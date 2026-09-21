# Device `*_overwrite_into` — evidence record (G2b-3, #1329)

Base: `origin/main` 4442fef8 (typed returning device transforms #1322, executor
caller alpha #1318). Branch `g2b3-device-overwrite-into`. No dependency change.

## What landed

`permute_overwrite_into`, `transpose_overwrite_into`,
`transpose_axes_overwrite_into` and `repartition_overwrite_into` on
`TensorMap<R, D, CudaStorage<D>>`, generic over `CudaPayload`, sharing one
private boundary `overwrite_tree_transform_cuda` — the device mirror of Host
`overwrite_tree_transform` (`tenet/src/typed.rs`).

## Host precondition -> device check

| # | Host step (`tenet/src/typed.rs::overwrite_tree_transform`) | Device | Error |
|---|---|---|---|
| 1 | `same_runtime` | same | `Error::RuntimeMismatch` |
| 2 | `typed_rule_identity` equality | same | `Error::RuleMismatch` |
| 3 | source owned **dense** (lazy adjoint rejected, not lowered) | same | `InvalidArgument("typed destination tree transform requires an ordinary dense CUDA source")` |
| 4 | destination owned dense | same | `InvalidArgument("destination must use ordinary dense CUDA storage")` |
| 5 | `Arc::ptr_eq(source.data, destination.data)` | same | `InvalidArgument("destination storage must not alias an input")` |
| 6 | `runtime.admitted_tree_pair_operation`, else build the operation | same | operation-build errors (expert layer) |
| 7 | if not admitted: `transformed_multiplicity_free` space equality | same | `InvalidArgument("destination fusion space or block layout does not match the operation result")` |
| 8 | `required_len` vs storage length | same (`TensorStorage::len`) | `InvalidArgument("destination storage length {a} does not match required length {r}")` |
| 9 | `Arc::strong_count(body) == 1 && strong_count(data) == 1` | same | `InvalidArgument("destination storage must be uniquely owned")` |
| — | (Host has none) | device-only placement check, first statement under the device lease | `Error::PlacementMismatch` |
| 10 | replay with the caller `alpha`, Overwrite | `CudaTreeTransformExecutor::replay(.., alpha, Overwrite)` | executor errors |
| 11 | if not admitted: `admit_exact_tree_pair_layout` | same, same Runtime store | — |

Steps 1-9 and the structure compile all run **before** `lease_cuda()`, as in
`tree_transform_cuda`; the Host context lease is dropped before the device
lease, so no lease nests.

Error equality evidence:

- `tenet/tests/typed_transform_host_side.rs::the_host_overwrite_into_preconditions_have_a_fixed_order_and_wording`
  pins the Host order and the exact Host strings, **ungated** (runs in ordinary
  CI, no device, no `cuda` feature);
- `tenet/tests/typed_cuda_transform_contracts.rs::device_overwrite_into_rejections_happen_before_any_device_work`
  runs each rejection on Host and on device and asserts
  `device == as_device_message(host)`, where `as_device_message` is the single
  documented substitution `"ordinary dense host {source,storage}"` ->
  `"ordinary dense CUDA {source,storage}"`.

## Rejection safety (no write before the last verdict)

The destination is the caller's device buffer, so the returning path's
"spend the upload first" argument (the Why-not comment at `tree_transform_cuda`)
does not transfer. Verified from source, no preflight was needed:

`tenet-operations/src/cuda_transform.rs::replay` decides, in order,
(a) the `Axpby(beta != 1)` capability boundary, (b) `validate_stage_a`
(structures, lengths, placement, context, capabilities),
(c) `prepare` -> `compile_device_plan` + `prepared_structure`, where every
region's expressibility and `validate_destination` verdict is taken, and
(d) `validate_stage_c` (regions, aliasing, workspace, coefficient readiness).
Only then does it `reserve_zero_template` and issue the first
`cuda_region_zero` over `dst`. So every rejection returns with the caller's
destination byte-identical. A rejection inside `prepare` also leaves no
executor state: `grow_workspace`, the coefficient upload and
`self.prepared.insert` all follow the failing steps. A `validate_stage_c`
rejection can leave a prepared entry behind — cache only, semantically neutral,
and unreachable from this boundary since steps 1-9 already pin the lengths,
placement and non-aliasing that Stage C re-checks.

Tests assert this operationally: every rejection case downloads the
destination before and after, and asserts `cuda_transfer_stats` delta ==
`CudaTransferStats::default()` and `cuda_tree_transform_stats()` unchanged.

## Short-circuit semantics

None, mirroring Host. Host `overwrite_tree_transform` has no identity branch,
so an identity axis list, `repartition` to the same split and a rank-0
`transpose` all still write `alpha * self` into the destination — unlike the
returning device `permute`/`repartition`/`transpose`, which clone. Gated by
`device_overwrite_into_has_no_identity_short_circuit` (device) and
`host_overwrite_into_clears_a_poisoned_destination_and_never_short_circuits`
(ungated).

## Caller alpha

`replay(.., alpha, Overwrite)` with the caller's value; the executor's #1318
rule applies it (descriptor alpha on Single moves and Multi scatters for
`alpha != 0`; the zero 1x1 operand with descriptor `1` for `alpha == 0`,
`-0.0` included by IEEE comparison), so `0 * src` is computed and a NaN or
infinite source propagates exactly as on Host. Disclosed rounding difference:
a Single block rounds as `alpha * (c * x)` where Host rounds `(alpha * c) * x`;
Multi blocks agree in order. Both are in the rustdoc.

## Layout admission

`admit_exact_tree_pair_layout` is the Runtime's own Host tree-pair store, keyed
on rule identity, operation and the two bound layouts — placement independent.
The device path feeds and reuses exactly that store, so a second call with the
same pair resolves the operation out of it (step 6 hits, step 7 is skipped);
`device_overwrite_into_admits_the_exact_layout_on_the_shared_runtime_store`
asserts entries unchanged, hits increased, equal results, and that a
destination that is *not* this operation's result still misses and is still
rejected.

## Cost contract

Warm: 0 H2D, 0 D2H, 0 device allocations — no #740 output initialisation,
because the destination is the caller's and Overwrite mode zeroes every
inactive layout itself. Cold: one coefficient upload per structure, at most one
workspace growth, one zero-template reservation. Asserted for all four methods
by `a_warm_device_overwrite_into_transfers_nothing_and_allocates_nothing`.

Measured disclosure from the A100 run: `alpha == 0` takes its 1x1 operand from
element 0 of the *context* zero template, so the first zero-scale replay on a
context whose template is still empty pays one `size_of::<D>()`-byte upload and
one device allocation. Once per context, not per call — the second zero-scale
call measures 0/0/0. This is in the rustdoc.

Kernel submission counters (`gemm_calls`, `copy_calls`) are non-zero on a warm
call by construction; the contract is about transfers and allocations, which is
why the tests compare those five fields rather than the whole
`CudaTransferStats`.

## Verification

Local (macOS, private target dir, removed afterwards):
`cargo fmt --all --check`; `cargo clippy --workspace --all-targets -D warnings`
for default and `--no-default-features --features cuda,cpu-faer`;
`cargo test -p tenet-rs -p tenet-network -p tenet-operations` (0 failures);
doctests for both feature sets; `RUSTDOCFLAGS=-D warnings cargo doc`.

Device (qg1, A100): see the run summaries recorded with this leaf's report.

## Residual

The storage-length mismatch (step 8) is mirrored and reachable in code, but is
not constructible through the public device API: every device tensor is built
with `required_len` storage and `validate_cuda_owned_metadata` rejects
anything else. It is covered by the Host twin's pinned message plus source
review, not by a device fixture.
