# Device general `contract` — fermionic core-right twist via destination scales (G2c-2, #1347)

Base: `origin/main` 14b2ffbf (device DynamicTree executor, #1353). Branch
`g2c2-device-contract-twist`. No dependency change. Design:
`reviews/gpu-phase-20260920/g2c-design.md` §10 P1-1/P1-2/P1-3 (authoritative)
and `g2c-independent-design-review.md`.

## What landed

A device `contract` whose Host `DynamicTree` artifact carries a fermionic
core-right twist now runs on device instead of returning
`UnsupportedOnDevice`, and the canonical form whose twist varies within one
coupled sector leaves the storage-direct core route for that artifact.

| Seam | Where | Visibility |
|---|---|---|
| `DynamicTreeExecutionArtifact::core_right_destination_scales`: sorted `(core-right destination block offset, θ_b ≠ 1)`, zero-element blocks left out (they can share an offset with the next block); built by `compile_rhs_contract_twist` in the loop that builds `core_right_twist` | `tenet-tensors/src/contract/dynamic.rs` | `pub(crate)` |
| `validate_uniform_multi_scales`: every live destination layout of one Multi block has the same θ, else `InvalidArgument` (internal, never a panic); run in `finish_dynamic_tree_execution_artifact` when the list is non-empty | same | `pub(super)` |
| `CudaTreeTransformExecutor::replay_with_destination_scales(ctx, structure, dst_structure, src_structure, dst, src, alpha, mode, destination_scales: &[(usize, C)])`; `replay` is it with `&[]` | `tenet-operations/src/cuda_transform.rs` | `#[doc(hidden)] pub` |
| `NonuniformTwist { Reject, Decline }` on `try_compile_oriented_storage_contract_plan`: the device core route declines, the Host storage-direct entries keep rejecting | `tenet-tensors/src/contract/resolution.rs`, `context.rs` | `pub(crate)` |
| `validate_storage_contract_request` (provider against the spaces, operand flags against the request), now also run by `compile_storage_contract_dynamic_tree`, so the `#[doc(hidden)] pub` second half is sound on its own (re-check P2) | `tenet-tensors/src/contract/context.rs` | private |

The typed device `contract` no longer rejects a non-empty core-right twist;
`execute_dynamic_tree_on_cuda` passes the list to the core-right source
transform (the physical rhs under `LhsRhs`, the physical lhs under `RhsLhs`)
and `&[]` to the other.

## Reference record

- TensorKit `blas_contract!`, `src/tensors/tensoroperations.jl:383-455`
  @`cfaa073`: when any `pB[1]` leg is dual, twist B after its `tensoradd!`
  over its dual codomain legs (`:429`), or A over its non-dual domain legs
  (`:419`) when A is the operand already copied (`:398-409`); `twist!` is a
  separate in-place pass.
- QSpace `contract_matchAB_groupC` → `getFermSigns` (`QSpace.cc:4399-4400`)
  and `contractDATA_group` folding the per-block sign into the GEMM scalar
  `afac = fA[ia]*fB[ib]` (`QSpace_aux.cc:105-115`) @`dd2cc7e`: the sign rides
  a scalar of an existing pass, as here.
- TeNeT Host: `execute_dynamic_tree_execution_artifact_impl` runs the
  core-right source transform with α = 1, then `execute_rhs_contract_twist`
  (`scale_strided` per twisted block, `dynamic.rs`). A twisted core-right
  operand is never borrowed (`rhs_source_is_borrowable`), so the transform
  always runs; the device re-checks this and returns an internal error
  otherwise.

| Host step | Device step |
|---|---|
| core-right transform, α = 1 | `replay_with_destination_scales(.., α = 1, Overwrite, θ list)`: the Single move / Multi scatter writing block b gets descriptor α·θ_b |
| `execute_rhs_contract_twist`: in-place `scale_strided` of every twisted block | none — folded into the write above (one pass fewer than Host and TensorKit) |
| packs, recoupling GEMM, inactive-layout zero fills (α-free) | unchanged: unscaled |

Why the fold is exact: the Host computes θ_b·(α·T(src))_b per block; a Single
block is one move, and a Multi block's GEMM output column for destination b is
scattered by one move, so scaling that move's descriptor by θ_b is the same
product. The compiler additionally asserts θ uniform per Multi block (θ
depends only on the destination codomain's uncoupled sectors at the
contracted positions, which one Multi block shares), so the fold is also "the
twist after the transform" at recoupling granularity. θ = ±1, so α·θ_b is
exact and non-zero whenever α ≠ 0.

α == 0 composition: a zero caller scale keeps `replay`'s zero-operand route
(descriptor 1, operand 0) and ignores θ: the written values are 0·x, NaN for
a NaN source, exactly the Host's θ·(0·x) up to the sign of an exact zero. The
contraction itself always replays sources at α = 1; the α == 0 path is
reachable only by a direct executor caller and is gated there.

Host behaviour for canonical non-uniform θ (unchanged): the Host eager
`contract` routes a twisted canonical contraction to `DynamicTree` with the
in-place twist; only the Host storage-direct entries
(`tensorcontract_fusion_dyn_direct_on_storage_raw`,
`tensorcontract_fusion_dyn_prelowered_direct_on_storage`, used by the device
canonical `contract_overwrite_into`) return "fermionic twist is nonuniform
within one RHS coupled-sector matrix", and still do. The device returning
`contract` now declines the scaled core there and runs the artifact.

Rust deviations: θ is kept as the provider coefficient type `C` and converted
per submission with `scale_by_coefficient`, so no per-dtype θ vector is
stored or uploaded; the lookup is a binary search of the sorted list by the
prepared move's destination offset (host-side, O(log T) per written block, and
skipped when the list is empty).

## Cost and contracts

- Device: no extra pass, no extra upload, no key change. The prepared
  structure, its uploaded coefficient vector, its `StructureKey` and its
  cuTENSOR plan signatures are those of the untwisted replay (a ±1 descriptor
  α adds no signature; `alphas()` in `cuda_tree_transform.rs` already covers
  −1). Gated: a scaled replay after an unscaled one of the same structure
  transfers nothing, allocates nothing, prepares no structure and misses no
  plan; a warm twisted typed `contract` uploads only its #740 output.
- Host: per compile of a twisted artifact, the scale list costs one boxed
  slice (16 B per twisted block; at most one reallocation when it is boxed)
  and the uniformity scan walks the core-right transform's Multi destination
  layouts once with an O(log T) lookup each. Untwisted and bosonic artifacts
  pay nothing: an empty `Vec` boxes without allocating and the scan is
  skipped for an empty list.
- Never bitwise across Host/device: the Host scales a zero block to `-0.0`
  where the device's unscaled Overwrite zero fill leaves `+0.0`.

## Correctness evidence

- Ungated, artifact level (`tenet-tensors/src/contract/storage_contract_tests.rs`):
  for every forced axis-order candidate and **both** orientations of fZ2 x
  U(1) and fZ2 (x) SU(2) fixtures (twist on one or both contracted legs,
  canonical non-uniform), the scale list equals the Host twist compiler's
  per-block twist of the *physical* core-right operand's transformed source
  (the physical rhs under LhsRhs, the physical lhs under RhsLhs, recomputed
  from that space alone) and the artifact's own in-place actions, is strictly
  sorted and ±1-valued; every fixture twists under LhsRhs and, per provider,
  the mixed-θ and canonical fixtures also under RhsLhs; each forced
  orientation's Host replay equals the eager Host contraction (review P2s); some artifact twists per fixture
  and some twisted core-right transform recoupled (the uniform assertion is
  exercised). A list flipped on one destination of a Multi block is an
  `InvalidArgument`. The canonical non-uniform case: the device core route
  returns `None`, the storage resolution is a twisted `DynamicTree`, and the
  Host storage-direct entry still rejects it before any GEMM. The
  Host-`Structure` pin now uses a positive `last_resolution_is_structure`
  flag (re-check P2).
- Ungated, oracle (`tenet/tests/typed_contract_host_oracle.rs`): Host
  `contract` == TensorKit's `blas_contract!` sequence with the twist on the B
  role == the same with the twist on the A role, for fZ2 x U(1) and
  fZ2 (x) SU(2) rank-4 fixtures (degeneracy > 1; twist on B's codomain side
  and from its domain side; mixed θ within one coupled sector; the canonical
  non-uniform form; lazy adjoint on either side), at f64/c64/f32/c32; each is
  non-vacuous (differs from the twist-free sequence). The FZ2 closed loops of
  `tk_fermionic_correspondence` as explicit `contract` calls equal
  TensorKit's −4, −14, −6, −6.
- Device, artifact level: same forced fermionic artifacts, device replay ==
  Host replay (transform + in-place twist), f64 and c64.
- Device, executor level (`tenet-operations/tests/cuda_tree_transform.rs`):
  `replay_with_destination_scales` == Host replay followed by an index-walk
  block scale, α ∈ {1, 0, −2.5} × alternating θ, four dtypes, Single,
  conjugated and recoupling fixtures, with a negative control; α = ±0 over a
  NaN source keeps the Host's NaN pattern; unsorted offsets rejected before
  any device work.
- Device, typed (`tenet/tests/typed_cuda_contract.rs`): device == Host ==
  both TensorKit twist roles for every fermionic fixture at all four dtypes;
  the FZ2 loops as explicit device `contract` calls equal TensorKit; warm
  twisted contraction = the #740 output upload only.

## Device run (A100, `cuda,cpu-faer`, `--test-threads=1`)

qg1, CUDA 12.6, GPU 0 with no process of any user, private target
`/data2/ryo-w/gpu-phase/g2c2-target` (removed afterwards). Verbatim, on the
final tree:

```
tenet-tensors --lib storage_contract -- --ignored
test contract::storage_contract_tests::device::a_nan_poisoned_core_destination_scratch_is_zeroed_on_its_inactive_blocks ... ok
test contract::storage_contract_tests::device::a_twisted_artifact_replays_with_destination_scales_as_the_host_scales_in_place ... ok
test contract::storage_contract_tests::device::an_identity_source_transform_borrows_either_operand_on_device ... ok
test contract::storage_contract_tests::device::every_forced_candidate_and_orientation_matches_the_host_replay ... ok
test result: ok. 4 passed; 0 failed; 0 ignored; 0 measured; 543 filtered out; finished in 9.92s
tenet-tensors --lib storage_contract (non-ignored, cuda build)
test result: ok. 10 passed; 0 failed; 4 ignored; 0 measured; 533 filtered out; finished in 2.17s
tests/typed_cuda_contract.rs -- --ignored
test result: ok. 10 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 8.69s
tests/typed_cuda_contract.rs (non-ignored)
test result: ok. 0 passed; 0 failed; 10 ignored; 0 measured; 0 filtered out; finished in 0.00s
tests/typed_contract_host_oracle.rs (ungated)
test result: ok. 4 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 2.32s
tenet-operations tests/cuda_tree_transform.rs -- --ignored
test result: ok. 29 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 8.58s
tenet-operations tests/cuda_tree_transform.rs (non-ignored)
test result: ok. 0 passed; 0 failed; 29 ignored; 0 measured; 0 filtered out; finished in 0.00s
cargo test -p tenet-rs --doc --no-default-features --features cuda,cpu-faer
test result: ok. 92 passed; 0 failed; 1 ignored; 0 measured; 0 filtered out; finished in 9.11s
```

Contracts measured (A100, f64, `fermionic_general(fZ2 (x) SU(2), mixed θ)`,
2584-byte output), warm:

```
fZ2xSU2 mixed θ f64: output 2584 B; warm CudaTransferStats { h2d_calls: 1, h2d_bytes: 2584, d2h_calls: 0, d2h_bytes: 0, device_allocs: 1, gemm_calls: 218, solver_calls: 0, copy_calls: 0 }; CudaTreeTransformStats { prepared_structures: 3, executor_bytes: 2912, workspace_bytes: 1328, context_scalar_operand_bytes: 8, required_plan_entries: 105 }
```

The untwisted `u1_rank_five` warm line is unchanged from G2c-1a (h2d 1 /
4488 B, allocs 1, gemm 126, `required_plan_entries` 50).

Full device suite (`cargo test --workspace --lib --tests --no-default-features
--features cuda,cpu-faer --no-fail-fast -- --ignored --skip
measure_checked_generic_transform_phases --skip axioms_ --skip itebd_ --skip
cross_library --test-threads=1`), 121 test binaries: **211 passed, 0 failed**
(G2c-1a: 204; +1 artifact-level, +2 net typed — the twist-rejection test is
replaced — and +4 executor-level). A first targeted run failed one new test
of its own making (a one-block fixture got an empty alternating θ list; the
list now starts at the first block) before the rerun above.

Local gates (macOS, private target, removed afterwards): `cargo fmt --all
--check`; `cargo clippy --workspace --all-targets -- -D warnings` and with
`--no-default-features --features cuda,cpu-faer`; `cargo test -p tenet-rs -p
tenet-tensors -p tenet-operations -p tenet-network` (1643 passed, 0 failed);
doctests (default and cuda); `RUSTDOCFLAGS="-D warnings" cargo doc
--workspace --no-deps` and the cuda doc build of tenet-rs, tenet-tensors,
tenet-operations.

## Residuals

- G2c-1b (#1346): device `contract_overwrite_into` stays canonical-only and
  therefore keeps the Host storage-direct rejection of a non-uniform twist;
  the destination-scale mechanism is what it will reuse for general axes.
- G2c-3 (#1348): network preflight must now admit twisted steps (P1-4's
  static fermionic rejection is no longer needed for correctness; the
  schedule-level decision remains that leaf's).
- `tk_fermionic_correspondence` itself (tenet-network, `tensor!`) is
  unchanged; its scalars are re-expressed as explicit contracts in
  `contract_cases::fz2_tensorkit_loops`, run on Host (ungated) and device.
