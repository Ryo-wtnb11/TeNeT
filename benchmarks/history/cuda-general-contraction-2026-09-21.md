# Device general-axes `contract` — evidence record (G2c-1a, #1345)

Base: `origin/main` 2f6f4a15 (device single-precision factorizations #1352),
rebased from 3b4c672c without conflict. Branch `g2c1a-device-general-contract`.
No dependency change. Design: `reviews/gpu-phase-20260920/g2c-design.md` §10
(authoritative) and its independent review.

## What landed

Device `contract` / `contract_ordered` on `TensorMap<R, D, CudaStorage<D>>`
(generic over `CudaPayload`: `f64`, `Complex64`, `f32`, `Complex32`) accept
arbitrary contracted and output axes, owned or lazy-adjoint.

| Seam | Where | Visibility |
|---|---|---|
| `try_compile_storage_contract_core_route` (free, lock-free) + `TensorContractFusionExecutionContext::compile_storage_contract_dynamic_tree`; `compile_storage_contract_resolution` = the two in order — the Host compile entry | `tenet-tensors/src/contract/context.rs` | `#[doc(hidden)] pub` |
| `StorageContractResolution` (opaque `Core` / `DynamicTree` route; `requires_core_right_twist`, `is_dynamic_tree`) | `tenet-tensors/src/contract/resolution.rs` | `#[doc(hidden)] pub` |
| `execute_storage_contract_resolution_on_cuda` + `CudaContractScratch` | `tenet-tensors/src/contract/dynamic/cuda.rs` (`cfg(feature = "cuda")`) | `#[doc(hidden)] pub` |
| `FusionBlockContractPlan::inactive_destination_regions` | `tenet-operations/src/fusion_replay.rs` | `pub` |
| `CudaDenseStorage::{capacity, set_active_len}` | `tenet-dense/src/cuda_adapter.rs` | `#[doc(hidden)] pub` |
| `Runtime::cuda_contract_scratch_bytes`, `CudaLease::split_contract` | `tenet/src/runtime.rs` | `pub` (cuda) / `pub(crate)` |
| `oriented_contract_destination` (split out of the Host lazy path) | `tenet/src/tensor_core.rs` | `pub(crate)` |

Deleted: the dead host-slice `tensorcontract_fusion_dynamic_plan_into_storage_context`
(`dynamic.rs`, `#[allow(dead_code)]`, which also ignored the RhsLhs twist
placement) with its three tests and the storage-origin scratch types only it
used. Reused rather than duplicated, as the design required.

## Reference record

- TensorKit `blas_contract!`, `src/tensors/tensoroperations.jl:383-455`
  @`cfaa073`: `tensoradd!` A and B into temporaries in matrix form (skipped
  when the permutation is shared), optional `twist!` (`:419`/`:429`), `mul!`,
  `tensoradd!` into C; orientation among four `contract_memcost` options
  (`contract!`, `:314-362`).
- QSpace `QSpace::contract`, `QSpace.cc:4141-4260` @`dd2cc7e`: `permute_to`
  both operands (a reference when the permutation is a no-op), grouped
  `contractDATA_group` GEMMs, `Permute` of the result; fermionic signs folded
  into the per-block GEMM scalar (`QSpace_aux.cc:105-115`).
- TeNeT Host `DynamicTree` (`execute_dynamic_tree_execution_artifact_impl`,
  `tenet-tensors/src/contract/dynamic.rs`) is exactly that decomposition, and
  the device path replays **the same artifact**: nothing categorical is
  re-derived, so the orientation / axis-order candidate, transform structures,
  borrow decisions, core plan and twist classification are the Host's.

| Host step | Device step |
|---|---|
| source transform into context scratch, `tree_transform_structure_overwrite_into_raw(.., 1)` | `CudaTreeTransformExecutor::replay(.., 1, Overwrite)` into the Runtime scratch; conjugation is the read flag |
| borrowed source (identity, unconjugated, same core layout) | the operand buffer itself |
| `execute_rhs_contract_twist` (in-place `scale_strided`) | none: a non-empty `core_right_twist` is rejected before the lease (G2c-2, #1347) |
| core `execute_raw` straight into `dst` (identity output) | `execute_direct_on_storage_prezeroed` into the fresh #740 output |
| `prepare_dst` (full `fill_zero`) + `execute_raw(β = 0)` into core scratch | `cuda_region_zero` of exactly `inactive_destination_regions()`, then the prezeroed core |
| output transform `tree_transform_structure_into_raw(.., 1, β = 0)` | `replay(.., 1, Overwrite)` |

Route choice. The `mul!` form keeps the existing storage-direct core (lazy
adjoints as GEMM operand flags; a uniform fermionic twist as per-job alpha —
unchanged), resolved by the free `try_compile_storage_contract_core_route`
with no Host context lock; everything else compiles the artifact with the
owned entry (two owned operands) or the prelowered entry (a lazy adjoint), the
same entries Host uses. The device route equals the Host's except in three
classes, each with equal results to dtype tolerance, never bitwise:

- Host `Structure` (a conjugated operand whose sectors are all self-dual, in
  core-form source order, with a non-core output): the device runs the
  prelowered `DynamicTree` artifact, a path the Host never takes for that
  class. Gated on SU(2), multi-block, degeneracy > 1, lazy lhs and lazy rhs,
  all four dtypes (`where_the_host_takes_its_structure_route_…`), with the
  Host resolution pinned as `Structure` in `storage_contract_tests.rs`;
- canonical owned operands with a uniform fermionic twist: Host takes
  `DynamicTree` with an in-place twist, the device the pre-existing scaled
  storage core;
- owned core geometry over a non-canonical storage layout (expert layouts
  only): Host packs/scatters in its core, the device takes `DynamicTree`.

Diagonal operands stay `UnsupportedOnDevice`. Behaviour change: an anyonic
device `contract` now errors even in canonical form, as on Host.

The GEMM seam now bounds every matrix view by the active length
(`check_matrix_bound`), not only by the allocation, so a narrowed buffer is
safe against any future caller, not only the ones that validate first.

Rust deviations: the scratch is narrowed to each replay's exact length
(`set_active_len`) instead of being a slice view, because a `CudaStorage` is a
whole allocation and the replay admits exact lengths only; the inactive-block
regions are converted to device regions per call (one small `Vec` per inactive
block).

## Scratch ownership, zeroing and budget

- Owner: `CudaDeviceState` (beside the tree-transform executor, same device
  mutex). Per (payload dtype, context): lhs, rhs, core-destination buffers.
  Grown only at a new high-water mark (one zero H2D per grown buffer, #740),
  never shrunk, released by `Runtime::clear_tree_transform_cache` under the
  same maintenance lease as the executor.
- Zeroing: source scratch never (Overwrite assigns every active block and
  zeroes every inactive layout); core-destination scratch exactly the plan's
  inactive destination blocks, skipped when there are none; the returned
  output nothing beyond its #740 initialisation.
- `contracted == 0` jobs cannot occur: a zero degeneracy is dropped from the
  leg, so no admitted operand has an empty contracted extent; a coupled sector
  with nothing to contract has no job and its block is an inactive block
  (pinned ungated, `a_zero_degeneracy_sector_never_yields_a_zero_extent_core_job`).
- Budget: bounded (high-water of the contractions run) and observable
  (`cuda_contract_scratch_bytes`, reported apart from
  `cuda_tree_transform_stats` so no byte is counted twice); not charged to
  `PlanCacheConfig::workspace_budget_bytes` (g2b-design §10 P1-2 precedent).

## Lock / lease order

All admission (runtime, anyonic, operand storage, destination derivation and
the Host's axis / leg validation) → the lock-free canonical core route; only on
a miss the Host context lease → compile the artifact → drop it → twist check → lock-free placement check → **one**
device lease (`split_contract`) → output upload → replay → drop. Nothing
under the device lease leases again.

## Contracts measured (A100, f64, `u1_rank_five`: rank 5 × rank 4, 561-element output)

```
cold  CudaTransferStats { h2d_calls: 7, h2d_bytes: 15592, d2h_calls: 0, d2h_bytes: 0, device_allocs: 7, gemm_calls: 126, .. }
warm  CudaTransferStats { h2d_calls: 1, h2d_bytes: 4488,  d2h_calls: 0, d2h_bytes: 0, device_allocs: 1, gemm_calls: 126, .. }
scratch 10136 B; CudaTreeTransformStats { prepared_structures: 3, executor_bytes: 968, workspace_bytes: 0, required_plan_entries: 50 }
```

Cold = the 4488-byte output + three scratch high-water uploads (10136 B) +
three coefficient payloads (968 B). Warm = the #740 output initialisation
only; scratch and transform state unchanged. A smaller contraction after a
larger one allocates nothing but its output (scratch narrowed, not
reallocated). A `-1` descriptor alpha adds no cuTENSOR plan signature
(`alphas()` of `tenet-operations/tests/cuda_tree_transform.rs` now includes
`-1`; `warm_scale_sweep` asserts plan misses unchanged). `gemm_calls` counts
every dot-general submission (region moves, zero fills and GEMMs), equal cold
and warm.

## Correctness evidence

- Forced orientations (`tenet-tensors/src/contract/storage_contract_tests.rs`,
  device): both orientations × every axis-order candidate, U(1) rank 5 and
  SU(2) recoupling, `f64` and `Complex64`, device replay == Host replay of the
  same artifact; identity-transform borrow of either operand (asserted to
  really borrow); NaN-poisoned core-destination scratch before a contraction
  with an inactive core block comes back finite and equal to Host.
- Typed lowering (`tenet/tests/typed_cuda_contract.rs`, device): device ==
  Host and device == TensorKit `blas_contract!` sequence on Host typed ops for
  U(1) rank 5, U(1) and SU(2) reordered whole-side, SU(2) codomain-against-
  domain, U(1) × SU(2), and both identity-transform fixtures, at all four
  dtypes; lazy-adjoint lhs and rhs at all four dtypes; device == physical-basis
  contraction for U(1)/SU(2), `f64`/`Complex64`; degeneracy > 1 and multiple
  blocks throughout.
- Oracle independence (`tenet/tests/typed_contract_host_oracle.rs`, ungated):
  Host == `blas_contract!` sequence (all fixtures, lazy cases, four dtypes) and
  Host == physical-basis contraction.
- Boundaries: fermionic dual contracted leg on a transformed core-right operand
  → `UnsupportedOnDevice`, transfer counters, transform stats and scratch
  unchanged; malformed axes / output order / non-dual pairing / runtime
  mismatch → the Host's error text, nothing submitted. Former rejection
  assertions for non-canonical axes in `typed_cuda_transfer.rs` inverted.
- Never bitwise across Host/device: GEMM summation order is cuTENSOR's; Host
  `-0.0` vs device `+0.0` on zero blocks.

## Device run (A100, `cuda,cpu-faer`, `--test-threads=1`)

qg1, CUDA 12.6, GPU 0 with no process of any user, private target
`/data2/ryo-w/gpu-phase/g2c1a-target` (removed afterwards). Verbatim:

```
tenet-tensors --lib storage_contract -- --ignored        (forced orientations, borrow, NaN scratch)
test result: ok. 3 passed; 0 failed; 0 ignored; 0 measured; 539 filtered out; finished in 4.51s
tenet-tensors --lib storage_contract (non-ignored, cuda build)
test result: ok. 6 passed; 0 failed; 3 ignored; 0 measured; 533 filtered out; finished in 0.55s
tests/typed_cuda_contract.rs -- --ignored
test result: ok. 7 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 5.42s
tests/typed_cuda_contract.rs (non-ignored)
test result: ok. 0 passed; 0 failed; 7 ignored; 0 measured; 0 filtered out; finished in 0.00s
tests/typed_contract_host_oracle.rs (ungated)
test result: ok. 2 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 1.29s
tenet-operations tests/cuda_tree_transform.rs -- --ignored   (alpha sweep incl. -1)
test result: ok. 25 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 7.81s
tests/typed_cuda_single_precision.rs -- --ignored
test result: ok. 11 passed; 0 failed; 0 ignored; 0 measured; 3 filtered out; finished in 4.10s
cargo test -p tenet-rs --doc --no-default-features --features cuda,cpu-faer
test result: ok. 92 passed; 0 failed; 1 ignored; 0 measured; 0 filtered out; finished in 8.87s
```

Full device suite (`cargo test --workspace --lib --tests --no-default-features
--features cuda,cpu-faer --no-fail-fast -- --ignored --skip
measure_checked_generic_transform_phases --skip axioms_ --skip itebd_ --skip
cross_library --test-threads=1`), 121 test binaries: **203 passed, 0 failed**.
(The first full run found one stale rejection assertion for non-canonical axes
in `typed_cuda_transfer.rs::typed_cuda_c64_contract_and_compose_match_host`;
it was inverted to device == Host and the rerun above is on the final tree.)

Local gates (macOS, private target, removed afterwards): `cargo fmt --all
--check`; `cargo clippy --workspace --all-targets -- -D warnings` and with
`--no-default-features --features cuda,cpu-faer`; `cargo test -p tenet-rs -p
tenet-tensors -p tenet-operations -p tenet-network` (1637 passed, 0 failed);
doctests 68 (default) and 92 (cuda) passed; `RUSTDOCFLAGS="-D warnings" cargo
doc --workspace --no-deps` and the cuda doc build of tenet-dense,
tenet-operations, tenet-tensors, tenet-rs.

## Device rerun after the independent source review

The review's P1 added the Host-`Structure` class fixture
(`su2_structure_cases`, pinned Host-side as `Structure` by
`a_self_dual_core_form_lazy_contraction_is_host_structure_and_device_dynamic_tree`)
and its P2s: the lock-free canonical core route, the active-length GEMM view
bound, tolerance-based compares with length asserts, the route-difference
wording, the anyonic disclosure and the checked-Generic gap. Same qg1 rules,
GPU 0, fresh private target (removed afterwards). Verbatim:

```
tenet-tensors storage_contract -- --ignored
test result: ok. 3 passed; 0 failed; 0 ignored; 0 measured; 540 filtered out; finished in 4.54s
tenet-tensors storage_contract (non-ignored, cuda build)
test result: ok. 7 passed; 0 failed; 3 ignored; 0 measured; 533 filtered out; finished in 0.76s
tests/typed_cuda_contract.rs -- --ignored
test result: ok. 8 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 5.92s
tests/typed_cuda_contract.rs (non-ignored)
test result: ok. 0 passed; 0 failed; 8 ignored; 0 measured; 0 filtered out; finished in 0.00s
tests/typed_contract_host_oracle.rs
test result: ok. 2 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 1.40s
cargo test -p tenet-rs --doc --no-default-features --features cuda,cpu-faer
test result: ok. 92 passed; 0 failed; 1 ignored; 0 measured; 0 filtered out; finished in 9.87s
```

Full device suite (same command as above), 121 binaries: **204 passed, 0
failed**.

## Residuals

- G2c-1b (#1346): device `contract_overwrite_into` stays canonical-only; the
  zeroing rule for an identity-output caller destination lands there.
- G2c-2 (#1347): the non-empty core-right twist (destination scales), and
  canonical axes with a non-uniform θ, which still return the storage-direct
  "fermionic twist is nonuniform" error.
- G2c-3 (#1348): `tensor!` device networks keep their canonical schedule
  predicate; preflight must classify the twist statically (design P1-4).
- Disclosed constants: the #740 output upload; one zero H2D per scratch buffer
  at each high-water mark; the Overwrite zero fills of inactive layouts over a
  fresh output; one host `Vec` per inactive core block per call.
