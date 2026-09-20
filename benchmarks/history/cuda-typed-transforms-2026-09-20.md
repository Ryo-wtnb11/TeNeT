# Typed structural transforms on device tensors (G2b-2, issue #1322)

Base: origin/main `f7d324fe` (executor with caller alpha, #1319). Branch
`g2b2-typed-device-transforms`. No dependency change.

## What landed

`permute`, `braid`, `transpose`, `transpose_axes` and `repartition` on
`TensorMap<R, D, CudaStorage<D>>`, generic over `CudaPayload` (`f64`,
`Complex64`), plus the Runtime ownership of the device tree-transform executor.

Semantics are the Host's by construction: the typed device method builds the
same `TreeTransformOperation`, runs the same rule validation, and compiles or
looks up the same `TreeTransformStructure` through the same
`RuleIdentity`-keyed store. Only the executor below that structure differs.

## Ownership and lock order

`RuntimeInner.cuda` is now `Option<Mutex<CudaDeviceState>>` with
`CudaDeviceState { dense: CudaDenseContext, tree_transform:
CudaTreeTransformExecutor }`. `CudaLease` is a newtype over the guard with
`Deref/DerefMut<Target = CudaDenseContext>`, so the ~20 existing `&mut *lease`
sites are untouched, plus `split() -> (&mut CudaDenseContext, &mut
CudaTreeTransformExecutor)` for a replay, which needs both at once.

Per returning transform, in order:

1. lazy-adjoint lowering (`lower_adjoint_tree_transform_operation`) and
   recursion on the owned parent — lease-free;
2. destination space derivation (`transformed_multiplicity_free`);
3. `lease_context()` → `compile_tree_pair_structure` (the Host store's own
   mutex is taken and released inside) → **the context lease is dropped**;
4. `lease_cuda()` → placement check → output upload → `replay(alpha = 1,
   Overwrite)` → drop.

At most one mutex is held at a time, and nothing reached under the device
lease leases again (`std::sync::Mutex` is not re-entrant). A compile-level
gate in `tenet/src/runtime.rs` pins both the `DerefMut` and the `split()`
shape.

Executor bounds come from the Runtime's own configuration at build:
`tree_transform_cache_info().entry_capacity()` prepared entries (so a
structure warm on the Host stays warm on device),
`DEFAULT_COEFFICIENT_BUDGET_BYTES` and `DEFAULT_PLAN_CACHE_BUDGET_BYTES`.
Nothing is charged to `PlanCacheConfig::workspace_budget_bytes`.

`Runtime::cuda_tree_transform_stats()` reports, read-only,
`prepared_structures`, `executor_bytes` (coefficient payloads + workspaces),
`workspace_bytes`, `context_scalar_operand_bytes` (owned by the context, not
the executor — counted once, separately) and `required_plan_entries`.
`clear_tree_transform_cache()` clears the Host store first, then takes the
device lease and clears the executor; never nested.

## Measured transfer contract (A100, CUDA 12.6, rustc 1.96.0)

Operation-matrix rows, `many-small` (64 blocks, degeneracy 4), rank-2 U(1)
endomorphism, columns `h2d_calls, h2d_bytes, d2h_calls, d2h_bytes,
device_allocs, gemm_calls, solver_calls, copy_calls`:

| operation | phase | h2d | h2d bytes | d2h | allocs | gemm | copy |
|---|---|---|---|---|---|---|---|
| permute (f64) | cold | 2 | 8704 | 0 | 2 | 64 | 0 |
| permute (f64) | warm | 1 | 8192 | 0 | 1 | 64 | 0 |
| braid (f64) | warm | 1 | 8192 | 0 | 1 | 64 | 0 |
| transpose (f64) | warm | 1 | 8192 | 0 | 1 | 64 | 0 |
| transpose_axes (f64) | warm | 1 | 8192 | 0 | 1 | 64 | 0 |
| repartition (f64) | warm | 1 | 8192 | 0 | 1 | 64 | 0 |
| permute (c64) | warm | 1 | 16384 | 0 | 1 | 64 | 0 |

8192 bytes is exactly the output (`1024 · size_of::<f64>()`). The cold row's
extra 512 bytes are the structure's 64 coefficients, uploaded once. Every row
reports `ok:host_value_equality` against the Host answer.

This is the #1304/#740 constant and nothing else: a warm returning call
uploads no structure data, downloads nothing, and allocates only its output.
`*_overwrite_into` (transfer-free when warm) is G2b-3.

The warm contract holds only while the Host store admits the structure: a
tree-transform cache byte budget of zero, or an entry over the store's
per-entry limit, hands back a fresh allocation per call and so re-uploads the
coefficient payload per call. Stated in each method's rustdoc.

## #1320 (misaligned whole-factor copy) exposure: none

The typed transform path issues no `copy_read_into` at all. The output is a
whole-buffer `CudaStorage::upload_owned`; the executor moves blocks with
`cuda_region_axpby` and zeroes inactive layouts with `cuda_region_zero`, both
of which submit `dot_general`. `cuda_zero_prefix` and `cuda_copy_region_into`
— the two `copy_read_into` wrappers #1320 is about — are not reached. Measured
confirmation: `copy_calls = 0` in every operation-matrix transform row, cold
and warm. No fixture was excluded for #1320.

## Device suite

`cargo test --workspace --lib --tests --no-default-features --features
cuda,cpu-faer --no-fail-fast -- --ignored --skip
measure_checked_generic_transform_phases --skip axioms_ --skip itebd_ --skip
cross_library --test-threads=1` on A100 (GPU 0, no other process):
**142 passed, 0 failed**, including the two new binaries (9 and 5 tests).

## Scope of this evidence

Timings in the operation matrix are one process's per-iteration medians of a
debug build and are a validation fixture only: no value here feeds a dispatch
decision, and no baseline CSV was regenerated. The device numbers that carry
weight are the transfer, allocation and submission counters, which are exact.
