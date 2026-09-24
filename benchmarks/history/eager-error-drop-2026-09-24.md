# Eager success paths construct no TeNeT error values (#1360)

Date 2026-09-24. Host: Apple M4 Max, macOS 15.5, rustc 1.96.0. Base
`origin/main` `c34934e5`. Example `tenet/examples/eager_overhead_ledger.rs`
built as in `benchmarks/eager_overhead_ledger.sh` (`--release`, default
features, `CARGO_PROFILE_RELEASE_DEBUG=line-tables-only`), Cargo.lock sha256
`d7e6daaf…39ff` (unchanged). Binary sha256: base
`bf5a9a241420da1599c3bdd4ba3de0fc0a281e495e2957fbcec756547f08fffa`, change
`f6e9cd49f070dbb5b9355a7042b4a63d7c98d66f5a73a4f830dd93a3baec8f41`.

Method: the ledger's phase sampling, `LEDGER_SAMPLE=<file> <bin> <sym> f64 <case> <op>`
(`LEDGER_THREADS` default `one`, 4 s, `/usr/bin/sample` at 1 ms), for every
E1 row with a nonzero `errdrop` column: U(1)/fZ2×U(1)/SU(2) × {compose,
contract, permute, repartition, qr_compact, restrict_leg} × {r2_s8_d2,
r4_s3_d4}. A main-thread sample under `sample_calls` counts as `errdrop` when
any frame matches the ledger rule `drop_in_place<[^>]*Error>`
(`benchmarks/eager_overhead_phases.py`), counted over the whole stack.

## Result (main-thread samples in the timed loop; `errdrop` split by owning crate)

| sym | case | op | base samples | base TeNeT errdrop | base Tenferro errdrop | change samples | change TeNeT errdrop | change Tenferro errdrop |
|---|---|---|---:|---:|---:|---:|---:|---:|
| SU2 | r2_s8_d2 | compose | 3356 | 115 | 40 | 3356 | 0 | 37 |
| SU2 | r2_s8_d2 | contract | 3349 | 86 | 7 | 3360 | 0 | 21 |
| SU2 | r2_s8_d2 | permute | 3348 | 107 | 0 | 3376 | 0 | 0 |
| SU2 | r2_s8_d2 | qr_compact | 3360 | 16 | 51 | 3351 | 0 | 57 |
| SU2 | r2_s8_d2 | repartition | 3369 | 62 | 0 | 3361 | 0 | 0 |
| SU2 | r2_s8_d2 | restrict_leg | 3380 | 386 | 0 | 3375 | 0 | 0 |
| SU2 | r4_s3_d4 | compose | 3332 | 3 | 2 | 3345 | 0 | 2 |
| SU2 | r4_s3_d4 | contract | 3338 | 13 | 19 | 3367 | 0 | 11 |
| SU2 | r4_s3_d4 | permute | 3369 | 32 | 26 | 3370 | 0 | 18 |
| SU2 | r4_s3_d4 | qr_compact | 3361 | 2 | 10 | 3351 | 0 | 6 |
| SU2 | r4_s3_d4 | repartition | 3365 | 6 | 0 | 3293 | 0 | 0 |
| SU2 | r4_s3_d4 | restrict_leg | 3420 | 240 | 0 | 3396 | 0 | 0 |
| U1 | r2_s8_d2 | compose | 3338 | 93 | 33 | 3355 | 0 | 33 |
| U1 | r2_s8_d2 | contract | 3315 | 93 | 19 | 3351 | 0 | 26 |
| U1 | r2_s8_d2 | permute | 3306 | 92 | 0 | 3375 | 0 | 0 |
| U1 | r2_s8_d2 | qr_compact | 3360 | 13 | 53 | 3356 | 0 | 59 |
| U1 | r2_s8_d2 | repartition | 3373 | 79 | 0 | 3367 | 0 | 0 |
| U1 | r2_s8_d2 | restrict_leg | 3400 | 408 | 0 | 3389 | 0 | 0 |
| U1 | r4_s3_d4 | compose | 3357 | 11 | 4 | 3355 | 0 | 6 |
| U1 | r4_s3_d4 | contract | 3350 | 25 | 11 | 3358 | 0 | 5 |
| U1 | r4_s3_d4 | permute | 3387 | 15 | 0 | 3339 | 0 | 0 |
| U1 | r4_s3_d4 | qr_compact | 3288 | 12 | 4 | 3342 | 0 | 10 |
| U1 | r4_s3_d4 | repartition | 3365 | 17 | 0 | 3264 | 0 | 0 |
| U1 | r4_s3_d4 | restrict_leg | 3413 | 214 | 0 | 3410 | 0 | 0 |
| fZ2xU1 | r2_s8_d2 | compose | 3369 | 111 | 33 | 3386 | 1 | 31 |
| fZ2xU1 | r2_s8_d2 | contract | 3322 | 79 | 12 | 3367 | 0 | 18 |
| fZ2xU1 | r2_s8_d2 | permute | 3346 | 83 | 0 | 3392 | 0 | 0 |
| fZ2xU1 | r2_s8_d2 | qr_compact | 3377 | 11 | 44 | 3361 | 0 | 48 |
| fZ2xU1 | r2_s8_d2 | repartition | 3353 | 55 | 0 | 3369 | 0 | 0 |
| fZ2xU1 | r2_s8_d2 | restrict_leg | 3387 | 354 | 0 | 3403 | 0 | 0 |
| fZ2xU1 | r4_s3_d4 | compose | 3380 | 14 | 2 | 3306 | 0 | 7 |
| fZ2xU1 | r4_s3_d4 | contract | 3362 | 35 | 5 | 3355 | 0 | 8 |
| fZ2xU1 | r4_s3_d4 | permute | 3373 | 8 | 0 | 3326 | 0 | 0 |
| fZ2xU1 | r4_s3_d4 | qr_compact | 3321 | 15 | 17 | 3335 | 0 | 16 |
| fZ2xU1 | r4_s3_d4 | repartition | 3348 | 17 | 0 | 3199 | 0 | 0 |
| fZ2xU1 | r4_s3_d4 | restrict_leg | 3388 | 248 | 0 | 3396 | 0 | 0 |

Base: every sampled row has TeNeT-owned error drops (`OperationError`,
`CoreError`, `FusionAlgebraError`, `DenseError`), 0.2–12 % of the row, largest
in `restrict_leg` (6–12 %). Change: no `tenet_*` error drop remains except
one sample in `TensorContractFusionExecutionContext::tensorcompose_fusion*`
(1 of ~3 400, fZ2×U(1) compose). Every other remaining `errdrop` sample is
`drop_in_place<tenferro_tensor_core::error::ValidationError>` inside Tenferro
(`checked_logical_element_count`, `validate_reachable_bounds`,
`checked_product`, `validate_mutable_no_overlap`, `col_major_strides`), which
is a dependency and outside this change. Sample counts are observations;
wall-clock time is not a gate.

## Where the base drops came from

`sample` does not show inlined frames, so each caller was traced to the eager
`Option::ok_or(<Error>::…)` in it or in its inlined helpers:
`host_scalar_kernels::{checked_strided_extrema_from, …}` via
`validate_raw_strided_bounds`; `oriented_fusion_restrict_into`;
`BlockStructure::block`; `checked_{u1,z2}_irrep`/SU(2) `checked_irrep` via
`try_dual_sector` in `prepare_fusion_tree_layout_checked`;
`fusion_replay::{validate_storage_range, validate_direct_plan_layouts,
canonical_job_range}`; `kernel_adapter::normalize_fused_layout`;
`transform_replay::checked_fused_index_len`; `tenet_dense::layout`,
`executor`, `strided_batch_run_layout`; `strided::element_count`;
`lowering::lower_tensorcontract_adjoint_axes`;
`contract::{context, dynamic_space}`; the compact-factor plan and output
helpers in `factorize.rs`. The `storage_end_exclusive` per-block walk the
E1 ledger sampled is no longer on these paths at `c34934e5` (#1358); its
`ok_or` sites were converted anyway.

## Form of the change

`ok_or(E)` became `ok_or_else(|| E)` with the same variant, fields and check
order. Where `clippy::unnecessary_lazy_evaluations` rejects the closure
(it treats a variant with `Copy` fields as cheap, although the enum has drop
glue), the site became `let Some(x) = … else { return Err(E) }` or a
`match` returning the same `Err(E)`; no lint was allowed. `OffsetError`
(`host_scalar_kernels.rs`) has no drop glue and was left eager.

Checks: `cargo fmt --check`, `cargo clippy --workspace --all-targets -- -D
warnings`, `cargo test --workspace` (2709 passed, 0 failed).
