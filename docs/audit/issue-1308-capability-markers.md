# Payload-dtype capability markers — public entry-point classification

Authority: TeNeT `98ebb52904633ac9346dfe96515060d9dad03dfa` (`origin/main` at the
time of the split), issue
[#1308](https://github.com/Ryo-wtnb11/TeNeT/issues/1308), plan
[#1065](https://github.com/Ryo-wtnb11/TeNeT/issues/1065), survey
`reviews/gpu-phase-20260920/single-precision-survey.md`.

This artifact is the classification the leaf was reviewed against. It is
revision-pinned evidence, not current capability authority: `tenet/src/typed.rs`
and `tenet/tests/capability_markers.rs` are.

## The three markers

| Marker | Definition (`tenet/src/typed.rs`) | Sealing |
| --- | --- | --- |
| `TensorScalar` | `pub trait TensorScalar: ScalarOps {}` | private supertrait `ScalarOps` (`pub(crate)`) — unchanged |
| `FactorizationScalar` | `pub trait FactorizationScalar: TensorScalar {}` | inherited from `TensorScalar`, the pattern `CudaPayload` already uses |
| `AdvancedLinalgScalar` | `pub trait AdvancedLinalgScalar: FactorizationScalar {}` | inherited from `TensorScalar` |

All three are implemented for `f64` and `num_complex::Complex64` only. No new
dtype, no tolerance change, no renamed public item; `TensorScalar` keeps both
its name and its base family, so every f64/Complex64 caller that compiles today
still compiles.

`WireScalar` (persistence, `tenet/src/typed/serialization.rs`) and `CudaPayload`
(device payload) are orthogonal capability traits and are unchanged.

## Public entry points by marker

### `TensorScalar` — base

Nothing below was re-bounded; this column records that the classification was
made, not skipped.

| Family | Entry points |
| --- | --- |
| Construction | `zeros`, `from_block_fn`, `rand`, `rand_with_seed`, `id`, `isomorphism`, `unitary`, `isometry`, `zeros_like`, `scalar`, `insert_left_unit`, `insert_right_unit`, `remove_unit` |
| Inspection | `provider`, `runtime`, `rank`, `numout`, `numin`, `numind`, `codomain*`, `domain*`, `block`, `block_count`, `blocks`, `block_fusion_trees`, `data`, `placement`, `leg_dim(s)`, `is_diagonal`, `diagonal_spectrum`, `detach_runtime`/`attach_runtime` |
| Arithmetic | `add`, `add_assign`, `scale`, `scale_assign`, `normalize`, `adjoint` |
| Reductions | `norm`, `norm_inf`, `norm_p`, `inner`, `dot`, `tr` |
| Predicates | `is_hermitian`, `is_antihermitian`, `is_isometric`, `is_unitary`, `project_hermitian`, `project_antihermitian` |
| Products | `contract`, `contract_ordered`, `compose`, `otimes`, `deligne_product`, `catdomain`, `catcodomain`, the `*_overwrite_into` expert forms |
| Structure | `permute`, `braid`, `transpose`, `transpose_axes`, `repartition`, `twist`, `twist_inverse`, `flip`, `flip_inverse` |
| Legs / diagonal | `restrict_leg`, `embed_leg`, `restrict_diagonal`, `diagonal`, `diagview`, `absorb` |
| Trace | `trace_pairs` |
| Physical dense | `to_physical_dense`, `project_physical_dense`, and the SU(2) pair |
| Spaces | all of `GradedSpace`, `LegSelection`, `TruncatedSelection`, including `find_truncated` (no payload) |
| Conversions | `to_c64`, `re`, `im` |
| Device (base half) | `to_cuda`, `to_host`, and the `CudaStorage` block at `impl<R, D: CudaPayload> TensorMap<R, D, CudaStorage<D>>` covering `adjoint`, `scale`, `add`, `zeros_like`, `normalize`, `norm`, `inner`, `dot`, `contract*`, `compose` |
| Network | every `D: TensorScalar` bound in `tenet-network` (`network.rs`, `plancache.rs`) and `tensor!` execution — unchanged, per the #1308 table |

### `FactorizationScalar`

| Entry point | Site |
| --- | --- |
| `qr_compact`, `left_orth` | `TypedTensorQrDispatch` + its public impl |
| `qr_full` | `TypedTensorFullQrDispatch` |
| `lq_compact`, `right_orth` | `TypedTensorLqDispatch` |
| `lq_full` | `TypedTensorFullLqDispatch` |
| `svd_compact`, `svd_full` | `TypedTensorSvdDispatch` |
| `svd_vals` | `TypedTensorSvdValsDispatch` |
| `svd_trunc` | `TypedTensorSvdTruncDispatch` |
| `eigh_full` | `TypedTensorEighDispatch` |
| `eigh_vals` | `TypedTensorEighValsDispatch` |
| `eigh_trunc` | `TypedTensorEighTruncDispatch` |
| `left_null`, `right_null` | `TypedTensorNullDispatch` |
| `left_polar`, `right_polar` | `TypedTensorPolarDispatch` |
| `is_posdef` | per-method `where D: FactorizationScalar` |
| device `svd_compact`, `svd_trunc`, `eigh_full`, `eigh_trunc` | `impl<R, D> TensorMap<R, D, CudaStorage<D>> where D: CudaPayload + FactorizationScalar` |

### `AdvancedLinalgScalar`

| Entry point | Site |
| --- | --- |
| `inv` | `TypedTensorInvDispatch` |
| `pinv` | `TypedTensorPinvDispatch` |
| `solve`, `solve_right` | `TypedTensorSolveDispatch` |
| `exp` | `TypedTensorExpDispatch` |
| `powi` | `TypedTensorPowiDispatch` |
| `sqrt` | per-method `where D: AdvancedLinalgScalar` |
| `eig_full` | `TypedTensorEigDispatch` |
| `eig_vals` | `TypedTensorEigValsDispatch` |
| `eig_trunc` | `TypedTensorEigTruncDispatch` |
| Checked-Generic `eig_full`/`eig_trunc` inherent impls | `D: AdvancedLinalgScalar + FactorScalar<Eig = Complex64>` |

## Ambiguous classifications and how they were resolved

Rule applied: when a family assignment was not forced by the #1308 table, the
**narrower (later)** marker wins.

| Entry point | Why ambiguous | Resolution | Reason |
| --- | --- | --- | --- |
| `sqrt` | Implemented elementwise over a diagonal bond tensor (`ScalarOps::sqrt_value`); it never calls a factorization, so on mechanics alone it is base | `AdvancedLinalgScalar` | It is the matrix-function family's square root — the `s.sqrt()` used around `svd_trunc` — and #1308 lists `sqrt` under matrix functions. Narrower marker per the rule. |
| `is_posdef` | A predicate returning `bool`, sitting among base predicates in the same impl block | `FactorizationScalar` | Its non-compact arm calls `eigh_vals`. A payload dtype that cannot be trusted to factorize cannot be trusted to answer this. |
| `is_isometric`, `is_unitary` | Also predicates in the same block, and the survey grouped isometry/unitary concepts with factorization | `TensorScalar` | They are `adjoint` + `compose` + `id` + `norm` only. No factorization is reachable, so the narrower marker would be unjustified rather than merely cautious. |
| `isometry`, `unitary`, `isomorphism`, `id` | The survey's draft table put these constructors under the factorization marker | `TensorScalar` | They are `zeros` plus a partial-identity fill (`Self::structural`); no factorization. #1308's table puts construction in the base family. |
| `diagonal_spectrum` | Returns a spectrum, which elsewhere means SVD/EIGH output | `TensorScalar` | It is `diag` readback of an already-stored compact spectrum; it computes nothing. |
| `absorb` | Bond absorption, adjacent to truncation workflows | `TensorScalar` | Composition/scaling only. |
| `find_truncated` | Truncation is otherwise a factorization concern | `TensorScalar` (unchanged) | It lives on `GradedSpace` and carries no payload at all, as #1308 states. |
| `eig_*` | Already carried `FactorScalar<Eig = Complex64>`, which arguably gated them | `AdvancedLinalgScalar` (in addition) | `FactorScalar` is a dense-payload accessor trait implemented for f32/Complex32 already; it is not an admission boundary. The marker is. |
| device `qr_compact` | The device QR block is `impl<R> TensorMap<R, f64, CudaStorage>` | untouched | Its payload is the concrete `f64`, so there is no generic bound to narrow. Complex64 device QR remains [#1271](https://github.com/Ryo-wtnb11/TeNeT/issues/1271). |
| `tenet-network` (44 bounds) | The network could plausibly be its own marker (the survey proposed `NetworkScalar`) | `TensorScalar` | No network path calls a factorization or a matrix function, and #1308's table places `tensor!` execution in the base family. A separate network marker would be a fourth name with nothing behind it. |

## Internal helper traits

`ScalarOps` (private) keeps its `FactorScalar` supertrait. `FactorScalar`
supplies dense payload accessors (`dense_slice`, `from_real`, `adjoint`,
`epsilon`, `widen_complex`) that the base family needs in every operation; it
does not itself implement factorization logic, which lives in free functions of
`tenet-matrixalgebra` bounded on it. Moving it off `ScalarOps` would have
re-plumbed the base family for no admission gain.

What the split does enforce is that the *dispatch* traits carrying
factorization and advanced logic —
`TypedTensor{Qr,FullQr,Lq,FullLq,Svd,SvdVals,SvdTrunc,Eigh,EighVals,EighTrunc,Null,Polar}Dispatch`
and
`TypedTensor{Inv,Solve,Pinv,Exp,Powi,Eig,EigVals,EigTrunc}Dispatch` —
are themselves unnameable for a `D: TensorScalar`-only caller, because their own
`where` clauses now require the marker. The private
`*_multiplicity_free` implementations of those operations carry the matching
per-method bound for the same reason.

## Re-bounded sites

| Location | `FactorizationScalar` | `AdvancedLinalgScalar` |
| --- | --- | --- |
| Dispatch trait definitions | 12 | 8 |
| `MultiplicityFreeAdmissionMode` dispatch impls | 12 | 8 |
| `CheckedGenericAdmissionMode` dispatch impls | 12 | 8 |
| Public `TensorMap` inherent impl blocks | 12 | 10 (8 public + 2 Checked-Generic `eig` blocks) |
| Device `CudaStorage` impl block | 1 | 0 |
| Per-method `where` clauses | 1 (`is_posdef`) | 1 (`sqrt`) |
| Private `*_multiplicity_free` helpers | 6 | 4 |
| **Production total** | **56** | **39** |
| Test/example helpers in this workspace | 7 (+5 outside `typed.rs`) | 6 (+6 outside `typed.rs`) |

## Workspace-internal generic callers that had to be re-bounded

Downstream code using the concrete `f64`/`Complex64` payloads is unaffected.
Code generic over `D: TensorScalar` that calls a narrowed method is not, and all
such callers inside this workspace are test scaffolding:

| File | Item | New bound |
| --- | --- | --- |
| `tenet/src/typed.rs` | `assert_compact_svd_reads_parent`, `assert_full_svd_reads_parent`, `assert_truncated_svd_reads_parent`, `assert_null_redirect`, `assert_polar_redirect`, `assert_polar_factors`, `assert_qr_lq_keeps_input_cache_cold` | `FactorizationScalar` |
| `tenet/src/typed.rs` | `assert_exp_uses_a_cold_logical_copy`, `assert_sqrt_uses_a_cold_logical_copy`, `assert_inverse_redirect`, `assert_pinv_redirect`, `assert_rank_deficient_polar_support` | `AdvancedLinalgScalar` |
| `tenet/tests/typed_facade.rs` | `compact_bond_trace` | `FactorizationScalar` |
| `tenet/src/typed.rs` | `assert_checked_generic_solve_acceptance` (`racah-generated`) | `AdvancedLinalgScalar` |
| `tenet/tests/checked_generic_facade.rs` | `assert_checked_generic_eigh_factors`, and under `racah-generated` `assert_sun_checked_generic_eigh`, `..._null_projectors`, `..._polar_qh` | `FactorizationScalar` |
| `tenet/tests/checked_generic_facade.rs` | under `racah-generated`: `assert_sun_checked_generic_powi_outer_multiplicity`, `..._inv`, `..._pinv`, `..._solve_right`, and the `SunEigInput` helper trait | `AdvancedLinalgScalar` |
| `tenet/examples/operation_matrix.rs` | `run_mf_eig` | `AdvancedLinalgScalar` |
| `tenet-network/examples/cuda_operation_matrix.rs` | `HarnessScalar` supertrait | `FactorizationScalar + CudaPayload` |

No production call site outside `typed.rs` needed a bound: `tenet-network`,
`tenet-krylov` and the persistence codec never reach a narrowed method.

## Evidence

- `tenet/tests/capability_markers.rs` — positive static assertions: each
  family's representative entry points type-check under exactly its marker, for
  both admitted dtypes.
- `compile_fail` doctests on `FactorizationScalar` and `AdvancedLinalgScalar` —
  a `D: TensorScalar` caller reaches neither `svd_compact` nor `exp`, and a
  `D: FactorizationScalar` caller does not reach `exp`. Each is paired with a
  compiling twin so a `compile_fail` cannot pass for an unrelated reason.

No benchmark: the change adds only marker traits with no methods and no
associated items, so every call still monomorphizes to the same code. Generated
code is unchanged.
