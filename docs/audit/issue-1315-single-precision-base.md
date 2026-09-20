# Single-precision base admission — what H1 opened and what stayed closed

Authority: TeNeT `f032d121` (`origin/main`, which carries the marker split
#1308), issue
[#1315](https://github.com/Ryo-wtnb11/TeNeT/issues/1315), plan
[#1065](https://github.com/Ryo-wtnb11/TeNeT/issues/1065), survey
`reviews/gpu-phase-20260920/single-precision-survey.md`, and the #1308
classification in `issue-1308-capability-markers.md` (which stays
revision-pinned at `053d5a74` and is not rewritten here).

This artifact is revision-pinned evidence, not current capability authority:
`tenet/src/typed.rs`, `tenet/tests/capability_markers.rs` and
`tenet/tests/single_precision_base.rs` are.

## Admission table

| Capability trait | `f64` | `Complex64` | `f32` | `Complex32` |
| --- | --- | --- | --- | --- |
| `TensorScalar` (base family) | yes | yes | **yes (H1)** | **yes (H1)** |
| `FactorizationScalar` | yes | yes | no | no |
| `AdvancedLinalgScalar` | yes | yes | no | no |
| `WireScalar` (persistence) | yes | yes | no | no |
| `CudaPayload` (device) | yes | yes | no | no |

The base family is exactly the one the #1308 table lists under `TensorScalar`:
construction, inspection, `add`/`scale`/`normalize`/`adjoint`, the reductions,
the tolerance-taking predicates, contraction/`compose`/`otimes`/`cat`, the
structural transforms, trace, `restrict_leg`/`embed_leg`/`restrict_diagonal`/
`diagview`, and `tensor!` network execution.

## Gates opened

| Site | Change |
| --- | --- |
| `tenet-operations/src/owned_overwrite_buffer.rs` | `unsafe impl ZeroBytes for f32` / `Complex32`, with the all-zero-bytes invariant stated per impl |
| `tenet/src/typed.rs` `ScalarOps` | impls for `f32` / `Complex32`; new associated type `Wide` |
| `tenet/src/typed.rs` `TensorScalar` | `impl TensorScalar for f32` / `Complex32` |
| `tenet/src/runtime.rs` `Ctxs` | `f32` / `c32` lanes, built on first use |
| `tenet/src/lib.rs` | `pub use num_complex::Complex32` |

Everything in `tenet-dense`, `tenet-operations`, `tenet-tensors` and
`tenet-matrixalgebra` that the base family needs was already generic over the
four dtypes before this leaf; no dtype dispatch was added there.

## Gates deliberately left closed

| Gate | Where | Why |
| --- | --- | --- |
| `FactorizationScalar` | `tenet/src/typed.rs` | factorization tolerances (numerical rank, Hermitian admission, truncation near-ties) need their own review — leaf H4 |
| `AdvancedLinalgScalar` | `tenet/src/typed.rs` | matrix functions and `inv` — leaf H5 |
| `WireScalar` | `tenet/src/typed/serialization.rs` | the wire format has tags for `f64`/`Complex64` only — leaf H8 |
| `CudaScalar` / `CudaPayload` | `tenet-dense/src/cuda_adapter.rs`, `tenet/src/typed.rs` | the device Hermitian constant is `64 * f64::EPSILON` and `download_values` accepts `Tensor::F64` only; both would misbehave silently at `f32` — leaves C0–C4 |
| dtype conversions (`to_f64`, `narrow_to_f32`, `to_c32`) | — | not written; leaf H2 |
| complex structural coefficients with a single-precision payload | `tenet/src/typed.rs` (`MultiplicityFreeContractExecution<R, Complex64>` exists for `Complex64` only) | a `Complex64 -> Complex32` coefficient action is a #1065 non-goal, so this stays a compile-time boundary, as it already was for `f64` payloads |
| `Complex64` Checked-Generic `eig` output | `tenet/src/typed.rs` | unreachable from the base family; leaf H6 |

## Semantics recorded

* **Structural coefficients stay `f64`.** They act on a single-precision
  payload through the existing `tenet-operations/src/scalar.rs`
  `RecouplingCoefficientAction<f64>` impls for `f32` and `Complex32`. This
  follows TensorKit (`promote_permute(Float32, I)` is `Float32` for a real
  sector scalar type). Narrowing an exactly-computed algebraic coefficient is
  not lossy narrowing of user data.
* **Recoupling scratch is per lane.** The GEMM-form recoupling matrix is
  converted once per `coefficient_structure_identity` into the *workspace* of
  the executing context, and each payload dtype owns its own `Ctx`, hence its
  own workspace. An `f32` and an `f64` tensor over the same structure therefore
  cannot read each other's converted scratch;
  `single_and_double_lanes_do_not_share_coefficient_scratch` runs both, on one
  runtime, interleaved and repeated.
* **Plan stores stay shared and `f64`-keyed.** `RuntimeTreeTransformStore<f64>`
  is bound into every real-coefficient lane, including the new ones: the plans
  are structural, not payload-typed, so nothing is duplicated per dtype.
* **`norm`/`norm_inf`/`norm_p` return `f64`; `inner`/`tr` return `D`.** Every
  region partial sum now accumulates in `ScalarOps::Wide` — `Self` for the
  double-precision pair, the double-precision member of the same field for the
  single-precision pair. `f64` and `Complex64` results are bitwise unchanged
  (`Wide = Self`, `widen` is the identity, and the emitted arithmetic is the
  same); `f32`/`Complex32` gain the accuracy of a double accumulator over a
  naive `f32` sum, which TeNeT chooses deliberately over TensorKit's scaled
  per-block `LinearAlgebra.norm`.
* **Mixed payload dtypes stay a compile error**, exactly as `f64 × Complex64`
  already was.
* **`rand` draws in the payload type.** `f32` takes 24 bits and divides by
  `2^23`, so every draw is exactly representable and the half-open `[-1, 1)`
  range survives — narrowing the 53-bit double draw would round to exactly
  `1.0` above `1 - 2^-25` (survey T11).
* **Tolerances have no defaults and are the caller's.** The base predicates
  document that a `tol` chosen for `f64` rejects a good `f32` tensor and must
  be scaled by the payload's epsilon.

## Costs

| Measurement | Before (`f032d121`) | After |
| --- | --- | --- |
| `Runtime::build()` allocation calls | 199 | 199 |
| `Runtime::build()` allocation bytes | 68 512 | 68 656 |
| first `Complex32` tensor on that runtime | — | 76 calls, against 10 for the second |

The 144 bytes are the two `Option<Box<_>>` lane slots and the retained lane
configuration per `Ctxs`, inside allocations the runtime already made. A lane
costs about 66 allocation calls, and is paid only by a program that names the
dtype. Pinned by `tenet/tests/single_precision_allocations.rs`.

The same test pins the per-operation contract: at a 605-entry payload, `f32`
and `f64` perform the identical 25 allocation calls, and `f32` allocates
exactly `3 * 605 * 4` fewer bytes — half the payload of each of the three
payload-sized buffers the measured pipeline produces. `Complex32` against
`Complex64` likewise.

## Evidence

- `tenet/tests/single_precision_base.rs` — the base family at `(f32, f64)` and
  `(Complex32, Complex64)` against the widened-input oracle, over U(1),
  fZ2 × U(1) × SU(2) (fermionic signs, nontrivial braiding, `dim(c) != 1`) and,
  under `racah-generated`, SU(3) with an outer-multiplicity vertex. Covers
  arithmetic, the reductions, every structural transform, lazy adjoint,
  contraction/compose/otimes/cat, rank 5, multi-block, compact diagonal
  storage, `restrict_leg`/`embed_leg`/`restrict_diagonal`/`diagview`, a permute
  round trip, and a negative control that the fixture is genuinely moved by the
  transforms.
- `tenet-network/tests/single_precision_network.rs` — `tensor!` chains, macro
  trace and plan-cache replay, same oracle.
- `tenet/tests/single_precision_allocations.rs` — the two cost contracts above.
- `tenet/src/runtime.rs::lazily_built_single_precision_lanes_inherit_the_runtime_configuration`
  — a deferred lane carries the eager lanes' CPU context, recoupling threads
  and cache policy.
- `tenet-operations/src/owned_overwrite_buffer.rs` tests — `zeroed_payload` at
  all four dtypes, plus a grow-and-drop check of the single-precision
  allocations (the Miri subject for the new `unsafe impl`s).
- `compile_fail` doctests on `TensorScalar`, each with a compiling `f64`/
  `Complex64` twin that differs only in the payload dtype: `svd_compact`,
  `exp`, `to_bytes_with`. The `to_cuda` rejection is pinned at `to_cuda`
  itself (already present for `Complex32`); its twin sits on `TensorScalar`
  under the `cuda` feature.

Miri was not run: `cargo-miri` is not installed in this environment, and the
brief forbids installing toolchains. The two new `unsafe impl`s add no code —
they widen the input domain of `zeroed_payload`, whose existing unsafe block is
unchanged — and the tests that would exercise them under Miri are named above.

## Tolerance

`K * sqrt(n) * eps(real(D)) * max(1, max|expected|)` with `K = 32` and `n` the
number of payload entries the operation sums over. `K` covers the recoupling
coefficients applied on top of the naive-sum bound; no test needed a larger
constant.

## Residual hazards for the later leaves

- **T1** `scaled_hermitian_residual_accepts` is `64 * f64::EPSILON`
  (`tenet-dense/src/cuda_adapter.rs`): it rejects every `f32` Hermitian block.
  A hard blocker for device EIGH — leaf C1.
- **`download_values`** accepts `Tensor::F64` only, so an `f32` device SVD/EIGH
  would error on its spectra — leaf C1.
- **T2** the `budget + 1e-15` absolute slack on discarded weight
  (`tenet-matrixalgebra/src/truncation.rs`) is noise-dominated at `f32`
  precision — leaf H4 must document it with a near-tie fixture.
- **T4** `CgOptions::default().rtol = 1e-12` is unreachable for an `f32`
  operator. `tenet-krylov` scalars are `f64` by contract, so nothing misfires
  today.
- **Checked-Generic `eig`** returns `Complex64` rather than `D::Eig`; admitting
  a single-precision payload to `AdvancedLinalgScalar` before H6 would widen
  silently.
- **`tenet::matrixalgebra::svd_compact`** on `core::TensorMap` already accepts
  `D: FactorScalar`, hence `f32`, outside every marker. Pre-existing expert-layer
  exception, unchanged by this leaf (#1308 review, P2-3).
