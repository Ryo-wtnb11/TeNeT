# Single-precision factorizations — what H4 opened and what stayed closed

Authority: TeNeT `a06c3a78` (`origin/main`, carrying the marker split #1308 and
the base admission #1315), issue
[#1324](https://github.com/Ryo-wtnb11/TeNeT/issues/1324), plan
[#1065](https://github.com/Ryo-wtnb11/TeNeT/issues/1065), survey
`reviews/gpu-phase-20260920/single-precision-survey.md`, and the earlier audits
`issue-1308-capability-markers.md` and `issue-1315-single-precision-base.md`,
which stay revision-pinned and are not rewritten here.

This artifact is revision-pinned evidence, not current capability authority:
`tenet/src/typed.rs`, `tenet-matrixalgebra/src/{factorize,truncation}.rs`,
`tenet/tests/capability_markers.rs` and
`tenet/tests/single_precision_factorizations.rs` are.

References used, with revisions:

* TensorKit.jl `cfaa073e4d1e3eb2167edcbdc3be9872f41e7d91`
  (`codebases/refs/TensorKit.jl`).
* MatrixAlgebraKit v0.6.9 (`~/.julia/packages/MatrixAlgebraKit/AbU5a`).
* QSpace v4: **no corresponding path.** QSpace data is `double` /
  `wbcomplex{double,double}` only, MEX input must be `mxDOUBLE_CLASS` and the
  BLAS bindings are `dgemm`/`zgemm`; it contributes nothing to single precision
  (survey §3).

## Admission table

| Capability trait | `f64` | `Complex64` | `f32` | `Complex32` |
| --- | --- | --- | --- | --- |
| `TensorScalar` (base family) | yes | yes | yes (H1) | yes (H1) |
| `FactorizationScalar` | yes | yes | **yes (H4)** | **yes (H4)** |
| `AdvancedLinalgScalar` | yes | yes | no | no |
| `WireScalar` (persistence) | yes | yes | no | no |
| `CudaPayload` (device) | yes | yes | no | no |

`FactorizationScalar` is the #1308 family verbatim: QR/LQ (compact and full),
SVD (compact, full, values, truncated), Hermitian eigendecomposition (full,
values, truncated), `left_orth`/`right_orth`, left/right null spaces,
left/right polar, and `is_posdef`. `GradedSpace::find_truncated` carries no
payload and stays on `TensorScalar`.

## Gates opened

| Site | Change |
| --- | --- |
| `tenet/src/typed.rs` | `impl FactorizationScalar for f32` / `Complex32` |
| `tenet-matrixalgebra/src/factorize.rs` | `impl SpectrumMagnitude for f32` / `Complex32` |

That is the whole production diff. Everything the family dispatches to was
already generic over `FactorScalar`, which `tenet-matrixalgebra` has
implemented for the single-precision pair since before #1065, with
component-typed `epsilon`/`safe_minimum`, a component-typed Hermitian
admission tolerance and an `eps(real(D))`-scaled numerical rank. No dispatch
trait, plan, cache or adapter needed a new arm: a `cargo check --workspace`
after adding the two marker impls compiled clean, which is the compile-time
proof that every `D: FactorizationScalar` body is instantiable at the new
dtypes.

`SpectrumMagnitude` was the one genuine gap. `TensorMap::diagview` returns
`SectorSpectrum<_, D>` in the **payload** dtype, and
`GradedSpace::find_truncated` takes any `V: SpectrumMagnitude`; without the two
new impls a single-precision `diagview` could not be fed to a truncation
decision at all. Both widen before taking the absolute value or the
hypotenuse, so a `Complex32` whose components straddle the `f32` range still
reports a finite magnitude. Neither offers `nonnegative_f64_slice`: an `f32`
slice is not an `f64` slice, so the selection materializes the magnitudes,
exactly as `Complex64` already does.

## Gates deliberately left closed

| Gate | Where | Why |
| --- | --- | --- |
| `AdvancedLinalgScalar` | `tenet/src/typed.rs` | matrix functions, `inv`/`pinv`/`solve` — leaf H5 |
| general `eig` | `tenet/src/typed.rs` | the Checked-Generic arm still returns `Complex64` rather than `D::Eig`; admitting a single-precision payload would widen silently — leaf H6 |
| `WireScalar` | `tenet/src/typed/serialization.rs` | the wire format has tags for `f64`/`Complex64` only — leaf H8 |
| `CudaScalar` / `CudaPayload` | `tenet-dense/src/cuda_adapter.rs`, `tenet/src/typed.rs` | the device Hermitian constant is `64 * f64::EPSILON` and `download_values` accepts `Tensor::F64` only — leaves C0–C4 |
| dtype conversions | — | leaf H2 |
| complex structural coefficients with a single-precision payload | `tenet/src/typed.rs` | a `Complex64 -> Complex32` coefficient action is a #1065 non-goal; unchanged compile-time boundary |

Pinned by `compile_fail` doctests on `TensorScalar`, each with a compiling
double-precision twin that differs only in the payload dtype: `exp`, `inv`,
`eig_full`, `to_bytes_with`, and `to_cuda` (at `to_cuda` itself, under the
`cuda` feature). The `svd_compact` `compile_fail` that #1315 put there is gone
— it is exactly what this leaf opens — and `tenet/tests/capability_markers.rs`
now instantiates its `factorization_family` helper at all four dtypes while
still refusing to instantiate `advanced_family` and `general_eig_family` at the
single-precision pair.

## Tolerance audit

Every constant reachable from the factorization family, classified as
**(a)** already `eps(real(D))`-scaled, **(b)** caller-supplied, **(c)**
`f64`-specific and changed, **(d)** `f64`-specific but dtype-independent by
design.

| Site | Value | Class | Change | Reference |
| --- | --- | --- | --- | --- |
| `factorize.rs::FactorScalar::epsilon` (f32/Complex32) | `f32::EPSILON` | a | none | LAPACK `slamch('P')`; MAK `defaulttol` is `eps(real(T))^(2/3)` (`src/common/defaults.jl:10`) |
| `factorize.rs::FactorScalar::safe_minimum` (f32/Complex32) | `f32::MIN_POSITIVE` | a | none | LAPACK `slamch('S')`, `sgebal.f:330` |
| `factorize.rs::HermitianReal::relative_tolerance` | `64 * Self::EPSILON` in the *component* type, selected by `hermitian_matrix_contents` | a | none | MAK `default_hermitian_tol(A) = eps(norm(A,Inf))^(3/4)` (`src/common/defaults.jl:44`) — also element-typed; TeNeT's is tighter and scale-invariant |
| `factorize.rs::numerical_rank_and_compact_basis` (null spaces, polar) | `D::epsilon() * max(m,n) * sigma_max` | a | none | TensorKit `_default_rtol(t) = eps(real(float(scalartype(t)))) * min(dim(domain), dim(codomain))` (`src/tensors/linalg.jl:284`); MAK null-space default `rtol = defaulttol(eltype(A))` (`src/algorithms.jl:277`). Both scale with `eps(real(T))`; TeNeT uses `max` where TensorKit uses `min`, a pre-existing and more conservative choice |
| `truncation.rs::kept_counts` `Truncation::Rank` | `budget + 1e-12` | d | none | The budget is a sum of `f64` quantum dimensions and the weights are `f64` at every payload dtype; nothing about it narrows. TensorKit `truncrank` (`src/factorizations/truncation.jl:261`) is exact |
| `truncation.rs::kept_counts` `Truncation::DiscardWeight` | `budget + 1e-15` | **d** (survey T2 proposed c) | **none — see below** | MAK `_truncerr_impl` (`src/implementations/truncation.jl:91-102`) has no slack at all, and accumulates in the element type |
| `truncation.rs` `Tolerance` / `ToleranceInf` thresholds | `max(atol, rtol * norm)` | b | none | MAK `findtruncated(::TruncationByValue)` (`src/implementations/truncation.jl:64`), same expression |
| `typed.rs::is_posdef` / `is_hermitian` / `is_isometric` / `is_unitary` | `tol * max(norm, 1)`, no default | b | rustdoc only | TensorKit `isapprox` default `Base.rtoldefault` = `sqrt(eps(real(T)))` (`src/tensors/abstracttensor.jl:684`); MAK `is_left_isometric(...; rtol = defaulttol(A))` (`src/common/matrixproperties.jl:49`) |
| `typed.rs` exact-zero singularity tests | `== 0.0` | d | none | exact by intent; TensorKit the same |
| `matrix_functions.rs` Padé-13 constants, Blue's scaled-norm constants | double-precision | d | none — and unreachable from this family | leaf H5 (survey T8/T9) |

### The discarded-weight slack (survey T2): analysed, deliberately unchanged

The brief asked for `budget + 1e-15` to be made relative to the epsilon of the
payload's real type. It should not be, and this is the reasoned record of why.

* **What the slack guards is not payload-typed.** The loop it sits in
  accumulates `discarded + weight * value * value` where `discarded`, `weight`
  and `value` are all `f64` *at every payload dtype*: `FactorScalar::real_spectrum`
  and `compute_f64_spectrum` widen a single-precision spectrum rather than
  recomputing it, `WeightedSpectrum::values` is `&[f64]`, and the quantum
  dimensions are `f64`. The arithmetic the slack absorbs is bit-for-bit the
  same at `f32` as at `f64`. A payload-dependent slack would therefore not
  absorb a payload-dependent error; it would change the *policy*.
* **The reference argues the same way, harder.** MatrixAlgebraKit's
  `_truncerr_impl` compares the cumulative tail against the budget with **no
  slack**, and it accumulates in the element type — so at `Float32` the
  reference's own decision is noisier than TeNeT's. TeNeT is already the more
  accurate of the two at single precision; widening its slack by
  `eps(f32)` would move it in the wrong direction and away from the reference.
* **What single precision actually changes is upstream of the decision**: the
  singular values and eigenvalues themselves carry a relative error of order
  `eps(f32) * cond`, so two values closer together than that are not ordered
  reliably. No slack fixes that, and the honest contract is the documented one
  — the kept *set* near a tie may differ from the double-precision run while
  the discarded *weight* agrees. That is stated on `FactorizationScalar` and on
  `Truncation`, and demonstrated by
  `a_near_tie_may_keep_a_different_state_but_not_a_different_weight`.
* **Bitwise neutrality was the tie-breaker the brief named.** Making the slack
  relative to the budget or to `norm^2` cannot keep `f64` decisions bitwise
  unchanged: `1e-15` absolute and `k * eps * norm^2` relative agree only at one
  scale. The brief's instruction for that case is to stop on the constant and
  report, which is what this section does.
* **Residual, recorded rather than fixed:** the slack being *absolute* is a
  genuine scale dependence, and it affects `f64` exactly as much as `f32` — a
  spectrum scaled up by `10^6` gets a relatively `10^12` times tighter slack.
  Changing it is a decision about the double-precision contract and needs its
  own leaf, with its own `f64` evidence. The survey (T2) reaches the same
  conclusion: "if changed, make it relative to `budget` in its own leaf
  (affects f64 too)". The reasoning is now a comment at the site so the next
  reader does not have to rediscover it.

### Evidence that `f64` / `Complex64` are bitwise unchanged

1. **By construction.** The production diff is two marker impls and two
   `SpectrumMagnitude` impls for types that had none. No expression on a
   `f64`/`Complex64` path is touched; no shared arithmetic is touched. Adding
   an impl for a new type cannot change the code generated for an existing one
   — there is no blanket impl and no specialization in this crate.
2. **By measurement.**
   `double_precision_truncation_decisions_are_bitwise_unchanged` pins eight
   `f64` truncation decisions — kept dimension and `error.to_bits()` — over a
   non-dyadic spectrum (multiples of a tenth, so no partial sum is exact),
   including a case whose budget is *exactly* the weight of the two smallest
   tails, where `0.1 * 0.1` twice sums to `0.020000000000000004`, four ulps
   above the budget, and the `1e-15` slack is the only reason the second tail
   goes. Every expected value was measured on a **detached `origin/main`
   worktree at `a06c3a78` with its own target directory**
   (`tenet-h4-basecheck-20260921` / `tenet-h4-basecheck-target-20260921`, both
   removed afterwards), not read back from this branch, and they match here bit
   for bit. Composite (`Rank ∧ DiscardWeight`) is included so the intersection
   path is pinned too.
3. The existing `f64` truncation and factorization suites
   (`truncation_composition.rs`, `truncation_composition_checked_generic.rs`,
   `truncspace.rs`, `mf_linalg_conformance.rs`, `checked_generic_facade.rs`,
   `user_decomp.rs`, `cpu_shape_svd.rs`) pass unchanged.

## Tolerance used in the tests

`K * sqrt(n) * eps(f32) * max(1, scale) * kappa` with `K = 32` (inherited from
#1315), `n` the payload length the operation sums over, `scale` the largest
magnitude in the oracle, and `kappa` the **measured** condition number of the
fixture — computed from the double-precision `svd_vals` of that very fixture,
asserted to stay below `1e4`, and printed in every failure message. `kappa`
multiplies the bound only where the quantity under test is a forward error:

| Law | `kappa` | Why |
| --- | --- | --- |
| reconstruction (`q∘r`, `u∘s∘vh`, `v∘d∘v†`, `w∘p`) | 1 | backward stable |
| isometry / co-isometry / unitarity | 1 | backward stable |
| gauge-fixed factors compared pointwise (`Q`, `R`, `L`, polar `W`, `P`) | measured | forward error of a solve |
| null-space annihilation | measured | forward error |
| spectra, truncation `error` | 1 | ordered, backward stable |

`PREDICATE_TOL = 1e-4` for `is_hermitian`/`is_posdef`: MatrixAlgebraKit's
`defaulttol` at `f32` is `eps(f32)^(2/3) = 2.4e-5`, the same order. No absolute
constant appears in any assertion about a computed quantity; the only bare
numbers are this documented predicate tolerance and the `kappa < 1e4` sanity
gate on the fixtures themselves.

## Evidence

* `tenet/tests/single_precision_factorizations.rs` — the whole family at
  `(f32, f64)` and `(Complex32, Complex64)` over U(1), fZ2 × U(1) × SU(2)
  (fermionic signs, nontrivial braiding, `dim(c) != 1`) and, under
  `racah-generated`, SU(3) on the Checked-Generic dispatch. Gauge-fixed factors
  against the widened-input oracle pointwise; gauge-dependent ones through
  reconstruction, isometry and annihilation identities; spectra pointwise and
  for their ordering convention (singular values descending and non-negative,
  Hermitian eigenvalues descending in magnitude); `qr`/`lq` positive real
  diagonal; truncation on a fixture away from ties; a compact-diagonal receiver
  including `find_truncated` fed from a single-precision `diagview`; a
  lazy-adjoint receiver; and, on the Checked-Generic path, a negative control
  that the lazy-adjoint rejection is worded identically at both precisions.
* `tenet/tests/single_precision_factorizations.rs::a_near_tie_may_keep_a_different_state_but_not_a_different_weight`
  — two states one `f32` ulp apart in different coupled sectors. Asserts the
  kept *count* and the discarded *weight*; reports, rather than asserts, which
  candidate survived.
* `tenet/tests/single_precision_factorizations.rs::double_precision_truncation_decisions_are_bitwise_unchanged`
  — the `f64` pins described above.
* `tenet/tests/single_precision_allocations.rs::single_precision_factorizations_allocate_as_often_as_double`
  — same factor shapes, no more allocation calls than the double-precision
  twin, and at least the narrowing of the returned factors in bytes. Measured
  (macOS/aarch64, `dense_threads(1)`, debug): `f64` 308 calls / 252 235 B,
  `f32` 279 / 202 423; `Complex64` 291 / 311 731, `Complex32` 291 / 237 987,
  at 2 226 produced payload entries. **Equality of call counts is not the
  contract**: the four dtypes already disagree among themselves at
  `origin/main`, because `FactorScalar::compute_f64_spectrum` is overridden for
  the double-precision pair and allocates a fresh spectrum vector where the
  single-precision pair reuses the caller's scratch. The assertion is the
  structural one — narrowing the payload never makes the factorization
  allocate more.
* `tenet/tests/capability_markers.rs` — `factorization_family` instantiated at
  all four dtypes, `advanced_family` and `general_eig_family` at the
  double-precision pair only.
* `tenet/tests/single_precision_oracle/mod.rs` — the oracle harness (exactly
  widening 24-bit draw stream, `Parts`, tolerance, fixtures), moved out of
  `single_precision_base.rs` so both single-precision suites share one
  definition rather than a copy. `single_precision_base.rs` is otherwise
  unchanged and still green.

## Residual hazards for the later leaves

* **The absolute discarded-weight slack** (above) is scale-dependent at *every*
  dtype. Own leaf, `f64` contract decision.
* **`SpectrumMagnitude` has no `nonnegative_f64_slice` for `f32`**, so a
  single-precision truncation materializes one `Vec<f64>` of magnitudes per
  sector where a `f64` spectrum borrows. Same asymptotics, one extra
  `O(rank)` allocation per sector per truncated factorization. Removing it
  would need a magnitude view over a foreign element type; not worth a new
  abstraction at this size.
* **Checked-Generic factorizations reject a lazy adjoint operand** ("checked
  Generic contraction currently requires direct owned tensors", "checked
  Generic svd_vals does not accept lazy adjoints"). Pre-existing, dtype-
  independent; the new suite pins that the rejection does not become
  dtype-dependent, and asserts the isometry laws on that path indirectly
  (through pointwise agreement with the `f64` factors, which
  `checked_generic_facade.rs` proves isometric) rather than through an adjoint
  contraction.
* **Multiplicity vertices are not exercised *inside* a factorization.** Every
  fixture here is rank two, so each coupled sector carries one fusion tree.
  Outer multiplicity is a property of block assembly, which
  `single_precision_base.rs` covers at SU(3) for the base family and the `f64`
  suites cover for factorizations; a single-precision rank-≥3 SU(3)
  factorization fixture would close the last corner. H5/H6 should add one when
  they extend this file.
* **T1 / `download_values` (C1)**, **T4 (`CgOptions::default().rtol`)**,
  **T8 Padé-13 constants (H5)**, **Checked-Generic `eig` widening (H6)**,
  **C0/C4 device** — unchanged from the #1315 audit.
* **CI:** the SU(3) coverage here is behind `racah-generated`. Whether that
  feature's job runs this file is tracked in #1293; until it does, that
  acceptance bullet is verified locally only.
* **`tenet::matrixalgebra::svd_compact`** on `core::TensorMap` still accepts
  `D: FactorScalar` outside every marker. Pre-existing expert-layer exception
  (#1308 review, P2-3) — and from this leaf on it is no longer a gap, since
  `f32` reaches the typed `svd_compact` too.
