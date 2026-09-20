# Single-precision base admission — what H1 opened and what stayed closed

Authority: TeNeT `89b1cde6` (`origin/main`; the branch was rebased onto it
after the independent review, and it carries the marker split #1308), issue
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
| `tenet/src/typed.rs` `ScalarOps` | impls for `f32` / `Complex32`; new supertrait `WideScalar<Wide: FactorScalar>` |
| `tenet-operations/src/scalar.rs` | new `WideScalar` trait (the reduction accumulator, one authority for every crate) |
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
  reduction that sums a whole block, coupled region or spectrum accumulates in
  `WideScalar::Wide` — `Self` for the double-precision pair, the
  double-precision member of the same field for the single-precision pair. The
  claim covers **all** of them, on every storage form, which is what the
  independent review returned the first revision for:

  | Path | Site |
  | --- | --- |
  | dense coupled region | `typed.rs::coupled_region_inner`, `weighted_inner`, `weighted_trace` |
  | compact diagonal | `typed.rs::compact_inner`, `tr_multiplicity_free`, `trace_pairs_multiplicity_free` |
  | lazy adjoint | `tenet-tensors/src/oriented_elementwise.rs::oriented_fusion_inner[_with]` and the `tenet-operations` kernel it calls, `bilinear_raw_strided_kernel_mapped` |

  `f64` and `Complex64` results are unchanged **bit for bit**: `Wide = Self`,
  `widen`/`narrow` are the identity, and where the oriented total applies a
  quantum-dimension weight the accumulator is multiplied by
  `coefficient_as_data(w)` — `w + 0i` for a complex accumulator — which is the
  exact expression the loop used before, rather than the componentwise
  `scale_by_coefficient`, so even the signed zeros and the non-finite cases
  agree. Pinned by
  `tenet/tests/reduction_accumulator_precision.rs` against bit patterns
  captured on `origin/main` 89b1cde6 in a detached worktree with its own
  target directory. Signed zeros and the componentwise-versus-full-complex
  weight product are the one difference no public input can expose, because a
  signed zero is flattened by the `+0.0` the accumulator starts from; the
  conservative form is kept regardless, and the test records why.
  `f32`/`Complex32` gain the accuracy and the range of a double accumulator
  over a naive `f32` sum, which TeNeT chooses deliberately over TensorKit's
  scaled per-block `LinearAlgebra.norm`.
* **Plan-cache observability is aggregated, not per dtype.**
  `Runtime::tree_transform_cache_info` and `clear_tree_transform_cache` report
  and clear single- and double-precision activity together, because the plans
  are structural and the store is shared. Recorded in the `TensorScalar`
  rustdoc. Nothing dtype-dependent is stored under a dtype-free key: the
  recoupling scratch is a `Vec<D>` inside a `Ctx<D>` workspace, so cross-dtype
  reuse is impossible by type, and the network pools key on `TypeId<(R,D,S)>`.
* **A new payload dtype is a minor source break (RFC 1105).** A payload dtype
  inferred *solely* from float literals, whose result then has a method called
  on it, now fails with `E0689`: with one float impl rustc unified `{float}`
  with `f64` through impl selection eagerly, with two it waits for the
  end-of-function fallback, which is too late for method resolution. Affects
  `diagonal`, `from_block_fn` closures returning literals, and `scale`/`add`
  coefficients on an un-annotated `zeros`/`id`/`rand`; the fix is to annotate
  the payload type. Documented in the `TensorScalar` rustdoc with a compiling
  example. One in-tree call site needed it
  (`tenet-network/tests/trace_macro.rs`).
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

Measured on macOS/aarch64, `dense_threads(1)`, debug build. These are
**evidence, not assertions**: an absolute allocation count depends on the
platform, the allocator and the core count the CPU context sizes its pool
from, so the tests assert only relations between measurements taken in the
same process.

| Measurement | Before (`f032d121`) | After |
| --- | --- | --- |
| `Runtime::build()` allocation calls | 199 | 199 |
| `Runtime::build()` allocation bytes | 68 512 | 68 656 |
| first `Complex32` permute on a runtime already warmed with `f64` | — | 58 calls, against 22 for the second |

The 144 bytes are the two `Option<Box<_>>` lane slots and the retained lane
configuration per `Ctxs`, inside allocations the runtime already made. A lane
costs about 36 allocation calls, and is paid only by a program that names the
dtype — but it is paid *per `Ctxs`*, not per `Runtime`: each pooled
`TensorExecutionContext` has an `mf` and a `generic` namespace, so a program
that uses single precision on `max_idle` pooled contexts builds up to
`2 * max_idle` lanes per dtype. That is the same shape as the double-precision
lanes it already pays for eagerly. The contract the test pins is structural and platform-independent: building a
`Runtime` costs the same whether or not single precision was used earlier in
the process, and on a runtime already warmed with the `f64` operation — which
fills the layout admission and the structure-keyed, `f64`-keyed plan store the
lanes share — the first single-precision operation still allocates more than
the second. That remaining difference *is* the lane.

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
- `tenet/tests/contraction_output_allocations.rs::owned_su2_single_precision_contraction_allocates_like_double`
  — the same contract on the SU(2) compose fixture, which is where the
  coefficient scratch is converted per structure identity.
- `tenet/tests/reduction_accumulator_precision.rs` — where the reductions
  accumulate: 39 `f64`/`Complex64` bit-for-bit pins plus 10 Checked-Generic
  ones (SU(3), behind `racah-generated`), over dense, compact-diagonal and
  lazy-adjoint storage, `norm`/`inner`/`tr`/full trace, an abelian provider
  with `dim(c) == 1` and a non-abelian one with `dim(c) == 2`, and a `-0.0`
  fixture on the lazy-adjoint complex path. The entries are non-dyadic, so a
  reordered or re-associated accumulation changes the bits — visible in the
  table itself, where `f64 su2 dense inner` and `f64 su2 lazy adjoint inner`
  differ in the last bit because the two paths sum in different orders. And
  three single-precision pins that distinguish a wide from a narrow
  accumulator by construction (one entry of `8192.0f32` and the rest `1.0f32`: `8192^2` is
  `2^26`, whose `f32` step is 8, so a narrow accumulator swallows every `1.0`
  and a wide one does not — the difference survives narrowing back to `f32`),
  plus an overflow fixture (`1e20f32`) whose `norm` must be finite on dense,
  compact and lazy-adjoint storage.
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
- **T8 (H5):** `tenet-matrixalgebra/src/matrix_functions.rs` Padé-13 constants
  are double-precision. They are representable and correct for `f32`, so this
  is an accepted cost (an over-accurate degree), not a defect — but H5 should
  record it rather than rediscover it.
- **C0/C4:** `Complex32` on CUDA must be assumed to share the unreleased
  complex-kernel defect that blocks `Complex64` device QR (tenferro-rs#1833:
  complex `zero_value` in `triu`/fill/diagonal kernels), and `f32` itself is
  unverified on NVRTC/cuTENSOR — TensorKit reports a Float32 cuTENSOR failure
  in its own wrapper. A device probe leaf comes first.
- **C3:** `tenet-dense/src/cuda_adapter.rs::warm_up` exercises `f64` only, so
  the first single-precision device call would pay NVRTC/cuSOLVER
  initialisation.
- **CI:** the Checked-Generic (SU(3)) coverage added here is behind
  `racah-generated`. Whether that feature's job runs these files is tracked in
  #1293; until it does, that acceptance bullet is verified locally only.
- **`tenet::matrixalgebra::svd_compact`** on `core::TensorMap` already accepts
  `D: FactorScalar`, hence `f32`, outside every marker. Pre-existing expert-layer
  exception, unchanged by this leaf (#1308 review, P2-3).
