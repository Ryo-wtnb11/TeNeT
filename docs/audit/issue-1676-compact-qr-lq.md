# Owned Host diagonal QR and LQ

Authority: `34aef370e5d220713300c04772b3caa144d83c52` (2026-10-02), issue
#1676 current body. Its approved semantic decision is `W = V`, including dual
orientation, for admitted owned compact diagonal inputs. Both factors retain
compact storage. The ordinary dense factor-space convention is unchanged.

## Reference equivalence

TensorKit `cfaa073e4d1e3eb2167edcbdc3be9872f41e7d91`,
`src/factorizations/diagonal.jl:16-42,61-65`, initializes QR as `(d, similar(d))`
and LQ as `(similar(d), d)`, then repacks each diagonal for MatrixAlgebraKit.
MatrixAlgebraKit 0.6.9 `src/implementations/qr.jl:_diagonal_qr!` (300-331),
`src/implementations/lq.jl:lq_diagonal!` (331-363), and
`src/common/safemethods.jl:sign_safe` (10-11) prescribe phase/absolute-value
factors and phase +1 at zero. Compact and full coincide on square blocks.

QSpace `dd2cc7e10dc7d3917b23309a44d1fe67adb4dc43`,
`Source/orthoQS.cc:172,237,271,352` dispatches to
`Source/mpsortho.cc:orthoQS` (1032-1074), then `SVD_Data::blockSVD` /
`wbSVD` (546-612). This is grouped SVD orthogonalization, with no corresponding
compact-diagonal QR/LQ specialization. TeNeT preserves sector grouping and
uses TensorKit's selected diagonal arithmetic; it does not imitate the SVD.

The input is the proved bond endomorphism `V <- V`. The output uses the exact
same admitted space, coupled-sector ids, dual flags, tree basis, block identity,
and diagonal order. There is no braiding, conjugation, transpose, accumulation,
packing, scatter, or backend submission. QR publishes phase then magnitude;
LQ reverses those two factors. Reusing the source space avoids the generic
factor planner's new nondual bond and makes both factors valid existing
`TypedData::Diagonal` values. No categorical coefficient is recomputed.

Rust deviations: immutable source and two newly owned spectra replace Julia's
in-place copied input. Scale-first complex phase formation (already used by
the polar seam) avoids loss of isometry for subnormal components. Nonfinite
components and magnitudes unrepresentable in the payload dtype decline direct
admission. Invalid sector/layout admission likewise retains the dense route.

For `C` sectors, `K = sum k_c`, `P = sum k_c^2`, the spectrum arithmetic and
publication are O(K + C) work/storage after validating the sector layout. The
owned output is intentional. There is no retained plan, workspace, or cache;
no P-element input/output payload and no dense solver are needed.

## Authority graph and consumers

- Public `typed/scalar.rs:{qr_compact,qr_full,lq_compact,lq_full}` lowers roles
  through `transform_ops.rs:with_leg_roles` before mode dispatch. Exact roles
  borrow; other roles first use the existing transform. Only an owned compact
  diagonal that survives lowering qualifies.
- `mode_dispatch.rs` selects the multiplicity-free Host methods. They share
  one diagonal QR spectrum admission seam; LQ exchanges its output order.
  Dense and lazy-adjoint paths keep their existing matrixalgebra methods.
  Lazy adjoint parents are dense by the `TypedAdjointView` invariant.
- Checked Generic dispatch in `mode_dispatch.rs` / `scalar.rs` and CUDA
  dispatch in `cuda_ops.rs` / `cuda_factor.rs` do not use this admission seam.
  CUDA upload expands Host compact storage; no device compact claim is made.
- There is no QR/LQ prepared or profiled Host compact wrapper. Benchmarks and
  ordinary callers reach the same public methods. Batched/CUDA planning and
  the static/expert dense matrixalgebra APIs remain independent dense paths.
- `diagview`, `dense_data`, `materialize`, `map_diagonal`, serialization,
  `compact_adjoint`, compact compose, norms and scalar functions all require
  the existing exact bond-endomorphism invariant. The source-space clone and
  sector-preserving spectra satisfy it without broadening representation.
- `polar.rs` shares only the scalar phase/magnitude routine; its existing
  geometry, dense publication and error ordering remain intact.

## Observable decision and verification

The factor domain/codomain preserves a dual input's dual orientation. Both
factors support `diagview`; callers requiring a dense buffer call `materialize`
before `dense_data`. Dense source inputs still return ordinary dense factors.
The independent oracle is the hand diagonal formula, reconstruction and
isometry, with explicit space equality and compact storage assertions.
The negative control uses a QR-forbidden executor: the old route invokes it.

Issue impact: #1676 SOLVED by this diagonal rule, subject to verification and
independent review; umbrella #1615 PARTIAL (other operation leaves remain).
#1690/#1691 polar semantics are unchanged; shared arithmetic is regression
covered. No dependency, cache, profiler or CUDA resource contract changes.

## Dense LQ control

The preexisting `extend_adjoint_col_major` StepBy/Take iterator acquired an
out-of-line per-row fold in Release after this change. Full-call controls
exposed a repeatable 3–5% real dense compact-LQ regression. Removing only the
new admission check did not remove it; forcing the outer helper boundary did
not remove it either. Explicit element pushes worsened both real and complex
controls. An exact-size index range passed to `Vec::extend` restored timing.
The adopted bounded change keeps the single reserve, column-major traversal,
conjugation and append semantics. A hand-ordered rectangular complex fixture
and all-scalar empty/append cases protect these semantics. This is an ordinary
dense data-movement loop; it adds no dispatch, framework, provider or cache.

Final gate and benchmark details, including experimental non-improvements and
excluded invalid binary reuse, are recorded in
[`benchmarks/issue_1676_results`](../../benchmarks/issue_1676_results/README.md).
Timing is measurement evidence, not a CI acceptance assertion.
