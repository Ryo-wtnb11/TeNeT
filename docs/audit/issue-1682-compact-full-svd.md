# Compact S for owned diagonal full SVD

Authority: main `8dd296d54d6a0eb87c1dae26ac0c1021c27a8fc7`, issue #1682
current body and #1615 structural rule. The leaf changes S storage for admitted
owned Host multiplicity-free diagonal inputs, preserving full-factor spaces.

## Reference and equivalence

TensorKit `cfaa073e4d1e3eb2167edcbdc3be9872f41e7d91`,
`src/factorizations/diagonal.jl:initialize_output(svd_full!, ..., DiagonalAlgorithm)`
(lines 50–58) selects dense U/Vh and `similar_diagonal` S. Its diagonal compact
SVD delegates to full SVD (68–72). MatrixAlgebraKit 0.6.9
`src/implementations/svd.jl:initialize_output` (96–102) and
`svd_full!(..., DiagonalAlgorithm)` (241–269) prescribe per-sector descending
absolute values, stable source-index tie order, U permutation and Vh phase;
zero phase is +1. TeNeT's existing diagonal compact factor builder implements
this sequence and scales complex phase formation to handle subnormals.

QSpace `dd2cc7e10dc7d3917b23309a44d1fe67adb4dc43`,
`Source/svdQS.cc:MPS_ORTHO_1` (151–209) permutes then calls
`Source/mpsortho.cc:getSVD` (1081–1115), which calls `SVD_Data::blockSVD`
(546–612) and truncation/publication. This is grouped sector SVD; no compact
input specialization corresponds to this leaf. TeNeT retains its reduced
sector grouping and uses the TensorKit diagonal dispatch's arithmetic.

The source invariant is exactly a bond endomorphism V <- V. Each reduced
block is square k_c by k_c. `region.rs:compact_bond_leg` chooses the nondual
leg W with coupled-sector dimensions min(k_c,k_c)=k_c. Full SVD's
`build_bound_factor_with_placement` and `rectangular_diagonal_bond_tensor`
choose the same sector dimensions and nondual orientation independently on
both sides. Hence reuse preserves U: V <- W, S: W <- W, Vh: W <- V, including
a dual V. The source external tree identities/order are validated by
`compact_factor_routes_preserve_tree_order`. There is no recoupling, braiding,
pivotal coefficient or fermion exchange. U and Vh are generally permutations,
so their existing dense representation is required; S is diagonal on W.

Rust uses borrowed input and newly owned outputs instead of Julia's copied
mutable input. The facade keeps its existing common payload dtype D for all
factors: complex S stores real magnitudes with zero imaginary part, whereas
TensorKit can return a real-valued S. No new scalar/result type is introduced.
Sort costs O(sum k_c log k_c), dense U/Vh publication costs
Theta(sum k_c^2), and S stores O(sum k_c). No dense input or S payload, solver,
retained plan, cache or backend workspace is needed on the direct route.
Nonfinite/unrepresentable spectra or declined layouts retain the existing
fallback. No new admission policy or dependency is introduced.

## Consumer map

`scalar.rs:svd_full` first lowers leg roles with `with_leg_roles`, then
`mode_dispatch.rs` selects the Host multiplicity-free method. Only an owned
Diagonal reaching that method qualifies. Dense and lazy inputs retain their
dense full-SVD route; changed roles retain the existing transform. Checked
Generic (`scalar.rs:svd_full_checked_generic`) and CUDA (`cuda_factor.rs`)
remain independent routes. Expert matrixalgebra full SVD still publishes dense
S. No prepared/profiled Host diagonal full-SVD wrapper exists.

`diagview`, `diagonal_spectrum`, `dense_data`, `materialize`, serialization,
compact compose, norms, and diagonal mapping consume the existing compact
endomorphism invariant. Algebraic consumers accept compact S; dense buffer
callers explicitly materialize. Dense/lazy full-SVD storage assertions remain
unchanged. No duplicate full-factor builder or semantic authority is added.
The existing diagonal compact-SVD builder and its independent hand gauge
oracles are shared and extended to full SVD.

Issue impact: #1682 SOLVED subject to gates and independent review; #1615
PARTIAL, with sibling leaves remaining. No cache, profiler, provider or CUDA
resource contract changes.
