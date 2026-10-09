# One compact factor plan for Host and CUDA (#1774)

Authority: TeNeT base `a596439b794c0e2988e8c19bc266f58dcc7fa595`; references
TensorKit.jl `cfaa073e`, QSpace v4 `dd2cc7e1`.

## Invariant

`CompactFactorPlan` (`tenet-matrixalgebra/src/factorize/compact_plan.rs`) is
the one authority for the compact bond, the factor spaces, the sector routes,
the tree layouts and the spectrum factor space of compact QR, compact SVD and
full EIGH, on Host and CUDA alike. The CUDA facade keeps only device
execution and its resources.

## Reference map

| Step | TensorKit `cfaa073e` | QSpace `dd2cc7e1` | TeNeT |
| --- | --- | --- | --- |
| Compact bond | `src/factorizations/matrixalgebrakit.jl:MAK.initialize_output(::typeof(qr_compact!))` and `(::typeof(svd_compact!))`: `infimum(fuse(codomain(t)), fuse(domain(t)))`; `src/spaces/gradedspace.jl:infimum` keeps sectors on both sides with `min` degeneracy | `Source/mpsortho.cc:SVD_Data::blockSVD`: one dense SVD per coupled-sector group, whose kept rank sizes the bond | `compact_bond_leg`: one sector per coupled-sector region (sectors on both sides) with `min(rows, cols)`; zero degeneracies dropped by `SectorLeg` |
| EIGH bond | `initialize_output(::typeof(eigh_full!))`: `V_D = fuse(domain(t))` | `Source/QSpace.cc:QSpace::EigenSymmetric`: one symmetric eigendecomposition per group | the same compact bond: an endomorphism's regions are square, so `min(n, n) = n` |
| Factor spaces | `similar(t, codomain(t) ← V)`, `similar(t, V ← domain(t))` | factor blocks assembled per group | `CompactFactorPlan::left_space` / `right_space` (the layouts built by `build_bound_factor_space`) |
| Spectrum factor | `similar_diagonal(t, real(T), V)` | singular values or eigenvalues per group | `CompactFactorPlan::bond_space`, the `spectrum_bond` rule (`leg_bond_space`) |
| Per-sector execution | `MAK.$f!(t, F, alg)` → `src/tensors/blockiterator.jl:foreachblock`: one dense call per coupled sector | `getQsum(..).groupRecs` then `toBlockMatrix` + `wbSVD` / `wbEigenS` per group | `CompactFactorPlan::routes`: one route per source region; zero-rank routes have no factor region |

QSpace has no symmetric-tensor QR (only dense `wbarray` QR), so it gives no
QR counterpart.

## What was ported

The output-initialization and per-block execution split: the plan is TensorKit's
`initialize_output` (one bond and factor-space rule, independent of storage)
plus the `foreachblock` sector pairing, compiled into routes. As in TensorKit,
a CUDA tensor and a Host tensor of the same space get the same factor spaces
through the same code.

## Rust, Tenferro and TeNeT differences

- The plan stores provider-free validated layouts and rebinds them to the
  caller's provider (`rebind_validated`); coupled-sector regions come from
  the per-structure cache, so building the plan after the device admission
  compiles nothing twice.
- Each route reports whether its factor region lists the source's trees with
  the source's offsets and shapes (`left_preserves_trees`,
  `right_preserves_trees`). The Host direct path publishes positionally only
  when every route does, else it scatters by tree identity; the device copies
  the whole factor on such a route and runs one GEMM per tree otherwise.
  TensorKit has a single block layout per sector, so it needs no such
  predicate.
- Device execution (cuSOLVER through Tenferro, gauge, selectors, copies)
  stays in the facade; moving it into this crate is a separate follow-up.

## Equivalence with the base CUDA planning

- Bond: base CUDA built the same leg from regions (QR, SVD) or from spectrum
  lengths `n` (EIGH); both equal `compact_bond_leg`.
- Factor spaces: the same `derive_from_final_homspace` calls on equal
  homspaces, in the same order (left, right), so provider errors keep their
  order.
- `W <- W`: the `spectrum_bond` rule reduces, in multiplicity-free mode, to
  the `derive_from_final_homspace(W <- W)` the base used.
- Tree alignment: `coupled_sector_regions` places each tree at the running
  extent in list order, so offsets are cumulative from 0 and sum to the
  region extent. The base CUDA conditions (offsets tile `[0, extent)`) hold
  for every region, and equal trees on the same legs have equal shapes; the
  base predicate and `*_preserves_trees` accept the same routes.
- SVD diagonal: the base order-insensitive tree match and
  `has_aligned_diagonal()` accept the same inputs (a single nondual bond leg
  has one tree per coupled sector on each side); the latter is the exact
  precondition of the stride-`rank + 1` diagonal copy.
- Zero-rank sectors cannot arise from public constructors (zero degeneracies
  are dropped); the plan's zero-rank routes are skipped on the device as on
  the Host.

## Evidence

- Cross-storage gate `device_factor_plan_publishes_the_host_spaces_at_every_payload`
  (`tenet/tests/typed_cuda_single_precision_factorizations.rs`): QR, SVD and
  EIGH at `f64`, `Complex64`, `f32` and `Complex32` on U(1) multi-tree sectors
  with a dual leg and one-sided sectors, and on SU(2); EIGH on a
  rank-deficient Gram operator and its unit shift. Device factor structure
  equals the Host's; values compare through gauge-independent identities.
- Structural allocations of the pre-lease planning fall in calls for every
  operation (QR constant in the sector count); SVD keeps a disclosed constant
  of up to 55 bytes for the plan's validated layouts. The numbers are in the
  #1774 implementation record.
