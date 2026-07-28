# TensorKit 0.17 user-API parity

One row per **user-facing** TensorKit 0.17.0 export (from `TensorKit.jl`'s
`export` lists), mapped primarily to the erased facade of the TeNeT user
layer (`tenet::prelude` — `Tensor`, `Space`, `Runtime`, and the factorization
return types). This is the lookup surface: a TensorKit user finds the
function they reach for under its 0.17 name here, or the rationale for why
TeNeT spells or gates it differently.

Per-item upstream `file:line` provenance for the public rustdoc of both
facades (erased and `tenet::typed`) lives in
[`tenet/references.md`](../tenet/references.md), against pinned upstream
revisions.

Reference source: `TensorKit v0.17.0` at
`~/.julia/packages/TensorKit/jCjQQ/src`. Names are the TK 0.17 canon
(`svd_trunc`/`left_orth` family — never `tsvd`/`leftorth`).

## Status legend

| Status | Meaning |
|---|---|
| **has** | Present under the same (or effectively same) name. |
| **has-different-name** | Present; TeNeT spells it differently. The mapping column *is* the alias — no thin wrapper is added when the different name is clearer and the concept is already discoverable. |
| **added** | Added in this parity sweep under the TK name (or a Rust-idiomatic `snake_case` of it). |
| **design-gated** | Not present; needs kernel/storage/solver work beyond a facade wrapper, or would reintroduce a known hazard. Rationale given. |
| **N/A** | No TeNeT analog by design (concept erased at the user layer, or category-theoretic surface TeNeT does not model). |

The TeNeT user layer is **immutable / `Result`-returning**: every in-place
TensorKit `foo!` / `foo!!` bang method maps to the out-of-place `foo` row and
is not separately listed unless its semantics differ.

The `tenet::operations` and `tenet::matrixalgebra` modules are curated
expert facades, not full implementation-crate re-exports. APIs outside those
allow-lists remain available through direct `tenet-tensors` or
`tenet-matrixalgebra` dependencies; see [the migration note](api_migration_587.md).

## Summary counts

Counts are table rows; a few rows bundle several closely-related exports
(e.g. `eigh_full` / `eigh_trunc` / `eigh_vals`).

| Status | Rows |
|---|---|
| has | 44 |
| has-different-name | 26 |
| added (this sweep) | 14 |
| design-gated | 24 |
| N/A | 8 |

Added this sweep: `Tensor::repartition`, `Tensor::zeros_like`,
`Tensor::insert_left_unit` / `insert_right_unit` / `remove_unit`,
`Tensor::is_hermitian` / `is_antihermitian` / `is_isometric` / `is_unitary` /
`is_posdef`, `Tensor::project_hermitian` / `project_antihermitian`,
`Space::has_sector`, `Space::oplus`, `Tensor::norm_p` / `TensorMap::norm_p`,
`Space::truncspace` / `GradedSpace::truncspace`.

---

## Constructors

| TK 0.17 | Status | TeNeT | Notes |
|---|---|---|---|
| `zero` | has-different-name | `Tensor::zeros` | Named for the plural leg-list constructor family. |
| `zerovector` | added | `Tensor::zeros_like` | Same spaces + dtype, zeroed. |
| `one` | has-different-name | `Tensor::id` | The multiplicative identity is the identity endomorphism. |
| `id` | has | `Tensor::id` | |
| `isomorphism` | has | `Tensor::isomorphism` | |
| `unitary` | has | `Tensor::unitary` | |
| `isometry` | has | `Tensor::isometry` | |
| `rand` | has | `Tensor::rand` / `rand_with_seed` | Entries uniform in `[-1, 1)` (TK `rand` is `[0, 1)`); use an explicit seed for reproducibility. |
| `randn` | design-gated | — | Needs a Gaussian `Fill` variant in the core layer; `rand` covers the common "random tensor" need. |
| `randisometry` | design-gated | — | Composes as `Tensor::rand(...).left_orth()?.0` at the call site; no dedicated constructor yet. |
| (block fill) | has | `Tensor::from_block_fn` | No TK export; per-block closure fill, dtype from the closure. |
| `*!` bang forms | N/A | — | Immutable facade; no in-place constructor convention. |

## Vector interface & scalar linear algebra

| TK 0.17 | Status | TeNeT | Notes |
|---|---|---|---|
| `norm` | has | `Tensor::norm` (+ `norm_inf`) | Quantum-dimension-weighted Frobenius. |
| `norm(t, p)` | added | `Tensor::norm_p` / `TensorMap::norm_p` | General exponent (`linalg.jl:257-275`): `p = Inf` is the max entry magnitude, finite `p > 0` is `(Σ_c dim(c)·norm(block_c, p)^p)^(1/p)`, and non-positive / NaN `p` is a typed error. A separate method because Rust has no overloading; `p = 2` / `p = Inf` delegate to `norm` / `norm_inf`. Compact diagonal storage stays `O(Σ_c k_c)`. |
| `dot` | has | `Tensor::dot` | |
| `inner` | has | `Tensor::inner` | |
| `normalize` | has | `Tensor::normalize` | Zero-norm not special-cased, as in TK. |
| `tr` | has | `Tensor::tr` (+ `trace_pairs` for partial) | `tr` is the positive ordinary trace; fermionic `trace_pairs` follows TensorKit contraction supertrace semantics. |
| `scale` | has | `Tensor::scale` / `scale_c64` | |
| `add` | has | `Tensor::add` / `add_c64` | `α·self + β·other`. |
| `axpy!` / `axpby!` | has-different-name | `Tensor::add` | Same `α`/`β` combination, out of place. |
| `mul!` | has-different-name | `Tensor::compose` / `contract` | Categorical composition (`A * B`). |
| `lmul!` / `rmul!` | has-different-name | `Tensor::scale` | Scalar (and diagonal, via `compose`) scaling. |
| `pinv` | has | `Tensor::pinv` | Pseudo-inverse with `rcond` cutoff. |
| `adjoint!` | has-different-name | `Tensor::adjoint` | Lazy, out of place. |
| `*!` / `*!!` bang forms | N/A | — | Immutable facade. |

## Index manipulation

| TK 0.17 | Status | TeNeT | Notes |
|---|---|---|---|
| `permute` | has | `Tensor::permute` | |
| `braid` | has | `Tensor::braid` | Explicit per-strand levels. |
| `transpose` | has | `Tensor::transpose` | Planar transpose. |
| `twist` | has | `Tensor::twist` | |
| `flip` (tensor) | has | `Tensor::flip` | |
| `repartition` | added | `Tensor::repartition` | Single split-point arg (domain rank fixed by `rank`). |
| `insertleftunit` / `insertrightunit` / `removeunit` | added | `Tensor::insert_left_unit` / `Tensor::insert_right_unit` / `Tensor::remove_unit` | Insertion takes the zero-based external slot plus a `dual` flag; removal requires that axis to carry exactly the vacuum sector with degeneracy one. |
| `catdomain` / `catcodomain` | has | `Tensor::catdomain` / `Tensor::catcodomain` | TensorKit names as Rust binary methods; concatenate the sole domain/codomain leg and place reduced blocks in adjacent column/row slabs. |
| `absorb` | has | `Tensor::absorb` | Immutable Rust form of TensorKit `absorb`: copies the shared prefix of every matching fusion-tree subblock from a source into a destination-shaped tensor; distinct from composition and diagonal absorption. Host only; device and mutable destination variants remain unsupported. |
| `*!` bang forms | N/A | — | Immutable facade. |

## Factorizations & matrix functions

| TK 0.17 | Status | TeNeT | Notes |
|---|---|---|---|
| `svd_compact` | has | `Tensor::svd_compact` | |
| `svd_full` | has | `Tensor::svd_full` | |
| `svd_trunc` | has | `Tensor::svd_trunc` → `SvdTrunc` | Truncation via `Truncation` (below). |
| `svd_vals` | has | `Tensor::svd_vals` | |
| `left_orth` / `right_orth` | has | `Tensor::left_orth` / `right_orth` | |
| `left_null` / `right_null` | has | `Tensor::left_null` / `right_null` | |
| `qr_null` / `lq_null` | has-different-name | `Tensor::left_null` / `right_null` | Same null-space factor. |
| `left_polar` / `right_polar` | has | `Tensor::left_polar` / `right_polar` | |
| `qr_full` / `qr_compact` | has | `Tensor::qr_full` / `qr_compact` | |
| `lq_full` / `lq_compact` | has | `Tensor::lq_full` / `lq_compact` | |
| `eigh_full` / `eigh_trunc` / `eigh_vals` | has | `Tensor::eigh_full` / `eigh_trunc` / `eigh_vals` | |
| `eig_full` / `eig_trunc` / `eig_vals` | has | `Tensor::eig_full` / `eig_trunc` / `eig_vals` | Outputs always c64. |
| `eigen` | has-different-name | `Tensor::eig_full` | |
| `exp` | has | `Tensor::exp` / `TensorMap::exp` | Any endomorphism, as in TK (`exp!`, `linalg.jl:420-428`, which checks only `domain == codomain`). Dense input dispatches on the blocks: Hermitian blocks keep the exact spectral route `v exp(d) v^H`, everything else takes blockwise scaling-and-squaring Padé [13/13] (Higham 2005) — the algorithm behind the `LinearAlgebra.exp!` TK calls — at `O(Σ_c n_c³)` with an `O(max_c n_c²)` workspace reused across sectors — the whole of the scratch on the canonical layout, while the non-contiguous fallback matricizes every sector first and adds `O(Σ_c n_c²)` — and, as in TK, `LAPACK.gebal!('B', ·)` balancing around the approximant (#577). Values therefore agree with TK to approximant error, not bitwise: TK drops to Padé degree 3/5/7/9 below `‖A‖₁ = 2.1` where TeNeT always uses [13/13]. Compact diagonal storage takes TK's `exp(::DiagonalTensorMap)` (`diagonal.jl:383-390`) instead: elementwise on the stored values, `O(Σ_c k_c)`, staying compact and — as upstream — with no hermiticity gate (#576 typed, #578 erased). |
| (matrix `sqrt` / `inv`) | has | `Tensor::sqrt` / `Tensor::inv` | LinearAlgebra surface; not a distinct TK export. |
| `ishermitian` | added | `Tensor::is_hermitian` | Non-endomorphism → `false`, not an error. |
| `isantihermitian` | added | `Tensor::is_antihermitian` | |
| `isisometric` | added | `Tensor::is_isometric` | |
| `isunitary` | added | `Tensor::is_unitary` | |
| `isposdef` | added | `Tensor::is_posdef` | Hermitian + all eigenvalues `> -tol`. |
| `project_hermitian` | added | `Tensor::project_hermitian` | `(t + t†)/2`. |
| `project_antihermitian` | added | `Tensor::project_antihermitian` | `(t − t†)/2`. |
| `project_isometric` | has-different-name | `Tensor::left_polar` (`.0`) | The polar isometric factor is the nearest isometry. |
| `rank` (numerical) | design-gated | — | Composes from `svd_vals` + a threshold at the call site. |
| `cond` | design-gated | — | Composes from `svd_vals` (max/min ratio) at the call site. |
| `sylvester` | design-gated | — | Sylvester-equation solver; no linear-solver surface on the facade. |
| `\` / `/` | design-gated | — | Per-block linear solves (`linalg.jl:397-417`), the honest primitive behind environment fitting and seam solves. `pinv` + `compose` is today's worse-conditioned workaround; named methods land with #594. |
| `^` (integer power) | design-gated | — | Power-by-squaring over `compose` / `inv` (`linalg.jl:44-47`); rides along with #594. |
| `DiagonalTensorMap` | design-gated | — | Compact diagonal storage exists internally and every reduction on it stays `O(Σ_c k_c)`, but it is only ever *produced* (by `svd_*` / `eigh_*`) — there is no user-facing type or constructor. #593 adds the construction surface. |
| `diag` / `diagm` / `isdiag` | design-gated | — | Same gate as `DiagonalTensorMap` (`linalg.jl:179-190`): build-from-spectrum, an `is_diagonal` predicate and a spectrum accessor land with #593. |
| `otimes` (`⊗`, tensor) | has-multiplicity-free | `Tensor::otimes` / `typed::TensorMap::otimes` | Tensor product in one category (`linalg.jl:556`), with exact external order `cod(A), cod(B); dom(A), dom(B)`. Codomain trees and domain trees are merged independently with checked F moves; no legs cross, no R symbol is evaluated, and no dense Kronecker temporary is built. Erased SU(3) is rejected: its outer multiplicities require a separate generic tree-merge kernel. |
| `deligneproduct` (`⊠`, tensor) | has-typed-only | `typed::TensorMap::deligne_product` | Deligne product (`linalg.jl:597`) over an explicit generic `ProductFusionRule`. Both component identities are checked; factor order and TeNeT's nested association are preserved. The erased facade has no generic product-provider admission and exposes no stub or pairwise enum matrix. |

## Spaces & sectors

| TK 0.17 | Status | TeNeT | Notes |
|---|---|---|---|
| `dual` | has | `Space::dual` | |
| `isdual` | has | `Space::is_dual` | |
| `dim` (space) | has | `Space::dim` | Quantum-dimension-weighted total. |
| `dim(V, c)` | has-different-name | `Space::degeneracy` | Per-sector degeneracy (`dim` is the weighted total). |
| `reduceddim` | has-different-name | `Space::degeneracy` | Reduced (per-sector) dimension. |
| `dims` (tensor legs) | has-different-name | `Tensor::leg_dims` / `leg_dim` | |
| `fuse` | has | `Space::fuse` / `fuse_all` | Collapses the factors into one graded leg, as TK's `fuse` does. |
| `otimes` (`⊗`, space) | design-gated | — | TK's space-level `⊗` builds a **factor-preserving** `ProductSpace`, which `Space::fuse` is not: fusing is lossy about the factors. TeNeT has no public product-space type, so there is nothing to return (see #595). Unrelated to TK's `ProductSector` / TeNeT's `ProductSector`, which is a *sector label* in a Deligne product category, not a list of legs. |
| `oplus` (`⊕`) | added | `Space::oplus` | Per-sector degeneracy sum; rule + duality guarded. |
| `ominus` (`⊖`) | design-gated | — | Space subtraction; niche, needs a negativity guard. |
| `flip` (space) | has-different-name | `Space::dual` | For an elementary space, `flip` and `dual` give isomorphic spaces; the twist-carrying distinction is internal to the fusion machinery. |
| `sectors` | has | `Space::sectors` / `try_sectors` / `su3_sectors` | |
| `hassector` | added | `Space::has_sector` | Boolean membership (SU(3) via `su3_degeneracy`). |
| `sectortype` / `spacetype` | N/A | — | The concrete sector/rule type is erased at the user layer; `SectorLabel` enumerates it instead. |
| `field` | N/A | — | Scalar field is carried by the `Dtype` token (`F64`/`C64`). |
| `unitspace` | has | `Space::unitspace` | Trivial-unit space for the receiver's rule; what `insert_left_unit` / `insert_right_unit` insert. |
| `isunitspace` | has | `Space::isunitspace` | `tenet/src/space.rs::Space::isunitspace`. Sector content only (exactly the vacuum with degeneracy one); the dual flag is ignored, as in TK. |
| `insertleftunit` / `insertrightunit` / `removeunit` (space) | N/A | — | TK's unit helpers operate on `ProductSpace` / `HomSpace` / tensors (`spaces/vectorspaces.jl:298-307`) — they insert or drop a *slot*. TeNeT's `Space` is one elementary graded leg with no slot list, so same-named `Space` methods would model the wrong object. The tensor-level analogs are `tenet/src/tensor.rs::Tensor::insert_left_unit` / `insert_right_unit` / `remove_unit`. Revisit only if a public factor-preserving product-space type is ever introduced. |
| `zerospace` | design-gated | — | No zero-space constructor on the facade. |
| `infimum` / `supremum` / `isisomorphic` / `ismonomorphic` / `isepimorphic` | design-gated | — | Space-lattice predicates; no facade surface. |
| `unit` / `allunits` / `timereversed` | N/A | — | Category-theoretic sector surface TeNeT does not model at the user layer. |

### Sector types

Sector types are not a public type of the *erased* user layer: a rule is chosen
by which `Space` constructor is called, and read back as a `SectorLabel`. The
provider-typed facade is the exception the rest of this section is measured
against — `tenet::typed::GradedSpace<R>` keeps the provider type and reports
`SectorCodec::Sector` labels, so a sector type is public there.

| TK 0.17 | Status | TeNeT | Notes |
|---|---|---|---|
| `Z2Irrep` / `U1Irrep` / `SU2Irrep` | has-different-name | `Space::z2` / `Space::u1` / `Space::su2` | Labels round-trip through `SectorLabel::Z2` / `U1` / `SU2`. |
| `FermionParity` | has-different-name | `Space::fz2` | Fermionic Z2; carries the braiding sign. |
| `SUNIrrep{3}` | has-different-name | `Space::su3` | Outer-multiplicity rule; `(p, q)` labels read back through `Space::su3_sectors` (they do not fit `SectorLabel`). |
| `ProductSector` (`⊠` of sectors) | has-different-name | `ProductFusionRuleExt::product` + `product_sector` (typed facade) | The canonical route, and as open as TK's `⊠` in what it admits: any ordered product of admitted providers, recursively nested, no new constructor. It is *not* identical in association — TK's `⊠` flattens, so `(A ⊠ B) ⊠ C` and `A ⊠ (B ⊠ C)` are the same `ProductSector{Tuple{A,B,C}}` there, while TeNeT keeps the nesting: factor order and association are Rust-type/label structure, never an automatic equivalence. `Space::product` / `Space::fz2_u1_su2` are two fixed erased conveniences kept for compatibility, not the extension mechanism (#610). |
| `ZNIrrep{N}` (`N > 2`) | design-gated | — | Nothing exists between `Z2` and `U(1)`: Z_N clock models, parafermions and Z_N gauge sectors all need one `FusionRule` + `SectorCodec` impl plus a `Space` constructor (#591). |
| `CU1Irrep` | design-gated | — | O(2) / broken-SU(2) sectors; same gate as `ZNIrrep{N}` (#591). |

## Block access & conversion

| TK 0.17 | Status | TeNeT | Notes |
|---|---|---|---|
| `block` / `blocks` | design-gated | — | Per-coupled-sector reduced-block view; `Tensor::data` / `data_c64` expose the flat storage buffer, but not a sector-indexed view (needs a sector→range map surface). |
| `blocksectors` | design-gated | — | Coupled-sector list; derivable but no direct facade accessor yet. |
| `blockdim` / `subblock` / `subblocks` | design-gated | — | Same as `block`. |
| `scalartype` | has-different-name | `Tensor::dtype` | Returns `Dtype`. |
| `storagetype` | N/A | — | Storage (host `Vec` / device) is erased; no user type parameter. |
| `scalar` | has | `Tensor::scalar` → `Scalar` | Rank-0 extraction. |
| (dense `Array`) | design-gated | — | Full dense materialization (fusion-tensor contraction); `data()`/`data_c64()` give the block buffer, not a dense array. |
| `complex` (widen) | has | `Tensor::to_c64` | |
| `real` / `imag` / `conj` | design-gated | — | On non-self-dual symmetric tensors these hit the coupled-sector mislabel hazard fixed in the adjoint fold; safe support needs that self-dual-guard machinery, not a wrapper. |

## Contraction & truncation

| TK 0.17 | Status | TeNeT | Notes |
|---|---|---|---|
| `@tensor` | has-different-name | `tensor!` (`tenet-network`) | Identifier-index proc-macro; no einsum string parser. |
| `@tensoropt` / `@ncon` / `ncon` | has-different-name | `tensor!` + planner | N-body order chosen by the greedy / opt-einsum-path / cotengra planner. |
| `contract!` | has-different-name | `Tensor::contract` / `contract_ordered` / `compose` | (Expert layer: `tensorcontract_into`.) |
| `scalar` | has | `Tensor::scalar` | |
| `@planar` / `@plansor` | design-gated | — | Planar-only diagram contraction; not exposed. |
| `notrunc` | has-different-name | `Truncation::Full` | |
| `truncrank` | has-different-name | `Truncation::rank` | |
| `trunctol` | has-different-name | `Truncation::absolute_cutoff` / `relative_cutoff` / `relative_inf_cutoff` | Checked constructors; `p=Inf` → `ToleranceInf`. |
| `truncerror` | has-different-name | `Truncation::relative_error` | Checked constructor bounding the discarded 2-norm tail. |
| (compose truncations) | has-different-name | `Truncation::and` | |
| `truncspace` | added | `Space::truncspace` / `GradedSpace::truncspace` → `Truncation::space` | Fixed per-sector prefix counts read off a target space (`truncation.jl:261-269`). Absent sector = rank zero, over-long requests clamp to the available prefix, and a profile from another fusion rule is a typed error. |
| `truncfilter` | design-gated | — | Arbitrary non-prefix index filters; the prefix-only decision layer does not model them (see the `truncation.rs` header). |

## Notes on deliberate omissions

- **Bang (`!`) methods.** TeNeT's user layer is immutable and `Result`-typed;
  the curated expert facade retains `tensorcontract_into`, `tensoradd_into`,
  `tensortrace_into`, `permute_into`, `braid_into`, and `transpose_into`.
  Broader unstable `_into` families require a direct `tenet-tensors` dependency.
- **Sector / space *type* introspection** (`sectortype`, `spacetype`,
  `storagetype`) is intentionally erased: `Tensor` and `Space` are rule- and
  storage-generic at the user layer, dispatching internally. `SectorLabel` and
  `Dtype` are the user-visible stand-ins.
- **`real`/`imag`/`conj`** are the one linear-algebra gap left open on purpose:
  they are safe on self-dual rules but mislabel coupled sectors on non-self-dual
  ones without the adjoint-fold self-dual guard. Design-gated until that guard
  is exposed, rather than shipped with a known correctness trap.
