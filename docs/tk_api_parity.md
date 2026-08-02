# TensorKit 0.17 user-API parity

One row per **user-facing** TensorKit 0.17.0 export (from `TensorKit.jl`'s
`export` lists), mapped to TeNeT's provider-typed user layer
(`tenet::prelude` — `TensorMap<R,D,S>`, `GradedSpace<R>`, and `Runtime`). This
is the lookup surface: a TensorKit user finds the
function they reach for under its 0.17 name here, or the rationale for why
TeNeT spells or gates it differently.

Per-item upstream `file:line` provenance for the public rustdoc lives in
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
| **N/A** | No TeNeT analog by design (the type system already carries it, or TeNeT does not model that category-theoretic surface). |

The TeNeT user layer is **immutable**, and fallible operations return `Result`:
every in-place
TensorKit `foo!` / `foo!!` bang method maps to the out-of-place `foo` row and
is not separately listed unless its semantics differ.

The `tenet::operations` and `tenet::matrixalgebra` modules are curated
expert facades, not full implementation-crate re-exports. APIs outside those
allow-lists remain available through direct `tenet-tensors` or
`tenet-matrixalgebra` dependencies; see [the migration note](api_migration_587.md).

## Constructors

| TK 0.17 | Status | TeNeT | Notes |
|---|---|---|---|
| `zero` | has-different-name | `TensorMap::zeros` | Named for the plural leg-list constructor family. |
| `zerovector` | added | `TensorMap::zeros_like` | Same spaces + scalar/storage type, zeroed. |
| `one` | has-different-name | `TensorMap::id` | The multiplicative identity is the identity endomorphism. |
| `id` | has | `TensorMap::id` | |
| `isomorphism` | has | `TensorMap::isomorphism` | |
| `unitary` | has | `TensorMap::unitary` | |
| `isometry` | has | `TensorMap::isometry` | |
| `rand` | has | `TensorMap::rand` / `rand_with_seed` | Entries uniform in `[-1, 1)` (TK `rand` is `[0, 1)`); use an explicit seed for reproducibility. |
| `randn` | design-gated | — | Needs a Gaussian `Fill` variant in the core layer; `rand` covers the common "random tensor" need. |
| `randisometry` | design-gated | — | Composes as `TensorMap::rand(...).left_orth()?.0` at the call site; no dedicated constructor yet. |
| (block fill) | has | `TensorMap::from_block_fn` | No TK export; per-block closure fill, dtype from the closure. |
| `*!` bang forms | N/A | — | Immutable facade; no in-place constructor convention. |

## Vector interface & scalar linear algebra

| TK 0.17 | Status | TeNeT | Notes |
|---|---|---|---|
| `norm` | has | `TensorMap::norm` (+ `norm_inf`) | Quantum-dimension-weighted Frobenius. |
| `norm(t, p)` | added | `TensorMap::norm_p` | General exponent (`linalg.jl:257-275`): `p = Inf` is the max entry magnitude, finite `p > 0` is `(Σ_c dim(c)·norm(block_c, p)^p)^(1/p)`, and non-positive / NaN `p` is a typed error. A separate method because Rust has no overloading; `p = 2` / `p = Inf` delegate to `norm` / `norm_inf`. Compact diagonal storage stays `O(Σ_c k_c)`. |
| `dot` | has | `TensorMap::dot` | |
| `inner` | has | `TensorMap::inner` | |
| `normalize` | has | `TensorMap::normalize` | Zero-norm not special-cased, as in TK. |
| `tr` | has | `TensorMap::tr` (+ `trace_pairs` for partial) | `tr` is the positive ordinary trace; fermionic `trace_pairs` follows TensorKit contraction supertrace semantics. |
| `scale` | has | `TensorMap::scale` | |
| `add` | has | `TensorMap::add` | `α·self + β·other`. |
| `axpy!` / `axpby!` | has-different-name | `TensorMap::add` | Same `α`/`β` combination, out of place. |
| `mul!` | has-different-name | `TensorMap::compose` / `contract` | Categorical composition (`A * B`). |
| `lmul!` / `rmul!` | has-different-name | `TensorMap::scale` | Scalar (and diagonal, via `compose`) scaling. |
| `pinv` | has | `TensorMap::pinv` | Pseudo-inverse with `rcond` cutoff. |
| `adjoint!` | has-different-name | `TensorMap::adjoint` | Lazy, out of place. |
| `*!` / `*!!` bang forms | N/A | — | Immutable facade. |

## Index manipulation

| TK 0.17 | Status | TeNeT | Notes |
|---|---|---|---|
| `permute` | has | `TensorMap::permute` | |
| `braid` | has | `TensorMap::braid` | Explicit per-strand levels. |
| `transpose` | has | `TensorMap::transpose` | Planar transpose. |
| `twist` | has | `TensorMap::twist` | |
| `flip` (tensor) | has | `TensorMap::flip` | |
| `repartition` | added | `TensorMap::repartition` | Single split-point arg (domain rank fixed by `rank`). |
| `insertleftunit` / `insertrightunit` / `removeunit` | added | `TensorMap::insert_left_unit` / `TensorMap::insert_right_unit` / `TensorMap::remove_unit` | Insertion takes the zero-based external slot plus a `dual` flag; removal requires that axis to carry exactly the vacuum sector with degeneracy one. |
| `catdomain` / `catcodomain` | has | `TensorMap::catdomain` / `TensorMap::catcodomain` | TensorKit names as Rust binary methods; concatenate the sole domain/codomain leg and place reduced blocks in adjacent column/row slabs. |
| `absorb` | has | `TensorMap::absorb` | Immutable Rust form of TensorKit `absorb`: copies the shared prefix of every matching fusion-tree subblock from a source into a destination-shaped tensor; distinct from composition and diagonal absorption. Host only; device and mutable destination variants remain unsupported. |
| `*!` bang forms | N/A | — | Immutable facade. |

## Factorizations & matrix functions

| TK 0.17 | Status | TeNeT | Notes |
|---|---|---|---|
| `svd_compact` | has | `TensorMap::svd_compact` | |
| `svd_full` | has | `TensorMap::svd_full` | |
| `svd_trunc` | has | `TensorMap::svd_trunc` → `SvdTrunc` | Truncation via `Truncation` (below). |
| `svd_vals` | has | `TensorMap::svd_vals` | |
| `left_orth` / `right_orth` | has | `TensorMap::left_orth` / `right_orth` | |
| `left_null` / `right_null` | has | `TensorMap::left_null` / `right_null` | |
| `qr_null` / `lq_null` | has-different-name | `TensorMap::left_null` / `right_null` | Same null-space factor. |
| `left_polar` / `right_polar` | has | `TensorMap::left_polar` / `right_polar` | |
| `qr_full` / `qr_compact` | has | `TensorMap::qr_full` / `qr_compact` | |
| `lq_full` / `lq_compact` | has | `TensorMap::lq_full` / `lq_compact` | |
| `eigh_full` / `eigh_trunc` / `eigh_vals` | has | `TensorMap::eigh_full` / `eigh_trunc` / `eigh_vals` | |
| `eig_full` / `eig_trunc` / `eig_vals` | has | `TensorMap::eig_full` / `eig_trunc` / `eig_vals` | Outputs always c64. |
| `eigen` | has-different-name | `TensorMap::eig_full` | |
| `exp` | has | `TensorMap::exp` | Any endomorphism, as in TK (`exp!`, `linalg.jl:420-428`, which checks only `domain == codomain`). Dense input dispatches on the blocks: Hermitian blocks keep the spectral route `v exp(d) v^H`, everything else takes blockwise scaling-and-squaring Padé [13/13] (Higham 2005) at `O(Σ_c n_c³)` with an `O(max_c n_c²)` workspace reused across sectors. Values agree with TK to approximant error, not bitwise: TK selects lower Padé degrees for small norms. Compact diagonal storage applies `exp` elementwise and stays compact. |
| (matrix `sqrt` / `inv`) | has | `TensorMap::sqrt` / `TensorMap::inv` | LinearAlgebra surface; not a distinct TK export. |
| `ishermitian` | added | `TensorMap::is_hermitian` | Non-endomorphism → `false`, not an error. |
| `isantihermitian` | added | `TensorMap::is_antihermitian` | |
| `isisometric` | added | `TensorMap::is_isometric` | |
| `isunitary` | added | `TensorMap::is_unitary` | |
| `isposdef` | added | `TensorMap::is_posdef` | Hermitian + all eigenvalues `> -tol`. |
| `project_hermitian` | added | `TensorMap::project_hermitian` | `(t + t†)/2`. |
| `project_antihermitian` | added | `TensorMap::project_antihermitian` | `(t − t†)/2`. |
| `project_isometric` | has-different-name | `TensorMap::left_polar` (`.0`) | The polar isometric factor is the nearest isometry. |
| `rank` (numerical) | design-gated | — | Composes from `svd_vals` + a threshold at the call site. |
| `cond` | design-gated | — | Composes from `svd_vals` (max/min ratio) at the call site. |
| `sylvester` | design-gated | — | Sylvester-equation solver; no linear-solver surface on the facade. |
| `\` / `/` | design-gated | — | Per-block linear solves (`linalg.jl:397-417`), the honest primitive behind environment fitting and seam solves. `pinv` + `compose` is today's worse-conditioned workaround; named methods land with #594. |
| `^` (integer power) | has | `TensorMap::powi` | Power-by-squaring over `compose` / `inv` (`linalg.jl:45-47`). |
| `DiagonalTensorMap` | has-different-name | compact `TensorMap` storage | No public diagonal type is added. `TensorMap::diagonal` publishes compact storage directly, and `diagonal_spectrum` reads it back without materializing. |
| `diag` / `diagm` / `isdiag` | has-different-name | `diagonal` / `diagonal_spectrum` / `is_diagonal` | `diagonal` takes one labelled value vector per bond sector. `is_diagonal(0.0)` matches TensorKit exact finite-data `isdiag`; positive tolerances use `max_offdiag <= tol * max(norm_inf, 1)`. |
| `otimes` (`⊗`, tensor) | has | `TensorMap::otimes` | Tensor product in one category (`linalg.jl:556`), with exact external order `cod(A), cod(B); dom(A), dom(B)`. |
| `deligneproduct` (`⊠`, tensor) | has | `TensorMap::deligne_product` | Deligne product (`linalg.jl:597`) over an explicit generic `ProductFusionRule`; component identities, factor order and nested association are preserved. |

## Spaces & sectors

| TK 0.17 | Status | TeNeT | Notes |
|---|---|---|---|
| `dual` | has-different-name | `GradedSpace::try_dual` | Provider failures are explicit. |
| `isdual` | has | `GradedSpace::is_dual` | |
| `dim` (space) | has | `GradedSpace::dim` | Quantum-dimension-weighted total. |
| `dim(V, c)` | has-different-name | `GradedSpace::degeneracy` | Per-sector degeneracy (`dim` is the weighted total). |
| `reduceddim` | has-different-name | `GradedSpace::degeneracy` | Reduced (per-sector) dimension. |
| `dims` (tensor legs) | has-different-name | `TensorMap::leg_dims` / `leg_dim` | |
| `fuse` | has | `GradedSpace::fuse` | Collapses two factors into one graded leg, as TK's `fuse` does. |
| `otimes` (`⊗`, space) | design-gated | — | TK's space-level `⊗` builds a **factor-preserving** `ProductSpace`, which `GradedSpace::fuse` is not. TeNeT has no public product-space type (see #595); `ProductSector` is instead a sector label in a Deligne product category. |
| `oplus` (`⊕`) | added | `GradedSpace::oplus` | Per-sector degeneracy sum; rule + duality guarded. |
| `ominus` (`⊖`) | design-gated | — | Space subtraction; niche, needs a negativity guard. |
| `flip` (space) | has-different-name | `GradedSpace::try_dual` | For an elementary space, `flip` and `dual` give isomorphic spaces. |
| `sectors` | has | `GradedSpace::sectors` | Returns the provider's concrete sector type. |
| `hassector` | added | `GradedSpace::has_sector` | Boolean membership for provider-labelled sectors. |
| `sectortype` / `spacetype` | N/A | Rust types `R::Sector` / `R` | Carried statically rather than queried at runtime. |
| `field` | N/A | scalar type parameter `D` | Carried statically by `TensorMap<R,D,S>`. |
| `unitspace` | has | `GradedSpace::unitspace` | Trivial-unit space for the receiver's rule; what `insert_left_unit` / `insert_right_unit` insert. |
| `isunitspace` | design-gated | — | No standalone predicate; `unitspace` constructs the canonical vacuum leg and tensor unit operations validate it. |
| `insertleftunit` / `insertrightunit` / `removeunit` (space) | N/A | — | TK's helpers insert/drop a slot in `ProductSpace` / `HomSpace`. `GradedSpace` is one elementary leg; tensor-level analogs are `TensorMap::insert_left_unit` / `insert_right_unit` / `remove_unit`. |
| `zerospace` | design-gated | — | No zero-space constructor on the facade. |
| `infimum` / `supremum` / `isisomorphic` / `ismonomorphic` / `isepimorphic` | design-gated | — | Space-lattice predicates; no facade surface. |
| `unit` / `allunits` / `timereversed` | N/A | — | Category-theoretic sector surface TeNeT does not model at the user layer. |

### Sector types

Sector and provider types are public parts of `GradedSpace<R>` and
`TensorMap<R,D,S>`. Readback uses the provider's `SectorCodec::Sector` directly.

| TK 0.17 | Status | TeNeT | Notes |
|---|---|---|---|
| `Z2Irrep` / `U1Irrep` / `SU2Irrep` | has | provider sector types of `Z2FusionRule` / `U1FusionRule` / `SU2FusionRule` | Labels round-trip without an erased enum. |
| `FermionParity` | has-different-name | `FermionParityFusionRule` with `Z2Irrep` labels | Carries the fermionic braiding sign. |
| `SUNIrrep{N}` | feature-gated | `SUNFusionRule` + `GradedSpace` | With `racah-generated`, SU(N) uses dynamic Dynkin labels through checked Generic admission. |
| `ProductSector` (`⊠` of sectors) | has | `ProductFusionRuleExt::product` + `product_sector` | Any ordered product of admitted providers, recursively nested, without a central constructor. Unlike TK's flattened tuple, TeNeT preserves Rust type/label association. |
| `ZNIrrep{N}` | has | `ZNFusionRule` / `ZNIrrep` | Checked Z_N provider for N >= 1. |
| `CU1Irrep` | has | `CU1FusionRule` / `CU1Irrep` | Checked O(2) / broken-SU(2) provider. |

## Block access & conversion

| TK 0.17 | Status | TeNeT | Notes |
|---|---|---|---|
| `block` / `blocks` | design-gated | — | Per-coupled-sector reduced-block view; Host `TensorMap::data` exposes the flat block buffer, but not a sector-indexed slice view. |
| `blocksectors` | has-different-name | `block_count` + `block_fusion_trees` | Provider-labelled fusion trees are available per stored block. |
| `blockdim` / `subblock` / `subblocks` | design-gated | — | Same as `block`. |
| `scalartype` | N/A | type parameter `D` | Static rather than queried at runtime. |
| `storagetype` | N/A | type parameter `S` | Static; placement metadata remains available for diagnostics. |
| `scalar` | has | `TensorMap::scalar` → `D` | Rank-0 extraction. |
| (dense `Array`) | design-gated | — | Full dense materialization (fusion-tensor contraction); Host `data()` gives the reduced-block buffer, not a dense array. |
| `complex` (widen) | has | `TensorMap::to_c64` | |
| `real` / `imag` | has-different-name | `TensorMap<_,Complex64>::re` / `im` | Return `TensorMap<_,f64>` and preserve compact storage where possible. |
| `conj` | design-gated | — | Elementwise conjugation is distinct from the implemented categorical `adjoint`. |

## Contraction & truncation

| TK 0.17 | Status | TeNeT | Notes |
|---|---|---|---|
| `@tensor` | has-different-name | `tensor!` (`tenet-network`) | Identifier-index proc-macro; no einsum string parser. |
| `@tensoropt` / `@ncon` / `ncon` | has-different-name | `tensor!` + planner | N-body order chosen by the greedy / opt-einsum-path / cotengra planner. |
| `contract!` | has-different-name | `TensorMap::contract` / `compose` | `contract_ordered` is a documented alias; expert layer: `tensorcontract_into`. |
| `scalar` | has | `TensorMap::scalar` | |
| `@planar` / `@plansor` | design-gated | — | Planar-only diagram contraction; not exposed. |
| `notrunc` | has-different-name | `Truncation::Full` | |
| `truncrank` | has-different-name | `Truncation::rank` | |
| `trunctol` | has-different-name | `Truncation::absolute_cutoff` / `relative_cutoff` / `relative_inf_cutoff` | Checked constructors; `p=Inf` → `ToleranceInf`. |
| `truncerror` | has-different-name | `Truncation::relative_error` | Checked constructor bounding the discarded 2-norm tail. |
| (compose truncations) | has-different-name | `Truncation::and` | |
| `truncspace` | added | `GradedSpace::truncspace` → `Truncation::space` | Fixed per-sector prefix counts read off a target space (`truncation.jl:261-269`). Absent sector = rank zero, over-long requests clamp to the available prefix, and a profile from another fusion rule is a typed error. |
| `truncfilter` | design-gated | — | Arbitrary non-prefix index filters; the prefix-only decision layer does not model them (see the `truncation.rs` header). |

## Notes on deliberate omissions

- **Bang (`!`) methods.** TeNeT's user layer is immutable and `Result`-typed;
  the curated expert facade retains `tensorcontract_into`, `tensoradd_into`,
  `tensortrace_into`, `permute_into`, `braid_into`, and `transpose_into`.
  Broader unstable `_into` families require a direct `tenet-tensors` dependency.
- **Type introspection.** `R`, `R::Sector`, `D`, and `S` carry provider, sector,
  scalar and storage types statically; runtime `sectortype` / `scalartype` /
  `storagetype` methods would duplicate Rust's type system.
- **Elementwise `conj`.** `re` and `im` are available for complex tensors.
  Elementwise conjugation remains distinct from categorical `adjoint` and is
  not exposed as an alias.
