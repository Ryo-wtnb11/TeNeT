# TensorKit 0.17 user-API parity

One row per **user-facing** TensorKit 0.17.0 export (from `TensorKit.jl`'s
`export` lists), mapped to the TeNeT user layer (`tenet::prelude` —
`Tensor`, `Space`, `Runtime`, and the factorization return types). This is the
lookup surface: a TensorKit user finds the function they reach for under its
0.17 name here, or the rationale for why TeNeT spells or gates it differently.

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

## Summary counts

Counts are table rows; a few rows bundle several closely-related exports
(e.g. `eigh_full` / `eigh_trunc` / `eigh_vals`).

| Status | Rows |
|---|---|
| has | 40 |
| has-different-name | 22 |
| added (this sweep) | 11 |
| design-gated | 18 |
| N/A | 7 |

Added this sweep: `Tensor::numout` / `numin` / `numind`, `Tensor::repartition`,
`Tensor::zeros_like`, `Tensor::is_hermitian` / `is_antihermitian` /
`is_isometric` / `is_unitary` / `is_posdef`, `Tensor::project_hermitian` /
`project_antihermitian`, `Space::has_sector`, `Space::oplus`.

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
| `insertleftunit` / `insertrightunit` / `removeunit` | design-gated | — | Needs a trivial-unit `Space` constructor on the facade (see `unitspace` below). |
| `catdomain` / `catcodomain` | has | `Tensor::catdomain` / `Tensor::catcodomain` | TensorKit names as Rust binary methods; concatenate the sole domain/codomain leg and place reduced blocks in adjacent column/row slabs. |
| `absorb` | design-gated | — | Copies the shared prefix of every matching fusion-tree subblock from a source into a destination-shaped tensor; distinct from composition and diagonal absorption. |
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
| `exp` | has | `Tensor::exp` | |
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

## Spaces & sectors

| TK 0.17 | Status | TeNeT | Notes |
|---|---|---|---|
| `dual` | has | `Space::dual` | |
| `isdual` | has | `Space::is_dual` | |
| `dim` (space) | has | `Space::dim` | Quantum-dimension-weighted total. |
| `dim(V, c)` | has-different-name | `Space::degeneracy` | Per-sector degeneracy (`dim` is the weighted total). |
| `reduceddim` | has-different-name | `Space::degeneracy` | Reduced (per-sector) dimension. |
| `dims` (tensor legs) | has-different-name | `Tensor::leg_dims` / `leg_dim` | |
| `fuse` / `otimes` (`⊗`) | has | `Space::fuse` / `fuse_all` | |
| `oplus` (`⊕`) | added | `Space::oplus` | Per-sector degeneracy sum; rule + duality guarded. |
| `ominus` (`⊖`) | design-gated | — | Space subtraction; niche, needs a negativity guard. |
| `flip` (space) | has-different-name | `Space::dual` | For an elementary space, `flip` and `dual` give isomorphic spaces; the twist-carrying distinction is internal to the fusion machinery. |
| `sectors` | has | `Space::sectors` / `try_sectors` / `su3_sectors` | |
| `hassector` | added | `Space::has_sector` | Boolean membership (SU(3) via `su3_degeneracy`). |
| `sectortype` / `spacetype` | N/A | — | The concrete sector/rule type is erased at the user layer; `SectorLabel` enumerates it instead. |
| `field` | N/A | — | Scalar field is carried by the `Dtype` token (`F64`/`C64`). |
| `unitspace` / `zerospace` / `leftunitspace` / `rightunitspace` / `isunitspace` | design-gated | — | No trivial-unit / zero `Space` constructor on the facade yet (blocks `insertunit`/`removeunit`). |
| `infimum` / `supremum` / `isisomorphic` / `ismonomorphic` / `isepimorphic` | design-gated | — | Space-lattice predicates; no facade surface. |
| `unit` / `allunits` / `deligneproduct` / `timereversed` | N/A | — | Category-theoretic sector surface TeNeT does not model at the user layer. |

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
| `truncfilter` / `truncspace` | design-gated | — | Non-prefix filters / space-target truncation; the prefix-only decision layer does not model them (see `truncation.rs` header). |

## Notes on deliberate omissions

- **Bang (`!`) methods.** TeNeT's user layer is immutable and `Result`-typed;
  the expert layer (`tenet::operations::*_into`) carries the in-place surface.
- **Sector / space *type* introspection** (`sectortype`, `spacetype`,
  `storagetype`) is intentionally erased: `Tensor` and `Space` are rule- and
  storage-generic at the user layer, dispatching internally. `SectorLabel` and
  `Dtype` are the user-visible stand-ins.
- **`real`/`imag`/`conj`** are the one linear-algebra gap left open on purpose:
  they are safe on self-dual rules but mislabel coupled sectors on non-self-dual
  ones without the adjoint-fold self-dual guard. Design-gated until that guard
  is exposed, rather than shipped with a known correctness trap.
