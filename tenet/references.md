# Provenance and references

Upstream source coordinates for the semantic claims made by `tenet`'s public
rustdoc. The rustdoc keeps the semantic anchor (the TensorKit /
MatrixAlgebraKit API name and the behavioral claim); the exact `file:line`
provenance lives here, against the pinned revisions below, so the docs never
carry coordinates that can silently rot.

Scope: public items of the erased facade (`tenet::prelude::Tensor`,
`src/tensor.rs`) and the typed facade (`tenet::typed`, `src/typed.rs`) whose
rustdoc carries an upstream semantic claim. Private and `pub(crate)` rustdoc
keeps its inline coordinates (implementation-facing, reviewed with the code).

## Pinned upstream revisions

All `source` coordinates below are line numbers in these exact trees
(immutable Julia package installs; the path slug is derived from the
`git-tree-sha1`, so the on-disk tree cannot drift from the pin):

| Upstream | Version | git-tree-sha1 | Local tree |
| --- | --- | --- | --- |
| TensorKit.jl | 0.17.0 | `025081058ed953b53aeea9bb8bdbdf241a1fa54f` | `~/.julia/packages/TensorKit/jCjQQ` |
| TensorKitSectors.jl | 0.3.4 | `bb54ef851826493b1ec37fd6a9a9ec573ccd5373` | `~/.julia/packages/TensorKitSectors/JS0fj` |
| MatrixAlgebraKit.jl | 0.6.9 | `0f958855fc2efdc59f1b6a5a066ee3026e87edb3` | `~/.julia/packages/MatrixAlgebraKit/AbU5a` |

Version/tree-sha1 pairs resolved from the local package installs'
`Project.toml` plus the General registry's `Versions.toml` (slug recomputed
with `Base.version_slug` to confirm the correspondence).

MatrixAlgebraKit is pinned for the factorization *names* the rustdoc anchors
to (`svd_compact`, `left_orth`, `DiagonalAlgorithm`, ...); every `source`
path below that mentions MatrixAlgebraKit behavior
(`factorizations/matrixalgebrakit.jl`, `factorizations/diagonal.jl`) is a
TensorKit file wrapping it. `tensors/tensoroperations.jl` and
`tensors/vectorinterface.jl` are likewise TensorKit's implementations of the
TensorOperations / VectorInterface interfaces.

## Coordinates

`source` paths are relative to the upstream package's `src/`. Rows flagged
**divergence** in the note provenance a documented deliberate difference —
the full semantic statement stays in the rustdoc; only the coordinate lives
here.

A row noted "both facades" covers the identical claim on the erased
(`Tensor::x`) and typed (`typed::TensorMap::x` / `typed::GradedSpace::x`)
item of the same name; separate per-facade rows exist where the claims or
coordinates differ. A struct's row also covers its documented public fields.
A source with no line range is a deliberate name-only row: the claim has no
single defensible upstream location (the note says why).

| Item | Upstream component | Revision | Source | Note |
| --- | --- | --- | --- | --- |
| `Scalar` | TensorKit | 0.17.0 | `tensors/linalg.jl:319-327` | erased return-scalar enum: TK `tr` (319-327) and `dot` (255) return the block scalartype — real tensors give real scalars |
| `SvdTrunc` | MatrixAlgebraKit | 0.6.9 | `interface/svd.jl:44-93` | `svd_trunc` docstring: returns `(U, S, Vᴴ, ϵ)`; both facades (identical claim on the erased and typed item) |
| `Tensor::add` | TensorKit | 0.17.0 | `tensors/vectorinterface.jl:67-99` | divergence: TK computes `β*ty + α*tx` (first coefficient on the *second* argument); tenet's `alpha` belongs to `self`; both facades (identical claim on the erased and typed item) |
| `Tensor::adjoint` | TensorKit | 0.17.0 | `tensors/adjoint.jl:19` | `Base.adjoint` returns the lazy `AdjointTensorMap` wrapper (struct at 9-12); the typed eager form is rowed at `typed::TensorMap::adjoint` |
| `Tensor::braid` | TensorKit | 0.17.0 | `tensors/indexmanipulations.jl:331-342` | both facades (identical claim on the erased and typed item) |
| `Tensor::compose` | TensorKit | 0.17.0 | `tensors/tensoroperations.jl:388-420` | fermionic supertrace twist applied only in `blas_contract!`, never in `mul!` |
| `Tensor::compose` | TensorKit | 0.17.0 | `tensors/linalg.jl:38-42` | categorical composition `compose` behind `A * B`; both facades (identical claim on the erased and typed item) |
| `Tensor::contract` | TensorKit | 0.17.0 | `tensors/tensoroperations.jl:119-146` | `TO.tensorcontract!` with default `pAB`; both facades (identical claim on the erased and typed item); see the `contract_ordered` rows for the `pAB` details |
| `Tensor::data` | TensorKit | 0.17.0 | `tensors/tensor.jl:10-35` | `TensorMap`'s flat `data` vector — the coupled-sector matrix layout tenet's storage claims equivalence with (also the erased module-doc claim) |
| `Tensor::dot` | TensorKit | 0.17.0 | `tensors/linalg.jl:255` | `LinearAlgebra.dot` alias of `inner`; both facades (identical claim on the erased and typed item) |
| `Tensor::eig_full` | MatrixAlgebraKit | 0.6.9 | `interface/eig.jl:12-33` | both facades (identical claim on the erased and typed item) |
| `Tensor::eig_trunc` | MatrixAlgebraKit | 0.6.9 | `interface/eig.jl:35-87` | both facades (identical claim on the erased and typed item) |
| `Tensor::eig_vals` | MatrixAlgebraKit | 0.6.9 | `interface/eig.jl:140-158` | both facades (identical claim on the erased and typed item) |
| `Tensor::eigh_full` | MatrixAlgebraKit | 0.6.9 | `interface/eigh.jl:14-37` | both facades (identical claim on the erased and typed item) |
| `Tensor::eigh_trunc` | MatrixAlgebraKit | 0.6.9 | `interface/eigh.jl:39-91` | both facades (identical claim on the erased and typed item) |
| `Tensor::eigh_vals` | MatrixAlgebraKit | 0.6.9 | `interface/eigh.jl:144-162` | both facades (identical claim on the erased and typed item) |
| `Tensor::exp` | TensorKit | 0.17.0 | `tensors/linalg.jl:44` | `exp` copies, then calls `exp!` |
| `Tensor::exp` | TensorKit | 0.17.0 | `tensors/linalg.jl:420-427` | `exp!`: `domain == codomain` check, per-block dense exponential |
| `Tensor::exp` | TensorKit | 0.17.0 | `tensors/diagonal.jl:383-390` | compact arm: `exp(::DiagonalTensorMap)` is unconditionally elementwise |
| `Tensor::flip` | TensorKit | 0.17.0 | `tensors/indexmanipulations.jl:8-29` | |
| `Tensor::flip` | TensorKit | 0.17.0 | `fusiontrees/braiding_manipulations.jl:384-413` | per-leg Z-isomorphism phase (χ, θ) of the fusion-tree `flip` |
| `Tensor::from_block_fn` | TensorKit | 0.17.0 | `tensors/tensor.jl` | divergence: no TK counterpart (name-only row — no single defensible range); TK's constructor surface is `undef`/`zeros`/`ones` (283-318) and the rand family (320-408); both facades (identical claim on the erased and typed item) |
| `Tensor::id` | TensorKit | 0.17.0 | `tensors/linalg.jl:75-82` | both facades (identical claim on the erased and typed item) |
| `Tensor::inner` | TensorKit | 0.17.0 | `tensors/vectorinterface.jl:114-123` | `VectorInterface.inner`, quantum-dimension weighted; TK `dot` alias at `tensors/linalg.jl:255`; both facades (identical claim on the erased and typed item) |
| `Tensor::inv` | TensorKit | 0.17.0 | `tensors/linalg.jl:375-387` | both facades (identical claim on the erased and typed item) |
| `Tensor::is_antihermitian` | TensorKit | 0.17.0 | `factorizations/factorizations.jl:72-76` | per coupled block via MAK `isantihermitian` (`common/matrixproperties.jl:90-102`); both facades (identical claim on the erased and typed item) |
| `Tensor::is_hermitian` | TensorKit | 0.17.0 | `factorizations/factorizations.jl:67-71` | `LinearAlgebra.ishermitian` alias at 77; per coupled block via MAK `ishermitian` (`common/matrixproperties.jl:66-78`); both facades (identical claim on the erased and typed item) |
| `Tensor::is_isometric` | MatrixAlgebraKit | 0.6.9 | `common/matrixproperties.jl:1-21` | TK applies it per coupled block (`factorizations/factorizations.jl:97-100`, `is_left_isometric`); both facades (identical claim on the erased and typed item) |
| `Tensor::is_posdef` | TensorKit | 0.17.0 | `factorizations/factorizations.jl:86-94` | `isposdef` + `isposdef!`, Cholesky per coupled block; both facades (identical claim on the erased and typed item) |
| `Tensor::is_unitary` | MatrixAlgebraKit | 0.6.9 | `common/matrixproperties.jl:23-38` | both facades (identical claim on the erased and typed item) |
| `Tensor::isometry` | TensorKit | 0.17.0 | `tensors/linalg.jl:149-158` | |
| `Tensor::isomorphism` | TensorKit | 0.17.0 | `tensors/linalg.jl:102-109` | |
| `Tensor::left_null` | MatrixAlgebraKit | 0.6.9 | `interface/orthnull.jl:167-244` | both facades (identical claim on the erased and typed item) |
| `Tensor::left_orth` | MatrixAlgebraKit | 0.6.9 | `interface/orthnull.jl:3-82` | TK 0.17 export of the MAK name; both facades (identical claim on the erased and typed item) |
| `Tensor::left_polar` | MatrixAlgebraKit | 0.6.9 | `interface/polar.jl:3-20` | both facades (identical claim on the erased and typed item); TK factor-space glue on the typed `left_polar` rows |
| `Tensor::lq_compact` | MatrixAlgebraKit | 0.6.9 | `interface/lq.jl:22-43` | both facades (identical claim on the erased and typed item); TK diagonal fast-path divergence on the typed `lq_compact` row |
| `Tensor::lq_full` | MatrixAlgebraKit | 0.6.9 | `interface/lq.jl:3-20` | both facades (identical claim on the erased and typed item) |
| `Tensor::norm` | TensorKit | 0.17.0 | `tensors/linalg.jl:257-275` | quantum-dimension-weighted Frobenius norm (`_norm` at `p = 2`); both facades (identical claim on the erased and typed item) |
| `Tensor::norm_inf` | TensorKit | 0.17.0 | `tensors/linalg.jl:262-265` | `_norm`'s `p == Inf` branch: maximum absolute stored entry, unweighted; both facades (identical claim on the erased and typed item) |
| `Tensor::norm_p` | TensorKit | 0.17.0 | `tensors/linalg.jl:257-275` | `norm(t, p)` and the `_norm` block reduction |
| `Tensor::normalize` | TensorKit | 0.17.0 | `tensors/linalg.jl:18-19` | both facades (identical claim on the erased and typed item) |
| `Tensor::permute` | TensorKit | 0.17.0 | `tensors/indexmanipulations.jl:242-259` | both facades (identical claim on the erased and typed item) |
| `Tensor::pinv` | TensorKit | 0.17.0 | `tensors/linalg.jl:388-396` | both facades (identical claim on the erased and typed item) |
| `Tensor::project_antihermitian` | MatrixAlgebraKit | 0.6.9 | `interface/projections.jl:16-29` | both facades (identical claim on the erased and typed item) |
| `Tensor::project_hermitian` | MatrixAlgebraKit | 0.6.9 | `interface/projections.jl:1-14` | both facades (identical claim on the erased and typed item) |
| `Tensor::qr_compact` | MatrixAlgebraKit | 0.6.9 | `interface/qr.jl:22-44` | both facades (identical claim on the erased and typed item); TK diagonal fast-path divergence on the typed `qr_compact` row |
| `Tensor::qr_full` | MatrixAlgebraKit | 0.6.9 | `interface/qr.jl:3-20` | both facades (identical claim on the erased and typed item); see also the typed `qr_full` row |
| `Tensor::repartition` | TensorKit | 0.17.0 | `tensors/indexmanipulations.jl:464-474` | both facades (identical claim on the erased and typed item) |
| `Tensor::right_null` | MatrixAlgebraKit | 0.6.9 | `interface/orthnull.jl:246-323` | both facades (identical claim on the erased and typed item) |
| `Tensor::right_orth` | MatrixAlgebraKit | 0.6.9 | `interface/orthnull.jl:84-163` | TK 0.17 export of the MAK name; both facades (identical claim on the erased and typed item) |
| `Tensor::right_polar` | MatrixAlgebraKit | 0.6.9 | `interface/polar.jl:22-40` | both facades (identical claim on the erased and typed item); TK factor-space glue on the typed `right_polar` row |
| `Tensor::scale` | TensorKit | 0.17.0 | `tensors/vectorinterface.jl:24-27` | `VectorInterface.scale`, behind `α * t`; both facades (identical claim on the erased and typed item) |
| `Tensor::space` | TensorKit | 0.17.0 | `tensors/abstracttensor.jl:196-201` | `space(t, i)` flat-leg convention (domain legs dualized) |
| `Tensor::sqrt` | TensorKit | 0.17.0 | `tensors/diagonal.jl:384-390` | `sqrt.(d.data)` elementwise on the diagonal; both facades (identical claim on the erased and typed item) |
| `Tensor::svd_compact` | MatrixAlgebraKit | 0.6.9 | `interface/svd.jl:23-42` | both facades (identical claim on the erased and typed item) |
| `Tensor::svd_full` | MatrixAlgebraKit | 0.6.9 | `interface/svd.jl:3-21` | both facades (identical claim on the erased and typed item) |
| `Tensor::svd_trunc` | MatrixAlgebraKit | 0.6.9 | `interface/svd.jl:44-93` | both facades (identical claim on the erased and typed item); return type rowed under `SvdTrunc` |
| `Tensor::svd_vals` | MatrixAlgebraKit | 0.6.9 | `interface/svd.jl:144-156` | both facades (identical claim on the erased and typed item) |
| `Tensor::tr` | TensorKit | 0.17.0 | `tensors/linalg.jl:319-327` | both facades (identical claim on the erased and typed item) |
| `Tensor::trace_pairs` | TensorKit | 0.17.0 | `tensors/tensoroperations.jl:72-87` | `TO.tensortrace!`; both facades (identical claim on the erased and typed item) |
| `Tensor::transpose` | TensorKit | 0.17.0 | `tensors/indexmanipulations.jl:401-411` | both facades (identical claim on the erased and typed item) |
| `Tensor::transpose_axes` | TensorKit | 0.17.0 | `tensors/indexmanipulations.jl:401-411` | the same TK `transpose`, reached with an explicit cyclic `Index2Tuple`; both facades (identical claim on the erased and typed item) |
| `Tensor::twist` | TensorKit | 0.17.0 | `tensors/indexmanipulations.jl:62-97` | `twist!` (62-78) and `twist` (90-97) |
| `Tensor::unitary` | TensorKit | 0.17.0 | `tensors/linalg.jl:129-132` | |
| `Tensor::zeros` | TensorKit | 0.17.0 | `tensors/tensor.jl:283-318` | the generated `zeros`/`ones` constructor pair; both facades (identical claim on the erased and typed item) |
| `tensor` (module doc) | TensorKit | 0.17.0 | `tensors/tensor.jl:10-35` | coupled-sector matrix storage-layout equivalence claim; see the `Tensor::data` row |
| `typed` (module doc) | TensorKitSectors | 0.3.4 | `product.jl:245-294` | `ProductSector` / Deligne product `⊠` |
| `typed::BlockFusionTrees` | TensorKit | 0.17.0 | `tensors/abstracttensor.jl:348-352` | named after `fusiontrees(t)`; its accessors mirror the `f₁.uncoupled`/`f₂.uncoupled`/coupled fields of TK fusion-tree pairs, and `blocksectors` (331-335) is the coarser surface it deliberately is not |
| `typed::EigTrunc` | MatrixAlgebraKit | 0.6.9 | `interface/eig.jl:35-87` | return-type struct for `eig_trunc` |
| `typed::EighTrunc` | MatrixAlgebraKit | 0.6.9 | `interface/eigh.jl:39-91` | return-type struct for `eigh_trunc`; field order `d`, `v` as MAK `initialize_output` |
| `typed::GradedSpace` | TensorKit | 0.17.0 | `spaces/gradedspace.jl:2-29` | the struct this leg type mirrors (docstring 2-25, definition 26-29) |
| `typed::GradedSpace::degeneracies` | TensorKit | 0.17.0 | `spaces/gradedspace.jl:96-101` | `dim(V, c)` |
| `typed::GradedSpace::sectors` | TensorKit | 0.17.0 | `spaces/gradedspace.jl:180-187` | divergence: tenet returns stored labels as-is, TK dualizes stored keys on read when `isdual(V)` |
| `typed::GradedSpace::truncspace` | TensorKit | 0.17.0 | `factorizations/truncation.jl:261-270` | `findtruncated(_svd)` for `TruncationSpace` — the semantics of the strategy; `truncspace` itself is `factorizations/truncation.jl:23-26` |
| `typed::GradedSpace::try_dual` | TensorKit | 0.17.0 | `spaces/gradedspace.jl:112` | divergence: TK flips the flag and dualizes lazily on read; tenet rewrites the stored sector table eagerly |
| `typed::GradedSpace::try_dual` | TensorKit | 0.17.0 | `spaces/vectorspaces.jl:69-73` | the `dual(dual(V)) == V` contract |
| `typed::GradedSpace::try_new` | TensorKit | 0.17.0 | `spaces/gradedspace.jl:70-85` | `GradedSpace` / `Vect[I]` constructor family |
| `typed::TensorMap` | TensorKit | 0.17.0 | `tensors/tensor.jl:10-35` | convention: payload scalar `T` independent of the sector type, as TK's `TensorMap{T, S, ...}` parameters separate them |
| `typed::TensorMap::absorb` | TensorKit | 0.17.0 | `tensors/linalg.jl:531-545` | `absorb` (531), `absorb!` (532-545); rank-check `DimensionError` at 533-534; shared-block `min` region copy at 538-543; both facades (identical claim on the erased and typed item) |
| `typed::TensorMap::adjoint` | TensorKit | 0.17.0 | `tensors/linalg.jl:218` | eager `adjoint!` into a fresh destination |
| `typed::TensorMap::block_count` | TensorKit | 0.17.0 | `tensors/abstracttensor.jl:331-335` | `length(blocksectors(t))` counts coupled sectors |
| `typed::TensorMap::catcodomain` | TensorKit | 0.17.0 | `tensors/linalg.jl:498-514` | domain match 499-500; codomain duality 503-504; direct sum `V = V1 ⊕ V2` 506; per-sector row slabs, `t1` first, 509-512; both facades (identical claim on the erased and typed item) |
| `typed::TensorMap::catdomain` | TensorKit | 0.17.0 | `tensors/linalg.jl:479-497` | codomain match 480-483; domain duality 486-487; direct sum `V = V1 ⊕ V2` 489; per-sector column slabs, `t1` first, 492-495; both facades (identical claim on the erased and typed item) |
| `typed::TensorMap::codomain_rank` | TensorKit | 0.17.0 | `tensors/abstracttensor.jl:239-241` | `numout` |
| `typed::TensorMap::codomain_spaces` | TensorKit | 0.17.0 | `tensors/abstracttensor.jl:204-214` | `codomain(t)` |
| `typed::TensorMap::contract_ordered` | TensorKit | 0.17.0 | `tensors/tensoroperations.jl:119-146` | `TO.tensorcontract!` and its `pAB` output permutation |
| `typed::TensorMap::contract_ordered` | TensorKit | 0.17.0 | `tensors/tensoroperations.jl:159-167` | destination structure `permute(compose(sA, sB), pAB)` |
| `typed::TensorMap::domain_rank` | TensorKit | 0.17.0 | `tensors/abstracttensor.jl:253-255` | `numin` |
| `typed::TensorMap::domain_spaces` | TensorKit | 0.17.0 | `tensors/abstracttensor.jl:217-226` | `domain(t)` |
| `typed::TensorMap::exp` | TensorKit | 0.17.0 | `tensors/linalg.jl:44` | `exp` copies, then calls `exp!` |
| `typed::TensorMap::exp` | TensorKit | 0.17.0 | `tensors/linalg.jl:420-427` | `exp!`: `domain == codomain` check, per-block dense exponential |
| `typed::TensorMap::flip` | TensorKit | 0.17.0 | `tensors/indexmanipulations.jl:21-29` | |
| `typed::TensorMap::flip` | TensorKit | 0.17.0 | `fusiontrees/braiding_manipulations.jl:384-413` | per-leg Z-isomorphism phase (χ, θ) of the fusion-tree `flip` |
| `typed::TensorMap::im` | TensorKit | 0.17.0 | `tensors/abstracttensor.jl:718-728` | `Base.imag` |
| `typed::TensorMap::insert_left_unit` | TensorKit | 0.17.0 | `tensors/indexmanipulations.jl:124-138` | both facades (identical claim on the erased and typed item) |
| `typed::TensorMap::insert_left_unit` | TensorKit | 0.17.0 | `tensors/indexmanipulations.jl:129-130` | `TensorMap` arm: payload reused, space rewrapped |
| `typed::TensorMap::insert_left_unit` | TensorKit | 0.17.0 | `tensors/indexmanipulations.jl:132-137` | non-`TensorMap` arm: `similar` + blockwise copy |
| `typed::TensorMap::insert_right_unit` | TensorKit | 0.17.0 | `tensors/indexmanipulations.jl:158-172` | both facades (identical claim on the erased and typed item) |
| `typed::TensorMap::isometry` | TensorKit | 0.17.0 | `tensors/linalg.jl:149-158` | |
| `typed::TensorMap::isomorphism` | TensorKit | 0.17.0 | `tensors/linalg.jl:102-109` | |
| `typed::TensorMap::left_polar` | TensorKit | 0.17.0 | `factorizations/matrixalgebrakit.jl:204-208` | factor spaces: `w` on the input's homspace, `p` on `domain ← domain` |
| `typed::TensorMap::left_polar` | TensorKit | 0.17.0 | `factorizations/diagonal.jl:8-14` | `DiagonalTensorMap` gets only `copy_input` for the polars — no diagonal polar specialization |
| `typed::TensorMap::leg_dims` | TensorKit | 0.17.0 | `tensors/abstracttensor.jl:196-201` | `space(t, i)` |
| `typed::TensorMap::lq_compact` | TensorKit | 0.17.0 | `factorizations/diagonal.jl:29-41,61-66` | divergence: TK's `DiagonalAlgorithm` LQ fast path not adopted (#613 Group 4) |
| `typed::TensorMap::norm_p` | TensorKit | 0.17.0 | `tensors/linalg.jl:257-275` | `norm(t, p)` and the `_norm` block reduction |
| `typed::TensorMap::numin` | TensorKit | 0.17.0 | `tensors/abstracttensor.jl:253-255` | both facades (identical claim on the erased and typed item) |
| `typed::TensorMap::numind` | TensorKit | 0.17.0 | `tensors/abstracttensor.jl:267` | both facades (identical claim on the erased and typed item) |
| `typed::TensorMap::numout` | TensorKit | 0.17.0 | `tensors/abstracttensor.jl:239-241` | both facades (identical claim on the erased and typed item) |
| `typed::TensorMap::qr_compact` | TensorKit | 0.17.0 | `factorizations/diagonal.jl:16-28,61-66` | divergence: TK's `DiagonalAlgorithm` QR fast path not adopted (#613 Group 4) |
| `typed::TensorMap::qr_full` | TensorKit | 0.17.0 | `factorizations/diagonal.jl:16-28,61-66` | same non-adoption as `qr_compact` |
| `typed::TensorMap::rand` | TensorKit | 0.17.0 | `tensors/tensor.jl:320-408` | the generated `rand`/`randn`/`randexp`/`randisometry` family; both facades (identical claim on the erased and typed item) |
| `typed::TensorMap::rand_with_seed` | TensorKit | 0.17.0 | `tensors/tensor.jl:320-408` | divergence: TK threads a caller-supplied `rng` (overloads at 363-406 inside the generated block), no integer-seed entry point |
| `typed::TensorMap::rank` | TensorKit | 0.17.0 | `tensors/abstracttensor.jl:267` | `numind` |
| `typed::TensorMap::re` | TensorKit | 0.17.0 | `tensors/abstracttensor.jl:707-717` | `Base.real` |
| `typed::TensorMap::remove_unit` | TensorKit | 0.17.0 | `tensors/indexmanipulations.jl:186-197` | |
| `typed::TensorMap::right_polar` | TensorKit | 0.17.0 | `factorizations/matrixalgebrakit.jl:210-214` | factor spaces: `p` on `codomain ← codomain`, `wh` on the input's homspace |
| `typed::TensorMap::scalar` | TensorKit | 0.17.0 | `tensors/tensoroperations.jl:446-451` | empty payload reads as zero |
| `typed::TensorMap::to_c64` | TensorKit | 0.17.0 | `tensors/abstracttensor.jl:696-705` | `Base.complex` |
| `typed::TensorMap::twist` | TensorKit | 0.17.0 | `tensors/indexmanipulations.jl:90-97` | `twist`; in-place `twist!` at 62-78 |
| `typed::TensorMap::twist` | TensorKit | 0.17.0 | `tensors/indexmanipulations.jl:34-51` | `has_shared_twist` identity-twist detection |
| `typed::TensorMap::twist` | TensorKit | 0.17.0 | `tensors/indexmanipulations.jl:91-93` | `copy = false` default shares `t` on identity twist |
| `typed::TensorMap::twist` | TensorKit | 0.17.0 | `tensors/diagonal.jl:84-89` | `similar(::DiagonalTensorMap)` preserves diagonal storage, so TK's diagonal twist stays diagonal |
| `typed::TensorMap::unitary` | TensorKit | 0.17.0 | `tensors/linalg.jl:129-132` | |
| `typed::TensorMap::zeros_like` | TensorKit | 0.17.0 | `tensors/vectorinterface.jl:7-20` | `zerovector`; both facades (identical claim on the erased and typed item) |
