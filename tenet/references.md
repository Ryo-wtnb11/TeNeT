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

| Item | Upstream component | Revision | Source | Note |
| --- | --- | --- | --- | --- |
| `Tensor::add` | TensorKit | 0.17.0 | `tensors/vectorinterface.jl:67-99` | divergence: TK computes `β*ty + α*tx` (first coefficient on the *second* argument); tenet's `alpha` belongs to `self` |
| `Tensor::compose` | TensorKit | 0.17.0 | `tensors/tensoroperations.jl:388-420` | fermionic supertrace twist applied only in `blas_contract!`, never in `mul!` |
| `Tensor::exp` | TensorKit | 0.17.0 | `tensors/linalg.jl:44` | `exp` copies, then calls `exp!` |
| `Tensor::exp` | TensorKit | 0.17.0 | `tensors/linalg.jl:420-427` | `exp!`: `domain == codomain` check, per-block dense exponential |
| `Tensor::exp` | TensorKit | 0.17.0 | `tensors/diagonal.jl:383-390` | compact arm: `exp(::DiagonalTensorMap)` is unconditionally elementwise |
| `Tensor::flip` | TensorKit | 0.17.0 | `tensors/indexmanipulations.jl:8-29` | |
| `Tensor::flip` | TensorKit | 0.17.0 | `fusiontrees/braiding_manipulations.jl:384-413` | per-leg Z-isomorphism phase (χ, θ) of the fusion-tree `flip` |
| `Tensor::id` | TensorKit | 0.17.0 | `tensors/linalg.jl:75-82` | |
| `Tensor::isometry` | TensorKit | 0.17.0 | `tensors/linalg.jl:149-158` | |
| `Tensor::isomorphism` | TensorKit | 0.17.0 | `tensors/linalg.jl:102-109` | |
| `Tensor::norm_p` | TensorKit | 0.17.0 | `tensors/linalg.jl:257-275` | `norm(t, p)` and the `_norm` block reduction |
| `Tensor::sqrt` | TensorKit | 0.17.0 | `tensors/diagonal.jl:384-390` | `sqrt.(d.data)` elementwise on the diagonal |
| `Tensor::twist` | TensorKit | 0.17.0 | `tensors/indexmanipulations.jl:62-97` | `twist!` (62-78) and `twist` (90-97) |
| `Tensor::unitary` | TensorKit | 0.17.0 | `tensors/linalg.jl:129-132` | |
| `typed` (module doc) | TensorKitSectors | 0.3.4 | `product.jl:245-294` | `ProductSector` / Deligne product `⊠` |
| `typed::GradedSpace::degeneracies` | TensorKit | 0.17.0 | `spaces/gradedspace.jl:96-101` | `dim(V, c)` |
| `typed::GradedSpace::sectors` | TensorKit | 0.17.0 | `spaces/gradedspace.jl:180-187` | divergence: tenet returns stored labels as-is, TK dualizes stored keys on read when `isdual(V)` |
| `typed::GradedSpace::truncspace` | TensorKit | 0.17.0 | `factorizations/truncation.jl:261-270` | `findtruncated(_svd)` for `TruncationSpace` — the semantics of the strategy; `truncspace` itself is `factorizations/truncation.jl:23-26` |
| `typed::GradedSpace::try_dual` | TensorKit | 0.17.0 | `spaces/gradedspace.jl:112` | divergence: TK flips the flag and dualizes lazily on read; tenet rewrites the stored sector table eagerly |
| `typed::GradedSpace::try_dual` | TensorKit | 0.17.0 | `spaces/vectorspaces.jl:69-73` | the `dual(dual(V)) == V` contract |
| `typed::GradedSpace::try_new` | TensorKit | 0.17.0 | `spaces/gradedspace.jl:70-85` | `GradedSpace` / `Vect[I]` constructor family |
| `typed::TensorMap::absorb` | TensorKit | 0.17.0 | `tensors/linalg.jl:531-545` | `absorb` (531), `absorb!` (532-545); rank-check `DimensionError` at 533-534; shared-block `min` region copy at 538-543 |
| `typed::TensorMap::adjoint` | TensorKit | 0.17.0 | `tensors/linalg.jl:218` | eager `adjoint!` into a fresh destination |
| `typed::TensorMap::block_count` | TensorKit | 0.17.0 | `tensors/abstracttensor.jl:331-335` | `length(blocksectors(t))` counts coupled sectors |
| `typed::TensorMap::catcodomain` | TensorKit | 0.17.0 | `tensors/linalg.jl:498-514` | domain match 499-500; codomain duality 503-504; direct sum `V = V1 ⊕ V2` 506; per-sector row slabs, `t1` first, 509-512 |
| `typed::TensorMap::catdomain` | TensorKit | 0.17.0 | `tensors/linalg.jl:479-497` | codomain match 480-483; domain duality 486-487; direct sum `V = V1 ⊕ V2` 489; per-sector column slabs, `t1` first, 492-495 |
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
| `typed::TensorMap::insert_left_unit` | TensorKit | 0.17.0 | `tensors/indexmanipulations.jl:124-138` | |
| `typed::TensorMap::insert_left_unit` | TensorKit | 0.17.0 | `tensors/indexmanipulations.jl:129-130` | `TensorMap` arm: payload reused, space rewrapped |
| `typed::TensorMap::insert_left_unit` | TensorKit | 0.17.0 | `tensors/indexmanipulations.jl:132-137` | non-`TensorMap` arm: `similar` + blockwise copy |
| `typed::TensorMap::insert_right_unit` | TensorKit | 0.17.0 | `tensors/indexmanipulations.jl:158-172` | |
| `typed::TensorMap::isometry` | TensorKit | 0.17.0 | `tensors/linalg.jl:149-158` | |
| `typed::TensorMap::isomorphism` | TensorKit | 0.17.0 | `tensors/linalg.jl:102-109` | |
| `typed::TensorMap::leg_dims` | TensorKit | 0.17.0 | `tensors/abstracttensor.jl:196-201` | `space(t, i)` |
| `typed::TensorMap::left_polar` | TensorKit | 0.17.0 | `factorizations/matrixalgebrakit.jl:204-208` | factor spaces: `w` on the input's homspace, `p` on `domain ← domain` |
| `typed::TensorMap::left_polar` | TensorKit | 0.17.0 | `factorizations/diagonal.jl:8-14` | `DiagonalTensorMap` gets only `copy_input` for the polars — no diagonal polar specialization |
| `typed::TensorMap::lq_compact` | TensorKit | 0.17.0 | `factorizations/diagonal.jl:29-41,61-66` | divergence: TK's `DiagonalAlgorithm` LQ fast path not adopted (#613 Group 4) |
| `typed::TensorMap::norm_p` | TensorKit | 0.17.0 | `tensors/linalg.jl:257-275` | `norm(t, p)` and the `_norm` block reduction |
| `typed::TensorMap::numin` | TensorKit | 0.17.0 | `tensors/abstracttensor.jl:253-255` | |
| `typed::TensorMap::numind` | TensorKit | 0.17.0 | `tensors/abstracttensor.jl:267` | |
| `typed::TensorMap::numout` | TensorKit | 0.17.0 | `tensors/abstracttensor.jl:239-241` | |
| `typed::TensorMap::qr_compact` | TensorKit | 0.17.0 | `factorizations/diagonal.jl:16-28,61-66` | divergence: TK's `DiagonalAlgorithm` QR fast path not adopted (#613 Group 4) |
| `typed::TensorMap::qr_full` | TensorKit | 0.17.0 | `factorizations/diagonal.jl:16-28,61-66` | same non-adoption as `qr_compact` |
| `typed::TensorMap::rand` | TensorKit | 0.17.0 | `tensors/tensor.jl:320-408` | the generated `rand`/`randn`/`randexp`/`randisometry` family |
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
| `typed::TensorMap::zeros_like` | TensorKit | 0.17.0 | `tensors/vectorinterface.jl:7-20` | `zerovector` |
