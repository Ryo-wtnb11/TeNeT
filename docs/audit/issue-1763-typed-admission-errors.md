# CUDA tree-transform admission error categories — reference provenance

Authority: TeNeT `f6f36b481be44680f0cdeb380f867b3db3da0fb5` (`origin/main`),
issue [#1763](https://github.com/Ryo-wtnb11/TeNeT/issues/1763). Owning code:
`tenet-operations/src/opaque_admission.rs`
(`From<TreeTransformAdmissionError> for OperationError`, `validate_stage_a`,
`map_task_error`).

This artifact is revision-pinned evidence, not current capability authority.

## Invariant

A failure that the Host tree-transform replay also detects reaches the caller
of the CUDA executor with the Host's `OperationError`, not a collapsed
`InvalidArgument`. Stage A order is unchanged: structures, checked workspace
arithmetic, exact lengths, then placement, context and capabilities, all
before any device work. Execution and successful replay are unchanged.

| Admission variant | Base mapping | Host producer of the same failure | New mapping |
| --- | --- | --- | --- |
| `Structure(t)` | `InvalidArgument` | `task_view.rs:validate_structures_and_lengths` → `structure_identity.rs:validate_structure_identity` | `StructureMismatch { tensor: t }` |
| `Length { expected, actual }` | `InvalidArgument` | `task_view.rs:validate_exact_len` | `ElementCountMismatch { expected, actual }` |
| `ArithmeticOverflow` | `InvalidArgument` | `task_view.rs:validate_workspace_requirements`, `transform_replay/mod.rs:checked_fused_index_len` | `ElementCountOverflow` |
| `ArithmeticOverflow` from Stage C `validate_region` (`Layout::array` of a storage's usable capacity) | `InvalidArgument` | none reachable: a real allocation's capacity always has a valid layout; the failure is an element-count overflow | `ElementCountOverflow` |
| `Task(e)` | `InvalidArgument` | the Host returns `e` itself | `e` |
| `Placement`, `Context`, `Capability`, `InvalidRegion`, `Aliasing`, `WorkspaceCapacity`, `CoefficientReadiness` | `InvalidArgument` | none: Host storage is borrowed slices with no placement, context, region or readiness to admit | `InvalidArgument` (unchanged) |

## Caller paths

All single-member CUDA replays reach the conversion through one function,
`cuda_transform.rs:CudaTreeTransformExecutor::replay_with_destination_scales`
(`replay` delegates to it):

- ordinary returning transforms: `tenet/src/typed/cuda_transform.rs:tree_transform_cuda`
  (`permute`, `braid`, `transpose`, `repartition`, `twist`);
- destination transforms: `tree_transform_into_cuda` (`*_into`, `Axpby(beta)`);
- prepared contraction replay: `tenet-tensors/src/contract/dynamic/cuda.rs`
  and `artifact.rs`.

The member-stack path (`CudaSingleMemberRegions::prepare` / `prepare_scaled`,
used by `tenet/src/typed/contract_batch.rs`) calls
`validate_structures_and_lengths` directly and already returned the Host
categories; it does not pass through this conversion and is unchanged.

## Reference search

Nothing was ported. Rust error enums have no literal reference equivalent;
the references only separate input-geometry checks from kernel execution,
which Stage A already does.

| Reference | Revision | Searched path | Finding |
| --- | --- | --- | --- |
| TensorKit.jl | `cfaa073e4d1e3eb2167edcbdc3be9872f41e7d91` | `src/tensors/indexmanipulations.jl:add_transform!` → `spacecheck_transform`; `add_transform_kernel!` (generic, `AbelianTreeTransformer`, `GenericTreeTransformer` methods) | Geometry is checked once in `@boundscheck spacecheck_transform`, which throws a typed `SpaceMismatch`; the kernels assume admitted inputs and only propagate backend exceptions. TeNeT keeps the same split: the structure/length category is the Host's typed one regardless of executor. |
| QSpace | `d2d3d7da6a59a2e8f2cb7dc8f33e7c345af59371` (`reviews/reference-sources/QSpace-current`) | `Source/QSpace.cc:QSpace<TQ,TD>::contract`, `contract_matchAB_groupC` | QDIM, permutation, rank, fdir and itag checks run before block work and abort via `wblog(FL,"ERR …")`; there are no error categories and no GPU path. |

## Preserved TeNeT contract

Host and CUDA return the same `OperationError` for each single supported
structure, length or overflow failure. The order differs for one combination:
the Host checks lengths before workspace overflow, while CUDA Stage A checks
overflow before lengths, so an input that is both mis-sized and
overflowing could report different errors. Real storage cannot produce that
combination. Executor-only resource verdicts stay
`InvalidArgument`; no public error type was added and no capability boundary
moved.
