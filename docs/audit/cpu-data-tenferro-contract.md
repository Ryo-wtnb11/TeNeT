# CPU data ownership and Tenferro contract

This is a dated source audit for [#1140] and its first evidence leaf [#1141].
It records the contract at the revisions below. Current source, tests, and
normative policy remain authoritative after those revisions.

[#1140]: https://github.com/Ryo-wtnb11/TeNeT/issues/1140
[#1141]: https://github.com/Ryo-wtnb11/TeNeT/issues/1141

## Exact authorities

| authority | revision | role |
| --- | --- | --- |
| TeNeT | `4ea8348dad1cf2dbc3907a89c1734aae29881580` | audited implementation |
| crates.io Tenferro 0.3.0 | `9505d5bfd60c890212f0ec44aa9cf07fef185f63` | installed dependency source |
| Tenferro upstream main | `a48866b1a0bb52e6f9712c485925105b14ea30b9` | later comparison only |
| TensorKit | `cfaa073e4d1e3eb2167edcbdc3be9872f41e7d91` | semantic and public-behavior reference |
| QSpace | `dd2cc7e10dc7d3917b23309a44d1fe67adb4dc43` | representation/dataflow reference |

`tenet-dense/Cargo.toml:61-66` specifies registry Tenferro `0.3.0`.
TeNeT does not commit `Cargo.lock`. Offline resolution in the audit worktree
selected version 0.3.0 for every Tenferro crate. Each cached package's
`.cargo_vcs_info.json` records
`9505d5bfd60c890212f0ec44aa9cf07fef185f63`. The generated ignored lock has
SHA-256 `455947c958ea1306757498b66f3eaa74c6d7bdbdccb4e9988ae2ffa15b4aeb41`.

Lock membership includes optional packages. It is not the active default CPU
graph. `tenet-dense` defaults to `cpu-faer`
(`tenet-dense/Cargo.toml:14-25`). The active Tenferro nodes are
`tenferro-cpu`, `tenferro-linalg`, and `tenferro-ad` with `cpu-faer`;
`tenferro-internal-ops` with `autodiff`; `tenferro-tensor` with its
`default`; and their core/runtime/internal support crates. The optional
`tenferro-gpu` package is present in the lock but absent from the active
resolve nodes.

The CI checkout of
`TENFERRO_REV=0ae5e1c769f256c2180a6e0c48eafdd6d878d8f1` and the adjacent
path-dependency comment are stale
(`.github/workflows/ci.yml:17,24-27,77-83`). No manifest patch or path
override makes that checkout part of Cargo resolution at the audited TeNeT
revision.

## A. Data ownership before Tenferro

The TeNeT ownership chain is:

```text
TensorMap space and block identity
  -> shared typed body and reduced payload
  -> operation-owned pack or borrowed dense region
  -> DenseRead / DenseWrite
  -> Tenferro TensorRead / TensorWrite
  -> owned result or overwrite destination
```

| layer | owner and observable contract | source |
| --- | --- | --- |
| Core tensor | `tenet_core::TensorMap` owns storage and an `Arc<BlockStructure>`; an admitted fusion space fixes the block layout. Packed storage length may be smaller than the full physical dense dimension. | `tenet-core/src/tensor_map.rs:1-14,353-389` |
| Storage | Placement, readable/writable storage, and reusable scratch are separate traits. Host ordinary payload and scratch are `Vec<T>`. | `tenet-core/src/storage.rs:8-72,74-109` |
| Typed tensor | `typed::TensorMap` shares its `Runtime` and representation. `TypedTensorBody` pairs a bound space with a separately shared payload. Clone shares; a write route must publish or obtain a unique payload. | `tenet/src/typed.rs:8890-8941,9042-9085` |
| Lazy/compact representations | A lazy adjoint owns its parent and logical space and may materialize once. Compact spectrum storage is distinct from the ordinary dense payload. | `tenet/src/typed.rs:8944-8968` |
| Borrowed dense boundary | `DenseView` and `DenseViewMut` borrow a slice plus shape, nonnegative strides, and offset; constructors validate layout bounds. | `tenet-dense/src/view.rs:6-35,84-105` |
| Backend result | `DenseTensor` owns a Tenferro `Tensor` behind an `Arc` and exposes read-only shape/dtype/slices. It is not `Clone`, and TeNeT has no use that clones the inner Arc. | `tenet-dense/src/tensor.rs:9-19,25-100` |

The final `Arc<Tensor>` adds a local control-block allocation without enabling
sharing in current source. It does not duplicate tensor payload bytes and is
not a Tenferro requirement. Removing it would be a narrow local ownership
simplification, outside #1141 unless measurements or a needed move boundary
justify the change.

`contract_overwrite_into` admits only the exact expected fusion space/layout,
ordinary dense Host storage, no input alias, and unique destination ownership
before mutation. Context-lease admission errors leave the destination
unchanged; after clearing, a later engine error may leave it zeroed or partially
written (`tenet/src/typed.rs:12957-13083`). The API does not promise
transactional rollback.

TensorKit's corresponding `TensorMap` owns one dense reduced-data vector and
its `TensorMapSpace`; its raw-data constructor takes ownership without a copy,
and block access returns reshaped views
(`src/tensors/tensor.jl:5-20,127-154,466-499`). QSpace owns QIDX structural
records, CGR data, and per-record runtime-rank DATA arrays; `permute` rebuilds
the destination records and traverses corresponding DATA/CGR entries
(`Source/QSpace.hh:57-75,233-259,1487-1490,2890-2948`). These references
support keeping structure with tensor ownership and using views/dataflow. They
do not require TeNeT to copy QSpace's separate C++ allocation pattern.

## B. Six shape-change cases

These cases must not be collapsed into one “dynamic shape” capability.

| change | current TeNeT admission and reuse contract | evidence required by #1141 |
| --- | --- | --- |
| 1. Values only | Reuse the plan, compatible private destinations, executor resources, and current layout; overwrite all result values. | Multi-step reused execution must equal fresh execution and an independent indexed dense oracle. |
| 2. Same sectors, changed degeneracies | The contraction order may remain valid. Full connected-space validation runs; size/layout-bound destinations are rebuilt. Scratch capacity may be reused independently. | Growth and shrink across at least three operands, then valid recovery. |
| 3. Same total dimension, redistributed sectors | Equal element count does not imply equal layout. Full sector metadata must mismatch and invalidate old replay state. | Same-total redistribution with no stale intermediate publication. |
| 4. Sectors appear/disappear, including zero/empty blocks | Rebuild admitted spaces, routes, and destinations; revalidate every connection. | Appearance, disappearance, and zero/empty cases across supported fixtures. |
| 5. Input rank/split or topology changes | `PlannedNetwork` rejects input rank/codomain-split drift. A changed graph topology requires a new `Network`/schedule. Order validity and order quality remain separate. | Rejection followed by valid recovery; no claim that one plan accepts new topology. |
| 6. SVD-dependent retained rank | This changes factor output spaces after numerical selection. It is not input-shape polymorphism. Build new factor spaces, layouts, and payloads for kept per-sector ranks. | Reconstruction, orthogonality/isometry, retained subspace, and discarded error; do not compare gauge-dependent factor entries. |

`PlannedNetwork::execute_with_workspace_meter` first checks input count,
`Runtime`, rule identity, and planned rank/split. It compares the complete
per-leg sector snapshot and storage reuse class; a mismatch validates connected
pairs and clears replay state before installing the new snapshot
(`tenet-network/src/network.rs:1644-1774`). The caller-owned
`NetworkExecutionWorkspace` contains private intermediate destinations plus
owner/runtime/rule/snapshot admission state
(`tenet-network/src/network.rs:943-969,1287-1318`). The final owned tensor
leaves the workspace (`:943-949,1621-1629`).

The existing
`workspace_drift_and_provider_allocation_changes_do_not_leave_stale_results`
checks degeneracy/provider/runtime drift and recovery with two operands
(`tenet-network/tests/typed_network.rs:384-440`). It does not carry a retained
intermediate through a three-or-more-operand chain. That is the concrete #1141
evidence gap.

The confirmed defect is in TeNeT's fusion `Structure` producer, before any
Tenferro call. The workspace correctly notices changed sector metadata, clears
replay state, and rebuilds the structure. The rebuilt structure can still omit
a valid non-self-dual sector or match unequal coupled sectors because its two
contraction predicates apply duality at the wrong layers. Workspace reuse made
the wrong answer visible, but neither retained workspace state nor a plan cache
is its root cause.

### Contraction equivalence and consumer map

The repair is confined to the two private predicates used by
`tensorcontract_fusion_block_specs_lowered` in
`tenet-tensors/src/contract/fusion/block_specs.rs:413-461,487-500,591-606,731-768`:

| invariant | reference correspondence | TeNeT boundary |
| --- | --- | --- |
| Contracted all-outgoing labels are dual across the two operands. | TensorKit `cfaa073e4d1e3eb2167edcbdc3be9872f41e7d91`, `src/tensors/tensoroperations.jl:TO.checkcontractible` (180-188), requires the dual external space of A to equal B. | `FusionTreePairKey::external_sectors` already dualizes stored domain sectors (`tenet-core/src/block_structure.rs:151-165`). `contracted_external_sectors_match` must therefore compare dual labels. |
| After permutation into matrix form, the stored contracted fusion-tree bases are identical. | TensorKit's sector matrix product joins equal coupled sectors (`src/tensors/linalg.jl:LinearAlgebra.mul!`, 330-372). QSpace `dd2cc7e10dc7d3917b23309a44d1fe67adb4dc43`, `Source/QSpace.cc:QSpace::contract_matchAB_groupC` (4267-4484), validates index metadata separately and then exact-matches stored contracted QIDX slices. | `contracted_fusion_tree_basis_matches` must compare the complete `FusionTreeKey`: uncoupled and coupled sectors, inner lines, vertices, and dual flags (`tenet-core/src/fusion_tree.rs:1224-1231`). The direct coupled-region path already requires equal stored column/row trees (`tenet-operations/src/fusion_replay.rs:509-515`). |

The single producer feeds core and transformed block-spec loops and is shared
by raw/legacy and prelowered structure constructors. Ordinary host,
dynamic/raw, prelowered overwrite, profiled, standalone backend, and explicit
`PreparedTensorContractFusion` execution consume structures produced there
(`tenet-tensors/src/contract/context.rs:1774-1945`). The plain dense
`TensorContractCache::get_or_compile` (`context.rs:188-259`) is a different
producer, and ordinary fusion resolution does not use that cache. Direct
canonical and DynamicTree lowering remain separate oracles. Checked Generic
network execution also remains separate; multiplicity-free network execution
selects retained overwrite or an owned result, while Checked Generic always
uses the returned route (`tenet-network/src/network.rs:1141-1245`). Raw storage
may compile `Structure` but rejects that execution route, and the public
prelowered device path accepts Core only, so no successful CUDA numerical path
consumes this repair.

The repair preserves the existing block-pair enumeration and O(r) contracted
rank matching and introduces no new allocation, cache, workspace, or copy/pack
mechanism. Correct execution must process every valid block, so allocation,
copy/pack, backend-call, and other resource totals can increase relative to the
incorrect baseline. Timings from an incorrect result are not performance
evidence.

For SVD, `svd_compact_factors_dyn_with_direction` either uses direct coupled
regions or constructs sector matricizations. The fallback builds factor spaces
from each sector's current rank, allocates final factor payloads, reuses
maximum-size operation-local factor workspaces across sectors, and scatters
each result into its admitted layout
(`tenet-matrixalgebra/src/factorize.rs:1631-1772`). Truncation then selects
kept ranks and returns newly wrapped factor spaces/payloads
(`tenet/src/typed.rs:14355-14395`).

Tenferro's dynamic truncated-SVD tutorial does not broaden this contract. It
compiles an explicit `[4,4]` input specification once and runs different
values; a runtime scalar changes only the truncated output extent from 2 to 3
(`docs/tutorials/dynamic-shape-truncated-svd.md:8-10,52-70,93-145,152-154`,
Tenferro `9505d5b` and `a48866b`). The design keeps backend sizes concrete
and explicitly avoids full input-shape polymorphism
(`docs/design/dynamic-symbolic-shapes.md:67-77,252-260,315-342`).

The runtime design does not expose its `ExecProgram` or `ScheduledGraph`
staging as a general public execution surface
(`docs/spec/backend-contract.md:20-47,57-80,240-279`). Shape guards run
before backend side effects (`:103-119`), while extension modules own their
runtime/cache state outside semantic operation identity
(`docs/spec/extension-op.md:257-280,616-656,756-781`). Those boundaries do
not move TeNeT's sector/layout authority into Tenferro.

Likewise, Tenferro's `changing_shape_prepare` benchmark runs 129 ordinary
eager einsums with different concrete shapes, beyond a 128-entry cache limit
(`crates/tenferro-einsum/benches/changing_shape_prepare.rs:10-12,51-96`).
It measures the complete ordinary-call sequence: planning/cache behavior,
numerical execution, eager materialization, and result readback. It does not
isolate preparation cost. A `ConcreteEinsumPlan` records exact input count,
dtype, and shape and rejects changed metadata
(`crates/tenferro-einsum/src/concrete.rs:856-874,949-986,1095-1107,1179-1202`
at `9505d5b`).

## C. Tenferro API integration

Installed 0.3.0 and upstream main must be kept separate:

| operation/boundary | installed 0.3.0 | upstream main | audited TeNeT use |
| --- | --- | --- | --- |
| Tensor owner | `TypedTensor`, `Tensor`, and `TensorValue` are move-only physical owners. `duplicate` is explicit. | Same. | Returned owned tensors are wrapped as `DenseTensor`. |
| Borrowed view | `TensorRead` accepts `&Tensor` or arbitrary-strided `TensorView`; `TensorWrite` accepts `&mut Tensor` or `TensorViewMut` and never resizes. | Same. | `tenferro_view{,_mut}` converts borrowed TeNeT layouts without a payload copy (`tenferro_adapter.rs:910-966`). |
| Dot/GEMM | Public borrowed/into APIs. | Same. | Direct `dot_general_read_into*` for dot, GEMM, accumulation, grouped and strided batches (`:459-478,682-751`). |
| Solve | Public `solve_read_into`; the CPU path may write directly into eligible caller storage. | Same, with a backend-session receiver. | Direct borrowed inputs and caller destination (`:577-615`). |
| SVD/QR/EIGH | Borrowed `*_read` input, owned outputs. No public decomposition `*_read_into`. | Still no decomposition `*_read_into`. | Trait defaults call owned decomposition, then copy/scatter every result to caller destinations (`tenet-dense/src/executor.rs:51-105,315-373`). |
| Values-only SVD | Provider implementation exists behind `#[doc(hidden)] LinalgBackend::svd_values(&Tensor)`; no public borrowed concrete extension. | Public owned `svdvals` and borrowed `svd_values_read`. | Materializes a borrowed view, calls full SVD, and discards vectors (`tenferro_adapter.rs:618-640`); owned by #880. |
| Session | Concrete linalg takes `&mut B: LinalgBackend`; CPU access uses public but `#[doc(hidden)] with_cpu_exec_session`. | Concrete linalg takes `&mut dyn BackendSession` and performs leaf routing internally. | `with_cpu_linalg` correctly scopes both callbacks for installed 0.3.0 (`:1000-1008`); no session is retained. |

Installed source evidence:

- `tenferro-tensor/src/types.rs:1007-1051,3727-3976,3978-4027` at
  `9505d5b`: owned, read, write, and move-only value types.
- `tenferro-linalg/src/tensor_ext.rs:123-190,617-719,818-873`:
  owned/borrowed decomposition and solve-into extensions.
- `tenferro-cpu/src/lib.rs:184-208` and
  `tenferro-cpu/src/exec_session.rs:30-41`: hidden borrowed CPU leaf session.

Upstream source evidence:

- `crates/tenferro-linalg/src/tensor_ext.rs:145-166,637-721,888-897,1800-1831,1923-1931,2315-2345`:
  backend-session concrete surface and public borrowed values-only path.
- `crates/tenferro-linalg/tests/integration/concrete_surface.rs:9-32,426-464`:
  fixed tuple/values-only behavior and direct owned/strided solve destination.

There is one further constructor limitation in both revisions:
`CpuBackend::from_context` always selects
`CpuBackendKind::default_compiled()`; context-plus-kind construction is
private (installed `tenferro-cpu/src/backend.rs:1609-1661`; upstream
`crates/tenferro-cpu/src/backend.rs:1712-1764`). TeNeT therefore cannot share
its `CpuContext` when an explicit non-default provider is requested and
creates a separate backend/thread pool
(`tenet-dense/src/tenferro_adapter.rs:138-163`).

The public borrowed/into/session APIs already suffice for per-call runtime-rank
views, dot/GEMM, and solve. They do not require a Tenferro graph, a second
prepared-plan owner, a common execution IR, or a cache hierarchy. Missing
decomposition destinations, public borrowed values-only SVD in installed
0.3.0, and context-plus-kind construction are narrow API gaps. An upstream
dependency update is a separate design decision.

## Measurement evidence and residual ownership

No timing or speedup result is claimed in this contract. #1141's benchmark
protocol covers independent multi-step correctness, cold and warm execution,
changing structure, many-small and few-large blocks, allocation calls and
requested bytes, and process-wide live-memory peaks. Actual results and raw
sample spreads will be recorded in the pull request and issue only after runs
at the exact immutable integrated commit. Copy/pack counts and private
workspace retention remain unmeasured where the public interfaces expose no
direct counter; allocation or byte totals are not proxies for either quantity.

[#880] retains values-only factorization work. [#1083] retains stream,
multi-device, event, and scheduling scope. [#1084] retains the optional common
execution-IR investigation. Completing #1141 does not close these issues,
establish CPU-wide performance parity, or complete #1140/#1139.

[#880]: https://github.com/Ryo-wtnb11/TeNeT/issues/880
[#1083]: https://github.com/Ryo-wtnb11/TeNeT/issues/1083
[#1084]: https://github.com/Ryo-wtnb11/TeNeT/issues/1084
