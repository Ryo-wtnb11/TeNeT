# Symmetric tensor runtime architecture review

Review date: 2026-08-12

Reviewed revisions:

- TeNeT: [`5390f64a`](https://github.com/Ryo-wtnb11/TeNeT/tree/5390f64a4e58b76f011f58e1576f632bfd569cf0)
- tenferro-rs: [`ec9b0602`](https://github.com/tensor4all/tenferro-rs/tree/ec9b06025774574a5df47e11ccdd730fb0c01c80)
- TensorKit.jl: [`64430c5`](https://github.com/QuantumKitHub/TensorKit.jl/tree/64430c522c7da3c13c8376fc9734a8ecc054a324)
- QSpace v4 public source: [`684bdb8`](https://bitbucket.org/qspace4u/qspace-v4-pub/commits/684bdb817ee6bc14154e12445252a390ff252e70)
- QSpace paper: [SciPost Phys. Codebases 40 (2024)](https://doi.org/10.21468/SciPostPhysCodeb.40)

This review treats TensorKit as the primary reference for categorical semantics and QSpace only as a comparison for separating structural work from reduced numerical execution. It does not assume that TeNeT should reproduce either implementation. Source facts, design inferences, and recommendations are distinguished below.

The work was split into three independent source audits: (1) TensorKit mathematics, tree transformers and caches; (2) QSpace contraction/store execution boundaries; and (3) TeNeT layers, hot paths, caches and open-issue overlap. The supervising review then checked the cited current revisions, reconciled terminology, and rejected proposals that merely duplicated TeNeT's existing completed-transform boundary. No benchmark was run during this review; performance statements are either linked to existing TeNeT measurements or explicitly presented as hypotheses to measure.

## 1. Executive summary

1. The design thesis—**TensorKit-like categorical semantics plus Rust-native ahead-of-execution lowering**—is sound. More importantly, current TeNeT already implements most of it. Runtime-rank operations are compiled into immutable categorical group plans, layout-resolved replay descriptors, strided copy/accumulate work, and pack/GEMM/scatter work. Fusion-tree traversal and F/R evaluation are not present in the warm numerical replay loop.
2. There is no P0 architectural blocker in the inspected CPU path. At the reviewed TeNeT revision, one independent P1 CPU defect remained: direct checked-Generic contraction bypassed the Runtime Generic lane and repeatedly constructed execution resources and contraction artifacts. This was resolved after remeasurement by #1079, which reused existing Runtime resources without adding a global operation cache.
3. `TreeTransformGroupPlan<T>` is a genuinely layout-independent categorical plan. `TreeTransformStructure<T>` is the layout-bound, replay-ready descriptor. Its name understates its role, but its contents are coherent: block layouts, offsets, categorical coefficient payload, GEMM jobs, pack/scatter descriptors, overwrite proof, and a host parallel schedule.
4. The material pre-CUDA gap is the execution interface, not categorical semantics. `TreeTransformBackend` is explicitly host-slice based. CUDA network execution accepts only direct contraction schedules and rejects required result/final permutations. A placement-aware replay/lowering boundary should consume the existing layout-bound descriptor without re-running symmetry reasoning.
5. TeNeT's dynamic rank is not dynamic kernel generation. A rank-17 permutation can use a baked generic N-D strided descriptor. Rank-specialized kernels should be optional backend choices supported by measurements, not public tensor types.
6. The current cache policy is substantially correct: Racah/provider authority owns generated F/R/CGC data; TeNeT retains bounded canonical structures and completed layout-bound transforms; workspace and numerical results are not globally cached. A proposed four-level global cache hierarchy should **not** be implemented by default.
7. TensorKit validates TeNeT's semantic and transformer shape: it builds completed transformers containing coefficients plus concrete strides/offsets. TensorKit's `GenericTreeTransformer` means “not `UniqueFusion`”; it is not specific to `GenericFusion` and also covers multiplicity-free `SimpleFusion`.
8. QSpace illustrates a carrier-algebra amortization pattern but does not prove its benefit for TeNeT's F/R lowering. It persists expensive CGT/X-symbol data and then contracts reduced blocks, but even a warm X-symbol path still performs structural lookup, consistency checks, and small outer-multiplicity transformations. TeNeT should not copy CGT, CGR, RC_STORE, a global mutable database, or demand-grown multiplicity gauges.
9. The finite backend vocabulary should stay internal and compact: fill/scale, strided copy-axpby (including permutation), pack, GEMM batches, scatter-axpby, and reduction. “Grouped” and “batched” are scheduling forms of GEMM, not necessarily separate semantic operations.
10. Before expanding CUDA, implement and benchmark the smallest placement-aware tree-transform replay seam, then add device-side permutation/pack/GEMM/scatter and network lowering. Multi-stream and multi-GPU scheduling must remain after symmetry/layout lowering.

## 2. Design thesis evaluation

The thesis is accepted with one qualification:

> TeNeT should keep a runtime-rank categorical frontend and lower it ahead of replay into a finite numerical vocabulary. It should not require one public, serializable mega-IR before a second backend proves that such an object is useful.

Current TeNeT already follows:

```text
runtime-rank operation
  -> provider-backed FusionTree/F/R reasoning
  -> TreeTransformGroupPlan<T>             (categorical, layout independent)
  -> TreeTransformStructure<T>             (layout bound, replay ready)
  -> strided copy/axpby OR pack/GEMM/scatter
  -> DenseExecutor / host kernels
```

The source explicitly calls `TreeTransformGroupPlan` an immutable categorical transform independent of storage layout (`tenet-operations/src/transform_plan.rs:642-650`). `TreeTransformStructure` resolves keys, block layouts, offsets, and pack/scatter descriptors once (`tenet-operations/src/transform_structure.rs:63-95`). Its recoupling plan already stores finite `DenseGemmBatchJob`s and precomputed run partitions (`:249-259`). Warm replay therefore does not interpret arbitrary-rank FusionTrees.

The remaining thesis violation is backend reach: the current public replay contract takes host-readable/writable storage or raw host slices (`tenet-operations/src/replay_backend.rs:22-79`). The file itself correctly says future device/MPI implementations should use a separate placement-aware trait (`:172-177`).

## 3. TensorKit / QSpace / TeNeT comparison

| Concern | TensorKit | QSpace | Current TeNeT | Verdict |
|---|---|---|---|---|
| Mathematical authority | Sector types, fusion styles, FusionTree, F/R, duality | Representation data and explicit arbitrary-rank CGTs; tensor-local CGR references them, while X-symbols are derived contraction data | Provider traits, sectors, FusionTrees, F/R/duality capabilities | Follow TensorKit-like semantics |
| Reduced tensor | Fusion-tree-indexed reduced blocks | `DATA` reduced multiplet tensors plus `QIDX`/`CGR` | `TensorMap` blocks indexed by canonical fusion-tree structure | Correct direction |
| Transform construction | `AbelianTreeTransformer` or `GenericTreeTransformer` | Per-operation block matching and CGR/X-symbol processing | categorical group plan then completed structure | TeNeT is more explicitly staged |
| Layout binding | Transformer contains strides, offsets, shapes | QIDX/DATA layout is tensor local | `TreeTransformStructure` owns block/layout table and replay descriptors | Sound |
| Warm numerical path | strided block transforms / small matrices | reduced contractions plus X/OM contraction | strided copy/axpby or pack/GEMM/scatter | Sound |
| Persistent structural data | process-global LRU caches | process-global RAM stores plus disk R/C/X stores | provider/Racah data plus bounded runtime/core caches | Do not copy QSpace persistence |
| Backend split | Julia/Strided views, CPU-centric abstraction | MATLAB MEX/C++/OpenMP/BLAS | CPU replay plus separate limited CUDA storage GEMM | Placement-aware transform replay is missing |

## 4. Terminology and mathematical model

For a symmetric tensor,

\[
T = \bigoplus_{\alpha} A_{\alpha}\otimes C_{\alpha},
\]

`A_α` is reduced numerical data on degeneracy spaces and `C_α` is the structural intertwiner selected by sectors, fusion trees, multiplicity labels, conventions, and category data.

| Concept | TensorKit | TeNeT | Storage/reconstruction |
|---|---|---|---|
| irrep / sector | `Sector` values | provider-associated sector type and opaque `SectorId` | sector labels/IDs are metadata; carrier vectors are not tensor payload |
| fusion rule | `a ⊗ b` enumerates each allowed output once; `Nsymbol(a,b,c)` gives its multiplicity | provider fusion/decomposition traits | queried during admission/enumeration |
| binary multiplicity | multiplicity label on fusion vertices | vertex multiplicity in fusion-tree keys/provider data | structural key, not dense tensor axis |
| FusionTree | `FusionTree{I,N,M,L}` | runtime-rank `FusionTreeKey`/pair keys and canonical structures | canonical symbolic key; runtime rank is a value |
| intertwiner | basis vector represented by a tree/path/multiplicity labels | canonical FusionTree block identity | implicit/symbolic, not an explicit dense CG tensor |
| CGC | optional carrier-space coordinates exposed by `fusiontensor`; not required by TensorKit's core F/R/tree semantics | provider/Racah-owned coefficient data where supplied | normally external to tensor storage |
| F-symbol | `Fsymbol`/recoupling | provider rigid-symbol capability | evaluated while compiling categorical transforms; coefficient payload may be retained |
| R-symbol | `Rsymbol`/braiding | provider braiding capability | evaluated during braid lowering; cheap values need not receive their own cache |
| duality/bending | dual sectors, bend/frobenius data | duality and rigid-symbol traits | semantic lowering, absent from replay |
| reduced tensor | `TensorMap` block data | `TensorMap` block storage | stored numerical payload |
| degeneracy space | user multiplicity \(\mathbb C^{n_a}\), `dim(V,a)=n_a`, distinct from intrinsic `dim(a)` | `SectorLeg` degeneracies and block layouts | dimensions determine concrete block shapes/strides |
| higher-rank multiplicity | number of admissible fusion paths including vertex labels | multiple FusionTree keys for fixed external labels | distinct structural blocks or recoupling matrix dimensions |
| product symmetry | product sector/category composition | product/provider composition where implemented | componentwise semantic data; layout is still a reduced tensor layout |

The following distinctions are mandatory:

- **Binary fusion multiplicity** is \(N_{ab}^{c}>1\) at one trivalent vertex.
- **Multiple allowed channels** means several values of \(c\), even if every \(N_{ab}^{c}\leq 1\).
- **Higher-rank fusion-path multiplicity** counts admissible intermediate labels and vertex multiplicities for fixed external/result labels.
- **QSpace outer multiplicity** is the dimension of the space of linearly independent complete CGTs for fixed external irreps. At rank three it can coincide with a binary multiplicity; at higher rank it need not. QSpace documents outer multiplicity for rank-four-and-higher SU(2) although binary SU(2) fusion is multiplicity-free.
- TensorKit `UniqueFusion` means each binary fusion has exactly one output sector; a scalar one-to-one tree transform is a consequence, not its definition. `SimpleFusion` can remain binary multiplicity-free yet require multiple paths and a matrix transformer. `GenericFusion` admits nontrivial binary multiplicity. TensorKit's type named [`GenericTreeTransformer`](https://github.com/QuantumKitHub/TensorKit.jl/blob/64430c522c7da3c13c8376fc9734a8ecc054a324/src/tensors/treetransformers.jl#L44-L153) handles both non-`UniqueFusion` styles.

TeNeT stores reduced blocks and canonical structural identities, not explicit carrier-space CGTs. It reconstructs operation-specific F/R/bend transformations during plan compilation, then retains the completed immutable replay structure in the bounded runtime cache.

## 5. Rust vs Julia execution-model implications

Julia can specialize methods for concrete sector, scalar, space, and storage types at runtime. TensorKit can therefore keep high-level generic calls close to specialized loops. Rust can monomorphize provider/scalar/backend types, but rank, sector count, tree count, and the contribution graph remain runtime values.

That is not a defect. These dimensions determine **data**, not an unbounded instruction set. Rust should represent runtime-rank axes as compact slices/IDs and compile them into fixed executor families. It should avoid:

- trait-object dispatch per block/task;
- heap allocation per contribution during replay;
- repeated hashing of FusionTree objects inside numerical loops;
- rank-specific public tensor types;
- runtime-generated kernels as a prerequisite for arbitrary rank.

An optional backend dispatch of ranks 2–4 to specialized kernels and all other ranks to a generic N-D kernel is reasonable only after the benchmark matrix shows the metadata loop or indexing arithmetic matters.

## 6. Current TeNeT architecture

### A. Mathematical / categorical layer

- Provider traits, `RuleIdentity`, sector admission, fusion/decomposition, duality, rigid/braided symbols: `tenet-core` and typed facade provider bounds.
- Runtime-rank `TreeTransformOperation` (`Transpose`, `Permute`, `Braid`): `tenet-operations/src/transform_key.rs`.
- Fusion-tree pair keys and canonical HomSpace/block structures: `tenet-core`.

### B. Canonical structural layer

- `TreeTransformGroupPlan<T>` and group block specs: categorical source/destination keys, coefficients or small recoupling matrices, logical axis mapping; no offsets/strides/backend (`transform_plan.rs:642-705`).
- Canonical HomSpace and fusion-tree layout tables in `tenet-core`.

### C. Structural-plan layer

`TreeTransformGroupPlan<T>` is the closest exact match. It is compile-local. Coefficient matrices may share a lazily materialized contiguous payload; all-scalar plans deliberately remain inline to avoid allocation.

### D. Layout layer

`TreeTransformStructure<T>` binds categorical groups to source/destination `BlockStructure`, layout table entries, offsets, overwrite coverage, and inactive destination layouts (`transform_structure.rs:63-95`).

### E. Execution plan layer

The same immutable completed structure carries:

- scalar replay descriptors;
- pack-column and scatter-column descriptors;
- scatter disjointness/grouping;
- `DenseGemmBatchJob[]` and precomputed batch runs;
- a host parallel schedule.

This is not a clean one-type-per-box diagram, but it is already a finite execution description. Splitting it solely for conceptual symmetry would add object and cache lifetime complexity without changing the hot path.

### F. Runtime/backend layer

- Host `TreeTransformBackend` and `DenseTreeTransformOperations` execute completed structures.
- `DenseExecutor` accepts batch GEMM jobs with disjoint destination guarantees (`tenet-dense/src/executor.rs:8-22`).
- CUDA currently provides storage-aware GEMM and selected typed operations, while network execution is accepted only if every contraction is direct and no result/final permutation is required (`tenet-network/src/network.rs:1563-1605`).

### G. Cache layer

- Provider/Racah: generated categorical coefficient authority.
- `tenet-core`: bounded/weak interning and canonical HomSpace/layout structure caches.
- Runtime: bounded completed `TreeTransformStructure<T>` LRU keyed by rule identity, operation, exact source/destination structure and storage conjugation (`tenet-tensors/src/tree_transform/cache.rs:43-123`). Defaults are 256 entries and an 8 MiB per-entry admission cap (`:115-117`), with a runtime-wide byte ledger.
- Task/request contexts: optional task-local caches and reusable workspaces.
- Not cached: ordinary numerical results, destination tensors, large workspace buffers globally, or backend stream state.

## 7. Runtime-rank handling review

The current direction is correct. `TreeTransformOperation` stores runtime-length permutations and precomputed axis positions. Compilation enumerates the structural cases once. Replay sees flattened layout entries and fixed descriptor arrays.

Recommended backend strategy:

```text
rank 2/3/4 and measured common stride patterns -> optional specialized kernel
otherwise                                  -> generic baked N-D strided kernel
matrix mixing                              -> pack + GEMM batch + scatter
```

The generic N-D task must contain extents, source/destination strides, offsets, coefficient mode, and a validated alias/overlap contract. It must not contain FusionTrees or call the provider. Rank specialization belongs in backend task selection, not the tensor type system or categorical plan.

## 8. Current cache architecture

### TensorKit

TensorKit uses [configurable cache styles](https://github.com/QuantumKitHub/TensorKit.jl/blob/64430c522c7da3c13c8376fc9734a8ecc054a324/src/auxiliary/caches.jl#L1-L173). It has two relevant memo layers:

- layout-independent `fsbraid`/`fstranspose`, keyed by tree pair/block plus operation and returning an algebraic scalar or matrix; `UniqueFusion` selects `NoCache`, while non-unique fusion uses a global LRU;
- layout-dependent `treebraider`/`treetransposer`, keyed by complete source/destination spaces plus operation and cached globally even for `UniqueFusion`.

Each global cache is a separate 10,000-entry, entry-count rather than byte-count LRU. Concurrent misses may safely duplicate construction. `TaskLocalCache` is implemented but current TensorKit functions do not select it.

The completed transformer value contains [shape/stride/offset structures](https://github.com/QuantumKitHub/TensorKit.jl/blob/64430c522c7da3c13c8376fc9734a8ecc054a324/src/tensors/treetransformers.jl#L10-L129). `AbelianTreeTransformer`—named for its implementation role but selected by `UniqueFusion`—retains a scalar and complete source/destination `(shape,strides,offset)`. `GenericTreeTransformer` retains the path-space matrix \(U\), one common shape per side, and per-tree `(strides,offset)` arrays. FusionTree objects are transient construction inputs; canonical destination/source ordering becomes matrix row/column order.

TensorKit therefore demonstrates that layout-independent and completed caches can coexist, but not that this extra lookup/retention is profitable in Rust. TeNeT should add a categorical cache only after measuring cross-layout reuse and accounting bytes, provider/gauge identity and synchronization.

Two implementation cautions should not be copied: `degeneracystructure` documentation says “object identity,” while its actual `HomSpace` key uses structural equality/hash; and raw `FusionTree` validity assertions are commented out, relying on trusted iterators/call paths. TeNeT checked constructors must continue to prove admissibility before publishing cached plans.

### TeNeT

The runtime completed-transform store is bounded, immutable-value, and compile-outside-lock. This is appropriate. The key includes semantic provider identity and exact structures; the value strongly owns the completed structure through `Arc`. Core canonical caches are bounded or weak. This avoids an indefinitely strongly owned graph.

Potential costs to measure, not assume:

- completed-cache mutex/hash cost for extremely small warm transforms;
- checked-Generic fallback scans when exact intern identity is unavailable;
- repeated categorical compilation across many distinct layouts sharing one category transform;
- core interner contention during concurrent cold planning.

### QSpace

QSpace's process-global in-memory `gCS`, `gRS`, and `gXS` objects support large, expensive, explicit carrier-space data; `gStore` performs I/O against persistent `CStore/`, `RStore/`, and `XStore/` directories. Full CGTs may be purged to references and reloaded. Persistent files and demand-grown outer-multiplicity bases are part of a coherent database. This is far more than ordinary memoization and is not an appropriate default for TeNeT.

### Decision on the proposed four levels

1. **Categorical transform cache:** do not add now. Racah/provider caches F/R/CGC authority, and TeNeT's categorical group plan is compile-local. Add only if cross-layout repeated planning is measured to dominate and the key can completely identify gauge/provider/conventions.
2. **Structural plan cache:** merge with the current completed runtime cache in the default design. A second retained plan would duplicate keys and object lifetime.
3. **Layout-bound plan cache:** this is the existing completed transformer cache; keep it bounded.
4. **Backend execution cache:** begin as an execution-context-local/device-local sidecar keyed by completed-plan identity, dtype, device and backend version. Do not make it global until device lowering is stable and reuse is measured.

Do not cache cheap scalar R values independently, one-shot operation metadata, tensor-value-dependent metadata, workspace, destination/result buffers, rapidly changing layouts, GPU stream-local state, or objects cheaper to reconstruct than hash and synchronize.

## 9. Current execution architecture

Cold/warm/hot staging is already visible:

```text
provider/FusionTree/F/R reasoning       cold
  -> categorical group plan            cold
  -> layout binding + replay schedule   cold/warm, cached
  -> workspace sizing/reuse             warm
  -> strided replay / GEMM batches      hot
```

Warm scalar blocks use baked strided copy/axpby descriptors. Matrix-valued recoupling uses pack, batched/grouped GEMM through `DenseExecutor`, and scatter. Plan-time run partitioning avoids recomputing GEMM group boundaries on every replay.

The inspected warm loop does not allocate FusionTrees, evaluate F/R, search sectors, canonicalize trees, or select providers. Some temporary-vector and allocation costs remain workspace/API dependent and should be measured with the existing operation-matrix harness.

## 10. What is already well designed

- **Explicit layout-independent seam:** `TreeTransformGroupPlan` states and enforces the categorical/layout boundary (`transform_plan.rs:642-650`).
- **Immutable completed descriptor:** derived plans cannot become stale through public mutation (`transform_structure.rs:63-95`).
- **Scalar/matrix separation:** scalar groups stay compact; matrix-valued groups use shared contiguous coefficients and GEMM-oriented replay.
- **Finite hot path:** copy/axpby and pack/GEMM/scatter, rather than provider callbacks.
- **Precomputed safety/scheduling facts:** destination disjointness, overwrite coverage, GEMM jobs and run partitions are compiled once (`transform_structure.rs:249-301`).
- **Bounded completed cache:** byte and entry budgets, per-entry admission, statistics, generation-safe publication, and immutable `Arc` values (`tree_transform/cache.rs:98-155`).
- **Correct cache ownership:** runtime-owned rather than a hidden process-global completed-operation cache.
- **Honest backend comment:** the host-slice trait is marked legacy/current and is not presented as the future device contract (`replay_backend.rs:22-28,172-177`).
- **CUDA preflight rejects unsupported semantics:** network execution fails rather than silently staging/permuting on host (`network.rs:1563-1605`).
- **Dynamic rank is value-level:** no unbounded family of rank-specific public types.

## 11. Problems and design smells

### [P1 at reviewed revision; resolved by #1079] Checked-Generic contraction bypasses Runtime execution-resource reuse

**Location:** `tenet/src/typed.rs:7076-7110`; `tenet-tensors/src/contract/checked_generic.rs:318-333,457-556`.

**Current:** the public checked-Generic contraction calls the standalone checked path instead of leasing the Runtime Generic execution lane. The path creates fresh transform backend/workspace state, recompiles operand transforms, HomSpaces, axis/block plans and output transform, and later constructs another transform backend plus fusion/kernel workspaces.

**Problem:** reusable execution resources and the explicit compile/replay boundary already exist elsewhere, but this reachable path does not use them. Checked-Generic host network steps inherit the same behavior.

**Why it matters / hot path:** issue #1009 historically measured representative SU(3)/SU(4) whole contractions at roughly 1.5 ms and 17k–19k caller allocations, but explicitly did not identify the dominant subphase and predates this reviewed SHA. Source proves repeated resource/artifact construction; its fraction of the measured gap remains unproved and must be remeasured.

**CPU:** material allocation/setup overhead. **GPU/multi-GPU:** not directly a CUDA defect, but leaving two ownership models would complicate later common lowering. **API/maintenance:** the standalone path duplicates the Runtime resource policy.

**TensorKit/QSpace comparison:** TensorKit reuses completed transformers; QSpace reuses carrier-algebra coefficients and derived X-symbols but rebuilds per-operation block matching and reduced contraction traversal. Neither implies a hidden global numerical-result cache.

**Recommended:** first route the existing checked compiler/executor through a Runtime-leased Generic backend/workspace. Then restore an explicit caller/network-owned prepare-and-replay handle only if current measurements justify compile-once reuse. The current `PreparedTensorContractFusion` is multiplicity-free-bound (`tenet-tensors/src/contract/context.rs:1752-1871`), and closed #601 previously owned Generic prepared replay; this is a measured regression/history question, not authorization for a fresh parallel hierarchy. Preserve eager semantics for ordinary `contract` and exact provider/layout witnesses.

**Keep current:** acceptable only if current phase measurements show that this repeated construction is negligible; otherwise reuse the existing Runtime lane. **Size:** medium.

### [P1] Host-slice tree-transform contract has no placement-aware successor

**Location:** `tenet-operations/src/replay_backend.rs:22-79,172-177`; `storage_scratch.rs`.

**Current:** completed structures are reusable, but the execution trait requires host-accessible tensors or raw host slices.

**Problem:** a CUDA/MPI backend cannot implement tree transforms over resident storage through this interface without host staging or a parallel API.

**Why it matters / hot path:** it blocks device-resident permutation, braid, repartition, pack and scatter. It would force synchronization/transfers around otherwise finite numerical work.

**CPU:** no regression today. **GPU/multi-GPU:** major blocker for asynchronous streams and resident networks. **API/maintenance:** adding device methods to the host trait would entangle placement with semantics.

**Comparison:** TensorKit does not solve Rust device placement; QSpace's principle supports a late numerical executor but its MEX implementation is not a reusable GPU design.

**Recommended:** add a narrow placement-aware replay/lowering contract that consumes `TreeTransformStructure` and backend-owned buffers/workspace. Keep the host trait intact during migration. Do not re-run F/R/provider logic.

**Keep current:** reasonable while CUDA supports direct GEMM only, but not before claiming general CUDA symmetric operations. **Size:** medium.

### [P1] CUDA network execution is direct-contraction-only

**Location:** `tenet-network/src/network.rs:1563-1605`.

**Current:** any non-direct operand axes, result permutation, or final permutation makes CUDA preflight reject the plan.

**Problem:** realistic symmetric contraction paths need layout transforms; the network planner cannot lower them to device tasks.

**Hot path:** unsupported rather than slow, which is correct but incomplete. **CPU:** none. **GPU/multi-GPU:** prevents general networks and future task scheduling. **API:** capability is explicit and safe.

**Recommended:** after the placement-aware transform seam, lower required permutations into the same device task stream; preserve preflight atomicity. **Keep current:** correct until device transforms exist. **Size:** large, staged.

### [P1] CUDA runtime ownership is not yet a multi-device scheduler

**Location:** typed runtime/CUDA context and operation entry points in `tenet/src/typed.rs`; `tenet-operations/src/cuda.rs`.

**Current:** CUDA typed operations take the coarse `RuntimeState` mutex and hold it across allocation and execution (`tenet/src/typed.rs:11349-11384`); CUDA does not use the Host execution-lane lease model. There is no explicit device/stream/event task scheduler.

**Problem:** a single mutable execution authority is adequate for correctness but cannot express concurrent streams, per-device workspaces, transfers, collectives, or multi-GPU placement.

**Hot path:** confirmed coarse lock scope can serialize callers. Direct contraction also creates and uploads a Host `Vec<f64>` of zeros for each destination while holding this lock (`typed.rs:11364-11366`), already tracked by #740. **CPU:** none. **GPU:** can limit overlap and adds a host allocation/upload. **Maintenance:** prematurely adding distributed scheduling now would be over-engineering.

**Recommended:** first measure lock/lease scope and single-device concurrency; then introduce device-local execution leases and scheduler-owned stream/event/workspace resources. **Keep current:** appropriate for the present single-device baseline. **Size:** medium then large.

### [P2] `TreeTransformStructure` combines backend-neutral replay facts and a host schedule

**Location:** `tenet-operations/src/transform_structure.rs:82-95,249-301`.

**Current:** layout, categorical payload, GEMM jobs, pack/scatter descriptors and `TreeTransformParallelSchedule` share one immutable object.

**Problem:** the host schedule may be irrelevant to CUDA, and the type name does not communicate layout binding/execution readiness.

**Hot path:** no demonstrated cost. **GPU:** can lead to unused metadata or pressure to make a shared object backend-specific. **Maintenance:** conceptual ambiguity.

**Recommended:** do not split yet. When implementing the second backend, extract only truly backend-specific scheduling into an execution-context sidecar; retain backend-neutral shapes, offsets, disjointness and GEMM jobs. Rename only as part of that migration. **Keep current:** preferred until the seam is exercised. **Size:** small/medium.

### [P2] No benchmark gate covers the full cold-to-device lowering matrix

**Location:** existing `benchmarks/operation_matrix.*`, issue-1007 audit and operation benches.

**Current:** useful cold/warm CPU evidence exists, but not the requested ranks/symmetry/block regimes nor device task/launch metrics.

**Problem:** cache splitting, specialized-rank kernels, grouped GEMM and execution-IR changes cannot be selected rationally.

**Recommended:** extend the existing harness before introducing new caches or major IR layers. **Keep current:** risks architecture by anecdote. **Size:** medium.

### [P2] Cross-layout categorical-plan reuse is intentionally not retained

**Location:** compile-local `TreeTransformGroupPlan`; completed runtime cache.

**Current:** distinct layouts can miss the completed cache and rebuild the same categorical transform.

**Problem:** this could matter for expensive generic multiplicity transforms across many degeneracy layouts, but existing evidence does not establish it.

**Recommended:** instrument F/R calls and categorical/binding time separately. Add a bounded categorical cache only if `construction cost × reuse` exceeds lookup, synchronization, memory, and key complexity. **Keep current:** preferred. **Size:** potentially medium.

### [P3] Public/internal names do not expose the complete staging model

**Location:** `TreeTransformStructure`, legacy `TreeTransformBackend` naming.

**Current:** comments are clearer than type names.

**Recommended:** documentation first; rename only with the device seam. This is not an architecture defect. **Size:** small.

No evidence was found for these requested anti-patterns in warm replay: F/R evaluation, tree canonicalization, sector search, tensor-value cache keys, global workspace caching, dense matrix allocation for scalar transforms, or provider/device state embedded in the categorical plan. `Arc` is used mainly for immutable heavy canonical/completed objects and tensor bodies, not per-contribution dynamic dispatch. Arc-clone and cache-lock costs remain benchmark questions.

## 12. Recommended data structures

The migration-minimal hierarchy is:

```rust
// Existing value-level semantic descriptor
TreeTransformOperation

// Existing immutable categorical plan; compile-local unless reuse is proven
TreeTransformGroupPlan<Coeff>

// Existing immutable layout-bound replay plan
TreeTransformStructure<Coeff>

// New backend-local sidecar, only when required
BackendExecutionPlan<BackendKey>

// Existing/new flat POD-like work descriptors
KernelTask / DenseGemmBatchJob / StridedReplay

// Operation/request-local reusable mutable storage
Workspace<Placement>
```

Ownership rules:

- Use cheap copied IDs (`RuleIdentity`, block/layout indices, plan identity) in keys/tasks.
- Use `Arc` for immutable, heavy, shared canonical structures and completed plans returned from caches. Do not put every child behind another `Arc`; vectors should own compact descriptors contiguously.
- Keep the categorical group plan value-owned during compilation. Its matrix payload may be shared when binding multiple compatible layouts, as current code does.
- Keep workspace uniquely mutable and execution-context/request-local. Never include it in cached immutable plans.
- Keep device buffers, streams, events, library handles, and allocator state backend/device local.
- Generic parameters should cover coefficient/scalar and backend entry points. Runtime rank and task counts remain slices/vectors.
- Serialization should target stable semantic/structural descriptors only after provider/gauge/schema identity is specified. Do not serialize pointers, `Arc` identity, streams, or opaque backend handles.

## 13. Recommended cache hierarchy

```text
Provider/Racah bounded coefficient data
    F/R/CGC/decomposition authority

tenet-core canonical structure caches
    weak/bounded HomSpace and FusionTree layout identities

Runtime completed-transform cache
    key: provider identity + operation + exact src/dst structure + convention
    value: immutable layout-bound replay plan

ExecutionContext backend lowering sidecar
    key: completed-plan ID + dtype + backend/device architecture
    value: grouping/workspace layout/backend metadata only
    initially operation/context local; retain only if reuse is measured
```

Admission criteria for every new cache: measured construction time, expected reuse distribution, complete identity/gauge key, bounded bytes/entries, no partial publication on failure, concurrency semantics, and telemetry. A cache hit must be cheaper than recomputation in the target regime.

## 14. Recommended execution IR

Use an internal tagged union or structure-of-arrays task stream only where it reduces duplicate backend logic. Minimal semantics:

| Primitive | CPU | GPU | rank | aliasing/workspace |
|---|---|---|---|---|
| `FillScale` | vector/strided loop | fill/scale kernel | N-D view | in-place allowed under explicit mode; no workspace |
| `CopyAxpby` | contiguous or baked strided loop | copy/permute-axpby kernel | generic N-D; specialize measured ranks | source/destination overlap declared; no hidden allocation |
| `Pack` | strided gather | gather/permute kernel | generic N-D | disjoint packed destination; backend workspace |
| `GemmBatch` | BLAS, serial/batched/grouped routing | cuBLAS grouped/strided/batched | matrices after lowering | destinations disjoint per batch or explicit reduction target |
| `ScatterAxpby` | strided scatter/accumulate | scatter kernel | generic N-D | contribution groups carry disjointness/atomic/reduction policy |
| `Reduce` | deterministic tree/chunk reduction | staged reduction | scalar/tile | explicit workspace and determinism mode |

`Scale`, `Axpy`, `PermuteCopy`, and `StridedCopy` need not be separate semantic variants if one descriptor plus flags produces better locality. Conversely, do not force scalar copy through the GEMM matrix path. `BatchedGemm` and `GroupedGemm` are backend routing decisions over `GemmBatch` jobs.

The executor must not reference a symmetry provider, FusionTree, F-symbol, or R-symbol. It may reference stable plan/layout IDs for validation and profiling, never for semantic interpretation.

## 15. CPU/GPU backend boundary

The shared boundary should include logical regions, shapes, strides, coefficients, contribution groups, overwrite/alias proofs, and workspace size/alignment. It should exclude pointers, streams, events, allocator objects, thread pools, device IDs, and kernel handles.

Backend lowering may legitimately choose backend-aware layouts or pack strategies. That choice belongs after the shared layout-bound descriptor and may return a backend sidecar. If a GPU-specific tensor layout becomes beneficial, record it as a placement/layout policy and re-bind explicitly; do not inject CUDA fields into categorical objects.

Many-small regime:

- coalesce equal-shape tasks;
- use grouped/strided batched GEMM;
- fuse scale with copy/permute/scatter;
- minimize launches and pointer-array construction;
- prebuild compact device task arrays.

Few-large regime:

- prefer direct GEMM and avoid pack/scatter;
- reuse workspace;
- overlap independent GEMMs and transfers with streams;
- avoid host synchronization between tasks.

Multi-GPU scheduling consumes already-lowered contribution groups, estimates compute/transfer cost, assigns devices, inserts transfers/collectives, and owns events. It must not repeat sector or F/R reasoning.

## 16. TensorKit: adopt and avoid

Adopt:

- FusionTree-based semantic basis;
- explicit fusion-style distinctions and multiplicity-aware F/R recoupling;
- reduced tensor representation;
- completed reusable transformer metadata;
- separate `UniqueFusion` scalar and multiple-tree matrix recoupling paths;
- source/destination shape/stride/offset resolution before numerical replay.

Do not copy literally:

- assumptions that Julia JIT and object dispatch will specialize hot calls;
- process-global cache defaults without Rust workload measurements;
- Julia allocation/object layouts;
- exact type names (`GenericTreeTransformer` does not mean GenericFusion);
- a layout-dependent transformer as proof that TeNeT should eliminate its existing layout-independent compile seam.

## 17. QSpace: adopt and avoid

Adopt as execution principles:

- remove expensive carrier/structural algebra from repeated reduced numerical contraction;
- retain small operation-specific structural transformations when their reuse is real;
- treat structural identity, gauge and version as correctness data;
- allow large structural data to have a different lifetime from tensor payload;
- perform numerical block work only after structural matching/lowering.

Avoid:

- explicit CGTs as a second semantic representation beside FusionTree/F/R;
- CGR/X-symbol-centric tensor semantics;
- process-global mutable `gCS`/`gRS`/`gXS`/`gStore` ownership;
- persistent `RStore/`/`CStore/`/`XStore/` and `RC_STORE` filesystem coupling;
- demand-grown outer-multiplicity bases and cross-job store synchronization;
- assuming QSpace has a fully detached operation-plan replay path—it still performs warm structural checks and matching.

An “X-like” TeNeT object, if ever useful, should be a deterministic small transform derived from provider identity plus FusionTree/F/R conventions, not a projection between explicit CGT databases.

## 18. Recommended target architecture

```text
 SymmetryProvider / Racah authority                         immutable shared
  sectors, fusion, duality, F/R/CGC     <bounded coefficient/cache boundary>
                 |
                 v
 FusionTree keys + canonical HomSpace structures             value IDs / Arc heavy tables
                 |
                 v
 TreeTransformOperation -> TreeTransformGroupPlan<C>          cold, value-owned
                 |                 <do not cache unless measured>
                 v
 Layout binder (degeneracies, block layouts, convention)
                 |
                 v
 TreeTransformStructure<C>                                    immutable Arc
  offsets, strides, coefficients, contribution groups,
  pack/scatter descriptors, GEMM jobs, alias proofs
                 |                 <bounded Runtime cache>
                 v
 placement-aware execution lowering
                 |
         +-------+----------------+
         |                        |
         v                        v
 Host task view              Device task view                 context-local
 thread partition           launch/group/workspace layout
         |                        |
         v                        v
 CPU executor               GPU executor                      mutable backend resources
 host workspace             streams/events/device workspace
         +-----------+------------+
                     v
              Runtime scheduler
         dependencies, slicing, placement,
         transfer/collective overlap, telemetry
```

The diagram is intentionally not seven mandatory cached Rust structs. The existing completed structure can serve both layout-bound plan and backend-neutral execution description until a second backend demonstrates a useful split.

For the backend half of this boundary, [tenferro-rs at the reviewed revision](https://github.com/tensor4all/tenferro-rs/tree/ec9b06025774574a5df47e11ccdd730fb0c01c80) is the primary implementation reference. Tenferro already owns private finite execution staging, placement-bound scheduling, explicit transfer and event domains, and bounded runtime/backend caches. TeNeT should therefore stop at symmetry-aware structural/layout lowering where possible and evaluate a supported Tenferro backend or extension boundary first. If that boundary cannot express efficient copy/pack/GEMM/scatter replay, the result should motivate a narrow upstream seam or retain a TeNeT-local compact view. TeNeT should not copy Tenferro's crate-private `ExecProgram` or `ScheduledGraph`, nor grow a second generic device runtime.

## 19. Migration roadmap

### Phase 1 — establish measurement and invariants

- **Prerequisite:** current CPU operation matrix and CUDA direct baseline.
- **Changes:** add phase timers/counters for categorical compile, layout bind, cache lookup, workspace preparation and replay; count F/R calls, hashes and allocations.
- **Benefit:** evidence for every subsequent split/cache/specialization.
- **Bench/test:** Section 20 matrix; preserve semantic cross-provider and complex-coefficient tests.
- **Risk:** low.

### Phase 2 — placement-aware tree-transform execution seam

- **Prerequisite:** immutable `TreeTransformStructure` and host replay tests.
- **Changes:** define backend/storage placement contract; expose compact read-only task/layout views; keep old host trait as adapter.
- **Benefit:** device/MPI replay without host slices or symmetry re-planning.
- **Bench/test:** identical CPU output through adapter; failure atomicity; no provider calls during replay.
- **Risk:** medium; avoid lifetime-heavy borrowed object graphs.

### Phase 3 — CUDA strided copy/pack/scatter and GEMM lowering

- **Prerequisite:** Phase 2 and stable device storage.
- **Changes:** backend sidecar, device task buffer, workspace sizing, generic N-D permutation; optional small-rank kernels only after evidence.
- **Benefit:** resident permutation/braid/repartition.
- **Bench/test:** host/device parity for the current f64 capability, alias cases, many-small launch counts. Complex coefficients require a separate complex device-storage prerequisite; current `CudaStorage` is f64-only.
- **Risk:** medium/high.

### Phase 4 — CUDA network plan admission

- **Prerequisite:** device transforms and direct contraction.
- **Changes:** lower required operand/result/final permutations into the device task DAG; preserve all-or-nothing preflight.
- **Benefit:** general network schedules rather than direct-only paths.
- **Bench/test:** multi-step networks, sliced execution, deterministic output assembly, zero host transfers during resident execution.
- **Risk:** high.

### Phase 5 — allocation-light CPU executor refinements

- **Prerequisite:** benchmark evidence.
- **Changes:** specialize only dominant strided patterns; improve grouped small-block routing; pre-size/reuse task metadata.
- **Benefit:** lower overhead in many-small blocks without changing semantics.
- **Bench/test:** allocation count and warm replay regression thresholds.
- **Risk:** low/medium.

### Phase 6 — asynchronous single-GPU scheduler

- **Prerequisite:** device task DAG and telemetry.
- **Changes:** device-local execution leases, stream/event scheduling, reusable workspace pool, overlap policy.
- **Benefit:** concurrency and transfer/compute overlap.
- **Bench/test:** concurrency, kernel gaps, synchronization count, deterministic modes.
- **Risk:** high.

### Phase 7 — multi-GPU placement

- **Prerequisite:** stable single-device task and scheduler model.
- **Changes:** device placement, transfer/collective tasks, cost model, per-device workspace budgets.
- **Benefit:** scale large contribution graphs.
- **Bench/test:** weak/strong scaling, communication overlap, memory balance, reproducibility.
- **Risk:** high; do not start from categorical code.

## 20. Benchmark plan

Use a factorial matrix, with a smaller mandatory CI subset and a full scheduled suite:

- rank: 2, 4, 8, 16;
- symmetry: Abelian, SU(2)-like multiplicity-free, generic multiplicity, braided/complex coefficients;
- block distribution: many-small, mixed, few-large;
- operations: permute, braid, repartition, contraction, repeated identical operation, cold distinct operation.

For each case report separately:

1. provider/category query and F/R call counts;
2. categorical-plan construction time;
3. layout binding/completed-plan construction time;
4. cache lookup time and hit/miss/eviction/admission status;
5. execution-lowering time;
6. numerical replay time;
7. allocation count and bytes by phase;
8. number and bytes of task descriptors/workspace;
9. hash lookups and cache-lock wait where observable;
10. CPU utilization and parallel efficiency;
11. GPU kernel/launch count, achieved occupancy, memory throughput, GEMM efficiency, stream idle gaps and synchronization count;
12. host-device transfer bytes and count.

Run at least four modes:

```text
cold process/provider + cold plan
warm provider/category + cold layout
warm completed plan + fresh workspace
warm completed plan + reused workspace
```

Compare:

- cache off vs current bounded completed cache;
- generic N-D vs any proposed rank-specialized kernel;
- scalar fast path vs matrix path;
- direct large GEMM vs packed/grouped small-block path;
- one thread vs configured CPU threads;
- one GPU stream vs proposed multi-stream scheduling.

Use medians plus tail latency and memory high-water mark. Reject a new cache if it improves synthetic planning but not end-to-end repeated workloads, or if its lookup/synchronization dominates small operations.

## 21. Top 10 concrete changes

Items are ordered. Existing issues should be extended rather than duplicated where scopes match.

1. **Host checked-Generic contraction: reuse Runtime execution resources — completed by [#1079](https://github.com/Ryo-wtnb11/TeNeT/pull/1079).** The checked path now uses the Generic lane; remeasurement did not justify an implicit ordinary-operation cache.
2. **Benchmark: cold/warm symmetric lowering matrix and phase counters.** Extend the existing operation-matrix harness with ranks, symmetry regimes, block distributions, F/R/hash/allocation counters and CUDA task metrics.
3. **CUDA: placement-aware tree-transform replay contract ([#1080](https://github.com/Ryo-wtnb11/TeNeT/issues/1080)).** Add the narrow successor anticipated by `replay_backend.rs`, consuming existing completed structures without host slices. Evaluate a supported Tenferro boundary first; otherwise motivate a narrow upstream seam or retain a TeNeT-local compact view.
4. **CUDA: device strided copy/axpby, pack and scatter tasks.** Preserve scalar fast paths and generic runtime rank.
5. **CUDA: decide backend-lowering retention ownership ([#1082](https://github.com/Ryo-wtnb11/TeNeT/issues/1082)).** Begin operation/context-local; first determine whether retention belongs to Tenferro's prepared/backend cache owners. Add a TeNeT sidecar only for metadata proven to remain TeNeT-specific and reusable.
6. **CUDA network: lower non-direct/result/final permutations.** Replace direct-only admission incrementally while keeping preflight atomicity.
7. **Runtime: measure and narrow CUDA lease/lock scope.** Introduce device-local execution leases only after the measurement demonstrates serialization.
8. **CUDA scheduler: streams/events/workspace ownership ([#1083](https://github.com/Ryo-wtnb11/TeNeT/issues/1083)).** Keep scheduling state out of structural plans and avoid duplicating Tenferro's execution-endpoint, transfer and event-domain runtime.
9. **CPU: benchmark-gated many-small grouping and documentation.** Evaluate grouped GEMM/fused replay and document the cold/warm/hot ownership boundary.
10. **Conditional only: cross-layout categorical transform cache.** Open implementation work only if Phase 1 proves repeated categorical compilation dominates and a complete deterministic gauge/provider key is available.

### Final judgment

TeNeT **does maintain a runtime-rank symmetric tensor API while lowering FusionTree/F/R semantics into reusable finite numerical work on CPU**. The arbitrary-rank frontend does not remain as arbitrary-rank categorical interpretation in the inspected hot replay path. With the CPU-first #1077 remeasurement/fix completed by #1079, the next CUDA-enabling step is to expose the already-lowered work through a placement-aware execution boundary. That is preferable to a categorical redesign or a new hierarchy of global caches.

Issue disposition from this audit:

- Completed Host leaf: [#1077](https://github.com/Ryo-wtnb11/TeNeT/issues/1077), checked-Generic Runtime resource reuse, implemented by [#1079](https://github.com/Ryo-wtnb11/TeNeT/pull/1079). Source inspection proves resource reuse; measurement did not show an allocation reduction, so no broader operation cache was added.
- CUDA umbrella and first implementation gate: [#3](https://github.com/Ryo-wtnb11/TeNeT/issues/3) and [#1080](https://github.com/Ryo-wtnb11/TeNeT/issues/1080), placement-aware tree-transform replay with Tenferro integration, a narrow upstream seam, and a TeNeT-local compact view retained as alternatives pending #1084.
- Investigation gates, not implementation commitments:
  - [#1081](https://github.com/Ryo-wtnb11/TeNeT/issues/1081), rank-specialized kernels versus generic N-D replay; generic dense execution should use Tenferro's backend contract, while reusable low-level kernel work belongs upstream in Tenferro or its kernel provider rather than TeNeT.
  - [#1082](https://github.com/Ryo-wtnb11/TeNeT/issues/1082), retained backend-lowering metadata and cache ownership; duplicate TeNeT retention is the default rejection.
  - [#1083](https://github.com/Ryo-wtnb11/TeNeT/issues/1083), TeNeT/Tenferro multi-stream and multi-device boundary.
  - [#1084](https://github.com/Ryo-wtnb11/TeNeT/issues/1084), whether any common TeNeT execution IR is needed now that Tenferro already owns private finite execution staging and scheduling.
- No issue was opened for transform-cache sharding or a separate categorical cache; both still lack evidence that an additional cache would beat current completed-structure reuse.
