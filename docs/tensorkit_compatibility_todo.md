# TensorKit Compatibility TODO

This file records TensorKit ecosystem behavior that TeNeT intentionally mirrors
or leaves unsupported until the categorical semantics are fixed. These items are
not optimization notes; they are semantic boundaries for the TensorKit-compatible
baseline.

## Storage-centric backend dispatch

TensorKit ecosystem behavior:

- `TensorMap` owns dense block storage as a storage type parameter. CPU tensors
  use host arrays and CUDA tensors use `CuArray` storage.
- `TreeTransformer` objects cache categorical block structure, recoupling
  matrices, and packed strided structures. They are not CPU/GPU backend objects.
- Tree-transform replay allocates temporary buffers with `similar(tdst.data,
  sz)` / `similar(tsrc.data, sz)`, so workspace placement follows tensor
  storage.
- Dense replay calls `StridedView`/`mul!`/`tensoradd!`; Julia dispatch then
  routes CPU storage to BLAS/Strided and GPU storage to CUDA/cuBLAS/cuTENSOR
  extensions.
- TensorOperations explicit backend/allocator arguments remain escape hatches,
  but ordinary tensor operations infer the execution path from the array
  storage and allocator.

TeNeT baseline:

- TensorMap-level code must not expose concrete runtimes such as strided-rs,
  MKL, cuBLAS, CUDA driver handles, or tenferro tensor types.
- Fusion-tree enumeration, sector matching, tree-pair recoupling, dense-axis
  permutations, and block replay descriptors remain TeNeT categorical data.
- Dense execution is selected from storage placement. Host storage currently
  uses the host dense executor; future CUDA storage must use device workspaces
  and device kernels without host `as_slice()` fallbacks.
- Low-level implementations such as strided-rs, tenferro, C++ BLAS, CUDA C++,
  cuBLAS, or cuTENSOR are private implementation details below the storage
  dispatch boundary.

Current TeNeT status:

- `tenet-dense` exposes host `DenseView` / `DenseViewMut` with
  `DensePlacement::Host`.
- `DefaultDenseExecutor` is no longer a public generic over a kernel adapter.
  The strided host matmul path is hidden behind the default host executor.
- Cargo features now forward CPU provider selection consistently to both
  tenferro and strided-rs:
  - `cpu-faer`
  - `cpu-blas`
  - `blas-accelerate`
  - `blas-openblas`
  - `blas-mkl`
  - `provider-inject`

Next required refactor:

- Generalize `TreeTransformBackend` and `TensorContractBackend` over storage
  types instead of assuming host `Vec<T>`.
- Make public non-`*_with` facades dispatch from destination/source placement.
  Keep `*_with` APIs as explicit testing and override hooks.
- Make tree-transform and contraction workspaces storage-aware, mirroring
  TensorKit's `similar(tdst.data, ...)` behavior.
- Add CUDA storage only after host storage-generic tests pass. A `cuda` Cargo
  feature must not be exposed until tree replay and contraction actually use
  device-resident workspaces and kernels.

Known temporary mismatch:

- Until the raw strided-rs API is merged upstream and tenferro-rs is updated to
  the same strided-rs source/revision, TeNeT can pull two `strided-einsum2`
  crates: one through tenferro-rs and one through TeNeT's raw-kernel adapter.
  This is a dependency hygiene issue, not a semantic difference; remove it by
  converging on the upstream raw API revision.

## AdjointTensorMap with explicit braid

TensorKit source:

- `/Users/ryowatanabe/.julia/packages/TensorKit/6Camk/src/tensors/indexmanipulations.jl:163`
- `# TODO: braid for AdjointTensorMap; think about how to map the levels argument.`

TeNeT status:

- TeNeT already models explicit braid as `TreeTransformOperationKey::Braid`
  with axis permutations plus separate codomain/domain levels. The unresolved
  boundary is only lazy source adjoint lowering for that explicit braid.
- Ordinary lazy-adjoint lowering for `tensoradd`, `tensortrace`, and
  `tensorcontract` is implemented by swapping the categorical source space,
  remapping axes with `adjoint_tensor_axis`, and carrying storage conjugation
  separately.
- For symmetric braiding, TensorKit's `braid` path short-circuits to
  adjoint-aware `permute`, so this TODO does not block Z2, FermionParity, or
  ordinary symmetric tensoradd/contract fixtures.
- TensorKit leaves this undefined. TeNeT defines it as an extension; it is not
  part of the current TensorKit-compatible baseline.
- TeNeT implements this as a gated extension. Public fusion tensoradd only
  enables `Braid + source_conjugate` when the rule reports
  `supports_unitary_braid_dagger()`. Rules without that capability still return
  `UnsupportedTreeTransformScope`.

TeNeT extension definition:

- This extension is only mathematically valid for dagger-compatible braided
  fusion categories where the braiding is unitary:
  `c_{a,b}^\dagger = c_{a,b}^{-1}`. If a rule cannot provide this semantics,
  `Braid + source_conjugate` must stay unsupported.
- A lazy source adjoint first replaces the source tensor map by the categorical
  adjoint space view, including the codomain/domain swap. TeNeT's existing
  0-based adjoint axis map is used:
  `axis < NOUT ? NIN + axis : axis - NOUT`.
- Braid levels are attached to source strands, not to output tuple positions.
  `TreeTransformOperationKey::Braid` stores them split by source
  codomain/domain axes, matching TensorKit's `add_braid!` split by
  `codomainind(tsrc)` and `domainind(tsrc)`. Lowering remaps those split source
  levels with `adjoint_tensor_axis` and carries each strand level with it.
- The adjoint of a non-symmetric braid reverses the crossing direction. In
  TeNeT's level encoding this is represented by reversing the relative level
  order:
  `level' = min_level + max_level - level`.
  This turns every TensorKit-style `levels[i] < levels[j]` crossing into the
  inverse `levels'[i] > levels'[j]` crossing, and conversely.
- TeNeT core already represents double-tree braid levels as
  `codomain_levels ++ reverse(domain_levels)`, matching TensorKit's
  `fsbraid` construction. The extension must preserve that representation
  after adjoint-axis remapping and level-order reversal.

Implemented test coverage:

- `tenet-core` verifies that reflecting levels with
  `level' = min_level + max_level - level` selects the inverse Artin branch for
  a direction-sensitive anyonic rule.
- `tenet-operations` verifies source-strand level remapping separately from
  output tuple positions.
- `tenet-operations` validates bad braid axes, level count mismatch, and
  duplicate levels at lowering time.
- Public `tensoradd_fusion_into` rejects `Braid + source_conjugate` when the
  rule does not declare unitary dagger-compatible braiding.
- Supported unitary phase anyonic fixtures compare public
  `source_conjugate + explicit Braid` against the explicit categorical
  sequence: adjoint-space view, inverse-level braid, then storage conjugation.
  Tensoradd covers codomain-only, domain-only, and mixed codomain/domain
  double-tree maps.

Remaining coverage to add before broadening beyond tensoradd:

- Add equivalent codomain-only, domain-only, and mixed codomain/domain
  double-tree coverage when other public operations expose explicit braid with
  source conjugation.
- Include nonscalar SU2 recoupling and nontrivial degeneracy shapes once that
  operation path is enabled beyond tensoradd.
- Keep symmetric `Permute` behavior separate from explicit `Braid`; symmetric
  braid short-circuiting must not hide mistakes in non-symmetric braid lowering.

## Fermionic tensor trace vs ordinary trace

TensorKit source:

- `TensorOperations.@tensor` lowers repeated indices to `tensortrace!`.
- TensorKit `tensortrace!` routes to `trace_permute!`, which applies categorical
  twists for fermionic sectors.
- `LinearAlgebra.tr(::TensorMap)` is not the same operation; it sums block traces
  with dimension factors and does not represent the `@tensor` supertrace path.

TeNeT status:

- Plain dense `tensortrace` remains dense/TensorOperations-like.
- Fusion `tensortrace_fusion_*` implements categorical trace factors and
  fermionic supertrace behavior.
- Plain `tensortrace` rejects fusion-tree tensors so categorical trace is not
  silently replaced by dense trace.

## TensorOperations categorical lowering coverage

TensorKit source:

- TensorOperations exposes `tensorcontract!(C, A, pA, conjA, B, pB, conjB,
  pAB, ...)`; `conjA` and `conjB` are independent API states.
- TensorKit lowers `conjA`/`conjB` lazily by replacing the selected source with
  `A'`/`B'` and remapping axes with `adjointtensorindices`.
- For rank `(1, 1)`, TensorKit's 1-based adjoint axis map swaps `1 <-> 2`.
  TeNeT's 0-based map swaps `0 <-> 1`.

TeNeT fixed coverage:

- Dense `tensoradd`, `tensortrace`, and `tensorcontract` keep TensorOperations
  dense semantics and reject fusion-tree tensors instead of silently using the
  wrong categorical operation.
- Fusion `tensoradd` has lazy source-adjoint lowering and context replay tests.
- Fusion `tensortrace` has fermionic supertrace, degeneracy diagonal, lazy
  adjoint, beta-once, and SU2 quantum-dimension coverage.
- Fusion `tensortrace` rejects nonsymmetric braiding for ordinary trace
  semantics, matching TensorKit/TensorOperations `tensortrace!` rather than
  treating an anyonic braid as a symmetric permute.
- Fusion `tensortrace` has TensorKit oracle coverage for a non-scalar
  FermionParity output trace with nonadjacent traced axes, and for a two-pair
  FermionParity supertrace. The oracle script is
  `tenet-operations/benchmarks/tensorkit_tensortrace_oracles.jl`.
- Fusion `tensorcontract` has lazy `lhs_conj`, `rhs_conj`, and `both_conj`
  coverage for scalar fusion blocks.
- Fusion `tensorcontract` has Z2 and FermionParity 2x2 degeneracy block
  coverage for:
  - `lhs_conj`: per sector `C_s = A_s† * B_s`
  - `rhs_conj`: per sector `C_s = A_s * B_s†`
  - `both_conj`: per sector `C_s = A_s† * B_s†`
- FermionParity W <- W no-braiding tests intentionally assert no extra
  fermionic twist for these three matrix-contract fixtures. TensorKit only
  inserts the tensorcontract twist when the effective contracted `B` leg is
  dual.
- Fusion `tensorcontract` has noncanonical SU2 source-transform fixtures with
  lazy `lhs_conj`, `rhs_conj`, and `both_conj`. The one-sided fixtures use
  TensorKit-matching dual-oriented source spaces, e.g. `dual(V)` legs for the
  non-adjoint side when needed. The Rust tests compare explicit-plan execution
  against the literal lowered sequence: source `tensoradd_fusion_into(...,
  source_conjugate=...)` transforms, then canonical contraction.
- The Julia oracle script records the corresponding TensorKit direct
  `tensorcontract!` result for the SU2 fixtures. TeNeT's internal fusion-tree
  block storage is not flat-order identical to TensorKit's `C.data` for that
  multi-tree SU2 case, so the Rust assertion uses the TeNeT explicit reference
  sequence rather than direct flat-data equality.
- Fusion `tensorcontract` has a ProductSector(FermionParity x U1 x SU2)
  component-channel fixture with SU2 recoupling and complex data. This fixture
  has scalar degeneracy blocks, so the Rust data is asserted directly against
  the TensorKit oracle values.

Remaining semantic boundary:

- Extend the Julia TensorKit oracle script
  `tenet-operations/benchmarks/tensorkit_contract_adjoint_oracles.jl` as new
  Z2/FermionParity 2x2 degeneracy fixtures are added.
- Do not silently lower explicit non-symmetric `Braid + source_conjugate` as an
  ordinary `permute`; that is categorically wrong for non-symmetric braiding.
  Keep the TeNeT extension gated by explicit rule capability.

## Storage-centric typed execution boundary

TensorKit source:

- TensorKit tensor objects keep categorical/block structure separate from the
  underlying storage object.
- Transformer/structure construction uses block structures and fusion-tree
  metadata; replay dispatch follows the storage/array backend.
- Low-level raw replay is still an array/slice/device-kernel concern, not a
  categorical API concern.

TeNeT fixed coverage:

- `TensorMap<T, NOUT, NIN, S, D>` already had a storage slot, but several
  `tenet-operations` typed APIs omitted `D` and therefore defaulted to
  `Vec<T>`.
- `TreeTransformBackend`, `TensorOperationsBackend`, and
  `TensorContractBackend` typed entry points now expose the storage slot.
- Current host implementations require `HostReadableStorage` /
  `HostWritableStorage` explicitly. This is intentional: raw replay still uses
  host slices, and TeNeT should not pretend CUDA/device storage is supported
  until device replay/workspace paths exist.
- Structure builders such as `TensorAddStructure::compile`,
  `TensorTraceStructure::compile`, `TensorContractStructure::compile`, and
  `TreeTransformStructure::compile` require only `TensorStorage`, because they
  inspect structure metadata and do not read tensor data.
- Public `*_execute_with` facades for tensoradd, tensortrace,
  tree-transform, and tensorcontract accept non-`Vec` host storage.
- Plain default host facades for tensoradd, tensortrace, and tensorcontract
  also accept non-`Vec` host storage.
- Tree-pair transform plan/cache/context APIs and tensoradd-fusion convenience
  APIs accept non-`Vec` host storage while still compiling transformer
  structures only from categorical/block metadata.
- Fusion contraction convenience, explicit-plan, and dynamic fallback APIs
  accept non-`Vec` host storage. Dynamic scratch and canonical replay remain
  host-backed internally, so the API uses host readable/writable bounds rather
  than pretending to support device storage.
- Fusion tensortrace convenience APIs and `TensorContractFusionExecutionContext`
  methods accept non-`Vec` host storage under the same host-read/write boundary.
- Tests include custom writable/read-only host storage wrappers to assert that
  typed structure compile and replay do not accidentally rely on `Vec<T>` and
  do not require source tensors to be writable.

Remaining implementation boundary:

- Raw replay methods still accept `&[T]` / `&mut [T]` and are host-only.
- Workspaces (`TreeTransformWorkspace`, `TensorContractWorkspace`) still own
  `Vec<T>`. Device/CUDA support requires storage-aware workspaces before any
  CUDA feature should be exposed.
- Host workspace implementations are named explicitly:
  `HostTreeTransformWorkspace`, `HostTensorContractWorkspace`,
  `HostCanonicalFusionBlockContractWorkspace`, and
  `HostDynamicFusionScratchWorkspace`. The old public workspace names remain
  type aliases for source compatibility. Public host workspace types report
  `Placement::Host`; future device/CUDA replay should add separate device
  workspace types rather than hiding device storage behind these host buffers.
- Host scalar strided primitives are isolated in `host_scalar_kernels.rs`.
  Tree/fusion replay should call this boundary for tensoradd, copy-scale,
  axpby, and scale rather than embedding raw strided loops in categorical
  planning code. This mirrors the TensorKit/Strided.jl split and is the
  replacement point for a future C++/CUDA low-level backend.
- Dense matmul has an explicit kernel boundary:
  `DenseKernelBackend` plus `DenseExecutorWithKernel<K>`. The default host
  executor still uses `StridedKernelBackend`, while future BLAS/C++/CUDA
  kernels can replace that layer without touching TensorMap/fusion algorithms.
- Higher-level default convenience functions currently instantiate host
  backends. Once device storage exists, these should dispatch from placement
  instead of exposing backend selection in user-facing APIs.
