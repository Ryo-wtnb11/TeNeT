# TeNeT

TeNeT is a Rust tensor-network library for symmetric tensors. It separates a
generic symmetric-tensor engine from symmetry-specific fusion data. A tensor is
a map `codomain <- domain` stored as fusion-tree-indexed reduced blocks.
Fusion-rule *providers* supply sector labels and categorical data, while the
engine handles block layouts and tensor operations; contraction-path planning
remains separate from execution.

Design priority, in this order: Rust-native maintainability and extensibility;
speed that survives dynamic-rank tensor networks; a usable high-level API.

All crates are at version `0.1.0`. The public API is not stabilized and
expert-layer types still move between crates as the layering settles.

## Architecture

The ordinary API keeps the symmetry provider, payload scalar, and storage type
concrete as `TensorMap<R, D, S>`, while tensor rank and sector content remain
runtime values. Operations dispatch on provider capabilities—fusion
multiplicity, braiding, rigidity, and checked symbol access—rather than on a
central symmetry enum.

Each tensor keeps the exact provider instance used to build it and a compiled
block layout: fusion-tree keys, per-block shapes and strides, and offsets into
one contiguous payload. Checked construction validates the complete structure
before returning a space, tensor, plan, or result. Compiled layouts and
recoupling data can then be reused only with matching categorical data and
gauge conventions. Contraction-plan reuse follows network topology and the
configured replan policy.

`Runtime` owns the bounded caches, workspace pools, and selectable dense
backends used by those operations. Contraction-path planners consume network
metadata and produce validated plans; TeNeT executes the plans locally over its
reduced blocks. This keeps mathematical structure, resource ownership,
planning, and kernel selection separate.

External libraries and papers are cited in function documentation,
[`tenet/references.md`](tenet/references.md), tests, or benchmark reports when
they support a particular convention, divergence, oracle, or comparison. They
do not define TeNeT's public contract.

## Symmetries are providers

`tenet-sectors` defines the provider vocabulary — `FusionRule`,
`CheckedFusionAlgebra`, `SectorCodec`, `MultiplicityFreeRigidSymbols`,
`GenericRigidSymbols`, `RuleIdentity` — and ships `ZNFusionRule` (including
`Z2FusionRule`), `FermionParityFusionRule` (fZ2), `U1FusionRule`,
`CU1FusionRule`, `SU2FusionRule`, `FibonacciFusionRule`, `ProductFusionRule`,
and feature-gated `SUNFusionRule`. Operations select trait capabilities rather
than a provider enum or a named-group dispatch branch.
[`docs/provider_interface.md`](docs/provider_interface.md) is the contract for
writing one.

`FibonacciFusionRule` supplies categorical data but does not implement
`SectorCodec`, so it is not available through the ordinary typed
`GradedSpace` / `TensorMap` API. `tenet-category-data` separately ships
`CategoryDataFibonacci`, which implements `SectorCodec` but has `Complex64`
categorical coefficients; the current multiplicity-free typed root requires
`f64`. SUN is available only with `racah-generated` and uses the checked Generic
path.

A product of providers is itself a provider, so

```text
FermionParityFusionRule.product(U1FusionRule).product(SU2FusionRule)
```

is `(fZ2 ⊠ U(1)) ⊠ SU(2)` — an ordered product of components, recursively
nested, without a central provider enum, dispatch arm or group-specific
constructor. Factor order and association are structure of the Rust type and of
the `ProductSector` label, never an automatic equivalence: `U(1) ⊠ fZ2` and
`fZ2 ⊠ U(1)` are both legal and are different types.

The product's coefficient scalar is promoted from its components', as in
TensorKitSectors, so a component with complex topological data composes with a
real-coefficient group provider: `Fibonacci ⊠ Z2` carries `Complex64`
coefficients while `fZ2 ⊠ U(1)` stays real.

The ordinary user API is `GradedSpace<R>` / `TensorMap<R, D, S>`. It keeps `R`
concrete, returns the provider's own labels (`SectorCodec::Sector`), and keeps
payload scalar `D` and storage `S` separate from the categorical coefficient
scalar. The multiplicity-free typed path currently requires categorical
coefficients of type `f64`. Checked Generic providers have a separate Host-only
admission and execution path. Supported typed CUDA paths are currently
multiplicity-free `f64`. A failed checked operation returns no output tensor or
partial factor tuple.

SU(2) representation algebra itself is not reimplemented here: `tenet-sectors`
delegates 3j/6j, F/R and Frobenius-Schur coefficients plus their caches to the
pinned [`racah`](https://github.com/Ryo-wtnb11/racah) crate. See
[`docs/su2_authority.md`](docs/su2_authority.md) for the pinned revision and the
compatibility protocol.

## Current typed capabilities

The [operation matrix](docs/audit/operation-matrix.md) lists the Host operations
covered by current tests. This includes tensor transformations and
contractions, matrix factorizations, static network traces, and versioned
`f64`/`Complex64` Host snapshots. Storage and device limits are summarized
below.

## Quick Start

Follow the [Quick Start tutorial source](tenet/src/tutorial.md#1-quick-start)
for the required sibling checkout and tested Tenferro revision. That tutorial
is rendered at the top of TeNeT's crate documentation.

The runnable [U(1) example](tenet/examples/quickstart.rs) is the single source
for the calculation. It builds two deterministic charge-preserving maps,
contracts them with `tensor!`, and checks the result:

```sh
cargo run -p tenet --example quickstart
```

For the full workspace checks and local API documentation, run:

```sh
cargo test --workspace
cargo doc --workspace --no-deps
```

For a complete tensor-network application, follow the guided
[U(1) iTEBD Heisenberg tutorial](docs/itebd_heisenberg.md) alongside its
runnable example.

## Backend Philosophy

**Operators say WHAT to compute; backends say WHICH kernel computes it.** The
compiled block layout above is what the kernels run over. The currently
selectable dense and planning backends have explicit selection points. Operator
and user-layer code express spaces, axes, conjugate flags, and output order —
whether that becomes a faer call, a BLAS `op='C'`, or a CUDA kernel is decided
by the selected backend. The full policy is
[`docs/backend_policy.md`](docs/backend_policy.md).

Three consequences worth knowing before you build a `Runtime`:

**Selection is runtime, at `Runtime::builder()`.** There are two independent
dense-backend settings, both defaulting to the pure-Rust faer path:

| builder call | picks the backend for |
| --- | --- |
| `.linalg_backend(LinalgBackend::Faer \| Blas)` | factorizations — SVD / QR / eigh / eig / inv / exp (LAPACK-style work). |
| `.gemm_backend(LinalgBackend::Faer \| Blas)` | the coupled-block contraction GEMM used by `compose`, `contract`, and recoupling execution (BLAS-style work). |
| `.with_dense_executor(Box<dyn DenseExecutor + Send>)` | a fully custom factorization backend; takes precedence over `linalg_backend`. |
| `.cuda(device)` (feature `cuda`) | device placement. |
| `.optimizer(Optimizer::…)` | the contraction-path planner (see below). |

Because OpenBLAS, MKL, and Accelerate cannot be linked simultaneously, *which*
BLAS is a compile-time `blas-*` feature; choosing faer versus the one linked
BLAS stays a runtime decision. `LinalgBackend::Blas` without a compiled
`cpu-blas` / `blas-*` provider fails in `build()` rather than falling back
silently.

**Planning is a backend too.** Contraction-order search is pluggable the same
way — built-in greedy, the pure-Rust `opt-einsum-path` optimizers, or Python
`cotengra` — and it is strictly a planner: TeNeT always executes the resulting
path itself. See [Contraction Planning](#contraction-planning).

**Backend choice affects performance, not tensor semantics.** Every backend
runs against the same oracle suite and is expected to produce the same tensor.
The caveat is floating point, not semantics: BLAS/LAPACK providers differ in
rounding and in decomposition gauge, so parity-sensitive workflows should pin
and test the backend they ship with.

Still hardcoded, and tracked as such in the policy doc: the transpose-free
contraction kernels inside `DenseTreeTransformOperations` are not yet a builder
choice.

## Crates

| crate | role |
| --- | --- |
| `tenet` | Public provider-typed `Runtime`, `GradedSpace<R>`, `TensorMap<R,D,S>`, tensor operations and decomposition results. |
| `tenet-category-data` | Pinned category tables and provenance, currently exposed through `CategoryDataFibonacci`. |
| `tenet-network` | `tensor!` frontend, `NetworkIR`, contraction-order optimizers, reusable `ContractionPlan`, plan cache, slicing metadata. |
| `tenet-macros` | Procedural macro implementation for `tensor!`. |
| `tenet-sectors` | Sector-algebra vocabulary and ready-to-use ZN/Z2, fZ2, U1, CU1, SU2, Fibonacci, product, and feature-gated SUN providers; re-exported by `tenet-core`. |
| `tenet-core` | Fusion-tree spaces and keys, block structures, and low-level statically-ranked tensor-map storage. |
| `tenet-tensors` | Symmetric tensor maps, tensor contraction/transform resolution, execution contexts, caches. |
| `tenet-operations` | TensorOperations-style tensoradd/contract/trace/permute lowering and execution support. |
| `tenet-dense` | Dense block execution boundary and CPU/GPU backend selection. |
| `tenet-matrixalgebra` | SVD/eigh/eig/QR/LQ/polar/matrix-function operations. |
| `tenet-krylov` | Matrix-free Krylov solvers for algorithm layers. v0: conjugate gradient over a `KrylovVector`/`LinearOperator` pair, real `f64` scalars, no dependencies. Not used by the tensor layer yet. |

## Contraction Planning

TeNeT separates path planning from tensor execution.

```text
tensor!(...) labels
  -> NetworkIR + DenseCostModel
  -> DenseContractionOptimizer
  -> ContractionPlan
  -> TensorMap::contract / TensorMap::permute execution
```

The planner sees only metadata:

- input label lists, such as `[["a", "b"], ["b", "c"]]`;
- output labels, such as `["a", "c"]`;
- label dimensions, such as `{ "a": 2, "b": 16, "c": 4 }`;
- optimizer configuration.

It does not receive raw tensor storage, fusion-tree blocks, dense buffers, or
tensor values. External optimizers return an active-pair path such as
`[[0, 1], [0, 1]]`; TeNeT validates that path, builds a `ContractionPlan`, then
executes the plan locally with `TensorMap::contract`.

The plan cache is topology-keyed: labels, adjoint markers, codomain/domain
splits, output labels, and optimizer choice are part of the key; concrete leg
dimensions are tracked as a snapshot for replan policy. The default policy is
`BakeOnce`, i.e. find a non-degenerate order once and reuse it across later
dimension drift.

## Planner Backends

| feature | backend | purpose |
| --- | --- | --- |
| default | built-in greedy | Fast deterministic baseline, no external dependency. |
| `opt-path` | `opt-einsum-path` crate | Pure-Rust path search: `auto`, `auto-hq`, `dp`, `optimal`, branch, random-greedy, memory limit. |
| `cotengra-python` | Python `cotengra` subprocess | Optional high-quality external planner, including cotengra hyper optimization and slicing decisions. |

`opt-einsum-path` receives a generated einsum equation plus shapes. This is for
path search only; TeNeT still executes the contraction itself.

The cotengra backend sends JSON over stdin/stdout to Python:

```json
{
  "inputs": [["a", "b"], ["b", "c"]],
  "output": ["a", "c"],
  "size_dict": {"a": 2, "b": 16, "c": 4},
  "config": {"method": "auto-hq", "minimize": "flops"}
}
```

The Python side calls `cotengra.array_contract_tree(...)` and returns
`tree.get_path()` plus optional sliced-index metadata.

## Features

| feature | effect |
| --- | --- |
| no default features | `tenet-sectors` and `tenet-core` build without a dense backend. Execution crates require a CPU backend or `provider-inject`; otherwise they fail with a backend-selection diagnostic. |
| `cpu-faer` | Default CPU dense backend. |
| `cpu-blas` | Enable the BLAS/LAPACK provider path selected through downstream backend features. |
| `blas-accelerate` | Accelerate-backed BLAS/LAPACK feature wiring. |
| `blas-openblas` | OpenBLAS-backed BLAS/LAPACK feature wiring. |
| `blas-mkl` | MKL-backed BLAS/LAPACK feature wiring. |
| `provider-inject` | Allow injecting a dense backend explicitly. |
| `cuda` | Compile the supported typed CUDA paths, currently multiplicity-free `f64`; a CPU feature is also required for Host-only execution used elsewhere. |
| `racah-generated` | Enable Racah-generated coefficient data and the checked Generic SUN provider through `tenet-sectors`, `tenet-core`, `tenet`, and `tenet-network`. |
| `opt-path` | Enable `opt-einsum-path` optimizers in `tenet-network`. Enable it on `tenet-network`, not on `tenet`: on `tenet` it is a marker that only adds the `Optimizer::{Optimal, DynamicProgramming, AutoHq}` variants. |
| `cotengra-python` | Enable the Python cotengra planner bridge in `tenet-network`. Same marker relationship: on `tenet` it only adds `Optimizer::CotengraPython` and its config types. |

For cotengra, create the Python environment with uv:

```sh
uv sync --project tools/cotengra-python
TENET_COTENGRA_UV_PROJECT=tools/cotengra-python \
  TENET_RUN_COTENGRA_PYTHON_TEST=1 \
  cargo test -p tenet-network --features cotengra-python
```

## Current Limitations

- Checked Generic providers use the same `TensorMap<R,D,S>` ownership model on
  Host storage, including ordinary network execution and static intra-operand
  trace. Checked Generic CUDA remains unsupported. Supported typed CUDA paths
  are currently multiplicity-free `f64`.
- Execution crates reject a no-default-features build because their convenience
  APIs require a concrete executor. Use `tenet-sectors` / `tenet-core` for
  backend-free types, or enable a CPU feature or `provider-inject` for the full
  workspace. CUDA is an additional backend feature and still requires one of
  those host backends.
- CUDA is compile-checked in CI, but requires a CUDA runner for runtime smoke
  tests; Host-only tree-transform execution is not used silently on a device.
- `cotengra-python` is a planner backend, not an executor backend.
- Cotengra slicing decisions can be represented as `SlicedPlan`, but ordinary
  sliced execution over `TensorMap` is not wired yet.
- External planners use dense effective dimensions. Symmetric block execution,
  fusion-tree bookkeeping, fermionic signs, and storage layout remain TeNeT
  execution responsibilities.
- `SectorId`, raw block order, and seeded random storage are internal
  representations rather than cross-version formats. See
  [`docs/sector_id_compatibility.md`](docs/sector_id_compatibility.md) for the
  packed product-sector migration and cache/fixture guidance.

## Documentation Map

- [`docs/itebd_heisenberg.md`](docs/itebd_heisenberg.md): guided U(1) iTEBD
  ground-state calculation for the infinite spin-1/2 Heisenberg chain.
- [`tenet/src/tutorial.md`](tenet/src/tutorial.md): user-layer tutorial with
  compiling examples.
- [`tenet/src/mathematics.md`](tenet/src/mathematics.md): tensor-map
  convention, duality, and categorical semantics.
- [`docs/provider_interface.md`](docs/provider_interface.md): what a symmetry
  provider must implement, which trait owns which data, and the current typed
  capability boundaries.
- [`docs/writing_style.md`](docs/writing_style.md): evidence and plain-language
  rules for TeNeT documentation.
- [`docs/sector_id_compatibility.md`](docs/sector_id_compatibility.md):
  `SectorId`, product-codec, storage-order, seeded-random, and cache
  compatibility contract.
- [`docs/su2_authority.md`](docs/su2_authority.md): the `racah` SU(2)
  coefficient authority — pinned revision, cache ownership, and the
  compatibility protocol for bumping it.
- [`docs/complexity_parity_policy.md`](docs/complexity_parity_policy.md): the
  structured-operation FLOP and storage-order contract.
- [`docs/backend_policy.md`](docs/backend_policy.md): backend selection and the
  policy for performance evidence.
- [`docs/cotengra_backend.md`](docs/cotengra_backend.md): cotengra setup,
  planner protocol, and current limitations.
- [`benchmarks/README.md`](benchmarks/README.md): benchmark harnesses, recorded
  results, and notes.

## Development Notes

Follow the [writing style guide](docs/writing_style.md). Claims about behavior
must match current source and tests; performance claims also need a current
measurement recorded in the pull request.
