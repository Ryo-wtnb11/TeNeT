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

An admitted tensor owns its exact provider authority and a compiled block
layout: fusion-tree keys, per-block shapes and strides, and offsets into one
contiguous payload. Checked admission validates the complete structure before
publishing a space, tensor, plan, or result. Fusion-tree layouts, recoupling
replay, and contraction plans can then be reused without changing their
provider or gauge snapshot.

`Runtime` owns the bounded caches, workspace pools, and selectable dense
backends used by those operations. Contraction-path planners consume network
metadata and produce validated plans; TeNeT executes the plans locally over its
reduced blocks. This keeps mathematical structure, resource ownership,
planning, and kernel selection at explicit seams.

External libraries and papers are cited in function documentation,
[`tenet/references.md`](tenet/references.md), tests, or benchmark reports when
they support a particular convention, divergence, oracle, or comparison. They
do not define TeNeT's public contract.

## Symmetries are providers

`tenet-sectors` owns the provider vocabulary — `FusionRule`,
`CheckedFusionAlgebra`, `SectorCodec`, `MultiplicityFreeRigidSymbols`,
`GenericRigidSymbols`, `RuleIdentity` — and the built-in symmetries (U(1), Z2,
fZ2, SU(2), Fibonacci, and their products) are ordinary implementations of
those traits. There is no provider enum and no symmetry-named dispatch arm in
any operation. The crate has zero workspace dependencies.
[`docs/provider_interface.md`](docs/provider_interface.md) is the contract for
writing one.

Two qualifications, because "no privileged status" would be too strong today.
The built-ins additionally implement a sealed layout-enumeration trait
(`LoweredMultiplicityFreeAlgebra`) that lets the cold enumeration work on
decoded labels instead of round-tripping `SectorId` per channel; an external
provider takes the same semantic path with a constant-factor penalty and cannot
opt in. And `tenet-core` names the built-in providers to implement it. Both are
recorded as debt, not as design.

Products are a provider combinator, not a list of blessed symmetries. A
product of providers is itself a provider, so

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
payload scalar `D` and storage `S` orthogonal to the categorical coefficient
scalar. Construction, transforms, contractions, reductions, factorizations,
matrix functions, compact diagonal storage, Host execution and supported CUDA
execution all use this ownership model. Checked admission is transactional: an
invalid or unrepresentable algebra publishes no layout or cache state.

SU(2) representation algebra itself is not reimplemented here: `tenet-sectors`
delegates 3j/6j, F/R and Frobenius-Schur coefficients plus their caches to the
pinned [`racah`](https://github.com/Ryo-wtnb11/racah) crate. See
[`docs/su2_authority.md`](docs/su2_authority.md) for the pinned revision and the
compatibility protocol.

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

## Backend Philosophy

**Operators say WHAT to compute; backends say WHICH kernel computes it.** The
compiled block layout above is what the kernels run over; the kernel itself is a
separate, replaceable decision. Every compute primitive that has more than one
plausible implementation is a trait with an explicit selection point, never a
hardcoded dependency. Operator and user-layer code express spaces, axes,
conjugate flags, and output order — whether that becomes a faer call, a BLAS
`op='C'`, or a CUDA kernel is the backend layer's business. The full rule set is
[`docs/backend_policy.md`](docs/backend_policy.md).

Three consequences worth knowing before you build a `Runtime`:

**Selection is runtime, at `Runtime::builder()`.** The dense provider is two
independent knobs, both defaulting to the pure-Rust faer path:

| builder call | picks the backend for |
| --- | --- |
| `.linalg_backend(LinalgBackend::Faer \| Blas)` | factorizations — SVD / QR / eigh / eig / inv / exp (LAPACK-style work). |
| `.gemm_backend(LinalgBackend::Faer \| Blas)` | the coupled-block contraction GEMM used by `compose` / `contract` and recoupling replay (BLAS-style work). |
| `.with_dense_executor(Box<dyn DenseExecutor + Send>)` | a fully custom factorization provider; takes precedence over `linalg_backend`. |
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

**Backend choice is a performance knob, not a semantics knob.** Every backend
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
| `tenet-network` | `tensor!` frontend, `NetworkIR`, contraction-order optimizers, reusable `ContractionPlan`, plan cache, slicing metadata. |
| `tenet-macros` | Procedural macro implementation for `tensor!`. |
| `tenet-sectors` | Sector-algebra vocabulary: fusion-rule/codec traits, `SectorId`, and the built-in irrep providers (U(1), Z2, fZ2, SU(2), Fibonacci, products). No workspace dependencies; re-exported wholesale by `tenet-core`. |
| `tenet-core` | Fusion-tree spaces and keys, block structures, and low-level statically-ranked tensor-map storage. |
| `tenet-tensors` | Symmetric tensor maps, tensor contraction/transform resolution, execution contexts, caches. |
| `tenet-operations` | TensorOperations-style tensoradd/contract/trace/permute lowering and replay support. |
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
  -> TensorMap::contract / TensorMap::permute replay
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
| no default features | Unsupported for execution crates; `tenet-operations` fails the build with a backend-selection diagnostic that the rest of the stack inherits. Leaf crates (`tenet-sectors`, `tenet-core`, `tenet-macros`, `tenet-krylov`) declare no features at all and remain backend-free. |
| `cpu-faer` | Default CPU dense backend. |
| `cpu-blas` | Enable the BLAS/LAPACK provider path selected through downstream backend features. |
| `blas-accelerate` | Accelerate-backed BLAS/LAPACK feature wiring. |
| `blas-openblas` | OpenBLAS-backed BLAS/LAPACK feature wiring. |
| `blas-mkl` | MKL-backed BLAS/LAPACK feature wiring. |
| `provider-inject` | Allow injecting a dense provider explicitly. |
| `cuda` | Compile CUDA execution paths where implemented; a CPU feature is also required for host-only replay. |
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

- Checked Generic providers use the same `TensorMap<R,D,S>` ownership model,
  but operations that require capabilities not yet supplied by their provider
  remain unavailable. `tensor!` supports Host tensors and canonical returning
  CUDA schedules; intra-operand trace lowering remains Host-only and returns
  `UnsupportedOnDevice` on CUDA.
- Execution crates reject a no-default-features build because their convenience
  APIs require a concrete executor. Use `tenet-sectors` / `tenet-core` for
  backend-free types, or enable a CPU feature or `provider-inject` for the full
  workspace. CUDA is an additional backend feature and still requires one of
  those host backends.
- CUDA is compile-checked in CI, but requires a CUDA runner for runtime smoke
  tests; host-only tree-transform replay is not silently used as device replay.
- `cotengra-python` is a planner backend, not an executor backend.
- Cotengra slicing decisions can be represented as `SlicedPlan`, but ordinary
  sliced execution over `TensorMap` is not wired yet.
- External planners use dense effective dimensions. Symmetric block execution,
  fusion-tree bookkeeping, fermionic signs, and storage layout remain TeNeT
  execution responsibilities.
- The fast cold layout enumeration is behind a sealed trait implemented for the
  built-in providers only. External providers are semantically equal and pay a
  constant factor on that path.
- `SectorId`, raw block order, and seeded random storage are internal
  representations rather than cross-version formats. See
  [`docs/sector_id_compatibility.md`](docs/sector_id_compatibility.md) for the
  packed product-sector migration and cache/fixture guidance.

## Documentation Map

- [`tenet/src/tutorial.md`](tenet/src/tutorial.md): user-layer tutorial with
  compiling examples.
- [`tenet/src/mathematics.md`](tenet/src/mathematics.md): tensor-map
  convention, duality, and categorical semantics.
- [`docs/provider_interface.md`](docs/provider_interface.md): what a symmetry
  provider must implement, which trait owns which data, and the current
  restrictions on external providers.
- [`docs/sector_id_compatibility.md`](docs/sector_id_compatibility.md):
  `SectorId`, product-codec, storage-order, seeded-random, and cache
  compatibility contract.
- [`docs/su2_authority.md`](docs/su2_authority.md): the `racah` SU(2)
  coefficient authority — pinned revision, cache ownership, and the
  compatibility protocol for bumping it.
- [`docs/complexity_parity_policy.md`](docs/complexity_parity_policy.md): the
  structured-operation FLOP and storage-order contract.
- [`docs/backend_policy.md`](docs/backend_policy.md): selectable dense
  transpose/GEMM backend design and measured thread scaling.
- [`docs/cotengra_backend.md`](docs/cotengra_backend.md): cotengra Python
  backend setup, latency, and limitations.
- [`benchmarks/README.md`](benchmarks/README.md): benchmark notes and measured
  performance work.

## Development Notes

Before architectural or semantic changes, read the repository review policy in
`../AGENTS.md`. Claims about an external correspondence should be checked
against the cited source as well as TeNeT's local semantic tests.
