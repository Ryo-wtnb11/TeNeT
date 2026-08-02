# TeNeT

TeNeT is a Rust tensor-network library for symmetric tensors. A tensor is a map
`codomain <- domain` stored as fusion-tree-indexed reduced blocks; symmetries
enter as fusion-rule *providers* rather than as a fixed symmetry list; and
contraction-path planning is separated from execution.

Design priority, in this order: Rust-native maintainability and extensibility;
speed that survives dynamic-rank tensor networks; TensorKit-level usability.

All crates are at version `0.1.0`. The public API is not stabilized and
expert-layer types still move between crates as the layering settles.

## Two reference frames

TeNeT is not a port of a single library. Two references divide the design, and
the split is visible in the crate layering.

**TensorKit (Julia) fixes the semantic model.** Spaces and tensor maps retain
their sector/provider type; categorical algorithms dispatch on *fusion
capability* — multiplicity-free versus generic symbols, braiding style,
rigidity — instead of on a symmetry enum; tensor maps are `codomain <- domain`
with explicit duality. Names follow the TensorKit 0.17 spelling, and
TensorKitSectors, TensorOperations, MatrixAlgebraKit, Strided.jl and
StridedViews.jl are the vocabulary for the layers below the user API.
[`docs/tk_api_parity.md`](docs/tk_api_parity.md) is the per-export lookup table
for anyone arriving from TensorKit.

**[QSpace][qspace] (Weichselbaum) fixes the compiled execution model** — it is the base of
the `Runtime`/backend layer. Quantum labels, block structure, coefficient
records and runtime-rank metadata stay *tensor-near*: an admitted tensor carries
its own block layout (fusion-tree keys plus per-block shape, strides and
offset into one contiguous payload), so an operation resolves the layout once
and then dispatches whole blocks into dense kernels, instead of rediscovering
the symmetry structure per element or per call. Fusion-tree layouts, complete
hom-space structures, recoupling replay and contraction plans are recorded and
reused rather than recomputed. QSpace is also the second oracle for non-abelian
conventions: SU(2) and the fZ2 ⊠ U(1) ⊠ SU(2) products are checked against its
fusion/recoupling (CGC) handling as well as against TensorKit.

Rust's part is to hold both at once: keep the concrete provider type `R` through
monomorphized execution — TensorKit's semantics without a vtable — over a
QSpace-style compiled block engine, whose kernels are selectable at
`Runtime::builder()` (see [Backend Philosophy](#backend-philosophy)).

[qspace]: https://bitbucket.org/qspace4u/workspace/repositories/

## Symmetries are providers

`tenet-sectors` owns the provider vocabulary — `FusionRule`,
`CheckedFusionAlgebra`, `SectorCodec`, `MultiplicityFreeRigidSymbols`,
`GenericRigidSymbols`, `RuleIdentity` — and the built-in symmetries (U(1), Z2,
fZ2, SU(2), Fibonacci, and their products) are ordinary implementations of
those traits, with no privileged
status in the engine. The crate has zero workspace dependencies.

Products are a provider combinator, not a list of blessed symmetries. A
product of providers is itself a provider, so

```text
FermionParityFusionRule.product(U1FusionRule).product(SU2FusionRule)
```

is `(fZ2 ⊠ U(1)) ⊠ SU(2)` — any ordered product of admitted components,
recursively nested, without a new `RuleKind`, dispatch arm or `Space`
constructor. Factor order and association are structure of the Rust type and of
the `ProductSector` label, never an automatic equivalence: `U(1) ⊠ fZ2` and
`fZ2 ⊠ U(1)` are both legal and are different types. `tenet::typed`'s module
documentation carries the compiling example; the erased `Space::product` and
`Space::fz2_u1_su2` are two fixed conveniences kept for compatibility and are
not the extension mechanism.

TeNeT has two peer user facades. The provider-typed
[`tenet::typed`](tenet/src/typed.rs) facade
(`GradedSpace<R>`, `TensorMap<R, D>`) keeps `R` concrete, returns the provider's
own labels (`SectorCodec::Sector`), and separates payload dtype `D` (`f64` /
`Complex64`) from the categorical coefficient scalar. It ships construction,
inspection, transforms and contractions, scalar/reduction operations,
factorizations and matrix functions, and compact-diagonal paths. Construction
uses transactional checked admission, so an invalid or unrepresentable algebra
publishes no layout, cache, or admission state.

The rule-erased `Runtime` / `Space` / `Tensor` facade is its peer for built-in
providers. It owns runtime dtype and placement dispatch and its own
lazy-adjoint state. `tensor!` belongs to the provider-typed facade: it executes
typed Host tensors through a per-plan workspace and canonical CUDA tensors
through the returning-only device path.
Neither facade replaces or wraps the other.

SU(2) representation algebra itself is not reimplemented here: `tenet-sectors`
delegates 3j/6j, F/R and Frobenius-Schur coefficients plus their caches to the
pinned [`racah`](https://github.com/Ryo-wtnb11/racah) crate. See
[`docs/su2_authority.md`](docs/su2_authority.md) for the pinned revision and the
compatibility protocol.

## Quick Start

```sh
cargo test --workspace
cargo doc --workspace --no-deps
```

The erased user layer — the shortest path to a working contraction:

```rust
use tenet::prelude::*;

fn main() -> Result<(), Error> {
    let rt = Runtime::builder().build()?;
    let v = Space::u1([(-1, 2), (0, 3), (1, 2)]);

    // Tensors are maps codomain <- domain.
    let a = Tensor::rand(&rt, Dtype::F64, [&v, &v], [&v, &v])?;
    let b = Tensor::rand(&rt, Dtype::F64, [&v, &v], [&v, &v])?;

    let c = a.compose(&b)?;
    assert_eq!((c.codomain_rank(), c.domain_rank()), (2, 2));

    Ok(())
}
```

`tensor!` does not expose an einsum string parser. Labels are Rust identifiers
inside the macro, `;` separates codomain and domain legs, `[]` is a scalar
output, and `conj(x)[...]` marks an adjoint operand.

The same U(1) space through the typed facade. The provider is an ordinary value
here, so a fusion rule defined outside this workspace substitutes for
`U1FusionRule` without touching the engine:

```rust
use std::sync::Arc;

use tenet::core::{U1FusionRule, U1Irrep};
use tenet::prelude::{Error, Runtime};
use tenet::typed::{GradedSpace, TensorMap};
use tenet_network::tensor;

fn main() -> Result<(), Error> {
    let rt = Runtime::builder().build()?;
    let u1 = Arc::new(U1FusionRule);

    let v = GradedSpace::try_new(
        Arc::clone(&u1),
        [(U1Irrep::new(-1), 2), (U1Irrep::new(0), 3), (U1Irrep::new(1), 2)],
        false,
    )?;

    let t: TensorMap<U1FusionRule, f64> = TensorMap::zeros(&rt, [&v], [&v])?;
    let u: TensorMap<U1FusionRule, f64> = TensorMap::zeros(&rt, [&v], [&v])?;
    assert_eq!(t.block_count(), 3);

    // Blocks report the provider's own labels, not SectorIds.
    let trees = t.block_fusion_trees(0)?;
    assert_eq!(trees.coupled(), &U1Irrep::new(0));

    // @tensor-style notation: [codomain; domain].
    let c = tensor!([i; k] = t[i; j] * u[j; k])?;
    assert_eq!((c.codomain_rank(), c.domain_rank()), (1, 1));

    Ok(())
}
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
| `tenet` | Public facade: the erased `Runtime`, `Space`, `Tensor`, scalar dtype, tensor methods, decomposition wrappers, and the provider-typed `tenet::typed` facade. |
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
  -> Tensor::contract / Tensor::permute replay
```

The planner sees only metadata:

- input label lists, such as `[["a", "b"], ["b", "c"]]`;
- output labels, such as `["a", "c"]`;
- label dimensions, such as `{ "a": 2, "b": 16, "c": 4 }`;
- optimizer configuration.

It does not receive raw tensor storage, fusion-tree blocks, dense buffers, or
tensor values. External optimizers return an active-pair path such as
`[[0, 1], [0, 1]]`; TeNeT validates that path, builds a `ContractionPlan`, then
executes the plan locally with `Tensor::contract`.

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

- The typed facade admits checked multiplicity-free providers. It has no
  conversion to or from the erased `Tensor`. `tensor!` supports Host tensors
  and canonical returning CUDA schedules; intra-operand trace lowering remains
  Host-only and returns `UnsupportedOnDevice` on CUDA.
- Execution crates reject a no-default-features build because their convenience
  APIs require a concrete executor. Use `tenet-sectors` / `tenet-core` for
  backend-free types, or enable a CPU feature or `provider-inject` for the full
  workspace. CUDA is an additional backend feature and still requires one of
  those host backends.
- CUDA is compile-checked in CI, but requires a CUDA runner for runtime smoke
  tests; host-only tree-transform replay is not silently used as device replay.
- `cotengra-python` is a planner backend, not an executor backend.
- Cotengra slicing decisions can be represented as `SlicedPlan`, but ordinary
  sliced execution over `Tensor` is not wired yet.
- External planners use dense effective dimensions. Symmetric block execution,
  fusion-tree bookkeeping, fermionic signs, and storage layout remain TeNeT
  execution responsibilities.
- `SectorId`, raw block order, and seeded random storage are internal
  representations rather than cross-version formats. See
  [`docs/sector_id_compatibility.md`](docs/sector_id_compatibility.md) for the
  packed product-sector migration and cache/fixture guidance.

## Documentation Map

- [`tenet/src/tutorial.md`](tenet/src/tutorial.md): user-layer tutorial with
  compiling examples.
- [`tenet/src/mathematics.md`](tenet/src/mathematics.md): tensor-map
  convention, duality, and categorical semantics.
- [`docs/tk_api_parity.md`](docs/tk_api_parity.md): TensorKit 0.17 user-API
  parity table — every user-facing export, its TeNeT name, and the rationale
  for anything spelled or gated differently. The lookup surface for a
  TensorKit user.
- [`docs/user_api_design.md`](docs/user_api_design.md): API design notes and
  TensorKit vocabulary alignment.
- [`docs/sector_id_compatibility.md`](docs/sector_id_compatibility.md):
  `SectorId`, product-codec, storage-order, seeded-random, and cache
  compatibility contract.
- [`docs/tensorkit_compatibility_table.md`](docs/tensorkit_compatibility_table.md):
  internal naming and compatibility table.
- [`docs/su2_authority.md`](docs/su2_authority.md): the `racah` SU(2)
  coefficient authority — pinned revision, cache ownership, and the
  compatibility protocol for bumping it.
- [`docs/complexity_parity_policy.md`](docs/complexity_parity_policy.md): the
  rule that TeNeT must match TensorKit's asymptotic FLOP/storage order.
- [`docs/backend_policy.md`](docs/backend_policy.md): selectable dense
  transpose/GEMM backend design and measured thread scaling.
- [`docs/cotengra_backend.md`](docs/cotengra_backend.md): cotengra Python
  backend setup, latency, and limitations.
- [`benchmarks/README.md`](benchmarks/README.md): benchmark notes and measured
  performance work.

## Development Notes

Before architectural or semantic changes, read the repository review policy in
`../AGENTS.md`. TeNeT changes that claim TensorKit compatibility should be
checked against the reference implementation, not only against local tests.
</content>
