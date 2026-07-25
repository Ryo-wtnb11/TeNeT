# TeNeT

TeNeT is a Rust tensor-network library for TensorKit-style symmetric tensors.
The public user layer is built around `Runtime`, `Space`, `Tensor`, and the
`tensor!` contraction macro. Lower layers keep the execution machinery explicit:
sector algebra and fusion rules, fusion-tree/block structure handling,
TensorOperations-style lowering, dense kernel dispatch, matrix decompositions,
contraction planning, and plan caching live in separate crates.

Symmetries enter through fusion-rule providers, not a fixed symmetry list.
`tenet-sectors` owns the provider vocabulary — `FusionRule`, `SectorCodec`,
`MultiplicityFreeRigidSymbols`, `GenericRigidSymbols`, `RuleIdentity` — and the
built-in rules (U(1), Z2, fZ2, SU(2), Fibonacci, and their products; SU(3)'s
tabulated rule lives in `tenet-core`) are ordinary implementations of it.
Admitting provider types defined outside the workspace is being built out; the
label-codec half of that contract (`SectorCodec`) is already in place.

All crates are at version `0.1.0`. The public API is not stabilized and
expert-layer types still move between crates as the layering settles — the
sector vocabulary was recently split out of `tenet-core` into `tenet-sectors`.
Public APIs are intended to stay Rust-native while matching the TensorKit
ecosystem's semantics closely: TensorKit, TensorKitSectors, TensorOperations,
MatrixAlgebraKit, Strided.jl, and StridedViews.jl are the reference vocabulary,
and names follow the TensorKit 0.17 spelling. [QSpace][qspace] (Weichselbaum)
is the second design reference, in two roles. It fixes the compiled execution
model that the `Runtime` layer follows: quantum labels, block structure,
coefficient records, and runtime-rank metadata stay tensor-near and feed
whole-tensor kernel dispatch, rather than being recomputed per operation. It is
also the numerical oracle for non-abelian conventions (SU(2), and the
fZ2 ⊠ U(1) ⊠ SU(2) products) — its fusion / recoupling (CGC) handling is the
second source to check conventions against, alongside TensorKit.
[`docs/tk_api_parity.md`](docs/tk_api_parity.md) is the per-export lookup table.

SU(2) representation algebra itself is not reimplemented here: `tenet-sectors`
delegates 3j/6j, F/R, and Frobenius-Schur coefficients plus their caches to the
pinned [`racah`](https://github.com/Ryo-wtnb11/racah) crate. See
[`docs/su2_authority.md`](docs/su2_authority.md) for the pinned revision and the
compatibility protocol.

[qspace]: https://bitbucket.org/qspace4u/workspace/repositories/

## Quick Start

```sh
cargo test --workspace
cargo doc --workspace --no-deps
```

Minimal user-layer example:

```rust
use tenet::prelude::*;
use tenet_network::tensor;

fn main() -> Result<(), Error> {
    let rt = Runtime::builder().build()?;
    let v = Space::u1([(-1, 2), (0, 3), (1, 2)]);

    // Tensors are maps codomain <- domain.
    let a = Tensor::rand(&rt, Dtype::F64, [&v, &v], [&v, &v])?;
    let b = Tensor::rand(&rt, Dtype::F64, [&v, &v], [&v, &v])?;

    // @tensor-style notation: [codomain; domain].
    let c = tensor!([i, j; g, h] = a[i, j; k, l] * b[k, l; g, h])?;
    assert_eq!((c.codomain_rank(), c.domain_rank()), (2, 2));

    Ok(())
}
```

`tensor!` does not expose an einsum string parser. Labels are Rust identifiers
inside the macro, `;` separates codomain and domain legs, `[]` is a scalar
output, and `conj(x)[...]` marks an adjoint operand.

## Backend Philosophy

**Operators say WHAT to compute; backends say WHICH kernel computes it.** Every
compute primitive that has more than one plausible implementation is a trait
with an explicit selection point, never a hardcoded dependency. Operator and
user-layer code express spaces, axes, conjugate flags, and output order —
whether that becomes a faer call, a BLAS `op='C'`, or a CUDA kernel is the
backend layer's business. The full rule set is
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
| `tenet` | Public facade: `Runtime`, `Space`, `Tensor`, scalar dtype, tensor methods, decomposition wrappers. |
| `tenet-network` | `tensor!` frontend, `NetworkIR`, contraction-order optimizers, reusable `ContractionPlan`, plan cache, slicing metadata. |
| `tenet-macros` | Procedural macro implementation for `tensor!`. |
| `tenet-sectors` | Sector-algebra vocabulary: fusion-rule/codec traits, `SectorId`, and the built-in irrep providers (U(1), Z2, fZ2, SU(2), Fibonacci, products). No workspace dependencies; re-exported wholesale by `tenet-core`. |
| `tenet-core` | Fusion-tree spaces and keys, block structures, low-level storage types, the typed `TensorMap`, and the tabulated SU(3) rule. |
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
