# TeNeT

TeNeT is a Rust tensor-network library for TensorKit-style symmetric tensors.
The public user layer is built around `Runtime`, `Space`, `Tensor`, and the
`tensor!` contraction macro. Lower layers keep the execution machinery explicit:
fusion-tree/block structure handling, TensorOperations-style lowering, dense
kernel dispatch, matrix decompositions, contraction planning, and plan caching
live in separate crates.

The implementation is currently an active rebuild. Public APIs are intended to
stay Rust-native while matching the TensorKit ecosystem's semantics closely:
TensorKit, TensorKitSectors, TensorOperations, MatrixAlgebraKit, Strided.jl, and
StridedViews.jl are the reference vocabulary. For non-abelian symmetric-tensor
conventions (SU(2), and the fZ2 ⊠ U(1) ⊠ SU(2) products), [QSpace][qspace]
(Weichselbaum) is an additional design and numerical reference alongside
TensorKit — its non-abelian fusion / recoupling (CGC) handling is a second
oracle to check conventions against.

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

## Crates

| crate | role |
| --- | --- |
| `tenet` | Public facade: `Runtime`, `Space`, `Tensor`, scalar dtype, tensor methods, decomposition wrappers. |
| `tenet-network` | `tensor!` frontend, `NetworkIR`, contraction-order optimizers, reusable `ContractionPlan`, plan cache, slicing metadata. |
| `tenet-macros` | Procedural macro implementation for `tensor!`. |
| `tenet-core` | Fusion rules, sectors, fusion-tree keys, block structures, and low-level storage types. |
| `tenet-tensors` | Symmetric tensor maps, tensor contraction/transform resolution, execution contexts, caches. |
| `tenet-operations` | TensorOperations-style tensoradd/contract/trace/permute lowering and replay support. |
| `tenet-dense` | Dense block execution boundary and CPU/GPU backend selection. |
| `tenet-matrixalgebra` | SVD/eigh/eig/QR/LQ/polar/matrix-function operations. |

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
| no default features | Unsupported for execution crates; the build fails with a backend-selection diagnostic. Leaf crates such as `tenet-core` remain backend-free. |
| `cpu-faer` | Default CPU dense backend. |
| `cpu-blas` | Enable the BLAS/LAPACK provider path selected through downstream backend features. |
| `blas-accelerate` | Accelerate-backed BLAS/LAPACK feature wiring. |
| `blas-openblas` | OpenBLAS-backed BLAS/LAPACK feature wiring. |
| `blas-mkl` | MKL-backed BLAS/LAPACK feature wiring. |
| `provider-inject` | Allow injecting a dense provider explicitly. |
| `cuda` | Compile CUDA execution paths where implemented; a CPU feature is also required for host-only replay. |
| `opt-path` | Enable `opt-einsum-path` optimizers in `tenet-network`. |
| `cotengra-python` | Enable the Python cotengra planner bridge in `tenet-network`. |

For cotengra, create the Python environment with uv:

```sh
uv sync --project tools/cotengra-python
TENET_COTENGRA_UV_PROJECT=tools/cotengra-python \
  TENET_RUN_COTENGRA_PYTHON_TEST=1 \
  cargo test -p tenet-network --features cotengra-python
```

## Current Limitations

- Execution crates reject a no-default-features build because their convenience
  APIs require a concrete executor. Use `tenet-core` for backend-free types, or
  enable a CPU/CUDA feature or `provider-inject` for the full workspace.
- CUDA is compile-checked in CI, but requires a CUDA runner for runtime smoke
  tests; host-only tree-transform replay is not silently used as device replay.
- `cotengra-python` is a planner backend, not an executor backend.
- Cotengra slicing decisions can be represented as `SlicedPlan`, but ordinary
  sliced execution over `Tensor` is not wired yet.
- External planners use dense effective dimensions. Symmetric block execution,
  fusion-tree bookkeeping, fermionic signs, and storage layout remain TeNeT
  execution responsibilities.
- BLAS/LAPACK backend choices can change floating-point and decomposition gauge
  behavior. Numerical parity-sensitive workflows should pin and test the chosen
  backend.

## Documentation Map

- [`tenet/src/tutorial.md`](tenet/src/tutorial.md): user-layer tutorial with
  compiling examples.
- [`tenet/src/mathematics.md`](tenet/src/mathematics.md): tensor-map
  convention, duality, and categorical semantics.
- [`docs/user_api_design.md`](docs/user_api_design.md): API design notes and
  TensorKit vocabulary alignment.
- [`docs/tensorkit_compatibility_table.md`](docs/tensorkit_compatibility_table.md):
  TensorKit naming and compatibility table.
- [`docs/cotengra_backend.md`](docs/cotengra_backend.md): cotengra Python
  backend setup, latency, and limitations.
- [`benchmarks/README.md`](benchmarks/README.md): benchmark notes and measured
  performance work.

## Development Notes

Before architectural or semantic changes, read the repository review policy in
`../AGENTS.md`. TeNeT changes that claim TensorKit compatibility should be
checked against the reference implementation, not only against local tests.
