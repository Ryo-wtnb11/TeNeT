# TeNeT

**Symmetric tensor primitives for composable algorithms in Rust.**

TeNeT provides small operations on block-sparse tensor maps: construct a space,
build a tensor, permute or braid its legs, contract, trace, or factorize it.
An algorithm written by a person or a coding agent can combine these operations
without implementing fusion trees, sector bookkeeping, or dense block layouts.

The caller chooses the algorithm, operation order, and when to move a tensor
between Host and CUDA storage. Within each supported operation, TeNeT handles
symmetry-aware decomposition, block layout and data movement, and submission of
dense work. [Tenferro](https://github.com/tensor4all/tenferro-rs) supplies the
dense kernels and provider resources. This division keeps individual operations
optimizable without hiding the algorithm inside a large workflow API.

TeNeT is under active development; the public API is not yet stable. Ordinary
applications use the `tenet-rs` package (imported as `tenet`). Add
`tenet-network` for the `tensor!` notation and network planning.

## Start with a tensor

This example creates two U(1)-symmetric maps and composes them. A `GradedSpace`
describes one leg by its charge sectors and degeneracies. A `TensorMap` is a map
`codomain <- domain`; only symmetry-allowed reduced blocks are stored.

```rust
use std::sync::Arc;
use tenet::sector::{U1FusionRule, U1Irrep};
use tenet::typed::{Error, GradedSpace, Runtime, TensorMap};

fn main() -> Result<(), Error> {
    let runtime = Runtime::builder().build()?;
    let leg = GradedSpace::try_new(
        Arc::new(U1FusionRule),
        [(U1Irrep::new(-1), 1), (U1Irrep::new(0), 2), (U1Irrep::new(1), 1)],
    )?;
    let a: TensorMap<U1FusionRule, f64> =
        TensorMap::rand_with_seed(&runtime, [&leg], [&leg], 1)?;
    let b: TensorMap<U1FusionRule, f64> =
        TensorMap::rand_with_seed(&runtime, [&leg], [&leg], 2)?;
    let c = a.compose(&b)?;
    println!("squared norm: {}", c.inner(&c)?);
    Ok(())
}
```

For a runnable indexed contraction with `tensor!`, use
[`tenet-network/examples/quickstart.rs`](tenet-network/examples/quickstart.rs):

```sh
cargo run -p tenet-network --example quickstart
```

## Choose the next primitive

| Task | Public entry point |
| --- | --- |
| Define legs and construct tensors | `GradedSpace`, `TensorMap::from_subblock_fn`, `zeros`, `isomorphism`, `rand_with_seed` |
| Change leg order or orientation | `permute`, `braid`, `repartition`, `adjoint` |
| Combine tensors | `compose`, `contract` with `ContractSpec`, `tensor!` for an indexed network |
| Reduce or inspect | `trace_pairs`, `inner`, `norm`, `blocks`, `subblocks` |
| Factorize | `qr_compact`, `svd_compact`, `eigh_full`, and their result types |
| Reuse repeated work | Explicit plan/workspace APIs where supported; `ComposePlan` and `ComposeWorkspace` for stacked composition |

These are operations, not a prescribed MPS, PEPS, or VMC algorithm. The caller
can select a contraction order, choose destinations and reusable state where an
API provides them, and combine calls into a larger computation. TeNeT remains
responsible for the symmetry semantics and dense work *inside* each call.
Available methods depend on the provider, scalar type, and storage placement;
unsupported combinations return explicit errors or have no matching method.
The [crate tutorial](tenet/src/tutorial.md) shows method signatures and
examples. Generate the local function-level Rust documentation with:

```sh
cargo doc -p tenet-rs -p tenet-network --no-deps --open
```

For repeated computations, an application can organize tensor instances into
batches and use an explicit prepared API when that operation provides one.
For a new algorithm, start with the ordinary operations above; their types and
errors expose the required spaces, axes, and supported placement to coding
agents as well as human callers.

## Symmetry and execution boundaries

TeNeT's engine is generic over a fusion-rule provider. Built-in providers
include ZN/Z2, fermion parity, U(1), CU(1), SU(2), Fibonacci, ordered products,
and feature-gated SUN. Providers supply sector labels and categorical data;
the engine owns fusion-tree-indexed reduced blocks, categorical transforms,
contraction layouts, and validation. A product such as
`FermionParityFusionRule.product(U1FusionRule)` is an ordered product, not an
automatic equivalence with the reverse order. See the
[provider interface](docs/provider_interface.md) for trait and capability
requirements. SU(2) coefficient generation delegates to
[`racah`](https://github.com/Ryo-wtnb11/racah).

`Runtime` owns execution resources. Host dense factorizations and contraction
GEMM can select faer or one compiled BLAS provider independently through
`Runtime::builder().linalg_backend(...)` and `.gemm_backend(...)`. CUDA is an
explicit feature and transfer path: `.cuda(device)` makes a device available,
and `to_cuda()` / `to_host()` move tensor payloads. TeNeT does not silently
transfer a tensor to make an unsupported device operation work. The
[backend policy](docs/backend_policy.md) describes selection and resource
ownership.

For a tensor network, `tenet-network` can select and reuse a contraction path.
The planner reads labels and dimensions, not payload data. TeNeT executes the
chosen path on reduced blocks. Built-in greedy planning needs no external
optimizer; optional `opt-path` and `cotengra-python` features add planners.
They do not replace TeNeT's execution engine.

## Current scope

- Host is the broadest typed execution path. Checked Generic providers have a
  narrower Host-only operation set. Fibonacci has a tested subset of typed
  operations; arbitrary-axis contraction and factorization are not claimed.
- CUDA supports a multiplicity-free `f64`/`Complex64` subset after explicit
  transfer. Supported device operations include contraction, transforms, trace,
  compact SVD and QR, and EIGH, subject to each method's bounds. Full and
  values-only SVD/QR, `eig`, and matrix functions have no device path.
  CUDA runtime tests require a CUDA runner; CI also checks that the feature
  compiles.
- `ComposePlan` / `ComposeWorkspace` support stacked composition, and
  `PreparedEighFull` supports batched EIGH. General member-batched `contract`
  is not public yet ([#1506](https://github.com/Ryo-wtnb11/TeNeT/issues/1506)).
- The default `tenet-rs` feature is `cpu-faer`. `cpu-blas` needs one linked
  `blas-accelerate`, `blas-openblas`, or `blas-mkl` feature. See the
  [manifest](tenet/Cargo.toml) and [backend policy](docs/backend_policy.md) for
  the full feature contract.

## Read next

- [Tutorial and API examples](tenet/src/tutorial.md): tensor construction,
  axes, contraction, decomposition, and backends.
- [Tensor-map mathematics](tenet/src/mathematics.md): duality, orientation, and
  categorical conventions.
- [U(1) iTEBD example](docs/itebd_heisenberg.md): a complete algorithm built
  from these operations.
- [Provider interface](docs/provider_interface.md): define another symmetry.
- [Benchmark records](benchmarks/README.md): revision-specific performance
  evidence; the README makes no general speed claim.
- [Design](docs/design.md): structural ownership, execution, reuse, and
  placement boundaries.
- [Coding agent rules](AGENTS.md): how to verify and change TeNeT.
- [Documentation map](docs/README.md): current guides and revision-pinned
  evidence.

To check the workspace locally:

```sh
cargo test --workspace
cargo doc --workspace --no-deps
```
