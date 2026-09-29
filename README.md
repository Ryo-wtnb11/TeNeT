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

## Operations and documentation

TeNeT provides `permute`, `braid`, `contract`, `trace_pairs`, `qr_compact`,
`svd_compact`, and other operations that an application can compose into its
own algorithm. The [tutorial](tenet/src/tutorial.md) shows their use and the
generated Rust documentation describes individual methods.

Supported operations depend on the symmetry provider, scalar type, and
storage placement. Host has the broadest operation set; CUDA uses explicit
`to_cuda()` and `to_host()` transfers and supports a smaller subset. See the
[documentation index](docs/README.md) for mathematical conventions, backend
capabilities, examples, and revision-specific evidence.

For development, read the [design](docs/design.md) and [coding agent rules](AGENTS.md).
