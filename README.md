# symtn

**Symmetric tensor primitives for composable algorithms in Rust.**

symtn provides small operations on block-sparse tensor maps: construct a space,
build a tensor, permute or braid its legs, contract, trace, or factorize it.
An algorithm written by a person or a coding agent can combine these operations
without implementing fusion trees, sector bookkeeping, or dense block layouts.

The caller chooses the algorithm, operation order, and when to move a tensor
between Host and CUDA storage. Within each supported operation, symtn handles
symmetry-aware decomposition, block layout and data movement, and submission of
dense work. [Tenferro](https://github.com/tensor4all/tenferro-rs) supplies the
dense kernels and provider resources. This division keeps individual operations
optimizable without hiding the algorithm inside a large workflow API.

symtn is under active development; the public API is not yet stable. Ordinary
applications use the `symtn` crate. Add
`symtn-network` for labeled tensor networks and network planning. The crates
are not yet published on crates.io; depend on the repository:

```toml
[dependencies]
symtn = { git = "https://github.com/Ryo-wtnb11/symtn-rs" }
symtn-network = { git = "https://github.com/Ryo-wtnb11/symtn-rs" }
```

## Acknowledgments and upstream work

symtn builds on [TensorKit.jl](https://github.com/QuantumKitHub/TensorKit.jl) and [QSpace](https://bitbucket.org/qspace4u/). We thank their authors and contributors.

See [provenance and references](symtn/references.md) for detailed source correspondences.

## Citation

**If you use symtn in research, please cite TensorKit and QSpace.** This requests scholarly credit for the work on which
symtn builds; it does not add a condition to the software license.

- **TensorKit:** Lukas Devos and Jutho Haegeman, *TensorKit.jl: A Julia package
  for large-scale tensor computations, with a hint of category theory* (2025),
  [doi:10.48550/arXiv.2508.10076](https://doi.org/10.48550/arXiv.2508.10076).
  See TensorKit's [official citation metadata](https://github.com/QuantumKitHub/TensorKit.jl/blob/main/CITATION.cff)
  for its preferred citation and software record.
- **QSpace:** Andreas Weichselbaum, *QSpace — An open-source tensor library for
  Abelian and non-Abelian symmetries*, SciPost Physics Codebases **40** (2024),
  [doi:10.21468/SciPostPhysCodeb.40](https://doi.org/10.21468/SciPostPhysCodeb.40).
  When using QSpace directly, also cite the release used, following the
  [QSpace publication's citation guidance](https://scipost.org/SciPostPhysCodeb.40).

## Start with a tensor

This example creates two U(1)-symmetric maps and composes them. A `GradedSpace`
describes one leg by its charge sectors and degeneracies. A `TensorMap` is a map
`codomain <- domain`; only symmetry-allowed reduced blocks are stored.

```rust
use std::sync::Arc;
use symtn::sector::{U1FusionRule, U1Irrep};
use symtn::typed::{Error, GradedSpace, Runtime, TensorMap};

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

For a runnable labeled contraction with `contract!` (index notation over
`Network::contract`), use
[`symtn-network/examples/quickstart.rs`](symtn-network/examples/quickstart.rs):

```sh
cargo run -p symtn-network --example quickstart
```

## Operations and documentation

symtn provides `permute`, `braid`, `contract`, `trace_pairs`, `qr_compact`,
`svd_compact`, and other operations that an application can compose into its
own algorithm. The [symtn crate guide](symtn/README.md) maps these operations
to methods and links to the tutorial.

Supported operations depend on the symmetry provider, scalar type, and
storage placement. Host has the broadest operation set; CUDA uses explicit
`to_cuda()` and `to_host()` transfers and supports a smaller subset. See the
[documentation index](docs/README.md) for mathematical conventions, backend
capabilities, examples, and revision-specific evidence.

For development, read the [design](docs/design.md) and [coding agent rules](AGENTS.md).
