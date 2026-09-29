# tenet

`tenet-rs` is TeNeT's public package; its library is imported as `tenet`.
Ordinary applications import `Runtime`, `RuntimeBuilder`, `GradedSpace`,
`TensorMap`, `LinalgBackend`, and `Truncation` from `tenet::typed`, and the
built-in symmetries such as `U1FusionRule` from `tenet::sector`. Every public
item has exactly one path.

Start with the [crate tutorial](src/tutorial.md#quick-start). The main
`TensorMap` operations are:

| Task | Methods |
| --- | --- |
| Construct | `from_subblock_fn`, `zeros`, `isomorphism`, `rand_with_seed` |
| Reorder and orient legs | `permute`, `braid`, `repartition`, `adjoint` |
| Combine | `compose`, `contract` with `ContractSpec` |
| Reduce and inspect | `trace_pairs`, `inner`, `norm`, `blocks`, `subblocks` |
| Factorize | `qr_compact`, `svd_compact`, `eigh_full` |

For index notation and network planning, add `tenet-network` and run its
[quickstart](../tenet-network/examples/quickstart.rs). Mathematical conventions
are in [tensor-map mathematics](src/mathematics.md).

The default host provider is Tenferro's resolved compiled default: BLAS when
its CPU build enables `cpu-blas`, otherwise `cpu-faer`; see the
[backend policy](../docs/backend_policy.md) and [`Cargo.toml`](Cargo.toml) for
alternatives. `cuda` and `racah-generated` add
the typed CUDA surface and SUN providers. The `opt-path` and `cotengra-python`
facade markers expose optimizer configuration; the planners themselves are
enabled in `tenet-network`.
