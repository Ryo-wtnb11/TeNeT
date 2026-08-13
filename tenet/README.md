# tenet

`tenet-rs` is TeNeT's public package; its library is imported as `tenet`.
Ordinary applications use `tenet::prelude::*` for `Runtime`, `RuntimeBuilder`,
`GradedSpace`, `TensorMap`, `LinalgBackend`, and `Truncation`.

Start with the [crate tutorial](src/tutorial.md#1-quick-start). For index
notation and network planning, add `tenet-network` and run its
[quickstart](../tenet-network/examples/quickstart.rs).

Crate-specific features are `cuda` for the typed CUDA surface,
`racah-generated` for SUN providers, and `cotengra-python` for Python
cotengra path search.
