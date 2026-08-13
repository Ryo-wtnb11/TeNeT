# tenet

`tenet-rs` is TeNeT's public package; its library is imported as `tenet`.
Ordinary applications use `tenet::prelude::*` for `Runtime`, `RuntimeBuilder`,
`GradedSpace`, `TensorMap`, `LinalgBackend`, and `Truncation`.

Start with the [crate tutorial](src/tutorial.md#1-quick-start). For index
notation and network planning, add `tenet-network` and run its
[quickstart](https://github.com/Ryo-wtnb11/TeNeT/blob/main/tenet-network/examples/quickstart.rs).

The default host backend is `cpu-faer`; see the root
[feature table](https://github.com/Ryo-wtnb11/TeNeT#feature-flags) and
[`Cargo.toml`](Cargo.toml) for alternatives. `cuda` and `racah-generated` add
the typed CUDA surface and SUN providers. The `opt-path` and `cotengra-python`
facade markers expose optimizer configuration; the planners themselves are
enabled in `tenet-network`.
