# tenet-network

The user-facing home of `tensor!` and explicit tensor-network planning.
`NetworkIR`, `Network`, `PlannedNetwork`, `ContractionPlan`, `Network::plan`,
optimizer traits, and slicing types let expert users inspect or reuse structural plans;
TeNeT executes the resulting plan locally.

The default host backend is `cpu-faer`; enable at least one host backend.
`opt-path` enables opt-einsum-path search and `cotengra-python` calls Python
cotengra for path search. `cuda` still requires a host backend, while
`racah-generated` forwards generated SUN support; see `Cargo.toml` for the
other provider and BLAS flags.

Run the [quickstart](examples/quickstart.rs) with:

```sh
cargo run -p tenet-network --example quickstart
```
