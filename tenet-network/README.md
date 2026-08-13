# tenet-network

The user-facing home of `tensor!` and explicit tensor-network planning.
`NetworkIR`, `Network`, `PlannedNetwork`, `ContractionPlan`, `plan`, optimizer
traits, and slicing types let expert users inspect or reuse structural plans;
TeNeT executes the resulting plan locally.

Features: `opt-path` enables opt-einsum-path search, `cotengra-python` calls
Python cotengra for path search, `cuda` enables the CUDA execution path, and
`racah-generated` forwards generated SUN support.

Run the [quickstart](examples/quickstart.rs) with:

```sh
cargo run -p tenet-network --example quickstart
```
