# TeNeT cotengra Python environment

This uv project is the reproducible Python environment for TeNeT's optional
`cotengra-python` planner backend.

Create or update the environment:

```sh
uv sync --project tools/cotengra-python
```

The environment pins `cotengra` plus `kahypar` (recommended hypergraph
partitioner) and `optuna` (Bayesian sampler). Both are required for the `hyper`
method to actually hyper-optimize — without them cotengra silently degrades to
the basic `labels` partitioner and purely random sampling.

Check the environment:

```sh
uv run --project tools/cotengra-python python -c \
  'import cotengra, kahypar, optuna; print(cotengra.__version__)'
```

Use it from Rust:

```rust
use tenet::plancache::{CotengraPythonConfig, Optimizer};

let optimizer = Optimizer::CotengraPython(
    CotengraPythonConfig::with_uv_project("/path/to/TeNeT/tools/cotengra-python"),
);
```

The opt-in integration test takes the project path from its own environment
(the library itself reads none):

```sh
TENET_COTENGRA_UV_PROJECT=$PWD/tools/cotengra-python TENET_RUN_COTENGRA_PYTHON_TEST=1 \
  cargo test -p tenet-network --features cotengra-python -- --ignored
```

The project path is passed to `uv` unchanged and resolved against the process
working directory, so prefer an absolute path.
