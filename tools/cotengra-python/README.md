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
use tenet::prelude::{CotengraPythonConfig, Optimizer};

let optimizer = Optimizer::CotengraPython(
    CotengraPythonConfig::with_uv_project("tools/cotengra-python"),
);
```

Or set the project path without changing Rust code:

```sh
TENET_COTENGRA_UV_PROJECT=tools/cotengra-python cargo test -p tenet-network --features cotengra-python
```

Relative `tools/cotengra-python` paths are resolved against the current working
directory first, then against the TeNeT workspace source tree. For long-running
applications outside this workspace, an absolute path is still the clearest
choice.
