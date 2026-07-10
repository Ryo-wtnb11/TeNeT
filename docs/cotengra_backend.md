# cotengra Python contraction-path backend

An **optional**, feature-gated path planner that shells out to the installed
Python [`cotengra`](https://github.com/jcmgray/cotengra) package for
contraction-order search (greedy / optimal / auto-hq / **hyper**) and slicing
decisions (slice / reconfigure / forest-reconfigure). The pure-Rust
`opt-einsum-path` optimizers remain the default; cotengra adds the
hyper-optimization tier (KaHyPar hypergraph partitioning + Bayesian
hyper-parameter search) that has no pure-Rust equivalent yet.

## Why a Python subprocess

Path planning is a coarse-grained, one-shot operation — input is an einsum
spec plus leg dimensions, output is a pairwise path (and an optional slicing).
It is off the hot per-contraction execution path and is cached per topology
(`ReplanPolicy::BakeOnce`), so the subprocess boundary costs little (see
[Latency](#latency)) while avoiding a reimplementation of cotengra's
hyper-optimizer. The bridge is JSON over stdin/stdout; nothing is linked into
the Rust process. `cotengrust` (a Rust cotengra port) is **AGPL**, so it is not
vendored — the Python package is called instead.

## Setup

The reproducible Python environment is the [uv](https://docs.astral.sh/uv/)
project under `tools/cotengra-python`:

```sh
uv sync --project tools/cotengra-python
```

Dependencies (`tools/cotengra-python/pyproject.toml`):

| package | role | without it |
|---------|------|-----------|
| `cotengra` | the path optimizer | — |
| `kahypar` | recommended hypergraph partitioner | `hyper` silently falls back to the basic `labels` partitioner |
| `optuna` | Bayesian hyper-parameter sampling | `hyper` silently falls back to purely random sampling |

`kahypar` and `optuna` are what make the `hyper` method actually
hyper-optimize; both ship as wheels (no C++ build on macOS/Linux). Verify:

```sh
uv run --project tools/cotengra-python python -c \
  'import cotengra, kahypar, optuna; print(cotengra.__version__)'
```

## Enabling it in a crate

Turn on the `cotengra-python` feature on `tenet` and `tenet-network`:

```toml
tenet         = { path = "...", features = ["cotengra-python"] }
tenet-network = { path = "...", features = ["cotengra-python"] }
```

## Using it from Rust

```rust
use tenet::prelude::{CotengraPythonConfig, Optimizer, PlanCacheConfig};

let optimizer = Optimizer::CotengraPython(
    CotengraPythonConfig::with_uv_project("tools/cotengra-python"),
);
let rt = Runtime::builder()
    .plan_cache(PlanCacheConfig { optimizer, ..Default::default() })
    .build()?;
```

The relative `tools/cotengra-python` path is resolved against the current
working directory first, then against `tenet-network`'s `CARGO_MANIFEST_DIR`
parent — so it works regardless of the caller's CWD (a downstream crate does
**not** need an absolute path). Overrides, in priority order:

- `CotengraPythonConfig::python(program)` or `TENET_COTENGRA_PYTHON` — run a
  specific interpreter directly.
- `TENET_COTENGRA_UV_PROJECT=<path>` — use `uv run --project <path> python`.
- otherwise `python3` on `PATH` (must have cotengra importable).

`CotengraPythonConfig` selects the `method` (`Greedy` / `Optimal` / `AutoHq` /
`Hyper`), `minimize` (flops / size / …), `max_repeats`, `seed`, and a
`slicing` config (none / slice / reconfigure / forest-reconfigure with
`target_size` / `step_size` / `max_repeats` / `allow_outer`).

## Latency

Measured on a 16-tensor periodic grid (warm uv cache), per planner call:

| step | time |
|------|------|
| subprocess + `uv run` + `import cotengra` | ~50 ms |
| greedy / auto search | sub-ms |
| auto-hq search | ~45 ms |
| hyper search (full kahypar + optuna, `max_repeats`≈64) | ~1 s |

So a full call is ~50 ms for greedy, ~100 ms for auto-hq, ~1 s for hyper. Every
call is amortized to **once per topology** by `ReplanPolicy::BakeOnce`. The
`hyper` cost is the real hyper-optimization price and pays off only on large /
hard networks where greedy is suboptimal; for the small local contractions of a
boundary method, greedy / auto-hq (or the pure-Rust `opt-einsum-path` drivers)
are the right choice. A persistent Python worker to remove the ~50 ms spawn is
**not** worth it at this latency.

## Limitations

- **Deployment**: opting in requires a Python environment (uv + cotengra)
  alongside the Rust binary. The pure-Rust `opt-einsum-path` default has no such
  requirement; a fully self-contained Rust `hyper` would need the AGPL-avoiding
  `tenet-cotengrust` port (roadmap TODO).
- **Sliced execution is not connected**: the backend *decides* a slicing
  (`SlicedPlan`, internal=summed / output=stacked) but the ordinary tensor
  executor does not yet execute the slices — path planning only. See
  [issue #93](https://github.com/Ryo-wtnb11/TeNeT/issues/93).
