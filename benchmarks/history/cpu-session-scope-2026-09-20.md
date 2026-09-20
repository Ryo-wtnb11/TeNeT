# CPU session scope — one Tenferro session per dense phase (#1283, S1a)

Date 2026-09-20. Host: Apple M4 Max, Darwin 24.5.0 arm64, rustc 1.96.0 /
cargo 1.96.0. Provider: `cpu-faer` unless a row says Accelerate.
Baseline revision `077b5b58` ("Give device operations a device-local lock"),
candidate `55eb6ca9` ("dense: hold one CPU session per dense phase").
Separate target directories per revision; the two `operation_matrix` binaries
hash differently:

```
59e5c655ce9926918683d191ffdbcb63d00cfc6d4bf45383f930a37957f7a441  base   operation_matrix
58e6518119b9f5875b5b41fb01c00bae5d004d57a3ea8c027b77d744294b6712  cand   operation_matrix
```

Cargo lock `cad23363c997e3895a5391be45ee97fb3cfe19ca049c53593bbc8f60580f59b2`,
features `cpu-faer,racah-generated`, `--locked --offline --release`.
Raw medians: `cpu-session-scope-2026-09-20-baseline.csv` /
`-candidate.csv` (one `benchmarks/operation_matrix.sh` run each,
`OP_MATRIX_OPERATION=oriented_uniform_run`).

## Which rows reach the changed code

`matmul_batch_axpby_ops_serial_typed` (`tenet-dense/src/tenferro_adapter.rs`)
is reached only through `matmul_batch_axpby_with_ops_into` with a non-Identity
operand op. The production path is
`tenet-operations/src/fusion_replay.rs:945-958` →
`tenet-tensors/src/contract/fusion_block.rs:296-315` →
`tenet-tensors/src/contract/backend.rs:481-517` →
`tenet-dense/src/tenferro_adapter.rs:1159` → `:360` → `:433`. The op comes from
operand orientation (`fusion_block.rs:1717-1719`, `:1741-1743`:
`Adjoint => MatrixOp::Adjoint`), never from shapes, and `MatrixOp::Transpose`
is never produced by the tensor layer.

Of the rows this benchmark drives, only two reach the serial route, both in
`DenseAdapter / oriented_uniform_run` (`tenet/examples/operation_matrix.rs:476`):

- `heterogeneous_c64_AI` — 8 alternating-shape jobs, `runs = [1; 8]`, so the
  equal-count branch at `:378` sends all 8 jobs through the serial route.
- `singleton_c64_AI` — 1 job, same branch.

The public `U1Public` rows (`direct_compose`, `lazy_lhs_compose`,
`lazy_lhs_contract`, `*_shape_cycle`) do not: their fixture is uniform, so
`strided_batch_runs` returns one long run and the strided-batch route at `:406`
takes it. No new benchmark row was added — the coverage already existed.

## Session counts per call (Tenferro CPU sessions entered)

| site | before | after |
|---|---:|---:|
| op-bearing serial batch, `heterogeneous_c64_AI` (8 jobs) | 8 | 1 |
| op-bearing serial batch, `singleton_c64_AI` (1 job) | 1 | 1 |
| `eig_vals` | 2 | 1 |
| grouped / strided batch routes, all other adapter rows | 1 | 1 |

Before, each session was opened by Tenferro inside the backend-level dot;
after, TeNeT opens one itself and issues the jobs through it. The new
`cpu_session_stats().sessions_opened` counts only the TeNeT-side entries, so it
reads 0 on the baseline for these rows and 1 on the candidate — the tests in
`tenet-dense/tests/cpu_session_scope.rs` assert the `1`.

## Host operation matrix

### RAYON_NUM_THREADS=1 (benchmarks/operation_matrix.sh, 3 script runs x 3 process samples)

| form | phase | base us/iter | candidate us/iter | delta |
|---|---|---:|---:|---:|
| few_large_c64_AA | first_fresh_executor_after_preflight | 130.334 | 129.083 | -1.0% |
| few_large_c64_AA | warm_fixed | 126.499 | 128.222 | +1.4% |
| few_large_c64_AI | first_fresh_executor_after_preflight | 128.291 | 127.417 | -0.7% |
| few_large_c64_AI | warm_fixed | 126.710 | 127.618 | +0.7% |
| few_large_c64_AI_shape_cycle | warm_shape_cycle | 124.299 | 124.203 | -0.1% |
| few_large_c64_II | first_fresh_executor_after_preflight | 127.084 | 126.250 | -0.7% |
| few_large_c64_II | warm_fixed | 125.345 | 125.599 | +0.2% |
| few_large_f64_II | first_fresh_executor_after_preflight | 32.167 | 34.625 | +7.6% |
| few_large_f64_II | warm_fixed | 30.795 | 31.174 | +1.2% |
| few_large_f64_TI | first_fresh_executor_after_preflight | 32.208 | 32.666 | +1.4% |
| few_large_f64_TI | warm_fixed | 31.745 | 31.749 | +0.0% |
| few_large_f64_TI_shape_cycle | warm_shape_cycle | 31.399 | 31.241 | -0.5% |
| heterogeneous_c64_AI | first_fresh_executor_after_preflight | 11.833 | 9.000 | -23.9% |
| heterogeneous_c64_AI | warm_fixed | 8.542 | 5.688 | -33.4% |
| many_small_c64_AA | first_fresh_executor_after_preflight | 4.042 | 4.041 | -0.0% |
| many_small_c64_AA | warm_fixed | 3.022 | 3.221 | +6.6% |
| many_small_c64_AI | first_fresh_executor_after_preflight | 4.083 | 4.333 | +6.1% |
| many_small_c64_AI | warm_fixed | 3.063 | 3.130 | +2.2% |
| many_small_c64_AI_shape_cycle | warm_shape_cycle | 3.709 | 3.783 | +2.0% |
| many_small_c64_II | first_fresh_executor_after_preflight | 3.458 | 3.625 | +4.8% |
| many_small_c64_II | warm_fixed | 2.139 | 2.176 | +1.7% |
| many_small_f64_II | first_fresh_executor_after_preflight | 4.333 | 4.375 | +1.0% |
| many_small_f64_II | warm_fixed | 1.868 | 1.832 | -1.9% |
| many_small_f64_TI | first_fresh_executor_after_preflight | 3.375 | 3.583 | +6.2% |
| many_small_f64_TI | warm_fixed | 2.324 | 2.341 | +0.7% |
| many_small_f64_TI_shape_cycle | warm_shape_cycle | 2.813 | 2.836 | +0.8% |
| minimum_run_c64_AI | first_fresh_executor_after_preflight | 2.125 | 1.959 | -7.8% |
| minimum_run_c64_AI | warm_fixed | 1.232 | 1.236 | +0.3% |
| minimum_run_f64_TI | first_fresh_executor_after_preflight | 2.084 | 2.167 | +4.0% |
| minimum_run_f64_TI | warm_fixed | 1.176 | 1.185 | +0.8% |
| singleton_c64_AI | first_fresh_executor_after_preflight | 36.125 | 37.792 | +4.6% |
| singleton_c64_AI | warm_fixed | 32.708 | 32.223 | -1.5% |

### default threads (binaries run directly, 3 runs x 3 process samples)

| form | phase | base us/iter | candidate us/iter | delta |
|---|---|---:|---:|---:|
| few_large_c64_AA | first_fresh_executor_after_preflight | 127.250 | 129.708 | +1.9% |
| few_large_c64_AA | warm_fixed | 126.358 | 128.838 | +2.0% |
| few_large_c64_AI | first_fresh_executor_after_preflight | 129.125 | 128.834 | -0.2% |
| few_large_c64_AI | warm_fixed | 128.519 | 127.427 | -0.8% |
| few_large_c64_AI_shape_cycle | warm_shape_cycle | 125.843 | 126.395 | +0.4% |
| few_large_c64_II | first_fresh_executor_after_preflight | 131.625 | 127.291 | -3.3% |
| few_large_c64_II | warm_fixed | 129.090 | 127.430 | -1.3% |
| few_large_f64_II | first_fresh_executor_after_preflight | 33.125 | 32.958 | -0.5% |
| few_large_f64_II | warm_fixed | 31.048 | 31.464 | +1.3% |
| few_large_f64_TI | first_fresh_executor_after_preflight | 34.000 | 33.417 | -1.7% |
| few_large_f64_TI | warm_fixed | 32.163 | 32.150 | -0.0% |
| few_large_f64_TI_shape_cycle | warm_shape_cycle | 32.238 | 31.812 | -1.3% |
| heterogeneous_c64_AI | first_fresh_executor_after_preflight | 11.916 | 8.750 | -26.6% |
| heterogeneous_c64_AI | warm_fixed | 8.820 | 5.741 | -34.9% |
| many_small_c64_AA | first_fresh_executor_after_preflight | 4.125 | 4.333 | +5.0% |
| many_small_c64_AA | warm_fixed | 3.144 | 3.178 | +1.1% |
| many_small_c64_AI | first_fresh_executor_after_preflight | 4.500 | 4.084 | -9.2% |
| many_small_c64_AI | warm_fixed | 3.141 | 3.125 | -0.5% |
| many_small_c64_AI_shape_cycle | warm_shape_cycle | 3.884 | 3.773 | -2.9% |
| many_small_c64_II | first_fresh_executor_after_preflight | 3.500 | 3.709 | +6.0% |
| many_small_c64_II | warm_fixed | 2.182 | 2.186 | +0.2% |
| many_small_f64_II | first_fresh_executor_after_preflight | 3.958 | 3.917 | -1.0% |
| many_small_f64_II | warm_fixed | 1.840 | 1.859 | +1.0% |
| many_small_f64_TI | first_fresh_executor_after_preflight | 3.500 | 3.709 | +6.0% |
| many_small_f64_TI | warm_fixed | 2.357 | 2.363 | +0.3% |
| many_small_f64_TI_shape_cycle | warm_shape_cycle | 2.962 | 2.871 | -3.1% |
| minimum_run_c64_AI | first_fresh_executor_after_preflight | 2.417 | 2.084 | -13.8% |
| minimum_run_c64_AI | warm_fixed | 1.305 | 1.308 | +0.2% |
| minimum_run_f64_TI | first_fresh_executor_after_preflight | 2.334 | 2.375 | +1.8% |
| minimum_run_f64_TI | warm_fixed | 1.212 | 1.220 | +0.7% |
| singleton_c64_AI | first_fresh_executor_after_preflight | 35.417 | 34.459 | -2.7% |
| singleton_c64_AI | warm_fixed | 34.565 | 32.588 | -5.7% |

`heterogeneous_c64_AI` is the only row on the changed path and is the only one
that moves outside run-to-run spread: −33% (threads=1) and −35% (default
threads) on `warm_fixed`, consistent across all three runs. `singleton_c64_AI`
is unchanged, as expected — one job was already one session.

Non-improvements, stated plainly: several rows that cannot reach the changed
code drift by a few percent in both directions. `many_small_c64_AA warm_fixed`
is consistently +6.6% at threads=1 (3.05/2.99/3.02 → 3.22/3.18/3.24) while
being neutral (+1.1%) at default threads. That row takes the strided-batch
route, which this change does not touch; it is unexplained binary-layout or
machine drift and is not claimed as a cost of the change. The
`first_fresh_executor_after_preflight` rows are single-iteration samples and
are not evidence in either direction.

## Concurrency and BLAS

A CPU session holds a process-global execution permit for its whole scope, so a
longer scope must be measured with more than one thread. Temporary harness
(`tenet-dense/examples/s1a_concurrent.rs`, not committed): N threads, one
`DefaultDenseExecutor` each, all issuing the same 8-job adjoint batch,
60000 iterations (20000 for Accelerate); median of 3 runs, µs per iteration per
thread.

| provider / threads env | threads | base | candidate | delta |
|---|---:|---:|---:|---:|
| faer, `RAYON_NUM_THREADS=1` | 1 | 9.44 | 6.01 | −36% |
| faer, `RAYON_NUM_THREADS=1` | 2 | 67.7 | 17.8 | −74% |
| faer, default threads | 1 | 129.1 | 16.9 | −87% |
| faer, default threads | 2 | 316.1 | 43.9 | −86% |
| faer, default threads | 4 | 947.0 | 118.9 | −87% |
| Accelerate BLAS | 1 | 132.5 | 134.4 | +1.5% |
| Accelerate BLAS | 2 | 316.6 | 290.6 | −8% |

Both revisions scale worse than linearly with threads — the permit serializes
execution either way — but the candidate is uniformly faster, so the longer
scope does not make contention worse here. Under faer with default threads the
per-session Rayon pool install dominates, which is why the gap is largest
there; note that this harness lets faer use all workers, while the
operation-matrix rows above pin `RAYON_NUM_THREADS=1`.

Accelerate is neutral at one thread, as the design predicted: BLAS is a
`ProviderDefaultExclusive` resolution, so `run_backend_session_cached` skips
`enter_managed_session` and there is no Rayon handoff to amortize — only the
permit and the engine mutex. The −8% at two threads is a single measurement of
one fixture and is not a general claim.

## Not measured

qg1 CPU (this run is the Mac only), OpenBLAS/MKL, and any CUDA row (S1a is
CPU-only by design F1).
