# CUDA baseline after the build-time backend warm-up, 2026-09-20 (GL-2)

Rerun of `benchmarks/cuda_operation_matrix.sh` on qg1 after #1278 moved the
tenferro backend library initialization (cuTENSOR `dlopen` + handle + plan,
cuSOLVER/cuBLAS `dlopen` + handles) from the first device submission to
`Runtime` construction. Protocol and column definitions:
`benchmarks/cuda_operation_matrix.md`. Comparison baseline:
`cuda-baseline-2026-09-20.md` (TeNeT `709f9e0c` + the GB harness).

This is a measurement record, not a CI gate and not a threshold any code
dispatches on.

Raw data: `cuda-baseline-2026-09-20-gl2.csv` (927 lines, header lines
included; 897 data rows, identical row set to the baseline). Build log:
`cuda-baseline-2026-09-20-gl2.build.log`.

## Authority and environment

`tenet_sha=d3d7fa4f+gl2-cuda-warm-up-at-build` — TeNeT `origin/main` at
`d3d7fa4fe...` plus this branch's diff. Same host, driver, toolchain, cuTENSOR
2.5.0, `Cargo.lock` (`lock_sha256=cad23363...`), features `cuda,cpu-faer`,
release, `CUDA_VISIBLE_DEVICES=0`, single-threaded BLAS/rayon as in the
baseline; the full pinned header is in the CSV. No other GPU work ran on the
device during the measurement.

All 897 rows report the same correctness verdicts as the baseline, cell for
cell by category: 448 `ok:host_value_equality`, 160 `ok:device_reconstruction`,
128 `ok:host_scalar_equality`, 64 `ok:host_spectrum_and_discard_weight`, 64
`ok:roundtrip_bit_equal`, 32 `no Host counterpart`. Zero mismatches, zero
unexpected skips.

## Runtime construction: where the cost went

New header line in the CSV:

```
# runtime_build_ns first=822545620 second=193163066
```

Two single observations, not medians, both in this one process:

- `first` = 823 ms: the process's first CUDA `Runtime`. It pays `cuInit`,
  primary-context retain, the CubeCL server start **and** the new warm-up.
- `second` = 193 ms: a second fresh `Runtime` in the same process. CUDA context
  and CubeCL server are warm; the tenferro `CudaBackend` — and therefore the
  cuTENSOR and cuSOLVER/cuBLAS handles — is new, so this 193 ms is essentially
  the warm-up alone.

193 ms matches the 185–657 ms that every first call used to pay, which is the
point: the cost is unavoidable and is now attributed to construction instead of
to whichever user operation happened to be first.

## First call on a fresh Runtime: before and after

Medians over all 200 device `cold` rows:

| set | before | after |
|---|---:|---:|
| all device `cold` rows (n=200) | min 185.0 ms, median 200.4 ms, max 657.4 ms | min 0.3 ms, median 2.6 ms, max 474.5 ms |
| non-factorization rows (n=144) | min 185.0 ms, median 197.3 ms, max 227.3 ms | min 0.3 ms, median 2.0 ms, max 34.2 ms |
| factorization rows (n=56) | min 206.8 ms, median 239.6 ms, max 657.4 ms | min 5.2 ms, median 42.3 ms, max 474.5 ms |

The flat ~200 ms floor is gone from every row.

Requested rows (`f64`, both fixture families; `warm` is the after-run's warm
per-iteration median, for scale):

| fixture | operation | first call before (ms) | first call after (ms) | warm after (ms) |
|---|---|---:|---:|---:|
| U1 f64 many-small | contract_direct | 185.0 | 2.4 | 1.774 |
| U1 f64 many-small | network_chain3 | 200.2 | 4.1 | 3.269 |
| U1 f64 many-small | eigh_full | 657.4 | 474.5 | 61.568 |
| U1 f64 many-small | qr_compact | 411.7 | 248.1 | 38.757 |
| U1 f64 few-large | contract_direct | 216.6 | 20.1 | 0.219 |
| U1 f64 few-large | network_chain3 | 202.2 | 17.4 | 0.483 |
| U1 f64 few-large | eigh_full | 485.2 | 387.8 | 7.880 |
| U1 f64 few-large | qr_compact | 222.6 | 24.4 | 3.010 |
| fZ2 f64 many-small | contract_direct | 189.4 | 0.6 | 0.074 |
| fZ2 f64 many-small | network_chain3 | 189.0 | 0.7 | 0.169 |
| fZ2 f64 many-small | eigh_full | 214.1 | 11.5 | 1.558 |
| fZ2 f64 many-small | qr_compact | 228.4 | 25.6 | 0.957 |
| fZ2 f64 few-large | contract_direct | 209.2 | 17.6 | 0.134 |
| fZ2 f64 few-large | network_chain3 | 202.1 | 17.4 | 0.252 |
| fZ2 f64 few-large | eigh_full | 232.8 | 31.2 | 3.994 |
| fZ2 f64 few-large | qr_compact | 225.9 | 22.9 | 1.898 |
| SU2 f64 many-small | contract_direct | 194.7 | 2.1 | 1.708 |
| SU2 f64 many-small | network_chain3 | 199.9 | 4.1 | 3.274 |
| SU2 f64 many-small | eigh_full | 258.2 | 51.7 | 41.172 |
| SU2 f64 many-small | qr_compact | 251.5 | 47.1 | 31.230 |
| SU2 f64 few-large | contract_direct | 212.9 | 17.8 | 0.184 |
| SU2 f64 few-large | network_chain3 | 202.4 | 20.1 | 0.372 |
| SU2 f64 few-large | eigh_full | 239.6 | 35.1 | 7.595 |
| SU2 f64 few-large | qr_compact | 225.7 | 22.9 | 2.569 |

Reading the numbers, plainly:

- `contract_direct` and `network_chain3` first calls collapse from ~190–220 ms
  to 0.6–20 ms. What is left is CubeCL's NVRTC compile of the kernel families
  the row first touches plus the cuTENSOR plan for its shape, both of which the
  warm-up deliberately does not cover (a 1×1 warm-up plan does not fit a
  64 × 64 contraction key).
- `eigh_full` and `qr_compact` drop but stay expensive on the rows whose *warm*
  cost is itself tens of ms (U1 many-small `eigh_full`: 474.5 ms first call
  against a 61.6 ms warm median; SU2 many-small `qr_compact`: 47.1 ms against
  31.2 ms warm). What remains there is per-shape cuSOLVER work and kernel JIT
  proportional to the block count, not handle creation: the handles are warm
  before the fixture is built. The largest single improvement in the table is
  U1 many-small `eigh_full`, 657.4 → 474.5 ms.
- No row's first call regressed beyond its warm cost. The warm-up runs once per
  `Runtime`, not per operation, so the warm columns should be unaffected; they
  are not identical, and the differences are reported rather than smoothed
  over. Of the 232 warm device rows, 24 differ from the baseline by more than
  20 % and 11 by more than 30 %:

  | row | before | after | delta |
  |---|---:|---:|---:|
  | c64 few-large `to_cuda` (U1, U1xfZ2, SU2) | 47.7-47.9 us | 98.0-100.2 us | +105 to +110 % |
  | fZ2 c64 few-large `to_cuda` | 26.2 us | 54.8 us | +109 % |
  | c64 many-small `to_cuda` (U1, U1xfZ2, SU2) | 9.1-9.4 us | 14.6-15.9 us | +55 to +74 % |
  | c64 few-large `scale` (U1, U1xfZ2) | 123.1-135.3 us | 179.1 us | +32 to +46 % |
  | SU2 c64 few-large `inner` / `norm` | 187.3-193.1 us | 127.2-129.1 us | -32 to -33 % |

  The remaining 13 sit between 20 % and 25 % and are mostly SU2 few-large rows
  moving in the faster direction. None of this has a code path behind it: the
  change touches `Runtime` construction only and nothing in transfer,
  elementwise or reduction execution. Every device counter in the matrix is
  byte-for-byte identical to the baseline — all 232 warm and 200 cold rows
  agree on `h2d_*`, `d2h_*`, `device_allocs`, `gemm_calls`, `solver_calls` and
  `copy_calls` — so these rows do the same work as before. They are
  tens-of-microseconds medians from a single process with no between-process
  variance estimate, so read them as run-to-run variation of this harness, not
  as an effect of this commit.

## What is still first-use cost

- **CubeCL NVRTC compilation** per distinct kernel, per process per device. The
  warm-up does not enumerate kernel families, and TeNeT does not own that
  cache. CubeCL's PTX disk cache is off by default and is enabled through
  CubeCL's own `cubecl.toml` (`[compilation] cache`) — Tenferro/CubeCL
  configuration, not a TeNeT knob.
- **cuTENSOR plan construction** per contraction key. Only the 1×1 warm-up plan
  is built at construction.
- The tenferro extension cache that holds the cuSOLVER/cuBLAS handles is an LRU
  (16 entries / 64 MiB): a long-running process that evicts them can pay their
  creation again. Nothing in TeNeT pins them.

## Measurement limits

Unchanged from the baseline record: one process per invocation (no
between-process median), no reachable device synchronization, no device-memory
high-water mark, and `runtime_build_ns` is two single observations rather than
a distribution.
