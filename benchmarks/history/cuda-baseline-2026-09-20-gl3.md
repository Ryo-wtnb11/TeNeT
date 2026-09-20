# CUDA record after the device-local execution lock, 2026-09-20 (GL-3)

Rerun of `benchmarks/cuda_operation_matrix.sh --device 0` on qg1 after #1281
moved the CUDA context out of `RuntimeState` into a device-local mutex on
`RuntimeInner`, so device operations take `Runtime::lease_cuda()` instead of
the coarse Runtime state lock. Protocol and column definitions:
`benchmarks/cuda_operation_matrix.md`. This is a measurement record, not a CI
gate and not a threshold any code dispatches on.

The counter columns are the subject: every device counter of all 896 data rows
(`h2d_calls`, `h2d_bytes`, `d2h_calls`, `d2h_bytes`, `device_allocs`,
`gemm_calls`, `solver_calls`, `copy_calls`, `barrier_d2h_calls`,
`barrier_d2h_bytes`) plus `alloc_calls`, `alloc_bytes` and the `check` verdict
is identical to the reference run, 0 differing cells, which is the invariant
GL-3 claims: the lock moved, no device work, transfer, allocation or result
did. The timing columns are informational only; warm device rows land within
0.90x-1.45x of the reference (median 1.06x), which is run-to-run scatter of
this single-process, single-device harness and supports no performance claim
in either direction. GL-3 makes no concurrency, overlap, multi-stream or
scaling claim: device operations still serialize on the device mutex and on
Tenferro's internal handle and plan locks.

## Comparison baseline

The reference is **not** `cuda-baseline-2026-09-20-gl2.csv`. That record was
measured at `d3d7fa4f` + the GL-2 diff, i.e. before #1279 landed, and #1279
(device destination reuse in the shared network workspace) legitimately changes
64 counter cells of the `network_chain3` rows (one fewer H2D upload and device
allocation, one D2D copy in the warm phase). Comparing GL-3 against it would
attribute a merged upstream change to this leaf.

The reference used here is therefore a same-host rerun of the exact merge base,
`ea4f3f5c`, recorded as `cuda-baseline-2026-09-20-gl3-base.csv` (build log
`…-gl3-base.build.log`). Both runs use the same host, driver, toolchain,
cuTENSOR 2.5.0, `Cargo.lock` (`lock_sha256=cad23363…`), features
`cuda,cpu-faer`, release, `CUDA_VISIBLE_DEVICES=0` and single-threaded
BLAS/rayon; the full pinned header is in each CSV. The two runs used separate
`CARGO_TARGET_DIR`s: with one shared target directory Cargo reuses the example
binary across two source checkouts of the same workspace, which silently
compares a revision against itself.

Raw data: `cuda-baseline-2026-09-20-gl3.csv` (927 lines including the pinned
header comments; 896 data rows, identical row set to the reference). Build log:
`cuda-baseline-2026-09-20-gl3.build.log`.

## Correctness verdicts

All 896 rows report the same verdicts as the reference, cell for cell: 448
`ok:host_value_equality`, 160 `ok:device_reconstruction`, 128
`ok:host_scalar_equality`, 64 `ok:host_spectrum_and_discard_weight`, 64
`ok:roundtrip_bit_equal`, 32 `no Host counterpart`. Zero mismatches, zero
unexpected skips.
