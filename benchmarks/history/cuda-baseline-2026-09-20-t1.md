# CUDA record after removing the redundant transfer host copies, 2026-09-20 (T1)

Rerun of `benchmarks/cuda_operation_matrix.sh --device 0` on qg1 after #1284
gave the CUDA boundary an owned-buffer upload (`CudaDenseStorage::upload_owned`)
and made downloads return Tenferro's vector instead of copying out of it.
Protocol and column definitions: `benchmarks/cuda_operation_matrix.md`. This is
a measurement record, not a CI gate and not a threshold any code dispatches on.

## Comparison baseline

`cuda-baseline-2026-09-20-gl3.csv`, the record of the current merge base
`077b5b58` on the same host, driver, toolchain, cuTENSOR 2.5.0, `Cargo.lock`
(`lock_sha256=cad23363…`), features `cuda,cpu-faer`, release,
`CUDA_VISIBLE_DEVICES=0` and single-threaded BLAS/rayon. The two runs used
separate `CARGO_TARGET_DIR`s, so neither reused the other's example binary.

Raw data: `cuda-baseline-2026-09-20-t1.csv` (927 lines including the pinned
header comments; 896 data rows, identical row set to the reference). Build log:
`cuda-baseline-2026-09-20-t1.build.log`.

## Device counters and correctness: unchanged

Every device counter of all 896 rows (`h2d_calls`, `h2d_bytes`, `d2h_calls`,
`d2h_bytes`, `device_allocs`, `gemm_calls`, `solver_calls`, `copy_calls`,
`barrier_d2h_calls`, `barrier_d2h_bytes`) is identical to the reference, cell
for cell: **0 differing cells**. The `check` column is identical too, 448
`ok:host_value_equality`, 160 `ok:device_reconstruction`, 128
`ok:host_scalar_equality`, 64 `ok:host_spectrum_and_discard_weight`, 64
`ok:roundtrip_bit_equal`, 32 `no Host counterpart`. The change moves host
buffers; it submits the same device work, moves the same device bytes and
returns the same values.

## Host allocations: the subject

432 of 896 rows drop `alloc_calls` and/or `alloc_bytes`; **no row rises**.
Totals over all rows: `alloc_calls` 1 769 889 → 1 763 073 (-6 816, -0.39%),
`alloc_bytes` 1 826 436 809 → 1 779 313 161 (-47 123 648, -2.58%). Bytes fall
much further than calls because every eliminated copy is a whole payload while
the surviving allocations are mostly small bookkeeping.

| operation | phase | rows | alloc_calls base -> T1 | alloc_bytes base -> T1 |
|---|---|---|---|---|
| add_lazy_fold | cold | 16 | 2476 -> 2444 | 7595776 -> 6144640 |
| add_lazy_fold | warm | 16 | 2300 -> 2268 | 7374464 -> 5923328 |
| add_owned | cold | 16 | 2384 -> 2352 | 7579392 -> 6128256 |
| add_owned | warm | 16 | 2208 -> 2176 | 7358080 -> 5906944 |
| compose | cold | 16 | 23224 -> 23208 | 8900000 -> 7449248 |
| compose | warm | 16 | 20348 -> 20332 | 7902016 -> 6451264 |
| contract_direct | cold | 16 | 23192 -> 23176 | 8899744 -> 7448992 |
| contract_direct | warm | 16 | 20316 -> 20300 | 7901760 -> 6451008 |
| contract_lazy_adjoint_lhs | cold | 16 | 24024 -> 24008 | 8919712 -> 7468960 |
| contract_lazy_adjoint_lhs | warm | 16 | 21148 -> 21132 | 7921728 -> 6470976 |
| eigh_full | cold | 16 | 360853 -> 358741 | 69647560 -> 65258568 |
| eigh_full | warm | 16 | 357780 -> 355668 | 68646016 -> 64257024 |
| inner | cold | 16 | 23774 -> 23742 | 1615120 -> 1605136 |
| inner | warm | 16 | 20834 -> 20802 | 610800 -> 600816 |
| network_chain3 | cold | 16 | 51772 -> 51740 | 17899488 -> 14997984 |
| network_chain3 | warm | 16 | 40956 -> 40940 | 8649216 -> 7198464 |
| norm | cold | 16 | 23774 -> 23742 | 1615120 -> 1605136 |
| norm | warm | 16 | 20834 -> 20802 | 610800 -> 600816 |
| qr_compact | cold | 8 | 70963 -> 70947 | 13458912 -> 12491744 |
| qr_compact | warm | 8 | 69547 -> 69531 | 13072400 -> 12105232 |
| scale | cold | 16 | 1632 -> 1600 | 7558016 -> 6107072 |
| scale | warm | 16 | 1456 -> 1424 | 7336704 -> 5885760 |
| svd_compact | cold | 16 | 100399 -> 99935 | 31689236 -> 27310228 |
| svd_compact | warm | 16 | 97554 -> 97090 | 30915568 -> 26536560 |
| svd_trunc_rank | cold | 16 | 87837 -> 87269 | 19288470 -> 17814134 |
| svd_trunc_rank | warm | 16 | 79302 -> 78734 | 17455468 -> 15981132 |
| to_host | first_after_setup | 16 | 144 -> 128 | 2909824 -> 1459072 |
| to_host | warm | 16 | 144 -> 128 | 2909824 -> 1459072 |

`to_host` is the cleanest reading: one download per row, one payload copy
removed, exactly one `alloc_calls` per row and half the bytes. `eigh_full`,
`svd_compact` and `svd_trunc_rank` drop the most calls because they upload one
selector per kept route. `to_cuda` rows are *unchanged*, which is the other
half of the invariant: `to_cuda` borrows a live Host tensor, so it still pays
the one copy Tenferro's owned-host-tensor upload requires.

## Timing: informational

Warm device rows land within 0.77x-1.18x of the reference (median 0.97x, 232
rows). That is run-to-run scatter of this single-process, single-device
harness; a host `memcpy` removed from a path dominated by device submission is
not expected to move a timer, and this record claims no speedup.
