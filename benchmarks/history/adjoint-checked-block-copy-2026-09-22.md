# Lazy-adjoint add and materialization use the checked strided block copy (#1399)

Date 2026-09-22. Host: Apple M4 Max, macOS 15.5, rustc 1.96.0.

- **Before:** `origin/main` `ed2aa606` plus this change's two new E1 rows only.
- **After:** #1399.
- **Build:** one freshly resolved Cargo lock (`7c32d6910a6fbc7e…`,
  `strided-kernel` 0.4.1) for both. Built one after the other into separate
  target directories with the build line of `benchmarks/eager_overhead_ledger.sh`.
- **Binaries (sha256 prefix):** before `938ca051ecd9c315`, after `24872a7ba7aaf192`.

Method: the E1 example with filters `add`, `add_adjoint` and `adjoint_data`,
`LEDGER_THREADS=one`, two passes in the order before, after, before, after, on
a quiet machine. The full ledger script was not rerun: no other E1 row reaches
the changed code.

- The existing `add` row is owned + owned and never reached the changed path.
  It is listed in the raw CSVs only as a noise reference (0.86–1.19×, same code).
- `add_adjoint` is new: `a.adjoint().add(&b, 1, 1)` with `b` owned on the adjoint
  space, so both passes (`Copy`, then `Axpy{1}`) run.
- `adjoint_data` is new: a fresh `a.adjoint()` and its first `data()`, so it
  includes the view's metadata admission.
- Raw pass-1 rows: `adjoint-checked-block-copy-2026-09-22-{before,after}.csv`.
- Timings are observations only. The change rests on its structure.

## Structural change

**Before.** `oriented_fusion_add_into` and `materialize_adjoint_data_dyn` ran
each block through `tensoradd_raw_strided_kernel_mapped`, which computes two
checked offsets and two slice bounds checks per element and never fuses axes.

**After.** Both build the block's `isize` strides once (`CheckedBlockAxes`,
extent-one axes dropped) and call
`StridedHostKernelAdapter::tensoradd_strided_checked`: bounds validated once,
layout normalized once, then the fused span walk. The element action is still
`raw_strided_action(alpha, beta)`, so results are bit-identical. The mapped
kernel is deleted.

## Allocation

Identical in every row: `add_adjoint` 3 calls, `adjoint_data` 8 calls, same
bytes. Pinned by `tenet/tests/adjoint_view_allocations.rs`
(`lazy_add_allocation_count_is_pinned_across_block_counts`: 3 calls for one to
many blocks; `first_lazy_materialization_allocates_once_per_payload_not_per_block`:
3 calls).


### add_adjoint

| sym | dtype | case | elements | before µs (pass 1 / 2) | after µs (pass 1 / 2) | after/before (min) | alloc calls/bytes before → after |
|---|---|---|---:|---:|---:|---:|---|
| U1 | c64 | r2_s8_d16 | 2048 | 7.31 / 8.17 | 4.10 / 4.07 | 0.56 | 3/32976 → 3/32976 |
| U1 | c64 | r2_s8_d2 | 32 | 1.81 / 1.78 | 1.82 / 1.88 | 1.02 | 3/720 → 3/720 |
| U1 | c64 | r3_s4_d4 | 768 | 7.23 / 7.19 | 4.85 / 4.81 | 0.67 | 3/12496 → 3/12496 |
| U1 | c64 | r4_s3_d4 | 4864 | 32.62 / 32.17 | 12.71 / 11.46 | 0.36 | 3/78032 → 3/78032 |
| U1 | c64 | r5_s2_d2 | 320 | 6.81 / 6.67 | 4.17 / 4.54 | 0.63 | 3/5328 → 3/5328 |
| U1 | f64 | r2_s8_d16 | 2048 | 7.02 / 6.44 | 2.85 / 2.66 | 0.41 | 3/16592 → 3/16592 |
| U1 | f64 | r2_s8_d2 | 32 | 1.58 / 1.74 | 1.98 / 2.05 | 1.25 | 3/464 → 3/464 |
| U1 | f64 | r3_s4_d4 | 768 | 6.31 / 6.90 | 3.93 / 4.35 | 0.62 | 3/6352 → 3/6352 |
| U1 | f64 | r4_s3_d4 | 4864 | 28.00 / 27.42 | 9.79 / 8.79 | 0.32 | 3/39120 → 3/39120 |
| U1 | f64 | r5_s2_d2 | 320 | 5.88 / 6.60 | 4.01 / 4.44 | 0.68 | 3/2768 → 3/2768 |
| fZ2xU1 | c64 | r2_s8_d16 | 2048 | 7.71 / 6.94 | 4.07 / 4.07 | 0.59 | 3/32976 → 3/32976 |
| fZ2xU1 | c64 | r2_s8_d2 | 32 | 1.78 / 1.62 | 1.83 / 2.06 | 1.13 | 3/720 → 3/720 |
| fZ2xU1 | c64 | r3_s4_d4 | 768 | 6.44 / 6.62 | 4.40 / 4.38 | 0.68 | 3/12496 → 3/12496 |
| fZ2xU1 | c64 | r4_s3_d4 | 4864 | 30.00 / 30.12 | 11.50 / 12.71 | 0.38 | 3/78032 → 3/78032 |
| fZ2xU1 | c64 | r5_s2_d2 | 320 | 6.75 / 6.73 | 4.61 / 4.21 | 0.63 | 3/5328 → 3/5328 |
| fZ2xU1 | f64 | r2_s8_d16 | 2048 | 6.98 / 7.00 | 2.66 / 2.81 | 0.38 | 3/16592 → 3/16592 |
| fZ2xU1 | f64 | r2_s8_d2 | 32 | 1.58 / 1.58 | 1.98 / 1.82 | 1.16 | 3/464 → 3/464 |
| fZ2xU1 | f64 | r3_s4_d4 | 768 | 7.21 / 6.46 | 3.96 / 4.33 | 0.61 | 3/6352 → 3/6352 |
| fZ2xU1 | f64 | r4_s3_d4 | 4864 | 26.96 / 26.83 | 9.00 / 9.88 | 0.34 | 3/39120 → 3/39120 |
| fZ2xU1 | f64 | r5_s2_d2 | 320 | 6.56 / 5.92 | 4.12 / 3.99 | 0.67 | 3/2768 → 3/2768 |
| SU2 | c64 | r2_s8_d16 | 2048 | 7.73 / 7.69 | 3.67 / 4.19 | 0.48 | 3/32976 → 3/32976 |
| SU2 | c64 | r2_s8_d2 | 32 | 1.64 / 1.63 | 2.02 / 2.06 | 1.24 | 3/720 → 3/720 |
| SU2 | c64 | r3_s4_d4 | 1472 | 14.79 / 14.54 | 9.83 / 8.88 | 0.61 | 3/23760 → 3/23760 |
| SU2 | c64 | r4_s3_d4 | 11776 | 73.21 / 75.79 | 28.75 / 29.12 | 0.39 | 3/188624 → 3/188624 |
| SU2 | c64 | r5_s2_d2 | 672 | 15.08 / 13.33 | 9.46 / 9.21 | 0.69 | 3/10960 → 3/10960 |
| SU2 | f64 | r2_s8_d16 | 2048 | 7.02 / 6.31 | 2.91 / 2.65 | 0.42 | 3/16592 → 3/16592 |
| SU2 | f64 | r2_s8_d2 | 32 | 1.63 / 1.74 | 1.79 / 1.97 | 1.10 | 3/464 → 3/464 |
| SU2 | f64 | r3_s4_d4 | 1472 | 14.42 / 13.83 | 8.12 / 8.88 | 0.59 | 3/11984 → 3/11984 |
| SU2 | f64 | r4_s3_d4 | 11776 | 69.21 / 72.00 | 23.33 / 25.21 | 0.34 | 3/94416 → 3/94416 |
| SU2 | f64 | r5_s2_d2 | 672 | 14.58 / 14.50 | 9.92 / 9.88 | 0.68 | 3/5584 → 3/5584 |

### adjoint_data

| sym | dtype | case | elements | before µs (pass 1 / 2) | after µs (pass 1 / 2) | after/before (min) | alloc calls/bytes before → after |
|---|---|---|---:|---:|---:|---:|---|
| U1 | c64 | r2_s8_d16 | 2048 | 4.49 / 3.75 | 2.40 / 2.64 | 0.64 | 8/33616 → 8/33616 |
| U1 | c64 | r2_s8_d2 | 32 | 1.20 / 1.22 | 1.46 / 1.35 | 1.12 | 8/1360 → 8/1360 |
| U1 | c64 | r3_s4_d4 | 768 | 3.97 / 3.89 | 2.91 / 2.67 | 0.69 | 8/13216 → 8/13216 |
| U1 | c64 | r4_s3_d4 | 4864 | 17.33 / 15.21 | 6.42 / 6.35 | 0.42 | 8/78832 → 8/78832 |
| U1 | c64 | r5_s2_d2 | 320 | 3.40 / 3.61 | 2.66 / 2.66 | 0.78 | 8/6208 → 8/6208 |
| U1 | f64 | r2_s8_d16 | 2048 | 3.42 / 3.81 | 2.18 / 2.12 | 0.62 | 8/17232 → 8/17232 |
| U1 | f64 | r2_s8_d2 | 32 | 1.20 / 1.19 | 1.38 / 1.36 | 1.14 | 8/1104 → 8/1104 |
| U1 | f64 | r3_s4_d4 | 768 | 3.62 / 3.56 | 2.54 / 2.72 | 0.71 | 8/7072 → 8/7072 |
| U1 | f64 | r4_s3_d4 | 4864 | 14.33 / 15.25 | 6.10 / 5.52 | 0.39 | 8/39920 → 8/39920 |
| U1 | f64 | r5_s2_d2 | 320 | 3.53 / 3.49 | 2.62 / 2.60 | 0.75 | 8/3648 → 8/3648 |
| fZ2xU1 | c64 | r2_s8_d16 | 2048 | 4.01 / 4.43 | 2.55 / 2.90 | 0.64 | 8/33616 → 8/33616 |
| fZ2xU1 | c64 | r2_s8_d2 | 32 | 1.51 / 1.37 | 1.69 / 1.59 | 1.16 | 8/1360 → 8/1360 |
| fZ2xU1 | c64 | r3_s4_d4 | 768 | 4.04 / 3.81 | 2.75 / 3.04 | 0.72 | 8/13216 → 8/13216 |
| fZ2xU1 | c64 | r4_s3_d4 | 4864 | 17.08 / 14.96 | 7.04 / 6.98 | 0.47 | 8/78832 → 8/78832 |
| fZ2xU1 | c64 | r5_s2_d2 | 320 | 3.44 / 3.69 | 2.48 / 2.58 | 0.72 | 8/6208 → 8/6208 |
| fZ2xU1 | f64 | r2_s8_d16 | 2048 | 3.57 / 3.99 | 2.57 / 2.34 | 0.66 | 8/17232 → 8/17232 |
| fZ2xU1 | f64 | r2_s8_d2 | 32 | 1.39 / 1.26 | 1.58 / 1.42 | 1.13 | 8/1104 → 8/1104 |
| fZ2xU1 | f64 | r3_s4_d4 | 768 | 3.71 / 3.75 | 2.59 / 2.58 | 0.70 | 8/7072 → 8/7072 |
| fZ2xU1 | f64 | r4_s3_d4 | 4864 | 14.33 / 13.88 | 6.38 / 6.17 | 0.44 | 8/39920 → 8/39920 |
| fZ2xU1 | f64 | r5_s2_d2 | 320 | 3.24 / 3.58 | 2.71 / 2.70 | 0.83 | 8/3648 → 8/3648 |
| SU2 | c64 | r2_s8_d16 | 2048 | 4.26 / 4.15 | 2.27 / 2.36 | 0.55 | 8/33616 → 8/33616 |
| SU2 | c64 | r2_s8_d2 | 32 | 1.16 / 1.17 | 1.46 / 1.29 | 1.11 | 8/1360 → 8/1360 |
| SU2 | c64 | r3_s4_d4 | 1472 | 7.27 / 7.31 | 5.31 / 5.35 | 0.73 | 8/24480 → 8/24480 |
| SU2 | c64 | r4_s3_d4 | 11776 | 35.04 / 35.25 | 15.83 / 14.42 | 0.41 | 8/189424 → 8/189424 |
| SU2 | c64 | r5_s2_d2 | 672 | 7.56 / 6.88 | 4.81 / 4.96 | 0.70 | 8/11840 → 8/11840 |
| SU2 | f64 | r2_s8_d16 | 2048 | 3.82 / 3.79 | 2.20 / 2.00 | 0.53 | 8/17232 → 8/17232 |
| SU2 | f64 | r2_s8_d2 | 32 | 1.20 / 1.23 | 1.24 / 1.39 | 1.04 | 8/1104 → 8/1104 |
| SU2 | f64 | r3_s4_d4 | 1472 | 6.19 / 6.29 | 4.56 / 5.06 | 0.74 | 8/12704 → 8/12704 |
| SU2 | f64 | r4_s3_d4 | 11776 | 35.38 / 32.75 | 14.96 / 13.38 | 0.41 | 8/95216 → 8/95216 |
| SU2 | f64 | r5_s2_d2 | 672 | 7.17 / 7.27 | 4.83 / 5.25 | 0.67 | 8/6464 → 8/6464 |


## Reading

- Payload-bound rows (`r4_s3_d4`, `r2_s8_d16`) take 0.32–0.64× the old time.
  For example, SU(2) c64 r4 `add_adjoint` goes from 73.2 to 28.8 µs.
- The 32-element floor (`r2_s8_d2`, eight 2×2 blocks) is 1.02–1.25× on
  `add_adjoint` and 1.04–1.16× on `adjoint_data`, that is +0.1 to +0.4 µs per
  call. This is the fixed per-kernel-call cost of normalizing the layout and
  validating two extents, against a 4-element per-element loop. It is paid 16
  times per `add_adjoint` call. The owned `add` noise reference moves by up
  to ±19 % at this size.
