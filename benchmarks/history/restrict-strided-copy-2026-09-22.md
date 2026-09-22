# restrict_leg / embed_leg move blocks with the fused strided block copy (#1362)

Date 2026-09-22. Host: Apple M4 Max, macOS 15.5, rustc 1.96.0.

- **Before:** `origin/main` `b99aee40`.
- **After:** #1362 rebased onto it, with inline kernel scratch.
- **Build:** both revisions came from one freshly resolved Cargo lock
  (`368bb6404ef757b2…`, `strided-kernel` 0.4.1). They were built one after the
  other into separate target directories with
  `CARGO_PROFILE_RELEASE_DEBUG=line-tables-only cargo build --release --locked
  --offline -p tenet-rs --example eager_overhead_ledger`.
- **Binaries (sha256 prefix):** before `084dc2f76ba0a286`, which is identical to
  the `db89ccc8` build because `b99aee40` changes only CI; after
  `728044dc4e34ec9d`.

Method: the E1 example (`eager-overhead-ledger-2026-09-21.md`) with filters
`restrict_leg` and `permute`, run with `LEDGER_THREADS=one`.

- Each revision ran two passes, in the order before, after, before, after.
  The machine was quiet: no cargo, rustc or julia process was running.
- `permute` is included because it runs on the same adapter scratch.
- The E1 example has no `embed_leg` row.
- Raw pass-1 rows are in
  `restrict-strided-copy-2026-09-22-{before,after}.csv` and
  `restrict-strided-copy-2026-09-22-permute-{before,after}.csv`.
- These timings are observations only. The change rests on its structure, and
  no timing threshold decides it.

## Structural change

**Before.** Each block ran `tensoradd_raw_strided_kernel_mapped` →
`raw_strided_combine_recurse_mapped`. For every element, this path computes two
checked `isize` offsets, converts both to `usize`, and does two slice bounds
checks. It walks one axis per recursion level and never fuses contiguous axes.

**After.** Each block runs
`StridedHostKernelAdapter::tensoradd_strided_checked`:

1. It validates both reachable extents once (`validate_raw_strided_bounds`).
2. It normalizes the layout once (`normalize_fused_layout`: drops extent-one
   axes, sorts by destination stride, fuses contiguous runs).
3. It runs the same `apply_fused_pair_slices` span walk that tree-transform
   replay (`permute`) uses.

The per-element arithmetic is unchanged, so `alpha = 1, beta = 0` is still a
bit-exact copy.

**Scratch.** The adapter's normalization and traversal scratch (`dims`, both
stride vectors, `index`) is now `SmallVec<[_; 8]>`. A fresh adapter per eager
call therefore allocates nothing up to rank 8. This also covers the other
call sites that build one per call. The per-block stride buffers in the
restriction are stack `SmallVec<[isize; 8]>` as well.

## Allocation

`restrict_leg` makes the same number of allocator calls and requests the same
number of bytes as `origin/main` in every row: 13 calls per E1 call. `permute`
is unchanged in every row. Its compiled replay already supplies its traversal
index from the execution workspace.

These properties are pinned by the following tests:

- `tenet-operations/tests/fusion_group_replay_allocations.rs`: a fresh
  adapter's checked copy at rank 8 makes zero allocator calls.
- `tenet/tests/restrict_leg_allocations.rs`: that fixture's `restrict_leg`
  makes 11 calls, the same count as the per-element kernel before this change.

### restrict_leg

| sym | dtype | case | elements | before µs (pass 1 / 2) | after µs (pass 1 / 2) | after/before (min) | alloc calls/bytes before → after |
|---|---|---|---:|---:|---:|---:|---|
| U1 | f64 | r2_s8_d2 | 32 | 1.35 / 1.12 | 1.45 / 1.13 | 1.01 | 13/1096 → 13/1096 |
| U1 | f64 | r2_s8_d16 | 2048 | 2.73 / 2.98 | 1.63 / 1.58 | 0.58 | 13/9160 → 13/9160 |
| U1 | f64 | r3_s4_d4 | 768 | 3.23 / 3.21 | 2.73 / 2.69 | 0.84 | 13/4088 → 13/4088 |
| U1 | f64 | r4_s3_d4 | 4864 | 12.12 / 12.29 | 7.25 / 7.19 | 0.59 | 13/20568 → 13/20568 |
| U1 | f64 | r5_s2_d2 | 320 | 3.43 / 3.44 | 2.36 / 2.30 | 0.67 | 13/2488 → 13/2488 |
| U1 | c64 | r2_s8_d2 | 32 | 1.21 / 1.21 | 1.23 / 1.27 | 1.02 | 13/1224 → 13/1224 |
| U1 | c64 | r2_s8_d16 | 2048 | 3.01 / 3.11 | 1.78 / 1.80 | 0.59 | 13/17352 → 13/17352 |
| U1 | c64 | r3_s4_d4 | 768 | 3.40 / 3.39 | 2.48 / 2.53 | 0.73 | 13/7160 → 13/7160 |
| U1 | c64 | r4_s3_d4 | 4864 | 13.71 / 13.46 | 6.21 / 6.58 | 0.46 | 13/40024 → 13/40024 |
| U1 | c64 | r5_s2_d2 | 320 | 3.54 / 3.49 | 2.35 / 2.33 | 0.67 | 13/3768 → 13/3768 |
| fZ2xU1 | f64 | r2_s8_d2 | 32 | 1.31 / 1.30 | 1.35 / 1.31 | 1.01 | 13/1096 → 13/1096 |
| fZ2xU1 | f64 | r2_s8_d16 | 2048 | 3.09 / 2.94 | 1.78 / 1.77 | 0.60 | 13/9160 → 13/9160 |
| fZ2xU1 | f64 | r3_s4_d4 | 768 | 3.38 / 3.33 | 2.85 / 2.78 | 0.83 | 13/4088 → 13/4088 |
| fZ2xU1 | f64 | r4_s3_d4 | 4864 | 12.25 / 12.33 | 7.42 / 7.38 | 0.60 | 13/20568 → 13/20568 |
| fZ2xU1 | f64 | r5_s2_d2 | 320 | 3.49 / 3.51 | 2.48 / 2.43 | 0.70 | 13/2488 → 13/2488 |
| fZ2xU1 | c64 | r2_s8_d2 | 32 | 1.38 / 1.40 | 1.42 / 1.45 | 1.03 | 13/1224 → 13/1224 |
| fZ2xU1 | c64 | r2_s8_d16 | 2048 | 3.14 / 3.31 | 1.97 / 2.07 | 0.63 | 13/17352 → 13/17352 |
| fZ2xU1 | c64 | r3_s4_d4 | 768 | 3.47 / 3.46 | 2.61 / 2.70 | 0.76 | 13/7160 → 13/7160 |
| fZ2xU1 | c64 | r4_s3_d4 | 4864 | 13.58 / 13.50 | 6.46 / 6.69 | 0.48 | 13/40024 → 13/40024 |
| fZ2xU1 | c64 | r5_s2_d2 | 320 | 3.56 / 3.57 | 2.51 / 2.50 | 0.70 | 13/3768 → 13/3768 |
| SU2 | f64 | r2_s8_d2 | 32 | 1.11 / 1.10 | 1.18 / 1.13 | 1.03 | 13/1096 → 13/1096 |
| SU2 | f64 | r2_s8_d16 | 2048 | 2.74 / 2.73 | 1.65 / 1.61 | 0.59 | 13/9160 → 13/9160 |
| SU2 | f64 | r3_s4_d4 | 1472 | 5.81 / 5.75 | 4.74 / 4.82 | 0.82 | 13/6904 → 13/6904 |
| SU2 | f64 | r4_s3_d4 | 11776 | 29.83 / 30.12 | 17.46 / 17.29 | 0.58 | 13/48216 → 13/48216 |
| SU2 | f64 | r5_s2_d2 | 672 | 6.71 / 6.71 | 4.40 / 4.33 | 0.65 | 13/3896 → 13/3896 |
| SU2 | c64 | r2_s8_d2 | 32 | 1.18 / 1.19 | 1.19 / 1.20 | 1.00 | 13/1224 → 13/1224 |
| SU2 | c64 | r2_s8_d16 | 2048 | 3.29 / 2.99 | 1.76 / 1.71 | 0.57 | 13/17352 → 13/17352 |
| SU2 | c64 | r3_s4_d4 | 1472 | 6.08 / 6.15 | 4.28 / 4.40 | 0.70 | 13/12792 → 13/12792 |
| SU2 | c64 | r4_s3_d4 | 11776 | 32.71 / 32.54 | 14.04 / 14.62 | 0.43 | 13/95320 → 13/95320 |
| SU2 | c64 | r5_s2_d2 | 672 | 6.81 / 6.79 | 4.39 / 4.36 | 0.64 | 13/6584 → 13/6584 |

### permute

| sym | dtype | case | elements | before µs (pass 1 / 2) | after µs (pass 1 / 2) | after/before (min) | alloc calls/bytes before → after |
|---|---|---|---:|---:|---:|---:|---|
| U1 | f64 | r2_s8_d2 | 32 | 0.94 / 0.94 | 0.89 / 0.91 | 0.95 | 12/1376 → 12/1376 |
| U1 | f64 | r2_s8_d16 | 2048 | 1.85 / 1.62 | 1.67 / 1.54 | 0.95 | 12/17504 → 12/17504 |
| U1 | f64 | r3_s4_d4 | 768 | 1.45 / 1.47 | 1.48 / 1.45 | 1.00 | 13/7432 → 13/7432 |
| U1 | f64 | r4_s3_d4 | 4864 | 4.56 / 4.47 | 4.39 / 4.38 | 0.98 | 13/40304 → 13/40304 |
| U1 | f64 | r5_s2_d2 | 320 | 1.22 / 1.26 | 1.19 / 1.22 | 0.98 | 14/4184 → 14/4184 |
| U1 | c64 | r2_s8_d2 | 32 | 1.00 / 1.03 | 1.00 / 1.00 | 1.00 | 12/1632 → 12/1632 |
| U1 | c64 | r2_s8_d16 | 2048 | 2.33 / 2.27 | 2.27 / 2.26 | 1.00 | 12/33888 → 12/33888 |
| U1 | c64 | r3_s4_d4 | 768 | 1.78 / 1.76 | 1.78 / 1.75 | 1.00 | 13/13576 → 13/13576 |
| U1 | c64 | r4_s3_d4 | 4864 | 6.08 / 6.15 | 6.04 / 6.06 | 0.99 | 13/79216 → 13/79216 |
| U1 | c64 | r5_s2_d2 | 320 | 1.36 / 1.32 | 1.34 / 1.33 | 1.01 | 14/6744 → 14/6744 |
| fZ2xU1 | f64 | r2_s8_d2 | 32 | 1.31 / 1.28 | 1.28 / 1.31 | 1.00 | 12/1376 → 12/1376 |
| fZ2xU1 | f64 | r2_s8_d16 | 2048 | 1.91 / 1.93 | 1.95 / 2.17 | 1.02 | 12/17504 → 12/17504 |
| fZ2xU1 | f64 | r3_s4_d4 | 768 | 1.62 / 1.67 | 1.68 / 1.66 | 1.03 | 13/7432 → 13/7432 |
| fZ2xU1 | f64 | r4_s3_d4 | 4864 | 4.60 / 4.69 | 4.57 / 4.99 | 0.99 | 13/40304 → 13/40304 |
| fZ2xU1 | f64 | r5_s2_d2 | 320 | 1.32 / 1.40 | 1.33 / 1.34 | 1.00 | 14/4184 → 14/4184 |
| fZ2xU1 | c64 | r2_s8_d2 | 32 | 1.41 / 1.44 | 1.42 / 1.40 | 0.99 | 12/1632 → 12/1632 |
| fZ2xU1 | c64 | r2_s8_d16 | 2048 | 2.64 / 2.68 | 2.68 / 2.65 | 1.00 | 12/33888 → 12/33888 |
| fZ2xU1 | c64 | r3_s4_d4 | 768 | 1.97 / 2.02 | 2.06 / 2.02 | 1.03 | 13/13576 → 13/13576 |
| fZ2xU1 | c64 | r4_s3_d4 | 4864 | 6.27 / 6.40 | 6.29 / 6.46 | 1.00 | 13/79216 → 13/79216 |
| fZ2xU1 | c64 | r5_s2_d2 | 320 | 1.46 / 1.48 | 1.48 / 1.48 | 1.01 | 14/6744 → 14/6744 |
| SU2 | f64 | r2_s8_d2 | 32 | 0.94 / 0.98 | 0.91 / 0.92 | 0.98 | 12/1376 → 12/1376 |
| SU2 | f64 | r2_s8_d16 | 2048 | 1.58 / 1.63 | 1.60 / 1.76 | 1.01 | 12/17504 → 12/17504 |
| SU2 | f64 | r3_s4_d4 | 1472 | 1.99 / 2.01 | 1.99 / 1.97 | 0.99 | 13/13064 → 13/13064 |
| SU2 | f64 | r4_s3_d4 | 11776 | 17.75 / 17.25 | 17.42 / 17.62 | 1.01 | 21/95792 → 21/95792 |
| SU2 | f64 | r5_s2_d2 | 672 | 4.90 / 4.79 | 4.81 / 4.88 | 1.00 | 18/7096 → 18/7096 |
| SU2 | c64 | r2_s8_d2 | 32 | 1.02 / 1.06 | 0.99 / 1.04 | 0.97 | 12/1632 → 12/1632 |
| SU2 | c64 | r2_s8_d16 | 2048 | 2.29 / 2.34 | 2.27 / 2.27 | 0.99 | 12/33888 → 12/33888 |
| SU2 | c64 | r3_s4_d4 | 1472 | 2.51 / 2.57 | 2.53 / 2.58 | 1.01 | 13/24840 → 13/24840 |
| SU2 | c64 | r4_s3_d4 | 11776 | 23.54 / 23.79 | 23.54 / 23.50 | 1.00 | 21/190000 → 21/190000 |
| SU2 | c64 | r5_s2_d2 | 672 | 5.23 / 5.17 | 5.29 / 5.35 | 1.02 | 18/12472 → 18/12472 |

## Reading

- On the payload-bound `restrict_leg` rows (`r4_s3_d4`, `r2_s8_d16`, `r5`),
  the new path takes 0.43–0.70× the old time:
  - U(1) f64 at r4 goes from 12.1 to 7.2 µs.
  - SU(2) c64 at r4 goes from 32.5 to 14.0 µs.
  - Most of what remains at r4 is per-call structure work outside the kernel
    (ledger constant 2).
- The 32-element floor (`r2_s8_d2`) comes out at 1.00–1.03× on the minimum of
  the two passes. That is within the ~±5 % spread between passes, and the
  allocation count now matches the old path.
- `permute` falls in 0.95–1.03× in every row, with the same allocation count.
