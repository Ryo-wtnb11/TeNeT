# restrict_leg / embed_leg move blocks with the fused strided copy (#1362)

Date 2026-09-22. Host: Apple M4 Max, macOS 15.5, rustc 1.96.0. Before:
`origin/main` `db89ccc8`; after: that commit plus #1362. Both built from one
freshly resolved Cargo lock (`368bb6404ef757b2…`, `strided-kernel` 0.4.1) into
separate target directories, sequentially, with
`CARGO_PROFILE_RELEASE_DEBUG=line-tables-only cargo build --release --locked
--offline -p tenet-rs --example eager_overhead_ledger`. Binaries (sha256
prefix): before `084dc2f76ba0a286`, after `9e8565cbe875b72a`.

Method: the E1 example (`eager-overhead-ledger-2026-09-21.md`), filter
`restrict_leg`, `LEDGER_THREADS=one`, two passes per revision run in the order
before, after, before, after on a quiet machine (no cargo, rustc or julia
process). The E1 example has no `embed_leg` row. Raw pass-1 rows are in
`restrict-strided-copy-2026-09-22-{before,after}.csv`. These timings are
observations. The change is justified by its structure, described below, and
the timings do not set any threshold.

## Structural change

Before this change, each block ran `tensoradd_raw_strided_kernel_mapped` →
`raw_strided_combine_recurse_mapped`, which does two checked `isize` offset
computations, two `usize` conversions and two slice bounds checks per element.
It walks one axis per recursion level and never fuses contiguous axes. After
this change, the kernel validates both reachable extents once per block
(`validate_raw_strided_bounds`), normalizes the block once
(`normalize_fused_layout`: drops extent-one axes, sorts by destination stride,
fuses contiguous runs), and runs the same `apply_fused_pair_slices` span walk
that tree-transform replay (`permute`) uses. The per-element arithmetic is
unchanged, so `alpha = 1, beta = 0` is still a bit-exact copy.

## Median per call

| sym | dtype | case | elements | before µs (pass 1 / 2) | after µs (pass 1 / 2) | after/before (min) |
|---|---|---|---:|---:|---:|---:|
| U1 | f64 | r2_s8_d2 | 32 | 1.11 / 1.12 | 1.25 / 1.16 | 1.04 |
| U1 | f64 | r2_s8_d16 | 2048 | 2.75 / 2.79 | 1.55 / 1.55 | 0.56 |
| U1 | f64 | r3_s4_d4 | 768 | 3.21 / 3.38 | 2.58 / 2.57 | 0.80 |
| U1 | f64 | r4_s3_d4 | 4864 | 12.21 / 12.33 | 6.83 / 6.85 | 0.56 |
| U1 | f64 | r5_s2_d2 | 320 | 3.43 / 3.42 | 2.06 / 2.09 | 0.60 |
| U1 | c64 | r2_s8_d2 | 32 | 1.22 / 1.17 | 1.22 / 1.24 | 1.04 |
| U1 | c64 | r2_s8_d16 | 2048 | 3.10 / 3.21 | 1.69 / 1.76 | 0.54 |
| U1 | c64 | r3_s4_d4 | 768 | 3.39 / 3.39 | 2.33 / 2.41 | 0.69 |
| U1 | c64 | r4_s3_d4 | 4864 | 13.54 / 13.46 | 5.92 / 5.75 | 0.43 |
| U1 | c64 | r5_s2_d2 | 320 | 3.50 / 3.43 | 2.10 / 2.16 | 0.61 |
| fZ2xU1 | f64 | r2_s8_d2 | 32 | 1.31 / 1.30 | 1.32 / 1.36 | 1.02 |
| fZ2xU1 | f64 | r2_s8_d16 | 2048 | 2.98 / 3.15 | 1.71 / 1.73 | 0.57 |
| fZ2xU1 | f64 | r3_s4_d4 | 768 | 3.36 / 3.33 | 2.60 / 2.65 | 0.78 |
| fZ2xU1 | f64 | r4_s3_d4 | 4864 | 12.25 / 12.38 | 6.94 / 6.96 | 0.57 |
| fZ2xU1 | f64 | r5_s2_d2 | 320 | 3.51 / 3.56 | 2.21 / 2.19 | 0.62 |
| fZ2xU1 | c64 | r2_s8_d2 | 32 | 1.45 / 1.51 | 1.42 / 1.42 | 0.98 |
| fZ2xU1 | c64 | r2_s8_d16 | 2048 | 3.18 / 3.23 | 1.89 / 1.89 | 0.59 |
| fZ2xU1 | c64 | r3_s4_d4 | 768 | 3.54 / 3.54 | 2.42 / 2.45 | 0.68 |
| fZ2xU1 | c64 | r4_s3_d4 | 4864 | 13.75 / 13.83 | 5.96 / 6.25 | 0.43 |
| fZ2xU1 | c64 | r5_s2_d2 | 320 | 3.62 / 3.76 | 2.21 / 2.27 | 0.61 |
| SU2 | f64 | r2_s8_d2 | 32 | 1.17 / 1.19 | 1.14 / 1.13 | 0.97 |
| SU2 | f64 | r2_s8_d16 | 2048 | 2.78 / 2.78 | 1.52 / 1.50 | 0.54 |
| SU2 | f64 | r3_s4_d4 | 1472 | 5.79 / 5.77 | 4.44 / 4.50 | 0.77 |
| SU2 | f64 | r4_s3_d4 | 11776 | 29.67 / 29.62 | 16.54 / 16.71 | 0.56 |
| SU2 | f64 | r5_s2_d2 | 672 | 6.73 / 6.73 | 3.93 / 3.96 | 0.58 |
| SU2 | c64 | r2_s8_d2 | 32 | 1.23 / 1.23 | 1.21 / 1.19 | 0.96 |
| SU2 | c64 | r2_s8_d16 | 2048 | 3.00 / 2.99 | 1.69 / 1.62 | 0.54 |
| SU2 | c64 | r3_s4_d4 | 1472 | 6.10 / 6.12 | 4.01 / 3.97 | 0.65 |
| SU2 | c64 | r4_s3_d4 | 11776 | 32.25 / 32.12 | 13.04 / 13.42 | 0.41 |
| SU2 | c64 | r5_s2_d2 | 672 | 6.96 / 6.92 | 3.94 / 4.03 | 0.57 |

## Allocation

Every row makes 4 more allocator calls and 128 more bytes per call (13 → 17
calls). The extra allocations are the adapter's normalization and traversal
scratch (`dims`, two stride vectors, index). There is one set per call, and
its size depends on the rank, not on the block count or degeneracy. The
per-block stride buffers are stack `SmallVec<[isize; 8]>`. The
payload-independence contract in `tenet/tests/restrict_leg_allocations.rs`
still holds.

## Reading

- The payload-bound rows (`r4_s3_d4`, `r2_s8_d16`, `r5`) take 0.41–0.61× the
  previous time. At r4, U(1) f64 goes from 12.2 µs to 6.8 µs, and SU(2) c64
  from 32 µs to 13 µs. The remainder at r4 is mostly the per-call structure
  work outside the kernel (ledger constant 2).
- The 32-element floor (`r2_s8_d2`) ranges from 0.96× to 1.04×. There the
  fixed cost of normalizing each block and of the four scratch allocations is
  about the same as the per-element saving.
