# Complete-structure cache hits skip the extent walk (#1358)

Date 2026-09-21. Host: Apple M4 Max, macOS 15.5, rustc 1.96.0. Before:
`origin/main` `827eebf7`; after: that commit plus #1358 (only
`PreparedFusionTreeLayout::build_complete_from_leg_degeneracies` and the
complete-structure cache hit/miss accounting in `tenet-core/src/fusion_space.rs`
changed). Both built from one Cargo lock
(`d68d3b40d33f0a7da2b78e9c36077dca097c2d006f84f443ffd571ec80f69232`,
`strided-kernel` 0.4.0) into separate target directories with
`CARGO_PROFILE_RELEASE_DEBUG=line-tables-only cargo build --release --locked
--offline -p tenet-rs --example eager_overhead_ledger`. Binaries: before
`0bc03824…9c866e`, after `f7413c2f…fe55f` (sha256).

Method: the E1 example (`eager-overhead-ledger-2026-09-21.md`) with
`LEDGER_THREADS=one`, two passes per revision interleaved before, after,
before, after on a quiet machine; stack samples with `LEDGER_SECONDS=2` for
f64, `r2_s8_d2` and `r4_s3_d4`, permute/repartition/restrict_leg/compose/contract,
all three symmetries. Raw pass-1 rows: `eager-structure-hit-walk-2026-09-21-{before,after}.csv`.

## Allocation (calls, requested bytes per warm call)

Allocation counts were identical between the two passes of each revision.
262 of 270 rows are unchanged. The 8 changed rows are all SU(2) rank-4/5
rows; the reduction is attributed to the removed walk's per-sector
`SmallVec<[_; 8]>` tree buffers, which spill to the heap when a coupled
sector has more than 8 codomain or domain trees:

| row (f64 and c64 identical deltas) | calls before → after | bytes delta |
|---|---:|---:|
| SU2 r4_s3_d4 contract | 179 → 167 | −8448 |
| SU2 r4_s3_d4 repartition | 17 → 11 | −4224 |
| SU2 r5_s2_d2 contract | 169 → 163 | −2816 |
| SU2 r5_s2_d2 repartition | 15 → 12 | −1408 |

No claim is made for other shapes; the removed buffers allocate only when a
coupled sector has more than 8 trees on one side.

## Walk share (stack samples, main thread, timed loop)

`build_complete` = samples with `build_complete_from_leg_degeneracies` on the
stack; `walk proxy` = those without the cache lookup, admission, builder,
key, or signature-validation frames below it (the walk is inlined into that
frame, so this is a proxy). ns = share × median of the same revision.

| row | base build_complete share (ns) | base walk-proxy share (ns) | after build_complete share (ns) | after walk-proxy |
|---|---:|---:|---:|---:|
| U1 r2_s8_d2 permute | 45.3% (764) | 42.4% (715) | 6.4% (60) | 1.1% (10) |
| U1 r2_s8_d2 repartition | 36.3% (503) | 33.7% (468) | 5.5% (45) | 1.2% (10) |
| U1 r2_s8_d2 restrict_leg | 40.9% (790) | 34.8% (673) | 11.2% (133) | 1.1% (13) |
| U1 r2_s8_d2 compose | 21.3% (740) | 20.5% (710) | 1.6% (42) | 0.1% (3) |
| U1 r2_s8_d2 contract | 27.9% (2870) | 26.8% (2756) | 3.6% (258) | 1.4% (98) |
| U1 r4_s3_d4 permute | 22.5% (1352) | 22.0% (1321) | 1.0% (44) | 0.3% (12) |
| U1 r4_s3_d4 repartition | 36.3% (1443) | 34.6% (1373) | 2.3% (54) | 0.3% (8) |
| U1 r4_s3_d4 restrict_leg | 10.1% (1383) | 9.3% (1276) | 1.1% (138) | 0.1% (16) |
| U1 r4_s3_d4 compose | 11.4% (1521) | 11.1% (1477) | 0.7% (84) | 0.5% (61) |
| U1 r4_s3_d4 contract | 23.2% (3574) | 22.1% (3393) | 1.6% (193) | 0.3% (39) |
| fZ2xU1 r2_s8_d2 permute | 34.9% (716) | 32.3% (662) | 4.6% (59) | 0.8% (10) |
| fZ2xU1 r2_s8_d2 repartition | 32.5% (532) | 29.4% (482) | 4.5% (50) | 0.9% (9) |
| fZ2xU1 r2_s8_d2 restrict_leg | 39.1% (843) | 33.3% (719) | 7.9% (110) | 0.8% (11) |
| fZ2xU1 r2_s8_d2 compose | 19.9% (748) | 18.3% (687) | 2.5% (74) | 0.2% (6) |
| fZ2xU1 r2_s8_d2 contract | 24.5% (2959) | 23.3% (2817) | 2.6% (227) | 1.2% (111) |
| fZ2xU1 r4_s3_d4 permute | 22.3% (1443) | 21.6% (1392) | 1.1% (51) | 0.1% (6) |
| fZ2xU1 r4_s3_d4 repartition | 33.7% (1392) | 32.4% (1335) | 3.0% (76) | 0.5% (13) |
| fZ2xU1 r4_s3_d4 restrict_leg | 12.6% (1764) | 11.0% (1536) | 1.6% (203) | 0.6% (81) |
| fZ2xU1 r4_s3_d4 compose | 10.7% (1447) | 10.6% (1438) | 1.0% (118) | 0.6% (71) |
| fZ2xU1 r4_s3_d4 contract | 20.4% (3297) | 19.3% (3117) | 2.1% (264) | 0.5% (58) |
| SU2 r2_s8_d2 permute | 41.7% (710) | 38.2% (650) | 6.9% (66) | 1.2% (11) |
| SU2 r2_s8_d2 repartition | 37.2% (500) | 35.0% (471) | 4.7% (38) | 1.1% (9) |
| SU2 r2_s8_d2 restrict_leg | 38.7% (737) | 33.9% (645) | 11.4% (133) | 0.9% (11) |
| SU2 r2_s8_d2 compose | 20.2% (722) | 17.9% (640) | 2.1% (57) | 0.2% (5) |
| SU2 r2_s8_d2 contract | 30.7% (3229) | 28.8% (3023) | 2.8% (201) | 0.6% (43) |
| SU2 r4_s3_d4 permute | 14.2% (3840) | 14.1% (3823) | 0.4% (94) | 0.1% (16) |
| SU2 r4_s3_d4 repartition | 42.5% (3640) | 41.7% (3567) | 1.4% (64) | 0.0% (0) |
| SU2 r4_s3_d4 restrict_leg | 9.2% (3024) | 8.7% (2854) | 0.6% (178) | 0.2% (59) |
| SU2 r4_s3_d4 compose | 7.6% (2652) | 7.6% (2629) | 0.1% (42) | 0.1% (21) |
| SU2 r4_s3_d4 contract | 20.7% (7092) | 20.1% (6912) | 1.0% (275) | 0.5% (120) |

The pre-lookup walk was 7–42 % of these calls (0.5–7 µs); after the change
the frame keeps 0.1–11 % (key construction, lookup, and `Arc` handling).
Part of the removed walk time was the per-block `storage_end_exclusive`
error construction (#1360), which warm hits no longer reach on this path.

## Timing (observation only, not a gate)

Median ns per call, the lower of two passes, f64. `scale` is a control that
does not reach the changed function.

| row | before median ns | after median ns | after/before |
|---|---:|---:|---:|
| U1 r2_s8_d2 permute | 1688 | 928 | 0.55 |
| U1 r2_s8_d2 repartition | 1385 | 823 | 0.59 |
| U1 r2_s8_d2 restrict_leg | 1930 | 1190 | 0.62 |
| U1 r2_s8_d2 compose | 3472 | 2656 | 0.76 |
| U1 r2_s8_d2 contract | 10292 | 7167 | 0.70 |
| U1 r2_s8_d2 scale | 49 | 50 | 1.01 |
| U1 r4_s3_d4 permute | 6000 | 4472 | 0.75 |
| U1 r4_s3_d4 repartition | 3972 | 2350 | 0.59 |
| U1 r4_s3_d4 restrict_leg | 13708 | 12458 | 0.91 |
| U1 r4_s3_d4 compose | 13292 | 11625 | 0.87 |
| U1 r4_s3_d4 contract | 15375 | 11792 | 0.77 |
| U1 r4_s3_d4 scale | 523 | 568 | 1.09 |
| fZ2xU1 r2_s8_d2 permute | 2050 | 1281 | 0.62 |
| fZ2xU1 r2_s8_d2 repartition | 1639 | 1106 | 0.68 |
| fZ2xU1 r2_s8_d2 restrict_leg | 2158 | 1399 | 0.65 |
| fZ2xU1 r2_s8_d2 compose | 3750 | 2938 | 0.78 |
| fZ2xU1 r2_s8_d2 contract | 12083 | 8896 | 0.74 |
| fZ2xU1 r2_s8_d2 scale | 50 | 50 | 1.00 |
| fZ2xU1 r4_s3_d4 permute | 6458 | 4611 | 0.71 |
| fZ2xU1 r4_s3_d4 repartition | 4125 | 2531 | 0.61 |
| fZ2xU1 r4_s3_d4 restrict_leg | 13958 | 12541 | 0.90 |
| fZ2xU1 r4_s3_d4 compose | 13583 | 12000 | 0.88 |
| fZ2xU1 r4_s3_d4 contract | 16125 | 12584 | 0.78 |
| fZ2xU1 r4_s3_d4 scale | 444 | 502 | 1.13 |
| SU2 r2_s8_d2 permute | 1701 | 943 | 0.55 |
| SU2 r2_s8_d2 repartition | 1344 | 806 | 0.60 |
| SU2 r2_s8_d2 restrict_leg | 1903 | 1162 | 0.61 |
| SU2 r2_s8_d2 compose | 3583 | 2740 | 0.76 |
| SU2 r2_s8_d2 contract | 10500 | 7312 | 0.70 |
| SU2 r2_s8_d2 scale | 49 | 50 | 1.00 |
| SU2 r4_s3_d4 permute | 27042 | 23833 | 0.88 |
| SU2 r4_s3_d4 repartition | 8562 | 4667 | 0.55 |
| SU2 r4_s3_d4 restrict_leg | 32709 | 30250 | 0.92 |
| SU2 r4_s3_d4 compose | 34708 | 31834 | 0.92 |
| SU2 r4_s3_d4 contract | 34334 | 26209 | 0.76 |
| SU2 r4_s3_d4 scale | 958 | 913 | 0.95 |

## Residual

No result layout is rebuilt on warm calls before or after this change (the
public warm test `tenet/tests/complete_structure_warm_hits.rs` asserts 0
misses, admissions, and evictions). Remaining warm per-call structure costs:
global write lock in `commit_layout` (#1366), lookup-key allocation (#1367),
conjugated-source block-structure rebuild (#1368).
