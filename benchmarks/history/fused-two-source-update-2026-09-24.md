# Fused two-source checked update and single-pass block layout (#1401)

Date 2026-09-24. Host: Apple M4 Max, macOS 15.5, rustc 1.96.0. The host was
shared (load average 35–47), so only ratios and op counts are meaningful.

- **Before:** `origin/main` `7110daac`. **After:** #1401 on that base.
- **Build:** the build line of `benchmarks/eager_overhead_ledger.sh`, one Cargo
  lock (`c103cfe3d105561c`) for both, separate target directories.
- **Binaries (sha256 prefix):** before `d93801df21763cd1`, after `5b966b84638f8e72`.
- **Method:** the E1 example with filters `add_adjoint`, `adjoint_data`,
  `restrict_leg` and `add`, `LEDGER_THREADS=one`, passes in the order before,
  after, before, after. `add` (owned + owned) does not reach the changed code
  and is the noise reference. Raw pass-1 rows:
  `fused-two-source-update-2026-09-24-{before,after}.csv`.
- **Timings are observations.** The acceptance gate is the op count below.

## Structural change

| per block | before | after |
|---|---|---|
| lazy `add`, both coefficients nonzero | 2 stride-conversion passes, 2 adapter normalizations, 2 span walks (the second re-reads and rewrites the destination) | 1 joint layout pass over dst/lhs/rhs, 1 span walk |
| lazy-adjoint `data()`, `restrict_leg` | 1 conversion pass + 1 normalization, 1 walk | 1 layout pass, 1 walk |
| scatter (`fusion_scatter_add_assign`) | unchanged: preflight conversion + 1 adapter call | unchanged |

`CheckedBlockLayout` converts every stride, drops extent-one axes, places each
axis in destination-stride order and folds the element count in one loop, then
fuses runs where all operands agree (Strided.jl `_mapreduce_fuse!`). Pinned by
`checked_block_paths_make_one_layout_pass_and_one_walk_per_block` (tenet-tensors)
and `one_layout_pass_and_one_walk_per_two_source_block` (tenet-operations),
debug-build counters.

## Allocation

Unchanged at rank ≤ 8 in every E1 row. Above rank 8 the counts return to
main's: the layout holds 16 non-unit axes inline.

| U(1) c64, lazy-adjoint ops | #1399 rank 10 | #1401 rank 10 | #1401 rank 18 |
|---|---:|---:|---:|
| lazy + owned `add` | 12 | 3 | 7 |
| lazy + lazy `add` | 12 | 3 | 7 |
| first `data()` | 9 | 3 | 6 |

Pinned by `block_stride_buffers_spill_only_past_rank_sixteen`.

## E1, r2_s8_d2 (32 elements, eight 2×2 blocks), min µs

| op | sym | dtype | before (pass 1 / 2) | after (pass 1 / 2) | after/before |
|---|---|---|---:|---:|---:|
| add_adjoint | U1 | f64 | 2.10 / 2.26 | 1.86 / 1.71 | 0.81 |
| add_adjoint | U1 | c64 | 2.34 / 2.20 | 1.92 / 1.80 | 0.82 |
| add_adjoint | fZ2xU1 | f64 | 2.30 / 2.09 | 1.87 / 1.75 | 0.84 |
| add_adjoint | fZ2xU1 | c64 | 2.41 / 2.35 | 1.82 / 1.85 | 0.78 |
| add_adjoint | SU2 | f64 | 2.22 / 2.31 | 1.73 / 1.81 | 0.78 |
| add_adjoint | SU2 | c64 | 2.35 / 2.21 | 1.90 / 1.88 | 0.85 |
| adjoint_data | U1 | f64 | 1.67 / 1.57 | 1.53 / 1.53 | 0.97 |
| adjoint_data | U1 | c64 | 1.83 / 1.64 | 1.69 / 1.63 | 1.00 |
| adjoint_data | fZ2xU1 | f64 | 1.88 / 1.77 | 1.96 / 1.85 | 1.04 |
| adjoint_data | fZ2xU1 | c64 | 2.23 / 2.17 | 2.27 / 2.03 | 0.94 |
| adjoint_data | SU2 | f64 | 1.58 / 1.61 | 1.61 / 1.56 | 0.99 |
| adjoint_data | SU2 | c64 | 1.76 / 1.73 | 1.76 / 1.68 | 0.97 |
| restrict_leg | U1 | f64 | 1.21 / 1.11 | 1.24 / 1.15 | 1.04 |
| restrict_leg | U1 | c64 | 1.28 / 1.17 | 1.45 / 1.30 | 1.11 |
| restrict_leg | fZ2xU1 | f64 | 1.48 / 1.38 | 1.36 / 1.40 | 0.99 |
| restrict_leg | fZ2xU1 | c64 | 1.62 / 1.58 | 1.68 / 1.62 | 1.03 |
| restrict_leg | SU2 | f64 | 1.19 / 1.14 | 1.21 / 1.27 | 1.06 |
| restrict_leg | SU2 | c64 | 1.29 / 1.23 | 1.36 / 1.36 | 1.10 |
| add (noise) | all six | | | | 0.93–1.07 |

- `add_adjoint` is 0.78–0.85× at r2_s8_d2 and 0.79–1.01× on every other case
  (raw CSVs). All rows are now below the #1399 per-element "before" column
  (1.58–1.81 µs on that day's host, not directly comparable in absolute terms).
- `adjoint_data` is within the noise band: it already made one call per block,
  and it gains only the second O(rank) pass.
- `restrict_leg` reads +0–11 %. Three more alternating passes on this host gave
  +2–5 %, of the same sign. The op count per block went from two layout passes
  to one; the residual is not explained structurally and is recorded as an
  observation.

## Kernel microprobe (sizes)

Eight blocks, dst column-major, lhs transposed (lazy-adjoint layout), rhs
column-major, `alpha = beta = 1` (`add_adjoint`), f64, min of 7 reps, two runs.
One binary (sha256 `994e472f714899f7`) times all three paths, ns per 8-block op:

| block | per-element scalar kernel ×2 | #1399 (fill + 2 adapter calls) | #1401 (fill_two + fused walk) |
|---|---:|---:|---:|
| 1×1 | 293–297 | 356–369 | 247–292 |
| 2×2 | 394–408 | 791–796 | 422–435 |
| 4×4 | 581–588 | 801–812 | 484–488 |
| 8×8 | 1273–1410 | 1032–1090 | 757–760 |
| 16×16 | 3781–3954 | 1807–1936 | 1840–1866 |
| 32×32 | 16437–18949 | 5550–6693 | 6029–6037 |

The 2×2 penalty of #1399 (2.0× the per-element kernel on this host) is gone
(1.03–1.07×); from 4×4 up the fused path is at or below both. At 16×16 and
32×32 the #1399 and #1401 columns overlap within the run-to-run spread.
