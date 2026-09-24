# One admission per op-bearing batch GEMM phase (#1291)

Date 2026-09-25. Host: Apple M4 Max (16 cores), macOS 15.5, rustc 1.96.0,
faer provider, default threads. Base `origin/main` `a6141371`.

Timings are observations only. The host was shared with other agents' builds
and tests during the runs (1-minute load average 31 to 70, recorded per run
in the CSV), so absolute numbers are noisy. The comparison rests on the
interleaved passes and on the medians below, not on any single row.

## Options

A mixed run partition of `matmul_batch_axpby_with_ops_into` issues each
batchable run as one strided backend call and, before this change, each
non-batchable run as its own serial session. Each of those entries takes the
process-wide execution permit and, on faer, one Rayon pool install.

- **(a) Coalesce.** Adjacent non-batchable runs become one serial call. Plan
  cache slots are absolute job indices (`cache_start + relative_slot`), so each
  GEMM keeps its slot, kernel, order, alpha/beta and conjugation.
- **(b) Scope.** The whole partition runs inside one Tenferro
  `CpuBackend::with_execution_scope` (the #1424 helper), so the strided calls
  and the serial sessions share one permit and pool install.

TeNeT counters for one phase (`cpu_session_stats`, sessions / admissions):

| partition | base | (a) | (b) | (a)+(b) |
|---|---|---|---|---|
| `[4,1,1,1]` | 3 / 4 | 1 / 2 | 3 / 1 | 1 / 1 |
| `[1,4,1,4]` | 2 / 4 | 2 / 4 | 2 / 1 | 2 / 1 |

(a) does nothing for alternating partitions; (b) makes every partition one
admission. They compose: inside the scope, (a) still saves the per-session
engine entry and buffer-pool loan.

Working set: unchanged by either. Both issue the same jobs in the same order
into the caller's output; neither stages or batches outputs. (b) adds the
scope's shared permit `Arc` (48 B, as in #1424) and one `CpuBackend` clone per
phase.

Permit-hold cost: under (b) the permit is held from the first to the last GEMM
of the phase, including the between-run view setup (a few `DenseView`
constructions, no allocation). A Tenferro call on another thread therefore
waits for the rest of the phase instead of the rest of the current run. The
wait is bounded by one phase, which is what a single-run phase already costs
today.

## Victim-latency harness

`gemm-partition-scope-2026-09-25-victim.rs` (kept uncompiled; copy to
`tenet-dense/examples/`). An aggressor thread runs one c64
(`Adjoint`, `Identity`) phase back to back; the main thread issues a 4x4 f64
`matmul_into` on its own executor every 50 to 200 µs and records its latency
(4000 samples). Block edge `n` = 8 (admission-dominated) and 48
(GEMM-dominated). All four modes come from one release binary (sha256 prefix
`16fc1be5712a45d1`) built from the change plus a temporary, uncommitted
`TENET_1291_MODE` switch that disables coalescing and/or the scope; with both
disabled it is the base algorithm. Three passes, ordered base, a, b, ab /
ab, b, a, base / base, a, b, ab. Raw rows:
`gemm-partition-scope-2026-09-25-victim.csv`.

Median of the three passes (µs). `solo` = phase time without the victim;
the victim-alone p50 (measured once per process, before the aggressor starts) ranges 9.04 to 11.75 over the CSV rows; its per-mode medians over the three passes are 9.62 to 10.50.

| partition | n | mode | solo phase | victim p50 | victim p99 |
|---|---:|---|---:|---:|---:|
| `[4,1,1,1]` | 8 | base | 86.4 | 57.2 | 100.0 |
| | | (a) | 34.7 | 46.3 | 98.6 |
| | | (b) | 17.5 | 32.6 | 69.2 |
| | | (a)+(b) | 17.2 | 33.6 | 66.1 |
| `[1,4,1,4]` | 8 | base | 86.3 | 57.0 | 101.6 |
| | | (a) | 86.9 | 42.5 | 84.2 |
| | | (b) | 17.2 | 33.4 | 69.3 |
| | | (a)+(b) | 16.7 | 32.0 | 59.7 |
| `[4,1,1,1]` | 48 | base | 393 | 128 | 359 |
| | | (a) | 329 | 141 | 319 |
| | | (b) | 324 | 245 | 547 |
| | | (a)+(b) | 312 | 244 | 491 |
| `[1,4,1,4]` | 48 | base | 550 | 145 | 344 |
| | | (a) | 497 | 146 | 327 |
| | | (b) | 507 | 398 | 721 |
| | | (a)+(b) | 450 | 382 | 671 |

Single-sample maxima in the milliseconds appear in every mode (for example
base `[1,4,1,4]` n=8 pass 1, 1416 µs) and track host load, not the mode.

## Reading

- Where admission dominates (n=8), (b) cuts the phase time about 5x and also
  lowers the victim's latency (p50 −41 to −44%, p99 −31 to −41% against
  base): the aggressor spends less time in the permit and in pool installs
  overall. (a) alone helps only where singletons are adjacent.
- Where GEMM work dominates (n=48), (b) gains 8 to 21% phase time, and the
  victim's p50 rises about 1.9x (`[4,1,1,1]`) to 2.7x (`[1,4,1,4]`), p99 about
  1.4x to 2.1x. That is the expected cost: the victim now waits for the
  remaining phase, not the remaining run. Its bound is one phase's GEMM time,
  the same bound a single-run phase already has.
- Aggregate throughput is not reduced: the permit serializes both threads'
  Tenferro work in every mode.

Choice: (a)+(b). The admission saving is structural (one per phase regardless
of partition shape); the latency cost is bounded by the phase and is the same
tradeoff #1424 accepted for the streaming factorization loops. No block size
or run count selects between the modes. A covering partition with a single run
is already one dispatch and skips the scope (no admission to save, and no
`CpuBackend` clone).

## Not measured

BLAS providers (their permit is provider-exclusive, so the victim bound is the
same phase but on every thread), qg1, and a victim that is itself a long
phase.
