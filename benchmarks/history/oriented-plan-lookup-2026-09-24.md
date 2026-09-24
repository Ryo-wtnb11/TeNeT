# Lazy-adjoint oriented plan taken from the Runtime store (#1414)

Date 2026-09-24. Host: Apple M4 Max, macOS 15.5, rustc 1.96.0.

- **Before:** `origin/main` `97662eda`.
- **After:** branch `oriented-plan-lookup`.
- **Build:** both revisions built one after the other into separate target
  directories with `CARGO_PROFILE_RELEASE_DEBUG=line-tables-only cargo build
  --release -p tenet-rs --example eager_overhead_ledger`. Binaries (sha256
  prefix): before `578625849c4d4d88`, after `d8353b157acb4202`.
- **Method:** E1 with the filters `contract`, `contract_conj`, `compose_conj`,
  `LEDGER_THREADS=one`, `nice -n 19`, two passes ordered before, after, before,
  after. Allocation counts matched between passes. Medians are the mean of the
  two passes. Raw pass-1 rows: `oriented-plan-lookup-2026-09-24-{before,after}.csv`.
- Timings are observations only.

## Change

`TreeTransformCache::get_or_compile_tree_pair_oriented` compiled the
adjoint-oriented tree-pair plan on every call and never consulted the
Runtime-owned transform store that the owned path uses. A runtime-bound
adjoint call now looks the plan up in that store. Its key is the ordinary
tree-pair key (rule, operation, destination and parent structures,
conjugation) plus the source orientation, so an adjoint-oriented entry never
aliases an ordinary entry with the same structures.

## E1 (facade), f64

Calls and bytes per warm call; the median is in µs. The c64 rows show the same
calls; their bytes are in the raw CSVs.

| op | symmetry | dtype | case | before calls / B | after calls / B | before med µs | after med µs |
|---|---|---|---|---:|---:|---:|---:|
| contract_conj | U1 | f64 | r2_s8_d2 | 474 / 61098 | 33 / 4950 | 20.00 | 5.82 |
| contract_conj | U1 | f64 | r2_s8_d16 | 474 / 77226 | 33 / 21078 | 24.56 | 9.80 |
| contract_conj | U1 | f64 | r3_s4_d4 | 1120 / 190500 | 36 / 10872 | 51.23 | 6.32 |
| contract_conj | U1 | f64 | r4_s3_d4 | 1879 / 554519 | 34 / 91546 | 130.08 | 10.83 |
| contract_conj | U1 | f64 | r5_s2_d2 | 1469 / 208980 | 38 / 7500 | 56.04 | 5.88 |
| contract_conj | fZ2xU1 | f64 | r2_s8_d2 | 484 / 61866 | 43 / 5718 | 23.62 | 8.31 |
| contract_conj | fZ2xU1 | f64 | r2_s8_d16 | 484 / 77994 | 43 / 21846 | 27.79 | 12.54 |
| contract_conj | fZ2xU1 | f64 | r3_s4_d4 | 1127 / 191076 | 43 / 11448 | 55.40 | 7.50 |
| contract_conj | fZ2xU1 | f64 | r4_s3_d4 | 1886 / 555095 | 41 / 92122 | 136.98 | 12.10 |
| contract_conj | fZ2xU1 | f64 | r5_s2_d2 | 1474 / 209428 | 43 / 7948 | 60.77 | 6.66 |
| contract_conj | SU2 | f64 | r2_s8_d2 | 474 / 60570 | 30 / 4294 | 21.60 | 6.24 |
| contract_conj | SU2 | f64 | r2_s8_d16 | 474 / 76698 | 30 / 20422 | 25.58 | 10.29 |
| contract_conj | SU2 | f64 | r3_s4_d4 | 2034 / 350384 | 31 / 15624 | 93.23 | 8.21 |
| contract_conj | SU2 | f64 | r4_s3_d4 | 3599 / 1378510 | 45 / 147562 | 327.56 | 29.77 |
| contract_conj | SU2 | f64 | r5_s2_d2 | 2333 / 362932 | 36 / 9244 | 108.63 | 12.10 |
| compose_conj | U1 | f64 | r2_s8_d2 | 16 / 1990 | 16 / 1990 | 2.66 | 2.65 |
| compose_conj | U1 | f64 | r2_s8_d16 | 16 / 18118 | 16 / 18118 | 5.74 | 5.66 |
| compose_conj | U1 | f64 | r3_s4_d4 | 16 / 1816 | 16 / 1816 | 4.37 | 4.35 |
| compose_conj | U1 | f64 | r4_s3_d4 | 22 / 113620 | 22 / 113620 | 14.85 | 14.71 |
| compose_conj | U1 | f64 | r5_s2_d2 | 16 / 2054 | 16 / 2054 | 3.57 | 3.56 |
| compose_conj | fZ2xU1 | f64 | r2_s8_d2 | 18 / 2342 | 18 / 2342 | 3.49 | 3.53 |
| compose_conj | fZ2xU1 | f64 | r2_s8_d16 | 18 / 18470 | 18 / 18470 | 6.61 | 6.57 |
| compose_conj | fZ2xU1 | f64 | r3_s4_d4 | 18 / 2200 | 18 / 2200 | 5.38 | 5.26 |
| compose_conj | fZ2xU1 | f64 | r4_s3_d4 | 25 / 114124 | 25 / 114124 | 16.38 | 16.15 |
| compose_conj | fZ2xU1 | f64 | r5_s2_d2 | 18 / 2422 | 18 / 2422 | 4.40 | 4.40 |
| compose_conj | SU2 | f64 | r2_s8_d2 | 16 / 1990 | 16 / 1990 | 2.70 | 2.72 |
| compose_conj | SU2 | f64 | r2_s8_d16 | 16 / 18118 | 16 / 18118 | 5.74 | 5.82 |
| compose_conj | SU2 | f64 | r3_s4_d4 | 16 / 1816 | 16 / 1816 | 4.84 | 4.83 |
| compose_conj | SU2 | f64 | r4_s3_d4 | 24 / 283603 | 24 / 283603 | 36.25 | 36.12 |
| compose_conj | SU2 | f64 | r5_s2_d2 | 16 / 2438 | 16 / 2438 | 3.87 | 3.97 |
| contract | U1 | f64 | r2_s8_d2 | 27 / 4416 | 27 / 4416 | 5.56 | 5.48 |
| contract | U1 | f64 | r2_s8_d16 | 27 / 20544 | 27 / 20544 | 9.43 | 9.30 |
| contract | U1 | f64 | r3_s4_d4 | 29 / 10160 | 29 / 10160 | 5.45 | 5.31 |
| contract | U1 | f64 | r4_s3_d4 | 29 / 91096 | 29 / 91096 | 9.55 | 9.26 |
| contract | U1 | f64 | r5_s2_d2 | 31 / 6800 | 31 / 6800 | 4.81 | 4.76 |
| contract | fZ2xU1 | f64 | r2_s8_d2 | 27 / 4416 | 27 / 4416 | 7.36 | 7.32 |
| contract | fZ2xU1 | f64 | r2_s8_d16 | 27 / 20544 | 27 / 20544 | 11.21 | 11.10 |
| contract | fZ2xU1 | f64 | r3_s4_d4 | 36 / 10736 | 36 / 10736 | 6.48 | 6.46 |
| contract | fZ2xU1 | f64 | r4_s3_d4 | 36 / 91672 | 36 / 91672 | 10.48 | 10.50 |
| contract | fZ2xU1 | f64 | r5_s2_d2 | 36 / 7248 | 36 / 7248 | 5.41 | 5.41 |
| contract | SU2 | f64 | r2_s8_d2 | 25 / 4096 | 25 / 4096 | 5.63 | 5.66 |
| contract | SU2 | f64 | r2_s8_d16 | 25 / 20224 | 25 / 20224 | 9.50 | 9.47 |
| contract | SU2 | f64 | r3_s4_d4 | 23 / 14832 | 23 / 14832 | 6.04 | 6.08 |
| contract | SU2 | f64 | r4_s3_d4 | 37 / 146584 | 37 / 146584 | 24.23 | 24.33 |
| contract | SU2 | f64 | r5_s2_d2 | 27 / 8432 | 27 / 8432 | 8.27 | 8.41 |

`compose_conj` does not reach the oriented compiler: its calls are unchanged in
every row. The remaining `contract_conj` excess over owned `contract` is +5 to
+9 calls, and +16 for fZ2×U1 `r2_s8_d2`. It lies outside plan compilation,
because a warm call compiles no plan. The likely source is the per-call
`FusionOperand::prepare` adjoint projection, but that was not profiled here.
