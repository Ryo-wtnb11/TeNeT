# Issue 1676 verification and full-call benchmark

## Revisions and reproduction

- Baseline: `34aef370e5d220713300c04772b3caa144d83c52`.
- Final production code: `122ef8c8a41f3bfc0dba2b853c237cb1216bae71`.
- Harness: `../issue_1676_qr_lq.rs`, SHA256
  `982ca04ec48acd37779ba71f4ff5954ed74031b77263a61a3003f8e3e4217b53`.
- Identical Cargo.lock SHA256:
  `cc03b84d11f9bacff29a6dcdab1259a2349b76f9ece45a60153f421ebc5d74ae`.
- Apple M4 Max (12 performance + 4 efficiency cores), macOS 15.5,
  aarch64-apple-darwin, rustc 1.96.0 `ac68faa20`, LLVM 22.1.2.
- Release/default features, explicitly Faer, dense threads 1;
  OMP/OPENBLAS/VECLIB/MKL/RAYON thread environment variables all 1.

Copy the harness to `tenet/examples/issue_1676_qr_lq.rs` in each checkout.
With the same compatible shared `CARGO_TARGET_DIR`, run the following
sequentially for each checkout, saving the resulting executable before
building the other revision:

```sh
cargo clean --release -p tenet-matrixalgebra -p tenet-rs
cargo build --release -p tenet-rs --example issue_1676_qr_lq
```

Distinct executable hashes and `--probe` output are retained beside this
report. The probe verifies the expected change from dense output / fresh bond
to compact output / exact source bond. Scoped cleaning is required because an
earlier cross-worktree shared-target build incorrectly reused one executable;
those runs are excluded, as documented in `experiments.md`.

Five pairs use execution order B/C, C/B, B/C, C/B, B/C. Each process measures
384 cases: U1, SU2, U1 × fermion parity; f32/f64/c32/c64;
(C,k,dual) = (1,8,false), (4,8,true), (1,64,false), (4,64,true);
diagonal and dense storage; compact/full QR and LQ. Dense controls materialize
the same diagonal-valued square input before measuring the factorization.
These are storage/dispatch controls, not a claim over every dense matrix shape.

Setup is outside timing. Each full public call includes allocation,
factor publication, and destruction. `first_ns` records the first call of a
case, but earlier cases share the runtime, so it is not independent cold-start
latency. After five warmups, one call records allocation calls and requested
bytes, then eleven samples of twenty calls yield the per-process median.
The allocator counter is disabled during timing (its atomic enable check is
still present). No dense-kernel-only timing or CPU affinity claim is made.

`summary.csv` takes the median of the five per-process medians and includes
min/max variability and allocation data. `../issue_1676_summarize.py` regenerates
it; all individual runs remain authoritative. Experimental CSVs are outside
the aggregation glob. Timing is not a CI gate.

## Correctness gates

See `gates.txt` for exact counts and coverage. The final production change was
checked by affected matrixalgebra/typed tests and doctests, workspace coverage,
formatting, strict all-target clippy, and strict rustdoc. The prior full
workspace run and final affected run are distinguished in that record.
The no-solver negative control is in `red.log`. Hand diagonal phases,
reconstruction, isometry, exact dual spaces, materialization/solver counters,
serialization and fallback admission are directly tested. Existing polar tests
exercise the extracted shared scalar arithmetic. CUDA and checked Generic
retain their routes; this is not device performance evidence.

## Final measurements

All 192 diagonal cases improve: median speedup 17.71–536.45×, geometric mean
86.23×. This is the removal of dense materialization/factorization for admitted
compact input, not a faster dense QR kernel. The diagonal arithmetic and owned
outputs scale with the stored spectrum length.

Representative full-call medians (nanoseconds; requested allocation bytes):

| Case | Baseline ns | Candidate ns | Baseline calls/bytes | Candidate calls/bytes |
| --- | ---: | ---: | ---: | ---: |
| U1 f32, C=1 k=8, diagonal compact QR | 3,764 | 197 | 32 / 11,844 | 9 / 588 |
| Dual U1 f64, C=4 k=64, diagonal compact QR | 253,570 | 975 | 251 / 1,296,896 | 15 / 4,880 |
| Dual SU2 c64, C=4 k=64, diagonal compact LQ | 642,922 | 1,252 | 276 / 4,085,044 | 15 / 8,976 |

For the 192 dense controls, baseline/candidate median ratios range
0.9701–1.0619 (geometric mean 1.0096). Every case retains exactly the same
allocation calls and bytes. The earlier systematic 3–5% real dense compact-LQ
regression is absent after the bounded append adjustment. The final real
compact-LQ ratios range 0.9878–1.0278. Every apparent dense slowdown has
intersecting baseline/candidate per-run ranges. Small cases still vary by a
few percent; this does not establish universal dense parity or a dense speedup.

The following rows show all five per-process medians, including the two worst
apparent dense slowdowns and two former regression cases. Pair order is
B/C, C/B, B/C, C/B, B/C; the slowdown changes sign in pair 3 for the first two.

| Dense case | Baseline ns, runs 1–5 | Candidate ns, runs 1–5 |
| --- | --- | --- |
| U1 f64, C=1 k=8, compact QR | 3647, 3670, 3683, 3747, 3664 | 3783, 3687, 3658, 3810, 3875 |
| U1 × fermion c64, C=1 k=8, full LQ | 4166, 4195, 4316, 4139, 4122 | 4318, 4312, 4204, 4285, 4262 |
| Dual SU2 f32, C=4 k=64, compact LQ | 200779, 200983, 200814, 204310, 201210 | 199227, 201385, 199968, 200054, 200810 |
| Dual U1 f64, C=4 k=64, compact LQ | 264460, 267100, 266064, 268454, 265764 | 260583, 261160, 261020, 261266, 263512 |

The worst small dense QR median changes from 3,670 to 3,783 ns (+113 ns,
3.08%). Its ranges are 3,647–3,747 and 3,658–3,875 ns. The next full-LQ case
changes from 4,166 to 4,285 ns (+119 ns, 2.86%), with ranges 4,122–4,316 and
4,204–4,318 ns. These non-improvements remain in the result rather than being
hidden by the aggregate. Further claims at this scale would require additional
measurement; no threshold or timing assertion is encoded in tests.
