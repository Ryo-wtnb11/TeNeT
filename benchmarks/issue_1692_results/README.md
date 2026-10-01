# Issue 1692: compact polar factor evidence

## Authority and method

- Baseline: TeNeT `8b931d34869d1f6075bb4580ea73a43d03bbc1f8`.
- Candidate production code: `b776a40a83ce17e7d8a518d97ef1a85e5b19d257` (this report is a later evidence-only commit).
- Harness: [`../issue_1692_polar.rs`](../issue_1692_polar.rs), SHA-256 `4a04e682f099fddbaabaca47b775d46e57f3d489b8c6137dfd4caa6664ad713a`. Both builds used the same `Cargo.lock` SHA-256 `cc03b84d11f9bacff29a6dcdab1259a2349b76f9ece45a60153f421ebc5d74ae`.
- Apple M4 Max (12 performance + 4 efficiency cores), macOS 15.5, arm64, rustc 1.96.0. Release/default features; explicit Faer factorization provider and one dense thread. `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `VECLIB_MAXIMUM_THREADS`, `MKL_NUM_THREADS`, and `RAYON_NUM_THREADS` were all 1.
- Full public `left_polar`/`right_polar` calls include factor publication and drop. Setup is outside timing. Each case has five warmups, one allocation-counted call, then eleven samples of twenty calls; the per-process median is recorded. Five paired processes in B/C, C/B, B/C, C/B, B/C order; `summary.csv` takes the median of per-process medians. `first_ns` is only first within this shared-runtime process, not independent cold-start latency.
- Matrix: U(1), SU(2), U(1) × fermion parity; f32/f64/c32/c64; one sector of width 8 and four dual sectors of width 64; compact diagonal and dense-materialized inputs. Dense controls have diagonal values but run the ordinary dense route. 96 cases per process; no CI timing threshold.

To reproduce, copy the harness into `tenet/examples/issue_1692_polar.rs` in both checkouts. Build each with `cargo build --release -p tenet-rs --example issue_1692_polar`; a scoped `cargo clean --release -p tenet-matrixalgebra -p tenet-rs` preceded each build because both worktrees shared `CARGO_TARGET_DIR`. Save each executable before rebuilding the other. The baseline executable SHA-256 was `9c901408d50e4fd2ad3995210e3875f22516c0079aa62c58ca328d0caaf6f4ee`; the final candidate was `0ec20a2c33e793fa165e20ff3d5ac38b31fc63441afc123a82a3e1d06862d453`. Their `--probe` outputs report all four factors dense in the baseline and compact in the candidate, with unchanged spaces. The probe requires both `dense_data()` rejection and `diagview()` success, because `diagview()` alone also succeeds for a dense endomorphism.

## Results

For the 48 compact-input cases, the median speedup range is 0.991–6.237× (geometric mean 1.932×). The one ratio below 1 is a 4 ns small-case difference with overlapping five-run ranges; none has a nonoverlapping slowdown. Requested allocation bytes decrease in all 48 cases. For one width-8 sector, allocation calls are unchanged; for four width-64 sectors, the candidate makes three more small metadata allocations while eliminating dense factor buffers. All 48 dense controls retain exactly the same allocation calls and requested bytes. Their timing ratios range 0.984–1.043× (geometric mean 1.003×), with no nonoverlapping slowdown across five runs. This does not prove universal dense-path parity.

| Full call | Baseline median ns | Candidate median ns | Baseline calls/bytes | Candidate calls/bytes |
| --- | ---: | ---: | ---: | ---: |
| U(1) f32, one width-8 sector, compact left | 435 | 429 | 14 / 1,422 | 14 / 950 |
| U(1) c64, four dual width-64 sectors, compact left | 8,041 | 1,670 | 17 / 533,544 | 20 / 9,416 |
| SU(2) f64, four dual width-64 sectors, compact right | 4,058 | 1,375 | 17 / 267,304 | 20 / 5,320 |
| U(1) × fermion c64, four dual width-64 sectors, compact right | 8,345 | 1,827 | 17 / 533,544 | 20 / 9,416 |
| U(1) f64, one width-8 sector, dense left control | 9,558 | 9,710 | 64 / 21,124 | 64 / 21,124 |

The first candidate revision built an intermediate pair vector and then unzipped it into compact spectra. Three initial paired runs showed two extra allocation calls and a small-case slowdown. Those raw runs are preserved under `diagnostic-first-candidate/`, **excluded** from `summary.csv`. The final code constructs phase/magnitude spectra once inside the shared validated admission. The five final pairs above use its distinct executable hash and verified probe.

## Correctness and gates

TDD red on the old dense representation, then green for exact bond spaces and compact storage. Independent hand phase/absolute-value values cover four dtypes; U(1), SU(2), product-fermion, dual and changed-role cases; zero, empty, subnormal, and nonfinite fallback. Reconstruction, isometry, positivity, zero input-materialization/SVD/GEMM counters, serialization, dense `materialize()`, and the unchanged dense expert factor bytes are tested. The full workspace test/doctest run passed on the original base. On the final allocation refactor, rebased onto `8b931d34`, `tenet-rs` and `tenet-matrixalgebra` tests/doctests, `cargo fmt --all --check`, all-target workspace clippy with `-D warnings`, strict workspace rustdoc, and workspace coverage passed; coverage thresholds: 188/188 files.

This benchmark does not measure CUDA, checked Generic, arbitrary dense values, or end-to-end algorithms. It supports the changed Host multiplicity-free compact-input route only.
