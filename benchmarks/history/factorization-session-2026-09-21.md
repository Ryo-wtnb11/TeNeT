# One CPU linalg session per compact factorization call (#1361)

Date 2026-09-21. Host: Apple M4 Max (12 P + 4 E cores), macOS 15.5,
rustc 1.96.0. Baseline `origin/main` `ae51bf84`; candidate is this commit.
Both binaries were built from the same `tenet/examples/eager_overhead_ledger.rs`
(this commit adds its `svd_compact` row; the baseline build used the same
example source), one after the other into separate target directories, with
`CARGO_PROFILE_RELEASE_DEBUG=line-tables-only cargo build --release --offline
-p tenet-rs --example eager_overhead_ledger`, default features (`cpu-faer`),
and the same freshly resolved lock (sha256 `368bb640…`, not committed).

Reproduce one row: `LEDGER_THREADS={one,default} eager_overhead_ledger
{qr_compact,svd_compact}`. Two rounds, each running base then candidate, with
the `one` and `default` layouts interleaved. Every invocation waited until no
other cargo, rustc, or julia process was running.

Raw data is committed next to this file:
`factorization-session-2026-09-21-{base,cand}-{one,default}-run{1,2}.csv`,
with the E1 ledger's columns. Each table value is the median of the two
per-run medians. The allocation columns come from run 1. They count calls on
the calling thread only, so they are valid for the `one` layout only: under
`default`, the session body runs on a Tenferro pool worker (see the E1 record).

Timing is an observation, not a gate. The gate is the session-count test
`tenet/tests/factorization_session_scope.rs`:

| fixture | coupled sectors | sessions per qr_compact / left_orth / svd_compact, base → candidate |
|---|---|---|
| U(1) f64, 1←1 | 8 | 8 → 1 |
| U(1) c64, 2←2 | 15 | 15 → 1 |
| SU(2) f64, 2←1 | 4 | 4 → 1 |
| SU(2) c64, 2←2 | 7 | 7 → 1 |
| fZ2×U(1) f64, 1←1 | 6 | 6 → 1 |
| fZ2×U(1) c64, 2←1 | 6 | 6 → 1 |

## f64 rows (c64 rows are in the CSVs and show the same pattern)

| threads | op | sym | dtype | case | coupled | base median µs | cand median µs | ratio | base allocs/bytes | cand allocs/bytes |
|---|---|---|---|---|---|---|---|---|---|---|
| default | qr_compact | SU2 | f64 | r2_s8_d16 | 8 | 136.60 | 58.73 | 0.43 | 42/90720 | 45/93536 |
| default | qr_compact | SU2 | f64 | r2_s8_d2 | 8 | 124.67 | 33.10 | 0.27 | 42/58464 | 45/61280 |
| default | qr_compact | SU2 | f64 | r4_s3_d4 | 5 | 708.69 | 591.60 | 0.83 | 115/246332 | 121/248956 |
| default | qr_compact | U1 | f64 | r2_s8_d16 | 8 | 137.31 | 58.73 | 0.43 | 42/90720 | 46/95056 |
| default | qr_compact | U1 | f64 | r2_s8_d2 | 8 | 124.69 | 32.83 | 0.26 | 42/58464 | 45/61280 |
| default | qr_compact | U1 | f64 | r4_s3_d4 | 5 | 148.29 | 101.23 | 0.68 | 111/135154 | 117/137778 |
| default | qr_compact | fZ2xU1 | f64 | r2_s8_d16 | 8 | 135.50 | 58.69 | 0.43 | 42/90720 | 45/93536 |
| default | qr_compact | fZ2xU1 | f64 | r2_s8_d2 | 8 | 128.75 | 33.44 | 0.26 | 42/58464 | 45/61280 |
| default | qr_compact | fZ2xU1 | f64 | r4_s3_d4 | 5 | 149.04 | 101.73 | 0.68 | 111/135154 | 117/137778 |
| default | svd_compact | SU2 | f64 | r2_s8_d16 | 8 | 259.48 | 191.83 | 0.74 | 66/148480 | 68/151040 |
| default | svd_compact | SU2 | f64 | r2_s8_d2 | 8 | 133.67 | 47.25 | 0.35 | 66/115328 | 68/117888 |
| default | svd_compact | SU2 | f64 | r4_s3_d4 | 5 | 1489.52 | 1451.46 | 0.97 | 135/283980 | 140/286444 |
| default | svd_compact | U1 | f64 | r2_s8_d16 | 8 | 259.98 | 191.69 | 0.74 | 67/150000 | 68/151040 |
| default | svd_compact | U1 | f64 | r2_s8_d2 | 8 | 131.10 | 47.83 | 0.36 | 66/115328 | 68/117888 |
| default | svd_compact | U1 | f64 | r4_s3_d4 | 5 | 453.58 | 422.67 | 0.93 | 131/172162 | 136/174626 |
| default | svd_compact | fZ2xU1 | f64 | r2_s8_d16 | 8 | 258.62 | 190.15 | 0.74 | 66/148480 | 68/151040 |
| default | svd_compact | fZ2xU1 | f64 | r2_s8_d2 | 8 | 130.73 | 47.85 | 0.37 | 66/115328 | 68/117888 |
| default | svd_compact | fZ2xU1 | f64 | r4_s3_d4 | 5 | 450.40 | 421.04 | 0.93 | 131/172162 | 136/174626 |
| one | qr_compact | SU2 | f64 | r2_s8_d16 | 8 | 44.31 | 41.92 | 0.95 | 130/225248 | 133/228064 |
| one | qr_compact | SU2 | f64 | r2_s8_d2 | 8 | 23.77 | 22.02 | 0.93 | 130/108512 | 133/111328 |
| one | qr_compact | SU2 | f64 | r4_s3_d4 | 5 | 198.17 | 199.52 | 1.01 | 283/1041393 | 289/1044017 |
| one | qr_compact | U1 | f64 | r2_s8_d16 | 8 | 44.19 | 41.98 | 0.95 | 130/225248 | 133/228064 |
| one | qr_compact | U1 | f64 | r2_s8_d2 | 8 | 23.73 | 22.25 | 0.94 | 130/108512 | 133/111328 |
| one | qr_compact | U1 | f64 | r4_s3_d4 | 5 | 87.23 | 86.10 | 0.99 | 193/424393 | 199/427017 |
| one | qr_compact | fZ2xU1 | f64 | r2_s8_d16 | 8 | 44.94 | 42.58 | 0.95 | 130/225248 | 133/228064 |
| one | qr_compact | fZ2xU1 | f64 | r2_s8_d2 | 8 | 24.15 | 22.69 | 0.94 | 130/108512 | 133/111328 |
| one | qr_compact | fZ2xU1 | f64 | r4_s3_d4 | 5 | 87.58 | 87.04 | 0.99 | 193/424393 | 199/427017 |
| one | svd_compact | SU2 | f64 | r2_s8_d16 | 8 | 178.71 | 173.65 | 0.97 | 162/376320 | 164/378880 |
| one | svd_compact | SU2 | f64 | r2_s8_d2 | 8 | 33.79 | 31.90 | 0.94 | 162/193792 | 164/196352 |
| one | svd_compact | SU2 | f64 | r4_s3_d4 | 5 | 1022.71 | 1021.71 | 1.00 | 320/1495415 | 325/1497879 |
| one | svd_compact | U1 | f64 | r2_s8_d16 | 8 | 178.60 | 173.69 | 0.97 | 162/376320 | 164/378880 |
| one | svd_compact | U1 | f64 | r2_s8_d2 | 8 | 34.00 | 31.94 | 0.94 | 162/193792 | 164/196352 |
| one | svd_compact | U1 | f64 | r4_s3_d4 | 5 | 392.13 | 387.73 | 0.99 | 222/636277 | 227/638741 |
| one | svd_compact | fZ2xU1 | f64 | r2_s8_d16 | 8 | 177.67 | 173.96 | 0.98 | 162/376320 | 164/378880 |
| one | svd_compact | fZ2xU1 | f64 | r2_s8_d2 | 8 | 34.62 | 32.71 | 0.94 | 162/193792 | 164/196352 |
| one | svd_compact | fZ2xU1 | f64 | r4_s3_d4 | 5 | 388.69 | 383.27 | 0.99 | 222/636277 | 227/638741 |

## Reading

- **Default threads.** Most of the gain is here. Every Tenferro session pays
  the pool hop (E1 constant 1, about 10–13 µs). One session per call instead
  of one per coupled sector takes the 8-sector QR from about 125 µs to about
  33 µs, and the 8-sector SVD from about 131 µs to about 48 µs. Rows whose
  kernel time dominates gain less (r4 SU(2) SVD: 0.96–0.97).
- **One thread.** The gain is only 3–7 % (8-sector 2×2 QR: 23.7 → 22.0–22.3 µs).
  The saved per-session entry is about 0.2 µs per block. So most of E1
  constant 5's 16.5 µs "backend" share is per-QR work that stays: Tenferro
  tensor wrapping, descriptor validation, and output conversion. This matches
  the independent review's correction to leaf 4. That per-op residue stays
  with the batched Tenferro wishlist (E1 leaf 7).
- **Allocations (one thread).** The candidate adds a fixed 2–6 calls and
  about 1–3 KB per call, for example 130 → 133 calls on the 8-sector QR. The
  extra calls are the batch's block, view, and flat-output vectors. The count
  does not grow with the sector count. Peak working set is unchanged, because
  the covered loops already kept every block's factors until the final concat.

## Reference

- TensorKit 0.17.1 (`f87ca7fe`), `src/factorizations/matrixalgebrakit.jl:35-47`:
  the generated `MAK.qr_compact!` / `svd_compact!` / … call
  `foreachblock` (`src/tensors/blockiterator.jl:33-46`), which makes one dense
  MatrixAlgebraKit call per coupled block. Julia has no execution-session
  concept, so no corresponding cost exists there.
- QSpace `d2d3d7da`, `Source/mpsortho.cc:546` `SVD_Data::blockSVD`: an
  OpenMP loop with one `wbSVD` per symmetry block. QSpace has no QR
  factorization in `orthoQS`, and no session concept either.
- So the session is a Rust/Tenferro-specific admission cost. This change keeps
  the reference granularity (one dense factorization per coupled block, in
  block order) and pays the admission once per call.
