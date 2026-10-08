# Issues 1799 + 1752: per-sector pinv / exp / polar

- Baseline: TeNeT `4bfb99ca5f1d2dfeb70d4a96819876aa224a7ab1`. Candidate: branch
  `issue-1799-1752-spectral-sector` at `12a9466d` (production code from `cba4d5a0`); after the
  rebase onto `d4e0a4ff` these are `80597cf0` / `71cd6b7d`, and `3ae2b6bb` adds this record.
- Harness: [`../issue_1799_1752_matrix_functions.rs`](../issue_1799_1752_matrix_functions.rs),
  SHA-256 `7ad15a97cad8f79c6bf83b23061ed57363f498397acfdcf78bd5bab65f352135`, copied into
  `tenet/examples/` of each checkout; `Cargo.lock` unchanged
  (`cf6f1228bceb7f261940a522b5db3431359681ff511b801ccc873395551c7677`).
- Apple M4 Max, macOS 15.5, rustc 1.96.0, release, default features, Faer provider, one dense
  thread, `OMP/OPENBLAS/VECLIB/RAYON_NUM_THREADS=1`. Separate target directories per checkout.
- Full public calls on `[V, V] <- [V, V]`, multiplicity-free U(1) (charges -1, 0, 1) and SU(2)
  (spins 0, 1/2, 1), degeneracy `k` = 2 and 6 per sector, f64 and Complex64. Three warmups, one
  allocation-counted call, eleven samples of five calls; per-process median. Five processes per
  build in B/H, H/B, B/H, H/B, B/H order; `summary.csv` is the median of the five medians.

## Results

Allocation calls and requested bytes fall for every pinv, Hermitian exp and polar case (no
factor-layout publication, no adjoint factor copies, no contraction output). General (Padé) exp is
unchanged except one call / 464 bytes for the boxed Padé workspace of the route enum.

Timing (median speedup base/head, geometric mean over the 8 cases per operation, range):

| operation | geomean | range |
| --- | ---: | ---: |
| pinv | 1.03 | 1.00–1.08 |
| pinv of a lazy adjoint | 0.99 | 0.93–1.04 |
| Hermitian exp | 1.01 | 0.95–1.07 |
| general exp | 1.00 | 0.97–1.06 |
| left / right polar | 0.96 / 0.97 | 0.90–1.03 |
| polar of a lazy adjoint | 0.98 / 0.97 | 0.90–1.03 |

At `k = 6` every operation is within run-to-run spread. At `k = 2` (5 coupled sectors of order
2–6) polar is 3–4 µs slower (~5–10 %): it now makes two dense GEMM calls per coupled sector
(`W`, `P`), and a `sample` profile of the head build shows the provider's per-call session entry
and descriptor setup in each `dot_general_into`. That the old contraction amortized this over
fewer provider calls is the likely cause, not measured separately. `pinv` makes one call per
sector and still wins by removing the factor copies.
Batching the per-sector GEMMs of these kernels into one provider call is the remaining constant
factor; no CI timing gate.

## Streaming polar (review follow-up)

`streaming_polar/` compares the staged polar (`3ae2b6bb`, the rebased candidate above) with the
streaming one (each sector: SVD → `W`/`P` GEMMs → landing, then dropped), same protocol on the
harness revision with peak live bytes and a third shape, U(1)/SU(2) with 7 leg sectors, `k = 2`
(harness SHA-256 `f98cb218b8e8c40e810e23c8eb4b3382d4a98f2dce6645b5ca7167f646ba5db6`). The
staged working set is `outputs + Σ_c (U_c, S_c, Vh_c)`, the streaming one `outputs + max_c` of a
sector's factors and the provider's SVD scratch. With 7 leg sectors peak live bytes fall 8–34 %
(e.g. SU(2) f64 left polar 1 390 600 → 966 920). With 3 leg sectors (5 coupled sectors) they rise
(e.g. U(1) f64 `k = 2` 17 560 → 21 944, SU(2) c64 `k = 6` 5 000 712 → 5 757 640): there the
provider's transient SVD scratch, which the staged loop paid before the outputs were allocated,
now overlaps the live outputs, and it exceeds the few staged factors it replaces. Time is
unchanged (geomean 0.995–1.003 over the polar rows); allocation calls are equal and bytes 480 B lower.
The SVD factors themselves remain provider-owned per call: the executor's `svd_into` has only the
default copy-through-owned implementation, so a reused factor buffer would add a copy, not remove
an allocation.
