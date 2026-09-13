# Generic sector assembler experiment (2026-09-13)

Issue: #1161. This experiment tested replacing the Generic factorization
assembler's repeated linear tree lookup with the existing hash-indexed
assembler. The production change was rejected: it improved the largest
synthetic tree grid but slowed every smaller control and increased allocation
requests and requested bytes in every case.

## Immutable source states

- Production authority: `2d0b129dabc569303b4a4bdc78f00164d05d47b4`.
- Corrected baseline plus test/diagnostic:
  `1dc528882ec88a3b339c02dc4a14bf3e8e167875`.
- Experimental wrapper-only candidate:
  `f933de0ae0506a82871d15eb593b56afa5016cb6`.
- Diagnostic SHA-256 in both measured states:
  `6f8b9691c4d809c224fd4755b6c7d2c986e2e95f98660ccc765cdf29d7f9af18`.
- Copied `benchmarks/cpu_shape_reuse.Cargo.lock` SHA-256:
  `455947c958ea1306757498b66f3eaa74c6d7bdbdccb4e9988ae2ffa15b4aeb41`.

The candidate differs from the corrected baseline only in
`tenet-matrixalgebra/src/factorize.rs` (4 insertions, 106 deletions). The
diagnostic passes its dense matrix input through `black_box` at the rejecting
executor boundary so matrix payload stores remain observable.

## Protocol

```text
RAYON_NUM_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1 CARGO_PROFILE_RELEASE_DEBUG=0 \
TENET_GENERIC_ASSEMBLER_ITERS=10000 TENET_GENERIC_ASSEMBLER_SAMPLES=9 \
CARGO_TARGET_DIR=/Users/ryowatanabe/Research/codes/MyTensorNetworks/libraries/tenet/target \
cargo run --locked --offline --release -p tenet-matrixalgebra \
  --example generic_assembler_measurement --no-default-features --features cpu-faer
```

Machine: Apple M4 Max (`aarch64-apple-darwin`), macOS 15.5; rustc 1.96.0,
LLVM 22.1.2. Threads were pinned to one. The lock resolves Racah 0.1.1; the
assembler makes no provider or Racah call.

Each case used 16 warmups and nine samples of 10,000 calls. Fixture admission
was outside the scope. The measured prefix includes input binding, checked
`svd_vals_dyn_checked_generic` routing, assembly, a rejecting first dense call,
error propagation, and destruction. Timing and allocation samples were
separate. Allocation results count requests and requested bytes, not successful
or live bytes. Fixtures use a synthetic admitted rank-four provider; they do
not represent a physical category, cross-rank performance, dense numerical
work, or end-to-end factorization.

## Median results

Positive latency deltas are slower.

| Case | Baseline ns | Candidate ns | Delta | Allocation calls | Requested bytes |
|---|---:|---:|---:|---:|---:|
| one sector, `T=1`, degeneracy 1 | 341.167 | 429.629 | +25.9% | 8 -> 14 | 1,320 -> 1,956 |
| one sector, `T=2`, degeneracy 1 | 389.246 | 804.200 | +106.6% | 13 -> 19 | 1,472 -> 2,180 |
| one sector, `T=4`, degeneracy 1 | 1,348.192 | 2,394.454 | +77.6% | 29 -> 37 | 2,016 -> 3,300 |
| one sector, `T=8`, degeneracy 1 | 7,215.508 | 8,065.821 | +11.8% | 87 -> 97 | 5,728 -> 8,724 |
| one sector, `T=16`, degeneracy 1 | 42,487.621 | 27,855.954 | -34.4% | 297 -> 309 | 16,992 -> 25,700 |
| one sector, `T=2`, degeneracy 3 | 961.150 | 1,372.333 | +42.8% | 13 -> 19 | 4,032 -> 4,740 |
| interleaved two sector, `T=8` | 15,107.350 | 17,245.758 | +14.2% | 172 -> 188 | 11,072 -> 16,604 |
| few-large, `T=1`, degeneracy 4 | 576.808 | 729.396 | +26.5% | 8 -> 14 | 3,360 -> 3,996 |

The `T=1` samples showed strong within-run frequency drift, so its exact
percentage is not a stable steady-state estimate. The other controls and the
allocation counts are sufficient to reject a broad production switch. No size
threshold, cache, dependency change, or alternative production implementation
was introduced.

Raw samples are in
`generic-sector-assembler-baseline-2026-09-13.csv` and
`generic-sector-assembler-candidate-2026-09-13.csv`.
