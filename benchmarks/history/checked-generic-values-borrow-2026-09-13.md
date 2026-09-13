# Checked Generic values input borrowing (2026-09-13)

Issue: #1166. Checked Generic SVD-, Hermitian-eigen-, and general-eigenvalue
operations now use the existing canonical coupled-sector regions. Canonical
matrices borrow the admitted tensor payload; padded or reordered expert layouts
retain the existing packer.

## Immutable source states

- Production base: `d729763bc045829b04c6e42b5d443bb434c43e7f`.
- Baseline plus measurement harness:
  `445173d5e6c4930509864081c052cc543b4a7c65`.
- Candidate plus the same measurement harness:
  `18a54159a91f0960558da2785b458a8c0b1949e5`.
- Harness SHA-256 in both measured states:
  `a74829c42d74ce76c9a31016adc4cc02a4268e49bc2a9ddb8aa00e4b972f39c3`.
- Copied `benchmarks/cpu_shape_reuse.Cargo.lock` SHA-256:
  `455947c958ea1306757498b66f3eaa74c6d7bdbdccb4e9988ae2ffa15b4aeb41`.

The lock resolves Tenferro 0.3.0 and Racah 0.1.1. No dependency changed.

## Protocol and boundary

```text
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
CARGO_INCREMENTAL=0 CARGO_PROFILE_RELEASE_DEBUG=0 \
TENET_GENERIC_VALUES_BORROW_CASES=1 \
TENET_GENERIC_ASSEMBLER_ITERS=2000 TENET_GENERIC_ASSEMBLER_SAMPLES=9 \
CARGO_TARGET_DIR=/Users/ryowatanabe/Research/codes/MyTensorNetworks/libraries/tenet/target \
cargo run --locked --offline --release -p tenet-matrixalgebra \
  --features cpu-faer --example generic_assembler_measurement
```

Machine: Apple M4 Max, 16 cores, 48 GB, `aarch64-apple-darwin`, macOS 15.5;
rustc 1.96.0, LLVM 22.1.2. Each case received a 100 ms untimed warmup, then
nine timing samples and separate allocation samples of 2,000 calls.

Fixture construction and checked admission are outside measurement. The timed
prefix includes `BoundDynamicTensorRef` construction, checked routing, matrix
selection or packing, the rejecting first dense call, error propagation, and
destruction. `black_box` consumes the dense input view at that boundary. It
does not run a dense numerical kernel or measure full factorization. Allocation
figures count requests and requested bytes, not successful or live bytes. The
rank-four provider is synthetic and structurally admitted; it is not a physical
category and uses no Racah coefficients.

## Median results

| Layout and shape | Baseline ns | Candidate ns | Allocation calls | Requested bytes |
|---|---:|---:|---:|---:|
| canonical, `T=2`, degeneracy 3 | 966.645 | 31.895 | 13 -> 1 | 4,032 -> 32 |
| padded, `T=2`, degeneracy 3 | 982.292 | 983.292 | 13 -> 13 | 4,032 -> 4,032 |
| canonical, `T=8`, degeneracy 1 | 7,392.895 | 77.334 | 87 -> 1 | 5,728 -> 32 |
| padded, `T=8`, degeneracy 1 | 7,578.875 | 7,221.729 | 87 -> 87 | 5,728 -> 5,728 |

The canonical cases remove the TeNeT-owned matrix payload pack in this measured
prefix. The padded controls keep identical allocation counts and requested
bytes; their timing difference is not attributed to an algorithm change. The
remaining one allocation and 32 requested bytes belong to the returned dense
error path. These results do not claim that Tenferro performs no internal copy
or that complete SVD/eigenvalue execution improves by the same ratio.

Raw samples are in
`checked-generic-values-borrow-baseline-2026-09-13.csv` and
`checked-generic-values-borrow-candidate-2026-09-13.csv`.
