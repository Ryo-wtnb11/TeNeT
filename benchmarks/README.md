# TeNeT vs TensorKit contraction microbench

Same workload on both sides: rank-4 tensors `A, B` in `V ⊗ V ← V ⊗ V`,
uniform degeneracy `d` per sector, warm structure caches, single-threaded
(BLAS 1 thread on Julia, `RAYON_NUM_THREADS=1` on Rust).

- `compose`:  `C[a b; g h] = A[a b; c d] * B[c d; g h]` (canonical route, no tree transforms)
- `swap`:     `C[a b; g h] = A[a b; c d] * B[d c; g h]` (source tree transforms)
- `swap+out`: `C[b a; g h] = A[a b; c d] * B[d c; g h]` (plus output transform)

Symmetries: U(1) `{-1,0,1}`, fZ2 `{0,1}`, SU(2) `{0,1/2,1}`, U(1)⊠fZ2
`{(-1,1),(0,0),(1,1)}`.

Run:

```
cargo run --release --example microbench_fusion -p tenet-operations -- <d> <min_ms>
MICROBENCH_PROFILE=1 RAYON_NUM_THREADS=1 target/release/examples/microbench_fusion <d> <min_ms>
julia benchmarks/tensorkit_microbench.jl <d> <min_ms>
```

## Results (2026-07-02, Apple M-series, Julia 1.11.6 / TensorKit 0.16.2)

µs per iteration.

### d = 4 (small blocks, overhead regime)

| symmetry | workload | TeNeT (faer 1T) | TensorKit | ratio |
|---|---|---:|---:|---:|
| U1 | compose | 121.7 | 8.9 | 13.7 |
| U1 | swap | 153.7 | 23.2 | 6.6 |
| U1 | swap+out | 120.0 | 34.9 | 3.4 |
| fZ2 | compose | 51.6 | 3.3 | 15.6 |
| fZ2 | swap | 72.0 | 9.4 | 7.7 |
| fZ2 | swap+out | 88.5 | 14.3 | 6.2 |
| SU2 | compose | 306.9 | 44.2 | 6.9 |
| SU2 | swap | 414.0 | 73.6 | 5.6 |
| SU2 | swap+out | 518.3 | 101.4 | 5.1 |
| U1⊠fZ2 | compose | 113.5 | 9.2 | 12.3 |
| U1⊠fZ2 | swap | 154.8 | 34.5 | 4.5 |
| U1⊠fZ2 | swap+out | 121.9 | 36.8 | 3.3 |

### d = 16 (large blocks, GEMM regime)

| symmetry | workload | TeNeT (faer 1T) | TeNeT (Accelerate) | TensorKit | ratio (faer) |
|---|---|---:|---:|---:|---:|
| U1 | compose | 46 880 | 22 048 | 27 293 | 1.72 |
| U1 | swap | 54 534 | | 31 755 | 1.72 |
| U1 | swap+out | 37 530 | | 30 774 | 1.22 |
| fZ2 | compose | 19 027 | | 9 863 | 1.93 |
| SU2 | compose | 153 108 | 59 874 | 99 781 | 1.53 |
| U1⊠fZ2 | compose | 48 803 | | 27 508 | 1.77 |

## Profile decomposition (MICROBENCH_PROFILE=1)

U1 `compose`, d=4: total 114.6 — pack 68.5 + scatter 34.5 (**90 %**),
matmul 9.0, plan/cache lookups 0.5, scratch prepare 0.9.

U1 `compose`, d=16: total 46 615 — matmul 27 158, pack 12 934, scatter 6 468.
TeNeT's matmul alone (27.2 ms with faer) equals TensorKit's entire iteration
(27.3 ms): the whole gap is pack/scatter.

## Conclusions

- **Cause of the gap is the per-call pack/GEMM/scatter replay**, not plan or
  cache lookup (<1 %), not scratch allocation, not the strided kernel speed,
  not threading. TensorKit stores each coupled-sector block as one contiguous
  matrix, so `compose` is a direct per-sector GEMM with zero packing; TeNeT's
  per-fusion-tree subblock layout re-packs the coupled matrix on every
  contraction.
- The `swap+out` anomaly (faster than `compose` for U1) is real: the canonical
  destination has fewer, larger matrix groups (1 vs 5), so less pack/scatter.
- GEMM backend replacement works as designed: with `blas-accelerate`, TeNeT
  `compose` at d=16 is already faster than TensorKit (22.0 vs 27.3 ms U1,
  59.9 vs 99.8 ms SU2) despite the pack/scatter overhead.
- Next optimization directions, in order of leverage:
  1. Pack-free fast path when a group's subblocks are already contiguous
     columns of the group matrix (alias instead of copy).
  2. TensorKit-style coupled-sector block layout so the canonical
     matricization is contiguous by construction.
  3. Reduce scatter cost via beta-aware direct-GEMM into destination views
     when strides permit.
