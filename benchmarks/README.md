# TeNeT vs TensorKit contraction microbench

Same workload on both sides: rank-4 tensors `A, B` in `V ⊗ V ← V ⊗ V`,
uniform degeneracy `d` per sector, warm structure caches, single-threaded.
**Both sides use the same BLAS (Apple Accelerate)**: TeNeT built with
`--no-default-features --features blas-accelerate`, Julia switched from the
default OpenBLAS via `using AppleAccelerate`. Comparing across different BLAS
implementations is misleading on Apple silicon — Accelerate's AMX GEMM is
5–15x faster than OpenBLAS at these sizes, which initially masqueraded as a
TeNeT win at d=16.

- `compose`:  `C[a b; g h] = A[a b; c d] * B[c d; g h]` (core-form route, no tree transforms)
- `swap`:     `C[a b; g h] = A[a b; c d] * B[d c; g h]` (source tree transforms)
- `swap+out`: `C[b a; g h] = A[a b; c d] * B[d c; g h]` (plus output transform)

Symmetries: U(1) `{-1,0,1}`, fZ2 `{0,1}`, SU(2) `{0,1/2,1}`, U(1)⊠fZ2
`{(-1,1),(0,0),(1,1)}`.

Run:

```
cargo build --release --example microbench_fusion -p tenet-tensors \
    --no-default-features --features blas-accelerate
RAYON_NUM_THREADS=1 [MICROBENCH_PROFILE=1] target/release/examples/microbench_fusion <d> <min_ms>
julia -e 'push!(ARGS,"<d>","<min_ms>"); using AppleAccelerate; include("benchmarks/tensorkit_microbench.jl")'
```

## Results (2026-07-11)

- tenet commit: `b5b5d82f4ec5495a25f05ef0d55686a4c616701e` (`main`, includes the
  #106 plan-time per-run GEMM routing fix)
- tenferro-rs pairing: `main` worktree at `d5c768c7` (path-dep symlink target)
- Julia 1.11.6, TensorKit v0.16.2 (`QuantumKitHub/TensorKit.jl#ld-mooncakerules`), AppleAccelerate.jl v0.7.0
  — TensorKit columns unchanged from the 2026-07-11 measurement (TK side has no
  code change here, so it was not re-run)
- Accelerate BLAS both sides, `RAYON_NUM_THREADS=1`
- TeNeT columns: median of 3 runs (300ms min per cell) on a quiet machine

µs per iteration.

### d = 4 (small blocks)

| symmetry | workload | TeNeT | TensorKit | ratio |
|---|---|---:|---:|---:|
| U1 | compose | 4.10 | 3.96 | 1.04 |
| U1 | swap | 9.72 | 18.82 | **0.52** |
| U1 | swap+out | 12.43 | 34.21 | **0.36** |
| fZ2 | compose | 2.09 | 1.86 | 1.12 |
| fZ2 | swap | 4.19 | 8.30 | **0.50** |
| fZ2 | swap+out | 7.01 | 14.17 | **0.49** |
| SU2 | compose | 7.19 | 7.45 | 0.97 |
| SU2 | swap | 30.02 | 44.78 | **0.67** |
| SU2 | swap+out | 44.63 | 78.02 | **0.57** |
| U1⊠fZ2 | compose | 3.93 | 3.96 | 0.99 |
| U1⊠fZ2 | swap | 9.70 | 19.25 | **0.50** |
| U1⊠fZ2 | swap+out | 12.27 | 34.21 | **0.36** |

### d = 16 (large blocks)

| symmetry | workload | TeNeT | TensorKit | ratio |
|---|---|---:|---:|---:|
| U1 | compose | 1 886.1 | 1 928.6 | 0.98 |
| U1 | swap | 2 645.3 | 4 167.2 | **0.63** |
| U1 | swap+out | 3 281.0 | 5 141.5 | **0.64** |
| fZ2 | compose | 678.6 | 695.1 | 0.98 |
| fZ2 | swap | 988.7 | 1 429.9 | **0.69** |
| fZ2 | swap+out | 1 265.9 | 1 993.1 | **0.64** |
| SU2 | compose | 7 741.3 | 7 953.3 | 0.97 |
| SU2 | swap | 11 915.0 | 17 040.7 | **0.70** |
| SU2 | swap+out | 15 075.6 | 19 519.4 | **0.77** |
| U1⊠fZ2 | compose | 1 881.4 | 1 935.9 | 0.97 |
| U1⊠fZ2 | swap | 2 662.0 | 3 736.3 | **0.71** |
| U1⊠fZ2 | swap+out | 3 283.2 | 5 364.8 | **0.61** |

At d=16 every workload is at parity or faster than TensorKit (compose is
GEMM-bound and matches `mul!` to ~3%; swap/swap+out beat TensorKit by
23–39% via the GEMM-based recoupling replay). At d=4, compose is now at
TensorKit parity (0.97–1.12x) and swap/swap+out are 0.36–0.67x (faster
than TensorKit).

The former d=4 compose residual (the strided-batch GEMM-routing seam,
[issue #103](https://github.com/Ryo-wtnb11/TeNeT/issues/103)) was eliminated
by the plan-time per-run GEMM routing in
[#106](https://github.com/Ryo-wtnb11/TeNeT/pull/106), restoring d=4 compose
to TensorKit parity and closing #103.

### Cold (structure compile) baseline

First call per workload (all structure caches cold), d=16: U1 compose 2.9 ms
/ swap 4.4 ms / swap+out 5.3 ms; SU2 compose 9.5 ms / swap 18.1 ms /
swap+out 21.0 ms — consistent with the earlier-recorded cold costs (compile
adds roughly one warm iteration at this scale).

### Profile decomposition (MICROBENCH_PROFILE=1, Accelerate)

U1 `compose`, d=16: total 1 903 — pure GEMM (matmul 1 903, pack/scatter 0),
matching TensorKit's `mul!` route exactly. U1 `swap`, d=16: total 2 804 =
tree-transform replay 718 (26%) + matmul 1 983 (71%) + scratch prep 102.
SU2 `swap+out`, d=16: total 17 044 = tree-transform replay 4 957 (29%) +
matmul 8 040 (47%) + output transform 3 692 (22%). At d=4 the same shape
holds proportionally (e.g. U1 `swap` total 16.0 = replay 6.2 + matmul 6.9 +
scratch prep 2.2) — no single stage dominates once packing is gone; the
gap left is per-call overhead, not a missing kernel.

## SVD cross-check against TensorKit

The Rust and Julia cross-check scripts fill every fusion-tree pair block of a
`V x V <- V x V` tensor with the same integer-hash function of the sector
labels and degeneracy indices, then print per-coupled-sector singular values
(invariant under tree ordering and per-tree basis conventions). U(1)
`{-1,0,1}` and SU(2) `{0,1/2}` at degeneracy 2: all 8 sector spectra agree
with TensorKit's `svd_compact` to 10 printed digits (max deviation 0.0) —
validating the fusion-space structure and the blockwise SVD end to end.

## History

The full chronological optimization history (packed-layout baseline →
coupled layout → strided-rs replay → fused loops → GEMM recoupling) lives
in the git history of this file.
