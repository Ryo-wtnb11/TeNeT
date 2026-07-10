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

- tenet commit: `906447dd80f8dbaa03af8c804d6b1a3764d17f0c` (`main`, includes #102
  Stage B1 generic-fusion braid and the #104 d=4 per-call-overhead fix)
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
| U1 | compose | 6.50 | 3.96 | 1.64 |
| U1 | swap | 12.63 | 18.82 | **0.67** |
| U1 | swap+out | 14.57 | 34.21 | **0.43** |
| fZ2 | compose | 2.34 | 1.86 | 1.26 |
| fZ2 | swap | 4.39 | 8.30 | **0.53** |
| fZ2 | swap+out | 7.30 | 14.17 | **0.52** |
| SU2 | compose | 10.62 | 7.45 | 1.43 |
| SU2 | swap | 33.32 | 44.78 | **0.74** |
| SU2 | swap+out | 49.65 | 78.02 | **0.64** |
| U1⊠fZ2 | compose | 6.37 | 3.96 | 1.61 |
| U1⊠fZ2 | swap | 12.64 | 19.25 | **0.66** |
| U1⊠fZ2 | swap+out | 14.64 | 34.21 | **0.43** |

### d = 16 (large blocks)

| symmetry | workload | TeNeT | TensorKit | ratio |
|---|---|---:|---:|---:|
| U1 | compose | 1 977.3 | 1 928.6 | 1.03 |
| U1 | swap | 2 721.8 | 4 167.2 | **0.65** |
| U1 | swap+out | 3 348.0 | 5 141.5 | **0.65** |
| fZ2 | compose | 689.6 | 695.1 | 0.99 |
| fZ2 | swap | 1 003.1 | 1 429.9 | **0.70** |
| fZ2 | swap+out | 1 272.4 | 1 993.1 | **0.64** |
| SU2 | compose | 8 242.7 | 7 953.3 | 1.04 |
| SU2 | swap | 12 239.2 | 17 040.7 | **0.72** |
| SU2 | swap+out | 15 598.5 | 19 519.4 | **0.80** |
| U1⊠fZ2 | compose | 1 981.4 | 1 935.9 | 1.02 |
| U1⊠fZ2 | swap | 2 782.6 | 3 736.3 | **0.74** |
| U1⊠fZ2 | swap+out | 3 517.2 | 5 364.8 | **0.66** |

At d=16 every workload is at parity or faster than TensorKit (compose is
GEMM-bound and matches `mul!` to ~4%; swap/swap+out beat TensorKit by
20–36% via the GEMM-based recoupling replay — swap/swap+out improved
another 8–13% over the pre-#104 measurement from the hybrid
stack/thread_local fused-pair dispatch). At d=4, compose is 1.26–1.64x
TensorKit (unchanged by #104 — see the residual note below); swap/swap+out
are 0.43–0.74x (faster than TensorKit), recovered from 0.57–0.84x pre-#104
now that the thread_local-scratch regression from #101/12748cf is fixed
for rank ≤ 8 (the d=4/d=16 workloads here).

**Known residual ([issue #103](https://github.com/Ryo-wtnb11/TeNeT/issues/103)):**
d=4 compose (and part of swap) still carries roughly +2.7µs / +1.4µs from an
unrelated change bundled in the same original commit: `tenet-dense` dropped
its `STRIDED_BATCH_MIN_JOBS` floor and loosened the strided-batch GEMM
routing condition to `jobs.len() >= 2` / `run_len > 1`, so d=4's small
5-group GEMMs now take the strided-batch seam, which is ~60% slower
per-call than the direct path at this size. Bisect first misattributed this
to eager error-struct construction in `recoupling_multi_block`; profiling
during the #104 fix showed that path never executes for U1 d=4 and traced
the regression to `core_matmul` itself, i.e. the GEMM-routing threshold.
#104 fixed the larger, independent thread_local-scratch regression
(swap/swap+out) via a hybrid dispatch; this GEMM-routing threshold is a
separate design decision left open in #103.

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
