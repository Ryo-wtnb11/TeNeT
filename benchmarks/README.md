# TeNeT vs TensorKit contraction microbench

Same workload on both sides: rank-4 tensors `A, B` in `V ⊗ V ← V ⊗ V`,
uniform degeneracy `d` per sector, warm structure caches, single-threaded.
**Both sides use the same BLAS (Apple Accelerate)**: TeNeT built with
`--no-default-features --features blas-accelerate`, Julia switched from the
default OpenBLAS via `using AppleAccelerate`. Comparing across different BLAS
implementations is misleading on Apple silicon — Accelerate's AMX GEMM is
5–15x faster than OpenBLAS at these sizes, which initially masqueraded as a
TeNeT win at d=16.

- `compose`:  `C[a b; g h] = A[a b; c d] * B[c d; g h]` (canonical route, no tree transforms)
- `swap`:     `C[a b; g h] = A[a b; c d] * B[d c; g h]` (source tree transforms)
- `swap+out`: `C[b a; g h] = A[a b; c d] * B[d c; g h]` (plus output transform)

Symmetries: U(1) `{-1,0,1}`, fZ2 `{0,1}`, SU(2) `{0,1/2,1}`, U(1)⊠fZ2
`{(-1,1),(0,0),(1,1)}`.

Run:

```
cargo build --release --example microbench_fusion -p tenet-operations \
    --no-default-features --features blas-accelerate
RAYON_NUM_THREADS=1 [MICROBENCH_PROFILE=1] target/release/examples/microbench_fusion <d> <min_ms>
julia -e 'push!(ARGS,"<d>","<min_ms>"); using AppleAccelerate; include("benchmarks/tensorkit_microbench.jl")'
```

## Results (2026-07-02, Apple M-series, Julia 1.11.6 / TensorKit 0.16.2, Accelerate both sides)

µs per iteration.

### d = 4 (small blocks)

| symmetry | workload | TeNeT | TensorKit | ratio |
|---|---|---:|---:|---:|
| U1 | compose | 133.9 | 4.0 | 33 |
| U1 | swap | 169.3 | 20.0 | 8.5 |
| U1 | swap+out | 124.1 | 34.5 | 3.6 |
| fZ2 | compose | 61.8 | 1.9 | 33 |
| fZ2 | swap | 80.2 | 8.5 | 9.4 |
| fZ2 | swap+out | 96.9 | 14.9 | 6.5 |
| SU2 | compose | 309.5 | 7.3 | 42 |
| SU2 | swap | 424.5 | 44.1 | 9.6 |
| SU2 | swap+out | 526.7 | 78.4 | 6.7 |
| U1⊠fZ2 | compose | 128.0 | 4.1 | 31 |
| U1⊠fZ2 | swap | 168.3 | 19.5 | 8.6 |
| U1⊠fZ2 | swap+out | 124.0 | 34.6 | 3.6 |

### d = 16 (large blocks)

| symmetry | workload | TeNeT | TensorKit | ratio |
|---|---|---:|---:|---:|
| U1 | compose | 22 889 | 1 844 | 12.4 |
| U1 | swap | 30 434 | 4 033 | 7.5 |
| U1 | swap+out | 23 135 | 4 709 | 4.9 |
| fZ2 | compose | 10 502 | 679 | 15.5 |
| fZ2 | swap | 13 557 | 1 415 | 9.6 |
| fZ2 | swap+out | 16 692 | 2 022 | 8.3 |
| SU2 | compose | 68 870 | 7 981 | 8.6 |
| SU2 | swap | 88 413 | 15 779 | 5.6 |
| SU2 | swap+out | 107 351 | 19 914 | 5.4 |
| U1⊠fZ2 | compose | 22 961 | 1 857 | 12.4 |
| U1⊠fZ2 | swap | 30 455 | 3 624 | 8.4 |
| U1⊠fZ2 | swap+out | 23 053 | 4 771 | 4.8 |

Reference (TeNeT with default `cpu-faer`, 1 thread): U1 compose d=16 is
46 880 — the faer GEMM alone (27 158) costs as much as TensorKit's whole
OpenBLAS iteration. GEMM backend choice matters, but it is not the main gap.

## Profile decomposition (MICROBENCH_PROFILE=1, Accelerate)

U1 `compose`, d=16: total 22 955 — **pack 13 049 + scatter 6 511 (85 %)**,
matmul 3 339, plan/cache lookups 3.6, scratch prepare 51.
U1 `compose`, d=4: total ~115 — pack+scatter ≈ 90 %, matmul ≈ 9,
plan/cache lookups 0.5.

TeNeT's matmul leg (3 339) is also ~1.8x TensorKit's entire `mul!` iteration
(1 847) — same five coupled-sector GEMMs, so there is a secondary GEMM-call
overhead worth a look once packing is gone.

## Coupled-sector layout results (same date, after `from_degeneracy_shapes_coupled` + direct GEMM)

With tensors and dynamic-route canonical scratch in the coupled-sector matrix
layout (`MICROBENCH_LAYOUT=coupled`), the fusion-block plan detects that each
group matrix already exists in storage and hands it to GEMM directly — no
pack, no scatter (locked by `coupled_layout_compose_uses_direct_gemm_groups`).

µs per iteration, Accelerate both sides:

| symmetry | workload | TeNeT packed | TeNeT coupled | TensorKit | coupled/TK |
|---|---|---:|---:|---:|---:|
| U1 | compose d=4 | 133.9 | 6.4 | 4.0 | 1.6 |
| U1 | compose d=16 | 22 889 | 1 846 | 1 844 | **1.00** |
| fZ2 | compose d=16 | 10 502 | 678 | 679 | **1.00** |
| SU2 | compose d=16 | 68 870 | 7 738 | 7 981 | **0.97** |
| U1⊠fZ2 | compose d=16 | 22 961 | 1 965 | 1 857 | 1.06 |
| U1 | swap d=16 | 30 434 | 15 908 | 4 033 | 3.9 |
| fZ2 | swap d=16 | 13 557 | 7 771 | 1 415 | 5.5 |
| SU2 | swap d=16 | 88 413 | 58 380 | 15 779 | 3.7 |
| U1 | swap+out d=16 | 23 135 | 18 463 | 4 709 | 3.9 |

Profile (U1 d=16, coupled): `compose` total 1 834 = matmul 1 832 —
pack/scatter are zero. `swap` total 15 920 = source tree transforms 13 833
(87 %) + matmul 1 965. The contraction leg is done; the remaining gap is the
tree-transform replay itself (TensorKit's permute-equivalent costs ~2.2 ms on
the same workload), which is the next optimization target — the same
pack/recoupling/scatter machinery now behind `HostKernelAdapter`.

## Conclusions

- **With identical BLAS, TeNeT is 3.6–42x slower**; the gap is the per-call
  pack/GEMM/scatter replay. Plan/cache lookup (<0.1 %), scratch allocation,
  strided kernel speed, and threading are all non-causes.
- TensorKit stores each coupled-sector block as one contiguous column-major
  matrix (rows = codomain trees × degeneracy, cols = domain trees ×
  degeneracy), so `compose` is a direct per-sector GEMM with zero packing,
  and transforms materialize into temporaries that are again GEMM-ready.
  TeNeT's per-fusion-tree subblock layout re-packs the coupled matrix on
  every contraction.
- The `swap+out` < `compose` anomaly for U1 is real: the canonical
  destination has 1 matrix group instead of 5, so less pack/scatter.
- Missing pieces for TensorKit-parity, in dependency order:
  1. Coupled-sector contiguous block layout in `BlockStructure` (subblock
     offsets/strides embedded in the per-sector matrix) plus a `block(c)`
     matrix view accessor.
  2. Canonical fusion-block plan emitting per-sector GEMM calls on block
     views (alpha/beta into destination blocks) instead of pack/GEMM/scatter;
     keep packing only as a fallback for non-contiguous cases.
  3. Dynamic-route canonical scratch tensors laid out in the same
     sector-matrix form so transformed sources feed GEMM directly.
  4. Per-GEMM call overhead audit (TeNeT's matmul leg is ~1.8x TensorKit's
     `mul!` for identical block shapes).
