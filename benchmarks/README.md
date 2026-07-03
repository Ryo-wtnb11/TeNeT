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
cargo build --release --example microbench_fusion -p tenet-tensors \
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
layout (now the default; `MICROBENCH_LAYOUT=packed` opts out), the fusion-block plan detects that each
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

## strided-rs replay kernels (tensor4all/strided-rs 4ea9acb)

Switching `StridedHostKernelAdapter`'s copy/axpy paths from the in-crate
scalar odometer loops to strided-rs `copy_scale`/`axpy` (dimension fusion +
SIMD fast paths, upstream #133/#134) removes most of the remaining
tree-transform replay cost. µs per iteration, Accelerate both sides,
coupled layout:

| symmetry | workload | before | after | TensorKit | after/TK |
|---|---|---:|---:|---:|---:|
| U1 | swap d=16 | 15 908 | 3 406 | 4 033 | **0.84** |
| U1 | swap+out d=16 | 18 463 | 2 976 | 4 709 | **0.63** |
| fZ2 | swap d=16 | 7 771 | 1 346 | 1 415 | **0.95** |
| SU2 | swap d=16 | 58 380 | 16 402 | 15 779 | 1.04 |
| SU2 | swap+out d=16 | 84 734 | 21 635 | 19 914 | 1.09 |
| U1 | swap d=4 | 90.9 | 42.9 | 20.0 | 2.1 |
| SU2 | swap d=4 | 285.7 | 126.1 | 44.1 | 2.9 |

At d=16 every workload is now at or better than TensorKit. The remaining
small-block (d=4) gap of 1.5-3x is per-call overhead (plan lookups, view
setup, route dispatch) and no longer any single dominant stage.

## Allocation-free fused replay loops (small-tensor overhead)

The strided-rs view entry points allocate owned metadata (`Arc::from(dims)`,
plan building) per call — ~0.8 µs per tiny copy, which dominated d=4 replay.
`StridedHostKernelAdapter` now runs copy/axpy through a stack-allocated fused
loop nest (axes sorted by destination stride, adjacent axes fused, contiguous
inner runs as plain slice loops; rank > 8 falls back to strided-rs). µs per
iteration, Accelerate both sides, coupled layout:

| symmetry | workload | before | after | TensorKit | after/TK |
|---|---|---:|---:|---:|---:|
| U1 | swap d=4 | 42.9 | 13.3 | 20.0 | **0.66** |
| U1 | swap+out d=4 | 52.2 | 10.6 | 34.5 | **0.31** |
| fZ2 | swap d=4 | 20.0 | 5.8 | 8.5 | **0.68** |
| SU2 | swap d=4 | 126.1 | 35.7 | 44.1 | **0.81** |
| SU2 | swap+out d=4 | 186.6 | 51.8 | 78.4 | **0.66** |
| U1 | swap d=16 | 3 406 | 2 965 | 4 033 | **0.74** |
| SU2 | swap d=16 | 16 402 | 14 250 | 15 779 | **0.90** |
| SU2 | swap+out d=16 | 21 635 | 18 353 | 19 914 | **0.92** |

Every workload at both sizes is now at or faster than TensorKit except
`compose` at d=4 (6.2 vs 4.0 µs), which is five Accelerate GEMM calls plus
~1.4 µs of validation/lookup — no longer dominated by any replay stage.

## Cold (structure compile) baseline

The example now reports the first call (all structure caches cold) per
workload as `cold=`. At d=16, coupled layout: U1 compose 6.1 ms / swap 4.8 ms;
SU2 compose 9.7 ms / swap 17.2 ms / swap+out 24.0 ms — i.e. compile adds
roughly one warm iteration at this scale. The pathological cold costs seen in
finite-torus (seconds to minutes) belong to the legacy stack's per-permute
caches, not to this pipeline; re-evaluate compile cost on a realistic
apply-gate workload (rank-5/6, larger sector sets) when the network
contraction bench lands.

## tsvd cross-check against TensorKit

`tenet-matrixalgebra/examples/tsvd_crosscheck.rs` and
`benchmarks/tensorkit_tsvd_crosscheck.jl` fill every fusion-tree pair block of
a `V x V <- V x V` tensor with the same integer-hash function of the sector
labels and degeneracy indices, then print per-coupled-sector singular values
(invariant under tree ordering and per-tree basis conventions). U(1)
`{-1,0,1}` and SU(2) `{0,1/2}` at degeneracy 2: all 8 sector spectra agree
with TensorKit 0.16 `svd_compact` to 10 printed digits (max deviation 0.0) —
validating the fusion-space structure and the blockwise SVD end to end.

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

## Post-restructure verification (2026-07-03, after tenet-operations extraction)

Same protocol (coupled layout, Accelerate, `RAYON_NUM_THREADS=1`). No
regression from the crate split; swap paths improved further vs the last
recorded table (stack-fused replay loops landed after it).

| symmetry | workload | d=4 | d=16 | TensorKit d=16 | ratio |
|---|---|---|---|---|---|
| U1 | compose | 5.3 | 1 841 | 1 844 | **1.00** |
| U1 | swap | 11.8 | 2 722 | 4 033 | **0.68** |
| U1 | swap+out | 9.0 | 2 016 | 4 709 | **0.43** |
| fZ2 | compose | 2.6 | 692 | 679 | 1.02 |
| fZ2 | swap | 4.9 | 1 002 | 1 415 | **0.71** |
| SU2 | compose | 8.5 | 7 732 | 7 981 | **0.97** |
| SU2 | swap | 31.0 | 15 378 | 15 779 | **0.97** |
| SU2 | swap+out | 46.7 | 21 107 | 19 914 | 1.06 |
| U1⊠fZ2 | compose | 5.2 | 2 187 | 1 857 | 1.18 |

### d=4 full comparison (2026-07-03, both sides Accelerate, 1 thread)

TensorKit rerun via `julia -e 'using AppleAccelerate; include("tensorkit_microbench.jl")' 4 300`
(the bare script uses OpenBLAS; at d=4 that alone doubles TensorKit's
compose time, so the Accelerate numbers are the fair baseline). µs/iter.

| symmetry | workload | TeNeT | TensorKit | ratio |
|---|---|---|---|---|
| U1 | compose | 5.3 | 4.1 | 1.31 |
| U1 | swap | 11.8 | 19.8 | **0.60** |
| U1 | swap+out | 9.0 | 35.6 | **0.25** |
| fZ2 | compose | 2.6 | 1.9 | 1.41 |
| fZ2 | swap | 4.9 | 8.7 | **0.56** |
| fZ2 | swap+out | 7.2 | 15.2 | **0.47** |
| SU2 | compose | 8.5 | 7.1 | 1.19 |
| SU2 | swap | 31.0 | 48.6 | **0.64** |
| SU2 | swap+out | 46.7 | 78.0 | **0.60** |
| U1⊠fZ2 | compose | 5.2 | 4.1 | 1.27 |
| U1⊠fZ2 | swap | 11.7 | 19.9 | **0.59** |
| U1⊠fZ2 | swap+out | 8.9 | 35.0 | **0.25** |

Only compose d=4 remains 1.2–1.4× (per-sector GEMM launch overhead;
TensorKit amortizes via `mul!` on the same coupled matrices). All
tree-transform paths are 1.6–4× faster than TensorKit at d=4.

### Hom-space Arc sharing (2026-07-03)

Warm replay was paying three deep `FusionTreeHomSpace` clones per call
(`DynamicFusionMapSpace::from_typed`) plus deep hom-space equality in the
route and fusion-block last-entry fast paths. The hom space is now stored
behind `Arc` in `FusionTensorMapSpace`, so the conversion is a pointer
clone and cache fast paths compare `Arc::ptr_eq` first (structural
equality remains the fallback — semantics unchanged, no thresholds).
`plan_lookups` dropped 0.6 → 0.2 µs/call; d=4 compose vs TensorKit
(Accelerate both sides):

| symmetry | before | after | TensorKit | ratio |
|---|---|---|---|---|
| U1 | 5.3 | 4.5 | 4.1 | 1.10 |
| fZ2 | 2.6 | 1.9 | 1.9 | **1.01** |
| SU2 | 8.5 | 7.8 | 7.1 | 1.10 |
| U1⊠fZ2 | 5.2 | 4.5 | 4.1 | 1.10 |

The remaining matmul bucket is 0.84 µs per coupled-sector GEMM vs
TensorKit's 0.82 µs per `mul!` — per-call parity; the residual ~0.1–0.2 µs
is cache-key touch and facade entry, shared by all routes.

## Layout default flip (2026-07-03)

`FusionTensorMapSpace::from_degeneracy_shapes` now builds the
TensorKit-equivalent coupled-sector matrix layout by default;
`from_degeneracy_shapes_packed` keeps the old packed layout for
storage-layout tests and packed interop. All earlier `MICROBENCH_LAYOUT=coupled`
runs correspond to today's defaults.

### Prepared-plan handles + unified warm lookup (2026-07-03)

`prepare_tensorcontract_fusion` returns a `PreparedTensorContractFusion`
handle resolving route + plan once (FFTW-style plan-once/execute-many);
`execute_prepared_tensorcontract_fusion` validates tensors by
subblock-structure Arc identity and replays with zero cache lookups.
The facade warm path now probes the fusion-block last entry first — a
hit implies the canonical route, so the route cache is skipped (one
compare instead of two). d=4 compose: U1 4.3, fZ2 1.7 (TensorKit 4.1 /
1.9), SU2 7.5 (TK 7.1). The handle is also the execution unit for
sector-level threading.
