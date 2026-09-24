# CUDA coefficient-1 moves as native strided copies (issue #1410, leaf M4)

Observation record for M4: an unscaled, unconjugated, overwrite region move
(`cuda_region_axpby` with `CudaRegionCoefficient::One`, `alpha == 1`,
`Overwrite`) is now one Tenferro typed `TensorViewCanonicalization::copy_into`
(native CubeCL `strided_to_strided_kernel`, a plain element assignment) instead
of a `dot_general` against a 1x1 coefficient of 1. The tree-transform executor
sends unit-coefficient Single blocks, packs and unscaled scatters there.
Timings are observations, never a CI gate and never a dispatch input.

## Environment

| Item | Value |
|---|---|
| Host | qg1, NVIDIA A100-SXM4-40GB, driver 560.35.05, `CUDA_VISIBLE_DEVICES=0` (no other process) |
| Toolkit | CUDA 12.6, cuTENSOR 2.5.0 via `TENFERRO_CUTENSOR_PATH` |
| Toolchain | rustc 1.96.0, release, `--no-default-features --features cuda,cpu-faer` |
| Threads | `RAYON/OPENBLAS/OMP/MKL_NUM_THREADS=1` |
| Base | `origin/main` `aed4a76e`, target `1410-bench/target-base` |
| Branch | `device-strided-copy`, target `1410-bench/target-branch` (separate target dirs) |
| Tenferro | 0.7.1 (tenferro-gpu, -tensor, -linalg), same `Cargo.lock` for both |

Binary sha256: base `cuda_operation_matrix` `a1211dba…312d3e`, branch
`cuda_operation_matrix` `4962aef0…74f151`, route probe `m4_move_bench`
`fb1098d2…62861a`, cold probe `m4_cold_probe` `4554d687…3c36`. The two probes
were temporary examples built from the branch and are not committed.

## 1. Transform rows of `cuda_operation_matrix` (base vs branch)

Default fixtures (`many-small` 64 blocks of 4x4, `few-large` 4 blocks of
64x64), 3 warm-up + 20 measured calls, interleaved `base, branch, branch, base`
on the same GPU. The matrix times submission (no device synchronization is
reachable through the public API), so these rows show host-side cost. Ratio is
branch/base of the median of the two runs' per-iteration medians.

| Rows | gemm/copy per call, base -> branch | warm ratio |
|---|---|---|
| U1, all 5 transforms, f64/c64, both families | 64/0 -> 0/64, 4/0 -> 0/4 | 0.21-0.64 |
| fZ2 / U1xfZ2 transpose, transpose_axes, repartition | all copy | 0.21-0.40 |
| fZ2 / U1xfZ2 permute, braid (sign blocks stay GEMM) | 2/0 -> 1/1, 64/0 -> 32/32 | 0.77-0.94 |
| SU2 permute/braid/transpose(_axes) | 64/0 -> 34/30, 4/0 -> 2/2 | 0.76-0.84 |
| SU2 repartition | 64/0 -> 63/1 | 0.93-0.97 |
| fZ2 c64 few-large braid | 2/0 -> 1/1 | 1.06 (133k vs 143k ns) |

Warm transfers are unchanged (one H2D, the #740 output). Cold H2D rises by one
(the context `1`, created with the structure) where a structure still has a
GEMM-route move, and is unchanged for all-unit structures, which no longer
upload a coefficient payload. Every row's check is `ok:host_value_equality`.
Non-transform rows keep identical submission counts; their time ratios are in
0.73-1.15 (noise).

**Cold outlier.** `U1 many-small permute cold` is 21.5x slower (base 3.6-3.8
ms, branch 73-87 ms, both dtypes): it is the first transform row of each dtype
in the process and pays the NVRTC compile of the copy kernel. See section 3.

## 2. Route probe: completion time per move (same binary, same GPU)

`m4_move_bench`: `k` moves of one layout per repetition, then a one-element
download on the same stream as the fence; per-move time = elapsed / `k`,
median of 21 repetitions, routes interleaved per repetition. Old route: the
GEMM with a 1x1 buffer coefficient holding 1 (the pre-M4 path); new: the copy.
Two runs; the second agrees within a few percent.

| Layout (dims, dst order) | k | GEMM ns/move | copy ns/move | copy/GEMM |
|---|---|---|---|---|
| [4,4] transpose | 64 | 44875 | 12786 | 0.28 |
| [4,4,4] (2,0,1) | 64 | 44481 | 12964 | 0.29 |
| [64,64] transpose | 16 | 52974 | 17604 | 0.33 |
| [64,64] identity | 16 | 54213 | 18046 | 0.33 |
| [512,512] transpose | 4 | 63782 | 32100 | 0.50 |
| [32,32,32] (2,0,1) | 4 | 63579 | 29197 | 0.46 |
| [16,16,16,16] reverse | 4 | 64617 | 29012 | 0.45 |
| [64,64,64] (2,0,1) | 2 | 74590 | 42515 | 0.57 |
| **[2048,2048] transpose f64** | 2 | 100024 | 234884 | **2.35** |
| **[2048,2048] transpose c64** | 2 | 154114 | 316123 | **2.05** |

(f64 shown; c64 is within 10% of it except where listed.) Per-move launch cost
drops by about 3x. For a 32 MiB (f64) / 64 MiB (c64) transposed block, the
native kernel is 2.0-2.4x slower than cuTENSOR's permute: it is a scalar
element loop without tiling, so one side of a transpose is uncoalesced.

## 3. Cold probe: first call per distinct layout

`m4_cold_probe`, fresh process per route, f64, transposes `[r, c]` for
`r in 2..12, c in {5, 7}` after one warm-up layout; first and second call each
followed by a one-element download.

| Route | first call, new layout | second call |
|---|---|---|
| GEMM (cuTENSOR plan) | 0.24-0.29 ms | 0.064-0.075 ms |
| copy, strided source (offset 1) | **38.7-41.0 ms, every layout** | 0.055-0.10 ms |
| copy, compact source at offset 0 | 35-44 ms per new destination-stride tuple, then 0.05 ms | 0.05 ms |

Cause, at source level: tenferro-gpu 0.7.1 `kernels/structural.rs`
`strided_to_strided_kernel` takes `dims`, both stride sequences and `len` as
`#[comptime]`, and `contiguous_to_view_kernel` takes the destination strides as
`#[comptime]`, so CubeCL compiles one NVRTC module per distinct fused layout
(identical in 0.6.0). The compiled modules stay in CubeCL's in-process cache,
which therefore grows with the number of distinct block layouts a process
moves. A transform's blocks sit at non-zero offsets, so it takes the first
row: roughly 40 ms once per new block layout per process, where the GEMM route
paid about 0.26 ms.

## Summary

- Correctness: coefficient-1 moves are bit-exact on device for every payload
  (C64 infinities, negative-zero components, NaN payloads, f32/f64 signalling
  NaNs), matching the host's coefficient-1 copy (#1406).
- Warm, small and medium blocks: fewer submissions of a cheaper kind, 0.2-0.9x
  host-side time and 0.3-0.6x completion time.
- Regressions: large transposed blocks complete 2.0-2.4x slower, and each new
  block layout costs a ~40 ms NVRTC compile on first use, with an in-process
  kernel cache that grows with the layout count. Both are properties of the
  upstream native kernel.
