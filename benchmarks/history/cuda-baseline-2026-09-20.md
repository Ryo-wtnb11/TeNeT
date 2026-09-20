# CUDA baseline, 2026-09-20

Revision-pinned device performance baseline of the existing CUDA path
(issue #1273, roadmap #1139). Produced by `benchmarks/cuda_operation_matrix.sh`
on qg1; protocol and column definitions are in
`benchmarks/cuda_operation_matrix.md`.

This is a measurement record. It is not a CI gate, it is not a speed claim, and
nothing here is a threshold that any code dispatches on. Rows where Host is
faster than the device are recorded plainly as non-improvements.

Raw data: `cuda-baseline-2026-09-20.csv` (926 rows, header lines included).
Build log: `cuda-baseline-2026-09-20.build.log`.

## Environment

```
host_os=Linux host_release=6.8.0-49-generic host_arch=x86_64
cpu=AMD EPYC 7742 64-Core Processor
gpu=0, NVIDIA A100-SXM4-40GB, 560.35.05, 40960 MiB
gpu=1, NVIDIA A100-SXM4-40GB, 560.35.05, 40960 MiB
gpu=2, NVIDIA A100-SXM4-40GB, 560.35.05, 40960 MiB
gpu=3, NVIDIA A100-SXM4-40GB, 560.35.05, 40960 MiB
gpu=4, NVIDIA A100-SXM4-40GB, 560.35.05, 40960 MiB
gpu=5, NVIDIA A100-SXM4-40GB, 560.35.05, 40960 MiB
gpu=6, NVIDIA A100-SXM4-40GB, 560.35.05, 40960 MiB
gpu=7, NVIDIA A100-SXM4-40GB, 560.35.05, 40960 MiB
nvcc=nvcc: NVIDIA (R) Cuda compiler driver Copyright (c) 2005-2024 NVIDIA Corporation Built on Tue_Oct_29_23:50:19_PDT_2024 Cuda compilation tools, release 12.6, V12.6.85 Build cuda_12.6.r12.6/compiler.35059454_0
rustc=rustc 1.96.0 (ac68faa20 2026-05-25) cargo=cargo 1.96.0 (30a34c682 2026-05-25)
tenet_sha=709f9e0c3ddf685cc21cc08693bbc8c0597bc46e+gb-cuda-operation-matrix-worktree dirty=false lock=Cargo.lock lock_sha256=cad23363c997e3895a5391be45ee97fb3cfe19ca049c53593bbc8f60580f59b2
cargo_package=tenet-network cargo_features=cuda,cpu-faer profile=release locked=true offline=true
cuda_paths TENFERRO_CUTENSOR_PATH=/home/ryo-w/.julia/artifacts/6e2b77b3f4385f09b93d41018b88ef1445b5c464/lib/libcutensor.so CUDA_PATH=/usr/local/cuda-12.6 CUDA_VISIBLE_DEVICES=0
cutensor_resolved=/home/ryo-w/.julia/artifacts/6e2b77b3f4385f09b93d41018b88ef1445b5c464/lib/libcutensor.so.2.5.0 cutensor_version=2.5.0
threads=RAYON_NUM_THREADS:1 OPENBLAS_NUM_THREADS:1 OMP_NUM_THREADS:1 MKL_NUM_THREADS:1
tenet_authority=709f9e0c3ddf685cc21cc08693bbc8c0597bc46e+gb-cuda-operation-matrix-worktree dirty=false
tenferro_authority=cudarc@0.19.9 checksum=804764d10e844da09765a7b2ca9641a0851523d1702efb0d7299d73e31b86e80; t4a-cubecl-cuda@0.10.0 checksum=cae9b959e07a1ea70ee5831e9097db5f9057cfd87bbde2db71ec19e6cfe6d958; tenferro-gpu@0.5.0 checksum=c87dd7e4bca837bd78775bf17f51a2e764c51a31ff0f468a1125037fc2d89870; tenferro-linalg@0.3.0 checksum=2b441a4dbe6ea1a2c9867fe6721f4f235fe327c6fdab0c61d3bb74f1fbd5dcee; tenferro-linalg@0.5.0 checksum=c62ee55bb010b64f9407aae83f66443e90f12eb9302bd2149eda8e0f73aaac60; tenferro-tensor@0.3.0 checksum=6a1e201682893cd49c471d2f27039464317ae12023db0dcd42d937ef236dbac4; tenferro-tensor@0.5.0 checksum=40eab4bf1e0db49db694cf8036878defd5eb973eaeb5a599ade7941e0b8921dd
cuda_env TENFERRO_CUTENSOR_PATH=/home/ryo-w/.julia/artifacts/6e2b77b3f4385f09b93d41018b88ef1445b5c464/lib/libcutensor.so CUDA_VISIBLE_DEVICES=0 device_ordinal=0
fixtures families=["many-small", "few-large"] blocks=[64, 4] degeneracy=[4, 64] rank=2 shape=endomorphism[V;V]
protocol fresh_Runtime_per_row fixture_before_timer first_call warmup=3 warm_iterations=20 statistic=per_iteration_median(+ns_min) correctness_checked_after_both_timed_phases
cold_scope=cold is the first call of the measured operation on a fresh Runtime; the fixture's own upload has already run, so process-global CUDA state and interned space structures may be warm. Rows whose measured operation is itself performed during fixture setup report first_after_setup instead of cold
sampling=one process per invocation; unlike the Host operation_matrix wrapper there is no three-process median, so a row carries this process's per-iteration median only
allocation_scope=caller-thread Rust allocation calls and requested bytes during the timed region; excludes worker threads, frees, and device memory
device_counter_scope=tenet_dense::cuda_transfer_stats deltas over the timed region; device_allocs counts uploads plus tenferro tensors wrapped as CudaDenseStorage and excludes tenferro-internal solver workspaces
peak_device_memory=NA: neither TeNeT nor tenferro exposes a device allocation high-water mark, and a per-row nvidia-smi sample would observe the whole device rather than this process's buffers
synchronization=NA: no device synchronization is reachable through TeNeT's public API, and tenferro's with_cubecl success path does not synchronize, so a timed region measures submission unless the operation itself downloads. The completion barrier after each timed call is norm() on one fixture input, which submits one GEMM per coupled block and downloads the per-block partials; only its downloads are recorded, in the barrier_* columns, and none of its traffic is included in the phase columns
tensorkit_cuda_comparison=absent: neither TensorKit nor QSpace has a matched CUDA fixture for these rows at this revision
```

All 8 A100s were visible to `nvidia-smi`; the run used ordinal 0
(`CUDA_VISIBLE_DEVICES=0`). Every row's correctness check passed
(`0` mismatches, `0` unexpected skips across all 926 rows).

Two notes on the recorded authority:

- The measurement host holds a source copy without git metadata, so the
  revision was supplied through `TENET_SHA`. It reads
  `709f9e0c... + gb-cuda-operation-matrix-worktree`: TeNeT `origin/main` at
  `709f9e0c3ddf685cc21cc08693bbc8c0597bc46e` plus this branch's diff, which is
  the observation counters, this harness, and this record. No other production
  code differs.
- `TENFERRO_CUTENSOR_PATH` must name the shared object
  (`.../lib/libcutensor.so`), not the directory; pointing it at the directory,
  as `reviews/gpu-phase-20260920/g0-plan.md` does, makes every cuTENSOR-backed
  operation fail with `cannot read file data: Is a directory`. The directory
  still belongs on `LD_LIBRARY_PATH`.

## Measurement limits

- **Single process.** Unlike the Host `operation_matrix.sh` wrapper, which
  takes a three-process median, this is one process per invocation. A timing
  here is this process's per-iteration median, with no between-process
  variance estimate.
- **Peak device memory: absent.** Neither TeNeT nor tenferro exposes a device
  allocation high-water mark, and an `nvidia-smi --query-gpu=memory.used`
  sample would observe the whole device rather than this process's buffers.
  `device_allocs` counts buffers, not bytes retained.
- **No synchronization is reachable** through TeNeT's public API, and
  tenferro's `with_cubecl` success path does not synchronize, so a timed region
  measures submission unless the operation itself downloads. The completion
  barrier after each timed call is `norm()` on one fixture input: it submits
  one GEMM per coupled block and downloads the per-block partials. Only its
  downloads are recorded (`barrier_d2h_*`); none of its traffic enters the
  phase columns, and its GEMMs are not recorded at all.

## Fixtures

Rank-2 endomorphisms `[V; V]`. `many-small` is 64 sectors of degeneracy 4
(64 coupled blocks of `4 x 4`); `few-large` is 4 sectors of degeneracy 64
(4 blocks of `64 x 64`). Fermion parity has two irreps, so `fZ2` realizes
2 blocks in both families; that is a provider capability, not a harness
decision.

## The first call on a fresh Runtime costs ~0.2 s

Every row that submits backend work pays **185-657 ms on its first call**,
median 200 ms, against warm medians of 58 us to 62 ms. The two pure transfer
rows do not: their first call is 3.8-157 us.

The cost is independent of provider, dtype, fixture family, block count and
operation, and the fixture's own upload has already run on that Runtime when
it is paid. That localizes it to the first cuTENSOR/cuSOLVER-backed submission
on a freshly built `CudaDenseContext`, not to CUDA context creation and not to
any TeNeT-side cache. `eigh_full` and `qr_compact` pay more than the flat
~200 ms (up to 657 ms), consistent with cuSOLVER initializing on top of it.

This is the single largest number in the table and it was invisible in the
first draft of this record, which probed each operation for its correctness
oracle before measuring, so the reported "cold" was really a second call. The
harness now returns the first call's output and validates after both timed
phases.

## Structural interpretation, one line per operation

Per-call counts are the warm per-iteration medians. `B` is the coupled block
count (64 / 4 / 2), `d` the degeneracy, `w = size_of::<D>()` (8 or 16).

- `to_cuda`: 1 H2D of `L * w` and 1 device allocation; nothing else crosses.
  `to_host`: 1 D2H of the same size, no device allocation.
- `contract_direct`: 1 H2D of `L_dst * w` **zeros** plus 1 device allocation
  per call, then `B` GEMM submissions. The upload is pure overhead: the output
  buffer is fully written by the GEMMs. This is the cost G3 targets.
- `contract_lazy_adjoint_lhs`: identical counters to `contract_direct`. The
  lazy adjoint costs no extra transfer, allocation, or GEMM; the operand flag
  is carried into `dot_general`.
- `compose`: identical counters to `contract_direct` for these endomorphisms.
- `scale`: 2 H2D (`8 B` coefficient scalar + the full output zero upload),
  2 device allocations, 1 GEMM.
- `add_owned` and `add_lazy_fold`: 2 H2D (`16 B` of coefficients + the full
  output zero upload), 2 device allocations, 2 GEMMs. The lazy fold adds no
  transfer over the owned form.
- `norm` and `inner`: 1 H2D of `B * 8 B` quantum-dimension weights and 1 D2H of
  `B * 8 B` partials, 1 device allocation, `B` GEMMs. The only rows whose timed
  region already contains a completing download.
- `svd_compact`: 3 bulk H2D (the three factor outputs), `B` D2H of `d * 8 B`
  each (one spectrum per coupled block), `B` cuSOLVER calls, `2B` region
  copies, `2B + 3` device allocations.
- `svd_trunc_rank`: `B` spectrum D2H and `B` cuSOLVER calls as above, plus the
  selector uploads and GEMMs of the non-aligned factor assembly; the bulk
  output uploads shrink because the truncated factors are assembled by copy.
- `eigh_full`: the most transfer-heavy row, and the **reorder selectors**, not
  the Hermiticity check, are what drives the uploads. Per call, exactly:

  | class | calls | bytes |
  |---|---|---|
  | eigenvalue and eigenvector output buffers | 2 | `2 * B * d^2 * w` |
  | `upload_selector` eigenvector-reorder selectors (`typed.rs` `eigh_trunc`) | `B` | `B * d^2 * w` |
  | Hermiticity pre-check input-scale scalar | `B` | `B * 8` |
  | **H2D total** | `2B + 2` | |
  | Hermiticity pre-check scalar reductions (input max, input sum-of-squares, residual max) | `3B` | `3B * 8` |
  | eigenvalue spectra | `B` | `B * d * 8` |
  | **D2H total** | `4B` | |

  The `B` GEMMs are the selector applications in `assemble_left_factor`, one
  per kept route. The check contributes one upload and three downloads per
  block rather than the four-download worst case because these fixtures are
  exactly Hermitian, so `cuda_is_hermitian_region` returns at
  `residual_scale == 0.0` before the second normalization. The formula
  reproduces every one of the 16 provider x dtype x family rows exactly; for
  `U1`/`f64`/`many-small` that is 130 H2D / 25088 B and 256 D2H / 3584 B, of
  which the selectors are 64 calls / 8192 B and the check only 64 calls /
  512 B of the uploads.
- `qr_compact`: 2 bulk H2D, **no** D2H (the positive-diagonal gauge is applied
  on device), `B` cuSOLVER calls, `2B` region copies, `2B + 2` device
  allocations.
- `network_chain3`: exactly `N - 1 = 2` H2D zero uploads and 2 device
  allocations for a 3-tensor chain, and `2B` GEMMs. Each contraction step pays
  the returning-path upload of `contract_direct`.

Dtype: every call **count** is identical for `f64` and `c64`; only the byte
counts double, exactly `size_of::<Complex64>() / size_of::<f64>()`, except the
`8 B` scalar metadata, which stays `f64` in both.

Provider: the call counts depend only on the coupled block count, not on the
categorical structure. `SU2` and `U1xfZ2` produce the same per-call transfer,
allocation, GEMM and solver counts as `U1` at equal block count.

## Non-improvements

At every fixture size measured here, Host is faster than the device on every
row that has a Host counterpart. Examples (`U1`, `f64`, warm medians):

| row | family | cuda | host |
|---|---|---|---|
| `contract_direct` | many-small | 1761 us | 34 us |
| `contract_direct` | few-large | 227 us | 60 us |
| `norm` | many-small | 1765 us | 0.9 us |
| `svd_compact` | few-large | 5289 us | 1312 us |
| `eigh_full` | many-small | 62580 us | 904 us |
| `network_chain3` | few-large | 461 us | 122 us |

These fixtures are far below the size where an A100 pays for its launch and
transfer overhead; the numbers are recorded as the current per-call structural
cost of the device path, not as a verdict on the device path's ceiling. The
`many-small` blow-up (64 blocks of `4 x 4`) is the per-block submission cost,
which is what the block/group batching work in the roadmap addresses. Adding
the ~0.2 s first-call cost, a device Runtime built for a single small operation
is never competitive at this revision.

## Absent references

Neither TensorKit nor QSpace has a matched CUDA fixture for these rows at this
revision, so no cross-library device comparison is recorded.

## Results

### Per-call device cost, `many-small`

`first` is the first call of the measured operation on a fresh Runtime; for the two
transfer rows the operation also runs during fixture setup, so their first call is
labelled `first_after_setup` in the CSV. All counter columns are warm per-iteration
medians.

| provider | dtype | operation | cuda first us | cuda warm us | host warm us | H2D calls/bytes | D2H calls/bytes | dev allocs | GEMM | solver | copy | caller allocs/bytes |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| U1 | f64 | `to_cuda` | 4.4 | 12.5 | NA | 1 / 8192 | 0 / 0 | 1 | 0 | 0 | 0 | 22 / 35448 |
| U1 | f64 | `to_host` | 27.4 | 27.6 | NA | 0 / 0 | 1 / 8192 | 0 | 0 | 0 | 0 | 9 / 16904 |
| U1 | f64 | `contract_direct` | 184993.4 | 1761.2 | 34.0 | 1 / 8192 | 0 / 0 | 1 | 64 | 0 | 0 | 3057 / 134968 |
| U1 | f64 | `contract_lazy_adjoint_lhs` | 200243.7 | 1775.8 | 38.0 | 1 / 8192 | 0 / 0 | 1 | 64 | 0 | 0 | 3185 / 138040 |
| U1 | f64 | `compose` | 201575.5 | 1770.0 | 34.1 | 1 / 8192 | 0 / 0 | 1 | 64 | 0 | 0 | 3059 / 134984 |
| U1 | f64 | `scale` | 209538.2 | 57.8 | 0.3 | 2 / 8200 | 0 / 0 | 2 | 1 | 0 | 0 | 91 / 46128 |
| U1 | f64 | `add_owned` | 195021.8 | 89.9 | 0.4 | 2 / 8208 | 0 / 0 | 2 | 2 | 0 | 0 | 138 / 47440 |
| U1 | f64 | `add_lazy_fold` | 190662.7 | 111.6 | 50.9 | 2 / 8208 | 0 / 0 | 2 | 2 | 0 | 0 | 145 / 49104 |
| U1 | f64 | `norm` | 206371.1 | 1765.1 | 0.9 | 1 / 512 | 1 / 512 | 1 | 64 | 0 | 0 | 3164 / 90048 |
| U1 | f64 | `inner` | 207353.1 | 1774.0 | 0.9 | 1 / 512 | 1 / 512 | 1 | 64 | 0 | 0 | 3164 / 90048 |
| U1 | f64 | `svd_compact` | 307810.8 | 61939.3 | 670.4 | 3 / 24576 | 64 / 2048 | 131 | 0 | 64 | 128 | 14882 / 1513224 |
| U1 | f64 | `svd_trunc_rank` | 250346.9 | 56238.3 | 663.7 | 11 / 544 | 64 / 2048 | 139 | 8 | 64 | 0 | 11432 / 1487764 |
| U1 | f64 | `eigh_full` | 657382.9 | 62579.7 | 904.0 | 130 / 25088 | 256 / 3584 | 194 | 64 | 64 | 0 | 53290 / 6915712 |
| U1 | f64 | `qr_compact` | 411710.5 | 39796.6 | 717.1 | 2 / 16384 | 0 / 0 | 130 | 0 | 64 | 128 | 21315 / 2607056 |
| U1 | f64 | `network_chain3` | 200156.4 | 3538.9 | 84.1 | 2 / 16384 | 0 / 0 | 2 | 128 | 0 | 0 | 6181 / 294584 |
| U1 | c64 | `to_cuda` | 6.4 | 9.1 | NA | 1 / 16384 | 0 / 0 | 1 | 0 | 0 | 0 | 22 / 68216 |
| U1 | c64 | `to_host` | 27.3 | 27.1 | NA | 0 / 0 | 1 / 16384 | 0 | 0 | 0 | 0 | 9 / 33288 |
| U1 | c64 | `contract_direct` | 192318.2 | 1776.3 | 36.9 | 1 / 16384 | 0 / 0 | 1 | 64 | 0 | 0 | 3057 / 175928 |
| U1 | c64 | `contract_lazy_adjoint_lhs` | 200535.5 | 1792.7 | 41.3 | 1 / 16384 | 0 / 0 | 1 | 64 | 0 | 0 | 3185 / 179000 |
| U1 | c64 | `compose` | 193538.0 | 1785.9 | 36.8 | 1 / 16384 | 0 / 0 | 1 | 64 | 0 | 0 | 3059 / 175944 |
| U1 | c64 | `scale` | 223938.7 | 62.8 | 0.9 | 2 / 16400 | 0 / 0 | 2 | 1 | 0 | 0 | 91 / 87120 |
| U1 | c64 | `add_owned` | 191936.8 | 93.3 | 1.5 | 2 / 16416 | 0 / 0 | 2 | 2 | 0 | 0 | 138 / 88480 |
| U1 | c64 | `add_lazy_fold` | 193898.1 | 118.5 | 53.6 | 2 / 16416 | 0 / 0 | 2 | 2 | 0 | 0 | 145 / 90144 |
| U1 | c64 | `norm` | 198358.9 | 1765.5 | 5.9 | 1 / 1024 | 1 / 1024 | 1 | 64 | 0 | 0 | 3164 / 93632 |
| U1 | c64 | `inner` | 193969.7 | 1771.9 | 3.4 | 1 / 1024 | 1 / 1024 | 1 | 64 | 0 | 0 | 3164 / 93632 |
| U1 | c64 | `svd_compact` | 338112.9 | 69070.3 | 790.7 | 3 / 49152 | 64 / 2048 | 131 | 0 | 64 | 128 | 14882 / 1636296 |
| U1 | c64 | `svd_trunc_rank` | 274098.3 | 73693.1 | 783.5 | 11 / 1088 | 64 / 2048 | 139 | 8 | 64 | 0 | 11432 / 1490676 |
| U1 | c64 | `eigh_full` | 505325.0 | 63456.9 | 630.8 | 130 / 49664 | 256 / 3584 | 194 | 64 | 64 | 0 | 56554 / 7826432 |
| U1 | c64 | `network_chain3` | 221398.4 | 3578.5 | 88.6 | 2 / 32768 | 0 / 0 | 2 | 128 | 0 | 0 | 6181 / 376504 |
| fZ2 | f64 | `to_cuda` | 3.8 | 5.0 | NA | 1 / 256 | 0 / 0 | 1 | 0 | 0 | 0 | 22 / 3704 |
| fZ2 | f64 | `to_host` | 25.1 | 23.2 | NA | 0 / 0 | 1 / 256 | 0 | 0 | 0 | 0 | 9 / 1032 |
| fZ2 | f64 | `contract_direct` | 189430.6 | 77.9 | 6.1 | 1 / 256 | 0 / 0 | 1 | 2 | 0 | 0 | 141 / 8072 |
| fZ2 | f64 | `contract_lazy_adjoint_lhs` | 189335.1 | 78.5 | 6.1 | 1 / 256 | 0 / 0 | 1 | 2 | 0 | 0 | 145 / 8168 |
| fZ2 | f64 | `compose` | 189370.9 | 77.6 | 6.0 | 1 / 256 | 0 / 0 | 1 | 2 | 0 | 0 | 143 / 8088 |
| fZ2 | f64 | `scale` | 189229.9 | 49.0 | 0.1 | 2 / 264 | 0 / 0 | 2 | 1 | 0 | 0 | 91 / 6448 |
| fZ2 | f64 | `add_owned` | 194061.8 | 78.1 | 0.1 | 2 / 272 | 0 / 0 | 2 | 2 | 0 | 0 | 138 / 7760 |
| fZ2 | f64 | `add_lazy_fold` | 191741.4 | 81.0 | 1.4 | 2 / 272 | 0 / 0 | 2 | 2 | 0 | 0 | 143 / 8400 |
| fZ2 | f64 | `norm` | 193435.9 | 86.5 | 0.1 | 1 / 16 | 1 / 16 | 1 | 2 | 0 | 0 | 127 / 4168 |
| fZ2 | f64 | `inner` | 193258.5 | 86.3 | 0.1 | 1 / 16 | 1 / 16 | 1 | 2 | 0 | 0 | 127 / 4168 |
| fZ2 | f64 | `svd_compact` | 210218.2 | 1784.9 | 23.5 | 3 / 768 | 2 / 64 | 7 | 0 | 2 | 4 | 549 / 56536 |
| fZ2 | f64 | `svd_trunc_rank` | 206847.6 | 1866.9 | 26.9 | 7 / 576 | 2 / 64 | 11 | 4 | 2 | 0 | 725 / 69324 |
| fZ2 | f64 | `eigh_full` | 214085.1 | 1325.3 | 19.3 | 6 / 784 | 8 / 112 | 8 | 2 | 2 | 0 | 1746 / 223806 |
| fZ2 | f64 | `qr_compact` | 228449.1 | 814.9 | 15.6 | 2 / 512 | 0 / 0 | 6 | 0 | 2 | 4 | 722 / 87658 |
| fZ2 | f64 | `network_chain3` | 188999.5 | 160.6 | 14.7 | 2 / 512 | 0 / 0 | 2 | 4 | 0 | 0 | 314 / 18472 |
| fZ2 | c64 | `to_cuda` | 5.2 | 4.6 | NA | 1 / 512 | 0 / 0 | 1 | 0 | 0 | 0 | 22 / 4728 |
| fZ2 | c64 | `to_host` | 26.6 | 24.8 | NA | 0 / 0 | 1 / 512 | 0 | 0 | 0 | 0 | 9 / 1544 |
| fZ2 | c64 | `contract_direct` | 191105.9 | 76.5 | 6.1 | 1 / 512 | 0 / 0 | 1 | 2 | 0 | 0 | 141 / 9352 |
| fZ2 | c64 | `contract_lazy_adjoint_lhs` | 192774.4 | 76.8 | 6.3 | 1 / 512 | 0 / 0 | 1 | 2 | 0 | 0 | 145 / 9448 |
| fZ2 | c64 | `compose` | 194365.7 | 77.8 | 6.2 | 1 / 512 | 0 / 0 | 1 | 2 | 0 | 0 | 143 / 9368 |
| fZ2 | c64 | `scale` | 195234.0 | 48.0 | 0.1 | 2 / 528 | 0 / 0 | 2 | 1 | 0 | 0 | 91 / 7760 |
| fZ2 | c64 | `add_owned` | 192761.2 | 76.8 | 0.2 | 2 / 544 | 0 / 0 | 2 | 2 | 0 | 0 | 138 / 9120 |
| fZ2 | c64 | `add_lazy_fold` | 193071.6 | 80.3 | 1.5 | 2 / 544 | 0 / 0 | 2 | 2 | 0 | 0 | 143 / 9760 |
| fZ2 | c64 | `norm` | 192946.5 | 98.3 | 0.2 | 1 / 32 | 1 / 32 | 1 | 2 | 0 | 0 | 126 / 4352 |
| fZ2 | c64 | `inner` | 197431.3 | 99.2 | 0.2 | 1 / 32 | 1 / 32 | 1 | 2 | 0 | 0 | 126 / 4352 |
| fZ2 | c64 | `svd_compact` | 208698.8 | 2603.1 | 26.9 | 3 / 1536 | 2 / 64 | 7 | 0 | 2 | 4 | 549 / 60382 |
| fZ2 | c64 | `svd_trunc_rank` | 208091.4 | 2684.8 | 31.3 | 7 / 1152 | 2 / 64 | 11 | 4 | 2 | 0 | 725 / 72210 |
| fZ2 | c64 | `eigh_full` | 216377.7 | 1404.3 | 22.5 | 6 / 1552 | 8 / 112 | 8 | 2 | 2 | 0 | 1848 / 252266 |
| fZ2 | c64 | `network_chain3` | 192217.2 | 161.4 | 15.1 | 2 / 1024 | 0 / 0 | 2 | 4 | 0 | 0 | 314 / 21032 |
| SU2 | f64 | `to_cuda` | 6.4 | 8.3 | NA | 1 / 8192 | 0 / 0 | 1 | 0 | 0 | 0 | 22 / 35448 |
| SU2 | f64 | `to_host` | 30.3 | 29.4 | NA | 0 / 0 | 1 / 8192 | 0 | 0 | 0 | 0 | 9 / 16904 |
| SU2 | f64 | `contract_direct` | 194661.9 | 1778.1 | 34.3 | 1 / 8192 | 0 / 0 | 1 | 64 | 0 | 0 | 3057 / 134968 |
| SU2 | f64 | `contract_lazy_adjoint_lhs` | 198804.8 | 1789.3 | 38.6 | 1 / 8192 | 0 / 0 | 1 | 64 | 0 | 0 | 3185 / 138040 |
| SU2 | f64 | `compose` | 196273.8 | 1783.1 | 34.3 | 1 / 8192 | 0 / 0 | 1 | 64 | 0 | 0 | 3059 / 134984 |
| SU2 | f64 | `scale` | 196753.4 | 57.7 | 0.3 | 2 / 8200 | 0 / 0 | 2 | 1 | 0 | 0 | 91 / 46128 |
| SU2 | f64 | `add_owned` | 194900.1 | 85.6 | 0.4 | 2 / 8208 | 0 / 0 | 2 | 2 | 0 | 0 | 138 / 47440 |
| SU2 | f64 | `add_lazy_fold` | 198364.2 | 111.8 | 50.8 | 2 / 8208 | 0 / 0 | 2 | 2 | 0 | 0 | 145 / 49104 |
| SU2 | f64 | `norm` | 198222.8 | 1763.0 | 0.7 | 1 / 512 | 1 / 512 | 1 | 64 | 0 | 0 | 3164 / 90048 |
| SU2 | f64 | `inner` | 196647.6 | 1747.5 | 0.8 | 1 / 512 | 1 / 512 | 1 | 64 | 0 | 0 | 3164 / 90048 |
| SU2 | f64 | `svd_compact` | 264370.8 | 61298.3 | 656.9 | 3 / 24576 | 64 / 2048 | 131 | 0 | 64 | 128 | 14882 / 1513224 |
| SU2 | f64 | `svd_trunc_rank` | 263688.7 | 58578.6 | 668.2 | 7 / 272 | 64 / 2048 | 135 | 4 | 64 | 0 | 11152 / 1476302 |
| SU2 | f64 | `eigh_full` | 258166.3 | 41633.0 | 525.9 | 130 / 25088 | 256 / 3584 | 194 | 64 | 64 | 0 | 53290 / 6915712 |
| SU2 | f64 | `qr_compact` | 251547.1 | 33015.2 | 740.8 | 2 / 16384 | 0 / 0 | 130 | 0 | 64 | 128 | 21315 / 2607056 |
| SU2 | f64 | `network_chain3` | 199942.0 | 3557.1 | 85.9 | 2 / 16384 | 0 / 0 | 2 | 128 | 0 | 0 | 6181 / 294584 |
| SU2 | c64 | `to_cuda` | 7.8 | 9.4 | NA | 1 / 16384 | 0 / 0 | 1 | 0 | 0 | 0 | 22 / 68216 |
| SU2 | c64 | `to_host` | 28.2 | 27.6 | NA | 0 / 0 | 1 / 16384 | 0 | 0 | 0 | 0 | 9 / 33288 |
| SU2 | c64 | `contract_direct` | 197217.5 | 1779.9 | 37.4 | 1 / 16384 | 0 / 0 | 1 | 64 | 0 | 0 | 3057 / 175928 |
| SU2 | c64 | `contract_lazy_adjoint_lhs` | 223080.6 | 1795.2 | 41.6 | 1 / 16384 | 0 / 0 | 1 | 64 | 0 | 0 | 3185 / 179000 |
| SU2 | c64 | `compose` | 196923.7 | 1781.7 | 37.0 | 1 / 16384 | 0 / 0 | 1 | 64 | 0 | 0 | 3059 / 175944 |
| SU2 | c64 | `scale` | 196321.6 | 62.9 | 0.9 | 2 / 16400 | 0 / 0 | 2 | 1 | 0 | 0 | 91 / 87120 |
| SU2 | c64 | `add_owned` | 194899.1 | 92.7 | 1.5 | 2 / 16416 | 0 / 0 | 2 | 2 | 0 | 0 | 138 / 88480 |
| SU2 | c64 | `add_lazy_fold` | 199339.3 | 119.3 | 53.8 | 2 / 16416 | 0 / 0 | 2 | 2 | 0 | 0 | 145 / 90144 |
| SU2 | c64 | `norm` | 196681.9 | 1773.1 | 2.8 | 1 / 1024 | 1 / 1024 | 1 | 64 | 0 | 0 | 3164 / 93632 |
| SU2 | c64 | `inner` | 198885.6 | 1754.9 | 2.8 | 1 / 1024 | 1 / 1024 | 1 | 64 | 0 | 0 | 3164 / 93632 |
| SU2 | c64 | `svd_compact` | 290061.9 | 68778.7 | 784.8 | 3 / 49152 | 64 / 2048 | 131 | 0 | 64 | 128 | 14882 / 1636296 |
| SU2 | c64 | `svd_trunc_rank` | 273102.7 | 65750.1 | 780.2 | 7 / 544 | 64 / 2048 | 135 | 4 | 64 | 0 | 11152 / 1477854 |
| SU2 | c64 | `eigh_full` | 256066.5 | 54506.8 | 631.9 | 130 / 49664 | 256 / 3584 | 194 | 64 | 64 | 0 | 56554 / 7826432 |
| SU2 | c64 | `network_chain3` | 196727.6 | 3567.8 | 90.6 | 2 / 32768 | 0 / 0 | 2 | 128 | 0 | 0 | 6181 / 376504 |
| U1xfZ2 | f64 | `to_cuda` | 7.1 | 8.2 | NA | 1 / 8192 | 0 / 0 | 1 | 0 | 0 | 0 | 22 / 35448 |
| U1xfZ2 | f64 | `to_host` | 30.8 | 27.6 | NA | 0 / 0 | 1 / 8192 | 0 | 0 | 0 | 0 | 9 / 16904 |
| U1xfZ2 | f64 | `contract_direct` | 194056.1 | 1774.9 | 41.4 | 1 / 8192 | 0 / 0 | 1 | 64 | 0 | 0 | 3057 / 134968 |
| U1xfZ2 | f64 | `contract_lazy_adjoint_lhs` | 197405.2 | 1785.1 | 45.5 | 1 / 8192 | 0 / 0 | 1 | 64 | 0 | 0 | 3185 / 138040 |
| U1xfZ2 | f64 | `compose` | 195939.5 | 1788.7 | 41.5 | 1 / 8192 | 0 / 0 | 1 | 64 | 0 | 0 | 3059 / 134984 |
| U1xfZ2 | f64 | `scale` | 195830.8 | 58.7 | 0.3 | 2 / 8200 | 0 / 0 | 2 | 1 | 0 | 0 | 91 / 46128 |
| U1xfZ2 | f64 | `add_owned` | 199373.7 | 86.3 | 0.4 | 2 / 8208 | 0 / 0 | 2 | 2 | 0 | 0 | 138 / 47440 |
| U1xfZ2 | f64 | `add_lazy_fold` | 197292.8 | 120.5 | 50.6 | 2 / 8208 | 0 / 0 | 2 | 2 | 0 | 0 | 145 / 49104 |
| U1xfZ2 | f64 | `norm` | 197836.8 | 1769.4 | 0.9 | 1 / 512 | 1 / 512 | 1 | 64 | 0 | 0 | 3164 / 90048 |
| U1xfZ2 | f64 | `inner` | 198570.5 | 1751.7 | 0.9 | 1 / 512 | 1 / 512 | 1 | 64 | 0 | 0 | 3164 / 90048 |
| U1xfZ2 | f64 | `svd_compact` | 267736.2 | 61642.0 | 680.7 | 3 / 24576 | 64 / 2048 | 131 | 0 | 64 | 128 | 14882 / 1513224 |
| U1xfZ2 | f64 | `svd_trunc_rank` | 263412.3 | 59138.6 | 695.5 | 11 / 544 | 64 / 2048 | 139 | 8 | 64 | 0 | 11432 / 1487764 |
| U1xfZ2 | f64 | `eigh_full` | 257734.5 | 52817.9 | 551.9 | 130 / 25088 | 256 / 3584 | 194 | 64 | 64 | 0 | 53290 / 6915712 |
| U1xfZ2 | f64 | `qr_compact` | 254429.9 | 32596.6 | 768.1 | 2 / 16384 | 0 / 0 | 130 | 0 | 64 | 128 | 21315 / 2607056 |
| U1xfZ2 | f64 | `network_chain3` | 199734.5 | 3587.2 | 99.8 | 2 / 16384 | 0 / 0 | 2 | 128 | 0 | 0 | 6181 / 294584 |
| U1xfZ2 | c64 | `to_cuda` | 12.2 | 9.3 | NA | 1 / 16384 | 0 / 0 | 1 | 0 | 0 | 0 | 22 / 68216 |
| U1xfZ2 | c64 | `to_host` | 27.8 | 27.5 | NA | 0 / 0 | 1 / 16384 | 0 | 0 | 0 | 0 | 9 / 33288 |
| U1xfZ2 | c64 | `contract_direct` | 197163.7 | 1771.5 | 43.5 | 1 / 16384 | 0 / 0 | 1 | 64 | 0 | 0 | 3057 / 175928 |
| U1xfZ2 | c64 | `contract_lazy_adjoint_lhs` | 200635.9 | 1789.9 | 48.8 | 1 / 16384 | 0 / 0 | 1 | 64 | 0 | 0 | 3185 / 179000 |
| U1xfZ2 | c64 | `compose` | 198797.6 | 1784.5 | 43.8 | 1 / 16384 | 0 / 0 | 1 | 64 | 0 | 0 | 3059 / 175944 |
| U1xfZ2 | c64 | `scale` | 196258.9 | 63.2 | 0.9 | 2 / 16400 | 0 / 0 | 2 | 1 | 0 | 0 | 91 / 87120 |
| U1xfZ2 | c64 | `add_owned` | 199294.7 | 94.0 | 1.5 | 2 / 16416 | 0 / 0 | 2 | 2 | 0 | 0 | 138 / 88480 |
| U1xfZ2 | c64 | `add_lazy_fold` | 197599.7 | 126.8 | 54.0 | 2 / 16416 | 0 / 0 | 2 | 2 | 0 | 0 | 145 / 90144 |
| U1xfZ2 | c64 | `norm` | 199018.8 | 1775.0 | 5.9 | 1 / 1024 | 1 / 1024 | 1 | 64 | 0 | 0 | 3164 / 93632 |
| U1xfZ2 | c64 | `inner` | 200865.2 | 1788.0 | 3.4 | 1 / 1024 | 1 / 1024 | 1 | 64 | 0 | 0 | 3164 / 93632 |
| U1xfZ2 | c64 | `svd_compact` | 292760.9 | 68870.9 | 809.8 | 3 / 49152 | 64 / 2048 | 131 | 0 | 64 | 128 | 14882 / 1636296 |
| U1xfZ2 | c64 | `svd_trunc_rank` | 275325.5 | 65327.0 | 798.5 | 11 / 1088 | 64 / 2048 | 139 | 8 | 64 | 0 | 11432 / 1490676 |
| U1xfZ2 | c64 | `eigh_full` | 249806.2 | 54317.0 | 651.3 | 130 / 49664 | 256 / 3584 | 194 | 64 | 64 | 0 | 56554 / 7826432 |
| U1xfZ2 | c64 | `network_chain3` | 197611.0 | 3620.2 | 103.9 | 2 / 32768 | 0 / 0 | 2 | 128 | 0 | 0 | 6181 / 376504 |

### Per-call device cost, `few-large`

`first` is the first call of the measured operation on a fresh Runtime; for the two
transfer rows the operation also runs during fixture setup, so their first call is
labelled `first_after_setup` in the CSV. All counter columns are warm per-iteration
medians.

| provider | dtype | operation | cuda first us | cuda warm us | host warm us | H2D calls/bytes | D2H calls/bytes | dev allocs | GEMM | solver | copy | caller allocs/bytes |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| U1 | f64 | `to_cuda` | 22.3 | 55.0 | NA | 1 / 131072 | 0 / 0 | 1 | 0 | 0 | 0 | 22 / 526968 |
| U1 | f64 | `to_host` | 42.3 | 41.5 | NA | 0 / 0 | 1 / 131072 | 0 | 0 | 0 | 0 | 9 / 262664 |
| U1 | f64 | `contract_direct` | 216647.9 | 226.9 | 59.9 | 1 / 131072 | 0 / 0 | 1 | 4 | 0 | 0 | 235 / 664824 |
| U1 | f64 | `contract_lazy_adjoint_lhs` | 208487.3 | 227.6 | 61.6 | 1 / 131072 | 0 / 0 | 1 | 4 | 0 | 0 | 243 / 665016 |
| U1 | f64 | `compose` | 206393.1 | 229.5 | 59.5 | 1 / 131072 | 0 / 0 | 1 | 4 | 0 | 0 | 237 / 664840 |
| U1 | f64 | `scale` | 190779.7 | 117.6 | 3.0 | 2 / 131080 | 0 / 0 | 2 | 1 | 0 | 0 | 91 / 660528 |
| U1 | f64 | `add_owned` | 192440.0 | 145.9 | 4.7 | 2 / 131088 | 0 / 0 | 2 | 2 | 0 | 0 | 138 / 661840 |
| U1 | f64 | `add_lazy_fold` | 194352.2 | 162.4 | 71.8 | 2 / 131088 | 0 / 0 | 2 | 2 | 0 | 0 | 143 / 662480 |
| U1 | f64 | `norm` | 193793.4 | 161.5 | 14.5 | 1 / 32 | 1 / 32 | 1 | 4 | 0 | 0 | 224 / 7008 |
| U1 | f64 | `inner` | 193887.0 | 161.6 | 14.5 | 1 / 32 | 1 / 32 | 1 | 4 | 0 | 0 | 224 / 7008 |
| U1 | f64 | `svd_compact` | 213273.0 | 5288.6 | 1311.9 | 3 / 393216 | 4 / 2048 | 11 | 0 | 4 | 8 | 1011 / 2065576 |
| U1 | f64 | `svd_trunc_rank` | 224652.4 | 5404.7 | 1284.3 | 11 / 139264 | 4 / 2048 | 19 | 8 | 4 | 0 | 1341 / 830096 |
| U1 | f64 | `eigh_full` | 485193.9 | 7248.0 | 1543.9 | 10 / 393248 | 16 / 2144 | 14 | 4 | 4 | 0 | 3408 / 2411644 |
| U1 | f64 | `qr_compact` | 222593.0 | 2974.8 | 306.2 | 2 / 262144 | 0 / 0 | 10 | 0 | 4 | 8 | 1386 / 1474372 |
| U1 | f64 | `network_chain3` | 202247.8 | 460.6 | 122.2 | 2 / 262144 | 0 / 0 | 2 | 8 | 0 | 0 | 502 / 1332056 |
| U1 | c64 | `to_cuda` | 74.6 | 47.7 | NA | 1 / 262144 | 0 / 0 | 1 | 0 | 0 | 0 | 22 / 1051256 |
| U1 | c64 | `to_host` | 156.7 | 59.9 | NA | 0 / 0 | 1 / 262144 | 0 | 0 | 0 | 0 | 9 / 524808 |
| U1 | c64 | `contract_direct` | 227261.5 | 286.0 | 190.3 | 1 / 262144 | 0 / 0 | 1 | 4 | 0 | 0 | 235 / 1320184 |
| U1 | c64 | `contract_lazy_adjoint_lhs` | 214545.5 | 285.9 | 195.6 | 1 / 262144 | 0 / 0 | 1 | 4 | 0 | 0 | 243 / 1320376 |
| U1 | c64 | `compose` | 215039.2 | 286.6 | 189.8 | 1 / 262144 | 0 / 0 | 1 | 4 | 0 | 0 | 237 / 1320200 |
| U1 | c64 | `scale` | 189749.1 | 123.1 | 12.3 | 2 / 262160 | 0 / 0 | 2 | 1 | 0 | 0 | 91 / 1315920 |
| U1 | c64 | `add_owned` | 193720.3 | 210.4 | 21.3 | 2 / 262176 | 0 / 0 | 2 | 2 | 0 | 0 | 138 / 1317280 |
| U1 | c64 | `add_lazy_fold` | 190645.7 | 214.7 | 110.5 | 2 / 262176 | 0 / 0 | 2 | 2 | 0 | 0 | 143 / 1317920 |
| U1 | c64 | `norm` | 190446.8 | 165.4 | 94.0 | 1 / 64 | 1 / 64 | 1 | 4 | 0 | 0 | 224 / 7232 |
| U1 | c64 | `inner` | 194065.0 | 166.6 | 53.3 | 1 / 64 | 1 / 64 | 1 | 4 | 0 | 0 | 224 / 7232 |
| U1 | c64 | `svd_compact` | 215438.5 | 8724.6 | 2534.4 | 3 / 786432 | 4 / 2048 | 11 | 0 | 4 | 8 | 1011 / 4031668 |
| U1 | c64 | `svd_trunc_rank` | 233352.7 | 7123.6 | 2549.2 | 11 / 278528 | 4 / 2048 | 19 | 8 | 4 | 0 | 1341 / 1526428 |
| U1 | c64 | `eigh_full` | 237049.1 | 7571.2 | 2190.4 | 10 / 786464 | 16 / 2144 | 14 | 4 | 4 | 0 | 3612 / 4426964 |
| U1 | c64 | `network_chain3` | 207194.6 | 578.2 | 385.7 | 2 / 524288 | 0 / 0 | 2 | 8 | 0 | 0 | 502 / 2642776 |
| fZ2 | f64 | `to_cuda` | 13.4 | 31.9 | NA | 1 / 65536 | 0 / 0 | 1 | 0 | 0 | 0 | 22 / 264824 |
| fZ2 | f64 | `to_host` | 34.3 | 33.1 | NA | 0 / 0 | 1 / 65536 | 0 | 0 | 0 | 0 | 9 / 131592 |
| fZ2 | f64 | `contract_direct` | 209239.2 | 130.3 | 31.9 | 1 / 65536 | 0 / 0 | 1 | 2 | 0 | 0 | 141 / 334472 |
| fZ2 | f64 | `contract_lazy_adjoint_lhs` | 208832.7 | 130.1 | 32.8 | 1 / 65536 | 0 / 0 | 1 | 2 | 0 | 0 | 145 / 334568 |
| fZ2 | f64 | `compose` | 209011.0 | 129.8 | 31.9 | 1 / 65536 | 0 / 0 | 1 | 2 | 0 | 0 | 143 / 334488 |
| fZ2 | f64 | `scale` | 190549.2 | 84.0 | 1.6 | 2 / 65544 | 0 / 0 | 2 | 1 | 0 | 0 | 91 / 332848 |
| fZ2 | f64 | `add_owned` | 192555.7 | 112.0 | 2.4 | 2 / 65552 | 0 / 0 | 2 | 2 | 0 | 0 | 138 / 334160 |
| fZ2 | f64 | `add_lazy_fold` | 192610.7 | 117.6 | 35.9 | 2 / 65552 | 0 / 0 | 2 | 2 | 0 | 0 | 143 / 334800 |
| fZ2 | f64 | `norm` | 196900.4 | 97.1 | 7.3 | 1 / 16 | 1 / 16 | 1 | 2 | 0 | 0 | 127 / 4168 |
| fZ2 | f64 | `inner` | 196576.1 | 97.0 | 7.3 | 1 / 16 | 1 / 16 | 1 | 2 | 0 | 0 | 127 / 4168 |
| fZ2 | f64 | `svd_compact` | 209986.9 | 2673.6 | 658.2 | 3 / 196608 | 2 / 1024 | 7 | 0 | 2 | 4 | 549 / 1037656 |
| fZ2 | f64 | `svd_trunc_rank` | 223576.7 | 2770.4 | 646.2 | 7 / 147456 | 2 / 1024 | 11 | 4 | 2 | 0 | 725 / 809964 |
| fZ2 | f64 | `eigh_full` | 232758.0 | 4068.3 | 774.8 | 6 / 196624 | 8 / 1072 | 8 | 2 | 2 | 0 | 1746 / 1210014 |
| fZ2 | f64 | `qr_compact` | 225904.4 | 1932.0 | 153.7 | 2 / 131072 | 0 / 0 | 6 | 0 | 2 | 4 | 722 / 740458 |
| fZ2 | f64 | `network_chain3` | 202109.1 | 262.4 | 67.9 | 2 / 131072 | 0 / 0 | 2 | 4 | 0 | 0 | 314 / 671272 |
| fZ2 | c64 | `to_cuda` | 47.4 | 26.2 | NA | 1 / 131072 | 0 / 0 | 1 | 0 | 0 | 0 | 22 / 526968 |
| fZ2 | c64 | `to_host` | 46.1 | 45.2 | NA | 0 / 0 | 1 / 131072 | 0 | 0 | 0 | 0 | 9 / 262664 |
| fZ2 | c64 | `contract_direct` | 223843.5 | 173.5 | 97.7 | 1 / 131072 | 0 / 0 | 1 | 2 | 0 | 0 | 141 / 662152 |
| fZ2 | c64 | `contract_lazy_adjoint_lhs` | 215289.3 | 171.2 | 100.5 | 1 / 131072 | 0 / 0 | 1 | 2 | 0 | 0 | 145 / 662248 |
| fZ2 | c64 | `compose` | 217624.4 | 172.2 | 97.8 | 1 / 131072 | 0 / 0 | 1 | 2 | 0 | 0 | 143 / 662168 |
| fZ2 | c64 | `scale` | 189874.8 | 98.1 | 6.2 | 2 / 131088 | 0 / 0 | 2 | 1 | 0 | 0 | 91 / 660560 |
| fZ2 | c64 | `add_owned` | 193562.2 | 162.6 | 10.7 | 2 / 131104 | 0 / 0 | 2 | 2 | 0 | 0 | 138 / 661920 |
| fZ2 | c64 | `add_lazy_fold` | 194272.7 | 165.4 | 53.7 | 2 / 131104 | 0 / 0 | 2 | 2 | 0 | 0 | 143 / 662560 |
| fZ2 | c64 | `norm` | 196272.4 | 123.9 | 47.0 | 1 / 32 | 1 / 32 | 1 | 2 | 0 | 0 | 126 / 4352 |
| fZ2 | c64 | `inner` | 194945.7 | 125.6 | 26.6 | 1 / 32 | 1 / 32 | 1 | 2 | 0 | 0 | 126 / 4352 |
| fZ2 | c64 | `svd_compact` | 214906.1 | 4620.5 | 1267.0 | 3 / 393216 | 2 / 1024 | 7 | 0 | 2 | 4 | 549 / 2020702 |
| fZ2 | c64 | `svd_trunc_rank` | 233429.9 | 4747.5 | 1280.5 | 7 / 294912 | 2 / 1024 | 11 | 4 | 2 | 0 | 725 / 1547250 |
| fZ2 | c64 | `eigh_full` | 242722.4 | 4580.2 | 1097.6 | 6 / 393232 | 8 / 1072 | 8 | 2 | 2 | 0 | 1848 / 2217674 |
| fZ2 | c64 | `network_chain3` | 212832.2 | 350.6 | 201.0 | 2 / 262144 | 0 / 0 | 2 | 4 | 0 | 0 | 314 / 1326632 |
| SU2 | f64 | `to_cuda` | 23.9 | 54.8 | NA | 1 / 131072 | 0 / 0 | 1 | 0 | 0 | 0 | 22 / 526968 |
| SU2 | f64 | `to_host` | 42.3 | 41.1 | NA | 0 / 0 | 1 / 131072 | 0 | 0 | 0 | 0 | 9 / 262664 |
| SU2 | f64 | `contract_direct` | 212899.4 | 227.6 | 59.3 | 1 / 131072 | 0 / 0 | 1 | 4 | 0 | 0 | 235 / 664824 |
| SU2 | f64 | `contract_lazy_adjoint_lhs` | 209442.0 | 228.9 | 61.8 | 1 / 131072 | 0 / 0 | 1 | 4 | 0 | 0 | 243 / 665016 |
| SU2 | f64 | `compose` | 211516.5 | 229.1 | 59.0 | 1 / 131072 | 0 / 0 | 1 | 4 | 0 | 0 | 237 / 664840 |
| SU2 | f64 | `scale` | 193491.6 | 119.5 | 3.0 | 2 / 131080 | 0 / 0 | 2 | 1 | 0 | 0 | 91 / 660528 |
| SU2 | f64 | `add_owned` | 195419.2 | 145.8 | 4.3 | 2 / 131088 | 0 / 0 | 2 | 2 | 0 | 0 | 138 / 661840 |
| SU2 | f64 | `add_lazy_fold` | 195276.0 | 150.9 | 71.9 | 2 / 131088 | 0 / 0 | 2 | 2 | 0 | 0 | 143 / 662480 |
| SU2 | f64 | `norm` | 195721.4 | 162.2 | 14.5 | 1 / 32 | 1 / 32 | 1 | 4 | 0 | 0 | 224 / 7008 |
| SU2 | f64 | `inner` | 198970.2 | 161.9 | 14.5 | 1 / 32 | 1 / 32 | 1 | 4 | 0 | 0 | 224 / 7008 |
| SU2 | f64 | `svd_compact` | 215043.1 | 5294.8 | 1310.8 | 3 / 393216 | 4 / 2048 | 11 | 0 | 4 | 8 | 1011 / 2065576 |
| SU2 | f64 | `svd_trunc_rank` | 227574.2 | 6259.4 | 1283.9 | 11 / 54608 | 4 / 2048 | 19 | 8 | 4 | 0 | 1503 / 529792 |
| SU2 | f64 | `eigh_full` | 239618.3 | 7915.7 | 1543.6 | 10 / 393248 | 16 / 2144 | 14 | 4 | 4 | 0 | 3408 / 2411644 |
| SU2 | f64 | `qr_compact` | 225713.5 | 3094.4 | 306.4 | 2 / 262144 | 0 / 0 | 10 | 0 | 4 | 8 | 1386 / 1474372 |
| SU2 | f64 | `network_chain3` | 202353.8 | 506.8 | 122.2 | 2 / 262144 | 0 / 0 | 2 | 8 | 0 | 0 | 502 / 1332056 |
| SU2 | c64 | `to_cuda` | 76.2 | 47.8 | NA | 1 / 262144 | 0 / 0 | 1 | 0 | 0 | 0 | 22 / 1051256 |
| SU2 | c64 | `to_host` | 62.9 | 61.6 | NA | 0 / 0 | 1 / 262144 | 0 | 0 | 0 | 0 | 9 / 524808 |
| SU2 | c64 | `contract_direct` | 220916.4 | 309.7 | 190.9 | 1 / 262144 | 0 / 0 | 1 | 4 | 0 | 0 | 235 / 1320184 |
| SU2 | c64 | `contract_lazy_adjoint_lhs` | 216131.1 | 311.4 | 196.2 | 1 / 262144 | 0 / 0 | 1 | 4 | 0 | 0 | 243 / 1320376 |
| SU2 | c64 | `compose` | 216928.8 | 313.7 | 190.8 | 1 / 262144 | 0 / 0 | 1 | 4 | 0 | 0 | 237 / 1320200 |
| SU2 | c64 | `scale` | 191813.2 | 139.6 | 12.3 | 2 / 262160 | 0 / 0 | 2 | 1 | 0 | 0 | 91 / 1315920 |
| SU2 | c64 | `add_owned` | 196542.7 | 232.5 | 22.4 | 2 / 262176 | 0 / 0 | 2 | 2 | 0 | 0 | 138 / 1317280 |
| SU2 | c64 | `add_lazy_fold` | 198225.1 | 239.1 | 110.3 | 2 / 262176 | 0 / 0 | 2 | 2 | 0 | 0 | 143 / 1317920 |
| SU2 | c64 | `norm` | 195863.5 | 187.3 | 53.1 | 1 / 64 | 1 / 64 | 1 | 4 | 0 | 0 | 224 / 7232 |
| SU2 | c64 | `inner` | 196572.7 | 193.1 | 53.3 | 1 / 64 | 1 / 64 | 1 | 4 | 0 | 0 | 224 / 7232 |
| SU2 | c64 | `svd_compact` | 219494.6 | 9169.4 | 2564.1 | 3 / 786432 | 4 / 2048 | 11 | 0 | 4 | 8 | 1011 / 4031668 |
| SU2 | c64 | `svd_trunc_rank` | 239595.8 | 8009.5 | 2552.3 | 11 / 109216 | 4 / 2048 | 19 | 8 | 4 | 0 | 1503 / 802844 |
| SU2 | c64 | `eigh_full` | 241461.2 | 8229.3 | 2194.5 | 10 / 786464 | 16 / 2144 | 14 | 4 | 4 | 0 | 3612 / 4426964 |
| SU2 | c64 | `network_chain3` | 212819.5 | 627.0 | 389.6 | 2 / 524288 | 0 / 0 | 2 | 8 | 0 | 0 | 502 / 2642776 |
| U1xfZ2 | f64 | `to_cuda` | 24.3 | 55.0 | NA | 1 / 131072 | 0 / 0 | 1 | 0 | 0 | 0 | 22 / 526968 |
| U1xfZ2 | f64 | `to_host` | 43.6 | 42.3 | NA | 0 / 0 | 1 / 131072 | 0 | 0 | 0 | 0 | 9 / 262664 |
| U1xfZ2 | f64 | `contract_direct` | 214224.2 | 229.4 | 60.1 | 1 / 131072 | 0 / 0 | 1 | 4 | 0 | 0 | 235 / 664824 |
| U1xfZ2 | f64 | `contract_lazy_adjoint_lhs` | 211692.4 | 229.4 | 61.4 | 1 / 131072 | 0 / 0 | 1 | 4 | 0 | 0 | 243 / 665016 |
| U1xfZ2 | f64 | `compose` | 211480.8 | 230.6 | 60.3 | 1 / 131072 | 0 / 0 | 1 | 4 | 0 | 0 | 237 / 664840 |
| U1xfZ2 | f64 | `scale` | 197102.8 | 117.2 | 3.0 | 2 / 131080 | 0 / 0 | 2 | 1 | 0 | 0 | 91 / 660528 |
| U1xfZ2 | f64 | `add_owned` | 197473.3 | 145.7 | 4.1 | 2 / 131088 | 0 / 0 | 2 | 2 | 0 | 0 | 138 / 661840 |
| U1xfZ2 | f64 | `add_lazy_fold` | 197856.6 | 153.8 | 72.0 | 2 / 131088 | 0 / 0 | 2 | 2 | 0 | 0 | 143 / 662480 |
| U1xfZ2 | f64 | `norm` | 198949.3 | 164.3 | 14.5 | 1 / 32 | 1 / 32 | 1 | 4 | 0 | 0 | 224 / 7008 |
| U1xfZ2 | f64 | `inner` | 201081.4 | 164.3 | 14.5 | 1 / 32 | 1 / 32 | 1 | 4 | 0 | 0 | 224 / 7008 |
| U1xfZ2 | f64 | `svd_compact` | 217764.0 | 5297.7 | 1314.9 | 3 / 393216 | 4 / 2048 | 11 | 0 | 4 | 8 | 1011 / 2065576 |
| U1xfZ2 | f64 | `svd_trunc_rank` | 229028.2 | 5420.0 | 1287.4 | 11 / 139264 | 4 / 2048 | 19 | 8 | 4 | 0 | 1341 / 830096 |
| U1xfZ2 | f64 | `eigh_full` | 240080.7 | 7204.1 | 1544.7 | 10 / 393248 | 16 / 2144 | 14 | 4 | 4 | 0 | 3408 / 2411644 |
| U1xfZ2 | f64 | `qr_compact` | 227186.0 | 3007.7 | 307.3 | 2 / 262144 | 0 / 0 | 10 | 0 | 4 | 8 | 1386 / 1474372 |
| U1xfZ2 | f64 | `network_chain3` | 205095.4 | 465.6 | 124.9 | 2 / 262144 | 0 / 0 | 2 | 8 | 0 | 0 | 502 / 1332056 |
| U1xfZ2 | c64 | `to_cuda` | 47.9 | 47.9 | NA | 1 / 262144 | 0 / 0 | 1 | 0 | 0 | 0 | 22 / 1051256 |
| U1xfZ2 | c64 | `to_host` | 64.4 | 61.1 | NA | 0 / 0 | 1 / 262144 | 0 | 0 | 0 | 0 | 9 / 524808 |
| U1xfZ2 | c64 | `contract_direct` | 224092.6 | 297.8 | 191.6 | 1 / 262144 | 0 / 0 | 1 | 4 | 0 | 0 | 235 / 1320184 |
| U1xfZ2 | c64 | `contract_lazy_adjoint_lhs` | 218162.5 | 301.8 | 196.9 | 1 / 262144 | 0 / 0 | 1 | 4 | 0 | 0 | 243 / 1320376 |
| U1xfZ2 | c64 | `compose` | 220267.9 | 302.5 | 190.6 | 1 / 262144 | 0 / 0 | 1 | 4 | 0 | 0 | 237 / 1320200 |
| U1xfZ2 | c64 | `scale` | 195706.4 | 135.3 | 12.3 | 2 / 262160 | 0 / 0 | 2 | 1 | 0 | 0 | 91 / 1315920 |
| U1xfZ2 | c64 | `add_owned` | 199704.9 | 226.2 | 21.3 | 2 / 262176 | 0 / 0 | 2 | 2 | 0 | 0 | 138 / 1317280 |
| U1xfZ2 | c64 | `add_lazy_fold` | 199772.3 | 230.1 | 110.7 | 2 / 262176 | 0 / 0 | 2 | 2 | 0 | 0 | 143 / 1317920 |
| U1xfZ2 | c64 | `norm` | 200902.9 | 177.8 | 94.0 | 1 / 64 | 1 / 64 | 1 | 4 | 0 | 0 | 224 / 7232 |
| U1xfZ2 | c64 | `inner` | 199605.1 | 177.0 | 53.2 | 1 / 64 | 1 / 64 | 1 | 4 | 0 | 0 | 224 / 7232 |
| U1xfZ2 | c64 | `svd_compact` | 222815.1 | 8766.9 | 2563.7 | 3 / 786432 | 4 / 2048 | 11 | 0 | 4 | 8 | 1011 / 4031668 |
| U1xfZ2 | c64 | `svd_trunc_rank` | 244470.0 | 7817.3 | 2564.5 | 11 / 278528 | 4 / 2048 | 19 | 8 | 4 | 0 | 1341 / 1526428 |
| U1xfZ2 | c64 | `eigh_full` | 243983.8 | 7473.4 | 2189.1 | 10 / 786464 | 16 / 2144 | 14 | 4 | 4 | 0 | 3612 / 4426964 |
| U1xfZ2 | c64 | `network_chain3` | 217049.3 | 605.3 | 387.8 | 2 / 524288 | 0 / 0 | 2 | 8 | 0 | 0 | 502 / 2642776 |
