# CUDA single-precision device probe (f32 / Complex32) — 2026-09-20

Leaf C0 of the single-precision plan (#1303; survey
`reviews/gpu-phase-20260920/single-precision-survey.md`, roadmap #1139/#1065).
Test-only: no production code, no dependency change. `CudaScalar` stays sealed
to `f64`/`Complex64`; the probe calls Tenferro 0.5.0 directly.

Probe: `tenet-dense/tests/cuda_single_precision_probe.rs`.

## Environment

| | |
|---|---|
| TeNeT base | `origin/main` `fca39850`, branch `c0-single-precision-probe` |
| Host | `qg1`, `/data2/ryo-w/gpu-phase/c0`, cargo 1.96.0 |
| GPU | NVIDIA A100-SXM4-40GB, driver 560.35.05, `CUDA_VISIBLE_DEVICES=0` |
| CUDA | 12.6 (`/usr/local/cuda-12.6`) |
| cuTENSOR | `libcutensor.so.2.5.0` via `TENFERRO_CUTENSOR_PATH` |
| Tenferro | 0.5.0 (`tenferro-tensor`, `tenferro-gpu`, `tenferro-linalg`) |
| Build | `dev` (unoptimized + debuginfo), `--no-default-features --features cuda,cpu-faer` |
| Command | `cargo test -p tenet-dense --no-default-features --features cuda,cpu-faer --test cuda_single_precision_probe -- --ignored --test-threads=1 --nocapture` |
| Result | `20 passed; 0 failed`, 6.96 s wall |

Oracles are independent host loops in the same dtype (never a TeNeT descriptor,
never the f64 device path). Pure moves are compared bitwise; arithmetic at
`16·√n·eps(real(T))`; factorization laws at `√eps(real(T))·√n` relative to the
fixture scale.

## Operation coverage

The probe's operation list is exactly what `tenet-dense/src/cuda_adapter.rs`
calls today, so each row is one Tenferro call a later `CudaScalar` f32/C32 leaf
would newly reach.

| Operation (Tenferro call) | Adapter site | f32 | Complex32 |
|---|---|---|---|
| `upload_tensor` / `download_tensor`, flat buffer | `CudaDenseStorage::upload_owned` / `download` | PASS (bitwise) | PASS (bitwise) |
| `backend_region_view{,_mut}`, strided + offset | `region_view_strided` / `region_view_mut` | PASS | PASS |
| `dot_general_read_into_accum`, rank-2 region GEMM, all 9 `MatrixOp` pairs (Identity/Transpose/Adjoint), α∈{1, −0.75+0.5i}, β∈{0,1}, m×k×n ∈ {3×4×2, 1×6×1, 5×1×4, 4×4×4} | `cuda_gemm_region_strided_into`, `cuda_gemm_region_with_ops_into`; `weighted_inner_cuda` (`tenet/src/typed.rs:11760`) uses the `Adjoint` row-view form | PASS | PASS |
| `dot_general_read_into_accum`, N-D strided region vs 1×1 ones, rank 3/4/5, permuted source and destination orders, conj, α/β | `cuda_axpby_owned` (`typed.rs:11529`) and the #1298/#1301 form | PASS (bitwise at α=1) | PASS (bitwise at α=1) |
| `dot_general_read_into_accum`, **outer product** (no contracted modes) into a permuted-stride region, `lhs_conj` | planned `otimes`/G2 | **PASS** | **PASS** |
| `to_contiguous_read` of a strided region | `cuda_is_hermitian_region` | PASS (bitwise) | PASS (bitwise) |
| `abs` | `cuda_is_hermitian_region`, `magnitudes_for_sum_squares` | PASS (output F32) | PASS (**output F32**, not C32) |
| `conj` | `cuda_is_hermitian_region` | n/a (skipped on the dtype invariant) | PASS (exact-zero A−Aᴴ residual) |
| `reduce_max` | `cuda_is_hermitian_region` | PASS (**returns F32**) | PASS (**returns F32**) |
| `reduce_sum_squares_read` | `cuda_is_hermitian_region` | PASS (**returns F32**) | PASS (**returns F32**) |
| `sub` | `cuda_is_hermitian_region` | PASS | PASS |
| `div` by a rank-0 **real** scalar | `cuda_is_hermitian_region` + `upload_scalar` | PASS | PASS |
| `div` by a rank-0 **f64** scalar (today's `upload_scalar`) | `cuda_adapter.rs:715` | **REJECTED** | **REJECTED** |
| `div` by a rank-0 **payload-typed complex** scalar | not a production shape | n/a | **REJECTED** |
| rank-0 scalar `upload_tensor`, real metadata `download_tensor` | `upload_scalar` / `download_values` | PASS (F32 payload) | PASS (F32 payload) |
| `copy_read_into` into a strided `ld > rows` region | `cuda_copy_region_into` | PASS (bitwise) | PASS (bitwise) |
| `svd_read` on a region, 5×3 / 3×5 / 4×4 | `cuda_svd_region` | PASS | PASS |
| `qr_with_options_read(PositiveDiagonal)`, 5×3 / 4×4 | `cuda_qr_region` | PASS | **FAIL (#1833)** |
| `eigh_read`, 4×4 / 6×6 | `cuda_eigh_region` | PASS | PASS |
| `solve` | not reached today (#1065 gate) | PASS (residual 0.000e0) | **FAIL (#1833)** |
| `lu` | not reached today (#1065 gate) | PASS (P/L/U 3×3, parity rank 0) | **FAIL (#1833)** |

Factorization laws checked in every PASS row: reconstruction
(`U diag(s) Vᵗ = A`, `QR = A`, `A v_j = λ_j v_j`), orthonormality (`UᴴU = I`,
`VᴴV = I`, `QᴴQ = I`, `VᴴV = I`), singular values non-negative and descending,
eigenvalues real and ascending, and `R_jj` real non-negative for the
positive-diagonal gauge.

## Exact failure texts

### Complex32 device QR — `qr_region_c32_is_blocked_by_the_complex_zero_kernel`

```
qr: backend failure: The server is in an invalid state

Caused by:
  [Launch(A compilation error happened during launch

Caused by:
  An error caused the compilation to fail

Caused by:
  [Compilation Error]
    default_program(156): error: no suitable constructor exists to convert from "uint32" (aka "unsigned int") to "float2"
      const cuFloatComplex l_16 = cuFloatComplex(uint32(0));
                                                 ^
    1 error detected in the compilation of "default_program".
```

The dumped kernel is `triu_kernel` over `cuFloatComplex*`. This is
tenferro-rs#1833 exactly as the survey predicted: `zero_value`/`one_value` are
`E::cast_from(0u32)` / `(1u32)` (tenferro-gpu 0.5.0
`src/kernels/helpers.rs:84-86`), which NVRTC cannot construct for
`cuFloatComplex`. The C32 form is the `float2` twin of the C64 failure behind
#1271; it is fixed upstream but unreleased in 0.5.0.

### Complex32 `solve` and `lu` — `lu_and_solve_behaviour_c32_is_blocked`

```
lu_factor: backend failure: The server is in an invalid state

Caused by:
  [Launch(A compilation error happened during launch

Caused by:
  An error caused the compilation to fail

Caused by:
  [Compilation Error]
    default_program(141): error: no suitable constructor exists to convert from "uint32" (aka "unsigned int") to "float2"
      const cuFloatComplex l_2 = cuFloatComplex(uint32(1));
                                                ^
    1 error detected in the compilation of "default_program".
```

Same defect class, `one_value` arm (the LU identity/permutation fill).

### Scalar dtype rejections — `hermitian_kernel_chain_*`

```
div(f32, f64 scalar): rejected with `div: dtype mismatch: expected F32, actual F64`
div(Complex32, f64 scalar): rejected with `div: dtype mismatch: expected C32, actual F64`
div(Complex32, Complex32 scalar): rejected with `div: incompatible shapes: lhs=[], rhs=[4, 4]`
```

## Findings the survey did not have, or got differently

1. **The TensorKit Float32 outer-product failure does not reproduce.**
   TensorKit's CUDA suite marks `⊗` as "broken because of cuTENSOR" for
   `Float32` (`test/cuda/tensors.jl:523`). Through Tenferro 0.5.0 on cuTENSOR
   2.5.0, the zero-contracted-mode `dot_general_read_into_accum` into a
   permuted-stride destination is correct for both F32 and C32. The TensorKit
   note is wrapper-specific and must not be carried into TeNeT as a boundary.

2. **Real metadata comes back as F32, not F64.** SVD singular values, EIGH
   eigenvalues, `abs`, `reduce_max` and `reduce_sum_squares` all produce
   `DType::F32` for a single-precision payload. `download_values`
   (`cuda_adapter.rs:677-694`) accepts `Tensor::F64` only and would fail with
   "expected f64 values, got F32". Confirmed hazard; C1 must widen it.

3. **`upload_scalar` must be typed by the real component, not by `f64` and not
   by the payload.** The f64 scalar is a hard dtype mismatch for both single
   dtypes. Independently, a *complex* rank-0 divisor is also rejected — with a
   shape error whose operands are printed reversed
   (`lhs=[], rhs=[4, 4]` for `div(normal, scalar)`). So the production shape is
   specifically "complex tensor ÷ real rank-0 scalar"; C1 must keep the divisor
   in `D::Real` and must not reach for payload-typed symmetry.

4. **`abs` narrows C32 → F32**, mirroring C64 → F64. The
   `magnitudes_for_sum_squares` skip-on-`!IS_COMPLEX` invariant carries over
   unchanged.

5. **F32 `lu`/`solve` work on device.** The f64 probe
   (`cuda-strided-region-probe-2026-09-20.md`) already showed the real lane
   works; this confirms it for f32 with a zero residual on a diagonally
   dominant 3×3. It does not make them supported in TeNeT: #1065 keeps `solve`
   behind its own proof gate (leaf H7).

6. **No notable first-call NVRTC stall per dtype.** Each of the 20 tests
   constructs its own `CudaBackend` (so cuTENSOR/cuSOLVER/cuBLAS init is paid
   20 times) and the whole `--test-threads=1` run finishes in 6.96 s wall.
   Nothing about F32/C32 kernel compilation stands out against the f64 baseline
   in `cuda-baseline-2026-09-20.md`; warm-up (leaf C3) should still cover each
   admitted dtype, but there is no new per-dtype cost signal here.

## Operations a later `CudaScalar` f32/Complex32 leaf must keep `Unsupported`

Device-blocked by tenferro-rs#1833 in Tenferro 0.5.0, for `Complex32` only:

- **`cuda_qr_region::<Complex32>`** — joins #1271 (C64 device QR). Leaf C4 must
  carry a compile-time boundary and a `compile_fail` doctest next to the C64
  one (`tenet/src/typed.rs:11340`), not a runtime fallback.
- **`lu` / `solve` for `Complex32`** — not reachable from TeNeT today; when
  leaf H7 admits `solve`, the device path must be an explicit capability error
  for C32.
- **Complex zero-fill / `triu`-shaped device kernels for `Complex32`** — any new
  G2/G3 primitive that materializes a complex zero or one on device inherits
  this failure. Write them generic over `CudaScalar` and gate C32 with the same
  boundary as C64.

Nothing is blocked for **f32**: every probed operation passes.

## Residual risks and what this probe does not prove

- **T1 (Hermitian admission constant) is not exercised numerically.** The
  fixture is exactly Hermitian, so the conjugate-transpose residual is exactly
  0 and `scaled_hermitian_residual_accepts` accepts at any constant. The probe
  only records that `64·f64::EPSILON = 1.42e-14` and `64·f32::EPSILON = 7.63e-6`
  differ by ~5·10⁸, so an f32 block with genuine rounding noise would be
  rejected by the current constant. Proving that belongs to C1, where the
  constant becomes `D::Real::EPSILON`-scaled.
- Debug build only; this is a correctness probe, no timing claim is made.
- One device (A100, sm_80), one cuTENSOR (2.5.0) and one CUDA (12.6). A
  different architecture or cuTENSOR could select different contraction
  algorithms.
- Fixtures are small (≤ 6×6 factorizations, ≤ rank 5 regions) and
  well-conditioned: this probes capability and semantics, not f32 conditioning
  behaviour on realistic TeNeT block sizes.
- TeNeT's own device path is not exercised; `CudaScalar` remains sealed, so
  nothing here proves the adapter's offsets, counters or error mapping for
  single precision.
