# `CudaScalar` for `f32` / `Complex32` with a real lane — leaf C1 (#1326)

Device evidence for admitting the two single-precision payloads to the sealed
`tenet-dense` trait `CudaScalar`. Survey
`reviews/gpu-phase-20260920/single-precision-survey.md` (C1 row); probe
`benchmarks/history/cuda-single-precision-probe-2026-09-20.md` (leaf C0, #1303).

No typed tensor API opens here: `tenet`'s `CudaPayload` stays `f64`/`Complex64`
(leaf C2). No dependency change.

## Environment

| | |
|---|---|
| TeNeT base | `origin/main` `eb9d50ce`, branch `c1-cuda-scalar-single-precision` |
| Host | `qg1`, `/data2/ryo-w/gpu-phase/c1`, private target `/data2/ryo-w/gpu-phase/c1-target` (removed after the run) |
| GPU | NVIDIA A100-SXM4-40GB, `CUDA_VISIBLE_DEVICES=1` (GPU 1 had no process of any user for the whole run) |
| CUDA / cuTENSOR | 12.6 (`/usr/local/cuda-12.6`) / `libcutensor.so.2.5.0` via `TENFERRO_CUTENSOR_PATH` |
| Tenferro | 0.5.0 (`tenferro-tensor`, `tenferro-gpu`, `tenferro-linalg`), unchanged |
| Build | `dev`, `--no-default-features --features cuda,cpu-faer` |

## The real lane

Every admitted payload names a real lane through Tenferro's own
`TensorScalar::Real` (`tenferro-tensor-0.5.0/src/types.rs:3477`, instantiated at
`:3721-3727`): `f32` for `f32`/`Complex32`, `f64` for `f64`/`Complex64`. C1 adds
one sealed trait, `CudaRealScalar`, over exactly those two types, and types
every real-metadata path by `D::Real` instead of by `f64`:

| Adapter entry point | Before | After |
|---|---|---|
| `download_values` | matched `Tensor::F64` only | generic over `R: CudaRealScalar`; the lane's values are downloaded and widened to `f64` on the host |
| `download_scalar` | `f64` | `R` |
| `upload_scalar` (rank-0 divisor) | `f64` | `R` — probe finding 3: an `F64` divisor is a dtype mismatch against an `F32`/`C32` tensor, and a payload-typed complex divisor is a shape error |
| `cuda_svd_region` singular values | `f64` download | `D::Real` download, widened |
| `cuda_eigh_region` eigenvalues | `f64` download | `D::Real` download, widened |
| `cuda_is_hermitian_region` reductions and divisors | `f64` | `D::Real` |
| Hermitian tolerance | `64 * f64::EPSILON` for every payload | `64 * eps(real(D))` |

Spectra stay `f64` at the TeNeT level: the widening happens immediately after
the transfer, so `Vec<f64>` remains the signature every host-side truncation
decision consumes, and the bytes moved are the lane's own.

## Hermitian tolerance

`tenet-dense/src/cuda_hermitian.rs` now owns the admission rule as pure host
arithmetic, with the tolerance as a parameter. The constant is the Host twin's:
`normwise_hermitian` (`tenet-matrixalgebra/src/factorize.rs:796`) tests
`||(A - A†)/2||_F <= R::relative_tolerance() * ||A||_F` with
`relative_tolerance() == 64 * R::EPSILON` for `R` in `{f32, f64}`
(`factorize.rs:177`, `:187`).

TensorKit agrees on the shape, not on a number: `ishermitian`
(`TensorKit.jl` `cfaa073e`, `src/factorizations/factorizations.jl:67`) forwards
each block to MatrixAlgebraKit 0.6.9 `ishermitian`
(`src/common/matrixproperties.jl:72`), which is exact equality by default and,
when a tolerance is requested, derives it from the payload's own epsilon —
`default_hermitian_tol(A) = eps(norm(A, Inf))^(3/4)` (`src/common/defaults.jl:43`),
used by `strided_ishermitian_approx` (`matrixproperties.jl:150`). Neither
reference ever measures a single-precision block against a double-precision
epsilon, which is what the device rule did before this leaf: `64 * eps(f64) =
1.4e-14` against the `64 * eps(f32) = 7.6e-6` an `f32` block is entitled to.

`f64` and `Complex64` decisions are unchanged — the same expression with the
same constant.

## Unsupported boundaries

Device QR of a **complex** payload is a typed `DenseError::Unsupported`, raised
by `ensure_device_constant_kernels` **before** `ensure_cuda_device` and before
any view, upload or submission. Cause: Tenferro 0.5.0 materializes a kernel
constant as `E::cast_from(0u32)` / `(1u32)`
(`tenferro-gpu-0.5.0/src/kernels/helpers.rs:84-86`), which NVRTC cannot
construct from `uint32`.

Scope note — the boundary covers **both** complex dtypes, not only the new one.
The first device run of this leaf exercised `Complex64` QR as a supposedly
supported case and it failed inside the launch:

```
Complex64 QR 5x3: Cuda backend error in cuda_qr: qr: backend failure: The server is in an invalid state
  [Compilation Error]
    default_program(156): error: no suitable constructor exists to convert from "uint32" (aka "unsigned int") to "double2"
      const cuDoubleComplex l_16 = cuDoubleComplex(uint32(0));
```

That is #1271, the `double2` arm of the identical defect (the `float2` arm is
tenferro-rs#1833, probe leaf C0). Gating only the single-precision arm would
have left the same defect reported two different ways for two callers of one
kernel, and would have made the capability const say something false about
`Complex64`. Both complex payloads therefore carry
`DEVICE_CONSTANT_KERNELS = false`. Nothing regresses: the typed layer offers
device QR for `f64` only (`tenet/src/typed.rs:11350`, with the `Complex64`
`compile_fail` doctests of #1271 at `:11320-11349`), so no TeNeT path reached
`cuda_qr_region::<Complex64>`; what changes is that a direct adapter call now
gets a typed capability error instead of an NVRTC compile log.

The same capability const covers LU and `solve` when leaf H7 admits them;
neither is reachable from the adapter today, so there is no code to gate. Both
real payloads are unaffected, and `f32` QR is exercised here against its laws.

## Neutrality for `f64` / `Complex64`

- `warm_up` submits `f64` work only and its documented counter deltas are
  unchanged (`h2d_calls` 3, `h2d_bytes` 24, `d2h_calls` 1, `d2h_bytes` 8,
  `device_allocs` 4, `gemm_calls` 1, `solver_calls` 1, `copy_calls` 0), pinned
  by `warm_up_costs_one_gemm_one_solver_call_and_a_bounded_fixed_traffic`. The
  scalar operands stay lazy per dtype, so a caller that never touches single
  precision pays nothing for it. Probe finding 6 measured no per-dtype NVRTC
  stall.
- `cuda_transfer_bytes_scale_with_the_payload_dtype` and
  `cuda_transfer_counters_attribute_one_upload_download_and_gemm` are unchanged
  and still pass.
- The context's scalar operands moved from two slots keyed by
  `IS_COMPLEX` to four keyed by dtype, because the operands are *payload*-typed:
  a shared slot would have handed an `f32` call the `f64` context `1`. The
  `f64` upload counts per fill are unchanged
  (`the_zero_template_is_uploaded_once_per_dtype_and_grows_monotonically`), and
  `scalar_operand_bytes` now charges each slot its own element size.

## Device results

All runs: `qg1`, GPU 1 (no process of any user on it for the whole run), `dev`
profile, `--test-threads=1`, private target `/data2/ryo-w/gpu-phase/c1-target`
(removed afterwards). Log `/data2/ryo-w/gpu-phase/c1-final2.log`.

Targeted device tests of the touched crates, non-ignored tests with the `cuda`
feature set, their doctests, and the full workspace device suite, in one run:

```
=== targeted ignored
exit_targeted=0
=== non-ignored
exit_nonignored=0
=== doc
exit_doc=0
=== full suite
exit_suite=0
```

535 tests passed, 0 failed in total; the full-suite section alone is
`passed 141 failed 0`.

The new four-dtype suite `tenet-dense/tests/cuda_scalar_dtypes.rs` (rerun
verbatim against the final source):

```
running 8 tests
test complex_device_qr_is_rejected_before_any_device_work ... ok
test device_spectra_are_widened_to_f64_for_every_dtype ... ok
test every_admitted_dtype_reports_its_own_tag_and_rejects_the_others ... ok
test region_eigh_obeys_its_laws_for_every_dtype ... ok
test region_gemm_matches_a_double_precision_oracle_for_every_dtype_and_op_pair ... ok
test region_qr_obeys_its_laws_for_every_supported_dtype ... ok
test region_svd_obeys_its_laws_for_every_dtype ... ok
test the_hermitian_rule_scales_with_the_payload_real_lane ... ok

test result: ok. 8 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 4.85s
```

The region primitive and the copy-route guard at all four dtypes:

```
test nd_region_axpby_matches_a_host_strided_loop_c32 ... ok
test nd_region_axpby_matches_a_host_strided_loop_c64 ... ok
test nd_region_axpby_matches_a_host_strided_loop_f32 ... ok
test nd_region_axpby_matches_a_host_strided_loop_f64 ... ok
test the_zero_template_is_uploaded_once_per_dtype_and_grows_monotonically ... ok
test the_context_zero_template_overwrites_exactly_its_region ... ok
test a_zero_coefficient_propagates_nan_from_the_source ... ok
test overwrite_is_independent_of_a_nan_poisoned_destination ... ok
test copy_region_is_correct_at_every_destination_offset ... ok
```

The `f64`/`Complex64` neutrality pins, unchanged and still passing:

```
test cuda_adapter::tests::cuda_hermitian_region_is_scaled_and_downloads_only_scalar_metadata ... ok
test cuda_adapter::tests::cuda_transfer_bytes_scale_with_the_payload_dtype ... ok
test cuda_adapter::tests::cuda_transfer_counters_attribute_one_upload_download_and_gemm ... ok
test cuda_adapter::tests::warm_up_costs_one_gemm_one_solver_call_and_a_bounded_fixed_traffic ... ok
```

The tree-transform executor, now instantiated for all four dtypes over the
structure-level and categorical fixtures:

```
test device_replay_matches_the_oracle_and_the_host_for_every_fixture ... ok
test every_caller_scale_matches_the_host_and_the_oracle ... ok
test overwrite_cleans_a_nan_poisoned_destination_including_inactive_layouts ... ok
test overwrite_cleans_a_nan_poisoned_destination_around_recoupling_blocks ... ok
test a_warm_complex_replay_is_transfer_free_and_plan_stable_for_every_caller_scale ... ok
test alternating_complex_recoupling_structures_upload_their_matrices_once_each ... ok
```

No timing claim is made: `dev` profile, correctness only. One device, one
cuTENSOR (2.5.0), one CUDA (12.6), small fixtures — this proves capability and
semantics, not `f32` conditioning on realistic block sizes.

## Residuals

- **C2** (`CudaPayload` for `f32`/`Complex32`): not opened here. The typed
  layer still seals `TensorScalar` and `CudaPayload` to `f64`/`Complex64`, so
  no user-visible single-precision device tensor exists yet.
- **C3** (warm-up per dtype): nothing was added. The libraries this warms are
  per backend instance, not per dtype, and the probe found no per-dtype stall;
  what a single-precision caller still pays on first use is the CubeCL kernel
  JIT for its dtype and its own scalar operands, both lazy.
- **C4** (device factorizations as a typed API): the adapter-level regions are
  proven for all four dtypes here (minus `Complex32` QR), but the typed
  `compile_fail` doctest next to the `Complex64` one (`tenet/src/typed.rs:11340`)
  belongs to C4, because no typed single-precision device entry point exists to
  reject yet.
