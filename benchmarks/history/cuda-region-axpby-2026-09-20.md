# CUDA region axpby adapter (issue #1301, leaf G2a-1)

Device evidence for `cuda_region_axpby` / `cuda_region_zero`, the strided N-D
region primitive of `reviews/gpu-phase-20260920/g2-design.md` §8: a strided,
offset, rank-N source region times a 1x1 **data** coefficient operand, written
(`beta = 0`) or accumulated (`beta = 1`) into a strided, offset, rank-N
destination region of a distinct buffer, with optional conjugation.

Correctness evidence only; no timings are claimed and none are needed — the
primitive performs the same one strided pass per block as the host, and the
probe (#1298) already recorded the plan-cache and transfer constants.

## Environment

| Item | Value |
|---|---|
| Host | qg1 (`GPUA100-gleap-01`), NVIDIA A100-SXM4-40GB, driver 560.35.05, `CUDA_VISIBLE_DEVICES=0` |
| Toolkit | CUDA 12.6, cuTENSOR 2.5.0 via `TENFERRO_CUTENSOR_PATH` (the `.so` file) |
| Toolchain | cargo/rustc 1.96.0, debug (`test` profile) |
| TeNeT | worktree `g2a1-region-axpby` at `origin/main` `fca39850`, Cargo.lock unchanged |
| Tenferro | `tenferro-tensor`/`-gpu`/`-linalg` 0.5.0 |
| Sources | `/data2/ryo-w/gpu-phase/g2a1`, target `/data2/ryo-w/gpu-phase/gl2-target` |

## Runs

New binary, `tenet-dense/tests/cuda_region_axpby.rs` (14 `#[ignore]` tests,
after the independent source review's P2 follow-ups):

```
cargo test -p tenet-dense --no-default-features --features cuda,cpu-faer \
  --test cuda_region_axpby -- --ignored --test-threads=1 --nocapture
test result: ok. 14 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 3.89s
```

Full device suite:

```
cargo test --workspace --lib --tests --no-default-features --features cuda,cpu-faer \
  --no-fail-fast -- --ignored --skip measure_checked_generic_transform_phases \
  --skip axioms_ --skip itebd_ --skip cross_library --test-threads=1
97 `test result: ok` lines, 76 device tests passed, 0 failed
```

The pre-existing device results are unchanged, including
`warm_up_costs_one_gemm_one_solver_call_and_a_bounded_fixed_traffic`: the
context's scalar operands are created on first use of their dtype, not during
warm-up, so the warm-up's documented counter deltas still hold exactly.

## What the 12 tests establish

| Test | Evidence |
|---|---|
| `nd_region_axpby_matches_a_host_strided_loop_f64` / `_c64` | ranks 3–6, source and destination axis orders differing, leading strides > 1, three blocks per buffer at unrelated offsets, `conj` on/off, `beta` ∈ {overwrite, accumulate}, coefficient 1 from the context and real/complex coefficients read from a shared coefficient vector at the block's own offset — against an independent host strided walk. Coefficient-1 cases are compared **bitwise**, so "finite payloads are unaffected by the multiply" is tested, not assumed; other coefficients at `1e-12` relative |
| `a_zero_coefficient_propagates_nan_from_the_source` | coefficient 0 read as a data operand keeps `0 * NaN = NaN`, as the host does; a descriptor alpha of 0 would have erased it |
| `overwrite_is_independent_of_a_nan_poisoned_destination` | `beta = 0` into an all-NaN destination writes clean values; untouched positions keep their NaN |
| `infinite_payload_behaviour_is_recorded` | see below |
| `the_context_zero_template_overwrites_exactly_its_region` | the context zero template read as a packed source zeroes a permuted-stride region and nothing else |
| `the_zero_template_is_uploaded_once_per_dtype_and_grows_monotonically` | 2 uploads on the first fill of a dtype (the `1` and the template), 0 for any shorter fill, exactly 1 when the template must grow, and the two dtypes are independent |
| `a_broadcast_source_axis_forms_an_outer_product` | a stride-0 *source* axis is accepted (outer product with ones); stride 0 on the destination is rejected |
| `a_diagonal_source_view_extracts_the_block_diagonal` | a merged `s_row + s_col` source stride reads the block diagonal, compared bitwise against the unmerged host walk |
| `every_rejection_is_typed_and_submits_no_device_work` | dims mismatch, coefficient offset out of bounds, source/destination out of bounds, non-injective destination, dtype mismatch — each the exact typed `DenseError`, with `h2d_calls`, `d2h_calls`, `device_allocs`, `gemm_calls`, `solver_calls` and `copy_calls` all 0 across the whole rejection phase |
| `a_zero_extent_region_is_a_no_op_without_a_submission` | a zero-extent region returns `Ok`, touches nothing, and submits nothing |
| `the_call_phase_moves_nothing_across_the_host_boundary` | 16 region calls: `h2d_calls = 0`, `h2d_bytes = 0`, `d2h_calls = 0`, `device_allocs = 0`, `gemm_calls = 16` |
| `gapped_and_interleaved_destinations_are_accepted` | a sub-block of a wider parent (`s_{k+1} > d_k·s_k`) and the boundary case `dims [2,2] strides [2,3]` — the layouts that separate the host's cumulative-span rule from a stricter one — are accepted by the backend and land exactly where the host walk says |
| `a_reserved_zero_template_makes_every_later_fill_transfer_free` | `reserve_zero_template` sizes the template once, three fills of different sizes then cost one upload in total (the `1` operand), `scalar_operand_bytes` reports the pinned bytes, and `release_scalar_operands` returns them |

## Recorded non-finite behaviour (the one disclosed deviation)

Verbatim, with a coefficient of 1:

```
f64 +/-inf through a coefficient of 1: [inf, -inf]
Complex64 +/-inf through a coefficient of 1: [Complex { re: NaN, im: NaN }, Complex { re: NaN, im: NaN }]
```

The host copies bit-exactly when the coefficient is 1
(`host_scalar_kernels.rs` alpha == 1 path) precisely to avoid this; the device
always multiplies by the 1x1 operand. For `f64` that is exact and infinities
survive. For `Complex64` the device's complex product evaluates a difference of
products that is `inf - inf` for `(inf, 0)` and for `(0, -inf)`, so **both**
components come back `NaN`: the complex product's `inf * 0` term is already
`NaN`, and the remaining multiply spreads it across both components. This is
stronger than the design's prediction that such a payload would merely "gain
NaN components". Finite payloads, and NaN
propagation through a zero coefficient, are unaffected; this is recorded, not
worked around, because hiding it would mean reintroducing a descriptor alpha
and losing NaN propagation everywhere else.

## Transfer contract, stated exactly

`cuda_region_axpby` with a caller-owned coefficient moves nothing across the
host boundary, ever, and allocates no device buffer. The context-owned
operands are the only exception, and they are one-off: `coeff = None` uploads
the one-element `1` on first use of a dtype, and `cuda_region_zero`
additionally uploads a zero template, re-uploading it only when a fill is
longer than the resident one. `reserve_zero_template` sizes it once so a
replay pays none; `scalar_operand_bytes` reports what is pinned and
`release_scalar_operands` frees it. G2a-2 is expected to charge those bytes to
the device workspace budget and to converge this template with the G3c
`CudaZeroTemplate` (`tenet/src/typed.rs`); this leaf deliberately does not
consolidate them.

Disclosed cost: Tenferro's stream slots are per thread, so a context-owned
operand used from another thread can force a device-wide synchronize per call.
Every such call today happens under the device lease, which serializes them.

## CI coverage

`cuda-check` only `cargo check`s the `cuda` feature, so the device tests and
the adapter are compile-checked there, not executed. The region descriptor and
every layout rule it enforces therefore live in `tenet-dense/src/cuda_region.rs`,
which is compiled (and tested) without the `cuda` feature: its 7 unit tests run
in the ordinary `cargo test -p tenet-dense` that CI does execute.

## Constraints this adapter honours

Carried over from `cuda-strided-region-probe-2026-09-20.md` and enforced here
before any device work:

1. **Non-negative strides.** `CudaRegion` stores unsigned strides, so a
   negative one is not expressible rather than rejected at the contraction.
2. **Injective destination.** Checked on the host with the same
   cumulative-span rule the host proves block layouts with
   (`tenet-core` `block_structure.rs::block_layout_is_proven_injective`) and
   reported as `DenseError::Unsupported`, instead of relying on Tenferro's
   view constructor. Device admission therefore equals the host's *proven*
   class; layouts the host admits only through its exact overlap fallback are
   `Unsupported` on device.
3. **No in-place transform.** `CudaDenseStorage` is neither `Clone` nor
   refcounted, so a shared and an exclusive borrow of one buffer cannot
   coexist; source, coefficient and destination are necessarily distinct.
4. **Zero-extent regions** return `Ok` with no submission.
5. **No rank limit** is imposed: the probe accepted every mode count to 72.
6. **`beta` ∈ {0, 1}** is a two-variant enum, not a validated scalar: 0.5.0 has
   no in-place strided scale.
