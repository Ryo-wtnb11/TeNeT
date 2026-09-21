# Single-precision device payloads — what C2 opened and what stayed closed

Authority: TeNeT `ab47d16c` (`origin/main`, #1339), issue
[#1336](https://github.com/Ryo-wtnb11/TeNeT/issues/1336), plan
[#1065](https://github.com/Ryo-wtnb11/TeNeT/issues/1065), survey
`reviews/gpu-phase-20260920/single-precision-survey.md` (row C2), and the
adapter leaf `benchmarks/history/cuda-scalar-single-precision-2026-09-21.md`
(#1326, C1).

This artifact is revision-pinned evidence, not current capability authority:
`tenet/src/typed.rs`, `tenet/tests/typed_cuda_single_precision.rs` and
`tenet-network/tests/typed_cuda_network.rs` are.

## Admission table — device families by payload dtype

| Device family | Gate | `f64` | `Complex64` | `f32` | `Complex32` |
| --- | --- | --- | --- | --- | --- |
| `to_cuda` / `to_host` | `CudaPayload` | yes | yes | **yes (C2)** | **yes (C2)** |
| `adjoint` (lazy), `scale`, `add`, `zeros_like`, `normalize` | `CudaPayload` | yes | yes | **yes (C2)** | **yes (C2)** |
| `norm`, `inner`, `dot` | `CudaPayload` | yes | yes | **yes (C2)** | **yes (C2)** |
| `contract`, `contract_ordered`, `compose`, `contract_overwrite_into[_with_template]` | `CudaPayload` | yes | yes | **yes (C2)** | **yes (C2)** |
| `permute`, `braid`, `transpose`, `transpose_axes`, `repartition` and their `*_overwrite_into` forms (#1339) | `CudaPayload` | yes | yes | **yes (C2)** | **yes (C2)** |
| `tensor!` / `Network` device replay, workspace reuse, zero template | `CudaPayload` | yes | yes | **yes (C2)** | **yes (C2)** |
| `svd_compact`, `eigh_full` | `CudaFactorizationPayload` | yes | yes | **no (C4)** | **no (C4)** |
| `svd_trunc`, `eigh_trunc` | `CudaFactorizationPayload` | `UnsupportedOnDevice` (#1297) | same | **no (C4)** | **no (C4)** |
| `qr_compact` | concrete `f64` payload | yes | no (#1271) | **no (C4)** | **no (C4)** |
| everything else (`eig`, `inv`, `solve`, matrix functions) | — | not a device operation | | | |

## Gates opened

| Site | Change |
| --- | --- |
| `tenet/src/typed.rs` `CudaPayload` | `impl CudaPayload for f32` / `Complex32` |

That is the whole production opening. Every device impl block of the base
family was already written `D: CudaPayload`, and the per-dtype plumbing the
survey worried about was already in place and was verified rather than assumed:

* **Execution lanes** are per dtype and lazily built since #1315
  (`tenet/src/runtime.rs` `Ctxs::f32_lane` / `c32_lane`, `Option<Box<Ctx<..>>>`
  fields), so a caller that never touches single precision still builds no
  single-precision lane.
* **Device scalar operands** are four slots keyed by dtype since #1326, not two
  keyed by `IS_COMPLEX` — pinned by the adapter test
  `every_admitted_dtype_owns_a_distinct_operand_slot`.
* **The context zero template** is per dtype
  (`CudaDenseContext::reserve_zero_template::<D>`), pinned by
  `the_zero_template_is_uploaded_once_per_dtype_and_grows_monotonically`.
* **Network workspace pools** key on `(provider, dtype, storage)`, so an `f32`
  and an `f64` network over the same topology cannot share a destination.
* **Structural coefficients stay `f64` on host** and are uploaded converted
  once per `(structure, dtype, context)`, into the executing `Ctx<D>`'s own
  workspace — unchanged from #1315, and the reason an `f32` and an `f64`
  tensor over one structure cannot read each other's converted scratch.

## Gates deliberately closed, and the mechanism

**Device factorizations.** The device `svd_compact`/`eigh_full` block was
bounded `D: CudaPayload + FactorizationScalar`. Since #1332 admitted single
precision to the *host* factorization family, both halves of that bound now
hold for `f32`/`Complex32`, so opening `CudaPayload` would have opened the
device factorizations silently — with the device SVD gauge, the device
Hermitian admission rule and the host-side truncation composition never
reviewed at single precision.

Mechanism: a new sealed marker, the narrowest one that expresses exactly this
boundary.

```rust
pub trait CudaFactorizationPayload: CudaPayload + FactorizationScalar {}
impl CudaFactorizationPayload for f64 {}
impl CudaFactorizationPayload for num_complex::Complex64 {}
```

The impl block's bound becomes `D: CudaFactorizationPayload`. It is sealed
through `CudaPayload`, which is sealed through `TensorScalar`, so no downstream
crate can widen it. Opening it is leaf C4's job.

Pinned by `compile_fail` doctests, each with a compiling twin differing only in
the bound or the dtype:

| `compile_fail` | Twin |
| --- | --- |
| `D: CudaPayload` cannot reach `svd_compact` | `D: CudaFactorizationPayload` can |
| `D: CudaPayload + FactorizationScalar` cannot reach `svd_compact` | same |
| `f32` device `svd_compact` | `f64` device `svd_compact` |
| `Complex32` device `eigh_full` | `Complex64` device `eigh_full` |
| `f32` device `qr_compact` | the block's own `f64` example |
| on `FactorizationScalar` — `f32` device `svd_compact` | `f64` device `svd_compact` |

The positive half of the same table (`f32`/`Complex32` *are* `CudaPayload`;
`f64`/`Complex64` *are* `CudaFactorizationPayload`) is the ungated test
`every_base_family_payload_is_a_device_payload`.

**`compile_fail` pins that moved.** The two pins on `TensorMap::to_cuda` that
rejected `f32`/`Complex32` uploads are now compiling examples, and the
`TensorScalar` twin they pointed at now shows a single-precision upload
alongside the double-precision one. Nothing lost a pin: what those two pins
protected has become the device factorization pins above.

## The device reduction accumulation contract

`norm`/`inner`/`dot` on `CudaStorage` have two halves with different
accumulators, and the difference is observable only at single precision.

| Half | Where | Accumulator | Why |
| --- | --- | --- | --- |
| within one coupled sector | device, inside the backend GEMM | the payload dtype | Tenferro 0.5.0 offers no widening reduction |
| across coupled sectors (quantum-dimension weighting and the total) | host, after the lease is released | `WideScalar::Wide` | the same accumulator every host reduction uses |

The device half is not TeNeT's choice to make. `dot_general_read_into_accum`
(`tenferro-gpu-0.5.0/src/cubecl/gemm.rs:726`) dispatches on the single dtype
shared by both operands and the destination, and `accum_erased` (`:743`) reads
all three as one `T`; the cuTENSOR compute descriptor is fixed per dtype with
no caller control — `CUTENSOR_COMPUTE_DESC_32F` for `f32` (`gemm.rs:101`),
`..._64F` for `f64` (`gemm.rs:134`). A widened device sum would need a second
pass over each region, which is a cost, not a free improvement. The device
reduction algorithm is therefore unchanged by this leaf.

The host half *was* changed, and is the only behavioral change here besides the
admission: `weighted_inner_cuda` now folds the downloaded per-sector scalars in
`WideScalar::Wide` and narrows once at the end, and `norm` takes its square
root from the wide accumulator before any narrowing. For `f64`/`Complex64`
`Wide = Self` and `widen`/`narrow` are the identity, so those results are
unchanged down to the emitted arithmetic. The reason is the same one #1315 gave
for the host: the *number of sectors* must not degrade a result, and a
single-precision cross-sector fold would have been a second, avoidable error
source on top of the device one.

**Consequence, documented and tested.** A `norm` that is finite on the host can
be `inf` on the device at single precision, because the within-sector sum
saturates near `3.4e38` where the host's `f64` accumulator does not. This is
reported as `inf`, not as a typed error — the same convention `f64` overflow
already has. Pinned by
`device_single_precision_norm_can_overflow_where_the_host_stays_finite`, whose
double-precision twin of the same fixture shape stays finite on both sides.

Reduction *tests* therefore use a stated device-accumulation bound,
`2 * terms * eps(real(D)) * scale`, not the host's wide-accumulator tolerance.

## Evidence

| Gate | File |
| --- | --- |
| Transfer, arithmetic, reductions, contraction/compose, `contract_overwrite_into`, transforms, `*_overwrite_into`, cost contracts, admission | `tenet/tests/typed_cuda_single_precision.rs` |
| Device network chains, warm replay, workspace reuse, warm counters | `tenet-network/tests/typed_cuda_network.rs` (the two `single_precision_*` gates) |
| Adapter-level four-dtype device evidence (regions, spectra, Hermitian rule) | `tenet-dense/tests/cuda_scalar_dtypes.rs` (#1326) |
| Host single-precision base semantics | `tenet/tests/single_precision_base.rs` (#1315) |
| Device run | `benchmarks/history/cuda-typed-single-precision-2026-09-21.md` |

Every device gate is one generic body instantiated for all four dtypes, so the
double-precision instantiation is the control. The oracle is always the host
result *at the same payload dtype*, never the `f64` host result: what these
prove is device/host agreement at the payload's own precision, on top of the
independently gated host single-precision semantics.

Cost contracts are relative and same-process: the `f32` run of a fixture is
compared against the `f64` run of the same fixture in the same test binary.
Call counts must be equal and transferred bytes exactly halved; no absolute
platform constant appears.

## Residuals

* **C3** (warm-up per admitted dtype) — nothing was added. The libraries warmed
  are per backend instance, not per dtype, and the C0 probe found no per-dtype
  NVRTC stall; a first single-precision call still pays its own CubeCL kernel
  JIT and its own scalar operands, both lazy.
* **C4** (device factorizations for single precision) — closed here by
  `CudaFactorizationPayload`, with the `compile_fail` pins above as the
  handover. `Complex32` device QR additionally inherits the tenferro-rs#1833
  `float2` constant defect, which the adapter already reports as a typed
  `Unsupported` (`DEVICE_CONSTANT_KERNELS = false` for both complex dtypes).
* **`tenet-network/examples/cuda_operation_matrix.rs`** — not extended with
  single-precision rows. Its `HarnessScalar` is bounded on the device
  factorization family, so adding `f32`/`Complex32` means splitting the harness
  into a base half and a factorization half first; that is not the mechanical
  change the leaf allowed. Its bound was re-expressed as
  `CudaFactorizationPayload` (one line) and its pinned baseline CSVs are
  untouched.
* **Persistence** (`WireScalar`) and the host advanced-linalg family stay
  closed for single precision; they are leaves H8 and H5, unchanged here.
