# Single-precision device factorizations — what C4 opened and what stayed closed

Authority: TeNeT `91d15f97` (`origin/main`, #1340), issue
[#1341](https://github.com/Ryo-wtnb11/TeNeT/issues/1341), plan
[#1065](https://github.com/Ryo-wtnb11/TeNeT/issues/1065), the adapter leaf
`benchmarks/history/cuda-scalar-single-precision-2026-09-21.md` (#1326, C1),
the device payload leaf `docs/audit/issue-1336-single-precision-device-payload.md`
(C2), and the host factorization leaf
`docs/audit/issue-1324-single-precision-factorizations.md`.

Reference evidence: this leaf is dtype admission with no algorithm change. It
inherits the reference record of #1324 (TensorKit `cfaa073`, MatrixAlgebraKit
0.6.9 tolerance rules) and of the #1326 adapter leaf; QSpace has no
corresponding path, because it is double-precision only.

This artifact is revision-pinned evidence, not current capability authority:
`tenet/src/typed.rs`, `tenet/tests/typed_cuda_single_precision_factorizations.rs`
and `tenet-dense/src/cuda_adapter.rs` are. It supersedes the admission table of
the #1336 artifact.

## Admission table — device families by payload dtype

| Device family | Gate | `f64` | `Complex64` | `f32` | `Complex32` |
| --- | --- | --- | --- | --- | --- |
| `to_cuda` / `to_host` | `CudaPayload` | yes | yes | yes (C2) | yes (C2) |
| `adjoint` (lazy), `scale`, `add`, `zeros_like`, `normalize` | `CudaPayload` | yes | yes | yes (C2) | yes (C2) |
| `norm`, `inner`, `dot` | `CudaPayload` | yes | yes | yes (C2) | yes (C2) |
| `contract`, `contract_ordered`, `compose`, `contract_overwrite_into[_with_template]` | `CudaPayload` | yes | yes | yes (C2) | yes (C2) |
| `permute`, `braid`, `transpose`, `transpose_axes`, `repartition` and their `*_overwrite_into` forms | `CudaPayload` | yes | yes | yes (C2) | yes (C2) |
| `tensor!` / `Network` device replay, workspace reuse, zero template | `CudaPayload` | yes | yes | yes (C2) | yes (C2) |
| `svd_compact`, `eigh_full` | `CudaFactorizationPayload` | yes | yes | **yes (C4)** | **yes (C4)** |
| `qr_compact` | `CudaQrPayload` | yes | no (#1271) | **yes (C4)** | no (tenferro-rs#1833) |
| `svd_trunc`, `eigh_trunc` | `CudaFactorizationPayload` | `UnsupportedOnDevice` (#1297) | same | same | same |
| everything else (`eig`, `inv`, `solve`, matrix functions) | — | not a device operation | | | |

## Gates opened, and the mechanism

| Site | Change |
| --- | --- |
| `tenet/src/typed.rs` `CudaFactorizationPayload` | `impl` for `f32` / `Complex32` |
| `tenet/src/typed.rs` `CudaQrPayload` | new marker; `impl` for `f64` / `f32` |
| `tenet/src/typed.rs` device `qr_compact` impl block | concrete `f64` → generic `D: CudaQrPayload` |

Nothing else in the device factorization block changed. Every helper it calls
was already written `D: CudaPayload` and every adapter entry point already
dispatched on the payload dtype, so the f64-specific assumptions were a short,
enumerable list (below) rather than a rewrite.

### Device QR: one authority for the capability

Device QR is *narrower* than the device factorization family, and the reason is
not precision. The backend's positive-diagonal gauge runs a `triu` kernel whose
zero constant Tenferro 0.5.0 materialized as `E::cast_from(0u32)`
(`tenferro-gpu-0.5.0/src/kernels/helpers.rs:84`), which NVRTC could not construct
for `cuFloatComplex` (tenferro-rs#1833) or `cuDoubleComplex` (#1271). Tenferro
0.6.0 compiles it; TeNeT keeps the gate until #1271 verifies the complex path.
It is a
**dtype capability**, and the adapter already owns it as
`CudaScalar::DEVICE_CONSTANT_KERNELS`, enforced before any device work by
`ensure_device_constant_kernels` (`tenet-dense/src/cuda_adapter.rs:1145`).

TeNeT must express that capability one level higher, as a *bound*, because the
#1271 `compile_fail` doctests pin a complex device QR as a compile-time
boundary and a runtime error would silently un-pin them. Stable Rust has no
`where D::CONST == true`, so `CudaQrPayload` is a hand-written projection of the
constant, held equal to it by a `const` assertion beside the marker:

```rust
const _: () = {
    use tenet_dense::CudaScalar;
    assert!(
        <f64 as CudaScalar>::DEVICE_CONSTANT_KERNELS
            && <f32 as CudaScalar>::DEVICE_CONSTANT_KERNELS,
        "a real payload lost its device constant kernels: drop its `CudaQrPayload` impl"
    );
    assert!(
        !<num_complex::Complex64 as CudaScalar>::DEVICE_CONSTANT_KERNELS
            && !<num_complex::Complex32 as CudaScalar>::DEVICE_CONSTANT_KERNELS,
        "a complex payload gained device constant kernels: give it a `CudaQrPayload` impl"
    );
};
```

That block ties the constant to literals, not to the marker's impl set, so it
alone would let an extra `CudaQrPayload` impl for a dtype whose constant is
`false` compile. The device `qr_compact` body therefore also asserts the
constant for its own `D`:

```rust
const {
    assert!(
        <D as tenet_dense::CudaScalar>::DEVICE_CONSTANT_KERNELS,
        "`CudaQrPayload` admits a dtype without device constant kernels: drop its impl"
    );
}
```

An inline `const` naming a generic parameter is evaluated at monomorphization,
so such an impl fails the first build that instantiates device QR at it (the
`f32_cuda_qr` doctest instantiates it at `f32`). The two checks cover opposite
directions: the block fires on an upstream flip of the constant without any
instantiation; the inline `const` fires on a new impl the block cannot see.

The constant remains the only place the capability is *decided*; the marker
cannot drift from it, because a drift fails the build with a message naming the
repair. `device_qr_admission_follows_the_adapter_capability_constant`
(`tenet/tests/typed_cuda_single_precision.rs`, ungated) states the same
equality from the test suite.

## f64-specific assumptions found in the device factorization block

The block was audited line by line for anything that read `f64` rather than the
payload. Four sites, three of them already correct:

| Site | Verdict |
| --- | --- |
| device `qr_compact` output buffers `vec![0.0; len]` | **fixed** → `vec![D::ZERO; len]` |
| device `qr_compact` `cuda_qr_region::<f64>` | **fixed** → `cuda_qr_region::<D>` |
| spectra `SectorSpectrum<f64>` from `cuda_svd_region` / `cuda_eigh_region` | already correct: the adapter downloads through `download_values::<D::Real>`, whose `CudaRealScalar::widen` is exact, so a `f32` spectrum reaches the host as the `f64` TeNeT decides in (#1326) |
| `fill_diagonal_values` writing `D::from_real(value)` | already correct: `FactorScalar::from_real` narrows per dtype (`tenet-matrixalgebra/src/factorize.rs:216` for `f32`, `:323` for `Complex32`) |
| the Hermitian admission tolerance | already correct: `HERMITIAN_TOLERANCE_EPSILONS * <D::Real as CudaRealScalar>::EPSILON` since C1 — the *reason* an `f32` block is admitted at all |
| the `#[cfg(test)]` observation hooks | already correct: they count *calls* (decompositions, copies, selector uploads, assembly GEMMs, output uploads), never bytes, so a 4-byte element changes nothing they record |
| the `#1320` alignment guard `permute_operand_offset_is_aligned(offset, size_of::<D>())` | already correct **and now load-bearing at a new set of offsets**: aligned iff `offset * size_of::<D>() ≡ 0 (mod 256)`, so `f32` needs `offset ≡ 0 (mod 64)` where `f64` needed `≡ 0 (mod 32)` |

There is no host-side threshold, no `as f64` narrowing and no absolute constant
anywhere in the block.

## `compile_fail` pins

| Pin | Before (#1336) | After (#1341) |
| --- | --- | --- |
| `D: CudaPayload` cannot reach `svd_compact` | `compile_fail` | unchanged |
| `D: CudaPayload + FactorizationScalar` cannot reach `svd_compact` | `compile_fail` | unchanged |
| `f32` device `svd_compact` | `compile_fail` | **compiles** |
| `Complex32` device `eigh_full` | `compile_fail` | **compiles** |
| `f32` device `qr_compact` | `compile_fail` | **compiles** |
| on `FactorizationScalar` — `f32` device `svd_compact` | `compile_fail` | **compiles**; the pin there is now `Complex32` device `qr_compact` |
| `Complex64` device `qr_compact` | `compile_fail` (#1271) | unchanged — and now fails on the `CudaQrPayload` bound rather than on a concrete `f64` impl block |
| `Complex32` device `qr_compact` | — | **new** `compile_fail` |
| `D: CudaFactorizationPayload` cannot reach `qr_compact` | — | **new** `compile_fail`, with `D: CudaQrPayload` as its twin |
| Checked-Generic providers have no device factorization | `compile_fail` | unchanged |
| `svd_trunc` at `f32` compiles and fails at runtime | — | **new** compiling example: #1297 is a capability error, not a bound |

The positive half is the ungated
`the_device_admission_markers_hold_exactly_where_the_table_says`.

## Evidence

| Gate | File |
| --- | --- |
| Device SVD / QR / EIGH at every admitted payload: structure, spectrum, isometry, reconstruction, positive-diagonal gauge, `\|lambda\|` ordering, Hermitian admission, rejection order, #1320 offsets, #1297 composition, cost contract | `tenet/tests/typed_cuda_single_precision_factorizations.rs` |
| Base family, reductions, transforms, overflow and overwrite contracts | `tenet/tests/typed_cuda_single_precision.rs` |
| Device network chains at U(1) and SU(2), warm cost at both lanes | `tenet-network/tests/typed_cuda_network.rs` |
| Adapter-level four-dtype device evidence | `tenet-dense/tests/cuda_scalar_dtypes.rs` (#1326) |
| Host single-precision factorization semantics | `tenet/tests/single_precision_factorizations.rs` (#1324) |
| Device run | `benchmarks/history/cuda-typed-single-precision-factorizations-2026-09-21.md` |

The oracle is always the **host** factorization of the same tensor at the same
payload dtype, read through gauge-independent identities: the device SVD and
`eigh` keep the raw cuSOLVER gauge, so `u`, `vh` and the eigenvectors are never
compared pointwise. `qr_compact` *is* gauge-fixed on device (positive
diagonal), so its factors are. Conditioning enters as
`kappa = sigma_max / sigma_min` measured from the double-precision twin of the
same fixture, exactly as the host suite measures it; `eps` is the payload's
own, so the `f64` instantiation of every generic body is a real control.

Cost contracts are relative and same-process: the single-precision run of a
fixture is compared against the double-precision run of the same fixture in the
same test binary. Call counts must be equal and transferred bytes exactly
halved; no absolute platform constant appears.

## Residuals

* **A compact-diagonal device receiver is unreachable** through the public API:
  `to_cuda` densifies a `TypedData::Diagonal` payload on upload
  (`tenet/src/typed.rs` `to_cuda`), so the `Diagonal` arm of
  `direct_cuda_storage` is defensive and has no public reproduction. Only the
  lazy-adjoint half of the "compact and lazy receivers" rejection order is
  testable here; the compact half is pinned on the host.
* **`Complex32` / `Complex64` device QR** stay closed pending #1271. The
  compile defect (tenferro-rs#1833) is fixed in Tenferro 0.6.0, which TeNeT
  adopted in #1370; lifting the gate needs its own device evidence.
* **`tenet-network/examples/cuda_operation_matrix.rs`** gained `f32` and
  `Complex32` rows *without* the base/factorization harness split #1336
  expected: that split existed only to work around `f32` not being a
  `CudaFactorizationPayload`, which this leaf removed. The rows sit behind a
  new `--precision double|single|all` flag whose default is `double`, so the
  pinned baseline CSVs keep their rows, their order and their `check` column.
  The flag exists because the harness builds a fresh `Runtime` per row and
  three lanes in one process exhaust cuTENSOR handles (`cutensorCreate` status
  14 after ~1900 rows on an A100); that retention is a pre-existing property of
  the harness, is not fixed here, and is recorded in
  `benchmarks/history/cuda-typed-single-precision-factorizations-2026-09-21.md`.
* **Persistence** (`WireScalar`) and the host advanced-linalg family stay
  closed for single precision; leaves H8 and H5, unchanged here.
* **Carried to a later device leaf** from the independent review (P2-2, P2-3),
  neither a C4 defect:
  * the #1320 offset test has no *nonzero* 64-element-aligned `f32` factor
    offset (e.g. `d0 = 8`, offset 64: aligned for `f32`, unaligned for
    `f64`/`Complex32`/`Complex64`); its `(4,2)`/`(2,2)` "controls" are in fact
    unaligned for three or all four dtypes;
  * device `qr_compact` has no `f32`-vs-`f64` call/byte cost contract; only
    SVD/EIGH are in the measured pair.
* **Fresh-`Runtime` churn** exhausting cuTENSOR (`cutensorCreate` status 14) is
  Ryo-wtnb11/TeNeT#1343.
* **`normalize` of an overflowed single-precision device norm** returns zeros
  with no error; that is a known limitation tracked in Ryo-wtnb11/TeNeT#1344,
  characterised (not endorsed) by
  `device_single_precision_normalize_of_an_overflowed_norm_is_all_zero`. *Resolved by #1344: the
  device `norm` accumulates in `f64`; that test was replaced by
  `device_norm_and_normalize_match_the_host_where_a_payload_sum_would_overflow_or_underflow`.*
