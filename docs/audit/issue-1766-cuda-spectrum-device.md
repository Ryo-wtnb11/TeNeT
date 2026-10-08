# CUDA spectrum device admission — reference provenance

Authority: TeNeT `0e1280780086b29f142a7b632bb892ddfc2eb92a` (`origin/main`),
issue [#1766](https://github.com/Ryo-wtnb11/TeNeT/issues/1766). Owning code:
`tenet-dense/src/cuda_adapter/factorization.rs` (`CudaSpectrum`,
`ensure_spectra_device`, `cuda_download_spectra`,
`cuda_download_batched_spectra`, `cuda_copy_spectrum_into`).

This artifact is revision-pinned evidence, not current capability authority.

## Invariant

Every `CudaSpectrum` records the device of the context whose solver produced
it (`cuda_svd_region`, `cuda_eigh_region`, `cuda_eigh_region_batched`), as
`CudaDenseStorage` does. Each consumer rejects a spectrum, and the copy also a
`dst`, on another device than the context with the crate's existing
`ensure_cuda_device` error. The check runs before any submission, allocation
or empty-input early return. Nothing else changes: dtype, layout, op names,
transfer and allocation counts on success.

## Reference search

Nothing was ported. This is a Rust/Tenferro resource-admission guard. None of
the references has an explicit device-identity admission step to correspond
to.

| Reference | Revision | Searched path | Finding |
| --- | --- | --- | --- |
| TensorKit.jl | `cfaa073e4d1e3eb2167edcbdc3be9872f41e7d91` | `ext/TensorKitCUDAExt/cutensormap.jl:CuTensorMap` (`TensorMap{T, S, N₁, N₂, CuVector{T, CUDA.DeviceMemory}}`) | The device is implicit in the CUDA.jl buffer and the task-local device context. No identity is stored on the tensor and no admission check exists. |
| TensorKit.jl | same | `src/factorizations/matrixalgebrakit.jl:MAK.initialize_output(::typeof(svd_compact!), …)` and `MAK.initialize_output(::typeof(eigh_full!), …)` (spectra via `similar_diagonal` over the input's storage type); `src/factorizations/diagonal.jl:MAK.svd_compact!` | The spectrum is a `DiagonalTensorMap` over the same storage type as the input. Its device follows the CUDA.jl allocation, with no explicit check before a later read or copy. |
| TensorOperations.jl | 5.8.0 (registry tree `WNMzk`) | `ext/TensorOperationscuTENSORExt.jl:TO.select_backend` (the `tensoradd!`/`tensortrace!`/`tensorcontract!` methods on `CuStridedView`), `_custrided` | Backend selection dispatches on array type only. Host arrays are marshalled to the current device and there is no device-identity comparison. No factorization path exists. |
| QSpace | `d2d3d7da6a59a2e8f2cb7dc8f33e7c345af59371` (`reviews/reference-sources/QSpace-current`) | case-insensitive grep for `cuda\|gpu` over the tree | No GPU source; no corresponding path. |

## Rust-specific deviation

Julia binds an array to its device implicitly through CUDA.jl ownership and
task-local state. TeNeT passes an explicit `CudaDenseContext` for one device
ordinal (the value given to Tenferro's `CudaBackend::new`). Device buffers
therefore carry that ordinal, and every entry point compares it with the
context before work. A mismatch is a typed error, not an implicit cross-device
access.
