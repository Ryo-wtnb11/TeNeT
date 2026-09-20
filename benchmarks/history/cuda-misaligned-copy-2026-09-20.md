# cudaErrorMisalignedAddress in the aligned whole-factor copy (#1320)

Date: 2026-09-20/21. Host: qg1, NVIDIA A100 (GPU 0, exclusive — no other
compute process on any GPU at run time). CUDA 12.6, cuTENSOR 2.5.0,
tenferro 0.5.0 (crates.io, pinned). Build: `--no-default-features --features
cuda,cpu-faer`, debug profile, `--test-threads=1`, private target dir
(`/data2/ryo-w/gpu-phase/fix1320-target`, removed afterwards).

Base: `origin/main` `f7d324fe` (rebased from `89b1cde6` after #1319 changed
`cuda_region_axpby`'s signature). Fix: `fa54d9ab`.

## 1. Defect

Device `svd_compact` / `qr_compact` take an aligned whole-factor route that
copies a compact factor straight into its sector region with
`cuda_copy_region_into` → Tenferro `copy_read_into`. When the target region
starts at an element offset whose byte product is not a multiple of 256, the
kernel launch fails with `cudaErrorMisalignedAddress`.

The failure is asynchronous. `copy_read_into` returns `Ok`; the error surfaces
at the next synchronizing call, which in the original report was
`cuda_download`. The CUDA error is sticky: the context is unusable afterwards,
so the Tenferro-level probe lives in its own test binary
(`tenet-dense/tests/cuda_misaligned_copy_probe.rs`) with exactly one test and
stops its sweep at the first failure.

## 2. Trigger matrix

Destination offsets only. The source view of `cuda_copy_region_into` is always
compact at element offset 0, so its pointer is the base allocation's and is
256-byte aligned by construction; Tenferro rejects a non-compact or offset
source outright (`permutation.rs:444`).

Measured through the typed layer, U(1) square tensor on leg
`[(0, d0), (1, d1)]`, f64 — the second sector's block starts at element
`d0 * d0`:

| `(d0, d1)` | dst offset | block | byte offset | 0.5.0 `copy_read_into` |
| --- | --- | --- | --- | --- |
| (2,2) | 4  | 2x2 | 32  | ok |
| (3,2) | 9  | 2x2 | 72  | FAIL |
| (3,3) | 9  | 3x3 | 72  | ok |
| (4,2) | 16 | 2x2 | 128 | ok |
| (5,2) | 25 | 2x2 | 200 | FAIL |
| (1,1) | 1  | 1x1 | 8   | ok |

Observed pattern: it faults when the shifted pointer is 8-byte but not
16-byte aligned *and* the extent lets cuTENSOR vectorize by two f64 — i.e. the
kernel selection, not the pointer alone, decides whether the lie is fatal.
Leading dimension and dtype do not change the rule, only which element offsets
are byte-unaligned (f64: multiples of 32 are safe; Complex64: multiples of 16).

Device evidence for the raw Tenferro behaviour:
`tenet-dense/tests/cuda_misaligned_copy_probe.rs` (see §6).

## 3. Root cause

`tenferro-gpu-0.5.0/src/cubecl/permutation.rs`:

- `resolve_prepared_device_region` (line 726) folds the view's element offset
  into the operand pointer (lines 743–752) but returns
  `alignment: CUDA_ALLOCATION_ALIGNMENT` (256) unconditionally (line 755).
- `copy_view_into` (lines 480–481, comment 493–498) passes those resolved
  values as the cuTENSOR *descriptor* alignment requirement for both operands.

cuTENSOR selects a vectorized kernel from the advertised alignment, so a
descriptor that claims 256 for a pointer that is only 8-byte aligned produces
a launch the hardware cannot execute.

Sibling paths are already truthful and show why the workaround works:

- `permutation.rs:397` (`to_contiguous_view`'s input operand) uses
  `view_descriptor_alignment_requirement::<T>()` (= `size_of::<T>()`).
- `gemm.rs:630` / `gemm.rs:677` return the same per-element value for every
  `View(_)` operand of a contraction. `dot_general_read_into_accum` — and
  therefore `cuda_region_axpby` — is safe at arbitrary offsets.

Upstream status: fixed on tenferro-rs `main` by `25379dd`
(tensor4all/tenferro-rs#1836), which added `resolved_operand` /
`device_address_alignment`. Verified against `origin/main`
`f6c20eb2c1451c62dccd5dd7b7eece2fdd24de88` (fetched 2026-09-20). The published
0.5.0 release does not contain it. Draft report / release request:
`reviews/gpu-phase-20260920/tenferro-issue-misaligned-copy.md`.

## 4. Fix

`tenet-dense/src/cuda_adapter.rs` `cuda_copy_region_into` — the single seam
every offset copy in TeNeT routes through — branches on the destination offset:

- byte offset a multiple of 256 → `copy_read_into` (`cutensorPermute`),
  unchanged;
- otherwise → `cuda_region_axpby` with `alpha = 1` and
  `CudaRegionCoefficient::One` (the context's shared `1`),
  `CudaRegionBeta::Overwrite`, source `CudaRegion::packed([rows, cols], 0)`,
  destination `CudaRegion::new([rows, cols], [1, dst_ld], dst_offset)`.

Guard predicate: `cuda_region.rs` `permute_operand_offset_is_aligned(offset,
element_bytes)`, with `CUTENSOR_PERMUTE_DESCRIPTOR_ALIGNMENT = 256`. It lives
in the CPU-testable layout module and has its own unit tests.

### Proof of the guard

The condition is the descriptor contract itself, not the kernel heuristic:

1. Tenferro advertises `CUDA_ALLOCATION_ALIGNMENT` = 256 bytes for both
   permutation operands of `copy_view_into`.
2. The base address satisfies 256. CubeCL's CUDA runtime reports
   `mem_alignment = 512` (`t4a-cubecl-cuda-0.10.0/src/runtime.rs:76`) and its
   memory pool pads every slice start to that alignment
   (`t4a-cubecl-runtime-0.10.0` `memory_pool/memory_page.rs:125-146`,
   `memory_management/memory_manage.rs:226` and `:253`), over pages obtained
   from `cuMemAllocAsync`/`cudaMalloc`. Nothing checks this at runtime, in
   TeNeT or in Tenferro: it is exactly the assumption
   `resolve_owned_operand` (`permutation.rs:680`) already makes for every
   owned operand, so the guard is no weaker than the offset-0 path that has
   been in production since G1b.
3. The operand pointer is `base + offset * size_of::<D>()`. Its alignment is
   ≥ 256 iff `offset * size_of::<D>() ≡ 0 (mod 256)`.
4. So the advertisement is truthful exactly on that set, and false everywhere
   else. The guard admits exactly the truthful set.

Anything narrower — "odd offset with even extent", or a 16-byte rule — would
depend on which vector width cuTENSOR picks for a given extent and dtype, i.e.
on a kernel-selection heuristic that a cuTENSOR update may change. An
offset-overflowing product is treated as unsafe rather than wrapped.

### Route choice and cost

Guarded, not unconditional, on structural grounds: offset 0 is provably
truthful and is the *only* offset the zero-template reset ever uses
(`cuda_zero_prefix`, the G3c destination reset), which is the hottest copy in
a warm `tensor!` replay. Keeping it on the permutation route preserves the
property its doc records — a packed prefix needs no entry of the contraction
plan LRU the caller's GEMMs share.

Cost difference per guarded call: one entry of the cuTENSOR contraction plan
cache (keyed by the region shape) instead of one entry of the permutation plan
cache; one `dot_general` submission instead of one `cutensorPermute`. Both move
`rows * cols` elements in one submission.

Allocation: steady state is allocation-free on both routes, but the *first*
region-route call per (context, dtype) uploads that context's one-element `1`
operand — one H2D call and one device allocation, counted as `h2d_calls` and
`device_allocs`. It is a one-off per context and dtype, shared with every
other `cuda_region_axpby`/`cuda_region_zero` user, and is already part of the
region primitive's documented transfer contract.

Errors: for valid input the routes are equivalent. For *invalid* input they
are not interchangeable — the region route validates through
`cuda_region_axpby`, so a dtype or device mismatch, an out-of-bounds region or
a non-injective destination is reported with `op = "cuda_region_axpby"` (and a
dtype mismatch becomes `DenseError::DTypeMismatch` rather than a
`"cuda_region"` backend error). The variant and `op` a caller observes
therefore depend on the destination offset. Both routes reject the same
inputs, no caller branches on either, and no test pins them; this is disclosed
on `cuda_copy_region_into`.

Disclosed deviation: `cuda_region_axpby` multiplies by a 1x1 `1` operand
rather than copying bits, and a Complex64 `±inf` payload comes back as `NaN`
(`benchmarks/history/cuda-region-axpby-2026-09-20.md`). Irrelevant at these
sites: the only caller is `copy_whole_factor`, whose source is a factor
cuSOLVER produced from a finite input; a non-finite input already fails the
factorization. Finite Complex64 values are bit-exact through a multiply by
`(1, 0)`, and f64 infinities survive unchanged.

## 5. Affected sites (authority graph)

Every TeNeT use of a Tenferro copy/permute with a possibly-offset view:

| Site | Offsets | Classification |
| --- | --- | --- |
| `cuda_adapter.rs cuda_copy_region_into` | destination arbitrary; source always 0 | **affected** — fixed here |
| `typed.rs copy_whole_factor` (svd/qr aligned whole-factor route) | passes `target.range().start`, any block offset | **affected** — routes through the seam above, no caller change |
| `typed.rs assemble_right_factor` / per-tree QR–SVD assembly | `cuda_gemm_region_into` | unaffected — contraction path, truthful view alignment (`gemm.rs:630/677`) |
| `cuda_adapter.rs cuda_zero_prefix` (+ `CudaZeroTemplate::reset_prefix`, G3c reset / zero template) | both operands at element offset 0 | unaffected — the 256-byte promise holds at offset 0 |
| `cuda_adapter.rs cuda_region_axpby` / `cuda_region_zero`, `tenet-operations/src/cuda_transform.rs` replay | arbitrary | unaffected — contraction path; this is the safe route the fix uses |
| `cuda_svd_region` / `cuda_qr_region` / `cuda_eigh_region` | offset source views | unaffected — cuSOLVER through `tenferro-linalg`, not `copy_view_into` |
| `cuda_matmul_region_into`, `cuda_gemm_region_*` | arbitrary | unaffected — contraction path |
| `cuda_is_hermitian_region` | offset source view | unaffected — it materializes through `to_contiguous_view`, whose input descriptor advertises the truthful `size_of::<T>()` (`permutation.rs:397`), not because it is a contraction |

`cuda_copy_region_into` and `cuda_zero_prefix` are the only two TeNeT call
sites of `copy_read_into` outside tests.

## 6. Counters

`copy_calls` still counts one per `cuda_copy_region_into` that moves data, on
either route — its documented meaning is unchanged. The region route
additionally counts one `gemm_calls`, because that is the submission it makes;
this is stated in the `CudaTransferStats` doc and asserted by
`cuda_copy_region_alignment.rs`.

No pinned counter in an existing test changed. The two that could have:

- `typed_cuda_network.rs` warm-chain gate `(h2d, device_allocs, copy_calls,
  d2h, gemm_calls) = (1, 1, 2, 0, 6)` — its two copies are zero-template
  resets at offset 0, which keep the permutation route.
- the QR/SVD observation gates count `observe_cuda_factor_copy`, a TeNeT-level
  count of `copy_whole_factor` calls, which the routing does not change.

## 7. Tests

- `tenet-dense/src/cuda_region.rs` (CPU, always run):
  `permute_offsets_are_admitted_by_the_descriptor_promise`,
  `permute_offset_overflow_is_not_admitted`.
- `tenet-dense/tests/cuda_copy_region_alignment.rs` (device, `#[ignore]`):
  offsets `{0,1,2,3,9,25,k-1,k,k+1,2k}` × blocks
  `{1x1,2x2,3x3,2x3,3x2,4x1,1x4}` × `ld ∈ {rows, rows+2}`, f64 and Complex64;
  bitwise against a host scatter, plus the route each offset must take.
- `tenet/tests/typed_cuda_transfer.rs`
  `typed_cuda_factorizations_handle_blocks_at_unaligned_offsets` (device):
  `svd_compact` and `qr_compact` (f64) and `svd_compact` (Complex64; device QR
  for c64 is unsupported, #1271) over `(d0,d1) ∈ {(3,2),(5,2),(3,3),(4,2),
  (2,2)}`, compared with Host through the existing reconstruction and
  gauge-invariant assertions.
- `tenet-dense/tests/cuda_misaligned_copy_probe.rs` (device, own binary):
  records raw Tenferro `copy_read_into` behaviour per destination offset. It
  records rather than gates — once TeNeT pins a tenferro release containing
  #1836 it will simply pass everywhere.

## 8. Device results

Tenferro-level reproduction, `cuda_misaligned_copy_probe`, verbatim:

```
copy_read_into f64 dst_offset 0 (0 bytes) 2x2: ok
copy_read_into f64 dst_offset 4 (32 bytes) 2x2: ok
copy_read_into f64 dst_offset 9 (72 bytes) 2x2: FAILED: download: cubecl_runtime_synchronize: backend failure: RuntimeError(cudaErrorMisalignedAddress, "misaligned address")
context poisoned after the first failure; stopping the sweep
tenferro-gpu: failed to synchronize CUDA runtime during Drop: cuda_runtime_drop: backend failure: RuntimeError(cudaErrorMisalignedAddress, "misaligned address")
```

This confirms both the defect and the stickiness: the error is raised at the
`download`, not at `copy_read_into`, and it still fires inside the runtime's
`Drop`. Offset 4 (32 bytes) passing while offset 9 (72 bytes) faults is exactly
why the guard is the descriptor contract and not the observed fault set — the
guard is conservative, refusing some offsets that happen to survive today's
kernel selection.

Full device suite, `--no-default-features --features cuda,cpu-faer
--no-fail-fast -- --ignored --skip measure_checked_generic_transform_phases
--skip axioms_ --skip itebd_ --skip cross_library --test-threads=1`:

```
binaries: 107 passed: 131 failed: 0 ignored: 0
```

Per-binary, verbatim:

```
cuda_copy_region_alignment: test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.98s
cuda_misaligned_copy_probe: test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 1.18s
typed_cuda_transfer:        test result: ok. 20 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 9.06s
cuda_region_axpby:          test result: ok. 17 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 4.30s
cuda_strided_region_probe:  test result: ok. 10 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 2.88s
```

The pre-fix base was not re-run to failure on device: the Tenferro-level probe
above establishes the defect directly, and the typed reproduction is recorded
in #1320.
