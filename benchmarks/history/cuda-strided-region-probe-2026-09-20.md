# CUDA strided N-D region probe (issue #1298)

Device evidence for the claim in `reviews/gpu-phase-20260920/basic-path-survey.md` §0:
released Tenferro 0.5.0 executes, on a real A100, a strided/offset rank-N source
region accumulated into a strided/offset rank-N destination region with α/β and
conjugation, expressed as `dot_general_read_into_accum` against a 1×1 operand.
Test-only leaf; no production code changed.

Test binary: `tenet-dense/tests/cuda_strided_region_probe.rs` (10 `#[ignore]`
device tests). The oracle in every item is an independent host strided loop over
the same index space; nothing is derived from a TeNeT descriptor.

## Environment

| Item | Value |
|---|---|
| Host | qg1 (`GPUA100-gleap-01`), NVIDIA A100-SXM4-40GB, driver 560.35.05, `CUDA_VISIBLE_DEVICES=0` |
| Toolkit | CUDA 12.6 (`nvcc` V12.6.85), cuTENSOR 2.5.0 via `TENFERRO_CUTENSOR_PATH` |
| Toolchain | rustc/cargo 1.96.0, debug (`test` profile) |
| TeNeT | worktree `g2p-device-probe` at `origin/main` `b46ddac4`, Cargo.lock unchanged |
| Tenferro | `tenferro-tensor`/`-gpu`/`-linalg` 0.5.0 |
| Command | `cargo test -p tenet-dense --no-default-features --features cuda,cpu-faer --test cuda_strided_region_probe -- --ignored --test-threads=1 --nocapture` |
| Result | `10 passed; 0 failed`, 3.76 s |

## Per-item results

| # | Scope item | Test | Result |
|---|---|---|---|
| 1 | rank 3–6 permuted-stride source at offset → permuted-stride destination at offset, β ∈ {0,1}, α real and complex, conj on/off, 3 blocks inside one flat buffer | `nd_strided_region_accumulation_matches_host_f64`, `..._c64` | **PASS**, bitwise for α = 1 |
| 2 | outer product (no contracting dims) with permuted output strides | `outer_product_into_permuted_region_matches_host` | **PASS** |
| 3 | diagonal read view (stride `s_i + s_j`) against a ones operand = partial trace | `diagonal_read_view_traces_block` | **PASS** |
| 4 | overlap/alias rejection, out-of-bounds, negative strides, zero-size blocks, rank limit | `overlapping_mutable_region_is_rejected`, `negative_stride_read_view_is_rejected`, `zero_extent_region_is_a_no_op_or_explicit_error`, `rank_beyond_cutensor_mode_limit_is_reported` | **PASS** (behaviour recorded below) |
| 5 | counters: transfers, plan-cache entries per distinct key, behaviour past 64 | `plan_cache_reuses_one_key_and_evicts_past_the_limit` | **PASS** |
| 6 | Complex64 `solve`/`lu` on CUDA | `complex_solve_and_lu_behaviour_is_recorded` | f64 **PASS**, Complex64 **FAILS** upstream (#1833 class) |

### Item 1 detail

Four shapes (`[2,3,4]`, `[3,2,2,3]`, `[2,2,3,2,2]`, `[2,2,2,2,2,3]`), each with a
source axis order and a *different* destination axis order, leading strides 2 and
3 respectively (so no axis is contiguous with the allocation start), three source
and three destination base offsets inside one flat buffer per side. Swept over
`conj ∈ {false,true}` × `β ∈ {0,1}` × α ∈ {1, −0.75, (0.5,−1.25) for c64}: 96
f64 and 144 Complex64 contractions.

- `α = 1` (pure copy / conj / single add) compared **bitwise**; it matched for
  every case, both dtypes. Note this is stronger than required: cuTENSOR's
  `α·x + β·y` with α = β = 1 reproduced the host's element order exactly.
- `α ≠ 1` compared at `1e-12` relative; matched.
- Untouched destination elements stayed bitwise identical (the oracle covers the
  whole buffer, not only the blocks).
- No host transfer occurs in the contraction phase: the probe's own H2D/D2H
  counter is asserted unchanged across the block loop.

This is the first device evidence for rank ≥ 3 views and for **permuted output
strides**, which upstream tests (2-D offset/ld regions only) and TeNeT's existing
A100 evidence (`cuda_gemm_region_strided_into`) did not cover.

### Item 4 detail — exact behaviour

| Case | Behaviour |
|---|---|
| Mutable destination view with a stride-0 axis | Rejected at view construction: `TypedTensor::backend_region_view_mut: mutable tensor layout may overlap physical elements; materialize a contiguous owner before requesting mutable access` |
| Read view running past the allocation | Rejected: `TypedTensor::backend_region_view: view metadata is out of borrowed-slice bounds` |
| Negative source stride | Rejected inside the contraction: `dot_general: invalid argument layout: cuTENSOR dot-general accumulation requires nonnegative view strides, got [1, -2, 1]; canonicalize the view on device first` |
| Zero-extent axis in the block | **Accepted**, destination bitwise unchanged (no-op) |
| High rank | Ranks 8, 16, 24, 32, 40, 48, 56, 63, 64, 65, 72 all **accepted** (see limitation below) |

Source/destination aliasing *inside one buffer* is not a runtime check at all:
`backend_region_view` borrows the tensor shared and `backend_region_view_mut`
borrows it exclusively, so an in-place strided transform is rejected by the Rust
borrow checker. Every returning TeNeT transform has distinct source and
destination tensors, so this is a boundary, not a blocker — but an in-place
device transform is *not* expressible through this primitive.

### Item 5 detail — counters

- Device allocations: the primitive writes into a caller-supplied destination
  view and returns `()`, so it creates no owned tensor. The only device memory it
  retains is the cuTENSOR plan cache's workspace (measured below). This probe has
  no direct read of the CubeCL allocator, so "no device allocation" is an
  argument from the API shape plus the retained-bytes number, not an allocator
  trace.
- Transfers: `Transfers { h2d: 3, d2h: 0 }` for the whole plan-cache test — the
  three operand uploads and nothing else across 96 contractions.
- Plan cache, default limit **64 entries**:
  - 16 replays of one (shape, strides) signature → `entries: 1, hits: 15,
    misses: 1, evictions: 0`, `retained_bytes: 14216` (one plan ≈ 14 KB).
  - 80 distinct signatures → `entries: 64, misses: 81, evictions: 17`,
    `retained_bytes: 899704` (≈ 900 KB for 64 plans, ≈ 14 KB each).
  - `set_cutensor_plan_cache_max_entries(256)` succeeds and is read back; it does
    not retroactively repopulate, as expected.

  So the survey's disclosed constant is confirmed: a transform whose replay
  touches more than 64 distinct block signatures rebuilds plans every pass. The
  cap is raiseable from TeNeT at ≈ 14 KB per entry, i.e. ≈ 3.6 MB for 256.

### Item 6 detail — Complex64 `solve` / `lu`

| dtype | `solve` | `lu` |
|---|---|---|
| f64 | succeeded, residual `6.661e-16` (host oracle `‖Ax − b‖`) | succeeded, `P=[3,3] L=[3,3] U=[3,3] parity=[]` |
| Complex64 | **failed** | **failed** |

Both Complex64 failures are the same NVRTC compilation error, raised from
`lu_factor`:

```
lu_factor: backend failure: The server is in an invalid state
Caused by: [Launch(A compilation error happened during launch
Caused by: [Compilation Error]
  default_program(141): error: no suitable constructor exists to convert from
  "uint32" (aka "unsigned int") to "double2"
    const cuDoubleComplex l_2 = cuDoubleComplex(uint32(1));
```

in the generated `lu_parity` kernel. This is exactly the defect class the survey
predicted (`tenferro-linalg` `gpu/kernels.rs:14-16` `one_value::<E>() =
E::cast_from(1u32)`, used by `lu_parity` `gpu/linalg.rs:406`) and it confirms the
survey's footnote that tenferro-rs#1833 — which cites only `tenferro-gpu`'s copy
of the constant — also applies to `tenferro-linalg`'s own copy. The f64 leg
succeeding in the same run establishes this is complex-only, not "no CUDA solve".

## Go / no-go for the G2 adapter leaf

**GO.** The survey's §0 route is device-verified for every layout G2 needs: rank
3–6, permuted destination strides, non-zero offsets on both sides, α/β, `lhs_conj`,
f64 and Complex64, multiple blocks per buffer, zero transfers and no extra device
buffer. Items 2 and 3 additionally clear the `otimes` and trace rows of the survey
matrix with the same primitive.

Constraints a G2 adapter must honour:

1. **Non-negative strides only.** A source region with a negative stride is
   rejected at the contraction (exact text above). Any TeNeT transform that would
   produce a reversed axis must canonicalize it into the destination's stride
   pattern instead.
2. **Destination views must be injective.** A repeated (stride-0) destination axis
   is rejected at view construction. Broadcast-style scatter is not available.
3. **No in-place transforms.** Source and destination must be different tensors
   (Rust borrow rule, not a runtime error). This matches the survey's finding that
   `beta ∉ {0,1}` in-place scaling is unsupported (W8).
4. **Zero-extent blocks are a silent no-op**, so the adapter need not special-case
   them, but must not rely on an error to detect an empty job.
5. **Plan cache**: 64 entries by default, ≈ 14 KB each. A transform replay with
   more than 64 distinct (extent, stride) signatures thrashes. The adapter should
   expose or raise the cap when the block-signature count is known; this is a
   constant, not an asymptotic, and is already recorded as Tenferro wish W9.
6. **Complex64 stays off every LU/QR-based path** (`solve`, `inv`, `exp`,
   `qr_compact`, `lq_*`, `svd_full`, null spaces) until tenferro-rs#1833 is fixed;
   f64 is unaffected.

## Limitations of this evidence

- The rank sweep past 6 pads with **unit** axes (extent 1), because 64 non-unit
  modes of extent ≥ 2 is not a realizable tensor. So "rank 72 accepted" means the
  mode count itself is not a barrier for unit-padded ranks; the highest fully
  non-unit rank exercised is 6. TeNeT's reachable ranks are far below either
  bound, so no limit was found that constrains G2.
- Device allocation is argued from the API shape and plan-cache retained bytes,
  not from an allocator counter (none is exposed by Tenferro 0.5.0).
- Timings are not reported: this is a correctness/capability probe, and the run
  is a debug build.
