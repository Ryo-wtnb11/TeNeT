# Device tree-transform executor, Single blocks — A100 record (2026-09-20)

Leaf G2a-2, issue #1304. Design: `reviews/gpu-phase-20260920/g2-design.md` §8
(authoritative revision after the independent design review). Primitive:
`cuda_region_axpby` / `cuda_region_zero` (#1301/#1302).

## Environment

- Host `qg1`, `CUDA_VISIBLE_DEVICES=1` (NVIDIA A100), CUDA 12.6, cuTENSOR 2.5.0
  via `TENFERRO_CUTENSOR_PATH`.
- Source `/data2/ryo-w/gpu-phase/g2a2` at branch `g2a2-single-block-executor`
  over `origin/main be46c9e0`; shared `CARGO_TARGET_DIR=/data2/ryo-w/gpu-phase/gl2-target`.
- Build: `--no-default-features --features cuda,cpu-faer`, `dev` profile
  (correctness evidence only; this leaf reports no timing).

## Correctness

New executor suite (`tenet-operations/tests/cuda_tree_transform.rs`), all
`#[ignore]`d device tests, `--test-threads=1`:

```
running 10 tests
test a_dropped_structure_releases_its_cached_coefficients ... ok
test a_second_context_prepares_its_own_device_state ... ok
test a_warm_replay_transfers_nothing_and_allocates_no_device_buffer ... ok
test alternating_structures_upload_their_coefficients_exactly_once_each ... ok
test both_payload_dtypes_share_one_structure_with_their_own_coefficients ... ok
test device_replay_matches_the_oracle_and_the_host_for_every_fixture ... ok
test mismatched_structures_and_lengths_are_rejected_by_admission ... ok
test more_signatures_than_the_default_plan_bound_raise_the_cap_without_thrashing ... ok
test overwrite_cleans_a_nan_poisoned_destination_including_inactive_layouts ... ok
test unsupported_modes_are_rejected_before_any_device_work ... ok

test result: ok. 10 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 3.17s
```

Oracles, independent of the structure under test:

1. An explicit index walk over each fixture's own block metadata
   (`tests/common/mod.rs`), which never reads a `TreeTransformStructure`, its
   baked fused layouts or any executor state. That walk is itself pinned
   against the **host** executor by `tests/tree_transform_device_oracle.rs`,
   which needs no GPU and therefore runs in CI (3 tests, including a negative
   control proving the oracle depends on the coefficient and on the axis map).
2. The host executor replaying the same compiled structure, compared element
   for element with the device result.

Covered: rank 2–6 permutes/transposes, fermionic signs (coefficient −1),
coefficients 0.625/0.75/2.5, conjugated sources (`storage_conjugate`), f64 and
Complex64 payloads, interleaved multi-block destination layouts (gapped strides
into a wider parent), inactive destination layouts, zero-extent blocks, both
destination modes. Tolerance 1e-12; bitwise equality is not asserted (the
device multiplies by the coefficient operand where the host copies — the
deviation recorded in `cuda-region-axpby-2026-09-20.md` §8/P1-3).

## Contracts measured on device

| Contract | Observation |
|---|---|
| Warm replay traffic | `h2d_calls = 0`, `d2h_calls = 0`, `device_allocs = 0`; submissions = 1 per active block + 1 per inactive destination layout (3 for the fixture) |
| Coefficient uploads | 2 structures × 3 alternating rounds → `h2d_calls = 2` total |
| Plan cache | 70 distinct fused signatures: cap raised from Tenferro's default 64 to ≥ 70; warm replay adds 0 `evictions` and 0 `misses` |
| Rejection order | `Axpby(β ∈ {0, 2, −1})` and a Multi-block structure → typed `UnsupportedDeviceTreeTransform` with **all eight** boundary counters and the whole plan-cache snapshot unchanged, and no cache entry created |
| Overwrite | NaN-poisoned destination comes back free of NaN; the accumulate negative control keeps its NaN |
| Cache purge | dropping the structure releases its entry on the next preparation |
| Context keying | the same structure on a second context prepares its own state (2 entries) |

## Full device suite

`cargo test --workspace --lib --tests --no-default-features --features
cuda,cpu-faer --no-fail-fast -- --ignored --skip
measure_checked_generic_transform_phases --skip axioms_ --skip itebd_ --skip
cross_library --test-threads=1`

```
EXIT=0
passed=86 failed=0
```

(`/data2/ryo-w/gpu-phase/g2a2-device3.log`, re-run after the final refactor;
`g2a2-device2.log` is the identical earlier run.) No process left on the GPU.

## Counter contract changed by the zero-template consolidation

`CudaZeroTemplate` (G3c) no longer owns a device buffer: the single zero
template is the `CudaDenseContext`'s, shared with the region primitive, and the
workspace keeps charging the length it reserved to `workspace_budget_bytes`.
The reset is therefore a region write (`gemm_calls`) instead of a
`cuda_copy_region_into` (`copy_calls`), and the template plus the `1` operand
are created once per context rather than once per workspace.
`typed_cuda_network.rs::warm_cuda_chain_uploads_nothing_for_its_reused_destinations`
is updated accordingly:

- first overwriting call: `(h2d_calls, copy_calls)` `(2, 2)` → `(3, 0)`;
- steady state: `(h2d, device_allocs, copies, d2h, submissions)`
  `(1, 1, 2, 0, 6)` → `(1, 1, 0, 0, 8)`.

The zero-traffic and zero-allocation halves of that contract are unchanged, and
`the_device_zero_template_is_charged_to_the_workspace_budget` passes unmodified.

## Complexity

Variables: `B` Single blocks, `n_b` elements of block `b`, `I` inactive
destination layouts, `m_i` their elements, `S` distinct fused signatures.

- Submissions: `B + I` (host: `B + I` strided passes). Elements moved:
  `Σ_b n_b + Σ_i m_i`, identical to host.
- Retained device bytes: one coefficient vector per prepared structure
  (`|coefficients| × size_of::<D>()`, bounded by a byte budget, LRU-evicted,
  purged when the structure dies) plus the context's shared operands
  (one element + the longest zeroed region ever reserved, over all callers).
- Host work per replay: none beyond the loop itself. The region descriptors
  are built once, when the structure is prepared, and a replay only submits
  them.
- Plan cache: `S` counted once per structure on the host from its baked fused
  layouts, and the backend cap raised monotonically to the sum over prepared
  structures, capped by 8 MiB / ≈14 KB per plan ≈ 585 entries. Disclosed: the
  count is a lower bound (operand alignment and the accumulation scalars are
  also part of Tenferro's plan key), which is what a monotonic raise needs.
