# Device tree-transform executor, Single blocks — A100 record (2026-09-20)

Leaf G2a-2, issue #1304. Design: `reviews/gpu-phase-20260920/g2-design.md` §8
(authoritative revision after the independent design review). Primitive:
`cuda_region_axpby` / `cuda_region_zero` (#1301/#1302).

## Environment

- Host `qg1`, `CUDA_VISIBLE_DEVICES=1` (NVIDIA A100), CUDA 12.6, cuTENSOR 2.5.0
  via `TENFERRO_CUTENSOR_PATH`.
- Source `/data2/ryo-w/gpu-phase/g2a2` at branch `g2a2-single-block-executor`
  over `origin/main 98ebb529`; shared `CARGO_TARGET_DIR=/data2/ryo-w/gpu-phase/gl2-target`.
- Build: `--no-default-features --features cuda,cpu-faer`, `dev` profile
  (correctness evidence only; this leaf reports no timing).

## Correctness

New executor suite (`tenet-operations/tests/cuda_tree_transform.rs`), all
`#[ignore]`d device tests, `--test-threads=1`:

```
running 11 tests
test a_coefficient_of_one_moves_f64_payloads_bitwise ... ok
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

test result: ok. 11 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 3.46s
```

Oracles, independent of the structure under test:

1. An explicit index walk over each fixture's own block metadata
   (`tests/common/mod.rs`), which never reads a `TreeTransformStructure`, its
   baked fused layouts or any executor state. That walk is itself pinned
   against the **host** executor by `tests/tree_transform_device_oracle.rs`,
   which needs no GPU and therefore runs in CI (5 tests, including a negative
   control proving the oracle depends on the coefficient and on the axis map,
   and a host replay of the layout the device reports as unsupported).
2. The host executor replaying the same compiled structure, compared element
   for element with the device result.

Covered: rank 2–6 permutes/transposes, fermionic signs (coefficient −1),
coefficients 0.625/0.75/2.5, conjugated sources (`storage_conjugate`), f64 and
Complex64 payloads, interleaved multi-block destination layouts (gapped strides
into a wider parent), inactive destination layouts, zero-extent blocks, both
destination modes. Tolerance 1e-12 in general, because the device multiplies
by the coefficient operand where the host copies (the deviation recorded in
`cuda-region-axpby-2026-09-20.md` / §8 P1-3); for the fixtures whose every
coefficient is exactly 1, f64 device results are asserted **bitwise** against
both the host and the oracle.

## Contracts measured on device

| Contract | Observation |
|---|---|
| Warm replay traffic | `h2d_calls = 0`, `d2h_calls = 0`, `device_allocs = 0`; submissions = 1 per active block + 1 per inactive destination layout (3 for the fixture) |
| Coefficient uploads | 2 structures × 3 alternating rounds → `h2d_calls = 2` total |
| Plan cache | 70 distinct fused signatures: cap raised from Tenferro's default 64 to ≥ 70; warm replay adds 0 `evictions` and 0 `misses` |
| Rejection order | `Axpby(β ∈ {0, 2, −1})`, a Multi-block structure, and a destination layout only the host's exact overlap fallback admits (dims [3,2] strides [2,3]) → typed errors with **all eight** boundary counters and the whole plan-cache snapshot unchanged, no cache entry created, and the destination's bytes byte-for-byte unchanged |
| Plan-cap negative control | the same 70-signature structure on a fresh context with plan budget 0: the bound stays at Tenferro's 64 and the warm replay's `evictions` grow |
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
passed=107 failed=0
```

(`/data2/ryo-w/gpu-phase/g2a2-device4.log`, the run after the review repairs.)
No process left on the GPU.

## G3c reset: one authority, unchanged kernel and unchanged counters

`CudaZeroTemplate` (G3c) no longer owns a device buffer — the single zero
template is the `CudaDenseContext`'s, shared with the region primitive, and the
workspace keeps charging the length it reserved to `workspace_budget_bytes`.
The **kernel is unchanged**: `cuda_zero_prefix` copies the template's packed
prefix with `copy_read_into` (`cutensorPermute`, its own plan cache, no
multiply and no `1` operand), which is what the workspace always did. An
earlier revision of this leaf routed the reset through `cuda_region_zero`
(`cutensorContract` against the 1×1 operand); that was reverted after the
independent source review, because it added a multiply per element, one H2D
for the `1` operand and one contraction-plan entry per destination length in
the 64-entry LRU that G3c's GEMMs share, for no G3c need.

`typed_cuda_network.rs::warm_cuda_chain_uploads_nothing_for_its_reused_destinations`
therefore keeps its original tuples, verified on device:

- first overwriting call `(h2d_calls, copy_calls) = (2, 2)`;
- steady state `(h2d, device_allocs, copies, d2h, gemm) = (1, 1, 2, 0, 6)`.

`the_device_zero_template_is_charged_to_the_workspace_budget` passes
unmodified.

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
  structures, capped by 8 MiB / an estimated 14 KB per plan ≈ 585 entries.
  Tenferro's contraction plan key is (dtype, the three operand layouts, their
  alignments, the operand operators, workspace preference): view alignment is
  the constant `size_of::<D>()` so offsets do not multiply keys, the
  accumulation scalars are *not* in the key, and conjugation is one value per
  structure. The per-structure count is therefore exact and the per-executor
  sum over-counts shared signatures — the safe direction for a cap.
- Host work per replay: the region descriptors are built once, when the
  structure is prepared, so the loop only submits them; each submission still
  builds its own Tenferro view metadata and config vectors inside the
  primitive.

## Residual (carried to later leaves)

- **P2-2**: the context zero template now outlives every workspace and has no
  production release path, so `workspace_budget_bytes` under-charges after a
  cache clear, a budget rejection or a workspace drop — bounded by one
  largest-destination buffer per dtype. Needs a release on the last workspace
  drop, or `scalar_operand_bytes` reported in runtime stats.
- **P2-3**: `retained_device_bytes` is an accessor nobody charges yet; #1304's
  "charged to the device workspace budget" stays PARTIAL until G2b wires the
  executor into the runtime's device state.
- **P2-6**: `StructureCache` bounds bytes but not entry count, does not charge
  its host-side descriptor vectors, and does a linear scan per replay; a map
  keyed on `(Weak::as_ptr, TypeId, context)` plus an entry cap is the fix when
  a workload makes it matter.
- **P2-11**: every `cuda_region_zero` fill reads a compact offset-0 source, so
  the inactive-layout fills could also leave the contraction path for
  `copy_read_into`; needs a device probe for N-D strided destinations first.
- Caller `alpha` (deviation 1) remains a G2b decision; the reviewer's
  recommendation is descriptor α for α ≠ 0 and the zero template's 1×1 prefix
  as the coefficient operand for α == 0.
