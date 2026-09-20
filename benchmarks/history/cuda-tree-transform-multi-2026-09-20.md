# Device tree-transform executor: recoupling (Multi) blocks — 2026-09-20

Leaf G2a-3 (issue #1310). Base `origin/main` 053d5a74, branch
`g2a3-multi-block-executor`. Design authority `reviews/gpu-phase-20260920/g2-design.md`
§8; host authority `tenet-operations/src/transform_replay.rs`.

## Environment

- A100-SXM4-40GB, driver 560.35.05, `CUDA_VISIBLE_DEVICES=1` (GPU 1 idle, no
  other user's process on it; all eight GPUs reported 1 MiB used and no compute
  apps before and after the run).
- CUDA 12.6, cuTENSOR 2.5.0, Tenferro 0.5.0 (no dependency change).
- `cargo 1.96.0` / `rustc 1.96.0`, debug profile, `--no-default-features
  --features cuda,cpu-faer`, `--test-threads=1`.
- Shared target `/data2/ryo-w/gpu-phase/gl2-target` (194G, `/data2` at 51%).

This is correctness and transfer-contract evidence. No timing was taken: the
lowering is a one-to-one port of the host's own pack → GEMM → scatter
decomposition, so its FLOP, launch and byte counts follow from the structure,
not from a measurement.

## Correspondence record

| Host (`transform_replay.rs`) | Device |
|---|---|
| `pack_layout_into_column`, scale `T::one()`, `source_conjugate` | `cuda_region_axpby`, context `1` operand, `Overwrite`, same conjugation flag |
| `recoupling_gemm_batch` / `matmul_batch_axpby_into`, α=1, β=0 | `cuda_matmul_region_into(rows, contracted, cols)` per `DenseGemmBatchJob`, serial `recoupling_plan.entries()` order |
| `scatter_column_into_layout`, alpha, `DestinationMode` | `cuda_region_axpby`, context `1` operand, β ∈ {Overwrite, Accumulate} |
| `ensure_recoupling_coefficients` (`C→D`, plan-entry order) | one upload per (structure, dtype, context), appended to the block-coefficient vector at `recoupling_base` |
| `TreeTransformWorkspaceRequirements` | two device buffers per (dtype, context), grown monotonically |

Both sides read the *same* baked fused arena: `bake_fused_layouts` bakes a Multi
source entry with its packed column as the destination and a Multi destination
entry with the packed column as the source, so pack and scatter share the host
normalizer's canonical `(dims, dst_strides, src_strides)` triples.

### GEMM orientation

TensorKit `mul!(dst, src, transpose(U))` (indexmanipulations.jl:626/:731
@cfaa073); host `recoupling_gemm_batch` reinterprets the row-major `U[dst, src]`
payload as the column-major `(src_count × dst_count)` matrix `Uᵀ`. The device
reads the same buffer as the `[k, n]` operand with leading dimension
`k = contracted = src_count`, so operand element `(s, d)` is `U[d][s]`.

Established as a test, not as an argument: `tests/common` fixtures use a
non-symmetric `U`, the CPU-only `tree_transform_device_oracle.rs` asserts the
transposed `U` gives a *different* expected buffer and that the host reproduces
the untransposed one, and `recoupling_rectangular` makes a transposed
orientation a shape error outright.

## Cache and workspace accounting

- Structure cache: key `(Weak structure identity, dtype, context)`; bounded by
  `DEFAULT_COEFFICIENT_BUDGET_BYTES` (16 MiB) and now also by
  `MAX_STRUCTURE_CACHE_ENTRIES` (32), which is what keeps the lookup scan O(1).
  One entry holds `len(coefficients) + Σ_j src_count_j·dst_count_j` elements.
- Workspace: one pair of buffers per (dtype, context), sized
  `Σ_b element_count_b · src_count_b` and `Σ_b element_count_b · dst_count_b`,
  never shrunk, reported by `workspace_device_bytes()` and included in
  `retained_device_bytes()`.
- Disclosed constants: Tenferro 0.5.0 creates a device buffer only by uploading
  one, so each workspace growth costs one H2D of zeros (values irrelevant —
  every column is fully written before it is read). Pack and scatter multiply by
  the context's `1` where the host copies, the same disclosed deviation G2a-1
  recorded for Single blocks.

Measured on A100 for `alternating_recoupling_structures_…`: the first round of
two alternating Multi structures pays 6 H2D calls (one coefficient-and-matrix
vector each, two workspace buffers, the context `1`, the zero template); the
second and third rounds pay 0 H2D and 0 device allocations.
`a_warm_recoupling_replay_…`: cold then warm, warm delta = 0 H2D, 0 D2H, 0
allocations, workspace bytes unchanged, 12 submissions (1 Single + 1 zero fill +
2×(2 packs + 1 GEMM + 2 scatters)).

## Device suite

New/changed tests, `cargo test -p tenet-operations --test cuda_tree_transform …
-- --ignored --test-threads=1`:

```
test result: ok. 15 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 4.87s
```

Full device suite, `cargo test --workspace --lib --tests --no-default-features
--features cuda,cpu-faer --no-fail-fast -- --ignored --skip
measure_checked_generic_transform_phases --skip axioms_ --skip itebd_ --skip
cross_library --test-threads=1` — 102 result lines, all `ok`, 111 tests passed,
0 failed, no panic and no `error:` in the log. Non-empty binaries:

```
test result: ok. 4 passed; 0 failed; 0 ignored; 0 measured; 71 filtered out; finished in 1.92s
test result: ok. 14 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 3.77s
test result: ok. 20 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 7.49s
test result: ok. 10 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 3.14s
test result: ok. 2 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.54s
test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 50 filtered out; finished in 0.89s
test result: ok. 10 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 3.46s
test result: ok. 15 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 4.62s
test result: ok. 13 passed; 0 failed; 0 ignored; 0 measured; 116 filtered out; finished in 5.52s
test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 82 filtered out; finished in 0.88s
test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 1 filtered out; finished in 0.89s
test result: ok. 19 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 8.38s
test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 539 filtered out; finished in 0.89s
```

## Residuals

- Real symmetry-provider fixtures (SU(2), fermionic product rules) exercise this
  path only through the typed `TensorMap` API, which is leaf G2b. The Multi
  semantics are covered here at the structure level with an independent oracle;
  the categorical fixtures land with G2b.
- The GEMM is submitted one job at a time, as the host's serial driver does.
  Grouping equal-shape jobs into one strided batch is Tenferro wishlist W9.
- A dtype change reallocates that (dtype, context)'s workspace rather than
  keeping one buffer per dtype forever; growth is monotonic within a dtype.
