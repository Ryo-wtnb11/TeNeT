# Device tree-transform executor: recoupling (Multi) blocks — 2026-09-20

Leaf G2a-3 (issue #1310). Base `origin/main` f032d121, branch
`g2a3-multi-block-executor`. Design authority `reviews/gpu-phase-20260920/g2-design.md`
§8; host authority `tenet-operations/src/transform_replay.rs`. Second revision
after `g2a3-independent-source-review.md` (P1 + nine P2s).

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

TensorKit `mul!(dst, src, transpose(U))` (indexmanipulations.jl:629/:701
@cfaa073); host `recoupling_gemm_batch` reinterprets the row-major `U[dst, src]`
payload as the column-major `(src_count × dst_count)` matrix `Uᵀ`. The device
reads the same buffer as the `[k, n]` operand with leading dimension
`k = contracted = src_count`, so operand element `(s, d)` is `U[d][s]`.

Established as a test, not as an argument: `tests/common` fixtures use a
non-symmetric `U`, the CPU-only `tree_transform_device_oracle.rs` asserts the
transposed `U` gives a *different* expected buffer and that the host reproduces
the untransposed one, and `recoupling_rectangular` makes a transposed
orientation a shape error outright.

## Categorical evidence (provider-compiled structures)

The acceptance gate's SU(2) and fermionic recoupling evidence is in
`tenet-tensors/tests/`, reached through the public
`TreeTransformCache::get_or_compile_tree_pair_structures_with_storage_conjugation`
— one level below the typed `TensorMap` API, so no G2b surface is needed.

- Fixtures: SU(2) permute / braid / planar transpose over the four-leg
  two-channel F move (degeneracy 1 and 2), and over a six-leg **five-channel**
  space whose `U` is **not** symmetric; the fZ2 ⊠ SU(2) counterparts of both,
  with and without `storage_conjugate`. Twelve compiled fixtures in total.
- **Which fermionic rule recouples:** plain fZ2 is `FusionStyleKind::Unique`, so
  every fusion tree of a key is forced, every F move is 1×1 and the compiled
  structure has `Single` blocks only — pinned by
  `a_fermionic_abelian_rule_compiles_to_single_blocks_only`. The product
  **fZ2 ⊠ SU(2)** keeps `BraidingStyleKind::Fermionic`
  (`Fermionic ⊞ Bosonic = Fermionic`) while `Unique ⊞ Simple = Simple` supplies
  the channels, so it is the one rule here whose recoupling matrix carries
  fermionic signs; `the_fermionic_recoupling_matrix_differs_from_the_bosonic_one`
  pins that its coefficients are not the bosonic ones.
- Oracle: a triple loop over `blocks()`, `layouts().{entry,shape,strides}` and
  `recoupling_coefficients_dst_src()` with **unbaked** strides — no baked arena,
  no recoupling plan, no packed column, no GEMM.
- Orientation on provider data: the two-channel SU(2) matrix is its own
  transpose (real 6j data, but blind to a reversed orientation), so the control
  uses the five-channel one, where the transposed walk gives a different buffer
  and both host and device land on the untransposed one.

## Cache and workspace accounting

- Structure cache: key `(Weak structure identity, dtype, context)`; bounded by
  `DEFAULT_COEFFICIENT_BUDGET_BYTES` (16 MiB) and by a constructor-supplied
  entry count defaulting to `DEFAULT_STRUCTURE_CACHE_ENTRIES` (256) — the host
  transform cache's own bound, so a working set that is warm on the host stays
  warm here. One entry holds `len(coefficients)` elements: the whole converted
  payload, indexed exactly as the structure indexes it, with **no re-pack** of
  the recoupling matrices (a Multi block's matrix is the run at its own
  `coefficient_start`). One key scan per replay resolves the entry to an index.
- Workspace: one pair of buffers per (dtype, context), sized
  `Σ_b element_count_b · src_count_b` and `Σ_b element_count_b · dst_count_b`,
  never shrunk, reported by `workspace_device_bytes()` and included in
  `retained_device_bytes()`.
- Disclosed constants: each workspace growth costs one H2D of zeros, the way
  every other device allocation in TeNeT is made (#740); Tenferro 0.5.0's
  `cubecl::Session::alloc_zero_output` is unusable here (broken for complex
  dtypes, tenferro-rs#1833; unpublished for kernel-written outputs), and G3c
  does not use it either. Values are irrelevant — every column is fully written
  before it is read. Pack and scatter multiply by the context's `1` where the
  host copies, the same disclosed deviation G2a-1 recorded for Single blocks.

Measured on A100 for `alternating_recoupling_structures_…`: the first round of
two alternating Multi structures pays 6 H2D calls (one coefficient-and-matrix
vector each, two workspace buffers, the context `1`, the zero template); the
second and third rounds pay 0 H2D and 0 device allocations.
`a_warm_recoupling_replay_…`: cold then warm, warm delta = 0 H2D, 0 D2H, 0
allocations, workspace bytes unchanged, 12 submissions (1 Single + 1 zero fill +
2×(2 packs + 1 GEMM + 2 scatters)).

## Device suite

New/changed tests, `cargo test -p tenet-operations --test cuda_tree_transform
-p tenet-tensors --test cuda_tree_transform_categorical … -- --ignored
--test-threads=1`:

```
test result: ok. 17 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 5.32s
test result: ok. 3 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 1.41s
```

Full device suite, `cargo test --workspace --lib --tests --no-default-features
--features cuda,cpu-faer --no-fail-fast -- --ignored --skip
measure_checked_generic_transform_phases --skip axioms_ --skip itebd_ --skip
cross_library --test-threads=1` — 105 result lines, all `ok`, 116 tests passed,
0 failed, no panic and no `error:` in the log. Non-empty binaries:

```
test result: ok. 4 passed; 0 failed; 0 ignored; 0 measured; 71 filtered out; finished in 1.79s
test result: ok. 14 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 3.56s
test result: ok. 20 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 7.04s
test result: ok. 10 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 2.93s
test result: ok. 2 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.54s
test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 50 filtered out; finished in 0.88s
test result: ok. 10 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 3.52s
test result: ok. 17 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 5.04s
test result: ok. 13 passed; 0 failed; 0 ignored; 0 measured; 116 filtered out; finished in 5.22s
test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 82 filtered out; finished in 0.88s
test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 1 filtered out; finished in 0.89s
test result: ok. 19 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 8.18s
test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 539 filtered out; finished in 0.85s
test result: ok. 3 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 1.39s
```

## Residuals

- A codomain/domain **repartition** needs a second tree-pair space (the
  destination's trees are split differently), so the categorical suite covers
  the `Transpose` *kind* through a planar rotation and leaves the repartitioning
  variant to G2b, where the typed API builds both spaces.
- The GEMM is submitted one job at a time, as the host's serial driver does.
  Grouping equal-shape jobs into one strided batch is Tenferro wishlist W9.
- A dtype change reallocates that (dtype, context)'s workspace rather than
  keeping one buffer per dtype forever; growth is monotonic within a dtype.
