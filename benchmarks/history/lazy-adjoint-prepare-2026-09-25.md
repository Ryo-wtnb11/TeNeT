# Warm lazy-adjoint contract: no per-call adjoint projection (#1419)

Date 2026-09-25. Host: Apple M4 Max, macOS 15.5, rustc 1.96.0.

- **Before:** `origin/main` `18161561`.
- **After:** branch `lazy-adjoint-prepare`.
- **Build:** `CARGO_PROFILE_RELEASE_DEBUG=line-tables-only cargo build --release
  --locked -p tenet-rs --example eager_overhead_ledger`, one target directory
  per revision. Binaries (sha256 prefix): before `b0b34537b3025494`, after
  `6e839e36f8382da8`.
- **Method:** E1, `LEDGER_THREADS=one`, filters `f64` plus one of `contract`,
  `contract_conj`, `compose_conj`, `nice -n 19`, two passes ordered before,
  after, before, after. Allocation counts and bytes matched between passes.
  Medians are the mean of the two passes. Raw pass-1 rows:
  `lazy-adjoint-prepare-2026-09-25-{before,after}.csv`.
- Timings are observations only.

## Attribution

A test-only counting allocator recorded a backtrace per allocation during one
warm call (release, line tables) and grouped the calls by their first TeNeT
frame. `contract_conj` minus owned `contract`, calls per warm call:

| site | U1 r4 | SU2 r4 | fZ2×U1 r4 | U1 r2 | SU2 r2 | fZ2×U1 r2 |
|---|---:|---:|---:|---:|---:|---:|
| `oriented_contract_destination`: axis-set `vec![false; n]` ×3 | +3 | +3 | +3 | +3 | +3 | +3 |
| `FusionOperand::prepare`: adjoint HomSpace `materialize` | +1 | +1 | +1 | +1 | +1 | +1 |
| `FusionOperand::prepare`: `storage_indices` Vec (+ one lookup per block) | +1 | +1 | +1 | +1 | +1 | +1 |
| `tensorcontract_fusion_block_specs_lowered`: contracted HomSpace built to compare with dst | | +1 | | | +1 | |
| `is_core_form_fusion_source_contract`: two axis-range Vecs | | +2 | | | +2 | |
| permute descriptor `SectorLeg::try_dual` (route) | | | | +4 | | +4 |
| `compile_rhs_contract_twist` actions, shapes, strides (route) | | | | | | +10 |
| owned route only: core-destination space, its contracted HomSpace and space | | | | −3 | −3 | −3 |
| **gap before** | **+5** | **+8** | **+5** | **+6** | **+5** | **+16** |
| **gap after** | **0** | **0** | **0** | **+1** | **−3** | **+11** |

Every other site matched one for one (the transformed-source spaces, plan and
artifact `Arc`s, block GEMM scratch, output payload).

## Change

- `FusionOperand::prepare` borrows the parent space's memoized adjoint HomSpace
  (#1368's `adjoint` slot) instead of materializing one per call. The slot now
  holds `Arc<AdjointMemo>`: the HomSpace fills first and the adjoint block
  structure only when `adjoint_view` asks, so a lazy-adjoint contraction does
  not retain an unused adjoint structure. The slot shrank from 24 to 16 bytes,
  which is the −24/−32 bytes on owned `contract` below.
- For a Complete parent, the logical-to-storage map is filled on first read.
  A warm call takes its oriented plan from the Runtime store (#1417) and reads
  it only on a store miss, so it performs no per-block lookup. A Subset parent
  still selects its keys eagerly, because the logical key list depends on them.
  The Complete validation (missing key, block count) runs where the map is
  first read, in the same order.
- `oriented_contract_destination` checks axis sets with `SmallVec<[bool; 16]>`,
  as `TensorContractAxisPlan` already does.
- `tensorcontract_fusion_block_specs_lowered` compares with
  `FusionTreeHomSpace::tensorcontract_homspace_matches` (same errors, same
  order) instead of building the contracted HomSpace, and the core-form checks
  compare against ranges. This also moves the expert conjugated-contract pin
  in `conjugated_contract_allocations.rs` for SU2 from 168 to 162.

Results are bitwise equal: the same plans and the same arithmetic run. A hash of
the warm output bits for U1, SU2 and fZ2×U1 at `r2_s8_d2` and `r4_s3_d4`
matched between the two revisions.

## E1 (facade), f64

Calls and bytes per warm call; the median is in µs.

| op | symmetry | case | before calls / B | after calls / B | before med µs | after med µs |
|---|---|---|---:|---:|---:|---:|
| contract_conj | U1 | r2_s8_d2 | 33 / 4950 | 28 / 4568 | 5.73 | 5.43 |
| contract_conj | U1 | r2_s8_d16 | 33 / 21078 | 28 / 20696 | 9.81 | 9.35 |
| contract_conj | U1 | r3_s4_d4 | 36 / 10872 | 31 / 10456 | 6.22 | 5.45 |
| contract_conj | U1 | r4_s3_d4 | 34 / 91546 | 29 / 91072 | 11.00 | 9.65 |
| contract_conj | U1 | r5_s2_d2 | 38 / 7500 | 33 / 7096 | 5.73 | 5.19 |
| contract_conj | fZ2×U1 | r2_s8_d2 | 43 / 5718 | 38 / 5336 | 8.17 | 7.75 |
| contract_conj | fZ2×U1 | r2_s8_d16 | 43 / 21846 | 38 / 21464 | 12.40 | 11.98 |
| contract_conj | fZ2×U1 | r3_s4_d4 | 43 / 11448 | 38 / 11032 | 7.36 | 6.74 |
| contract_conj | fZ2×U1 | r4_s3_d4 | 41 / 92122 | 36 / 91648 | 12.10 | 10.81 |
| contract_conj | fZ2×U1 | r5_s2_d2 | 43 / 7948 | 38 / 7544 | 6.70 | 5.85 |
| contract_conj | SU2 | r2_s8_d2 | 30 / 4294 | 22 / 3608 | 6.15 | 5.58 |
| contract_conj | SU2 | r2_s8_d16 | 30 / 20422 | 22 / 19736 | 10.17 | 9.57 |
| contract_conj | SU2 | r3_s4_d4 | 31 / 15624 | 23 / 14808 | 8.06 | 6.58 |
| contract_conj | SU2 | r4_s3_d4 | 45 / 147562 | 37 / 146560 | 29.31 | 27.23 |
| contract_conj | SU2 | r5_s2_d2 | 36 / 9244 | 28 / 8432 | 12.02 | 10.85 |
| compose_conj | U1 | r2_s8_d2 | 16 / 1990 | 13 / 1976 | 2.57 | 2.59 |
| compose_conj | U1 | r2_s8_d16 | 16 / 18118 | 13 / 18104 | 5.65 | 5.66 |
| compose_conj | U1 | r3_s4_d4 | 16 / 1816 | 13 / 1800 | 4.36 | 4.33 |
| compose_conj | U1 | r4_s3_d4 | 22 / 113620 | 19 / 113600 | 14.88 | 14.96 |
| compose_conj | U1 | r5_s2_d2 | 16 / 2054 | 13 / 2032 | 3.57 | 3.53 |
| compose_conj | fZ2×U1 | r2_s8_d2 | 18 / 2342 | 13 / 1976 | 3.41 | 3.08 |
| compose_conj | fZ2×U1 | r2_s8_d16 | 18 / 18470 | 13 / 18104 | 6.56 | 6.19 |
| compose_conj | fZ2×U1 | r3_s4_d4 | 18 / 2200 | 13 / 1800 | 5.23 | 4.63 |
| compose_conj | fZ2×U1 | r4_s3_d4 | 25 / 114124 | 20 / 113664 | 16.31 | 15.29 |
| compose_conj | fZ2×U1 | r5_s2_d2 | 18 / 2422 | 13 / 2032 | 4.32 | 3.74 |
| compose_conj | SU2 | r2_s8_d2 | 16 / 1990 | 13 / 1976 | 2.60 | 2.59 |
| compose_conj | SU2 | r2_s8_d16 | 16 / 18118 | 13 / 18104 | 5.73 | 5.60 |
| compose_conj | SU2 | r3_s4_d4 | 16 / 1816 | 13 / 1800 | 4.83 | 4.71 |
| compose_conj | SU2 | r4_s3_d4 | 24 / 283603 | 21 / 283583 | 36.85 | 36.85 |
| compose_conj | SU2 | r5_s2_d2 | 16 / 2438 | 13 / 2416 | 3.91 | 3.86 |
| contract | U1 | r2_s8_d2 | 27 / 4416 | 27 / 4384 | 5.45 | 5.36 |
| contract | U1 | r2_s8_d16 | 27 / 20544 | 27 / 20512 | 9.45 | 9.22 |
| contract | U1 | r3_s4_d4 | 29 / 10160 | 29 / 10136 | 5.42 | 5.43 |
| contract | U1 | r4_s3_d4 | 29 / 91096 | 29 / 91072 | 9.29 | 9.39 |
| contract | U1 | r5_s2_d2 | 31 / 6800 | 31 / 6776 | 4.61 | 4.69 |
| contract | fZ2×U1 | r2_s8_d2 | 27 / 4416 | 27 / 4384 | 7.11 | 7.20 |
| contract | fZ2×U1 | r2_s8_d16 | 27 / 20544 | 27 / 20512 | 11.15 | 11.08 |
| contract | fZ2×U1 | r3_s4_d4 | 36 / 10736 | 36 / 10712 | 6.36 | 6.34 |
| contract | fZ2×U1 | r4_s3_d4 | 36 / 91672 | 36 / 91648 | 10.38 | 10.48 |
| contract | fZ2×U1 | r5_s2_d2 | 36 / 7248 | 36 / 7224 | 5.33 | 5.35 |
| contract | SU2 | r2_s8_d2 | 25 / 4096 | 25 / 4064 | 5.46 | 5.39 |
| contract | SU2 | r2_s8_d16 | 25 / 20224 | 25 / 20192 | 9.24 | 9.14 |
| contract | SU2 | r3_s4_d4 | 23 / 14832 | 23 / 14808 | 5.95 | 5.96 |
| contract | SU2 | r4_s3_d4 | 37 / 146584 | 37 / 146560 | 24.25 | 24.40 |
| contract | SU2 | r5_s2_d2 | 27 / 8432 | 27 / 8408 | 8.19 | 8.28 |

## Residual

Warm `contract_conj` now equals owned `contract` on every `r4_s3_d4` row and
on SU2 `r3_s4_d4`. The rest of the gap is not per-call adjoint work:

- U1 and fZ2×U1 `r3`, `r5` (+2), U1 `r2` (+1): `SectorLeg::try_dual` in the
  permute descriptor of the transformed source. The logical adjoint moves
  different legs across the codomain/domain bar than the owned operand, so it
  dualizes more of them. Owned `contract` runs the same code per call (6 to 8
  dual builds on these rows). SU2 legs are self-dual and build nothing.
- fZ2×U1 `r2_s8_d2` (+11): the candidate selector picks another route for the
  adjoint operand, since the operand cannot be borrowed. That route twists the
  right operand (+10 twist-action allocations) instead of transforming the
  output. SU2 `r5_s2_d2` (+1) is one more recoupling GEMM batch in that
  route's replay.
- Both are per-call artifact compilation, which owned `contract` also runs on
  every warm call. They belong to the eager-artifact work, not to the adjoint
  boundary.
