# Warm contract/compose compile without rank-sized heap data (#1359)

Date 2026-09-22. Host: Apple M4 Max, macOS 15.5, rustc 1.96.0.

- **Before:** `origin/main` `ed2aa606`.
- **After:** #1359 option (A), the three commits on `allocation-free-contract-compile`
  (`5016c81e`, `cd420227`, `ae926913`; the last is CUDA-only).
- **Build:** one freshly resolved Cargo lock (`7c32d6910a6fbc7e…`), both
  revisions built one after the other into separate target directories with
  `CARGO_PROFILE_RELEASE_DEBUG=line-tables-only cargo build --release -p
  tenet-rs --example eager_overhead_ledger`. Binaries (sha256 prefix): before
  `81bbc31162e9730c`, after `5b3f41f8ff9904ef`.
- **Method:** the E1 example (`eager-overhead-ledger-2026-09-21.md`) with the
  filters `contract` and `compose`, `LEDGER_THREADS=one`, under `nice -n 19`,
  two passes in the order before, after, before, after. No cargo, rustc or
  julia process ran during the passes. Allocation counts were identical
  between passes. Medians below are the mean of the two passes' medians.
- **Raw pass-1 rows:** `warm-contract-compile-2026-09-22-{before,after}.csv`.
- Timings are observations only; the change rests on its structure.

## Structural change

The compile is the resolution a warm eager `contract`/`compose` builds per
call (resolutions stay per operation, as in TensorKit and QSpace; no cache
was added).

1. **Axis arithmetic on the stack.** `TensorContractAxisPlan`, lowered axes,
   contracted-axis candidates and `TensorContractSpecOwned` hold rank-sized
   lists in `SmallVec<[usize; 8]>`; `TreeTransformOperation` allocates only its
   shared slice. Plan candidates are scored as stack `CandidatePlan`s and only
   the winner is materialized. Before, one DynamicTree call compiled seven axis
   plans of six `Vec`s each and nine operations of four allocations each.
2. **One preflight, no rebuilt HomSpaces.** Destination checks prove
   "contracted HomSpace == destination" leg by leg
   (`OrientedFusionTreeHomSpace::tensorcontract_homspace_matches`), falling
   back to the old build-and-compare only for a mismatch or an unprovable dual.
   The core plan of plan-derived operands skips the second
   `CoreContractPreflight`. The canonical coupled-region walk skips tree-list
   compares when both sides are one structure content at one `nout`.
3. **Device (CUDA only).** The inactive-region list moved from the resolution
   into `CudaContractScratch`, rewritten in place at replay.

## Compile-only allocations (`tenet-tensors/tests/warm_contract_compile_allocations.rs`)

Warm, Runtime configuration (no context-local space cache, shared tree
store), `A(V^m ← V^n)` composed with `S(V^n ← V^n)`:

| route | before (rank 3 → 7) | after (every rank 3–7) |
|---|---|---|
| Core (`try_compile_storage_contract_core_route`) | 13 | 5 |
| plan (`prepare_tensorcontract_fusion_plan_dyn`) | 46 → 50 | 3 |
| DynamicTree (`compile_storage_contract_resolution`) | 138 → 150 | 25 |

## Host `contract_overwrite_into` on the G2c-3 device-overwrite fixtures

Warm Host calls, `dense_threads(1)`, allocation calls per call (f64 and c64
identical; the G2c-3 device run reported 188–13156 on the device path):

| fixture | before | after |
|---|---:|---:|
| U(1) rank 5 × 4, mixed sides | 258 | 39 |
| fZ2×SU(2) mixed θ | 234 | 105 |
| U(1) core route, inactive block | 30 | 13 |
| U(1) transformed lhs, identity output, inactive block | 149 | 31 |
| U(1) output transform over an inactive core block | 159 | 34 |

## E1 rows (f64 and c64, one thread)

### contract

| symmetry | dtype | case | blocks | before µs | after µs | after/before | alloc calls | alloc KiB |
|---|---|---|---:|---:|---:|---:|---:|---:|
| U1 | f64 | r2_s8_d2 | 8 | 7.83 | 6.69 | 0.85 | 152 → 35 | 7.3 → 4.8 |
| U1 | f64 | r2_s8_d16 | 8 | 11.02 | 9.67 | 0.88 | 152 → 35 | 23.1 → 20.6 |
| U1 | f64 | r3_s4_d4 | 12 | 7.33 | 5.74 | 0.78 | 150 → 35 | 13.4 → 10.5 |
| U1 | f64 | r4_s3_d4 | 19 | 11.73 | 9.96 | 0.85 | 159 → 43 | 94.3 → 90.9 |
| U1 | f64 | r5_s2_d2 | 10 | 7.15 | 5.21 | 0.73 | 159 → 37 | 11.8 → 7.5 |
| U1 | c64 | r2_s8_d2 | 8 | 7.94 | 6.15 | 0.77 | 152 → 35 | 7.6 → 5.1 |
| U1 | c64 | r2_s8_d16 | 8 | 16.17 | 14.29 | 0.88 | 152 → 35 | 39.1 → 36.6 |
| U1 | c64 | r3_s4_d4 | 12 | 7.95 | 6.30 | 0.79 | 150 → 35 | 19.4 → 16.5 |
| U1 | c64 | r4_s3_d4 | 19 | 16.29 | 14.27 | 0.88 | 159 → 43 | 180.3 → 176.9 |
| U1 | c64 | r5_s2_d2 | 10 | 7.25 | 5.42 | 0.75 | 159 → 37 | 14.3 → 10.0 |
| fZ2xU1 | f64 | r2_s8_d2 | 8 | 8.94 | 7.67 | 0.86 | 152 → 35 | 7.3 → 4.8 |
| fZ2xU1 | f64 | r2_s8_d16 | 8 | 12.77 | 11.29 | 0.88 | 152 → 35 | 23.1 → 20.6 |
| fZ2xU1 | f64 | r3_s4_d4 | 12 | 8.30 | 6.82 | 0.82 | 157 → 42 | 14.0 → 11.0 |
| fZ2xU1 | f64 | r4_s3_d4 | 19 | 12.79 | 11.44 | 0.89 | 166 → 50 | 94.8 → 91.5 |
| fZ2xU1 | f64 | r5_s2_d2 | 10 | 7.81 | 5.96 | 0.76 | 164 → 42 | 12.3 → 7.9 |
| fZ2xU1 | c64 | r2_s8_d2 | 8 | 9.54 | 8.10 | 0.85 | 152 → 35 | 7.6 → 5.1 |
| fZ2xU1 | c64 | r2_s8_d16 | 8 | 17.71 | 16.15 | 0.91 | 152 → 35 | 39.1 → 36.6 |
| fZ2xU1 | c64 | r3_s4_d4 | 12 | 9.08 | 7.49 | 0.82 | 157 → 42 | 20.0 → 17.0 |
| fZ2xU1 | c64 | r4_s3_d4 | 19 | 17.17 | 15.54 | 0.91 | 166 → 50 | 180.8 → 177.5 |
| fZ2xU1 | c64 | r5_s2_d2 | 10 | 7.71 | 6.09 | 0.79 | 164 → 42 | 14.8 → 10.4 |
| SU2 | f64 | r2_s8_d2 | 8 | 7.34 | 6.04 | 0.82 | 152 → 35 | 7.3 → 4.8 |
| SU2 | f64 | r2_s8_d16 | 8 | 11.19 | 10.04 | 0.90 | 152 → 35 | 23.1 → 20.6 |
| SU2 | f64 | r3_s4_d4 | 23 | 8.29 | 6.76 | 0.82 | 150 → 35 | 18.9 → 16.0 |
| SU2 | f64 | r4_s3_d4 | 46 | 26.83 | 24.81 | 0.92 | 167 → 51 | 148.5 → 145.1 |
| SU2 | f64 | r5_s2_d2 | 21 | 11.27 | 9.17 | 0.81 | 163 → 41 | 14.7 → 10.4 |
| SU2 | c64 | r2_s8_d2 | 8 | 7.88 | 6.30 | 0.80 | 152 → 35 | 7.6 → 5.1 |
| SU2 | c64 | r2_s8_d16 | 8 | 15.83 | 14.44 | 0.91 | 152 → 35 | 39.1 → 36.6 |
| SU2 | c64 | r3_s4_d4 | 23 | 9.34 | 7.75 | 0.83 | 150 → 35 | 30.4 → 27.5 |
| SU2 | c64 | r4_s3_d4 | 46 | 38.75 | 36.60 | 0.94 | 167 → 51 | 288.5 → 285.1 |
| SU2 | c64 | r5_s2_d2 | 21 | 11.52 | 9.53 | 0.83 | 163 → 41 | 19.9 → 15.6 |

Geometric mean after/before: 0.84 (range 0.73–0.94).

### compose

| symmetry | dtype | case | blocks | before µs | after µs | after/before | alloc calls | alloc KiB |
|---|---|---|---:|---:|---:|---:|---:|---:|
| U1 | f64 | r2_s8_d2 | 8 | 2.79 | 2.58 | 0.93 | 29 → 14 | 2.5 → 2.1 |
| U1 | f64 | r2_s8_d16 | 8 | 5.33 | 5.16 | 0.97 | 29 → 14 | 18.3 → 17.8 |
| U1 | f64 | r3_s4_d4 | 12 | 2.59 | 2.42 | 0.93 | 30 → 15 | 8.0 → 7.5 |
| U1 | f64 | r4_s3_d4 | 19 | 11.73 | 11.69 | 1.00 | 36 → 21 | 77.9 → 77.3 |
| U1 | f64 | r5_s2_d2 | 10 | 2.45 | 2.29 | 0.94 | 30 → 15 | 4.7 → 4.1 |
| U1 | c64 | r2_s8_d2 | 8 | 2.91 | 2.66 | 0.91 | 29 → 14 | 2.8 → 2.3 |
| U1 | c64 | r2_s8_d16 | 8 | 9.23 | 8.89 | 0.96 | 29 → 14 | 34.3 → 33.8 |
| U1 | c64 | r3_s4_d4 | 12 | 2.98 | 2.78 | 0.93 | 30 → 15 | 14.0 → 13.5 |
| U1 | c64 | r4_s3_d4 | 19 | 36.31 | 35.92 | 0.99 | 36 → 21 | 151.1 → 150.5 |
| U1 | c64 | r5_s2_d2 | 10 | 2.62 | 2.53 | 0.96 | 30 → 15 | 7.2 → 6.6 |
| fZ2xU1 | f64 | r2_s8_d2 | 8 | 3.02 | 2.73 | 0.91 | 29 → 14 | 2.5 → 2.1 |
| fZ2xU1 | f64 | r2_s8_d16 | 8 | 5.60 | 5.35 | 0.96 | 29 → 14 | 18.3 → 17.8 |
| fZ2xU1 | f64 | r3_s4_d4 | 12 | 2.70 | 2.53 | 0.94 | 30 → 15 | 8.0 → 7.5 |
| fZ2xU1 | f64 | r4_s3_d4 | 19 | 12.04 | 12.02 | 1.00 | 37 → 22 | 77.9 → 77.4 |
| fZ2xU1 | f64 | r5_s2_d2 | 10 | 2.54 | 2.33 | 0.92 | 30 → 15 | 4.7 → 4.1 |
| fZ2xU1 | c64 | r2_s8_d2 | 8 | 3.17 | 2.96 | 0.93 | 29 → 14 | 2.8 → 2.3 |
| fZ2xU1 | c64 | r2_s8_d16 | 8 | 9.46 | 9.25 | 0.98 | 29 → 14 | 34.3 → 33.8 |
| fZ2xU1 | c64 | r3_s4_d4 | 12 | 3.10 | 2.96 | 0.95 | 30 → 15 | 14.0 → 13.5 |
| fZ2xU1 | c64 | r4_s3_d4 | 19 | 36.54 | 36.15 | 0.99 | 37 → 22 | 151.1 → 150.6 |
| fZ2xU1 | c64 | r5_s2_d2 | 10 | 2.80 | 2.55 | 0.91 | 30 → 15 | 7.2 → 6.6 |
| SU2 | f64 | r2_s8_d2 | 8 | 2.80 | 2.56 | 0.91 | 29 → 14 | 2.5 → 2.1 |
| SU2 | f64 | r2_s8_d16 | 8 | 5.39 | 5.15 | 0.96 | 29 → 14 | 18.3 → 17.8 |
| SU2 | f64 | r3_s4_d4 | 23 | 3.03 | 2.68 | 0.89 | 30 → 15 | 13.5 → 13.0 |
| SU2 | f64 | r4_s3_d4 | 46 | 30.94 | 31.73 | 1.03 | 38 → 23 | 187.9 → 187.3 |
| SU2 | f64 | r5_s2_d2 | 21 | 2.71 | 2.57 | 0.95 | 30 → 15 | 7.4 → 6.9 |
| SU2 | c64 | r2_s8_d2 | 8 | 2.91 | 2.73 | 0.94 | 29 → 14 | 2.8 → 2.3 |
| SU2 | c64 | r2_s8_d16 | 8 | 9.27 | 9.03 | 0.97 | 29 → 14 | 34.3 → 33.8 |
| SU2 | c64 | r3_s4_d4 | 23 | 3.67 | 3.42 | 0.93 | 30 → 15 | 25.0 → 24.5 |
| SU2 | c64 | r4_s3_d4 | 46 | 124.94 | 124.00 | 0.99 | 38 → 23 | 371.1 → 370.5 |
| SU2 | c64 | r5_s2_d2 | 21 | 3.23 | 3.04 | 0.94 | 30 → 15 | 12.7 → 12.1 |

Geometric mean after/before: 0.95 (range 0.89–1.03).

## What remains per call

- Per transformed operand: the permuted HomSpace (TensorKit forms it too),
  one `Arc<SectorLegData>` per leg that crosses codomain and domain
  (`SectorLeg::dual`; the only rank-growing allocation left), the layout
  lookup key (#1367) and the commit lock (#1366), and the two `Arc`s the replay
  scratch shares.
- The core destination space when the output transform is not the identity,
  the plan vectors, the fermionic twist actions (O(blocks)), and the plan and
  artifact `Arc`s.
- The destination derivation (#1358/L1) and the Tenferro session entry (E1
  constant 5) are outside this leaf.
