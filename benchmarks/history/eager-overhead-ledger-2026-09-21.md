# Eager per-call overhead ledger — small many-block tensors (#1313)

Date 2026-09-21. Host: Apple M4 Max (12 P + 4 E cores), macOS 15.5 (24F74),
rustc 1.96.0 / cargo 1.96.0. TeNeT revision `origin/main` `28a9c67d` plus this
commit's benchmark sources only (no production code changed). Cargo lock
`cad23363c997e3895a5391be45ee97fb3cfe19ca049c53593bbc8f60580f59b2`, default
features (`cpu-faer`), `cargo build --release --locked --offline -p tenet-rs
--example eager_overhead_ledger` with `CARGO_PROFILE_RELEASE_DEBUG=line-tables-only`
(symbols for the sampler; codegen unchanged). Example binary sha256
`b5420876309325f11becc77cc968dba545a106c3de51f8db33dc52e9c2ed56fd`, built by
`benchmarks/eager_overhead_ledger.sh` from this commit's sources.

Reference: TensorKit.jl 0.17.1 at `f87ca7fe557abbc79561d23298028664ed5dbcd6`
(package tree `b8e781a8`, the pinned `benchmarks/tensorkit_benchmark`
environment, unchanged), Julia 1.11.6, `julia -t 1`, OpenBLAS with
`BLAS.set_num_threads(1)` (AppleAccelerate is in that environment but not
loaded by this script).

Reproduce: `benchmarks/eager_overhead_ledger.sh OUT_DIR` (refuses to start while
another cargo/rustc/julia runs), then
`benchmarks/eager_overhead_phases.py OUT_DIR/<row>.one.sample` per phase table.

Raw data committed next to this file:

- `eager-overhead-ledger-2026-09-21-tenet-{one,one-run2,default,tenet1,dense1}.csv`
- `eager-overhead-ledger-2026-09-21-tensorkit{,-run2}.csv`

Columns: `threads,symmetry,dtype,case,rank,blocks,coupled,elements,op,iterations,min_ns,median_ns,alloc_calls,alloc_bytes`.
`blocks` is the number of stored fusion-tree blocks, `coupled` the number of
coupled sectors; both are identical between TeNeT and TensorKit in every row
(checked), so the two libraries run the same block shapes.

## What was measured

Legs: one leg space `V` with `s` sectors of degeneracy `d` each, U(1) charges
centred on 0, fZ2×U(1) with parity = charge mod 2, SU(2) spins `0, 1/2, …`.

| case | tensor | blocks U(1) / fZ2×U(1) / SU(2) | elements (U(1)) |
|---|---|---|---:|
| `r2_s8_d2` | `V ← V`, 8 sectors, d = 2 | 8 / 8 / 8 | 32 |
| `r2_s8_d16` | `V ← V`, 8 sectors, d = 16 | 8 / 8 / 8 | 2048 |
| `r3_s4_d4` | `V⊗V ← V`, 4 sectors, d = 4 | 12 / 12 / 23 | 768 |
| `r4_s3_d4` | `V⊗V ← V⊗V`, 3 sectors, d = 4 | 19 / 19 / 46 | 4864 |
| `r5_s2_d2` | `V⊗V⊗V ← V⊗V`, 2 sectors, d = 2 | 10 / 10 / 21 | 320 |

| op | TeNeT call | TensorKit call |
|---|---|---|
| compose | `a.compose(&s)`, `s: dom ← dom` | `A * S` |
| contract | `a.contract(&m, &[0], &[1], 0..r)`, `m: V ← V` | `TO.tensorcontract(A, ((2..r),(1,)), false, M, ((2,),(1,)), false, ((1..r-1),(r,)))` |
| permute | all axes rotated by one, same codomain count | `permute(A, perm)` |
| repartition | `nc+1` if `nd ≥ 2` else `nc−1` | `repartition(A, N)` |
| qr_compact | `a.qr_compact()` | `qr_compact(A)` |
| restrict_leg | leg 0 restricted to the first `⌈d/2⌉` of every sector | `P ∘₀ A` with `P = isometry(V, W)'` (no restriction primitive exists; see below) |
| scale / add / norm | `a.scale(2)`, `a.add(&a2, 1, 1)`, `a.norm()` | `scale(A, 2)`, `A + A2`, `norm(A)` |

Timing (both libraries, same estimator): 30 ms warm-up, then batches of ≥ 10 µs
(the host timer ticks at ~42 ns) for 250 ms or 4000 batches; per-call minimum
and median over the batches. BenchmarkTools is not in the pinned Julia
environment and adding it would change a reference environment, so the same
manual estimator is used on both sides. Allocation: one warm call (TeNeT: counting
global allocator, calls + requested bytes on the calling thread, result drop
included; TensorKit: `@timed`, `gc_alloc_count` + bytes).

Thread layouts (`LEDGER_THREADS`): `one` = `Runtime::builder().dense_threads(1)`
(TeNeT's Rayon global pool and the Tenferro CPU context both 1); `default` = no
configuration (16 + 16 workers); `tenet1` = Rayon global pool 1, Tenferro context
default; `dense1` = Rayon global pool default, Tenferro context 1.

Noise: the second one-thread pass and the second TensorKit pass
(`*-run2.csv`) agree with the first to a median per-row ratio of 1.00 / 1.02
(10–90 %: 0.98–1.04 TeNeT, 0.97–1.09 TensorKit); an earlier full pass on the
pre-`black_box` build gave the same per-op geometric means within 0.05.
Allocation counts are identical across passes. Wall-clock numbers here are
observations, not CI gates.

## Headline: TeNeT vs TensorKit, f64, one thread (median µs)

| op | case | U(1) TeNeT / TK | fZ2×U(1) | SU(2) | TeNeT allocs U(1) calls / KiB | TK allocs U(1) |
|---|---|---:|---:|---:|---:|---:|
| compose | r2_s8_d2 | 4.04 / 1.38 (2.94×) | 3.85 / 1.62 (2.37×) | 3.60 / 1.57 (2.29×) | 29 / 2.5 | 56 / 2.9 |
| compose | r4_s3_d4 | 13.25 / 11.21 (1.18×) | 13.67 / 11.62 (1.18×) | 34.92 / 32.92 (1.06×) | 36 / 77.9 | 42 / 65.8 |
| contract | r2_s8_d2 | 10.67 / 3.17 (3.37×) | 12.83 / 3.58 (3.58×) | 10.83 / 3.58 (3.02×) | 152 / 7.3 | 122 / 6.2 |
| contract | r4_s3_d4 | 15.79 / 19.29 (0.82×) | 16.75 / 21.38 (0.78×) | 34.46 / 63.92 (0.54×) | 159 / 94.3 | 304 / 146.0 |
| permute | r2_s8_d2 | 1.79 / 1.08 (1.66×) | 2.23 / 1.13 (1.98×) | 1.82 / 1.22 (1.49×) | 12 / 1.3 | 62 / 3.0 |
| permute | r4_s3_d4 | 6.02 / 10.33 (0.58×) | 6.25 / 11.00 (0.57×) | 20.58 / 48.62 (0.42×) | 13 / 39.4 | 254 / 79.3 |
| repartition | r2_s8_d2 | 1.44 / 1.12 (1.28×) | 1.83 / 1.21 (1.51×) | 1.41 / 1.25 (1.13×) | 10 / 1.2 | 68 / 3.3 |
| repartition | r4_s3_d4 | 4.14 / 8.58 (0.48×) | 4.18 / 8.62 (0.48×) | 8.38 / 42.75 (0.20×) | 11 / 39.1 | 260 / 79.6 |
| qr_compact | r2_s8_d2 | 23.62 / 18.58 (1.27×) | 24.29 / 21.38 (1.14×) | 24.67 / 21.12 (1.17×) | 122 / 91.6 | 547 / 25.2 |
| qr_compact | r4_s3_d4 | 86.46 / 123.96 (0.70×) | 87.25 / 126.04 (0.69×) | 198.46 / 264.38 (0.75×) | 188 / 405.5 | 423 / 282.3 |
| restrict_leg | r2_s8_d2 | 1.99 / 2.34 (0.85×) | 2.32 / 2.68 (0.87×) | 1.98 / 2.64 (0.75×) | 13 / 1.1 | 32 / 1.5 |
| restrict_leg | r4_s3_d4 | 14.29 / 20.08 (0.71×) | 14.17 / 21.33 (0.66×) | 33.62 / 91.54 (0.37×) | 13 / 20.1 | 558 / 135.4 |
| scale | r2_s8_d2 | 0.05 / 0.16 (0.31×) | 0.05 / 0.21 (0.24×) | 0.05 / 0.22 (0.24×) | 3 / 0.5 | 4 / 0.4 |
| scale | r4_s3_d4 | 0.39 / 2.03 (0.19×) | 0.58 / 2.35 (0.25×) | 1.02 / 4.38 (0.23×) | 3 / 38.2 | 5 / 64.1 |
| add | r2_s8_d2 | 0.06 / 0.21 (0.29×) | 0.06 / 0.21 (0.29×) | 0.06 / 0.20 (0.32×) | 3 / 0.5 | 4 / 0.4 |
| add | r4_s3_d4 | 0.66 / 3.25 (0.20×) | 0.59 / 3.38 (0.18×) | 2.41 / 7.25 (0.33×) | 3 / 38.2 | 5 / 64.1 |
| norm | r2_s8_d2 | 0.00 / 0.02 | 0.00 / 0.02 | 0.02 / 0.40 | 0 / 0 | 0 / 0 |
| norm | r4_s3_d4 | 2.31 / 4.57 (0.51×) | 2.31 / 4.60 (0.50×) | 5.58 / 11.00 (0.51×) | 0 / 0 | 0 / 0 |

Over all 270 rows (3 symmetries × 2 dtypes × 5 cases × 9 ops), TeNeT/TensorKit
median-time ratio, geometric mean (min–max): compose 1.46 (0.86–2.94),
contract 1.32 (0.54–3.58), permute 0.83 (0.33–1.98), repartition 0.64
(0.17–1.51), qr_compact 0.78 (0.52–1.27), restrict_leg 0.54 (0.20–1.03), scale
0.26, add 0.28, norm 0.43 (0.06–1.70).

**Where the invariant fails.** TeNeT is slower than TensorKit exactly where the
payload is small and the call floor dominates: the per-call floor is ~3.5 µs
for `compose` (TK ~1.4), ~10 µs for `contract` (TK ~3.2), ~1.8 µs for
`permute` (TK ~1.1), ~1.4 µs for `repartition` (TK ~1.1), and ~2.9 µs per
coupled-sector QR (TK ~2.3). From ~20 blocks / ~5 k elements up, contract,
permute, repartition, qr_compact and restrict_leg are faster than TensorKit
(2–5× on the SU(2) transforms); f64 compose stays 1.06–1.18× slower at r4 and
1.4–1.7× at `r2_s8_d16` (Complex64 compose is at parity there). One more kernel-level
exception: Complex64 `norm` on abelian rules is 1.3–1.7× TensorKit (which calls
BLAS `nrm2` on the flat data vector); f64 `norm` is 0.5×.

**restrict_leg reference.** TensorKit has no degeneracy-range restriction;
its idiomatic equivalent, contraction with the adjoint of `isometry(V, W)`, is a
GEMM and does more work than TeNeT's gather. The ratio is recorded but is not a
parity statement; QSpace has no corresponding path either (below).

## Phase ledger (f64, one thread)

Method: the existing typed profiles (`TensorContractFusionProfile`,
`TreeTransformReplayProfile`) sit behind `*_profiled` expert entry points that
the public `TensorMap` methods never reach, so they cannot attribute the
facade's cost. No hook was added. Instead the example loops one operation for
4 s while macOS `/usr/bin/sample` records the call tree at 1 ms
(`LEDGER_SAMPLE`, ~3000 main-thread samples per row), and
`benchmarks/eager_overhead_phases.py` charges every sample to the deepest frame
matching a phase rule. Values are share × measured median, in µs per call;
`·` < 0.5 %. Statistical resolution is ~±2 % of the row. Hashing inside
interning is charged to hashing, allocation inside a phase to that phase;
`alloc` is the payload allocation/zeroing issued directly by the facade;
`errdrop` is `drop_in_place::<…Error>` on the success path (see constant 4).

| sym | case | op | median | valid | space | plan | hash | alloc | wksp | dispatch | backend | gemm/qr | strided | errdrop | drop | facade+loops |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| U1 | r2_s8_d2 | compose | 4.04 | 0.20 | 1.35 | 0.38 | · | 0.15 | · | 0.02 | 1.25 | 0.12 | · | 0.24 | 0.22 | 0.10 |
| U1 | r2_s8_d2 | contract | 10.67 | 0.52 | 4.54 | 2.39 | 0.16 | 0.17 | · | 0.11 | 1.14 | 0.08 | 0.10 | 0.69 | 0.68 | 0.08 |
| U1 | r2_s8_d2 | permute | 1.79 | 0.04 | 1.19 | 0.06 | 0.02 | 0.07 | 0.01 | 0.14 | · | · | · | 0.15 | 0.09 | 0.03 |
| U1 | r2_s8_d2 | repartition | 1.44 | 0.06 | 0.86 | 0.07 | 0.02 | 0.05 | · | 0.16 | · | · | · | 0.11 | 0.05 | 0.04 |
| U1 | r2_s8_d2 | qr_compact | 23.62 | · | 1.80 | 0.19 | · | · | · | 1.43 | 16.51 | 2.56 | · | 0.58 | 0.31 | · |
| U1 | r2_s8_d2 | restrict_leg | 1.99 | · | 0.85 | 0.11 | 0.03 | 0.09 | · | 0.52 | · | · | · | 0.22 | 0.08 | 0.08 |
| fZ2xU1 | r2_s8_d2 | compose | 3.85 | 0.13 | 1.40 | 0.48 | · | 0.09 | · | · | 1.05 | 0.12 | · | 0.23 | 0.23 | 0.08 |
| fZ2xU1 | r2_s8_d2 | contract | 12.83 | 0.51 | 6.21 | 2.69 | 0.17 | 0.21 | · | 0.08 | 1.31 | 0.12 | 0.08 | 0.75 | 0.60 | 0.10 |
| fZ2xU1 | r2_s8_d2 | permute | 2.23 | 0.03 | 1.58 | 0.07 | 0.03 | 0.06 | · | 0.15 | · | · | · | 0.16 | 0.10 | 0.05 |
| fZ2xU1 | r2_s8_d2 | repartition | 1.83 | 0.06 | 1.18 | 0.08 | 0.03 | 0.06 | · | 0.16 | · | · | · | 0.12 | 0.08 | 0.05 |
| fZ2xU1 | r2_s8_d2 | qr_compact | 24.29 | · | 2.19 | 0.32 | · | · | · | 1.47 | 15.98 | 2.85 | · | 0.79 | 0.39 | · |
| fZ2xU1 | r2_s8_d2 | restrict_leg | 2.32 | · | 1.09 | 0.13 | 0.03 | 0.10 | · | 0.53 | · | · | · | 0.28 | 0.08 | 0.09 |
| SU2 | r2_s8_d2 | compose | 3.60 | 0.18 | 1.21 | 0.36 | · | 0.12 | · | · | 1.12 | 0.10 | · | 0.22 | 0.18 | 0.07 |
| SU2 | r2_s8_d2 | contract | 10.83 | 0.41 | 4.81 | 2.33 | 0.14 | 0.20 | · | 0.08 | 1.30 | 0.14 | 0.09 | 0.67 | 0.53 | 0.14 |
| SU2 | r2_s8_d2 | permute | 1.82 | 0.03 | 1.21 | 0.11 | 0.02 | 0.04 | · | 0.15 | · | · | · | 0.14 | 0.08 | 0.04 |
| SU2 | r2_s8_d2 | repartition | 1.41 | 0.06 | 0.86 | 0.06 | 0.02 | 0.06 | · | 0.13 | · | · | · | 0.09 | 0.05 | 0.06 |
| SU2 | r2_s8_d2 | qr_compact | 24.67 | · | 1.98 | 0.26 | · | · | · | 1.59 | 16.76 | 2.62 | · | 0.81 | 0.43 | · |
| SU2 | r2_s8_d2 | restrict_leg | 1.98 | · | 0.85 | 0.11 | 0.02 | 0.10 | · | 0.49 | · | · | · | 0.26 | 0.08 | 0.08 |
| U1 | r4_s3_d4 | compose | 13.25 | 0.32 | 2.17 | 0.39 | · | 0.77 | · | · | 0.96 | 7.93 | · | 0.36 | 0.22 | 0.08 |
| U1 | r4_s3_d4 | contract | 15.79 | 0.55 | 5.41 | 2.22 | · | 0.77 | · | 0.12 | 0.92 | 1.67 | 2.46 | 0.89 | 0.59 | 0.10 |
| U1 | r4_s3_d4 | permute | 6.02 | 0.04 | 1.88 | 0.14 | 0.05 | 0.60 | · | 0.16 | · | · | 2.70 | 0.27 | 0.12 | 0.05 |
| U1 | r4_s3_d4 | repartition | 4.14 | 0.08 | 1.58 | 0.13 | 0.05 | 0.62 | · | 0.23 | · | · | 0.97 | 0.33 | 0.09 | 0.05 |
| U1 | r4_s3_d4 | qr_compact | 86.46 | · | 1.94 | 2.14 | 0.48 | 0.97 | · | 5.13 | 17.43 | 56.28 | · | 0.83 | 1.17 | · |
| U1 | r4_s3_d4 | restrict_leg | 14.29 | · | 1.50 | 0.12 | · | 0.33 | · | 11.09 | · | · | · | 0.93 | 0.08 | 0.21 |
| U1 | r4_s3_d4 | scale | 0.39 | · | · | · | · | 0.04 | · | · | · | · | · | · | · | 0.35 |
| U1 | r4_s3_d4 | add | 0.66 | · | · | · | · | 0.06 | · | · | · | · | · | · | · | 0.59 |
| U1 | r4_s3_d4 | norm | 2.31 | · | · | · | · | · | · | · | · | · | · | · | · | 2.31 |
| fZ2xU1 | r4_s3_d4 | compose | 13.67 | 0.22 | 2.04 | 0.48 | · | 0.82 | · | · | 1.05 | 8.34 | · | 0.39 | 0.17 | 0.09 |
| fZ2xU1 | r4_s3_d4 | contract | 16.75 | 0.46 | 5.95 | 2.70 | · | 0.81 | · | 0.14 | 1.04 | 1.57 | 2.38 | 0.89 | 0.60 | · |
| fZ2xU1 | r4_s3_d4 | permute | 6.25 | 0.03 | 1.80 | 0.11 | 0.03 | 0.67 | · | 0.19 | · | · | 2.91 | 0.30 | 0.12 | 0.07 |
| fZ2xU1 | r4_s3_d4 | repartition | 4.18 | 0.08 | 1.80 | 0.11 | 0.04 | 0.60 | · | 0.21 | · | · | 0.85 | 0.32 | 0.10 | 0.05 |
| fZ2xU1 | r4_s3_d4 | qr_compact | 87.25 | · | 2.72 | 2.32 | 0.74 | 1.38 | · | 4.99 | 16.28 | 56.59 | · | 0.80 | 1.17 | · |
| fZ2xU1 | r4_s3_d4 | restrict_leg | 14.17 | · | 1.64 | 0.14 | · | 0.30 | · | 10.94 | · | · | · | 0.81 | 0.10 | 0.21 |
| SU2 | r4_s3_d4 | compose | 34.92 | 0.22 | 3.33 | 0.42 | · | 0.92 | · | · | 1.04 | 27.99 | · | 0.65 | · | · |
| SU2 | r4_s3_d4 | contract | 34.46 | 0.53 | 8.60 | 2.51 | · | 1.13 | · | 0.81 | 6.99 | 4.95 | 6.13 | 1.98 | 0.54 | 0.19 |
| SU2 | r4_s3_d4 | permute | 20.58 | · | 2.79 | 0.16 | · | 1.12 | · | 0.57 | 6.16 | 1.46 | 7.21 | 0.85 | 0.13 | · |
| SU2 | r4_s3_d4 | repartition | 8.38 | 0.09 | 3.29 | 0.21 | · | 1.12 | · | 0.55 | · | · | 2.16 | 0.70 | 0.13 | 0.07 |
| SU2 | r4_s3_d4 | qr_compact | 198.46 | · | 3.08 | 2.23 | · | 2.69 | · | 8.40 | 24.81 | 153.77 | · | 1.38 | 1.05 | · |
| SU2 | r4_s3_d4 | restrict_leg | 33.62 | · | 3.13 | · | · | 0.85 | · | 26.98 | · | · | · | 2.00 | · | 0.41 |

`scale`/`add`/`norm` are a single inlined loop plus one payload allocation in
every row and are omitted for the small case (≤ 0.06 µs). The workspace lease
(`Runtime::lease_context`) is below 0.5 % everywhere.

Owning symbols behind the large columns (from the call trees):

- **space** — `DynamicFusionMapSpace::transformed_with_primer`
  (`tenet-tensors/src/contract/dynamic_space.rs:2254`) →
  `from_final_homspace_with_prepared` (`:2037`) →
  `tenet_core::PreparedFusionTreeLayout::build_complete_from_leg_degeneracies`
  / `DegeneracyBlock::new` for permute, repartition, and each contract operand;
  `BoundDynamicFusionMapSpace::contracted_multiplicity_free_ordered`
  (`dynamic_space.rs:1635`, called at `tenet/src/tensor_core.rs:427`) for the
  compose/contract output; `DynamicFusionMapSpace::core_dst_with_primer` for
  the contract core destination; `BoundDynamicFusionMapSpace::from_final_homspace`
  / `root_with_replaced_leg` for restrict.
- **plan** — contract: `resolution::compile_dynamic_tree_plan`
  (`tenet-tensors/src/contract/resolution.rs:211`) →
  `fusion::plan::prepare_tensorcontract_fusion_plan_dyn_raw_canonical`
  (`contract/fusion/plan.rs:460`), and
  `dynamic::finish_dynamic_tree_execution_artifact` → `compile_core_plan`
  (`resolution.rs:510`). compose: `FusionBlockContractPlan::try_from_canonical_coupled_regions`
  (`tenet-operations/src/fusion_replay.rs`).
- **backend** — `<DefaultDenseExecutor as DenseExecutor>::{matmul_batch_axpby_into, qr}`
  (`tenet-dense/src/tenferro_adapter.rs`) → Tenferro
  `CpuBackend::run_backend_session_cached` → `CpuOperationEntry::enter_managed_session`
  → `arbiter::with_execution_owner` → `CpuContext::install` → `ScopedJob::run`
  → `with_execution_resources`; inside, by-value `memmove`s of session structs,
  `Instant` reads (`mach_absolute_time`), `AllocationGroup::from_host_vec`,
  `validate_descriptor`, and on the way out `DenseTensor::into_f64_vec`.
- **dispatch (restrict)** — `tenet_tensors::oriented_fusion_restrict_into`
  (`tenet-tensors/src/oriented_elementwise.rs:113`) →
  `tensoradd_raw_strided_kernel_mapped` (`tenet-operations/src/host_scalar_kernels.rs:132`)
  → `raw_strided_combine_recurse_mapped` (`:1163`).

## Threads: default vs one thread

Median ratio to `one`, geometric mean over all rows (max):

| op | default | tenet1 (TeNeT Rayon = 1) | dense1 (Tenferro = 1) |
|---|---|---|---|
| compose | 2.93 (4.70) | 2.89 (4.52) | 1.00 (1.02) |
| contract | 2.23 (5.44) | 2.23 (5.22) | 1.00 (1.03) |
| permute | 1.30 (9.38) | 1.29 (9.47) | 1.00 (1.06) |
| repartition | 1.00 (1.05) | 1.00 (1.03) | 1.00 (1.10) |
| qr_compact | 3.03 (5.36) | 3.02 (5.28) | 1.00 (1.05) |
| restrict_leg | 1.00 (1.06) | 1.00 (1.04) | 1.00 (1.06) |
| scale / add / norm | 0.94–1.04 | 0.96–1.06 | 0.96–1.08 |

**Layer.** Pinning TeNeT's Rayon pool changes nothing; pinning only the
Tenferro CPU context removes the whole slowdown. The spawn/join is Tenferro's
per-session pool hop: `tenferro_cpu::context::CpuContext::install`
(`tenferro-cpu-0.5.0/src/context.rs:435` → `install_if_needed` `:375` →
Rayon `ThreadPool::install` `:370`) moves every backend session from the
calling thread onto a provider worker and blocks on a latch. The job itself runs
serially on one worker (worker busy time ≈ the one-thread kernel time); the
caller sleeps in `psynch_cvwait` under `install` for 73 % (compose), 85 % (qr)
and 71–95 % (SU(2) contract/permute/qr) of the default-thread samples (r4, f64).
Measured cost ≈ 10 µs per session (compose, one grouped GEMM session:
13.3 → 23.0 µs) and ≈ 12.6 µs per session for QR, which opens one session per
coupled sector (`r2_s8_d2`, 8 sectors: 23.6 → 125 µs). Operations with no
Tenferro session (repartition, restrict, scale/add/norm, abelian permute) are
unaffected; SU(2) permute enters Tenferro for its recoupling GEMM, hence the
9.4× maximum. TeNeT's own `rayon::join` replay schedule did not appear on any
default-thread stack at these sizes; the only fork seen is faer's internal
`join_context` inside Householder application for the largest SU(2) QR blocks,
on the Tenferro worker (Tenferro's context passes its parallelism to faer).

## Reference production dispatch

TensorKit (`~/.julia/packages/TensorKit/DQFb5/src`, rev `f87ca7fe`):

| op | production path |
|---|---|
| compose | `LinearAlgebra.mul!(tC, tA, tB)` (`tensors/linalg.jl:330`): sorted coupled-sector walk over `blocks`, one `mul!` per coupled sector. Same grouping as TeNeT (one GEMM job per coupled sector, TeNeT submits them as one grouped call). |
| contract | `TO.tensorcontract!` (`tensors/tensoroperations.jl:130`) → `blas_contract!` (`:392`): `isblascontractable` check, `tensoradd!` copy of each operand into BLAS form when needed (fermionic twist on the copied operand), then `mul!`. The permutation/partition logic is tuple arithmetic; no plan object is compiled per call. |
| permute | `permute` (`tensors/indexmanipulations.jl:242`) → `permute!` (`:217`) → `braid!` → `add_transform!` (`:554`) → `add_transform_kernel!` on the flat data vector with the `@cached` `treebraider` (`tensors/treetransformers.jl:162`; `AbelianTreeTransformer` / `GenericTreeTransformer`, `:655` / `:667`). |
| repartition | `repartition` (`:471`) → `repartition!` (`:442`) → `transpose!` with the `@cached` `treetransposer` (`treetransformers.jl:175`). |
| output structure | `similar(t, …, permute(space(t), p))`: the output `HomSpace` is an O(rank) tuple; its block layout comes from `@cached sectorstructure` / `degeneracystructure` (`spaces/structure.jl:41`, `:114`) keyed by the hashed `HomSpace`. |
| qr_compact | MatrixAlgebraKit `qr_compact!` generated in `factorizations/matrixalgebrakit.jl:35-47`: `foreachblock` → one dense `qr_compact!` per coupled sector. Same per-block granularity as TeNeT. |
| norm | `LinearAlgebra.norm(t::TensorMap)` (`tensors/linalg.jl:277`): `UniqueFusion` → `norm(t.data)` (one BLAS `nrm2` over the flat vector); otherwise per-block `_norm` with dimension weights. |
| scale / add | VectorInterface `scale` / `add` (`tensors/vectorinterface.jl:24`, `:67`) on the flat data vector. |
| restrict | no production path; the timed stand-in is an isometry contraction. |

QSpace (`reviews/reference-sources/QSpace-current`, `d2d3d7da6a59a2e8f2cb7dc8f33e7c345af59371`):
**no numbers.** The only MATLAB on this Mac is an R2026a trial whose licence
has expired (`matlab -batch` fails with MathWorks licence error −10.2), and no
`*.mexmaca64` build of QSpace exists anywhere under `libraries/`, so no QSpace
operation can run here. Production dispatch by source:

| op | production path |
|---|---|
| contract | `Class/@QSpace/contract.m` → `Source/contractQS.cc:mexFunction` (`:313`) → `QSpace<TQ,TD>::contract` (`Source/QSpace.cc:4155`): `permute_to` both operands (`Source/QSpace.hh:2994`), `contract_matchAB_groupC` (`QSpace.cc:4281`), then `contractDATA_group` per coupled group (`Source/QSpace_aux.cc:28`), OpenMP over groups when `QSP_NUM_THREADS > 1`. Each call also pays the MATLAB↔C++ `mxArray` conversion of both operands and the result. |
| permute | `Class/@QSpace/permute.m` → `Source/permuteQS.cc` → `QSpace<TQ,TD>::permute` (`Source/QSpace.hh:2891`). |
| QR | none: `orthoQS` (`Source/orthoQS.cc`) orthonormalizes through `SVD_Data::blockSVD` (`Source/mpsortho.hh:180`), block SVD, not QR. |
| restrict | none at degeneracy granularity: `getsub` (`Class/@QSpace/getsub.m`) selects whole records at MATLAB level. |
| norm / add / scale | `normQS.cc` → `QSpace<TQ,TD>::norm2` (`QSpace.cc:2258`); `plusQS.cc` → `QSpace<TQ,TD>::plus` (`QSpace.hh:1609`); scalar `mtimes` is a MATLAB loop over `data` cells (`Class/@QSpace/mtimes.m:28-37`). |

## Ranked avoidable constants

Ranked by the time they add at the small-shape floor, where TeNeT is behind
TensorKit. µs are f64, one thread, from the phase ledger.

1. **Tenferro pool hop per backend session under default threads — +10–13 µs
   per session.** Owner: `tenferro_cpu::context::CpuContext::install`
   (Tenferro 0.5.0), reached because the default `Runtime` builds its CPU
   context from the environment (`tenet/src/runtime.rs:1577-1581`,
   `SharedCpuContext::from_env`) and TeNeT enters one session per GEMM
   submission and one per coupled-sector QR. Structural cause: a
   cross-thread install whose fixed wake/latch cost does not depend on the
   work, applied to jobs that then run on one worker anyway. 2–5× on
   compose/contract/qr, 9.4× on SU(2) permute. TensorKit/QSpace: Julia
   `OhMyThreads` and QSpace OpenMP gate threading on work size
   (`use_threaded_transform`: `length(t.data) > Strided.MINTHREADLENGTH`,
   `indexmanipulations.jl:579`; QSpace `nc > 1 && QSP_NUM_THREADS > 1`).
2. **Per-call structure construction — 0.9–1.9 µs (permute, repartition,
   restrict), 1.2–1.4 µs small / 2.0–3.3 µs r4 (compose), 4.5–6.2 µs (contract: three spaces),
   up to 8.6 µs (SU(2) contract).** Owners: `transformed_with_primer`,
   `contracted_multiplicity_free_ordered`, `core_dst_with_primer`, all ending
   in `PreparedFusionTreeLayout::build_complete_from_leg_degeneracies`.
   Structural cause: the block layout of a result space is a pure function of
   (rule identity, interned hom space), but it is rebuilt — O(blocks × rank)
   work plus allocations — on every call. The runtime's per-context caches are
   set to `OperationCachePolicy::NoCache` (`tenet/src/runtime.rs:150,154,344`),
   so `DynamicFusionSpaceCache::get_or_compile_transformed_source` always takes
   its miss branch (`tenet-tensors/src/contract/dynamic.rs:2011-2033`), and
   the output spaces have no cache at all. TensorKit: O(rank) `HomSpace` tuple
   plus a hashed lookup of the `@cached` structure. This is the largest single
   TeNeT-owned constant and the main reason small `contract` is 3–3.6×
   TensorKit.
3. **Per-call contraction plan compilation — 2.2–2.7 µs (contract), 0.36–0.48
   µs (compose).** Owners listed under *plan* above. Structural cause: the axis
   plan, route, and core plan are recompiled from the operand spaces on every
   call; they depend only on (ranks, axis lists, output order, operand
   orientation) and on the already-interned structures. TensorKit does the
   same decisions as tuple arithmetic in `blas_contract!`.
4. **Eagerly constructed error values on the success path — 0.14–0.3 µs on
   small transforms/compose, 0.7–0.9 µs contract, 2.0 µs SU(2) contract/restrict
   (5–8 % of small operations).** `Option::ok_or(CoreError::…)` /
   `ok_or(OperationError::…)` builds an error of a type with drop glue before
   the check and drops it when the value is `Some`; the sampled
   `drop_in_place::<tenet_core::CoreError>` calls come mainly from
   `tenet_core::storage_end_exclusive` (`tenet-core/src/error.rs:374-377`),
   called per block from `DegeneracyBlock::new`, and the restrict loop
   (`oriented_elementwise.rs:162-182`). There are 346 `ok_or(<Error>::…)`
   sites in `tenet*/src`. Structural cause: an eager argument where a lazy one
   (`ok_or_else`) is equivalent.
5. **Tenferro session entry at one thread — ~1.0–1.3 µs per GEMM submission,
   ~2.0 µs per QR call.** Owner: Tenferro (`run_backend_session_cached`,
   `enter_managed_session`, `with_execution_owner`, `ScopedJob::run`, tensor
   wrapping and descriptor validation, `Instant` reads per session), plus
   TeNeT's QR result copy-out `DenseTensor::into_f64_vec`. QR pays it once per
   coupled sector: 16.5 µs of the 23.6 µs `r2_s8_d2` QR is backend entry for
   8 × (2×2) QRs whose kernel is 2.6 µs. TensorKit's per-block
   MatrixAlgebraKit call costs ~2.3 µs in total.
6. **restrict_leg gathers with a per-element checked scalar kernel — 11 µs for
   4864 elements (2.3 ns/element), 27 µs SU(2).** Owner:
   `oriented_fusion_restrict_into` → `tensoradd_raw_strided_kernel_mapped` →
   `raw_strided_combine_recurse_mapped`, which recomputes and range-checks both
   offsets for every element and applies an `alpha·src + beta·dst` action with
   `beta = 0`. A permute of the same payload through the strided-kernel copy
   costs 2.7 µs. Structural cause: the restriction is a per-block rectangular
   strided copy with bounds provable once per block. The output is also
   zero-filled before being fully overwritten (`tenet/src/typed.rs:14171`).
7. **Payload zero-fill before full overwrite — 0.6–1.1 µs at r4 (5–10 % of
   permute/repartition/compose).** `zeroed_payload` at
   `tenet/src/tensor_core.rs:434` (compose/contract), `vec![0; len]` in the
   tree-transform path (`tensor_core.rs:399`). Already owned by #1290.
8. **Complex64 abelian `norm` kernel — 1.3–1.7× TensorKit's BLAS `nrm2`** on
   the flat vector (f64 is 0.5×). Owner: the host reduction behind
   `TensorMap::norm`; overlaps #875 (route host inner reductions through the
   runtime backend).

Not constants: output allocation for owned results (by contract); the QR
per-block granularity (TensorKit and QSpace also factor per block); the
`compose` GEMM grouping (identical to TensorKit's per-coupled-sector `mul!`).

## Proposed follow-up leaves (one invariant each)

1. **TeNeT: result layouts are looked up, not rebuilt.** Invariant: a warm eager
   permute/repartition/restrict/compose/contract performs no
   `build_complete_from_leg_degeneracies` for a (rule, hom space) whose
   complete structure is already interned; asserted with a call counter
   (`observe_final_result_layout_build` exists) and bounded, keyed, resettable
   storage per policy. Constant 2.
2. **TeNeT: contraction plan work per call is O(rank) or reused.** Invariant: a
   warm `contract`/`compose` compiles no axis/route/core plan for an axis pattern
   and operand structure it has seen; counter-asserted. Constant 3.
3. **TeNeT: no eager error construction on hot success paths.** Invariant: zero
   `drop_in_place::<CoreError|OperationError>` calls on a successful warm
   eager operation; mechanical `ok_or` → `ok_or_else` on the listed paths plus
   the `clippy::or_fun_call` lint so it stays fixed. Constant 4.
4. **TeNeT: one Tenferro session per factorization phase.** Invariant:
   `qr_compact` (and the other compact factorizations) opens one CPU session per
   call regardless of the coupled-sector count, as #1283 did for grouped GEMM;
   `cpu_session_stats().sessions_opened` asserts it. Constants 1 and 5 (divides
   both by the block count).
5. **TeNeT: restrict/embed use the strided block copy.** Invariant:
   `restrict_leg`/`embed_leg` move each block with one bounds-checked-once
   strided copy (the kernel permute uses) and write an uninitialized
   destination; per-element cost within the permute kernel's on the same
   payload. Constant 6 (destination part with #1290).
6. **Thread policy for small dense phases (with Tenferro).** Invariant: under
   default threads a dense phase below a structural work gate runs on the
   calling thread, so no small eager operation is slower than at one thread.
   Needs a Tenferro capability (inline execution / work-size gate in
   `CpuContext::install`) — file upstream, mark this leaf blocked; the TeNeT
   side is only choosing the gate at the owning layer. Constant 1.
7. **Tenferro upstream (issue only): per-session entry cost.** Invariant: a
   one-thread CPU session entry costs O(1) small constant without by-value
   copies of large session structs or clock reads; QR returns into a
   caller-provided buffer. Constant 5. Candidate for the batched Tenferro
   wish-list.
8. **TeNeT: complex norm kernel parity.** Invariant: Complex64 abelian `norm`
   within TensorKit's `nrm2` time on the flat payload. Constant 8; fold into
   #875 if that issue owns the reduction path.

## Issue impact

- #1313: measurement and ledger delivered (**SOLVED** for the ledger acceptance;
  the invariant itself is **PARTIAL** — violated at the small-shape floor for
  compose/contract/permute/repartition/qr, with the causes above). QSpace
  numbers remain **BLOCKED** on a runnable MATLAB/QSpace build.
- #1290 (uninitialized writer): evidence for constant 7 and the restrict
  destination — **UNBLOCKED** by numbers, scope unchanged.
- #1291 (coalesce non-batchable GEMM runs): session entry is ~1–1.3 µs at one
  thread and ~10 µs at default threads, so each coalesced run saves that much —
  **PARTIAL** evidence.
- #875 (host reductions through the runtime backend): constant 8 — **PARTIAL**.
- #1287/#1288/#1140: consequence 1 of the B3 study (eager floor first) is now
  quantified; the prepared-handle design should wait for leaves 1–4, since
  constants 2–4 are eager-path work a handle would otherwise hide.

## Unsupported / residual

- QSpace numbers: not measured (licence/build absence above).
- Phase attribution was sampled for f64 only (both sizes, all symmetries,
  one and default threads for r4); Complex64 timing and allocation rows are in
  the CSVs. Phase shares are statistical (~±2 % of a row).
- TensorKit ran with OpenBLAS; its kernels differ from faer, so compare the
  floor (small cases) rather than kernel-bound rows for overhead.
- The fermionic `contract` in both libraries twists the dual contracted leg; the
  two are the same operation, but no value cross-check was part of this
  measurement leaf (correctness is covered by the existing oracles).
