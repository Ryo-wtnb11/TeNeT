# Adjoint block structure memoized on its space (#1368)

Date 2026-09-24. Host: Apple M4 Max, macOS 15.5, rustc 1.96.0.

- **Before:** `origin/main` `e354d10d` plus this change's E1 rows only.
- **After:** branch `adjoint-structure-memo`.
- **Build:** both revisions built one after the other into separate target
  directories with `CARGO_PROFILE_RELEASE_DEBUG=line-tables-only cargo build
  --release -p tenet-rs --example eager_overhead_ledger`. Binaries (sha256
  prefix): before `ae9c9031ce243ae3`, after `05b0eacafa04b3b0`.
- **Method:** E1 with the filters `contract`, `contract_conj`, `compose_conj`,
  `LEDGER_THREADS=one`, `nice -n 19`, two passes ordered before, after, before,
  after. Allocation counts matched between passes. Medians are the mean of the
  two passes. Raw pass-1 rows: `adjoint-structure-memo-2026-09-24-{before,after}.csv`.
- Timings are observations only.

## Change

`DynamicFusionMapSpace` holds `OnceLock<Result<(Arc<FusionTreeHomSpace>,
Arc<BlockStructure>), Box<OperationError>>>`. `adjoint_view` fills it on first
use and afterwards clones two `Arc`s. The slot is excluded from `PartialEq` and
from the `ValidatedDynamicFusionLayout` hash. Admission is still read at call
time.

## Which paths this reaches

The rebuild in `lowering::adjoint_block_structure_view` runs only through
`DynamicFusionMapSpace::adjoint_view`. On `e354d10d` that function is called by
the expert conjugated-source contraction (`TensorContractSpec` with a conjugate
flag: `get_or_compile_transformed_source`, before its cache lookup, and plan
preparation in `contract/fusion/plan.rs` and `block_specs.rs`), and through
`BoundDynamicFusionMapSpace::adjoint_view`.

The public eager `TensorMap::contract`/`compose` with a lazy `adjoint()`
operand does **not** call it. Those calls take the prelowered `FusionOperand`
route. A `/usr/bin/sample` of the E1 `contract_conj` row on `e354d10d` shows no
`adjoint_view` or `BlockStructure::from_blocks_with_rank` frame. So the E1 rows
below show no change from this leaf, apart from the inline slot's bytes.

## Adjoint-structure builds (`ADJOINT_VIEW_BUILDS`)

For 5 warm expert conjugated contractions `conj(A)·B` on one space, over U(1),
SU(2) and fZ2×U(1), for both f64 and c64:

| revision | builds |
|---|---:|
| before | 11 (two per call plus one while preparing) |
| after | 1 |

## Expert conjugated contraction, warm allocations

`tenet-tensors/tests/conjugated_contract_allocations.rs`, `A, B: V ← V` with
three sectors, `tensorcontract_fusion_dyn_into` with the lhs conjugated,
default context. The counts are the same at degeneracy 2 and 16.

| symmetry | before calls / bytes | after calls / bytes |
|---|---:|---:|
| U1 | 49 / 14088 | 11 / 1376 |
| SU2 | 188 / 11268 | 168 / 4616 |
| fZ2×U1 | 49 / 14088 | 11 / 1376 |

## E1 (facade), f64, selected rows

Calls and bytes per warm call; the median is in µs. The rows not shown behave
the same way, including every c64 row.

| op | symmetry | case | before calls / B | after calls / B | before med | after med |
|---|---|---|---:|---:|---:|---:|
| contract_conj | U1 | r2_s8_d2 | 474 / 61026 | 474 / 61098 | 20.25 | 20.29 |
| contract_conj | U1 | r2_s8_d16 | 474 / 77154 | 474 / 77226 | 25.31 | 24.42 |
| contract_conj | U1 | r4_s3_d4 | 1879 / 554447 | 1879 / 554519 | 131.79 | 131.10 |
| contract_conj | SU2 | r2_s8_d2 | 474 / 60498 | 474 / 60570 | 22.31 | 21.52 |
| contract_conj | SU2 | r4_s3_d4 | 3599 / 1378438 | 3599 / 1378510 | 336.04 | 334.08 |
| contract_conj | fZ2xU1 | r2_s8_d2 | 484 / 61794 | 484 / 61866 | 24.35 | 23.67 |
| contract_conj | fZ2xU1 | r4_s3_d4 | 1886 / 555023 | 1886 / 555095 | 139.31 | 137.79 |
| compose_conj | U1 | r2_s8_d2 | 16 / 1966 | 16 / 1990 | 2.68 | 2.66 |
| compose_conj | SU2 | r4_s3_d4 | 24 / 283579 | 24 / 283603 | 36.17 | 36.35 |
| compose_conj | fZ2xU1 | r4_s3_d4 | 25 / 114100 | 25 / 114124 | 16.33 | 16.19 |
| contract | U1 | r2_s8_d2 | 27 / 4320 | 27 / 4416 | 5.60 | 6.41 |
| contract | SU2 | r4_s3_d4 | 37 / 146512 | 37 / 146584 | 24.31 | 24.46 |
| contract | fZ2xU1 | r4_s3_d4 | 36 / 91600 | 36 / 91672 | 10.50 | 10.38 |

Allocation calls are unchanged on every E1 row. Bytes rise by 24 B for each
`DynamicFusionMapSpace` a call allocates (+24 to +96 B), because every space is
now 24 bytes larger.

E1 `contract_conj` is still 17–97× the owned `contract` in allocation
calls. The sample puts that cost in
`TreeTransformCache::get_or_compile_tree_pair_oriented` (`tree_transform/cache.rs`),
which recompiles the oriented source transform plan on every call without
consulting the runtime transform store. That is a separate defect, and this
leaf does not address it.

## Memory

- Filled memo, 3 blocks of rank 2: 3862 B charged
  (`BlockStructure::charged_retained_bytes` of the adjoint structure), plus
  one adjoint `FusionTreeHomSpace`. The memo is paid only by spaces that are
  ever adjointed and freed with the space.
- Unfilled slot: 24 B inline in every `DynamicFusionMapSpace`.
