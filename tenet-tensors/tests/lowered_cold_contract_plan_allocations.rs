//! #586 sweep guard: cold lowered enumeration + cold contraction plan-build
//! allocation, measured relatively (no absolute byte pins — platform
//! allocators differ; see #611 lesson on frozen bit pins).
//!
//! Two properties are pinned:
//! 1. The genuine cold mechanism (`prepare_fusion_tree_layout_lowered`)
//!    stays streaming — enumeration bytes track the admitted key count, not
//!    the candidate sector product — and builds the layout exactly once
//!    (a committed layout makes the second prepare a cache lookup).
//! 2. The erased-contract fusion route's cold plan build: the E1 baseline.
//!    While both context entries exist, the `_lowered` delegate and the plain
//!    twin must be allocation- and cache-statistics-identical; after E1 the
//!    plain entry is the production route and this test is its guard.

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::sync::Arc;

use tenet_core::{FusionProductSpace, FusionTreeHomSpace, SectorLeg, U1FusionRule, U1Irrep};
use tenet_tensors::{
    reset_global_operation_caches, BoundDynamicFusionMapSpace, OperationCachePolicy,
    OutputAxisOrder, RuleIdentity, TensorContractFusionExecutionContext, TensorContractSpec,
};

struct CountingAllocator;

thread_local! {
    static COUNTING: Cell<bool> = const { Cell::new(false) };
    static ALLOCATIONS: Cell<usize> = const { Cell::new(0) };
    static BYTES: Cell<usize> = const { Cell::new(0) };
}

unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        if COUNTING.with(Cell::get) {
            ALLOCATIONS.with(|count| count.set(count.get() + 1));
            BYTES.with(|bytes| bytes.set(bytes.get() + layout.size()));
        }
        unsafe { System.alloc(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }
}

#[global_allocator]
static GLOBAL: CountingAllocator = CountingAllocator;

fn measured<T>(operation: impl FnOnce() -> T) -> (T, usize, usize) {
    ALLOCATIONS.set(0);
    BYTES.set(0);
    COUNTING.set(true);
    let value = operation();
    COUNTING.set(false);
    (value, BYTES.with(Cell::get), ALLOCATIONS.with(Cell::get))
}

fn reset_all_caches() {
    reset_global_operation_caches();
    tenet_core::reset_core_intern_tables();
}

fn chain_homspace(sector_count: i32) -> FusionTreeHomSpace {
    let sectors = (0..sector_count)
        .map(|charge| (U1Irrep::new(charge).sector_id(), 1))
        .collect::<Vec<_>>();
    FusionTreeHomSpace::new(
        FusionProductSpace::new([SectorLeg::new(sectors.clone(), false)]),
        FusionProductSpace::new([SectorLeg::new(sectors, false)]),
    )
}

#[test]
fn cold_lowered_enumeration_streams_and_builds_once() {
    // Streaming: admitted keys grow linearly with the sector count while the
    // candidate space (codomain x domain sector pairs) grows quadratically.
    // An eager whole-table regression would push per-key bytes at N=64 far
    // above per-key bytes at N=8; the streaming path keeps them comparable.
    let per_key_bytes = |sector_count: i32| {
        reset_all_caches();
        let homspace = chain_homspace(sector_count);
        let (prepared, bytes, _) = measured(|| {
            homspace
                .prepare_fusion_tree_layout_lowered(&U1FusionRule)
                .unwrap()
        });
        let keys = prepared.commit();
        assert_eq!(keys.len(), sector_count as usize);
        bytes as f64 / keys.len() as f64
    };
    let small = per_key_bytes(8);
    let large = per_key_bytes(64);
    assert!(
        large <= 2.0 * small,
        "cold lowered enumeration regressed toward eager whole-table \
         materialization: {large:.1} bytes/key at N=64 vs {small:.1} at N=8"
    );

    // Builds-once: with the cold layout committed, the second prepare is a
    // cache lookup, not a rebuild.
    reset_all_caches();
    let homspace = chain_homspace(16);
    let (first, cold_bytes, _) = measured(|| {
        homspace
            .prepare_fusion_tree_layout_lowered(&U1FusionRule)
            .unwrap()
    });
    let cold_keys = first.commit();
    let (second, warm_bytes, _) = measured(|| {
        homspace
            .prepare_fusion_tree_layout_lowered(&U1FusionRule)
            .unwrap()
    });
    let warm_keys = second.commit();
    assert!(Arc::ptr_eq(&cold_keys, &warm_keys));
    assert!(
        warm_bytes * 8 <= cold_bytes,
        "second lowered prepare rebuilt the layout: warm {warm_bytes} bytes \
         vs cold {cold_bytes} bytes"
    );
}

struct ContractFixture {
    dst: BoundDynamicFusionMapSpace<U1FusionRule>,
    lhs: BoundDynamicFusionMapSpace<U1FusionRule>,
    rhs: BoundDynamicFusionMapSpace<U1FusionRule>,
    lhs_data: Vec<f64>,
    rhs_data: Vec<f64>,
}

/// Rank-3 x rank-3 U(1) contraction with a non-identity output order: the
/// erased facade's fusion route (tree-transform plan build), not the
/// direct-core fast path.
fn contract_fixture() -> ContractFixture {
    let provider = Arc::new(U1FusionRule);
    let leg = |dual| {
        SectorLeg::new(
            [
                (U1Irrep::new(0).sector_id(), 2),
                (U1Irrep::new(1).sector_id(), 2),
            ],
            dual,
        )
    };
    let lhs_hom = FusionTreeHomSpace::new(
        FusionProductSpace::new([leg(false), leg(true)]),
        FusionProductSpace::new([leg(false)]),
    );
    let rhs_hom = FusionTreeHomSpace::new(
        FusionProductSpace::new([leg(true)]),
        FusionProductSpace::new([leg(true), leg(false)]),
    );
    let lhs = BoundDynamicFusionMapSpace::from_final_homspace_multiplicity_free_lowered(
        Arc::clone(&provider),
        lhs_hom,
    )
    .unwrap();
    let rhs = BoundDynamicFusionMapSpace::from_final_homspace_multiplicity_free_lowered(
        Arc::clone(&provider),
        rhs_hom,
    )
    .unwrap();
    let dst = BoundDynamicFusionMapSpace::contracted_multiplicity_free_ordered(
        &lhs,
        &rhs,
        &[0],
        &[2],
        OutputAxisOrder::from_axes(&[2, 0, 3, 1]),
    )
    .unwrap();
    let lhs_data = (0..lhs.space().required_len().unwrap())
        .map(|index| index as f64 + 1.0)
        .collect::<Vec<_>>();
    let rhs_data = (0..rhs.space().required_len().unwrap())
        .map(|index| 0.5 * index as f64 - 2.0)
        .collect::<Vec<_>>();
    ContractFixture {
        dst,
        lhs,
        rhs,
        lhs_data,
        rhs_data,
    }
}

struct RouteRun {
    output: Vec<f64>,
    cold_bytes: usize,
    cold_allocations: usize,
    warm_bytes: usize,
    cold_misses: usize,
    cold_hits: usize,
    warm_misses: usize,
    warm_hits: usize,
}

fn run_route(lowered_entry: bool) -> RouteRun {
    reset_all_caches();
    let fixture = contract_fixture();
    let axes = || TensorContractSpec::new(&[0], &[2], OutputAxisOrder::from_axes(&[2, 0, 3, 1]));
    let mut context = TensorContractFusionExecutionContext::<f64, RuleIdentity>::default();
    context.set_cache_policy(OperationCachePolicy::TaskLocal);
    let mut output = vec![0.0; fixture.dst.space().required_len().unwrap()];
    let mut run =
        |output: &mut [f64],
         context: &mut TensorContractFusionExecutionContext<f64, RuleIdentity>| {
            // ponytail: the lowered arm dies with the twin in the sweep's removal
            // commit; the plain arm then remains as the production-route guard.
            if lowered_entry {
                context
                    .tensorcontract_fusion_dyn_into_lowered(
                        &fixture.dst,
                        output,
                        &fixture.lhs,
                        &fixture.lhs_data,
                        &fixture.rhs,
                        &fixture.rhs_data,
                        axes(),
                        1.0,
                        0.0,
                    )
                    .unwrap();
            } else {
                context
                    .tensorcontract_fusion_dyn_into(
                        &fixture.dst,
                        output,
                        &fixture.lhs,
                        &fixture.lhs_data,
                        &fixture.rhs,
                        &fixture.rhs_data,
                        axes(),
                        1.0,
                        0.0,
                    )
                    .unwrap();
            }
        };
    let ((), cold_bytes, cold_allocations) = measured(|| run(&mut output, &mut context));
    let cold_misses = context.dynamic_fusion_space_cache_misses();
    let cold_hits = context.dynamic_fusion_space_cache_hits();
    let mut warm_output = vec![0.0; output.len()];
    let ((), warm_bytes, _) = measured(|| run(&mut warm_output, &mut context));
    assert_eq!(warm_output, output);
    RouteRun {
        output,
        cold_bytes,
        cold_allocations,
        warm_bytes,
        cold_misses,
        cold_hits,
        warm_misses: context.dynamic_fusion_space_cache_misses(),
        warm_hits: context.dynamic_fusion_space_cache_hits(),
    }
}

#[test]
fn cold_contract_plan_build_parity_between_lowered_and_plain_entries() {
    // Discard one run: the first contraction in the process pays a one-time
    // warmup (lazy statics, thread-local init) that is route-independent and
    // would otherwise skew whichever arm runs first.
    let _ = run_route(false);
    let lowered = run_route(true);
    let plain = run_route(false);

    // The fixture must exercise the fusion plan-build route, not the
    // direct-core fast path: a cold run misses the dynamic fusion space
    // cache, a warm run hits it without new misses.
    assert!(lowered.cold_misses >= 3, "fixture took the core fast path");
    assert_eq!(lowered.warm_misses, lowered.cold_misses);
    assert!(lowered.warm_hits > lowered.cold_hits);
    assert!(
        lowered.warm_bytes < lowered.cold_bytes,
        "warm run rebuilt the cold plan: warm {} bytes vs cold {} bytes",
        lowered.warm_bytes,
        lowered.cold_bytes
    );

    // E1 pass bar: exact public output equality plus no additional cold
    // layout/plan-build allocation relative to the lowered baseline.
    assert_eq!(plain.output, lowered.output);
    assert_eq!(plain.cold_misses, lowered.cold_misses);
    assert_eq!(plain.cold_hits, lowered.cold_hits);
    assert_eq!(plain.warm_misses, lowered.warm_misses);
    assert_eq!(plain.warm_hits, lowered.warm_hits);
    assert_eq!(
        (plain.cold_bytes, plain.cold_allocations),
        (lowered.cold_bytes, lowered.cold_allocations),
        "plain entry changed cold plan-build allocation vs lowered baseline"
    );
    assert_eq!(plain.warm_bytes, lowered.warm_bytes);
}
