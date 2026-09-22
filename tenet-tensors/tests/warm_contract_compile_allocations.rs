//! #1359: a warm eager contraction compiles its resolution without heap
//! allocations for rank-sized data.

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::sync::{Arc, Mutex};

use tenet_core::{FusionProductSpace, FusionTreeHomSpace, SectorLeg, U1FusionRule, U1Irrep};
use tenet_tensors::{
    prepare_tensorcontract_fusion_plan_dyn, try_compile_storage_contract_core_route,
    BoundDynamicFusionMapSpace, FusionOperand, OperationCachePolicy, OutputAxisOrder, RuleIdentity,
    RuntimeTreeTransformStore, TensorContractFusionExecutionContext, TensorContractSpec,
};

struct CountingAllocator;

thread_local! {
    static COUNTING: Cell<bool> = const { Cell::new(false) };
    static ALLOCATIONS: Cell<usize> = const { Cell::new(0) };
}

unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        if COUNTING.get() {
            ALLOCATIONS.set(ALLOCATIONS.get() + 1);
        }
        unsafe { System.alloc(layout) }
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        if COUNTING.get() {
            ALLOCATIONS.set(ALLOCATIONS.get() + 1);
        }
        unsafe { System.alloc_zeroed(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        if COUNTING.get() {
            ALLOCATIONS.set(ALLOCATIONS.get() + 1);
        }
        unsafe { System.realloc(ptr, layout, new_size) }
    }
}

#[global_allocator]
static GLOBAL: CountingAllocator = CountingAllocator;

// Why a lock: the complete-structure and tree-transform caches are process
// global, and a concurrent test could turn a warm lookup into a miss.
static SERIAL: Mutex<()> = Mutex::new(());

/// Allocations of one warm call, including the drop of its result.
fn warm_allocations<T>(mut call: impl FnMut() -> T) -> usize {
    drop(call());
    drop(call());
    ALLOCATIONS.set(0);
    COUNTING.set(true);
    drop(call());
    COUNTING.set(false);
    ALLOCATIONS.get()
}

type Space = BoundDynamicFusionMapSpace<U1FusionRule>;

fn leg(dual: bool) -> SectorLeg {
    SectorLeg::new(
        [
            (U1Irrep::new(-1).sector_id(), 2),
            (U1Irrep::new(0).sector_id(), 3),
            (U1Irrep::new(1).sector_id(), 2),
        ],
        dual,
    )
}

fn space(provider: &Arc<U1FusionRule>, codomain: usize, domain: usize) -> Space {
    Space::from_final_homspace_multiplicity_free_checked(
        Arc::clone(provider),
        FusionTreeHomSpace::new(
            FusionProductSpace::new((0..codomain).map(|_| leg(false))),
            FusionProductSpace::new((0..domain).map(|_| leg(false))),
        ),
    )
    .unwrap()
}

/// Runtime-configured context: no context-local space cache, tree
/// transforms from one shared store.
fn runtime_like_context(
    store: &Arc<RuntimeTreeTransformStore<f64>>,
) -> TensorContractFusionExecutionContext<f64, RuleIdentity> {
    let mut context = TensorContractFusionExecutionContext::<f64, RuleIdentity>::default();
    context.set_cache_policy(OperationCachePolicy::NoCache);
    context
        .tree_context_mut()
        .cache_mut()
        .bind_runtime_store(Arc::downgrade(store));
    context
}

/// `A(V^codomain ← V^domain)` composed with `S(V^domain ← V^domain)` in
/// three ways: the canonical Core route, the same contraction with its two
/// leading open axes swapped (a DynamicTree route whose source and output
/// transforms keep every leg on its side), and the plan alone.
fn warm_compile_allocations(codomain: usize, domain: usize) -> [usize; 3] {
    let provider = Arc::new(U1FusionRule);
    let lhs = space(&provider, codomain, domain);
    let square = space(&provider, domain, domain);
    let lhs_domain = (codomain..codomain + domain).collect::<Vec<_>>();
    let square_codomain = (0..domain).collect::<Vec<_>>();
    let composed = Space::contracted_multiplicity_free_ordered(
        &lhs,
        &square,
        &lhs_domain,
        &square_codomain,
        OutputAxisOrder::identity(),
    )
    .unwrap();
    let core = warm_allocations(|| {
        try_compile_storage_contract_core_route(
            &composed,
            FusionOperand::direct(lhs.space()),
            FusionOperand::direct(square.space()),
            TensorContractSpec::with_default_output_order(&lhs_domain, &square_codomain),
        )
        .unwrap()
        .expect("canonical composition takes the core route")
    });

    let mut swapped = (0..codomain + domain).collect::<Vec<_>>();
    swapped.swap(0, 1);
    let axes = || {
        TensorContractSpec::new(
            &lhs_domain,
            &square_codomain,
            OutputAxisOrder::from_axes(&swapped),
        )
    };
    let dst = Space::contracted_multiplicity_free_ordered(
        &lhs,
        &square,
        &lhs_domain,
        &square_codomain,
        OutputAxisOrder::from_axes(&swapped),
    )
    .unwrap();
    let store = Arc::new(RuntimeTreeTransformStore::new(
        RuntimeTreeTransformStore::<f64>::DEFAULT_BYTE_BUDGET,
    ));
    let mut context = runtime_like_context(&store);
    let dynamic_tree = warm_allocations(|| {
        let resolution = context
            .compile_storage_contract_resolution(
                &dst,
                FusionOperand::direct(lhs.space()),
                FusionOperand::direct(square.space()),
                axes(),
            )
            .unwrap();
        assert!(resolution.is_dynamic_tree());
        resolution
    });
    let plan = warm_allocations(|| {
        prepare_tensorcontract_fusion_plan_dyn(&dst, &lhs, &square, axes()).unwrap()
    });
    [core, dynamic_tree, plan]
}

#[test]
fn warm_contract_compile_allocations_do_not_scale_with_rank() {
    let _serial = SERIAL
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    for (codomain, domain) in [(2, 1), (2, 2), (3, 2), (3, 3), (4, 3)] {
        let [core, dynamic_tree, plan] = warm_compile_allocations(codomain, domain);
        let rank = codomain + domain;
        // What: the same allocation count at every rank from 3 to 7.
        // Core: the plan's job, coefficient, run and inactive-region lists
        // plus its `Arc`. Plan: one shared slice per transform operation.
        // DynamicTree: that plan and its `Arc`; per transformed source its
        // permuted HomSpace, the layout lookup key (#1367), and the two
        // `Arc`s the replay scratch shares; the same for the core
        // destination; the core plan; the empty twist list; the artifact.
        assert_eq!(core, 5, "core route at rank {rank}");
        assert_eq!(plan, 3, "plan at rank {rank}");
        assert_eq!(dynamic_tree, 25, "dynamic-tree route at rank {rank}");
    }
}

/// `A(V^codomain ← V^domain)` contracted on its codomain axis 0 with
/// `M(V ← V)` on its domain axis, identity output (the E1 `contract` row):
/// A's source transform sends leg 0 to the domain and its `domain` legs to
/// the codomain, and M's transform swaps its two legs.
#[test]
fn warm_contract_compile_allocates_once_per_leg_that_changes_side() {
    let _serial = SERIAL
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    // Rank 2 is left out: there the planner takes the reversed orientation,
    // whose source transforms keep every leg on its side.
    for (codomain, domain) in [(2, 1), (1, 2), (2, 2), (3, 2), (2, 3), (3, 3)] {
        let provider = Arc::new(U1FusionRule);
        let lhs = space(&provider, codomain, domain);
        let matrix = space(&provider, 1, 1);
        let open = (0..codomain + domain).collect::<Vec<_>>();
        let axes = || TensorContractSpec::new(&[0], &[1], OutputAxisOrder::from_axes(&open));
        let dst = Space::contracted_multiplicity_free_ordered(
            &lhs,
            &matrix,
            &[0],
            &[1],
            OutputAxisOrder::from_axes(&open),
        )
        .unwrap();
        let store = Arc::new(RuntimeTreeTransformStore::new(
            RuntimeTreeTransformStore::<f64>::DEFAULT_BYTE_BUDGET,
        ));
        let mut context = runtime_like_context(&store);
        let allocations = warm_allocations(|| {
            context
                .compile_storage_contract_resolution(
                    &dst,
                    FusionOperand::direct(lhs.space()),
                    FusionOperand::direct(matrix.space()),
                    axes(),
                )
                .unwrap()
        });
        // What: the rank-independent DynamicTree compile (20 here: no core
        // destination, the output transform is the identity) plus exactly one
        // allocation per leg that changes side in a source transform. That
        // one is `SectorLeg::dual` building the dual leg's sector data while
        // the permuted HomSpace is formed; TensorKit's `dual(V)` shares the
        // sector data instead, and removing it is its own leaf, not rank
        // arithmetic this test could forbid.
        let crossing_legs = (1 + domain) + 2;
        assert_eq!(
            allocations,
            20 + crossing_legs,
            "rank {}",
            codomain + domain
        );
    }
}
