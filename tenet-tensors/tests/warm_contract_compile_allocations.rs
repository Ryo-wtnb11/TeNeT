//! #1359: a warm eager contraction compiles its resolution without heap
//! allocations for rank-sized data.

use std::sync::Arc;

use tenet_core::{FusionProductSpace, FusionTreeHomSpace, SectorLeg, U1FusionRule, U1Irrep};
use tenet_tensors::{
    prepare_tensorcontract_fusion_plan_dyn, try_compile_storage_contract_core_route,
    BoundDynamicFusionMapSpace, DirectCoreExecutor, FusionOperand, OperationCachePolicy,
    OutputAxisOrder, RuleIdentity, RuntimeTreeTransformStore, StorageContractResolution,
    TensorContractFusionExecutionContext, TensorContractSpec,
};

#[path = "../../tests/support/counting_alloc.rs"]
mod counting_alloc;

#[global_allocator]
static ALLOCATOR: counting_alloc::CountingAllocator = counting_alloc::CountingAllocator;

/// Allocations of one warm call, including the drop of its result.
fn warm_allocations<T>(mut call: impl FnMut() -> T) -> usize {
    drop(call());
    drop(call());
    counting_alloc::start();
    drop(call());
    let allocs = counting_alloc::stop();
    allocs.calls as usize
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

/// Same sectors with a degeneracy profile the `q -> -q` dual does not fix, so
/// the dual leg's map differs from the source leg's.
fn asymmetric_leg(dual: bool) -> SectorLeg {
    SectorLeg::new(
        [
            (U1Irrep::new(-1).sector_id(), 2),
            (U1Irrep::new(0).sector_id(), 3),
            (U1Irrep::new(1).sector_id(), 4),
        ],
        dual,
    )
}

fn space(provider: &Arc<U1FusionRule>, codomain: usize, domain: usize) -> Space {
    space_of(leg, provider, codomain, domain)
}

fn space_of(
    leg: fn(bool) -> SectorLeg,
    provider: &Arc<U1FusionRule>,
    codomain: usize,
    domain: usize,
) -> Space {
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

/// The storage ladder these rows pin: the lock-free canonical core, then the
/// `DynamicTree` artifact (the planner's `CopyC` rung is not measured here).
fn storage_ladder(
    context: &mut TensorContractFusionExecutionContext<f64, RuleIdentity>,
    dst: &Space,
    lhs: FusionOperand<'_>,
    rhs: FusionOperand<'_>,
    axes: TensorContractSpec<'_>,
) -> StorageContractResolution<f64> {
    match try_compile_storage_contract_core_route::<DirectCoreExecutor, _>(dst, lhs, rhs, axes)
        .unwrap()
        .hit()
    {
        Some(core) => core,
        None => context
            .compile_storage_contract_dynamic_tree(dst, lhs, rhs, axes)
            .unwrap(),
    }
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
        try_compile_storage_contract_core_route::<DirectCoreExecutor, _>(
            &composed,
            FusionOperand::direct(lhs.space()),
            FusionOperand::direct(square.space()),
            TensorContractSpec::with_default_output_order(&lhs_domain, &square_codomain),
        )
        .unwrap()
        .hit()
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
        let resolution = storage_ladder(
            &mut context,
            &dst,
            FusionOperand::direct(lhs.space()),
            FusionOperand::direct(square.space()),
            axes(),
        );
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
    let _serial = counting_alloc::serial();
    for (codomain, domain) in [(2, 1), (2, 2), (3, 2), (3, 3), (4, 3)] {
        let [core, dynamic_tree, plan] = warm_compile_allocations(codomain, domain);
        let rank = codomain + domain;
        // What: the same allocation count at every rank from 3 to 7.
        // Core: the plan's job, coefficient, run and inactive-region lists
        // plus its `Arc`. Plan: one shared slice per transform operation.
        // DynamicTree: that plan and its `Arc`; per transformed source its
        // permuted HomSpace and the two `Arc`s the replay scratch shares; the
        // same for the core destination; the core plan; the empty twist list;
        // the artifact. The layout lookup key is no longer one of them
        // (#1367): it borrows the HomSpace content.
        assert_eq!(core, 5, "core route at rank {rank}");
        assert_eq!(plan, 3, "plan at rank {rank}");
        assert_eq!(dynamic_tree, 19, "dynamic-tree route at rank {rank}");
    }
}

/// Allocations of one warm DynamicTree compile of
/// `A(V^codomain ← V^domain)` contracted on its codomain axis 0 with
/// `M(V ← V)` on its domain axis, identity output (the E1 `contract` row):
/// A's source transform sends leg 0 to the domain and its `domain` legs to
/// the codomain, and M's transform swaps its two legs.
fn crossing_compile_allocations(
    leg: fn(bool) -> SectorLeg,
    codomain: usize,
    domain: usize,
) -> usize {
    let provider = Arc::new(U1FusionRule);
    let lhs = space_of(leg, &provider, codomain, domain);
    let matrix = space_of(leg, &provider, 1, 1);
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
    warm_allocations(|| {
        storage_ladder(
            &mut context,
            &dst,
            FusionOperand::direct(lhs.space()),
            FusionOperand::direct(matrix.space()),
            axes(),
        )
    })
}

/// The route's own allocation count. Every shape but (1, 2) dualizes
/// `(1 + domain) + 2` legs while forming permuted HomSpaces.
fn crossing_compile_base(codomain: usize, domain: usize) -> usize {
    if (codomain, domain) == (1, 2) {
        19
    } else {
        16
    }
}

#[test]
fn warm_contract_compile_allocates_nothing_per_leg_that_changes_side() {
    let _serial = counting_alloc::serial();
    // Rank 2 is left out: there the planner takes the reversed orientation,
    // whose source transforms keep every leg on its side.
    for (codomain, domain) in [(2, 1), (1, 2), (2, 2), (3, 2), (2, 3), (3, 3)] {
        let allocations = crossing_compile_allocations(leg, codomain, domain);
        // What: the rank-independent DynamicTree compile, and nothing per leg
        // that changes side in a source transform. #1403 removed that
        // per-leg allocation: forming the permuted HomSpace dualizes
        // `(1 + domain) + 2` legs, and `SectorLeg::dual` now shares the
        // source leg's sector data, as TensorKit's `dual(V)` shares `V.dims`.
        //
        // Why shape (1, 2) is apart: its route forms no permuted HomSpace
        // with a crossing leg, so it never dualizes one.
        let base = crossing_compile_base(codomain, domain);
        assert_eq!(allocations, base, "rank {}", codomain + domain);
    }
}

/// #1403: a leg whose sector -> degeneracy map the dual moves shares its
/// storage too, so it allocates nothing per crossing leg either.
#[test]
fn warm_contract_compile_allocates_nothing_per_crossing_leg_the_dual_moves() {
    let _serial = counting_alloc::serial();
    for (codomain, domain) in [(2, 1), (1, 2), (2, 2), (3, 2), (2, 3), (3, 3)] {
        let moved = crossing_compile_allocations(asymmetric_leg, codomain, domain);
        let fixed = crossing_compile_allocations(leg, codomain, domain);
        let base = crossing_compile_base(codomain, domain);
        // What: the same count as legs the dual fixes, and as the route with
        // no crossing leg.
        assert_eq!(moved, fixed, "rank {}", codomain + domain);
        assert_eq!(moved, base, "rank {}", codomain + domain);
    }
}

/// Allocations of one warm `plan_contract` of `A(V^rank ← V^rank)` with
/// `B(V^rank ← V^rank)` on A's domain and B's codomain, both reversed, into
/// each operand's open legs reversed: TensorKit's `copyC` shape (C1p at rank
/// 2), so the planner builds the temporary space, its core and one permute.
/// Paired with the same contraction through the core-then-`DynamicTree`
/// ladder, the device route before #1857.
fn copy_c_plan_allocations(rank: usize) -> [usize; 2] {
    let provider = Arc::new(U1FusionRule);
    let lhs = space(&provider, rank, rank);
    let rhs = space(&provider, rank, rank);
    let lhs_axes = (rank..2 * rank).rev().collect::<Vec<_>>();
    let rhs_axes = (0..rank).rev().collect::<Vec<_>>();
    let output = (0..rank)
        .rev()
        .chain((rank..2 * rank).rev())
        .collect::<Vec<_>>();
    let dst = Space::contracted_multiplicity_free_partitioned(
        &lhs,
        &rhs,
        &lhs_axes,
        &rhs_axes,
        OutputAxisOrder::from_axes(&output),
        rank,
    )
    .unwrap();
    let store = Arc::new(RuntimeTreeTransformStore::new(
        RuntimeTreeTransformStore::<f64>::DEFAULT_BYTE_BUDGET,
    ));
    let mut context = runtime_like_context(&store);
    let copy_c = warm_allocations(|| {
        let resolution = context
            .plan_contract::<DirectCoreExecutor, _>(
                &dst,
                FusionOperand::direct(lhs.space()),
                FusionOperand::direct(rhs.space()),
                &lhs_axes,
                &rhs_axes,
                &output,
            )
            .unwrap();
        assert!(resolution.copy_c().is_some(), "rank {rank} takes CopyC");
        resolution
    });
    let dynamic_tree = warm_allocations(|| {
        let resolution = storage_ladder(
            &mut context,
            &dst,
            FusionOperand::direct(lhs.space()),
            FusionOperand::direct(rhs.space()),
            TensorContractSpec::new(&lhs_axes, &rhs_axes, OutputAxisOrder::from_axes(&output)),
        );
        assert!(resolution.is_dynamic_tree(), "rank {rank}");
        resolution
    });
    // #1993: every structure this fixture builds is retained; the old 8 MiB
    // per-entry limit bypassed the rank-5 transform and recompiled it per call.
    for info in [store.info(), store.plan_info(), store.group_info()] {
        assert_eq!(info.admission_bypasses(), 0, "rank {rank}: {info:?}");
    }
    [copy_c, dynamic_tree]
}

#[test]
fn warm_copy_c_planning_is_bounded_and_no_costlier_than_the_dynamic_tree() {
    let _serial = counting_alloc::serial();
    // Rank 6 is the #1998 acceptance: its complete HomSpace (formerly
    // 111 MB), its transform (formerly 249 MB) and its fusion-tree layouts
    // used to exceed the 64 MiB budgets or the 8 MiB entry limits, so each
    // warm call rebuilt them (#1993).
    for rank in [2, 3, 4, 5, 6] {
        let complete_bypasses = tenet_core::complete_hom_space_structure_cache_info().bypasses();
        let layout_bypasses = tenet_core::fusion_tree_layout_cache_info().admission_bypasses();
        let intern_bypasses =
            tenet_core::block_structure_intern_cache_info().oversized_admission_bypasses();
        let [copy_c, dynamic_tree] = copy_c_plan_allocations(rank);
        eprintln!("rank {rank}: warm CopyC plan {copy_c}, DynamicTree ladder {dynamic_tree}");
        // What: device-eager CopyC planning stays a small bounded count once
        // its structures are warm (9-10 through rank 4, 18 at ranks 5 and 6
        // where rank-sized inline axis vectors spill), and never above the
        // DynamicTree compile it replaced for this contraction. Rank 5 was
        // 850,760 before #1993; rank 6 was 23,467 before #1998.
        assert!(copy_c <= 20, "rank {rank}: {copy_c}");
        assert!(
            copy_c <= dynamic_tree,
            "rank {rank}: {copy_c} > {dynamic_tree}"
        );
        assert_eq!(
            tenet_core::complete_hom_space_structure_cache_info().bypasses(),
            complete_bypasses,
            "rank {rank}: a complete HomSpace structure bypassed the cache"
        );
        assert_eq!(
            tenet_core::fusion_tree_layout_cache_info().admission_bypasses(),
            layout_bypasses,
            "rank {rank}: a fusion-tree layout bypassed the cache"
        );
        assert_eq!(
            tenet_core::block_structure_intern_cache_info().oversized_admission_bypasses(),
            intern_bypasses,
            "rank {rank}: a block-structure content bypassed the interner"
        );
    }
}

#[test]
fn rank_six_block_structure_content_is_interned_across_derivations() {
    let _serial = counting_alloc::serial();
    // What (#1998): the U(1) `V^6 <- V^6` content (73,789 blocks) is
    // interned, so equal derivations share one content id while it is live.
    // Before #1998 its copied intern key exceeded the 8 MiB entry limit and
    // every derivation minted a new id, missing every id-keyed cache.
    let provider = Arc::new(U1FusionRule);
    let structure = Arc::clone(space(&provider, 6, 6).space().structure());
    let bypasses = tenet_core::block_structure_intern_cache_info().oversized_admission_bypasses();
    let derive = || {
        tenet_core::BlockStructure::from_parts(
            structure.sector_structure().clone(),
            structure.degeneracy_structure().clone(),
        )
        .unwrap()
    };
    let first = derive();
    let second = derive();
    assert_eq!(first.content_id(), structure.content_id());
    assert_eq!(second.content_id(), structure.content_id());
    assert_eq!(
        tenet_core::block_structure_intern_cache_info().oversized_admission_bypasses(),
        bypasses
    );
}
