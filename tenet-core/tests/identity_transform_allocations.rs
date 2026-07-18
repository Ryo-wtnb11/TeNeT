use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::hint::black_box;

use tenet_core::{
    multiplicity_free_braid_tree_block, multiplicity_free_permute_tree_pair_block,
    unique_permute_tree, FermionParityFusionRule, FusionProductSpace, FusionTreeBlockKey,
    FusionTreeHomSpace, FusionTreeKey, PreparedTreePairOperation, SU2FusionRule, SU2Irrep,
    SectorId, SectorLeg, U1FusionRule, U1Irrep, Z2FusionRule, Z2Irrep,
};

struct CountingAllocator;

thread_local! {
    static COUNTING: Cell<bool> = const { Cell::new(false) };
    static ALLOCATIONS: Cell<usize> = const { Cell::new(0) };
    static ALLOCATED_BYTES: Cell<usize> = const { Cell::new(0) };
}

unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let pointer = unsafe { System.alloc(layout) };
        if !pointer.is_null() && COUNTING.get() {
            ALLOCATIONS.set(ALLOCATIONS.get() + 1);
            ALLOCATED_BYTES.set(ALLOCATED_BYTES.get() + layout.size());
        }
        pointer
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        let pointer = unsafe { System.realloc(ptr, layout, new_size) };
        if !pointer.is_null() && COUNTING.get() {
            ALLOCATIONS.set(ALLOCATIONS.get() + 1);
            ALLOCATED_BYTES.set(ALLOCATED_BYTES.get() + new_size);
        }
        pointer
    }
}

#[global_allocator]
static ALLOCATOR: CountingAllocator = CountingAllocator;

#[test]
fn unique_identity_permute_does_not_allocate() {
    // What: the identity Unique permutation returns its inline tree key
    // without constructing permutation-level scratch.
    let tree = FusionTreeKey::try_new_for_rule(
        &Z2FusionRule,
        [Z2Irrep::ODD.sector_id()],
        Some(Z2Irrep::ODD.sector_id()),
        [false],
        [],
        [],
    )
    .unwrap();
    let _ = unique_permute_tree(&Z2FusionRule, &tree, &[0]).unwrap();

    ALLOCATIONS.set(0);
    COUNTING.set(true);
    let transformed = unique_permute_tree(&Z2FusionRule, &tree, &[0]).unwrap();
    COUNTING.set(false);

    assert_eq!(transformed.0, tree);
    assert_eq!(transformed.1, 1.0);
    assert_eq!(ALLOCATIONS.get(), 0);
}

#[test]
fn block_identity_permute_allocates_only_owned_output() {
    let source = FusionTreeBlockKey::pair_from_sector_ids(
        [1, 1],
        [0],
        Some(0),
        [false, false],
        [false],
        [],
        [],
        [],
        [],
    );
    let sources = [source.clone()];
    let _ =
        multiplicity_free_permute_tree_pair_block(&Z2FusionRule, &sources, &[0, 1], &[2]).unwrap();

    ALLOCATIONS.set(0);
    COUNTING.set(true);
    let transformed =
        multiplicity_free_permute_tree_pair_block(&Z2FusionRule, &sources, &[0, 1], &[2]).unwrap();
    COUNTING.set(false);

    // What: identity block permutation allocates only the intentional owned
    // outer result and its one owned row, with no level-vector temporaries.
    assert_eq!(transformed, vec![vec![(source, 1.0)]]);
    assert_eq!(ALLOCATIONS.get(), 2);
}

#[test]
fn compact_block_warm_allocations_do_not_restore_per_source_scratch() {
    let rule = SU2FusionRule;
    let spin_one = SU2Irrep::from_twice_spin(2).sector_id();
    let codomain: [SectorLeg; 8] = std::array::from_fn(|_| SectorLeg::new([(spin_one, 1)], false));
    let hom = FusionTreeHomSpace::new(
        FusionProductSpace::new(codomain),
        FusionProductSpace::new([SectorLeg::new([(spin_one, 1)], false)]),
    );
    let keys = hom.fusion_tree_keys(&rule);
    let sources = &keys[..16];
    let codomain_permutation = [7usize, 6, 5, 4, 3, 2, 1, 0];
    let domain_permutation = [8usize];
    let _ = multiplicity_free_permute_tree_pair_block(
        &rule,
        sources,
        &codomain_permutation,
        &domain_permutation,
    )
    .unwrap();

    let (output, allocations) = measured_allocations(|| {
        black_box(
            multiplicity_free_permute_tree_pair_block(
                &rule,
                sources,
                &codomain_permutation,
                &domain_permutation,
            )
            .unwrap(),
        )
    });

    // What: the fixed 16-source non-Abelian warm transform retains one compact
    // block workspace rather than recreating full-key scratch per source.
    assert_eq!(output.len(), sources.len());
    // Why not treat this as a general performance bound: it guards this fixed
    // regression fixture while release ABBA covers broader ranks and rules.
    assert!(
        allocations <= 3072,
        "compact warm block allocated {allocations} times"
    );
}

#[test]
fn shared_frame_decode_does_not_allocate_per_source_above_inline_rank() {
    for (rank, expected_per_extra_source) in [(9usize, 5usize), (16, 11)] {
        let even = Z2Irrep::EVEN.sector_id();
        let source = FusionTreeKey::try_new_for_rule(
            &Z2FusionRule,
            std::iter::repeat_n(even, rank),
            Some(even),
            std::iter::repeat_n(false, rank),
            std::iter::repeat_n(even, rank.saturating_sub(2)),
            std::iter::repeat_n(SectorId::new(1), rank.saturating_sub(1)),
        )
        .unwrap();
        let single = [source.clone()];
        let cohort = std::iter::repeat_n(source, 16).collect::<Vec<_>>();
        let mut permutation = (0..rank).collect::<Vec<_>>();
        permutation.swap(0, 1);
        let levels = (0..rank).collect::<Vec<_>>();

        let run = |sources: &[FusionTreeKey]| {
            multiplicity_free_braid_tree_block(&Z2FusionRule, sources, &permutation, &levels)
                .unwrap()
        };
        black_box(run(&single));
        black_box(run(&cohort));
        let (_, single_allocations) = measured_allocations(|| black_box(run(&single)));
        let (_, cohort_allocations) = measured_allocations(|| black_box(run(&cohort)));

        // What: duplicate rank-9/16 sources reuse one owned external frame.
        // The exact slope includes the intentional owned output and local
        // scratch costs. A reconstructed compact codomain frame would add two
        // allocations per source at either rank, so this equality detects it.
        assert_eq!(
            cohort_allocations - single_allocations,
            expected_per_extra_source * (cohort.len() - single.len()),
            "rank-{rank} shared-frame allocation slope changed"
        );
    }
}

#[test]
fn tree_pair_shared_frames_do_not_allocate_per_source_above_inline_rank() {
    for (rank, expected_per_extra_source) in [(9usize, 5usize), (16, 11)] {
        let even = Z2Irrep::EVEN.sector_id();
        let codomain = FusionTreeKey::try_new_for_rule(
            &Z2FusionRule,
            std::iter::repeat_n(even, rank),
            Some(even),
            std::iter::repeat_n(false, rank),
            std::iter::repeat_n(even, rank.saturating_sub(2)),
            std::iter::repeat_n(SectorId::new(1), rank.saturating_sub(1)),
        )
        .unwrap();
        let domain = FusionTreeKey::try_new_for_rule(
            &Z2FusionRule,
            std::iter::empty(),
            Some(even),
            std::iter::empty(),
            std::iter::empty(),
            std::iter::empty(),
        )
        .unwrap();
        let source = FusionTreeBlockKey::pair(codomain, domain);
        let single = [source.clone()];
        let cohort = std::iter::repeat_n(source, 16).collect::<Vec<_>>();
        let mut permutation = (0..rank).collect::<Vec<_>>();
        permutation.swap(0, 1);

        let run = |sources: &[FusionTreeBlockKey]| {
            multiplicity_free_permute_tree_pair_block(&Z2FusionRule, sources, &permutation, &[])
                .unwrap()
        };
        black_box(run(&single));
        black_box(run(&cohort));
        let (_, single_allocations) = measured_allocations(|| black_box(run(&single)));
        let (_, cohort_allocations) = measured_allocations(|| black_box(run(&cohort)));

        // What: both codomain and domain frames are borrowed-matched for every
        // tree-pair source after the first; the exact slope excludes rebuilding
        // either frame while retaining the public owned-output costs.
        assert_eq!(
            cohort_allocations - single_allocations,
            expected_per_extra_source * (cohort.len() - single.len()),
            "rank-{rank} tree-pair shared-frame allocation slope changed"
        );
    }
}

fn u1_homspace(rank: usize) -> FusionTreeHomSpace {
    let leg = || {
        SectorLeg::new(
            [
                (U1Irrep::new(-1).sector_id(), 2),
                (U1Irrep::new(0).sector_id(), 3),
                (U1Irrep::new(2).sector_id(), 1),
            ],
            false,
        )
    };
    let nout = rank / 2;
    FusionTreeHomSpace::new(
        FusionProductSpace::new((0..nout).map(|_| leg())),
        FusionProductSpace::new((nout..rank).map(|_| leg())),
    )
}

fn measured_allocations<T>(operation: impl FnOnce() -> T) -> (T, usize) {
    let (output, allocations, _) = measured_allocation_stats(operation);
    (output, allocations)
}

fn measured_allocation_stats<T>(operation: impl FnOnce() -> T) -> (T, usize, usize) {
    ALLOCATIONS.set(0);
    ALLOCATED_BYTES.set(0);
    COUNTING.set(true);
    let output = operation();
    COUNTING.set(false);
    (output, ALLOCATIONS.get(), ALLOCATED_BYTES.get())
}

#[test]
fn prepared_nonidentity_unique_operations_have_zero_prepare_and_stable_execute_allocations() {
    let odd = Z2Irrep::ODD.sector_id();
    let even = Z2Irrep::EVEN.sector_id();
    let tree = || {
        FusionTreeKey::try_new_for_rule(
            &FermionParityFusionRule,
            [odd, odd],
            Some(even),
            [false, true],
            [],
            [SectorId::new(1)],
        )
        .unwrap()
    };
    let source = FusionTreeBlockKey::pair(tree(), tree());
    let single = [source.clone()];
    let cohort = std::iter::repeat_n(source, 16).collect::<Vec<_>>();
    let permute = PreparedTreePairOperation::prepare_permute(
        &FermionParityFusionRule,
        2,
        2,
        &[3, 0],
        &[1, 2],
    )
    .unwrap();
    let braid = PreparedTreePairOperation::prepare_braid(
        &FermionParityFusionRule,
        2,
        2,
        &[3, 0],
        &[1, 2],
        &[7, 2],
        &[11, 5],
    )
    .unwrap();
    let transpose = PreparedTreePairOperation::prepare_transpose(2, 2, &[1, 3], &[0, 2]).unwrap();

    let preparations: [fn() -> PreparedTreePairOperation; 3] = [
        || {
            PreparedTreePairOperation::prepare_permute(
                &FermionParityFusionRule,
                2,
                2,
                &[3, 0],
                &[1, 2],
            )
            .unwrap()
        },
        || {
            PreparedTreePairOperation::prepare_braid(
                &FermionParityFusionRule,
                2,
                2,
                &[3, 0],
                &[1, 2],
                &[7, 2],
                &[11, 5],
            )
            .unwrap()
        },
        || PreparedTreePairOperation::prepare_transpose(2, 2, &[1, 3], &[0, 2]).unwrap(),
    ];
    for prepare in preparations {
        let (_, calls, bytes) = measured_allocation_stats(prepare);
        // What: valid rank<=8 operation validation and lowering fit entirely
        // in the prepared representation's inline storage.
        assert_eq!((calls, bytes), (0, 0));
    }

    for prepared in [&permute, &braid, &transpose] {
        let run = |sources: &[FusionTreeBlockKey]| {
            for source in sources {
                black_box(
                    prepared
                        .execute_unique_rigid(&FermionParityFusionRule, source)
                        .unwrap(),
                );
            }
        };
        run(&single);
        run(&cohort);
        let (_, single_calls, single_bytes) = measured_allocation_stats(|| run(&single));
        let (_, repeated_calls, repeated_bytes) = measured_allocation_stats(|| run(&single));
        let (_, cohort_calls, cohort_bytes) = measured_allocation_stats(|| run(&cohort));

        // What: repeated nonidentity replay has a stable allocation contract,
        // and a source cohort pays exactly the measured per-source cost.
        assert_eq!(
            (repeated_calls, repeated_bytes),
            (single_calls, single_bytes)
        );
        assert_eq!(cohort_calls, single_calls * cohort.len());
        assert_eq!(cohort_bytes, single_bytes * cohort.len());
    }
}

#[test]
fn rank_eight_or_less_same_side_homspace_derivation_is_heap_free() {
    let rule = U1FusionRule;
    for rank in [2usize, 4, 6, 8] {
        let homspace = u1_homspace(rank);
        let nout = rank / 2;
        let codomain_axes = (0..nout).rev().collect::<Vec<_>>();
        let domain_axes = (nout..rank).rev().collect::<Vec<_>>();

        let (_, select_allocations) = measured_allocations(|| {
            black_box(
                homspace
                    .select(&rule, &codomain_axes, &domain_axes)
                    .unwrap(),
            )
        });
        assert_eq!(
            select_allocations, 0,
            "rank-{rank} same-side HomSpace::select must stay heap-free"
        );
        let (_, checked_select_allocations) = measured_allocations(|| {
            black_box(
                homspace
                    .try_select_checked(&rule, &codomain_axes, &domain_axes)
                    .unwrap(),
            )
        });
        assert_eq!(
            checked_select_allocations, 0,
            "rank-{rank} checked same-side select added transient storage"
        );

        let (_, permute_allocations) = measured_allocations(|| {
            black_box(
                homspace
                    .permute(&rule, &codomain_axes, &domain_axes)
                    .unwrap(),
            )
        });
        assert_eq!(
            permute_allocations, 0,
            "rank-{rank} same-side HomSpace::permute must stay heap-free"
        );
        let (_, checked_permute_allocations) = measured_allocations(|| {
            black_box(
                homspace
                    .try_permute_checked(&rule, &codomain_axes, &domain_axes)
                    .unwrap(),
            )
        });
        assert_eq!(
            checked_permute_allocations, 0,
            "rank-{rank} checked same-side permute added transient storage"
        );

        let (_, compose_allocations) = measured_allocations(|| {
            black_box(FusionTreeHomSpace::compose(&rule, &homspace, &homspace).unwrap())
        });
        assert_eq!(
            compose_allocations, 0,
            "rank-{rank} HomSpace::compose must stay heap-free"
        );
    }
}

#[test]
fn rank_eight_or_less_crossing_derivation_allocates_only_final_dual_legs() {
    let rule = U1FusionRule;
    for rank in [2usize, 4, 6, 8] {
        let homspace = u1_homspace(rank);
        let nout = rank / 2;
        let output_axes = (0..rank).rev().collect::<Vec<_>>();
        let codomain_axes = &output_axes[..nout];
        let domain_axes = &output_axes[nout..];
        let lhs_axes = (nout..rank).collect::<Vec<_>>();
        let rhs_axes = (0..nout).collect::<Vec<_>>();

        let (_, permute_allocations) = measured_allocations(|| {
            black_box(homspace.permute(&rule, codomain_axes, domain_axes).unwrap())
        });
        assert_eq!(
            permute_allocations, rank,
            "rank-{rank} crossing permute must allocate one final LegData per crossed leg"
        );

        let (_, checked_permute_allocations) = measured_allocations(|| {
            black_box(
                homspace
                    .try_permute_checked(&rule, codomain_axes, domain_axes)
                    .unwrap(),
            )
        });
        // What: checked orientation keeps its transactional staging inline for
        // the common rank<=8 case and allocates only the final dual LegData.
        assert_eq!(
            checked_permute_allocations, rank,
            "rank-{rank} checked crossing permute added transient storage"
        );

        let (_, contract_allocations) = measured_allocations(|| {
            black_box(
                FusionTreeHomSpace::tensorcontract_homspace(
                    &rule,
                    &homspace,
                    &homspace,
                    &lhs_axes,
                    &rhs_axes,
                    &output_axes,
                    nout,
                )
                .unwrap(),
            )
        });
        assert_eq!(
            contract_allocations, rank,
            "rank-{rank} contraction must allocate only final crossed LegData"
        );

        let (_, checked_contract_allocations) = measured_allocations(|| {
            black_box(
                FusionTreeHomSpace::try_tensorcontract_homspace_checked(
                    &rule,
                    &homspace,
                    &homspace,
                    &lhs_axes,
                    &rhs_axes,
                    &output_axes,
                    nout,
                )
                .unwrap(),
            )
        });
        assert_eq!(
            checked_contract_allocations, rank,
            "rank-{rank} checked contraction added transient storage"
        );

        let (_, repeated_allocations) = measured_allocations(|| {
            for _ in 0..10 {
                black_box(
                    FusionTreeHomSpace::tensorcontract_homspace(
                        &rule,
                        &homspace,
                        &homspace,
                        &lhs_axes,
                        &rhs_axes,
                        &output_axes,
                        nout,
                    )
                    .unwrap(),
                );
            }
        });
        assert_eq!(
            repeated_allocations,
            rank * 10,
            "rank-{rank} contraction allocation bound must be independent of prior calls"
        );
    }
}
