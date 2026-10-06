use super::*;

fn degeneracy_cache_info() -> StructureCacheInfo {
    structure_cache_info(StructureCacheKind::DegeneracyStructure)
}

#[test]
fn complete_homspace_layout_cache_reuses_semantic_content_and_excludes_regions() {
    // Isolated like prepared_complete_structure_hits_without_rebuilding_layout:
    // this asserts process-global complete-structure cache counters, which
    // CACHE_TEST_LOCK does not protect from the crate's many ordinary,
    // unlocked complete-structure builds landing between two reads (#1903).
    if test_support::run_isolated_or_return(
        "TENET_CORE_COMPLETE_HOMSPACE_REUSE_ISOLATED",
        "tests::fusion_space::layout_caches::complete_homspace_layout_cache_reuses_semantic_content_and_excludes_regions",
    ) {
        return;
    }
    // What: independently constructed complete multiplicity-free U1,
    // SU2, and product HomSpaces share frozen content by value; cached
    // content never owns a wrapper-local coupled-region state.
    let _guard = test_support::CACHE_TEST_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    reset_core_intern_tables();

    fn assert_reused<R>(rule: &R, first_hom: FusionTreeHomSpace, second_hom: FusionTreeHomSpace)
    where
        R: MultiplicityFreeFusionRule,
    {
        let first = first_hom
            .coupled_subblock_structure_from_leg_degeneracies(rule)
            .unwrap();
        let info = degeneracy_cache_info();
        let interned = block_structure_intern_calls();
        // Live wrapper: the hit returns the canonical Arc itself, without
        // re-interning content.
        let second = second_hom
            .coupled_subblock_structure_from_leg_degeneracies(rule)
            .unwrap();
        assert!(Arc::ptr_eq(&first, &second));
        let after = degeneracy_cache_info();
        assert_eq!(after.hits(), info.hits() + 1);
        assert_eq!(after.misses(), info.misses());
        assert_eq!(after.admissions(), info.admissions());
        assert_eq!(block_structure_intern_calls(), interned);

        // Dead wrapper: content stays cached (a hit, not a miss), only the
        // wrapper is rebuilt; region state died with the old wrapper.
        let content = first.content_key();
        let region = first.weak_region_state();
        drop(first);
        drop(second);
        assert!(region.upgrade().is_none());
        let third = second_hom
            .coupled_subblock_structure_from_leg_degeneracies(rule)
            .unwrap();
        assert!(Arc::ptr_eq(&third.content_key(), &content));
        assert!(third.weak_region_state().upgrade().is_some());
        let after = degeneracy_cache_info();
        assert_eq!(after.hits(), info.hits() + 2);
        assert_eq!(after.misses(), info.misses());
        assert_eq!(after.admissions(), info.admissions());
        assert_eq!(block_structure_intern_calls(), interned);
        // The entry's Weak was refreshed: the next hit is the rebuilt Arc.
        let fourth = first_hom
            .coupled_subblock_structure_from_leg_degeneracies(rule)
            .unwrap();
        assert!(Arc::ptr_eq(&third, &fourth));
        let region = third.weak_region_state();
        drop(third);
        drop(fourth);
        assert!(region.upgrade().is_none());
    }

    let u1_hom = || FusionTreeHomSpace::from_sectors([(u1(1), 2)], [(u1(1), 3)]);
    assert_reused(&U1FusionRule, u1_hom(), u1_hom());
    let su2_hom = || FusionTreeHomSpace::from_sectors([(su2(1), 2)], [(su2(1), 3)]);
    assert_reused(&SU2FusionRule, su2_hom(), su2_hom());

    type Fz2U1 = ProductFusionRule<FermionParityFusionRule, U1FusionRule>;
    type Product = ProductFusionRule<Fz2U1, SU2FusionRule>;
    let pair = Fz2U1::new(FermionParityFusionRule, U1FusionRule);
    let sector = pair.encode_sector(z2_odd(), u1(1));
    let rule = Product::new(pair, SU2FusionRule);
    let product_hom = || {
        FusionTreeHomSpace::from_sectors(
            [(rule.encode_sector(sector, su2(1)), 2)],
            [(rule.encode_sector(sector, su2(1)), 3)],
        )
    };
    assert_reused(&rule, product_hom(), product_hom());

    let info = degeneracy_cache_info();
    assert_eq!(info.entries(), 3);
    assert_eq!(info.admissions(), 3);
    assert!(info.hits() >= 3);
}

#[test]
fn complete_homspace_layout_cache_concurrent_hits_share_one_canonical_arc() {
    // Isolated like tenet-tensors #649/#650's checked_bind_failure
    // test: this asserts absolute process-global admissions()/
    // misses()/hits() counts, which CACHE_TEST_LOCK does not protect
    // from the crate's many ordinary, unlocked complete-structure
    // builds landing in the same narrow window. Found flaking
    // (left: 3/2, right: 1 at line 2898-ish) during #1598/#1606
    // verification; same class as that pair's four named instances.
    if test_support::run_isolated_or_return(
            "TENET_CORE_COMPLETE_HOMSPACE_CONCURRENT_HITS_ISOLATED",
            "tests::fusion_space::layout_caches::complete_homspace_layout_cache_concurrent_hits_share_one_canonical_arc",
        ) {
            return;
        }
    // What: threads looking up one key concurrently, with and without a
    // live wrapper between rounds, all receive the same canonical Arc from
    // a single admission.
    let _guard = test_support::CACHE_TEST_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    reset_core_intern_tables();
    let hom = || FusionTreeHomSpace::from_sectors([(su2(1), 2), (su2(3), 1)], [(su2(1), 3)]);
    let held = hom()
        .coupled_subblock_structure_from_leg_degeneracies(&SU2FusionRule)
        .unwrap();

    let round = || {
        let barrier = std::sync::Barrier::new(4);
        let results = std::thread::scope(|scope| {
            let handles = (0..4)
                .map(|_| {
                    scope.spawn(|| {
                        barrier.wait();
                        hom()
                            .coupled_subblock_structure_from_leg_degeneracies(&SU2FusionRule)
                            .unwrap()
                    })
                })
                .collect::<Vec<_>>();
            handles
                .into_iter()
                .map(|handle| handle.join().unwrap())
                .collect::<Vec<_>>()
        });
        assert!(results.iter().all(|r| Arc::ptr_eq(r, &results[0])));
        results
    };

    let live = round();
    assert!(Arc::ptr_eq(&live[0], &held));
    drop(live);
    drop(held);
    let rebuilt = round();
    // The entry's wrapper is the rebuilt Arc.
    let key = CompleteHomSpaceStructureCacheKey::new(&SU2FusionRule, &hom());
    let before = degeneracy_cache_info().hits();
    let entry = degeneracy_structure_cache().get(&key).unwrap();
    assert!(Arc::ptr_eq(&entry.structure(), &rebuilt[0]));
    assert_eq!(degeneracy_cache_info().hits(), before + 1);
    assert_eq!(degeneracy_cache_info().admissions(), 1);
    assert_eq!(degeneracy_cache_info().misses(), 1);
    assert_eq!(degeneracy_cache_info().hits(), 9);
    drop(rebuilt);
}

#[test]
fn complete_homspace_layout_cache_keys_semantics_and_preserves_direct_layout() {
    // Isolated like tenet-tensors #649/#650's checked_bind_failure
    // test: this asserts an absolute process-global entries() count,
    // which CACHE_TEST_LOCK does not protect from the crate's many
    // ordinary, unlocked complete-structure builds landing in the
    // same narrow window. Found flaking (left: 5, right: 4 at line
    // 2960-ish) during #1598/#1606 verification; same class as that
    // pair's four named instances.
    if test_support::run_isolated_or_return(
            "TENET_CORE_COMPLETE_HOMSPACE_KEYS_SEMANTICS_ISOLATED",
            "tests::fusion_space::layout_caches::complete_homspace_layout_cache_keys_semantics_and_preserves_direct_layout",
        ) {
            return;
        }
    // What: rule identity, degeneracies, and dual flags are distinct
    // complete-layout keys, while cache admission preserves the direct
    // builder's ordered block tuples and required storage length.
    let _guard = test_support::CACHE_TEST_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    reset_core_intern_tables();

    let base = FusionTreeHomSpace::new(
        FusionProductSpace::new([SectorLeg::new([(u1(0), 2), (u1(1), 3)], false)]),
        FusionProductSpace::new([SectorLeg::new([(u1(0), 2), (u1(1), 3)], false)]),
    );
    let cached = base
        .coupled_subblock_structure_from_leg_degeneracies(&U1FusionRule)
        .unwrap();
    let layout = base.fusion_tree_layout_data_uncached(&U1FusionRule);
    let (sector, degeneracy) =
        coupled_subblock_parts_from_leg_degeneracies(&base, &layout).unwrap();
    let direct = BlockStructure::from_shared_parts(sector, degeneracy).unwrap();
    let signature = |structure: &BlockStructure| {
        (0..structure.block_count())
            .map(|index| {
                let block = structure.block(index).unwrap();
                (
                    block.key().clone(),
                    block.shape().to_vec(),
                    block.strides().to_vec(),
                    block.offset(),
                )
            })
            .collect::<Vec<_>>()
    };
    assert_eq!(signature(&cached), signature(&direct));
    assert_eq!(cached.required_len(), direct.required_len());

    let different_degeneracy = FusionTreeHomSpace::from_sectors([(u1(0), 4)], [(u1(0), 2)]);
    let dual = FusionTreeHomSpace::new(
        FusionProductSpace::new([SectorLeg::new([(u1(0), 2), (u1(1), 3)], true)]),
        FusionProductSpace::new([SectorLeg::new([(u1(0), 2), (u1(1), 3)], false)]),
    );
    let rule_changed = FusionTreeHomSpace::from_sectors([(z2_even(), 2)], [(z2_even(), 2)]);
    different_degeneracy
        .coupled_subblock_structure_from_leg_degeneracies(&U1FusionRule)
        .unwrap();
    dual.coupled_subblock_structure_from_leg_degeneracies(&U1FusionRule)
        .unwrap();
    rule_changed
        .coupled_subblock_structure_from_leg_degeneracies(&Z2FusionRule)
        .unwrap();
    assert_eq!(degeneracy_cache_info().entries(), 4);

    let live = Arc::clone(&cached);
    reset_core_intern_tables();
    assert_eq!(live.required_len(), cached.required_len());
    assert_eq!(degeneracy_cache_info().entries(), 0);
}

#[test]
fn fusion_layout_shape_and_fermionic_rule_provenance_do_not_alias() {
    // What: one sector layout may be shared across degeneracies, but concrete
    // shapes and bosonic/fermionic rule provenance select distinct structures/layouts.
    let _guard = test_support::CACHE_TEST_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    reset_core_intern_tables();
    let hom = FusionTreeHomSpace::new(
        FusionProductSpace::new([SectorLeg::new([(z2_even(), 1), (z2_odd(), 1)], false)]),
        FusionProductSpace::new([SectorLeg::new([(z2_even(), 1), (z2_odd(), 1)], false)]),
    );
    let bosonic_layout = hom.cached_fusion_tree_layout(&Z2FusionRule);
    let fermionic_layout = hom.cached_fusion_tree_layout(&FermionParityFusionRule);
    assert!(!Arc::ptr_eq(&bosonic_layout, &fermionic_layout));

    let small = hom
        .coupled_subblock_structure(&FermionParityFusionRule, 1, [vec![1, 1], vec![1, 1]])
        .unwrap();
    let large_hom = FusionTreeHomSpace::new(
        FusionProductSpace::new([SectorLeg::new([(z2_even(), 2), (z2_odd(), 3)], false)]),
        FusionProductSpace::new([SectorLeg::new([(z2_even(), 2), (z2_odd(), 3)], false)]),
    );
    let large = large_hom
        .coupled_subblock_structure(&FermionParityFusionRule, 1, [vec![2, 2], vec![3, 3]])
        .unwrap();
    assert!(!Arc::ptr_eq(&small, &large));
    assert_ne!(small.as_ref(), large.as_ref());

    let transient_hom =
        FusionTreeHomSpace::from_sectors([(U1Irrep::new(17), 4)], [(U1Irrep::new(17), 5)]);
    let transient = transient_hom
        .coupled_subblock_structure(&U1FusionRule, 1, [vec![4, 5]])
        .unwrap();
    let expired = Arc::downgrade(&transient);
    drop(transient);
    assert!(expired.upgrade().is_none());
    let rebuilt = transient_hom
        .coupled_subblock_structure(&U1FusionRule, 1, [vec![4, 5]])
        .unwrap();
    assert_eq!(rebuilt.block(0).unwrap().shape(), &[4, 5]);
}

#[test]
fn explicit_shape_coupled_structure_reuses_the_bounded_leg_cache() {
    // What: with shapes equal to the leg degeneracies, the explicit-shape
    // constructor is answered by the byte-bounded complete-HomSpace cache: one
    // leg-grid build on the miss, none on the hit, and the same shared value
    // as the leg-derived constructor.
    let _guard = test_support::CACHE_TEST_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    reset_core_intern_tables();
    let leg = SectorLeg::new([(u1(-1), 2), (u1(0), 3), (u1(2), 1)], false);
    let hom = FusionTreeHomSpace::new(
        FusionProductSpace::new([leg.clone(), leg.clone()]),
        FusionProductSpace::new([leg.clone(), leg]),
    );
    let layout = hom.fusion_tree_layout_data_uncached(&U1FusionRule);
    let grid_sides = layout
        .sectors
        .iter()
        .map(|sector| sector.row_count + sector.col_count)
        .sum::<usize>();
    let shapes = hom
        .fusion_tree_keys(&U1FusionRule)
        .iter()
        .map(|key| {
            key.codomain_uncoupled()
                .iter()
                .chain(key.domain_uncoupled())
                .map(|&sector| hom.codomain().legs()[0].degeneracy(sector).unwrap())
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();

    reset_coupled_grid_build_observations();
    let misses = complete_hom_space_miss_observations();
    let first = hom
        .coupled_subblock_structure(&U1FusionRule, 2, shapes.clone())
        .unwrap();
    assert_eq!(coupled_grid_build_observations().1, grid_sides);
    assert_eq!(complete_hom_space_miss_observations(), misses + 1);

    reset_coupled_grid_build_observations();
    let second = hom
        .coupled_subblock_structure(&U1FusionRule, 2, shapes)
        .unwrap();
    let from_legs = hom
        .coupled_subblock_structure_from_leg_degeneracies(&U1FusionRule)
        .unwrap();
    assert_eq!(coupled_grid_build_observations(), (0, 0));
    assert_eq!(complete_hom_space_miss_observations(), misses + 1);
    assert!(Arc::ptr_eq(&first, &second));
    assert!(Arc::ptr_eq(&first, &from_legs));
    assert!(first.storage_tiling_proven());
}

fn reset_test_homspace() -> FusionTreeHomSpace {
    FusionTreeHomSpace::from_sectors([(U1Irrep::new(3), 2)], [(U1Irrep::new(3), 5)])
}

/// One build of the complete structure, as `coupled_subblock_structure_from_leg_degeneracies`
/// performs it before admission.
fn complete_build(
    homspace: &FusionTreeHomSpace,
) -> (CompleteHomSpaceStructureCacheKey, Arc<BlockStructure>) {
    let key = CompleteHomSpaceStructureCacheKey::new(&U1FusionRule, homspace);
    let layout = homspace.cached_fusion_tree_layout(&U1FusionRule);
    let (sector, degeneracy) =
        coupled_subblock_parts_from_leg_degeneracies(homspace, &layout).unwrap();
    let built = BlockStructure::from_shared_parts(sector, degeneracy)
        .unwrap()
        .into_shared();
    (key, built)
}

#[test]
fn complete_cache_publishes_a_build_that_saw_no_reset() {
    let _guard = test_support::CACHE_TEST_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    reset_core_intern_tables();
    let homspace = reset_test_homspace();
    let epoch = core_reset_epoch();
    let (key, built) = complete_build(&homspace);

    let published = admit_complete_hom_space_structure(key, Arc::clone(&built), epoch);

    // What: the epoch check does not block an ordinary miss.
    assert!(Arc::ptr_eq(&published, &built));
    let hit = homspace
        .coupled_subblock_structure_from_leg_degeneracies(&U1FusionRule)
        .unwrap();
    assert!(Arc::ptr_eq(&hit, &built));
}

#[test]
fn complete_cache_drops_a_build_that_started_before_a_reset() {
    let _guard = test_support::CACHE_TEST_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    reset_core_intern_tables();
    let homspace = reset_test_homspace();
    // A1: a build records its epoch and interns content, then a reset runs,
    // then the build admits.
    let epoch = core_reset_epoch();
    let (key, stale) = complete_build(&homspace);
    reset_core_intern_tables();

    let returned = admit_complete_hom_space_structure(key, Arc::clone(&stale), epoch);

    // What: the straddling build keeps its correct result, but the cache does
    // not publish it, so a post-reset build mints a fresh content identity.
    assert!(Arc::ptr_eq(&returned, &stale));
    let fresh = homspace
        .coupled_subblock_structure_from_leg_degeneracies(&U1FusionRule)
        .unwrap();
    assert_ne!(fresh.content_id(), stale.content_id());
    assert_eq!(fresh.as_ref(), stale.as_ref());
}

#[test]
fn complete_cache_drops_a_build_that_ran_inside_a_reset() {
    let _guard = test_support::CACHE_TEST_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    reset_core_intern_tables();
    let homspace = reset_test_homspace();
    let source = homspace
        .coupled_subblock_structure_from_leg_degeneracies(&U1FusionRule)
        .unwrap();
    // A2: a build runs after the reset cleared the complete cache but before
    // it cleared the intern tables, so it interns the pre-reset content.
    let during = std::rc::Rc::new(std::cell::RefCell::new(None));
    let slot = std::rc::Rc::clone(&during);
    let hom = homspace.clone();
    crate::block_structure::MID_RESET_HOOK.with(|hook| {
        hook.set(Some(Box::new(move || {
            let epoch = core_reset_epoch();
            *slot.borrow_mut() = Some((epoch, complete_build(&hom)));
        })));
    });
    reset_core_intern_tables();
    let (epoch, (key, stale)) = during.borrow_mut().take().unwrap();
    assert_eq!(stale.content_id(), source.content_id());

    admit_complete_hom_space_structure(key, stale, epoch);

    // What: no pre-reset content identity is published after the reset.
    let fresh = homspace
        .coupled_subblock_structure_from_leg_degeneracies(&U1FusionRule)
        .unwrap();
    assert_ne!(fresh.content_id(), source.content_id());
}

#[test]
fn overlapping_resets_wait_so_the_epoch_stays_odd_until_the_first_finishes() {
    let _guard = test_support::CACHE_TEST_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    reset_core_intern_tables();
    let homspace = reset_test_homspace();
    // Kept alive so the intern table still resolves its content mid-reset.
    let source = homspace
        .coupled_subblock_structure_from_leg_degeneracies(&U1FusionRule)
        .unwrap();
    let source_id = source.content_id();
    let start = core_reset_epoch();
    let second_done = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let second = std::rc::Rc::new(std::cell::RefCell::new(None));
    let (slot, done, hom) = (
        std::rc::Rc::clone(&second),
        Arc::clone(&second_done),
        homspace.clone(),
    );
    crate::block_structure::MID_RESET_HOOK.with(|hook| {
        hook.set(Some(Box::new(move || {
            // Reset B starts on another thread while reset A is mid-way.
            let finished = Arc::clone(&done);
            *slot.borrow_mut() = Some(std::thread::spawn(move || {
                reset_core_intern_tables();
                finished.store(true, std::sync::atomic::Ordering::SeqCst);
            }));
            std::thread::sleep(std::time::Duration::from_millis(50));
            // What: B cannot run inside A, so the epoch stays odd and a build
            // here (which interns A's not-yet-cleared content) is not
            // published.
            assert!(!done.load(std::sync::atomic::Ordering::SeqCst));
            assert_eq!(core_reset_epoch(), start + 1);
            let inside = hom
                .coupled_subblock_structure_from_leg_degeneracies(&U1FusionRule)
                .unwrap();
            assert_eq!(inside.content_id(), source_id);
        })));
    });
    reset_core_intern_tables();
    second.borrow_mut().take().unwrap().join().unwrap();

    assert!(second_done.load(std::sync::atomic::Ordering::SeqCst));
    assert_eq!(core_reset_epoch(), start + 4);
    let after = homspace
        .coupled_subblock_structure_from_leg_degeneracies(&U1FusionRule)
        .unwrap();
    assert_ne!(after.content_id(), source_id);
}

#[test]
fn a_degeneracy_only_change_reuses_the_sector_structure() {
    // What (#2014, variable shape): two HomSpaces with the same sectors and
    // different degeneracies enumerate their fusion trees once and share one
    // sector structure; only the degeneracy structure differs. TensorKit
    // keys `sectorstructure` on the sectors alone (`structure.jl:41`).
    let _guard = test_support::CACHE_TEST_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    reset_core_intern_tables();
    let hom = |small: usize, large: usize| {
        let leg = SectorLeg::new([(u1(-1), small), (u1(0), large), (u1(1), small)], false);
        FusionTreeHomSpace::new(
            FusionProductSpace::new([leg.clone(), leg.clone()]),
            FusionProductSpace::new([leg]),
        )
    };
    reset_fusion_tree_layout_probe_side_effect_calls();
    let first = hom(1, 2)
        .coupled_subblock_structure_from_leg_degeneracies(&U1FusionRule)
        .unwrap();
    let second = hom(3, 5)
        .coupled_subblock_structure_from_leg_degeneracies(&U1FusionRule)
        .unwrap();
    assert_eq!(fusion_tree_layout_probe_side_effect_calls(), (1, 1));
    assert!(Arc::ptr_eq(
        &first.content_key().sector,
        &second.content_key().sector
    ));
    assert_ne!(first.required_len(), second.required_len());

    // The Generic builder shares through the same cache, under its own key.
    let su2_hom = |deg: usize| {
        let leg = SectorLeg::new([(su2(1), deg), (su2(2), deg + 1)], false);
        FusionTreeHomSpace::new(
            FusionProductSpace::new([leg.clone(), leg.clone()]),
            FusionProductSpace::new([leg]),
        )
    };
    let small = su2_hom(1)
        .coupled_subblock_structure_from_leg_degeneracies_generic(&SU2FusionRule)
        .unwrap();
    let large = su2_hom(4)
        .coupled_subblock_structure_from_leg_degeneracies_generic(&SU2FusionRule)
        .unwrap();
    assert!(Arc::ptr_eq(
        &small.content_key().sector,
        &large.content_key().sector
    ));
    assert_ne!(small.required_len(), large.required_len());

    // So does the checked Generic builder (#2030), once a walk succeeded.
    let checked = InfallibleGeneric::new(&SU2FusionRule);
    let checked_small = su2_hom(2)
        .coupled_subblock_structure_from_leg_degeneracies_generic_checked(&checked)
        .unwrap();
    let checked_large = su2_hom(5)
        .coupled_subblock_structure_from_leg_degeneracies_generic_checked(&checked)
        .unwrap();
    assert!(Arc::ptr_eq(
        &checked_small.content_key().sector,
        &checked_large.content_key().sector
    ));
}

/// A provider whose fusion channels repeat: `1 ⊗ 1` lists the vacuum twice.
#[derive(Clone, Copy, Debug)]
struct RepeatedChannelRule;

impl FusionRule for RepeatedChannelRule {
    fn rule_identity(&self) -> RuleIdentity {
        RuleIdentity::of_type::<Self>()
    }
    fn fusion_style(&self) -> FusionStyleKind {
        FusionStyleKind::Unique
    }
    fn braiding_style(&self) -> BraidingStyleKind {
        BraidingStyleKind::Bosonic
    }
    fn vacuum(&self) -> SectorId {
        SectorId::new(0)
    }
    fn fusion_channels(&self, left: SectorId, right: SectorId) -> SectorVec {
        let fused = SectorId::new((left.id() + right.id()) % 2);
        if left.id() == 1 && right.id() == 1 {
            smallvec![fused, fused]
        } else {
            smallvec![fused]
        }
    }
}

impl MultiplicityFreeFusionRule for RepeatedChannelRule {}

#[test]
fn repeated_fusion_channels_are_a_typed_error_not_a_duplicate_block() {
    // What (#2014 review F1): duplicate tree keys from a provider with
    // repeated channels stay `DuplicateBlockKey`, as before the sector
    // structure was shared, instead of a structure with two equal keys.
    let leg = || SectorLeg::new([(SectorId::new(0), 1), (SectorId::new(1), 1)], false);
    let hom = FusionTreeHomSpace::new(
        FusionProductSpace::new([leg(), leg(), leg()]),
        FusionProductSpace::new([leg()]),
    );
    let error = hom
        .coupled_subblock_structure_from_leg_degeneracies(&RepeatedChannelRule)
        .unwrap_err();
    assert!(
        matches!(error, CoreError::DuplicateBlockKey { .. }),
        "{error:?}"
    );
}
