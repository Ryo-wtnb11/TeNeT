use super::*;

#[test]
fn fusion_layout_local_cache_is_strict_insertion_order_and_resets_exactly() {
    // What: read hits do not promote FIFO order; entry eviction and reset
    // update charged bytes and counters deterministically in isolated state.
    let mut cache = FusionTreeLayoutCache::new(2, 100, 100);
    let (key0, layout0) = local_u1_layout(40_000);
    let (key1, layout1) = local_u1_layout(40_001);
    let (key2, layout2) = local_u1_layout(40_002);
    cache.admit(Arc::clone(&key0), layout0, 30);
    cache.admit(Arc::clone(&key1), layout1, 30);
    assert!(cache.lookup(&key0).is_some());
    cache.admit(Arc::clone(&key2), layout2, 30);

    assert!(cache.lookup(&key0).is_none());
    assert!(cache.lookup(&key1).is_some());
    assert!(cache.lookup(&key2).is_some());
    assert_eq!(cache.info().entries(), 2);
    assert_eq!(cache.info().charged_payload_bytes(), 60);
    assert_eq!(cache.info().evictions(), 1);

    cache.clear();
    assert_eq!(cache.info().entries(), 0);
    assert_eq!(cache.info().charged_payload_bytes(), 0);
    assert_eq!(cache.info().misses(), 0);
    assert_eq!(cache.info().evictions(), 0);
    assert_eq!(cache.info().admission_bypasses(), 0);
}

#[test]
fn fusion_layout_local_cache_enforces_byte_and_max_entry_admission() {
    // What: charged-byte pressure evicts oldest entries, while an oversized
    // entry is returned to its caller but never retained by the cache.
    let mut cache = FusionTreeLayoutCache::new(8, 50, 40);
    let (key0, layout0) = local_u1_layout(50_000);
    let (key1, layout1) = local_u1_layout(50_001);
    let (oversized_key, oversized_layout) = local_u1_layout(50_002);
    cache.admit(Arc::clone(&key0), layout0, 30);
    cache.admit(Arc::clone(&key1), layout1, 30);

    assert!(cache.lookup(&key0).is_none());
    assert!(cache.lookup(&key1).is_some());
    assert_eq!(cache.info().charged_payload_bytes(), 30);
    assert_eq!(cache.info().evictions(), 1);

    let returned = cache.admit(
        Arc::clone(&oversized_key),
        Arc::clone(&oversized_layout),
        41,
    );
    assert!(Arc::ptr_eq(&returned, &oversized_layout));
    assert!(cache.lookup(&oversized_key).is_none());
    assert_eq!(cache.info().entries(), 1);
    assert_eq!(cache.info().charged_payload_bytes(), 30);
    assert_eq!(cache.info().admission_bypasses(), 1);
}

#[test]
fn fusion_layout_local_cache_bypasses_oversized_rule_identity() {
    #[derive(Clone)]
    struct OversizedIdentityRule {
        identity: RuleIdentity,
    }

    impl FusionRule for OversizedIdentityRule {
        fn rule_identity(&self) -> RuleIdentity {
            self.identity.clone()
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
            smallvec![SectorId::new(left.id() ^ right.id())]
        }
    }

    impl MultiplicityFreeFusionRule for OversizedIdentityRule {}

    // What: canonical rule bytes participate in admission accounting, so
    // an identity alone above the per-entry limit is computed but not retained.
    let canonical_bytes = Arc::<[u8]>::from(vec![
        0;
        FUSION_TREE_LAYOUT_CACHE_MAX_ENTRY_BYTES
            .saturating_add(1)
    ]);
    let rule = OversizedIdentityRule {
        identity: RuleIdentity::from_canonical_bytes::<OversizedIdentityRule>(0, canonical_bytes),
    };
    assert!(rule.identity.charged_retained_bytes() > FUSION_TREE_LAYOUT_CACHE_MAX_ENTRY_BYTES);
    let hom =
        FusionTreeHomSpace::from_sectors([(SectorId::new(0), 1)], Vec::<(SectorId, usize)>::new());
    let key = Arc::new(FusionTreeHomSpaceCacheKey::new(&rule, &hom));
    let layout = Arc::new(fusion_tree_layout_from_data(
        next_fusion_tree_layout_id(),
        hom.fusion_tree_layout_data_uncached(&rule),
    ));
    let charged_bytes = charged_fusion_tree_layout_bytes(&key, &layout);
    assert!(charged_bytes > FUSION_TREE_LAYOUT_CACHE_MAX_ENTRY_BYTES);

    let mut cache = FusionTreeLayoutCache::new(
        8,
        FUSION_TREE_LAYOUT_CACHE_BYTE_BUDGET,
        FUSION_TREE_LAYOUT_CACHE_MAX_ENTRY_BYTES,
    );
    let returned = cache.admit(Arc::clone(&key), Arc::clone(&layout), charged_bytes);
    assert!(Arc::ptr_eq(&returned, &layout));
    assert!(cache.lookup(&key).is_none());
    let info = cache.info();
    assert_eq!(info.entries(), 0);
    assert_eq!(info.admission_bypasses(), 1);
}

#[test]
fn fusion_layout_local_cache_bypasses_oversized_product_rule_identity() {
    #[derive(Clone)]
    struct OversizedProductIdentityRule {
        identity: RuleIdentity,
    }

    impl FusionRule for OversizedProductIdentityRule {
        fn rule_identity(&self) -> RuleIdentity {
            self.identity.clone()
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
            smallvec![SectorId::new(left.id() ^ right.id())]
        }
    }

    impl MultiplicityFreeFusionRule for OversizedProductIdentityRule {}

    struct ProductIdentityCodec;

    // What: a product identity retains the canonical bytes of both child
    // identities, so core admission continues to reject an oversized key.
    let canonical_bytes = Arc::<[u8]>::from(vec![
        0;
        FUSION_TREE_LAYOUT_CACHE_MAX_ENTRY_BYTES
            .saturating_add(1)
    ]);
    let rule = OversizedProductIdentityRule {
        identity: RuleIdentity::compose_with_codec::<ProductIdentityCodec>(
            RuleIdentity::from_canonical_bytes::<OversizedProductIdentityRule>(0, canonical_bytes),
            RuleIdentity::of_type::<Z2FusionRule>(),
        ),
    };
    assert!(rule.identity.charged_retained_bytes() > FUSION_TREE_LAYOUT_CACHE_MAX_ENTRY_BYTES);
    let hom =
        FusionTreeHomSpace::from_sectors([(SectorId::new(0), 1)], Vec::<(SectorId, usize)>::new());
    let key = Arc::new(FusionTreeHomSpaceCacheKey::new(&rule, &hom));
    let layout = Arc::new(fusion_tree_layout_from_data(
        next_fusion_tree_layout_id(),
        hom.fusion_tree_layout_data_uncached(&rule),
    ));
    let charged_bytes = charged_fusion_tree_layout_bytes(&key, &layout);
    assert!(charged_bytes > FUSION_TREE_LAYOUT_CACHE_MAX_ENTRY_BYTES);

    let mut cache = FusionTreeLayoutCache::new(
        8,
        FUSION_TREE_LAYOUT_CACHE_BYTE_BUDGET,
        FUSION_TREE_LAYOUT_CACHE_MAX_ENTRY_BYTES,
    );
    let returned = cache.admit(Arc::clone(&key), Arc::clone(&layout), charged_bytes);
    assert!(Arc::ptr_eq(&returned, &layout));
    assert!(cache.lookup(&key).is_none());
    let info = cache.info();
    assert_eq!(info.entries(), 0);
    assert_eq!(info.admission_bypasses(), 1);
}

#[test]
fn complete_homspace_layout_cache_is_fifo_bounded_and_bypasses_one_over_limit() {
    // What: complete immutable layouts retain at most the configured
    // charge, evict the oldest admission without read promotion, and
    // return oversized eager content without retaining it.
    let hom = FusionTreeHomSpace::from_sectors([(u1(0), 1)], Vec::<(SectorId, usize)>::new());
    let structure = BlockStructure::trivial(&[1]).unwrap().into_shared();
    let key = || {
        Arc::new(CompleteHomSpaceStructureCacheKey {
            rule: RuleIdentity::new_unique::<usize>(),
            homspace: Arc::clone(&hom.content),
        })
    };
    let key0 = key();
    let key1 = key();
    let key2 = key();
    let mut cache = CompleteHomSpaceStructureCache::new(2, 20, 10);

    // Charged bytes include the entry's `Weak<BlockStructure>` word
    // (`size_of::<CompleteHomSpaceStructureCacheEntry>()` grew from 2 to
    // 3 words, 16 -> 24 bytes on 64-bit) and the wrapper `ArcInner` the
    // Weak keeps allocated after the last strong owner dies.
    let charged = charged_complete_hom_space_structure_bytes(&key0, &structure.content_key());
    assert!(
        charged
            >= std::mem::size_of::<CompleteHomSpaceStructureCacheKey>()
                + 3 * std::mem::size_of::<usize>()
                + 12 * std::mem::size_of::<usize>()
                + std::mem::size_of::<BlockStructure>()
    );

    cache.admit_built(Arc::clone(&key0), Arc::clone(&structure), 10);
    cache.admit_built(Arc::clone(&key1), Arc::clone(&structure), 10);
    assert!(cache.peek_counting_hit(&key0).is_some());
    cache.admit_built(Arc::clone(&key2), Arc::clone(&structure), 10);
    assert!(cache.peek_counting_hit(&key0).is_none());
    assert!(cache.peek_counting_hit(&key1).is_some());
    assert!(cache.peek_counting_hit(&key2).is_some());
    assert_eq!(cache.info().entries(), 2);
    assert_eq!(cache.info().charged_bytes(), 20);
    assert_eq!(cache.info().evictions(), 1);

    let oversize = key();
    let returned = cache.admit_built(Arc::clone(&oversize), Arc::clone(&structure), 11);
    assert!(Arc::ptr_eq(&returned, &structure));
    assert!(cache.peek_counting_hit(&oversize).is_none());
    assert_eq!(cache.info().entries(), 2);
    assert_eq!(cache.info().bypasses(), 1);
    assert_eq!(cache.info().hits(), 3);
    // Misses are completed builds reaching admission, bypass included.
    assert_eq!(cache.info().misses(), 4);
    assert_eq!(cache.info().admissions(), 3);

    cache.clear();
    let cleared = cache.info();
    assert_eq!(cleared.entries(), 0);
    assert_eq!(cleared.charged_bytes(), 0);
    assert_eq!(cleared.hits(), 0);
    assert_eq!(cleared.misses(), 0);
    assert_eq!(cleared.admissions(), 0);
    assert_eq!(cleared.evictions(), 0);
    assert_eq!(cleared.bypasses(), 0);
}

#[test]
fn complete_homspace_layout_cache_bounds_bind_by_bytes_and_bypass_outliers() {
    // Isolated like prepared_complete_structure_hits_without_rebuilding_layout:
    // this asserts process-global complete-structure cache counters, which
    // CACHE_TEST_LOCK does not protect from the crate's many ordinary,
    // unlocked complete-structure builds landing between two reads (#1903).
    if test_support::run_isolated_or_return(
        "TENET_CORE_COMPLETE_HOMSPACE_BOUNDS_ISOLATED",
        "tests::fusion_space::layout_caches::complete_homspace_layout_cache_bounds_bind_by_bytes_and_bypass_outliers",
    ) {
        return;
    }
    // What: at the production bounds the byte budget, not the entry cap,
    // evicts first for entries of the smallest measured median size
    // (4455 bytes, #1365 census); the budget holds two maximum-size
    // entries; an entry above the limit is returned uncached, one at the
    // limit is retained; the global cache reports these bounds and reset
    // zeroes only its activity.
    let hom = FusionTreeHomSpace::from_sectors([(u1(0), 1)], Vec::<(SectorId, usize)>::new());
    let structure = BlockStructure::trivial(&[1]).unwrap().into_shared();
    let key = || {
        Arc::new(CompleteHomSpaceStructureCacheKey {
            rule: RuleIdentity::new_unique::<usize>(),
            homspace: Arc::clone(&hom.content),
        })
    };
    let mut cache = CompleteHomSpaceStructureCache::new(
        COMPLETE_HOM_SPACE_STRUCTURE_CACHE_CAP,
        COMPLETE_HOM_SPACE_STRUCTURE_CACHE_BYTE_BUDGET,
        COMPLETE_HOM_SPACE_STRUCTURE_CACHE_MAX_ENTRY_BYTES,
    );
    let typical = 4455;
    let fits = COMPLETE_HOM_SPACE_STRUCTURE_CACHE_BYTE_BUDGET / typical;
    assert!(fits < COMPLETE_HOM_SPACE_STRUCTURE_CACHE_CAP);
    for _ in 0..fits {
        cache.admit_built(key(), Arc::clone(&structure), typical);
    }
    assert_eq!(cache.info().evictions(), 0);
    cache.admit_built(key(), Arc::clone(&structure), typical);
    let info = cache.info();
    assert_eq!(info.evictions(), 1);
    assert_eq!(info.entries(), fits);
    assert!(info.charged_bytes() <= info.byte_budget());

    let outlier = key();
    let returned = cache.admit_built(
        Arc::clone(&outlier),
        Arc::clone(&structure),
        COMPLETE_HOM_SPACE_STRUCTURE_CACHE_MAX_ENTRY_BYTES + 1,
    );
    assert!(Arc::ptr_eq(&returned, &structure));
    assert!(cache.peek_counting_hit(&outlier).is_none());
    assert_eq!(cache.info().bypasses(), 1);
    cache.clear();
    let at_limit = [key(), key()];
    for limit_key in &at_limit {
        cache.admit_built(
            Arc::clone(limit_key),
            Arc::clone(&structure),
            COMPLETE_HOM_SPACE_STRUCTURE_CACHE_MAX_ENTRY_BYTES,
        );
    }
    assert!(at_limit
        .iter()
        .all(|limit_key| cache.peek_counting_hit(limit_key).is_some()));
    assert_eq!((cache.info().evictions(), cache.info().bypasses()), (0, 0));

    let _guard = test_support::CACHE_TEST_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    reset_core_intern_tables();
    let global = complete_hom_space_structure_cache_info();
    assert_eq!(global.entry_capacity(), 1024);
    assert_eq!(global.byte_budget(), 4 * 1024 * 1024);
    assert_eq!(
        global.max_entry_bytes(),
        COMPLETE_HOM_SPACE_STRUCTURE_CACHE_MAX_ENTRY_BYTES
    );
    assert_eq!(
        (
            global.entries(),
            global.charged_bytes(),
            global.hits(),
            global.misses()
        ),
        (0, 0, 0, 0)
    );
    assert_eq!(
        (global.admissions(), global.evictions(), global.bypasses()),
        (0, 0, 0)
    );
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
        let info = complete_hom_space_structure_cache_info();
        let interned = block_structure_intern_calls();
        // Live wrapper: the hit returns the canonical Arc itself, without
        // re-interning content.
        let second = second_hom
            .coupled_subblock_structure_from_leg_degeneracies(rule)
            .unwrap();
        assert!(Arc::ptr_eq(&first, &second));
        let after = complete_hom_space_structure_cache_info();
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
        let after = complete_hom_space_structure_cache_info();
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

    let info = complete_hom_space_structure_cache_info();
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
    // The refreshed entry pairs the rebuilt wrapper with its own content.
    let key = CompleteHomSpaceStructureCacheKey::new(&SU2FusionRule, &hom());
    let cache = complete_hom_space_structure_cache().read().unwrap();
    let entry = cache.entries.peek(&key).unwrap();
    assert!(Arc::ptr_eq(&entry.content, &rebuilt[0].content_key()));
    assert!(Arc::ptr_eq(&entry.wrapper.upgrade().unwrap(), &rebuilt[0]));
    drop(cache);
    assert_eq!(complete_hom_space_structure_cache_info().admissions(), 1);
    assert_eq!(complete_hom_space_structure_cache_info().misses(), 1);
    assert_eq!(complete_hom_space_structure_cache_info().hits(), 8);
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
    let direct = BlockStructure::from_parts(sector, degeneracy).unwrap();
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
    assert_eq!(complete_hom_space_structure_cache_info().entries(), 4);

    let live = Arc::clone(&cached);
    reset_core_intern_tables();
    assert_eq!(live.required_len(), cached.required_len());
    assert_eq!(complete_hom_space_structure_cache_info().entries(), 0);
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
    assert_ne!(bosonic_layout.id, fermionic_layout.id);

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
