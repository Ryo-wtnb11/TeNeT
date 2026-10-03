use super::*;

#[test]
fn hom_space_id_is_idempotent() {
    let build = || {
        FusionTreeHomSpace::new(
            FusionProductSpace::new([u1_leg(1, 2, false)]),
            FusionProductSpace::new([u1_leg(1, 2, false)]),
        )
    };
    assert_eq!(build().id(), build().id());
}

#[test]
fn hom_space_clone_shares_content_but_not_unpublished_id_state() {
    // What: cloning reuses immutable HomSpace data while each handle keeps
    // its own lazy identity publication snapshot.
    let hom = FusionTreeHomSpace::new(
        FusionProductSpace::new([u1_leg(1, 2, false)]),
        FusionProductSpace::new([u1_leg(-1, 3, true)]),
    );
    let before_id = hom.clone();

    assert!(Arc::ptr_eq(&hom.content, &before_id.content));
    assert!(before_id.existing_id().is_none());

    let id = hom.id();
    assert!(before_id.existing_id().is_none());

    let after_id = hom.clone();
    assert!(Arc::ptr_eq(&hom.content, &after_id.content));
    assert_eq!(after_id.existing_id(), Some(id));
}

#[test]
fn hom_space_id_separates_dual_flip() {
    // Rank-1 duality analog of the #119 regression: flipping one leg's dual
    // bit must not alias, on either the codomain or the domain side.
    let base = FusionTreeHomSpace::new(
        FusionProductSpace::new([u1_leg(1, 2, false)]),
        FusionProductSpace::new([u1_leg(1, 2, false)]),
    );
    let cod_dual = FusionTreeHomSpace::new(
        FusionProductSpace::new([u1_leg(1, 2, true)]),
        FusionProductSpace::new([u1_leg(1, 2, false)]),
    );
    let dom_dual = FusionTreeHomSpace::new(
        FusionProductSpace::new([u1_leg(1, 2, false)]),
        FusionProductSpace::new([u1_leg(1, 2, true)]),
    );
    assert_ne!(base.id(), cod_dual.id());
    assert_ne!(base.id(), dom_dual.id());
}

#[test]
fn hom_space_id_separates_sectors_degeneracy_and_rank() {
    let base = FusionTreeHomSpace::new(
        FusionProductSpace::new([u1_leg(1, 2, false)]),
        FusionProductSpace::new([u1_leg(1, 2, false)]),
    )
    .id();
    let other_charge = FusionTreeHomSpace::new(
        FusionProductSpace::new([u1_leg(3, 2, false)]),
        FusionProductSpace::new([u1_leg(1, 2, false)]),
    )
    .id();
    let other_deg = FusionTreeHomSpace::new(
        FusionProductSpace::new([u1_leg(1, 5, false)]),
        FusionProductSpace::new([u1_leg(1, 2, false)]),
    )
    .id();
    let higher_rank = FusionTreeHomSpace::new(
        FusionProductSpace::new([u1_leg(1, 2, false), u1_leg(0, 2, false)]),
        FusionProductSpace::new([u1_leg(1, 2, false)]),
    )
    .id();
    assert_ne!(base, other_charge);
    assert_ne!(base, other_deg);
    assert_ne!(base, higher_rank);
}

#[test]
fn hom_space_id_remains_semantic_after_intern_eviction() {
    // What: floods the shared hom-space intern table past its cap, which
    // races `concurrent_equal_hom_spaces_share_semantic_identity` (asserts
    // ptr_eq on entries of that same table) if both run concurrently.
    let _guard = test_support::CACHE_TEST_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let build = || {
        FusionTreeHomSpace::new(
            FusionProductSpace::new([u1_leg(17, 2, false)]),
            FusionProductSpace::new([u1_leg(17, 3, true)]),
        )
    };
    let before = build().id();
    for charge in 10_000..10_000 + HOM_SPACE_INTERN_CAP as i32 + 1 {
        let _ = FusionTreeHomSpace::new(
            FusionProductSpace::new([u1_leg(charge, 1, false)]),
            FusionProductSpace::new([u1_leg(charge, 1, false)]),
        )
        .id();
    }
    let after = build().id();
    assert!(!Arc::ptr_eq(&before.key, &after.key));
    assert_eq!(before, after);
    let hash = |id: &HomSpaceId| {
        let mut state = rustc_hash::FxHasher::default();
        id.hash(&mut state);
        std::hash::Hasher::finish(&state)
    };
    assert_eq!(hash(&before), hash(&after));
    assert_eq!(
        hom_space_intern_table().read().unwrap().entries.len(),
        HOM_SPACE_INTERN_CAP
    );
}

#[test]
fn eager_hom_space_derivation_does_not_touch_lazy_id_interner() {
    let rule = U1FusionRule;
    let leg = SectorLeg::new([(u1(-1), 2), (u1(2), 1)], false);
    let hom = FusionTreeHomSpace::new(
        FusionProductSpace::new([leg.clone(), leg.clone()]),
        FusionProductSpace::new([leg.clone(), leg]),
    );

    let selected = hom.select(&rule, &[1, 0], &[3, 2]).unwrap();
    let permuted = hom.permute(&rule, &[1, 0], &[3, 2]).unwrap();
    let composed = FusionTreeHomSpace::compose(&rule, &hom, &hom).unwrap();
    let contracted = FusionTreeHomSpace::tensorcontract_homspace(
        &rule,
        &hom,
        &hom,
        &[2, 3],
        &[0, 1],
        &[0, 1, 2, 3],
        2,
    )
    .unwrap();
    assert!(selected.existing_id().is_none());
    assert!(permuted.existing_id().is_none());
    assert!(composed.existing_id().is_none());
    assert!(contracted.existing_id().is_none());

    let selected_id = selected.id();
    assert_eq!(selected_id, permuted.id());
    assert_eq!(composed.id(), contracted.id());
}

#[test]
fn concurrent_eager_hom_space_derivation_does_not_touch_lazy_id_interner() {
    let rule = U1FusionRule;
    let leg = SectorLeg::new([(u1(-1), 2), (u1(2), 1)], false);
    let hom = FusionTreeHomSpace::new(
        FusionProductSpace::new([leg.clone(), leg.clone()]),
        FusionProductSpace::new([leg.clone(), leg]),
    );
    std::thread::scope(|scope| {
        for _ in 0..8 {
            scope.spawn(|| {
                for _ in 0..100 {
                    let permuted = hom.permute(&rule, &[1, 0], &[3, 2]).unwrap();
                    let contracted = FusionTreeHomSpace::tensorcontract_homspace(
                        &rule,
                        &hom,
                        &hom,
                        &[2, 3],
                        &[0, 1],
                        &[0, 1, 2, 3],
                        2,
                    )
                    .unwrap();
                    assert!(permuted.existing_id().is_none());
                    assert!(contracted.existing_id().is_none());
                }
            });
        }
    });
}

#[test]
fn resetting_lazy_hom_space_interner_preserves_semantic_identity() {
    let _guard = test_support::CACHE_TEST_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    reset_hom_space_intern_table();
    let build = || {
        FusionTreeHomSpace::new(
            FusionProductSpace::new([u1_leg(23, 2, false)]),
            FusionProductSpace::new([u1_leg(-23, 3, true)]),
        )
    };
    let before = build().id();
    reset_hom_space_intern_table();
    let after = build().id();
    assert_eq!(before, after);
    assert!(!Arc::ptr_eq(&before.key, &after.key));
}

#[test]
fn fusion_tree_key_collision_falls_back_to_field_comparison() {
    // Why the hook exists: a genuine Fx collision on this fixture is not
    // constructible on demand, so the cache is forced equal to prove that
    // `Eq` and map insertion still distinguish keys by their fields.
    let (distinct, _, _) = key_hash_fixture();
    let forced = 0x5eed_u64;
    let left = distinct[0].clone().with_cached_hash_for_test(forced);
    for other in &distinct[1..] {
        let right = other.clone().with_cached_hash_for_test(forced);
        assert_eq!(fx_hash_of(&left), fx_hash_of(&right));
        assert_ne!(left, right);
        let mut map: rustc_hash::FxHashMap<FusionTreeKey, usize> = rustc_hash::FxHashMap::default();
        map.insert(left.clone(), 0);
        map.insert(right.clone(), 1);
        assert_eq!(map.len(), 2);
        assert_eq!(map[&left], 0);
        assert_eq!(map[&right], 1);
    }
}

#[test]
fn fusion_space_adjoint_view_preserves_custom_storage_footprint() {
    // What: adjoint swaps categorical sides and block axes while retaining
    // the already-admitted physical footprint; applying it twice is exact.
    let dense = TensorMapSpace::<1, 1>::from_dims([2], [3]).unwrap();
    let homspace = FusionTreeHomSpace::from_sector_ids([(0, 2)], [(0, 3)]);
    let structure = BlockStructure::from_blocks(vec![BlockSpec::with_key(
        BlockKey::ordinal(0),
        vec![2, 3],
        vec![1, 4],
        2,
    )
    .unwrap()])
    .unwrap();
    let source = FusionTensorMapSpace::new_unbound(dense, homspace, structure).unwrap();

    reset_exact_storage_fallback_count();
    let adjoint = source.adjoint_view().unwrap();
    assert_eq!(exact_storage_fallback_count(), 0);
    assert_eq!(adjoint.dense_space().codomain().dims(), &[3]);
    assert_eq!(adjoint.dense_space().domain().dims(), &[2]);
    assert_eq!(adjoint.homspace().codomain(), source.homspace().domain());
    assert_eq!(adjoint.homspace().domain(), source.homspace().codomain());
    let block = adjoint.subblock_structure().block(0).unwrap();
    assert_eq!(block.shape(), &[3, 2]);
    assert_eq!(block.strides(), &[4, 1]);
    assert_eq!(block.offset(), 2);
    assert_eq!(adjoint.adjoint_view().unwrap(), source);
}
