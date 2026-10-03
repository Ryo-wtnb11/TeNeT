use super::*;

#[test]
fn fusion_tree_key_hash_is_consistent_with_eq() {
    let (distinct, shared, fresh) = key_hash_fixture();
    let base = &distinct[0];
    for copy in [&shared, &fresh] {
        assert_eq!(base, copy);
        assert_eq!(base.cached_hash_for_test(), copy.cached_hash_for_test());
        assert_eq!(fx_hash_of(base), fx_hash_of(copy));
    }
    for key in &distinct {
        let recomputed = fusion_tree_key_hash(
            key.uncoupled(),
            key.coupled(),
            key.is_dual(),
            key.innerlines(),
            key.vertices(),
        );
        assert_eq!(key.cached_hash_for_test(), recomputed);
        // `Hash` writes exactly the cached value.
        let mut hasher = rustc_hash::FxHasher::default();
        hasher.write_u64(key.cached_hash_for_test());
        assert_eq!(fx_hash_of(key), hasher.finish());
    }
    // Pairwise set includes the shared-backing and fresh-backing
    // duplicates of the base key so the equal-key branch covers real
    // distinct objects, not only a key against itself.
    let mut pairwise = distinct.clone();
    pairwise.push(shared);
    pairwise.push(fresh);
    let mut equal_pairs_between_distinct_objects = 0;
    for (i, a) in pairwise.iter().enumerate() {
        for (j, b) in pairwise.iter().enumerate() {
            if a == b {
                assert_eq!(a.cached_hash_for_test(), b.cached_hash_for_test());
                if i != j {
                    equal_pairs_between_distinct_objects += 1;
                }
            }
        }
    }
    assert_eq!(equal_pairs_between_distinct_objects, 6);
    assert_eq!(
        pairwise.iter().collect::<rustc_hash::FxHashSet<_>>().len(),
        distinct.len()
    );
}

#[test]
fn fusion_tree_key_ordering_matches_field_wise_comparison() {
    let (mut keys, shared, fresh) = key_hash_fixture();
    keys.push(shared);
    keys.push(fresh);
    let fields = |k: &FusionTreeKey| {
        (
            k.uncoupled().to_vec(),
            k.coupled(),
            k.is_dual().to_vec(),
            k.innerlines().to_vec(),
            k.vertices().to_vec(),
        )
    };
    for a in &keys {
        for b in &keys {
            assert_eq!(a.cmp(b), fields(a).cmp(&fields(b)));
            assert_eq!(a.partial_cmp(b), Some(a.cmp(b)));
            assert_eq!(a.cmp(b) == std::cmp::Ordering::Equal, a == b);
        }
    }
}

#[test]
fn braided_fusion_tree_key_carries_a_fresh_hash() {
    let tree = FusionTreeKey::try_from_sector_ids([1, 1, 0], 0, [false, true, false], [0], [1, 1])
        .unwrap();
    let steps = [
        PreparedArtinStep {
            index: 1,
            inverse: false,
        },
        PreparedArtinStep {
            index: 0,
            inverse: true,
        },
    ];
    let (braided, _) =
        execute_unique_tree_braid_steps(&FermionParityFusionRule, &tree, steps).unwrap();
    assert_ne!(braided, tree);
    let recomputed = fusion_tree_key_hash(
        braided.uncoupled(),
        braided.coupled(),
        braided.is_dual(),
        braided.innerlines(),
        braided.vertices(),
    );
    assert_eq!(braided.cached_hash_for_test(), recomputed);
    // Hand oracle: (1⊗1→0)(0⊗0→0) with the vacuum leg braided to the
    // front becomes (0⊗1→1)(1⊗1→0); the dual flag travels with its leg.
    let expected =
        FusionTreeKey::try_from_sector_ids([0, 1, 1], 0, [false, false, true], [1], [1, 1])
            .unwrap();
    assert_eq!(braided, expected);
    assert_eq!(fx_hash_of(&braided), fx_hash_of(&expected));
}

#[test]
fn fusion_tree_key_debug_excludes_cache() {
    let key = FusionTreeKey::try_from_sector_ids([3, 4], 5, [false, true], [], [1]).unwrap();
    assert_eq!(
        format!("{key:?}"),
        "FusionTreeKey { uncoupled: [SectorId(3), SectorId(4)], coupled: SectorId(5), \
             is_dual: [false, true], innerlines: [], vertices: [MultiplicityIndex(1)] }"
    );
}

// Canary (#153) against silent growth of the hottest recoupling-plan key.
// A smaller representation is allowed; growth requires re-checking the
// compact Hash/Eq/Ord and allocation contracts.
#[test]
fn fusion_tree_key_size_has_not_silently_grown() {
    assert!(std::mem::size_of::<FusionTreeKey>() <= 264);
    assert_eq!(
        std::mem::size_of::<MultiplicityIndex>(),
        std::mem::size_of::<usize>()
    );
}

#[test]
fn checked_recursive_product_preserves_channel_order_and_valid_multiplicity() {
    // What: recursive checked products retain the established
    // right-outer/left-inner channel order and valid nsymbol products.
    type Pair = ProductFusionRule<FibonacciFusionRule, FibonacciFusionRule>;
    type Triple = ProductFusionRule<Pair, FibonacciFusionRule>;

    let pair = Pair::new(FibonacciFusionRule, FibonacciFusionRule);
    let pair_tau = pair
        .try_encode_sector(SectorId::new(1), SectorId::new(1))
        .unwrap();
    let triple = Triple::new(pair, FibonacciFusionRule);
    let input = triple
        .try_encode_sector(pair_tau, SectorId::new(1))
        .unwrap();
    let checked = triple.try_fusion_channels(input, input).unwrap();
    let infallible = triple.fusion_channels(input, input);
    assert_eq!(checked, infallible);

    let mut expected = SectorVec::new();
    for right in [SectorId::new(0), SectorId::new(1)] {
        for pair_right in [SectorId::new(0), SectorId::new(1)] {
            for pair_left in [SectorId::new(0), SectorId::new(1)] {
                let pair_channel = triple
                    .left_rule()
                    .try_encode_sector(pair_left, pair_right)
                    .unwrap();
                expected.push(triple.try_encode_sector(pair_channel, right).unwrap());
            }
        }
    }
    assert_eq!(checked, expected);
    for coupled in checked {
        assert_eq!(
            triple.try_nsymbol(input, input, coupled),
            Ok(triple.nsymbol(input, input, coupled))
        );
    }
}

#[test]
fn checked_sector_leg_dual_rejects_a_malformed_id_transactionally() {
    // What: a checked dual rejects an excluded raw ID without changing the
    // source leg.
    let source = SectorLeg::new([(excluded_u1_id(), 2), (u1(1), 3)], false);
    let before = source.clone();
    assert_eq!(
        source.try_dual(&U1FusionRule),
        Err(FusionAlgebraError::InvalidSector {
            sector: excluded_u1_id()
        })
    );
    assert_eq!(source, before);

    #[cfg(target_pointer_width = "64")]
    {
        type Rule = ProductFusionRule<U1FusionRule, Z2FusionRule, TensorKitProductCodec>;
        let rule = Rule::new(U1FusionRule, Z2FusionRule);
        let min = TensorKitProductCodec::encode(excluded_u1_id(), z2_odd());
        let product = SectorLeg::new([(min, 1)], false);
        assert_eq!(
            product.try_dual(&rule),
            Err(FusionAlgebraError::InvalidSector {
                sector: excluded_u1_id()
            })
        );
    }
}

#[test]
fn checked_enumeration_matches_infallible_for_simple_and_unique_rules() {
    // What: the checked SectorId enumerator reproduces the exact
    // CoupledFusionTrees set, order, and prepared layout keys of the
    // infallible hot path for a Simple (SU2) and a Unique (U1) provider —
    // the TensorKit tree-grid oracle plus a byte/order-identity regression.
    let _guard = crate::test_support::CACHE_TEST_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);

    let su2_space =
        FusionTreeHomSpace::from_sectors([(su2(1), 2), (su2(1), 2)], [(su2(1), 2), (su2(1), 2)]);
    for space in [su2_space.codomain(), su2_space.domain()] {
        assert_eq!(
            try_fusion_trees_by_coupled_for_space_checked(&SU2FusionRule, space).unwrap(),
            fusion_trees_by_coupled_for_space(&SU2FusionRule, space),
        );
    }
    // Multi-block: SU2 spin-1/2 x spin-1/2 couples to spin 0 and spin 1.
    assert!(
        try_fusion_trees_by_coupled_for_space_checked(&SU2FusionRule, su2_space.codomain())
            .unwrap()
            .len()
            >= 2
    );

    reset_core_intern_tables();
    let checked = su2_space
        .prepare_fusion_tree_layout_checked(&SU2FusionRule)
        .unwrap();
    let infallible = su2_space.prepare_fusion_tree_layout(&SU2FusionRule);
    assert_eq!(checked.keys(), infallible.keys());

    let u1_space = FusionTreeHomSpace::from_sectors(
        [
            (U1Irrep::new(1).sector_id(), 1),
            (U1Irrep::new(-1).sector_id(), 1),
        ],
        [(U1Irrep::new(0).sector_id(), 1)],
    );
    for space in [u1_space.codomain(), u1_space.domain()] {
        assert_eq!(
            try_fusion_trees_by_coupled_for_space_checked(&U1FusionRule, space).unwrap(),
            fusion_trees_by_coupled_for_space(&U1FusionRule, space),
        );
    }

    // Dual legs change the stored FusionTreeKey duality flags; the checked
    // walk must still reproduce the infallible key set exactly.
    let dual_space = FusionProductSpace::new([
        SectorLeg::new([(su2(1), 2)], true),
        SectorLeg::new([(su2(1), 2)], false),
    ]);
    assert!(dual_space.legs().iter().any(SectorLeg::is_dual));
    assert_eq!(
        try_fusion_trees_by_coupled_for_space_checked(&SU2FusionRule, &dual_space).unwrap(),
        fusion_trees_by_coupled_for_space(&SU2FusionRule, &dual_space),
    );
}
