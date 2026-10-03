use super::*;

fn malformed_simple_su2_tree_pair_tensors() -> (TensorMap<f64, 2, 0>, TensorMap<f64, 2, 0>) {
    let valid = all_codomain_fusion_tree_test_key_for_rule(
        &SU2FusionRule,
        [1, 1],
        0,
        [false, false],
        [],
        [1],
    );
    let malformed = BlockKey::from(
        FusionTreePairKey::try_pair_from_sector_ids(
            [1, 1],
            [],
            1,
            [false, false],
            [],
            [],
            [],
            [1],
            [],
        )
        .unwrap(),
    );
    let dst_structure = packed_fixture_structure(2, [(valid, vec![1, 1])]).unwrap();
    let src_structure = packed_fixture_structure(2, [(malformed, vec![1, 1])]).unwrap();
    let space = TensorMapSpace::<2, 0>::from_dims([1, 1], []).unwrap();
    let dst =
        TensorMap::<f64, 2, 0>::from_vec_with_structure(vec![0.0], space.clone(), dst_structure)
            .unwrap();
    let src =
        TensorMap::<f64, 2, 0>::from_vec_with_structure(vec![1.0], space, src_structure).unwrap();
    (dst, src)
}

pub(super) fn simple_su2_vertex_structure(vertex: usize) -> Arc<BlockStructure> {
    let vertices = (vertex != 0).then_some(vertex);
    let key = BlockKey::from(
        FusionTreePairKey::try_pair_from_sector_ids(
            [1, 1],
            [],
            0,
            [false, false],
            [],
            [],
            [],
            vertices,
            [],
        )
        .unwrap(),
    );
    Arc::new(packed_fixture_structure(2, [(key, vec![1, 1])]).unwrap())
}

#[derive(Clone)]
pub(super) struct AdmissionCountingSu2Rule {
    pub(super) nsymbol_calls: Arc<AtomicUsize>,
}

impl FusionRule for AdmissionCountingSu2Rule {
    fn rule_identity(&self) -> tenet_core::RuleIdentity {
        SU2FusionRule.rule_identity()
    }

    fn fusion_style(&self) -> FusionStyleKind {
        SU2FusionRule.fusion_style()
    }

    fn braiding_style(&self) -> BraidingStyleKind {
        SU2FusionRule.braiding_style()
    }

    fn vacuum(&self) -> SectorId {
        SU2FusionRule.vacuum()
    }

    fn dual(&self, sector: SectorId) -> SectorId {
        SU2FusionRule.dual(sector)
    }

    fn fusion_channels(&self, left: SectorId, right: SectorId) -> SectorVec {
        SU2FusionRule.fusion_channels(left, right)
    }

    fn nsymbol(&self, left: SectorId, right: SectorId, coupled: SectorId) -> usize {
        self.nsymbol_calls.fetch_add(1, Ordering::Relaxed);
        SU2FusionRule.nsymbol(left, right, coupled)
    }
}

impl MultiplicityFreeFusionRule for AdmissionCountingSu2Rule {}

impl MultiplicityFreeFusionSymbols for AdmissionCountingSu2Rule {
    type Scalar = f64;

    fn f_symbol_scalar(
        &self,
        left: SectorId,
        middle: SectorId,
        right: SectorId,
        coupled: SectorId,
        left_coupled: SectorId,
        right_coupled: SectorId,
    ) -> Self::Scalar {
        SU2FusionRule.f_symbol_scalar(left, middle, right, coupled, left_coupled, right_coupled)
    }

    fn r_symbol_scalar(&self, left: SectorId, right: SectorId, coupled: SectorId) -> Self::Scalar {
        SU2FusionRule.r_symbol_scalar(left, right, coupled)
    }
}

impl MultiplicityFreeRigidSymbols for AdmissionCountingSu2Rule {
    fn dim_scalar(&self, sector: SectorId) -> Self::Scalar {
        SU2FusionRule.dim_scalar(sector)
    }

    fn inv_dim_scalar(&self, sector: SectorId) -> Self::Scalar {
        SU2FusionRule.inv_dim_scalar(sector)
    }

    fn sqrt_dim_scalar(&self, sector: SectorId) -> Self::Scalar {
        SU2FusionRule.sqrt_dim_scalar(sector)
    }

    fn inv_sqrt_dim_scalar(&self, sector: SectorId) -> Self::Scalar {
        SU2FusionRule.inv_sqrt_dim_scalar(sector)
    }

    fn twist_scalar(&self, sector: SectorId) -> Self::Scalar {
        SU2FusionRule.twist_scalar(sector)
    }

    fn frobenius_schur_phase_scalar(&self, sector: SectorId) -> Self::Scalar {
        SU2FusionRule.frobenius_schur_phase_scalar(sector)
    }
}

pub(super) fn builtin_tree_cache_state(
    cache: &TreeTransformCache<f64, RuleIdentity>,
) -> (TreeTransformCacheStats, usize) {
    (cache.stats(), cache.structure_len())
}

fn assert_invalid_simple_vertex(error: OperationError) {
    assert_eq!(
        error,
        OperationError::Core(CoreError::MalformedFusionTree {
            message: "fusion tree has an invalid number of vertices",
        })
    );
}

fn valid_simple_source_and_nonalias_malformed_destination(
) -> (TensorMap<f64, 2, 0>, TensorMap<f64, 2, 0>) {
    let valid = all_codomain_fusion_tree_test_key_for_rule(
        &SU2FusionRule,
        [1, 1],
        0,
        [false, false],
        [],
        [1],
    );
    let malformed = BlockKey::from(
        FusionTreePairKey::try_pair_from_sector_ids(
            [1, 1],
            [],
            0,
            [false, false],
            [],
            [],
            [],
            [],
            [],
        )
        .unwrap(),
    );
    let src_structure = packed_fixture_structure(2, [(valid, vec![1, 1])]).unwrap();
    let dst_structure = packed_fixture_structure(2, [(malformed, vec![2, 1])]).unwrap();
    assert_ne!(src_structure.content_id(), dst_structure.content_id());
    let src = TensorMap::<f64, 2, 0>::from_vec_with_structure(
        vec![3.0],
        TensorMapSpace::<2, 0>::from_dims([1, 1], []).unwrap(),
        src_structure,
    )
    .unwrap();
    let dst = TensorMap::<f64, 2, 0>::from_vec_with_structure(
        vec![17.0, 19.0],
        TensorMapSpace::<2, 0>::from_dims([2, 1], []).unwrap(),
        dst_structure,
    )
    .unwrap();
    (dst, src)
}

#[test]
fn callback_builder_admits_whole_source_before_first_callback() {
    let valid = BlockKey::from(
        FusionTreePairKey::try_pair_from_sector_ids(
            [1, 1],
            [],
            0,
            [false, false],
            [],
            [],
            [],
            [1],
            [],
        )
        .unwrap(),
    );
    let malformed_later = BlockKey::from(
        FusionTreePairKey::try_pair_from_sector_ids(
            [1, 1],
            [],
            2,
            [false, false],
            [],
            [],
            [],
            [],
            [],
        )
        .unwrap(),
    );
    let structure =
        packed_fixture_structure(2, [(valid, vec![1, 1]), (malformed_later, vec![1, 1])]).unwrap();
    let callbacks = std::cell::Cell::new(0usize);

    let error = build_tree_transform_group_plan(
        &SU2FusionRule,
        TreeTransformOperation::permute([0, 1], []),
        &structure,
        |source| {
            callbacks.set(callbacks.get() + 1);
            Ok(vec![(source.clone(), 1.0)])
        },
    )
    .unwrap_err();

    // What: a malformed later source block prevents source-major callbacks
    // from observing the valid prefix.
    assert_invalid_simple_vertex(error);
    assert_eq!(callbacks.get(), 0);
}

#[test]
fn callback_builder_rejects_malformed_destination_before_assembly() {
    let source = FusionTreePairKey::pair(
        FusionTreeKey::try_new_for_rule(
            &SU2FusionRule,
            [SectorId::new(1), SectorId::new(1)],
            SectorId::new(0),
            [false, false],
            [],
            [MultiplicityIndex::ONE],
        )
        .unwrap(),
        FusionTreeKey::try_new_for_rule(&SU2FusionRule, [], SectorId::new(0), [], [], []).unwrap(),
    );
    let malformed_destination = FusionTreePairKey::try_pair_from_sector_ids(
        [1, 1],
        [],
        0,
        [false, false],
        [],
        [],
        [],
        [],
        [],
    )
    .unwrap();
    assert_ne!(source, malformed_destination);
    let structure = packed_fixture_structure(2, [(BlockKey::from(source), vec![1, 1])]).unwrap();

    let error = build_tree_transform_group_plan(
        &SU2FusionRule,
        TreeTransformOperation::permute([0, 1], []),
        &structure,
        |_| Ok(vec![(malformed_destination.clone(), 1.0)]),
    )
    .unwrap_err();

    // What: a malformed callback destination is rejected before group assembly
    // even though its external sectors match the admitted source.
    assert_invalid_simple_vertex(error);
}

#[test]
fn warm_structure_aliases_are_rejected_before_local_cache_lookup() {
    let _guard = crate::test_support::CACHE_TEST_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    reset_global_operation_caches();

    let valid_structure = simple_su2_vertex_structure(1);
    let invalid_structure = simple_su2_vertex_structure(0);
    assert_ne!(valid_structure.content_id(), invalid_structure.content_id());
    assert_ne!(
        valid_structure.block(0).unwrap().key(),
        invalid_structure.block(0).unwrap().key()
    );
    let operation = TreeTransformOperation::braid([1, 0], [], [0, 1], []);
    let mut cache = TreeTransformCache::<f64, RuleIdentity>::default();
    cache
        .get_or_compile_tree_pair_structures_with_storage_conjugation_ref(
            &SU2FusionRule,
            &operation,
            &valid_structure,
            &valid_structure,
            false,
        )
        .unwrap();

    let local_before = builtin_tree_cache_state(&cache);
    let error = cache
        .get_or_compile_tree_pair_structures_with_storage_conjugation_ref(
            &SU2FusionRule,
            &operation,
            &invalid_structure,
            &valid_structure,
            false,
        )
        .unwrap_err();
    assert_invalid_simple_vertex(error);

    // What: a malformed destination is rejected before local cache observation
    // changes, even after a valid sibling structure is warm.
    assert_eq!(builtin_tree_cache_state(&cache), local_before);

    let error = cache
        .get_or_compile_tree_pair_structures_with_storage_conjugation_ref(
            &SU2FusionRule,
            &operation,
            &valid_structure,
            &invalid_structure,
            false,
        )
        .unwrap_err();
    assert_invalid_simple_vertex(error);
    assert_eq!(builtin_tree_cache_state(&cache), local_before);

    let mut independent = TreeTransformCache::<f64, RuleIdentity>::default();
    let independent_before = builtin_tree_cache_state(&independent);
    let error = independent
        .get_or_compile_tree_pair_structures_with_storage_conjugation_ref(
            &SU2FusionRule,
            &operation,
            &invalid_structure,
            &valid_structure,
            false,
        )
        .unwrap_err();
    assert_invalid_simple_vertex(error);

    // What: an independent cache cannot use another context's completed
    // structure to admit a malformed raw structure.
    assert_eq!(builtin_tree_cache_state(&independent), independent_before);
}

#[test]
fn exact_warm_structure_reuses_prior_local_admission_proof() {
    let _guard = crate::test_support::CACHE_TEST_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    reset_global_operation_caches();

    let calls = Arc::new(AtomicUsize::new(0));
    let rule = AdmissionCountingSu2Rule {
        nsymbol_calls: Arc::clone(&calls),
    };
    let structure = simple_su2_vertex_structure(1);
    let operation = TreeTransformOperation::braid([1, 0], [], [0, 1], []);
    let mut cache = TreeTransformCache::<f64, RuleIdentity>::default();

    let cold = cache
        .get_or_compile_tree_pair_structures_with_storage_conjugation_ref(
            &rule, &operation, &structure, &structure, false,
        )
        .unwrap();
    assert!(calls.load(Ordering::Relaxed) > 0);

    calls.store(0, Ordering::Relaxed);
    let same_content = Arc::new((*structure).clone());
    assert_eq!(same_content.content_id(), structure.content_id());
    assert!(!Arc::ptr_eq(&same_content, &structure));
    let warm = cache
        .get_or_compile_tree_pair_structures_with_storage_conjugation_ref(
            &rule,
            &operation,
            &same_content,
            &same_content,
            false,
        )
        .unwrap();

    // What: an exact semantic-key/content replay reuses the LOCAL admission
    // proof carried by the published compiled structure.
    assert!(Arc::ptr_eq(&cold, &warm));
    assert_eq!(calls.load(Ordering::Relaxed), 0);

    cache.set_policy(OperationCachePolicy::NoCache);
    calls.store(0, Ordering::Relaxed);
    cache
        .get_or_compile_tree_pair_structures_with_storage_conjugation_ref(
            &rule, &operation, &structure, &structure, false,
        )
        .unwrap();

    // What: clearing retained entries removes proof reuse; the next call
    // performs LOCAL admission again.
    assert!(calls.load(Ordering::Relaxed) > 0);
}

#[test]
fn completed_structure_miss_and_hit_validate_capability_once() {
    use crate::tree_transform::{
        multiplicity_free_capability_validations, reset_multiplicity_free_capability_validations,
    };

    let structure = simple_su2_vertex_structure(1);
    let operation = TreeTransformOperation::braid([1, 0], [], [0, 1], []);
    let mut cache = TreeTransformCache::<f64, RuleIdentity>::default();

    reset_multiplicity_free_capability_validations();
    cache
        .get_or_compile_tree_pair_structures_with_storage_conjugation_ref(
            &SU2FusionRule,
            &operation,
            &structure,
            &structure,
            false,
        )
        .unwrap();
    assert_eq!(multiplicity_free_capability_validations(), 1);

    reset_multiplicity_free_capability_validations();
    cache
        .get_or_compile_tree_pair_structures_with_storage_conjugation_ref(
            &SU2FusionRule,
            &operation,
            &structure,
            &structure,
            false,
        )
        .unwrap();

    // What: misses hand the prior capability proof to the whole-HomSpace
    // compiler, while hits still perform their one admission check.
    assert_eq!(multiplicity_free_capability_validations(), 1);
}

#[test]
fn prelowered_same_content_roles_share_one_local_admission() {
    let _guard = crate::test_support::CACHE_TEST_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    reset_global_operation_caches();

    let calls = Arc::new(AtomicUsize::new(0));
    let rule = AdmissionCountingSu2Rule {
        nsymbol_calls: Arc::clone(&calls),
    };
    let logical = simple_su2_vertex_structure(1);
    let storage = Arc::new((*logical).clone());
    let destination = Arc::new((*logical).clone());
    assert_eq!(logical.content_id(), storage.content_id());
    assert_eq!(logical.content_id(), destination.content_id());
    let mut cache = TreeTransformCache::<f64, RuleIdentity>::default();

    cache
        .get_or_compile_tree_pair_prelowered(
            &rule,
            &TreeTransformOperation::braid([1, 0], [], [0, 1], []),
            &destination,
            &logical,
            &storage,
            false,
            Ok,
            Ok,
        )
        .unwrap();

    // What: logical, storage, and destination roles sharing immutable content
    // perform one LOCAL categorical admission, even through distinct Arcs.
    assert_eq!(calls.load(Ordering::Relaxed), 1);
}

#[test]
fn no_cache_tree_transform_paths_compile_without_retaining_structures() {
    let structure = simple_su2_vertex_structure(1);
    let space = TensorMapSpace::<2, 0>::from_dims([1, 1], []).unwrap();
    let src = TensorMap::<f64, 2, 0>::from_vec_with_structure(
        vec![1.0],
        space.clone(),
        (*structure).clone(),
    )
    .unwrap();
    let dst =
        TensorMap::<f64, 2, 0>::from_vec_with_structure(vec![0.0], space, (*structure).clone())
            .unwrap();
    let operation = TreeTransformOperation::braid([1, 0], [], [0, 1], []);

    let assert_uncached = |f: &mut dyn FnMut(&mut TreeTransformCache<f64, RuleIdentity>)| {
        let mut cache =
            TreeTransformCache::<f64, RuleIdentity>::with_policy(OperationCachePolicy::NoCache);
        f(&mut cache);

        // What: NoCache compiles eagerly without retaining the completed
        // structure.
        assert_eq!(cache.stats().structure_misses(), 1);
        assert_eq!(cache.structure_len(), 0);
    };

    assert_uncached(&mut |cache| {
        cache
            .get_or_compile_tree_pair(&SU2FusionRule, operation.clone(), &dst, &src)
            .unwrap();
    });
    assert_uncached(&mut |cache| {
        cache
            .get_or_compile_tree_pair_structures_with_storage_conjugation_ref(
                &SU2FusionRule,
                &operation,
                &structure,
                &structure,
                false,
            )
            .unwrap();
    });
    assert_uncached(&mut |cache| {
        cache
            .get_or_compile_tree_pair_prelowered(
                &SU2FusionRule,
                &operation,
                &structure,
                &structure,
                &structure,
                false,
                Ok,
                Ok,
            )
            .unwrap();
    });
    assert_uncached(&mut |cache| {
        let logical_keys = (0..structure.block_count())
            .map(|index| match structure.block(index).unwrap().key() {
                BlockKey::FusionTree(key) => key.clone(),
                _ => unreachable!("SU2 fixture uses fusion-tree keys"),
            })
            .collect::<Vec<_>>();
        let storage_indices = (0..logical_keys.len()).collect::<Vec<_>>();
        cache
            .get_or_compile_tree_pair_oriented(
                &SU2FusionRule,
                &operation,
                &structure,
                &logical_keys,
                || Ok(&storage_indices),
                &structure,
                tenet_core::FusionTreePairOrientation::Direct,
                structure.rank(),
                Ok,
            )
            .unwrap();
    });
    assert_uncached(&mut |cache| {
        cache
            .get_or_compile_all_codomain(&SU2FusionRule, operation.clone(), &dst, &src)
            .unwrap();
    });
}

#[test]
fn oriented_adjoint_projection_matches_materialized_logical_oracle() {
    let make_key = |coupled: usize| {
        let coupled = SectorId::new(coupled);
        FusionTreePairKey::pair(
            FusionTreeKey::try_new_for_rule(
                &SU2FusionRule,
                [SectorId::new(1), SectorId::new(1)],
                coupled,
                [false, false],
                [],
                [MultiplicityIndex::ONE],
            )
            .unwrap(),
            FusionTreeKey::try_new_for_rule(
                &SU2FusionRule,
                [SectorId::new(2), SectorId::new(2)],
                coupled,
                [false, false],
                [],
                [MultiplicityIndex::ONE],
            )
            .unwrap(),
        )
    };
    let storage_keys = [make_key(2), make_key(0)];
    let adjoint_key = |key: &FusionTreePairKey| {
        FusionTreePairKey::pair(key.domain_tree().clone(), key.codomain_tree().clone())
    };
    let logical_keys = [adjoint_key(&storage_keys[1]), adjoint_key(&storage_keys[0])];
    let storage = Arc::new(
        packed_fixture_structure(
            4,
            storage_keys
                .iter()
                .cloned()
                .map(|key| (key, vec![1usize; 4])),
        )
        .unwrap(),
    );
    let logical = Arc::new(
        packed_fixture_structure(
            4,
            logical_keys
                .iter()
                .cloned()
                .map(|key| (key, vec![1usize; 4])),
        )
        .unwrap(),
    );
    let operation = TreeTransformOperation::braid([1, 0], [3, 2], [0, 1], [2, 3]);
    let storage_indices = [1, 0];
    let storage_axes = [2, 3, 0, 1];

    let mut old_cache =
        TreeTransformCache::<f64, RuleIdentity>::with_policy(OperationCachePolicy::NoCache);
    let old = old_cache
        .get_or_compile_tree_pair_prelowered(
            &SU2FusionRule,
            &operation,
            &logical,
            &logical,
            &storage,
            true,
            |index| Ok(storage_indices[index]),
            |axis| Ok(storage_axes[axis]),
        )
        .unwrap();
    let mut oriented_cache =
        TreeTransformCache::<f64, RuleIdentity>::with_policy(OperationCachePolicy::NoCache);
    let oriented = oriented_cache
        .get_or_compile_tree_pair_oriented(
            &SU2FusionRule,
            &operation,
            &logical,
            &logical_keys,
            || Ok(&storage_indices),
            &storage,
            tenet_core::FusionTreePairOrientation::Adjoint,
            4,
            |axis| Ok(storage_axes[axis]),
        )
        .unwrap();

    // What: an adjoint operand with noncanonical parent block order compiles
    // and replays exactly like the former materialized logical structure.
    assert_eq!(oriented.as_ref(), old.as_ref());
    let space = TensorMapSpace::<2, 2>::from_dims([1, 1], [1, 1]).unwrap();
    let src = TensorMap::<f64, 2, 2>::from_vec_with_structure(
        vec![10.0, 20.0],
        space.clone(),
        (*storage).clone(),
    )
    .unwrap();
    let make_dst = || {
        TensorMap::<f64, 2, 2>::from_vec_with_structure(
            vec![0.0, 0.0],
            space.clone(),
            (*logical).clone(),
        )
        .unwrap()
    };
    let mut old_dst = make_dst();
    let mut oriented_dst = make_dst();
    let mut backend = DenseTreeTransformOperations::default();
    let mut workspace = TreeTransformWorkspace::default();
    tree_transform_execute_with(
        &mut backend,
        &mut workspace,
        &old,
        &mut old_dst,
        &src,
        1.0,
        0.0,
    )
    .unwrap();
    tree_transform_execute_with(
        &mut backend,
        &mut workspace,
        &oriented,
        &mut oriented_dst,
        &src,
        1.0,
        0.0,
    )
    .unwrap();
    assert_eq!(oriented_dst.data(), old_dst.data());
}

#[test]
fn runtime_bound_adjoint_oriented_plan_is_reused_and_keyed_by_orientation() {
    // Both trees carry sectors [1, 1], so the parent block set is closed under
    // the adjoint and the ordinary and oriented compilers accept the same
    // destination and parent structures: only the key separates their plans.
    let tree = |coupled: usize| {
        FusionTreeKey::try_new_for_rule(
            &SU2FusionRule,
            [SectorId::new(1), SectorId::new(1)],
            SectorId::new(coupled),
            [false, false],
            [],
            [MultiplicityIndex::ONE],
        )
        .unwrap()
    };
    let storage_keys = [
        FusionTreePairKey::pair(tree(2), tree(2)),
        FusionTreePairKey::pair(tree(0), tree(0)),
    ];
    let storage = Arc::new(
        packed_fixture_structure(
            4,
            storage_keys
                .iter()
                .cloned()
                .map(|key| (key, vec![1usize; 4])),
        )
        .unwrap(),
    );
    let destination = Arc::new(
        packed_fixture_structure(
            4,
            [storage_keys[1].clone(), storage_keys[0].clone()]
                .into_iter()
                .map(|key| (key, vec![1usize; 4])),
        )
        .unwrap(),
    );
    let logical_keys = [storage_keys[1].clone(), storage_keys[0].clone()];
    let storage_indices = [1, 0];
    let storage_axes = [2, 3, 0, 1];
    let operation = TreeTransformOperation::braid([1, 0], [3, 2], [0, 1], [2, 3]);
    let other_operation = TreeTransformOperation::braid([0, 1], [3, 2], [0, 1], [2, 3]);
    let oriented = |cache: &mut TreeTransformCache<f64, RuleIdentity>,
                    operation: &TreeTransformOperation,
                    orientation: tenet_core::FusionTreePairOrientation| {
        let direct = orientation == tenet_core::FusionTreePairOrientation::Direct;
        cache
            .get_or_compile_tree_pair_oriented(
                &SU2FusionRule,
                operation,
                &destination,
                if direct { &storage_keys } else { &logical_keys },
                || Ok(if direct { &[0, 1] } else { &storage_indices }),
                &storage,
                orientation,
                4,
                |axis| Ok(if direct { axis } else { storage_axes[axis] }),
            )
            .unwrap()
    };
    let adjoint = tenet_core::FusionTreePairOrientation::Adjoint;
    let store = Arc::new(RuntimeTreeTransformStore::<f64>::default());
    let mut cache = TreeTransformCache::<f64, RuleIdentity>::default();
    cache.bind_runtime_store(Arc::downgrade(&store));
    crate::tree_transform::take_oriented_tree_pair_compiles();

    let first = oriented(&mut cache, &operation, adjoint);
    let second = oriented(&mut cache, &operation, adjoint);

    // What: the second runtime-bound call reuses the Runtime-owned plan.
    assert_eq!(crate::tree_transform::take_oriented_tree_pair_compiles(), 1);
    assert!(Arc::ptr_eq(&first, &second));
    assert_eq!((store.info().hits(), store.info().misses()), (1, 1));
    let mut uncached =
        TreeTransformCache::<f64, RuleIdentity>::with_policy(OperationCachePolicy::NoCache);
    // What: the retained plan equals a fresh compile of the same inputs.
    assert_eq!(
        first.as_ref(),
        oriented(&mut uncached, &operation, adjoint).as_ref()
    );

    let conjugated = cache
        .get_or_compile_tree_pair_structures_with_storage_conjugation_ref(
            &SU2FusionRule,
            &operation,
            &destination,
            &storage,
            true,
        )
        .unwrap();
    let plain = cache
        .get_or_compile_tree_pair_structures_with_storage_conjugation_ref(
            &SU2FusionRule,
            &operation,
            &destination,
            &storage,
            false,
        )
        .unwrap();
    let other = oriented(&mut cache, &other_operation, adjoint);

    // What: an ordinary plan with either conjugation flag, and an adjoint plan
    // for another operation, each miss instead of aliasing the adjoint entry.
    assert!(!Arc::ptr_eq(&first, &conjugated));
    assert!(!Arc::ptr_eq(&first, &plain));
    assert!(!Arc::ptr_eq(&first, &other));
    assert_eq!((store.info().hits(), store.info().misses()), (1, 4));
    assert_eq!(store.info().entries(), 4);

    let direct = tenet_core::FusionTreePairOrientation::Direct;
    let direct_first = oriented(&mut cache, &operation, direct);
    let direct_second = oriented(&mut cache, &operation, direct);

    // What: a direct oriented plan is neither served from nor admitted to the
    // ordinary tree-pair entry whose key it would share.
    assert_eq!(crate::tree_transform::take_oriented_tree_pair_compiles(), 4);
    assert!(!Arc::ptr_eq(&direct_first, &plain));
    assert!(!Arc::ptr_eq(&direct_first, &direct_second));
    assert_eq!((store.info().hits(), store.info().misses()), (1, 4));
}

#[test]
fn adjoint_oriented_degeneracy_change_reuses_the_categorical_plan() {
    // What: a new parent layout with the same sectors misses the completed-
    // structure tier but rebuilds no categorical plan, and binding the cached
    // plan equals a fresh uncached compile of that layout.
    let tree = |coupled: usize| {
        FusionTreeKey::try_new_for_rule(
            &SU2FusionRule,
            [SectorId::new(1), SectorId::new(1)],
            SectorId::new(coupled),
            [false, false],
            [],
            [MultiplicityIndex::ONE],
        )
        .unwrap()
    };
    let storage_keys = [
        FusionTreePairKey::pair(tree(2), tree(2)),
        FusionTreePairKey::pair(tree(0), tree(0)),
    ];
    let logical_keys = [storage_keys[1].clone(), storage_keys[0].clone()];
    let storage_indices = [1, 0];
    let storage_axes = [2, 3, 0, 1];
    let operation = TreeTransformOperation::braid([1, 0], [3, 2], [0, 1], [2, 3]);
    let layouts = |degeneracy: usize| {
        let structure = |keys: [FusionTreePairKey; 2]| {
            Arc::new(
                packed_fixture_structure(4, keys.into_iter().map(|key| (key, vec![degeneracy; 4])))
                    .unwrap(),
            )
        };
        (
            structure(storage_keys.clone()),
            structure([storage_keys[1].clone(), storage_keys[0].clone()]),
        )
    };
    let oriented =
        |cache: &mut TreeTransformCache<f64, RuleIdentity>,
         (storage, destination): &(Arc<BlockStructure>, Arc<BlockStructure>)| {
            cache
                .get_or_compile_tree_pair_oriented(
                    &SU2FusionRule,
                    &operation,
                    destination,
                    &logical_keys,
                    || Ok(&storage_indices),
                    storage,
                    tenet_core::FusionTreePairOrientation::Adjoint,
                    4,
                    |axis| Ok(storage_axes[axis]),
                )
                .unwrap()
        };
    let store = Arc::new(RuntimeTreeTransformStore::<f64>::default());
    let mut cache = TreeTransformCache::<f64, RuleIdentity>::default();
    cache.bind_runtime_store(Arc::downgrade(&store));

    oriented(&mut cache, &layouts(1));
    assert_eq!(store.plan_info().misses(), 1);
    let wider = layouts(2);
    let warm = oriented(&mut cache, &wider);

    assert_eq!(store.info().misses(), 2);
    assert_eq!(
        (store.plan_info().hits(), store.plan_info().misses()),
        (1, 1)
    );
    let mut uncached =
        TreeTransformCache::<f64, RuleIdentity>::with_policy(OperationCachePolicy::NoCache);
    assert_eq!(warm.as_ref(), oriented(&mut uncached, &wider).as_ref());
}

#[test]
fn runtime_bound_adjoint_oriented_projection_errors_match_the_local_path() {
    let tree = |coupled: usize| {
        FusionTreeKey::try_new_for_rule(
            &SU2FusionRule,
            [SectorId::new(1), SectorId::new(1)],
            SectorId::new(coupled),
            [false, false],
            [],
            [MultiplicityIndex::ONE],
        )
        .unwrap()
    };
    let keys = [
        FusionTreePairKey::pair(tree(2), tree(2)),
        FusionTreePairKey::pair(tree(0), tree(0)),
    ];
    let storage = Arc::new(
        packed_fixture_structure(4, keys.iter().cloned().map(|key| (key, vec![1usize; 4])))
            .unwrap(),
    );
    let operation = TreeTransformOperation::braid([1, 0], [3, 2], [0, 1], [2, 3]);
    let duplicate = [keys[0].clone(), keys[0].clone()];
    let store = Arc::new(RuntimeTreeTransformStore::<f64>::default());
    let mut bound = TreeTransformCache::<f64, RuleIdentity>::default();
    bound.bind_runtime_store(Arc::downgrade(&store));
    let mut local =
        TreeTransformCache::<f64, RuleIdentity>::with_policy(OperationCachePolicy::NoCache);
    let malformed: [(&[FusionTreePairKey], &[usize]); 4] = [
        (&keys, &[1]),
        (&keys, &[1, 2]),
        (&duplicate, &[1, 0]),
        (&duplicate, &[1, 2]),
    ];
    for (logical_keys, storage_indices) in malformed {
        let compile = |cache: &mut TreeTransformCache<f64, RuleIdentity>| {
            let error = cache
                .get_or_compile_tree_pair_oriented(
                    &SU2FusionRule,
                    &operation,
                    &storage,
                    logical_keys,
                    || Ok(storage_indices),
                    &storage,
                    tenet_core::FusionTreePairOrientation::Adjoint,
                    4,
                    Ok,
                )
                .unwrap_err();
            format!("{error:?}")
        };

        // What: a runtime-bound miss reports the same first projection error
        // as the context-local path and admits nothing.
        assert_eq!(compile(&mut bound), compile(&mut local));
    }
    // What: every malformed projection is read only after a store miss, so
    // each one counts a miss and admits nothing.
    assert_eq!(store.info().entries(), 0);
    assert_eq!(store.info().misses(), 4);
}

#[test]
fn prelowered_compile_does_not_enter_the_completed_structure_lru() {
    let structure = simple_su2_vertex_structure(1);
    let operation_a = TreeTransformOperation::braid([1, 0], [], [0, 1], []);
    let operation_b = TreeTransformOperation::permute([0, 1], []);
    let operation_c = TreeTransformOperation::braid([1, 0], [], [1, 0], []);
    let mut cache = TreeTransformCache::<f64, RuleIdentity>::with_policy(
        OperationCachePolicy::task_local_lru(2),
    );

    let first_a = cache
        .get_or_compile_tree_pair_structures_with_storage_conjugation_ref(
            &SU2FusionRule,
            &operation_a,
            &structure,
            &structure,
            false,
        )
        .unwrap();
    cache
        .get_or_compile_tree_pair_prelowered(
            &SU2FusionRule,
            &operation_b,
            &structure,
            &structure,
            &structure,
            false,
            Ok,
            Ok,
        )
        .unwrap();
    let warm_a = cache
        .get_or_compile_tree_pair_structures_with_storage_conjugation_ref(
            &SU2FusionRule,
            &operation_a,
            &structure,
            &structure,
            false,
        )
        .unwrap();
    assert!(Arc::ptr_eq(&first_a, &warm_a));

    cache.reset_stats();
    cache
        .get_or_compile_tree_pair_structures_with_storage_conjugation_ref(
            &SU2FusionRule,
            &operation_c,
            &structure,
            &structure,
            false,
        )
        .unwrap();
    cache
        .get_or_compile_tree_pair_structures_with_storage_conjugation_ref(
            &SU2FusionRule,
            &operation_b,
            &structure,
            &structure,
            false,
        )
        .unwrap();

    // What: the prelowered compile does not occupy the completed-structure LRU;
    // the later ordinary B and C calls are both misses.
    assert_eq!(cache.stats().structure_hits(), 0);
    assert_eq!(cache.stats().structure_misses(), 2);
}

#[test]
fn failed_structure_compile_does_not_change_completed_structure_recency() {
    let structure = simple_su2_vertex_structure(1);
    let wrong_destination = Arc::new(
        packed_fixture_structure(
            2,
            [(
                BlockKey::from(
                    FusionTreePairKey::try_pair_from_sector_ids(
                        [2, 2],
                        [],
                        0,
                        [false, false],
                        [],
                        [],
                        [],
                        [1],
                        [],
                    )
                    .unwrap(),
                ),
                vec![1, 1],
            )],
        )
        .unwrap(),
    );
    let operation_a = TreeTransformOperation::braid([1, 0], [], [0, 1], []);
    let operation_b = TreeTransformOperation::permute([0, 1], []);
    let operation_c = TreeTransformOperation::braid([1, 0], [], [1, 0], []);
    let mut cache = TreeTransformCache::<f64, RuleIdentity>::with_policy(
        OperationCachePolicy::task_local_lru(2),
    );

    cache
        .get_or_compile_tree_pair_structures_with_storage_conjugation_ref(
            &SU2FusionRule,
            &operation_b,
            &structure,
            &structure,
            false,
        )
        .unwrap();
    let first_a = cache
        .get_or_compile_tree_pair_structures_with_storage_conjugation_ref(
            &SU2FusionRule,
            &operation_a,
            &structure,
            &structure,
            false,
        )
        .unwrap();
    cache
        .get_or_compile_tree_pair_structures_with_storage_conjugation_ref(
            &SU2FusionRule,
            &operation_b,
            &wrong_destination,
            &structure,
            false,
        )
        .unwrap_err();

    cache.reset_stats();
    let warm_a = cache
        .get_or_compile_tree_pair_structures_with_storage_conjugation_ref(
            &SU2FusionRule,
            &operation_a,
            &structure,
            &structure,
            false,
        )
        .unwrap();
    assert!(Arc::ptr_eq(&first_a, &warm_a));
    cache
        .get_or_compile_tree_pair_structures_with_storage_conjugation_ref(
            &SU2FusionRule,
            &operation_c,
            &structure,
            &structure,
            false,
        )
        .unwrap();
    let retained_a = cache
        .get_or_compile_tree_pair_structures_with_storage_conjugation_ref(
            &SU2FusionRule,
            &operation_a,
            &structure,
            &structure,
            false,
        )
        .unwrap();

    // What: a failed compile does not publish or promote an entry, while the
    // subsequent A hit keeps that completed structure resident when C arrives.
    assert!(Arc::ptr_eq(&first_a, &retained_a));
    assert_eq!(cache.stats().structure_hits(), 2);
    assert_eq!(cache.stats().structure_misses(), 1);
}

#[test]
fn warm_prelowered_raw_arc_aliases_do_not_observe_or_mutate_cache_state() {
    let _guard = crate::test_support::CACHE_TEST_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    reset_global_operation_caches();

    let valid = simple_su2_vertex_structure(1);
    let invalid = simple_su2_vertex_structure(0);
    assert_ne!(valid.content_id(), invalid.content_id());
    assert!(!Arc::ptr_eq(&valid, &invalid));
    let operation = TreeTransformOperation::braid([1, 0], [], [0, 1], []);
    let mut cache = TreeTransformCache::<f64, RuleIdentity>::default();
    cache
        .get_or_compile_tree_pair_prelowered(
            &SU2FusionRule,
            &operation,
            &valid,
            &valid,
            &valid,
            false,
            Ok,
            Ok,
        )
        .unwrap();
    let local_before = builtin_tree_cache_state(&cache);

    for (destination, logical_source, storage_source) in [
        (&valid, &invalid, &valid),
        (&valid, &valid, &invalid),
        (&invalid, &valid, &valid),
    ] {
        let error = cache
            .get_or_compile_tree_pair_prelowered(
                &SU2FusionRule,
                &operation,
                destination,
                logical_source,
                storage_source,
                false,
                Ok,
                Ok,
            )
            .unwrap_err();
        assert_invalid_simple_vertex(error);

        // What: each raw-Arc role is categorically admitted before a warm
        // completed-structure lookup can observe its shared content identity.
        assert_eq!(builtin_tree_cache_state(&cache), local_before);
    }
}

#[test]
fn invalid_simple_source_does_not_mutate_cached_or_no_cache_state() {
    // What: malformed categorical input is rejected before cache statistics or
    // retained completed-structure state change under either execution policy.
    for policy in [
        OperationCachePolicy::default(),
        OperationCachePolicy::NoCache,
    ] {
        let (dst, src) = malformed_simple_su2_tree_pair_tensors();
        let mut cache = TreeTransformCache::<f64, RuleIdentity>::with_policy(policy);
        let before_stats = cache.stats();

        let error = cache
            .get_or_compile_tree_pair(
                &SU2FusionRule,
                TreeTransformOperation::braid([0, 1], [], [0, 1], []),
                &dst,
                &src,
            )
            .unwrap_err();

        assert_eq!(
            error,
            OperationError::Core(CoreError::MalformedFusionTree {
                message: "fusion tree contains an inadmissible fusion vertex",
            })
        );
        assert_eq!(cache.stats(), before_stats);
        assert_eq!(cache.structure_len(), 0);
    }
}

#[test]
fn malformed_source_precedes_noncategorical_destination_without_cache_mutation() {
    let (_, src) = malformed_simple_su2_tree_pair_tensors();
    let dst_structure = packed_fixture_structure(2, [(BlockKey::opaque([9]), vec![1, 1])]).unwrap();
    let dst = TensorMap::<f64, 2, 0>::from_vec_with_structure(
        vec![0.0],
        TensorMapSpace::<2, 0>::from_dims([1, 1], []).unwrap(),
        dst_structure,
    )
    .unwrap();
    let mut cache = TreeTransformCache::<f64, RuleIdentity>::new();

    let error = cache
        .get_or_compile_tree_pair(
            &SU2FusionRule,
            TreeTransformOperation::permute([0, 1], []),
            &dst,
            &src,
        )
        .unwrap_err();

    // What: cold admission proves the categorical source before diagnosing
    // the destination namespace, without touching cache state.
    assert_eq!(
        error,
        OperationError::Core(CoreError::MalformedFusionTree {
            message: "fusion tree contains an inadmissible fusion vertex",
        })
    );
    assert_eq!(cache.stats(), TreeTransformCacheStats::default());
    assert!(cache.is_empty());
}

#[test]
fn invalid_operation_precedes_malformed_simple_source_without_cache_mutation() {
    use crate::tree_transform::{
        reset_tree_pair_operation_preparations, tree_pair_operation_preparations,
    };

    // What: operation syntax has deterministic precedence over categorical
    // source admission and neither failure is counted as a cache miss.
    let (dst, src) = malformed_simple_su2_tree_pair_tensors();
    let mut cache = TreeTransformCache::<f64, RuleIdentity>::new();

    reset_tree_pair_operation_preparations();
    let error = cache
        .get_or_compile_tree_pair(
            &SU2FusionRule,
            TreeTransformOperation::braid([0, 0], [], [0, 1], []),
            &dst,
            &src,
        )
        .unwrap_err();

    assert_eq!(
        error,
        OperationError::Core(CoreError::InvalidPermutation {
            permutation: vec![0, 0],
            rank: 2,
        })
    );
    assert_eq!(tree_pair_operation_preparations(), 0);
    assert_eq!(cache.stats(), TreeTransformCacheStats::default());
    assert!(cache.is_empty());
}

#[test]
fn invalid_operation_precedes_non_categorical_namespace_without_cache_mutation() {
    for key in [BlockKey::Dense, BlockKey::opaque([7, 11])] {
        let structure = BlockStructure::from_blocks(vec![BlockSpec::column_major_with_key(
            key.clone(),
            vec![1, 1],
            0,
        )
        .unwrap()])
        .unwrap();
        let space = TensorMapSpace::<2, 0>::from_dims([1, 1], []).unwrap();
        let dst = TensorMap::<f64, 2, 0>::from_vec_with_structure(
            vec![0.0],
            space.clone(),
            structure.clone(),
        )
        .unwrap();
        let src =
            TensorMap::<f64, 2, 0>::from_vec_with_structure(vec![1.0], space, structure).unwrap();
        let mut cache = TreeTransformCache::<f64, RuleIdentity>::new();

        let error = cache
            .get_or_compile_tree_pair(
                &SU2FusionRule,
                TreeTransformOperation::permute([0, 0], []),
                &dst,
                &src,
            )
            .unwrap_err();
        assert_eq!(
            error,
            OperationError::Core(CoreError::InvalidPermutation {
                permutation: vec![0, 0],
                rank: 2,
            })
        );
        assert_eq!(cache.stats(), TreeTransformCacheStats::default());
        assert!(cache.is_empty());

        let error = cache
            .get_or_compile_tree_pair(
                &SU2FusionRule,
                TreeTransformOperation::permute([0, 1], []),
                &dst,
                &src,
            )
            .unwrap_err();
        assert_eq!(
            error,
            OperationError::Core(CoreError::ExpectedFusionTreePairKey { actual: key.kind() })
        );
        // What: syntax and namespace failures occur before any cache lookup or
        // categorical coefficient provider can compile a row.
        assert_eq!(cache.stats(), TreeTransformCacheStats::default());
        assert!(cache.is_empty());
    }
}

#[test]
fn invalid_unique_source_does_not_count_eager_compile_misses() {
    // What: a failed Unique eager build reports no successful structure compile
    // in cache statistics.
    let valid = all_codomain_fusion_tree_test_key([1, 1], 0, [false, false], [], [1]);
    let malformed = BlockKey::from(
        FusionTreePairKey::try_pair_from_sector_ids(
            [1, 1],
            [],
            1,
            [false, false],
            [],
            [],
            [],
            [1],
            [],
        )
        .unwrap(),
    );
    let dst_structure = packed_fixture_structure(2, [(valid, vec![1, 1])]).unwrap();
    let src_structure = packed_fixture_structure(2, [(malformed, vec![1, 1])]).unwrap();
    let space = TensorMapSpace::<2, 0>::from_dims([1, 1], []).unwrap();
    let dst =
        TensorMap::<f64, 2, 0>::from_vec_with_structure(vec![0.0], space.clone(), dst_structure)
            .unwrap();
    let src =
        TensorMap::<f64, 2, 0>::from_vec_with_structure(vec![1.0], space, src_structure).unwrap();
    let mut cache = TreeTransformCache::<f64, RuleIdentity>::new();

    let error = cache
        .get_or_compile_tree_pair(
            &Z2FusionRule,
            TreeTransformOperation::permute([0, 1], []),
            &dst,
            &src,
        )
        .unwrap_err();

    assert_eq!(
        error,
        OperationError::Core(CoreError::MalformedFusionTree {
            message: "fusion tree contains an inadmissible fusion vertex",
        })
    );
    assert_eq!(cache.stats(), TreeTransformCacheStats::default());
    assert!(cache.is_empty());
}

#[test]
fn direct_tree_pair_rejects_nonalias_malformed_destination_without_writing() {
    let (mut dst, src) = valid_simple_source_and_nonalias_malformed_destination();
    let before = dst.data().to_vec();

    let error = tree_transform_into(
        &SU2FusionRule,
        TreeTransformOperation::permute([0, 1], []),
        &mut dst,
        &src,
        1.0,
        0.0,
    )
    .unwrap_err();

    // What: noncached tree-pair admission validates the destination before
    // replay can modify caller-owned storage.
    assert_invalid_simple_vertex(error);
    assert_eq!(dst.data(), before);
}

#[test]
fn typed_all_codomain_rejects_nonalias_malformed_destination_without_state_or_output_change() {
    let (mut dst, src) = valid_simple_source_and_nonalias_malformed_destination();
    let before_output = dst.data().to_vec();
    let mut context = TreeTransformExecutionContext::<f64, RuleIdentity>::default();
    let before_cache = builtin_tree_cache_state(context.cache());

    let error = context
        .all_codomain_tree_transform_into(
            &SU2FusionRule,
            TreeTransformOperation::permute([0, 1], []),
            &mut dst,
            &src,
            1.0,
            0.0,
        )
        .unwrap_err();

    // What: typed constructors canonicalize equal content identities, so this
    // nonalias fixture proves all-codomain destination admission without a
    // fabricated raw-Arc alias.
    assert_invalid_simple_vertex(error);
    assert_eq!(builtin_tree_cache_state(context.cache()), before_cache);
    assert_eq!(dst.data(), before_output);
}

#[test]
fn all_codomain_pair_mismatch_is_rejected_before_source_scope_or_cache_state() {
    // What: whole-pair categorical admission precedes the all-codomain source
    // restriction, so mismatched coupled sectors retain the core error.
    let nonempty_domain = BlockKey::from(FusionTreePairKey::pair(
        FusionTreeKey::try_new_for_rule(
            &SU2FusionRule,
            [SectorId::new(1), SectorId::new(1)],
            SectorId::new(0),
            [false, false],
            [],
            [MultiplicityIndex::ONE],
        )
        .unwrap(),
        FusionTreeKey::try_new_for_rule(
            &SU2FusionRule,
            [SectorId::new(0)],
            SectorId::new(0),
            [false],
            [],
            [],
        )
        .unwrap(),
    ));
    let mismatched = BlockKey::from(FusionTreePairKey::pair(
        FusionTreeKey::try_new_for_rule(
            &SU2FusionRule,
            [SectorId::new(1), SectorId::new(1)],
            SectorId::new(0),
            [false, false],
            [],
            [MultiplicityIndex::ONE],
        )
        .unwrap(),
        FusionTreeKey::try_new_for_rule(
            &SU2FusionRule,
            [SectorId::new(2)],
            SectorId::new(2),
            [false],
            [],
            [],
        )
        .unwrap(),
    ));
    let src_structure = packed_fixture_structure(
        3,
        [
            (nonempty_domain, vec![1, 1, 1]),
            (mismatched, vec![1, 1, 1]),
        ],
    )
    .unwrap();
    let dst_key =
        all_codomain_fusion_tree_test_key([0, 0, 0], 0, [false, false, false], [0], [1, 1]);
    let dst_structure = packed_fixture_structure(3, [(dst_key, vec![1, 1, 1])]).unwrap();
    let src_space = TensorMapSpace::<2, 1>::from_dims([1, 1], [1]).unwrap();
    let dst_space = TensorMapSpace::<3, 0>::from_dims([1, 1, 1], []).unwrap();
    let src =
        TensorMap::<f64, 2, 1>::from_vec_with_structure(vec![1.0, 2.0], src_space, src_structure)
            .unwrap();
    let dst = TensorMap::<f64, 3, 0>::from_vec_with_structure(vec![0.0], dst_space, dst_structure)
        .unwrap();
    let mut cache = TreeTransformCache::<f64, RuleIdentity>::new();

    let error = cache
        .get_or_compile_all_codomain(
            &SU2FusionRule,
            TreeTransformOperation::permute([0, 1, 2], []),
            &dst,
            &src,
        )
        .unwrap_err();

    assert_eq!(
        error,
        OperationError::Core(CoreError::MalformedFusionTree {
            message: "fusion tree pair requires matching coupled sectors",
        })
    );

    let error = cache
        .get_or_compile_all_codomain(
            &SU2FusionRule,
            TreeTransformOperation::permute([0, 0, 2], []),
            &dst,
            &src,
        )
        .unwrap_err();
    assert_eq!(
        error,
        OperationError::Core(CoreError::MalformedFusionTree {
            message: "fusion tree pair requires matching coupled sectors",
        })
    );

    let error = cache
        .get_or_compile_all_codomain(
            &SU2FusionRule,
            TreeTransformOperation::transpose([0, 1, 2], []),
            &dst,
            &src,
        )
        .unwrap_err();
    assert_eq!(
        error,
        OperationError::Core(CoreError::MalformedFusionTree {
            message: "fusion tree pair requires matching coupled sectors",
        })
    );

    // What: whole-pair source admission also precedes all-codomain operation
    // scope without publishing cache state.
    assert_eq!(cache.stats(), TreeTransformCacheStats::default());
    assert!(cache.is_empty());
}
