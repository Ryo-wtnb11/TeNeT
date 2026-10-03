use super::*;

#[test]
fn independent_tree_transform_contexts_compile_their_own_artifacts() {
    let _guard = crate::test_support::CACHE_TEST_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let src_key0 = all_codomain_fusion_tree_test_key_for_rule(
        &SU2FusionRule,
        [1, 1, 1, 1],
        0,
        [false, false, false, false],
        [0, 1],
        [1, 1, 1],
    );
    let src_key1 = all_codomain_fusion_tree_test_key_for_rule(
        &SU2FusionRule,
        [1, 1, 1, 1],
        0,
        [false, false, false, false],
        [2, 1],
        [1, 1, 1],
    );
    let block_structure = packed_fixture_structure(
        4,
        [
            (src_key0.clone(), vec![1, 1, 1, 1]),
            (src_key1.clone(), vec![1, 1, 1, 1]),
        ],
    )
    .unwrap();
    let space = TensorMapSpace::<4, 0>::from_dims([1, 1, 1, 1], []).unwrap();
    let src = TensorMap::<f64, 4, 0>::from_vec_with_structure(
        vec![3.0, -4.0],
        space.clone(),
        block_structure.clone(),
    )
    .unwrap();
    let operation = TreeTransformOperation::braid([0, 2, 1, 3], [], [23, 29, 31, 37], []);

    let calls = Arc::new(AtomicUsize::new(0));
    let rule = AdmissionCountingSu2Rule {
        nsymbol_calls: Arc::clone(&calls),
    };
    let mut first = TreeTransformExecutionContext::<f64, RuleIdentity>::default();
    let run = |context: &mut TreeTransformExecutionContext<f64, RuleIdentity>| {
        let mut dst = TensorMap::<f64, 4, 0>::from_vec_with_structure(
            vec![1.0, 2.0],
            space.clone(),
            block_structure.clone(),
        )
        .unwrap();
        context
            .tree_transform_into(&rule, operation.clone(), &mut dst, &src, 2.0, -1.0)
            .unwrap();
        dst.data().to_vec()
    };
    let first_data = run(&mut first);
    assert_eq!(first.cache().stats().structure_misses(), 1);
    assert!(calls.load(Ordering::Relaxed) > 0);

    calls.store(0, Ordering::Relaxed);
    let mut second = TreeTransformExecutionContext::<f64, RuleIdentity>::default();
    let second_data = run(&mut second);
    assert_eq!(second.cache().stats().structure_misses(), 1);
    assert!(calls.load(Ordering::Relaxed) > 0);

    // What: a fresh execution context recompiles the exact SU(2) transform
    // instead of inheriting another context's completed structure.
    assert_eq!(second_data, first_data);
}

#[test]
fn concurrent_tree_transform_contexts_do_not_share_compiled_artifacts() {
    let barrier = Arc::new(std::sync::Barrier::new(2));
    let run = |barrier: Arc<std::sync::Barrier>| {
        std::thread::spawn(move || {
            let calls = Arc::new(AtomicUsize::new(0));
            let rule = AdmissionCountingSu2Rule {
                nsymbol_calls: Arc::clone(&calls),
            };
            let structure = simple_su2_vertex_structure(1);
            let operation = TreeTransformOperation::braid([1, 0], [], [0, 1], []);
            let mut cache = TreeTransformCache::<f64, RuleIdentity>::default();
            barrier.wait();
            cache
                .get_or_compile_tree_pair_structures_with_storage_conjugation_ref(
                    &rule, &operation, &structure, &structure, false,
                )
                .unwrap();
            (
                calls.load(Ordering::Relaxed),
                builtin_tree_cache_state(&cache),
            )
        })
    };

    let left = run(Arc::clone(&barrier));
    let right = run(barrier);
    let (left_calls, left_state) = left.join().unwrap();
    let (right_calls, right_state) = right.join().unwrap();

    // What: simultaneous fresh contexts each perform categorical admission and
    // retain one private completed structure instead of observing sibling state.
    assert!(left_calls > 0);
    assert!(right_calls > 0);
    assert_eq!(left_state.0.structure_misses(), 1);
    assert_eq!(right_state.0.structure_misses(), 1);
    assert_eq!(left_state.1, 1);
    assert_eq!(right_state.1, 1);
}

#[test]
fn su2_two_by_two_f_move_uses_one_completed_structure_miss_compiler() {
    use crate::tree_transform::{reset_tree_pair_lowering_calls, tree_pair_lowering_calls};

    let src_key0 = all_codomain_fusion_tree_test_key_for_rule(
        &SU2FusionRule,
        [1, 1, 1, 1],
        0,
        [false, false, false, false],
        [0, 1],
        [1, 1, 1],
    );
    let src_key1 = all_codomain_fusion_tree_test_key_for_rule(
        &SU2FusionRule,
        [1, 1, 1, 1],
        0,
        [false, false, false, false],
        [2, 1],
        [1, 1, 1],
    );
    let block_structure = packed_fixture_structure(
        4,
        [
            (src_key0.clone(), vec![1, 1, 1, 1]),
            (src_key1.clone(), vec![1, 1, 1, 1]),
        ],
    )
    .unwrap();
    let src_space = TensorMapSpace::<4, 0>::from_dims([1, 1, 1, 1], []).unwrap();
    let dst_space = TensorMapSpace::<4, 0>::from_dims([1, 1, 1, 1], []).unwrap();
    let src = TensorMap::<f64, 4, 0>::from_vec_with_structure(
        vec![10.0, 20.0],
        src_space,
        block_structure.clone(),
    )
    .unwrap();
    let dst =
        TensorMap::<f64, 4, 0>::from_vec_with_structure(vec![0.0, 0.0], dst_space, block_structure)
            .unwrap();
    let direct_operation = TreeTransformOperation::braid([0, 2, 1, 3], [], [0, 1, 2, 3], []);
    let miss_operation = TreeTransformOperation::braid(vec![0, 2, 1, 3], [], vec![0, 1, 2, 3], []);
    let hit_operation = TreeTransformOperation::braid([0, 2, 1, 3], [], [0, 1, 2, 3], []);
    assert_eq!(direct_operation, miss_operation);
    assert_eq!(miss_operation, hit_operation);
    let group_count = src.structure().fusion_tree_groups().len();
    let mut cache = TreeTransformCache::<f64, RuleIdentity>::new();

    reset_tree_pair_lowering_calls();
    let direct = tree_transform_structure(&SU2FusionRule, direct_operation, &dst, &src).unwrap();
    assert_eq!(tree_pair_lowering_calls(), (group_count, 0));

    reset_tree_pair_lowering_calls();
    let miss = cache
        .get_or_compile_tree_pair(&SU2FusionRule, miss_operation, &dst, &src)
        .unwrap();
    assert_eq!(tree_pair_lowering_calls(), (group_count, 0));

    reset_tree_pair_lowering_calls();
    let hit = cache
        .get_or_compile_tree_pair(&SU2FusionRule, hit_operation, &dst, &src)
        .unwrap();
    assert_eq!(tree_pair_lowering_calls(), (0, 0));

    // What: independently constructed content-equal operations identify the
    // same completed structure without relying on Arc pointer identity.
    assert!(Arc::ptr_eq(&miss, &hit));
    assert_eq!(cache.structure_len(), 1);
    assert_eq!(cache.stats().structure_misses(), 1);
    assert_eq!(cache.stats().structure_hits(), 1);
    assert!(direct.has_pack_gemm_scatter_blocks());
    assert_eq!(direct.blocks(), miss.blocks());
    assert_eq!(direct.layouts(), miss.layouts());
    assert_eq!(miss.blocks(), hit.blocks());
    assert_eq!(miss.layouts(), hit.layouts());
    let coefficient_bits = |structure: &TreeTransformStructure<f64>| {
        structure
            .gathered_coefficients()
            .iter()
            .map(|coefficient| coefficient.to_bits())
            .collect::<Vec<_>>()
    };
    assert_eq!(coefficient_bits(&direct), coefficient_bits(&miss));
    assert_eq!(coefficient_bits(&miss), coefficient_bits(&hit));

    let replay = |structure: &TreeTransformStructure<f64>| {
        let mut output = dst.clone();
        let mut backend = DenseTreeTransformOperations::default();
        let mut workspace = TreeTransformWorkspace::default();
        tree_transform_execute_with(
            &mut backend,
            &mut workspace,
            structure,
            &mut output,
            &src,
            1.0,
            0.0,
        )
        .unwrap();
        output.data().to_vec()
    };
    let direct_output = replay(&direct);
    let miss_output = replay(&miss);
    let hit_output = replay(&hit);

    let replay_bits = |values: &[f64]| {
        values
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>()
    };
    assert_eq!(replay_bits(&direct_output), replay_bits(&miss_output));
    assert_eq!(replay_bits(&miss_output), replay_bits(&hit_output));
    assert!((direct_output[0] - 22.320_508_075_688_77).abs() < 1.0e-12);
    assert!((direct_output[1] + 1.339_745_962_155_612_7).abs() < 1.0e-12);
}

#[test]
fn tree_transform_cache_compiles_distinct_all_codomain_degeneracy_shapes() {
    let src_key0 = all_codomain_fusion_tree_test_key_for_rule(
        &SU2FusionRule,
        [1, 1, 1, 1],
        0,
        [false, false, false, false],
        [0, 1],
        [1, 1, 1],
    );
    let src_key1 = all_codomain_fusion_tree_test_key_for_rule(
        &SU2FusionRule,
        [1, 1, 1, 1],
        0,
        [false, false, false, false],
        [2, 1],
        [1, 1, 1],
    );
    let small_structure = packed_fixture_structure(
        4,
        [
            (src_key0.clone(), vec![1, 1, 1, 1]),
            (src_key1.clone(), vec![1, 1, 1, 1]),
        ],
    )
    .unwrap();
    let large_structure = packed_fixture_structure(
        4,
        [(src_key0, vec![2, 1, 1, 1]), (src_key1, vec![2, 1, 1, 1])],
    )
    .unwrap();
    let small_space = TensorMapSpace::<4, 0>::from_dims([1, 1, 1, 1], []).unwrap();
    let large_space = TensorMapSpace::<4, 0>::from_dims([2, 1, 1, 1], []).unwrap();
    let src = TensorMap::<f64, 4, 0>::from_vec_with_structure(
        vec![10.0, 20.0],
        small_space.clone(),
        small_structure.clone(),
    )
    .unwrap();
    let mut dst = TensorMap::<f64, 4, 0>::from_vec_with_structure(
        vec![0.0, 0.0],
        small_space,
        small_structure,
    )
    .unwrap();
    let src_large = TensorMap::<f64, 4, 0>::from_vec_with_structure(
        vec![1.0, 2.0, 3.0, 4.0],
        large_space.clone(),
        large_structure.clone(),
    )
    .unwrap();
    let dst_large = TensorMap::<f64, 4, 0>::from_vec_with_structure(
        vec![0.0, 0.0, 0.0, 0.0],
        large_space,
        large_structure,
    )
    .unwrap();
    let operation = TreeTransformOperation::braid([0, 2, 1, 3], [], [0, 1, 2, 3], []);
    let mut cache = TreeTransformCache::<f64, RuleIdentity>::new();

    {
        let structure = cache
            .get_or_compile_all_codomain(&SU2FusionRule, operation.clone(), &dst, &src)
            .unwrap();
        assert!(structure.has_pack_gemm_scatter_blocks());
    }
    assert_eq!(cache.structure_len(), 1);

    {
        let structure = cache
            .get_or_compile_all_codomain(&SU2FusionRule, operation.clone(), &dst, &src)
            .unwrap();
        assert!(structure.has_pack_gemm_scatter_blocks());
    }
    assert_eq!(cache.structure_len(), 1);

    {
        let structure = cache
            .get_or_compile_all_codomain(&SU2FusionRule, operation.clone(), &dst_large, &src_large)
            .unwrap();
        assert!(structure.has_pack_gemm_scatter_blocks());
    }
    assert_eq!(cache.structure_len(), 2);

    let structure = cache
        .get_or_compile_all_codomain(&SU2FusionRule, operation, &dst, &src)
        .unwrap();
    let mut backend = DenseTreeTransformOperations::default();
    let mut workspace = TreeTransformWorkspace::default();
    tree_transform_execute_with(
        &mut backend,
        &mut workspace,
        &structure,
        &mut dst,
        &src,
        1.0,
        0.0,
    )
    .unwrap();

    assert!((dst.data()[0] - 22.320_508_075_688_77).abs() < 1.0e-12);
    assert!((dst.data()[1] + 1.339_745_962_155_612_7).abs() < 1.0e-12);
}

#[test]
fn tree_transform_execution_context_reuses_all_codomain_cache() {
    let src_key0 = all_codomain_fusion_tree_test_key_for_rule(
        &SU2FusionRule,
        [1, 1, 1, 1],
        0,
        [false, false, false, false],
        [0, 1],
        [1, 1, 1],
    );
    let src_key1 = all_codomain_fusion_tree_test_key_for_rule(
        &SU2FusionRule,
        [1, 1, 1, 1],
        0,
        [false, false, false, false],
        [2, 1],
        [1, 1, 1],
    );
    let block_structure = packed_fixture_structure(
        4,
        [
            (src_key0.clone(), vec![1, 1, 1, 1]),
            (src_key1.clone(), vec![1, 1, 1, 1]),
        ],
    )
    .unwrap();
    let space = TensorMapSpace::<4, 0>::from_dims([1, 1, 1, 1], []).unwrap();
    let mut src = TensorMap::<f64, 4, 0>::from_vec_with_structure(
        vec![10.0, 20.0],
        space.clone(),
        block_structure.clone(),
    )
    .unwrap();
    let mut dst =
        TensorMap::<f64, 4, 0>::from_vec_with_structure(vec![0.0, 0.0], space, block_structure)
            .unwrap();
    let operation = TreeTransformOperation::braid([0, 2, 1, 3], [], [0, 1, 2, 3], []);
    let mut context = TreeTransformExecutionContext::<f64, RuleIdentity>::default();
    assert_eq!(context.cache().stats(), TreeTransformCacheStats::default());

    context
        .all_codomain_tree_transform_into(
            &SU2FusionRule,
            operation.clone(),
            &mut dst,
            &src,
            1.0,
            0.0,
        )
        .unwrap();

    assert_eq!(context.cache().structure_len(), 1);
    assert_eq!(context.cache().stats().structure_hits(), 0);
    assert_eq!(context.cache().stats().structure_misses(), 1);
    assert!((dst.data()[0] - 22.320_508_075_688_77).abs() < 1.0e-12);
    assert!((dst.data()[1] + 1.339_745_962_155_612_7).abs() < 1.0e-12);

    src.data_mut().copy_from_slice(&[3.0, -4.0]);
    dst.data_mut().copy_from_slice(&[1.0, 2.0]);
    context
        .all_codomain_tree_transform_into(&SU2FusionRule, operation, &mut dst, &src, 2.0, -1.0)
        .unwrap();

    assert_eq!(context.cache().structure_len(), 1);
    assert_eq!(context.cache().stats().structure_hits(), 1);
    assert_eq!(context.cache().stats().structure_misses(), 1);
    let c = 0.866_025_403_784_438_6;
    assert!((dst.data()[0] - (-1.0 + 2.0 * (0.5 * 3.0 + c * -4.0))).abs() < 1.0e-12);
    assert!((dst.data()[1] - (-2.0 + 2.0 * (c * 3.0 - 0.5 * -4.0))).abs() < 1.0e-12);
    context.cache_mut().reset_stats();
    assert_eq!(context.cache().stats(), TreeTransformCacheStats::default());
}

#[test]
fn all_codomain_artifacts_do_not_cross_context_boundaries() {
    let src_key0 = all_codomain_fusion_tree_test_key_for_rule(
        &SU2FusionRule,
        [1, 1, 1, 1],
        0,
        [false, false, false, false],
        [0, 1],
        [1, 1, 1],
    );
    let src_key1 = all_codomain_fusion_tree_test_key_for_rule(
        &SU2FusionRule,
        [1, 1, 1, 1],
        0,
        [false, false, false, false],
        [2, 1],
        [1, 1, 1],
    );
    let structure = packed_fixture_structure(
        4,
        [(src_key0, vec![1, 1, 1, 1]), (src_key1, vec![1, 1, 1, 1])],
    )
    .unwrap();
    let space = TensorMapSpace::<4, 0>::from_dims([1, 1, 1, 1], []).unwrap();
    let src = TensorMap::<f64, 4, 0>::from_vec_with_structure(
        vec![10.0, 20.0],
        space.clone(),
        structure.clone(),
    )
    .unwrap();
    let dst =
        TensorMap::<f64, 4, 0>::from_vec_with_structure(vec![0.0, 0.0], space, structure).unwrap();
    let operation = TreeTransformOperation::braid([0, 2, 1, 3], [], [0, 1, 2, 3], []);

    let mut first = TreeTransformCache::<f64, RuleIdentity>::default();
    let cold = first
        .get_or_compile_all_codomain(&SU2FusionRule, operation.clone(), &dst, &src)
        .unwrap();
    let warm = first
        .get_or_compile_all_codomain(&SU2FusionRule, operation.clone(), &dst, &src)
        .unwrap();
    assert!(Arc::ptr_eq(&cold, &warm));

    let mut fresh = TreeTransformCache::<f64, RuleIdentity>::default();
    let rebuilt = fresh
        .get_or_compile_all_codomain(&SU2FusionRule, operation, &dst, &src)
        .unwrap();

    // What: all-codomain completed structures remain private to the explicit
    // context while a fresh context rebuilds an equal artifact.
    assert!(!Arc::ptr_eq(&cold, &rebuilt));
    assert_eq!(cold.as_ref(), rebuilt.as_ref());
    assert_eq!(first.stats().structure_hits(), 1);
    assert_eq!(fresh.stats().structure_hits(), 0);
    assert_eq!(fresh.stats().structure_misses(), 1);
}

#[test]
fn tree_transform_execution_context_no_cache_rebuilds_without_retaining_entries() {
    let src_key0 = all_codomain_fusion_tree_test_key_for_rule(
        &SU2FusionRule,
        [1, 1, 1, 1],
        0,
        [false, false, false, false],
        [0, 1],
        [1, 1, 1],
    );
    let src_key1 = all_codomain_fusion_tree_test_key_for_rule(
        &SU2FusionRule,
        [1, 1, 1, 1],
        0,
        [false, false, false, false],
        [2, 1],
        [1, 1, 1],
    );
    let block_structure = packed_fixture_structure(
        4,
        [
            (src_key0.clone(), vec![1, 1, 1, 1]),
            (src_key1.clone(), vec![1, 1, 1, 1]),
        ],
    )
    .unwrap();
    let space = TensorMapSpace::<4, 0>::from_dims([1, 1, 1, 1], []).unwrap();
    let src = TensorMap::<f64, 4, 0>::from_vec_with_structure(
        vec![10.0, 20.0],
        space.clone(),
        block_structure.clone(),
    )
    .unwrap();
    let mut dst =
        TensorMap::<f64, 4, 0>::from_vec_with_structure(vec![0.0, 0.0], space, block_structure)
            .unwrap();
    let operation = TreeTransformOperation::braid([0, 2, 1, 3], [], [0, 1, 2, 3], []);
    let mut context = TreeTransformExecutionContext::<f64, RuleIdentity>::default();
    context
        .cache_mut()
        .set_policy(OperationCachePolicy::NoCache);

    for expected_misses in 1..=2 {
        context
            .all_codomain_tree_transform_into(
                &SU2FusionRule,
                operation.clone(),
                &mut dst,
                &src,
                1.0,
                0.0,
            )
            .unwrap();
        assert_eq!(context.cache().structure_len(), 0);
        assert_eq!(context.cache().stats().structure_hits(), 0);
        assert_eq!(context.cache().stats().structure_misses(), expected_misses);
    }

    let expected = dst.data().to_vec();
    for policy in [
        OperationCachePolicy::TaskLocal,
        OperationCachePolicy::task_local_lru(1),
    ] {
        let mut policy_dst = dst.clone();
        policy_dst.data_mut().fill(0.0);
        let mut policy_context = TreeTransformExecutionContext::<f64, RuleIdentity>::default();
        policy_context.cache_mut().set_policy(policy);
        policy_context
            .all_codomain_tree_transform_into(
                &SU2FusionRule,
                operation.clone(),
                &mut policy_dst,
                &src,
                1.0,
                0.0,
            )
            .unwrap();

        // What: cache ownership and eviction policy do not change the tensor
        // produced by the shared eager compiler.
        assert_eq!(policy_dst.data(), expected);
    }
}

#[test]
fn tree_transform_execution_context_default_is_bounded() {
    let context = TreeTransformExecutionContext::<f64, RuleIdentity>::default();

    // What: ordinary contexts retain at most the documented default number of
    // completed tree-transform structures.
    assert_eq!(
        context.cache().policy(),
        OperationCachePolicy::task_local_lru(256)
    );
}

#[test]
fn tree_transform_execution_context_task_local_lru_evicts_old_transformer() {
    let src_key0 = all_codomain_fusion_tree_test_key_for_rule(
        &SU2FusionRule,
        [1, 1, 1, 1],
        0,
        [false, false, false, false],
        [0, 1],
        [1, 1, 1],
    );
    let src_key1 = all_codomain_fusion_tree_test_key_for_rule(
        &SU2FusionRule,
        [1, 1, 1, 1],
        0,
        [false, false, false, false],
        [2, 1],
        [1, 1, 1],
    );
    let block_structure = packed_fixture_structure(
        4,
        [
            (src_key0.clone(), vec![1, 1, 1, 1]),
            (src_key1.clone(), vec![1, 1, 1, 1]),
        ],
    )
    .unwrap();
    let space = TensorMapSpace::<4, 0>::from_dims([1, 1, 1, 1], []).unwrap();
    let src = TensorMap::<f64, 4, 0>::from_vec_with_structure(
        vec![10.0, 20.0],
        space.clone(),
        block_structure.clone(),
    )
    .unwrap();
    let mut dst =
        TensorMap::<f64, 4, 0>::from_vec_with_structure(vec![0.0, 0.0], space, block_structure)
            .unwrap();
    let operation = TreeTransformOperation::braid([0, 2, 1, 3], [], [0, 1, 2, 3], []);
    let mut context = TreeTransformExecutionContext::<f64, RuleIdentity>::default();
    context
        .cache_mut()
        .set_policy(OperationCachePolicy::task_local_lru(1));

    context
        .tree_transform_into(&SU2FusionRule, operation.clone(), &mut dst, &src, 1.0, 0.0)
        .unwrap();
    assert_eq!(context.cache().structure_len(), 1);

    context
        .all_codomain_tree_transform_into(
            &SU2FusionRule,
            operation.clone(),
            &mut dst,
            &src,
            1.0,
            0.0,
        )
        .unwrap();
    assert_eq!(context.cache().structure_len(), 1);

    context
        .tree_transform_into(&SU2FusionRule, operation.clone(), &mut dst, &src, 1.0, 0.0)
        .unwrap();
    assert_eq!(context.cache().structure_len(), 1);
    assert_eq!(context.cache().stats().structure_hits(), 0);
    assert_eq!(context.cache().stats().structure_misses(), 3);

    context
        .cache_mut()
        .set_policy(OperationCachePolicy::task_local_lru(2));
    context
        .all_codomain_tree_transform_into(
            &SU2FusionRule,
            operation.clone(),
            &mut dst,
            &src,
            1.0,
            0.0,
        )
        .unwrap();
    context
        .tree_transform_into(&SU2FusionRule, operation.clone(), &mut dst, &src, 1.0, 0.0)
        .unwrap();

    context.cache_mut().reset_stats();
    for tree_pair in [false, true, false, true] {
        if tree_pair {
            context
                .tree_transform_into(&SU2FusionRule, operation.clone(), &mut dst, &src, 1.0, 0.0)
                .unwrap();
        } else {
            context
                .all_codomain_tree_transform_into(
                    &SU2FusionRule,
                    operation.clone(),
                    &mut dst,
                    &src,
                    1.0,
                    0.0,
                )
                .unwrap();
        }
    }
    assert_eq!(context.cache().stats().structure_hits(), 4);
    assert_eq!(context.cache().stats().structure_misses(), 0);

    let mut cloned = context.cache().clone();
    cloned.reset_stats();
    cloned
        .get_or_compile_tree_pair(
            &SU2FusionRule,
            TreeTransformOperation::permute([0, 1, 2, 3], []),
            &dst,
            &src,
        )
        .unwrap();
    cloned
        .get_or_compile_all_codomain(&SU2FusionRule, operation, &dst, &src)
        .unwrap();

    // What: cloning a bounded context preserves completed-structure recency.
    assert_eq!(cloned.stats().structure_hits(), 0);
    assert_eq!(cloned.stats().structure_misses(), 2);
}

#[test]
fn tree_transform_execution_context_separates_tree_pair_and_all_codomain_scopes() {
    let src_key0 = all_codomain_fusion_tree_test_key_for_rule(
        &SU2FusionRule,
        [1, 1, 1, 1],
        0,
        [false, false, false, false],
        [0, 1],
        [1, 1, 1],
    );
    let src_key1 = all_codomain_fusion_tree_test_key_for_rule(
        &SU2FusionRule,
        [1, 1, 1, 1],
        0,
        [false, false, false, false],
        [2, 1],
        [1, 1, 1],
    );
    let block_structure = packed_fixture_structure(
        4,
        [
            (src_key0.clone(), vec![1, 1, 1, 1]),
            (src_key1.clone(), vec![1, 1, 1, 1]),
        ],
    )
    .unwrap();
    let space = TensorMapSpace::<4, 0>::from_dims([1, 1, 1, 1], []).unwrap();
    let src = TensorMap::<f64, 4, 0>::from_vec_with_structure(
        vec![10.0, 20.0],
        space.clone(),
        block_structure.clone(),
    )
    .unwrap();
    let mut dst =
        TensorMap::<f64, 4, 0>::from_vec_with_structure(vec![0.0, 0.0], space, block_structure)
            .unwrap();
    let operation = TreeTransformOperation::braid([0, 2, 1, 3], [], [0, 1, 2, 3], []);
    let mut context = TreeTransformExecutionContext::<f64, RuleIdentity>::default();

    context
        .tree_transform_into(&SU2FusionRule, operation.clone(), &mut dst, &src, 1.0, 0.0)
        .unwrap();
    assert_eq!(context.cache().structure_len(), 1);

    dst.data_mut().copy_from_slice(&[0.0, 0.0]);
    context
        .all_codomain_tree_transform_into(&SU2FusionRule, operation, &mut dst, &src, 1.0, 0.0)
        .unwrap();

    assert_eq!(context.cache().structure_len(), 2);
    assert!((dst.data()[0] - 22.320_508_075_688_77).abs() < 1.0e-12);
    assert!((dst.data()[1] + 1.339_745_962_155_612_7).abs() < 1.0e-12);
}
