use super::*;
use std::cell::Cell;
use tenet_core::{
    BlockKey, BlockSpec, FusionProductSpace, FusionRule, FusionTensorMapSpace, SU2FusionRule,
    SectorLeg, TensorMapSpace, U1FusionRule, U1Irrep, Z2FusionRule,
};
use tenet_operations::TensorContractSpec;

use crate::tree_context::TreeTransformExecutionContext;
use crate::{BoundDynamicFusionMapSpace, DenseTreeTransformOperations};
use tenet_operations::OutputAxisOrder;

use super::super::dynamic_space::{checked_metadata_dispatcher, MetadataOutput, MetadataRequest};

thread_local! {
    static EXECUTION_PRIMER_CALLS: Cell<usize> = const { Cell::new(0) };
}

fn counting_su2_primer(
    rule: &SU2FusionRule,
    request: MetadataRequest<'_>,
) -> Result<MetadataOutput, OperationError> {
    EXECUTION_PRIMER_CALLS.with(|calls| calls.set(calls.get() + 1));
    checked_metadata_dispatcher(rule, request)
}

fn reset_execution_primer_calls() {
    EXECUTION_PRIMER_CALLS.with(|calls| calls.set(0));
}

fn execution_primer_calls() -> usize {
    EXECUTION_PRIMER_CALLS.with(Cell::get)
}

fn one_block_structure() -> Arc<BlockStructure> {
    Arc::new(
        BlockStructure::from_blocks_with_rank(
            1,
            vec![BlockSpec::column_major_with_key(BlockKey::opaque([0]), vec![2], 0).unwrap()],
        )
        .unwrap(),
    )
}

fn run_nan_poisoned_destination_case(
    scratch: &mut DynamicFusionScratchWorkspace<f64>,
    open_degeneracy: usize,
) -> (Vec<f64>, usize, usize) {
    let provider = Arc::new(U1FusionRule);
    let wide = || {
        SectorLeg::new(
            [
                (U1Irrep::new(0).sector_id(), 1),
                (U1Irrep::new(1).sector_id(), open_degeneracy),
                (U1Irrep::new(2).sector_id(), 1),
            ],
            false,
        )
    };
    let narrow = || {
        SectorLeg::new(
            [
                (U1Irrep::new(0).sector_id(), 2),
                (U1Irrep::new(1).sector_id(), 1),
            ],
            false,
        )
    };
    let space = |codomain, domain| {
        BoundDynamicFusionMapSpace::from_final_homspace_multiplicity_free(
            Arc::clone(&provider),
            FusionTreeHomSpace::new(
                FusionProductSpace::new(codomain),
                FusionProductSpace::new(domain),
            ),
        )
        .unwrap()
    };
    let lhs = space(vec![wide(), wide()], vec![narrow()]);
    let rhs = space(vec![narrow()], vec![wide()]);
    let output_order = OutputAxisOrder::from_axes(&[1, 0, 2]);
    let axes = TensorContractSpec::new(&[2], &[0], output_order);
    let dst = BoundDynamicFusionMapSpace::contracted_multiplicity_free_ordered(
        &lhs,
        &rhs,
        &[2],
        &[0],
        output_order,
    )
    .unwrap();
    let plan = super::super::fusion::prepare_tensorcontract_fusion_plan_dyn_raw(
        provider.as_ref(),
        dst.space(),
        lhs.space(),
        rhs.space(),
        axes,
    )
    .unwrap();
    assert!(!plan.output_transform_is_identity());

    let mut tree_context =
        TreeTransformExecutionContext::new(DenseTreeTransformOperations::default_executor());
    let mut cache = DynamicFusionSpaceCache::default();
    let artifact = compile_dynamic_tree_execution_artifact::<_, _, _, f64, _, false>(
        &mut tree_context,
        &mut cache,
        provider.as_ref(),
        encoded_layout_primer::<U1FusionRule>,
        &plan,
        dst.space(),
        lhs.space(),
        lhs.space().structure(),
        rhs.space(),
        rhs.space().structure(),
        None,
    )
    .unwrap();
    let core_dst = artifact.core_dst.as_ref().unwrap();
    let core_len = core_dst.space.required_len().unwrap();
    let inactive = artifact.block_plan.inactive_destination_regions().len();
    assert!(inactive > 0, "fixture must include an inactive core block");
    let lhs_data = (0..lhs.space().required_len().unwrap())
        .map(|index| (index as f64 * 0.37 + 0.1).sin())
        .collect::<Vec<_>>();
    let rhs_data = (0..rhs.space().required_len().unwrap())
        .map(|index| (index as f64 * 0.23 + 0.2).cos())
        .collect::<Vec<_>>();
    let execute = |tree_context: &mut TreeTransformExecutionContext<f64, crate::RuleIdentity>,
                   scratch: &mut DynamicFusionScratchWorkspace<f64>,
                   output: &mut [f64]| {
        execute_dynamic_tree_execution_artifact(
            tree_context,
            &mut DenseTreeTransformOperations::default(),
            &mut super::super::backend::TensorContractWorkspace::default(),
            &mut FusionBlockContractWorkspace::default(),
            scratch,
            &artifact,
            dst.space().structure(),
            output,
            &lhs_data,
            &rhs_data,
            1.0,
            0.0,
        )
        .unwrap();
    };
    let mut expected = vec![0.0; dst.space().required_len().unwrap()];
    execute(
        &mut tree_context,
        &mut DynamicFusionScratchWorkspace::default(),
        &mut expected,
    );
    let mut actual = vec![0.0; expected.len()];
    execute(&mut tree_context, scratch, &mut actual);
    assert!(
        scratch
            .dst_data_mut()
            .expect("replay retains destination scratch")
            .iter()
            .all(|value| value.is_finite()),
        "strong zero must initialize active and inactive core blocks"
    );
    assert_eq!(
        actual
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>(),
        expected
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>()
    );
    (actual, core_len, inactive)
}

#[test]
fn dynamic_core_strong_zero_overwrites_nan_scratch_after_shape_change() {
    let mut scratch = DynamicFusionScratchWorkspace::default();
    let (_, first_len, _) = run_nan_poisoned_destination_case(&mut scratch, 4);
    scratch
        .dst_data_mut()
        .expect("first replay retains destination scratch")
        .fill(f64::NAN);
    let (same_shape, same_len, same_inactive) = run_nan_poisoned_destination_case(&mut scratch, 4);
    assert_eq!(first_len, same_len);
    assert!(same_inactive > 0);
    assert!(same_shape.iter().all(|value| value.is_finite()));
    scratch
        .dst_data_mut()
        .expect("same-shape replay retains destination scratch")
        .fill(f64::NAN);

    let (actual, second_len, inactive) = run_nan_poisoned_destination_case(&mut scratch, 2);

    assert_ne!(first_len, second_len, "fixture must change scratch shape");
    assert!(inactive > 0);
    assert!(actual.iter().all(|value| value.is_finite()));
}

#[test]
fn dynamic_fusion_space_cache_default_is_bounded() {
    let cache = DynamicFusionSpaceCache::<crate::RuleIdentity>::default();

    assert_eq!(
        cache.policy(),
        OperationCachePolicy::task_local_lru(DEFAULT_OPERATION_CACHE_ENTRIES)
    );
}

#[test]
fn dynamic_fusion_space_cache_explicit_task_local_stays_unbounded() {
    let mut cache = DynamicFusionSpaceCache::<crate::RuleIdentity>::default();

    cache.set_policy(OperationCachePolicy::TaskLocal);

    assert_eq!(cache.policy(), OperationCachePolicy::TaskLocal);
}

#[test]
fn direct_prelowered_source_reuses_ordinary_transform_entry() {
    let rule = U1FusionRule;
    let charges = [-1, 0, 1].map(|charge| U1Irrep::new(charge).sector_id());
    let codomain = SectorLeg::new(charges.map(|sector| (sector, 1)), false);
    let domain = SectorLeg::new(charges.map(|sector| (rule.dual(sector), 1)), false);
    let homspace = FusionTreeHomSpace::new(
        FusionProductSpace::new([codomain]),
        FusionProductSpace::new([domain]),
    );
    let canonical = FusionTensorMapSpace::from_degeneracy_shapes_coupled(
        TensorMapSpace::<1, 1>::from_dims([3], [3]).unwrap(),
        homspace.clone(),
        &rule,
        vec![vec![1, 1]; 3],
    )
    .unwrap();
    let reversed_keys = (0..canonical.subblock_structure().block_count())
        .rev()
        .map(|index| {
            canonical
                .subblock_structure()
                .block(index)
                .unwrap()
                .key()
                .clone()
        })
        .collect::<Vec<_>>();
    let reordered = FusionTensorMapSpace::new_unbound(
        TensorMapSpace::<1, 1>::from_dims([3], [3]).unwrap(),
        homspace,
        crate::tests::packed_fixture_structure(
            2,
            reversed_keys.into_iter().map(|key| (key, vec![1, 1])),
        )
        .unwrap(),
    )
    .unwrap()
    .try_bind_rule(&rule)
    .unwrap();
    let source = DynamicFusionMapSpace::from_typed(&reordered);
    let layout = super::super::dynamic_space::FusionOperand::direct(&source)
        .prepare(
            &rule,
            super::super::dynamic_space::encoded_layout_primer::<U1FusionRule>,
        )
        .unwrap();
    assert!(layout.is_direct());
    let operation = TreeTransformOperation::transpose([1], [0]);
    let mut tree_context =
        TreeTransformExecutionContext::new(DenseTreeTransformOperations::default_executor());
    let mut cache = DynamicFusionSpaceCache::default();
    let ordinary_cold = cache
        .get_or_compile_transformed_source::<_, f64, _>(
            &mut tree_context,
            &rule,
            &source,
            source.structure(),
            &operation,
            false,
            super::super::dynamic_space::encoded_layout_primer::<U1FusionRule>,
        )
        .unwrap();
    let direct_hit = compile_prelowered_source_transform(
        &mut tree_context,
        &mut cache,
        &rule,
        &layout,
        &operation,
        super::super::dynamic_space::encoded_layout_primer::<U1FusionRule>,
    )
    .unwrap();

    // What: the Direct side of a mixed prelowered contraction uses the
    // ordinary parent-storage compiler and therefore the exact same entry.
    assert!(Arc::ptr_eq(
        &ordinary_cold.transform_structure,
        &direct_hit.transform_structure
    ));
    assert!(Arc::ptr_eq(&ordinary_cold.space, &direct_hit.space));
}

#[test]
fn execution_layout_primer_runs_only_after_dynamic_space_cache_misses() {
    // What: transformed-source and nonidentity-output core spaces invoke
    // the selected primer on a cold miss, while a task-local replay hit
    // returns the shared entry without invoking it again.
    let _guard = crate::test_support::CACHE_TEST_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let rule = SU2FusionRule;
    let homspace = FusionTreeHomSpace::from_sector_ids([(7, 3); 4], []);
    let key_count = homspace.fusion_tree_keys(&rule).len();
    let source =
        DynamicFusionMapSpace::from_degeneracy_shapes(&rule, homspace, vec![vec![3; 4]; key_count])
            .unwrap();
    let scalar_homspace = FusionTreeHomSpace::from_sector_ids([], []);
    let scalar =
        DynamicFusionMapSpace::from_degeneracy_shapes(&rule, scalar_homspace, [vec![]]).unwrap();
    let operation = TreeTransformOperation::permute([0, 2, 1, 3], []);
    let axes = TensorContractSpec::new(&[], &[], OutputAxisOrder::from_axes(&[0, 2, 1, 3]));
    let provider = Arc::new(rule);
    let source_bound =
        super::super::dynamic_space::BoundDynamicFusionMapSpace::bind_multiplicity_free(
            source.clone(),
            Arc::clone(&provider),
        )
        .unwrap();
    let scalar_bound =
        super::super::dynamic_space::BoundDynamicFusionMapSpace::bind_multiplicity_free(
            scalar.clone(),
            Arc::clone(&provider),
        )
        .unwrap();
    let output_bound = super::super::dynamic_space::BoundDynamicFusionMapSpace::contracted_multiplicity_free_ordered(
        &source_bound,
        &scalar_bound,
        axes.lhs_contracting_axes(),
        axes.rhs_contracting_axes(),
        axes.output_permutation(),
    )
    .unwrap();
    let output = output_bound.space().clone();
    let plan = super::super::fusion::prepare_tensorcontract_fusion_plan_dyn_raw(
        &rule, &output, &source, &scalar, axes,
    )
    .unwrap();
    assert!(!plan.output_transform_is_identity());

    crate::reset_global_operation_caches();
    tenet_core::reset_core_intern_tables();
    let mut tree_context =
        TreeTransformExecutionContext::new(DenseTreeTransformOperations::default_executor());
    let mut cache = DynamicFusionSpaceCache::default();
    reset_execution_primer_calls();
    let cold_transform = cache
        .get_or_compile_transformed_source::<_, f64, _>(
            &mut tree_context,
            &rule,
            &source,
            source.structure(),
            &operation,
            false,
            counting_su2_primer,
        )
        .unwrap();
    assert_eq!(execution_primer_calls(), 1);
    let warm_transform = cache
        .get_or_compile_transformed_source::<_, f64, _>(
            &mut tree_context,
            &rule,
            &source,
            source.structure(),
            &operation,
            false,
            counting_su2_primer,
        )
        .unwrap();
    assert_eq!(execution_primer_calls(), 1);
    assert_eq!(cold_transform.space, warm_transform.space);

    crate::reset_global_operation_caches();
    tenet_core::reset_core_intern_tables();
    reset_execution_primer_calls();
    let cold_core = cache
        .get_or_compile_core_dst::<_, f64, _>(
            &mut tree_context,
            &rule,
            &source,
            &scalar,
            &plan,
            &output,
            counting_su2_primer,
        )
        .unwrap();
    assert_eq!(execution_primer_calls(), 1);
    let warm_core = cache
        .get_or_compile_core_dst::<_, f64, _>(
            &mut tree_context,
            &rule,
            &source,
            &scalar,
            &plan,
            &output,
            counting_su2_primer,
        )
        .unwrap();
    assert_eq!(execution_primer_calls(), 1);
    assert_eq!(cold_core.space, warm_core.space);
    assert!(cache.stats().hits() >= 2);

    crate::reset_global_operation_caches();
    tenet_core::reset_core_intern_tables();
    let mut no_cache = DynamicFusionSpaceCache::default();
    no_cache.set_policy(OperationCachePolicy::NoCache);
    reset_execution_primer_calls();
    no_cache
        .get_or_compile_transformed_source::<_, f64, _>(
            &mut tree_context,
            &rule,
            &source,
            source.structure(),
            &TreeTransformOperation::permute([3, 1, 2, 0], []),
            false,
            counting_su2_primer,
        )
        .unwrap();
    assert_eq!(execution_primer_calls(), 1);
    assert_eq!(no_cache.len(), 0);
    assert_eq!(no_cache.stats().hits(), 0);

    crate::reset_global_operation_caches();
    tenet_core::reset_core_intern_tables();
    reset_execution_primer_calls();
    // The DynamicTree artifact compile primes layouts with the destination's
    // capability (#1858). The typed route takes `copyC` for this
    // permute-only contraction, so the artifact compile is driven directly.
    let output_bound = output_bound
        .clone()
        .with_test_layout_primer(counting_su2_primer);
    let compile =
        |context: &mut crate::TensorContractFusionExecutionContext<f64, crate::RuleIdentity>,
         output: &super::super::dynamic_space::BoundDynamicFusionMapSpace<_>,
         axes| {
            let resolution = context
                .compile_storage_contract_resolution(
                    output,
                    crate::FusionOperand::direct(source_bound.space()),
                    crate::FusionOperand::direct(scalar_bound.space()),
                    axes,
                )
                .unwrap();
            assert!(resolution.is_dynamic_tree());
        };
    let mut context =
        crate::TensorContractFusionExecutionContext::<f64, crate::RuleIdentity>::default();
    compile(&mut context, &output_bound, axes);
    let cold_calls = execution_primer_calls();
    assert!(cold_calls > 0);
    compile(&mut context, &output_bound, axes);
    assert_eq!(execution_primer_calls(), cold_calls);

    let mut no_cache_context =
        crate::TensorContractFusionExecutionContext::<f64, crate::RuleIdentity>::default();
    no_cache_context.set_cache_policy(OperationCachePolicy::NoCache);
    reset_execution_primer_calls();
    for expected_minimum in 1..=2 {
        compile(&mut no_cache_context, &output_bound, axes);
        assert!(execution_primer_calls() >= expected_minimum);
        assert_eq!(no_cache_context.dynamic_fusion_space_cache_len(), 0);
    }

    let other_axes = TensorContractSpec::new(&[], &[], OutputAxisOrder::from_axes(&[1, 0, 2, 3]));
    let other_output = super::super::dynamic_space::BoundDynamicFusionMapSpace::contracted_multiplicity_free_ordered(
        &source_bound,
        &scalar_bound,
        other_axes.lhs_contracting_axes(),
        other_axes.rhs_contracting_axes(),
        other_axes.output_permutation(),
    )
    .unwrap()
    .with_test_layout_primer(counting_su2_primer);
    let mut lru_context =
        crate::TensorContractFusionExecutionContext::<f64, crate::RuleIdentity>::default();
    lru_context.set_cache_policy(OperationCachePolicy::task_local_lru(1));
    reset_execution_primer_calls();
    for (output, axes) in [
        (&output_bound, axes),
        (&other_output, other_axes),
        (&output_bound, axes),
    ] {
        compile(&mut lru_context, output, axes);
        assert!(lru_context.dynamic_fusion_space_cache_len() <= 1);
    }
    assert!(execution_primer_calls() >= 3);
}

#[test]
fn dynamic_fusion_fast_space_key_requires_shared_structure_content_identity() {
    let first_structure = one_block_structure();
    let second_structure = one_block_structure();
    assert!(!Arc::ptr_eq(&first_structure, &second_structure));
    assert_eq!(first_structure.as_ref(), second_structure.as_ref());
    assert_ne!(first_structure.content_id(), second_structure.content_id());

    let homspace = Arc::new(FusionTreeHomSpace::from_sector_ids([(0, 2)], []));
    let first = DynamicFusionFastSpaceKey {
        nout: 1,
        homspace: homspace.as_ref().clone(),
        structure_id: first_structure.content_id(),
    };
    let second = DynamicFusionFastSpaceKey {
        nout: 1,
        homspace: homspace.as_ref().clone(),
        structure_id: second_structure.content_id(),
    };

    assert_ne!(first, second);

    let operation = TreeTransformOperation::permute([0], []);
    let first_transform = DynamicFusionTransformedSourceFastKey::<&'static str> {
        rule: "test",
        nout: 1,
        homspace: homspace.as_ref().clone(),
        replay_structure_id: first_structure.content_id(),
        operation: operation.clone(),
        source_conjugate: false,
    };
    let second_transform = DynamicFusionTransformedSourceFastKey::<&'static str> {
        rule: "test",
        nout: 1,
        homspace: homspace.as_ref().clone(),
        replay_structure_id: second_structure.content_id(),
        operation,
        source_conjugate: false,
    };
    assert_ne!(first_transform, second_transform);
}

#[test]
fn borrowable_core_layout_accepts_equal_structure_across_intern_reset() {
    let _guard = crate::test_support::CACHE_TEST_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let rule = Z2FusionRule;
    let build = || {
        FusionTensorMapSpace::from_degeneracy_shapes_coupled(
            TensorMapSpace::<1, 1>::from_dims([1], [1]).unwrap(),
            FusionTreeHomSpace::from_sector_ids([(0, 1)], [(0, 1)]),
            &rule,
            [vec![1, 1]],
        )
        .unwrap()
        .try_bind_rule(&rule)
        .unwrap()
    };
    let source = DynamicFusionMapSpace::from_typed(&build());
    let source_structure = Arc::clone(source.structure());

    crate::cache::reset_global_operation_caches();
    let core = DynamicFusionMapSpace::from_typed(&build());

    // What: equal live layouts created on opposite sides of an intern reset
    // remain borrowable even though their process-local content ids differ.
    assert_ne!(source_structure.content_id(), core.structure().content_id());
    assert!(source_is_borrowable_core_layout(
        &source,
        &source_structure,
        &core,
        &TreeTransformOperation::permute([0], [1]),
        false,
    ));
}

#[test]
fn borrowable_core_layout_defers_homspace_identity_until_after_cheap_gates() {
    // What: nonidentity and conjugating sources skip HomSpace identity,
    // while an otherwise borrowable identity layout performs one comparison.
    let rule = Z2FusionRule;
    let typed = FusionTensorMapSpace::from_degeneracy_shapes_coupled(
        TensorMapSpace::<1, 1>::from_dims([1], [1]).unwrap(),
        FusionTreeHomSpace::from_sector_ids([(0, 1)], [(0, 1)]),
        &rule,
        [vec![1, 1]],
    )
    .unwrap()
    .try_bind_rule(&rule)
    .unwrap();
    let source = DynamicFusionMapSpace::from_typed(&typed);
    let source_structure = Arc::clone(source.structure());

    reset_source_layout_homspace_id_comparisons();
    assert!(!source_is_borrowable_core_layout(
        &source,
        &source_structure,
        &source,
        &TreeTransformOperation::permute([1], [0]),
        false,
    ));
    assert_eq!(source_layout_homspace_id_comparisons(), 0);

    reset_source_layout_homspace_id_comparisons();
    assert!(!source_is_borrowable_core_layout(
        &source,
        &source_structure,
        &source,
        &TreeTransformOperation::permute([0], [1]),
        true,
    ));
    assert_eq!(source_layout_homspace_id_comparisons(), 0);

    reset_source_layout_homspace_id_comparisons();
    assert!(source_is_borrowable_core_layout(
        &source,
        &source_structure,
        &source,
        &TreeTransformOperation::permute([0], [1]),
        false,
    ));
    assert_eq!(source_layout_homspace_id_comparisons(), 1);
}
