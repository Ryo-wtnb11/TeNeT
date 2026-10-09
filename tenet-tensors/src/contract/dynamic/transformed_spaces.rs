//! The transformed-source and core-destination derivations of a
//! dynamic-tree contraction. Nothing here retains anything: the derived
//! spaces come from the complete-HomSpace cache and the transformers from
//! the process-global completed-transformer cache (#2014-3).

use super::*;

#[derive(Clone, Debug)]
pub(in crate::contract) struct DynamicFusionTransformedSourceEntry<C = f64> {
    pub(in crate::contract) space: Arc<DynamicFusionMapSpace>,
    pub(in crate::contract) replay_structure: Arc<BlockStructure>,
    pub(in crate::contract) transform_structure: TreeTransformStructure<C>,
}

#[derive(Clone, Debug)]
pub(in crate::contract) struct DynamicFusionCoreDstEntry<C = f64> {
    pub(in crate::contract) space: Arc<DynamicFusionMapSpace>,
    pub(in crate::contract) output_transform_structure: TreeTransformStructure<C>,
}

/// The core-layout space of a stored source and its transformer. A
/// conjugated source is read as its storage-ordered lazy adjoint, keyed on
/// the parent's storage: a per-call adjoint view would take a fresh content
/// id on every call and only evict live transformers.
#[allow(clippy::too_many_arguments)]
pub(super) fn compile_transformed_source<R, D, C, RuleKey, BT>(
    tree_context: &mut TreeTransformExecutionContext<D, RuleKey, C, BT>,
    rule: &R,
    src_space: &DynamicFusionMapSpace,
    src_storage_structure: &Arc<BlockStructure>,
    operation: &TreeTransformOperation,
    source_conjugate: bool,
    layout_primer: LayoutKeyBuilder<R>,
) -> Result<DynamicFusionTransformedSourceEntry<C>, OperationError>
where
    RuleKey: 'static + Clone + Eq + Hash + Send + Sync,
    R: MultiplicityFreeRigidSymbols<Scalar = C> + TreeTransformRuleCacheKey<Key = RuleKey>,
    D: DenseRecouplingScalar + RecouplingCoefficientAction<C>,
    C: DenseBlockScalar,
    BT: TreeTransformBackend<D, C>,
{
    if source_conjugate {
        let oriented = crate::FusionOperand::prepare_storage_ordered_adjoint(src_space, rule)?;
        return compile_transformed_source_oriented(
            tree_context,
            rule,
            &oriented,
            operation,
            layout_primer,
        );
    }
    let space = src_space.transformed_with_primer(rule, operation, layout_primer)?;
    let dst_structure = Arc::clone(space.structure());
    let transform_structure = tree_context.tree_structure(
        rule,
        operation,
        &dst_structure,
        TreeStructureSource::Stored {
            structure: src_storage_structure,
            storage_conjugate: false,
        },
    )?;
    Ok(DynamicFusionTransformedSourceEntry {
        space: Arc::new(space),
        replay_structure: Arc::clone(src_storage_structure),
        transform_structure,
    })
}

/// The core-layout space of a lazy adjoint read through its parent's
/// storage, and its oriented transformer.
pub(super) fn compile_transformed_source_oriented<R, D, C, RuleKey, BT>(
    tree_context: &mut TreeTransformExecutionContext<D, RuleKey, C, BT>,
    rule: &R,
    source: &FusionOperandLayout<'_>,
    operation: &TreeTransformOperation,
    layout_primer: LayoutKeyBuilder<R>,
) -> Result<DynamicFusionTransformedSourceEntry<C>, OperationError>
where
    RuleKey: 'static + Clone + Eq + Hash + Send + Sync,
    R: MultiplicityFreeRigidSymbols<Scalar = C> + TreeTransformRuleCacheKey<Key = RuleKey>,
    D: DenseRecouplingScalar + RecouplingCoefficientAction<C>,
    C: DenseBlockScalar,
    BT: TreeTransformBackend<D, C>,
{
    let space = source.transformed_space(rule, operation, layout_primer)?;
    let dst_structure = Arc::clone(space.structure());
    let transform_structure = tree_context.tree_structure(
        rule,
        operation,
        &dst_structure,
        TreeStructureSource::Oriented(source),
    )?;
    Ok(DynamicFusionTransformedSourceEntry {
        space: Arc::new(space),
        replay_structure: Arc::clone(source.storage_space().structure()),
        transform_structure,
    })
}

/// The core destination space and the output transformer into `output_dst`.
pub(super) fn compile_core_dst<R, D, C, RuleKey, BT>(
    tree_context: &mut TreeTransformExecutionContext<D, RuleKey, C, BT>,
    rule: &R,
    lhs: &DynamicFusionMapSpace,
    rhs: &DynamicFusionMapSpace,
    plan: &FusionContractPlan,
    output_dst: &DynamicFusionMapSpace,
    layout_primer: LayoutKeyBuilder<R>,
) -> Result<DynamicFusionCoreDstEntry<C>, OperationError>
where
    RuleKey: 'static + Clone + Eq + Hash + Send + Sync,
    R: MultiplicityFreeRigidSymbols<Scalar = C> + TreeTransformRuleCacheKey<Key = RuleKey>,
    D: DenseRecouplingScalar + RecouplingCoefficientAction<C>,
    C: DenseBlockScalar,
    BT: TreeTransformBackend<D, C>,
{
    let space = DynamicFusionMapSpace::core_dst_with_primer(rule, lhs, rhs, plan, layout_primer)?;
    let src_structure = Arc::clone(space.structure());
    let output_transform_structure = tree_context.tree_structure(
        rule,
        plan.output_transform(),
        output_dst.structure(),
        TreeStructureSource::Stored {
            structure: &src_structure,
            storage_conjugate: false,
        },
    )?;
    Ok(DynamicFusionCoreDstEntry {
        space: Arc::new(space),
        output_transform_structure,
    })
}
