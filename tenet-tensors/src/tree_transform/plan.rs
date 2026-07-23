use core::ops::{Add, Mul};
use rustc_hash::{FxHashMap, FxHashSet};
use smallvec::SmallVec;
use std::{collections::hash_map::Entry, sync::Arc};

use num_traits::Zero;
use tenet_core::{
    generic_braid_tree_pair_block_ordered, generic_permute_tree_pair_block_ordered,
    generic_transpose_tree_pair_block_ordered,
    multiplicity_free_braid_tree_pair_block_ordered_indexed,
    multiplicity_free_transpose_tree_pair_block_ordered_indexed, BlockKey, BlockKeyKind,
    BlockStructure, CoreError, FusionRule, FusionStyleKind, FusionTreeBlockGroup, FusionTreeKey,
    FusionTreePairKey, FusionTreePairOrientation, GenericBraidScalar, GenericRigidSymbols,
    LocallyValidatedFusionTreeBlockStructure, MultiplicityFreeFusionSymbols,
    MultiplicityFreeRigidSymbols, OrderedBlockLinearMap, OrderedBlockLinearStorage,
    PreparedTreePairOperation,
};

use crate::{OperationError, TreeTransformStructure};

use super::operation::{
    TreeTransformOperation, TreeTransformOperationKind, ValidateBraidingSupport,
};

pub use tenet_operations::transform_plan::{
    TreeTransformBlockSpec, TreeTransformGroupBlockSpec, TreeTransformGroupPlan,
    TreeTransformKeyBlockSpec,
};

#[cfg(test)]
std::thread_local! {
    static MULTIPLICITY_FREE_CAPABILITY_VALIDATIONS: std::cell::Cell<usize> = const {
        std::cell::Cell::new(0)
    };
}

#[cfg(test)]
pub(crate) fn reset_multiplicity_free_capability_validations() {
    MULTIPLICITY_FREE_CAPABILITY_VALIDATIONS.set(0);
}

#[cfg(test)]
pub(crate) fn multiplicity_free_capability_validations() -> usize {
    MULTIPLICITY_FREE_CAPABILITY_VALIDATIONS.get()
}

pub(crate) fn validate_multiplicity_free_tree_transform_capability<R>(
    rule: &R,
    operation: &TreeTransformOperation,
) -> Result<(), OperationError>
where
    R: FusionRule,
{
    #[cfg(test)]
    MULTIPLICITY_FREE_CAPABILITY_VALIDATIONS
        .set(MULTIPLICITY_FREE_CAPABILITY_VALIDATIONS.get() + 1);
    if !rule.fusion_style().is_multiplicity_free() {
        return Err(OperationError::UnsupportedFusionStyle {
            operation: Box::new(operation.clone()),
            style: rule.fusion_style(),
        });
    }
    operation.validate_braiding_support(rule)
}

pub(crate) fn validate_fusion_tree_key_namespace(
    structure: &BlockStructure,
) -> Result<(), OperationError> {
    match structure.sector_structure().key_kind() {
        None | Some(BlockKeyKind::FusionTree) => Ok(()),
        Some(actual) => Err(OperationError::from_core_preserving_context(
            CoreError::ExpectedFusionTreePairKey { actual },
        )),
    }
}

pub(crate) fn validate_tree_transform_rank_syntax(
    operation: &TreeTransformOperation,
    total_rank: usize,
) -> Result<(), OperationError> {
    PreparedTreePairOperation::validate_rank_syntax(
        total_rank,
        operation.codomain_permutation(),
        operation.domain_permutation(),
    )
    .map_err(OperationError::from_core_preserving_context)
}

pub(crate) fn validate_tree_pair_namespace_before_cache(
    operation: &TreeTransformOperation,
    structure: &BlockStructure,
) -> Result<(), OperationError> {
    match structure.sector_structure().key_kind() {
        None | Some(BlockKeyKind::FusionTree) => Ok(()),
        Some(_) => {
            // Why not probe the categorical cache for application keys: no
            // successful cold compile can publish such an entry. Syntax still
            // precedes namespace rejection to preserve the public error order.
            validate_tree_transform_rank_syntax(operation, structure.rank())?;
            validate_fusion_tree_key_namespace(structure)
        }
    }
}

pub(crate) fn validate_all_codomain_namespace_before_cache(
    operation: &TreeTransformOperation,
    structure: &BlockStructure,
) -> Result<(), OperationError> {
    match structure.sector_structure().key_kind() {
        None | Some(BlockKeyKind::FusionTree) => Ok(()),
        Some(_) => {
            // Why not apply rank-only syntax to categorical sources here:
            // all-codomain admission has an established pair/scope/source
            // precedence that depends on the actual tree split. Noncategorical
            // storage has no split, so only scope and total-rank syntax can
            // precede its namespace rejection.
            validate_all_codomain_operation_scope(operation)?;
            validate_tree_transform_rank_syntax(operation, structure.rank())?;
            validate_fusion_tree_key_namespace(structure)
        }
    }
}

pub(crate) fn validate_multiplicity_free_tree_pair_preflight<'rule, 'structure, R>(
    rule: &'rule R,
    operation: &TreeTransformOperation,
    src_structure: &'structure BlockStructure,
) -> Result<LocallyValidatedFusionTreeBlockStructure<'rule, 'structure, R>, OperationError>
where
    R: FusionRule,
{
    validate_multiplicity_free_tree_transform_capability(rule, operation)?;
    validate_multiplicity_free_tree_pair_preflight_after_capability(rule, operation, src_structure)
}

pub(crate) fn validate_multiplicity_free_tree_pair_preflight_after_capability<
    'rule,
    'structure,
    R,
>(
    rule: &'rule R,
    operation: &TreeTransformOperation,
    src_structure: &'structure BlockStructure,
) -> Result<LocallyValidatedFusionTreeBlockStructure<'rule, 'structure, R>, OperationError>
where
    R: FusionRule,
{
    validate_tree_transform_rank_syntax(operation, src_structure.rank())?;
    validate_fusion_tree_key_namespace(src_structure)?;
    validate_tree_transform_operation_syntax(operation, src_structure)?;
    LocallyValidatedFusionTreeBlockStructure::try_new(rule, src_structure)
        .map_err(OperationError::from_core_preserving_context)
}

pub(crate) fn validate_generic_tree_pair_preflight<'rule, 'structure, R>(
    rule: &'rule R,
    operation: &TreeTransformOperation,
    src_structure: &'structure BlockStructure,
) -> Result<LocallyValidatedFusionTreeBlockStructure<'rule, 'structure, R>, OperationError>
where
    R: FusionRule,
{
    if !rule.fusion_style().has_multiplicity() {
        return Err(OperationError::UnsupportedFusionStyle {
            operation: Box::new(operation.clone()),
            style: rule.fusion_style(),
        });
    }
    operation.validate_braiding_support(rule)?;
    validate_tree_transform_rank_syntax(operation, src_structure.rank())?;
    validate_fusion_tree_key_namespace(src_structure)?;
    validate_tree_transform_operation_syntax(operation, src_structure)?;
    LocallyValidatedFusionTreeBlockStructure::try_new(rule, src_structure)
        .map_err(OperationError::from_core_preserving_context)
}

pub(crate) struct LocallyValidatedAllCodomainFusionTreeBlockStructure<'rule, 'structure, R> {
    proof: LocallyValidatedFusionTreeBlockStructure<'rule, 'structure, R>,
}

impl<'rule, 'structure, R> LocallyValidatedAllCodomainFusionTreeBlockStructure<'rule, 'structure, R>
where
    R: FusionRule,
{
    fn try_new(
        rule: &'rule R,
        operation: &TreeTransformOperation,
        src_structure: &'structure BlockStructure,
    ) -> Result<Self, OperationError> {
        validate_multiplicity_free_tree_transform_capability(rule, operation)?;
        Self::try_new_after_capability(rule, operation, src_structure)
    }

    fn try_new_after_capability(
        rule: &'rule R,
        operation: &TreeTransformOperation,
        src_structure: &'structure BlockStructure,
    ) -> Result<Self, OperationError> {
        validate_all_codomain_namespace_before_cache(operation, src_structure)?;
        let mut first_pair_mismatch = None;
        let mut first_source_restriction = None;
        let mut first_syntax_error = None;
        let mut prepared_splits = SmallVec::<[(usize, usize); 4]>::new();
        for index in 0..src_structure.block_count() {
            let block = src_structure.block(index)?;
            let BlockKey::FusionTree(key) = block.key() else {
                continue;
            };
            if key.codomain_tree().coupled() != key.domain_tree().coupled() {
                first_pair_mismatch.get_or_insert(OperationError::Core(
                    CoreError::MalformedFusionTree {
                        message: "fusion tree pair requires matching coupled sectors",
                    },
                ));
            }
            if first_source_restriction.is_none() {
                first_source_restriction =
                    validate_all_codomain_fusion_tree_block(rule, index, key).err();
            }
            let split = (
                key.codomain_tree().uncoupled().len(),
                key.domain_tree().uncoupled().len(),
            );
            if !prepared_splits.contains(&split) {
                if first_syntax_error.is_none() {
                    first_syntax_error = prepare_tree_pair_operation_syntax(operation, split).err();
                }
                prepared_splits.push(split);
            }
        }
        if let Some(error) = first_pair_mismatch {
            return Err(error);
        }
        validate_all_codomain_operation_scope(operation)?;
        if let Some(error) = first_source_restriction {
            return Err(error);
        }
        if let Some(error) = first_syntax_error {
            return Err(error);
        }
        let proof = LocallyValidatedFusionTreeBlockStructure::try_new(rule, src_structure)
            .map_err(OperationError::from_core_preserving_context)?;
        Ok(Self { proof })
    }

    fn proof(&self) -> &LocallyValidatedFusionTreeBlockStructure<'rule, 'structure, R> {
        &self.proof
    }

    fn rule(&self) -> &'rule R {
        self.proof.rule()
    }

    fn structure(&self) -> &'structure BlockStructure {
        self.proof.structure()
    }
}

pub(crate) fn validate_multiplicity_free_all_codomain_preflight<'rule, 'structure, R>(
    rule: &'rule R,
    operation: &TreeTransformOperation,
    src_structure: &'structure BlockStructure,
) -> Result<LocallyValidatedAllCodomainFusionTreeBlockStructure<'rule, 'structure, R>, OperationError>
where
    R: FusionRule,
{
    LocallyValidatedAllCodomainFusionTreeBlockStructure::try_new(rule, operation, src_structure)
}

pub(crate) fn validate_multiplicity_free_all_codomain_preflight_after_capability<
    'rule,
    'structure,
    R,
>(
    rule: &'rule R,
    operation: &TreeTransformOperation,
    src_structure: &'structure BlockStructure,
) -> Result<LocallyValidatedAllCodomainFusionTreeBlockStructure<'rule, 'structure, R>, OperationError>
where
    R: FusionRule,
{
    LocallyValidatedAllCodomainFusionTreeBlockStructure::try_new_after_capability(
        rule,
        operation,
        src_structure,
    )
}

fn validate_tree_transform_operation_syntax(
    operation: &TreeTransformOperation,
    src_structure: &BlockStructure,
) -> Result<(), OperationError> {
    let mut prepared_splits = SmallVec::<[(usize, usize); 4]>::new();
    for index in 0..src_structure.block_count() {
        let block = src_structure.block(index)?;
        let BlockKey::FusionTree(key) = block.key() else {
            continue;
        };
        let split = (
            key.codomain_tree().uncoupled().len(),
            key.domain_tree().uncoupled().len(),
        );
        if prepared_splits.contains(&split) {
            continue;
        }
        prepare_tree_pair_operation_syntax(operation, split)?;
        prepared_splits.push(split);
    }
    Ok(())
}

/// Build a TensorKit-style grouped tree-transform plan for multiplicity-free
/// fusion rules.
///
/// This is the generic callback form: each source tree may map to multiple
/// destination trees, and duplicate destinations are accumulated into one
/// group-level recoupling matrix. `GenericFusion` with vertex multiplicities is
/// intentionally not represented by this scalar-coefficient API.
///
/// # Provider-domain precondition
///
/// Fusion-tree block keys in `src_structure`, and keys returned by `transform`,
/// follow [`tenet_core::FusionTreeKey::validate_for_rule`]'s provider-domain
/// precondition.
pub fn build_tree_transform_group_plan<T, R, F>(
    rule: &R,
    operation: TreeTransformOperation,
    src_structure: &BlockStructure,
    transform: F,
) -> Result<TreeTransformGroupPlan<T>, OperationError>
where
    R: FusionRule,
    T: Clone + Add<Output = T> + Zero,
    F: FnMut(&FusionTreePairKey) -> Result<Vec<(FusionTreePairKey, T)>, OperationError>,
{
    let source_proof =
        validate_multiplicity_free_tree_pair_preflight(rule, &operation, src_structure)?;
    let mut transform = transform;
    build_tree_transform_group_plan_validated(&source_proof, operation, |source| {
        let rows = transform(source)?;
        for (destination, _) in &rows {
            destination
                .validate_for_rule(source_proof.rule())
                .map_err(OperationError::from_core_preserving_context)?;
        }
        Ok(rows)
    })
}

fn build_tree_transform_group_plan_validated<T, R, F>(
    source_proof: &LocallyValidatedFusionTreeBlockStructure<'_, '_, R>,
    operation: TreeTransformOperation,
    mut transform: F,
) -> Result<TreeTransformGroupPlan<T>, OperationError>
where
    R: FusionRule,
    T: Clone + Add<Output = T> + Zero,
    F: FnMut(&FusionTreePairKey) -> Result<Vec<(FusionTreePairKey, T)>, OperationError>,
{
    let src_structure = source_proof.structure();
    let source_axes = operation_source_axes(&operation);

    let mut specs = Vec::new();
    for group in src_structure.fusion_tree_group_slice() {
        specs.extend(assemble_tree_pair_group_specs(
            src_structure,
            group,
            &source_axes,
            &mut |_, src_key| transform(src_key).map(Arc::new),
        )?);
    }

    Ok(TreeTransformGroupPlan::new(specs))
}

/// Standard all-codomain tree-transform builder for Unique and Simple
/// multiplicity-free rules.
///
/// # Provider-domain precondition
///
/// Fusion-tree block keys in `src_structure` follow
/// [`tenet_core::FusionTreeKey::validate_for_rule`]'s provider-domain
/// precondition.
pub fn build_all_codomain_tree_transform_group_plan<R>(
    rule: &R,
    operation: TreeTransformOperation,
    src_structure: &BlockStructure,
) -> Result<TreeTransformGroupPlan<R::Scalar>, OperationError>
where
    R: MultiplicityFreeFusionSymbols,
    R::Scalar: Clone + Add<Output = R::Scalar> + Mul<Output = R::Scalar> + Zero,
{
    if rule.fusion_style() == FusionStyleKind::Unique {
        build_unique_all_codomain_tree_transform_group_plan(rule, operation, src_structure)
    } else {
        build_multiplicity_free_all_codomain_tree_transform_group_plan(
            rule,
            operation,
            src_structure,
        )
    }
}

/// Standard full tree-pair transform builder for Unique and Simple
/// multiplicity-free rules.
///
/// # Provider-domain precondition
///
/// Fusion-tree block keys in `src_structure` follow
/// [`tenet_core::FusionTreeKey::validate_for_rule`]'s provider-domain
/// precondition.
pub fn build_tree_pair_transform_group_plan<R>(
    rule: &R,
    operation: TreeTransformOperation,
    src_structure: &BlockStructure,
) -> Result<TreeTransformGroupPlan<R::Scalar>, OperationError>
where
    R: MultiplicityFreeRigidSymbols,
    R::Scalar: Clone + Add<Output = R::Scalar> + Mul<Output = R::Scalar> + Zero,
{
    if rule.fusion_style() == FusionStyleKind::Unique {
        build_unique_tree_pair_transform_group_plan(rule, operation, src_structure)
    } else {
        build_multiplicity_free_tree_pair_transform_group_plan(rule, operation, src_structure)
    }
}

fn build_tree_pair_transform_group_plan_validated<R>(
    source_proof: &LocallyValidatedFusionTreeBlockStructure<'_, '_, R>,
    operation: TreeTransformOperation,
) -> Result<TreeTransformGroupPlan<R::Scalar>, OperationError>
where
    R: MultiplicityFreeRigidSymbols,
    R::Scalar: Clone + Add<Output = R::Scalar> + Mul<Output = R::Scalar> + Zero,
{
    if source_proof.rule().fusion_style() == FusionStyleKind::Unique {
        build_unique_tree_pair_transform_group_plan_validated(source_proof, operation)
    } else {
        build_multiplicity_free_tree_pair_transform_group_plan_validated(source_proof, operation)
    }
}

pub(crate) fn compile_multiplicity_free_tree_pair_structure<R>(
    rule: &R,
    operation: &TreeTransformOperation,
    dst_structure: Arc<BlockStructure>,
    src_structure: Arc<BlockStructure>,
    storage_conjugate: bool,
) -> Result<TreeTransformStructure<R::Scalar>, OperationError>
where
    R: MultiplicityFreeRigidSymbols,
    R::Scalar: Copy + Clone + Add<Output = R::Scalar> + Mul<Output = R::Scalar> + Zero,
{
    compile_multiplicity_free_tree_pair_structure_with(
        rule,
        operation,
        dst_structure,
        src_structure,
        storage_conjugate,
        |source_proof, operation| {
            build_tree_pair_transform_group_plan_validated(source_proof, operation.clone())
        },
    )
}

pub(crate) fn compile_multiplicity_free_tree_pair_structure_with_threads<R>(
    rule: &R,
    operation: &TreeTransformOperation,
    dst_structure: Arc<BlockStructure>,
    src_structure: Arc<BlockStructure>,
    storage_conjugate: bool,
    threads: usize,
) -> Result<TreeTransformStructure<R::Scalar>, OperationError>
where
    R: MultiplicityFreeRigidSymbols,
    R::Scalar:
        Copy + Clone + Add<Output = R::Scalar> + Mul<Output = R::Scalar> + Zero + Send + Sync,
{
    compile_multiplicity_free_tree_pair_structure_with(
        rule,
        operation,
        dst_structure,
        src_structure,
        storage_conjugate,
        |source_proof, operation| {
            build_tree_pair_transform_group_plan_validated_with_threads(
                source_proof,
                operation.clone(),
                threads,
            )
        },
    )
}

pub(crate) fn compile_multiplicity_free_tree_pair_structure_after_capability_with_threads<R>(
    rule: &R,
    operation: &TreeTransformOperation,
    dst_structure: Arc<BlockStructure>,
    src_structure: Arc<BlockStructure>,
    storage_conjugate: bool,
    threads: usize,
) -> Result<TreeTransformStructure<R::Scalar>, OperationError>
where
    R: MultiplicityFreeRigidSymbols,
    R::Scalar:
        Copy + Clone + Add<Output = R::Scalar> + Mul<Output = R::Scalar> + Zero + Send + Sync,
{
    let replay_src_structure = Arc::clone(&src_structure);
    let source_proof = validate_multiplicity_free_tree_pair_preflight_after_capability(
        rule,
        operation,
        &src_structure,
    )?;
    finish_multiplicity_free_tree_pair_structure(
        source_proof,
        operation,
        dst_structure,
        replay_src_structure,
        storage_conjugate,
        |source_proof, operation| {
            build_tree_pair_transform_group_plan_validated_with_threads(
                source_proof,
                operation.clone(),
                threads,
            )
        },
    )
}

fn compile_multiplicity_free_tree_pair_structure_with<R, F>(
    rule: &R,
    operation: &TreeTransformOperation,
    dst_structure: Arc<BlockStructure>,
    src_structure: Arc<BlockStructure>,
    storage_conjugate: bool,
    build_plan: F,
) -> Result<TreeTransformStructure<R::Scalar>, OperationError>
where
    R: MultiplicityFreeRigidSymbols,
    R::Scalar: Copy,
    F: FnOnce(
        &LocallyValidatedFusionTreeBlockStructure<'_, '_, R>,
        &TreeTransformOperation,
    ) -> Result<TreeTransformGroupPlan<R::Scalar>, OperationError>,
{
    let replay_src_structure = Arc::clone(&src_structure);
    let source_proof =
        validate_multiplicity_free_tree_pair_preflight(rule, operation, &src_structure)?;
    finish_multiplicity_free_tree_pair_structure(
        source_proof,
        operation,
        dst_structure,
        replay_src_structure,
        storage_conjugate,
        build_plan,
    )
}

fn finish_multiplicity_free_tree_pair_structure<R, F>(
    source_proof: LocallyValidatedFusionTreeBlockStructure<'_, '_, R>,
    operation: &TreeTransformOperation,
    dst_structure: Arc<BlockStructure>,
    src_structure: Arc<BlockStructure>,
    storage_conjugate: bool,
    build_plan: F,
) -> Result<TreeTransformStructure<R::Scalar>, OperationError>
where
    R: MultiplicityFreeRigidSymbols,
    R::Scalar: Copy,
    F: FnOnce(
        &LocallyValidatedFusionTreeBlockStructure<'_, '_, R>,
        &TreeTransformOperation,
    ) -> Result<TreeTransformGroupPlan<R::Scalar>, OperationError>,
{
    LocallyValidatedFusionTreeBlockStructure::try_new(source_proof.rule(), &dst_structure)
        .map_err(OperationError::from_core_preserving_context)?;
    let plan = build_plan(&source_proof, operation)?;
    drop(source_proof);
    plan.compile_shared_structures_with_storage_conjugation(
        dst_structure,
        src_structure,
        storage_conjugate,
    )
}

pub(crate) fn build_all_codomain_tree_transform_group_plan_validated_with_threads<R>(
    source_proof: &LocallyValidatedAllCodomainFusionTreeBlockStructure<'_, '_, R>,
    operation: TreeTransformOperation,
    threads: usize,
) -> Result<TreeTransformGroupPlan<R::Scalar>, OperationError>
where
    R: MultiplicityFreeFusionSymbols + Sync,
    R::Scalar: Clone + Add<Output = R::Scalar> + Mul<Output = R::Scalar> + Zero + Send + Sync,
{
    if source_proof.rule().fusion_style() == FusionStyleKind::Unique {
        build_unique_all_codomain_tree_transform_group_plan_validated(source_proof, operation)
    } else {
        build_multiplicity_free_all_codomain_tree_transform_group_plan_validated_with_threads(
            source_proof,
            operation,
            threads,
        )
    }
}

pub(crate) fn build_tree_pair_transform_group_plan_validated_with_threads<R>(
    source_proof: &LocallyValidatedFusionTreeBlockStructure<'_, '_, R>,
    operation: TreeTransformOperation,
    threads: usize,
) -> Result<TreeTransformGroupPlan<R::Scalar>, OperationError>
where
    R: MultiplicityFreeRigidSymbols,
    R::Scalar: Clone + Add<Output = R::Scalar> + Mul<Output = R::Scalar> + Zero + Send + Sync,
{
    if source_proof.rule().fusion_style() == FusionStyleKind::Unique {
        build_unique_tree_pair_transform_group_plan_validated(source_proof, operation)
    } else {
        build_multiplicity_free_tree_pair_transform_group_plan_validated_with_threads(
            source_proof,
            operation,
            threads,
        )
    }
}

#[cfg(test)]
pub(crate) fn build_unique_tree_transform_group_plan<T, R, F>(
    rule: &R,
    operation: TreeTransformOperation,
    src_structure: &BlockStructure,
    transform: F,
) -> Result<TreeTransformGroupPlan<T>, OperationError>
where
    R: FusionRule,
    T: Clone + Add<Output = T> + Zero,
    F: FnMut(&FusionTreePairKey) -> Result<(FusionTreePairKey, T), OperationError>,
{
    if rule.fusion_style() != FusionStyleKind::Unique {
        return Err(OperationError::UnsupportedFusionStyle {
            operation: Box::new(operation),
            style: rule.fusion_style(),
        });
    }
    let mut transform = transform;
    build_tree_transform_group_plan(rule, operation, src_structure, |source| {
        transform(source).map(|row| vec![row])
    })
}

pub(crate) fn build_unique_all_codomain_tree_transform_group_plan<R>(
    rule: &R,
    operation: TreeTransformOperation,
    src_structure: &BlockStructure,
) -> Result<TreeTransformGroupPlan<R::Scalar>, OperationError>
where
    R: MultiplicityFreeFusionSymbols,
    R::Scalar: Clone + Add<Output = R::Scalar> + Mul<Output = R::Scalar> + Zero,
{
    if rule.fusion_style() != FusionStyleKind::Unique {
        return Err(OperationError::UnsupportedFusionStyle {
            operation: Box::new(operation),
            style: rule.fusion_style(),
        });
    }
    let source_proof =
        validate_multiplicity_free_all_codomain_preflight(rule, &operation, src_structure)?;
    build_unique_all_codomain_tree_transform_group_plan_validated(&source_proof, operation)
}

fn build_unique_all_codomain_tree_transform_group_plan_validated<R>(
    source_proof: &LocallyValidatedAllCodomainFusionTreeBlockStructure<'_, '_, R>,
    operation: TreeTransformOperation,
) -> Result<TreeTransformGroupPlan<R::Scalar>, OperationError>
where
    R: MultiplicityFreeFusionSymbols,
    R::Scalar: Clone + Add<Output = R::Scalar> + Mul<Output = R::Scalar> + Zero,
{
    debug_assert_eq!(source_proof.rule().fusion_style(), FusionStyleKind::Unique);
    let proof = source_proof.proof();
    let src_structure = source_proof.structure();
    let source_axes = operation_source_axes(&operation);
    let mut specs = Vec::with_capacity(src_structure.block_count());
    for index in 0..src_structure.block_count() {
        let Some(src_key) = proof.fusion_tree_pair_key(index)? else {
            continue;
        };
        let mut rows = match operation.kind() {
            TreeTransformOperationKind::Permute => {
                proof.permute_codomain_rows_for_block_index(index, operation.codomain_permutation())
            }
            TreeTransformOperationKind::Braid => proof.braid_codomain_rows_for_block_index(
                index,
                operation.codomain_permutation(),
                operation.codomain_levels(),
            ),
            TreeTransformOperationKind::Transpose => {
                unreachable!("all-codomain admission rejected transpose")
            }
        }
        .map_err(OperationError::from_core_preserving_context)?;
        let Some((destination, coefficient)) = rows.pop() else {
            return Err(OperationError::EmptyTransformBlock);
        };
        if !rows.is_empty() {
            return Err(OperationError::StructureMismatch {
                tensor: "proof-bound Unique all-codomain cardinality",
            });
        }
        let dst_key = FusionTreePairKey::pair(destination, src_key.domain_tree().clone());
        specs.push(
            TreeTransformGroupBlockSpec::single(dst_key, src_key.clone(), coefficient)
                .with_shared_source_axes(Arc::clone(&source_axes)),
        );
    }
    Ok(TreeTransformGroupPlan::new(specs))
}

pub(crate) fn build_multiplicity_free_all_codomain_tree_transform_group_plan<R>(
    rule: &R,
    operation: TreeTransformOperation,
    src_structure: &BlockStructure,
) -> Result<TreeTransformGroupPlan<R::Scalar>, OperationError>
where
    R: MultiplicityFreeFusionSymbols,
    R::Scalar: Clone + Add<Output = R::Scalar> + Mul<Output = R::Scalar> + Zero,
{
    let source_proof =
        validate_multiplicity_free_all_codomain_preflight(rule, &operation, src_structure)?;
    build_multiplicity_free_all_codomain_tree_transform_group_plan_validated(
        &source_proof,
        operation,
    )
}

fn build_multiplicity_free_all_codomain_tree_transform_group_plan_validated<R>(
    source_proof: &LocallyValidatedAllCodomainFusionTreeBlockStructure<'_, '_, R>,
    operation: TreeTransformOperation,
) -> Result<TreeTransformGroupPlan<R::Scalar>, OperationError>
where
    R: MultiplicityFreeFusionSymbols,
    R::Scalar: Clone + Add<Output = R::Scalar> + Mul<Output = R::Scalar> + Zero,
{
    let src_structure = source_proof.structure();
    let source_axes = operation_source_axes(&operation);
    let mut specs = Vec::new();
    for group in src_structure.fusion_tree_group_slice() {
        specs.extend(build_one_multiplicity_free_all_codomain_group(
            source_proof,
            &operation,
            &source_axes,
            group,
        )?);
    }
    Ok(TreeTransformGroupPlan::new(specs))
}

fn build_one_multiplicity_free_all_codomain_group<R>(
    source_proof: &LocallyValidatedAllCodomainFusionTreeBlockStructure<'_, '_, R>,
    operation: &TreeTransformOperation,
    source_axes: &Arc<[usize]>,
    group: &FusionTreeBlockGroup,
) -> Result<Vec<TreeTransformGroupBlockSpec<R::Scalar>>, OperationError>
where
    R: MultiplicityFreeFusionSymbols,
    R::Scalar: Clone + Add<Output = R::Scalar> + Mul<Output = R::Scalar> + Zero,
{
    let mut unique_index_by_tree = FxHashMap::default();
    let mut unique_indices = SmallVec::<[usize; 8]>::new();
    let mut source_alignment = SmallVec::<[usize; 8]>::new();
    for &src_block_index in group.block_indices() {
        let Some(src_key) = source_proof.proof().fusion_tree_pair_key(src_block_index)? else {
            return Err(OperationError::ExpectedFusionTreeBlock {
                tensor: "src",
                index: src_block_index,
            });
        };
        let codomain_tree = src_key.codomain_tree();
        let unique_index = match unique_index_by_tree.get(codomain_tree) {
            Some(&index) => index,
            None => {
                let index = unique_indices.len();
                unique_index_by_tree.insert(codomain_tree, index);
                unique_indices.push(src_block_index);
                index
            }
        };
        source_alignment.push(unique_index);
    }

    let transformed = transform_all_codomain_rows_for_block_indices(
        source_proof,
        operation,
        &mut unique_indices.iter().copied(),
    )?;
    if transformed.len() != unique_indices.len() {
        return Err(OperationError::CoefficientCountMismatch {
            expected: unique_indices.len(),
            actual: transformed.len(),
        });
    }
    let transformed = transformed.into_iter().map(Arc::new).collect::<Vec<_>>();
    let mut source_cursor = 0usize;
    let mut rows_for = |_: usize, _: &FusionTreeKey| {
        let unique_index = source_alignment.get(source_cursor).copied();
        source_cursor += 1;
        unique_index
            .and_then(|index| transformed.get(index))
            .map(Arc::clone)
            .ok_or(OperationError::StructureMismatch {
                tensor: "compile-local all-codomain rows",
            })
    };
    let specs = if operation.is_identity_for(group.group_key().codomain_uncoupled().len(), 0) {
        assemble_identity_all_codomain_group_specs(
            source_proof.structure(),
            group,
            source_axes,
            &mut rows_for,
        )
    } else {
        assemble_all_codomain_group_specs(
            source_proof.structure(),
            group,
            source_axes,
            &mut rows_for,
        )
    }?;
    if source_cursor != source_alignment.len() {
        return Err(OperationError::StructureMismatch {
            tensor: "compile-local all-codomain row order",
        });
    }
    Ok(specs)
}

fn build_multiplicity_free_all_codomain_tree_transform_group_plan_validated_with_threads<R>(
    source_proof: &LocallyValidatedAllCodomainFusionTreeBlockStructure<'_, '_, R>,
    operation: TreeTransformOperation,
    threads: usize,
) -> Result<TreeTransformGroupPlan<R::Scalar>, OperationError>
where
    R: MultiplicityFreeFusionSymbols + Sync,
    R::Scalar: Clone + Add<Output = R::Scalar> + Mul<Output = R::Scalar> + Zero + Send + Sync,
{
    let src_structure = source_proof.structure();
    let source_axes = operation_source_axes(&operation);
    let groups = src_structure
        .fusion_tree_group_slice()
        .iter()
        .collect::<Vec<_>>();
    let completed = execute_staged_groups(groups, threads, |group| {
        build_one_multiplicity_free_all_codomain_group(
            source_proof,
            &operation,
            &source_axes,
            group,
        )
    })?;

    let mut specs = Vec::new();
    for group_specs in completed {
        specs.extend(group_specs);
    }
    Ok(TreeTransformGroupPlan::new(specs))
}

type TransformRows<K, T> = Vec<(K, T)>;

#[cfg(test)]
mod generic_preflight_tests {
    use super::{validate_generic_tree_pair_preflight, TreeTransformOperation};
    use tenet_core::{
        BlockKey, BlockSpec, BlockStructure, CoreError, FusionTreeKey, FusionTreePairKey,
        MultiplicityIndex, SectorId, Su3FusionRule,
    };
    use tenet_operations::OperationError;

    #[test]
    fn su3_generic_preflight_accepts_valid_permute_and_braid_before_categorical_admission() {
        let rule = Su3FusionRule::new();
        let eight = rule.sector_of(1, 1).unwrap();
        let vacuum = SectorId::new(0);
        let valid = FusionTreePairKey::pair(
            FusionTreeKey::try_new_for_rule(
                &rule,
                [eight, eight],
                vacuum,
                [false, false],
                [],
                [MultiplicityIndex::ONE],
            )
            .unwrap(),
            FusionTreeKey::try_new_for_rule(&rule, [], vacuum, [], [], []).unwrap(),
        );
        let valid_structure = BlockStructure::from_blocks(vec![BlockSpec::column_major_with_key(
            BlockKey::from(valid),
            vec![1, 1],
            0,
        )
        .unwrap()])
        .unwrap();

        for operation in [
            TreeTransformOperation::permute([1, 0], []),
            TreeTransformOperation::braid([1, 0], [], [0, 1], []),
        ] {
            // What: style-neutral syntax validation admits valid Generic
            // operations without constructing an Artin execution plan.
            validate_generic_tree_pair_preflight(&rule, &operation, &valid_structure).unwrap();
        }

        assert_eq!(
            FusionTreePairKey::try_pair_from_sector_ids(
                [eight.id(), eight.id()],
                [],
                vacuum.id(),
                [false, false],
                [],
                [],
                [],
                [0],
                [],
            )
            .unwrap_err(),
            CoreError::InvalidMultiplicityIndex { value: 0 }
        );

        let malformed = FusionTreePairKey::try_pair_from_sector_ids(
            [eight.id(), eight.id()],
            [],
            vacuum.id(),
            [false, false],
            [],
            [],
            [],
            [2],
            [],
        )
        .unwrap();
        let malformed_structure =
            BlockStructure::from_blocks(vec![BlockSpec::column_major_with_key(
                BlockKey::from(malformed),
                vec![1, 1],
                0,
            )
            .unwrap()])
            .unwrap();
        let error = match validate_generic_tree_pair_preflight(
            &rule,
            &TreeTransformOperation::braid([0, 0], [], [0, 1], []),
            &malformed_structure,
        ) {
            Ok(_) => panic!("invalid Generic operation unexpectedly admitted"),
            Err(error) => error,
        };

        // What: invalid operation syntax retains precedence over malformed
        // Generic categorical data.
        assert_eq!(
            error,
            OperationError::Core(CoreError::InvalidPermutation {
                permutation: vec![0, 0],
                rank: 2,
            })
        );
    }

    #[test]
    fn generic_preflight_checks_rank_syntax_before_opaque_namespace() {
        let rule = Su3FusionRule::new();
        let structure = BlockStructure::from_blocks(vec![BlockSpec::column_major_with_key(
            BlockKey::opaque([4, 2]),
            vec![1, 1],
            0,
        )
        .unwrap()])
        .unwrap();

        let syntax_error = match validate_generic_tree_pair_preflight(
            &rule,
            &TreeTransformOperation::permute([0, 0], []),
            &structure,
        ) {
            Ok(_) => panic!("invalid operation unexpectedly admitted"),
            Err(error) => error,
        };
        assert_eq!(
            syntax_error,
            OperationError::Core(CoreError::InvalidPermutation {
                permutation: vec![0, 0],
                rank: 2,
            })
        );

        let namespace_error = match validate_generic_tree_pair_preflight(
            &rule,
            &TreeTransformOperation::permute([0, 1], []),
            &structure,
        ) {
            Ok(_) => panic!("opaque namespace unexpectedly admitted"),
            Err(error) => error,
        };
        // What: Generic and multiplicity-free cold admission share the same
        // operation-before-namespace boundary.
        assert_eq!(
            namespace_error,
            OperationError::Core(CoreError::ExpectedFusionTreePairKey {
                actual: tenet_core::BlockKeyKind::Opaque,
            })
        );
    }
}

fn execute_staged_groups<I, O, F>(
    inputs: Vec<I>,
    threads: usize,
    build: F,
) -> Result<Vec<O>, OperationError>
where
    I: Send,
    O: Send,
    F: Fn(I) -> Result<O, OperationError> + Send + Sync,
{
    if threads <= 1 || inputs.len() <= 1 {
        return inputs.into_iter().map(build).collect();
    }

    use rayon::prelude::*;

    let batches = partition_staged_groups(inputs, threads)
        .into_par_iter()
        .map(|batch| batch.into_iter().map(&build).collect::<Result<Vec<_>, _>>())
        .collect::<Vec<_>>();
    flatten_ordered_batch_results(batches)
}

fn flatten_ordered_batch_results<O, E>(batches: Vec<Result<Vec<O>, E>>) -> Result<Vec<O>, E> {
    let mut outputs = Vec::new();
    // Why not collect the parallel iterator directly into `Result<Vec<_>, _>`:
    // Rayon deliberately leaves the winning error nondeterministic when
    // several batches fail. Folding ordered batch results here preserves the
    // serial source-group error precedence without publishing partial state.
    for batch in batches {
        outputs.extend(batch?);
    }
    Ok(outputs)
}

#[cfg(test)]
mod ordered_batch_tests {
    use super::flatten_ordered_batch_results;

    #[test]
    fn flatten_ordered_batches_selects_first_error_and_preserves_success_order() {
        // What: collector semantics follow source-batch order even when several
        // worker results contain distinct errors.
        let errors = vec![Ok(vec![0, 1]), Err("first"), Err("second")];
        assert_eq!(flatten_ordered_batch_results(errors), Err("first"));

        // What: successful batches flatten without reordering their rows.
        let successes: Vec<Result<Vec<usize>, &str>> =
            vec![Ok(vec![0, 1]), Ok(vec![2]), Ok(vec![3, 4])];
        assert_eq!(
            flatten_ordered_batch_results(successes),
            Ok(vec![0, 1, 2, 3, 4])
        );
    }
}

fn partition_staged_groups<I>(inputs: Vec<I>, threads: usize) -> Vec<Vec<I>> {
    let batch_count = threads.max(1).min(inputs.len());
    if batch_count == 0 {
        return Vec::new();
    }
    let minimum_size = inputs.len() / batch_count;
    let larger_batches = inputs.len() % batch_count;
    let mut inputs = inputs.into_iter();
    (0..batch_count)
        .map(|batch| {
            let size = minimum_size + usize::from(batch < larger_batches);
            inputs.by_ref().take(size).collect()
        })
        .collect()
}

#[cfg(test)]
pub(crate) fn partition_staged_groups_for_test<I>(inputs: Vec<I>, threads: usize) -> Vec<Vec<I>> {
    partition_staged_groups(inputs, threads)
}

fn transform_all_codomain_rows_for_block_indices<R>(
    proof: &LocallyValidatedAllCodomainFusionTreeBlockStructure<'_, '_, R>,
    operation: &TreeTransformOperation,
    block_indices: &mut dyn Iterator<Item = usize>,
) -> Result<Vec<TransformRows<FusionTreeKey, R::Scalar>>, OperationError>
where
    R: MultiplicityFreeFusionSymbols,
    R::Scalar: Clone + Add<Output = R::Scalar> + Mul<Output = R::Scalar>,
{
    let Some(first_index) = block_indices.next() else {
        return Ok(Vec::new());
    };
    let indices = std::iter::once(first_index).chain(block_indices);
    let transformed = match operation.kind() {
        TreeTransformOperationKind::Permute => proof
            .proof()
            .permute_codomain_rows_for_block_indices(indices, operation.codomain_permutation()),
        TreeTransformOperationKind::Braid => proof.proof().braid_codomain_rows_for_block_indices(
            indices,
            operation.codomain_permutation(),
            operation.codomain_levels(),
        ),
        TreeTransformOperationKind::Transpose => {
            unreachable!("all-codomain operation scope validation rejected transpose")
        }
    };
    transformed.map_err(OperationError::from_core_preserving_context)
}

fn assemble_identity_all_codomain_group_specs<T, F>(
    src_structure: &BlockStructure,
    group: &FusionTreeBlockGroup,
    source_axes: &Arc<[usize]>,
    rows_for: &mut F,
) -> Result<Vec<TreeTransformGroupBlockSpec<T>>, OperationError>
where
    T: Clone,
    F: FnMut(usize, &FusionTreeKey) -> Result<Arc<Vec<(FusionTreeKey, T)>>, OperationError>,
{
    let mut specs = Vec::with_capacity(group.block_indices().len());
    for &src_block_index in group.block_indices() {
        let block = src_structure.block(src_block_index)?;
        let BlockKey::FusionTree(src_key) = block.key() else {
            return Err(OperationError::ExpectedFusionTreeBlock {
                tensor: "src",
                index: src_block_index,
            });
        };
        let transformed = rows_for(src_block_index, src_key.codomain_tree())?;
        let [(dst_codomain_tree, coefficient)] = transformed.as_slice() else {
            return Err(OperationError::EmptyTransformBlock);
        };
        let dst_key =
            FusionTreePairKey::pair(dst_codomain_tree.clone(), src_key.domain_tree().clone());
        specs.push(
            TreeTransformGroupBlockSpec::single(dst_key, src_key.clone(), coefficient.clone())
                .with_shared_source_axes(Arc::clone(source_axes)),
        );
    }
    Ok(specs)
}

struct DenseRecouplingRows<T> {
    coefficients: Vec<T>,
    src_count: usize,
}

impl<T> DenseRecouplingRows<T>
where
    T: Clone + Add<Output = T> + Zero,
{
    fn new(src_count: usize) -> Self {
        Self {
            coefficients: Vec::with_capacity(src_count),
            src_count,
        }
    }

    fn push_zero_row(&mut self) -> usize {
        let row = self.coefficients.len() / self.src_count;
        self.coefficients
            .resize_with(self.coefficients.len() + self.src_count, T::zero);
        row
    }

    fn add(&mut self, dst_row: usize, src_column: usize, coefficient: T) {
        let index = dst_row * self.src_count + src_column;
        self.coefficients[index] = self.coefficients[index].clone() + coefficient;
    }

    fn into_coefficients(self) -> Vec<T> {
        self.coefficients
    }
}

fn assemble_all_codomain_group_specs<T, F>(
    src_structure: &BlockStructure,
    group: &FusionTreeBlockGroup,
    source_axes: &Arc<[usize]>,
    rows_for: &mut F,
) -> Result<Vec<TreeTransformGroupBlockSpec<T>>, OperationError>
where
    T: Clone + Add<Output = T> + Zero,
    F: FnMut(usize, &FusionTreeKey) -> Result<Arc<Vec<(FusionTreeKey, T)>>, OperationError>,
{
    let src_block_indices = group.block_indices();
    let mut src_keys = Vec::<FusionTreePairKey>::with_capacity(src_block_indices.len());
    let mut dst_keys = Vec::<FusionTreePairKey>::new();
    let mut dst_index_by_key = FxHashMap::<FusionTreePairKey, usize>::default();
    let mut rows = DenseRecouplingRows::new(src_block_indices.len());
    let mut direct_rows = Vec::with_capacity(src_block_indices.len());
    let mut direct_dst_keys = FxHashSet::default();
    let mut is_injective_singleton = true;

    for (src_column, &src_block_index) in src_block_indices.iter().enumerate() {
        let block = src_structure.block(src_block_index)?;
        let BlockKey::FusionTree(src_key) = block.key() else {
            return Err(OperationError::ExpectedFusionTreeBlock {
                tensor: "src",
                index: src_block_index,
            });
        };
        src_keys.push(src_key.clone());

        let transformed = rows_for(src_block_index, src_key.codomain_tree())?;
        if let [(dst_codomain_tree, coefficient)] = transformed.as_slice() {
            let dst_key =
                FusionTreePairKey::pair(dst_codomain_tree.clone(), src_key.domain_tree().clone());
            if !direct_dst_keys.insert(dst_key.clone()) {
                is_injective_singleton = false;
            }
            direct_rows.push((src_key.clone(), dst_key, coefficient.clone()));
        } else {
            is_injective_singleton = false;
        }
        for (dst_codomain_tree, coefficient) in transformed.iter() {
            let dst_key =
                FusionTreePairKey::pair(dst_codomain_tree.clone(), src_key.domain_tree().clone());
            let dst_row = if let Some(&dst_row) = dst_index_by_key.get(&dst_key) {
                dst_row
            } else {
                let dst_row = dst_keys.len();
                dst_index_by_key.insert(dst_key.clone(), dst_row);
                dst_keys.push(dst_key);
                let allocated_row = rows.push_zero_row();
                debug_assert_eq!(allocated_row, dst_row);
                allocated_row
            };
            rows.add(dst_row, src_column, coefficient.clone());
        }
    }

    if let Some(direct_specs) = lower_injective_singleton_rows(
        source_axes,
        direct_rows,
        src_keys.len(),
        is_injective_singleton,
    ) {
        return Ok(direct_specs);
    }
    if dst_keys.is_empty() {
        return Err(OperationError::EmptyTransformBlock);
    }

    Ok(vec![TreeTransformGroupBlockSpec::try_multi(
        dst_keys,
        src_keys,
        rows.into_coefficients(),
    )?
    .with_shared_source_axes(Arc::clone(source_axes))])
}

pub(crate) fn build_multiplicity_free_tree_pair_transform_group_plan<R>(
    rule: &R,
    operation: TreeTransformOperation,
    src_structure: &BlockStructure,
) -> Result<TreeTransformGroupPlan<R::Scalar>, OperationError>
where
    R: MultiplicityFreeRigidSymbols,
    R::Scalar: Clone + Add<Output = R::Scalar> + Mul<Output = R::Scalar> + Zero,
{
    let source_proof =
        validate_multiplicity_free_tree_pair_preflight(rule, &operation, src_structure)?;
    build_multiplicity_free_tree_pair_transform_group_plan_validated(&source_proof, operation)
}

fn build_multiplicity_free_tree_pair_transform_group_plan_validated<R>(
    source_proof: &LocallyValidatedFusionTreeBlockStructure<'_, '_, R>,
    operation: TreeTransformOperation,
) -> Result<TreeTransformGroupPlan<R::Scalar>, OperationError>
where
    R: MultiplicityFreeRigidSymbols,
    R::Scalar: Clone + Add<Output = R::Scalar> + Mul<Output = R::Scalar> + Zero,
{
    let src_structure = source_proof.structure();
    let source_axes = operation_source_axes(&operation);
    let mut primary_prepared = None;
    let mut additional_prepared = None::<FxHashMap<(usize, usize), PreparedTreePairOperation>>;
    let mut specs = Vec::new();
    for group in src_structure.fusion_tree_group_slice() {
        let source_split = (
            group.group_key().codomain_uncoupled().len(),
            group.group_key().domain_uncoupled().len(),
        );
        let prepared = prepared_tree_pair_operation_for_split(
            &mut primary_prepared,
            &mut additional_prepared,
            source_proof.rule(),
            &operation,
            source_split,
        )?;
        specs.extend(build_one_multiplicity_free_tree_pair_group(
            source_proof,
            &operation,
            &source_axes,
            group,
            prepared,
        )?);
    }
    Ok(TreeTransformGroupPlan::new(specs))
}

fn build_one_multiplicity_free_tree_pair_group<R>(
    source_proof: &LocallyValidatedFusionTreeBlockStructure<'_, '_, R>,
    operation: &TreeTransformOperation,
    source_axes: &Arc<[usize]>,
    group: &FusionTreeBlockGroup,
    prepared: &PreparedTreePairOperation,
) -> Result<Vec<TreeTransformGroupBlockSpec<R::Scalar>>, OperationError>
where
    R: MultiplicityFreeRigidSymbols,
    R::Scalar: Clone + Add<Output = R::Scalar> + Mul<Output = R::Scalar> + Zero,
{
    let source_split = (
        group.group_key().codomain_uncoupled().len(),
        group.group_key().domain_uncoupled().len(),
    );
    if operation.is_identity_for(source_split.0, source_split.1) {
        let mut rows_for = |index: usize, _: &FusionTreePairKey| {
            source_proof
                .execute_multiplicity_free_for_block_index(index, prepared)
                .map_err(OperationError::from_core_preserving_context)
                .map(Arc::new)
        };
        return assemble_identity_tree_pair_group_specs(
            source_proof.structure(),
            group,
            source_axes,
            &mut rows_for,
        );
    }

    let first_index = *group
        .block_indices()
        .first()
        .ok_or(OperationError::EmptyTransformBlock)?;
    source_proof.fusion_tree_pair_key(first_index)?.ok_or(
        OperationError::ExpectedFusionTreeBlock {
            tensor: "src",
            index: first_index,
        },
    )?;
    let indices = group.block_indices().iter().copied();
    let ordered = match operation.kind() {
        TreeTransformOperationKind::Transpose => source_proof
            .execute_multiplicity_free_transpose_ordered_for_block_indices_borrowed(
                indices, prepared,
            ),
        TreeTransformOperationKind::Permute | TreeTransformOperationKind::Braid => source_proof
            .execute_multiplicity_free_braid_ordered_for_block_indices_borrowed(indices, prepared),
    }
    .map_err(OperationError::from_core_preserving_context)?;
    assemble_ordered_tree_pair_group_specs(source_proof.structure(), group, source_axes, ordered)
}

fn build_multiplicity_free_tree_pair_transform_group_plan_validated_with_threads<R>(
    source_proof: &LocallyValidatedFusionTreeBlockStructure<'_, '_, R>,
    operation: TreeTransformOperation,
    threads: usize,
) -> Result<TreeTransformGroupPlan<R::Scalar>, OperationError>
where
    R: MultiplicityFreeRigidSymbols,
    R::Scalar: Clone + Add<Output = R::Scalar> + Mul<Output = R::Scalar> + Zero + Send + Sync,
{
    let src_structure = source_proof.structure();
    let source_axes = operation_source_axes(&operation);
    let mut primary_prepared = None;
    let mut additional_prepared = None::<FxHashMap<(usize, usize), PreparedTreePairOperation>>;
    let mut staged_groups = Vec::with_capacity(src_structure.fusion_tree_group_slice().len());
    for group in src_structure.fusion_tree_group_slice() {
        let source_split = (
            group.group_key().codomain_uncoupled().len(),
            group.group_key().domain_uncoupled().len(),
        );
        prepared_tree_pair_operation_for_split(
            &mut primary_prepared,
            &mut additional_prepared,
            source_proof.rule(),
            &operation,
            source_split,
        )?;
        // Why not stage the prepared value itself: ranks beyond its inline
        // step capacity would deep-clone spilled storage once per group.
        staged_groups.push((group, source_split));
    }

    let completed = execute_staged_groups(staged_groups, threads, |(group, source_split)| {
        let prepared = primary_prepared
            .as_ref()
            .filter(|(split, _)| *split == source_split)
            .map(|(_, prepared)| prepared)
            .or_else(|| {
                additional_prepared
                    .as_ref()
                    .and_then(|prepared| prepared.get(&source_split))
            })
            .expect("every staged source split was prepared");
        build_one_multiplicity_free_tree_pair_group(
            source_proof,
            &operation,
            &source_axes,
            group,
            prepared,
        )
    })?;

    let mut specs = Vec::new();
    for group_specs in completed {
        specs.extend(group_specs);
    }
    Ok(TreeTransformGroupPlan::new(specs))
}

pub(crate) fn build_oriented_tree_pair_transform_group_plan_with_threads<R>(
    rule: &R,
    operation: TreeTransformOperation,
    logical_keys: &[FusionTreePairKey],
    storage_structure: &BlockStructure,
    orientation: FusionTreePairOrientation,
    logical_rank: usize,
    storage_projection: &FxHashMap<&FusionTreePairKey, usize>,
    threads: usize,
) -> Result<TreeTransformGroupPlan<R::Scalar>, OperationError>
where
    R: MultiplicityFreeRigidSymbols,
    R::Scalar: Clone + Add<Output = R::Scalar> + Mul<Output = R::Scalar> + Zero + Send + Sync,
{
    validate_multiplicity_free_tree_transform_capability(rule, &operation)?;
    validate_tree_transform_rank_syntax(&operation, logical_rank)?;
    let source_axes = operation_source_axes(&operation);
    let mut group_indices = FxHashMap::default();
    let mut staged_groups = Vec::<(Vec<FusionTreePairKey>, Vec<usize>, (usize, usize))>::new();
    for key in logical_keys {
        let group_key = key.group_key();
        let group_index = match group_indices.entry(group_key) {
            Entry::Occupied(entry) => *entry.get(),
            Entry::Vacant(entry) => {
                let group_index = staged_groups.len();
                entry.insert(group_index);
                staged_groups.push((
                    Vec::new(),
                    Vec::new(),
                    (
                        key.codomain_tree().uncoupled().len(),
                        key.domain_tree().uncoupled().len(),
                    ),
                ));
                group_index
            }
        };
        staged_groups[group_index].0.push(key.clone());
        staged_groups[group_index]
            .1
            .push(
                *storage_projection
                    .get(key)
                    .ok_or(OperationError::StructureMismatch {
                        tensor: "oriented source projection",
                    })?,
            );
    }

    let mut primary_prepared = None;
    let mut additional_prepared = None::<FxHashMap<(usize, usize), PreparedTreePairOperation>>;
    for (_, _, source_split) in &staged_groups {
        prepared_tree_pair_operation_for_split(
            &mut primary_prepared,
            &mut additional_prepared,
            rule,
            &operation,
            *source_split,
        )?;
    }

    let completed = execute_staged_groups(staged_groups, threads, |group| {
        let (src_keys, storage_indices, source_split) = group;
        let prepared = primary_prepared
            .as_ref()
            .filter(|(split, _)| *split == source_split)
            .map(|(_, prepared)| prepared)
            .or_else(|| {
                additional_prepared
                    .as_ref()
                    .and_then(|prepared| prepared.get(&source_split))
            })
            .expect("every oriented source split was prepared");
        let ordered = match operation.kind() {
            TreeTransformOperationKind::Transpose => {
                multiplicity_free_transpose_tree_pair_block_ordered_indexed(
                    rule,
                    storage_structure,
                    &storage_indices,
                    orientation,
                    prepared,
                )
            }
            TreeTransformOperationKind::Permute | TreeTransformOperationKind::Braid => {
                multiplicity_free_braid_tree_pair_block_ordered_indexed(
                    rule,
                    storage_structure,
                    &storage_indices,
                    orientation,
                    prepared,
                )
            }
        }
        .map_err(OperationError::from_core_preserving_context)?;
        assemble_ordered_tree_pair_group_specs_from_keys(src_keys, &source_axes, ordered)
    })?;

    Ok(TreeTransformGroupPlan::from_specs(
        completed.into_iter().flatten(),
    ))
}

#[cfg(test)]
std::thread_local! {
    static ORDERED_TREE_PAIR_LOWERING_CALLS: std::cell::Cell<usize> =
        const { std::cell::Cell::new(0) };
    static LEGACY_TREE_PAIR_ASSEMBLY_CALLS: std::cell::Cell<usize> =
        const { std::cell::Cell::new(0) };
    static TREE_PAIR_OPERATION_PREPARATIONS: std::cell::Cell<usize> =
        const { std::cell::Cell::new(0) };
}

#[cfg(test)]
pub(crate) fn reset_tree_pair_lowering_calls() {
    ORDERED_TREE_PAIR_LOWERING_CALLS.set(0);
    LEGACY_TREE_PAIR_ASSEMBLY_CALLS.set(0);
}

#[cfg(test)]
pub(crate) fn tree_pair_lowering_calls() -> (usize, usize) {
    (
        ORDERED_TREE_PAIR_LOWERING_CALLS.get(),
        LEGACY_TREE_PAIR_ASSEMBLY_CALLS.get(),
    )
}

#[cfg(test)]
pub(crate) fn reset_tree_pair_operation_preparations() {
    TREE_PAIR_OPERATION_PREPARATIONS.set(0);
}

#[cfg(test)]
pub(crate) fn tree_pair_operation_preparations() -> usize {
    TREE_PAIR_OPERATION_PREPARATIONS.get()
}

fn assemble_ordered_tree_pair_group_specs<T>(
    src_structure: &BlockStructure,
    group: &FusionTreeBlockGroup,
    source_axes: &Arc<[usize]>,
    ordered: OrderedBlockLinearMap<FusionTreePairKey, T>,
) -> Result<Vec<TreeTransformGroupBlockSpec<T>>, OperationError>
where
    T: Clone + Zero,
{
    #[cfg(test)]
    ORDERED_TREE_PAIR_LOWERING_CALLS.set(ORDERED_TREE_PAIR_LOWERING_CALLS.get() + 1);

    let mut src_keys = Vec::with_capacity(group.block_indices().len());
    for &src_block_index in group.block_indices() {
        let block = src_structure.block(src_block_index)?;
        let BlockKey::FusionTree(src_key) = block.key() else {
            return Err(OperationError::ExpectedFusionTreeBlock {
                tensor: "src",
                index: src_block_index,
            });
        };
        src_keys.push(src_key.clone());
    }

    assemble_ordered_tree_pair_group_specs_from_keys(src_keys, source_axes, ordered)
}

fn assemble_ordered_tree_pair_group_specs_from_keys<T>(
    src_keys: Vec<FusionTreePairKey>,
    source_axes: &Arc<[usize]>,
    ordered: OrderedBlockLinearMap<FusionTreePairKey, T>,
) -> Result<Vec<TreeTransformGroupBlockSpec<T>>, OperationError>
where
    T: Clone + Zero,
{
    let (destinations, source_count, storage) = ordered.into_parts();
    if source_count != src_keys.len() {
        return Err(OperationError::StructureMismatch {
            tensor: "ordered tree-pair source columns",
        });
    }
    if destinations.is_empty() {
        return Err(OperationError::EmptyTransformBlock);
    }

    match storage {
        OrderedBlockLinearStorage::SingletonColumns {
            destination_rows,
            coefficients,
        } => {
            if destination_rows.len() != source_count || coefficients.len() != source_count {
                return Err(OperationError::StructureMismatch {
                    tensor: "ordered singleton tree-pair columns",
                });
            }
            let mut destination_seen = vec![false; destinations.len()];
            let mut is_injective = true;
            for &destination_row in &destination_rows {
                let Some(seen) = destination_seen.get_mut(destination_row) else {
                    return Err(OperationError::StructureMismatch {
                        tensor: "ordered singleton destination row",
                    });
                };
                if std::mem::replace(seen, true) {
                    is_injective = false;
                }
            }
            if is_injective {
                let mut destination_slots = destinations.into_iter().map(Some).collect::<Vec<_>>();
                return Ok(src_keys
                    .into_iter()
                    .zip(destination_rows)
                    .zip(coefficients)
                    .map(|((source, destination_row), coefficient)| {
                        TreeTransformGroupBlockSpec::single(
                            destination_slots[destination_row]
                                .take()
                                .expect("injective destination row is consumed once"),
                            source,
                            coefficient,
                        )
                        .with_shared_source_axes(Arc::clone(source_axes))
                    })
                    .collect());
            }

            let mut dense = vec![T::zero(); destinations.len().saturating_mul(source_count)];
            for (source, (destination_row, coefficient)) in
                destination_rows.into_iter().zip(coefficients).enumerate()
            {
                dense[destination_row * source_count + source] = coefficient;
            }
            Ok(vec![TreeTransformGroupBlockSpec::try_multi(
                destinations,
                src_keys,
                dense,
            )?
            .with_shared_source_axes(Arc::clone(source_axes))])
        }
        OrderedBlockLinearStorage::DenseDstSrc(coefficients) => {
            let expected = destinations.len().saturating_mul(source_count);
            if coefficients.len() != expected {
                return Err(OperationError::StructureMismatch {
                    tensor: "ordered dense tree-pair coefficients",
                });
            }
            let coefficients = coefficients
                .into_iter()
                .map(|coefficient| coefficient.unwrap_or_else(T::zero))
                .collect();
            Ok(vec![TreeTransformGroupBlockSpec::try_multi(
                destinations,
                src_keys,
                coefficients,
            )?
            .with_shared_source_axes(Arc::clone(source_axes))])
        }
    }
}

/// Generic-fusion (outer-multiplicity) tree-pair plan compile — the Stage B2c
/// dispatch receptacle for SU(3)/SO(N≥7)/Sp(N) rules. Parallel entry to
/// `build_multiplicity_free_tree_pair_transform_group_plan`: non-identity groups
/// are compiled as one ordered block map over full tree-pair keys, while the
/// identity path keeps the existing identity assembly.
///
/// This is a SEPARATE entry rather than a runtime branch inside the mult-free
/// builder because the two are disjoint at the type level:
/// `GenericRigidSymbols` and `MultiplicityFreeRigidSymbols` are never both
/// implemented by a real rule, so a mult-free rule can never name this
/// function's bound, let alone reach its row-generation body. Both paths do
/// intentionally share group-spec assembly, including structural monomial
/// lowering, while retaining separate fusion-style guards and symbol APIs.
/// Why not call this a byte-for-byte or blanket zero-cost guarantee: changes to
/// the shared assembler are expected to affect both paths; the guarantee is
/// that multiplicity-free rules never execute generic F/R-symbol logic. The
/// provider-owned [`FusionStyleKind`] gate below defends against a
/// `GenericRigidSymbols` implementation that reports a multiplicity-free
/// style. Fusion style is not duplicated in individual tree keys.
///
/// # Provider-domain precondition
///
/// Fusion-tree block keys in `src_structure` follow
/// [`tenet_core::FusionTreeKey::validate_for_rule`]'s provider-domain
/// precondition.
pub fn build_generic_tree_pair_transform_group_plan<R>(
    rule: &R,
    operation: TreeTransformOperation,
    src_structure: &BlockStructure,
) -> Result<TreeTransformGroupPlan<R::Scalar>, OperationError>
where
    R: GenericRigidSymbols,
    R::Scalar: GenericBraidScalar + Zero,
{
    let source_proof = validate_generic_tree_pair_preflight(rule, &operation, src_structure)?;
    build_generic_tree_pair_transform_group_plan_validated(&source_proof, operation)
}

pub(crate) fn build_generic_tree_pair_transform_group_plan_validated<R>(
    source_proof: &LocallyValidatedFusionTreeBlockStructure<'_, '_, R>,
    operation: TreeTransformOperation,
) -> Result<TreeTransformGroupPlan<R::Scalar>, OperationError>
where
    R: GenericRigidSymbols,
    R::Scalar: GenericBraidScalar + Zero,
{
    let src_structure = source_proof.structure();
    let source_axes = operation_source_axes(&operation);

    let mut specs = Vec::new();
    for group in src_structure.fusion_tree_group_slice() {
        if operation.is_identity_for(
            group.group_key().codomain_uncoupled().len(),
            group.group_key().domain_uncoupled().len(),
        ) {
            let mut rows_for = |index: usize, _: &FusionTreePairKey| {
                let rows = match operation.kind() {
                    TreeTransformOperationKind::Permute => source_proof
                        .generic_permute_tree_pair_for_block_index(
                            index,
                            operation.codomain_permutation(),
                            operation.domain_permutation(),
                        ),
                    TreeTransformOperationKind::Braid => source_proof
                        .generic_braid_tree_pair_for_block_index(
                            index,
                            operation.codomain_permutation(),
                            operation.domain_permutation(),
                            operation.codomain_levels(),
                            operation.domain_levels(),
                        ),
                    TreeTransformOperationKind::Transpose => source_proof
                        .generic_transpose_tree_pair_for_block_index(
                            index,
                            operation.codomain_permutation(),
                            operation.domain_permutation(),
                        ),
                }
                .map_err(OperationError::from_core_preserving_context)?;
                Ok(Arc::new(rows))
            };
            specs.extend(assemble_identity_tree_pair_group_specs(
                src_structure,
                group,
                &source_axes,
                &mut rows_for,
            )?);
        } else {
            let mut src_keys = Vec::with_capacity(group.block_indices().len());
            for &src_block_index in group.block_indices() {
                let block = src_structure.block(src_block_index)?;
                let BlockKey::FusionTree(src_key) = block.key() else {
                    return Err(OperationError::ExpectedFusionTreeBlock {
                        tensor: "src",
                        index: src_block_index,
                    });
                };
                src_keys.push(src_key.clone());
            }
            let ordered = match operation.kind() {
                TreeTransformOperationKind::Permute => generic_permute_tree_pair_block_ordered(
                    source_proof.rule(),
                    &src_keys,
                    operation.codomain_permutation(),
                    operation.domain_permutation(),
                ),
                TreeTransformOperationKind::Braid => generic_braid_tree_pair_block_ordered(
                    source_proof.rule(),
                    &src_keys,
                    operation.codomain_permutation(),
                    operation.domain_permutation(),
                    operation.codomain_levels(),
                    operation.domain_levels(),
                ),
                TreeTransformOperationKind::Transpose => generic_transpose_tree_pair_block_ordered(
                    source_proof.rule(),
                    &src_keys,
                    operation.codomain_permutation(),
                    operation.domain_permutation(),
                ),
            }
            .map_err(OperationError::from_core_preserving_context)?;
            specs.extend(assemble_ordered_tree_pair_group_specs(
                src_structure,
                group,
                &source_axes,
                ordered,
            )?);
        }
    }

    Ok(TreeTransformGroupPlan::new(specs))
}

fn assemble_identity_tree_pair_group_specs<T, F>(
    src_structure: &BlockStructure,
    group: &FusionTreeBlockGroup,
    source_axes: &Arc<[usize]>,
    rows_for: &mut F,
) -> Result<Vec<TreeTransformGroupBlockSpec<T>>, OperationError>
where
    T: Clone,
    F: FnMut(usize, &FusionTreePairKey) -> Result<Arc<Vec<(FusionTreePairKey, T)>>, OperationError>,
{
    let mut specs = Vec::with_capacity(group.block_indices().len());
    for &src_block_index in group.block_indices() {
        let block = src_structure.block(src_block_index)?;
        let BlockKey::FusionTree(src_key) = block.key() else {
            return Err(OperationError::ExpectedFusionTreeBlock {
                tensor: "src",
                index: src_block_index,
            });
        };
        let transformed = rows_for(src_block_index, src_key)?;
        let [(dst_key, coefficient)] = transformed.as_slice() else {
            return Err(OperationError::EmptyTransformBlock);
        };
        // Identity rows are singleton by construction. Why not synthesize the
        // coefficient here: consuming the validated row preserves scalar
        // conversion semantics across serial and parallel builders.
        specs.push(
            TreeTransformGroupBlockSpec::single(
                dst_key.clone(),
                src_key.clone(),
                coefficient.clone(),
            )
            .with_shared_source_axes(Arc::clone(source_axes)),
        );
    }
    Ok(specs)
}

/// Assemble one group's block specs (destination-key dedup plus the
/// `U[dst, src]` recoupling coefficient matrix) from per-tree recoupling
/// rows. Groups are independent, which is what lets the parallel compile map
/// over them.
fn assemble_tree_pair_group_specs<T, F>(
    src_structure: &BlockStructure,
    group: &FusionTreeBlockGroup,
    source_axes: &Arc<[usize]>,
    rows_for: &mut F,
) -> Result<Vec<TreeTransformGroupBlockSpec<T>>, OperationError>
where
    T: Clone + Add<Output = T> + Zero,
    F: FnMut(usize, &FusionTreePairKey) -> Result<Arc<Vec<(FusionTreePairKey, T)>>, OperationError>,
{
    #[cfg(test)]
    LEGACY_TREE_PAIR_ASSEMBLY_CALLS.set(LEGACY_TREE_PAIR_ASSEMBLY_CALLS.get() + 1);

    let src_block_indices = group.block_indices();
    let mut src_keys = Vec::<FusionTreePairKey>::with_capacity(src_block_indices.len());
    let mut dst_keys = Vec::<FusionTreePairKey>::new();
    let mut dst_index_by_key = FxHashMap::<FusionTreePairKey, usize>::default();
    let mut rows = DenseRecouplingRows::new(src_block_indices.len());
    let mut direct_rows = Vec::with_capacity(src_block_indices.len());
    let mut direct_dst_keys = FxHashSet::default();
    let mut is_injective_singleton = true;

    for (src_column, &src_block_index) in src_block_indices.iter().enumerate() {
        let block = src_structure.block(src_block_index)?;
        let BlockKey::FusionTree(src_key) = block.key() else {
            return Err(OperationError::ExpectedFusionTreeBlock {
                tensor: "src",
                index: src_block_index,
            });
        };
        src_keys.push(src_key.clone());

        let transformed = rows_for(src_block_index, src_key)?;
        if let [(dst_tree_key, coefficient)] = transformed.as_slice() {
            let dst_key = dst_tree_key.clone();
            if !direct_dst_keys.insert(dst_key.clone()) {
                is_injective_singleton = false;
            }
            direct_rows.push((src_key.clone(), dst_key, coefficient.clone()));
        } else {
            is_injective_singleton = false;
        }

        for (dst_tree_key, coefficient) in transformed.iter() {
            let dst_key = dst_tree_key.clone();
            let dst_row = if let Some(&dst_row) = dst_index_by_key.get(&dst_key) {
                dst_row
            } else {
                let dst_row = dst_keys.len();
                dst_index_by_key.insert(dst_key.clone(), dst_row);
                dst_keys.push(dst_key);
                let allocated_row = rows.push_zero_row();
                debug_assert_eq!(allocated_row, dst_row);
                allocated_row
            };
            rows.add(dst_row, src_column, coefficient.clone());
        }
    }

    if let Some(direct_specs) = lower_injective_singleton_rows(
        source_axes,
        direct_rows,
        src_keys.len(),
        is_injective_singleton,
    ) {
        return Ok(direct_specs);
    }
    if dst_keys.is_empty() {
        return Err(OperationError::EmptyTransformBlock);
    }

    Ok(vec![TreeTransformGroupBlockSpec::try_multi(
        dst_keys,
        src_keys,
        rows.into_coefficients(),
    )?
    .with_shared_source_axes(Arc::clone(source_axes))])
}

fn lower_injective_singleton_rows<T>(
    source_axes: &Arc<[usize]>,
    direct_rows: Vec<(FusionTreePairKey, FusionTreePairKey, T)>,
    src_count: usize,
    is_injective_singleton: bool,
) -> Option<Vec<TreeTransformGroupBlockSpec<T>>> {
    if !is_injective_singleton || direct_rows.len() != src_count || direct_rows.is_empty() {
        return None;
    }

    // Row cardinality plus destination-key injectivity proves independent
    // replay. Why not inspect coefficient values: phases and non-unit scalars
    // are valid direct maps, while numerical zeros cannot prove structure.
    Some(
        direct_rows
            .into_iter()
            .map(|(src_key, dst_key, coefficient)| {
                TreeTransformGroupBlockSpec::single(dst_key, src_key, coefficient)
                    .with_shared_source_axes(Arc::clone(source_axes))
            })
            .collect(),
    )
}

pub(crate) fn build_unique_tree_pair_transform_group_plan<R>(
    rule: &R,
    operation: TreeTransformOperation,
    src_structure: &BlockStructure,
) -> Result<TreeTransformGroupPlan<R::Scalar>, OperationError>
where
    R: MultiplicityFreeRigidSymbols,
    R::Scalar: Clone + Add<Output = R::Scalar> + Mul<Output = R::Scalar> + Zero,
{
    if rule.fusion_style() != FusionStyleKind::Unique {
        return Err(OperationError::UnsupportedFusionStyle {
            operation: Box::new(operation),
            style: rule.fusion_style(),
        });
    }
    let source_proof =
        validate_multiplicity_free_tree_pair_preflight(rule, &operation, src_structure)?;
    build_unique_tree_pair_transform_group_plan_validated(&source_proof, operation)
}

fn build_unique_tree_pair_transform_group_plan_validated<R>(
    source_proof: &LocallyValidatedFusionTreeBlockStructure<'_, '_, R>,
    operation: TreeTransformOperation,
) -> Result<TreeTransformGroupPlan<R::Scalar>, OperationError>
where
    R: MultiplicityFreeRigidSymbols,
    R::Scalar: Clone + Add<Output = R::Scalar> + Mul<Output = R::Scalar>,
{
    debug_assert_eq!(source_proof.rule().fusion_style(), FusionStyleKind::Unique);
    let src_structure = source_proof.structure();
    let source_axes = operation_source_axes(&operation);
    let mut primary_prepared = None;
    let mut additional_prepared = None::<FxHashMap<(usize, usize), PreparedTreePairOperation>>;
    let mut specs = Vec::with_capacity(src_structure.block_count());
    for index in 0..src_structure.block_count() {
        let Some(src_key) = source_proof.fusion_tree_pair_key(index)? else {
            continue;
        };
        let source_split = (
            src_key.codomain_tree().uncoupled().len(),
            src_key.domain_tree().uncoupled().len(),
        );
        let prepared = prepared_tree_pair_operation_for_split(
            &mut primary_prepared,
            &mut additional_prepared,
            source_proof.rule(),
            &operation,
            source_split,
        )?;
        let transformed = source_proof
            .execute_unique_rigid_for_block_index(index, prepared)
            .map_err(OperationError::from_core_preserving_context)?;
        specs.push(
            TreeTransformGroupBlockSpec::single(transformed.0, src_key.clone(), transformed.1)
                .with_shared_source_axes(Arc::clone(&source_axes)),
        );
    }
    Ok(TreeTransformGroupPlan::new(specs))
}

fn prepared_tree_pair_operation_for_split<'prepared, R>(
    primary: &'prepared mut Option<((usize, usize), PreparedTreePairOperation)>,
    additional: &'prepared mut Option<FxHashMap<(usize, usize), PreparedTreePairOperation>>,
    rule: &R,
    operation: &TreeTransformOperation,
    source_split: (usize, usize),
) -> Result<&'prepared PreparedTreePairOperation, OperationError>
where
    R: FusionRule,
{
    if primary.is_none() {
        *primary = Some((
            source_split,
            prepare_tree_pair_operation(rule, operation, source_split)?,
        ));
    }
    if let Some((primary_split, prepared)) = primary.as_ref() {
        if *primary_split == source_split {
            return Ok(prepared);
        }
    }
    let prepared_by_split = additional.get_or_insert_with(FxHashMap::default);
    let prepared = match prepared_by_split.entry(source_split) {
        Entry::Occupied(entry) => entry.into_mut(),
        Entry::Vacant(entry) => {
            entry.insert(prepare_tree_pair_operation(rule, operation, source_split)?)
        }
    };
    Ok(prepared)
}

fn prepare_tree_pair_operation<R>(
    rule: &R,
    operation: &TreeTransformOperation,
    (source_codomain_rank, source_domain_rank): (usize, usize),
) -> Result<PreparedTreePairOperation, OperationError>
where
    R: FusionRule,
{
    #[cfg(test)]
    TREE_PAIR_OPERATION_PREPARATIONS.set(TREE_PAIR_OPERATION_PREPARATIONS.get() + 1);
    match operation.kind() {
        TreeTransformOperationKind::Permute => PreparedTreePairOperation::prepare_permute(
            rule,
            source_codomain_rank,
            source_domain_rank,
            operation.codomain_permutation(),
            operation.domain_permutation(),
        ),
        TreeTransformOperationKind::Braid => PreparedTreePairOperation::prepare_braid(
            rule,
            source_codomain_rank,
            source_domain_rank,
            operation.codomain_permutation(),
            operation.domain_permutation(),
            operation.codomain_levels(),
            operation.domain_levels(),
        ),
        TreeTransformOperationKind::Transpose => PreparedTreePairOperation::prepare_transpose(
            source_codomain_rank,
            source_domain_rank,
            operation.codomain_permutation(),
            operation.domain_permutation(),
        ),
    }
    .map_err(OperationError::from_core_preserving_context)
}

fn prepare_tree_pair_operation_syntax(
    operation: &TreeTransformOperation,
    (source_codomain_rank, source_domain_rank): (usize, usize),
) -> Result<(), OperationError> {
    match operation.kind() {
        TreeTransformOperationKind::Permute => PreparedTreePairOperation::validate_permute_syntax(
            source_codomain_rank,
            source_domain_rank,
            operation.codomain_permutation(),
            operation.domain_permutation(),
        ),
        TreeTransformOperationKind::Braid => PreparedTreePairOperation::validate_braid_syntax(
            source_codomain_rank,
            source_domain_rank,
            operation.codomain_permutation(),
            operation.domain_permutation(),
            operation.codomain_levels(),
            operation.domain_levels(),
        ),
        TreeTransformOperationKind::Transpose => {
            PreparedTreePairOperation::validate_transpose_syntax(
                source_codomain_rank,
                source_domain_rank,
                operation.codomain_permutation(),
                operation.domain_permutation(),
            )
        }
    }
    .map_err(OperationError::from_core_preserving_context)
}

fn validate_all_codomain_operation_scope(
    operation: &TreeTransformOperation,
) -> Result<(), OperationError> {
    let scope_error = || OperationError::UnsupportedTreeTransformScope {
        operation: Box::new(operation.clone()),
        message: "all-codomain UniqueFusion lowering requires an empty domain operation",
    };

    match operation.kind() {
        TreeTransformOperationKind::Permute if operation.domain_permutation().is_empty() => Ok(()),
        TreeTransformOperationKind::Braid
            if operation.domain_permutation().is_empty() && operation.domain_levels().is_empty() =>
        {
            Ok(())
        }
        TreeTransformOperationKind::Permute | TreeTransformOperationKind::Braid => {
            Err(scope_error())
        }
        TreeTransformOperationKind::Transpose => Err(OperationError::UnsupportedTreeTransformScope {
            operation: Box::new(operation.clone()),
            message: "all-codomain UniqueFusion lowering supports explicit Permute or Braid operations",
        }),
    }
}

fn operation_source_axes(operation: &TreeTransformOperation) -> Arc<[usize]> {
    operation
        .codomain_permutation()
        .iter()
        .chain(operation.domain_permutation())
        .copied()
        .collect()
}

fn validate_all_codomain_fusion_tree_block<R>(
    rule: &R,
    index: usize,
    key: &FusionTreePairKey,
) -> Result<(), OperationError>
where
    R: FusionRule,
{
    let domain = key.domain_tree();
    let empty_domain_coupled_is_valid = domain.coupled() == rule.vacuum();
    if domain.uncoupled().is_empty()
        && empty_domain_coupled_is_valid
        && domain.is_dual().is_empty()
        && domain.innerlines().is_empty()
        && domain.vertices().is_empty()
    {
        return Ok(());
    }
    Err(OperationError::ExpectedAllCodomainFusionTree { index })
}
