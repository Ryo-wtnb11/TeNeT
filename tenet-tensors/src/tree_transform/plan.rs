use core::ops::{Add, Mul};
use rustc_hash::{FxHashMap, FxHashSet};
use smallvec::SmallVec;
use std::{collections::hash_map::Entry, sync::Arc};

use num_traits::Zero;
use tenet_core::{
    multiplicity_free_braid_tree_pair_block, multiplicity_free_permute_tree_pair_block,
    multiplicity_free_transpose_tree_pair_block, BlockKey, BlockStructure, CoreError, FusionRule,
    FusionStyleKind, FusionTreeBlockGroup, FusionTreeBlockKey, FusionTreeKey, GenericBraidScalar,
    GenericRigidSymbols, LocallyValidatedFusionTreeBlockStructure, MultiplicityFreeFusionSymbols,
    MultiplicityFreeRigidSymbols, PreparedTreePairOperation,
};

use crate::OperationError;

use super::operation::{TreeTransformOperation, ValidateBraidingSupport};

pub use tenet_operations::transform_plan::{
    TreeTransformBlockSpec, TreeTransformGroupBlockSpec, TreeTransformGroupPlan,
    TreeTransformKeyBlockSpec,
};

pub(crate) fn validate_multiplicity_free_tree_transform_capability<R>(
    rule: &R,
    operation: &TreeTransformOperation,
) -> Result<(), OperationError>
where
    R: FusionRule,
{
    if !rule.fusion_style().is_multiplicity_free() {
        return Err(OperationError::UnsupportedFusionStyle {
            operation: Box::new(operation.clone()),
            style: rule.fusion_style(),
        });
    }
    operation.validate_braiding_support(rule)
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
        let mut first_pair_mismatch = None;
        let mut first_source_restriction = None;
        let mut first_syntax_error = None;
        let mut prepared_splits = SmallVec::<[(usize, usize); 4]>::new();
        for index in 0..src_structure.block_count() {
            let block = src_structure.block(index)?;
            let BlockKey::FusionTree(key) = block.key() else {
                continue;
            };
            let normalized_coupled = |tree: &FusionTreeKey| {
                tree.coupled()
                    .or_else(|| tree.uncoupled().is_empty().then(|| rule.vacuum()))
            };
            if normalized_coupled(key.codomain_tree()) != normalized_coupled(key.domain_tree()) {
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
    F: FnMut(&FusionTreeBlockKey) -> Result<Vec<(FusionTreeBlockKey, T)>, OperationError>,
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
    F: FnMut(&FusionTreeBlockKey) -> Result<Vec<(FusionTreeBlockKey, T)>, OperationError>,
{
    let src_structure = source_proof.structure();
    let source_axes = operation_source_axes(&operation);

    let mut specs = Vec::new();
    for group in src_structure.fusion_tree_groups() {
        specs.extend(assemble_tree_pair_group_specs(
            src_structure,
            &group,
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

pub(crate) fn build_all_codomain_tree_transform_group_plan_validated<R>(
    source_proof: &LocallyValidatedAllCodomainFusionTreeBlockStructure<'_, '_, R>,
    operation: TreeTransformOperation,
) -> Result<TreeTransformGroupPlan<R::Scalar>, OperationError>
where
    R: MultiplicityFreeFusionSymbols,
    R::Scalar: Clone + Add<Output = R::Scalar> + Mul<Output = R::Scalar> + Zero,
{
    if source_proof.rule().fusion_style() == FusionStyleKind::Unique {
        build_unique_all_codomain_tree_transform_group_plan_validated(source_proof, operation)
    } else {
        build_multiplicity_free_all_codomain_tree_transform_group_plan_validated(
            source_proof,
            operation,
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

pub(crate) fn build_tree_pair_transform_group_plan_validated<R>(
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
    F: FnMut(&FusionTreeBlockKey) -> Result<(FusionTreeBlockKey, T), OperationError>,
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
        let Some(src_key) = proof.fusion_tree_block_key(index)? else {
            continue;
        };
        let mut rows = match &operation {
            TreeTransformOperation::Permute {
                codomain_permutation,
                ..
            } => proof.permute_codomain_rows_for_block_index(index, codomain_permutation),
            TreeTransformOperation::Braid {
                codomain_permutation,
                codomain_levels,
                ..
            } => proof.braid_codomain_rows_for_block_index(
                index,
                codomain_permutation,
                codomain_levels,
            ),
            TreeTransformOperation::Transpose { .. } => {
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
        let dst_key = FusionTreeBlockKey::pair(destination, src_key.domain_tree().clone());
        specs.push(
            TreeTransformGroupBlockSpec::single(
                src_key.group_key(),
                dst_key,
                src_key.clone(),
                coefficient,
            )
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
    for group in src_structure.fusion_tree_groups() {
        let mut rows_for = |index: usize, source: &FusionTreeKey| {
            transform_all_codomain_rows_for_block_index(source_proof, &operation, index, source)
                .map(Arc::new)
        };
        if operation.is_identity_for(group.group_key().codomain_uncoupled().len(), 0) {
            specs.extend(assemble_identity_all_codomain_group_specs(
                src_structure,
                &group,
                &source_axes,
                &mut rows_for,
            )?);
        } else {
            specs.extend(assemble_all_codomain_group_specs(
                src_structure,
                &group,
                &source_axes,
                &mut rows_for,
            )?);
        }
    }
    Ok(TreeTransformGroupPlan::new(specs))
}

/// Shape-independent all-codomain rows. One plan compile records one hit or
/// miss per distinct memo key, even when both accepted empty-domain encodings
/// (`None` and `Some(vacuum)`) reference the same codomain tree.
type TransformRows<K, T> = Vec<(K, T)>;
type SharedTransformRows<K, T> = Arc<TransformRows<K, T>>;
type TransformBlockRows<K, T> = Vec<TransformRows<K, T>>;
type StagedSources<'a, K, T> = SmallVec<[(usize, &'a K, Option<SharedTransformRows<K, T>>); 4]>;
type StagedSourceIndices = SmallVec<[usize; 4]>;

enum StagedSourceAlignment {
    Identity,
    Explicit(StagedSourceIndices),
}

pub(crate) type AllCodomainRowMemo<T, RuleKey> = FxHashMap<
    (RuleKey, TreeTransformOperation, FusionTreeKey),
    SharedTransformRows<FusionTreeKey, T>,
>;

struct StagedGroupRows<'a, K, T> {
    group: FusionTreeBlockGroup,
    // Why not inline an unbounded group: group cardinality has no algebraic
    // rank bound. Four bounds the fixed footprint; larger groups spill without
    // changing ownership, ordering, or transactional semantics.
    sources: StagedSources<'a, K, T>,
    // Why not rebuild a key map in each worker: group assemblers visit every
    // stored block once in block order, so this is the recoupling matrix's
    // block-column alignment. Callback key and exhaustion checks turn future
    // assembler-order drift into an error without rebuilding a per-group hash
    // table on the fully-warm path.
    source_alignment: StagedSourceAlignment,
}

struct CompletedGroupRows<K, T> {
    specs: Vec<TreeTransformGroupBlockSpec<T>>,
    computed: Vec<(K, SharedTransformRows<K, T>)>,
}

fn resolve_staged_group_rows<K, T, F>(
    mut sources: StagedSources<'_, K, T>,
    block_transform: F,
) -> Result<(StagedSources<'_, K, T>, Vec<(K, SharedTransformRows<K, T>)>), OperationError>
where
    K: Clone,
    F: FnOnce(&mut dyn Iterator<Item = usize>) -> Result<Vec<TransformRows<K, T>>, OperationError>,
{
    if sources.iter().all(|(_, _, rows)| rows.is_some()) {
        return Ok((sources, Vec::new()));
    }

    let missing_count = sources.iter().filter(|(_, _, rows)| rows.is_none()).count();
    let mut missing_indices = sources
        .iter()
        .filter_map(|(index, _, rows)| rows.is_none().then_some(*index));
    let batched = block_transform(&mut missing_indices)?;
    if missing_indices.next().is_some() || batched.len() != missing_count {
        return Err(OperationError::CoefficientCountMismatch {
            expected: missing_count,
            actual: batched.len(),
        });
    }

    let mut computed = Vec::with_capacity(missing_count);
    let mut missing_rows = batched.into_iter();
    // Rescan the short source group in order so publishing each computed row
    // needs no separately owned position list.
    for (_, key, slot) in &mut sources {
        if slot.is_some() {
            continue;
        }
        // Why not return source keys from the compact runner: preflight owns
        // the admitted keys and the runner preserves their exact order.
        // Pairing cloned keys with rows inside core would allocate another
        // outer Vec before this transactional publication step.
        let rows = missing_rows
            .next()
            .expect("validated block result covers every missing source");
        let rows = Arc::new(rows);
        *slot = Some(Arc::clone(&rows));
        computed.push(((*key).clone(), rows));
    }
    debug_assert!(missing_rows.next().is_none());
    Ok((sources, computed))
}

#[cfg(test)]
mod staged_row_resolution_tests {
    use super::{resolve_staged_group_rows, StagedSources};
    use std::sync::Arc;
    use tenet_operations::OperationError;

    #[test]
    fn partial_misses_resolve_in_source_order_without_changing_hits() {
        let keys = [0usize, 1, 2, 3];
        let hit_zero = Arc::new(vec![(10usize, 1i32)]);
        let hit_two = Arc::new(vec![(20usize, 2i32)]);
        let mut sources = StagedSources::new();
        sources.push((0, &keys[0], Some(Arc::clone(&hit_zero))));
        sources.push((1, &keys[1], None));
        sources.push((2, &keys[2], Some(Arc::clone(&hit_two))));
        sources.push((3, &keys[3], None));

        let (resolved, computed) = resolve_staged_group_rows(sources, |missing| {
            // What: partial misses reach one block transform in original
            // source order, independently of the intervening memo hits.
            assert!(missing.eq([1, 3]));
            Ok(vec![vec![(11, 3)], vec![(31, 4)]])
        })
        .unwrap();

        assert!(Arc::ptr_eq(resolved[0].2.as_ref().unwrap(), &hit_zero));
        assert!(Arc::ptr_eq(resolved[2].2.as_ref().unwrap(), &hit_two));
        assert_eq!(resolved[1].2.as_deref().unwrap(), &[(11, 3)]);
        assert_eq!(resolved[3].2.as_deref().unwrap(), &[(31, 4)]);
        assert_eq!(
            computed.iter().map(|(key, _)| *key).collect::<Vec<_>>(),
            [1, 3]
        );
    }

    #[test]
    fn coefficient_count_mismatch_does_not_publish_partial_rows() {
        let keys = [0usize, 1];
        let mut sources = StagedSources::new();
        sources.push((0, &keys[0], None));
        sources.push((1, &keys[1], None));

        let error = resolve_staged_group_rows(sources, |missing| {
            // What: validating the complete block result remains before any
            // staged row publication.
            assert!(missing.eq([0, 1]));
            Ok(vec![vec![(10, 1i32)]])
        })
        .unwrap_err();

        assert_eq!(
            error,
            OperationError::CoefficientCountMismatch {
                expected: 2,
                actual: 1,
            }
        );
    }
}

#[cfg(test)]
mod generic_preflight_tests {
    use super::{validate_generic_tree_pair_preflight, TreeTransformOperation};
    use tenet_core::{
        BlockKey, BlockSpec, BlockStructure, CoreError, FusionTreeBlockKey, FusionTreeKey,
        SectorId, Su3FusionRule,
    };
    use tenet_operations::OperationError;

    #[test]
    fn su3_generic_preflight_accepts_valid_permute_and_braid_before_categorical_admission() {
        let rule = Su3FusionRule::new();
        let eight = rule.sector_of(1, 1).unwrap();
        let vacuum = SectorId::new(0);
        let valid = FusionTreeBlockKey::pair(
            FusionTreeKey::try_new_for_rule(
                &rule,
                [eight, eight],
                Some(vacuum),
                [false, false],
                [],
                [SectorId::new(1)],
            )
            .unwrap(),
            FusionTreeKey::try_new_for_rule(&rule, [], Some(vacuum), [], [], []).unwrap(),
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

        let malformed = FusionTreeBlockKey::pair_from_sector_ids(
            [eight.id(), eight.id()],
            [],
            Some(vacuum.id()),
            [false, false],
            [],
            [],
            [],
            [0],
            [],
        );
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

#[allow(clippy::too_many_arguments)]
#[cfg(test)]
pub(crate) fn build_multiplicity_free_all_codomain_tree_transform_group_plan_memoized<R, RuleKey>(
    rule: &R,
    rule_key: &RuleKey,
    operation: TreeTransformOperation,
    src_structure: &BlockStructure,
    memo: &mut AllCodomainRowMemo<R::Scalar, RuleKey>,
    memo_hits: &mut usize,
    memo_misses: &mut usize,
    threads: usize,
) -> Result<TreeTransformGroupPlan<R::Scalar>, OperationError>
where
    R: MultiplicityFreeFusionSymbols + Sync,
    R::Scalar: Clone + Add<Output = R::Scalar> + Mul<Output = R::Scalar> + Zero,
    RuleKey: Clone + Eq + std::hash::Hash,
{
    let proof = validate_multiplicity_free_all_codomain_preflight(rule, &operation, src_structure)?;
    build_multiplicity_free_all_codomain_tree_transform_group_plan_memoized_validated(
        &proof,
        rule_key,
        operation,
        memo,
        memo_hits,
        memo_misses,
        threads,
    )
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn build_multiplicity_free_all_codomain_tree_transform_group_plan_memoized_validated<
    R,
    RuleKey,
>(
    proof: &LocallyValidatedAllCodomainFusionTreeBlockStructure<'_, '_, R>,
    rule_key: &RuleKey,
    operation: TreeTransformOperation,
    memo: &mut AllCodomainRowMemo<R::Scalar, RuleKey>,
    memo_hits: &mut usize,
    memo_misses: &mut usize,
    threads: usize,
) -> Result<TreeTransformGroupPlan<R::Scalar>, OperationError>
where
    R: MultiplicityFreeFusionSymbols + Sync,
    R::Scalar: Clone + Add<Output = R::Scalar> + Mul<Output = R::Scalar> + Zero,
    RuleKey: Clone + Eq + std::hash::Hash,
{
    build_multiplicity_free_all_codomain_tree_transform_group_plan_memoized_impl(
        proof,
        rule_key,
        operation,
        memo,
        memo_hits,
        memo_misses,
        threads,
        transform_all_codomain_rows_for_block_indices,
    )
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
    let transformed = match operation {
        TreeTransformOperation::Permute {
            codomain_permutation,
            ..
        } => proof
            .proof()
            .permute_codomain_rows_for_block_indices(indices, codomain_permutation),
        TreeTransformOperation::Braid {
            codomain_permutation,
            codomain_levels,
            ..
        } => proof.proof().braid_codomain_rows_for_block_indices(
            indices,
            codomain_permutation,
            codomain_levels,
        ),
        TreeTransformOperation::Transpose { .. } => {
            unreachable!("all-codomain operation scope validation rejected transpose")
        }
    };
    transformed.map_err(OperationError::from_core_preserving_context)
}

fn transform_all_codomain_rows_for_block_index<R>(
    proof: &LocallyValidatedAllCodomainFusionTreeBlockStructure<'_, '_, R>,
    operation: &TreeTransformOperation,
    block_index: usize,
    source: &FusionTreeKey,
) -> Result<TransformRows<FusionTreeKey, R::Scalar>, OperationError>
where
    R: MultiplicityFreeFusionSymbols,
    R::Scalar: Clone + Add<Output = R::Scalar> + Mul<Output = R::Scalar>,
{
    let actual_source = proof
        .proof()
        .fusion_tree_block_key(block_index)?
        .ok_or(OperationError::ExpectedFusionTreeBlock {
            tensor: "src",
            index: block_index,
        })?
        .codomain_tree();
    if actual_source != source {
        return Err(OperationError::StructureMismatch {
            tensor: "proof-bound all-codomain source",
        });
    }
    let rows = match operation {
        TreeTransformOperation::Permute {
            codomain_permutation,
            ..
        } => proof
            .proof()
            .permute_codomain_rows_for_block_index(block_index, codomain_permutation),
        TreeTransformOperation::Braid {
            codomain_permutation,
            codomain_levels,
            ..
        } => proof.proof().braid_codomain_rows_for_block_index(
            block_index,
            codomain_permutation,
            codomain_levels,
        ),
        TreeTransformOperation::Transpose { .. } => {
            unreachable!("all-codomain operation scope validation rejected transpose")
        }
    };
    rows.map_err(OperationError::from_core_preserving_context)
}

#[allow(clippy::too_many_arguments)]
fn build_multiplicity_free_all_codomain_tree_transform_group_plan_memoized_impl<R, RuleKey, F>(
    proof: &LocallyValidatedAllCodomainFusionTreeBlockStructure<'_, '_, R>,
    rule_key: &RuleKey,
    operation: TreeTransformOperation,
    memo: &mut AllCodomainRowMemo<R::Scalar, RuleKey>,
    memo_hits: &mut usize,
    memo_misses: &mut usize,
    threads: usize,
    block_transform: F,
) -> Result<TreeTransformGroupPlan<R::Scalar>, OperationError>
where
    R: MultiplicityFreeFusionSymbols + Sync,
    R::Scalar: Clone + Add<Output = R::Scalar> + Mul<Output = R::Scalar> + Zero,
    RuleKey: Clone + Eq + std::hash::Hash,
    F: Fn(
            &LocallyValidatedAllCodomainFusionTreeBlockStructure<'_, '_, R>,
            &TreeTransformOperation,
            &mut dyn Iterator<Item = usize>,
        ) -> Result<Vec<TransformRows<FusionTreeKey, R::Scalar>>, OperationError>
        + Send
        + Sync,
{
    let src_structure = proof.structure();
    let source_axes = operation_source_axes(&operation);
    let groups = src_structure.fusion_tree_groups();

    let mut counted_keys = FxHashSet::default();
    let mut staged_hits = 0;
    let mut staged_misses = 0;
    let mut staged_groups = Vec::with_capacity(groups.len());
    for group in groups {
        let mut group_source_index = FxHashMap::default();
        let mut sources = StagedSources::new();
        let mut source_indices = StagedSourceIndices::new();
        for &src_block_index in group.block_indices() {
            let Some(src_key) = proof.proof().fusion_tree_block_key(src_block_index)? else {
                return Err(OperationError::ExpectedFusionTreeBlock {
                    tensor: "src",
                    index: src_block_index,
                });
            };
            let codomain_tree = src_key.codomain_tree();
            if let Some(&source_index) = group_source_index.get(codomain_tree) {
                source_indices.push(source_index);
                continue;
            }
            // Why not coalesce absent rows across groups: a worker owns one
            // complete block transform plus assembly. Cross-group duplicates
            // may recompute, while ordered `entry` commit publishes only the
            // first result deterministically.
            let memo_key = (rule_key.clone(), operation.clone(), codomain_tree.clone());
            let rows = memo.get(&memo_key).map(Arc::clone);
            if counted_keys.insert(codomain_tree) {
                if rows.is_some() {
                    staged_hits += 1;
                } else {
                    staged_misses += 1;
                }
            }
            let source_index = sources.len();
            group_source_index.insert(codomain_tree, source_index);
            sources.push((src_block_index, codomain_tree, rows));
            source_indices.push(source_index);
        }
        staged_groups.push(StagedGroupRows {
            group,
            sources,
            source_alignment: StagedSourceAlignment::Explicit(source_indices),
        });
    }
    let completed = execute_staged_groups(staged_groups, threads, |staged| {
        let (resolved, computed) = resolve_staged_group_rows(staged.sources, |missing_indices| {
            block_transform(proof, &operation, missing_indices)
        })?;
        let StagedSourceAlignment::Explicit(source_indices) = staged.source_alignment else {
            return Err(OperationError::StructureMismatch {
                tensor: "staged all-codomain alignment",
            });
        };
        let mut source_cursor = 0usize;

        let mut rows_for = |_: usize, codomain_tree: &FusionTreeKey| {
            let index = source_indices.get(source_cursor).copied();
            source_cursor += 1;
            index
                .and_then(|index| resolved.get(index))
                .filter(|(_, key, _)| *key == codomain_tree)
                .and_then(|(_, _, rows)| rows.as_ref())
                .map(Arc::clone)
                .ok_or(OperationError::StructureMismatch {
                    tensor: "staged all-codomain rows",
                })
        };
        let specs =
            if operation.is_identity_for(staged.group.group_key().codomain_uncoupled().len(), 0) {
                assemble_identity_all_codomain_group_specs(
                    src_structure,
                    &staged.group,
                    &source_axes,
                    &mut rows_for,
                )
            } else {
                assemble_all_codomain_group_specs(
                    src_structure,
                    &staged.group,
                    &source_axes,
                    &mut rows_for,
                )
            }?;
        drop(rows_for);
        if source_cursor != staged.group.block_indices().len()
            || source_indices.len() != source_cursor
        {
            return Err(OperationError::StructureMismatch {
                tensor: "staged all-codomain row order",
            });
        }
        Ok(CompletedGroupRows { specs, computed })
    })?;

    let mut specs = Vec::new();
    for completed_group in completed {
        specs.extend(completed_group.specs);
        for (key, rows) in completed_group.computed {
            let memo_key = (rule_key.clone(), operation.clone(), key);
            memo.entry(memo_key).or_insert(rows);
        }
    }
    *memo_hits += staged_hits;
    *memo_misses += staged_misses;
    Ok(TreeTransformGroupPlan::new(specs))
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
            FusionTreeBlockKey::pair(dst_codomain_tree.clone(), src_key.domain_tree().clone());
        specs.push(
            TreeTransformGroupBlockSpec::single(
                group.group_key().clone(),
                dst_key,
                src_key.clone(),
                coefficient.clone(),
            )
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
    let mut src_keys = Vec::<BlockKey>::with_capacity(src_block_indices.len());
    let mut dst_keys = Vec::<BlockKey>::new();
    let mut dst_index_by_key = FxHashMap::<BlockKey, usize>::default();
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
        src_keys.push(BlockKey::from(src_key.clone()));

        let transformed = rows_for(src_block_index, src_key.codomain_tree())?;
        if let [(dst_codomain_tree, coefficient)] = transformed.as_slice() {
            let dst_key = BlockKey::from(FusionTreeBlockKey::pair(
                dst_codomain_tree.clone(),
                src_key.domain_tree().clone(),
            ));
            if !direct_dst_keys.insert(dst_key.clone()) {
                is_injective_singleton = false;
            }
            direct_rows.push((
                BlockKey::from(src_key.clone()),
                dst_key,
                coefficient.clone(),
            ));
        } else {
            is_injective_singleton = false;
        }
        for (dst_codomain_tree, coefficient) in transformed.iter() {
            let dst_key = BlockKey::from(FusionTreeBlockKey::pair(
                dst_codomain_tree.clone(),
                src_key.domain_tree().clone(),
            ));
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
        group,
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

    Ok(vec![TreeTransformGroupBlockSpec::multi(
        group.group_key().clone(),
        dst_keys,
        src_keys,
        rows.into_coefficients(),
    )
    .with_shared_source_axes(Arc::clone(source_axes))])
}

/// Shape-independent recoupling rows for one source tree under one
/// operation: the caching unit of TensorKit's `@cached` `fstranspose`/`fsbraid`. Rows survive
/// degeneracy (bond-dimension) changes because they depend only on the tree
/// keys, so chi sweeps recompile plans without recomputing F/R-symbol
/// contractions.
pub(crate) type TreePairRowMemo<T, RuleKey> = FxHashMap<
    (RuleKey, TreeTransformOperation, FusionTreeBlockKey),
    SharedTransformRows<FusionTreeBlockKey, T>,
>;

/// Batched tree-pair rows over a whole block's source trees
/// (all sharing uncoupled sectors, e.g. one [`FusionTreeBlockGroup`]). The
/// TensorKit 0.17 `fsbraid`/`fstranspose` batching: the bend/braid/cyclic step
/// structure is walked once for the block, not once per source. Returns rows
/// per source in `src_keys` order.
pub(crate) fn transformed_tree_pair_rows_block<R>(
    rule: &R,
    operation: &TreeTransformOperation,
    src_keys: &[FusionTreeBlockKey],
) -> Result<TransformBlockRows<FusionTreeBlockKey, R::Scalar>, OperationError>
where
    R: MultiplicityFreeRigidSymbols,
    R::Scalar: Clone + Add<Output = R::Scalar> + Mul<Output = R::Scalar> + Zero,
{
    match operation {
        TreeTransformOperation::Permute {
            codomain_permutation,
            domain_permutation,
        } => multiplicity_free_permute_tree_pair_block(
            rule,
            src_keys,
            codomain_permutation,
            domain_permutation,
        )
        .map_err(OperationError::from_core_preserving_context),
        TreeTransformOperation::Braid {
            codomain_permutation,
            domain_permutation,
            codomain_levels,
            domain_levels,
        } => multiplicity_free_braid_tree_pair_block(
            rule,
            src_keys,
            codomain_permutation,
            domain_permutation,
            codomain_levels,
            domain_levels,
        )
        .map_err(OperationError::from_core_preserving_context),
        TreeTransformOperation::Transpose {
            codomain_permutation,
            domain_permutation,
        } => multiplicity_free_transpose_tree_pair_block(
            rule,
            src_keys,
            codomain_permutation,
            domain_permutation,
        )
        .map_err(OperationError::from_core_preserving_context),
    }
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
    let mut specs = Vec::new();
    for group in src_structure.fusion_tree_groups() {
        let mut rows_for = |index: usize, source: &FusionTreeBlockKey| {
            transform_tree_pair_rows_for_block_index(source_proof, &operation, index, source)
                .map(Arc::new)
        };
        if operation.is_identity_for(
            group.group_key().codomain_uncoupled().len(),
            group.group_key().domain_uncoupled().len(),
        ) {
            specs.extend(assemble_identity_tree_pair_group_specs(
                src_structure,
                &group,
                &source_axes,
                &mut rows_for,
            )?);
        } else {
            specs.extend(assemble_tree_pair_group_specs(
                src_structure,
                &group,
                &source_axes,
                &mut rows_for,
            )?);
        }
    }
    Ok(TreeTransformGroupPlan::new(specs))
}

/// Generic-fusion (outer-multiplicity) tree-pair plan compile — the Stage B2c
/// dispatch receptacle for SU(3)/SO(N≥7)/Sp(N) rules. Parallel entry to
/// `build_multiplicity_free_tree_pair_transform_group_plan`: it reuses the
/// exact same group-spec assembly (`assemble_tree_pair_group_specs`, generic
/// over the coefficient type) and differs only in the recoupling-row source
/// (the validated per-block Generic tree-pair executor).
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
/// runtime `has_multiplicity` gate below defends against a
/// `GenericRigidSymbols` rule that reports a multiplicity-free style. A
/// `has_multiplicity()` dispatch over a dyn-style entry is a Stage B3 concern
/// (the SU(3) provider / generic-capable facade), where a caller can hold a rule
/// of unknown style.
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
    for group in src_structure.fusion_tree_groups() {
        let mut rows_for = |index: usize, _: &FusionTreeBlockKey| {
            let rows = match &operation {
                TreeTransformOperation::Permute {
                    codomain_permutation,
                    domain_permutation,
                } => source_proof.generic_permute_tree_pair_for_block_index(
                    index,
                    codomain_permutation,
                    domain_permutation,
                ),
                TreeTransformOperation::Braid {
                    codomain_permutation,
                    domain_permutation,
                    codomain_levels,
                    domain_levels,
                } => source_proof.generic_braid_tree_pair_for_block_index(
                    index,
                    codomain_permutation,
                    domain_permutation,
                    codomain_levels,
                    domain_levels,
                ),
                TreeTransformOperation::Transpose {
                    codomain_permutation,
                    domain_permutation,
                } => source_proof.generic_transpose_tree_pair_for_block_index(
                    index,
                    codomain_permutation,
                    domain_permutation,
                ),
            }
            .map_err(OperationError::from_core_preserving_context)?;
            Ok(Arc::new(rows))
        };
        if operation.is_identity_for(
            group.group_key().codomain_uncoupled().len(),
            group.group_key().domain_uncoupled().len(),
        ) {
            specs.extend(assemble_identity_tree_pair_group_specs(
                src_structure,
                &group,
                &source_axes,
                &mut rows_for,
            )?);
        } else {
            specs.extend(assemble_tree_pair_group_specs(
                src_structure,
                &group,
                &source_axes,
                &mut rows_for,
            )?);
        }
    }

    Ok(TreeTransformGroupPlan::new(specs))
}

/// Memoized plan build: recoupling rows come from a shape-independent
/// tree-granular memo (TensorKit `fstranspose`/`fsbraid` `@cached` analog), so recompiling
/// for a new degeneracy pattern reuses every F/R-symbol contraction.
///
/// Serial and parallel builds share the same staged group executor. Each
/// worker performs one whole-block row transform and assembles that group's
/// specs; memo entries and statistics are committed in source order only after
/// every group succeeds.
#[allow(clippy::too_many_arguments)]
#[cfg(test)]
pub(crate) fn build_multiplicity_free_tree_pair_transform_group_plan_memoized<R, RuleKey>(
    rule: &R,
    rule_key: &RuleKey,
    operation: TreeTransformOperation,
    src_structure: &BlockStructure,
    memo: &mut TreePairRowMemo<R::Scalar, RuleKey>,
    memo_hits: &mut usize,
    memo_misses: &mut usize,
    threads: usize,
) -> Result<TreeTransformGroupPlan<R::Scalar>, OperationError>
where
    R: MultiplicityFreeRigidSymbols,
    R::Scalar: Clone + Add<Output = R::Scalar> + Mul<Output = R::Scalar> + Zero,
    RuleKey: Clone + Eq + std::hash::Hash,
{
    let proof = validate_multiplicity_free_tree_pair_preflight(rule, &operation, src_structure)?;
    build_multiplicity_free_tree_pair_transform_group_plan_memoized_validated(
        &proof,
        rule_key,
        operation,
        memo,
        memo_hits,
        memo_misses,
        threads,
    )
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn build_multiplicity_free_tree_pair_transform_group_plan_memoized_validated<
    R,
    RuleKey,
>(
    proof: &LocallyValidatedFusionTreeBlockStructure<'_, '_, R>,
    rule_key: &RuleKey,
    operation: TreeTransformOperation,
    memo: &mut TreePairRowMemo<R::Scalar, RuleKey>,
    memo_hits: &mut usize,
    memo_misses: &mut usize,
    threads: usize,
) -> Result<TreeTransformGroupPlan<R::Scalar>, OperationError>
where
    R: MultiplicityFreeRigidSymbols,
    R::Scalar: Clone + Add<Output = R::Scalar> + Mul<Output = R::Scalar> + Zero,
    RuleKey: Clone + Eq + std::hash::Hash,
{
    build_multiplicity_free_tree_pair_transform_group_plan_memoized_impl(
        proof,
        rule_key,
        operation,
        memo,
        memo_hits,
        memo_misses,
        threads,
        transform_tree_pair_rows_for_block_indices,
    )
}

fn transform_tree_pair_rows_for_block_indices<R>(
    proof: &LocallyValidatedFusionTreeBlockStructure<'_, '_, R>,
    operation: &TreeTransformOperation,
    block_indices: &mut dyn Iterator<Item = usize>,
) -> Result<Vec<TransformRows<FusionTreeBlockKey, R::Scalar>>, OperationError>
where
    R: MultiplicityFreeRigidSymbols,
    R::Scalar: Clone + Add<Output = R::Scalar> + Mul<Output = R::Scalar>,
{
    let Some(first_index) = block_indices.next() else {
        return Ok(Vec::new());
    };
    let first = proof.fusion_tree_block_key(first_index)?.ok_or(
        OperationError::ExpectedFusionTreeBlock {
            tensor: "src",
            index: first_index,
        },
    )?;
    let prepared = prepare_tree_pair_operation(
        proof.rule(),
        operation,
        (
            first.codomain_tree().uncoupled().len(),
            first.domain_tree().uncoupled().len(),
        ),
    )?;
    let indices = std::iter::once(first_index).chain(block_indices);
    let transformed = match operation {
        TreeTransformOperation::Transpose { .. } => {
            proof.execute_multiplicity_free_transpose_for_block_indices(indices, prepared)
        }
        TreeTransformOperation::Permute { .. } | TreeTransformOperation::Braid { .. } => {
            proof.execute_multiplicity_free_braid_for_block_indices(indices, prepared)
        }
    };
    transformed.map_err(OperationError::from_core_preserving_context)
}

fn transform_tree_pair_rows_for_block_index<R>(
    proof: &LocallyValidatedFusionTreeBlockStructure<'_, '_, R>,
    operation: &TreeTransformOperation,
    block_index: usize,
    source: &FusionTreeBlockKey,
) -> Result<TransformRows<FusionTreeBlockKey, R::Scalar>, OperationError>
where
    R: MultiplicityFreeRigidSymbols,
    R::Scalar: Clone + Add<Output = R::Scalar> + Mul<Output = R::Scalar>,
{
    let actual_source = proof.fusion_tree_block_key(block_index)?.ok_or(
        OperationError::ExpectedFusionTreeBlock {
            tensor: "src",
            index: block_index,
        },
    )?;
    if actual_source != source {
        return Err(OperationError::StructureMismatch {
            tensor: "proof-bound tree-pair source",
        });
    }
    let prepared = prepare_tree_pair_operation(
        proof.rule(),
        operation,
        (
            source.codomain_tree().uncoupled().len(),
            source.domain_tree().uncoupled().len(),
        ),
    )?;
    proof
        .execute_multiplicity_free_for_block_index(block_index, &prepared)
        .map_err(OperationError::from_core_preserving_context)
}

#[allow(clippy::too_many_arguments)]
fn build_multiplicity_free_tree_pair_transform_group_plan_memoized_impl<R, RuleKey, F>(
    proof: &LocallyValidatedFusionTreeBlockStructure<'_, '_, R>,
    rule_key: &RuleKey,
    operation: TreeTransformOperation,
    memo: &mut TreePairRowMemo<R::Scalar, RuleKey>,
    memo_hits: &mut usize,
    memo_misses: &mut usize,
    threads: usize,
    block_transform: F,
) -> Result<TreeTransformGroupPlan<R::Scalar>, OperationError>
where
    R: MultiplicityFreeRigidSymbols,
    R::Scalar: Clone + Add<Output = R::Scalar> + Mul<Output = R::Scalar> + Zero,
    RuleKey: Clone + Eq + std::hash::Hash,
    F: Fn(
            &LocallyValidatedFusionTreeBlockStructure<'_, '_, R>,
            &TreeTransformOperation,
            &mut dyn Iterator<Item = usize>,
        ) -> Result<Vec<TransformRows<FusionTreeBlockKey, R::Scalar>>, OperationError>
        + Send
        + Sync,
{
    let src_structure = proof.structure();
    let source_axes = operation_source_axes(&operation);
    let groups = src_structure.fusion_tree_groups();

    let mut staged_hits = 0;
    let mut staged_misses = 0;
    let mut staged_groups = Vec::with_capacity(groups.len());
    for group in groups {
        let mut sources = StagedSources::new();
        for &src_block_index in group.block_indices() {
            let Some(src_key) = proof.fusion_tree_block_key(src_block_index)? else {
                return Err(OperationError::ExpectedFusionTreeBlock {
                    tensor: "src",
                    index: src_block_index,
                });
            };
            let memo_key = (rule_key.clone(), operation.clone(), src_key.clone());
            let rows = memo.get(&memo_key).map(Arc::clone);
            if rows.is_some() {
                staged_hits += 1;
            } else {
                staged_misses += 1;
            }
            sources.push((src_block_index, src_key, rows));
        }
        staged_groups.push(StagedGroupRows {
            group,
            sources,
            source_alignment: StagedSourceAlignment::Identity,
        });
    }
    let completed = execute_staged_groups(staged_groups, threads, |staged| {
        let (resolved, computed) = resolve_staged_group_rows(staged.sources, |missing_indices| {
            block_transform(proof, &operation, missing_indices)
        })?;
        if !matches!(staged.source_alignment, StagedSourceAlignment::Identity) {
            return Err(OperationError::StructureMismatch {
                tensor: "staged tree-pair alignment",
            });
        }
        let mut source_cursor = 0usize;

        let mut rows_for = |_: usize, src_key: &FusionTreeBlockKey| {
            let index = source_cursor;
            source_cursor += 1;
            resolved
                .get(index)
                .filter(|(_, key, _)| *key == src_key)
                .and_then(|(_, _, rows)| rows.as_ref())
                .map(Arc::clone)
                .ok_or(OperationError::StructureMismatch {
                    tensor: "staged tree-pair rows",
                })
        };
        let specs = if operation.is_identity_for(
            staged.group.group_key().codomain_uncoupled().len(),
            staged.group.group_key().domain_uncoupled().len(),
        ) {
            assemble_identity_tree_pair_group_specs(
                src_structure,
                &staged.group,
                &source_axes,
                &mut rows_for,
            )
        } else {
            assemble_tree_pair_group_specs(
                src_structure,
                &staged.group,
                &source_axes,
                &mut rows_for,
            )
        }?;
        drop(rows_for);
        if source_cursor != staged.group.block_indices().len() {
            return Err(OperationError::StructureMismatch {
                tensor: "staged tree-pair row order",
            });
        }
        Ok(CompletedGroupRows { specs, computed })
    })?;
    let mut specs = Vec::new();
    for completed_group in completed {
        specs.extend(completed_group.specs);
        for (key, rows) in completed_group.computed {
            memo.insert((rule_key.clone(), operation.clone(), key), rows);
        }
    }
    *memo_hits += staged_hits;
    *memo_misses += staged_misses;
    Ok(TreeTransformGroupPlan::new(specs))
}

#[cfg(test)]
#[allow(clippy::too_many_arguments)]
pub(crate) fn build_multiplicity_free_tree_pair_transform_group_plan_memoized_with_block_transform<
    R,
    RuleKey,
    F,
>(
    rule: &R,
    rule_key: &RuleKey,
    operation: TreeTransformOperation,
    src_structure: &BlockStructure,
    memo: &mut TreePairRowMemo<R::Scalar, RuleKey>,
    memo_hits: &mut usize,
    memo_misses: &mut usize,
    threads: usize,
    block_transform: F,
) -> Result<TreeTransformGroupPlan<R::Scalar>, OperationError>
where
    R: MultiplicityFreeRigidSymbols,
    R::Scalar: Clone + Add<Output = R::Scalar> + Mul<Output = R::Scalar> + Zero,
    RuleKey: Clone + Eq + std::hash::Hash,
    F: Fn(
            &R,
            &TreeTransformOperation,
            &[FusionTreeBlockKey],
        ) -> Result<TransformBlockRows<FusionTreeBlockKey, R::Scalar>, OperationError>
        + Send
        + Sync,
{
    let proof = validate_multiplicity_free_tree_pair_preflight(rule, &operation, src_structure)?;
    build_multiplicity_free_tree_pair_transform_group_plan_memoized_impl(
        &proof,
        rule_key,
        operation,
        memo,
        memo_hits,
        memo_misses,
        threads,
        |proof, operation, block_indices| {
            let size_hint = block_indices.size_hint();
            let mut source_keys = Vec::with_capacity(size_hint.1.unwrap_or(size_hint.0));
            for index in block_indices {
                source_keys.push(
                    proof
                        .fusion_tree_block_key(index)?
                        .ok_or(OperationError::ExpectedFusionTreeBlock {
                            tensor: "src",
                            index,
                        })?
                        .clone(),
                );
            }
            let rows = block_transform(proof.rule(), operation, &source_keys)?;
            Ok(rows)
        },
    )
}

fn assemble_identity_tree_pair_group_specs<T, F>(
    src_structure: &BlockStructure,
    group: &FusionTreeBlockGroup,
    source_axes: &Arc<[usize]>,
    rows_for: &mut F,
) -> Result<Vec<TreeTransformGroupBlockSpec<T>>, OperationError>
where
    T: Clone,
    F: FnMut(
        usize,
        &FusionTreeBlockKey,
    ) -> Result<Arc<Vec<(FusionTreeBlockKey, T)>>, OperationError>,
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
        // coefficient here: consuming the cached row preserves memo/stat and
        // scalar-conversion semantics across serial and parallel builders.
        specs.push(
            TreeTransformGroupBlockSpec::single(
                group.group_key().clone(),
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
    F: FnMut(
        usize,
        &FusionTreeBlockKey,
    ) -> Result<Arc<Vec<(FusionTreeBlockKey, T)>>, OperationError>,
{
    let src_block_indices = group.block_indices();
    let mut src_keys = Vec::<BlockKey>::with_capacity(src_block_indices.len());
    let mut dst_keys = Vec::<BlockKey>::new();
    let mut dst_index_by_key = FxHashMap::<BlockKey, usize>::default();
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
        src_keys.push(BlockKey::from(src_key.clone()));

        let transformed = rows_for(src_block_index, src_key)?;
        if let [(dst_tree_key, coefficient)] = transformed.as_slice() {
            let dst_key = BlockKey::from(dst_tree_key.clone());
            if !direct_dst_keys.insert(dst_key.clone()) {
                is_injective_singleton = false;
            }
            direct_rows.push((
                BlockKey::from(src_key.clone()),
                dst_key,
                coefficient.clone(),
            ));
        } else {
            is_injective_singleton = false;
        }

        for (dst_tree_key, coefficient) in transformed.iter() {
            let dst_key = BlockKey::from(dst_tree_key.clone());
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
        group,
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

    Ok(vec![TreeTransformGroupBlockSpec::multi(
        group.group_key().clone(),
        dst_keys,
        src_keys,
        rows.into_coefficients(),
    )
    .with_shared_source_axes(Arc::clone(source_axes))])
}

fn lower_injective_singleton_rows<T>(
    group: &FusionTreeBlockGroup,
    source_axes: &Arc<[usize]>,
    direct_rows: Vec<(BlockKey, BlockKey, T)>,
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
                TreeTransformGroupBlockSpec::single(
                    group.group_key().clone(),
                    dst_key,
                    src_key,
                    coefficient,
                )
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
        let Some(src_key) = source_proof.fusion_tree_block_key(index)? else {
            continue;
        };
        let source_split = (
            src_key.codomain_tree().uncoupled().len(),
            src_key.domain_tree().uncoupled().len(),
        );
        if primary_prepared.is_none() {
            primary_prepared = Some((
                source_split,
                prepare_tree_pair_operation(source_proof.rule(), &operation, source_split)?,
            ));
        }
        let transformed = if let Some((primary_split, prepared)) = primary_prepared.as_ref() {
            if *primary_split == source_split {
                source_proof.execute_unique_rigid_for_block_index(index, prepared)
            } else {
                let prepared_by_split = additional_prepared.get_or_insert_with(FxHashMap::default);
                let prepared = match prepared_by_split.entry(source_split) {
                    Entry::Occupied(entry) => entry.into_mut(),
                    Entry::Vacant(entry) => entry.insert(prepare_tree_pair_operation(
                        source_proof.rule(),
                        &operation,
                        source_split,
                    )?),
                };
                source_proof.execute_unique_rigid_for_block_index(index, prepared)
            }
        } else {
            unreachable!("first fusion-tree block prepares its source split")
        }
        .map_err(OperationError::from_core_preserving_context)?;
        specs.push(
            TreeTransformGroupBlockSpec::single(
                src_key.group_key(),
                transformed.0,
                src_key.clone(),
                transformed.1,
            )
            .with_shared_source_axes(Arc::clone(&source_axes)),
        );
    }
    Ok(TreeTransformGroupPlan::new(specs))
}

fn prepare_tree_pair_operation<R>(
    rule: &R,
    operation: &TreeTransformOperation,
    (source_codomain_rank, source_domain_rank): (usize, usize),
) -> Result<PreparedTreePairOperation, OperationError>
where
    R: FusionRule,
{
    match operation {
        TreeTransformOperation::Permute {
            codomain_permutation,
            domain_permutation,
        } => PreparedTreePairOperation::prepare_permute(
            rule,
            source_codomain_rank,
            source_domain_rank,
            codomain_permutation,
            domain_permutation,
        ),
        TreeTransformOperation::Braid {
            codomain_permutation,
            domain_permutation,
            codomain_levels,
            domain_levels,
        } => PreparedTreePairOperation::prepare_braid(
            rule,
            source_codomain_rank,
            source_domain_rank,
            codomain_permutation,
            domain_permutation,
            codomain_levels,
            domain_levels,
        ),
        TreeTransformOperation::Transpose {
            codomain_permutation,
            domain_permutation,
        } => PreparedTreePairOperation::prepare_transpose(
            source_codomain_rank,
            source_domain_rank,
            codomain_permutation,
            domain_permutation,
        ),
    }
    .map_err(OperationError::from_core_preserving_context)
}

fn prepare_tree_pair_operation_syntax(
    operation: &TreeTransformOperation,
    (source_codomain_rank, source_domain_rank): (usize, usize),
) -> Result<(), OperationError> {
    match operation {
        TreeTransformOperation::Permute {
            codomain_permutation,
            domain_permutation,
        } => PreparedTreePairOperation::validate_permute_syntax(
            source_codomain_rank,
            source_domain_rank,
            codomain_permutation,
            domain_permutation,
        ),
        TreeTransformOperation::Braid {
            codomain_permutation,
            domain_permutation,
            codomain_levels,
            domain_levels,
        } => PreparedTreePairOperation::validate_braid_syntax(
            source_codomain_rank,
            source_domain_rank,
            codomain_permutation,
            domain_permutation,
            codomain_levels,
            domain_levels,
        ),
        TreeTransformOperation::Transpose {
            codomain_permutation,
            domain_permutation,
        } => PreparedTreePairOperation::validate_transpose_syntax(
            source_codomain_rank,
            source_domain_rank,
            codomain_permutation,
            domain_permutation,
        ),
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

    match operation {
        TreeTransformOperation::Permute {
            domain_permutation,
            ..
        } if domain_permutation.is_empty() => Ok(()),
        TreeTransformOperation::Braid {
            domain_permutation,
            domain_levels,
            ..
        } if domain_permutation.is_empty() && domain_levels.is_empty() => Ok(()),
        TreeTransformOperation::Permute { .. } | TreeTransformOperation::Braid { .. } => {
            Err(scope_error())
        }
        TreeTransformOperation::Transpose { .. } => Err(OperationError::UnsupportedTreeTransformScope {
            operation: Box::new(operation.clone()),
            message: "all-codomain UniqueFusion lowering supports explicit Permute or Braid operations",
        }),
    }
}

fn operation_source_axes(operation: &TreeTransformOperation) -> Arc<[usize]> {
    match operation {
        TreeTransformOperation::Permute {
            codomain_permutation,
            domain_permutation,
        }
        | TreeTransformOperation::Braid {
            codomain_permutation,
            domain_permutation,
            ..
        }
        | TreeTransformOperation::Transpose {
            codomain_permutation,
            domain_permutation,
        } => codomain_permutation
            .iter()
            .chain(domain_permutation)
            .copied()
            .collect(),
    }
}

fn validate_all_codomain_fusion_tree_block<R>(
    rule: &R,
    index: usize,
    key: &FusionTreeBlockKey,
) -> Result<(), OperationError>
where
    R: FusionRule,
{
    let domain = key.domain_tree();
    let empty_domain_coupled_is_valid = domain
        .coupled()
        .map_or(true, |coupled| coupled == rule.vacuum());
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
