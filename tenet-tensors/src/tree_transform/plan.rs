use core::ops::{Add, Mul};
use rustc_hash::{FxHashMap, FxHashSet};
use std::sync::Arc;

use num_traits::Zero;
use tenet_core::{
    generic_braid_tree_pair, generic_permute_tree_pair, generic_transpose_tree_pair,
    multiplicity_free_braid_tree, multiplicity_free_braid_tree_pair,
    multiplicity_free_braid_tree_pair_block, multiplicity_free_permute_tree,
    multiplicity_free_permute_tree_pair, multiplicity_free_permute_tree_pair_block,
    multiplicity_free_transpose_tree_pair, multiplicity_free_transpose_tree_pair_block, BlockKey,
    BlockStructure, FusionRule, FusionTreeBlockGroup, FusionTreeBlockKey, FusionTreeKey,
    GenericBraidScalar, GenericRigidSymbols, MultiplicityFreeFusionSymbols,
    MultiplicityFreeRigidSymbols,
};
#[cfg(test)]
use tenet_core::{
    unique_braid_tree, unique_braid_tree_pair, unique_permute_tree, unique_permute_tree_pair,
    unique_transpose_tree_pair, FusionStyleKind, MultiplicityFreePivotalSymbols,
};

use crate::OperationError;

use super::operation::{TreeTransformOperation, ValidateBraidingSupport};

pub use tenet_operations::transform_plan::{
    TreeTransformBlockSpec, TreeTransformGroupBlockSpec, TreeTransformGroupPlan,
    TreeTransformKeyBlockSpec,
};

/// Build a TensorKit-style grouped tree-transform plan for multiplicity-free
/// fusion rules.
///
/// This is the generic callback form: each source tree may map to multiple
/// destination trees, and duplicate destinations are accumulated into one
/// group-level recoupling matrix. `GenericFusion` with vertex multiplicities is
/// intentionally not represented by this scalar-coefficient API.
pub fn build_tree_transform_group_plan<T, R, F>(
    rule: &R,
    operation: TreeTransformOperation,
    src_structure: &BlockStructure,
    mut transform: F,
) -> Result<TreeTransformGroupPlan<T>, OperationError>
where
    R: FusionRule,
    T: Clone + Add<Output = T> + Zero,
    F: FnMut(&FusionTreeBlockKey) -> Result<Vec<(FusionTreeBlockKey, T)>, OperationError>,
{
    if !rule.fusion_style().is_multiplicity_free() {
        return Err(OperationError::UnsupportedFusionStyle {
            operation,
            style: rule.fusion_style(),
        });
    }
    operation.validate_braiding_support(rule)?;
    let source_axes = operation_source_axes(&operation);

    let mut specs = Vec::new();
    for group in src_structure.fusion_tree_groups() {
        specs.extend(assemble_tree_pair_group_specs(
            src_structure,
            &group,
            &source_axes,
            &mut |src_key| transform(src_key).map(Arc::new),
        )?);
    }

    Ok(TreeTransformGroupPlan::new(specs))
}

/// Standard all-codomain tree-transform builder for Unique and Simple
/// multiplicity-free rules.
pub fn build_all_codomain_tree_transform_group_plan<R>(
    rule: &R,
    operation: TreeTransformOperation,
    src_structure: &BlockStructure,
) -> Result<TreeTransformGroupPlan<R::Scalar>, OperationError>
where
    R: MultiplicityFreeFusionSymbols,
    R::Scalar: Clone + Add<Output = R::Scalar> + Mul<Output = R::Scalar> + Zero,
{
    build_multiplicity_free_all_codomain_tree_transform_group_plan(rule, operation, src_structure)
}

/// Standard full tree-pair transform builder for Unique and Simple
/// multiplicity-free rules.
pub fn build_tree_pair_transform_group_plan<R>(
    rule: &R,
    operation: TreeTransformOperation,
    src_structure: &BlockStructure,
) -> Result<TreeTransformGroupPlan<R::Scalar>, OperationError>
where
    R: MultiplicityFreeRigidSymbols,
    R::Scalar: Clone + Add<Output = R::Scalar> + Mul<Output = R::Scalar> + Zero,
{
    build_multiplicity_free_tree_pair_transform_group_plan(rule, operation, src_structure)
}

#[cfg(test)]
pub(crate) fn build_unique_tree_transform_group_plan<T, R, F>(
    rule: &R,
    operation: TreeTransformOperation,
    src_structure: &BlockStructure,
    mut transform: F,
) -> Result<TreeTransformGroupPlan<T>, OperationError>
where
    R: FusionRule,
    F: FnMut(&FusionTreeBlockKey) -> Result<(FusionTreeBlockKey, T), OperationError>,
{
    if rule.fusion_style() != FusionStyleKind::Unique {
        return Err(OperationError::UnsupportedFusionStyle {
            operation,
            style: rule.fusion_style(),
        });
    }
    operation.validate_braiding_support(rule)?;
    let source_axes = operation_source_axes(&operation);

    let mut specs = Vec::with_capacity(src_structure.block_count());
    for index in 0..src_structure.block_count() {
        let block = src_structure.block(index)?;
        let BlockKey::FusionTree(src_key) = block.key() else {
            return Err(OperationError::ExpectedFusionTreeBlock {
                tensor: "src",
                index,
            });
        };
        let (dst_key, coefficient) = transform(src_key)?;
        specs.push(
            TreeTransformGroupBlockSpec::single(
                src_key.group_key(),
                dst_key,
                src_key.clone(),
                coefficient,
            )
            .with_source_axes(source_axes.clone()),
        );
    }

    Ok(TreeTransformGroupPlan::new(specs))
}

#[cfg(test)]
pub(crate) fn build_unique_all_codomain_tree_transform_group_plan<R>(
    rule: &R,
    operation: TreeTransformOperation,
    src_structure: &BlockStructure,
) -> Result<TreeTransformGroupPlan<R::Scalar>, OperationError>
where
    R: MultiplicityFreeFusionSymbols,
    R::Scalar: Mul<Output = R::Scalar>,
{
    if rule.fusion_style() != FusionStyleKind::Unique {
        return Err(OperationError::UnsupportedFusionStyle {
            operation,
            style: rule.fusion_style(),
        });
    }
    operation.validate_braiding_support(rule)?;
    validate_all_codomain_operation_scope(&operation)?;
    let source_axes = operation_source_axes(&operation);

    let mut specs = Vec::with_capacity(src_structure.block_count());
    for index in 0..src_structure.block_count() {
        let block = src_structure.block(index)?;
        let BlockKey::FusionTree(src_key) = block.key() else {
            return Err(OperationError::ExpectedFusionTreeBlock {
                tensor: "src",
                index,
            });
        };
        validate_all_codomain_fusion_tree_block(rule, index, src_key)?;

        let (dst_codomain_tree, coefficient) = match &operation {
            TreeTransformOperation::Permute {
                codomain_permutation,
                ..
            } => unique_permute_tree(rule, src_key.codomain_tree(), codomain_permutation)?,
            TreeTransformOperation::Braid {
                codomain_permutation,
                codomain_levels,
                ..
            } => unique_braid_tree(
                rule,
                src_key.codomain_tree(),
                codomain_permutation,
                codomain_levels,
            )?,
            TreeTransformOperation::Transpose { .. } => {
                unreachable!("all-codomain operation scope validation rejected transpose")
            }
        };
        let dst_key = FusionTreeBlockKey::pair(dst_codomain_tree, src_key.domain_tree().clone());
        specs.push(
            TreeTransformGroupBlockSpec::single(
                src_key.group_key(),
                dst_key,
                src_key.clone(),
                coefficient,
            )
            .with_source_axes(source_axes.clone()),
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
    build_multiplicity_free_all_codomain_tree_transform_group_plan_with_rows(
        rule,
        operation,
        src_structure,
        |operation, codomain_tree| {
            transformed_all_codomain_rows(rule, operation, codomain_tree).map(Arc::new)
        },
    )
}

fn build_multiplicity_free_all_codomain_tree_transform_group_plan_with_rows<R, F>(
    rule: &R,
    operation: TreeTransformOperation,
    src_structure: &BlockStructure,
    mut rows_for: F,
) -> Result<TreeTransformGroupPlan<R::Scalar>, OperationError>
where
    R: MultiplicityFreeFusionSymbols,
    R::Scalar: Clone + Add<Output = R::Scalar> + Mul<Output = R::Scalar> + Zero,
    F: FnMut(
        &TreeTransformOperation,
        &FusionTreeKey,
    ) -> Result<Arc<Vec<(FusionTreeKey, R::Scalar)>>, OperationError>,
{
    if !rule.fusion_style().is_multiplicity_free() {
        return Err(OperationError::UnsupportedFusionStyle {
            operation,
            style: rule.fusion_style(),
        });
    }
    operation.validate_braiding_support(rule)?;
    validate_all_codomain_operation_scope(&operation)?;
    let source_axes = operation_source_axes(&operation);

    let mut specs = Vec::new();
    for group in src_structure.fusion_tree_groups() {
        if operation.is_identity_for(group.group_key().codomain_uncoupled().len(), 0) {
            specs.extend(assemble_identity_all_codomain_group_specs(
                rule,
                src_structure,
                &group,
                &source_axes,
                &mut |codomain_tree| rows_for(&operation, codomain_tree),
            )?);
        } else {
            specs.extend(assemble_all_codomain_group_specs(
                rule,
                src_structure,
                &group,
                &source_axes,
                &mut |codomain_tree| rows_for(&operation, codomain_tree),
            )?);
        }
    }

    Ok(TreeTransformGroupPlan::new(specs))
}

pub(crate) type AllCodomainRowMemo<T, RuleKey> =
    FxHashMap<(RuleKey, TreeTransformOperation, FusionTreeKey), Arc<Vec<(FusionTreeKey, T)>>>;

fn transformed_all_codomain_rows<R>(
    rule: &R,
    operation: &TreeTransformOperation,
    codomain_tree: &FusionTreeKey,
) -> Result<Vec<(FusionTreeKey, R::Scalar)>, OperationError>
where
    R: MultiplicityFreeFusionSymbols,
    R::Scalar: Clone + Add<Output = R::Scalar> + Mul<Output = R::Scalar>,
{
    let rows = match operation {
        TreeTransformOperation::Permute {
            codomain_permutation,
            ..
        } => multiplicity_free_permute_tree(rule, codomain_tree, codomain_permutation),
        TreeTransformOperation::Braid {
            codomain_permutation,
            codomain_levels,
            ..
        } => {
            multiplicity_free_braid_tree(rule, codomain_tree, codomain_permutation, codomain_levels)
        }
        TreeTransformOperation::Transpose { .. } => {
            unreachable!("all-codomain operation scope validation rejected transpose")
        }
    };
    rows.map_err(OperationError::from_core_preserving_context)
}

#[allow(clippy::too_many_arguments)]
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
    if threads > 1 {
        return build_all_codomain_tree_transform_group_plan_parallel(
            rule,
            rule_key,
            operation,
            src_structure,
            memo,
            memo_hits,
            memo_misses,
            threads,
        );
    }
    build_multiplicity_free_all_codomain_tree_transform_group_plan_with_rows(
        rule,
        operation,
        src_structure,
        |operation, codomain_tree| {
            let memo_key = (rule_key.clone(), operation.clone(), codomain_tree.clone());
            if let Some(rows) = memo.get(&memo_key) {
                *memo_hits += 1;
                return Ok(Arc::clone(rows));
            }
            *memo_misses += 1;
            let rows = Arc::new(transformed_all_codomain_rows(
                rule,
                operation,
                codomain_tree,
            )?);
            memo.insert(memo_key, Arc::clone(&rows));
            Ok(rows)
        },
    )
}

#[allow(clippy::too_many_arguments)]
fn build_all_codomain_tree_transform_group_plan_parallel<R, RuleKey>(
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
    use rayon::prelude::*;

    if !rule.fusion_style().is_multiplicity_free() {
        return Err(OperationError::UnsupportedFusionStyle {
            operation,
            style: rule.fusion_style(),
        });
    }
    operation.validate_braiding_support(rule)?;
    validate_all_codomain_operation_scope(&operation)?;
    let source_axes = operation_source_axes(&operation);
    let groups = src_structure.fusion_tree_groups();

    let mut missing = Vec::new();
    let mut rows_by_codomain =
        FxHashMap::<FusionTreeKey, Arc<Vec<(FusionTreeKey, R::Scalar)>>>::default();
    for group in &groups {
        for &src_block_index in group.block_indices() {
            let block = src_structure.block(src_block_index)?;
            let BlockKey::FusionTree(src_key) = block.key() else {
                return Err(OperationError::ExpectedFusionTreeBlock {
                    tensor: "src",
                    index: src_block_index,
                });
            };
            validate_all_codomain_fusion_tree_block(rule, src_block_index, src_key)?;
            let codomain_tree = src_key.codomain_tree().clone();
            let memo_key = (rule_key.clone(), operation.clone(), codomain_tree.clone());
            if let Some(rows) = memo.get(&memo_key) {
                *memo_hits += 1;
                rows_by_codomain.insert(codomain_tree, Arc::clone(rows));
            } else {
                *memo_misses += 1;
                missing.push((memo_key, codomain_tree));
            }
        }
    }

    let (memo_keys, missing_codomain_trees): (Vec<_>, Vec<_>) = missing.into_iter().unzip();
    let chunk = missing_codomain_trees.len().div_ceil(threads).max(1);
    let computed: Vec<(FusionTreeKey, Arc<Vec<(FusionTreeKey, R::Scalar)>>)> =
        missing_codomain_trees
            .into_par_iter()
            .with_min_len(chunk)
            .map(|codomain_tree| {
                let rows = transformed_all_codomain_rows(rule, &operation, &codomain_tree)?;
                Ok((codomain_tree, Arc::new(rows)))
            })
            .collect::<Result<_, OperationError>>()?;

    for (memo_key, (codomain_tree, rows)) in memo_keys.into_iter().zip(computed) {
        rows_by_codomain.insert(codomain_tree, Arc::clone(&rows));
        memo.insert(memo_key, rows);
    }

    let group_chunk = groups.len().div_ceil(threads).max(1);
    let specs = groups
        .into_par_iter()
        .with_min_len(group_chunk)
        .map(|group| {
            let mut rows_for =
                |codomain_tree: &FusionTreeKey| match rows_by_codomain.get(codomain_tree) {
                    Some(rows) => Ok(Arc::clone(rows)),
                    None => {
                        transformed_all_codomain_rows(rule, &operation, codomain_tree).map(Arc::new)
                    }
                };
            if operation.is_identity_for(group.group_key().codomain_uncoupled().len(), 0) {
                assemble_identity_all_codomain_group_specs(
                    rule,
                    src_structure,
                    &group,
                    &source_axes,
                    &mut rows_for,
                )
            } else {
                assemble_all_codomain_group_specs(
                    rule,
                    src_structure,
                    &group,
                    &source_axes,
                    &mut rows_for,
                )
            }
        })
        .collect::<Result<Vec<_>, OperationError>>()?
        .into_iter()
        .flatten()
        .collect();

    Ok(TreeTransformGroupPlan::new(specs))
}

fn assemble_identity_all_codomain_group_specs<R, T, F>(
    rule: &R,
    src_structure: &BlockStructure,
    group: &FusionTreeBlockGroup,
    source_axes: &[usize],
    rows_for: &mut F,
) -> Result<Vec<TreeTransformGroupBlockSpec<T>>, OperationError>
where
    R: FusionRule,
    T: Clone,
    F: FnMut(&FusionTreeKey) -> Result<Arc<Vec<(FusionTreeKey, T)>>, OperationError>,
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
        validate_all_codomain_fusion_tree_block(rule, src_block_index, src_key)?;
        let transformed = rows_for(src_key.codomain_tree())?;
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
            .with_source_axes(source_axes.to_vec()),
        );
    }
    Ok(specs)
}

fn assemble_all_codomain_group_specs<R, T, F>(
    rule: &R,
    src_structure: &BlockStructure,
    group: &FusionTreeBlockGroup,
    source_axes: &[usize],
    rows_for: &mut F,
) -> Result<Vec<TreeTransformGroupBlockSpec<T>>, OperationError>
where
    R: FusionRule,
    T: Clone + Add<Output = T> + Zero,
    F: FnMut(&FusionTreeKey) -> Result<Arc<Vec<(FusionTreeKey, T)>>, OperationError>,
{
    let src_block_indices = group.block_indices();
    let mut src_keys = Vec::<BlockKey>::with_capacity(src_block_indices.len());
    let mut dst_keys = Vec::<BlockKey>::new();
    let mut dst_index_by_key = FxHashMap::<BlockKey, usize>::default();
    let mut rows = Vec::<Vec<T>>::new();
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
        validate_all_codomain_fusion_tree_block(rule, src_block_index, src_key)?;
        src_keys.push(BlockKey::from(src_key.clone()));

        let transformed = rows_for(src_key.codomain_tree())?;
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
        for row in &mut rows {
            row.push(T::zero());
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
                rows.push(vec![T::zero(); src_column + 1]);
                dst_row
            };
            rows[dst_row][src_column] = rows[dst_row][src_column].clone() + coefficient.clone();
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

    let src_count = src_keys.len();
    let mut recoupling_coefficients_dst_src = Vec::with_capacity(dst_keys.len() * src_count);
    for row in rows {
        recoupling_coefficients_dst_src.extend(row);
    }
    Ok(vec![TreeTransformGroupBlockSpec::multi(
        group.group_key().clone(),
        dst_keys,
        src_keys,
        recoupling_coefficients_dst_src,
    )
    .with_source_axes(source_axes.to_vec())])
}

/// Shape-independent recoupling rows for one source tree under one
/// operation: the caching unit of TensorKit's `@cached` `fstranspose`/`fsbraid`. Rows survive
/// degeneracy (bond-dimension) changes because they depend only on the tree
/// keys, so chi sweeps recompile plans without recomputing F/R-symbol
/// contractions.
pub(crate) type TreePairRowMemo<T, RuleKey> = FxHashMap<
    (RuleKey, TreeTransformOperation, FusionTreeBlockKey),
    Arc<Vec<(FusionTreeBlockKey, T)>>,
>;

pub(crate) fn transformed_tree_pair_rows<R>(
    rule: &R,
    operation: &TreeTransformOperation,
    src_key: &FusionTreeBlockKey,
) -> Result<Vec<(FusionTreeBlockKey, R::Scalar)>, OperationError>
where
    R: MultiplicityFreeRigidSymbols,
    R::Scalar: Clone + Add<Output = R::Scalar> + Mul<Output = R::Scalar> + Zero,
{
    let rows = match operation {
        TreeTransformOperation::Permute {
            codomain_permutation,
            domain_permutation,
        } => multiplicity_free_permute_tree_pair(
            rule,
            src_key,
            codomain_permutation,
            domain_permutation,
        ),
        TreeTransformOperation::Braid {
            codomain_permutation,
            domain_permutation,
            codomain_levels,
            domain_levels,
        } => multiplicity_free_braid_tree_pair(
            rule,
            src_key,
            codomain_permutation,
            domain_permutation,
            codomain_levels,
            domain_levels,
        ),
        TreeTransformOperation::Transpose {
            codomain_permutation,
            domain_permutation,
        } => multiplicity_free_transpose_tree_pair(
            rule,
            src_key,
            codomain_permutation,
            domain_permutation,
        ),
    };
    rows.map_err(OperationError::from_core_preserving_context)
}

/// Batched [`transformed_tree_pair_rows`] over a whole block's source trees
/// (all sharing uncoupled sectors, e.g. one [`FusionTreeBlockGroup`]). The
/// TensorKit 0.17 `fsbraid`/`fstranspose` batching: the bend/braid/cyclic step
/// structure is walked once for the block, not once per source. Returns rows
/// per source in `src_keys` order.
pub(crate) fn transformed_tree_pair_rows_block<R>(
    rule: &R,
    operation: &TreeTransformOperation,
    src_keys: &[FusionTreeBlockKey],
) -> Result<Vec<Vec<(FusionTreeBlockKey, R::Scalar)>>, OperationError>
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
    build_multiplicity_free_tree_pair_transform_group_plan_with_rows(
        rule,
        operation,
        src_structure,
        |operation, src_key| transformed_tree_pair_rows(rule, operation, src_key).map(Arc::new),
    )
}

/// Recoupling rows for one source tree pair under one operation, Generic-fusion
/// (outer-multiplicity) path. Generic sibling of [`transformed_tree_pair_rows`]:
/// identical operation → primitive dispatch, over the adversarially-verified
/// `generic_*_tree_pair` family (Stage B1/B2a/B2b). Adds no recoupling math.
pub(crate) fn transformed_generic_tree_pair_rows<R>(
    rule: &R,
    operation: &TreeTransformOperation,
    src_key: &FusionTreeBlockKey,
) -> Result<Vec<(FusionTreeBlockKey, R::Scalar)>, OperationError>
where
    R: GenericRigidSymbols,
    R::Scalar: GenericBraidScalar,
{
    let rows = match operation {
        TreeTransformOperation::Permute {
            codomain_permutation,
            domain_permutation,
        } => generic_permute_tree_pair(rule, src_key, codomain_permutation, domain_permutation),
        TreeTransformOperation::Braid {
            codomain_permutation,
            domain_permutation,
            codomain_levels,
            domain_levels,
        } => generic_braid_tree_pair(
            rule,
            src_key,
            codomain_permutation,
            domain_permutation,
            codomain_levels,
            domain_levels,
        ),
        TreeTransformOperation::Transpose {
            codomain_permutation,
            domain_permutation,
        } => generic_transpose_tree_pair(rule, src_key, codomain_permutation, domain_permutation),
    };
    rows.map_err(OperationError::from_core_preserving_context)
}

/// Generic-fusion (outer-multiplicity) tree-pair plan compile — the Stage B2c
/// dispatch receptacle for SU(3)/SO(N≥7)/Sp(N) rules. Parallel entry to
/// `build_multiplicity_free_tree_pair_transform_group_plan`: it reuses the
/// exact same group-spec assembly (`assemble_tree_pair_group_specs`, generic
/// over the coefficient type) and differs only in the recoupling-row source
/// (`transformed_generic_tree_pair_rows`).
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
pub fn build_generic_tree_pair_transform_group_plan<R>(
    rule: &R,
    operation: TreeTransformOperation,
    src_structure: &BlockStructure,
) -> Result<TreeTransformGroupPlan<R::Scalar>, OperationError>
where
    R: GenericRigidSymbols,
    R::Scalar: GenericBraidScalar + Zero,
{
    if !rule.fusion_style().has_multiplicity() {
        return Err(OperationError::UnsupportedFusionStyle {
            operation,
            style: rule.fusion_style(),
        });
    }
    operation.validate_braiding_support(rule)?;
    let source_axes = operation_source_axes(&operation);

    let mut specs = Vec::new();
    for group in src_structure.fusion_tree_groups() {
        if operation.is_identity_for(
            group.group_key().codomain_uncoupled().len(),
            group.group_key().domain_uncoupled().len(),
        ) {
            specs.extend(assemble_identity_tree_pair_group_specs(
                src_structure,
                &group,
                &source_axes,
                &mut |src_key| {
                    transformed_generic_tree_pair_rows(rule, &operation, src_key).map(Arc::new)
                },
            )?);
        } else {
            specs.extend(assemble_tree_pair_group_specs(
                src_structure,
                &group,
                &source_axes,
                &mut |src_key| {
                    transformed_generic_tree_pair_rows(rule, &operation, src_key).map(Arc::new)
                },
            )?);
        }
    }

    Ok(TreeTransformGroupPlan::new(specs))
}

/// Memoized plan build: recoupling rows come from a shape-independent
/// tree-granular memo (TensorKit `fstranspose`/`fsbraid` `@cached` analog), so recompiling
/// for a new degeneracy pattern reuses every F/R-symbol contraction.
///
/// `threads <= 1` is the untouched serial path; `threads > 1` runs the
/// parallel compile (see [`build_tree_pair_transform_group_plan_parallel`]),
/// which produces a plan identical to the serial build.
#[allow(clippy::too_many_arguments)]
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
    if threads > 1 {
        return build_tree_pair_transform_group_plan_parallel(
            rule,
            rule_key,
            operation,
            src_structure,
            memo,
            memo_hits,
            memo_misses,
            threads,
        );
    }

    // Pre-populate the memo one whole block at a time via the batched
    // (TensorKit 0.17 `fsbraid`) block transform, so the bend/braid step
    // structure is walked once per block instead of once per source tree. The
    // memo stays keyed per source tree, so assembly below is unchanged and a
    // partially-warm memo only computes the still-missing sources.
    if rule.fusion_style().is_multiplicity_free() {
        for group in src_structure.fusion_tree_groups() {
            let mut missing_keys = Vec::new();
            for &src_block_index in group.block_indices() {
                let block = src_structure.block(src_block_index)?;
                let BlockKey::FusionTree(src_key) = block.key() else {
                    return Err(OperationError::ExpectedFusionTreeBlock {
                        tensor: "src",
                        index: src_block_index,
                    });
                };
                let memo_key = (rule_key.clone(), operation.clone(), src_key.clone());
                if memo.contains_key(&memo_key) {
                    *memo_hits += 1;
                } else {
                    missing_keys.push(src_key.clone());
                }
            }
            if missing_keys.is_empty() {
                continue;
            }
            let batched = transformed_tree_pair_rows_block(rule, &operation, &missing_keys)?;
            for (src_key, rows) in missing_keys.into_iter().zip(batched) {
                let memo_key = (rule_key.clone(), operation.clone(), src_key);
                memo.insert(memo_key, Arc::new(rows));
                *memo_misses += 1;
            }
        }
    }

    build_multiplicity_free_tree_pair_transform_group_plan_with_rows(
        rule,
        operation,
        src_structure,
        |operation, src_key| {
            let memo_key = (rule_key.clone(), operation.clone(), src_key.clone());
            if let Some(rows) = memo.get(&memo_key) {
                return Ok(Arc::clone(rows));
            }
            // Unreachable when the block pre-pass ran (every source was
            // inserted); recomputing is pure, so stay correct for the
            // non-multiplicity-free fallthrough.
            let rows = Arc::new(transformed_tree_pair_rows(rule, operation, src_key)?);
            memo.insert(memo_key, Arc::clone(&rows));
            Ok(rows)
        },
    )
}

/// Parallel plan compile: the analog of TensorKit's threaded
/// `TreeTransformer` construction (`treetransformers.jl:69-90`, one work
/// item per fusion block over `min(nthreads, nblocks)` workers), on rayon's
/// global pool — the pool strided-kernel's threaded kernels and the parallel
/// replay already use — with `with_min_len` bounding the split count to
/// `threads`.
///
/// Two parallel phases with a serial merge between them, so the memo needs
/// no locks and the workspace `unsafe` ban is never tested:
///
/// 1. recoupling rows for memo-missing source trees, one work item per tree
///    (the F/R-symbol contractions, the dominant compile cost), collected
///    into a plain `Vec`;
/// 2. serial: stats + memo insertion in block order (identical counts and
///    entries to the serial build);
/// 3. group spec assembly (dst-key dedup + coefficient matrix), one work
///    item per fusion-tree group, reading the now-complete memo.
///
/// TensorKit gates construction threading on the thread count alone — row
/// cost scales with tree count, not degeneracy, so the replay byte-length
/// gate does not apply; a single missing row / single group degenerates to
/// a serial chunk by construction.
#[allow(clippy::too_many_arguments)]
fn build_tree_pair_transform_group_plan_parallel<R, RuleKey>(
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
    use rayon::prelude::*;

    if !rule.fusion_style().is_multiplicity_free() {
        return Err(OperationError::UnsupportedFusionStyle {
            operation,
            style: rule.fusion_style(),
        });
    }
    operation.validate_braiding_support(rule)?;
    let source_axes = operation_source_axes(&operation);
    let groups = src_structure.fusion_tree_groups();

    // Memo-missing source trees, in block order (block keys are unique
    // within a structure, so no dedup is needed). `rows_by_src` collects
    // this structure's rows keyed by tree only: phase 3 workers read it
    // instead of the memo, so the memo's RuleKey never crosses threads.
    let mut missing = Vec::new();
    let mut rows_by_src =
        FxHashMap::<FusionTreeBlockKey, Arc<Vec<(FusionTreeBlockKey, R::Scalar)>>>::default();
    for group in &groups {
        for &src_block_index in group.block_indices() {
            let block = src_structure.block(src_block_index)?;
            let BlockKey::FusionTree(src_key) = block.key() else {
                return Err(OperationError::ExpectedFusionTreeBlock {
                    tensor: "src",
                    index: src_block_index,
                });
            };
            let memo_key = (rule_key.clone(), operation.clone(), src_key.clone());
            if let Some(rows) = memo.get(&memo_key) {
                *memo_hits += 1;
                rows_by_src.insert(src_key.clone(), Arc::clone(rows));
            } else {
                *memo_misses += 1;
                missing.push((memo_key, src_key.clone()));
            }
        }
    }

    // Phase 1 (parallel): rows for the missing trees. The RuleKey is not
    // needed inside the workers (and carries no Send/Sync bound), so the
    // memo keys stay on this thread and zip back up in order afterwards.
    let (memo_keys, missing_src_keys): (Vec<_>, Vec<_>) = missing.into_iter().unzip();
    let chunk = missing_src_keys.len().div_ceil(threads).max(1);
    let computed: Vec<(
        FusionTreeBlockKey,
        Arc<Vec<(FusionTreeBlockKey, R::Scalar)>>,
    )> = missing_src_keys
        .into_par_iter()
        .with_min_len(chunk)
        .map(|src_key| {
            let rows = transformed_tree_pair_rows(rule, &operation, &src_key)?;
            Ok((src_key, Arc::new(rows)))
        })
        .collect::<Result<_, OperationError>>()?;
    // Phase 2 (serial): memo insertion, preserving block order.
    for (memo_key, (src_key, rows)) in memo_keys.into_iter().zip(computed) {
        rows_by_src.insert(src_key, Arc::clone(&rows));
        memo.insert(memo_key, rows);
    }

    // Phase 3 (parallel): per-group spec assembly against the now-complete
    // memo (every source tree was either a hit or inserted above).
    let group_chunk = groups.len().div_ceil(threads).max(1);
    let specs = groups
        .into_par_iter()
        .with_min_len(group_chunk)
        .map(|group| {
            let mut rows_for = |src_key: &FusionTreeBlockKey| match rows_by_src.get(src_key) {
                Some(rows) => Ok(Arc::clone(rows)),
                // Unreachable by construction (every tree was collected
                // above); recomputing is pure, so stay correct anyway.
                None => transformed_tree_pair_rows(rule, &operation, src_key).map(Arc::new),
            };
            if operation.is_identity_for(
                group.group_key().codomain_uncoupled().len(),
                group.group_key().domain_uncoupled().len(),
            ) {
                assemble_identity_tree_pair_group_specs(
                    src_structure,
                    &group,
                    &source_axes,
                    &mut rows_for,
                )
            } else {
                assemble_tree_pair_group_specs(src_structure, &group, &source_axes, &mut rows_for)
            }
        })
        .collect::<Result<Vec<_>, OperationError>>()?
        .into_iter()
        .flatten()
        .collect();

    Ok(TreeTransformGroupPlan::new(specs))
}

fn build_multiplicity_free_tree_pair_transform_group_plan_with_rows<R, F>(
    rule: &R,
    operation: TreeTransformOperation,
    src_structure: &BlockStructure,
    mut rows_for: F,
) -> Result<TreeTransformGroupPlan<R::Scalar>, OperationError>
where
    R: MultiplicityFreeRigidSymbols,
    R::Scalar: Clone + Add<Output = R::Scalar> + Mul<Output = R::Scalar> + Zero,
    F: FnMut(
        &TreeTransformOperation,
        &FusionTreeBlockKey,
    ) -> Result<Arc<Vec<(FusionTreeBlockKey, R::Scalar)>>, OperationError>,
{
    if !rule.fusion_style().is_multiplicity_free() {
        return Err(OperationError::UnsupportedFusionStyle {
            operation,
            style: rule.fusion_style(),
        });
    }
    operation.validate_braiding_support(rule)?;
    let source_axes = operation_source_axes(&operation);

    let mut specs = Vec::new();
    for group in src_structure.fusion_tree_groups() {
        if operation.is_identity_for(
            group.group_key().codomain_uncoupled().len(),
            group.group_key().domain_uncoupled().len(),
        ) {
            specs.extend(assemble_identity_tree_pair_group_specs(
                src_structure,
                &group,
                &source_axes,
                &mut |src_key| rows_for(&operation, src_key),
            )?);
        } else {
            specs.extend(assemble_tree_pair_group_specs(
                src_structure,
                &group,
                &source_axes,
                &mut |src_key| rows_for(&operation, src_key),
            )?);
        }
    }

    Ok(TreeTransformGroupPlan::new(specs))
}

fn assemble_identity_tree_pair_group_specs<T, F>(
    src_structure: &BlockStructure,
    group: &FusionTreeBlockGroup,
    source_axes: &[usize],
    rows_for: &mut F,
) -> Result<Vec<TreeTransformGroupBlockSpec<T>>, OperationError>
where
    T: Clone,
    F: FnMut(&FusionTreeBlockKey) -> Result<Arc<Vec<(FusionTreeBlockKey, T)>>, OperationError>,
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
        let transformed = rows_for(src_key)?;
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
            .with_source_axes(source_axes.to_vec()),
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
    source_axes: &[usize],
    rows_for: &mut F,
) -> Result<Vec<TreeTransformGroupBlockSpec<T>>, OperationError>
where
    T: Clone + Add<Output = T> + Zero,
    F: FnMut(&FusionTreeBlockKey) -> Result<Arc<Vec<(FusionTreeBlockKey, T)>>, OperationError>,
{
    let src_block_indices = group.block_indices();
    let mut src_keys = Vec::<BlockKey>::with_capacity(src_block_indices.len());
    let mut dst_keys = Vec::<BlockKey>::new();
    let mut dst_index_by_key = FxHashMap::<BlockKey, usize>::default();
    let mut rows = Vec::<Vec<T>>::new();
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

        let transformed = rows_for(src_key)?;
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

        for row in &mut rows {
            row.push(T::zero());
        }
        for (dst_tree_key, coefficient) in transformed.iter() {
            let dst_key = BlockKey::from(dst_tree_key.clone());
            let dst_row = if let Some(&dst_row) = dst_index_by_key.get(&dst_key) {
                dst_row
            } else {
                let dst_row = dst_keys.len();
                dst_index_by_key.insert(dst_key.clone(), dst_row);
                dst_keys.push(dst_key);
                rows.push(vec![T::zero(); src_column + 1]);
                dst_row
            };
            rows[dst_row][src_column] = rows[dst_row][src_column].clone() + coefficient.clone();
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

    let src_count = src_keys.len();
    let mut recoupling_coefficients_dst_src = Vec::with_capacity(dst_keys.len() * src_count);
    for row in rows {
        recoupling_coefficients_dst_src.extend(row);
    }
    Ok(vec![TreeTransformGroupBlockSpec::multi(
        group.group_key().clone(),
        dst_keys,
        src_keys,
        recoupling_coefficients_dst_src,
    )
    .with_source_axes(source_axes.to_vec())])
}

fn lower_injective_singleton_rows<T>(
    group: &FusionTreeBlockGroup,
    source_axes: &[usize],
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
                .with_source_axes(source_axes.to_vec())
            })
            .collect(),
    )
}

#[cfg(test)]
pub(crate) fn build_unique_tree_pair_transform_group_plan<R>(
    rule: &R,
    operation: TreeTransformOperation,
    src_structure: &BlockStructure,
) -> Result<TreeTransformGroupPlan<R::Scalar>, OperationError>
where
    R: MultiplicityFreePivotalSymbols,
    R::Scalar: Mul<Output = R::Scalar>,
{
    if rule.fusion_style() != FusionStyleKind::Unique {
        return Err(OperationError::UnsupportedFusionStyle {
            operation,
            style: rule.fusion_style(),
        });
    }
    operation.validate_braiding_support(rule)?;
    let source_axes = operation_source_axes(&operation);

    let mut specs = Vec::with_capacity(src_structure.block_count());
    for index in 0..src_structure.block_count() {
        let block = src_structure.block(index)?;
        let BlockKey::FusionTree(src_key) = block.key() else {
            return Err(OperationError::ExpectedFusionTreeBlock {
                tensor: "src",
                index,
            });
        };

        let (dst_key, coefficient) = match &operation {
            TreeTransformOperation::Permute {
                codomain_permutation,
                domain_permutation,
            } => unique_permute_tree_pair(rule, src_key, codomain_permutation, domain_permutation)?,
            TreeTransformOperation::Braid {
                codomain_permutation,
                domain_permutation,
                codomain_levels,
                domain_levels,
            } => unique_braid_tree_pair(
                rule,
                src_key,
                codomain_permutation,
                domain_permutation,
                codomain_levels,
                domain_levels,
            )?,
            TreeTransformOperation::Transpose {
                codomain_permutation,
                domain_permutation,
            } => {
                unique_transpose_tree_pair(rule, src_key, codomain_permutation, domain_permutation)?
            }
        };
        specs.push(
            TreeTransformGroupBlockSpec::single(
                src_key.group_key(),
                dst_key,
                src_key.clone(),
                coefficient,
            )
            .with_source_axes(source_axes.clone()),
        );
    }

    Ok(TreeTransformGroupPlan::new(specs))
}

fn validate_all_codomain_operation_scope(
    operation: &TreeTransformOperation,
) -> Result<(), OperationError> {
    let scope_error = || OperationError::UnsupportedTreeTransformScope {
        operation: operation.clone(),
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
            operation: operation.clone(),
            message: "all-codomain UniqueFusion lowering supports explicit Permute or Braid operations",
        }),
    }
}

fn operation_source_axes(operation: &TreeTransformOperation) -> Vec<usize> {
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
