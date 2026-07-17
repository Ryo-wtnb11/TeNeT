use core::ops::{Add, Mul};
use rustc_hash::{FxHashMap, FxHashSet};
use smallvec::SmallVec;
use std::sync::Arc;

use num_traits::Zero;
use tenet_core::{
    generic_braid_tree_pair, generic_permute_tree_pair, generic_transpose_tree_pair,
    multiplicity_free_braid_tree, multiplicity_free_braid_tree_block,
    multiplicity_free_braid_tree_pair, multiplicity_free_braid_tree_pair_block,
    multiplicity_free_permute_tree, multiplicity_free_permute_tree_block,
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
            operation: Box::new(operation),
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
            operation: Box::new(operation),
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
            operation: Box::new(operation),
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
            operation: Box::new(operation),
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

/// Shape-independent all-codomain rows. One plan compile records one hit or
/// miss per distinct memo key, even when both accepted empty-domain encodings
/// (`None` and `Some(vacuum)`) reference the same codomain tree.
type TransformRows<K, T> = Vec<(K, T)>;
type SharedTransformRows<K, T> = Arc<TransformRows<K, T>>;
type TransformBlockRows<K, T> = Vec<TransformRows<K, T>>;
type StagedSources<'a, K, T> = SmallVec<[(&'a K, Option<SharedTransformRows<K, T>>); 4]>;
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
    F: FnOnce(&[K]) -> Result<TransformBlockRows<K, T>, OperationError>,
{
    if sources.iter().all(|(_, rows)| rows.is_some()) {
        return Ok((sources, Vec::new()));
    }

    let missing_positions = sources
        .iter()
        .enumerate()
        .filter_map(|(position, (_, rows))| rows.is_none().then_some(position))
        .collect::<Vec<_>>();
    let missing_keys = missing_positions
        .iter()
        .map(|&position| sources[position].0.clone())
        .collect::<Vec<_>>();
    let batched = block_transform(&missing_keys)?;
    if batched.len() != missing_keys.len() {
        return Err(OperationError::CoefficientCountMismatch {
            expected: missing_keys.len(),
            actual: batched.len(),
        });
    }

    let mut computed = Vec::with_capacity(missing_keys.len());
    // Why not rebuild a key map: preflight makes staged sources unique, and
    // the block API preserves source order exactly as recoupling columns do.
    for ((position, key), rows) in missing_positions.into_iter().zip(missing_keys).zip(batched) {
        let rows = Arc::new(rows);
        sources[position].1 = Some(Arc::clone(&rows));
        computed.push((key, rows));
    }
    Ok((sources, computed))
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

pub(crate) fn transformed_all_codomain_rows_block<R>(
    rule: &R,
    operation: &TreeTransformOperation,
    codomain_trees: &[FusionTreeKey],
) -> Result<TransformBlockRows<FusionTreeKey, R::Scalar>, OperationError>
where
    R: MultiplicityFreeFusionSymbols,
    R::Scalar: Clone + Add<Output = R::Scalar> + Mul<Output = R::Scalar>,
{
    let rows = match operation {
        TreeTransformOperation::Permute {
            codomain_permutation,
            ..
        } => multiplicity_free_permute_tree_block(rule, codomain_trees, codomain_permutation),
        TreeTransformOperation::Braid {
            codomain_permutation,
            codomain_levels,
            ..
        } => multiplicity_free_braid_tree_block(
            rule,
            codomain_trees,
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
    build_multiplicity_free_all_codomain_tree_transform_group_plan_memoized_impl(
        rule,
        rule_key,
        operation,
        src_structure,
        memo,
        memo_hits,
        memo_misses,
        threads,
        transformed_all_codomain_rows_block,
    )
}

#[allow(clippy::too_many_arguments)]
fn build_multiplicity_free_all_codomain_tree_transform_group_plan_memoized_impl<R, RuleKey, F>(
    rule: &R,
    rule_key: &RuleKey,
    operation: TreeTransformOperation,
    src_structure: &BlockStructure,
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
            &R,
            &TreeTransformOperation,
            &[FusionTreeKey],
        ) -> Result<TransformBlockRows<FusionTreeKey, R::Scalar>, OperationError>
        + Send
        + Sync,
{
    if !rule.fusion_style().is_multiplicity_free() {
        return Err(OperationError::UnsupportedFusionStyle {
            operation: Box::new(operation),
            style: rule.fusion_style(),
        });
    }
    operation.validate_braiding_support(rule)?;
    validate_all_codomain_operation_scope(&operation)?;
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
            let block = src_structure.block(src_block_index)?;
            let BlockKey::FusionTree(src_key) = block.key() else {
                return Err(OperationError::ExpectedFusionTreeBlock {
                    tensor: "src",
                    index: src_block_index,
                });
            };
            validate_all_codomain_fusion_tree_block(rule, src_block_index, src_key)?;
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
            sources.push((codomain_tree, rows));
            source_indices.push(source_index);
        }
        staged_groups.push(StagedGroupRows {
            group,
            sources,
            source_alignment: StagedSourceAlignment::Explicit(source_indices),
        });
    }
    let completed = execute_staged_groups(staged_groups, threads, |staged| {
        let (resolved, computed) = resolve_staged_group_rows(staged.sources, |missing_keys| {
            block_transform(rule, &operation, missing_keys)
        })?;
        let StagedSourceAlignment::Explicit(source_indices) = staged.source_alignment else {
            return Err(OperationError::StructureMismatch {
                tensor: "staged all-codomain alignment",
            });
        };
        let mut source_cursor = 0usize;

        let mut rows_for = |codomain_tree: &FusionTreeKey| {
            let index = source_indices.get(source_cursor).copied();
            source_cursor += 1;
            index
                .and_then(|index| resolved.get(index))
                .filter(|(key, _)| *key == codomain_tree)
                .and_then(|(_, rows)| rows.as_ref())
                .map(Arc::clone)
                .ok_or(OperationError::StructureMismatch {
                    tensor: "staged all-codomain rows",
                })
        };
        let specs =
            if operation.is_identity_for(staged.group.group_key().codomain_uncoupled().len(), 0) {
                assemble_identity_all_codomain_group_specs(
                    rule,
                    src_structure,
                    &staged.group,
                    &source_axes,
                    &mut rows_for,
                )
            } else {
                assemble_all_codomain_group_specs(
                    rule,
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
    .with_source_axes(source_axes.to_vec())])
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
            operation: Box::new(operation),
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
/// Serial and parallel builds share the same staged group executor. Each
/// worker performs one whole-block row transform and assembles that group's
/// specs; memo entries and statistics are committed in source order only after
/// every group succeeds.
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
    build_multiplicity_free_tree_pair_transform_group_plan_memoized_impl(
        rule,
        rule_key,
        operation,
        src_structure,
        memo,
        memo_hits,
        memo_misses,
        threads,
        transformed_tree_pair_rows_block,
    )
}

#[allow(clippy::too_many_arguments)]
fn build_multiplicity_free_tree_pair_transform_group_plan_memoized_impl<R, RuleKey, F>(
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
    if !rule.fusion_style().is_multiplicity_free() {
        return Err(OperationError::UnsupportedFusionStyle {
            operation: Box::new(operation),
            style: rule.fusion_style(),
        });
    }
    operation.validate_braiding_support(rule)?;
    let source_axes = operation_source_axes(&operation);
    let groups = src_structure.fusion_tree_groups();

    let mut staged_hits = 0;
    let mut staged_misses = 0;
    let mut staged_groups = Vec::with_capacity(groups.len());
    for group in groups {
        let mut sources = StagedSources::new();
        for &src_block_index in group.block_indices() {
            let block = src_structure.block(src_block_index)?;
            let BlockKey::FusionTree(src_key) = block.key() else {
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
            sources.push((src_key, rows));
        }
        staged_groups.push(StagedGroupRows {
            group,
            sources,
            source_alignment: StagedSourceAlignment::Identity,
        });
    }
    let completed = execute_staged_groups(staged_groups, threads, |staged| {
        let (resolved, computed) = resolve_staged_group_rows(staged.sources, |missing_keys| {
            block_transform(rule, &operation, missing_keys)
        })?;
        if !matches!(staged.source_alignment, StagedSourceAlignment::Identity) {
            return Err(OperationError::StructureMismatch {
                tensor: "staged tree-pair alignment",
            });
        }
        let mut source_cursor = 0usize;

        let mut rows_for = |src_key: &FusionTreeBlockKey| {
            let index = source_cursor;
            source_cursor += 1;
            resolved
                .get(index)
                .filter(|(key, _)| *key == src_key)
                .and_then(|(_, rows)| rows.as_ref())
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
    build_multiplicity_free_tree_pair_transform_group_plan_memoized_impl(
        rule,
        rule_key,
        operation,
        src_structure,
        memo,
        memo_hits,
        memo_misses,
        threads,
        block_transform,
    )
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
            operation: Box::new(operation),
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
            operation: Box::new(operation),
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
