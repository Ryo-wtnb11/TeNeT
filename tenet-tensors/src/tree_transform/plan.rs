use core::ops::{Add, Mul};
use std::collections::HashMap;
use std::sync::Arc;

use num_traits::Zero;
use tenet_core::{
    multiplicity_free_braid_tree, multiplicity_free_braid_tree_pair,
    multiplicity_free_permute_tree, multiplicity_free_permute_tree_pair,
    multiplicity_free_transpose_tree_pair, BlockKey, BlockStructure, FusionRule,
    FusionTreeBlockGroup, FusionTreeBlockKey, FusionTreeGroupKey, MultiplicityFreeFusionSymbols,
    MultiplicityFreeRigidSymbols, TensorMap, TensorStorage,
};
#[cfg(test)]
use tenet_core::{
    unique_braid_tree, unique_braid_tree_pair, unique_permute_tree, unique_permute_tree_pair,
    unique_transpose_tree_pair, FusionStyleKind, MultiplicityFreePivotalSymbols,
};

use crate::{OperationError, TreeTransformStructure};

use super::helpers::{block_indices_for_keys, fusion_tree_group_block_keys};
use super::operation::TreeTransformOperationKey;

#[derive(Clone, Debug, PartialEq)]
pub struct TreeTransformBlockSpec<T> {
    pub(crate) dst_blocks: Vec<usize>,
    pub(crate) src_blocks: Vec<usize>,
    pub(crate) coefficients_src_by_dst: Vec<T>,
    pub(crate) source_axes: Option<Vec<usize>>,
}

impl<T> TreeTransformBlockSpec<T> {
    pub fn single(dst_block: usize, src_block: usize, coefficient: T) -> Self {
        Self {
            dst_blocks: vec![dst_block],
            src_blocks: vec![src_block],
            coefficients_src_by_dst: vec![coefficient],
            source_axes: None,
        }
    }

    pub fn multi(
        dst_blocks: Vec<usize>,
        src_blocks: Vec<usize>,
        coefficients_src_by_dst: Vec<T>,
    ) -> Self {
        Self {
            dst_blocks,
            src_blocks,
            coefficients_src_by_dst,
            source_axes: None,
        }
    }

    pub fn with_source_axes<I>(mut self, axes: I) -> Self
    where
        I: IntoIterator<Item = usize>,
    {
        self.source_axes = Some(axes.into_iter().collect());
        self
    }

    fn with_optional_source_axes(mut self, axes: Option<Vec<usize>>) -> Self {
        self.source_axes = axes;
        self
    }

    #[inline]
    pub fn dst_blocks(&self) -> &[usize] {
        &self.dst_blocks
    }

    #[inline]
    pub fn src_blocks(&self) -> &[usize] {
        &self.src_blocks
    }

    /// Recoupling matrix coefficients stored as `U[dst, src]` in row-major
    /// destination-by-source order: `coeff[src + dst * src_count]`.
    #[inline]
    pub fn coefficients_src_by_dst(&self) -> &[T] {
        &self.coefficients_src_by_dst
    }

    #[inline]
    pub fn source_axes(&self) -> Option<&[usize]> {
        self.source_axes.as_deref()
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct TreeTransformKeyBlockSpec<T> {
    dst_keys: Vec<BlockKey>,
    src_keys: Vec<BlockKey>,
    coefficients_src_by_dst: Vec<T>,
    source_axes: Option<Vec<usize>>,
}

impl<T> TreeTransformKeyBlockSpec<T> {
    pub fn single<KDst, KSrc>(dst_key: KDst, src_key: KSrc, coefficient: T) -> Self
    where
        KDst: Into<BlockKey>,
        KSrc: Into<BlockKey>,
    {
        Self {
            dst_keys: vec![dst_key.into()],
            src_keys: vec![src_key.into()],
            coefficients_src_by_dst: vec![coefficient],
            source_axes: None,
        }
    }

    pub fn multi<DstKeys, SrcKeys, KDst, KSrc>(
        dst_keys: DstKeys,
        src_keys: SrcKeys,
        coefficients_src_by_dst: Vec<T>,
    ) -> Self
    where
        DstKeys: IntoIterator<Item = KDst>,
        SrcKeys: IntoIterator<Item = KSrc>,
        KDst: Into<BlockKey>,
        KSrc: Into<BlockKey>,
    {
        Self {
            dst_keys: dst_keys.into_iter().map(Into::into).collect(),
            src_keys: src_keys.into_iter().map(Into::into).collect(),
            coefficients_src_by_dst,
            source_axes: None,
        }
    }

    pub fn with_source_axes<I>(mut self, axes: I) -> Self
    where
        I: IntoIterator<Item = usize>,
    {
        self.source_axes = Some(axes.into_iter().collect());
        self
    }

    #[inline]
    pub fn dst_keys(&self) -> &[BlockKey] {
        &self.dst_keys
    }

    #[inline]
    pub fn src_keys(&self) -> &[BlockKey] {
        &self.src_keys
    }

    /// Recoupling matrix coefficients stored as `U[dst, src]` in row-major
    /// destination-by-source order: `coeff[src + dst * src_count]`.
    #[inline]
    pub fn coefficients_src_by_dst(&self) -> &[T] {
        &self.coefficients_src_by_dst
    }
}

impl<T: Copy> TreeTransformKeyBlockSpec<T> {
    pub(crate) fn to_indexed_spec(
        &self,
        dst_structure: &BlockStructure,
        src_structure: &BlockStructure,
    ) -> Result<TreeTransformBlockSpec<T>, OperationError> {
        let dst_blocks = block_indices_for_keys(dst_structure, &self.dst_keys)?;
        let src_blocks = block_indices_for_keys(src_structure, &self.src_keys)?;

        Ok(TreeTransformBlockSpec::multi(
            dst_blocks,
            src_blocks,
            self.coefficients_src_by_dst.clone(),
        ))
        .map(|spec| spec.with_optional_source_axes(self.source_axes.clone()))
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct TreeTransformGroupBlockSpec<T> {
    group_key: FusionTreeGroupKey,
    dst_keys: Vec<BlockKey>,
    src_keys: Vec<BlockKey>,
    coefficients_src_by_dst: Vec<T>,
    source_axes: Option<Vec<usize>>,
}

impl<T> TreeTransformGroupBlockSpec<T> {
    pub fn single<KDst, KSrc>(
        group_key: FusionTreeGroupKey,
        dst_key: KDst,
        src_key: KSrc,
        coefficient: T,
    ) -> Self
    where
        KDst: Into<BlockKey>,
        KSrc: Into<BlockKey>,
    {
        Self {
            group_key,
            dst_keys: vec![dst_key.into()],
            src_keys: vec![src_key.into()],
            coefficients_src_by_dst: vec![coefficient],
            source_axes: None,
        }
    }

    pub fn multi<DstKeys, SrcKeys, KDst, KSrc>(
        group_key: FusionTreeGroupKey,
        dst_keys: DstKeys,
        src_keys: SrcKeys,
        coefficients_src_by_dst: Vec<T>,
    ) -> Self
    where
        DstKeys: IntoIterator<Item = KDst>,
        SrcKeys: IntoIterator<Item = KSrc>,
        KDst: Into<BlockKey>,
        KSrc: Into<BlockKey>,
    {
        Self {
            group_key,
            dst_keys: dst_keys.into_iter().map(Into::into).collect(),
            src_keys: src_keys.into_iter().map(Into::into).collect(),
            coefficients_src_by_dst,
            source_axes: None,
        }
    }

    pub fn with_source_axes<I>(mut self, axes: I) -> Self
    where
        I: IntoIterator<Item = usize>,
    {
        self.source_axes = Some(axes.into_iter().collect());
        self
    }

    pub fn from_block_groups(
        dst_structure: &BlockStructure,
        dst_group: &FusionTreeBlockGroup,
        src_structure: &BlockStructure,
        src_group: &FusionTreeBlockGroup,
        coefficients_src_by_dst: Vec<T>,
    ) -> Result<Self, OperationError> {
        let dst_keys = fusion_tree_group_block_keys(dst_structure, dst_group, "dst")?;
        let src_keys = fusion_tree_group_block_keys(src_structure, src_group, "src")?;
        let expected = dst_keys
            .len()
            .checked_mul(src_keys.len())
            .ok_or(OperationError::ElementCountOverflow)?;
        if coefficients_src_by_dst.len() != expected {
            return Err(OperationError::CoefficientCountMismatch {
                expected,
                actual: coefficients_src_by_dst.len(),
            });
        }
        Ok(Self::multi(
            src_group.group_key().clone(),
            dst_keys,
            src_keys,
            coefficients_src_by_dst,
        ))
    }

    #[inline]
    pub fn group_key(&self) -> &FusionTreeGroupKey {
        &self.group_key
    }

    #[inline]
    pub fn dst_keys(&self) -> &[BlockKey] {
        &self.dst_keys
    }

    #[inline]
    pub fn src_keys(&self) -> &[BlockKey] {
        &self.src_keys
    }

    /// Recoupling matrix coefficients stored as `U[dst, src]` in row-major
    /// destination-by-source order: `coeff[src + dst * src_count]`.
    #[inline]
    pub fn coefficients_src_by_dst(&self) -> &[T] {
        &self.coefficients_src_by_dst
    }
}

impl<T: Copy> TreeTransformGroupBlockSpec<T> {
    pub(crate) fn to_indexed_spec(
        &self,
        dst_structure: &BlockStructure,
        src_structure: &BlockStructure,
    ) -> Result<TreeTransformBlockSpec<T>, OperationError> {
        let dst_blocks = block_indices_for_keys(dst_structure, &self.dst_keys)?;
        let src_blocks = block_indices_for_keys(src_structure, &self.src_keys)?;

        Ok(TreeTransformBlockSpec::multi(
            dst_blocks,
            src_blocks,
            self.coefficients_src_by_dst.clone(),
        ))
        .map(|spec| spec.with_optional_source_axes(self.source_axes.clone()))
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct TreeTransformGroupPlan<T> {
    specs: Vec<TreeTransformGroupBlockSpec<T>>,
}

impl<T> TreeTransformGroupPlan<T> {
    pub fn new(specs: Vec<TreeTransformGroupBlockSpec<T>>) -> Self {
        Self { specs }
    }

    pub fn from_specs<I>(specs: I) -> Self
    where
        I: IntoIterator<Item = TreeTransformGroupBlockSpec<T>>,
    {
        Self::new(specs.into_iter().collect())
    }

    #[inline]
    pub fn specs(&self) -> &[TreeTransformGroupBlockSpec<T>] {
        &self.specs
    }

    pub fn into_specs(self) -> Vec<TreeTransformGroupBlockSpec<T>> {
        self.specs
    }
}

impl<T: Copy> TreeTransformGroupPlan<T> {
    pub fn compile<
        TDst,
        TSrc,
        const DST_NOUT: usize,
        const DST_NIN: usize,
        const SRC_NOUT: usize,
        const SRC_NIN: usize,
        SDst,
        SSrc,
        DDst,
        DSrc,
    >(
        &self,
        dst: &TensorMap<TDst, DST_NOUT, DST_NIN, SDst, DDst>,
        src: &TensorMap<TSrc, SRC_NOUT, SRC_NIN, SSrc, DSrc>,
    ) -> Result<TreeTransformStructure<T>, OperationError>
    where
        DDst: TensorStorage<TDst>,
        DSrc: TensorStorage<TSrc>,
    {
        TreeTransformStructure::compile_grouped(dst, src, &self.specs)
    }

    pub fn compile_structures(
        &self,
        dst_structure: &BlockStructure,
        src_structure: &BlockStructure,
    ) -> Result<TreeTransformStructure<T>, OperationError> {
        self.compile_structures_with_storage_conjugation(dst_structure, src_structure, false)
    }

    pub fn compile_structures_with_storage_conjugation(
        &self,
        dst_structure: &BlockStructure,
        src_structure: &BlockStructure,
        storage_conjugate: bool,
    ) -> Result<TreeTransformStructure<T>, OperationError> {
        TreeTransformStructure::compile_grouped_structures_with_storage_conjugation(
            dst_structure,
            src_structure,
            &self.specs,
            storage_conjugate,
        )
    }

    pub(crate) fn compile_shared_structures_with_storage_conjugation(
        &self,
        dst_structure: Arc<BlockStructure>,
        src_structure: Arc<BlockStructure>,
        storage_conjugate: bool,
    ) -> Result<TreeTransformStructure<T>, OperationError> {
        TreeTransformStructure::compile_grouped_shared_structures(
            dst_structure,
            src_structure,
            &self.specs,
            storage_conjugate,
        )
    }
}

/// Build a TensorKit-style grouped tree-transform plan for multiplicity-free
/// fusion rules.
///
/// This is the generic callback form: each source tree may map to multiple
/// destination trees, and duplicate destinations are accumulated into one
/// group-level recoupling matrix. `GenericFusion` with vertex multiplicities is
/// intentionally not represented by this scalar-coefficient API.
pub fn build_tree_transform_group_plan<T, R, F>(
    rule: &R,
    operation: TreeTransformOperationKey,
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
        let src_block_indices = group.block_indices();
        let mut src_keys = Vec::<BlockKey>::with_capacity(src_block_indices.len());
        let mut dst_keys = Vec::<BlockKey>::new();
        let mut dst_index_by_key = HashMap::<BlockKey, usize>::new();
        let mut rows = Vec::<Vec<T>>::new();

        for (src_column, &src_block_index) in src_block_indices.iter().enumerate() {
            let block = src_structure.block(src_block_index)?;
            let BlockKey::FusionTree(src_key) = block.key() else {
                return Err(OperationError::ExpectedFusionTreeBlock {
                    tensor: "src",
                    index: src_block_index,
                });
            };
            src_keys.push(BlockKey::from(src_key.clone()));

            for row in &mut rows {
                row.push(T::zero());
            }
            for (dst_tree_key, coefficient) in transform(src_key)? {
                let dst_key = BlockKey::from(dst_tree_key);
                let dst_row = if let Some(&dst_row) = dst_index_by_key.get(&dst_key) {
                    dst_row
                } else {
                    let dst_row = dst_keys.len();
                    dst_index_by_key.insert(dst_key.clone(), dst_row);
                    dst_keys.push(dst_key);
                    rows.push(vec![T::zero(); src_column + 1]);
                    dst_row
                };
                rows[dst_row][src_column] = rows[dst_row][src_column].clone() + coefficient;
            }
        }

        if dst_keys.is_empty() {
            return Err(OperationError::EmptyTransformBlock);
        }
        let src_count = src_keys.len();
        let mut coefficients_src_by_dst = Vec::with_capacity(dst_keys.len() * src_count);
        for row in rows {
            coefficients_src_by_dst.extend(row);
        }
        specs.push(
            TreeTransformGroupBlockSpec::multi(
                group.group_key().clone(),
                dst_keys,
                src_keys,
                coefficients_src_by_dst,
            )
            .with_source_axes(source_axes.clone()),
        );
    }

    Ok(TreeTransformGroupPlan::new(specs))
}

/// Standard all-codomain tree-transform builder for Unique and Simple
/// multiplicity-free rules.
pub fn build_all_codomain_tree_transform_group_plan<R>(
    rule: &R,
    operation: TreeTransformOperationKey,
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
    operation: TreeTransformOperationKey,
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
    operation: TreeTransformOperationKey,
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
    operation: TreeTransformOperationKey,
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
            TreeTransformOperationKey::Permute {
                codomain_permutation,
                ..
            } => unique_permute_tree(rule, src_key.codomain_tree(), codomain_permutation)?,
            TreeTransformOperationKey::Braid {
                codomain_permutation,
                codomain_levels,
                ..
            } => unique_braid_tree(
                rule,
                src_key.codomain_tree(),
                codomain_permutation,
                codomain_levels,
            )?,
            TreeTransformOperationKey::Transpose { .. } => {
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
    operation: TreeTransformOperationKey,
    src_structure: &BlockStructure,
) -> Result<TreeTransformGroupPlan<R::Scalar>, OperationError>
where
    R: MultiplicityFreeFusionSymbols,
    R::Scalar: Clone + Add<Output = R::Scalar> + Mul<Output = R::Scalar> + Zero,
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
        let src_block_indices = group.block_indices();
        let mut src_keys = Vec::<BlockKey>::with_capacity(src_block_indices.len());
        let mut dst_keys = Vec::<BlockKey>::new();
        let mut dst_index_by_key = HashMap::<BlockKey, usize>::new();
        let mut rows = Vec::<Vec<R::Scalar>>::new();

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

            let transformed = match &operation {
                TreeTransformOperationKey::Permute {
                    codomain_permutation,
                    ..
                } => multiplicity_free_permute_tree(
                    rule,
                    src_key.codomain_tree(),
                    codomain_permutation,
                )?,
                TreeTransformOperationKey::Braid {
                    codomain_permutation,
                    codomain_levels,
                    ..
                } => multiplicity_free_braid_tree(
                    rule,
                    src_key.codomain_tree(),
                    codomain_permutation,
                    codomain_levels,
                )?,
                TreeTransformOperationKey::Transpose { .. } => {
                    unreachable!("all-codomain operation scope validation rejected transpose")
                }
            };

            for row in &mut rows {
                row.push(R::Scalar::zero());
            }
            for (dst_codomain_tree, coefficient) in transformed {
                let dst_key = BlockKey::from(FusionTreeBlockKey::pair(
                    dst_codomain_tree,
                    src_key.domain_tree().clone(),
                ));
                let dst_row = if let Some(&dst_row) = dst_index_by_key.get(&dst_key) {
                    dst_row
                } else {
                    let dst_row = dst_keys.len();
                    dst_index_by_key.insert(dst_key.clone(), dst_row);
                    dst_keys.push(dst_key);
                    rows.push(vec![R::Scalar::zero(); src_column + 1]);
                    dst_row
                };
                rows[dst_row][src_column] = rows[dst_row][src_column].clone() + coefficient;
            }
        }

        let src_count = src_keys.len();
        let mut coefficients_src_by_dst = Vec::with_capacity(dst_keys.len() * src_count);
        for row in rows {
            coefficients_src_by_dst.extend(row);
        }
        specs.push(
            TreeTransformGroupBlockSpec::multi(
                group.group_key().clone(),
                dst_keys,
                src_keys,
                coefficients_src_by_dst,
            )
            .with_source_axes(source_axes.clone()),
        );
    }

    Ok(TreeTransformGroupPlan::new(specs))
}

pub(crate) fn build_multiplicity_free_tree_pair_transform_group_plan<R>(
    rule: &R,
    operation: TreeTransformOperationKey,
    src_structure: &BlockStructure,
) -> Result<TreeTransformGroupPlan<R::Scalar>, OperationError>
where
    R: MultiplicityFreeRigidSymbols,
    R::Scalar: Clone + Add<Output = R::Scalar> + Mul<Output = R::Scalar> + Zero,
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
        let src_block_indices = group.block_indices();
        let mut src_keys = Vec::<BlockKey>::with_capacity(src_block_indices.len());
        let mut dst_keys = Vec::<BlockKey>::new();
        let mut dst_index_by_key = HashMap::<BlockKey, usize>::new();
        let mut rows = Vec::<Vec<R::Scalar>>::new();

        for (src_column, &src_block_index) in src_block_indices.iter().enumerate() {
            let block = src_structure.block(src_block_index)?;
            let BlockKey::FusionTree(src_key) = block.key() else {
                return Err(OperationError::ExpectedFusionTreeBlock {
                    tensor: "src",
                    index: src_block_index,
                });
            };
            src_keys.push(BlockKey::from(src_key.clone()));

            let transformed = match &operation {
                TreeTransformOperationKey::Permute {
                    codomain_permutation,
                    domain_permutation,
                } => multiplicity_free_permute_tree_pair(
                    rule,
                    src_key,
                    codomain_permutation,
                    domain_permutation,
                )?,
                TreeTransformOperationKey::Braid {
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
                )?,
                TreeTransformOperationKey::Transpose {
                    codomain_permutation,
                    domain_permutation,
                } => multiplicity_free_transpose_tree_pair(
                    rule,
                    src_key,
                    codomain_permutation,
                    domain_permutation,
                )?,
            };

            for row in &mut rows {
                row.push(R::Scalar::zero());
            }
            for (dst_tree_key, coefficient) in transformed {
                let dst_key = BlockKey::from(dst_tree_key);
                let dst_row = if let Some(&dst_row) = dst_index_by_key.get(&dst_key) {
                    dst_row
                } else {
                    let dst_row = dst_keys.len();
                    dst_index_by_key.insert(dst_key.clone(), dst_row);
                    dst_keys.push(dst_key);
                    rows.push(vec![R::Scalar::zero(); src_column + 1]);
                    dst_row
                };
                rows[dst_row][src_column] = rows[dst_row][src_column].clone() + coefficient;
            }
        }

        let src_count = src_keys.len();
        let mut coefficients_src_by_dst = Vec::with_capacity(dst_keys.len() * src_count);
        for row in rows {
            coefficients_src_by_dst.extend(row);
        }
        specs.push(
            TreeTransformGroupBlockSpec::multi(
                group.group_key().clone(),
                dst_keys,
                src_keys,
                coefficients_src_by_dst,
            )
            .with_source_axes(source_axes.clone()),
        );
    }

    Ok(TreeTransformGroupPlan::new(specs))
}

#[cfg(test)]
pub(crate) fn build_unique_tree_pair_transform_group_plan<R>(
    rule: &R,
    operation: TreeTransformOperationKey,
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
            TreeTransformOperationKey::Permute {
                codomain_permutation,
                domain_permutation,
            } => unique_permute_tree_pair(rule, src_key, codomain_permutation, domain_permutation)?,
            TreeTransformOperationKey::Braid {
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
            TreeTransformOperationKey::Transpose {
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
    operation: &TreeTransformOperationKey,
) -> Result<(), OperationError> {
    let scope_error = || OperationError::UnsupportedTreeTransformScope {
        operation: operation.clone(),
        message: "all-codomain UniqueFusion lowering requires an empty domain operation",
    };

    match operation {
        TreeTransformOperationKey::Permute {
            domain_permutation,
            ..
        } if domain_permutation.is_empty() => Ok(()),
        TreeTransformOperationKey::Braid {
            domain_permutation,
            domain_levels,
            ..
        } if domain_permutation.is_empty() && domain_levels.is_empty() => Ok(()),
        TreeTransformOperationKey::Permute { .. } | TreeTransformOperationKey::Braid { .. } => {
            Err(scope_error())
        }
        TreeTransformOperationKey::Transpose { .. } => Err(OperationError::UnsupportedTreeTransformScope {
            operation: operation.clone(),
            message: "all-codomain UniqueFusion lowering supports explicit Permute or Braid operations",
        }),
    }
}

fn operation_source_axes(operation: &TreeTransformOperationKey) -> Vec<usize> {
    match operation {
        TreeTransformOperationKey::Permute {
            codomain_permutation,
            domain_permutation,
        }
        | TreeTransformOperationKey::Braid {
            codomain_permutation,
            domain_permutation,
            ..
        }
        | TreeTransformOperationKey::Transpose {
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
