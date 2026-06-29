#![forbid(unsafe_code)]

//! TensorOperations-style lowering for TeNeT.
//!
//! Public/core tensor code talks in terms of TeNeT-owned block views. This crate
//! lowers those views to strided-rs kernels at the same granularity that
//! TensorKit uses Strided.jl/StridedViews.jl internally.

use core::fmt;
use core::ops::{Add, Mul};
use std::collections::HashMap;
use std::sync::Arc;

use num_complex::{Complex32, Complex64};
use num_traits::{One, Zero};
use tenet_core::{
    BlockKey, BlockLayout, BlockStructure, BlockView, BlockViewMut, CoreError,
    FusionTreeBlockGroup, FusionTreeGroupKey, TensorMap,
};
use tenet_dense::{
    DefaultDenseExecutor, DenseError, DenseExecutor, DenseRead, DenseView, DenseViewMut, DenseWrite,
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AxisPermutation<'a> {
    Identity,
    Axes(&'a [usize]),
}

impl<'a> AxisPermutation<'a> {
    #[inline]
    pub fn identity() -> Self {
        Self::Identity
    }

    #[inline]
    pub fn from_axes(axes: &'a [usize]) -> Self {
        Self::Axes(axes)
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TensorAddStructure {
    rank: usize,
    axes: Vec<usize>,
    terms: Vec<TensorAddStructureTerm>,
    descriptor: TensorAddDescriptor,
    dst_structure: Arc<BlockStructure>,
    src_structure: Arc<BlockStructure>,
}

pub fn tensoradd_structure<TDst, TSrc, const NOUT: usize, const NIN: usize, SDst, SSrc>(
    dst: &TensorMap<TDst, NOUT, NIN, SDst>,
    src: &TensorMap<TSrc, NOUT, NIN, SSrc>,
    permutation: AxisPermutation<'_>,
) -> Result<TensorAddStructure, OperationError> {
    TensorAddStructure::compile(dst, src, permutation)
}

impl TensorAddStructure {
    pub fn compile<TDst, TSrc, const NOUT: usize, const NIN: usize, SDst, SSrc>(
        dst: &TensorMap<TDst, NOUT, NIN, SDst>,
        src: &TensorMap<TSrc, NOUT, NIN, SSrc>,
        permutation: AxisPermutation<'_>,
    ) -> Result<Self, OperationError> {
        Self::compile_shared_structures(
            Arc::clone(dst.structure()),
            Arc::clone(src.structure()),
            permutation,
        )
    }

    pub fn compile_structures(
        dst_structure: &BlockStructure,
        src_structure: &BlockStructure,
        permutation: AxisPermutation<'_>,
    ) -> Result<Self, OperationError> {
        Self::compile_shared_structures(
            Arc::new(dst_structure.clone()),
            Arc::new(src_structure.clone()),
            permutation,
        )
    }

    fn compile_shared_structures(
        dst_structure: Arc<BlockStructure>,
        src_structure: Arc<BlockStructure>,
        permutation: AxisPermutation<'_>,
    ) -> Result<Self, OperationError> {
        if dst_structure.block_count() != src_structure.block_count() {
            return Err(OperationError::BlockCountMismatch {
                dst: dst_structure.block_count(),
                src: src_structure.block_count(),
            });
        }

        let rank = dst_structure.rank();
        if src_structure.rank() != rank {
            return Err(OperationError::StructureRankMismatch {
                expected: rank,
                actual: src_structure.rank(),
            });
        }
        let axes = permutation_axes(permutation, rank)?;
        let src_for_dst = dst_structure
            .pair_block_indices_from(&src_structure)
            .map_err(OperationError::from_core_preserving_context)?;
        let mut terms = Vec::with_capacity(dst_structure.block_count());

        for dst_index in 0..dst_structure.block_count() {
            let dst_block = dst_structure.block(dst_index)?;
            if dst_block.shape().len() != rank {
                return Err(OperationError::RankMismatch {
                    expected: rank,
                    actual: dst_block.shape().len(),
                });
            }
            let src_index = src_for_dst[dst_index];
            let src_block = src_structure.block(src_index)?;
            if src_block.shape().len() != rank {
                return Err(OperationError::RankMismatch {
                    expected: rank,
                    actual: src_block.shape().len(),
                });
            }

            terms.push(TensorAddStructureTerm {
                key: dst_block.key().clone(),
                dst_block: dst_index,
                src_block: src_index,
            });
        }

        let descriptor =
            TensorAddDescriptor::compile(rank, &axes, &terms, &dst_structure, &src_structure)?;

        Ok(Self {
            rank,
            axes,
            terms,
            descriptor,
            dst_structure,
            src_structure,
        })
    }

    #[inline]
    pub fn rank(&self) -> usize {
        self.rank
    }

    #[inline]
    pub fn axes(&self) -> &[usize] {
        &self.axes
    }

    #[inline]
    pub fn terms(&self) -> &[TensorAddStructureTerm] {
        &self.terms
    }

    #[inline]
    fn descriptor(&self) -> &TensorAddDescriptor {
        &self.descriptor
    }

    fn validate_replay_structures(
        &self,
        dst_structure: &Arc<BlockStructure>,
        src_structure: &Arc<BlockStructure>,
    ) -> Result<(), OperationError> {
        validate_structure_identity("dst", &self.dst_structure, dst_structure)?;
        validate_structure_identity("src", &self.src_structure, src_structure)
    }

    pub fn execute_with<B, T, const NOUT: usize, const NIN: usize, S>(
        &self,
        backend: &mut B,
        allocator: &mut B::Allocator,
        dst: &mut TensorMap<T, NOUT, NIN, S>,
        src: &TensorMap<T, NOUT, NIN, S>,
        alpha: T,
        beta: T,
    ) -> Result<(), OperationError>
    where
        B: TensorOperationsBackend,
        T: Copy
            + Add<T, Output = T>
            + Mul<T, Output = T>
            + PartialEq
            + Zero
            + One
            + strided_kernel::MaybeSendSync,
    {
        backend.tensoradd_structure_into(allocator, self, dst, src, alpha, beta)
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TensorAddStructureTerm {
    key: BlockKey,
    dst_block: usize,
    src_block: usize,
}

impl TensorAddStructureTerm {
    #[inline]
    pub fn key(&self) -> &BlockKey {
        &self.key
    }

    #[inline]
    pub fn dst_block(&self) -> usize {
        self.dst_block
    }

    #[inline]
    pub fn src_block(&self) -> usize {
        self.src_block
    }
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub(crate) struct TensorAddDescriptor {
    terms: Vec<TensorAddDescriptorTerm>,
    shapes: Vec<usize>,
    dst_strides: Vec<isize>,
    src_strides: Vec<isize>,
}

impl TensorAddDescriptor {
    #[inline]
    pub fn terms(&self) -> &[TensorAddDescriptorTerm] {
        &self.terms
    }

    fn reserve(&mut self, term_count: usize, rank: usize) {
        self.terms.reserve(term_count);
        let entry_count = term_count.saturating_mul(rank);
        self.shapes.reserve(entry_count);
        self.dst_strides.reserve(entry_count);
        self.src_strides.reserve(entry_count);
    }

    fn shape(&self, term: &TensorAddDescriptorTerm) -> &[usize] {
        &self.shapes[term.layout_start..term.layout_start + term.rank]
    }

    fn dst_strides(&self, term: &TensorAddDescriptorTerm) -> &[isize] {
        &self.dst_strides[term.layout_start..term.layout_start + term.rank]
    }

    fn src_strides(&self, term: &TensorAddDescriptorTerm) -> &[isize] {
        &self.src_strides[term.layout_start..term.layout_start + term.rank]
    }

    fn compile(
        rank: usize,
        axes: &[usize],
        terms: &[TensorAddStructureTerm],
        dst_structure: &BlockStructure,
        src_structure: &BlockStructure,
    ) -> Result<Self, OperationError> {
        let mut descriptor = Self::default();
        descriptor.reserve(terms.len(), rank);

        for term in terms {
            let dst_block = dst_structure.block(term.dst_block())?;
            let src_block = src_structure.block(term.src_block())?;
            if dst_block.shape().len() != rank {
                return Err(OperationError::RankMismatch {
                    expected: rank,
                    actual: dst_block.shape().len(),
                });
            }
            if src_block.shape().len() != rank {
                return Err(OperationError::RankMismatch {
                    expected: rank,
                    actual: src_block.shape().len(),
                });
            }

            let layout_start = descriptor.shapes.len();
            for (dst_axis, &src_axis) in axes.iter().enumerate() {
                let dst_dim = dst_block.shape()[dst_axis];
                let src_dim = src_block.shape()[src_axis];
                if dst_dim != src_dim {
                    let src_shape = axes
                        .iter()
                        .map(|&axis| src_block.shape()[axis])
                        .collect::<Vec<_>>();
                    return Err(OperationError::ShapeMismatch {
                        dst: dst_block.shape().to_vec(),
                        src: src_shape,
                    });
                }
                descriptor.shapes.push(dst_dim);
                descriptor.dst_strides.push(
                    isize::try_from(dst_block.strides()[dst_axis]).map_err(|_| {
                        OperationError::StrideOverflow {
                            value: dst_block.strides()[dst_axis],
                        }
                    })?,
                );
                descriptor.src_strides.push(
                    isize::try_from(src_block.strides()[src_axis]).map_err(|_| {
                        OperationError::StrideOverflow {
                            value: src_block.strides()[src_axis],
                        }
                    })?,
                );
            }

            descriptor.terms.push(TensorAddDescriptorTerm {
                dst_block: term.dst_block(),
                src_block: term.src_block(),
                layout_start,
                rank,
                dst_offset: offset_to_isize(dst_block.offset())?,
                src_offset: offset_to_isize(src_block.offset())?,
            });
        }

        Ok(descriptor)
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct TensorAddDescriptorTerm {
    dst_block: usize,
    src_block: usize,
    layout_start: usize,
    rank: usize,
    dst_offset: isize,
    src_offset: isize,
}

#[derive(Clone, Debug, PartialEq)]
pub struct TreeTransformBlockSpec<T> {
    dst_blocks: Vec<usize>,
    src_blocks: Vec<usize>,
    coefficients_src_by_dst: Vec<T>,
}

impl<T> TreeTransformBlockSpec<T> {
    pub fn single(dst_block: usize, src_block: usize, coefficient: T) -> Self {
        Self {
            dst_blocks: vec![dst_block],
            src_blocks: vec![src_block],
            coefficients_src_by_dst: vec![coefficient],
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
        }
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
}

#[derive(Clone, Debug, PartialEq)]
pub struct TreeTransformKeyBlockSpec<T> {
    dst_keys: Vec<BlockKey>,
    src_keys: Vec<BlockKey>,
    coefficients_src_by_dst: Vec<T>,
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
        }
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
    fn to_indexed_spec(
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
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct TreeTransformGroupBlockSpec<T> {
    group_key: FusionTreeGroupKey,
    dst_keys: Vec<BlockKey>,
    src_keys: Vec<BlockKey>,
    coefficients_src_by_dst: Vec<T>,
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
        }
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
    fn to_indexed_spec(
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
    pub fn compile<TDst, TSrc, const NOUT: usize, const NIN: usize, SDst, SSrc>(
        &self,
        dst: &TensorMap<TDst, NOUT, NIN, SDst>,
        src: &TensorMap<TSrc, NOUT, NIN, SSrc>,
    ) -> Result<TreeTransformStructure<T>, OperationError> {
        TreeTransformStructure::compile_grouped(dst, src, &self.specs)
    }

    pub fn compile_structures(
        &self,
        dst_structure: &BlockStructure,
        src_structure: &BlockStructure,
    ) -> Result<TreeTransformStructure<T>, OperationError> {
        TreeTransformStructure::compile_grouped_structures(
            dst_structure,
            src_structure,
            &self.specs,
        )
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub enum TreeTransformOperationKey {
    Transpose {
        codomain_permutation: Vec<usize>,
        domain_permutation: Vec<usize>,
    },
    Braid {
        codomain_permutation: Vec<usize>,
        domain_permutation: Vec<usize>,
        codomain_levels: Vec<usize>,
        domain_levels: Vec<usize>,
    },
}

impl TreeTransformOperationKey {
    pub fn transpose<Codomain, Domain>(
        codomain_permutation: Codomain,
        domain_permutation: Domain,
    ) -> Self
    where
        Codomain: IntoIterator<Item = usize>,
        Domain: IntoIterator<Item = usize>,
    {
        Self::Transpose {
            codomain_permutation: codomain_permutation.into_iter().collect(),
            domain_permutation: domain_permutation.into_iter().collect(),
        }
    }

    pub fn braid<Codomain, Domain, CodomainLevels, DomainLevels>(
        codomain_permutation: Codomain,
        domain_permutation: Domain,
        codomain_levels: CodomainLevels,
        domain_levels: DomainLevels,
    ) -> Self
    where
        Codomain: IntoIterator<Item = usize>,
        Domain: IntoIterator<Item = usize>,
        CodomainLevels: IntoIterator<Item = usize>,
        DomainLevels: IntoIterator<Item = usize>,
    {
        Self::Braid {
            codomain_permutation: codomain_permutation.into_iter().collect(),
            domain_permutation: domain_permutation.into_iter().collect(),
            codomain_levels: codomain_levels.into_iter().collect(),
            domain_levels: domain_levels.into_iter().collect(),
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub struct TreeTransformGroupPlanKey {
    operation: TreeTransformOperationKey,
    groups: Vec<TreeTransformCachedGroupKey>,
}

impl TreeTransformGroupPlanKey {
    pub fn new<Groups>(operation: TreeTransformOperationKey, groups: Groups) -> Self
    where
        Groups: IntoIterator<Item = TreeTransformCachedGroupKey>,
    {
        Self {
            operation,
            groups: groups.into_iter().collect(),
        }
    }

    pub fn from_plan<T>(
        operation: TreeTransformOperationKey,
        plan: &TreeTransformGroupPlan<T>,
    ) -> Self {
        Self::new(
            operation,
            plan.specs()
                .iter()
                .map(TreeTransformCachedGroupKey::from_spec),
        )
    }

    #[inline]
    pub fn operation(&self) -> &TreeTransformOperationKey {
        &self.operation
    }

    #[inline]
    pub fn groups(&self) -> &[TreeTransformCachedGroupKey] {
        &self.groups
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub struct TreeTransformCachedGroupKey {
    group_key: FusionTreeGroupKey,
    dst_keys: Vec<BlockKey>,
    src_keys: Vec<BlockKey>,
}

impl TreeTransformCachedGroupKey {
    pub fn new<DstKeys, SrcKeys, KDst, KSrc>(
        group_key: FusionTreeGroupKey,
        dst_keys: DstKeys,
        src_keys: SrcKeys,
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
        }
    }

    pub fn from_spec<T>(spec: &TreeTransformGroupBlockSpec<T>) -> Self {
        Self {
            group_key: spec.group_key().clone(),
            dst_keys: spec.dst_keys().to_vec(),
            src_keys: spec.src_keys().to_vec(),
        }
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
}

#[derive(Clone, Debug)]
pub struct TreeTransformGroupPlanCache<T> {
    plans: HashMap<TreeTransformGroupPlanKey, TreeTransformGroupPlan<T>>,
}

impl<T> Default for TreeTransformGroupPlanCache<T> {
    fn default() -> Self {
        Self {
            plans: HashMap::new(),
        }
    }
}

impl<T> TreeTransformGroupPlanCache<T> {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn len(&self) -> usize {
        self.plans.len()
    }

    pub fn is_empty(&self) -> bool {
        self.plans.is_empty()
    }

    pub fn get(&self, key: &TreeTransformGroupPlanKey) -> Option<&TreeTransformGroupPlan<T>> {
        self.plans.get(key)
    }

    pub fn insert(
        &mut self,
        key: TreeTransformGroupPlanKey,
        plan: TreeTransformGroupPlan<T>,
    ) -> Option<TreeTransformGroupPlan<T>> {
        self.plans.insert(key, plan)
    }

    pub fn get_or_insert_with<F>(
        &mut self,
        key: TreeTransformGroupPlanKey,
        build: F,
    ) -> &TreeTransformGroupPlan<T>
    where
        F: FnOnce() -> TreeTransformGroupPlan<T>,
    {
        self.plans.entry(key).or_insert_with(build)
    }
}

fn fusion_tree_group_block_keys(
    structure: &BlockStructure,
    group: &FusionTreeBlockGroup,
    tensor: &'static str,
) -> Result<Vec<BlockKey>, OperationError> {
    let mut keys = Vec::with_capacity(group.block_indices().len());
    for &index in group.block_indices() {
        let block = structure.block(index).map_err(|err| match err {
            CoreError::BlockIndexOutOfBounds { index, count } => {
                OperationError::BlockIndexOutOfBounds {
                    tensor,
                    index,
                    count,
                }
            }
            other => OperationError::Core(other),
        })?;
        match block.key().fusion_tree_group_key() {
            Some(actual) if &actual == group.group_key() => keys.push(block.key().clone()),
            _ => return Err(OperationError::FusionTreeGroupMismatch { tensor, index }),
        }
    }
    Ok(keys)
}

fn block_indices_for_keys(
    structure: &BlockStructure,
    keys: &[BlockKey],
) -> Result<Vec<usize>, OperationError> {
    keys.iter()
        .map(|key| {
            structure
                .find_block_index_by_key(key)
                .ok_or_else(|| OperationError::MissingBlockKey { key: key.clone() })
        })
        .collect()
}

#[derive(Clone, Debug, PartialEq)]
pub struct TreeTransformStructure<T> {
    rank: usize,
    blocks: Vec<TreeTransformBlock>,
    layouts: TreeTransformLayoutTable,
    coefficients_src_by_dst: Vec<T>,
    dst_structure: Arc<BlockStructure>,
    src_structure: Arc<BlockStructure>,
}

impl<T: Copy> TreeTransformStructure<T> {
    pub fn compile<TDst, TSrc, const NOUT: usize, const NIN: usize, SDst, SSrc>(
        dst: &TensorMap<TDst, NOUT, NIN, SDst>,
        src: &TensorMap<TSrc, NOUT, NIN, SSrc>,
        specs: &[TreeTransformBlockSpec<T>],
    ) -> Result<Self, OperationError> {
        Self::compile_shared_structures(
            Arc::clone(dst.structure()),
            Arc::clone(src.structure()),
            specs,
        )
    }

    pub fn compile_structures(
        dst_structure: &BlockStructure,
        src_structure: &BlockStructure,
        specs: &[TreeTransformBlockSpec<T>],
    ) -> Result<Self, OperationError> {
        Self::compile_shared_structures(
            Arc::new(dst_structure.clone()),
            Arc::new(src_structure.clone()),
            specs,
        )
    }

    pub fn compile_keyed<TDst, TSrc, const NOUT: usize, const NIN: usize, SDst, SSrc>(
        dst: &TensorMap<TDst, NOUT, NIN, SDst>,
        src: &TensorMap<TSrc, NOUT, NIN, SSrc>,
        specs: &[TreeTransformKeyBlockSpec<T>],
    ) -> Result<Self, OperationError> {
        Self::compile_keyed_shared_structures(
            Arc::clone(dst.structure()),
            Arc::clone(src.structure()),
            specs,
        )
    }

    pub fn compile_keyed_structures(
        dst_structure: &BlockStructure,
        src_structure: &BlockStructure,
        specs: &[TreeTransformKeyBlockSpec<T>],
    ) -> Result<Self, OperationError> {
        Self::compile_keyed_shared_structures(
            Arc::new(dst_structure.clone()),
            Arc::new(src_structure.clone()),
            specs,
        )
    }

    pub fn compile_grouped<TDst, TSrc, const NOUT: usize, const NIN: usize, SDst, SSrc>(
        dst: &TensorMap<TDst, NOUT, NIN, SDst>,
        src: &TensorMap<TSrc, NOUT, NIN, SSrc>,
        specs: &[TreeTransformGroupBlockSpec<T>],
    ) -> Result<Self, OperationError> {
        Self::compile_grouped_shared_structures(
            Arc::clone(dst.structure()),
            Arc::clone(src.structure()),
            specs,
        )
    }

    pub fn compile_grouped_structures(
        dst_structure: &BlockStructure,
        src_structure: &BlockStructure,
        specs: &[TreeTransformGroupBlockSpec<T>],
    ) -> Result<Self, OperationError> {
        Self::compile_grouped_shared_structures(
            Arc::new(dst_structure.clone()),
            Arc::new(src_structure.clone()),
            specs,
        )
    }

    fn compile_grouped_shared_structures(
        dst_structure: Arc<BlockStructure>,
        src_structure: Arc<BlockStructure>,
        specs: &[TreeTransformGroupBlockSpec<T>],
    ) -> Result<Self, OperationError> {
        let indexed_specs = specs
            .iter()
            .map(|spec| spec.to_indexed_spec(&dst_structure, &src_structure))
            .collect::<Result<Vec<_>, _>>()?;
        Self::compile_shared_structures(dst_structure, src_structure, &indexed_specs)
    }

    fn compile_keyed_shared_structures(
        dst_structure: Arc<BlockStructure>,
        src_structure: Arc<BlockStructure>,
        specs: &[TreeTransformKeyBlockSpec<T>],
    ) -> Result<Self, OperationError> {
        let indexed_specs = specs
            .iter()
            .map(|spec| spec.to_indexed_spec(&dst_structure, &src_structure))
            .collect::<Result<Vec<_>, _>>()?;
        Self::compile_shared_structures(dst_structure, src_structure, &indexed_specs)
    }

    fn compile_shared_structures(
        dst_structure: Arc<BlockStructure>,
        src_structure: Arc<BlockStructure>,
        specs: &[TreeTransformBlockSpec<T>],
    ) -> Result<Self, OperationError> {
        let rank = dst_structure.rank();
        if src_structure.rank() != rank {
            return Err(OperationError::StructureRankMismatch {
                expected: rank,
                actual: src_structure.rank(),
            });
        }

        let mut layouts = TreeTransformLayoutTable::default();
        let mut blocks = Vec::with_capacity(specs.len());
        let mut coefficients_src_by_dst = Vec::new();
        let mut touched_dst_blocks = vec![false; dst_structure.block_count()];

        for spec in specs {
            if spec.dst_blocks.is_empty() || spec.src_blocks.is_empty() {
                return Err(OperationError::EmptyTransformBlock);
            }
            let src_count = spec.src_blocks.len();
            let dst_count = spec.dst_blocks.len();
            let expected_coefficients = src_count
                .checked_mul(dst_count)
                .ok_or(OperationError::ElementCountOverflow)?;
            if spec.coefficients_src_by_dst.len() != expected_coefficients {
                return Err(OperationError::CoefficientCountMismatch {
                    expected: expected_coefficients,
                    actual: spec.coefficients_src_by_dst.len(),
                });
            }

            for &dst_block in &spec.dst_blocks {
                let touched = touched_dst_blocks.get_mut(dst_block).ok_or(
                    OperationError::BlockIndexOutOfBounds {
                        tensor: "dst",
                        index: dst_block,
                        count: dst_structure.block_count(),
                    },
                )?;
                if *touched {
                    return Err(OperationError::DuplicateTransformDestination { dst_block });
                }
                *touched = true;
            }

            let dst_layout_start = layouts.entry_count();
            let mut element_count = None;
            for &dst_block in &spec.dst_blocks {
                let block = dst_structure.block(dst_block)?;
                let layout_element_count =
                    layouts.push_block(rank, block.shape(), block.strides(), block.offset())?;
                match element_count {
                    Some(expected) if expected != layout_element_count => {
                        return Err(OperationError::ElementCountMismatch {
                            expected,
                            actual: layout_element_count,
                        });
                    }
                    Some(_) => {}
                    None => element_count = Some(layout_element_count),
                }
            }

            let src_layout_start = layouts.entry_count();
            for &src_block in &spec.src_blocks {
                let block = src_structure.block(src_block)?;
                let layout_element_count =
                    layouts.push_block(rank, block.shape(), block.strides(), block.offset())?;
                match element_count {
                    Some(expected) if expected != layout_element_count => {
                        return Err(OperationError::ElementCountMismatch {
                            expected,
                            actual: layout_element_count,
                        });
                    }
                    Some(_) => {}
                    None => element_count = Some(layout_element_count),
                }
            }
            let element_count = element_count.expect("validated non-empty block");
            let coefficient_start = coefficients_src_by_dst.len();
            coefficients_src_by_dst.extend_from_slice(&spec.coefficients_src_by_dst);

            if src_count == 1 && dst_count == 1 {
                let dst_layout = layouts.entry(dst_layout_start);
                let src_layout = layouts.entry(src_layout_start);
                if layouts.shape(dst_layout) != layouts.shape(src_layout) {
                    return Err(OperationError::ShapeMismatch {
                        dst: layouts.shape(dst_layout).to_vec(),
                        src: layouts.shape(src_layout).to_vec(),
                    });
                }
                blocks.push(TreeTransformBlock::Single {
                    dst_layout: dst_layout_start,
                    src_layout: src_layout_start,
                    coefficient: coefficient_start,
                });
            } else {
                blocks.push(TreeTransformBlock::Multi {
                    dst_layout_start,
                    dst_count,
                    src_layout_start,
                    src_count,
                    coefficient_start,
                    element_count,
                });
            }
        }

        Ok(Self {
            rank,
            blocks,
            layouts,
            coefficients_src_by_dst,
            dst_structure,
            src_structure,
        })
    }

    #[inline]
    pub fn rank(&self) -> usize {
        self.rank
    }

    #[inline]
    pub fn block_count(&self) -> usize {
        self.blocks.len()
    }

    pub fn workspace_lens(&self) -> (usize, usize) {
        self.blocks
            .iter()
            .fold((0, 0), |(max_src, max_dst), block| match block {
                TreeTransformBlock::Single { .. } => (max_src, max_dst),
                TreeTransformBlock::Multi {
                    dst_count,
                    src_count,
                    element_count,
                    ..
                } => (
                    max_src.max(element_count.saturating_mul(*src_count)),
                    max_dst.max(element_count.saturating_mul(*dst_count)),
                ),
            })
    }

    pub fn workspace_len(&self) -> usize {
        let (source, destination) = self.workspace_lens();
        source.max(destination)
    }

    pub fn has_pack_gemm_scatter_blocks(&self) -> bool {
        self.blocks
            .iter()
            .any(|block| matches!(block, TreeTransformBlock::Multi { .. }))
    }

    fn coefficient(&self, index: usize) -> T {
        self.coefficients_src_by_dst[index]
    }

    fn validate_replay_structures(
        &self,
        dst_structure: &Arc<BlockStructure>,
        src_structure: &Arc<BlockStructure>,
    ) -> Result<(), OperationError> {
        validate_structure_identity("dst", &self.dst_structure, dst_structure)?;
        validate_structure_identity("src", &self.src_structure, src_structure)
    }
}

#[derive(Clone, Debug, PartialEq)]
enum TreeTransformBlock {
    Single {
        dst_layout: usize,
        src_layout: usize,
        coefficient: usize,
    },
    Multi {
        dst_layout_start: usize,
        dst_count: usize,
        src_layout_start: usize,
        src_count: usize,
        coefficient_start: usize,
        element_count: usize,
    },
}

#[derive(Clone, Debug, Default, PartialEq)]
struct TreeTransformLayoutTable {
    entries: Vec<TreeTransformLayout>,
    shapes: Vec<usize>,
    strides: Vec<isize>,
    packed_strides: Vec<isize>,
}

impl TreeTransformLayoutTable {
    fn entry_count(&self) -> usize {
        self.entries.len()
    }

    fn entry(&self, index: usize) -> &TreeTransformLayout {
        &self.entries[index]
    }

    fn shape(&self, layout: &TreeTransformLayout) -> &[usize] {
        &self.shapes[layout.layout_start..layout.layout_start + layout.rank]
    }

    fn strides(&self, layout: &TreeTransformLayout) -> &[isize] {
        &self.strides[layout.layout_start..layout.layout_start + layout.rank]
    }

    fn packed_strides(&self, layout: &TreeTransformLayout) -> &[isize] {
        &self.packed_strides[layout.layout_start..layout.layout_start + layout.rank]
    }

    fn push_block(
        &mut self,
        rank: usize,
        shape: &[usize],
        strides: &[usize],
        offset: usize,
    ) -> Result<usize, OperationError> {
        if shape.len() != rank {
            return Err(OperationError::RankMismatch {
                expected: rank,
                actual: shape.len(),
            });
        }
        if strides.len() != rank {
            return Err(OperationError::RankMismatch {
                expected: rank,
                actual: strides.len(),
            });
        }
        let element_count = element_count(shape)?;
        let layout_start = self.shapes.len();
        let packed_strides = column_major_strides_isize(shape)?;
        self.shapes.extend_from_slice(shape);
        self.strides.extend(
            strides
                .iter()
                .map(|&stride| {
                    isize::try_from(stride)
                        .map_err(|_| OperationError::StrideOverflow { value: stride })
                })
                .collect::<Result<Vec<_>, _>>()?,
        );
        self.packed_strides.extend_from_slice(&packed_strides);
        self.entries.push(TreeTransformLayout {
            layout_start,
            rank,
            offset: offset_to_isize(offset)?,
            element_count,
        });
        Ok(element_count)
    }
}

#[derive(Clone, Debug, PartialEq)]
struct TreeTransformLayout {
    layout_start: usize,
    rank: usize,
    offset: isize,
    element_count: usize,
}

#[derive(Clone, Debug)]
pub struct TreeTransformWorkspace<T> {
    zero_strides: Vec<isize>,
    source: Vec<T>,
    destination: Vec<T>,
}

impl<T> Default for TreeTransformWorkspace<T> {
    fn default() -> Self {
        Self {
            zero_strides: Vec::new(),
            source: Vec::new(),
            destination: Vec::new(),
        }
    }
}

impl<T> TreeTransformWorkspace<T> {
    pub fn source_len(&self) -> usize {
        self.source.len()
    }

    pub fn destination_len(&self) -> usize {
        self.destination.len()
    }
}

pub trait TreeTransformBackend<T>
where
    T: Copy
        + Add<T, Output = T>
        + Mul<T, Output = T>
        + PartialEq
        + Zero
        + One
        + strided_kernel::MaybeSendSync,
{
    type Workspace;

    fn tree_transform_structure_into<const NOUT: usize, const NIN: usize, S>(
        &mut self,
        workspace: &mut Self::Workspace,
        structure: &TreeTransformStructure<T>,
        dst: &mut TensorMap<T, NOUT, NIN, S>,
        src: &TensorMap<T, NOUT, NIN, S>,
        alpha: T,
        beta: T,
    ) -> Result<(), OperationError>;
}

pub trait TensorOperationsBackend {
    type Allocator;

    fn copy_block_into<T>(
        &mut self,
        allocator: &mut Self::Allocator,
        dst: BlockViewMut<'_, T>,
        src: BlockView<'_, T>,
    ) -> Result<(), OperationError>
    where
        T: Copy + strided_kernel::MaybeSendSync;

    fn tensoradd_structure_into<T, const NOUT: usize, const NIN: usize, S>(
        &mut self,
        allocator: &mut Self::Allocator,
        structure: &TensorAddStructure,
        dst: &mut TensorMap<T, NOUT, NIN, S>,
        src: &TensorMap<T, NOUT, NIN, S>,
        alpha: T,
        beta: T,
    ) -> Result<(), OperationError>
    where
        T: Copy
            + Add<T, Output = T>
            + Mul<T, Output = T>
            + PartialEq
            + Zero
            + One
            + strided_kernel::MaybeSendSync;
}

#[derive(Clone, Debug, Default)]
pub struct HostAllocator {
    zero_strides: Vec<isize>,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct HostTensorOperations;

#[derive(Debug)]
pub struct DenseTreeTransformOperations<E = DefaultDenseExecutor> {
    dense: E,
}

impl DenseTreeTransformOperations<DefaultDenseExecutor> {
    pub fn default_executor() -> Self {
        Self {
            dense: DefaultDenseExecutor::new(),
        }
    }
}

impl<E> DenseTreeTransformOperations<E> {
    pub fn new(dense: E) -> Self {
        Self { dense }
    }

    pub fn dense(&self) -> &E {
        &self.dense
    }

    pub fn dense_mut(&mut self) -> &mut E {
        &mut self.dense
    }
}

impl Default for DenseTreeTransformOperations<DefaultDenseExecutor> {
    fn default() -> Self {
        Self::default_executor()
    }
}

impl TensorOperationsBackend for HostTensorOperations {
    type Allocator = HostAllocator;

    fn copy_block_into<T>(
        &mut self,
        _allocator: &mut Self::Allocator,
        dst: BlockViewMut<'_, T>,
        src: BlockView<'_, T>,
    ) -> Result<(), OperationError>
    where
        T: Copy + strided_kernel::MaybeSendSync,
    {
        copy_block_with_strided_kernel(dst, src)
    }

    fn tensoradd_structure_into<T, const NOUT: usize, const NIN: usize, S>(
        &mut self,
        allocator: &mut Self::Allocator,
        structure: &TensorAddStructure,
        dst: &mut TensorMap<T, NOUT, NIN, S>,
        src: &TensorMap<T, NOUT, NIN, S>,
        alpha: T,
        beta: T,
    ) -> Result<(), OperationError>
    where
        T: Copy
            + Add<T, Output = T>
            + Mul<T, Output = T>
            + PartialEq
            + Zero
            + One
            + strided_kernel::MaybeSendSync,
    {
        tensoradd_structure_with_strided_kernel(allocator, structure, dst, src, alpha, beta)
    }
}

impl<T> TreeTransformBackend<T> for HostTensorOperations
where
    T: Copy
        + Add<T, Output = T>
        + Mul<T, Output = T>
        + PartialEq
        + Zero
        + One
        + strided_kernel::MaybeSendSync,
{
    type Workspace = TreeTransformWorkspace<T>;

    fn tree_transform_structure_into<const NOUT: usize, const NIN: usize, S>(
        &mut self,
        workspace: &mut Self::Workspace,
        structure: &TreeTransformStructure<T>,
        dst: &mut TensorMap<T, NOUT, NIN, S>,
        src: &TensorMap<T, NOUT, NIN, S>,
        alpha: T,
        beta: T,
    ) -> Result<(), OperationError> {
        tree_transform_structure_with_strided_kernel(workspace, structure, dst, src, alpha, beta)
    }
}

#[doc(hidden)]
pub trait DenseRecouplingScalar:
    Copy
    + Add<Self, Output = Self>
    + Mul<Self, Output = Self>
    + PartialEq
    + Zero
    + One
    + strided_kernel::MaybeSendSync
    + 'static
{
    fn dense_read(view: DenseView<'_, Self>) -> DenseRead<'_>;
    fn dense_write(view: DenseViewMut<'_, Self>) -> DenseWrite<'_>;
}

macro_rules! impl_dense_recoupling_scalar {
    ($ty:ty, $read_variant:ident, $write_variant:ident) => {
        impl DenseRecouplingScalar for $ty {
            fn dense_read(view: DenseView<'_, Self>) -> DenseRead<'_> {
                DenseRead::$read_variant(view)
            }

            fn dense_write(view: DenseViewMut<'_, Self>) -> DenseWrite<'_> {
                DenseWrite::$write_variant(view)
            }
        }
    };
}

impl_dense_recoupling_scalar!(f32, F32, F32);
impl_dense_recoupling_scalar!(f64, F64, F64);
impl_dense_recoupling_scalar!(Complex32, C32, C32);
impl_dense_recoupling_scalar!(Complex64, C64, C64);

impl<E, T> TreeTransformBackend<T> for DenseTreeTransformOperations<E>
where
    E: DenseExecutor,
    T: DenseRecouplingScalar,
{
    type Workspace = TreeTransformWorkspace<T>;

    fn tree_transform_structure_into<const NOUT: usize, const NIN: usize, S>(
        &mut self,
        workspace: &mut Self::Workspace,
        structure: &TreeTransformStructure<T>,
        dst: &mut TensorMap<T, NOUT, NIN, S>,
        src: &TensorMap<T, NOUT, NIN, S>,
        alpha: T,
        beta: T,
    ) -> Result<(), OperationError> {
        tree_transform_structure_with_dense_recoupling(
            &mut self.dense,
            workspace,
            structure,
            dst,
            src,
            alpha,
            beta,
        )
    }
}

pub fn tensorcopy_into<T, const NOUT: usize, const NIN: usize, S>(
    dst: &mut TensorMap<T, NOUT, NIN, S>,
    src: &TensorMap<T, NOUT, NIN, S>,
) -> Result<(), OperationError>
where
    T: Copy + strided_kernel::MaybeSendSync,
{
    let mut backend = HostTensorOperations;
    let mut allocator = HostAllocator::default();
    tensorcopy_into_with(&mut backend, &mut allocator, dst, src)
}

pub fn tensorcopy_into_with<B, T, const NOUT: usize, const NIN: usize, S>(
    backend: &mut B,
    allocator: &mut B::Allocator,
    dst: &mut TensorMap<T, NOUT, NIN, S>,
    src: &TensorMap<T, NOUT, NIN, S>,
) -> Result<(), OperationError>
where
    B: TensorOperationsBackend,
    T: Copy + strided_kernel::MaybeSendSync,
{
    backend.copy_block_into(allocator, dst.subblock_mut()?, src.subblock()?)
}

pub fn tensoradd_into<T, const NOUT: usize, const NIN: usize, S>(
    dst: &mut TensorMap<T, NOUT, NIN, S>,
    src: &TensorMap<T, NOUT, NIN, S>,
    permutation: AxisPermutation<'_>,
    alpha: T,
    beta: T,
) -> Result<(), OperationError>
where
    T: Copy
        + Add<T, Output = T>
        + Mul<T, Output = T>
        + PartialEq
        + Zero
        + One
        + strided_kernel::MaybeSendSync,
{
    let mut backend = HostTensorOperations;
    let mut allocator = HostAllocator::default();
    tensoradd_into_with(
        &mut backend,
        &mut allocator,
        dst,
        src,
        permutation,
        alpha,
        beta,
    )
}

pub fn tensoradd_into_with<B, T, const NOUT: usize, const NIN: usize, S>(
    backend: &mut B,
    allocator: &mut B::Allocator,
    dst: &mut TensorMap<T, NOUT, NIN, S>,
    src: &TensorMap<T, NOUT, NIN, S>,
    permutation: AxisPermutation<'_>,
    alpha: T,
    beta: T,
) -> Result<(), OperationError>
where
    B: TensorOperationsBackend,
    T: Copy
        + Add<T, Output = T>
        + Mul<T, Output = T>
        + PartialEq
        + Zero
        + One
        + strided_kernel::MaybeSendSync,
{
    let structure = tensoradd_structure(dst, src, permutation)?;
    tensoradd_execute_with(backend, allocator, &structure, dst, src, alpha, beta)
}

pub fn tensoradd_execute_with<B, T, const NOUT: usize, const NIN: usize, S>(
    backend: &mut B,
    allocator: &mut B::Allocator,
    structure: &TensorAddStructure,
    dst: &mut TensorMap<T, NOUT, NIN, S>,
    src: &TensorMap<T, NOUT, NIN, S>,
    alpha: T,
    beta: T,
) -> Result<(), OperationError>
where
    B: TensorOperationsBackend,
    T: Copy
        + Add<T, Output = T>
        + Mul<T, Output = T>
        + PartialEq
        + Zero
        + One
        + strided_kernel::MaybeSendSync,
{
    structure.execute_with(backend, allocator, dst, src, alpha, beta)
}

pub fn tree_transform_execute_with<B, T, const NOUT: usize, const NIN: usize, S>(
    backend: &mut B,
    workspace: &mut B::Workspace,
    structure: &TreeTransformStructure<T>,
    dst: &mut TensorMap<T, NOUT, NIN, S>,
    src: &TensorMap<T, NOUT, NIN, S>,
    alpha: T,
    beta: T,
) -> Result<(), OperationError>
where
    B: TreeTransformBackend<T>,
    T: Copy
        + Add<T, Output = T>
        + Mul<T, Output = T>
        + PartialEq
        + Zero
        + One
        + strided_kernel::MaybeSendSync,
{
    backend.tree_transform_structure_into(workspace, structure, dst, src, alpha, beta)
}

pub fn tensoradd_assign_into<T, const NOUT: usize, const NIN: usize, S>(
    dst: &mut TensorMap<T, NOUT, NIN, S>,
    src: &TensorMap<T, NOUT, NIN, S>,
    alpha: T,
) -> Result<(), OperationError>
where
    T: Copy
        + Add<T, Output = T>
        + Mul<T, Output = T>
        + PartialEq
        + Zero
        + One
        + strided_kernel::MaybeSendSync,
{
    tensoradd_into(dst, src, AxisPermutation::identity(), alpha, T::zero())
}

pub fn tensoradd_add_into<T, const NOUT: usize, const NIN: usize, S>(
    dst: &mut TensorMap<T, NOUT, NIN, S>,
    src: &TensorMap<T, NOUT, NIN, S>,
    alpha: T,
) -> Result<(), OperationError>
where
    T: Copy
        + Add<T, Output = T>
        + Mul<T, Output = T>
        + PartialEq
        + Zero
        + One
        + strided_kernel::MaybeSendSync,
{
    tensoradd_into(dst, src, AxisPermutation::identity(), alpha, T::one())
}

pub fn copy_into<T>(dst: BlockViewMut<'_, T>, src: BlockView<'_, T>) -> Result<(), OperationError>
where
    T: Copy + strided_kernel::MaybeSendSync,
{
    let mut backend = HostTensorOperations;
    let mut allocator = HostAllocator::default();
    backend.copy_block_into(&mut allocator, dst, src)
}

pub fn scaled_assign_into<T>(
    dst: BlockViewMut<'_, T>,
    src: BlockView<'_, T>,
    alpha: T,
) -> Result<(), OperationError>
where
    T: Copy
        + Add<T, Output = T>
        + Mul<T, Output = T>
        + PartialEq
        + Zero
        + One
        + strided_kernel::MaybeSendSync,
{
    let mut allocator = HostAllocator::default();
    tensoradd_block_with_strided_kernel(&mut allocator, dst, src, alpha, T::zero())
}

pub fn scaled_add_into<T>(
    dst: BlockViewMut<'_, T>,
    src: BlockView<'_, T>,
    alpha: T,
) -> Result<(), OperationError>
where
    T: Copy
        + Add<T, Output = T>
        + Mul<T, Output = T>
        + PartialEq
        + Zero
        + One
        + strided_kernel::MaybeSendSync,
{
    let mut allocator = HostAllocator::default();
    tensoradd_block_with_strided_kernel(&mut allocator, dst, src, alpha, T::one())
}

fn copy_block_with_strided_kernel<T>(
    dst: BlockViewMut<'_, T>,
    src: BlockView<'_, T>,
) -> Result<(), OperationError>
where
    T: Copy + strided_kernel::MaybeSendSync,
{
    let mut dst = strided_write(dst)?;
    let src = strided_read(src)?;
    strided_kernel::copy_into(&mut dst, &src).map_err(strided_error)
}

fn tensoradd_structure_with_strided_kernel<T, const NOUT: usize, const NIN: usize, S>(
    allocator: &mut HostAllocator,
    structure: &TensorAddStructure,
    dst: &mut TensorMap<T, NOUT, NIN, S>,
    src: &TensorMap<T, NOUT, NIN, S>,
    alpha: T,
    beta: T,
) -> Result<(), OperationError>
where
    T: Copy
        + Add<T, Output = T>
        + Mul<T, Output = T>
        + PartialEq
        + Zero
        + One
        + strided_kernel::MaybeSendSync,
{
    let descriptor = structure.descriptor();
    structure.validate_replay_structures(dst.structure(), src.structure())?;
    if dst.structure().block_count() != descriptor.terms().len() {
        return Err(OperationError::BlockCountMismatch {
            dst: dst.structure().block_count(),
            src: descriptor.terms().len(),
        });
    }
    if src.structure().block_count() != descriptor.terms().len() {
        return Err(OperationError::BlockCountMismatch {
            dst: descriptor.terms().len(),
            src: src.structure().block_count(),
        });
    }

    let zero_strides = &mut allocator.zero_strides;
    let dst_data = dst.data_mut();
    let src_data = src.data();
    for term in descriptor.terms() {
        tensoradd_prepared_block_with_strided_kernel(
            zero_strides,
            descriptor,
            term,
            dst_data,
            src_data,
            alpha,
            beta,
        )?;
    }
    Ok(())
}

fn tree_transform_structure_with_strided_kernel<T, const NOUT: usize, const NIN: usize, S>(
    workspace: &mut TreeTransformWorkspace<T>,
    structure: &TreeTransformStructure<T>,
    dst: &mut TensorMap<T, NOUT, NIN, S>,
    src: &TensorMap<T, NOUT, NIN, S>,
    alpha: T,
    beta: T,
) -> Result<(), OperationError>
where
    T: Copy
        + Add<T, Output = T>
        + Mul<T, Output = T>
        + PartialEq
        + Zero
        + One
        + strided_kernel::MaybeSendSync,
{
    structure.validate_replay_structures(dst.structure(), src.structure())?;
    let dst_data = dst.data_mut();
    let src_data = src.data();

    for block in &structure.blocks {
        match *block {
            TreeTransformBlock::Single {
                dst_layout,
                src_layout,
                coefficient,
            } => tree_transform_single_with_strided_kernel(
                &mut workspace.zero_strides,
                &structure.layouts,
                structure.layouts.entry(dst_layout),
                structure.layouts.entry(src_layout),
                structure.coefficient(coefficient),
                dst_data,
                src_data,
                alpha,
                beta,
            )?,
            TreeTransformBlock::Multi {
                dst_layout_start,
                dst_count,
                src_layout_start,
                src_count,
                coefficient_start,
                element_count,
            } => tree_transform_multi_with_pack_gemm_scatter(
                workspace,
                &structure.layouts,
                dst_layout_start,
                dst_count,
                src_layout_start,
                src_count,
                coefficient_start,
                element_count,
                &structure.coefficients_src_by_dst,
                dst_data,
                src_data,
                alpha,
                beta,
            )?,
        }
    }
    Ok(())
}

fn tree_transform_structure_with_dense_recoupling<E, T, const NOUT: usize, const NIN: usize, S>(
    dense: &mut E,
    workspace: &mut TreeTransformWorkspace<T>,
    structure: &TreeTransformStructure<T>,
    dst: &mut TensorMap<T, NOUT, NIN, S>,
    src: &TensorMap<T, NOUT, NIN, S>,
    alpha: T,
    beta: T,
) -> Result<(), OperationError>
where
    E: DenseExecutor,
    T: DenseRecouplingScalar,
{
    structure.validate_replay_structures(dst.structure(), src.structure())?;
    let dst_data = dst.data_mut();
    let src_data = src.data();

    for block in &structure.blocks {
        match *block {
            TreeTransformBlock::Single {
                dst_layout,
                src_layout,
                coefficient,
            } => tree_transform_single_with_strided_kernel(
                &mut workspace.zero_strides,
                &structure.layouts,
                structure.layouts.entry(dst_layout),
                structure.layouts.entry(src_layout),
                structure.coefficient(coefficient),
                dst_data,
                src_data,
                alpha,
                beta,
            )?,
            TreeTransformBlock::Multi {
                dst_layout_start,
                dst_count,
                src_layout_start,
                src_count,
                coefficient_start,
                element_count,
            } => tree_transform_multi_with_dense_recoupling(
                dense,
                workspace,
                &structure.layouts,
                dst_layout_start,
                dst_count,
                src_layout_start,
                src_count,
                coefficient_start,
                element_count,
                &structure.coefficients_src_by_dst,
                dst_data,
                src_data,
                alpha,
                beta,
            )?,
        }
    }
    Ok(())
}

fn tensoradd_block_with_strided_kernel<T>(
    allocator: &mut HostAllocator,
    dst: BlockViewMut<'_, T>,
    src: BlockView<'_, T>,
    alpha: T,
    beta: T,
) -> Result<(), OperationError>
where
    T: Copy
        + Add<T, Output = T>
        + Mul<T, Output = T>
        + PartialEq
        + Zero
        + One
        + strided_kernel::MaybeSendSync,
{
    let mut dst = strided_write(dst)?;
    let src = strided_read(src)?;

    if dst.dims() != src.dims() {
        return Err(OperationError::ShapeMismatch {
            dst: dst.dims().to_vec(),
            src: src.dims().to_vec(),
        });
    }

    if beta.is_zero() {
        strided_kernel::copy_scale(&mut dst, &src, alpha).map_err(strided_error)
    } else {
        if !beta.is_one() {
            scale_destination(&mut allocator.zero_strides, &mut dst, beta)?;
        }
        strided_kernel::axpy(&mut dst, &src, alpha).map_err(strided_error)
    }
}

fn tensoradd_prepared_block_with_strided_kernel<T>(
    zero_strides: &mut Vec<isize>,
    descriptor: &TensorAddDescriptor,
    term: &TensorAddDescriptorTerm,
    dst_data: &mut [T],
    src_data: &[T],
    alpha: T,
    beta: T,
) -> Result<(), OperationError>
where
    T: Copy
        + Add<T, Output = T>
        + Mul<T, Output = T>
        + PartialEq
        + Zero
        + One
        + strided_kernel::MaybeSendSync,
{
    let shape = descriptor.shape(term);
    let dst_strides = descriptor.dst_strides(term);
    let src_strides = descriptor.src_strides(term);
    let mut dst =
        strided_kernel::StridedViewMut::new(dst_data, shape, dst_strides, term.dst_offset)
            .map_err(strided_error)?;
    let src = strided_kernel::StridedView::<T>::new(src_data, shape, src_strides, term.src_offset)
        .map_err(strided_error)?;

    if beta.is_zero() {
        strided_kernel::copy_scale(&mut dst, &src, alpha).map_err(strided_error)
    } else {
        if !beta.is_one() {
            scale_destination(zero_strides, &mut dst, beta)?;
        }
        strided_kernel::axpy(&mut dst, &src, alpha).map_err(strided_error)
    }
}

fn tree_transform_single_with_strided_kernel<T>(
    zero_strides: &mut Vec<isize>,
    layouts: &TreeTransformLayoutTable,
    dst_layout: &TreeTransformLayout,
    src_layout: &TreeTransformLayout,
    coefficient: T,
    dst_data: &mut [T],
    src_data: &[T],
    alpha: T,
    beta: T,
) -> Result<(), OperationError>
where
    T: Copy
        + Add<T, Output = T>
        + Mul<T, Output = T>
        + PartialEq
        + Zero
        + One
        + strided_kernel::MaybeSendSync,
{
    let shape = layouts.shape(dst_layout);
    let mut dst = strided_kernel::StridedViewMut::new(
        dst_data,
        shape,
        layouts.strides(dst_layout),
        dst_layout.offset,
    )
    .map_err(strided_error)?;
    let src = strided_kernel::StridedView::<T>::new(
        src_data,
        shape,
        layouts.strides(src_layout),
        src_layout.offset,
    )
    .map_err(strided_error)?;
    let scale = alpha * coefficient;
    if beta.is_zero() {
        strided_kernel::copy_scale(&mut dst, &src, scale).map_err(strided_error)
    } else {
        if !beta.is_one() {
            scale_destination(zero_strides, &mut dst, beta)?;
        }
        strided_kernel::axpy(&mut dst, &src, scale).map_err(strided_error)
    }
}

#[allow(clippy::too_many_arguments)]
fn tree_transform_multi_with_pack_gemm_scatter<T>(
    workspace: &mut TreeTransformWorkspace<T>,
    layouts: &TreeTransformLayoutTable,
    dst_layout_start: usize,
    dst_count: usize,
    src_layout_start: usize,
    src_count: usize,
    coefficient_start: usize,
    element_count: usize,
    coefficients_src_by_dst: &[T],
    dst_data: &mut [T],
    src_data: &[T],
    alpha: T,
    beta: T,
) -> Result<(), OperationError>
where
    T: Copy
        + Add<T, Output = T>
        + Mul<T, Output = T>
        + PartialEq
        + Zero
        + One
        + strided_kernel::MaybeSendSync,
{
    let source_len = element_count
        .checked_mul(src_count)
        .ok_or(OperationError::ElementCountOverflow)?;
    let destination_len = element_count
        .checked_mul(dst_count)
        .ok_or(OperationError::ElementCountOverflow)?;
    workspace.source.resize(source_len, T::zero());
    workspace.destination.resize(destination_len, T::zero());

    for src_index in 0..src_count {
        let layout = layouts.entry(src_layout_start + src_index);
        pack_layout_into_column(
            layouts,
            layout,
            src_data,
            &mut workspace.source,
            src_index * element_count,
        )?;
    }

    apply_recoupling_matrix_src_times_u_transpose(
        &mut workspace.destination,
        &workspace.source,
        coefficients_src_by_dst,
        coefficient_start,
        element_count,
        src_count,
        dst_count,
    )?;

    for dst_index in 0..dst_count {
        let layout = layouts.entry(dst_layout_start + dst_index);
        scatter_column_into_layout(
            &mut workspace.zero_strides,
            layouts,
            layout,
            &workspace.destination,
            dst_index * element_count,
            dst_data,
            alpha,
            beta,
        )?;
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn tree_transform_multi_with_dense_recoupling<E, T>(
    dense: &mut E,
    workspace: &mut TreeTransformWorkspace<T>,
    layouts: &TreeTransformLayoutTable,
    dst_layout_start: usize,
    dst_count: usize,
    src_layout_start: usize,
    src_count: usize,
    coefficient_start: usize,
    element_count: usize,
    coefficients_src_by_dst: &[T],
    dst_data: &mut [T],
    src_data: &[T],
    alpha: T,
    beta: T,
) -> Result<(), OperationError>
where
    E: DenseExecutor,
    T: DenseRecouplingScalar,
{
    let source_len = element_count
        .checked_mul(src_count)
        .ok_or(OperationError::ElementCountOverflow)?;
    let destination_len = element_count
        .checked_mul(dst_count)
        .ok_or(OperationError::ElementCountOverflow)?;
    workspace.source.resize(source_len, T::zero());
    workspace.destination.resize(destination_len, T::zero());

    for src_index in 0..src_count {
        let layout = layouts.entry(src_layout_start + src_index);
        pack_layout_into_column(
            layouts,
            layout,
            src_data,
            &mut workspace.source,
            src_index * element_count,
        )?;
    }

    apply_recoupling_matrix_with_dense_executor(
        dense,
        &mut workspace.destination,
        &workspace.source,
        coefficients_src_by_dst,
        coefficient_start,
        element_count,
        src_count,
        dst_count,
    )?;

    for dst_index in 0..dst_count {
        let layout = layouts.entry(dst_layout_start + dst_index);
        scatter_column_into_layout(
            &mut workspace.zero_strides,
            layouts,
            layout,
            &workspace.destination,
            dst_index * element_count,
            dst_data,
            alpha,
            beta,
        )?;
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn apply_recoupling_matrix_src_times_u_transpose<T>(
    destination: &mut [T],
    source: &[T],
    coefficients_src_by_dst: &[T],
    coefficient_start: usize,
    element_count: usize,
    src_count: usize,
    dst_count: usize,
) -> Result<(), OperationError>
where
    T: Copy + Add<T, Output = T> + Mul<T, Output = T> + Zero,
{
    let source_len = element_count
        .checked_mul(src_count)
        .ok_or(OperationError::ElementCountOverflow)?;
    let destination_len = element_count
        .checked_mul(dst_count)
        .ok_or(OperationError::ElementCountOverflow)?;
    let coefficient_count = src_count
        .checked_mul(dst_count)
        .ok_or(OperationError::ElementCountOverflow)?;
    let coefficient_end = coefficient_start
        .checked_add(coefficient_count)
        .ok_or(OperationError::ElementCountOverflow)?;

    if source.len() != source_len {
        return Err(OperationError::ElementCountMismatch {
            expected: source_len,
            actual: source.len(),
        });
    }
    if destination.len() != destination_len {
        return Err(OperationError::ElementCountMismatch {
            expected: destination_len,
            actual: destination.len(),
        });
    }
    if coefficients_src_by_dst.len() < coefficient_end {
        return Err(OperationError::CoefficientCountMismatch {
            expected: coefficient_end,
            actual: coefficients_src_by_dst.len(),
        });
    }

    // TensorKit's dense-vector GenericTreeTransformer uses `U[dst, src]` and
    // computes `buffer_dst = buffer_src * transpose(U)` after packing source
    // trees as columns. Keep this as the backend-replaceable boundary.
    for dst_index in 0..dst_count {
        let dst_column_start = dst_index * element_count;
        let coefficient_row_start = coefficient_start + dst_index * src_count;
        for element in 0..element_count {
            let mut sum = T::zero();
            for src_index in 0..src_count {
                let coeff = coefficients_src_by_dst[coefficient_row_start + src_index];
                let src_value = source[element + src_index * element_count];
                sum = sum + src_value * coeff;
            }
            destination[dst_column_start + element] = sum;
        }
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn apply_recoupling_matrix_with_dense_executor<E, T>(
    dense: &mut E,
    destination: &mut [T],
    source: &[T],
    coefficients_src_by_dst: &[T],
    coefficient_start: usize,
    element_count: usize,
    src_count: usize,
    dst_count: usize,
) -> Result<(), OperationError>
where
    E: DenseExecutor,
    T: DenseRecouplingScalar,
{
    let source_len = element_count
        .checked_mul(src_count)
        .ok_or(OperationError::ElementCountOverflow)?;
    let destination_len = element_count
        .checked_mul(dst_count)
        .ok_or(OperationError::ElementCountOverflow)?;
    let coefficient_count = src_count
        .checked_mul(dst_count)
        .ok_or(OperationError::ElementCountOverflow)?;
    let coefficient_end = coefficient_start
        .checked_add(coefficient_count)
        .ok_or(OperationError::ElementCountOverflow)?;

    if source.len() != source_len {
        return Err(OperationError::ElementCountMismatch {
            expected: source_len,
            actual: source.len(),
        });
    }
    if destination.len() != destination_len {
        return Err(OperationError::ElementCountMismatch {
            expected: destination_len,
            actual: destination.len(),
        });
    }
    if coefficients_src_by_dst.len() < coefficient_end {
        return Err(OperationError::CoefficientCountMismatch {
            expected: coefficient_end,
            actual: coefficients_src_by_dst.len(),
        });
    }

    let source_shape = [element_count, src_count];
    let source_strides = [1, element_count];
    let coefficient_shape = [src_count, dst_count];
    let coefficient_strides = [1, src_count];
    let destination_shape = [element_count, dst_count];
    let destination_strides = [1, element_count];

    let lhs = T::dense_read(
        DenseView::new(source, &source_shape, &source_strides, 0).map_err(OperationError::Dense)?,
    );
    let rhs = T::dense_read(
        DenseView::new(
            coefficients_src_by_dst,
            &coefficient_shape,
            &coefficient_strides,
            coefficient_start,
        )
        .map_err(OperationError::Dense)?,
    );
    let output = T::dense_write(
        DenseViewMut::new(destination, &destination_shape, &destination_strides, 0)
            .map_err(OperationError::Dense)?,
    );
    dense
        .matmul_into(output, lhs, rhs)
        .map_err(OperationError::Dense)
}

fn pack_layout_into_column<T>(
    layouts: &TreeTransformLayoutTable,
    layout: &TreeTransformLayout,
    src_data: &[T],
    packed: &mut [T],
    packed_offset: usize,
) -> Result<(), OperationError>
where
    T: Copy + strided_kernel::MaybeSendSync,
{
    let shape = layouts.shape(layout);
    let mut dst = strided_kernel::StridedViewMut::new(
        packed,
        shape,
        layouts.packed_strides(layout),
        offset_to_isize(packed_offset)?,
    )
    .map_err(strided_error)?;
    let src = strided_kernel::StridedView::<T>::new(
        src_data,
        shape,
        layouts.strides(layout),
        layout.offset,
    )
    .map_err(strided_error)?;
    strided_kernel::copy_into(&mut dst, &src).map_err(strided_error)
}

fn scatter_column_into_layout<T>(
    zero_strides: &mut Vec<isize>,
    layouts: &TreeTransformLayoutTable,
    layout: &TreeTransformLayout,
    packed: &[T],
    packed_offset: usize,
    dst_data: &mut [T],
    alpha: T,
    beta: T,
) -> Result<(), OperationError>
where
    T: Copy
        + Add<T, Output = T>
        + Mul<T, Output = T>
        + PartialEq
        + Zero
        + One
        + strided_kernel::MaybeSendSync,
{
    let shape = layouts.shape(layout);
    let mut dst = strided_kernel::StridedViewMut::new(
        dst_data,
        shape,
        layouts.strides(layout),
        layout.offset,
    )
    .map_err(strided_error)?;
    let src = strided_kernel::StridedView::<T>::new(
        packed,
        shape,
        layouts.packed_strides(layout),
        offset_to_isize(packed_offset)?,
    )
    .map_err(strided_error)?;

    if beta.is_zero() {
        strided_kernel::copy_scale(&mut dst, &src, alpha).map_err(strided_error)
    } else {
        if !beta.is_one() {
            scale_destination(zero_strides, &mut dst, beta)?;
        }
        strided_kernel::axpy(&mut dst, &src, alpha).map_err(strided_error)
    }
}

fn scale_destination<T>(
    zero_strides: &mut Vec<isize>,
    dst: &mut strided_kernel::StridedViewMut<'_, T>,
    beta: T,
) -> Result<(), OperationError>
where
    T: Copy + Mul<T, Output = T> + strided_kernel::MaybeSendSync,
{
    let scalar = [beta];
    zero_strides.clear();
    zero_strides.resize(dst.ndim(), 0);
    let beta_view =
        strided_kernel::StridedView::<T>::new(&scalar, dst.dims(), zero_strides.as_slice(), 0)
            .map_err(strided_error)?;
    strided_kernel::mul(dst, &beta_view).map_err(strided_error)
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum OperationError {
    Core(CoreError),
    Dense(DenseError),
    BlockIndexOutOfBounds {
        tensor: &'static str,
        index: usize,
        count: usize,
    },
    BlockCountMismatch {
        dst: usize,
        src: usize,
    },
    CoefficientCountMismatch {
        expected: usize,
        actual: usize,
    },
    DuplicateTransformDestination {
        dst_block: usize,
    },
    ElementCountMismatch {
        expected: usize,
        actual: usize,
    },
    ElementCountOverflow,
    EmptyTransformBlock,
    InvalidPermutation {
        axes: Vec<usize>,
        rank: usize,
    },
    FusionTreeGroupMismatch {
        tensor: &'static str,
        index: usize,
    },
    RankMismatch {
        expected: usize,
        actual: usize,
    },
    StructureMismatch {
        tensor: &'static str,
    },
    StructureRankMismatch {
        expected: usize,
        actual: usize,
    },
    MissingBlockKey {
        key: BlockKey,
    },
    ShapeMismatch {
        dst: Vec<usize>,
        src: Vec<usize>,
    },
    StrideOverflow {
        value: usize,
    },
    OffsetOverflow {
        value: usize,
    },
    StridedKernel {
        message: String,
    },
}

impl fmt::Display for OperationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Core(err) => err.fmt(f),
            Self::Dense(err) => err.fmt(f),
            Self::BlockIndexOutOfBounds {
                tensor,
                index,
                count,
            } => {
                write!(
                    f,
                    "{tensor} block index {index} is out of bounds for {count} blocks"
                )
            }
            Self::BlockCountMismatch { dst, src } => {
                write!(f, "block count mismatch: dst {dst}, src {src}")
            }
            Self::CoefficientCountMismatch { expected, actual } => {
                write!(
                    f,
                    "coefficient count mismatch: expected {expected}, got {actual}"
                )
            }
            Self::DuplicateTransformDestination { dst_block } => {
                write!(
                    f,
                    "tree transform destination block {dst_block} appears in more than one block"
                )
            }
            Self::ElementCountMismatch { expected, actual } => {
                write!(
                    f,
                    "element count mismatch: expected {expected}, got {actual}"
                )
            }
            Self::ElementCountOverflow => write!(f, "element count overflow"),
            Self::EmptyTransformBlock => {
                write!(f, "tree transform block has no source or destination")
            }
            Self::InvalidPermutation { axes, rank } => {
                write!(f, "invalid axis permutation {axes:?} for rank {rank}")
            }
            Self::FusionTreeGroupMismatch { tensor, index } => {
                write!(
                    f,
                    "{tensor} block {index} does not match the fusion-tree group"
                )
            }
            Self::RankMismatch { expected, actual } => {
                write!(f, "rank mismatch: expected {expected}, got {actual}")
            }
            Self::StructureMismatch { tensor } => {
                write!(
                    f,
                    "{tensor} tensor structure does not match compiled structure"
                )
            }
            Self::StructureRankMismatch { expected, actual } => {
                write!(
                    f,
                    "block structure rank mismatch: expected {expected}, got {actual}"
                )
            }
            Self::MissingBlockKey { key } => {
                write!(f, "missing matching block for key {key:?}")
            }
            Self::ShapeMismatch { dst, src } => {
                write!(f, "shape mismatch: dst {dst:?}, src {src:?}")
            }
            Self::StrideOverflow { value } => {
                write!(f, "stride {value} does not fit in strided-rs isize")
            }
            Self::OffsetOverflow { value } => {
                write!(f, "offset {value} does not fit in strided-rs isize")
            }
            Self::StridedKernel { message } => write!(f, "strided kernel error: {message}"),
        }
    }
}

impl std::error::Error for OperationError {}

impl From<CoreError> for OperationError {
    fn from(value: CoreError) -> Self {
        Self::Core(value)
    }
}

impl OperationError {
    fn from_core_preserving_context(value: CoreError) -> Self {
        match value {
            CoreError::MissingBlockKey { key } => Self::MissingBlockKey { key },
            other => Self::Core(other),
        }
    }
}

fn strided_read<'a, T>(
    view: BlockView<'a, T>,
) -> Result<strided_kernel::StridedView<'a, T>, OperationError> {
    let layout = view.layout();
    let strides = strides_to_isize(layout.strides())?;
    let offset = offset_to_isize(layout.offset())?;
    strided_kernel::StridedView::new(view.data(), layout.shape(), &strides, offset)
        .map_err(strided_error)
}

fn strided_write<'a, T>(
    view: BlockViewMut<'a, T>,
) -> Result<strided_kernel::StridedViewMut<'a, T>, OperationError> {
    let (data, layout) = view.into_parts();
    let strides = strides_to_isize(layout.strides())?;
    let offset = offset_to_isize(layout.offset())?;
    strided_kernel::StridedViewMut::new(data, layout.shape(), &strides, offset)
        .map_err(strided_error)
}

fn strides_to_isize(strides: &[usize]) -> Result<Vec<isize>, OperationError> {
    strides
        .iter()
        .map(|&stride| {
            isize::try_from(stride).map_err(|_| OperationError::StrideOverflow { value: stride })
        })
        .collect()
}

fn offset_to_isize(offset: usize) -> Result<isize, OperationError> {
    isize::try_from(offset).map_err(|_| OperationError::OffsetOverflow { value: offset })
}

fn element_count(shape: &[usize]) -> Result<usize, OperationError> {
    shape.iter().try_fold(1usize, |acc, &dim| {
        acc.checked_mul(dim)
            .ok_or(OperationError::ElementCountOverflow)
    })
}

fn column_major_strides_isize(shape: &[usize]) -> Result<Vec<isize>, OperationError> {
    let mut stride = 1usize;
    let mut strides = Vec::with_capacity(shape.len());
    for &dim in shape {
        strides.push(
            isize::try_from(stride)
                .map_err(|_| OperationError::StrideOverflow { value: stride })?,
        );
        stride = stride
            .checked_mul(dim)
            .ok_or(OperationError::ElementCountOverflow)?;
    }
    Ok(strides)
}

fn validate_structure_identity(
    tensor: &'static str,
    expected: &Arc<BlockStructure>,
    actual: &Arc<BlockStructure>,
) -> Result<(), OperationError> {
    if Arc::ptr_eq(expected, actual) || expected.as_ref() == actual.as_ref() {
        Ok(())
    } else {
        Err(OperationError::StructureMismatch { tensor })
    }
}

fn permutation_axes(
    permutation: AxisPermutation<'_>,
    rank: usize,
) -> Result<Vec<usize>, OperationError> {
    match permutation {
        AxisPermutation::Identity => Ok((0..rank).collect()),
        AxisPermutation::Axes(axes) => {
            if axes.len() != rank {
                return Err(OperationError::InvalidPermutation {
                    axes: axes.to_vec(),
                    rank,
                });
            }
            let mut seen = vec![false; rank];
            for &axis in axes {
                if axis >= rank || seen[axis] {
                    return Err(OperationError::InvalidPermutation {
                        axes: axes.to_vec(),
                        rank,
                    });
                }
                seen[axis] = true;
            }
            Ok(axes.to_vec())
        }
    }
}

fn strided_error(err: strided_kernel::StridedError) -> OperationError {
    OperationError::StridedKernel {
        message: err.to_string(),
    }
}

#[allow(dead_code)]
fn _assert_layout_owned_by_tenet(_layout: BlockLayout<'_>) {}

#[cfg(test)]
mod tests {
    use super::*;
    use num_complex::{Complex32, Complex64};
    use std::fmt::Debug;
    use tenet_core::{FusionTreeBlockKey, TensorMapSpace};

    fn fusion_tree_test_key<
        const COD: usize,
        const DOM: usize,
        const COD_DUAL: usize,
        const DOM_DUAL: usize,
    >(
        codomain: [usize; COD],
        domain: [usize; DOM],
        coupled: usize,
        codomain_is_dual: [bool; COD_DUAL],
        domain_is_dual: [bool; DOM_DUAL],
    ) -> BlockKey {
        BlockKey::from(FusionTreeBlockKey::pair_from_sector_ids(
            codomain,
            domain,
            Some(coupled),
            codomain_is_dual,
            domain_is_dual,
            [coupled + 100],
            [coupled + 200],
            [coupled + 300],
            [coupled + 400],
        ))
    }

    #[test]
    fn copy_into_uses_strided_kernel_for_transposed_views() {
        let src_data = [1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0];
        let src_shape = [3, 2];
        let src_strides = [2, 1];
        let dst_shape = [3, 2];
        let dst_strides = [1, 3];
        let mut dst_data = [0.0_f64; 6];

        let src = BlockView::new(&src_data, &src_shape, &src_strides, 0).unwrap();
        let dst = BlockViewMut::new(&mut dst_data, &dst_shape, &dst_strides, 0).unwrap();
        copy_into(dst, src).unwrap();

        assert_eq!(dst_data, [1.0, 3.0, 5.0, 2.0, 4.0, 6.0]);
    }

    #[test]
    fn scaled_assign_into_uses_strided_kernel() {
        let src_data = [1.0_f64, 2.0, 3.0, 4.0];
        let shape = [2, 2];
        let src_strides = [2, 1];
        let dst_strides = [1, 2];
        let mut dst_data = [0.0_f64; 4];

        let src = BlockView::new(&src_data, &shape, &src_strides, 0).unwrap();
        let dst = BlockViewMut::new(&mut dst_data, &shape, &dst_strides, 0).unwrap();
        scaled_assign_into(dst, src, 2.0).unwrap();

        assert_eq!(dst_data, [2.0, 6.0, 4.0, 8.0]);
    }

    #[test]
    fn scaled_add_into_uses_strided_kernel() {
        let src_data = [1.0_f64, 2.0, 3.0, 4.0];
        let shape = [2, 2];
        let strides = [1, 2];
        let mut dst_data = [10.0_f64, 20.0, 30.0, 40.0];

        let src = BlockView::new(&src_data, &shape, &strides, 0).unwrap();
        let dst = BlockViewMut::new(&mut dst_data, &shape, &strides, 0).unwrap();
        scaled_add_into(dst, src, 3.0).unwrap();

        assert_eq!(dst_data, [13.0, 26.0, 39.0, 52.0]);
    }

    fn assert_tensorcopy_dtype<T>(values: Vec<T>, fill: T)
    where
        T: Copy + Clone + Debug + PartialEq + strided_kernel::MaybeSendSync,
    {
        let space = TensorMapSpace::<2, 0>::from_dims([2, 2], []).unwrap();
        let src = TensorMap::<T, 2, 0>::from_vec(values.clone(), space.clone()).unwrap();
        let mut dst = TensorMap::<T, 2, 0>::filled(fill, space).unwrap();

        tensorcopy_into(&mut dst, &src).unwrap();

        assert_eq!(dst.data(), values.as_slice());
    }

    fn assert_tensoradd_dtype<T>(
        values: Vec<T>,
        fill: T,
        alpha: T,
        assign_expected: Vec<T>,
        add_expected: Vec<T>,
    ) where
        T: Copy
            + Clone
            + Debug
            + PartialEq
            + Add<T, Output = T>
            + Mul<T, Output = T>
            + Zero
            + One
            + strided_kernel::MaybeSendSync,
    {
        let space = TensorMapSpace::<2, 0>::from_dims([2, 2], []).unwrap();
        let src = TensorMap::<T, 2, 0>::from_vec(values.clone(), space.clone()).unwrap();

        let mut assign_dst = TensorMap::<T, 2, 0>::filled(fill, space.clone()).unwrap();
        tensoradd_assign_into(&mut assign_dst, &src, alpha).unwrap();
        assert_eq!(assign_dst.data(), assign_expected.as_slice());

        let mut add_dst = TensorMap::<T, 2, 0>::filled(fill, space).unwrap();
        tensoradd_add_into(&mut add_dst, &src, alpha).unwrap();
        assert_eq!(add_dst.data(), add_expected.as_slice());
    }

    fn assert_tensoradd_general_dtype<T>(
        values: Vec<T>,
        fill: T,
        alpha: T,
        beta: T,
        expected: Vec<T>,
    ) where
        T: Copy
            + Clone
            + Debug
            + PartialEq
            + Add<T, Output = T>
            + Mul<T, Output = T>
            + Zero
            + One
            + strided_kernel::MaybeSendSync,
    {
        let space = TensorMapSpace::<2, 0>::from_dims([2, 2], []).unwrap();
        let src = TensorMap::<T, 2, 0>::from_vec(values, space.clone()).unwrap();
        let mut dst = TensorMap::<T, 2, 0>::filled(fill, space).unwrap();

        tensoradd_into(&mut dst, &src, AxisPermutation::identity(), alpha, beta).unwrap();

        assert_eq!(dst.data(), expected.as_slice());
    }

    fn assert_tree_single_dtype<T>(
        values: Vec<T>,
        fill: T,
        coefficient: T,
        alpha: T,
        beta: T,
        expected: Vec<T>,
    ) where
        T: Copy
            + Clone
            + Debug
            + PartialEq
            + Add<T, Output = T>
            + Mul<T, Output = T>
            + Zero
            + One
            + strided_kernel::MaybeSendSync,
    {
        let space = TensorMapSpace::<2, 0>::from_dims([2, 2], []).unwrap();
        let src = TensorMap::<T, 2, 0>::from_vec(values, space.clone()).unwrap();
        let mut dst = TensorMap::<T, 2, 0>::filled(fill, space).unwrap();
        let structure = TreeTransformStructure::compile(
            &dst,
            &src,
            &[TreeTransformBlockSpec::single(0, 0, coefficient)],
        )
        .unwrap();
        let mut backend = HostTensorOperations;
        let mut workspace = TreeTransformWorkspace::default();

        tree_transform_execute_with(
            &mut backend,
            &mut workspace,
            &structure,
            &mut dst,
            &src,
            alpha,
            beta,
        )
        .unwrap();

        assert_eq!(dst.data(), expected.as_slice());
    }

    fn assert_tree_multi_dtype<T>(
        coefficients: Vec<T>,
        alpha: T,
        beta: T,
        fill: T,
        expected: Vec<T>,
    ) where
        T: Copy
            + Clone
            + Debug
            + PartialEq
            + Add<T, Output = T>
            + Mul<T, Output = T>
            + Zero
            + One
            + strided_kernel::MaybeSendSync,
    {
        let space = TensorMapSpace::<2, 0>::from_dims([4, 2], []).unwrap();
        let src_structure =
            BlockStructure::packed_column_major(2, [vec![2, 2], vec![2, 2]]).unwrap();
        let dst_structure =
            BlockStructure::packed_column_major(2, [vec![4, 1], vec![4, 1]]).unwrap();
        let src = TensorMap::<T, 2, 0>::from_vec_with_structure(
            vec![
                T::one(),
                T::one() + T::one(),
                T::one() + T::one() + T::one(),
                T::one() + T::one() + T::one() + T::one(),
                T::one() + T::one() + T::one() + T::one() + T::one(),
                T::one() + T::one() + T::one() + T::one() + T::one() + T::one(),
                T::one() + T::one() + T::one() + T::one() + T::one() + T::one() + T::one(),
                T::one()
                    + T::one()
                    + T::one()
                    + T::one()
                    + T::one()
                    + T::one()
                    + T::one()
                    + T::one(),
            ],
            space.clone(),
            src_structure,
        )
        .unwrap();
        let mut dst =
            TensorMap::<T, 2, 0>::from_vec_with_structure(vec![fill; 8], space, dst_structure)
                .unwrap();
        let structure = TreeTransformStructure::compile(
            &dst,
            &src,
            &[TreeTransformBlockSpec::multi(
                vec![0, 1],
                vec![0, 1],
                coefficients,
            )],
        )
        .unwrap();
        let mut backend = HostTensorOperations;
        let mut workspace = TreeTransformWorkspace::default();

        tree_transform_execute_with(
            &mut backend,
            &mut workspace,
            &structure,
            &mut dst,
            &src,
            alpha,
            beta,
        )
        .unwrap();

        assert_eq!(dst.data(), expected.as_slice());
        assert_eq!(workspace.source_len(), 8);
        assert_eq!(workspace.destination_len(), 8);
    }

    fn assert_tree_multi_tensorkit_orientation_dtype<T>(
        src_values: Vec<T>,
        coefficients_src_by_dst: Vec<T>,
        alpha: T,
        beta: T,
        fill: T,
        expected: Vec<T>,
    ) where
        T: Copy
            + Clone
            + Debug
            + PartialEq
            + Add<T, Output = T>
            + Mul<T, Output = T>
            + Zero
            + One
            + strided_kernel::MaybeSendSync,
    {
        let src_space = TensorMapSpace::<2, 0>::from_dims([6, 1], []).unwrap();
        let dst_space = TensorMapSpace::<2, 0>::from_dims([4, 1], []).unwrap();
        let src_structure =
            BlockStructure::packed_column_major(2, [vec![2, 1], vec![2, 1], vec![2, 1]]).unwrap();
        let dst_structure =
            BlockStructure::packed_column_major(2, [vec![2, 1], vec![2, 1]]).unwrap();
        let src =
            TensorMap::<T, 2, 0>::from_vec_with_structure(src_values, src_space, src_structure)
                .unwrap();
        let mut dst =
            TensorMap::<T, 2, 0>::from_vec_with_structure(vec![fill; 4], dst_space, dst_structure)
                .unwrap();
        let structure = TreeTransformStructure::compile(
            &dst,
            &src,
            &[TreeTransformBlockSpec::multi(
                vec![0, 1],
                vec![0, 1, 2],
                coefficients_src_by_dst,
            )],
        )
        .unwrap();
        let mut backend = HostTensorOperations;
        let mut workspace = TreeTransformWorkspace::default();

        tree_transform_execute_with(
            &mut backend,
            &mut workspace,
            &structure,
            &mut dst,
            &src,
            alpha,
            beta,
        )
        .unwrap();

        assert_eq!(dst.data(), expected.as_slice());
        assert_eq!(workspace.source_len(), 6);
        assert_eq!(workspace.destination_len(), 4);
    }

    fn assert_tree_multi_tensorkit_orientation_dense_dtype<T>(
        src_values: Vec<T>,
        coefficients_src_by_dst: Vec<T>,
        alpha: T,
        beta: T,
        fill: T,
        expected: Vec<T>,
    ) where
        T: DenseRecouplingScalar + Clone + Debug,
    {
        let src_space = TensorMapSpace::<2, 0>::from_dims([6, 1], []).unwrap();
        let dst_space = TensorMapSpace::<2, 0>::from_dims([4, 1], []).unwrap();
        let src_structure =
            BlockStructure::packed_column_major(2, [vec![2, 1], vec![2, 1], vec![2, 1]]).unwrap();
        let dst_structure =
            BlockStructure::packed_column_major(2, [vec![2, 1], vec![2, 1]]).unwrap();
        let src =
            TensorMap::<T, 2, 0>::from_vec_with_structure(src_values, src_space, src_structure)
                .unwrap();
        let mut dst =
            TensorMap::<T, 2, 0>::from_vec_with_structure(vec![fill; 4], dst_space, dst_structure)
                .unwrap();
        let structure = TreeTransformStructure::compile(
            &dst,
            &src,
            &[TreeTransformBlockSpec::multi(
                vec![0, 1],
                vec![0, 1, 2],
                coefficients_src_by_dst,
            )],
        )
        .unwrap();
        let mut backend = DenseTreeTransformOperations::default();
        let mut workspace = TreeTransformWorkspace::default();

        tree_transform_execute_with(
            &mut backend,
            &mut workspace,
            &structure,
            &mut dst,
            &src,
            alpha,
            beta,
        )
        .unwrap();

        assert_eq!(dst.data(), expected.as_slice());
        assert_eq!(workspace.source_len(), 6);
        assert_eq!(workspace.destination_len(), 4);
    }

    fn assert_tree_multi_keyed_dtype<T>(
        src_values: Vec<T>,
        coefficients_src_by_dst: Vec<T>,
        expected: Vec<T>,
    ) where
        T: Copy
            + Clone
            + Debug
            + PartialEq
            + Add<T, Output = T>
            + Mul<T, Output = T>
            + Zero
            + One
            + strided_kernel::MaybeSendSync,
    {
        let key10 = BlockKey::sector_ids([10]);
        let key20 = BlockKey::sector_ids([20]);
        let key100 = BlockKey::sector_ids([100]);
        let key200 = BlockKey::sector_ids([200]);
        let key300 = BlockKey::sector_ids([300]);
        let src_space = TensorMapSpace::<2, 0>::from_dims([6, 1], []).unwrap();
        let dst_space = TensorMapSpace::<2, 0>::from_dims([4, 1], []).unwrap();
        let src_structure = BlockStructure::packed_column_major_with_keys(
            2,
            [
                (key100.clone(), vec![2, 1]),
                (key300.clone(), vec![2, 1]),
                (key200.clone(), vec![2, 1]),
            ],
        )
        .unwrap();
        let dst_structure = BlockStructure::packed_column_major_with_keys(
            2,
            [(key20.clone(), vec![2, 1]), (key10.clone(), vec![2, 1])],
        )
        .unwrap();
        let src =
            TensorMap::<T, 2, 0>::from_vec_with_structure(src_values, src_space, src_structure)
                .unwrap();
        let mut dst = TensorMap::<T, 2, 0>::from_vec_with_structure(
            vec![T::zero(); 4],
            dst_space,
            dst_structure,
        )
        .unwrap();
        let structure = TreeTransformStructure::compile_keyed(
            &dst,
            &src,
            &[TreeTransformKeyBlockSpec::multi(
                vec![key10, key20],
                vec![key100, key200, key300],
                coefficients_src_by_dst,
            )],
        )
        .unwrap();
        let mut backend = HostTensorOperations;
        let mut workspace = TreeTransformWorkspace::default();

        tree_transform_execute_with(
            &mut backend,
            &mut workspace,
            &structure,
            &mut dst,
            &src,
            T::one(),
            T::zero(),
        )
        .unwrap();

        assert_eq!(structure.block_count(), 1);
        assert_eq!(dst.data(), expected.as_slice());
        assert_eq!(workspace.source_len(), 6);
        assert_eq!(workspace.destination_len(), 4);
    }

    #[derive(Default)]
    struct CountingDenseExecutor {
        dot_general_into_calls: usize,
    }

    impl DenseExecutor for CountingDenseExecutor {
        fn svd(
            &mut self,
            _input: DenseRead<'_>,
        ) -> Result<Vec<tenet_dense::DenseTensor>, DenseError> {
            unreachable!("tree transform does not call svd")
        }

        fn qr(
            &mut self,
            _input: DenseRead<'_>,
        ) -> Result<Vec<tenet_dense::DenseTensor>, DenseError> {
            unreachable!("tree transform does not call qr")
        }

        fn eigh(
            &mut self,
            _input: DenseRead<'_>,
        ) -> Result<Vec<tenet_dense::DenseTensor>, DenseError> {
            unreachable!("tree transform does not call eigh")
        }

        fn dot_general_into(
            &mut self,
            output: DenseWrite<'_>,
            lhs: DenseRead<'_>,
            rhs: DenseRead<'_>,
            config: &tenet_dense::DenseDotConfig,
        ) -> Result<(), DenseError> {
            self.dot_general_into_calls += 1;
            assert_eq!(config, &tenet_dense::DenseDotConfig::matmul());

            // This mock pins the TensorKit-style `mul!` boundary only:
            // `buffer_src :: (blocksize, n_src)` times `U^T :: (n_src, n_dst)`
            // into `buffer_dst :: (blocksize, n_dst)`. Numerical GEMM behavior
            // is covered by the DefaultDenseExecutor test.
            let (mut output, lhs, rhs) = match (output, lhs, rhs) {
                (DenseWrite::F64(output), DenseRead::F64(lhs), DenseRead::F64(rhs)) => {
                    (output, lhs, rhs)
                }
                _ => panic!("counting executor only covers f64 recoupling"),
            };

            assert_eq!(lhs.shape(), &[2, 3]);
            assert_eq!(lhs.strides(), &[1, 2]);
            assert_eq!(lhs.offset(), 0);
            assert_eq!(lhs.data(), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);

            assert_eq!(rhs.shape(), &[3, 2]);
            assert_eq!(rhs.strides(), &[1, 3]);
            assert_eq!(rhs.offset(), 0);
            assert_eq!(rhs.data(), &[10.0, 100.0, 1000.0, 20.0, 200.0, 2000.0]);

            assert_eq!(output.shape(), &[2, 2]);
            assert_eq!(output.strides(), &[1, 2]);
            assert_eq!(output.offset(), 0);

            let out_strides = output.strides().to_vec();
            let out_offset = output.offset();
            let out_data = output.data_mut();
            out_data[out_offset] = 5310.0;
            out_data[out_offset + out_strides[0]] = 6420.0;
            out_data[out_offset + out_strides[1]] = 10620.0;
            out_data[out_offset + out_strides[0] + out_strides[1]] = 12840.0;
            Ok(())
        }
    }

    #[test]
    fn tensorcopy_supports_all_storage_dtypes() {
        assert_tensorcopy_dtype(vec![1.0_f32, 2.0, 3.0, 4.0], 0.0);
        assert_tensorcopy_dtype(vec![1.0_f64, 2.0, 3.0, 4.0], 0.0);
        assert_tensorcopy_dtype(vec![1_i32, 2, 3, 4], 0);
        assert_tensorcopy_dtype(vec![1_i64, 2, 3, 4], 0);
        assert_tensorcopy_dtype(vec![true, false, true, false], false);
        assert_tensorcopy_dtype(
            vec![
                Complex32::new(1.0, 1.0),
                Complex32::new(2.0, -1.0),
                Complex32::new(3.0, 0.5),
                Complex32::new(4.0, -0.5),
            ],
            Complex32::new(0.0, 0.0),
        );
        assert_tensorcopy_dtype(
            vec![
                Complex64::new(1.0, 1.0),
                Complex64::new(2.0, -1.0),
                Complex64::new(3.0, 0.5),
                Complex64::new(4.0, -0.5),
            ],
            Complex64::new(0.0, 0.0),
        );
    }

    #[test]
    fn tensoradd_assign_and_add_support_all_numeric_dtypes() {
        assert_tensoradd_dtype(
            vec![1.0_f32, 2.0, 3.0, 4.0],
            10.0,
            2.0,
            vec![2.0, 4.0, 6.0, 8.0],
            vec![12.0, 14.0, 16.0, 18.0],
        );
        assert_tensoradd_dtype(
            vec![1.0_f64, 2.0, 3.0, 4.0],
            10.0,
            2.0,
            vec![2.0, 4.0, 6.0, 8.0],
            vec![12.0, 14.0, 16.0, 18.0],
        );
        assert_tensoradd_dtype(
            vec![1_i32, 2, 3, 4],
            10,
            2,
            vec![2, 4, 6, 8],
            vec![12, 14, 16, 18],
        );
        assert_tensoradd_dtype(
            vec![1_i64, 2, 3, 4],
            10,
            2,
            vec![2, 4, 6, 8],
            vec![12, 14, 16, 18],
        );
        assert_tensoradd_dtype(
            vec![
                Complex32::new(1.0, 1.0),
                Complex32::new(2.0, -1.0),
                Complex32::new(3.0, 0.5),
                Complex32::new(4.0, -0.5),
            ],
            Complex32::new(10.0, 0.0),
            Complex32::new(2.0, 0.0),
            vec![
                Complex32::new(2.0, 2.0),
                Complex32::new(4.0, -2.0),
                Complex32::new(6.0, 1.0),
                Complex32::new(8.0, -1.0),
            ],
            vec![
                Complex32::new(12.0, 2.0),
                Complex32::new(14.0, -2.0),
                Complex32::new(16.0, 1.0),
                Complex32::new(18.0, -1.0),
            ],
        );
        assert_tensoradd_dtype(
            vec![
                Complex64::new(1.0, 1.0),
                Complex64::new(2.0, -1.0),
                Complex64::new(3.0, 0.5),
                Complex64::new(4.0, -0.5),
            ],
            Complex64::new(10.0, 0.0),
            Complex64::new(2.0, 0.0),
            vec![
                Complex64::new(2.0, 2.0),
                Complex64::new(4.0, -2.0),
                Complex64::new(6.0, 1.0),
                Complex64::new(8.0, -1.0),
            ],
            vec![
                Complex64::new(12.0, 2.0),
                Complex64::new(14.0, -2.0),
                Complex64::new(16.0, 1.0),
                Complex64::new(18.0, -1.0),
            ],
        );
    }

    #[test]
    fn tensoradd_general_beta_supports_all_numeric_dtypes() {
        assert_tensoradd_general_dtype(
            vec![1.0_f32, 2.0, 3.0, 4.0],
            10.0,
            2.0,
            3.0,
            vec![32.0, 34.0, 36.0, 38.0],
        );
        assert_tensoradd_general_dtype(
            vec![1.0_f64, 2.0, 3.0, 4.0],
            10.0,
            2.0,
            3.0,
            vec![32.0, 34.0, 36.0, 38.0],
        );
        assert_tensoradd_general_dtype(vec![1_i32, 2, 3, 4], 10, 2, 3, vec![32, 34, 36, 38]);
        assert_tensoradd_general_dtype(vec![1_i64, 2, 3, 4], 10, 2, 3, vec![32, 34, 36, 38]);
        assert_tensoradd_general_dtype(
            vec![
                Complex32::new(1.0, 1.0),
                Complex32::new(2.0, -1.0),
                Complex32::new(3.0, 0.5),
                Complex32::new(4.0, -0.5),
            ],
            Complex32::new(10.0, 1.0),
            Complex32::new(2.0, 0.0),
            Complex32::new(3.0, 0.0),
            vec![
                Complex32::new(32.0, 5.0),
                Complex32::new(34.0, 1.0),
                Complex32::new(36.0, 4.0),
                Complex32::new(38.0, 2.0),
            ],
        );
        assert_tensoradd_general_dtype(
            vec![
                Complex64::new(1.0, 1.0),
                Complex64::new(2.0, -1.0),
                Complex64::new(3.0, 0.5),
                Complex64::new(4.0, -0.5),
            ],
            Complex64::new(10.0, 1.0),
            Complex64::new(2.0, 0.0),
            Complex64::new(3.0, 0.0),
            vec![
                Complex64::new(32.0, 5.0),
                Complex64::new(34.0, 1.0),
                Complex64::new(36.0, 4.0),
                Complex64::new(38.0, 2.0),
            ],
        );
    }

    #[test]
    fn tree_transform_single_replay_supports_all_numeric_dtypes() {
        assert_tree_single_dtype(
            vec![1.0_f32, 2.0, 3.0, 4.0],
            10.0,
            3.0,
            2.0,
            4.0,
            vec![46.0, 52.0, 58.0, 64.0],
        );
        assert_tree_single_dtype(
            vec![1.0_f64, 2.0, 3.0, 4.0],
            10.0,
            3.0,
            2.0,
            4.0,
            vec![46.0, 52.0, 58.0, 64.0],
        );
        assert_tree_single_dtype(vec![1_i32, 2, 3, 4], 10, 3, 2, 4, vec![46, 52, 58, 64]);
        assert_tree_single_dtype(vec![1_i64, 2, 3, 4], 10, 3, 2, 4, vec![46, 52, 58, 64]);
        assert_tree_single_dtype(
            vec![
                Complex32::new(1.0, 1.0),
                Complex32::new(2.0, -1.0),
                Complex32::new(3.0, 0.5),
                Complex32::new(4.0, -0.5),
            ],
            Complex32::new(10.0, 1.0),
            Complex32::new(3.0, 0.0),
            Complex32::new(2.0, 0.0),
            Complex32::new(4.0, 0.0),
            vec![
                Complex32::new(46.0, 10.0),
                Complex32::new(52.0, -2.0),
                Complex32::new(58.0, 7.0),
                Complex32::new(64.0, 1.0),
            ],
        );
        assert_tree_single_dtype(
            vec![
                Complex64::new(1.0, 1.0),
                Complex64::new(2.0, -1.0),
                Complex64::new(3.0, 0.5),
                Complex64::new(4.0, -0.5),
            ],
            Complex64::new(10.0, 1.0),
            Complex64::new(3.0, 0.0),
            Complex64::new(2.0, 0.0),
            Complex64::new(4.0, 0.0),
            vec![
                Complex64::new(46.0, 10.0),
                Complex64::new(52.0, -2.0),
                Complex64::new(58.0, 7.0),
                Complex64::new(64.0, 1.0),
            ],
        );
    }

    #[test]
    fn tree_transform_multi_pack_gemm_scatter_supports_all_numeric_dtypes() {
        assert_tree_multi_dtype(
            vec![2.0_f32, 3.0, 5.0, 7.0],
            2.0,
            10.0,
            1.0,
            vec![44.0, 54.0, 64.0, 74.0, 90.0, 114.0, 138.0, 162.0],
        );
        assert_tree_multi_dtype(
            vec![2.0_f64, 3.0, 5.0, 7.0],
            2.0,
            10.0,
            1.0,
            vec![44.0, 54.0, 64.0, 74.0, 90.0, 114.0, 138.0, 162.0],
        );
        assert_tree_multi_dtype(
            vec![2_i32, 3, 5, 7],
            2,
            10,
            1,
            vec![44, 54, 64, 74, 90, 114, 138, 162],
        );
        assert_tree_multi_dtype(
            vec![2_i64, 3, 5, 7],
            2,
            10,
            1,
            vec![44, 54, 64, 74, 90, 114, 138, 162],
        );
        assert_tree_multi_dtype(
            vec![
                Complex32::new(2.0, 0.0),
                Complex32::new(3.0, 0.0),
                Complex32::new(5.0, 0.0),
                Complex32::new(7.0, 0.0),
            ],
            Complex32::new(2.0, 0.0),
            Complex32::new(10.0, 0.0),
            Complex32::new(1.0, 1.0),
            vec![
                Complex32::new(44.0, 10.0),
                Complex32::new(54.0, 10.0),
                Complex32::new(64.0, 10.0),
                Complex32::new(74.0, 10.0),
                Complex32::new(90.0, 10.0),
                Complex32::new(114.0, 10.0),
                Complex32::new(138.0, 10.0),
                Complex32::new(162.0, 10.0),
            ],
        );
        assert_tree_multi_dtype(
            vec![
                Complex64::new(2.0, 0.0),
                Complex64::new(3.0, 0.0),
                Complex64::new(5.0, 0.0),
                Complex64::new(7.0, 0.0),
            ],
            Complex64::new(2.0, 0.0),
            Complex64::new(10.0, 0.0),
            Complex64::new(1.0, 1.0),
            vec![
                Complex64::new(44.0, 10.0),
                Complex64::new(54.0, 10.0),
                Complex64::new(64.0, 10.0),
                Complex64::new(74.0, 10.0),
                Complex64::new(90.0, 10.0),
                Complex64::new(114.0, 10.0),
                Complex64::new(138.0, 10.0),
                Complex64::new(162.0, 10.0),
            ],
        );
    }

    #[test]
    fn tree_transform_multi_uses_tensorkit_recoupling_orientation_for_all_numeric_dtypes() {
        assert_tree_multi_tensorkit_orientation_dtype(
            vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0],
            vec![10.0, 100.0, 1000.0, 20.0, 200.0, 2000.0],
            2.0,
            3.0,
            1.0,
            vec![10623.0, 12843.0, 21243.0, 25683.0],
        );
        assert_tree_multi_tensorkit_orientation_dtype(
            vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0],
            vec![10.0, 100.0, 1000.0, 20.0, 200.0, 2000.0],
            2.0,
            3.0,
            1.0,
            vec![10623.0, 12843.0, 21243.0, 25683.0],
        );
        assert_tree_multi_tensorkit_orientation_dtype(
            vec![1_i32, 2, 3, 4, 5, 6],
            vec![10, 100, 1000, 20, 200, 2000],
            2,
            3,
            1,
            vec![10623, 12843, 21243, 25683],
        );
        assert_tree_multi_tensorkit_orientation_dtype(
            vec![1_i64, 2, 3, 4, 5, 6],
            vec![10, 100, 1000, 20, 200, 2000],
            2,
            3,
            1,
            vec![10623, 12843, 21243, 25683],
        );
        assert_tree_multi_tensorkit_orientation_dtype(
            vec![
                Complex32::new(1.0, 0.0),
                Complex32::new(2.0, 0.0),
                Complex32::new(3.0, 0.0),
                Complex32::new(4.0, 0.0),
                Complex32::new(5.0, 0.0),
                Complex32::new(6.0, 0.0),
            ],
            vec![
                Complex32::new(10.0, 0.0),
                Complex32::new(100.0, 0.0),
                Complex32::new(1000.0, 0.0),
                Complex32::new(20.0, 0.0),
                Complex32::new(200.0, 0.0),
                Complex32::new(2000.0, 0.0),
            ],
            Complex32::new(2.0, 0.0),
            Complex32::new(3.0, 0.0),
            Complex32::new(1.0, 1.0),
            vec![
                Complex32::new(10623.0, 3.0),
                Complex32::new(12843.0, 3.0),
                Complex32::new(21243.0, 3.0),
                Complex32::new(25683.0, 3.0),
            ],
        );
        assert_tree_multi_tensorkit_orientation_dtype(
            vec![
                Complex64::new(1.0, 0.0),
                Complex64::new(2.0, 0.0),
                Complex64::new(3.0, 0.0),
                Complex64::new(4.0, 0.0),
                Complex64::new(5.0, 0.0),
                Complex64::new(6.0, 0.0),
            ],
            vec![
                Complex64::new(10.0, 0.0),
                Complex64::new(100.0, 0.0),
                Complex64::new(1000.0, 0.0),
                Complex64::new(20.0, 0.0),
                Complex64::new(200.0, 0.0),
                Complex64::new(2000.0, 0.0),
            ],
            Complex64::new(2.0, 0.0),
            Complex64::new(3.0, 0.0),
            Complex64::new(1.0, 1.0),
            vec![
                Complex64::new(10623.0, 3.0),
                Complex64::new(12843.0, 3.0),
                Complex64::new(21243.0, 3.0),
                Complex64::new(25683.0, 3.0),
            ],
        );
    }

    #[test]
    fn tree_transform_dense_backend_matches_tensorkit_recoupling_orientation_for_gemm_dtypes() {
        assert_tree_multi_tensorkit_orientation_dense_dtype(
            vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0],
            vec![10.0, 100.0, 1000.0, 20.0, 200.0, 2000.0],
            2.0,
            3.0,
            1.0,
            vec![10623.0, 12843.0, 21243.0, 25683.0],
        );
        assert_tree_multi_tensorkit_orientation_dense_dtype(
            vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0],
            vec![10.0, 100.0, 1000.0, 20.0, 200.0, 2000.0],
            2.0,
            3.0,
            1.0,
            vec![10623.0, 12843.0, 21243.0, 25683.0],
        );
        assert_tree_multi_tensorkit_orientation_dense_dtype(
            vec![
                Complex32::new(1.0, 0.0),
                Complex32::new(2.0, 0.0),
                Complex32::new(3.0, 0.0),
                Complex32::new(4.0, 0.0),
                Complex32::new(5.0, 0.0),
                Complex32::new(6.0, 0.0),
            ],
            vec![
                Complex32::new(10.0, 0.0),
                Complex32::new(100.0, 0.0),
                Complex32::new(1000.0, 0.0),
                Complex32::new(20.0, 0.0),
                Complex32::new(200.0, 0.0),
                Complex32::new(2000.0, 0.0),
            ],
            Complex32::new(2.0, 0.0),
            Complex32::new(3.0, 0.0),
            Complex32::new(1.0, 1.0),
            vec![
                Complex32::new(10623.0, 3.0),
                Complex32::new(12843.0, 3.0),
                Complex32::new(21243.0, 3.0),
                Complex32::new(25683.0, 3.0),
            ],
        );
        assert_tree_multi_tensorkit_orientation_dense_dtype(
            vec![
                Complex64::new(1.0, 0.0),
                Complex64::new(2.0, 0.0),
                Complex64::new(3.0, 0.0),
                Complex64::new(4.0, 0.0),
                Complex64::new(5.0, 0.0),
                Complex64::new(6.0, 0.0),
            ],
            vec![
                Complex64::new(10.0, 0.0),
                Complex64::new(100.0, 0.0),
                Complex64::new(1000.0, 0.0),
                Complex64::new(20.0, 0.0),
                Complex64::new(200.0, 0.0),
                Complex64::new(2000.0, 0.0),
            ],
            Complex64::new(2.0, 0.0),
            Complex64::new(3.0, 0.0),
            Complex64::new(1.0, 1.0),
            vec![
                Complex64::new(10623.0, 3.0),
                Complex64::new(12843.0, 3.0),
                Complex64::new(21243.0, 3.0),
                Complex64::new(25683.0, 3.0),
            ],
        );
    }

    #[test]
    fn tree_transform_dense_backend_calls_dense_matmul_for_multi_tree_blocks() {
        let src_space = TensorMapSpace::<2, 0>::from_dims([6, 1], []).unwrap();
        let dst_space = TensorMapSpace::<2, 0>::from_dims([4, 1], []).unwrap();
        let src_structure =
            BlockStructure::packed_column_major(2, [vec![2, 1], vec![2, 1], vec![2, 1]]).unwrap();
        let dst_structure =
            BlockStructure::packed_column_major(2, [vec![2, 1], vec![2, 1]]).unwrap();
        let src = TensorMap::<f64, 2, 0>::from_vec_with_structure(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            src_space,
            src_structure,
        )
        .unwrap();
        let mut dst =
            TensorMap::<f64, 2, 0>::from_vec_with_structure(vec![1.0; 4], dst_space, dst_structure)
                .unwrap();
        let structure = TreeTransformStructure::compile(
            &dst,
            &src,
            &[TreeTransformBlockSpec::multi(
                vec![0, 1],
                vec![0, 1, 2],
                vec![10.0, 100.0, 1000.0, 20.0, 200.0, 2000.0],
            )],
        )
        .unwrap();
        let mut backend = DenseTreeTransformOperations::new(CountingDenseExecutor::default());
        let mut workspace = TreeTransformWorkspace::default();

        tree_transform_execute_with(
            &mut backend,
            &mut workspace,
            &structure,
            &mut dst,
            &src,
            2.0,
            3.0,
        )
        .unwrap();

        assert_eq!(backend.dense().dot_general_into_calls, 1);
        assert_eq!(dst.data(), &[10623.0, 12843.0, 21243.0, 25683.0]);
    }

    #[test]
    fn tree_transform_compile_keyed_pairs_tree_blocks_by_key_not_index_for_all_numeric_dtypes() {
        assert_tree_multi_keyed_dtype(
            vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0],
            vec![10.0, 100.0, 1000.0, 20.0, 200.0, 2000.0],
            vec![7020.0, 9240.0, 3510.0, 4620.0],
        );
        assert_tree_multi_keyed_dtype(
            vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0],
            vec![10.0, 100.0, 1000.0, 20.0, 200.0, 2000.0],
            vec![7020.0, 9240.0, 3510.0, 4620.0],
        );
        assert_tree_multi_keyed_dtype(
            vec![1_i32, 2, 3, 4, 5, 6],
            vec![10, 100, 1000, 20, 200, 2000],
            vec![7020, 9240, 3510, 4620],
        );
        assert_tree_multi_keyed_dtype(
            vec![1_i64, 2, 3, 4, 5, 6],
            vec![10, 100, 1000, 20, 200, 2000],
            vec![7020, 9240, 3510, 4620],
        );
        assert_tree_multi_keyed_dtype(
            vec![
                Complex32::new(1.0, 0.0),
                Complex32::new(2.0, 0.0),
                Complex32::new(3.0, 0.0),
                Complex32::new(4.0, 0.0),
                Complex32::new(5.0, 0.0),
                Complex32::new(6.0, 0.0),
            ],
            vec![
                Complex32::new(10.0, 0.0),
                Complex32::new(100.0, 0.0),
                Complex32::new(1000.0, 0.0),
                Complex32::new(20.0, 0.0),
                Complex32::new(200.0, 0.0),
                Complex32::new(2000.0, 0.0),
            ],
            vec![
                Complex32::new(7020.0, 0.0),
                Complex32::new(9240.0, 0.0),
                Complex32::new(3510.0, 0.0),
                Complex32::new(4620.0, 0.0),
            ],
        );
        assert_tree_multi_keyed_dtype(
            vec![
                Complex64::new(1.0, 0.0),
                Complex64::new(2.0, 0.0),
                Complex64::new(3.0, 0.0),
                Complex64::new(4.0, 0.0),
                Complex64::new(5.0, 0.0),
                Complex64::new(6.0, 0.0),
            ],
            vec![
                Complex64::new(10.0, 0.0),
                Complex64::new(100.0, 0.0),
                Complex64::new(1000.0, 0.0),
                Complex64::new(20.0, 0.0),
                Complex64::new(200.0, 0.0),
                Complex64::new(2000.0, 0.0),
            ],
            vec![
                Complex64::new(7020.0, 0.0),
                Complex64::new(9240.0, 0.0),
                Complex64::new(3510.0, 0.0),
                Complex64::new(4620.0, 0.0),
            ],
        );
    }

    #[test]
    fn tensoradd_with_backend_allocator_applies_axis_permutation() {
        let src_space = TensorMapSpace::<2, 0>::from_dims([2, 3], []).unwrap();
        let dst_space = TensorMapSpace::<2, 0>::from_dims([3, 2], []).unwrap();
        let src = TensorMap::<f64, 2, 0>::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], src_space)
            .unwrap();
        let mut dst = TensorMap::<f64, 2, 0>::filled(10.0, dst_space).unwrap();
        let mut backend = HostTensorOperations;
        let mut allocator = HostAllocator::default();

        tensoradd_into_with(
            &mut backend,
            &mut allocator,
            &mut dst,
            &src,
            AxisPermutation::from_axes(&[1, 0]),
            2.0,
            3.0,
        )
        .unwrap();

        assert_eq!(dst.data(), &[32.0, 36.0, 40.0, 34.0, 38.0, 42.0]);
    }

    #[test]
    fn tensoradd_structure_precomputes_permutation_pairing_and_descriptor() {
        let src_space = TensorMapSpace::<2, 0>::from_dims([2, 3], []).unwrap();
        let dst_space = TensorMapSpace::<2, 0>::from_dims([3, 2], []).unwrap();
        let src = TensorMap::<f64, 2, 0>::from_vec(vec![1.0; 6], src_space).unwrap();
        let dst = TensorMap::<f64, 2, 0>::filled(0.0, dst_space).unwrap();

        let structure =
            TensorAddStructure::compile(&dst, &src, AxisPermutation::from_axes(&[1, 0])).unwrap();

        assert_eq!(structure.rank(), 2);
        assert_eq!(structure.axes(), &[1, 0]);
        assert_eq!(structure.terms().len(), 1);
        assert_eq!(structure.terms()[0].key(), &BlockKey::trivial());
        assert_eq!(structure.terms()[0].dst_block(), 0);
        assert_eq!(structure.terms()[0].src_block(), 0);
    }

    #[test]
    fn tensoradd_structure_replays_without_recompiling() {
        let src_space = TensorMapSpace::<2, 0>::from_dims([2, 3], []).unwrap();
        let dst_space = TensorMapSpace::<2, 0>::from_dims([3, 2], []).unwrap();
        let src = TensorMap::<f64, 2, 0>::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], src_space)
            .unwrap();
        let mut dst = TensorMap::<f64, 2, 0>::filled(10.0, dst_space).unwrap();
        let structure =
            TensorAddStructure::compile(&dst, &src, AxisPermutation::from_axes(&[1, 0])).unwrap();
        let mut backend = HostTensorOperations;
        let mut allocator = HostAllocator::default();

        tensoradd_execute_with(
            &mut backend,
            &mut allocator,
            &structure,
            &mut dst,
            &src,
            2.0,
            0.0,
        )
        .unwrap();
        tensoradd_execute_with(
            &mut backend,
            &mut allocator,
            &structure,
            &mut dst,
            &src,
            1.0,
            1.0,
        )
        .unwrap();

        assert_eq!(dst.data(), &[3.0, 9.0, 15.0, 6.0, 12.0, 18.0]);
    }

    #[test]
    fn tensoradd_structure_compiles_concrete_shape_and_replays_it() {
        let src_space = TensorMapSpace::<2, 0>::from_dims([4, 5], []).unwrap();
        let dst_space = TensorMapSpace::<2, 0>::from_dims([5, 4], []).unwrap();
        let src = TensorMap::<f64, 2, 0>::from_vec((1..=20).map(|x| x as f64).collect(), src_space)
            .unwrap();
        let mut dst = TensorMap::<f64, 2, 0>::filled(0.0, dst_space).unwrap();
        let structure =
            TensorAddStructure::compile(&dst, &src, AxisPermutation::from_axes(&[1, 0])).unwrap();
        let mut backend = HostTensorOperations;
        let mut allocator = HostAllocator::default();

        tensoradd_execute_with(
            &mut backend,
            &mut allocator,
            &structure,
            &mut dst,
            &src,
            2.0,
            0.0,
        )
        .unwrap();

        assert_eq!(
            dst.data(),
            &[
                2.0, 10.0, 18.0, 26.0, 34.0, 4.0, 12.0, 20.0, 28.0, 36.0, 6.0, 14.0, 22.0, 30.0,
                38.0, 8.0, 16.0, 24.0, 32.0, 40.0,
            ]
        );
    }

    #[test]
    fn tensoradd_structure_replays_multiple_packed_blocks() {
        let src_space = TensorMapSpace::<2, 0>::from_dims([4, 4], []).unwrap();
        let dst_space = TensorMapSpace::<2, 0>::from_dims([4, 4], []).unwrap();
        let src_structure =
            BlockStructure::packed_column_major(2, [vec![2, 3], vec![1, 4]]).unwrap();
        let dst_structure =
            BlockStructure::packed_column_major(2, [vec![3, 2], vec![4, 1]]).unwrap();
        let src = TensorMap::<f64, 2, 0>::from_vec_with_structure(
            (1..=10).map(|x| x as f64).collect(),
            src_space,
            src_structure,
        )
        .unwrap();
        let mut dst = TensorMap::<f64, 2, 0>::from_vec_with_structure(
            vec![0.0; 10],
            dst_space,
            dst_structure,
        )
        .unwrap();
        let structure =
            TensorAddStructure::compile(&dst, &src, AxisPermutation::from_axes(&[1, 0])).unwrap();
        let mut backend = HostTensorOperations;
        let mut allocator = HostAllocator::default();

        tensoradd_execute_with(
            &mut backend,
            &mut allocator,
            &structure,
            &mut dst,
            &src,
            2.0,
            0.0,
        )
        .unwrap();

        assert_eq!(
            dst.data(),
            &[2.0, 6.0, 10.0, 4.0, 8.0, 12.0, 14.0, 16.0, 18.0, 20.0]
        );
    }

    #[test]
    fn tensoradd_structure_pairs_blocks_by_key_not_index() {
        let src_space = TensorMapSpace::<2, 0>::from_dims([4, 4], []).unwrap();
        let dst_space = TensorMapSpace::<2, 0>::from_dims([4, 4], []).unwrap();
        let src_structure = BlockStructure::packed_column_major_with_keys(
            2,
            [
                (BlockKey::sector_ids([10]), vec![2, 3]),
                (BlockKey::sector_ids([20]), vec![1, 4]),
            ],
        )
        .unwrap();
        let dst_structure = BlockStructure::packed_column_major_with_keys(
            2,
            [
                (BlockKey::sector_ids([20]), vec![4, 1]),
                (BlockKey::sector_ids([10]), vec![3, 2]),
            ],
        )
        .unwrap();
        let src = TensorMap::<f64, 2, 0>::from_vec_with_structure(
            (1..=10).map(|x| x as f64).collect(),
            src_space,
            src_structure,
        )
        .unwrap();
        let mut dst = TensorMap::<f64, 2, 0>::from_vec_with_structure(
            vec![0.0; 10],
            dst_space,
            dst_structure,
        )
        .unwrap();
        let structure =
            TensorAddStructure::compile(&dst, &src, AxisPermutation::from_axes(&[1, 0])).unwrap();
        let mut backend = HostTensorOperations;
        let mut allocator = HostAllocator::default();

        assert_eq!(structure.terms()[0].key(), &BlockKey::sector_ids([20]));
        assert_eq!(structure.terms()[0].dst_block(), 0);
        assert_eq!(structure.terms()[0].src_block(), 1);
        assert_eq!(structure.terms()[1].key(), &BlockKey::sector_ids([10]));
        assert_eq!(structure.terms()[1].dst_block(), 1);
        assert_eq!(structure.terms()[1].src_block(), 0);

        tensoradd_execute_with(
            &mut backend,
            &mut allocator,
            &structure,
            &mut dst,
            &src,
            2.0,
            0.0,
        )
        .unwrap();

        assert_eq!(
            dst.data(),
            &[14.0, 16.0, 18.0, 20.0, 2.0, 6.0, 10.0, 4.0, 8.0, 12.0]
        );
    }

    #[test]
    fn tensoradd_structure_rejects_invalid_permutation_at_compile_time() {
        let space = TensorMapSpace::<2, 0>::from_dims([2, 2], []).unwrap();
        let src = TensorMap::<f64, 2, 0>::filled(1.0, space.clone()).unwrap();
        let dst = TensorMap::<f64, 2, 0>::filled(0.0, space).unwrap();

        let err = TensorAddStructure::compile(&dst, &src, AxisPermutation::from_axes(&[0, 0]))
            .unwrap_err();

        assert_eq!(
            err,
            OperationError::InvalidPermutation {
                axes: vec![0, 0],
                rank: 2,
            }
        );
    }

    #[test]
    fn tensoradd_structure_rejects_incompatible_shape_at_compile_time() {
        let src_space = TensorMapSpace::<2, 0>::from_dims([4, 5], []).unwrap();
        let dst_space = TensorMapSpace::<2, 0>::from_dims([4, 5], []).unwrap();
        let src = TensorMap::<f64, 2, 0>::filled(1.0, src_space).unwrap();
        let dst = TensorMap::<f64, 2, 0>::filled(0.0, dst_space).unwrap();

        let err = TensorAddStructure::compile(&dst, &src, AxisPermutation::from_axes(&[1, 0]))
            .unwrap_err();

        assert_eq!(
            err,
            OperationError::ShapeMismatch {
                dst: vec![4, 5],
                src: vec![5, 4],
            }
        );
    }

    #[test]
    fn tensoradd_structure_rejects_incompatible_replay_structure() {
        let compile_src_space = TensorMapSpace::<2, 0>::from_dims([2, 3], []).unwrap();
        let compile_dst_space = TensorMapSpace::<2, 0>::from_dims([3, 2], []).unwrap();
        let compile_src = TensorMap::<f64, 2, 0>::filled(1.0, compile_src_space).unwrap();
        let compile_dst = TensorMap::<f64, 2, 0>::filled(0.0, compile_dst_space).unwrap();
        let structure = TensorAddStructure::compile(
            &compile_dst,
            &compile_src,
            AxisPermutation::from_axes(&[1, 0]),
        )
        .unwrap();

        let src_space = TensorMapSpace::<2, 0>::from_dims([4, 5], []).unwrap();
        let dst_space = TensorMapSpace::<2, 0>::from_dims([5, 4], []).unwrap();
        let src = TensorMap::<f64, 2, 0>::filled(1.0, src_space).unwrap();
        let mut dst = TensorMap::<f64, 2, 0>::filled(0.0, dst_space).unwrap();
        let mut backend = HostTensorOperations;
        let mut allocator = HostAllocator::default();

        let err = tensoradd_execute_with(
            &mut backend,
            &mut allocator,
            &structure,
            &mut dst,
            &src,
            1.0,
            0.0,
        )
        .unwrap_err();

        assert_eq!(err, OperationError::StructureMismatch { tensor: "dst" });
    }

    #[test]
    fn tree_transform_rejects_invalid_block_specs_at_compile_time() {
        let space = TensorMapSpace::<2, 0>::from_dims([4, 2], []).unwrap();
        let structure = BlockStructure::packed_column_major(2, [vec![2, 2], vec![2, 2]]).unwrap();
        let src = TensorMap::<f64, 2, 0>::from_vec_with_structure(
            vec![1.0; 8],
            space.clone(),
            structure.clone(),
        )
        .unwrap();
        let dst = TensorMap::<f64, 2, 0>::from_vec_with_structure(vec![0.0; 8], space, structure)
            .unwrap();

        let err = TreeTransformStructure::compile(
            &dst,
            &src,
            &[TreeTransformBlockSpec::multi(
                vec![0, 1],
                vec![0, 1],
                vec![1.0, 2.0],
            )],
        )
        .unwrap_err();
        assert_eq!(
            err,
            OperationError::CoefficientCountMismatch {
                expected: 4,
                actual: 2,
            }
        );

        let err = TreeTransformStructure::compile(
            &dst,
            &src,
            &[
                TreeTransformBlockSpec::single(0, 0, 1.0),
                TreeTransformBlockSpec::single(0, 1, 1.0),
            ],
        )
        .unwrap_err();
        assert_eq!(
            err,
            OperationError::DuplicateTransformDestination { dst_block: 0 }
        );
    }

    #[test]
    fn tree_transform_compile_keyed_rejects_missing_tree_block_key() {
        let src_space = TensorMapSpace::<2, 0>::from_dims([2, 2], []).unwrap();
        let dst_space = TensorMapSpace::<2, 0>::from_dims([2, 2], []).unwrap();
        let key1 = BlockKey::sector_ids([1]);
        let key2 = BlockKey::sector_ids([2]);
        let src_structure =
            BlockStructure::packed_column_major_with_keys(2, [(key1.clone(), vec![2, 2])]).unwrap();
        let dst_structure =
            BlockStructure::packed_column_major_with_keys(2, [(key1.clone(), vec![2, 2])]).unwrap();
        let src =
            TensorMap::<f64, 2, 0>::from_vec_with_structure(vec![1.0; 4], src_space, src_structure)
                .unwrap();
        let dst =
            TensorMap::<f64, 2, 0>::from_vec_with_structure(vec![0.0; 4], dst_space, dst_structure)
                .unwrap();

        let err = TreeTransformStructure::compile_keyed(
            &dst,
            &src,
            &[TreeTransformKeyBlockSpec::single(key2.clone(), key1, 1.0)],
        )
        .unwrap_err();

        assert_eq!(err, OperationError::MissingBlockKey { key: key2 });
    }

    #[test]
    fn tree_transform_group_block_spec_preserves_group_identity_and_ordered_keys() {
        let group_key = FusionTreeGroupKey::from_sector_ids([10, 20], [30], [false, true], [true]);
        let dst_key1 = BlockKey::sector_ids([101, 201]);
        let dst_key2 = BlockKey::sector_ids([102, 202]);
        let src_key = BlockKey::sector_ids([301, 401]);
        let spec = TreeTransformGroupBlockSpec::multi(
            group_key.clone(),
            [dst_key1.clone(), dst_key2.clone()],
            [src_key.clone()],
            vec![2.0_f64, 3.0],
        );

        assert_eq!(spec.group_key(), &group_key);
        assert_eq!(
            spec.group_key()
                .codomain_uncoupled()
                .iter()
                .map(|sector| sector.id())
                .collect::<Vec<_>>(),
            vec![10, 20]
        );
        assert_eq!(
            spec.group_key()
                .domain_uncoupled()
                .iter()
                .map(|sector| sector.id())
                .collect::<Vec<_>>(),
            vec![30]
        );
        assert_eq!(spec.group_key().codomain_is_dual(), &[false, true]);
        assert_eq!(spec.group_key().domain_is_dual(), &[true]);
        assert_eq!(spec.dst_keys(), &[dst_key1, dst_key2]);
        assert_eq!(spec.src_keys(), &[src_key]);
        assert_eq!(spec.coefficients_src_by_dst(), &[2.0, 3.0]);
    }

    #[test]
    fn tree_transform_compile_grouped_lowers_to_replay_ready_structure() {
        let group_key = FusionTreeGroupKey::from_sector_ids([10, 20], [30], [false, true], [true]);
        let key10 = BlockKey::sector_ids([10]);
        let key20 = BlockKey::sector_ids([20]);
        let key100 = BlockKey::sector_ids([100]);
        let key200 = BlockKey::sector_ids([200]);
        let key300 = BlockKey::sector_ids([300]);
        let src_space = TensorMapSpace::<2, 0>::from_dims([6, 1], []).unwrap();
        let dst_space = TensorMapSpace::<2, 0>::from_dims([4, 1], []).unwrap();
        let src_structure = BlockStructure::packed_column_major_with_keys(
            2,
            [
                (key100.clone(), vec![2, 1]),
                (key300.clone(), vec![2, 1]),
                (key200.clone(), vec![2, 1]),
            ],
        )
        .unwrap();
        let dst_structure = BlockStructure::packed_column_major_with_keys(
            2,
            [(key20.clone(), vec![2, 1]), (key10.clone(), vec![2, 1])],
        )
        .unwrap();
        let src = TensorMap::<f64, 2, 0>::from_vec_with_structure(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            src_space,
            src_structure,
        )
        .unwrap();
        let mut dst =
            TensorMap::<f64, 2, 0>::from_vec_with_structure(vec![0.0; 4], dst_space, dst_structure)
                .unwrap();
        let structure = TreeTransformStructure::compile_grouped(
            &dst,
            &src,
            &[TreeTransformGroupBlockSpec::multi(
                group_key,
                [key10, key20],
                [key100, key200, key300],
                vec![10.0, 100.0, 1000.0, 20.0, 200.0, 2000.0],
            )],
        )
        .unwrap();
        let mut backend = HostTensorOperations;
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

        assert_eq!(structure.block_count(), 1);
        assert_eq!(dst.data(), &[7020.0, 9240.0, 3510.0, 4620.0]);
        assert_eq!(workspace.source_len(), 6);
        assert_eq!(workspace.destination_len(), 4);
    }

    #[test]
    fn tree_transform_compile_grouped_rejects_missing_tree_block_key() {
        let src_space = TensorMapSpace::<2, 0>::from_dims([2, 2], []).unwrap();
        let dst_space = TensorMapSpace::<2, 0>::from_dims([2, 2], []).unwrap();
        let group_key = FusionTreeGroupKey::from_sector_ids([1], [1], [false], [true]);
        let present_key = BlockKey::sector_ids([1]);
        let missing_key = BlockKey::sector_ids([2]);
        let src_structure =
            BlockStructure::packed_column_major_with_keys(2, [(present_key.clone(), vec![2, 2])])
                .unwrap();
        let dst_structure =
            BlockStructure::packed_column_major_with_keys(2, [(present_key.clone(), vec![2, 2])])
                .unwrap();
        let src =
            TensorMap::<f64, 2, 0>::from_vec_with_structure(vec![1.0; 4], src_space, src_structure)
                .unwrap();
        let dst =
            TensorMap::<f64, 2, 0>::from_vec_with_structure(vec![0.0; 4], dst_space, dst_structure)
                .unwrap();

        let err = TreeTransformStructure::compile_grouped(
            &dst,
            &src,
            &[TreeTransformGroupBlockSpec::single(
                group_key,
                missing_key.clone(),
                present_key,
                1.0,
            )],
        )
        .unwrap_err();

        assert_eq!(err, OperationError::MissingBlockKey { key: missing_key });
    }

    #[test]
    fn tree_transform_group_block_spec_from_groups_uses_source_group_and_ordered_keys() {
        let src_key1 = fusion_tree_test_key([10, 20], [30], 5, [false, true], [true]);
        let src_key2 = fusion_tree_test_key([10, 20], [30], 6, [false, true], [true]);
        let dst_key1 = fusion_tree_test_key([20, 10], [30], 7, [true, false], [true]);
        let dst_key2 = fusion_tree_test_key([20, 10], [30], 8, [true, false], [true]);
        let src_structure = BlockStructure::packed_column_major_with_keys(
            2,
            [
                (src_key1.clone(), vec![1, 1]),
                (src_key2.clone(), vec![1, 1]),
            ],
        )
        .unwrap();
        let dst_structure = BlockStructure::packed_column_major_with_keys(
            2,
            [
                (dst_key1.clone(), vec![1, 1]),
                (dst_key2.clone(), vec![1, 1]),
            ],
        )
        .unwrap();
        let src_groups = src_structure.fusion_tree_groups();
        let dst_groups = dst_structure.fusion_tree_groups();

        let spec = TreeTransformGroupBlockSpec::from_block_groups(
            &dst_structure,
            &dst_groups[0],
            &src_structure,
            &src_groups[0],
            vec![1.0_f64, 2.0, 3.0, 4.0],
        )
        .unwrap();

        assert_eq!(spec.group_key(), src_groups[0].group_key());
        assert_ne!(spec.group_key(), dst_groups[0].group_key());
        assert_eq!(spec.src_keys(), &[src_key1, src_key2]);
        assert_eq!(spec.dst_keys(), &[dst_key1, dst_key2]);
        assert_eq!(spec.coefficients_src_by_dst(), &[1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn tree_transform_group_plan_compiles_across_degeneracy_shapes_without_layout_leakage() {
        let src_key1 = fusion_tree_test_key([10, 20], [30], 5, [false, true], [true]);
        let src_key2 = fusion_tree_test_key([10, 20], [30], 6, [false, true], [true]);
        let dst_key1 = fusion_tree_test_key([20, 10], [30], 7, [true, false], [true]);
        let dst_key2 = fusion_tree_test_key([20, 10], [30], 8, [true, false], [true]);
        let src_small = BlockStructure::packed_column_major_with_keys(
            2,
            [
                (src_key1.clone(), vec![2, 1]),
                (src_key2.clone(), vec![2, 1]),
            ],
        )
        .unwrap();
        let dst_small = BlockStructure::packed_column_major_with_keys(
            2,
            [
                (dst_key1.clone(), vec![2, 1]),
                (dst_key2.clone(), vec![2, 1]),
            ],
        )
        .unwrap();
        let src_large = BlockStructure::packed_column_major_with_keys(
            2,
            [(src_key1, vec![3, 1]), (src_key2, vec![3, 1])],
        )
        .unwrap();
        let dst_large = BlockStructure::packed_column_major_with_keys(
            2,
            [(dst_key1, vec![3, 1]), (dst_key2, vec![3, 1])],
        )
        .unwrap();
        let spec = TreeTransformGroupBlockSpec::from_block_groups(
            &dst_small,
            &dst_small.fusion_tree_groups()[0],
            &src_small,
            &src_small.fusion_tree_groups()[0],
            vec![1.0_f64, 0.0, 0.0, 1.0],
        )
        .unwrap();
        let plan = TreeTransformGroupPlan::new(vec![spec]);
        let key = TreeTransformGroupPlanKey::from_plan(
            TreeTransformOperationKey::transpose([1, 0], [0]),
            &plan,
        );
        let large_spec = TreeTransformGroupBlockSpec::from_block_groups(
            &dst_large,
            &dst_large.fusion_tree_groups()[0],
            &src_large,
            &src_large.fusion_tree_groups()[0],
            vec![1.0_f64, 0.0, 0.0, 1.0],
        )
        .unwrap();
        let large_plan = TreeTransformGroupPlan::new(vec![large_spec]);
        let large_key = TreeTransformGroupPlanKey::from_plan(
            TreeTransformOperationKey::transpose([1, 0], [0]),
            &large_plan,
        );
        let mut cache = TreeTransformGroupPlanCache::new();

        cache.insert(key.clone(), plan.clone());

        let small_structure = plan.compile_structures(&dst_small, &src_small).unwrap();
        let cached = cache.get(&large_key).unwrap();
        let large_structure = cached.compile_structures(&dst_large, &src_large).unwrap();

        assert_eq!(key, large_key);
        assert_eq!(cache.len(), 1);
        assert_eq!(plan.specs().len(), 1);
        assert_eq!(small_structure.block_count(), 1);
        assert_eq!(large_structure.block_count(), 1);
        assert_eq!(small_structure.workspace_lens(), (4, 4));
        assert_eq!(large_structure.workspace_lens(), (6, 6));
    }

    #[test]
    fn tree_transform_group_plan_cache_key_tracks_operation_but_not_coefficients() {
        let group_key = FusionTreeGroupKey::from_sector_ids([10, 20], [30], [false, true], [true]);
        let dst_key = fusion_tree_test_key([20, 10], [30], 7, [true, false], [true]);
        let src_key = fusion_tree_test_key([10, 20], [30], 5, [false, true], [true]);
        let plan_a = TreeTransformGroupPlan::new(vec![TreeTransformGroupBlockSpec::single(
            group_key.clone(),
            dst_key.clone(),
            src_key.clone(),
            2.0_f64,
        )]);
        let plan_b = TreeTransformGroupPlan::new(vec![TreeTransformGroupBlockSpec::single(
            group_key, dst_key, src_key, 3.0_f64,
        )]);

        let transpose = TreeTransformGroupPlanKey::from_plan(
            TreeTransformOperationKey::transpose([1, 0], [0]),
            &plan_a,
        );
        let same_operation_different_coefficients = TreeTransformGroupPlanKey::from_plan(
            TreeTransformOperationKey::transpose([1, 0], [0]),
            &plan_b,
        );
        let different_permutation = TreeTransformGroupPlanKey::from_plan(
            TreeTransformOperationKey::transpose([0, 1], [0]),
            &plan_a,
        );
        let braid = TreeTransformGroupPlanKey::from_plan(
            TreeTransformOperationKey::braid([1, 0], [0], [2], [0]),
            &plan_a,
        );

        assert_eq!(transpose, same_operation_different_coefficients);
        assert_ne!(transpose, different_permutation);
        assert_ne!(transpose, braid);
    }

    #[test]
    fn tree_transform_group_block_spec_rejects_group_structure_mismatch() {
        let src_key = fusion_tree_test_key([10, 20], [30], 5, [false, true], [true]);
        let src_structure =
            BlockStructure::packed_column_major_with_keys(2, [(src_key, vec![1, 1])]).unwrap();
        let dense_structure = BlockStructure::trivial(&[1, 1]).unwrap();
        let src_groups = src_structure.fusion_tree_groups();

        let err = TreeTransformGroupBlockSpec::<f64>::from_block_groups(
            &dense_structure,
            &src_groups[0],
            &src_structure,
            &src_groups[0],
            vec![1.0],
        )
        .unwrap_err();

        assert_eq!(
            err,
            OperationError::FusionTreeGroupMismatch {
                tensor: "dst",
                index: 0,
            }
        );
    }

    #[test]
    fn tree_transform_rejects_incompatible_single_tree_shapes() {
        let src_space = TensorMapSpace::<2, 0>::from_dims([2, 2], []).unwrap();
        let dst_space = TensorMapSpace::<2, 0>::from_dims([4, 1], []).unwrap();
        let src = TensorMap::<f64, 2, 0>::from_vec(vec![1.0; 4], src_space).unwrap();
        let dst = TensorMap::<f64, 2, 0>::filled(0.0, dst_space).unwrap();

        let err = TreeTransformStructure::compile(
            &dst,
            &src,
            &[TreeTransformBlockSpec::single(0, 0, 1.0)],
        )
        .unwrap_err();

        assert_eq!(
            err,
            OperationError::ShapeMismatch {
                dst: vec![4, 1],
                src: vec![2, 2],
            }
        );
    }

    #[test]
    fn tree_transform_rejects_mismatched_multi_tree_element_count() {
        let src_space = TensorMapSpace::<2, 0>::from_dims([4, 2], []).unwrap();
        let dst_space = TensorMapSpace::<2, 0>::from_dims([3, 2], []).unwrap();
        let src_structure =
            BlockStructure::packed_column_major(2, [vec![2, 2], vec![2, 2]]).unwrap();
        let dst_structure =
            BlockStructure::packed_column_major(2, [vec![3, 1], vec![3, 1]]).unwrap();
        let src =
            TensorMap::<f64, 2, 0>::from_vec_with_structure(vec![1.0; 8], src_space, src_structure)
                .unwrap();
        let dst =
            TensorMap::<f64, 2, 0>::from_vec_with_structure(vec![0.0; 6], dst_space, dst_structure)
                .unwrap();

        let err = TreeTransformStructure::compile(
            &dst,
            &src,
            &[TreeTransformBlockSpec::multi(
                vec![0, 1],
                vec![0, 1],
                vec![1.0, 0.0, 0.0, 1.0],
            )],
        )
        .unwrap_err();

        assert_eq!(
            err,
            OperationError::ElementCountMismatch {
                expected: 3,
                actual: 4,
            }
        );
    }
}
