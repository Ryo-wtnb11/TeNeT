#![forbid(unsafe_code)]

//! TensorOperations-style lowering for TeNeT.
//!
//! Public/core tensor code talks in terms of TeNeT-owned block views. This crate
//! lowers those views to strided-rs kernels at the same granularity that
//! TensorKit uses Strided.jl/StridedViews.jl internally.

use core::ops::{Add, Mul};
use std::collections::HashMap;
use std::hash::Hash;
use std::sync::Arc;

use num_complex::{Complex32, Complex64};
use num_traits::{One, Zero};
use tenet_core::{
    multiplicity_free_braid_tree, multiplicity_free_braid_tree_pair,
    multiplicity_free_permute_tree, multiplicity_free_permute_tree_pair,
    multiplicity_free_transpose_tree_pair, BlockKey, BlockLayout, BlockStructure, BlockView,
    BlockViewMut, CoreError, FusionRule, FusionTensorMapSpace, FusionTreeBlockGroup,
    FusionTreeBlockKey, FusionTreeGroupKey, FusionTreeHomSpace, FusionTreeKey,
    MultiplicityFreeFusionSymbols, MultiplicityFreeRigidSymbols, SectorId, TensorMap,
};
#[cfg(test)]
use tenet_core::{
    unique_braid_tree, unique_braid_tree_pair, unique_permute_tree, unique_permute_tree_pair,
    unique_transpose_tree_pair, FusionStyleKind, MultiplicityFreePivotalSymbols,
};
use tenet_dense::{
    DefaultDenseExecutor, DenseDotConfig, DenseExecutor, DenseRead, DenseView, DenseViewMut,
    DenseWrite,
};

mod axis;
mod error;
mod tree_transform;

pub use axis::{AxisPermutation, OwnedTensorContractAxisSpec, TensorContractAxisSpec};
pub use error::OperationError;
pub use tree_transform::{
    TreeTransformBuiltinRuleCacheKey, TreeTransformOperationKey, TreeTransformPlanScope,
    TreeTransformProductRuleCacheKey, TreeTransformRuleCacheKey,
};

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

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TensorContractFusionExplicitPlan {
    lhs_transform: TreeTransformOperationKey,
    rhs_transform: TreeTransformOperationKey,
    output_transform: TreeTransformOperationKey,
    canonical_axes: OwnedTensorContractAxisSpec,
    canonical_dst_nout: usize,
    canonical_dst_nin: usize,
    lhs_canonical_nout: usize,
    lhs_canonical_nin: usize,
    rhs_canonical_nout: usize,
    rhs_canonical_nin: usize,
}

impl TensorContractFusionExplicitPlan {
    #[inline]
    pub fn lhs_transform(&self) -> &TreeTransformOperationKey {
        &self.lhs_transform
    }

    #[inline]
    pub fn rhs_transform(&self) -> &TreeTransformOperationKey {
        &self.rhs_transform
    }

    #[inline]
    pub fn output_transform(&self) -> &TreeTransformOperationKey {
        &self.output_transform
    }

    #[inline]
    pub fn canonical_axes(&self) -> &OwnedTensorContractAxisSpec {
        &self.canonical_axes
    }

    #[inline]
    pub fn canonical_dst_nout(&self) -> usize {
        self.canonical_dst_nout
    }

    #[inline]
    pub fn canonical_dst_nin(&self) -> usize {
        self.canonical_dst_nin
    }

    fn output_transform_is_identity(&self) -> bool {
        let canonical_rank = self.canonical_dst_nout + self.canonical_dst_nin;
        match &self.output_transform {
            TreeTransformOperationKey::Permute {
                codomain_permutation,
                domain_permutation,
            } => {
                codomain_permutation
                    .iter()
                    .copied()
                    .eq(0..self.canonical_dst_nout)
                    && domain_permutation
                        .iter()
                        .copied()
                        .eq(self.canonical_dst_nout..canonical_rank)
            }
            _ => false,
        }
    }

    #[inline]
    pub fn lhs_canonical_nout(&self) -> usize {
        self.lhs_canonical_nout
    }

    #[inline]
    pub fn lhs_canonical_nin(&self) -> usize {
        self.lhs_canonical_nin
    }

    #[inline]
    pub fn rhs_canonical_nout(&self) -> usize {
        self.rhs_canonical_nout
    }

    #[inline]
    pub fn rhs_canonical_nin(&self) -> usize {
        self.rhs_canonical_nin
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct TensorContractStructure<C = f64> {
    dst_rank: usize,
    lhs_rank: usize,
    rhs_rank: usize,
    lhs_contracting_axes: Vec<usize>,
    rhs_contracting_axes: Vec<usize>,
    output_axes: Vec<usize>,
    terms: Vec<TensorContractStructureTerm<C>>,
    descriptor: TensorContractDescriptor<C>,
    dst_structure: Arc<BlockStructure>,
    lhs_structure: Arc<BlockStructure>,
    rhs_structure: Arc<BlockStructure>,
}

pub fn tensorcontract_structure<
    TDst,
    TLhs,
    TRhs,
    const DST_NOUT: usize,
    const DST_NIN: usize,
    const LHS_NOUT: usize,
    const LHS_NIN: usize,
    const RHS_NOUT: usize,
    const RHS_NIN: usize,
    SDst,
    SLhs,
    SRhs,
>(
    dst: &TensorMap<TDst, DST_NOUT, DST_NIN, SDst>,
    lhs: &TensorMap<TLhs, LHS_NOUT, LHS_NIN, SLhs>,
    rhs: &TensorMap<TRhs, RHS_NOUT, RHS_NIN, SRhs>,
    axes: TensorContractAxisSpec<'_>,
) -> Result<TensorContractStructure, OperationError> {
    TensorContractStructure::compile(dst, lhs, rhs, axes)
}

pub fn tensorcontract_fusion_structure<
    R,
    TDst,
    TLhs,
    TRhs,
    const DST_NOUT: usize,
    const DST_NIN: usize,
    const LHS_NOUT: usize,
    const LHS_NIN: usize,
    const RHS_NOUT: usize,
    const RHS_NIN: usize,
    SDst,
    SLhs,
    SRhs,
>(
    rule: &R,
    dst: &TensorMap<TDst, DST_NOUT, DST_NIN, SDst>,
    lhs: &TensorMap<TLhs, LHS_NOUT, LHS_NIN, SLhs>,
    rhs: &TensorMap<TRhs, RHS_NOUT, RHS_NIN, SRhs>,
    axes: TensorContractAxisSpec<'_>,
) -> Result<TensorContractStructure, OperationError>
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64>,
{
    let dst_fusion = dst
        .fusion_space()
        .ok_or(OperationError::Core(CoreError::MissingFusionSpace))?;
    let lhs_fusion = lhs
        .fusion_space()
        .ok_or(OperationError::Core(CoreError::MissingFusionSpace))?;
    let rhs_fusion = rhs
        .fusion_space()
        .ok_or(OperationError::Core(CoreError::MissingFusionSpace))?;
    let block_specs =
        tensorcontract_fusion_block_specs(rule, dst_fusion, lhs_fusion, rhs_fusion, axes)?;
    TensorContractStructure::compile_with_block_specs(dst, lhs, rhs, axes, &block_specs)
}

pub fn tensorcontract_fusion_block_specs<
    R,
    const DST_NOUT: usize,
    const DST_NIN: usize,
    const LHS_NOUT: usize,
    const LHS_NIN: usize,
    const RHS_NOUT: usize,
    const RHS_NIN: usize,
>(
    rule: &R,
    dst: &FusionTensorMapSpace<DST_NOUT, DST_NIN>,
    lhs: &FusionTensorMapSpace<LHS_NOUT, LHS_NIN>,
    rhs: &FusionTensorMapSpace<RHS_NOUT, RHS_NIN>,
    axes: TensorContractAxisSpec<'_>,
) -> Result<Vec<TensorContractBlockSpec>, OperationError>
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64>,
{
    let axis_plan = TensorContractAxisPlan::compile(
        lhs.subblock_structure().rank(),
        rhs.subblock_structure().rank(),
        dst.subblock_structure().rank(),
        axes,
    )?;
    let expected_homspace = FusionTreeHomSpace::tensorcontract_homspace(
        rule,
        lhs.homspace(),
        rhs.homspace(),
        axes.lhs_contracting_axes(),
        axes.rhs_contracting_axes(),
        axis_plan.output_axes.as_slice(),
        DST_NOUT,
    )
    .map_err(OperationError::from_core_preserving_context)?;
    if &expected_homspace != dst.homspace() {
        return Err(OperationError::StructureMismatch { tensor: "dst" });
    }
    if !is_canonical_fusion_source_contract(
        lhs.homspace(),
        rhs.homspace(),
        axis_plan.lhs_contracting_axes.as_slice(),
        axis_plan.rhs_contracting_axes.as_slice(),
    ) {
        return Err(OperationError::UnsupportedTensorContractScope {
            message: "fusion contraction requiring source tree-pair transforms is not implemented; pre-transform operands explicitly",
        });
    }
    if is_canonical_fusion_compose_contract(
        lhs.homspace(),
        rhs.homspace(),
        axis_plan.lhs_contracting_axes.as_slice(),
        axis_plan.rhs_contracting_axes.as_slice(),
        axis_plan.output_axes.as_slice(),
        DST_NOUT,
    ) {
        return tensorcontract_canonical_fusion_block_specs(rule, dst, lhs, rhs, &axis_plan);
    }

    tensorcontract_transformed_fusion_block_specs(rule, dst, lhs, rhs, &axis_plan, DST_NOUT)
}

pub fn tensorcontract_fusion_explicit_plan<
    R,
    const DST_NOUT: usize,
    const DST_NIN: usize,
    const LHS_NOUT: usize,
    const LHS_NIN: usize,
    const RHS_NOUT: usize,
    const RHS_NIN: usize,
>(
    rule: &R,
    dst: &FusionTensorMapSpace<DST_NOUT, DST_NIN>,
    lhs: &FusionTensorMapSpace<LHS_NOUT, LHS_NIN>,
    rhs: &FusionTensorMapSpace<RHS_NOUT, RHS_NIN>,
    axes: TensorContractAxisSpec<'_>,
) -> Result<TensorContractFusionExplicitPlan, OperationError>
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64>,
{
    let axis_plan = TensorContractAxisPlan::compile(
        lhs.subblock_structure().rank(),
        rhs.subblock_structure().rank(),
        dst.subblock_structure().rank(),
        axes,
    )?;
    let expected_homspace = FusionTreeHomSpace::tensorcontract_homspace(
        rule,
        lhs.homspace(),
        rhs.homspace(),
        axes.lhs_contracting_axes(),
        axes.rhs_contracting_axes(),
        axis_plan.output_axes.as_slice(),
        DST_NOUT,
    )
    .map_err(OperationError::from_core_preserving_context)?;
    if &expected_homspace != dst.homspace() {
        return Err(OperationError::StructureMismatch { tensor: "dst" });
    }

    let lhs_canonical_nout = axis_plan.lhs_open_axes.len();
    let lhs_canonical_nin = axis_plan.lhs_contracting_axes.len();
    let rhs_canonical_nout = axis_plan.rhs_contracting_axes.len();
    let rhs_canonical_nin = axis_plan.rhs_open_axes.len();
    let canonical_dst_nout = lhs_canonical_nout;
    let canonical_dst_nin = rhs_canonical_nin;
    let canonical_output_rank = canonical_dst_nout + canonical_dst_nin;
    let output_transform = TreeTransformOperationKey::permute(
        axis_plan.output_axes[..DST_NOUT].to_vec(),
        axis_plan.output_axes[DST_NOUT..].to_vec(),
    );
    Ok(TensorContractFusionExplicitPlan {
        lhs_transform: TreeTransformOperationKey::permute(
            axis_plan.lhs_open_axes,
            axis_plan.lhs_contracting_axes,
        ),
        rhs_transform: TreeTransformOperationKey::permute(
            axis_plan.rhs_contracting_axes,
            axis_plan.rhs_open_axes,
        ),
        canonical_axes: OwnedTensorContractAxisSpec::new(
            (lhs_canonical_nout..lhs_canonical_nout + lhs_canonical_nin).collect(),
            (0..rhs_canonical_nout).collect(),
            (0..canonical_output_rank).collect(),
        ),
        output_transform,
        canonical_dst_nout,
        canonical_dst_nin,
        lhs_canonical_nout,
        lhs_canonical_nin,
        rhs_canonical_nout,
        rhs_canonical_nin,
    })
}

fn tensorcontract_canonical_fusion_block_specs<
    R,
    const DST_NOUT: usize,
    const DST_NIN: usize,
    const LHS_NOUT: usize,
    const LHS_NIN: usize,
    const RHS_NOUT: usize,
    const RHS_NIN: usize,
>(
    rule: &R,
    dst: &FusionTensorMapSpace<DST_NOUT, DST_NIN>,
    lhs: &FusionTensorMapSpace<LHS_NOUT, LHS_NIN>,
    rhs: &FusionTensorMapSpace<RHS_NOUT, RHS_NIN>,
    axis_plan: &TensorContractAxisPlan,
) -> Result<Vec<TensorContractBlockSpec>, OperationError>
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64>,
{
    let mut specs = Vec::new();
    for lhs_index in 0..lhs.subblock_structure().block_count() {
        let lhs_block = lhs.subblock_structure().block(lhs_index)?;
        let BlockKey::FusionTree(lhs_key) = lhs_block.key() else {
            continue;
        };
        let lhs_external = lhs_key.external_sectors(rule);
        for rhs_index in 0..rhs.subblock_structure().block_count() {
            let rhs_block = rhs.subblock_structure().block(rhs_index)?;
            let BlockKey::FusionTree(rhs_key) = rhs_block.key() else {
                continue;
            };
            let rhs_external = rhs_key.external_sectors(rule);
            if !contracted_external_sectors_match(
                &lhs_external,
                &rhs_external,
                axis_plan.lhs_contracting_axes.as_slice(),
                axis_plan.rhs_contracting_axes.as_slice(),
            ) {
                continue;
            }
            if !contracted_fusion_tree_basis_matches(
                rule,
                lhs_key.domain_tree(),
                rhs_key.codomain_tree(),
            ) {
                continue;
            }
            let dst_key = FusionTreeBlockKey::pair(
                lhs_key.codomain_tree().clone(),
                rhs_key.domain_tree().clone(),
            );
            let dst_external = dst_key.external_sectors(rule);
            let expected_external =
                contracted_output_external_sectors(&lhs_external, &rhs_external, &axis_plan);
            if dst_external != expected_external {
                return Err(OperationError::StructureMismatch { tensor: "dst" });
            }
            let dst_keys = dst
                .homspace()
                .fusion_tree_keys_from_external_sectors(rule, &dst_external)
                .map_err(OperationError::from_core_preserving_context)?;
            if !dst_keys.contains(&dst_key) {
                return Err(OperationError::MissingBlockKey {
                    key: BlockKey::from(dst_key),
                });
            }
            let dst_index = dst.find_subblock_index(&dst_key).ok_or_else(|| {
                OperationError::MissingBlockKey {
                    key: BlockKey::from(dst_key.clone()),
                }
            })?;
            specs.push(TensorContractBlockSpec::with_coefficient(
                dst_index,
                lhs_index,
                rhs_index,
                rule.scalar_one(),
            ));
        }
    }
    Ok(specs)
}

fn tensorcontract_transformed_fusion_block_specs<
    R,
    const DST_NOUT: usize,
    const DST_NIN: usize,
    const LHS_NOUT: usize,
    const LHS_NIN: usize,
    const RHS_NOUT: usize,
    const RHS_NIN: usize,
>(
    rule: &R,
    dst: &FusionTensorMapSpace<DST_NOUT, DST_NIN>,
    lhs: &FusionTensorMapSpace<LHS_NOUT, LHS_NIN>,
    rhs: &FusionTensorMapSpace<RHS_NOUT, RHS_NIN>,
    axis_plan: &TensorContractAxisPlan,
    dst_codomain_rank: usize,
) -> Result<Vec<TensorContractBlockSpec>, OperationError>
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64>,
{
    let output_codomain_axes = &axis_plan.output_axes[..dst_codomain_rank];
    let output_domain_axes = &axis_plan.output_axes[dst_codomain_rank..];
    let mut specs = Vec::new();
    for lhs_index in 0..lhs.subblock_structure().block_count() {
        let lhs_block = lhs.subblock_structure().block(lhs_index)?;
        let BlockKey::FusionTree(lhs_key) = lhs_block.key() else {
            continue;
        };
        let lhs_terms = multiplicity_free_permute_tree_pair(
            rule,
            lhs_key,
            axis_plan.lhs_open_axes.as_slice(),
            axis_plan.lhs_contracting_axes.as_slice(),
        )
        .map_err(OperationError::from_core_preserving_context)?;
        for rhs_index in 0..rhs.subblock_structure().block_count() {
            let rhs_block = rhs.subblock_structure().block(rhs_index)?;
            let BlockKey::FusionTree(rhs_key) = rhs_block.key() else {
                continue;
            };
            let rhs_terms = multiplicity_free_permute_tree_pair(
                rule,
                rhs_key,
                axis_plan.rhs_contracting_axes.as_slice(),
                axis_plan.rhs_open_axes.as_slice(),
            )
            .map_err(OperationError::from_core_preserving_context)?;

            for (lhs_canonical, lhs_coeff) in &lhs_terms {
                for (rhs_canonical, rhs_coeff) in &rhs_terms {
                    if !contracted_fusion_tree_basis_matches(
                        rule,
                        lhs_canonical.domain_tree(),
                        rhs_canonical.codomain_tree(),
                    ) {
                        continue;
                    }
                    let canonical_dst_key = FusionTreeBlockKey::pair(
                        lhs_canonical.codomain_tree().clone(),
                        rhs_canonical.domain_tree().clone(),
                    );
                    let dst_terms = multiplicity_free_permute_tree_pair(
                        rule,
                        &canonical_dst_key,
                        output_codomain_axes,
                        output_domain_axes,
                    )
                    .map_err(OperationError::from_core_preserving_context)?;
                    for (dst_key, dst_coeff) in dst_terms {
                        let dst_index = dst.find_subblock_index(&dst_key).ok_or_else(|| {
                            OperationError::MissingBlockKey {
                                key: BlockKey::from(dst_key.clone()),
                            }
                        })?;
                        specs.push(TensorContractBlockSpec::with_coefficient(
                            dst_index,
                            lhs_index,
                            rhs_index,
                            *lhs_coeff * *rhs_coeff * dst_coeff,
                        ));
                    }
                }
            }
        }
    }
    Ok(specs)
}

impl TensorContractStructure {
    pub fn compile<
        TDst,
        TLhs,
        TRhs,
        const DST_NOUT: usize,
        const DST_NIN: usize,
        const LHS_NOUT: usize,
        const LHS_NIN: usize,
        const RHS_NOUT: usize,
        const RHS_NIN: usize,
        SDst,
        SLhs,
        SRhs,
    >(
        dst: &TensorMap<TDst, DST_NOUT, DST_NIN, SDst>,
        lhs: &TensorMap<TLhs, LHS_NOUT, LHS_NIN, SLhs>,
        rhs: &TensorMap<TRhs, RHS_NOUT, RHS_NIN, SRhs>,
        axes: TensorContractAxisSpec<'_>,
    ) -> Result<Self, OperationError> {
        Self::compile_shared_structures(
            Arc::clone(dst.structure()),
            Arc::clone(lhs.structure()),
            Arc::clone(rhs.structure()),
            axes,
        )
    }

    pub fn compile_structures(
        dst_structure: &BlockStructure,
        lhs_structure: &BlockStructure,
        rhs_structure: &BlockStructure,
        axes: TensorContractAxisSpec<'_>,
    ) -> Result<Self, OperationError> {
        Self::compile_shared_structures(
            Arc::new(dst_structure.clone()),
            Arc::new(lhs_structure.clone()),
            Arc::new(rhs_structure.clone()),
            axes,
        )
    }

    pub fn compile_with_block_specs<
        TDst,
        TLhs,
        TRhs,
        const DST_NOUT: usize,
        const DST_NIN: usize,
        const LHS_NOUT: usize,
        const LHS_NIN: usize,
        const RHS_NOUT: usize,
        const RHS_NIN: usize,
        SDst,
        SLhs,
        SRhs,
    >(
        dst: &TensorMap<TDst, DST_NOUT, DST_NIN, SDst>,
        lhs: &TensorMap<TLhs, LHS_NOUT, LHS_NIN, SLhs>,
        rhs: &TensorMap<TRhs, RHS_NOUT, RHS_NIN, SRhs>,
        axes: TensorContractAxisSpec<'_>,
        block_specs: &[TensorContractBlockSpec],
    ) -> Result<Self, OperationError> {
        Self::compile_shared_structures_with_block_specs(
            Arc::clone(dst.structure()),
            Arc::clone(lhs.structure()),
            Arc::clone(rhs.structure()),
            axes,
            block_specs,
        )
    }

    pub fn compile_structures_with_block_specs(
        dst_structure: &BlockStructure,
        lhs_structure: &BlockStructure,
        rhs_structure: &BlockStructure,
        axes: TensorContractAxisSpec<'_>,
        block_specs: &[TensorContractBlockSpec],
    ) -> Result<Self, OperationError> {
        Self::compile_shared_structures_with_block_specs(
            Arc::new(dst_structure.clone()),
            Arc::new(lhs_structure.clone()),
            Arc::new(rhs_structure.clone()),
            axes,
            block_specs,
        )
    }

    fn compile_shared_structures(
        dst_structure: Arc<BlockStructure>,
        lhs_structure: Arc<BlockStructure>,
        rhs_structure: Arc<BlockStructure>,
        axes: TensorContractAxisSpec<'_>,
    ) -> Result<Self, OperationError> {
        if dst_structure.block_count() != 1
            || lhs_structure.block_count() != 1
            || rhs_structure.block_count() != 1
        {
            return Err(OperationError::UnsupportedTensorContractScope {
                message: "block-sparse contraction enumeration is not implemented yet",
            });
        }

        let block_specs = [TensorContractBlockSpec::new(0, 0, 0)];
        Self::compile_shared_structures_with_block_specs(
            dst_structure,
            lhs_structure,
            rhs_structure,
            axes,
            &block_specs,
        )
    }

    fn compile_shared_structures_with_block_specs(
        dst_structure: Arc<BlockStructure>,
        lhs_structure: Arc<BlockStructure>,
        rhs_structure: Arc<BlockStructure>,
        axes: TensorContractAxisSpec<'_>,
        block_specs: &[TensorContractBlockSpec],
    ) -> Result<Self, OperationError> {
        let dst_rank = dst_structure.rank();
        let lhs_rank = lhs_structure.rank();
        let rhs_rank = rhs_structure.rank();
        let axis_plan = TensorContractAxisPlan::compile(lhs_rank, rhs_rank, dst_rank, axes)?;
        let mut terms = Vec::with_capacity(block_specs.len());
        for spec in block_specs {
            validate_block_index("dst", spec.dst_block(), dst_structure.block_count())?;
            validate_block_index("lhs", spec.lhs_block(), lhs_structure.block_count())?;
            validate_block_index("rhs", spec.rhs_block(), rhs_structure.block_count())?;
            let dst_block = dst_structure.block(spec.dst_block())?;
            terms.push(TensorContractStructureTerm {
                key: dst_block.key().clone(),
                dst_block: spec.dst_block(),
                lhs_block: spec.lhs_block(),
                rhs_block: spec.rhs_block(),
                coefficient: spec.coefficient(),
            });
        }
        let descriptor = TensorContractDescriptor::compile(
            &axis_plan,
            &terms,
            &dst_structure,
            &lhs_structure,
            &rhs_structure,
        )?;

        Ok(Self {
            dst_rank,
            lhs_rank,
            rhs_rank,
            lhs_contracting_axes: axis_plan.lhs_contracting_axes,
            rhs_contracting_axes: axis_plan.rhs_contracting_axes,
            output_axes: axis_plan.output_axes,
            terms,
            descriptor,
            dst_structure,
            lhs_structure,
            rhs_structure,
        })
    }
}

impl<C> TensorContractStructure<C>
where
    C: Copy + One,
{
    #[inline]
    pub fn dst_rank(&self) -> usize {
        self.dst_rank
    }

    #[inline]
    pub fn lhs_rank(&self) -> usize {
        self.lhs_rank
    }

    #[inline]
    pub fn rhs_rank(&self) -> usize {
        self.rhs_rank
    }

    #[inline]
    pub fn lhs_contracting_axes(&self) -> &[usize] {
        &self.lhs_contracting_axes
    }

    #[inline]
    pub fn rhs_contracting_axes(&self) -> &[usize] {
        &self.rhs_contracting_axes
    }

    /// Destination-axis to canonical-output-axis mapping. The canonical output
    /// order is TensorOperations' `(lhs open axes..., rhs open axes...)`.
    #[inline]
    pub fn output_axes(&self) -> &[usize] {
        &self.output_axes
    }

    #[inline]
    pub fn terms(&self) -> &[TensorContractStructureTerm<C>] {
        &self.terms
    }

    #[inline]
    fn descriptor(&self) -> &TensorContractDescriptor<C> {
        &self.descriptor
    }

    fn validate_replay_structures(
        &self,
        dst_structure: &Arc<BlockStructure>,
        lhs_structure: &Arc<BlockStructure>,
        rhs_structure: &Arc<BlockStructure>,
    ) -> Result<(), OperationError> {
        validate_structure_identity("dst", &self.dst_structure, dst_structure)?;
        validate_structure_identity("lhs", &self.lhs_structure, lhs_structure)?;
        validate_structure_identity("rhs", &self.rhs_structure, rhs_structure)
    }

    pub fn execute_with<
        B,
        D,
        const DST_NOUT: usize,
        const DST_NIN: usize,
        const LHS_NOUT: usize,
        const LHS_NIN: usize,
        const RHS_NOUT: usize,
        const RHS_NIN: usize,
        SDst,
        SLhs,
        SRhs,
    >(
        &self,
        backend: &mut B,
        workspace: &mut B::Workspace,
        dst: &mut TensorMap<D, DST_NOUT, DST_NIN, SDst>,
        lhs: &TensorMap<D, LHS_NOUT, LHS_NIN, SLhs>,
        rhs: &TensorMap<D, RHS_NOUT, RHS_NIN, SRhs>,
        alpha: D,
        beta: D,
    ) -> Result<(), OperationError>
    where
        B: TensorContractBackend<D, C>,
        D: DenseBlockScalar + RecouplingCoefficientAction<C>,
    {
        backend.tensorcontract_structure_into(workspace, self, dst, lhs, rhs, alpha, beta)
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TensorContractBlockSpec<C = f64> {
    dst_block: usize,
    lhs_block: usize,
    rhs_block: usize,
    coefficient: C,
}

impl<C> TensorContractBlockSpec<C>
where
    C: One,
{
    pub fn new(dst_block: usize, lhs_block: usize, rhs_block: usize) -> Self {
        Self::with_coefficient(dst_block, lhs_block, rhs_block, C::one())
    }
}

impl<C> TensorContractBlockSpec<C> {
    pub const fn with_coefficient(
        dst_block: usize,
        lhs_block: usize,
        rhs_block: usize,
        coefficient: C,
    ) -> Self {
        Self {
            dst_block,
            lhs_block,
            rhs_block,
            coefficient,
        }
    }

    #[inline]
    pub fn dst_block(&self) -> usize {
        self.dst_block
    }

    #[inline]
    pub fn lhs_block(&self) -> usize {
        self.lhs_block
    }

    #[inline]
    pub fn rhs_block(&self) -> usize {
        self.rhs_block
    }

    #[inline]
    pub fn coefficient(&self) -> C
    where
        C: Copy,
    {
        self.coefficient
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct TensorContractStructureTerm<C = f64> {
    key: BlockKey,
    dst_block: usize,
    lhs_block: usize,
    rhs_block: usize,
    coefficient: C,
}

impl<C> TensorContractStructureTerm<C> {
    #[inline]
    pub fn key(&self) -> &BlockKey {
        &self.key
    }

    #[inline]
    pub fn dst_block(&self) -> usize {
        self.dst_block
    }

    #[inline]
    pub fn lhs_block(&self) -> usize {
        self.lhs_block
    }

    #[inline]
    pub fn rhs_block(&self) -> usize {
        self.rhs_block
    }

    #[inline]
    pub fn coefficient(&self) -> C
    where
        C: Copy,
    {
        self.coefficient
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct TensorContractAxisPlan {
    lhs_contracting_axes: Vec<usize>,
    rhs_contracting_axes: Vec<usize>,
    lhs_open_axes: Vec<usize>,
    rhs_open_axes: Vec<usize>,
    output_axes: Vec<usize>,
}

impl TensorContractAxisPlan {
    fn compile(
        lhs_rank: usize,
        rhs_rank: usize,
        dst_rank: usize,
        axes: TensorContractAxisSpec<'_>,
    ) -> Result<Self, OperationError> {
        if axes.lhs_contracting_axes().len() != axes.rhs_contracting_axes().len() {
            return Err(OperationError::ContractAxisCountMismatch {
                lhs: axes.lhs_contracting_axes().len(),
                rhs: axes.rhs_contracting_axes().len(),
            });
        }
        let lhs_seen = validate_axis_subset("lhs", axes.lhs_contracting_axes(), lhs_rank)?;
        let rhs_seen = validate_axis_subset("rhs", axes.rhs_contracting_axes(), rhs_rank)?;

        let lhs_open_axes = (0..lhs_rank)
            .filter(|&axis| !lhs_seen[axis])
            .collect::<Vec<_>>();
        let rhs_open_axes = (0..rhs_rank)
            .filter(|&axis| !rhs_seen[axis])
            .collect::<Vec<_>>();
        let canonical_output_rank = lhs_open_axes.len() + rhs_open_axes.len();

        let output_axes = permutation_axes(axes.output_permutation(), canonical_output_rank)?;
        if output_axes.len() != dst_rank {
            return Err(OperationError::StructureRankMismatch {
                expected: output_axes.len(),
                actual: dst_rank,
            });
        }

        Ok(Self {
            lhs_contracting_axes: axes.lhs_contracting_axes().to_vec(),
            rhs_contracting_axes: axes.rhs_contracting_axes().to_vec(),
            lhs_open_axes,
            rhs_open_axes,
            output_axes,
        })
    }
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct TensorContractDescriptor<C = f64> {
    dot_config: DenseDotConfig,
    terms: Vec<TensorContractDescriptorTerm<C>>,
    lhs_shapes: Vec<usize>,
    lhs_strides: Vec<usize>,
    rhs_shapes: Vec<usize>,
    rhs_strides: Vec<usize>,
    output_shapes: Vec<usize>,
    output_strides: Vec<usize>,
    scatter_shapes: Vec<usize>,
    dst_strides: Vec<isize>,
    workspace_strides: Vec<isize>,
}

impl<C> TensorContractDescriptor<C>
where
    C: Copy + One,
{
    #[inline]
    pub fn terms(&self) -> &[TensorContractDescriptorTerm<C>] {
        &self.terms
    }

    #[inline]
    fn dot_config(&self) -> &DenseDotConfig {
        &self.dot_config
    }

    fn lhs_shape(&self, term: &TensorContractDescriptorTerm<C>) -> &[usize] {
        &self.lhs_shapes[term.lhs_layout_start..term.lhs_layout_start + term.lhs_rank]
    }

    fn lhs_strides(&self, term: &TensorContractDescriptorTerm<C>) -> &[usize] {
        &self.lhs_strides[term.lhs_layout_start..term.lhs_layout_start + term.lhs_rank]
    }

    fn rhs_shape(&self, term: &TensorContractDescriptorTerm<C>) -> &[usize] {
        &self.rhs_shapes[term.rhs_layout_start..term.rhs_layout_start + term.rhs_rank]
    }

    fn rhs_strides(&self, term: &TensorContractDescriptorTerm<C>) -> &[usize] {
        &self.rhs_strides[term.rhs_layout_start..term.rhs_layout_start + term.rhs_rank]
    }

    fn output_shape(&self, term: &TensorContractDescriptorTerm<C>) -> &[usize] {
        &self.output_shapes[term.output_layout_start..term.output_layout_start + term.output_rank]
    }

    fn output_strides(&self, term: &TensorContractDescriptorTerm<C>) -> &[usize] {
        &self.output_strides[term.output_layout_start..term.output_layout_start + term.output_rank]
    }

    fn scatter_shape(&self, term: &TensorContractDescriptorTerm<C>) -> &[usize] {
        &self.scatter_shapes
            [term.scatter_layout_start..term.scatter_layout_start + term.output_rank]
    }

    fn dst_strides(&self, term: &TensorContractDescriptorTerm<C>) -> &[isize] {
        &self.dst_strides[term.scatter_layout_start..term.scatter_layout_start + term.output_rank]
    }

    fn workspace_strides(&self, term: &TensorContractDescriptorTerm<C>) -> &[isize] {
        &self.workspace_strides
            [term.scatter_layout_start..term.scatter_layout_start + term.output_rank]
    }

    fn compile(
        axis_plan: &TensorContractAxisPlan,
        terms: &[TensorContractStructureTerm<C>],
        dst_structure: &BlockStructure,
        lhs_structure: &BlockStructure,
        rhs_structure: &BlockStructure,
    ) -> Result<Self, OperationError> {
        let mut descriptor = Self {
            dot_config: DenseDotConfig::new(
                axis_plan.lhs_contracting_axes.clone(),
                axis_plan.rhs_contracting_axes.clone(),
                Vec::new(),
                Vec::new(),
            ),
            terms: Vec::new(),
            lhs_shapes: Vec::new(),
            lhs_strides: Vec::new(),
            rhs_shapes: Vec::new(),
            rhs_strides: Vec::new(),
            output_shapes: Vec::new(),
            output_strides: Vec::new(),
            scatter_shapes: Vec::new(),
            dst_strides: Vec::new(),
            workspace_strides: Vec::new(),
        };

        let lhs_rank = lhs_structure.rank();
        let rhs_rank = rhs_structure.rank();
        let output_rank = dst_structure.rank();
        descriptor.terms.reserve(terms.len());
        descriptor.lhs_shapes.reserve(terms.len() * lhs_rank);
        descriptor.lhs_strides.reserve(terms.len() * lhs_rank);
        descriptor.rhs_shapes.reserve(terms.len() * rhs_rank);
        descriptor.rhs_strides.reserve(terms.len() * rhs_rank);
        descriptor.output_shapes.reserve(terms.len() * output_rank);
        descriptor.output_strides.reserve(terms.len() * output_rank);
        descriptor.scatter_shapes.reserve(terms.len() * output_rank);
        descriptor.dst_strides.reserve(terms.len() * output_rank);
        descriptor
            .workspace_strides
            .reserve(terms.len() * output_rank);

        let mut seen_dst_blocks = Vec::<usize>::new();
        for term in terms {
            let lhs_block = lhs_structure.block(term.lhs_block())?;
            let rhs_block = rhs_structure.block(term.rhs_block())?;
            let dst_block = dst_structure.block(term.dst_block())?;
            let lhs_layout_start = descriptor.lhs_shapes.len();
            descriptor.lhs_shapes.extend_from_slice(lhs_block.shape());
            descriptor
                .lhs_strides
                .extend_from_slice(lhs_block.strides());
            let rhs_layout_start = descriptor.rhs_shapes.len();
            descriptor.rhs_shapes.extend_from_slice(rhs_block.shape());
            descriptor
                .rhs_strides
                .extend_from_slice(rhs_block.strides());

            let lhs_contract_shape = axis_plan
                .lhs_contracting_axes
                .iter()
                .map(|&axis| lhs_block.shape()[axis])
                .collect::<Vec<_>>();
            let rhs_contract_shape = axis_plan
                .rhs_contracting_axes
                .iter()
                .map(|&axis| rhs_block.shape()[axis])
                .collect::<Vec<_>>();
            if lhs_contract_shape != rhs_contract_shape {
                return Err(OperationError::ShapeMismatch {
                    dst: lhs_contract_shape,
                    src: rhs_contract_shape,
                });
            }
            let output_shape = axis_plan
                .lhs_open_axes
                .iter()
                .map(|&axis| lhs_block.shape()[axis])
                .chain(
                    axis_plan
                        .rhs_open_axes
                        .iter()
                        .map(|&axis| rhs_block.shape()[axis]),
                )
                .collect::<Vec<_>>();
            let output_strides = column_major_strides_usize(&output_shape)?;
            let workspace_len = element_count(&output_shape)?;
            let output_layout_start = descriptor.output_shapes.len();
            descriptor.output_shapes.extend_from_slice(&output_shape);
            descriptor.output_strides.extend_from_slice(&output_strides);

            let scatter_shape = axis_plan
                .output_axes
                .iter()
                .map(|&axis| output_shape[axis])
                .collect::<Vec<_>>();
            if dst_block.shape() != scatter_shape.as_slice() {
                return Err(OperationError::ShapeMismatch {
                    dst: dst_block.shape().to_vec(),
                    src: scatter_shape,
                });
            }
            let scatter_layout_start = descriptor.scatter_shapes.len();
            descriptor
                .scatter_shapes
                .extend_from_slice(dst_block.shape());
            for (dst_axis, &workspace_axis) in axis_plan.output_axes.iter().enumerate() {
                descriptor.dst_strides.push(
                    isize::try_from(dst_block.strides()[dst_axis]).map_err(|_| {
                        OperationError::StrideOverflow {
                            value: dst_block.strides()[dst_axis],
                        }
                    })?,
                );
                descriptor.workspace_strides.push(
                    isize::try_from(output_strides[workspace_axis]).map_err(|_| {
                        OperationError::StrideOverflow {
                            value: output_strides[workspace_axis],
                        }
                    })?,
                );
            }
            let apply_beta = !seen_dst_blocks.contains(&term.dst_block());
            if apply_beta {
                seen_dst_blocks.push(term.dst_block());
            }
            descriptor.terms.push(TensorContractDescriptorTerm {
                dst_block: term.dst_block(),
                lhs_block: term.lhs_block(),
                rhs_block: term.rhs_block(),
                lhs_layout_start,
                rhs_layout_start,
                output_layout_start,
                scatter_layout_start,
                lhs_rank,
                rhs_rank,
                output_rank,
                lhs_offset: lhs_block.offset(),
                rhs_offset: rhs_block.offset(),
                dst_offset: offset_to_isize(dst_block.offset())?,
                workspace_len,
                apply_beta,
                coefficient: term.coefficient(),
            });
        }

        Ok(descriptor)
    }
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct TensorContractDescriptorTerm<C = f64> {
    dst_block: usize,
    lhs_block: usize,
    rhs_block: usize,
    lhs_layout_start: usize,
    rhs_layout_start: usize,
    output_layout_start: usize,
    scatter_layout_start: usize,
    lhs_rank: usize,
    rhs_rank: usize,
    output_rank: usize,
    lhs_offset: usize,
    rhs_offset: usize,
    dst_offset: isize,
    workspace_len: usize,
    apply_beta: bool,
    coefficient: C,
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
    pub fn compile<
        TDst,
        TSrc,
        const DST_NOUT: usize,
        const DST_NIN: usize,
        const SRC_NOUT: usize,
        const SRC_NIN: usize,
        SDst,
        SSrc,
    >(
        &self,
        dst: &TensorMap<TDst, DST_NOUT, DST_NIN, SDst>,
        src: &TensorMap<TSrc, SRC_NOUT, SRC_NIN, SSrc>,
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
        specs.push(TreeTransformGroupBlockSpec::multi(
            group.group_key().clone(),
            dst_keys,
            src_keys,
            coefficients_src_by_dst,
        ));
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
fn build_unique_tree_transform_group_plan<T, R, F>(
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
        specs.push(TreeTransformGroupBlockSpec::single(
            src_key.group_key(),
            dst_key,
            src_key.clone(),
            coefficient,
        ));
    }

    Ok(TreeTransformGroupPlan::new(specs))
}

#[cfg(test)]
fn build_unique_all_codomain_tree_transform_group_plan<R>(
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
        specs.push(TreeTransformGroupBlockSpec::single(
            src_key.group_key(),
            dst_key,
            src_key.clone(),
            coefficient,
        ));
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
        specs.push(TreeTransformGroupBlockSpec::multi(
            group.group_key().clone(),
            dst_keys,
            src_keys,
            coefficients_src_by_dst,
        ));
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
        specs.push(TreeTransformGroupBlockSpec::multi(
            group.group_key().clone(),
            dst_keys,
            src_keys,
            coefficients_src_by_dst,
        ));
    }

    Ok(TreeTransformGroupPlan::new(specs))
}

#[cfg(test)]
fn build_unique_tree_pair_transform_group_plan<R>(
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
        specs.push(TreeTransformGroupBlockSpec::single(
            src_key.group_key(),
            dst_key,
            src_key.clone(),
            coefficient,
        ));
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

#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub struct TreeTransformSectorPlanKey<RuleKey> {
    rule: RuleKey,
    scope: TreeTransformPlanScope,
    operation: TreeTransformOperationKey,
    source_groups: Vec<TreeTransformSourceGroupKey>,
}

impl<RuleKey> TreeTransformSectorPlanKey<RuleKey>
where
    RuleKey: Clone + Eq + Hash,
{
    pub fn tree_pair<R>(
        rule: &R,
        operation: TreeTransformOperationKey,
        src_structure: &BlockStructure,
    ) -> Result<Self, OperationError>
    where
        R: TreeTransformRuleCacheKey<Key = RuleKey>,
    {
        Self::new(
            rule.tree_transform_rule_cache_key(),
            TreeTransformPlanScope::TreePair,
            operation,
            src_structure,
        )
    }

    pub fn all_codomain<R>(
        rule: &R,
        operation: TreeTransformOperationKey,
        src_structure: &BlockStructure,
    ) -> Result<Self, OperationError>
    where
        R: TreeTransformRuleCacheKey<Key = RuleKey>,
    {
        Self::new(
            rule.tree_transform_rule_cache_key(),
            TreeTransformPlanScope::AllCodomain,
            operation,
            src_structure,
        )
    }

    fn new(
        rule: RuleKey,
        scope: TreeTransformPlanScope,
        operation: TreeTransformOperationKey,
        src_structure: &BlockStructure,
    ) -> Result<Self, OperationError> {
        let source_groups = src_structure
            .fusion_tree_groups()
            .into_iter()
            .map(|group| TreeTransformSourceGroupKey::from_group(src_structure, &group))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self {
            rule,
            scope,
            operation,
            source_groups,
        })
    }

    #[inline]
    pub fn rule(&self) -> &RuleKey {
        &self.rule
    }

    #[inline]
    pub fn scope(&self) -> TreeTransformPlanScope {
        self.scope
    }

    #[inline]
    pub fn operation(&self) -> &TreeTransformOperationKey {
        &self.operation
    }

    #[inline]
    pub fn source_groups(&self) -> &[TreeTransformSourceGroupKey] {
        &self.source_groups
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub struct TreeTransformSourceGroupKey {
    group_key: FusionTreeGroupKey,
    src_keys: Vec<BlockKey>,
}

impl TreeTransformSourceGroupKey {
    fn from_group(
        structure: &BlockStructure,
        group: &FusionTreeBlockGroup,
    ) -> Result<Self, OperationError> {
        Ok(Self {
            group_key: group.group_key().clone(),
            src_keys: fusion_tree_group_block_keys(structure, group, "src")?,
        })
    }

    #[inline]
    pub fn group_key(&self) -> &FusionTreeGroupKey {
        &self.group_key
    }

    #[inline]
    pub fn src_keys(&self) -> &[BlockKey] {
        &self.src_keys
    }
}

#[cfg(test)]
#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub struct TreeTransformGroupPlanKey {
    operation: TreeTransformOperationKey,
    groups: Vec<TreeTransformCachedGroupKey>,
}

#[cfg(test)]
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

#[cfg(test)]
#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub struct TreeTransformCachedGroupKey {
    group_key: FusionTreeGroupKey,
    dst_keys: Vec<BlockKey>,
    src_keys: Vec<BlockKey>,
}

#[cfg(test)]
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

#[cfg(test)]
#[derive(Clone, Debug)]
pub struct TreeTransformGroupPlanCache<T> {
    plans: HashMap<TreeTransformGroupPlanKey, TreeTransformGroupPlan<T>>,
}

#[cfg(test)]
impl<T> Default for TreeTransformGroupPlanCache<T> {
    fn default() -> Self {
        Self {
            plans: HashMap::new(),
        }
    }
}

#[cfg(test)]
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

#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub struct BlockStructureCacheKey {
    rank: usize,
    blocks: Vec<BlockStructureCacheBlockKey>,
}

impl BlockStructureCacheKey {
    pub fn from_structure(structure: &BlockStructure) -> Result<Self, OperationError> {
        let mut blocks = Vec::with_capacity(structure.block_count());
        for index in 0..structure.block_count() {
            let block = structure.block(index)?;
            blocks.push(BlockStructureCacheBlockKey {
                key: block.key().clone(),
                shape: block.shape().to_vec(),
                strides: block.strides().to_vec(),
                offset: block.offset(),
            });
        }
        Ok(Self {
            rank: structure.rank(),
            blocks,
        })
    }

    #[inline]
    pub fn rank(&self) -> usize {
        self.rank
    }

    #[inline]
    pub fn blocks(&self) -> &[BlockStructureCacheBlockKey] {
        &self.blocks
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub struct BlockStructureCacheBlockKey {
    key: BlockKey,
    shape: Vec<usize>,
    strides: Vec<usize>,
    offset: usize,
}

impl BlockStructureCacheBlockKey {
    #[inline]
    pub fn key(&self) -> &BlockKey {
        &self.key
    }

    #[inline]
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    #[inline]
    pub fn strides(&self) -> &[usize] {
        &self.strides
    }

    #[inline]
    pub fn offset(&self) -> usize {
        self.offset
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub struct TreeTransformStructureCacheKey<PlanKey> {
    plan: PlanKey,
    dst: BlockStructureCacheKey,
    src: BlockStructureCacheKey,
}

impl<PlanKey> TreeTransformStructureCacheKey<PlanKey>
where
    PlanKey: Clone,
{
    pub fn from_structures(
        plan: PlanKey,
        dst_structure: &BlockStructure,
        src_structure: &BlockStructure,
    ) -> Result<Self, OperationError> {
        Ok(Self {
            plan,
            dst: BlockStructureCacheKey::from_structure(dst_structure)?,
            src: BlockStructureCacheKey::from_structure(src_structure)?,
        })
    }

    #[inline]
    pub fn plan(&self) -> &PlanKey {
        &self.plan
    }

    #[inline]
    pub fn dst(&self) -> &BlockStructureCacheKey {
        &self.dst
    }

    #[inline]
    pub fn src(&self) -> &BlockStructureCacheKey {
        &self.src
    }
}

#[derive(Clone, Debug)]
pub struct TreeTransformStructureCache<T, PlanKey> {
    structures: HashMap<TreeTransformStructureCacheKey<PlanKey>, TreeTransformStructure<T>>,
}

impl<T, PlanKey> Default for TreeTransformStructureCache<T, PlanKey> {
    fn default() -> Self {
        Self {
            structures: HashMap::new(),
        }
    }
}

impl<T, PlanKey> TreeTransformStructureCache<T, PlanKey>
where
    PlanKey: Clone + Eq + Hash,
{
    pub fn new() -> Self {
        Self::default()
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.structures.len()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.structures.is_empty()
    }

    pub fn get(
        &self,
        key: &TreeTransformStructureCacheKey<PlanKey>,
    ) -> Option<&TreeTransformStructure<T>> {
        self.structures.get(key)
    }

    pub fn insert(
        &mut self,
        key: TreeTransformStructureCacheKey<PlanKey>,
        structure: TreeTransformStructure<T>,
    ) -> Option<TreeTransformStructure<T>> {
        self.structures.insert(key, structure)
    }
}

#[derive(Clone, Debug)]
pub struct TreeTransformCache<T, RuleKey> {
    plans: HashMap<TreeTransformSectorPlanKey<RuleKey>, TreeTransformGroupPlan<T>>,
    structures: TreeTransformStructureCache<T, TreeTransformSectorPlanKey<RuleKey>>,
    stats: TreeTransformCacheStats,
}

pub type TreePairTransformCache<T, RuleKey> = TreeTransformCache<T, RuleKey>;

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct TreeTransformCacheStats {
    plan_hits: usize,
    plan_misses: usize,
    structure_hits: usize,
    structure_misses: usize,
}

impl TreeTransformCacheStats {
    #[inline]
    pub fn plan_hits(self) -> usize {
        self.plan_hits
    }

    #[inline]
    pub fn plan_misses(self) -> usize {
        self.plan_misses
    }

    #[inline]
    pub fn structure_hits(self) -> usize {
        self.structure_hits
    }

    #[inline]
    pub fn structure_misses(self) -> usize {
        self.structure_misses
    }
}

impl<T, RuleKey> Default for TreeTransformCache<T, RuleKey> {
    fn default() -> Self {
        Self {
            plans: HashMap::new(),
            structures: TreeTransformStructureCache::default(),
            stats: TreeTransformCacheStats::default(),
        }
    }
}

impl<T, RuleKey> TreeTransformCache<T, RuleKey>
where
    RuleKey: Clone + Eq + Hash,
{
    pub fn new() -> Self {
        Self::default()
    }

    #[inline]
    pub fn plan_len(&self) -> usize {
        self.plans.len()
    }

    #[inline]
    pub fn structure_len(&self) -> usize {
        self.structures.len()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.plans.is_empty() && self.structures.is_empty()
    }

    #[inline]
    pub fn stats(&self) -> TreeTransformCacheStats {
        self.stats
    }

    pub fn reset_stats(&mut self) {
        self.stats = TreeTransformCacheStats::default();
    }

    pub fn get_or_compile_tree_pair<
        R,
        TDst,
        TSrc,
        const DST_NOUT: usize,
        const DST_NIN: usize,
        const SRC_NOUT: usize,
        const SRC_NIN: usize,
        SDst,
        SSrc,
    >(
        &mut self,
        rule: &R,
        operation: TreeTransformOperationKey,
        dst: &TensorMap<TDst, DST_NOUT, DST_NIN, SDst>,
        src: &TensorMap<TSrc, SRC_NOUT, SRC_NIN, SSrc>,
    ) -> Result<&TreeTransformStructure<T>, OperationError>
    where
        R: MultiplicityFreeRigidSymbols<Scalar = T> + TreeTransformRuleCacheKey<Key = RuleKey>,
        T: Copy + Clone + Add<Output = T> + Mul<Output = T> + Zero,
    {
        let plan_key =
            TreeTransformSectorPlanKey::tree_pair(rule, operation.clone(), src.structure())?;
        if self.plans.contains_key(&plan_key) {
            self.stats.plan_hits += 1;
        } else {
            self.stats.plan_misses += 1;
            let plan = build_tree_pair_transform_group_plan(rule, operation, src.structure())?;
            self.plans.insert(plan_key.clone(), plan);
        }
        self.get_or_compile_structure(plan_key, dst, src)
    }

    pub fn get_or_compile_all_codomain<
        R,
        TDst,
        TSrc,
        const DST_NOUT: usize,
        const DST_NIN: usize,
        const SRC_NOUT: usize,
        const SRC_NIN: usize,
        SDst,
        SSrc,
    >(
        &mut self,
        rule: &R,
        operation: TreeTransformOperationKey,
        dst: &TensorMap<TDst, DST_NOUT, DST_NIN, SDst>,
        src: &TensorMap<TSrc, SRC_NOUT, SRC_NIN, SSrc>,
    ) -> Result<&TreeTransformStructure<T>, OperationError>
    where
        R: MultiplicityFreeFusionSymbols<Scalar = T> + TreeTransformRuleCacheKey<Key = RuleKey>,
        T: Copy + Clone + Add<Output = T> + Mul<Output = T> + Zero,
    {
        let plan_key =
            TreeTransformSectorPlanKey::all_codomain(rule, operation.clone(), src.structure())?;
        if self.plans.contains_key(&plan_key) {
            self.stats.plan_hits += 1;
        } else {
            self.stats.plan_misses += 1;
            let plan =
                build_all_codomain_tree_transform_group_plan(rule, operation, src.structure())?;
            self.plans.insert(plan_key.clone(), plan);
        }
        self.get_or_compile_structure(plan_key, dst, src)
    }

    fn get_or_compile_structure<
        TDst,
        TSrc,
        const DST_NOUT: usize,
        const DST_NIN: usize,
        const SRC_NOUT: usize,
        const SRC_NIN: usize,
        SDst,
        SSrc,
    >(
        &mut self,
        plan_key: TreeTransformSectorPlanKey<RuleKey>,
        dst: &TensorMap<TDst, DST_NOUT, DST_NIN, SDst>,
        src: &TensorMap<TSrc, SRC_NOUT, SRC_NIN, SSrc>,
    ) -> Result<&TreeTransformStructure<T>, OperationError>
    where
        T: Copy,
    {
        let structure_key = TreeTransformStructureCacheKey::from_structures(
            plan_key.clone(),
            dst.structure(),
            src.structure(),
        )?;
        if self.structures.get(&structure_key).is_some() {
            self.stats.structure_hits += 1;
        } else {
            self.stats.structure_misses += 1;
            let plan = self
                .plans
                .get(&plan_key)
                .expect("tree transform plan inserted before structure compile");
            let structure = plan.compile(dst, src)?;
            self.structures.insert(structure_key.clone(), structure);
        }
        Ok(self
            .structures
            .get(&structure_key)
            .expect("tree transform structure inserted before return"))
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

/// Replay-ready tree-transform descriptor.
///
/// This is the TensorKit-style transformer-build boundary: construction resolves
/// tree keys, coefficients, block layouts, offsets, and pack/scatter descriptors
/// against concrete source and destination structures. Hot paths should build
/// this once and replay it with [`tree_transform_execute_with`] while reusing a
/// backend and workspace.
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
    pub fn compile<
        TDst,
        TSrc,
        const DST_NOUT: usize,
        const DST_NIN: usize,
        const SRC_NOUT: usize,
        const SRC_NIN: usize,
        SDst,
        SSrc,
    >(
        dst: &TensorMap<TDst, DST_NOUT, DST_NIN, SDst>,
        src: &TensorMap<TSrc, SRC_NOUT, SRC_NIN, SSrc>,
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

    pub fn compile_keyed<
        TDst,
        TSrc,
        const DST_NOUT: usize,
        const DST_NIN: usize,
        const SRC_NOUT: usize,
        const SRC_NIN: usize,
        SDst,
        SSrc,
    >(
        dst: &TensorMap<TDst, DST_NOUT, DST_NIN, SDst>,
        src: &TensorMap<TSrc, SRC_NOUT, SRC_NIN, SSrc>,
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

    pub fn compile_grouped<
        TDst,
        TSrc,
        const DST_NOUT: usize,
        const DST_NIN: usize,
        const SRC_NOUT: usize,
        const SRC_NIN: usize,
        SDst,
        SSrc,
    >(
        dst: &TensorMap<TDst, DST_NOUT, DST_NIN, SDst>,
        src: &TensorMap<TSrc, SRC_NOUT, SRC_NIN, SSrc>,
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
    coefficients: Vec<T>,
}

impl<T> Default for TreeTransformWorkspace<T> {
    fn default() -> Self {
        Self {
            zero_strides: Vec::new(),
            source: Vec::new(),
            destination: Vec::new(),
            coefficients: Vec::new(),
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

pub trait TreeTransformScalar:
    Copy
    + Add<Self, Output = Self>
    + Mul<Self, Output = Self>
    + PartialEq
    + Zero
    + One
    + strided_kernel::MaybeSendSync
{
}

impl<T> TreeTransformScalar for T where
    T: Copy
        + Add<T, Output = T>
        + Mul<T, Output = T>
        + PartialEq
        + Zero
        + One
        + strided_kernel::MaybeSendSync
{
}

/// Action of a categorical recoupling coefficient on tensor storage data.
///
/// TensorKit allows, for example, real SU(2) coefficients to act on complex
/// tensor blocks. Rust needs that conversion boundary to be explicit.
pub trait RecouplingCoefficientAction<C>: Copy {
    fn scale_by_coefficient(self, coefficient: C) -> Self;
    fn coefficient_as_data(coefficient: C) -> Self;
}

macro_rules! impl_same_recoupling_coefficient_action {
    ($($ty:ty),+ $(,)?) => {
        $(
            impl RecouplingCoefficientAction<$ty> for $ty {
                #[inline]
                fn scale_by_coefficient(self, coefficient: $ty) -> Self {
                    self * coefficient
                }

                #[inline]
                fn coefficient_as_data(coefficient: $ty) -> Self {
                    coefficient
                }
            }
        )+
    };
}

impl_same_recoupling_coefficient_action!(f32, f64, i32, i64, Complex32, Complex64);

impl RecouplingCoefficientAction<f64> for f32 {
    #[inline]
    fn scale_by_coefficient(self, coefficient: f64) -> Self {
        self * coefficient as f32
    }

    #[inline]
    fn coefficient_as_data(coefficient: f64) -> Self {
        coefficient as f32
    }
}

impl RecouplingCoefficientAction<f32> for f64 {
    #[inline]
    fn scale_by_coefficient(self, coefficient: f32) -> Self {
        self * f64::from(coefficient)
    }

    #[inline]
    fn coefficient_as_data(coefficient: f32) -> Self {
        f64::from(coefficient)
    }
}

impl RecouplingCoefficientAction<f32> for Complex32 {
    #[inline]
    fn scale_by_coefficient(self, coefficient: f32) -> Self {
        self * coefficient
    }

    #[inline]
    fn coefficient_as_data(coefficient: f32) -> Self {
        Self::new(coefficient, 0.0)
    }
}

impl RecouplingCoefficientAction<f64> for Complex32 {
    #[inline]
    fn scale_by_coefficient(self, coefficient: f64) -> Self {
        self * coefficient as f32
    }

    #[inline]
    fn coefficient_as_data(coefficient: f64) -> Self {
        Self::new(coefficient as f32, 0.0)
    }
}

impl RecouplingCoefficientAction<f32> for Complex64 {
    #[inline]
    fn scale_by_coefficient(self, coefficient: f32) -> Self {
        self * f64::from(coefficient)
    }

    #[inline]
    fn coefficient_as_data(coefficient: f32) -> Self {
        Self::new(f64::from(coefficient), 0.0)
    }
}

impl RecouplingCoefficientAction<f64> for Complex64 {
    #[inline]
    fn scale_by_coefficient(self, coefficient: f64) -> Self {
        self * coefficient
    }

    #[inline]
    fn coefficient_as_data(coefficient: f64) -> Self {
        Self::new(coefficient, 0.0)
    }
}

pub trait TreeTransformBackend<D, C>
where
    D: TreeTransformScalar,
    C: Copy,
{
    type Workspace;

    fn tree_transform_structure_into<
        const DST_NOUT: usize,
        const DST_NIN: usize,
        const SRC_NOUT: usize,
        const SRC_NIN: usize,
        SDst,
        SSrc,
    >(
        &mut self,
        workspace: &mut Self::Workspace,
        structure: &TreeTransformStructure<C>,
        dst: &mut TensorMap<D, DST_NOUT, DST_NIN, SDst>,
        src: &TensorMap<D, SRC_NOUT, SRC_NIN, SSrc>,
        alpha: D,
        beta: D,
    ) -> Result<(), OperationError>;
}

pub trait TensorContractBackend<D, C = f64>
where
    D: DenseBlockScalar + RecouplingCoefficientAction<C>,
    C: Copy + One,
{
    type Workspace;

    fn tensorcontract_structure_into<
        const DST_NOUT: usize,
        const DST_NIN: usize,
        const LHS_NOUT: usize,
        const LHS_NIN: usize,
        const RHS_NOUT: usize,
        const RHS_NIN: usize,
        SDst,
        SLhs,
        SRhs,
    >(
        &mut self,
        workspace: &mut Self::Workspace,
        structure: &TensorContractStructure<C>,
        dst: &mut TensorMap<D, DST_NOUT, DST_NIN, SDst>,
        lhs: &TensorMap<D, LHS_NOUT, LHS_NIN, SLhs>,
        rhs: &TensorMap<D, RHS_NOUT, RHS_NIN, SRhs>,
        alpha: D,
        beta: D,
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

#[derive(Clone, Debug)]
pub struct TensorContractWorkspace<T> {
    output: Vec<T>,
    zero_strides: Vec<isize>,
}

impl<T> Default for TensorContractWorkspace<T> {
    fn default() -> Self {
        Self {
            output: Vec::new(),
            zero_strides: Vec::new(),
        }
    }
}

impl<T> TensorContractWorkspace<T> {
    #[inline]
    pub fn output_len(&self) -> usize {
        self.output.len()
    }
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

#[derive(Debug)]
pub struct TreeTransformExecutionContext<D, RuleKey, C = D, B = DenseTreeTransformOperations>
where
    D: TreeTransformScalar,
    C: Copy,
    B: TreeTransformBackend<D, C>,
{
    backend: B,
    workspace: B::Workspace,
    cache: TreeTransformCache<C, RuleKey>,
}

impl<D, RuleKey, C, B> TreeTransformExecutionContext<D, RuleKey, C, B>
where
    D: TreeTransformScalar,
    C: Copy,
    B: TreeTransformBackend<D, C>,
{
    pub fn with_parts(
        backend: B,
        workspace: B::Workspace,
        cache: TreeTransformCache<C, RuleKey>,
    ) -> Self {
        Self {
            backend,
            workspace,
            cache,
        }
    }

    #[inline]
    pub fn backend(&self) -> &B {
        &self.backend
    }

    #[inline]
    pub fn backend_mut(&mut self) -> &mut B {
        &mut self.backend
    }

    #[inline]
    pub fn workspace(&self) -> &B::Workspace {
        &self.workspace
    }

    #[inline]
    pub fn workspace_mut(&mut self) -> &mut B::Workspace {
        &mut self.workspace
    }

    #[inline]
    pub fn cache(&self) -> &TreeTransformCache<C, RuleKey> {
        &self.cache
    }

    #[inline]
    pub fn cache_mut(&mut self) -> &mut TreeTransformCache<C, RuleKey> {
        &mut self.cache
    }

    pub fn into_parts(self) -> (B, B::Workspace, TreeTransformCache<C, RuleKey>) {
        (self.backend, self.workspace, self.cache)
    }
}

impl<D, RuleKey, C, B> TreeTransformExecutionContext<D, RuleKey, C, B>
where
    D: TreeTransformScalar,
    C: Copy,
    RuleKey: Clone + Eq + Hash,
    B: TreeTransformBackend<D, C>,
    B::Workspace: Default,
{
    pub fn new(backend: B) -> Self {
        Self::with_parts(backend, B::Workspace::default(), TreeTransformCache::new())
    }
}

impl<D, RuleKey, C, B> Default for TreeTransformExecutionContext<D, RuleKey, C, B>
where
    D: TreeTransformScalar,
    C: Copy,
    RuleKey: Clone + Eq + Hash,
    B: TreeTransformBackend<D, C> + Default,
    B::Workspace: Default,
{
    fn default() -> Self {
        Self::new(B::default())
    }
}

impl<D, RuleKey, C, B> TreeTransformExecutionContext<D, RuleKey, C, B>
where
    D: TreeTransformScalar,
    C: Copy + Clone + Add<Output = C> + Mul<Output = C> + Zero,
    RuleKey: Clone + Eq + Hash,
    B: TreeTransformBackend<D, C>,
{
    pub fn tree_pair_transform_into<
        R,
        const DST_NOUT: usize,
        const DST_NIN: usize,
        const SRC_NOUT: usize,
        const SRC_NIN: usize,
        SDst,
        SSrc,
    >(
        &mut self,
        rule: &R,
        operation: TreeTransformOperationKey,
        dst: &mut TensorMap<D, DST_NOUT, DST_NIN, SDst>,
        src: &TensorMap<D, SRC_NOUT, SRC_NIN, SSrc>,
        alpha: D,
        beta: D,
    ) -> Result<(), OperationError>
    where
        R: MultiplicityFreeRigidSymbols<Scalar = C> + TreeTransformRuleCacheKey<Key = RuleKey>,
    {
        let Self {
            backend,
            workspace,
            cache,
        } = self;
        let structure = cache.get_or_compile_tree_pair(rule, operation, dst, src)?;
        backend.tree_transform_structure_into(workspace, structure, dst, src, alpha, beta)
    }

    pub fn all_codomain_tree_transform_into<
        R,
        const DST_NOUT: usize,
        const DST_NIN: usize,
        const SRC_NOUT: usize,
        const SRC_NIN: usize,
        SDst,
        SSrc,
    >(
        &mut self,
        rule: &R,
        operation: TreeTransformOperationKey,
        dst: &mut TensorMap<D, DST_NOUT, DST_NIN, SDst>,
        src: &TensorMap<D, SRC_NOUT, SRC_NIN, SSrc>,
        alpha: D,
        beta: D,
    ) -> Result<(), OperationError>
    where
        R: MultiplicityFreeFusionSymbols<Scalar = C> + TreeTransformRuleCacheKey<Key = RuleKey>,
    {
        let Self {
            backend,
            workspace,
            cache,
        } = self;
        let structure = cache.get_or_compile_all_codomain(rule, operation, dst, src)?;
        backend.tree_transform_structure_into(workspace, structure, dst, src, alpha, beta)
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

impl<D, C> TreeTransformBackend<D, C> for HostTensorOperations
where
    D: Copy
        + Add<D, Output = D>
        + Mul<D, Output = D>
        + PartialEq
        + Zero
        + One
        + strided_kernel::MaybeSendSync
        + RecouplingCoefficientAction<C>,
    C: Copy,
{
    type Workspace = TreeTransformWorkspace<D>;

    fn tree_transform_structure_into<
        const DST_NOUT: usize,
        const DST_NIN: usize,
        const SRC_NOUT: usize,
        const SRC_NIN: usize,
        SDst,
        SSrc,
    >(
        &mut self,
        workspace: &mut Self::Workspace,
        structure: &TreeTransformStructure<C>,
        dst: &mut TensorMap<D, DST_NOUT, DST_NIN, SDst>,
        src: &TensorMap<D, SRC_NOUT, SRC_NIN, SSrc>,
        alpha: D,
        beta: D,
    ) -> Result<(), OperationError> {
        tree_transform_structure_with_strided_kernel(workspace, structure, dst, src, alpha, beta)
    }
}

#[doc(hidden)]
pub trait DenseBlockScalar:
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

#[doc(hidden)]
pub trait DenseRecouplingScalar: DenseBlockScalar + RecouplingCoefficientAction<Self> {}

impl<T> DenseRecouplingScalar for T where T: DenseBlockScalar + RecouplingCoefficientAction<Self> {}

macro_rules! impl_dense_block_scalar {
    ($ty:ty, $read_variant:ident, $write_variant:ident) => {
        impl DenseBlockScalar for $ty {
            fn dense_read(view: DenseView<'_, Self>) -> DenseRead<'_> {
                DenseRead::$read_variant(view)
            }

            fn dense_write(view: DenseViewMut<'_, Self>) -> DenseWrite<'_> {
                DenseWrite::$write_variant(view)
            }
        }
    };
}

impl_dense_block_scalar!(f32, F32, F32);
impl_dense_block_scalar!(f64, F64, F64);
impl_dense_block_scalar!(Complex32, C32, C32);
impl_dense_block_scalar!(Complex64, C64, C64);

impl<E, D, C> TreeTransformBackend<D, C> for DenseTreeTransformOperations<E>
where
    E: DenseExecutor,
    D: DenseRecouplingScalar + RecouplingCoefficientAction<C>,
    C: Copy,
{
    type Workspace = TreeTransformWorkspace<D>;

    fn tree_transform_structure_into<
        const DST_NOUT: usize,
        const DST_NIN: usize,
        const SRC_NOUT: usize,
        const SRC_NIN: usize,
        SDst,
        SSrc,
    >(
        &mut self,
        workspace: &mut Self::Workspace,
        structure: &TreeTransformStructure<C>,
        dst: &mut TensorMap<D, DST_NOUT, DST_NIN, SDst>,
        src: &TensorMap<D, SRC_NOUT, SRC_NIN, SSrc>,
        alpha: D,
        beta: D,
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

impl<E, D, C> TensorContractBackend<D, C> for DenseTreeTransformOperations<E>
where
    E: DenseExecutor,
    D: DenseBlockScalar + RecouplingCoefficientAction<C>,
    C: Copy + One,
{
    type Workspace = TensorContractWorkspace<D>;

    fn tensorcontract_structure_into<
        const DST_NOUT: usize,
        const DST_NIN: usize,
        const LHS_NOUT: usize,
        const LHS_NIN: usize,
        const RHS_NOUT: usize,
        const RHS_NIN: usize,
        SDst,
        SLhs,
        SRhs,
    >(
        &mut self,
        workspace: &mut Self::Workspace,
        structure: &TensorContractStructure<C>,
        dst: &mut TensorMap<D, DST_NOUT, DST_NIN, SDst>,
        lhs: &TensorMap<D, LHS_NOUT, LHS_NIN, SLhs>,
        rhs: &TensorMap<D, RHS_NOUT, RHS_NIN, SRhs>,
        alpha: D,
        beta: D,
    ) -> Result<(), OperationError> {
        tensorcontract_structure_with_dense_executor(
            &mut self.dense,
            workspace,
            structure,
            dst,
            lhs,
            rhs,
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

pub fn tensorcontract_into<
    D,
    const DST_NOUT: usize,
    const DST_NIN: usize,
    const LHS_NOUT: usize,
    const LHS_NIN: usize,
    const RHS_NOUT: usize,
    const RHS_NIN: usize,
    SDst,
    SLhs,
    SRhs,
>(
    dst: &mut TensorMap<D, DST_NOUT, DST_NIN, SDst>,
    lhs: &TensorMap<D, LHS_NOUT, LHS_NIN, SLhs>,
    rhs: &TensorMap<D, RHS_NOUT, RHS_NIN, SRhs>,
    axes: TensorContractAxisSpec<'_>,
    alpha: D,
    beta: D,
) -> Result<(), OperationError>
where
    D: DenseBlockScalar + RecouplingCoefficientAction<f64>,
{
    let mut backend = DenseTreeTransformOperations::default_executor();
    let mut workspace = TensorContractWorkspace::default();
    tensorcontract_into_with(
        &mut backend,
        &mut workspace,
        dst,
        lhs,
        rhs,
        axes,
        alpha,
        beta,
    )
}

pub fn tensorcontract_into_with<
    B,
    D,
    const DST_NOUT: usize,
    const DST_NIN: usize,
    const LHS_NOUT: usize,
    const LHS_NIN: usize,
    const RHS_NOUT: usize,
    const RHS_NIN: usize,
    SDst,
    SLhs,
    SRhs,
>(
    backend: &mut B,
    workspace: &mut B::Workspace,
    dst: &mut TensorMap<D, DST_NOUT, DST_NIN, SDst>,
    lhs: &TensorMap<D, LHS_NOUT, LHS_NIN, SLhs>,
    rhs: &TensorMap<D, RHS_NOUT, RHS_NIN, SRhs>,
    axes: TensorContractAxisSpec<'_>,
    alpha: D,
    beta: D,
) -> Result<(), OperationError>
where
    B: TensorContractBackend<D, f64>,
    D: DenseBlockScalar + RecouplingCoefficientAction<f64>,
{
    let structure = tensorcontract_structure(dst, lhs, rhs, axes)?;
    tensorcontract_execute_with(backend, workspace, &structure, dst, lhs, rhs, alpha, beta)
}

pub fn tensorcontract_fusion_into<
    R,
    D,
    const DST_NOUT: usize,
    const DST_NIN: usize,
    const LHS_NOUT: usize,
    const LHS_NIN: usize,
    const RHS_NOUT: usize,
    const RHS_NIN: usize,
    SDst,
    SLhs,
    SRhs,
>(
    rule: &R,
    dst: &mut TensorMap<D, DST_NOUT, DST_NIN, SDst>,
    lhs: &TensorMap<D, LHS_NOUT, LHS_NIN, SLhs>,
    rhs: &TensorMap<D, RHS_NOUT, RHS_NIN, SRhs>,
    axes: TensorContractAxisSpec<'_>,
    alpha: D,
    beta: D,
) -> Result<(), OperationError>
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64>,
    D: DenseBlockScalar + RecouplingCoefficientAction<f64>,
{
    let mut backend = DenseTreeTransformOperations::default_executor();
    let mut workspace = TensorContractWorkspace::default();
    tensorcontract_fusion_into_with(
        &mut backend,
        &mut workspace,
        rule,
        dst,
        lhs,
        rhs,
        axes,
        alpha,
        beta,
    )
}

/// Execute a TensorKit-style fusion contraction through explicit source
/// tree-pair transforms.
///
/// This is the reference-safe path for contractions whose source operands are
/// not already in canonical compose form. The caller provides the canonical
/// temporary tensors because their ranks are determined by the chosen
/// contraction axes and therefore cannot be constructed generically from the
/// original const-generic tensor ranks.
///
/// The sequence is:
///
/// 1. transform `lhs` to `(lhs open) <- (lhs contracted)`;
/// 2. transform `rhs` to `(rhs contracted) <- (rhs open)`;
/// 3. run the fusion contraction from those canonical operands into `dst`,
///    which must already be the canonical output tree-pair shape.
///
/// Use [`tensorcontract_fusion_explicit_plan_into_canonical_dst`] when the
/// requested output permutation needs a final tree-pair transform.
pub fn tensorcontract_fusion_via_tree_pair_transforms_into<
    R,
    D,
    const DST_NOUT: usize,
    const DST_NIN: usize,
    const LHS_NOUT: usize,
    const LHS_NIN: usize,
    const RHS_NOUT: usize,
    const RHS_NIN: usize,
    const LHS_CAN_NOUT: usize,
    const LHS_CAN_NIN: usize,
    const RHS_CAN_NOUT: usize,
    const RHS_CAN_NIN: usize,
    SDst,
    SLhs,
    SRhs,
    SLhsCan,
    SRhsCan,
>(
    rule: &R,
    dst: &mut TensorMap<D, DST_NOUT, DST_NIN, SDst>,
    lhs_canonical: &mut TensorMap<D, LHS_CAN_NOUT, LHS_CAN_NIN, SLhsCan>,
    rhs_canonical: &mut TensorMap<D, RHS_CAN_NOUT, RHS_CAN_NIN, SRhsCan>,
    lhs: &TensorMap<D, LHS_NOUT, LHS_NIN, SLhs>,
    rhs: &TensorMap<D, RHS_NOUT, RHS_NIN, SRhs>,
    axes: TensorContractAxisSpec<'_>,
    alpha: D,
    beta: D,
) -> Result<(), OperationError>
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64>,
    D: DenseRecouplingScalar + RecouplingCoefficientAction<f64>,
{
    let plan = tensorcontract_fusion_explicit_plan(
        rule,
        dst.fusion_space()
            .ok_or(OperationError::Core(CoreError::MissingFusionSpace))?,
        lhs.fusion_space()
            .ok_or(OperationError::Core(CoreError::MissingFusionSpace))?,
        rhs.fusion_space()
            .ok_or(OperationError::Core(CoreError::MissingFusionSpace))?,
        axes,
    )?;
    tensorcontract_fusion_explicit_plan_into(
        rule,
        &plan,
        dst,
        lhs_canonical,
        rhs_canonical,
        lhs,
        rhs,
        alpha,
        beta,
    )
}

const EXPLICIT_OUTPUT_TRANSFORM_REQUIRES_CANONICAL_DST: &str =
    "explicit fusion contraction with output tree-pair transform requires caller-owned canonical_dst";

pub fn tensorcontract_fusion_explicit_plan_into<
    R,
    D,
    const DST_NOUT: usize,
    const DST_NIN: usize,
    const LHS_NOUT: usize,
    const LHS_NIN: usize,
    const RHS_NOUT: usize,
    const RHS_NIN: usize,
    const LHS_CAN_NOUT: usize,
    const LHS_CAN_NIN: usize,
    const RHS_CAN_NOUT: usize,
    const RHS_CAN_NIN: usize,
    SDst,
    SLhs,
    SRhs,
    SLhsCan,
    SRhsCan,
>(
    rule: &R,
    plan: &TensorContractFusionExplicitPlan,
    dst: &mut TensorMap<D, DST_NOUT, DST_NIN, SDst>,
    lhs_canonical: &mut TensorMap<D, LHS_CAN_NOUT, LHS_CAN_NIN, SLhsCan>,
    rhs_canonical: &mut TensorMap<D, RHS_CAN_NOUT, RHS_CAN_NIN, SRhsCan>,
    lhs: &TensorMap<D, LHS_NOUT, LHS_NIN, SLhs>,
    rhs: &TensorMap<D, RHS_NOUT, RHS_NIN, SRhs>,
    alpha: D,
    beta: D,
) -> Result<(), OperationError>
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64>,
    D: DenseRecouplingScalar + RecouplingCoefficientAction<f64>,
{
    let mut tree_backend = DenseTreeTransformOperations::default_executor();
    let mut contract_backend = DenseTreeTransformOperations::default_executor();
    let mut tree_workspace = TreeTransformWorkspace::default();
    let mut contract_workspace = TensorContractWorkspace::default();
    tensorcontract_fusion_explicit_plan_into_with(
        &mut tree_backend,
        &mut tree_workspace,
        &mut contract_backend,
        &mut contract_workspace,
        rule,
        plan,
        dst,
        lhs_canonical,
        rhs_canonical,
        lhs,
        rhs,
        alpha,
        beta,
    )
}

pub fn tensorcontract_fusion_explicit_plan_into_canonical_dst<
    R,
    D,
    const DST_NOUT: usize,
    const DST_NIN: usize,
    const DST_CAN_NOUT: usize,
    const DST_CAN_NIN: usize,
    const LHS_NOUT: usize,
    const LHS_NIN: usize,
    const RHS_NOUT: usize,
    const RHS_NIN: usize,
    const LHS_CAN_NOUT: usize,
    const LHS_CAN_NIN: usize,
    const RHS_CAN_NOUT: usize,
    const RHS_CAN_NIN: usize,
    SDst,
    SDstCan,
    SLhs,
    SRhs,
    SLhsCan,
    SRhsCan,
>(
    rule: &R,
    plan: &TensorContractFusionExplicitPlan,
    dst: &mut TensorMap<D, DST_NOUT, DST_NIN, SDst>,
    canonical_dst: &mut TensorMap<D, DST_CAN_NOUT, DST_CAN_NIN, SDstCan>,
    lhs_canonical: &mut TensorMap<D, LHS_CAN_NOUT, LHS_CAN_NIN, SLhsCan>,
    rhs_canonical: &mut TensorMap<D, RHS_CAN_NOUT, RHS_CAN_NIN, SRhsCan>,
    lhs: &TensorMap<D, LHS_NOUT, LHS_NIN, SLhs>,
    rhs: &TensorMap<D, RHS_NOUT, RHS_NIN, SRhs>,
    alpha: D,
    beta: D,
) -> Result<(), OperationError>
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64>,
    D: DenseRecouplingScalar + RecouplingCoefficientAction<f64>,
{
    let mut tree_backend = DenseTreeTransformOperations::default_executor();
    let mut contract_backend = DenseTreeTransformOperations::default_executor();
    let mut tree_workspace = TreeTransformWorkspace::default();
    let mut contract_workspace = TensorContractWorkspace::default();
    tensorcontract_fusion_explicit_plan_into_canonical_dst_with(
        &mut tree_backend,
        &mut tree_workspace,
        &mut contract_backend,
        &mut contract_workspace,
        rule,
        plan,
        dst,
        canonical_dst,
        lhs_canonical,
        rhs_canonical,
        lhs,
        rhs,
        alpha,
        beta,
    )
}

pub fn tensorcontract_fusion_explicit_plan_into_with<
    BT,
    BC,
    R,
    D,
    const DST_NOUT: usize,
    const DST_NIN: usize,
    const LHS_NOUT: usize,
    const LHS_NIN: usize,
    const RHS_NOUT: usize,
    const RHS_NIN: usize,
    const LHS_CAN_NOUT: usize,
    const LHS_CAN_NIN: usize,
    const RHS_CAN_NOUT: usize,
    const RHS_CAN_NIN: usize,
    SDst,
    SLhs,
    SRhs,
    SLhsCan,
    SRhsCan,
>(
    tree_backend: &mut BT,
    tree_workspace: &mut BT::Workspace,
    contract_backend: &mut BC,
    contract_workspace: &mut BC::Workspace,
    rule: &R,
    plan: &TensorContractFusionExplicitPlan,
    dst: &mut TensorMap<D, DST_NOUT, DST_NIN, SDst>,
    lhs_canonical: &mut TensorMap<D, LHS_CAN_NOUT, LHS_CAN_NIN, SLhsCan>,
    rhs_canonical: &mut TensorMap<D, RHS_CAN_NOUT, RHS_CAN_NIN, SRhsCan>,
    lhs: &TensorMap<D, LHS_NOUT, LHS_NIN, SLhs>,
    rhs: &TensorMap<D, RHS_NOUT, RHS_NIN, SRhs>,
    alpha: D,
    beta: D,
) -> Result<(), OperationError>
where
    BT: TreeTransformBackend<D, f64>,
    BC: TensorContractBackend<D, f64>,
    R: MultiplicityFreeRigidSymbols<Scalar = f64>,
    D: DenseRecouplingScalar + RecouplingCoefficientAction<f64>,
{
    if !plan.output_transform_is_identity()
        || DST_NOUT != plan.canonical_dst_nout()
        || DST_NIN != plan.canonical_dst_nin()
    {
        return Err(OperationError::UnsupportedTensorContractScope {
            message: EXPLICIT_OUTPUT_TRANSFORM_REQUIRES_CANONICAL_DST,
        });
    }
    if LHS_CAN_NOUT != plan.lhs_canonical_nout() || LHS_CAN_NIN != plan.lhs_canonical_nin() {
        return Err(OperationError::StructureRankMismatch {
            expected: plan.lhs_canonical_nout() + plan.lhs_canonical_nin(),
            actual: LHS_CAN_NOUT + LHS_CAN_NIN,
        });
    }
    if RHS_CAN_NOUT != plan.rhs_canonical_nout() || RHS_CAN_NIN != plan.rhs_canonical_nin() {
        return Err(OperationError::StructureRankMismatch {
            expected: plan.rhs_canonical_nout() + plan.rhs_canonical_nin(),
            actual: RHS_CAN_NOUT + RHS_CAN_NIN,
        });
    }

    lhs_canonical.data_mut().fill(D::zero());
    rhs_canonical.data_mut().fill(D::zero());
    tree_pair_transform_into_with(
        tree_backend,
        tree_workspace,
        rule,
        plan.lhs_transform().clone(),
        lhs_canonical,
        lhs,
        D::one(),
        D::zero(),
    )?;
    tree_pair_transform_into_with(
        tree_backend,
        tree_workspace,
        rule,
        plan.rhs_transform().clone(),
        rhs_canonical,
        rhs,
        D::one(),
        D::zero(),
    )?;

    tensorcontract_fusion_into_with(
        contract_backend,
        contract_workspace,
        rule,
        dst,
        lhs_canonical,
        rhs_canonical,
        plan.canonical_axes().as_spec(),
        alpha,
        beta,
    )
}

pub fn tensorcontract_fusion_explicit_plan_into_canonical_dst_with<
    BT,
    BC,
    R,
    D,
    const DST_NOUT: usize,
    const DST_NIN: usize,
    const DST_CAN_NOUT: usize,
    const DST_CAN_NIN: usize,
    const LHS_NOUT: usize,
    const LHS_NIN: usize,
    const RHS_NOUT: usize,
    const RHS_NIN: usize,
    const LHS_CAN_NOUT: usize,
    const LHS_CAN_NIN: usize,
    const RHS_CAN_NOUT: usize,
    const RHS_CAN_NIN: usize,
    SDst,
    SDstCan,
    SLhs,
    SRhs,
    SLhsCan,
    SRhsCan,
>(
    tree_backend: &mut BT,
    tree_workspace: &mut BT::Workspace,
    contract_backend: &mut BC,
    contract_workspace: &mut BC::Workspace,
    rule: &R,
    plan: &TensorContractFusionExplicitPlan,
    dst: &mut TensorMap<D, DST_NOUT, DST_NIN, SDst>,
    canonical_dst: &mut TensorMap<D, DST_CAN_NOUT, DST_CAN_NIN, SDstCan>,
    lhs_canonical: &mut TensorMap<D, LHS_CAN_NOUT, LHS_CAN_NIN, SLhsCan>,
    rhs_canonical: &mut TensorMap<D, RHS_CAN_NOUT, RHS_CAN_NIN, SRhsCan>,
    lhs: &TensorMap<D, LHS_NOUT, LHS_NIN, SLhs>,
    rhs: &TensorMap<D, RHS_NOUT, RHS_NIN, SRhs>,
    alpha: D,
    beta: D,
) -> Result<(), OperationError>
where
    BT: TreeTransformBackend<D, f64>,
    BC: TensorContractBackend<D, f64>,
    R: MultiplicityFreeRigidSymbols<Scalar = f64>,
    D: DenseRecouplingScalar + RecouplingCoefficientAction<f64>,
{
    if DST_CAN_NOUT != plan.canonical_dst_nout() || DST_CAN_NIN != plan.canonical_dst_nin() {
        return Err(OperationError::StructureRankMismatch {
            expected: plan.canonical_dst_nout() + plan.canonical_dst_nin(),
            actual: DST_CAN_NOUT + DST_CAN_NIN,
        });
    }
    if LHS_CAN_NOUT != plan.lhs_canonical_nout() || LHS_CAN_NIN != plan.lhs_canonical_nin() {
        return Err(OperationError::StructureRankMismatch {
            expected: plan.lhs_canonical_nout() + plan.lhs_canonical_nin(),
            actual: LHS_CAN_NOUT + LHS_CAN_NIN,
        });
    }
    if RHS_CAN_NOUT != plan.rhs_canonical_nout() || RHS_CAN_NIN != plan.rhs_canonical_nin() {
        return Err(OperationError::StructureRankMismatch {
            expected: plan.rhs_canonical_nout() + plan.rhs_canonical_nin(),
            actual: RHS_CAN_NOUT + RHS_CAN_NIN,
        });
    }

    lhs_canonical.data_mut().fill(D::zero());
    rhs_canonical.data_mut().fill(D::zero());
    canonical_dst.data_mut().fill(D::zero());
    tree_pair_transform_into_with(
        tree_backend,
        tree_workspace,
        rule,
        plan.lhs_transform().clone(),
        lhs_canonical,
        lhs,
        D::one(),
        D::zero(),
    )?;
    tree_pair_transform_into_with(
        tree_backend,
        tree_workspace,
        rule,
        plan.rhs_transform().clone(),
        rhs_canonical,
        rhs,
        D::one(),
        D::zero(),
    )?;

    tensorcontract_fusion_into_with(
        contract_backend,
        contract_workspace,
        rule,
        canonical_dst,
        lhs_canonical,
        rhs_canonical,
        plan.canonical_axes().as_spec(),
        alpha,
        D::zero(),
    )?;

    tree_pair_transform_into_with(
        tree_backend,
        tree_workspace,
        rule,
        plan.output_transform().clone(),
        dst,
        canonical_dst,
        D::one(),
        beta,
    )
}

pub fn tensorcontract_fusion_into_with<
    B,
    R,
    D,
    const DST_NOUT: usize,
    const DST_NIN: usize,
    const LHS_NOUT: usize,
    const LHS_NIN: usize,
    const RHS_NOUT: usize,
    const RHS_NIN: usize,
    SDst,
    SLhs,
    SRhs,
>(
    backend: &mut B,
    workspace: &mut B::Workspace,
    rule: &R,
    dst: &mut TensorMap<D, DST_NOUT, DST_NIN, SDst>,
    lhs: &TensorMap<D, LHS_NOUT, LHS_NIN, SLhs>,
    rhs: &TensorMap<D, RHS_NOUT, RHS_NIN, SRhs>,
    axes: TensorContractAxisSpec<'_>,
    alpha: D,
    beta: D,
) -> Result<(), OperationError>
where
    B: TensorContractBackend<D, f64>,
    R: MultiplicityFreeRigidSymbols<Scalar = f64>,
    D: DenseBlockScalar + RecouplingCoefficientAction<f64>,
{
    let structure = tensorcontract_fusion_structure(rule, dst, lhs, rhs, axes)?;
    tensorcontract_execute_with(backend, workspace, &structure, dst, lhs, rhs, alpha, beta)
}

pub fn tensorcontract_execute_with<
    B,
    D,
    C,
    const DST_NOUT: usize,
    const DST_NIN: usize,
    const LHS_NOUT: usize,
    const LHS_NIN: usize,
    const RHS_NOUT: usize,
    const RHS_NIN: usize,
    SDst,
    SLhs,
    SRhs,
>(
    backend: &mut B,
    workspace: &mut B::Workspace,
    structure: &TensorContractStructure<C>,
    dst: &mut TensorMap<D, DST_NOUT, DST_NIN, SDst>,
    lhs: &TensorMap<D, LHS_NOUT, LHS_NIN, SLhs>,
    rhs: &TensorMap<D, RHS_NOUT, RHS_NIN, SRhs>,
    alpha: D,
    beta: D,
) -> Result<(), OperationError>
where
    B: TensorContractBackend<D, C>,
    D: DenseBlockScalar + RecouplingCoefficientAction<C>,
    C: Copy + One,
{
    structure.execute_with(backend, workspace, dst, lhs, rhs, alpha, beta)
}

pub fn tree_transform_execute_with<
    B,
    D,
    C,
    const DST_NOUT: usize,
    const DST_NIN: usize,
    const SRC_NOUT: usize,
    const SRC_NIN: usize,
    SDst,
    SSrc,
>(
    backend: &mut B,
    workspace: &mut B::Workspace,
    structure: &TreeTransformStructure<C>,
    dst: &mut TensorMap<D, DST_NOUT, DST_NIN, SDst>,
    src: &TensorMap<D, SRC_NOUT, SRC_NIN, SSrc>,
    alpha: D,
    beta: D,
) -> Result<(), OperationError>
where
    B: TreeTransformBackend<D, C>,
    D: Copy
        + Add<D, Output = D>
        + Mul<D, Output = D>
        + PartialEq
        + Zero
        + One
        + strided_kernel::MaybeSendSync,
    C: Copy,
{
    backend.tree_transform_structure_into(workspace, structure, dst, src, alpha, beta)
}

/// Build a replay-ready tree-pair transform structure.
///
/// This builds the replay-ready descriptor used by hot paths. It performs the
/// categorical tree-pair lowering and compiles the result against the actual
/// `dst` and `src` block structures. The returned structure can be reused with
/// [`tree_transform_execute_with`] as long as replay tensors have matching
/// structures.
pub fn tree_pair_transform_structure<
    R,
    TDst,
    TSrc,
    const DST_NOUT: usize,
    const DST_NIN: usize,
    const SRC_NOUT: usize,
    const SRC_NIN: usize,
    SDst,
    SSrc,
>(
    rule: &R,
    operation: TreeTransformOperationKey,
    dst: &TensorMap<TDst, DST_NOUT, DST_NIN, SDst>,
    src: &TensorMap<TSrc, SRC_NOUT, SRC_NIN, SSrc>,
) -> Result<TreeTransformStructure<R::Scalar>, OperationError>
where
    R: MultiplicityFreeRigidSymbols,
    R::Scalar: Copy + Add<Output = R::Scalar> + Mul<Output = R::Scalar> + Zero,
{
    let plan = build_tree_pair_transform_group_plan(rule, operation, src.structure())?;
    plan.compile(dst, src)
}

/// Compile and execute a tree-pair transform in one call.
///
/// This is a convenience API. It rebuilds the transform structure on every call;
/// hot tensor-network loops should call [`tree_pair_transform_structure`] once
/// and replay the returned structure with [`tree_transform_execute_with`].
pub fn tree_pair_transform_into<
    R,
    D,
    const DST_NOUT: usize,
    const DST_NIN: usize,
    const SRC_NOUT: usize,
    const SRC_NIN: usize,
    SDst,
    SSrc,
>(
    rule: &R,
    operation: TreeTransformOperationKey,
    dst: &mut TensorMap<D, DST_NOUT, DST_NIN, SDst>,
    src: &TensorMap<D, SRC_NOUT, SRC_NIN, SSrc>,
    alpha: D,
    beta: D,
) -> Result<(), OperationError>
where
    R: MultiplicityFreeRigidSymbols,
    R::Scalar: Copy + Clone + Add<Output = R::Scalar> + Mul<Output = R::Scalar> + Zero,
    D: DenseRecouplingScalar + RecouplingCoefficientAction<R::Scalar>,
{
    let mut backend = DenseTreeTransformOperations::default();
    let mut workspace = TreeTransformWorkspace::default();
    tree_pair_transform_into_with(
        &mut backend,
        &mut workspace,
        rule,
        operation,
        dst,
        src,
        alpha,
        beta,
    )
}

/// Compile and execute a tree-pair transform with caller-owned backend/workspace.
///
/// The backend and workspace are reused, but the transform structure is still
/// rebuilt on every call. This mirrors a TensorKit-style one-call transformer
/// application with explicit execution resources, not a cached transformer.
/// Use [`tree_pair_transform_into_with_context`] when the categorical plan and
/// replay descriptor should be cached behind a caller-owned context. Use
/// [`tree_pair_transform_structure`] plus [`tree_transform_execute_with`] for
/// the tightest loop when the exact replay descriptor is already known.
pub fn tree_pair_transform_into_with<
    B,
    R,
    D,
    const DST_NOUT: usize,
    const DST_NIN: usize,
    const SRC_NOUT: usize,
    const SRC_NIN: usize,
    SDst,
    SSrc,
>(
    backend: &mut B,
    workspace: &mut B::Workspace,
    rule: &R,
    operation: TreeTransformOperationKey,
    dst: &mut TensorMap<D, DST_NOUT, DST_NIN, SDst>,
    src: &TensorMap<D, SRC_NOUT, SRC_NIN, SSrc>,
    alpha: D,
    beta: D,
) -> Result<(), OperationError>
where
    B: TreeTransformBackend<D, R::Scalar>,
    R: MultiplicityFreeRigidSymbols,
    R::Scalar: Copy + Add<Output = R::Scalar> + Mul<Output = R::Scalar> + Zero,
    D: TreeTransformScalar,
{
    let structure = tree_pair_transform_structure(rule, operation, dst, src)?;
    tree_transform_execute_with(backend, workspace, &structure, dst, src, alpha, beta)
}

pub fn tree_pair_transform_into_with_context<
    B,
    R,
    D,
    RuleKey,
    const DST_NOUT: usize,
    const DST_NIN: usize,
    const SRC_NOUT: usize,
    const SRC_NIN: usize,
    SDst,
    SSrc,
>(
    context: &mut TreeTransformExecutionContext<D, RuleKey, R::Scalar, B>,
    rule: &R,
    operation: TreeTransformOperationKey,
    dst: &mut TensorMap<D, DST_NOUT, DST_NIN, SDst>,
    src: &TensorMap<D, SRC_NOUT, SRC_NIN, SSrc>,
    alpha: D,
    beta: D,
) -> Result<(), OperationError>
where
    B: TreeTransformBackend<D, R::Scalar>,
    R: MultiplicityFreeRigidSymbols + TreeTransformRuleCacheKey<Key = RuleKey>,
    RuleKey: Clone + Eq + Hash,
    R::Scalar: Copy + Clone + Add<Output = R::Scalar> + Mul<Output = R::Scalar> + Zero,
    D: TreeTransformScalar,
{
    context.tree_pair_transform_into(rule, operation, dst, src, alpha, beta)
}

pub fn all_codomain_tree_transform_into_with_context<
    B,
    R,
    D,
    RuleKey,
    const DST_NOUT: usize,
    const DST_NIN: usize,
    const SRC_NOUT: usize,
    const SRC_NIN: usize,
    SDst,
    SSrc,
>(
    context: &mut TreeTransformExecutionContext<D, RuleKey, R::Scalar, B>,
    rule: &R,
    operation: TreeTransformOperationKey,
    dst: &mut TensorMap<D, DST_NOUT, DST_NIN, SDst>,
    src: &TensorMap<D, SRC_NOUT, SRC_NIN, SSrc>,
    alpha: D,
    beta: D,
) -> Result<(), OperationError>
where
    B: TreeTransformBackend<D, R::Scalar>,
    R: MultiplicityFreeFusionSymbols + TreeTransformRuleCacheKey<Key = RuleKey>,
    RuleKey: Clone + Eq + Hash,
    R::Scalar: Copy + Clone + Add<Output = R::Scalar> + Mul<Output = R::Scalar> + Zero,
    D: TreeTransformScalar,
{
    context.all_codomain_tree_transform_into(rule, operation, dst, src, alpha, beta)
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

fn tensorcontract_structure_with_dense_executor<
    E,
    D,
    C,
    const DST_NOUT: usize,
    const DST_NIN: usize,
    const LHS_NOUT: usize,
    const LHS_NIN: usize,
    const RHS_NOUT: usize,
    const RHS_NIN: usize,
    SDst,
    SLhs,
    SRhs,
>(
    dense: &mut E,
    workspace: &mut TensorContractWorkspace<D>,
    structure: &TensorContractStructure<C>,
    dst: &mut TensorMap<D, DST_NOUT, DST_NIN, SDst>,
    lhs: &TensorMap<D, LHS_NOUT, LHS_NIN, SLhs>,
    rhs: &TensorMap<D, RHS_NOUT, RHS_NIN, SRhs>,
    alpha: D,
    beta: D,
) -> Result<(), OperationError>
where
    E: DenseExecutor,
    D: DenseBlockScalar + RecouplingCoefficientAction<C>,
    C: Copy + One,
{
    structure.validate_replay_structures(dst.structure(), lhs.structure(), rhs.structure())?;
    let descriptor = structure.descriptor();
    let lhs_data = lhs.data();
    let rhs_data = rhs.data();
    let dst_data = dst.data_mut();

    for term in descriptor.terms() {
        workspace.output.resize(term.workspace_len, D::zero());
        let lhs = D::dense_read(
            DenseView::new(
                lhs_data,
                descriptor.lhs_shape(term),
                descriptor.lhs_strides(term),
                term.lhs_offset,
            )
            .map_err(OperationError::Dense)?,
        );
        let rhs = D::dense_read(
            DenseView::new(
                rhs_data,
                descriptor.rhs_shape(term),
                descriptor.rhs_strides(term),
                term.rhs_offset,
            )
            .map_err(OperationError::Dense)?,
        );
        let output = D::dense_write(
            DenseViewMut::new(
                &mut workspace.output,
                descriptor.output_shape(term),
                descriptor.output_strides(term),
                0,
            )
            .map_err(OperationError::Dense)?,
        );
        dense
            .dot_general_into(output, lhs, rhs, descriptor.dot_config())
            .map_err(OperationError::Dense)?;

        let term_alpha = alpha.scale_by_coefficient(term.coefficient);
        let term_beta = if term.apply_beta { beta } else { D::one() };
        tensoradd_raw_strided_kernel(
            &mut workspace.zero_strides,
            dst_data,
            &workspace.output,
            descriptor.scatter_shape(term),
            descriptor.dst_strides(term),
            descriptor.workspace_strides(term),
            term.dst_offset,
            0,
            term_alpha,
            term_beta,
        )?;
    }
    Ok(())
}

fn tree_transform_structure_with_strided_kernel<
    D,
    C,
    const DST_NOUT: usize,
    const DST_NIN: usize,
    const SRC_NOUT: usize,
    const SRC_NIN: usize,
    SDst,
    SSrc,
>(
    workspace: &mut TreeTransformWorkspace<D>,
    structure: &TreeTransformStructure<C>,
    dst: &mut TensorMap<D, DST_NOUT, DST_NIN, SDst>,
    src: &TensorMap<D, SRC_NOUT, SRC_NIN, SSrc>,
    alpha: D,
    beta: D,
) -> Result<(), OperationError>
where
    D: Copy
        + Add<D, Output = D>
        + Mul<D, Output = D>
        + PartialEq
        + Zero
        + One
        + strided_kernel::MaybeSendSync
        + RecouplingCoefficientAction<C>,
    C: Copy,
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

fn tree_transform_structure_with_dense_recoupling<
    E,
    D,
    C,
    const DST_NOUT: usize,
    const DST_NIN: usize,
    const SRC_NOUT: usize,
    const SRC_NIN: usize,
    SDst,
    SSrc,
>(
    dense: &mut E,
    workspace: &mut TreeTransformWorkspace<D>,
    structure: &TreeTransformStructure<C>,
    dst: &mut TensorMap<D, DST_NOUT, DST_NIN, SDst>,
    src: &TensorMap<D, SRC_NOUT, SRC_NIN, SSrc>,
    alpha: D,
    beta: D,
) -> Result<(), OperationError>
where
    E: DenseExecutor,
    D: DenseRecouplingScalar + RecouplingCoefficientAction<C>,
    C: Copy,
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
    tensoradd_raw_strided_kernel(
        zero_strides,
        dst_data,
        src_data,
        descriptor.shape(term),
        descriptor.dst_strides(term),
        descriptor.src_strides(term),
        term.dst_offset,
        term.src_offset,
        alpha,
        beta,
    )
}

#[allow(clippy::too_many_arguments)]
fn tensoradd_raw_strided_kernel<T>(
    zero_strides: &mut Vec<isize>,
    dst_data: &mut [T],
    src_data: &[T],
    shape: &[usize],
    dst_strides: &[isize],
    src_strides: &[isize],
    dst_offset: isize,
    src_offset: isize,
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
    let mut dst = strided_kernel::StridedViewMut::new(dst_data, shape, dst_strides, dst_offset)
        .map_err(strided_error)?;
    let src = strided_kernel::StridedView::<T>::new(src_data, shape, src_strides, src_offset)
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

fn tree_transform_single_with_strided_kernel<D, C>(
    zero_strides: &mut Vec<isize>,
    layouts: &TreeTransformLayoutTable,
    dst_layout: &TreeTransformLayout,
    src_layout: &TreeTransformLayout,
    coefficient: C,
    dst_data: &mut [D],
    src_data: &[D],
    alpha: D,
    beta: D,
) -> Result<(), OperationError>
where
    D: Copy
        + Add<D, Output = D>
        + Mul<D, Output = D>
        + PartialEq
        + Zero
        + One
        + strided_kernel::MaybeSendSync
        + RecouplingCoefficientAction<C>,
    C: Copy,
{
    let shape = layouts.shape(dst_layout);
    let mut dst = strided_kernel::StridedViewMut::new(
        dst_data,
        shape,
        layouts.strides(dst_layout),
        dst_layout.offset,
    )
    .map_err(strided_error)?;
    let src = strided_kernel::StridedView::<D>::new(
        src_data,
        shape,
        layouts.strides(src_layout),
        src_layout.offset,
    )
    .map_err(strided_error)?;
    let scale = alpha.scale_by_coefficient(coefficient);
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
fn tree_transform_multi_with_pack_gemm_scatter<D, C>(
    workspace: &mut TreeTransformWorkspace<D>,
    layouts: &TreeTransformLayoutTable,
    dst_layout_start: usize,
    dst_count: usize,
    src_layout_start: usize,
    src_count: usize,
    coefficient_start: usize,
    element_count: usize,
    coefficients_src_by_dst: &[C],
    dst_data: &mut [D],
    src_data: &[D],
    alpha: D,
    beta: D,
) -> Result<(), OperationError>
where
    D: Copy
        + Add<D, Output = D>
        + Mul<D, Output = D>
        + PartialEq
        + Zero
        + One
        + strided_kernel::MaybeSendSync
        + RecouplingCoefficientAction<C>,
    C: Copy,
{
    let source_len = element_count
        .checked_mul(src_count)
        .ok_or(OperationError::ElementCountOverflow)?;
    let destination_len = element_count
        .checked_mul(dst_count)
        .ok_or(OperationError::ElementCountOverflow)?;
    workspace.source.resize(source_len, D::zero());
    workspace.destination.resize(destination_len, D::zero());

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
fn tree_transform_multi_with_dense_recoupling<E, D, C>(
    dense: &mut E,
    workspace: &mut TreeTransformWorkspace<D>,
    layouts: &TreeTransformLayoutTable,
    dst_layout_start: usize,
    dst_count: usize,
    src_layout_start: usize,
    src_count: usize,
    coefficient_start: usize,
    element_count: usize,
    coefficients_src_by_dst: &[C],
    dst_data: &mut [D],
    src_data: &[D],
    alpha: D,
    beta: D,
) -> Result<(), OperationError>
where
    E: DenseExecutor,
    D: DenseRecouplingScalar + RecouplingCoefficientAction<C>,
    C: Copy,
{
    let source_len = element_count
        .checked_mul(src_count)
        .ok_or(OperationError::ElementCountOverflow)?;
    let destination_len = element_count
        .checked_mul(dst_count)
        .ok_or(OperationError::ElementCountOverflow)?;
    workspace.source.resize(source_len, D::zero());
    workspace.destination.resize(destination_len, D::zero());

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

    let coefficient_count = src_count
        .checked_mul(dst_count)
        .ok_or(OperationError::ElementCountOverflow)?;
    let coefficient_end = coefficient_start
        .checked_add(coefficient_count)
        .ok_or(OperationError::ElementCountOverflow)?;
    if coefficients_src_by_dst.len() < coefficient_end {
        return Err(OperationError::CoefficientCountMismatch {
            expected: coefficient_end,
            actual: coefficients_src_by_dst.len(),
        });
    }
    workspace.coefficients.clear();
    workspace.coefficients.extend(
        coefficients_src_by_dst[coefficient_start..coefficient_end]
            .iter()
            .copied()
            .map(D::coefficient_as_data),
    );

    apply_recoupling_matrix_with_dense_executor(
        dense,
        &mut workspace.destination,
        &workspace.source,
        &workspace.coefficients,
        0,
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
fn apply_recoupling_matrix_src_times_u_transpose<D, C>(
    destination: &mut [D],
    source: &[D],
    coefficients_src_by_dst: &[C],
    coefficient_start: usize,
    element_count: usize,
    src_count: usize,
    dst_count: usize,
) -> Result<(), OperationError>
where
    D: Copy + Add<D, Output = D> + Zero + RecouplingCoefficientAction<C>,
    C: Copy,
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
            let mut sum = D::zero();
            for src_index in 0..src_count {
                let coeff = coefficients_src_by_dst[coefficient_row_start + src_index];
                let src_value = source[element + src_index * element_count];
                sum = sum + src_value.scale_by_coefficient(coeff);
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

fn column_major_strides_usize(shape: &[usize]) -> Result<Vec<usize>, OperationError> {
    let mut stride = 1usize;
    let mut strides = Vec::with_capacity(shape.len());
    for &dim in shape {
        strides.push(stride);
        stride = stride
            .checked_mul(dim)
            .ok_or(OperationError::ElementCountOverflow)?;
    }
    Ok(strides)
}

fn validate_axis_subset(
    tensor: &'static str,
    axes: &[usize],
    rank: usize,
) -> Result<Vec<bool>, OperationError> {
    let mut seen = vec![false; rank];
    for &axis in axes {
        if axis >= rank || seen[axis] {
            return Err(OperationError::InvalidAxisSet {
                tensor,
                axes: axes.to_vec(),
                rank,
            });
        }
        seen[axis] = true;
    }
    Ok(seen)
}

fn contracted_external_sectors_match(
    lhs_external: &[SectorId],
    rhs_external: &[SectorId],
    lhs_axes: &[usize],
    rhs_axes: &[usize],
) -> bool {
    lhs_axes
        .iter()
        .zip(rhs_axes)
        .all(|(&lhs_axis, &rhs_axis)| lhs_external[lhs_axis] == rhs_external[rhs_axis])
}

fn contracted_fusion_tree_basis_matches<R>(
    rule: &R,
    lhs_domain: &FusionTreeKey,
    rhs_codomain: &FusionTreeKey,
) -> bool
where
    R: FusionRule,
{
    lhs_domain.uncoupled().len() == rhs_codomain.uncoupled().len()
        && lhs_domain.innerlines().len() == rhs_codomain.innerlines().len()
        && lhs_domain.vertices() == rhs_codomain.vertices()
        && lhs_domain.is_dual() == rhs_codomain.is_dual()
        && lhs_domain
            .uncoupled()
            .iter()
            .copied()
            .map(|sector| rule.dual(sector))
            .eq(rhs_codomain.uncoupled().iter().copied())
        && lhs_domain
            .innerlines()
            .iter()
            .copied()
            .map(|sector| rule.dual(sector))
            .eq(rhs_codomain.innerlines().iter().copied())
        && rule.dual(lhs_domain.coupled().unwrap_or_else(|| rule.vacuum()))
            == rhs_codomain.coupled().unwrap_or_else(|| rule.vacuum())
}

fn contracted_output_external_sectors(
    lhs_external: &[SectorId],
    rhs_external: &[SectorId],
    axis_plan: &TensorContractAxisPlan,
) -> Vec<SectorId> {
    let mut canonical = axis_plan
        .lhs_open_axes
        .iter()
        .map(|&axis| lhs_external[axis])
        .collect::<Vec<_>>();
    canonical.extend(
        axis_plan
            .rhs_open_axes
            .iter()
            .map(|&axis| rhs_external[axis]),
    );
    axis_plan
        .output_axes
        .iter()
        .map(|&axis| canonical[axis])
        .collect()
}

fn is_canonical_fusion_compose_contract(
    lhs: &FusionTreeHomSpace,
    rhs: &FusionTreeHomSpace,
    lhs_contracting_axes: &[usize],
    rhs_contracting_axes: &[usize],
    output_axes: &[usize],
    dst_codomain_rank: usize,
) -> bool {
    let canonical_output_rank = lhs.codomain().len() + rhs.domain().len();
    let canonical_output_axes = (0..canonical_output_rank).collect::<Vec<_>>();
    is_canonical_fusion_source_contract(lhs, rhs, lhs_contracting_axes, rhs_contracting_axes)
        && output_axes == canonical_output_axes.as_slice()
        && dst_codomain_rank == lhs.codomain().len()
}

fn is_canonical_fusion_source_contract(
    lhs: &FusionTreeHomSpace,
    rhs: &FusionTreeHomSpace,
    lhs_contracting_axes: &[usize],
    rhs_contracting_axes: &[usize],
) -> bool {
    let lhs_domain_axes =
        (lhs.codomain().len()..lhs.codomain().len() + lhs.domain().len()).collect::<Vec<_>>();
    let rhs_codomain_axes = (0..rhs.codomain().len()).collect::<Vec<_>>();
    lhs_contracting_axes == lhs_domain_axes.as_slice()
        && rhs_contracting_axes == rhs_codomain_axes.as_slice()
}

fn validate_block_index(
    tensor: &'static str,
    index: usize,
    count: usize,
) -> Result<(), OperationError> {
    if index < count {
        Ok(())
    } else {
        Err(OperationError::BlockIndexOutOfBounds {
            tensor,
            index,
            count,
        })
    }
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
mod tests;
