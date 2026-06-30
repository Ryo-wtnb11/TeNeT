#![forbid(unsafe_code)]

//! TensorOperations-style lowering for TeNeT.
//!
//! Public/core tensor code talks in terms of TeNeT-owned block views. This crate
//! lowers those views to strided-rs kernels at the same granularity that
//! TensorKit uses Strided.jl/StridedViews.jl internally.

use core::fmt;
use core::ops::{Add, Mul};
use std::any::TypeId;
use std::collections::HashMap;
use std::hash::Hash;
use std::sync::Arc;

use num_complex::{Complex32, Complex64};
use num_traits::{One, Zero};
use tenet_core::{
    multiplicity_free_braid_tree, multiplicity_free_braid_tree_pair,
    multiplicity_free_permute_tree, multiplicity_free_permute_tree_pair,
    multiplicity_free_transpose_tree_pair, BlockKey, BlockLayout, BlockStructure, BlockView,
    BlockViewMut, BraidingStyleKind, CoreError, FermionParityFusionRule, FusionRule,
    FusionStyleKind, FusionTensorMapSpace, FusionTreeBlockGroup, FusionTreeBlockKey,
    FusionTreeGroupKey, FusionTreeHomSpace, FusionTreeKey, MultiplicityFreeFusionSymbols,
    MultiplicityFreeRigidSymbols, ProductFusionRule, ProductSectorCodec, SU2FusionRule, SectorId,
    TensorMap, U1FusionRule, Z2FusionRule,
};
#[cfg(test)]
use tenet_core::{
    unique_braid_tree, unique_braid_tree_pair, unique_permute_tree, unique_permute_tree_pair,
    unique_transpose_tree_pair, MultiplicityFreePivotalSymbols,
};
use tenet_dense::{
    DefaultDenseExecutor, DenseDotConfig, DenseError, DenseExecutor, DenseRead, DenseView,
    DenseViewMut, DenseWrite,
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

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct TensorContractAxisSpec<'a> {
    lhs_contracting_axes: &'a [usize],
    rhs_contracting_axes: &'a [usize],
    output_permutation: AxisPermutation<'a>,
}

impl<'a> TensorContractAxisSpec<'a> {
    pub fn new(
        lhs_contracting_axes: &'a [usize],
        rhs_contracting_axes: &'a [usize],
        output_permutation: AxisPermutation<'a>,
    ) -> Self {
        Self {
            lhs_contracting_axes,
            rhs_contracting_axes,
            output_permutation,
        }
    }

    pub fn canonical(lhs_contracting_axes: &'a [usize], rhs_contracting_axes: &'a [usize]) -> Self {
        Self::new(
            lhs_contracting_axes,
            rhs_contracting_axes,
            AxisPermutation::identity(),
        )
    }

    #[inline]
    pub fn lhs_contracting_axes(&self) -> &'a [usize] {
        self.lhs_contracting_axes
    }

    #[inline]
    pub fn rhs_contracting_axes(&self) -> &'a [usize] {
        self.rhs_contracting_axes
    }

    #[inline]
    pub fn output_permutation(&self) -> AxisPermutation<'a> {
        self.output_permutation
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
        validate_all_codomain_fusion_tree_block(index, src_key)?;

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
            validate_all_codomain_fusion_tree_block(src_block_index, src_key)?;
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

fn validate_all_codomain_fusion_tree_block(
    index: usize,
    key: &FusionTreeBlockKey,
) -> Result<(), OperationError> {
    let domain = key.domain_tree();
    if domain.uncoupled().is_empty()
        && domain.coupled().is_none()
        && domain.is_dual().is_empty()
        && domain.innerlines().is_empty()
        && domain.vertices().is_empty()
    {
        return Ok(());
    }
    Err(OperationError::ExpectedAllCodomainFusionTree { index })
}

#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub enum TreeTransformOperationKey {
    Transpose {
        codomain_permutation: Vec<usize>,
        domain_permutation: Vec<usize>,
    },
    Permute {
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
    /// Build a planar transpose operation.
    ///
    /// The two permutations follow TensorKit's `Index2Tuple` convention:
    /// both `codomain_permutation` and `domain_permutation` contain source
    /// tensor axis numbers in the full `0..numind` range. They are not local
    /// permutations within the old codomain/domain parts. For example, for a
    /// `(NOUT, NIN) = (2, 1)` tensor, keeping the domain leg in the domain uses
    /// `domain_permutation = [2]`, not `[0]`.
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

    /// Build a symmetric-braiding permutation operation.
    ///
    /// Axis numbering follows TensorKit's `Index2Tuple` convention; see
    /// [`Self::transpose`].
    pub fn permute<Codomain, Domain>(
        codomain_permutation: Codomain,
        domain_permutation: Domain,
    ) -> Self
    where
        Codomain: IntoIterator<Item = usize>,
        Domain: IntoIterator<Item = usize>,
    {
        Self::Permute {
            codomain_permutation: codomain_permutation.into_iter().collect(),
            domain_permutation: domain_permutation.into_iter().collect(),
        }
    }

    /// Build an explicit braid operation with source-axis permutations and levels.
    ///
    /// Axis numbering follows TensorKit's `Index2Tuple` convention; see
    /// [`Self::transpose`]. `codomain_levels` and `domain_levels` are the
    /// levels of the source axes selected by each output tuple.
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

    pub fn requires_symmetric_braiding(&self) -> bool {
        matches!(self, Self::Permute { .. })
    }

    pub fn validate_braiding_support<R>(&self, rule: &R) -> Result<(), OperationError>
    where
        R: FusionRule,
    {
        if self.requires_symmetric_braiding() && !rule.braiding_style().is_symmetric() {
            return Err(OperationError::UnsupportedBraidingStyle {
                operation: self.clone(),
                style: rule.braiding_style(),
            });
        }
        Ok(())
    }
}

pub trait TreeTransformRuleCacheKey {
    type Key: Clone + Eq + Hash;

    fn tree_transform_rule_cache_key(&self) -> Self::Key;
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum TreeTransformBuiltinRuleCacheKey {
    Z2,
    FermionParity,
    U1,
    SU2,
}

impl TreeTransformRuleCacheKey for Z2FusionRule {
    type Key = TreeTransformBuiltinRuleCacheKey;

    fn tree_transform_rule_cache_key(&self) -> Self::Key {
        TreeTransformBuiltinRuleCacheKey::Z2
    }
}

impl TreeTransformRuleCacheKey for FermionParityFusionRule {
    type Key = TreeTransformBuiltinRuleCacheKey;

    fn tree_transform_rule_cache_key(&self) -> Self::Key {
        TreeTransformBuiltinRuleCacheKey::FermionParity
    }
}

impl TreeTransformRuleCacheKey for U1FusionRule {
    type Key = TreeTransformBuiltinRuleCacheKey;

    fn tree_transform_rule_cache_key(&self) -> Self::Key {
        TreeTransformBuiltinRuleCacheKey::U1
    }
}

impl TreeTransformRuleCacheKey for SU2FusionRule {
    type Key = TreeTransformBuiltinRuleCacheKey;

    fn tree_transform_rule_cache_key(&self) -> Self::Key {
        TreeTransformBuiltinRuleCacheKey::SU2
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub struct TreeTransformProductRuleCacheKey<LeftKey, RightKey> {
    left: LeftKey,
    right: RightKey,
    codec: TypeId,
}

impl<LeftKey, RightKey> TreeTransformProductRuleCacheKey<LeftKey, RightKey> {
    pub fn new<Codec>(left: LeftKey, right: RightKey) -> Self
    where
        Codec: 'static,
    {
        Self {
            left,
            right,
            codec: TypeId::of::<Codec>(),
        }
    }

    #[inline]
    pub fn left(&self) -> &LeftKey {
        &self.left
    }

    #[inline]
    pub fn right(&self) -> &RightKey {
        &self.right
    }
}

impl<LeftRule, RightRule, Codec> TreeTransformRuleCacheKey
    for ProductFusionRule<LeftRule, RightRule, Codec>
where
    LeftRule: TreeTransformRuleCacheKey,
    RightRule: TreeTransformRuleCacheKey,
    Codec: ProductSectorCodec + 'static,
{
    type Key = TreeTransformProductRuleCacheKey<LeftRule::Key, RightRule::Key>;

    fn tree_transform_rule_cache_key(&self) -> Self::Key {
        TreeTransformProductRuleCacheKey::new::<Codec>(
            self.left_rule().tree_transform_rule_cache_key(),
            self.right_rule().tree_transform_rule_cache_key(),
        )
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum TreeTransformPlanScope {
    AllCodomain,
    TreePair,
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
    ContractAxisCountMismatch {
        lhs: usize,
        rhs: usize,
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
    ExpectedFusionTreeBlock {
        tensor: &'static str,
        index: usize,
    },
    ExpectedAllCodomainFusionTree {
        index: usize,
    },
    InvalidPermutation {
        axes: Vec<usize>,
        rank: usize,
    },
    InvalidAxisSet {
        tensor: &'static str,
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
    UnsupportedFusionStyle {
        operation: TreeTransformOperationKey,
        style: FusionStyleKind,
    },
    UnsupportedBraidingStyle {
        operation: TreeTransformOperationKey,
        style: BraidingStyleKind,
    },
    UnsupportedTreeTransformScope {
        operation: TreeTransformOperationKey,
        message: &'static str,
    },
    UnsupportedTensorContractScope {
        message: &'static str,
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
            Self::ContractAxisCountMismatch { lhs, rhs } => {
                write!(f, "contracting axis count mismatch: lhs {lhs}, rhs {rhs}")
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
            Self::ExpectedFusionTreeBlock { tensor, index } => {
                write!(f, "{tensor} block {index} is not a fusion-tree block")
            }
            Self::ExpectedAllCodomainFusionTree { index } => {
                write!(
                    f,
                    "source fusion-tree block {index} is not an all-codomain tree"
                )
            }
            Self::InvalidPermutation { axes, rank } => {
                write!(f, "invalid axis permutation {axes:?} for rank {rank}")
            }
            Self::InvalidAxisSet { tensor, axes, rank } => {
                write!(f, "invalid {tensor} axis set {axes:?} for rank {rank}")
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
            Self::UnsupportedFusionStyle { operation, style } => {
                write!(
                    f,
                    "unsupported fusion style {style:?} for tree transform operation {operation:?}"
                )
            }
            Self::UnsupportedBraidingStyle { operation, style } => {
                write!(
                    f,
                    "unsupported braiding style {style:?} for tree transform operation {operation:?}"
                )
            }
            Self::UnsupportedTreeTransformScope { operation, message } => {
                write!(
                    f,
                    "unsupported tree transform scope for operation {operation:?}: {message}"
                )
            }
            Self::UnsupportedTensorContractScope { message } => {
                write!(f, "unsupported tensor contraction scope: {message}")
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
    let lhs_domain_axes =
        (lhs.codomain().len()..lhs.codomain().len() + lhs.domain().len()).collect::<Vec<_>>();
    let rhs_codomain_axes = (0..rhs.codomain().len()).collect::<Vec<_>>();
    let canonical_output_rank = lhs.codomain().len() + rhs.domain().len();
    let canonical_output_axes = (0..canonical_output_rank).collect::<Vec<_>>();
    lhs_contracting_axes == lhs_domain_axes.as_slice()
        && rhs_contracting_axes == rhs_codomain_axes.as_slice()
        && output_axes == canonical_output_axes.as_slice()
        && dst_codomain_rank == lhs.codomain().len()
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
mod tests {
    use super::*;
    use num_complex::{Complex32, Complex64};
    use std::fmt::Debug;
    use tenet_core::{
        BlockSpec, BraidingStyleKind, FermionParityFusionRule, FusionProductSpace,
        FusionTensorMapSpace, FusionTreeHomSpace, FusionTreeKey, MultiplicityFreeFusionRule,
        MultiplicityFreeFusionSymbols, ProductFusionRule, SU2FusionRule, SU2Irrep, SectorId,
        SectorLeg, TensorMapSpace, U1FusionRule, U1Irrep, Z2FusionRule,
    };

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

    fn expect_tree_key(key: &BlockKey) -> FusionTreeBlockKey {
        match key {
            BlockKey::FusionTree(tree) => tree.clone(),
            BlockKey::Dense => panic!("test expected a fusion-tree key"),
        }
    }

    fn empty_fusion_tree() -> FusionTreeKey {
        FusionTreeKey::new(
            Vec::<SectorId>::new(),
            None,
            Vec::<bool>::new(),
            Vec::<SectorId>::new(),
            Vec::<SectorId>::new(),
        )
    }

    fn all_codomain_fusion_tree_test_key<
        const COD: usize,
        const COD_DUAL: usize,
        const COD_INNER: usize,
        const COD_VERTICES: usize,
    >(
        codomain: [usize; COD],
        coupled: Option<usize>,
        codomain_is_dual: [bool; COD_DUAL],
        codomain_innerlines: [usize; COD_INNER],
        codomain_vertices: [usize; COD_VERTICES],
    ) -> BlockKey {
        BlockKey::from(FusionTreeBlockKey::pair(
            FusionTreeKey::from_sector_ids(
                codomain,
                coupled,
                codomain_is_dual,
                codomain_innerlines,
                codomain_vertices,
            ),
            empty_fusion_tree(),
        ))
    }

    type FpU1Rule = ProductFusionRule<FermionParityFusionRule, U1FusionRule>;
    type FpU1Su2Rule = ProductFusionRule<FpU1Rule, SU2FusionRule>;

    fn fz2_u1_su2_tree_pair_fixture() -> (
        FpU1Su2Rule,
        FusionTensorMapSpace<2, 1>,
        FusionTensorMapSpace<2, 1>,
        [SectorId; 2],
    ) {
        let left_rule = FpU1Rule::default();
        let rule = FpU1Su2Rule::default();
        let even = SectorId::new(0);
        let odd = SectorId::new(1);
        let left_sector =
            |parity, charge| left_rule.encode_sector(parity, U1Irrep::new(charge).sector_id());
        let sector = |parity, charge, twice_spin| {
            rule.encode_sector(
                left_sector(parity, charge),
                SU2Irrep::from_twice_spin(twice_spin).sector_id(),
            )
        };

        let a = sector(odd, 1, 1);
        let b = sector(odd, -1, 1);
        let c0 = sector(even, 0, 0);
        let c1 = sector(even, 0, 2);
        let src_hom = FusionTreeHomSpace::new(
            FusionProductSpace::new([SectorLeg::new([a], false), SectorLeg::new([b], false)]),
            FusionProductSpace::new([SectorLeg::new([c0, c1], false)]),
        );
        let dst_hom = FusionTreeHomSpace::new(
            FusionProductSpace::new([SectorLeg::new([b], false), SectorLeg::new([a], false)]),
            FusionProductSpace::new([SectorLeg::new([c0, c1], false)]),
        );
        let src_space = FusionTensorMapSpace::from_degeneracy_shapes(
            TensorMapSpace::<2, 1>::from_dims([1, 1], [1]).unwrap(),
            src_hom,
            &rule,
            [vec![1, 1, 1], vec![1, 1, 1]],
        )
        .unwrap();
        let dst_space = FusionTensorMapSpace::from_degeneracy_shapes(
            TensorMapSpace::<2, 1>::from_dims([1, 1], [1]).unwrap(),
            dst_hom,
            &rule,
            [vec![1, 1, 1], vec![1, 1, 1]],
        )
        .unwrap();

        (rule, src_space, dst_space, [c0, c1])
    }

    fn single_transform_coefficient_for_coupled(
        plan: &TreeTransformGroupPlan<f64>,
        coupled: SectorId,
    ) -> f64 {
        let mut found = None;
        for spec in plan.specs() {
            assert_eq!(spec.src_keys().len(), 1);
            assert_eq!(spec.dst_keys().len(), 1);
            assert_eq!(spec.coefficients_src_by_dst().len(), 1);
            let dst_coupled = expect_tree_key(&spec.dst_keys()[0]).coupled().unwrap();
            if dst_coupled == coupled {
                assert!(found.is_none(), "duplicate coefficient for {coupled:?}");
                found = Some(spec.coefficients_src_by_dst()[0]);
            }
        }
        found.unwrap_or_else(|| panic!("missing coefficient for {coupled:?}"))
    }

    fn expected_single_tree_pair_replay(
        plan: &TreeTransformGroupPlan<f64>,
        dst_structure: &BlockStructure,
        src_structure: &BlockStructure,
        initial_dst: &[f64],
        src_data: &[f64],
        alpha: f64,
        beta: f64,
    ) -> Vec<f64> {
        let mut expected = initial_dst
            .iter()
            .map(|value| beta * value)
            .collect::<Vec<_>>();
        for spec in plan.specs() {
            assert_eq!(spec.src_keys().len(), 1);
            assert_eq!(spec.dst_keys().len(), 1);
            assert_eq!(spec.coefficients_src_by_dst().len(), 1);
            let src_key = &spec.src_keys()[0];
            let dst_key = &spec.dst_keys()[0];
            let src_offset = src_structure.block_by_key(src_key).unwrap().offset();
            let dst_offset = dst_structure.block_by_key(dst_key).unwrap().offset();
            expected[dst_offset] +=
                alpha * spec.coefficients_src_by_dst()[0] * src_data[src_offset];
        }
        expected
    }

    fn column_major_structure_like(
        structure: &BlockStructure,
        shape: Vec<usize>,
    ) -> BlockStructure {
        let blocks = (0..structure.block_count())
            .map(|index| (structure.block(index).unwrap().key().clone(), shape.clone()));
        BlockStructure::packed_column_major_with_keys(structure.rank(), blocks).unwrap()
    }

    #[derive(Clone, Copy, Debug)]
    struct UniqueZ2Rule;

    impl FusionRule for UniqueZ2Rule {
        fn fusion_style(&self) -> FusionStyleKind {
            FusionStyleKind::Unique
        }

        fn braiding_style(&self) -> BraidingStyleKind {
            BraidingStyleKind::Bosonic
        }

        fn vacuum(&self) -> SectorId {
            SectorId::new(0)
        }

        fn fusion_channels(&self, left: SectorId, right: SectorId) -> Vec<SectorId> {
            vec![SectorId::new((left.id() + right.id()) % 2)]
        }
    }

    impl MultiplicityFreeFusionRule for UniqueZ2Rule {}

    impl MultiplicityFreeFusionSymbols for UniqueZ2Rule {
        type Scalar = f64;

        fn scalar_one(&self) -> Self::Scalar {
            1.0
        }

        fn scalar_conj(&self, value: Self::Scalar) -> Self::Scalar {
            value
        }

        fn f_symbol_scalar(
            &self,
            _left: SectorId,
            _middle: SectorId,
            _right: SectorId,
            _coupled: SectorId,
            _left_coupled: SectorId,
            _right_coupled: SectorId,
        ) -> Self::Scalar {
            1.0
        }

        fn r_symbol_scalar(
            &self,
            _left: SectorId,
            _right: SectorId,
            _coupled: SectorId,
        ) -> Self::Scalar {
            1.0
        }
    }

    impl MultiplicityFreePivotalSymbols for UniqueZ2Rule {
        fn bendright_scalar(
            &self,
            _left_coupled: SectorId,
            _bent_sector: SectorId,
            _coupled: SectorId,
            _bent_leg_is_dual: bool,
        ) -> Self::Scalar {
            1.0
        }

        fn foldright_scalar(
            &self,
            _source: &FusionTreeBlockKey,
            _destination: &FusionTreeBlockKey,
        ) -> Self::Scalar {
            1.0
        }
    }

    #[derive(Clone, Copy, Debug)]
    struct UniqueAnyonicRule;

    impl FusionRule for UniqueAnyonicRule {
        fn fusion_style(&self) -> FusionStyleKind {
            FusionStyleKind::Unique
        }

        fn braiding_style(&self) -> BraidingStyleKind {
            BraidingStyleKind::Anyonic
        }

        fn vacuum(&self) -> SectorId {
            SectorId::new(0)
        }

        fn fusion_channels(&self, left: SectorId, right: SectorId) -> Vec<SectorId> {
            vec![SectorId::new((left.id() + right.id()) % 2)]
        }
    }

    impl MultiplicityFreeFusionRule for UniqueAnyonicRule {}

    impl MultiplicityFreeFusionSymbols for UniqueAnyonicRule {
        type Scalar = f64;

        fn scalar_one(&self) -> Self::Scalar {
            1.0
        }

        fn scalar_conj(&self, value: Self::Scalar) -> Self::Scalar {
            value
        }

        fn f_symbol_scalar(
            &self,
            _left: SectorId,
            _middle: SectorId,
            _right: SectorId,
            _coupled: SectorId,
            _left_coupled: SectorId,
            _right_coupled: SectorId,
        ) -> Self::Scalar {
            1.0
        }

        fn r_symbol_scalar(
            &self,
            left: SectorId,
            right: SectorId,
            _coupled: SectorId,
        ) -> Self::Scalar {
            if left.id() == 1 && right.id() == 1 {
                -2.0
            } else {
                1.0
            }
        }
    }

    impl MultiplicityFreePivotalSymbols for UniqueAnyonicRule {
        fn bendright_scalar(
            &self,
            _left_coupled: SectorId,
            _bent_sector: SectorId,
            _coupled: SectorId,
            _bent_leg_is_dual: bool,
        ) -> Self::Scalar {
            1.0
        }

        fn foldright_scalar(
            &self,
            _source: &FusionTreeBlockKey,
            _destination: &FusionTreeBlockKey,
        ) -> Self::Scalar {
            1.0
        }
    }

    #[derive(Clone, Copy, Debug)]
    struct UniquePlanarRule;

    impl FusionRule for UniquePlanarRule {
        fn fusion_style(&self) -> FusionStyleKind {
            FusionStyleKind::Unique
        }

        fn braiding_style(&self) -> BraidingStyleKind {
            BraidingStyleKind::NoBraiding
        }

        fn vacuum(&self) -> SectorId {
            SectorId::new(0)
        }

        fn fusion_channels(&self, left: SectorId, right: SectorId) -> Vec<SectorId> {
            vec![SectorId::new((left.id() + right.id()) % 2)]
        }
    }

    impl MultiplicityFreeFusionRule for UniquePlanarRule {}

    #[derive(Clone, Copy, Debug)]
    struct SimpleSu2Rule;

    impl FusionRule for SimpleSu2Rule {
        fn fusion_style(&self) -> FusionStyleKind {
            FusionStyleKind::Simple
        }

        fn braiding_style(&self) -> BraidingStyleKind {
            BraidingStyleKind::Bosonic
        }

        fn vacuum(&self) -> SectorId {
            SectorId::new(0)
        }

        fn fusion_channels(&self, left: SectorId, right: SectorId) -> Vec<SectorId> {
            let min = left.id().abs_diff(right.id());
            let max = left.id() + right.id();
            (min..=max).step_by(2).map(SectorId::new).collect()
        }
    }

    impl MultiplicityFreeFusionRule for SimpleSu2Rule {}

    #[derive(Clone, Copy, Debug)]
    struct GenericMultiplicityRule;

    impl FusionRule for GenericMultiplicityRule {
        fn fusion_style(&self) -> FusionStyleKind {
            FusionStyleKind::Generic
        }

        fn braiding_style(&self) -> BraidingStyleKind {
            BraidingStyleKind::Anyonic
        }

        fn vacuum(&self) -> SectorId {
            SectorId::new(0)
        }

        fn fusion_channels(&self, left: SectorId, right: SectorId) -> Vec<SectorId> {
            match (left.id(), right.id()) {
                (1, 1) => vec![SectorId::new(0), SectorId::new(1)],
                (0, x) | (x, 0) => vec![SectorId::new(x)],
                _ => Vec::new(),
            }
        }

        fn nsymbol(&self, left: SectorId, right: SectorId, coupled: SectorId) -> usize {
            match (left.id(), right.id(), coupled.id()) {
                (1, 1, 1) => 2,
                _ => usize::from(self.fusion_channels(left, right).contains(&coupled)),
            }
        }
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
            + strided_kernel::MaybeSendSync
            + RecouplingCoefficientAction<T>,
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
            + strided_kernel::MaybeSendSync
            + RecouplingCoefficientAction<T>,
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

    fn assert_tree_single_mixed_dtype<D, C>(
        values: Vec<D>,
        fill: D,
        coefficient: C,
        alpha: D,
        beta: D,
        expected: Vec<D>,
    ) where
        D: TreeTransformScalar + RecouplingCoefficientAction<C> + Clone + Debug,
        C: Copy,
    {
        let space = TensorMapSpace::<2, 0>::from_dims([2, 2], []).unwrap();
        let src = TensorMap::<D, 2, 0>::from_vec(values, space.clone()).unwrap();
        let mut dst = TensorMap::<D, 2, 0>::filled(fill, space).unwrap();
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

    fn assert_tree_multi_mixed_dtype<D, C>(
        src_values: Vec<D>,
        coefficients_src_by_dst: Vec<C>,
        alpha: D,
        beta: D,
        fill: D,
        expected: Vec<D>,
    ) where
        D: TreeTransformScalar + RecouplingCoefficientAction<C> + Clone + Debug,
        C: Copy,
    {
        let space = TensorMapSpace::<2, 0>::from_dims([4, 2], []).unwrap();
        let src_structure =
            BlockStructure::packed_column_major(2, [vec![2, 2], vec![2, 2]]).unwrap();
        let dst_structure =
            BlockStructure::packed_column_major(2, [vec![4, 1], vec![4, 1]]).unwrap();
        let src =
            TensorMap::<D, 2, 0>::from_vec_with_structure(src_values, space.clone(), src_structure)
                .unwrap();
        let mut dst =
            TensorMap::<D, 2, 0>::from_vec_with_structure(vec![fill; 8], space, dst_structure)
                .unwrap();
        let structure = TreeTransformStructure::compile(
            &dst,
            &src,
            &[TreeTransformBlockSpec::multi(
                vec![0, 1],
                vec![0, 1],
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
            + strided_kernel::MaybeSendSync
            + RecouplingCoefficientAction<T>,
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
            + strided_kernel::MaybeSendSync
            + RecouplingCoefficientAction<T>,
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
    fn tree_transform_single_replay_supports_complex_data_with_real_coefficients() {
        assert_tree_single_mixed_dtype(
            vec![
                Complex32::new(1.0, 1.0),
                Complex32::new(2.0, -1.0),
                Complex32::new(3.0, 0.5),
                Complex32::new(4.0, -0.5),
            ],
            Complex32::new(10.0, 1.0),
            3.0_f64,
            Complex32::new(2.0, 0.0),
            Complex32::new(4.0, 0.0),
            vec![
                Complex32::new(46.0, 10.0),
                Complex32::new(52.0, -2.0),
                Complex32::new(58.0, 7.0),
                Complex32::new(64.0, 1.0),
            ],
        );
        assert_tree_single_mixed_dtype(
            vec![
                Complex64::new(1.0, 1.0),
                Complex64::new(2.0, -1.0),
                Complex64::new(3.0, 0.5),
                Complex64::new(4.0, -0.5),
            ],
            Complex64::new(10.0, 1.0),
            3.0_f64,
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
    fn tree_transform_multi_pack_gemm_scatter_supports_complex_data_with_real_coefficients() {
        assert_tree_multi_mixed_dtype(
            vec![
                Complex32::new(1.0, 0.0),
                Complex32::new(2.0, 0.0),
                Complex32::new(3.0, 0.0),
                Complex32::new(4.0, 0.0),
                Complex32::new(5.0, 0.0),
                Complex32::new(6.0, 0.0),
                Complex32::new(7.0, 0.0),
                Complex32::new(8.0, 0.0),
            ],
            vec![2.0_f64, 3.0, 5.0, 7.0],
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
        assert_tree_multi_mixed_dtype(
            vec![
                Complex64::new(1.0, 0.0),
                Complex64::new(2.0, 0.0),
                Complex64::new(3.0, 0.0),
                Complex64::new(4.0, 0.0),
                Complex64::new(5.0, 0.0),
                Complex64::new(6.0, 0.0),
                Complex64::new(7.0, 0.0),
                Complex64::new(8.0, 0.0),
            ],
            vec![2.0_f64, 3.0, 5.0, 7.0],
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
    fn tensorcontract_structure_precomputes_canonical_dense_descriptor() {
        let lhs_space = TensorMapSpace::<2, 0>::from_dims([2, 3], []).unwrap();
        let rhs_space = TensorMapSpace::<2, 0>::from_dims([3, 2], []).unwrap();
        let dst_space = TensorMapSpace::<2, 0>::from_dims([2, 2], []).unwrap();
        let lhs = TensorMap::<f64, 2, 0>::from_vec(vec![1.0; 6], lhs_space).unwrap();
        let rhs = TensorMap::<f64, 2, 0>::from_vec(vec![1.0; 6], rhs_space).unwrap();
        let dst = TensorMap::<f64, 2, 0>::filled(0.0, dst_space).unwrap();

        let structure = TensorContractStructure::compile(
            &dst,
            &lhs,
            &rhs,
            TensorContractAxisSpec::canonical(&[1], &[0]),
        )
        .unwrap();

        assert_eq!(structure.dst_rank(), 2);
        assert_eq!(structure.lhs_rank(), 2);
        assert_eq!(structure.rhs_rank(), 2);
        assert_eq!(structure.lhs_contracting_axes(), &[1]);
        assert_eq!(structure.rhs_contracting_axes(), &[0]);
        assert_eq!(structure.output_axes(), &[0, 1]);
        assert_eq!(structure.terms().len(), 1);
        assert_eq!(structure.terms()[0].key(), &BlockKey::trivial());
        assert_eq!(structure.terms()[0].dst_block(), 0);
        assert_eq!(structure.terms()[0].lhs_block(), 0);
        assert_eq!(structure.terms()[0].rhs_block(), 0);
    }

    #[test]
    fn tensorcontract_into_uses_dense_backend_for_matmul_and_alpha_beta() {
        let lhs_space = TensorMapSpace::<2, 0>::from_dims([2, 3], []).unwrap();
        let rhs_space = TensorMapSpace::<2, 0>::from_dims([3, 2], []).unwrap();
        let dst_space = TensorMapSpace::<2, 0>::from_dims([2, 2], []).unwrap();
        let lhs = TensorMap::<f64, 2, 0>::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], lhs_space)
            .unwrap();
        let rhs =
            TensorMap::<f64, 2, 0>::from_vec(vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0], rhs_space)
                .unwrap();
        let mut dst = TensorMap::<f64, 2, 0>::filled(1.0, dst_space).unwrap();

        tensorcontract_into(
            &mut dst,
            &lhs,
            &rhs,
            TensorContractAxisSpec::canonical(&[1], &[0]),
            2.0,
            3.0,
        )
        .unwrap();

        assert_eq!(dst.data(), &[155.0, 203.0, 209.0, 275.0]);
    }

    #[derive(Default)]
    struct ContractLayoutCheckingDenseExecutor {
        dot_general_into_calls: usize,
    }

    impl DenseExecutor for ContractLayoutCheckingDenseExecutor {
        fn svd(
            &mut self,
            _input: DenseRead<'_>,
        ) -> Result<Vec<tenet_dense::DenseTensor>, DenseError> {
            unreachable!("tensor contraction does not call svd")
        }

        fn qr(
            &mut self,
            _input: DenseRead<'_>,
        ) -> Result<Vec<tenet_dense::DenseTensor>, DenseError> {
            unreachable!("tensor contraction does not call qr")
        }

        fn eigh(
            &mut self,
            _input: DenseRead<'_>,
        ) -> Result<Vec<tenet_dense::DenseTensor>, DenseError> {
            unreachable!("tensor contraction does not call eigh")
        }

        fn dot_general_into(
            &mut self,
            output: DenseWrite<'_>,
            lhs: DenseRead<'_>,
            rhs: DenseRead<'_>,
            config: &DenseDotConfig,
        ) -> Result<(), DenseError> {
            self.dot_general_into_calls += 1;
            assert_eq!(
                config,
                &DenseDotConfig::new(vec![1], vec![1], Vec::new(), Vec::new())
            );
            let (mut output, lhs, rhs) = match (output, lhs, rhs) {
                (DenseWrite::F64(output), DenseRead::F64(lhs), DenseRead::F64(rhs)) => {
                    (output, lhs, rhs)
                }
                _ => panic!("layout-checking executor only covers f64"),
            };

            assert_eq!(lhs.shape(), &[2, 3]);
            assert_eq!(lhs.strides(), &[1, 2]);
            assert_eq!(lhs.offset(), 0);
            assert_eq!(rhs.shape(), &[4, 3]);
            assert_eq!(rhs.strides(), &[1, 4]);
            assert_eq!(rhs.offset(), 0);
            assert_eq!(output.shape(), &[2, 4]);
            assert_eq!(output.strides(), &[1, 2]);
            assert_eq!(output.offset(), 0);

            output
                .data_mut()
                .copy_from_slice(&[115.0, 148.0, 124.0, 160.0, 133.0, 172.0, 142.0, 184.0]);
            Ok(())
        }
    }

    #[test]
    fn tensorcontract_structure_honors_output_permutation_with_workspace_scatter() {
        let lhs_space = TensorMapSpace::<2, 0>::from_dims([2, 3], []).unwrap();
        let rhs_space = TensorMapSpace::<2, 0>::from_dims([4, 3], []).unwrap();
        let dst_space = TensorMapSpace::<2, 0>::from_dims([4, 2], []).unwrap();
        let lhs = TensorMap::<f64, 2, 0>::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], lhs_space)
            .unwrap();
        let rhs = TensorMap::<f64, 2, 0>::from_vec(
            (7..=18).map(|value| value as f64).collect(),
            rhs_space,
        )
        .unwrap();
        let mut dst = TensorMap::<f64, 2, 0>::filled(0.0, dst_space).unwrap();
        let structure = TensorContractStructure::compile(
            &dst,
            &lhs,
            &rhs,
            TensorContractAxisSpec::new(&[1], &[1], AxisPermutation::from_axes(&[1, 0])),
        )
        .unwrap();
        let mut backend =
            DenseTreeTransformOperations::new(ContractLayoutCheckingDenseExecutor::default());
        let mut workspace = TensorContractWorkspace::default();

        tensorcontract_execute_with(
            &mut backend,
            &mut workspace,
            &structure,
            &mut dst,
            &lhs,
            &rhs,
            1.0,
            0.0,
        )
        .unwrap();

        assert_eq!(backend.dense().dot_general_into_calls, 1);
        assert_eq!(
            dst.data(),
            &[115.0, 124.0, 133.0, 142.0, 148.0, 160.0, 172.0, 184.0]
        );
        assert_eq!(workspace.output_len(), 8);
    }

    fn assert_tensorcontract_scalar_dtype<D>(lhs_value: D, rhs_value: D, fill: D, expected: D)
    where
        D: DenseBlockScalar + RecouplingCoefficientAction<f64> + Clone + Debug,
    {
        let lhs_space = TensorMapSpace::<2, 0>::from_dims([1, 1], []).unwrap();
        let rhs_space = TensorMapSpace::<2, 0>::from_dims([1, 1], []).unwrap();
        let dst_space = TensorMapSpace::<2, 0>::from_dims([1, 1], []).unwrap();
        let lhs = TensorMap::<D, 2, 0>::from_vec(vec![lhs_value], lhs_space).unwrap();
        let rhs = TensorMap::<D, 2, 0>::from_vec(vec![rhs_value], rhs_space).unwrap();
        let mut dst = TensorMap::<D, 2, 0>::filled(fill, dst_space).unwrap();

        tensorcontract_into(
            &mut dst,
            &lhs,
            &rhs,
            TensorContractAxisSpec::canonical(&[1], &[0]),
            D::one() + D::one(),
            D::one() + D::one() + D::one(),
        )
        .unwrap();

        assert_eq!(dst.data(), &[expected]);
    }

    fn assert_weighted_tensorcontract_scalar_dtype<D>(
        lhs_value: D,
        rhs_value: D,
        fill: D,
        expected: D,
    ) where
        D: DenseBlockScalar + RecouplingCoefficientAction<f64> + Clone + Debug,
    {
        let lhs_space = TensorMapSpace::<2, 0>::from_dims([1, 1], []).unwrap();
        let rhs_space = TensorMapSpace::<2, 0>::from_dims([1, 1], []).unwrap();
        let dst_space = TensorMapSpace::<2, 0>::from_dims([1, 1], []).unwrap();
        let lhs = TensorMap::<D, 2, 0>::from_vec(vec![lhs_value], lhs_space).unwrap();
        let rhs = TensorMap::<D, 2, 0>::from_vec(vec![rhs_value], rhs_space).unwrap();
        let mut dst = TensorMap::<D, 2, 0>::from_vec(vec![fill], dst_space).unwrap();
        let structure = TensorContractStructure::compile_with_block_specs(
            &dst,
            &lhs,
            &rhs,
            TensorContractAxisSpec::canonical(&[1], &[0]),
            &[TensorContractBlockSpec::with_coefficient(0, 0, 0, 0.5)],
        )
        .unwrap();

        tensorcontract_execute_with(
            &mut DenseTreeTransformOperations::default_executor(),
            &mut TensorContractWorkspace::default(),
            &structure,
            &mut dst,
            &lhs,
            &rhs,
            D::one() + D::one(),
            D::one() + D::one() + D::one(),
        )
        .unwrap();

        assert_eq!(dst.data(), &[expected]);
    }

    #[test]
    fn tensorcontract_dense_backend_covers_all_gemm_dtypes() {
        assert_tensorcontract_scalar_dtype(2.0_f32, 3.0_f32, 5.0_f32, 27.0_f32);
        assert_tensorcontract_scalar_dtype(2.0_f64, 3.0_f64, 5.0_f64, 27.0_f64);
        assert_tensorcontract_scalar_dtype(
            Complex32::new(2.0, 0.0),
            Complex32::new(3.0, 0.0),
            Complex32::new(5.0, 0.0),
            Complex32::new(27.0, 0.0),
        );
        assert_tensorcontract_scalar_dtype(
            Complex64::new(2.0, 0.0),
            Complex64::new(3.0, 0.0),
            Complex64::new(5.0, 0.0),
            Complex64::new(27.0, 0.0),
        );
    }

    #[test]
    fn tensorcontract_weighted_terms_support_all_gemm_dtypes() {
        assert_weighted_tensorcontract_scalar_dtype(2.0_f32, 3.0_f32, 5.0_f32, 21.0_f32);
        assert_weighted_tensorcontract_scalar_dtype(2.0_f64, 3.0_f64, 5.0_f64, 21.0_f64);
        assert_weighted_tensorcontract_scalar_dtype(
            Complex32::new(2.0, 1.0),
            Complex32::new(3.0, -1.0),
            Complex32::new(5.0, 2.0),
            Complex32::new(22.0, 7.0),
        );
        assert_weighted_tensorcontract_scalar_dtype(
            Complex64::new(2.0, 1.0),
            Complex64::new(3.0, -1.0),
            Complex64::new(5.0, 2.0),
            Complex64::new(22.0, 7.0),
        );
    }

    #[test]
    fn tensorcontract_structure_rejects_invalid_axes_at_compile_time() {
        let lhs_space = TensorMapSpace::<2, 0>::from_dims([2, 3], []).unwrap();
        let rhs_space = TensorMapSpace::<2, 0>::from_dims([3, 2], []).unwrap();
        let dst_space = TensorMapSpace::<2, 0>::from_dims([2, 2], []).unwrap();
        let lhs = TensorMap::<f64, 2, 0>::filled(1.0, lhs_space).unwrap();
        let rhs = TensorMap::<f64, 2, 0>::filled(1.0, rhs_space).unwrap();
        let dst = TensorMap::<f64, 2, 0>::filled(0.0, dst_space).unwrap();

        let duplicate = TensorContractStructure::compile(
            &dst,
            &lhs,
            &rhs,
            TensorContractAxisSpec::canonical(&[1, 1], &[0, 1]),
        )
        .unwrap_err();
        assert_eq!(
            duplicate,
            OperationError::InvalidAxisSet {
                tensor: "lhs",
                axes: vec![1, 1],
                rank: 2,
            }
        );

        let count = TensorContractStructure::compile(
            &dst,
            &lhs,
            &rhs,
            TensorContractAxisSpec::canonical(&[1], &[0, 1]),
        )
        .unwrap_err();
        assert_eq!(
            count,
            OperationError::ContractAxisCountMismatch { lhs: 1, rhs: 2 }
        );
    }

    #[test]
    fn tensorcontract_structure_rejects_dimension_and_output_mismatch_at_compile_time() {
        let lhs_space = TensorMapSpace::<2, 0>::from_dims([2, 3], []).unwrap();
        let rhs_space = TensorMapSpace::<2, 0>::from_dims([4, 2], []).unwrap();
        let dst_space = TensorMapSpace::<2, 0>::from_dims([2, 4], []).unwrap();
        let lhs = TensorMap::<f64, 2, 0>::filled(1.0, lhs_space).unwrap();
        let rhs = TensorMap::<f64, 2, 0>::filled(1.0, rhs_space).unwrap();
        let dst = TensorMap::<f64, 2, 0>::filled(0.0, dst_space).unwrap();

        let contracted_dim = TensorContractStructure::compile(
            &dst,
            &lhs,
            &rhs,
            TensorContractAxisSpec::canonical(&[1], &[1]),
        )
        .unwrap_err();
        assert_eq!(
            contracted_dim,
            OperationError::ShapeMismatch {
                dst: vec![3],
                src: vec![2],
            }
        );

        let wrong_dst_space = TensorMapSpace::<2, 0>::from_dims([4, 2], []).unwrap();
        let wrong_dst = TensorMap::<f64, 2, 0>::filled(0.0, wrong_dst_space).unwrap();
        let valid_rhs_space = TensorMapSpace::<2, 0>::from_dims([3, 2], []).unwrap();
        let valid_rhs = TensorMap::<f64, 2, 0>::filled(1.0, valid_rhs_space).unwrap();
        let output = TensorContractStructure::compile(
            &wrong_dst,
            &lhs,
            &valid_rhs,
            TensorContractAxisSpec::canonical(&[1], &[0]),
        )
        .unwrap_err();
        assert_eq!(
            output,
            OperationError::ShapeMismatch {
                dst: vec![4, 2],
                src: vec![2, 2],
            }
        );
    }

    #[derive(Default)]
    struct PanicDenseExecutor;

    impl DenseExecutor for PanicDenseExecutor {
        fn svd(
            &mut self,
            _input: DenseRead<'_>,
        ) -> Result<Vec<tenet_dense::DenseTensor>, DenseError> {
            unreachable!("tensor contraction does not call svd")
        }

        fn qr(
            &mut self,
            _input: DenseRead<'_>,
        ) -> Result<Vec<tenet_dense::DenseTensor>, DenseError> {
            unreachable!("tensor contraction does not call qr")
        }

        fn eigh(
            &mut self,
            _input: DenseRead<'_>,
        ) -> Result<Vec<tenet_dense::DenseTensor>, DenseError> {
            unreachable!("tensor contraction does not call eigh")
        }

        fn dot_general_into(
            &mut self,
            _output: DenseWrite<'_>,
            _lhs: DenseRead<'_>,
            _rhs: DenseRead<'_>,
            _config: &DenseDotConfig,
        ) -> Result<(), DenseError> {
            panic!("replay structure mismatch must be rejected before dense execution")
        }
    }

    #[test]
    fn tensorcontract_structure_rejects_incompatible_replay_structure_before_dense_execution() {
        let compile_lhs_space = TensorMapSpace::<2, 0>::from_dims([2, 3], []).unwrap();
        let compile_rhs_space = TensorMapSpace::<2, 0>::from_dims([3, 2], []).unwrap();
        let compile_dst_space = TensorMapSpace::<2, 0>::from_dims([2, 2], []).unwrap();
        let compile_lhs = TensorMap::<f64, 2, 0>::filled(1.0, compile_lhs_space).unwrap();
        let compile_rhs = TensorMap::<f64, 2, 0>::filled(1.0, compile_rhs_space).unwrap();
        let compile_dst = TensorMap::<f64, 2, 0>::filled(0.0, compile_dst_space).unwrap();
        let structure = TensorContractStructure::compile(
            &compile_dst,
            &compile_lhs,
            &compile_rhs,
            TensorContractAxisSpec::canonical(&[1], &[0]),
        )
        .unwrap();

        let lhs_space = TensorMapSpace::<2, 0>::from_dims([4, 3], []).unwrap();
        let rhs_space = TensorMapSpace::<2, 0>::from_dims([3, 2], []).unwrap();
        let dst_space = TensorMapSpace::<2, 0>::from_dims([4, 2], []).unwrap();
        let lhs = TensorMap::<f64, 2, 0>::filled(1.0, lhs_space).unwrap();
        let rhs = TensorMap::<f64, 2, 0>::filled(1.0, rhs_space).unwrap();
        let mut dst = TensorMap::<f64, 2, 0>::filled(0.0, dst_space).unwrap();
        let mut backend = DenseTreeTransformOperations::new(PanicDenseExecutor);
        let mut workspace = TensorContractWorkspace::default();

        let err = tensorcontract_execute_with(
            &mut backend,
            &mut workspace,
            &structure,
            &mut dst,
            &lhs,
            &rhs,
            1.0,
            0.0,
        )
        .unwrap_err();

        assert_eq!(err, OperationError::StructureMismatch { tensor: "dst" });
    }

    #[test]
    fn tensorcontract_structure_rejects_multiblock_until_block_sparse_enumeration_exists() {
        let dense = BlockStructure::trivial(&[2, 2]).unwrap();
        let multiblock = BlockStructure::packed_column_major(2, [vec![1, 2], vec![1, 2]]).unwrap();

        let err = TensorContractStructure::compile_structures(
            &dense,
            &multiblock,
            &dense,
            TensorContractAxisSpec::canonical(&[1], &[0]),
        )
        .unwrap_err();

        assert_eq!(
            err,
            OperationError::UnsupportedTensorContractScope {
                message: "block-sparse contraction enumeration is not implemented yet",
            }
        );
    }

    #[test]
    fn tensorcontract_structure_replays_explicit_block_terms_and_applies_beta_once() {
        let lhs_space = TensorMapSpace::<2, 0>::from_dims([2, 2], []).unwrap();
        let rhs_space = TensorMapSpace::<2, 0>::from_dims([2, 2], []).unwrap();
        let dst_space = TensorMapSpace::<2, 0>::from_dims([1, 1], []).unwrap();
        let lhs_structure = BlockStructure::packed_column_major_with_keys(
            2,
            [
                (BlockKey::sector_ids([10]), vec![1, 2]),
                (BlockKey::sector_ids([20]), vec![1, 2]),
            ],
        )
        .unwrap();
        let rhs_structure = BlockStructure::packed_column_major_with_keys(
            2,
            [
                (BlockKey::sector_ids([30]), vec![2, 1]),
                (BlockKey::sector_ids([40]), vec![2, 1]),
            ],
        )
        .unwrap();
        let dst_structure = BlockStructure::packed_column_major_with_keys(
            2,
            [(BlockKey::sector_ids([99]), vec![1, 1])],
        )
        .unwrap();
        let lhs = TensorMap::<f64, 2, 0>::from_vec_with_structure(
            vec![1.0, 2.0, 3.0, 4.0],
            lhs_space,
            lhs_structure,
        )
        .unwrap();
        let rhs = TensorMap::<f64, 2, 0>::from_vec_with_structure(
            vec![5.0, 6.0, 7.0, 8.0],
            rhs_space,
            rhs_structure,
        )
        .unwrap();
        let mut dst =
            TensorMap::<f64, 2, 0>::from_vec_with_structure(vec![10.0], dst_space, dst_structure)
                .unwrap();
        let structure = TensorContractStructure::compile_with_block_specs(
            &dst,
            &lhs,
            &rhs,
            TensorContractAxisSpec::canonical(&[1], &[0]),
            &[
                TensorContractBlockSpec::with_coefficient(0, 0, 0, 0.5),
                TensorContractBlockSpec::with_coefficient(0, 1, 1, 2.0),
            ],
        )
        .unwrap();

        tensorcontract_execute_with(
            &mut DenseTreeTransformOperations::default_executor(),
            &mut TensorContractWorkspace::default(),
            &structure,
            &mut dst,
            &lhs,
            &rhs,
            2.0,
            3.0,
        )
        .unwrap();

        assert_eq!(structure.terms().len(), 2);
        assert_eq!(dst.data(), &[259.0]);
    }

    #[test]
    fn tensorcontract_structure_rejects_invalid_explicit_block_term_at_compile_time() {
        let dense = BlockStructure::trivial(&[1, 1]).unwrap();

        let err = TensorContractStructure::compile_structures_with_block_specs(
            &dense,
            &dense,
            &dense,
            TensorContractAxisSpec::canonical(&[1], &[0]),
            &[TensorContractBlockSpec::new(0, 1, 0)],
        )
        .unwrap_err();

        assert_eq!(
            err,
            OperationError::BlockIndexOutOfBounds {
                tensor: "lhs",
                index: 1,
                count: 1,
            }
        );
    }

    #[test]
    fn tensorcontract_fusion_structure_enumerates_z2_compose_blocks_and_replays() {
        let rule = Z2FusionRule;
        let leg = || SectorLeg::new([SectorId::new(0), SectorId::new(1)], false);
        let lhs_space = FusionTensorMapSpace::from_degeneracy_shapes(
            TensorMapSpace::<1, 1>::from_dims([1], [1]).unwrap(),
            FusionTreeHomSpace::new(
                FusionProductSpace::new([leg()]),
                FusionProductSpace::new([leg()]),
            ),
            &rule,
            [vec![1, 1], vec![1, 1]],
        )
        .unwrap();
        let rhs_space = FusionTensorMapSpace::from_degeneracy_shapes(
            TensorMapSpace::<1, 1>::from_dims([1], [1]).unwrap(),
            FusionTreeHomSpace::new(
                FusionProductSpace::new([leg()]),
                FusionProductSpace::new([leg()]),
            ),
            &rule,
            [vec![1, 1], vec![1, 1]],
        )
        .unwrap();
        let dst_space = FusionTensorMapSpace::from_degeneracy_shapes(
            TensorMapSpace::<1, 1>::from_dims([1], [1]).unwrap(),
            FusionTreeHomSpace::new(
                FusionProductSpace::new([leg()]),
                FusionProductSpace::new([leg()]),
            ),
            &rule,
            [vec![1, 1], vec![1, 1]],
        )
        .unwrap();
        let lhs =
            TensorMap::<f64, 1, 1>::from_vec_with_fusion_space(vec![2.0, 3.0], lhs_space).unwrap();
        let rhs =
            TensorMap::<f64, 1, 1>::from_vec_with_fusion_space(vec![5.0, 7.0], rhs_space).unwrap();
        let mut dst =
            TensorMap::<f64, 1, 1>::from_vec_with_fusion_space(vec![10.0, 20.0], dst_space)
                .unwrap();

        let specs = tensorcontract_fusion_block_specs(
            &rule,
            dst.fusion_space().unwrap(),
            lhs.fusion_space().unwrap(),
            rhs.fusion_space().unwrap(),
            TensorContractAxisSpec::canonical(&[1], &[0]),
        )
        .unwrap();
        assert_eq!(
            specs,
            vec![
                TensorContractBlockSpec::new(0, 0, 0),
                TensorContractBlockSpec::new(1, 1, 1),
            ]
        );

        tensorcontract_fusion_into(
            &rule,
            &mut dst,
            &lhs,
            &rhs,
            TensorContractAxisSpec::canonical(&[1], &[0]),
            2.0,
            3.0,
        )
        .unwrap();

        assert_eq!(dst.data(), &[50.0, 102.0]);
    }

    #[test]
    fn tensorcontract_fusion_block_specs_enumerates_su2_innerline_blocks_from_homspace() {
        let rule = SU2FusionRule;
        let half = SectorId::new(1);
        let lhs_space = FusionTensorMapSpace::from_degeneracy_shapes(
            TensorMapSpace::<3, 1>::from_dims([1, 1, 1], [1]).unwrap(),
            FusionTreeHomSpace::from_sector_ids([1, 1, 1], [1]),
            &rule,
            [vec![1, 1, 1, 1], vec![1, 1, 1, 1]],
        )
        .unwrap();
        let rhs_space = FusionTensorMapSpace::from_degeneracy_shapes(
            TensorMapSpace::<1, 1>::from_dims([1], [1]).unwrap(),
            FusionTreeHomSpace::from_sector_ids([1], [1]),
            &rule,
            [vec![1, 1]],
        )
        .unwrap();
        let dst_space = FusionTensorMapSpace::from_degeneracy_shapes(
            TensorMapSpace::<3, 1>::from_dims([1, 1, 1], [1]).unwrap(),
            FusionTreeHomSpace::from_sector_ids([1, 1, 1], [1]),
            &rule,
            [vec![1, 1, 1, 1], vec![1, 1, 1, 1]],
        )
        .unwrap();

        let specs = tensorcontract_fusion_block_specs(
            &rule,
            &dst_space,
            &lhs_space,
            &rhs_space,
            TensorContractAxisSpec::canonical(&[3], &[0]),
        )
        .unwrap();

        assert_eq!(
            specs,
            vec![
                TensorContractBlockSpec::new(0, 0, 0),
                TensorContractBlockSpec::new(1, 1, 0),
            ]
        );
        assert_eq!(
            dst_space
                .homspace()
                .fusion_tree_keys_from_external_sectors(&rule, &[half, half, half, half])
                .unwrap()
                .len(),
            2
        );
    }

    #[test]
    fn tensorcontract_fusion_block_specs_rejects_missing_destination_subblock() {
        let rule = Z2FusionRule;
        let leg = || SectorLeg::new([SectorId::new(0), SectorId::new(1)], false);
        let lhs_space = FusionTensorMapSpace::from_degeneracy_shapes(
            TensorMapSpace::<1, 1>::from_dims([1], [1]).unwrap(),
            FusionTreeHomSpace::new(
                FusionProductSpace::new([leg()]),
                FusionProductSpace::new([leg()]),
            ),
            &rule,
            [vec![1, 1], vec![1, 1]],
        )
        .unwrap();
        let rhs_space = FusionTensorMapSpace::from_degeneracy_shapes(
            TensorMapSpace::<1, 1>::from_dims([1], [1]).unwrap(),
            FusionTreeHomSpace::new(
                FusionProductSpace::new([leg()]),
                FusionProductSpace::new([leg()]),
            ),
            &rule,
            [vec![1, 1], vec![1, 1]],
        )
        .unwrap();
        let dst_hom = FusionTreeHomSpace::new(
            FusionProductSpace::new([leg()]),
            FusionProductSpace::new([leg()]),
        );
        let keys = dst_hom.fusion_tree_keys(&rule);
        let dst_structure =
            BlockStructure::packed_column_major_with_keys(2, [(keys[0].clone(), vec![1, 1])])
                .unwrap();
        let dst_space = FusionTensorMapSpace::new(
            TensorMapSpace::<1, 1>::from_dims([1], [1]).unwrap(),
            dst_hom,
            dst_structure,
        )
        .unwrap();

        let err = tensorcontract_fusion_block_specs(
            &rule,
            &dst_space,
            &lhs_space,
            &rhs_space,
            TensorContractAxisSpec::canonical(&[1], &[0]),
        )
        .unwrap_err();

        assert_eq!(
            err,
            OperationError::MissingBlockKey {
                key: keys[1].clone().into()
            }
        );
    }

    #[test]
    fn tensorcontract_fusion_block_specs_enumerates_noncanonical_tree_transform_terms() {
        let rule = Z2FusionRule;
        let leg = |is_dual| SectorLeg::new([SectorId::new(0)], is_dual);
        let fusion_space = FusionTensorMapSpace::from_degeneracy_shapes(
            TensorMapSpace::<1, 1>::from_dims([1], [1]).unwrap(),
            FusionTreeHomSpace::new(
                FusionProductSpace::new([leg(false)]),
                FusionProductSpace::new([leg(false)]),
            ),
            &rule,
            [vec![1, 1]],
        )
        .unwrap();
        let transformed_dst_space = FusionTensorMapSpace::from_degeneracy_shapes(
            TensorMapSpace::<1, 1>::from_dims([1], [1]).unwrap(),
            FusionTreeHomSpace::new(
                FusionProductSpace::new([leg(true)]),
                FusionProductSpace::new([leg(true)]),
            ),
            &rule,
            [vec![1, 1]],
        )
        .unwrap();

        let specs = tensorcontract_fusion_block_specs(
            &rule,
            &transformed_dst_space,
            &fusion_space,
            &fusion_space,
            TensorContractAxisSpec::canonical(&[0], &[1]),
        )
        .unwrap();

        assert_eq!(
            specs,
            vec![TensorContractBlockSpec::with_coefficient(0, 0, 0, 1.0)]
        );

        let specs = tensorcontract_fusion_block_specs(
            &rule,
            &transformed_dst_space,
            &fusion_space,
            &fusion_space,
            TensorContractAxisSpec::new(&[1], &[0], AxisPermutation::from_axes(&[1, 0])),
        )
        .unwrap();

        assert_eq!(
            specs,
            vec![TensorContractBlockSpec::with_coefficient(0, 0, 0, 1.0)]
        );
    }

    #[test]
    fn tensorcontract_fusion_noncanonical_replays_transformed_product_block() {
        let rule = Z2FusionRule;
        let leg = |is_dual| SectorLeg::new([SectorId::new(0)], is_dual);
        let src_space = FusionTensorMapSpace::from_degeneracy_shapes(
            TensorMapSpace::<1, 1>::from_dims([1], [1]).unwrap(),
            FusionTreeHomSpace::new(
                FusionProductSpace::new([leg(false)]),
                FusionProductSpace::new([leg(false)]),
            ),
            &rule,
            [vec![1, 1]],
        )
        .unwrap();
        let dst_space = FusionTensorMapSpace::from_degeneracy_shapes(
            TensorMapSpace::<1, 1>::from_dims([1], [1]).unwrap(),
            FusionTreeHomSpace::new(
                FusionProductSpace::new([leg(true)]),
                FusionProductSpace::new([leg(true)]),
            ),
            &rule,
            [vec![1, 1]],
        )
        .unwrap();
        let lhs = TensorMap::<f64, 1, 1>::from_vec_with_fusion_space(vec![2.0], src_space.clone())
            .unwrap();
        let rhs = TensorMap::<f64, 1, 1>::from_vec_with_fusion_space(vec![5.0], src_space).unwrap();
        let mut dst =
            TensorMap::<f64, 1, 1>::from_vec_with_fusion_space(vec![7.0], dst_space).unwrap();

        tensorcontract_fusion_into(
            &rule,
            &mut dst,
            &lhs,
            &rhs,
            TensorContractAxisSpec::canonical(&[0], &[1]),
            3.0,
            11.0,
        )
        .unwrap();

        assert_eq!(dst.data(), &[107.0]);
    }

    #[test]
    fn tensorcontract_fusion_output_recoupling_uses_su2_coefficients() {
        let rule = SU2FusionRule;
        let src_key = all_codomain_fusion_tree_test_key(
            [1, 1, 1, 1],
            Some(0),
            [false, false, false, false],
            [0, 1],
            [1, 1, 1],
        );
        let dst_key0 = src_key.clone();
        let dst_key1 = all_codomain_fusion_tree_test_key(
            [1, 1, 1, 1],
            Some(0),
            [false, false, false, false],
            [2, 1],
            [1, 1, 1],
        );
        let scalar_key = BlockKey::from(FusionTreeBlockKey::pair(
            empty_fusion_tree(),
            empty_fusion_tree(),
        ));
        let lhs_space = FusionTensorMapSpace::new(
            TensorMapSpace::<4, 0>::from_dims([1, 1, 1, 1], []).unwrap(),
            FusionTreeHomSpace::from_sector_ids([1, 1, 1, 1], []),
            BlockStructure::packed_column_major_with_keys(4, [(src_key, vec![1, 1, 1, 1])])
                .unwrap(),
        )
        .unwrap();
        let rhs_space = FusionTensorMapSpace::new(
            TensorMapSpace::<0, 0>::from_dims([], []).unwrap(),
            FusionTreeHomSpace::from_sector_ids([], []),
            BlockStructure::packed_column_major_with_keys(0, [(scalar_key, vec![])]).unwrap(),
        )
        .unwrap();
        let dst_space = FusionTensorMapSpace::new(
            TensorMapSpace::<4, 0>::from_dims([1, 1, 1, 1], []).unwrap(),
            FusionTreeHomSpace::from_sector_ids([1, 1, 1, 1], []),
            BlockStructure::packed_column_major_with_keys(
                4,
                [(dst_key0, vec![1, 1, 1, 1]), (dst_key1, vec![1, 1, 1, 1])],
            )
            .unwrap(),
        )
        .unwrap();
        let lhs =
            TensorMap::<f64, 4, 0>::from_vec_with_fusion_space(vec![10.0], lhs_space).unwrap();
        let rhs = TensorMap::<f64, 0, 0>::from_vec_with_fusion_space(vec![5.0], rhs_space).unwrap();
        let mut dst =
            TensorMap::<f64, 4, 0>::from_vec_with_fusion_space(vec![1.0, 2.0], dst_space).unwrap();

        let specs = tensorcontract_fusion_block_specs(
            &rule,
            dst.fusion_space().unwrap(),
            lhs.fusion_space().unwrap(),
            rhs.fusion_space().unwrap(),
            TensorContractAxisSpec::new(&[], &[], AxisPermutation::from_axes(&[0, 2, 1, 3])),
        )
        .unwrap();
        assert_eq!(specs.len(), 2);
        assert_eq!(specs[0].dst_block(), 0);
        assert_eq!(specs[1].dst_block(), 1);
        assert!((specs[0].coefficient() - 0.5).abs() < 1.0e-12);
        assert!((specs[1].coefficient() - 0.866_025_403_784_438_6).abs() < 1.0e-12);

        tensorcontract_fusion_into(
            &rule,
            &mut dst,
            &lhs,
            &rhs,
            TensorContractAxisSpec::new(&[], &[], AxisPermutation::from_axes(&[0, 2, 1, 3])),
            2.0,
            3.0,
        )
        .unwrap();

        assert!((dst.data()[0] - 53.0).abs() < 1.0e-12);
        assert!((dst.data()[1] - 92.602_540_378_443_86).abs() < 1.0e-12);
    }

    #[test]
    fn tensorcontract_fusion_su2_keeps_contracted_tree_basis_with_degeneracy() {
        let rule = SU2FusionRule;
        let lhs_hom = FusionTreeHomSpace::from_sector_ids([1], [1, 1, 1]);
        let rhs_hom = FusionTreeHomSpace::from_sector_ids([1, 1, 1], [1]);
        let lhs_keys = lhs_hom.fusion_tree_keys(&rule);
        let rhs_keys = rhs_hom.fusion_tree_keys(&rule);
        assert_eq!(lhs_keys.len(), 2);
        assert_eq!(rhs_keys.len(), 2);
        assert_ne!(
            lhs_keys[0].domain_tree().innerlines()[0],
            lhs_keys[1].domain_tree().innerlines()[0]
        );
        let lhs_space = FusionTensorMapSpace::from_degeneracy_shapes(
            TensorMapSpace::<1, 3>::from_dims([2], [2, 2, 2]).unwrap(),
            lhs_hom,
            &rule,
            [vec![2, 2, 2, 2], vec![2, 2, 2, 2]],
        )
        .unwrap();
        let rhs_space = FusionTensorMapSpace::from_degeneracy_shapes(
            TensorMapSpace::<3, 1>::from_dims([2, 2, 2], [2]).unwrap(),
            rhs_hom,
            &rule,
            [vec![2, 2, 2, 2], vec![2, 2, 2, 2]],
        )
        .unwrap();
        let dst_hom = FusionTreeHomSpace::from_sector_ids([1], [1]);
        let dst_keys = dst_hom.fusion_tree_keys(&rule);
        assert_eq!(dst_keys.len(), 1);
        let dst_space = FusionTensorMapSpace::new(
            TensorMapSpace::<1, 1>::from_dims([2], [2]).unwrap(),
            dst_hom,
            BlockStructure::packed_column_major_with_keys(
                2,
                [(dst_keys[0].clone(), vec![2, 2])],
            )
            .unwrap(),
        )
        .unwrap();
        let lhs_data = (0..32)
            .map(|index| 0.25 + index as f64)
            .collect::<Vec<_>>();
        let rhs_data = (0..32)
            .map(|index| 10.0 - 0.5 * index as f64)
            .collect::<Vec<_>>();
        let initial_dst = vec![1.0, -2.0, 3.0, -4.0];
        let lhs = TensorMap::<f64, 1, 3>::from_vec_with_fusion_space(lhs_data, lhs_space).unwrap();
        let rhs = TensorMap::<f64, 3, 1>::from_vec_with_fusion_space(rhs_data, rhs_space).unwrap();
        let mut dst =
            TensorMap::<f64, 1, 1>::from_vec_with_fusion_space(initial_dst.clone(), dst_space)
                .unwrap();
        let axes = TensorContractAxisSpec::canonical(&[1, 2, 3], &[0, 1, 2]);
        let specs = tensorcontract_fusion_block_specs(
            &rule,
            dst.fusion_space().unwrap(),
            lhs.fusion_space().unwrap(),
            rhs.fusion_space().unwrap(),
            axes,
        )
        .unwrap();

        assert_eq!(specs.len(), 2);
        for spec in &specs {
            let lhs_key = match lhs.structure().block(spec.lhs_block()).unwrap().key() {
                BlockKey::FusionTree(key) => key,
                BlockKey::Dense => panic!("expected lhs fusion-tree block"),
            };
            let rhs_key = match rhs.structure().block(spec.rhs_block()).unwrap().key() {
                BlockKey::FusionTree(key) => key,
                BlockKey::Dense => panic!("expected rhs fusion-tree block"),
            };
            assert_eq!(
                lhs_key.domain_tree().innerlines()[0],
                rhs_key.codomain_tree().innerlines()[0],
                "contracted SU2 tree basis must not cross-contract"
            );
        }

        let alpha = 1.25;
        let beta = -0.5;
        let mut expected = initial_dst
            .into_iter()
            .map(|value| beta * value)
            .collect::<Vec<_>>();
        for spec in &specs {
            let lhs_offset = lhs.structure().block(spec.lhs_block()).unwrap().offset();
            let rhs_offset = rhs.structure().block(spec.rhs_block()).unwrap().offset();
            for lhs_open in 0..2 {
                for rhs_open in 0..2 {
                    let mut sum = 0.0;
                    for a in 0..2 {
                        for b in 0..2 {
                            for c in 0..2 {
                                let lhs_index = lhs_offset + lhs_open + 2 * a + 4 * b + 8 * c;
                                let rhs_index = rhs_offset + a + 2 * b + 4 * c + 8 * rhs_open;
                                sum += lhs.data()[lhs_index] * rhs.data()[rhs_index];
                            }
                        }
                    }
                    expected[lhs_open + 2 * rhs_open] += alpha * spec.coefficient() * sum;
                }
            }
        }

        tensorcontract_fusion_into(&rule, &mut dst, &lhs, &rhs, axes, alpha, beta).unwrap();

        for (&actual, expected) in dst.data().iter().zip(expected) {
            assert!(
                (actual - expected).abs() < 1.0e-10,
                "actual {actual} expected {expected}"
            );
        }
    }

    #[test]
    fn tensorcontract_fusion_product_noncanonical_supports_complex_data() {
        let (rule, src_space, dst_space, _) = fz2_u1_su2_tree_pair_fixture();
        let scalar_key = BlockKey::from(FusionTreeBlockKey::pair(
            empty_fusion_tree(),
            empty_fusion_tree(),
        ));
        let rhs_space = FusionTensorMapSpace::new(
            TensorMapSpace::<0, 0>::from_dims([], []).unwrap(),
            FusionTreeHomSpace::from_sector_ids([], []),
            BlockStructure::packed_column_major_with_keys(0, [(scalar_key, vec![])]).unwrap(),
        )
        .unwrap();
        let lhs = TensorMap::<Complex64, 2, 1>::from_vec_with_fusion_space(
            vec![Complex64::new(1.0, 2.0), Complex64::new(3.0, -1.0)],
            src_space,
        )
        .unwrap();
        let rhs = TensorMap::<Complex64, 0, 0>::from_vec_with_fusion_space(
            vec![Complex64::new(2.0, 0.5)],
            rhs_space,
        )
        .unwrap();
        let mut dst = TensorMap::<Complex64, 2, 1>::from_vec_with_fusion_space(
            vec![Complex64::new(5.0, 1.0), Complex64::new(-2.0, 4.0)],
            dst_space,
        )
        .unwrap();
        let axes = TensorContractAxisSpec::new(&[], &[], AxisPermutation::from_axes(&[1, 0, 2]));
        let specs = tensorcontract_fusion_block_specs(
            &rule,
            dst.fusion_space().unwrap(),
            lhs.fusion_space().unwrap(),
            rhs.fusion_space().unwrap(),
            axes,
        )
        .unwrap();
        assert!(!specs.is_empty());

        let mut expected = dst
            .data()
            .iter()
            .map(|&value| Complex64::new(3.0, 0.0) * value)
            .collect::<Vec<_>>();
        let lhs_structure = lhs.structure();
        let rhs_structure = rhs.structure();
        let dst_structure = dst.structure();
        for spec in &specs {
            let lhs_offset = lhs_structure.block(spec.lhs_block()).unwrap().offset();
            let rhs_offset = rhs_structure.block(spec.rhs_block()).unwrap().offset();
            let dst_offset = dst_structure.block(spec.dst_block()).unwrap().offset();
            expected[dst_offset] += Complex64::new(2.0 * spec.coefficient(), 0.0)
                * lhs.data()[lhs_offset]
                * rhs.data()[rhs_offset];
        }

        tensorcontract_fusion_into(
            &rule,
            &mut dst,
            &lhs,
            &rhs,
            axes,
            Complex64::new(2.0, 0.0),
            Complex64::new(3.0, 0.0),
        )
        .unwrap();

        for (&actual, expected) in dst.data().iter().zip(expected) {
            let delta = actual - expected;
            assert!(delta.re.abs() < 1.0e-12 && delta.im.abs() < 1.0e-12);
        }
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
    fn unique_tree_transform_plan_builder_creates_single_specs_in_source_order() {
        let src_key1 = fusion_tree_test_key([1, 0], [1], 1, [false, false], [false]);
        let src_key2 = fusion_tree_test_key([0, 1], [1], 1, [false, false], [false]);
        let dst_key1 = fusion_tree_test_key([0, 1], [1], 1, [false, false], [false]);
        let dst_key2 = fusion_tree_test_key([1, 0], [1], 1, [false, false], [false]);
        let src_tree1 = expect_tree_key(&src_key1);
        let src_tree2 = expect_tree_key(&src_key2);
        let dst_tree1 = expect_tree_key(&dst_key1);
        let dst_tree2 = expect_tree_key(&dst_key2);
        let src_structure = BlockStructure::packed_column_major_with_keys(
            2,
            [
                (src_key1.clone(), vec![1, 1]),
                (src_key2.clone(), vec![1, 1]),
            ],
        )
        .unwrap();

        let plan = build_unique_tree_transform_group_plan(
            &UniqueZ2Rule,
            TreeTransformOperationKey::transpose([1, 0], [0]),
            &src_structure,
            |src| {
                if src == &src_tree1 {
                    Ok((dst_tree1.clone(), 2.0_f64))
                } else if src == &src_tree2 {
                    Ok((dst_tree2.clone(), 3.0_f64))
                } else {
                    panic!("unexpected source key {src:?}")
                }
            },
        )
        .unwrap();

        assert_eq!(plan.specs().len(), 2);
        assert_eq!(plan.specs()[0].group_key(), &src_tree1.group_key());
        assert_eq!(plan.specs()[0].src_keys(), &[src_key1]);
        assert_eq!(plan.specs()[0].dst_keys(), &[dst_key1]);
        assert_eq!(plan.specs()[0].coefficients_src_by_dst(), &[2.0]);
        assert_eq!(plan.specs()[1].group_key(), &src_tree2.group_key());
        assert_eq!(plan.specs()[1].src_keys(), &[src_key2]);
        assert_eq!(plan.specs()[1].dst_keys(), &[dst_key2]);
        assert_eq!(plan.specs()[1].coefficients_src_by_dst(), &[3.0]);
    }

    #[test]
    fn single_output_unique_tree_transform_helper_rejects_simple_fusion() {
        let src_key = fusion_tree_test_key([1, 1, 1], [1], 1, [false, false, false], [false]);
        let src_structure =
            BlockStructure::packed_column_major_with_keys(4, [(src_key, vec![1, 1, 1, 1])])
                .unwrap();
        let operation = TreeTransformOperationKey::transpose([2, 1, 0], [0]);

        let err = build_unique_tree_transform_group_plan(
            &SimpleSu2Rule,
            operation.clone(),
            &src_structure,
            |_| -> Result<(FusionTreeBlockKey, f64), OperationError> {
                unreachable!("non-Unique fusion must be rejected before transforming keys")
            },
        )
        .unwrap_err();

        assert_eq!(
            err,
            OperationError::UnsupportedFusionStyle {
                operation,
                style: FusionStyleKind::Simple,
            }
        );
    }

    #[test]
    fn tree_transform_plan_builder_accepts_simple_multi_destination_callback() {
        let src_key0 = all_codomain_fusion_tree_test_key(
            [1, 1, 1, 1],
            Some(0),
            [false, false, false, false],
            [0, 1],
            [1, 1, 1],
        );
        let src_key1 = all_codomain_fusion_tree_test_key(
            [1, 1, 1, 1],
            Some(0),
            [false, false, false, false],
            [2, 1],
            [1, 1, 1],
        );
        let src_tree0 = expect_tree_key(&src_key0);
        let src_tree1 = expect_tree_key(&src_key1);
        let src_structure = BlockStructure::packed_column_major_with_keys(
            4,
            [
                (src_key0.clone(), vec![1, 1, 1, 1]),
                (src_key1.clone(), vec![1, 1, 1, 1]),
            ],
        )
        .unwrap();
        let operation = TreeTransformOperationKey::braid([0, 2, 1, 3], [], [0, 1, 2, 3], []);

        let plan =
            build_tree_transform_group_plan(&SimpleSu2Rule, operation, &src_structure, |src| {
                if src == &src_tree0 {
                    Ok(vec![
                        (src_tree0.clone(), 0.5_f64),
                        (src_tree1.clone(), 0.866_025_403_784_438_6),
                    ])
                } else if src == &src_tree1 {
                    Ok(vec![
                        (src_tree0.clone(), 0.866_025_403_784_438_6),
                        (src_tree1.clone(), -0.5),
                    ])
                } else {
                    panic!("unexpected source key {src:?}")
                }
            })
            .unwrap();

        assert_eq!(plan.specs().len(), 1);
        let spec = &plan.specs()[0];
        assert_eq!(spec.src_keys(), &[src_key0.clone(), src_key1.clone()]);
        assert_eq!(spec.dst_keys(), &[src_key0, src_key1]);
        assert_eq!(
            spec.coefficients_src_by_dst(),
            &[0.5, 0.866_025_403_784_438_6, 0.866_025_403_784_438_6, -0.5]
        );
    }

    #[test]
    fn multiplicity_free_su2_plan_builder_creates_generic_recoupling_block() {
        let src_key0 = all_codomain_fusion_tree_test_key(
            [1, 1, 1, 1],
            Some(0),
            [false, false, false, false],
            [0, 1],
            [1, 1, 1],
        );
        let src_key1 = all_codomain_fusion_tree_test_key(
            [1, 1, 1, 1],
            Some(0),
            [false, false, false, false],
            [2, 1],
            [1, 1, 1],
        );
        let src_structure = BlockStructure::packed_column_major_with_keys(
            4,
            [
                (src_key0.clone(), vec![1, 1, 1, 1]),
                (src_key1.clone(), vec![1, 1, 1, 1]),
            ],
        )
        .unwrap();
        let operation = TreeTransformOperationKey::braid([0, 2, 1, 3], [], [0, 1, 2, 3], []);

        let plan =
            build_all_codomain_tree_transform_group_plan(&SU2FusionRule, operation, &src_structure)
                .unwrap();

        assert_eq!(plan.specs().len(), 1);
        let spec = &plan.specs()[0];
        assert_eq!(spec.src_keys(), &[src_key0.clone(), src_key1.clone()]);
        assert_eq!(spec.dst_keys(), &[src_key0, src_key1]);
        let expected = [0.5, 0.866_025_403_784_438_6, 0.866_025_403_784_438_6, -0.5];
        assert_eq!(spec.coefficients_src_by_dst().len(), expected.len());
        for (&actual, expected) in spec.coefficients_src_by_dst().iter().zip(expected) {
            assert!(
                (actual - expected).abs() < 1.0e-12,
                "coefficient {actual} != {expected}"
            );
        }

        let src_space = TensorMapSpace::<4, 0>::from_dims([1, 1, 1, 1], []).unwrap();
        let dst_space = TensorMapSpace::<4, 0>::from_dims([1, 1, 1, 1], []).unwrap();
        let src = TensorMap::<f64, 4, 0>::from_vec_with_structure(
            vec![10.0, 20.0],
            src_space,
            src_structure.clone(),
        )
        .unwrap();
        let mut dst = TensorMap::<f64, 4, 0>::from_vec_with_structure(
            vec![0.0, 0.0],
            dst_space,
            src_structure.clone(),
        )
        .unwrap();
        let structure = plan
            .compile_structures(&src_structure, &src_structure)
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

        assert!(structure.has_pack_gemm_scatter_blocks());
        assert!((dst.data()[0] - 22.320_508_075_688_77).abs() < 1.0e-12);
        assert!((dst.data()[1] + 1.339_745_962_155_612_7).abs() < 1.0e-12);
    }

    #[test]
    fn tree_pair_transform_public_helper_executes_su2_recoupling_block() {
        let src_key0 = all_codomain_fusion_tree_test_key(
            [1, 1, 1, 1],
            Some(0),
            [false, false, false, false],
            [0, 1],
            [1, 1, 1],
        );
        let src_key1 = all_codomain_fusion_tree_test_key(
            [1, 1, 1, 1],
            Some(0),
            [false, false, false, false],
            [2, 1],
            [1, 1, 1],
        );
        let structure = BlockStructure::packed_column_major_with_keys(
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
            structure.clone(),
        )
        .unwrap();
        let mut dst =
            TensorMap::<f64, 4, 0>::from_vec_with_structure(vec![0.0, 0.0], dst_space, structure)
                .unwrap();
        let operation = TreeTransformOperationKey::braid([0, 2, 1, 3], [], [0, 1, 2, 3], []);

        let compiled =
            tree_pair_transform_structure(&SU2FusionRule, operation.clone(), &dst, &src).unwrap();
        assert!(compiled.has_pack_gemm_scatter_blocks());
        tree_pair_transform_into(&SU2FusionRule, operation, &mut dst, &src, 1.0, 0.0).unwrap();

        assert!((dst.data()[0] - 22.320_508_075_688_77).abs() < 1.0e-12);
        assert!((dst.data()[1] + 1.339_745_962_155_612_7).abs() < 1.0e-12);
    }

    #[test]
    fn tree_pair_transform_structure_replays_su2_recoupling_without_recompiling() {
        let src_key0 = all_codomain_fusion_tree_test_key(
            [1, 1, 1, 1],
            Some(0),
            [false, false, false, false],
            [0, 1],
            [1, 1, 1],
        );
        let src_key1 = all_codomain_fusion_tree_test_key(
            [1, 1, 1, 1],
            Some(0),
            [false, false, false, false],
            [2, 1],
            [1, 1, 1],
        );
        let block_structure = BlockStructure::packed_column_major_with_keys(
            4,
            [
                (src_key0.clone(), vec![1, 1, 1, 1]),
                (src_key1.clone(), vec![1, 1, 1, 1]),
            ],
        )
        .unwrap();
        let src_space = TensorMapSpace::<4, 0>::from_dims([1, 1, 1, 1], []).unwrap();
        let dst_space = TensorMapSpace::<4, 0>::from_dims([1, 1, 1, 1], []).unwrap();
        let mut src = TensorMap::<f64, 4, 0>::from_vec_with_structure(
            vec![10.0, 20.0],
            src_space,
            block_structure.clone(),
        )
        .unwrap();
        let mut dst = TensorMap::<f64, 4, 0>::from_vec_with_structure(
            vec![0.0, 0.0],
            dst_space,
            block_structure,
        )
        .unwrap();
        let operation = TreeTransformOperationKey::braid([0, 2, 1, 3], [], [0, 1, 2, 3], []);
        let structure =
            tree_pair_transform_structure(&SU2FusionRule, operation, &dst, &src).unwrap();
        let mut backend = DenseTreeTransformOperations::default();
        let mut workspace = TreeTransformWorkspace::default();
        let expected = |initial: [f64; 2], source: [f64; 2], alpha: f64, beta: f64| {
            let c = 0.866_025_403_784_438_6;
            [
                beta * initial[0] + alpha * (0.5 * source[0] + c * source[1]),
                beta * initial[1] + alpha * (c * source[0] - 0.5 * source[1]),
            ]
        };

        assert!(structure.has_pack_gemm_scatter_blocks());
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
        let expected_first = expected([0.0, 0.0], [10.0, 20.0], 1.0, 0.0);
        assert!((dst.data()[0] - expected_first[0]).abs() < 1.0e-12);
        assert!((dst.data()[1] - expected_first[1]).abs() < 1.0e-12);
        assert_eq!(workspace.source_len(), 2);
        assert_eq!(workspace.destination_len(), 2);

        src.data_mut().copy_from_slice(&[3.0, -4.0]);
        dst.data_mut().copy_from_slice(&[1.0, 2.0]);
        tree_transform_execute_with(
            &mut backend,
            &mut workspace,
            &structure,
            &mut dst,
            &src,
            2.0,
            -1.0,
        )
        .unwrap();
        let expected_second = expected([1.0, 2.0], [3.0, -4.0], 2.0, -1.0);
        assert!((dst.data()[0] - expected_second[0]).abs() < 1.0e-12);
        assert!((dst.data()[1] - expected_second[1]).abs() < 1.0e-12);
    }

    #[test]
    fn tree_transform_cache_reuses_su2_recoupling_descriptor() {
        let src_key0 = all_codomain_fusion_tree_test_key(
            [1, 1, 1, 1],
            Some(0),
            [false, false, false, false],
            [0, 1],
            [1, 1, 1],
        );
        let src_key1 = all_codomain_fusion_tree_test_key(
            [1, 1, 1, 1],
            Some(0),
            [false, false, false, false],
            [2, 1],
            [1, 1, 1],
        );
        let block_structure = BlockStructure::packed_column_major_with_keys(
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
        let mut dst = TensorMap::<f64, 4, 0>::from_vec_with_structure(
            vec![0.0, 0.0],
            dst_space,
            block_structure,
        )
        .unwrap();
        let operation = TreeTransformOperationKey::braid([0, 2, 1, 3], [], [0, 1, 2, 3], []);
        let mut cache = TreeTransformCache::<f64, TreeTransformBuiltinRuleCacheKey>::new();

        {
            let structure = cache
                .get_or_compile_tree_pair(&SU2FusionRule, operation.clone(), &dst, &src)
                .unwrap();
            assert!(structure.has_pack_gemm_scatter_blocks());
        }
        assert_eq!(cache.plan_len(), 1);
        assert_eq!(cache.structure_len(), 1);

        {
            let structure = cache
                .get_or_compile_tree_pair(&SU2FusionRule, operation.clone(), &dst, &src)
                .unwrap();
            assert!(structure.has_pack_gemm_scatter_blocks());
        }
        assert_eq!(cache.plan_len(), 1);
        assert_eq!(cache.structure_len(), 1);

        let structure = cache
            .get_or_compile_tree_pair(&SU2FusionRule, operation, &dst, &src)
            .unwrap();
        let mut backend = DenseTreeTransformOperations::default();
        let mut workspace = TreeTransformWorkspace::default();
        tree_transform_execute_with(
            &mut backend,
            &mut workspace,
            structure,
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
    fn tree_transform_cache_reuses_all_codomain_plan_across_degeneracy_shapes() {
        let src_key0 = all_codomain_fusion_tree_test_key(
            [1, 1, 1, 1],
            Some(0),
            [false, false, false, false],
            [0, 1],
            [1, 1, 1],
        );
        let src_key1 = all_codomain_fusion_tree_test_key(
            [1, 1, 1, 1],
            Some(0),
            [false, false, false, false],
            [2, 1],
            [1, 1, 1],
        );
        let small_structure = BlockStructure::packed_column_major_with_keys(
            4,
            [
                (src_key0.clone(), vec![1, 1, 1, 1]),
                (src_key1.clone(), vec![1, 1, 1, 1]),
            ],
        )
        .unwrap();
        let large_structure = BlockStructure::packed_column_major_with_keys(
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
        let operation = TreeTransformOperationKey::braid([0, 2, 1, 3], [], [0, 1, 2, 3], []);
        let mut cache = TreeTransformCache::<f64, TreeTransformBuiltinRuleCacheKey>::new();

        {
            let structure = cache
                .get_or_compile_all_codomain(&SU2FusionRule, operation.clone(), &dst, &src)
                .unwrap();
            assert!(structure.has_pack_gemm_scatter_blocks());
        }
        assert_eq!(cache.plan_len(), 1);
        assert_eq!(cache.structure_len(), 1);

        {
            let structure = cache
                .get_or_compile_all_codomain(&SU2FusionRule, operation.clone(), &dst, &src)
                .unwrap();
            assert!(structure.has_pack_gemm_scatter_blocks());
        }
        assert_eq!(cache.plan_len(), 1);
        assert_eq!(cache.structure_len(), 1);

        {
            let structure = cache
                .get_or_compile_all_codomain(
                    &SU2FusionRule,
                    operation.clone(),
                    &dst_large,
                    &src_large,
                )
                .unwrap();
            assert!(structure.has_pack_gemm_scatter_blocks());
        }
        assert_eq!(cache.plan_len(), 1);
        assert_eq!(cache.structure_len(), 2);

        let structure = cache
            .get_or_compile_all_codomain(&SU2FusionRule, operation, &dst, &src)
            .unwrap();
        let mut backend = DenseTreeTransformOperations::default();
        let mut workspace = TreeTransformWorkspace::default();
        tree_transform_execute_with(
            &mut backend,
            &mut workspace,
            structure,
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
        let src_key0 = all_codomain_fusion_tree_test_key(
            [1, 1, 1, 1],
            Some(0),
            [false, false, false, false],
            [0, 1],
            [1, 1, 1],
        );
        let src_key1 = all_codomain_fusion_tree_test_key(
            [1, 1, 1, 1],
            Some(0),
            [false, false, false, false],
            [2, 1],
            [1, 1, 1],
        );
        let block_structure = BlockStructure::packed_column_major_with_keys(
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
        let operation = TreeTransformOperationKey::braid([0, 2, 1, 3], [], [0, 1, 2, 3], []);
        let mut context =
            TreeTransformExecutionContext::<f64, TreeTransformBuiltinRuleCacheKey>::default();
        assert_eq!(context.cache().stats(), TreeTransformCacheStats::default());

        all_codomain_tree_transform_into_with_context(
            &mut context,
            &SU2FusionRule,
            operation.clone(),
            &mut dst,
            &src,
            1.0,
            0.0,
        )
        .unwrap();

        assert_eq!(context.cache().plan_len(), 1);
        assert_eq!(context.cache().structure_len(), 1);
        assert_eq!(context.cache().stats().plan_hits(), 0);
        assert_eq!(context.cache().stats().plan_misses(), 1);
        assert_eq!(context.cache().stats().structure_hits(), 0);
        assert_eq!(context.cache().stats().structure_misses(), 1);
        assert!((dst.data()[0] - 22.320_508_075_688_77).abs() < 1.0e-12);
        assert!((dst.data()[1] + 1.339_745_962_155_612_7).abs() < 1.0e-12);

        src.data_mut().copy_from_slice(&[3.0, -4.0]);
        dst.data_mut().copy_from_slice(&[1.0, 2.0]);
        context
            .all_codomain_tree_transform_into(&SU2FusionRule, operation, &mut dst, &src, 2.0, -1.0)
            .unwrap();

        assert_eq!(context.cache().plan_len(), 1);
        assert_eq!(context.cache().structure_len(), 1);
        assert_eq!(context.cache().stats().plan_hits(), 1);
        assert_eq!(context.cache().stats().plan_misses(), 1);
        assert_eq!(context.cache().stats().structure_hits(), 1);
        assert_eq!(context.cache().stats().structure_misses(), 1);
        let c = 0.866_025_403_784_438_6;
        assert!((dst.data()[0] - (-1.0 + 2.0 * (0.5 * 3.0 + c * -4.0))).abs() < 1.0e-12);
        assert!((dst.data()[1] - (-2.0 + 2.0 * (c * 3.0 - 0.5 * -4.0))).abs() < 1.0e-12);
        context.cache_mut().reset_stats();
        assert_eq!(context.cache().stats(), TreeTransformCacheStats::default());
    }

    #[test]
    fn tree_transform_execution_context_separates_tree_pair_and_all_codomain_scopes() {
        let src_key0 = all_codomain_fusion_tree_test_key(
            [1, 1, 1, 1],
            Some(0),
            [false, false, false, false],
            [0, 1],
            [1, 1, 1],
        );
        let src_key1 = all_codomain_fusion_tree_test_key(
            [1, 1, 1, 1],
            Some(0),
            [false, false, false, false],
            [2, 1],
            [1, 1, 1],
        );
        let block_structure = BlockStructure::packed_column_major_with_keys(
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
        let operation = TreeTransformOperationKey::braid([0, 2, 1, 3], [], [0, 1, 2, 3], []);
        let mut context =
            TreeTransformExecutionContext::<f64, TreeTransformBuiltinRuleCacheKey>::default();

        context
            .tree_pair_transform_into(&SU2FusionRule, operation.clone(), &mut dst, &src, 1.0, 0.0)
            .unwrap();
        assert_eq!(context.cache().plan_len(), 1);
        assert_eq!(context.cache().structure_len(), 1);

        dst.data_mut().copy_from_slice(&[0.0, 0.0]);
        context
            .all_codomain_tree_transform_into(&SU2FusionRule, operation, &mut dst, &src, 1.0, 0.0)
            .unwrap();

        assert_eq!(context.cache().plan_len(), 2);
        assert_eq!(context.cache().structure_len(), 2);
        assert!((dst.data()[0] - 22.320_508_075_688_77).abs() < 1.0e-12);
        assert!((dst.data()[1] + 1.339_745_962_155_612_7).abs() < 1.0e-12);
    }

    #[test]
    fn tree_pair_plan_builder_handles_su2_one_by_one_domain_crossing() {
        let src_key = BlockKey::from(FusionTreeBlockKey::pair_from_sector_ids(
            [1],
            [1],
            Some(1),
            [false],
            [false],
            [],
            [],
            [],
            [],
        ));
        let expected_dst_key = BlockKey::from(FusionTreeBlockKey::pair_from_sector_ids(
            [1],
            [1],
            Some(1),
            [true],
            [true],
            [],
            [],
            [],
            [],
        ));
        let src_structure =
            BlockStructure::packed_column_major_with_keys(2, [(src_key.clone(), vec![1, 1])])
                .unwrap();
        let dst_structure = BlockStructure::packed_column_major_with_keys(
            2,
            [(expected_dst_key.clone(), vec![1, 1])],
        )
        .unwrap();

        let plan = build_tree_pair_transform_group_plan(
            &SU2FusionRule,
            TreeTransformOperationKey::permute([1], [0]),
            &src_structure,
        )
        .unwrap();

        assert_eq!(plan.specs().len(), 1);
        let spec = &plan.specs()[0];
        assert_eq!(spec.src_keys(), &[src_key]);
        assert_eq!(spec.dst_keys(), &[expected_dst_key]);
        assert_eq!(spec.coefficients_src_by_dst().len(), 1);
        assert!((spec.coefficients_src_by_dst()[0] - 1.0).abs() < 1.0e-12);
        plan.compile_structures(&dst_structure, &src_structure)
            .unwrap();
    }

    #[test]
    fn tree_pair_transform_public_helper_executes_su2_domain_crossing() {
        let src_key = BlockKey::from(FusionTreeBlockKey::pair_from_sector_ids(
            [1],
            [1],
            Some(1),
            [false],
            [false],
            [],
            [],
            [],
            [],
        ));
        let expected_dst_key = BlockKey::from(FusionTreeBlockKey::pair_from_sector_ids(
            [1],
            [1],
            Some(1),
            [true],
            [true],
            [],
            [],
            [],
            [],
        ));
        let src_structure =
            BlockStructure::packed_column_major_with_keys(2, [(src_key, vec![1, 1])]).unwrap();
        let dst_structure = BlockStructure::packed_column_major_with_keys(
            2,
            [(expected_dst_key.clone(), vec![1, 1])],
        )
        .unwrap();
        let src_space = TensorMapSpace::<1, 1>::from_dims([1], [1]).unwrap();
        let dst_space = TensorMapSpace::<1, 1>::from_dims([1], [1]).unwrap();
        let src =
            TensorMap::<f64, 1, 1>::from_vec_with_structure(vec![7.0], src_space, src_structure)
                .unwrap();
        let mut dst =
            TensorMap::<f64, 1, 1>::from_vec_with_structure(vec![2.0], dst_space, dst_structure)
                .unwrap();
        let operation = TreeTransformOperationKey::permute([1], [0]);

        tree_pair_transform_into(&SU2FusionRule, operation, &mut dst, &src, 3.0, 5.0).unwrap();

        assert_eq!(dst.structure().block(0).unwrap().key(), &expected_dst_key);
        assert!((dst.data()[0] - 31.0).abs() < 1.0e-12);
    }

    #[test]
    fn tree_pair_transform_public_helper_executes_su2_with_complex_data() {
        let src_key = BlockKey::from(FusionTreeBlockKey::pair_from_sector_ids(
            [1],
            [1],
            Some(1),
            [false],
            [false],
            [],
            [],
            [],
            [],
        ));
        let expected_dst_key = BlockKey::from(FusionTreeBlockKey::pair_from_sector_ids(
            [1],
            [1],
            Some(1),
            [true],
            [true],
            [],
            [],
            [],
            [],
        ));
        let src_structure =
            BlockStructure::packed_column_major_with_keys(2, [(src_key, vec![1, 1])]).unwrap();
        let dst_structure = BlockStructure::packed_column_major_with_keys(
            2,
            [(expected_dst_key.clone(), vec![1, 1])],
        )
        .unwrap();
        let src_space = TensorMapSpace::<1, 1>::from_dims([1], [1]).unwrap();
        let dst_space = TensorMapSpace::<1, 1>::from_dims([1], [1]).unwrap();
        let src = TensorMap::<Complex64, 1, 1>::from_vec_with_structure(
            vec![Complex64::new(7.0, 1.0)],
            src_space,
            src_structure,
        )
        .unwrap();
        let mut dst = TensorMap::<Complex64, 1, 1>::from_vec_with_structure(
            vec![Complex64::new(2.0, -3.0)],
            dst_space,
            dst_structure,
        )
        .unwrap();
        let operation = TreeTransformOperationKey::permute([1], [0]);

        tree_pair_transform_into(
            &SU2FusionRule,
            operation,
            &mut dst,
            &src,
            Complex64::new(3.0, 0.0),
            Complex64::new(5.0, 0.0),
        )
        .unwrap();

        assert_eq!(dst.structure().block(0).unwrap().key(), &expected_dst_key);
        assert!((dst.data()[0] - Complex64::new(31.0, -12.0)).norm() < 1.0e-12);
    }

    #[test]
    fn tree_pair_operation_key_uses_tensorkit_global_source_axes() {
        let src_key = fusion_tree_test_key([1, 0], [1], 1, [false, false], [false]);
        let src_structure =
            BlockStructure::packed_column_major_with_keys(3, [(src_key, vec![1, 1, 1])]).unwrap();

        let local_domain_identity = build_tree_pair_transform_group_plan(
            &Z2FusionRule,
            TreeTransformOperationKey::permute([1, 0], [0]),
            &src_structure,
        )
        .unwrap_err();
        assert_eq!(
            local_domain_identity,
            OperationError::Core(CoreError::InvalidPermutation {
                permutation: vec![1, 0, 0],
                rank: 3,
            })
        );

        build_tree_pair_transform_group_plan(
            &Z2FusionRule,
            TreeTransformOperationKey::permute([1, 0], [2]),
            &src_structure,
        )
        .unwrap();
    }

    #[test]
    fn tree_pair_transform_public_helper_executes_split_changing_permute() {
        let src_key = BlockKey::from(FusionTreeBlockKey::pair_from_sector_ids(
            [1],
            [0, 1],
            Some(1),
            [false],
            [false, true],
            [],
            [],
            [],
            [1],
        ));
        let src_tree = expect_tree_key(&src_key);
        let operation = TreeTransformOperationKey::permute([0, 2], [1]);
        let (dst_tree, coefficient) =
            unique_permute_tree_pair(&Z2FusionRule, &src_tree, &[0, 2], &[1]).unwrap();
        let dst_key = BlockKey::from(dst_tree);
        let src_structure =
            BlockStructure::packed_column_major_with_keys(3, [(src_key, vec![1, 1, 1])]).unwrap();
        let dst_structure =
            BlockStructure::packed_column_major_with_keys(3, [(dst_key.clone(), vec![1, 1, 1])])
                .unwrap();
        let src_space = TensorMapSpace::<1, 2>::from_dims([1], [1, 1]).unwrap();
        let dst_space = TensorMapSpace::<2, 1>::from_dims([1, 1], [1]).unwrap();
        let src =
            TensorMap::<f64, 1, 2>::from_vec_with_structure(vec![7.0], src_space, src_structure)
                .unwrap();
        let mut dst =
            TensorMap::<f64, 2, 1>::from_vec_with_structure(vec![2.0], dst_space, dst_structure)
                .unwrap();

        tree_pair_transform_into(&Z2FusionRule, operation, &mut dst, &src, 3.0, 5.0).unwrap();

        assert_eq!(dst.structure().block(0).unwrap().key(), &dst_key);
        assert_eq!(dst.data(), &[3.0 * coefficient * 7.0 + 5.0 * 2.0]);
    }

    #[test]
    fn tree_pair_transform_public_helper_compiles_against_actual_destination_structure() {
        let src_key = BlockKey::from(FusionTreeBlockKey::pair_from_sector_ids(
            [1],
            [0, 1],
            Some(1),
            [false],
            [false, true],
            [],
            [],
            [],
            [1],
        ));
        let src_tree = expect_tree_key(&src_key);
        let operation = TreeTransformOperationKey::permute([0, 2], [1]);
        let (dst_tree, _) =
            unique_permute_tree_pair(&Z2FusionRule, &src_tree, &[0, 2], &[1]).unwrap();
        let expected_missing = BlockKey::from(dst_tree);
        let src_structure =
            BlockStructure::packed_column_major_with_keys(3, [(src_key.clone(), vec![1, 1, 1])])
                .unwrap();
        let wrong_dst_structure =
            BlockStructure::packed_column_major_with_keys(3, [(src_key, vec![1, 1, 1])]).unwrap();
        let src_space = TensorMapSpace::<1, 2>::from_dims([1], [1, 1]).unwrap();
        let dst_space = TensorMapSpace::<2, 1>::from_dims([1, 1], [1]).unwrap();
        let src =
            TensorMap::<f64, 1, 2>::from_vec_with_structure(vec![7.0], src_space, src_structure)
                .unwrap();
        let dst = TensorMap::<f64, 2, 1>::from_vec_with_structure(
            vec![0.0],
            dst_space,
            wrong_dst_structure,
        )
        .unwrap();

        let err = tree_pair_transform_structure(&Z2FusionRule, operation, &dst, &src).unwrap_err();

        assert_eq!(
            err,
            OperationError::MissingBlockKey {
                key: expected_missing,
            }
        );
    }

    #[test]
    fn multiplicity_free_product_tree_pair_plan_builder_handles_fz2_u1_su2_blocks() {
        let (rule, src_space, dst_space, [c0, c1]) = fz2_u1_su2_tree_pair_fixture();
        let src_structure = src_space.subblock_structure();
        let dst_structure = dst_space.subblock_structure();

        let plan = build_tree_pair_transform_group_plan(
            &rule,
            TreeTransformOperationKey::permute([1, 0], [2]),
            src_structure,
        )
        .unwrap();

        assert_eq!(plan.specs().len(), 2);
        assert!((single_transform_coefficient_for_coupled(&plan, c0) - 1.0).abs() < 1.0e-12);
        assert!((single_transform_coefficient_for_coupled(&plan, c1) + 1.0).abs() < 1.0e-12);
        plan.compile_structures(dst_structure, src_structure)
            .unwrap();
    }

    #[test]
    fn tree_pair_transform_public_helper_executes_product_fz2_u1_su2_blocks() {
        let (rule, src_space, dst_space, [c0, c1]) = fz2_u1_su2_tree_pair_fixture();
        let operation = TreeTransformOperationKey::permute([1, 0], [2]);
        let src =
            TensorMap::<f64, 2, 1>::from_vec_with_fusion_space(vec![10.0, 20.0], src_space.clone())
                .unwrap();
        let mut dst =
            TensorMap::<f64, 2, 1>::from_vec_with_fusion_space(vec![1.0, 2.0], dst_space.clone())
                .unwrap();
        let initial_dst = dst.data().to_vec();
        let plan = build_tree_pair_transform_group_plan(&rule, operation.clone(), src.structure())
            .unwrap();
        assert!((single_transform_coefficient_for_coupled(&plan, c0) - 1.0).abs() < 1.0e-12);
        assert!((single_transform_coefficient_for_coupled(&plan, c1) + 1.0).abs() < 1.0e-12);
        let mut expected = initial_dst
            .iter()
            .map(|value| 3.0 * value)
            .collect::<Vec<_>>();
        for spec in plan.specs() {
            let src_key = &spec.src_keys()[0];
            let dst_key = &spec.dst_keys()[0];
            let src_offset = src.structure().block_by_key(src_key).unwrap().offset();
            let dst_offset = dst.structure().block_by_key(dst_key).unwrap().offset();
            expected[dst_offset] +=
                2.0 * spec.coefficients_src_by_dst()[0] * src.data()[src_offset];
        }

        tree_pair_transform_into(&rule, operation, &mut dst, &src, 2.0, 3.0).unwrap();

        assert_eq!(dst.structure(), dst_space.subblock_structure());
        for (actual, expected) in dst.data().iter().zip(expected) {
            assert!(
                (actual - expected).abs() < 1.0e-12,
                "actual {actual} != expected {expected}"
            );
        }
    }

    #[test]
    fn tree_pair_transform_public_helper_executes_product_with_complex_data() {
        let (rule, src_space, dst_space, [c0, c1]) = fz2_u1_su2_tree_pair_fixture();
        let operation = TreeTransformOperationKey::permute([1, 0], [2]);
        let src = TensorMap::<Complex64, 2, 1>::from_vec_with_fusion_space(
            vec![Complex64::new(10.0, 1.0), Complex64::new(20.0, -2.0)],
            src_space.clone(),
        )
        .unwrap();
        let mut dst = TensorMap::<Complex64, 2, 1>::from_vec_with_fusion_space(
            vec![Complex64::new(1.0, 3.0), Complex64::new(2.0, -4.0)],
            dst_space.clone(),
        )
        .unwrap();
        let initial_dst = dst.data().to_vec();
        let plan = build_tree_pair_transform_group_plan(&rule, operation.clone(), src.structure())
            .unwrap();
        assert!((single_transform_coefficient_for_coupled(&plan, c0) - 1.0).abs() < 1.0e-12);
        assert!((single_transform_coefficient_for_coupled(&plan, c1) + 1.0).abs() < 1.0e-12);
        let alpha = Complex64::new(2.0, 0.0);
        let beta = Complex64::new(3.0, 0.0);
        let mut expected = initial_dst
            .iter()
            .map(|value| *value * beta)
            .collect::<Vec<_>>();
        for spec in plan.specs() {
            let src_key = &spec.src_keys()[0];
            let dst_key = &spec.dst_keys()[0];
            let src_offset = src.structure().block_by_key(src_key).unwrap().offset();
            let dst_offset = dst.structure().block_by_key(dst_key).unwrap().offset();
            expected[dst_offset] = expected[dst_offset]
                + src.data()[src_offset].scale_by_coefficient(spec.coefficients_src_by_dst()[0])
                    * alpha;
        }

        tree_pair_transform_into(&rule, operation, &mut dst, &src, alpha, beta).unwrap();

        assert_eq!(dst.structure(), dst_space.subblock_structure());
        assert_eq!(dst.data(), expected.as_slice());
    }

    #[test]
    fn tree_pair_transform_structure_replays_product_without_recompiling() {
        let (rule, src_space, dst_space, [c0, c1]) = fz2_u1_su2_tree_pair_fixture();
        let operation = TreeTransformOperationKey::permute([1, 0], [2]);
        let mut src =
            TensorMap::<f64, 2, 1>::from_vec_with_fusion_space(vec![10.0, 20.0], src_space.clone())
                .unwrap();
        let mut dst =
            TensorMap::<f64, 2, 1>::from_vec_with_fusion_space(vec![1.0, 2.0], dst_space.clone())
                .unwrap();
        let plan = build_tree_pair_transform_group_plan(&rule, operation.clone(), src.structure())
            .unwrap();
        let structure = tree_pair_transform_structure(&rule, operation, &dst, &src).unwrap();
        let mut backend = DenseTreeTransformOperations::default();
        let mut workspace = TreeTransformWorkspace::default();

        assert!((single_transform_coefficient_for_coupled(&plan, c0) - 1.0).abs() < 1.0e-12);
        assert!((single_transform_coefficient_for_coupled(&plan, c1) + 1.0).abs() < 1.0e-12);
        assert_eq!(structure.block_count(), 2);
        assert!(!structure.has_pack_gemm_scatter_blocks());
        let expected_first = expected_single_tree_pair_replay(
            &plan,
            dst.structure(),
            src.structure(),
            dst.data(),
            src.data(),
            2.0,
            3.0,
        );
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
        for (actual, expected) in dst.data().iter().zip(expected_first) {
            assert!(
                (actual - expected).abs() < 1.0e-12,
                "actual {actual} != expected {expected}"
            );
        }
        assert_eq!(workspace.source_len(), 0);
        assert_eq!(workspace.destination_len(), 0);

        src.data_mut().copy_from_slice(&[4.0, 5.0]);
        dst.data_mut().copy_from_slice(&[6.0, 7.0]);
        let expected_second = expected_single_tree_pair_replay(
            &plan,
            dst.structure(),
            src.structure(),
            dst.data(),
            src.data(),
            -1.0,
            0.5,
        );
        tree_transform_execute_with(
            &mut backend,
            &mut workspace,
            &structure,
            &mut dst,
            &src,
            -1.0,
            0.5,
        )
        .unwrap();
        for (actual, expected) in dst.data().iter().zip(expected_second) {
            assert!(
                (actual - expected).abs() < 1.0e-12,
                "actual {actual} != expected {expected}"
            );
        }
    }

    #[test]
    fn tree_transform_cache_reuses_product_plan_across_degeneracy_shapes() {
        let (rule, src_space, dst_space, _) = fz2_u1_su2_tree_pair_fixture();
        type RuleKey = <FpU1Su2Rule as TreeTransformRuleCacheKey>::Key;
        let operation = TreeTransformOperationKey::permute([1, 0], [2]);
        let src =
            TensorMap::<f64, 2, 1>::from_vec_with_fusion_space(vec![10.0, 20.0], src_space.clone())
                .unwrap();
        let mut dst =
            TensorMap::<f64, 2, 1>::from_vec_with_fusion_space(vec![1.0, 2.0], dst_space.clone())
                .unwrap();
        let src_large_structure =
            column_major_structure_like(src_space.subblock_structure(), vec![2, 1, 1]);
        let dst_large_structure =
            column_major_structure_like(dst_space.subblock_structure(), vec![2, 1, 1]);
        let large_space = TensorMapSpace::<2, 1>::from_dims([2, 1], [1]).unwrap();
        let src_large = TensorMap::<f64, 2, 1>::from_vec_with_structure(
            vec![1.0, 2.0, 3.0, 4.0],
            large_space.clone(),
            src_large_structure,
        )
        .unwrap();
        let dst_large = TensorMap::<f64, 2, 1>::from_vec_with_structure(
            vec![0.0, 0.0, 0.0, 0.0],
            large_space,
            dst_large_structure,
        )
        .unwrap();
        let mut cache = TreeTransformCache::<f64, RuleKey>::new();

        {
            let structure = cache
                .get_or_compile_tree_pair(&rule, operation.clone(), &dst, &src)
                .unwrap();
            assert_eq!(structure.block_count(), 2);
        }
        assert_eq!(cache.plan_len(), 1);
        assert_eq!(cache.structure_len(), 1);

        {
            let structure = cache
                .get_or_compile_tree_pair(&rule, operation.clone(), &dst, &src)
                .unwrap();
            assert_eq!(structure.block_count(), 2);
        }
        assert_eq!(cache.plan_len(), 1);
        assert_eq!(cache.structure_len(), 1);

        {
            let structure = cache
                .get_or_compile_tree_pair(&rule, operation, &dst_large, &src_large)
                .unwrap();
            assert_eq!(structure.block_count(), 2);
        }
        assert_eq!(cache.plan_len(), 1);
        assert_eq!(cache.structure_len(), 2);

        let structure = cache
            .get_or_compile_tree_pair(
                &rule,
                TreeTransformOperationKey::permute([1, 0], [2]),
                &dst,
                &src,
            )
            .unwrap();
        let plan = build_tree_pair_transform_group_plan(
            &rule,
            TreeTransformOperationKey::permute([1, 0], [2]),
            src.structure(),
        )
        .unwrap();
        let expected = expected_single_tree_pair_replay(
            &plan,
            dst.structure(),
            src.structure(),
            dst.data(),
            src.data(),
            2.0,
            3.0,
        );
        let mut backend = DenseTreeTransformOperations::default();
        let mut workspace = TreeTransformWorkspace::default();
        tree_transform_execute_with(
            &mut backend,
            &mut workspace,
            structure,
            &mut dst,
            &src,
            2.0,
            3.0,
        )
        .unwrap();
        for (actual, expected) in dst.data().iter().zip(expected) {
            assert!(
                (actual - expected).abs() < 1.0e-12,
                "actual {actual} != expected {expected}"
            );
        }
    }

    #[test]
    fn tree_transform_execution_context_reuses_product_tree_pair_cache() {
        let (rule, src_space, dst_space, _) = fz2_u1_su2_tree_pair_fixture();
        type RuleKey = <FpU1Su2Rule as TreeTransformRuleCacheKey>::Key;
        let operation = TreeTransformOperationKey::permute([1, 0], [2]);
        let mut src =
            TensorMap::<f64, 2, 1>::from_vec_with_fusion_space(vec![10.0, 20.0], src_space.clone())
                .unwrap();
        let mut dst =
            TensorMap::<f64, 2, 1>::from_vec_with_fusion_space(vec![1.0, 2.0], dst_space.clone())
                .unwrap();
        let plan = build_tree_pair_transform_group_plan(&rule, operation.clone(), src.structure())
            .unwrap();
        let mut context = TreeTransformExecutionContext::<f64, RuleKey>::default();
        let expected_first = expected_single_tree_pair_replay(
            &plan,
            dst.structure(),
            src.structure(),
            dst.data(),
            src.data(),
            2.0,
            3.0,
        );

        context
            .tree_pair_transform_into(&rule, operation.clone(), &mut dst, &src, 2.0, 3.0)
            .unwrap();

        assert_eq!(context.cache().plan_len(), 1);
        assert_eq!(context.cache().structure_len(), 1);
        for (actual, expected) in dst.data().iter().zip(expected_first) {
            assert!(
                (actual - expected).abs() < 1.0e-12,
                "actual {actual} != expected {expected}"
            );
        }

        src.data_mut().copy_from_slice(&[4.0, 5.0]);
        dst.data_mut().copy_from_slice(&[6.0, 7.0]);
        let expected_second = expected_single_tree_pair_replay(
            &plan,
            dst.structure(),
            src.structure(),
            dst.data(),
            src.data(),
            -1.0,
            0.5,
        );
        tree_pair_transform_into_with_context(
            &mut context,
            &rule,
            operation,
            &mut dst,
            &src,
            -1.0,
            0.5,
        )
        .unwrap();

        assert_eq!(context.cache().plan_len(), 1);
        assert_eq!(context.cache().structure_len(), 1);
        for (actual, expected) in dst.data().iter().zip(expected_second) {
            assert!(
                (actual - expected).abs() < 1.0e-12,
                "actual {actual} != expected {expected}"
            );
        }
    }

    #[test]
    fn tree_transform_execution_context_misses_on_different_tree_pair_operation() {
        let (rule, src_space, dst_space, _) = fz2_u1_su2_tree_pair_fixture();
        type RuleKey = <FpU1Su2Rule as TreeTransformRuleCacheKey>::Key;
        let src =
            TensorMap::<f64, 2, 1>::from_vec_with_fusion_space(vec![10.0, 20.0], src_space.clone())
                .unwrap();
        let mut dst =
            TensorMap::<f64, 2, 1>::from_vec_with_fusion_space(vec![1.0, 2.0], dst_space.clone())
                .unwrap();
        let mut context = TreeTransformExecutionContext::<f64, RuleKey>::default();

        context
            .tree_pair_transform_into(
                &rule,
                TreeTransformOperationKey::permute([1, 0], [2]),
                &mut dst,
                &src,
                1.0,
                0.0,
            )
            .unwrap();
        assert_eq!(context.cache().plan_len(), 1);
        assert_eq!(context.cache().structure_len(), 1);

        dst.data_mut().copy_from_slice(&[1.0, 2.0]);
        context
            .tree_pair_transform_into(
                &rule,
                TreeTransformOperationKey::braid([1, 0], [2], [1, 0], [2]),
                &mut dst,
                &src,
                1.0,
                0.0,
            )
            .unwrap();

        assert_eq!(context.cache().plan_len(), 2);
        assert_eq!(context.cache().structure_len(), 2);
    }

    #[test]
    fn unique_tree_transform_plan_builder_rejects_generic_fusion() {
        let src_key = fusion_tree_test_key([1, 1], [1], 1, [false, false], [false]);
        let src_structure =
            BlockStructure::packed_column_major_with_keys(3, [(src_key, vec![1, 1, 1])]).unwrap();
        let operation = TreeTransformOperationKey::braid([1, 0], [0], [1, 0], [0]);

        let err = build_unique_tree_transform_group_plan(
            &GenericMultiplicityRule,
            operation.clone(),
            &src_structure,
            |_| -> Result<(FusionTreeBlockKey, f64), OperationError> {
                unreachable!("GenericFusion must be rejected before transforming keys")
            },
        )
        .unwrap_err();

        assert_eq!(
            err,
            OperationError::UnsupportedFusionStyle {
                operation,
                style: FusionStyleKind::Generic,
            }
        );
    }

    #[test]
    fn tree_transform_operation_key_distinguishes_permute_from_explicit_braid() {
        assert!(TreeTransformOperationKey::permute([1, 0], [0]).requires_symmetric_braiding());
        assert!(!TreeTransformOperationKey::transpose([1, 0], [0]).requires_symmetric_braiding());
        assert!(!TreeTransformOperationKey::braid([1, 0], [0], [1, 0], [0])
            .requires_symmetric_braiding());
    }

    #[test]
    fn unique_tree_transform_plan_builder_rejects_permute_without_symmetric_braiding() {
        let src_key = fusion_tree_test_key([1, 0], [1], 1, [false, false], [false]);
        let src_structure =
            BlockStructure::packed_column_major_with_keys(3, [(src_key, vec![1, 1, 1])]).unwrap();
        let operation = TreeTransformOperationKey::permute([1, 0], [0]);

        let err = build_unique_tree_transform_group_plan(
            &UniqueAnyonicRule,
            operation.clone(),
            &src_structure,
            |_| -> Result<(FusionTreeBlockKey, f64), OperationError> {
                unreachable!("permutation must reject non-symmetric braiding before key transform")
            },
        )
        .unwrap_err();

        assert_eq!(
            err,
            OperationError::UnsupportedBraidingStyle {
                operation,
                style: BraidingStyleKind::Anyonic,
            }
        );
    }

    #[test]
    fn unique_tree_transform_plan_builder_defers_explicit_no_braiding_to_crossing_logic() {
        let src_key = fusion_tree_test_key([1, 0], [1], 1, [false, false], [false]);
        let src_tree = expect_tree_key(&src_key);
        let src_structure =
            BlockStructure::packed_column_major_with_keys(3, [(src_key.clone(), vec![1, 1, 1])])
                .unwrap();

        let plan = build_unique_tree_transform_group_plan(
            &UniquePlanarRule,
            TreeTransformOperationKey::braid([1, 0], [0], [1, 0], [0]),
            &src_structure,
            |src| Ok((src.clone(), 1.0_f64)),
        )
        .unwrap();

        assert_eq!(plan.specs().len(), 1);
        assert_eq!(plan.specs()[0].group_key(), &src_tree.group_key());
        assert_eq!(plan.specs()[0].src_keys(), &[src_key.clone()]);
        assert_eq!(plan.specs()[0].dst_keys(), &[src_key]);
        assert_eq!(plan.specs()[0].coefficients_src_by_dst(), &[1.0]);
    }

    #[test]
    fn unique_all_codomain_braid_plan_builder_lowers_codomain_single_tree() {
        let src_key = all_codomain_fusion_tree_test_key([1, 1], Some(0), [false, true], [], [1]);
        let expected_dst_key =
            all_codomain_fusion_tree_test_key([1, 1], Some(0), [true, false], [], [1]);
        let src_tree = expect_tree_key(&src_key);
        let src_structure =
            BlockStructure::packed_column_major_with_keys(2, [(src_key.clone(), vec![1, 1])])
                .unwrap();

        let plan = build_unique_all_codomain_tree_transform_group_plan(
            &UniqueAnyonicRule,
            TreeTransformOperationKey::braid(
                [1, 0],
                Vec::<usize>::new(),
                [0, 1],
                Vec::<usize>::new(),
            ),
            &src_structure,
        )
        .unwrap();

        assert_eq!(plan.specs().len(), 1);
        assert_eq!(plan.specs()[0].group_key(), &src_tree.group_key());
        assert_eq!(plan.specs()[0].src_keys(), &[src_key]);
        assert_eq!(plan.specs()[0].dst_keys(), &[expected_dst_key]);
        assert_eq!(plan.specs()[0].coefficients_src_by_dst(), &[-2.0]);
    }

    #[test]
    fn unique_all_codomain_permute_plan_builder_lowers_symmetric_permutation() {
        let src_key = all_codomain_fusion_tree_test_key([1, 0], Some(1), [false, true], [], [1]);
        let expected_dst_key =
            all_codomain_fusion_tree_test_key([0, 1], Some(1), [true, false], [], [1]);
        let src_structure =
            BlockStructure::packed_column_major_with_keys(2, [(src_key.clone(), vec![1, 1])])
                .unwrap();

        let plan = build_unique_all_codomain_tree_transform_group_plan(
            &UniqueZ2Rule,
            TreeTransformOperationKey::permute([1, 0], Vec::<usize>::new()),
            &src_structure,
        )
        .unwrap();

        assert_eq!(plan.specs().len(), 1);
        assert_eq!(plan.specs()[0].src_keys(), &[src_key]);
        assert_eq!(plan.specs()[0].dst_keys(), &[expected_dst_key]);
        assert_eq!(plan.specs()[0].coefficients_src_by_dst(), &[1.0]);
    }

    #[test]
    fn unique_all_codomain_plan_builder_rejects_domain_operation_scope() {
        let src_key = all_codomain_fusion_tree_test_key([1, 0], Some(1), [false, false], [], [1]);
        let src_structure =
            BlockStructure::packed_column_major_with_keys(2, [(src_key, vec![1, 1])]).unwrap();
        let operation = TreeTransformOperationKey::braid([1, 0], [0], [0, 1], [0]);

        let err = build_unique_all_codomain_tree_transform_group_plan(
            &UniqueZ2Rule,
            operation.clone(),
            &src_structure,
        )
        .unwrap_err();

        assert_eq!(
            err,
            OperationError::UnsupportedTreeTransformScope {
                operation,
                message: "all-codomain UniqueFusion lowering requires an empty domain operation",
            }
        );
    }

    #[test]
    fn unique_all_codomain_plan_builder_rejects_nonempty_domain_tree() {
        let src_key = fusion_tree_test_key([1, 0], [1], 1, [false, false], [false]);
        let src_structure =
            BlockStructure::packed_column_major_with_keys(3, [(src_key, vec![1, 1, 1])]).unwrap();

        let err = build_unique_all_codomain_tree_transform_group_plan(
            &UniqueZ2Rule,
            TreeTransformOperationKey::permute([1, 0], Vec::<usize>::new()),
            &src_structure,
        )
        .unwrap_err();

        assert_eq!(
            err,
            OperationError::ExpectedAllCodomainFusionTree { index: 0 }
        );
    }

    #[test]
    fn unique_all_codomain_permute_plan_builder_rejects_nonsymmetric_braiding() {
        let src_key = all_codomain_fusion_tree_test_key([1, 1], Some(0), [false, false], [], [1]);
        let src_structure =
            BlockStructure::packed_column_major_with_keys(2, [(src_key, vec![1, 1])]).unwrap();
        let operation = TreeTransformOperationKey::permute([1, 0], Vec::<usize>::new());

        let err = build_unique_all_codomain_tree_transform_group_plan(
            &UniqueAnyonicRule,
            operation.clone(),
            &src_structure,
        )
        .unwrap_err();

        assert_eq!(
            err,
            OperationError::UnsupportedBraidingStyle {
                operation,
                style: BraidingStyleKind::Anyonic,
            }
        );
    }

    #[test]
    fn unique_tree_pair_plan_builder_lowers_domain_only_permutation() {
        let src_key = BlockKey::from(FusionTreeBlockKey::pair_from_sector_ids(
            [1],
            [0, 1],
            Some(1),
            [false],
            [false, true],
            [],
            [],
            [],
            [1],
        ));
        let expected_dst_key = BlockKey::from(FusionTreeBlockKey::pair_from_sector_ids(
            [1],
            [1, 0],
            Some(1),
            [false],
            [true, false],
            [],
            [],
            [],
            [1],
        ));
        let src_tree = expect_tree_key(&src_key);
        let src_structure =
            BlockStructure::packed_column_major_with_keys(3, [(src_key.clone(), vec![1, 1, 1])])
                .unwrap();

        let plan = build_unique_tree_pair_transform_group_plan(
            &UniqueZ2Rule,
            TreeTransformOperationKey::permute([0], [2, 1]),
            &src_structure,
        )
        .unwrap();

        assert_eq!(plan.specs().len(), 1);
        assert_eq!(plan.specs()[0].group_key(), &src_tree.group_key());
        assert_eq!(plan.specs()[0].src_keys(), &[src_key]);
        assert_eq!(plan.specs()[0].dst_keys(), &[expected_dst_key]);
        assert_eq!(plan.specs()[0].coefficients_src_by_dst(), &[1.0]);
    }

    #[test]
    fn unique_tree_pair_plan_builder_lowers_codomain_domain_crossing_braid() {
        let src_key = BlockKey::from(FusionTreeBlockKey::pair_from_sector_ids(
            [1],
            [1],
            Some(1),
            [false],
            [true],
            [],
            [],
            [],
            [],
        ));
        let expected_dst_key = BlockKey::from(FusionTreeBlockKey::pair_from_sector_ids(
            [1],
            [1],
            Some(1),
            [false],
            [true],
            [],
            [],
            [],
            [],
        ));
        let src_structure =
            BlockStructure::packed_column_major_with_keys(2, [(src_key.clone(), vec![1, 1])])
                .unwrap();

        let plan = build_unique_tree_pair_transform_group_plan(
            &UniqueAnyonicRule,
            TreeTransformOperationKey::braid([1], [0], [0], [1]),
            &src_structure,
        )
        .unwrap();

        assert_eq!(plan.specs().len(), 1);
        assert_eq!(plan.specs()[0].src_keys(), &[src_key]);
        assert_eq!(plan.specs()[0].dst_keys(), &[expected_dst_key]);
        assert_eq!(plan.specs()[0].coefficients_src_by_dst(), &[-2.0]);
    }

    #[test]
    fn unique_tree_pair_plan_builder_lowers_cyclic_transpose() {
        let src_key = BlockKey::from(FusionTreeBlockKey::pair_from_sector_ids(
            [1],
            [1],
            Some(1),
            [false],
            [true],
            [],
            [],
            [],
            [],
        ));
        let expected_dst_key = src_key.clone();
        let src_structure =
            BlockStructure::packed_column_major_with_keys(2, [(src_key.clone(), vec![1, 1])])
                .unwrap();
        let operation = TreeTransformOperationKey::transpose([1], [0]);

        let plan =
            build_unique_tree_pair_transform_group_plan(&UniqueZ2Rule, operation, &src_structure)
                .unwrap();

        assert_eq!(plan.specs().len(), 1);
        assert_eq!(plan.specs()[0].src_keys(), &[src_key]);
        assert_eq!(plan.specs()[0].dst_keys(), &[expected_dst_key]);
        assert_eq!(plan.specs()[0].coefficients_src_by_dst(), &[1.0]);
    }

    #[test]
    fn unique_tree_pair_plan_builder_lowers_rank_four_cyclic_transpose() {
        let src_key = BlockKey::from(FusionTreeBlockKey::pair_from_sector_ids(
            [1, 0],
            [1, 0],
            Some(1),
            [false, false],
            [false, false],
            [],
            [],
            [1],
            [1],
        ));
        let expected_dst_key = BlockKey::from(FusionTreeBlockKey::pair_from_sector_ids(
            [1, 1],
            [0, 0],
            Some(0),
            [true, false],
            [false, true],
            [],
            [],
            [1],
            [1],
        ));
        let src_structure =
            BlockStructure::packed_column_major_with_keys(4, [(src_key.clone(), vec![1, 1, 1, 1])])
                .unwrap();
        let operation = TreeTransformOperationKey::transpose([2, 0], [3, 1]);

        let plan =
            build_unique_tree_pair_transform_group_plan(&UniqueZ2Rule, operation, &src_structure)
                .unwrap();

        assert_eq!(plan.specs().len(), 1);
        assert_eq!(plan.specs()[0].src_keys(), &[src_key]);
        assert_eq!(plan.specs()[0].dst_keys(), &[expected_dst_key]);
        assert_eq!(plan.specs()[0].coefficients_src_by_dst(), &[1.0]);
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
    fn tree_transform_sector_plan_key_is_rule_scope_and_source_sector_only() {
        let src_key1 = fusion_tree_test_key([10, 20], [30], 5, [false, true], [true]);
        let src_key2 = fusion_tree_test_key([10, 20], [30], 6, [false, true], [true]);
        let src_small = BlockStructure::packed_column_major_with_keys(
            2,
            [
                (src_key1.clone(), vec![2, 1]),
                (src_key2.clone(), vec![2, 1]),
            ],
        )
        .unwrap();
        let src_large = BlockStructure::packed_column_major_with_keys(
            2,
            [(src_key1, vec![3, 1]), (src_key2, vec![3, 1])],
        )
        .unwrap();
        let operation = TreeTransformOperationKey::transpose([1, 0], [0]);

        let z2_small =
            TreeTransformSectorPlanKey::tree_pair(&Z2FusionRule, operation.clone(), &src_small)
                .unwrap();
        let z2_large =
            TreeTransformSectorPlanKey::tree_pair(&Z2FusionRule, operation.clone(), &src_large)
                .unwrap();
        let fermion = TreeTransformSectorPlanKey::tree_pair(
            &FermionParityFusionRule,
            operation.clone(),
            &src_small,
        )
        .unwrap();
        let all_codomain =
            TreeTransformSectorPlanKey::all_codomain(&Z2FusionRule, operation, &src_small).unwrap();

        assert_eq!(z2_small, z2_large);
        assert_ne!(z2_small, fermion);
        assert_ne!(z2_small, all_codomain);
    }

    #[test]
    fn tree_transform_structure_cache_key_tracks_concrete_layout() {
        let key = fusion_tree_test_key([10, 20], [30], 5, [false, true], [true]);
        let src = BlockStructure::from_blocks_with_rank(
            2,
            vec![BlockSpec::with_key(key.clone(), vec![2, 3], vec![1, 2], 0).unwrap()],
        )
        .unwrap();
        let shape_changed = BlockStructure::from_blocks_with_rank(
            2,
            vec![BlockSpec::with_key(key.clone(), vec![3, 2], vec![1, 3], 0).unwrap()],
        )
        .unwrap();
        let stride_changed = BlockStructure::from_blocks_with_rank(
            2,
            vec![BlockSpec::with_key(key.clone(), vec![2, 3], vec![2, 1], 0).unwrap()],
        )
        .unwrap();
        let offset_changed = BlockStructure::from_blocks_with_rank(
            2,
            vec![BlockSpec::with_key(key.clone(), vec![2, 3], vec![1, 2], 1).unwrap()],
        )
        .unwrap();
        let plan_key = TreeTransformSectorPlanKey::tree_pair(
            &Z2FusionRule,
            TreeTransformOperationKey::transpose([1, 0], [0]),
            &src,
        )
        .unwrap();
        let base =
            TreeTransformStructureCacheKey::from_structures(plan_key.clone(), &src, &src).unwrap();

        assert_ne!(
            base,
            TreeTransformStructureCacheKey::from_structures(plan_key.clone(), &shape_changed, &src)
                .unwrap()
        );
        assert_ne!(
            base,
            TreeTransformStructureCacheKey::from_structures(
                plan_key.clone(),
                &stride_changed,
                &src
            )
            .unwrap()
        );
        assert_ne!(
            base,
            TreeTransformStructureCacheKey::from_structures(plan_key, &offset_changed, &src)
                .unwrap()
        );
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
