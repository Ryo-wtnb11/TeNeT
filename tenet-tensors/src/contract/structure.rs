use std::sync::Arc;

use num_traits::One;
use tenet_core::{
    BlockKey, BlockStructure, HostReadableStorage, HostWritableStorage, TensorMap, TensorStorage,
};
use tenet_dense::DenseDotConfig;

use crate::strided::{column_major_strides_usize, element_count, offset_to_isize};
use crate::{DenseBlockScalar, OperationError, RecouplingCoefficientAction};
use tenet_operations::structure_identity::validate_structure_identity;
use tenet_operations::{permutation_axes, TensorContractSpec};

use super::backend::TensorContractBackend;

#[derive(Clone, Debug, PartialEq)]
pub struct TensorContractStructure<C = f64> {
    dst_rank: usize,
    lhs_rank: usize,
    rhs_rank: usize,
    pub(super) lhs_contracting_axes: Vec<usize>,
    pub(super) rhs_contracting_axes: Vec<usize>,
    pub(super) output_axes: Vec<usize>,
    lhs_conjugate: bool,
    rhs_conjugate: bool,
    terms: Vec<TensorContractStructureTerm<C>>,
    descriptor: TensorContractDescriptor<C>,
    dst_structure: Arc<BlockStructure>,
    lhs_structure: Arc<BlockStructure>,
    rhs_structure: Arc<BlockStructure>,
    lhs_storage_structure: Arc<BlockStructure>,
    rhs_storage_structure: Arc<BlockStructure>,
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
    DDst,
    DLhs,
    DRhs,
>(
    dst: &TensorMap<TDst, DST_NOUT, DST_NIN, SDst, DDst>,
    lhs: &TensorMap<TLhs, LHS_NOUT, LHS_NIN, SLhs, DLhs>,
    rhs: &TensorMap<TRhs, RHS_NOUT, RHS_NIN, SRhs, DRhs>,
    axes: TensorContractSpec<'_>,
) -> Result<TensorContractStructure, OperationError>
where
    DDst: TensorStorage<TDst>,
    DLhs: TensorStorage<TLhs>,
    DRhs: TensorStorage<TRhs>,
{
    TensorContractStructure::compile(dst, lhs, rhs, axes)
}

pub(crate) const PLAIN_TENSORCONTRACT_FUSION_REQUIRES_FUSION_API: &str =
    "plain tensorcontract does not lower fusion-tree blocks; use tensorcontract_fusion_*";
pub(crate) const PLAIN_TENSORCONTRACT_BLOCK_SPARSE_UNSUPPORTED: &str =
    "block-sparse contraction enumeration is not implemented yet";

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
        DDst,
        DLhs,
        DRhs,
    >(
        dst: &TensorMap<TDst, DST_NOUT, DST_NIN, SDst, DDst>,
        lhs: &TensorMap<TLhs, LHS_NOUT, LHS_NIN, SLhs, DLhs>,
        rhs: &TensorMap<TRhs, RHS_NOUT, RHS_NIN, SRhs, DRhs>,
        axes: TensorContractSpec<'_>,
    ) -> Result<Self, OperationError>
    where
        DDst: TensorStorage<TDst>,
        DLhs: TensorStorage<TLhs>,
        DRhs: TensorStorage<TRhs>,
    {
        if dst.fusion_space().is_some()
            || lhs.fusion_space().is_some()
            || rhs.fusion_space().is_some()
        {
            return Err(OperationError::UnsupportedTensorContractScope {
                message: PLAIN_TENSORCONTRACT_FUSION_REQUIRES_FUSION_API,
            });
        }
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
        axes: TensorContractSpec<'_>,
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
        DDst,
        DLhs,
        DRhs,
    >(
        dst: &TensorMap<TDst, DST_NOUT, DST_NIN, SDst, DDst>,
        lhs: &TensorMap<TLhs, LHS_NOUT, LHS_NIN, SLhs, DLhs>,
        rhs: &TensorMap<TRhs, RHS_NOUT, RHS_NIN, SRhs, DRhs>,
        axes: TensorContractSpec<'_>,
        block_specs: &[TensorContractBlockSpec],
    ) -> Result<Self, OperationError>
    where
        DDst: TensorStorage<TDst>,
        DLhs: TensorStorage<TLhs>,
        DRhs: TensorStorage<TRhs>,
    {
        Self::compile_shared_structures_with_block_specs(
            Arc::clone(dst.structure()),
            Arc::clone(lhs.structure()),
            Arc::clone(rhs.structure()),
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
        axes: TensorContractSpec<'_>,
        block_specs: &[TensorContractBlockSpec],
    ) -> Result<Self, OperationError> {
        Self::compile_shared_structures_with_block_specs(
            Arc::new(dst_structure.clone()),
            Arc::new(lhs_structure.clone()),
            Arc::new(rhs_structure.clone()),
            Arc::new(lhs_structure.clone()),
            Arc::new(rhs_structure.clone()),
            axes,
            block_specs,
        )
    }

    pub(crate) fn compile_shared_structures_with_block_specs_and_storage(
        dst_structure: Arc<BlockStructure>,
        lhs_structure: Arc<BlockStructure>,
        rhs_structure: Arc<BlockStructure>,
        lhs_storage_structure: Arc<BlockStructure>,
        rhs_storage_structure: Arc<BlockStructure>,
        axes: TensorContractSpec<'_>,
        block_specs: &[TensorContractBlockSpec],
    ) -> Result<Self, OperationError> {
        Self::compile_shared_structures_with_block_specs(
            dst_structure,
            lhs_structure,
            rhs_structure,
            lhs_storage_structure,
            rhs_storage_structure,
            axes,
            block_specs,
        )
    }

    fn compile_shared_structures(
        dst_structure: Arc<BlockStructure>,
        lhs_structure: Arc<BlockStructure>,
        rhs_structure: Arc<BlockStructure>,
        axes: TensorContractSpec<'_>,
    ) -> Result<Self, OperationError> {
        if dst_structure.block_count() != 1
            || lhs_structure.block_count() != 1
            || rhs_structure.block_count() != 1
        {
            return Err(OperationError::UnsupportedTensorContractScope {
                message: PLAIN_TENSORCONTRACT_BLOCK_SPARSE_UNSUPPORTED,
            });
        }

        let block_specs = [TensorContractBlockSpec::new(0, 0, 0)];
        Self::compile_shared_structures_with_block_specs(
            Arc::clone(&dst_structure),
            Arc::clone(&lhs_structure),
            Arc::clone(&rhs_structure),
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
        lhs_storage_structure: Arc<BlockStructure>,
        rhs_storage_structure: Arc<BlockStructure>,
        axes: TensorContractSpec<'_>,
        block_specs: &[TensorContractBlockSpec],
    ) -> Result<Self, OperationError> {
        Self::compile_shared_structures_with_block_specs_inner(
            dst_structure,
            lhs_structure,
            rhs_structure,
            lhs_storage_structure,
            rhs_storage_structure,
            axes,
            block_specs,
        )
    }

    fn compile_shared_structures_with_block_specs_inner(
        dst_structure: Arc<BlockStructure>,
        lhs_structure: Arc<BlockStructure>,
        rhs_structure: Arc<BlockStructure>,
        lhs_storage_structure: Arc<BlockStructure>,
        rhs_storage_structure: Arc<BlockStructure>,
        axes: TensorContractSpec<'_>,
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
            lhs_conjugate: axis_plan.lhs_conjugate,
            rhs_conjugate: axis_plan.rhs_conjugate,
            terms,
            descriptor,
            dst_structure,
            lhs_structure,
            rhs_structure,
            lhs_storage_structure,
            rhs_storage_structure,
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
    pub fn lhs_conjugate(&self) -> bool {
        self.lhs_conjugate
    }

    #[inline]
    pub fn rhs_conjugate(&self) -> bool {
        self.rhs_conjugate
    }

    #[inline]
    pub fn terms(&self) -> &[TensorContractStructureTerm<C>] {
        &self.terms
    }

    #[inline]
    pub(super) fn descriptor(&self) -> &TensorContractDescriptor<C> {
        &self.descriptor
    }

    #[cfg(test)]
    pub(crate) fn dense_route_kind(&self) -> TensorContractDenseRouteKind {
        self.descriptor.dense_route_kind()
    }

    #[cfg(test)]
    pub(crate) fn dense_route_contracting_axes(&self) -> (&[usize], &[usize]) {
        (
            self.descriptor.lhs_contracting_axes(),
            self.descriptor.rhs_contracting_axes(),
        )
    }

    pub(super) fn validate_replay_structures(
        &self,
        dst_structure: &Arc<BlockStructure>,
        lhs_structure: &Arc<BlockStructure>,
        rhs_structure: &Arc<BlockStructure>,
    ) -> Result<(), OperationError> {
        validate_structure_identity("dst", &self.dst_structure, dst_structure)?;
        validate_structure_identity("lhs", &self.lhs_storage_structure, lhs_structure)?;
        validate_structure_identity("rhs", &self.rhs_storage_structure, rhs_structure)
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
        DDst,
        DLhs,
        DRhs,
    >(
        &self,
        backend: &mut B,
        workspace: &mut B::Workspace,
        dst: &mut TensorMap<D, DST_NOUT, DST_NIN, SDst, DDst>,
        lhs: &TensorMap<D, LHS_NOUT, LHS_NIN, SLhs, DLhs>,
        rhs: &TensorMap<D, RHS_NOUT, RHS_NIN, SRhs, DRhs>,
        alpha: D,
        beta: D,
    ) -> Result<(), OperationError>
    where
        B: TensorContractBackend<D, C>,
        D: DenseBlockScalar + RecouplingCoefficientAction<C>,
        DDst: HostWritableStorage<D>,
        DLhs: HostReadableStorage<D>,
        DRhs: HostReadableStorage<D>,
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
pub(super) struct TensorContractAxisPlan {
    pub(super) lhs_contracting_axes: Vec<usize>,
    pub(super) rhs_contracting_axes: Vec<usize>,
    pub(super) lhs_open_axes: Vec<usize>,
    pub(super) rhs_open_axes: Vec<usize>,
    pub(super) output_axes: Vec<usize>,
    pub(super) lhs_conjugate: bool,
    pub(super) rhs_conjugate: bool,
}

impl TensorContractAxisPlan {
    pub(super) fn compile(
        lhs_rank: usize,
        rhs_rank: usize,
        dst_rank: usize,
        axes: TensorContractSpec<'_>,
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
            lhs_conjugate: axes.lhs_conjugate(),
            rhs_conjugate: axes.rhs_conjugate(),
        })
    }
}

#[derive(Clone, Debug, PartialEq)]
pub(super) struct TensorContractDescriptor<C = f64> {
    dot_config: DenseDotConfig,
    dense_route_kind: TensorContractDenseRouteKind,
    dense_route_order: TensorContractDenseRouteOrder,
    lhs_contracting_axes: Vec<usize>,
    rhs_contracting_axes: Vec<usize>,
    lhs_open_axes: Vec<usize>,
    rhs_open_axes: Vec<usize>,
    lhs_conjugate: bool,
    rhs_conjugate: bool,
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
    pub(super) fn dot_config(&self) -> &DenseDotConfig {
        &self.dot_config
    }

    #[cfg(test)]
    #[inline]
    pub(crate) fn dense_route_kind(&self) -> TensorContractDenseRouteKind {
        self.dense_route_kind
    }

    #[inline]
    pub(super) fn dense_route_order(&self) -> TensorContractDenseRouteOrder {
        self.dense_route_order
    }

    #[inline]
    pub(super) fn lhs_contracting_axes(&self) -> &[usize] {
        &self.lhs_contracting_axes
    }

    #[inline]
    pub(super) fn rhs_contracting_axes(&self) -> &[usize] {
        &self.rhs_contracting_axes
    }

    #[inline]
    pub(super) fn lhs_open_axes(&self) -> &[usize] {
        &self.lhs_open_axes
    }

    #[inline]
    pub(super) fn rhs_open_axes(&self) -> &[usize] {
        &self.rhs_open_axes
    }

    #[inline]
    pub(super) fn lhs_conjugate(&self) -> bool {
        self.lhs_conjugate
    }

    #[inline]
    pub(super) fn rhs_conjugate(&self) -> bool {
        self.rhs_conjugate
    }

    pub(super) fn lhs_shape(&self, term: &TensorContractDescriptorTerm<C>) -> &[usize] {
        &self.lhs_shapes[term.lhs_layout_start..term.lhs_layout_start + term.lhs_rank]
    }

    pub(super) fn lhs_strides(&self, term: &TensorContractDescriptorTerm<C>) -> &[usize] {
        &self.lhs_strides[term.lhs_layout_start..term.lhs_layout_start + term.lhs_rank]
    }

    pub(super) fn rhs_shape(&self, term: &TensorContractDescriptorTerm<C>) -> &[usize] {
        &self.rhs_shapes[term.rhs_layout_start..term.rhs_layout_start + term.rhs_rank]
    }

    pub(super) fn rhs_strides(&self, term: &TensorContractDescriptorTerm<C>) -> &[usize] {
        &self.rhs_strides[term.rhs_layout_start..term.rhs_layout_start + term.rhs_rank]
    }

    pub(super) fn output_shape(&self, term: &TensorContractDescriptorTerm<C>) -> &[usize] {
        &self.output_shapes[term.output_layout_start..term.output_layout_start + term.output_rank]
    }

    pub(super) fn output_strides(&self, term: &TensorContractDescriptorTerm<C>) -> &[usize] {
        &self.output_strides[term.output_layout_start..term.output_layout_start + term.output_rank]
    }

    pub(super) fn scatter_shape(&self, term: &TensorContractDescriptorTerm<C>) -> &[usize] {
        &self.scatter_shapes
            [term.scatter_layout_start..term.scatter_layout_start + term.output_rank]
    }

    pub(super) fn dst_strides(&self, term: &TensorContractDescriptorTerm<C>) -> &[isize] {
        &self.dst_strides[term.scatter_layout_start..term.scatter_layout_start + term.output_rank]
    }

    pub(super) fn workspace_strides(&self, term: &TensorContractDescriptorTerm<C>) -> &[isize] {
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
        let dense_route = TensorContractDenseRoute::select(
            axis_plan,
            terms,
            dst_structure,
            lhs_structure,
            rhs_structure,
        )?;
        let mut descriptor = Self {
            dot_config: DenseDotConfig::new(
                dense_route.dot_lhs_contracting_axes(),
                dense_route.dot_rhs_contracting_axes(),
                Vec::new(),
                Vec::new(),
            ),
            dense_route_kind: dense_route.kind,
            dense_route_order: dense_route.order,
            lhs_contracting_axes: dense_route.lhs_contracting_axes,
            rhs_contracting_axes: dense_route.rhs_contracting_axes,
            lhs_open_axes: axis_plan.lhs_open_axes.clone(),
            rhs_open_axes: axis_plan.rhs_open_axes.clone(),
            lhs_conjugate: axis_plan.lhs_conjugate,
            rhs_conjugate: axis_plan.rhs_conjugate,
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

            let lhs_contract_shape = descriptor
                .lhs_contracting_axes
                .iter()
                .map(|&axis| lhs_block.shape()[axis])
                .collect::<Vec<_>>();
            let rhs_contract_shape = descriptor
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
            let semantic_output_shape = axis_plan
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
            let output_shape = dense_route
                .output_axes
                .iter()
                .map(|&axis| semantic_output_shape[axis])
                .collect::<Vec<_>>();
            let output_strides = column_major_strides_usize(&output_shape)?;
            let workspace_len = element_count(&output_shape)?;
            let output_layout_start = descriptor.output_shapes.len();
            descriptor.output_shapes.extend_from_slice(&output_shape);
            descriptor.output_strides.extend_from_slice(&output_strides);

            let scatter_shape = axis_plan
                .output_axes
                .iter()
                .map(|&axis| semantic_output_shape[axis])
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
            let workspace_axis_by_semantic_axis =
                inverse_permutation(&dense_route.output_axes, output_shape.len())?;
            for (dst_axis, &semantic_axis) in axis_plan.output_axes.iter().enumerate() {
                let workspace_axis = workspace_axis_by_semantic_axis[semantic_axis];
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

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum TensorContractDenseRouteKind {
    ForwardSortLhsContractingAxes,
    ForwardSortRhsContractingAxes,
    ReverseSortLhsContractingAxes,
    ReverseSortRhsContractingAxes,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum TensorContractDenseRouteOrder {
    LhsRhs,
    RhsLhs,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct TensorContractDenseRoute {
    kind: TensorContractDenseRouteKind,
    order: TensorContractDenseRouteOrder,
    lhs_contracting_axes: Vec<usize>,
    rhs_contracting_axes: Vec<usize>,
    output_axes: Vec<usize>,
}

impl TensorContractDenseRoute {
    fn select<C>(
        axis_plan: &TensorContractAxisPlan,
        terms: &[TensorContractStructureTerm<C>],
        dst_structure: &BlockStructure,
        lhs_structure: &BlockStructure,
        rhs_structure: &BlockStructure,
    ) -> Result<Self, OperationError> {
        let lhs_sort = sortperm(&axis_plan.lhs_contracting_axes);
        let lhs_sorted_by_lhs = take_by_permutation(&axis_plan.lhs_contracting_axes, &lhs_sort);
        let rhs_sorted_by_lhs = take_by_permutation(&axis_plan.rhs_contracting_axes, &lhs_sort);

        let rhs_sort = sortperm(&axis_plan.rhs_contracting_axes);
        let lhs_sorted_by_rhs = take_by_permutation(&axis_plan.lhs_contracting_axes, &rhs_sort);
        let rhs_sorted_by_rhs = take_by_permutation(&axis_plan.rhs_contracting_axes, &rhs_sort);

        let forward_output_axes = forward_output_axes(axis_plan);
        let reverse_output_axes = reverse_output_axes(axis_plan);

        let lhs_cost = dense_route_memcost(
            axis_plan,
            terms,
            dst_structure,
            lhs_structure,
            rhs_structure,
            &lhs_sorted_by_lhs,
            &rhs_sorted_by_lhs,
            &forward_output_axes,
            TensorContractDenseRouteOrder::LhsRhs,
        )?;
        let rhs_cost = dense_route_memcost(
            axis_plan,
            terms,
            dst_structure,
            lhs_structure,
            rhs_structure,
            &lhs_sorted_by_rhs,
            &rhs_sorted_by_rhs,
            &forward_output_axes,
            TensorContractDenseRouteOrder::LhsRhs,
        )?;

        let forward = if lhs_cost <= rhs_cost {
            Self {
                kind: TensorContractDenseRouteKind::ForwardSortLhsContractingAxes,
                order: TensorContractDenseRouteOrder::LhsRhs,
                lhs_contracting_axes: lhs_sorted_by_lhs.clone(),
                rhs_contracting_axes: rhs_sorted_by_lhs.clone(),
                output_axes: forward_output_axes.clone(),
            }
        } else {
            Self {
                kind: TensorContractDenseRouteKind::ForwardSortRhsContractingAxes,
                order: TensorContractDenseRouteOrder::LhsRhs,
                lhs_contracting_axes: lhs_sorted_by_rhs.clone(),
                rhs_contracting_axes: rhs_sorted_by_rhs.clone(),
                output_axes: forward_output_axes.clone(),
            }
        };

        if axis_plan.lhs_conjugate || axis_plan.rhs_conjugate {
            return Ok(forward);
        }

        let reverse_lhs_cost = dense_route_memcost(
            axis_plan,
            terms,
            dst_structure,
            lhs_structure,
            rhs_structure,
            &lhs_sorted_by_lhs,
            &rhs_sorted_by_lhs,
            &reverse_output_axes,
            TensorContractDenseRouteOrder::RhsLhs,
        )?;
        let reverse_rhs_cost = dense_route_memcost(
            axis_plan,
            terms,
            dst_structure,
            lhs_structure,
            rhs_structure,
            &lhs_sorted_by_rhs,
            &rhs_sorted_by_rhs,
            &reverse_output_axes,
            TensorContractDenseRouteOrder::RhsLhs,
        )?;

        let reverse = if reverse_lhs_cost <= reverse_rhs_cost {
            Self {
                kind: TensorContractDenseRouteKind::ReverseSortLhsContractingAxes,
                order: TensorContractDenseRouteOrder::RhsLhs,
                lhs_contracting_axes: lhs_sorted_by_lhs,
                rhs_contracting_axes: rhs_sorted_by_lhs,
                output_axes: reverse_output_axes.clone(),
            }
        } else {
            Self {
                kind: TensorContractDenseRouteKind::ReverseSortRhsContractingAxes,
                order: TensorContractDenseRouteOrder::RhsLhs,
                lhs_contracting_axes: lhs_sorted_by_rhs,
                rhs_contracting_axes: rhs_sorted_by_rhs,
                output_axes: reverse_output_axes,
            }
        };

        let forward_cost = lhs_cost.min(rhs_cost);
        let reverse_cost = reverse_lhs_cost.min(reverse_rhs_cost);
        if forward_cost <= reverse_cost {
            Ok(forward)
        } else {
            Ok(reverse)
        }
    }

    fn dot_lhs_contracting_axes(&self) -> Vec<usize> {
        match self.order {
            TensorContractDenseRouteOrder::LhsRhs => self.lhs_contracting_axes.clone(),
            TensorContractDenseRouteOrder::RhsLhs => {
                let mut axes = self.rhs_contracting_axes.clone();
                axes.reverse();
                axes
            }
        }
    }

    fn dot_rhs_contracting_axes(&self) -> Vec<usize> {
        match self.order {
            TensorContractDenseRouteOrder::LhsRhs => self.rhs_contracting_axes.clone(),
            TensorContractDenseRouteOrder::RhsLhs => {
                let mut axes = self.lhs_contracting_axes.clone();
                axes.reverse();
                axes
            }
        }
    }
}

fn dense_route_memcost<C>(
    axis_plan: &TensorContractAxisPlan,
    terms: &[TensorContractStructureTerm<C>],
    dst_structure: &BlockStructure,
    lhs_structure: &BlockStructure,
    rhs_structure: &BlockStructure,
    lhs_contracting_axes: &[usize],
    rhs_contracting_axes: &[usize],
    output_axes: &[usize],
    order: TensorContractDenseRouteOrder,
) -> Result<usize, OperationError> {
    let mut cost = 0usize;
    let output_dst_axes = dst_axes_for_route_output(axis_plan, output_axes)?;
    let first_output_len = match order {
        TensorContractDenseRouteOrder::LhsRhs => axis_plan.lhs_open_axes.len(),
        TensorContractDenseRouteOrder::RhsLhs => axis_plan.rhs_open_axes.len(),
    };
    for term in terms {
        let dst_block = dst_structure.block(term.dst_block())?;
        let lhs_block = lhs_structure.block(term.lhs_block())?;
        let rhs_block = rhs_structure.block(term.rhs_block())?;
        let lhs_needs_copy = match order {
            TensorContractDenseRouteOrder::LhsRhs => !is_dense_contractable_layout(
                lhs_block.shape(),
                lhs_block.strides(),
                &axis_plan.lhs_open_axes,
                lhs_contracting_axes,
                axis_plan.lhs_conjugate,
            )?,
            TensorContractDenseRouteOrder::RhsLhs => {
                let mut reversed_lhs_contracting_axes = lhs_contracting_axes.to_vec();
                reversed_lhs_contracting_axes.reverse();
                !is_dense_contractable_layout(
                    lhs_block.shape(),
                    lhs_block.strides(),
                    &reversed_lhs_contracting_axes,
                    &axis_plan.lhs_open_axes,
                    axis_plan.lhs_conjugate,
                )?
            }
        };
        if lhs_needs_copy {
            cost = cost
                .checked_add(element_count(lhs_block.shape())?)
                .ok_or(OperationError::ElementCountOverflow)?;
        }
        let rhs_needs_copy = match order {
            TensorContractDenseRouteOrder::LhsRhs => !is_dense_contractable_layout(
                rhs_block.shape(),
                rhs_block.strides(),
                rhs_contracting_axes,
                &axis_plan.rhs_open_axes,
                axis_plan.rhs_conjugate,
            )?,
            TensorContractDenseRouteOrder::RhsLhs => {
                let mut reversed_rhs_contracting_axes = rhs_contracting_axes.to_vec();
                reversed_rhs_contracting_axes.reverse();
                !is_dense_contractable_layout(
                    rhs_block.shape(),
                    rhs_block.strides(),
                    &axis_plan.rhs_open_axes,
                    &reversed_rhs_contracting_axes,
                    axis_plan.rhs_conjugate,
                )?
            }
        };
        if rhs_needs_copy {
            cost = cost
                .checked_add(element_count(rhs_block.shape())?)
                .ok_or(OperationError::ElementCountOverflow)?;
        }
        if !is_dense_destination_layout(
            dst_block.shape(),
            dst_block.strides(),
            &output_dst_axes[..first_output_len.min(output_dst_axes.len())],
            &output_dst_axes[first_output_len.min(output_dst_axes.len())..],
        )? {
            cost = cost
                .checked_add(element_count(dst_block.shape())?)
                .ok_or(OperationError::ElementCountOverflow)?;
        }
    }
    Ok(cost)
}

fn forward_output_axes(axis_plan: &TensorContractAxisPlan) -> Vec<usize> {
    (0..axis_plan.lhs_open_axes.len() + axis_plan.rhs_open_axes.len()).collect()
}

fn reverse_output_axes(axis_plan: &TensorContractAxisPlan) -> Vec<usize> {
    let lhs_open = axis_plan.lhs_open_axes.len();
    let rhs_open = axis_plan.rhs_open_axes.len();
    (lhs_open..lhs_open + rhs_open).chain(0..lhs_open).collect()
}

fn dst_axes_for_route_output(
    axis_plan: &TensorContractAxisPlan,
    route_output_axes: &[usize],
) -> Result<Vec<usize>, OperationError> {
    let dst_axis_by_semantic_axis =
        inverse_permutation(&axis_plan.output_axes, route_output_axes.len())?;
    Ok(route_output_axes
        .iter()
        .map(|&semantic_axis| dst_axis_by_semantic_axis[semantic_axis])
        .collect())
}

fn inverse_permutation(values: &[usize], len: usize) -> Result<Vec<usize>, OperationError> {
    let mut inverse = vec![usize::MAX; len];
    for (index, &value) in values.iter().enumerate() {
        if value >= len || inverse[value] != usize::MAX {
            return Err(OperationError::InvalidAxisSet {
                tensor: "permutation",
                axes: values.to_vec(),
                rank: len,
            });
        }
        inverse[value] = index;
    }
    if inverse.iter().any(|&index| index == usize::MAX) {
        return Err(OperationError::InvalidAxisSet {
            tensor: "permutation",
            axes: values.to_vec(),
            rank: len,
        });
    }
    Ok(inverse)
}

fn is_dense_destination_layout(
    shape: &[usize],
    strides: &[usize],
    first_axes: &[usize],
    second_axes: &[usize],
) -> Result<bool, OperationError> {
    let first_shape = first_axes
        .iter()
        .map(|&axis| shape[axis])
        .collect::<Vec<_>>();
    let first_strides = first_axes
        .iter()
        .map(|&axis| strides[axis])
        .collect::<Vec<_>>();
    let second_shape = second_axes
        .iter()
        .map(|&axis| shape[axis])
        .collect::<Vec<_>>();
    let second_strides = second_axes
        .iter()
        .map(|&axis| strides[axis])
        .collect::<Vec<_>>();
    let (first_fusable, _, first_stride) = can_fuse_strided_dims(&first_shape, &first_strides)?;
    let (second_fusable, _, _) = can_fuse_strided_dims(&second_shape, &second_strides)?;
    Ok(first_fusable && first_stride == 1 && second_fusable)
}

fn is_dense_contractable_layout(
    shape: &[usize],
    strides: &[usize],
    first_axes: &[usize],
    second_axes: &[usize],
    conjugate: bool,
) -> Result<bool, OperationError> {
    let first_shape = first_axes
        .iter()
        .map(|&axis| shape[axis])
        .collect::<Vec<_>>();
    let first_strides = first_axes
        .iter()
        .map(|&axis| strides[axis])
        .collect::<Vec<_>>();
    let second_shape = second_axes
        .iter()
        .map(|&axis| shape[axis])
        .collect::<Vec<_>>();
    let second_strides = second_axes
        .iter()
        .map(|&axis| strides[axis])
        .collect::<Vec<_>>();
    let (first_fusable, _, first_stride) = can_fuse_strided_dims(&first_shape, &first_strides)?;
    let (second_fusable, _, second_stride) = can_fuse_strided_dims(&second_shape, &second_strides)?;
    let stride_condition = if conjugate {
        second_stride == 1
    } else {
        first_stride == 1 || second_stride == 1
    };
    Ok(first_fusable && second_fusable && stride_condition)
}

fn can_fuse_strided_dims(
    dims: &[usize],
    strides: &[usize],
) -> Result<(bool, usize, usize), OperationError> {
    if dims.is_empty() {
        return Ok((true, 1, 1));
    }
    if dims[0] == 0 {
        return Ok((true, 0, 1));
    }
    if dims[0] == 1 {
        return can_fuse_strided_dims(&dims[1..], &strides[1..]);
    }

    let (tail_fusable, tail_dim, tail_stride) = can_fuse_strided_dims(&dims[1..], &strides[1..])?;
    let expected_tail_stride = dims[0]
        .checked_mul(strides[0])
        .ok_or(OperationError::ElementCountOverflow)?;
    let fused_dim = dims[0]
        .checked_mul(tail_dim)
        .ok_or(OperationError::ElementCountOverflow)?;
    if tail_fusable && (tail_stride == expected_tail_stride || tail_dim == 1) {
        let fused_stride = if fused_dim <= 1 { 1 } else { strides[0] };
        Ok((true, fused_dim, fused_stride))
    } else {
        Ok((false, fused_dim, strides[0]))
    }
}

fn sortperm(values: &[usize]) -> Vec<usize> {
    let mut permutation = (0..values.len()).collect::<Vec<_>>();
    permutation.sort_by_key(|&index| values[index]);
    permutation
}

fn take_by_permutation(values: &[usize], permutation: &[usize]) -> Vec<usize> {
    permutation.iter().map(|&index| values[index]).collect()
}

#[derive(Clone, Debug, PartialEq)]
pub(super) struct TensorContractDescriptorTerm<C = f64> {
    dst_block: usize,
    lhs_block: usize,
    rhs_block: usize,
    pub(super) lhs_layout_start: usize,
    pub(super) rhs_layout_start: usize,
    pub(super) output_layout_start: usize,
    pub(super) scatter_layout_start: usize,
    pub(super) lhs_rank: usize,
    pub(super) rhs_rank: usize,
    pub(super) output_rank: usize,
    pub(super) lhs_offset: usize,
    pub(super) rhs_offset: usize,
    pub(super) dst_offset: isize,
    pub(super) workspace_len: usize,
    pub(super) apply_beta: bool,
    pub(super) coefficient: C,
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
