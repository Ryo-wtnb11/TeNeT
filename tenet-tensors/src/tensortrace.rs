use core::ops::{Add, Mul};
use std::sync::Arc;

use num_traits::{One, Zero};
use tenet_core::{
    multiplicity_free_permute_tree_pair, split_fusion_tree, BlockKey, BlockStructure, FusionRule,
    FusionTensorMapSpace, FusionTreeBlockKey, FusionTreeHomSpace, FusionTreeKey,
    HostReadableStorage, HostWritableStorage, MultiplicityFreeRigidSymbols, SectorLeg, TensorMap,
    TensorStorage,
};

use crate::contract::{BoundDynamicFusionMapSpace, DynamicFusionMapSpace};
use crate::lowering::{adjoint_fusion_space_view, lower_tensortrace_source_adjoint_axes};
use crate::strided::offset_to_isize;
use crate::{tensortrace_raw_strided_kernel, tensortrace_raw_strided_kernel_add_with_coefficient};
use tenet_operations::structure_identity::validate_structure_identity;
use tenet_operations::OperationError;
use tenet_operations::TensorTraceAxisSpec;
use tenet_operations::{ConjugateValue, RealStructuralCoefficient, RecouplingCoefficientAction};

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TensorTraceStructure {
    dst_rank: usize,
    src_rank: usize,
    output_axes: Vec<usize>,
    trace_lhs_axes: Vec<usize>,
    trace_rhs_axes: Vec<usize>,
    terms: Vec<TensorTraceStructureTerm>,
    descriptor: TensorTraceDescriptor,
    dst_structure: Arc<BlockStructure>,
    src_structure: Arc<BlockStructure>,
}

pub fn tensortrace_structure<
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
    dst: &TensorMap<TDst, DST_NOUT, DST_NIN, SDst, DDst>,
    src: &TensorMap<TSrc, SRC_NOUT, SRC_NIN, SSrc, DSrc>,
    axes: TensorTraceAxisSpec<'_>,
) -> Result<TensorTraceStructure, OperationError>
where
    DDst: TensorStorage<TDst>,
    DSrc: TensorStorage<TSrc>,
{
    TensorTraceStructure::compile(dst, src, axes)
}

pub fn tensortrace_fusion_structure<
    R,
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
    rule: &R,
    dst: &TensorMap<TDst, DST_NOUT, DST_NIN, SDst, DDst>,
    src: &TensorMap<TSrc, SRC_NOUT, SRC_NIN, SSrc, DSrc>,
    axes: TensorTraceAxisSpec<'_>,
) -> Result<TensorTraceFusionStructure<R::Scalar>, OperationError>
where
    DDst: TensorStorage<TDst>,
    DSrc: TensorStorage<TSrc>,
    R: MultiplicityFreeRigidSymbols,
    R::Scalar: Clone
        + Add<Output = R::Scalar>
        + Mul<Output = R::Scalar>
        + Zero
        + RealStructuralCoefficient,
{
    let dst_fusion = dst
        .fusion_space()
        .ok_or(tenet_core::CoreError::MissingFusionSpace)?;
    let src_fusion = src
        .fusion_space()
        .ok_or(tenet_core::CoreError::MissingFusionSpace)?;
    if axes.source_conjugate() {
        let adjoint_src = adjoint_fusion_space_view(src_fusion)?;
        let adjoint_axes = lower_tensortrace_source_adjoint_axes::<SRC_NOUT, SRC_NIN>(axes)?;
        TensorTraceFusionStructure::compile_fusion_spaces_with_storage_structure(
            rule,
            dst_fusion,
            &adjoint_src,
            Arc::clone(src.structure()),
            adjoint_axes.as_spec(),
        )
    } else {
        TensorTraceFusionStructure::compile_fusion_spaces(rule, dst_fusion, src_fusion, axes)
    }
}

pub(crate) const PLAIN_TENSORTRACE_FUSION_REQUIRES_FUSION_API: &str =
    "plain tensortrace does not lower fusion-tree blocks; use tensortrace_fusion_*";
pub(crate) const PLAIN_TENSORTRACE_BLOCK_SPARSE_UNSUPPORTED: &str =
    "block-sparse tensortrace enumeration is not implemented yet";
pub(crate) const FUSION_TENSORTRACE_REQUIRES_SYMMETRIC_BRAIDING: &str =
    "fusion tensortrace requires symmetric braiding";

#[derive(Clone, Debug, PartialEq)]
pub struct TensorTraceFusionStructure<C> {
    dst_rank: usize,
    src_rank: usize,
    output_axes: Vec<usize>,
    trace_lhs_axes: Vec<usize>,
    trace_rhs_axes: Vec<usize>,
    terms: Vec<TensorTraceFusionStructureTerm<C>>,
    descriptor: TensorTraceDescriptor,
    dst_structure: Arc<BlockStructure>,
    src_structure: Arc<BlockStructure>,
    src_storage_structure: Arc<BlockStructure>,
}

impl<C> TensorTraceFusionStructure<C> {
    pub fn compile_fusion_spaces<
        R,
        const DST_NOUT: usize,
        const DST_NIN: usize,
        const SRC_NOUT: usize,
        const SRC_NIN: usize,
    >(
        rule: &R,
        dst: &FusionTensorMapSpace<DST_NOUT, DST_NIN>,
        src: &FusionTensorMapSpace<SRC_NOUT, SRC_NIN>,
        axes: TensorTraceAxisSpec<'_>,
    ) -> Result<Self, OperationError>
    where
        R: MultiplicityFreeRigidSymbols<Scalar = C>,
        C: Clone + Add<Output = C> + Mul<Output = C> + Zero + RealStructuralCoefficient,
    {
        dst.validate_rule(rule).map_err(OperationError::Core)?;
        src.validate_rule(rule).map_err(OperationError::Core)?;
        Self::compile_fusion_spaces_with_storage_structure(
            rule,
            dst,
            src,
            Arc::clone(src.subblock_structure()),
            axes,
        )
    }

    pub(crate) fn compile_fusion_spaces_with_storage_structure<
        R,
        const DST_NOUT: usize,
        const DST_NIN: usize,
        const SRC_NOUT: usize,
        const SRC_NIN: usize,
    >(
        rule: &R,
        dst: &FusionTensorMapSpace<DST_NOUT, DST_NIN>,
        src: &FusionTensorMapSpace<SRC_NOUT, SRC_NIN>,
        src_storage_structure: Arc<BlockStructure>,
        axes: TensorTraceAxisSpec<'_>,
    ) -> Result<Self, OperationError>
    where
        R: MultiplicityFreeRigidSymbols<Scalar = C>,
        C: Clone + Add<Output = C> + Mul<Output = C> + Zero + RealStructuralCoefficient,
    {
        dst.validate_rule(rule)?;
        src.validate_rule(rule)?;
        Self::compile_fusion_parts(
            rule,
            dst.homspace(),
            Arc::clone(dst.subblock_structure()),
            src.homspace(),
            Arc::clone(src.subblock_structure()),
            src_storage_structure,
            DST_NOUT,
            axes,
        )
    }

    /// Dynamic-rank [`Self::compile_fusion_spaces`] retaining the source
    /// provider authority.
    pub fn compile_fusion_dyn<R>(
        dst: &BoundDynamicFusionMapSpace<R>,
        src: &BoundDynamicFusionMapSpace<R>,
        axes: TensorTraceAxisSpec<'_>,
    ) -> Result<Self, OperationError>
    where
        R: MultiplicityFreeRigidSymbols<Scalar = C>,
        C: Clone + Add<Output = C> + Mul<Output = C> + Zero + RealStructuralCoefficient,
    {
        // Why not accept a separate rule: trace coefficients are categorical
        // semantics, so they must come from the provider that proved the source.
        Self::compile_fusion_dyn_raw(src.provider(), dst.space(), src.space(), axes)
    }

    pub(crate) fn compile_fusion_dyn_raw<R>(
        rule: &R,
        dst: &DynamicFusionMapSpace,
        src: &DynamicFusionMapSpace,
        axes: TensorTraceAxisSpec<'_>,
    ) -> Result<Self, OperationError>
    where
        R: MultiplicityFreeRigidSymbols<Scalar = C>,
        C: Clone + Add<Output = C> + Mul<Output = C> + Zero + RealStructuralCoefficient,
    {
        dst.validate_rule(rule)?;
        src.validate_rule(rule)?;
        Self::compile_fusion_parts(
            rule,
            dst.homspace(),
            Arc::clone(dst.structure()),
            src.homspace(),
            Arc::clone(src.structure()),
            Arc::clone(src.structure()),
            dst.nout(),
            axes,
        )
    }

    /// Rank-runtime core shared by the const-generic and dynamic compiles.
    #[allow(clippy::too_many_arguments)]
    fn compile_fusion_parts<R>(
        rule: &R,
        dst_homspace: &FusionTreeHomSpace,
        dst_structure: Arc<BlockStructure>,
        src_homspace: &FusionTreeHomSpace,
        src_structure: Arc<BlockStructure>,
        src_storage_structure: Arc<BlockStructure>,
        dst_codomain_rank: usize,
        axes: TensorTraceAxisSpec<'_>,
    ) -> Result<Self, OperationError>
    where
        R: MultiplicityFreeRigidSymbols<Scalar = C>,
        C: Clone + Add<Output = C> + Mul<Output = C> + Zero + RealStructuralCoefficient,
    {
        if !rule.braiding_style().is_symmetric() {
            return Err(OperationError::UnsupportedTensorContractScope {
                message: FUSION_TENSORTRACE_REQUIRES_SYMMETRIC_BRAIDING,
            });
        }
        let axis_plan =
            TensorTraceAxisPlan::compile(src_structure.rank(), dst_structure.rank(), axes)?;
        validate_fusion_trace_homspace(
            rule,
            dst_homspace,
            src_homspace,
            &axis_plan,
            dst_codomain_rank,
        )?;
        let terms = build_fusion_trace_terms(
            rule,
            &dst_structure,
            &src_structure,
            &axis_plan,
            dst_codomain_rank,
        )?;
        let dense_terms = terms
            .iter()
            .map(|term| TensorTraceStructureTerm {
                key: BlockKey::from(term.dst_key.clone()),
                dst_block: term.dst_block,
                src_block: term.src_block,
            })
            .collect::<Vec<_>>();
        let descriptor = TensorTraceDescriptor::compile(
            &axis_plan,
            &dense_terms,
            &dst_structure,
            &src_structure,
        )?;

        Ok(Self {
            dst_rank: dst_structure.rank(),
            src_rank: src_structure.rank(),
            output_axes: axis_plan.output_axes,
            trace_lhs_axes: axis_plan.trace_lhs_axes,
            trace_rhs_axes: axis_plan.trace_rhs_axes,
            terms,
            descriptor,
            dst_structure,
            src_structure,
            src_storage_structure,
        })
    }

    #[inline]
    pub fn dst_rank(&self) -> usize {
        self.dst_rank
    }

    #[inline]
    pub fn src_rank(&self) -> usize {
        self.src_rank
    }

    #[inline]
    pub fn output_axes(&self) -> &[usize] {
        &self.output_axes
    }

    #[inline]
    pub fn trace_lhs_axes(&self) -> &[usize] {
        &self.trace_lhs_axes
    }

    #[inline]
    pub fn trace_rhs_axes(&self) -> &[usize] {
        &self.trace_rhs_axes
    }

    #[inline]
    pub fn terms(&self) -> &[TensorTraceFusionStructureTerm<C>] {
        &self.terms
    }

    #[inline]
    pub(crate) fn descriptor(&self) -> &TensorTraceDescriptor {
        &self.descriptor
    }

    pub(crate) fn validate_replay_structures(
        &self,
        dst_structure: &Arc<BlockStructure>,
        src_structure: &Arc<BlockStructure>,
    ) -> Result<(), OperationError> {
        validate_structure_identity("dst", &self.dst_structure, dst_structure)?;
        validate_structure_identity("src", &self.src_storage_structure, src_structure)
    }

    pub fn execute_with<
        B,
        T,
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
        backend: &mut B,
        allocator: &mut B::Allocator,
        dst: &mut TensorMap<T, DST_NOUT, DST_NIN, SDst, DDst>,
        src: &TensorMap<T, SRC_NOUT, SRC_NIN, SSrc, DSrc>,
        alpha: T,
        beta: T,
    ) -> Result<(), OperationError>
    where
        B: crate::TensorTraceOperationsBackend,
        T: Copy
            + Add<T, Output = T>
            + Mul<T, Output = T>
            + PartialEq
            + Zero
            + One
            + ConjugateValue
            + RecouplingCoefficientAction<C>
            + strided_kernel::MaybeSendSync,
        C: Copy,
        DDst: HostWritableStorage<T>,
        DSrc: HostReadableStorage<T>,
    {
        backend.tensortrace_fusion_structure_into(allocator, self, dst, src, alpha, beta)
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct TensorTraceFusionStructureTerm<C> {
    dst_key: FusionTreeBlockKey,
    src_key: FusionTreeBlockKey,
    dst_block: usize,
    src_block: usize,
    coefficient: C,
}

impl<C> TensorTraceFusionStructureTerm<C> {
    #[inline]
    pub fn dst_key(&self) -> &FusionTreeBlockKey {
        &self.dst_key
    }

    #[inline]
    pub fn src_key(&self) -> &FusionTreeBlockKey {
        &self.src_key
    }

    #[inline]
    pub fn dst_block(&self) -> usize {
        self.dst_block
    }

    #[inline]
    pub fn src_block(&self) -> usize {
        self.src_block
    }

    #[inline]
    pub fn coefficient(&self) -> &C {
        &self.coefficient
    }
}

impl TensorTraceStructure {
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
        dst: &TensorMap<TDst, DST_NOUT, DST_NIN, SDst, DDst>,
        src: &TensorMap<TSrc, SRC_NOUT, SRC_NIN, SSrc, DSrc>,
        axes: TensorTraceAxisSpec<'_>,
    ) -> Result<Self, OperationError>
    where
        DDst: TensorStorage<TDst>,
        DSrc: TensorStorage<TSrc>,
    {
        if dst.fusion_space().is_some() || src.fusion_space().is_some() {
            return Err(OperationError::UnsupportedTensorContractScope {
                message: PLAIN_TENSORTRACE_FUSION_REQUIRES_FUSION_API,
            });
        }
        Self::compile_shared_structures(
            Arc::clone(dst.structure()),
            Arc::clone(src.structure()),
            axes,
        )
    }

    pub fn compile_structures(
        dst_structure: &BlockStructure,
        src_structure: &BlockStructure,
        axes: TensorTraceAxisSpec<'_>,
    ) -> Result<Self, OperationError> {
        Self::compile_shared_structures(
            Arc::new(dst_structure.clone()),
            Arc::new(src_structure.clone()),
            axes,
        )
    }

    fn compile_shared_structures(
        dst_structure: Arc<BlockStructure>,
        src_structure: Arc<BlockStructure>,
        axes: TensorTraceAxisSpec<'_>,
    ) -> Result<Self, OperationError> {
        if dst_structure.block_count() != 1 || src_structure.block_count() != 1 {
            return Err(OperationError::UnsupportedTensorContractScope {
                message: PLAIN_TENSORTRACE_BLOCK_SPARSE_UNSUPPORTED,
            });
        }

        let axis_plan =
            TensorTraceAxisPlan::compile(src_structure.rank(), dst_structure.rank(), axes)?;
        let dst_block = dst_structure.block(0)?;
        let terms = vec![TensorTraceStructureTerm {
            key: dst_block.key().clone(),
            dst_block: 0,
            src_block: 0,
        }];
        let descriptor =
            TensorTraceDescriptor::compile(&axis_plan, &terms, &dst_structure, &src_structure)?;

        Ok(Self {
            dst_rank: dst_structure.rank(),
            src_rank: src_structure.rank(),
            output_axes: axis_plan.output_axes,
            trace_lhs_axes: axis_plan.trace_lhs_axes,
            trace_rhs_axes: axis_plan.trace_rhs_axes,
            terms,
            descriptor,
            dst_structure,
            src_structure,
        })
    }

    #[inline]
    pub fn dst_rank(&self) -> usize {
        self.dst_rank
    }

    #[inline]
    pub fn src_rank(&self) -> usize {
        self.src_rank
    }

    #[inline]
    pub fn output_axes(&self) -> &[usize] {
        &self.output_axes
    }

    #[inline]
    pub fn trace_lhs_axes(&self) -> &[usize] {
        &self.trace_lhs_axes
    }

    #[inline]
    pub fn trace_rhs_axes(&self) -> &[usize] {
        &self.trace_rhs_axes
    }

    #[inline]
    pub fn terms(&self) -> &[TensorTraceStructureTerm] {
        &self.terms
    }

    #[inline]
    pub(crate) fn descriptor(&self) -> &TensorTraceDescriptor {
        &self.descriptor
    }

    pub(crate) fn validate_replay_structures(
        &self,
        dst_structure: &Arc<BlockStructure>,
        src_structure: &Arc<BlockStructure>,
    ) -> Result<(), OperationError> {
        validate_structure_identity("dst", &self.dst_structure, dst_structure)?;
        validate_structure_identity("src", &self.src_structure, src_structure)
    }

    pub fn execute_with<
        B,
        T,
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
        backend: &mut B,
        allocator: &mut B::Allocator,
        dst: &mut TensorMap<T, DST_NOUT, DST_NIN, SDst, DDst>,
        src: &TensorMap<T, SRC_NOUT, SRC_NIN, SSrc, DSrc>,
        alpha: T,
        beta: T,
    ) -> Result<(), OperationError>
    where
        B: crate::TensorTraceOperationsBackend,
        T: Copy
            + Add<T, Output = T>
            + Mul<T, Output = T>
            + PartialEq
            + Zero
            + One
            + ConjugateValue
            + strided_kernel::MaybeSendSync,
        DDst: HostWritableStorage<T>,
        DSrc: HostReadableStorage<T>,
    {
        backend.tensortrace_structure_into(allocator, self, dst, src, alpha, beta)
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TensorTraceStructureTerm {
    key: BlockKey,
    dst_block: usize,
    src_block: usize,
}

impl TensorTraceStructureTerm {
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
pub(crate) struct TensorTraceDescriptor {
    source_conjugate: bool,
    terms: Vec<TensorTraceDescriptorTerm>,
    output_shape: Vec<usize>,
    trace_shape: Vec<usize>,
    dst_strides: Vec<isize>,
    src_output_strides: Vec<isize>,
    src_trace_strides: Vec<isize>,
}

impl TensorTraceDescriptor {
    #[inline]
    pub(crate) fn terms(&self) -> &[TensorTraceDescriptorTerm] {
        &self.terms
    }

    #[inline]
    pub(crate) fn source_conjugate(&self) -> bool {
        self.source_conjugate
    }

    pub(crate) fn output_shape(&self, term: &TensorTraceDescriptorTerm) -> &[usize] {
        &self.output_shape[term.output_layout_start..term.output_layout_start + term.output_rank]
    }

    pub(crate) fn trace_shape(&self, term: &TensorTraceDescriptorTerm) -> &[usize] {
        &self.trace_shape[term.trace_layout_start..term.trace_layout_start + term.trace_rank]
    }

    pub(crate) fn dst_strides(&self, term: &TensorTraceDescriptorTerm) -> &[isize] {
        &self.dst_strides[term.output_layout_start..term.output_layout_start + term.output_rank]
    }

    pub(crate) fn src_output_strides(&self, term: &TensorTraceDescriptorTerm) -> &[isize] {
        &self.src_output_strides
            [term.output_layout_start..term.output_layout_start + term.output_rank]
    }

    pub(crate) fn src_trace_strides(&self, term: &TensorTraceDescriptorTerm) -> &[isize] {
        &self.src_trace_strides[term.trace_layout_start..term.trace_layout_start + term.trace_rank]
    }

    fn compile(
        axis_plan: &TensorTraceAxisPlan,
        terms: &[TensorTraceStructureTerm],
        dst_structure: &BlockStructure,
        src_structure: &BlockStructure,
    ) -> Result<Self, OperationError> {
        let mut descriptor = Self {
            source_conjugate: axis_plan.source_conjugate,
            ..Self::default()
        };
        descriptor.terms.reserve(terms.len());
        descriptor
            .output_shape
            .reserve(terms.len().saturating_mul(axis_plan.output_axes.len()));
        descriptor
            .trace_shape
            .reserve(terms.len().saturating_mul(axis_plan.trace_lhs_axes.len()));
        descriptor
            .dst_strides
            .reserve(terms.len().saturating_mul(axis_plan.output_axes.len()));
        descriptor
            .src_output_strides
            .reserve(terms.len().saturating_mul(axis_plan.output_axes.len()));
        descriptor
            .src_trace_strides
            .reserve(terms.len().saturating_mul(axis_plan.trace_lhs_axes.len()));

        for term in terms {
            let dst_block = dst_structure.block(term.dst_block())?;
            let src_block = src_structure.block(term.src_block())?;
            let output_layout_start = descriptor.output_shape.len();
            for (dst_axis, &src_axis) in axis_plan.output_axes.iter().enumerate() {
                let dst_dim = dst_block.shape()[dst_axis];
                let src_dim = src_block.shape()[src_axis];
                if dst_dim != src_dim {
                    let src_shape = axis_plan
                        .output_axes
                        .iter()
                        .map(|&axis| src_block.shape()[axis])
                        .collect::<Vec<_>>();
                    return Err(OperationError::ShapeMismatch {
                        dst: dst_block.shape().to_vec(),
                        src: src_shape,
                    });
                }
                descriptor.output_shape.push(dst_dim);
                descriptor
                    .dst_strides
                    .push(stride_to_isize(dst_block.strides()[dst_axis])?);
                descriptor
                    .src_output_strides
                    .push(stride_to_isize(src_block.strides()[src_axis])?);
            }

            let trace_layout_start = descriptor.trace_shape.len();
            for (&lhs_axis, &rhs_axis) in axis_plan
                .trace_lhs_axes
                .iter()
                .zip(axis_plan.trace_rhs_axes.iter())
            {
                let lhs_dim = src_block.shape()[lhs_axis];
                let rhs_dim = src_block.shape()[rhs_axis];
                if lhs_dim != rhs_dim {
                    return Err(OperationError::ShapeMismatch {
                        dst: vec![lhs_dim],
                        src: vec![rhs_dim],
                    });
                }
                descriptor.trace_shape.push(lhs_dim);
                let trace_stride = src_block.strides()[lhs_axis]
                    .checked_add(src_block.strides()[rhs_axis])
                    .ok_or(OperationError::ElementCountOverflow)?;
                descriptor
                    .src_trace_strides
                    .push(stride_to_isize(trace_stride)?);
            }

            descriptor.terms.push(TensorTraceDescriptorTerm {
                dst_block: term.dst_block(),
                src_block: term.src_block(),
                output_layout_start,
                trace_layout_start,
                output_rank: axis_plan.output_axes.len(),
                trace_rank: axis_plan.trace_lhs_axes.len(),
                dst_offset: offset_to_isize(dst_block.offset())?,
                src_offset: offset_to_isize(src_block.offset())?,
            });
        }

        Ok(descriptor)
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct TensorTraceDescriptorTerm {
    pub(crate) dst_block: usize,
    pub(crate) src_block: usize,
    pub(crate) output_layout_start: usize,
    pub(crate) trace_layout_start: usize,
    pub(crate) output_rank: usize,
    pub(crate) trace_rank: usize,
    pub(crate) dst_offset: isize,
    pub(crate) src_offset: isize,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct TensorTraceAxisPlan {
    output_axes: Vec<usize>,
    trace_lhs_axes: Vec<usize>,
    trace_rhs_axes: Vec<usize>,
    source_conjugate: bool,
}

impl TensorTraceAxisPlan {
    fn compile(
        src_rank: usize,
        dst_rank: usize,
        axes: TensorTraceAxisSpec<'_>,
    ) -> Result<Self, OperationError> {
        if axes.trace_lhs_axes().len() != axes.trace_rhs_axes().len() {
            return Err(OperationError::TraceAxisCountMismatch {
                lhs: axes.trace_lhs_axes().len(),
                rhs: axes.trace_rhs_axes().len(),
            });
        }
        if axes.output_axes().len() != dst_rank {
            return Err(OperationError::RankMismatch {
                expected: dst_rank,
                actual: axes.output_axes().len(),
            });
        }

        let mut seen = vec![false; src_rank];
        mark_axes("trace output", axes.output_axes(), src_rank, &mut seen)?;
        mark_axes("trace lhs", axes.trace_lhs_axes(), src_rank, &mut seen)?;
        mark_axes("trace rhs", axes.trace_rhs_axes(), src_rank, &mut seen)?;
        if seen.iter().any(|&axis_seen| !axis_seen) {
            let mut all_axes = Vec::with_capacity(
                axes.output_axes().len()
                    + axes.trace_lhs_axes().len()
                    + axes.trace_rhs_axes().len(),
            );
            all_axes.extend_from_slice(axes.output_axes());
            all_axes.extend_from_slice(axes.trace_lhs_axes());
            all_axes.extend_from_slice(axes.trace_rhs_axes());
            return Err(OperationError::InvalidAxisSet {
                tensor: "trace source",
                axes: all_axes,
                rank: src_rank,
            });
        }

        Ok(Self {
            output_axes: axes.output_axes().to_vec(),
            trace_lhs_axes: axes.trace_lhs_axes().to_vec(),
            trace_rhs_axes: axes.trace_rhs_axes().to_vec(),
            source_conjugate: axes.source_conjugate(),
        })
    }
}

fn build_fusion_trace_terms<R>(
    rule: &R,
    dst_structure: &BlockStructure,
    src_structure: &BlockStructure,
    axis_plan: &TensorTraceAxisPlan,
    dst_codomain_rank: usize,
) -> Result<Vec<TensorTraceFusionStructureTerm<R::Scalar>>, OperationError>
where
    R: MultiplicityFreeRigidSymbols,
    R::Scalar: Clone + Add<Output = R::Scalar> + Mul<Output = R::Scalar> + Zero,
{
    let mut codomain_permutation =
        Vec::with_capacity(dst_codomain_rank + axis_plan.trace_lhs_axes.len());
    codomain_permutation.extend_from_slice(&axis_plan.output_axes[..dst_codomain_rank]);
    codomain_permutation.extend_from_slice(&axis_plan.trace_lhs_axes);
    let mut domain_permutation = Vec::with_capacity(
        axis_plan.output_axes.len() - dst_codomain_rank + axis_plan.trace_rhs_axes.len(),
    );
    domain_permutation.extend_from_slice(&axis_plan.output_axes[dst_codomain_rank..]);
    domain_permutation.extend_from_slice(&axis_plan.trace_rhs_axes);

    let mut terms = Vec::new();
    for src_block_index in 0..src_structure.block_count() {
        let src_block = src_structure.block(src_block_index)?;
        let BlockKey::FusionTree(src_key) = src_block.key() else {
            return Err(OperationError::ExpectedFusionTreeBlock {
                tensor: "src",
                index: src_block_index,
            });
        };
        for (permuted_key, permutation_coefficient) in multiplicity_free_permute_tree_pair(
            rule,
            src_key,
            &codomain_permutation,
            &domain_permutation,
        )
        .map_err(OperationError::from_core_preserving_context)?
        {
            let (dst_codomain_tree, trace_codomain_tree) =
                split_fusion_tree(rule, permuted_key.codomain_tree(), dst_codomain_rank)
                    .map_err(OperationError::from_core_preserving_context)?;
            let (dst_domain_tree, trace_domain_tree) = split_fusion_tree(
                rule,
                permuted_key.domain_tree(),
                axis_plan.output_axes.len() - dst_codomain_rank,
            )
            .map_err(OperationError::from_core_preserving_context)?;
            if trace_codomain_tree != trace_domain_tree {
                continue;
            }

            let trace_factor = trace_channel_factor(rule, &trace_codomain_tree)
                .map_err(OperationError::from_core_preserving_context)?;
            let coefficient = permutation_coefficient * trace_factor;
            let dst_key = FusionTreeBlockKey::pair(dst_codomain_tree, dst_domain_tree);
            let dst_block = dst_structure
                .find_block_index_by_fusion_tree_key(&dst_key)
                .ok_or_else(|| OperationError::MissingBlockKey {
                    key: BlockKey::from(dst_key.clone()),
                })?;
            terms.push(TensorTraceFusionStructureTerm {
                dst_key,
                src_key: src_key.clone(),
                dst_block,
                src_block: src_block_index,
                coefficient,
            });
        }
    }
    Ok(terms)
}

fn trace_channel_factor<R>(
    rule: &R,
    trace_tree: &FusionTreeKey,
) -> Result<R::Scalar, tenet_core::CoreError>
where
    R: MultiplicityFreeRigidSymbols,
    R::Scalar: Mul<Output = R::Scalar>,
{
    let coupled = trace_tree.coupled().unwrap_or_else(|| rule.vacuum());
    let first = trace_tree.uncoupled().first().copied().ok_or(
        tenet_core::CoreError::MalformedFusionTree {
            message: "trace channel requires at least one uncoupled sector",
        },
    )?;
    let mut factor = rule.dim_scalar(coupled) * rule.inv_dim_scalar(first);
    for (&sector, &is_dual) in trace_tree
        .uncoupled()
        .iter()
        .zip(trace_tree.is_dual())
        .skip(1)
    {
        if !is_dual {
            factor = factor * rule.twist_scalar(sector);
        }
    }
    Ok(factor)
}

fn validate_fusion_trace_homspace<R>(
    rule: &R,
    dst: &FusionTreeHomSpace,
    src: &FusionTreeHomSpace,
    axis_plan: &TensorTraceAxisPlan,
    dst_codomain_rank: usize,
) -> Result<(), OperationError>
where
    R: FusionRule,
{
    let dst_domain_rank = axis_plan.output_axes.len() - dst_codomain_rank;
    let expected = src
        .select(
            rule,
            &axis_plan.output_axes[..dst_codomain_rank],
            &axis_plan.output_axes[dst_codomain_rank..],
        )
        .map_err(OperationError::from_core_preserving_context)?;
    if expected != *dst {
        return Err(OperationError::StructureMismatch { tensor: "dst" });
    }
    if dst.domain().len() != dst_domain_rank {
        return Err(OperationError::StructureRankMismatch {
            expected: dst_domain_rank,
            actual: dst.domain().len(),
        });
    }
    for (&lhs_axis, &rhs_axis) in axis_plan
        .trace_lhs_axes
        .iter()
        .zip(axis_plan.trace_rhs_axes.iter())
    {
        let lhs = outward_axis_leg(rule, src, lhs_axis)?;
        let rhs = outward_axis_leg(rule, src, rhs_axis)?;
        if lhs != dual_sector_leg(rule, &rhs) {
            return Err(OperationError::StructureMismatch {
                tensor: "trace axes",
            });
        }
    }
    Ok(())
}

fn outward_axis_leg<R>(
    rule: &R,
    homspace: &FusionTreeHomSpace,
    axis: usize,
) -> Result<SectorLeg, OperationError>
where
    R: FusionRule,
{
    if axis < homspace.codomain().len() {
        Ok(homspace.codomain().legs()[axis].clone())
    } else if axis < homspace.rank() {
        Ok(dual_sector_leg(
            rule,
            &homspace.domain().legs()[axis - homspace.codomain().len()],
        ))
    } else {
        Err(OperationError::InvalidAxisSet {
            tensor: "trace source",
            axes: vec![axis],
            rank: homspace.rank(),
        })
    }
}

fn dual_sector_leg<R>(rule: &R, leg: &SectorLeg) -> SectorLeg
where
    R: FusionRule,
{
    leg.dual(rule)
}

pub(crate) fn tensortrace_structure_with_strided_kernel<
    T,
    const DST_NOUT: usize,
    const DST_NIN: usize,
    const SRC_NOUT: usize,
    const SRC_NIN: usize,
    SDst,
    SSrc,
    DDst,
    DSrc,
>(
    _allocator: &mut crate::HostAllocator,
    structure: &TensorTraceStructure,
    dst: &mut TensorMap<T, DST_NOUT, DST_NIN, SDst, DDst>,
    src: &TensorMap<T, SRC_NOUT, SRC_NIN, SSrc, DSrc>,
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
        + ConjugateValue
        + strided_kernel::MaybeSendSync,
    DDst: HostWritableStorage<T>,
    DSrc: HostReadableStorage<T>,
{
    let dst_structure = Arc::clone(dst.structure());
    let src_structure = Arc::clone(src.structure());
    structure.validate_replay_structures(&dst_structure, &src_structure)?;
    let descriptor = structure.descriptor();
    for term in descriptor.terms() {
        tensortrace_raw_strided_kernel(
            dst.data_mut(),
            src.data(),
            descriptor.output_shape(term),
            descriptor.trace_shape(term),
            descriptor.dst_strides(term),
            descriptor.src_output_strides(term),
            descriptor.src_trace_strides(term),
            term.dst_offset,
            term.src_offset,
            descriptor.source_conjugate(),
            alpha,
            beta,
        )?;
    }
    Ok(())
}

pub(crate) fn tensortrace_fusion_structure_with_strided_kernel<
    T,
    C,
    const DST_NOUT: usize,
    const DST_NIN: usize,
    const SRC_NOUT: usize,
    const SRC_NIN: usize,
    SDst,
    SSrc,
    DDst,
    DSrc,
>(
    _allocator: &mut crate::HostAllocator,
    structure: &TensorTraceFusionStructure<C>,
    dst: &mut TensorMap<T, DST_NOUT, DST_NIN, SDst, DDst>,
    src: &TensorMap<T, SRC_NOUT, SRC_NIN, SSrc, DSrc>,
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
        + ConjugateValue
        + RecouplingCoefficientAction<C>
        + strided_kernel::MaybeSendSync,
    C: Copy,
    DDst: HostWritableStorage<T>,
    DSrc: HostReadableStorage<T>,
{
    let dst_structure = Arc::clone(dst.structure());
    let src_structure = Arc::clone(src.structure());
    structure.validate_replay_structures(&dst_structure, &src_structure)?;
    scale_trace_destination(dst.data_mut(), beta);
    let descriptor = structure.descriptor();
    for (term, fusion_term) in descriptor.terms().iter().zip(structure.terms()) {
        tensortrace_raw_strided_kernel_add_with_coefficient(
            dst.data_mut(),
            src.data(),
            descriptor.output_shape(term),
            descriptor.trace_shape(term),
            descriptor.dst_strides(term),
            descriptor.src_output_strides(term),
            descriptor.src_trace_strides(term),
            term.dst_offset,
            term.src_offset,
            descriptor.source_conjugate(),
            alpha,
            fusion_term.coefficient,
        )?;
    }
    Ok(())
}

/// Dynamic-rank fusion tensortrace: partial (or full) trace of `src` over
/// the `axes` trace pairs into caller-allocated `dst_data`
/// (`dst = beta * dst + alpha * trace(src)`), operating on
/// provider-bound dynamic spaces plus raw coupled-layout slices —
/// the dynamic analog of [`crate::tensortrace_fusion_into`], sharing the
/// same term compilation (TensorKit `tensortrace!` semantics: quantum
/// dimension factors and twists, i.e. the fermionic supertrace).
#[allow(clippy::too_many_arguments)]
pub fn tensortrace_fusion_dyn_into<R, D>(
    dst_space: &BoundDynamicFusionMapSpace<R>,
    dst_data: &mut [D],
    src_space: &BoundDynamicFusionMapSpace<R>,
    src_data: &[D],
    axes: TensorTraceAxisSpec<'_>,
    alpha: D,
    beta: D,
) -> Result<(), OperationError>
where
    R: MultiplicityFreeRigidSymbols,
    R::Scalar:
        Copy + Add<Output = R::Scalar> + Mul<Output = R::Scalar> + Zero + RealStructuralCoefficient,
    D: Copy
        + Add<D, Output = D>
        + Mul<D, Output = D>
        + PartialEq
        + Zero
        + One
        + ConjugateValue
        + RecouplingCoefficientAction<R::Scalar>
        + strided_kernel::MaybeSendSync,
{
    tensortrace_fusion_dyn_into_raw(
        src_space.provider(),
        dst_space.space(),
        dst_data,
        src_space.space(),
        src_data,
        axes,
        alpha,
        beta,
    )
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn tensortrace_fusion_dyn_into_raw<R, D>(
    rule: &R,
    dst_space: &DynamicFusionMapSpace,
    dst_data: &mut [D],
    src_space: &DynamicFusionMapSpace,
    src_data: &[D],
    axes: TensorTraceAxisSpec<'_>,
    alpha: D,
    beta: D,
) -> Result<(), OperationError>
where
    R: MultiplicityFreeRigidSymbols,
    R::Scalar:
        Copy + Add<Output = R::Scalar> + Mul<Output = R::Scalar> + Zero + RealStructuralCoefficient,
    D: Copy
        + Add<D, Output = D>
        + Mul<D, Output = D>
        + PartialEq
        + Zero
        + One
        + ConjugateValue
        + RecouplingCoefficientAction<R::Scalar>
        + strided_kernel::MaybeSendSync,
{
    let structure =
        TensorTraceFusionStructure::compile_fusion_dyn_raw(rule, dst_space, src_space, axes)?;
    scale_trace_destination(dst_data, beta);
    let descriptor = structure.descriptor();
    for (term, fusion_term) in descriptor.terms().iter().zip(structure.terms()) {
        tensortrace_raw_strided_kernel_add_with_coefficient(
            dst_data,
            src_data,
            descriptor.output_shape(term),
            descriptor.trace_shape(term),
            descriptor.dst_strides(term),
            descriptor.src_output_strides(term),
            descriptor.src_trace_strides(term),
            term.dst_offset,
            term.src_offset,
            descriptor.source_conjugate(),
            alpha,
            fusion_term.coefficient,
        )?;
    }
    Ok(())
}

fn scale_trace_destination<T>(dst: &mut [T], beta: T)
where
    T: Copy + Mul<T, Output = T> + PartialEq + Zero + One,
{
    if beta.is_one() {
        return;
    }
    if beta.is_zero() {
        dst.fill(T::zero());
    } else {
        for value in dst {
            *value = *value * beta;
        }
    }
}

fn mark_axes(
    tensor: &'static str,
    axes: &[usize],
    rank: usize,
    seen: &mut [bool],
) -> Result<(), OperationError> {
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
    Ok(())
}

fn stride_to_isize(stride: usize) -> Result<isize, OperationError> {
    isize::try_from(stride).map_err(|_| OperationError::StrideOverflow { value: stride })
}
