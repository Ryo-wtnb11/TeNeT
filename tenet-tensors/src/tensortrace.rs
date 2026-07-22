use core::ops::{Add, Mul};
use std::sync::Arc;

use num_traits::{One, Zero};
use tenet_core::{
    multiplicity_free_permute_tree_pair_block_indexed, split_fusion_tree, BlockKey, BlockStructure,
    CheckedFusionAlgebra, CheckedFusionSpaceError, FusionRule, FusionStyleKind,
    FusionTensorMapSpace, FusionTreeHomSpace, FusionTreeKey, FusionTreePairKey,
    FusionTreePairOrientation, HostReadableStorage, HostWritableStorage,
    MultiplicityFreeRigidSymbols, OrientedFusionTreeHomSpace, SectorLeg, TensorMap, TensorStorage,
};

use crate::contract::{BoundDynamicFusionMapSpace, DynamicFusionMapSpace};
use crate::lowering::{
    lower_tensortrace_source_adjoint_axes, lower_tensortrace_source_adjoint_axes_dyn,
};
use crate::strided::offset_to_isize;
use crate::{tensortrace_raw_strided_kernel, tensortrace_raw_strided_kernel_add_with_coefficient};
use tenet_operations::structure_identity::validate_structure_identity;
use tenet_operations::transform_structure::validate_destination_layouts_injective;
use tenet_operations::TensorTraceAxisSpec;
use tenet_operations::{axpby_raw_strided_kernel_trusted, scale_raw_strided_kernel_trusted};
use tenet_operations::{try_tensortrace_owned_raw, OperationError, OwnedTraceTerm};
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

/// Compiles a fusion-aware trace into reusable structural terms.
///
/// Valid terms and their source accumulation follow global source-block order.
/// For expert layouts built from an incomplete explicit block structure,
/// errors between distinct external-sector groups likewise follow the first
/// source block encountered globally. On the non-`Unique` grouped path
/// (currently `Simple` fusion), one external-sector group is transformed
/// atomically, so a symbol or tree-transform error from a later member of that
/// group can precede an earlier member's destination
/// [`OperationError::MissingBlockKey`].
///
/// This uses the same block-level lowering granularity as TensorKit and QSpace;
/// those references do not establish identical malformed-layout error
/// ordering. Why not recover per-source error precedence within a group: doing
/// so would replay the same F/R traversal for every source tree.
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
        let adjoint_axes = lower_tensortrace_source_adjoint_axes::<SRC_NOUT, SRC_NIN>(axes)?;
        TensorTraceFusionStructure::compile_fusion_spaces_oriented(
            rule,
            dst_fusion,
            src_fusion,
            FusionTreePairOrientation::Adjoint,
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

#[derive(Clone, Copy)]
struct OrientedTraceSource<'a> {
    homspace: OrientedFusionTreeHomSpace<'a>,
    structure: &'a Arc<BlockStructure>,
    orientation: FusionTreePairOrientation,
    storage_nout: usize,
    storage_nin: usize,
}

impl<'a> OrientedTraceSource<'a> {
    fn new(
        homspace: &'a FusionTreeHomSpace,
        structure: &'a Arc<BlockStructure>,
        storage_nout: usize,
        storage_nin: usize,
        orientation: FusionTreePairOrientation,
    ) -> Self {
        Self {
            homspace: OrientedFusionTreeHomSpace::new(homspace, orientation),
            structure,
            orientation,
            storage_nout,
            storage_nin,
        }
    }

    fn logical_axis_to_storage(self, axis: usize) -> usize {
        match self.orientation {
            FusionTreePairOrientation::Direct => axis,
            FusionTreePairOrientation::Adjoint => {
                if axis < self.storage_nin {
                    self.storage_nout + axis
                } else {
                    axis - self.storage_nin
                }
            }
        }
    }

    fn source_key(self, index: usize) -> Result<FusionTreePairKey, OperationError> {
        let block = self.structure.block(index)?;
        let BlockKey::FusionTree(key) = block.key() else {
            return Err(OperationError::ExpectedFusionTreeBlock {
                tensor: "src",
                index,
            });
        };
        Ok(match self.orientation {
            FusionTreePairOrientation::Direct => key.clone(),
            FusionTreePairOrientation::Adjoint => {
                FusionTreePairKey::pair(key.domain_tree().clone(), key.codomain_tree().clone())
            }
        })
    }

    fn validate_source_keys(self) -> Result<(), OperationError> {
        for index in 0..self.structure.block_count() {
            let block = self.structure.block(index)?;
            if !matches!(block.key(), BlockKey::FusionTree(_)) {
                return Err(OperationError::ExpectedFusionTreeBlock {
                    tensor: "src",
                    index,
                });
            }
        }
        Ok(())
    }
}

#[cfg(test)]
thread_local! {
    static TRACE_TRANSFORM_SOURCES: std::cell::RefCell<Vec<usize>> = const {
        std::cell::RefCell::new(Vec::new())
    };
    static TRACE_COMPILER_GEOMETRY_DERIVATIONS: std::cell::Cell<(usize, usize)> = const {
        std::cell::Cell::new((0, 0))
    };
}

#[cfg(test)]
pub(crate) fn reset_trace_transform_invocations() {
    TRACE_TRANSFORM_SOURCES.with_borrow_mut(Vec::clear);
}

#[cfg(test)]
pub(crate) fn take_trace_transform_invocations() -> usize {
    take_trace_transform_sources().len()
}

#[cfg(test)]
pub(crate) fn take_trace_transform_sources() -> Vec<usize> {
    TRACE_TRANSFORM_SOURCES.take()
}

#[cfg(test)]
pub(crate) fn take_trace_compiler_geometry_derivations() -> (usize, usize) {
    TRACE_COMPILER_GEOMETRY_DERIVATIONS.replace((0, 0))
}

#[inline]
fn record_trace_transform_invocation(_src_block_index: usize) {
    #[cfg(test)]
    TRACE_TRANSFORM_SOURCES.with_borrow_mut(|sources| sources.push(_src_block_index));
}

#[inline]
fn record_trace_axis_plan_derivation() {
    #[cfg(test)]
    TRACE_COMPILER_GEOMETRY_DERIVATIONS.with(|counts| {
        let (axis_plans, selected_homspaces) = counts.get();
        counts.set((axis_plans.saturating_add(1), selected_homspaces));
    });
}

#[inline]
fn record_trace_selected_homspace_derivation() {
    #[cfg(test)]
    TRACE_COMPILER_GEOMETRY_DERIVATIONS.with(|counts| {
        let (axis_plans, selected_homspaces) = counts.get();
        counts.set((axis_plans, selected_homspaces.saturating_add(1)));
    });
}

/// A compiled fusion-aware trace whose valid terms retain global source order.
///
/// See [`tensortrace_fusion_structure`] for the non-`Unique` grouped path's
/// block-atomic error-ordering contract for expert incomplete layouts.
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
        Self::compile_fusion_spaces_oriented(
            rule,
            dst,
            src,
            FusionTreePairOrientation::Direct,
            axes,
        )
    }

    fn compile_fusion_spaces_oriented<
        R,
        const DST_NOUT: usize,
        const DST_NIN: usize,
        const SRC_NOUT: usize,
        const SRC_NIN: usize,
    >(
        rule: &R,
        dst: &FusionTensorMapSpace<DST_NOUT, DST_NIN>,
        src: &FusionTensorMapSpace<SRC_NOUT, SRC_NIN>,
        orientation: FusionTreePairOrientation,
        axes: TensorTraceAxisSpec<'_>,
    ) -> Result<Self, OperationError>
    where
        R: MultiplicityFreeRigidSymbols<Scalar = C>,
        C: Clone + Add<Output = C> + Mul<Output = C> + Zero + RealStructuralCoefficient,
    {
        dst.validate_rule(rule)?;
        src.validate_rule(rule)?;
        let source = OrientedTraceSource::new(
            src.homspace(),
            src.subblock_structure(),
            SRC_NOUT,
            SRC_NIN,
            orientation,
        );
        Self::compile_fusion_parts(
            rule,
            dst.homspace(),
            Arc::clone(dst.subblock_structure()),
            source,
            DST_NOUT,
            axes,
        )
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
        Self::compile_fusion_dyn_raw_with_preflight(rule, dst, src, axes, |_, _| Ok(None))
    }

    fn compile_fusion_dyn_checked_raw<R>(
        rule: &R,
        dst: &DynamicFusionMapSpace,
        src: &DynamicFusionMapSpace,
        axes: TensorTraceAxisSpec<'_>,
    ) -> Result<Self, OperationError>
    where
        R: MultiplicityFreeRigidSymbols<Scalar = C> + CheckedFusionAlgebra,
        C: Clone + Add<Output = C> + Mul<Output = C> + Zero + RealStructuralCoefficient,
    {
        Self::compile_fusion_dyn_raw_with_preflight(
            rule,
            dst,
            src,
            axes,
            |logical_src, axis_plan| {
                checked_fusion_trace_geometry(rule, logical_src, axis_plan, dst.nout()).map(Some)
            },
        )
    }

    fn compile_fusion_dyn_raw_with_preflight<R, F>(
        rule: &R,
        dst: &DynamicFusionMapSpace,
        src: &DynamicFusionMapSpace,
        axes: TensorTraceAxisSpec<'_>,
        preflight: F,
    ) -> Result<Self, OperationError>
    where
        R: MultiplicityFreeRigidSymbols<Scalar = C>,
        C: Clone + Add<Output = C> + Mul<Output = C> + Zero + RealStructuralCoefficient,
        F: FnOnce(
            OrientedFusionTreeHomSpace<'_>,
            &TensorTraceAxisPlan,
        ) -> Result<Option<CheckedTraceGeometry>, OperationError>,
    {
        dst.validate_rule(rule)?;
        src.validate_rule(rule)?;
        let orientation = if axes.source_conjugate() {
            FusionTreePairOrientation::Adjoint
        } else {
            FusionTreePairOrientation::Direct
        };
        let lowered_axes = lower_tensortrace_source_adjoint_axes_dyn(src.nout(), src.nin(), axes)?;
        let source = OrientedTraceSource::new(
            src.homspace(),
            src.structure(),
            src.nout(),
            src.nin(),
            orientation,
        );
        Self::compile_fusion_parts_with_preflight(
            rule,
            dst.homspace(),
            Arc::clone(dst.structure()),
            source,
            dst.nout(),
            lowered_axes.as_spec(),
            preflight,
        )
    }

    /// Rank-runtime core shared by the const-generic and dynamic compiles.
    #[allow(clippy::too_many_arguments)]
    fn compile_fusion_parts<R>(
        rule: &R,
        dst_homspace: &FusionTreeHomSpace,
        dst_structure: Arc<BlockStructure>,
        src: OrientedTraceSource<'_>,
        dst_codomain_rank: usize,
        axes: TensorTraceAxisSpec<'_>,
    ) -> Result<Self, OperationError>
    where
        R: MultiplicityFreeRigidSymbols<Scalar = C>,
        C: Clone + Add<Output = C> + Mul<Output = C> + Zero + RealStructuralCoefficient,
    {
        Self::compile_fusion_parts_with_preflight(
            rule,
            dst_homspace,
            dst_structure,
            src,
            dst_codomain_rank,
            axes,
            |_, _| Ok(None),
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn compile_fusion_parts_with_preflight<R, F>(
        rule: &R,
        dst_homspace: &FusionTreeHomSpace,
        dst_structure: Arc<BlockStructure>,
        src: OrientedTraceSource<'_>,
        dst_codomain_rank: usize,
        axes: TensorTraceAxisSpec<'_>,
        preflight: F,
    ) -> Result<Self, OperationError>
    where
        R: MultiplicityFreeRigidSymbols<Scalar = C>,
        C: Clone + Add<Output = C> + Mul<Output = C> + Zero + RealStructuralCoefficient,
        F: FnOnce(
            OrientedFusionTreeHomSpace<'_>,
            &TensorTraceAxisPlan,
        ) -> Result<Option<CheckedTraceGeometry>, OperationError>,
    {
        if !rule.braiding_style().is_symmetric() {
            return Err(OperationError::UnsupportedTensorContractScope {
                message: FUSION_TENSORTRACE_REQUIRES_SYMMETRIC_BRAIDING,
            });
        }
        let axis_plan =
            TensorTraceAxisPlan::compile(src.structure.rank(), dst_structure.rank(), axes)?;
        let checked_geometry = preflight(src.homspace, &axis_plan)?;
        // Why not cache this per-call result: eager compilation only needs to
        // pass the checked selection forward instead of deriving it again.
        validate_fusion_trace_homspace(
            rule,
            dst_homspace,
            src.homspace,
            &axis_plan,
            dst_codomain_rank,
            checked_geometry,
        )?;
        let terms =
            build_fusion_trace_terms(rule, &dst_structure, src, &axis_plan, dst_codomain_rank)?;
        let dense_terms = terms
            .iter()
            .map(|term| TensorTraceStructureTerm {
                key: BlockKey::from(term.dst_key.clone()),
                dst_block: term.dst_block,
                src_block: term.src_block,
            })
            .collect::<Vec<_>>();
        let descriptor =
            TensorTraceDescriptor::compile_oriented(&axis_plan, &dense_terms, &dst_structure, src)?;
        validate_destination_layouts_injective(
            &dst_structure,
            "tensor trace destination layouts overlap",
        )?;

        Ok(Self {
            dst_rank: dst_structure.rank(),
            src_rank: src.structure.rank(),
            output_axes: axis_plan.output_axes,
            trace_lhs_axes: axis_plan.trace_lhs_axes,
            trace_rhs_axes: axis_plan.trace_rhs_axes,
            terms,
            descriptor,
            dst_structure,
            src_structure: Arc::clone(src.structure),
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
    dst_key: FusionTreePairKey,
    src_key: FusionTreePairKey,
    dst_block: usize,
    src_block: usize,
    coefficient: C,
}

impl<C> TensorTraceFusionStructureTerm<C> {
    #[inline]
    pub fn dst_key(&self) -> &FusionTreePairKey {
        &self.dst_key
    }

    #[inline]
    pub fn src_key(&self) -> &FusionTreePairKey {
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
        validate_destination_layouts_injective(
            &dst_structure,
            "tensor trace destination layouts overlap",
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
    destination_layouts: Vec<TensorTraceDestinationLayout>,
    destination_shape: Vec<usize>,
    destination_strides: Vec<isize>,
    destination_zero_strides: Vec<isize>,
    output_shape: Vec<usize>,
    trace_shape: Vec<usize>,
    dst_strides: Vec<isize>,
    src_output_strides: Vec<isize>,
    src_trace_strides: Vec<isize>,
    destination_producers: Vec<usize>,
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

    #[inline]
    fn destination_layouts(&self) -> &[TensorTraceDestinationLayout] {
        &self.destination_layouts
    }

    fn destination_shape(&self, layout: &TensorTraceDestinationLayout) -> &[usize] {
        &self.destination_shape[layout.layout_start..layout.layout_start + layout.rank]
    }

    fn destination_strides(&self, layout: &TensorTraceDestinationLayout) -> &[isize] {
        &self.destination_strides[layout.layout_start..layout.layout_start + layout.rank]
    }

    fn destination_zero_strides(&self, layout: &TensorTraceDestinationLayout) -> &[isize] {
        &self.destination_zero_strides[..layout.rank]
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

    fn destination_producer_offsets(&self) -> &[usize] {
        &self.destination_producers[..self.destination_layouts.len() + 1]
    }

    fn destination_producer_indices(&self) -> &[usize] {
        &self.destination_producers[self.destination_layouts.len() + 1..]
    }

    fn compile(
        axis_plan: &TensorTraceAxisPlan,
        terms: &[TensorTraceStructureTerm],
        dst_structure: &BlockStructure,
        src_structure: &BlockStructure,
    ) -> Result<Self, OperationError> {
        Self::compile_with_source_axis_map(axis_plan, terms, dst_structure, src_structure, |axis| {
            axis
        })
    }

    fn compile_oriented(
        axis_plan: &TensorTraceAxisPlan,
        terms: &[TensorTraceStructureTerm],
        dst_structure: &BlockStructure,
        src: OrientedTraceSource<'_>,
    ) -> Result<Self, OperationError> {
        Self::compile_with_source_axis_map(axis_plan, terms, dst_structure, src.structure, |axis| {
            src.logical_axis_to_storage(axis)
        })
    }

    fn compile_with_source_axis_map<F>(
        axis_plan: &TensorTraceAxisPlan,
        terms: &[TensorTraceStructureTerm],
        dst_structure: &BlockStructure,
        src_structure: &BlockStructure,
        source_axis: F,
    ) -> Result<Self, OperationError>
    where
        F: Fn(usize) -> usize,
    {
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
        descriptor
            .destination_layouts
            .reserve(dst_structure.block_count());
        descriptor
            .destination_zero_strides
            .resize(dst_structure.rank(), 0);

        for term in terms {
            let dst_block = dst_structure.block(term.dst_block())?;
            let src_block = src_structure.block(term.src_block())?;
            let output_layout_start = descriptor.output_shape.len();
            for (dst_axis, &src_axis) in axis_plan.output_axes.iter().enumerate() {
                let src_axis = source_axis(src_axis);
                let dst_dim = dst_block.shape()[dst_axis];
                let src_dim = src_block.shape()[src_axis];
                if dst_dim != src_dim {
                    let src_shape = axis_plan
                        .output_axes
                        .iter()
                        .map(|&axis| src_block.shape()[source_axis(axis)])
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
                let lhs_axis = source_axis(lhs_axis);
                let rhs_axis = source_axis(rhs_axis);
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

        for dst_block in 0..dst_structure.block_count() {
            let block = dst_structure.block(dst_block)?;
            let layout_start = descriptor.destination_shape.len();
            descriptor
                .destination_shape
                .extend_from_slice(block.shape());
            for &stride in block.strides() {
                descriptor
                    .destination_strides
                    .push(stride_to_isize(stride)?);
            }
            descriptor
                .destination_layouts
                .push(TensorTraceDestinationLayout {
                    layout_start,
                    rank: block.shape().len(),
                    offset: offset_to_isize(block.offset())?,
                });
        }
        descriptor.compile_destination_producer_partition(dst_structure.block_count())?;

        Ok(descriptor)
    }

    fn compile_destination_producer_partition(
        &mut self,
        block_count: usize,
    ) -> Result<(), OperationError> {
        let offsets_len = block_count
            .checked_add(1)
            .ok_or(OperationError::ElementCountOverflow)?;
        let total_len = offsets_len
            .checked_add(self.terms.len())
            .ok_or(OperationError::ElementCountOverflow)?;
        self.destination_producers.resize(total_len, 0);
        let (offsets, indices) = self.destination_producers.split_at_mut(offsets_len);
        for term in &self.terms {
            let end = offsets.get_mut(term.dst_block + 1).ok_or(
                OperationError::BlockIndexOutOfBounds {
                    tensor: "trace destination",
                    index: term.dst_block,
                    count: block_count,
                },
            )?;
            *end = end
                .checked_add(1)
                .ok_or(OperationError::ElementCountOverflow)?;
        }

        for block in 0..block_count {
            offsets[block + 1] = offsets[block]
                .checked_add(offsets[block + 1])
                .ok_or(OperationError::ElementCountOverflow)?;
        }
        for (term_index, term) in self.terms.iter().enumerate().rev() {
            let cursor = &mut offsets[term.dst_block + 1];
            *cursor = cursor
                .checked_sub(1)
                .ok_or(OperationError::ElementCountOverflow)?;
            indices[*cursor] = term_index;
        }
        for block in 0..block_count {
            offsets[block] = offsets[block + 1];
        }
        offsets[block_count] = self.terms.len();
        Ok(())
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct TensorTraceDestinationLayout {
    layout_start: usize,
    rank: usize,
    offset: isize,
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
        record_trace_axis_plan_derivation();
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

struct CheckedTraceGeometry {
    selected_homspace: FusionTreeHomSpace,
    trace_pairs_match: bool,
}

fn build_fusion_trace_terms<R>(
    rule: &R,
    dst_structure: &BlockStructure,
    src: OrientedTraceSource<'_>,
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

    src.validate_source_keys()?;
    let mut rows_by_source = (0..src.structure.block_count())
        .map(|_| None)
        .collect::<Vec<Option<Vec<(FusionTreePairKey, R::Scalar)>>>>();
    let is_unique = rule.fusion_style() == FusionStyleKind::Unique;
    let groups = if is_unique {
        &[][..]
    } else {
        src.structure.fusion_tree_group_slice()
    };
    let mut group_by_source = if is_unique {
        Vec::new()
    } else {
        vec![None; src.structure.block_count()]
    };
    if !is_unique {
        for (group_index, group) in groups.iter().enumerate() {
            for &src_block_index in group.block_indices() {
                group_by_source[src_block_index] = Some(group_index);
            }
        }
    }

    let mut terms = Vec::new();
    for src_block_index in 0..src.structure.block_count() {
        if is_unique {
            record_trace_transform_invocation(src_block_index);
            let source_indices = [src_block_index];
            let mut rows = multiplicity_free_permute_tree_pair_block_indexed(
                rule,
                src.structure,
                &source_indices,
                src.orientation,
                &codomain_permutation,
                &domain_permutation,
            )
            .map_err(OperationError::from_core_preserving_context)?;
            rows_by_source[src_block_index] = rows.pop();
        } else if rows_by_source[src_block_index].is_none() {
            let group_index =
                group_by_source[src_block_index].ok_or(OperationError::InvalidArgument {
                    message: "trace source block was not assigned to a fusion group",
                })?;
            let group = &groups[group_index];
            record_trace_transform_invocation(src_block_index);
            let group_rows = multiplicity_free_permute_tree_pair_block_indexed(
                rule,
                src.structure,
                group.block_indices(),
                src.orientation,
                &codomain_permutation,
                &domain_permutation,
            )
            .map_err(OperationError::from_core_preserving_context)?;
            if group_rows.len() != group.block_indices().len() {
                return Err(OperationError::InvalidArgument {
                    message: "trace block transform returned the wrong source row count",
                });
            }
            for (&src_block_index, rows) in group.block_indices().iter().zip(group_rows) {
                rows_by_source[src_block_index] = Some(rows);
            }
        }

        // Why not transform every group up front: a later group's symbol error
        // must not overtake an earlier source's lowering error. The whole current
        // group stays atomic because per-source replay would duplicate its F/R
        // traversal; expert incomplete structures can therefore observe changed
        // error timing only between members of that same group.
        let rows =
            rows_by_source[src_block_index]
                .take()
                .ok_or(OperationError::InvalidArgument {
                    message: "trace source block was not assigned to a fusion group",
                })?;
        lower_fusion_trace_source_rows(
            rule,
            dst_structure,
            axis_plan,
            dst_codomain_rank,
            src_block_index,
            src,
            rows,
            &mut terms,
        )?;
    }
    Ok(terms)
}

#[cfg(test)]
pub(crate) fn build_fusion_trace_terms_for_test<R>(
    rule: &R,
    dst_structure: &BlockStructure,
    src_structure: &BlockStructure,
    axes: TensorTraceAxisSpec<'_>,
    dst_codomain_rank: usize,
) -> Result<Vec<TensorTraceFusionStructureTerm<R::Scalar>>, OperationError>
where
    R: MultiplicityFreeRigidSymbols,
    R::Scalar: Clone + Add<Output = R::Scalar> + Mul<Output = R::Scalar> + Zero,
{
    let axis_plan = TensorTraceAxisPlan::compile(src_structure.rank(), dst_structure.rank(), axes)?;
    let homspace = FusionTreeHomSpace::new(
        tenet_core::FusionProductSpace::new([]),
        tenet_core::FusionProductSpace::new([]),
    );
    let src_structure = Arc::new(src_structure.clone());
    let src = OrientedTraceSource::new(
        &homspace,
        &src_structure,
        src_structure.rank(),
        0,
        FusionTreePairOrientation::Direct,
    );
    build_fusion_trace_terms(rule, dst_structure, src, &axis_plan, dst_codomain_rank)
}

#[allow(clippy::too_many_arguments)]
fn lower_fusion_trace_source_rows<R>(
    rule: &R,
    dst_structure: &BlockStructure,
    axis_plan: &TensorTraceAxisPlan,
    dst_codomain_rank: usize,
    src_block_index: usize,
    src: OrientedTraceSource<'_>,
    rows: Vec<(FusionTreePairKey, R::Scalar)>,
    terms: &mut Vec<TensorTraceFusionStructureTerm<R::Scalar>>,
) -> Result<(), OperationError>
where
    R: MultiplicityFreeRigidSymbols,
    R::Scalar: Clone + Mul<Output = R::Scalar>,
{
    for (permuted_key, permutation_coefficient) in rows {
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
        let dst_key = FusionTreePairKey::pair(dst_codomain_tree, dst_domain_tree);
        let dst_block = dst_structure
            .find_block_index_by_fusion_tree_pair(&dst_key)
            .ok_or_else(|| OperationError::MissingBlockKey {
                key: Box::new(BlockKey::from(dst_key.clone())),
            })?;
        terms.push(TensorTraceFusionStructureTerm {
            dst_key,
            src_key: src.source_key(src_block_index)?,
            dst_block,
            src_block: src_block_index,
            coefficient,
        });
    }
    Ok(())
}

fn trace_channel_factor<R>(
    rule: &R,
    trace_tree: &FusionTreeKey,
) -> Result<R::Scalar, tenet_core::CoreError>
where
    R: MultiplicityFreeRigidSymbols,
    R::Scalar: Mul<Output = R::Scalar>,
{
    let coupled = trace_tree.coupled();
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
    src: OrientedFusionTreeHomSpace<'_>,
    axis_plan: &TensorTraceAxisPlan,
    dst_codomain_rank: usize,
    checked_geometry: Option<CheckedTraceGeometry>,
) -> Result<(), OperationError>
where
    R: FusionRule,
{
    let (selected, checked_trace_pairs_match) = match checked_geometry {
        Some(CheckedTraceGeometry {
            selected_homspace,
            trace_pairs_match,
        }) => (selected_homspace, Some(trace_pairs_match)),
        None => {
            record_trace_selected_homspace_derivation();
            (
                src.select(
                    rule,
                    &axis_plan.output_axes[..dst_codomain_rank],
                    &axis_plan.output_axes[dst_codomain_rank..],
                )
                .map_err(OperationError::from_core_preserving_context)?,
                None,
            )
        }
    };
    if selected != *dst {
        return Err(OperationError::StructureMismatch { tensor: "dst" });
    }
    let dst_domain_rank = axis_plan.output_axes.len() - dst_codomain_rank;
    if dst.domain().len() != dst_domain_rank {
        return Err(OperationError::StructureRankMismatch {
            expected: dst_domain_rank,
            actual: dst.domain().len(),
        });
    }
    if let Some(trace_pairs_match) = checked_trace_pairs_match {
        if !trace_pairs_match {
            return Err(OperationError::StructureMismatch {
                tensor: "trace axes",
            });
        }
        return Ok(());
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
    homspace: OrientedFusionTreeHomSpace<'_>,
    axis: usize,
) -> Result<SectorLeg, OperationError>
where
    R: FusionRule,
{
    homspace
        .external_axis_leg(rule, axis)
        .ok_or_else(|| OperationError::InvalidAxisSet {
            tensor: "trace source",
            axes: vec![axis],
            rank: homspace.rank(),
        })
}

fn dual_sector_leg<R>(rule: &R, leg: &SectorLeg) -> SectorLeg
where
    R: FusionRule,
{
    leg.dual(rule)
}

fn outward_axis_leg_checked<R>(
    rule: &R,
    homspace: OrientedFusionTreeHomSpace<'_>,
    axis: usize,
) -> Result<SectorLeg, OperationError>
where
    R: FusionRule + CheckedFusionAlgebra,
{
    homspace
        .try_external_axis_leg(rule, axis)
        .map_err(|error| OperationError::FusionAlgebra(Box::new(error)))?
        .ok_or_else(|| OperationError::InvalidAxisSet {
            tensor: "trace source",
            axes: vec![axis],
            rank: homspace.rank(),
        })
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
    let dst_len = dst.data_mut().len();
    validate_trace_data_extents(&dst_structure, dst_len, &src_structure, src.data().len())?;
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
    let dst_len = dst.data_mut().len();
    validate_trace_data_extents(&dst_structure, dst_len, &src_structure, src.data().len())?;
    let descriptor = structure.descriptor();
    scale_trace_destination_layouts(descriptor, dst.data_mut(), beta)?;
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

/// Checked lowered dynamic trace. Built-in lowered rules should use this
/// entry point when typed machine-boundary errors are required; the legacy
/// wrapper above remains available to custom fusion rules.
#[allow(clippy::too_many_arguments)]
pub fn tensortrace_fusion_dyn_into_checked<R, D>(
    dst_space: &BoundDynamicFusionMapSpace<R>,
    dst_data: &mut [D],
    src_space: &BoundDynamicFusionMapSpace<R>,
    src_data: &[D],
    axes: TensorTraceAxisSpec<'_>,
    alpha: D,
    beta: D,
) -> Result<(), OperationError>
where
    R: MultiplicityFreeRigidSymbols + CheckedFusionAlgebra,
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
    tensortrace_fusion_dyn_into_checked_raw(
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

/// Internal owned-output trace path for built-in host storage.
///
/// This compiles the oriented trace semantics once, then either uses the
/// private-initialization writer or replays that same structure into an
/// initialized destination when canonical physical coverage is unavailable.
#[doc(hidden)]
pub fn tensortrace_fusion_dyn_owned<R, D>(
    dst_space: &BoundDynamicFusionMapSpace<R>,
    src_space: &BoundDynamicFusionMapSpace<R>,
    src_data: &[D],
    axes: TensorTraceAxisSpec<'_>,
    alpha: D,
) -> Result<Vec<D>, OperationError>
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
    let structure = TensorTraceFusionStructure::compile_fusion_dyn_raw(
        src_space.provider(),
        dst_space.space(),
        src_space.space(),
        axes,
    )?;
    tensortrace_fusion_dyn_structure_owned(
        &structure,
        dst_space.space(),
        src_space.space(),
        src_data,
        alpha,
    )
}

/// Checked built-in counterpart of [`tensortrace_fusion_dyn_owned`].
#[doc(hidden)]
pub fn tensortrace_fusion_dyn_owned_checked<R, D>(
    dst_space: &BoundDynamicFusionMapSpace<R>,
    src_space: &BoundDynamicFusionMapSpace<R>,
    src_data: &[D],
    axes: TensorTraceAxisSpec<'_>,
    alpha: D,
) -> Result<Vec<D>, OperationError>
where
    R: MultiplicityFreeRigidSymbols + CheckedFusionAlgebra,
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
    let structure = TensorTraceFusionStructure::compile_fusion_dyn_checked_raw(
        src_space.provider(),
        dst_space.space(),
        src_space.space(),
        axes,
    )?;
    tensortrace_fusion_dyn_structure_owned(
        &structure,
        dst_space.space(),
        src_space.space(),
        src_data,
        alpha,
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
    tensortrace_fusion_dyn_structure_into_raw(
        &structure, dst_space, dst_data, src_space, src_data, alpha, beta,
    )
}

fn tensortrace_fusion_dyn_structure_owned<C, D>(
    structure: &TensorTraceFusionStructure<C>,
    dst_space: &DynamicFusionMapSpace,
    src_space: &DynamicFusionMapSpace,
    src_data: &[D],
    alpha: D,
) -> Result<Vec<D>, OperationError>
where
    C: Copy,
    D: Copy
        + Add<D, Output = D>
        + Mul<D, Output = D>
        + PartialEq
        + Zero
        + One
        + ConjugateValue
        + RecouplingCoefficientAction<C>
        + strided_kernel::MaybeSendSync,
{
    let descriptor = structure.descriptor();
    if descriptor.terms().len() != structure.terms().len() {
        return Err(OperationError::CoefficientCountMismatch {
            expected: descriptor.terms().len(),
            actual: structure.terms().len(),
        });
    }
    for (term, fusion_term) in descriptor.terms().iter().zip(structure.terms()) {
        if term.dst_block != fusion_term.dst_block() || term.src_block != fusion_term.src_block() {
            return Err(OperationError::StructureMismatch {
                tensor: "trace term",
            });
        }
    }

    if let Some(data) = try_tensortrace_owned_raw(
        dst_space.structure(),
        dst_space.nout(),
        src_space.structure(),
        src_data,
        descriptor.source_conjugate(),
        descriptor.terms().len(),
        descriptor.destination_producer_indices(),
        descriptor.destination_producer_offsets(),
        |term_index| {
            let term = &descriptor.terms()[term_index];
            let fusion_term = &structure.terms()[term_index];
            OwnedTraceTerm::new(
                term.dst_block,
                term.src_block,
                descriptor.output_shape(term),
                descriptor.trace_shape(term),
                descriptor.src_output_strides(term),
                descriptor.src_trace_strides(term),
                fusion_term.coefficient,
            )
        },
        alpha,
    )? {
        return Ok(data);
    }

    let required_len = dst_space
        .structure()
        .required_len()
        .map_err(OperationError::from_core_preserving_context)?;
    let mut data = vec![D::zero(); required_len];
    tensortrace_fusion_dyn_structure_into_raw(
        structure,
        dst_space,
        &mut data,
        src_space,
        src_data,
        alpha,
        D::zero(),
    )?;
    Ok(data)
}

#[allow(clippy::too_many_arguments)]
fn tensortrace_fusion_dyn_structure_into_raw<C, D>(
    structure: &TensorTraceFusionStructure<C>,
    dst_space: &DynamicFusionMapSpace,
    dst_data: &mut [D],
    src_space: &DynamicFusionMapSpace,
    src_data: &[D],
    alpha: D,
    beta: D,
) -> Result<(), OperationError>
where
    C: Copy,
    D: Copy
        + Add<D, Output = D>
        + Mul<D, Output = D>
        + PartialEq
        + Zero
        + One
        + ConjugateValue
        + RecouplingCoefficientAction<C>
        + strided_kernel::MaybeSendSync,
{
    validate_trace_data_extents(
        dst_space.structure(),
        dst_data.len(),
        src_space.structure(),
        src_data.len(),
    )?;
    let descriptor = structure.descriptor();
    scale_trace_destination_layouts(descriptor, dst_data, beta)?;
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

/// Preflights finite-label trace metadata and returns the selected result HomSpace.
///
/// The user facade calls this before materializing source data or constructing
/// a result layout, so a checked algebra failure cannot publish either derived
/// state first.
#[doc(hidden)]
pub fn tensortrace_fusion_dyn_selected_homspace_checked<R>(
    src: &BoundDynamicFusionMapSpace<R>,
    axes: TensorTraceAxisSpec<'_>,
    dst_nout: usize,
) -> Result<FusionTreeHomSpace, OperationError>
where
    R: FusionRule + CheckedFusionAlgebra,
{
    src.space().validate_rule(src.provider())?;
    let orientation = if axes.source_conjugate() {
        FusionTreePairOrientation::Adjoint
    } else {
        FusionTreePairOrientation::Direct
    };
    let lowered_axes =
        lower_tensortrace_source_adjoint_axes_dyn(src.space().nout(), src.space().nin(), axes)?;
    let lowered_spec = lowered_axes.as_spec();
    let axis_plan = TensorTraceAxisPlan::compile(
        src.space().rank(),
        lowered_spec.output_axes().len(),
        lowered_spec,
    )?;
    checked_fusion_trace_geometry(
        src.provider(),
        OrientedFusionTreeHomSpace::new(src.space().homspace(), orientation),
        &axis_plan,
        dst_nout,
    )
    .map(|geometry| geometry.selected_homspace)
}

/// Checked lowered trace entry point. The legacy dynamic API intentionally
/// keeps its `MultiplicityFreeRigidSymbols` bound for custom rules; lowered
/// built-in callers use this wrapper to validate metadata before compilation.
#[allow(clippy::too_many_arguments)]
pub(crate) fn tensortrace_fusion_dyn_into_checked_raw<R, D>(
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
    R: MultiplicityFreeRigidSymbols + CheckedFusionAlgebra,
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
    let structure = TensorTraceFusionStructure::compile_fusion_dyn_checked_raw(
        rule, dst_space, src_space, axes,
    )?;
    tensortrace_fusion_dyn_structure_into_raw(
        &structure, dst_space, dst_data, src_space, src_data, alpha, beta,
    )
}

fn checked_fusion_trace_geometry<R>(
    rule: &R,
    src_homspace: OrientedFusionTreeHomSpace<'_>,
    axis_plan: &TensorTraceAxisPlan,
    dst_nout: usize,
) -> Result<CheckedTraceGeometry, OperationError>
where
    R: FusionRule + CheckedFusionAlgebra,
{
    if dst_nout > axis_plan.output_axes.len() {
        return Err(OperationError::RankMismatch {
            expected: axis_plan.output_axes.len(),
            actual: dst_nout,
        });
    }
    // Why not widen the public rule bound: custom encoded rules retain their
    // established infallible contract while lowered built-ins close overflow.
    record_trace_selected_homspace_derivation();
    let selected = src_homspace
        .try_select_checked(
            rule,
            &axis_plan.output_axes[..dst_nout],
            &axis_plan.output_axes[dst_nout..],
        )
        .map_err(|error| match error {
            CheckedFusionSpaceError::FusionAlgebra(error) => OperationError::FusionAlgebra(error),
            CheckedFusionSpaceError::Core(error) => OperationError::Core(*error),
            _ => OperationError::InvalidArgument {
                message: "checked trace metadata error",
            },
        })?;
    let mut trace_pairs_match = true;
    for (&lhs_axis, &rhs_axis) in axis_plan
        .trace_lhs_axes
        .iter()
        .zip(axis_plan.trace_rhs_axes.iter())
    {
        let lhs = outward_axis_leg_checked(rule, src_homspace, lhs_axis)?;
        let rhs = outward_axis_leg_checked(rule, src_homspace, rhs_axis)?;
        let rhs_dual = rhs
            .try_dual(rule)
            .map_err(|error| OperationError::FusionAlgebra(Box::new(error)))?;
        lhs.try_dual(rule)
            .map_err(|error| OperationError::FusionAlgebra(Box::new(error)))?;
        trace_pairs_match &= lhs == rhs_dual;
    }
    Ok(CheckedTraceGeometry {
        selected_homspace: selected,
        trace_pairs_match,
    })
}

fn validate_trace_data_extents(
    dst_structure: &BlockStructure,
    dst_len: usize,
    src_structure: &BlockStructure,
    src_len: usize,
) -> Result<(), OperationError> {
    let expected_dst = dst_structure
        .required_len()
        .map_err(OperationError::from_core_preserving_context)?;
    if dst_len != expected_dst {
        return Err(OperationError::ElementCountMismatch {
            expected: expected_dst,
            actual: dst_len,
        });
    }
    let expected_src = src_structure
        .required_len()
        .map_err(OperationError::from_core_preserving_context)?;
    if src_len != expected_src {
        return Err(OperationError::ElementCountMismatch {
            expected: expected_src,
            actual: src_len,
        });
    }
    Ok(())
}

fn scale_trace_destination_layouts<T>(
    descriptor: &TensorTraceDescriptor,
    dst: &mut [T],
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
{
    if beta.is_one() {
        return Ok(());
    }
    if beta.is_zero() {
        let zero = [T::zero()];
        for layout in descriptor.destination_layouts() {
            axpby_raw_strided_kernel_trusted(
                dst,
                &zero,
                descriptor.destination_shape(layout),
                descriptor.destination_strides(layout),
                descriptor.destination_zero_strides(layout),
                layout.offset,
                0,
                T::one(),
                beta,
            )?;
        }
        return Ok(());
    }
    for layout in descriptor.destination_layouts() {
        scale_raw_strided_kernel_trusted(
            dst,
            descriptor.destination_shape(layout),
            descriptor.destination_strides(layout),
            layout.offset,
            beta,
        )?;
    }
    Ok(())
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
