use std::collections::HashSet;

use rustc_hash::FxHashMap;
use std::sync::Arc;

use tenet_core::{
    BlockKey, CategoricalScalar, CoupledMatrixSide, FusionRule, FusionTreeHomSpace, FusionTreeKey,
    FusionTreePairOrientation, MultiplicityFreeRigidSymbols, OrientedFusionTreeHomSpace, SectorId,
};
#[cfg(test)]
use tenet_core::{HostReadableStorage, HostWritableStorage};

use crate::strided::{
    column_major_strides_isize, column_major_strides_usize, element_count, offset_to_isize,
    strides_to_isize,
};
use crate::{DenseBlockScalar, HostKernelAdapter, OperationError, RecouplingCoefficientAction};
use tenet_operations::TensorContractSpec;

#[cfg(test)]
pub(crate) use tenet_operations::fusion_replay::StorageGemm;
use tenet_operations::fusion_replay::{
    direct_group_matrix_offset_generic, fusion_scale_block_layouts_excluding,
    FusionBlockContractGroupPlan, FusionBlockMatrixGroup, FusionStridedBlockLayout,
    FusionSubblockMatrixLayout, MatrixOp, Rank2GemmBatchJob,
};
pub(crate) use tenet_operations::fusion_replay::{
    FusionBlockContractPlan, FusionBlockContractWorkspace, Rank2Gemm,
};

/// Validate category identity before a contraction route reads sectors or
/// symbols through the supplied rule.
///
/// Why-not validate during replay: a compiled plan is already tied to
/// validated spaces, so replay checks would charge every warm execution for a
/// compile-time invariant.
pub(crate) fn validate_fusion_contract_rule<R>(
    rule: &R,
    dst: &DynamicFusionMapSpace,
    lhs: &DynamicFusionMapSpace,
    rhs: &DynamicFusionMapSpace,
) -> Result<(), OperationError>
where
    R: FusionRule,
{
    dst.validate_rule(rule)?;
    lhs.validate_rule(rule)?;
    rhs.validate_rule(rule)
}

use super::backend::TensorContractBackend;
use super::dynamic_space::{DynamicFusionMapSpace, FusionOperandLayout};
use super::fusion::reject_fusion_contract_conjugation;
use super::structure::TensorContractAxisPlan;

mod compile;
#[cfg(test)]
mod tests;
pub(crate) use compile::*;

pub(super) struct CoreContractPreflight<'a, R> {
    rule: &'a R,
    dst_homspace: &'a FusionTreeHomSpace,
    lhs_homspace: OrientedFusionTreeHomSpace<'a>,
    rhs_homspace: OrientedFusionTreeHomSpace<'a>,
    axis_plan: TensorContractAxisPlan,
}

pub(super) struct ValidatedCoreContract<'a, R> {
    preflight: CoreContractPreflight<'a, R>,
}

impl<'a, R> CoreContractPreflight<'a, R>
where
    R: FusionRule,
{
    pub(super) fn compile(
        rule: &'a R,
        dst: &'a DynamicFusionMapSpace,
        lhs: &'a DynamicFusionMapSpace,
        rhs: &'a DynamicFusionMapSpace,
        axes: TensorContractSpec<'_>,
    ) -> Result<Self, OperationError> {
        validate_fusion_contract_rule(rule, dst, lhs, rhs)?;
        Self::compile_homspaces(rule, dst.homspace(), lhs.homspace(), rhs.homspace(), axes)
    }

    pub(super) fn compile_homspaces(
        rule: &'a R,
        dst_homspace: &'a FusionTreeHomSpace,
        lhs_homspace: &'a FusionTreeHomSpace,
        rhs_homspace: &'a FusionTreeHomSpace,
        axes: TensorContractSpec<'_>,
    ) -> Result<Self, OperationError> {
        Self::compile_oriented(
            rule,
            dst_homspace,
            OrientedFusionTreeHomSpace::new(lhs_homspace, FusionTreePairOrientation::Direct),
            OrientedFusionTreeHomSpace::new(rhs_homspace, FusionTreePairOrientation::Direct),
            axes,
        )
    }

    pub(super) fn compile_oriented(
        rule: &'a R,
        dst_homspace: &'a FusionTreeHomSpace,
        lhs_homspace: OrientedFusionTreeHomSpace<'a>,
        rhs_homspace: OrientedFusionTreeHomSpace<'a>,
        axes: TensorContractSpec<'_>,
    ) -> Result<Self, OperationError> {
        #[cfg(test)]
        CORE_CONTRACT_PREFLIGHTS.set(CORE_CONTRACT_PREFLIGHTS.get() + 1);
        let axis_plan = TensorContractAxisPlan::compile(
            lhs_homspace.rank(),
            rhs_homspace.rank(),
            dst_homspace.rank(),
            axes,
        )?;
        Ok(Self {
            rule,
            dst_homspace,
            lhs_homspace,
            rhs_homspace,
            axis_plan,
        })
    }

    pub(super) fn has_conjugation(&self) -> bool {
        self.axis_plan.lhs_conjugate || self.axis_plan.rhs_conjugate
    }

    pub(super) fn validate_core_geometry(
        self,
    ) -> Result<Option<ValidatedCoreContract<'a, R>>, OperationError> {
        if !is_core_form_source(
            self.lhs_homspace.rank(),
            self.lhs_homspace.nout(),
            self.rhs_homspace.nout(),
            &self.axis_plan,
        ) || !is_core_form_output(
            self.dst_homspace.codomain().len(),
            self.lhs_homspace.nout(),
            self.rhs_homspace.rank(),
            self.rhs_homspace.nout(),
            &self.axis_plan,
        ) {
            return Ok(None);
        }
        if !core_homspace_matches(
            self.rule,
            self.lhs_homspace,
            self.rhs_homspace,
            self.axis_plan.lhs_contracting_axes.as_slice(),
            self.axis_plan.rhs_contracting_axes.as_slice(),
            self.axis_plan.output_axes.as_slice(),
            self.dst_homspace,
        )? {
            return Err(OperationError::StructureMismatch { tensor: "dst" });
        }
        Ok(Some(ValidatedCoreContract { preflight: self }))
    }

    pub(super) fn require_core_geometry(
        self,
    ) -> Result<ValidatedCoreContract<'a, R>, OperationError> {
        self.validate_core_geometry()?
            .ok_or(OperationError::UnsupportedTensorContractScope {
                message: "core fusion-block contraction requires core source and output axes",
            })
    }
}

impl<'a, R> ValidatedCoreContract<'a, R> {
    pub(super) fn rule(&self) -> &'a R {
        self.preflight.rule
    }

    pub(super) fn rhs_homspace(&self) -> OrientedFusionTreeHomSpace<'a> {
        self.preflight.rhs_homspace
    }

    pub(super) fn rhs_contracting_axes(&self) -> &[usize] {
        &self.preflight.axis_plan.rhs_contracting_axes
    }
}

/// The core destination check: `dst` must be the contracted HomSpace of
/// the two oriented operands. Decided by leg comparison, without building
/// that HomSpace, with its errors in its order.
fn core_homspace_matches<R>(
    rule: &R,
    lhs: OrientedFusionTreeHomSpace<'_>,
    rhs: OrientedFusionTreeHomSpace<'_>,
    lhs_contracting_axes: &[usize],
    rhs_contracting_axes: &[usize],
    output_axes: &[usize],
    dst: &FusionTreeHomSpace,
) -> Result<bool, OperationError>
where
    R: FusionRule,
{
    #[cfg(test)]
    EXPECTED_CORE_HOMSPACE_DERIVATIONS.set(EXPECTED_CORE_HOMSPACE_DERIVATIONS.get() + 1);
    OrientedFusionTreeHomSpace::tensorcontract_homspace_matches(
        rule,
        lhs,
        rhs,
        lhs_contracting_axes,
        rhs_contracting_axes,
        output_axes,
        dst.codomain().len(),
        dst,
    )
    .map_err(OperationError::from_core_preserving_context)
}

#[cfg(test)]
thread_local! {
    static EXPECTED_CORE_HOMSPACE_DERIVATIONS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
    static CORE_CONTRACT_PREFLIGHTS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

#[cfg(test)]
pub(crate) fn reset_core_contract_derivations() {
    super::structure::reset_tensor_contract_axis_plan_compiles();
    EXPECTED_CORE_HOMSPACE_DERIVATIONS.set(0);
    CORE_CONTRACT_PREFLIGHTS.set(0);
    super::dynamic_space::reset_derived_homspace_builds();
}

/// Asserts what [`super::resolution::compile_derived_core_plan`] relies on:
/// the full core preflight and destination check accept the plan-derived
/// operands. Leaves the compile counters as they were, so tests keep
/// counting the production work only.
#[cfg(debug_assertions)]
pub(crate) fn debug_assert_core_geometry<R>(
    rule: &R,
    dst: &DynamicFusionMapSpace,
    lhs: &DynamicFusionMapSpace,
    rhs: &DynamicFusionMapSpace,
    core_axes: TensorContractSpec<'_>,
) where
    R: FusionRule,
{
    #[cfg(test)]
    let counts = (
        CORE_CONTRACT_PREFLIGHTS.get(),
        EXPECTED_CORE_HOMSPACE_DERIVATIONS.get(),
        super::structure::tensor_contract_axis_plan_compiles(),
    );
    let checked = reject_fusion_contract_conjugation(core_axes).and_then(|()| {
        CoreContractPreflight::compile(rule, dst, lhs, rhs, core_axes)?
            .require_core_geometry()
            .map(|_| ())
    });
    #[cfg(test)]
    {
        CORE_CONTRACT_PREFLIGHTS.set(counts.0);
        EXPECTED_CORE_HOMSPACE_DERIVATIONS.set(counts.1);
        super::structure::set_tensor_contract_axis_plan_compiles(counts.2);
    }
    debug_assert!(
        checked.is_ok(),
        "plan-derived core operands failed the core preflight: {checked:?}"
    );
}

/// What one contraction compile did: core preflights, core destination
/// checks, and permuted or contracted HomSpaces built.
#[cfg(test)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct ContractCompileCounts {
    pub(crate) preflights: usize,
    pub(crate) core_destination_checks: usize,
    pub(crate) derived_homspace_builds: usize,
}

#[cfg(test)]
pub(crate) fn contract_compile_counts() -> ContractCompileCounts {
    ContractCompileCounts {
        preflights: CORE_CONTRACT_PREFLIGHTS.get(),
        core_destination_checks: EXPECTED_CORE_HOMSPACE_DERIVATIONS.get(),
        derived_homspace_builds: super::dynamic_space::derived_homspace_builds(),
    }
}

#[cfg(test)]
pub(crate) fn core_contract_derivations() -> (usize, usize) {
    (
        super::structure::tensor_contract_axis_plan_compiles(),
        EXPECTED_CORE_HOMSPACE_DERIVATIONS.get(),
    )
}

/// Adapts a [`TensorContractBackend`] + workspace pair onto the replay
/// layer's [`Rank2Gemm`] seam.
pub(crate) struct BackendRank2Gemm<'a, B, W, C = f64> {
    pub(crate) backend: &'a mut B,
    pub(crate) workspace: &'a mut W,
    pub(crate) coefficient: std::marker::PhantomData<C>,
}

impl<'a, B, W, C> BackendRank2Gemm<'a, B, W, C> {
    pub(crate) fn new(backend: &'a mut B, workspace: &'a mut W) -> Self {
        Self {
            backend,
            workspace,
            coefficient: std::marker::PhantomData,
        }
    }
}

impl<'a, B, D, C> Rank2Gemm<D> for BackendRank2Gemm<'a, B, B::Workspace, C>
where
    B: TensorContractBackend<D, C>,
    D: DenseBlockScalar + RecouplingCoefficientAction<C>,
    C: Copy + num_traits::One,
{
    fn matmul_rank2(
        &mut self,
        dst: &mut [D],
        lhs: &[D],
        rhs: &[D],
        rows: usize,
        contracted: usize,
        cols: usize,
        alpha: D,
        beta: D,
    ) -> Result<(), OperationError> {
        self.backend.matmul_rank2_axpby_into_raw(
            self.workspace,
            dst,
            lhs,
            rhs,
            rows,
            contracted,
            cols,
            alpha,
            beta,
        )
    }

    fn matmul_rank2_batch(
        &mut self,
        dst: &mut [D],
        lhs: &[D],
        rhs: &[D],
        jobs: &[Rank2GemmBatchJob],
        runs: &[usize],
        alpha: D,
        beta: D,
    ) -> Result<(), OperationError>
    where
        D: Copy,
    {
        self.backend.matmul_rank2_batch_axpby_into_raw(
            self.workspace,
            dst,
            lhs,
            rhs,
            jobs,
            runs,
            alpha,
            beta,
        )
    }

    fn matmul_rank2_batch_with_ops(
        &mut self,
        dst: &mut [D],
        lhs: &[D],
        rhs: &[D],
        jobs: &[Rank2GemmBatchJob],
        runs: &[usize],
        lhs_op: MatrixOp,
        rhs_op: MatrixOp,
        alpha: D,
        beta: D,
    ) -> Result<(), OperationError>
    where
        D: Copy,
    {
        self.backend.matmul_rank2_batch_with_ops_axpby_into_raw(
            self.workspace,
            dst,
            lhs,
            rhs,
            jobs,
            runs,
            lhs_op,
            rhs_op,
            alpha,
            beta,
        )
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn tensorcontract_core_fusion_blocks_into_raw<A, B, R, D, C>(
    kernels: &mut A,
    backend: &mut B,
    workspace: &mut B::Workspace,
    rule: &R,
    dst_space: &DynamicFusionMapSpace,
    dst_data: &mut [D],
    lhs_space: &DynamicFusionMapSpace,
    lhs_data: &[D],
    rhs_space: &DynamicFusionMapSpace,
    rhs_data: &[D],
    axes: TensorContractSpec<'_>,
    alpha: D,
    beta: D,
) -> Result<(), OperationError>
where
    A: HostKernelAdapter<D>,
    B: TensorContractBackend<D, C>,
    R: MultiplicityFreeRigidSymbols<Scalar = C>,
    D: DenseBlockScalar + RecouplingCoefficientAction<C>,
    C: DenseBlockScalar,
{
    let plan = compile_fusion_block_contract_plan(rule, dst_space, lhs_space, rhs_space, axes)?;
    tensorcontract_core_fusion_blocks_with_plan_into_raw(
        kernels, backend, workspace, &plan, dst_space, dst_data, lhs_space, lhs_data, rhs_space,
        rhs_data, alpha, beta,
    )
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn tensorcontract_core_fusion_blocks_with_plan_into_raw<A, B, D, C>(
    kernels: &mut A,
    backend: &mut B,
    workspace: &mut B::Workspace,
    plan: &FusionBlockContractPlan<C>,
    dst_space: &DynamicFusionMapSpace,
    dst_data: &mut [D],
    lhs_space: &DynamicFusionMapSpace,
    lhs_data: &[D],
    rhs_space: &DynamicFusionMapSpace,
    rhs_data: &[D],
    alpha: D,
    beta: D,
) -> Result<(), OperationError>
where
    A: HostKernelAdapter<D>,
    B: TensorContractBackend<D, C>,
    D: DenseBlockScalar + RecouplingCoefficientAction<C>,
    C: DenseBlockScalar,
{
    let mut fusion_workspace = FusionBlockContractWorkspace::<D>::default();
    plan.execute_raw(
        kernels,
        &mut BackendRank2Gemm::new(backend, workspace),
        &mut fusion_workspace,
        dst_space.structure(),
        dst_data,
        lhs_space.structure(),
        lhs_data,
        rhs_space.structure(),
        rhs_data,
        alpha,
        beta,
    )
}

/// Generic-fusion (Stage B3c-1) sibling of the multiplicity-free core classifier:
/// identical predicate, relaxed to any [`FusionRule`]. The homspace-shape check
/// (`tensorcontract_homspace`) and the axis-form checks are already fully
/// symmetry-agnostic — only the mult-free trait bound differed.
fn is_core_form_fusion_block_contract_generic<R>(
    rule: &R,
    dst_space: &DynamicFusionMapSpace,
    lhs_space: &DynamicFusionMapSpace,
    rhs_space: &DynamicFusionMapSpace,
    axes: TensorContractSpec<'_>,
) -> Result<bool, OperationError>
where
    R: FusionRule,
{
    reject_fusion_contract_conjugation(axes)?;
    let axis_plan = TensorContractAxisPlan::compile(
        lhs_space.rank(),
        rhs_space.rank(),
        dst_space.rank(),
        axes,
    )?;
    if !is_core_form_source(
        lhs_space.rank(),
        lhs_space.nout(),
        rhs_space.nout(),
        &axis_plan,
    ) || !is_core_form_output(
        dst_space.nout(),
        lhs_space.nout(),
        rhs_space.rank(),
        rhs_space.nout(),
        &axis_plan,
    ) {
        return Ok(false);
    }
    let expected_homspace = FusionTreeHomSpace::tensorcontract_homspace(
        rule,
        lhs_space.homspace(),
        rhs_space.homspace(),
        axes.lhs_contracting_axes(),
        axes.rhs_contracting_axes(),
        axis_plan.output_axes.as_slice(),
        dst_space.nout(),
    )
    .map_err(OperationError::from_core_preserving_context)?;
    if expected_homspace != *dst_space.homspace() {
        return Err(OperationError::StructureMismatch { tensor: "dst" });
    }
    Ok(true)
}

fn is_core_form_source(
    lhs_rank: usize,
    lhs_nout: usize,
    rhs_nout: usize,
    axis_plan: &TensorContractAxisPlan,
) -> bool {
    axis_plan
        .lhs_contracting_axes
        .iter()
        .copied()
        .eq(lhs_nout..lhs_rank)
        && axis_plan
            .rhs_contracting_axes
            .iter()
            .copied()
            .eq(0..rhs_nout)
}

fn is_core_form_output(
    dst_nout: usize,
    lhs_nout: usize,
    rhs_rank: usize,
    rhs_nout: usize,
    axis_plan: &TensorContractAxisPlan,
) -> bool {
    let output_rank = lhs_nout + (rhs_rank - rhs_nout);
    dst_nout == lhs_nout && axis_plan.output_axes.iter().copied().eq(0..output_rank)
}
