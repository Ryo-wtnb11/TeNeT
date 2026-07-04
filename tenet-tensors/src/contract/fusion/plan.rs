use tenet_core::{FusionTensorMapSpace, FusionTreeHomSpace, MultiplicityFreeRigidSymbols};

use crate::lowering::lower_tensorcontract_adjoint_axes;
use crate::{OperationError, TreeTransformOperation};
use tenet_operations::{TensorContractSpec, TensorContractSpecOwned};

use super::super::dynamic_space::DynamicFusionMapSpace;
use super::super::structure::TensorContractAxisPlan;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct FusionContractPlan {
    lhs_transform: TreeTransformOperation,
    rhs_transform: TreeTransformOperation,
    output_transform: TreeTransformOperation,
    core_axes: TensorContractSpecOwned,
    core_dst_open_lhs_rank: usize,
    core_dst_open_rhs_rank: usize,
    lhs_open_rank: usize,
    lhs_contract_rank: usize,
    rhs_contract_rank: usize,
    rhs_open_rank: usize,
    lhs_source_conjugate: bool,
    rhs_source_conjugate: bool,
}

impl FusionContractPlan {
    #[inline]
    pub fn lhs_transform(&self) -> &TreeTransformOperation {
        &self.lhs_transform
    }

    #[inline]
    pub fn rhs_transform(&self) -> &TreeTransformOperation {
        &self.rhs_transform
    }

    #[inline]
    pub fn output_transform(&self) -> &TreeTransformOperation {
        &self.output_transform
    }

    #[inline]
    pub fn core_axes(&self) -> &TensorContractSpecOwned {
        &self.core_axes
    }

    #[inline]
    pub fn core_dst_open_lhs_rank(&self) -> usize {
        self.core_dst_open_lhs_rank
    }

    #[inline]
    pub fn core_dst_open_rhs_rank(&self) -> usize {
        self.core_dst_open_rhs_rank
    }

    pub(crate) fn output_transform_is_identity(&self) -> bool {
        let core_rank = self.core_dst_open_lhs_rank + self.core_dst_open_rhs_rank;
        match &self.output_transform {
            TreeTransformOperation::Permute {
                codomain_permutation,
                domain_permutation,
            } => {
                codomain_permutation
                    .iter()
                    .copied()
                    .eq(0..self.core_dst_open_lhs_rank)
                    && domain_permutation
                        .iter()
                        .copied()
                        .eq(self.core_dst_open_lhs_rank..core_rank)
            }
            _ => false,
        }
    }

    #[inline]
    pub fn lhs_open_rank(&self) -> usize {
        self.lhs_open_rank
    }

    #[inline]
    pub fn lhs_contract_rank(&self) -> usize {
        self.lhs_contract_rank
    }

    #[inline]
    pub fn rhs_contract_rank(&self) -> usize {
        self.rhs_contract_rank
    }

    #[inline]
    pub fn rhs_open_rank(&self) -> usize {
        self.rhs_open_rank
    }

    #[inline]
    pub fn lhs_source_conjugate(&self) -> bool {
        self.lhs_source_conjugate
    }

    #[inline]
    pub fn rhs_source_conjugate(&self) -> bool {
        self.rhs_source_conjugate
    }
}

pub fn prepare_tensorcontract_fusion_plan<
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
    axes: TensorContractSpec<'_>,
) -> Result<FusionContractPlan, OperationError>
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64>,
{
    prepare_tensorcontract_fusion_plan_dyn(
        rule,
        &DynamicFusionMapSpace::from_typed(dst),
        &DynamicFusionMapSpace::from_typed(lhs),
        &DynamicFusionMapSpace::from_typed(rhs),
        axes,
    )
}

/// Dynamic-rank variant of [`prepare_tensorcontract_fusion_plan`]: same
/// lowering, spaces given as [`DynamicFusionMapSpace`] handles.
pub fn prepare_tensorcontract_fusion_plan_dyn<R>(
    rule: &R,
    dst: &DynamicFusionMapSpace,
    lhs: &DynamicFusionMapSpace,
    rhs: &DynamicFusionMapSpace,
    axes: TensorContractSpec<'_>,
) -> Result<FusionContractPlan, OperationError>
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64>,
{
    let lowered_axes =
        lower_tensorcontract_adjoint_axes(lhs.nout(), lhs.nin(), rhs.nout(), rhs.nin(), axes)?;
    let lhs_adjoint;
    let lhs = if axes.lhs_conjugate() {
        lhs_adjoint = lhs.adjoint_view()?;
        &lhs_adjoint
    } else {
        lhs
    };
    let rhs_adjoint;
    let rhs = if axes.rhs_conjugate() {
        rhs_adjoint = rhs.adjoint_view()?;
        &rhs_adjoint
    } else {
        rhs
    };
    prepare_tensorcontract_fusion_plan_from_spaces(
        rule,
        dst,
        lhs,
        rhs,
        lowered_axes.as_spec(),
        lowered_axes.lhs_storage_conjugate(),
        lowered_axes.rhs_storage_conjugate(),
    )
}

fn prepare_tensorcontract_fusion_plan_from_spaces<R>(
    rule: &R,
    dst: &DynamicFusionMapSpace,
    lhs: &DynamicFusionMapSpace,
    rhs: &DynamicFusionMapSpace,
    axes: TensorContractSpec<'_>,
    lhs_source_conjugate: bool,
    rhs_source_conjugate: bool,
) -> Result<FusionContractPlan, OperationError>
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64>,
{
    let dst_nout = dst.nout();
    let axis_plan = TensorContractAxisPlan::compile(lhs.rank(), rhs.rank(), dst.rank(), axes)?;
    let expected_homspace = FusionTreeHomSpace::tensorcontract_homspace(
        rule,
        lhs.homspace(),
        rhs.homspace(),
        axes.lhs_contracting_axes(),
        axes.rhs_contracting_axes(),
        axis_plan.output_axes.as_slice(),
        dst_nout,
    )
    .map_err(OperationError::from_core_preserving_context)?;
    if &expected_homspace != dst.homspace() {
        return Err(OperationError::StructureMismatch { tensor: "dst" });
    }

    let lhs_open_rank = axis_plan.lhs_open_axes.len();
    let lhs_contract_rank = axis_plan.lhs_contracting_axes.len();
    let rhs_contract_rank = axis_plan.rhs_contracting_axes.len();
    let rhs_open_rank = axis_plan.rhs_open_axes.len();
    let core_dst_open_lhs_rank = lhs_open_rank;
    let core_dst_open_rhs_rank = rhs_open_rank;
    let core_output_rank = core_dst_open_lhs_rank + core_dst_open_rhs_rank;
    let output_transform = TreeTransformOperation::permute(
        axis_plan.output_axes[..dst_nout].to_vec(),
        axis_plan.output_axes[dst_nout..].to_vec(),
    );
    Ok(FusionContractPlan {
        lhs_transform: TreeTransformOperation::permute(
            axis_plan.lhs_open_axes,
            axis_plan.lhs_contracting_axes,
        ),
        rhs_transform: TreeTransformOperation::permute(
            axis_plan.rhs_contracting_axes,
            axis_plan.rhs_open_axes,
        ),
        core_axes: TensorContractSpecOwned::new(
            (lhs_open_rank..lhs_open_rank + lhs_contract_rank).collect(),
            (0..rhs_contract_rank).collect(),
            (0..core_output_rank).collect(),
        ),
        output_transform,
        core_dst_open_lhs_rank,
        core_dst_open_rhs_rank,
        lhs_open_rank,
        lhs_contract_rank,
        rhs_contract_rank,
        rhs_open_rank,
        lhs_source_conjugate,
        rhs_source_conjugate,
    })
}
