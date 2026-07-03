use tenet_core::{FusionTensorMapSpace, FusionTreeHomSpace, MultiplicityFreeRigidSymbols};

use crate::axis::{OwnedTensorContractAxisSpec, TensorContractAxisSpec};
use crate::lowering::{adjoint_fusion_space_view, lower_tensorcontract_adjoint_axes};
use crate::{OperationError, TreeTransformOperationKey};

use super::super::structure::TensorContractAxisPlan;

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
    lhs_source_conjugate: bool,
    rhs_source_conjugate: bool,
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

    pub(crate) fn output_transform_is_identity(&self) -> bool {
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

    #[inline]
    pub fn lhs_source_conjugate(&self) -> bool {
        self.lhs_source_conjugate
    }

    #[inline]
    pub fn rhs_source_conjugate(&self) -> bool {
        self.rhs_source_conjugate
    }
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
    let lowered_axes =
        lower_tensorcontract_adjoint_axes::<LHS_NOUT, LHS_NIN, RHS_NOUT, RHS_NIN>(axes)?;
    if axes.lhs_conjugate() && axes.rhs_conjugate() {
        let lhs_adjoint = adjoint_fusion_space_view(lhs)?;
        let rhs_adjoint = adjoint_fusion_space_view(rhs)?;
        return tensorcontract_fusion_explicit_plan_from_spaces(
            rule,
            dst,
            &lhs_adjoint,
            &rhs_adjoint,
            lowered_axes.as_spec(),
            lowered_axes.lhs_storage_conjugate(),
            lowered_axes.rhs_storage_conjugate(),
        );
    }
    if axes.lhs_conjugate() {
        let lhs_adjoint = adjoint_fusion_space_view(lhs)?;
        return tensorcontract_fusion_explicit_plan_from_spaces(
            rule,
            dst,
            &lhs_adjoint,
            rhs,
            lowered_axes.as_spec(),
            lowered_axes.lhs_storage_conjugate(),
            lowered_axes.rhs_storage_conjugate(),
        );
    }
    if axes.rhs_conjugate() {
        let rhs_adjoint = adjoint_fusion_space_view(rhs)?;
        return tensorcontract_fusion_explicit_plan_from_spaces(
            rule,
            dst,
            lhs,
            &rhs_adjoint,
            lowered_axes.as_spec(),
            lowered_axes.lhs_storage_conjugate(),
            lowered_axes.rhs_storage_conjugate(),
        );
    }
    tensorcontract_fusion_explicit_plan_from_spaces(
        rule,
        dst,
        lhs,
        rhs,
        lowered_axes.as_spec(),
        false,
        false,
    )
}

fn tensorcontract_fusion_explicit_plan_from_spaces<
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
    lhs_source_conjugate: bool,
    rhs_source_conjugate: bool,
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
        lhs_source_conjugate,
        rhs_source_conjugate,
    })
}
