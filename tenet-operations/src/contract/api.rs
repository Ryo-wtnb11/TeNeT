use num_traits::One;
use tenet_core::{
    CoreError, HostReadableStorage, HostWritableStorage, MultiplicityFreeRigidSymbols, TensorMap,
};

use crate::axis::{AxisPermutation, TensorContractAxisSpec};
use crate::lowering::adjoint_fusion_space_view;
use crate::{
    build_tree_pair_transform_group_plan, tree_pair_transform_into_with, DenseBlockScalar,
    DenseRecouplingScalar, DenseTreeTransformOperations, OperationError,
    RecouplingCoefficientAction, TreeTransformBackend, TreeTransformWorkspace,
};

use super::backend::{TensorContractBackend, TensorContractWorkspace};
use super::dynamic::{
    tensorcontract_fusion_dynamic_plan_into_with,
    tensorcontract_fusion_dynamic_transforms_into_with,
};
use super::dynamic_space::DynamicFusionMapSpace;
use super::fusion::{
    tensorcontract_fusion_explicit_plan, tensorcontract_fusion_structure,
    TensorContractFusionExplicitPlan, EXPLICIT_OUTPUT_TRANSFORM_REQUIRES_CANONICAL_DST,
    SOURCE_TRANSFORM_REQUIRES_EXPLICIT,
};
use super::fusion_block::{
    is_canonical_fusion_block_contract, tensorcontract_canonical_fusion_blocks_into_raw,
};
use super::structure::{tensorcontract_structure, TensorContractStructure};

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
    DDst,
    DLhs,
    DRhs,
>(
    dst: &mut TensorMap<D, DST_NOUT, DST_NIN, SDst, DDst>,
    lhs: &TensorMap<D, LHS_NOUT, LHS_NIN, SLhs, DLhs>,
    rhs: &TensorMap<D, RHS_NOUT, RHS_NIN, SRhs, DRhs>,
    axes: TensorContractAxisSpec<'_>,
    alpha: D,
    beta: D,
) -> Result<(), OperationError>
where
    D: DenseBlockScalar + RecouplingCoefficientAction<f64>,
    DDst: HostWritableStorage<D>,
    DLhs: HostReadableStorage<D>,
    DRhs: HostReadableStorage<D>,
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

pub fn tensorproduct_into<
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
    dst: &mut TensorMap<D, DST_NOUT, DST_NIN, SDst, DDst>,
    lhs: &TensorMap<D, LHS_NOUT, LHS_NIN, SLhs, DLhs>,
    rhs: &TensorMap<D, RHS_NOUT, RHS_NIN, SRhs, DRhs>,
    output_permutation: AxisPermutation<'_>,
    alpha: D,
    beta: D,
) -> Result<(), OperationError>
where
    D: DenseBlockScalar + RecouplingCoefficientAction<f64>,
    DDst: HostWritableStorage<D>,
    DLhs: HostReadableStorage<D>,
    DRhs: HostReadableStorage<D>,
{
    tensorcontract_into(
        dst,
        lhs,
        rhs,
        TensorContractAxisSpec::new(&[], &[], output_permutation),
        alpha,
        beta,
    )
}

pub fn tensorproduct_into_with_conjugation<
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
    dst: &mut TensorMap<D, DST_NOUT, DST_NIN, SDst, DDst>,
    lhs: &TensorMap<D, LHS_NOUT, LHS_NIN, SLhs, DLhs>,
    rhs: &TensorMap<D, RHS_NOUT, RHS_NIN, SRhs, DRhs>,
    output_permutation: AxisPermutation<'_>,
    lhs_conjugate: bool,
    rhs_conjugate: bool,
    alpha: D,
    beta: D,
) -> Result<(), OperationError>
where
    D: DenseBlockScalar + RecouplingCoefficientAction<f64>,
    DDst: HostWritableStorage<D>,
    DLhs: HostReadableStorage<D>,
    DRhs: HostReadableStorage<D>,
{
    tensorcontract_into(
        dst,
        lhs,
        rhs,
        TensorContractAxisSpec::new_with_conjugation(
            &[],
            &[],
            output_permutation,
            lhs_conjugate,
            rhs_conjugate,
        ),
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
    DDst,
    DLhs,
    DRhs,
>(
    backend: &mut B,
    workspace: &mut B::Workspace,
    dst: &mut TensorMap<D, DST_NOUT, DST_NIN, SDst, DDst>,
    lhs: &TensorMap<D, LHS_NOUT, LHS_NIN, SLhs, DLhs>,
    rhs: &TensorMap<D, RHS_NOUT, RHS_NIN, SRhs, DRhs>,
    axes: TensorContractAxisSpec<'_>,
    alpha: D,
    beta: D,
) -> Result<(), OperationError>
where
    B: TensorContractBackend<D, f64>,
    D: DenseBlockScalar + RecouplingCoefficientAction<f64>,
    DDst: HostWritableStorage<D>,
    DLhs: HostReadableStorage<D>,
    DRhs: HostReadableStorage<D>,
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
    DDst,
    DLhs,
    DRhs,
>(
    rule: &R,
    dst: &mut TensorMap<D, DST_NOUT, DST_NIN, SDst, DDst>,
    lhs: &TensorMap<D, LHS_NOUT, LHS_NIN, SLhs, DLhs>,
    rhs: &TensorMap<D, RHS_NOUT, RHS_NIN, SRhs, DRhs>,
    axes: TensorContractAxisSpec<'_>,
    alpha: D,
    beta: D,
) -> Result<(), OperationError>
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64>,
    D: DenseRecouplingScalar + RecouplingCoefficientAction<f64>,
    DDst: HostWritableStorage<D>,
    DLhs: HostReadableStorage<D>,
    DRhs: HostReadableStorage<D>,
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

pub fn tensorproduct_fusion_into<
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
    DDst,
    DLhs,
    DRhs,
>(
    rule: &R,
    dst: &mut TensorMap<D, DST_NOUT, DST_NIN, SDst, DDst>,
    lhs: &TensorMap<D, LHS_NOUT, LHS_NIN, SLhs, DLhs>,
    rhs: &TensorMap<D, RHS_NOUT, RHS_NIN, SRhs, DRhs>,
    output_permutation: AxisPermutation<'_>,
    alpha: D,
    beta: D,
) -> Result<(), OperationError>
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64>,
    D: DenseRecouplingScalar + RecouplingCoefficientAction<f64>,
    DDst: HostWritableStorage<D>,
    DLhs: HostReadableStorage<D>,
    DRhs: HostReadableStorage<D>,
{
    tensorcontract_fusion_into(
        rule,
        dst,
        lhs,
        rhs,
        TensorContractAxisSpec::new(&[], &[], output_permutation),
        alpha,
        beta,
    )
}

pub fn tensorproduct_fusion_into_with_conjugation<
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
    DDst,
    DLhs,
    DRhs,
>(
    rule: &R,
    dst: &mut TensorMap<D, DST_NOUT, DST_NIN, SDst, DDst>,
    lhs: &TensorMap<D, LHS_NOUT, LHS_NIN, SLhs, DLhs>,
    rhs: &TensorMap<D, RHS_NOUT, RHS_NIN, SRhs, DRhs>,
    output_permutation: AxisPermutation<'_>,
    lhs_conjugate: bool,
    rhs_conjugate: bool,
    alpha: D,
    beta: D,
) -> Result<(), OperationError>
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64>,
    D: DenseRecouplingScalar + RecouplingCoefficientAction<f64>,
    DDst: HostWritableStorage<D>,
    DLhs: HostReadableStorage<D>,
    DRhs: HostReadableStorage<D>,
{
    tensorcontract_fusion_into(
        rule,
        dst,
        lhs,
        rhs,
        TensorContractAxisSpec::new_with_conjugation(
            &[],
            &[],
            output_permutation,
            lhs_conjugate,
            rhs_conjugate,
        ),
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
    DDst,
    DLhs,
    DRhs,
    DLhsCan,
    DRhsCan,
>(
    rule: &R,
    dst: &mut TensorMap<D, DST_NOUT, DST_NIN, SDst, DDst>,
    lhs_canonical: &mut TensorMap<D, LHS_CAN_NOUT, LHS_CAN_NIN, SLhsCan, DLhsCan>,
    rhs_canonical: &mut TensorMap<D, RHS_CAN_NOUT, RHS_CAN_NIN, SRhsCan, DRhsCan>,
    lhs: &TensorMap<D, LHS_NOUT, LHS_NIN, SLhs, DLhs>,
    rhs: &TensorMap<D, RHS_NOUT, RHS_NIN, SRhs, DRhs>,
    axes: TensorContractAxisSpec<'_>,
    alpha: D,
    beta: D,
) -> Result<(), OperationError>
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64>,
    D: DenseRecouplingScalar + RecouplingCoefficientAction<f64>,
    DDst: HostWritableStorage<D>,
    DLhs: HostReadableStorage<D>,
    DRhs: HostReadableStorage<D>,
    DLhsCan: HostWritableStorage<D> + HostReadableStorage<D>,
    DRhsCan: HostWritableStorage<D> + HostReadableStorage<D>,
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
    DDst,
    DLhs,
    DRhs,
    DLhsCan,
    DRhsCan,
>(
    rule: &R,
    plan: &TensorContractFusionExplicitPlan,
    dst: &mut TensorMap<D, DST_NOUT, DST_NIN, SDst, DDst>,
    lhs_canonical: &mut TensorMap<D, LHS_CAN_NOUT, LHS_CAN_NIN, SLhsCan, DLhsCan>,
    rhs_canonical: &mut TensorMap<D, RHS_CAN_NOUT, RHS_CAN_NIN, SRhsCan, DRhsCan>,
    lhs: &TensorMap<D, LHS_NOUT, LHS_NIN, SLhs, DLhs>,
    rhs: &TensorMap<D, RHS_NOUT, RHS_NIN, SRhs, DRhs>,
    alpha: D,
    beta: D,
) -> Result<(), OperationError>
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64>,
    D: DenseRecouplingScalar + RecouplingCoefficientAction<f64>,
    DDst: HostWritableStorage<D>,
    DLhs: HostReadableStorage<D>,
    DRhs: HostReadableStorage<D>,
    DLhsCan: HostWritableStorage<D> + HostReadableStorage<D>,
    DRhsCan: HostWritableStorage<D> + HostReadableStorage<D>,
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
    DDst,
    DDstCan,
    DLhs,
    DRhs,
    DLhsCan,
    DRhsCan,
>(
    rule: &R,
    plan: &TensorContractFusionExplicitPlan,
    dst: &mut TensorMap<D, DST_NOUT, DST_NIN, SDst, DDst>,
    canonical_dst: &mut TensorMap<D, DST_CAN_NOUT, DST_CAN_NIN, SDstCan, DDstCan>,
    lhs_canonical: &mut TensorMap<D, LHS_CAN_NOUT, LHS_CAN_NIN, SLhsCan, DLhsCan>,
    rhs_canonical: &mut TensorMap<D, RHS_CAN_NOUT, RHS_CAN_NIN, SRhsCan, DRhsCan>,
    lhs: &TensorMap<D, LHS_NOUT, LHS_NIN, SLhs, DLhs>,
    rhs: &TensorMap<D, RHS_NOUT, RHS_NIN, SRhs, DRhs>,
    alpha: D,
    beta: D,
) -> Result<(), OperationError>
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64>,
    D: DenseRecouplingScalar + RecouplingCoefficientAction<f64>,
    DDst: HostWritableStorage<D>,
    DDstCan: HostWritableStorage<D> + HostReadableStorage<D>,
    DLhs: HostReadableStorage<D>,
    DRhs: HostReadableStorage<D>,
    DLhsCan: HostWritableStorage<D> + HostReadableStorage<D>,
    DRhsCan: HostWritableStorage<D> + HostReadableStorage<D>,
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
    DDst,
    DLhs,
    DRhs,
    DLhsCan,
    DRhsCan,
>(
    tree_backend: &mut BT,
    tree_workspace: &mut BT::Workspace,
    contract_backend: &mut BC,
    contract_workspace: &mut BC::Workspace,
    rule: &R,
    plan: &TensorContractFusionExplicitPlan,
    dst: &mut TensorMap<D, DST_NOUT, DST_NIN, SDst, DDst>,
    lhs_canonical: &mut TensorMap<D, LHS_CAN_NOUT, LHS_CAN_NIN, SLhsCan, DLhsCan>,
    rhs_canonical: &mut TensorMap<D, RHS_CAN_NOUT, RHS_CAN_NIN, SRhsCan, DRhsCan>,
    lhs: &TensorMap<D, LHS_NOUT, LHS_NIN, SLhs, DLhs>,
    rhs: &TensorMap<D, RHS_NOUT, RHS_NIN, SRhs, DRhs>,
    alpha: D,
    beta: D,
) -> Result<(), OperationError>
where
    BT: TreeTransformBackend<D, f64>,
    BC: TensorContractBackend<D, f64>,
    R: MultiplicityFreeRigidSymbols<Scalar = f64>,
    D: DenseRecouplingScalar + RecouplingCoefficientAction<f64>,
    DDst: HostWritableStorage<D>,
    DLhs: HostReadableStorage<D>,
    DRhs: HostReadableStorage<D>,
    DLhsCan: HostWritableStorage<D> + HostReadableStorage<D>,
    DRhsCan: HostWritableStorage<D> + HostReadableStorage<D>,
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
    tree_pair_transform_into_with_optional_storage_conjugation(
        tree_backend,
        tree_workspace,
        rule,
        plan.lhs_transform().clone(),
        lhs_canonical,
        lhs,
        plan.lhs_source_conjugate(),
        D::one(),
        D::zero(),
    )?;
    tree_pair_transform_into_with_optional_storage_conjugation(
        tree_backend,
        tree_workspace,
        rule,
        plan.rhs_transform().clone(),
        rhs_canonical,
        rhs,
        plan.rhs_source_conjugate(),
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
    DDst,
    DDstCan,
    DLhs,
    DRhs,
    DLhsCan,
    DRhsCan,
>(
    tree_backend: &mut BT,
    tree_workspace: &mut BT::Workspace,
    contract_backend: &mut BC,
    contract_workspace: &mut BC::Workspace,
    rule: &R,
    plan: &TensorContractFusionExplicitPlan,
    dst: &mut TensorMap<D, DST_NOUT, DST_NIN, SDst, DDst>,
    canonical_dst: &mut TensorMap<D, DST_CAN_NOUT, DST_CAN_NIN, SDstCan, DDstCan>,
    lhs_canonical: &mut TensorMap<D, LHS_CAN_NOUT, LHS_CAN_NIN, SLhsCan, DLhsCan>,
    rhs_canonical: &mut TensorMap<D, RHS_CAN_NOUT, RHS_CAN_NIN, SRhsCan, DRhsCan>,
    lhs: &TensorMap<D, LHS_NOUT, LHS_NIN, SLhs, DLhs>,
    rhs: &TensorMap<D, RHS_NOUT, RHS_NIN, SRhs, DRhs>,
    alpha: D,
    beta: D,
) -> Result<(), OperationError>
where
    BT: TreeTransformBackend<D, f64>,
    BC: TensorContractBackend<D, f64>,
    R: MultiplicityFreeRigidSymbols<Scalar = f64>,
    D: DenseRecouplingScalar + RecouplingCoefficientAction<f64>,
    DDst: HostWritableStorage<D>,
    DDstCan: HostWritableStorage<D> + HostReadableStorage<D>,
    DLhs: HostReadableStorage<D>,
    DRhs: HostReadableStorage<D>,
    DLhsCan: HostWritableStorage<D> + HostReadableStorage<D>,
    DRhsCan: HostWritableStorage<D> + HostReadableStorage<D>,
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
    tree_pair_transform_into_with_optional_storage_conjugation(
        tree_backend,
        tree_workspace,
        rule,
        plan.lhs_transform().clone(),
        lhs_canonical,
        lhs,
        plan.lhs_source_conjugate(),
        D::one(),
        D::zero(),
    )?;
    tree_pair_transform_into_with_optional_storage_conjugation(
        tree_backend,
        tree_workspace,
        rule,
        plan.rhs_transform().clone(),
        rhs_canonical,
        rhs,
        plan.rhs_source_conjugate(),
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

#[allow(clippy::too_many_arguments)]
fn tree_pair_transform_into_with_optional_storage_conjugation<
    B,
    R,
    D,
    const DST_NOUT: usize,
    const DST_NIN: usize,
    const SRC_NOUT: usize,
    const SRC_NIN: usize,
    SDst,
    SSrc,
    DDst,
    DSrc,
>(
    backend: &mut B,
    workspace: &mut B::Workspace,
    rule: &R,
    operation: crate::TreeTransformOperationKey,
    dst: &mut TensorMap<D, DST_NOUT, DST_NIN, SDst, DDst>,
    src: &TensorMap<D, SRC_NOUT, SRC_NIN, SSrc, DSrc>,
    source_conjugate: bool,
    alpha: D,
    beta: D,
) -> Result<(), OperationError>
where
    B: TreeTransformBackend<D, f64>,
    R: MultiplicityFreeRigidSymbols<Scalar = f64>,
    D: DenseRecouplingScalar + RecouplingCoefficientAction<f64>,
    DDst: HostWritableStorage<D>,
    DSrc: HostReadableStorage<D>,
{
    if !source_conjugate {
        return tree_pair_transform_into_with(
            backend, workspace, rule, operation, dst, src, alpha, beta,
        );
    }

    let src_fusion = src
        .fusion_space()
        .ok_or(OperationError::Core(CoreError::MissingFusionSpace))?;
    let adjoint_src = adjoint_fusion_space_view(src_fusion)?;
    let dst_structure = std::sync::Arc::clone(dst.structure());
    let src_replay_structure = std::sync::Arc::clone(adjoint_src.subblock_structure());
    let plan = build_tree_pair_transform_group_plan(rule, operation, &src_replay_structure)?;
    let structure = plan.compile_structures_with_storage_conjugation(
        &dst_structure,
        &src_replay_structure,
        true,
    )?;
    backend.tree_transform_structure_into_raw(
        workspace,
        &structure,
        &dst_structure,
        &src_replay_structure,
        dst.data_mut(),
        src.data(),
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
    DDst,
    DLhs,
    DRhs,
>(
    backend: &mut B,
    workspace: &mut B::Workspace,
    rule: &R,
    dst: &mut TensorMap<D, DST_NOUT, DST_NIN, SDst, DDst>,
    lhs: &TensorMap<D, LHS_NOUT, LHS_NIN, SLhs, DLhs>,
    rhs: &TensorMap<D, RHS_NOUT, RHS_NIN, SRhs, DRhs>,
    axes: TensorContractAxisSpec<'_>,
    alpha: D,
    beta: D,
) -> Result<(), OperationError>
where
    B: TensorContractBackend<D, f64>,
    R: MultiplicityFreeRigidSymbols<Scalar = f64>,
    D: DenseRecouplingScalar + RecouplingCoefficientAction<f64>,
    DDst: HostWritableStorage<D>,
    DLhs: HostReadableStorage<D>,
    DRhs: HostReadableStorage<D>,
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
    let dst_dynamic = DynamicFusionMapSpace::from_typed(dst_fusion);
    let lhs_dynamic = DynamicFusionMapSpace::from_typed(lhs_fusion);
    let rhs_dynamic = DynamicFusionMapSpace::from_typed(rhs_fusion);
    if !axes.lhs_conjugate()
        && !axes.rhs_conjugate()
        && is_canonical_fusion_block_contract(rule, &dst_dynamic, &lhs_dynamic, &rhs_dynamic, axes)?
    {
        return tensorcontract_canonical_fusion_blocks_into_raw(
            backend,
            workspace,
            rule,
            &dst_dynamic,
            dst.data_mut(),
            &lhs_dynamic,
            lhs.data(),
            &rhs_dynamic,
            rhs.data(),
            axes,
            alpha,
            beta,
        );
    }

    match tensorcontract_fusion_structure(rule, dst, lhs, rhs, axes) {
        Ok(structure) => {
            tensorcontract_execute_with(backend, workspace, &structure, dst, lhs, rhs, alpha, beta)
        }
        Err(OperationError::UnsupportedTensorContractScope {
            message: SOURCE_TRANSFORM_REQUIRES_EXPLICIT,
        }) => tensorcontract_fusion_dynamic_transforms_into_with(
            backend, workspace, rule, dst, lhs, rhs, axes, alpha, beta,
        ),
        Err(err) => Err(err),
    }
}

#[allow(clippy::too_many_arguments)]
pub fn tensorcontract_fusion_into_with_backends<
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
    SDst,
    SLhs,
    SRhs,
    DDst,
    DLhs,
    DRhs,
>(
    tree_backend: &mut BT,
    tree_workspace: &mut BT::Workspace,
    contract_backend: &mut BC,
    contract_workspace: &mut BC::Workspace,
    rule: &R,
    dst: &mut TensorMap<D, DST_NOUT, DST_NIN, SDst, DDst>,
    lhs: &TensorMap<D, LHS_NOUT, LHS_NIN, SLhs, DLhs>,
    rhs: &TensorMap<D, RHS_NOUT, RHS_NIN, SRhs, DRhs>,
    axes: TensorContractAxisSpec<'_>,
    alpha: D,
    beta: D,
) -> Result<(), OperationError>
where
    BT: TreeTransformBackend<D, f64>,
    BC: TensorContractBackend<D, f64>,
    R: MultiplicityFreeRigidSymbols<Scalar = f64>,
    D: DenseRecouplingScalar + RecouplingCoefficientAction<f64>,
    DDst: HostWritableStorage<D>,
    DLhs: HostReadableStorage<D>,
    DRhs: HostReadableStorage<D>,
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
    let dst_dynamic = DynamicFusionMapSpace::from_typed(dst_fusion);
    let lhs_dynamic = DynamicFusionMapSpace::from_typed(lhs_fusion);
    let rhs_dynamic = DynamicFusionMapSpace::from_typed(rhs_fusion);
    if !axes.lhs_conjugate()
        && !axes.rhs_conjugate()
        && is_canonical_fusion_block_contract(rule, &dst_dynamic, &lhs_dynamic, &rhs_dynamic, axes)?
    {
        return tensorcontract_canonical_fusion_blocks_into_raw(
            contract_backend,
            contract_workspace,
            rule,
            &dst_dynamic,
            dst.data_mut(),
            &lhs_dynamic,
            lhs.data(),
            &rhs_dynamic,
            rhs.data(),
            axes,
            alpha,
            beta,
        );
    }

    match tensorcontract_fusion_structure(rule, dst, lhs, rhs, axes) {
        Ok(structure) => tensorcontract_execute_with(
            contract_backend,
            contract_workspace,
            &structure,
            dst,
            lhs,
            rhs,
            alpha,
            beta,
        ),
        Err(OperationError::UnsupportedTensorContractScope {
            message: SOURCE_TRANSFORM_REQUIRES_EXPLICIT,
        }) => {
            let plan = tensorcontract_fusion_explicit_plan(
                rule, dst_fusion, lhs_fusion, rhs_fusion, axes,
            )?;
            tensorcontract_fusion_dynamic_plan_into_with(
                tree_backend,
                tree_workspace,
                contract_backend,
                contract_workspace,
                rule,
                &plan,
                dst,
                lhs,
                rhs,
                alpha,
                beta,
            )
        }
        Err(err) => Err(err),
    }
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
    DDst,
    DLhs,
    DRhs,
>(
    backend: &mut B,
    workspace: &mut B::Workspace,
    structure: &TensorContractStructure<C>,
    dst: &mut TensorMap<D, DST_NOUT, DST_NIN, SDst, DDst>,
    lhs: &TensorMap<D, LHS_NOUT, LHS_NIN, SLhs, DLhs>,
    rhs: &TensorMap<D, RHS_NOUT, RHS_NIN, SRhs, DRhs>,
    alpha: D,
    beta: D,
) -> Result<(), OperationError>
where
    B: TensorContractBackend<D, C>,
    D: DenseBlockScalar + RecouplingCoefficientAction<C>,
    C: Copy + One,
    DDst: HostWritableStorage<D>,
    DLhs: HostReadableStorage<D>,
    DRhs: HostReadableStorage<D>,
{
    structure.execute_with(backend, workspace, dst, lhs, rhs, alpha, beta)
}
