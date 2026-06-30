use num_traits::One;
use tenet_core::{CoreError, MultiplicityFreeRigidSymbols, TensorMap};

use crate::axis::TensorContractAxisSpec;
use crate::{
    tree_pair_transform_into_with, DenseBlockScalar, DenseRecouplingScalar,
    DenseTreeTransformOperations, OperationError, RecouplingCoefficientAction,
    TreeTransformBackend, TreeTransformWorkspace,
};

use super::backend::{TensorContractBackend, TensorContractWorkspace};
use super::fusion::{
    tensorcontract_fusion_explicit_plan, tensorcontract_fusion_structure,
    TensorContractFusionExplicitPlan, EXPLICIT_OUTPUT_TRANSFORM_REQUIRES_CANONICAL_DST,
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
