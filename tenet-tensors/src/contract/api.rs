use num_traits::One;
use tenet_core::{
    CoreError, HostReadableStorage, HostWritableStorage, MultiplicityFreeRigidSymbols, TensorMap,
};

use crate::lowering::adjoint_fusion_space_view;
use crate::{
    build_tree_pair_transform_group_plan, tree_transform_into_with,
    tree_transform_overwrite_into_with, DenseBlockScalar, DenseRecouplingScalar,
    DenseTreeTransformOperations, OperationError, RecouplingCoefficientAction,
    TreeTransformBackend, TreeTransformWorkspace,
};
use tenet_operations::{OutputAxisOrder, TensorContractSpec};

use super::backend::{TensorContractBackend, TensorContractWorkspace};
use super::dynamic::{
    tensorcontract_fusion_dynamic_plan_into_with,
    tensorcontract_fusion_dynamic_transforms_into_with,
};
use super::dynamic_space::DynamicFusionMapSpace;
use super::fusion::{
    prepare_tensorcontract_fusion_plan, tensorcontract_fusion_structure, FusionContractPlan,
    EXPLICIT_OUTPUT_TRANSFORM_REQUIRES_CORE_DST, SOURCE_TRANSFORM_REQUIRES_EXPLICIT,
};
use super::fusion_block::{
    compile_fusion_block_contract_plan_validated, is_core_form_fusion_block_contract,
    tensorcontract_core_fusion_blocks_with_plan_into_raw, validate_fusion_contract_rule,
};
use super::resolution::rhs_contract_requires_twist;
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
    axes: TensorContractSpec<'_>,
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
    output_permutation: OutputAxisOrder<'_>,
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
        TensorContractSpec::new(&[], &[], output_permutation),
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
    output_permutation: OutputAxisOrder<'_>,
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
        TensorContractSpec::new_with_conjugation(
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
    axes: TensorContractSpec<'_>,
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
    axes: TensorContractSpec<'_>,
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
    output_permutation: OutputAxisOrder<'_>,
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
        TensorContractSpec::new(&[], &[], output_permutation),
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
    output_permutation: OutputAxisOrder<'_>,
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
        TensorContractSpec::new_with_conjugation(
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
/// not already in core compose form. The caller provides the core
/// temporary tensors because their ranks are determined by the chosen
/// contraction axes and therefore cannot be constructed generically from the
/// original const-generic tensor ranks.
///
/// The sequence is:
///
/// 1. transform `lhs` to `(lhs open) <- (lhs contracted)`;
/// 2. transform `rhs` to `(rhs contracted) <- (rhs open)`;
/// 3. run the fusion contraction from those core operands into `dst`,
///    which must already be the core output tree-pair shape.
///
/// Use [`tensorcontract_fusion_prepared_into_core_dst`] when the
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
    lhs_core: &mut TensorMap<D, LHS_CAN_NOUT, LHS_CAN_NIN, SLhsCan, DLhsCan>,
    rhs_core: &mut TensorMap<D, RHS_CAN_NOUT, RHS_CAN_NIN, SRhsCan, DRhsCan>,
    lhs: &TensorMap<D, LHS_NOUT, LHS_NIN, SLhs, DLhs>,
    rhs: &TensorMap<D, RHS_NOUT, RHS_NIN, SRhs, DRhs>,
    axes: TensorContractSpec<'_>,
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
    let plan = prepare_tensorcontract_fusion_plan(
        rule,
        dst.fusion_space()
            .ok_or(OperationError::Core(CoreError::MissingFusionSpace))?,
        lhs.fusion_space()
            .ok_or(OperationError::Core(CoreError::MissingFusionSpace))?,
        rhs.fusion_space()
            .ok_or(OperationError::Core(CoreError::MissingFusionSpace))?,
        axes,
    )?;
    tensorcontract_fusion_prepared_into(rule, &plan, dst, lhs_core, rhs_core, lhs, rhs, alpha, beta)
}

pub fn tensorcontract_fusion_prepared_into<
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
    plan: &FusionContractPlan,
    dst: &mut TensorMap<D, DST_NOUT, DST_NIN, SDst, DDst>,
    lhs_core: &mut TensorMap<D, LHS_CAN_NOUT, LHS_CAN_NIN, SLhsCan, DLhsCan>,
    rhs_core: &mut TensorMap<D, RHS_CAN_NOUT, RHS_CAN_NIN, SRhsCan, DRhsCan>,
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
    tensorcontract_fusion_prepared_into_with(
        &mut tree_backend,
        &mut tree_workspace,
        &mut contract_backend,
        &mut contract_workspace,
        rule,
        plan,
        dst,
        lhs_core,
        rhs_core,
        lhs,
        rhs,
        alpha,
        beta,
    )
}

pub fn tensorcontract_fusion_prepared_into_core_dst<
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
    plan: &FusionContractPlan,
    dst: &mut TensorMap<D, DST_NOUT, DST_NIN, SDst, DDst>,
    core_dst: &mut TensorMap<D, DST_CAN_NOUT, DST_CAN_NIN, SDstCan, DDstCan>,
    lhs_core: &mut TensorMap<D, LHS_CAN_NOUT, LHS_CAN_NIN, SLhsCan, DLhsCan>,
    rhs_core: &mut TensorMap<D, RHS_CAN_NOUT, RHS_CAN_NIN, SRhsCan, DRhsCan>,
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
    tensorcontract_fusion_prepared_into_core_dst_with(
        &mut tree_backend,
        &mut tree_workspace,
        &mut contract_backend,
        &mut contract_workspace,
        rule,
        plan,
        dst,
        core_dst,
        lhs_core,
        rhs_core,
        lhs,
        rhs,
        alpha,
        beta,
    )
}

pub fn tensorcontract_fusion_prepared_into_with<
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
    plan: &FusionContractPlan,
    dst: &mut TensorMap<D, DST_NOUT, DST_NIN, SDst, DDst>,
    lhs_core: &mut TensorMap<D, LHS_CAN_NOUT, LHS_CAN_NIN, SLhsCan, DLhsCan>,
    rhs_core: &mut TensorMap<D, RHS_CAN_NOUT, RHS_CAN_NIN, SRhsCan, DRhsCan>,
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
        || DST_NOUT != plan.core_dst_open_lhs_rank()
        || DST_NIN != plan.core_dst_open_rhs_rank()
    {
        return Err(OperationError::UnsupportedTensorContractScope {
            message: EXPLICIT_OUTPUT_TRANSFORM_REQUIRES_CORE_DST,
        });
    }
    if LHS_CAN_NOUT != plan.lhs_open_rank() || LHS_CAN_NIN != plan.lhs_contract_rank() {
        return Err(OperationError::StructureRankMismatch {
            expected: plan.lhs_open_rank() + plan.lhs_contract_rank(),
            actual: LHS_CAN_NOUT + LHS_CAN_NIN,
        });
    }
    if RHS_CAN_NOUT != plan.rhs_contract_rank() || RHS_CAN_NIN != plan.rhs_open_rank() {
        return Err(OperationError::StructureRankMismatch {
            expected: plan.rhs_contract_rank() + plan.rhs_open_rank(),
            actual: RHS_CAN_NOUT + RHS_CAN_NIN,
        });
    }

    tree_transform_overwrite_with_optional_storage_conjugation(
        tree_backend,
        tree_workspace,
        rule,
        plan.lhs_transform().clone(),
        lhs_core,
        lhs,
        plan.lhs_source_conjugate(),
        D::one(),
    )?;
    tree_transform_overwrite_with_optional_storage_conjugation(
        tree_backend,
        tree_workspace,
        rule,
        plan.rhs_transform().clone(),
        rhs_core,
        rhs,
        plan.rhs_source_conjugate(),
        D::one(),
    )?;

    tensorcontract_fusion_into_with(
        contract_backend,
        contract_workspace,
        rule,
        dst,
        lhs_core,
        rhs_core,
        plan.core_axes().as_spec(),
        alpha,
        beta,
    )
}

pub fn tensorcontract_fusion_prepared_into_core_dst_with<
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
    plan: &FusionContractPlan,
    dst: &mut TensorMap<D, DST_NOUT, DST_NIN, SDst, DDst>,
    core_dst: &mut TensorMap<D, DST_CAN_NOUT, DST_CAN_NIN, SDstCan, DDstCan>,
    lhs_core: &mut TensorMap<D, LHS_CAN_NOUT, LHS_CAN_NIN, SLhsCan, DLhsCan>,
    rhs_core: &mut TensorMap<D, RHS_CAN_NOUT, RHS_CAN_NIN, SRhsCan, DRhsCan>,
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
    if DST_CAN_NOUT != plan.core_dst_open_lhs_rank() || DST_CAN_NIN != plan.core_dst_open_rhs_rank()
    {
        return Err(OperationError::StructureRankMismatch {
            expected: plan.core_dst_open_lhs_rank() + plan.core_dst_open_rhs_rank(),
            actual: DST_CAN_NOUT + DST_CAN_NIN,
        });
    }
    if LHS_CAN_NOUT != plan.lhs_open_rank() || LHS_CAN_NIN != plan.lhs_contract_rank() {
        return Err(OperationError::StructureRankMismatch {
            expected: plan.lhs_open_rank() + plan.lhs_contract_rank(),
            actual: LHS_CAN_NOUT + LHS_CAN_NIN,
        });
    }
    if RHS_CAN_NOUT != plan.rhs_contract_rank() || RHS_CAN_NIN != plan.rhs_open_rank() {
        return Err(OperationError::StructureRankMismatch {
            expected: plan.rhs_contract_rank() + plan.rhs_open_rank(),
            actual: RHS_CAN_NOUT + RHS_CAN_NIN,
        });
    }

    core_dst.data_mut().fill(D::zero());
    tree_transform_overwrite_with_optional_storage_conjugation(
        tree_backend,
        tree_workspace,
        rule,
        plan.lhs_transform().clone(),
        lhs_core,
        lhs,
        plan.lhs_source_conjugate(),
        D::one(),
    )?;
    tree_transform_overwrite_with_optional_storage_conjugation(
        tree_backend,
        tree_workspace,
        rule,
        plan.rhs_transform().clone(),
        rhs_core,
        rhs,
        plan.rhs_source_conjugate(),
        D::one(),
    )?;

    tensorcontract_fusion_into_with(
        contract_backend,
        contract_workspace,
        rule,
        core_dst,
        lhs_core,
        rhs_core,
        plan.core_axes().as_spec(),
        alpha,
        D::zero(),
    )?;

    tree_transform_into_with(
        tree_backend,
        tree_workspace,
        rule,
        plan.output_transform().clone(),
        dst,
        core_dst,
        D::one(),
        beta,
    )
}

#[allow(clippy::too_many_arguments)]
fn tree_transform_overwrite_with_optional_storage_conjugation<
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
    operation: crate::TreeTransformOperation,
    dst: &mut TensorMap<D, DST_NOUT, DST_NIN, SDst, DDst>,
    src: &TensorMap<D, SRC_NOUT, SRC_NIN, SSrc, DSrc>,
    source_conjugate: bool,
    alpha: D,
) -> Result<(), OperationError>
where
    B: TreeTransformBackend<D, f64>,
    R: MultiplicityFreeRigidSymbols<Scalar = f64>,
    D: DenseRecouplingScalar + RecouplingCoefficientAction<f64>,
    DDst: HostWritableStorage<D>,
    DSrc: HostReadableStorage<D>,
{
    // Why not use the generic beta=0 route: this destination is private core
    // scratch and the explicit overwrite replay already owns every logical element.
    if !source_conjugate {
        return tree_transform_overwrite_into_with(
            backend, workspace, rule, operation, dst, src, alpha,
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
    backend.tree_transform_structure_overwrite_into_raw(
        workspace,
        &structure,
        &dst_structure,
        &src_replay_structure,
        dst.data_mut(),
        src.data(),
        alpha,
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
    axes: TensorContractSpec<'_>,
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
    validate_fusion_contract_rule(rule, &dst_dynamic, &lhs_dynamic, &rhs_dynamic)?;
    if !axes.lhs_conjugate()
        && !axes.rhs_conjugate()
        && is_core_form_fusion_block_contract(rule, &dst_dynamic, &lhs_dynamic, &rhs_dynamic, axes)?
        && !rhs_contract_requires_twist(rule, &rhs_dynamic, axes)?
    {
        let plan = compile_fusion_block_contract_plan_validated(
            rule,
            &dst_dynamic,
            &lhs_dynamic,
            &rhs_dynamic,
            axes,
        )?;
        if plan.is_fully_direct() {
            return tensorcontract_core_fusion_blocks_with_plan_into_raw(
                &mut crate::StridedHostKernelAdapter::with_transpose_backend(
                    backend.transpose_backend(),
                ),
                backend,
                workspace,
                &plan,
                &dst_dynamic,
                dst.data_mut(),
                &lhs_dynamic,
                lhs.data(),
                &rhs_dynamic,
                rhs.data(),
                alpha,
                beta,
            );
        }
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
    axes: TensorContractSpec<'_>,
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
    validate_fusion_contract_rule(rule, &dst_dynamic, &lhs_dynamic, &rhs_dynamic)?;
    if !axes.lhs_conjugate()
        && !axes.rhs_conjugate()
        && is_core_form_fusion_block_contract(rule, &dst_dynamic, &lhs_dynamic, &rhs_dynamic, axes)?
        && !rhs_contract_requires_twist(rule, &rhs_dynamic, axes)?
    {
        let plan = compile_fusion_block_contract_plan_validated(
            rule,
            &dst_dynamic,
            &lhs_dynamic,
            &rhs_dynamic,
            axes,
        )?;
        if plan.is_fully_direct() {
            return tensorcontract_core_fusion_blocks_with_plan_into_raw(
                &mut crate::StridedHostKernelAdapter::with_transpose_backend(
                    contract_backend.transpose_backend(),
                ),
                contract_backend,
                contract_workspace,
                &plan,
                &dst_dynamic,
                dst.data_mut(),
                &lhs_dynamic,
                lhs.data(),
                &rhs_dynamic,
                rhs.data(),
                alpha,
                beta,
            );
        }
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
            let plan =
                prepare_tensorcontract_fusion_plan(rule, dst_fusion, lhs_fusion, rhs_fusion, axes)?;
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
