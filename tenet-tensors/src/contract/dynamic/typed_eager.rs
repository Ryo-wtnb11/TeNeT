use super::*;

use tenet_core::{CoreError, HostReadableStorage, HostWritableStorage, TensorMap, TensorStorage};
use tenet_operations::TensorContractSpec;

use super::super::fusion_block::tensorcontract_core_fusion_blocks_into_raw;
use crate::lowering::adjoint_fusion_space_view;
use crate::tree_transform::build_tree_pair_transform_group_plan;

#[allow(clippy::too_many_arguments)]
pub(crate) fn tensorcontract_fusion_dynamic_plan_into_with<
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
    plan: &FusionContractPlan,
    dst: &mut TensorMap<D, DST_NOUT, DST_NIN, SDst, DDst>,
    lhs: &TensorMap<D, LHS_NOUT, LHS_NIN, SLhs, DLhs>,
    rhs: &TensorMap<D, RHS_NOUT, RHS_NIN, SRhs, DRhs>,
    alpha: D,
    beta: D,
) -> Result<(), OperationError>
where
    BT: TreeTransformBackend<D, R::Scalar>,
    BC: TensorContractBackend<D, R::Scalar>,
    R: MultiplicityFreeRigidSymbols,
    R::Scalar: DenseBlockScalar,
    D: DenseRecouplingScalar + RecouplingCoefficientAction<R::Scalar>,
    DDst: HostWritableStorage<D>,
    DLhs: HostReadableStorage<D>,
    DRhs: HostReadableStorage<D>,
{
    let reverse = plan.orientation() == FusionContractOrientation::RhsLhs;
    let lhs_source_space = DynamicFusionMapSpace::from_typed(
        lhs.fusion_space()
            .ok_or(OperationError::Core(CoreError::MissingFusionSpace))?,
    );
    let lhs_transformed = transformed_source_space_and_structure(
        rule,
        lhs,
        plan.lhs_transform(),
        plan.lhs_source_conjugate(),
    )?;
    let lhs_layout_borrowable = source_is_borrowable_core_layout(
        &lhs_source_space,
        lhs.structure(),
        &lhs_transformed.0,
        plan.lhs_transform(),
        plan.lhs_source_conjugate(),
    );
    let (rhs_space, rhs_replay_structure) = transformed_source_space_and_structure(
        rule,
        rhs,
        plan.rhs_transform(),
        plan.rhs_source_conjugate(),
    )?;
    let rhs_source_space = DynamicFusionMapSpace::from_typed(
        rhs.fusion_space()
            .ok_or(OperationError::Core(CoreError::MissingFusionSpace))?,
    );
    let rhs_layout_borrowable = source_is_borrowable_core_layout(
        &rhs_source_space,
        rhs.structure(),
        &rhs_space,
        plan.rhs_transform(),
        plan.rhs_source_conjugate(),
    );
    let SourceBorrowing {
        lhs_borrowed,
        rhs_borrowed,
        twist_lhs,
    } = resolve_source_borrowing(
        rule,
        plan,
        &lhs_transformed.0,
        &rhs_space,
        lhs_layout_borrowable,
        rhs_layout_borrowable,
    )?;
    let core_right_space = if reverse {
        &lhs_transformed.0
    } else {
        &rhs_space
    };
    let core_right_homspace = core_right_space.homspace();
    let mut lhs_core = (!lhs_borrowed)
        .then(|| DynamicFusionScratch::<D>::zeroed(Arc::new(lhs_transformed.0.clone())))
        .transpose()?;
    let mut rhs_core = (!rhs_borrowed)
        .then(|| DynamicFusionScratch::<D>::zeroed(Arc::new(rhs_space.clone())))
        .transpose()?;

    if let Some(lhs_core) = lhs_core.as_mut() {
        tree_pair_transform_typed_to_dynamic(
            tree_backend,
            tree_workspace,
            rule,
            plan.lhs_transform().clone(),
            lhs_core,
            lhs,
            &lhs_transformed.1,
            plan.lhs_source_conjugate(),
            D::one(),
        )?;
        if twist_lhs {
            let lhs_scratch_space = lhs_core.space().clone();
            apply_contract_twist(
                &mut crate::StridedHostKernelAdapter::default(),
                rule,
                &lhs_scratch_space,
                core_right_homspace,
                !reverse,
                lhs_core.data_mut(),
                plan.core_axes().as_spec().rhs_contracting_axes(),
            )?;
        }
    }
    if let Some(rhs_core) = rhs_core.as_mut() {
        tree_pair_transform_typed_to_dynamic(
            tree_backend,
            tree_workspace,
            rule,
            plan.rhs_transform().clone(),
            rhs_core,
            rhs,
            &rhs_replay_structure,
            plan.rhs_source_conjugate(),
            D::one(),
        )?;
        if !twist_lhs {
            let rhs_scratch_space = rhs_core.space().clone();
            apply_contract_twist(
                &mut crate::StridedHostKernelAdapter::default(),
                rule,
                &rhs_scratch_space,
                core_right_homspace,
                reverse,
                rhs_core.data_mut(),
                plan.core_axes().as_spec().rhs_contracting_axes(),
            )?;
        }
    }

    let physical_lhs_core = match lhs_core.as_ref() {
        Some(scratch) => CoreSource::from_host_scratch(scratch),
        None => CoreSource::borrowed(&lhs_transformed.0, lhs.data()),
    };
    let physical_rhs_core = select_core_source(rhs_borrowed, &rhs_space, rhs.data(), || {
        CoreSource::from_host_scratch(
            rhs_core
                .as_ref()
                .expect("non-borrowed RHS materialized before core contraction"),
        )
    });
    let (lhs_core, rhs_core_view) = if reverse {
        (physical_rhs_core, physical_lhs_core)
    } else {
        (physical_lhs_core, physical_rhs_core)
    };

    if plan.output_transform_is_identity() {
        let dst_space = DynamicFusionMapSpace::from_typed(
            dst.fusion_space()
                .ok_or(OperationError::Core(CoreError::MissingFusionSpace))?,
        );
        let dst_structure = std::sync::Arc::clone(dst.structure());
        return tensorcontract_dynamic_core_into_raw(
            contract_backend,
            contract_workspace,
            rule,
            &dst_space,
            &dst_structure,
            dst.data_mut(),
            lhs_core,
            rhs_core_view,
            plan.core_axes().as_spec(),
            alpha,
            beta,
        );
    }

    let core_dst_space =
        DynamicFusionMapSpace::core_dst(rule, lhs_core.space(), rhs_core_view.space(), plan)?;
    let mut core_dst = DynamicFusionScratch::<D>::zeroed(Arc::new(core_dst_space))?;
    let core_dst_space_for_contract = core_dst.space().clone();
    let core_dst_structure = std::sync::Arc::clone(core_dst.space().structure());
    tensorcontract_dynamic_core_into_raw(
        contract_backend,
        contract_workspace,
        rule,
        &core_dst_space_for_contract,
        &core_dst_structure,
        core_dst.data_mut(),
        lhs_core,
        rhs_core_view,
        plan.core_axes().as_spec(),
        alpha,
        D::zero(),
    )?;
    tree_pair_transform_dynamic_to_typed(
        tree_backend,
        tree_workspace,
        rule,
        plan.output_transform().clone(),
        dst,
        &core_dst,
        D::one(),
        beta,
    )
}

fn transformed_source_space_and_structure<
    R,
    D,
    const SRC_NOUT: usize,
    const SRC_NIN: usize,
    SSrc,
    DSrc,
>(
    rule: &R,
    src: &TensorMap<D, SRC_NOUT, SRC_NIN, SSrc, DSrc>,
    operation: &TreeTransformOperation,
    source_conjugate: bool,
) -> Result<(DynamicFusionMapSpace, std::sync::Arc<BlockStructure>), OperationError>
where
    R: MultiplicityFreeRigidSymbols,
    DSrc: TensorStorage<D>,
{
    let src_fusion = src
        .fusion_space()
        .ok_or(OperationError::Core(CoreError::MissingFusionSpace))?;
    if source_conjugate {
        let adjoint = adjoint_fusion_space_view(rule, src_fusion)?;
        let replay_structure = std::sync::Arc::clone(adjoint.subblock_structure());
        let space = DynamicFusionMapSpace::transformed_from_typed(rule, &adjoint, operation)?;
        Ok((space, replay_structure))
    } else {
        let replay_structure = std::sync::Arc::clone(src.structure());
        let space = DynamicFusionMapSpace::transformed_from_typed(rule, src_fusion, operation)?;
        Ok((space, replay_structure))
    }
}

#[expect(
    clippy::too_many_arguments,
    reason = "the typed-to-dynamic replay boundary keeps backend resources, transform, structures, buffers, conjugation, and alpha explicit"
)]
fn tree_pair_transform_typed_to_dynamic<
    BT,
    R,
    D,
    const SRC_NOUT: usize,
    const SRC_NIN: usize,
    SSrc,
    DSrc,
>(
    tree_backend: &mut BT,
    tree_workspace: &mut BT::Workspace,
    rule: &R,
    operation: TreeTransformOperation,
    dst: &mut DynamicFusionScratch<D>,
    src: &TensorMap<D, SRC_NOUT, SRC_NIN, SSrc, DSrc>,
    src_replay_structure: &std::sync::Arc<BlockStructure>,
    source_conjugate: bool,
    alpha: D,
) -> Result<(), OperationError>
where
    BT: TreeTransformBackend<D, R::Scalar>,
    R: MultiplicityFreeRigidSymbols,
    R::Scalar: DenseBlockScalar,
    D: DenseRecouplingScalar + RecouplingCoefficientAction<R::Scalar>,
    DSrc: HostReadableStorage<D>,
{
    let plan = build_tree_pair_transform_group_plan(rule, operation, src_replay_structure)?;
    let structure = plan.compile_structures_with_storage_conjugation(
        dst.space().structure(),
        src_replay_structure,
        source_conjugate,
    )?;
    let dst_structure = std::sync::Arc::clone(dst.space().structure());
    tree_backend.tree_transform_structure_overwrite_into_raw(
        tree_workspace,
        &structure,
        &dst_structure,
        src_replay_structure,
        dst.data_mut(),
        src.data(),
        alpha,
        &[],
    )
}

#[expect(
    clippy::too_many_arguments,
    reason = "the dynamic-to-typed replay boundary keeps backend resources, transform, operands, and alpha/beta explicit"
)]
fn tree_pair_transform_dynamic_to_typed<
    BT,
    R,
    D,
    const DST_NOUT: usize,
    const DST_NIN: usize,
    SDst,
    DDst,
>(
    tree_backend: &mut BT,
    tree_workspace: &mut BT::Workspace,
    rule: &R,
    operation: TreeTransformOperation,
    dst: &mut TensorMap<D, DST_NOUT, DST_NIN, SDst, DDst>,
    src: &DynamicFusionScratch<D>,
    alpha: D,
    beta: D,
) -> Result<(), OperationError>
where
    BT: TreeTransformBackend<D, R::Scalar>,
    R: MultiplicityFreeRigidSymbols,
    R::Scalar: DenseBlockScalar,
    D: DenseRecouplingScalar + RecouplingCoefficientAction<R::Scalar>,
    DDst: HostWritableStorage<D>,
{
    let plan = build_tree_pair_transform_group_plan(rule, operation, src.space().structure())?;
    let structure = plan.compile_structures(dst.structure(), src.space().structure())?;
    let dst_structure = std::sync::Arc::clone(dst.structure());
    let src_structure = std::sync::Arc::clone(src.space().structure());
    tree_backend.tree_transform_structure_into_raw(
        tree_workspace,
        &structure,
        &dst_structure,
        &src_structure,
        dst.data_mut(),
        src.data(),
        alpha,
        beta,
    )
}

#[allow(clippy::too_many_arguments)]
fn tensorcontract_dynamic_core_into_raw<B, R, D>(
    backend: &mut B,
    workspace: &mut B::Workspace,
    rule: &R,
    dst_space: &DynamicFusionMapSpace,
    dst_structure: &std::sync::Arc<BlockStructure>,
    dst_data: &mut [D],
    lhs: CoreSource<'_, D>,
    rhs: CoreSource<'_, D>,
    axes: TensorContractSpec<'_>,
    alpha: D,
    beta: D,
) -> Result<(), OperationError>
where
    B: TensorContractBackend<D, R::Scalar>,
    R: MultiplicityFreeRigidSymbols,
    R::Scalar: DenseBlockScalar,
    D: DenseRecouplingScalar + RecouplingCoefficientAction<R::Scalar>,
{
    let _ = dst_structure;
    tensorcontract_core_fusion_blocks_into_raw(
        &mut crate::StridedHostKernelAdapter::default(),
        backend,
        workspace,
        rule,
        dst_space,
        dst_data,
        lhs.space(),
        lhs.data(),
        rhs.space(),
        rhs.data(),
        axes,
        alpha,
        beta,
    )
}
