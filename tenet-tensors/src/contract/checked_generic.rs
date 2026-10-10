use std::sync::Arc;

use num_traits::Zero;
use tenet_core::{
    CheckedGenericAdmissionMode, CheckedGenericRigidSymbols, CoreError, FusionTreeHomSpace,
    RuleIdentity, StructurallyValidatedFusionTreeSubset,
};
#[cfg(test)]
use tenet_operations::DenseTreeTransformOperations;
use tenet_operations::{ContractDestinationInit, TensorContractSpec, TreeTransformBackend};

use crate::mode::{PlanningAlgebra, TreeStructureSource};
use crate::tree_transform::{CheckedGenericPlanError, CheckedPendingCoefficients};
use crate::{
    zeroed_payload, ConjugateValue, DenseRecouplingScalar, OperationError,
    RecouplingCoefficientAction, ZeroBytes,
};

use super::context::{plan_compose_in, PlanTarget, TensorContractFusionExecutionContext};
use super::dynamic_space::{
    BoundDynamicFusionMapSpace, FusionOperand, PreparedCheckedGenericDynamicSpace,
};
use super::fusion::{
    compile_tensorcontract_fusion_plan_from_ranks, orient_fusion_contract_plan,
    select_complete_bosonic_contract_candidate, ContractAxisOrderCandidate,
    FusionContractOrientation,
};
use super::fusion_block::{
    compile_checked_generic_core_plan, BackendRank2Gemm, FusionBlockContractWorkspace, Rank2Gemm,
};
use super::resolution::HostEagerExecutor;
use super::route_host::Stage;
use super::structure::TensorContractAxisPlan;

type CheckedContractResult<P, D> = Result<
    (BoundDynamicFusionMapSpace<P>, Vec<D>),
    CheckedGenericPlanError<<P as tenet_core::CheckedGenericFusion>::Error>,
>;
// Why not box the transformed arm: this value is operation-local and boxing
// would add a heap allocation to every nonidentity source transform.
#[allow(clippy::large_enum_variant)]
enum CheckedStagedOperand<'a> {
    Borrowed(&'a super::DynamicFusionMapSpace),
    Transformed {
        prepared: PreparedCheckedGenericDynamicSpace,
        replay: tenet_operations::TreeTransformStructure<f64>,
        structure: Arc<tenet_core::BlockStructure>,
    },
}

impl CheckedStagedOperand<'_> {
    fn homspace(&self) -> &FusionTreeHomSpace {
        match self {
            Self::Borrowed(space) => space.homspace(),
            Self::Transformed { prepared, .. } => prepared.homspace(),
        }
    }

    fn nout(&self) -> usize {
        match self {
            Self::Borrowed(space) => space.nout(),
            Self::Transformed { prepared, .. } => prepared.nout(),
        }
    }

    fn structure(&self) -> &Arc<tenet_core::BlockStructure> {
        match self {
            Self::Borrowed(space) => space.structure(),
            Self::Transformed { structure, .. } => structure,
        }
    }

    /// `(preview, committed)`; a borrowed operand is already committed.
    fn commit(
        self,
    ) -> (
        Arc<tenet_core::BlockStructure>,
        Arc<tenet_core::BlockStructure>,
    ) {
        match self {
            Self::Borrowed(space) => (Arc::clone(space.structure()), Arc::clone(space.structure())),
            Self::Transformed {
                prepared,
                structure,
                ..
            } => (structure, prepared.commit_structure()),
        }
    }
}

#[cfg(test)]
fn same_axes(lhs: &[usize], rhs: &[usize]) -> bool {
    let mut lhs = lhs.to_vec();
    let mut rhs = rhs.to_vec();
    lhs.sort_unstable();
    rhs.sort_unstable();
    lhs == rhs
}

fn validate_source_structure<E>(
    space: &super::DynamicFusionMapSpace,
) -> Result<(), CheckedGenericPlanError<E>> {
    StructurallyValidatedFusionTreeSubset::try_new(space.homspace(), space.structure())?;
    Ok(())
}

struct CheckedContractLocal {
    output_rank: usize,
    axis_plan: TensorContractAxisPlan,
}

fn validate_contract_local<P, D>(
    lhs_space: &BoundDynamicFusionMapSpace<P>,
    lhs_data: &[D],
    rhs_space: &BoundDynamicFusionMapSpace<P>,
    rhs_data: &[D],
    axes: TensorContractSpec<'_>,
    dst_nout: usize,
) -> Result<CheckedContractLocal, CheckedGenericPlanError<P::Error>>
where
    P: CheckedGenericRigidSymbols<Scalar = f64>,
{
    if axes.lhs_conjugate() || axes.rhs_conjugate() {
        return Err(OperationError::UnsupportedTensorContractScope {
            message: "checked Generic contraction currently requires eager direct operands",
        }
        .into());
    }
    let output_rank = lhs_space
        .space()
        .rank()
        .checked_sub(axes.lhs_contracting_axes().len())
        .and_then(|rank| {
            rhs_space
                .space()
                .rank()
                .checked_sub(axes.rhs_contracting_axes().len())
                .and_then(|rhs| rank.checked_add(rhs))
        })
        .ok_or(OperationError::ElementCountOverflow)?;
    let axis_plan = TensorContractAxisPlan::compile(
        lhs_space.space().rank(),
        rhs_space.space().rank(),
        output_rank,
        axes,
    )?;
    if dst_nout > output_rank {
        return Err(CoreError::StructureRankMismatch {
            expected: output_rank,
            actual: dst_nout,
        }
        .into());
    }
    for (data, space) in [(lhs_data, lhs_space), (rhs_data, rhs_space)] {
        let expected = space.space().required_len()?;
        if data.len() != expected {
            return Err(OperationError::ElementCountMismatch {
                expected,
                actual: data.len(),
            }
            .into());
        }
        validate_source_structure::<P::Error>(space.space())?;
    }
    Ok(CheckedContractLocal {
        output_rank,
        axis_plan,
    })
}

fn staged_transform<'a, P>(
    coefficients: &mut CheckedPendingCoefficients,
    authority: &BoundDynamicFusionMapSpace<P>,
    provider: &P,
    source: &'a BoundDynamicFusionMapSpace<P>,
    operation: &crate::TreeTransformOperation,
) -> Result<CheckedStagedOperand<'a>, CheckedGenericPlanError<P::Error>>
where
    P: CheckedGenericRigidSymbols<Scalar = f64>,
{
    if operation.is_identity_for(source.space().nout(), source.space().nin()) {
        return Ok(CheckedStagedOperand::Borrowed(source.space()));
    }
    let prepared = authority.prepare_final_homspace_generic_from_checked(provider, || {
        source
            .space()
            .homspace()
            .try_permute_generic_checked(
                provider,
                operation.codomain_permutation(),
                operation.domain_permutation(),
            )
            .map_err(CheckedGenericPlanError::from)
    })?;
    let destination = prepared.shared_structure();
    let replay = <CheckedGenericAdmissionMode as PlanningAlgebra<P>>::tree_structure(
        coefficients,
        provider,
        operation,
        &destination,
        TreeStructureSource::Stored {
            structure: source.space().structure(),
            storage_conjugate: false,
        },
    )?;
    Ok(CheckedStagedOperand::Transformed {
        prepared,
        replay,
        structure: destination,
    })
}

fn execute_transform<D, B>(
    backend: &mut B,
    workspace: &mut B::Workspace,
    replay: &tenet_operations::TreeTransformStructure<f64>,
    destination: &Arc<tenet_core::BlockStructure>,
    source: &Arc<tenet_core::BlockStructure>,
    destination_data: &mut [D],
    source_data: &[D],
) -> Result<(), OperationError>
where
    D: DenseRecouplingScalar + RecouplingCoefficientAction<f64>,
    B: TreeTransformBackend<D, f64>,
{
    backend.tree_transform_structure_into_raw(
        workspace,
        replay,
        destination,
        source,
        destination_data,
        source_data,
        D::one(),
        D::zero(),
    )
}

fn execute_staged_transform<D, B>(
    backend: &mut B,
    workspace: &mut B::Workspace,
    staged: &CheckedStagedOperand<'_>,
    source: &Arc<tenet_core::BlockStructure>,
    source_data: &[D],
) -> Result<Option<Vec<D>>, OperationError>
where
    D: DenseRecouplingScalar + RecouplingCoefficientAction<f64> + Copy + Zero + ZeroBytes,
    B: TreeTransformBackend<D, f64>,
{
    let CheckedStagedOperand::Transformed {
        prepared,
        replay,
        structure,
    } = staged
    else {
        return Ok(None);
    };
    let mut data = zeroed_payload(prepared.required_len());
    execute_transform(
        backend,
        workspace,
        replay,
        structure,
        source,
        &mut data,
        source_data,
    )?;
    Ok(Some(data))
}

#[cfg(test)]
/// Contracts two direct checked Generic tensors using the shared stable
/// candidate policy. The result retains the left provider allocation.
#[doc(hidden)]
pub fn tensorcontract_owned_checked_generic<P, D>(
    lhs_space: &BoundDynamicFusionMapSpace<P>,
    lhs_data: &[D],
    rhs_space: &BoundDynamicFusionMapSpace<P>,
    rhs_data: &[D],
    axes: TensorContractSpec<'_>,
) -> CheckedContractResult<P, D>
where
    P: CheckedGenericRigidSymbols<Scalar = f64>,
    D: DenseRecouplingScalar
        + RecouplingCoefficientAction<f64>
        + ConjugateValue
        + Copy
        + Zero
        + ZeroBytes,
{
    let mut transform_backend = DenseTreeTransformOperations::default();
    let mut transform_workspaces = Default::default();
    let mut contract_backend = DenseTreeTransformOperations::default();
    let mut contract_workspace = Default::default();
    let mut fusion_workspace = FusionBlockContractWorkspace::default();
    tensorcontract_owned_checked_generic_with_resources(
        lhs_space,
        lhs_data,
        rhs_space,
        rhs_data,
        axes,
        None,
        &mut transform_backend,
        &mut transform_workspaces,
        &mut BackendRank2Gemm::<_, _, f64>::new(&mut contract_backend, &mut contract_workspace),
        &mut fusion_workspace,
    )
}

/// Runtime-context variant of [`tensorcontract_owned_checked_generic`], with
/// the result split after its first `codomain_rank` output axes
/// (TensorOperations `pAB`).
#[doc(hidden)]
pub fn tensorcontract_owned_checked_generic_in_context<P, D>(
    context: &mut TensorContractFusionExecutionContext<D, RuleIdentity>,
    lhs_space: &BoundDynamicFusionMapSpace<P>,
    lhs_data: &[D],
    rhs_space: &BoundDynamicFusionMapSpace<P>,
    rhs_data: &[D],
    axes: TensorContractSpec<'_>,
    codomain_rank: usize,
) -> CheckedContractResult<P, D>
where
    P: CheckedGenericRigidSymbols<Scalar = f64>,
    D: DenseRecouplingScalar
        + RecouplingCoefficientAction<f64>
        + ConjugateValue
        + Copy
        + Zero
        + ZeroBytes,
{
    let (
        transform_backend,
        transform_workspaces,
        contract_backend,
        contract_workspace,
        fusion_workspace,
    ) = context.checked_generic_resources_mut();
    tensorcontract_owned_checked_generic_with_resources(
        lhs_space,
        lhs_data,
        rhs_space,
        rhs_data,
        axes,
        Some(codomain_rank),
        transform_backend,
        transform_workspaces,
        &mut BackendRank2Gemm::<_, _, f64>::new(contract_backend, contract_workspace),
        fusion_workspace,
    )
}

#[allow(clippy::too_many_arguments)]
fn tensorcontract_owned_checked_generic_with_resources<P, D, G, B>(
    lhs_space: &BoundDynamicFusionMapSpace<P>,
    lhs_data: &[D],
    rhs_space: &BoundDynamicFusionMapSpace<P>,
    rhs_data: &[D],
    axes: TensorContractSpec<'_>,
    codomain_rank: Option<usize>,
    transform_backend: &mut B,
    transform_workspaces: &mut [B::Workspace; 3],
    core_gemm: &mut G,
    fusion_workspace: &mut FusionBlockContractWorkspace<D>,
) -> CheckedContractResult<P, D>
where
    P: CheckedGenericRigidSymbols<Scalar = f64>,
    D: DenseRecouplingScalar
        + RecouplingCoefficientAction<f64>
        + ConjugateValue
        + Copy
        + Zero
        + ZeroBytes,
    G: Rank2Gemm<D>,
    B: TreeTransformBackend<D, f64>,
{
    let dst_nout = match codomain_rank {
        Some(rank) => rank,
        None => lhs_space
            .space()
            .rank()
            .checked_sub(axes.lhs_contracting_axes().len())
            .ok_or(OperationError::ElementCountOverflow)?,
    };
    let CheckedContractLocal {
        output_rank,
        axis_plan,
    } = validate_contract_local(lhs_space, lhs_data, rhs_space, rhs_data, axes, dst_nout)?;
    let provider = crate::admission::admit_checked_generic_pair(lhs_space, rhs_space)?;
    // General-axis contraction may braid, twist and insert the fermionic
    // supertrace sign; the checked Generic core is Bosonic only. Canonical
    // composition crosses no legs and skips this boundary.
    <CheckedGenericAdmissionMode as PlanningAlgebra<P>>::core_alpha(
        provider,
        FusionOperand::direct(rhs_space.space()).oriented_homspace(),
        axes.rhs_contracting_axes(),
        [],
    )?;
    let destination = lhs_space.prepare_final_homspace_generic_from_checked(provider, || {
        FusionTreeHomSpace::try_tensorcontract_homspace_generic_checked(
            provider,
            lhs_space.space().homspace(),
            rhs_space.space().homspace(),
            axes.lhs_contracting_axes(),
            axes.rhs_contracting_axes(),
            &axis_plan.output_axes,
            dst_nout,
        )
        .map_err(CheckedGenericPlanError::from)
    })?;
    let (candidate, orientation) = select_complete_bosonic_contract_candidate(
        dst_nout,
        output_rank,
        destination.required_len(),
        lhs_space.space().nout(),
        lhs_space.space().rank(),
        lhs_space.space().required_len()?,
        rhs_space.space().nout(),
        rhs_space.space().rank(),
        rhs_space.space().required_len()?,
        axes,
    )?;
    execute_preselected_checked_generic_contract(
        lhs_space,
        lhs_data,
        rhs_space,
        rhs_data,
        axes,
        dst_nout,
        &candidate,
        orientation,
        provider,
        output_rank,
        destination,
        transform_backend,
        transform_workspaces,
        core_gemm,
        fusion_workspace,
    )
}

/// Canonical composition (TensorKit `mul!`) of two direct checked Generic
/// tensors: `lhs.domain` is glued to `rhs.codomain` in order.
///
/// It plans through the shared [`plan_compose`](super::plan_compose) rung in
/// the checked mode and replays on the Host route executor: one
/// coefficient-free block GEMM per coupled sector, or the checked irregular
/// core for a non-canonical tiling. No leg crosses another, hence no
/// braiding style is required. The destination is staged, planned over as
/// a preview, and committed only after the replay succeeds (#2063).
#[doc(hidden)]
pub fn tensorcompose_owned_checked_generic_in_context<P, D>(
    context: &mut TensorContractFusionExecutionContext<D, RuleIdentity>,
    lhs_space: &BoundDynamicFusionMapSpace<P>,
    lhs_data: &[D],
    rhs_space: &BoundDynamicFusionMapSpace<P>,
    rhs_data: &[D],
) -> CheckedContractResult<P, D>
where
    P: CheckedGenericRigidSymbols<Scalar = f64>,
    D: DenseRecouplingScalar
        + RecouplingCoefficientAction<f64>
        + ConjugateValue
        + Copy
        + Zero
        + ZeroBytes,
{
    let lhs_nout = lhs_space.space().nout();
    let lhs_axes = (lhs_nout..lhs_space.space().rank()).collect::<Vec<_>>();
    let rhs_axes = (0..rhs_space.space().nout()).collect::<Vec<_>>();
    let axes = TensorContractSpec::with_default_output_order(&lhs_axes, &rhs_axes);
    let CheckedContractLocal { axis_plan, .. } =
        validate_contract_local(lhs_space, lhs_data, rhs_space, rhs_data, axes, lhs_nout)?;
    let provider = crate::admission::admit_checked_generic_pair(lhs_space, rhs_space)?;
    let destination = lhs_space.prepare_final_homspace_generic_from_checked(provider, || {
        FusionTreeHomSpace::try_tensorcontract_homspace_generic_checked(
            provider,
            lhs_space.space().homspace(),
            rhs_space.space().homspace(),
            &lhs_axes,
            &rhs_axes,
            &axis_plan.output_axes,
            lhs_nout,
        )
        .map_err(CheckedGenericPlanError::from)
    })?;
    let preview = destination.preview();
    let resolution = plan_compose_in::<HostEagerExecutor, CheckedGenericAdmissionMode, P>(
        PlanTarget {
            rule: provider,
            space: &preview,
            authority: lhs_space,
        },
        FusionOperand::direct(lhs_space.space()),
        FusionOperand::direct(rhs_space.space()),
    )?;
    #[cfg(test)]
    context.record_contract_route(&resolution);
    let mut data = zeroed_payload(destination.required_len());
    context.execute_contract_route_host(
        &resolution,
        preview.structure(),
        &mut data,
        (lhs_space.space().structure(), lhs_data),
        (rhs_space.space().structure(), rhs_data),
        D::one(),
        ContractDestinationInit::Zeroed,
    )?;
    let destination = lhs_space.commit_final_homspace_generic_bound_checked(destination)?;
    Ok((destination, data))
}

#[cfg(test)]
/// Runs exactly one caller-selected checked Generic contraction candidate.
///
/// Every destination remains a read-only staged structure until replay and
/// backend execution succeed. The sole publication is the final commit under
/// the left operand's provider allocation.
#[allow(clippy::too_many_arguments)]
pub(crate) fn tensorcontract_owned_checked_generic_preselected<P, D>(
    lhs_space: &BoundDynamicFusionMapSpace<P>,
    lhs_data: &[D],
    rhs_space: &BoundDynamicFusionMapSpace<P>,
    rhs_data: &[D],
    axes: TensorContractSpec<'_>,
    dst_nout: usize,
    candidate: &ContractAxisOrderCandidate,
    orientation: FusionContractOrientation,
) -> CheckedContractResult<P, D>
where
    P: CheckedGenericRigidSymbols<Scalar = f64>,
    D: DenseRecouplingScalar
        + RecouplingCoefficientAction<f64>
        + ConjugateValue
        + Copy
        + Zero
        + ZeroBytes,
{
    let mut backend = DenseTreeTransformOperations::default();
    let mut workspace = Default::default();
    let mut transform_backend = DenseTreeTransformOperations::default();
    let mut transform_workspaces = Default::default();
    let mut fusion_workspace = FusionBlockContractWorkspace::default();
    tensorcontract_owned_checked_generic_preselected_with_core_gemm(
        lhs_space,
        lhs_data,
        rhs_space,
        rhs_data,
        axes,
        dst_nout,
        candidate,
        orientation,
        &mut transform_backend,
        &mut transform_workspaces,
        &mut BackendRank2Gemm::<_, _, f64>::new(&mut backend, &mut workspace),
        &mut fusion_workspace,
    )
}

#[cfg(test)]
#[allow(clippy::too_many_arguments)]
fn tensorcontract_owned_checked_generic_preselected_with_core_gemm<P, D, G, B>(
    lhs_space: &BoundDynamicFusionMapSpace<P>,
    lhs_data: &[D],
    rhs_space: &BoundDynamicFusionMapSpace<P>,
    rhs_data: &[D],
    axes: TensorContractSpec<'_>,
    dst_nout: usize,
    candidate: &ContractAxisOrderCandidate,
    orientation: FusionContractOrientation,
    transform_backend: &mut B,
    transform_workspaces: &mut [B::Workspace; 3],
    core_gemm: &mut G,
    fusion_workspace: &mut FusionBlockContractWorkspace<D>,
) -> CheckedContractResult<P, D>
where
    P: CheckedGenericRigidSymbols<Scalar = f64>,
    D: DenseRecouplingScalar
        + RecouplingCoefficientAction<f64>
        + ConjugateValue
        + Copy
        + Zero
        + ZeroBytes,
    G: Rank2Gemm<D>,
    B: TreeTransformBackend<D, f64>,
{
    if !same_axes(axes.lhs_contracting_axes(), candidate.lhs())
        || !same_axes(axes.rhs_contracting_axes(), candidate.rhs())
    {
        return Err(OperationError::InvalidArgument {
            message: "preselected candidate must preserve contracted axis sets",
        }
        .into());
    }
    let candidate_axes =
        TensorContractSpec::new(candidate.lhs(), candidate.rhs(), axes.output_permutation());
    let CheckedContractLocal {
        output_rank,
        axis_plan,
    } = validate_contract_local(
        lhs_space,
        lhs_data,
        rhs_space,
        rhs_data,
        candidate_axes,
        dst_nout,
    )?;
    let provider = crate::admission::admit_checked_generic_pair(lhs_space, rhs_space)?;
    // General-axis contraction may braid, twist and insert the fermionic
    // supertrace sign; the checked Generic core is Bosonic only. Canonical
    // composition crosses no legs and skips this boundary.
    <CheckedGenericAdmissionMode as PlanningAlgebra<P>>::core_alpha(
        provider,
        FusionOperand::direct(rhs_space.space()).oriented_homspace(),
        candidate.rhs(),
        [],
    )?;
    let destination = lhs_space.prepare_final_homspace_generic_from_checked(provider, || {
        FusionTreeHomSpace::try_tensorcontract_homspace_generic_checked(
            provider,
            lhs_space.space().homspace(),
            rhs_space.space().homspace(),
            candidate.lhs(),
            candidate.rhs(),
            &axis_plan.output_axes,
            dst_nout,
        )
        .map_err(CheckedGenericPlanError::from)
    })?;
    execute_preselected_checked_generic_contract(
        lhs_space,
        lhs_data,
        rhs_space,
        rhs_data,
        axes,
        dst_nout,
        candidate,
        orientation,
        provider,
        output_rank,
        destination,
        transform_backend,
        transform_workspaces,
        core_gemm,
        fusion_workspace,
    )
}

#[allow(clippy::too_many_arguments)]
fn execute_preselected_checked_generic_contract<P, D, G, B>(
    lhs_space: &BoundDynamicFusionMapSpace<P>,
    lhs_data: &[D],
    rhs_space: &BoundDynamicFusionMapSpace<P>,
    rhs_data: &[D],
    axes: TensorContractSpec<'_>,
    dst_nout: usize,
    candidate: &ContractAxisOrderCandidate,
    orientation: FusionContractOrientation,
    provider: &P,
    output_rank: usize,
    destination: PreparedCheckedGenericDynamicSpace,
    transform_backend: &mut B,
    transform_workspaces: &mut [B::Workspace; 3],
    core_gemm: &mut G,
    fusion_workspace: &mut FusionBlockContractWorkspace<D>,
) -> CheckedContractResult<P, D>
where
    P: CheckedGenericRigidSymbols<Scalar = f64>,
    D: DenseRecouplingScalar
        + RecouplingCoefficientAction<f64>
        + ConjugateValue
        + Copy
        + Zero
        + ZeroBytes,
    G: Rank2Gemm<D>,
    B: TreeTransformBackend<D, f64>,
{
    let candidate_axes =
        TensorContractSpec::new(candidate.lhs(), candidate.rhs(), axes.output_permutation());
    let plan = orient_fusion_contract_plan(
        compile_tensorcontract_fusion_plan_from_ranks(
            dst_nout,
            output_rank,
            lhs_space.space().rank(),
            rhs_space.space().rank(),
            candidate_axes,
            false,
            false,
        )?,
        orientation,
    );

    // Call-owned: the staged and output transforms' composed coefficients
    // publish only after this call's commit.
    let mut coefficients = CheckedPendingCoefficients::new();
    let lhs_prepared = staged_transform(
        &mut coefficients,
        lhs_space,
        provider,
        lhs_space,
        plan.lhs_transform(),
    )?;
    let rhs_prepared = staged_transform(
        &mut coefficients,
        lhs_space,
        provider,
        rhs_space,
        plan.rhs_transform(),
    )?;

    let (core_left, core_right, core_left_structure, core_right_structure) = match orientation {
        FusionContractOrientation::LhsRhs => (
            &lhs_prepared,
            &rhs_prepared,
            lhs_prepared.structure(),
            rhs_prepared.structure(),
        ),
        FusionContractOrientation::RhsLhs => (
            &rhs_prepared,
            &lhs_prepared,
            rhs_prepared.structure(),
            lhs_prepared.structure(),
        ),
    };
    let core_axes = plan.core_axes().as_spec();
    let core_axis_plan = TensorContractAxisPlan::compile(
        core_left.homspace().rank(),
        core_right.homspace().rank(),
        output_rank,
        core_axes,
    )?;
    let core_destination =
        lhs_space.prepare_final_homspace_generic_from_checked(provider, || {
            FusionTreeHomSpace::try_tensorcontract_homspace_generic_checked(
                provider,
                core_left.homspace(),
                core_right.homspace(),
                core_axes.lhs_contracting_axes(),
                core_axes.rhs_contracting_axes(),
                &core_axis_plan.output_axes,
                plan.core_dst_open_lhs_rank(),
            )
            .map_err(CheckedGenericPlanError::from)
        })?;
    let core_structure = core_destination.shared_structure();
    let core_plan = compile_checked_generic_core_plan(
        &core_structure,
        core_destination.nout(),
        core_left_structure,
        core_left.nout(),
        core_right_structure,
        core_right.nout(),
        core_axes,
    )?;

    let destination_structure = destination.shared_structure();
    let output_replay = if plan.output_transform_is_identity() {
        None
    } else {
        Some(
            <CheckedGenericAdmissionMode as PlanningAlgebra<P>>::tree_structure(
                &mut coefficients,
                provider,
                plan.output_transform(),
                &destination_structure,
                TreeStructureSource::Stored {
                    structure: &core_structure,
                    storage_conjugate: false,
                },
            )?,
        )
    };

    let lhs_transformed = execute_staged_transform(
        transform_backend,
        &mut transform_workspaces[Stage::Lhs as usize],
        &lhs_prepared,
        lhs_space.space().structure(),
        lhs_data,
    )?;
    let rhs_transformed = execute_staged_transform(
        transform_backend,
        &mut transform_workspaces[Stage::Rhs as usize],
        &rhs_prepared,
        rhs_space.space().structure(),
        rhs_data,
    )?;

    let lhs_core_data = lhs_transformed.as_deref().unwrap_or(lhs_data);
    let rhs_core_data = rhs_transformed.as_deref().unwrap_or(rhs_data);

    let (core_lhs_data, core_rhs_data) = match orientation {
        FusionContractOrientation::LhsRhs => (lhs_core_data, rhs_core_data),
        FusionContractOrientation::RhsLhs => (rhs_core_data, lhs_core_data),
    };
    let mut kernels = crate::StridedHostKernelAdapter::default();
    let mut data = zeroed_payload(destination.required_len());
    if let Some(output_replay) = output_replay {
        let mut core_data = zeroed_payload(core_destination.required_len());
        core_plan.execute_raw_zeroed(
            &mut kernels,
            core_gemm,
            fusion_workspace,
            &core_structure,
            &mut core_data,
            core_left_structure,
            core_lhs_data,
            core_right_structure,
            core_rhs_data,
            D::one(),
        )?;
        execute_transform(
            transform_backend,
            &mut transform_workspaces[Stage::Output as usize],
            &output_replay,
            &destination_structure,
            &core_structure,
            &mut data,
            &core_data,
        )?;
    } else {
        core_plan.execute_raw_zeroed(
            &mut kernels,
            core_gemm,
            fusion_workspace,
            &core_structure,
            &mut data,
            core_left_structure,
            core_lhs_data,
            core_right_structure,
            core_rhs_data,
            D::one(),
        )?;
    }
    let destination = lhs_space.commit_final_homspace_generic_bound_checked(destination)?;
    // Only after the fallible destination commit: a failed call commits no
    // intermediate and publishes nothing. These commits cannot fail; a lost
    // or refused admission leaves its preview unpublishable.
    let committed = [
        (
            destination_structure,
            Arc::clone(destination.space().structure()),
        ),
        lhs_prepared.commit(),
        rhs_prepared.commit(),
        (core_structure, core_destination.commit_structure()),
    ];
    coefficients.flush_committed(&committed);
    Ok((destination, data))
}

#[cfg(test)]
mod tests {
    use std::cell::{Cell, RefCell};

    use crate::tests::GenericMultiplicityRule;
    use tenet_core::{
        BraidingStyleKind, CheckedGenericFusion, CoupledSectorFold, FusionProductSpace, FusionRule,
        FusionStyleKind, GenericFArray, GenericRMatrix, InfallibleGeneric, RuleIdentity, SectorId,
        SectorLeg, SectorVec,
    };

    use super::*;
    use crate::contract::fusion::contracted_axis_order_candidates;

    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    enum Query {
        Dual,
        Channel,
        N,
        F,
        R,
        Rigidity,
    }

    impl Query {
        const COUNT: usize = 6;
    }

    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    enum Event {
        Identity,
        Style,
        Braiding,
        Query(Query),
    }

    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    struct SpyError(Query);

    impl std::fmt::Display for SpyError {
        fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(formatter, "injected {:?} failure", self.0)
        }
    }

    impl std::error::Error for SpyError {}

    /// Checked-only wrapper: deliberately does not implement `FusionRule`.
    struct CheckedGenericSpy {
        rule: GenericMultiplicityRule,
        calls: Cell<[usize; Query::COUNT]>,
        events: RefCell<Vec<Event>>,
        fail: Cell<Option<Query>>,
        malformed: Cell<Option<Query>>,
    }

    impl CheckedGenericSpy {
        fn new() -> Self {
            Self {
                rule: GenericMultiplicityRule,
                calls: Cell::new([0; Query::COUNT]),
                events: RefCell::new(Vec::new()),
                fail: Cell::new(None),
                malformed: Cell::new(None),
            }
        }

        fn hit(&self, query: Query) -> Result<(), SpyError> {
            self.events.borrow_mut().push(Event::Query(query));
            let mut calls = self.calls.get();
            calls[query as usize] += 1;
            self.calls.set(calls);
            if self.fail.get() == Some(query) {
                Err(SpyError(query))
            } else {
                Ok(())
            }
        }

        fn reset(&self) {
            self.calls.set([0; Query::COUNT]);
            self.events.borrow_mut().clear();
            self.fail.set(None);
            self.malformed.set(None);
        }

        fn count(&self, query: Query) -> usize {
            self.calls.get()[query as usize]
        }

        fn algebra_calls(&self) -> usize {
            self.calls.get().into_iter().sum()
        }
    }

    impl CheckedGenericFusion for CheckedGenericSpy {
        type Error = SpyError;

        // Why its own identity rather than the wrapped rule's: the composed
        // coefficients are cached per identity, and a sibling test's
        // publication under the shared rule's identity would skip this
        // spy's provider queries.
        fn rule_identity(&self) -> RuleIdentity {
            self.events.borrow_mut().push(Event::Identity);
            RuleIdentity::of_type::<Self>()
        }

        fn fusion_style(&self) -> FusionStyleKind {
            self.events.borrow_mut().push(Event::Style);
            FusionRule::fusion_style(&self.rule)
        }

        fn braiding_style(&self) -> BraidingStyleKind {
            self.events.borrow_mut().push(Event::Braiding);
            BraidingStyleKind::Bosonic
        }

        fn vacuum(&self) -> SectorId {
            FusionRule::vacuum(&self.rule)
        }

        fn try_dual(&self, sector: SectorId) -> Result<SectorId, Self::Error> {
            self.hit(Query::Dual)?;
            Ok(FusionRule::dual(&self.rule, sector))
        }

        fn try_fusion_channels(
            &self,
            left: SectorId,
            right: SectorId,
        ) -> Result<SectorVec, Self::Error> {
            self.hit(Query::Channel)?;
            Ok(FusionRule::fusion_channels(&self.rule, left, right))
        }

        fn try_fusion_channels_in_table(
            &self,
            left: SectorId,
            right: SectorId,
        ) -> Result<SectorVec, Self::Error> {
            self.hit(Query::Channel)?;
            Ok(FusionRule::fusion_channels(&self.rule, left, right))
        }

        // Counts one query for the fold itself, as the engine's failure
        // budgets are stated per provider call rather than per inner channel
        // lookup. The classification is the trait default over this rule.
        fn try_coupled_sector_fold(
            &self,
            effective: &[SectorId],
        ) -> Result<CoupledSectorFold, Self::Error> {
            self.hit(Query::Channel)?;
            match InfallibleGeneric::new(&self.rule).try_coupled_sector_fold(effective) {
                Ok(fold) => Ok(fold),
                Err(never) => match never {},
            }
        }

        fn try_nsymbol(
            &self,
            left: SectorId,
            right: SectorId,
            coupled: SectorId,
        ) -> Result<usize, Self::Error> {
            self.hit(Query::N)?;
            Ok(FusionRule::nsymbol(&self.rule, left, right, coupled))
        }
    }

    impl CheckedGenericRigidSymbols for CheckedGenericSpy {
        type Scalar = f64;

        fn try_sqrt_dim_scalar(&self, sector: SectorId) -> Result<f64, Self::Error> {
            self.hit(Query::Rigidity)?;
            let _ = sector;
            Ok(1.0)
        }

        fn try_inv_sqrt_dim_scalar(&self, sector: SectorId) -> Result<f64, Self::Error> {
            self.hit(Query::Rigidity)?;
            let _ = sector;
            Ok(1.0)
        }

        fn try_frobenius_schur_phase_scalar(&self, sector: SectorId) -> Result<f64, Self::Error> {
            self.hit(Query::Rigidity)?;
            let _ = sector;
            Ok(1.0)
        }

        fn try_f_symbol_generic(
            &self,
            a: SectorId,
            b: SectorId,
            c: SectorId,
            d: SectorId,
            e: SectorId,
            f: SectorId,
        ) -> Result<GenericFArray<f64>, Self::Error> {
            self.hit(Query::F)?;
            let shape = (
                self.rule.nsymbol(a, b, e),
                self.rule.nsymbol(e, c, d),
                self.rule.nsymbol(b, c, f),
                self.rule.nsymbol(a, f, d),
            );
            let mut data = vec![0.0; shape.0 * shape.1 * shape.2 * shape.3];
            let cols = shape.0 * shape.1;
            let rows = shape.2 * shape.3;
            for index in 0..cols.min(rows) {
                data[index * rows + index] = 1.0;
            }
            let symbol = GenericFArray::new(data, shape);
            if self.malformed.get() == Some(Query::F) {
                Ok(GenericFArray::new(
                    symbol.data().to_vec(),
                    (1, 1, symbol.data().len(), 1),
                ))
            } else {
                Ok(symbol)
            }
        }

        fn try_r_symbol_generic(
            &self,
            a: SectorId,
            b: SectorId,
            c: SectorId,
        ) -> Result<GenericRMatrix<f64>, Self::Error> {
            self.hit(Query::R)?;
            let size = self.rule.nsymbol(a, b, c);
            let mut data = vec![0.0; size * size];
            for index in 0..size {
                data[index * size + index] = 1.0;
            }
            let symbol = GenericRMatrix::new(data, size, size);
            if self.malformed.get() == Some(Query::R) {
                Ok(GenericRMatrix::new(
                    symbol.data().to_vec(),
                    1,
                    symbol.data().len(),
                ))
            } else {
                Ok(symbol)
            }
        }
    }

    struct FailingGemm;

    impl Rank2Gemm<f64> for FailingGemm {
        #[allow(clippy::too_many_arguments)]
        fn matmul_rank2(
            &mut self,
            _dst: &mut [f64],
            _lhs: &[f64],
            _rhs: &[f64],
            _rows: usize,
            _contracted: usize,
            _cols: usize,
            _alpha: f64,
            _beta: f64,
        ) -> Result<(), OperationError> {
            Err(OperationError::StridedKernel {
                message: "injected checked Generic core failure".into(),
            })
        }
    }

    fn homspace(_rule: &GenericMultiplicityRule, nout: usize, nin: usize) -> FusionTreeHomSpace {
        let leg = || SectorLeg::new([(SectorId::new(1), 1)], false);
        FusionTreeHomSpace::new(
            FusionProductSpace::new((0..nout).map(|_| leg())),
            FusionProductSpace::new((0..nin).map(|_| leg())),
        )
    }

    #[allow(clippy::arc_with_non_send_sync)]
    fn bound_pair(
        nout: usize,
        nin: usize,
    ) -> (
        Arc<CheckedGenericSpy>,
        BoundDynamicFusionMapSpace<CheckedGenericSpy>,
        Arc<CheckedGenericSpy>,
        BoundDynamicFusionMapSpace<CheckedGenericSpy>,
    ) {
        let left = Arc::new(CheckedGenericSpy::new());
        let right = Arc::new(CheckedGenericSpy::new());
        let homspace = homspace(&left.rule, nout, nin);
        let lhs = BoundDynamicFusionMapSpace::from_final_homspace_generic_checked(
            Arc::clone(&left),
            homspace.clone(),
        )
        .unwrap();
        let rhs = BoundDynamicFusionMapSpace::from_final_homspace_generic_checked(
            Arc::clone(&right),
            homspace,
        )
        .unwrap();
        left.reset();
        right.reset();
        (left, lhs, right, rhs)
    }

    #[test]
    #[allow(clippy::arc_with_non_send_sync)]
    fn preselected_checked_generic_uses_left_authority_and_commits_left_owner() {
        const ISOLATED: &str = "TENET_CHECKED_GENERIC_CONTRACT_AUTHORITY_ISOLATED";
        if std::env::var_os(ISOLATED).is_none() {
            let output = std::process::Command::new(std::env::current_exe().unwrap())
                .args([
                    "--exact",
                    "contract::checked_generic::tests::preselected_checked_generic_uses_left_authority_and_commits_left_owner",
                ])
                .env(ISOLATED, "1")
                .output()
                .unwrap();
            assert!(
                output.status.success(),
                "isolated authority test failed:\n{}",
                String::from_utf8_lossy(&output.stderr)
            );
            return;
        }

        let (left, lhs, right, rhs) = bound_pair(1, 1);
        tenet_core::clear_structure_caches();
        let candidate = contracted_axis_order_candidates(&[1], &[0]).remove(0);
        let lhs_data = vec![1.0; lhs.space().required_len().unwrap()];
        let rhs_data = vec![2.0; rhs.space().required_len().unwrap()];

        let (output, data) = tensorcontract_owned_checked_generic_preselected(
            &lhs,
            &lhs_data,
            &rhs,
            &rhs_data,
            TensorContractSpec::with_default_output_order(&[1], &[0]),
            1,
            &candidate,
            FusionContractOrientation::LhsRhs,
        )
        .unwrap();

        assert!(Arc::ptr_eq(output.provider_arc(), &left));
        assert!(!Arc::ptr_eq(output.provider_arc(), &right));
        assert_eq!(right.algebra_calls(), 0);
        assert!(left.algebra_calls() > 0);
        assert_eq!(data.len(), output.space().required_len().unwrap());
        // The destination's one identity read and admission style, then the
        // final guard's style: commit reuses the admitted identity (#2046).
        assert!(left
            .events
            .borrow()
            .ends_with(&[Event::Identity, Event::Style, Event::Style]));

        left.reset();
        right.reset();
        let (warm_output, warm_data) = tensorcontract_owned_checked_generic_preselected(
            &lhs,
            &lhs_data,
            &rhs,
            &rhs_data,
            TensorContractSpec::with_default_output_order(&[1], &[0]),
            1,
            &candidate,
            FusionContractOrientation::LhsRhs,
        )
        .unwrap();
        assert!(Arc::ptr_eq(warm_output.provider_arc(), &left));
        assert!(!Arc::ptr_eq(warm_output.provider_arc(), &right));
        assert_eq!(right.algebra_calls(), 0);
        assert_eq!(warm_data, data);
        // The destination's one identity read and admission style, then the
        // final guard's style: commit reuses the admitted identity (#2046).
        assert!(left
            .events
            .borrow()
            .ends_with(&[Event::Identity, Event::Style, Event::Style]));
    }

    #[test]
    #[allow(clippy::arc_with_non_send_sync)]
    fn preselected_checked_generic_rejects_local_axes_before_provider_queries() {
        let (left, lhs, right, rhs) = bound_pair(1, 1);
        let candidate = contracted_axis_order_candidates(&[9], &[0]).remove(0);
        let error = tensorcontract_owned_checked_generic_preselected(
            &lhs,
            &vec![0.0; lhs.space().required_len().unwrap()],
            &rhs,
            &vec![0.0; rhs.space().required_len().unwrap()],
            TensorContractSpec::with_default_output_order(&[9], &[0]),
            1,
            &candidate,
            FusionContractOrientation::LhsRhs,
        )
        .unwrap_err();

        assert!(matches!(error, CheckedGenericPlanError::Operation(_)));
        assert_eq!(left.algebra_calls(), 0);
        assert_eq!(right.algebra_calls(), 0);
    }

    #[test]
    #[allow(clippy::arc_with_non_send_sync)]
    fn preselected_checked_generic_executes_one_transforming_candidate() {
        let (left, lhs, right, rhs) = bound_pair(2, 2);
        let candidate = contracted_axis_order_candidates(&[3, 2], &[0, 1]).remove(0);
        let lhs_data = vec![1.0; lhs.space().required_len().unwrap()];
        let rhs_data = vec![2.0; rhs.space().required_len().unwrap()];

        let (output, data) = tensorcontract_owned_checked_generic_preselected(
            &lhs,
            &lhs_data,
            &rhs,
            &rhs_data,
            TensorContractSpec::with_default_output_order(&[3, 2], &[0, 1]),
            2,
            &candidate,
            FusionContractOrientation::RhsLhs,
        )
        .unwrap();

        assert!(Arc::ptr_eq(output.provider_arc(), &left));
        assert_eq!(right.algebra_calls(), 0);
        assert!(left.count(Query::F) > 0);
        assert!(left.count(Query::R) > 0);
        assert_eq!(data.len(), output.space().required_len().unwrap());
    }

    #[test]
    #[allow(clippy::arc_with_non_send_sync)]
    fn a_warm_checked_generic_contraction_converts_at_most_one_pack_per_stage() {
        // Isolated: a concurrent cache clear would force a rebuild.
        const ISOLATED: &str = "TENET_CHECKED_GENERIC_CONTRACT_PACKS_ISOLATED";
        if std::env::var_os(ISOLATED).is_none() {
            let output = std::process::Command::new(std::env::current_exe().unwrap())
                .args([
                    "--exact",
                    "contract::checked_generic::tests::a_warm_checked_generic_contraction_converts_at_most_one_pack_per_stage",
                ])
                .env(ISOLATED, "1")
                .output()
                .unwrap();
            assert!(
                output.status.success(),
                "isolated pack test failed:\n{}",
                String::from_utf8_lossy(&output.stderr)
            );
            return;
        }

        // Why (#2101): each stage replays into its own context workspace, so
        // no stage evicts another's Multi pack: a warm call converts at most
        // one pack per stage (a shared workspace would convert three into one
        // slot). #2063 tightens this to none: a warm call binds the completed
        // transformers of the first call, whose packs are already converted.
        let (_left, lhs, _right, rhs) = bound_pair(2, 2);
        let lhs_data = (0..lhs.space().required_len().unwrap())
            .map(|index| index as f64 - 1.5)
            .collect::<Vec<_>>();
        let rhs_data = (0..rhs.space().required_len().unwrap())
            .map(|index| 2.0 - index as f64)
            .collect::<Vec<_>>();
        let mut context =
            TensorContractFusionExecutionContext::<f64, tenet_core::RuleIdentity>::default();
        let mut run = || {
            let (_, data) = tensorcontract_owned_checked_generic_in_context(
                &mut context,
                &lhs,
                &lhs_data,
                &rhs,
                &rhs_data,
                TensorContractSpec::new(
                    &[3, 1],
                    &[0, 3],
                    tenet_operations::OutputAxisOrder::Axes(&[2, 0, 3, 1]),
                ),
                2,
            )
            .unwrap();
            (data, context.eager_coefficient_pack_builds())
        };
        let (cold_data, cold) = run();
        assert_eq!(cold, [1, 1, 1], "all three stages recouple");
        let mut previous = cold;
        for _ in 0..3 {
            let (data, builds) = run();
            assert_eq!(data, cold_data);
            assert_eq!(builds, previous, "a warm call converts no pack");
            previous = builds;
        }
    }

    #[test]
    #[allow(clippy::arc_with_non_send_sync)]
    fn preselected_checked_generic_preserves_late_provider_and_shape_errors() {
        const ISOLATED: &str = "TENET_CHECKED_GENERIC_CONTRACT_PROVIDER_FAILURE_ISOLATED";
        if std::env::var_os(ISOLATED).is_none() {
            let output = std::process::Command::new(std::env::current_exe().unwrap())
                .args([
                    "--exact",
                    "contract::checked_generic::tests::preselected_checked_generic_preserves_late_provider_and_shape_errors",
                ])
                .env(ISOLATED, "1")
                .output()
                .unwrap();
            assert!(
                output.status.success(),
                "isolated provider-failure test failed:\n{}",
                String::from_utf8_lossy(&output.stderr)
            );
            return;
        }

        let (left, lhs, right, rhs) = bound_pair(2, 2);
        tenet_core::clear_structure_caches();
        let candidate = contracted_axis_order_candidates(&[3, 2], &[0, 1]).remove(0);
        let lhs_data = vec![1.0; lhs.space().required_len().unwrap()];
        let rhs_data = vec![2.0; rhs.space().required_len().unwrap()];

        for query in [Query::Dual, Query::Channel, Query::N, Query::F, Query::R] {
            left.reset();
            right.reset();
            left.fail.set(Some(query));
            let error = tensorcontract_owned_checked_generic_preselected(
                &lhs,
                &lhs_data,
                &rhs,
                &rhs_data,
                TensorContractSpec::with_default_output_order(&[3, 2], &[0, 1]),
                2,
                &candidate,
                FusionContractOrientation::LhsRhs,
            )
            .unwrap_err();
            assert!(matches!(
                error,
                CheckedGenericPlanError::Provider(SpyError(actual)) if actual == query
            ));
            assert_eq!(right.algebra_calls(), 0);
            assert_eq!(
                tenet_core::structure_cache_info(
                    tenet_core::StructureCacheKind::DegeneracyStructure
                )
                .entries(),
                0
            );
        }

        left.reset();
        right.reset();
        left.malformed.set(Some(Query::R));
        let error = tensorcontract_owned_checked_generic_preselected(
            &lhs,
            &lhs_data,
            &rhs,
            &rhs_data,
            TensorContractSpec::with_default_output_order(&[3, 2], &[0, 1]),
            2,
            &candidate,
            FusionContractOrientation::LhsRhs,
        )
        .unwrap_err();
        assert!(matches!(
            error,
            CheckedGenericPlanError::SymbolShape { symbol: "R", .. }
        ));
        assert_eq!(right.algebra_calls(), 0);
        assert_eq!(
            tenet_core::structure_cache_info(tenet_core::StructureCacheKind::DegeneracyStructure)
                .entries(),
            0
        );
    }

    #[test]
    #[allow(clippy::arc_with_non_send_sync)]
    fn preselected_checked_generic_backend_failure_does_not_commit() {
        const ISOLATED: &str = "TENET_CHECKED_GENERIC_CONTRACT_FAILURE_ISOLATED";
        if std::env::var_os(ISOLATED).is_none() {
            let output = std::process::Command::new(std::env::current_exe().unwrap())
                .args([
                    "--exact",
                    "contract::checked_generic::tests::preselected_checked_generic_backend_failure_does_not_commit",
                ])
                .env(ISOLATED, "1")
                .output()
                .unwrap();
            assert!(
                output.status.success(),
                "isolated failure-atomicity test failed:\n{}",
                String::from_utf8_lossy(&output.stderr)
            );
            return;
        }

        let (left, lhs, right, rhs) = bound_pair(1, 1);
        tenet_core::clear_structure_caches();
        let candidate = contracted_axis_order_candidates(&[1], &[0]).remove(0);
        let mut transform_backend = DenseTreeTransformOperations::default();
        let mut transform_workspaces = Default::default();
        let mut fusion_workspace = FusionBlockContractWorkspace::default();
        let error = tensorcontract_owned_checked_generic_preselected_with_core_gemm(
            &lhs,
            &vec![1.0; lhs.space().required_len().unwrap()],
            &rhs,
            &vec![2.0; rhs.space().required_len().unwrap()],
            TensorContractSpec::with_default_output_order(&[1], &[0]),
            1,
            &candidate,
            FusionContractOrientation::LhsRhs,
            &mut transform_backend,
            &mut transform_workspaces,
            &mut FailingGemm,
            &mut fusion_workspace,
        )
        .unwrap_err();

        assert!(matches!(
            error,
            CheckedGenericPlanError::Operation(OperationError::StridedKernel { .. })
        ));
        assert_eq!(right.algebra_calls(), 0);
        assert!(left.algebra_calls() > 0);
        // The layout walks succeeded before the backend failed, so their
        // pure-data layouts may be published (#2030); no block structure is.
        assert_eq!(
            tenet_core::structure_cache_info(tenet_core::StructureCacheKind::DegeneracyStructure)
                .entries(),
            0
        );
        assert_eq!(
            tenet_core::structure_cache_info(tenet_core::StructureCacheKind::DegeneracyStructure)
                .entries(),
            0
        );
    }

    #[test]
    #[allow(clippy::arc_with_non_send_sync)]
    fn preselected_checked_generic_success_commits_once() {
        const ISOLATED: &str = "TENET_CHECKED_GENERIC_CONTRACT_SUCCESS_ISOLATED";
        if std::env::var_os(ISOLATED).is_none() {
            let output = std::process::Command::new(std::env::current_exe().unwrap())
                .args([
                    "--exact",
                    "contract::checked_generic::tests::preselected_checked_generic_success_commits_once",
                ])
                .env(ISOLATED, "1")
                .output()
                .unwrap();
            assert!(
                output.status.success(),
                "isolated commit test failed:\n{}",
                String::from_utf8_lossy(&output.stderr)
            );
            return;
        }

        let (left, lhs, right, rhs) = bound_pair(1, 1);
        tenet_core::clear_structure_caches();
        let candidate = contracted_axis_order_candidates(&[1], &[0]).remove(0);
        let (output, _) = tensorcontract_owned_checked_generic_preselected(
            &lhs,
            &vec![1.0; lhs.space().required_len().unwrap()],
            &rhs,
            &vec![2.0; rhs.space().required_len().unwrap()],
            TensorContractSpec::with_default_output_order(&[1], &[0]),
            1,
            &candidate,
            FusionContractOrientation::LhsRhs,
        )
        .unwrap();

        assert!(Arc::ptr_eq(output.provider_arc(), &left));
        assert_eq!(right.algebra_calls(), 0);
        assert_eq!(
            tenet_core::structure_cache_info(tenet_core::StructureCacheKind::DegeneracyStructure)
                .entries(),
            1
        );
        // The destination's one identity read and admission style, then the
        // final guard's style: commit reuses the admitted identity (#2046).
        assert!(left
            .events
            .borrow()
            .ends_with(&[Event::Identity, Event::Style, Event::Style]));
    }

    /// Rank-1 legs with the listed `(sector, degeneracy)` pairs on each side.
    fn homspace_with(codomain: &[(usize, usize)], domain: &[(usize, usize)]) -> FusionTreeHomSpace {
        let leg = |sectors: &[(usize, usize)]| {
            SectorLeg::new(
                sectors
                    .iter()
                    .map(|&(sector, degeneracy)| (SectorId::new(sector), degeneracy)),
                false,
            )
        };
        FusionTreeHomSpace::new(
            FusionProductSpace::new([leg(codomain)]),
            FusionProductSpace::new([leg(domain)]),
        )
    }

    #[allow(clippy::arc_with_non_send_sync)]
    fn bound_space(homspace: FusionTreeHomSpace) -> BoundDynamicFusionMapSpace<CheckedGenericSpy> {
        BoundDynamicFusionMapSpace::from_final_homspace_generic_checked(
            Arc::new(CheckedGenericSpy::new()),
            homspace,
        )
        .unwrap()
    }

    fn assert_inactive_sector_zero_and_active_product<D>(
        one: D,
        two: D,
        six: D,
        is_positive_zero: fn(D) -> bool,
    ) where
        D: DenseRecouplingScalar
            + RecouplingCoefficientAction<f64>
            + ConjugateValue
            + Copy
            + Zero
            + ZeroBytes
            + std::fmt::Debug,
    {
        // lhs `[0:1, 1:2] <- [1:3]` and rhs `[1:3] <- [0:1, 1:2]` couple only
        // to sector 1; the destination `[0:1, 1:2] <- [0:1, 1:2]` also has the
        // sector-0 block, which no GEMM writes.
        let lhs = bound_space(homspace_with(&[(0, 1), (1, 2)], &[(1, 3)]));
        let rhs = bound_space(homspace_with(&[(1, 3)], &[(0, 1), (1, 2)]));
        let lhs_data = vec![one; lhs.space().required_len().unwrap()];
        let rhs_data = vec![two; rhs.space().required_len().unwrap()];
        let (output, data) = tensorcontract_owned_checked_generic(
            &lhs,
            &lhs_data,
            &rhs,
            &rhs_data,
            TensorContractSpec::with_default_output_order(&[1], &[0]),
        )
        .unwrap();
        let structure = output.space().structure();
        assert_eq!(structure.block_count(), 2);
        assert_eq!(data.len(), 1 + 4);
        for index in 0..structure.block_count() {
            // The sector-0 block is the 1x1 one, the sector-1 block the 2x2 one.
            let block = structure.block(index).unwrap();
            let range = block.offset()..block.offset() + block.element_count().unwrap();
            match range.len() {
                1 => assert!(data[range].iter().all(|&v| is_positive_zero(v))),
                4 => assert!(data[range].iter().all(|&v| v == six)),
                other => panic!("unexpected block length {other}"),
            }
        }
    }

    #[test]
    fn owned_checked_generic_contract_leaves_inactive_sectors_exactly_zero() {
        assert_inactive_sector_zero_and_active_product::<f64>(1.0, 2.0, 6.0, |v| v.to_bits() == 0);
        assert_inactive_sector_zero_and_active_product::<num_complex::Complex64>(
            num_complex::Complex64::new(1.0, 0.0),
            num_complex::Complex64::new(2.0, 0.0),
            num_complex::Complex64::new(6.0, 0.0),
            |v| v.re.to_bits() == 0 && v.im.to_bits() == 0,
        );
    }

    struct FailingAtJob {
        calls: usize,
        fail_at: usize,
    }

    impl Rank2Gemm<f64> for FailingAtJob {
        #[allow(clippy::too_many_arguments)]
        fn matmul_rank2(
            &mut self,
            dst: &mut [f64],
            _lhs: &[f64],
            _rhs: &[f64],
            _rows: usize,
            _contracted: usize,
            _cols: usize,
            _alpha: f64,
            _beta: f64,
        ) -> Result<(), OperationError> {
            let call = self.calls;
            self.calls += 1;
            if call == self.fail_at {
                return Err(OperationError::StridedKernel {
                    message: "injected failure on a later GEMM job".into(),
                });
            }
            dst.fill(1.0);
            Ok(())
        }
    }

    /// Resident `(cache 2, cache 3)` entries.
    fn cache_entries() -> (usize, usize) {
        let entries = |kind| tenet_core::structure_cache_info(kind).entries();
        (
            entries(tenet_core::StructureCacheKind::DegeneracyStructure),
            entries(tenet_core::StructureCacheKind::CompletedTreeTransformer),
        )
    }

    fn coefficient_admissions() -> u64 {
        tenet_core::structure_cache_info(tenet_core::StructureCacheKind::TreeTransformCoefficients)
            .admissions()
    }

    #[test]
    #[allow(clippy::arc_with_non_send_sync)]
    fn checked_contraction_publishes_composed_coefficients_only_after_its_commit() {
        const ISOLATED: &str = "TENET_CHECKED_GENERIC_CONTRACT_COEFFICIENTS_ISOLATED";
        if std::env::var_os(ISOLATED).is_none() {
            let output = std::process::Command::new(std::env::current_exe().unwrap())
                .args([
                    "--exact",
                    "contract::checked_generic::tests::checked_contraction_publishes_composed_coefficients_only_after_its_commit",
                ])
                .env(ISOLATED, "1")
                .output()
                .unwrap();
            assert!(
                output.status.success(),
                "isolated coefficient-publication test failed:\n{}",
                String::from_utf8_lossy(&output.stderr)
            );
            return;
        }

        let (left, lhs, right, rhs) = bound_pair(2, 2);
        tenet_core::clear_structure_caches();
        let candidate = contracted_axis_order_candidates(&[3, 2], &[0, 1]).remove(0);
        let lhs_data = vec![1.0; lhs.space().required_len().unwrap()];
        let rhs_data = vec![2.0; rhs.space().required_len().unwrap()];
        fn contract_with<G: Rank2Gemm<f64>>(
            (lhs, lhs_data, rhs, rhs_data): (
                &BoundDynamicFusionMapSpace<CheckedGenericSpy>,
                &[f64],
                &BoundDynamicFusionMapSpace<CheckedGenericSpy>,
                &[f64],
            ),
            candidate: &ContractAxisOrderCandidate,
            gemm: &mut G,
        ) -> CheckedContractResult<CheckedGenericSpy, f64> {
            let mut transform_backend = DenseTreeTransformOperations::default();
            let mut transform_workspaces = Default::default();
            let mut fusion_workspace = FusionBlockContractWorkspace::default();
            tensorcontract_owned_checked_generic_preselected_with_core_gemm(
                lhs,
                lhs_data,
                rhs,
                rhs_data,
                TensorContractSpec::with_default_output_order(&[3, 2], &[0, 1]),
                2,
                candidate,
                FusionContractOrientation::RhsLhs,
                &mut transform_backend,
                &mut transform_workspaces,
                gemm,
                &mut fusion_workspace,
            )
        }
        let operands = (&lhs, lhs_data.as_slice(), &rhs, rhs_data.as_slice());
        let failing = || {
            contract_with(
                operands,
                &candidate,
                &mut FailingAtJob {
                    calls: 0,
                    fail_at: 0,
                },
            )
        };
        let succeeding = || {
            let mut backend = DenseTreeTransformOperations::default();
            let mut workspace = Default::default();
            contract_with(
                operands,
                &candidate,
                &mut BackendRank2Gemm::<_, _, f64>::new(&mut backend, &mut workspace),
            )
        };

        // What: the staged and output transforms build their groups, then
        // the core GEMM fails before the commit. Nothing is published, so a
        // retry replays the identical F/R provider ledger and error.
        let mut ledgers = Vec::new();
        for _attempt in 0..2 {
            left.reset();
            right.reset();
            crate::tree_transform::take_coefficient_group_activity();
            let error = failing().unwrap_err();
            assert!(matches!(
                error,
                CheckedGenericPlanError::Operation(OperationError::StridedKernel { .. })
            ));
            let groups = crate::tree_transform::take_coefficient_group_activity();
            assert!(groups.misses > 0);
            assert_eq!((groups.hits, groups.publications), (0, 0));
            assert_eq!(coefficient_admissions(), 0);
            // Caches 2 and 3 too: a failed call commits no intermediate and
            // publishes no transformer it built.
            let transformers = crate::tree_transform::take_completed_transformer_activity();
            assert_eq!((transformers.builds, transformers.publications), (3, 0));
            assert_eq!(cache_entries(), (0, 0));
            assert!(left.count(Query::F) > 0 && left.count(Query::R) > 0);
            // The F/R ledger: layout walks may publish pure-data layouts
            // (#2030), so structural queries can shrink on a retry.
            ledgers.push(
                left.events
                    .borrow()
                    .iter()
                    .copied()
                    .filter(|event| matches!(event, Event::Query(Query::F | Query::R)))
                    .collect::<Vec<_>>(),
            );
        }
        assert_eq!(ledgers[0], ledgers[1]);

        // What: a committed call publishes its groups, its three
        // intermediates (cache 2, with the destination) and its three
        // transformers (cache 3); a repeat binds those transformers, so it
        // makes no group lookup and no F or R query and returns the same bits.
        left.reset();
        let (first_space, first) = succeeding().unwrap();
        assert!(coefficient_admissions() > 0);
        let transformers = crate::tree_transform::take_completed_transformer_activity();
        assert_eq!((transformers.builds, transformers.publications), (3, 3));
        // The destination and the three intermediates span two HomSpaces.
        let committed = cache_entries();
        assert_eq!(committed, (2, 3));
        left.reset();
        crate::tree_transform::take_coefficient_group_activity();
        let (repeat_space, repeat) = succeeding().unwrap();
        let groups = crate::tree_transform::take_coefficient_group_activity();
        assert_eq!((groups.hits, groups.misses, groups.publications), (0, 0, 0));
        let transformers = crate::tree_transform::take_completed_transformer_activity();
        assert_eq!(
            (
                transformers.hits,
                transformers.builds,
                transformers.publications
            ),
            (3, 0, 0)
        );
        assert_eq!(cache_entries(), committed);
        assert_eq!((left.count(Query::F), left.count(Query::R)), (0, 0));
        assert_eq!(first_space.space(), repeat_space.space());
        assert_eq!(
            first.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            repeat.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
        );
    }

    #[test]
    #[allow(clippy::arc_with_non_send_sync)]
    fn intermediates_sharing_a_homspace_converge_on_one_id_within_the_call() {
        const ISOLATED: &str = "TENET_CHECKED_GENERIC_CONTRACT_SHARED_HOMSPACE_ISOLATED";
        if std::env::var_os(ISOLATED).is_none() {
            let output = std::process::Command::new(std::env::current_exe().unwrap())
                .args([
                    "--exact",
                    "contract::checked_generic::tests::intermediates_sharing_a_homspace_converge_on_one_id_within_the_call",
                ])
                .env(ISOLATED, "1")
                .output()
                .unwrap();
            assert!(
                output.status.success(),
                "isolated shared-HomSpace test failed:\n{}",
                String::from_utf8_lossy(&output.stderr)
            );
            return;
        }

        // What: contracting the codomain leg of `V <- V` with the domain leg
        // of another stages both operands to `V* <- V*`, which the core
        // shares; the transposed output transforms from that core. Within
        // the first call the rhs and core previews lose their admission to
        // the staged lhs, so their transformers publish under its id: the
        // second call builds nothing.
        let (_left, lhs, _right, rhs) = bound_pair(1, 1);
        tenet_core::clear_structure_caches();
        let lhs_data = [1.5_f64];
        let rhs_data = [-2.0];
        let candidate = contracted_axis_order_candidates(&[0], &[1]).remove(0);
        let run = || {
            tensorcontract_owned_checked_generic_preselected(
                &lhs,
                &lhs_data,
                &rhs,
                &rhs_data,
                TensorContractSpec::new(
                    &[0],
                    &[1],
                    tenet_operations::OutputAxisOrder::Axes(&[1, 0]),
                ),
                1,
                &candidate,
                FusionContractOrientation::LhsRhs,
            )
            .unwrap()
        };
        crate::tree_transform::take_completed_transformer_activity();
        let (cold_space, cold) = run();
        let transformers = crate::tree_transform::take_completed_transformer_activity();
        assert_eq!((transformers.hits, transformers.builds), (0, 3));
        let committed = cache_entries();
        for _ in 0..3 {
            let (space, data) = run();
            let transformers = crate::tree_transform::take_completed_transformer_activity();
            assert_eq!(
                (
                    transformers.hits,
                    transformers.builds,
                    transformers.publications
                ),
                (3, 0, 0)
            );
            assert_eq!(cache_entries(), committed);
            assert_eq!(space.space(), cold_space.space());
            assert_eq!(data[0].to_bits(), cold[0].to_bits());
        }
    }

    #[test]
    #[allow(clippy::arc_with_non_send_sync)]
    fn a_reset_straddling_the_publication_leaves_nothing_resident() {
        const ISOLATED: &str = "TENET_CHECKED_GENERIC_CONTRACT_RESET_STRADDLE_ISOLATED";
        if std::env::var_os(ISOLATED).is_none() {
            let output = std::process::Command::new(std::env::current_exe().unwrap())
                .args([
                    "--exact",
                    "contract::checked_generic::tests::a_reset_straddling_the_publication_leaves_nothing_resident",
                ])
                .env(ISOLATED, "1")
                .output()
                .unwrap();
            assert!(
                output.status.success(),
                "isolated reset-straddle test failed:\n{}",
                String::from_utf8_lossy(&output.stderr)
            );
            return;
        }

        // What: a clear between the commits and the first transformer
        // publication refuses every staged publication (the call's epoch is
        // older), the result is unchanged, and the next call rebuilds.
        let (_left, lhs, _right, rhs) = bound_pair(2, 2);
        let lhs_data = (0..lhs.space().required_len().unwrap())
            .map(|index| index as f64 - 1.5)
            .collect::<Vec<_>>();
        let rhs_data = (0..rhs.space().required_len().unwrap())
            .map(|index| 2.0 - index as f64)
            .collect::<Vec<_>>();
        let mut context =
            TensorContractFusionExecutionContext::<f64, tenet_core::RuleIdentity>::default();
        let mut run = || {
            tensorcontract_owned_checked_generic_in_context(
                &mut context,
                &lhs,
                &lhs_data,
                &rhs,
                &rhs_data,
                TensorContractSpec::new(
                    &[3, 1],
                    &[0, 3],
                    tenet_operations::OutputAxisOrder::Axes(&[2, 0, 3, 1]),
                ),
                2,
            )
            .unwrap()
            .1
        };
        tenet_core::clear_structure_caches();
        let reference = run();
        tenet_core::clear_structure_caches();
        crate::tree_transform::take_completed_transformer_activity();
        crate::tree_transform::before_next_completed_publication(Box::new(
            tenet_core::clear_structure_caches,
        ));
        let straddled = run();
        let transformers = crate::tree_transform::take_completed_transformer_activity();
        assert_eq!((transformers.builds, transformers.publications), (3, 3));
        assert_eq!(cache_entries().1, 0);
        assert_eq!(straddled, reference);
        let rebuilt = run();
        let transformers = crate::tree_transform::take_completed_transformer_activity();
        assert_eq!((transformers.hits, transformers.builds), (0, 3));
        assert_eq!(rebuilt, reference);
    }

    #[test]
    #[allow(clippy::arc_with_non_send_sync)]
    fn checked_generic_gemm_failure_on_a_later_job_returns_the_error_without_publishing() {
        // What: with two coupled sectors the second GEMM job fails after the
        // first wrote its block; the owner returns `Err` (no panic) and no
        // output space or payload is returned.
        let lhs = bound_space(homspace_with(&[(0, 2), (1, 2)], &[(0, 2), (1, 2)]));
        let rhs = bound_space(homspace_with(&[(0, 2), (1, 2)], &[(0, 2), (1, 2)]));
        let candidate = contracted_axis_order_candidates(&[1], &[0]).remove(0);
        let mut transform_backend = DenseTreeTransformOperations::default();
        let mut transform_workspaces = Default::default();
        let mut fusion_workspace = FusionBlockContractWorkspace::default();
        let mut gemm = FailingAtJob {
            calls: 0,
            fail_at: 1,
        };
        let error = tensorcontract_owned_checked_generic_preselected_with_core_gemm(
            &lhs,
            &vec![1.0; lhs.space().required_len().unwrap()],
            &rhs,
            &vec![2.0; rhs.space().required_len().unwrap()],
            TensorContractSpec::with_default_output_order(&[1], &[0]),
            1,
            &candidate,
            FusionContractOrientation::LhsRhs,
            &mut transform_backend,
            &mut transform_workspaces,
            &mut gemm,
            &mut fusion_workspace,
        )
        .unwrap_err();
        assert_eq!(gemm.calls, 2);
        assert!(matches!(
            error,
            CheckedGenericPlanError::Operation(OperationError::StridedKernel { .. })
        ));
    }
}
