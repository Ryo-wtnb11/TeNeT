use tenet_core::{
    CheckedFusionAlgebra, CheckedFusionSpaceError, FusionRule, FusionSpaceAdmission,
    FusionTensorMapSpace, FusionTreeHomSpace, LoweredMultiplicityFreeAlgebra,
    MultiplicityFreeRigidSymbols,
};

use crate::lowering::lower_tensorcontract_adjoint_axes;
use crate::{OperationError, TreeTransformOperation, TreeTransformOperationKind};
use tenet_operations::{OutputAxisOrder, TensorContractSpec, TensorContractSpecOwned};

use super::super::dynamic_space::{
    BoundDynamicFusionMapSpace, DynamicFusionMapSpace, FusionOperandLayout, LayoutKeyBuilder,
    TransformedLayoutProbe,
};
use super::super::structure::TensorContractAxisPlan;

type HomSpaceBuilder<R> = fn(
    &R,
    &FusionTreeHomSpace,
    &FusionTreeHomSpace,
    &[usize],
    &[usize],
    &[usize],
    usize,
) -> Result<FusionTreeHomSpace, OperationError>;

fn encoded_homspace_builder<R: FusionRule>(
    rule: &R,
    lhs: &FusionTreeHomSpace,
    rhs: &FusionTreeHomSpace,
    lhs_axes: &[usize],
    rhs_axes: &[usize],
    output_axes: &[usize],
    dst_rank: usize,
) -> Result<FusionTreeHomSpace, OperationError> {
    FusionTreeHomSpace::tensorcontract_homspace(
        rule,
        lhs,
        rhs,
        lhs_axes,
        rhs_axes,
        output_axes,
        dst_rank,
    )
    .map_err(OperationError::from_core_preserving_context)
}

fn lowered_homspace_builder<R: LoweredMultiplicityFreeAlgebra + CheckedFusionAlgebra>(
    rule: &R,
    lhs: &FusionTreeHomSpace,
    rhs: &FusionTreeHomSpace,
    lhs_axes: &[usize],
    rhs_axes: &[usize],
    output_axes: &[usize],
    dst_rank: usize,
) -> Result<FusionTreeHomSpace, OperationError> {
    FusionTreeHomSpace::try_tensorcontract_homspace_checked(
        rule,
        lhs,
        rhs,
        lhs_axes,
        rhs_axes,
        output_axes,
        dst_rank,
    )
    .map_err(|error| match error {
        CheckedFusionSpaceError::Core(error) => {
            OperationError::from_core_preserving_context(*error)
        }
        CheckedFusionSpaceError::FusionAlgebra(error) => OperationError::FusionAlgebra(error),
        _ => OperationError::InvalidArgument {
            message: "unknown checked fusion metadata error",
        },
    })
}

#[cfg(test)]
std::thread_local! {
    static CANDIDATE_SCORE_CALLS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
    static CONTRACT_PREFLIGHT_CALLS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
    static CANDIDATE_LAYOUT_PROBE_CALLS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

#[cfg(test)]
pub(crate) fn reset_candidate_score_calls() {
    CANDIDATE_SCORE_CALLS.set(0);
}

#[cfg(test)]
pub(crate) fn candidate_score_calls() -> usize {
    CANDIDATE_SCORE_CALLS.get()
}

#[cfg(test)]
pub(crate) fn reset_candidate_build_calls() {
    CONTRACT_PREFLIGHT_CALLS.set(0);
    CANDIDATE_LAYOUT_PROBE_CALLS.set(0);
}

#[cfg(test)]
pub(crate) fn candidate_build_calls() -> (usize, usize) {
    (
        CONTRACT_PREFLIGHT_CALLS.get(),
        CANDIDATE_LAYOUT_PROBE_CALLS.get(),
    )
}

/// A paired ordering of the contracted axes.
///
/// The two vectors are a single permutation: entries at the same position
/// remain paired. Keeping this as one value prevents a cost model from
/// accidentally sorting one operand independently and changing the
/// contraction semantics. This is deliberately only a preparation primitive;
/// it does not select a runtime winner yet.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct ContractAxisOrderCandidate {
    lhs: Vec<usize>,
    rhs: Vec<usize>,
}

impl ContractAxisOrderCandidate {
    #[inline]
    pub(crate) fn lhs(&self) -> &[usize] {
        &self.lhs
    }

    #[inline]
    pub(crate) fn rhs(&self) -> &[usize] {
        &self.rhs
    }
}

/// Operand orientation for a fusion-level contraction candidate.
///
/// Keeping orientation separate prevents reversed candidates from overloading
/// paired-axis ordering with a second meaning.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum FusionContractOrientation {
    LhsRhs,
    RhsLhs,
}

const FORWARD_ORIENTATIONS: [FusionContractOrientation; 1] = [FusionContractOrientation::LhsRhs];
const CACHED_ORIENTATIONS: [FusionContractOrientation; 2] = [
    FusionContractOrientation::LhsRhs,
    FusionContractOrientation::RhsLhs,
];

/// Provider-independent structural facts used by the current fusion selector.
///
/// The value owns no provider, transformed HomSpace, backend, or cache handle.
/// It is therefore safe to carry into later route lowering without retaining
/// the semantic machinery used to prove these facts.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct FusionContractCandidateFacts {
    axis_order: ContractAxisOrderCandidate,
    orientation: FusionContractOrientation,
    lhs_conjugate: bool,
    rhs_conjugate: bool,
    lhs_exact_identity_borrowable: bool,
    rhs_exact_identity_borrowable: bool,
    rhs_requires_twist: bool,
    output_exact_identity: bool,
    lhs_materialized_elements: usize,
    rhs_materialized_elements: usize,
    output_materialized_elements: usize,
    total_materialized_elements: usize,
}

impl FusionContractCandidateFacts {
    #[cfg(test)]
    #[inline]
    pub(crate) fn axis_order(&self) -> &ContractAxisOrderCandidate {
        &self.axis_order
    }

    #[cfg(test)]
    #[inline]
    pub(crate) fn orientation(&self) -> FusionContractOrientation {
        self.orientation
    }

    #[cfg(test)]
    #[inline]
    pub(crate) fn lhs_conjugate(&self) -> bool {
        self.lhs_conjugate
    }

    #[cfg(test)]
    #[inline]
    pub(crate) fn rhs_conjugate(&self) -> bool {
        self.rhs_conjugate
    }

    #[cfg(test)]
    #[inline]
    pub(crate) fn lhs_exact_identity_borrowable(&self) -> bool {
        self.lhs_exact_identity_borrowable
    }

    #[cfg(test)]
    #[inline]
    pub(crate) fn rhs_exact_identity_borrowable(&self) -> bool {
        self.rhs_exact_identity_borrowable
    }

    #[cfg(test)]
    #[inline]
    pub(crate) fn rhs_requires_twist(&self) -> bool {
        self.rhs_requires_twist
    }

    #[cfg(test)]
    #[inline]
    pub(crate) fn output_exact_identity(&self) -> bool {
        self.output_exact_identity
    }

    #[cfg(test)]
    #[inline]
    pub(crate) fn lhs_materialized_elements(&self) -> usize {
        self.lhs_materialized_elements
    }

    #[cfg(test)]
    #[inline]
    pub(crate) fn rhs_materialized_elements(&self) -> usize {
        self.rhs_materialized_elements
    }

    #[cfg(test)]
    #[inline]
    pub(crate) fn output_materialized_elements(&self) -> usize {
        self.output_materialized_elements
    }

    #[inline]
    pub(crate) fn total_materialized_elements(&self) -> usize {
        self.total_materialized_elements
    }
}

#[derive(Debug)]
struct ScoredFusionContractCandidate {
    plan: FusionContractPlan,
    facts: FusionContractCandidateFacts,
}

/// Build the side-sorted candidates for a contraction.
///
/// Sorting is stable and always applies the same permutation to both sides.
/// No fermionic sign is computed here: that belongs to the tree-transform
/// execution of the selected candidate.
pub(crate) fn contracted_axis_order_candidates(
    lhs: &[usize],
    rhs: &[usize],
) -> Vec<ContractAxisOrderCandidate> {
    assert_eq!(
        lhs.len(),
        rhs.len(),
        "paired contraction axes must have equal length"
    );
    let mut lhs_order = (0..lhs.len()).collect::<Vec<_>>();
    lhs_order.sort_by_key(|&i| lhs[i]);
    let lhs_sorted = ContractAxisOrderCandidate {
        lhs: lhs_order.iter().map(|&i| lhs[i]).collect(),
        rhs: lhs_order.iter().map(|&i| rhs[i]).collect(),
    };
    let mut rhs_order = (0..rhs.len()).collect::<Vec<_>>();
    rhs_order.sort_by_key(|&i| rhs[i]);
    let rhs_sorted = ContractAxisOrderCandidate {
        lhs: rhs_order.iter().map(|&i| lhs[i]).collect(),
        rhs: rhs_order.iter().map(|&i| rhs[i]).collect(),
    };
    let mut candidates = vec![lhs_sorted];
    if !candidates.contains(&rhs_sorted) {
        candidates.push(rhs_sorted);
    }
    candidates
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct FusionContractPlan {
    orientation: FusionContractOrientation,
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
    pub(crate) fn orientation(&self) -> FusionContractOrientation {
        self.orientation
    }

    pub(crate) fn require_forward_scratch(&self) -> Result<(), OperationError> {
        if self.orientation == FusionContractOrientation::RhsLhs {
            return Err(OperationError::UnsupportedTensorContractScope {
                message: "caller-owned fusion contraction scratch supports only LhsRhs orientation",
            });
        }
        Ok(())
    }

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
        self.output_transform.kind() == TreeTransformOperationKind::Permute
            && self
                .output_transform
                .codomain_permutation()
                .iter()
                .copied()
                .eq(0..self.core_dst_open_lhs_rank)
            && self
                .output_transform
                .domain_permutation()
                .iter()
                .copied()
                .eq(self.core_dst_open_lhs_rank..core_rank)
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
    dst.validate_rule(rule)?;
    lhs.validate_rule(rule)?;
    rhs.validate_rule(rule)?;
    prepare_tensorcontract_fusion_plan_dyn_raw(
        rule,
        &DynamicFusionMapSpace::from_typed(dst),
        &DynamicFusionMapSpace::from_typed(lhs),
        &DynamicFusionMapSpace::from_typed(rhs),
        axes,
    )
}

/// Dynamic-rank variant of [`prepare_tensorcontract_fusion_plan`] retaining
/// the checked provider authority carried by the input spaces.
pub fn prepare_tensorcontract_fusion_plan_dyn<R>(
    dst: &BoundDynamicFusionMapSpace<R>,
    lhs: &BoundDynamicFusionMapSpace<R>,
    rhs: &BoundDynamicFusionMapSpace<R>,
    axes: TensorContractSpec<'_>,
) -> Result<FusionContractPlan, OperationError>
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64>,
{
    // Why not accept a separate rule: the bound spaces are the semantic
    // authority. The raw core still performs cheap identity checks without
    // re-enumerating any fusion-tree grid.
    prepare_tensorcontract_fusion_plan_dyn_raw(
        lhs.provider(),
        dst.space(),
        lhs.space(),
        rhs.space(),
        axes,
    )
}

#[cfg(test)]
pub(crate) fn prepare_tensorcontract_fusion_plan_dyn_lowered<R>(
    dst: &BoundDynamicFusionMapSpace<R>,
    lhs: &BoundDynamicFusionMapSpace<R>,
    rhs: &BoundDynamicFusionMapSpace<R>,
    axes: TensorContractSpec<'_>,
) -> Result<FusionContractPlan, OperationError>
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64>
        + LoweredMultiplicityFreeAlgebra
        + CheckedFusionAlgebra,
{
    let rule = lhs.provider();
    dst.space().validate_rule(rule)?;
    lhs.space().validate_rule(rule)?;
    rhs.space().validate_rule(rule)?;
    let lowered_axes = lower_tensorcontract_adjoint_axes(
        lhs.space().nout(),
        lhs.space().nin(),
        rhs.space().nout(),
        rhs.space().nin(),
        axes,
    )?;
    let lhs_adjoint;
    let lhs_space = if axes.lhs_conjugate() {
        lhs_adjoint = lhs.space().adjoint_view()?;
        &lhs_adjoint
    } else {
        lhs.space()
    };
    let rhs_adjoint;
    let rhs_space = if axes.rhs_conjugate() {
        rhs_adjoint = rhs.space().adjoint_view()?;
        &rhs_adjoint
    } else {
        rhs.space()
    };
    select_tensorcontract_fusion_plan_from_spaces_with_probe(
        rule,
        dst.space(),
        lhs_space,
        rhs_space,
        lowered_axes.as_spec(),
        lowered_axes.lhs_storage_conjugate(),
        lowered_axes.rhs_storage_conjugate(),
        lowered_layout_probe::<R>,
        lowered_homspace_builder::<R>,
        Some(dst.layout_primer()),
    )
}

pub(crate) fn prepare_tensorcontract_fusion_plan_dyn_raw<R>(
    rule: &R,
    dst: &DynamicFusionMapSpace,
    lhs: &DynamicFusionMapSpace,
    rhs: &DynamicFusionMapSpace,
    axes: TensorContractSpec<'_>,
) -> Result<FusionContractPlan, OperationError>
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64>,
{
    prepare_tensorcontract_fusion_plan_dyn_raw_with_orientations(
        rule,
        dst,
        lhs,
        rhs,
        axes,
        &FORWARD_ORIENTATIONS,
    )
}

pub(crate) fn prepare_tensorcontract_fusion_plan_dyn_raw_canonical<R>(
    rule: &R,
    dst: &DynamicFusionMapSpace,
    lhs: &DynamicFusionMapSpace,
    rhs: &DynamicFusionMapSpace,
    axes: TensorContractSpec<'_>,
) -> Result<FusionContractPlan, OperationError>
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64>,
{
    prepare_tensorcontract_fusion_plan_dyn_raw_with_orientations(
        rule,
        dst,
        lhs,
        rhs,
        axes,
        &CACHED_ORIENTATIONS,
    )
}

fn prepare_tensorcontract_fusion_plan_dyn_raw_with_orientations<R>(
    rule: &R,
    dst: &DynamicFusionMapSpace,
    lhs: &DynamicFusionMapSpace,
    rhs: &DynamicFusionMapSpace,
    axes: TensorContractSpec<'_>,
    orientations: &[FusionContractOrientation],
) -> Result<FusionContractPlan, OperationError>
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64>,
{
    dst.validate_rule(rule)?;
    lhs.validate_rule(rule)?;
    rhs.validate_rule(rule)?;
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
    select_tensorcontract_fusion_plan_from_spaces_with_orientations(
        rule,
        dst,
        lhs,
        rhs,
        lowered_axes.as_spec(),
        lowered_axes.lhs_storage_conjugate(),
        lowered_axes.rhs_storage_conjugate(),
        orientations,
    )
}

pub(crate) fn prepare_tensorcontract_fusion_plan_dyn_prelowered_canonical<R>(
    rule: &R,
    dst: &DynamicFusionMapSpace,
    lhs: &FusionOperandLayout<'_>,
    rhs: &FusionOperandLayout<'_>,
    axes: TensorContractSpec<'_>,
    primer: LayoutKeyBuilder<R>,
) -> Result<FusionContractPlan, OperationError>
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64>,
{
    prepare_tensorcontract_fusion_plan_from_operands(
        rule,
        dst,
        lhs,
        rhs,
        axes,
        |rule, source: &FusionOperandLayout<'_>, operation, _| {
            if source.is_direct() {
                encoded_layout_probe(rule, source.storage_space(), operation, None)
            } else {
                source.transformed_layout_probe(
                    rule,
                    operation,
                    super::super::dynamic_space::encoded_layout_primer::<R>,
                )
            }
        },
        encoded_homspace_builder::<R>,
        primer,
    )
}

/// Prepare a plan with an explicitly paired contraction-axis ordering.
///
/// This is intentionally crate-private: it is an oracle seam for validating
/// candidate permutations before a layout cost model is enabled. The caller
/// must provide the same axis sets as `axes`; only their order may differ.
#[cfg(test)]
pub(crate) fn prepare_tensorcontract_fusion_plan_dyn_raw_with_axis_order<R>(
    rule: &R,
    dst: &DynamicFusionMapSpace,
    lhs: &DynamicFusionMapSpace,
    rhs: &DynamicFusionMapSpace,
    axes: TensorContractSpec<'_>,
    candidate: &ContractAxisOrderCandidate,
) -> Result<FusionContractPlan, OperationError>
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64>,
{
    fn same_axes(a: &[usize], b: &[usize]) -> bool {
        let mut a = a.to_vec();
        let mut b = b.to_vec();
        a.sort_unstable();
        b.sort_unstable();
        a == b
    }
    if !same_axes(axes.lhs_contracting_axes(), candidate.lhs())
        || !same_axes(axes.rhs_contracting_axes(), candidate.rhs())
    {
        return Err(OperationError::InvalidArgument {
            message: "candidate must preserve contracted axis sets",
        });
    }
    let candidate_axes = TensorContractSpec::new_with_conjugation(
        candidate.lhs(),
        candidate.rhs(),
        axes.output_permutation(),
        axes.lhs_conjugate(),
        axes.rhs_conjugate(),
    );
    prepare_tensorcontract_fusion_plan_dyn_raw_fixed(rule, dst, lhs, rhs, candidate_axes)
}

#[cfg(test)]
pub(crate) fn prepare_tensorcontract_fusion_plan_dyn_raw_with_axis_order_and_orientation<R>(
    rule: &R,
    dst: &DynamicFusionMapSpace,
    lhs: &DynamicFusionMapSpace,
    rhs: &DynamicFusionMapSpace,
    axes: TensorContractSpec<'_>,
    candidate: &ContractAxisOrderCandidate,
    orientation: FusionContractOrientation,
) -> Result<FusionContractPlan, OperationError>
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64>,
{
    let plan = prepare_tensorcontract_fusion_plan_dyn_raw_with_axis_order(
        rule, dst, lhs, rhs, axes, candidate,
    )?;
    Ok(orient_fusion_contract_plan(plan, orientation))
}

fn orient_fusion_contract_plan(
    mut plan: FusionContractPlan,
    orientation: FusionContractOrientation,
) -> FusionContractPlan {
    if orientation == FusionContractOrientation::RhsLhs {
        let lhs_open_rank = plan.lhs_open_rank;
        let rhs_open_rank = plan.rhs_open_rank;
        let contract_rank = plan.lhs_contract_rank;
        let semantic_to_core = |axis: usize| {
            if axis < lhs_open_rank {
                axis + rhs_open_rank
            } else {
                axis - lhs_open_rank
            }
        };
        if plan.output_transform.kind() != TreeTransformOperationKind::Permute {
            unreachable!("fusion contraction output lowering uses a permutation");
        }
        let output_axes = plan
            .output_transform
            .codomain_permutation()
            .iter()
            .chain(plan.output_transform.domain_permutation())
            .copied()
            .map(semantic_to_core)
            .collect::<Vec<_>>();
        let dst_nout = plan.output_transform.codomain_permutation().len();
        if plan.lhs_transform.kind() != TreeTransformOperationKind::Permute {
            unreachable!("fusion contraction source lowering uses a permutation");
        }
        let lhs_open_axes = plan.lhs_transform.codomain_permutation().to_vec();
        let lhs_contracting_axes = plan.lhs_transform.domain_permutation().to_vec();
        if plan.rhs_transform.kind() != TreeTransformOperationKind::Permute {
            unreachable!("fusion contraction source lowering uses a permutation");
        }
        let rhs_contracting_axes = plan.rhs_transform.codomain_permutation().to_vec();
        let rhs_open_axes = plan.rhs_transform.domain_permutation().to_vec();
        plan.orientation = orientation;
        plan.lhs_transform = TreeTransformOperation::permute(lhs_contracting_axes, lhs_open_axes);
        plan.rhs_transform = TreeTransformOperation::permute(rhs_open_axes, rhs_contracting_axes);
        plan.core_axes = TensorContractSpecOwned::new(
            (rhs_open_rank..rhs_open_rank + contract_rank).collect(),
            (0..contract_rank).collect(),
            (0..lhs_open_rank + rhs_open_rank).collect(),
        );
        plan.output_transform = TreeTransformOperation::permute(
            output_axes[..dst_nout].iter().copied(),
            output_axes[dst_nout..].iter().copied(),
        );
        plan.core_dst_open_lhs_rank = rhs_open_rank;
        plan.core_dst_open_rhs_rank = lhs_open_rank;
    }
    plan
}

#[cfg(test)]
fn prepare_tensorcontract_fusion_plan_dyn_raw_fixed<R>(
    rule: &R,
    dst: &DynamicFusionMapSpace,
    lhs: &DynamicFusionMapSpace,
    rhs: &DynamicFusionMapSpace,
    axes: TensorContractSpec<'_>,
) -> Result<FusionContractPlan, OperationError>
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64>,
{
    dst.validate_rule(rule)?;
    lhs.validate_rule(rule)?;
    rhs.validate_rule(rule)?;
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
    validate_tensorcontract_fusion_plan_inputs(
        rule,
        dst,
        lhs,
        rhs,
        lowered_axes.as_spec(),
        encoded_homspace_builder::<R>,
    )?;
    compile_tensorcontract_fusion_plan_from_spaces(
        dst,
        lhs,
        rhs,
        lowered_axes.as_spec(),
        lowered_axes.lhs_storage_conjugate(),
        lowered_axes.rhs_storage_conjugate(),
    )
}

#[cfg(test)]
pub(crate) fn prepare_tensorcontract_fusion_candidate_facts_dyn_raw<R>(
    rule: &R,
    dst: &DynamicFusionMapSpace,
    lhs: &DynamicFusionMapSpace,
    rhs: &DynamicFusionMapSpace,
    axes: TensorContractSpec<'_>,
) -> Result<Vec<FusionContractCandidateFacts>, OperationError>
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64>,
{
    dst.validate_rule(rule)?;
    lhs.validate_rule(rule)?;
    rhs.validate_rule(rule)?;
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
    fusion_contract_candidate_facts_from_spaces_with_probe(
        rule,
        dst,
        lhs,
        rhs,
        lowered_axes.as_spec(),
        lowered_axes.lhs_storage_conjugate(),
        lowered_axes.rhs_storage_conjugate(),
        encoded_layout_probe::<R>,
        encoded_homspace_builder::<R>,
        None,
        &CACHED_ORIENTATIONS,
    )
}

#[allow(clippy::too_many_arguments)]
fn select_tensorcontract_fusion_plan_from_spaces_with_orientations<R>(
    rule: &R,
    dst: &DynamicFusionMapSpace,
    lhs: &DynamicFusionMapSpace,
    rhs: &DynamicFusionMapSpace,
    axes: TensorContractSpec<'_>,
    lhs_source_conjugate: bool,
    rhs_source_conjugate: bool,
    orientations: &[FusionContractOrientation],
) -> Result<FusionContractPlan, OperationError>
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64>,
{
    select_tensorcontract_fusion_plan_from_spaces_with_probe_and_orientations(
        rule,
        dst,
        lhs,
        rhs,
        axes,
        lhs_source_conjugate,
        rhs_source_conjugate,
        encoded_layout_probe::<R>,
        encoded_homspace_builder::<R>,
        None,
        orientations,
    )
}

#[cfg(test)]
#[allow(clippy::too_many_arguments)]
fn fusion_contract_candidate_facts_from_spaces_with_probe<R>(
    rule: &R,
    dst: &DynamicFusionMapSpace,
    lhs: &DynamicFusionMapSpace,
    rhs: &DynamicFusionMapSpace,
    axes: TensorContractSpec<'_>,
    lhs_source_conjugate: bool,
    rhs_source_conjugate: bool,
    probe: LayoutProbeBuilder<R>,
    homspace_builder: HomSpaceBuilder<R>,
    primer: Option<LayoutKeyBuilder<R>>,
    orientations: &[FusionContractOrientation],
) -> Result<Vec<FusionContractCandidateFacts>, OperationError>
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64>,
{
    validate_tensorcontract_fusion_plan_inputs(rule, dst, lhs, rhs, axes, homspace_builder)?;
    orientations
        .iter()
        .flat_map(|&orientation| {
            contracted_axis_order_candidates(
                axes.lhs_contracting_axes(),
                axes.rhs_contracting_axes(),
            )
            .into_iter()
            .map(move |candidate| (orientation, candidate))
        })
        .map(|(orientation, candidate)| {
            let candidate_axes = TensorContractSpec::new(
                candidate.lhs(),
                candidate.rhs(),
                axes.output_permutation(),
            );
            let plan = orient_fusion_contract_plan(
                compile_tensorcontract_fusion_plan_from_spaces(
                    dst,
                    lhs,
                    rhs,
                    candidate_axes,
                    lhs_source_conjugate,
                    rhs_source_conjugate,
                )?,
                orientation,
            );
            score_fusion_contract_candidate(rule, dst, lhs, rhs, candidate, plan, probe, primer)
                .map(|scored| scored.facts)
        })
        .collect()
}

#[cfg(test)]
type LayoutProbeBuilder<R> = for<'a> fn(
    &'a R,
    &'a DynamicFusionMapSpace,
    &'a TreeTransformOperation,
    Option<LayoutKeyBuilder<R>>,
) -> Result<TransformedLayoutProbe, OperationError>;

trait ContractPlanSource {
    fn homspace(&self) -> &FusionTreeHomSpace;
    fn nout(&self) -> usize;
    fn rank(&self) -> usize;
    fn admission(&self) -> &FusionSpaceAdmission;
    fn storage_space(&self) -> &DynamicFusionMapSpace;
}

impl ContractPlanSource for DynamicFusionMapSpace {
    fn homspace(&self) -> &FusionTreeHomSpace {
        self.homspace()
    }

    fn nout(&self) -> usize {
        self.nout()
    }

    fn rank(&self) -> usize {
        self.rank()
    }

    fn admission(&self) -> &FusionSpaceAdmission {
        self.admission()
    }

    fn storage_space(&self) -> &DynamicFusionMapSpace {
        self
    }
}

impl ContractPlanSource for FusionOperandLayout<'_> {
    fn homspace(&self) -> &FusionTreeHomSpace {
        self.homspace()
    }

    fn nout(&self) -> usize {
        self.nout()
    }

    fn rank(&self) -> usize {
        self.rank()
    }

    fn admission(&self) -> &FusionSpaceAdmission {
        self.admission()
    }

    fn storage_space(&self) -> &DynamicFusionMapSpace {
        self.storage_space()
    }
}

fn encoded_layout_probe<R>(
    rule: &R,
    space: &DynamicFusionMapSpace,
    operation: &TreeTransformOperation,
    _primer: Option<LayoutKeyBuilder<R>>,
) -> Result<TransformedLayoutProbe, OperationError>
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64>,
{
    space.transformed_layout_probe(rule, operation)
}

fn lowered_layout_probe<R>(
    rule: &R,
    space: &DynamicFusionMapSpace,
    operation: &TreeTransformOperation,
    primer: Option<LayoutKeyBuilder<R>>,
) -> Result<TransformedLayoutProbe, OperationError>
where
    R: LoweredMultiplicityFreeAlgebra + CheckedFusionAlgebra,
{
    space.transformed_layout_probe_with_primer(
        rule,
        operation,
        primer.expect("lowered layout probe requires metadata primer"),
    )
}

#[cfg(test)]
fn select_tensorcontract_fusion_plan_from_spaces_with_probe<R>(
    rule: &R,
    dst: &DynamicFusionMapSpace,
    lhs: &DynamicFusionMapSpace,
    rhs: &DynamicFusionMapSpace,
    axes: TensorContractSpec<'_>,
    lhs_source_conjugate: bool,
    rhs_source_conjugate: bool,
    probe: LayoutProbeBuilder<R>,
    homspace_builder: HomSpaceBuilder<R>,
    primer: Option<LayoutKeyBuilder<R>>,
) -> Result<FusionContractPlan, OperationError>
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64>,
{
    select_tensorcontract_fusion_plan_from_spaces_with_probe_and_orientations(
        rule,
        dst,
        lhs,
        rhs,
        axes,
        lhs_source_conjugate,
        rhs_source_conjugate,
        probe,
        homspace_builder,
        primer,
        &FORWARD_ORIENTATIONS,
    )
}

#[allow(clippy::too_many_arguments)]
fn select_tensorcontract_fusion_plan_from_spaces_with_probe_and_orientations<R, S, P>(
    rule: &R,
    dst: &DynamicFusionMapSpace,
    lhs: &S,
    rhs: &S,
    axes: TensorContractSpec<'_>,
    lhs_source_conjugate: bool,
    rhs_source_conjugate: bool,
    probe: P,
    homspace_builder: HomSpaceBuilder<R>,
    primer: Option<LayoutKeyBuilder<R>>,
    orientations: &[FusionContractOrientation],
) -> Result<FusionContractPlan, OperationError>
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64>,
    S: ContractPlanSource,
    P: for<'a> Fn(
            &'a R,
            &'a S,
            &'a TreeTransformOperation,
            Option<LayoutKeyBuilder<R>>,
        ) -> Result<TransformedLayoutProbe, OperationError>
        + Copy,
{
    validate_tensorcontract_fusion_plan_inputs(rule, dst, lhs, rhs, axes, homspace_builder)?;
    let candidates =
        contracted_axis_order_candidates(axes.lhs_contracting_axes(), axes.rhs_contracting_axes());
    let complete = matches!(dst.admission(), FusionSpaceAdmission::Complete(_))
        && matches!(lhs.admission(), FusionSpaceAdmission::Complete(_))
        && matches!(rhs.admission(), FusionSpaceAdmission::Complete(_));

    let mut best = None;
    let mut first_candidate_error = None;
    for &orientation in orientations {
        for candidate in &candidates {
            let scored = (|| {
                let candidate_axes = TensorContractSpec::new(
                    candidate.lhs(),
                    candidate.rhs(),
                    axes.output_permutation(),
                );
                let plan = orient_fusion_contract_plan(
                    compile_tensorcontract_fusion_plan_from_ranks(
                        dst.nout(),
                        dst.rank(),
                        lhs.rank(),
                        rhs.rank(),
                        candidate_axes,
                        lhs_source_conjugate,
                        rhs_source_conjugate,
                    )?,
                    orientation,
                );
                if complete {
                    let facts = score_complete_fusion_contract_candidate(
                        rule,
                        dst,
                        lhs,
                        rhs,
                        candidate.clone(),
                        &plan,
                    )?;
                    Ok(ScoredFusionContractCandidate { plan, facts })
                } else {
                    score_fusion_contract_candidate(
                        rule,
                        dst,
                        lhs,
                        rhs,
                        candidate.clone(),
                        plan,
                        probe,
                        primer,
                    )
                }
            })();
            let scored = match scored {
                Ok(scored) => scored,
                Err(error) => {
                    // Why not fail immediately: candidate-local layout capability
                    // does not invalidate a later stable contraction candidate.
                    first_candidate_error.get_or_insert(error);
                    continue;
                }
            };
            if best
                .as_ref()
                .is_none_or(|best: &ScoredFusionContractCandidate| {
                    scored.facts.total_materialized_elements()
                        < best.facts.total_materialized_elements()
                })
            {
                best = Some(scored);
            }
        }
    }
    match best {
        Some(best) => Ok(best.plan),
        None => Err(first_candidate_error
            .expect("paired contraction always has at least the LHS-sorted candidate")),
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn prepare_tensorcontract_fusion_plan_dyn_prelowered_with_primer_canonical<R>(
    rule: &R,
    dst: &DynamicFusionMapSpace,
    lhs: &FusionOperandLayout<'_>,
    rhs: &FusionOperandLayout<'_>,
    axes: TensorContractSpec<'_>,
    primer: LayoutKeyBuilder<R>,
) -> Result<FusionContractPlan, OperationError>
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64>
        + LoweredMultiplicityFreeAlgebra
        + CheckedFusionAlgebra,
{
    prepare_tensorcontract_fusion_plan_from_operands(
        rule,
        dst,
        lhs,
        rhs,
        axes,
        |rule, source: &FusionOperandLayout<'_>, operation, primer| {
            let primer = primer.ok_or(OperationError::InvalidArgument {
                message: "lowered prelowered plan requires a layout primer",
            })?;
            if source.is_direct() {
                lowered_layout_probe(rule, source.storage_space(), operation, Some(primer))
            } else {
                source.transformed_layout_probe(rule, operation, primer)
            }
        },
        lowered_homspace_builder::<R>,
        primer,
    )
}

#[allow(clippy::too_many_arguments)]
fn prepare_tensorcontract_fusion_plan_from_operands<R, P>(
    rule: &R,
    dst: &DynamicFusionMapSpace,
    lhs: &FusionOperandLayout<'_>,
    rhs: &FusionOperandLayout<'_>,
    axes: TensorContractSpec<'_>,
    probe: P,
    homspace_builder: HomSpaceBuilder<R>,
    primer: LayoutKeyBuilder<R>,
) -> Result<FusionContractPlan, OperationError>
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64>,
    P: for<'a> Fn(
            &'a R,
            &'a FusionOperandLayout<'_>,
            &'a TreeTransformOperation,
            Option<LayoutKeyBuilder<R>>,
        ) -> Result<TransformedLayoutProbe, OperationError>
        + Copy,
{
    dst.validate_rule(rule)?;
    if axes.lhs_conjugate() != lhs.storage_conjugate()
        || axes.rhs_conjugate() != rhs.storage_conjugate()
    {
        return Err(OperationError::InvalidArgument {
            message: "prelowered operand flags must match the contraction cache key",
        });
    }
    let logical_axes = TensorContractSpec::new(
        axes.lhs_contracting_axes(),
        axes.rhs_contracting_axes(),
        axes.output_permutation(),
    );
    select_tensorcontract_fusion_plan_from_spaces_with_probe_and_orientations(
        rule,
        dst,
        lhs,
        rhs,
        logical_axes,
        lhs.storage_conjugate(),
        rhs.storage_conjugate(),
        probe,
        homspace_builder,
        Some(primer),
        &CACHED_ORIENTATIONS,
    )
}

fn source_contract_homspace_requires_twist<R>(
    rule: &R,
    source: &FusionTreeHomSpace,
    contracting_axes: &[usize],
) -> Result<bool, OperationError>
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64>,
{
    // Why not permute a loser HomSpace: external-axis duality is unchanged by
    // the candidate permutation, so core-right twist can be read at its source axes.
    super::super::resolution::rhs_contract_homspace_requires_twist(
        rule,
        source,
        TensorContractSpec::new(&[], contracting_axes, OutputAxisOrder::identity()),
    )
}

// Why not resolve every length eagerly: identity-borrowed storage never uses
// that count and must not gain an overflow/error from an unused route.
enum CandidateRequiredLen<'a> {
    Known(usize),
    Space(&'a DynamicFusionMapSpace),
}

impl CandidateRequiredLen<'_> {
    fn get(self) -> Result<usize, OperationError> {
        match self {
            Self::Known(len) => Ok(len),
            Self::Space(space) => space
                .required_len()
                .map_err(OperationError::from_core_preserving_context),
        }
    }
}

struct FusionContractMaterializationInputs<'a> {
    lhs_exact_identity_borrowable: bool,
    rhs_exact_identity_borrowable: bool,
    core_right_requires_twist: bool,
    lhs_required_len: CandidateRequiredLen<'a>,
    rhs_required_len: CandidateRequiredLen<'a>,
    output_required_len: CandidateRequiredLen<'a>,
}

fn fusion_contract_candidate_facts(
    axis_order: ContractAxisOrderCandidate,
    plan: &FusionContractPlan,
    inputs: FusionContractMaterializationInputs<'_>,
) -> Result<FusionContractCandidateFacts, OperationError> {
    let reverse = plan.orientation() == FusionContractOrientation::RhsLhs;
    let lhs_requires_twist = reverse && inputs.core_right_requires_twist;
    let rhs_requires_twist = !reverse && inputs.core_right_requires_twist;
    let lhs_materialized_elements = if inputs.lhs_exact_identity_borrowable && !lhs_requires_twist {
        0
    } else {
        inputs.lhs_required_len.get()?
    };
    let rhs_materialized_elements = if inputs.rhs_exact_identity_borrowable && !rhs_requires_twist {
        0
    } else {
        inputs.rhs_required_len.get()?
    };
    let output_exact_identity = plan.output_transform_is_identity();
    let output_materialized_elements = if output_exact_identity {
        0
    } else {
        inputs.output_required_len.get()?
    };
    let total_materialized_elements = lhs_materialized_elements
        .checked_add(rhs_materialized_elements)
        .and_then(|total| total.checked_add(output_materialized_elements))
        .ok_or_else(|| {
            OperationError::from_core_preserving_context(
                tenet_core::CoreError::ElementCountOverflow,
            )
        })?;
    Ok(FusionContractCandidateFacts {
        axis_order,
        orientation: plan.orientation(),
        lhs_conjugate: plan.lhs_source_conjugate(),
        rhs_conjugate: plan.rhs_source_conjugate(),
        lhs_exact_identity_borrowable: inputs.lhs_exact_identity_borrowable,
        rhs_exact_identity_borrowable: inputs.rhs_exact_identity_borrowable,
        rhs_requires_twist,
        output_exact_identity,
        lhs_materialized_elements,
        rhs_materialized_elements,
        output_materialized_elements,
        total_materialized_elements,
    })
}

fn score_complete_fusion_contract_candidate<R, S>(
    rule: &R,
    dst: &DynamicFusionMapSpace,
    lhs: &S,
    rhs: &S,
    axis_order: ContractAxisOrderCandidate,
    plan: &FusionContractPlan,
) -> Result<FusionContractCandidateFacts, OperationError>
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64>,
    S: ContractPlanSource,
{
    #[cfg(test)]
    CANDIDATE_SCORE_CALLS.set(CANDIDATE_SCORE_CALLS.get() + 1);
    // Why not probe the transformed layouts: Complete admission proves the
    // canonical full basis, so permutation preserves reduced element count and
    // an exact identity operation preserves the source structure.
    let lhs_exact_identity_borrowable = super::super::dynamic::source_layout_metadata_is_borrowable(
        lhs.storage_space(),
        lhs.nout(),
        lhs.rank(),
        || true,
        plan.lhs_transform(),
        plan.lhs_source_conjugate(),
    );
    let rhs_exact_identity_borrowable = super::super::dynamic::source_layout_metadata_is_borrowable(
        rhs.storage_space(),
        rhs.nout(),
        rhs.rank(),
        || true,
        plan.rhs_transform(),
        plan.rhs_source_conjugate(),
    );
    let (core_right, core_right_axes) = match plan.orientation() {
        FusionContractOrientation::LhsRhs => (rhs, axis_order.rhs()),
        FusionContractOrientation::RhsLhs => (lhs, axis_order.lhs()),
    };
    let core_right_requires_twist =
        source_contract_homspace_requires_twist(rule, core_right.homspace(), core_right_axes)?;
    fusion_contract_candidate_facts(
        axis_order,
        plan,
        FusionContractMaterializationInputs {
            lhs_exact_identity_borrowable,
            rhs_exact_identity_borrowable,
            core_right_requires_twist,
            lhs_required_len: CandidateRequiredLen::Space(lhs.storage_space()),
            rhs_required_len: CandidateRequiredLen::Space(rhs.storage_space()),
            output_required_len: CandidateRequiredLen::Space(dst),
        },
    )
}

fn score_fusion_contract_candidate<R, S, P>(
    rule: &R,
    dst: &DynamicFusionMapSpace,
    lhs: &S,
    rhs: &S,
    axis_order: ContractAxisOrderCandidate,
    plan: FusionContractPlan,
    probe: P,
    primer: Option<LayoutKeyBuilder<R>>,
) -> Result<ScoredFusionContractCandidate, OperationError>
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64>,
    S: ContractPlanSource,
    P: for<'a> Fn(
            &'a R,
            &'a S,
            &'a TreeTransformOperation,
            Option<LayoutKeyBuilder<R>>,
        ) -> Result<TransformedLayoutProbe, OperationError>
        + Copy,
{
    #[cfg(test)]
    {
        CANDIDATE_SCORE_CALLS.set(CANDIDATE_SCORE_CALLS.get() + 1);
        CANDIDATE_LAYOUT_PROBE_CALLS.set(CANDIDATE_LAYOUT_PROBE_CALLS.get() + 2);
    }
    let lhs_core = probe(rule, lhs, plan.lhs_transform(), primer)?;
    let rhs_core = probe(rule, rhs, plan.rhs_transform(), primer)?;
    let lhs_exact_identity_borrowable = super::super::dynamic::source_layout_metadata_is_borrowable(
        lhs.storage_space(),
        lhs_core.nout,
        lhs_core.homspace.rank(),
        || lhs_core.homspace == *lhs.homspace(),
        plan.lhs_transform(),
        plan.lhs_source_conjugate(),
    ) && lhs_core.source_structure_matches;
    let rhs_exact_identity_borrowable = super::super::dynamic::source_layout_metadata_is_borrowable(
        rhs.storage_space(),
        rhs_core.nout,
        rhs_core.homspace.rank(),
        || rhs_core.homspace == *rhs.homspace(),
        plan.rhs_transform(),
        plan.rhs_source_conjugate(),
    ) && rhs_core.source_structure_matches;
    let reverse = plan.orientation() == FusionContractOrientation::RhsLhs;
    let core_right = if reverse { &lhs_core } else { &rhs_core };
    let core_right_requires_twist = super::super::resolution::rhs_contract_homspace_requires_twist(
        rule,
        &core_right.homspace,
        plan.core_axes().as_spec(),
    )?;
    let facts = fusion_contract_candidate_facts(
        axis_order,
        &plan,
        FusionContractMaterializationInputs {
            lhs_exact_identity_borrowable,
            rhs_exact_identity_borrowable,
            core_right_requires_twist,
            lhs_required_len: CandidateRequiredLen::Known(lhs_core.required_len),
            rhs_required_len: CandidateRequiredLen::Known(rhs_core.required_len),
            output_required_len: CandidateRequiredLen::Space(dst),
        },
    )?;
    Ok(ScoredFusionContractCandidate { plan, facts })
}

fn validate_tensorcontract_fusion_plan_inputs<R, S>(
    rule: &R,
    dst: &DynamicFusionMapSpace,
    lhs: &S,
    rhs: &S,
    axes: TensorContractSpec<'_>,
    homspace_builder: HomSpaceBuilder<R>,
) -> Result<(), OperationError>
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64>,
    S: ContractPlanSource,
{
    #[cfg(test)]
    CONTRACT_PREFLIGHT_CALLS.set(CONTRACT_PREFLIGHT_CALLS.get() + 1);
    let axis_plan = TensorContractAxisPlan::compile(lhs.rank(), rhs.rank(), dst.rank(), axes)?;
    let expected_homspace = homspace_builder(
        rule,
        lhs.homspace(),
        rhs.homspace(),
        axes.lhs_contracting_axes(),
        axes.rhs_contracting_axes(),
        axis_plan.output_axes.as_slice(),
        dst.nout(),
    )?;
    if &expected_homspace != dst.homspace() {
        return Err(OperationError::StructureMismatch { tensor: "dst" });
    }
    Ok(())
}

#[cfg(test)]
fn compile_tensorcontract_fusion_plan_from_spaces(
    dst: &DynamicFusionMapSpace,
    lhs: &DynamicFusionMapSpace,
    rhs: &DynamicFusionMapSpace,
    axes: TensorContractSpec<'_>,
    lhs_source_conjugate: bool,
    rhs_source_conjugate: bool,
) -> Result<FusionContractPlan, OperationError> {
    compile_tensorcontract_fusion_plan_from_ranks(
        dst.nout(),
        dst.rank(),
        lhs.rank(),
        rhs.rank(),
        axes,
        lhs_source_conjugate,
        rhs_source_conjugate,
    )
}

#[allow(clippy::too_many_arguments)]
fn compile_tensorcontract_fusion_plan_from_ranks(
    dst_nout: usize,
    dst_rank: usize,
    lhs_rank: usize,
    rhs_rank: usize,
    axes: TensorContractSpec<'_>,
    lhs_source_conjugate: bool,
    rhs_source_conjugate: bool,
) -> Result<FusionContractPlan, OperationError> {
    let axis_plan = TensorContractAxisPlan::compile(lhs_rank, rhs_rank, dst_rank, axes)?;
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
        orientation: FusionContractOrientation::LhsRhs,
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

#[cfg(test)]
mod tests {
    use super::{
        candidate_build_calls, contracted_axis_order_candidates,
        prepare_tensorcontract_fusion_candidate_facts_dyn_raw,
        prepare_tensorcontract_fusion_plan_dyn_prelowered_canonical,
        prepare_tensorcontract_fusion_plan_dyn_prelowered_with_primer_canonical,
        prepare_tensorcontract_fusion_plan_dyn_raw,
        prepare_tensorcontract_fusion_plan_dyn_raw_canonical,
        prepare_tensorcontract_fusion_plan_dyn_raw_with_axis_order_and_orientation,
        reset_candidate_build_calls, reset_candidate_score_calls, FusionContractOrientation,
    };
    use crate::contract::dynamic_space::{
        encoded_layout_primer, lowered_metadata_dispatcher, MetadataOutput, MetadataRequest,
    };
    use crate::contract::{DynamicFusionMapSpace, FusionOperand};
    use crate::{TreeTransformOperation, TreeTransformOperationKind};
    use std::sync::Arc;
    use tenet_core::{
        BlockKey, FermionParityFusionRule, FusionProductSpace, FusionTensorMapSpace,
        FusionTreeHomSpace, SU2FusionRule, SectorId, SectorLeg, TensorMapSpace, U1FusionRule,
    };
    use tenet_operations::{OutputAxisOrder, TensorContractSpec};

    std::thread_local! {
        static REJECT_NEXT_PROBE: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
        static OPERAND_PRIMER_CALLS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
    }

    fn rejecting_operand_primer(
        _rule: &U1FusionRule,
        _request: MetadataRequest<'_>,
    ) -> Result<MetadataOutput, crate::OperationError> {
        OPERAND_PRIMER_CALLS.set(OPERAND_PRIMER_CALLS.get() + 1);
        Err(crate::OperationError::InvalidArgument {
            message: "encoded operand plan called supplied primer",
        })
    }

    fn counting_lowered_operand_primer(
        rule: &U1FusionRule,
        request: MetadataRequest<'_>,
    ) -> Result<MetadataOutput, crate::OperationError> {
        OPERAND_PRIMER_CALLS.set(OPERAND_PRIMER_CALLS.get() + 1);
        lowered_metadata_dispatcher(rule, request)
    }

    fn reject_next_probe(
        rule: &U1FusionRule,
        space: &DynamicFusionMapSpace,
        operation: &TreeTransformOperation,
        primer: Option<super::LayoutKeyBuilder<U1FusionRule>>,
    ) -> Result<super::TransformedLayoutProbe, crate::OperationError> {
        if REJECT_NEXT_PROBE.replace(false) {
            Err(crate::OperationError::UnsupportedTensorContractScope {
                message: "test candidate layout unavailable",
            })
        } else {
            super::encoded_layout_probe(rule, space, operation, primer)
        }
    }

    fn single_sector_typed_space(
        rule: &U1FusionRule,
        dimensions: [usize; 4],
    ) -> FusionTensorMapSpace<2, 2> {
        let homspace = FusionTreeHomSpace::from_sector_ids(
            [(0, dimensions[0]), (0, dimensions[1])],
            [(0, dimensions[2]), (0, dimensions[3])],
        );
        FusionTensorMapSpace::from_degeneracy_shapes(
            TensorMapSpace::<2, 2>::from_dims(
                [dimensions[0], dimensions[1]],
                [dimensions[2], dimensions[3]],
            )
            .unwrap(),
            homspace,
            rule,
            [dimensions.to_vec()],
        )
        .unwrap()
    }

    fn single_sector_space(rule: &U1FusionRule, dimensions: [usize; 4]) -> DynamicFusionMapSpace {
        DynamicFusionMapSpace::from_typed(&single_sector_typed_space(rule, dimensions))
    }

    fn subset_copy(
        rule: &U1FusionRule,
        complete: &FusionTensorMapSpace<2, 2>,
    ) -> DynamicFusionMapSpace {
        let subset = FusionTensorMapSpace::from_shared_subblock_structure(
            complete.dense_space().clone(),
            complete.homspace().clone(),
            Arc::clone(complete.subblock_structure()),
        )
        .unwrap()
        .try_bind_rule(rule)
        .unwrap();
        DynamicFusionMapSpace::from_typed(&subset)
    }

    fn selected_contract_axes(
        lhs_dimensions: [usize; 4],
        rhs_dimensions: [usize; 4],
    ) -> (Vec<usize>, Vec<usize>) {
        let rule = U1FusionRule;
        let lhs = single_sector_space(&rule, lhs_dimensions);
        let rhs = single_sector_space(&rule, rhs_dimensions);
        let dst = single_sector_space(
            &rule,
            [
                lhs_dimensions[0],
                lhs_dimensions[1],
                rhs_dimensions[2],
                rhs_dimensions[3],
            ],
        );
        let plan = prepare_tensorcontract_fusion_plan_dyn_raw(
            &rule,
            &dst,
            &lhs,
            &rhs,
            TensorContractSpec::with_default_output_order(&[2, 3], &[1, 0]),
        )
        .unwrap();
        assert_eq!(
            plan.lhs_transform().kind(),
            TreeTransformOperationKind::Permute,
            "lhs lowering must be a permutation"
        );
        assert_eq!(
            plan.rhs_transform().kind(),
            TreeTransformOperationKind::Permute,
            "rhs lowering must be a permutation"
        );
        (
            plan.lhs_transform().domain_permutation().to_vec(),
            plan.rhs_transform().codomain_permutation().to_vec(),
        )
    }

    #[test]
    fn operand_plan_uses_the_route_specific_layout_primer() {
        let rule = U1FusionRule;
        let lhs_typed = single_sector_typed_space(&rule, [2, 3, 4, 5]);
        let rhs_typed = single_sector_typed_space(&rule, [5, 4, 6, 7]);
        let lhs = subset_copy(&rule, &lhs_typed);
        let rhs = subset_copy(&rule, &rhs_typed);
        let dst = single_sector_space(&rule, [2, 3, 6, 7]);
        let lhs_layout = FusionOperand::direct(&lhs)
            .prepare(&rule, encoded_layout_primer::<U1FusionRule>)
            .unwrap();
        let rhs_layout = FusionOperand::direct(&rhs)
            .prepare(&rule, encoded_layout_primer::<U1FusionRule>)
            .unwrap();
        let axes = || TensorContractSpec::with_default_output_order(&[2, 3], &[1, 0]);

        OPERAND_PRIMER_CALLS.set(0);
        prepare_tensorcontract_fusion_plan_dyn_prelowered_canonical(
            &rule,
            &dst,
            &lhs_layout,
            &rhs_layout,
            axes(),
            rejecting_operand_primer,
        )
        .unwrap();
        assert_eq!(OPERAND_PRIMER_CALLS.get(), 0);

        prepare_tensorcontract_fusion_plan_dyn_prelowered_with_primer_canonical(
            &rule,
            &dst,
            &lhs_layout,
            &rhs_layout,
            axes(),
            counting_lowered_operand_primer,
        )
        .unwrap();
        assert!(OPERAND_PRIMER_CALLS.get() > 0);
    }

    #[test]
    fn candidates_keep_lhs_rhs_pairs_intact() {
        let candidates = contracted_axis_order_candidates(&[3, 1, 2], &[6, 8, 4]);
        assert_eq!(candidates[0].lhs(), &[1, 2, 3]);
        assert_eq!(candidates[0].rhs(), &[8, 4, 6]);
        assert_eq!(candidates[1].lhs(), &[2, 3, 1]);
        assert_eq!(candidates[1].rhs(), &[4, 6, 8]);
        assert_eq!(candidates.len(), 2);
    }

    #[test]
    fn already_canonical_axes_do_not_duplicate_candidates() {
        let candidates = contracted_axis_order_candidates(&[0, 2], &[1, 3]);
        assert_eq!(candidates.len(), 1);
        assert_eq!(candidates[0].lhs(), &[0, 2]);
        assert_eq!(candidates[0].rhs(), &[1, 3]);
    }

    #[test]
    fn duplicate_axis_values_use_stable_pair_order() {
        let candidates = contracted_axis_order_candidates(&[2, 1, 2], &[7, 3, 5]);
        assert_eq!(candidates[0].lhs(), &[1, 2, 2]);
        assert_eq!(candidates[0].rhs(), &[3, 7, 5]);
    }

    #[test]
    fn lhs_sorted_candidate_wins_when_rhs_is_the_smaller_replay() {
        // What: crossed pairs lower to the LHS-sorted plan when only the small RHS is materialized.
        assert_eq!(
            selected_contract_axes([16, 16, 2, 3], [3, 2, 1, 1]),
            (vec![2, 3], vec![1, 0])
        );
    }

    #[test]
    fn rhs_sorted_candidate_wins_when_lhs_is_the_smaller_replay() {
        // What: the same crossed pairs lower to the RHS-sorted plan after operand sizes reverse.
        assert_eq!(
            selected_contract_axes([1, 1, 2, 3], [3, 2, 16, 16]),
            (vec![3, 2], vec![0, 1])
        );
    }

    #[test]
    fn canonical_selector_chooses_reverse_when_it_avoids_both_forward_transforms() {
        let rule = U1FusionRule;
        let lhs = single_sector_space(&rule, [2, 3, 5, 7]);
        let rhs = single_sector_space(&rule, [11, 13, 3, 2]);
        let dst = single_sector_space(&rule, [11, 13, 5, 7]);
        let axes =
            TensorContractSpec::new(&[1, 0], &[2, 3], OutputAxisOrder::from_axes(&[2, 3, 0, 1]));
        let facts =
            prepare_tensorcontract_fusion_candidate_facts_dyn_raw(&rule, &dst, &lhs, &rhs, axes)
                .unwrap();
        let selected = super::prepare_tensorcontract_fusion_plan_dyn_raw_canonical(
            &rule, &dst, &lhs, &rhs, axes,
        )
        .unwrap();

        assert_eq!(facts.len(), 4);
        assert!(facts[..2]
            .iter()
            .all(|candidate| candidate.total_materialized_elements() > 0));
        assert!(
            facts[2..]
                .iter()
                .map(|candidate| candidate.total_materialized_elements())
                .min()
                .unwrap()
                < facts[..2]
                    .iter()
                    .map(|candidate| candidate.total_materialized_elements())
                    .min()
                    .unwrap()
        );
        assert_eq!(selected.orientation(), FusionContractOrientation::RhsLhs);
        assert_eq!(
            selected.lhs_transform(),
            &TreeTransformOperation::permute([1, 0], [2, 3])
        );
        assert_eq!(
            selected.rhs_transform(),
            &TreeTransformOperation::permute([0, 1], [2, 3])
        );
        assert!(selected.output_transform_is_identity());
    }

    #[test]
    fn equal_cost_keeps_lhs_first_across_repeats_and_thread_counts() {
        // What: an equal allocation cost always selects the stable LHS-sorted candidate.
        for threads in [1, 2, 4] {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap();
            for _ in 0..4 {
                assert_eq!(
                    pool.install(|| selected_contract_axes([1, 1, 2, 3], [3, 2, 1, 1])),
                    (vec![2, 3], vec![1, 0])
                );
            }
        }
    }

    #[test]
    fn complete_selector_skips_loser_layouts_and_matches_exact_fallback() {
        let rule = U1FusionRule;
        let lhs_complete = single_sector_typed_space(&rule, [16, 16, 2, 3]);
        let rhs_complete = single_sector_typed_space(&rule, [3, 2, 1, 1]);
        let dst_complete = single_sector_typed_space(&rule, [16, 16, 1, 1]);
        let lhs = DynamicFusionMapSpace::from_typed(&lhs_complete);
        let rhs = DynamicFusionMapSpace::from_typed(&rhs_complete);
        let dst = DynamicFusionMapSpace::from_typed(&dst_complete);
        let axes = TensorContractSpec::with_default_output_order(&[2, 3], &[1, 0]);

        reset_candidate_score_calls();
        reset_candidate_build_calls();
        let complete =
            prepare_tensorcontract_fusion_plan_dyn_raw_canonical(&rule, &dst, &lhs, &rhs, axes)
                .unwrap();
        // What: Complete cold selection performs one semantic HomSpace preflight
        // and scores all stable candidates without constructing their layouts.
        assert_eq!(super::candidate_score_calls(), 4);
        assert_eq!(candidate_build_calls(), (1, 0));

        reset_candidate_score_calls();
        reset_candidate_build_calls();
        let exact = prepare_tensorcontract_fusion_plan_dyn_raw_canonical(
            &rule,
            &subset_copy(&rule, &dst_complete),
            &subset_copy(&rule, &lhs_complete),
            &subset_copy(&rule, &rhs_complete),
            axes,
        )
        .unwrap();
        // What: the conservative expert fallback probes both operands for each
        // candidate and selects the same mathematical winner.
        assert_eq!(super::candidate_score_calls(), 4);
        assert_eq!(candidate_build_calls(), (1, 8));
        assert_eq!(complete, exact);
    }

    #[test]
    fn complete_fz2_twist_cost_matches_exact_probe() {
        let rule = FermionParityFusionRule;
        let odd = SectorId::new(1);
        let space = |codomain_dual: bool, domain_dual: bool| {
            FusionTensorMapSpace::from_degeneracy_shapes(
                TensorMapSpace::<1, 1>::from_dims([1], [1]).unwrap(),
                FusionTreeHomSpace::new(
                    FusionProductSpace::new([SectorLeg::new([(odd, 1)], codomain_dual)]),
                    FusionProductSpace::new([SectorLeg::new([(odd, 1)], domain_dual)]),
                ),
                &rule,
                [vec![1, 1]],
            )
            .unwrap()
        };
        let lhs = DynamicFusionMapSpace::from_typed(&space(false, true));
        let rhs = DynamicFusionMapSpace::from_typed(&space(true, false));
        let dst = DynamicFusionMapSpace::from_typed(&space(false, false));
        let axis_order = contracted_axis_order_candidates(&[1], &[0]).remove(0);
        let axes =
            TensorContractSpec::with_default_output_order(axis_order.lhs(), axis_order.rhs());
        let plan = super::compile_tensorcontract_fusion_plan_from_spaces(
            &dst, &lhs, &rhs, axes, false, false,
        )
        .unwrap();

        let complete = super::score_complete_fusion_contract_candidate(
            &rule,
            &dst,
            &lhs,
            &rhs,
            axis_order.clone(),
            &plan,
        )
        .unwrap();
        let exact = super::score_fusion_contract_candidate(
            &rule,
            &dst,
            &lhs,
            &rhs,
            axis_order,
            plan,
            super::encoded_layout_probe,
            None,
        )
        .unwrap()
        .facts;

        // What: Complete structural scoring observes the same odd RHS twist
        // and materialization cost as the exact transformed-layout probe.
        assert!(complete.rhs_requires_twist());
        assert_eq!(complete, exact);
    }

    #[test]
    fn exact_fallback_skips_an_unavailable_loser() {
        let rule = U1FusionRule;
        let lhs = single_sector_typed_space(&rule, [1, 1, 2, 3]);
        let rhs = single_sector_typed_space(&rule, [3, 2, 16, 16]);
        let dst = single_sector_typed_space(&rule, [1, 1, 16, 16]);
        let lhs = subset_copy(&rule, &lhs);
        let rhs = subset_copy(&rule, &rhs);
        let dst = subset_copy(&rule, &dst);
        let axes = TensorContractSpec::with_default_output_order(&[2, 3], &[1, 0]);
        let expected =
            prepare_tensorcontract_fusion_plan_dyn_raw(&rule, &dst, &lhs, &rhs, axes).unwrap();

        REJECT_NEXT_PROBE.set(true);
        let actual = super::select_tensorcontract_fusion_plan_from_spaces_with_probe(
            &rule,
            &dst,
            &lhs,
            &rhs,
            axes,
            false,
            false,
            reject_next_probe,
            super::encoded_homspace_builder,
            None,
        )
        .unwrap();

        // What: a candidate-local capability failure cannot reject a later
        // stable candidate that selects the same valid winner.
        assert_eq!(actual, expected);
    }

    #[test]
    fn asymmetric_u1_candidates_expose_separate_source_components() {
        // What: each paired U1 candidate reports the exact source it must
        // materialize, rather than exposing only the winning total.
        let rule = U1FusionRule;
        let lhs = single_sector_space(&rule, [16, 16, 2, 3]);
        let rhs = single_sector_space(&rule, [3, 2, 1, 1]);
        let dst = single_sector_space(&rule, [16, 16, 1, 1]);
        let facts = prepare_tensorcontract_fusion_candidate_facts_dyn_raw(
            &rule,
            &dst,
            &lhs,
            &rhs,
            TensorContractSpec::with_default_output_order(&[2, 3], &[1, 0]),
        )
        .unwrap();

        assert_eq!(facts.len(), 4);
        assert_eq!(facts[0].axis_order().lhs(), &[2, 3]);
        assert_eq!(facts[0].axis_order().rhs(), &[1, 0]);
        assert_eq!(facts[0].orientation(), FusionContractOrientation::LhsRhs);
        assert!(!facts[0].lhs_conjugate());
        assert!(!facts[0].rhs_conjugate());
        assert!(facts[0].lhs_exact_identity_borrowable());
        assert!(!facts[0].rhs_exact_identity_borrowable());
        assert!(!facts[0].rhs_requires_twist());
        assert!(facts[0].output_exact_identity());
        assert_eq!(facts[0].lhs_materialized_elements(), 0);
        assert_eq!(facts[0].rhs_materialized_elements(), 6);
        assert_eq!(facts[0].output_materialized_elements(), 0);
        assert_eq!(facts[0].total_materialized_elements(), 6);

        assert_eq!(facts[1].axis_order().lhs(), &[3, 2]);
        assert_eq!(facts[1].axis_order().rhs(), &[0, 1]);
        assert!(!facts[1].lhs_exact_identity_borrowable());
        assert!(facts[1].rhs_exact_identity_borrowable());
        assert_eq!(facts[1].lhs_materialized_elements(), 1_536);
        assert_eq!(facts[1].rhs_materialized_elements(), 0);
        assert_eq!(facts[1].output_materialized_elements(), 0);
        assert_eq!(facts[1].total_materialized_elements(), 1_536);
        assert_eq!(facts[2].orientation(), FusionContractOrientation::RhsLhs);
        assert_eq!(facts[3].orientation(), FusionContractOrientation::RhsLhs);
    }

    #[test]
    fn nonidentity_output_is_one_checked_score_component() {
        // What: a requested nonidentity output order contributes exactly one
        // complete destination buffer to every current candidate.
        let rule = U1FusionRule;
        let lhs = single_sector_space(&rule, [11, 13, 2, 3]);
        let rhs = single_sector_space(&rule, [3, 2, 5, 7]);
        let dst = single_sector_space(&rule, [13, 11, 7, 5]);
        let facts = prepare_tensorcontract_fusion_candidate_facts_dyn_raw(
            &rule,
            &dst,
            &lhs,
            &rhs,
            TensorContractSpec::new(&[2, 3], &[1, 0], OutputAxisOrder::from_axes(&[1, 0, 3, 2])),
        )
        .unwrap();
        let dst_elements = dst.required_len().unwrap();

        assert_eq!(facts.len(), 4);
        for candidate in facts {
            assert!(!candidate.output_exact_identity());
            assert_eq!(candidate.output_materialized_elements(), dst_elements);
            assert_eq!(
                candidate.total_materialized_elements(),
                candidate.lhs_materialized_elements()
                    + candidate.rhs_materialized_elements()
                    + dst_elements
            );
        }
    }

    #[test]
    fn source_conjugation_is_an_explicit_candidate_fact() {
        // What: adjoint lowering records storage conjugation independently
        // from axis order and charges the affected source exactly once.
        let rule = U1FusionRule;
        let sector = tenet_core::U1Irrep::new(0).sector_id();
        let vector_space = || {
            FusionTensorMapSpace::from_degeneracy_shapes(
                TensorMapSpace::<1, 0>::from_dims([1], []).unwrap(),
                FusionTreeHomSpace::new(
                    FusionProductSpace::new([SectorLeg::new([(sector, 1)], false)]),
                    FusionProductSpace::new(Vec::<SectorLeg>::new()),
                ),
                &rule,
                [vec![1]],
            )
            .unwrap()
        };
        let lhs_typed = vector_space();
        let rhs_typed = vector_space();
        let dst_typed = FusionTensorMapSpace::from_degeneracy_shapes(
            TensorMapSpace::<1, 1>::from_dims([1], [1]).unwrap(),
            FusionTreeHomSpace::new(
                FusionProductSpace::new([SectorLeg::new([(sector, 1)], true)]),
                FusionProductSpace::new([SectorLeg::new([(sector, 1)], true)]),
            ),
            &rule,
            [vec![1, 1]],
        )
        .unwrap();
        let lhs = DynamicFusionMapSpace::from_typed(&lhs_typed);
        let rhs = DynamicFusionMapSpace::from_typed(&rhs_typed);
        let dst = DynamicFusionMapSpace::from_typed(&dst_typed);
        let facts = prepare_tensorcontract_fusion_candidate_facts_dyn_raw(
            &rule,
            &dst,
            &lhs,
            &rhs,
            TensorContractSpec::new_with_conjugation(
                &[],
                &[],
                OutputAxisOrder::identity(),
                true,
                false,
            ),
        )
        .unwrap();

        assert_eq!(facts.len(), 2);
        assert!(facts[0].lhs_conjugate());
        assert!(!facts[0].rhs_conjugate());
        assert!(!facts[0].lhs_exact_identity_borrowable());
        assert_eq!(facts[0].lhs_materialized_elements(), 1);
        assert_eq!(
            facts[0].total_materialized_elements(),
            facts[0].lhs_materialized_elements()
                + facts[0].rhs_materialized_elements()
                + facts[0].output_materialized_elements()
        );
        assert_eq!(facts[1].orientation(), FusionContractOrientation::RhsLhs);
        assert!(!facts[1].lhs_exact_identity_borrowable());

        let candidate = contracted_axis_order_candidates(&[], &[]).remove(0);
        let reverse = prepare_tensorcontract_fusion_plan_dyn_raw_with_axis_order_and_orientation(
            &rule,
            &dst,
            &lhs,
            &rhs,
            TensorContractSpec::new_with_conjugation(
                &[],
                &[],
                OutputAxisOrder::identity(),
                true,
                false,
            ),
            &candidate,
            FusionContractOrientation::RhsLhs,
        )
        .unwrap();
        assert_eq!(reverse.orientation(), FusionContractOrientation::RhsLhs);
        assert!(reverse.lhs_source_conjugate());
        assert!(!reverse.rhs_source_conjugate());
    }

    #[test]
    fn selector_cost_materializes_identity_operand_with_missing_structural_zero() {
        // What: an identity-axis operand with an incomplete SU2 tree grid is
        // charged for the complete core layout instead of treated as borrowed.
        let rule = SU2FusionRule;
        let lhs_typed = FusionTensorMapSpace::from_degeneracy_shapes(
            TensorMapSpace::<0, 0>::from_dims([], []).unwrap(),
            FusionTreeHomSpace::from_sector_ids([], []),
            &rule,
            [vec![]],
        )
        .unwrap();
        let rhs_hom = FusionTreeHomSpace::from_sector_ids([], [(1, 1), (1, 1), (1, 1), (1, 1)]);
        let rhs_keys = rhs_hom.fusion_tree_keys(&rule);
        assert_eq!(rhs_keys.len(), 2);
        let rhs_typed = FusionTensorMapSpace::new_unbound(
            TensorMapSpace::<0, 4>::from_dims([], [1, 1, 1, 1]).unwrap(),
            rhs_hom.clone(),
            crate::tests::packed_fixture_structure(
                4,
                [(BlockKey::from(rhs_keys[0].clone()), vec![1, 1, 1, 1])],
            )
            .unwrap(),
        )
        .unwrap()
        .try_bind_rule(&rule)
        .unwrap();
        let dst_hom = rhs_hom.permute(&rule, &[0, 1, 2, 3], &[]).unwrap();
        let dst_keys = dst_hom.fusion_tree_keys(&rule);
        let dst_typed = FusionTensorMapSpace::new_unbound(
            TensorMapSpace::<4, 0>::from_dims([1, 1, 1, 1], []).unwrap(),
            dst_hom,
            crate::tests::packed_fixture_structure(
                4,
                dst_keys
                    .iter()
                    .cloned()
                    .map(|key| (BlockKey::from(key), vec![1, 1, 1, 1])),
            )
            .unwrap(),
        )
        .unwrap()
        .try_bind_rule(&rule)
        .unwrap();
        let lhs = DynamicFusionMapSpace::from_typed(&lhs_typed);
        let rhs = DynamicFusionMapSpace::from_typed(&rhs_typed);
        let dst = DynamicFusionMapSpace::from_typed(&dst_typed);
        let facts = prepare_tensorcontract_fusion_candidate_facts_dyn_raw(
            &rule,
            &dst,
            &lhs,
            &rhs,
            TensorContractSpec::new(&[], &[], OutputAxisOrder::from_axes(&[0, 1, 2, 3])),
        )
        .unwrap();

        assert_eq!(facts.len(), 2);
        assert!(!facts[0].rhs_exact_identity_borrowable());
        assert_eq!(facts[0].lhs_materialized_elements(), 0);
        assert_eq!(facts[0].rhs_materialized_elements(), 2);
        assert_eq!(facts[0].output_materialized_elements(), 2);
        assert_eq!(facts[0].total_materialized_elements(), 4);
    }
}
