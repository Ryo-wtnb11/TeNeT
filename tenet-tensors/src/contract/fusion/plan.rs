use tenet_core::{
    CheckedFusionAlgebra, CheckedFusionSpaceError, FusionRule, FusionTensorMapSpace,
    FusionTreeHomSpace, LoweredMultiplicityFreeAlgebra, MultiplicityFreeRigidSymbols,
};

use crate::lowering::lower_tensorcontract_adjoint_axes;
use crate::{OperationError, TreeTransformOperation};
use tenet_operations::{TensorContractSpec, TensorContractSpecOwned};

use super::super::dynamic_space::{
    BoundDynamicFusionMapSpace, DynamicFusionMapSpace, LayoutKeyBuilder, TransformedLayoutProbe,
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
}

#[cfg(test)]
pub(crate) fn reset_candidate_score_calls() {
    CANDIDATE_SCORE_CALLS.set(0);
}

#[cfg(test)]
pub(crate) fn candidate_score_calls() -> usize {
    CANDIDATE_SCORE_CALLS.get()
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
/// Slice 1 has only the existing forward route. Keeping orientation in the
/// facts prevents later reversed candidates from overloading paired-axis
/// ordering with a second meaning.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum FusionContractOrientation {
    LhsRhs,
}

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
    select_tensorcontract_fusion_plan_from_spaces(
        rule,
        dst,
        lhs,
        rhs,
        lowered_axes.as_spec(),
        lowered_axes.lhs_storage_conjugate(),
        lowered_axes.rhs_storage_conjugate(),
    )
}

pub(crate) fn prepare_tensorcontract_fusion_plan_dyn_prelowered<R>(
    rule: &R,
    dst: &DynamicFusionMapSpace,
    lhs: &DynamicFusionMapSpace,
    rhs: &DynamicFusionMapSpace,
    axes: TensorContractSpec<'_>,
    lhs_storage_conjugate: bool,
    rhs_storage_conjugate: bool,
) -> Result<FusionContractPlan, OperationError>
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64>,
{
    dst.validate_rule(rule)?;
    lhs.validate_rule(rule)?;
    rhs.validate_rule(rule)?;
    if axes.lhs_conjugate() != lhs_storage_conjugate
        || axes.rhs_conjugate() != rhs_storage_conjugate
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
    select_tensorcontract_fusion_plan_from_spaces(
        rule,
        dst,
        lhs,
        rhs,
        logical_axes,
        lhs_storage_conjugate,
        rhs_storage_conjugate,
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
    )
}

fn select_tensorcontract_fusion_plan_from_spaces<R>(
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
    select_tensorcontract_fusion_plan_from_spaces_with_probe(
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
) -> Result<Vec<FusionContractCandidateFacts>, OperationError>
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64>,
{
    validate_tensorcontract_fusion_plan_inputs(rule, dst, lhs, rhs, axes, homspace_builder)?;
    contracted_axis_order_candidates(axes.lhs_contracting_axes(), axes.rhs_contracting_axes())
        .into_iter()
        .map(|candidate| {
            let candidate_axes = TensorContractSpec::new(
                candidate.lhs(),
                candidate.rhs(),
                axes.output_permutation(),
            );
            let plan = prepare_tensorcontract_fusion_plan_from_spaces(
                rule,
                dst,
                lhs,
                rhs,
                candidate_axes,
                lhs_source_conjugate,
                rhs_source_conjugate,
            )?;
            score_fusion_contract_candidate(rule, dst, lhs, rhs, candidate, plan, probe, primer)
                .map(|scored| scored.facts)
        })
        .collect()
}

type LayoutProbeBuilder<R> = for<'a> fn(
    &'a R,
    &'a DynamicFusionMapSpace,
    &'a TreeTransformOperation,
    Option<LayoutKeyBuilder<R>>,
) -> Result<TransformedLayoutProbe, OperationError>;

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
    validate_tensorcontract_fusion_plan_inputs(rule, dst, lhs, rhs, axes, homspace_builder)?;
    let candidates =
        contracted_axis_order_candidates(axes.lhs_contracting_axes(), axes.rhs_contracting_axes());
    let mut best = None;
    for candidate in candidates {
        let candidate_axes =
            TensorContractSpec::new(candidate.lhs(), candidate.rhs(), axes.output_permutation());
        let plan = prepare_tensorcontract_fusion_plan_from_spaces(
            rule,
            dst,
            lhs,
            rhs,
            candidate_axes,
            lhs_source_conjugate,
            rhs_source_conjugate,
        )?;
        let scored =
            score_fusion_contract_candidate(rule, dst, lhs, rhs, candidate, plan, probe, primer)?;
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
    Ok(best
        .expect("paired contraction always has at least the LHS-sorted candidate")
        .plan)
}

pub(crate) fn prepare_tensorcontract_fusion_plan_dyn_prelowered_with_primer<R>(
    rule: &R,
    dst: &DynamicFusionMapSpace,
    lhs: &DynamicFusionMapSpace,
    rhs: &DynamicFusionMapSpace,
    axes: TensorContractSpec<'_>,
    lhs_storage_conjugate: bool,
    rhs_storage_conjugate: bool,
    primer: LayoutKeyBuilder<R>,
) -> Result<FusionContractPlan, OperationError>
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64>
        + LoweredMultiplicityFreeAlgebra
        + CheckedFusionAlgebra,
{
    dst.validate_rule(rule)?;
    lhs.validate_rule(rule)?;
    rhs.validate_rule(rule)?;
    if axes.lhs_conjugate() != lhs_storage_conjugate
        || axes.rhs_conjugate() != rhs_storage_conjugate
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
    select_tensorcontract_fusion_plan_from_spaces_with_probe(
        rule,
        dst,
        lhs,
        rhs,
        logical_axes,
        lhs_storage_conjugate,
        rhs_storage_conjugate,
        lowered_layout_probe::<R>,
        lowered_homspace_builder::<R>,
        Some(primer),
    )
}

fn score_fusion_contract_candidate<R>(
    rule: &R,
    dst: &DynamicFusionMapSpace,
    lhs: &DynamicFusionMapSpace,
    rhs: &DynamicFusionMapSpace,
    axis_order: ContractAxisOrderCandidate,
    plan: FusionContractPlan,
    probe: LayoutProbeBuilder<R>,
    primer: Option<LayoutKeyBuilder<R>>,
) -> Result<ScoredFusionContractCandidate, OperationError>
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64>,
{
    #[cfg(test)]
    CANDIDATE_SCORE_CALLS.set(CANDIDATE_SCORE_CALLS.get() + 1);
    let lhs_core = probe(rule, lhs, plan.lhs_transform(), primer)?;
    let rhs_core = probe(rule, rhs, plan.rhs_transform(), primer)?;
    let lhs_borrowed = super::super::dynamic::source_layout_metadata_is_borrowable(
        lhs,
        lhs_core.nout,
        lhs_core.homspace.rank(),
        || lhs_core.homspace == *lhs.homspace(),
        plan.lhs_transform(),
        plan.lhs_source_conjugate(),
    ) && lhs_core.source_structure_matches;
    let rhs_requires_twist = super::super::resolution::rhs_contract_homspace_requires_twist(
        rule,
        &rhs_core.homspace,
        plan.core_axes().as_spec(),
    )?;
    let rhs_exact_identity_borrowable = super::super::dynamic::source_layout_metadata_is_borrowable(
        rhs,
        rhs_core.nout,
        rhs_core.homspace.rank(),
        || rhs_core.homspace == *rhs.homspace(),
        plan.rhs_transform(),
        plan.rhs_source_conjugate(),
    ) && rhs_core.source_structure_matches;
    let lhs_materialized_elements = if lhs_borrowed {
        0
    } else {
        lhs_core.required_len
    };
    let rhs_materialized_elements = if rhs_exact_identity_borrowable && !rhs_requires_twist {
        0
    } else {
        rhs_core.required_len
    };
    let lhs_rhs_materialized_elements = lhs_materialized_elements
        .checked_add(rhs_materialized_elements)
        .ok_or_else(|| {
            OperationError::from_core_preserving_context(
                tenet_core::CoreError::ElementCountOverflow,
            )
        })?;
    let output_exact_identity = plan.output_transform_is_identity();
    let output_materialized_elements = if output_exact_identity {
        0
    } else {
        dst.required_len()
            .map_err(OperationError::from_core_preserving_context)?
    };
    let total_materialized_elements = lhs_rhs_materialized_elements
        .checked_add(output_materialized_elements)
        .ok_or_else(|| {
            OperationError::from_core_preserving_context(
                tenet_core::CoreError::ElementCountOverflow,
            )
        })?;
    let facts = FusionContractCandidateFacts {
        axis_order,
        orientation: FusionContractOrientation::LhsRhs,
        lhs_conjugate: plan.lhs_source_conjugate(),
        rhs_conjugate: plan.rhs_source_conjugate(),
        lhs_exact_identity_borrowable: lhs_borrowed,
        rhs_exact_identity_borrowable,
        rhs_requires_twist,
        output_exact_identity,
        lhs_materialized_elements,
        rhs_materialized_elements,
        output_materialized_elements,
        total_materialized_elements,
    };
    Ok(ScoredFusionContractCandidate { plan, facts })
}

fn validate_tensorcontract_fusion_plan_inputs<R>(
    rule: &R,
    dst: &DynamicFusionMapSpace,
    lhs: &DynamicFusionMapSpace,
    rhs: &DynamicFusionMapSpace,
    axes: TensorContractSpec<'_>,
    homspace_builder: HomSpaceBuilder<R>,
) -> Result<(), OperationError>
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64>,
{
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

#[cfg(test)]
mod tests {
    use super::{
        contracted_axis_order_candidates, prepare_tensorcontract_fusion_candidate_facts_dyn_raw,
        prepare_tensorcontract_fusion_plan_dyn_raw, FusionContractOrientation,
    };
    use crate::contract::DynamicFusionMapSpace;
    use crate::TreeTransformOperation;
    use tenet_core::{
        BlockKey, FusionProductSpace, FusionTensorMapSpace, FusionTreeHomSpace, SU2FusionRule,
        SectorLeg, TensorMapSpace, U1FusionRule,
    };
    use tenet_operations::{OutputAxisOrder, TensorContractSpec};

    fn single_sector_space(rule: &U1FusionRule, dimensions: [usize; 4]) -> DynamicFusionMapSpace {
        let homspace = FusionTreeHomSpace::from_sector_ids(
            [(0, dimensions[0]), (0, dimensions[1])],
            [(0, dimensions[2]), (0, dimensions[3])],
        );
        DynamicFusionMapSpace::from_degeneracy_shapes(rule, homspace, [dimensions]).unwrap()
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
        let TreeTransformOperation::Permute {
            domain_permutation: lhs_contract,
            ..
        } = plan.lhs_transform()
        else {
            panic!("lhs lowering must be a permutation");
        };
        let TreeTransformOperation::Permute {
            codomain_permutation: rhs_contract,
            ..
        } = plan.rhs_transform()
        else {
            panic!("rhs lowering must be a permutation");
        };
        (lhs_contract.to_vec(), rhs_contract.to_vec())
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

        assert_eq!(facts.len(), 2);
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

        assert_eq!(facts.len(), 2);
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

        assert_eq!(facts.len(), 1);
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

        assert_eq!(facts.len(), 1);
        assert!(!facts[0].rhs_exact_identity_borrowable());
        assert_eq!(facts[0].lhs_materialized_elements(), 0);
        assert_eq!(facts[0].rhs_materialized_elements(), 2);
        assert_eq!(facts[0].output_materialized_elements(), 2);
        assert_eq!(facts[0].total_materialized_elements(), 4);
    }
}
