use tenet_core::{FusionTensorMapSpace, FusionTreeHomSpace, MultiplicityFreeRigidSymbols};

use crate::lowering::lower_tensorcontract_adjoint_axes;
use crate::{OperationError, TreeTransformOperation};
use tenet_operations::{TensorContractSpec, TensorContractSpecOwned};

use super::super::dynamic_space::{BoundDynamicFusionMapSpace, DynamicFusionMapSpace};
use super::super::structure::TensorContractAxisPlan;

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

/// Build the canonical and side-sorted candidates for a contraction.
///
/// Sorting is stable and always applies the same permutation to both sides.
/// The canonical candidate is first and is therefore the authority until a
/// layout-aware cost model is introduced. No fermionic sign is computed here:
/// that belongs to the tree-transform execution of the selected candidate.
pub(crate) fn contracted_axis_order_candidates(
    lhs: &[usize],
    rhs: &[usize],
) -> Vec<ContractAxisOrderCandidate> {
    assert_eq!(
        lhs.len(),
        rhs.len(),
        "paired contraction axes must have equal length"
    );
    let canonical = ContractAxisOrderCandidate {
        lhs: lhs.to_vec(),
        rhs: rhs.to_vec(),
    };
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
    let mut candidates = vec![canonical];
    if !candidates.contains(&lhs_sorted) {
        candidates.push(lhs_sorted);
    }
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
    prepare_tensorcontract_fusion_plan_dyn_raw(rule, dst, lhs, rhs, candidate_axes)
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
    use super::contracted_axis_order_candidates;

    #[test]
    fn candidates_keep_lhs_rhs_pairs_intact() {
        let candidates = contracted_axis_order_candidates(&[3, 1, 2], &[6, 8, 4]);
        assert_eq!(candidates[0].lhs(), &[3, 1, 2]);
        assert_eq!(candidates[0].rhs(), &[6, 8, 4]);
        assert_eq!(candidates[1].lhs(), &[1, 2, 3]);
        assert_eq!(candidates[1].rhs(), &[8, 4, 6]);
        assert_eq!(candidates[2].lhs(), &[2, 3, 1]);
        assert_eq!(candidates[2].rhs(), &[4, 6, 8]);
        assert_eq!(candidates.len(), 3);
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
        assert_eq!(candidates[1].lhs(), &[1, 2, 2]);
        assert_eq!(candidates[1].rhs(), &[3, 7, 5]);
    }
}
