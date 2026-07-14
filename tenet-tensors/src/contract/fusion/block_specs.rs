use std::sync::Arc;

use tenet_core::{
    multiplicity_free_permute_tree_pair, BlockKey, BraidingStyleKind, CoreError, FusionRule,
    FusionTensorMapSpace, FusionTreeBlockKey, FusionTreeHomSpace, FusionTreeKey,
    MultiplicityFreeRigidSymbols, SectorId, TensorMap, TensorStorage,
};

use crate::lowering::lower_tensorcontract_adjoint_axes;
use crate::OperationError;
use tenet_operations::TensorContractSpec;

use super::super::dynamic_space::DynamicFusionMapSpace;
use super::super::fusion_block::validate_fusion_contract_rule;
use super::super::structure::{
    TensorContractAxisPlan, TensorContractBlockSpec, TensorContractStructure,
};

/// Every sector on every fusion tree of `space` equals its own dual. Used to
/// gate the Structure route's conjugate (categorical-adjoint) block matching,
/// which is only correct for self-dual symmetries.
fn all_sectors_self_dual<R>(rule: &R, space: &DynamicFusionMapSpace) -> bool
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64>,
{
    let tree_self_dual = |tree: &FusionTreeKey| {
        tree.uncoupled().iter().all(|&s| rule.dual(s) == s)
            && tree.coupled().is_none_or(|c| rule.dual(c) == c)
    };
    space
        .homspace()
        .fusion_tree_keys(rule)
        .iter()
        .all(|key| tree_self_dual(key.codomain_tree()) && tree_self_dual(key.domain_tree()))
}

pub fn tensorcontract_fusion_structure<
    R,
    TDst,
    TLhs,
    TRhs,
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
    dst: &TensorMap<TDst, DST_NOUT, DST_NIN, SDst, DDst>,
    lhs: &TensorMap<TLhs, LHS_NOUT, LHS_NIN, SLhs, DLhs>,
    rhs: &TensorMap<TRhs, RHS_NOUT, RHS_NIN, SRhs, DRhs>,
    axes: TensorContractSpec<'_>,
) -> Result<TensorContractStructure, OperationError>
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64>,
    DDst: TensorStorage<TDst>,
    DLhs: TensorStorage<TLhs>,
    DRhs: TensorStorage<TRhs>,
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
    tensorcontract_fusion_structure_dyn(
        rule,
        &DynamicFusionMapSpace::from_typed(dst_fusion),
        &DynamicFusionMapSpace::from_typed(lhs_fusion),
        &DynamicFusionMapSpace::from_typed(rhs_fusion),
        Arc::clone(lhs.structure()),
        Arc::clone(rhs.structure()),
        axes,
    )
}

/// Dynamic-rank variant of [`tensorcontract_fusion_structure`]. The storage
/// structures are the layouts the source data slices are replayed with (for
/// unconjugated operands these are the spaces' own subblock structures).
pub fn tensorcontract_fusion_structure_dyn<R>(
    rule: &R,
    dst: &DynamicFusionMapSpace,
    lhs: &DynamicFusionMapSpace,
    rhs: &DynamicFusionMapSpace,
    lhs_storage_structure: Arc<tenet_core::BlockStructure>,
    rhs_storage_structure: Arc<tenet_core::BlockStructure>,
    axes: TensorContractSpec<'_>,
) -> Result<TensorContractStructure, OperationError>
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64>,
{
    dst.validate_rule(rule)?;
    lhs.validate_rule(rule)?;
    rhs.validate_rule(rule)?;
    // The categorical-adjoint (conjugate) block matching in this Structure route
    // is only correct when the conjugated operand's sectors are all self-dual
    // (a sector equal to its dual). For non-self-dual sectors — e.g. a U(1)
    // charge q whose dual is -q ≠ q — it mislabels the coupled sector of the
    // output block (pairing q with -q across codomain/domain), producing an
    // invalid `MissingBlockKey`. This was only ever exercised on self-dual
    // symmetries (Z2, fermion parity, SU(2)). Decline to the DynamicTree route,
    // which handles the adjoint via `adjoint_view` + a data-only storage
    // conjugation correctly for any symmetry (still copy-free). Verified against
    // the eager `adjoint_dyn` oracle for U(1).
    if (axes.lhs_conjugate() && !all_sectors_self_dual(rule, lhs))
        || (axes.rhs_conjugate() && !all_sectors_self_dual(rule, rhs))
    {
        return Err(OperationError::UnsupportedTensorContractScope {
            message: SOURCE_TRANSFORM_REQUIRES_EXPLICIT,
        });
    }
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
    tensorcontract_fusion_structure_from_spaces(
        rule,
        dst,
        lhs,
        rhs,
        lhs_storage_structure,
        rhs_storage_structure,
        lowered_axes.as_spec(),
    )
}

fn tensorcontract_fusion_structure_from_spaces<R>(
    rule: &R,
    dst: &DynamicFusionMapSpace,
    lhs: &DynamicFusionMapSpace,
    rhs: &DynamicFusionMapSpace,
    lhs_storage_structure: std::sync::Arc<tenet_core::BlockStructure>,
    rhs_storage_structure: std::sync::Arc<tenet_core::BlockStructure>,
    axes: TensorContractSpec<'_>,
) -> Result<TensorContractStructure, OperationError>
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64>,
{
    let block_specs = tensorcontract_fusion_block_specs_lowered(rule, dst, lhs, rhs, axes)?;
    TensorContractStructure::compile_shared_structures_with_block_specs_and_storage(
        std::sync::Arc::clone(dst.structure()),
        std::sync::Arc::clone(lhs.structure()),
        std::sync::Arc::clone(rhs.structure()),
        lhs_storage_structure,
        rhs_storage_structure,
        axes,
        &block_specs,
    )
}

pub fn tensorcontract_fusion_block_specs<
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
) -> Result<Vec<TensorContractBlockSpec>, OperationError>
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64>,
{
    let dst = DynamicFusionMapSpace::from_typed(dst);
    let lhs = DynamicFusionMapSpace::from_typed(lhs);
    let rhs = DynamicFusionMapSpace::from_typed(rhs);
    validate_fusion_contract_rule(rule, &dst, &lhs, &rhs)?;
    reject_fusion_contract_conjugation(axes)?;
    tensorcontract_fusion_block_specs_lowered(rule, &dst, &lhs, &rhs, axes)
}

fn tensorcontract_fusion_block_specs_lowered<R>(
    rule: &R,
    dst: &DynamicFusionMapSpace,
    lhs: &DynamicFusionMapSpace,
    rhs: &DynamicFusionMapSpace,
    axes: TensorContractSpec<'_>,
) -> Result<Vec<TensorContractBlockSpec>, OperationError>
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
    if !is_core_form_fusion_source_contract(
        lhs.homspace(),
        rhs.homspace(),
        axis_plan.lhs_contracting_axes.as_slice(),
        axis_plan.rhs_contracting_axes.as_slice(),
    ) {
        return Err(OperationError::UnsupportedTensorContractScope {
            message: SOURCE_TRANSFORM_REQUIRES_EXPLICIT,
        });
    }
    if is_core_form_fusion_compose_contract(
        lhs.homspace(),
        rhs.homspace(),
        axis_plan.lhs_contracting_axes.as_slice(),
        axis_plan.rhs_contracting_axes.as_slice(),
        axis_plan.output_axes.as_slice(),
        dst_nout,
    ) {
        return tensorcontract_core_fusion_block_specs(rule, dst, lhs, rhs, &axis_plan);
    }

    tensorcontract_transformed_fusion_block_specs(rule, dst, lhs, rhs, &axis_plan, dst_nout)
}

fn tensorcontract_core_fusion_block_specs<R>(
    rule: &R,
    dst: &DynamicFusionMapSpace,
    lhs: &DynamicFusionMapSpace,
    rhs: &DynamicFusionMapSpace,
    axis_plan: &TensorContractAxisPlan,
) -> Result<Vec<TensorContractBlockSpec>, OperationError>
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64>,
{
    let mut specs = Vec::new();
    for lhs_index in 0..lhs.structure().block_count() {
        let lhs_block = lhs.structure().block(lhs_index)?;
        let BlockKey::FusionTree(lhs_key) = lhs_block.key() else {
            continue;
        };
        let lhs_external = lhs_key.external_sectors(rule);
        for rhs_index in 0..rhs.structure().block_count() {
            let rhs_block = rhs.structure().block(rhs_index)?;
            let BlockKey::FusionTree(rhs_key) = rhs_block.key() else {
                continue;
            };
            let rhs_external = rhs_key.external_sectors(rule);
            if !contracted_external_sectors_match(
                &lhs_external,
                &rhs_external,
                axis_plan.lhs_contracting_axes.as_slice(),
                axis_plan.rhs_contracting_axes.as_slice(),
            ) {
                continue;
            }
            if !contracted_fusion_tree_basis_matches(
                rule,
                lhs_key.domain_tree(),
                rhs_key.codomain_tree(),
            ) {
                continue;
            }
            let dst_key = FusionTreeBlockKey::pair(
                lhs_key.codomain_tree().clone(),
                rhs_key.domain_tree().clone(),
            );
            let dst_external = dst_key.external_sectors(rule);
            let expected_external =
                contracted_output_external_sectors(&lhs_external, &rhs_external, axis_plan);
            if dst_external != expected_external {
                return Err(OperationError::StructureMismatch { tensor: "dst" });
            }
            // The sorted, exact-key block lookup below already reports a
            // missing dst key. A prior `fusion_tree_keys_from_external_sectors`
            // + `contains` membership test is redundant — dst's structure
            // blocks are exactly the homspace's valid fusion trees, so
            // `find_block_index_by_fusion_tree_key` is `Some` iff the key is a
            // valid output tree — and it costs an uncached fusion-tree
            // enumeration + allocation per block pair (issue #52).
            let dst_index = dst
                .structure()
                .find_block_index_by_fusion_tree_key(&dst_key)
                .ok_or_else(|| OperationError::MissingBlockKey {
                    key: BlockKey::from(dst_key.clone()),
                })?;
            let coefficient = rhs_contract_twist_factor(
                rule,
                rhs.homspace(),
                axis_plan.rhs_contracting_axes.as_slice(),
                rhs_key.codomain_tree(),
            )?;
            specs.push(TensorContractBlockSpec::with_coefficient(
                dst_index,
                lhs_index,
                rhs_index,
                coefficient,
            ));
        }
    }
    Ok(specs)
}

fn tensorcontract_transformed_fusion_block_specs<R>(
    rule: &R,
    dst: &DynamicFusionMapSpace,
    lhs: &DynamicFusionMapSpace,
    rhs: &DynamicFusionMapSpace,
    axis_plan: &TensorContractAxisPlan,
    dst_codomain_rank: usize,
) -> Result<Vec<TensorContractBlockSpec>, OperationError>
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64>,
{
    let output_codomain_axes = &axis_plan.output_axes[..dst_codomain_rank];
    let output_domain_axes = &axis_plan.output_axes[dst_codomain_rank..];
    let mut specs = Vec::new();
    for lhs_index in 0..lhs.structure().block_count() {
        let lhs_block = lhs.structure().block(lhs_index)?;
        let BlockKey::FusionTree(lhs_key) = lhs_block.key() else {
            continue;
        };
        let lhs_terms = multiplicity_free_permute_tree_pair(
            rule,
            lhs_key,
            axis_plan.lhs_open_axes.as_slice(),
            axis_plan.lhs_contracting_axes.as_slice(),
        )
        .map_err(OperationError::from_core_preserving_context)?;
        for rhs_index in 0..rhs.structure().block_count() {
            let rhs_block = rhs.structure().block(rhs_index)?;
            let BlockKey::FusionTree(rhs_key) = rhs_block.key() else {
                continue;
            };
            let rhs_terms = multiplicity_free_permute_tree_pair(
                rule,
                rhs_key,
                axis_plan.rhs_contracting_axes.as_slice(),
                axis_plan.rhs_open_axes.as_slice(),
            )
            .map_err(OperationError::from_core_preserving_context)?;

            for (lhs_core, lhs_coeff) in &lhs_terms {
                for (rhs_core, rhs_coeff) in &rhs_terms {
                    if !contracted_fusion_tree_basis_matches(
                        rule,
                        lhs_core.domain_tree(),
                        rhs_core.codomain_tree(),
                    ) {
                        continue;
                    }
                    let core_dst_key = FusionTreeBlockKey::pair(
                        lhs_core.codomain_tree().clone(),
                        rhs_core.domain_tree().clone(),
                    );
                    let rhs_twist = rhs_contract_twist_factor(
                        rule,
                        rhs.homspace(),
                        axis_plan.rhs_contracting_axes.as_slice(),
                        rhs_core.codomain_tree(),
                    )?;
                    let dst_terms = multiplicity_free_permute_tree_pair(
                        rule,
                        &core_dst_key,
                        output_codomain_axes,
                        output_domain_axes,
                    )
                    .map_err(OperationError::from_core_preserving_context)?;
                    for (dst_key, dst_coeff) in dst_terms {
                        let dst_index = dst
                            .structure()
                            .find_block_index_by_fusion_tree_key(&dst_key)
                            .ok_or_else(|| OperationError::MissingBlockKey {
                                key: BlockKey::from(dst_key.clone()),
                            })?;
                        specs.push(TensorContractBlockSpec::with_coefficient(
                            dst_index,
                            lhs_index,
                            rhs_index,
                            *lhs_coeff * *rhs_coeff * dst_coeff * rhs_twist,
                        ));
                    }
                }
            }
        }
    }
    Ok(specs)
}

pub(crate) const EXPLICIT_OUTPUT_TRANSFORM_REQUIRES_CORE_DST: &str =
    "explicit fusion contraction with output tree-pair transform requires caller-owned core_dst";
pub(crate) const FUSION_TENSORCONTRACT_CONJUGATION_REQUIRES_CATEGORICAL_ADJOINT: &str =
    "fusion tensorcontract with conjugation requires categorical adjoint lowering";
pub(crate) const SOURCE_TRANSFORM_REQUIRES_EXPLICIT: &str =
    "fusion contraction requiring source tree-pair transforms is not implemented; pre-transform operands explicitly";

pub(crate) fn reject_fusion_contract_conjugation(
    axes: TensorContractSpec<'_>,
) -> Result<(), OperationError> {
    if axes.lhs_conjugate() || axes.rhs_conjugate() {
        return Err(OperationError::UnsupportedTensorContractScope {
            message: FUSION_TENSORCONTRACT_CONJUGATION_REQUIRES_CATEGORICAL_ADJOINT,
        });
    }
    Ok(())
}

pub(crate) fn rhs_contract_twist_factor<R>(
    rule: &R,
    rhs: &FusionTreeHomSpace,
    rhs_contracting_axes: &[usize],
    rhs_core_codomain: &FusionTreeKey,
) -> Result<f64, OperationError>
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64>,
{
    if rule.braiding_style() != BraidingStyleKind::Fermionic {
        return Ok(rule.scalar_one());
    }
    if rhs_contracting_axes.len() != rhs_core_codomain.uncoupled().len() {
        return Err(OperationError::StructureRankMismatch {
            expected: rhs_contracting_axes.len(),
            actual: rhs_core_codomain.uncoupled().len(),
        });
    }
    let mut factor = rule.scalar_one();
    for (position, &axis) in rhs_contracting_axes.iter().enumerate() {
        if external_axis_is_dual(rhs, axis)? {
            factor *= rule.twist_scalar(rhs_core_codomain.uncoupled()[position]);
        }
    }
    Ok(factor)
}

pub(crate) fn external_axis_is_dual(
    homspace: &FusionTreeHomSpace,
    axis: usize,
) -> Result<bool, OperationError> {
    homspace
        .external_axis_is_dual(axis)
        .ok_or_else(|| OperationError::InvalidAxisSet {
            tensor: "rhs",
            axes: vec![axis],
            rank: homspace.rank(),
        })
}

fn contracted_external_sectors_match(
    lhs_external: &[SectorId],
    rhs_external: &[SectorId],
    lhs_axes: &[usize],
    rhs_axes: &[usize],
) -> bool {
    lhs_axes
        .iter()
        .zip(rhs_axes)
        .all(|(&lhs_axis, &rhs_axis)| lhs_external[lhs_axis] == rhs_external[rhs_axis])
}

pub(crate) fn contracted_fusion_tree_basis_matches<R>(
    rule: &R,
    lhs_domain: &FusionTreeKey,
    rhs_codomain: &FusionTreeKey,
) -> bool
where
    R: FusionRule,
{
    lhs_domain.uncoupled().len() == rhs_codomain.uncoupled().len()
        && lhs_domain.innerlines().len() == rhs_codomain.innerlines().len()
        && lhs_domain.vertices() == rhs_codomain.vertices()
        && lhs_domain.is_dual() == rhs_codomain.is_dual()
        && lhs_domain
            .uncoupled()
            .iter()
            .copied()
            .map(|sector| rule.dual(sector))
            .eq(rhs_codomain.uncoupled().iter().copied())
        && lhs_domain
            .innerlines()
            .iter()
            .copied()
            .map(|sector| rule.dual(sector))
            .eq(rhs_codomain.innerlines().iter().copied())
        && rule.dual(lhs_domain.coupled().unwrap_or_else(|| rule.vacuum()))
            == rhs_codomain.coupled().unwrap_or_else(|| rule.vacuum())
}

fn contracted_output_external_sectors(
    lhs_external: &[SectorId],
    rhs_external: &[SectorId],
    axis_plan: &TensorContractAxisPlan,
) -> Vec<SectorId> {
    let mut core = axis_plan
        .lhs_open_axes
        .iter()
        .map(|&axis| lhs_external[axis])
        .collect::<Vec<_>>();
    core.extend(
        axis_plan
            .rhs_open_axes
            .iter()
            .map(|&axis| rhs_external[axis]),
    );
    axis_plan
        .output_axes
        .iter()
        .map(|&axis| core[axis])
        .collect()
}

fn is_core_form_fusion_compose_contract(
    lhs: &FusionTreeHomSpace,
    rhs: &FusionTreeHomSpace,
    lhs_contracting_axes: &[usize],
    rhs_contracting_axes: &[usize],
    output_axes: &[usize],
    dst_codomain_rank: usize,
) -> bool {
    let core_output_rank = lhs.codomain().len() + rhs.domain().len();
    let core_output_axes = (0..core_output_rank).collect::<Vec<_>>();
    is_core_form_fusion_source_contract(lhs, rhs, lhs_contracting_axes, rhs_contracting_axes)
        && output_axes == core_output_axes.as_slice()
        && dst_codomain_rank == lhs.codomain().len()
}

fn is_core_form_fusion_source_contract(
    lhs: &FusionTreeHomSpace,
    rhs: &FusionTreeHomSpace,
    lhs_contracting_axes: &[usize],
    rhs_contracting_axes: &[usize],
) -> bool {
    let lhs_domain_axes =
        (lhs.codomain().len()..lhs.codomain().len() + lhs.domain().len()).collect::<Vec<_>>();
    let rhs_codomain_axes = (0..rhs.codomain().len()).collect::<Vec<_>>();
    lhs_contracting_axes == lhs_domain_axes.as_slice()
        && rhs_contracting_axes == rhs_codomain_axes.as_slice()
}
