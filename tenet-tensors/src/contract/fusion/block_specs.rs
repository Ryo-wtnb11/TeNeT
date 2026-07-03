use std::sync::Arc;

use tenet_core::{
    multiplicity_free_permute_tree_pair, BlockKey, BraidingStyleKind, CoreError, FusionRule,
    FusionTensorMapSpace, FusionTreeBlockKey, FusionTreeHomSpace, FusionTreeKey,
    MultiplicityFreeRigidSymbols, SectorId, TensorMap, TensorStorage,
};

use crate::axis::TensorContractAxisSpec;
use crate::lowering::{adjoint_fusion_space_view, lower_tensorcontract_adjoint_axes};
use crate::OperationError;

use super::super::structure::{
    TensorContractAxisPlan, TensorContractBlockSpec, TensorContractStructure,
};

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
    axes: TensorContractAxisSpec<'_>,
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
    let lowered_axes =
        lower_tensorcontract_adjoint_axes::<LHS_NOUT, LHS_NIN, RHS_NOUT, RHS_NIN>(axes)?;
    if axes.lhs_conjugate() && axes.rhs_conjugate() {
        let lhs_adjoint = adjoint_fusion_space_view(lhs_fusion)?;
        let rhs_adjoint = adjoint_fusion_space_view(rhs_fusion)?;
        tensorcontract_fusion_structure_from_spaces(
            rule,
            dst_fusion,
            &lhs_adjoint,
            &rhs_adjoint,
            Arc::clone(lhs.structure()),
            Arc::clone(rhs.structure()),
            lowered_axes.as_spec(),
        )
    } else if axes.lhs_conjugate() {
        let lhs_adjoint = adjoint_fusion_space_view(lhs_fusion)?;
        tensorcontract_fusion_structure_from_spaces(
            rule,
            dst_fusion,
            &lhs_adjoint,
            rhs_fusion,
            Arc::clone(lhs.structure()),
            Arc::clone(rhs.structure()),
            lowered_axes.as_spec(),
        )
    } else if axes.rhs_conjugate() {
        let rhs_adjoint = adjoint_fusion_space_view(rhs_fusion)?;
        tensorcontract_fusion_structure_from_spaces(
            rule,
            dst_fusion,
            lhs_fusion,
            &rhs_adjoint,
            Arc::clone(lhs.structure()),
            Arc::clone(rhs.structure()),
            lowered_axes.as_spec(),
        )
    } else {
        tensorcontract_fusion_structure_from_spaces(
            rule,
            dst_fusion,
            lhs_fusion,
            rhs_fusion,
            Arc::clone(lhs.structure()),
            Arc::clone(rhs.structure()),
            lowered_axes.as_spec(),
        )
    }
}

fn tensorcontract_fusion_structure_from_spaces<
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
    lhs_storage_structure: std::sync::Arc<tenet_core::BlockStructure>,
    rhs_storage_structure: std::sync::Arc<tenet_core::BlockStructure>,
    axes: TensorContractAxisSpec<'_>,
) -> Result<TensorContractStructure, OperationError>
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64>,
{
    let block_specs = tensorcontract_fusion_block_specs_lowered(rule, dst, lhs, rhs, axes)?;
    TensorContractStructure::compile_shared_structures_with_block_specs_and_storage(
        std::sync::Arc::clone(dst.subblock_structure()),
        std::sync::Arc::clone(lhs.subblock_structure()),
        std::sync::Arc::clone(rhs.subblock_structure()),
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
    axes: TensorContractAxisSpec<'_>,
) -> Result<Vec<TensorContractBlockSpec>, OperationError>
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64>,
{
    reject_fusion_contract_conjugation(axes)?;
    tensorcontract_fusion_block_specs_lowered(rule, dst, lhs, rhs, axes)
}

fn tensorcontract_fusion_block_specs_lowered<
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
) -> Result<Vec<TensorContractBlockSpec>, OperationError>
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
    if !is_canonical_fusion_source_contract(
        lhs.homspace(),
        rhs.homspace(),
        axis_plan.lhs_contracting_axes.as_slice(),
        axis_plan.rhs_contracting_axes.as_slice(),
    ) {
        return Err(OperationError::UnsupportedTensorContractScope {
            message: SOURCE_TRANSFORM_REQUIRES_EXPLICIT,
        });
    }
    if is_canonical_fusion_compose_contract(
        lhs.homspace(),
        rhs.homspace(),
        axis_plan.lhs_contracting_axes.as_slice(),
        axis_plan.rhs_contracting_axes.as_slice(),
        axis_plan.output_axes.as_slice(),
        DST_NOUT,
    ) {
        return tensorcontract_canonical_fusion_block_specs(rule, dst, lhs, rhs, &axis_plan);
    }

    tensorcontract_transformed_fusion_block_specs(rule, dst, lhs, rhs, &axis_plan, DST_NOUT)
}

fn tensorcontract_canonical_fusion_block_specs<
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
    axis_plan: &TensorContractAxisPlan,
) -> Result<Vec<TensorContractBlockSpec>, OperationError>
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64>,
{
    let mut specs = Vec::new();
    for lhs_index in 0..lhs.subblock_structure().block_count() {
        let lhs_block = lhs.subblock_structure().block(lhs_index)?;
        let BlockKey::FusionTree(lhs_key) = lhs_block.key() else {
            continue;
        };
        let lhs_external = lhs_key.external_sectors(rule);
        for rhs_index in 0..rhs.subblock_structure().block_count() {
            let rhs_block = rhs.subblock_structure().block(rhs_index)?;
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
            let dst_keys = dst
                .homspace()
                .fusion_tree_keys_from_external_sectors(rule, &dst_external)
                .map_err(OperationError::from_core_preserving_context)?;
            if !dst_keys.contains(&dst_key) {
                return Err(OperationError::MissingBlockKey {
                    key: BlockKey::from(dst_key),
                });
            }
            let dst_index = dst.find_subblock_index(&dst_key).ok_or_else(|| {
                OperationError::MissingBlockKey {
                    key: BlockKey::from(dst_key.clone()),
                }
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

fn tensorcontract_transformed_fusion_block_specs<
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
    axis_plan: &TensorContractAxisPlan,
    dst_codomain_rank: usize,
) -> Result<Vec<TensorContractBlockSpec>, OperationError>
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64>,
{
    let output_codomain_axes = &axis_plan.output_axes[..dst_codomain_rank];
    let output_domain_axes = &axis_plan.output_axes[dst_codomain_rank..];
    let mut specs = Vec::new();
    for lhs_index in 0..lhs.subblock_structure().block_count() {
        let lhs_block = lhs.subblock_structure().block(lhs_index)?;
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
        for rhs_index in 0..rhs.subblock_structure().block_count() {
            let rhs_block = rhs.subblock_structure().block(rhs_index)?;
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

            for (lhs_canonical, lhs_coeff) in &lhs_terms {
                for (rhs_canonical, rhs_coeff) in &rhs_terms {
                    if !contracted_fusion_tree_basis_matches(
                        rule,
                        lhs_canonical.domain_tree(),
                        rhs_canonical.codomain_tree(),
                    ) {
                        continue;
                    }
                    let canonical_dst_key = FusionTreeBlockKey::pair(
                        lhs_canonical.codomain_tree().clone(),
                        rhs_canonical.domain_tree().clone(),
                    );
                    let rhs_twist = rhs_contract_twist_factor(
                        rule,
                        rhs.homspace(),
                        axis_plan.rhs_contracting_axes.as_slice(),
                        rhs_canonical.codomain_tree(),
                    )?;
                    let dst_terms = multiplicity_free_permute_tree_pair(
                        rule,
                        &canonical_dst_key,
                        output_codomain_axes,
                        output_domain_axes,
                    )
                    .map_err(OperationError::from_core_preserving_context)?;
                    for (dst_key, dst_coeff) in dst_terms {
                        let dst_index = dst.find_subblock_index(&dst_key).ok_or_else(|| {
                            OperationError::MissingBlockKey {
                                key: BlockKey::from(dst_key.clone()),
                            }
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

pub(crate) const EXPLICIT_OUTPUT_TRANSFORM_REQUIRES_CANONICAL_DST: &str =
    "explicit fusion contraction with output tree-pair transform requires caller-owned canonical_dst";
pub(crate) const FUSION_TENSORCONTRACT_CONJUGATION_REQUIRES_CATEGORICAL_ADJOINT: &str =
    "fusion tensorcontract with conjugation requires categorical adjoint lowering";
pub(crate) const SOURCE_TRANSFORM_REQUIRES_EXPLICIT: &str =
    "fusion contraction requiring source tree-pair transforms is not implemented; pre-transform operands explicitly";

pub(crate) fn reject_fusion_contract_conjugation(
    axes: TensorContractAxisSpec<'_>,
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
    rhs_canonical_codomain: &FusionTreeKey,
) -> Result<f64, OperationError>
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64>,
{
    if rule.braiding_style() != BraidingStyleKind::Fermionic {
        return Ok(rule.scalar_one());
    }
    if rhs_contracting_axes.len() != rhs_canonical_codomain.uncoupled().len() {
        return Err(OperationError::StructureRankMismatch {
            expected: rhs_contracting_axes.len(),
            actual: rhs_canonical_codomain.uncoupled().len(),
        });
    }
    let mut factor = rule.scalar_one();
    for (position, &axis) in rhs_contracting_axes.iter().enumerate() {
        if external_axis_is_dual(rhs, axis)? {
            factor *= rule.twist_scalar(rhs_canonical_codomain.uncoupled()[position]);
        }
    }
    Ok(factor)
}

pub(crate) fn external_axis_is_dual(
    homspace: &FusionTreeHomSpace,
    axis: usize,
) -> Result<bool, OperationError> {
    if axis < homspace.codomain().len() {
        Ok(homspace.codomain().legs()[axis].is_dual())
    } else if axis < homspace.rank() {
        Ok(!homspace.domain().legs()[axis - homspace.codomain().len()].is_dual())
    } else {
        Err(OperationError::InvalidAxisSet {
            tensor: "rhs",
            axes: vec![axis],
            rank: homspace.rank(),
        })
    }
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
    let mut canonical = axis_plan
        .lhs_open_axes
        .iter()
        .map(|&axis| lhs_external[axis])
        .collect::<Vec<_>>();
    canonical.extend(
        axis_plan
            .rhs_open_axes
            .iter()
            .map(|&axis| rhs_external[axis]),
    );
    axis_plan
        .output_axes
        .iter()
        .map(|&axis| canonical[axis])
        .collect()
}

fn is_canonical_fusion_compose_contract(
    lhs: &FusionTreeHomSpace,
    rhs: &FusionTreeHomSpace,
    lhs_contracting_axes: &[usize],
    rhs_contracting_axes: &[usize],
    output_axes: &[usize],
    dst_codomain_rank: usize,
) -> bool {
    let canonical_output_rank = lhs.codomain().len() + rhs.domain().len();
    let canonical_output_axes = (0..canonical_output_rank).collect::<Vec<_>>();
    is_canonical_fusion_source_contract(lhs, rhs, lhs_contracting_axes, rhs_contracting_axes)
        && output_axes == canonical_output_axes.as_slice()
        && dst_codomain_rank == lhs.codomain().len()
}

fn is_canonical_fusion_source_contract(
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
