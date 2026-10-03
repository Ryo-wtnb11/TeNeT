use super::*;

#[derive(Clone, Debug)]
pub(super) struct RhsTwistAction<C = f64> {
    pub(super) shape: Vec<usize>,
    pub(super) strides: Vec<isize>,
    pub(super) offset: isize,
    pub(super) factor: C,
}

/// The fermionic contraction twist of one materialized core operand `space`:
/// `θ` of each block is read from its contracted fusion tree — the codomain
/// tree of a core-right block, the domain tree of a core-left block (equal to
/// the codomain tree of every core-right block it meets) — at the positions
/// where `core_right`'s contracted leg is dual, as TensorKit twists `B`'s
/// dual codomain legs or the matching `A` domain legs (`blas_contract!`,
/// src/tensors/tensoroperations.jl:419/429 @cfaa073).
///
/// Also appends `(block offset, θ_b)` for every non-empty twisted block to
/// `destination_scales` when given (unsorted; a zero-element block is left
/// out because it can share its offset with the next block).
pub(super) fn compile_contract_twist<R>(
    rule: &R,
    space: &DynamicFusionMapSpace,
    core_right: &FusionTreeHomSpace,
    space_is_core_left: bool,
    rhs_contracting_axes: &[usize],
    mut destination_scales: Option<&mut Vec<(usize, R::Scalar)>>,
) -> Result<Arc<[RhsTwistAction<R::Scalar>]>, OperationError>
where
    R: MultiplicityFreeRigidSymbols,
    R::Scalar: DenseBlockScalar,
{
    if rule.braiding_style() != tenet_core::BraidingStyleKind::Fermionic {
        return Ok(Arc::from([]));
    }
    let structure = std::sync::Arc::clone(space.structure());
    let mut actions = Vec::new();
    for index in 0..structure.block_count() {
        let block = structure
            .block(index)
            .map_err(OperationError::from_core_preserving_context)?;
        let tenet_core::BlockKey::FusionTree(key) = block.key() else {
            continue;
        };
        let contracted_tree = if space_is_core_left {
            key.domain_tree()
        } else {
            key.codomain_tree()
        };
        let factor = super::fusion::rhs_contract_twist_factor(
            rule,
            core_right,
            rhs_contracting_axes,
            contracted_tree,
        )?;
        if factor != R::Scalar::one() {
            if let Some(scales) = destination_scales.as_deref_mut() {
                if !block.shape().contains(&0) {
                    scales.push((block.offset(), factor));
                }
            }
            actions.push(RhsTwistAction {
                shape: block.shape().to_vec(),
                strides: tenet_operations::strided::strides_to_isize(block.strides())?,
                offset: tenet_operations::strided::offset_to_isize(block.offset())?,
                factor,
            });
        }
    }
    Ok(actions.into())
}

/// The Host twist compiler's `(offset, θ)` list for `space`, sorted: the test
/// oracle for one physical operand's twist, independent of any artifact.
#[cfg(test)]
pub(in crate::contract) fn contract_twist_scales<R>(
    rule: &R,
    space: &DynamicFusionMapSpace,
    core_right: &FusionTreeHomSpace,
    space_is_core_left: bool,
    rhs_contracting_axes: &[usize],
) -> Result<Vec<(usize, R::Scalar)>, OperationError>
where
    R: MultiplicityFreeRigidSymbols,
    R::Scalar: DenseBlockScalar,
{
    let mut scales = Vec::new();
    compile_contract_twist(
        rule,
        space,
        core_right,
        space_is_core_left,
        rhs_contracting_axes,
        Some(&mut scales),
    )?;
    scales.sort_unstable_by_key(|&(offset, _)| offset);
    Ok(scales)
}

/// θ_b of the destination block at `offset`: one where `scales` (sorted by
/// offset) has no entry.
fn destination_scale<C: DenseBlockScalar>(scales: &[(usize, C)], offset: usize) -> C {
    scales
        .binary_search_by_key(&offset, |&(block, _)| block)
        .map_or_else(|_| C::one(), |index| scales[index].1)
}

/// Asserts θ is uniform over the destination layouts of every Multi block of
/// the twisted operand's transform. Scaling each scatter by its own block's θ
/// would be exact either way; what this pins is the invariant behind it: θ
/// depends only on the destination's contracted uncoupled sectors (codomain
/// of a core-right block, domain of a core-left one), which one Multi block
/// shares, so a non-uniform list means the
/// twist and the transform disagree about block identity — an internal
/// inconsistency of the compiled artifact, reported before any replay.
pub(in crate::contract) fn validate_uniform_multi_scales<C: DenseBlockScalar>(
    structure: &TreeTransformStructure<C>,
    scales: &[(usize, C)],
) -> Result<(), OperationError> {
    let layouts = structure.layouts();
    for block in structure.blocks() {
        let tenet_operations::TreeTransformBlock::Multi {
            dst_layout_start,
            dst_count,
            ..
        } = *block
        else {
            continue;
        };
        let mut first = None;
        for index in dst_layout_start..dst_layout_start + dst_count {
            let layout = layouts.entry(index);
            if layout.element_count == 0 {
                continue;
            }
            let offset =
                usize::try_from(layout.offset).map_err(|_| OperationError::InvalidArgument {
                    message: "twisted transform destination layout has a negative offset",
                })?;
            let theta = destination_scale(scales, offset);
            match first {
                None => first = Some(theta),
                Some(expected) if expected == theta => {}
                Some(_) => {
                    return Err(OperationError::InvalidArgument {
                        message: "fermionic contraction twist is not uniform within one \
                                  recoupling block",
                    })
                }
            }
        }
    }
    Ok(())
}

pub(super) fn apply_contract_twist<A, R, D>(
    kernels: &mut A,
    rule: &R,
    space: &DynamicFusionMapSpace,
    core_right: &FusionTreeHomSpace,
    space_is_core_left: bool,
    data: &mut [D],
    rhs_contracting_axes: &[usize],
) -> Result<(), OperationError>
where
    A: crate::HostKernelAdapter<D>,
    R: MultiplicityFreeRigidSymbols,
    R::Scalar: DenseBlockScalar,
    D: DenseRecouplingScalar + RecouplingCoefficientAction<R::Scalar>,
{
    let actions = compile_contract_twist(
        rule,
        space,
        core_right,
        space_is_core_left,
        rhs_contracting_axes,
        None,
    )?;
    execute_contract_twist(kernels, data, &actions)
}

/// Applies the fermionic supertrace actions compiled with the tree artifact.
/// Why not retain the rule here: numerical replay must not re-enter categorical
/// coefficient evaluation or rebuild strided descriptors.
pub(super) fn execute_contract_twist<A, D, C>(
    kernels: &mut A,
    data: &mut [D],
    actions: &[RhsTwistAction<C>],
) -> Result<(), OperationError>
where
    A: crate::HostKernelAdapter<D>,
    D: DenseRecouplingScalar + RecouplingCoefficientAction<C>,
    C: Copy,
{
    for action in actions {
        kernels.scale_strided(
            data,
            &action.shape,
            &action.strides,
            action.offset,
            D::coefficient_as_data(action.factor),
        )?;
    }
    Ok(())
}
