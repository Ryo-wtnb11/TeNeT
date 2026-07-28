use std::collections::HashMap;

use tenet_core::{
    merge_fusion_trees_multiplicity_free, BlockKey, CanonicalUnitFusionRule, CheckedFusionAlgebra,
    CheckedFusionSpaceError, CoreError, FusionProductSpace, FusionTreeHomSpace, FusionTreePairKey,
    MultiplicityFreeRigidSymbols, PreparedTreePairOperation, RuleIdentity,
};
use tenet_matrixalgebra::SectorSpectrum;
use tenet_tensors::{
    BoundDynamicFusionMapSpace, BoundDynamicTensorRef, DynamicFusionMapSpace, FusionOperand,
    OutputAxisOrder, RecouplingCoefficientAction, TensorContractSpec, TreeTransformOperation,
    TreeTransformOperationKind, TreeTransformRuleCacheKey,
};

use crate::runtime::Ctx;
use crate::tensor::{internal_layout_error, UserScalar};

/// Positive integer power with no identity seed.
pub(crate) fn pow_by_squaring<T: Clone, E>(
    mut power: T,
    mut exponent: u32,
    mut compose: impl FnMut(&T, &T) -> Result<T, E>,
) -> Result<T, E> {
    debug_assert!(exponent > 0);
    while exponent & 1 == 0 {
        power = compose(&power, &power)?;
        exponent >>= 1;
    }
    let mut result = power.clone();
    exponent >>= 1;
    while exponent != 0 {
        power = compose(&power, &power)?;
        if exponent & 1 != 0 {
            result = compose(&result, &power)?;
        }
        exponent >>= 1;
    }
    Ok(result)
}

/// Transforms a compact diagonal spectrum through a rank-(1,1) leg swap
/// without ever building the `Σ_c k_c²` dense payload — TensorKit 0.17
/// `src/tensors/diagonal.jl:215-242`, where `permute`/`transpose` of a
/// `DiagonalTensorMap` re-labels the stored diagonal instead of materializing
/// it.
///
/// The geometry this accepts is exactly the one already proved for the erased
/// facade: rank `(1, 1)`, codomain permutation `[1]`, domain permutation `[0]`,
/// and a `Permute` or `Transpose` operation — see [`is_rank_one_diagonal_swap`],
/// which both facades ask before calling this. Under it every source block
/// lowers to a **single** destination term, so the whole transform is one real
/// coefficient per sector applied to that sector's stored values. The
/// single-term property is asserted here rather than assumed: zero or several
/// terms is an engine invariant break, not a caller mistake, and is reported as
/// one.
///
/// Why the guard is not widened: an explicit braid, a higher rank, or a Generic
/// (non-multiplicity-free) fusion rule can lower one source block to a *sum* of
/// destination terms, which is no longer a per-sector scaling of a diagonal and
/// has no compact single-term predicate proved for it. Those keep the dense
/// fallback.
///
/// `V` is the stored value type and is only ever acted on by the real
/// coefficient, so the `f64`, `Complex64` and real-valued-`Complex64` storages
/// all share this one body. The bound is the crate's own
/// [`RecouplingCoefficientAction`] rather than a bare `Mul<f64>`: it is the
/// seam every other recoupling coefficient in the engine acts through, and it
/// is already a supertrait of the typed facade's payload scalar, so neither
/// facade has to widen a public bound to reach this helper.
pub(crate) fn transform_rank_one_diagonal_spectrum<R, V>(
    rule: &R,
    source: &DynamicFusionMapSpace,
    destination: &DynamicFusionMapSpace,
    operation: &TreeTransformOperation,
    spectrum: &[SectorSpectrum<V>],
) -> Result<Vec<SectorSpectrum<V>>, crate::error::Error>
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64>,
    V: RecouplingCoefficientAction<f64>,
{
    let source_structure = source.structure();
    if source_structure.block_count() != spectrum.len() {
        return Err(internal_layout_error(
            "compact diagonal spectrum does not cover its rank-one block structure",
        ));
    }
    let spectrum_by_sector = spectrum
        .iter()
        .map(|entry| (entry.sector, entry))
        .collect::<HashMap<_, _>>();
    let mut output_by_sector = HashMap::with_capacity(spectrum.len());
    let prepared = match operation.kind() {
        TreeTransformOperationKind::Permute => PreparedTreePairOperation::prepare_permute(
            rule,
            1,
            1,
            operation.codomain_permutation(),
            operation.domain_permutation(),
        )?,
        TreeTransformOperationKind::Transpose => PreparedTreePairOperation::prepare_transpose(
            1,
            1,
            operation.codomain_permutation(),
            operation.domain_permutation(),
        )?,
        TreeTransformOperationKind::Braid => {
            return Err(internal_layout_error(
                "compact diagonal swap does not accept an explicit braid",
            ));
        }
    };

    for index in 0..source_structure.block_count() {
        let block = source_structure.block(index)?;
        let BlockKey::FusionTree(source) = block.key() else {
            return Err(internal_layout_error(
                "compact diagonal storage requires fusion-tree blocks",
            ));
        };
        let source_sector = source.codomain_tree().coupled();
        let entry = spectrum_by_sector.get(&source_sector).ok_or_else(|| {
            internal_layout_error("compact diagonal spectrum is missing a rank-one block sector")
        })?;
        let rows = prepared.execute_multiplicity_free(rule, source)?;
        let mut rows = rows.into_iter();
        let (destination, coefficient) = rows.next().ok_or_else(|| {
            internal_layout_error("rank-one diagonal swap produced no destination term")
        })?;
        if rows.next().is_some() {
            return Err(internal_layout_error(
                "rank-one diagonal swap produced multiple destination terms",
            ));
        }
        let entry = SectorSpectrum {
            sector: destination.codomain_tree().coupled(),
            values: entry
                .values
                .iter()
                .copied()
                .map(|value| value.scale_by_coefficient(coefficient))
                .collect(),
        };
        if output_by_sector.insert(entry.sector, entry).is_some() {
            return Err(internal_layout_error(
                "rank-one diagonal swap produced duplicate destination sectors",
            ));
        }
    }

    let destination_structure = destination.structure();
    let mut output = Vec::with_capacity(spectrum.len());
    for index in 0..destination_structure.block_count() {
        let block = destination_structure.block(index)?;
        let BlockKey::FusionTree(destination) = block.key() else {
            return Err(internal_layout_error(
                "compact diagonal destination requires fusion-tree blocks",
            ));
        };
        let sector = destination.codomain_tree().coupled();
        let entry = output_by_sector.remove(&sector).ok_or_else(|| {
            internal_layout_error("rank-one diagonal swap is missing a destination block sector")
        })?;
        let [rows, columns] = block.shape() else {
            return Err(internal_layout_error(
                "compact diagonal destination block is not a matrix",
            ));
        };
        if rows != columns || entry.values.len() != *rows {
            return Err(internal_layout_error(
                "rank-one diagonal spectrum does not match its destination block shape",
            ));
        }
        output.push(entry);
    }
    if !output_by_sector.is_empty() || output.len() != spectrum.len() {
        return Err(internal_layout_error(
            "rank-one diagonal swap destination does not cover its compact spectrum",
        ));
    }
    Ok(output)
}

/// TensorKit 0.17 `src/tensors/diagonal.jl:379-381`: `isposdef` on a
/// `DiagonalTensorMap` reads the stored diagonal values rather than
/// factorizing. The caller owns the two steps that come first — the Hermiticity
/// gate and `threshold = tol * max(norm, 1)` — so this is only the comparison.
///
/// Why the real part and not the modulus: the route this replaces
/// eigendecomposes the materialization with the *Hermitian* solver, which reads
/// one triangle and therefore only ever sees the diagonal's real part. A
/// stored value whose imaginary part is not negligible cannot reach here — the
/// Hermiticity gate rejected it — so taking the real part is the same answer,
/// not a looser one.
///
/// Strict: a value exactly at the threshold is `false`, so a positive
/// *semi*definite spectrum is rejected. That is TensorKit's answer on both of
/// its routes — the diagonal one cited above is `all(isposdef, d.data)`, i.e.
/// strict positivity per stored value, and the general one is Cholesky-based,
/// which fails on a singular matrix.
pub(crate) fn compact_is_posdef<V>(spectrum: &[SectorSpectrum<V>], threshold: f64) -> bool
where
    V: tenet_matrixalgebra::FactorScalar,
{
    spectrum
        .iter()
        .flat_map(|entry| entry.values.iter())
        .all(|&value| value.widen_complex().re > threshold)
}

/// Whether `operation` on a tensor of these ranks is the rank-(1,1) leg swap
/// [`transform_rank_one_diagonal_spectrum`] is proved for. Shared so the two
/// facades cannot drift on which geometries take the compact route.
pub(crate) fn is_rank_one_diagonal_swap(
    codomain_rank: usize,
    domain_rank: usize,
    operation: &TreeTransformOperation,
) -> bool {
    codomain_rank == 1
        && domain_rank == 1
        && operation.codomain_permutation() == [1]
        && operation.domain_permutation() == [0]
        && matches!(
            operation.kind(),
            TreeTransformOperationKind::Permute | TreeTransformOperationKind::Transpose
        )
}

// Test-only observability at the two owned multiplicity-free seams, in the
// erased facade's `ORDERED_CONTRACT_FUSED_ROUTE` style (`tensor.rs`): armed
// thread-locals that count executions of the seam the current thread runs
// through. Both facades route here — the erased host fusion path calls
// `tensorcontract_owned_multiplicity_free` too — so a gate over these counters
// pins "one fused contraction, no separate permute transform" without a
// facade-side hook. Observability only: nothing behavioral reads them.
#[cfg(test)]
thread_local! {
    pub(crate) static CONTRACT_SEAM_CALLS: std::cell::Cell<Option<usize>> =
        const { std::cell::Cell::new(None) };
    pub(crate) static TREE_TRANSFORM_SEAM_CALLS: std::cell::Cell<Option<usize>> =
        const { std::cell::Cell::new(None) };
}

#[cfg(test)]
fn observe_contract_seam_call() {
    CONTRACT_SEAM_CALLS.with(|observation| {
        if let Some(calls) = observation.get() {
            observation.set(Some(calls + 1));
        }
    });
}

#[cfg(test)]
fn observe_tree_transform_seam_call() {
    TREE_TRANSFORM_SEAM_CALLS.with(|observation| {
        if let Some(calls) = observation.get() {
            observation.set(Some(calls + 1));
        }
    });
}

/// Executes one owned multiplicity-free transform from a validated provider
/// binding. The caller keeps user-layer dispatch and representation policy;
/// this helper owns only typed destination derivation and the direct/fallback
/// execution choice.
pub(crate) fn tree_transform_owned_multiplicity_free<R, D>(
    context: &mut Ctx<D, RuleIdentity>,
    input: BoundDynamicTensorRef<'_, R, D>,
    operation: TreeTransformOperation,
) -> Result<(BoundDynamicFusionMapSpace<R>, Vec<D>), tenet_tensors::OperationError>
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64> + TreeTransformRuleCacheKey<Key = RuleIdentity>,
    D: UserScalar,
{
    #[cfg(test)]
    observe_tree_transform_seam_call();
    let destination = input.space().transformed_multiplicity_free(&operation)?;
    let dst_space = destination.space();
    if let Some(data) = context
        .tree_context_mut()
        .try_tree_transform_dyn_overwrite_owned(
            input.space().provider(),
            &operation,
            dst_space.structure(),
            input.space().space().structure(),
            dst_space.nout(),
            input.data(),
            D::from_real(1.0),
        )?
    {
        return Ok((destination, data));
    }

    let mut data = vec![D::from_real(0.0); dst_space.required_len()?];
    context.tree_context_mut().tree_transform_dyn_into(
        input.space().provider(),
        operation,
        dst_space.structure(),
        input.space().space().structure(),
        &mut data,
        input.data(),
        D::from_real(1.0),
        D::from_real(0.0),
    )?;
    Ok((destination, data))
}

pub(crate) fn tensorcontract_owned_multiplicity_free<R, D>(
    context: &mut Ctx<D, RuleIdentity>,
    lhs: BoundDynamicTensorRef<'_, R, D>,
    rhs: BoundDynamicTensorRef<'_, R, D>,
    lhs_axes: &[usize],
    rhs_axes: &[usize],
    output_order: OutputAxisOrder<'_>,
) -> Result<(BoundDynamicFusionMapSpace<R>, Vec<D>), tenet_tensors::OperationError>
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64> + TreeTransformRuleCacheKey<Key = RuleIdentity>,
    D: UserScalar,
{
    #[cfg(test)]
    observe_contract_seam_call();
    let destination = BoundDynamicFusionMapSpace::contracted_multiplicity_free_ordered(
        lhs.space(),
        rhs.space(),
        lhs_axes,
        rhs_axes,
        output_order,
    )?;
    let mut data = vec![D::from_real(0.0); destination.space().required_len()?];
    context.tensorcontract_fusion_dyn_into(
        &destination,
        &mut data,
        lhs.space(),
        lhs.data(),
        rhs.space(),
        rhs.data(),
        TensorContractSpec::new(lhs_axes, rhs_axes, output_order),
        D::from_real(1.0),
        D::from_real(0.0),
    )?;
    Ok((destination, data))
}

/// TensorKit tensor product: merge codomain trees with codomain trees and
/// domain trees with domain trees, without crossing either pair of legs.
pub(crate) fn tensorproduct_owned_multiplicity_free<R, D>(
    lhs: BoundDynamicTensorRef<'_, R, D>,
    rhs: BoundDynamicTensorRef<'_, R, D>,
) -> Result<(BoundDynamicFusionMapSpace<R>, Vec<D>), tenet_tensors::OperationError>
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64>
        + CheckedFusionAlgebra
        + CanonicalUnitFusionRule
        + TreeTransformRuleCacheKey<Key = RuleIdentity>,
    D: UserScalar,
{
    let rule = lhs.space().provider();
    if rule.rule_identity() != rhs.space().provider().rule_identity() {
        return Err(tenet_tensors::OperationError::Core(
            CoreError::FusionRuleMismatch {
                expected: rule.rule_identity(),
                actual: rhs.space().provider().rule_identity(),
            },
        ));
    }
    let lhs_hom = lhs.space().space().homspace();
    let rhs_hom = rhs.space().space().homspace();
    let homspace = FusionTreeHomSpace::new(
        FusionProductSpace::new(
            lhs_hom
                .codomain()
                .legs()
                .iter()
                .chain(rhs_hom.codomain().legs())
                .cloned(),
        ),
        FusionProductSpace::new(
            lhs_hom
                .domain()
                .legs()
                .iter()
                .chain(rhs_hom.domain().legs())
                .cloned(),
        ),
    );
    let prepared_destination = homspace
        .prepare_fusion_tree_layout_checked(rule)
        .map_err(|error| tenet_tensors::OperationError::FusionAlgebra(Box::new(error)))?;
    let destination_indices = prepared_destination
        .keys()
        .iter()
        .cloned()
        .enumerate()
        .map(|(index, key)| (key, index))
        .collect::<HashMap<_, _>>();

    struct Contribution {
        lhs: usize,
        rhs: usize,
        destination: usize,
        coefficient: f64,
    }
    let lhs_structure = lhs.space().space().structure();
    let rhs_structure = rhs.space().space().structure();
    let mut contributions = Vec::new();
    for lhs_index in 0..lhs_structure.block_count() {
        let BlockKey::FusionTree(lhs_key) = lhs_structure.block(lhs_index)?.key() else {
            return Err(tenet_tensors::OperationError::ExpectedFusionTreeBlock {
                tensor: "lhs",
                index: lhs_index,
            });
        };
        for rhs_index in 0..rhs_structure.block_count() {
            let BlockKey::FusionTree(rhs_key) = rhs_structure.block(rhs_index)?.key() else {
                return Err(tenet_tensors::OperationError::ExpectedFusionTreeBlock {
                    tensor: "rhs",
                    index: rhs_index,
                });
            };
            for coupled in rule
                .try_fusion_channels(
                    lhs_key.codomain_tree().coupled(),
                    rhs_key.codomain_tree().coupled(),
                )
                .map_err(|error| tenet_tensors::OperationError::FusionAlgebra(Box::new(error)))?
            {
                let codomain = merge_fusion_trees_multiplicity_free(
                    rule,
                    lhs_key.codomain_tree(),
                    rhs_key.codomain_tree(),
                    coupled,
                )
                .map_err(tensor_product_checked_error)?;
                let domain = merge_fusion_trees_multiplicity_free(
                    rule,
                    lhs_key.domain_tree(),
                    rhs_key.domain_tree(),
                    coupled,
                )
                .map_err(tensor_product_checked_error)?;
                for (codomain, codomain_coefficient) in &codomain {
                    for (domain, domain_coefficient) in &domain {
                        let key = FusionTreePairKey::pair(codomain.clone(), domain.clone());
                        let destination_index =
                            destination_indices.get(&key).copied().ok_or_else(|| {
                                tenet_tensors::OperationError::MissingBlockKey {
                                    key: Box::new(BlockKey::FusionTree(key)),
                                }
                            })?;
                        contributions.push(Contribution {
                            lhs: lhs_index,
                            rhs: rhs_index,
                            destination: destination_index,
                            coefficient: *codomain_coefficient
                                * rule.scalar_conj(*domain_coefficient),
                        });
                    }
                }
            }
        }
    }

    // Checked provider work and exact destination-key validation precede
    // destination admission. Only deterministic prepared-key replay remains.
    let destination = lhs.space().derive_from_final_homspace(homspace)?;
    let dst_structure = destination.space().structure();
    if dst_structure.block_count() != prepared_destination.keys().len()
        || prepared_destination
            .keys()
            .iter()
            .enumerate()
            .any(|(index, key)| {
                dst_structure.find_block_index_by_fusion_tree_pair(key) != Some(index)
            })
    {
        return Err(tenet_tensors::OperationError::StructureMismatch {
            tensor: "tensor-product destination",
        });
    }

    // Multiple recoupling paths deliberately accumulate below.
    let mut data = vec![D::from_real(0.0); destination.space().required_len()?];
    let lhs_nout = lhs.space().space().nout();
    let rhs_nout = rhs.space().space().nout();
    for contribution in contributions {
        let lhs_block = lhs_structure.block(contribution.lhs)?;
        let rhs_block = rhs_structure.block(contribution.rhs)?;
        let dst_block = dst_structure.block(contribution.destination)?;
        scatter_tensor_product_block(
            lhs.data(),
            lhs_block,
            lhs_nout,
            rhs.data(),
            rhs_block,
            rhs_nout,
            &mut data,
            dst_block,
            contribution.coefficient,
        )?;
    }
    Ok((destination, data))
}

fn tensor_product_checked_error(error: CheckedFusionSpaceError) -> tenet_tensors::OperationError {
    match error {
        CheckedFusionSpaceError::Core(error) => tenet_tensors::OperationError::Core(*error),
        CheckedFusionSpaceError::FusionAlgebra(error) => {
            tenet_tensors::OperationError::FusionAlgebra(error)
        }
        _ => tenet_tensors::OperationError::InvalidArgument {
            message: "unknown checked fusion-tree merge error",
        },
    }
}

#[allow(clippy::too_many_arguments)]
fn scatter_tensor_product_block<D: UserScalar>(
    lhs_data: &[D],
    lhs: tenet_core::BlockRef<'_>,
    lhs_nout: usize,
    rhs_data: &[D],
    rhs: tenet_core::BlockRef<'_>,
    rhs_nout: usize,
    destination_data: &mut [D],
    destination: tenet_core::BlockRef<'_>,
    coefficient: f64,
) -> Result<(), tenet_tensors::OperationError> {
    if !destination
        .shape()
        .iter()
        .copied()
        .eq(lhs.shape()[..lhs_nout]
            .iter()
            .chain(&rhs.shape()[..rhs_nout])
            .chain(&lhs.shape()[lhs_nout..])
            .chain(&rhs.shape()[rhs_nout..])
            .copied())
    {
        return Err(tenet_tensors::OperationError::StructureMismatch {
            tensor: "tensor-product destination",
        });
    }

    let lhs_count = lhs.element_count()?;
    let rhs_count = rhs.element_count()?;
    let mut lhs_coordinates = vec![0; lhs.shape().len()];
    let mut rhs_coordinates = vec![0; rhs.shape().len()];
    for _ in 0..lhs_count {
        let lhs_position = block_position(lhs.offset(), lhs.strides(), &lhs_coordinates);
        rhs_coordinates.fill(0);
        for _ in 0..rhs_count {
            let rhs_position = block_position(rhs.offset(), rhs.strides(), &rhs_coordinates);
            let destination_position = destination.offset()
                + lhs_coordinates[..lhs_nout]
                    .iter()
                    .zip(&destination.strides()[..lhs_nout])
                    .map(|(&index, &stride)| index * stride)
                    .sum::<usize>()
                + rhs_coordinates[..rhs_nout]
                    .iter()
                    .zip(&destination.strides()[lhs_nout..lhs_nout + rhs_nout])
                    .map(|(&index, &stride)| index * stride)
                    .sum::<usize>()
                + lhs_coordinates[lhs_nout..]
                    .iter()
                    .zip(
                        &destination.strides()[lhs_nout + rhs_nout
                            ..lhs_nout + rhs_nout + lhs.shape().len() - lhs_nout],
                    )
                    .map(|(&index, &stride)| index * stride)
                    .sum::<usize>()
                + rhs_coordinates[rhs_nout..]
                    .iter()
                    .zip(
                        &destination.strides()
                            [lhs_nout + rhs_nout + lhs.shape().len() - lhs_nout..],
                    )
                    .map(|(&index, &stride)| index * stride)
                    .sum::<usize>();
            let value =
                (lhs_data[lhs_position] * rhs_data[rhs_position]).scale_by_coefficient(coefficient);
            destination_data[destination_position] = destination_data[destination_position] + value;
            increment_coordinates(&mut rhs_coordinates, rhs.shape());
        }
        increment_coordinates(&mut lhs_coordinates, lhs.shape());
    }
    Ok(())
}

fn block_position(offset: usize, strides: &[usize], coordinates: &[usize]) -> usize {
    offset
        + coordinates
            .iter()
            .zip(strides)
            .map(|(&index, &stride)| index * stride)
            .sum::<usize>()
}

fn increment_coordinates(coordinates: &mut [usize], shape: &[usize]) {
    for (coordinate, &extent) in coordinates.iter_mut().zip(shape) {
        *coordinate += 1;
        if *coordinate < extent {
            return;
        }
        *coordinate = 0;
    }
}

/// Categorical composition (TensorKit `A * B` / `mul!`) of two owned
/// multiplicity-free operands.
///
/// Differs from [`tensorcontract_owned_multiplicity_free`] in exactly two
/// places: the output order is fixed to the identity (composition has no
/// re-ordering freedom — the open axes keep their sides), and the seam is the
/// composition one, which never inserts the fermionic supertrace twist.
///
/// The operands are always direct: this facade has no lazy adjoint view, so
/// there is no conjugated storage for [`tenet_tensors::FusionOperand`] to
/// separate from its logical geometry.
pub(crate) fn tensorcompose_owned_multiplicity_free<R, D>(
    context: &mut Ctx<D, RuleIdentity>,
    lhs: BoundDynamicTensorRef<'_, R, D>,
    rhs: BoundDynamicTensorRef<'_, R, D>,
    lhs_axes: &[usize],
    rhs_axes: &[usize],
) -> Result<(BoundDynamicFusionMapSpace<R>, Vec<D>), tenet_tensors::OperationError>
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64> + TreeTransformRuleCacheKey<Key = RuleIdentity>,
    D: UserScalar,
{
    let destination = BoundDynamicFusionMapSpace::contracted_multiplicity_free_ordered(
        lhs.space(),
        rhs.space(),
        lhs_axes,
        rhs_axes,
        OutputAxisOrder::identity(),
    )?;
    let mut data = vec![D::from_real(0.0); destination.space().required_len()?];
    context.tensorcompose_fusion_dyn_into(
        &destination,
        &mut data,
        FusionOperand::direct(lhs.space().space()),
        lhs.data(),
        FusionOperand::direct(rhs.space().space()),
        rhs.data(),
        lhs_axes,
        rhs_axes,
        D::from_real(1.0),
        D::from_real(0.0),
    )?;
    Ok((destination, data))
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use tenet_core::{
        BlockKey, BlockSpec, BlockStructure, BraidingStyleKind, FusionProductSpace, FusionRule,
        FusionStyleKind, MultiplicityFreeFusionRule, MultiplicityFreeFusionSymbols,
        MultiplicityFreeRigidSymbols, RuleIdentity, SectorId, SectorLeg, SectorVec, Z2FusionRule,
    };
    use tenet_tensors::{
        BoundDynamicFusionMapSpace, BoundDynamicTensorRef, OutputAxisOrder, TreeTransformOperation,
    };

    use super::{
        pow_by_squaring, scatter_tensor_product_block, tensorcontract_owned_multiplicity_free,
        tree_transform_owned_multiplicity_free,
    };
    use crate::runtime::Ctx;

    #[test]
    fn power_by_squaring_has_logarithmic_composition_count() {
        let mut compositions = 0;
        let power = pow_by_squaring(3_u64, 13, |left, right| {
            compositions += 1;
            Ok::<_, ()>(left * right)
        })
        .unwrap();
        assert_eq!(power, 3_u64.pow(13));
        assert_eq!(compositions, 5);

        compositions = 0;
        assert_eq!(
            pow_by_squaring(1_u64, 1 << 31, |left, right| {
                compositions += 1;
                Ok::<_, ()>(left * right)
            })
            .unwrap(),
            1
        );
        assert_eq!(compositions, 31);
    }

    #[test]
    fn tensor_product_scatter_accumulates_asymmetric_strided_blocks() {
        // What: two contributions to one destination add rather than
        // overwrite. The gapped source layouts make a contiguous-slice
        // shortcut fail, and the unequal shapes pin the external order.
        let lhs_structure = BlockStructure::from_blocks(vec![BlockSpec::with_key(
            BlockKey::trivial(),
            vec![2, 3],
            vec![2, 5],
            0,
        )
        .unwrap()])
        .unwrap();
        let rhs_structure = BlockStructure::from_blocks(vec![BlockSpec::with_key(
            BlockKey::trivial(),
            vec![4, 2],
            vec![1, 7],
            0,
        )
        .unwrap()])
        .unwrap();
        let destination_structure = BlockStructure::trivial(&[2, 4, 3, 2]).unwrap();
        let lhs_block = lhs_structure.only_block().unwrap();
        let rhs_block = rhs_structure.only_block().unwrap();
        let destination_block = destination_structure.only_block().unwrap();
        let mut lhs = vec![f64::NAN; 13];
        let mut rhs = vec![f64::NAN; 11];
        for j in 0..3 {
            for i in 0..2 {
                lhs[i * 2 + j * 5] = (1 + i + 10 * j) as f64;
            }
        }
        for j in 0..2 {
            for i in 0..4 {
                rhs[i + j * 7] = (2 + i + 10 * j) as f64;
            }
        }
        let mut actual = vec![0.0; 48];
        for _ in 0..2 {
            scatter_tensor_product_block(
                &lhs,
                lhs_block,
                1,
                &rhs,
                rhs_block,
                1,
                &mut actual,
                destination_block,
                0.5,
            )
            .unwrap();
        }

        let mut expected = vec![0.0; 48];
        for rhs_domain in 0..2 {
            for lhs_domain in 0..3 {
                for rhs_codomain in 0..4 {
                    for lhs_codomain in 0..2 {
                        let position =
                            lhs_codomain + 2 * rhs_codomain + 8 * lhs_domain + 24 * rhs_domain;
                        expected[position] = (1 + lhs_codomain + 10 * lhs_domain) as f64
                            * (2 + rhs_codomain + 10 * rhs_domain) as f64;
                    }
                }
            }
        }
        assert_eq!(actual, expected);
    }

    /// Deliberately outside the user-layer rule enum: this exercises the typed
    /// core with a provider an application can define without `LoweredMultiplicityFreeAlgebra`.
    struct ExternalZ2;

    impl FusionRule for ExternalZ2 {
        fn rule_identity(&self) -> RuleIdentity {
            RuleIdentity::of_type::<Self>()
        }

        fn fusion_style(&self) -> FusionStyleKind {
            Z2FusionRule.fusion_style()
        }

        fn braiding_style(&self) -> BraidingStyleKind {
            Z2FusionRule.braiding_style()
        }

        fn vacuum(&self) -> SectorId {
            Z2FusionRule.vacuum()
        }

        fn supports_unitary_braid_dagger(&self) -> bool {
            Z2FusionRule.supports_unitary_braid_dagger()
        }

        fn dual(&self, sector: SectorId) -> SectorId {
            Z2FusionRule.dual(sector)
        }

        fn fusion_channels(&self, left: SectorId, right: SectorId) -> SectorVec {
            Z2FusionRule.fusion_channels(left, right)
        }
    }

    impl MultiplicityFreeFusionRule for ExternalZ2 {}

    impl MultiplicityFreeFusionSymbols for ExternalZ2 {
        type Scalar = f64;

        fn scalar_one(&self) -> Self::Scalar {
            Z2FusionRule.scalar_one()
        }

        fn scalar_conj(&self, value: Self::Scalar) -> Self::Scalar {
            Z2FusionRule.scalar_conj(value)
        }

        fn has_trivial_associator_gauge(&self) -> bool {
            Z2FusionRule.has_trivial_associator_gauge()
        }

        fn f_symbol_scalar(
            &self,
            left: SectorId,
            middle: SectorId,
            right: SectorId,
            coupled: SectorId,
            left_coupled: SectorId,
            right_coupled: SectorId,
        ) -> Self::Scalar {
            Z2FusionRule.f_symbol_scalar(left, middle, right, coupled, left_coupled, right_coupled)
        }

        fn r_symbol_scalar(
            &self,
            left: SectorId,
            right: SectorId,
            coupled: SectorId,
        ) -> Self::Scalar {
            Z2FusionRule.r_symbol_scalar(left, right, coupled)
        }
    }

    impl MultiplicityFreeRigidSymbols for ExternalZ2 {
        fn dim_scalar(&self, sector: SectorId) -> Self::Scalar {
            Z2FusionRule.dim_scalar(sector)
        }

        fn inv_dim_scalar(&self, sector: SectorId) -> Self::Scalar {
            Z2FusionRule.inv_dim_scalar(sector)
        }

        fn sqrt_dim_scalar(&self, sector: SectorId) -> Self::Scalar {
            Z2FusionRule.sqrt_dim_scalar(sector)
        }

        fn inv_sqrt_dim_scalar(&self, sector: SectorId) -> Self::Scalar {
            Z2FusionRule.inv_sqrt_dim_scalar(sector)
        }

        fn twist_scalar(&self, sector: SectorId) -> Self::Scalar {
            Z2FusionRule.twist_scalar(sector)
        }

        fn frobenius_schur_phase_scalar(&self, sector: SectorId) -> Self::Scalar {
            Z2FusionRule.frobenius_schur_phase_scalar(sector)
        }
    }

    #[test]
    fn external_multiplicity_free_provider_matches_direct_transform() {
        let provider = Arc::new(ExternalZ2);
        let leg = SectorLeg::new([(SectorId::new(0), 2)], false);
        let source = BoundDynamicFusionMapSpace::from_degeneracy_shapes(
            Arc::clone(&provider),
            tenet_core::FusionTreeHomSpace::new(
                FusionProductSpace::new([leg.clone()]),
                FusionProductSpace::new([leg]),
            ),
            [vec![2, 2]],
        )
        .unwrap();
        let operation = TreeTransformOperation::permute([1], [0]);
        let source_data = vec![1.0, 2.0, 3.0, 4.0];

        let expected_destination = source.transformed_multiplicity_free(&operation).unwrap();
        let mut expected_data = vec![0.0; expected_destination.space().required_len().unwrap()];
        let mut direct = Ctx::<f64, RuleIdentity>::default();
        direct
            .tree_context_mut()
            .tree_transform_dyn_into(
                provider.as_ref(),
                operation.clone(),
                expected_destination.space().structure(),
                source.space().structure(),
                &mut expected_data,
                &source_data,
                1.0,
                0.0,
            )
            .unwrap();

        let input: tenet_matrixalgebra::BoundDynamicTensorRef<'_, ExternalZ2, f64> =
            BoundDynamicTensorRef::try_new(&source, &source_data).unwrap();
        let mut context = Ctx::<f64, RuleIdentity>::default();
        let (actual_destination, actual_data) =
            tree_transform_owned_multiplicity_free(&mut context, input, operation).unwrap();

        assert_eq!(actual_destination.space(), expected_destination.space());
        assert_eq!(actual_data, expected_data);
    }

    #[test]
    fn external_multiplicity_free_provider_contracts_direct_with_output_order() {
        let provider = Arc::new(ExternalZ2);
        let lhs_codomain = SectorLeg::new([(SectorId::new(0), 2)], false);
        let contracted = SectorLeg::new([(SectorId::new(0), 3)], false);
        let rhs_domain = SectorLeg::new([(SectorId::new(0), 4)], false);
        let lhs = BoundDynamicFusionMapSpace::from_degeneracy_shapes(
            Arc::clone(&provider),
            tenet_core::FusionTreeHomSpace::new(
                FusionProductSpace::new([lhs_codomain.clone()]),
                FusionProductSpace::new([contracted.clone()]),
            ),
            [vec![2, 3]],
        )
        .unwrap();
        let rhs = BoundDynamicFusionMapSpace::from_degeneracy_shapes(
            Arc::clone(&provider),
            tenet_core::FusionTreeHomSpace::new(
                FusionProductSpace::new([contracted]),
                FusionProductSpace::new([rhs_domain.clone()]),
            ),
            [vec![3, 4]],
        )
        .unwrap();
        let lhs_data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let rhs_data = vec![
            7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0,
        ];

        let lhs = BoundDynamicTensorRef::try_new(&lhs, &lhs_data).unwrap();
        let rhs = BoundDynamicTensorRef::try_new(&rhs, &rhs_data).unwrap();
        let mut context = Ctx::<f64, RuleIdentity>::default();
        let (destination, data) = tensorcontract_owned_multiplicity_free(
            &mut context,
            lhs,
            rhs,
            &[1],
            &[0],
            OutputAxisOrder::from_axes(&[1, 0]),
        )
        .unwrap();

        let expected_destination = tenet_core::FusionTreeHomSpace::new(
            FusionProductSpace::new([SectorLeg::new([(SectorId::new(0), 4)], true)]),
            FusionProductSpace::new([SectorLeg::new([(SectorId::new(0), 2)], true)]),
        );
        assert_eq!(destination.space().homspace(), &expected_destination);
        assert_eq!(
            data,
            [76.0, 103.0, 130.0, 157.0, 100.0, 136.0, 172.0, 208.0]
        );
    }

    /// A rank-3 Z2 tensor map built through the typed facade, so the gate
    /// below measures the whole typed route, not this module in isolation.
    fn typed_z2_facade_tensor() -> (
        crate::typed::Runtime,
        crate::typed::TensorMap<Z2FusionRule, f64>,
    ) {
        let runtime = crate::typed::Runtime::builder().build().unwrap();
        let leg = crate::typed::GradedSpace::try_new(
            Arc::new(Z2FusionRule),
            [
                (tenet_core::Z2Irrep::EVEN, 2),
                (tenet_core::Z2Irrep::ODD, 3),
            ],
            false,
        )
        .unwrap();
        let tensor =
            crate::typed::TensorMap::from_block_fn(&runtime, [&leg, &leg], [&leg], |_, indices| {
                (indices[0] * 7 + indices[1] * 3 + indices[2]) as f64 + 1.0
            })
            .unwrap();
        (runtime, tensor)
    }

    /// Arms both seam counters, runs `operation`, and returns
    /// `(contract seam calls, tree-transform seam calls)`.
    fn seam_calls<T>(operation: impl FnOnce() -> T) -> (usize, usize) {
        super::CONTRACT_SEAM_CALLS.with(|observation| observation.set(Some(0)));
        super::TREE_TRANSFORM_SEAM_CALLS.with(|observation| observation.set(Some(0)));
        let _output = operation();
        (
            super::CONTRACT_SEAM_CALLS
                .with(|observation| observation.replace(None))
                .unwrap(),
            super::TREE_TRANSFORM_SEAM_CALLS
                .with(|observation| observation.replace(None))
                .unwrap(),
        )
    }

    #[test]
    fn typed_ordered_contract_is_one_fused_seam_call_and_no_permute_transform() {
        // What (#580 group 6, gate 1): a typed `contract_ordered` with a
        // non-identity output order runs the fused contraction seam exactly
        // once and never a separate permute transform — the typed sibling of
        // the erased `ORDERED_CONTRACT_FUSED_ROUTE` gate in `tensor.rs`.
        let (_runtime, tensor) = typed_z2_facade_tensor();

        // Negative control: the counters can see the sequential shape at all.
        // A contract followed by a public permute is one fused call plus one
        // transform, which is what a regressed contract-then-permute alias
        // would look like.
        let (fused, transforms) = seam_calls(|| {
            tensor
                .contract(&tensor, &[2], &[0], &[0, 1, 2, 3])
                .unwrap()
                .permute(&[1, 0], &[3, 2])
                .unwrap()
        });
        assert_eq!((fused, transforms), (1, 1));

        // The gate: the non-identity order is folded into the one fused call.
        let (fused, transforms) = seam_calls(|| {
            tensor
                .contract_ordered(&tensor, &[2], &[0], &[1, 0, 3, 2])
                .unwrap()
        });
        assert_eq!(
            (fused, transforms),
            (1, 0),
            "ordered contract must be one fused contraction, not contract-then-permute"
        );
    }
}
