use std::collections::HashMap;

use tenet_core::{
    merge_fusion_trees_generic_checked, merge_fusion_trees_multiplicity_free, BlockKey,
    CanonicalUnitFusionRule, CheckedFusionAlgebra, CheckedFusionSpaceError, CheckedGenericFusion,
    CheckedGenericRigidSymbols, CheckedGenericStructureError, CheckedGenericSymbolError, CoreError,
    FusionProductSpace, FusionStyleKind, FusionTreeHomSpace, FusionTreePairKey,
    FusionTreePairOrientation, GenericBraidScalar, MultiplicityFreeRigidSymbols, MultiplicityIndex,
    OrientedFusionTreeHomSpace, PreparedTreePairOperation, RuleIdentity,
};
use tenet_matrixalgebra::SectorSpectrum;
use tenet_tensors::{
    BoundDynamicFusionMapSpace, BoundDynamicTensorRef, DynamicFusionMapSpace, FusionOperand,
    OutputAxisOrder, RecouplingCoefficientAction, TensorContractSpec, TreeTransformOperation,
    TreeTransformOperationKind, TreeTransformRuleCacheKey,
};

use crate::runtime::Ctx;
use crate::tensor::internal_layout_error;
use crate::typed::ScalarOps;

/// Error from fallible checked-Generic tensor-product execution.
///
/// Provider failures and malformed F arrays remain distinct from local tensor
/// layout failures. Tensor product is not a tree-transform plan, so this type
/// deliberately does not reuse `CheckedGenericPlanError`.
#[derive(Debug)]
pub enum CheckedGenericTensorProductError<E> {
    /// The checked provider rejected an algebra or coefficient query.
    Provider(E),
    /// A returned F array did not have the shape fixed by its sectors.
    SymbolShape {
        /// The malformed categorical symbol.
        symbol: &'static str,
        /// Shape required by the fusion multiplicities.
        expected: Vec<usize>,
        /// Shape returned by the provider.
        actual: Vec<usize>,
    },
    /// A provider-independent categorical invariant failed.
    Core(CoreError),
    /// A local tensor layout or payload invariant failed.
    Operation(tenet_tensors::OperationError),
}

impl<E> From<CoreError> for CheckedGenericTensorProductError<E> {
    fn from(error: CoreError) -> Self {
        Self::Core(error)
    }
}

impl<E> From<tenet_tensors::OperationError> for CheckedGenericTensorProductError<E> {
    fn from(error: tenet_tensors::OperationError) -> Self {
        Self::Operation(error)
    }
}

impl<E> From<CheckedGenericStructureError<E>> for CheckedGenericTensorProductError<E> {
    fn from(error: CheckedGenericStructureError<E>) -> Self {
        match error {
            CheckedGenericStructureError::Provider(error) => Self::Provider(error),
            CheckedGenericStructureError::Core(error) => Self::Core(error),
        }
    }
}

impl<E: core::fmt::Display> core::fmt::Display for CheckedGenericTensorProductError<E> {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Provider(error) => error.fmt(formatter),
            Self::SymbolShape {
                symbol,
                expected,
                actual,
            } => write!(
                formatter,
                "{symbol} shape mismatch: expected {expected:?}, got {actual:?}"
            ),
            Self::Core(error) => error.fmt(formatter),
            Self::Operation(error) => error.fmt(formatter),
        }
    }
}

impl<E: std::error::Error + 'static> std::error::Error for CheckedGenericTensorProductError<E> {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Provider(error) => Some(error),
            Self::Core(error) => Some(error),
            Self::Operation(error) => Some(error),
            Self::SymbolShape { .. } => None,
        }
    }
}

fn map_checked_tensor_product_symbol_error<E>(
    error: CheckedGenericSymbolError<E>,
) -> CheckedGenericTensorProductError<E> {
    match error {
        CheckedGenericSymbolError::Provider(error) => {
            CheckedGenericTensorProductError::Provider(error)
        }
        CheckedGenericSymbolError::Shape {
            symbol,
            expected,
            actual,
        } => CheckedGenericTensorProductError::SymbolShape {
            symbol,
            expected,
            actual,
        },
        CheckedGenericSymbolError::Core(error) => CheckedGenericTensorProductError::Core(error),
    }
}

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
    D: ScalarOps,
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
    D: ScalarOps,
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

pub(crate) enum OrientedContractionKind {
    Contract,
    Compose,
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn tensorcontract_oriented_multiplicity_free<R, D>(
    context: &mut Ctx<D, RuleIdentity>,
    lhs_authority: &BoundDynamicFusionMapSpace<R>,
    lhs: FusionOperand<'_>,
    lhs_data: &[D],
    rhs_authority: &BoundDynamicFusionMapSpace<R>,
    rhs: FusionOperand<'_>,
    rhs_data: &[D],
    lhs_axes: &[usize],
    rhs_axes: &[usize],
    output_order: OutputAxisOrder<'_>,
    kind: OrientedContractionKind,
) -> Result<(BoundDynamicFusionMapSpace<R>, Vec<D>), tenet_tensors::OperationError>
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64>
        + CheckedFusionAlgebra
        + TreeTransformRuleCacheKey<Key = RuleIdentity>,
    D: ScalarOps,
{
    if lhs_authority.provider().rule_identity() != rhs_authority.provider().rule_identity() {
        return Err(tenet_tensors::OperationError::from_core_preserving_context(
            CoreError::FusionRuleMismatch {
                expected: lhs_authority.provider().rule_identity(),
                actual: rhs_authority.provider().rule_identity(),
            },
        ));
    }
    if lhs_axes.len() != rhs_axes.len() {
        return Err(tenet_tensors::OperationError::ContractAxisCountMismatch {
            lhs: lhs_axes.len(),
            rhs: rhs_axes.len(),
        });
    }
    let lhs_rank = lhs.storage_space().rank();
    let rhs_rank = rhs.storage_space().rank();
    // Keep `TensorContractAxisPlan::compile`'s public error order before any
    // oriented homspace/provider work. That plan is private to tenet-tensors;
    // re-exporting it just to share these three syntax checks would widen the
    // expert API.
    for (tensor, axes, rank) in [("lhs", lhs_axes, lhs_rank), ("rhs", rhs_axes, rhs_rank)] {
        let mut seen = vec![false; rank];
        if axes.iter().any(|&axis| {
            if axis >= rank || seen[axis] {
                true
            } else {
                seen[axis] = true;
                false
            }
        }) {
            return Err(tenet_tensors::OperationError::InvalidAxisSet {
                tensor,
                axes: axes.to_vec(),
                rank,
            });
        }
    }
    let lhs_open_rank = lhs_rank - lhs_axes.len();
    let rhs_open_rank = rhs_rank - rhs_axes.len();
    let identity_axes;
    let output_axes = match output_order {
        OutputAxisOrder::Identity => {
            identity_axes = (0..lhs_open_rank + rhs_open_rank).collect::<Vec<_>>();
            identity_axes.as_slice()
        }
        OutputAxisOrder::Axes(axes) => axes,
    };
    if output_axes.len() != lhs_open_rank + rhs_open_rank || {
        let mut seen = vec![false; output_axes.len()];
        output_axes.iter().any(|&axis| {
            if axis >= seen.len() || seen[axis] {
                true
            } else {
                seen[axis] = true;
                false
            }
        })
    } {
        return Err(tenet_tensors::OperationError::InvalidPermutation {
            axes: output_axes.to_vec(),
            rank: lhs_open_rank + rhs_open_rank,
        });
    }
    let lhs_orientation = if lhs.storage_conjugate() {
        FusionTreePairOrientation::Adjoint
    } else {
        FusionTreePairOrientation::Direct
    };
    let rhs_orientation = if rhs.storage_conjugate() {
        FusionTreePairOrientation::Adjoint
    } else {
        FusionTreePairOrientation::Direct
    };
    let homspace = OrientedFusionTreeHomSpace::try_tensorcontract_homspace_checked(
        lhs_authority.provider(),
        OrientedFusionTreeHomSpace::new(lhs.storage_space().homspace(), lhs_orientation),
        OrientedFusionTreeHomSpace::new(rhs.storage_space().homspace(), rhs_orientation),
        lhs_axes,
        rhs_axes,
        output_axes,
        lhs_open_rank,
    )
    .map_err(|error| match error {
        tenet_core::CheckedFusionSpaceError::Core(error) => {
            tenet_tensors::OperationError::from_core_preserving_context(*error)
        }
        tenet_core::CheckedFusionSpaceError::FusionAlgebra(error) => {
            tenet_tensors::OperationError::FusionAlgebra(error)
        }
        _ => tenet_tensors::OperationError::InvalidArgument {
            message: "unknown checked fusion metadata error",
        },
    })?;
    let destination = lhs_authority.derive_from_final_homspace(homspace)?;
    let mut data = vec![D::from_real(0.0); destination.space().required_len()?];
    match kind {
        OrientedContractionKind::Compose => context.tensorcompose_fusion_dyn_into(
            &destination,
            &mut data,
            lhs,
            lhs_data,
            rhs,
            rhs_data,
            lhs_axes,
            rhs_axes,
            D::from_real(1.0),
            D::from_real(0.0),
        )?,
        OrientedContractionKind::Contract => context.tensorcontract_fusion_dyn_prelowered_into(
            &destination,
            &mut data,
            lhs,
            lhs_data,
            rhs,
            rhs_data,
            TensorContractSpec::new_with_conjugation(
                lhs_axes,
                rhs_axes,
                output_order,
                lhs.storage_conjugate(),
                rhs.storage_conjugate(),
            ),
            D::from_real(1.0),
            D::from_real(0.0),
        )?,
    }
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
    D: ScalarOps,
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

type CheckedGenericTensorProductResult<R, D> = Result<
    (BoundDynamicFusionMapSpace<R>, Vec<D>),
    CheckedGenericTensorProductError<<R as CheckedGenericFusion>::Error>,
>;

#[cfg(test)]
static FAIL_CHECKED_TENSOR_PRODUCT_BEFORE_SCATTER: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);
#[cfg(test)]
static CHECKED_TENSOR_PRODUCT_COMMIT_COUNT: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(0);
#[cfg(test)]
std::thread_local! {
    static CHECKED_TENSOR_PRODUCT_RHS_STRUCTURE_OVERRIDE:
        std::cell::RefCell<Option<tenet_core::BlockStructure>> = const {
            std::cell::RefCell::new(None)
        };
}

fn validate_tensor_product_source_structure<E>(
    tensor: &'static str,
    structure: &tenet_core::BlockStructure,
) -> Result<(), CheckedGenericTensorProductError<E>> {
    for index in 0..structure.block_count() {
        let block = structure.block(index)?;
        let BlockKey::FusionTree(key) = block.key() else {
            return Err(
                tenet_tensors::OperationError::ExpectedFusionTreeBlock { tensor, index }.into(),
            );
        };
        if block.shape().len() != structure.rank() || block.strides().len() != structure.rank() {
            return Err(CoreError::StructureRankMismatch {
                expected: structure.rank(),
                actual: block.shape().len(),
            }
            .into());
        }
        let key_rank = key.codomain_tree().uncoupled().len() + key.domain_tree().uncoupled().len();
        if key_rank != structure.rank() {
            return Err(CoreError::StructureRankMismatch {
                expected: structure.rank(),
                actual: key_rank,
            }
            .into());
        }
        for tree in [key.codomain_tree(), key.domain_tree()] {
            let rank = tree.uncoupled().len();
            if tree.is_dual().len() != rank
                || tree.innerlines().len() != rank.saturating_sub(2)
                || tree.vertices().len() != rank.saturating_sub(1)
            {
                return Err(CoreError::MalformedFusionTree {
                    message: "tensor-product source fusion-tree arrays have inconsistent lengths",
                }
                .into());
            }
        }
        if key.codomain_tree().coupled() != key.domain_tree().coupled() {
            return Err(CoreError::MalformedFusionTree {
                message: "tensor-product source tree pair has mismatched coupled sectors",
            }
            .into());
        }
    }
    Ok(())
}

fn validate_tensor_product_source_structures<E>(
    lhs: &tenet_core::BlockStructure,
    rhs: &tenet_core::BlockStructure,
) -> Result<(), CheckedGenericTensorProductError<E>> {
    validate_tensor_product_source_structure("lhs", lhs)?;
    #[cfg(test)]
    {
        return CHECKED_TENSOR_PRODUCT_RHS_STRUCTURE_OVERRIDE.with(|override_structure| {
            let override_structure = override_structure.borrow();
            validate_tensor_product_source_structure(
                "rhs",
                override_structure.as_ref().unwrap_or(rhs),
            )
        });
    }
    #[cfg(not(test))]
    validate_tensor_product_source_structure("rhs", rhs)
}

pub(crate) fn tensorproduct_owned_checked_generic<R, D>(
    lhs_space: &BoundDynamicFusionMapSpace<R>,
    lhs_data: &[D],
    rhs_space: &BoundDynamicFusionMapSpace<R>,
    rhs_data: &[D],
) -> CheckedGenericTensorProductResult<R, D>
where
    R: CheckedGenericRigidSymbols<Scalar = f64>,
    D: ScalarOps,
{
    let rule = lhs_space.provider();
    let lhs_identity = rule.rule_identity();
    let rhs_identity = rhs_space.provider().rule_identity();
    if lhs_identity != rhs_identity {
        return Err(CoreError::FusionRuleMismatch {
            expected: lhs_identity,
            actual: rhs_identity,
        }
        .into());
    }
    for actual in [rule.fusion_style(), rhs_space.provider().fusion_style()] {
        if actual != FusionStyleKind::Generic {
            return Err(CoreError::UnsupportedFusionStyle {
                expected: FusionStyleKind::Generic,
                actual,
            }
            .into());
        }
    }

    // Local storage validation is deliberately after identity/style rejection
    // and before any provider enumeration or symbol query.
    let lhs = BoundDynamicTensorRef::try_new(lhs_space, lhs_data)?;
    let rhs = BoundDynamicTensorRef::try_new(rhs_space, rhs_data)?;
    let lhs_structure = lhs.space().space().structure();
    let rhs_structure = rhs.space().space().structure();
    validate_tensor_product_source_structures::<R::Error>(lhs_structure, rhs_structure)?;
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
    let prepared = lhs
        .space()
        .prepare_final_homspace_generic_with_checked(rule, homspace)?;
    let destination_indices = (0..prepared.structure().block_count())
        .map(|index| {
            let block = prepared.structure().block(index)?;
            let BlockKey::FusionTree(key) = block.key() else {
                return Err(tenet_tensors::OperationError::ExpectedFusionTreeBlock {
                    tensor: "tensor-product destination",
                    index,
                });
            };
            Ok((key.clone(), index))
        })
        .collect::<Result<HashMap<_, _>, tenet_tensors::OperationError>>()?;

    struct Contribution {
        lhs: usize,
        rhs: usize,
        destination: usize,
        coefficient: f64,
    }
    let mut contributions = Vec::new();
    for lhs_index in 0..lhs_structure.block_count() {
        let BlockKey::FusionTree(lhs_key) = lhs_structure.block(lhs_index)?.key() else {
            return Err(tenet_tensors::OperationError::ExpectedFusionTreeBlock {
                tensor: "lhs",
                index: lhs_index,
            }
            .into());
        };
        for rhs_index in 0..rhs_structure.block_count() {
            let BlockKey::FusionTree(rhs_key) = rhs_structure.block(rhs_index)?.key() else {
                return Err(tenet_tensors::OperationError::ExpectedFusionTreeBlock {
                    tensor: "rhs",
                    index: rhs_index,
                }
                .into());
            };
            let left_root = lhs_key.codomain_tree().coupled();
            let right_root = rhs_key.codomain_tree().coupled();
            let channels = rule
                .try_fusion_channels(left_root, right_root)
                .map_err(CheckedGenericTensorProductError::Provider)?;
            for coupled in channels {
                let multiplicity = rule
                    .try_nsymbol(left_root, right_root, coupled)
                    .map_err(CheckedGenericTensorProductError::Provider)?;
                for mu in 1..=multiplicity {
                    let mu = MultiplicityIndex::new(mu).ok_or(
                        tenet_tensors::OperationError::InvalidArgument {
                            message: "invalid Generic root multiplicity",
                        },
                    )?;
                    let codomain = merge_fusion_trees_generic_checked(
                        rule,
                        lhs_key.codomain_tree(),
                        rhs_key.codomain_tree(),
                        coupled,
                        mu,
                    )
                    .map_err(map_checked_tensor_product_symbol_error)?;
                    let domain = merge_fusion_trees_generic_checked(
                        rule,
                        lhs_key.domain_tree(),
                        rhs_key.domain_tree(),
                        coupled,
                        mu,
                    )
                    .map_err(map_checked_tensor_product_symbol_error)?;
                    for (codomain, codomain_coefficient) in &codomain {
                        for (domain, domain_coefficient) in &domain {
                            let key = FusionTreePairKey::pair(codomain.clone(), domain.clone());
                            let destination =
                                destination_indices.get(&key).copied().ok_or_else(|| {
                                    tenet_tensors::OperationError::MissingBlockKey {
                                        key: Box::new(BlockKey::FusionTree(key)),
                                    }
                                })?;
                            contributions.push(Contribution {
                                lhs: lhs_index,
                                rhs: rhs_index,
                                destination,
                                coefficient: *codomain_coefficient
                                    * domain_coefficient.braid_conj(),
                            });
                        }
                    }
                }
            }
        }
    }

    // All provider and F-array work is complete before payload allocation.
    // Scatter targets the staged structure; publishing the bound destination
    // is the final operation below.
    let mut data = vec![D::from_real(0.0); prepared.required_len()];
    #[cfg(test)]
    if FAIL_CHECKED_TENSOR_PRODUCT_BEFORE_SCATTER.swap(false, std::sync::atomic::Ordering::Relaxed)
    {
        return Err(tenet_tensors::OperationError::StructureMismatch {
            tensor: "forced late tensor-product scatter failure",
        }
        .into());
    }
    let lhs_nout = lhs.space().space().nout();
    let rhs_nout = rhs.space().space().nout();
    for contribution in contributions {
        scatter_tensor_product_block(
            lhs.data(),
            lhs_structure.block(contribution.lhs)?,
            lhs_nout,
            rhs.data(),
            rhs_structure.block(contribution.rhs)?,
            rhs_nout,
            &mut data,
            prepared.structure().block(contribution.destination)?,
            contribution.coefficient,
        )?;
    }
    let destination = lhs
        .space()
        .commit_final_homspace_generic_bound_checked(prepared)?;
    #[cfg(test)]
    CHECKED_TENSOR_PRODUCT_COMMIT_COUNT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
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
fn scatter_tensor_product_block<D: ScalarOps>(
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
    D: ScalarOps,
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
    use std::convert::Infallible;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;

    use tenet_core::{
        BlockKey, BlockSpec, BlockStructure, BraidingStyleKind, CheckedGenericFusion,
        CheckedGenericRigidSymbols, CoreError, FusionProductSpace, FusionRule, FusionStyleKind,
        FusionTreeHomSpace, FusionTreeKey, FusionTreePairKey, GenericFArray, GenericRMatrix,
        MultiplicityFreeFusionRule, MultiplicityFreeFusionSymbols, MultiplicityFreeRigidSymbols,
        RuleIdentity, SectorId, SectorLeg, SectorVec, Z2FusionRule,
    };
    use tenet_tensors::{
        BoundDynamicFusionMapSpace, BoundDynamicTensorRef, OutputAxisOrder, TreeTransformOperation,
    };

    use super::{
        pow_by_squaring, scatter_tensor_product_block, tensorcontract_owned_multiplicity_free,
        tensorproduct_owned_checked_generic, tree_transform_owned_multiplicity_free,
        CHECKED_TENSOR_PRODUCT_COMMIT_COUNT, CHECKED_TENSOR_PRODUCT_RHS_STRUCTURE_OVERRIDE,
        FAIL_CHECKED_TENSOR_PRODUCT_BEFORE_SCATTER,
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
        let trace = pow_by_squaring("a".to_string(), 13, |left, right| {
            Ok::<_, ()>(format!("({left}*{right})"))
        })
        .unwrap();
        assert_eq!(trace, "((a*((a*a)*(a*a)))*(((a*a)*(a*a))*((a*a)*(a*a))))");

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

    struct CheckedTensorProductSpy {
        algebra_queries: AtomicUsize,
        f_queries: AtomicUsize,
    }

    impl CheckedTensorProductSpy {
        fn reset(&self) {
            self.algebra_queries.store(0, Ordering::Relaxed);
            self.f_queries.store(0, Ordering::Relaxed);
        }
    }

    impl CheckedGenericFusion for CheckedTensorProductSpy {
        type Error = Infallible;

        fn rule_identity(&self) -> RuleIdentity {
            RuleIdentity::of_type::<Self>()
        }

        fn fusion_style(&self) -> FusionStyleKind {
            FusionStyleKind::Generic
        }

        fn braiding_style(&self) -> BraidingStyleKind {
            BraidingStyleKind::Bosonic
        }

        fn vacuum(&self) -> SectorId {
            SectorId::new(0)
        }

        fn try_dual(&self, sector: SectorId) -> Result<SectorId, Self::Error> {
            self.algebra_queries.fetch_add(1, Ordering::Relaxed);
            Ok(sector)
        }

        fn try_fusion_channels(
            &self,
            _left: SectorId,
            _right: SectorId,
        ) -> Result<SectorVec, Self::Error> {
            self.algebra_queries.fetch_add(1, Ordering::Relaxed);
            Ok([SectorId::new(0)].into_iter().collect())
        }

        fn try_fusion_channels_in_table(
            &self,
            left: SectorId,
            right: SectorId,
        ) -> Result<SectorVec, Self::Error> {
            self.try_fusion_channels(left, right)
        }

        fn try_nsymbol(
            &self,
            _left: SectorId,
            _right: SectorId,
            _coupled: SectorId,
        ) -> Result<usize, Self::Error> {
            self.algebra_queries.fetch_add(1, Ordering::Relaxed);
            Ok(1)
        }
    }

    impl CheckedGenericRigidSymbols for CheckedTensorProductSpy {
        type Scalar = f64;

        fn try_sqrt_dim_scalar(&self, _sector: SectorId) -> Result<f64, Self::Error> {
            Ok(1.0)
        }

        fn try_inv_sqrt_dim_scalar(&self, _sector: SectorId) -> Result<f64, Self::Error> {
            Ok(1.0)
        }

        fn try_frobenius_schur_phase_scalar(&self, _sector: SectorId) -> Result<f64, Self::Error> {
            Ok(1.0)
        }

        fn try_f_symbol_generic(
            &self,
            _a: SectorId,
            _b: SectorId,
            _c: SectorId,
            _d: SectorId,
            _e: SectorId,
            _f: SectorId,
        ) -> Result<GenericFArray<f64>, Self::Error> {
            self.f_queries.fetch_add(1, Ordering::Relaxed);
            Ok(GenericFArray::new(vec![1.0], (1, 1, 1, 1)))
        }

        fn try_r_symbol_generic(
            &self,
            _a: SectorId,
            _b: SectorId,
            _c: SectorId,
        ) -> Result<GenericRMatrix<f64>, Self::Error> {
            Ok(GenericRMatrix::new(vec![1.0], 1, 1))
        }
    }

    fn checked_tensor_product_source(
        provider: Arc<CheckedTensorProductSpy>,
    ) -> (
        BoundDynamicFusionMapSpace<CheckedTensorProductSpy>,
        Vec<f64>,
    ) {
        let leg = || SectorLeg::new([(SectorId::new(0), 1)], false);
        let homspace = FusionTreeHomSpace::new(
            FusionProductSpace::new([leg(), leg()]),
            FusionProductSpace::new([leg(), leg()]),
        );
        let space =
            BoundDynamicFusionMapSpace::from_final_homspace_generic_checked(provider, homspace)
                .unwrap();
        let data = vec![1.0; space.space().required_len().unwrap()];
        (space, data)
    }

    #[test]
    fn checked_tensor_product_rejects_a_later_malformed_key_before_provider_queries() {
        let provider = Arc::new(CheckedTensorProductSpy {
            algebra_queries: AtomicUsize::new(0),
            f_queries: AtomicUsize::new(0),
        });
        let (source, data) = checked_tensor_product_source(Arc::clone(&provider));
        let sector = SectorId::new(0);
        let tree =
            FusionTreeKey::try_new_for_rule(&Z2FusionRule, [sector], sector, [false], [], [])
                .unwrap();
        let valid = BlockSpec::column_major_with_key(
            BlockKey::FusionTree(FusionTreePairKey::pair(tree.clone(), tree)),
            vec![1, 1],
            0,
        )
        .unwrap();
        let rank_zero =
            FusionTreeKey::try_new_for_rule(&Z2FusionRule, [], sector, [], [], []).unwrap();
        let malformed = BlockSpec::column_major_with_key(
            BlockKey::FusionTree(FusionTreePairKey::pair(rank_zero.clone(), rank_zero)),
            vec![1, 1],
            1,
        )
        .unwrap();
        CHECKED_TENSOR_PRODUCT_RHS_STRUCTURE_OVERRIDE.with(|override_structure| {
            *override_structure.borrow_mut() =
                Some(BlockStructure::from_blocks(vec![valid, malformed]).unwrap());
        });
        provider.reset();

        let error =
            tensorproduct_owned_checked_generic(&source, &data, &source, &data).unwrap_err();
        CHECKED_TENSOR_PRODUCT_RHS_STRUCTURE_OVERRIDE.with(|override_structure| {
            override_structure.borrow_mut().take();
        });

        assert!(matches!(
            error,
            super::CheckedGenericTensorProductError::Core(CoreError::StructureRankMismatch {
                expected: 2,
                actual: 0
            })
        ));
        assert_eq!(provider.algebra_queries.load(Ordering::Relaxed), 0);
        assert_eq!(provider.f_queries.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn checked_tensor_product_late_scatter_failure_precedes_publication() {
        let provider = Arc::new(CheckedTensorProductSpy {
            algebra_queries: AtomicUsize::new(0),
            f_queries: AtomicUsize::new(0),
        });
        let (source, data) = checked_tensor_product_source(Arc::clone(&provider));
        provider.reset();
        CHECKED_TENSOR_PRODUCT_COMMIT_COUNT.store(0, Ordering::Relaxed);
        FAIL_CHECKED_TENSOR_PRODUCT_BEFORE_SCATTER.store(true, Ordering::Relaxed);

        let error =
            tensorproduct_owned_checked_generic(&source, &data, &source, &data).unwrap_err();

        assert!(matches!(
            error,
            super::CheckedGenericTensorProductError::Operation(_)
        ));
        assert!(provider.algebra_queries.load(Ordering::Relaxed) > 0);
        assert!(provider.f_queries.load(Ordering::Relaxed) > 0);
        assert_eq!(
            CHECKED_TENSOR_PRODUCT_COMMIT_COUNT.load(Ordering::Relaxed),
            0
        );
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
