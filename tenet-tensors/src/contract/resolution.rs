//! Eager contraction route compilation.
//!
//! Ordinary calls resolve per operation, as TensorKit and QSpace do. Explicit
//! prepared handles own the returned [`Resolution`] and any complete dynamic
//! execution artifact required for lookup-free replay.

use std::sync::Arc;

use tenet_core::{FusionTreeHomSpace, FusionTreePairOrientation, MultiplicityFreeRigidSymbols};

use super::structure::TensorContractStructure;
use crate::{DenseBlockScalar, OperationError};
use tenet_operations::axis::{OutputAxisOrder, TensorContractSpec};
use tenet_operations::fusion_replay::FusionBlockContractPlan;
use tenet_operations::TensorContractFusionProfile;

use super::dynamic_space::{DynamicFusionMapSpace, FusionOperand, FusionOperandLayout};
use super::fusion::{
    contracted_axis_order_candidates, external_axis_is_dual, is_core_form_fusion_source_contract,
    rhs_contract_twist_factor_oriented, FusionContractOrientation, FusionContractPlan,
    CACHED_ORIENTATIONS,
};
use super::fusion_block::{
    compile_fusion_block_contract_plan_core_geometry,
    compile_fusion_block_contract_plan_prelowered_validated,
    compile_fusion_block_contract_plan_validated, try_compile_oriented_canonical_core_plan,
    try_compile_scaled_canonical_core_plan, CoreContractPreflight, ValidatedCoreContract,
};

/// Resolved execution artifact for one contraction key: the route decision
/// and its compiled plan are one value, never cached separately.
#[derive(Clone, Debug)]
pub(crate) enum Resolution<C = f64> {
    /// Coupled-sector direct GEMM (TensorKit `mul!` shape).
    Core(Arc<FusionBlockContractPlan<C>>),
    /// [`Self::Core`] of the swapped candidate B·A (TensorKit
    /// `blas_contract!(C, B, reverse(pB), A, reverse(pA), pAB′)`): the plan's
    /// lhs is the caller's rhs, and replay passes the operands swapped.
    SwappedCore(Arc<FusionBlockContractPlan<C>>),
    /// Source/output tree transforms around a core contraction
    /// (TensorKit `@tensor` shape).
    DynamicTree(Arc<FusionContractPlan>),
    /// Dense one-shot structure for source/output transforms (TeNeT
    /// optimization over the faithful transform-then-contract path).
    Structure(Arc<TensorContractStructure<C>>),
}

/// Host-compiled, owned route of one contraction whose payloads are not
/// host slices (the device path): everything categorical — route choice,
/// orientation and axis order, source/output transform structures, borrow
/// decisions, core plan and twist classification — is decided here, on the
/// host, by the same compilers the Host contraction runs; a storage executor
/// only replays it.
///
/// Created by
/// [`TensorContractFusionExecutionContext::compile_storage_contract_resolution`](super::TensorContractFusionExecutionContext::compile_storage_contract_resolution).
#[doc(hidden)]
#[derive(Clone, Debug)]
pub struct StorageContractResolution<C = f64> {
    pub(crate) route: StorageContractRoute<C>,
}

impl<C: DenseBlockScalar> StorageContractRoute<C> {
    pub(crate) fn block_plan_is_fully_direct(&self) -> bool {
        match self {
            Self::Core(plan) | Self::SwappedCore(plan) => plan.is_fully_direct(),
            Self::DynamicTree(artifact) => artifact.block_plan_is_fully_direct(),
        }
    }

    /// The core plan whose GEMMs this route runs.
    #[cfg(feature = "cuda")]
    pub(crate) fn block_plan(&self) -> &FusionBlockContractPlan<C> {
        match self {
            Self::Core(plan) | Self::SwappedCore(plan) => plan,
            Self::DynamicTree(artifact) => artifact.block_plan(),
        }
    }
}

#[derive(Clone, Debug)]
pub(crate) enum StorageContractRoute<C> {
    /// Canonical fully-direct coupled-sector GEMM batch over the parent
    /// buffers (lazy adjoints as GEMM operand flags, a uniform fermionic twist
    /// as per-job alpha).
    #[cfg_attr(not(feature = "cuda"), allow(dead_code))]
    Core(Arc<FusionBlockContractPlan<C>>),
    /// [`Self::Core`] of the swapped candidate B·A; replay passes the
    /// operands swapped (see [`Resolution::SwappedCore`]).
    #[cfg_attr(not(feature = "cuda"), allow(dead_code))]
    SwappedCore(Arc<FusionBlockContractPlan<C>>),
    /// Source tree transforms → fully-direct core GEMM → output transform.
    DynamicTree(Arc<super::dynamic::DynamicTreeExecutionArtifact<C>>),
}

impl<C: DenseBlockScalar> StorageContractResolution<C> {
    /// A negatively strided inactive core block is rejected here, before the
    /// device lease; the device replay converts the plan's inactive blocks
    /// into its lease scratch only when it has to zero them.
    pub(crate) fn new(route: StorageContractRoute<C>) -> Result<Self, OperationError> {
        #[cfg(feature = "cuda")]
        super::dynamic::cuda::validate_inactive_regions(route.block_plan())?;
        Ok(Self { route })
    }

    /// True when the route needs the fermionic contraction twist of one
    /// materialized operand: the Host scales it in place after its source
    /// transform, the device folds it into that transform's destination
    /// writes.
    pub fn requires_source_twist(&self) -> bool {
        match &self.route {
            StorageContractRoute::Core(_) | StorageContractRoute::SwappedCore(_) => false,
            StorageContractRoute::DynamicTree(artifact) => artifact.requires_source_twist(),
        }
    }

    /// The core plan's inactive destination blocks when the core GEMMs write
    /// the caller's destination directly (the `Core` route, or an identity
    /// output); `None` when an output transform writes it.
    #[cfg(test)]
    pub(crate) fn direct_destination_inactive_blocks(&self) -> Option<usize> {
        match &self.route {
            StorageContractRoute::Core(plan) | StorageContractRoute::SwappedCore(plan) => {
                Some(plan.inactive_destination_regions().len())
            }
            StorageContractRoute::DynamicTree(artifact) => {
                artifact.direct_destination_inactive_blocks()
            }
        }
    }

    /// True when the route runs source/output tree transforms around the core.
    pub fn is_dynamic_tree(&self) -> bool {
        matches!(self.route, StorageContractRoute::DynamicTree(_))
    }
}

/// Compiles the route and plan for one ordinary contraction.
pub(crate) fn compile_resolution<R>(
    rule: &R,
    dst: &DynamicFusionMapSpace,
    lhs: &DynamicFusionMapSpace,
    rhs: &DynamicFusionMapSpace,
    axes: TensorContractSpec<'_>,
    compile_structure: impl FnOnce() -> Result<
        Option<Arc<TensorContractStructure<R::Scalar>>>,
        OperationError,
    >,
    compile_dynamic: impl FnOnce() -> Result<Arc<FusionContractPlan>, OperationError>,
) -> Result<Resolution<R::Scalar>, OperationError>
where
    R: MultiplicityFreeRigidSymbols,
    R::Scalar: DenseBlockScalar,
{
    compile_resolution_with_profile::<R, false>(
        rule,
        dst,
        lhs,
        rhs,
        axes,
        compile_structure,
        compile_dynamic,
        None,
    )
}

/// Compiles and attributes the ordinary eager route without introducing a
/// reusable execution artifact.
#[expect(
    clippy::too_many_arguments,
    reason = "profiled resolution keeps three spaces, TensorContractSpec, two lazy compilers, and profile sink explicit"
)]
pub(crate) fn compile_resolution_profiled<R>(
    rule: &R,
    dst: &DynamicFusionMapSpace,
    lhs: &DynamicFusionMapSpace,
    rhs: &DynamicFusionMapSpace,
    axes: TensorContractSpec<'_>,
    compile_structure: impl FnOnce() -> Result<
        Option<Arc<TensorContractStructure<R::Scalar>>>,
        OperationError,
    >,
    compile_dynamic: impl FnOnce() -> Result<Arc<FusionContractPlan>, OperationError>,
    profile: &mut TensorContractFusionProfile,
) -> Result<Resolution<R::Scalar>, OperationError>
where
    R: MultiplicityFreeRigidSymbols,
    R::Scalar: DenseBlockScalar,
{
    compile_resolution_with_profile::<R, true>(
        rule,
        dst,
        lhs,
        rhs,
        axes,
        compile_structure,
        compile_dynamic,
        Some(profile),
    )
}

#[allow(clippy::too_many_arguments)]
fn compile_resolution_with_profile<R, const PROFILED: bool>(
    rule: &R,
    dst: &DynamicFusionMapSpace,
    lhs: &DynamicFusionMapSpace,
    rhs: &DynamicFusionMapSpace,
    axes: TensorContractSpec<'_>,
    compile_structure: impl FnOnce() -> Result<
        Option<Arc<TensorContractStructure<R::Scalar>>>,
        OperationError,
    >,
    compile_dynamic: impl FnOnce() -> Result<Arc<FusionContractPlan>, OperationError>,
    mut profile: Option<&mut TensorContractFusionProfile>,
) -> Result<Resolution<R::Scalar>, OperationError>
where
    R: MultiplicityFreeRigidSymbols,
    R::Scalar: DenseBlockScalar,
{
    let preflight_start = PROFILED.then(std::time::Instant::now);
    let preflight = CoreContractPreflight::compile(rule, dst, lhs, rhs, axes)?;
    if !preflight.has_conjugation() {
        if let Some(validated) = preflight.validate_core_geometry()? {
            if !validated_rhs_contract_requires_twist(&validated)? {
                record_resolution_preflight(&mut profile, preflight_start);
                let block_plan_start = profile.as_ref().map(|_| std::time::Instant::now());
                let plan = compile_fusion_block_contract_plan_validated(validated, dst, lhs, rhs)?;
                if let Some(start) = block_plan_start {
                    profile
                        .as_deref_mut()
                        .expect("profiled route compilation carries a profile")
                        .core_block_plan_build += start.elapsed();
                }
                return Ok(Resolution::Core(Arc::new(plan)));
            }
        }
        let candidate = try_zero_copy_contract_candidates(
            rule,
            dst.nout(),
            FusionOperand::direct(lhs),
            FusionOperand::direct(rhs),
            axes,
            false,
            |core_lhs, core_rhs, core_axes, orientation| {
                let (core_lhs, core_rhs) = (core_lhs.storage_space(), core_rhs.storage_space());
                let Some(validated) =
                    CoreContractPreflight::compile(rule, dst, core_lhs, core_rhs, core_axes)?
                        .validate_core_geometry()?
                else {
                    return Ok(None);
                };
                let plan = compile_fusion_block_contract_plan_validated(
                    validated, dst, core_lhs, core_rhs,
                )?;
                Ok(Some(Resolution::core(plan, orientation)))
            },
        )?;
        record_resolution_preflight(&mut profile, preflight_start);
        if let Some(resolution) = candidate {
            return Ok(resolution);
        }
        return compile_dynamic_tree_plan::<R::Scalar, PROFILED>(compile_dynamic, &mut profile);
    }
    if let Some(structure) = compile_structure()? {
        record_resolution_preflight(&mut profile, preflight_start);
        return Ok(Resolution::Structure(structure));
    }
    record_resolution_preflight(&mut profile, preflight_start);
    compile_dynamic_tree_plan::<R::Scalar, PROFILED>(compile_dynamic, &mut profile)
}

fn record_resolution_preflight(
    profile: &mut Option<&mut TensorContractFusionProfile>,
    start: Option<std::time::Instant>,
) {
    if let Some(start) = start {
        profile
            .as_deref_mut()
            .expect("profiled route compilation carries a profile")
            .resolution_preflight += start.elapsed();
    }
}

fn compile_dynamic_tree_plan<C, const PROFILED: bool>(
    compile_dynamic: impl FnOnce() -> Result<Arc<FusionContractPlan>, OperationError>,
    profile: &mut Option<&mut TensorContractFusionProfile>,
) -> Result<Resolution<C>, OperationError> {
    let start = PROFILED.then(std::time::Instant::now);
    let plan = compile_dynamic()?;
    if let Some(start) = start {
        profile
            .as_deref_mut()
            .expect("profiled route compilation carries a profile")
            .dynamic_tree_plan_build += start.elapsed();
    }
    Ok(Resolution::DynamicTree(plan))
}

/// Compiles one contraction whose logical and storage spaces are already
/// separated by a validated lazy-adjoint boundary.
#[allow(clippy::too_many_arguments)]
pub(crate) fn compile_prelowered_resolution<R>(
    rule: &R,
    dst: &DynamicFusionMapSpace,
    lhs: &FusionOperandLayout<'_>,
    rhs: &FusionOperandLayout<'_>,
    axes: TensorContractSpec<'_>,
    compile_structure: impl FnOnce() -> Result<
        Option<Arc<TensorContractStructure<R::Scalar>>>,
        OperationError,
    >,
    compile_dynamic: impl FnOnce() -> Result<Arc<FusionContractPlan>, OperationError>,
) -> Result<Resolution<R::Scalar>, OperationError>
where
    R: MultiplicityFreeRigidSymbols,
    R::Scalar: DenseBlockScalar,
{
    let preflight = CoreContractPreflight::compile_oriented(
        rule,
        dst.homspace(),
        lhs.oriented_homspace(),
        rhs.oriented_homspace(),
        axes,
    )?;
    let has_conjugation = preflight.has_conjugation();
    if let Some(validated) = preflight.validate_core_geometry()? {
        if !validated_rhs_contract_requires_twist(&validated)? {
            let plan =
                compile_fusion_block_contract_plan_prelowered_validated(validated, dst, lhs, rhs)?;
            return Ok(Resolution::Core(Arc::new(plan)));
        }
    }
    if !has_conjugation {
        return Ok(Resolution::DynamicTree(compile_dynamic()?));
    }
    if let Some(structure) = compile_structure()? {
        return Ok(Resolution::Structure(structure));
    }
    Ok(Resolution::DynamicTree(compile_dynamic()?))
}

/// Tries the parent-owned coupled-region route before an adjoint operand
/// derives logical block keys, on every zero-copy TensorKit candidate (see
/// [`try_zero_copy_contract_candidates`]). A miss is not an error: the exact
/// projection is then prepared by the general prelowered lowering path.
pub(crate) fn try_compile_oriented_canonical_core_resolution<R>(
    rule: &R,
    dst: &DynamicFusionMapSpace,
    lhs: FusionOperand<'_>,
    rhs: FusionOperand<'_>,
    axes: TensorContractSpec<'_>,
) -> Result<Option<Resolution<R::Scalar>>, OperationError>
where
    R: MultiplicityFreeRigidSymbols,
    R::Scalar: DenseBlockScalar,
{
    try_zero_copy_contract_candidates(
        rule,
        dst.nout(),
        lhs,
        rhs,
        axes,
        false,
        |core_lhs, core_rhs, core_axes, orientation| {
            let preflight = CoreContractPreflight::compile_oriented(
                rule,
                dst.homspace(),
                core_lhs.oriented_homspace(),
                core_rhs.oriented_homspace(),
                core_axes,
            )?;
            let Some(validated) = preflight.validate_core_geometry()? else {
                return Ok(None);
            };
            let plan = try_compile_oriented_canonical_core_plan(
                &validated,
                dst,
                core_lhs.storage_space(),
                core_rhs.storage_space(),
            )?;
            Ok(plan.map(|plan| Resolution::core(plan, orientation)))
        },
    )
}

impl<C> Resolution<C> {
    fn core(plan: FusionBlockContractPlan<C>, orientation: FusionContractOrientation) -> Self {
        match orientation {
            FusionContractOrientation::LhsRhs => Self::Core(Arc::new(plan)),
            FusionContractOrientation::RhsLhs => Self::SwappedCore(Arc::new(plan)),
        }
    }
}

/// Runs `compile` on the TensorKit `contract!` candidates
/// (`_contract_candidates`: pairs sorted by lhs or by rhs keys, times A·B or
/// B·A) that need no source or output materialization, in TensorKit's tie
/// order m1..m4, and returns the first compiled route.
///
/// A candidate qualifies when, over the *oriented* operand HomSpaces, its
/// core-left operand contracts exactly its domain and its core-right operand
/// exactly its codomain, both in order, and its output is the identity: the
/// core GEMM then reads both parent buffers, a lazy adjoint as a GEMM
/// adjoint flag. This is TensorKit `_contract_memcost == 0` with
/// `has_shared_permute(::AdjointTensorMap)` (indexmanipulations.jl L282)
/// delegating to the parent: a lazy adjoint whose permuted index order is its
/// natural adjoint order is free. Unless `twist_consumable`, a candidate
/// whose core-right contracted legs carry the fermionic supertrace twist is
/// excluded, since `blas_contract!` must then copy one operand to twist it.
///
/// Why before the DynamicTree scorer, not inside it: DynamicTree cannot
/// borrow a conjugated source (`source_layout_permutation_is_borrowable`),
/// so a zero-cost score there would select a candidate that materializes.
/// Every qualifying candidate has cost zero, the global minimum, so taking
/// the first in tie order selects what `select_best_scored_contract_candidate`
/// would under TensorKit's cost; when none compiles, the DynamicTree scorer
/// runs unchanged. The checks here are axis and dual-flag comparisons only;
/// `compile` runs the categorical preflight for a qualifying candidate.
pub(crate) fn try_zero_copy_contract_candidates<'a, R, T>(
    rule: &R,
    dst_nout: usize,
    lhs: FusionOperand<'a>,
    rhs: FusionOperand<'a>,
    axes: TensorContractSpec<'_>,
    twist_consumable: bool,
    mut compile: impl FnMut(
        FusionOperand<'a>,
        FusionOperand<'a>,
        TensorContractSpec<'_>,
        FusionContractOrientation,
    ) -> Result<Option<T>, OperationError>,
) -> Result<Option<T>, OperationError>
where
    R: MultiplicityFreeRigidSymbols,
{
    let (lhs_axes, rhs_axes) = (axes.lhs_contracting_axes(), axes.rhs_contracting_axes());
    let contracted = lhs_axes.len();
    let (lhs_rank, rhs_rank) = (lhs.storage_space().rank(), rhs.storage_space().rank());
    if contracted != rhs_axes.len() || contracted > lhs_rank || contracted > rhs_rank {
        return Ok(None);
    }
    let lhs_open = lhs_rank - contracted;
    let open = lhs_open + rhs_rank - contracted;
    let output_is = |expected: &mut dyn Iterator<Item = usize>| match axes.output_permutation() {
        OutputAxisOrder::Identity => (0..open).eq(expected),
        OutputAxisOrder::Axes(output) => output.iter().copied().eq(expected),
    };
    // pAB′: B·A writes (rhs open | lhs open).
    let output_ok = [
        output_is(&mut (0..open)),
        output_is(&mut (lhs_open..open).chain(0..lhs_open)),
    ];
    if !output_ok.contains(&true) {
        return Ok(None);
    }
    let fermionic = rule.braiding_style() == tenet_core::BraidingStyleKind::Fermionic;
    let candidates = contracted_axis_order_candidates(lhs_axes, rhs_axes);
    for (orientation, output_ok) in CACHED_ORIENTATIONS.into_iter().zip(output_ok) {
        if !output_ok {
            continue;
        }
        for candidate in &candidates {
            let (core_lhs, core_rhs, core_lhs_axes, core_rhs_axes) = match orientation {
                FusionContractOrientation::LhsRhs => (lhs, rhs, candidate.lhs(), candidate.rhs()),
                FusionContractOrientation::RhsLhs => (rhs, lhs, candidate.rhs(), candidate.lhs()),
            };
            let (left, right) = (core_lhs.oriented_homspace(), core_rhs.oriented_homspace());
            if dst_nout != left.nout()
                || !core_lhs_axes.iter().copied().eq(left.nout()..left.rank())
                || !core_rhs_axes.iter().copied().eq(0..right.nout())
            {
                continue;
            }
            if fermionic
                && !twist_consumable
                && core_rhs_axes
                    .iter()
                    .any(|&axis| right.external_axis_is_dual(axis) == Some(true))
            {
                continue;
            }
            let core_axes = TensorContractSpec::new_with_conjugation(
                core_lhs_axes,
                core_rhs_axes,
                OutputAxisOrder::identity(),
                core_lhs.storage_conjugate(),
                core_rhs.storage_conjugate(),
            );
            if let Some(route) = compile(core_lhs, core_rhs, core_axes, orientation)? {
                return Ok(Some(route));
            }
        }
    }
    Ok(None)
}

/// Compiles the storage-only route with one categorical preflight. Ordinary
/// host resolution keeps its existing profiled path.
pub(crate) fn compile_storage_resolution<R>(
    rule: &R,
    dst: &DynamicFusionMapSpace,
    lhs: &DynamicFusionMapSpace,
    rhs: &DynamicFusionMapSpace,
    axes: TensorContractSpec<'_>,
    compile_structure: impl FnOnce() -> Result<
        Option<Arc<TensorContractStructure<R::Scalar>>>,
        OperationError,
    >,
    compile_dynamic: impl FnOnce() -> Result<Arc<FusionContractPlan>, OperationError>,
) -> Result<Resolution<R::Scalar>, OperationError>
where
    R: MultiplicityFreeRigidSymbols,
    R::Scalar: DenseBlockScalar,
{
    let preflight = CoreContractPreflight::compile(rule, dst, lhs, rhs, axes)?;
    if !preflight.has_conjugation() {
        if let Some(validated) = preflight.validate_core_geometry()? {
            if validated_rhs_contract_requires_twist(&validated)? {
                if let Some(plan) = try_compile_scaled_storage_contract_plan(
                    rule,
                    &validated,
                    dst,
                    lhs,
                    rhs,
                    FusionTreePairOrientation::Direct,
                    NonuniformTwist::Reject,
                )? {
                    return Ok(Resolution::Core(Arc::new(plan)));
                }
            } else {
                return compile_fusion_block_contract_plan_validated(validated, dst, lhs, rhs)
                    .map(Arc::new)
                    .map(Resolution::Core);
            }
        }
        return Ok(Resolution::DynamicTree(compile_dynamic()?));
    }
    if let Some(structure) = compile_structure()? {
        return Ok(Resolution::Structure(structure));
    }
    Ok(Resolution::DynamicTree(compile_dynamic()?))
}

/// What the storage-direct Core route does with a fermionic twist that is not
/// uniform within one RHS coupled-sector matrix, which no per-job GEMM alpha
/// can express.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum NonuniformTwist {
    /// Report `UnsupportedTensorContractScope`: the Host storage-direct
    /// entries, which keep their existing error.
    Reject,
    /// Decline the Core route (`Ok(None)`), so the caller compiles the
    /// `DynamicTree` artifact, which applies the twist per block: the device
    /// contraction.
    Decline,
}

fn try_compile_scaled_storage_contract_plan<R>(
    rule: &R,
    validated: &ValidatedCoreContract<'_, R>,
    dst: &DynamicFusionMapSpace,
    lhs: &DynamicFusionMapSpace,
    rhs: &DynamicFusionMapSpace,
    rhs_orientation: FusionTreePairOrientation,
    nonuniform: NonuniformTwist,
) -> Result<Option<FusionBlockContractPlan<R::Scalar>>, OperationError>
where
    R: MultiplicityFreeRigidSymbols,
    R::Scalar: DenseBlockScalar,
{
    let Some(regions) = rhs
        .structure()
        .coupled_sector_regions(rhs.nout())
        .map_err(OperationError::from_core_preserving_context)?
    else {
        return Ok(None);
    };
    let mut alpha_by_coupled = Vec::with_capacity(regions.len());
    for region in regions.iter() {
        let mut row_trees = match rhs_orientation {
            FusionTreePairOrientation::Direct => region.row_trees(),
            FusionTreePairOrientation::Adjoint => region.col_trees(),
        }
        .iter();
        let Some(first) = row_trees.next() else {
            return Err(OperationError::UnsupportedTensorContractScope {
                message: "canonical RHS coupled region has no row fusion tree",
            });
        };
        let alpha = rhs_contract_twist_factor_oriented(
            rule,
            validated.rhs_homspace(),
            validated.rhs_contracting_axes(),
            first.tree(),
        )?;
        for extent in row_trees {
            if rhs_contract_twist_factor_oriented(
                rule,
                validated.rhs_homspace(),
                validated.rhs_contracting_axes(),
                extent.tree(),
            )? != alpha
            {
                if nonuniform == NonuniformTwist::Decline {
                    return Ok(None);
                }
                return Err(OperationError::UnsupportedTensorContractScope {
                    message: "fermionic twist is nonuniform within one RHS coupled-sector matrix",
                });
            }
        }
        alpha_by_coupled.push((region.coupled(), alpha));
    }
    try_compile_scaled_canonical_core_plan(validated, dst, lhs, rhs, &alpha_by_coupled)
}

/// Compiles the canonical storage contraction directly from parent spaces and
/// lazy operand orientation, before any logical-key projection is prepared.
pub(crate) fn try_compile_oriented_storage_contract_plan<R>(
    rule: &R,
    dst: &DynamicFusionMapSpace,
    lhs: FusionOperand<'_>,
    rhs: FusionOperand<'_>,
    axes: TensorContractSpec<'_>,
    nonuniform: NonuniformTwist,
) -> Result<Option<Arc<FusionBlockContractPlan<R::Scalar>>>, OperationError>
where
    R: MultiplicityFreeRigidSymbols,
    R::Scalar: DenseBlockScalar,
{
    let preflight = CoreContractPreflight::compile_oriented(
        rule,
        dst.homspace(),
        lhs.oriented_homspace(),
        rhs.oriented_homspace(),
        axes,
    )?;
    let Some(validated) = preflight.validate_core_geometry()? else {
        return Ok(None);
    };
    let plan = if validated_rhs_contract_requires_twist(&validated)? {
        try_compile_scaled_storage_contract_plan(
            rule,
            &validated,
            dst,
            lhs.storage_space(),
            rhs.storage_space(),
            rhs.orientation(),
            nonuniform,
        )?
    } else {
        try_compile_oriented_canonical_core_plan(
            &validated,
            dst,
            lhs.storage_space(),
            rhs.storage_space(),
        )?
    };
    Ok(plan.map(Arc::new))
}

/// [`try_compile_oriented_storage_contract_plan`] over every zero-copy
/// TensorKit candidate ([`try_zero_copy_contract_candidates`]); a uniform
/// twist is consumed as per-job alpha, a nonuniform one declines.
pub(crate) fn try_compile_oriented_storage_contract_candidate_plan<R>(
    rule: &R,
    dst: &DynamicFusionMapSpace,
    lhs: FusionOperand<'_>,
    rhs: FusionOperand<'_>,
    axes: TensorContractSpec<'_>,
) -> Result<Option<StorageContractRoute<R::Scalar>>, OperationError>
where
    R: MultiplicityFreeRigidSymbols,
    R::Scalar: DenseBlockScalar,
{
    try_zero_copy_contract_candidates(
        rule,
        dst.nout(),
        lhs,
        rhs,
        axes,
        true,
        |core_lhs, core_rhs, core_axes, orientation| {
            let plan = try_compile_oriented_storage_contract_plan(
                rule,
                dst,
                core_lhs,
                core_rhs,
                core_axes,
                NonuniformTwist::Decline,
            )?;
            Ok(plan.map(|plan| match orientation {
                FusionContractOrientation::LhsRhs => StorageContractRoute::Core(plan),
                FusionContractOrientation::RhsLhs => StorageContractRoute::SwappedCore(plan),
            }))
        },
    )
}

/// Twist-free counterpart used only by tensor-map composition.
pub(crate) fn try_compile_oriented_storage_composition_plan<R>(
    rule: &R,
    dst: &DynamicFusionMapSpace,
    lhs: FusionOperand<'_>,
    rhs: FusionOperand<'_>,
    axes: TensorContractSpec<'_>,
) -> Result<Option<Arc<FusionBlockContractPlan<R::Scalar>>>, OperationError>
where
    R: MultiplicityFreeRigidSymbols,
    R::Scalar: DenseBlockScalar,
{
    let preflight = CoreContractPreflight::compile_oriented(
        rule,
        dst.homspace(),
        lhs.oriented_homspace(),
        rhs.oriented_homspace(),
        axes,
    )?;
    let Some(validated) = preflight.validate_core_geometry()? else {
        return Ok(None);
    };
    try_compile_oriented_canonical_core_plan(
        &validated,
        dst,
        lhs.storage_space(),
        rhs.storage_space(),
    )
    .map(|plan| plan.map(Arc::new))
}

/// Compiles the coupled block plan for already-materialized core operands.
pub(crate) fn compile_core_plan<R>(
    rule: &R,
    dst: &DynamicFusionMapSpace,
    lhs: &DynamicFusionMapSpace,
    rhs: &DynamicFusionMapSpace,
    axes: TensorContractSpec<'_>,
) -> Result<Arc<FusionBlockContractPlan<R::Scalar>>, OperationError>
where
    R: MultiplicityFreeRigidSymbols,
    R::Scalar: DenseBlockScalar,
{
    let preflight = CoreContractPreflight::compile(rule, dst, lhs, rhs, axes)?;
    super::fusion::reject_fusion_contract_conjugation(axes)?;
    let validated = preflight.require_core_geometry()?;
    compile_fusion_block_contract_plan_validated(validated, dst, lhs, rhs).map(Arc::new)
}

/// Compiles the coupled block plan of the core operands a
/// [`FusionContractPlan`] derived from an already validated contraction.
///
/// Why no second [`CoreContractPreflight`]: the plan's source transforms
/// place each operand's open and contracted legs in core order and its core
/// axes are the canonical `(open, contracted) x (contracted, open)` pairing,
/// so the core source and output form hold by construction. The core
/// destination is either derived from these two operands with those core
/// axes or, for an identity output transform, the destination whose
/// contracted HomSpace the plan compile already checked. The contracted leg
/// pairs are the checked pairs in candidate order, and every derived space
/// is bound to `rule`. Re-running the preflight would only repeat those
/// checks on every warm call; `compile_core_plan` keeps them for operands
/// that did not come from such a plan.
pub(crate) fn compile_derived_core_plan<R>(
    rule: &R,
    dst: &DynamicFusionMapSpace,
    lhs: &DynamicFusionMapSpace,
    rhs: &DynamicFusionMapSpace,
    core_axes: TensorContractSpec<'_>,
) -> Result<Arc<FusionBlockContractPlan<R::Scalar>>, OperationError>
where
    R: MultiplicityFreeRigidSymbols,
    R::Scalar: DenseBlockScalar,
{
    // Debug builds re-check the construction argument above, so an internal
    // inconsistency (a wrong space-cache entry, diverging HomSpace
    // derivations) still fails loudly in tests instead of reaching the block
    // plan.
    #[cfg(debug_assertions)]
    super::fusion_block::debug_assert_core_geometry(rule, dst, lhs, rhs, core_axes);
    #[cfg(not(debug_assertions))]
    let _ = core_axes;
    compile_fusion_block_contract_plan_core_geometry(rule, dst, lhs, rhs).map(Arc::new)
}

/// Compiles TensorKit `mul!` composition without inserting a fermionic
/// supertrace twist.
#[allow(clippy::too_many_arguments)]
pub(crate) fn compile_composition_plan<R>(
    rule: &R,
    dst: &DynamicFusionMapSpace,
    lhs: &FusionOperandLayout<'_>,
    rhs: &FusionOperandLayout<'_>,
    axes: TensorContractSpec<'_>,
) -> Result<Arc<FusionBlockContractPlan<R::Scalar>>, OperationError>
where
    R: MultiplicityFreeRigidSymbols,
    R::Scalar: DenseBlockScalar,
{
    let validated = CoreContractPreflight::compile_oriented(
        rule,
        dst.homspace(),
        lhs.oriented_homspace(),
        rhs.oriented_homspace(),
        axes,
    )?
    .require_core_geometry()?;
    compile_fusion_block_contract_plan_prelowered_validated(validated, dst, lhs, rhs).map(Arc::new)
}

/// True when the fermionic supertrace twist can be nontrivial: such
/// contractions take the dynamic route, where the twist is applied during
/// rhs materialization; the core direct-GEMM route stays
/// coefficient-free (TensorKit mul! parity).
pub(crate) fn rhs_contract_requires_twist<R>(
    rule: &R,
    rhs: &DynamicFusionMapSpace,
    axes: TensorContractSpec<'_>,
) -> Result<bool, OperationError>
where
    R: MultiplicityFreeRigidSymbols,
{
    rhs_contract_homspace_requires_twist(rule, rhs.homspace(), axes)
}

fn validated_rhs_contract_requires_twist<R>(
    validated: &ValidatedCoreContract<'_, R>,
) -> Result<bool, OperationError>
where
    R: MultiplicityFreeRigidSymbols,
{
    if validated.rule().braiding_style() != tenet_core::BraidingStyleKind::Fermionic {
        return Ok(false);
    }
    Ok(validated
        .rhs_contracting_axes()
        .iter()
        .copied()
        .any(|axis| {
            validated
                .rhs_homspace()
                .external_axis_is_dual(axis)
                .expect("core preflight validated every rhs contraction axis")
        }))
}

pub(crate) fn rhs_contract_homspace_requires_twist<R>(
    rule: &R,
    rhs: &FusionTreeHomSpace,
    axes: TensorContractSpec<'_>,
) -> Result<bool, OperationError>
where
    R: MultiplicityFreeRigidSymbols,
{
    rhs_contract_axes_require_twist(rule, rhs, axes.rhs_contracting_axes())
}

/// True when these contracting axes put a contraction in the direct core-GEMM
/// source form — `lhs`'s whole domain paired in order with `rhs`'s whole
/// codomain — with no fermionic supertrace twist on `rhs`. In that form the
/// identity output order resolves to [`Resolution::Core`]; any other output
/// order resolves to the dynamic tree route.
#[doc(hidden)]
pub fn contraction_sources_are_untwisted_core_form<R>(
    rule: &R,
    lhs: &FusionTreeHomSpace,
    rhs: &FusionTreeHomSpace,
    lhs_contracting_axes: &[usize],
    rhs_contracting_axes: &[usize],
) -> bool
where
    R: MultiplicityFreeRigidSymbols,
{
    lhs_contracting_axes.len() == rhs_contracting_axes.len()
        && is_core_form_fusion_source_contract(lhs, rhs, lhs_contracting_axes, rhs_contracting_axes)
        && matches!(
            rhs_contract_axes_require_twist(rule, rhs, rhs_contracting_axes),
            Ok(false)
        )
}

fn rhs_contract_axes_require_twist<R>(
    rule: &R,
    rhs: &FusionTreeHomSpace,
    rhs_contracting_axes: &[usize],
) -> Result<bool, OperationError>
where
    R: MultiplicityFreeRigidSymbols,
{
    if rule.braiding_style() != tenet_core::BraidingStyleKind::Fermionic {
        return Ok(false);
    }
    for &axis in rhs_contracting_axes {
        if external_axis_is_dual(rhs, axis)? {
            return Ok(true);
        }
    }
    Ok(false)
}

#[cfg(test)]
mod tests {
    use super::*;
    use tenet_core::{
        FermionParityFusionRule, FusionProductSpace, FusionTreeHomSpace, ProductFusionRuleExt,
        SU2FusionRule, SU2Irrep, SectorId, SectorLeg, U1FusionRule, U1Irrep,
    };

    fn single_sector_matrix_space<R>(
        rule: &R,
        sector: SectorId,
        codomain_dual: bool,
        domain_dual: bool,
    ) -> DynamicFusionMapSpace
    where
        R: MultiplicityFreeRigidSymbols<Scalar = f64>,
    {
        let hom = FusionTreeHomSpace::new(
            FusionProductSpace::new([SectorLeg::new([(sector, 1)], codomain_dual)]),
            FusionProductSpace::new([SectorLeg::new([(sector, 1)], domain_dual)]),
        );
        let count = hom.fusion_tree_keys(rule).len();
        DynamicFusionMapSpace::from_degeneracy_shapes(rule, hom, vec![vec![1, 1]; count]).unwrap()
    }

    #[test]
    fn rhs_twist_requirement_uses_external_domain_dual_and_product_parity() {
        let fermion = FermionParityFusionRule;
        let odd = SectorId::new(1);
        let cases = [
            (false, false, 0, false),
            (true, false, 0, true),
            (false, false, 1, true),
            (false, true, 1, false),
        ];
        for (codomain_dual, domain_dual, rhs_axis, expected) in cases {
            let rhs = single_sector_matrix_space(&fermion, odd, codomain_dual, domain_dual);
            let rhs_axes = [rhs_axis];
            let axes = TensorContractSpec::with_default_output_order(&[0], &rhs_axes);
            // What: codomain uses its stored dual flag, while domain external
            // duality is the inverse of its stored flag.
            assert_eq!(
                rhs_contract_requires_twist(&fermion, &rhs, axes).unwrap(),
                expected
            );
        }

        let fp_u1 = FermionParityFusionRule.product(U1FusionRule);
        let odd_charge = fp_u1.encode_sector(odd, U1Irrep::new(0).sector_id());
        let product = fp_u1.product(SU2FusionRule);
        let odd_product =
            product.encode_sector(odd_charge, SU2Irrep::from_twice_spin(0).sector_id());
        let rhs = single_sector_matrix_space(&product, odd_product, true, false);
        // What: a bosonic U(1) x SU(2) component does not erase the odd fZ2
        // twist on an externally dual product-sector axis.
        assert!(rhs_contract_requires_twist(
            &product,
            &rhs,
            TensorContractSpec::with_default_output_order(&[0], &[0]),
        )
        .unwrap());
    }

    #[test]
    fn core_resolution_derives_geometry_once() {
        let rule = U1FusionRule;
        let zero = U1Irrep::new(0).sector_id();
        let lhs = single_sector_matrix_space(&rule, zero, false, false);
        let rhs = single_sector_matrix_space(&rule, zero, false, false);
        let dst = single_sector_matrix_space(&rule, zero, false, false);
        super::super::fusion_block::reset_core_contract_derivations();

        let resolution = compile_resolution(
            &rule,
            &dst,
            &lhs,
            &rhs,
            TensorContractSpec::with_default_output_order(&[1], &[0]),
            || panic!("core contraction must not compile a dense structure"),
            || panic!("core contraction must not compile tree transforms"),
        )
        .unwrap();

        // What: one core compiler invocation derives each geometry authority once.
        assert!(matches!(resolution, Resolution::Core(_)));
        assert_eq!(
            super::super::fusion_block::core_contract_derivations(),
            (1, 1)
        );
    }

    #[test]
    fn storage_core_resolution_derives_geometry_once() {
        let rule = U1FusionRule;
        let zero = U1Irrep::new(0).sector_id();
        let lhs = single_sector_matrix_space(&rule, zero, false, false);
        let rhs = single_sector_matrix_space(&rule, zero, false, false);
        let dst = single_sector_matrix_space(&rule, zero, false, false);
        super::super::fusion_block::reset_core_contract_derivations();

        let resolution = compile_storage_resolution(
            &rule,
            &dst,
            &lhs,
            &rhs,
            TensorContractSpec::with_default_output_order(&[1], &[0]),
            || panic!("core storage contraction must not compile a dense structure"),
            || panic!("core storage contraction must not compile tree transforms"),
        )
        .unwrap();

        assert!(matches!(resolution, Resolution::Core(_)));
        assert_eq!(
            super::super::fusion_block::core_contract_derivations(),
            (1, 1)
        );
    }

    mod warm_compile_counts {
        use std::sync::Arc;

        use tenet_core::{
            FusionProductSpace, FusionTreeHomSpace, SectorLeg, U1FusionRule, U1Irrep,
        };
        use tenet_operations::{OutputAxisOrder, TensorContractSpec};

        use crate::contract::fusion_block::{
            contract_compile_counts, reset_core_contract_derivations, ContractCompileCounts,
        };
        use crate::contract::{BoundDynamicFusionMapSpace, FusionOperand};
        use crate::{
            ContractDestinationInit, OperationCachePolicy, RuleIdentity,
            TensorContractFusionExecutionContext,
        };

        type Space = BoundDynamicFusionMapSpace<U1FusionRule>;

        fn space(provider: &Arc<U1FusionRule>, codomain: usize, domain: usize) -> Space {
            let leg = || {
                SectorLeg::new(
                    [
                        (U1Irrep::new(-1).sector_id(), 2),
                        (U1Irrep::new(0).sector_id(), 3),
                        (U1Irrep::new(1).sector_id(), 1),
                    ],
                    false,
                )
            };
            Space::from_final_homspace_multiplicity_free_checked(
                Arc::clone(provider),
                FusionTreeHomSpace::new(
                    FusionProductSpace::new((0..codomain).map(|_| leg())),
                    FusionProductSpace::new((0..domain).map(|_| leg())),
                ),
            )
            .unwrap()
        }

        /// Counts of the third of three identical calls.
        fn warm_counts(mut call: impl FnMut()) -> ContractCompileCounts {
            call();
            call();
            reset_core_contract_derivations();
            call();
            contract_compile_counts()
        }

        #[test]
        fn warm_contract_compile_runs_one_preflight_and_rebuilds_no_homspace() {
            let provider = Arc::new(U1FusionRule);
            let lhs = space(&provider, 2, 1);
            let matrix = space(&provider, 1, 1);
            let mut context = TensorContractFusionExecutionContext::<f64, RuleIdentity>::default();
            context.set_cache_policy(OperationCachePolicy::NoCache);
            let lhs_data = vec![1.0; lhs.space().required_len().unwrap()];
            let matrix_data = vec![0.5; matrix.space().required_len().unwrap()];
            for output in [[0usize, 1, 2], [1, 0, 2]] {
                let axes = TensorContractSpec::new(&[0], &[1], OutputAxisOrder::from_axes(&output));
                let dst = Space::contracted_multiplicity_free_ordered(
                    &lhs,
                    &matrix,
                    &[0],
                    &[1],
                    OutputAxisOrder::from_axes(&output),
                )
                .unwrap();
                let mut dst_data = vec![0.0; dst.space().required_len().unwrap()];
                let host = warm_counts(|| {
                    context
                        .tensorcontract_fusion_dyn_into_with_init(
                            &dst,
                            &mut dst_data,
                            &lhs,
                            &lhs_data,
                            &matrix,
                            &matrix_data,
                            axes,
                            1.0,
                            ContractDestinationInit::Axpby(0.0),
                        )
                        .unwrap();
                });
                let mut core_dst = false;
                let storage = warm_counts(|| {
                    let resolution = context
                        .compile_storage_contract_resolution(
                            &dst,
                            FusionOperand::direct(lhs.space()),
                            FusionOperand::direct(matrix.space()),
                            axes,
                        )
                        .unwrap();
                    assert!(resolution.is_dynamic_tree());
                    core_dst = resolution.direct_destination_inactive_blocks().is_none();
                });
                // What: at most one core preflight per call (the Host
                // route's; the device route rejects every candidate on axes
                // alone, before any preflight), no core
                // destination check for a non-core request, and only the
                // HomSpaces TensorKit also forms: one permuted HomSpace per
                // transformed source plus the core destination when the
                // output transform is not the identity.
                // The fixture covers both output forms.
                assert_eq!(core_dst, output != [0, 1, 2]);
                let core_dst = usize::from(core_dst);
                let expected = ContractCompileCounts {
                    preflights: 1,
                    core_destination_checks: 0,
                    derived_homspace_builds: 2 + core_dst,
                };
                assert_eq!(host, expected, "host output order {output:?}");
                assert_eq!(
                    storage,
                    ContractCompileCounts {
                        preflights: 0,
                        ..expected
                    },
                    "storage output order {output:?}"
                );
            }
        }

        #[test]
        fn warm_core_compile_checks_the_destination_once_without_building_it() {
            let provider = Arc::new(U1FusionRule);
            let lhs = space(&provider, 2, 2);
            let square = space(&provider, 2, 2);
            let dst = Space::contracted_multiplicity_free_ordered(
                &lhs,
                &square,
                &[2, 3],
                &[0, 1],
                OutputAxisOrder::identity(),
            )
            .unwrap();
            let axes = TensorContractSpec::with_default_output_order(&[2, 3], &[0, 1]);
            let mut context = TensorContractFusionExecutionContext::<f64, RuleIdentity>::default();
            context.set_cache_policy(OperationCachePolicy::NoCache);
            let lhs_data = vec![1.0; lhs.space().required_len().unwrap()];
            let square_data = vec![0.5; square.space().required_len().unwrap()];
            let mut dst_data = vec![0.0; dst.space().required_len().unwrap()];
            let counts = warm_counts(|| {
                context
                    .tensorcontract_fusion_dyn_into_with_init(
                        &dst,
                        &mut dst_data,
                        &lhs,
                        &lhs_data,
                        &square,
                        &square_data,
                        axes,
                        1.0,
                        ContractDestinationInit::Zeroed,
                    )
                    .unwrap();
            });
            assert_eq!(
                counts,
                ContractCompileCounts {
                    preflights: 1,
                    core_destination_checks: 1,
                    derived_homspace_builds: 0,
                }
            );
        }
    }
}
