//! The one trace compile pipeline (#1856): a shared preflight (pair
//! duality, selected HomSpace) and a shared term builder (split, trace
//! channel match, channel factor, destination lookup), with the admission
//! mode supplying only its symbol access and errors.
//!
//! TensorKit `trace_permute!` / `_trace_permute!`
//! (`src/tensors/tensoroperations.jl`): check the pair duality and the
//! selected space, then per source tree pair permute to `(p…, q…)`, split at
//! the open ranks, keep `g₁ == g₂`, and scale by
//! `dim(coupled)/dim(first) · Π twist(non-dual)`.
//!
//! Why hook traits rather than one generic body: the multiplicity-free and
//! checked Generic modes keep different symbol access and error types; the
//! hooks are the narrowest layer that owns those differences. Both modes
//! detect the same misuses in TensorKit's order (#1872): braiding, axes,
//! selection, destination space (when there is one), pair duality.

use super::*;
use crate::tree_transform::{PendingCoefficientGroups, TraceColumnReuse, TransformerMode};
use crate::TreeTransformOperation;
use tenet_core::{
    generic_permute_tree_pair_block_indexed_checked, BlockSourceColumns, BraidingStyleKind,
    CheckedGenericAdmissionMode, MultiplicityFreeAdmissionMode, RuleIdentity,
};

/// What the trace preflight asks of an admission mode.
pub(super) trait TracePreflightMode<R> {
    type Error;

    fn operation(error: OperationError) -> Self::Error;

    fn validate_source(src: &BoundDynamicFusionMapSpace<R>) -> Result<(), Self::Error>;

    fn select(
        rule: &R,
        src: OrientedFusionTreeHomSpace<'_>,
        codomain_axes: &[usize],
        domain_axes: &[usize],
    ) -> Result<FusionTreeHomSpace, Self::Error>;

    fn pair_is_dual(
        rule: &R,
        src: OrientedFusionTreeHomSpace<'_>,
        lhs_axis: usize,
        rhs_axis: usize,
    ) -> Result<bool, Self::Error>;

    fn braiding_style(rule: &R) -> BraidingStyleKind;
}

/// The one trace preflight, in TensorKit `trace_permute!`'s order: the
/// braiding gate, the axes, the selected output HomSpace, then the pair
/// duality, which the caller raises through
/// [`TracePreflight::require_dual_pairs`] once any destination space has been
/// compared.
pub(super) fn trace_preflight<M, R>(
    src: &BoundDynamicFusionMapSpace<R>,
    axes: TensorTraceAxisSpec<'_>,
    dst_nout: usize,
) -> Result<TracePreflight, M::Error>
where
    M: TracePreflightMode<R>,
{
    M::validate_source(src)?;
    // TensorKit `trace_permute!` gates the braiding first: a trace is defined
    // only for symmetric braiding.
    crate::admission::require_symmetric_braiding(
        M::braiding_style(src.provider()),
        crate::admission::SymmetricBraidingOp::Trace,
    )
    .map_err(M::operation)?;
    let orientation = if axes.source_conjugate() {
        FusionTreePairOrientation::Adjoint
    } else {
        FusionTreePairOrientation::Direct
    };
    let lowered_axes =
        lower_tensortrace_source_adjoint_axes_dyn(src.space().nout(), src.space().nin(), axes)
            .map_err(M::operation)?;
    let lowered_spec = lowered_axes.as_spec();
    let axis_plan = TensorTraceAxisPlan::compile(
        src.space().rank(),
        lowered_spec.output_axes().len(),
        lowered_spec,
    )
    .map_err(M::operation)?;
    trace_geometry::<M, R>(
        src.provider(),
        OrientedFusionTreeHomSpace::new(src.space().homspace(), orientation),
        &axis_plan,
        dst_nout,
    )
}

/// The preflight's geometry half, for a compile that already holds its axis
/// plan.
pub(super) fn trace_geometry<M, R>(
    rule: &R,
    src_homspace: OrientedFusionTreeHomSpace<'_>,
    axis_plan: &TensorTraceAxisPlan,
    dst_nout: usize,
) -> Result<TracePreflight, M::Error>
where
    M: TracePreflightMode<R>,
{
    if dst_nout > axis_plan.output_axes.len() {
        return Err(M::operation(OperationError::RankMismatch {
            expected: axis_plan.output_axes.len(),
            actual: dst_nout,
        }));
    }
    let selected_homspace = M::select(
        rule,
        src_homspace,
        &axis_plan.output_axes[..dst_nout],
        &axis_plan.output_axes[dst_nout..],
    )?;
    let mut trace_pairs_match = true;
    for (&lhs_axis, &rhs_axis) in axis_plan
        .trace_lhs_axes
        .iter()
        .zip(axis_plan.trace_rhs_axes.iter())
    {
        trace_pairs_match &= M::pair_is_dual(rule, src_homspace, lhs_axis, rhs_axis)?;
    }
    Ok(TracePreflight {
        selected_homspace,
        trace_pairs_match,
    })
}

impl<R> TracePreflightMode<R> for MultiplicityFreeAdmissionMode
where
    R: FusionRule + CheckedFusionAlgebra,
{
    type Error = OperationError;

    #[inline]
    fn operation(error: OperationError) -> Self::Error {
        error
    }

    fn validate_source(src: &BoundDynamicFusionMapSpace<R>) -> Result<(), Self::Error> {
        src.space().validate_rule(src.provider())?;
        Ok(())
    }

    fn select(
        rule: &R,
        src: OrientedFusionTreeHomSpace<'_>,
        codomain_axes: &[usize],
        domain_axes: &[usize],
    ) -> Result<FusionTreeHomSpace, Self::Error> {
        // Why not widen the public rule bound: custom encoded rules retain their
        // established infallible contract while lowered built-ins close overflow.
        record_trace_selected_homspace_derivation();
        src.try_select_checked(rule, codomain_axes, domain_axes)
            .map_err(|error| match error {
                CheckedFusionSpaceError::FusionAlgebra(error) => {
                    OperationError::FusionAlgebra(error)
                }
                CheckedFusionSpaceError::Core(error) => OperationError::Core(*error),
                _ => OperationError::InvalidArgument {
                    message: "checked trace metadata error",
                },
            })
    }

    fn pair_is_dual(
        rule: &R,
        src: OrientedFusionTreeHomSpace<'_>,
        lhs_axis: usize,
        rhs_axis: usize,
    ) -> Result<bool, Self::Error> {
        let lhs = outward_axis_leg_checked(rule, src, lhs_axis)?;
        let rhs = outward_axis_leg_checked(rule, src, rhs_axis)?;
        let rhs_dual = rhs
            .try_dual(rule)
            .map_err(|error| OperationError::FusionAlgebra(Box::new(error)))?;
        lhs.try_dual(rule)
            .map_err(|error| OperationError::FusionAlgebra(Box::new(error)))?;
        Ok(lhs == rhs_dual)
    }

    fn braiding_style(rule: &R) -> BraidingStyleKind {
        FusionRule::braiding_style(rule)
    }
}

impl<R> TracePreflightMode<R> for CheckedGenericAdmissionMode
where
    R: CheckedGenericFusion,
{
    type Error = CheckedGenericPlanError<R::Error>;

    #[inline]
    fn operation(error: OperationError) -> Self::Error {
        CheckedGenericPlanError::Operation(error)
    }

    fn validate_source(_src: &BoundDynamicFusionMapSpace<R>) -> Result<(), Self::Error> {
        Ok(())
    }

    fn select(
        rule: &R,
        src: OrientedFusionTreeHomSpace<'_>,
        codomain_axes: &[usize],
        domain_axes: &[usize],
    ) -> Result<FusionTreeHomSpace, Self::Error> {
        src.try_select_generic_checked(rule, codomain_axes, domain_axes)
            .map_err(CheckedGenericPlanError::from)
    }

    fn pair_is_dual(
        rule: &R,
        src: OrientedFusionTreeHomSpace<'_>,
        lhs_axis: usize,
        rhs_axis: usize,
    ) -> Result<bool, Self::Error> {
        let lhs = src
            .try_external_axis_leg_generic(rule, lhs_axis)
            .map_err(CheckedGenericPlanError::from)?
            .ok_or_else(|| {
                CheckedGenericPlanError::Operation(OperationError::InvalidArgument {
                    message: "trace lhs axis has no external leg",
                })
            })?;
        let rhs = src
            .try_external_axis_leg_generic(rule, rhs_axis)
            .map_err(CheckedGenericPlanError::from)?
            .ok_or_else(|| {
                CheckedGenericPlanError::Operation(OperationError::InvalidArgument {
                    message: "trace rhs axis has no external leg",
                })
            })?;
        // Why the whole leg: TensorKit `trace_permute!` requires
        // `space(tsrc, q₁) == dual(space(tsrc, q₂))`; comparing only the first
        // sector admitted non-dual pairs and summed only the sectors they share.
        let rhs_dual = rhs
            .try_dual_generic(rule)
            .map_err(CheckedGenericPlanError::from)?;
        Ok(lhs == rhs_dual)
    }

    fn braiding_style(rule: &R) -> BraidingStyleKind {
        CheckedGenericFusion::braiding_style(rule)
    }
}

/// What the trace term builder asks of an admission mode.
pub(super) trait TraceTermMode<R> {
    type Scalar: Clone + Mul<Output = Self::Scalar> + Send + Sync + 'static;
    type Error;

    /// The composer and admission family a cache-4 trace entry is keyed by.
    const MODE: TransformerMode;

    fn rule_identity(rule: &R) -> RuleIdentity;

    fn operation(error: OperationError) -> Self::Error;

    /// Hands every source block's permuted tree-pair rows, in source block
    /// order, to [`TraceLowering::lower`]; returns the permutation groups it
    /// built, for the caller to publish once its whole call succeeded.
    fn lower_source_rows(
        rule: &R,
        lowering: &TraceLowering<'_>,
        codomain_permutation: &[usize],
        domain_permutation: &[usize],
        terms: &mut Vec<TensorTraceFusionStructureTerm<Self::Scalar>>,
    ) -> Result<PendingCoefficientGroups, Self::Error>;

    fn split(
        rule: &R,
        tree: &FusionTreeKey,
        rank: usize,
    ) -> Result<(FusionTreeKey, FusionTreeKey), Self::Error>;

    /// `dim(coupled)/dim(first) · Π twist(non-dual)`, in the mode's own
    /// arithmetic (the two modes round differently, see F4 of #1856).
    fn channel_factor(rule: &R, trace_tree: &FusionTreeKey) -> Result<Self::Scalar, Self::Error>;
}

/// The trace geometry every source row is lowered against.
pub(super) struct TraceLowering<'a> {
    pub(super) dst_structure: &'a BlockStructure,
    pub(super) src: OrientedTraceSource<'a>,
    pub(super) axis_plan: &'a TensorTraceAxisPlan,
    pub(super) dst_codomain_rank: usize,
}

impl TraceLowering<'_> {
    /// Lowers one source block's permuted rows: split at the open ranks,
    /// keep the rows whose traced trees match, then scale by the channel
    /// factor and address the destination block.
    #[allow(clippy::too_many_arguments)]
    #[inline]
    fn lower<'r, M, R, I>(
        &self,
        rule: &R,
        src_block_index: usize,
        src_key: impl Fn() -> Result<FusionTreePairKey, M::Error>,
        rows: I,
        terms: &mut Vec<TensorTraceFusionStructureTerm<M::Scalar>>,
    ) -> Result<(), M::Error>
    where
        M: TraceTermMode<R>,
        M::Scalar: 'r,
        I: IntoIterator<Item = (&'r FusionTreePairKey, &'r M::Scalar)>,
    {
        for (permuted_key, permutation_coefficient) in rows {
            let (dst_codomain_tree, trace_codomain_tree) =
                M::split(rule, permuted_key.codomain_tree(), self.dst_codomain_rank)?;
            let (dst_domain_tree, trace_domain_tree) = M::split(
                rule,
                permuted_key.domain_tree(),
                self.axis_plan.output_axes.len() - self.dst_codomain_rank,
            )?;
            if trace_codomain_tree != trace_domain_tree {
                continue;
            }
            let trace_factor = M::channel_factor(rule, &trace_codomain_tree)?;
            let coefficient = permutation_coefficient.clone() * trace_factor;
            let dst_key = FusionTreePairKey::pair(dst_codomain_tree, dst_domain_tree);
            let dst_block = self
                .dst_structure
                .find_block_index_by_fusion_tree_pair(&dst_key)
                .ok_or_else(|| {
                    M::operation(OperationError::MissingBlockKey {
                        key: Box::new(BlockKey::from(dst_key.clone())),
                    })
                })?;
            terms.push(TensorTraceFusionStructureTerm {
                dst_key,
                src_key: src_key()?,
                dst_block,
                src_block: src_block_index,
                coefficient,
            });
        }
        Ok(())
    }
}

/// The one trace term builder: TensorKit's `(p…, q…)` permutation, then each
/// source block's rows lowered by [`TraceLowering::lower`]. Returns the
/// terms and the permutation groups built on a cache-4 miss; the caller
/// publishes them only after its whole call succeeded, or drops them.
#[allow(clippy::type_complexity)]
pub(super) fn build_trace_terms<M, R>(
    rule: &R,
    dst_structure: &BlockStructure,
    src: OrientedTraceSource<'_>,
    axis_plan: &TensorTraceAxisPlan,
    dst_codomain_rank: usize,
) -> Result<
    (
        Vec<TensorTraceFusionStructureTerm<M::Scalar>>,
        PendingCoefficientGroups,
    ),
    M::Error,
>
where
    M: TraceTermMode<R>,
{
    let mut codomain_permutation =
        Vec::with_capacity(dst_codomain_rank + axis_plan.trace_lhs_axes.len());
    codomain_permutation.extend_from_slice(&axis_plan.output_axes[..dst_codomain_rank]);
    codomain_permutation.extend_from_slice(&axis_plan.trace_lhs_axes);
    let mut domain_permutation = Vec::with_capacity(
        axis_plan.output_axes.len() - dst_codomain_rank + axis_plan.trace_rhs_axes.len(),
    );
    domain_permutation.extend_from_slice(&axis_plan.output_axes[dst_codomain_rank..]);
    domain_permutation.extend_from_slice(&axis_plan.trace_rhs_axes);

    let lowering = TraceLowering {
        dst_structure,
        src,
        axis_plan,
        dst_codomain_rank,
    };
    let mut terms = Vec::new();
    let pending = M::lower_source_rows(
        rule,
        &lowering,
        &codomain_permutation,
        &domain_permutation,
        &mut terms,
    )?;
    Ok((terms, pending))
}

/// Lowers every source block in block order, resolving a fusion group's
/// permutation columns once, when its first member is reached: from cache 4
/// when resident (no admission and no composition, see
/// [`TraceColumnReuse`]), otherwise by `compose_group`, staged for the
/// caller to publish.
///
/// Why not compose every group up front: a later group's symbol or
/// admission error must not overtake an earlier source's lowering error.
/// The current group stays atomic because per-source replay would repeat
/// its F/R traversal (TensorKit `_trace_permute!` permutes a whole fusion
/// block), so provider errors can change order only between members of one
/// group (#2054).
fn lower_by_fusion_group<M, R>(
    rule: &R,
    lowering: &TraceLowering<'_>,
    codomain_permutation: &[usize],
    domain_permutation: &[usize],
    terms: &mut Vec<TensorTraceFusionStructureTerm<M::Scalar>>,
    mut compose_group: impl FnMut(
        &[usize],
    )
        -> Result<BlockSourceColumns<FusionTreePairKey, M::Scalar>, M::Error>,
) -> Result<PendingCoefficientGroups, M::Error>
where
    M: TraceTermMode<R>,
{
    let src = lowering.src;
    // Keyed by the storage keys and orientation (not the oriented keys): a
    // warm lookup then allocates no swapped key.
    let mut reuse = TraceColumnReuse::<M::Scalar>::new(
        M::rule_identity(rule),
        M::MODE,
        TreeTransformOperation::permute(
            codomain_permutation.iter().copied(),
            domain_permutation.iter().copied(),
        ),
        src.orientation,
    );
    let mut storage_keys = Vec::new();
    let block_count = src.structure.block_count();
    let groups = src.structure.fusion_tree_group_slice();
    // Each source's group and its position (column) in that group.
    let mut member_of_source = vec![None; block_count];
    for (group_index, group) in groups.iter().enumerate() {
        for (position, &src_block_index) in group.block_indices().iter().enumerate() {
            member_of_source[src_block_index] = Some((group_index, position));
        }
    }
    let mut columns_by_group = (0..groups.len()).map(|_| None).collect::<Vec<_>>();
    for (src_block_index, member) in member_of_source.into_iter().enumerate() {
        let (group_index, position) = member.ok_or_else(|| {
            M::operation(OperationError::InvalidArgument {
                message: "trace source block was not assigned to a fusion group",
            })
        })?;
        let columns = match &mut columns_by_group[group_index] {
            Some(columns) => columns,
            slot @ None => {
                let group = &groups[group_index];
                record_trace_transform_invocation(src_block_index);
                storage_keys.clear();
                for &index in group.block_indices() {
                    storage_keys.push(src.storage_key(index).map_err(M::operation)?);
                }
                let columns = match reuse.lookup((group.group_key(), &storage_keys)) {
                    Ok(columns) => columns,
                    Err(hash) => {
                        let columns = compose_group(group.block_indices())?;
                        reuse.stage(hash, (group.group_key(), &storage_keys), columns)
                    }
                };
                if columns.source_count() != group.block_indices().len() {
                    return Err(M::operation(OperationError::InvalidArgument {
                        message: "trace block transform returned the wrong source column count",
                    }));
                }
                slot.insert(columns)
            }
        };
        let destinations = columns.destinations();
        lowering.lower::<M, R, _>(
            rule,
            src_block_index,
            || src.source_key(src_block_index).map_err(M::operation),
            columns
                .column(position)
                .iter()
                .map(|(row, coefficient)| (&destinations[*row], coefficient)),
            terms,
        )?;
    }
    Ok(reuse.into_pending())
}

impl<R> TraceTermMode<R> for MultiplicityFreeAdmissionMode
where
    R: MultiplicityFreeRigidSymbols,
    R::Scalar: Clone + Add<Output = R::Scalar> + Mul<Output = R::Scalar> + Zero,
{
    type Scalar = R::Scalar;
    type Error = OperationError;

    const MODE: TransformerMode = TransformerMode::MultiplicityFree;

    fn rule_identity(rule: &R) -> RuleIdentity {
        FusionRule::rule_identity(rule)
    }

    #[inline]
    fn operation(error: OperationError) -> Self::Error {
        error
    }

    /// TensorKit's dispatch: one permuted row per tree pair for unique
    /// fusion (`_trace_permute!(::UniqueFusion)`), otherwise one block-indexed
    /// permute per fusion group (`fusionblocks`).
    fn lower_source_rows(
        rule: &R,
        lowering: &TraceLowering<'_>,
        codomain_permutation: &[usize],
        domain_permutation: &[usize],
        terms: &mut Vec<TensorTraceFusionStructureTerm<Self::Scalar>>,
    ) -> Result<PendingCoefficientGroups, Self::Error> {
        let src = lowering.src;
        src.validate_source_keys()?;
        if rule.fusion_style() == FusionStyleKind::Unique {
            let (source_codomain_rank, source_domain_rank) = match src.orientation {
                FusionTreePairOrientation::Direct => (src.storage_nout, src.storage_nin),
                FusionTreePairOrientation::Adjoint => (src.storage_nin, src.storage_nout),
            };
            let prepared = PreparedTreePairOperation::prepare_permute(
                rule,
                source_codomain_rank,
                source_domain_rank,
                codomain_permutation,
                domain_permutation,
            )
            .map_err(OperationError::from_core_preserving_context)?;
            for src_block_index in 0..src.structure.block_count() {
                record_trace_transform_invocation(src_block_index);
                let row = prepared
                    .execute_unique_rigid(rule, &src.source_key(src_block_index)?)
                    .map_err(OperationError::from_core_preserving_context)?;
                lowering.lower::<Self, R, _>(
                    rule,
                    src_block_index,
                    || src.source_key(src_block_index),
                    core::iter::once((&row.0, &row.1)),
                    terms,
                )?;
            }
            // TensorKit `NoCache` for Unique: one uncached phase per tree.
            return Ok(PendingCoefficientGroups::default());
        }

        lower_by_fusion_group::<Self, R>(
            rule,
            lowering,
            codomain_permutation,
            domain_permutation,
            terms,
            |indices| {
                multiplicity_free_permute_tree_pair_block_indexed(
                    rule,
                    src.structure,
                    indices,
                    src.orientation,
                    codomain_permutation,
                    domain_permutation,
                )
                .map_err(OperationError::from_core_preserving_context)
            },
        )
    }

    #[inline]
    fn split(
        rule: &R,
        tree: &FusionTreeKey,
        rank: usize,
    ) -> Result<(FusionTreeKey, FusionTreeKey), Self::Error> {
        split_fusion_tree(rule, tree, rank).map_err(OperationError::from_core_preserving_context)
    }

    #[inline]
    fn channel_factor(rule: &R, trace_tree: &FusionTreeKey) -> Result<Self::Scalar, Self::Error> {
        let coupled = trace_tree.coupled();
        let first = trace_tree.uncoupled().first().copied().ok_or(
            tenet_core::CoreError::MalformedFusionTree {
                message: "trace channel requires at least one uncoupled sector",
            },
        );
        let first = first.map_err(OperationError::from_core_preserving_context)?;
        let mut factor = rule.dim_scalar(coupled) * rule.inv_dim_scalar(first);
        for (&sector, &is_dual) in trace_tree
            .uncoupled()
            .iter()
            .zip(trace_tree.is_dual())
            .skip(1)
        {
            if !is_dual {
                factor = factor * rule.twist_scalar(sector);
            }
        }
        Ok(factor)
    }
}

impl<R> TraceTermMode<R> for CheckedGenericAdmissionMode
where
    R: CheckedGenericPivotal,
    R::Scalar: Clone + Mul<Output = R::Scalar> + Zero + 'static,
{
    type Scalar = R::Scalar;
    type Error = CheckedGenericPlanError<R::Error>;

    const MODE: TransformerMode = TransformerMode::CheckedGeneric;

    fn rule_identity(rule: &R) -> RuleIdentity {
        CheckedGenericFusion::rule_identity(rule)
    }

    #[inline]
    fn operation(error: OperationError) -> Self::Error {
        CheckedGenericPlanError::Operation(error)
    }

    /// TensorKit's Generic dispatch: one whole-basis checked permute per
    /// fusion group, admitting each member once (#2054).
    fn lower_source_rows(
        rule: &R,
        lowering: &TraceLowering<'_>,
        codomain_permutation: &[usize],
        domain_permutation: &[usize],
        terms: &mut Vec<TensorTraceFusionStructureTerm<Self::Scalar>>,
    ) -> Result<PendingCoefficientGroups, Self::Error> {
        let src = lowering.src;
        src.validate_source_keys()
            .map_err(CheckedGenericPlanError::Operation)?;
        lower_by_fusion_group::<Self, R>(
            rule,
            lowering,
            codomain_permutation,
            domain_permutation,
            terms,
            |indices| {
                generic_permute_tree_pair_block_indexed_checked(
                    rule,
                    src.structure,
                    indices,
                    src.orientation,
                    codomain_permutation,
                    domain_permutation,
                )
                .map_err(map_checked_generic_trace_symbol_error)
            },
        )
    }

    fn split(
        rule: &R,
        tree: &FusionTreeKey,
        rank: usize,
    ) -> Result<(FusionTreeKey, FusionTreeKey), Self::Error> {
        split_fusion_tree_generic_checked(rule, tree, rank)
            .map_err(map_checked_generic_trace_structure_error)
    }

    fn channel_factor(rule: &R, trace_tree: &FusionTreeKey) -> Result<Self::Scalar, Self::Error> {
        let first = trace_tree.uncoupled().first().copied().ok_or_else(|| {
            CheckedGenericPlanError::Core(tenet_core::CoreError::MalformedFusionTree {
                message: "trace channel requires at least one uncoupled sector",
            })
        })?;
        let sqrt_coupled = rule
            .try_sqrt_dim_scalar(trace_tree.coupled())
            .map_err(CheckedGenericPlanError::Provider)?;
        let inv_sqrt_first = rule
            .try_inv_sqrt_dim_scalar(first)
            .map_err(CheckedGenericPlanError::Provider)?;
        // The categorical trace factor is dim(coupled)/dim(first), expressed
        // through the Generic provider's square-root dimension interface.
        let mut factor =
            sqrt_coupled.clone() * sqrt_coupled * inv_sqrt_first.clone() * inv_sqrt_first;
        for (&sector, &is_dual) in trace_tree
            .uncoupled()
            .iter()
            .zip(trace_tree.is_dual())
            .skip(1)
        {
            if !is_dual {
                factor = factor
                    * rule
                        .try_twist_scalar(sector)
                        .map_err(CheckedGenericPlanError::Provider)?;
            }
        }
        Ok(factor)
    }
}
