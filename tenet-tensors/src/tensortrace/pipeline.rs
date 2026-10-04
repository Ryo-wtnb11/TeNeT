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
//! checked Generic modes keep different error variants, and until #1872
//! different timing for a non-dual pair; the hooks are the narrowest layer
//! that owns those differences.

use super::*;
use tenet_core::{CheckedGenericAdmissionMode, MultiplicityFreeAdmissionMode};

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

    /// `Some` rejects a non-dual pair at once; `None` records it in
    /// [`CheckedTraceGeometry::trace_pairs_match`] for the compile to reject.
    fn non_dual_error() -> Option<Self::Error>;
}

/// The one trace preflight: lowers the axes onto the stored orientation,
/// then selects the output HomSpace and checks the pairs.
pub(super) fn trace_preflight<M, R>(
    src: &BoundDynamicFusionMapSpace<R>,
    axes: TensorTraceAxisSpec<'_>,
    dst_nout: usize,
) -> Result<CheckedTraceGeometry, M::Error>
where
    M: TracePreflightMode<R>,
{
    M::validate_source(src)?;
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
) -> Result<CheckedTraceGeometry, M::Error>
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
        if !M::pair_is_dual(rule, src_homspace, lhs_axis, rhs_axis)? {
            if let Some(error) = M::non_dual_error() {
                return Err(error);
            }
            trace_pairs_match = false;
        }
    }
    Ok(CheckedTraceGeometry {
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

    fn non_dual_error() -> Option<Self::Error> {
        None
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

    fn non_dual_error() -> Option<Self::Error> {
        Some(CheckedGenericPlanError::Operation(
            OperationError::UnsupportedTensorContractScope {
                message: "trace pairs must contain dual sectors",
            },
        ))
    }
}

/// What the trace term builder asks of an admission mode.
pub(super) trait TraceTermMode<R> {
    type Scalar: Clone + Mul<Output = Self::Scalar>;
    type Error;

    fn operation(error: OperationError) -> Self::Error;

    /// Hands every source block's permuted tree-pair rows, in source block
    /// order, to [`TraceLowering::lower`].
    fn lower_source_rows(
        rule: &R,
        lowering: &TraceLowering<'_>,
        codomain_permutation: &[usize],
        domain_permutation: &[usize],
        terms: &mut Vec<TensorTraceFusionStructureTerm<Self::Scalar>>,
    ) -> Result<(), Self::Error>;

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
    fn lower<M, R, I>(
        &self,
        rule: &R,
        src_block_index: usize,
        src_key: impl Fn() -> Result<FusionTreePairKey, M::Error>,
        rows: I,
        terms: &mut Vec<TensorTraceFusionStructureTerm<M::Scalar>>,
    ) -> Result<(), M::Error>
    where
        M: TraceTermMode<R>,
        I: IntoIterator<Item = (FusionTreePairKey, M::Scalar)>,
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
            let coefficient = permutation_coefficient * trace_factor;
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
/// source block's rows lowered by [`TraceLowering::lower`].
pub(super) fn build_trace_terms<M, R>(
    rule: &R,
    dst_structure: &BlockStructure,
    src: OrientedTraceSource<'_>,
    axis_plan: &TensorTraceAxisPlan,
    dst_codomain_rank: usize,
) -> Result<Vec<TensorTraceFusionStructureTerm<M::Scalar>>, M::Error>
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
    M::lower_source_rows(
        rule,
        &lowering,
        &codomain_permutation,
        &domain_permutation,
        &mut terms,
    )?;
    Ok(terms)
}

impl<R> TraceTermMode<R> for MultiplicityFreeAdmissionMode
where
    R: MultiplicityFreeRigidSymbols,
    R::Scalar: Clone + Add<Output = R::Scalar> + Mul<Output = R::Scalar> + Zero,
{
    type Scalar = R::Scalar;
    type Error = OperationError;

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
    ) -> Result<(), Self::Error> {
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
                    core::iter::once(row),
                    terms,
                )?;
            }
            return Ok(());
        }

        let groups = src.structure.fusion_tree_group_slice();
        let mut rows_by_source = (0..src.structure.block_count())
            .map(|_| None)
            .collect::<Vec<Option<Vec<(FusionTreePairKey, R::Scalar)>>>>();
        let mut group_by_source = vec![None; src.structure.block_count()];
        for (group_index, group) in groups.iter().enumerate() {
            for &src_block_index in group.block_indices() {
                group_by_source[src_block_index] = Some(group_index);
            }
        }

        for src_block_index in 0..src.structure.block_count() {
            if rows_by_source[src_block_index].is_none() {
                let group_index =
                    group_by_source[src_block_index].ok_or(OperationError::InvalidArgument {
                        message: "trace source block was not assigned to a fusion group",
                    })?;
                let group = &groups[group_index];
                record_trace_transform_invocation(src_block_index);
                let group_rows = multiplicity_free_permute_tree_pair_block_indexed(
                    rule,
                    src.structure,
                    group.block_indices(),
                    src.orientation,
                    codomain_permutation,
                    domain_permutation,
                )
                .map_err(OperationError::from_core_preserving_context)?;
                if group_rows.len() != group.block_indices().len() {
                    return Err(OperationError::InvalidArgument {
                        message: "trace block transform returned the wrong source row count",
                    });
                }
                for (&src_block_index, rows) in group.block_indices().iter().zip(group_rows) {
                    rows_by_source[src_block_index] = Some(rows);
                }
            }

            // Why not transform every group up front: a later group's symbol error
            // must not overtake an earlier source's lowering error. The whole current
            // group stays atomic because per-source replay would duplicate its F/R
            // traversal; expert incomplete structures can therefore observe changed
            // error timing only between members of that same group.
            let rows =
                rows_by_source[src_block_index]
                    .take()
                    .ok_or(OperationError::InvalidArgument {
                        message: "trace source block was not assigned to a fusion group",
                    })?;
            lowering.lower::<Self, R, _>(
                rule,
                src_block_index,
                || src.source_key(src_block_index),
                rows,
                terms,
            )?;
        }
        Ok(())
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
    R::Scalar: Clone + Mul<Output = R::Scalar> + Zero,
{
    type Scalar = R::Scalar;
    type Error = CheckedGenericPlanError<R::Error>;

    #[inline]
    fn operation(error: OperationError) -> Self::Error {
        CheckedGenericPlanError::Operation(error)
    }

    /// One checked permute per source tree pair: tenet-core has no
    /// block-indexed Generic permute yet (F3 of #1856).
    fn lower_source_rows(
        rule: &R,
        lowering: &TraceLowering<'_>,
        codomain_permutation: &[usize],
        domain_permutation: &[usize],
        terms: &mut Vec<TensorTraceFusionStructureTerm<Self::Scalar>>,
    ) -> Result<(), Self::Error> {
        let src = lowering.src;
        for src_block_index in 0..src.structure.block_count() {
            let src_key = src
                .source_key(src_block_index)
                .map_err(CheckedGenericPlanError::Operation)?;
            validate_generic_fusion_tree_pair_checked(rule, &src_key).map_err(
                |error| match error {
                    tenet_core::CheckedGenericStructureError::Provider(error) => {
                        CheckedGenericPlanError::Provider(error)
                    }
                    tenet_core::CheckedGenericStructureError::Core(error) => {
                        CheckedGenericPlanError::Core(error)
                    }
                },
            )?;
            let rows = generic_permute_tree_pair_checked(
                rule,
                &src_key,
                codomain_permutation,
                domain_permutation,
            )
            .map_err(map_checked_generic_trace_symbol_error)?;
            lowering.lower::<Self, R, _>(
                rule,
                src_block_index,
                || Ok(src_key.clone()),
                rows,
                terms,
            )?;
        }
        Ok(())
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
