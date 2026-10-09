use super::*;

#[derive(Clone, Debug)]
pub(crate) struct DynamicTreeExecutionArtifact<C = f64> {
    pub(in crate::contract) orientation: FusionContractOrientation,
    pub(in crate::contract) lhs_transform: DynamicFusionTransformedSourceEntry<C>,
    pub(in crate::contract) rhs_transform: DynamicFusionTransformedSourceEntry<C>,
    pub(in crate::contract) lhs_borrowed: bool,
    pub(in crate::contract) rhs_borrowed: bool,
    /// The fermionic contraction twist scales the physical lhs's
    /// materialized core source (else the physical rhs's); meaningful only
    /// when `source_twist_destination_scales` is non-empty.
    pub(in crate::contract) twist_lhs: bool,
    /// The twist as `(twisted destination block offset, θ_b ≠ 1)`, sorted by
    /// offset and without zero-element blocks: the single representation
    /// every executor folds into the move writing block b of the twisted
    /// source transform (Host `tree_transform_structure_overwrite_into_raw`
    /// and member replay, device `replay_with_destination_scales`).
    source_twist_destination_scales: Box<[(usize, C)]>,
    pub(in crate::contract) core_dst: Option<DynamicFusionCoreDstEntry<C>>,
    pub(in crate::contract) block_plan: Arc<FusionBlockContractPlan<C>>,
}

impl<C: DenseBlockScalar> DynamicTreeExecutionArtifact<C> {
    #[cfg(test)]
    pub(crate) fn test_has_core_dst(&self) -> bool {
        self.core_dst.is_some()
    }

    #[cfg(test)]
    pub(crate) fn test_lhs_transform(
        &self,
    ) -> (
        &TreeTransformStructure<C>,
        &Arc<BlockStructure>,
        &Arc<BlockStructure>,
        &DynamicFusionMapSpace,
    ) {
        (
            &self.lhs_transform.transform_structure,
            self.lhs_transform.space.structure(),
            &self.lhs_transform.replay_structure,
            &self.lhs_transform.space,
        )
    }

    #[cfg(test)]
    pub(crate) fn orientation(&self) -> FusionContractOrientation {
        self.orientation
    }

    pub(crate) fn requires_source_twist(&self) -> bool {
        !self.source_twist_destination_scales.is_empty()
    }

    /// `(lhs, rhs)` in the core GEMM's operand order: swapped for an
    /// `RhsLhs` orientation.
    #[inline]
    pub(crate) fn core_order<T>(&self, lhs: T, rhs: T) -> (T, T) {
        if self.orientation == FusionContractOrientation::RhsLhs {
            (rhs, lhs)
        } else {
            (lhs, rhs)
        }
    }

    /// The physical lhs or rhs value, whichever source the contraction twist
    /// scales; meaningful only when [`Self::requires_source_twist`].
    #[inline]
    pub(crate) fn on_twisted<T>(&self, lhs: T, rhs: T) -> T {
        if self.twist_lhs {
            lhs
        } else {
            rhs
        }
    }

    /// The structure the core GEMMs write: the output transform's source, or
    /// the caller's destination when the output transform is the identity.
    pub(crate) fn core_dst_structure<'a>(
        &'a self,
        dst: &'a Arc<BlockStructure>,
    ) -> &'a Arc<BlockStructure> {
        self.core_dst
            .as_ref()
            .map_or(dst, |entry| entry.space.structure())
    }

    /// The twist folds of the lhs and rhs source stages: the twisted stage
    /// carries [`Self::source_twist_destination_scales`], the other none. A
    /// twisted source is never borrowed (checked at construction).
    pub(crate) fn stage_scales(&self) -> [&[(usize, C)]; 2] {
        let scales = &self.source_twist_destination_scales[..];
        let (lhs, rhs) = self.on_twisted((scales, &[][..]), (&[][..], scales));
        [lhs, rhs]
    }

    /// Whether the twist, if any, scales the physical lhs's core source.
    #[cfg(test)]
    pub(crate) fn twists_lhs(&self) -> bool {
        self.twist_lhs
    }

    #[cfg(test)]
    pub(crate) fn source_twist_destination_scales(&self) -> &[(usize, C)] {
        &self.source_twist_destination_scales
    }

    pub(crate) fn block_plan(&self) -> &FusionBlockContractPlan<C> {
        &self.block_plan
    }

    pub(crate) fn block_plan_is_fully_direct(&self) -> bool {
        self.block_plan.is_fully_direct()
    }

    /// The core plan's inactive destination blocks when the core GEMMs write
    /// the caller's destination directly (identity output); `None` when an
    /// output transform writes it.
    #[cfg(test)]
    pub(crate) fn direct_destination_inactive_blocks(&self) -> Option<usize> {
        self.core_dst
            .is_none()
            .then(|| self.block_plan.inactive_destination_regions().len())
    }

    #[cfg(test)]
    pub(crate) fn borrowed_sources(&self) -> (bool, bool) {
        (self.lhs_borrowed, self.rhs_borrowed)
    }

    /// The destination space of the physical lhs (`true`) or rhs source
    /// transform, whichever orientation the artifact has.
    #[cfg(test)]
    pub(crate) fn physical_core_space(&self, lhs: bool) -> &DynamicFusionMapSpace {
        if lhs {
            &self.lhs_transform.space
        } else {
            &self.rhs_transform.space
        }
    }

    #[cfg(test)]
    pub(crate) fn twisted_transform_structure(&self) -> &TreeTransformStructure<C> {
        if self.twist_lhs {
            &self.lhs_transform.transform_structure
        } else {
            &self.rhs_transform.transform_structure
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn compile_dynamic_tree_execution_artifact<RuleKey, BT, R, D, C, const PROFILED: bool>(
    tree_context: &mut TreeTransformExecutionContext<D, RuleKey, C, BT>,
    rule: &R,
    layout_primer: LayoutKeyBuilder<R>,
    plan: &FusionContractPlan,
    dst_space: &DynamicFusionMapSpace,
    lhs_space: &DynamicFusionMapSpace,
    lhs_structure: &Arc<BlockStructure>,
    rhs_space: &DynamicFusionMapSpace,
    rhs_structure: &Arc<BlockStructure>,
    profile: Option<&mut TensorContractFusionProfile>,
) -> Result<DynamicTreeExecutionArtifact<C>, OperationError>
where
    RuleKey: 'static + Clone + Eq + std::hash::Hash + Send + Sync,
    BT: TreeTransformBackend<D, C>,
    R: MultiplicityFreeRigidSymbols<Scalar = C> + TreeTransformRuleCacheKey<Key = RuleKey>,
    D: DenseRecouplingScalar + RecouplingCoefficientAction<C>,
    C: DenseBlockScalar,
{
    let source_start = PROFILED.then(std::time::Instant::now);
    let lhs_transform = compile_transformed_source(
        tree_context,
        rule,
        lhs_space,
        lhs_structure,
        plan.lhs_transform(),
        plan.lhs_source_conjugate(),
        layout_primer,
    )?;
    let lhs_layout_borrowable = source_is_borrowable_core_layout(
        lhs_space,
        lhs_structure,
        &lhs_transform.space,
        plan.lhs_transform(),
        plan.lhs_source_conjugate(),
    );
    let rhs_transform = compile_transformed_source(
        tree_context,
        rule,
        rhs_space,
        rhs_structure,
        plan.rhs_transform(),
        plan.rhs_source_conjugate(),
        layout_primer,
    )?;
    let rhs_layout_borrowable = source_is_borrowable_core_layout(
        rhs_space,
        rhs_structure,
        &rhs_transform.space,
        plan.rhs_transform(),
        plan.rhs_source_conjugate(),
    );
    let borrowing = resolve_source_borrowing(
        rule,
        plan,
        &lhs_transform.space,
        &rhs_transform.space,
        lhs_layout_borrowable,
        rhs_layout_borrowable,
    )?;
    finish_dynamic_tree_execution_artifact::<_, _, _, _, _, PROFILED>(
        tree_context,
        rule,
        layout_primer,
        plan,
        dst_space,
        lhs_transform,
        rhs_transform,
        borrowing,
        source_start,
        profile,
    )
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn compile_prelowered_dynamic_tree_execution_artifact<
    RuleKey,
    BT,
    R,
    D,
    C,
    const PROFILED: bool,
>(
    tree_context: &mut TreeTransformExecutionContext<D, RuleKey, C, BT>,
    rule: &R,
    layout_primer: LayoutKeyBuilder<R>,
    plan: &FusionContractPlan,
    dst_space: &DynamicFusionMapSpace,
    lhs: &FusionOperandLayout<'_>,
    rhs: &FusionOperandLayout<'_>,
    profile: Option<&mut TensorContractFusionProfile>,
) -> Result<DynamicTreeExecutionArtifact<C>, OperationError>
where
    RuleKey: 'static + Clone + Eq + std::hash::Hash + Send + Sync,
    BT: TreeTransformBackend<D, C>,
    R: MultiplicityFreeRigidSymbols<Scalar = C> + TreeTransformRuleCacheKey<Key = RuleKey>,
    D: DenseRecouplingScalar + RecouplingCoefficientAction<C>,
    C: DenseBlockScalar,
{
    let source_start = PROFILED.then(std::time::Instant::now);
    debug_assert_eq!(
        plan.lhs_source_conjugate(),
        lhs.orientation() == FusionTreePairOrientation::Adjoint
    );
    debug_assert_eq!(
        plan.rhs_source_conjugate(),
        rhs.orientation() == FusionTreePairOrientation::Adjoint
    );
    let lhs_direct = lhs.is_direct();
    let lhs_transform = compile_prelowered_source_transform(
        tree_context,
        rule,
        lhs,
        plan.lhs_transform(),
        layout_primer,
    )?;
    let lhs_layout_borrowable = lhs_direct
        && source_is_borrowable_core_layout(
            lhs.storage_space(),
            lhs.storage_space().structure(),
            &lhs_transform.space,
            plan.lhs_transform(),
            false,
        );
    let rhs_direct = rhs.is_direct();
    let rhs_transform = compile_prelowered_source_transform(
        tree_context,
        rule,
        rhs,
        plan.rhs_transform(),
        layout_primer,
    )?;
    let rhs_layout_borrowable = rhs_direct
        && source_is_borrowable_core_layout(
            rhs.storage_space(),
            rhs.storage_space().structure(),
            &rhs_transform.space,
            plan.rhs_transform(),
            false,
        );
    let borrowing = resolve_source_borrowing(
        rule,
        plan,
        &lhs_transform.space,
        &rhs_transform.space,
        lhs_layout_borrowable,
        rhs_layout_borrowable,
    )?;
    let artifact = finish_dynamic_tree_execution_artifact::<_, _, _, _, _, PROFILED>(
        tree_context,
        rule,
        layout_primer,
        plan,
        dst_space,
        lhs_transform,
        rhs_transform,
        borrowing,
        source_start,
        profile,
    )?;
    Ok(artifact)
}

pub(super) fn compile_prelowered_source_transform<RuleKey, BT, R, D, C>(
    tree_context: &mut TreeTransformExecutionContext<D, RuleKey, C, BT>,
    rule: &R,
    source: &FusionOperandLayout<'_>,
    operation: &TreeTransformOperation,
    layout_primer: LayoutKeyBuilder<R>,
) -> Result<DynamicFusionTransformedSourceEntry<C>, OperationError>
where
    RuleKey: 'static + Clone + Eq + std::hash::Hash + Send + Sync,
    BT: TreeTransformBackend<D, C>,
    R: MultiplicityFreeRigidSymbols<Scalar = C> + TreeTransformRuleCacheKey<Key = RuleKey>,
    D: DenseRecouplingScalar + RecouplingCoefficientAction<C>,
    C: DenseBlockScalar,
{
    if source.is_direct() {
        compile_transformed_source(
            tree_context,
            rule,
            source.storage_space(),
            source.storage_space().structure(),
            operation,
            false,
            layout_primer,
        )
    } else {
        compile_transformed_source_oriented(tree_context, rule, source, operation, layout_primer)
    }
}

#[allow(clippy::too_many_arguments)]
fn finish_dynamic_tree_execution_artifact<RuleKey, BT, R, D, C, const PROFILED: bool>(
    tree_context: &mut TreeTransformExecutionContext<D, RuleKey, C, BT>,
    rule: &R,
    layout_primer: LayoutKeyBuilder<R>,
    plan: &FusionContractPlan,
    dst_space: &DynamicFusionMapSpace,
    lhs_transform: DynamicFusionTransformedSourceEntry<C>,
    rhs_transform: DynamicFusionTransformedSourceEntry<C>,
    borrowing: SourceBorrowing,
    source_start: Option<std::time::Instant>,
    mut profile: Option<&mut TensorContractFusionProfile>,
) -> Result<DynamicTreeExecutionArtifact<C>, OperationError>
where
    RuleKey: 'static + Clone + Eq + std::hash::Hash + Send + Sync,
    BT: TreeTransformBackend<D, C>,
    R: MultiplicityFreeRigidSymbols<Scalar = C> + TreeTransformRuleCacheKey<Key = RuleKey>,
    D: DenseRecouplingScalar + RecouplingCoefficientAction<C>,
    C: DenseBlockScalar,
{
    let physical_lhs_core_space = lhs_transform.space.clone();
    let physical_rhs_core_space = rhs_transform.space.clone();
    let reverse = plan.orientation() == FusionContractOrientation::RhsLhs;
    let (core_left_space, core_right_space) = if reverse {
        (&physical_rhs_core_space, &physical_lhs_core_space)
    } else {
        (&physical_lhs_core_space, &physical_rhs_core_space)
    };
    let twisted_transform = if borrowing.twist_lhs {
        &lhs_transform
    } else {
        &rhs_transform
    };
    let source_twist_destination_scales = compile_contract_twist(
        rule,
        &twisted_transform.space,
        core_right_space.homspace(),
        borrowing.twist_lhs != reverse,
        plan.core_axes().as_spec().rhs_contracting_axes(),
    )?;
    if !source_twist_destination_scales.is_empty() {
        validate_uniform_multi_scales(
            &twisted_transform.transform_structure,
            &source_twist_destination_scales,
        )?;
    }
    if let Some(start) = source_start {
        profile
            .as_deref_mut()
            .expect("profiled compilation carries a profile")
            .source_space_lookup += start.elapsed();
        #[cfg(test)]
        PROFILED_ARTIFACT_COMPILE_PHASES
            .with(|phases| phases.set(phases.get() | PROFILED_ARTIFACT_SOURCE_PHASE));
    }

    let core_dst_start = PROFILED.then(std::time::Instant::now);
    let core_dst = if plan.output_transform_is_identity() {
        None
    } else {
        Some(compile_core_dst(
            tree_context,
            rule,
            core_left_space,
            core_right_space,
            plan,
            dst_space,
            layout_primer,
        )?)
    };
    if core_dst.is_some() {
        if let Some(start) = core_dst_start {
            profile
                .as_deref_mut()
                .expect("profiled compilation carries a profile")
                .core_dst_space_lookup += start.elapsed();
            #[cfg(test)]
            PROFILED_ARTIFACT_COMPILE_PHASES
                .with(|phases| phases.set(phases.get() | PROFILED_ARTIFACT_CORE_DST_PHASE));
        }
    }
    let block_dst_space = core_dst
        .as_ref()
        .map_or(dst_space, |entry| entry.space.as_ref());
    let block_plan_start = PROFILED.then(std::time::Instant::now);
    let block_plan = super::resolution::compile_derived_core_plan(
        rule,
        block_dst_space,
        core_left_space,
        core_right_space,
        plan.core_axes().as_spec(),
    )?;
    if let Some(start) = block_plan_start {
        profile
            .expect("profiled compilation carries a profile")
            .core_block_plan_build += start.elapsed();
        #[cfg(test)]
        PROFILED_ARTIFACT_COMPILE_PHASES
            .with(|phases| phases.set(phases.get() | PROFILED_ARTIFACT_BLOCK_PLAN_PHASE));
    }
    // Why not check this in every executor: Host, member and device replays
    // all fold the twist into the transform writing the twisted source's
    // owned scratch, and borrowing never picks the twisted side
    // (`resolve_source_borrowing`).
    let twisted_borrowed = if borrowing.twist_lhs {
        borrowing.lhs_borrowed
    } else {
        borrowing.rhs_borrowed
    };
    if !source_twist_destination_scales.is_empty() && twisted_borrowed {
        return Err(OperationError::InvalidArgument {
            message: "contraction twist must scale an owned source",
        });
    }
    Ok(DynamicTreeExecutionArtifact {
        orientation: plan.orientation(),
        lhs_transform,
        rhs_transform,
        lhs_borrowed: borrowing.lhs_borrowed,
        rhs_borrowed: borrowing.rhs_borrowed,
        twist_lhs: borrowing.twist_lhs,
        source_twist_destination_scales: source_twist_destination_scales.into_boxed_slice(),
        core_dst,
        block_plan,
    })
}
