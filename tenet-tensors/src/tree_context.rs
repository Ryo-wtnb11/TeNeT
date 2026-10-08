use core::ops::{Add, Mul};
use std::hash::Hash;
use std::marker::PhantomData;
use std::sync::Arc;

use num_traits::Zero;
use tenet_core::{
    BlockKey, BlockStructure, CategoricalScalar, CheckedGenericRigidSymbols, FusionTreeHomSpace,
    FusionTreePairOrientation, HostReadableStorage, HostWritableStorage,
    MultiplicityFreeAdmissionMode, MultiplicityFreeFusionSymbols, MultiplicityFreeRigidSymbols,
    Placement, RuleIdentity, TensorMap,
};

use crate::contract::{
    tree_transform_operation_axes, BoundDynamicFusionMapSpace, FusionOperand,
    PreparedCheckedGenericDynamicSpace,
};
use crate::mode::{PlanningAlgebra, TreeStructureSource};
use crate::tree_transform::{
    build_checked_generic_tree_pair_transform_group_plan_validated, lookup_bound,
    publish_committed, publishable, validate_checked_generic_tree_pair_plan_preflight,
    CheckedGenericPlanError, CheckedPendingCoefficients, CoefficientGroupReuse,
    CompletedTransformerKey, OrientedBasisOrder, TransformerMode, TreeTransformOperation,
    TreeTransformPlanning, TreeTransformRuleCacheKey, TreeTransformScope,
};
use crate::{
    validate_oriented_fusion_layout, RecouplingCoefficientAction, ReportsPlacement,
    TreeTransformReplayProfile, TreeTransformStructure,
};
use tenet_dense::DefaultDenseExecutor;
use tenet_operations::OperationError;
use tenet_operations::TreeTransformScalar;
use tenet_operations::{DenseTreeTransformOperations, TreeTransformBackend};

enum CheckedTreeTransformInputKind<'a, P, D> {
    Direct {
        space: &'a BoundDynamicFusionMapSpace<P>,
        data: &'a [D],
    },
    Adjoint {
        logical_space: &'a BoundDynamicFusionMapSpace<P>,
        parent_space: &'a BoundDynamicFusionMapSpace<P>,
        parent_data: &'a [D],
    },
}

/// Borrowed direct or lazy-adjoint request for the checked transform owner.
#[doc(hidden)]
pub struct CheckedTreeTransformInput<'a, P, D> {
    kind: CheckedTreeTransformInputKind<'a, P, D>,
}

impl<'a, P, D> CheckedTreeTransformInput<'a, P, D> {
    pub fn direct(space: &'a BoundDynamicFusionMapSpace<P>, data: &'a [D]) -> Self {
        Self {
            kind: CheckedTreeTransformInputKind::Direct { space, data },
        }
    }

    pub fn adjoint(
        logical_space: &'a BoundDynamicFusionMapSpace<P>,
        parent_space: &'a BoundDynamicFusionMapSpace<P>,
        parent_data: &'a [D],
    ) -> Self {
        Self {
            kind: CheckedTreeTransformInputKind::Adjoint {
                logical_space,
                parent_space,
                parent_data,
            },
        }
    }
}

#[cfg(test)]
/// Applies one checked Generic permute, braid, or transpose and returns its owned output.
///
/// Provider queries and replay compilation finish against an uninterned
/// destination preview. The destination structure becomes visible only after
/// those fallible stages succeed.
#[doc(hidden)]
#[allow(clippy::type_complexity)]
pub fn tree_transform_dyn_owned_checked_generic<P, D>(
    operation: TreeTransformOperation,
    src_space: &BoundDynamicFusionMapSpace<P>,
    src_data: &[D],
    alpha: D,
) -> Result<(BoundDynamicFusionMapSpace<P>, Vec<D>), CheckedGenericPlanError<P::Error>>
where
    P: CheckedGenericRigidSymbols,
    P::Scalar: CategoricalScalar + Copy + Zero + Sync + 'static,
    D: crate::DenseRecouplingScalar
        + RecouplingCoefficientAction<P::Scalar>
        + crate::ConjugateValue,
{
    let mut context = TreeTransformExecutionContext::<D, RuleIdentity, P::Scalar>::default();
    tree_transform_dyn_owned_checked_generic_in_context(
        &mut context,
        operation,
        src_space,
        src_data,
        alpha,
    )
}

#[cfg(test)]
/// Runtime-context variant of [`tree_transform_dyn_owned_checked_generic`].
///
/// A completed structure is published only after replay and destination commit.
#[doc(hidden)]
#[allow(clippy::type_complexity)]
pub fn tree_transform_dyn_owned_checked_generic_in_context<P, D, B>(
    context: &mut TreeTransformExecutionContext<D, RuleIdentity, P::Scalar, B>,
    operation: TreeTransformOperation,
    src_space: &BoundDynamicFusionMapSpace<P>,
    src_data: &[D],
    alpha: D,
) -> Result<(BoundDynamicFusionMapSpace<P>, Vec<D>), CheckedGenericPlanError<P::Error>>
where
    P: CheckedGenericRigidSymbols,
    P::Scalar: CategoricalScalar
        + Copy
        + Clone
        + Add<Output = P::Scalar>
        + Mul<Output = P::Scalar>
        + Zero
        + Send
        + Sync
        + 'static,
    D: crate::DenseRecouplingScalar
        + RecouplingCoefficientAction<P::Scalar>
        + crate::ConjugateValue,
    B: TreeTransformBackend<D, P::Scalar>,
{
    tree_transform_dyn_owned_checked_generic_input_in_context(
        context,
        operation,
        CheckedTreeTransformInput::direct(src_space, src_data),
        alpha,
    )
}

/// Common checked owner for direct and lazy-adjoint borrowed input.
#[doc(hidden)]
#[allow(clippy::type_complexity)]
pub fn tree_transform_dyn_owned_checked_generic_input_in_context<P, D, B>(
    context: &mut TreeTransformExecutionContext<D, RuleIdentity, P::Scalar, B>,
    operation: TreeTransformOperation,
    input: CheckedTreeTransformInput<'_, P, D>,
    alpha: D,
) -> Result<(BoundDynamicFusionMapSpace<P>, Vec<D>), CheckedGenericPlanError<P::Error>>
where
    P: CheckedGenericRigidSymbols,
    P::Scalar: CategoricalScalar
        + Copy
        + Clone
        + Add<Output = P::Scalar>
        + Mul<Output = P::Scalar>
        + Zero
        + Send
        + Sync
        + 'static,
    D: crate::DenseRecouplingScalar
        + RecouplingCoefficientAction<P::Scalar>
        + crate::ConjugateValue,
    B: TreeTransformBackend<D, P::Scalar>,
{
    let (logical_space, storage_space, src_data, operand) = match input.kind {
        CheckedTreeTransformInputKind::Direct { space, data } => {
            (space, space, data, FusionOperand::direct(space.space()))
        }
        CheckedTreeTransformInputKind::Adjoint {
            logical_space,
            parent_space,
            parent_data,
        } => (
            logical_space,
            parent_space,
            parent_data,
            FusionOperand::adjoint(parent_space.space()),
        ),
    };
    let source = logical_space.space();
    let storage_source = storage_space.space();
    let provider = logical_space.provider();
    let expected = storage_source.required_len()?;
    if src_data.len() != expected {
        return Err(OperationError::ElementCountMismatch {
            expected,
            actual: src_data.len(),
        }
        .into());
    }
    // Before admission: a clear during this request leaves its transformer
    // and composed coefficients unpublished.
    let mut coefficients = CheckedPendingCoefficients::new();
    let epoch = tenet_core::core_reset_epoch();
    let identity = std::cell::OnceCell::new();
    let destination = std::cell::OnceCell::new();
    let source_proof = std::cell::OnceCell::new();
    let (codomain_axes, domain_axes) = tree_transform_operation_axes(&operation);
    let structure =
        FusionTreeHomSpace::prepare_complete_coupled_subblock_structure_generic_checked_with::<
            _,
            CheckedGenericPlanError<P::Error>,
            _,
        >(provider, |provider_identity| {
            let actual = source.validate_transformed_generic_checked_identity(provider_identity)?;
            if provider.fusion_style() != tenet_core::FusionStyleKind::Generic {
                return Err(tenet_core::CoreError::UnsupportedFusionStyle {
                    expected: tenet_core::FusionStyleKind::Generic,
                    actual: provider.fusion_style(),
                }
                .into());
            }
            if operand.storage_conjugate() {
                if !Arc::ptr_eq(logical_space.provider_arc(), storage_space.provider_arc()) {
                    return Err(OperationError::StructureMismatch {
                        tensor: "checked adjoint provider",
                    }
                    .into());
                }
                if storage_source
                    .validate_transformed_generic_checked_identity(provider_identity)?
                    != actual
                {
                    return Err(OperationError::StructureMismatch {
                        tensor: "checked adjoint identity",
                    }
                    .into());
                }
                if source.nout() != storage_source.nin()
                    || source.nin() != storage_source.nout()
                    || source.homspace().codomain() != storage_source.homspace().domain()
                    || source.homspace().domain() != storage_source.homspace().codomain()
                {
                    return Err(OperationError::StructureMismatch {
                        tensor: "checked adjoint relation",
                    }
                    .into());
                }
                validate_oriented_fusion_layout(source.structure(), operand)?;
            }
            let proof = validate_checked_generic_tree_pair_plan_preflight(
                provider,
                &operation,
                source.structure(),
            )?;
            let homspace = source.homspace().try_permute_generic_checked(
                provider,
                codomain_axes,
                domain_axes,
            )?;
            identity.set(actual).expect("checked producer runs once");
            if source_proof.set(proof).is_err() {
                unreachable!("checked producer runs once");
            }
            destination
                .set(homspace.clone())
                .expect("checked producer runs once");
            Ok(homspace)
        })?;
    let identity = identity
        .into_inner()
        .expect("successful checked producer records identity");
    let prepared = PreparedCheckedGenericDynamicSpace::from_complete_parts(
        codomain_axes.len(),
        domain_axes.len(),
        destination
            .into_inner()
            .expect("successful checked producer records destination"),
        structure,
        identity.clone(),
    );
    let source_proof = source_proof
        .into_inner()
        .expect("successful checked producer records source proof");
    let logical_source_key = operand
        .storage_conjugate()
        .then(|| source.structure().as_ref());
    let key = |destination: &BlockStructure| {
        CompletedTransformerKey::new::<P::Scalar>(
            identity.clone(),
            TransformerMode::CheckedGeneric,
            TreeTransformScope::TreePair,
            &operation,
            FusionTreePairOrientation::Direct,
            OrientedBasisOrder::Canonical,
            operand.storage_conjugate(),
            logical_source_key,
            destination,
            storage_source.structure(),
        )
    };
    let dst_preview = Arc::new(prepared.structure().clone());
    // A complete-cache hit makes the preview canonical, so its key can hit;
    // a staged candidate's fresh id never does, and is never published.
    let cached =
        lookup_bound::<P::Scalar>(&key(&dst_preview), &dst_preview, storage_source.structure());
    let compiled = cached.is_none();
    let replay = match cached {
        Some(replay) => replay,
        None => {
            // Why the logical source's groups key the coefficients: the plan
            // is built on it; storage conjugation applies at binding.
            let reuse = CoefficientGroupReuse::<P::Scalar>::new(
                identity.clone(),
                TransformerMode::CheckedGeneric,
                TreeTransformScope::TreePair,
                &operation,
                FusionTreePairOrientation::Direct,
            );
            let plan = build_checked_generic_tree_pair_transform_group_plan_validated(
                operation.clone(),
                &source_proof,
                &reuse,
            )?;
            coefficients.stage(reuse.into_pending());
            if operand.storage_conjugate() {
                let logical_to_storage_block = |logical_index| {
                    let logical_block = source.structure().block(logical_index)?;
                    let BlockKey::FusionTree(logical_key) = logical_block.key() else {
                        return Err(OperationError::StructureMismatch {
                            tensor: "checked logical source",
                        });
                    };
                    operand.storage_block_index(logical_key)
                };
                plan.compile_shared_structures_with_storage_mapping(
                    Arc::clone(&dst_preview),
                    source.structure(),
                    Arc::clone(storage_source.structure()),
                    logical_to_storage_block,
                    |axis| operand.storage_axis(axis),
                    true,
                )?
            } else {
                plan.compile_shared_structures_with_storage_conjugation(
                    Arc::clone(&dst_preview),
                    Arc::clone(source.structure()),
                    false,
                )?
            }
        }
    };
    let mut dst_data = vec![D::zero(); prepared.required_len()];

    context.backend.tree_transform_structure_into_raw(
        &mut context.workspace,
        &replay,
        &dst_preview,
        storage_source.structure(),
        &mut dst_data,
        src_data,
        alpha,
        D::zero(),
    )?;
    let dst_space = logical_space.commit_final_homspace_generic_bound_checked(prepared)?;
    coefficients.flush();
    // Publication follows commit: only now are the destination's ids
    // committed, and only a resident (canonical) destination is keyed.
    let committed = dst_space.space().structure();
    if compiled
        && committed.as_ref() == dst_preview.as_ref()
        && publishable(
            [committed.as_ref(), storage_source.structure().as_ref()]
                .into_iter()
                .chain(logical_source_key),
        )
    {
        publish_committed(&key(committed), replay.replay_core(), epoch);
    }
    Ok((dst_space, dst_data))
}

#[allow(clippy::too_many_arguments)]
#[inline]
fn replay_structure_overwrite<D, C, B>(
    backend: &mut B,
    workspace: &mut B::Workspace,
    structure: &TreeTransformStructure<C>,
    dst_structure: &Arc<BlockStructure>,
    src_structure: &Arc<BlockStructure>,
    dst_data: &mut [D],
    src_data: &[D],
    alpha: D,
    profile: Option<&mut TreeTransformReplayProfile>,
) -> Result<(), OperationError>
where
    D: TreeTransformScalar,
    C: Copy,
    B: TreeTransformBackend<D, C>,
{
    match profile {
        Some(profile) => backend.tree_transform_structure_overwrite_into_raw_profiled(
            workspace,
            structure,
            dst_structure,
            src_structure,
            dst_data,
            src_data,
            alpha,
            &[],
            profile,
        ),
        None => backend.tree_transform_structure_overwrite_into_raw(
            workspace,
            structure,
            dst_structure,
            src_structure,
            dst_data,
            src_data,
            alpha,
            &[],
        ),
    }
}

#[derive(Debug)]
pub struct TreeTransformExecutionContext<D, RuleKey, C = D, B = DenseTreeTransformOperations>
where
    D: TreeTransformScalar,
    C: Copy,
    B: TreeTransformBackend<D, C>,
{
    backend: B,
    workspace: B::Workspace,
    planning: TreeTransformPlanning,
    rule_key: PhantomData<fn() -> RuleKey>,
}

impl<D, RuleKey, C, B> TreeTransformExecutionContext<D, RuleKey, C, B>
where
    D: TreeTransformScalar,
    C: Copy,
    B: TreeTransformBackend<D, C>,
{
    /// Completed tree transformers are resolved through the process-global
    /// cache (`tenet::cache`); a context owns only its backend and workspace.
    pub fn with_parts(backend: B, workspace: B::Workspace) -> Self {
        Self {
            backend,
            workspace,
            planning: TreeTransformPlanning::default(),
            rule_key: PhantomData,
        }
    }

    #[inline]
    pub fn backend(&self) -> &B {
        &self.backend
    }

    #[inline]
    pub fn backend_mut(&mut self) -> &mut B {
        &mut self.backend
    }

    #[inline]
    pub fn workspace(&self) -> &B::Workspace {
        &self.workspace
    }

    #[inline]
    pub(crate) fn backend_workspace_mut(&mut self) -> (&mut B, &mut B::Workspace) {
        (&mut self.backend, &mut self.workspace)
    }

    pub fn into_parts(self) -> (B, B::Workspace) {
        (self.backend, self.workspace)
    }
}

impl<D, RuleKey, C>
    TreeTransformExecutionContext<D, RuleKey, C, DenseTreeTransformOperations<DefaultDenseExecutor>>
where
    D: crate::DenseRecouplingScalar + RecouplingCoefficientAction<C> + crate::ConjugateValue,
    C: 'static + Copy + Clone + Add<Output = C> + Mul<Output = C> + Zero + Send + Sync,
    RuleKey: 'static + Clone + Eq + Hash + Send + Sync,
{
    /// Attempts the built-in uninitialised writer used by owned tensor
    /// transforms. `Ok(None)` means the proof was unavailable and no output
    /// was allocated. The writer runs under the backend's replay worker
    /// count: the parallel schedule splits the destination on its compiled
    /// slice-disjoint boundaries, so thread count never forces the
    /// zero-then-replay fallback.
    ///
    /// This concrete cross-crate entrypoint is internal and unstable despite
    /// being public for `tenet`; downstream callers must not rely on it.
    #[doc(hidden)]
    #[allow(clippy::too_many_arguments)]
    pub fn try_tree_transform_dyn_overwrite_owned<R>(
        &mut self,
        rule: &R,
        operation: &TreeTransformOperation,
        dst_structure: &Arc<BlockStructure>,
        src_structure: &Arc<BlockStructure>,
        nout: usize,
        src_data: &[D],
        alpha: D,
    ) -> Result<Option<Vec<D>>, OperationError>
    where
        R: MultiplicityFreeRigidSymbols<Scalar = C> + TreeTransformRuleCacheKey<Key = RuleKey>,
    {
        let structure = self.tree_structure(
            rule,
            operation,
            dst_structure,
            TreeStructureSource::Stored {
                structure: src_structure,
                storage_conjugate: false,
            },
        )?;
        self.backend
            .try_tree_transform_structure_overwrite_owned_raw(
                &mut self.workspace,
                &structure,
                dst_structure,
                src_structure,
                nout,
                src_data,
                alpha,
            )
    }
}

impl<D, RuleKey, C, B> TreeTransformExecutionContext<D, RuleKey, C, B>
where
    D: TreeTransformScalar,
    C: Copy,
    B: TreeTransformBackend<D, C> + ReportsPlacement,
    B::Workspace: ReportsPlacement,
{
    #[inline]
    pub fn backend_placement(&self) -> Placement {
        self.backend.placement()
    }

    #[inline]
    pub fn workspace_placement(&self) -> Placement {
        self.workspace.placement()
    }

    #[inline]
    pub fn is_host_context(&self) -> bool {
        self.backend.is_host_placement() && self.workspace.is_host_placement()
    }
}

impl<D, RuleKey, C, B> TreeTransformExecutionContext<D, RuleKey, C, B>
where
    D: TreeTransformScalar,
    C: Copy,
    RuleKey: Clone + Eq + Hash,
    B: TreeTransformBackend<D, C>,
    B::Workspace: Default,
{
    pub fn new(backend: B) -> Self {
        Self::with_parts(backend, B::Workspace::default())
    }
}

impl<D, RuleKey, C, B> Default for TreeTransformExecutionContext<D, RuleKey, C, B>
where
    D: TreeTransformScalar,
    C: Copy,
    RuleKey: Clone + Eq + Hash,
    B: TreeTransformBackend<D, C> + Default,
    B::Workspace: Default,
{
    fn default() -> Self {
        Self::new(B::default())
    }
}

impl<D, RuleKey, C, B> TreeTransformExecutionContext<D, RuleKey, C, B>
where
    D: TreeTransformScalar,
    C: 'static + Copy + Clone + Add<Output = C> + Mul<Output = C> + Zero + Send + Sync,
    RuleKey: 'static + Clone + Eq + Hash + Send + Sync,
    B: TreeTransformBackend<D, C>,
{
    pub fn tree_transform_into<
        R,
        const DST_NOUT: usize,
        const DST_NIN: usize,
        const SRC_NOUT: usize,
        const SRC_NIN: usize,
        SDst,
        SSrc,
        DDst,
        DSrc,
    >(
        &mut self,
        rule: &R,
        operation: TreeTransformOperation,
        dst: &mut TensorMap<D, DST_NOUT, DST_NIN, SDst, DDst>,
        src: &TensorMap<D, SRC_NOUT, SRC_NIN, SSrc, DSrc>,
        alpha: D,
        beta: D,
    ) -> Result<(), OperationError>
    where
        R: MultiplicityFreeRigidSymbols<Scalar = C> + TreeTransformRuleCacheKey<Key = RuleKey>,
        DDst: HostWritableStorage<D>,
        DSrc: HostReadableStorage<D>,
    {
        let structure = self.tree_structure(
            rule,
            &operation,
            dst.structure(),
            TreeStructureSource::Stored {
                structure: src.structure(),
                storage_conjugate: false,
            },
        )?;
        self.backend.tree_transform_structure_into(
            &mut self.workspace,
            &structure,
            dst,
            src,
            alpha,
            beta,
        )
    }

    pub fn tree_transform_overwrite_into<
        R,
        const DST_NOUT: usize,
        const DST_NIN: usize,
        const SRC_NOUT: usize,
        const SRC_NIN: usize,
        SDst,
        SSrc,
        DDst,
        DSrc,
    >(
        &mut self,
        rule: &R,
        operation: TreeTransformOperation,
        dst: &mut TensorMap<D, DST_NOUT, DST_NIN, SDst, DDst>,
        src: &TensorMap<D, SRC_NOUT, SRC_NIN, SSrc, DSrc>,
        alpha: D,
    ) -> Result<(), OperationError>
    where
        R: MultiplicityFreeRigidSymbols<Scalar = C> + TreeTransformRuleCacheKey<Key = RuleKey>,
        DDst: HostWritableStorage<D>,
        DSrc: HostReadableStorage<D>,
    {
        let dst_structure = Arc::clone(dst.structure());
        let src_structure = Arc::clone(src.structure());
        self.tree_transform_overwrite_into_raw_with_storage_conjugation(
            rule,
            operation,
            &dst_structure,
            &src_structure,
            dst.data_mut(),
            src.data(),
            false,
            alpha,
        )
    }

    /// Dynamic-rank tree transform (permute / braid / transpose): operates
    /// on raw slices plus their block structures, through the same
    /// structure-compile cache as the typed facade. `dst_data` must be
    /// zero-filled (or carry the `beta`-scaled accumuland) and sized for
    /// `dst_structure.required_len()`.
    #[allow(clippy::too_many_arguments)]
    pub fn tree_transform_dyn_into<R>(
        &mut self,
        rule: &R,
        operation: TreeTransformOperation,
        dst_structure: &Arc<BlockStructure>,
        src_structure: &Arc<BlockStructure>,
        dst_data: &mut [D],
        src_data: &[D],
        alpha: D,
        beta: D,
    ) -> Result<(), OperationError>
    where
        R: MultiplicityFreeRigidSymbols<Scalar = C> + TreeTransformRuleCacheKey<Key = RuleKey>,
    {
        self.tree_transform_into_raw_with_storage_conjugation(
            rule,
            operation,
            dst_structure,
            src_structure,
            dst_data,
            src_data,
            false,
            alpha,
            beta,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn tree_transform_dyn_into_ref<R>(
        &mut self,
        rule: &R,
        operation: &TreeTransformOperation,
        dst_structure: &Arc<BlockStructure>,
        src_structure: &Arc<BlockStructure>,
        dst_data: &mut [D],
        src_data: &[D],
        alpha: D,
        beta: D,
    ) -> Result<(), OperationError>
    where
        R: MultiplicityFreeRigidSymbols<Scalar = C> + TreeTransformRuleCacheKey<Key = RuleKey>,
    {
        let structure = self.tree_structure(
            rule,
            operation,
            dst_structure,
            TreeStructureSource::Stored {
                structure: src_structure,
                storage_conjugate: false,
            },
        )?;
        self.backend.tree_transform_structure_into_raw(
            &mut self.workspace,
            &structure,
            dst_structure,
            src_structure,
            dst_data,
            src_data,
            alpha,
            beta,
        )
    }

    #[allow(clippy::too_many_arguments)]
    #[doc(hidden)]
    pub fn tree_transform_dyn_overwrite_into_ref<R>(
        &mut self,
        rule: &R,
        operation: &TreeTransformOperation,
        dst_structure: &Arc<BlockStructure>,
        src_structure: &Arc<BlockStructure>,
        dst_data: &mut [D],
        src_data: &[D],
        alpha: D,
    ) -> Result<(), OperationError>
    where
        R: MultiplicityFreeRigidSymbols<Scalar = C> + TreeTransformRuleCacheKey<Key = RuleKey>,
    {
        self.compile_and_replay_overwrite(
            |context| {
                context.tree_structure(
                    rule,
                    operation,
                    dst_structure,
                    TreeStructureSource::Stored {
                        structure: src_structure,
                        storage_conjugate: false,
                    },
                )
            },
            dst_structure,
            src_structure,
            dst_data,
            src_data,
            alpha,
        )
    }

    /// Resolves the completed [`TreeTransformStructure`] for one
    /// multiplicity-free operation through the process-global cache, without
    /// replaying it. The returned handle is cheap: a hit clones three `Arc`s.
    ///
    /// This is the seam a *non-host* executor needs: everything categorical
    /// (validation, the group plan, F/R moves, the `RuleIdentity`-keyed cache)
    /// happens here, on the host, and what comes back is layout-only. A device
    /// replay compiles through this before it takes its device lease, so the
    /// host caches and the device lock are never held at the same time.
    ///
    /// `storage_conjugate` is fixed to `false`: a lazy adjoint is lowered onto
    /// its parent by the caller, exactly as the host facade does.
    ///
    /// This concrete cross-crate entrypoint is internal and unstable despite
    /// being public for `tenet`; downstream callers must not rely on it.
    #[doc(hidden)]
    pub fn compile_tree_pair_structure<R>(
        &mut self,
        rule: &R,
        operation: &TreeTransformOperation,
        dst_structure: &Arc<BlockStructure>,
        src_structure: &Arc<BlockStructure>,
    ) -> Result<TreeTransformStructure<C>, OperationError>
    where
        R: MultiplicityFreeRigidSymbols<Scalar = C> + TreeTransformRuleCacheKey<Key = RuleKey>,
    {
        self.tree_structure(
            rule,
            operation,
            dst_structure,
            TreeStructureSource::Stored {
                structure: src_structure,
                storage_conjugate: false,
            },
        )
    }

    /// The one tree-structure resolution of the planner and the eager
    /// transform entries: the multiplicity-free
    /// [`PlanningAlgebra::tree_structure`] with this context's planning state.
    pub(crate) fn tree_structure<R>(
        &mut self,
        rule: &R,
        operation: &TreeTransformOperation,
        dst_structure: &Arc<BlockStructure>,
        src: TreeStructureSource<'_>,
    ) -> Result<TreeTransformStructure<C>, OperationError>
    where
        R: MultiplicityFreeRigidSymbols<Scalar = C> + TreeTransformRuleCacheKey<Key = RuleKey>,
    {
        self.planning
            .set_recoupling_threads(self.backend.recoupling_threads());
        <MultiplicityFreeAdmissionMode as PlanningAlgebra<R>>::tree_structure(
            &mut self.planning,
            rule,
            operation,
            dst_structure,
            src,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn tree_transform_into_raw_with_storage_conjugation<R>(
        &mut self,
        rule: &R,
        operation: TreeTransformOperation,
        dst_structure: &std::sync::Arc<BlockStructure>,
        src_structure: &std::sync::Arc<BlockStructure>,
        dst_data: &mut [D],
        src_data: &[D],
        storage_conjugate: bool,
        alpha: D,
        beta: D,
    ) -> Result<(), OperationError>
    where
        R: MultiplicityFreeRigidSymbols<Scalar = C> + TreeTransformRuleCacheKey<Key = RuleKey>,
    {
        let structure = self.tree_structure(
            rule,
            &operation,
            dst_structure,
            TreeStructureSource::Stored {
                structure: src_structure,
                storage_conjugate,
            },
        )?;
        self.backend.tree_transform_structure_into_raw(
            &mut self.workspace,
            &structure,
            dst_structure,
            src_structure,
            dst_data,
            src_data,
            alpha,
            beta,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn tree_transform_into_raw_oriented<R>(
        &mut self,
        rule: &R,
        operation: TreeTransformOperation,
        dst_structure: &std::sync::Arc<BlockStructure>,
        source: &crate::contract::FusionOperandLayout<'_>,
        dst_data: &mut [D],
        src_data: &[D],
        alpha: D,
        beta: D,
    ) -> Result<(), OperationError>
    where
        R: MultiplicityFreeRigidSymbols<Scalar = C> + TreeTransformRuleCacheKey<Key = RuleKey>,
    {
        let structure = self.tree_structure(
            rule,
            &operation,
            dst_structure,
            TreeStructureSource::Oriented(source),
        )?;
        self.backend.tree_transform_structure_into_raw(
            &mut self.workspace,
            &structure,
            dst_structure,
            source.storage_space().structure(),
            dst_data,
            src_data,
            alpha,
            beta,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn tree_transform_overwrite_into_raw_with_storage_conjugation<R>(
        &mut self,
        rule: &R,
        operation: TreeTransformOperation,
        dst_structure: &std::sync::Arc<BlockStructure>,
        src_structure: &std::sync::Arc<BlockStructure>,
        dst_data: &mut [D],
        src_data: &[D],
        storage_conjugate: bool,
        alpha: D,
    ) -> Result<(), OperationError>
    where
        R: MultiplicityFreeRigidSymbols<Scalar = C> + TreeTransformRuleCacheKey<Key = RuleKey>,
    {
        self.compile_and_replay_overwrite(
            |context| {
                context.tree_structure(
                    rule,
                    &operation,
                    dst_structure,
                    TreeStructureSource::Stored {
                        structure: src_structure,
                        storage_conjugate,
                    },
                )
            },
            dst_structure,
            src_structure,
            dst_data,
            src_data,
            alpha,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn compile_and_replay_overwrite<F>(
        &mut self,
        compile: F,
        dst_structure: &Arc<BlockStructure>,
        src_structure: &Arc<BlockStructure>,
        dst_data: &mut [D],
        src_data: &[D],
        alpha: D,
    ) -> Result<(), OperationError>
    where
        F: FnOnce(&mut Self) -> Result<TreeTransformStructure<C>, OperationError>,
    {
        let structure = compile(self)?;
        let Self {
            backend, workspace, ..
        } = self;
        replay_structure_overwrite(
            backend,
            workspace,
            &structure,
            dst_structure,
            src_structure,
            dst_data,
            src_data,
            alpha,
            None,
        )
    }

    /// Replays an already resolved transformer, beta-accumulating: the
    /// typed exact-layout hit's entry.
    ///
    /// This concrete cross-crate entrypoint is internal and unstable despite
    /// being public for `tenet`; downstream callers must not rely on it.
    #[doc(hidden)]
    #[allow(clippy::too_many_arguments)]
    pub fn replay_tree_transform_dyn_into(
        &mut self,
        structure: &TreeTransformStructure<C>,
        dst_structure: &Arc<BlockStructure>,
        src_structure: &Arc<BlockStructure>,
        dst_data: &mut [D],
        src_data: &[D],
        alpha: D,
        beta: D,
    ) -> Result<(), OperationError> {
        self.tree_transform_structure_into_raw(
            structure,
            dst_structure,
            src_structure,
            dst_data,
            src_data,
            alpha,
            beta,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn tree_transform_structure_into_raw(
        &mut self,
        structure: &TreeTransformStructure<C>,
        dst_structure: &Arc<BlockStructure>,
        src_structure: &Arc<BlockStructure>,
        dst_data: &mut [D],
        src_data: &[D],
        alpha: D,
        beta: D,
    ) -> Result<(), OperationError> {
        let Self {
            backend, workspace, ..
        } = self;
        backend.tree_transform_structure_into_raw(
            workspace,
            structure,
            dst_structure,
            src_structure,
            dst_data,
            src_data,
            alpha,
            beta,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn tree_transform_structure_overwrite_into_raw(
        &mut self,
        structure: &TreeTransformStructure<C>,
        dst_structure: &Arc<BlockStructure>,
        src_structure: &Arc<BlockStructure>,
        dst_data: &mut [D],
        src_data: &[D],
        alpha: D,
    ) -> Result<(), OperationError> {
        let Self {
            backend, workspace, ..
        } = self;
        replay_structure_overwrite(
            backend,
            workspace,
            structure,
            dst_structure,
            src_structure,
            dst_data,
            src_data,
            alpha,
            None,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn tree_transform_structure_into_raw_profiled(
        &mut self,
        structure: &TreeTransformStructure<C>,
        dst_structure: &Arc<BlockStructure>,
        src_structure: &Arc<BlockStructure>,
        dst_data: &mut [D],
        src_data: &[D],
        alpha: D,
        beta: D,
        profile: &mut TreeTransformReplayProfile,
    ) -> Result<(), OperationError> {
        let Self {
            backend, workspace, ..
        } = self;
        backend.tree_transform_structure_into_raw_profiled(
            workspace,
            structure,
            dst_structure,
            src_structure,
            dst_data,
            src_data,
            alpha,
            beta,
            profile,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn tree_transform_structure_overwrite_into_raw_profiled(
        &mut self,
        structure: &TreeTransformStructure<C>,
        dst_structure: &Arc<BlockStructure>,
        src_structure: &Arc<BlockStructure>,
        dst_data: &mut [D],
        src_data: &[D],
        alpha: D,
        profile: &mut TreeTransformReplayProfile,
    ) -> Result<(), OperationError> {
        let Self {
            backend, workspace, ..
        } = self;
        replay_structure_overwrite(
            backend,
            workspace,
            structure,
            dst_structure,
            src_structure,
            dst_data,
            src_data,
            alpha,
            Some(profile),
        )
    }

    pub fn all_codomain_tree_transform_into<
        R,
        const DST_NOUT: usize,
        const DST_NIN: usize,
        const SRC_NOUT: usize,
        const SRC_NIN: usize,
        SDst,
        SSrc,
        DDst,
        DSrc,
    >(
        &mut self,
        rule: &R,
        operation: TreeTransformOperation,
        dst: &mut TensorMap<D, DST_NOUT, DST_NIN, SDst, DDst>,
        src: &TensorMap<D, SRC_NOUT, SRC_NIN, SSrc, DSrc>,
        alpha: D,
        beta: D,
    ) -> Result<(), OperationError>
    where
        R: MultiplicityFreeFusionSymbols<Scalar = C>
            + TreeTransformRuleCacheKey<Key = RuleKey>
            + Sync,
        DDst: HostWritableStorage<D>,
        DSrc: HostReadableStorage<D>,
    {
        let Self {
            backend,
            workspace,
            planning,
            ..
        } = self;
        planning.set_recoupling_threads(backend.recoupling_threads());
        let structure =
            planning.resolve_all_codomain(rule, &operation, dst.structure(), src.structure())?;
        backend.tree_transform_structure_into(workspace, &structure, dst, src, alpha, beta)
    }
}
