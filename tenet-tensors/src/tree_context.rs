use core::ops::{Add, Mul};
use std::hash::Hash;
use std::marker::PhantomData;
use std::sync::Arc;

use num_traits::Zero;
use tenet_core::{
    BlockStructure, CheckedGenericAdmissionMode, CheckedGenericRigidSymbols, HostReadableStorage,
    HostWritableStorage, MultiplicityFreeAdmissionMode, MultiplicityFreeFusionSymbols,
    MultiplicityFreeRigidSymbols, Placement, RuleIdentity, TensorMap,
};

use crate::contract::{BoundDynamicFusionMapSpace, FusionOperand};
use crate::mode::{PlanningAlgebra, StagedTransform, TreeStructureSource};
use crate::tree_transform::{
    CheckedGenericPlanError, TreeTransformOperation, TreeTransformPlanning,
    TreeTransformRuleCacheKey,
};
use crate::{
    RecouplingCoefficientAction, ReportsPlacement, TreeTransformReplayProfile,
    TreeTransformStructure,
};
use tenet_dense::DefaultDenseExecutor;
use tenet_operations::OperationError;
use tenet_operations::TreeTransformScalar;
use tenet_operations::{DenseTreeTransformOperations, TreeTransformBackend};

#[allow(clippy::too_many_arguments)]
#[inline]
pub(crate) fn replay_structure_overwrite<D, C, B>(
    backend: &mut B,
    workspace: &mut B::Workspace,
    structure: &TreeTransformStructure<C>,
    dst_structure: &Arc<BlockStructure>,
    src_structure: &Arc<BlockStructure>,
    dst_data: &mut [D],
    src_data: &[D],
    alpha: D,
    destination_scales: &[(usize, C)],
    profile: Option<&mut TreeTransformReplayProfile>,
) -> Result<(), OperationError>
where
    D: TreeTransformScalar + RecouplingCoefficientAction<C>,
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
            destination_scales,
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
            destination_scales,
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

impl<D, C>
    TreeTransformExecutionContext<
        D,
        RuleIdentity,
        C,
        DenseTreeTransformOperations<DefaultDenseExecutor>,
    >
where
    D: crate::DenseRecouplingScalar + RecouplingCoefficientAction<C> + crate::ConjugateValue,
    C: 'static + Copy + Clone + Add<Output = C> + Mul<Output = C> + Zero + Send + Sync,
{
    /// The owned multiplicity-free permute, braid, transpose or repartition
    /// `alpha · operation(src)` of a direct source, into a fresh payload.
    ///
    /// This concrete cross-crate entrypoint is internal and unstable despite
    /// being public for `tenet`; downstream callers must not rely on it.
    #[doc(hidden)]
    pub fn tree_transform_owned_multiplicity_free_in<R>(
        &mut self,
        src: &BoundDynamicFusionMapSpace<R>,
        src_data: &[D],
        operation: &TreeTransformOperation,
        alpha: D,
    ) -> Result<(BoundDynamicFusionMapSpace<R>, Vec<D>), OperationError>
    where
        R: MultiplicityFreeRigidSymbols<Scalar = C> + TreeTransformRuleCacheKey<Key = RuleIdentity>,
    {
        self.tree_transform_owned_in::<MultiplicityFreeAdmissionMode, R>(
            src, None, src_data, operation, alpha,
        )
    }

    /// The one owned eager tree transform of both modes: stage the
    /// destination (checked: admit and preflight the source), resolve the
    /// structure with the staging's proof, write it through the built-in
    /// overwrite writer or else zero-fill and replay, then commit and
    /// publish.
    fn tree_transform_owned_in<M, R>(
        &mut self,
        logical: &BoundDynamicFusionMapSpace<R>,
        adjoint_of: Option<&BoundDynamicFusionMapSpace<R>>,
        src_data: &[D],
        operation: &TreeTransformOperation,
        alpha: D,
    ) -> Result<(BoundDynamicFusionMapSpace<R>, Vec<D>), M::Error>
    where
        M: PlanningAlgebra<R, Scalar = C>,
    {
        let StagedTransform {
            preview,
            nout,
            mut stage,
            proof,
        } = M::stage_transform(logical, adjoint_of, src_data.len(), operation)?;
        let (source, src_structure) = match adjoint_of {
            None => {
                let structure = logical.space().structure();
                (
                    TreeStructureSource::Stored {
                        structure,
                        storage_conjugate: false,
                    },
                    structure,
                )
            }
            Some(parent) => (
                TreeStructureSource::StorageMapped {
                    logical: logical.space().structure(),
                    operand: FusionOperand::adjoint(parent.space()),
                },
                parent.space().structure(),
            ),
        };
        let structure = M::tree_structure_with(
            M::transform_cache(self.planning(), &mut stage),
            logical.provider(),
            operation,
            &preview,
            source,
            Some(&proof),
        )?;
        let data = match self
            .backend
            .try_tree_transform_structure_overwrite_owned_raw(
                &mut self.workspace,
                &structure,
                &preview,
                src_structure,
                nout,
                src_data,
                alpha,
            )? {
            Some(data) => data,
            None => {
                let mut data =
                    vec![D::zero(); preview.required_len().map_err(OperationError::from)?];
                self.tree_transform_structure_into_raw(
                    &structure,
                    &preview,
                    src_structure,
                    &mut data,
                    src_data,
                    alpha,
                    D::zero(),
                )?;
                data
            }
        };
        let destination = M::commit_transform(logical, stage, preview)?;
        Ok((destination, data))
    }
}

impl<D>
    TreeTransformExecutionContext<
        D,
        RuleIdentity,
        f64,
        DenseTreeTransformOperations<DefaultDenseExecutor>,
    >
where
    D: crate::DenseRecouplingScalar + RecouplingCoefficientAction<f64> + crate::ConjugateValue,
{
    /// The owned checked Generic permute, braid, transpose or repartition
    /// `alpha · operation(logical)`, into a fresh payload. `adjoint_of` is
    /// `None` for a direct source stored as `logical`, or the parent whose
    /// storage `src_data` is when `logical` is that parent's lazy adjoint.
    ///
    /// Provider queries and replay compilation finish against an uninterned
    /// destination preview; the destination structure, composed coefficients
    /// and completed transformer become visible only after those fallible
    /// stages succeed.
    ///
    /// This concrete cross-crate entrypoint is internal and unstable despite
    /// being public for `tenet`; downstream callers must not rely on it.
    #[doc(hidden)]
    #[allow(clippy::type_complexity)]
    pub fn tree_transform_owned_checked_generic_in<R>(
        &mut self,
        logical: &BoundDynamicFusionMapSpace<R>,
        adjoint_of: Option<&BoundDynamicFusionMapSpace<R>>,
        src_data: &[D],
        operation: &TreeTransformOperation,
        alpha: D,
    ) -> Result<(BoundDynamicFusionMapSpace<R>, Vec<D>), CheckedGenericPlanError<R::Error>>
    where
        R: CheckedGenericRigidSymbols<Scalar = f64>,
    {
        self.tree_transform_owned_in::<CheckedGenericAdmissionMode, R>(
            logical, adjoint_of, src_data, operation, alpha,
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
        D: RecouplingCoefficientAction<C>,
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
        D: RecouplingCoefficientAction<C>,
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
        <MultiplicityFreeAdmissionMode as PlanningAlgebra<R>>::tree_structure(
            self.planning(),
            rule,
            operation,
            dst_structure,
            src,
        )
    }

    /// This context's multiplicity-free planning state, set to recouple on
    /// its backend's threads: the [`PlanningAlgebra::StructureCache`] a
    /// multiplicity-free resolution scope plans with.
    pub(crate) fn planning(&mut self) -> &mut TreeTransformPlanning {
        self.planning
            .set_recoupling_threads(self.backend.recoupling_threads());
        &mut self.planning
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
        D: RecouplingCoefficientAction<C>,
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
        D: RecouplingCoefficientAction<C>,
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
            &[],
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

    /// Overwrite replay; the move writing destination block `b` is scaled by
    /// `θ_b` from `destination_scales` (see
    /// [`TreeTransformBackend::tree_transform_structure_overwrite_into_raw`]).
    #[cfg(test)]
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn tree_transform_structure_overwrite_into_raw(
        &mut self,
        structure: &TreeTransformStructure<C>,
        dst_structure: &Arc<BlockStructure>,
        src_structure: &Arc<BlockStructure>,
        dst_data: &mut [D],
        src_data: &[D],
        alpha: D,
        destination_scales: &[(usize, C)],
    ) -> Result<(), OperationError>
    where
        D: RecouplingCoefficientAction<C>,
    {
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
            destination_scales,
            None,
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
