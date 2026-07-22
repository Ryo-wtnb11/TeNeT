use core::ops::{Add, Mul};
use std::hash::Hash;
use std::sync::Arc;

use num_traits::Zero;
use tenet_core::{
    BlockStructure, GenericBraidScalar, GenericRigidSymbols, HostReadableStorage,
    HostWritableStorage, MultiplicityFreeFusionSymbols, MultiplicityFreeRigidSymbols, Placement,
    ScratchStorage, SimilarStorage, TensorMap,
};

use crate::cache::OperationCachePolicy;
use crate::storage_scratch::StorageTreeTransformWorkspace;
use crate::tree_transform::{
    TreeTransformCache, TreeTransformOperation, TreeTransformRuleCacheKey,
};
use crate::{
    RecouplingCoefficientAction, ReportsPlacement, TreeTransformReplayProfile,
    TreeTransformStructure,
};
use tenet_dense::DefaultDenseExecutor;
use tenet_operations::tree_transform_structure_with_storage_workspace_strided_kernel;
use tenet_operations::OperationError;
use tenet_operations::TreeTransformScalar;
use tenet_operations::{DenseTreeTransformOperations, TreeTransformBackend};

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
    cache: TreeTransformCache<C, RuleKey>,
}

impl<D, RuleKey, C, B> TreeTransformExecutionContext<D, RuleKey, C, B>
where
    D: TreeTransformScalar,
    C: Copy,
    B: TreeTransformBackend<D, C>,
{
    pub fn with_parts(
        backend: B,
        workspace: B::Workspace,
        cache: TreeTransformCache<C, RuleKey>,
    ) -> Self {
        Self {
            backend,
            workspace,
            cache,
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
    pub fn cache(&self) -> &TreeTransformCache<C, RuleKey> {
        &self.cache
    }

    #[inline]
    pub fn cache_mut(&mut self) -> &mut TreeTransformCache<C, RuleKey> {
        &mut self.cache
    }

    /// Replaces this context's completed tree-transform retention policy.
    pub fn set_cache_policy(&mut self, policy: OperationCachePolicy)
    where
        RuleKey: Clone + Eq + Hash,
    {
        self.cache.set_policy(policy);
    }

    pub fn into_parts(self) -> (B, B::Workspace, TreeTransformCache<C, RuleKey>) {
        (self.backend, self.workspace, self.cache)
    }
}

impl<D, RuleKey, C>
    TreeTransformExecutionContext<D, RuleKey, C, DenseTreeTransformOperations<DefaultDenseExecutor>>
where
    D: crate::DenseRecouplingScalar + RecouplingCoefficientAction<C> + crate::ConjugateValue,
    C: 'static + Copy + Clone + Add<Output = C> + Mul<Output = C> + Zero + Send + Sync,
    RuleKey: 'static + Clone + Eq + Hash + Send + Sync,
{
    /// Attempts the serial built-in writer used by owned tensor transforms.
    /// `Ok(None)` means the proof was unavailable and no output was allocated.
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
        if self.backend.recoupling_threads() != 1 {
            return Ok(None);
        }
        self.cache.set_recoupling_threads(1);
        let structure = self
            .cache
            .get_or_compile_tree_pair_structures_with_storage_conjugation_ref(
                rule,
                operation,
                dst_structure,
                src_structure,
                false,
            )?;
        tenet_operations::try_tree_transform_structure_overwrite_owned_raw(
            self.backend.dense_mut(),
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
        Self::with_parts(backend, B::Workspace::default(), TreeTransformCache::new())
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
        let Self {
            backend,
            workspace,
            cache,
        } = self;
        // One knob: compile parallelism follows the backend's replay setting.
        cache.set_recoupling_threads(backend.recoupling_threads());
        let structure = cache.get_or_compile_tree_pair(rule, operation, dst, src)?;
        backend.tree_transform_structure_into(workspace, &structure, dst, src, alpha, beta)
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
        let Self {
            backend,
            workspace,
            cache,
        } = self;
        cache.set_recoupling_threads(backend.recoupling_threads());
        let structure = cache.get_or_compile_tree_pair_structures_with_storage_conjugation_ref(
            rule,
            operation,
            dst_structure,
            src_structure,
            false,
        )?;
        backend.tree_transform_structure_into_raw(
            workspace,
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
            |cache| {
                cache.get_or_compile_tree_pair_structures_with_storage_conjugation_ref(
                    rule,
                    operation,
                    dst_structure,
                    src_structure,
                    false,
                )
            },
            dst_structure,
            src_structure,
            dst_data,
            src_data,
            alpha,
        )
    }

    #[allow(dead_code)]
    pub(crate) fn tree_transform_into_storage_workspace<
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
        storage_workspace: &mut StorageTreeTransformWorkspace<DSrc::Similar, DDst::Similar>,
        rule: &R,
        operation: TreeTransformOperation,
        dst: &mut TensorMap<D, DST_NOUT, DST_NIN, SDst, DDst>,
        src: &TensorMap<D, SRC_NOUT, SRC_NIN, SSrc, DSrc>,
        alpha: D,
        beta: D,
    ) -> Result<(), OperationError>
    where
        R: MultiplicityFreeRigidSymbols<Scalar = C> + TreeTransformRuleCacheKey<Key = RuleKey>,
        C: Clone,
        D: RecouplingCoefficientAction<C>,
        DDst: HostWritableStorage<D> + SimilarStorage<D>,
        DSrc: HostReadableStorage<D> + SimilarStorage<D>,
        DDst::Similar: HostWritableStorage<D> + ScratchStorage<D>,
        DSrc::Similar: HostWritableStorage<D> + ScratchStorage<D>,
    {
        self.cache
            .set_recoupling_threads(self.backend.recoupling_threads());
        let structure = self
            .cache
            .get_or_compile_tree_pair(rule, operation, dst, src)?;
        tree_transform_structure_with_storage_workspace_strided_kernel(
            &mut crate::StridedHostKernelAdapter::default(),
            storage_workspace,
            &structure,
            dst,
            src,
            alpha,
            beta,
        )
    }

    pub(crate) fn get_or_compile_tree_pair_structure_with_storage_conjugation<R>(
        &mut self,
        rule: &R,
        operation: TreeTransformOperation,
        dst_structure: &Arc<BlockStructure>,
        src_structure: &Arc<BlockStructure>,
        storage_conjugate: bool,
    ) -> Result<Arc<TreeTransformStructure<C>>, OperationError>
    where
        R: MultiplicityFreeRigidSymbols<Scalar = C> + TreeTransformRuleCacheKey<Key = RuleKey>,
    {
        self.cache
            .set_recoupling_threads(self.backend.recoupling_threads());
        self.cache
            .get_or_compile_tree_pair_structures_with_storage_conjugation(
                rule,
                operation,
                dst_structure,
                src_structure,
                storage_conjugate,
            )
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn get_or_compile_tree_pair_structure_prelowered<R, FBlock, FAxis>(
        &mut self,
        rule: &R,
        operation: &TreeTransformOperation,
        dst_structure: &Arc<BlockStructure>,
        logical_src_structure: &Arc<BlockStructure>,
        storage_src_structure: &Arc<BlockStructure>,
        storage_conjugate: bool,
        logical_to_storage_block: FBlock,
        logical_to_storage_axis: FAxis,
    ) -> Result<Arc<TreeTransformStructure<C>>, OperationError>
    where
        R: MultiplicityFreeRigidSymbols<Scalar = C> + TreeTransformRuleCacheKey<Key = RuleKey>,
        FBlock: Fn(usize) -> Result<usize, OperationError>,
        FAxis: Fn(usize) -> Result<usize, OperationError>,
    {
        self.cache
            .set_recoupling_threads(self.backend.recoupling_threads());
        self.cache.get_or_compile_tree_pair_prelowered(
            rule,
            operation,
            dst_structure,
            logical_src_structure,
            storage_src_structure,
            storage_conjugate,
            logical_to_storage_block,
            logical_to_storage_axis,
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
        let Self {
            backend,
            workspace,
            cache,
        } = self;
        // One knob: compile parallelism follows the backend's replay setting.
        cache.set_recoupling_threads(backend.recoupling_threads());
        let structure = cache.get_or_compile_tree_pair_structures_with_storage_conjugation(
            rule,
            operation,
            dst_structure,
            src_structure,
            storage_conjugate,
        )?;
        backend.tree_transform_structure_into_raw(
            workspace,
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
            |cache| {
                cache.get_or_compile_tree_pair_structures_with_storage_conjugation(
                    rule,
                    operation,
                    dst_structure,
                    src_structure,
                    storage_conjugate,
                )
            },
            dst_structure,
            src_structure,
            dst_data,
            src_data,
            alpha,
        )
    }

    /// Generic-fusion (SU(3)) dynamic-rank tree transform: the raw-slice
    /// analogue of [`Self::tree_transform_dyn_into`], routed through the
    /// non-memoized generic cache sibling. This is the path the top-level
    /// `tenet::Tensor` SU(3) `permute`/`braid`/`transpose` take.
    #[allow(clippy::too_many_arguments)]
    pub fn tree_transform_dyn_into_generic<R>(
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
        R: GenericRigidSymbols<Scalar = C> + TreeTransformRuleCacheKey<Key = RuleKey>,
        C: GenericBraidScalar,
    {
        let Self {
            backend,
            workspace,
            cache,
        } = self;
        cache.set_recoupling_threads(backend.recoupling_threads());
        let structure = cache.get_or_compile_tree_pair_structures_generic(
            rule,
            operation,
            dst_structure,
            src_structure,
        )?;
        backend.tree_transform_structure_into_raw(
            workspace,
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
    pub fn tree_transform_dyn_overwrite_into_generic<R>(
        &mut self,
        rule: &R,
        operation: TreeTransformOperation,
        dst_structure: &Arc<BlockStructure>,
        src_structure: &Arc<BlockStructure>,
        dst_data: &mut [D],
        src_data: &[D],
        alpha: D,
    ) -> Result<(), OperationError>
    where
        R: GenericRigidSymbols<Scalar = C> + TreeTransformRuleCacheKey<Key = RuleKey>,
        C: GenericBraidScalar,
    {
        self.compile_and_replay_overwrite(
            |cache| {
                cache.get_or_compile_tree_pair_structures_generic(
                    rule,
                    operation,
                    dst_structure,
                    src_structure,
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
        F: FnOnce(
            &mut TreeTransformCache<C, RuleKey>,
        ) -> Result<Arc<TreeTransformStructure<C>>, OperationError>,
    {
        let Self {
            backend,
            workspace,
            cache,
        } = self;
        cache.set_recoupling_threads(backend.recoupling_threads());
        let structure = compile(cache)?;
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
            backend,
            workspace,
            cache: _,
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
            backend,
            workspace,
            cache: _,
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
            backend,
            workspace,
            cache: _,
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
            backend,
            workspace,
            cache: _,
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
            cache,
        } = self;
        let structure = cache.get_or_compile_all_codomain(rule, operation, dst, src)?;
        backend.tree_transform_structure_into(workspace, &structure, dst, src, alpha, beta)
    }
}
