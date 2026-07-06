use core::ops::{Add, Mul};
use std::sync::Arc;

use num_traits::{One, Zero};
use tenet_core::{
    BlockStructure, BlockView, BlockViewMut, HostReadableStorage, HostWritableStorage, Placement,
    TensorMap,
};
use tenet_dense::{DefaultDenseExecutor, DenseExecutor};

use crate::ReportsPlacement;

use crate::{
    copy_block_with_strided_kernel, tensoradd_structure_with_strided_kernel,
    tree_transform_structure_with_strided_kernel,
    tree_transform_structure_with_structural_recoupling, ConjugateValue, DenseRecouplingScalar,
    OperationError, RecouplingCoefficientAction, StridedHostKernelAdapter, TensorAddStructure,
    TreeTransformReplayProfile, TreeTransformScalar, TreeTransformStructure,
    TreeTransformWorkspace,
};

/// Legacy/current tree-transform execution contract over host-accessible data.
///
/// The raw replay methods take host slices. New code that specifically depends
/// on this host-slice contract may use `HostTreeTransformBackend`; future
/// placement-aware/device/MPI transform traits should not inherit from this
/// raw-slice API.
pub trait TreeTransformBackend<D, C>
where
    D: TreeTransformScalar,
    C: Copy,
{
    type Workspace;

    /// Worker count this backend is configured to run transforms with.
    /// The execution context mirrors it into the plan-compile cache, so the
    /// one configured knob drives both replay and compile parallelism;
    /// backends without a thread setting stay serial.
    fn recoupling_threads(&self) -> usize {
        1
    }

    fn tree_transform_structure_into<
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
        workspace: &mut Self::Workspace,
        structure: &TreeTransformStructure<C>,
        dst: &mut TensorMap<D, DST_NOUT, DST_NIN, SDst, DDst>,
        src: &TensorMap<D, SRC_NOUT, SRC_NIN, SSrc, DSrc>,
        alpha: D,
        beta: D,
    ) -> Result<(), OperationError>
    where
        DDst: HostWritableStorage<D>,
        DSrc: HostReadableStorage<D>;

    fn tree_transform_structure_into_raw(
        &mut self,
        workspace: &mut Self::Workspace,
        structure: &TreeTransformStructure<C>,
        dst_structure: &Arc<BlockStructure>,
        src_structure: &Arc<BlockStructure>,
        dst_data: &mut [D],
        src_data: &[D],
        alpha: D,
        beta: D,
    ) -> Result<(), OperationError>;

    #[allow(clippy::too_many_arguments)]
    fn tree_transform_structure_into_raw_profiled(
        &mut self,
        workspace: &mut Self::Workspace,
        structure: &TreeTransformStructure<C>,
        dst_structure: &Arc<BlockStructure>,
        src_structure: &Arc<BlockStructure>,
        dst_data: &mut [D],
        src_data: &[D],
        alpha: D,
        beta: D,
        profile: &mut TreeTransformReplayProfile,
    ) -> Result<(), OperationError> {
        let start = std::time::Instant::now();
        let result = self.tree_transform_structure_into_raw(
            workspace,
            structure,
            dst_structure,
            src_structure,
            dst_data,
            src_data,
            alpha,
            beta,
        );
        profile.total += start.elapsed();
        result
    }
}

/// Explicit marker for the legacy host-slice tree-transform backend family.
///
/// `TreeTransformBackend` keeps the existing method-bearing public trait for
/// source compatibility. This marker means “implements the host-slice replay
/// contract,” not necessarily “physically CPU-native.” Future device/MPI
/// transform backends should use separate placement-aware execution traits.
pub trait HostTreeTransformBackend<D, C>: TreeTransformBackend<D, C>
where
    D: TreeTransformScalar,
    C: Copy,
{
}

impl<B, D, C> HostTreeTransformBackend<D, C> for B
where
    B: TreeTransformBackend<D, C> + ?Sized,
    D: TreeTransformScalar,
    C: Copy,
{
}

pub trait TensorOperationsBackend {
    type Allocator;

    fn copy_block_into<T>(
        &mut self,
        allocator: &mut Self::Allocator,
        dst: BlockViewMut<'_, T>,
        src: BlockView<'_, T>,
    ) -> Result<(), OperationError>
    where
        T: Copy + strided_kernel::MaybeSendSync;

    fn tensoradd_structure_into<T, const NOUT: usize, const NIN: usize, S, DDst, DSrc>(
        &mut self,
        allocator: &mut Self::Allocator,
        structure: &TensorAddStructure,
        dst: &mut TensorMap<T, NOUT, NIN, S, DDst>,
        src: &TensorMap<T, NOUT, NIN, S, DSrc>,
        alpha: T,
        beta: T,
    ) -> Result<(), OperationError>
    where
        T: Copy
            + Add<T, Output = T>
            + Mul<T, Output = T>
            + PartialEq
            + Zero
            + One
            + ConjugateValue
            + strided_kernel::MaybeSendSync,
        DDst: HostWritableStorage<T>,
        DSrc: HostReadableStorage<T>;
}

/// Host scratch workspace for tensoradd/tensortrace/copy replay.
///
/// This is not a general allocator: it currently owns host-side scratch used
/// by strided replay. The legacy `HostAllocator` name remains as a type alias
/// for source compatibility.
#[derive(Clone, Debug, Default)]
pub struct HostTensorOperationsWorkspace {
    pub(crate) zero_strides: Vec<isize>,
}

impl HostTensorOperationsWorkspace {
    #[inline]
    pub fn placement(&self) -> Placement {
        Placement::Host
    }

    #[inline]
    pub fn is_host_workspace(&self) -> bool {
        self.placement() == Placement::Host
    }
}

impl ReportsPlacement for HostTensorOperationsWorkspace {
    #[inline]
    fn placement(&self) -> Placement {
        Placement::Host
    }
}

pub type HostAllocator = HostTensorOperationsWorkspace;

#[derive(Clone, Copy, Debug, Default)]
pub struct HostTensorOperations;

impl HostTensorOperations {
    #[inline]
    pub fn placement(&self) -> Placement {
        Placement::Host
    }
}

impl ReportsPlacement for HostTensorOperations {
    #[inline]
    fn placement(&self) -> Placement {
        Placement::Host
    }
}

/// Default minimum destination length before a `recoupling_threads > 1`
/// setting actually goes parallel; below it the replay stays serial.
///
/// Mirrors TensorKit's gate `length(t.data) > Strided.MINTHREADLENGTH`
/// (Strided.jl `MINTHREADLENGTH = 1 << 15`, the same value strided-kernel
/// uses internally but does not export). Configurable per backend via
/// [`DenseTreeTransformOperations::set_transform_parallel_min_len`].
pub const TRANSFORM_PARALLEL_MIN_LEN: usize = 1 << 15;

#[derive(Debug)]
pub struct DenseTreeTransformOperations<E = DefaultDenseExecutor> {
    dense: E,
    // Replay parallelism is a property of this backend: worker count for the
    // tree-transform replay phases (1 = serial, the default).
    recoupling_threads: usize,
    // Size gate paired with recoupling_threads; see TRANSFORM_PARALLEL_MIN_LEN.
    transform_parallel_min_len: usize,
}

impl DenseTreeTransformOperations<DefaultDenseExecutor> {
    pub fn default_executor() -> Self {
        Self::new(DefaultDenseExecutor::new())
    }

    pub fn with_threads(threads: usize) -> Result<Self, OperationError> {
        Ok(Self::new(
            DefaultDenseExecutor::with_threads(threads).map_err(OperationError::Dense)?,
        ))
    }
}

impl<E> DenseTreeTransformOperations<E> {
    pub fn new(dense: E) -> Self {
        Self {
            dense,
            recoupling_threads: 1,
            transform_parallel_min_len: TRANSFORM_PARALLEL_MIN_LEN,
        }
    }

    #[inline]
    pub fn placement(&self) -> Placement {
        Placement::Host
    }

    pub fn dense(&self) -> &E {
        &self.dense
    }

    pub fn dense_mut(&mut self) -> &mut E {
        &mut self.dense
    }

    /// Worker count for tree-transform replays (default 1 = serial).
    #[inline]
    pub fn recoupling_threads(&self) -> usize {
        self.recoupling_threads
    }

    /// Sets the tree-transform replay worker count. `0` is treated as `1`
    /// (serial); values `> 1` parallelize replays whose destination length
    /// exceeds [`Self::transform_parallel_min_len`].
    pub fn set_recoupling_threads(&mut self, threads: usize) {
        self.recoupling_threads = threads.max(1);
    }

    /// Minimum destination length before `recoupling_threads > 1` goes
    /// parallel (default [`TRANSFORM_PARALLEL_MIN_LEN`]).
    #[inline]
    pub fn transform_parallel_min_len(&self) -> usize {
        self.transform_parallel_min_len
    }

    pub fn set_transform_parallel_min_len(&mut self, min_len: usize) {
        self.transform_parallel_min_len = min_len;
    }

    /// Effective worker count for one replay: the configured count when
    /// parallelism is enabled and the destination is past the size gate,
    /// otherwise 1 (the untouched serial path).
    #[inline]
    fn effective_recoupling_threads(&self, dst_len: usize) -> usize {
        if self.recoupling_threads > 1 && dst_len > self.transform_parallel_min_len {
            self.recoupling_threads
        } else {
            1
        }
    }
}

impl<E> ReportsPlacement for DenseTreeTransformOperations<E> {
    #[inline]
    fn placement(&self) -> Placement {
        Placement::Host
    }
}

impl Default for DenseTreeTransformOperations<DefaultDenseExecutor> {
    fn default() -> Self {
        Self::default_executor()
    }
}

/// Explicit marker for the current host-slice tensor operation backend family.
///
/// `TensorOperationsBackend` keeps the existing method-bearing public trait for
/// source compatibility. This marker makes host-only bounds explicit without
/// forcing downstream custom backends to rewrite their existing impls.
pub trait HostTensorOperationsBackend: TensorOperationsBackend {}

impl<B> HostTensorOperationsBackend for B where B: TensorOperationsBackend + ?Sized {}

impl TensorOperationsBackend for HostTensorOperations {
    type Allocator = HostTensorOperationsWorkspace;

    fn copy_block_into<T>(
        &mut self,
        _allocator: &mut Self::Allocator,
        dst: BlockViewMut<'_, T>,
        src: BlockView<'_, T>,
    ) -> Result<(), OperationError>
    where
        T: Copy + strided_kernel::MaybeSendSync,
    {
        copy_block_with_strided_kernel(dst, src)
    }

    fn tensoradd_structure_into<T, const NOUT: usize, const NIN: usize, S, DDst, DSrc>(
        &mut self,
        allocator: &mut Self::Allocator,
        structure: &TensorAddStructure,
        dst: &mut TensorMap<T, NOUT, NIN, S, DDst>,
        src: &TensorMap<T, NOUT, NIN, S, DSrc>,
        alpha: T,
        beta: T,
    ) -> Result<(), OperationError>
    where
        T: Copy
            + Add<T, Output = T>
            + Mul<T, Output = T>
            + PartialEq
            + Zero
            + One
            + ConjugateValue
            + strided_kernel::MaybeSendSync,
        DDst: HostWritableStorage<T>,
        DSrc: HostReadableStorage<T>,
    {
        tensoradd_structure_with_strided_kernel(allocator, structure, dst, src, alpha, beta)
    }
}

impl<D, C> TreeTransformBackend<D, C> for HostTensorOperations
where
    D: Copy
        + Add<D, Output = D>
        + Mul<D, Output = D>
        + PartialEq
        + Zero
        + One
        + ConjugateValue
        + strided_kernel::MaybeSendSync
        + RecouplingCoefficientAction<C>,
    C: Copy,
{
    type Workspace = TreeTransformWorkspace<D>;

    fn tree_transform_structure_into<
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
        workspace: &mut Self::Workspace,
        structure: &TreeTransformStructure<C>,
        dst: &mut TensorMap<D, DST_NOUT, DST_NIN, SDst, DDst>,
        src: &TensorMap<D, SRC_NOUT, SRC_NIN, SSrc, DSrc>,
        alpha: D,
        beta: D,
    ) -> Result<(), OperationError>
    where
        DDst: HostWritableStorage<D>,
        DSrc: HostReadableStorage<D>,
    {
        tree_transform_structure_with_strided_kernel(
            &mut StridedHostKernelAdapter,
            workspace,
            structure,
            dst,
            src,
            alpha,
            beta,
        )
    }

    fn tree_transform_structure_into_raw(
        &mut self,
        workspace: &mut Self::Workspace,
        structure: &TreeTransformStructure<C>,
        dst_structure: &Arc<BlockStructure>,
        src_structure: &Arc<BlockStructure>,
        dst_data: &mut [D],
        src_data: &[D],
        alpha: D,
        beta: D,
    ) -> Result<(), OperationError> {
        crate::tree_transform_structure_with_strided_kernel_raw(
            &mut StridedHostKernelAdapter,
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
}

impl<E, D, C> TreeTransformBackend<D, C> for DenseTreeTransformOperations<E>
where
    E: DenseExecutor,
    D: DenseRecouplingScalar + RecouplingCoefficientAction<C> + ConjugateValue,
    C: Copy + Sync,
{
    type Workspace = TreeTransformWorkspace<D>;

    #[inline]
    fn recoupling_threads(&self) -> usize {
        DenseTreeTransformOperations::recoupling_threads(self)
    }

    fn tree_transform_structure_into<
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
        workspace: &mut Self::Workspace,
        structure: &TreeTransformStructure<C>,
        dst: &mut TensorMap<D, DST_NOUT, DST_NIN, SDst, DDst>,
        src: &TensorMap<D, SRC_NOUT, SRC_NIN, SSrc, DSrc>,
        alpha: D,
        beta: D,
    ) -> Result<(), OperationError>
    where
        DDst: HostWritableStorage<D>,
        DSrc: HostReadableStorage<D>,
    {
        let threads = self.effective_recoupling_threads(dst.storage().len());
        tree_transform_structure_with_structural_recoupling(
            &mut StridedHostKernelAdapter,
            &mut self.dense,
            workspace,
            structure,
            dst,
            src,
            alpha,
            beta,
            threads,
        )
    }

    fn tree_transform_structure_into_raw(
        &mut self,
        workspace: &mut Self::Workspace,
        structure: &TreeTransformStructure<C>,
        dst_structure: &Arc<BlockStructure>,
        src_structure: &Arc<BlockStructure>,
        dst_data: &mut [D],
        src_data: &[D],
        alpha: D,
        beta: D,
    ) -> Result<(), OperationError> {
        let threads = self.effective_recoupling_threads(dst_data.len());
        crate::tree_transform_structure_with_structural_recoupling_raw(
            &mut StridedHostKernelAdapter,
            &mut self.dense,
            workspace,
            structure,
            dst_structure,
            src_structure,
            dst_data,
            src_data,
            alpha,
            beta,
            threads,
        )
    }

    fn tree_transform_structure_into_raw_profiled(
        &mut self,
        workspace: &mut Self::Workspace,
        structure: &TreeTransformStructure<C>,
        dst_structure: &Arc<BlockStructure>,
        src_structure: &Arc<BlockStructure>,
        dst_data: &mut [D],
        src_data: &[D],
        alpha: D,
        beta: D,
        profile: &mut TreeTransformReplayProfile,
    ) -> Result<(), OperationError> {
        let threads = self.effective_recoupling_threads(dst_data.len());
        crate::tree_transform_structure_with_structural_recoupling_raw_profiled(
            &mut StridedHostKernelAdapter,
            &mut self.dense,
            workspace,
            structure,
            dst_structure,
            src_structure,
            dst_data,
            src_data,
            alpha,
            beta,
            threads,
            profile,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_tensor_operations_backend<B: TensorOperationsBackend>() {}
    fn assert_host_tensor_operations_backend<B: HostTensorOperationsBackend>() {}
    fn assert_host_tree_transform_backend<B, D, C>()
    where
        B: HostTreeTransformBackend<D, C>,
        D: TreeTransformScalar,
        C: Copy,
    {
    }

    #[test]
    fn host_tensor_operations_keeps_compatibility_backend_names() {
        let backend = HostTensorOperations;
        let tree_backend = DenseTreeTransformOperations::default();

        assert_eq!(backend.placement(), Placement::Host);
        assert_eq!(tree_backend.placement(), Placement::Host);
        assert_tensor_operations_backend::<HostTensorOperations>();
        assert_host_tensor_operations_backend::<HostTensorOperations>();
        assert_host_tree_transform_backend::<HostTensorOperations, f64, f64>();
        assert_host_tree_transform_backend::<DenseTreeTransformOperations, f64, f64>();
    }

    #[test]
    fn host_allocator_alias_keeps_workspace_shape() {
        let workspace = HostTensorOperationsWorkspace::default();
        let alias = HostAllocator::default();

        assert_eq!(workspace.placement(), Placement::Host);
        assert!(workspace.is_host_workspace());
        assert_eq!(alias.placement(), Placement::Host);
        assert_eq!(workspace.zero_strides.len(), 0);
        assert_eq!(alias.zero_strides.len(), workspace.zero_strides.len());
    }
}
