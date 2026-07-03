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
    tensortrace_fusion_structure_with_strided_kernel, tensortrace_structure_with_strided_kernel,
    tree_transform_structure_with_strided_kernel,
    tree_transform_structure_with_structural_recoupling, ConjugateValue, DenseRecouplingScalar,
    OperationError, RecouplingCoefficientAction, StridedHostKernelAdapter, TensorAddStructure,
    TensorTraceFusionStructure, TensorTraceStructure, TreeTransformReplayProfile,
    TreeTransformScalar, TreeTransformStructure, TreeTransformWorkspace,
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

    fn tensortrace_structure_into<
        T,
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
        allocator: &mut Self::Allocator,
        structure: &TensorTraceStructure,
        dst: &mut TensorMap<T, DST_NOUT, DST_NIN, SDst, DDst>,
        src: &TensorMap<T, SRC_NOUT, SRC_NIN, SSrc, DSrc>,
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

    fn tensortrace_fusion_structure_into<
        T,
        C,
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
        allocator: &mut Self::Allocator,
        structure: &TensorTraceFusionStructure<C>,
        dst: &mut TensorMap<T, DST_NOUT, DST_NIN, SDst, DDst>,
        src: &TensorMap<T, SRC_NOUT, SRC_NIN, SSrc, DSrc>,
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
            + RecouplingCoefficientAction<C>
            + strided_kernel::MaybeSendSync,
        C: Copy,
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

#[derive(Debug)]
pub struct DenseTreeTransformOperations<E = DefaultDenseExecutor> {
    dense: E,
}

impl DenseTreeTransformOperations<DefaultDenseExecutor> {
    pub fn default_executor() -> Self {
        Self {
            dense: DefaultDenseExecutor::new(),
        }
    }
}

impl<E> DenseTreeTransformOperations<E> {
    pub fn new(dense: E) -> Self {
        Self { dense }
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

    fn tensortrace_structure_into<
        T,
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
        allocator: &mut Self::Allocator,
        structure: &TensorTraceStructure,
        dst: &mut TensorMap<T, DST_NOUT, DST_NIN, SDst, DDst>,
        src: &TensorMap<T, SRC_NOUT, SRC_NIN, SSrc, DSrc>,
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
        tensortrace_structure_with_strided_kernel(allocator, structure, dst, src, alpha, beta)
    }

    fn tensortrace_fusion_structure_into<
        T,
        C,
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
        allocator: &mut Self::Allocator,
        structure: &TensorTraceFusionStructure<C>,
        dst: &mut TensorMap<T, DST_NOUT, DST_NIN, SDst, DDst>,
        src: &TensorMap<T, SRC_NOUT, SRC_NIN, SSrc, DSrc>,
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
            + RecouplingCoefficientAction<C>
            + strided_kernel::MaybeSendSync,
        C: Copy,
        DDst: HostWritableStorage<T>,
        DSrc: HostReadableStorage<T>,
    {
        tensortrace_fusion_structure_with_strided_kernel(
            allocator, structure, dst, src, alpha, beta,
        )
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
        tree_transform_structure_with_structural_recoupling(
            &mut StridedHostKernelAdapter,
            &mut self.dense,
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
