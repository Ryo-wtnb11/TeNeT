use core::ops::{Add, Mul};
use std::hash::Hash;

use num_traits::{One, Zero};
use tenet_core::{
    BlockView, BlockViewMut, CoreError, GenericBraidScalar, GenericRigidSymbols,
    HostReadableStorage, HostWritableStorage, MultiplicityFreeRigidSymbols, TensorMap,
    TensorStorage,
};

use crate::lowering::{adjoint_fusion_space_view, lower_tensoradd_source_operation};
use crate::tensortrace::{
    tensortrace_fusion_structure, tensortrace_structure, TensorTraceFusionStructure,
    TensorTraceStructure,
};
use crate::tree_context::TreeTransformExecutionContext;
use crate::tree_transform::{
    build_generic_tree_pair_transform_group_plan, build_tree_pair_transform_group_plan,
    TreeTransformOperation, TreeTransformRuleCacheKey,
};
use tenet_operations::OperationError;
use tenet_operations::TreeTransformStructure;
use tenet_operations::{tensoradd_block_with_strided_kernel, TreeTransformWorkspace};
use tenet_operations::{
    tensoradd_structure, tensoradd_structure_with_conjugation, TensorAddStructure,
};
use tenet_operations::{
    ConjugateValue, DenseRecouplingScalar, RealStructuralCoefficient, RecouplingCoefficientAction,
    TreeTransformScalar,
};
use tenet_operations::{
    DenseTreeTransformOperations, HostAllocator, HostTensorOperations, TensorOperationsBackend,
    TreeTransformBackend,
};
use tenet_operations::{OutputAxisOrder, TensorTraceAxisSpec};

pub fn tensorcopy_into<T, const NOUT: usize, const NIN: usize, S, DDst, DSrc>(
    dst: &mut TensorMap<T, NOUT, NIN, S, DDst>,
    src: &TensorMap<T, NOUT, NIN, S, DSrc>,
) -> Result<(), OperationError>
where
    T: Copy + strided_kernel::MaybeSendSync,
    DDst: HostWritableStorage<T>,
    DSrc: HostReadableStorage<T>,
{
    let mut backend = HostTensorOperations;
    let mut allocator = HostAllocator::default();
    tensorcopy_into_with(&mut backend, &mut allocator, dst, src)
}

pub fn tensorcopy_into_with<B, T, const NOUT: usize, const NIN: usize, S, DDst, DSrc>(
    backend: &mut B,
    allocator: &mut B::Allocator,
    dst: &mut TensorMap<T, NOUT, NIN, S, DDst>,
    src: &TensorMap<T, NOUT, NIN, S, DSrc>,
) -> Result<(), OperationError>
where
    B: TensorOperationsBackend,
    T: Copy + strided_kernel::MaybeSendSync,
    DDst: HostWritableStorage<T>,
    DSrc: HostReadableStorage<T>,
{
    backend.copy_block_into(allocator, dst.subblock_mut()?, src.subblock()?)
}

pub fn tensoradd_into<T, const NOUT: usize, const NIN: usize, S, DDst, DSrc>(
    dst: &mut TensorMap<T, NOUT, NIN, S, DDst>,
    src: &TensorMap<T, NOUT, NIN, S, DSrc>,
    permutation: OutputAxisOrder<'_>,
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
    let mut backend = HostTensorOperations;
    let mut allocator = HostAllocator::default();
    tensoradd_into_with(
        &mut backend,
        &mut allocator,
        dst,
        src,
        permutation,
        alpha,
        beta,
    )
}

pub fn tensoradd_into_with_conjugation<T, const NOUT: usize, const NIN: usize, S, DDst, DSrc>(
    dst: &mut TensorMap<T, NOUT, NIN, S, DDst>,
    src: &TensorMap<T, NOUT, NIN, S, DSrc>,
    permutation: OutputAxisOrder<'_>,
    source_conjugate: bool,
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
    let mut backend = HostTensorOperations;
    let mut allocator = HostAllocator::default();
    tensoradd_into_with_backend_and_conjugation(
        &mut backend,
        &mut allocator,
        dst,
        src,
        permutation,
        source_conjugate,
        alpha,
        beta,
    )
}

pub fn tensoradd_into_with<B, T, const NOUT: usize, const NIN: usize, S, DDst, DSrc>(
    backend: &mut B,
    allocator: &mut B::Allocator,
    dst: &mut TensorMap<T, NOUT, NIN, S, DDst>,
    src: &TensorMap<T, NOUT, NIN, S, DSrc>,
    permutation: OutputAxisOrder<'_>,
    alpha: T,
    beta: T,
) -> Result<(), OperationError>
where
    B: TensorOperationsBackend,
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
    let structure = tensoradd_structure(dst, src, permutation)?;
    tensoradd_execute_with(backend, allocator, &structure, dst, src, alpha, beta)
}

#[allow(clippy::too_many_arguments)]
pub fn tensoradd_into_with_backend_and_conjugation<
    B,
    T,
    const NOUT: usize,
    const NIN: usize,
    S,
    DDst,
    DSrc,
>(
    backend: &mut B,
    allocator: &mut B::Allocator,
    dst: &mut TensorMap<T, NOUT, NIN, S, DDst>,
    src: &TensorMap<T, NOUT, NIN, S, DSrc>,
    permutation: OutputAxisOrder<'_>,
    source_conjugate: bool,
    alpha: T,
    beta: T,
) -> Result<(), OperationError>
where
    B: TensorOperationsBackend,
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
    let structure = tensoradd_structure_with_conjugation(dst, src, permutation, source_conjugate)?;
    tensoradd_execute_with(backend, allocator, &structure, dst, src, alpha, beta)
}

pub fn tensoradd_execute_with<B, T, const NOUT: usize, const NIN: usize, S, DDst, DSrc>(
    backend: &mut B,
    allocator: &mut B::Allocator,
    structure: &TensorAddStructure,
    dst: &mut TensorMap<T, NOUT, NIN, S, DDst>,
    src: &TensorMap<T, NOUT, NIN, S, DSrc>,
    alpha: T,
    beta: T,
) -> Result<(), OperationError>
where
    B: TensorOperationsBackend,
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
    structure.execute_with(backend, allocator, dst, src, alpha, beta)
}

#[allow(clippy::too_many_arguments)]
pub fn tensoradd_fusion_into<
    R,
    D,
    const DST_NOUT: usize,
    const DST_NIN: usize,
    const SRC_NOUT: usize,
    const SRC_NIN: usize,
    SDst,
    SSrc,
    DDst,
    DSrc,
>(
    rule: &R,
    dst: &mut TensorMap<D, DST_NOUT, DST_NIN, SDst, DDst>,
    src: &TensorMap<D, SRC_NOUT, SRC_NIN, SSrc, DSrc>,
    operation: TreeTransformOperation,
    source_conjugate: bool,
    alpha: D,
    beta: D,
) -> Result<(), OperationError>
where
    R: MultiplicityFreeRigidSymbols,
    R::Scalar: Copy + Clone + Add<Output = R::Scalar> + Mul<Output = R::Scalar> + Zero,
    D: DenseRecouplingScalar + RecouplingCoefficientAction<R::Scalar>,
    DDst: HostWritableStorage<D>,
    DSrc: HostReadableStorage<D>,
{
    let mut backend = DenseTreeTransformOperations::default();
    let mut workspace = TreeTransformWorkspace::default();
    tensoradd_fusion_into_with(
        &mut backend,
        &mut workspace,
        rule,
        dst,
        src,
        operation,
        source_conjugate,
        alpha,
        beta,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn tensoradd_fusion_into_with<
    B,
    R,
    D,
    const DST_NOUT: usize,
    const DST_NIN: usize,
    const SRC_NOUT: usize,
    const SRC_NIN: usize,
    SDst,
    SSrc,
    DDst,
    DSrc,
>(
    backend: &mut B,
    workspace: &mut B::Workspace,
    rule: &R,
    dst: &mut TensorMap<D, DST_NOUT, DST_NIN, SDst, DDst>,
    src: &TensorMap<D, SRC_NOUT, SRC_NIN, SSrc, DSrc>,
    operation: TreeTransformOperation,
    source_conjugate: bool,
    alpha: D,
    beta: D,
) -> Result<(), OperationError>
where
    B: TreeTransformBackend<D, R::Scalar>,
    R: MultiplicityFreeRigidSymbols,
    R::Scalar: Copy + Clone + Add<Output = R::Scalar> + Mul<Output = R::Scalar> + Zero,
    D: TreeTransformScalar,
    DDst: HostWritableStorage<D>,
    DSrc: HostReadableStorage<D>,
{
    let dst_fusion = dst
        .fusion_space()
        .ok_or(OperationError::Core(CoreError::MissingFusionSpace))?;
    let src_fusion = src
        .fusion_space()
        .ok_or(OperationError::Core(CoreError::MissingFusionSpace))?;
    dst_fusion
        .validate_rule(rule)
        .map_err(OperationError::Core)?;
    src_fusion
        .validate_rule(rule)
        .map_err(OperationError::Core)?;
    if source_conjugate
        && matches!(&operation, TreeTransformOperation::Braid { .. })
        && !rule.supports_unitary_braid_dagger()
    {
        return Err(OperationError::UnsupportedTreeTransformScope {
            operation,
            message:
                "source adjoint explicit braid requires a unitary dagger-compatible braiding rule",
        });
    }
    let lowered =
        lower_tensoradd_source_operation::<SRC_NOUT, SRC_NIN>(operation, source_conjugate)?;
    if lowered.storage_conjugate() {
        let adjoint_src = adjoint_fusion_space_view(src_fusion)?;
        let dst_structure = std::sync::Arc::clone(dst.structure());
        let src_replay_structure = std::sync::Arc::clone(adjoint_src.subblock_structure());
        let plan = build_tree_pair_transform_group_plan(
            rule,
            lowered.into_operation(),
            &src_replay_structure,
        )?;
        let structure = plan.compile_structures_with_storage_conjugation(
            &dst_structure,
            &src_replay_structure,
            true,
        )?;
        backend.tree_transform_structure_into_raw(
            workspace,
            &structure,
            &dst_structure,
            &src_replay_structure,
            dst.data_mut(),
            src.data(),
            alpha,
            beta,
        )
    } else {
        tree_transform_into_with(
            backend,
            workspace,
            rule,
            lowered.into_operation(),
            dst,
            src,
            alpha,
            beta,
        )
    }
}

#[allow(clippy::too_many_arguments)]
pub fn tensoradd_fusion_into_with_context<
    B,
    R,
    D,
    RuleKey,
    const DST_NOUT: usize,
    const DST_NIN: usize,
    const SRC_NOUT: usize,
    const SRC_NIN: usize,
    SDst,
    SSrc,
    DDst,
    DSrc,
>(
    context: &mut TreeTransformExecutionContext<D, RuleKey, R::Scalar, B>,
    rule: &R,
    dst: &mut TensorMap<D, DST_NOUT, DST_NIN, SDst, DDst>,
    src: &TensorMap<D, SRC_NOUT, SRC_NIN, SSrc, DSrc>,
    operation: TreeTransformOperation,
    source_conjugate: bool,
    alpha: D,
    beta: D,
) -> Result<(), OperationError>
where
    B: TreeTransformBackend<D, R::Scalar>,
    R: MultiplicityFreeRigidSymbols + TreeTransformRuleCacheKey<Key = RuleKey>,
    RuleKey: 'static + Clone + Eq + Hash + Send + Sync,
    R::Scalar: 'static
        + Copy
        + Clone
        + Add<Output = R::Scalar>
        + Mul<Output = R::Scalar>
        + Zero
        + Send
        + Sync,
    D: TreeTransformScalar,
    DDst: HostWritableStorage<D>,
    DSrc: HostReadableStorage<D>,
{
    let dst_fusion = dst
        .fusion_space()
        .ok_or(OperationError::Core(CoreError::MissingFusionSpace))?;
    let src_fusion = src
        .fusion_space()
        .ok_or(OperationError::Core(CoreError::MissingFusionSpace))?;
    dst_fusion
        .validate_rule(rule)
        .map_err(OperationError::Core)?;
    src_fusion
        .validate_rule(rule)
        .map_err(OperationError::Core)?;
    if source_conjugate
        && matches!(&operation, TreeTransformOperation::Braid { .. })
        && !rule.supports_unitary_braid_dagger()
    {
        return Err(OperationError::UnsupportedTreeTransformScope {
            operation,
            message:
                "source adjoint explicit braid requires a unitary dagger-compatible braiding rule",
        });
    }
    let lowered =
        lower_tensoradd_source_operation::<SRC_NOUT, SRC_NIN>(operation, source_conjugate)?;
    if lowered.storage_conjugate() {
        let adjoint_src = adjoint_fusion_space_view(src_fusion)?;
        let dst_structure = std::sync::Arc::clone(dst.structure());
        let src_replay_structure = std::sync::Arc::clone(adjoint_src.subblock_structure());
        context.tree_transform_into_raw_with_storage_conjugation(
            rule,
            lowered.into_operation(),
            &dst_structure,
            &src_replay_structure,
            dst.data_mut(),
            src.data(),
            true,
            alpha,
            beta,
        )
    } else {
        tree_transform_into_with_context(
            context,
            rule,
            lowered.into_operation(),
            dst,
            src,
            alpha,
            beta,
        )
    }
}

pub fn tensortrace_into<
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
    dst: &mut TensorMap<T, DST_NOUT, DST_NIN, SDst, DDst>,
    src: &TensorMap<T, SRC_NOUT, SRC_NIN, SSrc, DSrc>,
    axes: TensorTraceAxisSpec<'_>,
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
    let mut backend = HostTensorOperations;
    let mut allocator = HostAllocator::default();
    tensortrace_into_with(&mut backend, &mut allocator, dst, src, axes, alpha, beta)
}

pub fn tensortrace_into_with<
    B,
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
    backend: &mut B,
    allocator: &mut B::Allocator,
    dst: &mut TensorMap<T, DST_NOUT, DST_NIN, SDst, DDst>,
    src: &TensorMap<T, SRC_NOUT, SRC_NIN, SSrc, DSrc>,
    axes: TensorTraceAxisSpec<'_>,
    alpha: T,
    beta: T,
) -> Result<(), OperationError>
where
    B: crate::TensorTraceOperationsBackend,
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
    let structure = tensortrace_structure(dst, src, axes)?;
    tensortrace_execute_with(backend, allocator, &structure, dst, src, alpha, beta)
}

pub fn tensortrace_execute_with<
    B,
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
    backend: &mut B,
    allocator: &mut B::Allocator,
    structure: &TensorTraceStructure,
    dst: &mut TensorMap<T, DST_NOUT, DST_NIN, SDst, DDst>,
    src: &TensorMap<T, SRC_NOUT, SRC_NIN, SSrc, DSrc>,
    alpha: T,
    beta: T,
) -> Result<(), OperationError>
where
    B: crate::TensorTraceOperationsBackend,
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
    structure.execute_with(backend, allocator, dst, src, alpha, beta)
}

pub fn tensortrace_fusion_into<
    R,
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
    rule: &R,
    dst: &mut TensorMap<T, DST_NOUT, DST_NIN, SDst, DDst>,
    src: &TensorMap<T, SRC_NOUT, SRC_NIN, SSrc, DSrc>,
    axes: TensorTraceAxisSpec<'_>,
    alpha: T,
    beta: T,
) -> Result<(), OperationError>
where
    R: MultiplicityFreeRigidSymbols,
    R::Scalar: Clone
        + Add<Output = R::Scalar>
        + Mul<Output = R::Scalar>
        + Zero
        + Copy
        + RealStructuralCoefficient,
    T: Copy
        + Add<T, Output = T>
        + Mul<T, Output = T>
        + PartialEq
        + Zero
        + One
        + ConjugateValue
        + RecouplingCoefficientAction<R::Scalar>
        + strided_kernel::MaybeSendSync,
    DDst: HostWritableStorage<T>,
    DSrc: HostReadableStorage<T>,
{
    let mut backend = HostTensorOperations;
    let mut allocator = HostAllocator::default();
    tensortrace_fusion_into_with(
        &mut backend,
        &mut allocator,
        rule,
        dst,
        src,
        axes,
        alpha,
        beta,
    )
}

pub fn tensortrace_fusion_into_with<
    B,
    R,
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
    backend: &mut B,
    allocator: &mut B::Allocator,
    rule: &R,
    dst: &mut TensorMap<T, DST_NOUT, DST_NIN, SDst, DDst>,
    src: &TensorMap<T, SRC_NOUT, SRC_NIN, SSrc, DSrc>,
    axes: TensorTraceAxisSpec<'_>,
    alpha: T,
    beta: T,
) -> Result<(), OperationError>
where
    B: crate::TensorTraceOperationsBackend,
    R: MultiplicityFreeRigidSymbols,
    R::Scalar: Clone
        + Add<Output = R::Scalar>
        + Mul<Output = R::Scalar>
        + Zero
        + Copy
        + RealStructuralCoefficient,
    T: Copy
        + Add<T, Output = T>
        + Mul<T, Output = T>
        + PartialEq
        + Zero
        + One
        + ConjugateValue
        + RecouplingCoefficientAction<R::Scalar>
        + strided_kernel::MaybeSendSync,
    DDst: HostWritableStorage<T>,
    DSrc: HostReadableStorage<T>,
{
    let structure = tensortrace_fusion_structure(rule, dst, src, axes)?;
    tensortrace_fusion_execute_with(backend, allocator, &structure, dst, src, alpha, beta)
}

pub fn tensortrace_fusion_execute_with<
    B,
    C,
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
    backend: &mut B,
    allocator: &mut B::Allocator,
    structure: &TensorTraceFusionStructure<C>,
    dst: &mut TensorMap<T, DST_NOUT, DST_NIN, SDst, DDst>,
    src: &TensorMap<T, SRC_NOUT, SRC_NIN, SSrc, DSrc>,
    alpha: T,
    beta: T,
) -> Result<(), OperationError>
where
    B: crate::TensorTraceOperationsBackend,
    C: Copy,
    T: Copy
        + Add<T, Output = T>
        + Mul<T, Output = T>
        + PartialEq
        + Zero
        + One
        + ConjugateValue
        + RecouplingCoefficientAction<C>
        + strided_kernel::MaybeSendSync,
    DDst: HostWritableStorage<T>,
    DSrc: HostReadableStorage<T>,
{
    structure.execute_with(backend, allocator, dst, src, alpha, beta)
}

pub fn tree_transform_execute_with<
    B,
    D,
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
    backend: &mut B,
    workspace: &mut B::Workspace,
    structure: &TreeTransformStructure<C>,
    dst: &mut TensorMap<D, DST_NOUT, DST_NIN, SDst, DDst>,
    src: &TensorMap<D, SRC_NOUT, SRC_NIN, SSrc, DSrc>,
    alpha: D,
    beta: D,
) -> Result<(), OperationError>
where
    B: TreeTransformBackend<D, C>,
    D: Copy
        + Add<D, Output = D>
        + Mul<D, Output = D>
        + PartialEq
        + Zero
        + One
        + ConjugateValue
        + strided_kernel::MaybeSendSync,
    C: Copy,
    DDst: HostWritableStorage<D>,
    DSrc: HostReadableStorage<D>,
{
    backend.tree_transform_structure_into(workspace, structure, dst, src, alpha, beta)
}

/// Build a replay-ready tree-pair transform structure.
///
/// This builds the replay-ready descriptor used by hot paths. It performs the
/// categorical tree-pair lowering and compiles the result against the actual
/// `dst` and `src` block structures. The returned structure can be reused with
/// [`tree_transform_execute_with`] as long as replay tensors have matching
/// structures.
pub fn tree_transform_structure<
    R,
    TDst,
    TSrc,
    const DST_NOUT: usize,
    const DST_NIN: usize,
    const SRC_NOUT: usize,
    const SRC_NIN: usize,
    SDst,
    SSrc,
    DDst,
    DSrc,
>(
    rule: &R,
    operation: TreeTransformOperation,
    dst: &TensorMap<TDst, DST_NOUT, DST_NIN, SDst, DDst>,
    src: &TensorMap<TSrc, SRC_NOUT, SRC_NIN, SSrc, DSrc>,
) -> Result<TreeTransformStructure<R::Scalar>, OperationError>
where
    R: MultiplicityFreeRigidSymbols,
    R::Scalar: Copy + Add<Output = R::Scalar> + Mul<Output = R::Scalar> + Zero,
    DDst: TensorStorage<TDst>,
    DSrc: TensorStorage<TSrc>,
{
    let plan = build_tree_pair_transform_group_plan(rule, operation, src.structure())?;
    plan.compile(dst, src)
}

/// Compile and execute a tree-pair transform in one call.
///
/// This is a convenience API. It rebuilds the transform structure on every call;
/// hot tensor-network loops should call [`tree_transform_structure`] once
/// and replay the returned structure with [`tree_transform_execute_with`].
pub fn tree_transform_into<
    R,
    D,
    const DST_NOUT: usize,
    const DST_NIN: usize,
    const SRC_NOUT: usize,
    const SRC_NIN: usize,
    SDst,
    SSrc,
    DDst,
    DSrc,
>(
    rule: &R,
    operation: TreeTransformOperation,
    dst: &mut TensorMap<D, DST_NOUT, DST_NIN, SDst, DDst>,
    src: &TensorMap<D, SRC_NOUT, SRC_NIN, SSrc, DSrc>,
    alpha: D,
    beta: D,
) -> Result<(), OperationError>
where
    R: MultiplicityFreeRigidSymbols,
    R::Scalar: Copy + Clone + Add<Output = R::Scalar> + Mul<Output = R::Scalar> + Zero,
    D: DenseRecouplingScalar + RecouplingCoefficientAction<R::Scalar>,
    DDst: HostWritableStorage<D>,
    DSrc: HostReadableStorage<D>,
{
    let mut backend = DenseTreeTransformOperations::default();
    let mut workspace = TreeTransformWorkspace::default();
    tree_transform_into_with(
        &mut backend,
        &mut workspace,
        rule,
        operation,
        dst,
        src,
        alpha,
        beta,
    )
}

/// Compile and execute a tree-pair transform with caller-owned backend/workspace.
///
/// The backend and workspace are reused, but the transform structure is still
/// rebuilt on every call. This mirrors a TensorKit-style one-call transformer
/// application with explicit execution resources, not a cached transformer.
/// Use [`tree_transform_into_with_context`] when the categorical plan and
/// replay descriptor should be cached behind a caller-owned context. Use
/// [`tree_transform_structure`] plus [`tree_transform_execute_with`] for
/// the tightest loop when the exact replay descriptor is already known.
pub fn tree_transform_into_with<
    B,
    R,
    D,
    const DST_NOUT: usize,
    const DST_NIN: usize,
    const SRC_NOUT: usize,
    const SRC_NIN: usize,
    SDst,
    SSrc,
    DDst,
    DSrc,
>(
    backend: &mut B,
    workspace: &mut B::Workspace,
    rule: &R,
    operation: TreeTransformOperation,
    dst: &mut TensorMap<D, DST_NOUT, DST_NIN, SDst, DDst>,
    src: &TensorMap<D, SRC_NOUT, SRC_NIN, SSrc, DSrc>,
    alpha: D,
    beta: D,
) -> Result<(), OperationError>
where
    B: TreeTransformBackend<D, R::Scalar>,
    R: MultiplicityFreeRigidSymbols,
    R::Scalar: Copy + Add<Output = R::Scalar> + Mul<Output = R::Scalar> + Zero,
    D: TreeTransformScalar,
    DDst: HostWritableStorage<D>,
    DSrc: HostReadableStorage<D>,
{
    let structure = tree_transform_structure(rule, operation, dst, src)?;
    tree_transform_execute_with(backend, workspace, &structure, dst, src, alpha, beta)
}

pub fn tree_transform_into_with_context<
    B,
    R,
    D,
    RuleKey,
    const DST_NOUT: usize,
    const DST_NIN: usize,
    const SRC_NOUT: usize,
    const SRC_NIN: usize,
    SDst,
    SSrc,
    DDst,
    DSrc,
>(
    context: &mut TreeTransformExecutionContext<D, RuleKey, R::Scalar, B>,
    rule: &R,
    operation: TreeTransformOperation,
    dst: &mut TensorMap<D, DST_NOUT, DST_NIN, SDst, DDst>,
    src: &TensorMap<D, SRC_NOUT, SRC_NIN, SSrc, DSrc>,
    alpha: D,
    beta: D,
) -> Result<(), OperationError>
where
    B: TreeTransformBackend<D, R::Scalar>,
    R: MultiplicityFreeRigidSymbols + TreeTransformRuleCacheKey<Key = RuleKey>,
    RuleKey: 'static + Clone + Eq + Hash + Send + Sync,
    R::Scalar: 'static
        + Copy
        + Clone
        + Add<Output = R::Scalar>
        + Mul<Output = R::Scalar>
        + Zero
        + Send
        + Sync,
    D: TreeTransformScalar,
    DDst: HostWritableStorage<D>,
    DSrc: HostReadableStorage<D>,
{
    context.tree_transform_into(rule, operation, dst, src, alpha, beta)
}

/// TensorKit `permute!`: symmetric-braiding permutation of tensor legs, written into `dst`.
///
/// Thin wrapper over [`tree_transform_into`] with
/// [`TreeTransformOperation::permute`]; see [`TreeTransformOperation::permute`]
/// for the axis-numbering convention.
pub fn permute_into<
    R,
    D,
    const DST_NOUT: usize,
    const DST_NIN: usize,
    const SRC_NOUT: usize,
    const SRC_NIN: usize,
    SDst,
    SSrc,
    DDst,
    DSrc,
>(
    rule: &R,
    codomain_permutation: impl IntoIterator<Item = usize>,
    domain_permutation: impl IntoIterator<Item = usize>,
    dst: &mut TensorMap<D, DST_NOUT, DST_NIN, SDst, DDst>,
    src: &TensorMap<D, SRC_NOUT, SRC_NIN, SSrc, DSrc>,
    alpha: D,
    beta: D,
) -> Result<(), OperationError>
where
    R: MultiplicityFreeRigidSymbols,
    R::Scalar: Copy + Clone + Add<Output = R::Scalar> + Mul<Output = R::Scalar> + Zero,
    D: DenseRecouplingScalar + RecouplingCoefficientAction<R::Scalar>,
    DDst: HostWritableStorage<D>,
    DSrc: HostReadableStorage<D>,
{
    tree_transform_into(
        rule,
        TreeTransformOperation::permute(codomain_permutation, domain_permutation),
        dst,
        src,
        alpha,
        beta,
    )
}

/// TensorKit `permute!`: symmetric-braiding permutation of tensor legs, with caller-owned backend/workspace.
///
/// Thin wrapper over [`tree_transform_into_with`] with
/// [`TreeTransformOperation::permute`].
pub fn permute_into_with<
    B,
    R,
    D,
    const DST_NOUT: usize,
    const DST_NIN: usize,
    const SRC_NOUT: usize,
    const SRC_NIN: usize,
    SDst,
    SSrc,
    DDst,
    DSrc,
>(
    backend: &mut B,
    workspace: &mut B::Workspace,
    rule: &R,
    codomain_permutation: impl IntoIterator<Item = usize>,
    domain_permutation: impl IntoIterator<Item = usize>,
    dst: &mut TensorMap<D, DST_NOUT, DST_NIN, SDst, DDst>,
    src: &TensorMap<D, SRC_NOUT, SRC_NIN, SSrc, DSrc>,
    alpha: D,
    beta: D,
) -> Result<(), OperationError>
where
    B: TreeTransformBackend<D, R::Scalar>,
    R: MultiplicityFreeRigidSymbols,
    R::Scalar: Copy + Add<Output = R::Scalar> + Mul<Output = R::Scalar> + Zero,
    D: TreeTransformScalar,
    DDst: HostWritableStorage<D>,
    DSrc: HostReadableStorage<D>,
{
    tree_transform_into_with(
        backend,
        workspace,
        rule,
        TreeTransformOperation::permute(codomain_permutation, domain_permutation),
        dst,
        src,
        alpha,
        beta,
    )
}

/// TensorKit `permute!`: symmetric-braiding permutation of tensor legs, with a caller-owned caching execution context.
///
/// Thin wrapper over [`tree_transform_into_with_context`] with
/// [`TreeTransformOperation::permute`].
pub fn permute_into_with_context<
    B,
    R,
    D,
    RuleKey,
    const DST_NOUT: usize,
    const DST_NIN: usize,
    const SRC_NOUT: usize,
    const SRC_NIN: usize,
    SDst,
    SSrc,
    DDst,
    DSrc,
>(
    context: &mut TreeTransformExecutionContext<D, RuleKey, R::Scalar, B>,
    rule: &R,
    codomain_permutation: impl IntoIterator<Item = usize>,
    domain_permutation: impl IntoIterator<Item = usize>,
    dst: &mut TensorMap<D, DST_NOUT, DST_NIN, SDst, DDst>,
    src: &TensorMap<D, SRC_NOUT, SRC_NIN, SSrc, DSrc>,
    alpha: D,
    beta: D,
) -> Result<(), OperationError>
where
    B: TreeTransformBackend<D, R::Scalar>,
    R: MultiplicityFreeRigidSymbols + TreeTransformRuleCacheKey<Key = RuleKey>,
    RuleKey: 'static + Clone + Eq + Hash + Send + Sync,
    R::Scalar: 'static
        + Copy
        + Clone
        + Add<Output = R::Scalar>
        + Mul<Output = R::Scalar>
        + Zero
        + Send
        + Sync,
    D: TreeTransformScalar,
    DDst: HostWritableStorage<D>,
    DSrc: HostReadableStorage<D>,
{
    tree_transform_into_with_context(
        context,
        rule,
        TreeTransformOperation::permute(codomain_permutation, domain_permutation),
        dst,
        src,
        alpha,
        beta,
    )
}

/// TensorKit `braid!`: explicit braid with source-axis levels, written into `dst`.
///
/// Thin wrapper over [`tree_transform_into`] with
/// [`TreeTransformOperation::braid`]; see [`TreeTransformOperation::braid`]
/// for the axis-numbering convention.
pub fn braid_into<
    R,
    D,
    const DST_NOUT: usize,
    const DST_NIN: usize,
    const SRC_NOUT: usize,
    const SRC_NIN: usize,
    SDst,
    SSrc,
    DDst,
    DSrc,
>(
    rule: &R,
    codomain_permutation: impl IntoIterator<Item = usize>,
    domain_permutation: impl IntoIterator<Item = usize>,
    codomain_levels: impl IntoIterator<Item = usize>,
    domain_levels: impl IntoIterator<Item = usize>,
    dst: &mut TensorMap<D, DST_NOUT, DST_NIN, SDst, DDst>,
    src: &TensorMap<D, SRC_NOUT, SRC_NIN, SSrc, DSrc>,
    alpha: D,
    beta: D,
) -> Result<(), OperationError>
where
    R: MultiplicityFreeRigidSymbols,
    R::Scalar: Copy + Clone + Add<Output = R::Scalar> + Mul<Output = R::Scalar> + Zero,
    D: DenseRecouplingScalar + RecouplingCoefficientAction<R::Scalar>,
    DDst: HostWritableStorage<D>,
    DSrc: HostReadableStorage<D>,
{
    tree_transform_into(
        rule,
        TreeTransformOperation::braid(
            codomain_permutation,
            domain_permutation,
            codomain_levels,
            domain_levels,
        ),
        dst,
        src,
        alpha,
        beta,
    )
}

/// TensorKit `braid!`: explicit braid with source-axis levels, with caller-owned backend/workspace.
///
/// Thin wrapper over [`tree_transform_into_with`] with
/// [`TreeTransformOperation::braid`].
pub fn braid_into_with<
    B,
    R,
    D,
    const DST_NOUT: usize,
    const DST_NIN: usize,
    const SRC_NOUT: usize,
    const SRC_NIN: usize,
    SDst,
    SSrc,
    DDst,
    DSrc,
>(
    backend: &mut B,
    workspace: &mut B::Workspace,
    rule: &R,
    codomain_permutation: impl IntoIterator<Item = usize>,
    domain_permutation: impl IntoIterator<Item = usize>,
    codomain_levels: impl IntoIterator<Item = usize>,
    domain_levels: impl IntoIterator<Item = usize>,
    dst: &mut TensorMap<D, DST_NOUT, DST_NIN, SDst, DDst>,
    src: &TensorMap<D, SRC_NOUT, SRC_NIN, SSrc, DSrc>,
    alpha: D,
    beta: D,
) -> Result<(), OperationError>
where
    B: TreeTransformBackend<D, R::Scalar>,
    R: MultiplicityFreeRigidSymbols,
    R::Scalar: Copy + Add<Output = R::Scalar> + Mul<Output = R::Scalar> + Zero,
    D: TreeTransformScalar,
    DDst: HostWritableStorage<D>,
    DSrc: HostReadableStorage<D>,
{
    tree_transform_into_with(
        backend,
        workspace,
        rule,
        TreeTransformOperation::braid(
            codomain_permutation,
            domain_permutation,
            codomain_levels,
            domain_levels,
        ),
        dst,
        src,
        alpha,
        beta,
    )
}

/// TensorKit `braid!`: explicit braid with source-axis levels, with a caller-owned caching execution context.
///
/// Thin wrapper over [`tree_transform_into_with_context`] with
/// [`TreeTransformOperation::braid`].
pub fn braid_into_with_context<
    B,
    R,
    D,
    RuleKey,
    const DST_NOUT: usize,
    const DST_NIN: usize,
    const SRC_NOUT: usize,
    const SRC_NIN: usize,
    SDst,
    SSrc,
    DDst,
    DSrc,
>(
    context: &mut TreeTransformExecutionContext<D, RuleKey, R::Scalar, B>,
    rule: &R,
    codomain_permutation: impl IntoIterator<Item = usize>,
    domain_permutation: impl IntoIterator<Item = usize>,
    codomain_levels: impl IntoIterator<Item = usize>,
    domain_levels: impl IntoIterator<Item = usize>,
    dst: &mut TensorMap<D, DST_NOUT, DST_NIN, SDst, DDst>,
    src: &TensorMap<D, SRC_NOUT, SRC_NIN, SSrc, DSrc>,
    alpha: D,
    beta: D,
) -> Result<(), OperationError>
where
    B: TreeTransformBackend<D, R::Scalar>,
    R: MultiplicityFreeRigidSymbols + TreeTransformRuleCacheKey<Key = RuleKey>,
    RuleKey: 'static + Clone + Eq + Hash + Send + Sync,
    R::Scalar: 'static
        + Copy
        + Clone
        + Add<Output = R::Scalar>
        + Mul<Output = R::Scalar>
        + Zero
        + Send
        + Sync,
    D: TreeTransformScalar,
    DDst: HostWritableStorage<D>,
    DSrc: HostReadableStorage<D>,
{
    tree_transform_into_with_context(
        context,
        rule,
        TreeTransformOperation::braid(
            codomain_permutation,
            domain_permutation,
            codomain_levels,
            domain_levels,
        ),
        dst,
        src,
        alpha,
        beta,
    )
}

/// TensorKit `transpose!`: planar transpose of tensor legs, written into `dst`.
///
/// Thin wrapper over [`tree_transform_into`] with
/// [`TreeTransformOperation::transpose`]; see [`TreeTransformOperation::transpose`]
/// for the axis-numbering convention.
pub fn transpose_into<
    R,
    D,
    const DST_NOUT: usize,
    const DST_NIN: usize,
    const SRC_NOUT: usize,
    const SRC_NIN: usize,
    SDst,
    SSrc,
    DDst,
    DSrc,
>(
    rule: &R,
    codomain_permutation: impl IntoIterator<Item = usize>,
    domain_permutation: impl IntoIterator<Item = usize>,
    dst: &mut TensorMap<D, DST_NOUT, DST_NIN, SDst, DDst>,
    src: &TensorMap<D, SRC_NOUT, SRC_NIN, SSrc, DSrc>,
    alpha: D,
    beta: D,
) -> Result<(), OperationError>
where
    R: MultiplicityFreeRigidSymbols,
    R::Scalar: Copy + Clone + Add<Output = R::Scalar> + Mul<Output = R::Scalar> + Zero,
    D: DenseRecouplingScalar + RecouplingCoefficientAction<R::Scalar>,
    DDst: HostWritableStorage<D>,
    DSrc: HostReadableStorage<D>,
{
    tree_transform_into(
        rule,
        TreeTransformOperation::transpose(codomain_permutation, domain_permutation),
        dst,
        src,
        alpha,
        beta,
    )
}

/// TensorKit `transpose!`: planar transpose of tensor legs, with caller-owned backend/workspace.
///
/// Thin wrapper over [`tree_transform_into_with`] with
/// [`TreeTransformOperation::transpose`].
pub fn transpose_into_with<
    B,
    R,
    D,
    const DST_NOUT: usize,
    const DST_NIN: usize,
    const SRC_NOUT: usize,
    const SRC_NIN: usize,
    SDst,
    SSrc,
    DDst,
    DSrc,
>(
    backend: &mut B,
    workspace: &mut B::Workspace,
    rule: &R,
    codomain_permutation: impl IntoIterator<Item = usize>,
    domain_permutation: impl IntoIterator<Item = usize>,
    dst: &mut TensorMap<D, DST_NOUT, DST_NIN, SDst, DDst>,
    src: &TensorMap<D, SRC_NOUT, SRC_NIN, SSrc, DSrc>,
    alpha: D,
    beta: D,
) -> Result<(), OperationError>
where
    B: TreeTransformBackend<D, R::Scalar>,
    R: MultiplicityFreeRigidSymbols,
    R::Scalar: Copy + Add<Output = R::Scalar> + Mul<Output = R::Scalar> + Zero,
    D: TreeTransformScalar,
    DDst: HostWritableStorage<D>,
    DSrc: HostReadableStorage<D>,
{
    tree_transform_into_with(
        backend,
        workspace,
        rule,
        TreeTransformOperation::transpose(codomain_permutation, domain_permutation),
        dst,
        src,
        alpha,
        beta,
    )
}

/// TensorKit `transpose!`: planar transpose of tensor legs, with a caller-owned caching execution context.
///
/// Thin wrapper over [`tree_transform_into_with_context`] with
/// [`TreeTransformOperation::transpose`].
pub fn transpose_into_with_context<
    B,
    R,
    D,
    RuleKey,
    const DST_NOUT: usize,
    const DST_NIN: usize,
    const SRC_NOUT: usize,
    const SRC_NIN: usize,
    SDst,
    SSrc,
    DDst,
    DSrc,
>(
    context: &mut TreeTransformExecutionContext<D, RuleKey, R::Scalar, B>,
    rule: &R,
    codomain_permutation: impl IntoIterator<Item = usize>,
    domain_permutation: impl IntoIterator<Item = usize>,
    dst: &mut TensorMap<D, DST_NOUT, DST_NIN, SDst, DDst>,
    src: &TensorMap<D, SRC_NOUT, SRC_NIN, SSrc, DSrc>,
    alpha: D,
    beta: D,
) -> Result<(), OperationError>
where
    B: TreeTransformBackend<D, R::Scalar>,
    R: MultiplicityFreeRigidSymbols + TreeTransformRuleCacheKey<Key = RuleKey>,
    RuleKey: 'static + Clone + Eq + Hash + Send + Sync,
    R::Scalar: 'static
        + Copy
        + Clone
        + Add<Output = R::Scalar>
        + Mul<Output = R::Scalar>
        + Zero
        + Send
        + Sync,
    D: TreeTransformScalar,
    DDst: HostWritableStorage<D>,
    DSrc: HostReadableStorage<D>,
{
    tree_transform_into_with_context(
        context,
        rule,
        TreeTransformOperation::transpose(codomain_permutation, domain_permutation),
        dst,
        src,
        alpha,
        beta,
    )
}

// ======================================================================
// Stage B3a: Generic-fusion (outer-multiplicity) facade siblings.
//
// These lift the Stage B2c plan builder
// (`build_generic_tree_pair_transform_group_plan`) to the TensorMap facade so
// SU(3)/SO(N≥7)/Sp(N) rules can drive `permute`/`braid`/`transpose` at the same
// level as the multiplicity-free API. They are *siblings*, not runtime
// branches: `GenericRigidSymbols` and `MultiplicityFreeRigidSymbols` are never
// both implemented by a real rule, so a mult-free rule can never name these
// bounds and the mult-free facade functions above stay byte-for-byte untouched
// (the structural zero-cost guarantee). `TreeTransformGroupPlan::compile` is
// coefficient-generic (`T: Copy`) and execution
// (`tree_transform_execute_with`) is generic over the coefficient type, so no
// recoupling math is added here — the wiring only swaps the plan builder.
//
// Deferred to Stage B3b (recorded rather than built, since nothing can exercise
// them yet — no keyed Generic rule provider exists):
//   * The `_with_context` / cache path (`TreeTransformCache::get_or_compile_*`)
//     requires `TreeTransformRuleCacheKey`, which no Generic rule implements
//     until the B3b SU(3) table provider lands. A Generic cache sibling now
//     would be untestable dead code. When B3b adds a keyed Generic rule its
//     `Key` is a fresh associated type, so the cache — monomorphized per
//     `RuleKey` — shares no map with the mult-free
//     `TreeTransformBuiltinRuleCacheKey` instance and cannot collide.
//   * The top-level `tenet::Tensor` erases its rule behind the closed
//     `RuleKind` enum (mult-free variants only); reaching Generic from there
//     needs a new `RuleKind` variant + provider, which is B3b.
//   * The all-codomain Generic lowering (plan.rs guards #4/#5) is only reached
//     via the cache (`get_or_compile_all_codomain`); the non-cached facade
//     path here only does tree-pair, so it stays deferred as in B2c.
// ======================================================================

/// Generic-fusion sibling of [`tree_transform_structure`]: builds the Stage B2c
/// [`build_generic_tree_pair_transform_group_plan`] and compiles it against the
/// live `dst`/`src` block structures.
pub fn tree_transform_structure_generic<
    R,
    TDst,
    TSrc,
    const DST_NOUT: usize,
    const DST_NIN: usize,
    const SRC_NOUT: usize,
    const SRC_NIN: usize,
    SDst,
    SSrc,
    DDst,
    DSrc,
>(
    rule: &R,
    operation: TreeTransformOperation,
    dst: &TensorMap<TDst, DST_NOUT, DST_NIN, SDst, DDst>,
    src: &TensorMap<TSrc, SRC_NOUT, SRC_NIN, SSrc, DSrc>,
) -> Result<TreeTransformStructure<R::Scalar>, OperationError>
where
    R: GenericRigidSymbols,
    R::Scalar: GenericBraidScalar + Copy + Zero,
    DDst: TensorStorage<TDst>,
    DSrc: TensorStorage<TSrc>,
{
    let plan = build_generic_tree_pair_transform_group_plan(rule, operation, src.structure())?;
    plan.compile(dst, src)
}

/// Generic-fusion sibling of [`tree_transform_into_with`] with caller-owned
/// backend/workspace.
#[allow(clippy::too_many_arguments)]
pub fn tree_transform_into_with_generic<
    B,
    R,
    D,
    const DST_NOUT: usize,
    const DST_NIN: usize,
    const SRC_NOUT: usize,
    const SRC_NIN: usize,
    SDst,
    SSrc,
    DDst,
    DSrc,
>(
    backend: &mut B,
    workspace: &mut B::Workspace,
    rule: &R,
    operation: TreeTransformOperation,
    dst: &mut TensorMap<D, DST_NOUT, DST_NIN, SDst, DDst>,
    src: &TensorMap<D, SRC_NOUT, SRC_NIN, SSrc, DSrc>,
    alpha: D,
    beta: D,
) -> Result<(), OperationError>
where
    B: TreeTransformBackend<D, R::Scalar>,
    R: GenericRigidSymbols,
    R::Scalar: GenericBraidScalar + Copy + Zero,
    D: TreeTransformScalar,
    DDst: HostWritableStorage<D>,
    DSrc: HostReadableStorage<D>,
{
    let structure = tree_transform_structure_generic(rule, operation, dst, src)?;
    tree_transform_execute_with(backend, workspace, &structure, dst, src, alpha, beta)
}

/// Generic-fusion sibling of [`tree_transform_into`]: compile-and-execute a
/// tree-pair transform with a default dense backend.
#[allow(clippy::too_many_arguments)]
pub fn tree_transform_into_generic<
    R,
    D,
    const DST_NOUT: usize,
    const DST_NIN: usize,
    const SRC_NOUT: usize,
    const SRC_NIN: usize,
    SDst,
    SSrc,
    DDst,
    DSrc,
>(
    rule: &R,
    operation: TreeTransformOperation,
    dst: &mut TensorMap<D, DST_NOUT, DST_NIN, SDst, DDst>,
    src: &TensorMap<D, SRC_NOUT, SRC_NIN, SSrc, DSrc>,
    alpha: D,
    beta: D,
) -> Result<(), OperationError>
where
    R: GenericRigidSymbols,
    R::Scalar: GenericBraidScalar + Copy + Zero,
    D: DenseRecouplingScalar + RecouplingCoefficientAction<R::Scalar>,
    DDst: HostWritableStorage<D>,
    DSrc: HostReadableStorage<D>,
{
    let mut backend = DenseTreeTransformOperations::default();
    let mut workspace = TreeTransformWorkspace::default();
    tree_transform_into_with_generic(
        &mut backend,
        &mut workspace,
        rule,
        operation,
        dst,
        src,
        alpha,
        beta,
    )
}

/// Generic-fusion sibling of [`permute_into`].
#[allow(clippy::too_many_arguments)]
pub fn permute_into_generic<
    R,
    D,
    const DST_NOUT: usize,
    const DST_NIN: usize,
    const SRC_NOUT: usize,
    const SRC_NIN: usize,
    SDst,
    SSrc,
    DDst,
    DSrc,
>(
    rule: &R,
    codomain_permutation: impl IntoIterator<Item = usize>,
    domain_permutation: impl IntoIterator<Item = usize>,
    dst: &mut TensorMap<D, DST_NOUT, DST_NIN, SDst, DDst>,
    src: &TensorMap<D, SRC_NOUT, SRC_NIN, SSrc, DSrc>,
    alpha: D,
    beta: D,
) -> Result<(), OperationError>
where
    R: GenericRigidSymbols,
    R::Scalar: GenericBraidScalar + Copy + Zero,
    D: DenseRecouplingScalar + RecouplingCoefficientAction<R::Scalar>,
    DDst: HostWritableStorage<D>,
    DSrc: HostReadableStorage<D>,
{
    tree_transform_into_generic(
        rule,
        TreeTransformOperation::permute(codomain_permutation, domain_permutation),
        dst,
        src,
        alpha,
        beta,
    )
}

/// Generic-fusion sibling of [`braid_into`].
#[allow(clippy::too_many_arguments)]
pub fn braid_into_generic<
    R,
    D,
    const DST_NOUT: usize,
    const DST_NIN: usize,
    const SRC_NOUT: usize,
    const SRC_NIN: usize,
    SDst,
    SSrc,
    DDst,
    DSrc,
>(
    rule: &R,
    codomain_permutation: impl IntoIterator<Item = usize>,
    domain_permutation: impl IntoIterator<Item = usize>,
    codomain_levels: impl IntoIterator<Item = usize>,
    domain_levels: impl IntoIterator<Item = usize>,
    dst: &mut TensorMap<D, DST_NOUT, DST_NIN, SDst, DDst>,
    src: &TensorMap<D, SRC_NOUT, SRC_NIN, SSrc, DSrc>,
    alpha: D,
    beta: D,
) -> Result<(), OperationError>
where
    R: GenericRigidSymbols,
    R::Scalar: GenericBraidScalar + Copy + Zero,
    D: DenseRecouplingScalar + RecouplingCoefficientAction<R::Scalar>,
    DDst: HostWritableStorage<D>,
    DSrc: HostReadableStorage<D>,
{
    tree_transform_into_generic(
        rule,
        TreeTransformOperation::braid(
            codomain_permutation,
            domain_permutation,
            codomain_levels,
            domain_levels,
        ),
        dst,
        src,
        alpha,
        beta,
    )
}

/// Generic-fusion sibling of [`transpose_into`].
#[allow(clippy::too_many_arguments)]
pub fn transpose_into_generic<
    R,
    D,
    const DST_NOUT: usize,
    const DST_NIN: usize,
    const SRC_NOUT: usize,
    const SRC_NIN: usize,
    SDst,
    SSrc,
    DDst,
    DSrc,
>(
    rule: &R,
    codomain_permutation: impl IntoIterator<Item = usize>,
    domain_permutation: impl IntoIterator<Item = usize>,
    dst: &mut TensorMap<D, DST_NOUT, DST_NIN, SDst, DDst>,
    src: &TensorMap<D, SRC_NOUT, SRC_NIN, SSrc, DSrc>,
    alpha: D,
    beta: D,
) -> Result<(), OperationError>
where
    R: GenericRigidSymbols,
    R::Scalar: GenericBraidScalar + Copy + Zero,
    D: DenseRecouplingScalar + RecouplingCoefficientAction<R::Scalar>,
    DDst: HostWritableStorage<D>,
    DSrc: HostReadableStorage<D>,
{
    tree_transform_into_generic(
        rule,
        TreeTransformOperation::transpose(codomain_permutation, domain_permutation),
        dst,
        src,
        alpha,
        beta,
    )
}

pub fn tensoradd_assign_into<T, const NOUT: usize, const NIN: usize, S, DDst, DSrc>(
    dst: &mut TensorMap<T, NOUT, NIN, S, DDst>,
    src: &TensorMap<T, NOUT, NIN, S, DSrc>,
    alpha: T,
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
    tensoradd_into(dst, src, OutputAxisOrder::identity(), alpha, T::zero())
}

pub fn tensoradd_add_into<T, const NOUT: usize, const NIN: usize, S, DDst, DSrc>(
    dst: &mut TensorMap<T, NOUT, NIN, S, DDst>,
    src: &TensorMap<T, NOUT, NIN, S, DSrc>,
    alpha: T,
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
    tensoradd_into(dst, src, OutputAxisOrder::identity(), alpha, T::one())
}

pub fn copy_into<T>(dst: BlockViewMut<'_, T>, src: BlockView<'_, T>) -> Result<(), OperationError>
where
    T: Copy + strided_kernel::MaybeSendSync,
{
    let mut backend = HostTensorOperations;
    let mut allocator = HostAllocator::default();
    backend.copy_block_into(&mut allocator, dst, src)
}

pub fn scaled_assign_into<T>(
    dst: BlockViewMut<'_, T>,
    src: BlockView<'_, T>,
    alpha: T,
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
{
    let mut allocator = HostAllocator::default();
    tensoradd_block_with_strided_kernel(&mut allocator, dst, src, alpha, T::zero())
}

pub fn scaled_add_into<T>(
    dst: BlockViewMut<'_, T>,
    src: BlockView<'_, T>,
    alpha: T,
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
{
    let mut allocator = HostAllocator::default();
    tensoradd_block_with_strided_kernel(&mut allocator, dst, src, alpha, T::one())
}
