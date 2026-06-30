use core::ops::{Add, Mul};
use std::hash::Hash;

use num_traits::{One, Zero};
use tenet_core::{
    BlockView, BlockViewMut, MultiplicityFreeFusionSymbols, MultiplicityFreeRigidSymbols, TensorMap,
};

use crate::axis::AxisPermutation;
use crate::backend::{
    DenseTreeTransformOperations, HostAllocator, HostTensorOperations, TensorOperationsBackend,
    TreeTransformBackend,
};
use crate::error::OperationError;
use crate::host_kernels::{tensoradd_block_with_strided_kernel, TreeTransformWorkspace};
use crate::scalar::{DenseRecouplingScalar, RecouplingCoefficientAction, TreeTransformScalar};
use crate::tensoradd::{tensoradd_structure, TensorAddStructure};
use crate::tree_context::TreeTransformExecutionContext;
use crate::tree_structure::TreeTransformStructure;
use crate::tree_transform::{
    build_tree_pair_transform_group_plan, TreeTransformOperationKey, TreeTransformRuleCacheKey,
};

pub fn tensorcopy_into<T, const NOUT: usize, const NIN: usize, S>(
    dst: &mut TensorMap<T, NOUT, NIN, S>,
    src: &TensorMap<T, NOUT, NIN, S>,
) -> Result<(), OperationError>
where
    T: Copy + strided_kernel::MaybeSendSync,
{
    let mut backend = HostTensorOperations;
    let mut allocator = HostAllocator::default();
    tensorcopy_into_with(&mut backend, &mut allocator, dst, src)
}

pub fn tensorcopy_into_with<B, T, const NOUT: usize, const NIN: usize, S>(
    backend: &mut B,
    allocator: &mut B::Allocator,
    dst: &mut TensorMap<T, NOUT, NIN, S>,
    src: &TensorMap<T, NOUT, NIN, S>,
) -> Result<(), OperationError>
where
    B: TensorOperationsBackend,
    T: Copy + strided_kernel::MaybeSendSync,
{
    backend.copy_block_into(allocator, dst.subblock_mut()?, src.subblock()?)
}

pub fn tensoradd_into<T, const NOUT: usize, const NIN: usize, S>(
    dst: &mut TensorMap<T, NOUT, NIN, S>,
    src: &TensorMap<T, NOUT, NIN, S>,
    permutation: AxisPermutation<'_>,
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
        + strided_kernel::MaybeSendSync,
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

pub fn tensoradd_into_with<B, T, const NOUT: usize, const NIN: usize, S>(
    backend: &mut B,
    allocator: &mut B::Allocator,
    dst: &mut TensorMap<T, NOUT, NIN, S>,
    src: &TensorMap<T, NOUT, NIN, S>,
    permutation: AxisPermutation<'_>,
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
        + strided_kernel::MaybeSendSync,
{
    let structure = tensoradd_structure(dst, src, permutation)?;
    tensoradd_execute_with(backend, allocator, &structure, dst, src, alpha, beta)
}

pub fn tensoradd_execute_with<B, T, const NOUT: usize, const NIN: usize, S>(
    backend: &mut B,
    allocator: &mut B::Allocator,
    structure: &TensorAddStructure,
    dst: &mut TensorMap<T, NOUT, NIN, S>,
    src: &TensorMap<T, NOUT, NIN, S>,
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
        + strided_kernel::MaybeSendSync,
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
>(
    backend: &mut B,
    workspace: &mut B::Workspace,
    structure: &TreeTransformStructure<C>,
    dst: &mut TensorMap<D, DST_NOUT, DST_NIN, SDst>,
    src: &TensorMap<D, SRC_NOUT, SRC_NIN, SSrc>,
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
        + strided_kernel::MaybeSendSync,
    C: Copy,
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
pub fn tree_pair_transform_structure<
    R,
    TDst,
    TSrc,
    const DST_NOUT: usize,
    const DST_NIN: usize,
    const SRC_NOUT: usize,
    const SRC_NIN: usize,
    SDst,
    SSrc,
>(
    rule: &R,
    operation: TreeTransformOperationKey,
    dst: &TensorMap<TDst, DST_NOUT, DST_NIN, SDst>,
    src: &TensorMap<TSrc, SRC_NOUT, SRC_NIN, SSrc>,
) -> Result<TreeTransformStructure<R::Scalar>, OperationError>
where
    R: MultiplicityFreeRigidSymbols,
    R::Scalar: Copy + Add<Output = R::Scalar> + Mul<Output = R::Scalar> + Zero,
{
    let plan = build_tree_pair_transform_group_plan(rule, operation, src.structure())?;
    plan.compile(dst, src)
}

/// Compile and execute a tree-pair transform in one call.
///
/// This is a convenience API. It rebuilds the transform structure on every call;
/// hot tensor-network loops should call [`tree_pair_transform_structure`] once
/// and replay the returned structure with [`tree_transform_execute_with`].
pub fn tree_pair_transform_into<
    R,
    D,
    const DST_NOUT: usize,
    const DST_NIN: usize,
    const SRC_NOUT: usize,
    const SRC_NIN: usize,
    SDst,
    SSrc,
>(
    rule: &R,
    operation: TreeTransformOperationKey,
    dst: &mut TensorMap<D, DST_NOUT, DST_NIN, SDst>,
    src: &TensorMap<D, SRC_NOUT, SRC_NIN, SSrc>,
    alpha: D,
    beta: D,
) -> Result<(), OperationError>
where
    R: MultiplicityFreeRigidSymbols,
    R::Scalar: Copy + Clone + Add<Output = R::Scalar> + Mul<Output = R::Scalar> + Zero,
    D: DenseRecouplingScalar + RecouplingCoefficientAction<R::Scalar>,
{
    let mut backend = DenseTreeTransformOperations::default();
    let mut workspace = TreeTransformWorkspace::default();
    tree_pair_transform_into_with(
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
/// Use [`tree_pair_transform_into_with_context`] when the categorical plan and
/// replay descriptor should be cached behind a caller-owned context. Use
/// [`tree_pair_transform_structure`] plus [`tree_transform_execute_with`] for
/// the tightest loop when the exact replay descriptor is already known.
pub fn tree_pair_transform_into_with<
    B,
    R,
    D,
    const DST_NOUT: usize,
    const DST_NIN: usize,
    const SRC_NOUT: usize,
    const SRC_NIN: usize,
    SDst,
    SSrc,
>(
    backend: &mut B,
    workspace: &mut B::Workspace,
    rule: &R,
    operation: TreeTransformOperationKey,
    dst: &mut TensorMap<D, DST_NOUT, DST_NIN, SDst>,
    src: &TensorMap<D, SRC_NOUT, SRC_NIN, SSrc>,
    alpha: D,
    beta: D,
) -> Result<(), OperationError>
where
    B: TreeTransformBackend<D, R::Scalar>,
    R: MultiplicityFreeRigidSymbols,
    R::Scalar: Copy + Add<Output = R::Scalar> + Mul<Output = R::Scalar> + Zero,
    D: TreeTransformScalar,
{
    let structure = tree_pair_transform_structure(rule, operation, dst, src)?;
    tree_transform_execute_with(backend, workspace, &structure, dst, src, alpha, beta)
}

pub fn tree_pair_transform_into_with_context<
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
>(
    context: &mut TreeTransformExecutionContext<D, RuleKey, R::Scalar, B>,
    rule: &R,
    operation: TreeTransformOperationKey,
    dst: &mut TensorMap<D, DST_NOUT, DST_NIN, SDst>,
    src: &TensorMap<D, SRC_NOUT, SRC_NIN, SSrc>,
    alpha: D,
    beta: D,
) -> Result<(), OperationError>
where
    B: TreeTransformBackend<D, R::Scalar>,
    R: MultiplicityFreeRigidSymbols + TreeTransformRuleCacheKey<Key = RuleKey>,
    RuleKey: Clone + Eq + Hash,
    R::Scalar: Copy + Clone + Add<Output = R::Scalar> + Mul<Output = R::Scalar> + Zero,
    D: TreeTransformScalar,
{
    context.tree_pair_transform_into(rule, operation, dst, src, alpha, beta)
}

pub fn all_codomain_tree_transform_into_with_context<
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
>(
    context: &mut TreeTransformExecutionContext<D, RuleKey, R::Scalar, B>,
    rule: &R,
    operation: TreeTransformOperationKey,
    dst: &mut TensorMap<D, DST_NOUT, DST_NIN, SDst>,
    src: &TensorMap<D, SRC_NOUT, SRC_NIN, SSrc>,
    alpha: D,
    beta: D,
) -> Result<(), OperationError>
where
    B: TreeTransformBackend<D, R::Scalar>,
    R: MultiplicityFreeFusionSymbols + TreeTransformRuleCacheKey<Key = RuleKey>,
    RuleKey: Clone + Eq + Hash,
    R::Scalar: Copy + Clone + Add<Output = R::Scalar> + Mul<Output = R::Scalar> + Zero,
    D: TreeTransformScalar,
{
    context.all_codomain_tree_transform_into(rule, operation, dst, src, alpha, beta)
}

pub fn tensoradd_assign_into<T, const NOUT: usize, const NIN: usize, S>(
    dst: &mut TensorMap<T, NOUT, NIN, S>,
    src: &TensorMap<T, NOUT, NIN, S>,
    alpha: T,
) -> Result<(), OperationError>
where
    T: Copy
        + Add<T, Output = T>
        + Mul<T, Output = T>
        + PartialEq
        + Zero
        + One
        + strided_kernel::MaybeSendSync,
{
    tensoradd_into(dst, src, AxisPermutation::identity(), alpha, T::zero())
}

pub fn tensoradd_add_into<T, const NOUT: usize, const NIN: usize, S>(
    dst: &mut TensorMap<T, NOUT, NIN, S>,
    src: &TensorMap<T, NOUT, NIN, S>,
    alpha: T,
) -> Result<(), OperationError>
where
    T: Copy
        + Add<T, Output = T>
        + Mul<T, Output = T>
        + PartialEq
        + Zero
        + One
        + strided_kernel::MaybeSendSync,
{
    tensoradd_into(dst, src, AxisPermutation::identity(), alpha, T::one())
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
        + strided_kernel::MaybeSendSync,
{
    let mut allocator = HostAllocator::default();
    tensoradd_block_with_strided_kernel(&mut allocator, dst, src, alpha, T::one())
}
