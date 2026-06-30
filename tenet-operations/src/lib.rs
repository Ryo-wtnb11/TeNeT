#![forbid(unsafe_code)]

//! TensorOperations-style lowering for TeNeT.
//!
//! Public/core tensor code talks in terms of TeNeT-owned block views. This crate
//! lowers those views to strided-rs kernels at the same granularity that
//! TensorKit uses Strided.jl/StridedViews.jl internally.

use core::ops::{Add, Mul};
use std::hash::Hash;
use std::sync::Arc;

use num_traits::{One, Zero};
use tenet_core::{
    BlockLayout, BlockStructure, BlockView, BlockViewMut, MultiplicityFreeFusionSymbols,
    MultiplicityFreeRigidSymbols, TensorMap,
};

mod axis;
mod backend;
mod cache;
mod contract;
mod error;
mod host_kernels;
mod scalar;
mod strided;
mod tensoradd;
mod tree_structure;
mod tree_transform;

pub use axis::{AxisPermutation, OwnedTensorContractAxisSpec, TensorContractAxisSpec};
pub use backend::{
    DenseTreeTransformOperations, HostAllocator, HostTensorOperations, TensorOperationsBackend,
    TreeTransformBackend,
};
pub use cache::{
    BlockStructureCacheBlockKey, BlockStructureCacheKey, TreeTransformStructureCache,
    TreeTransformStructureCacheKey,
};
#[cfg(test)]
pub(crate) use contract::{
    contracted_fusion_tree_basis_matches, EXPLICIT_OUTPUT_TRANSFORM_REQUIRES_CANONICAL_DST,
};
pub use contract::{
    tensorcontract_execute_with, tensorcontract_fusion_block_specs,
    tensorcontract_fusion_explicit_plan, tensorcontract_fusion_explicit_plan_into,
    tensorcontract_fusion_explicit_plan_into_canonical_dst,
    tensorcontract_fusion_explicit_plan_into_canonical_dst_with,
    tensorcontract_fusion_explicit_plan_into_with, tensorcontract_fusion_into,
    tensorcontract_fusion_into_with, tensorcontract_fusion_structure,
    tensorcontract_fusion_via_tree_pair_transforms_into, tensorcontract_into,
    tensorcontract_into_with, tensorcontract_structure, TensorContractBackend,
    TensorContractBlockSpec, TensorContractFusionExplicitPlan, TensorContractStructure,
    TensorContractStructureTerm, TensorContractWorkspace,
};
pub use error::OperationError;
pub use host_kernels::TreeTransformWorkspace;
pub(crate) use host_kernels::{
    copy_block_with_strided_kernel, tensoradd_block_with_strided_kernel,
    tensoradd_raw_strided_kernel, tensoradd_structure_with_strided_kernel,
    tree_transform_structure_with_dense_recoupling, tree_transform_structure_with_strided_kernel,
};
pub use scalar::{
    DenseBlockScalar, DenseRecouplingScalar, RecouplingCoefficientAction, TreeTransformScalar,
};
use strided::{column_major_strides_usize, element_count, offset_to_isize};
pub use tensoradd::{tensoradd_structure, TensorAddStructure, TensorAddStructureTerm};
pub use tree_structure::TreeTransformStructure;
pub(crate) use tree_structure::{
    TreeTransformBlock, TreeTransformLayout, TreeTransformLayoutTable,
};
pub use tree_transform::{
    build_all_codomain_tree_transform_group_plan, build_tree_pair_transform_group_plan,
    build_tree_transform_group_plan, TreePairTransformCache, TreeTransformBlockSpec,
    TreeTransformBuiltinRuleCacheKey, TreeTransformCache, TreeTransformCacheStats,
    TreeTransformGroupBlockSpec, TreeTransformGroupPlan, TreeTransformKeyBlockSpec,
    TreeTransformOperationKey, TreeTransformPlanScope, TreeTransformProductRuleCacheKey,
    TreeTransformRuleCacheKey, TreeTransformSectorPlanKey, TreeTransformSourceGroupKey,
};
#[cfg(test)]
pub(crate) use tree_transform::{
    build_unique_all_codomain_tree_transform_group_plan,
    build_unique_tree_pair_transform_group_plan, build_unique_tree_transform_group_plan,
    TreeTransformGroupPlanCache, TreeTransformGroupPlanKey,
};

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
    pub fn workspace_mut(&mut self) -> &mut B::Workspace {
        &mut self.workspace
    }

    #[inline]
    pub fn cache(&self) -> &TreeTransformCache<C, RuleKey> {
        &self.cache
    }

    #[inline]
    pub fn cache_mut(&mut self) -> &mut TreeTransformCache<C, RuleKey> {
        &mut self.cache
    }

    pub fn into_parts(self) -> (B, B::Workspace, TreeTransformCache<C, RuleKey>) {
        (self.backend, self.workspace, self.cache)
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
    C: Copy + Clone + Add<Output = C> + Mul<Output = C> + Zero,
    RuleKey: Clone + Eq + Hash,
    B: TreeTransformBackend<D, C>,
{
    pub fn tree_pair_transform_into<
        R,
        const DST_NOUT: usize,
        const DST_NIN: usize,
        const SRC_NOUT: usize,
        const SRC_NIN: usize,
        SDst,
        SSrc,
    >(
        &mut self,
        rule: &R,
        operation: TreeTransformOperationKey,
        dst: &mut TensorMap<D, DST_NOUT, DST_NIN, SDst>,
        src: &TensorMap<D, SRC_NOUT, SRC_NIN, SSrc>,
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
        let structure = cache.get_or_compile_tree_pair(rule, operation, dst, src)?;
        backend.tree_transform_structure_into(workspace, structure, dst, src, alpha, beta)
    }

    pub fn all_codomain_tree_transform_into<
        R,
        const DST_NOUT: usize,
        const DST_NIN: usize,
        const SRC_NOUT: usize,
        const SRC_NIN: usize,
        SDst,
        SSrc,
    >(
        &mut self,
        rule: &R,
        operation: TreeTransformOperationKey,
        dst: &mut TensorMap<D, DST_NOUT, DST_NIN, SDst>,
        src: &TensorMap<D, SRC_NOUT, SRC_NIN, SSrc>,
        alpha: D,
        beta: D,
    ) -> Result<(), OperationError>
    where
        R: MultiplicityFreeFusionSymbols<Scalar = C> + TreeTransformRuleCacheKey<Key = RuleKey>,
    {
        let Self {
            backend,
            workspace,
            cache,
        } = self;
        let structure = cache.get_or_compile_all_codomain(rule, operation, dst, src)?;
        backend.tree_transform_structure_into(workspace, structure, dst, src, alpha, beta)
    }
}

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

pub(crate) fn validate_structure_identity(
    tensor: &'static str,
    expected: &Arc<BlockStructure>,
    actual: &Arc<BlockStructure>,
) -> Result<(), OperationError> {
    if Arc::ptr_eq(expected, actual) || expected.as_ref() == actual.as_ref() {
        Ok(())
    } else {
        Err(OperationError::StructureMismatch { tensor })
    }
}

pub(crate) fn permutation_axes(
    permutation: AxisPermutation<'_>,
    rank: usize,
) -> Result<Vec<usize>, OperationError> {
    match permutation {
        AxisPermutation::Identity => Ok((0..rank).collect()),
        AxisPermutation::Axes(axes) => {
            if axes.len() != rank {
                return Err(OperationError::InvalidPermutation {
                    axes: axes.to_vec(),
                    rank,
                });
            }
            let mut seen = vec![false; rank];
            for &axis in axes {
                if axis >= rank || seen[axis] {
                    return Err(OperationError::InvalidPermutation {
                        axes: axes.to_vec(),
                        rank,
                    });
                }
                seen[axis] = true;
            }
            Ok(axes.to_vec())
        }
    }
}

#[allow(dead_code)]
fn _assert_layout_owned_by_tenet(_layout: BlockLayout<'_>) {}

#[cfg(test)]
mod tests;
