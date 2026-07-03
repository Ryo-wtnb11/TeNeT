use std::collections::{HashMap, HashSet, VecDeque};
use std::hash::Hash;
use std::sync::Arc;

use tenet_core::{
    BlockKey, BlockStructure, FusionTreeHomSpace, FusionTreeKey, HostReadableStorage,
    HostWritableStorage, MultiplicityFreeRigidSymbols, SectorId,
};

use crate::cache::{
    rebuild_lru_order_from_keys, touch_lru_key, BlockStructureCacheKey, OperationCachePolicy,
};
use crate::strided::{
    column_major_strides_isize, column_major_strides_usize, element_count, offset_to_isize,
    strides_to_isize,
};
use crate::{
    DenseBlockScalar, HostKernelAdapter, OperationError, RecouplingCoefficientAction,
    TreeTransformRuleCacheKey,
};
use tenet_operations::{AxisPermutation, OwnedTensorContractAxisSpec, TensorContractAxisSpec};

use tenet_operations::fusion_replay::{
    direct_group_matrix_offset, fusion_scale_block_layouts_excluding,
    CanonicalFusionBlockContractGroupPlan, FusionBlockMatrixGroup, FusionStridedBlockLayout,
    FusionSubblockMatrixLayout,
};
pub(crate) use tenet_operations::fusion_replay::{
    CanonicalFusionBlockContractPlan, CanonicalFusionBlockContractWorkspace, Rank2Gemm, StorageGemm,
};

use super::backend::TensorContractBackend;
use super::dynamic_space::DynamicFusionMapSpace;
use super::fusion::reject_fusion_contract_conjugation;
use super::structure::TensorContractAxisPlan;

/// Adapts a [`TensorContractBackend`] + workspace pair onto the replay
/// layer's [`Rank2Gemm`] seam.
pub(crate) struct BackendRank2Gemm<'a, B, W> {
    pub(crate) backend: &'a mut B,
    pub(crate) workspace: &'a mut W,
}

impl<'a, B, D> Rank2Gemm<D> for BackendRank2Gemm<'a, B, B::Workspace>
where
    B: TensorContractBackend<D, f64>,
    D: DenseBlockScalar + RecouplingCoefficientAction<f64>,
{
    fn matmul_rank2(
        &mut self,
        dst: &mut [D],
        lhs: &[D],
        rhs: &[D],
        rows: usize,
        contracted: usize,
        cols: usize,
        alpha: D,
        beta: D,
    ) -> Result<(), OperationError> {
        self.backend.matmul_rank2_axpby_into_raw(
            self.workspace,
            dst,
            lhs,
            rhs,
            rows,
            contracted,
            cols,
            alpha,
            beta,
        )
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn tensorcontract_canonical_fusion_blocks_into_raw<A, B, R, D>(
    kernels: &mut A,
    backend: &mut B,
    workspace: &mut B::Workspace,
    rule: &R,
    dst_space: &DynamicFusionMapSpace,
    dst_data: &mut [D],
    lhs_space: &DynamicFusionMapSpace,
    lhs_data: &[D],
    rhs_space: &DynamicFusionMapSpace,
    rhs_data: &[D],
    axes: TensorContractAxisSpec<'_>,
    alpha: D,
    beta: D,
) -> Result<(), OperationError>
where
    A: HostKernelAdapter<D>,
    B: TensorContractBackend<D, f64>,
    R: MultiplicityFreeRigidSymbols<Scalar = f64>,
    D: DenseBlockScalar + RecouplingCoefficientAction<f64>,
{
    let plan =
        compile_canonical_fusion_block_contract_plan(rule, dst_space, lhs_space, rhs_space, axes)?;
    let mut fusion_workspace = CanonicalFusionBlockContractWorkspace::<D>::default();
    plan.execute_raw(
        kernels,
        &mut BackendRank2Gemm { backend, workspace },
        &mut fusion_workspace,
        dst_space.structure(),
        dst_data,
        lhs_space.structure(),
        lhs_data,
        rhs_space.structure(),
        rhs_data,
        alpha,
        beta,
    )
}

pub(crate) fn is_canonical_fusion_block_contract<R>(
    rule: &R,
    dst_space: &DynamicFusionMapSpace,
    lhs_space: &DynamicFusionMapSpace,
    rhs_space: &DynamicFusionMapSpace,
    axes: TensorContractAxisSpec<'_>,
) -> Result<bool, OperationError>
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64>,
{
    reject_fusion_contract_conjugation(axes)?;
    let axis_plan = TensorContractAxisPlan::compile(
        lhs_space.rank(),
        rhs_space.rank(),
        dst_space.rank(),
        axes,
    )?;
    if !is_canonical_source(lhs_space, rhs_space, &axis_plan)
        || !is_canonical_output(dst_space, lhs_space, rhs_space, &axis_plan)
    {
        return Ok(false);
    }
    let expected_homspace = FusionTreeHomSpace::tensorcontract_homspace(
        rule,
        lhs_space.homspace(),
        rhs_space.homspace(),
        axes.lhs_contracting_axes(),
        axes.rhs_contracting_axes(),
        axis_plan.output_axes.as_slice(),
        dst_space.nout(),
    )
    .map_err(OperationError::from_core_preserving_context)?;
    if expected_homspace != *dst_space.homspace() {
        return Err(OperationError::StructureMismatch { tensor: "dst" });
    }
    Ok(true)
}

fn validate_canonical_compose<R>(
    rule: &R,
    dst_space: &DynamicFusionMapSpace,
    lhs_space: &DynamicFusionMapSpace,
    rhs_space: &DynamicFusionMapSpace,
    axes: TensorContractAxisSpec<'_>,
) -> Result<(), OperationError>
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64>,
{
    if is_canonical_fusion_block_contract(rule, dst_space, lhs_space, rhs_space, axes)? {
        Ok(())
    } else {
        Err(OperationError::UnsupportedTensorContractScope {
            message: "canonical fusion-block contraction requires canonical source and output axes",
        })
    }
}

fn is_canonical_source(
    lhs_space: &DynamicFusionMapSpace,
    rhs_space: &DynamicFusionMapSpace,
    axis_plan: &TensorContractAxisPlan,
) -> bool {
    let lhs_domain_axes = (lhs_space.nout()..lhs_space.rank()).collect::<Vec<_>>();
    let rhs_codomain_axes = (0..rhs_space.nout()).collect::<Vec<_>>();
    axis_plan.lhs_contracting_axes == lhs_domain_axes
        && axis_plan.rhs_contracting_axes == rhs_codomain_axes
}

fn is_canonical_output(
    dst_space: &DynamicFusionMapSpace,
    lhs_space: &DynamicFusionMapSpace,
    rhs_space: &DynamicFusionMapSpace,
    axis_plan: &TensorContractAxisPlan,
) -> bool {
    let output_rank = lhs_space.nout() + (rhs_space.rank() - rhs_space.nout());
    let canonical_output_axes = (0..output_rank).collect::<Vec<_>>();
    dst_space.nout() == lhs_space.nout() && axis_plan.output_axes == canonical_output_axes
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::RefCell;
    use std::rc::Rc;
    use tenet_core::{
        FusionProductSpace, FusionTensorMapSpace, HostReadableStorage, HostWritableStorage,
        SectorLeg, TensorMap, TensorMapSpace, TensorStorage, Trivial, Z2FusionRule,
    };
    use tenet_core::{Placement, SimilarStorage};
    use tenet_operations::fusion_replay::HostCanonicalFusionBlockContractWorkspace;
    use tenet_operations::storage_scratch::StorageFusionBlockContractWorkspace;
    use tenet_operations::ReportsPlacement;

    use crate::{DenseTreeTransformOperations, TensorContractWorkspace};

    #[derive(Clone, Debug, Eq, PartialEq)]
    struct ScratchAllocation {
        label: &'static str,
        len: usize,
    }

    #[derive(Clone, Debug)]
    struct TrackingStorage<T> {
        data: Vec<T>,
        label: &'static str,
        allocations: Rc<RefCell<Vec<ScratchAllocation>>>,
    }

    #[derive(Clone, Debug)]
    struct TrackingScratch<T> {
        data: Vec<T>,
    }

    impl<T> TrackingStorage<T> {
        fn new(
            data: Vec<T>,
            label: &'static str,
            allocations: Rc<RefCell<Vec<ScratchAllocation>>>,
        ) -> Self {
            Self {
                data,
                label,
                allocations,
            }
        }
    }

    impl<T> TensorStorage<T> for TrackingStorage<T> {
        fn len(&self) -> usize {
            self.data.len()
        }

        fn placement(&self) -> Placement {
            Placement::Host
        }
    }

    impl<T> HostReadableStorage<T> for TrackingStorage<T> {
        fn as_slice(&self) -> &[T] {
            &self.data
        }
    }

    impl<T> HostWritableStorage<T> for TrackingStorage<T> {
        fn as_mut_slice(&mut self) -> &mut [T] {
            &mut self.data
        }
    }

    impl<T: Clone> SimilarStorage<T> for TrackingStorage<T> {
        type Similar = TrackingScratch<T>;

        fn similar_filled(&self, len: usize, value: T) -> Self::Similar
        where
            T: Clone,
        {
            self.allocations.borrow_mut().push(ScratchAllocation {
                label: self.label,
                len,
            });
            TrackingScratch {
                data: vec![value; len],
            }
        }
    }

    impl<T> TensorStorage<T> for TrackingScratch<T> {
        fn len(&self) -> usize {
            self.data.len()
        }

        fn placement(&self) -> Placement {
            Placement::Host
        }
    }

    impl<T> HostReadableStorage<T> for TrackingScratch<T> {
        fn as_slice(&self) -> &[T] {
            &self.data
        }
    }

    impl<T> HostWritableStorage<T> for TrackingScratch<T> {
        fn as_mut_slice(&mut self) -> &mut [T] {
            &mut self.data
        }
    }

    impl<T: Clone> tenet_core::ScratchStorage<T> for TrackingScratch<T> {
        fn reset_filled(&mut self, len: usize, value: T)
        where
            T: Clone,
        {
            self.data.clear();
            self.data.resize(len, value);
        }
    }

    /// Storage with no host-slice access: compiling the storage-direct path
    /// against this type proves the seam has no host contract.
    #[derive(Debug)]
    struct OpaqueStorage<T> {
        cells: Vec<T>,
    }

    impl<T> TensorStorage<T> for OpaqueStorage<T> {
        fn len(&self) -> usize {
            self.cells.len()
        }

        fn placement(&self) -> Placement {
            Placement::Host
        }
    }

    struct NaiveOpaqueGemm;

    impl StorageGemm<f64, OpaqueStorage<f64>, OpaqueStorage<f64>, OpaqueStorage<f64>>
        for NaiveOpaqueGemm
    {
        fn matmul_range_into(
            &mut self,
            dst: &mut OpaqueStorage<f64>,
            dst_offset: usize,
            lhs: &OpaqueStorage<f64>,
            lhs_offset: usize,
            rhs: &OpaqueStorage<f64>,
            rhs_offset: usize,
            rows: usize,
            contracted: usize,
            cols: usize,
        ) -> Result<(), OperationError> {
            for col in 0..cols {
                for row in 0..rows {
                    let mut sum = 0.0;
                    for inner in 0..contracted {
                        sum += lhs.cells[lhs_offset + row + rows * inner]
                            * rhs.cells[rhs_offset + inner + contracted * col];
                    }
                    dst.cells[dst_offset + row + rows * col] = sum;
                }
            }
            Ok(())
        }
    }

    #[test]
    fn storage_direct_replay_runs_without_host_slice_contract() {
        let rule = Z2FusionRule;
        let leg = || SectorLeg::new([SectorId::new(0), SectorId::new(1)], false);
        let fusion_space = || {
            FusionTensorMapSpace::from_degeneracy_shapes(
                TensorMapSpace::<1, 1>::from_dims([1], [1]).unwrap(),
                FusionTreeHomSpace::new(
                    FusionProductSpace::new([leg()]),
                    FusionProductSpace::new([leg()]),
                ),
                &rule,
                [vec![1, 1], vec![1, 1]],
            )
            .unwrap()
        };
        let space = fusion_space();
        let plan = compile_canonical_fusion_block_contract_plan(
            &rule,
            &DynamicFusionMapSpace::from_typed(&space),
            &DynamicFusionMapSpace::from_typed(&space),
            &DynamicFusionMapSpace::from_typed(&space),
            TensorContractAxisSpec::canonical(&[1], &[0]),
        )
        .unwrap();

        let lhs = OpaqueStorage {
            cells: vec![2.0, 3.0],
        };
        let rhs = OpaqueStorage {
            cells: vec![5.0, 7.0],
        };
        let mut dst = OpaqueStorage {
            cells: vec![0.0, 0.0],
        };
        plan.execute_direct_on_storage(&mut NaiveOpaqueGemm, &mut dst, &lhs, &rhs)
            .unwrap();

        // Two 1x1 sector matrices: dst = lhs * rhs per sector.
        assert_eq!(dst.cells, vec![10.0, 21.0]);
    }

    #[test]
    fn storage_direct_replay_matches_host_execute_raw() {
        let rule = Z2FusionRule;
        let leg = || SectorLeg::new([SectorId::new(0), SectorId::new(1)], false);
        let fusion_space = |dims: usize| {
            FusionTensorMapSpace::from_degeneracy_shapes_coupled(
                TensorMapSpace::<1, 1>::from_dims([2 * dims], [2 * dims]).unwrap(),
                FusionTreeHomSpace::new(
                    FusionProductSpace::new([leg()]),
                    FusionProductSpace::new([leg()]),
                ),
                &rule,
                [vec![dims, dims], vec![dims, dims]],
            )
            .unwrap()
        };
        let space = fusion_space(2);
        let len = space.required_len().unwrap();
        let lhs_data: Vec<f64> = (0..len).map(|i| 0.5 * i as f64 - 1.0).collect();
        let rhs_data: Vec<f64> = (0..len).map(|i| 1.5 - 0.25 * i as f64).collect();
        let plan = compile_canonical_fusion_block_contract_plan(
            &rule,
            &DynamicFusionMapSpace::from_typed(&space),
            &DynamicFusionMapSpace::from_typed(&space),
            &DynamicFusionMapSpace::from_typed(&space),
            TensorContractAxisSpec::canonical(&[1], &[0]),
        )
        .unwrap();

        let mut backend = DenseTreeTransformOperations::default();
        let mut workspace = TensorContractWorkspace::default();
        let mut expected = vec![0.0; len];
        let structure = std::sync::Arc::clone(space.subblock_structure());
        let mut fusion_workspace = CanonicalFusionBlockContractWorkspace::<f64>::default();
        plan.execute_raw(
            &mut crate::StridedHostKernelAdapter,
            &mut BackendRank2Gemm {
                backend: &mut backend,
                workspace: &mut workspace,
            },
            &mut fusion_workspace,
            &structure,
            &mut expected,
            &structure,
            &lhs_data,
            &structure,
            &rhs_data,
            1.0,
            0.0,
        )
        .unwrap();

        let mut direct = vec![0.0; len];
        let mut gemm = HostStorageGemm::new(&mut backend, &mut workspace);
        plan.execute_direct_on_storage(
            &mut gemm,
            &mut direct,
            &lhs_data.clone(),
            &rhs_data.clone(),
        )
        .unwrap();

        assert_eq!(direct, expected);
    }

    #[test]
    fn canonical_fusion_block_workspace_is_explicit_host_workspace() {
        let workspace = HostCanonicalFusionBlockContractWorkspace::<f64>::default();
        let alias = CanonicalFusionBlockContractWorkspace::<f64>::default();

        assert_eq!(workspace.placement(), Placement::Host);
        assert!(workspace.is_host_placement());
        assert_eq!(alias.placement(), Placement::Host);
    }

    #[test]
    fn canonical_fusion_block_storage_workspace_allocates_pack_scratch_from_operands_and_output_from_destination(
    ) {
        let rule = Z2FusionRule;
        let leg = || SectorLeg::new([SectorId::new(0), SectorId::new(1)], false);
        let fusion_space = || {
            FusionTensorMapSpace::from_degeneracy_shapes(
                TensorMapSpace::<1, 1>::from_dims([1], [1]).unwrap(),
                FusionTreeHomSpace::new(
                    FusionProductSpace::new([leg()]),
                    FusionProductSpace::new([leg()]),
                ),
                &rule,
                [vec![1, 1], vec![1, 1]],
            )
            .unwrap()
        };
        let allocations = Rc::new(RefCell::new(Vec::new()));
        let lhs =
            TensorMap::<f64, 1, 1, Trivial, TrackingStorage<f64>>::from_storage_with_fusion_space(
                TrackingStorage::new(vec![2.0, 3.0], "lhs", allocations.clone()),
                fusion_space(),
            )
            .unwrap();
        let rhs =
            TensorMap::<f64, 1, 1, Trivial, TrackingStorage<f64>>::from_storage_with_fusion_space(
                TrackingStorage::new(vec![5.0, 7.0], "rhs", allocations.clone()),
                fusion_space(),
            )
            .unwrap();
        let mut dst =
            TensorMap::<f64, 1, 1, Trivial, TrackingStorage<f64>>::from_storage_with_fusion_space(
                TrackingStorage::new(vec![10.0, 20.0], "destination", allocations.clone()),
                fusion_space(),
            )
            .unwrap();
        let plan = compile_canonical_fusion_block_contract_plan(
            &rule,
            &DynamicFusionMapSpace::from_typed(dst.fusion_space().unwrap()),
            &DynamicFusionMapSpace::from_typed(lhs.fusion_space().unwrap()),
            &DynamicFusionMapSpace::from_typed(rhs.fusion_space().unwrap()),
            TensorContractAxisSpec::canonical(&[1], &[0]),
        )
        .unwrap();
        let mut backend = DenseTreeTransformOperations::default();
        let mut workspace = TensorContractWorkspace::default();
        let mut fusion_workspace = StorageFusionBlockContractWorkspace::<
            TrackingScratch<f64>,
            TrackingScratch<f64>,
            TrackingScratch<f64>,
        >::default();

        plan.execute_storage_workspace(
            &mut crate::StridedHostKernelAdapter,
            &mut BackendRank2Gemm {
                backend: &mut backend,
                workspace: &mut workspace,
            },
            &mut fusion_workspace,
            &mut dst,
            &lhs,
            &rhs,
            2.0,
            3.0,
        )
        .unwrap();

        assert_eq!(dst.data(), &[50.0, 102.0]);
        // Pack/scatter scratch is gone from the canonical route: replay is
        // direct GEMM on storage, so no workspace allocations occur.
        assert_eq!(allocations.borrow().as_slice(), &[]);
    }
}

pub(crate) fn compile_canonical_fusion_block_contract_plan<R>(
    rule: &R,
    dst_space: &DynamicFusionMapSpace,
    lhs_space: &DynamicFusionMapSpace,
    rhs_space: &DynamicFusionMapSpace,
    axes: TensorContractAxisSpec<'_>,
) -> Result<CanonicalFusionBlockContractPlan, OperationError>
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64>,
{
    reject_fusion_contract_conjugation(axes)?;
    // Axis validation happens inside validate_canonical_compose.
    validate_canonical_compose(rule, dst_space, lhs_space, rhs_space, axes)?;

    let lhs_layout = FusionBlockMatrixLayout::compile(rule, lhs_space)?;
    let rhs_layout = FusionBlockMatrixLayout::compile(rule, rhs_space)?;
    let dst_layout = FusionBlockMatrixLayout::compile(rule, dst_space)?;

    let mut groups = Vec::new();
    let mut active_dst_blocks = HashSet::<usize>::new();
    for lhs_group in lhs_layout.groups {
        let Some(rhs_group) = rhs_layout.group(lhs_group.coupled) else {
            continue;
        };
        let Some(dst_group) = dst_layout.group(lhs_group.coupled) else {
            continue;
        };
        for block_index in &dst_group.block_indices {
            debug_assert!(
                !active_dst_blocks.contains(block_index),
                "canonical fusion-block dst subblock must be scattered exactly once"
            );
        }
        active_dst_blocks.extend(dst_group.block_indices.iter().copied());
        groups.push(CanonicalFusionBlockContractGroupPlan::new(
            lhs_group,
            rhs_group.clone(),
            dst_group.clone(),
        )?);
    }
    Ok(CanonicalFusionBlockContractPlan::from_parts(
        Arc::clone(dst_space.structure()),
        Arc::clone(lhs_space.structure()),
        Arc::clone(rhs_space.structure()),
        fusion_scale_block_layouts_excluding(dst_space.structure(), &active_dst_blocks)?,
        groups,
    ))
}

/// Host implementation of [`StorageGemm`] over host-readable storages.
#[allow(dead_code)]
pub(crate) struct HostStorageGemm<'a, B, W> {
    backend: &'a mut B,
    workspace: &'a mut W,
}

impl<'a, B, W> HostStorageGemm<'a, B, W> {
    #[allow(dead_code)]
    pub(crate) fn new(backend: &'a mut B, workspace: &'a mut W) -> Self {
        Self { backend, workspace }
    }
}

impl<'a, B, D, DDst, DLhs, DRhs> StorageGemm<D, DDst, DLhs, DRhs>
    for HostStorageGemm<'a, B, B::Workspace>
where
    B: TensorContractBackend<D, f64>,
    D: DenseBlockScalar + RecouplingCoefficientAction<f64>,
    DDst: HostWritableStorage<D>,
    DLhs: HostReadableStorage<D>,
    DRhs: HostReadableStorage<D>,
{
    fn matmul_range_into(
        &mut self,
        dst: &mut DDst,
        dst_offset: usize,
        lhs: &DLhs,
        lhs_offset: usize,
        rhs: &DRhs,
        rhs_offset: usize,
        rows: usize,
        contracted: usize,
        cols: usize,
    ) -> Result<(), OperationError> {
        let dst_len = rows * cols;
        let lhs_len = rows * contracted;
        let rhs_len = contracted * cols;
        self.backend.matmul_rank2_into_raw(
            self.workspace,
            &mut dst.as_mut_slice()[dst_offset..dst_offset + dst_len],
            &lhs.as_slice()[lhs_offset..lhs_offset + lhs_len],
            &rhs.as_slice()[rhs_offset..rhs_offset + rhs_len],
            rows,
            contracted,
            cols,
        )
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(crate) struct CanonicalFusionBlockContractCacheStats {
    hits: usize,
    fast_hits: usize,
    misses: usize,
}

impl CanonicalFusionBlockContractCacheStats {
    #[inline]
    pub(crate) fn hits(self) -> usize {
        self.hits
    }

    #[inline]
    pub(crate) fn fast_hits(self) -> usize {
        self.fast_hits
    }

    #[inline]
    pub(crate) fn misses(self) -> usize {
        self.misses
    }
}

#[derive(Clone, Debug)]
pub(crate) struct CanonicalFusionBlockContractCache<RuleKey> {
    last: Option<CanonicalFusionBlockContractLastEntry<RuleKey>>,
    fast_plans: HashMap<
        CanonicalFusionBlockContractFastKey<RuleKey>,
        Arc<CanonicalFusionBlockContractPlan>,
    >,
    plans: HashMap<
        CanonicalFusionBlockContractCacheKey<RuleKey>,
        Arc<CanonicalFusionBlockContractPlan>,
    >,
    plan_lru_order: VecDeque<CanonicalFusionBlockContractCacheKey<RuleKey>>,
    policy: OperationCachePolicy,
    stats: CanonicalFusionBlockContractCacheStats,
}

impl<RuleKey> Default for CanonicalFusionBlockContractCache<RuleKey> {
    fn default() -> Self {
        Self {
            last: None,
            fast_plans: HashMap::new(),
            plans: HashMap::new(),
            plan_lru_order: VecDeque::new(),
            policy: OperationCachePolicy::default(),
            stats: CanonicalFusionBlockContractCacheStats::default(),
        }
    }
}

impl<RuleKey> CanonicalFusionBlockContractCache<RuleKey>
where
    RuleKey: Clone + Eq + Hash,
{
    #[inline]
    pub(crate) fn len(&self) -> usize {
        self.plans.len()
    }

    #[inline]
    pub(crate) fn stats(&self) -> CanonicalFusionBlockContractCacheStats {
        self.stats
    }

    pub(crate) fn set_policy(&mut self, policy: OperationCachePolicy) {
        self.policy = policy;
        self.last = None;
        self.fast_plans.clear();
        if !policy.stores_entries() {
            self.plans.clear();
            self.plan_lru_order.clear();
        } else if let Some(max_entries) = policy.max_entries() {
            rebuild_lru_order_from_keys(&self.plans, &mut self.plan_lru_order);
            self.enforce_lru_limit(max_entries);
        }
    }

    fn touch_plan(&mut self, key: &CanonicalFusionBlockContractCacheKey<RuleKey>) {
        if self.policy.max_entries().is_some() && self.plans.contains_key(key) {
            touch_lru_key(&mut self.plan_lru_order, key);
        }
    }

    fn insert_plan(
        &mut self,
        key: CanonicalFusionBlockContractCacheKey<RuleKey>,
        fast_key: CanonicalFusionBlockContractFastKey<RuleKey>,
        plan: Arc<CanonicalFusionBlockContractPlan>,
    ) {
        if !self.policy.stores_entries() {
            return;
        }
        self.plans.insert(key.clone(), Arc::clone(&plan));
        self.fast_plans.insert(fast_key, plan);
        if self.policy.max_entries().is_some() {
            self.touch_plan(&key);
        }
        if let Some(max_entries) = self.policy.max_entries() {
            self.enforce_lru_limit(max_entries);
        }
    }

    fn enforce_lru_limit(&mut self, max_entries: usize) {
        let mut evicted = false;
        while self.plans.len() > max_entries {
            let Some(oldest) = self.plan_lru_order.pop_front() else {
                break;
            };
            evicted |= self.plans.remove(&oldest).is_some();
        }
        if evicted {
            self.fast_plans.clear();
            self.last = None;
        }
    }

    /// Probe the most-recent-entry fast path without LRU bookkeeping. A hit
    /// implies the canonical fusion-block route was already decided for this
    /// exact (rule, spaces, axes) key, so callers may skip the route cache.
    pub(crate) fn probe_last(
        &mut self,
        rule_key: &RuleKey,
        dst_space: &DynamicFusionMapSpace,
        lhs_space: &DynamicFusionMapSpace,
        rhs_space: &DynamicFusionMapSpace,
        axes: TensorContractAxisSpec<'_>,
    ) -> Option<Arc<CanonicalFusionBlockContractPlan>> {
        if !self.policy.stores_entries() {
            return None;
        }
        let plan = self.last.as_ref().and_then(|last| {
            if last.matches(rule_key, dst_space, lhs_space, rhs_space, axes) {
                Some(Arc::clone(&last.plan))
            } else {
                None
            }
        });
        if plan.is_some() {
            self.stats.hits += 1;
            self.stats.fast_hits += 1;
        }
        plan
    }

    pub(crate) fn get_or_compile<R>(
        &mut self,
        rule: &R,
        dst_space: &DynamicFusionMapSpace,
        lhs_space: &DynamicFusionMapSpace,
        rhs_space: &DynamicFusionMapSpace,
        axes: TensorContractAxisSpec<'_>,
    ) -> Result<Arc<CanonicalFusionBlockContractPlan>, OperationError>
    where
        R: MultiplicityFreeRigidSymbols<Scalar = f64> + TreeTransformRuleCacheKey<Key = RuleKey>,
    {
        let rule_key = rule.tree_transform_rule_cache_key();
        let raw_axes = RawTensorContractAxisSpecKey::from_axes(axes);
        if self.policy.stores_entries() {
            let refresh_lru = self.policy.max_entries().is_some();
            let last_hit = self.last.as_ref().and_then(|last| {
                if last.matches(&rule_key, dst_space, lhs_space, rhs_space, axes) {
                    Some((
                        refresh_lru.then(|| last.key.clone()).flatten(),
                        Arc::clone(&last.plan),
                    ))
                } else {
                    None
                }
            });
            if let Some((key, plan)) = last_hit {
                self.stats.hits += 1;
                self.stats.fast_hits += 1;
                if let Some(key) = key.as_ref() {
                    self.touch_plan(key);
                }
                return Ok(plan);
            }
        }
        let axis_plan = TensorContractAxisPlan::compile(
            lhs_space.rank(),
            rhs_space.rank(),
            dst_space.rank(),
            axes,
        )?;
        let axes_key = OwnedTensorContractAxisSpec::new_with_conjugation(
            axis_plan.lhs_contracting_axes,
            axis_plan.rhs_contracting_axes,
            axis_plan.output_axes,
            axis_plan.lhs_conjugate,
            axis_plan.rhs_conjugate,
        );
        let fast_key = CanonicalFusionBlockContractFastKey {
            rule: rule_key.clone(),
            dst: CanonicalFusionBlockFastSpaceKey::from_space(dst_space),
            lhs: CanonicalFusionBlockFastSpaceKey::from_space(lhs_space),
            rhs: CanonicalFusionBlockFastSpaceKey::from_space(rhs_space),
            axes: axes_key.clone(),
        };
        if self.policy.stores_entries() {
            let lru_key = if self.policy.max_entries().is_some() {
                Some(CanonicalFusionBlockContractCacheKey::from_parts(
                    rule_key.clone(),
                    dst_space,
                    lhs_space,
                    rhs_space,
                    axes_key.clone(),
                )?)
            } else {
                None
            };
            if let Some(plan) = self.fast_plans.get(&fast_key) {
                let plan = Arc::clone(plan);
                self.stats.hits += 1;
                self.stats.fast_hits += 1;
                if let Some(key) = lru_key.as_ref() {
                    self.touch_plan(key);
                }
                self.last = Some(CanonicalFusionBlockContractLastEntry {
                    key: lru_key,
                    rule: rule_key,
                    dst: CanonicalFusionBlockLastSpaceKey::from_space(dst_space),
                    lhs: CanonicalFusionBlockLastSpaceKey::from_space(lhs_space),
                    rhs: CanonicalFusionBlockLastSpaceKey::from_space(rhs_space),
                    axes: raw_axes,
                    plan: Arc::clone(&plan),
                });
                return Ok(plan);
            }
        }

        let key = CanonicalFusionBlockContractCacheKey::from_parts(
            rule_key.clone(),
            dst_space,
            lhs_space,
            rhs_space,
            axes_key,
        )?;
        if !self.policy.stores_entries() {
            self.stats.misses += 1;
            return Ok(Arc::new(compile_canonical_fusion_block_contract_plan(
                rule, dst_space, lhs_space, rhs_space, axes,
            )?));
        }
        if let Some(plan) = self.plans.get(&key) {
            self.stats.hits += 1;
            let plan = Arc::clone(plan);
            self.touch_plan(&key);
            self.fast_plans.insert(fast_key, Arc::clone(&plan));
            self.last = Some(CanonicalFusionBlockContractLastEntry {
                key: Some(key.clone()),
                rule: rule_key,
                dst: CanonicalFusionBlockLastSpaceKey::from_space(dst_space),
                lhs: CanonicalFusionBlockLastSpaceKey::from_space(lhs_space),
                rhs: CanonicalFusionBlockLastSpaceKey::from_space(rhs_space),
                axes: raw_axes,
                plan: Arc::clone(&plan),
            });
            return Ok(plan);
        } else {
            self.stats.misses += 1;
            let plan = compile_canonical_fusion_block_contract_plan(
                rule, dst_space, lhs_space, rhs_space, axes,
            )?;
            let plan = Arc::new(plan);
            let last_key = key.clone();
            self.insert_plan(key, fast_key, Arc::clone(&plan));
            self.last = Some(CanonicalFusionBlockContractLastEntry {
                key: Some(last_key),
                rule: rule_key,
                dst: CanonicalFusionBlockLastSpaceKey::from_space(dst_space),
                lhs: CanonicalFusionBlockLastSpaceKey::from_space(lhs_space),
                rhs: CanonicalFusionBlockLastSpaceKey::from_space(rhs_space),
                axes: raw_axes,
                plan: Arc::clone(&plan),
            });
            return Ok(plan);
        }
    }
}

#[derive(Clone, Debug)]
struct CanonicalFusionBlockContractLastEntry<RuleKey> {
    key: Option<CanonicalFusionBlockContractCacheKey<RuleKey>>,
    rule: RuleKey,
    dst: CanonicalFusionBlockLastSpaceKey,
    lhs: CanonicalFusionBlockLastSpaceKey,
    rhs: CanonicalFusionBlockLastSpaceKey,
    axes: RawTensorContractAxisSpecKey,
    plan: Arc<CanonicalFusionBlockContractPlan>,
}

impl<RuleKey> CanonicalFusionBlockContractLastEntry<RuleKey>
where
    RuleKey: Eq,
{
    fn matches(
        &self,
        rule: &RuleKey,
        dst: &DynamicFusionMapSpace,
        lhs: &DynamicFusionMapSpace,
        rhs: &DynamicFusionMapSpace,
        axes: TensorContractAxisSpec<'_>,
    ) -> bool {
        &self.rule == rule
            && self.dst.matches(dst)
            && self.lhs.matches(lhs)
            && self.rhs.matches(rhs)
            && self.axes.matches(axes)
    }
}

#[derive(Clone, Debug)]
struct CanonicalFusionBlockLastSpaceKey {
    nout: usize,
    homspace: Arc<FusionTreeHomSpace>,
    structure: Arc<BlockStructure>,
}

impl CanonicalFusionBlockLastSpaceKey {
    fn from_space(space: &DynamicFusionMapSpace) -> Self {
        Self {
            nout: space.nout(),
            homspace: Arc::clone(space.homspace_arc()),
            structure: Arc::clone(space.structure()),
        }
    }

    fn matches(&self, space: &DynamicFusionMapSpace) -> bool {
        self.nout == space.nout()
            && Arc::ptr_eq(&self.structure, space.structure())
            && (Arc::ptr_eq(&self.homspace, space.homspace_arc())
                || *self.homspace == *space.homspace())
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct RawTensorContractAxisSpecKey {
    lhs_contracting_axes: Vec<usize>,
    rhs_contracting_axes: Vec<usize>,
    output_permutation: RawAxisPermutationKey,
    lhs_conjugate: bool,
    rhs_conjugate: bool,
}

impl RawTensorContractAxisSpecKey {
    fn from_axes(axes: TensorContractAxisSpec<'_>) -> Self {
        Self {
            lhs_contracting_axes: axes.lhs_contracting_axes().to_vec(),
            rhs_contracting_axes: axes.rhs_contracting_axes().to_vec(),
            output_permutation: RawAxisPermutationKey::from_axes(axes.output_permutation()),
            lhs_conjugate: axes.lhs_conjugate(),
            rhs_conjugate: axes.rhs_conjugate(),
        }
    }

    fn matches(&self, axes: TensorContractAxisSpec<'_>) -> bool {
        self.lhs_contracting_axes == axes.lhs_contracting_axes()
            && self.rhs_contracting_axes == axes.rhs_contracting_axes()
            && self.output_permutation.matches(axes.output_permutation())
            && self.lhs_conjugate == axes.lhs_conjugate()
            && self.rhs_conjugate == axes.rhs_conjugate()
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
enum RawAxisPermutationKey {
    Identity,
    Axes(Vec<usize>),
}

impl RawAxisPermutationKey {
    fn from_axes(axes: AxisPermutation<'_>) -> Self {
        match axes {
            AxisPermutation::Identity => Self::Identity,
            AxisPermutation::Axes(axes) => Self::Axes(axes.to_vec()),
        }
    }

    fn matches(&self, axes: AxisPermutation<'_>) -> bool {
        match (self, axes) {
            (Self::Identity, AxisPermutation::Identity) => true,
            (Self::Axes(stored), AxisPermutation::Axes(axes)) => stored == axes,
            _ => false,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Hash)]
struct CanonicalFusionBlockContractFastKey<RuleKey> {
    rule: RuleKey,
    dst: CanonicalFusionBlockFastSpaceKey,
    lhs: CanonicalFusionBlockFastSpaceKey,
    rhs: CanonicalFusionBlockFastSpaceKey,
    axes: OwnedTensorContractAxisSpec,
}

#[derive(Clone, Debug, Eq, PartialEq, Hash)]
struct CanonicalFusionBlockFastSpaceKey {
    nout: usize,
    homspace: FusionTreeHomSpace,
    structure_ptr: usize,
}

impl CanonicalFusionBlockFastSpaceKey {
    fn from_space(space: &DynamicFusionMapSpace) -> Self {
        Self {
            nout: space.nout(),
            homspace: space.homspace().clone(),
            structure_ptr: Arc::as_ptr(space.structure()) as usize,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Hash)]
struct CanonicalFusionBlockContractCacheKey<RuleKey> {
    rule: RuleKey,
    dst_nout: usize,
    dst_homspace: FusionTreeHomSpace,
    dst_structure: BlockStructureCacheKey,
    lhs_nout: usize,
    lhs_homspace: FusionTreeHomSpace,
    lhs_structure: BlockStructureCacheKey,
    rhs_nout: usize,
    rhs_homspace: FusionTreeHomSpace,
    rhs_structure: BlockStructureCacheKey,
    axes: OwnedTensorContractAxisSpec,
}

impl<RuleKey> CanonicalFusionBlockContractCacheKey<RuleKey>
where
    RuleKey: Clone + Eq + Hash,
{
    fn from_parts(
        rule: RuleKey,
        dst: &DynamicFusionMapSpace,
        lhs: &DynamicFusionMapSpace,
        rhs: &DynamicFusionMapSpace,
        axes: OwnedTensorContractAxisSpec,
    ) -> Result<Self, OperationError> {
        Ok(Self {
            rule,
            dst_nout: dst.nout(),
            dst_homspace: dst.homspace().clone(),
            dst_structure: BlockStructureCacheKey::from_structure(dst.structure())?,
            lhs_nout: lhs.nout(),
            lhs_homspace: lhs.homspace().clone(),
            lhs_structure: BlockStructureCacheKey::from_structure(lhs.structure())?,
            rhs_nout: rhs.nout(),
            rhs_homspace: rhs.homspace().clone(),
            rhs_structure: BlockStructureCacheKey::from_structure(rhs.structure())?,
            axes,
        })
    }
}

#[derive(Clone, Debug)]
struct FusionBlockMatrixLayout {
    groups: Vec<FusionBlockMatrixGroup>,
}

impl FusionBlockMatrixLayout {
    fn compile<R>(rule: &R, space: &DynamicFusionMapSpace) -> Result<Self, OperationError>
    where
        R: MultiplicityFreeRigidSymbols<Scalar = f64>,
    {
        let mut builders = Vec::<FusionBlockMatrixGroupBuilder>::new();
        let mut group_indices = HashMap::<SectorId, usize>::new();
        for block_index in 0..space.structure().block_count() {
            let block = space.structure().block(block_index)?;
            let BlockKey::FusionTree(key) = block.key() else {
                return Err(OperationError::ExpectedFusionTreeBlock {
                    tensor: "fusion",
                    index: block_index,
                });
            };
            let coupled = coupled_sector(rule, key.codomain_tree());
            if coupled != coupled_sector(rule, key.domain_tree()) {
                return Err(OperationError::FusionTreeGroupMismatch {
                    tensor: "fusion",
                    index: block_index,
                });
            }
            let group_index = if let Some(&group_index) = group_indices.get(&coupled) {
                group_index
            } else {
                let group_index = builders.len();
                group_indices.insert(coupled, group_index);
                builders.push(FusionBlockMatrixGroupBuilder::new(coupled));
                group_index
            };
            let row_dim = element_count(&block.shape()[..space.nout()])?;
            let col_dim = element_count(&block.shape()[space.nout()..])?;
            builders[group_index].add_tree_pair(
                key.codomain_tree().clone(),
                row_dim,
                key.domain_tree().clone(),
                col_dim,
                block_index,
            )?;
        }
        let mut groups = Vec::with_capacity(builders.len());
        for builder in builders {
            groups.push(builder.finish(rule, space)?);
        }
        Ok(Self { groups })
    }

    fn group(&self, coupled: SectorId) -> Option<&FusionBlockMatrixGroup> {
        self.groups.iter().find(|group| group.coupled == coupled)
    }
}

#[derive(Clone, Debug)]
struct FusionBlockMatrixGroupBuilder {
    coupled: SectorId,
    row_offsets: HashMap<FusionTreeKey, TreeMatrixOffset>,
    col_offsets: HashMap<FusionTreeKey, TreeMatrixOffset>,
    tree_pairs: HashSet<(FusionTreeKey, FusionTreeKey)>,
    blocks: Vec<usize>,
    occupied_elements: usize,
    rows: usize,
    cols: usize,
}

impl FusionBlockMatrixGroupBuilder {
    fn new(coupled: SectorId) -> Self {
        Self {
            coupled,
            row_offsets: HashMap::new(),
            col_offsets: HashMap::new(),
            tree_pairs: HashSet::new(),
            blocks: Vec::new(),
            occupied_elements: 0,
            rows: 0,
            cols: 0,
        }
    }

    fn add_tree_pair(
        &mut self,
        row_tree: FusionTreeKey,
        row_dim: usize,
        col_tree: FusionTreeKey,
        col_dim: usize,
        block_index: usize,
    ) -> Result<(), OperationError> {
        if !self.tree_pairs.insert((row_tree.clone(), col_tree.clone())) {
            return Err(OperationError::StructureMismatch { tensor: "fusion" });
        }
        match self.row_offsets.get(&row_tree) {
            Some(offset) if offset.dim != row_dim => {
                return Err(OperationError::ShapeMismatch {
                    dst: vec![offset.dim],
                    src: vec![row_dim],
                });
            }
            Some(_) => {}
            None => {
                let offset = self.rows;
                self.rows = self
                    .rows
                    .checked_add(row_dim)
                    .ok_or(OperationError::ElementCountOverflow)?;
                self.row_offsets.insert(
                    row_tree,
                    TreeMatrixOffset {
                        offset,
                        dim: row_dim,
                    },
                );
            }
        }
        match self.col_offsets.get(&col_tree) {
            Some(offset) if offset.dim != col_dim => {
                return Err(OperationError::ShapeMismatch {
                    dst: vec![offset.dim],
                    src: vec![col_dim],
                });
            }
            Some(_) => {}
            None => {
                let offset = self.cols;
                self.cols = self
                    .cols
                    .checked_add(col_dim)
                    .ok_or(OperationError::ElementCountOverflow)?;
                self.col_offsets.insert(
                    col_tree,
                    TreeMatrixOffset {
                        offset,
                        dim: col_dim,
                    },
                );
            }
        }
        let block_elements = row_dim
            .checked_mul(col_dim)
            .ok_or(OperationError::ElementCountOverflow)?;
        self.occupied_elements = self
            .occupied_elements
            .checked_add(block_elements)
            .ok_or(OperationError::ElementCountOverflow)?;
        self.blocks.push(block_index);
        Ok(())
    }

    fn finish<R>(
        self,
        rule: &R,
        space: &DynamicFusionMapSpace,
    ) -> Result<FusionBlockMatrixGroup, OperationError>
    where
        R: MultiplicityFreeRigidSymbols<Scalar = f64>,
    {
        let mut subblocks = Vec::with_capacity(self.blocks.len());
        let block_indices = self.blocks;
        for &block_index in &block_indices {
            let block = space.structure().block(block_index)?;
            let BlockKey::FusionTree(key) = block.key() else {
                return Err(OperationError::ExpectedFusionTreeBlock {
                    tensor: "fusion",
                    index: block_index,
                });
            };
            let row = self
                .row_offsets
                .get(key.codomain_tree())
                .expect("row tree offset collected before finish");
            let col = self
                .col_offsets
                .get(key.domain_tree())
                .expect("column tree offset collected before finish");
            let mut matrix_strides = Vec::<isize>::with_capacity(block.shape().len());
            matrix_strides.extend(column_major_strides_isize(&block.shape()[..space.nout()])?);
            let domain_strides = column_major_strides_usize(&block.shape()[space.nout()..])?;
            for stride in domain_strides {
                let matrix_stride = stride
                    .checked_mul(self.rows)
                    .ok_or(OperationError::ElementCountOverflow)?;
                matrix_strides.push(isize::try_from(matrix_stride).map_err(|_| {
                    OperationError::StrideOverflow {
                        value: matrix_stride,
                    }
                })?);
            }
            let matrix_offset = col
                .offset
                .checked_mul(self.rows)
                .and_then(|offset| offset.checked_add(row.offset))
                .ok_or(OperationError::ElementCountOverflow)?;
            let matrix_offset = offset_to_isize(matrix_offset)?;
            // Coefficient-free by contract (TensorKit mul! parity): fermionic
            // supertrace twists are applied during rhs materialization on the
            // dynamic route, never inside the GEMM plan.
            let coefficient = rule.scalar_one();
            subblocks.push(FusionSubblockMatrixLayout {
                block: FusionStridedBlockLayout {
                    shape: block.shape().to_vec(),
                    strides: strides_to_isize(block.strides())?,
                    offset: offset_to_isize(block.offset())?,
                },
                matrix_offset,
                matrix_strides,
                coefficient,
            });
        }
        let matrix_elements = self
            .rows
            .checked_mul(self.cols)
            .ok_or(OperationError::ElementCountOverflow)?;
        let covers_matrix = self.occupied_elements == matrix_elements;
        let direct_offset = direct_group_matrix_offset(&subblocks, covers_matrix);
        Ok(FusionBlockMatrixGroup {
            coupled: self.coupled,
            rows: self.rows,
            cols: self.cols,
            needs_clear: !covers_matrix,
            direct_offset,
            block_indices,
            subblocks,
        })
    }
}

#[derive(Clone, Copy, Debug)]
struct TreeMatrixOffset {
    offset: usize,
    dim: usize,
}

fn coupled_sector<R>(rule: &R, tree: &FusionTreeKey) -> SectorId
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64>,
{
    tree.coupled().unwrap_or_else(|| rule.vacuum())
}
