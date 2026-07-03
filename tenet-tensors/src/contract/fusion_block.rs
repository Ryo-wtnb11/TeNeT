use std::collections::{HashMap, HashSet, VecDeque};
use std::hash::Hash;
use std::sync::Arc;

use num_traits::{One, Zero};
use tenet_core::{
    BlockKey, BlockStructure, FusionTreeHomSpace, FusionTreeKey, HostReadableStorage,
    HostWritableStorage, MultiplicityFreeRigidSymbols, Placement, ScratchStorage, SectorId,
    SimilarStorage, TensorStorage,
};

use crate::axis::{AxisPermutation, OwnedTensorContractAxisSpec, TensorContractAxisSpec};
use crate::cache::{
    rebuild_lru_order_from_keys, touch_lru_key, BlockStructureCacheKey, OperationCachePolicy,
};
use crate::host_scratch::HostScratchBuffer;
use crate::storage_scratch::{
    FusionBlockContractScratchBuffers, StorageFusionBlockContractWorkspace,
};
use crate::strided::{
    column_major_strides_isize, column_major_strides_usize, element_count, offset_to_isize,
    strides_to_isize,
};
use crate::structure_identity::validate_structure_identity;
use crate::{
    DenseBlockScalar, HostKernelAdapter, OperationError, RecouplingCoefficientAction,
    ReportsPlacement, TreeTransformRuleCacheKey,
};

use super::backend::TensorContractBackend;
use super::dynamic_space::DynamicFusionMapSpace;
use super::fusion::{reject_fusion_contract_conjugation, rhs_contract_twist_factor};
use super::profile::TensorContractFusionProfile;
use super::structure::TensorContractAxisPlan;

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
        CanonicalFusionBlockContractPlan::compile(rule, dst_space, lhs_space, rhs_space, axes)?;
    let mut fusion_workspace = CanonicalFusionBlockContractWorkspace::<D>::default();
    plan.execute_raw(
        kernels,
        backend,
        workspace,
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

/// Host scratch workspace for canonical fusion-block contraction.
///
/// This workspace packs blocks into host `Vec<T>` buffers before dense replay.
/// Device execution needs a separate device workspace.
#[derive(Clone, Debug)]
pub(crate) struct HostCanonicalFusionBlockContractWorkspace<T> {
    buffers: HostFusionBlockContractBuffers<T>,
}

pub(crate) type CanonicalFusionBlockContractWorkspace<T> =
    HostCanonicalFusionBlockContractWorkspace<T>;

impl<T> Default for HostCanonicalFusionBlockContractWorkspace<T> {
    fn default() -> Self {
        Self {
            buffers: HostFusionBlockContractBuffers::default(),
        }
    }
}

impl<T> ReportsPlacement for HostCanonicalFusionBlockContractWorkspace<T> {
    #[inline]
    fn placement(&self) -> Placement {
        Placement::Host
    }
}

#[derive(Clone, Debug)]
struct HostFusionBlockContractBuffers<T> {
    packed: FusionBlockContractScratchBuffers<
        HostScratchBuffer<T>,
        HostScratchBuffer<T>,
        HostScratchBuffer<T>,
    >,
}

impl<T> Default for HostFusionBlockContractBuffers<T> {
    fn default() -> Self {
        Self {
            packed: FusionBlockContractScratchBuffers::default(),
        }
    }
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
        let plan = CanonicalFusionBlockContractPlan::compile(
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
        let plan = CanonicalFusionBlockContractPlan::compile(
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
            &mut backend,
            &mut workspace,
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
        let plan = CanonicalFusionBlockContractPlan::compile(
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
            &mut backend,
            &mut workspace,
            &mut fusion_workspace,
            &mut dst,
            &lhs,
            &rhs,
            2.0,
            3.0,
        )
        .unwrap();

        assert_eq!(dst.data(), &[50.0, 102.0]);
        // The second group reuses the first group's buffers (same placement),
        // so exactly one allocation per slot happens.
        assert_eq!(
            allocations.borrow().as_slice(),
            &[
                ScratchAllocation {
                    label: "lhs",
                    len: 1,
                },
                ScratchAllocation {
                    label: "rhs",
                    len: 1,
                },
                ScratchAllocation {
                    label: "destination",
                    len: 1,
                },
            ],
        );
    }
}

#[derive(Clone, Debug)]
pub(crate) struct CanonicalFusionBlockContractPlan {
    dst_structure: Arc<BlockStructure>,
    lhs_structure: Arc<BlockStructure>,
    rhs_structure: Arc<BlockStructure>,
    inactive_dst_scale_blocks: Vec<FusionScaleBlockLayout>,
    groups: Vec<CanonicalFusionBlockContractGroupPlan>,
}

impl CanonicalFusionBlockContractPlan {
    pub(crate) fn compile<R>(
        rule: &R,
        dst_space: &DynamicFusionMapSpace,
        lhs_space: &DynamicFusionMapSpace,
        rhs_space: &DynamicFusionMapSpace,
        axes: TensorContractAxisSpec<'_>,
    ) -> Result<Self, OperationError>
    where
        R: MultiplicityFreeRigidSymbols<Scalar = f64>,
    {
        reject_fusion_contract_conjugation(axes)?;
        validate_canonical_compose(rule, dst_space, lhs_space, rhs_space, axes)?;
        let axis_plan = TensorContractAxisPlan::compile(
            lhs_space.rank(),
            rhs_space.rank(),
            dst_space.rank(),
            axes,
        )?;

        let lhs_layout = FusionBlockMatrixLayout::compile(rule, lhs_space, None)?;
        let rhs_layout = FusionBlockMatrixLayout::compile(
            rule,
            rhs_space,
            Some(&axis_plan.rhs_contracting_axes),
        )?;
        let dst_layout = FusionBlockMatrixLayout::compile(rule, dst_space, None)?;

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
            groups.push(CanonicalFusionBlockContractGroupPlan::compile(
                lhs_group,
                rhs_group.clone(),
                dst_group.clone(),
            )?);
        }
        Ok(Self {
            dst_structure: Arc::clone(dst_space.structure()),
            lhs_structure: Arc::clone(lhs_space.structure()),
            rhs_structure: Arc::clone(rhs_space.structure()),
            inactive_dst_scale_blocks: fusion_scale_block_layouts_excluding(
                dst_space.structure(),
                &active_dst_blocks,
            )?,
            groups,
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn execute_raw<A, B, D>(
        &self,
        kernels: &mut A,
        backend: &mut B,
        workspace: &mut B::Workspace,
        fusion_workspace: &mut CanonicalFusionBlockContractWorkspace<D>,
        dst_structure: &Arc<BlockStructure>,
        dst_data: &mut [D],
        lhs_structure: &Arc<BlockStructure>,
        lhs_data: &[D],
        rhs_structure: &Arc<BlockStructure>,
        rhs_data: &[D],
        alpha: D,
        beta: D,
    ) -> Result<(), OperationError>
    where
        A: HostKernelAdapter<D>,
        B: TensorContractBackend<D, f64>,
        D: DenseBlockScalar + RecouplingCoefficientAction<f64>,
    {
        self.validate_replay_inputs(
            dst_structure,
            dst_data.len(),
            lhs_structure,
            lhs_data.len(),
            rhs_structure,
            rhs_data.len(),
        )?;
        scale_all_blocks(kernels, &self.inactive_dst_scale_blocks, dst_data, beta)?;

        let trivial_scale = alpha.is_one() && beta.is_zero();
        for group in &self.groups {
            if !group.is_fully_direct(trivial_scale) {
                fusion_workspace
                    .buffers
                    .prepare(group.lhs.rows, group.lhs.cols, group.rhs.cols)?;
                fusion_workspace
                    .buffers
                    .clear_inputs(group.lhs.needs_clear, group.rhs.needs_clear);
            }
            execute_group_with_scratch_buffers(
                kernels,
                backend,
                workspace,
                group,
                &mut fusion_workspace.buffers.packed,
                dst_data,
                lhs_data,
                rhs_data,
                alpha,
                beta,
            )?;
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn execute_raw_profiled<A, B, D>(
        &self,
        kernels: &mut A,
        backend: &mut B,
        workspace: &mut B::Workspace,
        fusion_workspace: &mut CanonicalFusionBlockContractWorkspace<D>,
        dst_structure: &Arc<BlockStructure>,
        dst_data: &mut [D],
        lhs_structure: &Arc<BlockStructure>,
        lhs_data: &[D],
        rhs_structure: &Arc<BlockStructure>,
        rhs_data: &[D],
        alpha: D,
        beta: D,
        profile: &mut TensorContractFusionProfile,
    ) -> Result<(), OperationError>
    where
        A: HostKernelAdapter<D>,
        B: TensorContractBackend<D, f64>,
        D: DenseBlockScalar + RecouplingCoefficientAction<f64>,
    {
        let total_start = std::time::Instant::now();

        let start = std::time::Instant::now();
        self.validate_replay_inputs(
            dst_structure,
            dst_data.len(),
            lhs_structure,
            lhs_data.len(),
            rhs_structure,
            rhs_data.len(),
        )?;
        profile.canonical_validate += start.elapsed();

        let start = std::time::Instant::now();
        scale_all_blocks(kernels, &self.inactive_dst_scale_blocks, dst_data, beta)?;
        profile.canonical_scale += start.elapsed();

        let trivial_scale = alpha.is_one() && beta.is_zero();
        for group in &self.groups {
            profile.canonical_contract_groups += 1;

            if !group.is_fully_direct(trivial_scale) {
                let start = std::time::Instant::now();
                fusion_workspace
                    .buffers
                    .prepare(group.lhs.rows, group.lhs.cols, group.rhs.cols)?;
                fusion_workspace
                    .buffers
                    .clear_inputs(group.lhs.needs_clear, group.rhs.needs_clear);
                profile.canonical_workspace_prepare += start.elapsed();
            }

            if group.lhs.direct_offset.is_none() {
                let start = std::time::Instant::now();
                pack_group(
                    kernels,
                    &group.lhs,
                    lhs_data,
                    fusion_workspace.buffers.packed.lhs_mut().as_mut_slice(),
                )?;
                profile.canonical_pack_lhs += start.elapsed();
            } else {
                profile.canonical_direct_pack_skips += 1;
            }

            if group.rhs.direct_offset.is_none() {
                let start = std::time::Instant::now();
                pack_group(
                    kernels,
                    &group.rhs,
                    rhs_data,
                    fusion_workspace.buffers.packed.rhs_mut().as_mut_slice(),
                )?;
                profile.canonical_pack_rhs += start.elapsed();
            } else {
                profile.canonical_direct_pack_skips += 1;
            }

            let dst_direct = if trivial_scale {
                group.dst.direct_offset
            } else {
                None
            };
            let start = std::time::Instant::now();
            {
                let (lhs, rhs, dst) = fusion_workspace.buffers.packed.inputs_and_destination_mut();
                let lhs_slice = direct_or_scratch_slice(
                    lhs_data,
                    group.lhs.direct_offset,
                    group.lhs.rows,
                    group.lhs.cols,
                    lhs.as_slice(),
                )?;
                let rhs_slice = direct_or_scratch_slice(
                    rhs_data,
                    group.rhs.direct_offset,
                    group.rhs.rows,
                    group.rhs.cols,
                    rhs.as_slice(),
                )?;
                match dst_direct {
                    Some(base) => {
                        let dst_slice =
                            direct_slice_mut(dst_data, base, group.dst.rows, group.dst.cols)?;
                        matmul_group_plan(
                            backend, workspace, group, lhs_slice, rhs_slice, dst_slice,
                        )?;
                        profile.canonical_direct_gemm_groups += 1;
                    }
                    None => {
                        matmul_group_plan(
                            backend,
                            workspace,
                            group,
                            lhs_slice,
                            rhs_slice,
                            dst.as_mut_slice(),
                        )?;
                    }
                }
            }
            profile.canonical_matmul += start.elapsed();

            if dst_direct.is_none() {
                let start = std::time::Instant::now();
                scatter_group(
                    kernels,
                    &group.dst,
                    dst_data,
                    fusion_workspace.buffers.packed.destination().as_slice(),
                    alpha,
                    beta,
                )?;
                profile.canonical_scatter += start.elapsed();
            }
        }

        profile.canonical_contract_total += total_start.elapsed();
        Ok(())
    }

    #[allow(dead_code)]
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn execute_storage_workspace<
        A,
        B,
        D,
        const DST_NOUT: usize,
        const DST_NIN: usize,
        const LHS_NOUT: usize,
        const LHS_NIN: usize,
        const RHS_NOUT: usize,
        const RHS_NIN: usize,
        SDst,
        SLhs,
        SRhs,
        DDst,
        DLhs,
        DRhs,
    >(
        &self,
        kernels: &mut A,
        backend: &mut B,
        workspace: &mut B::Workspace,
        fusion_workspace: &mut StorageFusionBlockContractWorkspace<
            DLhs::Similar,
            DRhs::Similar,
            DDst::Similar,
        >,
        dst: &mut tenet_core::TensorMap<D, DST_NOUT, DST_NIN, SDst, DDst>,
        lhs: &tenet_core::TensorMap<D, LHS_NOUT, LHS_NIN, SLhs, DLhs>,
        rhs: &tenet_core::TensorMap<D, RHS_NOUT, RHS_NIN, SRhs, DRhs>,
        alpha: D,
        beta: D,
    ) -> Result<(), OperationError>
    where
        A: HostKernelAdapter<D>,
        B: TensorContractBackend<D, f64>,
        D: DenseBlockScalar + RecouplingCoefficientAction<f64>,
        DDst: HostWritableStorage<D> + SimilarStorage<D>,
        DDst::Similar: HostWritableStorage<D> + ScratchStorage<D>,
        DLhs: HostReadableStorage<D> + SimilarStorage<D>,
        DLhs::Similar: HostWritableStorage<D> + ScratchStorage<D>,
        DRhs: HostReadableStorage<D> + SimilarStorage<D>,
        DRhs::Similar: HostWritableStorage<D> + ScratchStorage<D>,
    {
        let dst_structure = Arc::clone(dst.structure());
        let lhs_structure = Arc::clone(lhs.structure());
        let rhs_structure = Arc::clone(rhs.structure());
        self.validate_replay_inputs(
            &dst_structure,
            dst.storage().len(),
            &lhs_structure,
            lhs.storage().len(),
            &rhs_structure,
            rhs.storage().len(),
        )?;
        scale_all_blocks(
            kernels,
            &self.inactive_dst_scale_blocks,
            dst.data_mut(),
            beta,
        )?;

        let lhs_data = lhs.data();
        let rhs_data = rhs.data();
        for group in &self.groups {
            let lens =
                fusion_block_group_scratch_lens(group.lhs.rows, group.lhs.cols, group.rhs.cols)?;
            fusion_workspace.prepare_from_storages(
                lhs.storage(),
                rhs.storage(),
                dst.storage(),
                lens.lhs,
                lens.rhs,
                lens.destination,
                D::zero(),
            );
            execute_group_with_scratch_buffers(
                kernels,
                backend,
                workspace,
                group,
                fusion_workspace.buffers_mut(),
                dst.data_mut(),
                lhs_data,
                rhs_data,
                alpha,
                beta,
            )?;
        }
        Ok(())
    }

    /// Storage-aware raw replay for callers whose operands are scratch buffers
    /// rather than `TensorMap`s (the dynamic canonical route).
    ///
    /// Pack scratch allocation origins are passed explicitly: LHS pack scratch
    /// from `lhs_alloc`, RHS pack scratch from `rhs_alloc`, and matmul output
    /// scratch from `dst_alloc`, while replay itself consumes the raw slices.
    #[allow(dead_code)]
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn execute_storage_raw<A, B, D, SLhs, SRhs, SDst>(
        &self,
        kernels: &mut A,
        backend: &mut B,
        workspace: &mut B::Workspace,
        fusion_workspace: &mut StorageFusionBlockContractWorkspace<
            SLhs::Similar,
            SRhs::Similar,
            SDst::Similar,
        >,
        lhs_alloc: &SLhs,
        rhs_alloc: &SRhs,
        dst_alloc: &SDst,
        dst_structure: &Arc<BlockStructure>,
        dst_data: &mut [D],
        lhs_structure: &Arc<BlockStructure>,
        lhs_data: &[D],
        rhs_structure: &Arc<BlockStructure>,
        rhs_data: &[D],
        alpha: D,
        beta: D,
    ) -> Result<(), OperationError>
    where
        A: HostKernelAdapter<D>,
        B: TensorContractBackend<D, f64>,
        D: DenseBlockScalar + RecouplingCoefficientAction<f64>,
        SLhs: SimilarStorage<D>,
        SLhs::Similar: HostWritableStorage<D> + ScratchStorage<D>,
        SRhs: SimilarStorage<D>,
        SRhs::Similar: HostWritableStorage<D> + ScratchStorage<D>,
        SDst: SimilarStorage<D>,
        SDst::Similar: HostWritableStorage<D> + ScratchStorage<D>,
    {
        self.validate_replay_inputs(
            dst_structure,
            dst_data.len(),
            lhs_structure,
            lhs_data.len(),
            rhs_structure,
            rhs_data.len(),
        )?;
        scale_all_blocks(kernels, &self.inactive_dst_scale_blocks, dst_data, beta)?;

        for group in &self.groups {
            let lens =
                fusion_block_group_scratch_lens(group.lhs.rows, group.lhs.cols, group.rhs.cols)?;
            fusion_workspace.prepare_from_storages(
                lhs_alloc,
                rhs_alloc,
                dst_alloc,
                lens.lhs,
                lens.rhs,
                lens.destination,
                D::zero(),
            );
            execute_group_with_scratch_buffers(
                kernels,
                backend,
                workspace,
                group,
                fusion_workspace.buffers_mut(),
                dst_data,
                lhs_data,
                rhs_data,
                alpha,
                beta,
            )?;
        }
        Ok(())
    }

    /// Storage-aware replay writing into a destination `TensorMap` while the
    /// LHS/RHS operands are raw canonical scratch slices (the dynamic route
    /// with an identity output transform).
    ///
    /// Pack scratch allocation origins: LHS pack from `lhs_alloc`, RHS pack
    /// from `rhs_alloc`, and matmul output scratch from the destination
    /// tensor's own storage.
    #[allow(dead_code)]
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn execute_storage_raw_sources<
        A,
        B,
        D,
        const DST_NOUT: usize,
        const DST_NIN: usize,
        SDst,
        SLhs,
        SRhs,
        DDst,
    >(
        &self,
        kernels: &mut A,
        backend: &mut B,
        workspace: &mut B::Workspace,
        fusion_workspace: &mut StorageFusionBlockContractWorkspace<
            SLhs::Similar,
            SRhs::Similar,
            DDst::Similar,
        >,
        lhs_alloc: &SLhs,
        rhs_alloc: &SRhs,
        dst: &mut tenet_core::TensorMap<D, DST_NOUT, DST_NIN, SDst, DDst>,
        lhs_structure: &Arc<BlockStructure>,
        lhs_data: &[D],
        rhs_structure: &Arc<BlockStructure>,
        rhs_data: &[D],
        alpha: D,
        beta: D,
    ) -> Result<(), OperationError>
    where
        A: HostKernelAdapter<D>,
        B: TensorContractBackend<D, f64>,
        D: DenseBlockScalar + RecouplingCoefficientAction<f64>,
        SLhs: SimilarStorage<D>,
        SLhs::Similar: HostWritableStorage<D> + ScratchStorage<D>,
        SRhs: SimilarStorage<D>,
        SRhs::Similar: HostWritableStorage<D> + ScratchStorage<D>,
        DDst: HostWritableStorage<D> + SimilarStorage<D>,
        DDst::Similar: HostWritableStorage<D> + ScratchStorage<D>,
    {
        let dst_structure = Arc::clone(dst.structure());
        self.validate_replay_inputs(
            &dst_structure,
            dst.storage().len(),
            lhs_structure,
            lhs_data.len(),
            rhs_structure,
            rhs_data.len(),
        )?;
        scale_all_blocks(
            kernels,
            &self.inactive_dst_scale_blocks,
            dst.data_mut(),
            beta,
        )?;

        for group in &self.groups {
            let lens =
                fusion_block_group_scratch_lens(group.lhs.rows, group.lhs.cols, group.rhs.cols)?;
            fusion_workspace.prepare_from_storages(
                lhs_alloc,
                rhs_alloc,
                dst.storage(),
                lens.lhs,
                lens.rhs,
                lens.destination,
                D::zero(),
            );
            execute_group_with_scratch_buffers(
                kernels,
                backend,
                workspace,
                group,
                fusion_workspace.buffers_mut(),
                dst.data_mut(),
                lhs_data,
                rhs_data,
                alpha,
                beta,
            )?;
        }
        Ok(())
    }

    /// Executes the contraction purely over storage handles.
    ///
    /// This is the device-side replay seam: the bounds require only
    /// [`TensorStorage`], so no host-slice contract leaks into the path. It
    /// supports exactly the fully-direct coupled-layout case with `alpha = 1`,
    /// `beta = 0` and no inactive destination blocks; every other case must
    /// use the host replay paths until the corresponding device kernels
    /// (pack/scatter, scale, tree transforms) exist behind their own seams.
    #[allow(dead_code)]
    pub(crate) fn execute_direct_on_storage<G, D, DDst, DLhs, DRhs>(
        &self,
        gemm: &mut G,
        dst: &mut DDst,
        lhs: &DLhs,
        rhs: &DRhs,
    ) -> Result<(), OperationError>
    where
        G: StorageGemm<D, DDst, DLhs, DRhs>,
        DDst: TensorStorage<D>,
        DLhs: TensorStorage<D>,
        DRhs: TensorStorage<D>,
    {
        if !self.inactive_dst_scale_blocks.is_empty() {
            return Err(OperationError::UnsupportedTensorContractScope {
                message: "storage-direct replay requires full destination coverage",
            });
        }
        for group in &self.groups {
            let (Some(lhs_base), Some(rhs_base), Some(dst_base)) = (
                group.lhs.direct_offset,
                group.rhs.direct_offset,
                group.dst.direct_offset,
            ) else {
                return Err(OperationError::UnsupportedTensorContractScope {
                    message: "storage-direct replay requires the coupled-sector matrix layout",
                });
            };
            validate_storage_range(lhs.len(), lhs_base, group.lhs.rows, group.lhs.cols)?;
            validate_storage_range(rhs.len(), rhs_base, group.rhs.rows, group.rhs.cols)?;
            validate_storage_range(dst.len(), dst_base, group.dst.rows, group.dst.cols)?;
            gemm.matmul_range_into(
                dst,
                dst_base,
                lhs,
                lhs_base,
                rhs,
                rhs_base,
                group.lhs.rows,
                group.lhs.cols,
                group.rhs.cols,
            )?;
        }
        Ok(())
    }

    fn validate_replay_structures(
        &self,
        dst_structure: &Arc<BlockStructure>,
        lhs_structure: &Arc<BlockStructure>,
        rhs_structure: &Arc<BlockStructure>,
    ) -> Result<(), OperationError> {
        validate_structure_identity("dst", &self.dst_structure, dst_structure)?;
        validate_structure_identity("lhs", &self.lhs_structure, lhs_structure)?;
        validate_structure_identity("rhs", &self.rhs_structure, rhs_structure)
    }

    fn validate_replay_inputs(
        &self,
        dst_structure: &Arc<BlockStructure>,
        dst_len: usize,
        lhs_structure: &Arc<BlockStructure>,
        lhs_len: usize,
        rhs_structure: &Arc<BlockStructure>,
        rhs_len: usize,
    ) -> Result<(), OperationError> {
        self.validate_replay_structures(dst_structure, lhs_structure, rhs_structure)?;
        validate_storage_len(dst_structure, dst_len)?;
        validate_storage_len(lhs_structure, lhs_len)?;
        validate_storage_len(rhs_structure, rhs_len)
    }
}

/// Placement-aware block GEMM over storage ranges.
///
/// The device-side replay seam for canonical fusion-block contraction:
/// `dst[dst_offset..][rows x cols] = lhs[lhs_offset..][rows x contracted] *
/// rhs[rhs_offset..][contracted x cols]` as column-major matrices, with no
/// host-slice contract in the trait. The host implementation wraps a
/// [`TensorContractBackend`]; device implementations submit kernels against
/// device storage handles.
pub(crate) trait StorageGemm<D, DDst, DLhs, DRhs> {
    #[allow(clippy::too_many_arguments)]
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
    ) -> Result<(), OperationError>;
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

fn validate_storage_range(
    storage_len: usize,
    base: usize,
    rows: usize,
    cols: usize,
) -> Result<(), OperationError> {
    let len = rows
        .checked_mul(cols)
        .ok_or(OperationError::ElementCountOverflow)?;
    let end = base
        .checked_add(len)
        .ok_or(OperationError::ElementCountOverflow)?;
    if end > storage_len {
        return Err(OperationError::ElementCountMismatch {
            expected: end,
            actual: storage_len,
        });
    }
    Ok(())
}

fn validate_storage_len(
    structure: &BlockStructure,
    actual_len: usize,
) -> Result<(), OperationError> {
    let expected = structure
        .required_len()
        .map_err(OperationError::from_core_preserving_context)?;
    if actual_len != expected {
        return Err(OperationError::ElementCountMismatch {
            expected,
            actual: actual_len,
        });
    }
    Ok(())
}

#[derive(Clone, Debug)]
struct CanonicalFusionBlockContractGroupPlan {
    lhs: FusionBlockMatrixGroup,
    rhs: FusionBlockMatrixGroup,
    dst: FusionBlockMatrixGroup,
}

impl CanonicalFusionBlockContractGroupPlan {
    /// True when GEMM can read both operands from storage and write the
    /// destination group matrix in place (no pack, no scatter).
    fn is_fully_direct(&self, trivial_scale: bool) -> bool {
        trivial_scale
            && self.lhs.direct_offset.is_some()
            && self.rhs.direct_offset.is_some()
            && self.dst.direct_offset.is_some()
    }

    fn compile(
        lhs: FusionBlockMatrixGroup,
        rhs: FusionBlockMatrixGroup,
        dst: FusionBlockMatrixGroup,
    ) -> Result<Self, OperationError> {
        if lhs.cols != rhs.rows {
            return Err(OperationError::ShapeMismatch {
                dst: vec![lhs.cols],
                src: vec![rhs.rows],
            });
        }
        if dst.rows != lhs.rows || dst.cols != rhs.cols {
            return Err(OperationError::ShapeMismatch {
                dst: vec![dst.rows, dst.cols],
                src: vec![lhs.rows, rhs.cols],
            });
        }

        Ok(Self { lhs, rhs, dst })
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
            return Ok(Arc::new(CanonicalFusionBlockContractPlan::compile(
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
            let plan = CanonicalFusionBlockContractPlan::compile(
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

impl<T> HostFusionBlockContractBuffers<T>
where
    T: Clone + Zero,
{
    fn prepare(
        &mut self,
        lhs_rows: usize,
        contracted: usize,
        rhs_cols: usize,
    ) -> Result<(), OperationError> {
        let lens = fusion_block_group_scratch_lens(lhs_rows, contracted, rhs_cols)?;
        self.packed.lhs_mut().resize_filled(lens.lhs, T::zero());
        self.packed.rhs_mut().resize_filled(lens.rhs, T::zero());
        self.packed
            .destination_mut()
            .resize_filled(lens.destination, T::zero());
        Ok(())
    }

    fn clear_inputs(&mut self, clear_lhs: bool, clear_rhs: bool) {
        if clear_lhs {
            self.packed.lhs_mut().fill(T::zero());
        }
        if clear_rhs {
            self.packed.rhs_mut().fill(T::zero());
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct FusionBlockContractScratchLens {
    lhs: usize,
    rhs: usize,
    destination: usize,
}

fn fusion_block_group_scratch_lens(
    lhs_rows: usize,
    contracted: usize,
    rhs_cols: usize,
) -> Result<FusionBlockContractScratchLens, OperationError> {
    let lhs = lhs_rows
        .checked_mul(contracted)
        .ok_or(OperationError::ElementCountOverflow)?;
    let rhs = contracted
        .checked_mul(rhs_cols)
        .ok_or(OperationError::ElementCountOverflow)?;
    let destination = lhs_rows
        .checked_mul(rhs_cols)
        .ok_or(OperationError::ElementCountOverflow)?;
    Ok(FusionBlockContractScratchLens {
        lhs,
        rhs,
        destination,
    })
}

#[derive(Clone, Debug)]
struct FusionBlockMatrixLayout {
    groups: Vec<FusionBlockMatrixGroup>,
}

impl FusionBlockMatrixLayout {
    fn compile<R>(
        rule: &R,
        space: &DynamicFusionMapSpace,
        rhs_contracting_axes: Option<&[usize]>,
    ) -> Result<Self, OperationError>
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
            groups.push(builder.finish(rule, space, rhs_contracting_axes)?);
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
        rhs_contracting_axes: Option<&[usize]>,
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
            let coefficient = if let Some(rhs_contracting_axes) = rhs_contracting_axes {
                rhs_contract_twist_factor(
                    rule,
                    space.homspace(),
                    rhs_contracting_axes,
                    key.codomain_tree(),
                )?
            } else {
                rule.scalar_one()
            };
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

#[derive(Clone, Debug)]
struct FusionBlockMatrixGroup {
    coupled: SectorId,
    rows: usize,
    cols: usize,
    // False only when the group's subblocks cover the packed matrix exactly.
    // Sparse fusion layouts keep this true so stale workspace cannot leak into GEMM.
    needs_clear: bool,
    // Storage offset of the group matrix when the operand's subblocks already
    // form it in place (coupled-sector matrix layout, unit coefficients):
    // packing is the identity copy and replay can hand storage to GEMM
    // directly.
    direct_offset: Option<usize>,
    block_indices: Vec<usize>,
    subblocks: Vec<FusionSubblockMatrixLayout>,
}

fn direct_group_matrix_offset(
    subblocks: &[FusionSubblockMatrixLayout],
    covers_matrix: bool,
) -> Option<usize> {
    if !covers_matrix {
        return None;
    }
    let mut base: Option<isize> = None;
    for subblock in subblocks {
        if subblock.coefficient != 1.0 {
            return None;
        }
        let strides_match = subblock
            .block
            .shape
            .iter()
            .zip(subblock.block.strides.iter().zip(&subblock.matrix_strides))
            .all(|(&dim, (&stride, &matrix_stride))| dim <= 1 || stride == matrix_stride);
        if !strides_match {
            return None;
        }
        let offset = subblock.block.offset - subblock.matrix_offset;
        if offset < 0 {
            return None;
        }
        match base {
            None => base = Some(offset),
            Some(existing) if existing != offset => return None,
            Some(_) => {}
        }
    }
    base.and_then(|offset| usize::try_from(offset).ok())
}

#[derive(Clone, Debug)]
struct FusionSubblockMatrixLayout {
    block: FusionStridedBlockLayout,
    matrix_offset: isize,
    matrix_strides: Vec<isize>,
    coefficient: f64,
}

#[derive(Clone, Debug)]
struct FusionStridedBlockLayout {
    shape: Vec<usize>,
    strides: Vec<isize>,
    offset: isize,
}

#[derive(Clone, Debug)]
struct FusionScaleBlockLayout {
    block: FusionStridedBlockLayout,
}

fn fusion_scale_block_layouts_excluding(
    structure: &BlockStructure,
    excluded_blocks: &HashSet<usize>,
) -> Result<Vec<FusionScaleBlockLayout>, OperationError> {
    let mut layouts = Vec::with_capacity(structure.block_count());
    for block_index in 0..structure.block_count() {
        if excluded_blocks.contains(&block_index) {
            continue;
        }
        let block = structure.block(block_index)?;
        layouts.push(FusionScaleBlockLayout {
            block: FusionStridedBlockLayout {
                shape: block.shape().to_vec(),
                strides: strides_to_isize(block.strides())?,
                offset: offset_to_isize(block.offset())?,
            },
        });
    }
    Ok(layouts)
}

fn coupled_sector<R>(rule: &R, tree: &FusionTreeKey) -> SectorId
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64>,
{
    tree.coupled().unwrap_or_else(|| rule.vacuum())
}

fn pack_group<A, T>(
    kernels: &mut A,
    group: &FusionBlockMatrixGroup,
    data: &[T],
    packed: &mut [T],
) -> Result<(), OperationError>
where
    A: HostKernelAdapter<T>,
    T: Copy + RecouplingCoefficientAction<f64>,
{
    for layout in &group.subblocks {
        kernels.copy_scale_strided(
            packed,
            data,
            &layout.block.shape,
            &layout.matrix_strides,
            &layout.block.strides,
            layout.matrix_offset,
            layout.block.offset,
            false,
            T::coefficient_as_data(layout.coefficient),
        )?;
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn execute_group_with_scratch_buffers<A, B, D, LhsScratch, RhsScratch, DestinationScratch>(
    kernels: &mut A,
    backend: &mut B,
    workspace: &mut B::Workspace,
    group: &CanonicalFusionBlockContractGroupPlan,
    scratch: &mut FusionBlockContractScratchBuffers<LhsScratch, RhsScratch, DestinationScratch>,
    dst_data: &mut [D],
    lhs_data: &[D],
    rhs_data: &[D],
    alpha: D,
    beta: D,
) -> Result<(), OperationError>
where
    A: HostKernelAdapter<D>,
    B: TensorContractBackend<D, f64>,
    D: DenseBlockScalar + RecouplingCoefficientAction<f64>,
    LhsScratch: HostWritableStorage<D>,
    RhsScratch: HostWritableStorage<D>,
    DestinationScratch: HostWritableStorage<D>,
{
    if group.lhs.direct_offset.is_none() {
        pack_group(
            kernels,
            &group.lhs,
            lhs_data,
            scratch.lhs_mut().as_mut_slice(),
        )?;
    }
    if group.rhs.direct_offset.is_none() {
        pack_group(
            kernels,
            &group.rhs,
            rhs_data,
            scratch.rhs_mut().as_mut_slice(),
        )?;
    }
    let dst_direct = if alpha.is_one() && beta.is_zero() {
        group.dst.direct_offset
    } else {
        None
    };
    let (lhs_scratch, rhs_scratch, dst_scratch) = scratch.inputs_and_destination_mut();
    let lhs_slice = direct_or_scratch_slice(
        lhs_data,
        group.lhs.direct_offset,
        group.lhs.rows,
        group.lhs.cols,
        lhs_scratch.as_slice(),
    )?;
    let rhs_slice = direct_or_scratch_slice(
        rhs_data,
        group.rhs.direct_offset,
        group.rhs.rows,
        group.rhs.cols,
        rhs_scratch.as_slice(),
    )?;
    match dst_direct {
        Some(base) => {
            let dst_slice = direct_slice_mut(dst_data, base, group.dst.rows, group.dst.cols)?;
            matmul_group_plan(backend, workspace, group, lhs_slice, rhs_slice, dst_slice)
        }
        None => {
            matmul_group_plan(
                backend,
                workspace,
                group,
                lhs_slice,
                rhs_slice,
                dst_scratch.as_mut_slice(),
            )?;
            scatter_group(
                kernels,
                &group.dst,
                dst_data,
                dst_scratch.as_slice(),
                alpha,
                beta,
            )
        }
    }
}

fn direct_matrix_len(rows: usize, cols: usize) -> Result<usize, OperationError> {
    rows.checked_mul(cols)
        .ok_or(OperationError::ElementCountOverflow)
}

fn direct_or_scratch_slice<'a, T>(
    data: &'a [T],
    direct_offset: Option<usize>,
    rows: usize,
    cols: usize,
    scratch: &'a [T],
) -> Result<&'a [T], OperationError> {
    match direct_offset {
        Some(base) => {
            let len = direct_matrix_len(rows, cols)?;
            let end = base
                .checked_add(len)
                .ok_or(OperationError::ElementCountOverflow)?;
            data.get(base..end)
                .ok_or(OperationError::ElementCountMismatch {
                    expected: end,
                    actual: data.len(),
                })
        }
        None => Ok(scratch),
    }
}

fn direct_slice_mut<T>(
    data: &mut [T],
    base: usize,
    rows: usize,
    cols: usize,
) -> Result<&mut [T], OperationError> {
    let len = direct_matrix_len(rows, cols)?;
    let end = base
        .checked_add(len)
        .ok_or(OperationError::ElementCountOverflow)?;
    let actual = data.len();
    data.get_mut(base..end)
        .ok_or(OperationError::ElementCountMismatch {
            expected: end,
            actual,
        })
}

fn scatter_group<A, T>(
    kernels: &mut A,
    group: &FusionBlockMatrixGroup,
    data: &mut [T],
    packed: &[T],
    alpha: T,
    beta: T,
) -> Result<(), OperationError>
where
    A: HostKernelAdapter<T>,
    T: Copy,
{
    for layout in &group.subblocks {
        kernels.axpby_strided(
            data,
            packed,
            &layout.block.shape,
            &layout.block.strides,
            &layout.matrix_strides,
            layout.block.offset,
            layout.matrix_offset,
            alpha,
            beta,
        )?;
    }
    Ok(())
}

fn scale_all_blocks<A, T>(
    kernels: &mut A,
    blocks: &[FusionScaleBlockLayout],
    data: &mut [T],
    beta: T,
) -> Result<(), OperationError>
where
    A: HostKernelAdapter<T>,
    T: Copy + One + PartialEq,
{
    if beta.is_one() {
        return Ok(());
    }
    for layout in blocks {
        kernels.scale_strided(
            data,
            &layout.block.shape,
            &layout.block.strides,
            layout.block.offset,
            beta,
        )?;
    }
    Ok(())
}

fn matmul_group_plan<B, D>(
    backend: &mut B,
    workspace: &mut B::Workspace,
    group: &CanonicalFusionBlockContractGroupPlan,
    lhs: &[D],
    rhs: &[D],
    dst: &mut [D],
) -> Result<(), OperationError>
where
    B: TensorContractBackend<D, f64>,
    D: DenseBlockScalar + RecouplingCoefficientAction<f64>,
{
    backend.matmul_rank2_into_raw(
        workspace,
        dst,
        lhs,
        rhs,
        group.lhs.rows,
        group.lhs.cols,
        group.rhs.cols,
    )
}
