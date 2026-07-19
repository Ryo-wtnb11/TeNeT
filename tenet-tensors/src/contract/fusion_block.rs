use std::collections::HashSet;

use rustc_hash::FxHashMap;
use std::sync::Arc;

use tenet_core::{
    BlockKey, FusionRule, FusionTreeHomSpace, FusionTreeKey, HostReadableStorage,
    HostWritableStorage, MultiplicityFreeRigidSymbols, SectorId,
};

use crate::lowering::prelowered_storage_block_index;
use crate::strided::{
    column_major_strides_isize, column_major_strides_usize, element_count, offset_to_isize,
    strides_to_isize,
};
use crate::{DenseBlockScalar, HostKernelAdapter, OperationError, RecouplingCoefficientAction};
use tenet_operations::TensorContractSpec;

use tenet_operations::fusion_replay::{
    direct_group_matrix_offset, fusion_scale_block_layouts_excluding, FusionBlockContractGroupPlan,
    FusionBlockMatrixGroup, FusionStridedBlockLayout, FusionSubblockMatrixLayout, MatrixOp,
    Rank2GemmBatchJob,
};
pub(crate) use tenet_operations::fusion_replay::{
    FusionBlockContractPlan, FusionBlockContractWorkspace, Rank2Gemm, StorageGemm,
};

/// Validate category identity before a contraction route reads sectors or
/// symbols through the supplied rule.
///
/// Why-not validate during replay: a compiled plan is already tied to
/// validated spaces, so replay checks would charge every warm execution for a
/// compile-time invariant.
pub(crate) fn validate_fusion_contract_rule<R>(
    rule: &R,
    dst: &DynamicFusionMapSpace,
    lhs: &DynamicFusionMapSpace,
    rhs: &DynamicFusionMapSpace,
) -> Result<(), OperationError>
where
    R: FusionRule,
{
    dst.validate_rule(rule)?;
    lhs.validate_rule(rule)?;
    rhs.validate_rule(rule)
}

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

    fn matmul_rank2_batch(
        &mut self,
        dst: &mut [D],
        lhs: &[D],
        rhs: &[D],
        jobs: &[Rank2GemmBatchJob],
        runs: &[usize],
        alpha: D,
        beta: D,
    ) -> Result<(), OperationError>
    where
        D: Copy,
    {
        self.backend.matmul_rank2_batch_axpby_into_raw(
            self.workspace,
            dst,
            lhs,
            rhs,
            jobs,
            runs,
            alpha,
            beta,
        )
    }

    fn matmul_rank2_batch_with_ops(
        &mut self,
        dst: &mut [D],
        lhs: &[D],
        rhs: &[D],
        jobs: &[Rank2GemmBatchJob],
        runs: &[usize],
        lhs_op: MatrixOp,
        rhs_op: MatrixOp,
        alpha: D,
        beta: D,
    ) -> Result<(), OperationError>
    where
        D: Copy,
    {
        self.backend.matmul_rank2_batch_with_ops_axpby_into_raw(
            self.workspace,
            dst,
            lhs,
            rhs,
            jobs,
            runs,
            lhs_op,
            rhs_op,
            alpha,
            beta,
        )
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn tensorcontract_core_fusion_blocks_into_raw<A, B, R, D>(
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
    axes: TensorContractSpec<'_>,
    alpha: D,
    beta: D,
) -> Result<(), OperationError>
where
    A: HostKernelAdapter<D>,
    B: TensorContractBackend<D, f64>,
    R: MultiplicityFreeRigidSymbols<Scalar = f64>,
    D: DenseBlockScalar + RecouplingCoefficientAction<f64>,
{
    let plan = compile_fusion_block_contract_plan(rule, dst_space, lhs_space, rhs_space, axes)?;
    tensorcontract_core_fusion_blocks_with_plan_into_raw(
        kernels, backend, workspace, &plan, dst_space, dst_data, lhs_space, lhs_data, rhs_space,
        rhs_data, alpha, beta,
    )
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn tensorcontract_core_fusion_blocks_with_plan_into_raw<A, B, D>(
    kernels: &mut A,
    backend: &mut B,
    workspace: &mut B::Workspace,
    plan: &FusionBlockContractPlan,
    dst_space: &DynamicFusionMapSpace,
    dst_data: &mut [D],
    lhs_space: &DynamicFusionMapSpace,
    lhs_data: &[D],
    rhs_space: &DynamicFusionMapSpace,
    rhs_data: &[D],
    alpha: D,
    beta: D,
) -> Result<(), OperationError>
where
    A: HostKernelAdapter<D>,
    B: TensorContractBackend<D, f64>,
    D: DenseBlockScalar + RecouplingCoefficientAction<f64>,
{
    let mut fusion_workspace = FusionBlockContractWorkspace::<D>::default();
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

pub(crate) fn is_core_form_fusion_block_contract<R>(
    rule: &R,
    dst_space: &DynamicFusionMapSpace,
    lhs_space: &DynamicFusionMapSpace,
    rhs_space: &DynamicFusionMapSpace,
    axes: TensorContractSpec<'_>,
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
    if !is_core_form_source(lhs_space, rhs_space, &axis_plan)
        || !is_core_form_output(dst_space, lhs_space, rhs_space, &axis_plan)
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

fn validate_core_compose<R>(
    rule: &R,
    dst_space: &DynamicFusionMapSpace,
    lhs_space: &DynamicFusionMapSpace,
    rhs_space: &DynamicFusionMapSpace,
    axes: TensorContractSpec<'_>,
) -> Result<(), OperationError>
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64>,
{
    if is_core_form_fusion_block_contract(rule, dst_space, lhs_space, rhs_space, axes)? {
        Ok(())
    } else {
        Err(OperationError::UnsupportedTensorContractScope {
            message: "core fusion-block contraction requires core source and output axes",
        })
    }
}

/// Generic-fusion (Stage B3c-1) sibling of [`is_core_form_fusion_block_contract`]:
/// identical predicate, relaxed to any [`FusionRule`]. The homspace-shape check
/// (`tensorcontract_homspace`) and the axis-form checks are already fully
/// symmetry-agnostic — only the mult-free trait bound differed.
fn is_core_form_fusion_block_contract_generic<R>(
    rule: &R,
    dst_space: &DynamicFusionMapSpace,
    lhs_space: &DynamicFusionMapSpace,
    rhs_space: &DynamicFusionMapSpace,
    axes: TensorContractSpec<'_>,
) -> Result<bool, OperationError>
where
    R: FusionRule,
{
    reject_fusion_contract_conjugation(axes)?;
    let axis_plan = TensorContractAxisPlan::compile(
        lhs_space.rank(),
        rhs_space.rank(),
        dst_space.rank(),
        axes,
    )?;
    if !is_core_form_source(lhs_space, rhs_space, &axis_plan)
        || !is_core_form_output(dst_space, lhs_space, rhs_space, &axis_plan)
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

fn is_core_form_source(
    lhs_space: &DynamicFusionMapSpace,
    rhs_space: &DynamicFusionMapSpace,
    axis_plan: &TensorContractAxisPlan,
) -> bool {
    let lhs_domain_axes = (lhs_space.nout()..lhs_space.rank()).collect::<Vec<_>>();
    let rhs_codomain_axes = (0..rhs_space.nout()).collect::<Vec<_>>();
    axis_plan.lhs_contracting_axes == lhs_domain_axes
        && axis_plan.rhs_contracting_axes == rhs_codomain_axes
}

fn is_core_form_output(
    dst_space: &DynamicFusionMapSpace,
    lhs_space: &DynamicFusionMapSpace,
    rhs_space: &DynamicFusionMapSpace,
    axis_plan: &TensorContractAxisPlan,
) -> bool {
    let output_rank = lhs_space.nout() + (rhs_space.rank() - rhs_space.nout());
    let core_output_axes = (0..output_rank).collect::<Vec<_>>();
    dst_space.nout() == lhs_space.nout() && axis_plan.output_axes == core_output_axes
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::RefCell;
    use std::rc::Rc;
    use tenet_core::{
        FusionProductSpace, FusionTensorMapSpace, HostReadableStorage, HostWritableStorage,
        SU2FusionRule, SectorLeg, TensorMap, TensorMapSpace, TensorStorage, Trivial, Z2FusionRule,
    };
    use tenet_core::{Placement, SimilarStorage};
    use tenet_operations::fusion_replay::HostFusionBlockContractWorkspace;
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
    fn incomplete_su2_grid_is_nonborrowed_and_keeps_sparse_group_clear() {
        // What: a legal SU2 structure missing off-diagonal tree pairs is
        // charged as RHS materialization and its packed matrix remains marked
        // for clearing before replay.
        let rule = SU2FusionRule;
        let scalar = FusionTensorMapSpace::from_degeneracy_shapes(
            TensorMapSpace::<0, 0>::from_dims([], []).unwrap(),
            FusionTreeHomSpace::from_sector_ids([], []),
            &rule,
            [vec![]],
        )
        .unwrap();
        let homspace = FusionTreeHomSpace::from_sector_ids(
            [(1, 1), (1, 1), (1, 1), (1, 1)],
            [(1, 1), (1, 1), (1, 1), (1, 1)],
        );
        let keys = homspace.fusion_tree_keys(&rule);
        let diagonal_keys = keys
            .iter()
            .filter(|key| key.codomain_tree() == key.domain_tree())
            .cloned()
            .collect::<Vec<_>>();
        assert_eq!(keys.len(), 14);
        assert_eq!(diagonal_keys.len(), 6);
        let sparse = FusionTensorMapSpace::new_unbound(
            TensorMapSpace::<4, 4>::from_dims([1, 1, 1, 1], [1, 1, 1, 1]).unwrap(),
            homspace.clone(),
            crate::tests::packed_fixture_structure(
                8,
                diagonal_keys
                    .into_iter()
                    .map(|key| (BlockKey::from(key), vec![1; 8])),
            )
            .unwrap(),
        )
        .unwrap()
        .try_bind_rule(&rule)
        .unwrap();
        let complete = FusionTensorMapSpace::from_degeneracy_shapes(
            TensorMapSpace::<4, 4>::from_dims([1, 1, 1, 1], [1, 1, 1, 1]).unwrap(),
            homspace,
            &rule,
            vec![vec![1; 8]; keys.len()],
        )
        .unwrap();
        let scalar = DynamicFusionMapSpace::from_typed(&scalar);
        let sparse = DynamicFusionMapSpace::from_typed(&sparse);
        let complete = DynamicFusionMapSpace::from_typed(&complete);

        let facts = crate::contract::prepare_tensorcontract_fusion_candidate_facts_dyn_raw(
            &rule,
            &complete,
            &scalar,
            &sparse,
            TensorContractSpec::new(&[], &[], tenet_operations::OutputAxisOrder::identity()),
        )
        .unwrap();
        assert_eq!(facts.len(), 1);
        assert_eq!(facts[0].lhs_materialized_elements(), 0);
        assert!(!facts[0].rhs_exact_identity_borrowable());
        assert_eq!(facts[0].rhs_materialized_elements(), 14);
        assert_eq!(facts[0].output_materialized_elements(), 14);

        let layout = FusionBlockMatrixLayout::compile(&rule, &sparse).unwrap();
        assert_eq!(layout.groups.len(), 3);
        assert_eq!(
            layout
                .groups
                .iter()
                .filter(|group| group.needs_clear)
                .count(),
            2
        );
        for group in &layout.groups {
            assert_eq!(
                group.needs_clear,
                group.subblocks.len() != group.rows * group.cols
            );
        }
    }

    #[test]
    fn storage_direct_replay_runs_without_host_slice_contract() {
        let rule = Z2FusionRule;
        let leg = || SectorLeg::new([(SectorId::new(0), 1), (SectorId::new(1), 1)], false);
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
        let plan = compile_fusion_block_contract_plan(
            &rule,
            &DynamicFusionMapSpace::from_typed(&space),
            &DynamicFusionMapSpace::from_typed(&space),
            &DynamicFusionMapSpace::from_typed(&space),
            TensorContractSpec::with_default_output_order(&[1], &[0]),
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
        let leg = |dims: usize| {
            SectorLeg::new([(SectorId::new(0), dims), (SectorId::new(1), dims)], false)
        };
        let fusion_space = |dims: usize| {
            FusionTensorMapSpace::from_degeneracy_shapes_coupled(
                TensorMapSpace::<1, 1>::from_dims([2 * dims], [2 * dims]).unwrap(),
                FusionTreeHomSpace::new(
                    FusionProductSpace::new([leg(dims)]),
                    FusionProductSpace::new([leg(dims)]),
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
        let plan = compile_fusion_block_contract_plan(
            &rule,
            &DynamicFusionMapSpace::from_typed(&space),
            &DynamicFusionMapSpace::from_typed(&space),
            &DynamicFusionMapSpace::from_typed(&space),
            TensorContractSpec::with_default_output_order(&[1], &[0]),
        )
        .unwrap();

        let mut backend = DenseTreeTransformOperations::default();
        let mut workspace = TensorContractWorkspace::default();
        let mut expected = vec![0.0; len];
        let structure = std::sync::Arc::clone(space.subblock_structure());
        let mut fusion_workspace = FusionBlockContractWorkspace::<f64>::default();
        plan.execute_raw(
            &mut crate::StridedHostKernelAdapter::default(),
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

    fn z2_adjoint_mapping_spaces() -> (
        DynamicFusionMapSpace,
        DynamicFusionMapSpace,
        FusionBlockMatrixLayout,
        FusionBlockMatrixLayout,
    ) {
        let rule = Z2FusionRule;
        let leg = || SectorLeg::new([(SectorId::new(0), 2), (SectorId::new(1), 2)], false);
        let storage = FusionTensorMapSpace::from_degeneracy_shapes(
            TensorMapSpace::<1, 1>::from_dims([4], [4]).unwrap(),
            FusionTreeHomSpace::new(
                FusionProductSpace::new([leg()]),
                FusionProductSpace::new([leg()]),
            ),
            &rule,
            [vec![2, 2], vec![2, 2]],
        )
        .unwrap();
        let logical = crate::lowering::adjoint_fusion_space_view(&storage).unwrap();
        let logical = DynamicFusionMapSpace::from_typed(&logical);
        let storage = DynamicFusionMapSpace::from_typed(&storage);
        let logical_layout = FusionBlockMatrixLayout::compile(&rule, &logical).unwrap();
        let storage_layout = FusionBlockMatrixLayout::compile(&rule, &storage).unwrap();
        (logical, storage, logical_layout, storage_layout)
    }

    #[test]
    fn prelowered_mapping_rejects_mismatched_tree_ordering() {
        let (logical, storage, mut logical_layout, storage_layout) = z2_adjoint_mapping_spaces();
        let logical_group = logical_layout.groups.first_mut().unwrap();
        logical_group.subblocks[0].matrix_offset += 1;

        let error = map_logical_group_to_storage(
            logical_group,
            &logical,
            &storage,
            &storage_layout,
            MatrixOp::Adjoint,
        )
        .unwrap_err();
        assert!(matches!(
            error,
            OperationError::StructureMismatch {
                tensor: "prelowered tree ordering"
            }
        ));
    }

    #[test]
    fn prelowered_non_direct_physical_layout_yields_fallback_plan() {
        let (logical, storage, logical_layout, mut storage_layout) = z2_adjoint_mapping_spaces();
        let logical_group = logical_layout.groups[0].clone();
        storage_layout.groups[0].direct_offset = None;
        let physical = map_logical_group_to_storage(
            &logical_group,
            &logical,
            &storage,
            &storage_layout,
            MatrixOp::Adjoint,
        )
        .unwrap();
        assert_eq!(physical.direct_offset, None);

        let group =
            FusionBlockContractGroupPlan::new(physical.clone(), physical, logical_group).unwrap();
        let plan = FusionBlockContractPlan::from_parts_with_ops(
            Arc::clone(logical.structure()),
            Arc::clone(storage.structure()),
            Arc::clone(storage.structure()),
            Vec::new(),
            vec![group],
            MatrixOp::Adjoint,
            MatrixOp::Adjoint,
        )
        .unwrap();

        // What: a valid but non-direct physical layout is a route decision,
        // not a user-visible Core compilation error.
        assert!(!plan.is_fully_direct());
    }

    /// GPU vertical: the same core direct replay executed on CUDA
    /// storage must reproduce the host result bit-for-bit (same GEMM
    /// ordering, overwrite semantics). Requires a CUDA device; run with
    /// `cargo test --features cuda -- --ignored`.
    #[cfg(feature = "cuda")]
    #[test]
    #[ignore]
    fn storage_direct_replay_on_cuda_matches_host() {
        use tenet_dense::CudaDenseContext;
        use tenet_operations::cuda::{CudaStorage, CudaStorageGemm};

        let rule = Z2FusionRule;
        let leg = || SectorLeg::new([(SectorId::new(0), 3), (SectorId::new(1), 3)], false);
        let space = FusionTensorMapSpace::from_degeneracy_shapes_coupled(
            TensorMapSpace::<1, 1>::from_dims([6], [6]).unwrap(),
            FusionTreeHomSpace::new(
                FusionProductSpace::new([leg()]),
                FusionProductSpace::new([leg()]),
            ),
            &rule,
            [vec![3, 3], vec![3, 3]],
        )
        .unwrap();
        let len = space.required_len().unwrap();
        let lhs_data: Vec<f64> = (0..len).map(|i| 0.5 * i as f64 - 1.0).collect();
        let rhs_data: Vec<f64> = (0..len).map(|i| 1.5 - 0.25 * i as f64).collect();
        let plan = compile_fusion_block_contract_plan(
            &rule,
            &DynamicFusionMapSpace::from_typed(&space),
            &DynamicFusionMapSpace::from_typed(&space),
            &DynamicFusionMapSpace::from_typed(&space),
            TensorContractSpec::with_default_output_order(&[1], &[0]),
        )
        .unwrap();

        let mut backend = DenseTreeTransformOperations::default();
        let mut workspace = TensorContractWorkspace::default();
        let mut expected = vec![0.0; len];
        let mut gemm = HostStorageGemm::new(&mut backend, &mut workspace);
        plan.execute_direct_on_storage(&mut gemm, &mut expected, &lhs_data, &rhs_data)
            .unwrap();

        let mut ctx = CudaDenseContext::new(0).unwrap();
        let lhs_dev = CudaStorage::upload(&ctx, &lhs_data).unwrap();
        let rhs_dev = CudaStorage::upload(&ctx, &rhs_data).unwrap();
        let mut dst_dev = CudaStorage::upload(&ctx, &vec![0.0; len]).unwrap();
        plan.execute_direct_on_storage(
            &mut CudaStorageGemm::new(&mut ctx),
            &mut dst_dev,
            &lhs_dev,
            &rhs_dev,
        )
        .unwrap();
        let result = dst_dev.download(&ctx).unwrap();

        assert_eq!(result, expected);
    }

    #[test]
    fn core_fusion_block_workspace_is_explicit_host_workspace() {
        let workspace = HostFusionBlockContractWorkspace::<f64>::default();
        let alias = FusionBlockContractWorkspace::<f64>::default();

        assert_eq!(workspace.placement(), Placement::Host);
        assert!(workspace.is_host_placement());
        assert_eq!(alias.placement(), Placement::Host);
    }

    #[test]
    fn core_fusion_block_storage_workspace_allocates_pack_scratch_from_operands_and_output_from_destination(
    ) {
        let rule = Z2FusionRule;
        let leg = || SectorLeg::new([(SectorId::new(0), 1), (SectorId::new(1), 1)], false);
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
        let plan = compile_fusion_block_contract_plan(
            &rule,
            &DynamicFusionMapSpace::from_typed(dst.fusion_space().unwrap()),
            &DynamicFusionMapSpace::from_typed(lhs.fusion_space().unwrap()),
            &DynamicFusionMapSpace::from_typed(rhs.fusion_space().unwrap()),
            TensorContractSpec::with_default_output_order(&[1], &[0]),
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
            &mut crate::StridedHostKernelAdapter::default(),
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
        // Pack/scatter scratch is gone from the core route: replay is
        // direct GEMM on storage, so no workspace allocations occur.
        assert_eq!(allocations.borrow().as_slice(), &[]);
    }
}

pub(crate) fn compile_fusion_block_contract_plan<R>(
    rule: &R,
    dst_space: &DynamicFusionMapSpace,
    lhs_space: &DynamicFusionMapSpace,
    rhs_space: &DynamicFusionMapSpace,
    axes: TensorContractSpec<'_>,
) -> Result<FusionBlockContractPlan, OperationError>
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64>,
{
    validate_fusion_contract_rule(rule, dst_space, lhs_space, rhs_space)?;
    compile_fusion_block_contract_plan_validated(rule, dst_space, lhs_space, rhs_space, axes)
}

pub(crate) fn compile_fusion_block_contract_plan_validated<R>(
    rule: &R,
    dst_space: &DynamicFusionMapSpace,
    lhs_space: &DynamicFusionMapSpace,
    rhs_space: &DynamicFusionMapSpace,
    axes: TensorContractSpec<'_>,
) -> Result<FusionBlockContractPlan, OperationError>
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64>,
{
    reject_fusion_contract_conjugation(axes)?;
    // Axis validation happens inside validate_core_compose.
    validate_core_compose(rule, dst_space, lhs_space, rhs_space, axes)?;

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
                "core fusion-block dst subblock must be scattered exactly once"
            );
        }
        active_dst_blocks.extend(dst_group.block_indices.iter().copied());
        groups.push(FusionBlockContractGroupPlan::new(
            lhs_group,
            rhs_group.clone(),
            dst_group.clone(),
        )?);
    }
    FusionBlockContractPlan::from_parts(
        Arc::clone(dst_space.structure()),
        Arc::clone(lhs_space.structure()),
        Arc::clone(rhs_space.structure()),
        fusion_scale_block_layouts_excluding(dst_space.structure(), &active_dst_blocks)?,
        groups,
    )
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn compile_fusion_block_contract_plan_prelowered<R>(
    rule: &R,
    dst_space: &DynamicFusionMapSpace,
    lhs_logical: &DynamicFusionMapSpace,
    lhs_storage: &DynamicFusionMapSpace,
    rhs_logical: &DynamicFusionMapSpace,
    rhs_storage: &DynamicFusionMapSpace,
    axes: TensorContractSpec<'_>,
    lhs_op: MatrixOp,
    rhs_op: MatrixOp,
) -> Result<FusionBlockContractPlan, OperationError>
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64>,
{
    validate_fusion_contract_rule(rule, dst_space, lhs_logical, rhs_logical)?;
    lhs_storage.validate_rule(rule)?;
    rhs_storage.validate_rule(rule)?;
    validate_core_compose(rule, dst_space, lhs_logical, rhs_logical, axes)?;

    let lhs_logical_layout = FusionBlockMatrixLayout::compile(rule, lhs_logical)?;
    let lhs_storage_layout = FusionBlockMatrixLayout::compile(rule, lhs_storage)?;
    let rhs_logical_layout = FusionBlockMatrixLayout::compile(rule, rhs_logical)?;
    let rhs_storage_layout = FusionBlockMatrixLayout::compile(rule, rhs_storage)?;
    let dst_layout = FusionBlockMatrixLayout::compile(rule, dst_space)?;

    let mut groups = Vec::new();
    let mut active_dst_blocks = HashSet::<usize>::new();
    for lhs_group in lhs_logical_layout.groups {
        let Some(rhs_group) = rhs_logical_layout.group(lhs_group.coupled) else {
            continue;
        };
        let Some(dst_group) = dst_layout.group(lhs_group.coupled) else {
            continue;
        };
        let lhs_physical = map_logical_group_to_storage(
            &lhs_group,
            lhs_logical,
            lhs_storage,
            &lhs_storage_layout,
            lhs_op,
        )?;
        let rhs_physical = map_logical_group_to_storage(
            rhs_group,
            rhs_logical,
            rhs_storage,
            &rhs_storage_layout,
            rhs_op,
        )?;
        for block_index in &dst_group.block_indices {
            debug_assert!(
                !active_dst_blocks.contains(block_index),
                "core fusion-block dst subblock must be scattered exactly once"
            );
        }
        active_dst_blocks.extend(dst_group.block_indices.iter().copied());
        groups.push(FusionBlockContractGroupPlan::new(
            lhs_physical,
            rhs_physical,
            dst_group.clone(),
        )?);
    }
    FusionBlockContractPlan::from_parts_with_ops(
        Arc::clone(dst_space.structure()),
        Arc::clone(lhs_storage.structure()),
        Arc::clone(rhs_storage.structure()),
        fusion_scale_block_layouts_excluding(dst_space.structure(), &active_dst_blocks)?,
        groups,
        lhs_op,
        rhs_op,
    )
}

fn map_logical_group_to_storage(
    logical_group: &FusionBlockMatrixGroup,
    logical_space: &DynamicFusionMapSpace,
    storage_space: &DynamicFusionMapSpace,
    storage_layout: &FusionBlockMatrixLayout,
    op: MatrixOp,
) -> Result<FusionBlockMatrixGroup, OperationError> {
    let storage_conjugate = op != MatrixOp::Identity;
    let map_block = prelowered_storage_block_index(logical_space, storage_space, storage_conjugate);
    let mapped = logical_group
        .block_indices
        .iter()
        .map(|&index| map_block(index))
        .collect::<Result<HashSet<_>, _>>()?;
    let storage_group = storage_layout
        .groups
        .iter()
        .find(|group| {
            group.block_indices.len() == mapped.len()
                && group
                    .block_indices
                    .iter()
                    .all(|index| mapped.contains(index))
        })
        .ok_or(OperationError::StructureMismatch {
            tensor: "prelowered physical group",
        })?;

    let expected_dims = match op {
        MatrixOp::Identity => (storage_group.rows, storage_group.cols),
        MatrixOp::Transpose | MatrixOp::Adjoint => (storage_group.cols, storage_group.rows),
    };
    if (logical_group.rows, logical_group.cols) != expected_dims {
        return Err(OperationError::ShapeMismatch {
            dst: vec![logical_group.rows, logical_group.cols],
            src: vec![expected_dims.0, expected_dims.1],
        });
    }
    for &logical_index in &logical_group.block_indices {
        let storage_index = map_block(logical_index)?;
        let logical = logical_space.structure().block(logical_index)?;
        let storage = storage_space.structure().block(storage_index)?;
        let split = storage_space.nout();
        let expected_shape = match op {
            MatrixOp::Identity => storage.shape().to_vec(),
            MatrixOp::Transpose | MatrixOp::Adjoint => storage.shape()[split..]
                .iter()
                .chain(&storage.shape()[..split])
                .copied()
                .collect(),
        };
        // Why not compare logical offsets/strides: the categorical adjoint
        // space has its own canonical packed layout. Replay never addresses
        // that layout; the checked parent group below is the physical authority.
        if logical.shape() != expected_shape {
            return Err(OperationError::StructureMismatch {
                tensor: "prelowered block layout",
            });
        }
        let logical_position = logical_group
            .block_indices
            .iter()
            .position(|&index| index == logical_index)
            .expect("logical block belongs to its group");
        let storage_position = storage_group
            .block_indices
            .iter()
            .position(|&index| index == storage_index)
            .expect("mapped storage block belongs to its group");
        let logical_offset = usize::try_from(
            logical_group.subblocks[logical_position].matrix_offset,
        )
        .map_err(|_| OperationError::StructureMismatch {
            tensor: "prelowered logical tree offset",
        })?;
        let storage_offset = usize::try_from(
            storage_group.subblocks[storage_position].matrix_offset,
        )
        .map_err(|_| OperationError::StructureMismatch {
            tensor: "prelowered physical tree offset",
        })?;
        let logical_tree_offset = (
            logical_offset % logical_group.rows,
            logical_offset / logical_group.rows,
        );
        let storage_tree_offset = (
            storage_offset % storage_group.rows,
            storage_offset / storage_group.rows,
        );
        let expected_tree_offset = match op {
            MatrixOp::Identity => storage_tree_offset,
            MatrixOp::Transpose | MatrixOp::Adjoint => {
                (storage_tree_offset.1, storage_tree_offset.0)
            }
        };
        if logical_tree_offset != expected_tree_offset {
            return Err(OperationError::StructureMismatch {
                tensor: "prelowered tree ordering",
            });
        }
    }
    let mut physical = logical_group.clone();
    physical.direct_offset = storage_group.direct_offset;
    physical.block_indices = storage_group.block_indices.clone();
    physical.subblocks = storage_group.subblocks.clone();
    Ok(physical)
}

/// Generic-fusion (Stage B3c-1) sibling of [`compile_fusion_block_contract_plan`]:
/// the SU(N) core/compose (fully-direct GEMM) route. Byte-identical plan
/// structure to the mult-free path — the coupled-block GEMM is symmetry-
/// agnostic, so the ONLY difference is that outer-multiplicity fusion trees
/// (vertex-labelled blocks, e.g. SU(3) `N(8,8,8)=2`) are grouped/paired
/// correctly by the group-agnostic block structure. The per-subblock
/// coefficient is `1.0` (SU(N) is bosonic → no supertrace twist, exactly what
/// the mult-free `rule.scalar_one()` returns for a bosonic rule).
///
/// A contraction whose source or output is NOT in core form (open contracted
/// legs needing a source tree-pair transform) is an explicit B3c-2 error: the
/// generic source-transform contract path is Stage B3c-2, not this stage.
pub(crate) fn compile_fusion_block_contract_plan_generic<R>(
    rule: &R,
    dst_space: &DynamicFusionMapSpace,
    lhs_space: &DynamicFusionMapSpace,
    rhs_space: &DynamicFusionMapSpace,
    axes: TensorContractSpec<'_>,
) -> Result<FusionBlockContractPlan, OperationError>
where
    R: FusionRule,
{
    validate_fusion_contract_rule(rule, dst_space, lhs_space, rhs_space)?;
    // Hardening guard (adversarial review, Stage B3c-1 refute pass): the
    // per-subblock `coefficient` computed below in `finish_generic` is
    // hardcoded to `1.0`, which assumes a bosonic rule (no supertrace twist).
    // That assumption is correct for every Generic rule shipped today
    // (SU(N) is bosonic), but silently drops a twist for a hypothetical
    // future non-bosonic Generic rule instead of failing loudly. Guard it
    // here so a non-bosonic rule gets the same explicit B3c-2 scope error as
    // any other unsupported generic-contract shape, rather than a silently
    // wrong coefficient.
    if rule.braiding_style() != tenet_core::BraidingStyleKind::Bosonic {
        return Err(OperationError::UnsupportedTensorContractScope {
            message: "non-bosonic Generic fusion contraction requires twist handling; \
                      the core/compose (fully-direct GEMM) route assumes bosonic braiding \
                      (coefficient = 1.0), which is Stage B3c-2, not this stage",
        });
    }
    reject_fusion_contract_conjugation(axes)?;
    if !is_core_form_fusion_block_contract_generic(rule, dst_space, lhs_space, rhs_space, axes)? {
        return Err(OperationError::UnsupportedTensorContractScope {
            message: "generic (SU(N)) fusion contraction supports only the core/compose \
                      (fully-direct GEMM) route; source tree-pair transforms are Stage B3c-2",
        });
    }

    let lhs_layout = FusionBlockMatrixLayout::compile_generic(lhs_space)?;
    let rhs_layout = FusionBlockMatrixLayout::compile_generic(rhs_space)?;
    let dst_layout = FusionBlockMatrixLayout::compile_generic(dst_space)?;

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
                "core fusion-block dst subblock must be scattered exactly once"
            );
        }
        active_dst_blocks.extend(dst_group.block_indices.iter().copied());
        groups.push(FusionBlockContractGroupPlan::new(
            lhs_group,
            rhs_group.clone(),
            dst_group.clone(),
        )?);
    }
    FusionBlockContractPlan::from_parts(
        Arc::clone(dst_space.structure()),
        Arc::clone(lhs_space.structure()),
        Arc::clone(rhs_space.structure()),
        fusion_scale_block_layouts_excluding(dst_space.structure(), &active_dst_blocks)?,
        groups,
    )
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
        let mut group_indices = FxHashMap::<SectorId, usize>::default();
        for block_index in 0..space.structure().block_count() {
            let block = space.structure().block(block_index)?;
            let BlockKey::FusionTree(key) = block.key() else {
                return Err(OperationError::ExpectedFusionTreeBlock {
                    tensor: "fusion",
                    index: block_index,
                });
            };
            let coupled = coupled_sector(key.codomain_tree());
            if coupled != coupled_sector(key.domain_tree()) {
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

    /// Generic-fusion (Stage B3c-1) sibling of [`Self::compile`]: relaxed to any
    /// [`FusionRule`] (the layout only needs `coupled()`/`vacuum()` to group
    /// blocks by coupled sector — no F/R symbols). Outer-multiplicity vertex
    /// labels ride in the fusion-tree keys, so multiplicity blocks land in the
    /// right coupled group automatically.
    fn compile_generic(space: &DynamicFusionMapSpace) -> Result<Self, OperationError> {
        let mut builders = Vec::<FusionBlockMatrixGroupBuilder>::new();
        let mut group_indices = FxHashMap::<SectorId, usize>::default();
        for block_index in 0..space.structure().block_count() {
            let block = space.structure().block(block_index)?;
            let BlockKey::FusionTree(key) = block.key() else {
                return Err(OperationError::ExpectedFusionTreeBlock {
                    tensor: "fusion",
                    index: block_index,
                });
            };
            let coupled = coupled_sector_generic(key.codomain_tree());
            if coupled != coupled_sector_generic(key.domain_tree()) {
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
            groups.push(builder.finish_generic(space)?);
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
    row_offsets: FxHashMap<FusionTreeKey, TreeMatrixOffset>,
    col_offsets: FxHashMap<FusionTreeKey, TreeMatrixOffset>,
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
            row_offsets: FxHashMap::default(),
            col_offsets: FxHashMap::default(),
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

    /// Generic-fusion (Stage B3c-1) sibling of [`Self::finish`]: byte-for-byte
    /// the same block/matrix layout, with the coefficient fixed to `1.0`.
    /// SU(N) is bosonic, so there is no supertrace twist — exactly the value
    /// `rule.scalar_one()` returns on the mult-free path. Takes no rule (the
    /// layout math is symmetry-agnostic once blocks are grouped by coupled
    /// sector).
    fn finish_generic(
        self,
        space: &DynamicFusionMapSpace,
    ) -> Result<FusionBlockMatrixGroup, OperationError> {
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
            // SU(N) bosonic: no supertrace twist → coefficient 1.0 (TensorKit
            // mul! parity, matching the mult-free `rule.scalar_one()`). This
            // assumes a bosonic rule; `compile_fusion_block_contract_plan_generic`
            // guards the entry against non-bosonic rules so that assumption is
            // never silently violated here.
            let coefficient = 1.0_f64;
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

fn coupled_sector(tree: &FusionTreeKey) -> SectorId {
    tree.coupled()
}

/// Generic-fusion sibling retained as a named algorithm boundary.
fn coupled_sector_generic(tree: &FusionTreeKey) -> SectorId {
    tree.coupled()
}

#[cfg(test)]
mod rule_identity_tests {
    use super::*;
    use tenet_core::{CoreError, FusionTreeHomSpace, TabulatedFusionRule};
    use tenet_operations::OutputAxisOrder;

    #[test]
    fn generic_block_plan_rejects_spaces_from_different_tabulated_rules() {
        const SU3: &[u8] = include_bytes!("../../../tenet-core/src/su3_table.bin");
        const SU4: &[u8] = include_bytes!("../../../tenet-core/src/testdata/su4_table.bin");
        let su3 = TabulatedFusionRule::try_from_bytes(SU3, "su3-table.bin").unwrap();
        let su4 = TabulatedFusionRule::try_from_bytes(SU4, "su4-table.bin").unwrap();
        let make = |rule: &TabulatedFusionRule| {
            DynamicFusionMapSpace::from_degeneracy_shapes_generic(
                rule,
                FusionTreeHomSpace::from_sector_ids([], []),
                [vec![]],
            )
            .unwrap()
        };
        let dst = make(&su3);
        let lhs = make(&su3);
        let rhs = make(&su4);

        let error = compile_fusion_block_contract_plan_generic(
            &su3,
            &dst,
            &lhs,
            &rhs,
            TensorContractSpec::new(&[], &[], OutputAxisOrder::identity()),
        )
        .unwrap_err();
        assert!(matches!(
            error,
            OperationError::Core(CoreError::FusionRuleMismatch { .. })
        ));
    }
}
