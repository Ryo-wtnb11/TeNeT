use std::collections::HashSet;

use rustc_hash::FxHashMap;
use std::sync::Arc;

use tenet_core::{
    BlockKey, FusionRule, FusionTreeHomSpace, FusionTreeKey, FusionTreePairOrientation,
    HostReadableStorage, HostWritableStorage, MultiplicityFreeRigidSymbols,
    OrientedFusionTreeHomSpace, SectorId,
};

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
use super::dynamic_space::{DynamicFusionMapSpace, FusionOperandLayout};
use super::fusion::reject_fusion_contract_conjugation;
use super::structure::TensorContractAxisPlan;

pub(super) struct CoreContractPreflight<'a, R> {
    rule: &'a R,
    dst_homspace: &'a FusionTreeHomSpace,
    lhs_homspace: OrientedFusionTreeHomSpace<'a>,
    rhs_homspace: OrientedFusionTreeHomSpace<'a>,
    axis_plan: TensorContractAxisPlan,
}

pub(super) struct ValidatedCoreContract<'a, R> {
    preflight: CoreContractPreflight<'a, R>,
}

impl<'a, R> CoreContractPreflight<'a, R>
where
    R: FusionRule,
{
    pub(super) fn compile(
        rule: &'a R,
        dst: &'a DynamicFusionMapSpace,
        lhs: &'a DynamicFusionMapSpace,
        rhs: &'a DynamicFusionMapSpace,
        axes: TensorContractSpec<'_>,
    ) -> Result<Self, OperationError> {
        validate_fusion_contract_rule(rule, dst, lhs, rhs)?;
        Self::compile_homspaces(rule, dst.homspace(), lhs.homspace(), rhs.homspace(), axes)
    }

    pub(super) fn compile_homspaces(
        rule: &'a R,
        dst_homspace: &'a FusionTreeHomSpace,
        lhs_homspace: &'a FusionTreeHomSpace,
        rhs_homspace: &'a FusionTreeHomSpace,
        axes: TensorContractSpec<'_>,
    ) -> Result<Self, OperationError> {
        Self::compile_oriented(
            rule,
            dst_homspace,
            OrientedFusionTreeHomSpace::new(lhs_homspace, FusionTreePairOrientation::Direct),
            OrientedFusionTreeHomSpace::new(rhs_homspace, FusionTreePairOrientation::Direct),
            axes,
        )
    }

    pub(super) fn compile_oriented(
        rule: &'a R,
        dst_homspace: &'a FusionTreeHomSpace,
        lhs_homspace: OrientedFusionTreeHomSpace<'a>,
        rhs_homspace: OrientedFusionTreeHomSpace<'a>,
        axes: TensorContractSpec<'_>,
    ) -> Result<Self, OperationError> {
        let axis_plan = TensorContractAxisPlan::compile(
            lhs_homspace.rank(),
            rhs_homspace.rank(),
            dst_homspace.rank(),
            axes,
        )?;
        Ok(Self {
            rule,
            dst_homspace,
            lhs_homspace,
            rhs_homspace,
            axis_plan,
        })
    }

    pub(super) fn has_conjugation(&self) -> bool {
        self.axis_plan.lhs_conjugate || self.axis_plan.rhs_conjugate
    }

    pub(super) fn validate_core_geometry(
        self,
    ) -> Result<Option<ValidatedCoreContract<'a, R>>, OperationError> {
        if !is_core_form_source(
            self.lhs_homspace.rank(),
            self.lhs_homspace.nout(),
            self.rhs_homspace.nout(),
            &self.axis_plan,
        ) || !is_core_form_output(
            self.dst_homspace.codomain().len(),
            self.lhs_homspace.nout(),
            self.rhs_homspace.rank(),
            self.rhs_homspace.nout(),
            &self.axis_plan,
        ) {
            return Ok(None);
        }
        let expected_homspace = derive_expected_core_homspace(
            self.rule,
            self.lhs_homspace,
            self.rhs_homspace,
            self.axis_plan.lhs_contracting_axes.as_slice(),
            self.axis_plan.rhs_contracting_axes.as_slice(),
            self.axis_plan.output_axes.as_slice(),
            self.dst_homspace.codomain().len(),
        )?;
        if expected_homspace != *self.dst_homspace {
            return Err(OperationError::StructureMismatch { tensor: "dst" });
        }
        Ok(Some(ValidatedCoreContract { preflight: self }))
    }

    pub(super) fn require_core_geometry(
        self,
    ) -> Result<ValidatedCoreContract<'a, R>, OperationError> {
        self.validate_core_geometry()?
            .ok_or(OperationError::UnsupportedTensorContractScope {
                message: "core fusion-block contraction requires core source and output axes",
            })
    }
}

impl<'a, R> ValidatedCoreContract<'a, R> {
    pub(super) fn rule(&self) -> &'a R {
        self.preflight.rule
    }

    pub(super) fn rhs_homspace(&self) -> OrientedFusionTreeHomSpace<'a> {
        self.preflight.rhs_homspace
    }

    pub(super) fn rhs_contracting_axes(&self) -> &[usize] {
        &self.preflight.axis_plan.rhs_contracting_axes
    }
}

fn derive_expected_core_homspace<R>(
    rule: &R,
    lhs: OrientedFusionTreeHomSpace<'_>,
    rhs: OrientedFusionTreeHomSpace<'_>,
    lhs_contracting_axes: &[usize],
    rhs_contracting_axes: &[usize],
    output_axes: &[usize],
    dst_nout: usize,
) -> Result<FusionTreeHomSpace, OperationError>
where
    R: FusionRule,
{
    #[cfg(test)]
    EXPECTED_CORE_HOMSPACE_DERIVATIONS.set(EXPECTED_CORE_HOMSPACE_DERIVATIONS.get() + 1);
    OrientedFusionTreeHomSpace::tensorcontract_homspace(
        rule,
        lhs,
        rhs,
        lhs_contracting_axes,
        rhs_contracting_axes,
        output_axes,
        dst_nout,
    )
    .map_err(OperationError::from_core_preserving_context)
}

#[cfg(test)]
thread_local! {
    static EXPECTED_CORE_HOMSPACE_DERIVATIONS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

#[cfg(test)]
pub(crate) fn reset_core_contract_derivations() {
    super::structure::reset_tensor_contract_axis_plan_compiles();
    EXPECTED_CORE_HOMSPACE_DERIVATIONS.set(0);
}

#[cfg(test)]
pub(crate) fn core_contract_derivations() -> (usize, usize) {
    (
        super::structure::tensor_contract_axis_plan_compiles(),
        EXPECTED_CORE_HOMSPACE_DERIVATIONS.get(),
    )
}

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

/// Generic-fusion (Stage B3c-1) sibling of the multiplicity-free core classifier:
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
    if !is_core_form_source(
        lhs_space.rank(),
        lhs_space.nout(),
        rhs_space.nout(),
        &axis_plan,
    ) || !is_core_form_output(
        dst_space.nout(),
        lhs_space.nout(),
        rhs_space.rank(),
        rhs_space.nout(),
        &axis_plan,
    ) {
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
    lhs_rank: usize,
    lhs_nout: usize,
    rhs_nout: usize,
    axis_plan: &TensorContractAxisPlan,
) -> bool {
    axis_plan
        .lhs_contracting_axes
        .iter()
        .copied()
        .eq(lhs_nout..lhs_rank)
        && axis_plan
            .rhs_contracting_axes
            .iter()
            .copied()
            .eq(0..rhs_nout)
}

fn is_core_form_output(
    dst_nout: usize,
    lhs_nout: usize,
    rhs_rank: usize,
    rhs_nout: usize,
    axis_plan: &TensorContractAxisPlan,
) -> bool {
    let output_rank = lhs_nout + (rhs_rank - rhs_nout);
    dst_nout == lhs_nout && axis_plan.output_axes.iter().copied().eq(0..output_rank)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::{Cell, RefCell};
    use std::rc::Rc;
    use tenet_core::{
        BlockStructure, BraidingStyleKind, CoreError, FusionProductSpace, FusionStyleKind,
        FusionTensorMapSpace, FusionTreePairKey, HostReadableStorage, HostWritableStorage,
        MultiplicityIndex, ProductFusionRule, SU2FusionRule, SU2Irrep, SectorLeg, SectorVec,
        TensorMap, TensorMapSpace, TensorStorage, Trivial, U1FusionRule, U1Irrep, Z2FusionRule,
    };
    use tenet_core::{Placement, SimilarStorage};
    use tenet_operations::fusion_replay::HostFusionBlockContractWorkspace;
    use tenet_operations::storage_scratch::StorageFusionBlockContractWorkspace;
    use tenet_operations::ReportsPlacement;

    use crate::{DenseTreeTransformOperations, TensorContractWorkspace};

    fn reset_layout_lookups() {
        FUSION_LAYOUT_LOOKUPS.with(|lookups| lookups.set(0));
        FUSION_LAYOUT_COMPILES.set(0);
    }

    fn layout_lookups() -> usize {
        FUSION_LAYOUT_LOOKUPS.with(Cell::get)
    }

    fn layout_compiles() -> usize {
        FUSION_LAYOUT_COMPILES.get()
    }

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
        assert_eq!(facts.len(), 2);
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

        // What: the actual incomplete SU2 grid passes the executable plan's
        // packed-matrix geometry proof, not only the layout builder's count.
        let mut active = HashSet::new();
        let groups = layout
            .groups
            .iter()
            .cloned()
            .map(|group| {
                active.extend(group.block_indices.iter().copied());
                FusionBlockContractGroupPlan::new(group.clone(), group.clone(), group)
            })
            .collect::<Result<Vec<_>, _>>()
            .unwrap();
        FusionBlockContractPlan::from_parts(
            Arc::clone(sparse.structure()),
            Arc::clone(sparse.structure()),
            Arc::clone(sparse.structure()),
            fusion_scale_block_layouts_excluding(sparse.structure(), &active).unwrap(),
            groups,
        )
        .unwrap();
    }

    fn assert_missing_input_group_scales_destination<R>(rule: &R, coupled: [SectorId; 2])
    where
        R: MultiplicityFreeRigidSymbols<Scalar = f64>,
    {
        let codomain = SectorLeg::new(coupled.map(|sector| (sector, 1)), false);
        let domain = SectorLeg::new(coupled.map(|sector| (sector, 1)), false);
        let homspace = FusionTreeHomSpace::new(
            FusionProductSpace::new([codomain]),
            FusionProductSpace::new([domain]),
        );
        let keys = homspace.fusion_tree_keys(rule);
        assert_eq!(keys.len(), 2);
        let full_order = vec![keys[1].clone(), keys[0].clone()];
        let make_space = |keys: Vec<FusionTreePairKey>| {
            FusionTensorMapSpace::new_unbound(
                TensorMapSpace::<1, 1>::from_dims([2], [2]).unwrap(),
                homspace.clone(),
                crate::tests::packed_fixture_structure(
                    2,
                    keys.into_iter().map(|key| (key, vec![1, 1])),
                )
                .unwrap(),
            )
            .unwrap()
            .try_bind_rule(rule)
            .unwrap()
        };
        let lhs = DynamicFusionMapSpace::from_typed(&make_space(full_order.clone()));
        let rhs = DynamicFusionMapSpace::from_typed(&make_space(vec![keys[0].clone()]));
        let dst = DynamicFusionMapSpace::from_typed(&make_space(full_order));

        reset_layout_lookups();
        let plan = compile_fusion_block_contract_plan(
            rule,
            &dst,
            &lhs,
            &rhs,
            TensorContractSpec::with_default_output_order(&[1], &[0]),
        )
        .unwrap();
        // What: each destination group performs one LHS and one RHS lookup;
        // ordinary joins never build or query a prelowered block locator.
        assert_eq!(layout_lookups(), 4);

        let mut output = vec![11.0, 7.0];
        let mut dense = DenseTreeTransformOperations::default();
        let mut dense_workspace = TensorContractWorkspace::default();
        let mut fusion_workspace = FusionBlockContractWorkspace::<f64>::default();
        plan.execute_raw(
            &mut crate::StridedHostKernelAdapter::default(),
            &mut BackendRank2Gemm {
                backend: &mut dense,
                workspace: &mut dense_workspace,
            },
            &mut fusion_workspace,
            dst.structure(),
            &mut output,
            lhs.structure(),
            &[3.0, 2.0],
            rhs.structure(),
            &[5.0],
            2.0,
            3.0,
        )
        .unwrap();

        // What: the missing RHS group is structural zero and receives beta
        // exactly once while the matched group applies alpha and beta.
        assert_eq!(output, vec![33.0, 41.0]);
    }

    #[test]
    fn destination_order_join_preserves_structural_zero_for_builtin_rules() {
        assert_missing_input_group_scales_destination(
            &U1FusionRule,
            [U1Irrep::new(-1).sector_id(), U1Irrep::new(1).sector_id()],
        );
        assert_missing_input_group_scales_destination(
            &SU2FusionRule,
            [
                SU2Irrep::from_twice_spin(0).sector_id(),
                SU2Irrep::from_twice_spin(1).sector_id(),
            ],
        );
        type U1Su2Rule = ProductFusionRule<U1FusionRule, SU2FusionRule>;
        let product = U1Su2Rule::default();
        assert_missing_input_group_scales_destination(
            &product,
            [
                product.encode_sector(
                    U1Irrep::new(0).sector_id(),
                    SU2Irrep::from_twice_spin(0).sector_id(),
                ),
                product.encode_sector(
                    U1Irrep::new(1).sector_id(),
                    SU2Irrep::from_twice_spin(1).sector_id(),
                ),
            ],
        );
    }

    #[test]
    fn canonical_region_join_scales_missing_sector_without_layout_compile() {
        let rule = Z2FusionRule;
        let outer = || SectorLeg::new([(SectorId::new(0), 1), (SectorId::new(1), 1)], false);
        let inner = || SectorLeg::new([(SectorId::new(0), 1)], false);
        let space = |codomain: SectorLeg, domain: SectorLeg, dims, shapes| {
            FusionTensorMapSpace::from_degeneracy_shapes_coupled(
                dims,
                FusionTreeHomSpace::new(
                    FusionProductSpace::new([codomain]),
                    FusionProductSpace::new([domain]),
                ),
                &rule,
                shapes,
            )
            .unwrap()
        };
        let lhs = DynamicFusionMapSpace::from_typed(&space(
            outer(),
            inner(),
            TensorMapSpace::<1, 1>::from_dims([2], [1]).unwrap(),
            vec![vec![1, 1]],
        ));
        let rhs = DynamicFusionMapSpace::from_typed(&space(
            inner(),
            outer(),
            TensorMapSpace::<1, 1>::from_dims([1], [2]).unwrap(),
            vec![vec![1, 1]],
        ));
        let dst = DynamicFusionMapSpace::from_typed(&space(
            outer(),
            outer(),
            TensorMapSpace::<1, 1>::from_dims([2], [2]).unwrap(),
            vec![vec![1, 1], vec![1, 1]],
        ));

        reset_layout_lookups();
        let plan = compile_fusion_block_contract_plan(
            &rule,
            &dst,
            &lhs,
            &rhs,
            TensorContractSpec::with_default_output_order(&[1], &[0]),
        )
        .unwrap();
        assert_eq!(layout_compiles(), 0);

        let mut output = vec![11.0, 7.0];
        let mut dense = DenseTreeTransformOperations::default();
        let mut dense_workspace = TensorContractWorkspace::default();
        let mut fusion_workspace = FusionBlockContractWorkspace::<f64>::default();
        plan.execute_raw(
            &mut crate::StridedHostKernelAdapter::default(),
            &mut BackendRank2Gemm {
                backend: &mut dense,
                workspace: &mut dense_workspace,
            },
            &mut fusion_workspace,
            dst.structure(),
            &mut output,
            lhs.structure(),
            &[3.0],
            rhs.structure(),
            &[5.0],
            2.0,
            3.0,
        )
        .unwrap();

        // What: the matched sector applies alpha and beta, while the absent
        // inner sector applies beta exactly once across its contiguous range.
        assert_eq!(output, vec![63.0, 21.0]);
    }

    #[derive(Clone, Copy)]
    struct LayoutToyGenericRule;

    impl FusionRule for LayoutToyGenericRule {
        fn rule_identity(&self) -> tenet_core::RuleIdentity {
            tenet_core::RuleIdentity::of_type::<Self>()
        }

        fn fusion_style(&self) -> FusionStyleKind {
            FusionStyleKind::Generic
        }

        fn braiding_style(&self) -> BraidingStyleKind {
            BraidingStyleKind::Bosonic
        }

        fn vacuum(&self) -> SectorId {
            SectorId::new(0)
        }

        fn dual(&self, sector: SectorId) -> SectorId {
            sector
        }

        fn fusion_channels(&self, left: SectorId, right: SectorId) -> SectorVec {
            match (left.id(), right.id()) {
                (0, 0) => [SectorId::new(0)].into_iter().collect(),
                (1, 1) => [SectorId::new(1)].into_iter().collect(),
                _ => SectorVec::new(),
            }
        }

        fn nsymbol(&self, left: SectorId, right: SectorId, coupled: SectorId) -> usize {
            if (left.id(), right.id(), coupled.id()) == (1, 1, 1) {
                2
            } else {
                usize::from(self.fusion_channels(left, right).contains(&coupled))
            }
        }
    }

    fn toy_vertex_tree(coupled: usize, vertex: usize) -> FusionTreeKey {
        FusionTreeKey::try_new_for_rule(
            &LayoutToyGenericRule,
            [SectorId::new(coupled); 2],
            SectorId::new(coupled),
            [false; 2],
            [],
            [MultiplicityIndex::new(vertex).unwrap()],
        )
        .unwrap()
    }

    #[test]
    fn generic_layout_keeps_interleaved_cartesian_vertex_order() {
        let vacuum = FusionTreePairKey::pair(toy_vertex_tree(0, 1), toy_vertex_tree(0, 1));
        let pair = |row_vertex, col_vertex| {
            FusionTreePairKey::pair(
                toy_vertex_tree(1, row_vertex),
                toy_vertex_tree(1, col_vertex),
            )
        };
        let ordered = vec![
            pair(2, 2),
            vacuum.clone(),
            pair(1, 1),
            pair(1, 2),
            pair(2, 1),
        ];
        let duplicate = crate::tests::packed_fixture_structure(
            4,
            [ordered[0].clone(), ordered[0].clone()]
                .into_iter()
                .map(|key| (key, vec![1; 4])),
        )
        .unwrap_err();
        // What: an exact duplicate tree pair remains a typed construction error.
        assert!(matches!(duplicate, CoreError::DuplicateBlockKey { .. }));

        let leg = || SectorLeg::new([(SectorId::new(0), 1), (SectorId::new(1), 1)], false);
        let space = FusionTensorMapSpace::new_unbound(
            TensorMapSpace::<2, 2>::from_dims([2, 2], [2, 2]).unwrap(),
            FusionTreeHomSpace::new(
                FusionProductSpace::new([leg(), leg()]),
                FusionProductSpace::new([leg(), leg()]),
            ),
            crate::tests::packed_fixture_structure(
                4,
                ordered.into_iter().map(|key| (key, vec![1; 4])),
            )
            .unwrap(),
        )
        .unwrap();
        let dynamic = DynamicFusionMapSpace::from_typed(&space);
        let layout = FusionBlockMatrixLayout::compile_generic(&dynamic).unwrap();

        // What: first destination occurrence fixes group order, while distinct
        // row/column vertex labels retain their full Cartesian block set.
        assert_eq!(
            layout
                .groups
                .iter()
                .map(|group| group.coupled)
                .collect::<Vec<_>>(),
            vec![SectorId::new(1), SectorId::new(0)]
        );
        let multiplicity = layout.group(SectorId::new(1)).unwrap();
        assert_eq!(multiplicity.block_indices, vec![0, 2, 3, 4]);
        assert_eq!((multiplicity.rows, multiplicity.cols), (2, 2));
        assert_eq!(
            multiplicity
                .subblocks
                .iter()
                .map(|subblock| subblock.matrix_offset)
                .collect::<Vec<_>>(),
            vec![0, 3, 1, 2]
        );
        assert!(!multiplicity.needs_clear);

        reset_layout_lookups();
        assert!(layout.group(SectorId::new(1)).is_some());
        assert!(layout.group(SectorId::new(0)).is_some());
        assert!(layout.group(SectorId::new(9)).is_none());
        // What: finalized coupled-sector hits and misses use one indexed probe each.
        assert_eq!(layout_lookups(), 3);
    }

    #[test]
    fn generic_multiplicity_grid_uses_canonical_region_gemm() {
        let rule = LayoutToyGenericRule;
        let leg = || SectorLeg::new([(SectorId::new(1), 1)], false);
        let homspace = || {
            FusionTreeHomSpace::new(
                FusionProductSpace::new([leg(), leg()]),
                FusionProductSpace::new([leg(), leg()]),
            )
        };
        let space = || {
            let homspace = homspace();
            let key_count = homspace.fusion_tree_keys_generic(&rule).unwrap().len();
            assert_eq!(key_count, 4);
            DynamicFusionMapSpace::from_degeneracy_shapes_generic(
                &rule,
                homspace,
                vec![vec![1; 4]; key_count],
            )
            .unwrap()
        };
        let lhs = space();
        let rhs = space();
        let dst = space();

        reset_layout_lookups();
        let plan = compile_fusion_block_contract_plan_generic(
            &rule,
            &dst,
            &lhs,
            &rhs,
            TensorContractSpec::with_default_output_order(&[2, 3], &[0, 1]),
        )
        .unwrap();
        assert_eq!(layout_compiles(), 0);

        let mut output = vec![0.0; 4];
        let mut dense = DenseTreeTransformOperations::default();
        let mut dense_workspace = TensorContractWorkspace::default();
        let mut fusion_workspace = FusionBlockContractWorkspace::<f64>::default();
        plan.execute_raw(
            &mut crate::StridedHostKernelAdapter::default(),
            &mut BackendRank2Gemm {
                backend: &mut dense,
                workspace: &mut dense_workspace,
            },
            &mut fusion_workspace,
            dst.structure(),
            &mut output,
            lhs.structure(),
            &[1.0, 2.0, 3.0, 4.0],
            rhs.structure(),
            &[5.0, 6.0, 7.0, 8.0],
            1.0,
            0.0,
        )
        .unwrap();

        // What: outer-multiplicity vertices remain matrix rows/columns on the
        // direct canonical route.
        assert_eq!(output, vec![23.0, 34.0, 31.0, 46.0]);
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
        reset_layout_lookups();
        let plan = compile_fusion_block_contract_plan(
            &rule,
            &DynamicFusionMapSpace::from_typed(&space),
            &DynamicFusionMapSpace::from_typed(&space),
            &DynamicFusionMapSpace::from_typed(&space),
            TensorContractSpec::with_default_output_order(&[1], &[0]),
        )
        .unwrap();
        // What: canonical tensor-owned regions bypass the per-tree layout compiler.
        assert_eq!(layout_compiles(), 0);

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

    fn z2_adjoint_mapping_spaces() -> (DynamicFusionMapSpace, DynamicFusionMapSpace) {
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
        let logical = crate::lowering::adjoint_fusion_space_view(&rule, &storage).unwrap();
        let logical = DynamicFusionMapSpace::from_typed(&logical);
        let storage = DynamicFusionMapSpace::from_typed(&storage);
        (logical, storage)
    }

    #[test]
    fn nonselfdual_u1_adjoint_projects_logical_order_to_parent_blocks() {
        let rule = U1FusionRule;
        let charges = [-1, 0, 1].map(|charge| U1Irrep::new(charge).sector_id());
        let codomain = SectorLeg::new(charges.map(|sector| (sector, 1)), false);
        let domain = SectorLeg::new(charges.map(|sector| (rule.dual(sector), 1)), false);
        let homspace = FusionTreeHomSpace::new(
            FusionProductSpace::new([codomain]),
            FusionProductSpace::new([domain]),
        );
        let canonical = FusionTensorMapSpace::from_degeneracy_shapes(
            TensorMapSpace::<1, 1>::from_dims([3], [3]).unwrap(),
            homspace.clone(),
            &rule,
            vec![vec![1, 1]; 3],
        )
        .unwrap();
        let reversed_keys = (0..canonical.subblock_structure().block_count())
            .rev()
            .map(|index| {
                canonical
                    .subblock_structure()
                    .block(index)
                    .unwrap()
                    .key()
                    .clone()
            })
            .collect::<Vec<_>>();
        let reordered = FusionTensorMapSpace::new_unbound(
            TensorMapSpace::<1, 1>::from_dims([3], [3]).unwrap(),
            homspace,
            crate::tests::packed_fixture_structure(
                2,
                reversed_keys.into_iter().map(|key| (key, vec![1, 1])),
            )
            .unwrap(),
        )
        .unwrap()
        .try_bind_rule(&rule)
        .unwrap();
        let storage = DynamicFusionMapSpace::from_typed(&reordered);
        let operand = crate::FusionOperand::adjoint(&storage)
            .prepare(
                &rule,
                super::super::dynamic_space::encoded_layout_primer::<U1FusionRule>,
            )
            .unwrap();
        let mapped = FusionBlockMatrixLayout::compile_operand(&rule, &operand, MatrixOp::Adjoint)
            .unwrap()
            .groups
            .iter()
            .flat_map(|group| group.block_indices.iter().copied())
            .collect::<Vec<_>>();

        // What: canonical non-self-dual adjoint keys address the reordered
        // parent blocks directly, without a materialized logical structure.
        assert_eq!(mapped, vec![2, 1, 0]);
    }

    #[test]
    fn direct_operand_keeps_noncanonical_parent_tree_order() {
        let rule = Z2FusionRule;
        let leg = || SectorLeg::new([(SectorId::new(0), 1), (SectorId::new(1), 1)], false);
        let homspace = FusionTreeHomSpace::new(
            FusionProductSpace::new([leg(), leg()]),
            FusionProductSpace::new([leg(), leg()]),
        );
        let keys = homspace.fusion_tree_keys(&rule);
        let shapes = vec![vec![1; 4]; keys.len()];
        let canonical = FusionTensorMapSpace::from_degeneracy_shapes_coupled(
            TensorMapSpace::<2, 2>::from_dims([2, 2], [2, 2]).unwrap(),
            homspace.clone(),
            &rule,
            shapes.clone(),
        )
        .unwrap();
        let mut reordered = keys.iter().cloned().zip(shapes).collect::<Vec<_>>();
        let mut start = 0usize;
        while start < reordered.len() {
            let coupled = reordered[start].0.codomain_tree().coupled();
            let end = start
                + reordered[start..]
                    .iter()
                    .take_while(|(key, _)| key.codomain_tree().coupled() == coupled)
                    .count();
            reordered[start..end].reverse();
            start = end;
        }
        let storage = FusionTensorMapSpace::new_unbound(
            TensorMapSpace::<2, 2>::from_dims([2, 2], [2, 2]).unwrap(),
            homspace,
            BlockStructure::coupled_sector_matrix_with_keys(&rule, 2, 4, reordered).unwrap(),
        )
        .unwrap()
        .try_bind_rule(&rule)
        .unwrap();
        let canonical = DynamicFusionMapSpace::from_typed(&canonical);
        let storage = DynamicFusionMapSpace::from_typed(&storage);
        let canonical_regions = canonical
            .structure()
            .coupled_sector_regions(canonical.nout())
            .unwrap()
            .unwrap();
        let storage_regions = storage
            .structure()
            .coupled_sector_regions(storage.nout())
            .unwrap()
            .unwrap();
        assert!(canonical_regions
            .iter()
            .zip(storage_regions.iter())
            .all(|(lhs, rhs)| (lhs.rows(), lhs.cols()) == (rhs.rows(), rhs.cols())));
        assert!(canonical_regions
            .iter()
            .zip(storage_regions.iter())
            .any(|(lhs, rhs)| lhs.row_trees() != rhs.row_trees()
                || lhs.col_trees() != rhs.col_trees()));

        let operand = crate::FusionOperand::direct(&storage)
            .prepare(
                &rule,
                super::super::dynamic_space::encoded_layout_primer::<Z2FusionRule>,
            )
            .unwrap();

        // What: Direct orientation keeps the tensor-owned block order and
        // introduces no canonical projection beside the parent structure.
        assert!(operand.is_direct());
        for index in 0..storage.structure().block_count() {
            assert_eq!(operand.storage_index(index).unwrap(), index);
            assert_eq!(
                BlockKey::from(operand.logical_key(index).unwrap().clone()),
                storage.structure().block(index).unwrap().key().clone()
            );
        }
    }

    #[test]
    fn adjoint_operand_without_canonical_regions_uses_exact_fallback() {
        let rule = Z2FusionRule;
        let (logical, storage) = z2_adjoint_mapping_spaces();
        super::super::dynamic_space::reset_fusion_operand_projection_prepares();
        let lhs = crate::FusionOperand::adjoint(&storage)
            .prepare(
                &rule,
                super::super::dynamic_space::encoded_layout_primer::<Z2FusionRule>,
            )
            .unwrap();
        let rhs = crate::FusionOperand::adjoint(&storage)
            .prepare(
                &rule,
                super::super::dynamic_space::encoded_layout_primer::<Z2FusionRule>,
            )
            .unwrap();

        reset_layout_lookups();
        let _plan = super::super::resolution::compile_composition_plan(
            &rule,
            &logical,
            &lhs,
            &rhs,
            TensorContractSpec::with_default_output_order_and_conjugation(&[1], &[0], true, true),
        )
        .unwrap();

        // What: a transposed logical structure without canonical coupled
        // regions prepares the exact projection and retains the tree-mapped
        // implementation.
        assert_eq!(
            super::super::dynamic_space::fusion_operand_projection_prepares(),
            2
        );
        assert!(layout_compiles() > 0);
    }

    #[test]
    fn rank22_adjoint_reentry_keeps_matrix_orientation() {
        let rule = Z2FusionRule;
        let leg = || SectorLeg::new([(SectorId::new(0), 2), (SectorId::new(1), 2)], false);
        let homspace = FusionTreeHomSpace::new(
            FusionProductSpace::new([leg(), leg()]),
            FusionProductSpace::new([leg(), leg()]),
        );
        let shapes = vec![vec![2; 4]; homspace.fusion_tree_keys(&rule).len()];
        let typed = FusionTensorMapSpace::from_degeneracy_shapes_coupled(
            TensorMapSpace::<2, 2>::from_dims([4, 4], [4, 4]).unwrap(),
            homspace,
            &rule,
            shapes,
        )
        .unwrap();
        let space = DynamicFusionMapSpace::from_typed(&typed);
        let lhs = crate::FusionOperand::adjoint(&space)
            .prepare(
                &rule,
                super::super::dynamic_space::encoded_layout_primer::<Z2FusionRule>,
            )
            .unwrap();
        let rhs = crate::FusionOperand::adjoint(&space)
            .prepare(
                &rule,
                super::super::dynamic_space::encoded_layout_primer::<Z2FusionRule>,
            )
            .unwrap();
        let plan = super::super::resolution::compile_composition_plan(
            &rule,
            &space,
            &lhs,
            &rhs,
            TensorContractSpec::with_default_output_order_and_conjugation(
                &[2, 3],
                &[0, 1],
                true,
                true,
            ),
        )
        .unwrap();

        let len = space.required_len().unwrap();
        let lhs_data = (0..len)
            .map(|index| (index % 17) as f64 - 8.0)
            .collect::<Vec<_>>();
        let rhs_data = (0..len)
            .map(|index| (index % 13) as f64 - 6.0)
            .collect::<Vec<_>>();
        let mut actual = vec![0.0; len];
        let mut backend = DenseTreeTransformOperations::default();
        let mut workspace = TensorContractWorkspace::default();
        plan.execute_raw(
            &mut crate::StridedHostKernelAdapter::default(),
            &mut BackendRank2Gemm {
                backend: &mut backend,
                workspace: &mut workspace,
            },
            &mut FusionBlockContractWorkspace::default(),
            space.structure(),
            &mut actual,
            space.structure(),
            &lhs_data,
            space.structure(),
            &rhs_data,
            1.0,
            0.0,
        )
        .unwrap();

        let mut expected = vec![0.0; len];
        for region in space
            .structure()
            .coupled_sector_regions(space.nout())
            .unwrap()
            .unwrap()
            .iter()
        {
            let start = region.range().start;
            let rows = region.rows();
            let cols = region.cols();
            for col in 0..rows {
                for row in 0..cols {
                    expected[start + row + cols * col] = (0..rows)
                        .map(|contracted| {
                            lhs_data[start + contracted + rows * row]
                                * rhs_data[start + col + rows * contracted]
                        })
                        .sum();
                }
            }
        }

        // What: post-projection core re-entry retains both lazy-adjoint
        // matrix operations instead of replaying parent storage as identity.
        assert_eq!(actual, expected);
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

fn compile_direct_coupled_region_plan(
    dst_space: &DynamicFusionMapSpace,
    lhs_storage: &DynamicFusionMapSpace,
    rhs_storage: &DynamicFusionMapSpace,
    lhs_op: MatrixOp,
    rhs_op: MatrixOp,
) -> Result<Option<FusionBlockContractPlan>, OperationError> {
    FusionBlockContractPlan::try_from_canonical_coupled_regions_with_ops(
        dst_space.structure(),
        dst_space.nout(),
        lhs_storage.structure(),
        lhs_storage.nout(),
        rhs_storage.structure(),
        rhs_storage.nout(),
        lhs_op,
        rhs_op,
    )
}

pub(crate) fn try_compile_oriented_canonical_core_plan<R>(
    validated: &ValidatedCoreContract<'_, R>,
    dst_space: &DynamicFusionMapSpace,
    lhs_storage: &DynamicFusionMapSpace,
    rhs_storage: &DynamicFusionMapSpace,
) -> Result<Option<FusionBlockContractPlan>, OperationError> {
    let matrix_op = |orientation| match orientation {
        FusionTreePairOrientation::Direct => MatrixOp::Identity,
        FusionTreePairOrientation::Adjoint => MatrixOp::Adjoint,
    };
    compile_direct_coupled_region_plan(
        dst_space,
        lhs_storage,
        rhs_storage,
        matrix_op(validated.preflight.lhs_homspace.orientation()),
        matrix_op(validated.preflight.rhs_homspace.orientation()),
    )
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
    reject_fusion_contract_conjugation(axes)?;
    let validated = CoreContractPreflight::compile_homspaces(
        rule,
        dst_space.homspace(),
        lhs_space.homspace(),
        rhs_space.homspace(),
        axes,
    )?
    .require_core_geometry()?;
    compile_fusion_block_contract_plan_validated(validated, dst_space, lhs_space, rhs_space)
}

pub(crate) fn compile_fusion_block_contract_plan_validated<R>(
    validated: ValidatedCoreContract<'_, R>,
    dst_space: &DynamicFusionMapSpace,
    lhs_space: &DynamicFusionMapSpace,
    rhs_space: &DynamicFusionMapSpace,
) -> Result<FusionBlockContractPlan, OperationError>
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64>,
{
    let rule = validated.preflight.rule;

    if let Some(plan) = compile_direct_coupled_region_plan(
        dst_space,
        lhs_space,
        rhs_space,
        MatrixOp::Identity,
        MatrixOp::Identity,
    )? {
        return Ok(plan);
    }

    let lhs_layout = FusionBlockMatrixLayout::compile(rule, lhs_space)?;
    let rhs_layout = FusionBlockMatrixLayout::compile(rule, rhs_space)?;
    let dst_layout = FusionBlockMatrixLayout::compile(rule, dst_space)?;

    let mut groups = Vec::new();
    let mut active_dst_blocks = HashSet::<usize>::new();
    for dst_group in dst_layout.groups {
        let Some(lhs_group) = lhs_layout.group(dst_group.coupled) else {
            continue;
        };
        let Some(rhs_group) = rhs_layout.group(dst_group.coupled) else {
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
            lhs_group.clone(),
            rhs_group.clone(),
            dst_group,
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

pub(crate) fn compile_fusion_block_contract_plan_prelowered_validated<R>(
    validated: ValidatedCoreContract<'_, R>,
    dst_space: &DynamicFusionMapSpace,
    lhs: &FusionOperandLayout<'_>,
    rhs: &FusionOperandLayout<'_>,
) -> Result<FusionBlockContractPlan, OperationError>
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64>,
{
    let rule = validated.preflight.rule;
    let lhs_op = match lhs.orientation() {
        FusionTreePairOrientation::Direct => MatrixOp::Identity,
        FusionTreePairOrientation::Adjoint => MatrixOp::Adjoint,
    };
    let rhs_op = match rhs.orientation() {
        FusionTreePairOrientation::Direct => MatrixOp::Identity,
        FusionTreePairOrientation::Adjoint => MatrixOp::Adjoint,
    };
    if let Some(plan) = try_compile_oriented_canonical_core_plan(
        &validated,
        dst_space,
        lhs.storage_space(),
        rhs.storage_space(),
    )? {
        return Ok(plan);
    }

    let compile_source = |source: &FusionOperandLayout<'_>, op| {
        if source.is_direct() {
            FusionBlockMatrixLayout::compile(rule, source.storage_space())
        } else {
            FusionBlockMatrixLayout::compile_operand(rule, source, op)
        }
    };
    let lhs_layout = compile_source(lhs, lhs_op)?;
    let rhs_layout = compile_source(rhs, rhs_op)?;
    let dst_layout = FusionBlockMatrixLayout::compile(rule, dst_space)?;

    let mut groups = Vec::new();
    let mut active_dst_blocks = HashSet::<usize>::new();
    for dst_group in dst_layout.groups {
        let Some(lhs_group) = lhs_layout.group(dst_group.coupled) else {
            continue;
        };
        let Some(rhs_group) = rhs_layout.group(dst_group.coupled) else {
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
            lhs_group.clone(),
            rhs_group.clone(),
            dst_group,
        )?);
    }
    FusionBlockContractPlan::from_parts_with_ops(
        Arc::clone(dst_space.structure()),
        Arc::clone(lhs.storage_space().structure()),
        Arc::clone(rhs.storage_space().structure()),
        fusion_scale_block_layouts_excluding(dst_space.structure(), &active_dst_blocks)?,
        groups,
        lhs_op,
        rhs_op,
    )
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

    if let Some(plan) = compile_direct_coupled_region_plan(
        dst_space,
        lhs_space,
        rhs_space,
        MatrixOp::Identity,
        MatrixOp::Identity,
    )? {
        return Ok(plan);
    }

    let lhs_layout = FusionBlockMatrixLayout::compile_generic(lhs_space)?;
    let rhs_layout = FusionBlockMatrixLayout::compile_generic(rhs_space)?;
    let dst_layout = FusionBlockMatrixLayout::compile_generic(dst_space)?;

    let mut groups = Vec::new();
    let mut active_dst_blocks = HashSet::<usize>::new();
    for dst_group in dst_layout.groups {
        let Some(lhs_group) = lhs_layout.group(dst_group.coupled) else {
            continue;
        };
        let Some(rhs_group) = rhs_layout.group(dst_group.coupled) else {
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
            lhs_group.clone(),
            rhs_group.clone(),
            dst_group,
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
    // Why not sort for a merge walk: expert block layouts retain first
    // destination occurrence order, while this compile-time map preserves it.
    group_indices: FxHashMap<SectorId, usize>,
}

impl FusionBlockMatrixLayout {
    fn compile<R>(rule: &R, space: &DynamicFusionMapSpace) -> Result<Self, OperationError>
    where
        R: MultiplicityFreeRigidSymbols<Scalar = f64>,
    {
        #[cfg(test)]
        FUSION_LAYOUT_COMPILES.set(FUSION_LAYOUT_COMPILES.get() + 1);
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
        Ok(Self {
            groups,
            group_indices,
        })
    }

    fn compile_operand<R>(
        rule: &R,
        source: &FusionOperandLayout<'_>,
        op: MatrixOp,
    ) -> Result<Self, OperationError>
    where
        R: MultiplicityFreeRigidSymbols<Scalar = f64>,
    {
        #[cfg(test)]
        FUSION_LAYOUT_COMPILES.set(FUSION_LAYOUT_COMPILES.get() + 1);
        let mut builders = Vec::<FusionBlockMatrixGroupBuilder>::new();
        let mut group_indices = FxHashMap::<SectorId, usize>::default();
        let storage = source.storage_space();
        for logical_index in 0..source.logical_block_count() {
            let key = source.logical_key(logical_index)?;
            let storage_index = source.storage_index(logical_index)?;
            let block = storage.structure().block(storage_index)?;
            let coupled = coupled_sector(key.codomain_tree());
            if coupled != coupled_sector(key.domain_tree()) {
                return Err(OperationError::FusionTreeGroupMismatch {
                    tensor: "fusion",
                    index: logical_index,
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
            let split = storage.nout();
            let (row_shape, col_shape) = match source.orientation() {
                FusionTreePairOrientation::Direct => {
                    (&block.shape()[..split], &block.shape()[split..])
                }
                FusionTreePairOrientation::Adjoint => {
                    (&block.shape()[split..], &block.shape()[..split])
                }
            };
            builders[group_index].add_tree_pair_mapped(
                key.codomain_tree().clone(),
                element_count(row_shape)?,
                key.domain_tree().clone(),
                element_count(col_shape)?,
                logical_index,
                storage_index,
            )?;
        }
        let mut groups = Vec::with_capacity(builders.len());
        for builder in builders {
            groups.push(builder.finish_operand(rule, source, op)?);
        }
        Ok(Self {
            groups,
            group_indices,
        })
    }

    /// Generic-fusion (Stage B3c-1) sibling of [`Self::compile`]: relaxed to any
    /// [`FusionRule`] (the layout only needs `coupled()`/`vacuum()` to group
    /// blocks by coupled sector — no F/R symbols). Outer-multiplicity vertex
    /// labels ride in the fusion-tree keys, so multiplicity blocks land in the
    /// right coupled group automatically.
    fn compile_generic(space: &DynamicFusionMapSpace) -> Result<Self, OperationError> {
        #[cfg(test)]
        FUSION_LAYOUT_COMPILES.set(FUSION_LAYOUT_COMPILES.get() + 1);
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
        Ok(Self {
            groups,
            group_indices,
        })
    }

    fn group(&self, coupled: SectorId) -> Option<&FusionBlockMatrixGroup> {
        #[cfg(test)]
        record_fusion_group_lookup();
        self.group_indices
            .get(&coupled)
            .and_then(|&group_index| self.groups.get(group_index))
    }
}

#[cfg(test)]
thread_local! {
    static FUSION_LAYOUT_LOOKUPS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
    static FUSION_LAYOUT_COMPILES: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

#[cfg(test)]
fn record_fusion_group_lookup() {
    FUSION_LAYOUT_LOOKUPS.with(|lookups| {
        lookups.set(lookups.get() + 1);
    });
}

#[derive(Clone, Debug)]
struct FusionBlockMatrixGroupBuilder {
    coupled: SectorId,
    row_offsets: FxHashMap<FusionTreeKey, TreeMatrixOffset>,
    col_offsets: FxHashMap<FusionTreeKey, TreeMatrixOffset>,
    tree_pairs: HashSet<(FusionTreeKey, FusionTreeKey)>,
    blocks: Vec<usize>,
    logical_blocks: Option<Vec<usize>>,
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
            logical_blocks: None,
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
        self.add_tree_pair_entry(row_tree, row_dim, col_tree, col_dim, block_index)
    }

    #[allow(clippy::too_many_arguments)]
    fn add_tree_pair_mapped(
        &mut self,
        row_tree: FusionTreeKey,
        row_dim: usize,
        col_tree: FusionTreeKey,
        col_dim: usize,
        logical_block_index: usize,
        storage_block_index: usize,
    ) -> Result<(), OperationError> {
        self.add_tree_pair_entry(row_tree, row_dim, col_tree, col_dim, storage_block_index)?;
        self.logical_blocks
            .get_or_insert_with(Vec::new)
            .push(logical_block_index);
        Ok(())
    }

    fn add_tree_pair_entry(
        &mut self,
        row_tree: FusionTreeKey,
        row_dim: usize,
        col_tree: FusionTreeKey,
        col_dim: usize,
        storage_block_index: usize,
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
        self.blocks.push(storage_block_index);
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

    fn finish_operand<R>(
        self,
        rule: &R,
        source: &FusionOperandLayout<'_>,
        op: MatrixOp,
    ) -> Result<FusionBlockMatrixGroup, OperationError>
    where
        R: MultiplicityFreeRigidSymbols<Scalar = f64>,
    {
        let storage = source.storage_space();
        let physical_rows = match op {
            MatrixOp::Identity => self.rows,
            MatrixOp::Transpose | MatrixOp::Adjoint => self.cols,
        };
        let logical_blocks =
            self.logical_blocks
                .as_deref()
                .ok_or(OperationError::StructureMismatch {
                    tensor: "oriented fusion group",
                })?;
        if logical_blocks.len() != self.blocks.len() {
            return Err(OperationError::StructureMismatch {
                tensor: "oriented fusion group",
            });
        }
        let mut subblocks = Vec::with_capacity(self.blocks.len());
        for (&logical_index, &storage_index) in logical_blocks.iter().zip(&self.blocks) {
            let key = source.logical_key(logical_index)?;
            let block = storage.structure().block(storage_index)?;
            let row = self
                .row_offsets
                .get(key.codomain_tree())
                .expect("row tree offset collected before finish");
            let col = self
                .col_offsets
                .get(key.domain_tree())
                .expect("column tree offset collected before finish");
            let split = storage.nout();
            let mut matrix_strides = Vec::<isize>::with_capacity(block.shape().len());
            matrix_strides.extend(column_major_strides_isize(&block.shape()[..split])?);
            for stride in column_major_strides_usize(&block.shape()[split..])? {
                let matrix_stride = stride
                    .checked_mul(physical_rows)
                    .ok_or(OperationError::ElementCountOverflow)?;
                matrix_strides.push(isize::try_from(matrix_stride).map_err(|_| {
                    OperationError::StrideOverflow {
                        value: matrix_stride,
                    }
                })?);
            }
            let matrix_offset = match op {
                MatrixOp::Identity => col
                    .offset
                    .checked_mul(self.rows)
                    .and_then(|offset| offset.checked_add(row.offset)),
                MatrixOp::Transpose | MatrixOp::Adjoint => row
                    .offset
                    .checked_mul(self.cols)
                    .and_then(|offset| offset.checked_add(col.offset)),
            }
            .ok_or(OperationError::ElementCountOverflow)?;
            subblocks.push(FusionSubblockMatrixLayout {
                block: FusionStridedBlockLayout {
                    shape: block.shape().to_vec(),
                    strides: strides_to_isize(block.strides())?,
                    offset: offset_to_isize(block.offset())?,
                },
                matrix_offset: offset_to_isize(matrix_offset)?,
                matrix_strides,
                coefficient: rule.scalar_one(),
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
            block_indices: self.blocks,
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
