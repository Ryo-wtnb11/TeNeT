use super::*;
use std::cell::Cell;
use tenet_core::Placement;
use tenet_core::{
    BlockStructure, BraidingStyleKind, CoreError, FusionProductSpace, FusionStyleKind,
    FusionTensorMapSpace, FusionTreePairKey, MultiplicityIndex, ProductFusionRule, SU2FusionRule,
    SU2Irrep, SectorLeg, SectorVec, TensorMapSpace, TensorStorage, U1FusionRule, U1Irrep,
    Z2FusionRule,
};
use tenet_operations::fusion_replay::HostFusionBlockContractWorkspace;
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
    let scalar = FusionTensorMapSpace::from_degeneracy_shapes_coupled(
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
    let complete = FusionTensorMapSpace::from_degeneracy_shapes_coupled(
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

    let layout = FusionBlockMatrixLayout::compile(&sparse)
        .unwrap()
        .finish_all(|group| group.finish(&rule, &sparse));
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
        &mut BackendRank2Gemm::<_, _, f64>::new(&mut dense, &mut dense_workspace),
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
            product.encode_component_ids(
                U1Irrep::new(0).sector_id(),
                SU2Irrep::from_twice_spin(0).sector_id(),
            ),
            product.encode_component_ids(
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
        &mut BackendRank2Gemm::<_, _, f64>::new(&mut dense, &mut dense_workspace),
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
        crate::tests::packed_fixture_structure(4, ordered.into_iter().map(|key| (key, vec![1; 4])))
            .unwrap(),
    )
    .unwrap();
    let dynamic = DynamicFusionMapSpace::from_typed(&space);
    let mut layout = FusionBlockMatrixLayout::compile(&dynamic).unwrap();
    let finished = layout
        .clone()
        .finish_all(|group| group.finish_generic::<f64>(dynamic.structure(), dynamic.nout()));

    // What: first destination occurrence fixes group order, while distinct
    // row/column vertex labels retain their full Cartesian block set.
    assert_eq!(
        finished
            .groups
            .iter()
            .map(|group| group.coupled)
            .collect::<Vec<_>>(),
        vec![SectorId::new(1), SectorId::new(0)]
    );
    let multiplicity = &finished.groups[0];
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
    assert!(layout.take_group(SectorId::new(1)).is_some());
    assert!(layout.take_group(SectorId::new(0)).is_some());
    assert!(layout.take_group(SectorId::new(9)).is_none());
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
    let plan = compile_fusion_block_contract_plan_generic::<_, f64>(
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
        &mut BackendRank2Gemm::<_, _, f64>::new(&mut dense, &mut dense_workspace),
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
        FusionTensorMapSpace::from_degeneracy_shapes_coupled(
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
    plan.execute_direct_on_storage_prezeroed(&mut NaiveOpaqueGemm, &mut dst, &lhs, &rhs)
        .unwrap();

    // Two 1x1 sector matrices: dst = lhs * rhs per sector.
    assert_eq!(dst.cells, vec![10.0, 21.0]);
}

#[test]
fn storage_direct_replay_matches_host_execute_raw() {
    let rule = Z2FusionRule;
    let leg =
        |dims: usize| SectorLeg::new([(SectorId::new(0), dims), (SectorId::new(1), dims)], false);
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
        &mut BackendRank2Gemm::<_, _, f64>::new(&mut backend, &mut workspace),
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
    plan.execute_direct_on_storage_prezeroed(
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
    let storage = FusionTensorMapSpace::from_degeneracy_shapes_coupled(
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
    let canonical = FusionTensorMapSpace::from_degeneracy_shapes_coupled(
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
    let mapped = FusionBlockMatrixLayout::compile_operand(&operand)
        .unwrap()
        .finish_all(|group| group.finish_operand(&rule, &operand, MatrixOp::Adjoint))
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
    assert!(
        canonical_regions
            .iter()
            .zip(storage_regions.iter())
            .any(|(lhs, rhs)| lhs.row_trees() != rhs.row_trees()
                || lhs.col_trees() != rhs.col_trees())
    );

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
        TensorContractSpec::with_default_output_order_and_conjugation(&[2, 3], &[0, 1], true, true),
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
        &mut BackendRank2Gemm::<_, _, f64>::new(&mut backend, &mut workspace),
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
    plan.execute_direct_on_storage_prezeroed(&mut gemm, &mut expected, &lhs_data, &rhs_data)
        .unwrap();

    let mut ctx = CudaDenseContext::new(0).unwrap();
    let lhs_dev = CudaStorage::upload(&ctx, &lhs_data).unwrap();
    let rhs_dev = CudaStorage::upload(&ctx, &rhs_data).unwrap();
    let mut dst_dev = CudaStorage::upload(&ctx, &vec![0.0; len]).unwrap();
    plan.execute_direct_on_storage_prezeroed(
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
