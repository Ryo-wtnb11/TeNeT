use super::*;

#[derive(Clone, Copy, Debug)]
struct Z2xZ3PointedRule;

impl Z2xZ3PointedRule {
    const fn encode(z2: usize, z3: usize) -> SectorId {
        SectorId::new((z2 % 2) + 2 * (z3 % 3))
    }

    const fn decode(sector: SectorId) -> (usize, usize) {
        (sector.id() % 2, (sector.id() / 2) % 3)
    }
}

impl FusionRule for Z2xZ3PointedRule {
    fn rule_identity(&self) -> RuleIdentity {
        RuleIdentity::of_type::<Self>()
    }
    fn fusion_style(&self) -> FusionStyleKind {
        FusionStyleKind::Unique
    }

    fn braiding_style(&self) -> BraidingStyleKind {
        BraidingStyleKind::Bosonic
    }

    fn vacuum(&self) -> SectorId {
        Self::encode(0, 0)
    }

    fn dual(&self, sector: SectorId) -> SectorId {
        let (z2, z3) = Self::decode(sector);
        Self::encode((2 - z2) % 2, (3 - z3) % 3)
    }

    fn fusion_channels(&self, left: SectorId, right: SectorId) -> SectorVec {
        let (left_z2, left_z3) = Self::decode(left);
        let (right_z2, right_z3) = Self::decode(right);
        smallvec![Self::encode(
            (left_z2 + right_z2) % 2,
            (left_z3 + right_z3) % 3,
        )]
    }
}

impl MultiplicityFreeFusionRule for Z2xZ3PointedRule {}

#[test]
fn block_view_validates_column_major_layout() {
    let data = [10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0];
    let shape = [2, 3];
    let strides = [1, 2];
    let view = BlockView::new(&data, &shape, &strides, 1).unwrap();
    assert_eq!(view.shape(), &[2, 3]);
    assert_eq!(view.strides(), &[1, 2]);
    assert_eq!(view.get(&[0, 0]), Some(&11.0));
    assert_eq!(view.get(&[1, 2]), Some(&16.0));
    assert_eq!(view.get(&[0]), None);
    assert_eq!(view.get(&[2, 0]), None);
    assert_eq!(view.get(&[0, 3]), None);
}

#[test]
fn block_view_rejects_out_of_bounds_layout() {
    let data = [0.0; 6];
    let shape = [2, 3];
    let strides = [1, 4];
    let err = BlockView::new(&data, &shape, &strides, 0).unwrap_err();
    assert_eq!(err, CoreError::OutOfBounds);
}

#[test]
fn trivial_tensormap_exposes_single_column_major_subblock() {
    let space = TensorMapSpace::<2, 1>::from_dims([2, 3], [4]).unwrap();
    let tensor =
        TensorMap::<f64, 2, 1>::from_vec((0..24).map(|x| x as f64).collect(), space).unwrap();

    assert_eq!(tensor.dim(), 24);
    assert_eq!(tensor.dims(), &[2, 3, 4]);
    assert_eq!(tensor.placement(), Placement::Host);
    assert_eq!(tensor.structure().block_count(), 1);

    let block = tensor.subblock().unwrap();
    assert_eq!(
        tensor.structure().block(0).unwrap().key(),
        &BlockKey::trivial()
    );
    assert_eq!(block.shape(), &[2, 3, 4]);
    assert_eq!(block.strides(), &[1, 2, 6]);
    assert_eq!(block.offset(), 0);
    assert_eq!(block.data()[23], 23.0);
}

#[test]
fn tensormap_subblock_mut_by_tree_updates_selected_storage() {
    let key =
        FusionTreePairKey::try_pair_from_sector_ids([1], [1], 1, [false], [true], [], [], [], [])
            .unwrap();
    let other =
        FusionTreePairKey::try_pair_from_sector_ids([0], [0], 0, [false], [true], [], [], [], [])
            .unwrap();
    let structure = packed_fixture_structure(
        2,
        [
            (BlockKey::from(other), vec![1, 2]),
            (BlockKey::from(key.clone()), vec![2, 1]),
        ],
    )
    .unwrap();
    let space = TensorMapSpace::<1, 1>::from_dims([3], [2]).unwrap();
    let mut tensor =
        TensorMap::<i32, 1, 1>::from_vec_with_structure(vec![1, 2, 3, 4], space, structure)
            .unwrap();

    {
        let mut view = tensor.subblock_mut_by_tree(&key).unwrap();
        let offset = view.offset();
        view.data_mut()[offset] = 30;
        view.data_mut()[offset + 1] = 40;
    }

    assert_eq!(tensor.data(), &[1, 2, 30, 40]);
}

#[test]
fn product_subblock_by_sectors_handles_simple_fusion_channels_without_manual_tree_keys() {
    type FpU1Rule = ProductFusionRule<FermionParityFusionRule, U1FusionRule>;
    type FpU1Su2Rule = ProductFusionRule<FpU1Rule, SU2FusionRule>;
    let left_rule = FpU1Rule::default();
    let rule = FpU1Su2Rule::default();
    let left_sector = |parity, charge| left_rule.encode_component_ids(parity, u1(charge));
    let sector = |parity, charge, twice_spin| {
        rule.encode_component_ids(left_sector(parity, charge), su2(twice_spin))
    };

    let a = sector(z2_odd(), 1, 1);
    let b = sector(z2_odd(), -1, 1);
    let c0 = sector(z2_even(), 0, 0);
    let c1 = sector(z2_even(), 0, 2);
    let dense = TensorMapSpace::<2, 1>::from_dims([1, 1], [1]).unwrap();
    let hom = FusionTreeHomSpace::new(
        FusionProductSpace::new([
            SectorLeg::new([(a, 1)], false),
            SectorLeg::new([(b, 1)], false),
        ]),
        FusionProductSpace::new([SectorLeg::new([(c0, 1), (c1, 1)], false)]),
    );
    let fusion_space = FusionTensorMapSpace::from_degeneracy_shapes_coupled(
        dense,
        hom,
        &rule,
        [vec![1, 1, 1], vec![1, 1, 1]],
    )
    .unwrap();
    let tensor =
        TensorMap::<i32, 2, 1>::from_vec_with_fusion_space(vec![100, 200], fusion_space).unwrap();

    let c0_block = tensor.subblock_by_sectors(&rule, &[a, b, c0]).unwrap();
    let c1_block = tensor.subblock_by_sectors(&rule, &[a, b, c1]).unwrap();
    assert_eq!(c0_block.offset(), 0);
    assert_eq!(c0_block.data()[c0_block.offset()], 100);
    assert_eq!(c1_block.offset(), 1);
    assert_eq!(c1_block.data()[c1_block.offset()], 200);

    let all_c0_blocks = tensor.subblocks_by_sectors(&rule, &[a, b, c0]).unwrap();
    assert_eq!(all_c0_blocks.len(), 1);
    assert_eq!(all_c0_blocks[0].offset(), 0);
}

#[test]
fn tensormap_subblock_by_sectors_matches_z2_unique() {
    let rule = Z2FusionRule;
    let dense = TensorMapSpace::<1, 1>::from_dims([2], [2]).unwrap();
    let hom = FusionTreeHomSpace::new(
        FusionProductSpace::new([SectorLeg::new(
            [(SectorId::new(0), 1), (SectorId::new(1), 1)],
            false,
        )]),
        FusionProductSpace::new([SectorLeg::new(
            [(SectorId::new(0), 1), (SectorId::new(1), 1)],
            false,
        )]),
    );
    let fusion_space = FusionTensorMapSpace::from_degeneracy_shapes_coupled(
        dense,
        hom,
        &rule,
        [vec![1, 1], vec![1, 1]],
    )
    .unwrap();
    let tensor =
        TensorMap::<i32, 1, 1>::from_vec_with_fusion_space(vec![10, 20], fusion_space).unwrap();

    let block = tensor
        .subblock_by_sectors(&rule, &[SectorId::new(1), SectorId::new(1)])
        .unwrap();

    assert_eq!(block.offset(), 1);
    assert_eq!(block.data()[block.offset()], 20);
}

#[test]
fn tensormap_subblock_by_sectors_dualizes_z4_domain_sector() {
    let rule = Z4PointedRule;
    let dense = TensorMapSpace::<1, 1>::from_dims([1], [1]).unwrap();
    let hom = FusionTreeHomSpace::new(
        FusionProductSpace::new([SectorLeg::new([(SectorId::new(1), 1)], false)]),
        FusionProductSpace::new([SectorLeg::new([(SectorId::new(1), 1)], false)]),
    );
    let fusion_space =
        FusionTensorMapSpace::from_degeneracy_shapes_coupled(dense, hom, &rule, [vec![1, 1]])
            .unwrap();
    let tensor =
        TensorMap::<f64, 1, 1>::from_vec_with_fusion_space(vec![3.5], fusion_space).unwrap();

    let block = tensor
        .subblock_by_sectors(&rule, &[SectorId::new(1), SectorId::new(3)])
        .unwrap();

    assert_eq!(block.offset(), 0);
    assert_eq!(block.data()[0], 3.5);
}

#[test]
fn tensormap_subblock_by_sectors_handles_fermionic_z2_key() {
    let rule = FermionParityFusionRule;
    let dense = TensorMapSpace::<1, 1>::from_dims([1], [1]).unwrap();
    let hom = FusionTreeHomSpace::from_sector_ids([(1, 1)], [(1, 1)]);
    let fusion_space =
        FusionTensorMapSpace::from_degeneracy_shapes_coupled(dense, hom, &rule, [vec![1, 1]])
            .unwrap();
    let mut tensor =
        TensorMap::<i32, 1, 1>::from_vec_with_fusion_space(vec![7], fusion_space).unwrap();

    {
        let mut block = tensor
            .subblock_mut_by_sectors(&rule, &[SectorId::new(1), SectorId::new(1)])
            .unwrap();
        let offset = block.offset();
        block.data_mut()[offset] = 11;
    }

    assert_eq!(tensor.data(), &[11]);
}

#[test]
fn tensormap_subblock_by_sectors_handles_product_pointed_rule() {
    let rule = Z2xZ3PointedRule;
    let codomain_sector = Z2xZ3PointedRule::encode(1, 2);
    let domain_tree_sector = rule.dual(Z2xZ3PointedRule::encode(1, 1));
    let dense = TensorMapSpace::<1, 1>::from_dims([1], [1]).unwrap();
    let hom = FusionTreeHomSpace::new(
        FusionProductSpace::new([SectorLeg::new([(codomain_sector, 1)], false)]),
        FusionProductSpace::new([SectorLeg::new([(domain_tree_sector, 1)], false)]),
    );
    let fusion_space =
        FusionTensorMapSpace::from_degeneracy_shapes_coupled(dense, hom, &rule, [vec![1, 1]])
            .unwrap();
    let tensor =
        TensorMap::<i32, 1, 1>::from_vec_with_fusion_space(vec![42], fusion_space).unwrap();

    let block = tensor
        .subblock_by_sectors(&rule, &[codomain_sector, Z2xZ3PointedRule::encode(1, 1)])
        .unwrap();

    assert_eq!(block.data()[block.offset()], 42);
}

#[test]
fn subblock_by_sectors_requires_fusion_tensor_space() {
    let rule = Z2FusionRule;
    let space = TensorMapSpace::<1, 1>::from_dims([1], [1]).unwrap();
    let tensor = TensorMap::<f64, 1, 1>::from_vec(vec![1.0], space).unwrap();

    let err = tensor
        .subblock_by_sectors(&rule, &[SectorId::new(0), SectorId::new(0)])
        .unwrap_err();

    assert_eq!(err, CoreError::MissingFusionSpace);
}

#[test]
fn tensormap_accepts_packed_block_structure() {
    let space = TensorMapSpace::<2, 0>::from_dims([4, 4], []).unwrap();
    let structure = BlockStructure::packed_column_major(2, [vec![2, 3], vec![1, 4]]).unwrap();
    let tensor = TensorMap::<f64, 2, 0>::from_vec_with_structure(
        (0..10).map(|x| x as f64).collect(),
        space,
        structure,
    )
    .unwrap();

    assert_eq!(tensor.data().len(), 10);
    assert_eq!(tensor.dim(), 10);
    assert_eq!(tensor.storage_dim(), 10);
    assert_eq!(tensor.dense_dim(), 16);
    assert_eq!(tensor.structure().rank(), 2);

    let first = tensor.block(0).unwrap();
    assert_eq!(first.shape(), &[2, 3]);
    assert_eq!(first.offset(), 0);

    let second = tensor.block(1).unwrap();
    assert_eq!(second.shape(), &[1, 4]);
    assert_eq!(second.offset(), 6);
}

#[test]
fn tensormap_rejects_structure_rank_that_does_not_match_space_rank() {
    let space = TensorMapSpace::<2, 0>::from_dims([2, 3], []).unwrap();
    let structure = BlockStructure::packed_column_major(1, [vec![6]]).unwrap();
    let err = TensorMap::<f64, 2, 0>::from_vec_with_structure(vec![0.0; 6], space, structure)
        .unwrap_err();

    assert_eq!(
        err,
        CoreError::StructureRankMismatch {
            expected: 2,
            actual: 1
        }
    );
}

#[test]
fn tensormap_rejects_incorrect_data_length() {
    let space = TensorMapSpace::<1, 1>::from_dims([2], [3]).unwrap();
    let err = TensorMap::<f64, 1, 1>::from_vec(vec![0.0; 5], space).unwrap_err();
    assert_eq!(
        err,
        CoreError::DimensionMismatch {
            expected: 6,
            actual: 5
        }
    );
}

#[derive(Debug)]
struct OpaqueReportedStorage {
    reported_len: std::cell::Cell<usize>,
}

impl TensorStorage<i32> for OpaqueReportedStorage {
    fn len(&self) -> usize {
        self.reported_len.get()
    }

    fn placement(&self) -> Placement {
        Placement::Host
    }
}

#[test]
fn generic_storage_extent_rechecks_interior_mutable_reported_length() {
    // What: custom opaque storage is still supported, but changing its
    // reported extent after construction is rejected in both directions.
    for reported_len in [1, 3] {
        let space = TensorMapSpace::<1, 0>::from_dims([2], []).unwrap();
        let tensor =
            TensorMap::<i32, 1, 0, Trivial, OpaqueReportedStorage>::from_storage_with_structure(
                OpaqueReportedStorage {
                    reported_len: std::cell::Cell::new(2),
                },
                space,
                BlockStructure::packed_column_major(1, [vec![2]]).unwrap(),
            )
            .unwrap();
        tensor.storage().reported_len.set(reported_len);

        assert_eq!(
            tensor.validate_storage_extent(reported_len),
            Err(CoreError::DimensionMismatch {
                expected: 2,
                actual: reported_len,
            })
        );
    }
}

#[test]
fn host_execution_rejects_slice_and_reported_extent_disagreement() {
    // What: host execution checks the actual slice independently of the
    // length reported during construction, for both short and oversized
    // external storage, before callbacks, views, or writes.
    for actual_len in [1, 3] {
        let tensor = adversarial_host_tensor(actual_len);
        assert_eq!(tensor.data().len(), actual_len);
        assert_host_execution_rejects_extent(
            tensor,
            CoreError::DimensionMismatch {
                expected: 2,
                actual: actual_len,
            },
        );
    }
}

#[test]
fn fusion_subblock_getters_reject_inexact_host_slices() {
    // What: fusion-tree and external-sector getter siblings share the same
    // exact host-slice boundary for immutable and mutable access.
    for actual_len in [0, 2] {
        let (mut tensor, key) = adversarial_fusion_host_tensor(actual_len);
        let error = CoreError::DimensionMismatch {
            expected: 1,
            actual: actual_len,
        };
        let sectors = [Z2Irrep::EVEN.sector_id(), Z2Irrep::EVEN.sector_id()];

        assert_eq!(tensor.subblock_by_tree(&key).unwrap_err(), error);
        assert_eq!(
            tensor
                .subblock_by_sectors(&Z2FusionRule, &sectors)
                .unwrap_err(),
            error
        );
        assert_eq!(
            tensor
                .subblocks_by_sectors(&Z2FusionRule, &sectors)
                .unwrap_err(),
            error
        );
        assert_eq!(tensor.subblock_mut_by_tree(&key).unwrap_err(), error);
        assert_eq!(
            tensor
                .subblock_mut_by_sectors(&Z2FusionRule, &sectors)
                .unwrap_err(),
            error
        );
        assert_eq!(tensor.data(), vec![10; actual_len]);
    }
}

#[test]
fn data_mut_changes_elements_without_changing_storage_length() {
    // What: ordinary host mutation remains available through a fixed-length
    // slice after the concrete mutable-storage escape is removed.
    let space = TensorMapSpace::<1, 0>::from_dims([2], []).unwrap();
    let mut tensor = TensorMap::<i32, 1, 0>::from_vec(vec![1, 2], space).unwrap();
    let len = tensor.data_mut().len();

    tensor.data_mut()[1] = 7;

    assert_eq!(tensor.data(), &[1, 7]);
    assert_eq!(tensor.data().len(), len);
}

#[test]
fn tensormap_allocates_similar_storage_from_backing_storage() {
    let space = TensorMapSpace::<1, 0>::from_dims([2], []).unwrap();
    let tensor = TensorMap::<f64, 1, 0>::from_vec(vec![1.0, 2.0], space).unwrap();
    let scratch = tensor.similar_storage_filled(3, 0.0);

    assert_eq!(scratch, vec![0.0; 3]);
    assert_eq!(scratch.placement(), tensor.placement());
}
