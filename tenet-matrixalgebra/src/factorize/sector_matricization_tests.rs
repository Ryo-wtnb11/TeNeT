use super::*;
use tenet_core::{BlockSpec, FusionTreePairKey, Z2FusionRule};

struct PayloadFreeGeometry {
    sector: SectorId,
    rows: usize,
    cols: usize,
    row_tree: FusionTreeKey,
    row_shape: Vec<usize>,
    col_tree: FusionTreeKey,
    col_shape: Vec<usize>,
}

impl SectorGeometry for PayloadFreeGeometry {
    fn sector(&self) -> SectorId {
        self.sector
    }

    fn rows(&self) -> usize {
        self.rows
    }

    fn cols(&self) -> usize {
        self.cols
    }

    fn tree_count(&self, _side: FactorSide) -> usize {
        1
    }

    fn tree(&self, side: FactorSide, index: usize) -> Option<TreeExtentRef<'_>> {
        if index != 0 {
            return None;
        }
        let (tree, shape) = match side {
            FactorSide::Left => (&self.row_tree, self.row_shape.as_slice()),
            FactorSide::Right => (&self.col_tree, self.col_shape.as_slice()),
        };
        Some(TreeExtentRef {
            tree,
            offset: 0,
            shape,
        })
    }
}

#[derive(Clone, Copy)]
struct TestGenericRule;

impl FusionRule for TestGenericRule {
    fn rule_identity(&self) -> tenet_core::RuleIdentity {
        tenet_core::RuleIdentity::of_type::<Self>()
    }

    fn fusion_style(&self) -> tenet_core::FusionStyleKind {
        tenet_core::FusionStyleKind::Generic
    }

    fn braiding_style(&self) -> tenet_core::BraidingStyleKind {
        tenet_core::BraidingStyleKind::Bosonic
    }

    fn vacuum(&self) -> SectorId {
        SectorId::new(0)
    }

    fn dual(&self, sector: SectorId) -> SectorId {
        sector
    }

    fn fusion_channels(&self, left: SectorId, right: SectorId) -> tenet_core::SectorVec {
        match (left.id(), right.id()) {
            (0, sector) | (sector, 0) => [SectorId::new(sector)].into_iter().collect(),
            (1, 1) => [SectorId::new(0), SectorId::new(1)].into_iter().collect(),
            _ => tenet_core::SectorVec::new(),
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

fn z2_pair(codomain: [usize; 2], domain: [usize; 2], coupled: usize) -> FusionTreePairKey {
    let pair = FusionTreePairKey::try_pair_from_sector_ids(
        codomain,
        domain,
        coupled,
        [false; 2],
        [false; 2],
        std::iter::empty::<usize>(),
        std::iter::empty::<usize>(),
        [1],
        [1],
    )
    .unwrap();
    pair.validate_for_rule(&Z2FusionRule).unwrap();
    pair
}

#[test]
fn left_factor_publication_borrows_real_geometry_for_complex_output() {
    let x = SectorId::new(1);
    let vacuum = SectorId::new(0);
    let source_key = FusionTreePairKey::try_pair_from_sector_ids(
        [1, 1, 1],
        [1],
        1,
        [false; 3],
        [false],
        [0],
        std::iter::empty::<usize>(),
        [1, 1],
        std::iter::empty::<usize>(),
    )
    .unwrap();
    source_key.validate_for_rule(&Z2FusionRule).unwrap();
    let homspace = FusionTreeHomSpace::new(
        FusionProductSpace::new([
            SectorLeg::new([(x, 2)], false),
            SectorLeg::new([(x, 1)], false),
            SectorLeg::new([(x, 3)], false),
        ]),
        FusionProductSpace::new([SectorLeg::new([(x, 1)], false)]),
    );
    let authority = BoundDynamicFusionMapSpace::from_final_homspace_multiplicity_free(
        Arc::new(Z2FusionRule),
        homspace.clone(),
    )
    .unwrap();
    let geometry = [PayloadFreeGeometry {
        sector: x,
        rows: 6,
        cols: 1,
        row_tree: source_key.codomain_tree().clone(),
        row_shape: vec![2, 1, 3],
        col_tree: source_key.domain_tree().clone(),
        col_shape: vec![1],
    }];
    let row_tree_before = geometry[0].row_tree.clone();
    let col_tree_before = geometry[0].col_tree.clone();
    let expected = (0..12)
        .map(|index| Complex64::new(index as f64 + 0.5, 10.0 - index as f64))
        .collect::<Vec<_>>();
    let mut pairs = [FactorPair {
        sector: x,
        kept: 2,
        left: expected.clone(),
        left_rows: 6,
        right: Vec::new(),
        right_leading: 0,
    }];

    let selected_ptr = pairs[0].left.as_ptr();
    reset_one_sided_publication_probe();
    let factor = build_left_bound_factor(&authority, &homspace, &geometry, &mut pairs).unwrap();

    let probe = one_sided_publication_probe();
    assert_eq!(
        (
            probe.canonical_publications,
            probe.fallback_publications,
            probe.owner_reused,
            probe.appended_elements,
        ),
        (1, 0, 1, 0)
    );
    assert!(std::ptr::eq(selected_ptr, factor.data().as_ptr()));
    assert!(pairs[0].left.is_empty());
    assert_eq!(geometry[0].row_tree, row_tree_before);
    assert_eq!(geometry[0].col_tree, col_tree_before);
    assert_eq!(geometry[0].row_shape, [2, 1, 3]);
    assert_eq!(geometry[0].col_shape, [1]);
    let block = factor.space().space().structure().block(0).unwrap();
    assert_eq!(block.shape(), [2, 1, 3, 2]);
    let BlockKey::FusionTree(key) = block.key() else {
        panic!("left factor must retain the complete fusion-tree key")
    };
    assert_eq!(key.codomain_tree(), &row_tree_before);
    assert_eq!(key.domain_tree().uncoupled(), &[x]);
    assert_eq!(key.domain_tree().coupled(), x);
    assert_eq!(key.domain_tree().vertices(), &[]);
    assert_eq!(key.domain_tree().innerlines(), &[]);
    assert_eq!(factor.data().len(), expected.len());
    for bond in 0..2 {
        for third in 0..3 {
            for first in 0..2 {
                let output = block.offset()
                    + first * block.strides()[0]
                    + third * block.strides()[2]
                    + bond * block.strides()[3];
                let matrix_row = first + 2 * third;
                assert_eq!(factor.data()[output], expected[matrix_row + 6 * bond]);
            }
        }
    }
    assert_eq!(vacuum, row_tree_before.innerlines()[0]);
}

fn generic_pair(coupled: usize, row_vertex: usize, col_vertex: usize) -> FusionTreePairKey {
    FusionTreePairKey::try_pair_from_sector_ids(
        [1, 1],
        [1, 1],
        coupled,
        [false; 2],
        [false; 2],
        std::iter::empty::<usize>(),
        std::iter::empty::<usize>(),
        [row_vertex],
        [col_vertex],
    )
    .unwrap()
}

fn full_identity_pair(
    row_inner: usize,
    row_dual: bool,
    col_inner: usize,
    col_dual: bool,
) -> FusionTreePairKey {
    FusionTreePairKey::try_pair_from_sector_ids(
        [1, 2, 3],
        [4, 5, 6],
        9,
        [false, row_dual, false],
        [true, false, col_dual],
        [row_inner],
        [col_inner],
        [1, 2],
        [2, 1],
    )
    .unwrap()
}

#[test]
fn packed_and_region_geometry_preserve_dual_and_innerline_tree_identity() {
    let mut row_trees = vec![
        full_identity_pair(7, false, 11, false)
            .codomain_tree()
            .clone(),
        full_identity_pair(8, true, 11, false)
            .codomain_tree()
            .clone(),
    ];
    let mut col_trees = vec![
        full_identity_pair(7, false, 10, false)
            .domain_tree()
            .clone(),
        full_identity_pair(7, false, 11, true).domain_tree().clone(),
    ];
    row_trees.sort();
    col_trees.sort();
    assert_ne!(row_trees[0].innerlines(), row_trees[1].innerlines());
    assert_ne!(row_trees[0].is_dual(), row_trees[1].is_dual());
    assert_ne!(col_trees[0].innerlines(), col_trees[1].innerlines());
    assert_ne!(col_trees[0].is_dual(), col_trees[1].is_dual());

    let row_shapes = [vec![1, 2, 1], vec![2, 1, 1]];
    let col_shapes = [vec![1, 3, 1], vec![1, 1, 1]];
    let row_extents = row_shapes
        .iter()
        .map(|shape| shape.iter().product::<usize>())
        .collect::<Vec<_>>();
    let col_extents = col_shapes
        .iter()
        .map(|shape| shape.iter().product::<usize>())
        .collect::<Vec<_>>();
    let rows = row_extents.iter().sum::<usize>();
    let cols = col_extents.iter().sum::<usize>();
    let mut blocks = Vec::new();
    let mut col_offset = 0;
    for (col, (col_tree, col_shape)) in col_trees.iter().zip(&col_shapes).enumerate() {
        let mut row_offset = 0;
        for (row, (row_tree, row_shape)) in row_trees.iter().zip(&row_shapes).enumerate() {
            let mut shape = row_shape.clone();
            shape.extend_from_slice(col_shape);
            let mut strides = Vec::with_capacity(shape.len());
            let mut stride = 1;
            for &dimension in row_shape {
                strides.push(stride);
                stride *= dimension;
            }
            stride = rows;
            for &dimension in col_shape {
                strides.push(stride);
                stride *= dimension;
            }
            blocks.push(
                BlockSpec::with_key(
                    FusionTreePairKey::pair(row_tree.clone(), col_tree.clone()).into(),
                    shape,
                    strides,
                    row_offset + rows * col_offset,
                )
                .unwrap(),
            );
            row_offset += row_extents[row];
        }
        col_offset += col_extents[col];
    }
    let structure = BlockStructure::from_blocks_with_rank(6, blocks).unwrap();
    let data = (0..rows * cols)
        .map(|value| value as f64)
        .collect::<Vec<_>>();
    let regions = structure.coupled_sector_regions(3).unwrap().unwrap();
    let packed = sector_matricizations_generic(&structure, &data, 3).unwrap();
    assert_eq!(regions.len(), 1);
    assert_eq!(packed.len(), 1);
    assert!(matches!(
        generic_input_matricizations(&structure, &data, 3).unwrap(),
        InputMatricizations::Regions { .. }
    ));
    let region = &regions[0];
    let matrix = &packed[0];
    assert_eq!(matrix.data, data);
    assert_eq!(region.sector(), SectorId::new(9));
    assert_eq!((region.rows(), region.cols()), (4, 4));
    assert_eq!(region.sector(), matrix.sector());
    assert_eq!(
        (region.rows(), region.cols()),
        (matrix.rows(), matrix.cols())
    );
    for side in [FactorSide::Left, FactorSide::Right] {
        let (expected_trees, expected_offsets, expected_shapes) = match side {
            FactorSide::Left => (&row_trees, [0, 2], &row_shapes),
            FactorSide::Right => (&col_trees, [0, 3], &col_shapes),
        };
        assert_eq!(region.tree_count(side), matrix.tree_count(side));
        for index in 0..region.tree_count(side) {
            let borrowed = region.tree(side, index).unwrap();
            let owned = matrix.tree(side, index).unwrap();
            assert_eq!(borrowed.tree, &expected_trees[index]);
            assert_eq!(borrowed.offset, expected_offsets[index]);
            assert_eq!(borrowed.shape, expected_shapes[index]);
            assert_eq!(owned.tree, &expected_trees[index]);
            assert_eq!(owned.offset, expected_offsets[index]);
            assert_eq!(owned.shape, expected_shapes[index]);
            assert_eq!(borrowed.tree, owned.tree);
            assert_eq!(borrowed.offset, owned.offset);
            assert_eq!(borrowed.shape, owned.shape);
        }
    }
}

#[test]
fn sector_matricizations_preserve_encounter_order_and_padded_block_values() {
    // What: noncanonical storage packs repeated row/column trees into
    // first-encounter sector geometry without changing block copy order.
    let structure = BlockStructure::from_blocks_with_rank(
        4,
        vec![
            BlockSpec::with_key(
                z2_pair([0, 1], [1, 0], 1).into(),
                vec![1, 2, 3, 1],
                vec![2, 9, 1, 20],
                3,
            )
            .unwrap(),
            BlockSpec::with_key(
                z2_pair([0, 0], [1, 1], 0).into(),
                vec![2, 1, 1, 2],
                vec![2, 20, 1, 7],
                30,
            )
            .unwrap(),
            BlockSpec::with_key(
                z2_pair([0, 0], [0, 0], 0).into(),
                vec![2, 1, 2, 1],
                vec![3, 20, 1, 50],
                50,
            )
            .unwrap(),
            BlockSpec::with_key(
                z2_pair([1, 1], [1, 1], 0).into(),
                vec![1, 2, 1, 2],
                vec![4, 1, 20, 7],
                70,
            )
            .unwrap(),
            BlockSpec::with_key(
                z2_pair([1, 1], [0, 0], 0).into(),
                vec![1, 2, 2, 1],
                vec![4, 1, 5, 20],
                90,
            )
            .unwrap(),
        ],
    )
    .unwrap();
    let mut data = vec![0.0; structure.required_len().unwrap()];
    for (position, value) in [
        (3, 101.0),
        (12, 102.0),
        (4, 103.0),
        (13, 104.0),
        (5, 105.0),
        (14, 106.0),
        (30, 1.0),
        (32, 2.0),
        (37, 5.0),
        (39, 6.0),
        (50, 3.0),
        (53, 4.0),
        (51, 7.0),
        (54, 8.0),
        (70, 9.0),
        (71, 10.0),
        (77, 13.0),
        (78, 14.0),
        (90, 11.0),
        (91, 12.0),
        (95, 15.0),
        (96, 16.0),
    ] {
        data[position] = value;
    }

    let matrices = sector_matricizations::<f64>(&structure, &data, 2).unwrap();

    assert_eq!(
        matrices
            .iter()
            .map(|matrix| matrix.sector)
            .collect::<Vec<_>>(),
        [SectorId::new(1), SectorId::new(0)]
    );
    assert_eq!(matrices[0].rows, 2);
    assert_eq!(matrices[0].cols, 3);
    assert_eq!(matrices[0].data, [101.0, 102.0, 103.0, 104.0, 105.0, 106.0]);
    assert_eq!(
        matrices[0].row_trees,
        vec![(
            z2_pair([0, 1], [1, 0], 1).codomain_tree().clone(),
            0,
            vec![1, 2],
        )]
    );
    assert_eq!(
        matrices[0].col_trees,
        vec![(
            z2_pair([0, 1], [1, 0], 1).domain_tree().clone(),
            0,
            vec![3, 1],
        )]
    );

    assert_eq!(matrices[1].rows, 4);
    assert_eq!(matrices[1].cols, 4);
    assert_eq!(
        matrices[1].data,
        [1.0, 2.0, 9.0, 10.0, 5.0, 6.0, 13.0, 14.0, 3.0, 4.0, 11.0, 12.0, 7.0, 8.0, 15.0, 16.0,]
    );
    assert_eq!(
        matrices[1].row_trees,
        vec![
            (
                z2_pair([0, 0], [1, 1], 0).codomain_tree().clone(),
                0,
                vec![2, 1],
            ),
            (
                z2_pair([1, 1], [1, 1], 0).codomain_tree().clone(),
                2,
                vec![1, 2],
            ),
        ]
    );
    assert_eq!(
        matrices[1].col_trees,
        vec![
            (
                z2_pair([0, 0], [1, 1], 0).domain_tree().clone(),
                0,
                vec![1, 2],
            ),
            (
                z2_pair([0, 0], [0, 0], 0).domain_tree().clone(),
                2,
                vec![2, 1],
            ),
        ]
    );
}

#[test]
fn generic_sector_matricizations_preserve_full_tree_identity_and_exact_layout() {
    let structure = BlockStructure::from_blocks_with_rank(
        4,
        vec![
            BlockSpec::with_key(
                generic_pair(1, 2, 1).into(),
                vec![1, 2, 2, 1],
                vec![1, 17, 3, 40],
                5,
            )
            .unwrap(),
            BlockSpec::with_key(
                generic_pair(0, 1, 1).into(),
                vec![1, 1, 1, 1],
                vec![1, 7, 5, 3],
                80,
            )
            .unwrap(),
            BlockSpec::with_key(
                generic_pair(1, 1, 2).into(),
                vec![1, 1, 1, 3],
                vec![1, 8, 2, 4],
                60,
            )
            .unwrap(),
            BlockSpec::with_key(
                generic_pair(1, 1, 1).into(),
                vec![1, 1, 2, 1],
                vec![1, 9, 5, 30],
                40,
            )
            .unwrap(),
        ],
    )
    .unwrap();
    let mut real = vec![0.0; structure.required_len().unwrap()];
    for (index, value) in [
        (5, 11.0),
        (22, 12.0),
        (8, 13.0),
        (25, 14.0),
        (80, 90.0),
        (60, 31.0),
        (64, 32.0),
        (68, 33.0),
        (40, 21.0),
        (45, 22.0),
    ] {
        real[index] = value;
    }
    let expected = [
        11.0, 12.0, 21.0, 13.0, 14.0, 22.0, 0.0, 0.0, 31.0, 0.0, 0.0, 32.0, 0.0, 0.0, 33.0,
    ];

    let matrices = sector_matricizations_generic(&structure, &real, 2).unwrap();
    assert_eq!(
        matrices
            .iter()
            .map(|matrix| matrix.sector)
            .collect::<Vec<_>>(),
        [SectorId::new(1), SectorId::new(0)]
    );
    assert_eq!((matrices[0].rows, matrices[0].cols), (3, 5));
    assert_eq!(matrices[0].data, expected);
    assert_eq!(matrices[1].data, [90.0]);
    assert_eq!(
        matrices[0]
            .row_trees
            .iter()
            .map(|(tree, offset, shape)| (tree.vertices()[0].get(), *offset, shape.clone()))
            .collect::<Vec<_>>(),
        [(2, 0, vec![1, 2]), (1, 2, vec![1, 1])]
    );
    assert_eq!(
        matrices[0]
            .col_trees
            .iter()
            .map(|(tree, offset, shape)| (tree.vertices()[0].get(), *offset, shape.clone()))
            .collect::<Vec<_>>(),
        [(1, 0, vec![2, 1]), (2, 2, vec![1, 3])]
    );

    let complex = real
        .iter()
        .map(|&value| Complex64::new(value, -value / 10.0))
        .collect::<Vec<_>>();
    let complex_matrices = sector_matricizations_generic(&structure, &complex, 2).unwrap();
    assert_eq!(
        complex_matrices[0].data,
        expected.map(|value| Complex64::new(value, -value / 10.0))
    );
    assert_eq!(complex_matrices[1].data, [Complex64::new(90.0, -9.0)]);

    let empty = BlockStructure::from_blocks_with_rank(4, Vec::new()).unwrap();
    assert!(sector_matricizations_generic::<f64>(&empty, &[], 2)
        .unwrap()
        .is_empty());

    let scalar_key =
        FusionTreePairKey::try_pair_from_sector_ids([], [], 0, [], [], [], [], [], []).unwrap();
    let scalar = BlockStructure::from_blocks_with_rank(
        0,
        vec![BlockSpec::with_key(scalar_key.into(), vec![], vec![], 1).unwrap()],
    )
    .unwrap();
    assert_eq!(
        sector_matricizations_generic(&scalar, &[0.0, 7.0], 0).unwrap()[0].data,
        [7.0]
    );

    let zero_extent = BlockStructure::from_blocks_with_rank(
        4,
        vec![BlockSpec::with_key(
            generic_pair(1, 1, 1).into(),
            vec![0, 1, 1, 1],
            vec![1, 1, 1, 1],
            0,
        )
        .unwrap()],
    )
    .unwrap();
    let zero_matrix = sector_matricizations_generic::<f64>(&zero_extent, &[], 2).unwrap();
    assert_eq!((zero_matrix[0].rows, zero_matrix[0].cols), (0, 1));
    assert!(zero_matrix[0].data.is_empty());

    let non_fusion = BlockStructure::from_blocks_with_rank(
        1,
        vec![BlockSpec::with_key(BlockKey::opaque([7]), vec![1], vec![1], 0).unwrap()],
    )
    .unwrap();
    assert!(matches!(
        sector_matricizations_generic::<f64>(&non_fusion, &[1.0], 1),
        Err(OperationError::ExpectedFusionTreeBlock {
            tensor: "tsvd",
            index: 0
        })
    ));
}

#[test]
fn generic_sector_matricizations_keep_tree_offsets_matrix_local() {
    // Raw block structures can pair trees whose coupled sectors differ. A
    // domain tree shared by two codomain-selected matrices therefore has
    // an independent column offset in each matrix.
    let sector_one = generic_pair(1, 1, 1);
    let sector_zero = generic_pair(0, 1, 1);
    let row_one = sector_one.codomain_tree().clone();
    let row_zero = sector_zero.codomain_tree().clone();
    let shared_domain = sector_one.domain_tree().clone();
    let other_domain = sector_zero.domain_tree().clone();
    let structure = BlockStructure::from_blocks_with_rank(
        4,
        vec![
            BlockSpec::with_key(
                FusionTreePairKey::pair(row_one.clone(), other_domain.clone()).into(),
                vec![1, 1, 1, 1],
                vec![1, 1, 1, 1],
                1,
            )
            .unwrap(),
            BlockSpec::with_key(
                FusionTreePairKey::pair(row_zero.clone(), shared_domain.clone()).into(),
                vec![1, 1, 1, 2],
                vec![1, 1, 1, 1],
                4,
            )
            .unwrap(),
            BlockSpec::with_key(
                FusionTreePairKey::pair(row_one.clone(), shared_domain.clone()).into(),
                vec![1, 1, 1, 2],
                vec![1, 1, 1, 1],
                8,
            )
            .unwrap(),
        ],
    )
    .unwrap();
    let mut data = vec![0.0; structure.required_len().unwrap()];
    data[1] = 11.0;
    data[4..6].copy_from_slice(&[21.0, 22.0]);
    data[8..10].copy_from_slice(&[31.0, 32.0]);

    let matrices = sector_matricizations_generic(&structure, &data, 2).unwrap();

    assert_eq!(matrices.len(), 2);
    assert_eq!(matrices[0].sector, SectorId::new(1));
    assert_eq!((matrices[0].rows, matrices[0].cols), (1, 3));
    assert_eq!(matrices[0].data, [11.0, 31.0, 32.0]);
    assert_eq!(
        matrices[0]
            .col_trees
            .iter()
            .map(|(tree, offset, _)| (tree, *offset))
            .collect::<Vec<_>>(),
        [(&other_domain, 0), (&shared_domain, 1)]
    );
    assert_eq!(matrices[1].sector, SectorId::new(0));
    assert_eq!((matrices[1].rows, matrices[1].cols), (1, 2));
    assert_eq!(matrices[1].data, [21.0, 22.0]);
    assert_eq!(matrices[1].col_trees[0].0, shared_domain);
    assert_eq!(matrices[1].col_trees[0].1, 0);
}

fn z2_single_sector_matrix(
    rows: usize,
    cols: usize,
) -> (FusionTreeHomSpace, SectorMatricization<f64>) {
    let even = SectorId::new(0);
    let homspace = FusionTreeHomSpace::new(
        FusionProductSpace::new([SectorLeg::new([(even, rows)], false)]),
        FusionProductSpace::new([SectorLeg::new([(even, cols)], false)]),
    );
    let key = homspace.fusion_tree_keys_generic(&TestGenericRule).unwrap()[0].clone();
    (
        homspace,
        SectorMatricization {
            sector: even,
            rows,
            cols,
            row_trees: vec![(key.codomain_tree().clone(), 0, vec![rows])],
            col_trees: vec![(key.domain_tree().clone(), 0, vec![cols])],
            data: vec![0.0; rows * cols],
        },
    )
}

#[test]
fn mf_one_sided_placement_uses_direct_and_adjoint_source_sides() {
    let (homspace, mut matrix) = z2_single_sector_matrix(2, 3);
    matrix.rows = 3;
    matrix.row_trees[0].1 = 1;
    matrix.cols = 5;
    matrix.col_trees[0].1 = 2;
    matrix.data = Vec::new();
    let authority = BoundDynamicFusionMapSpace::from_final_homspace_multiplicity_free(
        Arc::new(Z2FusionRule),
        homspace.clone(),
    )
    .unwrap();
    let adjoint = tenet_tensors::adjoint_bound_space_dyn(&authority).unwrap();
    let values = |len: usize, base: f64| {
        (0..len)
            .map(|index| Complex64::new(base + index as f64, -base - index as f64 / 10.0))
            .collect::<Vec<_>>()
    };

    let cases = [
        (
            &authority,
            authority.space().homspace(),
            FactorSide::Left,
            FactorPlacement::Direct,
            2,
            FactorPair {
                sector: SectorId::new(0),
                kept: 99,
                left: values(6, 10.0),
                left_rows: 3,
                right: Vec::new(),
                right_leading: 0,
            },
            vec![1, 2, 4, 5],
        ),
        (
            &authority,
            authority.space().homspace(),
            FactorSide::Right,
            FactorPlacement::Direct,
            3,
            FactorPair {
                sector: SectorId::new(0),
                kept: 99,
                left: Vec::new(),
                left_rows: 0,
                right: values(15, 20.0),
                right_leading: 3,
            },
            (6..15).collect(),
        ),
        (
            &adjoint,
            adjoint.space().homspace(),
            FactorSide::Left,
            FactorPlacement::Adjoint,
            4,
            FactorPair {
                sector: SectorId::new(0),
                kept: 99,
                left: values(20, 30.0),
                left_rows: 5,
                right: Vec::new(),
                right_leading: 0,
            },
            vec![2, 3, 4, 7, 8, 9, 12, 13, 14, 17, 18, 19],
        ),
        (
            &adjoint,
            adjoint.space().homspace(),
            FactorSide::Right,
            FactorPlacement::Adjoint,
            2,
            FactorPair {
                sector: SectorId::new(0),
                kept: 99,
                left: Vec::new(),
                left_rows: 0,
                right: values(6, 40.0),
                right_leading: 2,
            },
            (2..6).collect(),
        ),
    ];

    for (authority, output_hom, side, placement, bond, mut pair, selected) in cases {
        let source = match side {
            FactorSide::Left => &pair.left,
            FactorSide::Right => &pair.right,
        };
        let expected = selected
            .into_iter()
            .map(|index| source[index])
            .collect::<Vec<_>>();
        let factor = build_bound_factor_with_placement(
            authority,
            output_hom,
            std::slice::from_ref(&matrix),
            std::slice::from_mut(&mut pair),
            &BTreeMap::from([(SectorId::new(0), bond)]),
            side,
            placement,
        )
        .unwrap();
        assert_eq!(factor.data(), expected);
    }
}

#[test]
fn mf_one_sided_pair_errors_precede_tree_traversal() {
    let (homspace, mut matrix) = z2_single_sector_matrix(1, 1);
    let authority = BoundDynamicFusionMapSpace::from_final_homspace_multiplicity_free(
        Arc::new(Z2FusionRule),
        homspace.clone(),
    )
    .unwrap();
    matrix.row_trees.clear();
    matrix.col_trees.clear();
    let dimensions = BTreeMap::from([(SectorId::new(0), 1)]);
    let pair = |sector| FactorPair {
        sector,
        kept: 1,
        left: vec![1.0],
        left_rows: 1,
        right: vec![2.0],
        right_leading: 1,
    };

    reset_placement_index_probe();
    let error = build_bound_factor_with_placement(
        &authority,
        &homspace,
        std::slice::from_ref(&matrix),
        &mut [pair(SectorId::new(0)), pair(SectorId::new(1))],
        &dimensions,
        FactorSide::Left,
        FactorPlacement::Direct,
    )
    .unwrap_err();
    assert!(matches!(
        error,
        OperationError::UnsupportedTensorContractScope {
            message: "factor sector absent from the source tensor"
        }
    ));
    assert_eq!(placement_index_probe(), PlacementIndexProbe::default());

    let error = build_bound_factor_with_placement::<_, f64, _>(
        &authority,
        &homspace,
        std::slice::from_ref(&matrix),
        &mut [],
        &dimensions,
        FactorSide::Right,
        FactorPlacement::Direct,
    )
    .unwrap_err();
    assert!(matches!(
        error,
        OperationError::UnsupportedTensorContractScope {
            message: "factor rank absent for a populated source sector"
        }
    ));
}

#[test]
fn generic_pair_publication_falls_back_for_padded_staged_geometry() {
    let (homspace, matrix) = z2_single_sector_matrix(2, 1);
    let provider = Arc::new(TestGenericRule);
    reset_generic_pair_publication_probe();
    reset_scatter_visit_probe();

    let (left, right) = build_left_right_bound_pair_generic(
        &provider,
        &homspace,
        &[matrix],
        vec![FactorPair {
            sector: SectorId::new(0),
            kept: 1,
            left: vec![2.0, 3.0],
            left_rows: 2,
            right: vec![5.0, 99.0],
            right_leading: 2,
        }],
    )
    .unwrap();

    assert_eq!(left.data(), [2.0, 3.0]);
    assert_eq!(right.data(), [5.0]);
    // G_s = 1 still pays the grouping pass (B = 1 per side) on top of the
    // F = 1 scattered block; disclosed, not dispatched on.
    assert_eq!(
        scatter_visit_probe(),
        ScatterVisitProbe {
            left_grouped: 1,
            right_grouped: 1,
            left_groups_built: 1,
            right_groups_built: 1,
            left_visits: 1,
            right_visits: 1,
        }
    );
    let probe = generic_pair_publication_probe();
    assert_eq!(
        (probe.canonical_publications, probe.fallback_publications),
        (0, 1)
    );
    assert_eq!(
        (
            probe.left_scattered_elements,
            probe.right_scattered_elements
        ),
        (2, 1)
    );
}

fn vertex_tree_factor_fixture(
    reverse: bool,
) -> (
    FusionTreeHomSpace,
    SectorMatricization<Complex64>,
    FactorPair<Complex64>,
) {
    let x = SectorId::new(1);
    let homspace = FusionTreeHomSpace::new(
        FusionProductSpace::new([
            SectorLeg::new([(x, 2)], false),
            SectorLeg::new([(x, 1)], false),
        ]),
        FusionProductSpace::new([
            SectorLeg::new([(x, 1)], false),
            SectorLeg::new([(x, 3)], false),
        ]),
    );
    let keys = homspace.fusion_tree_keys_generic(&TestGenericRule).unwrap();
    let mut row_trees = Vec::new();
    let mut col_trees = Vec::new();
    for key in keys
        .iter()
        .filter(|key| coupled_of_generic(key.codomain_tree()) == x)
    {
        if !row_trees.contains(key.codomain_tree()) {
            row_trees.push(key.codomain_tree().clone());
        }
        if !col_trees.contains(key.domain_tree()) {
            col_trees.push(key.domain_tree().clone());
        }
    }
    assert_eq!(row_trees.len(), 2);
    assert_eq!(col_trees.len(), 2);
    assert_eq!(
        row_trees
            .iter()
            .map(|tree| tree.vertices()[0].get())
            .collect::<Vec<_>>(),
        [1, 2]
    );
    assert_eq!(
        col_trees
            .iter()
            .map(|tree| tree.vertices()[0].get())
            .collect::<Vec<_>>(),
        [1, 2]
    );
    if reverse {
        row_trees.reverse();
        col_trees.reverse();
    }
    let row_trees = row_trees
        .into_iter()
        .enumerate()
        .map(|(index, tree)| (tree, 2 * index, vec![2, 1]))
        .collect();
    let col_trees = col_trees
        .into_iter()
        .enumerate()
        .map(|(index, tree)| (tree, 3 * index, vec![1, 3]))
        .collect();
    let left = (0..8)
        .map(|index| Complex64::new(10.0 + index as f64, -1.0 - index as f64 / 4.0))
        .collect::<Vec<_>>();
    let right = (0..12)
        .map(|index| Complex64::new(30.0 + index as f64, 2.0 + index as f64 / 3.0))
        .collect::<Vec<_>>();
    (
        homspace,
        SectorMatricization {
            sector: x,
            rows: 4,
            cols: 6,
            row_trees,
            col_trees,
            data: vec![Complex64::new(0.0, 0.0); 24],
        },
        FactorPair {
            sector: x,
            kept: 2,
            left,
            left_rows: 4,
            right,
            right_leading: 2,
        },
    )
}

#[test]
fn one_sided_row_placement_uses_the_first_matching_duplicate() {
    let a = generic_pair(1, 1, 1).codomain_tree().clone();
    let b = generic_pair(1, 2, 1).codomain_tree().clone();
    let c = generic_pair(0, 1, 1).codomain_tree().clone();
    let matrix = SectorMatricization {
        sector: SectorId::new(1),
        rows: 7,
        cols: 0,
        row_trees: vec![
            (a.clone(), 0, vec![2]),
            (b.clone(), 2, vec![2]),
            (a.clone(), 4, vec![2]),
            (c.clone(), 6, vec![1]),
        ],
        col_trees: Vec::new(),
        data: Vec::<f64>::new(),
    };
    let index = PlacementIndex::new(std::slice::from_ref(&matrix), &[FactorSide::Left]);
    let offsets = [&b, &c, &a].map(|tree| {
        index
            .placement(SectorId::new(1), FactorSide::Left, tree)
            .unwrap()
            .0
    });

    assert_eq!(offsets, [2, 6, 0]);
}

#[test]
fn placement_index_distinguishes_dual_innerline_and_vertex_trees() {
    let base = full_identity_pair(7, false, 11, false);
    let dual = full_identity_pair(7, true, 11, true);
    let inner = full_identity_pair(8, false, 12, false);
    // Same sectors, duals and inner lines as `base`; only the vertices differ.
    let vertex = FusionTreePairKey::try_pair_from_sector_ids(
        [1, 2, 3],
        [4, 5, 6],
        9,
        [false, false, false],
        [true, false, false],
        [7],
        [11],
        [2, 2],
        [1, 1],
    )
    .unwrap();
    let (row_vertex, col_vertex) = (vertex.codomain_tree().clone(), vertex.domain_tree().clone());
    let matrix = SectorMatricization {
        sector: SectorId::new(9),
        rows: 4,
        cols: 4,
        row_trees: vec![
            (base.codomain_tree().clone(), 0, vec![1]),
            (dual.codomain_tree().clone(), 1, vec![1]),
            (inner.codomain_tree().clone(), 2, vec![1]),
            (row_vertex.clone(), 3, vec![1]),
        ],
        col_trees: vec![
            (base.domain_tree().clone(), 0, vec![1]),
            (dual.domain_tree().clone(), 1, vec![1]),
            (inner.domain_tree().clone(), 2, vec![1]),
            (col_vertex.clone(), 3, vec![1]),
        ],
        data: Vec::<f64>::new(),
    };
    let index = PlacementIndex::new(
        std::slice::from_ref(&matrix),
        &[FactorSide::Left, FactorSide::Right],
    );
    let sector = SectorId::new(9);
    assert_eq!(
        [
            base.codomain_tree(),
            dual.codomain_tree(),
            inner.codomain_tree(),
            &row_vertex
        ]
        .map(|tree| index.placement(sector, FactorSide::Left, tree).unwrap().0),
        [0, 1, 2, 3]
    );
    assert_eq!(
        [
            base.domain_tree(),
            dual.domain_tree(),
            inner.domain_tree(),
            &col_vertex
        ]
        .map(|tree| index.placement(sector, FactorSide::Right, tree).unwrap().0),
        [0, 1, 2, 3]
    );
    // A domain tree is never a codomain tree of this matrix and vice versa.
    assert!(matches!(
        index.placement(sector, FactorSide::Left, base.domain_tree()),
        Err(OperationError::UnsupportedTensorContractScope {
            message: "factor codomain tree absent from the source matricization"
        })
    ));
    assert!(matches!(
        index.placement(sector, FactorSide::Right, base.codomain_tree()),
        Err(OperationError::UnsupportedTensorContractScope {
            message: "factor domain tree absent from the source matricization"
        })
    ));
}

#[test]
fn mf_one_sided_duplicate_source_trees_publish_the_first_entry() {
    fn run<D: FactorScalar + fmt::Debug>(values: impl Fn(usize) -> D) {
        let (homspace, mut matrix) = z2_single_sector_matrix(2, 3);
        let authority = BoundDynamicFusionMapSpace::from_final_homspace_multiplicity_free(
            Arc::new(Z2FusionRule),
            homspace.clone(),
        )
        .unwrap();
        let matrix = {
            let row = matrix.row_trees[0].0.clone();
            let col = matrix.col_trees[0].0.clone();
            matrix.rows = 4;
            matrix.row_trees = vec![(row.clone(), 0, vec![2]), (row, 2, vec![2])];
            matrix.cols = 6;
            matrix.col_trees = vec![(col.clone(), 0, vec![3]), (col, 3, vec![3])];
            SectorMatricization {
                sector: matrix.sector,
                rows: matrix.rows,
                cols: matrix.cols,
                row_trees: matrix.row_trees,
                col_trees: matrix.col_trees,
                data: Vec::<D>::new(),
            }
        };
        let source = (0..12).map(&values).collect::<Vec<_>>();
        let cases = [
            (FactorSide::Left, 2, 4, vec![0, 1, 4, 5], vec![2, 3, 6, 7]),
            (FactorSide::Right, 2, 2, (0..6).collect(), (6..12).collect()),
        ];
        for (side, bond, leading, first, second) in cases {
            let mut pair = match side {
                FactorSide::Left => FactorPair {
                    sector: SectorId::new(0),
                    kept: 99,
                    left: source.clone(),
                    left_rows: leading,
                    right: Vec::new(),
                    right_leading: 0,
                },
                FactorSide::Right => FactorPair {
                    sector: SectorId::new(0),
                    kept: 99,
                    left: Vec::new(),
                    left_rows: 0,
                    right: source.clone(),
                    right_leading: leading,
                },
            };
            reset_one_sided_publication_probe();
            reset_placement_index_probe();
            let factor = build_bound_factor_with_placement(
                &authority,
                &homspace,
                std::slice::from_ref(&matrix),
                std::slice::from_mut(&mut pair),
                &BTreeMap::from([(SectorId::new(0), bond)]),
                side,
                FactorPlacement::Direct,
            )
            .unwrap();
            let pick = |indices: &[usize]| indices.iter().map(|&i| source[i]).collect::<Vec<_>>();
            assert_eq!(factor.data(), pick(&first));
            assert_ne!(pick(&first), pick(&second));
            assert_eq!(one_sided_publication_probe().fallback_publications, 1);
            assert_eq!(
                placement_index_probe(),
                PlacementIndexProbe {
                    index_builds: 1,
                    indexed_sides: 1,
                    indexed_trees: 2,
                    lookups: 1,
                }
            );
        }
    }
    run(|k| k as f64 + 0.5);
    run(|k| Complex64::new(k as f64 + 0.5, -(k as f64)));
}

/// Z_8 with every leg carrying all eight charges: every coupled sector has
/// eight row trees and eight column trees, so F_c = T_c = 8 per sector.
fn z8_many_tree_legs() -> [Vec<(SectorId, usize)>; 4] {
    let all = (0..8).map(|s| (SectorId::new(s), 1)).collect::<Vec<_>>();
    [all.clone(), all.clone(), all.clone(), all]
}

#[test]
fn mf_one_sided_many_tree_fallback_indexes_each_populated_sector_once() {
    fn run<D: FactorScalar + fmt::Debug>(values: impl Fn(usize) -> D) {
        let fresh = || {
            two_leg_geometry::<_, D>(
                Arc::new(tenet_core::ZNFusionRule::new(8).unwrap()),
                z8_many_tree_legs(),
            )
        };
        let (authority, matrices) = fresh();
        assert_eq!(matrices.len(), 8);
        assert!(matrices
            .iter()
            .all(|m| m.row_trees.len() == 8 && m.col_trees.len() == 8));
        for (side, placement) in [
            (FactorSide::Left, FactorPlacement::Direct),
            (FactorSide::Right, FactorPlacement::Direct),
        ] {
            let source_trees = source_trees_for(side, placement);
            let dimensions = matrices
                .iter()
                .map(|m| (m.sector, source_extent(m, source_trees)))
                .collect::<BTreeMap<_, _>>();
            let build = |matrices: &[SectorMatricization<D>], pairs: &mut [FactorPair<D>]| {
                reset_one_sided_publication_probe();
                reset_placement_index_probe();
                let factor = build_bound_factor_with_placement(
                    &authority,
                    authority.space().homspace(),
                    matrices,
                    pairs,
                    &dimensions,
                    side,
                    placement,
                )
                .unwrap();
                (
                    factor.data().to_vec(),
                    one_sided_publication_probe(),
                    placement_index_probe(),
                )
            };
            let staged = |matrices: &[SectorMatricization<D>]| {
                staged_one_sided_pairs(matrices, &dimensions, side, source_trees, &values)
            };

            let mut pairs = staged(&matrices);
            let (reference, probe, index) = build(&matrices, &mut pairs);
            assert_eq!(probe.canonical_publications, 1);
            assert_eq!(index, PlacementIndexProbe::default());

            // All eight sectors populated, reversed: F = T = 64 over
            // G_s = 8 in one table (formerly one per sector), so 128
            // hashes replace the former 8 * 8 * 8 = 512 key comparisons.
            let (_, mut reversed) = fresh();
            reversed.reverse();
            let mut pairs = staged(&reversed);
            let (data, probe, index) = build(&reversed, &mut pairs);
            assert_eq!(probe.fallback_publications, 1);
            assert_eq!(data, reference);
            assert_eq!(
                index,
                PlacementIndexProbe {
                    index_builds: 1,
                    indexed_sides: 8,
                    indexed_trees: 64,
                    lookups: 64,
                }
            );
            assert!(index.indexed_trees + index.lookups < 8 * 8 * 8);

            // Identity-only sectors 0 (before), 3 (between) and 7 (after)
            // publish canonically without any placement index.
            let (_, kept) = fresh();
            let kept = kept
                .into_iter()
                .filter(|m| ![0, 3, 7].contains(&m.sector.id()))
                .collect::<Vec<_>>();
            let mut pairs = staged(&kept);
            let selected = pairs
                .iter()
                .map(|pair| selected_of(pair, side).clone())
                .collect::<Vec<_>>();
            let (data, probe, index) = build(&kept, &mut pairs);
            assert_eq!(
                (probe.canonical_publications, probe.fallback_publications),
                (1, 0)
            );
            assert_eq!(index, PlacementIndexProbe::default());
            let mut start = 0usize;
            for matrix in &matrices {
                let len = source_extent(matrix, source_trees) * dimensions[&matrix.sector];
                if ![0, 3, 7].contains(&matrix.sector.id()) {
                    assert_eq!(&data[start..start + len], &reference[start..start + len]);
                }
                start += len;
            }
            assert_eq!(start, data.len());
            let (authority, _) = fresh();
            let factor_space = build_bound_factor_space(
                &authority,
                authority.space().homspace(),
                SectorLeg::new(dimensions.iter().map(|(&s, &d)| (s, d)), false),
                side,
            )
            .unwrap();
            assert_literal_one_sided_layout(
                factor_space.space().structure(),
                &data,
                &kept,
                &selected,
                &dimensions,
                side,
                source_trees,
            );
        }
    }
    run(|k| k as f64 + 0.25);
    run(|k| Complex64::new(k as f64 + 0.25, 0.5 - k as f64));
}

/// Three `{0, 1}` legs per side under the multiplicity-two test rule:
/// coupled sector 1 has 14 trees per side and sector 0 has 6, counting
/// inner lines and vertices, so F_c = T_c per sector (unit degeneracies).
#[test]
fn checked_one_sided_many_tree_publication_reuses_one_index_per_sector() {
    fn run<D: FactorScalar + fmt::Debug>(values: impl Fn(usize) -> D) {
        let rule = TestGenericRule;
        let provider = Arc::new(InfallibleGeneric::new(&rule));
        let leg = || SectorLeg::new([(SectorId::new(0), 1), (SectorId::new(1), 1)], false);
        let homspace = FusionTreeHomSpace::new(
            FusionProductSpace::new([leg(), leg(), leg()]),
            FusionProductSpace::new([leg(), leg(), leg()]),
        );
        let space = BoundDynamicFusionMapSpace::from_final_homspace_generic_checked(
            Arc::clone(&provider),
            homspace.clone(),
        )
        .unwrap();
        let zeros = vec![D::zero(); space.space().required_len().unwrap()];
        let fresh = || sector_matricizations(space.space().structure(), &zeros, 3).unwrap();
        let matrices = fresh();
        assert_eq!(
            matrices
                .iter()
                .map(|m| (m.sector.id(), m.row_trees.len(), m.col_trees.len()))
                .collect::<Vec<_>>(),
            [(0, 6, 6), (1, 14, 14)]
        );
        for side in [FactorSide::Left, FactorSide::Right] {
            let dimensions = matrices
                .iter()
                .map(|m| (m.sector, source_extent(m, side)))
                .collect::<BTreeMap<_, _>>();
            let build = |matrices: &[SectorMatricization<D>], pairs: &mut [FactorPair<D>]| {
                reset_one_sided_publication_probe();
                reset_placement_index_probe();
                let factor = build_bound_factor_generic_checked(
                    &provider,
                    &homspace,
                    matrices,
                    pairs,
                    &dimensions,
                    side,
                )
                .unwrap();
                (
                    factor.data().to_vec(),
                    one_sided_publication_probe(),
                    placement_index_probe(),
                )
            };
            let staged = |matrices: &[SectorMatricization<D>]| {
                staged_one_sided_pairs(matrices, &dimensions, side, side, &values)
            };

            // Canonical transfer: prevalidation alone, one lookup per key,
            // and no plan allocation for the all-populated input.
            let mut pairs = staged(&matrices);
            let (reference, probe, index) = build(&matrices, &mut pairs);
            assert_eq!(probe.canonical_publications, 1);
            assert_eq!(probe.plan_bytes, 0);
            assert_eq!(
                index,
                PlacementIndexProbe {
                    index_builds: 1,
                    indexed_sides: 2,
                    indexed_trees: 20,
                    lookups: 20,
                }
            );

            // Fallback: the single prevalidation table is reused (formerly
            // one table per sector, G_s = 2), so lookups double without a
            // rebuild; 60 hashes replace the former
            // 2 * (6 * 6 + 14 * 14) = 464 key comparisons.
            let mut reversed = fresh();
            reversed.reverse();
            let mut pairs = staged(&reversed);
            let (data, probe, index) = build(&reversed, &mut pairs);
            assert_eq!(probe.fallback_publications, 1);
            assert_eq!(data, reference);
            assert_eq!(
                index,
                PlacementIndexProbe {
                    index_builds: 1,
                    indexed_sides: 2,
                    indexed_trees: 20,
                    lookups: 40,
                }
            );
            assert!(index.indexed_trees + index.lookups < 2 * (6 * 6 + 14 * 14));

            // Identity-only sector after (keep 0) and before (keep 1) the
            // populated sector: only the populated sector is indexed.
            let lens = matrices
                .iter()
                .map(|m| source_extent(m, side) * dimensions[&m.sector])
                .collect::<Vec<_>>();
            for keep in [0usize, 1] {
                let kept = fresh()
                    .into_iter()
                    .filter(|m| m.sector.id() == keep)
                    .collect::<Vec<_>>();
                let mut pairs = staged(&kept);
                let selected = pairs
                    .iter()
                    .map(|pair| selected_of(pair, side).clone())
                    .collect::<Vec<_>>();
                let (data, probe, index) = build(&kept, &mut pairs);
                assert_eq!(
                    (probe.canonical_publications, probe.fallback_publications),
                    (1, 0)
                );
                assert_eq!(
                    index,
                    PlacementIndexProbe {
                        index_builds: 1,
                        indexed_sides: 1,
                        indexed_trees: kept[0].row_trees.len(),
                        lookups: kept[0].row_trees.len(),
                    }
                );
                let (first, second) = data.split_at(lens[0]);
                let (reference_first, reference_second) = reference.split_at(lens[0]);
                assert_eq!(second.len(), lens[1]);
                if keep == 0 {
                    assert_eq!(first, reference_first);
                } else {
                    assert_eq!(second, reference_second);
                }
                let checked_factor = build_bound_factor_generic_checked(
                    &provider,
                    &homspace,
                    &kept,
                    &mut staged(&kept),
                    &dimensions,
                    side,
                )
                .unwrap();
                assert_literal_one_sided_layout(
                    checked_factor.space().space().structure(),
                    &data,
                    &kept,
                    &selected,
                    &dimensions,
                    side,
                    side,
                );
            }
        }
    }
    run(|k| k as f64 + 0.25);
    run(|k| Complex64::new(k as f64 + 0.25, 0.5 - k as f64));
}

#[test]
fn generic_pair_fallback_validation_indexes_each_sector_once() {
    let (sector_count, trees_per_sector) = (4usize, 8usize);
    let mut matrices = Vec::new();
    let mut left_keys = Vec::new();
    let mut right_keys = Vec::new();
    for sector in 0..sector_count {
        let bond = generic_pair(sector, 1, 1).codomain_tree().clone();
        let mut row_trees = Vec::new();
        let mut col_trees = Vec::new();
        for tree_index in 0..trees_per_sector {
            let source = generic_pair(sector, tree_index + 1, tree_index + 1);
            left_keys.push(FusionTreePairKey::pair(
                source.codomain_tree().clone(),
                bond.clone(),
            ));
            right_keys.push(FusionTreePairKey::pair(
                bond.clone(),
                source.domain_tree().clone(),
            ));
            row_trees.push((source.codomain_tree().clone(), tree_index, vec![1, 1]));
            col_trees.push((source.domain_tree().clone(), tree_index, vec![1, 1]));
        }
        matrices.push(SectorMatricization {
            sector: SectorId::new(sector),
            rows: trees_per_sector,
            cols: trees_per_sector,
            row_trees,
            col_trees,
            data: Vec::<f64>::new(),
        });
    }
    // Interleave the key order across sectors so each sector's index is
    // reused across non-adjacent keys.
    left_keys.reverse();
    right_keys.reverse();
    let total = sector_count * trees_per_sector;
    for (side, keys) in [
        (FactorSide::Left, &left_keys),
        (FactorSide::Right, &right_keys),
    ] {
        reset_generic_pair_publication_probe();
        reset_placement_index_probe();
        let mut matrix_by_sector = None;
        assert!(
            validate_generic_factor_keys(keys, side, None, &matrices, &mut matrix_by_sector)
                .unwrap()
        );
        let probe = generic_pair_publication_probe();
        let lookups = match side {
            FactorSide::Left => probe.fallback_row_lookups,
            FactorSide::Right => probe.fallback_col_lookups,
        };
        assert_eq!(lookups, total);
        let index = placement_index_probe();
        assert_eq!(
            index,
            PlacementIndexProbe {
                index_builds: 1,
                indexed_sides: sector_count,
                indexed_trees: total,
                lookups: total,
            }
        );
        // F + T = 64 hashes versus the former F * T_c = 32 * 8 = 256
        // comparisons.
        assert!(index.indexed_trees + index.lookups < total * trees_per_sector);
    }
}

#[test]
fn checked_one_sided_placement_preserves_both_sides_and_tree_orders() {
    let rule = TestGenericRule;
    let provider = Arc::new(InfallibleGeneric::new(&rule));
    let dimensions = BTreeMap::from([(SectorId::new(1), 2)]);

    for reverse in [false, true] {
        let (homspace, matrix, pair) = vertex_tree_factor_fixture(reverse);
        for side in [FactorSide::Left, FactorSide::Right] {
            let (_, _, mut staged) = vertex_tree_factor_fixture(reverse);
            let factor = build_bound_factor_generic_checked(
                &provider,
                &homspace,
                std::slice::from_ref(&matrix),
                std::slice::from_mut(&mut staged),
                &dimensions,
                side,
            )
            .unwrap();
            let expected = match (side, reverse) {
                (FactorSide::Left, false) => pair.left.clone(),
                (FactorSide::Right, false) => pair.right.clone(),
                (FactorSide::Left, true) => vec![
                    pair.left[2],
                    pair.left[3],
                    pair.left[0],
                    pair.left[1],
                    pair.left[6],
                    pair.left[7],
                    pair.left[4],
                    pair.left[5],
                ],
                (FactorSide::Right, true) => vec![
                    pair.right[6],
                    pair.right[7],
                    pair.right[8],
                    pair.right[9],
                    pair.right[10],
                    pair.right[11],
                    pair.right[0],
                    pair.right[1],
                    pair.right[2],
                    pair.right[3],
                    pair.right[4],
                    pair.right[5],
                ],
            };
            assert_eq!(factor.data(), expected);
            assert!(Arc::ptr_eq(factor.space().provider_arc(), &provider));
            let selected_vertices = (0..factor.space().space().structure().block_count())
                .map(|index| factor.space().space().structure().block(index).unwrap())
                .map(|block| match block.key() {
                    BlockKey::FusionTree(key) => match side {
                        FactorSide::Left => key.codomain_tree().vertices()[0].get(),
                        FactorSide::Right => key.domain_tree().vertices()[0].get(),
                    },
                    _ => unreachable!("checked factor has fusion-tree keys"),
                })
                .collect::<Vec<_>>();
            assert_eq!(selected_vertices, [1, 2]);
        }
    }
}

#[test]
fn checked_one_sided_reports_missing_pairs_and_full_trees() {
    let rule = TestGenericRule;
    let provider = Arc::new(InfallibleGeneric::new(&rule));
    let (homspace, matrix, _) = vertex_tree_factor_fixture(false);
    let dimensions = BTreeMap::from([(SectorId::new(1), 2)]);
    for side in [FactorSide::Left, FactorSide::Right] {
        reset_placement_index_probe();
        let error = build_bound_factor_generic_checked::<_, Complex64, _>(
            &provider,
            &homspace,
            std::slice::from_ref(&matrix),
            &mut [],
            &dimensions,
            side,
        )
        .unwrap_err();
        assert!(matches!(
            error,
            CheckedGenericFactorPlanError::Operation(
                OperationError::UnsupportedTensorContractScope {
                    message: "factor rank absent for a populated source sector"
                }
            )
        ));
        // Prevalidation indexed the sole sector and looked up both keys;
        // the fallback rejected the missing pair before its own lookup.
        assert_eq!(
            placement_index_probe(),
            PlacementIndexProbe {
                index_builds: 1,
                indexed_sides: 1,
                indexed_trees: 2,
                lookups: 2,
            }
        );
    }

    for side in [FactorSide::Left, FactorSide::Right] {
        let (_, mut wrong, mut pair) = vertex_tree_factor_fixture(false);
        match side {
            FactorSide::Left => wrong.row_trees[0].0 = wrong.row_trees[1].0.clone(),
            FactorSide::Right => wrong.col_trees[0].0 = wrong.col_trees[1].0.clone(),
        }
        let error = build_bound_factor_generic_checked(
            &provider,
            &homspace,
            std::slice::from_ref(&wrong),
            std::slice::from_mut(&mut pair),
            &dimensions,
            side,
        )
        .unwrap_err();
        let expected = match side {
            FactorSide::Left => "factor codomain tree absent from the source matricization",
            FactorSide::Right => "factor domain tree absent from the source matricization",
        };
        assert!(matches!(
            error,
            CheckedGenericFactorPlanError::Operation(
                OperationError::UnsupportedTensorContractScope { message }
            ) if message == expected
        ));
    }
}

#[test]
fn generic_pair_publication_preserves_vertex_tree_payload_placement() {
    let provider = Arc::new(TestGenericRule);
    let (homspace, matrix, pair) = vertex_tree_factor_fixture(false);
    let expected_left = pair.left.clone();
    let expected_right = pair.right.clone();
    reset_generic_pair_publication_probe();
    let (left, right) =
        build_left_right_bound_pair_generic(&provider, &homspace, &[matrix], vec![pair]).unwrap();
    assert_eq!(left.data(), expected_left);
    assert_eq!(right.data(), expected_right);
    let probe = generic_pair_publication_probe();
    assert_eq!(
        (probe.canonical_publications, probe.fallback_publications),
        (1, 0)
    );
    assert_eq!(
        (probe.fallback_row_lookups, probe.fallback_col_lookups),
        (0, 0)
    );
    assert_eq!(
        (probe.left_scatter_calls, probe.right_scatter_calls),
        (0, 0)
    );
    assert_eq!(probe.output_blocks_visited, 4);
    assert_eq!(
        probe.ordered_key_validation_events,
        2 * probe.output_blocks_visited
    );

    let (homspace, matrix, pair) = vertex_tree_factor_fixture(true);
    let left_source = pair.left.clone();
    let right_source = pair.right.clone();
    reset_generic_pair_publication_probe();
    reset_placement_index_probe();
    let (left, right) =
        build_left_right_bound_pair_generic(&provider, &homspace, &[matrix], vec![pair]).unwrap();
    assert_eq!(
        left.data(),
        [
            left_source[2],
            left_source[3],
            left_source[0],
            left_source[1],
            left_source[6],
            left_source[7],
            left_source[4],
            left_source[5],
        ]
    );
    assert_eq!(
        right.data(),
        [
            right_source[6],
            right_source[7],
            right_source[8],
            right_source[9],
            right_source[10],
            right_source[11],
            right_source[0],
            right_source[1],
            right_source[2],
            right_source[3],
            right_source[4],
            right_source[5],
        ]
    );
    let probe = generic_pair_publication_probe();
    assert_eq!(
        (probe.canonical_publications, probe.fallback_publications),
        (0, 1)
    );
    assert_eq!(
        (probe.fallback_row_lookups, probe.fallback_col_lookups),
        (2, 2)
    );
    assert_eq!(
        (probe.left_scatter_calls, probe.right_scatter_calls),
        (2, 2)
    );
    // Row and column key validation each build a one-sided table; the
    // scatter loop shares one two-sided table (formerly four tables).
    assert_eq!(
        placement_index_probe(),
        PlacementIndexProbe {
            index_builds: 3,
            indexed_sides: 4,
            indexed_trees: 8,
            lookups: 8,
        }
    );
}

#[test]
fn generic_pair_canonical_validation_scales_linearly_in_sectors_and_trees() {
    for (sector_count, trees_per_sector) in [(1, 1), (1, 4), (4, 1), (4, 3)] {
        let mut matrices = Vec::with_capacity(sector_count);
        let mut pairs = Vec::with_capacity(sector_count);
        let mut left_keys = Vec::with_capacity(sector_count * trees_per_sector);
        let mut right_keys = Vec::with_capacity(sector_count * trees_per_sector);
        let mut left_blocks = Vec::with_capacity(sector_count * trees_per_sector);
        let mut right_blocks = Vec::with_capacity(sector_count * trees_per_sector);
        let mut output_offset = 0usize;
        for sector in 0..sector_count {
            let bond = FusionTreePairKey::try_pair_from_sector_ids(
                [sector],
                [sector],
                sector,
                [false],
                [false],
                [],
                [],
                [],
                [],
            )
            .unwrap()
            .codomain_tree()
            .clone();
            let mut row_trees = Vec::with_capacity(trees_per_sector);
            let mut col_trees = Vec::with_capacity(trees_per_sector);
            for tree_index in 0..trees_per_sector {
                let source = generic_pair(sector, tree_index + 1, tree_index + 1);
                let row_tree = source.codomain_tree().clone();
                let col_tree = source.domain_tree().clone();
                let left_key = FusionTreePairKey::pair(row_tree.clone(), bond.clone());
                let right_key = FusionTreePairKey::pair(bond.clone(), col_tree.clone());
                left_blocks.push(
                    BlockSpec::with_key(
                        left_key.clone().into(),
                        vec![1, 1, 1],
                        vec![1, 1, trees_per_sector],
                        output_offset + tree_index,
                    )
                    .unwrap(),
                );
                right_blocks.push(
                    BlockSpec::with_key(
                        right_key.clone().into(),
                        vec![1, 1, 1],
                        vec![1, 1, 1],
                        output_offset + tree_index,
                    )
                    .unwrap(),
                );
                left_keys.push(left_key);
                right_keys.push(right_key);
                row_trees.push((row_tree, tree_index, vec![1, 1]));
                col_trees.push((col_tree, tree_index, vec![1, 1]));
            }
            let sector = SectorId::new(sector);
            matrices.push(SectorMatricization {
                sector,
                rows: trees_per_sector,
                cols: trees_per_sector,
                row_trees,
                col_trees,
                data: vec![0.0; trees_per_sector * trees_per_sector],
            });
            pairs.push(FactorPair {
                sector,
                kept: 1,
                left: (0..trees_per_sector)
                    .map(|index| 100.0 * sector.id() as f64 + index as f64)
                    .collect(),
                left_rows: trees_per_sector,
                right: (0..trees_per_sector)
                    .map(|index| -100.0 * sector.id() as f64 - index as f64)
                    .collect(),
                right_leading: 1,
            });
            output_offset += trees_per_sector;
        }
        let ranks = pairs
            .iter()
            .map(|pair| SectorRank {
                sector: pair.sector,
                kept: pair.kept,
            })
            .collect::<Vec<_>>();
        let left_structure = BlockStructure::from_blocks_with_rank(3, left_blocks).unwrap();
        let right_structure = BlockStructure::from_blocks_with_rank(3, right_blocks).unwrap();
        let expected_left = pairs
            .iter()
            .flat_map(|pair| pair.left.iter().copied())
            .collect::<Vec<_>>();
        let expected_right = pairs
            .iter()
            .flat_map(|pair| pair.right.iter().copied())
            .collect::<Vec<_>>();

        reset_generic_pair_publication_probe();
        let mut matrix_by_sector = None;
        let mut left_cursor = FactorTreeCursor::new(&matrices, &ranks);
        assert!(validate_generic_factor_keys(
            &left_keys,
            FactorSide::Left,
            Some(&mut left_cursor),
            &matrices,
            &mut matrix_by_sector,
        )
        .unwrap());
        let mut right_cursor = FactorTreeCursor::new(&matrices, &ranks);
        assert!(validate_generic_factor_keys(
            &right_keys,
            FactorSide::Right,
            Some(&mut right_cursor),
            &matrices,
            &mut matrix_by_sector,
        )
        .unwrap());
        assert!(factor_output_is_canonical(
            &left_structure,
            Some(&left_keys),
            &matrices,
            &pairs,
            expected_left.len(),
            FactorSide::Left,
        ));
        assert!(factor_output_is_canonical(
            &right_structure,
            Some(&right_keys),
            &matrices,
            &pairs,
            expected_right.len(),
            FactorSide::Right,
        ));
        let output_blocks = left_keys.len() + right_keys.len();
        let probe = generic_pair_publication_probe();
        assert_eq!(probe.output_blocks_visited, output_blocks);
        assert_eq!(probe.ordered_key_validation_events, 2 * output_blocks);
        assert_eq!(
            (probe.fallback_row_lookups, probe.fallback_col_lookups),
            (0, 0)
        );
        let (left, right) =
            publish_generic_factor_pairs(pairs, expected_left.len(), expected_right.len());
        assert_eq!(left, expected_left);
        assert_eq!(right, expected_right);
    }
}

#[test]
fn generic_pair_validation_preserves_missing_sector_and_tree_errors() {
    let (homspace, matrix) = z2_single_sector_matrix(1, 1);
    let provider = Arc::new(TestGenericRule);
    let pair = || FactorPair {
        sector: SectorId::new(0),
        kept: 1,
        left: vec![1.0],
        left_rows: 1,
        right: vec![1.0],
        right_leading: 1,
    };

    let missing_sector = build_left_right_bound_pair_generic(
        &provider,
        &homspace,
        &[] as &[SectorMatricization<f64>],
        vec![pair()],
    )
    .unwrap_err();
    assert!(matches!(
        missing_sector,
        OperationError::UnsupportedTensorContractScope {
            message: "factor tree references a coupled sector absent from the source tensor"
        }
    ));

    let wrong =
        FusionTreePairKey::try_pair_from_sector_ids([1], [0], 0, [false], [false], [], [], [], [])
            .unwrap()
            .codomain_tree()
            .clone();
    let mut wrong_row = matrix;
    wrong_row.row_trees[0].0 = wrong.clone();
    wrong_row.col_trees[0].0 = wrong.clone();
    let missing_row =
        build_left_right_bound_pair_generic(&provider, &homspace, &[wrong_row], vec![pair()])
            .unwrap_err();
    assert!(matches!(
        missing_row,
        OperationError::UnsupportedTensorContractScope {
            message: "factor codomain tree absent from the source matricization"
        }
    ));

    let (_, mut wrong_col) = z2_single_sector_matrix(1, 1);
    wrong_col.col_trees[0].0 = wrong;
    let missing_col =
        build_left_right_bound_pair_generic(&provider, &homspace, &[wrong_col], vec![pair()])
            .unwrap_err();
    assert!(matches!(
        missing_col,
        OperationError::UnsupportedTensorContractScope {
            message: "factor domain tree absent from the source matricization"
        }
    ));
}

#[test]
fn generic_pair_publication_handles_zero_kept_sector() {
    let (homspace, matrix) = z2_single_sector_matrix(2, 1);
    reset_generic_pair_publication_probe();
    let (left, right) = build_left_right_bound_pair_generic(
        &Arc::new(TestGenericRule),
        &homspace,
        &[matrix],
        vec![FactorPair {
            sector: SectorId::new(0),
            kept: 0,
            left: Vec::<f64>::new(),
            left_rows: 2,
            right: Vec::new(),
            right_leading: 0,
        }],
    )
    .unwrap();
    assert!(left.data().is_empty());
    assert!(right.data().is_empty());
    assert_eq!(
        generic_pair_publication_probe(),
        GenericPairPublicationProbe {
            canonical_publications: 1,
            ..GenericPairPublicationProbe::default()
        }
    );

    let empty = FusionProductSpace::new(std::iter::empty::<SectorLeg>());
    let empty_hom = FusionTreeHomSpace::new(empty.clone(), empty);
    let (left, right) = build_left_right_bound_pair_generic::<_, f64, SectorMatricization<f64>>(
        &Arc::new(TestGenericRule),
        &empty_hom,
        &[],
        Vec::new(),
    )
    .unwrap();
    assert!(left.data().is_empty());
    assert!(right.data().is_empty());
}

#[test]
fn generic_pair_publication_handles_scalar_matrix() {
    let empty = FusionProductSpace::new(std::iter::empty::<SectorLeg>());
    let homspace = FusionTreeHomSpace::new(empty.clone(), empty);
    let key = homspace.fusion_tree_keys_generic(&TestGenericRule).unwrap()[0].clone();
    reset_generic_pair_publication_probe();
    let (left, right) = build_left_right_bound_pair_generic(
        &Arc::new(TestGenericRule),
        &homspace,
        &[SectorMatricization {
            sector: SectorId::new(0),
            rows: 1,
            cols: 1,
            row_trees: vec![(key.codomain_tree().clone(), 0, Vec::new())],
            col_trees: vec![(key.domain_tree().clone(), 0, Vec::new())],
            data: vec![7.0],
        }],
        vec![FactorPair {
            sector: SectorId::new(0),
            kept: 1,
            left: vec![2.0],
            left_rows: 1,
            right: vec![3.5],
            right_leading: 1,
        }],
    )
    .unwrap();
    assert_eq!(left.data(), [2.0]);
    assert_eq!(right.data(), [3.5]);
    let probe = generic_pair_publication_probe();
    assert_eq!(probe.canonical_publications, 1);
    assert_eq!((probe.left_owner_reused, probe.right_owner_reused), (1, 1));
}

/// Two-leg codomain/domain hom space matricized exactly as production
/// does, so tree orders and offsets are the real ones.
fn two_leg_geometry<R, D>(
    rule: Arc<R>,
    legs: [Vec<(SectorId, usize)>; 4],
) -> (BoundDynamicFusionMapSpace<R>, Vec<SectorMatricization<D>>)
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64>,
    D: FactorScalar,
{
    let [a, b, c, d] = legs;
    let homspace = FusionTreeHomSpace::new(
        FusionProductSpace::new([SectorLeg::new(a, false), SectorLeg::new(b, false)]),
        FusionProductSpace::new([SectorLeg::new(c, false), SectorLeg::new(d, false)]),
    );
    let authority =
        BoundDynamicFusionMapSpace::from_final_homspace_multiplicity_free(rule, homspace).unwrap();
    let data = vec![D::zero(); authority.space().required_len().unwrap()];
    let matrices = sector_matricizations(authority.space().structure(), &data, 2).unwrap();
    (authority, matrices)
}

fn z2_two_sector_geometry<D: FactorScalar>() -> (
    BoundDynamicFusionMapSpace<Z2FusionRule>,
    Vec<SectorMatricization<D>>,
) {
    let even = SectorId::new(0);
    let odd = SectorId::new(1);
    two_leg_geometry(
        Arc::new(Z2FusionRule),
        [
            vec![(even, 2), (odd, 1)],
            vec![(even, 1), (odd, 3)],
            vec![(even, 1), (odd, 2)],
            vec![(even, 2), (odd, 1)],
        ],
    )
}

fn source_trees_for(side: FactorSide, placement: FactorPlacement) -> FactorSide {
    match (side, placement) {
        (FactorSide::Left, FactorPlacement::Direct)
        | (FactorSide::Right, FactorPlacement::Adjoint) => FactorSide::Left,
        (FactorSide::Right, FactorPlacement::Direct)
        | (FactorSide::Left, FactorPlacement::Adjoint) => FactorSide::Right,
    }
}

fn source_extent<D>(matrix: &SectorMatricization<D>, source_trees: FactorSide) -> usize {
    match source_trees {
        FactorSide::Left => matrix.rows,
        FactorSide::Right => matrix.cols,
    }
}

/// Stages `a x b` (left) or `b x a` (right) selected factors per sector
/// with `kept` deliberately unrelated to the admitted bond; the opposite
/// side carries a sentinel payload that must survive publication.
fn staged_one_sided_pairs<D: FactorScalar>(
    matrices: &[SectorMatricization<D>],
    dimensions: &BTreeMap<SectorId, usize>,
    side: FactorSide,
    source_trees: FactorSide,
    values: &dyn Fn(usize) -> D,
) -> Vec<FactorPair<D>> {
    matrices
        .iter()
        .map(|matrix| {
            let a = source_extent(matrix, source_trees);
            let b = dimensions[&matrix.sector];
            let tag = matrix.sector.id() + 1;
            let selected = (0..a * b)
                .map(|k| values(1000 * tag + k))
                .collect::<Vec<_>>();
            let opposite = (0..3).map(|k| values(7 * tag + k)).collect();
            match side {
                FactorSide::Left => FactorPair {
                    sector: matrix.sector,
                    kept: 99,
                    left: selected,
                    left_rows: a,
                    right: opposite,
                    right_leading: 5,
                },
                FactorSide::Right => FactorPair {
                    sector: matrix.sector,
                    kept: 99,
                    left: opposite,
                    left_rows: 5,
                    right: selected,
                    right_leading: b,
                },
            }
        })
        .collect()
}

fn selected_of<D>(pair: &FactorPair<D>, side: FactorSide) -> &Vec<D> {
    match side {
        FactorSide::Left => &pair.left,
        FactorSide::Right => &pair.right,
    }
}

fn opposite_of<D>(pair: &FactorPair<D>, side: FactorSide) -> &Vec<D> {
    match side {
        FactorSide::Left => &pair.right,
        FactorSide::Right => &pair.left,
    }
}

/// Checks every output element against the literal coordinates
/// `F[o + q + a*j]` (left) / `F[j + b*(o + q)]` (right), independent of
/// the production layout proof. A sector without a matricization is
/// checked against a separately built `d x d` identity with `o` counted
/// over its earlier output blocks (the output tree order is its basis).
fn assert_literal_one_sided_layout<D>(
    structure: &BlockStructure,
    data: &[D],
    matrices: &[SectorMatricization<D>],
    selected: &[Vec<D>],
    dimensions: &BTreeMap<SectorId, usize>,
    side: FactorSide,
    source_trees: FactorSide,
) where
    D: FactorScalar + PartialEq + fmt::Debug,
{
    assert_literal_one_sided_blocks(
        structure,
        data,
        matrices,
        Some(selected),
        dimensions,
        side,
        source_trees,
    );
}

/// [`assert_literal_one_sided_layout`] that skips populated sectors when
/// `selected` is `None` (numerical factors), keeping the literal identity
/// check and the exact output coverage.
fn assert_literal_one_sided_blocks<D, M>(
    structure: &BlockStructure,
    data: &[D],
    matrices: &[SectorMatricization<M>],
    selected: Option<&[Vec<D>]>,
    dimensions: &BTreeMap<SectorId, usize>,
    side: FactorSide,
    source_trees: FactorSide,
) where
    D: FactorScalar + PartialEq + fmt::Debug,
{
    let mut verified = 0usize;
    let mut identity_offsets = BTreeMap::<SectorId, usize>::new();
    for index in 0..structure.block_count() {
        let block = structure.block(index).unwrap();
        let BlockKey::FusionTree(key) = block.key() else {
            panic!("factor blocks carry fusion-tree keys")
        };
        let tree = match side {
            FactorSide::Left => key.codomain_tree(),
            FactorSide::Right => key.domain_tree(),
        };
        let shape = block.shape();
        let strides = block.strides();
        let total = shape.iter().product::<usize>();
        let identity;
        let (o, a, b, factor) = if let Some((matrix_index, matrix)) = matrices
            .iter()
            .enumerate()
            .find(|(_, matrix)| matrix.sector == tree.coupled())
        {
            let Some(selected) = selected else {
                verified += total;
                continue;
            };
            let trees = match source_trees {
                FactorSide::Left => &matrix.row_trees,
                FactorSide::Right => &matrix.col_trees,
            };
            let (_, o, tree_shape) = trees
                .iter()
                .find(|(candidate, _, _)| candidate == tree)
                .unwrap();
            let b = dimensions[&matrix.sector];
            assert_eq!(total, tree_shape.iter().product::<usize>() * b);
            (
                *o,
                source_extent(matrix, source_trees),
                b,
                &selected[matrix_index],
            )
        } else {
            let d = dimensions[&tree.coupled()];
            let mut literal = vec![D::zero(); d * d];
            for diagonal in 0..d {
                literal[diagonal + d * diagonal] = D::one();
            }
            identity = literal;
            let o = identity_offsets.entry(tree.coupled()).or_default();
            let start = *o;
            *o += total / d;
            (start, d, d, &identity)
        };
        for flat in 0..total {
            let mut remaining = flat;
            let mut destination = block.offset();
            let mut coordinates = vec![0usize; shape.len()];
            for axis in 0..shape.len() {
                coordinates[axis] = remaining % shape[axis];
                remaining /= shape[axis];
                destination += coordinates[axis] * strides[axis];
            }
            let tree_axes = match side {
                FactorSide::Left => 0..shape.len() - 1,
                FactorSide::Right => 1..shape.len(),
            };
            let mut q = 0usize;
            let mut span = 1usize;
            for axis in tree_axes {
                q += coordinates[axis] * span;
                span *= shape[axis];
            }
            let j = match side {
                FactorSide::Left => coordinates[shape.len() - 1],
                FactorSide::Right => coordinates[0],
            };
            let expected = match side {
                FactorSide::Left => factor[o + q + a * j],
                FactorSide::Right => factor[j + b * (o + q)],
            };
            assert_eq!(data[destination], expected);
            verified += 1;
        }
    }
    assert_eq!(verified, data.len());
}

fn mf_one_sided_canonical_transfer_case<D>(values: &dyn Fn(usize) -> D)
where
    D: FactorScalar + PartialEq + fmt::Debug,
{
    let (authority, matrices) = z2_two_sector_geometry::<D>();
    let adjoint = tenet_tensors::adjoint_bound_space_dyn(&authority).unwrap();
    let row_dimensions = matrices
        .iter()
        .map(|matrix| (matrix.sector, matrix.rows))
        .collect::<BTreeMap<_, _>>();
    let col_dimensions = matrices
        .iter()
        .map(|matrix| (matrix.sector, matrix.cols))
        .collect::<BTreeMap<_, _>>();
    assert_eq!(matrices.len(), 2);
    assert!(matrices.iter().all(|matrix| matrix.rows != matrix.cols
        && matrix.row_trees.len() == 2
        && matrix.col_trees.len() == 2));
    assert_ne!(matrices[0].rows, matrices[1].rows);

    let cases = [
        (&authority, FactorSide::Left, FactorPlacement::Direct),
        (&authority, FactorSide::Right, FactorPlacement::Direct),
        (&adjoint, FactorSide::Left, FactorPlacement::Adjoint),
        (&adjoint, FactorSide::Right, FactorPlacement::Adjoint),
    ];
    for (space, side, placement) in cases {
        let source_trees = source_trees_for(side, placement);
        let dimensions = match source_trees {
            FactorSide::Left => &row_dimensions,
            FactorSide::Right => &col_dimensions,
        };
        let mut pairs = staged_one_sided_pairs(&matrices, dimensions, side, source_trees, values);
        let selected = pairs
            .iter()
            .map(|pair| selected_of(pair, side).clone())
            .collect::<Vec<_>>();
        let opposite = pairs
            .iter()
            .map(|pair| opposite_of(pair, side).clone())
            .collect::<Vec<_>>();

        reset_one_sided_publication_probe();
        let factor = build_bound_factor_with_placement(
            space,
            space.space().homspace(),
            &matrices,
            &mut pairs,
            dimensions,
            side,
            placement,
        )
        .unwrap();

        let probe = one_sided_publication_probe();
        assert_eq!(
            (probe.canonical_publications, probe.fallback_publications),
            (1, 0)
        );
        assert_eq!(probe.appended_elements, selected[1].len());
        // All-populated input allocates no plan, as the boolean proof did.
        assert_eq!(probe.plan_bytes, 0);
        assert_literal_one_sided_layout(
            factor.space().space().structure(),
            factor.data(),
            &matrices,
            &selected,
            dimensions,
            side,
            source_trees,
        );
        for (index, pair) in pairs.iter().enumerate() {
            assert!(selected_of(pair, side).is_empty());
            assert_eq!(opposite_of(pair, side), &opposite[index]);
            assert_eq!((pair.kept, pair.left_rows, pair.right_leading).0, 99);
        }
    }
}

#[test]
fn mf_one_sided_canonical_transfer_real_full_svd_bonds() {
    mf_one_sided_canonical_transfer_case(&|k| k as f64 * 0.5 - 7.0);
}

#[test]
fn mf_one_sided_canonical_transfer_complex_full_svd_bonds() {
    mf_one_sided_canonical_transfer_case(&|k| Complex64::new(k as f64, -(k as f64) / 3.0));
}

fn assert_buffers_intact<D>(pairs: &[FactorPair<D>], side: FactorSide) {
    assert!(pairs.iter().all(|pair| !selected_of(pair, side).is_empty()));
}

#[test]
fn mf_one_sided_fallbacks_match_canonical_output_and_keep_buffers() {
    let (authority, matrices) = z2_two_sector_geometry::<f64>();
    let homspace = authority.space().homspace();
    let side = FactorSide::Left;
    let placement = FactorPlacement::Direct;
    let row_dimensions = matrices
        .iter()
        .map(|matrix| (matrix.sector, matrix.rows))
        .collect::<BTreeMap<_, _>>();
    let staged = |scale: f64| {
        staged_one_sided_pairs(&matrices, &row_dimensions, side, side, &|k| {
            scale * (k as f64 + 1.0)
        })
    };
    let build = |matrices: &[SectorMatricization<f64>],
                 pairs: &mut [FactorPair<f64>],
                 dimensions: &BTreeMap<SectorId, usize>| {
        reset_one_sided_publication_probe();
        let result = build_bound_factor_with_placement(
            &authority, homspace, matrices, pairs, dimensions, side, placement,
        );
        (result, one_sided_publication_probe())
    };
    let canonical = |scale: f64| {
        let mut pairs = staged(scale);
        let (factor, probe) = build(&matrices, &mut pairs, &row_dimensions);
        assert_eq!(
            (probe.canonical_publications, probe.fallback_publications),
            (1, 0)
        );
        factor.unwrap().data().to_vec()
    };
    let reference = canonical(1.0);
    let sector_len = matrices[0].rows * matrices[0].rows;

    // Reordered matricizations: same admitted output, scatter path.
    let (_, mut reversed) = z2_two_sector_geometry::<f64>();
    reversed.reverse();
    let mut pairs = staged(1.0);
    pairs.reverse();
    let (factor, probe) = build(&reversed, &mut pairs, &row_dimensions);
    assert_eq!(
        (probe.canonical_publications, probe.fallback_publications),
        (0, 1)
    );
    assert_eq!(factor.unwrap().data(), reference);
    assert_buffers_intact(&pairs, side);

    // Reordered pairs alone.
    let mut pairs = staged(1.0);
    pairs.reverse();
    let (factor, probe) = build(&matrices, &mut pairs, &row_dimensions);
    assert_eq!(probe.fallback_publications, 1);
    assert_eq!(factor.unwrap().data(), reference);
    assert_buffers_intact(&pairs, side);

    // Padded source geometry: the even factor carries one extra leading
    // row that no tree covers, so tree offsets no longer start at zero.
    let (_, mut padded) = z2_two_sector_geometry::<f64>();
    padded[0].rows += 1;
    for tree in &mut padded[0].row_trees {
        tree.1 += 1;
    }
    let mut pairs =
        staged_one_sided_pairs(&padded, &row_dimensions, side, side, &|k| k as f64 + 1.0);
    let selected = pairs
        .iter()
        .map(|pair| pair.left.clone())
        .collect::<Vec<_>>();
    let (factor, probe) = build(&padded, &mut pairs, &row_dimensions);
    assert_eq!(probe.fallback_publications, 1);
    let factor = factor.unwrap();
    assert_eq!(factor.data().len(), reference.len());
    assert_literal_one_sided_layout(
        factor.space().space().structure(),
        factor.data(),
        &padded,
        &selected,
        &row_dimensions,
        side,
        side,
    );
    assert_buffers_intact(&pairs, side);

    // Identity-only sector after (drop odd) and before (drop even) the
    // populated sector publishes canonically: the populated region equals
    // the canonical one and the identity matches the literal oracle.
    for kept in [0..1, 1..2] {
        let mut pairs = staged(1.0);
        let selected = pairs[kept.clone()]
            .iter()
            .map(|pair| pair.left.clone())
            .collect::<Vec<_>>();
        let (factor, probe) = build(
            &matrices[kept.clone()],
            &mut pairs[kept.clone()],
            &row_dimensions,
        );
        assert_eq!(
            (probe.canonical_publications, probe.fallback_publications),
            (1, 0)
        );
        let factor = factor.unwrap();
        let populated = if kept.start == 0 {
            0..sector_len
        } else {
            sector_len..reference.len()
        };
        assert_eq!(&factor.data()[populated.clone()], &reference[populated]);
        assert_literal_one_sided_layout(
            factor.space().space().structure(),
            factor.data(),
            &matrices[kept],
            &selected,
            &row_dimensions,
            side,
            side,
        );
    }

    // Missing pair for a populated sector: unchanged error.
    let mut pairs = staged(1.0);
    let (result, probe) = build(&matrices, &mut pairs[..1], &row_dimensions);
    assert!(matches!(
        result.unwrap_err(),
        OperationError::UnsupportedTensorContractScope {
            message: "factor rank absent for a populated source sector"
        }
    ));
    assert_eq!(probe.fallback_publications, 1);
    assert_buffers_intact(&pairs, side);

    // Extraneous pair: unchanged error before any publication decision.
    let mut pairs = staged(1.0);
    pairs.push(FactorPair {
        sector: SectorId::new(5),
        kept: 1,
        left: vec![1.0],
        left_rows: 1,
        right: Vec::new(),
        right_leading: 0,
    });
    let (result, probe) = build(&matrices, &mut pairs, &row_dimensions);
    assert!(matches!(
        result.unwrap_err(),
        OperationError::UnsupportedTensorContractScope {
            message: "factor sector absent from the source tensor"
        }
    ));
    assert_eq!(
        (probe.canonical_publications, probe.fallback_publications),
        (0, 0)
    );
    assert_buffers_intact(&pairs, side);

    // Duplicate pair records for the even sector with different payloads:
    // the last record wins, on the scatter path.
    let alternate = canonical(2.0);
    let mut pairs = staged(1.0);
    let mut duplicate = staged(2.0);
    pairs.push(duplicate.swap_remove(0));
    let (factor, probe) = build(&matrices, &mut pairs, &row_dimensions);
    assert_eq!(probe.fallback_publications, 1);
    let factor = factor.unwrap();
    assert_eq!(&factor.data()[..sector_len], &alternate[..sector_len]);
    assert_eq!(&factor.data()[sector_len..], &reference[sector_len..]);
    assert_ne!(&alternate[..sector_len], &reference[..sector_len]);
    assert_buffers_intact(&pairs, side);
}

#[test]
fn mf_one_sided_identity_sector_between_populated_sectors_publishes_canonically() {
    let rule = Arc::new(tenet_core::ZNFusionRule::new(3).unwrap());
    let sectors = [SectorId::new(0), SectorId::new(1), SectorId::new(2)];
    let [s0, s1, s2] = sectors;
    let (authority, matrices) = two_leg_geometry::<_, f64>(
        rule,
        [
            vec![(s0, 1), (s1, 2), (s2, 1)],
            vec![(s0, 2), (s1, 1), (s2, 1)],
            vec![(s0, 1), (s1, 1), (s2, 2)],
            vec![(s0, 2), (s1, 1), (s2, 1)],
        ],
    );
    assert_eq!(
        matrices
            .iter()
            .map(|matrix| matrix.sector)
            .collect::<Vec<_>>(),
        sectors
    );
    let side = FactorSide::Right;
    let col_dimensions = matrices
        .iter()
        .map(|matrix| (matrix.sector, matrix.cols))
        .collect::<BTreeMap<_, _>>();
    let build = |matrices: &[SectorMatricization<f64>], pairs: &mut [FactorPair<f64>]| {
        reset_one_sided_publication_probe();
        let factor = build_bound_factor_with_placement(
            &authority,
            authority.space().homspace(),
            matrices,
            pairs,
            &col_dimensions,
            side,
            FactorPlacement::Direct,
        )
        .unwrap();
        (factor.data().to_vec(), one_sided_publication_probe())
    };
    let mut pairs =
        staged_one_sided_pairs(&matrices, &col_dimensions, side, side, &|k| k as f64 + 0.25);
    let (reference, probe) = build(&matrices, &mut pairs);
    assert_eq!(
        (probe.canonical_publications, probe.fallback_publications),
        (1, 0)
    );

    let (_, mut without_middle) = two_leg_geometry::<_, f64>(
        Arc::new(tenet_core::ZNFusionRule::new(3).unwrap()),
        [
            vec![(s0, 1), (s1, 2), (s2, 1)],
            vec![(s0, 2), (s1, 1), (s2, 1)],
            vec![(s0, 1), (s1, 1), (s2, 2)],
            vec![(s0, 2), (s1, 1), (s2, 1)],
        ],
    );
    without_middle.remove(1);
    let mut pairs = staged_one_sided_pairs(&without_middle, &col_dimensions, side, side, &|k| {
        k as f64 + 0.25
    });
    let selected = pairs
        .iter()
        .map(|pair| pair.right.clone())
        .collect::<Vec<_>>();
    reset_one_sided_publication_probe();
    let factor = build_bound_factor_with_placement(
        &authority,
        authority.space().homspace(),
        &without_middle,
        &mut pairs,
        &col_dimensions,
        side,
        FactorPlacement::Direct,
    )
    .unwrap();
    let probe = one_sided_publication_probe();
    assert_eq!(
        (probe.canonical_publications, probe.fallback_publications),
        (1, 0)
    );
    let first = matrices[0].cols * matrices[0].cols;
    let middle = matrices[1].cols * matrices[1].cols;
    // The first factor owns the output; the identity is written in place
    // and the last factor appended.
    assert_eq!(probe.appended_elements, middle + selected[1].len());
    let data = factor.data();
    assert_eq!(&data[..first], &reference[..first]);
    assert_eq!(&data[first + middle..], &reference[first + middle..]);
    assert_literal_one_sided_layout(
        factor.space().space().structure(),
        data,
        &without_middle,
        &selected,
        &col_dimensions,
        side,
        side,
    );

    // Reordered records around the identity sector keep the tree-identity
    // scatter and publish the same output.
    let (_, mut reversed) = two_leg_geometry::<_, f64>(
        Arc::new(tenet_core::ZNFusionRule::new(3).unwrap()),
        [
            vec![(s0, 1), (s1, 2), (s2, 1)],
            vec![(s0, 2), (s1, 1), (s2, 1)],
            vec![(s0, 1), (s1, 1), (s2, 2)],
            vec![(s0, 2), (s1, 1), (s2, 1)],
        ],
    );
    reversed.remove(1);
    reversed.reverse();
    let mut pairs =
        staged_one_sided_pairs(&reversed, &col_dimensions, side, side, &|k| k as f64 + 0.25);
    let (scattered, probe) = build(&reversed, &mut pairs);
    assert_eq!(
        (probe.canonical_publications, probe.fallback_publications),
        (0, 1)
    );
    assert_buffers_intact(&pairs, side);
    assert_eq!(scattered, data);
}

/// U(1) MPS site `(left x phys) <- right` with a codomain-only coupled
/// charge 2 and a domain-only charge 3; charge 0 is square (full rank,
/// omitted from both null spaces) and charge 1 is `6 x 4`.
fn u1_mps_side_only_space() -> BoundDynamicFusionMapSpace<tenet_core::U1FusionRule> {
    let q = |charge| tenet_core::U1Irrep::new(charge).sector_id();
    let homspace = FusionTreeHomSpace::new(
        FusionProductSpace::new([
            SectorLeg::new([(q(0), 3), (q(1), 3)], false),
            SectorLeg::new([(q(0), 1), (q(1), 1)], false),
        ]),
        FusionProductSpace::new([SectorLeg::new([(q(0), 3), (q(1), 4), (q(3), 2)], false)]),
    );
    BoundDynamicFusionMapSpace::from_final_homspace_multiplicity_free(
        Arc::new(tenet_core::U1FusionRule),
        homspace,
    )
    .unwrap()
}

/// Byte-traffic contract for side-only publication: every full/null factor
/// of the MPS site is published canonically, and the only elements
/// written beyond the moved first owner are the appended factors and the
/// in-place identities. The scatter fallback instead zero-fills all `P`
/// output elements and then scatters every populated block.
fn side_only_publication_bytes_case<D>(values: impl Fn(usize) -> D)
where
    D: FactorScalar + PartialEq + fmt::Debug,
{
    let space = u1_mps_side_only_space();
    let data = (0..space.space().required_len().unwrap())
        .map(&values)
        .collect::<Vec<_>>();
    let input = BoundDynamicTensorRef::try_new(&space, &data).unwrap();
    let matrices = sector_matricizations(space.space().structure(), &data, 2).unwrap();
    assert_eq!(
        matrices
            .iter()
            .map(|matrix| (matrix.rows, matrix.cols))
            .collect::<Vec<_>>(),
        [(3, 3), (6, 4)]
    );
    let q = |charge| tenet_core::U1Irrep::new(charge).sector_id();
    let codomain = BTreeMap::from([(q(0), 3), (q(1), 6), (q(2), 3)]);
    let domain = BTreeMap::from([(q(0), 3), (q(1), 4), (q(3), 2)]);
    let mut dense = tenet_dense::DefaultDenseExecutor::new();
    let bytes = |elements: usize| elements * std::mem::size_of::<D>();
    let check = |factor: &BoundDynFactor<_, D>,
                 dimensions: &BTreeMap<SectorId, usize>,
                 side: FactorSide| {
        assert_literal_one_sided_blocks(
            factor.space().space().structure(),
            factor.data(),
            &matrices,
            None,
            dimensions,
            side,
            side,
        );
    };
    let publication = |expected_written: usize| {
        let probe = one_sided_publication_probe();
        reset_one_sided_publication_probe();
        assert_eq!(
            (probe.canonical_publications, probe.fallback_publications),
            (1, 0)
        );
        // An identity sector is the only case that allocates a plan.
        assert_ne!(probe.plan_bytes, 0);
        assert_eq!(bytes(probe.appended_elements), bytes(expected_written));
    };

    // Left null: charge 1 null basis 6 x 2 owns the buffer, charge 2 adds
    // a 3 x 3 identity, charge 0 is absent (P = 12 + 9, fallback 21 + 12 + 3).
    reset_one_sided_publication_probe();
    let null = left_null_dyn(&mut dense, &input).unwrap();
    publication(9);
    assert_eq!(null.data().len(), 21);
    check(
        &null,
        &BTreeMap::from([(q(1), 2), (q(2), 3)]),
        FactorSide::Left,
    );
    // Right null: charges 0 and 1 have full column rank and are absent;
    // the charge-3 identity (2 x 2) is written into a fresh buffer.
    let null = right_null_dyn(&mut dense, &input).unwrap();
    publication(4);
    assert_eq!(null.data().len(), 4);
    check(&null, &BTreeMap::from([(q(3), 2)]), FactorSide::Right);

    let Qr { q: left, r: right } = qr_full_dyn(&mut dense, &input).unwrap();
    let _ = one_sided_publication_probe();
    check(&left, &codomain, FactorSide::Left);
    check(&right, &codomain, FactorSide::Right);
    let Lq { l: left, q: right } = lq_full_dyn(&mut dense, &input).unwrap();
    check(&left, &domain, FactorSide::Left);
    check(&right, &domain, FactorSide::Right);
    let svd = svd_full_dyn(&mut dense, &input).unwrap();
    check(svd.u(), &codomain, FactorSide::Left);
    check(svd.vh(), &domain, FactorSide::Right);
    let probe = one_sided_publication_probe();
    assert_eq!(
        (probe.canonical_publications, probe.fallback_publications),
        (6, 0)
    );
    // Q: 36 + 9 identity; R: 24 (charge 2 has no domain block); L: 24;
    // Q': 16 + 4 identity; U: 36 + 9; Vh: 16 + 4 (first owners moved).
    assert_eq!(
        bytes(probe.appended_elements),
        bytes(45 + 24 + 24 + 20 + 45 + 20)
    );
}

#[test]
fn side_only_publication_writes_each_output_element_once_real() {
    side_only_publication_bytes_case(|k| ((k * k) as f64 * 0.37 + k as f64).sin());
}

#[test]
fn side_only_publication_writes_each_output_element_once_complex() {
    side_only_publication_bytes_case(|k| {
        Complex64::new(
            ((k * k) as f64 * 0.37 + k as f64).sin(),
            ((k * k) as f64 * 0.23 - k as f64).cos(),
        )
    });
}

#[test]
fn mf_one_sided_zero_bond_sector_between_populated_sectors() {
    // Pinned outcome (a): the admitted structure omits zero-extent blocks,
    // so a zero-bond sector contributes no blocks and an empty factor; the
    // populated neighbours still transfer canonically.
    let legs = |s0, s1, s2| {
        [
            vec![(s0, 1), (s1, 2), (s2, 1)],
            vec![(s0, 2), (s1, 1), (s2, 1)],
            vec![(s0, 1), (s1, 1), (s2, 2)],
            vec![(s0, 2), (s1, 1), (s2, 1)],
        ]
    };
    let [s0, s1, s2] = [SectorId::new(0), SectorId::new(1), SectorId::new(2)];
    let (authority, matrices) = two_leg_geometry::<_, f64>(
        Arc::new(tenet_core::ZNFusionRule::new(3).unwrap()),
        legs(s0, s1, s2),
    );
    assert_eq!(
        matrices
            .iter()
            .map(|matrix| matrix.sector)
            .collect::<Vec<_>>(),
        [s0, s1, s2]
    );
    let side = FactorSide::Right;
    let dimensions = BTreeMap::from([(s0, matrices[0].cols), (s1, 0), (s2, matrices[2].cols)]);
    let mut pairs = staged_one_sided_pairs(&matrices, &dimensions, side, side, &|k| k as f64 - 0.5);
    pairs[1].kept = 0;
    assert!(pairs[1].right.is_empty());
    let selected = pairs
        .iter()
        .map(|pair| pair.right.clone())
        .collect::<Vec<_>>();
    let opposite = pairs
        .iter()
        .map(|pair| pair.left.clone())
        .collect::<Vec<_>>();

    reset_one_sided_publication_probe();
    let factor = build_bound_factor_with_placement(
        &authority,
        authority.space().homspace(),
        &matrices,
        &mut pairs,
        &dimensions,
        side,
        FactorPlacement::Direct,
    )
    .unwrap();

    let probe = one_sided_publication_probe();
    assert_eq!(
        (probe.canonical_publications, probe.fallback_publications),
        (1, 0)
    );
    assert_eq!(probe.appended_elements, selected[2].len());
    assert_eq!(factor.data().len(), selected[0].len() + selected[2].len());
    assert_literal_one_sided_layout(
        factor.space().space().structure(),
        factor.data(),
        &matrices,
        &selected,
        &dimensions,
        side,
        side,
    );
    for (index, pair) in pairs.iter().enumerate() {
        assert!(pair.right.is_empty());
        assert_eq!(pair.left, opposite[index]);
    }
}

/// One even sector whose selected tree has zero legs: the factor block on
/// that side is rank-1 (`shape == [b]`). `padding` extra source rows or
/// columns precede the tree so the canonical proof declines and the
/// scatter fallback publishes the block.
fn zero_leg_side_matrix<D: FactorScalar>(
    zero_leg_side: FactorSide,
    extent: usize,
    padding: usize,
) -> (FusionTreeHomSpace, SectorMatricization<D>) {
    let even = SectorId::new(0);
    let leg = FusionProductSpace::new([SectorLeg::new([(even, extent)], false)]);
    let homspace = match zero_leg_side {
        FactorSide::Left => FusionTreeHomSpace::new(FusionProductSpace::new([]), leg),
        FactorSide::Right => FusionTreeHomSpace::new(leg, FusionProductSpace::new([])),
    };
    let key = homspace.fusion_tree_keys_generic(&TestGenericRule).unwrap()[0].clone();
    let (rows, cols, row_trees, col_trees) = match zero_leg_side {
        FactorSide::Left => (
            1 + padding,
            extent,
            vec![(key.codomain_tree().clone(), padding, vec![])],
            vec![(key.domain_tree().clone(), 0, vec![extent])],
        ),
        FactorSide::Right => (
            extent,
            1 + padding,
            vec![(key.codomain_tree().clone(), 0, vec![extent])],
            vec![(key.domain_tree().clone(), padding, vec![])],
        ),
    };
    (
        homspace,
        SectorMatricization {
            sector: even,
            rows,
            cols,
            row_trees,
            col_trees,
            data: vec![D::zero(); rows * cols],
        },
    )
}

/// Rank-1 blocks on both sides through the MF and the checked one-sided
/// owners (#1197). Right: `b x cols` column-major, the block at column
/// `o` is `F[j + b*o]`; Left: `a x b`, the block at row `o` is
/// `F[o + a*j]`. Before the fix the Right block was read with the Left
/// stride (`F[o + b*j]`), which for `b = 3, cols = 2, o = 1` reaches
/// `F[7]` past the six-element factor and panicked on the slice bound.
fn one_sided_rank1_fallback_case<D>(values: &dyn Fn(usize) -> D)
where
    D: FactorScalar + PartialEq + fmt::Debug,
{
    let even = SectorId::new(0);
    let b = 3usize;
    let padding = 1usize;
    let factor = (0..2 * b).map(values).collect::<Vec<_>>();
    let pair = |side| match side {
        FactorSide::Right => FactorPair {
            sector: even,
            kept: 99,
            left: vec![values(500)],
            left_rows: 1,
            right: factor.clone(),
            right_leading: b,
        },
        FactorSide::Left => FactorPair {
            sector: even,
            kept: 99,
            left: factor.clone(),
            left_rows: 1 + padding,
            right: vec![values(500)],
            right_leading: 1,
        },
    };
    let cases = [
        (
            FactorSide::Right,
            (0..b).map(|j| factor[j + b * padding]).collect::<Vec<_>>(),
        ),
        (
            FactorSide::Left,
            (0..b)
                .map(|j| factor[padding + (1 + padding) * j])
                .collect::<Vec<_>>(),
        ),
    ];
    for (side, expected) in cases {
        let (homspace, matrix) = zero_leg_side_matrix::<D>(side, 2, padding);
        let dimensions = BTreeMap::from([(even, b)]);
        let selected = factor.clone();

        let authority = BoundDynamicFusionMapSpace::from_final_homspace_multiplicity_free(
            Arc::new(Z2FusionRule),
            homspace.clone(),
        )
        .unwrap();
        let mut pairs = [pair(side)];
        reset_one_sided_publication_probe();
        let mf = build_bound_factor_with_placement(
            &authority,
            &homspace,
            std::slice::from_ref(&matrix),
            &mut pairs,
            &dimensions,
            side,
            FactorPlacement::Direct,
        )
        .unwrap();
        let probe = one_sided_publication_probe();
        assert_eq!(
            (probe.canonical_publications, probe.fallback_publications),
            (0, 1)
        );
        let block = mf.space().space().structure().block(0).unwrap();
        assert_eq!(block.shape(), [b]);
        assert_eq!(mf.data(), expected);
        assert_eq!(selected_of(&pairs[0], side), &selected);

        let provider = Arc::new(InfallibleGeneric::new(&TestGenericRule));
        let mut pairs = [pair(side)];
        reset_one_sided_publication_probe();
        let checked = build_bound_factor_generic_checked(
            &provider,
            &homspace,
            std::slice::from_ref(&matrix),
            &mut pairs,
            &dimensions,
            side,
        )
        .unwrap();
        let probe = one_sided_publication_probe();
        assert_eq!(
            (probe.canonical_publications, probe.fallback_publications),
            (0, 1)
        );
        assert_eq!(checked.data(), expected);
        assert_eq!(selected_of(&pairs[0], side), &selected);
    }
}

#[test]
fn one_sided_rank1_fallback_scatters_right_and_left_layouts_real() {
    one_sided_rank1_fallback_case(&|k| k as f64 + 0.5);
}

#[test]
fn one_sided_rank1_fallback_scatters_right_and_left_layouts_complex() {
    one_sided_rank1_fallback_case(&|k| Complex64::new(k as f64, -(k as f64) - 0.25));
}

#[test]
fn generic_pair_fallback_scatters_rank1_right_block_with_right_layout() {
    let b = 3usize;
    let padding = 1usize;
    let (homspace, matrix) = zero_leg_side_matrix::<f64>(FactorSide::Right, 2, padding);
    let provider = Arc::new(TestGenericRule);
    let left = (0..2 * b).map(|k| 10.0 + k as f64).collect::<Vec<_>>();
    let right = (0..b * (1 + padding))
        .map(|k| 20.0 + k as f64)
        .collect::<Vec<_>>();
    reset_generic_pair_publication_probe();

    let (left_factor, right_factor) = build_left_right_bound_pair_generic(
        &provider,
        &homspace,
        std::slice::from_ref(&matrix),
        vec![FactorPair {
            sector: SectorId::new(0),
            kept: b,
            left: left.clone(),
            left_rows: 2,
            right: right.clone(),
            right_leading: b,
        }],
    )
    .unwrap();

    let probe = generic_pair_publication_probe();
    assert_eq!(
        (probe.canonical_publications, probe.fallback_publications),
        (0, 1)
    );
    assert_eq!(left_factor.data(), left);
    let block = right_factor.space().space().structure().block(0).unwrap();
    assert_eq!(block.shape(), [b]);
    assert_eq!(
        right_factor.data(),
        (0..b).map(|j| right[j + b * padding]).collect::<Vec<_>>()
    );
}

#[test]
fn checked_one_sided_canonical_transfer_across_sectors_and_extra_pairs() {
    let rule = TestGenericRule;
    let provider = Arc::new(InfallibleGeneric::new(&rule));
    let even = SectorId::new(0);
    let odd = SectorId::new(1);
    let homspace = FusionTreeHomSpace::new(
        FusionProductSpace::new([
            SectorLeg::new([(even, 2), (odd, 1)], false),
            SectorLeg::new([(even, 1), (odd, 3)], false),
        ]),
        FusionProductSpace::new([
            SectorLeg::new([(even, 1), (odd, 2)], false),
            SectorLeg::new([(even, 2), (odd, 1)], false),
        ]),
    );
    let space = BoundDynamicFusionMapSpace::from_final_homspace_generic_checked(
        Arc::clone(&provider),
        homspace.clone(),
    )
    .unwrap();
    let zeros = vec![Complex64::new(0.0, 0.0); space.space().required_len().unwrap()];
    let matrices = sector_matricizations(space.space().structure(), &zeros, 2).unwrap();
    assert_eq!(matrices.len(), 2);
    let values = |k: usize| Complex64::new(0.5 * k as f64, 2.0 - k as f64);

    for side in [FactorSide::Left, FactorSide::Right] {
        let dimensions = matrices
            .iter()
            .map(|matrix| (matrix.sector, source_extent(matrix, side)))
            .collect::<BTreeMap<_, _>>();
        let mut pairs = staged_one_sided_pairs(&matrices, &dimensions, side, side, &values);
        let selected = pairs
            .iter()
            .map(|pair| selected_of(pair, side).clone())
            .collect::<Vec<_>>();
        let opposite = pairs
            .iter()
            .map(|pair| opposite_of(pair, side).clone())
            .collect::<Vec<_>>();
        reset_one_sided_publication_probe();
        let factor = build_bound_factor_generic_checked(
            &provider,
            &homspace,
            &matrices,
            &mut pairs,
            &dimensions,
            side,
        )
        .unwrap();
        let probe = one_sided_publication_probe();
        assert_eq!(
            (probe.canonical_publications, probe.fallback_publications),
            (1, 0)
        );
        assert_eq!(probe.appended_elements, selected[1].len());
        // All-populated input allocates no plan, as the boolean proof did.
        assert_eq!(probe.plan_bytes, 0);
        assert!(Arc::ptr_eq(factor.space().provider_arc(), &provider));
        assert_literal_one_sided_layout(
            factor.space().space().structure(),
            factor.data(),
            &matrices,
            &selected,
            &dimensions,
            side,
            side,
        );
        for (index, pair) in pairs.iter().enumerate() {
            assert!(selected_of(pair, side).is_empty());
            assert_eq!(opposite_of(pair, side), &opposite[index]);
        }

        // An extra unused pair is ignored on the scatter path, never an
        // error, and the output is identical.
        let mut pairs = staged_one_sided_pairs(&matrices, &dimensions, side, side, &values);
        pairs.push(FactorPair {
            sector: SectorId::new(7),
            kept: 1,
            left: vec![Complex64::new(1.0, 1.0)],
            left_rows: 1,
            right: vec![Complex64::new(1.0, 1.0)],
            right_leading: 1,
        });
        reset_one_sided_publication_probe();
        let fallback = build_bound_factor_generic_checked(
            &provider,
            &homspace,
            &matrices,
            &mut pairs,
            &dimensions,
            side,
        )
        .unwrap();
        let probe = one_sided_publication_probe();
        assert_eq!(
            (probe.canonical_publications, probe.fallback_publications),
            (0, 1)
        );
        assert_eq!(fallback.data(), factor.data());
        assert_buffers_intact(&pairs, side);
    }
}

#[test]
fn checked_one_sided_vertex_trees_transfer_sole_owner_or_fall_back() {
    let rule = TestGenericRule;
    let provider = Arc::new(InfallibleGeneric::new(&rule));
    let dimensions = BTreeMap::from([(SectorId::new(1), 2)]);
    for reverse in [false, true] {
        for side in [FactorSide::Left, FactorSide::Right] {
            let (homspace, matrix, mut pair) = vertex_tree_factor_fixture(reverse);
            let (_, _, reference) = vertex_tree_factor_fixture(reverse);
            let selected_ptr = selected_of(&pair, side).as_ptr();
            reset_one_sided_publication_probe();
            let factor = build_bound_factor_generic_checked(
                &provider,
                &homspace,
                std::slice::from_ref(&matrix),
                std::slice::from_mut(&mut pair),
                &dimensions,
                side,
            )
            .unwrap();
            let probe = one_sided_publication_probe();
            assert_eq!(opposite_of(&pair, side), opposite_of(&reference, side));
            if reverse {
                assert_eq!(
                    (probe.canonical_publications, probe.fallback_publications),
                    (0, 1)
                );
                assert_eq!(selected_of(&pair, side), selected_of(&reference, side));
            } else {
                assert_eq!(
                    (
                        probe.canonical_publications,
                        probe.fallback_publications,
                        probe.owner_reused,
                        probe.appended_elements,
                    ),
                    (1, 0, 1, 0)
                );
                assert!(std::ptr::eq(selected_ptr, factor.data().as_ptr()));
                assert!(selected_of(&pair, side).is_empty());
                assert_eq!(factor.data(), selected_of(&reference, side));
            }
        }
    }
}

/// Records every checked provider query in call order so that two
/// publication routes can be compared query by query.
struct RecordingGeneric {
    rule: TestGenericRule,
    log: RefCell<Vec<String>>,
}

impl RecordingGeneric {
    fn new() -> Self {
        Self {
            rule: TestGenericRule,
            log: RefCell::new(Vec::new()),
        }
    }

    fn record<T>(&self, entry: String, value: T) -> Result<T, std::convert::Infallible> {
        self.log.borrow_mut().push(entry);
        Ok(value)
    }

    fn log(&self) -> Vec<String> {
        self.log.borrow().clone()
    }
}

impl CheckedGenericFusion for RecordingGeneric {
    type Error = std::convert::Infallible;

    fn rule_identity(&self) -> tenet_core::RuleIdentity {
        self.rule.rule_identity()
    }

    fn fusion_style(&self) -> tenet_core::FusionStyleKind {
        self.rule.fusion_style()
    }

    fn braiding_style(&self) -> tenet_core::BraidingStyleKind {
        self.rule.braiding_style()
    }

    fn vacuum(&self) -> SectorId {
        self.rule.vacuum()
    }

    fn try_dual(&self, sector: SectorId) -> Result<SectorId, Self::Error> {
        self.record(format!("dual {sector:?}"), self.rule.dual(sector))
    }

    fn try_fusion_channels(
        &self,
        left: SectorId,
        right: SectorId,
    ) -> Result<tenet_core::SectorVec, Self::Error> {
        self.record(
            format!("channels {left:?} {right:?}"),
            self.rule.fusion_channels(left, right),
        )
    }

    fn try_fusion_channels_in_table(
        &self,
        left: SectorId,
        right: SectorId,
    ) -> Result<tenet_core::SectorVec, Self::Error> {
        self.record(
            format!("channels_in_table {left:?} {right:?}"),
            self.rule.fusion_channels(left, right),
        )
    }

    fn try_nsymbol(
        &self,
        left: SectorId,
        right: SectorId,
        coupled: SectorId,
    ) -> Result<usize, Self::Error> {
        self.record(
            format!("nsymbol {left:?} {right:?} {coupled:?}"),
            self.rule.nsymbol(left, right, coupled),
        )
    }
}

/// The output HomSpace exactly as `build_bound_factor_generic_checked`
/// derives it, built independently of the returned factor.
fn one_sided_output_hom(
    homspace: &FusionTreeHomSpace,
    dimensions: &BTreeMap<SectorId, usize>,
    side: FactorSide,
) -> FusionTreeHomSpace {
    let bond = SectorLeg::new(
        dimensions.iter().map(|(&sector, &dim)| (sector, dim)),
        false,
    );
    match side {
        FactorSide::Left => {
            FusionTreeHomSpace::new(homspace.codomain().clone(), FusionProductSpace::new([bond]))
        }
        FactorSide::Right => {
            FusionTreeHomSpace::new(FusionProductSpace::new([bond]), homspace.domain().clone())
        }
    }
}

fn block_rows(structure: &BlockStructure) -> Vec<(BlockKey, Vec<usize>, Vec<usize>, usize)> {
    (0..structure.block_count())
        .map(|index| structure.block(index).unwrap())
        .map(|block| {
            (
                block.key().clone(),
                block.shape().to_vec(),
                block.strides().to_vec(),
                block.offset(),
            )
        })
        .collect()
}

/// Two-sector checked geometry with `keep` selecting which sectors carry a
/// matricization; the others become identity-only output sectors.
type TwoSectorCheckedFixture = (
    FusionTreeHomSpace,
    Vec<SectorMatricization<Complex64>>,
    BTreeMap<SectorId, usize>,
    BTreeMap<SectorId, usize>,
);

fn two_sector_checked_fixture(keep: &dyn Fn(SectorId) -> bool) -> TwoSectorCheckedFixture {
    let rule = TestGenericRule;
    let provider = Arc::new(InfallibleGeneric::new(&rule));
    let even = SectorId::new(0);
    let odd = SectorId::new(1);
    let homspace = FusionTreeHomSpace::new(
        FusionProductSpace::new([
            SectorLeg::new([(even, 2), (odd, 1)], false),
            SectorLeg::new([(even, 1), (odd, 3)], false),
        ]),
        FusionProductSpace::new([
            SectorLeg::new([(even, 1), (odd, 2)], false),
            SectorLeg::new([(even, 2), (odd, 1)], false),
        ]),
    );
    let space =
        BoundDynamicFusionMapSpace::from_final_homspace_generic_checked(provider, homspace.clone())
            .unwrap();
    let zeros = vec![Complex64::new(0.0, 0.0); space.space().required_len().unwrap()];
    let matrices = sector_matricizations(space.space().structure(), &zeros, 2).unwrap();
    assert_eq!(matrices.len(), 2);
    let row_dimensions = matrices
        .iter()
        .map(|matrix| (matrix.sector, matrix.rows))
        .collect();
    let col_dimensions = matrices
        .iter()
        .map(|matrix| (matrix.sector, matrix.cols))
        .collect();
    let matrices = matrices
        .into_iter()
        .filter(|matrix| keep(matrix.sector))
        .collect();
    (homspace, matrices, row_dimensions, col_dimensions)
}

#[test]
#[allow(clippy::arc_with_non_send_sync)] // The checked API needs Arc identity; the recorder is a single-threaded RefCell log.
fn checked_one_sided_enumerates_the_output_layout_once() {
    // What: one publication issues exactly the provider queries of a
    // single layout enumeration; the former route (keys, then a bound
    // space) issued that identical sequence twice, and the committed
    // block order equals the enumerated key order.
    let dimensions = BTreeMap::from([(SectorId::new(1), 2)]);
    for reverse in [false, true] {
        for side in [FactorSide::Left, FactorSide::Right] {
            let (homspace, matrix, mut pair) = vertex_tree_factor_fixture(reverse);
            let recorder = Arc::new(RecordingGeneric::new());
            let factor = build_bound_factor_generic_checked(
                &recorder,
                &homspace,
                std::slice::from_ref(&matrix),
                std::slice::from_mut(&mut pair),
                &dimensions,
                side,
            )
            .unwrap();
            let once = recorder.log();
            // Vertex fixture: the one-leg bond side folds once, the
            // two-leg side queries channels and the multiplicity of its
            // single vertex.
            assert_eq!(once.len(), 3, "{once:?}");

            let former = Arc::new(RecordingGeneric::new());
            let output_hom = one_sided_output_hom(&homspace, &dimensions, side);
            let keys = output_hom
                .fusion_tree_keys_generic_checked(former.as_ref())
                .unwrap();
            let expected = BoundDynamicFusionMapSpace::from_final_homspace_generic_checked(
                Arc::clone(&former),
                output_hom,
            )
            .unwrap();
            let twice = former.log();
            assert_eq!(twice.len(), 2 * once.len());
            assert_eq!(&twice[..once.len()], once.as_slice());
            assert_eq!(&twice[once.len()..], once.as_slice());

            let block_keys = block_rows(factor.space().space().structure())
                .into_iter()
                .map(|(key, ..)| key)
                .collect::<Vec<_>>();
            let key_order = keys.into_iter().map(BlockKey::from).collect::<Vec<_>>();
            assert_eq!(block_keys, key_order);
            assert_eq!(
                block_rows(factor.space().space().structure()),
                block_rows(expected.space().structure())
            );
            assert!(Arc::ptr_eq(factor.space().provider_arc(), &recorder));
        }
    }
}

#[test]
#[allow(clippy::arc_with_non_send_sync)] // The checked API needs Arc identity; the recorder is a single-threaded RefCell log.
fn checked_one_sided_placement_error_precedes_bound_space() {
    // What: a populated output key whose tree the matricization lacks
    // fails after the single enumeration and before any bound space or
    // publication exists; the staged factor buffers stay untouched.
    let dimensions = BTreeMap::from([(SectorId::new(1), 2)]);
    for side in [FactorSide::Left, FactorSide::Right] {
        let (homspace, mut matrix, mut pair) = vertex_tree_factor_fixture(false);
        let (_, _, reference) = vertex_tree_factor_fixture(false);
        match side {
            FactorSide::Left => matrix.row_trees.truncate(1),
            FactorSide::Right => matrix.col_trees.truncate(1),
        }
        let recorder = Arc::new(RecordingGeneric::new());
        reset_one_sided_publication_probe();
        reset_placement_index_probe();
        let error = build_bound_factor_generic_checked(
            &recorder,
            &homspace,
            std::slice::from_ref(&matrix),
            std::slice::from_mut(&mut pair),
            &dimensions,
            side,
        )
        .unwrap_err();
        let expected_message = match side {
            FactorSide::Left => "factor codomain tree absent from the source matricization",
            FactorSide::Right => "factor domain tree absent from the source matricization",
        };
        assert!(matches!(
            error,
            CheckedGenericFactorPlanError::Operation(
                OperationError::UnsupportedTensorContractScope { message }
            ) if message == expected_message
        ));
        assert_eq!(recorder.log().len(), 3);
        let probe = one_sided_publication_probe();
        assert_eq!(
            (probe.canonical_publications, probe.fallback_publications),
            (0, 0)
        );
        let index = placement_index_probe();
        assert_eq!(
            (index.index_builds, index.indexed_sides, index.indexed_trees),
            (1, 1, 1)
        );
        assert_eq!(index.lookups, 2);
        assert_eq!(pair.left, reference.left);
        assert_eq!(pair.right, reference.right);
    }
}

#[test]
fn checked_one_sided_structure_matches_two_enumeration_construction() {
    // What: with identity-only sectors before or after the populated one,
    // the committed structure, required length, data length and provider
    // binding equal those of a space built by the former separate
    // enumeration, for both sides and both scalar types.
    let rule = TestGenericRule;
    let provider = Arc::new(InfallibleGeneric::new(&rule));
    let even = SectorId::new(0);
    let odd = SectorId::new(1);
    let keeps: [&dyn Fn(SectorId) -> bool; 3] =
        [&|_| true, &|sector| sector == odd, &|sector| sector == even];
    for keep in keeps {
        let (homspace, matrices, row_dimensions, col_dimensions) = two_sector_checked_fixture(keep);
        for side in [FactorSide::Left, FactorSide::Right] {
            let dimensions = match side {
                FactorSide::Left => &row_dimensions,
                FactorSide::Right => &col_dimensions,
            };
            let expected = BoundDynamicFusionMapSpace::from_final_homspace_generic_checked(
                Arc::clone(&provider),
                one_sided_output_hom(&homspace, dimensions, side),
            )
            .unwrap();
            let expected_len = expected.space().required_len().unwrap();

            let complex = |k: usize| Complex64::new(0.5 * k as f64, 2.0 - k as f64);
            let mut pairs = staged_one_sided_pairs(&matrices, dimensions, side, side, &complex);
            let factor = build_bound_factor_generic_checked(
                &provider, &homspace, &matrices, &mut pairs, dimensions, side,
            )
            .unwrap();
            assert_eq!(
                block_rows(factor.space().space().structure()),
                block_rows(expected.space().structure())
            );
            assert_eq!(factor.space().space().required_len().unwrap(), expected_len);
            assert_eq!(factor.data().len(), expected_len);
            assert_eq!(
                factor.space().space().homspace(),
                expected.space().homspace()
            );
            assert!(Arc::ptr_eq(factor.space().provider_arc(), &provider));

            let real_matrices = matrices
                .iter()
                .map(|matrix| SectorMatricization {
                    sector: matrix.sector,
                    rows: matrix.rows,
                    cols: matrix.cols,
                    row_trees: matrix.row_trees.clone(),
                    col_trees: matrix.col_trees.clone(),
                    data: Vec::<f64>::new(),
                })
                .collect::<Vec<_>>();
            let real = |k: usize| 0.25 * k as f64;
            let mut pairs = staged_one_sided_pairs(&real_matrices, dimensions, side, side, &real);
            let factor = build_bound_factor_generic_checked(
                &provider,
                &homspace,
                &real_matrices,
                &mut pairs,
                dimensions,
                side,
            )
            .unwrap();
            assert_eq!(
                block_rows(factor.space().space().structure()),
                block_rows(expected.space().structure())
            );
            assert_eq!(factor.data().len(), expected_len);
        }
    }
}

/// Z_4 fusion declared under Generic style so the generic paired builders
/// accept it; multiplicity-free, so trees and blocks match the abelian
/// geometry.
struct Z4GenericRule;

static Z4_GENERIC: Z4GenericRule = Z4GenericRule;

impl FusionRule for Z4GenericRule {
    fn rule_identity(&self) -> tenet_core::RuleIdentity {
        tenet_core::RuleIdentity::of_type::<Self>()
    }

    fn fusion_style(&self) -> tenet_core::FusionStyleKind {
        tenet_core::FusionStyleKind::Generic
    }

    fn braiding_style(&self) -> tenet_core::BraidingStyleKind {
        tenet_core::BraidingStyleKind::Bosonic
    }

    fn vacuum(&self) -> SectorId {
        SectorId::new(0)
    }

    fn dual(&self, sector: SectorId) -> SectorId {
        SectorId::new((4 - sector.id() % 4) % 4)
    }

    fn fusion_channels(&self, left: SectorId, right: SectorId) -> tenet_core::SectorVec {
        [SectorId::new((left.id() + right.id()) % 4)]
            .into_iter()
            .collect()
    }

    fn nsymbol(&self, left: SectorId, right: SectorId, coupled: SectorId) -> usize {
        usize::from((left.id() + right.id()) % 4 == coupled.id())
    }
}

/// Two-leg codomain/domain hom space under [`Z4GenericRule`], matricized
/// exactly as the generic production path does.
fn z4_generic_geometry<D: FactorScalar>(
    legs: [Vec<(SectorId, usize)>; 4],
) -> (FusionTreeHomSpace, Vec<SectorMatricization<D>>) {
    let [a, b, c, d] = legs;
    let homspace = FusionTreeHomSpace::new(
        FusionProductSpace::new([SectorLeg::new(a, false), SectorLeg::new(b, false)]),
        FusionProductSpace::new([SectorLeg::new(c, false), SectorLeg::new(d, false)]),
    );
    let space = BoundDynamicFusionMapSpace::from_final_homspace_generic_checked(
        Arc::new(InfallibleGeneric::new(&Z4_GENERIC)),
        homspace.clone(),
    )
    .unwrap();
    let zeros = vec![D::zero(); space.space().required_len().unwrap()];
    let matrices = sector_matricizations_generic(space.space().structure(), &zeros, 2).unwrap();
    (homspace, matrices)
}

/// Every leg carries all four charges: G_s = 4 coupled sectors with four
/// row and four column trees each, so each factor side has B = F = 16
/// output blocks.
fn z4_all_charge_legs() -> [Vec<(SectorId, usize)>; 4] {
    let all = (0..4).map(|s| (SectorId::new(s), 1)).collect::<Vec<_>>();
    [all.clone(), all.clone(), all.clone(), all]
}

/// Stages `rows x kept` left and `kept x cols` right factors per sector
/// with `kept = min(rows, cols)` and sector-tagged values.
fn staged_pairs<D: FactorScalar>(
    matrices: &[SectorMatricization<D>],
    values: &dyn Fn(usize) -> D,
) -> Vec<FactorPair<D>> {
    matrices
        .iter()
        .map(|matrix| {
            let kept = matrix.rows.min(matrix.cols);
            let tag = matrix.sector.id() + 1;
            FactorPair {
                sector: matrix.sector,
                kept,
                left: (0..matrix.rows * kept)
                    .map(|k| values(1000 * tag + k))
                    .collect(),
                left_rows: matrix.rows,
                right: (0..kept * matrix.cols)
                    .map(|k| values(5000 * tag + k))
                    .collect(),
                right_leading: kept,
            }
        })
        .collect()
}

#[test]
fn generic_pair_fallback_scatters_each_output_block_once() {
    // What: a paired fallback publication over G_s = 4 matricizations
    // groups each factor side once and iterates only the F blocks of the
    // matricization being scattered, publishing the same data as the
    // canonical path, for both paired builders and both scalar types.
    fn run<D: FactorScalar + fmt::Debug>(values: impl Fn(usize) -> D) {
        let fresh = || z4_generic_geometry::<D>(z4_all_charge_legs());
        let (homspace, matrices) = fresh();
        assert_eq!(matrices.len(), 4);
        assert!(matrices
            .iter()
            .all(|m| m.row_trees.len() == 4 && m.col_trees.len() == 4));
        let plain_provider = Arc::new(Z4GenericRule);
        let checked_provider = Arc::new(InfallibleGeneric::new(&Z4_GENERIC));

        let build = |matrices: &[SectorMatricization<D>]| {
            reset_generic_pair_publication_probe();
            reset_scatter_visit_probe();
            let (left, right) = build_left_right_bound_pair_generic(
                &plain_provider,
                &homspace,
                matrices,
                staged_pairs(matrices, &values),
            )
            .unwrap();
            let plain = (
                left.data().to_vec(),
                right.data().to_vec(),
                generic_pair_publication_probe(),
                scatter_visit_probe(),
            );
            reset_generic_pair_publication_probe();
            reset_scatter_visit_probe();
            let (left, right) = build_left_right_bound_pair_generic_checked(
                &checked_provider,
                &homspace,
                matrices,
                staged_pairs(matrices, &values),
            )
            .unwrap();
            assert_eq!(
                (
                    left.space().space().structure().block_count(),
                    right.space().space().structure().block_count()
                ),
                (16, 16)
            );
            let checked = (
                left.data().to_vec(),
                right.data().to_vec(),
                generic_pair_publication_probe(),
                scatter_visit_probe(),
            );
            (plain, checked)
        };

        let (reference, checked_reference) = build(&matrices);
        for (_, _, probe, visits) in [&reference, &checked_reference] {
            assert_eq!(probe.canonical_publications, 1);
            assert_eq!(*visits, ScatterVisitProbe::default());
        }

        let (_, mut reversed) = fresh();
        reversed.reverse();
        let (plain, checked) = build(&reversed);
        for ((left, right, probe, visits), (left_ref, right_ref, ..)) in
            [(&plain, &reference), (&checked, &checked_reference)]
        {
            assert_eq!(probe.fallback_publications, 1);
            assert_eq!((left, right), (left_ref, right_ref));
            assert_eq!(
                (probe.left_scatter_calls, probe.right_scatter_calls),
                (16, 16)
            );
            // Formerly G_s * B = 4 * 16 = 64 block visits per side; now
            // one grouping pass over B = 16 plus the F = 16 scattered
            // blocks.
            assert_eq!(
                *visits,
                ScatterVisitProbe {
                    left_grouped: 16,
                    right_grouped: 16,
                    left_groups_built: 1,
                    right_groups_built: 1,
                    left_visits: 16,
                    right_visits: 16,
                }
            );
            assert!(visits.left_grouped + visits.left_visits < 4 * 16);
            assert!(visits.right_grouped + visits.right_visits < 4 * 16);
        }
    }
    run(|k| k as f64 + 0.125);
    run(|k| Complex64::new(k as f64 + 0.125, 0.75 - k as f64));
}

#[test]
fn generic_pair_validation_reports_left_defect_before_right_across_sectors() {
    // What: validation precedence, not scatter order. With a defective
    // domain tree in the first sector and a defective codomain tree in a
    // later sector, the codomain error fires for both paired builders
    // because `validate_generic_factor_keys` checks every left key before
    // any right key. Scatter-time placement errors are unreachable in the
    // paired builders: validation has already checked every key.
    let two = || {
        let leg = || vec![(SectorId::new(0), 1), (SectorId::new(1), 1)];
        z4_generic_geometry::<f64>([leg(), leg(), leg(), leg()])
    };
    let (homspace, matrices) = two();
    assert_eq!(
        matrices.iter().map(|m| m.sector.id()).collect::<Vec<_>>(),
        [0, 1, 2]
    );
    let plain_provider = Arc::new(Z4GenericRule);
    let checked_provider = Arc::new(InfallibleGeneric::new(&Z4_GENERIC));
    let foreign_row = matrices[0].row_trees[0].0.clone();
    let foreign_col = matrices[1].col_trees[0].0.clone();
    let corrupt = |bad_first_col: bool, bad_later_row: bool| {
        let (_, mut matrices) = two();
        if bad_first_col {
            matrices[0].col_trees[0].0 = foreign_col.clone();
        }
        if bad_later_row {
            matrices[1].row_trees[0].0 = foreign_row.clone();
        }
        matrices
    };
    let codomain = "factor codomain tree absent from the source matricization";
    let domain = "factor domain tree absent from the source matricization";
    for (bad_first_col, bad_later_row, expected) in [
        (true, true, codomain),
        (true, false, domain),
        (false, true, codomain),
    ] {
        let matrices = corrupt(bad_first_col, bad_later_row);
        let error = build_left_right_bound_pair_generic(
            &plain_provider,
            &homspace,
            &matrices,
            staged_pairs(&matrices, &|k| k as f64),
        )
        .unwrap_err();
        assert!(matches!(
            error,
            OperationError::UnsupportedTensorContractScope { message } if message == expected
        ));
        let error = build_left_right_bound_pair_generic_checked(
            &checked_provider,
            &homspace,
            &matrices,
            staged_pairs(&matrices, &|k| k as f64),
        )
        .unwrap_err();
        assert!(matches!(
            error,
            CheckedGenericFactorPlanError::Operation(
                OperationError::UnsupportedTensorContractScope { message }
            ) if message == expected
        ));
    }
}

/// The paired output HomSpaces exactly as
/// `build_left_right_bound_pair_generic_checked` derives them, built
/// independently of the returned factors.
fn paired_output_homs<D>(
    homspace: &FusionTreeHomSpace,
    pairs: &[FactorPair<D>],
) -> (FusionTreeHomSpace, FusionTreeHomSpace) {
    let bond = SectorLeg::new(pairs.iter().map(|pair| (pair.sector, pair.kept)), false);
    (
        FusionTreeHomSpace::new(
            homspace.codomain().clone(),
            FusionProductSpace::new([bond.clone()]),
        ),
        FusionTreeHomSpace::new(FusionProductSpace::new([bond]), homspace.domain().clone()),
    )
}

/// One checked factor space built the former way: keys enumerated, then
/// the bound space enumerated again.
fn two_enumeration_space<R: CheckedGenericFusion>(
    provider: &Arc<R>,
    hom: FusionTreeHomSpace,
) -> (Vec<FusionTreePairKey>, BoundDynamicFusionMapSpace<R>) {
    let keys = hom
        .fusion_tree_keys_generic_checked(provider.as_ref())
        .unwrap();
    let space =
        BoundDynamicFusionMapSpace::from_final_homspace_generic_checked(Arc::clone(provider), hom)
            .unwrap();
    (keys, space)
}

fn assert_factor_matches_space<R: CheckedGenericFusion, D: FactorScalar + fmt::Debug>(
    factor: &BoundDynFactor<R, D>,
    keys: &[FusionTreePairKey],
    expected: &BoundDynamicFusionMapSpace<R>,
    provider: &Arc<R>,
) {
    let rows = block_rows(factor.space().space().structure());
    assert_eq!(rows, block_rows(expected.space().structure()));
    assert_eq!(
        rows.into_iter().map(|(key, ..)| key).collect::<Vec<_>>(),
        keys.iter().cloned().map(BlockKey::from).collect::<Vec<_>>()
    );
    let expected_len = expected.space().required_len().unwrap();
    assert_eq!(factor.space().space().required_len().unwrap(), expected_len);
    assert_eq!(factor.data().len(), expected_len);
    assert_eq!(
        factor.space().space().homspace(),
        expected.space().homspace()
    );
    assert!(Arc::ptr_eq(factor.space().provider_arc(), provider));
}

/// Checked paired publication equals the two-enumeration construction in
/// structure, key order, length, homspace and provider binding, and the
/// unchecked paired builder in data.
fn assert_checked_pair_matches_two_enumeration_construction<
    P: FusionRule,
    R: CheckedGenericFusion,
    D: FactorScalar + fmt::Debug,
    M: SectorGeometry,
>(
    plain: &Arc<P>,
    checked: &Arc<R>,
    homspace: &FusionTreeHomSpace,
    matrices: &[M],
    pairs: &dyn Fn() -> Vec<FactorPair<D>>,
    expected_publications: (usize, usize),
) {
    let (left_hom, right_hom) = paired_output_homs(homspace, &pairs());
    let (left_keys, left_expected) = two_enumeration_space(checked, left_hom);
    let (right_keys, right_expected) = two_enumeration_space(checked, right_hom);
    let (plain_left, plain_right) =
        build_left_right_bound_pair_generic(plain, homspace, matrices, pairs()).unwrap();
    reset_generic_pair_publication_probe();
    let (left, right) =
        build_left_right_bound_pair_generic_checked(checked, homspace, matrices, pairs()).unwrap();
    let probe = generic_pair_publication_probe();
    assert_eq!(
        (probe.canonical_publications, probe.fallback_publications),
        expected_publications
    );
    assert_eq!(probe.output_blocks_visited, 0);
    assert_eq!(
        probe.ordered_key_validation_events,
        left_keys.len() + right_keys.len()
    );
    assert_factor_matches_space(&left, &left_keys, &left_expected, checked);
    assert_factor_matches_space(&right, &right_keys, &right_expected, checked);
    assert_eq!(left.data(), plain_left.data());
    assert_eq!(right.data(), plain_right.data());
}

#[test]
#[allow(clippy::arc_with_non_send_sync)] // The checked API needs Arc identity; the recorder is a single-threaded RefCell log.
fn checked_paired_enumerates_each_factor_layout_once() {
    // What: one paired publication issues exactly the provider queries of
    // one left enumeration followed by one right enumeration; the former
    // route (keys, then a bound space, per side) issued each of those
    // sequences twice. Canonical (`reverse = false`) and fallback
    // (`reverse = true`) geometries publish the same structure and data
    // as the two-enumeration construction.
    for reverse in [false, true] {
        let (homspace, matrix, pair) = vertex_tree_factor_fixture(reverse);
        let pairs = || vec![vertex_tree_factor_fixture(reverse).2];
        let recorder = Arc::new(RecordingGeneric::new());
        build_left_right_bound_pair_generic_checked(
            &recorder,
            &homspace,
            std::slice::from_ref(&matrix),
            vec![pair],
        )
        .unwrap();
        let once = recorder.log();

        let (left_hom, right_hom) = paired_output_homs(&homspace, &pairs());
        let side_log = |hom: &FusionTreeHomSpace| {
            let probe = RecordingGeneric::new();
            hom.prepare_coupled_subblock_structure_from_leg_degeneracies_generic_checked(&probe)
                .unwrap();
            probe.log()
        };
        let (once_left, once_right) = (side_log(&left_hom), side_log(&right_hom));
        // Vertex fixture: each side folds its one-leg bond once and
        // queries channels plus the multiplicity of its single vertex.
        assert_eq!((once_left.len(), once_right.len()), (3, 3));
        assert_eq!(once, [once_left.clone(), once_right.clone()].concat());

        let former = Arc::new(RecordingGeneric::new());
        two_enumeration_space(&former, left_hom);
        two_enumeration_space(&former, right_hom);
        assert_eq!(
            former.log(),
            [once_left.clone(), once_left, once_right.clone(), once_right].concat()
        );

        assert_checked_pair_matches_two_enumeration_construction(
            &Arc::new(TestGenericRule),
            &recorder,
            &homspace,
            std::slice::from_ref(&matrix),
            &pairs,
            if reverse { (0, 1) } else { (1, 0) },
        );
    }
}

#[test]
fn checked_paired_structure_matches_two_enumeration_construction() {
    // What: over G_s = 4 Z_4 matricizations with F = 16 blocks per side,
    // canonical and reversed (fallback) geometries, `f64` and
    // `Complex64`, the checked pair equals the two-enumeration
    // construction and the unchecked builder's data.
    fn run<D: FactorScalar + fmt::Debug>(values: impl Fn(usize) -> D) {
        let fresh = || z4_generic_geometry::<D>(z4_all_charge_legs());
        let (homspace, matrices) = fresh();
        let (_, mut reversed) = fresh();
        reversed.reverse();
        let plain = Arc::new(Z4GenericRule);
        let checked = Arc::new(InfallibleGeneric::new(&Z4_GENERIC));
        for (matrices, publications) in [(&matrices, (1, 0)), (&reversed, (0, 1))] {
            assert_checked_pair_matches_two_enumeration_construction(
                &plain,
                &checked,
                &homspace,
                matrices,
                &|| staged_pairs(matrices, &values),
                publications,
            );
        }
    }
    run(|k| k as f64 + 0.125);
    run(|k| Complex64::new(k as f64 + 0.125, 0.75 - k as f64));
}

#[test]
#[allow(clippy::arc_with_non_send_sync)] // The checked API needs Arc identity; the recorder is a single-threaded RefCell log.
fn checked_paired_placement_error_precedes_bound_space() {
    // What: a factor key whose tree the matricization lacks fails after
    // that side's single enumeration and before any publication; the
    // left side is validated completely before the right side is
    // enumerated. Formerly the left error surfaced after 3 queries and
    // the right error after 9 (left keys, left space, right keys); now
    // after 3 and 6.
    for side in [FactorSide::Left, FactorSide::Right] {
        let (homspace, mut matrix, pair) = vertex_tree_factor_fixture(false);
        match side {
            FactorSide::Left => matrix.row_trees.truncate(1),
            FactorSide::Right => matrix.col_trees.truncate(1),
        }
        let recorder = Arc::new(RecordingGeneric::new());
        reset_generic_pair_publication_probe();
        let error = build_left_right_bound_pair_generic_checked(
            &recorder,
            &homspace,
            std::slice::from_ref(&matrix),
            vec![pair],
        )
        .unwrap_err();
        let (expected_message, expected_calls) = match side {
            FactorSide::Left => (
                "factor codomain tree absent from the source matricization",
                3,
            ),
            FactorSide::Right => ("factor domain tree absent from the source matricization", 6),
        };
        assert!(matches!(
            error,
            CheckedGenericFactorPlanError::Operation(
                OperationError::UnsupportedTensorContractScope { message }
            ) if message == expected_message
        ));
        assert_eq!(recorder.log().len(), expected_calls);
        let probe = generic_pair_publication_probe();
        assert_eq!(
            (probe.canonical_publications, probe.fallback_publications),
            (0, 0)
        );
    }
}
