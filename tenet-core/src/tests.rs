mod tests {
    use super::*;
    use smallvec::smallvec;

    /// Fixture layout: subblocks packed contiguously in key order. Not a product
    /// layout (the only one is the coupled sector matrix); fixtures use it to
    /// exercise the arbitrary-strided-view contract of [`BlockStructure`].
    fn packed_fixture_structure<I, K>(rank: usize, blocks: I) -> Result<BlockStructure, CoreError>
    where
        I: IntoIterator<Item = (K, Vec<usize>)>,
        K: Into<BlockKey>,
    {
        let mut keys = Vec::new();
        let mut shapes = Vec::new();
        for (key, shape) in blocks {
            keys.push(key.into());
            shapes.push(shape);
        }
        BlockStructure::from_parts(
            SectorStructure::from_keys(rank, keys)?,
            DegeneracyStructure::packed_column_major(rank, shapes)?,
        )
    }

    #[test]
    fn block_fn_construction_is_layout_independent() {
        let rule = Z2FusionRule;
        let leg = |dual| SectorLeg::new([(z2_even(), 2), (z2_odd(), 2)], dual);
        let homspace = || {
            FusionTreeHomSpace::new(
                FusionProductSpace::new([leg(false), leg(false)]),
                FusionProductSpace::new([leg(false), leg(false)]),
            )
        };
        let shapes = |hom: &FusionTreeHomSpace| {
            hom.fusion_tree_keys(&rule)
                .iter()
                .map(|_| vec![2usize; 4])
                .collect::<Vec<_>>()
        };
        let dense = || TensorMapSpace::<2, 2>::from_dims([4, 4], [4, 4]).unwrap();
        let hom = homspace();
        let packed_structure =
            packed_fixture_structure(4, hom.fusion_tree_keys(&rule).into_iter().zip(shapes(&hom)))
                .unwrap();
        let packed_space =
            FusionTensorMapSpace::<2, 2>::new(dense(), hom.clone(), packed_structure).unwrap();
        let coupled_space = FusionTensorMapSpace::<2, 2>::from_degeneracy_shapes_coupled(
            dense(),
            hom.clone(),
            &rule,
            shapes(&hom),
        )
        .unwrap();

        let fill = |key: &BlockKey, indices: &[usize]| -> f64 {
            let BlockKey::FusionTree(tree) = key else {
                panic!("fusion tree keys expected");
            };
            let mut value = 0.0;
            for (axis, &sector) in tree
                .codomain_tree()
                .uncoupled()
                .iter()
                .chain(tree.domain_tree().uncoupled())
                .enumerate()
            {
                value += (axis as f64 + 1.0) * (sector.id() as f64 + 0.5);
            }
            for (axis, &index) in indices.iter().enumerate() {
                value += (axis as f64 + 2.0) * index as f64;
            }
            value
        };
        let packed =
            TensorMap::<f64, 2, 2>::from_block_fn_with_fusion_space(packed_space, 0.0, fill)
                .unwrap();
        let coupled =
            TensorMap::<f64, 2, 2>::from_block_fn_with_fusion_space(coupled_space, 0.0, fill)
                .unwrap();

        // Raw storage differs between layouts...
        assert_ne!(packed.data(), coupled.data());
        // ...but the logical block content is identical.
        let mut packed_elements = Vec::new();
        packed
            .for_each_block_element(|key, indices, value| {
                packed_elements.push((key.clone(), indices.to_vec(), *value));
            })
            .unwrap();
        let mut cursor = 0;
        coupled
            .for_each_block_element(|key, indices, value| {
                let (expected_key, expected_indices, expected_value) = &packed_elements[cursor];
                assert_eq!(key, expected_key);
                assert_eq!(indices, expected_indices.as_slice());
                assert_eq!(value, expected_value);
                cursor += 1;
            })
            .unwrap();
        assert_eq!(cursor, packed_elements.len());
    }

    #[test]
    fn coupled_layout_embeds_subblocks_into_sector_matrices() {
        let rule = Z2FusionRule;
        let leg = |degeneracy, dual| {
            SectorLeg::new([(z2_even(), degeneracy), (z2_odd(), degeneracy)], dual)
        };
        let homspace = FusionTreeHomSpace::new(
            FusionProductSpace::new([leg(2, false), leg(3, false)]),
            FusionProductSpace::new([leg(2, false), leg(3, false)]),
        );
        let keys = homspace.fusion_tree_keys(&rule);
        let shapes = keys
            .iter()
            .map(|_| vec![2usize, 3, 2, 3])
            .collect::<Vec<_>>();
        let packed = FusionTensorMapSpace::<2, 2>::from_degeneracy_shapes(
            TensorMapSpace::<2, 2>::from_dims([10, 10], [10, 10]).unwrap(),
            homspace.clone(),
            &rule,
            shapes.clone(),
        )
        .unwrap();
        let coupled = FusionTensorMapSpace::<2, 2>::from_degeneracy_shapes_coupled(
            TensorMapSpace::<2, 2>::from_dims([10, 10], [10, 10]).unwrap(),
            homspace,
            &rule,
            shapes,
        )
        .unwrap();

        let packed_structure = packed.subblock_structure();
        let coupled_structure = coupled.subblock_structure();
        assert_eq!(
            packed_structure.block_count(),
            coupled_structure.block_count()
        );
        assert_eq!(
            packed_structure.required_len().unwrap(),
            coupled_structure.required_len().unwrap()
        );

        // Two coupled sectors (even/odd), each with two codomain and two
        // domain trees of subblock row dim 6 and column dim 6: sector matrices
        // are 12 x 12.
        let matrix_rows = 12usize;
        let mut covered = vec![false; coupled_structure.required_len().unwrap()];
        for index in 0..coupled_structure.block_count() {
            let packed_block = packed_structure.block(index).unwrap();
            let coupled_block = coupled_structure.block(index).unwrap();
            assert_eq!(packed_block.key(), coupled_block.key());
            assert_eq!(packed_block.shape(), coupled_block.shape());
            // Codomain legs stay column-major inside the row block; domain
            // legs step whole matrix columns.
            assert_eq!(coupled_block.strides()[0], 1);
            assert_eq!(coupled_block.strides()[1], 2);
            assert_eq!(coupled_block.strides()[2], matrix_rows);
            assert_eq!(coupled_block.strides()[3], matrix_rows * 2);
            for i3 in 0..3 {
                for i2 in 0..2 {
                    for i1 in 0..3 {
                        for i0 in 0..2 {
                            let strides = coupled_block.strides();
                            let position = coupled_block.offset()
                                + i0 * strides[0]
                                + i1 * strides[1]
                                + i2 * strides[2]
                                + i3 * strides[3];
                            assert!(
                                !covered[position],
                                "coupled layout must not overlap between subblocks"
                            );
                            covered[position] = true;
                        }
                    }
                }
            }
        }
        assert!(
            covered.iter().all(|&flag| flag),
            "coupled layout must cover the sector matrices without holes"
        );
    }

    fn u1(charge: i32) -> SectorId {
        U1Irrep::new(charge).sector_id()
    }

    fn z2_even() -> SectorId {
        Z2Irrep::EVEN.sector_id()
    }

    fn z2_odd() -> SectorId {
        Z2Irrep::ODD.sector_id()
    }

    fn su2(twice_spin: usize) -> SectorId {
        SU2Irrep::from_twice_spin(twice_spin).sector_id()
    }

    #[derive(Clone, Copy, Debug)]
    struct BranchingMultiplicityFreeRule;

    impl FusionRule for BranchingMultiplicityFreeRule {
        fn fusion_style(&self) -> FusionStyleKind {
            FusionStyleKind::Simple
        }

        fn braiding_style(&self) -> BraidingStyleKind {
            BraidingStyleKind::Bosonic
        }

        fn vacuum(&self) -> SectorId {
            SectorId::new(0)
        }

        fn dual(&self, sector: SectorId) -> SectorId {
            match sector.id() {
                3 => SectorId::new(1),
                other => SectorId::new(other),
            }
        }

        fn fusion_channels(&self, left: SectorId, right: SectorId) -> SectorVec {
            match (left.id(), right.id()) {
                (0, x) | (x, 0) => smallvec![SectorId::new(x)],
                (1, 1) => smallvec![SectorId::new(0), SectorId::new(2)],
                (1, 2) | (2, 1) => smallvec![SectorId::new(1), SectorId::new(3)],
                (2, 2) => smallvec![SectorId::new(0)],
                _ => SmallVec::new(),
            }
        }
    }

    impl MultiplicityFreeFusionRule for BranchingMultiplicityFreeRule {}

    #[derive(Clone, Copy, Debug)]
    struct UnsortedFusionIteratorOrderRule;

    impl FusionRule for UnsortedFusionIteratorOrderRule {
        fn fusion_style(&self) -> FusionStyleKind {
            FusionStyleKind::Simple
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
                (0, x) | (x, 0) => smallvec![SectorId::new(x)],
                (1, 1) => smallvec![SectorId::new(2), SectorId::new(0)],
                (1, 2) | (2, 1) => smallvec![SectorId::new(1)],
                (2, 2) => smallvec![SectorId::new(0)],
                _ => SmallVec::new(),
            }
        }
    }

    impl MultiplicityFreeFusionRule for UnsortedFusionIteratorOrderRule {}

    #[derive(Clone, Copy, Debug)]
    struct Z4PointedRule;

    impl FusionRule for Z4PointedRule {
        fn fusion_style(&self) -> FusionStyleKind {
            FusionStyleKind::Unique
        }

        fn braiding_style(&self) -> BraidingStyleKind {
            BraidingStyleKind::Bosonic
        }

        fn vacuum(&self) -> SectorId {
            SectorId::new(0)
        }

        fn dual(&self, sector: SectorId) -> SectorId {
            SectorId::new((4 - sector.id() % 4) % 4)
        }

        fn fusion_channels(&self, left: SectorId, right: SectorId) -> SectorVec {
            smallvec![SectorId::new((left.id() + right.id()) % 4)]
        }
    }

    impl MultiplicityFreeFusionRule for Z4PointedRule {}

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

    #[derive(Clone, Copy, Debug)]
    struct PlanarZ2Rule;

    impl FusionRule for PlanarZ2Rule {
        fn fusion_style(&self) -> FusionStyleKind {
            FusionStyleKind::Unique
        }

        fn braiding_style(&self) -> BraidingStyleKind {
            BraidingStyleKind::NoBraiding
        }

        fn vacuum(&self) -> SectorId {
            SectorId::new(0)
        }

        fn fusion_channels(&self, left: SectorId, right: SectorId) -> SectorVec {
            smallvec![SectorId::new((left.id() + right.id()) % 2)]
        }
    }

    impl MultiplicityFreeFusionRule for PlanarZ2Rule {}

    impl MultiplicityFreeFusionSymbols for PlanarZ2Rule {
        type Scalar = f64;

        fn scalar_one(&self) -> Self::Scalar {
            1.0
        }

        fn scalar_conj(&self, value: Self::Scalar) -> Self::Scalar {
            value
        }

        fn f_symbol_scalar(
            &self,
            _left: SectorId,
            _middle: SectorId,
            _right: SectorId,
            _coupled: SectorId,
            _left_coupled: SectorId,
            _right_coupled: SectorId,
        ) -> Self::Scalar {
            1.0
        }

        fn r_symbol_scalar(
            &self,
            _left: SectorId,
            _right: SectorId,
            _coupled: SectorId,
        ) -> Self::Scalar {
            1.0
        }
    }

    #[derive(Clone, Copy, Debug)]
    struct AsymmetricAnyonicRule;

    impl FusionRule for AsymmetricAnyonicRule {
        fn fusion_style(&self) -> FusionStyleKind {
            FusionStyleKind::Unique
        }

        fn braiding_style(&self) -> BraidingStyleKind {
            BraidingStyleKind::Anyonic
        }

        fn vacuum(&self) -> SectorId {
            SectorId::new(0)
        }

        fn fusion_channels(&self, left: SectorId, right: SectorId) -> SectorVec {
            match (left.id(), right.id()) {
                (0, x) | (x, 0) => smallvec![SectorId::new(x)],
                (1, 2) | (2, 1) => smallvec![SectorId::new(3)],
                (3, 1) | (1, 3) => smallvec![SectorId::new(2)],
                (3, 2) | (2, 3) => smallvec![SectorId::new(1)],
                _ => smallvec![SectorId::new((left.id() + right.id()) % 4)],
            }
        }
    }

    impl MultiplicityFreeFusionRule for AsymmetricAnyonicRule {}

    impl MultiplicityFreeFusionSymbols for AsymmetricAnyonicRule {
        type Scalar = f64;

        fn scalar_one(&self) -> Self::Scalar {
            1.0
        }

        fn scalar_conj(&self, value: Self::Scalar) -> Self::Scalar {
            value
        }

        fn f_symbol_scalar(
            &self,
            _left: SectorId,
            _middle: SectorId,
            _right: SectorId,
            _coupled: SectorId,
            _left_coupled: SectorId,
            _right_coupled: SectorId,
        ) -> Self::Scalar {
            11.0
        }

        fn r_symbol_scalar(
            &self,
            left: SectorId,
            right: SectorId,
            _coupled: SectorId,
        ) -> Self::Scalar {
            match (left.id(), right.id()) {
                (1, 2) => 5.0,
                (2, 1) => 7.0,
                (3, 2) => 13.0,
                (2, 3) => 17.0,
                (1, 3) => 19.0,
                (3, 1) => 23.0,
                _ => 1.0,
            }
        }
    }

    impl MultiplicityFreePivotalSymbols for AsymmetricAnyonicRule {
        fn bendright_scalar(
            &self,
            _left_coupled: SectorId,
            _bent_sector: SectorId,
            _coupled: SectorId,
            _bent_leg_is_dual: bool,
        ) -> Self::Scalar {
            1.0
        }

        fn foldright_scalar(
            &self,
            _source: &FusionTreeBlockKey,
            _destination: &FusionTreeBlockKey,
        ) -> Self::Scalar {
            1.0
        }
    }

    fn fusion_tree_pair_order(keys: &[FusionTreeBlockKey]) -> Vec<(Vec<usize>, Vec<usize>, usize)> {
        keys.iter()
            .map(|key| {
                (
                    sector_ids(key.codomain_uncoupled()),
                    sector_ids(key.domain_uncoupled()),
                    key.coupled().expect("test keys have a coupled sector").id(),
                )
            })
            .collect()
    }

    fn sector_ids(sectors: &[SectorId]) -> Vec<usize> {
        sectors.iter().map(|sector| sector.id()).collect()
    }

    #[test]
    fn fusion_style_kind_matches_tensorkit_multiplicity_free_split() {
        assert!(FusionStyleKind::Unique.is_multiplicity_free());
        assert!(FusionStyleKind::Simple.is_multiplicity_free());
        assert!(!FusionStyleKind::Generic.is_multiplicity_free());
        assert!(!FusionStyleKind::Unique.has_multiple_outputs());
        assert!(FusionStyleKind::Simple.has_multiple_outputs());
        assert!(FusionStyleKind::Generic.has_multiple_outputs());
        assert!(!FusionStyleKind::Unique.has_multiplicity());
        assert!(!FusionStyleKind::Simple.has_multiplicity());
        assert!(FusionStyleKind::Generic.has_multiplicity());
    }

    #[test]
    fn braiding_style_kind_matches_tensorkit_hierarchy() {
        assert!(!BraidingStyleKind::NoBraiding.has_braiding());
        assert!(BraidingStyleKind::Bosonic.has_braiding());
        assert!(BraidingStyleKind::Fermionic.has_braiding());
        assert!(BraidingStyleKind::Anyonic.has_braiding());

        assert!(!BraidingStyleKind::NoBraiding.is_symmetric());
        assert!(BraidingStyleKind::Bosonic.is_symmetric());
        assert!(BraidingStyleKind::Fermionic.is_symmetric());
        assert!(!BraidingStyleKind::Anyonic.is_symmetric());

        assert!(BraidingStyleKind::Bosonic.is_bosonic());
        assert!(!BraidingStyleKind::Fermionic.is_bosonic());
        assert_eq!(
            BraidingStyleKind::Bosonic.combined_with(BraidingStyleKind::Fermionic),
            BraidingStyleKind::Fermionic
        );
        assert_eq!(
            BraidingStyleKind::Fermionic.combined_with(BraidingStyleKind::Anyonic),
            BraidingStyleKind::Anyonic
        );
        assert_eq!(
            BraidingStyleKind::Anyonic.combined_with(BraidingStyleKind::NoBraiding),
            BraidingStyleKind::NoBraiding
        );
    }

    #[test]
    fn fusion_rule_exposes_unique_outputs_and_nsymbol_separately() {
        let z2 = Z2FusionRule;
        let su2 = SU2FusionRule;

        assert_eq!(z2.fusion_style(), FusionStyleKind::Unique);
        assert_eq!(
            z2.fusion_channels(SectorId::new(1), SectorId::new(1))
                .to_vec(),
            vec![SectorId::new(0)]
        );
        assert_eq!(
            z2.nsymbol(SectorId::new(1), SectorId::new(1), SectorId::new(0)),
            1
        );
        assert_eq!(
            z2.nsymbol(SectorId::new(1), SectorId::new(1), SectorId::new(1)),
            0
        );

        assert_eq!(su2.fusion_style(), FusionStyleKind::Simple);
        assert_eq!(
            su2.fusion_channels(SectorId::new(1), SectorId::new(1))
                .to_vec(),
            vec![SectorId::new(0), SectorId::new(2)]
        );
        assert_eq!(
            su2.nsymbol(SectorId::new(1), SectorId::new(1), SectorId::new(2)),
            1
        );
    }

    #[test]
    fn multiplicity_free_symbols_are_a_separate_scalar_api() {
        let z2 = Z2FusionRule;

        assert_eq!(z2.scalar_one(), 1.0);
        assert_eq!(
            z2.f_symbol_scalar(
                SectorId::new(1),
                SectorId::new(1),
                SectorId::new(1),
                SectorId::new(1),
                SectorId::new(0),
                SectorId::new(0),
            ),
            1.0
        );
        assert_eq!(
            z2.r_symbol_scalar(SectorId::new(1), SectorId::new(1), SectorId::new(0)),
            1.0
        );
    }

    #[test]
    fn unique_artin_braid_first_allows_unit_crossing_without_braiding() {
        let tree = FusionTreeKey::from_sector_ids([0, 1], Some(1), [false, true], [], [1]);

        let (braided, coefficient) = unique_artin_braid_first(&PlanarZ2Rule, &tree).unwrap();

        assert_eq!(coefficient, 1.0);
        assert_eq!(braided.uncoupled(), &[SectorId::new(1), SectorId::new(0)]);
        assert_eq!(braided.is_dual(), &[true, false]);
        assert_eq!(braided.coupled(), Some(SectorId::new(1)));
        assert!(braided.innerlines().is_empty());
        assert_eq!(braided.vertices(), &[SectorId::new(1)]);
    }

    #[test]
    fn unique_artin_braid_first_rejects_nonunit_crossing_without_braiding() {
        let tree = FusionTreeKey::from_sector_ids([1, 1], Some(0), [false, false], [], [1]);

        let err = unique_artin_braid_first(&PlanarZ2Rule, &tree).unwrap_err();

        assert_eq!(
            err,
            CoreError::UnsupportedSectorBraid {
                left: SectorId::new(1),
                right: SectorId::new(1),
                style: BraidingStyleKind::NoBraiding,
            }
        );
    }

    #[test]
    fn unique_artin_braid_first_uses_r_symbol_for_first_crossing() {
        let tree = FusionTreeKey::from_sector_ids([1, 1], Some(0), [false, true], [], [1]);

        let (braided, coefficient) =
            unique_artin_braid_first(&FermionParityFusionRule, &tree).unwrap();

        assert_eq!(coefficient, -1.0);
        assert_eq!(braided.uncoupled(), &[SectorId::new(1), SectorId::new(1)]);
        assert_eq!(braided.is_dual(), &[true, false]);
        assert_eq!(braided.coupled(), Some(SectorId::new(0)));
        assert!(braided.innerlines().is_empty());
        assert_eq!(braided.vertices(), &[SectorId::new(1)]);
    }

    #[test]
    fn unique_artin_braid_first_uses_first_innerline_for_rank_three() {
        let tree =
            FusionTreeKey::from_sector_ids([1, 1, 1], Some(1), [false, false, false], [0], [1, 1]);

        let (braided, coefficient) =
            unique_artin_braid_first(&FermionParityFusionRule, &tree).unwrap();

        assert_eq!(coefficient, -1.0);
        assert_eq!(
            braided.uncoupled(),
            &[SectorId::new(1), SectorId::new(1), SectorId::new(1)]
        );
        assert_eq!(braided.innerlines(), &[SectorId::new(0)]);
        assert_eq!(braided.vertices(), &[SectorId::new(1), SectorId::new(1)]);
    }

    #[test]
    fn unique_artin_braid_at_updates_innerline_for_later_unit_crossing() {
        let tree =
            FusionTreeKey::from_sector_ids([1, 0, 1], Some(0), [false, false, true], [1], [1, 1]);

        let (braided, coefficient) = unique_artin_braid_at(&PlanarZ2Rule, &tree, 1).unwrap();

        assert_eq!(coefficient, 1.0);
        assert_eq!(
            braided.uncoupled(),
            &[SectorId::new(1), SectorId::new(1), SectorId::new(0)]
        );
        assert_eq!(braided.is_dual(), &[false, true, false]);
        assert_eq!(braided.innerlines(), &[SectorId::new(0)]);
        assert_eq!(braided.vertices(), &[SectorId::new(1), SectorId::new(1)]);
    }

    #[test]
    fn unique_artin_braid_at_uses_f_and_r_symbols_for_later_crossing() {
        let tree =
            FusionTreeKey::from_sector_ids([1, 1, 1], Some(1), [false, true, false], [0], [1, 1]);

        let (braided, coefficient) =
            unique_artin_braid_at(&FermionParityFusionRule, &tree, 1).unwrap();

        assert_eq!(coefficient, -1.0);
        assert_eq!(
            braided.uncoupled(),
            &[SectorId::new(1), SectorId::new(1), SectorId::new(1)]
        );
        assert_eq!(braided.is_dual(), &[false, false, true]);
        assert_eq!(braided.innerlines(), &[SectorId::new(0)]);
        assert_eq!(braided.vertices(), &[SectorId::new(1), SectorId::new(1)]);
    }

    #[test]
    fn unique_artin_braid_at_rejects_out_of_range_index() {
        let tree = FusionTreeKey::from_sector_ids([1, 1], Some(0), [false, false], [], [1]);

        let err = unique_artin_braid_at(&FermionParityFusionRule, &tree, 1).unwrap_err();

        assert_eq!(err, CoreError::InvalidBraidIndex { index: 1, rank: 2 });
    }

    #[test]
    fn permutation_to_adjacent_swaps_matches_tensorkit_order() {
        assert_eq!(
            permutation_to_adjacent_swaps(&[2, 0, 1], 3).unwrap(),
            vec![1, 0]
        );
        assert_eq!(
            permutation_to_adjacent_swaps(&[3, 0, 2, 1], 4).unwrap(),
            vec![2, 1, 0, 2]
        );
    }

    #[test]
    fn unique_braid_tree_replays_tensorkit_swap_order_and_level_updates() {
        let tree =
            FusionTreeKey::from_sector_ids([1, 1, 1], Some(1), [false, false, false], [0], [1, 1]);

        let (braided, coefficient) =
            unique_braid_tree(&FermionParityFusionRule, &tree, &[2, 0, 1], &[0, 1, 2]).unwrap();

        assert_eq!(coefficient, 1.0);
        assert_eq!(
            braided.uncoupled(),
            &[SectorId::new(1), SectorId::new(1), SectorId::new(1)]
        );
        assert_eq!(braided.is_dual(), &[false, false, false]);
        assert_eq!(braided.coupled(), Some(SectorId::new(1)));
        assert_eq!(braided.innerlines(), &[SectorId::new(0)]);
        assert_eq!(braided.vertices(), &[SectorId::new(1), SectorId::new(1)]);
    }

    #[test]
    fn unique_braid_tree_uses_inverse_artin_branch_from_levels() {
        let tree = FusionTreeKey::from_sector_ids([1, 2], Some(3), [false, false], [], [1]);

        let (braided_forward, forward) =
            unique_braid_tree(&AsymmetricAnyonicRule, &tree, &[1, 0], &[0, 1]).unwrap();
        let (braided_inverse, inverse) =
            unique_braid_tree(&AsymmetricAnyonicRule, &tree, &[1, 0], &[1, 0]).unwrap();

        assert_eq!(forward, 5.0);
        assert_eq!(inverse, 7.0);
        assert_eq!(braided_forward, braided_inverse);
        assert_eq!(
            braided_forward.uncoupled(),
            &[SectorId::new(2), SectorId::new(1)]
        );
        assert_eq!(braided_forward.coupled(), Some(SectorId::new(3)));
    }

    #[test]
    fn unique_braid_tree_reflected_levels_select_inverse_artin_branch() {
        let tree = FusionTreeKey::from_sector_ids([1, 2], Some(3), [false, false], [], [1]);
        let levels = [3, 8];
        let min_level = levels.iter().copied().min().unwrap();
        let max_level = levels.iter().copied().max().unwrap();
        let reflected_levels = levels
            .iter()
            .map(|&level| min_level + max_level - level)
            .collect::<Vec<_>>();

        let (forward_tree, forward_coeff) =
            unique_braid_tree(&AsymmetricAnyonicRule, &tree, &[1, 0], &levels).unwrap();
        let (inverse_tree, inverse_coeff) =
            unique_braid_tree(&AsymmetricAnyonicRule, &tree, &[1, 0], &reflected_levels).unwrap();

        assert_eq!(reflected_levels, vec![8, 3]);
        assert_eq!(forward_tree, inverse_tree);
        assert_eq!(forward_coeff, 5.0);
        assert_eq!(inverse_coeff, 7.0);
    }

    #[test]
    fn unique_braid_tree_rejects_invalid_permutation_and_level_count() {
        let tree = FusionTreeKey::from_sector_ids([1, 2], Some(3), [false, false], [], [1]);

        let err = unique_braid_tree(&AsymmetricAnyonicRule, &tree, &[1, 1], &[0, 1]).unwrap_err();
        assert_eq!(
            err,
            CoreError::InvalidPermutation {
                permutation: vec![1, 1],
                rank: 2,
            }
        );

        let err = unique_braid_tree(&AsymmetricAnyonicRule, &tree, &[1, 0], &[0]).unwrap_err();
        assert_eq!(
            err,
            CoreError::DimensionMismatch {
                expected: 2,
                actual: 1,
            }
        );
    }

    #[test]
    fn unique_permute_tree_requires_symmetric_braiding() {
        let tree = FusionTreeKey::from_sector_ids([1, 2], Some(3), [false, false], [], [1]);

        let err = unique_permute_tree(&AsymmetricAnyonicRule, &tree, &[1, 0]).unwrap_err();

        assert_eq!(
            err,
            CoreError::UnsupportedBraidingStyle {
                expected: "symmetric braiding",
                actual: BraidingStyleKind::Anyonic,
            }
        );
    }

    #[test]
    fn linearize_tree_pair_permutation_matches_tensorkit_zero_based_formula() {
        assert_eq!(
            linearize_tree_pair_permutation(&[0, 1], &[2, 3], 2, 2).unwrap(),
            vec![0, 1, 2, 3]
        );
        assert_eq!(
            linearize_tree_pair_permutation(&[3, 0], &[1, 2], 2, 2).unwrap(),
            vec![2, 0, 3, 1]
        );

        let err = linearize_tree_pair_permutation(&[0, 0], &[1, 2], 2, 2).unwrap_err();
        assert_eq!(
            err,
            CoreError::InvalidPermutation {
                permutation: vec![0, 0, 1, 2],
                rank: 4,
            }
        );
    }

    #[test]
    fn unique_repartition_tree_pair_moves_domain_to_reversed_dual_codomain() {
        let source = FusionTreeBlockKey::pair_from_sector_ids(
            [1],
            [0, 1],
            Some(1),
            [false],
            [false, true],
            [],
            [],
            [],
            [1],
        );

        let (all_out, coefficient) =
            unique_repartition_tree_pair(&Z2FusionRule, &source, 3).unwrap();

        assert_eq!(coefficient, 1.0);
        assert_eq!(
            all_out.codomain_uncoupled(),
            &[SectorId::new(1), SectorId::new(1), SectorId::new(0)]
        );
        assert_eq!(all_out.codomain_is_dual(), &[false, false, true]);
        assert_eq!(all_out.codomain_innerlines(), &[SectorId::new(0)]);
        assert_eq!(
            all_out.codomain_vertices(),
            &[SectorId::new(1), SectorId::new(1)]
        );
        assert!(all_out.domain_uncoupled().is_empty());
        assert_eq!(all_out.domain_tree().coupled(), Some(SectorId::new(0)));
    }

    #[test]
    fn unique_braid_tree_pair_matches_single_tree_when_domain_is_empty() {
        let source = FusionTreeBlockKey::pair(
            FusionTreeKey::from_sector_ids([1, 1], Some(0), [false, true], [], [1]),
            FusionTreeKey::new(
                Vec::<SectorId>::new(),
                None,
                Vec::<bool>::new(),
                Vec::<SectorId>::new(),
                Vec::<SectorId>::new(),
            ),
        );

        let (braided, coefficient) = unique_braid_tree_pair(
            &FermionParityFusionRule,
            &source,
            &[1, 0],
            &[],
            &[0, 1],
            &[],
        )
        .unwrap();

        assert_eq!(coefficient, -1.0);
        assert_eq!(
            braided.codomain_uncoupled(),
            &[SectorId::new(1), SectorId::new(1)]
        );
        assert_eq!(braided.codomain_is_dual(), &[true, false]);
        assert!(braided.domain_uncoupled().is_empty());
        assert_eq!(braided.domain_tree().coupled(), None);
    }

    #[test]
    fn unique_permute_tree_pair_handles_domain_only_swap() {
        let source = FusionTreeBlockKey::pair_from_sector_ids(
            [1],
            [0, 1],
            Some(1),
            [false],
            [false, true],
            [],
            [],
            [],
            [1],
        );

        let (permuted, coefficient) =
            unique_permute_tree_pair(&Z2FusionRule, &source, &[0], &[2, 1]).unwrap();

        assert_eq!(coefficient, 1.0);
        assert_eq!(permuted.codomain_uncoupled(), &[SectorId::new(1)]);
        assert_eq!(
            permuted.domain_uncoupled(),
            &[SectorId::new(1), SectorId::new(0)]
        );
        assert_eq!(permuted.domain_is_dual(), &[true, false]);
        assert_eq!(permuted.domain_vertices(), &[SectorId::new(1)]);
    }

    #[test]
    fn unique_permute_tree_pair_includes_codomain_domain_crossing() {
        let source = FusionTreeBlockKey::pair_from_sector_ids(
            [1],
            [1],
            Some(1),
            [false],
            [true],
            [],
            [],
            [],
            [],
        );

        let (permuted, coefficient) =
            unique_permute_tree_pair(&FermionParityFusionRule, &source, &[1], &[0]).unwrap();

        assert_eq!(coefficient, -1.0);
        assert_eq!(permuted.codomain_uncoupled(), &[SectorId::new(1)]);
        assert_eq!(permuted.codomain_is_dual(), &[false]);
        assert_eq!(permuted.domain_uncoupled(), &[SectorId::new(1)]);
        assert_eq!(permuted.domain_is_dual(), &[true]);
    }

    #[test]
    fn unique_transpose_tree_pair_is_cyclic_and_reversible() {
        let source = FusionTreeBlockKey::pair_from_sector_ids(
            [1],
            [1],
            Some(1),
            [false],
            [true],
            [],
            [],
            [],
            [],
        );

        let (transposed, coefficient) =
            unique_transpose_tree_pair(&Z2FusionRule, &source, &[1], &[0]).unwrap();
        let (roundtrip, inverse_coefficient) =
            unique_transpose_tree_pair(&Z2FusionRule, &transposed, &[1], &[0]).unwrap();

        assert_eq!(coefficient, 1.0);
        assert_eq!(inverse_coefficient, 1.0);
        assert_eq!(roundtrip, source);
    }

    #[test]
    fn unique_transpose_tree_pair_matches_tensorkit_clockwise_cycle() {
        let source = FusionTreeBlockKey::pair_from_sector_ids(
            [1, 0],
            [1, 0],
            Some(1),
            [false, false],
            [false, false],
            [],
            [],
            [1],
            [1],
        );
        let expected = FusionTreeBlockKey::pair_from_sector_ids(
            [0, 0],
            [1, 1],
            Some(0),
            [false, true],
            [true, false],
            [],
            [],
            [1],
            [1],
        );

        let (transposed, coefficient) =
            unique_transpose_tree_pair(&Z2FusionRule, &source, &[1, 3], &[0, 2]).unwrap();

        assert_eq!(coefficient, 1.0);
        assert_eq!(transposed, expected);
    }

    #[test]
    fn unique_transpose_tree_pair_matches_tensorkit_anticlockwise_cycle() {
        let source = FusionTreeBlockKey::pair_from_sector_ids(
            [1, 0],
            [1, 0],
            Some(1),
            [false, false],
            [false, false],
            [],
            [],
            [1],
            [1],
        );
        let expected = FusionTreeBlockKey::pair_from_sector_ids(
            [1, 1],
            [0, 0],
            Some(0),
            [true, false],
            [false, true],
            [],
            [],
            [1],
            [1],
        );

        let (transposed, coefficient) =
            unique_transpose_tree_pair(&Z2FusionRule, &source, &[2, 0], &[3, 1]).unwrap();

        assert_eq!(coefficient, 1.0);
        assert_eq!(transposed, expected);
    }

    #[test]
    fn unique_transpose_tree_pair_rejects_noncyclic_permutation() {
        let source = FusionTreeBlockKey::pair_from_sector_ids(
            [1, 0],
            [1],
            Some(1),
            [false, false],
            [false],
            [],
            [],
            [1],
            [],
        );

        let err = unique_transpose_tree_pair(&Z2FusionRule, &source, &[0, 2], &[1]).unwrap_err();

        assert_eq!(
            err,
            CoreError::InvalidPermutation {
                permutation: vec![0, 2, 1],
                rank: 3,
            }
        );
    }

    #[test]
    fn block_view_validates_column_major_layout() {
        let data = [0.0; 6];
        let shape = [2, 3];
        let strides = [1, 2];
        let view = BlockView::new(&data, &shape, &strides, 0).unwrap();
        assert_eq!(view.shape(), &[2, 3]);
        assert_eq!(view.strides(), &[1, 2]);
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
    fn block_structure_finds_fusion_tree_subblock_by_key() {
        let first = FusionTreeBlockKey::pair_from_sector_ids(
            [1],
            [1],
            Some(1),
            [false],
            [true],
            [],
            [],
            [],
            [],
        );
        let second = FusionTreeBlockKey::pair_from_sector_ids(
            [0],
            [0],
            Some(0),
            [false],
            [true],
            [],
            [],
            [],
            [],
        );
        let structure = packed_fixture_structure(
            2,
            [
                (BlockKey::from(second.clone()), vec![1, 4]),
                (BlockKey::from(first.clone()), vec![2, 3]),
            ],
        )
        .unwrap();

        let first_block = structure.fusion_tree_block(&first).unwrap();
        let second_block = structure
            .block_by_key(&BlockKey::from(second.clone()))
            .unwrap();

        assert_eq!(first_block.key(), &BlockKey::from(first));
        assert_eq!(first_block.shape(), &[2, 3]);
        assert_eq!(first_block.offset(), 4);
        assert_eq!(second_block.key(), &BlockKey::from(second));
        assert_eq!(second_block.shape(), &[1, 4]);
        assert_eq!(second_block.offset(), 0);
    }

    #[test]
    fn tensormap_subblock_by_tree_returns_matching_view() {
        let first = FusionTreeBlockKey::pair_from_sector_ids(
            [1],
            [1],
            Some(1),
            [false],
            [true],
            [],
            [],
            [],
            [],
        );
        let second = FusionTreeBlockKey::pair_from_sector_ids(
            [0],
            [0],
            Some(0),
            [false],
            [true],
            [],
            [],
            [],
            [],
        );
        let structure = packed_fixture_structure(
            2,
            [
                (BlockKey::from(second.clone()), vec![1, 2]),
                (BlockKey::from(first.clone()), vec![2, 2]),
            ],
        )
        .unwrap();
        let space = TensorMapSpace::<1, 1>::from_dims([3], [3]).unwrap();
        let tensor = TensorMap::<i32, 1, 1>::from_vec_with_structure(
            vec![10, 20, 30, 40, 50, 60],
            space,
            structure,
        )
        .unwrap();

        let first_view = tensor.subblock_by_tree(&first).unwrap();
        let second_view = tensor.block_by_key(&BlockKey::from(second)).unwrap();

        assert_eq!(first_view.shape(), &[2, 2]);
        assert_eq!(first_view.offset(), 2);
        assert_eq!(
            &first_view.data()[first_view.offset()..first_view.offset() + 4],
            &[30, 40, 50, 60]
        );
        assert_eq!(second_view.shape(), &[1, 2]);
        assert_eq!(second_view.offset(), 0);
    }

    #[test]
    fn tensormap_subblock_mut_by_tree_updates_selected_storage() {
        let key = FusionTreeBlockKey::pair_from_sector_ids(
            [1],
            [1],
            Some(1),
            [false],
            [true],
            [],
            [],
            [],
            [],
        );
        let structure = packed_fixture_structure(
            2,
            [
                (BlockKey::sector_ids([0]), vec![1, 2]),
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
    fn subblock_by_tree_reports_missing_fusion_tree_key() {
        let existing = FusionTreeBlockKey::pair_from_sector_ids(
            [0],
            [0],
            Some(0),
            [false],
            [true],
            [],
            [],
            [],
            [],
        );
        let missing = FusionTreeBlockKey::pair_from_sector_ids(
            [1],
            [1],
            Some(1),
            [false],
            [true],
            [],
            [],
            [],
            [],
        );
        let structure =
            packed_fixture_structure(2, [(BlockKey::from(existing), vec![1, 1])]).unwrap();
        let space = TensorMapSpace::<1, 1>::from_dims([1], [1]).unwrap();
        let tensor =
            TensorMap::<f64, 1, 1>::from_vec_with_structure(vec![1.0], space, structure).unwrap();

        let err = tensor.subblock_by_tree(&missing).unwrap_err();

        assert_eq!(
            err,
            CoreError::MissingBlockKey {
                key: BlockKey::from(missing),
            }
        );
    }

    #[test]
    fn public_u1_irrep_roundtrips_compact_ids_and_fuses() {
        let rule = U1FusionRule;
        let charges = [
            U1Irrep::new(-2),
            U1Irrep::new(-1),
            U1Irrep::new(0),
            U1Irrep::new(1),
            U1Irrep::new(2),
        ];
        let ids = charges.map(SectorId::from);

        assert_eq!(
            ids,
            [
                SectorId::new(3),
                SectorId::new(1),
                SectorId::new(0),
                SectorId::new(2),
                SectorId::new(4),
            ]
        );
        for charge in charges {
            assert_eq!(U1Irrep::from_sector_id(charge.sector_id()), Some(charge));
        }
        assert_eq!(rule.vacuum(), U1Irrep::new(0).sector_id());
        assert_eq!(
            rule.dual(U1Irrep::new(3).sector_id()),
            U1Irrep::new(-3).sector_id()
        );
        assert_eq!(
            rule.fusion_channels(U1Irrep::new(-2).sector_id(), U1Irrep::new(5).sector_id())
                .to_vec(),
            vec![U1Irrep::new(3).sector_id()]
        );
    }

    #[test]
    fn product_sector_codec_uses_tensorkit_diagonal_component_order() {
        let expected = [
            (0, 0),
            (0, 1),
            (1, 0),
            (0, 2),
            (1, 1),
            (2, 0),
            (0, 3),
            (1, 2),
            (2, 1),
            (3, 0),
        ];

        for (id, &(left, right)) in expected.iter().enumerate() {
            let encoded = TensorKitProductCodec::encode(SectorId::new(left), SectorId::new(right));
            assert_eq!(encoded, SectorId::new(id));
            assert_eq!(
                TensorKitProductCodec::decode(encoded),
                Some((SectorId::new(left), SectorId::new(right)))
            );
        }
    }

    #[test]
    fn product_sector_api_exposes_only_generic_composition() {
        let pair = product_sector(z2_odd(), u1(2));
        let encoded = pair.sector_id_with::<TensorKitProductCodec>();
        assert_eq!(encoded, TensorKitProductCodec::encode(z2_odd(), u1(2)));
        assert_eq!(pair.left(), &z2_odd());
        assert_eq!(pair.right(), &u1(2));

        let left_rule = product_fusion_rule(FermionParityFusionRule, U1FusionRule);
        let chained_rule = FermionParityFusionRule
            .product(U1FusionRule)
            .product(SU2FusionRule);
        let left_sector = |parity, charge| left_rule.encode_sector(parity, u1(charge));
        let chained_sector = |parity, charge, twice_spin| {
            chained_rule.encode_sector(left_sector(parity, charge), su2(twice_spin))
        };

        let a = chained_sector(z2_odd(), 1, 1);
        let b = chained_sector(z2_odd(), -1, 1);
        let c0 = chained_sector(z2_even(), 0, 0);
        let c2 = chained_sector(z2_even(), 0, 2);

        assert_eq!(chained_rule.fusion_style(), FusionStyleKind::Simple);
        assert_eq!(chained_rule.braiding_style(), BraidingStyleKind::Fermionic);
        assert_eq!(chained_rule.fusion_channels(a, b).to_vec(), vec![c0, c2]);
    }

    #[test]
    fn product_fusion_rule_combines_fermion_parity_and_u1_componentwise() {
        type FpU1Rule = ProductFusionRule<FermionParityFusionRule, U1FusionRule>;
        let rule = FpU1Rule::default();
        let sector = |parity, charge| rule.encode_sector(parity, u1(charge));
        let odd_two = sector(z2_odd(), 2);
        let odd_minus_five = sector(z2_odd(), -5);
        let even_minus_three = sector(z2_even(), -3);

        assert_eq!(rule.fusion_style(), FusionStyleKind::Unique);
        assert_eq!(rule.braiding_style(), BraidingStyleKind::Fermionic);
        assert_eq!(rule.vacuum(), sector(z2_even(), 0));
        assert_eq!(rule.dual(odd_two), sector(z2_odd(), -2));
        assert_eq!(
            rule.fusion_channels(odd_two, odd_minus_five).to_vec(),
            vec![even_minus_three]
        );
        assert_eq!(rule.nsymbol(odd_two, odd_minus_five, even_minus_three), 1);
        assert_eq!(
            rule.r_symbol_scalar(odd_two, odd_minus_five, even_minus_three),
            -1.0
        );
        assert_eq!(rule.sqrt_dim_scalar(odd_two), 1.0);
    }

    #[test]
    fn product_fusion_rule_nested_fz2_u1_su2_channels_and_symbols_match_tensorkit() {
        type FpU1Rule = ProductFusionRule<FermionParityFusionRule, U1FusionRule>;
        type FpU1Su2Rule = ProductFusionRule<FpU1Rule, SU2FusionRule>;
        let left_rule = FpU1Rule::default();
        let rule = FpU1Su2Rule::default();
        let left_sector = |parity, charge| left_rule.encode_sector(parity, u1(charge));
        let sector = |parity, charge, twice_spin| {
            rule.encode_sector(left_sector(parity, charge), su2(twice_spin))
        };

        let a = sector(z2_odd(), 1, 1);
        let b = sector(z2_odd(), -1, 1);
        let c0 = sector(z2_even(), 0, 0);
        let c2 = sector(z2_even(), 0, 2);

        assert_eq!(rule.fusion_style(), FusionStyleKind::Simple);
        assert_eq!(rule.braiding_style(), BraidingStyleKind::Fermionic);
        assert_eq!(rule.dual(a), sector(z2_odd(), -1, 1));
        assert_eq!(rule.fusion_channels(a, b).to_vec(), vec![c0, c2]);
        assert_eq!(rule.r_symbol_scalar(a, b, c0), 1.0);
        assert_eq!(rule.r_symbol_scalar(a, b, c2), -1.0);
        assert!((rule.sqrt_dim_scalar(c2) - 3.0_f64.sqrt()).abs() < 1.0e-12);

        let vacuum_left = left_sector(z2_even(), 0);
        let spin_half = rule.encode_sector(vacuum_left, su2(1));
        let spin_zero = rule.encode_sector(vacuum_left, su2(0));
        assert!(
            (rule.f_symbol_scalar(
                spin_half, spin_half, spin_half, spin_half, spin_zero, spin_zero,
            ) + 0.5)
                .abs()
                < 1.0e-12
        );
    }

    #[test]
    fn product_fusion_tree_homspace_matches_tensorkit_fz2_u1_su2_fixture() {
        type FpU1Rule = ProductFusionRule<FermionParityFusionRule, U1FusionRule>;
        type FpU1Su2Rule = ProductFusionRule<FpU1Rule, SU2FusionRule>;
        let left_rule = FpU1Rule::default();
        let rule = FpU1Su2Rule::default();
        let left_sector = |parity, charge| left_rule.encode_sector(parity, u1(charge));
        let sector = |parity, charge, twice_spin| {
            rule.encode_sector(left_sector(parity, charge), su2(twice_spin))
        };

        let a = sector(z2_odd(), 1, 1);
        let b = sector(z2_odd(), -1, 1);
        let c0 = sector(z2_even(), 0, 0);
        let c1 = sector(z2_even(), 0, 2);
        assert_eq!(a.id(), 43);
        assert_eq!(b.id(), 19);
        assert_eq!(c0.id(), 0);
        assert_eq!(c1.id(), 3);

        let hom = FusionTreeHomSpace::new(
            FusionProductSpace::new([
                SectorLeg::new([(a, 1)], false),
                SectorLeg::new([(b, 1)], false),
            ]),
            FusionProductSpace::new([SectorLeg::new([(c0, 1), (c1, 1)], false)]),
        );
        let keys = hom.fusion_tree_keys(&rule);

        assert_eq!(keys.len(), 2);
        for (key, coupled) in keys.iter().zip([c0, c1]) {
            assert_eq!(key.coupled(), Some(coupled));
            assert_eq!(key.codomain_uncoupled(), &[a, b]);
            assert_eq!(key.domain_uncoupled(), &[coupled]);
            assert_eq!(key.codomain_is_dual(), &[false, false]);
            assert_eq!(key.domain_is_dual(), &[false]);
            assert_eq!(key.codomain_innerlines(), &[]);
            assert_eq!(key.domain_innerlines(), &[]);
            assert_eq!(key.codomain_vertices(), &[SectorId::new(1)]);
            assert_eq!(key.domain_vertices(), &[]);
        }
    }

    #[test]
    fn product_subblock_by_sectors_handles_simple_fusion_channels_without_manual_tree_keys() {
        type FpU1Rule = ProductFusionRule<FermionParityFusionRule, U1FusionRule>;
        type FpU1Su2Rule = ProductFusionRule<FpU1Rule, SU2FusionRule>;
        let left_rule = FpU1Rule::default();
        let rule = FpU1Su2Rule::default();
        let left_sector = |parity, charge| left_rule.encode_sector(parity, u1(charge));
        let sector = |parity, charge, twice_spin| {
            rule.encode_sector(left_sector(parity, charge), su2(twice_spin))
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
        let fusion_space = FusionTensorMapSpace::from_degeneracy_shapes(
            dense,
            hom,
            &rule,
            [vec![1, 1, 1], vec![1, 1, 1]],
        )
        .unwrap();
        let tensor =
            TensorMap::<i32, 2, 1>::from_vec_with_fusion_space(vec![100, 200], fusion_space)
                .unwrap();

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
    fn product_external_domain_sector_is_dualized_componentwise() {
        type FpU1Rule = ProductFusionRule<FermionParityFusionRule, U1FusionRule>;
        let rule = FpU1Rule::default();
        let a = rule.encode_sector(z2_odd(), u1(2));
        let external_domain = rule.dual(a);
        let hom = FusionTreeHomSpace::new(
            FusionProductSpace::new([SectorLeg::new([(a, 1)], false)]),
            FusionProductSpace::new([SectorLeg::new([(a, 1)], false)]),
        );

        let keys = hom
            .fusion_tree_keys_from_external_sectors(&rule, &[a, external_domain])
            .unwrap();
        assert_eq!(keys.len(), 1);
        assert_eq!(keys[0].codomain_uncoupled(), &[a]);
        assert_eq!(keys[0].domain_uncoupled(), &[a]);
        assert_eq!(keys[0].coupled(), Some(a));

        let err = hom
            .fusion_tree_keys_from_external_sectors(&rule, &[a, a])
            .unwrap_err();
        assert_eq!(
            err,
            CoreError::InvalidSector {
                sector: external_domain
            }
        );
    }

    #[test]
    #[should_panic(expected = "Z2 fusion received an invalid sector")]
    fn product_fusion_rule_panics_on_component_invalid_sector_like_existing_rules() {
        type FpU1Rule = ProductFusionRule<FermionParityFusionRule, U1FusionRule>;
        let rule = FpU1Rule::default();
        let invalid_left_component = rule.encode_sector(SectorId::new(2), u1(0));
        let valid = rule.encode_sector(z2_even(), u1(0));

        let _ = rule.fusion_channels(invalid_left_component, valid);
    }

    #[test]
    fn public_su2_irrep_fusion_channels_match_doubled_spin_order() {
        let rule = SU2FusionRule;

        assert_eq!(
            rule.fusion_channels(
                SU2Irrep::from_twice_spin(1).sector_id(),
                SU2Irrep::from_twice_spin(2).sector_id(),
            )
            .to_vec(),
            vec![
                SU2Irrep::from_twice_spin(1).sector_id(),
                SU2Irrep::from_twice_spin(3).sector_id(),
            ]
        );
    }

    #[test]
    fn public_su2_f_and_r_symbols_match_tensorkit_values() {
        let rule = SU2FusionRule;
        let s = |twice_spin| SU2Irrep::from_twice_spin(twice_spin).sector_id();
        let cases = [
            ((1, 1, 1, 1, 0, 0), -0.5),
            ((1, 1, 1, 1, 0, 2), 0.866_025_403_784_438_6),
            ((1, 1, 1, 1, 2, 0), 0.866_025_403_784_438_6),
            ((1, 1, 1, 1, 2, 2), 0.5),
            ((1, 2, 1, 2, 1, 1), -1.0 / 3.0),
            ((2, 2, 2, 2, 0, 2), -0.577_350_269_189_625_7),
            ((2, 2, 2, 2, 2, 2), 0.5),
            ((1, 1, 2, 2, 1, 1), 0.0),
        ];

        for ((a, b, c, d, e, f), expected) in cases {
            let actual = rule.f_symbol_scalar(s(a), s(b), s(c), s(d), s(e), s(f));
            assert!(
                (actual - expected).abs() < 1.0e-12,
                "F({a},{b},{c},{d},{e},{f}) = {actual}, expected {expected}"
            );
        }
        assert_eq!(rule.r_symbol_scalar(s(1), s(1), s(0)), -1.0);
        assert_eq!(rule.r_symbol_scalar(s(1), s(1), s(2)), 1.0);
        assert_eq!(rule.r_symbol_scalar(s(1), s(2), s(0)), 0.0);
    }

    #[derive(Clone, Debug)]
    struct ProbeTreeKey(usize);

    static PROBE_TREE_KEY_EQ_CALLS: std::sync::atomic::AtomicUsize =
        std::sync::atomic::AtomicUsize::new(0);
    static PROBE_TREE_KEY_HASH_CALLS: std::sync::atomic::AtomicUsize =
        std::sync::atomic::AtomicUsize::new(0);

    impl PartialEq for ProbeTreeKey {
        fn eq(&self, other: &Self) -> bool {
            PROBE_TREE_KEY_EQ_CALLS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            self.0 == other.0
        }
    }

    impl Eq for ProbeTreeKey {}

    impl std::hash::Hash for ProbeTreeKey {
        fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
            PROBE_TREE_KEY_HASH_CALLS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            std::hash::Hash::hash(&self.0, state);
        }
    }

    #[test]
    fn fusion_term_accumulator_keeps_singleton_path_and_hashes_multi_terms() {
        PROBE_TREE_KEY_EQ_CALLS.store(0, std::sync::atomic::Ordering::Relaxed);
        PROBE_TREE_KEY_HASH_CALLS.store(0, std::sync::atomic::Ordering::Relaxed);
        let mut singleton = FusionTermAccumulator::new();
        singleton.push(ProbeTreeKey(7), 3usize);
        let singleton_terms = singleton.into_vec();
        assert_eq!(singleton_terms.len(), 1);
        let (singleton_key, singleton_coefficient) = &singleton_terms[0];
        assert_eq!(singleton_key.0, 7);
        assert_eq!(*singleton_coefficient, 3);
        assert_eq!(
            PROBE_TREE_KEY_HASH_CALLS.load(std::sync::atomic::Ordering::Relaxed),
            0
        );

        const DISTINCT: usize = 512;
        const ROUNDS: usize = 4;
        PROBE_TREE_KEY_EQ_CALLS.store(0, std::sync::atomic::Ordering::Relaxed);
        PROBE_TREE_KEY_HASH_CALLS.store(0, std::sync::atomic::Ordering::Relaxed);
        let mut accumulator = FusionTermAccumulator::new();
        for _ in 0..ROUNDS {
            for key in 0..DISTINCT {
                accumulator.push(ProbeTreeKey(key), 1usize);
            }
        }
        let terms = accumulator.into_vec();
        assert_eq!(terms.len(), DISTINCT);
        for (index, (key, coefficient)) in terms.iter().enumerate() {
            assert_eq!(key.0, index);
            assert_eq!(*coefficient, ROUNDS);
        }
        let eq_calls = PROBE_TREE_KEY_EQ_CALLS.load(std::sync::atomic::Ordering::Relaxed);
        assert!(
            eq_calls < DISTINCT * ROUNDS * 8,
            "HashMap-backed accumulation should stay linear; saw {eq_calls} equality checks"
        );
        assert!(
            PROBE_TREE_KEY_HASH_CALLS.load(std::sync::atomic::Ordering::Relaxed) > DISTINCT,
            "multi-term accumulation should use the hash path"
        );
    }

    #[test]
    fn multiplicity_free_su2_braid_expands_innerline_channels() {
        let rule = SU2FusionRule;
        let tree = FusionTreeKey::from_sector_ids(
            [1, 1, 1, 1],
            Some(0),
            [false, false, false, false],
            [0, 1],
            [1, 1, 1],
        );

        let braided =
            multiplicity_free_braid_tree(&rule, &tree, &[0, 2, 1, 3], &[0, 1, 2, 3]).unwrap();

        assert_eq!(braided.len(), 2);
        assert_eq!(braided[0].0.uncoupled(), &[SectorId::new(1); 4]);
        assert_eq!(
            braided[0].0.innerlines(),
            &[SectorId::new(0), SectorId::new(1)]
        );
        assert!((braided[0].1 - 0.5).abs() < 1.0e-12);
        assert_eq!(
            braided[1].0.innerlines(),
            &[SectorId::new(2), SectorId::new(1)]
        );
        assert!((braided[1].1 - 0.866_025_403_784_438_6).abs() < 1.0e-12);
    }

    #[test]
    fn multiplicity_free_su2_repartition_matches_tensorkit_bend_factor() {
        let rule = SU2FusionRule;
        let source = FusionTreeBlockKey::pair_from_sector_ids(
            [1],
            [1],
            Some(1),
            [false],
            [false],
            [],
            [],
            [],
            [],
        );

        let all_codomain = multiplicity_free_repartition_tree_pair(&rule, &source, 2).unwrap();
        assert_eq!(all_codomain.len(), 1);
        assert_eq!(
            all_codomain[0].0.codomain_uncoupled(),
            &[SectorId::new(1); 2]
        );
        assert_eq!(all_codomain[0].0.codomain_is_dual(), &[false, true]);
        assert_eq!(all_codomain[0].0.codomain_innerlines(), &[]);
        assert_eq!(all_codomain[0].0.codomain_vertices(), &[SectorId::new(1)]);
        assert_eq!(
            all_codomain[0].0.codomain_tree().coupled(),
            Some(SectorId::new(0))
        );
        assert_eq!(all_codomain[0].0.domain_uncoupled(), &[]);
        assert_eq!(
            all_codomain[0].0.domain_tree().coupled(),
            Some(SectorId::new(0))
        );
        assert!((all_codomain[0].1 - 2.0_f64.sqrt()).abs() < 1.0e-12);

        let all_domain = multiplicity_free_repartition_tree_pair(&rule, &source, 0).unwrap();
        assert_eq!(all_domain.len(), 1);
        assert_eq!(all_domain[0].0.codomain_uncoupled(), &[]);
        assert_eq!(
            all_domain[0].0.codomain_tree().coupled(),
            Some(SectorId::new(0))
        );
        assert_eq!(all_domain[0].0.domain_uncoupled(), &[SectorId::new(1); 2]);
        assert_eq!(all_domain[0].0.domain_is_dual(), &[false, true]);
        assert!((all_domain[0].1 - 2.0_f64.sqrt()).abs() < 1.0e-12);
    }

    #[test]
    fn multiplicity_free_su2_permute_tree_pair_matches_tensorkit_swap() {
        let rule = SU2FusionRule;
        let source = FusionTreeBlockKey::pair_from_sector_ids(
            [1],
            [1],
            Some(1),
            [false],
            [false],
            [],
            [],
            [],
            [],
        );

        let permuted = multiplicity_free_permute_tree_pair(&rule, &source, &[1], &[0]).unwrap();

        assert_eq!(permuted.len(), 1);
        assert_eq!(permuted[0].0.codomain_uncoupled(), &[SectorId::new(1)]);
        assert_eq!(permuted[0].0.domain_uncoupled(), &[SectorId::new(1)]);
        assert_eq!(permuted[0].0.codomain_is_dual(), &[true]);
        assert_eq!(permuted[0].0.domain_is_dual(), &[true]);
        assert_eq!(
            permuted[0].0.codomain_tree().coupled(),
            Some(SectorId::new(1))
        );
        assert_eq!(
            permuted[0].0.domain_tree().coupled(),
            Some(SectorId::new(1))
        );
        assert!((permuted[0].1 - 1.0).abs() < 1.0e-12);
    }

    fn u1_nonselfdual_tree_pair_fixture() -> FusionTreeBlockKey {
        FusionTreeBlockKey::pair(
            FusionTreeKey::new(
                [u1(1), u1(2)],
                Some(u1(3)),
                [false, false],
                Vec::<SectorId>::new(),
                [SectorId::new(1)],
            ),
            FusionTreeKey::new(
                [u1(3)],
                Some(u1(3)),
                [false],
                Vec::<SectorId>::new(),
                Vec::<SectorId>::new(),
            ),
        )
    }

    #[test]
    fn u1_bendright_dualizes_visible_sector_and_flips_isdual_like_tensorkit() {
        let out = multiplicity_free_bendright_tree_pair(
            &U1FusionRule,
            &u1_nonselfdual_tree_pair_fixture(),
        )
        .unwrap();

        assert_eq!(out.len(), 1);
        assert_eq!(out[0].1, 1.0);
        assert_eq!(out[0].0.codomain_uncoupled(), &[u1(1)]);
        assert_eq!(out[0].0.codomain_tree().coupled(), Some(u1(1)));
        assert_eq!(out[0].0.codomain_is_dual(), &[false]);
        assert_eq!(out[0].0.codomain_innerlines(), &[]);
        assert_eq!(out[0].0.codomain_vertices(), &[]);
        assert_eq!(out[0].0.domain_uncoupled(), &[u1(3), u1(-2)]);
        assert_eq!(out[0].0.domain_tree().coupled(), Some(u1(1)));
        assert_eq!(out[0].0.domain_is_dual(), &[false, true]);
        assert_eq!(out[0].0.domain_innerlines(), &[]);
        assert_eq!(out[0].0.domain_vertices(), &[SectorId::new(1)]);
    }

    #[test]
    fn u1_foldright_dualizes_first_visible_sector_and_flips_isdual_like_tensorkit() {
        let out = multiplicity_free_foldright_tree_pair(
            &U1FusionRule,
            &u1_nonselfdual_tree_pair_fixture(),
        )
        .unwrap();

        assert_eq!(out.len(), 1);
        assert_eq!(out[0].1, 1.0);
        assert_eq!(out[0].0.codomain_uncoupled(), &[u1(2)]);
        assert_eq!(out[0].0.codomain_tree().coupled(), Some(u1(2)));
        assert_eq!(out[0].0.codomain_is_dual(), &[false]);
        assert_eq!(out[0].0.codomain_innerlines(), &[]);
        assert_eq!(out[0].0.codomain_vertices(), &[]);
        assert_eq!(out[0].0.domain_uncoupled(), &[u1(-1), u1(3)]);
        assert_eq!(out[0].0.domain_tree().coupled(), Some(u1(2)));
        assert_eq!(out[0].0.domain_is_dual(), &[true, false]);
        assert_eq!(out[0].0.domain_innerlines(), &[]);
        assert_eq!(out[0].0.domain_vertices(), &[SectorId::new(1)]);
    }

    #[test]
    fn u1_repartition_to_all_domain_matches_tensorkit_nonselfdual_fixture() {
        let out = multiplicity_free_repartition_tree_pair(
            &U1FusionRule,
            &u1_nonselfdual_tree_pair_fixture(),
            0,
        )
        .unwrap();

        assert_eq!(out.len(), 1);
        assert_eq!(out[0].1, 1.0);
        assert_eq!(out[0].0.codomain_uncoupled(), &[]);
        assert_eq!(out[0].0.codomain_tree().coupled(), Some(u1(0)));
        assert_eq!(out[0].0.codomain_is_dual(), &[]);
        assert_eq!(out[0].0.codomain_innerlines(), &[]);
        assert_eq!(out[0].0.codomain_vertices(), &[]);
        assert_eq!(out[0].0.domain_uncoupled(), &[u1(3), u1(-2), u1(-1)]);
        assert_eq!(out[0].0.domain_tree().coupled(), Some(u1(0)));
        assert_eq!(out[0].0.domain_is_dual(), &[false, true, true]);
        assert_eq!(out[0].0.domain_innerlines(), &[u1(1)]);
        assert_eq!(
            out[0].0.domain_vertices(),
            &[SectorId::new(1), SectorId::new(1)]
        );
    }

    #[test]
    fn u1_repartition_to_all_codomain_matches_tensorkit_nonselfdual_fixture() {
        let out = multiplicity_free_repartition_tree_pair(
            &U1FusionRule,
            &u1_nonselfdual_tree_pair_fixture(),
            3,
        )
        .unwrap();

        assert_eq!(out.len(), 1);
        assert_eq!(out[0].1, 1.0);
        assert_eq!(out[0].0.codomain_uncoupled(), &[u1(1), u1(2), u1(-3)]);
        assert_eq!(out[0].0.codomain_tree().coupled(), Some(u1(0)));
        assert_eq!(out[0].0.codomain_is_dual(), &[false, false, true]);
        assert_eq!(out[0].0.codomain_innerlines(), &[u1(3)]);
        assert_eq!(
            out[0].0.codomain_vertices(),
            &[SectorId::new(1), SectorId::new(1)]
        );
        assert_eq!(out[0].0.domain_uncoupled(), &[]);
        assert_eq!(out[0].0.domain_tree().coupled(), Some(u1(0)));
        assert_eq!(out[0].0.domain_is_dual(), &[]);
        assert_eq!(out[0].0.domain_innerlines(), &[]);
        assert_eq!(out[0].0.domain_vertices(), &[]);
    }

    #[test]
    fn u1_transpose_cyclic_23_1_matches_tensorkit_nonselfdual_fixture() {
        let out = multiplicity_free_transpose_tree_pair(
            &U1FusionRule,
            &u1_nonselfdual_tree_pair_fixture(),
            &[1, 2],
            &[0],
        )
        .unwrap();

        assert_eq!(out.len(), 1);
        assert_eq!(out[0].1, 1.0);
        assert_eq!(out[0].0.codomain_uncoupled(), &[u1(2), u1(-3)]);
        assert_eq!(out[0].0.codomain_tree().coupled(), Some(u1(-1)));
        assert_eq!(out[0].0.codomain_is_dual(), &[false, true]);
        assert_eq!(out[0].0.codomain_innerlines(), &[]);
        assert_eq!(out[0].0.codomain_vertices(), &[SectorId::new(1)]);
        assert_eq!(out[0].0.domain_uncoupled(), &[u1(-1)]);
        assert_eq!(out[0].0.domain_tree().coupled(), Some(u1(-1)));
        assert_eq!(out[0].0.domain_is_dual(), &[true]);
        assert_eq!(out[0].0.domain_innerlines(), &[]);
        assert_eq!(out[0].0.domain_vertices(), &[]);
    }

    #[test]
    fn typed_sector_homspace_builds_u1_tree_key() {
        let rule = U1FusionRule;
        let hom = FusionTreeHomSpace::from_sectors([(U1Irrep::new(2), 1)], [(U1Irrep::new(2), 1)]);

        let key = hom
            .unique_fusion_tree_key_from_external_sectors(
                &rule,
                &[U1Irrep::new(2).sector_id(), U1Irrep::new(-2).sector_id()],
            )
            .unwrap();

        assert_eq!(key.codomain_uncoupled(), &[U1Irrep::new(2).sector_id()]);
        assert_eq!(key.domain_uncoupled(), &[U1Irrep::new(2).sector_id()]);
        assert_eq!(key.coupled(), Some(U1Irrep::new(2).sector_id()));
    }

    #[test]
    fn fusion_tensor_space_builds_subblockstructure_from_homspace() {
        let rule = Z2FusionRule;
        let dense = TensorMapSpace::<1, 1>::from_dims([2], [2]).unwrap();
        let hom = FusionTreeHomSpace::new(
            FusionProductSpace::new([SectorLeg::new(
                [(SectorId::new(0), 1), (SectorId::new(1), 3)],
                false,
            )]),
            FusionProductSpace::new([SectorLeg::new(
                [(SectorId::new(0), 2), (SectorId::new(1), 1)],
                false,
            )]),
        );

        let fusion_space = FusionTensorMapSpace::from_degeneracy_shapes(
            dense,
            hom,
            &rule,
            [vec![1, 2], vec![3, 1]],
        )
        .unwrap();

        assert_eq!(fusion_space.subblock_structure().block_count(), 2);
        assert_eq!(fusion_space.required_len().unwrap(), 5);
        assert_eq!(
            fusion_space.subblock_structure().block(0).unwrap().key(),
            &BlockKey::from(FusionTreeBlockKey::pair_from_sector_ids(
                [0],
                [0],
                Some(0),
                [false],
                [false],
                [],
                [],
                [],
                [],
            ))
        );
        assert_eq!(
            fusion_space.subblock_structure().block(1).unwrap().shape(),
            &[3, 1]
        );
    }

    #[test]
    fn fusion_tensor_space_rejects_homspace_rank_mismatch() {
        let rule = Z2FusionRule;
        let dense = TensorMapSpace::<1, 1>::from_dims([2], [2]).unwrap();
        let hom = FusionTreeHomSpace::from_sector_ids([(0, 1), (1, 1)], [(0, 1)]);

        let err = FusionTensorMapSpace::from_degeneracy_shapes(
            dense,
            hom,
            &rule,
            [vec![1, 1], vec![1, 1]],
        )
        .unwrap_err();

        assert_eq!(
            err,
            CoreError::StructureRankMismatch {
                expected: 1,
                actual: 2,
            }
        );
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
        let fusion_space = FusionTensorMapSpace::from_degeneracy_shapes(
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
            FusionTensorMapSpace::from_degeneracy_shapes(dense, hom, &rule, [vec![1, 1]]).unwrap();
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
            FusionTensorMapSpace::from_degeneracy_shapes(dense, hom, &rule, [vec![1, 1]]).unwrap();
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
            FusionTensorMapSpace::from_degeneracy_shapes(dense, hom, &rule, [vec![1, 1]]).unwrap();
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
    fn packed_block_structure_records_rank_offsets_and_required_len() {
        let structure = BlockStructure::packed_column_major(2, [vec![2, 3], vec![1, 4]]).unwrap();

        assert_eq!(structure.rank(), 2);
        assert_eq!(structure.block_count(), 2);
        assert_eq!(structure.sector_structure().block_count(), 2);
        assert_eq!(structure.degeneracy_structure().block_count(), 2);
        let first = structure.block(0).unwrap();
        assert_eq!(first.key(), &BlockKey::ordinal(0));
        assert_eq!(first.shape(), &[2, 3]);
        assert_eq!(first.strides(), &[1, 2]);
        assert_eq!(first.offset(), 0);
        let second = structure.block(1).unwrap();
        assert_eq!(second.key(), &BlockKey::ordinal(1));
        assert_eq!(second.shape(), &[1, 4]);
        assert_eq!(second.strides(), &[1, 1]);
        assert_eq!(second.offset(), 6);
        assert_eq!(structure.required_len().unwrap(), 10);
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
    fn block_structure_rejects_duplicate_keys() {
        let first =
            BlockSpec::column_major_with_key(BlockKey::sector_ids([7]), vec![2, 2], 0).unwrap();
        let second =
            BlockSpec::column_major_with_key(BlockKey::sector_ids([7]), vec![1, 3], 4).unwrap();

        let err = BlockStructure::from_blocks_with_rank(2, vec![first, second]).unwrap_err();

        assert_eq!(
            err,
            CoreError::DuplicateBlockKey {
                key: BlockKey::sector_ids([7])
            }
        );
    }

    #[test]
    fn fusion_tree_group_key_records_external_sector_tuples_and_duality() {
        let group = FusionTreeGroupKey::from_sector_ids([2, 3], [5], [false, true], [true]);

        assert_eq!(
            group.codomain_uncoupled(),
            &[SectorId::new(2), SectorId::new(3)]
        );
        assert_eq!(group.domain_uncoupled(), &[SectorId::new(5)]);
        assert_eq!(group.codomain_is_dual(), &[false, true]);
        assert_eq!(group.domain_is_dual(), &[true]);

        let same = FusionTreeGroupKey::new(
            [SectorId::new(2), SectorId::new(3)],
            [SectorId::new(5)],
            [false, true],
            [true],
        );
        assert_eq!(group, same);
    }

    #[test]
    fn fusion_tree_block_key_records_tensorkit_subblock_pair_fields() {
        let key = FusionTreeBlockKey::pair_from_sector_ids(
            [2, 3],
            [5, 7],
            Some(11),
            [false, true],
            [true, false],
            [13],
            [17],
            [19, 23],
            [29, 31],
        );

        assert_eq!(
            key.codomain_uncoupled(),
            &[SectorId::new(2), SectorId::new(3)]
        );
        assert_eq!(
            key.domain_uncoupled(),
            &[SectorId::new(5), SectorId::new(7)]
        );
        assert_eq!(key.coupled(), Some(SectorId::new(11)));
        assert_eq!(key.codomain_is_dual(), &[false, true]);
        assert_eq!(key.domain_is_dual(), &[true, false]);
        assert_eq!(key.codomain_innerlines(), &[SectorId::new(13)]);
        assert_eq!(key.domain_innerlines(), &[SectorId::new(17)]);
        assert_eq!(
            key.codomain_vertices(),
            &[SectorId::new(19), SectorId::new(23)]
        );
        assert_eq!(
            key.domain_vertices(),
            &[SectorId::new(29), SectorId::new(31)]
        );

        let group = key.group_key();
        assert_eq!(group.codomain_uncoupled(), key.codomain_uncoupled());
        assert_eq!(group.domain_uncoupled(), key.domain_uncoupled());
        assert_eq!(group.codomain_is_dual(), key.codomain_is_dual());
        assert_eq!(group.domain_is_dual(), key.domain_is_dual());
    }

    #[test]
    fn fusion_tree_homspace_generates_canonical_coupled_sector_order() {
        let rule = BranchingMultiplicityFreeRule;
        let hom = FusionTreeHomSpace::from_sector_ids([(1, 1), (1, 1)], [(1, 1), (1, 1)]);

        let keys = hom.fusion_tree_keys(&rule);

        assert_eq!(keys.len(), 2);
        assert_eq!(keys[0].coupled(), Some(SectorId::new(0)));
        assert_eq!(keys[1].coupled(), Some(SectorId::new(2)));
        assert_eq!(
            keys[0].codomain_uncoupled(),
            &[SectorId::new(1), SectorId::new(1)]
        );
        assert_eq!(
            keys[0].domain_uncoupled(),
            &[SectorId::new(1), SectorId::new(1)]
        );
        assert!(keys[0].codomain_innerlines().is_empty());
        assert!(keys[0].domain_innerlines().is_empty());
        assert_eq!(keys[0].codomain_vertices(), &[SectorId::new(1)]);
        assert_eq!(keys[0].domain_vertices(), &[SectorId::new(1)]);

        let sector = hom.sector_structure(&rule).unwrap();
        let groups = sector.fusion_tree_groups();
        assert_eq!(groups.len(), 1);
        assert_eq!(groups[0].block_indices(), &[0, 1]);
        assert_eq!(
            groups[0].group_key(),
            &FusionTreeGroupKey::from_sector_ids([1, 1], [1, 1], [false, false], [false, false])
        );
    }

    #[test]
    fn unique_homspace_builds_subblock_key_from_external_sectors() {
        let rule = Z2FusionRule;
        let hom = FusionTreeHomSpace::from_sector_ids([(1, 1)], [(1, 1)]);

        let key = hom
            .unique_fusion_tree_key_from_external_sectors(
                &rule,
                &[SectorId::new(1), SectorId::new(1)],
            )
            .unwrap();

        assert_eq!(key.codomain_uncoupled(), &[SectorId::new(1)]);
        assert_eq!(key.domain_uncoupled(), &[SectorId::new(1)]);
        assert_eq!(key.coupled(), Some(SectorId::new(1)));
        assert_eq!(key.codomain_is_dual(), &[false]);
        assert_eq!(key.domain_is_dual(), &[false]);
    }

    #[test]
    fn unique_homspace_dualizes_domain_external_sectors_like_tensorkit() {
        let rule = Z4PointedRule;
        let hom = FusionTreeHomSpace::new(
            FusionProductSpace::new([SectorLeg::new([(SectorId::new(1), 1)], false)]),
            FusionProductSpace::new([SectorLeg::new([(SectorId::new(1), 1)], false)]),
        );

        let key = hom
            .unique_fusion_tree_key_from_external_sectors(
                &rule,
                &[SectorId::new(1), SectorId::new(3)],
            )
            .unwrap();

        assert_eq!(key.codomain_uncoupled(), &[SectorId::new(1)]);
        assert_eq!(key.domain_uncoupled(), &[SectorId::new(1)]);
        assert_eq!(key.coupled(), Some(SectorId::new(1)));
    }

    #[test]
    fn fusion_tree_block_key_external_sectors_restore_visible_domain_sector() {
        let rule = Z4PointedRule;
        let hom = FusionTreeHomSpace::new(
            FusionProductSpace::new([SectorLeg::new([(SectorId::new(1), 1)], true)]),
            FusionProductSpace::new([SectorLeg::new([(SectorId::new(1), 1)], false)]),
        );
        let key = hom
            .unique_fusion_tree_key_from_external_sectors(
                &rule,
                &[SectorId::new(1), SectorId::new(3)],
            )
            .unwrap();

        assert_eq!(key.codomain_uncoupled(), &[SectorId::new(1)]);
        assert_eq!(key.domain_uncoupled(), &[SectorId::new(1)]);
        assert_eq!(
            key.external_sectors(&rule),
            vec![SectorId::new(1), SectorId::new(3)]
        );
        assert_eq!(key.external_is_dual(), vec![true, false]);
    }

    #[test]
    fn fusion_tree_homspace_compose_matches_nonselfdual_domain_convention() {
        // TensorKit: `A * B` needs `domain(A) == codomain(B)` as spaces, so
        // the stored legs pair verbatim even for non-self-dual sectors
        // (Julia check: `rand(U1Space(0=>1,1=>1) ← same) * itself` works).
        let rule = U1FusionRule;
        let physical = u1(1);
        let lhs = FusionTreeHomSpace::new(
            FusionProductSpace::new([SectorLeg::new([(u1(2), 1)], false)]),
            FusionProductSpace::new([SectorLeg::new([(physical, 1)], false)]),
        );
        let rhs = FusionTreeHomSpace::new(
            FusionProductSpace::new([SectorLeg::new([(physical, 1)], false)]),
            FusionProductSpace::new([SectorLeg::new([(u1(3), 1)], false)]),
        );

        let composed = FusionTreeHomSpace::compose(&rule, &lhs, &rhs).unwrap();

        assert_eq!(composed.codomain().legs()[0].sectors(), &[u1(2)]);
        assert_eq!(composed.domain().legs()[0].sectors(), &[u1(3)]);
    }

    #[test]
    fn fusion_tree_homspace_select_dualizes_axes_like_tensorkit() {
        let rule = U1FusionRule;
        let hom = FusionTreeHomSpace::new(
            FusionProductSpace::new([
                SectorLeg::new([(u1(1), 1)], false),
                SectorLeg::new([(u1(2), 1)], true),
            ]),
            FusionProductSpace::new([
                SectorLeg::new([(u1(3), 1)], false),
                SectorLeg::new([(u1(-5), 1)], true),
            ]),
        );

        let selected = hom.select(&rule, &[2, 0], &[1, 3]).unwrap();

        assert_eq!(selected.codomain().legs()[0].sectors(), &[u1(-3)]);
        assert!(selected.codomain().legs()[0].is_dual());
        assert_eq!(selected.codomain().legs()[1].sectors(), &[u1(1)]);
        assert!(!selected.codomain().legs()[1].is_dual());
        assert_eq!(selected.domain().legs()[0].sectors(), &[u1(-2)]);
        assert!(!selected.domain().legs()[0].is_dual());
        assert_eq!(selected.domain().legs()[1].sectors(), &[u1(-5)]);
        assert!(selected.domain().legs()[1].is_dual());
    }

    #[test]
    fn fusion_tree_homspace_permute_requires_full_axis_permutation() {
        let rule = U1FusionRule;
        let hom = FusionTreeHomSpace::from_sectors([(u1(0), 1), (u1(1), 1)], [(u1(2), 1)]);

        let err = hom.permute(&rule, &[0], &[2]).unwrap_err();
        assert_eq!(
            err,
            CoreError::InvalidPermutation {
                permutation: vec![0, 2],
                rank: 3,
            }
        );

        let err = hom.permute(&rule, &[0, 0], &[2]).unwrap_err();
        assert_eq!(
            err,
            CoreError::InvalidPermutation {
                permutation: vec![0, 0, 2],
                rank: 3,
            }
        );
    }

    #[test]
    fn fusion_tree_homspace_tensorcontract_preserves_canonical_compose() {
        let rule = U1FusionRule;
        let lhs = FusionTreeHomSpace::new(
            FusionProductSpace::new([SectorLeg::new([(u1(2), 1)], false)]),
            FusionProductSpace::new([SectorLeg::new([(u1(1), 1)], false)]),
        );
        let rhs = FusionTreeHomSpace::new(
            FusionProductSpace::new([SectorLeg::new([(u1(1), 1)], false)]),
            FusionProductSpace::new([SectorLeg::new([(u1(3), 1)], false)]),
        );

        let expected = FusionTreeHomSpace::compose(&rule, &lhs, &rhs).unwrap();
        let actual =
            FusionTreeHomSpace::tensorcontract_homspace(&rule, &lhs, &rhs, &[1], &[0], &[0, 1], 1)
                .unwrap();

        assert_eq!(actual, expected);
    }

    #[test]
    fn fusion_tree_homspace_tensorcontract_matches_tensorkit_structural_formula() {
        let rule = U1FusionRule;
        let lhs = FusionTreeHomSpace::new(
            FusionProductSpace::new([SectorLeg::new([(u1(1), 1)], false)]),
            FusionProductSpace::new([SectorLeg::new([(u1(5), 1)], false)]),
        );
        let rhs = FusionTreeHomSpace::new(
            FusionProductSpace::new([SectorLeg::new([(u1(7), 1)], false)]),
            FusionProductSpace::new([SectorLeg::new([(u1(1), 1)], false)]),
        );

        let lhs_permuted = lhs.permute(&rule, &[1], &[0]).unwrap();
        let rhs_permuted = rhs.permute(&rule, &[1], &[0]).unwrap();
        let expected = FusionTreeHomSpace::compose(&rule, &lhs_permuted, &rhs_permuted)
            .unwrap()
            .permute(&rule, &[0], &[1])
            .unwrap();
        let actual =
            FusionTreeHomSpace::tensorcontract_homspace(&rule, &lhs, &rhs, &[0], &[1], &[0, 1], 1)
                .unwrap();

        assert_eq!(actual, expected);
        assert_eq!(actual.codomain().legs()[0].sectors(), &[u1(-5)]);
        assert!(actual.codomain().legs()[0].is_dual());
        assert_eq!(actual.domain().legs()[0].sectors(), &[u1(-7)]);
        assert!(actual.domain().legs()[0].is_dual());
    }

    #[test]
    fn fusion_tree_homspace_tensorcontract_accepts_output_permutation_structurally() {
        let rule = U1FusionRule;
        let lhs = FusionTreeHomSpace::new(
            FusionProductSpace::new([SectorLeg::new([(u1(2), 1)], false)]),
            FusionProductSpace::new([SectorLeg::new([(u1(1), 1)], false)]),
        );
        let rhs = FusionTreeHomSpace::new(
            FusionProductSpace::new([SectorLeg::new([(u1(1), 1)], false)]),
            FusionProductSpace::new([SectorLeg::new([(u1(3), 1)], false)]),
        );

        let composed =
            FusionTreeHomSpace::tensorcontract_homspace(&rule, &lhs, &rhs, &[1], &[0], &[1, 0], 1)
                .unwrap();
        assert_eq!(composed.codomain().len(), 1);
        assert_eq!(composed.domain().len(), 1);
        assert_eq!(composed.codomain().legs()[0].sectors(), &[u1(-3)]);
        assert!(composed.codomain().legs()[0].is_dual());
        assert_eq!(composed.domain().legs()[0].sectors(), &[u1(-2)]);
        assert!(composed.domain().legs()[0].is_dual());
    }

    #[test]
    fn fusion_tree_homspace_compose_rejects_unmatched_contracted_sector() {
        // Pairing a domain leg with the *dual* codomain leg is a
        // SpaceMismatch in TensorKit (`(X ← V) * (V' ← Y)` fails).
        let rule = U1FusionRule;
        let lhs = FusionTreeHomSpace::new(
            FusionProductSpace::new([SectorLeg::new([(u1(0), 1)], false)]),
            FusionProductSpace::new([SectorLeg::new([(u1(1), 1)], false)]),
        );
        let rhs = FusionTreeHomSpace::new(
            FusionProductSpace::new([SectorLeg::new([(rule.dual(u1(1)), 1)], false)]),
            FusionProductSpace::new([SectorLeg::new([(u1(0), 1)], false)]),
        );

        let err = FusionTreeHomSpace::compose(&rule, &lhs, &rhs).unwrap_err();

        assert_eq!(
            err,
            CoreError::SectorMismatch {
                expected: u1(1),
                actual: rule.dual(u1(1)),
            }
        );
    }

    #[test]
    fn unique_homspace_rejects_invalid_external_sector_tuple() {
        let rule = Z4PointedRule;
        let hom = FusionTreeHomSpace::new(
            FusionProductSpace::new([SectorLeg::new([(SectorId::new(1), 1)], false)]),
            FusionProductSpace::new([SectorLeg::new([(SectorId::new(1), 1)], false)]),
        );

        let err = hom
            .unique_fusion_tree_key_from_external_sectors(
                &rule,
                &[SectorId::new(1), SectorId::new(2)],
            )
            .unwrap_err();

        assert_eq!(
            err,
            CoreError::InvalidSector {
                sector: SectorId::new(2),
            }
        );
    }

    #[test]
    fn fusion_tree_homspace_generates_innerline_paths_for_simple_fusion() {
        let rule = BranchingMultiplicityFreeRule;
        let hom = FusionTreeHomSpace::from_sector_ids([(1, 1), (1, 1), (1, 1)], [(1, 1)]);

        let keys = hom.fusion_tree_keys(&rule);

        assert_eq!(keys.len(), 2);
        assert_eq!(keys[0].coupled(), Some(SectorId::new(1)));
        assert_eq!(keys[1].coupled(), Some(SectorId::new(1)));
        assert_eq!(keys[0].codomain_innerlines(), &[SectorId::new(0)]);
        assert_eq!(keys[1].codomain_innerlines(), &[SectorId::new(2)]);
        assert_eq!(
            keys[0].codomain_vertices(),
            &[SectorId::new(1), SectorId::new(1)]
        );
        assert!(keys[0].domain_innerlines().is_empty());
        assert!(keys[0].domain_vertices().is_empty());
        assert_eq!(keys[0].domain_uncoupled(), &[SectorId::new(1)]);

        let groups = hom.fusion_tree_groups(&rule).unwrap();
        assert_eq!(groups.len(), 1);
        assert_eq!(groups[0].block_indices(), &[0, 1]);
    }

    #[test]
    fn fusion_tree_homspace_matches_tensorkit_z2_fusiontreelist_order() {
        let rule = Z2FusionRule;
        let leg = || SectorLeg::new([(SectorId::new(0), 1), (SectorId::new(1), 1)], false);
        let hom = FusionTreeHomSpace::new(
            FusionProductSpace::new([leg(), leg()]),
            FusionProductSpace::new([leg(), leg()]),
        );

        let keys = hom.fusion_tree_keys(&rule);

        // TensorKit.jl 6Camk:
        // V=Vect[Z2Irrep](0=>1,1=>1); W=(V⊗V)←(V⊗V);
        // [(f1.uncoupled, f2.uncoupled, f1.coupled) for (f1,f2) in fusiontrees(W)]
        assert_eq!(
            fusion_tree_pair_order(&keys),
            vec![
                (vec![0, 0], vec![0, 0], 0),
                (vec![1, 1], vec![0, 0], 0),
                (vec![0, 0], vec![1, 1], 0),
                (vec![1, 1], vec![1, 1], 0),
                (vec![1, 0], vec![1, 0], 1),
                (vec![0, 1], vec![1, 0], 1),
                (vec![1, 0], vec![0, 1], 1),
                (vec![0, 1], vec![0, 1], 1),
            ]
        );

        let groups = hom.fusion_tree_groups(&rule).unwrap();
        assert_eq!(groups.len(), keys.len());
        for (index, group) in groups.iter().enumerate() {
            assert_eq!(group.block_indices(), &[index]);
        }
    }

    #[test]
    fn fusion_tree_key_cache_hits_across_degeneracy_and_keeps_dual_signature() {
        let rule = SU2FusionRule;
        let mk_leg = |degeneracy| {
            SectorLeg::new(
                [
                    (SU2Irrep::from_twice_spin(0).sector_id(), degeneracy),
                    (SU2Irrep::from_twice_spin(1).sector_id(), degeneracy + 1),
                ],
                false,
            )
        };
        let hom_small = FusionTreeHomSpace::new(
            FusionProductSpace::new([mk_leg(1), mk_leg(1)]),
            FusionProductSpace::new([mk_leg(1)]),
        );
        let hom_large = FusionTreeHomSpace::new(
            FusionProductSpace::new([mk_leg(4), mk_leg(4)]),
            FusionProductSpace::new([mk_leg(4)]),
        );

        let small_layout = hom_small.cached_fusion_tree_layout(&rule);
        let large_layout = hom_large.cached_fusion_tree_layout(&rule);
        assert!(Arc::ptr_eq(&small_layout, &large_layout));
        assert_eq!(small_layout.keys.as_ref(), large_layout.keys.as_ref());

        let dual_hom = FusionTreeHomSpace::new(
            FusionProductSpace::new([mk_leg(1).dual(&rule), mk_leg(1)]),
            FusionProductSpace::new([mk_leg(1)]),
        );
        let dual_layout = dual_hom.cached_fusion_tree_layout(&rule);
        assert!(!Arc::ptr_eq(&small_layout, &dual_layout));
        assert_ne!(small_layout.keys.as_ref(), dual_layout.keys.as_ref());
    }

    #[test]
    fn fusion_tree_homspace_matches_tensorkit_su2_simple_order() {
        let rule = SU2FusionRule;
        let leg = || {
            SectorLeg::new(
                [
                    (SectorId::new(0), 1),
                    (SectorId::new(1), 1),
                    (SectorId::new(2), 1),
                ],
                false,
            )
        };
        let hom = FusionTreeHomSpace::new(
            FusionProductSpace::new([leg(), leg()]),
            FusionProductSpace::new([leg()]),
        );

        let keys = hom.fusion_tree_keys(&rule);

        // TensorKit.jl 6Camk with sector id = twice spin:
        // V=Vect[SU2Irrep](0=>1,1//2=>1,1=>1); W=(V⊗V)←V;
        // [(2f1.uncoupled, 2f2.uncoupled, 2f1.coupled) for (f1,f2) in fusiontrees(W)]
        assert_eq!(
            fusion_tree_pair_order(&keys),
            vec![
                (vec![0, 0], vec![0], 0),
                (vec![1, 1], vec![0], 0),
                (vec![2, 2], vec![0], 0),
                (vec![1, 0], vec![1], 1),
                (vec![0, 1], vec![1], 1),
                (vec![2, 1], vec![1], 1),
                (vec![1, 2], vec![1], 1),
                (vec![2, 0], vec![2], 2),
                (vec![1, 1], vec![2], 2),
                (vec![0, 2], vec![2], 2),
                (vec![2, 2], vec![2], 2),
            ]
        );
        assert!(keys
            .iter()
            .all(|key| key.codomain_vertices() == [SectorId::new(1)]));
        assert!(keys.iter().all(|key| key.domain_vertices().is_empty()));
    }

    #[test]
    fn braid_tree_pair_block_matches_per_source() {
        use std::collections::BTreeMap;
        let rule = SU2FusionRule;
        let leg = || {
            SectorLeg::new(
                [
                    (SectorId::new(0), 1),
                    (SectorId::new(1), 1),
                    (SectorId::new(2), 1),
                ],
                false,
            )
        };
        // (V⊗V⊗V) ← V spans many uncoupled blocks; test each block.
        let hom = FusionTreeHomSpace::new(
            FusionProductSpace::new([leg(), leg(), leg()]),
            FusionProductSpace::new([leg()]),
        );
        let keys = hom.fusion_tree_keys(&rule);

        // Group source tree-pairs by their uncoupled block (the batching unit).
        let mut blocks: BTreeMap<Vec<usize>, Vec<FusionTreeBlockKey>> = BTreeMap::new();
        for key in &keys {
            let tag: Vec<usize> = key
                .codomain_tree()
                .uncoupled()
                .iter()
                .chain(key.domain_tree().uncoupled())
                .map(|s| s.id())
                .collect();
            blocks.entry(tag).or_default().push(key.clone());
        }

        // Global leg indices: codomain legs 0,1,2 and domain leg 3. Reverse the
        // codomain, keep the domain leg in place.
        let codomain_permutation = [2usize, 1, 0];
        let domain_permutation = [3usize];
        let mut checked_blocks = 0;
        for src_keys in blocks.values() {
            let batched = multiplicity_free_permute_tree_pair_block(
                &rule,
                src_keys,
                &codomain_permutation,
                &domain_permutation,
            )
            .unwrap();
            assert_eq!(batched.len(), src_keys.len());
            for (src, batched_rows) in src_keys.iter().zip(&batched) {
                let per_source = multiplicity_free_permute_tree_pair(
                    &rule,
                    src,
                    &codomain_permutation,
                    &domain_permutation,
                )
                .unwrap();
                // Compare as key -> coefficient maps within double-precision tol.
                let mut want: BTreeMap<FusionTreeBlockKey, f64> = BTreeMap::new();
                for (k, c) in &per_source {
                    *want.entry(k.clone()).or_insert(0.0) += c;
                }
                let mut got: BTreeMap<FusionTreeBlockKey, f64> = BTreeMap::new();
                for (k, c) in batched_rows {
                    *got.entry(k.clone()).or_insert(0.0) += c;
                }
                assert_eq!(
                    want.keys().collect::<Vec<_>>(),
                    got.keys().collect::<Vec<_>>(),
                    "destination trees differ for a source in block"
                );
                for (k, wc) in &want {
                    let gc = got[k];
                    assert!(
                        (wc - gc).abs() <= 1e-12 * (1.0 + wc.abs()),
                        "coefficient mismatch {wc} vs {gc}"
                    );
                }
            }
            checked_blocks += 1;
        }
        assert!(checked_blocks > 0, "expected at least one block");
    }

    #[test]
    fn fusion_tree_homspace_matches_tensorkit_su2_innerline_order() {
        let rule = SU2FusionRule;
        let hom = FusionTreeHomSpace::from_sector_ids([(1, 1), (1, 1), (1, 1)], [(1, 1)]);

        let keys = hom.fusion_tree_keys(&rule);

        // TensorKit.jl 6Camk with sector id = twice spin:
        // V=Vect[SU2Irrep](1//2=>1); W=(V⊗V⊗V)←V;
        // codomain innerlines for fusiontrees(W) are [0], then [2].
        assert_eq!(keys.len(), 2);
        assert_eq!(keys[0].codomain_innerlines(), &[SectorId::new(0)]);
        assert_eq!(keys[1].codomain_innerlines(), &[SectorId::new(2)]);
        assert_eq!(
            fusion_tree_pair_order(&keys),
            vec![(vec![1, 1, 1], vec![1], 1), (vec![1, 1, 1], vec![1], 1),]
        );

        let groups = hom.fusion_tree_groups(&rule).unwrap();
        assert_eq!(groups.len(), 1);
        assert_eq!(groups[0].block_indices(), &[0, 1]);
    }

    #[test]
    fn fusion_tree_homspace_external_sectors_preserve_su2_simple_innerline_order() {
        let rule = SU2FusionRule;
        let half = SectorId::new(1);
        let hom = FusionTreeHomSpace::from_sector_ids([(1, 1), (1, 1), (1, 1)], [(1, 1)]);

        let keys = hom
            .fusion_tree_keys_from_external_sectors(&rule, &[half, half, half, half])
            .unwrap();

        assert_eq!(keys.len(), 2);
        assert_eq!(keys[0].codomain_uncoupled(), &[half, half, half]);
        assert_eq!(keys[0].domain_uncoupled(), &[half]);
        assert_eq!(keys[0].codomain_innerlines(), &[SectorId::new(0)]);
        assert_eq!(keys[1].codomain_innerlines(), &[SectorId::new(2)]);
        assert_eq!(
            fusion_tree_pair_order(&keys),
            vec![(vec![1, 1, 1], vec![1], 1), (vec![1, 1, 1], vec![1], 1),]
        );
    }

    #[test]
    fn tensormap_subblocks_by_sectors_returns_all_su2_simple_innerline_blocks() {
        let rule = SU2FusionRule;
        let half = SectorId::new(1);
        let dense = TensorMapSpace::<3, 1>::from_dims([1, 1, 1], [1]).unwrap();
        let hom = FusionTreeHomSpace::from_sector_ids([(1, 1), (1, 1), (1, 1)], [(1, 1)]);
        let fusion_space = FusionTensorMapSpace::from_degeneracy_shapes(
            dense,
            hom,
            &rule,
            [vec![1, 1, 1, 1], vec![1, 1, 1, 1]],
        )
        .unwrap();
        let tensor =
            TensorMap::<i32, 3, 1>::from_vec_with_fusion_space(vec![11, 22], fusion_space).unwrap();

        let blocks = tensor
            .subblocks_by_sectors(&rule, &[half, half, half, half])
            .unwrap();

        assert_eq!(blocks.len(), 2);
        assert_eq!(blocks[0].offset(), 0);
        assert_eq!(blocks[0].data()[blocks[0].offset()], 11);
        assert_eq!(blocks[1].offset(), 1);
        assert_eq!(blocks[1].data()[blocks[1].offset()], 22);

        let err = tensor
            .subblock_by_sectors(&rule, &[half, half, half, half])
            .unwrap_err();
        assert_eq!(
            err,
            CoreError::BlockCountMismatch {
                expected: 1,
                actual: 2,
            }
        );
    }

    #[test]
    fn fusion_tree_homspace_uses_tensorkit_parent_iterator_order_not_ord_sort() {
        let rule = UnsortedFusionIteratorOrderRule;
        let hom = FusionTreeHomSpace::from_sector_ids([(1, 1), (1, 1), (1, 1)], [(1, 1)]);

        let keys = hom.fusion_tree_keys(&rule);

        // TensorKit rank >= 3 iterator picks the parent line from
        // `coupled ⊗ dual(last)` order. This toy rule returns 1 ⊗ 1 as [2, 0],
        // deliberately opposite to `SectorId` Ord, so an Ord-based replay would
        // produce [0], [2].
        assert_eq!(keys.len(), 2);
        assert_eq!(keys[0].codomain_innerlines(), &[SectorId::new(2)]);
        assert_eq!(keys[1].codomain_innerlines(), &[SectorId::new(0)]);
    }

    #[test]
    fn fusion_tree_homspace_uses_visible_dual_space_sector_label_like_tensorkit() {
        let rule = U1FusionRule;
        let minus_one = U1Irrep::new(-1);
        let hom = FusionTreeHomSpace::new(
            FusionProductSpace::new([SectorLeg::new([(minus_one, 1)], true)]),
            FusionProductSpace::new([SectorLeg::new([(minus_one, 1)], false)]),
        );

        let keys = hom.fusion_tree_keys(&rule);

        // TensorKit:
        // collect(sectors(Vect[U1Irrep](1=>1)')) == [U1Irrep(-1)]
        // fusiontrees((U1Irrep(-1),), U1Irrep(-1), (true,)) keeps uncoupled = -1.
        assert_eq!(keys.len(), 1);
        assert_eq!(keys[0].coupled(), Some(minus_one.into()));
        assert_eq!(keys[0].codomain_uncoupled(), &[minus_one.into()]);
        assert_eq!(keys[0].codomain_is_dual(), &[true]);
        assert_eq!(keys[0].domain_uncoupled(), &[minus_one.into()]);
        assert_eq!(keys[0].domain_is_dual(), &[false]);
    }

    #[test]
    fn fusion_tree_homspace_does_not_dualize_selected_dual_leg_again() {
        let rule = BranchingMultiplicityFreeRule;
        let hom = FusionTreeHomSpace::new(
            FusionProductSpace::new([SectorLeg::new([(SectorId::new(1), 1)], true)]),
            FusionProductSpace::from_sector_ids([(1, 1)]),
        );

        let keys = hom.fusion_tree_keys(&rule);

        assert_eq!(keys.len(), 1);
        assert_eq!(keys[0].coupled(), Some(SectorId::new(1)));
        assert_eq!(keys[0].codomain_uncoupled(), &[SectorId::new(1)]);
        assert_eq!(keys[0].codomain_is_dual(), &[true]);
        assert_eq!(keys[0].domain_uncoupled(), &[SectorId::new(1)]);
        assert_eq!(keys[0].domain_is_dual(), &[false]);
    }

    #[test]
    fn fusion_tree_homspace_fusionblocks_follow_domain_outer_codomain_inner_order() {
        let rule = BranchingMultiplicityFreeRule;
        let hom = FusionTreeHomSpace::new(
            FusionProductSpace::new([
                SectorLeg::new([(SectorId::new(1), 1), (SectorId::new(2), 1)], false),
                SectorLeg::new([(SectorId::new(1), 1)], false),
            ]),
            FusionProductSpace::new([SectorLeg::new(
                [(SectorId::new(1), 1), (SectorId::new(2), 1)],
                false,
            )]),
        );

        let groups = hom.fusion_tree_groups(&rule).unwrap();

        assert_eq!(groups.len(), 2);
        assert_eq!(
            groups[0].group_key(),
            &FusionTreeGroupKey::from_sector_ids([2, 1], [1], [false, false], [false])
        );
        assert_eq!(
            groups[1].group_key(),
            &FusionTreeGroupKey::from_sector_ids([1, 1], [2], [false, false], [false])
        );
    }

    #[test]
    fn fusion_tree_groups_preserve_structure_order_and_ignore_internal_tree_data() {
        let first = BlockKey::from(FusionTreeBlockKey::pair_from_sector_ids(
            [10, 20],
            [30],
            Some(5),
            [false, true],
            [true],
            [101],
            [201],
            [301, 302],
            [401],
        ));
        let second = BlockKey::from(FusionTreeBlockKey::pair_from_sector_ids(
            [1],
            [2, 3],
            Some(4),
            [true],
            [false, true],
            [],
            [202],
            [303],
            [402, 403],
        ));
        let same_group_as_first = BlockKey::from(FusionTreeBlockKey::pair_from_sector_ids(
            [10, 20],
            [30],
            Some(6),
            [false, true],
            [true],
            [102],
            [203],
            [304, 305],
            [404],
        ));

        let keys = vec![first.clone(), second.clone(), same_group_as_first.clone()];
        let sector = SectorStructure::from_keys(2, keys.clone()).unwrap();
        let block_structure =
            packed_fixture_structure(2, keys.into_iter().map(|key| (key, vec![1, 1]))).unwrap();

        let sector_groups = sector.fusion_tree_groups();
        let block_groups = block_structure.fusion_tree_groups();
        assert_eq!(sector_groups, block_groups);
        assert_eq!(sector_groups.len(), 2);
        assert_eq!(sector_groups[0].block_indices(), &[0, 2]);
        assert_eq!(sector_groups[1].block_indices(), &[1]);
        assert_eq!(
            sector_groups[0].group_key(),
            &FusionTreeGroupKey::from_sector_ids([10, 20], [30], [false, true], [true])
        );
        assert_eq!(
            sector_groups[1].group_key(),
            &FusionTreeGroupKey::from_sector_ids([1], [2, 3], [true], [false, true])
        );
    }

    #[test]
    fn fusion_tree_groups_ignore_dense_blocks() {
        let key = BlockKey::from(FusionTreeBlockKey::pair_from_sector_ids(
            [7],
            [8],
            Some(9),
            [false],
            [true],
            [],
            [],
            [],
            [],
        ));
        let sector = SectorStructure::from_keys(2, [BlockKey::trivial(), key]).unwrap();
        let groups = sector.fusion_tree_groups();

        assert_eq!(groups.len(), 1);
        assert_eq!(groups[0].block_indices(), &[1]);
        assert_eq!(
            groups[0].group_key(),
            &FusionTreeGroupKey::from_sector_ids([7], [8], [false], [true])
        );

        let dense = BlockStructure::trivial(&[2, 3]).unwrap();
        assert!(dense.fusion_tree_groups().is_empty());
    }

    #[test]
    fn block_structure_separates_sector_and_degeneracy_data() {
        let sector = SectorStructure::from_keys(
            2,
            [BlockKey::sector_ids([0, 1]), BlockKey::sector_ids([1, 0])],
        )
        .unwrap();
        let degeneracy =
            DegeneracyStructure::packed_column_major(2, [vec![2, 3], vec![3, 2]]).unwrap();
        let structure = BlockStructure::from_parts(sector, degeneracy).unwrap();

        assert_eq!(structure.rank(), 2);
        assert_eq!(
            structure.sector_structure().key(0).unwrap(),
            &BlockKey::sector_ids([0, 1])
        );
        assert_eq!(
            structure.sector_structure().key(1).unwrap(),
            &BlockKey::sector_ids([1, 0])
        );
        assert_eq!(
            structure.degeneracy_structure().block(0).unwrap().shape(),
            &[2, 3]
        );
        assert_eq!(
            structure.degeneracy_structure().block(1).unwrap().offset(),
            6
        );
        assert_eq!(structure.required_len().unwrap(), 12);
    }

    #[test]
    fn sector_structure_pairs_compact_keys_without_map_lookup() {
        let dst = SectorStructure::from_keys(
            2,
            [
                BlockKey::sector_ids([2]),
                BlockKey::sector_ids([0]),
                BlockKey::sector_ids([1]),
            ],
        )
        .unwrap();
        let src = SectorStructure::from_keys(
            2,
            [
                BlockKey::sector_ids([0]),
                BlockKey::sector_ids([1]),
                BlockKey::sector_ids([2]),
            ],
        )
        .unwrap();

        assert!(src.has_compact_lookup());
        assert_eq!(dst.find_index(&BlockKey::sector_ids([0])), Some(1));
        assert_eq!(src.find_index(&BlockKey::sector_ids([2])), Some(2));
        assert_eq!(dst.pair_indices_from(&src).unwrap(), vec![2, 0, 1]);
    }

    #[test]
    fn sector_structure_pairs_general_fusion_keys_by_sorted_merge() {
        let key_a = BlockKey::sectors([SectorId::new(0), SectorId::new(1)]);
        let key_b = BlockKey::sectors([SectorId::new(1), SectorId::new(0)]);
        let dst = SectorStructure::from_keys(2, [key_b.clone(), key_a.clone()]).unwrap();
        let src = SectorStructure::from_keys(2, [key_a.clone(), key_b.clone()]).unwrap();

        assert!(!src.has_compact_lookup());
        assert_eq!(dst.find_index(&key_a), Some(1));
        assert_eq!(src.find_index(&key_b), Some(1));
        assert_eq!(dst.pair_indices_from(&src).unwrap(), vec![1, 0]);
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

    #[test]
    fn vec_storage_allocates_similar_host_scratch() {
        let storage = vec![1.0_f64, 2.0];
        let scratch = storage.similar_filled(4, 0.5);

        assert_eq!(scratch, vec![0.5; 4]);
        assert_eq!(scratch.placement(), Placement::Host);
    }

    #[test]
    fn tensormap_allocates_similar_storage_from_backing_storage() {
        let space = TensorMapSpace::<1, 0>::from_dims([2], []).unwrap();
        let tensor = TensorMap::<f64, 1, 0>::from_vec(vec![1.0, 2.0], space).unwrap();
        let scratch = tensor.similar_storage_filled(3, 0.0);

        assert_eq!(scratch, vec![0.0; 3]);
        assert_eq!(scratch.placement(), tensor.placement());
    }

    #[test]
    fn split_fusion_tree_matches_tensorkit_front_tail_convention() {
        let rule = SU2FusionRule;
        let half = SU2Irrep::from_twice_spin(1).sector_id();
        let one = SU2Irrep::from_twice_spin(2).sector_id();
        let tree = FusionTreeKey::new(
            [half, half, one],
            Some(one),
            [false, false, true],
            [SectorId::new(0)],
            [SectorId::new(1), SectorId::new(1)],
        );

        let (front, tail) = split_fusion_tree(&rule, &tree, 2).unwrap();

        assert_eq!(front.uncoupled(), &[half, half]);
        assert_eq!(front.coupled(), Some(SectorId::new(0)));
        assert_eq!(front.is_dual(), &[false, false]);
        assert_eq!(front.innerlines(), &[]);
        assert_eq!(front.vertices(), &[SectorId::new(1)]);
        assert_eq!(tail.uncoupled(), &[SectorId::new(0), one]);
        assert_eq!(tail.coupled(), Some(one));
        assert_eq!(tail.is_dual(), &[false, true]);
        assert_eq!(tail.innerlines(), &[]);
        assert_eq!(tail.vertices(), &[SectorId::new(1)]);
    }

    #[test]
    fn rigid_symbols_separate_twist_from_frobenius_schur_phase() {
        let fermion = FermionParityFusionRule;
        let odd = SectorId::new(1);
        assert_eq!(fermion.dim_scalar(odd), 1.0);
        assert_eq!(fermion.twist_scalar(odd), -1.0);
        assert_eq!(fermion.frobenius_schur_phase_scalar(odd), 1.0);

        let su2 = SU2FusionRule;
        let half = SU2Irrep::from_twice_spin(1).sector_id();
        assert_eq!(su2.dim_scalar(half), 2.0);
        assert_eq!(su2.twist_scalar(half), 1.0);
        assert_eq!(su2.frobenius_schur_phase_scalar(half), -1.0);
    }
}
