/// Test-only synchronization for tenet-core's process-global intern tables
/// (block-structure content/arc tables, the hom-space intern table).
///
/// Why-not (the alternatives this replaces): a per-test-file `#[serial]`
/// dependency would serialize this whole (large) test module for the sake of
/// a handful of tests; making the tables test-scoped (thread-local, or wiped
/// between tests) would stop exercising the process-global design these
/// tables actually ship with, and would hide the exact "concurrent
/// reset/flood lands between two reads" bugs this suite exists to catch
/// (see tenet-tensors #169, #172 for the shape of the bug).
///
/// So: one process-wide `Mutex`, taken by every test that either mutates
/// shared intern-table state (`reset_core_intern_tables`, LRU-cap floods) or
/// asserts on it (`Arc::ptr_eq` of interned values, table lengths, content
/// ids). Poison-tolerant: a panicking test must not cascade spurious
/// failures onto every other test sharing the lock.
pub(crate) mod test_support {
    pub(crate) static CACHE_TEST_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
}

mod tests {
    use super::*;
    use smallvec::smallvec;
    use std::hash::{Hash, Hasher};

    #[test]
    fn sector_leg_treats_zero_degeneracy_as_an_absent_sector() {
        // What: explicit zero degeneracies and omitted sectors identify the same
        // mathematical leg, fusion-tree keys, and hash identity.
        let even = z2_even();
        let odd = z2_odd();
        let omitted = SectorLeg::new([(even, 2)], false);
        let explicit_zero = SectorLeg::new([(odd, 0), (even, 2)], false);

        assert_eq!(explicit_zero, omitted);
        assert_eq!(explicit_zero.degeneracy(odd), None);

        let hash = |leg: &SectorLeg| {
            let mut state = std::collections::hash_map::DefaultHasher::new();
            leg.hash(&mut state);
            state.finish()
        };
        assert_eq!(hash(&explicit_zero), hash(&omitted));

        let hom = |leg| {
            FusionTreeHomSpace::new(
                FusionProductSpace::new([leg]),
                FusionProductSpace::new([SectorLeg::new([(even, 2)], false)]),
            )
        };
        assert_eq!(
            hom(explicit_zero).fusion_tree_keys(&Z2FusionRule),
            hom(omitted).fusion_tree_keys(&Z2FusionRule)
        );
    }

    #[test]
    fn sector_leg_try_new_reports_every_duplicate_sector_declaration() {
        // What: duplicate sector declarations are construction errors
        // independent of degeneracy and input order.
        let sector = z2_even();
        for pairs in [
            [(sector, 2), (sector, 2)],
            [(sector, 2), (sector, 3)],
            [(sector, 0), (sector, 2)],
            [(sector, 2), (sector, 0)],
            [(sector, 0), (sector, 0)],
        ] {
            assert_eq!(
                SectorLeg::try_new(pairs, false),
                Err(SectorLegConstructionError::DuplicateSector { sector })
            );
        }
    }

    #[test]
    #[should_panic(expected = "appears multiple times")]
    fn sector_leg_new_preserves_the_infallible_panic_boundary() {
        // What: the compatibility constructor still rejects duplicate positive
        // sectors instead of silently selecting one declaration.
        let sector = z2_even();
        let _ = SectorLeg::new([(sector, 2), (sector, 2)], false);
    }

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
            packed_fixture_structure(
                4,
                hom.fusion_tree_keys(&rule).iter().cloned().zip(shapes(&hom)),
            )
                .unwrap();
        let packed_space =
            FusionTensorMapSpace::<2, 2>::new_unbound(dense(), hom.clone(), packed_structure)
                .unwrap();
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

    #[test]
    fn coupled_sector_dimensions_cover_empty_and_multiplicity_free_products() {
        // What: product-space dimensions use the tensor unit for rank zero,
        // annihilate on an empty leg, and reproduce U1/SU2 fusion dimensions.
        let empty = FusionProductSpace::new(std::iter::empty::<SectorLeg>());
        assert_eq!(
            empty
                .coupled_sector_block_dimensions(&U1FusionRule)
                .unwrap(),
            BTreeMap::from([(u1(0), 1)])
        );

        let empty_leg = FusionProductSpace::new([SectorLeg::new(
            std::iter::empty::<(SectorId, usize)>(),
            false,
        )]);
        assert!(empty_leg
            .coupled_sector_block_dimensions(&U1FusionRule)
            .unwrap()
            .is_empty());

        let u1_leg = SectorLeg::new([(u1(0), 1), (u1(1), 1)], false);
        let u1_product = FusionProductSpace::new([u1_leg.clone(), u1_leg]);
        assert_eq!(
            u1_product
                .coupled_sector_block_dimensions(&U1FusionRule)
                .unwrap(),
            BTreeMap::from([(u1(0), 1), (u1(1), 2), (u1(2), 1)])
        );

        let half = SectorLeg::new([(su2(1), 1)], false);
        let su2_product = FusionProductSpace::new([half.clone(), half]);
        assert_eq!(
            su2_product
                .coupled_sector_block_dimensions(&SU2FusionRule)
                .unwrap(),
            BTreeMap::from([(su2(0), 1), (su2(2), 1)])
        );
    }

    #[test]
    fn coupled_sector_dimensions_keep_outward_labels_for_dual_legs() {
        // What: the pivotal dual flag does not dualize an already-outward U1
        // sector label a second time.
        let product =
            FusionProductSpace::new([SectorLeg::new([(u1(-3), 2)], true)]);
        assert_eq!(
            product
                .coupled_sector_block_dimensions(&U1FusionRule)
                .unwrap(),
            BTreeMap::from([(u1(-3), 2)])
        );
    }

    #[derive(Clone, Copy, Debug)]
    struct IsomorphismMultiplicityRule;

    impl FusionRule for IsomorphismMultiplicityRule {
        fn rule_identity(&self) -> RuleIdentity {
            RuleIdentity::of_type::<Self>()
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

        fn fusion_channels(&self, left: SectorId, right: SectorId) -> SectorVec {
            match (left.id(), right.id()) {
                (0, sector) | (sector, 0) => smallvec![SectorId::new(sector)],
                (1, 1) => smallvec![SectorId::new(2)],
                _ => SectorVec::new(),
            }
        }

        fn nsymbol(&self, left: SectorId, right: SectorId, coupled: SectorId) -> usize {
            if (left.id(), right.id(), coupled.id()) == (1, 1, 2) {
                2
            } else {
                usize::from(self.fusion_channels(left, right).contains(&coupled))
            }
        }
    }

    #[test]
    fn coupled_sector_dimensions_include_outer_multiplicity_and_check_overflow() {
        // What: a generic fusion channel contributes N(a,b,c), and dimension
        // arithmetic reports overflow rather than wrapping.
        let sector = SectorId::new(1);
        let product = FusionProductSpace::new([
            SectorLeg::new([(sector, 3)], false),
            SectorLeg::new([(sector, 5)], false),
        ]);
        assert_eq!(
            product
                .coupled_sector_block_dimensions(&IsomorphismMultiplicityRule)
                .unwrap(),
            BTreeMap::from([(SectorId::new(2), 30)])
        );

        let overflowing = FusionProductSpace::new([
            SectorLeg::new([(sector, usize::MAX)], false),
            SectorLeg::new([(sector, 1)], false),
        ]);
        assert_eq!(
            overflowing.coupled_sector_block_dimensions(&IsomorphismMultiplicityRule),
            Err(CoreError::ElementCountOverflow)
        );
    }

    #[derive(Clone, Copy, Debug)]
    struct IncompleteDimensionFoldRule;

    impl FusionRule for IncompleteDimensionFoldRule {
        fn rule_identity(&self) -> RuleIdentity {
            RuleIdentity::of_type::<Self>()
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

        fn fusion_channels(&self, _left: SectorId, _right: SectorId) -> SectorVec {
            SectorVec::new()
        }

        fn coupled_sector_fold(&self, _effective: &[SectorId]) -> CoupledSectorFold {
            CoupledSectorFold {
                tainted: vec![SectorId::new(1)],
                ..CoupledSectorFold::default()
            }
        }
    }

    #[test]
    fn coupled_sector_dimensions_reject_incomplete_bounded_folds() {
        // What: an incomplete bounded fusion result is a typed error, never a
        // silently truncated sector-dimension map.
        let product =
            FusionProductSpace::new([SectorLeg::new([(SectorId::new(1), 1)], false)]);
        assert!(matches!(
            product.coupled_sector_block_dimensions(&IncompleteDimensionFoldRule),
            Err(CoreError::FusionOutsideTable { .. })
        ));
    }

    #[derive(Clone, Copy, Debug)]
    struct BranchingMultiplicityFreeRule;

    impl FusionRule for BranchingMultiplicityFreeRule {
        fn rule_identity(&self) -> RuleIdentity { RuleIdentity::of_type::<Self>() }
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
        fn rule_identity(&self) -> RuleIdentity { RuleIdentity::of_type::<Self>() }
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
        fn rule_identity(&self) -> RuleIdentity { RuleIdentity::of_type::<Self>() }
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
        fn rule_identity(&self) -> RuleIdentity { RuleIdentity::of_type::<Self>() }
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
        fn rule_identity(&self) -> RuleIdentity { RuleIdentity::of_type::<Self>() }
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

    impl MultiplicityFreeRigidSymbols for PlanarZ2Rule {
        fn dim_scalar(&self, _sector: SectorId) -> Self::Scalar {
            1.0
        }

        fn inv_dim_scalar(&self, _sector: SectorId) -> Self::Scalar {
            1.0
        }

        fn sqrt_dim_scalar(&self, _sector: SectorId) -> Self::Scalar {
            1.0
        }

        fn inv_sqrt_dim_scalar(&self, _sector: SectorId) -> Self::Scalar {
            1.0
        }

        fn twist_scalar(&self, _sector: SectorId) -> Self::Scalar {
            1.0
        }

        fn frobenius_schur_phase_scalar(&self, _sector: SectorId) -> Self::Scalar {
            1.0
        }
    }

    #[derive(Clone, Copy, Debug)]
    struct IdentitySymbolPanicRule;

    impl FusionRule for IdentitySymbolPanicRule {
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
            SectorId::new(0)
        }

        fn fusion_channels(&self, left: SectorId, right: SectorId) -> SectorVec {
            smallvec![SectorId::new(left.id() ^ right.id())]
        }
    }

    impl MultiplicityFreeFusionRule for IdentitySymbolPanicRule {}

    impl MultiplicityFreeFusionSymbols for IdentitySymbolPanicRule {
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
            panic!("identity braid evaluated an F symbol")
        }

        fn r_symbol_scalar(
            &self,
            _left: SectorId,
            _right: SectorId,
            _coupled: SectorId,
        ) -> Self::Scalar {
            panic!("identity braid evaluated an R symbol")
        }
    }

    impl MultiplicityFreePivotalSymbols for IdentitySymbolPanicRule {
        fn bendright_scalar(
            &self,
            _left_coupled: SectorId,
            _bent_sector: SectorId,
            _coupled: SectorId,
            _bent_leg_is_dual: bool,
        ) -> Self::Scalar {
            panic!("identity braid evaluated a bend symbol")
        }

        fn foldright_scalar(
            &self,
            _source: &FusionTreePairKey,
            _destination: &FusionTreePairKey,
        ) -> Self::Scalar {
            panic!("identity braid evaluated a fold symbol")
        }
    }

    impl MultiplicityFreeRigidSymbols for IdentitySymbolPanicRule {
        fn dim_scalar(&self, _sector: SectorId) -> Self::Scalar {
            1.0
        }

        fn inv_dim_scalar(&self, _sector: SectorId) -> Self::Scalar {
            1.0
        }

        fn sqrt_dim_scalar(&self, _sector: SectorId) -> Self::Scalar {
            1.0
        }

        fn inv_sqrt_dim_scalar(&self, _sector: SectorId) -> Self::Scalar {
            1.0
        }

        fn twist_scalar(&self, _sector: SectorId) -> Self::Scalar {
            1.0
        }

        fn frobenius_schur_phase_scalar(&self, _sector: SectorId) -> Self::Scalar {
            1.0
        }
    }

    #[derive(Clone, Copy, Debug)]
    struct MisreportedGenericMultiplicityFreeRule;

    impl FusionRule for MisreportedGenericMultiplicityFreeRule {
        fn rule_identity(&self) -> RuleIdentity {
            RuleIdentity::of_type::<Self>()
        }

        fn fusion_style(&self) -> FusionStyleKind {
            FusionStyleKind::Generic
        }

        fn braiding_style(&self) -> BraidingStyleKind {
            BraidingStyleKind::Bosonic
        }

        fn vacuum(&self) -> SectorId {
            IdentitySymbolPanicRule.vacuum()
        }

        fn fusion_channels(&self, left: SectorId, right: SectorId) -> SectorVec {
            IdentitySymbolPanicRule.fusion_channels(left, right)
        }
    }

    impl MultiplicityFreeFusionRule for MisreportedGenericMultiplicityFreeRule {}

    impl MultiplicityFreeFusionSymbols for MisreportedGenericMultiplicityFreeRule {
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
            panic!("misreported Generic provider evaluated an F symbol")
        }

        fn r_symbol_scalar(
            &self,
            _left: SectorId,
            _right: SectorId,
            _coupled: SectorId,
        ) -> Self::Scalar {
            panic!("misreported Generic provider evaluated an R symbol")
        }
    }

    impl MultiplicityFreePivotalSymbols for MisreportedGenericMultiplicityFreeRule {
        fn bendright_scalar(
            &self,
            _left_coupled: SectorId,
            _bent_sector: SectorId,
            _coupled: SectorId,
            _bent_leg_is_dual: bool,
        ) -> Self::Scalar {
            panic!("misreported Generic provider evaluated a bend symbol")
        }

        fn foldright_scalar(
            &self,
            _source: &FusionTreePairKey,
            _destination: &FusionTreePairKey,
        ) -> Self::Scalar {
            panic!("misreported Generic provider evaluated a fold symbol")
        }
    }

    impl MultiplicityFreeRigidSymbols for MisreportedGenericMultiplicityFreeRule {
        fn dim_scalar(&self, _sector: SectorId) -> Self::Scalar {
            1.0
        }

        fn inv_dim_scalar(&self, _sector: SectorId) -> Self::Scalar {
            1.0
        }

        fn sqrt_dim_scalar(&self, _sector: SectorId) -> Self::Scalar {
            1.0
        }

        fn inv_sqrt_dim_scalar(&self, _sector: SectorId) -> Self::Scalar {
            1.0
        }

        fn twist_scalar(&self, _sector: SectorId) -> Self::Scalar {
            1.0
        }

        fn frobenius_schur_phase_scalar(&self, _sector: SectorId) -> Self::Scalar {
            1.0
        }
    }

    #[derive(Debug, Default)]
    struct SplitOnlyCountingRule {
        n_calls: std::sync::atomic::AtomicUsize,
        f_calls: std::sync::atomic::AtomicUsize,
        r_calls: std::sync::atomic::AtomicUsize,
        bend_calls: std::sync::atomic::AtomicUsize,
    }

    impl FusionRule for SplitOnlyCountingRule {
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
            SectorId::new(0)
        }

        fn fusion_channels(&self, left: SectorId, right: SectorId) -> SectorVec {
            smallvec![SectorId::new(left.id() ^ right.id())]
        }

        fn nsymbol(&self, left: SectorId, right: SectorId, coupled: SectorId) -> usize {
            self.n_calls
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            usize::from(left.id() ^ right.id() == coupled.id())
        }
    }

    impl MultiplicityFreeFusionRule for SplitOnlyCountingRule {}

    impl MultiplicityFreeFusionSymbols for SplitOnlyCountingRule {
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
            self.f_calls
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            1.0
        }

        fn r_symbol_scalar(
            &self,
            _left: SectorId,
            _right: SectorId,
            _coupled: SectorId,
        ) -> Self::Scalar {
            self.r_calls
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            1.0
        }
    }

    impl MultiplicityFreeRigidSymbols for SplitOnlyCountingRule {
        fn dim_scalar(&self, _sector: SectorId) -> Self::Scalar {
            1.0
        }

        fn inv_dim_scalar(&self, _sector: SectorId) -> Self::Scalar {
            1.0
        }

        fn sqrt_dim_scalar(&self, _sector: SectorId) -> Self::Scalar {
            1.0
        }

        fn inv_sqrt_dim_scalar(&self, _sector: SectorId) -> Self::Scalar {
            1.0
        }

        fn twist_scalar(&self, _sector: SectorId) -> Self::Scalar {
            1.0
        }

        fn frobenius_schur_phase_scalar(&self, _sector: SectorId) -> Self::Scalar {
            1.0
        }
    }

    impl MultiplicityFreePivotalSymbols for SplitOnlyCountingRule {
        fn bendright_scalar(
            &self,
            _left_coupled: SectorId,
            _bent_sector: SectorId,
            _coupled: SectorId,
            _bent_leg_is_dual: bool,
        ) -> Self::Scalar {
            self.bend_calls
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            1.0
        }

        fn foldright_scalar(
            &self,
            _source: &FusionTreePairKey,
            _destination: &FusionTreePairKey,
        ) -> Self::Scalar {
            1.0
        }
    }

    fn legacy_split_only_tree_pair_route<R>(
        rule: &R,
        source: &FusionTreePairKey,
        target_codomain_rank: usize,
    ) -> Result<Vec<(FusionTreePairKey, R::Scalar)>, CoreError>
    where
        R: MultiplicityFreeRigidSymbols,
        R::Scalar: Clone + Add<Output = R::Scalar> + Mul<Output = R::Scalar>,
    {
        // What: freeze the pre-shortcut composition as an independent oracle:
        // repartition to all-codomain, apply the identity tree braid, then
        // repartition back to the requested split.
        let total_rank =
            source.codomain_tree().uncoupled().len() + source.domain_tree().uncoupled().len();
        let identity = (0..total_rank).collect::<Vec<_>>();
        let levels = identity.clone();
        let all_codomain =
            multiplicity_free_repartition_tree_pair(rule, source, total_rank)?;
        let braided = compose_tree_pair_terms(rule, all_codomain, |rule, key| {
            multiplicity_free_braid_tree(
                rule,
                key.codomain_tree(),
                &identity,
                &levels,
            )
            .map(|terms| {
                terms
                    .into_iter()
                    .map(|(tree, coefficient)| {
                        (
                            FusionTreePairKey::pair(tree, key.domain_tree().clone()),
                            coefficient,
                        )
                    })
                    .collect::<Vec<_>>()
            })
        })?;
        multiplicity_free_repartition_terms(rule, braided, target_codomain_rank)
    }

    #[derive(Clone, Copy, Debug)]
    struct AsymmetricAnyonicRule;

    impl FusionRule for AsymmetricAnyonicRule {
        fn rule_identity(&self) -> RuleIdentity { RuleIdentity::of_type::<Self>() }
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
            _source: &FusionTreePairKey,
            _destination: &FusionTreePairKey,
        ) -> Self::Scalar {
            1.0
        }
    }

    impl MultiplicityFreeRigidSymbols for AsymmetricAnyonicRule {
        fn dim_scalar(&self, _sector: SectorId) -> Self::Scalar {
            1.0
        }

        fn inv_dim_scalar(&self, _sector: SectorId) -> Self::Scalar {
            1.0
        }

        fn sqrt_dim_scalar(&self, _sector: SectorId) -> Self::Scalar {
            1.0
        }

        fn inv_sqrt_dim_scalar(&self, _sector: SectorId) -> Self::Scalar {
            1.0
        }

        fn twist_scalar(&self, _sector: SectorId) -> Self::Scalar {
            1.0
        }

        fn frobenius_schur_phase_scalar(&self, _sector: SectorId) -> Self::Scalar {
            1.0
        }
    }

    #[derive(Clone, Copy, Debug)]
    struct UncertifiedCustomSymbolsRule;

    impl FusionRule for UncertifiedCustomSymbolsRule {
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
            SectorId::new(0)
        }

        fn fusion_channels(&self, left: SectorId, right: SectorId) -> SectorVec {
            smallvec![SectorId::new(left.id() ^ right.id())]
        }
    }

    impl MultiplicityFreeFusionRule for UncertifiedCustomSymbolsRule {}

    impl MultiplicityFreeFusionSymbols for UncertifiedCustomSymbolsRule {
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
            2.0
        }

        fn r_symbol_scalar(
            &self,
            _left: SectorId,
            _right: SectorId,
            coupled: SectorId,
        ) -> Self::Scalar {
            if coupled.id() == 0 { 3.0 } else { 5.0 }
        }
    }

    #[derive(Clone, Copy, Debug)]
    struct ComplexAsymmetricUniqueRule;

    impl FusionRule for ComplexAsymmetricUniqueRule {
        fn rule_identity(&self) -> RuleIdentity {
            RuleIdentity::of_type::<Self>()
        }

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
            smallvec![SectorId::new((left.id() + right.id()) % 4)]
        }
    }

    impl MultiplicityFreeFusionRule for ComplexAsymmetricUniqueRule {}

    impl MultiplicityFreeFusionSymbols for ComplexAsymmetricUniqueRule {
        type Scalar = Complex64;

        fn scalar_one(&self) -> Self::Scalar {
            Complex64::new(1.0, 0.0)
        }

        fn scalar_conj(&self, value: Self::Scalar) -> Self::Scalar {
            value.conj()
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
            Complex64::new(1.0, 0.0)
        }

        fn r_symbol_scalar(
            &self,
            left: SectorId,
            right: SectorId,
            _coupled: SectorId,
        ) -> Self::Scalar {
            let angle = match (left.id(), right.id()) {
                (1, 2) => std::f64::consts::FRAC_PI_3,
                (2, 1) => std::f64::consts::FRAC_PI_6,
                _ => 0.0,
            };
            Complex64::from_polar(1.0, angle)
        }
    }

    fn fusion_tree_pair_order(keys: &[FusionTreePairKey]) -> Vec<(Vec<usize>, Vec<usize>, usize)> {
        keys.iter()
            .map(|key| {
                (
                    sector_ids(key.codomain_uncoupled()),
                    sector_ids(key.domain_uncoupled()),
                    key.coupled().id(),
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
        let tree = FusionTreeKey::try_from_sector_ids([0, 1], 1, [false, true], [], [1]).unwrap();

        let (braided, coefficient) = unique_artin_braid_first(&PlanarZ2Rule, &tree).unwrap();

        assert_eq!(coefficient, 1.0);
        assert_eq!(braided.uncoupled(), &[SectorId::new(1), SectorId::new(0)]);
        assert_eq!(braided.is_dual(), &[true, false]);
        assert_eq!(braided.coupled(), SectorId::new(1));
        assert!(braided.innerlines().is_empty());
        assert_eq!(braided.vertices(), &[MultiplicityIndex::ONE]);
    }

    #[test]
    fn unique_artin_braid_first_rejects_nonunit_crossing_without_braiding() {
        let tree = FusionTreeKey::try_from_sector_ids([1, 1], 0, [false, false], [], [1]).unwrap();

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
        let tree = FusionTreeKey::try_from_sector_ids([1, 1], 0, [false, true], [], [1]).unwrap();

        let (braided, coefficient) =
            unique_artin_braid_first(&FermionParityFusionRule, &tree).unwrap();

        assert_eq!(coefficient, -1.0);
        assert_eq!(braided.uncoupled(), &[SectorId::new(1), SectorId::new(1)]);
        assert_eq!(braided.is_dual(), &[true, false]);
        assert_eq!(braided.coupled(), SectorId::new(0));
        assert!(braided.innerlines().is_empty());
        assert_eq!(braided.vertices(), &[MultiplicityIndex::ONE]);
    }

    #[test]
    fn unique_artin_braid_first_uses_first_innerline_for_rank_three() {
        let tree =
            FusionTreeKey::try_from_sector_ids([1, 1, 1], 1, [false, false, false], [0], [1, 1]).unwrap();

        let (braided, coefficient) =
            unique_artin_braid_first(&FermionParityFusionRule, &tree).unwrap();

        assert_eq!(coefficient, -1.0);
        assert_eq!(
            braided.uncoupled(),
            &[SectorId::new(1), SectorId::new(1), SectorId::new(1)]
        );
        assert_eq!(braided.innerlines(), &[SectorId::new(0)]);
        assert_eq!(braided.vertices(), &[MultiplicityIndex::ONE, MultiplicityIndex::ONE]);
    }

    #[test]
    fn unique_artin_braid_at_updates_innerline_for_later_unit_crossing() {
        let tree =
            FusionTreeKey::try_from_sector_ids([1, 0, 1], 0, [false, false, true], [1], [1, 1]).unwrap();

        let (braided, coefficient) = unique_artin_braid_at(&PlanarZ2Rule, &tree, 1).unwrap();

        assert_eq!(coefficient, 1.0);
        assert_eq!(
            braided.uncoupled(),
            &[SectorId::new(1), SectorId::new(1), SectorId::new(0)]
        );
        assert_eq!(braided.is_dual(), &[false, true, false]);
        assert_eq!(braided.innerlines(), &[SectorId::new(0)]);
        assert_eq!(braided.vertices(), &[MultiplicityIndex::ONE, MultiplicityIndex::ONE]);
    }

    #[test]
    fn unique_artin_braid_at_uses_f_and_r_symbols_for_later_crossing() {
        let tree =
            FusionTreeKey::try_from_sector_ids([1, 1, 1], 1, [false, true, false], [0], [1, 1]).unwrap();

        let (braided, coefficient) =
            unique_artin_braid_at(&FermionParityFusionRule, &tree, 1).unwrap();

        assert_eq!(coefficient, -1.0);
        assert_eq!(
            braided.uncoupled(),
            &[SectorId::new(1), SectorId::new(1), SectorId::new(1)]
        );
        assert_eq!(braided.is_dual(), &[false, false, true]);
        assert_eq!(braided.innerlines(), &[SectorId::new(0)]);
        assert_eq!(braided.vertices(), &[MultiplicityIndex::ONE, MultiplicityIndex::ONE]);
    }

    #[test]
    fn unique_artin_braid_at_rejects_out_of_range_index() {
        let tree = FusionTreeKey::try_from_sector_ids([1, 1], 0, [false, false], [], [1]).unwrap();

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
            FusionTreeKey::try_from_sector_ids([1, 1, 1], 1, [false, false, false], [0], [1, 1]).unwrap();

        let (braided, coefficient) =
            unique_braid_tree(&FermionParityFusionRule, &tree, &[2, 0, 1], &[0, 1, 2]).unwrap();

        assert_eq!(coefficient, 1.0);
        assert_eq!(
            braided.uncoupled(),
            &[SectorId::new(1), SectorId::new(1), SectorId::new(1)]
        );
        assert_eq!(braided.is_dual(), &[false, false, false]);
        assert_eq!(braided.coupled(), SectorId::new(1));
        assert_eq!(braided.innerlines(), &[SectorId::new(0)]);
        assert_eq!(braided.vertices(), &[MultiplicityIndex::ONE, MultiplicityIndex::ONE]);
    }

    #[test]
    fn unique_braid_tree_uses_inverse_artin_branch_from_levels() {
        let tree = FusionTreeKey::try_from_sector_ids([1, 2], 3, [false, false], [], [1]).unwrap();

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
        assert_eq!(braided_forward.coupled(), SectorId::new(3));
    }

    #[test]
    fn unique_braid_tree_reflected_levels_select_inverse_artin_branch() {
        let tree = FusionTreeKey::try_from_sector_ids([1, 2], 3, [false, false], [], [1]).unwrap();
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
    fn symmetric_unique_direct_braid_matches_artin_replay_exactly() {
        // What: every rank-4 fZ2 permutation has the exact tree and sign of
        // TensorKit's adjacent-Artin semantics, including multiplication order.
        let tree = FusionTreeKey::try_from_sector_ids(
            [1, 1, 0, 1], 1,
            [false, true, false, true],
            [0, 0],
            [1, 1, 1],
        ).unwrap();
        let levels = [0, 1, 2, 3];
        let mut permutation = [0usize, 1, 2, 3];
        loop {
            let direct =
                unique_braid_tree(&FermionParityFusionRule, &tree, &permutation, &levels)
                    .unwrap();

            let mut replay_tree = tree.clone();
            let mut replay_coefficient = 1.0;
            let mut replay_levels = levels;
            for swap in permutation_to_adjacent_swaps(&permutation, 4).unwrap() {
                let inverse = replay_levels[swap] > replay_levels[swap + 1];
                let (next, coefficient) = unique_artin_braid_at_with_inverse(
                    &FermionParityFusionRule,
                    &replay_tree,
                    swap,
                    inverse,
                )
                .unwrap();
                replay_tree = next;
                replay_coefficient *= coefficient;
                replay_levels.swap(swap, swap + 1);
            }
            assert_eq!(direct, (replay_tree, replay_coefficient));

            let Some(pivot) =
                (0..permutation.len() - 1).rfind(|&index| permutation[index] < permutation[index + 1])
            else {
                break;
            };
            let successor = (pivot + 1..permutation.len())
                .rfind(|&index| permutation[index] > permutation[pivot])
                .unwrap();
            permutation.swap(pivot, successor);
            permutation[pivot + 1..].reverse();
        }
    }

    #[test]
    fn rule_aware_constructor_rejects_inadmissible_unique_innerline() {
        // What: a rule-aware constructor rejects a stored vertex that is
        // outside the Unique rule's fusion graph.
        assert_eq!(
            FusionTreeKey::try_new_for_rule(
            &FermionParityFusionRule,
            [SectorId::new(1), SectorId::new(1), SectorId::new(1)], SectorId::new(1),
            [false, false, false],
            [SectorId::new(1)],
            [MultiplicityIndex::ONE, MultiplicityIndex::ONE],
            )
            .unwrap_err(),
            CoreError::MalformedFusionTree {
                message: "fusion tree contains an inadmissible fusion vertex",
            }
        );
    }

    #[test]
    fn rule_aware_tree_validation_covers_rank_shape_style_and_vertices() {
        // What: categorical interpretation requires the canonical empty-tree
        // vacuum and rejects every representable malformed tree field.
        let empty_vacuum = FusionTreeKey::new([], z2_even(), [], [], []);
        empty_vacuum.validate_for_rule(&Z2FusionRule).unwrap();
        assert_eq!(
            FusionTreeKey::new([], z2_odd(), [], [], [])
                .validate_for_rule(&Z2FusionRule)
                .unwrap_err(),
            CoreError::MalformedFusionTree {
                message: "rank-0 fusion tree coupled sector must equal the vacuum",
            }
        );

        let rank_one =
            FusionTreeKey::new([z2_odd()], z2_odd(), [true], [], []);
        rank_one.validate_for_rule(&Z2FusionRule).unwrap();
        assert_eq!(
            FusionTreeKey::new([z2_odd()], z2_even(), [true], [], [])
                .validate_for_rule(&Z2FusionRule)
                .unwrap_err(),
            CoreError::MalformedFusionTree {
                message: "rank-1 fusion tree coupled sector must equal its uncoupled sector",
            }
        );

        let bad_shapes = [
            (
                FusionTreeKey::new([z2_odd(), z2_odd()], z2_even(), [false], [], [MultiplicityIndex::ONE]),
                "fusion tree sectors and duality flags must have matching length",
            ),
            (
                FusionTreeKey::new(
                    [z2_odd(), z2_odd(), z2_odd()], z2_odd(),
                    [false; 3],
                    [],
                    [MultiplicityIndex::ONE; 2],
                ),
                "fusion tree has an invalid number of innerlines",
            ),
            (
                FusionTreeKey::new(
                    [z2_odd(), z2_odd()], z2_even(),
                    [false; 2],
                    [],
                    [],
                ),
                "fusion tree has an invalid number of vertices",
            ),
        ];
        for (tree, message) in bad_shapes {
            assert_eq!(
                tree.validate_for_rule(&Z2FusionRule).unwrap_err(),
                CoreError::MalformedFusionTree { message }
            );
        }

        for tree in [
            FusionTreeKey::new(
                [z2_odd(); 4], z2_even(),
                [false; 4],
                [z2_even(), z2_even()],
                [MultiplicityIndex::ONE; 3],
            ),
            FusionTreeKey::new(
                [z2_odd(); 4], z2_odd(),
                [false; 4],
                [z2_even(), z2_odd()],
                [MultiplicityIndex::ONE; 3],
            ),
        ] {
            assert_eq!(
                tree.validate_for_rule(&Z2FusionRule).unwrap_err(),
                CoreError::MalformedFusionTree {
                    message: "fusion tree contains an inadmissible fusion vertex",
                }
            );
        }
    }

    #[test]
    fn rule_aware_constructor_enforces_one_based_vertex_bounds() {
        // What: vertex labels are checked against the actual N-symbol for
        // both multiplicity-free and Generic rules.
        assert_eq!(
            FusionTreeKey::try_new_for_rule(
                &Z2FusionRule,
                [z2_odd(), z2_odd()], z2_even(),
                [false; 2],
                [],
                [MultiplicityIndex::new(2).expect("test multiplicity label is one-based")],
            )
            .unwrap_err(),
            CoreError::MalformedFusionTree {
                message: "fusion tree vertex label exceeds its fusion multiplicity",
            }
        );
        assert_eq!(
            MultiplicityIndex::try_from(0).unwrap_err(),
            CoreError::InvalidMultiplicityIndex { value: 0 }
        );
        assert_eq!(
            FusionTreeKey::try_new_for_rule(
                &ToyOmRule,
                [SectorId::new(ToyOmRule::A), SectorId::new(ToyOmRule::A)],
                SectorId::new(ToyOmRule::C),
                [false; 2],
                [],
                [MultiplicityIndex::new(3).unwrap()],
            )
            .unwrap_err(),
            CoreError::MalformedFusionTree {
                message: "fusion tree vertex label exceeds its fusion multiplicity",
            }
        );
    }

    #[test]
    fn checked_tree_constructors_report_builtin_closure_without_unwinding() {
        // What: raw checked construction preserves exact finite-algebra
        // failures instead of entering the infallible provider path.
        assert_eq!(
            FusionTreeKey::try_new_for_rule_checked(
                &U1FusionRule,
                [u1(i32::MAX), u1(1)],
                u1(0),
                [false; 2],
                [],
                [MultiplicityIndex::ONE],
            ),
            Err(CheckedFusionSpaceError::FusionAlgebra(Box::new(
                FusionAlgebraError::U1FusionOverflow {
                    left: i32::MAX,
                    right: 1,
                }
            )))
        );
        assert_eq!(
            FusionTreeKey::try_from_sector_ids_for_rule_checked(
                &SU2FusionRule,
                [128, 127],
                0,
                [false; 2],
                [],
                [1],
            ),
            Err(CheckedFusionSpaceError::FusionAlgebra(Box::new(
                FusionAlgebraError::FusionNotRepresentable {
                    left: su2(128),
                    right: su2(127),
                }
            )))
        );
    }

    fn assert_invalid_rank_one_checked<R>(
        rule: &R,
        invalid: SectorId,
        expected: FusionAlgebraError,
    ) where
        R: CheckedFusionAlgebra,
    {
        let tree = FusionTreeKey::new([invalid], invalid, [false], [], []);
        assert_eq!(
            tree.validate_for_rule_checked(rule),
            Err(CheckedFusionSpaceError::FusionAlgebra(Box::new(expected)))
        );
    }

    #[test]
    fn checked_rank_one_validates_ids_by_unit_fusion() {
        // What: rank-one raw imports reject every unrepresentable built-in ID,
        // while U1 MIN remains valid without computing its overflowing dual.
        let invalid = SectorId::new(2);
        assert_invalid_rank_one_checked(
            &Z2FusionRule,
            invalid,
            FusionAlgebraError::InvalidSector { sector: invalid },
        );
        assert_invalid_rank_one_checked(
            &FermionParityFusionRule,
            invalid,
            FusionAlgebraError::InvalidSector { sector: invalid },
        );
        assert_invalid_rank_one_checked(
            &FibonacciFusionRule,
            invalid,
            FusionAlgebraError::InvalidSector { sector: invalid },
        );
        let invalid_su2 = SectorId::new(255);
        assert_invalid_rank_one_checked(
            &SU2FusionRule,
            invalid_su2,
            FusionAlgebraError::InvalidSector {
                sector: invalid_su2,
            },
        );
        #[cfg(target_pointer_width = "64")]
        {
            let invalid_u1 = SectorId::new(u32::MAX as usize + 1);
            assert_invalid_rank_one_checked(
                &U1FusionRule,
                invalid_u1,
                FusionAlgebraError::InvalidSector { sector: invalid_u1 },
            );

            type Layout = ProductSectorLayout<Fz2SectorLayout, U1SectorLayout>;
            type Codec = PackedProductCodec<Fz2SectorLayout, U1SectorLayout>;
            type Rule = ProductFusionRule<FermionParityFusionRule, U1FusionRule, Codec>;
            let rule = Rule::new(FermionParityFusionRule, U1FusionRule);
            let invalid_product = SectorId::new(1usize << Layout::BITS);
            assert_invalid_rank_one_checked(
                &rule,
                invalid_product,
                FusionAlgebraError::ProductCodec(
                    ProductSectorCodecError::InvalidHighBits {
                        sector: invalid_product,
                        total_bits: Layout::BITS,
                    },
                ),
            );
        }

        FusionTreeKey::new([u1(i32::MIN)], u1(i32::MIN), [false], [], [])
            .validate_for_rule_checked(&U1FusionRule)
            .unwrap();
    }

    #[test]
    fn checked_tree_distinguishes_absent_channels_from_algebra_failure() {
        // What: a representable but mathematically absent stored channel is a
        // malformed tree, while invalid stored IDs and recursive product
        // closure keep their exact algebra causes.
        assert_eq!(
            FusionTreeKey::new(
                [u1(1), u1(2)],
                u1(4),
                [false; 2],
                [],
                [MultiplicityIndex::ONE],
            )
            .validate_for_rule_checked(&U1FusionRule),
            Err(CheckedFusionSpaceError::Core(Box::new(
                CoreError::MalformedFusionTree {
                    message: "fusion tree contains an inadmissible fusion vertex",
                }
            )))
        );
        let invalid = SectorId::new(2);
        assert_eq!(
            FusionTreeKey::new(
                [z2_odd(), z2_odd()],
                invalid,
                [false; 2],
                [],
                [MultiplicityIndex::ONE],
            )
            .validate_for_rule_checked(&Z2FusionRule),
            Err(CheckedFusionSpaceError::FusionAlgebra(Box::new(
                FusionAlgebraError::InvalidSector { sector: invalid }
            )))
        );

        #[cfg(target_pointer_width = "64")]
        {
            type Codec = PackedProductCodec<Fz2SectorLayout, U1SectorLayout>;
            type Rule = ProductFusionRule<FermionParityFusionRule, U1FusionRule, Codec>;
            let rule = Rule::new(FermionParityFusionRule, U1FusionRule);
            let left = Codec::encode(z2_even(), u1(i32::MAX));
            let right = Codec::encode(z2_odd(), u1(1));
            let coupled = Codec::encode(z2_odd(), u1(0));
            assert_eq!(
                FusionTreeKey::new(
                    [left, right],
                    coupled,
                    [false; 2],
                    [],
                    [MultiplicityIndex::ONE],
                )
                .validate_for_rule_checked(&rule),
                Err(CheckedFusionSpaceError::FusionAlgebra(Box::new(
                    FusionAlgebraError::U1FusionOverflow {
                        left: i32::MAX,
                        right: 1,
                    }
                )))
            );
        }
    }

    #[derive(Debug, Default)]
    struct CheckedTreeProbe {
        channel_calls: AtomicUsize,
        nsymbol_calls: AtomicUsize,
        legacy_nsymbol_calls: AtomicUsize,
    }

    impl FusionRule for CheckedTreeProbe {
        fn rule_identity(&self) -> RuleIdentity {
            RuleIdentity::of_type::<Self>()
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

        fn fusion_channels(&self, _left: SectorId, _right: SectorId) -> SectorVec {
            core::iter::once(SectorId::new(0)).collect()
        }

        fn nsymbol(&self, _left: SectorId, _right: SectorId, _coupled: SectorId) -> usize {
            self.legacy_nsymbol_calls.fetch_add(1, Ordering::Relaxed);
            1
        }
    }

    impl CheckedFusionAlgebra for CheckedTreeProbe {
        fn try_dual_sector(&self, sector: SectorId) -> Result<SectorId, FusionAlgebraError> {
            Ok(sector)
        }

        fn try_fusion_channels(
            &self,
            left: SectorId,
            right: SectorId,
        ) -> Result<SectorVec, FusionAlgebraError> {
            self.channel_calls.fetch_add(1, Ordering::Relaxed);
            if left == SectorId::new(9) {
                Err(FusionAlgebraError::FusionNotRepresentable { left, right })
            } else {
                Ok(self.fusion_channels(left, right))
            }
        }

        fn try_nsymbol(
            &self,
            _left: SectorId,
            _right: SectorId,
            _coupled: SectorId,
        ) -> Result<usize, FusionAlgebraError> {
            self.nsymbol_calls.fetch_add(1, Ordering::Relaxed);
            Ok(1)
        }
    }

    #[test]
    fn checked_tree_normalizes_structure_before_provider_calls() {
        // What: shape and rank errors touch no checked provider operation, and
        // generated-channel failure precedes stored multiplicity validation.
        let rule = CheckedTreeProbe::default();
        for tree in [
            FusionTreeKey::new(
                [SectorId::new(0); 2],
                SectorId::new(0),
                [false],
                [],
                [MultiplicityIndex::ONE],
            ),
            FusionTreeKey::new(
                [SectorId::new(0)],
                SectorId::new(1),
                [false],
                [],
                [],
            ),
        ] {
            assert!(matches!(
                tree.validate_for_rule_checked(&rule),
                Err(CheckedFusionSpaceError::Core(_))
            ));
        }
        assert_eq!(rule.channel_calls.load(Ordering::Relaxed), 0);
        assert_eq!(rule.nsymbol_calls.load(Ordering::Relaxed), 0);

        let failure = FusionTreeKey::new(
            [SectorId::new(9), SectorId::new(0)],
            SectorId::new(0),
            [false; 2],
            [],
            [MultiplicityIndex::new(2).unwrap()],
        )
        .validate_for_rule_checked(&rule);
        assert_eq!(
            failure,
            Err(CheckedFusionSpaceError::FusionAlgebra(Box::new(
                FusionAlgebraError::FusionNotRepresentable {
                    left: SectorId::new(9),
                    right: SectorId::new(0),
                }
            )))
        );
        assert_eq!(rule.channel_calls.load(Ordering::Relaxed), 1);
        assert_eq!(rule.nsymbol_calls.load(Ordering::Relaxed), 0);

        rule.channel_calls.store(0, Ordering::Relaxed);
        rule.nsymbol_calls.store(0, Ordering::Relaxed);
        assert_eq!(
            FusionTreeKey::new(
                [SectorId::new(0); 2],
                SectorId::new(0),
                [false; 2],
                [],
                [MultiplicityIndex::new(2).unwrap()],
            )
            .validate_for_rule_checked(&rule),
            Err(CheckedFusionSpaceError::Core(Box::new(
                CoreError::MalformedFusionTree {
                    message: "fusion tree vertex label exceeds its fusion multiplicity",
                }
            )))
        );
        assert_eq!(rule.channel_calls.load(Ordering::Relaxed), 1);
        assert_eq!(rule.nsymbol_calls.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn checked_tree_pair_preserves_validation_precedence() {
        // What: pair validation reports codomain, then domain, then coupled
        // mismatch without reordering finite-algebra checks.
        let bad_shape = FusionTreeKey::new(
            [u1(0); 2],
            u1(0),
            [false],
            [],
            [MultiplicityIndex::ONE],
        );
        let overflow = FusionTreeKey::new(
            [u1(i32::MAX), u1(1)],
            u1(0),
            [false; 2],
            [],
            [MultiplicityIndex::ONE],
        );
        assert!(matches!(
            FusionTreePairKey::pair(bad_shape, overflow.clone())
                .validate_for_rule_checked(&U1FusionRule),
            Err(CheckedFusionSpaceError::Core(_))
        ));
        assert_eq!(
            FusionTreePairKey::pair(
                FusionTreeKey::new([], u1(0), [], [], []),
                overflow,
            )
            .validate_for_rule_checked(&U1FusionRule),
            Err(CheckedFusionSpaceError::FusionAlgebra(Box::new(
                FusionAlgebraError::U1FusionOverflow {
                    left: i32::MAX,
                    right: 1,
                }
            )))
        );
        assert_eq!(
            FusionTreePairKey::pair(
                FusionTreeKey::new([u1(1)], u1(1), [false], [], []),
                FusionTreeKey::new([u1(2)], u1(2), [false], [], []),
            )
            .validate_for_rule_checked(&U1FusionRule),
            Err(CheckedFusionSpaceError::Core(Box::new(
                CoreError::MalformedFusionTree {
                    message: "fusion tree pair requires matching coupled sectors",
                }
            )))
        );
    }

    #[test]
    fn fusion_subset_structural_proof_precedes_algebra_and_preserves_legacy_work() {
        // What: HomSpace metadata is proved without provider calls, while the
        // legacy categorical phase retains the former local-validator work.
        let scalar = FusionTreeKey::new([], SectorId::new(0), [], [], []);
        let malformed = FusionTreeKey::new(
            [SectorId::new(0); 2],
            SectorId::new(0),
            [false],
            [],
            [MultiplicityIndex::ONE],
        );
        let malformed_structure = packed_fixture_structure(
            2,
            [(
                FusionTreePairKey::pair(malformed, scalar.clone()),
                vec![1, 1],
            )],
        )
        .unwrap();
        let homspace = FusionTreeHomSpace::from_sector_ids([(0, 1), (0, 1)], []);
        let checked = CheckedTreeProbe::default();
        assert!(matches!(
            homspace.validate_subblock_structure_subset_checked(
                &checked,
                &malformed_structure,
            ),
            Err(CheckedFusionSpaceError::Core(_))
        ));
        assert_eq!(checked.channel_calls.load(Ordering::Relaxed), 0);
        assert_eq!(checked.nsymbol_calls.load(Ordering::Relaxed), 0);

        let valid = FusionTreeKey::new(
            [SectorId::new(0); 2],
            SectorId::new(0),
            [false; 2],
            [],
            [MultiplicityIndex::ONE],
        );
        let coupling_probe = CheckedTreeProbe::default();
        let mismatched_structure = packed_fixture_structure(
            2,
            [
                (
                    FusionTreePairKey::pair(valid.clone(), scalar.clone()),
                    vec![1, 1],
                ),
                (
                    FusionTreePairKey::pair(
                        valid.clone(),
                        FusionTreeKey::new([], SectorId::new(1), [], [], []),
                    ),
                    vec![1, 1],
                ),
            ],
        )
        .unwrap();
        assert!(matches!(
            homspace.validate_subblock_structure_subset_checked(
                &coupling_probe,
                &mismatched_structure,
            ),
            Err(CheckedFusionSpaceError::Core(_))
        ));
        assert_eq!(coupling_probe.channel_calls.load(Ordering::Relaxed), 0);
        assert_eq!(coupling_probe.nsymbol_calls.load(Ordering::Relaxed), 0);

        let structure = packed_fixture_structure(
            2,
            [(FusionTreePairKey::pair(valid, scalar), vec![1, 1])],
        )
        .unwrap();
        let direct = CheckedTreeProbe::default();
        LocallyValidatedFusionTreeBlockStructure::try_new(&direct, &structure).unwrap();
        let expected_calls = direct.legacy_nsymbol_calls.load(Ordering::Relaxed);
        let admitted = CheckedTreeProbe::default();
        homspace
            .validate_subblock_structure_subset(&admitted, &structure)
            .unwrap();
        assert!(expected_calls > 0);
        assert_eq!(
            admitted.legacy_nsymbol_calls.load(Ordering::Relaxed),
            expected_calls
        );
    }

    #[test]
    fn checked_layout_and_space_admission_reject_finite_nonclosure_transactionally() {
        // What: caller-order layout validation and same-rule legacy admission
        // surface the exact finite-algebra error without publishing a new stamp.
        let scalar_zero = FusionTreeKey::new([], SectorId::new(0), [], [], []);
        let valid = FusionTreeKey::new(
            [SectorId::new(0); 2],
            SectorId::new(0),
            [false; 2],
            [],
            [MultiplicityIndex::ONE],
        );
        let malformed = FusionTreeKey::new(
            [SectorId::new(0); 2],
            SectorId::new(0),
            [false],
            [],
            [MultiplicityIndex::ONE],
        );
        let structural_probe = CheckedTreeProbe::default();
        assert!(matches!(
            BlockStructure::coupled_sector_matrix_with_keys_checked(
                &structural_probe,
                2,
                2,
                vec![
                    (
                        FusionTreePairKey::pair(valid.clone(), scalar_zero.clone()),
                        vec![1, 1],
                    ),
                    (
                        FusionTreePairKey::pair(malformed, scalar_zero.clone()),
                        vec![1, 1],
                    ),
                ],
            ),
            Err(CheckedFusionSpaceError::Core(_))
        ));
        assert_eq!(structural_probe.channel_calls.load(Ordering::Relaxed), 0);
        assert_eq!(structural_probe.nsymbol_calls.load(Ordering::Relaxed), 0);

        let split_probe = CheckedTreeProbe::default();
        assert_eq!(
            BlockStructure::coupled_sector_matrix_with_keys_checked(
                &split_probe,
                3,
                2,
                vec![(
                    FusionTreePairKey::pair(valid.clone(), scalar_zero.clone()),
                    vec![1, 1],
                )],
            ),
            Err(CheckedFusionSpaceError::Core(Box::new(
                CoreError::StructureRankMismatch {
                    expected: 2,
                    actual: 3,
                },
            )))
        );
        assert_eq!(split_probe.channel_calls.load(Ordering::Relaxed), 0);
        assert_eq!(split_probe.nsymbol_calls.load(Ordering::Relaxed), 0);
        assert_eq!(
            BlockStructure::coupled_sector_matrix_with_keys(
                &CheckedTreeProbe::default(),
                3,
                2,
                vec![(FusionTreePairKey::pair(valid, scalar_zero), vec![1, 1])],
            ),
            Err(CoreError::StructureRankMismatch {
                expected: 2,
                actual: 3,
            })
        );

        let overflow = FusionTreeKey::new(
            [u1(i32::MAX), u1(1)],
            u1(0),
            [false; 2],
            [],
            [MultiplicityIndex::ONE],
        );
        let scalar = FusionTreeKey::new([], u1(0), [], [], []);
        let pair = FusionTreePairKey::pair(overflow, scalar);
        let expected = CheckedFusionSpaceError::FusionAlgebra(Box::new(
            FusionAlgebraError::U1FusionOverflow {
                left: i32::MAX,
                right: 1,
            },
        ));
        reset_block_structure_intern_calls();
        assert_eq!(
            BlockStructure::coupled_sector_matrix_with_keys_checked(
                &U1FusionRule,
                2,
                2,
                vec![(pair.clone(), vec![1, 1])],
            ),
            Err(expected.clone())
        );
        assert_eq!(block_structure_intern_calls(), 0);

        let probe = CheckedTreeProbe::default();
        let failing_pair = FusionTreePairKey::pair(
            FusionTreeKey::new(
                [SectorId::new(9), SectorId::new(0)],
                SectorId::new(0),
                [false; 2],
                [],
                [MultiplicityIndex::ONE],
            ),
            FusionTreeKey::new([], SectorId::new(0), [], [], []),
        );
        let homspace = FusionTreeHomSpace::new(
            FusionProductSpace::new([
                SectorLeg::new([(SectorId::new(9), 1)], false),
                SectorLeg::new([(SectorId::new(0), 1)], false),
            ]),
            FusionProductSpace::new([]),
        );
        let legacy = FusionTensorMapSpace::<2, 0>::new_unbound(
            TensorMapSpace::from_dims([1, 1], []).unwrap(),
            homspace,
            packed_fixture_structure(2, [(failing_pair, vec![1, 1])]).unwrap(),
        )
        .unwrap()
        .try_bind_rule(&probe)
        .unwrap();
        assert!(matches!(legacy.admission(), FusionSpaceAdmission::Subset(_)));
        assert_eq!(
            legacy.try_bind_rule_checked(&probe),
            Err(CheckedFusionSpaceError::FusionAlgebra(Box::new(
                FusionAlgebraError::FusionNotRepresentable {
                    left: SectorId::new(9),
                    right: SectorId::new(0),
                },
            )))
        );
    }

    #[test]
    fn checked_coupled_layout_finishes_structural_preflight_before_algebra() {
        // What: incomplete grids, conflicting row extents, and overflowing
        // dimensions fail before checked algebra and publish no structure.
        let tree = |left| {
            FusionTreeKey::new(
                [SectorId::new(left), SectorId::new(0)],
                SectorId::new(0),
                [false; 2],
                [],
                [MultiplicityIndex::ONE],
            )
        };
        let row_zero = tree(0);
        let row_one = tree(1);
        let col_zero = tree(2);
        let col_one = tree(3);

        let wrong_split = CheckedTreeProbe::default();
        let rank_one = FusionTreeKey::new(
            [SectorId::new(0)],
            SectorId::new(0),
            [false],
            [],
            [],
        );
        reset_block_structure_intern_calls();
        assert_eq!(
            BlockStructure::coupled_sector_matrix_with_keys_checked(
                &wrong_split,
                2,
                2,
                vec![(
                    FusionTreePairKey::pair(rank_one.clone(), rank_one),
                    vec![1; 2],
                )],
            ),
            Err(CheckedFusionSpaceError::Core(Box::new(
                CoreError::FusionSpaceSplitMismatch {
                    expected_nout: 2,
                    expected_nin: 0,
                    actual_nout: 1,
                    actual_nin: 1,
                },
            )))
        );
        assert_eq!(wrong_split.channel_calls.load(Ordering::Relaxed), 0);
        assert_eq!(wrong_split.nsymbol_calls.load(Ordering::Relaxed), 0);
        assert_eq!(block_structure_intern_calls(), 0);

        let missing_grid = CheckedTreeProbe::default();
        reset_block_structure_intern_calls();
        assert_eq!(
            BlockStructure::coupled_sector_matrix_with_keys_checked(
                &missing_grid,
                2,
                4,
                vec![
                    (
                        FusionTreePairKey::pair(row_zero.clone(), col_zero.clone()),
                        vec![1; 4],
                    ),
                    (
                        FusionTreePairKey::pair(row_one, col_one.clone()),
                        vec![1; 4],
                    ),
                ],
            ),
            Err(CheckedFusionSpaceError::Core(Box::new(
                CoreError::BlockCountMismatch {
                    expected: 4,
                    actual: 2,
                },
            )))
        );
        assert_eq!(missing_grid.channel_calls.load(Ordering::Relaxed), 0);
        assert_eq!(missing_grid.nsymbol_calls.load(Ordering::Relaxed), 0);
        assert_eq!(block_structure_intern_calls(), 0);

        let conflicting_extent = CheckedTreeProbe::default();
        reset_block_structure_intern_calls();
        assert_eq!(
            BlockStructure::coupled_sector_matrix_with_keys_checked(
                &conflicting_extent,
                2,
                4,
                vec![
                    (
                        FusionTreePairKey::pair(row_zero.clone(), col_zero.clone()),
                        vec![1; 4],
                    ),
                    (
                        FusionTreePairKey::pair(row_zero.clone(), col_one.clone()),
                        vec![2, 1, 1, 1],
                    ),
                ],
            ),
            Err(CheckedFusionSpaceError::Core(Box::new(
                CoreError::DimensionMismatch {
                    expected: 1,
                    actual: 2,
                },
            )))
        );
        assert_eq!(
            conflicting_extent.channel_calls.load(Ordering::Relaxed),
            0
        );
        assert_eq!(
            conflicting_extent.nsymbol_calls.load(Ordering::Relaxed),
            0
        );
        assert_eq!(block_structure_intern_calls(), 0);

        let overflowing_extent = CheckedTreeProbe::default();
        reset_block_structure_intern_calls();
        assert_eq!(
            BlockStructure::coupled_sector_matrix_with_keys_checked(
                &overflowing_extent,
                2,
                4,
                vec![(
                    FusionTreePairKey::pair(row_zero, col_zero),
                    vec![usize::MAX, 2, 1, 1],
                )],
            ),
            Err(CheckedFusionSpaceError::Core(Box::new(
                CoreError::ElementCountOverflow,
            )))
        );
        assert_eq!(
            overflowing_extent.channel_calls.load(Ordering::Relaxed),
            0
        );
        assert_eq!(
            overflowing_extent.nsymbol_calls.load(Ordering::Relaxed),
            0
        );
        assert_eq!(block_structure_intern_calls(), 0);
    }

    #[test]
    fn checked_revalidation_preserves_complete_admission() {
        // What: adding finite-algebra proof to canonical built-in Complete
        // spaces preserves both their layouts and complete-grid admission.
        fn assert_rule<R>(rule: &R, sector: SectorId)
        where
            R: MultiplicityFreeFusionRule + CheckedFusionAlgebra,
        {
            let space = FusionTensorMapSpace::from_degeneracy_shapes(
                TensorMapSpace::<1, 1>::from_dims([1], [1]).unwrap(),
                FusionTreeHomSpace::from_sectors([(sector, 1)], [(sector, 1)]),
                rule,
                [vec![1, 1]],
            )
            .unwrap();
            let structure = Arc::clone(space.subblock_structure());
            assert!(matches!(space.admission(), FusionSpaceAdmission::Complete(_)));
            let checked = space.try_bind_rule_checked(rule).unwrap();
            assert!(matches!(
                checked.admission(),
                FusionSpaceAdmission::Complete(_)
            ));
            assert!(Arc::ptr_eq(&structure, checked.subblock_structure()));
        }

        assert_rule(&Z2FusionRule, z2_even());
        assert_rule(&FermionParityFusionRule, z2_odd());
        assert_rule(&U1FusionRule, u1(7));
        assert_rule(&SU2FusionRule, su2(3));
        assert_rule(&FibonacciFusionRule, SectorId::new(1));

        type Fz2U1 = ProductFusionRule<FermionParityFusionRule, U1FusionRule>;
        type Triple = ProductFusionRule<Fz2U1, SU2FusionRule>;
        let pair = Fz2U1::new(FermionParityFusionRule, U1FusionRule);
        let pair_sector = pair.encode_sector(z2_odd(), u1(2));
        let triple = Triple::new(pair, SU2FusionRule);
        let triple_sector = triple.encode_sector(pair_sector, su2(1));
        assert_rule(&triple, triple_sector);
    }

    fn assert_checked_tree_matches_infallible<R>(rule: &R, tree: FusionTreeKey)
    where
        R: CheckedFusionAlgebra,
    {
        tree.validate_for_rule(rule).unwrap();
        tree.validate_for_rule_checked(rule).unwrap();
    }

    #[test]
    fn checked_tree_matches_closed_builtin_and_nested_product_rules() {
        // What: checked validation accepts the same valid raw trees as the
        // established validator across built-in and recursive product rules.
        assert_checked_tree_matches_infallible(
            &Z2FusionRule,
            FusionTreeKey::new(
                [z2_odd(); 2],
                z2_even(),
                [false; 2],
                [],
                [MultiplicityIndex::ONE],
            ),
        );
        assert_checked_tree_matches_infallible(
            &FermionParityFusionRule,
            FusionTreeKey::new(
                [z2_odd(); 2],
                z2_even(),
                [false; 2],
                [],
                [MultiplicityIndex::ONE],
            ),
        );
        assert_checked_tree_matches_infallible(
            &U1FusionRule,
            FusionTreeKey::new(
                [u1(2), u1(-1)],
                u1(1),
                [false; 2],
                [],
                [MultiplicityIndex::ONE],
            ),
        );
        assert_checked_tree_matches_infallible(
            &SU2FusionRule,
            FusionTreeKey::new(
                [su2(1); 2],
                su2(0),
                [false; 2],
                [],
                [MultiplicityIndex::ONE],
            ),
        );
        assert_checked_tree_matches_infallible(
            &FibonacciFusionRule,
            FusionTreeKey::new(
                [SectorId::new(1); 2],
                SectorId::new(1),
                [false; 2],
                [],
                [MultiplicityIndex::ONE],
            ),
        );

        #[cfg(target_pointer_width = "64")]
        {
            type InnerCodec = PackedProductCodec<Fz2SectorLayout, U1SectorLayout>;
            type InnerRule =
                ProductFusionRule<FermionParityFusionRule, U1FusionRule, InnerCodec>;
            type InnerLayout = ProductSectorLayout<Fz2SectorLayout, U1SectorLayout>;
            type Codec = PackedProductCodec<InnerLayout, Su2SectorLayout>;
            type Rule = ProductFusionRule<InnerRule, SU2FusionRule, Codec>;
            let rule = Rule::new(
                InnerRule::new(FermionParityFusionRule, U1FusionRule),
                SU2FusionRule,
            );
            let sector = |parity, charge, spin| {
                Codec::encode(InnerCodec::encode(parity, charge), spin)
            };
            assert_checked_tree_matches_infallible(
                &rule,
                FusionTreeKey::new(
                    [sector(z2_odd(), u1(2), su2(1)), sector(z2_odd(), u1(-1), su2(1))],
                    sector(z2_even(), u1(1), su2(0)),
                    [false; 2],
                    [],
                    [MultiplicityIndex::ONE],
                ),
            );
        }

        assert_checked_tree_matches_infallible(
            &CheckedTreeProbe::default(),
            FusionTreeKey::new(
                [SectorId::new(0); 2],
                SectorId::new(0),
                [false; 2],
                [],
                [MultiplicityIndex::ONE],
            ),
        );
    }

    #[test]
    fn raw_tree_pair_constructor_rejects_zero_vertices_in_source_order() {
        let domain_vertices = std::iter::once_with(|| {
            panic!("domain vertices must not be consumed after a codomain error")
        });
        let error = FusionTreePairKey::try_pair_from_sector_ids(
            [1, 1],
            [1, 1],
            0,
            [false; 2],
            [false; 2],
            [],
            [],
            [0],
            domain_vertices,
        )
        .unwrap_err();
        // What: the public numeric import rejects zero while constructing the
        // codomain and does not continue into the domain after that error.
        assert_eq!(error, CoreError::InvalidMultiplicityIndex { value: 0 });

        assert_eq!(
            FusionTreePairKey::try_pair_from_sector_ids(
                [1, 1],
                [1, 1],
                0,
                [false; 2],
                [false; 2],
                [],
                [],
                [1],
                [0],
            )
            .unwrap_err(),
            CoreError::InvalidMultiplicityIndex { value: 0 },
        );
    }

    #[test]
    fn tree_pair_validation_requires_exact_coupled_sector() {
        // What: pair compatibility compares the same coupled sector on both
        // sides, including the canonical rank-zero vacuum.
        let empty_pair = FusionTreePairKey::pair(
            FusionTreeKey::new([], u1(0), [], [], []),
            FusionTreeKey::new([], u1(0), [], [], []),
        );
        empty_pair.validate_for_rule(&U1FusionRule).unwrap();
        FusionTreePairKey::pair(
            FusionTreeKey::new([], u1(0), [], [], []),
            FusionTreeKey::new([u1(0)], u1(0), [false], [], []),
        )
        .validate_for_rule(&U1FusionRule)
        .unwrap();
        assert_eq!(
            FusionTreePairKey::pair(
                FusionTreeKey::new([], u1(0), [], [], []),
                FusionTreeKey::new([u1(1)], u1(1), [false], [], []),
            )
            .validate_for_rule(&U1FusionRule)
            .unwrap_err(),
            CoreError::MalformedFusionTree {
                message: "fusion tree pair requires matching coupled sectors",
            }
        );

        let codomain = FusionTreeKey::try_new_for_rule(
            &U1FusionRule,
            [u1(1), u1(2)], u1(3),
            [false; 2],
            [],
            [MultiplicityIndex::ONE],
        )
        .unwrap();
        let same_c_domain = FusionTreeKey::try_new_for_rule(
            &U1FusionRule,
            [u1(4), u1(-1)], u1(3),
            [true, false],
            [],
            [MultiplicityIndex::ONE],
        )
        .unwrap();
        FusionTreePairKey::pair(codomain.clone(), same_c_domain)
            .validate_for_rule(&U1FusionRule)
            .unwrap();
        let dual_c_domain = FusionTreeKey::try_new_for_rule(
            &U1FusionRule,
            [u1(-4), u1(1)], u1(-3),
            [false; 2],
            [],
            [MultiplicityIndex::ONE],
        )
        .unwrap();
        assert_eq!(
            FusionTreePairKey::pair(codomain, dual_c_domain)
                .validate_for_rule(&U1FusionRule)
                .unwrap_err(),
            CoreError::MalformedFusionTree {
                message: "fusion tree pair requires matching coupled sectors",
            }
        );
    }

    #[test]
    fn fusion_tree_block_structure_proof_reports_the_first_physical_invalid_pair() {
        // What: structure admission reports the earliest malformed
        // fusion-tree pair in physical block order.
        let valid = FusionTreePairKey::pair(
            FusionTreeKey::try_new_for_rule(
                &U1FusionRule,
                [u1(1), u1(-1)], u1(0),
                [false; 2],
                [],
                [MultiplicityIndex::ONE],
            )
            .unwrap(),
            FusionTreeKey::new([], u1(0), [], [], []),
        );
        let mismatched_pair = FusionTreePairKey::pair(
            FusionTreeKey::try_new_for_rule(
                &U1FusionRule,
                [u1(1)], u1(1),
                [false],
                [],
                [],
            )
            .unwrap(),
            FusionTreeKey::try_new_for_rule(
                &U1FusionRule,
                [u1(2)], u1(2),
                [false],
                [],
                [],
            )
            .unwrap(),
        );
        let later_bad_shape = FusionTreePairKey::pair(
            FusionTreeKey::new([u1(1), u1(-1)], u1(0), [false], [], [MultiplicityIndex::ONE]),
            FusionTreeKey::new([], u1(0), [], [], []),
        );
        let structure = BlockStructure::from_blocks(vec![
            BlockSpec::column_major_with_key(valid.into(), vec![1, 1], 0).unwrap(),
            BlockSpec::column_major_with_key(mismatched_pair.into(), vec![1, 1], 1).unwrap(),
            BlockSpec::column_major_with_key(later_bad_shape.into(), vec![1, 1], 2).unwrap(),
        ])
        .unwrap();

        let error =
            match LocallyValidatedFusionTreeBlockStructure::try_new(&U1FusionRule, &structure) {
                Ok(_) => panic!("malformed structure unexpectedly admitted"),
                Err(error) => error,
            };
        assert_eq!(
            error,
            CoreError::MalformedFusionTree {
                message: "fusion tree pair requires matching coupled sectors",
            }
        );
    }

    #[test]
    fn fusion_tree_block_structure_proof_rejects_nontrivial_vertex_for_unique_provider() {
        let invalid = FusionTreePairKey::pair(
            FusionTreeKey::new(
                [z2_odd(), z2_odd()],
                z2_even(),
                [false; 2],
                [],
                [MultiplicityIndex::new(2).unwrap()],
            ),
            FusionTreeKey::new([z2_even()], z2_even(), [false], [], []),
        );
        let structure = BlockStructure::from_blocks(vec![
            BlockSpec::column_major_with_key(invalid.into(), vec![1, 1, 1], 0).unwrap(),
        ])
        .unwrap();

        // What: a raw label-two key cannot acquire the local proof required
        // by compact multiplicity-free batch execution.
        let error =
            match LocallyValidatedFusionTreeBlockStructure::try_new(&Z2FusionRule, &structure) {
                Ok(_) => panic!("nontrivial multiplicity label unexpectedly admitted"),
                Err(error) => error,
            };
        assert_eq!(
            error,
            CoreError::MalformedFusionTree {
                message: "fusion tree vertex label exceeds its fusion multiplicity",
            }
        );
    }

    #[test]
    fn coupled_sector_constructor_validates_in_caller_order_before_sorting() {
        let first = FusionTreePairKey::pair(
            FusionTreeKey::new(
                [SectorId::new(1), SectorId::new(2)],
                SectorId::new(3),
                [false; 2],
                [],
                [MultiplicityIndex::new(2).unwrap()],
            ),
            FusionTreeKey::new(
                [SectorId::new(3)],
                SectorId::new(3),
                [false],
                [],
                [],
            ),
        );
        let later_lower_coupled = FusionTreePairKey::pair(
            FusionTreeKey::new(
                [SectorId::new(0), SectorId::new(1)],
                SectorId::new(1),
                [false],
                [],
                [MultiplicityIndex::ONE],
            ),
            FusionTreeKey::new(
                [SectorId::new(1)],
                SectorId::new(1),
                [false],
                [],
                [],
            ),
        );

        // What: the first caller-supplied categorical error wins even though
        // coupled-sector layout order would move the later key before it.
        assert_eq!(
            BlockStructure::coupled_sector_matrix_with_keys(
                &IdentitySymbolPanicRule,
                2,
                3,
                vec![
                    (first, vec![1, 1, 1]),
                    (later_lower_coupled, vec![1, 1, 1]),
                ],
            )
            .unwrap_err(),
            CoreError::MalformedFusionTree {
                message: "fusion tree vertex label exceeds its fusion multiplicity",
            }
        );
    }

    #[test]
    fn fusion_tree_block_structure_proof_rejects_non_categorical_namespaces() {
        // What: LOCAL categorical admission rejects both anonymous dense
        // storage and arbitrary application routing keys.
        for key in [BlockKey::Dense, BlockKey::opaque([7, 11])] {
            let structure = BlockStructure::from_blocks(vec![
                BlockSpec::column_major_with_key(key.clone(), vec![1], 0).unwrap(),
            ])
            .unwrap();
            let error =
                match LocallyValidatedFusionTreeBlockStructure::try_new(&Z2FusionRule, &structure) {
                    Ok(_) => panic!("non-categorical structure unexpectedly admitted"),
                    Err(error) => error,
                };
            assert_eq!(
                error,
                CoreError::ExpectedFusionTreePairKey { actual: key.kind() }
            );
        }
    }

    #[test]
    fn fusion_tree_block_structure_proof_rejects_local_rank_mismatch() {
        // What: categorical admission requires every tree pair to describe the
        // same number of external legs as its containing block structure.
        let key = FusionTreePairKey::pair(
            FusionTreeKey::try_new_for_rule(
                &Z2FusionRule,
                [z2_even()], z2_even(),
                [false],
                [],
                [],
            )
            .unwrap(),
            FusionTreeKey::new([], z2_even(), [], [], []),
        );
        let structure = BlockStructure::from_blocks(vec![
            BlockSpec::column_major_with_key(key.into(), vec![1, 1], 0).unwrap(),
        ])
        .unwrap();

        let error = match LocallyValidatedFusionTreeBlockStructure::try_new(&Z2FusionRule, &structure) {
            Ok(_) => panic!("rank-mismatched structure unexpectedly admitted"),
            Err(error) => error,
        };
        assert_eq!(
            error,
            CoreError::StructureRankMismatch {
                expected: 2,
                actual: 1,
            }
        );
    }

    #[test]
    fn malformed_compact_tree_key_is_panic_free_for_structure_lookup() {
        // What: a raw one-leg key with missing dual metadata remains a fallible
        // categorical input rather than panicking during structure indexing.
        let malformed = FusionTreePairKey::pair(
            FusionTreeKey::new([z2_even()], z2_even(), [], [], []),
            FusionTreeKey::new([], z2_even(), [], [], []),
        );
        let structure = BlockStructure::from_blocks(vec![
            BlockSpec::column_major_with_key(malformed.clone().into(), vec![1], 0).unwrap(),
        ])
        .unwrap();

        assert_eq!(
            structure.find_block_index_by_fusion_tree_pair(&malformed),
            Some(0)
        );
        assert_eq!(
            structure.fusion_tree_pair_block(&malformed).unwrap().key(),
            &BlockKey::from(malformed)
        );
    }

    #[test]
    fn fusion_tree_block_structure_proof_indexes_only_its_bound_structure() {
        // What: a successful proof exposes borrowed keys by physical index from
        // its exact categorical structure, including out-of-bounds errors.
        let valid = FusionTreePairKey::pair(
            FusionTreeKey::try_new_for_rule(
                &U1FusionRule,
                [u1(1), u1(-1)], u1(0),
                [false; 2],
                [],
                [MultiplicityIndex::ONE],
            )
            .unwrap(),
            FusionTreeKey::new([], u1(0), [], [], []),
        );
        let structure = BlockStructure::from_blocks(vec![
            BlockSpec::column_major_with_key(valid.clone().into(), vec![1, 1], 0).unwrap(),
        ])
        .unwrap();

        let proof =
            LocallyValidatedFusionTreeBlockStructure::try_new(&U1FusionRule, &structure).unwrap();
        assert!(std::ptr::eq(proof.rule(), &U1FusionRule));
        assert!(std::ptr::eq(proof.structure(), &structure));
        let canonical = proof.fusion_tree_pair_key(0);
        #[allow(deprecated)]
        let legacy = proof.fusion_tree_block_key(0);
        assert_eq!(canonical, legacy);
        assert_eq!(canonical.unwrap(), Some(&valid));
        let canonical_missing = proof.fusion_tree_pair_key(1);
        #[allow(deprecated)]
        let legacy_missing = proof.fusion_tree_block_key(1);
        assert_eq!(canonical_missing, legacy_missing);
        assert_eq!(
            canonical_missing.unwrap_err(),
            CoreError::BlockIndexOutOfBounds { index: 1, count: 1 }
        );
    }

    #[test]
    fn validated_per_index_permute_preserves_planar_braiding_rejection_before_index_lookup() {
        let structure = BlockStructure::empty(0);
        let proof =
            LocallyValidatedFusionTreeBlockStructure::try_new(&PlanarZ2Rule, &structure).unwrap();

        // What: a proof-consuming per-index call observes the symmetric-
        // braiding boundary before attempting to read its requested block.
        assert_eq!(
            proof.permute_codomain_rows_for_block_index(0, &[]),
            Err(CoreError::UnsupportedBraidingStyle {
                expected: "symmetric braiding",
                actual: BraidingStyleKind::NoBraiding,
            })
        );
    }

    #[test]
    fn multiplicity_free_block_boundaries_reject_misreported_style_before_empty_identity() {
        let rule = MisreportedGenericMultiplicityFreeRule;
        let structure = BlockStructure::empty(0);
        let proof =
            LocallyValidatedFusionTreeBlockStructure::try_new(&rule, &structure).unwrap();
        let identity = PreparedTreePairOperation::prepare_transpose(0, 0, &[], &[]).unwrap();
        let expected = CoreError::UnsupportedFusionStyle {
            expected: FusionStyleKind::Simple,
            actual: FusionStyleKind::Generic,
        };

        // What: neither proof-bound nor raw block entry points let empty or
        // identity work bypass the runtime fusion-style capability.
        assert_eq!(
            proof
                .execute_multiplicity_free_braid_for_block_indices(
                    std::iter::empty(),
                    identity.clone(),
                )
                .unwrap_err(),
            expected
        );
        assert_eq!(
            proof
                .execute_multiplicity_free_transpose_for_block_indices(
                    std::iter::empty(),
                    identity,
                )
                .unwrap_err(),
            expected
        );
        assert_eq!(
            multiplicity_free_braid_tree_block(&rule, &[], &[], &[]).unwrap_err(),
            expected
        );
        assert_eq!(
            multiplicity_free_permute_tree_block(&rule, &[], &[]).unwrap_err(),
            expected
        );
        assert_eq!(
            multiplicity_free_braid_tree_pair_block(&rule, &[], &[], &[], &[], &[])
                .unwrap_err(),
            expected
        );
        assert_eq!(
            multiplicity_free_permute_tree_pair_block(&rule, &[], &[], &[]).unwrap_err(),
            expected
        );
        assert_eq!(
            multiplicity_free_transpose_tree_pair_block(&rule, &[], &[], &[]).unwrap_err(),
            expected
        );
    }

    #[test]
    fn unique_proof_hook_rechecks_style_and_prepared_source_rank() {
        // What: downstream callers cannot use a general categorical proof to
        // bypass the Unique-style or prepared source-split contract.
        let vacuum = su2(0);
        let half = su2(1);
        let simple_pair = FusionTreePairKey::pair(
            FusionTreeKey::try_new_for_rule(
                &SU2FusionRule,
                [half, half], vacuum,
                [false; 2],
                [],
                [MultiplicityIndex::ONE],
            )
            .unwrap(),
            FusionTreeKey::try_new_for_rule(
                &SU2FusionRule,
                [], vacuum,
                [],
                [],
                [],
            )
            .unwrap(),
        );
        let simple_structure =
            packed_fixture_structure(2, [(simple_pair, vec![1, 1])]).unwrap();
        let simple_proof =
            LocallyValidatedFusionTreeBlockStructure::try_new(&SU2FusionRule, &simple_structure).unwrap();
        let rank_two_identity =
            PreparedTreePairOperation::prepare_transpose(2, 0, &[0, 1], &[]).unwrap();
        assert_eq!(
            simple_proof
                .execute_unique_rigid_for_block_index(0, &rank_two_identity)
                .unwrap_err(),
            CoreError::UnsupportedFusionStyle {
                expected: FusionStyleKind::Unique,
                actual: FusionStyleKind::Simple,
            }
        );

        let unique_pair = FusionTreePairKey::pair(
            FusionTreeKey::try_new_for_rule(
                &U1FusionRule,
                [u1(1), u1(-1)], u1(0),
                [false; 2],
                [],
                [MultiplicityIndex::ONE],
            )
            .unwrap(),
            FusionTreeKey::try_new_for_rule(&U1FusionRule, [], u1(0), [], [], []).unwrap(),
        );
        let unique_structure =
            packed_fixture_structure(2, [(unique_pair, vec![1, 1])]).unwrap();
        let unique_proof =
            LocallyValidatedFusionTreeBlockStructure::try_new(&U1FusionRule, &unique_structure).unwrap();
        let rank_one_identity =
            PreparedTreePairOperation::prepare_transpose(1, 0, &[0], &[]).unwrap();
        assert_eq!(
            unique_proof
                .execute_unique_rigid_for_block_index(0, &rank_one_identity)
                .unwrap_err(),
            CoreError::DimensionMismatch {
                expected: 1,
                actual: 2,
            }
        );
    }

    #[test]
    fn borrowed_block_execution_rejects_prepared_family_and_source_split_mismatch() {
        let vacuum = su2(0);
        let half = su2(1);
        let pair = FusionTreePairKey::pair(
            FusionTreeKey::try_new_for_rule(
                &SU2FusionRule,
                [half, half],
                vacuum,
                [false; 2],
                [],
                [MultiplicityIndex::ONE],
            )
            .unwrap(),
            FusionTreeKey::try_new_for_rule(
                &SU2FusionRule,
                [],
                vacuum,
                [],
                [],
                [],
            )
            .unwrap(),
        );
        let structure = packed_fixture_structure(2, [(pair, vec![1, 1])]).unwrap();
        let proof =
            LocallyValidatedFusionTreeBlockStructure::try_new(&SU2FusionRule, &structure).unwrap();
        let transpose =
            PreparedTreePairOperation::prepare_transpose(2, 0, &[1], &[0]).unwrap();
        let transpose_identity =
            PreparedTreePairOperation::prepare_transpose(2, 0, &[0, 1], &[]).unwrap();
        let braid =
            PreparedTreePairOperation::prepare_permute(&SU2FusionRule, 2, 0, &[1, 0], &[])
                .unwrap();
        let wrong_split =
            PreparedTreePairOperation::prepare_permute(&SU2FusionRule, 1, 1, &[1], &[0])
                .unwrap();

        // What: safe borrowed block executors reject a prepared operation from
        // the other family before its private plan variant reaches execution.
        assert_eq!(
            proof
                .execute_multiplicity_free_braid_ordered_for_block_indices_borrowed(
                    [0],
                    &transpose,
                )
                .unwrap_err(),
            CoreError::MalformedFusionTree {
                message: "prepared tree-pair operation is incompatible with braid block execution",
            }
        );
        assert_eq!(
            proof
                .execute_multiplicity_free_braid_ordered_for_block_indices_borrowed(
                    [0],
                    &transpose_identity,
                )
                .unwrap_err(),
            CoreError::MalformedFusionTree {
                message: "prepared tree-pair operation is incompatible with braid block execution",
            }
        );
        assert_eq!(
            proof
                .execute_multiplicity_free_transpose_ordered_for_block_indices_borrowed(
                    [0], &braid,
                )
                .unwrap_err(),
            CoreError::MalformedFusionTree {
                message:
                    "prepared tree-pair operation is incompatible with transpose block execution",
            }
        );

        // What: operation family is a source-independent preflight. Empty and
        // invalid-index calls cannot bypass it, while a valid empty operation
        // remains the empty linear map.
        assert_eq!(
            proof
                .execute_multiplicity_free_braid_ordered_for_block_indices_borrowed(
                    std::iter::empty(),
                    &transpose,
                )
                .unwrap_err(),
            CoreError::MalformedFusionTree {
                message: "prepared tree-pair operation is incompatible with braid block execution",
            }
        );
        assert_eq!(
            proof
                .execute_multiplicity_free_transpose_ordered_for_block_indices_borrowed(
                    std::iter::empty(),
                    &braid,
                )
                .unwrap_err(),
            CoreError::MalformedFusionTree {
                message:
                    "prepared tree-pair operation is incompatible with transpose block execution",
            }
        );
        assert_eq!(
            proof
                .execute_multiplicity_free_braid_ordered_for_block_indices(
                    [usize::MAX],
                    transpose.clone(),
                )
                .unwrap_err(),
            CoreError::MalformedFusionTree {
                message: "prepared tree-pair operation is incompatible with braid block execution",
            }
        );
        assert_eq!(
            proof
                .execute_multiplicity_free_transpose_ordered_for_block_indices(
                    [usize::MAX],
                    braid.clone(),
                )
                .unwrap_err(),
            CoreError::MalformedFusionTree {
                message:
                    "prepared tree-pair operation is incompatible with transpose block execution",
            }
        );
        for empty in [
            proof
                .execute_multiplicity_free_braid_ordered_for_block_indices_borrowed(
                    std::iter::empty(),
                    &braid,
                )
                .unwrap(),
            proof
                .execute_multiplicity_free_transpose_ordered_for_block_indices_borrowed(
                    std::iter::empty(),
                    &transpose,
                )
                .unwrap(),
        ] {
            assert!(empty.destinations().is_empty());
            assert_eq!(empty.source_count(), 0);
        }

        // What: preparation for the right operation family is still bound to
        // its exact source codomain/domain split.
        assert_eq!(
            proof
                .execute_multiplicity_free_braid_ordered_for_block_indices_borrowed(
                    [0],
                    &wrong_split,
                )
                .unwrap_err(),
            CoreError::DimensionMismatch {
                expected: 1,
                actual: 2,
            }
        );
    }

    #[test]
    fn block_preflight_rejects_prepared_capability_before_empty_or_invalid_source() {
        let structure = BlockStructure::empty(0);
        let proof =
            LocallyValidatedFusionTreeBlockStructure::try_new(&PlanarZ2Rule, &structure).unwrap();
        let symmetric_identity =
            PreparedTreePairOperation::prepare_permute(&Z2FusionRule, 0, 0, &[], &[]).unwrap();
        let expected = CoreError::UnsupportedBraidingStyle {
            expected: "symmetric braiding",
            actual: BraidingStyleKind::NoBraiding,
        };

        // What: rule capability belongs to operation preflight, so neither an
        // empty iterator nor an invalid source index can bypass it.
        assert_eq!(
            proof
                .execute_multiplicity_free_braid_ordered_for_block_indices_borrowed(
                    std::iter::empty(),
                    &symmetric_identity,
                )
                .unwrap_err(),
            expected
        );
        assert_eq!(
            proof
                .execute_multiplicity_free_braid_ordered_for_block_indices(
                    [usize::MAX],
                    symmetric_identity,
                )
                .unwrap_err(),
            expected
        );
    }

    #[test]
    fn rule_validation_preserves_builtin_multiplicity_free_keys() {
        // What: validation is observational for representative abelian,
        // non-abelian, anyonic, fermionic, and product fusion trees.
        fn assert_preserved<R: FusionRule>(rule: &R, tree: FusionTreeKey) {
            let before = tree.clone();
            tree.validate_for_rule(rule).unwrap();
            assert_eq!(tree, before);
        }

        assert_preserved(
            &U1FusionRule,
            FusionTreeKey::try_new_for_rule(
                &U1FusionRule,
                [u1(1), u1(-2), u1(3)], u1(2),
                [false; 3],
                [u1(-1)],
                [MultiplicityIndex::ONE; 2],
            )
            .unwrap(),
        );
        let half = su2(1);
        let one = su2(2);
        assert_preserved(
            &SU2FusionRule,
            FusionTreeKey::try_new_for_rule(
                &SU2FusionRule,
                [half, half, one], one,
                [false, true, false],
                [su2(0)],
                [MultiplicityIndex::ONE; 2],
            )
            .unwrap(),
        );
        assert_preserved(
            &FibonacciFusionRule,
            FusionTreeKey::try_new_for_rule(
                &FibonacciFusionRule,
                [SectorId::new(1); 3], SectorId::new(1),
                [false; 3],
                [SectorId::new(0)],
                [MultiplicityIndex::ONE; 2],
            )
            .unwrap(),
        );
        assert_preserved(
            &FermionParityFusionRule,
            FusionTreeKey::try_new_for_rule(
                &FermionParityFusionRule,
                [z2_odd(); 3], z2_odd(),
                [false, true, false],
                [z2_even()],
                [MultiplicityIndex::ONE; 2],
            )
            .unwrap(),
        );

        type ProductRule = ProductFusionRule<U1FusionRule, SU2FusionRule>;
        let product = ProductRule::default();
        let left = product.encode_sector(u1(1), half);
        let right = product.encode_sector(u1(-1), half);
        let coupled = product.encode_sector(u1(0), su2(0));
        assert_preserved(
            &product,
            FusionTreeKey::try_new_for_rule(
                &product,
                [left, right], coupled,
                [false; 2],
                [],
                [MultiplicityIndex::ONE],
            )
            .unwrap(),
        );
    }

    #[test]
    fn inadmissible_tree_is_rejected_before_every_identity_path() {
        // What: a shape-correct but inadmissible tree cannot pass through
        // scalar, prepared, or whole-block identity shortcuts.
        fn assert_inadmissible<T: std::fmt::Debug>(result: Result<T, CoreError>) {
            assert_eq!(
                result.unwrap_err(),
                CoreError::MalformedFusionTree {
                    message: "fusion tree contains an inadmissible fusion vertex",
                }
            );
        }

        let rule = SplitOnlyCountingRule::default();
        let odd = SectorId::new(1);
        let vacuum = SectorId::new(0);
        let source = FusionTreeKey::new(
            [odd; 3], vacuum,
            [false; 3],
            [vacuum],
            [MultiplicityIndex::ONE; 2],
        );
        let identity = [0, 1, 2];
        let levels = [0, 1, 2];
        assert_inadmissible(multiplicity_free_braid_tree(&rule, &source, &identity, &levels));
        assert_inadmissible(multiplicity_free_permute_tree(
            &rule, &source, &identity,
        ));
        assert_inadmissible(multiplicity_free_braid_tree_block(
            &rule,
            std::slice::from_ref(&source),
            &identity,
            &levels,
        ));
        assert_inadmissible(multiplicity_free_permute_tree_block(
            &rule,
            std::slice::from_ref(&source),
            &identity,
        ));

        let pair = FusionTreePairKey::pair(
            source,
            FusionTreeKey::new([], vacuum, [], [], []),
        );
        let prepared = PreparedTreePairOperation::prepare_braid(
            &rule,
            3,
            0,
            &identity,
            &[],
            &levels,
            &[],
        )
        .unwrap();
        assert_inadmissible(prepared.execute_multiplicity_free(&rule, &pair));
        assert_inadmissible(multiplicity_free_braid_tree_pair(
            &rule, &pair, &identity, &[], &levels, &[],
        ));
        assert_inadmissible(multiplicity_free_permute_tree_pair(
            &rule, &pair, &identity, &[],
        ));
        assert_inadmissible(multiplicity_free_transpose_tree_pair(
            &rule, &pair, &identity, &[],
        ));
        assert_inadmissible(multiplicity_free_braid_tree_pair_block(
            &rule,
            std::slice::from_ref(&pair),
            &identity,
            &[],
            &levels,
            &[],
        ));
        assert_inadmissible(multiplicity_free_permute_tree_pair_block(
            &rule,
            std::slice::from_ref(&pair),
            &identity,
            &[],
        ));
        assert_inadmissible(multiplicity_free_transpose_tree_pair_block(
            &rule,
            std::slice::from_ref(&pair),
            &identity,
            &[],
        ));
        assert_eq!(
            rule.f_calls.load(std::sync::atomic::Ordering::Relaxed),
            0
        );
        assert_eq!(
            rule.r_calls.load(std::sync::atomic::Ordering::Relaxed),
            0
        );
    }

    #[test]
    fn public_operation_errors_precede_categorical_source_errors() {
        // What: deterministic argument and capability failures retain their
        // public precedence over a malformed source.
        let tau = SectorId::new(1);
        let vacuum = SectorId::new(0);
        let invalid = FusionTreeKey::new(
            [tau; 3], vacuum,
            [false; 3],
            [vacuum],
            [MultiplicityIndex::ONE; 2],
        );
        assert_eq!(
            multiplicity_free_braid_tree(
                &FibonacciFusionRule,
                &invalid,
                &[0, 1, 2],
                &[],
            )
            .unwrap_err(),
            CoreError::DimensionMismatch {
                expected: 3,
                actual: 0,
            }
        );
        assert!(matches!(
            multiplicity_free_braid_tree(
                &FibonacciFusionRule,
                &invalid,
                &[0, 0, 2],
                &[0, 1, 2],
            ),
            Err(CoreError::InvalidPermutation { .. })
        ));
        assert_eq!(
            unique_braid_tree(
                &FibonacciFusionRule,
                &invalid,
                &[0, 1, 2],
                &[0, 1, 2],
            )
            .unwrap_err(),
            CoreError::UnsupportedFusionStyle {
                expected: FusionStyleKind::Unique,
                actual: FusionStyleKind::Simple,
            }
        );
        assert!(matches!(
            multiplicity_free_permute_tree(
                &FibonacciFusionRule,
                &invalid,
                &[0, 1, 2],
            ),
            Err(CoreError::UnsupportedBraidingStyle { .. })
        ));
        assert_eq!(
            multiplicity_free_braid_tree_block(
                &FibonacciFusionRule,
                std::slice::from_ref(&invalid),
                &[0, 1, 2],
                &[],
            )
            .unwrap_err(),
            CoreError::DimensionMismatch {
                expected: 3,
                actual: 0,
            }
        );

        let pair = FusionTreePairKey::pair(
            invalid,
            FusionTreeKey::new([], vacuum, [], [], []),
        );
        let wrong_rank = PreparedTreePairOperation::prepare_braid(
            &FibonacciFusionRule,
            2,
            0,
            &[0, 1],
            &[],
            &[0, 1],
            &[],
        )
        .unwrap();
        assert_eq!(
            wrong_rank
                .execute_multiplicity_free(&FibonacciFusionRule, &pair)
                .unwrap_err(),
            CoreError::DimensionMismatch {
                expected: 2,
                actual: 3,
            }
        );
    }

    #[test]
    fn malformed_sources_do_not_evaluate_symbols_or_bends() {
        // What: categorical rejection occurs before F, R, or pivotal bend
        // providers can observe a malformed source.
        let rule = SplitOnlyCountingRule::default();
        let invalid = FusionTreeKey::new(
            [SectorId::new(1); 2], SectorId::new(1),
            [false; 2],
            [],
            [MultiplicityIndex::ONE],
        );
        assert!(unique_braid_tree(&rule, &invalid, &[0, 1], &[0, 1]).is_err());
        let pair = FusionTreePairKey::pair(
            invalid,
            FusionTreeKey::new(
                [SectorId::new(1)], SectorId::new(1),
                [false],
                [],
                [],
            ),
        );
        assert!(unique_repartition_tree_pair(&rule, &pair, 1).is_err());
        assert_eq!(
            rule.f_calls.load(std::sync::atomic::Ordering::Relaxed),
            0
        );
        assert_eq!(
            rule.r_calls.load(std::sync::atomic::Ordering::Relaxed),
            0
        );
        assert_eq!(
            rule.bend_calls
                .load(std::sync::atomic::Ordering::Relaxed),
            0
        );
    }

    #[test]
    fn block_validation_is_source_major_and_runs_once_per_source() {
        // What: block proofs report the first source in slice order and do one
        // N-symbol validation pass before identity execution.
        let tau = SectorId::new(1);
        let vacuum = SectorId::new(0);
        let invalid_first = FusionTreeKey::new(
            [tau; 3], vacuum,
            [false; 3],
            [vacuum],
            [MultiplicityIndex::ONE; 2],
        );
        let valid_first = FusionTreeKey::new(
            [tau; 3], tau,
            [false; 3],
            [vacuum],
            [MultiplicityIndex::ONE; 2],
        );
        let different_group = FusionTreeKey::new(
            [tau; 2], vacuum,
            [false; 2],
            [],
            [MultiplicityIndex::ONE],
        );
        assert_eq!(
            multiplicity_free_braid_tree_block(
                &FibonacciFusionRule,
                &[invalid_first, different_group.clone()],
                &[0, 1, 2],
                &[0, 1, 2],
            )
            .unwrap_err(),
            CoreError::MalformedFusionTree {
                message: "fusion tree contains an inadmissible fusion vertex",
            }
        );
        assert_eq!(
            multiplicity_free_braid_tree_block(
                &FibonacciFusionRule,
                &[valid_first, different_group],
                &[0, 1, 2],
                &[0, 1, 2],
            )
            .unwrap_err(),
            CoreError::MalformedFusionTree {
                message: "fusion-tree keys must share one group",
            }
        );

        let rule = SplitOnlyCountingRule::default();
        let valid = FusionTreeKey::new(
            [SectorId::new(1); 2], SectorId::new(0),
            [false; 2],
            [],
            [MultiplicityIndex::ONE],
        );
        let rows = multiplicity_free_braid_tree_block(
            &rule,
            &[valid.clone(), valid.clone()],
            &[0, 1],
            &[0, 1],
        )
        .unwrap();
        assert_eq!(rows.len(), 2);
        assert_eq!(
            rule.n_calls.load(std::sync::atomic::Ordering::Relaxed),
            2
        );

        rule.n_calls
            .store(0, std::sync::atomic::Ordering::Relaxed);
        let pair = FusionTreePairKey::pair(
            valid,
            FusionTreeKey::new(
                [SectorId::new(0)], SectorId::new(0),
                [false],
                [],
                [],
            ),
        );
        let rows = multiplicity_free_permute_tree_pair_block(
            &rule,
            &[pair.clone(), pair],
            &[0, 1],
            &[2],
        )
        .unwrap();
        assert_eq!(rows.len(), 2);
        assert_eq!(
            rule.n_calls.load(std::sync::atomic::Ordering::Relaxed),
            2
        );
    }

    #[test]
    fn unique_direct_braid_eligibility_excludes_rank_zero() {
        // What: the canonical empty tree carries the vacuum, but no rank-zero
        // operation is eligible for a direct braid rebuild.
        let empty = FusionTreeKey::try_new_for_rule(
            &Z2FusionRule,
            [],
            Z2FusionRule.vacuum(),
            [],
            [],
            [],
        )
        .unwrap();

        assert_eq!(empty.coupled(), Z2FusionRule.vacuum());
        assert!(!is_unique_direct_braid_source(&Z2FusionRule, &empty));
    }

    #[test]
    fn uncertified_custom_symbols_stay_on_general_artin_path() {
        // What: the public provider trait does not runtime-check coherence, so
        // arbitrary custom F/R data must remain on the default-false Artin
        // boundary instead of claiming TensorKit's certified F=1 shortcut.
        let rule = UncertifiedCustomSymbolsRule;
        assert!(!rule.has_trivial_associator_gauge());
        let tree =
            FusionTreeKey::try_from_sector_ids([1, 1, 1], 1, [false, false, false], [0], [1, 1]).unwrap();

        let (destination, coefficient) =
            unique_braid_tree(&rule, &tree, &[0, 2, 1], &[0, 1, 2]).unwrap();
        let expected = unique_artin_braid_at_with_inverse(&rule, &tree, 1, false).unwrap();

        assert_eq!((destination, coefficient), expected);
        assert_eq!(coefficient, 30.0);
    }

    #[test]
    fn complex_unique_prepared_steps_keep_asymmetric_inverse_orientation() {
        // What: prepared Artin steps retain Complex64 conjugation and reverse
        // R-symbol argument order for an inverse anyonic crossing.
        let rule = ComplexAsymmetricUniqueRule;
        let tree = FusionTreeKey::try_from_sector_ids([1, 2], 3, [false, true], [], [1]).unwrap();

        let forward = unique_braid_tree(&rule, &tree, &[1, 0], &[0, 1]).unwrap();
        let inverse = unique_braid_tree(&rule, &tree, &[1, 0], &[1, 0]).unwrap();
        let expected_forward =
            Complex64::from_polar(1.0, std::f64::consts::FRAC_PI_3);
        let expected_inverse =
            Complex64::from_polar(1.0, -std::f64::consts::FRAC_PI_6);

        assert_eq!(forward.0, inverse.0);
        assert!((forward.1 - expected_forward).norm() < 1.0e-12);
        assert!((inverse.1 - expected_inverse).norm() < 1.0e-12);
    }

    #[test]
    fn prepared_tree_pair_operation_size_excludes_a_second_step_arena() {
        // What: the expert plan stores one inline Artin lowering, not parallel
        // Artin and inversion arrays for mutually exclusive execution paths.
        assert!(std::mem::size_of::<PreparedTreePairOperation>() <= 600);
    }

    #[test]
    fn unique_braid_tree_rejects_invalid_permutation_and_level_count() {
        let tree = FusionTreeKey::try_from_sector_ids([1, 2], 3, [false, false], [], [1]).unwrap();

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
        let tree = FusionTreeKey::try_from_sector_ids([1, 2], 3, [false, false], [], [1]).unwrap();

        let err = unique_permute_tree(&AsymmetricAnyonicRule, &tree, &[1, 0]).unwrap_err();

        assert_eq!(
            err,
            CoreError::UnsupportedBraidingStyle {
                expected: "symmetric braiding",
                actual: BraidingStyleKind::Anyonic,
            }
        );
    }

    // --- Stage 0 spike: complex-`Scalar` fusion rule through the recoupling
    // engine. tenet-core has so far only ever instantiated `Scalar = f64`
    // providers; Fibonacci anyons need `Scalar = Complex64`. This probe rule
    // is *not* a physical anyon model (its F-symbol is a constant 1, so it
    // makes no pentagon claim) — it exists purely to prove that
    // `multiplicity_free_braid_tree` / `FusionTermAccumulator` compile and
    // run correctly when `Scalar: num_complex::Complex64` (Add/Mul/Clone from
    // `num_complex`, plus a genuinely complex `scalar_conj` = `.conj()`).
    #[derive(Clone, Copy, Debug)]
    struct ComplexScalarProbeRule;

    impl FusionRule for ComplexScalarProbeRule {
        fn rule_identity(&self) -> RuleIdentity { RuleIdentity::of_type::<Self>() }
        fn fusion_style(&self) -> FusionStyleKind {
            FusionStyleKind::Simple
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
                // Fibonacci-shaped multi-channel fusion: x⊗x = {vacuum, x}.
                // This is what forces `FusionStyleKind::Simple` (not `Unique`)
                // and exercises the multi-term loop in the braid engine.
                (1, 1) => smallvec![SectorId::new(0), SectorId::new(1)],
                _ => SectorVec::new(),
            }
        }
    }

    impl MultiplicityFreeFusionRule for ComplexScalarProbeRule {}

    const PROBE_ANGLE_ALPHA: f64 = std::f64::consts::FRAC_PI_3;
    const PROBE_ANGLE_BETA: f64 = 2.0 * std::f64::consts::FRAC_PI_3;

    impl MultiplicityFreeFusionSymbols for ComplexScalarProbeRule {
        type Scalar = num_complex::Complex64;

        fn scalar_one(&self) -> Self::Scalar {
            num_complex::Complex64::new(1.0, 0.0)
        }

        fn scalar_conj(&self, value: Self::Scalar) -> Self::Scalar {
            value.conj()
        }

        // Trivial associator (1 on every allowed channel, since
        // `fusion_channels` already zeroes out disallowed ones via the
        // engine's `nsymbol` gate): this probe only needs to exercise the
        // complex-scalar plumbing, not satisfy the pentagon identity.
        fn f_symbol_scalar(
            &self,
            _left: SectorId,
            _middle: SectorId,
            _right: SectorId,
            _coupled: SectorId,
            _left_coupled: SectorId,
            _right_coupled: SectorId,
        ) -> Self::Scalar {
            num_complex::Complex64::new(1.0, 0.0)
        }

        // The one place a genuine complex phase enters: R^{xx}_vacuum = e^{iα},
        // R^{xx}_x = e^{iβ}, distinct angles so the two channels are
        // distinguishable in the assertions below.
        fn r_symbol_scalar(
            &self,
            left: SectorId,
            right: SectorId,
            coupled: SectorId,
        ) -> Self::Scalar {
            if self.nsymbol(left, right, coupled) == 0 {
                return num_complex::Complex64::new(0.0, 0.0);
            }
            if left.id() == 0 || right.id() == 0 {
                return num_complex::Complex64::new(1.0, 0.0);
            }
            if coupled.id() == 0 {
                num_complex::Complex64::from_polar(1.0, PROBE_ANGLE_ALPHA)
            } else {
                num_complex::Complex64::from_polar(1.0, PROBE_ANGLE_BETA)
            }
        }
    }

    #[test]
    fn complex_scalar_r_symbol_and_conjugate_inverse_braid_stage0_spike() {
        let rule = ComplexScalarProbeRule;
        for coupled in [0usize, 1usize] {
            let tree = FusionTreeKey::try_from_sector_ids([1, 1], coupled, [false, false], [], [1]).unwrap();
            let expected = num_complex::Complex64::from_polar(
                1.0,
                if coupled == 0 {
                    PROBE_ANGLE_ALPHA
                } else {
                    PROBE_ANGLE_BETA
                },
            );

            let forward = multiplicity_free_braid_tree(&rule, &tree, &[1, 0], &[0, 1]).unwrap();
            assert_eq!(forward.len(), 1);
            assert!((forward[0].1 - expected).norm() < 1.0e-12);

            // Reflected levels select the inverse-artin branch: the
            // coefficient must come back as the complex conjugate, proving
            // `scalar_conj` (not just `Clone`/`Mul`) is wired through for a
            // non-real `Scalar`.
            let backward = multiplicity_free_braid_tree(&rule, &tree, &[1, 0], &[1, 0]).unwrap();
            assert_eq!(backward.len(), 1);
            assert!((backward[0].1 - expected.conj()).norm() < 1.0e-12);
        }
    }

    #[test]
    fn complex_scalar_braid_tree_expands_multichannel_loop_stage0_spike() {
        // Rank-3 tree with an index>0 swap: exercises the `fusion_channels(a,
        // d)` loop branch of `multiplicity_free_artin_braid_at_with_inverse`
        // (f_symbol_scalar * r_symbol_scalar * scalar_conj composition) —
        // this is the part of the engine Fibonacci's Simple-fusion braid
        // actually needs (the rank-2 spike above only reaches the
        // single-r-symbol `index == 0` branch).
        let rule = ComplexScalarProbeRule;
        let tree =
            FusionTreeKey::try_from_sector_ids([1, 1, 1], 1, [false, false, false], [1], [1, 1]).unwrap();

        let braided = multiplicity_free_braid_tree(&rule, &tree, &[0, 2, 1], &[0, 1, 2]).unwrap();

        // Hand-derived from the engine formula in the index>0 branch with
        // this rule's constant F=1: coefficient(c') = R(c,d,e) * conj(R(a,d,c')).
        // Here a=b=c=d=e=x, so R(c,d,e) = e^{iβ}; c'=vacuum -> R(a,d,c')=e^{iα},
        // c'=x -> e^{iβ}.
        assert_eq!(braided.len(), 2);
        let coeff_for = |innerline: usize| {
            braided
                .iter()
                .find(|(t, _)| t.innerlines() == [SectorId::new(innerline)])
                .unwrap()
                .1
        };
        let expected_vacuum_channel =
            num_complex::Complex64::from_polar(1.0, PROBE_ANGLE_BETA - PROBE_ANGLE_ALPHA);
        let expected_x_channel = num_complex::Complex64::new(1.0, 0.0);
        assert!((coeff_for(0) - expected_vacuum_channel).norm() < 1.0e-12);
        assert!((coeff_for(1) - expected_x_channel).norm() < 1.0e-12);
    }

    // --- Stage 1: FibonacciFusionRule provider tests.

    #[test]
    fn fibonacci_fusion_channels_and_style_match_tensorkitsectors() {
        let rule = FibonacciFusionRule;
        let vacuum = SectorId::new(0);
        let tau = SectorId::new(1);

        assert_eq!(rule.fusion_style(), FusionStyleKind::Simple);
        assert_eq!(rule.braiding_style(), BraidingStyleKind::Anyonic);
        assert_eq!(rule.vacuum(), vacuum);
        assert_eq!(rule.dual(tau), tau);

        assert_eq!(rule.fusion_channels(vacuum, tau).to_vec(), vec![tau]);
        assert_eq!(rule.fusion_channels(tau, vacuum).to_vec(), vec![tau]);
        assert_eq!(
            rule.fusion_channels(tau, tau).to_vec(),
            vec![vacuum, tau]
        );
        assert_eq!(rule.nsymbol(tau, tau, vacuum), 1);
        assert_eq!(rule.nsymbol(tau, tau, tau), 1);
        // "zero if one tau and two ones" (anyons.jl:113): exactly two vacuum
        // legs and one tau is never an allowed fusion outcome.
        assert_eq!(rule.nsymbol(vacuum, vacuum, tau), 0);
        assert_eq!(rule.nsymbol(vacuum, tau, vacuum), 0);
    }

    #[test]
    fn fibonacci_f_r_dim_twist_match_tensorkitsectors_anyons_jl() {
        // Numeric oracle: every constant here is transcribed directly from
        // `~/.julia/packages/TensorKitSectors/tugbK/src/anyons.jl` (Fsymbol
        // lines 115-137, Rsymbol lines 139-146, dim line 83) plus the two
        // generic fallbacks Fibonacci does not override (`sectors.jl:646-647`
        // for twist, `sectors.jl:461-469` for the Frobenius-Schur phase).
        let rule = FibonacciFusionRule;
        let vacuum = SectorId::new(0);
        let tau = SectorId::new(1);
        let phi = (1.0 + 5.0_f64.sqrt()) / 2.0;
        let cispi = |x: f64| Complex64::from_polar(1.0, std::f64::consts::PI * x);
        let close = |a: Complex64, b: Complex64| (a - b).norm() < 1.0e-12;

        // F^{τττ}_τ 2x2 block, keyed by (left_coupled, right_coupled).
        assert!(close(
            rule.f_symbol_scalar(tau, tau, tau, tau, vacuum, vacuum),
            Complex64::new(1.0 / phi, 0.0)
        ));
        assert!(close(
            rule.f_symbol_scalar(tau, tau, tau, tau, tau, tau),
            Complex64::new(-1.0 / phi, 0.0)
        ));
        assert!(close(
            rule.f_symbol_scalar(tau, tau, tau, tau, vacuum, tau),
            Complex64::new(1.0 / phi.sqrt(), 0.0)
        ));
        assert!(close(
            rule.f_symbol_scalar(tau, tau, tau, tau, tau, vacuum),
            Complex64::new(1.0 / phi.sqrt(), 0.0)
        ));
        // Every allowed configuration touching the vacuum leg is F = 1.
        assert_eq!(
            rule.f_symbol_scalar(vacuum, tau, tau, tau, tau, tau),
            Complex64::new(1.0, 0.0)
        );
        assert_eq!(
            rule.f_symbol_scalar(tau, vacuum, tau, tau, tau, tau),
            Complex64::new(1.0, 0.0)
        );
        // Disallowed configuration (`a ⊗ f = d` gate fails: vacuum⊗vacuum
        // never fuses to tau) is F = 0.
        assert_eq!(
            rule.f_symbol_scalar(vacuum, vacuum, vacuum, tau, vacuum, vacuum),
            Complex64::new(0.0, 0.0)
        );

        assert!(close(
            rule.r_symbol_scalar(tau, tau, vacuum),
            cispi(4.0 / 5.0)
        ));
        assert!(close(
            rule.r_symbol_scalar(tau, tau, tau),
            cispi(-3.0 / 5.0)
        ));
        assert_eq!(rule.r_symbol_scalar(vacuum, tau, tau), Complex64::new(1.0, 0.0));
        assert_eq!(rule.r_symbol_scalar(tau, vacuum, tau), Complex64::new(1.0, 0.0));

        assert_eq!(rule.dim_scalar(vacuum), Complex64::new(1.0, 0.0));
        assert!(close(rule.dim_scalar(tau), Complex64::new(phi, 0.0)));
        assert!(close(
            rule.sqrt_dim_scalar(tau),
            Complex64::new(phi.sqrt(), 0.0)
        ));
        assert!(close(
            rule.inv_sqrt_dim_scalar(tau),
            Complex64::new(1.0 / phi.sqrt(), 0.0)
        ));

        assert_eq!(rule.twist_scalar(vacuum), Complex64::new(1.0, 0.0));
        assert!(close(rule.twist_scalar(tau), cispi(-4.0 / 5.0)));

        assert_eq!(
            rule.frobenius_schur_phase_scalar(vacuum),
            Complex64::new(1.0, 0.0)
        );
        assert_eq!(
            rule.frobenius_schur_phase_scalar(tau),
            Complex64::new(1.0, 0.0)
        );
    }

    struct FibonacciFAdmissibilityProbe {
        calls: std::sync::Mutex<Vec<[SectorId; 6]>>,
        complex_f_phase: bool,
    }

    impl FibonacciFAdmissibilityProbe {
        const SENTINEL: Complex64 = Complex64::new(97.0, -31.0);

        fn new() -> Self {
            Self {
                calls: std::sync::Mutex::new(Vec::new()),
                complex_f_phase: false,
            }
        }

        fn with_complex_f_phase() -> Self {
            Self {
                calls: std::sync::Mutex::new(Vec::new()),
                complex_f_phase: true,
            }
        }

        fn take_calls(&self) -> Vec<[SectorId; 6]> {
            std::mem::take(
                &mut *self
                    .calls
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner),
            )
        }
    }

    impl FusionRule for FibonacciFAdmissibilityProbe {
        fn rule_identity(&self) -> RuleIdentity {
            RuleIdentity::of_type::<Self>()
        }

        fn fusion_style(&self) -> FusionStyleKind {
            FibonacciFusionRule.fusion_style()
        }

        fn braiding_style(&self) -> BraidingStyleKind {
            FibonacciFusionRule.braiding_style()
        }

        fn vacuum(&self) -> SectorId {
            FibonacciFusionRule.vacuum()
        }

        fn supports_unitary_braid_dagger(&self) -> bool {
            FibonacciFusionRule.supports_unitary_braid_dagger()
        }

        fn dual(&self, sector: SectorId) -> SectorId {
            FibonacciFusionRule.dual(sector)
        }

        fn fusion_channels(&self, left: SectorId, right: SectorId) -> SectorVec {
            FibonacciFusionRule.fusion_channels(left, right)
        }

        fn nsymbol(&self, left: SectorId, right: SectorId, coupled: SectorId) -> usize {
            FibonacciFusionRule.nsymbol(left, right, coupled)
        }
    }

    impl MultiplicityFreeFusionRule for FibonacciFAdmissibilityProbe {}

    impl MultiplicityFreeFusionSymbols for FibonacciFAdmissibilityProbe {
        type Scalar = Complex64;

        fn scalar_one(&self) -> Self::Scalar {
            FibonacciFusionRule.scalar_one()
        }

        fn scalar_conj(&self, value: Self::Scalar) -> Self::Scalar {
            FibonacciFusionRule.scalar_conj(value)
        }

        fn f_symbol_scalar(
            &self,
            left: SectorId,
            middle: SectorId,
            right: SectorId,
            coupled: SectorId,
            left_coupled: SectorId,
            right_coupled: SectorId,
        ) -> Self::Scalar {
            let call = [
                left,
                middle,
                right,
                coupled,
                left_coupled,
                right_coupled,
            ];
            self.calls
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .push(call);
            let admissible = FibonacciFusionRule.nsymbol(left, middle, left_coupled) != 0
                && FibonacciFusionRule.nsymbol(left_coupled, right, coupled) != 0
                && FibonacciFusionRule.nsymbol(middle, right, right_coupled) != 0
                && FibonacciFusionRule.nsymbol(left, right_coupled, coupled) != 0;
            if admissible {
                let value = FibonacciFusionRule.f_symbol_scalar(
                    left,
                    middle,
                    right,
                    coupled,
                    left_coupled,
                    right_coupled,
                );
                if self.complex_f_phase {
                    value * Complex64::new(0.6, 0.8)
                } else {
                    value
                }
            } else {
                Self::SENTINEL
            }
        }

        fn r_symbol_scalar(
            &self,
            left: SectorId,
            right: SectorId,
            coupled: SectorId,
        ) -> Self::Scalar {
            FibonacciFusionRule.r_symbol_scalar(left, right, coupled)
        }
    }

    impl MultiplicityFreeRigidSymbols for FibonacciFAdmissibilityProbe {
        fn dim_scalar(&self, sector: SectorId) -> Self::Scalar {
            FibonacciFusionRule.dim_scalar(sector)
        }

        fn inv_dim_scalar(&self, sector: SectorId) -> Self::Scalar {
            FibonacciFusionRule.inv_dim_scalar(sector)
        }

        fn sqrt_dim_scalar(&self, sector: SectorId) -> Self::Scalar {
            FibonacciFusionRule.sqrt_dim_scalar(sector)
        }

        fn inv_sqrt_dim_scalar(&self, sector: SectorId) -> Self::Scalar {
            FibonacciFusionRule.inv_sqrt_dim_scalar(sector)
        }

        fn twist_scalar(&self, sector: SectorId) -> Self::Scalar {
            FibonacciFusionRule.twist_scalar(sector)
        }

        fn frobenius_schur_phase_scalar(&self, sector: SectorId) -> Self::Scalar {
            FibonacciFusionRule.frobenius_schur_phase_scalar(sector)
        }
    }

    fn fibonacci_multi_associator_counterexample() -> (FusionTreeKey, FusionTreeKey) {
        let long =
            FusionTreeKey::try_from_sector_ids([1; 4], 1, [false; 4], [1, 0], [1, 1, 1])
                .unwrap();
        let short =
            FusionTreeKey::try_from_sector_ids([1; 3], 1, [false; 3], [0], [1, 1]).unwrap();
        let tau = SectorId::new(1);
        assert!(
            collect_fusion_trees_for_coupled(
                &FibonacciFusionRule,
                &[tau; 4],
                &[false; 4],
                &[tau; 4],
                tau,
            )
            .contains(&long)
        );
        assert!(
            collect_fusion_trees_for_coupled(
                &FibonacciFusionRule,
                &[tau; 3],
                &[false; 3],
                &[tau; 3],
                tau,
            )
            .contains(&short)
        );
        (long, short)
    }

    fn assert_fibonacci_f_calls_are_admissible(calls: &[[SectorId; 6]]) {
        for &[left, middle, right, coupled, left_coupled, right_coupled] in calls {
            assert_ne!(
                FibonacciFusionRule.nsymbol(left, middle, left_coupled),
                0
            );
            assert_ne!(
                FibonacciFusionRule.nsymbol(left_coupled, right, coupled),
                0
            );
            assert_ne!(
                FibonacciFusionRule.nsymbol(middle, right, right_coupled),
                0
            );
            assert_ne!(
                FibonacciFusionRule.nsymbol(left, right_coupled, coupled),
                0
            );
        }
    }

    #[test]
    fn fibonacci_multi_associator_filters_cross_inadmissible_candidates() {
        let rule = FibonacciFAdmissibilityProbe::new();
        let (long, short) = fibonacci_multi_associator_counterexample();
        let first = long.uncoupled()[0];
        let right = long.uncoupled()[2];
        let (middle_left, middle_right) = fusion_tree_vertex_neighbors(&long, 2).unwrap();
        let (short_left, short_right) = fusion_tree_vertex_neighbors(&short, 1).unwrap();

        // What: both stored trees are valid, but their staged cross vertex is
        // absent and therefore does not name an F-symbol coefficient.
        assert_ne!(
            rule.nsymbol(middle_left, right, middle_right),
            0
        );
        assert_ne!(rule.nsymbol(short_left, right, short_right), 0);
        assert_ne!(rule.nsymbol(first, short_left, middle_left), 0);
        assert_eq!(rule.nsymbol(first, short_right, middle_right), 0);
        assert_eq!(
            rule.f_symbol_scalar(
                first,
                short_left,
                right,
                middle_right,
                middle_left,
                short_right,
            ),
            FibonacciFAdmissibilityProbe::SENTINEL
        );
        assert_eq!(rule.take_calls().len(), 1);

        assert_eq!(
            multiplicity_free_multi_associator_scalar(&rule, &long, &short).unwrap(),
            None
        );
        assert!(rule.take_calls().is_empty());
    }

    #[test]
    fn fibonacci_multi_fmove_forward_and_inverse_call_only_admissible_f() {
        let rule = FibonacciFAdmissibilityProbe::new();
        let (long, short) = fibonacci_multi_associator_counterexample();
        let actual_forward = multiplicity_free_multi_fmove_tree(&rule, &long).unwrap();
        let phi = (1.0 + 5.0_f64.sqrt()) / 2.0;
        // What: TensorKit's Stage-1 candidate intersection retains these two
        // tails in this order; Stage 2 gives the listed Fibonacci F products.
        assert_eq!(
            actual_forward
                .iter()
                .map(|(tree, _)| (tree.coupled(), tree.innerlines().to_vec()))
                .collect::<Vec<_>>(),
            vec![
                (SectorId::new(0), vec![SectorId::new(1)]),
                (SectorId::new(1), vec![SectorId::new(1)]),
            ]
        );
        assert!(
            (actual_forward[0].1 - Complex64::new(1.0 / phi, 0.0)).norm() < 1.0e-12
        );
        assert!(
            (actual_forward[1].1 - Complex64::new(1.0 / phi.sqrt(), 0.0)).norm()
                < 1.0e-12
        );
        let calls = rule.take_calls();
        assert!(!calls.is_empty());
        assert_fibonacci_f_calls_are_admissible(&calls);

        let actual_inverse = multiplicity_free_multi_fmove_inv_tree(
            &rule,
            SectorId::new(1),
            SectorId::new(1),
            &short,
            false,
        )
        .unwrap();
        // What: TensorKit's right-to-left inverse construction retains these
        // two rank-4 trees in canonical order and conjugates the same real
        // Fibonacci coefficients.
        assert_eq!(
            actual_inverse
                .iter()
                .map(|(tree, _)| (tree.coupled(), tree.innerlines().to_vec()))
                .collect::<Vec<_>>(),
            vec![
                (
                    SectorId::new(1),
                    vec![SectorId::new(0), SectorId::new(1)]
                ),
                (
                    SectorId::new(1),
                    vec![SectorId::new(1), SectorId::new(1)]
                ),
            ]
        );
        assert!(
            (actual_inverse[0].1 - Complex64::new(1.0 / phi, 0.0)).norm() < 1.0e-12
        );
        assert!(
            (actual_inverse[1].1 - Complex64::new(1.0 / phi.sqrt(), 0.0)).norm()
                < 1.0e-12
        );
        let calls = rule.take_calls();
        assert!(!calls.is_empty());
        assert_fibonacci_f_calls_are_admissible(&calls);
    }

    #[test]
    fn fibonacci_multi_fmove_low_ranks_keep_their_existing_contracts() {
        let rule = FibonacciFAdmissibilityProbe::new();
        let vacuum = SectorId::new(0);
        let tau = SectorId::new(1);
        let empty = FusionTreeKey::new([], vacuum, [], [], []);
        let rank_one = FusionTreeKey::new([tau], tau, [false], [], []);
        let rank_two =
            FusionTreeKey::new([tau, tau], vacuum, [false, false], [], [MultiplicityIndex::ONE]);

        // What: forward rank 0/1/2 and inverse rank 0/1 retain their prior
        // results and do not enter an F-symbol provider.
        assert_eq!(
            multiplicity_free_multi_fmove_tree(&rule, &empty),
            multiplicity_free_multi_fmove_tree(&FibonacciFusionRule, &empty)
        );
        assert_eq!(
            multiplicity_free_multi_fmove_tree(&rule, &rank_one),
            multiplicity_free_multi_fmove_tree(&FibonacciFusionRule, &rank_one)
        );
        assert_eq!(
            multiplicity_free_multi_fmove_tree(&rule, &rank_two),
            multiplicity_free_multi_fmove_tree(&FibonacciFusionRule, &rank_two)
        );
        assert_eq!(
            multiplicity_free_multi_fmove_inv_tree(&rule, tau, tau, &empty, false),
            multiplicity_free_multi_fmove_inv_tree(
                &FibonacciFusionRule,
                tau,
                tau,
                &empty,
                false,
            )
        );
        assert_eq!(
            multiplicity_free_multi_fmove_inv_tree(&rule, tau, vacuum, &rank_one, false),
            multiplicity_free_multi_fmove_inv_tree(
                &FibonacciFusionRule,
                tau,
                vacuum,
                &rank_one,
                false,
            )
        );
        assert!(rule.take_calls().is_empty());

        // What: inverse rank 2 still performs its one associator step, with
        // unchanged data and an admissible provider call.
        assert_eq!(
            multiplicity_free_multi_fmove_inv_tree(&rule, tau, tau, &rank_two, false),
            multiplicity_free_multi_fmove_inv_tree(
                &FibonacciFusionRule,
                tau,
                tau,
                &rank_two,
                false,
            )
        );
        let calls = rule.take_calls();
        assert!(!calls.is_empty());
        assert_fibonacci_f_calls_are_admissible(&calls);
    }

    #[test]
    fn grouped_multi_fmove_matches_legacy_order_and_reuses_stage_symbols() {
        let rule = FibonacciFAdmissibilityProbe::with_complex_f_phase();
        let tau = SectorId::new(1);
        let trees = collect_fusion_trees_for_coupled(
            &rule,
            &[tau; 6],
            &[false; 6],
            &[tau; 6],
            tau,
        );
        let mut fixture = None;
        for tree in trees {
            let grouped = multiplicity_free_multi_fmove_tree(&rule, &tree).unwrap();
            let grouped_calls = rule.take_calls();
            let legacy =
                multiplicity_free_multi_fmove_tree_legacy_oracle(&rule, &tree).unwrap();
            let legacy_calls = rule.take_calls();
            if grouped_calls.len() < legacy_calls.len() {
                fixture = Some((tree, grouped, grouped_calls, legacy, legacy_calls));
                break;
            }
        }
        let (tree, grouped, grouped_calls, legacy, legacy_calls) =
            fixture.expect("rank-six Fibonacci must repeat stage-local F arguments");

        // What: grouped forward execution preserves the legacy candidate order
        // and coefficients while evaluating one complete F sextuple per stage.
        assert_eq!(grouped, legacy);
        assert_eq!(grouped_calls.len(), 7);
        assert_eq!(legacy_calls.len(), 18);
        assert_fibonacci_f_calls_are_admissible(&grouped_calls);
        assert!(grouped
            .iter()
            .any(|(_, coefficient)| coefficient.im.abs() > 1.0e-12));

        let tail = grouped
            .first()
            .expect("rank-six Fibonacci forward move has a tail")
            .0
            .clone();
        let grouped_inverse =
            multiplicity_free_multi_fmove_inv_tree(&rule, tau, tree.coupled(), &tail, false)
                .unwrap();
        let grouped_inverse_calls = rule.take_calls();
        let legacy_inverse = multiplicity_free_multi_fmove_inv_tree_legacy_oracle(
            &rule,
            tau,
            tree.coupled(),
            &tail,
            false,
        )
        .unwrap();
        let legacy_inverse_calls = rule.take_calls();

        // What: inverse execution uses the same canonical candidates and applies
        // conjugation after the same grouped associator products.
        assert_eq!(grouped_inverse, legacy_inverse);
        assert!(grouped_inverse_calls.len() <= legacy_inverse_calls.len());
        assert_fibonacci_f_calls_are_admissible(&grouped_inverse_calls);
        assert!(grouped_inverse
            .iter()
            .any(|(_, coefficient)| coefficient.im.abs() > 1.0e-12));
    }

    struct UniqueFAdmissibilityProbe {
        calls: std::sync::Mutex<Vec<[SectorId; 6]>>,
    }

    impl UniqueFAdmissibilityProbe {
        fn new() -> Self {
            Self {
                calls: std::sync::Mutex::new(Vec::new()),
            }
        }

        fn take_calls(&self) -> Vec<[SectorId; 6]> {
            std::mem::take(
                &mut *self
                    .calls
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner),
            )
        }
    }

    impl FusionRule for UniqueFAdmissibilityProbe {
        fn rule_identity(&self) -> RuleIdentity {
            RuleIdentity::of_type::<Self>()
        }

        fn fusion_style(&self) -> FusionStyleKind {
            Z2FusionRule.fusion_style()
        }

        fn braiding_style(&self) -> BraidingStyleKind {
            Z2FusionRule.braiding_style()
        }

        fn vacuum(&self) -> SectorId {
            Z2FusionRule.vacuum()
        }

        fn supports_unitary_braid_dagger(&self) -> bool {
            Z2FusionRule.supports_unitary_braid_dagger()
        }

        fn dual(&self, sector: SectorId) -> SectorId {
            Z2FusionRule.dual(sector)
        }

        fn fusion_channels(&self, left: SectorId, right: SectorId) -> SectorVec {
            Z2FusionRule.fusion_channels(left, right)
        }

        fn nsymbol(&self, left: SectorId, right: SectorId, coupled: SectorId) -> usize {
            Z2FusionRule.nsymbol(left, right, coupled)
        }
    }

    impl MultiplicityFreeFusionRule for UniqueFAdmissibilityProbe {}

    impl MultiplicityFreeFusionSymbols for UniqueFAdmissibilityProbe {
        type Scalar = f64;

        fn scalar_one(&self) -> Self::Scalar {
            Z2FusionRule.scalar_one()
        }

        fn scalar_conj(&self, value: Self::Scalar) -> Self::Scalar {
            Z2FusionRule.scalar_conj(value)
        }

        fn f_symbol_scalar(
            &self,
            left: SectorId,
            middle: SectorId,
            right: SectorId,
            coupled: SectorId,
            left_coupled: SectorId,
            right_coupled: SectorId,
        ) -> Self::Scalar {
            self.calls
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .push([
                    left,
                    middle,
                    right,
                    coupled,
                    left_coupled,
                    right_coupled,
                ]);
            let admissible = Z2FusionRule.nsymbol(left, middle, left_coupled) != 0
                && Z2FusionRule.nsymbol(left_coupled, right, coupled) != 0
                && Z2FusionRule.nsymbol(middle, right, right_coupled) != 0
                && Z2FusionRule.nsymbol(left, right_coupled, coupled) != 0;
            if admissible {
                Z2FusionRule.f_symbol_scalar(
                    left,
                    middle,
                    right,
                    coupled,
                    left_coupled,
                    right_coupled,
                )
            } else {
                997.0
            }
        }

        fn r_symbol_scalar(
            &self,
            left: SectorId,
            right: SectorId,
            coupled: SectorId,
        ) -> Self::Scalar {
            Z2FusionRule.r_symbol_scalar(left, right, coupled)
        }
    }

    impl MultiplicityFreeRigidSymbols for UniqueFAdmissibilityProbe {
        fn dim_scalar(&self, sector: SectorId) -> Self::Scalar {
            Z2FusionRule.dim_scalar(sector)
        }

        fn inv_dim_scalar(&self, sector: SectorId) -> Self::Scalar {
            Z2FusionRule.inv_dim_scalar(sector)
        }

        fn sqrt_dim_scalar(&self, sector: SectorId) -> Self::Scalar {
            Z2FusionRule.sqrt_dim_scalar(sector)
        }

        fn inv_sqrt_dim_scalar(&self, sector: SectorId) -> Self::Scalar {
            Z2FusionRule.inv_sqrt_dim_scalar(sector)
        }

        fn twist_scalar(&self, sector: SectorId) -> Self::Scalar {
            Z2FusionRule.twist_scalar(sector)
        }

        fn frobenius_schur_phase_scalar(&self, sector: SectorId) -> Self::Scalar {
            Z2FusionRule.frobenius_schur_phase_scalar(sector)
        }
    }

    #[test]
    fn unique_multi_fmove_callers_preserve_admissible_z2_results() {
        let rule = UniqueFAdmissibilityProbe::new();
        let odd = z2_odd();
        let even = z2_even();
        let long = FusionTreeKey::new(
            [odd; 4],
            even,
            [false; 4],
            [even, odd],
            [MultiplicityIndex::ONE; 3],
        );
        let short = FusionTreeKey::new(
            [odd; 3],
            odd,
            [false; 3],
            [even],
            [MultiplicityIndex::ONE; 2],
        );

        // What: the Unique-fusion wrappers that share the associator boundary
        // retain exact output trees, coefficients, and conjugation direction.
        assert_eq!(
            unique_rigid_multi_fmove_tree(&rule, &long),
            unique_rigid_multi_fmove_tree(&Z2FusionRule, &long)
        );
        let calls = rule.take_calls();
        assert!(!calls.is_empty());
        for &[left, middle, right, coupled, left_coupled, right_coupled] in &calls {
            assert_ne!(Z2FusionRule.nsymbol(left, middle, left_coupled), 0);
            assert_ne!(Z2FusionRule.nsymbol(left_coupled, right, coupled), 0);
            assert_ne!(Z2FusionRule.nsymbol(middle, right, right_coupled), 0);
            assert_ne!(Z2FusionRule.nsymbol(left, right_coupled, coupled), 0);
        }

        assert_eq!(
            unique_rigid_multi_fmove_inv_tree(&rule, odd, even, &short, false),
            unique_rigid_multi_fmove_inv_tree(&Z2FusionRule, odd, even, &short, false)
        );
        let calls = rule.take_calls();
        assert!(!calls.is_empty());
        for &[left, middle, right, coupled, left_coupled, right_coupled] in &calls {
            assert_ne!(Z2FusionRule.nsymbol(left, middle, left_coupled), 0);
            assert_ne!(Z2FusionRule.nsymbol(left_coupled, right, coupled), 0);
            assert_ne!(Z2FusionRule.nsymbol(middle, right, right_coupled), 0);
            assert_ne!(Z2FusionRule.nsymbol(left, right_coupled, coupled), 0);
        }
    }

    #[test]
    fn fibonacci_braid_then_inverse_braid_is_identity() {
        // Self-consistency (a): braiding a crossing and then undoing it
        // (reflected levels select the inverse-artin branch) must return the
        // exact original tree with total coefficient 1 — this only holds
        // because R^{ττ}_* is a genuine unit-modulus phase.
        let rule = FibonacciFusionRule;
        for coupled in [0usize, 1usize] {
            let tree =
                FusionTreeKey::try_from_sector_ids([1, 1], coupled, [false, false], [], [1]).unwrap();

            let forward = multiplicity_free_braid_tree(&rule, &tree, &[1, 0], &[0, 1]).unwrap();
            assert_eq!(forward.len(), 1);
            let backward =
                multiplicity_free_braid_tree(&rule, &forward[0].0, &[1, 0], &[1, 0]).unwrap();
            assert_eq!(backward.len(), 1);

            assert_eq!(backward[0].0.uncoupled(), tree.uncoupled());
            assert_eq!(backward[0].0.coupled(), tree.coupled());
            let total = forward[0].1 * backward[0].1;
            assert!((total - Complex64::new(1.0, 0.0)).norm() < 1.0e-12);
        }

        // Same check through the rank > 2 loop branch, where the round trip
        // additionally exercises the F-symbol: this only returns to the
        // identity because TensorKitSectors' F^{τττ}_τ block is a genuine
        // (real, orthogonal) unitary matrix.
        let tree =
            FusionTreeKey::try_from_sector_ids([1, 1, 1], 1, [false, false, false], [1], [1, 1]).unwrap();
        let forward = multiplicity_free_braid_tree(&rule, &tree, &[0, 2, 1], &[0, 1, 2]).unwrap();
        assert_eq!(forward.len(), 2);
        let mut total = Complex64::new(0.0, 0.0);
        for (intermediate, coeff) in &forward {
            let backward =
                multiplicity_free_braid_tree(&rule, intermediate, &[0, 2, 1], &[0, 2, 1]).unwrap();
            for (roundtrip, back_coeff) in &backward {
                if roundtrip.innerlines() == tree.innerlines() {
                    total += *coeff * *back_coeff;
                }
            }
        }
        assert!((total - Complex64::new(1.0, 0.0)).norm() < 1.0e-12);
    }

    #[test]
    fn fibonacci_braid_tree_end_to_end_matches_hand_derived_coefficients() {
        // End-to-end: Simple fusion + Anyonic braiding + complex Scalar
        // through `multiplicity_free_braid_tree` on a rank-3 tree, with
        // coefficients hand-derived from the engine's own
        // R(c,d,e) * conj(F(d,a,b,e,c',c) * R(a,d,c')) formula
        // (`multiplicity_free_artin_braid_at_with_inverse`, index > 0
        // branch) substituting TensorKitSectors' F/R values directly.
        let rule = FibonacciFusionRule;
        let tree =
            FusionTreeKey::try_from_sector_ids([1, 1, 1], 1, [false, false, false], [1], [1, 1]).unwrap();

        let braided = multiplicity_free_braid_tree(&rule, &tree, &[0, 2, 1], &[0, 1, 2]).unwrap();
        assert_eq!(braided.len(), 2);

        let phi = (1.0 + 5.0_f64.sqrt()) / 2.0;
        let cispi = |x: f64| Complex64::from_polar(1.0, std::f64::consts::PI * x);
        let coeff_for = |innerline: usize| {
            braided
                .iter()
                .find(|(t, _)| t.innerlines() == [SectorId::new(innerline)])
                .unwrap()
                .1
        };

        let expected_vacuum_channel = Complex64::new(1.0 / phi.sqrt(), 0.0) * cispi(3.0 / 5.0);
        let expected_tau_channel = Complex64::new(-1.0 / phi, 0.0);
        assert!((coeff_for(0) - expected_vacuum_channel).norm() < 1.0e-12);
        assert!((coeff_for(1) - expected_tau_channel).norm() < 1.0e-12);
    }

    #[test]
    fn fibonacci_elementary_artin_rows_match_tensorkit_coefficients() {
        // What: the private elementary Artin operation preserves TensorKit's
        // destination order and the independently substituted F/R coefficients
        // for both crossing orientations.
        let rule = FibonacciFusionRule;
        let tree =
            FusionTreeKey::try_from_sector_ids([1, 1, 1], 1, [false, false, false], [1], [1, 1]).unwrap();
        let phi = (1.0 + 5.0_f64.sqrt()) / 2.0;
        let cispi = |x: f64| Complex64::from_polar(1.0, std::f64::consts::PI * x);

        let forward =
            multiplicity_free_artin_braid_at_with_inverse(&rule, &tree, 1, false).unwrap();
        let inverse =
            multiplicity_free_artin_braid_at_with_inverse(&rule, &tree, 1, true).unwrap();

        assert_eq!(
            forward
                .iter()
                .map(|(key, _)| key.innerlines()[0].id())
                .collect::<Vec<_>>(),
            vec![0, 1]
        );
        assert_eq!(
            inverse
                .iter()
                .map(|(key, _)| key.innerlines()[0].id())
                .collect::<Vec<_>>(),
            vec![0, 1]
        );
        let forward_expected = [
            Complex64::new(1.0 / phi.sqrt(), 0.0) * cispi(3.0 / 5.0),
            Complex64::new(-1.0 / phi, 0.0),
        ];
        let inverse_expected = [
            Complex64::new(1.0 / phi.sqrt(), 0.0) * cispi(-3.0 / 5.0),
            Complex64::new(-1.0 / phi, 0.0),
        ];
        for ((_, actual), expected) in forward.iter().zip(forward_expected) {
            assert!((*actual - expected).norm() < 1.0e-12);
        }
        for ((_, actual), expected) in inverse.iter().zip(inverse_expected) {
            assert!((*actual - expected).norm() < 1.0e-12);
        }
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
    fn identity_braid_tree_pair_skips_symbols_and_repartition() {
        // What: an exact same-split braid is source => one for both Unique and
        // multiplicity-free entry points, without consulting F/R/bend data.
        let source = FusionTreePairKey::try_pair_from_sector_ids(
            [1],
            [1], 1,
            [false],
            [false],
            [],
            [],
            [],
            [],
        ).unwrap();

        let unique = unique_braid_tree_pair(
            &IdentitySymbolPanicRule,
            &source,
            &[0],
            &[1],
            &[19],
            &[3],
        )
        .unwrap();
        assert_eq!(unique, (source.clone(), 1.0));

        let multiplicity_free = multiplicity_free_braid_tree_pair(
            &IdentitySymbolPanicRule,
            &source,
            &[0],
            &[1],
            &[19],
            &[3],
        )
        .unwrap();
        assert_eq!(multiplicity_free, vec![(source.clone(), 1.0)]);

        let block = multiplicity_free_braid_tree_pair_block(
            &IdentitySymbolPanicRule,
            &[source.clone()],
            &[0],
            &[1],
            &[19],
            &[3],
        )
        .unwrap();
        assert_eq!(block, vec![vec![(source, 1.0)]]);
    }

    #[test]
    fn braid_tree_block_matches_per_source_su2_rows() {
        use std::collections::BTreeMap;

        let rule = SU2FusionRule;
        let sources = [
            FusionTreeKey::try_from_sector_ids(
                [1, 1, 1], 1,
                [false, false, false],
                [0],
                [1, 1],
            ).unwrap(),
            FusionTreeKey::try_from_sector_ids(
                [1, 1, 1], 1,
                [false, false, false],
                [2],
                [1, 1],
            ).unwrap(),
        ];

        let block =
            multiplicity_free_braid_tree_block(&rule, &sources, &[0, 2, 1], &[0, 1, 2]).unwrap();
        assert_eq!(block.len(), sources.len());
        for (source, block_rows) in sources.iter().zip(&block) {
            let per_source =
                multiplicity_free_braid_tree(&rule, source, &[0, 2, 1], &[0, 1, 2]).unwrap();
            let collect = |rows: &[(FusionTreeKey, f64)]| {
                let mut coefficients = BTreeMap::<FusionTreeKey, f64>::new();
                for (key, coefficient) in rows {
                    *coefficients.entry(key.clone()).or_default() += coefficient;
                }
                coefficients
            };
            let expected = collect(&per_source);
            let actual = collect(block_rows);

            // What: the whole-block walk preserves every destination tree and
            // coefficient produced by independent SU(2) source transforms.
            assert_eq!(
                expected.keys().collect::<Vec<_>>(),
                actual.keys().collect::<Vec<_>>()
            );
            for (key, expected_coefficient) in expected {
                let actual_coefficient = actual[&key];
                assert!(
                    (expected_coefficient - actual_coefficient).abs()
                        <= 1.0e-12 * (1.0 + expected_coefficient.abs())
                );
            }
        }
    }

    #[test]
    fn tree_block_admits_every_source_before_group_check_and_symbols() {
        let base =
            FusionTreeKey::try_from_sector_ids([1, 2], 3, [false, false], [], [1]).unwrap();
        let valid_mixed = [
            FusionTreeKey::try_from_sector_ids([1, 4], 5, [false, false], [], [1]).unwrap(),
            FusionTreeKey::try_from_sector_ids([1, 2], 3, [false, true], [], [1]).unwrap(),
        ];
        let expected = CoreError::MalformedFusionTree {
            message: "fusion-tree keys must share one group",
        };

        for other in valid_mixed {
            let sources = [base.clone(), other];
            // What: admitted mixed groups fail before the panic-on-symbol
            // fixture can evaluate F or R data.
            assert_eq!(
                multiplicity_free_braid_tree_block(
                    &IdentitySymbolPanicRule,
                    &sources,
                    &[0, 1],
                    &[0, 1],
                )
                .unwrap_err(),
                expected
            );
            assert_eq!(
                multiplicity_free_permute_tree_block(
                    &IdentitySymbolPanicRule,
                    &sources,
                    &[1, 0],
                )
                .unwrap_err(),
                expected
            );
        }

        let invalid_vertex = FusionTreeKey::new(
            base.uncoupled().iter().copied(),
            base.coupled(),
            base.is_dual().iter().copied(),
            base.innerlines().iter().copied(),
            [MultiplicityIndex::new(2).unwrap()],
        );
        let invalid_sources = [base, invalid_vertex];
        let expected_vertex = CoreError::MalformedFusionTree {
            message: "fusion tree vertex label exceeds its fusion multiplicity",
        };
        // What: provider-owned multiplicity admission checks every source
        // before a block shortcut can discard explicit vertex identity.
        assert_eq!(
            multiplicity_free_braid_tree_block(
                &IdentitySymbolPanicRule,
                &invalid_sources,
                &[0, 1],
                &[0, 1],
            )
            .unwrap_err(),
            expected_vertex
        );
    }

    #[test]
    fn tree_block_empty_and_identity_contracts_preserve_source_order() {
        let empty: &[FusionTreeKey] = &[];
        assert_eq!(
            multiplicity_free_braid_tree_block(
                &IdentitySymbolPanicRule,
                empty,
                &[],
                &[],
            )
            .unwrap(),
            Vec::<Vec<(FusionTreeKey, f64)>>::new()
        );
        assert_eq!(
            multiplicity_free_permute_tree_block(&IdentitySymbolPanicRule, empty, &[]).unwrap(),
            Vec::<Vec<(FusionTreeKey, f64)>>::new()
        );

        let half = su2(1);
        let sources = [
            FusionTreeKey::try_from_sector_ids(
                [half.id(), half.id()], su2(0).id(),
                [false; 2],
                [],
                [1],
            ).unwrap(),
            FusionTreeKey::try_from_sector_ids(
                [half.id(), half.id()], su2(2).id(),
                [false; 2],
                [],
                [1],
            ).unwrap(),
        ];
        let expected = sources
            .iter()
            .cloned()
            .map(|source| vec![(source, 1.0)])
            .collect::<Vec<_>>();
        // What: distinct coupled labels remain one external-sector group and
        // the symbol-free identity path returns exact rows in source order.
        assert_eq!(
            multiplicity_free_braid_tree_block(
                &SU2FusionRule,
                &sources,
                &[0, 1],
                &[13, 5],
            )
            .unwrap(),
            expected
        );
        assert_eq!(
            multiplicity_free_permute_tree_block(
                &SU2FusionRule,
                &sources,
                &[0, 1],
            )
            .unwrap(),
            expected
        );
    }

    fn tree_pair_group_fixture(
        codomain: &[usize],
        domain: &[usize],
        coupled: usize,
        codomain_dual: &[bool],
        domain_dual: &[bool],
    ) -> FusionTreePairKey {
        let codomain_vertices = vec![1; codomain.len().saturating_sub(1)];
        let domain_vertices = vec![1; domain.len().saturating_sub(1)];
        FusionTreePairKey::try_pair_from_sector_ids(
            codomain.iter().copied(),
            domain.iter().copied(), coupled,
            codomain_dual.iter().copied(),
            domain_dual.iter().copied(),
            [],
            [],
            codomain_vertices,
            domain_vertices,
        ).unwrap()
    }

    fn assert_mixed_tree_pair_block_group_is_rejected<R>(
        rule: &R,
        keys: &[FusionTreePairKey],
        expected: CoreError,
    )
    where
        R: MultiplicityFreeRigidSymbols,
        R::Scalar: Clone + Add<Output = R::Scalar> + Mul<Output = R::Scalar> + std::fmt::Debug,
    {
        // What: identity and non-identity braid entry paths reject before
        // returning a shared coefficient matrix or evaluating rigid symbols.
        assert_eq!(
            multiplicity_free_braid_tree_pair_block(
                rule,
                keys,
                &[0, 1],
                &[2],
                &[0, 1],
                &[2],
            )
            .unwrap_err(),
            expected
        );
        assert_eq!(
            multiplicity_free_braid_tree_pair_block(
                rule,
                keys,
                &[1, 0],
                &[2],
                &[0, 1],
                &[2],
            )
            .unwrap_err(),
            expected
        );

        // What: symmetric permutation validates the same block invariant on
        // both its identity shortcut and general braid delegation.
        assert_eq!(
            multiplicity_free_permute_tree_pair_block(rule, keys, &[0, 1], &[2]).unwrap_err(),
            expected
        );
        assert_eq!(
            multiplicity_free_permute_tree_pair_block(rule, keys, &[1, 0], &[2]).unwrap_err(),
            expected
        );

        // What: planar transpose validates before either its identity return or
        // cyclic repartition path can consume symbols.
        assert_eq!(
            multiplicity_free_transpose_tree_pair_block(rule, keys, &[0, 1], &[2]).unwrap_err(),
            expected
        );
        assert_eq!(
            multiplicity_free_transpose_tree_pair_block(rule, keys, &[1, 2], &[0]).unwrap_err(),
            expected
        );
    }

    #[test]
    fn tree_pair_block_apis_reject_mixed_fusion_tree_groups_before_symbols() {
        let base = tree_pair_group_fixture(&[1, 2], &[3], 3, &[false, false], &[false]);
        let mixed = [
            tree_pair_group_fixture(&[1, 4], &[5], 5, &[false, false], &[false]),
            tree_pair_group_fixture(&[1, 2], &[3], 3, &[false, true], &[false]),
            tree_pair_group_fixture(&[1, 2], &[3], 3, &[false, false], &[true]),
        ];

        for other in mixed {
            let keys = [base.clone(), other];
            let snapshot = keys.clone();
            assert_mixed_tree_pair_block_group_is_rejected(
                &IdentitySymbolPanicRule,
                &keys,
                CoreError::MalformedFusionTree {
                    message: TREE_PAIR_BLOCK_GROUP_ERROR,
                },
            );
            // What: validation errors do not alter caller-owned source keys.
            assert_eq!(keys, snapshot);
        }
    }

    #[test]
    fn tree_pair_block_apis_reject_mixed_product_sector_components() {
        type FpU1Rule = ProductFusionRule<FermionParityFusionRule, U1FusionRule>;
        let rule = FpU1Rule::default();
        let sector_a = rule.encode_sector(z2_even(), u1(2)).id();
        let sector_b = rule.encode_sector(z2_even(), u1(3)).id();
        let coupled_a = rule.encode_sector(z2_even(), u1(4)).id();
        let coupled_b = rule.encode_sector(z2_even(), u1(5)).id();
        let keys = [
            tree_pair_group_fixture(
                &[sector_a, sector_a],
                &[coupled_a],
                coupled_a,
                &[false; 2],
                &[false],
            ),
            tree_pair_group_fixture(
                &[sector_a, sector_b],
                &[coupled_b],
                coupled_b,
                &[false; 2],
                &[false],
            ),
        ];

        // What: a changed component of an interned product-sector label is a
        // different shared basis group, even when ranks and duality match.
        assert_mixed_tree_pair_block_group_is_rejected(
            &rule,
            &keys,
            CoreError::MalformedFusionTree {
                message: TREE_PAIR_BLOCK_GROUP_ERROR,
            },
        );
    }

    #[test]
    fn tree_pair_block_empty_and_valid_group_identity_policies_are_stable() {
        let rule = IdentitySymbolPanicRule;
        let empty: &[FusionTreePairKey] = &[];
        // What: an empty block remains a valid empty coefficient transform.
        assert_eq!(
            multiplicity_free_braid_tree_pair_block(&rule, empty, &[], &[], &[], &[]).unwrap(),
            Vec::<Vec<(FusionTreePairKey, f64)>>::new()
        );
        assert_eq!(
            multiplicity_free_permute_tree_pair_block(&rule, empty, &[], &[]).unwrap(),
            Vec::<Vec<(FusionTreePairKey, f64)>>::new()
        );
        assert_eq!(
            multiplicity_free_transpose_tree_pair_block(&rule, empty, &[], &[]).unwrap(),
            Vec::<Vec<(FusionTreePairKey, f64)>>::new()
        );

        let half = su2(1).id();
        let keys = vec![
            tree_pair_group_fixture(&[half, half], &[half, half], su2(0).id(), &[false; 2], &[false; 2]),
            tree_pair_group_fixture(&[half, half], &[half, half], su2(2).id(), &[false; 2], &[false; 2]),
        ];
        let expected = keys
            .iter()
            .cloned()
            .map(|key| vec![(key, 1.0)])
            .collect::<Vec<_>>();
        // What: distinct coupled labels remain valid basis states in one
        // external-sector group, preserving source order, destinations, and
        // coefficients on the symbol-free identity path.
        assert_eq!(
            multiplicity_free_braid_tree_pair_block(
                &SU2FusionRule,
                &keys,
                &[0, 1],
                &[2, 3],
                &[0, 1],
                &[2, 3],
            )
            .unwrap(),
            expected
        );
    }

    #[test]
    fn indexed_adjoint_tree_pair_block_reuses_parent_group_order() {
        let half = su2(1).id();
        let keys = [
            tree_pair_group_fixture(
                &[half, half],
                &[half, half],
                su2(0).id(),
                &[false; 2],
                &[false; 2],
            ),
            tree_pair_group_fixture(
                &[half, half],
                &[half, half],
                su2(0).id(),
                &[true, false],
                &[false; 2],
            ),
            tree_pair_group_fixture(
                &[half, half],
                &[half, half],
                su2(2).id(),
                &[false; 2],
                &[false; 2],
            ),
            tree_pair_group_fixture(
                &[half, half],
                &[half, half],
                su2(2).id(),
                &[true, false],
                &[false; 2],
            ),
        ];
        let structure = packed_fixture_structure(
            4,
            keys.iter().cloned().map(|key| (key, vec![1; 4])),
        )
        .unwrap();
        let eager_adjoint_keys = keys
            .iter()
            .map(|key| {
                FusionTreePairKey::pair(
                    key.domain_tree().clone(),
                    key.codomain_tree().clone(),
                )
            })
            .collect::<Vec<_>>();
        let eager_adjoint = packed_fixture_structure(
            4,
            eager_adjoint_keys
                .iter()
                .cloned()
                .map(|key| (key, vec![1; 4])),
        )
        .unwrap();

        // What: side swapping is a bijection on group labels, so an adjoint
        // projection preserves the parent group partition and storage order.
        assert_eq!(
            structure
                .fusion_tree_group_slice()
                .iter()
                .map(FusionTreeBlockGroup::block_indices)
                .collect::<Vec<_>>(),
            eager_adjoint
                .fusion_tree_group_slice()
                .iter()
                .map(FusionTreeBlockGroup::block_indices)
                .collect::<Vec<_>>(),
        );

        for group in structure.fusion_tree_group_slice() {
            let rows = multiplicity_free_permute_tree_pair_block_indexed(
                &SU2FusionRule,
                &structure,
                group.block_indices(),
                FusionTreePairOrientation::Adjoint,
                &[0, 1],
                &[2, 3],
            )
            .unwrap();
            // What: indexed adjoint identity emits logical swapped keys once,
            // in parent source order, without changing compact transform rows.
            assert_eq!(
                rows,
                group
                    .block_indices()
                    .iter()
                    .map(|&index| vec![(eager_adjoint_keys[index].clone(), 1.0)])
                    .collect::<Vec<_>>()
            );

            let prepared = PreparedTreePairOperation::prepare_permute(
                &SU2FusionRule,
                2,
                2,
                &[0, 1],
                &[2, 3],
            )
            .unwrap();
            let ordered = multiplicity_free_braid_tree_pair_block_ordered_indexed(
                &SU2FusionRule,
                &structure,
                group.block_indices(),
                FusionTreePairOrientation::Adjoint,
                &prepared,
            )
            .unwrap();
            // What: the ordered indexed kernel exposes the same logical
            // adjoint columns while retaining parent block-index order.
            assert_eq!(ordered.source_count(), group.block_indices().len());
            assert_eq!(
                ordered.destinations(),
                group
                    .block_indices()
                    .iter()
                    .map(|&index| eager_adjoint_keys[index].clone())
                    .collect::<Vec<_>>()
            );

            let prepared = PreparedTreePairOperation::prepare_transpose(
                2,
                2,
                &[0, 1],
                &[2, 3],
            )
            .unwrap();
            let transposed = multiplicity_free_transpose_tree_pair_block_ordered_indexed(
                &SU2FusionRule,
                &structure,
                group.block_indices(),
                FusionTreePairOrientation::Adjoint,
                &prepared,
            )
            .unwrap();
            // What: transpose uses the same oriented parent projection and
            // preserves the canonical logical source-column count.
            assert_eq!(transposed.source_count(), group.block_indices().len());
            assert_eq!(transposed.destinations(), ordered.destinations());
        }
    }

    #[test]
    fn empty_block_permute_preserves_braiding_style_error_precedence() {
        // What: an empty source block does not bypass the public permutation
        // API's symmetric-braiding capability check.
        assert_eq!(
            multiplicity_free_permute_tree_pair_block(
                &FibonacciFusionRule,
                &[],
                &[],
                &[],
            )
            .unwrap_err(),
            CoreError::UnsupportedBraidingStyle {
                expected: "symmetric braiding",
                actual: BraidingStyleKind::Anyonic,
            }
        );
    }

    #[test]
    fn tree_pair_block_apis_reject_invalid_later_vertex_before_symbols() {
        let codomain =
            FusionTreeKey::try_from_sector_ids([1, 2], 3, [false, false], [], [1]).unwrap();
        let domain = FusionTreeKey::try_from_sector_ids([3], 3, [false], [], []).unwrap();
        let base = FusionTreePairKey::pair(codomain.clone(), domain.clone());
        let invalid_codomain = FusionTreeKey::new(
            codomain.uncoupled().iter().copied(),
            codomain.coupled(),
            codomain.is_dual().iter().copied(),
            codomain.innerlines().iter().copied(),
            [MultiplicityIndex::new(2).unwrap()],
        );
        let invalid = FusionTreePairKey::pair(invalid_codomain, domain);

        // What: every source pair is categorically admitted before block
        // identity shortcuts or symbol evaluation.
        assert_mixed_tree_pair_block_group_is_rejected(
            &IdentitySymbolPanicRule,
            &[base, invalid],
            CoreError::MalformedFusionTree {
                message: "fusion tree vertex label exceeds its fusion multiplicity",
            },
        );
    }

    #[test]
    fn split_only_tree_pair_braid_uses_only_the_required_bend() {
        // What: moving the split from 1|2 to 2|1 with unchanged linearized
        // external-leg order evaluates one bend and no braid symbols, for both
        // the single-source and block entry points.
        let source = FusionTreePairKey::try_pair_from_sector_ids(
            [1],
            [0, 1], 1,
            [false],
            [false, true],
            [],
            [],
            [],
            [1],
        ).unwrap();

        let rule = SplitOnlyCountingRule::default();
        let single = multiplicity_free_braid_tree_pair(
            &rule,
            &source,
            &[0, 2],
            &[1],
            &[0],
            &[1, 2],
        )
        .unwrap();
        assert_eq!(rule.f_calls.load(std::sync::atomic::Ordering::Relaxed), 1);
        assert_eq!(rule.r_calls.load(std::sync::atomic::Ordering::Relaxed), 0);

        rule.f_calls
            .store(0, std::sync::atomic::Ordering::Relaxed);
        let block = multiplicity_free_braid_tree_pair_block(
            &rule,
            std::slice::from_ref(&source),
            &[0, 2],
            &[1],
            &[0],
            &[1, 2],
        )
        .unwrap();
        assert_eq!(rule.f_calls.load(std::sync::atomic::Ordering::Relaxed), 1);
        assert_eq!(rule.r_calls.load(std::sync::atomic::Ordering::Relaxed), 0);
        assert_eq!(block, vec![single]);
    }

    #[test]
    fn split_only_su2_braid_matches_legacy_composition_in_both_directions() {
        // What: SU(2) 2|2 -> 3|1 and 2|2 -> 1|3 retain the exact tree keys,
        // dual flags, and bend coefficients of the old all-codomain route.
        let source = FusionTreePairKey::try_pair_from_sector_ids(
            [1, 2],
            [2, 1], 1,
            [false, true],
            [true, false],
            [],
            [],
            [1],
            [1],
        ).unwrap();

        for (codomain_axes, domain_axes, target_rank) in
            [(&[0, 1, 3][..], &[2][..], 3), (&[0][..], &[2, 3, 1][..], 1)]
        {
            let actual = multiplicity_free_braid_tree_pair(
                &SU2FusionRule,
                &source,
                codomain_axes,
                domain_axes,
                &[0, 1],
                &[2, 3],
            )
            .unwrap();
            let expected =
                legacy_split_only_tree_pair_route(&SU2FusionRule, &source, target_rank).unwrap();
            assert_eq!(actual.len(), expected.len());
            for ((actual_key, actual_coefficient), (expected_key, expected_coefficient)) in
                actual.iter().zip(&expected)
            {
                assert_eq!(actual_key, expected_key);
                assert!((actual_coefficient - expected_coefficient).abs() < 1.0e-12);
            }
        }
    }

    #[test]
    fn split_only_tree_pair_braid_handles_empty_split_boundaries() {
        // What: the two extreme repartitions 0|2 -> 2|0 and 2|0 -> 0|2
        // preserve the same keys, dual flags, and coefficients as the direct
        // primitive, including an empty codomain or domain tree.
        let all_domain = FusionTreePairKey::try_pair_from_sector_ids(
            [],
            [1, 1], 0,
            [],
            [false, true],
            [],
            [],
            [],
            [1],
        ).unwrap();
        let to_codomain = multiplicity_free_braid_tree_pair(
            &SU2FusionRule,
            &all_domain,
            &[1, 0],
            &[],
            &[],
            &[0, 1],
        )
        .unwrap();
        let expected_codomain =
            multiplicity_free_repartition_tree_pair(&SU2FusionRule, &all_domain, 2).unwrap();
        assert_eq!(to_codomain, expected_codomain);

        let all_codomain = &to_codomain[0].0;
        let to_domain = multiplicity_free_braid_tree_pair(
            &SU2FusionRule,
            all_codomain,
            &[],
            &[1, 0],
            &[0, 1],
            &[],
        )
        .unwrap();
        let expected_domain =
            multiplicity_free_repartition_tree_pair(&SU2FusionRule, all_codomain, 0).unwrap();
        assert_eq!(to_domain, expected_domain);
    }

    #[test]
    fn split_only_nested_product_braid_matches_legacy_composition() {
        // What: a non-Abelian fZ2 x U(1) x SU(2) tree preserves the product
        // bend sign and duality bookkeeping of the old all-codomain route in
        // both split directions.
        type FpU1Rule = ProductFusionRule<FermionParityFusionRule, U1FusionRule>;
        type ProductRule = ProductFusionRule<FpU1Rule, SU2FusionRule>;
        let left_rule = FpU1Rule::default();
        let rule = ProductRule::default();
        let coupled = rule.encode_sector(
            left_rule.encode_sector(z2_even(), u1(0)),
            su2(1),
        );
        let domain_left = rule.encode_sector(
            left_rule.encode_sector(z2_odd(), u1(1)),
            su2(1),
        );
        let domain_right = rule.encode_sector(
            left_rule.encode_sector(z2_odd(), u1(-1)),
            su2(2),
        );
        let source = FusionTreePairKey::pair(
            FusionTreeKey::try_new_for_rule(
                &rule,
                [coupled], coupled,
                [false],
                [],
                [],
            )
            .unwrap(),
            FusionTreeKey::try_new_for_rule(
                &rule,
                [domain_left, domain_right], coupled,
                [false, true],
                [],
                [MultiplicityIndex::ONE],
            )
            .unwrap(),
        );

        let forward = multiplicity_free_braid_tree_pair(
            &rule,
            &source,
            &[0, 2],
            &[1],
            &[0],
            &[1, 2],
        )
        .unwrap();
        let expected = legacy_split_only_tree_pair_route(&rule, &source, 2).unwrap();
        assert_eq!(forward.len(), expected.len());
        assert!(forward[0].1 < 0.0);
        for ((actual_key, actual_coefficient), (expected_key, expected_coefficient)) in
            forward.iter().zip(&expected)
        {
            assert_eq!(actual_key, expected_key);
            assert!((actual_coefficient - expected_coefficient).abs() < 1.0e-12);
        }

        let reverse_source = &forward[0].0;
        let reverse = multiplicity_free_braid_tree_pair(
            &rule,
            reverse_source,
            &[0],
            &[2, 1],
            &[0, 1],
            &[2],
        )
        .unwrap();
        let reverse_expected =
            legacy_split_only_tree_pair_route(&rule, reverse_source, 1).unwrap();
        assert_eq!(reverse.len(), reverse_expected.len());
        assert!(reverse[0].1 < 0.0);
        for ((actual_key, actual_coefficient), (expected_key, expected_coefficient)) in
            reverse.iter().zip(&reverse_expected)
        {
            assert_eq!(actual_key, expected_key);
            assert!((actual_coefficient - expected_coefficient).abs() < 1.0e-12);
        }
    }

    #[test]
    fn nonidentity_tree_pair_braid_does_not_enter_split_only_path() {
        // What: changing the split does not suppress a real external-leg
        // permutation, and malformed axis maps still fail validation.
        let source = FusionTreePairKey::try_pair_from_sector_ids(
            [1],
            [0, 1], 1,
            [false],
            [false, true],
            [],
            [],
            [],
            [1],
        ).unwrap();

        let rule = SplitOnlyCountingRule::default();
        multiplicity_free_braid_tree_pair(
            &rule,
            &source,
            &[2, 0],
            &[1],
            &[0],
            &[1, 2],
        )
        .unwrap();
        assert!(rule.r_calls.load(std::sync::atomic::Ordering::Relaxed) > 0);

        for (codomain_axes, domain_axes) in
            [(&[0, 0][..], &[1][..]), (&[0, 3][..], &[1][..]), (&[0][..], &[1][..])]
        {
            assert!(multiplicity_free_braid_tree_pair(
                &rule,
                &source,
                codomain_axes,
                domain_axes,
                &[0],
                &[1, 2],
            )
            .is_err());
        }
    }

    #[test]
    fn identity_braid_tree_pair_validates_levels_before_symbol_free_return() {
        // What: malformed levels remain errors even when the axis map is the
        // exact current split.
        let source = FusionTreePairKey::try_pair_from_sector_ids(
            [1],
            [1], 1,
            [false],
            [false],
            [],
            [],
            [],
            [],
        ).unwrap();

        assert!(unique_braid_tree_pair(
            &IdentitySymbolPanicRule,
            &source,
            &[0],
            &[1],
            &[],
            &[3],
        )
        .is_err());
        assert!(multiplicity_free_braid_tree_pair(
            &IdentitySymbolPanicRule,
            &source,
            &[0],
            &[1],
            &[19],
            &[],
        )
        .is_err());
        assert!(unique_braid_tree_pair(
            &IdentitySymbolPanicRule,
            &source,
            &[0],
            &[0],
            &[19],
            &[3],
        )
        .is_err());
    }

    #[test]
    fn unique_identity_tree_operations_reject_simple_fusion_rules() {
        // What: identity axes do not let a Simple rule enter APIs whose
        // contract requires Unique fusion.
        let tree = FusionTreeKey::try_from_sector_ids([1], 1, [false], [], []).unwrap();
        let expected = CoreError::UnsupportedFusionStyle {
            expected: FusionStyleKind::Unique,
            actual: FusionStyleKind::Simple,
        };

        assert_eq!(
            unique_braid_tree(&SU2FusionRule, &tree, &[0], &[7]).unwrap_err(),
            expected
        );
        assert_eq!(
            unique_permute_tree(&SU2FusionRule, &tree, &[0]).unwrap_err(),
            expected
        );
    }

    #[test]
    fn same_split_transpose_of_dual_tree_pair_skips_bend_symbols() {
        // What: a real codomain/domain tree pair at its current 2|1 split is
        // source => one without consulting bend/fold data.
        let source = FusionTreePairKey::try_pair_from_sector_ids(
            [1, 0],
            [1], 1,
            [false, true],
            [true],
            [],
            [],
            [1],
            [],
        ).unwrap();

        assert_eq!(
            unique_transpose_tree_pair(&IdentitySymbolPanicRule, &source, &[0, 1], &[2])
                .unwrap(),
            (source, 1.0)
        );
    }

    #[test]
    fn identity_braid_rows_are_exact_for_supported_symmetry_families_and_rank_zero() {
        // What: identity rows preserve their exact source key and unit
        // coefficient for fermionic, non-abelian, product, and scalar spaces.
        let pair = |sector: SectorId| {
            FusionTreePairKey::try_pair_from_sector_ids(
                [sector.id()],
                [sector.id()], sector.id(),
                [false],
                [false],
                [],
                [],
                [],
                [],
            ).unwrap()
        };

        let fz2_source = pair(z2_odd());
        assert_eq!(
            unique_braid_tree_pair(
                &FermionParityFusionRule,
                &fz2_source,
                &[0],
                &[1],
                &[41],
                &[2],
            )
            .unwrap(),
            (fz2_source, 1.0)
        );

        let su2_source = pair(su2(1));
        assert_eq!(
            multiplicity_free_braid_tree_pair(
                &SU2FusionRule,
                &su2_source,
                &[0],
                &[1],
                &[13],
                &[5],
            )
            .unwrap(),
            vec![(su2_source, 1.0)]
        );

        type FpU1Rule = ProductFusionRule<FermionParityFusionRule, U1FusionRule>;
        type FpU1Su2Rule = ProductFusionRule<FpU1Rule, SU2FusionRule>;
        let left_rule = FpU1Rule::default();
        let product_rule = FpU1Su2Rule::default();
        let product_sector = product_rule.encode_sector(
            left_rule.encode_sector(z2_odd(), u1(2)),
            su2(1),
        );
        let product_source = pair(product_sector);
        assert_eq!(
            multiplicity_free_braid_tree_pair(
                &product_rule,
                &product_source,
                &[0],
                &[1],
                &[8],
                &[3],
            )
            .unwrap(),
            vec![(product_source, 1.0)]
        );

        let scalar_source = FusionTreePairKey::try_pair_from_sector_ids(
            Vec::<usize>::new(),
            Vec::<usize>::new(), z2_even().id(),
            Vec::<bool>::new(),
            Vec::<bool>::new(),
            Vec::<usize>::new(),
            Vec::<usize>::new(),
            Vec::<usize>::new(),
            Vec::<usize>::new(),
        ).unwrap();
        assert_eq!(
            unique_braid_tree_pair(
                &Z2FusionRule,
                &scalar_source,
                &[],
                &[],
                &[],
                &[],
            )
            .unwrap(),
            (scalar_source, 1.0)
        );
    }

    #[test]
    fn unique_repartition_tree_pair_moves_domain_to_reversed_dual_codomain() {
        let source = FusionTreePairKey::try_pair_from_sector_ids(
            [1],
            [0, 1], 1,
            [false],
            [false, true],
            [],
            [],
            [],
            [1],
        ).unwrap();

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
            &[MultiplicityIndex::ONE, MultiplicityIndex::ONE]
        );
        assert!(all_out.domain_uncoupled().is_empty());
        assert_eq!(all_out.domain_tree().coupled(), SectorId::new(0));
    }

    #[test]
    fn unique_braid_tree_pair_matches_single_tree_when_domain_is_empty() {
        let source = FusionTreePairKey::pair(
            FusionTreeKey::try_from_sector_ids([1, 1], 0, [false, true], [], [1]).unwrap(),
            FusionTreeKey::new(
                Vec::<SectorId>::new(),
                FermionParityFusionRule.vacuum(),
                Vec::<bool>::new(),
                Vec::<SectorId>::new(),
                Vec::<MultiplicityIndex>::new(),
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
        assert_eq!(
            braided.domain_tree().coupled(),
            FermionParityFusionRule.vacuum()
        );
    }

    #[test]
    fn unique_permute_tree_pair_handles_domain_only_swap() {
        let source = FusionTreePairKey::try_pair_from_sector_ids(
            [1],
            [0, 1], 1,
            [false],
            [false, true],
            [],
            [],
            [],
            [1],
        ).unwrap();

        let (permuted, coefficient) =
            unique_permute_tree_pair(&Z2FusionRule, &source, &[0], &[2, 1]).unwrap();

        assert_eq!(coefficient, 1.0);
        assert_eq!(permuted.codomain_uncoupled(), &[SectorId::new(1)]);
        assert_eq!(
            permuted.domain_uncoupled(),
            &[SectorId::new(1), SectorId::new(0)]
        );
        assert_eq!(permuted.domain_is_dual(), &[true, false]);
        assert_eq!(permuted.domain_vertices(), &[MultiplicityIndex::ONE]);
    }

    #[test]
    fn unique_permute_tree_pair_includes_codomain_domain_crossing() {
        let source = FusionTreePairKey::try_pair_from_sector_ids(
            [1],
            [1], 1,
            [false],
            [true],
            [],
            [],
            [],
            [],
        ).unwrap();

        let (permuted, coefficient) =
            unique_permute_tree_pair(&FermionParityFusionRule, &source, &[1], &[0]).unwrap();

        assert_eq!(coefficient, -1.0);
        assert_eq!(permuted.codomain_uncoupled(), &[SectorId::new(1)]);
        assert_eq!(permuted.codomain_is_dual(), &[false]);
        assert_eq!(permuted.domain_uncoupled(), &[SectorId::new(1)]);
        assert_eq!(permuted.domain_is_dual(), &[true]);
    }

    #[test]
    fn prepared_fermionic_domain_crossing_is_exactly_negative_one() {
        // What: an actual odd fZ2 domain leg crossing an odd codomain leg
        // retains the exact TensorKit fermionic phase through prepared replay.
        let source = FusionTreePairKey::try_pair_from_sector_ids(
            [1],
            [1], 1,
            [false],
            [true],
            [],
            [],
            [],
            [],
        ).unwrap();
        let prepared = PreparedTreePairOperation::prepare_permute(
            &FermionParityFusionRule,
            1,
            1,
            &[1],
            &[0],
        )
        .unwrap();

        let (destination, coefficient) = prepared
            .execute_unique_rigid(&FermionParityFusionRule, &source)
            .unwrap();

        assert_eq!(coefficient, -1.0);
        assert_eq!(destination.codomain_uncoupled(), &[SectorId::new(1)]);
        assert_eq!(destination.domain_uncoupled(), &[SectorId::new(1)]);
        assert_eq!(destination.codomain_is_dual(), &[false]);
        assert_eq!(destination.domain_is_dual(), &[true]);
    }

    #[test]
    fn prepared_permute_reverses_domain_levels_before_artin_lowering() {
        // What: TensorKit linearizes incoming legs in reverse order, so a
        // domain-only swap is prepared as the inverse Artin generator.
        let prepared =
            PreparedTreePairOperation::prepare_permute(&Z2FusionRule, 1, 2, &[0], &[2, 1])
                .unwrap();
        let PreparedTreePairPlan::Braid(braid) = prepared.plan else {
            panic!("domain-only swap must prepare a braid");
        };

        assert_eq!(
            braid.artin_steps.as_slice(),
            &[PreparedArtinStep {
                index: 1,
                inverse: true,
            }]
        );
    }

    #[test]
    fn prepared_permute_revalidates_symmetric_capability_for_reused_rule() {
        // What: a prepared permutation cannot become an identity, repartition,
        // or general braid when executed with a non-symmetric provider.
        let source = FusionTreePairKey::try_pair_from_sector_ids(
            [1],
            [1], 1,
            [false],
            [false],
            [],
            [],
            [],
            [],
        ).unwrap();
        let plans = [
            PreparedTreePairOperation::prepare_permute(&Z2FusionRule, 1, 1, &[0], &[1]).unwrap(),
            PreparedTreePairOperation::prepare_permute(&Z2FusionRule, 1, 1, &[0, 1], &[]).unwrap(),
            PreparedTreePairOperation::prepare_permute(&Z2FusionRule, 1, 1, &[1], &[0]).unwrap(),
        ];
        assert!(matches!(plans[0].plan, PreparedTreePairPlan::Identity));
        assert!(matches!(
            plans[1].plan,
            PreparedTreePairPlan::Repartition
        ));
        assert!(matches!(plans[2].plan, PreparedTreePairPlan::Braid(_)));

        for prepared in plans {
            assert_eq!(
                prepared.execute_unique_rigid(&AsymmetricAnyonicRule, &source),
                Err(CoreError::UnsupportedBraidingStyle {
                    expected: "symmetric braiding",
                    actual: BraidingStyleKind::Anyonic,
                })
            );
        }
    }

    #[test]
    fn prepared_transpose_fixes_both_cycle_directions_once() {
        // What: clockwise and anticlockwise TensorKit cyclic permutations are
        // lowered to one fixed direction/count before any source execution.
        let source = FusionTreePairKey::try_pair_from_sector_ids(
            [1, 0],
            [1, 0], 1,
            [false, false],
            [false, false],
            [],
            [],
            [1],
            [1],
        ).unwrap();
        let cases = [
            (
                &[1, 3][..],
                &[0, 2][..],
                PreparedCycleDirection::Clockwise,
            ),
            (
                &[2, 0][..],
                &[3, 1][..],
                PreparedCycleDirection::Anticlockwise,
            ),
        ];
        for (codomain, domain, expected_direction) in cases {
            let prepared =
                PreparedTreePairOperation::prepare_transpose(2, 2, codomain, domain).unwrap();
            assert_eq!(
                prepared.plan,
                PreparedTreePairPlan::Transpose {
                    direction: expected_direction,
                    count: 1,
                }
            );
            let actual = prepared.execute_unique_rigid(&Z2FusionRule, &source).unwrap();
            let repartitioned = unique_rigid_repartition_tree_pair_unchecked(
                &Z2FusionRule,
                &source,
                codomain.len(),
            )
            .unwrap();
            let (oracle_tree, cycle_coefficient) = match expected_direction {
                PreparedCycleDirection::Clockwise => {
                    unique_rigid_cycle_clockwise_tree_pair(&Z2FusionRule, &repartitioned.0).unwrap()
                }
                PreparedCycleDirection::Anticlockwise => {
                    unique_rigid_cycle_anticlockwise_tree_pair(&Z2FusionRule, &repartitioned.0)
                        .unwrap()
                }
            };
            let oracle = (oracle_tree, repartitioned.1 * cycle_coefficient);
            assert_eq!(actual, oracle);
        }
    }

    #[test]
    fn unique_prepared_executor_rejects_simple_for_every_plan_variant() {
        // What: a general multiplicity-free plan never becomes a Unique plan
        // merely because its operation variant is an identity or repartition.
        let source = FusionTreePairKey::try_pair_from_sector_ids(
            [1],
            [1], 1,
            [false],
            [false],
            [],
            [],
            [],
            [],
        ).unwrap();
        let plans = [
            PreparedTreePairOperation::prepare_braid(
                &SU2FusionRule,
                1,
                1,
                &[0],
                &[1],
                &[0],
                &[1],
            )
            .unwrap(),
            PreparedTreePairOperation::prepare_braid(
                &SU2FusionRule,
                1,
                1,
                &[0, 1],
                &[],
                &[0],
                &[1],
            )
            .unwrap(),
            PreparedTreePairOperation::prepare_braid(
                &SU2FusionRule,
                1,
                1,
                &[1],
                &[0],
                &[0],
                &[1],
            )
            .unwrap(),
            PreparedTreePairOperation::prepare_transpose(1, 1, &[1], &[0]).unwrap(),
        ];
        assert!(matches!(plans[0].plan, PreparedTreePairPlan::Identity));
        assert!(matches!(plans[1].plan, PreparedTreePairPlan::Repartition));
        assert!(matches!(plans[2].plan, PreparedTreePairPlan::Braid(_)));
        assert!(matches!(
            plans[3].plan,
            PreparedTreePairPlan::Transpose { .. }
        ));

        for prepared in plans {
            assert_eq!(
                prepared.execute_unique_rigid(&SU2FusionRule, &source),
                Err(CoreError::UnsupportedFusionStyle {
                    expected: FusionStyleKind::Unique,
                    actual: FusionStyleKind::Simple,
                })
            );
        }
    }

    #[test]
    fn prepared_operation_preserves_validation_error_precedence() {
        // What: level dimensions still fail before identity short-circuiting,
        // while transpose still reports the original unlinearized permutation.
        assert_eq!(
            PreparedTreePairOperation::prepare_braid(
                &SU2FusionRule,
                1,
                1,
                &[0],
                &[1],
                &[],
                &[1],
            ),
            Err(CoreError::DimensionMismatch {
                expected: 1,
                actual: 0,
            })
        );
        assert_eq!(
            PreparedTreePairOperation::prepare_transpose(2, 1, &[0, 2], &[2]),
            Err(CoreError::InvalidPermutation {
                permutation: vec![0, 2, 2],
                rank: 3,
            })
        );
    }

    #[test]
    fn public_prepared_braid_preserves_level_then_style_error_precedence() {
        let rule = Su3FusionRule::new();

        // What: malformed level lengths retain precedence over the public
        // multiplicity-free style gate.
        assert_eq!(
            PreparedTreePairOperation::prepare_braid(
                &rule,
                2,
                0,
                &[0, 0],
                &[],
                &[0],
                &[],
            ),
            Err(CoreError::DimensionMismatch {
                expected: 2,
                actual: 1,
            })
        );

        // What: once level lengths are valid, the existing public API rejects
        // Generic fusion before inspecting an invalid permutation.
        assert_eq!(
            PreparedTreePairOperation::prepare_braid(
                &rule,
                2,
                0,
                &[0, 0],
                &[],
                &[0, 1],
                &[],
            ),
            Err(CoreError::UnsupportedFusionStyle {
                expected: FusionStyleKind::Simple,
                actual: FusionStyleKind::Generic,
            })
        );
    }

    #[test]
    fn validation_only_tree_pair_syntax_handles_large_and_rank_zero_maps() {
        let codomain_rank = 10;
        let domain_rank = 9;
        let codomain = (0..codomain_rank).collect::<Vec<_>>();
        let domain =
            (codomain_rank..codomain_rank + domain_rank).collect::<Vec<_>>();
        let codomain_levels = (0..codomain_rank).collect::<Vec<_>>();
        let domain_levels = (codomain_rank..codomain_rank + domain_rank).collect::<Vec<_>>();

        // What: the validation-only API handles ranks beyond SmallVec's inline
        // permutation capacity without requiring a prepared Artin plan.
        PreparedTreePairOperation::validate_permute_syntax(
            codomain_rank,
            domain_rank,
            &codomain,
            &domain,
        )
        .unwrap();
        PreparedTreePairOperation::validate_braid_syntax(
            codomain_rank,
            domain_rank,
            &codomain,
            &domain,
            &codomain_levels,
            &domain_levels,
        )
        .unwrap();
        PreparedTreePairOperation::validate_transpose_syntax(
            codomain_rank,
            domain_rank,
            &codomain,
            &domain,
        )
        .unwrap();

        // What: the empty tensor map remains a valid identity operation.
        PreparedTreePairOperation::validate_permute_syntax(0, 0, &[], &[]).unwrap();
        PreparedTreePairOperation::validate_braid_syntax(0, 0, &[], &[], &[], &[]).unwrap();
        PreparedTreePairOperation::validate_transpose_syntax(0, 0, &[], &[]).unwrap();
    }

    #[test]
    fn unique_transpose_tree_pair_is_cyclic_and_reversible() {
        let source = FusionTreePairKey::try_pair_from_sector_ids(
            [1],
            [1], 1,
            [false],
            [true],
            [],
            [],
            [],
            [],
        ).unwrap();

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
        let source = FusionTreePairKey::try_pair_from_sector_ids(
            [1, 0],
            [1, 0], 1,
            [false, false],
            [false, false],
            [],
            [],
            [1],
            [1],
        ).unwrap();
        let expected = FusionTreePairKey::try_pair_from_sector_ids(
            [0, 0],
            [1, 1], 0,
            [false, true],
            [true, false],
            [],
            [],
            [1],
            [1],
        ).unwrap();

        let (transposed, coefficient) =
            unique_transpose_tree_pair(&Z2FusionRule, &source, &[1, 3], &[0, 2]).unwrap();

        assert_eq!(coefficient, 1.0);
        assert_eq!(transposed, expected);
    }

    #[test]
    fn unique_transpose_tree_pair_matches_tensorkit_anticlockwise_cycle() {
        let source = FusionTreePairKey::try_pair_from_sector_ids(
            [1, 0],
            [1, 0], 1,
            [false, false],
            [false, false],
            [],
            [],
            [1],
            [1],
        ).unwrap();
        let expected = FusionTreePairKey::try_pair_from_sector_ids(
            [1, 1],
            [0, 0], 0,
            [true, false],
            [false, true],
            [],
            [],
            [1],
            [1],
        ).unwrap();

        let (transposed, coefficient) =
            unique_transpose_tree_pair(&Z2FusionRule, &source, &[2, 0], &[3, 1]).unwrap();

        assert_eq!(coefficient, 1.0);
        assert_eq!(transposed, expected);
    }

    #[test]
    fn unique_transpose_tree_pair_rejects_noncyclic_permutation() {
        let source = FusionTreePairKey::try_pair_from_sector_ids(
            [1, 0],
            [1], 1,
            [false, false],
            [false],
            [],
            [],
            [1],
            [],
        ).unwrap();

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
        let first = FusionTreePairKey::try_pair_from_sector_ids(
            [1],
            [1], 1,
            [false],
            [true],
            [],
            [],
            [],
            [],
        ).unwrap();
        let second = FusionTreePairKey::try_pair_from_sector_ids(
            [0],
            [0], 0,
            [false],
            [true],
            [],
            [],
            [],
            [],
        ).unwrap();
        let structure = packed_fixture_structure(
            2,
            [
                (BlockKey::from(second.clone()), vec![1, 4]),
                (BlockKey::from(first.clone()), vec![2, 3]),
            ],
        )
        .unwrap();

        let first_block = structure.fusion_tree_pair_block(&first).unwrap();
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
        let first = FusionTreePairKey::try_pair_from_sector_ids(
            [1],
            [1], 1,
            [false],
            [true],
            [],
            [],
            [],
            [],
        ).unwrap();
        let second = FusionTreePairKey::try_pair_from_sector_ids(
            [0],
            [0], 0,
            [false],
            [true],
            [],
            [],
            [],
            [],
        ).unwrap();
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
        let key = FusionTreePairKey::try_pair_from_sector_ids(
            [1],
            [1], 1,
            [false],
            [true],
            [],
            [],
            [],
            [],
        ).unwrap();
        let other = FusionTreePairKey::try_pair_from_sector_ids(
            [0],
            [0], 0,
            [false],
            [true],
            [],
            [],
            [],
            [],
        ).unwrap();
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
    fn subblock_by_tree_reports_missing_fusion_tree_key() {
        let existing = FusionTreePairKey::try_pair_from_sector_ids(
            [0],
            [0], 0,
            [false],
            [true],
            [],
            [],
            [],
            [],
        ).unwrap();
        let missing = FusionTreePairKey::try_pair_from_sector_ids(
            [1],
            [1], 1,
            [false],
            [true],
            [],
            [],
            [],
            [],
        ).unwrap();
        let structure =
            packed_fixture_structure(2, [(BlockKey::from(existing), vec![1, 1])]).unwrap();
        let space = TensorMapSpace::<1, 1>::from_dims([1], [1]).unwrap();
        let tensor =
            TensorMap::<f64, 1, 1>::from_vec_with_structure(vec![1.0], space, structure).unwrap();

        let err = tensor.subblock_by_tree(&missing).unwrap_err();

        assert_eq!(
            err,
            CoreError::MissingBlockKey {
                key: Box::new(BlockKey::from(missing)),
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

    #[cfg(target_pointer_width = "64")]
    #[test]
    fn packed_product_codec_is_association_independent() {
        // What: fixed-width product IDs flatten numerically regardless of the
        // source-level association used to build the same ordered leaves.
        type FpU1Codec = PackedProductCodec<Fz2SectorLayout, U1SectorLayout>;
        type FpU1Layout = ProductSectorLayout<Fz2SectorLayout, U1SectorLayout>;
        type LeftAssociated = PackedProductCodec<FpU1Layout, Su2SectorLayout>;
        type U1Su2Codec = PackedProductCodec<U1SectorLayout, Su2SectorLayout>;
        type U1Su2Layout = ProductSectorLayout<U1SectorLayout, Su2SectorLayout>;
        type RightAssociated = PackedProductCodec<Fz2SectorLayout, U1Su2Layout>;

        for (parity, charge, twice_spin) in [
            (z2_even(), i32::MIN, 0),
            (z2_odd(), -1, 1),
            (z2_even(), 0, 2),
            (z2_odd(), i32::MAX, 254),
        ] {
            let left = FpU1Codec::encode(parity, u1(charge));
            let left_associated = LeftAssociated::encode(left, su2(twice_spin));
            let right = U1Su2Codec::encode(u1(charge), su2(twice_spin));
            let right_associated = RightAssociated::encode(parity, right);
            assert_eq!(left_associated, right_associated);
        }
    }

    #[cfg(target_pointer_width = "64")]
    #[test]
    fn packed_product_codec_covers_the_builtin_leaf_domains() {
        // What: the codec represents the complete i32 U1 label domain
        // together with every currently supported SU2 label; algebraic
        // overflow behavior is tracked separately in issue #274.
        type FpU1Codec = PackedProductCodec<Fz2SectorLayout, U1SectorLayout>;
        type FpU1Layout = ProductSectorLayout<Fz2SectorLayout, U1SectorLayout>;
        type TripleCodec = PackedProductCodec<FpU1Layout, Su2SectorLayout>;

        for charge in [i32::MIN, -1, 0, 1, i32::MAX] {
            for twice_spin in [0, 1, 127, 254] {
                let inner = FpU1Codec::encode(z2_odd(), u1(charge));
                let encoded = TripleCodec::encode(inner, su2(twice_spin));
                let (decoded_inner, decoded_spin) = TripleCodec::decode(encoded).unwrap();
                let (decoded_parity, decoded_charge) =
                    FpU1Codec::decode(decoded_inner).unwrap();
                assert_eq!(decoded_parity, z2_odd());
                assert_eq!(decoded_charge, u1(charge));
                assert_eq!(decoded_spin, su2(twice_spin));
            }
        }
    }

    struct PermissiveOneBitLayout;

    impl PackedSectorLayout for PermissiveOneBitLayout {
        const BITS: u32 = 1;

        fn validate(_sector: SectorId) -> Result<(), ProductSectorCodecError> {
            Ok(())
        }
    }

    struct ZeroBitLayout;

    impl PackedSectorLayout for ZeroBitLayout {
        const BITS: u32 = 0;

        fn validate(sector: SectorId) -> Result<(), ProductSectorCodecError> {
            (sector.id() == 0)
                .then_some(())
                .ok_or(ProductSectorCodecError::InvalidHighBits {
                    sector,
                    total_bits: 0,
                })
        }
    }

    struct FullWidthLayout;

    impl PackedSectorLayout for FullWidthLayout {
        const BITS: u32 = usize::BITS;

        fn validate(_sector: SectorId) -> Result<(), ProductSectorCodecError> {
            Ok(())
        }
    }

    #[test]
    fn packed_product_codec_reports_invalid_components_and_widths() {
        // What: malformed packed IDs and layouts wider than usize fail with a
        // typed reason instead of panicking, wrapping, or silently truncating.
        type FpU1Codec = PackedProductCodec<Fz2SectorLayout, U1SectorLayout>;
        let invalid_parity = FpU1Codec::encode_checked(SectorId::new(2), u1(0));
        assert!(matches!(
            invalid_parity,
            Err(ProductSectorCodecError::ComponentOutOfRange {
                component: ProductSectorComponent::Left,
                ..
            })
        ));

        type MaliciousCodec = PackedProductCodec<PermissiveOneBitLayout, Fz2SectorLayout>;
        assert!(matches!(
            MaliciousCodec::encode_checked(SectorId::new(2), SectorId::new(0)),
            Err(ProductSectorCodecError::ComponentOutOfRange {
                component: ProductSectorComponent::Left,
                ..
            })
        ));

        type TripleLayout =
            ProductSectorLayout<ProductSectorLayout<Fz2SectorLayout, U1SectorLayout>, Su2SectorLayout>;
        assert_eq!(TripleLayout::BITS, 41);
        #[cfg(target_pointer_width = "64")]
        {
            let invalid_high_bits = SectorId::new(1usize << TripleLayout::BITS);
            assert!(matches!(
                TripleLayout::validate(invalid_high_bits),
                Err(ProductSectorCodecError::InvalidHighBits { .. })
            ));
        }

        type TooWide = PackedProductCodec<
            ProductSectorLayout<U1SectorLayout, U1SectorLayout>,
            U1SectorLayout,
        >;
        assert!(matches!(
            TooWide::encode_checked(SectorId::new(0), SectorId::new(0)),
            Err(ProductSectorCodecError::WidthOverflow { .. })
        ));

        type FullThenZero = PackedProductCodec<FullWidthLayout, ZeroBitLayout>;
        let full = FullThenZero::encode_checked(SectorId::new(usize::MAX), SectorId::new(0))
            .unwrap();
        assert_eq!(full, SectorId::new(usize::MAX));
        assert_eq!(
            FullThenZero::decode_checked(full).unwrap(),
            (SectorId::new(usize::MAX), SectorId::new(0))
        );

        type ZeroThenParity = PackedProductCodec<ZeroBitLayout, Fz2SectorLayout>;
        assert_eq!(
            ZeroThenParity::encode_checked(SectorId::new(0), SectorId::new(1)).unwrap(),
            SectorId::new(1)
        );
    }

    #[test]
    fn packed_and_tensorkit_codecs_remain_distinct_compatible_options() {
        // What: the expert Cantor codec keeps its historical IDs while the
        // packed codec has a distinct rule identity and round-trips the same
        // semantic components.
        type PackedCodec = PackedProductCodec<Fz2SectorLayout, U1SectorLayout>;
        type PackedRule =
            ProductFusionRule<FermionParityFusionRule, U1FusionRule, PackedCodec>;
        type CantorRule = ProductFusionRule<FermionParityFusionRule, U1FusionRule>;

        let parity = z2_odd();
        let charge = u1(2);
        let packed = PackedCodec::encode(parity, charge);
        let cantor = TensorKitProductCodec::encode(parity, charge);
        assert_ne!(packed, cantor);
        assert_eq!(PackedCodec::decode(packed), Some((parity, charge)));
        assert_eq!(
            TensorKitProductCodec::decode(cantor),
            Some((parity, charge))
        );
        assert_ne!(
            PackedRule::default().rule_identity(),
            CantorRule::default().rule_identity()
        );
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
    fn product_rule_reuses_its_memoized_identity_node() {
        let rule = product_fusion_rule(Z2FusionRule, U1FusionRule);
        assert!(rule.identity.get().is_none());

        let first = rule.rule_identity();
        let cached = rule.identity.get().unwrap() as *const RuleIdentity;
        let second = rule.rule_identity();

        assert_eq!(first, second);
        assert_eq!(cached, rule.identity.get().unwrap() as *const RuleIdentity);
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
            assert_eq!(key.coupled(), coupled);
            assert_eq!(key.codomain_uncoupled(), &[a, b]);
            assert_eq!(key.domain_uncoupled(), &[coupled]);
            assert_eq!(key.codomain_is_dual(), &[false, false]);
            assert_eq!(key.domain_is_dual(), &[false]);
            assert_eq!(key.codomain_innerlines(), &[]);
            assert_eq!(key.domain_innerlines(), &[]);
            assert_eq!(key.codomain_vertices(), &[MultiplicityIndex::ONE]);
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
        assert_eq!(keys[0].coupled(), a);

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
        let tree = FusionTreeKey::try_from_sector_ids(
            [1, 1, 1, 1], 0,
            [false, false, false, false],
            [0, 1],
            [1, 1, 1],
        ).unwrap();

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
        let source = FusionTreePairKey::try_pair_from_sector_ids(
            [1],
            [1], 1,
            [false],
            [false],
            [],
            [],
            [],
            [],
        ).unwrap();

        let all_codomain = multiplicity_free_repartition_tree_pair(&rule, &source, 2).unwrap();
        assert_eq!(all_codomain.len(), 1);
        assert_eq!(
            all_codomain[0].0.codomain_uncoupled(),
            &[SectorId::new(1); 2]
        );
        assert_eq!(all_codomain[0].0.codomain_is_dual(), &[false, true]);
        assert_eq!(all_codomain[0].0.codomain_innerlines(), &[]);
        assert_eq!(all_codomain[0].0.codomain_vertices(), &[MultiplicityIndex::ONE]);
        assert_eq!(
            all_codomain[0].0.codomain_tree().coupled(),
            SectorId::new(0)
        );
        assert_eq!(all_codomain[0].0.domain_uncoupled(), &[]);
        assert_eq!(
            all_codomain[0].0.domain_tree().coupled(),
            SectorId::new(0)
        );
        assert!((all_codomain[0].1 - 2.0_f64.sqrt()).abs() < 1.0e-12);

        let all_domain = multiplicity_free_repartition_tree_pair(&rule, &source, 0).unwrap();
        assert_eq!(all_domain.len(), 1);
        assert_eq!(all_domain[0].0.codomain_uncoupled(), &[]);
        assert_eq!(
            all_domain[0].0.codomain_tree().coupled(),
            SectorId::new(0)
        );
        assert_eq!(all_domain[0].0.domain_uncoupled(), &[SectorId::new(1); 2]);
        assert_eq!(all_domain[0].0.domain_is_dual(), &[false, true]);
        assert!((all_domain[0].1 - 2.0_f64.sqrt()).abs() < 1.0e-12);
    }

    #[test]
    fn multiplicity_free_su2_permute_tree_pair_matches_tensorkit_swap() {
        let rule = SU2FusionRule;
        let source = FusionTreePairKey::try_pair_from_sector_ids(
            [1],
            [1], 1,
            [false],
            [false],
            [],
            [],
            [],
            [],
        ).unwrap();

        let permuted = multiplicity_free_permute_tree_pair(&rule, &source, &[1], &[0]).unwrap();

        assert_eq!(permuted.len(), 1);
        assert_eq!(permuted[0].0.codomain_uncoupled(), &[SectorId::new(1)]);
        assert_eq!(permuted[0].0.domain_uncoupled(), &[SectorId::new(1)]);
        assert_eq!(permuted[0].0.codomain_is_dual(), &[true]);
        assert_eq!(permuted[0].0.domain_is_dual(), &[true]);
        assert_eq!(
            permuted[0].0.codomain_tree().coupled(),
            SectorId::new(1)
        );
        assert_eq!(
            permuted[0].0.domain_tree().coupled(),
            SectorId::new(1)
        );
        assert!((permuted[0].1 - 1.0).abs() < 1.0e-12);
    }

    #[test]
    fn prepared_simple_pair_matches_explicit_generic_composition() {
        // What: a prepared Simple-fusion pair operation equals the explicit
        // repartition -> all-codomain Artin -> repartition composition.
        let rule = SU2FusionRule;
        let source = FusionTreePairKey::try_pair_from_sector_ids(
            [1],
            [1], 1,
            [false],
            [false],
            [],
            [],
            [],
            [],
        ).unwrap();
        let prepared =
            PreparedTreePairOperation::prepare_permute(&rule, 1, 1, &[1], &[0]).unwrap();
        let actual = prepared
            .execute_multiplicity_free(&rule, &source)
            .unwrap();

        let all_codomain = multiplicity_free_repartition_tree_pair(&rule, &source, 2).unwrap();
        let braided = compose_tree_pair_terms(&rule, all_codomain, |rule, key| {
            multiplicity_free_braid_tree(
                rule,
                key.codomain_tree(),
                &[1, 0],
                &[0, 1],
            )
            .map(|terms| {
                terms
                    .into_iter()
                    .map(|(tree, coefficient)| {
                        (
                            FusionTreePairKey::pair(
                                tree,
                                key.domain_tree().clone(),
                            ),
                            coefficient,
                        )
                    })
                    .collect::<Vec<_>>()
            })
        })
        .unwrap();
        let expected = multiplicity_free_repartition_terms(&rule, braided, 1).unwrap();

        assert_eq!(
            actual.iter().map(|(key, _)| key).collect::<Vec<_>>(),
            expected.iter().map(|(key, _)| key).collect::<Vec<_>>()
        );
        for ((_, actual), (_, expected)) in actual.iter().zip(expected) {
            assert!((*actual - expected).abs() < 1.0e-12);
        }
    }

    fn u1_nonselfdual_tree_pair_fixture() -> FusionTreePairKey {
        FusionTreePairKey::pair(
            FusionTreeKey::new(
                [u1(1), u1(2)], u1(3),
                [false, false],
                Vec::<SectorId>::new(),
                [MultiplicityIndex::ONE],
            ),
            FusionTreeKey::new(
                [u1(3)], u1(3),
                [false],
                Vec::<SectorId>::new(),
                Vec::<MultiplicityIndex>::new(),
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
        assert_eq!(out[0].0.codomain_tree().coupled(), u1(1));
        assert_eq!(out[0].0.codomain_is_dual(), &[false]);
        assert_eq!(out[0].0.codomain_innerlines(), &[]);
        assert_eq!(out[0].0.codomain_vertices(), &[]);
        assert_eq!(out[0].0.domain_uncoupled(), &[u1(3), u1(-2)]);
        assert_eq!(out[0].0.domain_tree().coupled(), u1(1));
        assert_eq!(out[0].0.domain_is_dual(), &[false, true]);
        assert_eq!(out[0].0.domain_innerlines(), &[]);
        assert_eq!(out[0].0.domain_vertices(), &[MultiplicityIndex::ONE]);
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
        assert_eq!(out[0].0.codomain_tree().coupled(), u1(2));
        assert_eq!(out[0].0.codomain_is_dual(), &[false]);
        assert_eq!(out[0].0.codomain_innerlines(), &[]);
        assert_eq!(out[0].0.codomain_vertices(), &[]);
        assert_eq!(out[0].0.domain_uncoupled(), &[u1(-1), u1(3)]);
        assert_eq!(out[0].0.domain_tree().coupled(), u1(2));
        assert_eq!(out[0].0.domain_is_dual(), &[true, false]);
        assert_eq!(out[0].0.domain_innerlines(), &[]);
        assert_eq!(out[0].0.domain_vertices(), &[MultiplicityIndex::ONE]);
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
        assert_eq!(out[0].0.codomain_tree().coupled(), u1(0));
        assert_eq!(out[0].0.codomain_is_dual(), &[] as &[bool]);
        assert_eq!(out[0].0.codomain_innerlines(), &[]);
        assert_eq!(out[0].0.codomain_vertices(), &[]);
        assert_eq!(out[0].0.domain_uncoupled(), &[u1(3), u1(-2), u1(-1)]);
        assert_eq!(out[0].0.domain_tree().coupled(), u1(0));
        assert_eq!(out[0].0.domain_is_dual(), &[false, true, true]);
        assert_eq!(out[0].0.domain_innerlines(), &[u1(1)]);
        assert_eq!(
            out[0].0.domain_vertices(),
            &[MultiplicityIndex::ONE, MultiplicityIndex::ONE]
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
        assert_eq!(out[0].0.codomain_tree().coupled(), u1(0));
        assert_eq!(out[0].0.codomain_is_dual(), &[false, false, true]);
        assert_eq!(out[0].0.codomain_innerlines(), &[u1(3)]);
        assert_eq!(
            out[0].0.codomain_vertices(),
            &[MultiplicityIndex::ONE, MultiplicityIndex::ONE]
        );
        assert_eq!(out[0].0.domain_uncoupled(), &[]);
        assert_eq!(out[0].0.domain_tree().coupled(), u1(0));
        assert_eq!(out[0].0.domain_is_dual(), &[] as &[bool]);
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
        assert_eq!(out[0].0.codomain_tree().coupled(), u1(-1));
        assert_eq!(out[0].0.codomain_is_dual(), &[false, true]);
        assert_eq!(out[0].0.codomain_innerlines(), &[]);
        assert_eq!(out[0].0.codomain_vertices(), &[MultiplicityIndex::ONE]);
        assert_eq!(out[0].0.domain_uncoupled(), &[u1(-1)]);
        assert_eq!(out[0].0.domain_tree().coupled(), u1(-1));
        assert_eq!(out[0].0.domain_is_dual(), &[true]);
        assert_eq!(out[0].0.domain_innerlines(), &[]);
        assert_eq!(out[0].0.domain_vertices(), &[]);
    }

    #[test]
    fn nested_product_elementary_bend_keeps_the_fermionic_phase() {
        // What: the elementary bend of an odd fZ2 pair retains the negative
        // product-category phase independently of the block/per-source runners.
        type FpU1Rule = ProductFusionRule<FermionParityFusionRule, U1FusionRule>;
        type ProductRule = ProductFusionRule<FpU1Rule, SU2FusionRule>;
        let left_rule = FpU1Rule::default();
        let rule = ProductRule::default();
        let coupled =
            rule.encode_sector(left_rule.encode_sector(z2_even(), u1(0)), su2(1));
        let odd_half =
            rule.encode_sector(left_rule.encode_sector(z2_odd(), u1(1)), su2(1));
        let odd_one =
            rule.encode_sector(left_rule.encode_sector(z2_odd(), u1(-1)), su2(2));
        let source = FusionTreePairKey::pair(
            FusionTreeKey::try_new_for_rule(
                &rule,
                [coupled], coupled,
                [false],
                [],
                [],
            )
            .unwrap(),
            FusionTreeKey::try_new_for_rule(
                &rule,
                [odd_half, odd_one], coupled,
                [false, true],
                [],
                [MultiplicityIndex::ONE],
            )
            .unwrap(),
        );

        let bent = multiplicity_free_bendleft_tree_pair(&rule, &source).unwrap();
        assert_eq!(bent.len(), 1);
        assert!(bent[0].1 < 0.0);
        let restored = multiplicity_free_bendright_tree_pair(&rule, &bent[0].0).unwrap();
        assert_eq!(restored.len(), 1);
        assert_eq!(restored[0].0, source);
        assert!((bent[0].1 * restored[0].1 - 1.0).abs() < 1.0e-12);
    }

    #[test]
    fn bendright_preserves_local_error_precedence_before_missing_duality() {
        // What: local coupled/innerline errors remain observable before a
        // missing final duality flag when several malformed fields coexist.
        let missing_innerline = FusionTreePairKey::pair(
            FusionTreeKey::try_from_sector_ids(
                [1, 1, 1], 1,
                [false, false],
                [],
                [1, 1],
            ).unwrap(),
            FusionTreeKey::try_from_sector_ids([], 0, [], [], []).unwrap(),
        );
        assert_eq!(
            multiplicity_free_bendright_tree_pair(&SU2FusionRule, &missing_innerline).unwrap_err(),
            CoreError::MalformedFusionTree {
                message: "bendright requires the last codomain innerline",
            }
        );

        let mismatched_coupled = FusionTreePairKey::pair(
            FusionTreeKey::try_from_sector_ids([1, 1], 0, [false], [], [1]).unwrap(),
            FusionTreeKey::try_from_sector_ids([1], 1, [false], [], []).unwrap(),
        );
        assert_eq!(
            multiplicity_free_bendright_tree_pair(&SU2FusionRule, &mismatched_coupled).unwrap_err(),
            CoreError::MalformedFusionTree {
                message: "fusion tree pair requires matching coupled sectors",
            }
        );
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
        assert_eq!(key.coupled(), U1Irrep::new(2).sector_id());
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
            &BlockKey::from(FusionTreePairKey::try_pair_from_sector_ids(
                [0],
                [0], 0,
                [false],
                [false],
                [],
                [],
                [],
                [],
            ).unwrap())
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
            BlockSpec::column_major_with_key(BlockKey::opaque([7]), vec![2, 2], 0).unwrap();
        let second =
            BlockSpec::column_major_with_key(BlockKey::opaque([7]), vec![1, 3], 4).unwrap();

        let err = BlockStructure::from_blocks_with_rank(2, vec![first, second]).unwrap_err();

        assert_eq!(
            err,
            CoreError::DuplicateBlockKey {
                key: Box::new(BlockKey::opaque([7]))
            }
        );
    }

    #[test]
    fn block_structure_validates_degeneracy_before_sector_keys() {
        // What: preparing an owned block structure preserves the historical
        // error order when both degeneracy metadata and sector keys are bad.
        let key = BlockKey::opaque([7]);
        let first = BlockSpec::column_major_with_key(key.clone(), vec![2, 2], 0).unwrap();
        let malformed = BlockSpec {
            key,
            shape: smallvec![1, 3],
            strides: smallvec![1],
            offset: 4,
        };

        assert_eq!(
            BlockStructure::from_blocks_with_rank(2, vec![first, malformed]),
            Err(CoreError::RankMismatch {
                shape: 2,
                strides: 1,
            })
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
    fn fusion_tree_pair_key_records_tensorkit_subblock_pair_fields() {
        let key = FusionTreePairKey::try_pair_from_sector_ids(
            [2, 3],
            [5, 7], 11,
            [false, true],
            [true, false],
            [13],
            [17],
            [19, 23],
            [29, 31],
        ).unwrap();

        assert_eq!(
            key.codomain_uncoupled(),
            &[SectorId::new(2), SectorId::new(3)]
        );
        assert_eq!(
            key.domain_uncoupled(),
            &[SectorId::new(5), SectorId::new(7)]
        );
        assert_eq!(key.coupled(), SectorId::new(11));
        assert_eq!(key.codomain_is_dual(), &[false, true]);
        assert_eq!(key.domain_is_dual(), &[true, false]);
        assert_eq!(key.codomain_innerlines(), &[SectorId::new(13)]);
        assert_eq!(key.domain_innerlines(), &[SectorId::new(17)]);
        assert_eq!(
            key.codomain_vertices(),
            &[
                MultiplicityIndex::new(19).unwrap(),
                MultiplicityIndex::new(23).unwrap(),
            ]
        );
        assert_eq!(
            key.domain_vertices(),
            &[
                MultiplicityIndex::new(29).unwrap(),
                MultiplicityIndex::new(31).unwrap(),
            ]
        );

        let group = key.group_key();
        assert_eq!(group.codomain_uncoupled(), key.codomain_uncoupled());
        assert_eq!(group.domain_uncoupled(), key.domain_uncoupled());
        assert_eq!(group.codomain_is_dual(), key.codomain_is_dual());
        assert_eq!(group.domain_is_dual(), key.domain_is_dual());
    }

    #[test]
    fn sorted_lookup_distinguishes_rank1_tree_duality() {
        let nondual = FusionTreePairKey::try_pair_from_sector_ids(
            [0],
            [], 0,
            [false],
            [],
            [],
            [],
            [],
            [],
        ).unwrap();
        let dual = FusionTreePairKey::try_pair_from_sector_ids(
            [0],
            [], 0,
            [true],
            [],
            [],
            [],
            [],
            [],
        ).unwrap();
        let structure = SectorStructure::from_keys(1, [BlockKey::from(nondual)]).unwrap();

        assert!(!structure.has_compact_lookup());
        assert_eq!(structure.find_index(&BlockKey::from(dual.clone())), None);
        assert_eq!(structure.find_fusion_tree_pair_index(&dual), None);
    }

    fn materialized_leg_tuple_oracle(space: &FusionProductSpace) -> Vec<Vec<FusionTreeLeg>> {
        fn visit(
            legs: &[SectorLeg],
            remaining: usize,
            current: &mut [FusionTreeLeg],
            out: &mut Vec<Vec<FusionTreeLeg>>,
        ) {
            if remaining == 0 {
                out.push(current.to_vec());
                return;
            }
            let index = remaining - 1;
            for &sector in legs[index].sectors() {
                current[index] = FusionTreeLeg::new(sector, legs[index].is_dual());
                visit(legs, remaining - 1, current, out);
            }
        }

        let mut out = Vec::new();
        let mut current = vec![FusionTreeLeg::new(SectorId::new(0), false); space.len()];
        visit(space.legs(), space.len(), &mut current, &mut out);
        out
    }

    fn materialized_multiplicity_free_group_oracle<R>(
        rule: &R,
        space: &FusionProductSpace,
    ) -> Vec<CoupledFusionTrees>
    where
        R: MultiplicityFreeFusionRule,
    {
        let mut grouped = Vec::<CoupledFusionTrees>::new();
        let mut index: FxHashMap<SectorId, usize> = FxHashMap::default();
        for tuple in materialized_leg_tuple_oracle(space) {
            let uncoupled = tuple.iter().map(|leg| leg.sector()).collect::<Vec<_>>();
            let is_dual = tuple.iter().map(|leg| leg.is_dual()).collect::<Vec<_>>();
            let effective = uncoupled.clone();
            for coupled in reachable_coupled_sectors(rule, &effective) {
                let trees = collect_fusion_trees_for_coupled(
                    rule, &uncoupled, &is_dual, &effective, coupled,
                );
                match index.get(&coupled) {
                    Some(&i) => grouped[i].trees.extend(trees),
                    None => {
                        index.insert(coupled, grouped.len());
                        grouped.push(CoupledFusionTrees { coupled, trees });
                    }
                }
            }
        }
        grouped.sort_by_key(|group| group.coupled);
        grouped
    }

    fn materialized_multiplicity_free_key_oracle<R>(
        rule: &R,
        codomain: &FusionProductSpace,
        domain: &FusionProductSpace,
    ) -> Vec<FusionTreePairKey>
    where
        R: MultiplicityFreeFusionRule,
    {
        let codomain = materialized_multiplicity_free_group_oracle(rule, codomain);
        let domain = materialized_multiplicity_free_group_oracle(rule, domain);
        merge_generic_tree_groups(&codomain, &domain)
    }

    fn materialized_generic_fold_oracle<R>(
        rule: &R,
        space: &FusionProductSpace,
    ) -> CoupledSectorFold
    where
        R: FusionRule,
    {
        let mut aggregate = CoupledSectorFold::default();
        let mut clean_set = Vec::new();
        for tuple in materialized_leg_tuple_oracle(space) {
            let effective = tuple.iter().map(|leg| leg.sector()).collect::<Vec<_>>();
            let fold = rule.coupled_sector_fold(&effective);
            clean_set.extend(fold.clean);
            aggregate.tainted.extend(fold.tainted);
            aggregate.out_of_table.extend(fold.out_of_table);
            aggregate.poisoned |= fold.poisoned;
        }
        aggregate.tainted.sort_unstable();
        aggregate.tainted.dedup();
        aggregate.out_of_table.sort();
        aggregate.out_of_table.dedup();
        clean_set.sort_unstable();
        clean_set.dedup();
        clean_set.retain(|sector| !aggregate.tainted.contains(sector));
        aggregate.clean = clean_set;
        if aggregate.poisoned {
            let mut demoted = std::mem::take(&mut aggregate.clean);
            aggregate.tainted.append(&mut demoted);
            aggregate.tainted.sort_unstable();
            aggregate.tainted.dedup();
        }
        aggregate
    }

    #[test]
    fn fusion_tree_keys_match_materialized_cartesian_oracle_across_ranks_and_duals() {
        // What: public key enumeration keeps the old tuple order without
        // reusing the production visitor in the test oracle.
        let rule = Z4PointedRule;
        let leg = |sectors: &[usize], is_dual| {
            SectorLeg::new(
                sectors
                    .iter()
                    .copied()
                    .map(|sector| (SectorId::new(sector), 1usize)),
                is_dual,
            )
        };

        let cases = [
            (
                "empty",
                FusionProductSpace::new([]),
                FusionProductSpace::new([]),
            ),
            (
                "rank1",
                FusionProductSpace::new([leg(&[1, 3], true)]),
                FusionProductSpace::new([leg(&[1, 3], false)]),
            ),
            (
                "rank2",
                FusionProductSpace::new([leg(&[0, 1], false), leg(&[2, 3], true)]),
                FusionProductSpace::new([leg(&[1, 2, 3], true)]),
            ),
            (
                "rank8",
                FusionProductSpace::new(
                    (0..8).map(|axis| leg(&[axis % 4, (axis + 1) % 4], axis % 2 == 1)),
                ),
                FusionProductSpace::new([leg(&[0, 2], true)]),
            ),
        ];

        for (name, codomain, domain) in cases {
            let expected =
                materialized_multiplicity_free_key_oracle(&rule, &codomain, &domain);
            let hom = FusionTreeHomSpace::new(codomain, domain);
            assert_eq!(hom.fusion_tree_keys(&rule).as_ref(), expected.as_slice(), "{name}");
        }
    }

    #[test]
    fn generic_fusion_tree_keys_error_uses_materialized_codomain_fold_order() {
        // What: Generic full-space construction still reports the first
        // non-clean side using the old materialized tuple fold semantics.
        let rule = su3();
        let eight = su3_id(1, 1);
        let t27 = su3_id(2, 2);
        let codomain = FusionProductSpace::new([
            SectorLeg::new([(eight, 1usize), (t27, 1usize)], false),
            SectorLeg::new([(eight, 1usize)], false),
        ]);
        let domain = FusionProductSpace::new([
            SectorLeg::new([(t27, 1usize)], false),
            SectorLeg::new([(eight, 1usize)], false),
        ]);
        let expected =
            fusion_fold_error_message("codomain", &materialized_generic_fold_oracle(&rule, &codomain));

        let hom = FusionTreeHomSpace::new(codomain, domain);
        let message = hom.fusion_tree_keys_generic(&rule).unwrap_err().to_string();

        assert_eq!(message, expected);
    }

    #[test]
    fn selected_leg_tuple_visitor_is_fallible_and_restartable() {
        // What: an early visitor error stops traversal without corrupting the
        // reusable scratch used by a later traversal.
        let space = FusionProductSpace::new([
            SectorLeg::new([(SectorId::new(0), 1), (SectorId::new(1), 1)], false),
            SectorLeg::new([(SectorId::new(2), 1), (SectorId::new(3), 1)], true),
        ]);
        let expected = materialized_leg_tuple_oracle(&space)
            .into_iter()
            .map(|tuple| {
                tuple
                    .into_iter()
                    .map(|leg| (leg.sector(), leg.is_dual()))
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();

        let mut partial = Vec::new();
        let err = space
            .try_visit_selected_leg_tuples(&mut |tuple| {
                partial.push(
                    tuple
                        .iter()
                        .map(|leg| (leg.sector(), leg.is_dual()))
                        .collect::<Vec<_>>(),
                );
                if partial.len() == 2 {
                    Err("stop")
                } else {
                    Ok(())
                }
            })
            .unwrap_err();
        assert_eq!(err, "stop");
        assert_eq!(partial, expected[..2]);

        let mut restarted = Vec::new();
        space
            .try_visit_selected_leg_tuples::<(), _>(&mut |tuple| {
                restarted.push(
                    tuple
                        .iter()
                        .map(|leg| (leg.sector(), leg.is_dual()))
                        .collect::<Vec<_>>(),
                );
                Ok(())
            })
            .unwrap();
        assert_eq!(restarted, expected);
    }

    #[test]
    fn fusion_tree_homspace_generates_canonical_coupled_sector_order() {
        let rule = BranchingMultiplicityFreeRule;
        let hom = FusionTreeHomSpace::from_sector_ids([(1, 1), (1, 1)], [(1, 1), (1, 1)]);

        let keys = hom.fusion_tree_keys(&rule);

        assert_eq!(keys.len(), 2);
        assert_eq!(keys[0].coupled(), SectorId::new(0));
        assert_eq!(keys[1].coupled(), SectorId::new(2));
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
        assert_eq!(keys[0].codomain_vertices(), &[MultiplicityIndex::ONE]);
        assert_eq!(keys[0].domain_vertices(), &[MultiplicityIndex::ONE]);

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
    fn uncached_coupled_layout_probe_does_not_publish_identity_or_cache_state() {
        // What: the cold cost probe computes layout size/equality without
        // interning BlockStructure content or publishing a fusion-layout entry.
        let _guard = test_support::CACHE_TEST_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let rule = Z2FusionRule;
        let leg = || {
            SectorLeg::new(
                [(Z2Irrep::EVEN.sector_id(), 2), (Z2Irrep::ODD.sector_id(), 3)],
                false,
            )
        };
        let homspace = FusionTreeHomSpace::new(
            FusionProductSpace::new([leg(), leg()]),
            FusionProductSpace::new([leg(), leg()]),
        );
        let source = homspace
            .coupled_subblock_structure_from_leg_degeneracies(&rule)
            .unwrap();

        crate::reset_block_structure_intern_calls();
        crate::reset_fusion_tree_layout_probe_side_effect_calls();
        crate::reset_hom_space_intern_calls();
        let (required_len, source_matches) = homspace
            .coupled_subblock_layout_probe_uncached(&rule, source.as_ref())
            .unwrap();

        assert_eq!(required_len, source.required_len().unwrap());
        assert!(source_matches);
        assert_eq!(crate::block_structure_intern_calls(), 0);
        assert_eq!(
            crate::fusion_tree_layout_probe_side_effect_calls(),
            (0, 0)
        );
        assert_eq!(crate::hom_space_intern_calls(), 0);
        let cached_again = homspace
            .coupled_subblock_structure_from_leg_degeneracies(&rule)
            .unwrap();
        assert!(std::sync::Arc::ptr_eq(&source, &cached_again));
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
        assert_eq!(key.coupled(), SectorId::new(1));
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
        assert_eq!(key.coupled(), SectorId::new(1));
    }

    #[test]
    fn fusion_tree_pair_key_external_sectors_restore_visible_domain_sector() {
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

    fn legacy_select<R: FusionRule>(
        rule: &R,
        homspace: &FusionTreeHomSpace,
        codomain_axes: &[usize],
        domain_axes: &[usize],
    ) -> FusionTreeHomSpace {
        FusionTreeHomSpace::new(
            FusionProductSpace::new(
                codomain_axes
                    .iter()
                    .map(|&axis| homspace.external_axis_leg(rule, axis)),
            ),
            FusionProductSpace::new(domain_axes.iter().map(|&axis| {
                homspace.external_axis_leg(rule, axis).dual(rule)
            })),
        )
    }

    fn legacy_tensorcontract_homspace<R: FusionRule>(
        rule: &R,
        lhs: &FusionTreeHomSpace,
        rhs: &FusionTreeHomSpace,
        lhs_axes: &[usize],
        rhs_axes: &[usize],
        output_axes: &[usize],
        nout: usize,
    ) -> FusionTreeHomSpace {
        let lhs_open = (0..lhs.rank())
            .filter(|axis| !lhs_axes.contains(axis))
            .collect::<Vec<_>>();
        let rhs_open = (0..rhs.rank())
            .filter(|axis| !rhs_axes.contains(axis))
            .collect::<Vec<_>>();
        let lhs = legacy_select(rule, lhs, &lhs_open, lhs_axes);
        let rhs = legacy_select(rule, rhs, rhs_axes, &rhs_open);
        let composed = FusionTreeHomSpace::compose(rule, &lhs, &rhs).unwrap();
        legacy_select(
            rule,
            &composed,
            &output_axes[..nout],
            &output_axes[nout..],
        )
    }

    fn assert_direct_contract_matches_legacy<R: CheckedFusionAlgebra>(
        rule: &R,
        lhs: &FusionTreeHomSpace,
        rhs: &FusionTreeHomSpace,
        lhs_axes: &[usize],
        rhs_axes: &[usize],
        output_axes: &[usize],
        nout: usize,
    ) {
        let expected = legacy_tensorcontract_homspace(
            rule,
            lhs,
            rhs,
            lhs_axes,
            rhs_axes,
            output_axes,
            nout,
        );
        let actual = FusionTreeHomSpace::tensorcontract_homspace(
            rule,
            lhs,
            rhs,
            lhs_axes,
            rhs_axes,
            output_axes,
            nout,
        )
        .unwrap();
        assert_eq!(actual, expected);
        let checked = FusionTreeHomSpace::try_tensorcontract_homspace_checked(
            rule,
            lhs,
            rhs,
            lhs_axes,
            rhs_axes,
            output_axes,
            nout,
        )
        .unwrap();
        assert_eq!(checked, actual);
    }

    #[test]
    fn direct_homspace_derivation_matches_old_sequence_for_supported_rules() {
        let mixed_leg = |sectors: &[(SectorId, usize)], dual| {
            SectorLeg::new(sectors.iter().copied(), dual)
        };

        let u1_sectors = [(u1(-2), 1), (u1(1), 2)];
        let u1_hom = FusionTreeHomSpace::new(
            FusionProductSpace::new([
                mixed_leg(&u1_sectors, false),
                mixed_leg(&u1_sectors, true),
            ]),
            FusionProductSpace::new([
                mixed_leg(&u1_sectors, true),
                mixed_leg(&u1_sectors, false),
            ]),
        );
        assert_direct_contract_matches_legacy(
            &U1FusionRule,
            &u1_hom,
            &u1_hom,
            &[3, 2],
            &[0, 1],
            &[2, 0, 3, 1],
            2,
        );

        let parity = [(SectorId::new(0), 2), (SectorId::new(1), 1)];
        let fz2_hom = FusionTreeHomSpace::new(
            FusionProductSpace::new([
                mixed_leg(&parity, false),
                mixed_leg(&parity, true),
            ]),
            FusionProductSpace::new([
                mixed_leg(&parity, true),
                mixed_leg(&parity, false),
            ]),
        );
        assert_direct_contract_matches_legacy(
            &FermionParityFusionRule,
            &fz2_hom,
            &fz2_hom,
            &[3, 2],
            &[0, 1],
            &[1, 3, 0, 2],
            2,
        );

        let su2 = [
            (SU2Irrep::from_twice_spin(0).sector_id(), 2),
            (SU2Irrep::from_twice_spin(1).sector_id(), 1),
        ];
        let su2_hom = FusionTreeHomSpace::new(
            FusionProductSpace::new([
                mixed_leg(&su2, false),
                mixed_leg(&su2, true),
            ]),
            FusionProductSpace::new([
                mixed_leg(&su2, true),
                mixed_leg(&su2, false),
            ]),
        );
        assert_direct_contract_matches_legacy(
            &SU2FusionRule,
            &su2_hom,
            &su2_hom,
            &[3, 2],
            &[0, 1],
            &[3, 1, 2, 0],
            2,
        );

        type FpU1Rule = ProductFusionRule<FermionParityFusionRule, U1FusionRule>;
        type ProductRule = ProductFusionRule<FpU1Rule, SU2FusionRule>;
        let product_rule = ProductRule::new(
            FpU1Rule::new(FermionParityFusionRule, U1FusionRule),
            SU2FusionRule,
        );
        let encode = |parity, charge, spin| {
            let inner = TensorKitProductCodec::encode(
                SectorId::new(parity),
                U1Irrep::new(charge).sector_id(),
            );
            TensorKitProductCodec::encode(inner, SU2Irrep::from_twice_spin(spin).sector_id())
        };
        let product = [(encode(0, 0, 0), 2), (encode(1, 1, 1), 1)];
        let product_hom = FusionTreeHomSpace::new(
            FusionProductSpace::new([
                mixed_leg(&product, false),
                mixed_leg(&product, true),
            ]),
            FusionProductSpace::new([
                mixed_leg(&product, true),
                mixed_leg(&product, false),
            ]),
        );
        assert_direct_contract_matches_legacy(
            &product_rule,
            &product_hom,
            &product_hom,
            &[3, 2],
            &[0, 1],
            &[2, 1, 3, 0],
            2,
        );
    }

    fn assert_checked_contract_all_orientations<R>(
        rule: &R,
        matched: SectorLeg,
        mismatched: SectorLeg,
        open: SectorLeg,
    ) where
        R: CheckedFusionAlgebra,
    {
        let assert_same_result = |lhs: &FusionTreeHomSpace, rhs: &FusionTreeHomSpace| {
            let infallible =
                FusionTreeHomSpace::tensorcontract_homspace(rule, lhs, rhs, &[0], &[0], &[], 0);
            let checked = FusionTreeHomSpace::try_tensorcontract_homspace_checked(
                rule,
                lhs,
                rhs,
                &[0],
                &[0],
                &[],
                0,
            );
            match (infallible, checked) {
                (Ok(expected), Ok(actual)) => assert_eq!(actual, expected),
                (
                    Err(expected),
                    Err(CheckedFusionSpaceError::Core(actual)),
                ) => assert_eq!(*actual, expected),
                (_, Err(CheckedFusionSpaceError::FusionAlgebra(error))) => {
                    panic!("closed fixture unexpectedly failed checked algebra: {error}")
                }
                (expected, actual) => {
                    panic!("checked/infallible contraction results differ: {expected:?} vs {actual:?}")
                }
            }
        };

        for lhs_axis in 0..2 {
            for rhs_axis in 0..2 {
                let lhs_stored = if lhs_axis == 0 {
                    matched.dual(rule)
                } else {
                    matched.clone()
                };
                let rhs_stored = if rhs_axis == 0 {
                    matched.clone()
                } else {
                    matched.dual(rule)
                };
                let lhs = if lhs_axis == 0 {
                    FusionTreeHomSpace::new(
                        FusionProductSpace::new([lhs_stored]),
                        FusionProductSpace::new([open.clone()]),
                    )
                } else {
                    FusionTreeHomSpace::new(
                        FusionProductSpace::new([open.clone()]),
                        FusionProductSpace::new([lhs_stored]),
                    )
                };
                let rhs = if rhs_axis == 0 {
                    FusionTreeHomSpace::new(
                        FusionProductSpace::new([rhs_stored]),
                        FusionProductSpace::new([open.clone()]),
                    )
                } else {
                    FusionTreeHomSpace::new(
                        FusionProductSpace::new([open.clone()]),
                        FusionProductSpace::new([rhs_stored]),
                    )
                };
                assert_direct_contract_matches_legacy(
                    rule,
                    &lhs,
                    &rhs,
                    &[lhs_axis],
                    &[rhs_axis],
                    &[1, 0],
                    1,
                );

                let bad_rhs_stored = if rhs_axis == 0 {
                    mismatched.clone()
                } else {
                    mismatched.dual(rule)
                };
                let bad_rhs = if rhs_axis == 0 {
                    FusionTreeHomSpace::new(
                        FusionProductSpace::new([bad_rhs_stored]),
                        FusionProductSpace::new([open.clone()]),
                    )
                } else {
                    FusionTreeHomSpace::new(
                        FusionProductSpace::new([open.clone()]),
                        FusionProductSpace::new([bad_rhs_stored]),
                    )
                };
                let lhs_contract = if lhs_axis == 0 { 0 } else { 1 };
                let rhs_contract = if rhs_axis == 0 { 0 } else { 1 };
                let infallible = FusionTreeHomSpace::tensorcontract_homspace(
                    rule,
                    &lhs,
                    &bad_rhs,
                    &[lhs_contract],
                    &[rhs_contract],
                    &[1, 0],
                    1,
                );
                let checked = FusionTreeHomSpace::try_tensorcontract_homspace_checked(
                    rule,
                    &lhs,
                    &bad_rhs,
                    &[lhs_contract],
                    &[rhs_contract],
                    &[1, 0],
                    1,
                );
                match (infallible, checked) {
                    (Err(expected), Err(CheckedFusionSpaceError::Core(actual))) => {
                        assert_eq!(*actual, expected)
                    }
                    (expected, actual) => panic!(
                        "checked/infallible mismatch error differs: {expected:?} vs {actual:?}"
                    ),
                }
            }
        }

        // Exercise the direct codomain/domain form in addition to the four
        // stored-side combinations above.
        let direct_lhs = FusionTreeHomSpace::new(
            FusionProductSpace::new([matched.clone()]),
            FusionProductSpace::new([]),
        );
        let direct_rhs = FusionTreeHomSpace::new(
            FusionProductSpace::new([]),
            FusionProductSpace::new([matched]),
        );
        assert_same_result(&direct_lhs, &direct_rhs);
    }

    #[test]
    fn checked_tensorcontract_matches_all_closed_rules_and_leg_orientations() {
        // What: checked contraction preserves valid HomSpaces and structural
        // errors across all four stored-side orientations and multi-sector
        // membership for every built-in multiplicity-free family.
        let fixture = |sectors: &[(SectorId, usize)], mismatch: &[(SectorId, usize)]| {
            (
                SectorLeg::new(sectors.iter().copied(), false),
                SectorLeg::new(mismatch.iter().copied(), false),
            )
        };
        let (z2, z2_bad) = fixture(
            &[(z2_even(), 1), (z2_odd(), 2)],
            &[(z2_even(), 1), (z2_odd(), 3)],
        );
        assert_checked_contract_all_orientations(
            &Z2FusionRule,
            z2,
            z2_bad,
            SectorLeg::new([(z2_even(), 1)], false),
        );
        let (fz2, fz2_bad) = fixture(
            &[(z2_even(), 2), (z2_odd(), 1)],
            &[(z2_even(), 3), (z2_odd(), 1)],
        );
        assert_checked_contract_all_orientations(
            &FermionParityFusionRule,
            fz2,
            fz2_bad,
            SectorLeg::new([(z2_even(), 1)], false),
        );
        let (u1_leg, u1_bad) = fixture(
            &[(u1(-2), 1), (u1(1), 2)],
            &[(u1(-2), 1), (u1(1), 3)],
        );
        assert_checked_contract_all_orientations(
            &U1FusionRule,
            u1_leg,
            u1_bad,
            SectorLeg::new([(u1(0), 1)], false),
        );
        let spin0 = su2(0);
        let spin_half = su2(1);
        let (su2_leg, su2_bad) = fixture(
            &[(spin0, 1), (spin_half, 2)],
            &[(spin0, 1), (spin_half, 3)],
        );
        assert_checked_contract_all_orientations(
            &SU2FusionRule,
            su2_leg,
            su2_bad,
            SectorLeg::new([(spin0, 1)], false),
        );
        let (fibonacci, fibonacci_bad) = fixture(
            &[(SectorId::new(0), 1), (SectorId::new(1), 2)],
            &[(SectorId::new(0), 1), (SectorId::new(1), 3)],
        );
        assert_checked_contract_all_orientations(
            &FibonacciFusionRule,
            fibonacci,
            fibonacci_bad,
            SectorLeg::new([(SectorId::new(0), 1)], false),
        );

        #[cfg(target_pointer_width = "64")]
        {
            type Rule = ProductFusionRule<U1FusionRule, Z2FusionRule, TensorKitProductCodec>;
            let rule = Rule::new(U1FusionRule, Z2FusionRule);
            let first = TensorKitProductCodec::encode(u1(-2), z2_even());
            let second = TensorKitProductCodec::encode(u1(1), z2_odd());
            let vacuum = TensorKitProductCodec::encode(u1(0), z2_even());
            let (product, product_bad) =
                fixture(&[(first, 1), (second, 2)], &[(first, 1), (second, 3)]);
            assert_checked_contract_all_orientations(
                &rule,
                product,
                product_bad,
                SectorLeg::new([(vacuum, 1)], false),
            );
        }
    }

    #[derive(Clone)]
    struct DualCountingRule<R> {
        inner: R,
        dual_calls: Arc<AtomicUsize>,
    }

    impl<R> DualCountingRule<R>
    where
        R: FusionRule,
    {
        fn new(inner: R) -> Self {
            Self {
                inner,
                dual_calls: Arc::new(AtomicUsize::new(0)),
            }
        }

        fn reset_dual_calls(&self) {
            self.dual_calls.store(0, Ordering::Relaxed);
        }

        fn dual_calls(&self) -> usize {
            self.dual_calls.load(Ordering::Relaxed)
        }
    }

    impl<R> FusionRule for DualCountingRule<R>
    where
        R: FusionRule,
    {
        fn rule_identity(&self) -> RuleIdentity {
            self.inner.rule_identity()
        }

        fn fusion_style(&self) -> FusionStyleKind {
            self.inner.fusion_style()
        }

        fn braiding_style(&self) -> BraidingStyleKind {
            self.inner.braiding_style()
        }

        fn vacuum(&self) -> SectorId {
            self.inner.vacuum()
        }

        fn dual(&self, sector: SectorId) -> SectorId {
            self.dual_calls.fetch_add(1, Ordering::Relaxed);
            self.inner.dual(sector)
        }

        fn fusion_channels(&self, left: SectorId, right: SectorId) -> SectorVec {
            self.inner.fusion_channels(left, right)
        }
    }

    impl<R> CheckedFusionAlgebra for DualCountingRule<R>
    where
        R: CheckedFusionAlgebra,
    {
        fn try_dual_sector(&self, sector: SectorId) -> Result<SectorId, FusionAlgebraError> {
            self.dual_calls.fetch_add(1, Ordering::Relaxed);
            self.inner.try_dual_sector(sector)
        }

        fn try_fusion_channels(
            &self,
            left: SectorId,
            right: SectorId,
        ) -> Result<SectorVec, FusionAlgebraError> {
            self.inner.try_fusion_channels(left, right)
        }

        fn try_nsymbol(
            &self,
            left: SectorId,
            right: SectorId,
            coupled: SectorId,
        ) -> Result<usize, FusionAlgebraError> {
            self.inner.try_nsymbol(left, right, coupled)
        }
    }

    fn assert_oriented_validation_dual_calls_are_linear<R>(
        rule: &DualCountingRule<R>,
        sectors: impl IntoIterator<Item = (SectorId, usize)>,
        sector_count: usize,
    ) where
        R: FusionRule,
    {
        let leg = SectorLeg::new(sectors, false);
        let lhs = OrientedLegView::borrowed(&leg).toggled();
        let rhs = OrientedLegView::borrowed(&leg).toggled();
        rule.reset_dual_calls();
        validate_oriented_composed_leg(rule, lhs, rhs).unwrap();
        assert_eq!(rule.dual_calls(), 2 * sector_count);
    }

    #[test]
    fn oriented_composed_leg_membership_is_linear_in_dual_operations() {
        const SECTORS: usize = 256;
        let u1_rule = DualCountingRule::new(U1FusionRule);
        assert_oriented_validation_dual_calls_are_linear(
            &u1_rule,
            (0..SECTORS).map(|index| (u1(index as i32 - 97), index % 5 + 1)),
            SECTORS,
        );

        let product_rule = DualCountingRule::new(ProductFusionRule::<
            U1FusionRule,
            FermionParityFusionRule,
            TensorKitProductCodec,
        >::new(U1FusionRule, FermionParityFusionRule));
        assert_oriented_validation_dual_calls_are_linear(
            &product_rule,
            (0..SECTORS).map(|index| {
                (
                    TensorKitProductCodec::encode(
                        u1(index as i32 - 113),
                        SectorId::new(index % 2),
                    ),
                    index % 7 + 1,
                )
            }),
            SECTORS,
        );
    }

    #[test]
    fn oriented_composed_leg_invalid_path_preserves_legacy_error_order() {
        let rule = U1FusionRule;
        let lhs = SectorLeg::new([(u1(-3), 1), (u1(2), 2)], false);
        let rhs = SectorLeg::new([(u1(-3), 4), (u1(2), 2)], false);
        let lhs_view = OrientedLegView::borrowed(&lhs).toggled();
        let rhs_view = OrientedLegView::borrowed(&rhs).toggled();
        let expected =
            validate_composed_leg(&lhs_view.materialize(&rule), &rhs_view.materialize(&rule));
        assert_eq!(
            validate_oriented_composed_leg(&rule, lhs_view, rhs_view),
            expected
        );
    }

    #[test]
    fn generic_su3_select_matches_old_sequence() {
        let rule = su3();
        let leg = SectorLeg::new([(su3_id(0, 0), 2), (su3_id(1, 0), 1)], false);
        let hom = FusionTreeHomSpace::new(
            FusionProductSpace::new([leg.clone(), leg.clone()]),
            FusionProductSpace::new([leg.clone(), leg]),
        );
        let expected = legacy_select(&rule, &hom, &[3, 0], &[2, 1]);
        let actual = hom.select(&rule, &[3, 0], &[2, 1]).unwrap();
        assert_eq!(actual, expected);
    }

    fn legacy_leg_degeneracy_structure<R>(
        rule: &R,
        homspace: &FusionTreeHomSpace,
    ) -> Arc<BlockStructure>
    where
        R: MultiplicityFreeFusionRule,
    {
        let keys = homspace.fusion_tree_keys(rule);
        let blocks = keys
            .iter()
            .map(|key| {
                (
                    key.clone(),
                    homspace.degeneracy_shape_for_key(key).unwrap().to_vec(),
                )
            })
            .collect();
        BlockStructure::coupled_sector_matrix_with_keys(
            rule,
            homspace.codomain().len(),
            homspace.rank(),
            blocks,
        )
        .unwrap()
        .into_shared()
    }

    fn assert_direct_leg_degeneracy_structure_matches_legacy<R>(
        rule: &R,
        homspace: &FusionTreeHomSpace,
    ) where
        R: MultiplicityFreeFusionRule,
    {
        let expected = legacy_leg_degeneracy_structure(rule, homspace);
        let actual = homspace
            .coupled_subblock_structure_from_leg_degeneracies(rule)
            .unwrap();
        assert_eq!(actual, expected);
        assert_eq!(actual.content_id(), expected.content_id());
        assert_eq!(actual.required_len().unwrap(), expected.required_len().unwrap());
    }

    fn assert_coupled_grid_layout_matches_key_reconstruction<R>(
        rule: &R,
        homspace: &FusionTreeHomSpace,
    ) where
        R: MultiplicityFreeFusionRule,
    {
        let reconstructed =
            reconstructed_fusion_tree_layout_data_from_keys(homspace.fusion_tree_keys_uncached(rule));
        let direct = homspace.fusion_tree_layout_data_uncached(rule);

        assert_eq!(direct.keys, reconstructed.keys);
        assert_eq!(direct.sectors.len(), reconstructed.sectors.len());
        for (actual, expected) in direct.sectors.iter().zip(&reconstructed.sectors) {
            assert_eq!(actual.start, expected.start);
            assert_eq!(actual.row_count, expected.row_count);
            assert_eq!(actual.col_count, expected.col_count);
            assert_eq!(
                expected.row_key_offsets,
                (0..actual.row_count).collect::<Vec<_>>()
            );
            assert_eq!(
                expected.col_key_offsets,
                (0..actual.col_count)
                    .map(|col| col * actual.row_count)
                    .collect::<Vec<_>>()
            );
            assert_eq!(
                expected.entries,
                (0..actual.col_count)
                    .flat_map(|col| {
                        (0..actual.row_count)
                            .map(move |row| FusionTreeBlockLayoutEntry { row, col })
                    })
                    .collect::<Vec<_>>()
            );
        }

        let direct_parts =
            coupled_subblock_parts_from_leg_degeneracies(homspace, &direct).unwrap();
        let reconstructed_parts = legacy_leg_degeneracy_structure(rule, homspace);
        assert_eq!(direct_parts.0, *reconstructed_parts.sector_structure());
        assert_eq!(direct_parts.1, *reconstructed_parts.degeneracy_structure());
        assert_eq!(
            direct_parts.1.required_len().unwrap(),
            reconstructed_parts.required_len().unwrap()
        );
    }

    #[test]
    fn direct_leg_degeneracy_layout_matches_legacy_for_supported_rules() {
        let _guard = test_support::CACHE_TEST_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let mixed_leg = |sectors: &[(SectorId, usize)], dual| {
            SectorLeg::new(sectors.iter().copied(), dual)
        };
        let build = |sectors: &[(SectorId, usize)]| {
            FusionTreeHomSpace::new(
                FusionProductSpace::new([
                    mixed_leg(sectors, false),
                    mixed_leg(sectors, true),
                ]),
                FusionProductSpace::new([
                    mixed_leg(sectors, true),
                    mixed_leg(sectors, false),
                ]),
            )
        };

        let u1_hom = build(&[(u1(-2), 2), (u1(1), 1)]);
        assert_direct_leg_degeneracy_structure_matches_legacy(&U1FusionRule, &u1_hom);
        assert_coupled_grid_layout_matches_key_reconstruction(&U1FusionRule, &u1_hom);

        let parity_hom = build(&[(SectorId::new(0), 3), (SectorId::new(1), 2)]);
        assert_direct_leg_degeneracy_structure_matches_legacy(
            &FermionParityFusionRule,
            &parity_hom,
        );
        assert_coupled_grid_layout_matches_key_reconstruction(
            &FermionParityFusionRule,
            &parity_hom,
        );

        let su2_hom = build(&[
            (SU2Irrep::from_twice_spin(0).sector_id(), 2),
            (SU2Irrep::from_twice_spin(1).sector_id(), 1),
        ]);
        assert_direct_leg_degeneracy_structure_matches_legacy(&SU2FusionRule, &su2_hom);
        assert_coupled_grid_layout_matches_key_reconstruction(&SU2FusionRule, &su2_hom);

        let product_rule = product_fusion_rule(FermionParityFusionRule, U1FusionRule);
        let product_hom = build(&[
            (product_rule.encode_sector(SectorId::new(0), u1(-1)), 2),
            (product_rule.encode_sector(SectorId::new(1), u1(2)), 1),
        ]);
        assert_direct_leg_degeneracy_structure_matches_legacy(&product_rule, &product_hom);
        assert_coupled_grid_layout_matches_key_reconstruction(&product_rule, &product_hom);

        let scalar =
            FusionTreeHomSpace::new(FusionProductSpace::new([]), FusionProductSpace::new([]));
        assert_direct_leg_degeneracy_structure_matches_legacy(&U1FusionRule, &scalar);
        assert_coupled_grid_layout_matches_key_reconstruction(&U1FusionRule, &scalar);

        let rank_one = FusionTreeHomSpace::new(
            FusionProductSpace::new([SectorLeg::new([(u1(0), 5)], true)]),
            FusionProductSpace::new(Vec::<SectorLeg>::new()),
        );
        assert_direct_leg_degeneracy_structure_matches_legacy(&U1FusionRule, &rank_one);
        assert_coupled_grid_layout_matches_key_reconstruction(&U1FusionRule, &rank_one);
        let rank_one_block = rank_one
            .coupled_subblock_structure_from_leg_degeneracies(&U1FusionRule)
            .unwrap();
        assert_eq!(rank_one_block.required_len().unwrap(), 5);
        assert_eq!(rank_one_block.block(0).unwrap().shape(), &[5]);
        assert_eq!(rank_one_block.block(0).unwrap().strides(), &[1]);
        assert_eq!(rank_one_block.block(0).unwrap().offset(), 0);

        let rank_one_domain = FusionTreeHomSpace::new(
            FusionProductSpace::new(Vec::<SectorLeg>::new()),
            FusionProductSpace::new([SectorLeg::new([(u1(0), 7)], true)]),
        );
        let rank_one_domain_block = rank_one_domain
            .coupled_subblock_structure_from_leg_degeneracies(&U1FusionRule)
            .unwrap();
        assert_eq!(rank_one_domain_block.required_len().unwrap(), 7);
        assert_eq!(rank_one_domain_block.block(0).unwrap().shape(), &[7]);
        assert_eq!(rank_one_domain_block.block(0).unwrap().strides(), &[1]);
        assert_eq!(rank_one_domain_block.block(0).unwrap().offset(), 0);

        let empty_domain = FusionTreeHomSpace::new(
            FusionProductSpace::new([
                SectorLeg::new([(u1(-1), 2), (u1(0), 1)], false),
                SectorLeg::new([(u1(0), 3), (u1(1), 1)], false),
            ]),
            FusionProductSpace::new(Vec::<SectorLeg>::new()),
        );
        assert_direct_leg_degeneracy_structure_matches_legacy(&U1FusionRule, &empty_domain);
        assert_coupled_grid_layout_matches_key_reconstruction(&U1FusionRule, &empty_domain);
    }

    #[test]
    fn canonical_coupled_grid_matches_literal_u1_sector_matrices() {
        let homspace = FusionTreeHomSpace::new(
            FusionProductSpace::new([
                SectorLeg::new([(u1(0), 2), (u1(1), 3)], false),
                SectorLeg::new([(u1(0), 5), (u1(1), 7)], true),
            ]),
            FusionProductSpace::new([
                SectorLeg::new([(u1(0), 11), (u1(1), 13)], true),
                SectorLeg::new([(u1(0), 17), (u1(1), 19)], false),
            ]),
        );

        let structure = homspace
            .coupled_subblock_structure_from_leg_degeneracies(&U1FusionRule)
            .unwrap();
        let expected = [
            (
                [u1(0), u1(0)],
                [u1(0), u1(0)],
                u1(0),
                [2, 5, 11, 17],
                [1, 2, 10, 110],
                0,
            ),
            (
                [u1(1), u1(0)],
                [u1(1), u1(0)],
                u1(1),
                [3, 5, 13, 17],
                [1, 3, 29, 377],
                1870,
            ),
            (
                [u1(0), u1(1)],
                [u1(1), u1(0)],
                u1(1),
                [2, 7, 13, 17],
                [1, 2, 29, 377],
                1885,
            ),
            (
                [u1(1), u1(0)],
                [u1(0), u1(1)],
                u1(1),
                [3, 5, 11, 19],
                [1, 3, 29, 319],
                8279,
            ),
            (
                [u1(0), u1(1)],
                [u1(0), u1(1)],
                u1(1),
                [2, 7, 11, 19],
                [1, 2, 29, 319],
                8294,
            ),
            (
                [u1(1), u1(1)],
                [u1(1), u1(1)],
                u1(2),
                [3, 7, 13, 19],
                [1, 3, 21, 273],
                14340,
            ),
        ];

        assert_eq!(structure.block_count(), expected.len());
        for (index, (codomain, domain, coupled, shape, strides, offset)) in
            expected.iter().enumerate()
        {
            let block = structure.block(index).unwrap();
            let BlockKey::FusionTree(key) = block.key() else {
                panic!("canonical symmetric block must use a fusion-tree key");
            };
            assert_eq!(key.codomain_uncoupled(), codomain);
            assert_eq!(key.domain_uncoupled(), domain);
            assert_eq!(key.codomain_is_dual(), &[false, true]);
            assert_eq!(key.domain_is_dual(), &[true, false]);
            assert_eq!(key.coupled(), *coupled);
            assert_eq!(block.shape(), shape);
            assert_eq!(block.strides(), strides);
            assert_eq!(block.offset(), *offset);
        }
        assert_eq!(structure.required_len().unwrap(), 19527);

        let regions = structure.coupled_sector_regions(2).unwrap().unwrap();
        assert_eq!(regions.len(), 3);
        assert_eq!(
            regions
                .iter()
                .map(|region| (
                    region.coupled(),
                    region.rows(),
                    region.cols(),
                    region.range(),
                ))
                .collect::<Vec<_>>(),
            vec![
                (u1(0), 10, 187, 0..1870),
                (u1(1), 29, 430, 1870..14340),
                (u1(2), 21, 247, 14340..19527),
            ]
        );
    }

    fn assert_literal_binary_choice_sector_grids<R>(
        rule: &R,
        vacuum: SectorId,
        external: SectorId,
        mut groups: Vec<(SectorId, Vec<[SectorId; 2]>)>,
    ) where
        R: MultiplicityFreeFusionRule,
    {
        let homspace = FusionTreeHomSpace::new(
            FusionProductSpace::new([
                SectorLeg::new([(vacuum, 2), (external, 3)], false),
                SectorLeg::new([(vacuum, 5), (external, 7)], true),
            ]),
            FusionProductSpace::new([
                SectorLeg::new([(vacuum, 11), (external, 13)], true),
                SectorLeg::new([(vacuum, 17), (external, 19)], false),
            ]),
        );

        groups.sort_by_key(|(coupled, _)| *coupled);
        let layout = homspace.fusion_tree_layout_data_uncached(rule);
        assert_eq!(layout.sectors.len(), groups.len());
        let expected_key_count = groups
            .iter()
            .map(|(_, trees)| trees.len() * trees.len())
            .sum::<usize>();
        assert_eq!(layout.keys.len(), expected_key_count);

        let structure = homspace
            .coupled_subblock_structure_from_leg_degeneracies(rule)
            .unwrap();
        let regions = structure.coupled_sector_regions(2).unwrap().unwrap();
        assert_eq!(regions.len(), groups.len());
        let mut block_index = 0usize;
        let mut sector_offset = 0usize;
        for (sector_index, ((expected_coupled, trees), sector)) in
            groups.iter().zip(&layout.sectors).enumerate()
        {
            assert_eq!(
                (sector.start, sector.row_count, sector.col_count),
                (block_index, trees.len(), trees.len())
            );

            let row_shapes = trees
                .iter()
                .map(|tree| {
                    [
                        if tree[0] == vacuum { 2 } else { 3 },
                        if tree[1] == vacuum { 5 } else { 7 },
                    ]
                })
                .collect::<Vec<_>>();
            let col_shapes = trees
                .iter()
                .map(|tree| {
                    [
                        if tree[0] == vacuum { 11 } else { 13 },
                        if tree[1] == vacuum { 17 } else { 19 },
                    ]
                })
                .collect::<Vec<_>>();
            let row_dims = row_shapes
                .iter()
                .map(|shape| shape[0] * shape[1])
                .collect::<Vec<_>>();
            let col_dims = col_shapes
                .iter()
                .map(|shape| shape[0] * shape[1])
                .collect::<Vec<_>>();
            let matrix_rows = row_dims.iter().sum::<usize>();
            let matrix_cols = col_dims.iter().sum::<usize>();
            let mut col_offset = 0usize;
            for (col, domain_tree) in trees.iter().enumerate() {
                let mut row_offset = 0usize;
                for (row, codomain_tree) in trees.iter().enumerate() {
                    let key = &layout.keys[block_index];
                    assert_eq!(key.codomain_uncoupled(), codomain_tree);
                    assert_eq!(key.domain_uncoupled(), domain_tree);
                    assert_eq!(key.codomain_is_dual(), &[false, true]);
                    assert_eq!(key.domain_is_dual(), &[true, false]);
                    assert_eq!(key.coupled(), *expected_coupled);

                    let block = structure.block(block_index).unwrap();
                    assert_eq!(
                        block.shape(),
                        &[
                            row_shapes[row][0],
                            row_shapes[row][1],
                            col_shapes[col][0],
                            col_shapes[col][1],
                        ]
                    );
                    assert_eq!(
                        block.strides(),
                        &[
                            1,
                            row_shapes[row][0],
                            matrix_rows,
                            matrix_rows * col_shapes[col][0],
                        ]
                    );
                    assert_eq!(
                        block.offset(),
                        sector_offset + row_offset + matrix_rows * col_offset
                    );
                    row_offset += row_dims[row];
                    block_index += 1;
                }
                col_offset += col_dims[col];
            }

            let sector_end = sector_offset + matrix_rows * matrix_cols;
            assert_eq!(regions[sector_index].coupled(), *expected_coupled);
            assert_eq!(regions[sector_index].rows(), matrix_rows);
            assert_eq!(regions[sector_index].cols(), matrix_cols);
            assert_eq!(regions[sector_index].range(), sector_offset..sector_end);
            sector_offset = sector_end;
        }
        assert_eq!(block_index, expected_key_count);
        assert_eq!(structure.required_len().unwrap(), sector_offset);
    }

    #[test]
    fn canonical_coupled_grid_has_literal_nonabelian_and_product_metadata() {
        assert_literal_binary_choice_sector_grids(
            &FermionParityFusionRule,
            z2_even(),
            z2_odd(),
            vec![
                (z2_even(), vec![[z2_even(), z2_even()], [z2_odd(), z2_odd()]]),
                (z2_odd(), vec![[z2_odd(), z2_even()], [z2_even(), z2_odd()]]),
            ],
        );
        assert_literal_binary_choice_sector_grids(
            &SU2FusionRule,
            su2(0),
            su2(1),
            vec![
                (su2(0), vec![[su2(0), su2(0)], [su2(1), su2(1)]]),
                (su2(1), vec![[su2(1), su2(0)], [su2(0), su2(1)]]),
                (su2(2), vec![[su2(1), su2(1)]]),
            ],
        );

        type Fz2U1Codec = PackedProductCodec<Fz2SectorLayout, U1SectorLayout>;
        type Fz2U1Layout = ProductSectorLayout<Fz2SectorLayout, U1SectorLayout>;
        type Fz2U1Rule =
            ProductFusionRule<FermionParityFusionRule, U1FusionRule, Fz2U1Codec>;
        type TripleCodec = PackedProductCodec<Fz2U1Layout, Su2SectorLayout>;
        type TripleRule = ProductFusionRule<Fz2U1Rule, SU2FusionRule, TripleCodec>;

        let triple_rule = TripleRule::new(
            Fz2U1Rule::new(FermionParityFusionRule, U1FusionRule),
            SU2FusionRule,
        );
        let vacuum =
            TripleCodec::encode(Fz2U1Codec::encode(z2_even(), u1(0)), su2(0));
        let external =
            TripleCodec::encode(Fz2U1Codec::encode(z2_odd(), u1(0)), su2(1));
        let second_channel =
            TripleCodec::encode(Fz2U1Codec::encode(z2_even(), u1(0)), su2(2));
        assert_literal_binary_choice_sector_grids(
            &triple_rule,
            vacuum,
            external,
            vec![
                (vacuum, vec![[vacuum, vacuum], [external, external]]),
                (
                    external,
                    vec![[external, vacuum], [vacuum, external]],
                ),
                (second_channel, vec![[external, external]]),
            ],
        );
    }

    #[test]
    fn canonical_su2_innerline_grid_matches_literal_sector_matrices() {
        let half = su2(1);
        let homspace = FusionTreeHomSpace::new(
            FusionProductSpace::new([
                SectorLeg::new([(half, 2)], false),
                SectorLeg::new([(half, 3)], true),
                SectorLeg::new([(half, 5)], false),
            ]),
            FusionProductSpace::new([
                SectorLeg::new([(half, 7)], true),
                SectorLeg::new([(half, 11)], false),
                SectorLeg::new([(half, 13)], true),
            ]),
        );

        let layout = homspace.fusion_tree_layout_data_uncached(&SU2FusionRule);
        assert_eq!(layout.keys.len(), 5);
        assert_eq!(layout.sectors.len(), 2);
        assert_eq!(
            (
                layout.sectors[0].start,
                layout.sectors[0].row_count,
                layout.sectors[0].col_count,
            ),
            (0, 2, 2)
        );
        assert_eq!(
            (
                layout.sectors[1].start,
                layout.sectors[1].row_count,
                layout.sectors[1].col_count,
            ),
            (4, 1, 1)
        );
        let expected_innerlines = [
            (su2(0), su2(0), su2(1)),
            (su2(2), su2(0), su2(1)),
            (su2(0), su2(2), su2(1)),
            (su2(2), su2(2), su2(1)),
            (su2(2), su2(2), su2(3)),
        ];
        for (key, &(codomain_inner, domain_inner, coupled)) in
            layout.keys.iter().zip(&expected_innerlines)
        {
            assert_eq!(key.codomain_uncoupled(), &[half, half, half]);
            assert_eq!(key.domain_uncoupled(), &[half, half, half]);
            assert_eq!(key.codomain_is_dual(), &[false, true, false]);
            assert_eq!(key.domain_is_dual(), &[true, false, true]);
            assert_eq!(key.codomain_innerlines(), &[codomain_inner]);
            assert_eq!(key.domain_innerlines(), &[domain_inner]);
            assert_eq!(key.coupled(), coupled);
        }

        let structure = homspace
            .coupled_subblock_structure_from_leg_degeneracies(&SU2FusionRule)
            .unwrap();
        let expected_offsets = [0, 30, 60060, 60090, 120120];
        for (index, &offset) in expected_offsets.iter().enumerate() {
            let block = structure.block(index).unwrap();
            assert_eq!(block.shape(), &[2, 3, 5, 7, 11, 13]);
            if index < 4 {
                assert_eq!(block.strides(), &[1, 2, 6, 60, 420, 4620]);
            } else {
                assert_eq!(block.strides(), &[1, 2, 6, 30, 210, 2310]);
            }
            assert_eq!(block.offset(), offset);
        }
        assert_eq!(structure.required_len().unwrap(), 150150);
        let regions = structure.coupled_sector_regions(3).unwrap().unwrap();
        assert_eq!(regions.len(), 2);
        assert_eq!(
            (
                regions[0].coupled(),
                regions[0].rows(),
                regions[0].cols(),
                regions[0].range(),
            ),
            (su2(1), 60, 2002, 0..120120)
        );
        assert_eq!(
            (
                regions[1].coupled(),
                regions[1].rows(),
                regions[1].cols(),
                regions[1].range(),
            ),
            (su2(3), 30, 1001, 120120..150150)
        );
    }

    #[test]
    fn explicit_shape_coupled_grid_reports_extent_overflow() {
        let homspace = FusionTreeHomSpace::new(
            FusionProductSpace::new([
                SectorLeg::new([(u1(0), usize::MAX)], false),
                SectorLeg::new([(u1(0), 2)], false),
            ]),
            FusionProductSpace::new(Vec::<SectorLeg>::new()),
        );

        let error = homspace
            .coupled_subblock_structure(&U1FusionRule, 2, [[usize::MAX, 2]])
            .unwrap_err();
        assert_eq!(error, CoreError::ElementCountOverflow);
    }

    fn assert_direct_generic_leg_degeneracy_structure_matches_legacy<R>(
        rule: &R,
        homspace: &FusionTreeHomSpace,
    ) where
        R: FusionRule,
    {
        let keys = homspace.fusion_tree_keys_generic(rule).unwrap();
        let blocks = keys
            .iter()
            .map(|key| {
                (
                    key.clone(),
                    homspace.degeneracy_shape_for_key(key).unwrap().to_vec(),
                )
            })
            .collect();
        let expected = BlockStructure::coupled_sector_matrix_with_keys(
            rule,
            homspace.codomain().len(),
            homspace.rank(),
            blocks,
        )
        .unwrap()
        .into_shared();
        let actual = homspace
            .coupled_subblock_structure_from_leg_degeneracies_generic(rule)
            .unwrap();
        assert_eq!(actual, expected);
        assert_eq!(actual.content_id(), expected.content_id());
    }

    #[test]
    fn direct_generic_leg_degeneracy_layout_matches_legacy() {
        let _guard = test_support::CACHE_TEST_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let rule = UnitaryToyOmRule;
        let a = SectorId::new(UnitaryToyOmRule::A);
        let c = SectorId::new(UnitaryToyOmRule::C);
        let homspace = FusionTreeHomSpace::new(
            FusionProductSpace::new([
                SectorLeg::new([(a, 2)], false),
                SectorLeg::new([(a, 2)], false),
            ]),
            FusionProductSpace::new([SectorLeg::new([(c, 3)], false)]),
        );
        assert_direct_generic_leg_degeneracy_structure_matches_legacy(&rule, &homspace);

        let su3_rule = su3();
        let eight = su3_id(1, 1);
        let su3_homspace = FusionTreeHomSpace::new(
            FusionProductSpace::new([SectorLeg::new([(eight, 2)], false)]),
            FusionProductSpace::new([SectorLeg::new([(eight, 3)], false)]),
        );
        assert_direct_generic_leg_degeneracy_structure_matches_legacy(&su3_rule, &su3_homspace);
    }

    #[test]
    fn canonical_coupled_grid_derives_each_row_and_column_once() {
        let rule = U1FusionRule;
        let leg = SectorLeg::new([(u1(-1), 2), (u1(0), 3), (u1(2), 1)], false);
        let homspace = FusionTreeHomSpace::new(
            FusionProductSpace::new([leg.clone(), leg.clone()]),
            FusionProductSpace::new([leg.clone(), leg]),
        );
        reset_coupled_grid_build_observations();
        let layout = homspace.fusion_tree_layout_data_uncached(&rule);
        let expected_derivations = layout
            .sectors
            .iter()
            .map(|sector| sector.row_count + sector.col_count)
            .sum::<usize>();
        let actual =
            coupled_subblock_parts_from_leg_degeneracies(&homspace, &layout).unwrap();
        let expected = legacy_leg_degeneracy_structure(&rule, &homspace);
        assert_eq!(actual.0, *expected.sector_structure());
        assert_eq!(actual.1, *expected.degeneracy_structure());
        assert_eq!(
            coupled_grid_build_observations(),
            (0, expected_derivations)
        );
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
        assert_eq!(keys[0].coupled(), SectorId::new(1));
        assert_eq!(keys[1].coupled(), SectorId::new(1));
        assert_eq!(keys[0].codomain_innerlines(), &[SectorId::new(0)]);
        assert_eq!(keys[1].codomain_innerlines(), &[SectorId::new(2)]);
        assert_eq!(
            keys[0].codomain_vertices(),
            &[MultiplicityIndex::ONE, MultiplicityIndex::ONE]
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
        let _guard = test_support::CACHE_TEST_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
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
    fn fusion_layout_identity_hashes_inner_semantics_not_arc_address() {
        // What: independently allocated identity Arcs compare and hash by their
        // complete rule/sector/duality value, while distinct rules and splits do not alias.
        let hom = FusionTreeHomSpace::new(
            FusionProductSpace::new([
                SectorLeg::new([(z2_even(), 1), (z2_odd(), 1)], false),
                SectorLeg::new([(z2_even(), 1)], true),
            ]),
            FusionProductSpace::new([SectorLeg::new([(z2_odd(), 1)], false)]),
        );
        let first = Arc::new(FusionTreeHomSpaceCacheKey::new(&Z2FusionRule, &hom));
        let second = Arc::new(FusionTreeHomSpaceCacheKey::new(&Z2FusionRule, &hom));
        assert!(!Arc::ptr_eq(&first, &second));
        assert_eq!(first, second);

        let mut first_hash = rustc_hash::FxHasher::default();
        first.hash(&mut first_hash);
        let mut second_hash = rustc_hash::FxHasher::default();
        second.hash(&mut second_hash);
        assert_eq!(first_hash.finish(), second_hash.finish());

        let fermionic = Arc::new(FusionTreeHomSpaceCacheKey::new(
            &FermionParityFusionRule,
            &hom,
        ));
        assert_ne!(first, fermionic);

        let repartitioned = FusionTreeHomSpace::new(
            FusionProductSpace::new([SectorLeg::new(
                [(z2_even(), 1), (z2_odd(), 1)],
                false,
            )]),
            FusionProductSpace::new([
                SectorLeg::new([(z2_even(), 1)], true),
                SectorLeg::new([(z2_odd(), 1)], false),
            ]),
        );
        let repartitioned = Arc::new(FusionTreeHomSpaceCacheKey::new(
            &Z2FusionRule,
            &repartitioned,
        ));
        assert_ne!(first, repartitioned);
    }

    #[test]
    fn fusion_layout_global_churn_and_reset_preserve_coupled_structure() {
        // What: cap overflow gives the rebuilt layout a fresh non-recycling id;
        // coupled content remains equal, and reset cannot stale-alias live old values.
        let _guard = test_support::CACHE_TEST_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        reset_core_intern_tables();
        let rule = U1FusionRule;
        let hom = FusionTreeHomSpace::from_sectors([(U1Irrep::new(0), 2)], [(U1Irrep::new(0), 3)]);
        let old_layout = hom.cached_fusion_tree_layout(&rule);
        let old_id = old_layout.id;
        let old_structure = hom
            .coupled_subblock_structure(&rule, 1, [vec![2, 3]])
            .unwrap();
        drop(old_layout);

        for charge in 1..=(FUSION_TREE_LAYOUT_CACHE_CAP as i32 + 64) {
            let distinct = FusionTreeHomSpace::from_sectors(
                [(U1Irrep::new(charge), 1)],
                [(U1Irrep::new(charge), 1)],
            );
            let _ = distinct.fusion_tree_keys(&rule);
        }
        let global_info = fusion_tree_layout_cache_info();
        assert!(global_info.entries() <= global_info.entry_capacity());
        assert!(global_info.charged_payload_bytes() <= global_info.byte_budget());

        let rebuilt_layout = hom.cached_fusion_tree_layout(&rule);
        assert!(rebuilt_layout.id > old_id);
        let reused_structure = hom
            .coupled_subblock_structure(&rule, 1, [vec![2, 3]])
            .unwrap();
        assert_eq!(old_structure.as_ref(), reused_structure.as_ref());

        reset_core_intern_tables();
        let after_reset_layout = hom.cached_fusion_tree_layout(&rule);
        let after_reset_structure = hom
            .coupled_subblock_structure(&rule, 1, [vec![2, 3]])
            .unwrap();
        assert!(!Arc::ptr_eq(&rebuilt_layout, &after_reset_layout));
        assert!(after_reset_layout.id > rebuilt_layout.id);
        assert!(!Arc::ptr_eq(&old_structure, &after_reset_structure));
        assert_eq!(old_structure.as_ref(), after_reset_structure.as_ref());
    }

    fn local_u1_layout(
        charge: i32,
    ) -> (
        Arc<FusionTreeHomSpaceCacheKey>,
        Arc<FusionTreeHomSpaceLayout>,
    ) {
        let rule = U1FusionRule;
        let hom = FusionTreeHomSpace::from_sectors(
            [(U1Irrep::new(charge), 1)],
            [(U1Irrep::new(charge), 1)],
        );
        let key = Arc::new(FusionTreeHomSpaceCacheKey::new(&rule, &hom));
        let layout = Arc::new(fusion_tree_layout_from_data(
            next_fusion_tree_layout_id(),
            hom.fusion_tree_layout_data_uncached(&rule),
        ));
        (key, layout)
    }

    fn assert_lowered_keys_match_encoded_oracle<R>(rule: &R, hom: &FusionTreeHomSpace)
    where
        R: LoweredMultiplicityFreeAlgebra,
    {
        let encoded = hom.fusion_tree_keys_uncached(rule);
        let lowered = hom.try_fusion_tree_keys_uncached_lowered(rule).unwrap();
        assert_eq!(lowered, encoded);

        let encoded_layout = hom.fusion_tree_layout_data_uncached(rule);
        let lowered_layout = hom
            .try_fusion_tree_layout_data_uncached_lowered(rule)
            .unwrap();
        assert_eq!(lowered_layout.keys, encoded_layout.keys);
        assert_eq!(lowered_layout.sectors.len(), encoded_layout.sectors.len());
        for (actual, expected) in lowered_layout
            .sectors
            .iter()
            .zip(&encoded_layout.sectors)
        {
            assert_eq!(actual.start, expected.start);
            assert_eq!(actual.row_count, expected.row_count);
            assert_eq!(actual.col_count, expected.col_count);
        }
    }

    fn singleton_rank_hom(sector: SectorId, rank: usize) -> FusionTreeHomSpace {
        let side = |invert_dual| {
            FusionProductSpace::new((0..rank).map(|axis| {
                SectorLeg::new([(sector, axis % 3 + 1)], (axis % 2 == 0) ^ invert_dual)
            }))
        };
        FusionTreeHomSpace::new(side(false), side(true))
    }

    #[test]
    fn lowered_builder_matches_encoded_oracle_for_builtin_ranks_and_products() {
        // What: every persistent key field and key order stays identical for
        // ranks 0 through 6 across all built-in multiplicity-free algebras.
        for rank in 0..=6 {
            assert_lowered_keys_match_encoded_oracle(
                &U1FusionRule,
                &singleton_rank_hom(u1(1), rank),
            );
            assert_lowered_keys_match_encoded_oracle(
                &Z2FusionRule,
                &singleton_rank_hom(z2_odd(), rank),
            );
            assert_lowered_keys_match_encoded_oracle(
                &FermionParityFusionRule,
                &singleton_rank_hom(z2_odd(), rank),
            );
            assert_lowered_keys_match_encoded_oracle(
                &SU2FusionRule,
                &singleton_rank_hom(su2(1), rank),
            );
        }

        type U1Fz2Codec = PackedProductCodec<U1SectorLayout, Fz2SectorLayout>;
        type U1Fz2Rule =
            ProductFusionRule<U1FusionRule, FermionParityFusionRule, U1Fz2Codec>;
        let pair_rule = U1Fz2Rule::new(U1FusionRule, FermionParityFusionRule);
        let pair_sector = U1Fz2Codec::encode(u1(1), z2_odd());

        type Fz2U1Codec = PackedProductCodec<Fz2SectorLayout, U1SectorLayout>;
        type Fz2U1Layout = ProductSectorLayout<Fz2SectorLayout, U1SectorLayout>;
        type TripleCodec = PackedProductCodec<Fz2U1Layout, Su2SectorLayout>;
        type Fz2U1Rule =
            ProductFusionRule<FermionParityFusionRule, U1FusionRule, Fz2U1Codec>;
        type TripleRule = ProductFusionRule<Fz2U1Rule, SU2FusionRule, TripleCodec>;
        let triple_rule = TripleRule::new(
            Fz2U1Rule::new(FermionParityFusionRule, U1FusionRule),
            SU2FusionRule,
        );
        let triple_sector =
            TripleCodec::encode(Fz2U1Codec::encode(z2_odd(), u1(1)), su2(1));
        let triple_pair_coupled =
            TripleCodec::encode(Fz2U1Codec::encode(z2_even(), u1(2)), su2(0));

        for rank in 0..=6 {
            assert_lowered_keys_match_encoded_oracle(
                &pair_rule,
                &singleton_rank_hom(pair_sector, rank),
            );
            assert_lowered_keys_match_encoded_oracle(
                &triple_rule,
                &singleton_rank_hom(triple_sector, rank),
            );
        }

        let triple_vacuum =
            TripleCodec::encode(Fz2U1Codec::encode(z2_even(), u1(0)), su2(0));
        let multi_tuple_side = |invert_dual| {
            FusionProductSpace::new((0..4).map(|axis| {
                SectorLeg::new(
                    [(triple_vacuum, 1), (triple_sector, 2)],
                    (axis % 2 == 0) ^ invert_dual,
                )
            }))
        };
        let multi_tuple_rank_eight =
            FusionTreeHomSpace::new(multi_tuple_side(false), multi_tuple_side(true));
        assert_lowered_keys_match_encoded_oracle(&triple_rule, &multi_tuple_rank_eight);

        for dual_mask in 0usize..(1 << 3) {
            let all_dual_masks = FusionTreeHomSpace::new(
                FusionProductSpace::new([
                    SectorLeg::new([(triple_sector, 2)], dual_mask & 1 != 0),
                    SectorLeg::new([(triple_sector, 3)], dual_mask & 2 != 0),
                ]),
                FusionProductSpace::new([SectorLeg::new(
                    [(triple_pair_coupled, 4)],
                    dual_mask & 4 != 0,
                )]),
            );
            assert_lowered_keys_match_encoded_oracle(&triple_rule, &all_dual_masks);
        }

        let asymmetric = FusionTreeHomSpace::new(
            FusionProductSpace::new([
                SectorLeg::new([(triple_sector, 2)], true),
                SectorLeg::new([(triple_sector, 3)], false),
            ]),
            FusionProductSpace::new([
                SectorLeg::new(
                    [(
                        TripleCodec::encode(
                            Fz2U1Codec::encode(z2_even(), u1(2)),
                            su2(0),
                        ),
                        4,
                    )],
                    true,
                ),
                SectorLeg::new(
                    [(
                        TripleCodec::encode(
                            Fz2U1Codec::encode(z2_even(), u1(0)),
                            su2(0),
                        ),
                        5,
                    )],
                    false,
                ),
            ]),
        );
        assert_lowered_keys_match_encoded_oracle(&triple_rule, &asymmetric);
    }

    #[test]
    fn lowered_and_encoded_entries_share_the_same_layout_cache() {
        // What: old-first and lowered-first construction converge on the same
        // Arc rather than publishing parallel layouts for one semantic key.
        let _guard = test_support::CACHE_TEST_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let hom = singleton_rank_hom(su2(1), 4);

        reset_core_intern_tables();
        let encoded_first = hom.cached_fusion_tree_layout(&SU2FusionRule);
        let lowered_second = hom
            .try_cached_fusion_tree_layout_lowered(&SU2FusionRule)
            .unwrap();
        assert!(Arc::ptr_eq(&encoded_first, &lowered_second));

        reset_core_intern_tables();
        let lowered_first = hom
            .try_cached_fusion_tree_layout_lowered(&SU2FusionRule)
            .unwrap();
        let encoded_second = hom.cached_fusion_tree_layout(&SU2FusionRule);
        assert!(Arc::ptr_eq(&lowered_first, &encoded_second));
    }

    #[test]
    fn lowered_layout_cache_hit_performs_no_decode_or_channel_work() {
        // What: a warm hit returns the retained layout without entering any
        // typed decode, forward fold, or backward fusion enumeration.
        let _guard = test_support::CACHE_TEST_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        reset_core_intern_tables();
        let hom = singleton_rank_hom(su2(1), 5);
        reset_lowered_layout_build_observations();
        let cold = hom.try_fusion_tree_keys_lowered(&SU2FusionRule).unwrap();
        let (cold_decodes, cold_channels) = lowered_layout_build_observations();
        assert!(cold_decodes > 0);
        assert!(cold_channels > 0);

        reset_lowered_layout_build_observations();
        let warm = hom.try_fusion_tree_keys_lowered(&SU2FusionRule).unwrap();
        assert!(Arc::ptr_eq(&cold, &warm));
        assert_eq!(lowered_layout_build_observations(), (0, 0));
    }

    #[test]
    fn prepared_lowered_layout_publishes_only_at_commit() {
        // What: cold preparation enumerates exactly once but does not consume
        // identity or cache admission until its explicit commit point.
        let _guard = test_support::CACHE_TEST_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        reset_core_intern_tables();
        reset_fusion_tree_layout_probe_side_effect_calls();
        reset_lowered_layout_build_observations();
        let hom = singleton_rank_hom(su2(1), 5);

        let prepared = hom
            .prepare_fusion_tree_layout_lowered(&SU2FusionRule)
            .unwrap();
        let cold_work = lowered_layout_build_observations();
        assert!(cold_work.0 > 0);
        assert!(cold_work.1 > 0);
        // Why not inspect global cache totals: unrelated parallel tests may
        // populate the same process cache. These thread-local probes attribute
        // publication exactly to this transaction.
        assert_eq!(fusion_tree_layout_probe_side_effect_calls(), (0, 0));

        let keys = prepared.commit();
        assert!(!keys.is_empty());
        assert_eq!(fusion_tree_layout_probe_side_effect_calls(), (1, 1));
    }

    #[test]
    fn prepared_lowered_final_structure_reuses_one_checked_enumeration() {
        // What: cold lowered preparation and the direct leg-degeneracy builder
        // match the established single-pass structure without a second
        // decode/channel enumeration or early cache publication.
        let _guard = test_support::CACHE_TEST_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let hom = singleton_rank_hom(su2(1), 5);
        reset_core_intern_tables();
        let expected = hom
            .coupled_subblock_structure_from_leg_degeneracies(&SU2FusionRule)
            .unwrap();

        reset_core_intern_tables();
        reset_fusion_tree_layout_probe_side_effect_calls();
        reset_lowered_layout_build_observations();
        let prepared = hom
            .prepare_fusion_tree_layout_lowered(&SU2FusionRule)
            .unwrap();
        let prepare_work = lowered_layout_build_observations();
        assert!(prepare_work.0 > 0);
        assert!(prepare_work.1 > 0);
        assert_eq!(fusion_tree_layout_probe_side_effect_calls(), (0, 0));

        reset_lowered_layout_build_observations();
        let actual = prepared.build_from_leg_degeneracies(&hom).unwrap();
        assert_eq!(lowered_layout_build_observations(), (0, 0));
        assert_eq!(actual, expected);
        assert_eq!(fusion_tree_layout_probe_side_effect_calls(), (0, 0));

        prepared.commit();
        assert_eq!(fusion_tree_layout_probe_side_effect_calls(), (1, 1));
    }

    #[test]
    fn prepared_lowered_final_structure_checks_signature_but_reads_target_degeneracies() {
        // What: a prepared layout rejects another same-rank sector signature
        // without publication, while the same sectors/duality with different
        // degeneracies are accepted as the target structure authority.
        let _guard = test_support::CACHE_TEST_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        reset_core_intern_tables();
        let source = FusionTreeHomSpace::new(
            FusionProductSpace::new([SectorLeg::new([(u1(1), 2)], false)]),
            FusionProductSpace::new([SectorLeg::new([(u1(1), 3)], true)]),
        );
        let prepared = source
            .prepare_fusion_tree_layout_lowered(&U1FusionRule)
            .unwrap();
        reset_fusion_tree_layout_probe_side_effect_calls();
        let mismatched = FusionTreeHomSpace::new(
            FusionProductSpace::new([SectorLeg::new([(u1(2), 2)], false)]),
            FusionProductSpace::new([SectorLeg::new([(u1(2), 3)], true)]),
        );

        let error = prepared
            .build_from_leg_degeneracies(&mismatched)
            .unwrap_err();
        assert_eq!(
            error,
            CoreError::MalformedFusionTree {
                message: "prepared layout does not match HomSpace sector signature",
            }
        );
        assert_eq!(fusion_tree_layout_probe_side_effect_calls(), (0, 0));
        let duality_mismatched = FusionTreeHomSpace::new(
            FusionProductSpace::new([SectorLeg::new([(u1(1), 2)], true)]),
            FusionProductSpace::new([SectorLeg::new([(u1(1), 3)], true)]),
        );
        assert_eq!(
            prepared
                .build_from_leg_degeneracies(&duality_mismatched)
                .unwrap_err(),
            CoreError::MalformedFusionTree {
                message: "prepared layout does not match HomSpace sector signature",
            }
        );
        assert_eq!(fusion_tree_layout_probe_side_effect_calls(), (0, 0));

        let target = FusionTreeHomSpace::new(
            FusionProductSpace::new([SectorLeg::new([(u1(1), 5)], false)]),
            FusionProductSpace::new([SectorLeg::new([(u1(1), 7)], true)]),
        );
        let structure = prepared.build_from_leg_degeneracies(&target).unwrap();
        assert_eq!(
            structure
                .degeneracy_structure()
                .blocks()
                .first()
                .unwrap()
                .shape(),
            &[5, 7]
        );
        assert_eq!(fusion_tree_layout_probe_side_effect_calls(), (0, 0));
    }

    #[test]
    fn cached_lowered_preparation_is_observationally_read_only() {
        // What: preparing an already-cached layout performs no enumeration,
        // ID issue, or admission when abandoned.
        let _guard = test_support::CACHE_TEST_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        reset_core_intern_tables();
        let hom = singleton_rank_hom(su2(1), 5);
        hom.try_fusion_tree_keys_lowered(&SU2FusionRule).unwrap();
        reset_fusion_tree_layout_probe_side_effect_calls();
        reset_lowered_layout_build_observations();

        let prepared = hom
            .prepare_fusion_tree_layout_lowered(&SU2FusionRule)
            .unwrap();
        assert!(!prepared.keys().is_empty());
        drop(prepared);

        assert_eq!(lowered_layout_build_observations(), (0, 0));
        assert_eq!(fusion_tree_layout_probe_side_effect_calls(), (0, 0));
    }

    #[test]
    fn cached_lowered_commit_readmits_after_core_reset() {
        // What: a cached preparation that survives reset republishes its exact
        // retained keys without consuming a fresh layout identity.
        let _guard = test_support::CACHE_TEST_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        reset_core_intern_tables();
        let hom = singleton_rank_hom(su2(1), 5);
        hom.try_fusion_tree_keys_lowered(&SU2FusionRule).unwrap();
        let prepared = hom
            .prepare_fusion_tree_layout_lowered(&SU2FusionRule)
            .unwrap();
        let retained = prepared.keys_arc();
        reset_core_intern_tables();
        reset_fusion_tree_layout_probe_side_effect_calls();

        let committed = prepared.commit();

        assert!(Arc::ptr_eq(&retained, &committed));
        assert_eq!(fusion_tree_layout_probe_side_effect_calls(), (0, 1));
    }

    #[test]
    fn concurrent_lowered_commits_share_one_layout_admission() {
        // What: two cold preparations racing to commit converge on one Arc
        // and one cache miss without the losing transaction issuing an ID.
        let _guard = test_support::CACHE_TEST_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        reset_core_intern_tables();
        let hom = singleton_rank_hom(su2(1), 5);
        let barrier = Arc::new(std::sync::Barrier::new(2));
        let handles = (0..2)
            .map(|_| {
                let hom = hom.clone();
                let barrier = Arc::clone(&barrier);
                std::thread::spawn(move || {
                    reset_fusion_tree_layout_probe_side_effect_calls();
                    let prepared = hom
                        .prepare_fusion_tree_layout_lowered(&SU2FusionRule)
                        .unwrap();
                    barrier.wait();
                    let keys = prepared.commit();
                    (keys, fusion_tree_layout_probe_side_effect_calls())
                })
            })
            .collect::<Vec<_>>();
        let mut results = handles
            .into_iter()
            .map(|handle| handle.join().unwrap())
            .collect::<Vec<_>>();
        let (second, second_calls) = results.pop().unwrap();
        let (first, first_calls) = results.pop().unwrap();

        assert!(Arc::ptr_eq(&first, &second));
        assert_eq!(first_calls.0 + second_calls.0, 1);
        assert_eq!(first_calls.1 + second_calls.1, 1);
    }

    #[test]
    fn lowered_builder_reports_malformed_ids_and_algebra_closure_without_panicking() {
        // What: packed decode remains a non-algebra lowered error, while U(1),
        // SU(2), and recursive product closure failures retain exact causes.
        type Codec = PackedProductCodec<Fz2SectorLayout, U1SectorLayout>;
        type Rule =
            ProductFusionRule<FermionParityFusionRule, U1FusionRule, Codec>;
        let rule = Rule::new(FermionParityFusionRule, U1FusionRule);
        let malformed = singleton_rank_hom(SectorId::new(usize::MAX), 1);
        let error = malformed
            .try_fusion_tree_keys_uncached_lowered(&rule)
            .unwrap_err();
        assert_eq!(
            error.static_message(),
            "built-in fusion-tree layout contains an invalid product sector"
        );
        assert_eq!(
            error.clone().into_checked_fusion_algebra(),
            FusionAlgebraError::ProductCodec(
                Codec::decode_checked(SectorId::new(usize::MAX)).unwrap_err(),
            )
        );
        assert!(error.into_fusion_algebra().is_err());

        let invalid_z2 = singleton_rank_hom(SectorId::new(2), 1)
            .try_fusion_tree_keys_uncached_lowered(&Z2FusionRule)
            .unwrap_err();
        assert_eq!(
            invalid_z2.into_checked_fusion_algebra(),
            FusionAlgebraError::InvalidSector {
                sector: SectorId::new(2),
            }
        );

        let u1_overflow = FusionTreeHomSpace::new(
            FusionProductSpace::new([
                SectorLeg::new([(u1(i32::MAX), 1)], false),
                SectorLeg::new([(u1(1), 1)], false),
            ]),
            FusionProductSpace::new(Vec::<SectorLeg>::new()),
        );
        let error = u1_overflow
            .try_fusion_tree_keys_uncached_lowered(&U1FusionRule)
            .unwrap_err();
        assert_eq!(
            error.into_fusion_algebra().unwrap(),
            FusionAlgebraError::U1FusionOverflow {
                left: i32::MAX,
                right: 1,
            }
        );

        let u1_dual_overflow = FusionTreeHomSpace::new(
            FusionProductSpace::new([
                SectorLeg::new([(u1(0), 1)], false),
                SectorLeg::new([(u1(0), 1)], false),
                SectorLeg::new([(u1(i32::MIN), 1)], false),
            ]),
            FusionProductSpace::new(Vec::<SectorLeg>::new()),
        );
        let error = u1_dual_overflow
            .try_fusion_tree_keys_uncached_lowered(&U1FusionRule)
            .unwrap_err();
        assert_eq!(
            error.into_fusion_algebra().unwrap(),
            FusionAlgebraError::U1DualOverflow { charge: i32::MIN }
        );

        let su2_overflow = FusionTreeHomSpace::new(
            FusionProductSpace::new([
                SectorLeg::new([(su2(128), 1)], false),
                SectorLeg::new([(su2(127), 1)], false),
            ]),
            FusionProductSpace::new([]),
        );
        let error = su2_overflow
            .try_fusion_tree_keys_uncached_lowered(&SU2FusionRule)
            .unwrap_err();
        assert_eq!(
            error.into_fusion_algebra().unwrap(),
            FusionAlgebraError::FusionNotRepresentable {
                left: su2(128),
                right: su2(127),
            }
        );

        let pair_overflow = FusionTreeHomSpace::new(
            FusionProductSpace::new([
                SectorLeg::new([(Codec::encode(z2_even(), u1(i32::MAX)), 1)], false),
                SectorLeg::new([(Codec::encode(z2_odd(), u1(1)), 1)], false),
            ]),
            FusionProductSpace::new([]),
        );
        let error = pair_overflow
            .try_fusion_tree_keys_uncached_lowered(&rule)
            .unwrap_err();
        assert_eq!(
            error.into_fusion_algebra().unwrap(),
            FusionAlgebraError::U1FusionOverflow {
                left: i32::MAX,
                right: 1,
            }
        );

        type PairLayout = ProductSectorLayout<Fz2SectorLayout, U1SectorLayout>;
        type TripleCodec = PackedProductCodec<PairLayout, Su2SectorLayout>;
        type TripleRule = ProductFusionRule<Rule, SU2FusionRule, TripleCodec>;
        let triple_rule = TripleRule::new(rule, SU2FusionRule);
        let triple = |parity, charge, spin| {
            TripleCodec::encode(Codec::encode(parity, charge), spin)
        };
        let triple_overflow = FusionTreeHomSpace::new(
            FusionProductSpace::new([
                SectorLeg::new(
                    [(triple(z2_even(), u1(i32::MAX), su2(0)), 1)],
                    false,
                ),
                SectorLeg::new([(triple(z2_odd(), u1(1), su2(1)), 1)], false),
            ]),
            FusionProductSpace::new([]),
        );
        let error = triple_overflow
            .try_fusion_tree_keys_uncached_lowered(&triple_rule)
            .unwrap_err();
        assert_eq!(
            error.into_fusion_algebra().unwrap(),
            FusionAlgebraError::U1FusionOverflow {
                left: i32::MAX,
                right: 1,
            }
        );
    }

    fn assert_failed_lowered_build_is_transactional<R>(
        rule: &R,
        hom: &FusionTreeHomSpace,
        expected: FusionAlgebraError,
    ) where
        R: LoweredMultiplicityFreeAlgebra,
    {
        let _guard = test_support::CACHE_TEST_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        reset_core_intern_tables();
        reset_fusion_tree_layout_probe_side_effect_calls();
        reset_hom_space_intern_calls();
        reset_block_structure_intern_calls();
        let error = hom.try_fusion_tree_keys_lowered(rule).unwrap_err();
        assert_eq!(error.into_fusion_algebra().unwrap(), expected);
        assert_eq!(fusion_tree_layout_probe_side_effect_calls(), (0, 0));
        assert_eq!(hom_space_intern_calls(), 0);
        assert_eq!(block_structure_intern_calls(), 0);
    }

    #[test]
    fn failed_lowered_algebra_builds_publish_no_identity_or_intern_state() {
        // What: invalid built-in U1, SU2, and product closure leaves layout
        // identity/admission, HomSpace, and BlockStructure state untouched.
        let u1_overflow = FusionTreeHomSpace::new(
            FusionProductSpace::new([
                SectorLeg::new([(u1(i32::MAX), 1)], false),
                SectorLeg::new([(u1(1), 1)], false),
            ]),
            FusionProductSpace::new([]),
        );
        assert_failed_lowered_build_is_transactional(
            &U1FusionRule,
            &u1_overflow,
            FusionAlgebraError::U1FusionOverflow {
                left: i32::MAX,
                right: 1,
            },
        );

        let su2_overflow = FusionTreeHomSpace::new(
            FusionProductSpace::new([
                SectorLeg::new([(su2(128), 1)], false),
                SectorLeg::new([(su2(127), 1)], false),
            ]),
            FusionProductSpace::new([]),
        );
        assert_failed_lowered_build_is_transactional(
            &SU2FusionRule,
            &su2_overflow,
            FusionAlgebraError::FusionNotRepresentable {
                left: su2(128),
                right: su2(127),
            },
        );

        type Codec = PackedProductCodec<Fz2SectorLayout, U1SectorLayout>;
        type Rule = ProductFusionRule<FermionParityFusionRule, U1FusionRule, Codec>;
        let product_rule = Rule::new(FermionParityFusionRule, U1FusionRule);
        let product_overflow = FusionTreeHomSpace::new(
            FusionProductSpace::new([
                SectorLeg::new([(Codec::encode(z2_even(), u1(i32::MAX)), 1)], false),
                SectorLeg::new([(Codec::encode(z2_odd(), u1(1)), 1)], false),
            ]),
            FusionProductSpace::new([]),
        );
        assert_failed_lowered_build_is_transactional(
            &product_rule,
            &product_overflow,
            FusionAlgebraError::U1FusionOverflow {
                left: i32::MAX,
                right: 1,
            },
        );
    }

    #[test]
    fn empty_lowered_leg_short_circuits_before_other_leg_decode() {
        // What: an empty product has no tuples and returns empty even when a
        // different leg carries an ID that would fail lowered decoding.
        let hom = FusionTreeHomSpace::new(
            FusionProductSpace::new([
                SectorLeg::new([(SectorId::new(usize::MAX), 1)], false),
                SectorLeg::new(Vec::<(SectorId, usize)>::new(), false),
            ]),
            FusionProductSpace::new(Vec::<SectorLeg>::new()),
        );
        let keys = hom
            .try_fusion_tree_keys_uncached_lowered(&U1FusionRule)
            .unwrap();
        assert!(keys.is_empty());
    }

    #[test]
    fn expert_cantor_product_keeps_the_encoded_fallback() {
        // What: custom-codec/expert product construction remains available via
        // the unchanged encoded public entry and preserves its historical IDs.
        type Rule = ProductFusionRule<FermionParityFusionRule, U1FusionRule>;
        let rule = Rule::new(FermionParityFusionRule, U1FusionRule);
        let sector = TensorKitProductCodec::encode(z2_odd(), u1(2));
        let hom = singleton_rank_hom(sector, 3);
        assert_eq!(
            hom.fusion_tree_keys(&rule).as_ref(),
            hom.fusion_tree_keys_uncached(&rule)
        );
    }

    #[test]
    fn fusion_layout_local_cache_is_strict_insertion_order_and_resets_exactly() {
        // What: read hits do not promote FIFO order; entry eviction and reset
        // update charged bytes and counters deterministically in isolated state.
        let mut cache = FusionTreeLayoutCache::new(2, 100, 100);
        let (key0, layout0) = local_u1_layout(40_000);
        let (key1, layout1) = local_u1_layout(40_001);
        let (key2, layout2) = local_u1_layout(40_002);
        cache.admit(Arc::clone(&key0), layout0, 30);
        cache.admit(Arc::clone(&key1), layout1, 30);
        assert!(cache.lookup(&key0).is_some());
        cache.admit(Arc::clone(&key2), layout2, 30);

        assert!(cache.lookup(&key0).is_none());
        assert!(cache.lookup(&key1).is_some());
        assert!(cache.lookup(&key2).is_some());
        assert_eq!(cache.info().entries(), 2);
        assert_eq!(cache.info().charged_payload_bytes(), 60);
        assert_eq!(cache.info().evictions(), 1);

        cache.clear();
        assert_eq!(cache.info().entries(), 0);
        assert_eq!(cache.info().charged_payload_bytes(), 0);
        assert_eq!(cache.info().misses(), 0);
        assert_eq!(cache.info().evictions(), 0);
        assert_eq!(cache.info().admission_bypasses(), 0);
    }

    #[test]
    fn fusion_layout_local_cache_enforces_byte_and_max_entry_admission() {
        // What: charged-byte pressure evicts oldest entries, while an oversized
        // entry is returned to its caller but never retained by the cache.
        let mut cache = FusionTreeLayoutCache::new(8, 50, 40);
        let (key0, layout0) = local_u1_layout(50_000);
        let (key1, layout1) = local_u1_layout(50_001);
        let (oversized_key, oversized_layout) = local_u1_layout(50_002);
        cache.admit(Arc::clone(&key0), layout0, 30);
        cache.admit(Arc::clone(&key1), layout1, 30);

        assert!(cache.lookup(&key0).is_none());
        assert!(cache.lookup(&key1).is_some());
        assert_eq!(cache.info().charged_payload_bytes(), 30);
        assert_eq!(cache.info().evictions(), 1);

        let returned = cache.admit(Arc::clone(&oversized_key), Arc::clone(&oversized_layout), 41);
        assert!(Arc::ptr_eq(&returned, &oversized_layout));
        assert!(cache.lookup(&oversized_key).is_none());
        assert_eq!(cache.info().entries(), 1);
        assert_eq!(cache.info().charged_payload_bytes(), 30);
        assert_eq!(cache.info().admission_bypasses(), 1);
    }

    #[test]
    fn fusion_layout_local_cache_bypasses_oversized_rule_identity() {
        #[derive(Clone)]
        struct OversizedIdentityRule {
            identity: RuleIdentity,
        }

        impl FusionRule for OversizedIdentityRule {
            fn rule_identity(&self) -> RuleIdentity {
                self.identity.clone()
            }

            fn fusion_style(&self) -> FusionStyleKind {
                FusionStyleKind::Unique
            }

            fn braiding_style(&self) -> BraidingStyleKind {
                BraidingStyleKind::Bosonic
            }

            fn vacuum(&self) -> SectorId {
                SectorId::new(0)
            }

            fn fusion_channels(&self, left: SectorId, right: SectorId) -> SectorVec {
                smallvec![SectorId::new(left.id() ^ right.id())]
            }
        }

        impl MultiplicityFreeFusionRule for OversizedIdentityRule {}

        // What: canonical rule bytes participate in admission accounting, so
        // an identity alone above the per-entry limit is computed but not retained.
        let canonical_bytes = Arc::<[u8]>::from(vec![
            0;
            FUSION_TREE_LAYOUT_CACHE_MAX_ENTRY_BYTES
                .saturating_add(1)
        ]);
        let rule = OversizedIdentityRule {
            identity: RuleIdentity::from_canonical_bytes::<OversizedIdentityRule>(
                0,
                canonical_bytes,
            ),
        };
        assert!(
            rule.identity.charged_retained_bytes()
                > FUSION_TREE_LAYOUT_CACHE_MAX_ENTRY_BYTES
        );
        let hom = FusionTreeHomSpace::from_sectors(
            [(SectorId::new(0), 1)],
            Vec::<(SectorId, usize)>::new(),
        );
        let key = Arc::new(FusionTreeHomSpaceCacheKey::new(&rule, &hom));
        let layout = Arc::new(fusion_tree_layout_from_data(
            next_fusion_tree_layout_id(),
            hom.fusion_tree_layout_data_uncached(&rule),
        ));
        let charged_bytes = charged_fusion_tree_layout_bytes(&key, &layout);
        assert!(charged_bytes > FUSION_TREE_LAYOUT_CACHE_MAX_ENTRY_BYTES);

        let mut cache = FusionTreeLayoutCache::new(
            8,
            FUSION_TREE_LAYOUT_CACHE_BYTE_BUDGET,
            FUSION_TREE_LAYOUT_CACHE_MAX_ENTRY_BYTES,
        );
        let returned = cache.admit(Arc::clone(&key), Arc::clone(&layout), charged_bytes);
        assert!(Arc::ptr_eq(&returned, &layout));
        assert!(cache.lookup(&key).is_none());
        let info = cache.info();
        assert_eq!(info.entries(), 0);
        assert_eq!(info.admission_bypasses(), 1);
    }

    #[test]
    fn fusion_layout_shape_and_fermionic_rule_provenance_do_not_alias() {
        // What: one sector layout may be shared across degeneracies, but concrete
        // shapes and bosonic/fermionic rule provenance select distinct structures/layouts.
        let _guard = test_support::CACHE_TEST_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        reset_core_intern_tables();
        let hom = FusionTreeHomSpace::new(
            FusionProductSpace::new([SectorLeg::new(
                [(z2_even(), 1), (z2_odd(), 1)],
                false,
            )]),
            FusionProductSpace::new([SectorLeg::new(
                [(z2_even(), 1), (z2_odd(), 1)],
                false,
            )]),
        );
        let bosonic_layout = hom.cached_fusion_tree_layout(&Z2FusionRule);
        let fermionic_layout = hom.cached_fusion_tree_layout(&FermionParityFusionRule);
        assert_ne!(bosonic_layout.id, fermionic_layout.id);

        let small = hom
            .coupled_subblock_structure(&FermionParityFusionRule, 1, [vec![1, 1], vec![1, 1]])
            .unwrap();
        let large_hom = FusionTreeHomSpace::new(
            FusionProductSpace::new([SectorLeg::new(
                [(z2_even(), 2), (z2_odd(), 3)],
                false,
            )]),
            FusionProductSpace::new([SectorLeg::new(
                [(z2_even(), 2), (z2_odd(), 3)],
                false,
            )]),
        );
        let large = large_hom
            .coupled_subblock_structure(&FermionParityFusionRule, 1, [vec![2, 2], vec![3, 3]])
            .unwrap();
        assert!(!Arc::ptr_eq(&small, &large));
        assert_ne!(small.as_ref(), large.as_ref());

        let transient_hom = FusionTreeHomSpace::from_sectors(
            [(U1Irrep::new(17), 4)],
            [(U1Irrep::new(17), 5)],
        );
        let transient = transient_hom
            .coupled_subblock_structure(&U1FusionRule, 1, [vec![4, 5]])
            .unwrap();
        let expired = Arc::downgrade(&transient);
        drop(transient);
        assert!(expired.upgrade().is_none());
        let rebuilt = transient_hom
            .coupled_subblock_structure(&U1FusionRule, 1, [vec![4, 5]])
            .unwrap();
        assert_eq!(rebuilt.block(0).unwrap().shape(), &[4, 5]);
    }

    #[test]
    fn fusion_layout_lookup_and_reset_are_concurrent_safe_after_poison() {
        // What: a poisoned layout lock and concurrent lookup/reset cannot publish
        // partial layout content or return a structure with the wrong keys/shapes.
        let _guard = test_support::CACHE_TEST_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        reset_core_intern_tables();
        let poisoned = std::panic::catch_unwind(|| {
            let _write = fusion_tree_layout_cache()
                .write()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            panic!("poison fusion layout cache for recovery test");
        });
        assert!(poisoned.is_err());

        let workers = (0..4)
            .map(|worker| {
                std::thread::spawn(move || {
                    let rule = U1FusionRule;
                    for iteration in 0..64 {
                        if worker == 0 && iteration % 8 == 0 {
                            reset_core_intern_tables();
                        }
                        let charge = worker * 100 + iteration;
                        let hom = FusionTreeHomSpace::from_sectors(
                            [(U1Irrep::new(charge), 2)],
                            [(U1Irrep::new(charge), 3)],
                        );
                        let keys = hom.fusion_tree_keys(&rule);
                        assert_eq!(keys.len(), 1);
                        assert_eq!(keys[0].coupled(), U1Irrep::new(charge).sector_id());
                        let structure = hom
                            .coupled_subblock_structure(&rule, 1, [vec![2, 3]])
                            .unwrap();
                        assert_eq!(structure.block_count(), 1);
                        assert_eq!(structure.block(0).unwrap().shape(), &[2, 3]);
                    }
                })
            })
            .collect::<Vec<_>>();
        for worker in workers {
            worker.join().unwrap();
        }
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
            .all(|key| key.codomain_vertices() == [MultiplicityIndex::ONE]));
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
        let mut blocks: BTreeMap<Vec<usize>, Vec<FusionTreePairKey>> = BTreeMap::new();
        for key in keys.iter() {
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
                let mut want: BTreeMap<FusionTreePairKey, f64> = BTreeMap::new();
                for (k, c) in &per_source {
                    *want.entry(k.clone()).or_insert(0.0) += c;
                }
                let mut got: BTreeMap<FusionTreePairKey, f64> = BTreeMap::new();
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

    fn compact_operator_cohort_fixture<R>(
        rule: &R,
        external: SectorId,
        coupled: SectorId,
    ) -> Vec<FusionTreePairKey>
    where
        R: MultiplicityFreeRigidSymbols<Scalar = f64> + MultiplicityFreeFusionRule,
    {
        let codomain: [SectorLeg; 8] =
            std::array::from_fn(|_| SectorLeg::new([(external, 1)], false));
        let hom = FusionTreeHomSpace::new(
            FusionProductSpace::new(codomain),
            FusionProductSpace::new([SectorLeg::new([(coupled, 1)], false)]),
        );
        let keys = hom.fusion_tree_keys(rule).to_vec();
        assert!(
            keys.len() >= 16,
            "fixture must expose every requested source cohort"
        );
        keys
    }

    fn assert_compact_operator_cohorts<R>(
        rule: &R,
        sources: &[FusionTreePairKey],
    ) where
        R: MultiplicityFreeRigidSymbols<Scalar = f64> + MultiplicityFreeFusionRule,
    {
        for cohort_len in [1usize, 2, 4, 8, 16] {
            let cohort = &sources[..cohort_len];

            // What: a direct compact bend-left produces the exact public
            // full-key rows in source and first-appearance destination order.
            let group = validate_tree_pair_block_group_for_rule(rule, cohort)
                .unwrap()
                .unwrap();
            let basis = CompactMultiplicityFreeTreePairBasis::from_group(group).unwrap();
            let (basis, columns) = compact_bendleft_block_first(rule, basis).unwrap();
            let got = scatter_compact_block(basis, columns);
            let want = cohort
                .iter()
                .map(|source| {
                    multiplicity_free_bendleft_tree_pair(rule, source)
                        .unwrap()
                        .into_vec()
                })
                .collect::<Vec<_>>();
            assert_eq!(got, want);

            // What: a direct compact bend-right preserves the same one-row
            // oracle and ordering for every requested block cohort size.
            let group = validate_tree_pair_block_group_for_rule(rule, cohort)
                .unwrap()
                .unwrap();
            let basis = CompactMultiplicityFreeTreePairBasis::from_group(group).unwrap();
            let (basis, columns) = compact_bendright_block_first(rule, basis).unwrap();
            let got = scatter_compact_block(basis, columns);
            let want = cohort
                .iter()
                .map(|source| {
                    multiplicity_free_bendright_tree_pair(rule, source)
                        .unwrap()
                        .into_vec()
                })
                .collect::<Vec<_>>();
            assert_eq!(got, want);

            // What: compact non-first Artin rows use the public F/R kernel and
            // retain its channel order for SU(2)-branching source cohorts.
            let group = validate_tree_pair_block_group_for_rule(rule, cohort)
                .unwrap()
                .unwrap();
            let basis = CompactMultiplicityFreeTreePairBasis::from_group(group).unwrap();
            let (basis, columns) =
                compact_codomain_artin_block_first(rule, basis, 3, false).unwrap();
            let got = scatter_compact_block(basis, columns);
            let want = cohort
                .iter()
                .map(|source| {
                    let domain = source.domain_tree().clone();
                    multiplicity_free_artin_braid_at_with_inverse(
                        rule,
                        source.codomain_tree(),
                        3,
                        false,
                    )
                    .unwrap()
                    .into_iter()
                    .map(|(codomain, coefficient)| {
                        (
                            FusionTreePairKey::pair(codomain, domain.clone()),
                            coefficient,
                        )
                    })
                    .collect::<Vec<_>>()
                })
                .collect::<Vec<_>>();
            assert_eq!(got, want);
        }
    }

    #[test]
    fn compact_block_operators_match_su2_cohort_oracles() {
        let rule = SU2FusionRule;
        let sources = compact_operator_cohort_fixture(&rule, su2(2), su2(2));

        assert_compact_operator_cohorts(&rule, &sources);
    }

    #[test]
    fn compact_block_operators_match_fermionic_product_cohort_oracles() {
        type FpU1Rule = ProductFusionRule<FermionParityFusionRule, U1FusionRule>;
        type ProductRule = ProductFusionRule<FpU1Rule, SU2FusionRule>;
        let left = FpU1Rule::default();
        let rule = ProductRule::default();
        let external =
            rule.encode_sector(left.encode_sector(z2_odd(), u1(0)), su2(2));
        let coupled =
            rule.encode_sector(left.encode_sector(z2_even(), u1(0)), su2(2));
        let sources = compact_operator_cohort_fixture(&rule, external, coupled);

        assert_compact_operator_cohorts(&rule, &sources);
    }

    fn assert_all_codomain_compact_cohorts<R>(
        rule: &R,
        sources: &[FusionTreeKey],
    ) where
        R: MultiplicityFreeFusionSymbols<Scalar = f64> + MultiplicityFreeFusionRule,
    {
        let cases = [
            ([1usize, 0, 2, 3, 4, 5, 6, 7], [0usize, 1, 2, 3, 4, 5, 6, 7]),
            ([2usize, 0, 1, 3, 4, 5, 6, 7], [2usize, 0, 1, 3, 4, 5, 6, 7]),
        ];
        for cohort_len in [1usize, 2, 4, 8, 16] {
            let cohort = &sources[..cohort_len];
            for (permutation, levels) in cases {
                let got =
                    multiplicity_free_braid_tree_block(rule, cohort, &permutation, &levels)
                        .unwrap();
                let want = cohort
                    .iter()
                    .map(|source| {
                        multiplicity_free_braid_tree(rule, source, &permutation, &levels)
                            .unwrap()
                    })
                    .collect::<Vec<_>>();
                assert_eq!(got.len(), want.len());
                for (got_row, want_row) in got.iter().zip(&want) {
                    // What: compact all-codomain execution preserves the scalar
                    // kernel's destination order as well as every full key.
                    assert_eq!(
                        got_row.iter().map(|(key, _)| key).collect::<Vec<_>>(),
                        want_row.iter().map(|(key, _)| key).collect::<Vec<_>>()
                    );
                    assert_eq!(got_row.len(), want_row.len());
                    for ((_, got_coefficient), (_, want_coefficient)) in
                        got_row.iter().zip(want_row)
                    {
                        assert!(
                            (got_coefficient - want_coefficient).abs()
                                <= 1.0e-12 * (1.0 + want_coefficient.abs())
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn all_codomain_compact_block_matches_su2_cohorts() {
        let rule = SU2FusionRule;
        let sources = compact_operator_cohort_fixture(&rule, su2(2), su2(2))
            .into_iter()
            .map(|source| source.codomain_tree().clone())
            .collect::<Vec<_>>();

        assert_all_codomain_compact_cohorts(&rule, &sources);
    }

    #[test]
    fn all_codomain_compact_block_matches_fermionic_product_cohorts() {
        type FpU1Rule = ProductFusionRule<FermionParityFusionRule, U1FusionRule>;
        type ProductRule = ProductFusionRule<FpU1Rule, SU2FusionRule>;
        let left = FpU1Rule::default();
        let rule = ProductRule::default();
        let external =
            rule.encode_sector(left.encode_sector(z2_odd(), u1(0)), su2(2));
        let coupled =
            rule.encode_sector(left.encode_sector(z2_even(), u1(0)), su2(2));
        let sources = compact_operator_cohort_fixture(&rule, external, coupled)
            .into_iter()
            .map(|source| source.codomain_tree().clone())
            .collect::<Vec<_>>();

        assert_all_codomain_compact_cohorts(&rule, &sources);
    }

    #[test]
    fn generic_tree_cannot_enter_multiplicity_free_projection() {
        // What: a genuine SU(3) 8 x 8 -> 8 multiplicity-two tree is rejected
        // before the compact path can erase its vertex identity.
        let rule = su3();
        let eight = su3_id(1, 1);
        let tree = FusionTreeKey::try_new_for_rule(
            &rule,
            [eight, eight],
            eight,
            [false, false],
            [],
            [MultiplicityIndex::new(2).unwrap()],
        )
        .unwrap();

        let error = match project_multiplicity_free_tree(&rule, &tree) {
            Ok(_) => panic!("Generic tree entered multiplicity-free projection"),
            Err(error) => error,
        };
        assert_eq!(
            error,
            CoreError::UnsupportedFusionStyle {
                expected: FusionStyleKind::Simple,
                actual: FusionStyleKind::Generic,
            }
        );

        // A proof is an indexed view of the exact slice it checked. Creating
        // another key later cannot extend that authority.
        let checked = FusionTreeKey::try_from_sector_ids([1, 1], 0, [false; 2], [], [1]).unwrap();
        let separate =
            FusionTreeKey::try_from_sector_ids([1, 1], 0, [false; 2], [], [2]).unwrap();
        let projection =
            MultiplicityFreeTreeProjection::checked(&SU2FusionRule, std::slice::from_ref(&checked))
                .unwrap();
        assert!(projection.tree_at(0).is_some());
        assert!(projection.tree_at(1).is_none());
        assert_eq!(separate.vertices(), &[MultiplicityIndex::new(2).unwrap()]);
    }

    #[test]
    fn compact_block_error_does_not_publish_partial_rows() {
        let rule = SU2FusionRule;
        let sources = compact_operator_cohort_fixture(&rule, su2(2), su2(2));
        let valid = sources[..4].to_vec();
        let permutation = [7usize, 6, 5, 4, 3, 2, 1, 0];
        let domain = [8usize];
        let baseline =
            multiplicity_free_permute_tree_pair_block(&rule, &valid, &permutation, &domain)
                .unwrap();

        let mut malformed = valid.clone();
        let source = &valid[1];
        let codomain = source.codomain_tree();
        let shortened = FusionTreeKey::new(
            codomain.uncoupled().iter().copied(),
            codomain.coupled(),
            codomain.is_dual().iter().copied(),
            codomain.innerlines()[..codomain.innerlines().len() - 1]
                .iter()
                .copied(),
            codomain.vertices().iter().copied(),
        );
        malformed[1] =
            FusionTreePairKey::pair(shortened, source.domain_tree().clone());
        let snapshot = malformed.clone();

        // What: an error after earlier source rows were staged leaves caller
        // keys unchanged and cannot affect a later successful block transform.
        assert!(multiplicity_free_permute_tree_pair_block(
            &rule,
            &malformed,
            &permutation,
            &domain,
        )
        .is_err());
        assert_eq!(malformed, snapshot);
        assert_eq!(
            multiplicity_free_permute_tree_pair_block(
                &rule,
                &valid,
                &permutation,
                &domain,
            )
            .unwrap(),
            baseline
        );
    }

    #[test]
    fn compact_repartition_preserves_source_major_error_precedence() {
        let codomain = |coupled, innerlines: &[SectorId]| {
            FusionTreeKey::new(
                [u1(1), u1(1), u1(1), u1(1)], coupled,
                [false; 4],
                innerlines.iter().copied(),
                [MultiplicityIndex::ONE; 3],
            )
        };
        let domain = |coupled| {
            FusionTreeKey::new(
                [u1(4)], coupled,
                [false],
                [],
                [],
            )
        };
        let sources = [
            FusionTreePairKey::pair(
                codomain(u1(4), &[u1(2)]),
                domain(u1(4)),
            ),
            FusionTreePairKey::pair(
                codomain(u1(4), &[]),
                domain(u1(99)),
            ),
        ];

        // What: the public categorical boundary rejects source 0's malformed
        // tree before compact bend-local validation can inspect later sources.
        assert_eq!(
            multiplicity_free_permute_tree_pair_block(
                &U1FusionRule,
                &sources,
                &[0, 1],
                &[4, 3, 2],
            )
            .unwrap_err(),
            CoreError::MalformedFusionTree {
                message: "fusion tree has an invalid number of innerlines",
            }
        );
    }

    trait TransposeOracleScalar {
        fn oracle_distance(&self, other: &Self) -> f64;
        fn oracle_magnitude(&self) -> f64;
    }

    impl TransposeOracleScalar for f64 {
        fn oracle_distance(&self, other: &Self) -> f64 {
            (self - other).abs()
        }

        fn oracle_magnitude(&self) -> f64 {
            self.abs()
        }
    }

    impl TransposeOracleScalar for Complex64 {
        fn oracle_distance(&self, other: &Self) -> f64 {
            (self - other).norm()
        }

        fn oracle_magnitude(&self) -> f64 {
            self.norm()
        }
    }

    fn assert_compact_transpose_matches_full_key_oracle<R>(
        rule: &R,
        sources: &[FusionTreePairKey],
        codomain_permutation: &[usize],
        domain_permutation: &[usize],
        expect_compact_boundary: bool,
    ) where
        R: MultiplicityFreeRigidSymbols,
        R::Scalar: Clone
            + Add<Output = R::Scalar>
            + Mul<Output = R::Scalar>
            + std::fmt::Debug
            + TransposeOracleScalar,
    {
        reset_compact_block_dimensions();
        let compact = multiplicity_free_transpose_tree_pair_block(
            rule,
            sources,
            codomain_permutation,
            domain_permutation,
        )
        .unwrap();
        let dimensions = compact_block_dimensions();
        let group = validate_tree_pair_block_group_for_rule(rule, sources)
            .unwrap()
            .expect("oracle cohort is nonempty");
        let prepared = PreparedTreePairOperation::prepare_transpose(
            group.codomain_rank,
            group.domain_rank,
            codomain_permutation,
            domain_permutation,
        )
        .unwrap();
        let ordered =
            multiplicity_free_transpose_tree_pair_block_ordered_validated(group, &prepared)
                .unwrap();
        let full_key = multiplicity_free_transpose_tree_pair_block_full_key_oracle(
            rule,
            sources,
            codomain_permutation,
            domain_permutation,
        )
        .unwrap();

        let mut expected_destinations = Vec::new();
        for source_rows in &compact {
            for (destination, _) in source_rows {
                if !expected_destinations.contains(destination) {
                    expected_destinations.push(destination.clone());
                }
            }
        }
        assert_eq!(ordered.destinations(), expected_destinations);
        assert_eq!(ordered.source_count(), sources.len());

        let mut ordered_coefficients =
            vec![None; ordered.destinations().len().saturating_mul(sources.len())];
        match ordered.storage() {
            OrderedBlockLinearStorage::SingletonColumns {
                destination_rows,
                coefficients,
            } => {
                assert_eq!(destination_rows.len(), sources.len());
                assert_eq!(coefficients.len(), sources.len());
                for (source, (&destination_row, coefficient)) in
                    destination_rows.iter().zip(coefficients).enumerate()
                {
                    ordered_coefficients[destination_row * sources.len() + source] =
                        Some(coefficient.clone());
                }
            }
            OrderedBlockLinearStorage::DenseDstSrc(coefficients) => {
                assert_eq!(coefficients.len(), ordered_coefficients.len());
                ordered_coefficients.clone_from_slice(coefficients);
            }
        }
        for destination_row in 0..ordered.destinations().len() {
            assert!(
                ordered_coefficients[destination_row * sources.len()
                    ..(destination_row + 1) * sources.len()]
                    .iter()
                    .any(Option::is_some),
                "ordered maps omit structurally empty destination rows"
            );
        }
        for (source, source_rows) in compact.iter().enumerate() {
            for (destination_row, destination) in ordered.destinations().iter().enumerate() {
                let expected = source_rows
                    .iter()
                    .find(|(candidate, _)| candidate == destination)
                    .map(|(_, coefficient)| coefficient);
                let actual =
                    ordered_coefficients[destination_row * sources.len() + source].as_ref();
                assert_eq!(actual.is_some(), expected.is_some());
                if let (Some(actual), Some(expected)) = (actual, expected) {
                    assert!(
                        actual.oracle_distance(expected)
                            <= 1.0e-12 * (1.0 + expected.oracle_magnitude()),
                        "ordered coefficient mismatch {expected:?} vs {actual:?}"
                    );
                }
            }
        }

        assert_eq!(compact.len(), full_key.len());
        for (compact_rows, full_key_rows) in compact.iter().zip(&full_key) {
            // What: compact execution preserves the legacy per-source
            // destination order and every categorical label.
            assert_eq!(
                compact_rows.iter().map(|(key, _)| key).collect::<Vec<_>>(),
                full_key_rows.iter().map(|(key, _)| key).collect::<Vec<_>>()
            );
            assert_eq!(compact_rows.len(), full_key_rows.len());
            for ((_, actual), (_, expected)) in compact_rows.iter().zip(full_key_rows) {
                assert!(
                    actual.oracle_distance(expected)
                        <= 1.0e-12 * (1.0 + expected.oracle_magnitude()),
                    "coefficient mismatch {expected:?} vs {actual:?}"
                );
            }
        }

        if expect_compact_boundary {
            let dimensions = dimensions.expect("nonidentity compact transform records dimensions");
            let destinations = full_key
                .iter()
                .flatten()
                .map(|(key, _)| key)
                .collect::<std::collections::BTreeSet<_>>();
            // What: the dense coefficient matrix is exactly the canonical
            // reachable block basis by the caller's source columns.
            assert_eq!(dimensions.destination_rows, destinations.len());
            assert_eq!(dimensions.source_columns, sources.len());
            assert_eq!(
                dimensions.coefficient_slots,
                dimensions.destination_rows * dimensions.source_columns
            );
            assert_eq!(
                dimensions.coefficient_bytes,
                dimensions.coefficient_slots * std::mem::size_of::<Option<R::Scalar>>()
            );
        } else {
            assert_eq!(dimensions, None);
        }
    }

    #[test]
    fn ordered_block_linear_storage_preserves_absent_and_zero_structure() {
        let rule = SU2FusionRule;
        let half = SectorLeg::new([(su2(1), 1)], false);
        let hom = FusionTreeHomSpace::new(
            FusionProductSpace::new([half.clone(), half.clone(), half]),
            FusionProductSpace::new([SectorLeg::new([(su2(1), 1)], false)]),
        );
        let keys = hom.fusion_tree_keys(&rule);
        assert_eq!(keys.len(), 2);
        let group = validate_tree_pair_block_group_for_rule(&rule, &keys)
            .unwrap()
            .expect("SU2 cohort is nonempty");

        let singleton_basis = CompactMultiplicityFreeTreePairBasis::from_group(group).unwrap();
        let mut singleton_columns = DenseColumns::with_capacity(2, 2);
        let row0 = singleton_columns.push_empty_row();
        let row1 = singleton_columns.push_empty_row();
        singleton_columns.row_mut(row0)[0] = Some(0.0);
        singleton_columns.row_mut(row1)[1] = Some(2.0);
        let singleton = order_compact_block(singleton_basis, singleton_columns);
        assert_eq!(singleton.destinations(), keys.as_ref());
        assert_eq!(
            singleton.storage(),
            &OrderedBlockLinearStorage::SingletonColumns {
                destination_rows: vec![0, 1],
                coefficients: vec![0.0, 2.0],
            }
        );

        let dense_basis = CompactMultiplicityFreeTreePairBasis::from_group(group).unwrap();
        let mut dense_columns = DenseColumns::with_capacity(2, 2);
        let row0 = dense_columns.push_empty_row();
        let row1 = dense_columns.push_empty_row();
        dense_columns.row_mut(row0)[0] = Some(0.0);
        dense_columns.row_mut(row1)[0] = Some(1.0);
        dense_columns.row_mut(row1)[1] = Some(2.0);
        let dense = order_compact_block(dense_basis, dense_columns);
        assert_eq!(dense.destinations(), keys.as_ref());
        assert_eq!(
            dense.storage(),
            &OrderedBlockLinearStorage::DenseDstSrc(vec![
                Some(0.0),
                None,
                Some(1.0),
                Some(2.0),
            ])
        );
    }

    #[test]
    fn ordered_transpose_preserves_custom_source_cohort_order() {
        let rule = FibonacciFAdmissibilityProbe::with_complex_f_phase();
        let tau = || SectorLeg::new([(SectorId::new(1), 1)], false);
        let hom = FusionTreeHomSpace::new(
            FusionProductSpace::new([tau(), tau()]),
            FusionProductSpace::new([tau(), tau()]),
        );
        let mut sources = hom.fusion_tree_keys(&rule).as_ref().to_vec();
        assert!(sources.len() > 1);
        sources.reverse();

        // What: a caller-selected source subset/order determines destination
        // first appearance; canonical HomSpace basis order is not substituted.
        assert_compact_transpose_matches_full_key_oracle(
            &rule,
            &sources,
            &[1, 3],
            &[0, 2],
            true,
        );
        assert_compact_transpose_matches_full_key_oracle(
            &rule,
            &sources[..2],
            &[1, 3],
            &[0, 2],
            true,
        );
    }

    #[test]
    fn transpose_tree_pair_block_matches_full_key_su2_cycles_and_repartition() {
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
        let mut blocks: BTreeMap<Vec<usize>, Vec<FusionTreePairKey>> = BTreeMap::new();
        for key in keys.iter() {
            let tag: Vec<usize> = key
                .codomain_tree()
                .uncoupled()
                .iter()
                .chain(key.domain_tree().uncoupled())
                .map(|s| s.id())
                .collect();
            blocks.entry(tag).or_default().push(key.clone());
        }

        let mut checked_blocks = 0;
        for (codomain_permutation, domain_permutation, uses_dense_block) in [
            (vec![3usize], vec![2usize, 1, 0], true),
            (vec![1usize, 2, 3], vec![0usize], true),
            (vec![0usize, 1], vec![3usize, 2], false),
            (vec![0usize, 1, 2, 3], vec![], false),
        ] {
            for src_keys in blocks.values() {
                assert_compact_transpose_matches_full_key_oracle(
                    &rule,
                    src_keys,
                    &codomain_permutation,
                    &domain_permutation,
                    uses_dense_block,
                );
                checked_blocks += 1;
            }
        }
        assert!(checked_blocks > 0, "expected at least one block");
    }

    #[test]
    fn transpose_tree_pair_block_matches_full_key_fermionic_product_cycle() {
        type FpU1Rule = ProductFusionRule<FermionParityFusionRule, U1FusionRule>;
        type ProductRule = ProductFusionRule<FpU1Rule, SU2FusionRule>;
        let left = FpU1Rule::default();
        let rule = ProductRule::default();
        let coupled = rule.encode_sector(left.encode_sector(z2_even(), u1(0)), su2(1));
        let odd_half = rule.encode_sector(left.encode_sector(z2_odd(), u1(1)), su2(1));
        let odd_one = rule.encode_sector(left.encode_sector(z2_odd(), u1(-1)), su2(2));
        let source = FusionTreePairKey::pair(
            FusionTreeKey::try_new_for_rule(
                &rule,
                [coupled],
                coupled,
                [false],
                [],
                [],
            )
            .unwrap(),
            FusionTreeKey::try_new_for_rule(
                &rule,
                [odd_half, odd_one],
                coupled,
                [false, true],
                [],
                [MultiplicityIndex::ONE],
            )
            .unwrap(),
        );

        // What: nested fZ2 x U1 x SU2 keeps the fermionic pivotal phase,
        // non-self-dual charge, and non-Abelian channel through a cycle.
        assert_compact_transpose_matches_full_key_oracle(
            &rule,
            &[source],
            &[2, 1],
            &[0],
            true,
        );
    }

    #[test]
    fn transpose_tree_pair_block_matches_low_rank_and_nonselfdual_u1_oracles() {
        let empty = FusionTreeKey::new([], u1(0), [], [], []);
        let rank_zero = [FusionTreePairKey::pair(empty.clone(), empty)];
        // What: scalar transpose remains the symbol-free identity operation.
        assert_compact_transpose_matches_full_key_oracle(
            &U1FusionRule,
            &rank_zero,
            &[],
            &[],
            false,
        );

        let rank_one_hom = FusionTreeHomSpace::new(
            FusionProductSpace::new([SectorLeg::new([(u1(0), 1)], false)]),
            FusionProductSpace::new([]),
        );
        let rank_one = rank_one_hom.fusion_tree_keys(&U1FusionRule);
        assert_eq!(rank_one.len(), 1);
        // What: moving one vacuum leg across the partition uses only the final
        // full-key reconstruction boundary.
        assert_compact_transpose_matches_full_key_oracle(
            &U1FusionRule,
            &rank_one,
            &[],
            &[0],
            false,
        );

        // What: a non-self-dual U1 cycle preserves sector dualization and flags.
        assert_compact_transpose_matches_full_key_oracle(
            &U1FusionRule,
            &[u1_nonselfdual_tree_pair_fixture()],
            &[1, 2],
            &[0],
            true,
        );
    }

    #[test]
    fn transpose_tree_pair_block_matches_fz2_odd_pivotal_oracle() {
        let hom = FusionTreeHomSpace::new(
            FusionProductSpace::new([
                SectorLeg::new([(z2_odd(), 1)], true),
                SectorLeg::new([(z2_odd(), 1)], false),
            ]),
            FusionProductSpace::new([SectorLeg::new([(z2_even(), 1)], false)]),
        );
        let sources = hom.fusion_tree_keys(&FermionParityFusionRule);
        assert!(!sources.is_empty());

        // What: cycling a dual odd leg retains the Frobenius-Schur/pivotal sign.
        assert_compact_transpose_matches_full_key_oracle(
            &FermionParityFusionRule,
            &sources,
            &[1, 2],
            &[0],
            true,
        );
    }

    #[test]
    fn transpose_tree_pair_block_matches_complex_f_oracle() {
        let rule = FibonacciFAdmissibilityProbe::with_complex_f_phase();
        let tau = || SectorLeg::new([(SectorId::new(1), 1)], false);
        let hom = FusionTreeHomSpace::new(
            FusionProductSpace::new([tau(), tau()]),
            FusionProductSpace::new([tau(), tau()]),
        );
        let sources = hom.fusion_tree_keys(&rule);
        assert!(sources.len() > 1);
        for (codomain_permutation, domain_permutation, expected_direction) in [
            (
                [1usize, 3],
                [0usize, 2],
                PreparedCycleDirection::Clockwise,
            ),
            (
                [2usize, 0],
                [3usize, 1],
                PreparedCycleDirection::Anticlockwise,
            ),
        ] {
            let prepared = PreparedTreePairOperation::prepare_transpose(
                2,
                2,
                &codomain_permutation,
                &domain_permutation,
            )
            .unwrap();
            assert!(matches!(
                prepared.plan,
                PreparedTreePairPlan::Transpose { direction, .. }
                    if direction == expected_direction
            ));
            let rows = multiplicity_free_transpose_tree_pair_block(
                &rule,
                &sources,
                &codomain_permutation,
                &domain_permutation,
            )
            .unwrap();
            assert!(rows
                .iter()
                .flatten()
                .any(|(_, coefficient)| coefficient.im.abs() > 1.0e-12));

            // What: both compact cycle directions preserve ordered multi-row
            // non-real F products and conjugation against the old full keys.
            assert_compact_transpose_matches_full_key_oracle(
                &rule,
                &sources,
                &codomain_permutation,
                &domain_permutation,
                true,
            );
        }
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
        assert_eq!(keys[0].coupled(), minus_one.into());
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
        assert_eq!(keys[0].coupled(), SectorId::new(1));
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
        let first = BlockKey::from(FusionTreePairKey::try_pair_from_sector_ids(
            [10, 20],
            [30], 5,
            [false, true],
            [true],
            [101],
            [201],
            [301, 302],
            [401],
        ).unwrap());
        let second = BlockKey::from(FusionTreePairKey::try_pair_from_sector_ids(
            [1],
            [2, 3], 4,
            [true],
            [false, true],
            [],
            [202],
            [303],
            [402, 403],
        ).unwrap());
        let same_group_as_first = BlockKey::from(FusionTreePairKey::try_pair_from_sector_ids(
            [10, 20],
            [30], 6,
            [false, true],
            [true],
            [102],
            [203],
            [304, 305],
            [404],
        ).unwrap());

        let keys = vec![first.clone(), second.clone(), same_group_as_first.clone()];
        let sector = SectorStructure::from_keys(2, keys.clone()).unwrap();
        let block_structure =
            packed_fixture_structure(2, keys.into_iter().map(|key| (key, vec![1, 1]))).unwrap();

        let sector_groups = sector.fusion_tree_groups();
        let block_groups = block_structure.fusion_tree_groups();
        assert_eq!(sector_groups, block_groups);
        let mut legacy_groups = Vec::<FusionTreeBlockGroup>::new();
        for (index, key) in [first, second, same_group_as_first].iter().enumerate() {
            let group_key = key.fusion_tree_group_key().unwrap();
            if let Some(group) = legacy_groups
                .iter_mut()
                .find(|group| group.group_key() == &group_key)
            {
                group.block_indices.push(index);
            } else {
                legacy_groups.push(FusionTreeBlockGroup::new(group_key, vec![index]));
            }
        }
        // What: construction-time metadata is exactly the former eager
        // first-appearance grouping, including interleaved storage indices.
        assert_eq!(sector_groups, legacy_groups);
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
    fn sector_structure_rejects_every_mixed_key_kind_pair() {
        let fusion = BlockKey::from(FusionTreePairKey::try_pair_from_sector_ids(
            [7],
            [8], 9,
            [false],
            [true],
            [],
            [],
            [],
            [],
        ).unwrap());
        let keys = [
            BlockKey::trivial(),
            BlockKey::opaque([7]),
            fusion,
        ];
        for expected in 0..keys.len() {
            for actual in 0..keys.len() {
                if expected == actual {
                    continue;
                }
                assert_eq!(
                    SectorStructure::from_keys(
                        2,
                        [keys[expected].clone(), keys[actual].clone()]
                    )
                    .unwrap_err(),
                    CoreError::MixedBlockKeyKinds {
                        expected: keys[expected].kind(),
                        actual: keys[actual].kind(),
                    }
                );
            }
        }

        let dense = BlockStructure::trivial(&[2, 3]).unwrap();
        let empty = BlockStructure::empty(2);
        assert!(dense.fusion_tree_groups().is_empty());
        assert!(empty.fusion_tree_groups().is_empty());
    }

    #[test]
    fn opaque_block_key_words_are_rank_independent_application_identity() {
        let key = OpaqueBlockKey::from_words([3, 5, 8, 13, 21]);
        let block_key = BlockKey::from(key.clone());
        let rank_one = SectorStructure::from_keys(1, [block_key.clone()]).unwrap();
        let rank_seven = SectorStructure::from_keys(7, [block_key.clone()]).unwrap();

        // What: opaque word count has no relationship to tensor rank, and the
        // public constructors preserve all application routing words.
        assert_eq!(key.words(), &[3, 5, 8, 13, 21]);
        assert_eq!(OpaqueBlockKey::new(vec![3, 5, 8, 13, 21]), key);
        assert_eq!(rank_one.key(0).unwrap(), &block_key);
        assert_eq!(rank_seven.key(0).unwrap(), &block_key);
        assert_eq!(OpaqueBlockKey::ordinal(34).words(), &[34]);
    }

    #[allow(deprecated)]
    #[test]
    fn deprecated_sector_key_constructors_are_opaque_compatibility_helpers() {
        let from_sectors = BlockKey::sectors([SectorId::new(3), SectorId::new(5)]);
        let from_ids = BlockKey::sector_ids([3, 5]);

        // What: legacy numeric block labels preserve routing identity without
        // being promoted to categorical fusion-tree pairs.
        assert_eq!(from_sectors, BlockKey::opaque([3, 5]));
        assert_eq!(from_ids, BlockKey::opaque([3, 5]));
        assert!(matches!(from_sectors, BlockKey::Opaque(_)));
        assert!(matches!(from_ids, BlockKey::Opaque(_)));
    }

    #[test]
    fn empty_sector_structure_has_one_canonical_namespace_free_form() {
        let constructed =
            SectorStructure::from_keys(3, std::iter::empty::<BlockKey>()).unwrap();
        let canonical = SectorStructure::empty(3);

        // What: both public empty constructors produce the same namespace-free
        // structure and never allocate a meaningless compact lookup.
        assert_eq!(constructed, canonical);
        assert_eq!(constructed.key_kind(), None);
        assert!(!constructed.has_compact_lookup());
    }

    #[test]
    fn block_structure_separates_sector_and_degeneracy_data() {
        let sector = SectorStructure::from_keys(
            2,
            [BlockKey::opaque([0, 1]), BlockKey::opaque([1, 0])],
        )
        .unwrap();
        let degeneracy =
            DegeneracyStructure::packed_column_major(2, [vec![2, 3], vec![3, 2]]).unwrap();
        let structure = BlockStructure::from_parts(sector, degeneracy).unwrap();

        assert_eq!(structure.rank(), 2);
        assert_eq!(
            structure.sector_structure().key(0).unwrap(),
            &BlockKey::opaque([0, 1])
        );
        assert_eq!(
            structure.sector_structure().key(1).unwrap(),
            &BlockKey::opaque([1, 0])
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
                BlockKey::opaque([2]),
                BlockKey::opaque([0]),
                BlockKey::opaque([1]),
            ],
        )
        .unwrap();
        let src = SectorStructure::from_keys(
            2,
            [
                BlockKey::opaque([0]),
                BlockKey::opaque([1]),
                BlockKey::opaque([2]),
            ],
        )
        .unwrap();

        assert!(src.has_compact_lookup());
        assert_eq!(dst.find_index(&BlockKey::opaque([0])), Some(1));
        assert_eq!(src.find_index(&BlockKey::opaque([2])), Some(2));
        assert_eq!(dst.pair_indices_from(&src).unwrap(), vec![2, 0, 1]);
    }

    #[test]
    fn lookup_never_aliases_different_key_namespaces() {
        let dense = SectorStructure::dense(0);
        let opaque_zero = SectorStructure::from_keys(0, [BlockKey::ordinal(0)]).unwrap();
        let fusion_pair = FusionTreePairKey::try_pair_from_sector_ids(
            [0],
            [], 0,
            [false],
            [],
            [],
            [],
            [],
            [],
        ).unwrap();
        let fusion = SectorStructure::from_keys(1, [fusion_pair.clone()]).unwrap();
        let opaque_one = SectorStructure::from_keys(1, [BlockKey::ordinal(1)]).unwrap();

        // What: compact integer routing is only an accelerator inside the
        // Dense/Opaque namespaces and never establishes categorical identity.
        assert_eq!(dense.find_index(&BlockKey::ordinal(0)), None);
        assert_eq!(opaque_zero.find_index(&BlockKey::Dense), None);
        assert_eq!(
            opaque_one.find_index(&BlockKey::from(fusion_pair.clone())),
            None
        );
        assert_eq!(
            opaque_one.find_fusion_tree_pair_index(&fusion_pair),
            None
        );
        assert!(!fusion.has_compact_lookup());
        assert_eq!(
            opaque_one.pair_indices_from(&fusion),
            Err(CoreError::MixedBlockKeyKinds {
                expected: BlockKeyKind::Opaque,
                actual: BlockKeyKind::FusionTree,
            })
        );
    }

    #[allow(deprecated)]
    #[test]
    fn deprecated_fusion_tree_lookup_forwarders_match_canonical_results() {
        let present = FusionTreePairKey::try_pair_from_sector_ids(
            [1],
            [], 1,
            [false],
            [],
            [],
            [],
            [],
            [],
        ).unwrap();
        let missing = FusionTreePairKey::try_pair_from_sector_ids(
            [2],
            [], 2,
            [false],
            [],
            [],
            [],
            [],
            [],
        ).unwrap();
        let structure = BlockStructure::from_blocks(vec![
            BlockSpec::column_major_with_key(present.clone().into(), vec![1], 0).unwrap(),
        ])
        .unwrap();
        let sectors = structure.sector_structure();

        // What: all three renamed lookup/block APIs retain exact success and
        // missing behavior through their deprecated forwarding layer.
        assert_eq!(
            sectors.find_fusion_tree_index(&present),
            sectors.find_fusion_tree_pair_index(&present)
        );
        assert_eq!(
            sectors.find_fusion_tree_index(&missing),
            sectors.find_fusion_tree_pair_index(&missing)
        );
        assert_eq!(
            structure.find_block_index_by_fusion_tree_key(&present),
            structure.find_block_index_by_fusion_tree_pair(&present)
        );
        assert_eq!(
            structure.find_block_index_by_fusion_tree_key(&missing),
            structure.find_block_index_by_fusion_tree_pair(&missing)
        );
        assert_eq!(
            structure.fusion_tree_block(&present).unwrap().key(),
            structure.fusion_tree_pair_block(&present).unwrap().key()
        );
        assert_eq!(
            structure.fusion_tree_block(&missing).unwrap_err(),
            structure.fusion_tree_pair_block(&missing).unwrap_err()
        );
    }

    #[test]
    fn sector_structure_pairs_general_opaque_keys_by_sorted_merge() {
        let key_a = BlockKey::opaque([0, 1]);
        let key_b = BlockKey::opaque([1, 0]);
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

    #[derive(Debug)]
    struct AdversarialHostStorage<T> {
        data: Vec<T>,
        reported_len: std::cell::Cell<usize>,
    }

    impl<T> TensorStorage<T> for AdversarialHostStorage<T> {
        fn len(&self) -> usize {
            self.reported_len.get()
        }

        fn placement(&self) -> Placement {
            Placement::Host
        }
    }

    impl<T> HostReadableStorage<T> for AdversarialHostStorage<T> {
        fn as_slice(&self) -> &[T] {
            &self.data
        }
    }

    impl<T> HostWritableStorage<T> for AdversarialHostStorage<T> {
        fn as_mut_slice(&mut self) -> &mut [T] {
            &mut self.data
        }
    }

    type AdversarialHostTensor =
        TensorMap<i32, 1, 0, Trivial, AdversarialHostStorage<i32>>;

    fn adversarial_host_tensor(actual_len: usize) -> AdversarialHostTensor {
        let space = TensorMapSpace::<1, 0>::from_dims([2], []).unwrap();
        let storage = AdversarialHostStorage {
            data: (0..actual_len).map(|value| value as i32 + 10).collect(),
            reported_len: std::cell::Cell::new(2),
        };
        AdversarialHostTensor::from_storage_with_structure(
            storage,
            space,
            BlockStructure::packed_column_major(1, [vec![2]]).unwrap(),
        )
        .unwrap()
    }

    fn assert_host_execution_rejects_extent(
        mut tensor: AdversarialHostTensor,
        error: CoreError,
    ) {
        let before = tensor.data().to_vec();
        let visits = std::cell::Cell::new(0);
        assert_eq!(
            tensor.for_each_block_element(|_, _, _| visits.set(visits.get() + 1)),
            Err(error.clone())
        );
        assert_eq!(visits.get(), 0);

        let fills = std::cell::Cell::new(0);
        assert_eq!(
            tensor.fill_block_elements(|_, _| {
                fills.set(fills.get() + 1);
                -1
            }),
            Err(error.clone())
        );
        assert_eq!(fills.get(), 0);
        assert_eq!(tensor.data(), before);

        assert_eq!(tensor.subblock().unwrap_err(), error);
        assert_eq!(tensor.block(0).unwrap_err(), error);
        assert_eq!(
            tensor.block_by_key(&BlockKey::ordinal(0)).unwrap_err(),
            error
        );
        assert_eq!(tensor.subblock_mut().unwrap_err(), error);
        assert_eq!(tensor.block_mut(0).unwrap_err(), error);
        assert_eq!(
            tensor
                .block_mut_by_key(&BlockKey::ordinal(0))
                .unwrap_err(),
            error
        );
        assert_eq!(tensor.data(), before);
    }

    #[test]
    fn host_execution_rejects_reported_extent_changes_before_callbacks_or_writes() {
        // What: safe interior mutability in external storage cannot make a
        // short or oversized reported extent reach callbacks, views, or writes.
        for reported_len in [1, 3] {
            let tensor = adversarial_host_tensor(2);
            tensor.storage().reported_len.set(reported_len);
            assert_host_execution_rejects_extent(
                tensor,
                CoreError::DimensionMismatch {
                    expected: 2,
                    actual: reported_len,
                },
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

    fn adversarial_fusion_host_tensor(
        actual_len: usize,
    ) -> (
        TensorMap<i32, 1, 1, Trivial, AdversarialHostStorage<i32>>,
        FusionTreePairKey,
    ) {
        let rule = Z2FusionRule;
        let fusion_space = FusionTensorMapSpace::from_degeneracy_shapes(
            TensorMapSpace::<1, 1>::from_dims([1], [1]).unwrap(),
            FusionTreeHomSpace::from_sectors(
                [(Z2Irrep::EVEN, 1)],
                [(Z2Irrep::EVEN, 1)],
            ),
            &rule,
            [vec![1, 1]],
        )
        .unwrap();
        let key = fusion_space.homspace().fusion_tree_keys(&rule)[0].clone();
        let tensor = TensorMap::from_storage_with_fusion_space(
            AdversarialHostStorage {
                data: vec![10; actual_len],
                reported_len: std::cell::Cell::new(1),
            },
            fusion_space,
        )
        .unwrap();
        (tensor, key)
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
    fn fusion_sector_getters_report_sector_metadata_errors_before_storage_extent() {
        // What: immutable sector getters preserve the mutable getter's error
        // precedence when both the sector tuple and host slice are malformed.
        let sectors = [Z2Irrep::EVEN.sector_id()];
        let metadata_error = CoreError::DimensionMismatch {
            expected: 2,
            actual: 1,
        };

        for actual_len in [0, 2] {
            let (mut tensor, _) = adversarial_fusion_host_tensor(actual_len);

            assert_eq!(
                tensor
                    .subblocks_by_sectors(&Z2FusionRule, &sectors)
                    .unwrap_err(),
                metadata_error
            );
            assert_eq!(
                tensor
                    .subblock_by_sectors(&Z2FusionRule, &sectors)
                    .unwrap_err(),
                metadata_error
            );
            assert_eq!(
                tensor
                    .subblock_mut_by_sectors(&Z2FusionRule, &sectors)
                    .unwrap_err(),
                metadata_error
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
            [half, half, one], one,
            [false, false, true],
            [SectorId::new(0)],
            [MultiplicityIndex::ONE, MultiplicityIndex::ONE],
        );

        let (front, tail) = split_fusion_tree(&rule, &tree, 2).unwrap();

        assert_eq!(front.uncoupled(), &[half, half]);
        assert_eq!(front.coupled(), SectorId::new(0));
        assert_eq!(front.is_dual(), &[false, false]);
        assert_eq!(front.innerlines(), &[]);
        assert_eq!(front.vertices(), &[MultiplicityIndex::ONE]);
        assert_eq!(tail.uncoupled(), &[SectorId::new(0), one]);
        assert_eq!(tail.coupled(), one);
        assert_eq!(tail.is_dual(), &[false, true]);
        assert_eq!(tail.innerlines(), &[]);
        assert_eq!(tail.vertices(), &[MultiplicityIndex::ONE]);
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

    // --- Stage A: outer-multiplicity (Generic fusion) foundation ----------
    //
    // `ToyOmRule` is purely synthetic (following `AsymmetricAnyonicRule`
    // above as a template): sector 1 ("a") fuses with itself to sector 3
    // ("c") via two independent channels, i.e. N(a,a,c) = 2. It exists only
    // to exercise the `GenericFusionSymbols` wiring and provider-owned
    // `FusionStyleKind::Generic` gate added in this stage.
    // Pentagon/hexagon are NOT required to hold — see
    // scratchpad/toy-om-stageA-plan.md, this is wiring validation only, not
    // a physical anyon model. The recoupling engine (recouple wrapper,
    // `UnsupportedFusionStyle` guards) does not consume this rule; that is
    // explicitly Stage B.
    #[derive(Clone, Copy, Debug)]
    struct ToyOmRule;

    impl ToyOmRule {
        const VACUUM: usize = 0;
        const A: usize = 1;
        const C: usize = 3;
    }

    impl FusionRule for ToyOmRule {
        fn rule_identity(&self) -> RuleIdentity { RuleIdentity::of_type::<Self>() }
        fn fusion_style(&self) -> FusionStyleKind {
            FusionStyleKind::Generic
        }

        fn braiding_style(&self) -> BraidingStyleKind {
            BraidingStyleKind::Bosonic
        }

        fn vacuum(&self) -> SectorId {
            SectorId::new(Self::VACUUM)
        }

        fn fusion_channels(&self, left: SectorId, right: SectorId) -> SectorVec {
            match (left.id(), right.id()) {
                (Self::VACUUM, x) | (x, Self::VACUUM) => smallvec![SectorId::new(x)],
                (Self::A, Self::A) => smallvec![SectorId::new(Self::C)],
                (Self::A, Self::C) | (Self::C, Self::A) => smallvec![SectorId::new(Self::A)],
                _ => smallvec![SectorId::new(Self::VACUUM)],
            }
        }

        fn nsymbol(&self, left: SectorId, right: SectorId, coupled: SectorId) -> usize {
            // The one artificial outer multiplicity this toy rule carries:
            // a (x) a -> c has two independent fusion channels (N=2). Every
            // other triple falls back to the multiplicity-free default (0
            // or 1, from whether `coupled` is a fusion channel of
            // `left (x) right`) — this is the override the design doc calls
            // for instead of trying to encode multiplicity through repeated
            // `fusion_channels` entries.
            if (left.id(), right.id(), coupled.id()) == (Self::A, Self::A, Self::C) {
                2
            } else {
                usize::from(self.fusion_channels(left, right).contains(&coupled))
            }
        }
    }

    impl GenericFusionSymbols for ToyOmRule {
        type Scalar = f64;

        fn f_symbol_generic(
            &self,
            a: SectorId,
            b: SectorId,
            c: SectorId,
            d: SectorId,
            e: SectorId,
            f: SectorId,
        ) -> GenericFArray<Self::Scalar> {
            let n_mu = self.nsymbol(a, b, e);
            let n_nu = self.nsymbol(e, c, d);
            let n_kappa = self.nsymbol(b, c, f);
            let n_lambda = self.nsymbol(a, f, d);
            let ids = (a.id(), b.id(), c.id(), d.id(), e.id(), f.id());
            if ids == (Self::A, Self::A, Self::VACUUM, Self::C, Self::C, Self::A) {
                // mu and lambda both range over the N(a,a,c)=2 channel here
                // (nu = N(c,0,c) = 1, kappa = N(a,0,a) = 1 are trivial), so
                // this F-block is genuinely a 2x2 matrix. Filled with a
                // pi/4 rotation — an actual orthogonal matrix, not just
                // shape-correct filler — because the design doc wants F's
                // (mu, lambda) block orthogonal so a later (Stage B)
                // braid * inverse == identity self-consistency check has
                // something real to check.
                let s = std::f64::consts::FRAC_1_SQRT_2;
                GenericFArray::new(vec![s, -s, s, s], (n_mu, n_nu, n_kappa, n_lambda))
            } else {
                let total = n_mu * n_nu * n_kappa * n_lambda;
                GenericFArray::new(vec![1.0; total], (n_mu, n_nu, n_kappa, n_lambda))
            }
        }

        fn r_symbol_generic(
            &self,
            a: SectorId,
            b: SectorId,
            c: SectorId,
        ) -> GenericRMatrix<Self::Scalar> {
            let rows = self.nsymbol(a, b, c);
            let cols = self.nsymbol(b, a, c);
            if (a.id(), b.id(), c.id()) == (Self::A, Self::A, Self::C) {
                GenericRMatrix::new(vec![1.0, 0.0, 0.0, 1.0], rows, cols)
            } else {
                GenericRMatrix::new(vec![1.0; rows * cols], rows, cols)
            }
        }
    }

    #[test]
    fn toy_om_rule_nsymbol_reports_outer_multiplicity() {
        let rule = ToyOmRule;
        let a = SectorId::new(ToyOmRule::A);
        let c = SectorId::new(ToyOmRule::C);
        let vacuum = SectorId::new(ToyOmRule::VACUUM);

        assert_eq!(rule.fusion_style(), FusionStyleKind::Generic);
        assert_eq!(rule.nsymbol(a, a, c), 2);
        // Everything else in this toy rule stays multiplicity-free (0 or 1).
        assert_eq!(rule.nsymbol(a, vacuum, a), 1);
        assert_eq!(rule.nsymbol(a, a, vacuum), 0);
    }

    #[test]
    fn generic_split_preserves_multiplicity_style_and_vertex_labels() {
        // What: splitting a valid Generic tree returns two keys that remain
        // valid for the same rule and retain the selected multiplicity basis.
        let rule = ToyOmRule;
        let a = SectorId::new(ToyOmRule::A);
        let c = SectorId::new(ToyOmRule::C);
        let source = FusionTreeKey::try_new_for_rule(
            &rule,
            [a, a, a], a,
            [false; 3],
            [c],
            [MultiplicityIndex::new(2).expect("test multiplicity label is one-based"), MultiplicityIndex::ONE],
        )
        .unwrap();

        let (front, tail) = split_fusion_tree(&rule, &source, 2).unwrap();

        front.validate_for_rule(&rule).unwrap();
        tail.validate_for_rule(&rule).unwrap();
        assert_eq!(front.vertices(), &[MultiplicityIndex::new(2).unwrap()]);
        assert_eq!(tail.vertices(), &[MultiplicityIndex::ONE]);
    }

    #[test]
    fn generic_identity_braid_rejects_an_inadmissible_source() {
        // What: the Generic identity shortcut is behind the same categorical
        // boundary as multiplicity-free tree operations.
        let a = SectorId::new(ToyOmRule::A);
        let c = SectorId::new(ToyOmRule::C);
        let invalid = FusionTreeKey::new(
            [a; 3], c,
            [false; 3],
            [c],
            [MultiplicityIndex::ONE; 2],
        );

        assert_eq!(
            generic_braid_tree(&ToyOmRule, &invalid, &[0, 1, 2], &[0, 1, 2])
                .unwrap_err(),
            CoreError::MalformedFusionTree {
                message: "fusion tree contains an inadmissible fusion vertex",
            }
        );
    }

    #[derive(Debug, Default)]
    struct CrossIncompatibleGenericRule {
        f_calls: std::sync::atomic::AtomicUsize,
    }

    impl CrossIncompatibleGenericRule {
        const VACUUM: usize = 0;
        const A: usize = 1;
        const B: usize = 2;
        const C: usize = 3;
        const X: usize = 4;
        const E: usize = 5;
        const D: usize = 6;
        const F: usize = 7;
        const G: usize = 8;
        const Q: usize = 9;
    }

    impl FusionRule for CrossIncompatibleGenericRule {
        fn rule_identity(&self) -> RuleIdentity {
            RuleIdentity::of_type::<Self>()
        }

        fn fusion_style(&self) -> FusionStyleKind {
            FusionStyleKind::Generic
        }

        fn braiding_style(&self) -> BraidingStyleKind {
            BraidingStyleKind::Bosonic
        }

        fn vacuum(&self) -> SectorId {
            SectorId::new(Self::VACUUM)
        }

        fn fusion_channels(&self, left: SectorId, right: SectorId) -> SectorVec {
            let channel = match (left.id(), right.id()) {
                (Self::VACUUM, sector) | (sector, Self::VACUUM) => Some(sector),
                (Self::A, Self::B) => Some(Self::E),
                (Self::E, Self::C) => Some(Self::D),
                (Self::D, Self::X) => Some(Self::Q),
                (Self::B, Self::C) => Some(Self::F),
                (Self::F, Self::X) => Some(Self::G),
                (Self::G, Self::X) => Some(Self::F),
                (Self::A, Self::G) => Some(Self::Q),
                (Self::A, Self::Q) => Some(Self::G),
                (Self::Q, Self::X) => Some(Self::D),
                (Self::D, Self::C) => Some(Self::E),
                _ => None,
            };
            channel
                .map(|sector| smallvec![SectorId::new(sector)])
                .unwrap_or_default()
        }
    }

    impl GenericFusionSymbols for CrossIncompatibleGenericRule {
        type Scalar = f64;

        fn f_symbol_generic(
            &self,
            _a: SectorId,
            _b: SectorId,
            _c: SectorId,
            _d: SectorId,
            _e: SectorId,
            _f: SectorId,
        ) -> GenericFArray<Self::Scalar> {
            self.f_calls
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            GenericFArray::new(vec![997.0], (1, 1, 1, 1))
        }

        fn r_symbol_generic(
            &self,
            a: SectorId,
            b: SectorId,
            c: SectorId,
        ) -> GenericRMatrix<Self::Scalar> {
            let rows = self.nsymbol(a, b, c);
            let cols = self.nsymbol(b, a, c);
            GenericRMatrix::new(vec![1.0; rows * cols], rows, cols)
        }
    }

    #[test]
    fn generic_multi_fmove_filters_cross_incompatible_trees_before_f() {
        // What: forward and inverse multi-F moves discard individually valid
        // long/short trees whose cross-tree F vertex does not exist, even when
        // the provider returns a nonzero sentinel outside its valid domain.
        let rule = CrossIncompatibleGenericRule::default();
        let sector = SectorId::new;
        let long = FusionTreeKey::new(
            [
                sector(CrossIncompatibleGenericRule::A),
                sector(CrossIncompatibleGenericRule::B),
                sector(CrossIncompatibleGenericRule::C),
                sector(CrossIncompatibleGenericRule::X),
            ], sector(CrossIncompatibleGenericRule::Q),
            [false; 4],
            [
                sector(CrossIncompatibleGenericRule::E),
                sector(CrossIncompatibleGenericRule::D),
            ],
            [MultiplicityIndex::ONE; 3],
        );
        let short = FusionTreeKey::new(
            [
                sector(CrossIncompatibleGenericRule::B),
                sector(CrossIncompatibleGenericRule::C),
                sector(CrossIncompatibleGenericRule::X),
            ], sector(CrossIncompatibleGenericRule::G),
            [false; 3],
            [sector(CrossIncompatibleGenericRule::F)],
            [MultiplicityIndex::ONE; 2],
        );
        long.validate_for_rule(&rule).unwrap();
        short.validate_for_rule(&rule).unwrap();

        assert!(generic_multi_fmove_tree(&rule, &long).unwrap().is_empty());
        assert!(generic_multi_fmove_inv_tree(
            &rule,
            sector(CrossIncompatibleGenericRule::A),
            sector(CrossIncompatibleGenericRule::Q),
            &short,
            false,
        )
        .unwrap()
        .is_empty());
        assert_eq!(
            rule.f_calls.load(std::sync::atomic::Ordering::Relaxed),
            0
        );
    }

    #[test]
    fn toy_om_rule_f_symbol_generic_has_nsymbol_shaped_block() {
        let rule = ToyOmRule;
        let a = SectorId::new(ToyOmRule::A);
        let c = SectorId::new(ToyOmRule::C);
        let vacuum = SectorId::new(ToyOmRule::VACUUM);

        // F(a,a,0,c,c,a): shape (N(a,a,c), N(c,0,c), N(a,0,a), N(a,a,c))
        // = (2, 1, 1, 2).
        let f = rule.f_symbol_generic(a, a, vacuum, c, c, a);
        assert_eq!(f.shape(), (2, 1, 1, 2));
        assert_eq!(f.data().len(), 4);

        // The (mu, lambda) 2x2 block (nu = kappa = 0) is an orthogonal
        // rotation: R^T R == I.
        let m = [
            [*f.get(0, 0, 0, 0), *f.get(0, 0, 0, 1)],
            [*f.get(1, 0, 0, 0), *f.get(1, 0, 0, 1)],
        ];
        for col in 0..2 {
            for other in 0..2 {
                let dot: f64 = (0..2).map(|row| m[row][col] * m[row][other]).sum();
                let expected = if col == other { 1.0 } else { 0.0 };
                assert!(
                    (dot - expected).abs() < 1e-12,
                    "F (mu,lambda) block is not orthogonal at columns {col},{other}: {dot}"
                );
            }
        }
    }

    #[test]
    fn toy_om_rule_r_symbol_generic_has_nsymbol_shaped_matrix() {
        let rule = ToyOmRule;
        let a = SectorId::new(ToyOmRule::A);
        let c = SectorId::new(ToyOmRule::C);

        let r = rule.r_symbol_generic(a, a, c);
        assert_eq!(r.shape(), (2, 2));
        assert_eq!(r.data().len(), 4);
    }

    fn generic_tree_pair(vertices_first: usize, vertices_second: usize) -> (FusionTreeKey, FusionTreeKey) {
        let a = SectorId::new(ToyOmRule::A);
        let c = SectorId::new(ToyOmRule::C);
        let first = FusionTreeKey::new([a, a], c, [false, false], [], [MultiplicityIndex::new(vertices_first).expect("test multiplicity label is one-based")]);
        let second = FusionTreeKey::new([a, a], c, [false, false], [], [MultiplicityIndex::new(vertices_second).expect("test multiplicity label is one-based")]);
        (first, second)
    }

    #[test]
    fn fusion_tree_key_generic_distinguishes_vertices() {
        // What: multiplicity vertices are always part of categorical identity;
        // fusion style comes from the provider rather than a duplicated flag.
        let (first, second) = generic_tree_pair(1, 2);
        assert_ne!(first, second);
        assert_ne!(first.cmp(&second), std::cmp::Ordering::Equal);

        let mut set = std::collections::HashSet::new();
        set.insert(first);
        set.insert(second);
        assert_eq!(set.len(), 2, "Generic keys differing in vertices must stay distinct");
    }

    #[test]
    fn tree_pair_axis_validation_remains_linear_above_inline_bitset_capacity() {
        let rule = Z2FusionRule;
        let rank = 129;
        let codomain = FusionTreeKey::try_new_for_rule(
            &rule,
            vec![SectorId::new(0); rank], SectorId::new(0),
            vec![false; rank],
            vec![SectorId::new(0); rank - 2],
            vec![MultiplicityIndex::ONE; rank - 1],
        )
        .unwrap();
        let domain =
            FusionTreeKey::try_new_for_rule(&rule, [], SectorId::new(0), [], [], [])
                .unwrap();
        let pair = FusionTreePairKey::pair(codomain, domain);
        let identity = (0..rank).collect::<Vec<_>>();

        let rows =
            multiplicity_free_permute_tree_pair(&rule, &pair, &identity, &[]).unwrap();
        assert_eq!(rows, vec![(pair.clone(), 1.0)]);

        let mut duplicate = identity;
        duplicate[rank - 1] = rank - 2;
        assert!(matches!(
            multiplicity_free_permute_tree_pair(&rule, &pair, &duplicate, &[]),
            Err(CoreError::InvalidPermutation { .. })
        ));
    }

    // --- Stage B1: Generic-fusion Artin braid (braid × inverse == identity) --
    //
    // `UnitaryToyOmRule` is a *new* rule (pure addition — Stage A's
    // `ToyOmRule` is left byte-for-byte untouched) with the same fusion
    // structure (a⊗a→c has N=2, everything else N≤1, mixing an OM vertex with
    // multiplicity-1 vertices), but with UNITARY F/R blocks so that TensorKit's
    // inverse-braid identity actually holds:
    //
    //   * R(a,a,c) is a genuine 2×2 rotation (unitary), so the braid mixes the
    //     two outer-multiplicity channels non-trivially — the round-trip test
    //     is real, not vacuous.
    //   * F(a,a,a,a,c,c) — the only F block the rank-3 braid touches — is the
    //     2×2 IDENTITY. This is deliberate: the toy rule is not
    //     hexagon-consistent, and for the index>1 braid the round-trip
    //     coefficient works out to `Rθᵀ·M·Rθ·M` (M = that F block); with
    //     nontrivial rotations Rθ, M that equals I *iff* M = I. A hexagon-
    //     consistent model would let a nontrivial F cancel, but B1 only needs
    //     the wiring + inverse-adjoint handling exercised, which the nontrivial
    //     R already does. (Since both braided legs are the sector `a`,
    //     R(a,b,c)=R(b,a,c), so no hexagon relation between the two R's is
    //     needed for index==0 either.)
    //   * F(a,a,0,c,c,a) is a genuine π/4 rotation, kept only so the unitarity
    //     assertion test has a non-identity unitary F block to check. It is not
    //     on any braid path here.
    #[derive(Clone, Copy, Debug)]
    struct UnitaryToyOmRule;

    impl UnitaryToyOmRule {
        const VACUUM: usize = 0;
        const A: usize = 1;
        const C: usize = 3;
        // R(a,a,c) rotation angle. Any nonzero angle whose sin/cos are both
        // nonzero makes the braid genuinely spread over both OM channels.
        const R_THETA: f64 = std::f64::consts::PI / 5.0;
    }

    impl FusionRule for UnitaryToyOmRule {
        fn rule_identity(&self) -> RuleIdentity { RuleIdentity::of_type::<Self>() }
        fn fusion_style(&self) -> FusionStyleKind {
            FusionStyleKind::Generic
        }

        fn braiding_style(&self) -> BraidingStyleKind {
            // Bosonic => has_braiding() (needed to pass the NoBraiding guard);
            // the actual crossings are governed by the (non-symmetric) R blocks.
            BraidingStyleKind::Bosonic
        }

        fn vacuum(&self) -> SectorId {
            SectorId::new(Self::VACUUM)
        }

        fn fusion_channels(&self, left: SectorId, right: SectorId) -> SectorVec {
            match (left.id(), right.id()) {
                (Self::VACUUM, x) | (x, Self::VACUUM) => smallvec![SectorId::new(x)],
                (Self::A, Self::A) => smallvec![SectorId::new(Self::C)],
                (Self::A, Self::C) | (Self::C, Self::A) => smallvec![SectorId::new(Self::A)],
                _ => smallvec![SectorId::new(Self::VACUUM)],
            }
        }

        fn nsymbol(&self, left: SectorId, right: SectorId, coupled: SectorId) -> usize {
            if (left.id(), right.id(), coupled.id()) == (Self::A, Self::A, Self::C) {
                2
            } else {
                usize::from(self.fusion_channels(left, right).contains(&coupled))
            }
        }
    }

    impl GenericFusionSymbols for UnitaryToyOmRule {
        type Scalar = f64;

        fn f_symbol_generic(
            &self,
            a: SectorId,
            b: SectorId,
            c: SectorId,
            d: SectorId,
            e: SectorId,
            f: SectorId,
        ) -> GenericFArray<Self::Scalar> {
            let n_mu = self.nsymbol(a, b, e);
            let n_nu = self.nsymbol(e, c, d);
            let n_kappa = self.nsymbol(b, c, f);
            let n_lambda = self.nsymbol(a, f, d);
            let shape = (n_mu, n_nu, n_kappa, n_lambda);
            let ids = (a.id(), b.id(), c.id(), d.id(), e.id(), f.id());
            if ids == (Self::A, Self::A, Self::VACUUM, Self::C, Self::C, Self::A) {
                // (mu, lambda) 2×2 block (nu = kappa = 0): a real π/4 rotation.
                // Kept only for the unitarity assertion — not on a braid path.
                let s = std::f64::consts::FRAC_1_SQRT_2;
                GenericFArray::new(vec![s, -s, s, s], shape)
            } else if ids == (Self::A, Self::A, Self::A, Self::A, Self::C, Self::C) {
                // The one F block the rank-3 braid reads: shape (2,1,2,1), a
                // 2×2 in (mu, kappa). IDENTITY (see the module comment for why
                // it must be I, not a rotation). Row-major over
                // (mu, nu, kappa, lambda): flat idx = mu*2 + kappa.
                GenericFArray::new(vec![1.0, 0.0, 0.0, 1.0], shape)
            } else {
                // Defensive default: identity on the leading diagonal of the
                // flattened ((mu,nu) × (kappa,lambda)) matrix. For 1×1 blocks
                // this is [1.0] (unitary); for any larger square block it is a
                // genuine unitary. No such block is reached by these tests, but
                // an all-ones fill would be silently non-unitary if one were.
                let rows = n_mu * n_nu;
                let cols = n_kappa * n_lambda;
                let mut data = vec![0.0; rows * cols];
                for r in 0..rows {
                    if r < cols {
                        data[r * cols + r] = 1.0;
                    }
                }
                GenericFArray::new(data, shape)
            }
        }

        fn r_symbol_generic(
            &self,
            a: SectorId,
            b: SectorId,
            c: SectorId,
        ) -> GenericRMatrix<Self::Scalar> {
            let rows = self.nsymbol(a, b, c);
            let cols = self.nsymbol(b, a, c);
            if (a.id(), b.id(), c.id()) == (Self::A, Self::A, Self::C) {
                // 2×2 rotation Rθ = [[cosθ, -sinθ], [sinθ, cosθ]] (unitary),
                // row-major. This is the block that makes the braid non-trivial.
                let (s, c_) = Self::R_THETA.sin_cos();
                GenericRMatrix::new(vec![c_, -s, s, c_], rows, cols)
            } else {
                // Every other block is 1×1 with modulus-1 entry (unitary).
                GenericRMatrix::new(vec![1.0; rows * cols], rows, cols)
            }
        }
    }

    // Rank-2 tree [a, a] -> c with a single OM vertex label `vertex`.
    fn unitary_rank2_tree(vertex: usize) -> FusionTreeKey {
        let a = SectorId::new(UnitaryToyOmRule::A);
        let c = SectorId::new(UnitaryToyOmRule::C);
        FusionTreeKey::new([a, a], c, [false, false], [], [MultiplicityIndex::new(vertex).expect("test multiplicity label is one-based")])
    }

    // Rank-3 tree [a, a, a] -> a: fuse a⊗a->c (OM vertex `vertex1`, N=2), then
    // c⊗a->a (vertex2, forced label 1). Innerline [c]. Mixes an OM vertex with
    // a multiplicity-1 vertex.
    fn unitary_rank3_tree(vertex1: usize) -> FusionTreeKey {
        let a = SectorId::new(UnitaryToyOmRule::A);
        let c = SectorId::new(UnitaryToyOmRule::C);
        FusionTreeKey::new(
            [a, a, a], a,
            [false, false, false],
            [c],
            [MultiplicityIndex::new(vertex1).expect("test multiplicity label is one-based"), MultiplicityIndex::ONE],
        )
    }

    // Braid at `index` (inv=false) then braid every output at `index`
    // (inv=true), summing coefficients per final tree. TensorKit's inverse
    // identity: this must equal the input tree with coefficient 1.
    fn braid_then_inverse_braid(
        rule: &UnitaryToyOmRule,
        tree: &FusionTreeKey,
        index: usize,
    ) -> std::collections::HashMap<FusionTreeKey, f64> {
        let forward = generic_artin_braid_at_with_inverse(rule, tree, index, false).unwrap();
        let mut totals: std::collections::HashMap<FusionTreeKey, f64> =
            std::collections::HashMap::new();
        for (mid, c_forward) in forward {
            let inverse = generic_artin_braid_at_with_inverse(rule, &mid, index, true).unwrap();
            for (final_tree, c_inverse) in inverse {
                *totals.entry(final_tree).or_insert(0.0) += c_forward * c_inverse;
            }
        }
        totals
    }

    fn assert_is_identity_on(
        totals: &std::collections::HashMap<FusionTreeKey, f64>,
        original: &FusionTreeKey,
    ) {
        let on_original = totals.get(original).copied().unwrap_or(0.0);
        assert!(
            (on_original - 1.0).abs() < 1e-12,
            "braid × inverse must be 1 on the original tree, got {on_original}"
        );
        for (tree, coeff) in totals {
            if tree != original {
                assert!(
                    coeff.abs() < 1e-12,
                    "braid × inverse must be 0 off the original tree, got {coeff}"
                );
            }
        }
    }

    #[test]
    fn unitary_toy_om_rule_r_and_f_blocks_are_unitary() {
        // Precondition of the round-trip tests: the F/R blocks the braid uses
        // are genuinely unitary. Assert it here so the identity tests below
        // rest on a checked assumption, not an asserted-by-fiat one.
        let rule = UnitaryToyOmRule;
        let a = SectorId::new(UnitaryToyOmRule::A);
        let c = SectorId::new(UnitaryToyOmRule::C);
        let vacuum = SectorId::new(UnitaryToyOmRule::VACUUM);

        // R(a,a,c): 2×2, must satisfy Rᵀ R = I.
        let r = rule.r_symbol_generic(a, a, c);
        assert_eq!(r.shape(), (2, 2));
        for col in 0..2 {
            for other in 0..2 {
                let dot: f64 = (0..2).map(|row| r.get(row, col) * r.get(row, other)).sum();
                let expected = if col == other { 1.0 } else { 0.0 };
                assert!(
                    (dot - expected).abs() < 1e-12,
                    "R(a,a,c) is not unitary at columns {col},{other}: {dot}"
                );
            }
        }

        // F(a,a,a,a,c,c): the braid's F block, shape (2,1,2,1) => 2×2 in
        // (mu,kappa). Must be unitary (it is the identity here).
        let f_braid = rule.f_symbol_generic(a, a, a, a, c, c);
        assert_eq!(f_braid.shape(), (2, 1, 2, 1));
        for mu in 0..2 {
            for other in 0..2 {
                let dot: f64 = (0..2)
                    .map(|kappa| f_braid.get(mu, 0, kappa, 0) * f_braid.get(other, 0, kappa, 0))
                    .sum();
                let expected = if mu == other { 1.0 } else { 0.0 };
                assert!(
                    (dot - expected).abs() < 1e-12,
                    "F(a,a,a,a,c,c) is not unitary at rows {mu},{other}: {dot}"
                );
            }
        }

        // F(a,a,0,c,c,a): the non-identity unitary F block, (mu,lambda) 2×2.
        let f_rot = rule.f_symbol_generic(a, a, vacuum, c, c, a);
        assert_eq!(f_rot.shape(), (2, 1, 1, 2));
        for col in 0..2 {
            for other in 0..2 {
                let dot: f64 = (0..2)
                    .map(|row| f_rot.get(row, 0, 0, col) * f_rot.get(row, 0, 0, other))
                    .sum();
                let expected = if col == other { 1.0 } else { 0.0 };
                assert!(
                    (dot - expected).abs() < 1e-12,
                    "F(a,a,0,c,c,a) is not unitary at columns {col},{other}: {dot}"
                );
            }
        }
    }

    #[test]
    fn generic_braid_index0_output_count_matches_nsymbol() {
        // Test (2): the forward braid of [a,a]->c at index 0 produces exactly
        // N(a,a,c) = 2 output trees, labelled with vertices 1 and 2 (both
        // nonzero because Rθ is a full rotation).
        let rule = UnitaryToyOmRule;
        let tree = unitary_rank2_tree(1);
        let outputs = generic_artin_braid_at_with_inverse(&rule, &tree, 0, false).unwrap();
        assert_eq!(outputs.len(), 2, "expected N(a,a,c)=2 output trees");
        let mut labels: Vec<usize> = outputs
            .iter()
            .map(|(t, _)| t.vertices()[0].get())
            .collect();
        labels.sort_unstable();
        assert_eq!(labels, vec![1, 2], "output vertex labels must be {{1,2}}");
    }

    #[test]
    fn generic_braid_index1_output_count_matches_nsymbol() {
        // Test (2), index>0 branch: braid of [a,a,a]->a at index 1 produces the
        // single c'=c channel, σ ∈ {1,2} (N(a,a,c)=2), λ = 1 (N(c,a,a)=1).
        let rule = UnitaryToyOmRule;
        let tree = unitary_rank3_tree(1);
        let outputs = generic_artin_braid_at_with_inverse(&rule, &tree, 1, false).unwrap();
        assert_eq!(outputs.len(), 2, "expected 2 (σ) output trees at index 1");
        let mut labels: Vec<(usize, usize)> = outputs
            .iter()
            .map(|(t, _)| {
                assert_eq!(t.innerlines(), &[SectorId::new(UnitaryToyOmRule::C)]);
                (t.vertices()[0].get(), t.vertices()[1].get())
            })
            .collect();
        labels.sort_unstable();
        assert_eq!(labels, vec![(1, 1), (2, 1)]);
    }

    #[test]
    fn generic_braid_inverse_is_identity_rank2_index0() {
        // Test (1), index==0 branch, over every enumerated vertex assignment.
        let rule = UnitaryToyOmRule;
        for vertex in 1..=2 {
            let tree = unitary_rank2_tree(vertex);
            let totals = braid_then_inverse_braid(&rule, &tree, 0);
            assert_is_identity_on(&totals, &tree);
        }
    }

    #[test]
    fn generic_braid_inverse_is_identity_rank3_index0() {
        // Test (1) + (4): index==0 on a rank>2 tree (uses the innerline as the
        // coupled sector for R), over every OM vertex assignment.
        let rule = UnitaryToyOmRule;
        for vertex1 in 1..=2 {
            let tree = unitary_rank3_tree(vertex1);
            let totals = braid_then_inverse_braid(&rule, &tree, 0);
            assert_is_identity_on(&totals, &tree);
        }
    }

    #[test]
    fn generic_braid_inverse_is_identity_rank3_index1() {
        // Test (1) + (4): the index>1 branch (F-move + R·F̄·R̄ contraction),
        // over every OM vertex assignment.
        let rule = UnitaryToyOmRule;
        for vertex1 in 1..=2 {
            let tree = unitary_rank3_tree(vertex1);
            let totals = braid_then_inverse_braid(&rule, &tree, 1);
            assert_is_identity_on(&totals, &tree);
        }
    }

    #[test]
    fn generic_braid_tree_roundtrip_is_identity() {
        // Exercises `generic_braid_tree`: braid [a,a]->c with the swap
        // permutation under levels [0,1] (=> inv=false), then braid the outputs
        // back under levels [1,0] (=> inv=true). The composite must be identity.
        let rule = UnitaryToyOmRule;
        for vertex in 1..=2 {
            let tree = unitary_rank2_tree(vertex);
            let forward = generic_braid_tree(&rule, &tree, &[1, 0], &[0, 1]).unwrap();
            let mut totals: std::collections::HashMap<FusionTreeKey, f64> =
                std::collections::HashMap::new();
            for (mid, c_forward) in forward {
                let back = generic_braid_tree(&rule, &mid, &[1, 0], &[1, 0]).unwrap();
                for (final_tree, c_back) in back {
                    *totals.entry(final_tree).or_insert(0.0) += c_forward * c_back;
                }
            }
            assert_is_identity_on(&totals, &tree);
        }
    }

    #[test]
    fn generic_braid_tree_identity_permutation_is_noop() {
        // The identity permutation decomposes to zero swaps: the tree returns
        // unchanged with coefficient 1.
        let rule = UnitaryToyOmRule;
        let tree = unitary_rank3_tree(2);
        let out = generic_braid_tree(&rule, &tree, &[0, 1, 2], &[0, 1, 2]).unwrap();
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].0, tree);
        assert!((out[0].1 - 1.0).abs() < 1e-12);
    }

    // ===================================================================
    // ADVERSARIAL REFUTATION (refute/b1-verify2): stress the nontrivial-F
    // path that the B1 round-trip tests deliberately avoid (F forced = I).
    // ===================================================================

    // Same structure as UnitaryToyOmRule, but F(a,a,a,a,c,c) — the block the
    // index>1 braid actually reads — is a NONTRIVIAL 2x2 rotation (in the
    // (mu,kappa) plane), not the identity. A single elementary braid needs no
    // hexagon consistency, so we can (a) compare the impl's coefficients to an
    // INDEPENDENT re-evaluation of TK's formula written here from scratch, and
    // (b) assert the elementary braid matrix is unitary (F,R all unitary).
    #[derive(Clone, Copy, Debug)]
    struct RefuteOmRule;

    impl RefuteOmRule {
        const VACUUM: usize = 0;
        const A: usize = 1;
        const C: usize = 3;
        const R_THETA: f64 = std::f64::consts::PI / 5.0;
        const F_PHI: f64 = std::f64::consts::PI / 7.0; // nontrivial F angle
    }

    impl FusionRule for RefuteOmRule {
        fn rule_identity(&self) -> RuleIdentity { RuleIdentity::of_type::<Self>() }
        fn fusion_style(&self) -> FusionStyleKind {
            FusionStyleKind::Generic
        }
        fn braiding_style(&self) -> BraidingStyleKind {
            BraidingStyleKind::Bosonic
        }
        fn vacuum(&self) -> SectorId {
            SectorId::new(Self::VACUUM)
        }
        fn fusion_channels(&self, left: SectorId, right: SectorId) -> SectorVec {
            match (left.id(), right.id()) {
                (Self::VACUUM, x) | (x, Self::VACUUM) => smallvec![SectorId::new(x)],
                (Self::A, Self::A) => smallvec![SectorId::new(Self::C)],
                (Self::A, Self::C) | (Self::C, Self::A) => smallvec![SectorId::new(Self::A)],
                _ => smallvec![SectorId::new(Self::VACUUM)],
            }
        }
        fn nsymbol(&self, left: SectorId, right: SectorId, coupled: SectorId) -> usize {
            if (left.id(), right.id(), coupled.id()) == (Self::A, Self::A, Self::C) {
                2
            } else {
                usize::from(self.fusion_channels(left, right).contains(&coupled))
            }
        }
    }

    impl GenericFusionSymbols for RefuteOmRule {
        type Scalar = f64;
        fn f_symbol_generic(
            &self,
            a: SectorId,
            b: SectorId,
            c: SectorId,
            d: SectorId,
            e: SectorId,
            f: SectorId,
        ) -> GenericFArray<Self::Scalar> {
            let shape = (
                self.nsymbol(a, b, e),
                self.nsymbol(e, c, d),
                self.nsymbol(b, c, f),
                self.nsymbol(a, f, d),
            );
            let ids = (a.id(), b.id(), c.id(), d.id(), e.id(), f.id());
            if ids == (Self::A, Self::A, Self::A, Self::A, Self::C, Self::C) {
                // shape (2,1,2,1); row-major flat idx = mu*2 + kappa. Rotation
                // R_phi in the (mu,kappa) plane: F[mu,0,kappa,0] = Rphi[mu,kappa].
                let (s, co) = Self::F_PHI.sin_cos();
                GenericFArray::new(vec![co, -s, s, co], shape)
            } else {
                // 1x1 blocks = 1.0 on every path these tests touch.
                let n = shape.0 * shape.1 * shape.2 * shape.3;
                GenericFArray::new(vec![1.0; n], shape)
            }
        }
        fn r_symbol_generic(
            &self,
            a: SectorId,
            b: SectorId,
            c: SectorId,
        ) -> GenericRMatrix<Self::Scalar> {
            let rows = self.nsymbol(a, b, c);
            let cols = self.nsymbol(b, a, c);
            if (a.id(), b.id(), c.id()) == (Self::A, Self::A, Self::C) {
                let (s, co) = Self::R_THETA.sin_cos();
                GenericRMatrix::new(vec![co, -s, s, co], rows, cols)
            } else {
                GenericRMatrix::new(vec![1.0; rows * cols], rows, cols)
            }
        }
    }

    fn refute_rank3_tree(vertex1: usize) -> FusionTreeKey {
        let a = SectorId::new(RefuteOmRule::A);
        let c = SectorId::new(RefuteOmRule::C);
        FusionTreeKey::new(
            [a, a, a], a,
            [false, false, false],
            [c],
            [MultiplicityIndex::new(vertex1).expect("test multiplicity label is one-based"), MultiplicityIndex::ONE],
        )
    }

    // INDEPENDENT re-evaluation of TK braiding_manipulations.jl:170-194 for the
    // rank-3 tree [a,a,a]->a braided at index 1. Written directly from the TK
    // source WITHOUT calling generic_artin_braid_at_with_inverse. Returns the
    // 2x2 braid matrix M[sigma][mu] over the OM channel (nu=lambda=rho=0 fixed,
    // c'=c, n_sigma=n_kappa=2).
    // Index-form loops kept on purpose: they mirror the TK formula verbatim.
    #[allow(clippy::needless_range_loop)]
    fn independent_braid_matrix_index1(rule: &RefuteOmRule, inverse: bool) -> [[f64; 2]; 2] {
        let a = SectorId::new(RefuteOmRule::A);
        let c = SectorId::new(RefuteOmRule::C);
        // TK naming for i>1 (0-based index==1 here): a=inner_ext[i-1]=leg0=a,
        // b=uncoupled[i]=a, c=inner_ext[i]=c, d=uncoupled[i+1]=a, e=coupled=a.
        let (a_s, b_s, c_s, d_s, e_s) = (a, a, c, a, a);
        let c_prime = c; // only channel in a (x) a
        let nu = 0usize; // vertices[i]=1 -> 0-based 0
        let n_sigma = rule.nsymbol(a_s, d_s, c_prime); // 2
        let n_lambda = rule.nsymbol(c_prime, b_s, e_s); // 1
        let n_rho = rule.nsymbol(d_s, c_s, e_s); // 1
        let n_kappa = rule.nsymbol(d_s, a_s, c_prime); // 2
        assert_eq!((n_sigma, n_lambda, n_rho, n_kappa), (2, 1, 1, 2));
        // Rmat1 = inv ? R(d,c,e)' : R(c,d,e); Rmat2 = inv ? R(d,a,c')' : R(a,d,c')
        let rmat1 = if inverse {
            rule.r_symbol_generic(d_s, c_s, e_s)
        } else {
            rule.r_symbol_generic(c_s, d_s, e_s)
        };
        let rmat2 = if inverse {
            rule.r_symbol_generic(d_s, a_s, c_prime)
        } else {
            rule.r_symbol_generic(a_s, d_s, c_prime)
        };
        let fmat = rule.f_symbol_generic(d_s, a_s, b_s, e_s, c_prime, c_s);
        let mut m = [[0.0f64; 2]; 2];
        for mu in 0..2usize {
            for sigma in 0..n_sigma {
                let lambda = 0usize;
                let mut coeff = 0.0f64;
                for rho in 0..n_rho {
                    for kappa in 0..n_kappa {
                        // Rmat1[nu,rho] (adjoint => conj(base[rho,nu]))
                        let r1 = if inverse {
                            rmat1.get(rho, nu) // 1x1, conj of real = itself
                        } else {
                            rmat1.get(nu, rho)
                        };
                        // conj(Fmat[kappa,lambda,mu,rho]); real => itself
                        let fc = fmat.get(kappa, lambda, mu, rho);
                        // conj(Rmat2[sigma,kappa]); inv => base[kappa,sigma]
                        let r2 = if inverse {
                            rmat2.get(kappa, sigma)
                        } else {
                            rmat2.get(sigma, kappa)
                        };
                        coeff += r1 * fc * r2;
                    }
                }
                m[sigma][mu] = coeff;
            }
        }
        m
    }

    // Extract the impl's 2x2 braid matrix M[sigma][mu] for the same case.
    fn impl_braid_matrix_index1(rule: &RefuteOmRule, inverse: bool) -> [[f64; 2]; 2] {
        let mut m = [[0.0f64; 2]; 2];
        for mu1 in 1..=2usize {
            let tree = refute_rank3_tree(mu1);
            let outs =
                generic_artin_braid_at_with_inverse(rule, &tree, 1, inverse).unwrap();
            for (out, coeff) in outs {
                assert_eq!(out.innerlines(), &[SectorId::new(RefuteOmRule::C)]);
                assert_eq!(out.vertices()[1].get(), 1, "lambda must be 1");
                let sigma = out.vertices()[0].get() - 1;
                m[sigma][mu1 - 1] = coeff;
            }
        }
        m
    }

    #[test]
    fn refute_impl_matches_independent_tk_formula_nontrivial_f() {
        let rule = RefuteOmRule;
        for &inverse in &[false, true] {
            let indep = independent_braid_matrix_index1(&rule, inverse);
            let got = impl_braid_matrix_index1(&rule, inverse);
            for s in 0..2 {
                for m in 0..2 {
                    assert!(
                        (indep[s][m] - got[s][m]).abs() < 1e-12,
                        "inverse={inverse} mismatch at [{s}][{m}]: indep={} impl={}",
                        indep[s][m],
                        got[s][m]
                    );
                }
            }
        }
    }

    #[test]
    fn refute_elementary_braid_is_unitary_nontrivial_f() {
        // With all R,F unitary the elementary braid matrix must be unitary,
        // even though F here is a nontrivial rotation (no hexagon needed).
        let rule = RefuteOmRule;
        let m = impl_braid_matrix_index1(&rule, false);
        // M^T M == I
        for i in 0..2 {
            for j in 0..2 {
                let dot: f64 = (0..2).map(|k| m[k][i] * m[k][j]).sum();
                let expect = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (dot - expect).abs() < 1e-12,
                    "braid matrix not unitary at ({i},{j}): {dot}"
                );
            }
        }
    }

    #[test]
    fn refute_roundtrip_matches_analytic_not_identity_when_f_nontrivial() {
        // For a NON-hexagon-consistent F, forward-then-inverse does NOT recover
        // the input for the index>0 path: the composite = M_inv * M_fwd, which
        // analytically is R(theta)^T . R(phi) . R(theta) . R(phi) = R(2*phi)
        // (commuting SO(2)), = I iff phi=0 (F=I). This is exactly why B1's own
        // round-trip test forces F=I; it is NOT a bug. Here we confirm the impl
        // reproduces the analytic composite (independent M_inv . M_fwd) and that
        // it deviates from identity by precisely R(2*phi).
        let rule = RefuteOmRule;
        let m_fwd = independent_braid_matrix_index1(&rule, false);
        let m_inv = independent_braid_matrix_index1(&rule, true);
        // Analytic composite M_inv . M_fwd (apply forward, then inverse).
        let mut comp = [[0.0f64; 2]; 2];
        for s in 0..2 {
            for m in 0..2 {
                comp[s][m] = (0..2).map(|t| m_inv[s][t] * m_fwd[t][m]).sum();
            }
        }
        // Impl round-trip matrix over mu -> final sigma.
        let mut impl_rt = [[0.0f64; 2]; 2];
        for mu1 in 1..=2usize {
            let tree = refute_rank3_tree(mu1);
            let mut col: std::collections::HashMap<usize, f64> =
                std::collections::HashMap::new();
            for (mid, cf) in generic_artin_braid_at_with_inverse(&rule, &tree, 1, false).unwrap()
            {
                for (fin, ci) in
                    generic_artin_braid_at_with_inverse(&rule, &mid, 1, true).unwrap()
                {
                    assert_eq!(fin.innerlines(), &[SectorId::new(RefuteOmRule::C)]);
                    let sigma = fin.vertices()[0].get() - 1;
                    *col.entry(sigma).or_insert(0.0) += cf * ci;
                }
            }
            for (sigma, v) in col {
                impl_rt[sigma][mu1 - 1] = v;
            }
        }
        // (a) impl reproduces the analytic composite.
        for s in 0..2 {
            for m in 0..2 {
                assert!(
                    (impl_rt[s][m] - comp[s][m]).abs() < 1e-12,
                    "impl round-trip != analytic at [{s}][{m}]: {} vs {}",
                    impl_rt[s][m],
                    comp[s][m]
                );
            }
        }
        // (b) composite == R(2*phi), NOT identity.
        let (s2, c2) = (2.0 * RefuteOmRule::F_PHI).sin_cos();
        let r2phi = [[c2, -s2], [s2, c2]];
        for s in 0..2 {
            for m in 0..2 {
                assert!((comp[s][m] - r2phi[s][m]).abs() < 1e-12);
            }
        }
        assert!(
            (comp[0][0] - 1.0).abs() > 1e-3,
            "sanity: nontrivial-F round-trip should deviate from identity"
        );
    }

    // ---- TK numeric oracle: real A4Irrep(3) sector, GenericFusion N=2 ----
    //
    // Values transcribed from TensorKit.jl v0.17 + TensorKitSectors v0.3.9
    // (A4Irrep, FusionStyle == GenericFusion()) — computed by TK's OWN
    // artin_braid on a FusionTreeBlock, independent of this port. We model just
    // the a=b=c=d=e=c'=3 OM sub-block (the braid coefficient formula is local:
    // it reads only F(3,3,3,3,3,3), R(3,3,3), which we transcribe exactly),
    // and reproduce it with generic_artin_braid_at_with_inverse on the tree
    // [3,3,3,3]->0, inner=[3,3], braided at index 1 (= TK i=2). If the F index
    // order [κ,λ,μ,ρ] or any conj/adjoint were transposed, this rich
    // (non-diagonal) 4x4 (μ,ν)->(σ,λ) block would not match.
    #[derive(Clone, Copy, Debug)]
    struct A4SubBlockRule;
    impl A4SubBlockRule {
        const VACUUM: usize = 0;
        const THREE: usize = 3;
        // F(3,3,3,3,3,3), row-major over (κ,λ,μ,ρ), dims (2,2,2,2). Verbatim
        // from TK (oracle4.jl FLAT_F_rowmajor_klmr), -0.0 normalised to 0.0.
        const F333333: [f64; 16] = [
            0.5, 0.0, 0.0, -0.5, 0.0, -0.5, -0.5, 0.0, 0.0, -0.5, -0.5, 0.0, -0.5, 0.0, 0.0, 0.5,
        ];
        // R(3,3,3) = [[-1,0],[0,1]] (TK).
        const R333: [f64; 4] = [-1.0, 0.0, 0.0, 1.0];
    }
    impl FusionRule for A4SubBlockRule {
        fn rule_identity(&self) -> RuleIdentity { RuleIdentity::of_type::<Self>() }
        fn fusion_style(&self) -> FusionStyleKind {
            FusionStyleKind::Generic
        }
        fn braiding_style(&self) -> BraidingStyleKind {
            BraidingStyleKind::Bosonic
        }
        fn vacuum(&self) -> SectorId {
            SectorId::new(Self::VACUUM)
        }
        fn fusion_channels(&self, left: SectorId, right: SectorId) -> SectorVec {
            // Truncated to the OM channel; only fusion_channels(3,3) is read by
            // the index-1 braid (for c' enumeration). 3⊗3 ∋ 3 with N=2 in A4.
            if (left.id(), right.id()) == (Self::THREE, Self::THREE) {
                smallvec![SectorId::new(Self::THREE)]
            } else {
                smallvec![SectorId::new(Self::VACUUM)]
            }
        }
        fn nsymbol(&self, left: SectorId, right: SectorId, coupled: SectorId) -> usize {
            if (left.id(), right.id(), coupled.id()) == (Self::THREE, Self::THREE, Self::THREE) {
                2
            } else {
                usize::from(self.fusion_channels(left, right).contains(&coupled))
            }
        }
    }
    impl GenericFusionSymbols for A4SubBlockRule {
        type Scalar = f64;
        fn f_symbol_generic(
            &self,
            a: SectorId,
            b: SectorId,
            c: SectorId,
            d: SectorId,
            e: SectorId,
            f: SectorId,
        ) -> GenericFArray<Self::Scalar> {
            let all3 = [a, b, c, d, e, f]
                .iter()
                .all(|s| s.id() == Self::THREE);
            assert!(all3, "only F(3,3,3,3,3,3) is modelled");
            GenericFArray::new(Self::F333333.to_vec(), (2, 2, 2, 2))
        }
        fn r_symbol_generic(
            &self,
            a: SectorId,
            b: SectorId,
            c: SectorId,
        ) -> GenericRMatrix<Self::Scalar> {
            assert_eq!(
                (a.id(), b.id(), c.id()),
                (Self::THREE, Self::THREE, Self::THREE),
                "only R(3,3,3) is modelled"
            );
            GenericRMatrix::new(Self::R333.to_vec(), 2, 2)
        }
    }

    #[test]
    #[allow(clippy::type_complexity)] // oracle table typed to match the TK dump verbatim
    fn tk_oracle_a4_generic_braid_matches_tensorkit() {
        let rule = A4SubBlockRule;
        let three = SectorId::new(A4SubBlockRule::THREE);
        let vac = SectorId::new(A4SubBlockRule::VACUUM);
        // TK oracle sub-block (identical for inv=false and inv=true here):
        // (mu,nu) -> (sigma,lambda) => coeff.  [oracle4.jl SUBBLOCK]
        let oracle: [((usize, usize), (usize, usize), f64); 8] = [
            ((1, 1), (1, 1), 0.5),
            ((2, 2), (1, 1), 0.5),
            ((2, 1), (2, 1), 0.5),
            ((1, 2), (2, 1), -0.5),
            ((2, 1), (1, 2), -0.5),
            ((1, 2), (1, 2), 0.5),
            ((1, 1), (2, 2), 0.5),
            ((2, 2), (2, 2), 0.5),
        ];
        for &inverse in &[false, true] {
            // Build impl matrix keyed by ((mu,nu),(sigma,lambda)).
            let mut got: std::collections::HashMap<((usize, usize), (usize, usize)), f64> =
                std::collections::HashMap::new();
            for mu in 1..=2usize {
                for nu in 1..=2usize {
                    let tree = FusionTreeKey::new(
                        [three, three, three, three], vac,
                        [false, false, false, false],
                        [three, three],
                        [MultiplicityIndex::new(mu).expect("test multiplicity label is one-based"), MultiplicityIndex::new(nu).expect("test multiplicity label is one-based"), MultiplicityIndex::ONE],
                    );
                    for (out, coeff) in
                        generic_artin_braid_at_with_inverse(&rule, &tree, 1, inverse).unwrap()
                    {
                        assert_eq!(out.innerlines(), &[three, three], "c'=3, e=3 unchanged");
                        assert_eq!(out.vertices()[2].get(), 1);
                        let sigma = out.vertices()[0].get();
                        let lambda = out.vertices()[1].get();
                        got.insert(((mu, nu), (sigma, lambda)), coeff);
                    }
                }
            }
            // Every oracle entry must be reproduced.
            for &(inp, outp, val) in &oracle {
                let g = got.get(&(inp, outp)).copied().unwrap_or(0.0);
                assert!(
                    (g - val).abs() < 1e-10,
                    "inv={inverse} {inp:?}->{outp:?}: impl={g} TK={val}"
                );
            }
            // And the impl must produce NO nonzero outside the oracle set.
            for (&(inp, outp), &g) in &got {
                if g.abs() > 1e-10 {
                    let known = oracle
                        .iter()
                        .any(|&(i, o, v)| i == inp && o == outp && (v - g).abs() < 1e-10);
                    assert!(known, "inv={inverse} spurious nonzero {inp:?}->{outp:?}={g}");
                }
            }
        }
    }

    // ===================== Stage B2a: Generic bend / repartition =============
    //
    // Oracle & gate rule: the REAL A4Irrep(3) outer-multiplicity sub-block.
    // A4Irrep(3) is self-dual (dual(3)=3), 3⊗3 = {0,1,2,3} with N(3,3,3)=2 (the
    // only outer multiplicity) AND 3⊗3 ∋ vacuum — so it is genuinely rigid and
    // can bend. (Contrast the braid-only `UnitaryToyOmRule`: its sector `a` has
    // NO proper dual — a⊗a ∌ vacuum — so every B-symbol there is degenerate and
    // bending is undefined. Extending it was therefore impossible without
    // changing its fusion structure; a new rigid rule is used instead, per the
    // "stop if existing code must change" constraint.)
    //
    // Constants computed out-of-band from TensorKit v0.16.2 + TensorKitSectors
    // v0.3.6 (git-tree-sha1 334a0ed5a0a0088a2b6fe7a39f78dda928038d85), by
    // TensorKit's OWN Bsymbol / Asymbol / bendright, independent of this port.
    //
    // DISCRIMINATING POWER (honest note): for A4Irrep(3), Bsymbol(3,3,3) AND
    // Asymbol(3,3,3) are BOTH the 2×2 identity (asserted below) — the A4 3-irrep
    // bend has no off-diagonal channel mixing. So this oracle does NOT catch a
    // μ↔ν transpose in the B-matrix indexing. It DOES pin the tree surgery, the
    // √dim(c)·(1/√dim(a)) coefficient (√3 vs 1 across the varied innerline in the
    // rank-3 table — a sqrt/invsqrt swap changes these), μ = last-codomain-vertex
    // selection, ν → domain vertex-label storage, and the F→B derivation. A
    // μ↔ν-discriminating BEND oracle needs a non-diagonal Bsymbol (e.g. SU(3));
    // none is available here as verified constants — flagged for B2b.
    #[derive(Clone, Copy, Debug)]
    struct A4BendRule;

    impl FusionRule for A4BendRule {
        fn rule_identity(&self) -> RuleIdentity { RuleIdentity::of_type::<Self>() }
        fn fusion_style(&self) -> FusionStyleKind {
            FusionStyleKind::Generic
        }
        fn braiding_style(&self) -> BraidingStyleKind {
            // Unused: bend/repartition are planar (no braiding). Bosonic keeps
            // the rule well-formed.
            BraidingStyleKind::Bosonic
        }
        fn vacuum(&self) -> SectorId {
            SectorId::new(0)
        }
        fn dual(&self, sector: SectorId) -> SectorId {
            // A4: 0 and 3 self-dual; the two nontrivial 1-dim irreps 1,2 are
            // each other's dual (verified in Julia). Only dual(3)=3 is exercised
            // by the bend, but the full map is correct.
            match sector.id() {
                1 => SectorId::new(2),
                2 => SectorId::new(1),
                _ => sector,
            }
        }
        fn fusion_channels(&self, left: SectorId, right: SectorId) -> SectorVec {
            match (left.id(), right.id()) {
                (0, x) | (x, 0) => smallvec![SectorId::new(x)],
                (3, 3) => smallvec![
                    SectorId::new(0),
                    SectorId::new(1),
                    SectorId::new(2),
                    SectorId::new(3)
                ],
                (3, _) | (_, 3) => smallvec![SectorId::new(3)],
                // {1,2}⊗{1,2}: never touched by the bend trees; defensive stub.
                _ => smallvec![SectorId::new(0)],
            }
        }
        fn nsymbol(&self, left: SectorId, right: SectorId, coupled: SectorId) -> usize {
            if (left.id(), right.id(), coupled.id()) == (3, 3, 3) {
                2
            } else {
                usize::from(self.fusion_channels(left, right).contains(&coupled))
            }
        }
    }

    impl GenericFusionSymbols for A4BendRule {
        type Scalar = f64;
        fn f_symbol_generic(
            &self,
            a: SectorId,
            b: SectorId,
            c: SectorId,
            d: SectorId,
            e: SectorId,
            f: SectorId,
        ) -> GenericFArray<Self::Scalar> {
            let inv = 1.0 / 3.0_f64.sqrt(); // 1/√dim(3)
            let ids = (a.id(), b.id(), c.id(), d.id(), e.id(), f.id());
            match ids {
                // F(3,3,3,3,3,0): the block Bsymbol(3,3,3) reshapes, shape
                // (N(3,3,3),N(3,3,3),1,1)=(2,2,1,1). (μ,ν)=(1/√3)·I, row-major.
                (3, 3, 3, 3, 3, 0) => GenericFArray::new(vec![inv, 0.0, 0.0, inv], (2, 2, 1, 1)),
                // F(x,3,3,x,3,0), x∈{0,1,2}: Bsymbol(x,3,3), shape (1,1,1,1)=[1].
                (0, 3, 3, 0, 3, 0) | (1, 3, 3, 1, 3, 0) | (2, 3, 3, 2, 3, 0) => {
                    GenericFArray::new(vec![1.0], (1, 1, 1, 1))
                }
                // F(3,3,3,3,x,0), x∈{0,1,2}: Bsymbol(3,3,x) (the return bend when
                // the intermediate coupled sector is x), (1,1,1,1)=[1/3].
                (3, 3, 3, 3, 0, 0) | (3, 3, 3, 3, 1, 0) | (3, 3, 3, 3, 2, 0) => {
                    GenericFArray::new(vec![1.0 / 3.0], (1, 1, 1, 1))
                }
                // F(3,3,3,3,0,3): the block Asymbol(3,3,3) reshapes, shape
                // (1,1,N(3,3,3),N(3,3,3))=(1,1,2,2). (κ,λ)=(1/√3)·I, row-major.
                (3, 3, 3, 3, 0, 3) => GenericFArray::new(vec![inv, 0.0, 0.0, inv], (1, 1, 2, 2)),
                _ => panic!("A4BendRule: unmodelled F{ids:?}"),
            }
        }
        fn r_symbol_generic(
            &self,
            _a: SectorId,
            _b: SectorId,
            _c: SectorId,
        ) -> GenericRMatrix<Self::Scalar> {
            // Unused: planar duality moves never braid. Trivial 1×1 stub.
            GenericRMatrix::new(vec![1.0], 1, 1)
        }
    }

    impl GenericRigidSymbols for A4BendRule {
        fn sqrt_dim_scalar(&self, sector: SectorId) -> Self::Scalar {
            if sector.id() == 3 {
                3.0_f64.sqrt()
            } else {
                1.0
            }
        }
        fn inv_sqrt_dim_scalar(&self, sector: SectorId) -> Self::Scalar {
            if sector.id() == 3 {
                1.0 / 3.0_f64.sqrt()
            } else {
                1.0
            }
        }
        fn frobenius_schur_phase_scalar(&self, _sector: SectorId) -> Self::Scalar {
            // All A4 sectors reached here have FS phase +1 (verified in Julia).
            1.0
        }
    }

    #[derive(Clone, Copy, Debug)]
    struct MisreportedSimpleA4Rule;

    impl FusionRule for MisreportedSimpleA4Rule {
        fn rule_identity(&self) -> RuleIdentity {
            RuleIdentity::of_type::<Self>()
        }

        fn fusion_style(&self) -> FusionStyleKind {
            FusionStyleKind::Simple
        }

        fn braiding_style(&self) -> BraidingStyleKind {
            A4BendRule.braiding_style()
        }

        fn vacuum(&self) -> SectorId {
            A4BendRule.vacuum()
        }

        fn dual(&self, sector: SectorId) -> SectorId {
            A4BendRule.dual(sector)
        }

        fn fusion_channels(&self, left: SectorId, right: SectorId) -> SectorVec {
            A4BendRule.fusion_channels(left, right)
        }

        fn nsymbol(&self, left: SectorId, right: SectorId, coupled: SectorId) -> usize {
            A4BendRule.nsymbol(left, right, coupled)
        }
    }

    impl GenericFusionSymbols for MisreportedSimpleA4Rule {
        type Scalar = f64;

        fn f_symbol_generic(
            &self,
            a: SectorId,
            b: SectorId,
            c: SectorId,
            d: SectorId,
            e: SectorId,
            f: SectorId,
        ) -> GenericFArray<Self::Scalar> {
            A4BendRule.f_symbol_generic(a, b, c, d, e, f)
        }

        fn r_symbol_generic(
            &self,
            a: SectorId,
            b: SectorId,
            c: SectorId,
        ) -> GenericRMatrix<Self::Scalar> {
            A4BendRule.r_symbol_generic(a, b, c)
        }
    }

    impl GenericRigidSymbols for MisreportedSimpleA4Rule {
        fn sqrt_dim_scalar(&self, sector: SectorId) -> Self::Scalar {
            A4BendRule.sqrt_dim_scalar(sector)
        }

        fn inv_sqrt_dim_scalar(&self, sector: SectorId) -> Self::Scalar {
            A4BendRule.inv_sqrt_dim_scalar(sector)
        }

        fn frobenius_schur_phase_scalar(&self, sector: SectorId) -> Self::Scalar {
            A4BendRule.frobenius_schur_phase_scalar(sector)
        }
    }

    fn a4_three() -> SectorId {
        SectorId::new(3)
    }

    // cod [3,3]->3 (vertex μ), dom [3]->3.
    fn a4_pair_rank2(mu: usize) -> FusionTreePairKey {
        let t = a4_three();
        let cod = FusionTreeKey::new([t, t], t, [false, false], [], [MultiplicityIndex::new(mu).expect("test multiplicity label is one-based")]);
        let dom = FusionTreeKey::new([t], t, [false], [], []);
        FusionTreePairKey::pair(cod, dom)
    }

    // cod [3,3,3]->3, inner=[x] (vertices v1,v2), dom [3]->3.
    fn a4_pair_rank3(inner: usize, v1: usize, v2: usize) -> FusionTreePairKey {
        let t = a4_three();
        let cod = FusionTreeKey::new(
            [t, t, t], t,
            [false, false, false],
            [SectorId::new(inner)],
            [MultiplicityIndex::new(v1).expect("test multiplicity label is one-based"), MultiplicityIndex::new(v2).expect("test multiplicity label is one-based")],
        );
        let dom = FusionTreeKey::new([t], t, [false], [], []);
        FusionTreePairKey::pair(cod, dom)
    }

    #[test]
    fn generic_proof_hook_rechecks_reported_style() {
        // What: the proof-consuming Generic hook rejects a provider that
        // implements Generic symbols but reports a multiplicity-free style.
        let empty = BlockStructure::empty(0);
        let wrong_style =
            LocallyValidatedFusionTreeBlockStructure::try_new(&MisreportedSimpleA4Rule, &empty).unwrap();
        assert_eq!(
            wrong_style
                .generic_permute_tree_pair_for_block_index(0, &[], &[])
                .unwrap_err(),
            CoreError::UnsupportedFusionStyle {
                expected: FusionStyleKind::Generic,
                actual: FusionStyleKind::Simple,
            }
        );
    }

    fn round_trip_bend(
        rule: &A4BendRule,
        pair: &FusionTreePairKey,
    ) -> std::collections::HashMap<FusionTreePairKey, f64> {
        let mut totals = std::collections::HashMap::new();
        for (mid, c1) in generic_bendright_tree_pair(rule, pair).unwrap() {
            for (out, c2) in generic_bendleft_tree_pair(rule, &mid).unwrap() {
                *totals.entry(out).or_insert(0.0) += c1 * c2;
            }
        }
        totals
    }

    fn assert_identity_map(
        totals: &std::collections::HashMap<FusionTreePairKey, f64>,
        expected_self: &FusionTreePairKey,
        label: &str,
    ) {
        for (key, coeff) in totals {
            let want = if key == expected_self { 1.0 } else { 0.0 };
            assert!(
                (coeff - want).abs() < 1e-12,
                "{label}: coeff {coeff} for is_self={} (want {want})",
                key == expected_self
            );
        }
        assert!(
            (totals.get(expected_self).copied().unwrap_or(0.0) - 1.0).abs() < 1e-12,
            "{label}: self coefficient missing"
        );
    }

    // Gate 1: bendright∘bendleft == identity (the B-matrix is a Hom-space
    // isomorphism), enumerated over all vertex assignments, rank 2 and 3.
    #[test]
    fn b2a_generic_bend_round_trip_identity() {
        let rule = A4BendRule;
        let t = a4_three();
        // Premise the round-trip depends on: N(a,b,c)==N(c,dual(b),a) so the
        // bend is square/invertible on the bent triple (a=b=c=3).
        assert_eq!(rule.nsymbol(t, t, t), rule.nsymbol(t, rule.dual(t), t));
        assert_eq!(rule.nsymbol(t, t, t), 2);

        for mu in 1..=2 {
            let pair = a4_pair_rank2(mu);
            assert_identity_map(&round_trip_bend(&rule, &pair), &pair, &format!("rank2 μ={mu}"));
        }
        // rank 3: inner∈{0,1,2} forces v1=v2=1 (N=1); inner=3 opens both OM
        // vertices v1,v2∈{1,2}. All vertex assignments enumerated.
        for inner in 0..=2 {
            let pair = a4_pair_rank3(inner, 1, 1);
            assert_identity_map(
                &round_trip_bend(&rule, &pair),
                &pair,
                &format!("rank3 inner={inner}"),
            );
        }
        for v1 in 1..=2 {
            for v2 in 1..=2 {
                let pair = a4_pair_rank3(3, v1, v2);
                assert_identity_map(
                    &round_trip_bend(&rule, &pair),
                    &pair,
                    &format!("rank3 inner=3 v=({v1},{v2})"),
                );
            }
        }
    }

    // Gate 2: repartition(N) then repartition back to the original N == identity.
    // via_n=1 exercises one bend each way; via_n=0 exercises two bends each way,
    // covering the rank-1-codomain (left_coupled=vacuum) branch of bendright.
    #[test]
    fn b2a_generic_repartition_round_trip_identity() {
        let rule = A4BendRule;
        for via_n in [1usize, 0usize] {
            for mu in 1..=2 {
                let pair = a4_pair_rank2(mu); // codomain rank 2
                let mut totals = std::collections::HashMap::new();
                for (mid, c1) in generic_repartition_tree_pair(&rule, &pair, via_n).unwrap() {
                    for (out, c2) in generic_repartition_tree_pair(&rule, &mid, 2).unwrap() {
                        *totals.entry(out).or_insert(0.0) += c1 * c2;
                    }
                }
                assert_identity_map(&totals, &pair, &format!("repartition via {via_n} μ={mu}"));
            }
        }
    }

    // Oracle: b_symbol_generic / a_symbol_generic match TK's Bsymbol / Asymbol.
    #[test]
    fn b2a_a4_b_and_a_symbol_match_tensorkit() {
        let rule = A4BendRule;
        let t = a4_three();
        // TK: Bsymbol(3,3,3) == I₂, Asymbol(3,3,3) == I₂ (TKS v0.3.6).
        let b = rule.b_symbol_generic(t, t, t);
        assert_eq!(b.shape(), (2, 2));
        let a = rule.a_symbol_generic(t, t, t);
        assert_eq!(a.shape(), (2, 2));
        for i in 0..2 {
            for j in 0..2 {
                let want = if i == j { 1.0 } else { 0.0 };
                assert!((b.get(i, j) - want).abs() < 1e-10, "B[{i},{j}]={}", b.get(i, j));
                assert!((a.get(i, j) - want).abs() < 1e-10, "A[{i},{j}]={}", a.get(i, j));
            }
        }
    }

    // Focused unit test for the domain-empty keep-last overwrite: mirrors TK's
    // block assignment `U[row, col] = coeff` (duality_manipulations.jl:110),
    // where every ν collapses onto the same output key (no vertex to store) and
    // the LAST non-zero ν wins. A4's Bsymbol is diagonal so it never puts two
    // non-zeros in one row — this needs a synthetic non-diagonal B. b_symbol is
    // overridden directly (default-method override), so no F is consulted.
    #[derive(Clone, Copy, Debug)]
    struct OverwriteProbeRule;
    impl FusionRule for OverwriteProbeRule {
        fn rule_identity(&self) -> RuleIdentity { RuleIdentity::of_type::<Self>() }
        fn fusion_style(&self) -> FusionStyleKind {
            FusionStyleKind::Generic
        }
        fn braiding_style(&self) -> BraidingStyleKind {
            BraidingStyleKind::Bosonic
        }
        fn vacuum(&self) -> SectorId {
            SectorId::new(0)
        }
        fn fusion_channels(&self, left: SectorId, right: SectorId) -> SectorVec {
            match (left.id(), right.id()) {
                (0, x) | (x, 0) => smallvec![SectorId::new(x)],
                (1, 1) => smallvec![SectorId::new(0)],
                _ => smallvec![SectorId::new(0)],
            }
        }
        fn nsymbol(&self, left: SectorId, right: SectorId, coupled: SectorId) -> usize {
            if (left.id(), right.id(), coupled.id()) == (1, 1, 0) {
                2
            } else {
                usize::from(self.fusion_channels(left, right).contains(&coupled))
            }
        }
    }
    impl GenericFusionSymbols for OverwriteProbeRule {
        type Scalar = f64;
        fn f_symbol_generic(
            &self,
            _a: SectorId,
            _b: SectorId,
            _c: SectorId,
            _d: SectorId,
            _e: SectorId,
            _f: SectorId,
        ) -> GenericFArray<Self::Scalar> {
            unreachable!("b_symbol_generic is overridden; F is never read")
        }
        fn r_symbol_generic(
            &self,
            _a: SectorId,
            _b: SectorId,
            _c: SectorId,
        ) -> GenericRMatrix<Self::Scalar> {
            GenericRMatrix::new(vec![1.0], 1, 1)
        }
    }
    impl GenericRigidSymbols for OverwriteProbeRule {
        fn sqrt_dim_scalar(&self, _sector: SectorId) -> Self::Scalar {
            1.0
        }
        fn inv_sqrt_dim_scalar(&self, _sector: SectorId) -> Self::Scalar {
            1.0
        }
        fn frobenius_schur_phase_scalar(&self, _sector: SectorId) -> Self::Scalar {
            1.0
        }
        fn b_symbol_generic(
            &self,
            _a: SectorId,
            _b: SectorId,
            _c: SectorId,
        ) -> GenericRMatrix<Self::Scalar> {
            // Row 0 = [0.3, 0.7]: two non-zeros, distinct, so keep-last is
            // distinguishable from keep-first (0.3) and from sum (1.0).
            GenericRMatrix::new(vec![0.3, 0.7, 0.0, 0.0], 2, 2)
        }
    }

    #[test]
    fn b2a_generic_bendright_domain_empty_keeps_last_nu() {
        let rule = OverwriteProbeRule;
        let a = SectorId::new(1);
        let c = rule.vacuum();
        // cod [1,1]->vac (vertex 1); dom []->vac (EMPTY domain ⇒ ν has nowhere to go).
        let cod = FusionTreeKey::new([a, a], c, [false, false], [], [MultiplicityIndex::ONE]);
        let dom = FusionTreeKey::new([], c, [], [], []);
        let pair = FusionTreePairKey::pair(cod, dom);
        pair.validate_for_rule(&rule).unwrap();
        let out = generic_bendright_tree_pair(&rule, &pair).unwrap();
        assert_eq!(out.len(), 1, "empty domain collapses ν to one key");
        // coeff0 = √dim(vac)·(1/√dim(1)) = 1; keep-last ⇒ B[0,1] = 0.7.
        assert!((out[0].1 - 0.7).abs() < 1e-12, "keep-last ν: got {}", out[0].1);
    }

    // Oracle: tree-level bendright tables vs TensorKit's own bendright.
    #[test]
    fn b2a_a4_bendright_tree_table_matches_tensorkit() {
        let rule = A4BendRule;
        let sq3 = 3.0_f64.sqrt();

        // --- rank 2: cod [3,3]->3 (μ), dom [3]->3.  TK (probe5.jl):
        //   μ=1 -> dom vertex 1, coeff 1 ;  μ=2 -> dom vertex 2, coeff 1.
        for mu in 1..=2 {
            let out = generic_bendright_tree_pair(&rule, &a4_pair_rank2(mu)).unwrap();
            let nonzero: Vec<_> = out.iter().filter(|(_, c)| c.abs() > 1e-10).collect();
            assert_eq!(nonzero.len(), 1, "rank2 μ={mu} expects one nonzero");
            let (key, coeff) = nonzero[0];
            assert_eq!(key.codomain_tree().uncoupled(), [a4_three()], "rank2 cod");
            assert!(key.codomain_tree().vertices().is_empty(), "rank2 cod rank-1 no vtx");
            assert_eq!(key.domain_tree().vertices()[0].get(), mu, "rank2 ν == μ (B diagonal)");
            assert!((coeff - 1.0).abs() < 1e-10, "rank2 μ={mu} coeff {coeff}");
        }

        // --- rank 3: cod [3,3,3]->3 inner=[x] (v1,v2), dom [3]->3.
        // TK (probe6.jl): coeff = √dim(3)/√dim(inner) = √3 (inner∈{0,1,2}) or 1
        // (inner=3); output cod vertex = v1, dom vertex = ν = v2 (B=I diagonal).
        // Table rows: (inner, v1, v2) -> (cod_vtx, dom_vtx, coeff).
        let table: [(usize, usize, usize, usize, usize, f64); 7] = [
            (0, 1, 1, 1, 1, sq3),
            (1, 1, 1, 1, 1, sq3),
            (2, 1, 1, 1, 1, sq3),
            (3, 1, 1, 1, 1, 1.0),
            (3, 2, 1, 2, 1, 1.0),
            (3, 1, 2, 1, 2, 1.0),
            (3, 2, 2, 2, 2, 1.0),
        ];
        for (inner, v1, v2, cod_vtx, dom_vtx, coeff) in table {
            let out = generic_bendright_tree_pair(&rule, &a4_pair_rank3(inner, v1, v2)).unwrap();
            let nonzero: Vec<_> = out.iter().filter(|(_, c)| c.abs() > 1e-10).collect();
            assert_eq!(nonzero.len(), 1, "rank3 ({inner},{v1},{v2}) one nonzero");
            let (key, got) = nonzero[0];
            let left_coupled = key.codomain_tree().coupled().id();
            assert_eq!(left_coupled, inner, "rank3 left_coupled == innerline");
            assert_eq!(key.codomain_tree().vertices()[0].get(), cod_vtx, "rank3 cod vtx");
            assert_eq!(key.domain_tree().vertices()[0].get(), dom_vtx, "rank3 dom vtx");
            assert!((got - coeff).abs() < 1e-10, "rank3 ({inner},{v1},{v2}) coeff {got} want {coeff}");
        }
    }

    // ============ REFUTE(b2a): μ↔ν / κ↔λ transpose discriminator ============
    //
    // The A4 oracle CANNOT catch a B-matrix (or A-matrix) index transpose: for
    // A4Irrep(3) both Bsymbol and Asymbol are the 2×2 IDENTITY, which is its own
    // transpose. This synthetic rule closes that gap with a *deliberately
    // non-symmetric* B and A block (B[0,1]≠B[1,0], A[0,1]≠A[1,0]).
    //
    // The reshape-collapse is transpose-free in Julia (verified out-of-band:
    // `reshape(F,(N1,N2))[μ,ν]==F[μ,ν,1,1]` for trailing singletons and
    // `[κ,λ]==F[1,1,κ,λ]` for leading singletons), so the CORRECT reading is
    //   B[μ,ν] = √dim(a)·√dim(b)·invsqrtdim(c) · F(a,b,dual(b),a,c,unit)[μ,ν,0,0]
    //   A[κ,λ] = √dim(a)·√dim(b)·invsqrtdim(c) · conj(κ_a·F(dual(a),a,b,b,unit,c)[0,0,κ,λ]).
    // A μ↔ν (or κ↔λ) swap in the impl would read F[ν,μ,0,0] / F[0,0,λ,κ] and
    // produce the TRANSPOSE — which THIS test detects and the A4 oracle does not.
    #[derive(Clone, Copy, Debug)]
    struct TransposeProbeRule;
    // Sector 1 is self-dual with dim 4 (so √dim=2, exercising the coeff factor);
    // 1⊗1 = {0 (rigidity), 1 (with N=2)}. Only the (1,1,1) block is non-trivial.
    impl FusionRule for TransposeProbeRule {
        fn rule_identity(&self) -> RuleIdentity { RuleIdentity::of_type::<Self>() }
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
            sector // 0 and 1 both self-dual
        }
        fn fusion_channels(&self, left: SectorId, right: SectorId) -> SectorVec {
            match (left.id(), right.id()) {
                (0, x) | (x, 0) => smallvec![SectorId::new(x)],
                (1, 1) => smallvec![SectorId::new(0), SectorId::new(1)],
                _ => smallvec![SectorId::new(0)],
            }
        }
        fn nsymbol(&self, left: SectorId, right: SectorId, coupled: SectorId) -> usize {
            if (left.id(), right.id(), coupled.id()) == (1, 1, 1) {
                2 // the single outer multiplicity
            } else {
                usize::from(self.fusion_channels(left, right).contains(&coupled))
            }
        }
    }
    // The raw F data, in TK [μ,ν,κ,λ] semantic order, row-major — the SAME bytes
    // the from-scratch oracle below reads directly.
    const TP_FB: [f64; 4] = [0.3, 0.7, 0.9, 0.1]; // F(1,1,1,1,1,0)[μ,ν] block, non-symmetric
    const TP_FA: [f64; 4] = [0.2, 0.5, 0.6, 0.4]; // F(1,1,1,1,0,1)[κ,λ] block, non-symmetric
    impl GenericFusionSymbols for TransposeProbeRule {
        type Scalar = f64;
        fn f_symbol_generic(
            &self,
            a: SectorId,
            b: SectorId,
            c: SectorId,
            d: SectorId,
            e: SectorId,
            f: SectorId,
        ) -> GenericFArray<Self::Scalar> {
            match (a.id(), b.id(), c.id(), d.id(), e.id(), f.id()) {
                // B block: shape (N(1,1,1),N(1,1,1),1,1) = (2,2,1,1).
                (1, 1, 1, 1, 1, 0) => GenericFArray::new(TP_FB.to_vec(), (2, 2, 1, 1)),
                // A block: shape (1,1,N(1,1,1),N(1,1,1)) = (1,1,2,2).
                (1, 1, 1, 1, 0, 1) => GenericFArray::new(TP_FA.to_vec(), (1, 1, 2, 2)),
                other => panic!("TransposeProbeRule: unmodelled F{other:?}"),
            }
        }
        fn r_symbol_generic(
            &self,
            _a: SectorId,
            _b: SectorId,
            _c: SectorId,
        ) -> GenericRMatrix<Self::Scalar> {
            GenericRMatrix::new(vec![1.0], 1, 1)
        }
    }
    impl GenericRigidSymbols for TransposeProbeRule {
        fn sqrt_dim_scalar(&self, sector: SectorId) -> Self::Scalar {
            if sector.id() == 1 { 2.0 } else { 1.0 } // dim(1)=4
        }
        fn inv_sqrt_dim_scalar(&self, sector: SectorId) -> Self::Scalar {
            if sector.id() == 1 { 0.5 } else { 1.0 }
        }
        fn frobenius_schur_phase_scalar(&self, _sector: SectorId) -> Self::Scalar {
            1.0
        }
    }

    // Independent from-scratch TK evaluation of the reshape formula — explicit
    // index loops, NO call into b_symbol_generic / a_symbol_generic.
    fn tp_expected_b() -> [[f64; 2]; 2] {
        let factor = 2.0 * 2.0 * 0.5; // √dim(1)·√dim(1)·invsqrtdim(1) = 2
        let mut b = [[0.0; 2]; 2];
        for mu in 0..2 {
            for nu in 0..2 {
                b[mu][nu] = factor * TP_FB[mu * 2 + nu]; // F[μ,ν,0,0]
            }
        }
        b
    }
    fn tp_expected_a() -> [[f64; 2]; 2] {
        let factor = 2.0 * 2.0 * 0.5;
        let kappa_a = 1.0f64; // FS phase, real
        let mut a = [[0.0; 2]; 2];
        for k in 0..2 {
            for l in 0..2 {
                // conj(κ_a · F[0,0,κ,λ]) · factor; all real here.
                a[k][l] = factor * (kappa_a * TP_FA[k * 2 + l]);
            }
        }
        a
    }

    #[test]
    fn refute_b2a_b_symbol_is_not_transposed() {
        let rule = TransposeProbeRule;
        let s = SectorId::new(1);
        let b = rule.b_symbol_generic(s, s, s);
        assert_eq!(b.shape(), (2, 2));
        let want = tp_expected_b();
        // Sanity: the oracle itself must be non-symmetric, else no discrimination.
        assert!((want[0][1] - want[1][0]).abs() > 0.1, "oracle B must be non-symmetric");
        for mu in 0..2 {
            for nu in 0..2 {
                assert!(
                    (b.get(mu, nu) - want[mu][nu]).abs() < 1e-12,
                    "B[{mu},{nu}]={} want {} (μ↔ν transpose?)",
                    b.get(mu, nu),
                    want[mu][nu]
                );
            }
        }
    }

    #[test]
    fn refute_b2a_a_symbol_is_not_transposed() {
        // a_symbol_generic is UNUSED by any other B2a test (fold is B2b), so this
        // is the ONLY thing exercising its κ↔λ index order today.
        let rule = TransposeProbeRule;
        let s = SectorId::new(1);
        let a = rule.a_symbol_generic(s, s, s);
        assert_eq!(a.shape(), (2, 2));
        let want = tp_expected_a();
        assert!((want[0][1] - want[1][0]).abs() > 0.1, "oracle A must be non-symmetric");
        for k in 0..2 {
            for l in 0..2 {
                assert!(
                    (a.get(k, l) - want[k][l]).abs() < 1e-12,
                    "A[{k},{l}]={} want {} (κ↔λ transpose?)",
                    a.get(k, l),
                    want[k][l]
                );
            }
        }
    }

    #[test]
    fn refute_b2a_bendright_uses_b_row_not_column() {
        // End-to-end: bend the codomain vertex μ; the ν output distribution must
        // equal ROW μ of B (coeff = coeff0·Bmat[μ,ν], coeff0=1 here). A μ↔ν swap
        // in the Bmat.get(μ,ν) call would emit COLUMN μ instead.
        let rule = TransposeProbeRule;
        let s = SectorId::new(1);
        let b = tp_expected_b();
        for mu in 1..=2usize {
            // cod [1,1]->1 vertex μ ; dom [1]->1.
            let cod = FusionTreeKey::new([s, s], s, [false, false], [], [MultiplicityIndex::new(mu).expect("test multiplicity label is one-based")]);
            let dom = FusionTreeKey::new([s], s, [false], [], []);
            let pair = FusionTreePairKey::pair(cod, dom);
            let out = generic_bendright_tree_pair(&rule, &pair).unwrap();
            // Collect coeff keyed by output domain vertex label (=ν+1).
            let mut got = [0.0f64; 2];
            for (key, coeff) in &out {
                let nu = key.domain_tree().vertices()[0].get(); // 1-based ν label
                got[nu - 1] = *coeff;
            }
            let row = &b[mu - 1];
            for nu in 0..2 {
                assert!(
                    (got[nu] - row[nu]).abs() < 1e-12,
                    "μ={mu}: ν={nu} coeff {} want ROW-μ {} (transpose ⇒ column-μ)",
                    got[nu],
                    row[nu]
                );
            }
            // Guard: distinguishable from the column (transposed) reading.
            let col = [b[0][mu - 1], b[1][mu - 1]];
            assert!(
                (got[0] - col[0]).abs() > 1e-9 || (got[1] - col[1]).abs() > 1e-9,
                "μ={mu}: row and column coincide — test cannot discriminate"
            );
        }
    }

    // ================= Stage B2b: Generic fold / multi_Fmove =================
    //
    // ORACLE PROVENANCE. All numeric constants below are TensorKit's own values
    // for `A4Irrep(3)`, extracted by running the mirrored source:
    //   TensorKit.jl @ git cfaa073 (v0.17.0), TensorKitSectors v0.3.9
    //   (Fsymbol values identical to v0.3.6, the version the B2a A4 constants
    //   cite — the A4 category data is version-stable).
    // The `multi_Fmove` tables come from `TensorKit.multi_Fmove(f)` and the
    // `foldright` tables from `TensorKit.foldright(FusionTreeBlock)`, both called
    // directly on A4 fusion trees. F-symbol arrays are transcribed ROW-MAJOR
    // over the TK axis order (μ, ν, κ, λ) — Julia's `vec()` is column-major, so
    // the transcription applies `permutedims(F,(4,3,2,1))` first. (The B2a
    // A4BendRule blocks are all transpose-symmetric, so they never exposed this;
    // the non-trivial F(3,3,3,3,3,3) block below does.)

    // Full A4Irrep(3) fusion rule with the COMPLETE F(3,3,3,3,e,f) table — the
    // B2a A4BendRule only modelled the handful of bend/A-symbol tuples, which is
    // insufficient for multi_Fmove/associator (they consult every (e,f)).
    #[derive(Clone, Copy, Debug)]
    struct A4FoldRule;
    impl FusionRule for A4FoldRule {
        fn rule_identity(&self) -> RuleIdentity { RuleIdentity::of_type::<Self>() }
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
            match sector.id() {
                1 => SectorId::new(2),
                2 => SectorId::new(1),
                _ => sector,
            }
        }
        fn fusion_channels(&self, left: SectorId, right: SectorId) -> SectorVec {
            match (left.id(), right.id()) {
                (0, x) | (x, 0) => smallvec![SectorId::new(x)],
                (3, 3) => smallvec![
                    SectorId::new(0),
                    SectorId::new(1),
                    SectorId::new(2),
                    SectorId::new(3)
                ],
                (3, _) | (_, 3) => smallvec![SectorId::new(3)],
                (1, 1) => smallvec![SectorId::new(2)],
                (2, 2) => smallvec![SectorId::new(1)],
                (1, 2) | (2, 1) => smallvec![SectorId::new(0)],
                _ => smallvec![SectorId::new(0)],
            }
        }
        fn nsymbol(&self, left: SectorId, right: SectorId, coupled: SectorId) -> usize {
            if (left.id(), right.id(), coupled.id()) == (3, 3, 3) {
                2
            } else {
                usize::from(self.fusion_channels(left, right).contains(&coupled))
            }
        }
    }
    impl GenericFusionSymbols for A4FoldRule {
        type Scalar = f64;
        fn f_symbol_generic(
            &self,
            a: SectorId,
            b: SectorId,
            c: SectorId,
            d: SectorId,
            e: SectorId,
            f: SectorId,
        ) -> GenericFArray<Self::Scalar> {
            let s = 1.0 / 3.0_f64.sqrt(); // 1/√3
            let m = -1.0 / (2.0 * 3.0_f64.sqrt()); // -1/(2√3)
            let hs = 3.0_f64.sqrt() / 2.0; // √3/2
            let ids = (a.id(), b.id(), c.id(), d.id(), e.id(), f.id());
            // The only non-trivial family: F(3,3,3,3,e,f). ROW-MAJOR (μ,ν,κ,λ).
            match ids {
                (3, 3, 3, 3, 0, 0) | (3, 3, 3, 3, 0, 1) | (3, 3, 3, 3, 0, 2)
                | (3, 3, 3, 3, 1, 0) | (3, 3, 3, 3, 1, 1) | (3, 3, 3, 3, 1, 2)
                | (3, 3, 3, 3, 2, 0) | (3, 3, 3, 3, 2, 1) | (3, 3, 3, 3, 2, 2) => {
                    GenericFArray::new(vec![1.0 / 3.0], (1, 1, 1, 1))
                }
                // A-symbol reshape F(3,3,b,b,0,3) for b∈{1,2}: TK gives [1] (1×1).
                (3, 3, 1, 1, 0, 3) | (3, 3, 2, 2, 0, 3) => {
                    GenericFArray::new(vec![1.0], (1, 1, 1, 1))
                }
                (3, 3, 3, 3, 0, 3) => GenericFArray::new(vec![s, 0.0, 0.0, s], (1, 1, 2, 2)),
                (3, 3, 3, 3, 1, 3) => GenericFArray::new(vec![m, -0.5, 0.5, m], (1, 1, 2, 2)),
                (3, 3, 3, 3, 2, 3) => GenericFArray::new(vec![m, 0.5, -0.5, m], (1, 1, 2, 2)),
                (3, 3, 3, 3, 3, 0) => GenericFArray::new(vec![s, 0.0, 0.0, s], (2, 2, 1, 1)),
                (3, 3, 3, 3, 3, 1) => GenericFArray::new(vec![m, 0.5, -0.5, m], (2, 2, 1, 1)),
                (3, 3, 3, 3, 3, 2) => GenericFArray::new(vec![m, -0.5, 0.5, m], (2, 2, 1, 1)),
                (3, 3, 3, 3, 3, 3) => GenericFArray::new(
                    vec![
                        0.5, 0.0, 0.0, -0.5, 0.0, -0.5, -0.5, 0.0, 0.0, -0.5, -0.5, 0.0, -0.5, 0.0,
                        0.0, 0.5,
                    ],
                    (2, 2, 2, 2),
                ),
                // The 9 non-trivial off-family blocks the CYCLE bends reach
                // (√3/2 = 0.8660254038). TK row-major (μ,ν,κ,λ) values.
                (1, 3, 3, 3, 3, 3) => {
                    GenericFArray::new(vec![-0.5, hs, -hs, -0.5], (1, 2, 2, 1))
                }
                (2, 3, 3, 3, 3, 3) => {
                    GenericFArray::new(vec![-0.5, -hs, hs, -0.5], (1, 2, 2, 1))
                }
                (3, 1, 3, 3, 3, 3) => {
                    GenericFArray::new(vec![-0.5, -hs, hs, -0.5], (1, 2, 1, 2))
                }
                (3, 2, 3, 3, 3, 3) => {
                    GenericFArray::new(vec![-0.5, hs, -hs, -0.5], (1, 2, 1, 2))
                }
                (3, 3, 1, 3, 3, 3) => {
                    GenericFArray::new(vec![-0.5, hs, -hs, -0.5], (2, 1, 1, 2))
                }
                (3, 3, 2, 3, 3, 3) => {
                    GenericFArray::new(vec![-0.5, -hs, hs, -0.5], (2, 1, 1, 2))
                }
                (3, 3, 3, 0, 3, 3) => {
                    GenericFArray::new(vec![1.0, 0.0, 0.0, 1.0], (2, 1, 2, 1))
                }
                (3, 3, 3, 1, 3, 3) => {
                    GenericFArray::new(vec![-0.5, hs, -hs, -0.5], (2, 1, 2, 1))
                }
                (3, 3, 3, 2, 3, 3) => {
                    GenericFArray::new(vec![-0.5, -hs, hs, -0.5], (2, 1, 2, 1))
                }
                // Everything else valid in A4 is a singleton block equal to [1]:
                // any F with a vacuum a/b/c leg, and the residual all-singleton
                // triples (e.g. F(3,1,3,0,3,3)). Shape-aware so a genuinely
                // unmodelled MULTI-dim block still panics instead of silently
                // returning a wrong scalar.
                _ => {
                    let shape = (
                        self.nsymbol(a, b, e),
                        self.nsymbol(e, c, d),
                        self.nsymbol(b, c, f),
                        self.nsymbol(a, f, d),
                    );
                    if shape == (1, 1, 1, 1) {
                        GenericFArray::new(vec![1.0], shape)
                    } else {
                        panic!("A4FoldRule: unmodelled non-singleton F{ids:?} shape={shape:?}");
                    }
                }
            }
        }
        fn r_symbol_generic(
            &self,
            _a: SectorId,
            _b: SectorId,
            _c: SectorId,
        ) -> GenericRMatrix<Self::Scalar> {
            GenericRMatrix::new(vec![1.0], 1, 1)
        }
    }
    impl GenericRigidSymbols for A4FoldRule {
        fn sqrt_dim_scalar(&self, sector: SectorId) -> Self::Scalar {
            if sector.id() == 3 { 3.0_f64.sqrt() } else { 1.0 }
        }
        fn inv_sqrt_dim_scalar(&self, sector: SectorId) -> Self::Scalar {
            if sector.id() == 3 { 1.0 / 3.0_f64.sqrt() } else { 1.0 }
        }
        fn frobenius_schur_phase_scalar(&self, _sector: SectorId) -> Self::Scalar {
            1.0
        }
    }

    fn a4f_rank3(inner: usize, v1: usize, v2: usize) -> FusionTreeKey {
        let t = SectorId::new(3);
        FusionTreeKey::new(
            [t, t, t], t,
            [false, false, false],
            [SectorId::new(inner)],
            [MultiplicityIndex::new(v1).expect("test multiplicity label is one-based"), MultiplicityIndex::new(v2).expect("test multiplicity label is one-based")],
        )
    }

    // Look up the coeff vector for the multi_Fmove output tree with the given
    // coupled sector and (single) vertex label.
    fn find_coeff(
        out: &[(FusionTreeKey, Vec<f64>)],
        coupled: usize,
        vtx: usize,
    ) -> Vec<f64> {
        out.iter()
            .find(|(tr, _)| {
                tr.coupled().id() == coupled && tr.vertices()[0].get() == vtx
            })
            .unwrap_or_else(|| panic!("no output tree coupled={coupled} vtx={vtx}"))
            .1
            .clone()
    }

    fn assert_vec(got: &[f64], want: &[f64], label: &str) {
        assert_eq!(got.len(), want.len(), "{label}: length {} != {}", got.len(), want.len());
        for (i, (g, w)) in got.iter().zip(want).enumerate() {
            assert!((g - w).abs() < 1e-10, "{label}[{i}]: {g} != {w}");
        }
    }

    // Gate 4a (A4 oracle): multi_Fmove of every rank-3 A4 (3,3,3)->3 tree matches
    // TensorKit.multi_Fmove exactly, INCLUDING the coefficient VECTORS. The
    // inner=3 rows exercise the non-trivial F(3,3,3,3,3,3) block, so the vector
    // machinery (F-slice selection, μ/ν/κ vertex indexing, λ free axis) is fully
    // discriminated here — unlike the B2a bend oracle whose A/B are I₂.
    #[test]
    fn b2b_a4_multi_fmove_matches_tensorkit() {
        let rule = A4FoldRule;
        let s = 1.0 / 3.0_f64.sqrt();
        let m = -1.0 / (2.0 * 3.0_f64.sqrt());
        let o = 1.0 / 3.0;
        // (inner,v1,v2) -> [(coupled, vtx, coeff)]. TK.multi_Fmove gold values.
        type Row = (usize, usize, usize, Vec<(usize, usize, Vec<f64>)>);
        let table: Vec<Row> = vec![
            (0, 1, 1, vec![
                (0, 1, vec![o]), (1, 1, vec![o]), (2, 1, vec![o]),
                (3, 1, vec![s, 0.0]), (3, 2, vec![0.0, s])]),
            (1, 1, 1, vec![
                (0, 1, vec![o]), (1, 1, vec![o]), (2, 1, vec![o]),
                (3, 1, vec![m, -0.5]), (3, 2, vec![0.5, m])]),
            (2, 1, 1, vec![
                (0, 1, vec![o]), (1, 1, vec![o]), (2, 1, vec![o]),
                (3, 1, vec![m, 0.5]), (3, 2, vec![-0.5, m])]),
            (3, 1, 1, vec![
                (0, 1, vec![s]), (1, 1, vec![m]), (2, 1, vec![m]),
                (3, 1, vec![0.5, 0.0]), (3, 2, vec![0.0, -0.5])]),
            (3, 1, 2, vec![
                (0, 1, vec![0.0]), (1, 1, vec![0.5]), (2, 1, vec![-0.5]),
                (3, 1, vec![0.0, -0.5]), (3, 2, vec![-0.5, 0.0])]),
            (3, 2, 1, vec![
                (0, 1, vec![0.0]), (1, 1, vec![-0.5]), (2, 1, vec![0.5]),
                (3, 1, vec![0.0, -0.5]), (3, 2, vec![-0.5, 0.0])]),
            (3, 2, 2, vec![
                (0, 1, vec![s]), (1, 1, vec![m]), (2, 1, vec![m]),
                (3, 1, vec![-0.5, 0.0]), (3, 2, vec![0.0, 0.5])]),
        ];
        for (inner, v1, v2, outputs) in table {
            let out = generic_multi_fmove_tree(&rule, &a4f_rank3(inner, v1, v2)).unwrap();
            assert_eq!(out.len(), 5, "in({inner},{v1},{v2}): 5 tails");
            for (coupled, vtx, want) in outputs {
                let got = find_coeff(&out, coupled, vtx);
                assert_vec(&got, &want, &format!("in({inner},{v1},{v2}) out(c={coupled},v={vtx})"));
            }
        }
    }

    // Build the full foldright coefficient map keyed by output tree pair. The
    // output collapses multiple (codomain', domain') paths per pair (the A-matrix
    // contraction) — the accumulator already summed them.
    fn foldright_map(
        rule: &A4FoldRule,
        pair: &FusionTreePairKey,
    ) -> std::collections::HashMap<FusionTreePairKey, f64> {
        let mut map = std::collections::HashMap::new();
        for (out, coeff) in generic_foldright_tree_pair(rule, pair).unwrap() {
            *map.entry(out).or_insert(0.0) += coeff;
        }
        map
    }

    // Gate 4b (A4 oracle): tree-level foldright U-matrix vs TensorKit.foldright.
    // rank-2 codomain: dst domain first leg dualized, U == I₂ (A4 Asymbol=I₂).
    #[test]
    fn b2b_a4_foldright_rank2_matches_tensorkit() {
        let rule = A4FoldRule;
        let t = SectorId::new(3);
        for mu in 1..=2 {
            // src: cod [3,3]->3 (vtx μ), dom [3]->3.
            let cod = FusionTreeKey::new([t, t], t, [false, false], [], [MultiplicityIndex::new(mu).expect("test multiplicity label is one-based")]);
            let dom = FusionTreeKey::new([t], t, [false], [], []);
            let map = foldright_map(&rule, &FusionTreePairKey::pair(cod, dom));
            // TK dst: cod [3]->3, dom [3,3]->3 (isdual=(true,false)) vtx μ, U[μ,μ]=1.
            let exp_cod =
                FusionTreeKey::new([t], t, [false], [], []);
            let exp_dom =
                FusionTreeKey::new([t, t], t, [true, false], [], [MultiplicityIndex::new(mu).expect("test multiplicity label is one-based")]);
            let exp = FusionTreePairKey::pair(exp_cod, exp_dom);
            for (key, coeff) in &map {
                let want = if key == &exp { 1.0 } else { 0.0 };
                assert!((coeff - want).abs() < 1e-10, "rank2 μ={mu}: coeff {coeff} want {want}");
            }
            assert!((map.get(&exp).copied().unwrap_or(0.0) - 1.0).abs() < 1e-10, "rank2 μ={mu} self");
        }
    }

    // Gate 4b (A4 oracle): rank-3 foldright — the 7×7 U-matrix that fully
    // exercises F(3,3,3,3,3,3) combined with the √dim coeff factors (the ±√3/2 =
    // ±0.8660 entries). This is the strongest fold discriminator available.
    #[test]
    fn b2b_a4_foldright_rank3_matches_tensorkit() {
        let rule = A4FoldRule;
        let t = SectorId::new(3);
        let dom = FusionTreeKey::new([t], t, [false], [], []);
        // src columns (cod [3,3,3]->3 inner=x vtx=(v1,v2), dom [3]->3).
        let cols: [(usize, usize, usize); 7] = [
            (0, 1, 1), (1, 1, 1), (2, 1, 1), (3, 1, 1), (3, 2, 1), (3, 1, 2), (3, 2, 2),
        ];
        // dst rows: (cod_coupled, cod_vtx, dom_coupled, dom_vtx). dom isdual=(true,false).
        let rows: [(usize, usize, usize, usize); 7] = [
            (0, 1, 0, 1), (1, 1, 1, 1), (2, 1, 2, 1),
            (3, 1, 3, 1), (3, 2, 3, 1), (3, 1, 3, 2), (3, 2, 3, 2),
        ];
        let sq = 1.0 / 3.0_f64.sqrt(); // 0.57735
        let hs = 3.0_f64.sqrt() / 2.0; // 0.86603
        // TK.foldright U (row,col) nonzeros; zeros elsewhere. 1-based -> 0-based.
        let u: [[f64; 7]; 7] = [
            [sq, sq, sq, 1.0, 0.0, 0.0, 1.0],
            [sq, sq, sq, -0.5, -hs, hs, -0.5],
            [sq, sq, sq, -0.5, hs, -hs, -0.5],
            [sq, -0.5 * sq, -0.5 * sq, 0.5, 0.0, 0.0, -0.5],
            [0.0, 0.5, -0.5, 0.0, -0.5, -0.5, 0.0],
            [0.0, -0.5, 0.5, 0.0, -0.5, -0.5, 0.0],
            [sq, -0.5 * sq, -0.5 * sq, -0.5, 0.0, 0.0, 0.5],
        ];
        for (ci, &(inner, v1, v2)) in cols.iter().enumerate() {
            let cod = a4f_rank3(inner, v1, v2);
            let pair = FusionTreePairKey::pair(cod, dom.clone());
            let map = foldright_map(&rule, &pair);
            for (ri, &(cc, cv, dc, dv)) in rows.iter().enumerate() {
                let ex_cod = FusionTreeKey::new(
                    [t, t], SectorId::new(cc), [false, false], [], [MultiplicityIndex::new(cv).expect("test multiplicity label is one-based")],
                );
                let ex_dom = FusionTreeKey::new(
                    [t, t], SectorId::new(dc), [true, false], [], [MultiplicityIndex::new(dv).expect("test multiplicity label is one-based")],
                );
                let key = FusionTreePairKey::pair(ex_cod, ex_dom);
                let got = map.get(&key).copied().unwrap_or(0.0);
                assert!(
                    (got - u[ri][ci]).abs() < 1e-10,
                    "U[row{ri},col{ci}] (in={inner},{v1},{v2}) got {got} want {}",
                    u[ri][ci]
                );
            }
        }
    }

    // Gate 1: fold round-trip identity. foldright then foldleft returns the
    // original pair with coefficient 1 (A-unitarity: A A† = I on the bent
    // triple). Enumerated over all rank-3 A4 vertex assignments.
    #[test]
    fn b2b_a4_fold_round_trip_identity() {
        let rule = A4FoldRule;
        let t = SectorId::new(3);
        let dom = FusionTreeKey::new([t], t, [false], [], []);
        for (inner, v1, v2) in
            [(0, 1, 1), (1, 1, 1), (2, 1, 1), (3, 1, 1), (3, 2, 1), (3, 1, 2), (3, 2, 2)]
        {
            let pair = FusionTreePairKey::pair(a4f_rank3(inner, v1, v2), dom.clone());
            let mut totals = std::collections::HashMap::new();
            for (mid, c1) in generic_foldright_tree_pair(&rule, &pair).unwrap() {
                for (out, c2) in generic_foldleft_tree_pair(&rule, &mid).unwrap() {
                    *totals.entry(out).or_insert(0.0) += c1 * c2;
                }
            }
            for (key, coeff) in &totals {
                let want = if key == &pair { 1.0 } else { 0.0 };
                assert!(
                    (coeff - want).abs() < 1e-10,
                    "fold rt in({inner},{v1},{v2}): coeff {coeff} want {want}"
                );
            }
            assert!(
                (totals.get(&pair).copied().unwrap_or(0.0) - 1.0).abs() < 1e-10,
                "fold rt in({inner},{v1},{v2}): self missing"
            );
        }
    }

    // Gate 2: cycle round-trip. cycleclockwise then cycleanticlockwise == id.
    #[test]
    fn b2b_a4_cycle_round_trip_identity() {
        let rule = A4FoldRule;
        let t = SectorId::new(3);
        let dom = FusionTreeKey::new([t], t, [false], [], []);
        for (inner, v1, v2) in [(0, 1, 1), (3, 1, 1), (3, 2, 1), (3, 1, 2), (3, 2, 2)] {
            let pair = FusionTreePairKey::pair(a4f_rank3(inner, v1, v2), dom.clone());
            let mut totals = std::collections::HashMap::new();
            for (mid, c1) in generic_cycle_clockwise_tree_pair(&rule, &pair).unwrap() {
                for (out, c2) in generic_cycle_anticlockwise_tree_pair(&rule, &mid).unwrap() {
                    *totals.entry(out).or_insert(0.0) += c1 * c2;
                }
            }
            for (key, coeff) in &totals {
                let want = if key == &pair { 1.0 } else { 0.0 };
                assert!(
                    (coeff - want).abs() < 1e-10,
                    "cycle rt in({inner},{v1},{v2}): coeff {coeff} want {want}"
                );
            }
            assert!(
                (totals.get(&pair).copied().unwrap_or(0.0) - 1.0).abs() < 1e-10,
                "cycle rt in({inner},{v1},{v2}): self missing"
            );
        }
    }

    // Residual (c): domain-rank ≥ 2. All prior generic bend/fold tests use a
    // rank-1 domain; these exercise multi_Fmove_inv on a rank-2 domain (its
    // candidates are rank-3, so the associator F-chain runs on the domain side)
    // and the rank-2-domain bend surgery. Round-trip identities, all A4 vertex
    // assignments enumerated.
    #[test]
    fn b2b_a4_fold_round_trip_domain_rank2() {
        let rule = A4FoldRule;
        let t = SectorId::new(3);
        for cod_mu in 1..=2 {
            for (dom_inner, dv) in [(0, 1), (1, 1), (2, 1), (3, 1), (3, 2)] {
                // cod [3,3]->3 (vtx cod_mu); dom [3,3]->3 inner=dom_inner (vtx dv).
                let cod = FusionTreeKey::new(
                    [t, t], t, [false, false], [], [MultiplicityIndex::new(cod_mu).expect("test multiplicity label is one-based")],
                );
                let dom = FusionTreeKey::new(
                    [t, t], t, [false, false], [], [MultiplicityIndex::new(dv).expect("test multiplicity label is one-based")],
                );
                let _ = dom_inner; // rank-2 dom has no innerline; kept for label clarity
                let pair = FusionTreePairKey::pair(cod, dom);
                let mut totals = std::collections::HashMap::new();
                for (mid, c1) in generic_foldright_tree_pair(&rule, &pair).unwrap() {
                    for (out, c2) in generic_foldleft_tree_pair(&rule, &mid).unwrap() {
                        *totals.entry(out).or_insert(0.0) += c1 * c2;
                    }
                }
                for (key, coeff) in &totals {
                    let want = if key == &pair { 1.0 } else { 0.0 };
                    assert!(
                        (coeff - want).abs() < 1e-10,
                        "fold rt dom-rank2 cod_mu={cod_mu} dv={dv}: {coeff} want {want}"
                    );
                }
                assert!(
                    (totals.get(&pair).copied().unwrap_or(0.0) - 1.0).abs() < 1e-10,
                    "fold rt dom-rank2 cod_mu={cod_mu} dv={dv}: self missing"
                );
            }
        }
    }

    #[test]
    fn b2b_a4_bend_round_trip_domain_rank2() {
        let rule = A4FoldRule;
        let t = SectorId::new(3);
        for cod_mu in 1..=2 {
            for dv in 1..=2 {
                let cod = FusionTreeKey::new(
                    [t, t], t, [false, false], [], [MultiplicityIndex::new(cod_mu).expect("test multiplicity label is one-based")],
                );
                let dom = FusionTreeKey::new(
                    [t, t], t, [false, false], [], [MultiplicityIndex::new(dv).expect("test multiplicity label is one-based")],
                );
                let pair = FusionTreePairKey::pair(cod, dom);
                let mut totals = std::collections::HashMap::new();
                for (mid, c1) in generic_bendright_tree_pair(&rule, &pair).unwrap() {
                    for (out, c2) in generic_bendleft_tree_pair(&rule, &mid).unwrap() {
                        *totals.entry(out).or_insert(0.0) += c1 * c2;
                    }
                }
                for (key, coeff) in &totals {
                    let want = if key == &pair { 1.0 } else { 0.0 };
                    assert!(
                        (coeff - want).abs() < 1e-10,
                        "bend rt dom-rank2 cod_mu={cod_mu} dv={dv}: {coeff} want {want}"
                    );
                }
                assert!(
                    (totals.get(&pair).copied().unwrap_or(0.0) - 1.0).abs() < 1e-10,
                    "bend rt dom-rank2 cod_mu={cod_mu} dv={dv}: self missing"
                );
            }
        }
    }

    // ===================== Residual (b): complex conj path =====================
    //
    // The B2a complex path (`GenericBraidScalar for Complex64`, the fold's
    // `coeff₂.braid_conj()`, and `a_symbol_generic`'s inner `conj`) was only
    // verified by source-matching. This closes it numerically: a synthetic
    // Complex64 Generic rule whose A-move / B-move are a genuinely complex 2×2
    // UNITARY U (non-Hermitian, non-real). Self-dual sector 1, N(1,1,1)=2,
    // dim=1 (all coeff factors = 1). U = (1/√2)[[1, i],[i, 1]].
    //
    // From `a_symbol_generic`: A[κ,λ] = conj(κ_a · F(1,1,1,1,0,1)[0,0,κ,λ]) with
    // κ_a=1, so setting the F block to conj(U) gives A = U. Likewise B = U from
    // the F(1,1,1,1,1,0) block. A wrong conj (missing/extra) or a μ↔ν transpose
    // flips the sign of the imaginary parts and fails both the direct check and
    // the round-trip (which needs U U† = I).
    #[derive(Clone, Copy, Debug)]
    struct ComplexUnitaryRule;
    impl FusionRule for ComplexUnitaryRule {
        fn rule_identity(&self) -> RuleIdentity { RuleIdentity::of_type::<Self>() }
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
            sector // 0 and 1 self-dual
        }
        fn fusion_channels(&self, left: SectorId, right: SectorId) -> SectorVec {
            match (left.id(), right.id()) {
                (0, x) | (x, 0) => smallvec![SectorId::new(x)],
                (1, 1) => smallvec![SectorId::new(0), SectorId::new(1)],
                _ => smallvec![SectorId::new(0)],
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
    fn cx(re: f64, im: f64) -> Complex64 {
        Complex64::new(re, im)
    }
    // U = (1/√2)[[1, i],[i, 1]], row-major.
    fn cx_u() -> [Complex64; 4] {
        let r = 1.0 / 2.0_f64.sqrt();
        [cx(r, 0.0), cx(0.0, r), cx(0.0, r), cx(r, 0.0)]
    }
    impl GenericFusionSymbols for ComplexUnitaryRule {
        type Scalar = Complex64;
        fn f_symbol_generic(
            &self,
            a: SectorId,
            b: SectorId,
            c: SectorId,
            d: SectorId,
            e: SectorId,
            f: SectorId,
        ) -> GenericFArray<Self::Scalar> {
            let u = cx_u();
            match (a.id(), b.id(), c.id(), d.id(), e.id(), f.id()) {
                // A block: A = conj(U) reshaped => set F = conj(U). shape (1,1,2,2).
                (1, 1, 1, 1, 0, 1) => GenericFArray::new(
                    vec![u[0].conj(), u[1].conj(), u[2].conj(), u[3].conj()],
                    (1, 1, 2, 2),
                ),
                // B block: B = U reshaped. shape (2,2,1,1).
                (1, 1, 1, 1, 1, 0) => {
                    GenericFArray::new(vec![u[0], u[1], u[2], u[3]], (2, 2, 1, 1))
                }
                (aa, bb, cc, _, _, _) if aa == 0 || bb == 0 || cc == 0 => {
                    GenericFArray::new(vec![cx(1.0, 0.0)], (1, 1, 1, 1))
                }
                other => panic!("ComplexUnitaryRule: unmodelled F{other:?}"),
            }
        }
        fn r_symbol_generic(
            &self,
            _a: SectorId,
            _b: SectorId,
            _c: SectorId,
        ) -> GenericRMatrix<Self::Scalar> {
            GenericRMatrix::new(vec![cx(1.0, 0.0)], 1, 1)
        }
    }
    impl GenericRigidSymbols for ComplexUnitaryRule {
        fn sqrt_dim_scalar(&self, _sector: SectorId) -> Self::Scalar {
            cx(1.0, 0.0)
        }
        fn inv_sqrt_dim_scalar(&self, _sector: SectorId) -> Self::Scalar {
            cx(1.0, 0.0)
        }
        fn frobenius_schur_phase_scalar(&self, _sector: SectorId) -> Self::Scalar {
            cx(1.0, 0.0)
        }
    }

    // Direct: foldright distributes ROW μ of the COMPLEX A-matrix (=U) to the
    // domain vertices ν. coeff(out ν) = coeff0·A[μ,ν] = U[μ,ν]. A missing conj
    // or a μ↔ν swap would produce conj(U)/Uᵀ — distinct complex numbers.
    #[test]
    fn refute_b2b_complex_foldright_reads_a_row_unconjugated() {
        let rule = ComplexUnitaryRule;
        let s = SectorId::new(1);
        let u = cx_u();
        // Sanity: U genuinely complex and non-Hermitian.
        assert!(u[1].im.abs() > 0.1, "U must be complex");
        assert!((u[1] - u[2].conj()).norm() > 0.1, "U must be non-Hermitian");
        for mu in 1..=2usize {
            let cod = FusionTreeKey::new([s, s], s, [false, false], [], [MultiplicityIndex::new(mu).expect("test multiplicity label is one-based")]);
            let dom = FusionTreeKey::new([s], s, [false], [], []);
            let pair = FusionTreePairKey::pair(cod, dom);
            let out = generic_foldright_tree_pair(&rule, &pair).unwrap();
            let mut got = [cx(0.0, 0.0); 2];
            for (key, coeff) in &out {
                let nu = key.domain_tree().vertices()[0].get();
                got[nu - 1] = *coeff;
            }
            for nu in 0..2 {
                let want = u[(mu - 1) * 2 + nu]; // ROW μ of U
                assert!(
                    (got[nu] - want).norm() < 1e-10,
                    "μ={mu} ν={nu}: {} want ROW-μ {} (conj/transpose?)",
                    got[nu],
                    want
                );
            }
            // Distinguishable from the conjugated reading.
            let want_conj = u[(mu - 1) * 2].conj();
            assert!(
                (got[0] - want_conj).norm() > 1e-9 || u[(mu - 1) * 2].im.abs() < 1e-12,
                "conj reading coincides — test cannot discriminate"
            );
        }
    }

    // Round-trip with a COMPLEX unitary A: foldright∘foldleft == id requires
    // U U† = I, so the conj in the return fold must be exactly right.
    #[test]
    fn b2b_complex_fold_round_trip_identity() {
        let rule = ComplexUnitaryRule;
        let s = SectorId::new(1);
        for mu in 1..=2usize {
            let cod = FusionTreeKey::new([s, s], s, [false, false], [], [MultiplicityIndex::new(mu).expect("test multiplicity label is one-based")]);
            let dom = FusionTreeKey::new([s], s, [false], [], []);
            let pair = FusionTreePairKey::pair(cod, dom);
            let mut totals: std::collections::HashMap<FusionTreePairKey, Complex64> =
                std::collections::HashMap::new();
            for (mid, c1) in generic_foldright_tree_pair(&rule, &pair).unwrap() {
                for (out, c2) in generic_foldleft_tree_pair(&rule, &mid).unwrap() {
                    *totals.entry(out).or_insert(cx(0.0, 0.0)) += c1 * c2;
                }
            }
            for (key, coeff) in &totals {
                let want = if key == &pair { cx(1.0, 0.0) } else { cx(0.0, 0.0) };
                assert!(
                    (coeff - want).norm() < 1e-10,
                    "cx fold rt μ={mu}: {coeff} want {want}"
                );
            }
            assert!(
                (totals.get(&pair).copied().unwrap_or(cx(0.0, 0.0)) - cx(1.0, 0.0)).norm() < 1e-10,
                "cx fold rt μ={mu}: self missing"
            );
        }
    }

    // Bend round-trip with a COMPLEX unitary B: bendright∘bendleft == id.
    #[test]
    fn b2b_complex_bend_round_trip_identity() {
        let rule = ComplexUnitaryRule;
        let s = SectorId::new(1);
        for mu in 1..=2usize {
            let cod = FusionTreeKey::new([s, s], s, [false, false], [], [MultiplicityIndex::new(mu).expect("test multiplicity label is one-based")]);
            let dom = FusionTreeKey::new([s], s, [false], [], []);
            let pair = FusionTreePairKey::pair(cod, dom);
            let mut totals: std::collections::HashMap<FusionTreePairKey, Complex64> =
                std::collections::HashMap::new();
            for (mid, c1) in generic_bendright_tree_pair(&rule, &pair).unwrap() {
                for (out, c2) in generic_bendleft_tree_pair(&rule, &mid).unwrap() {
                    *totals.entry(out).or_insert(cx(0.0, 0.0)) += c1 * c2;
                }
            }
            for (key, coeff) in &totals {
                let want = if key == &pair { cx(1.0, 0.0) } else { cx(0.0, 0.0) };
                assert!(
                    (coeff - want).norm() < 1e-10,
                    "cx bend rt μ={mu}: {coeff} want {want}"
                );
            }
        }
    }

    // ==== REFUTE (adversarial): coeff2 adjoint conj on GENUINELY complex data ====
    //
    // Gap found by the verifier: in `ComplexUnitaryRule` the domain vector
    // `coeff2` is always a real UNIT vector (rank-1 domain → seed case), so the
    // `coeff₂'` adjoint (TK `duality_manipulations.jl:279`) and the
    // `multi_Fmove_inv = conj(associator)` step (TK `basic_manipulations.jl:
    // 439/462`) are NEVER exercised on complex data by any existing test — the
    // A4 oracle is fully real, and the complex fold round-trip cancels a
    // consistent double conj error.
    //
    // This synthetic (deliberately NOT pentagon-consistent — a pure algebraic
    // fixture) rule drives one COMPLEX interior F into `coeff2` while keeping the
    // A-matrix REAL, isolating the two conj sites:
    //   * F(1,1,2,2,0,3) = 1  (real)  ⇒ Asymbol(1,2,3) is real, = 1.
    //   * F(1,3,3,2,2,3) = w  (complex) ⇒ multi_associator seed = w.
    // TK's `multi_Fmove_inv` returns conj(associator) = conj(w); TK's foldright
    // contracts coeff₂' (a SECOND conj) against transpose(A)·coeff₁, so the two
    // conjs cancel and the observable foldright coefficient is the RAW
    // associator w. Test A pins `multi_Fmove_inv = conj(w)` alone (breaks the
    // double-error symmetry); Test B pins the foldright net = w. Together they
    // rule out both a single and a double conj slip. A single missing conj at
    // EITHER site flips the observable to conj(w) ≠ w.
    #[derive(Clone, Copy, Debug)]
    struct Coeff2ConjRule;
    fn c2c_w() -> Complex64 {
        Complex64::new(0.6, 0.8) // |w| = 1, genuinely complex
    }
    impl FusionRule for Coeff2ConjRule {
        fn rule_identity(&self) -> RuleIdentity { RuleIdentity::of_type::<Self>() }
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
            sector // all self-dual
        }
        fn fusion_channels(&self, left: SectorId, right: SectorId) -> SectorVec {
            match (left.id(), right.id()) {
                (0, x) | (x, 0) => smallvec![SectorId::new(x)],
                (1, 1) => smallvec![SectorId::new(0)],
                (1, 2) | (2, 1) => smallvec![SectorId::new(3)],
                (1, 3) | (3, 1) => smallvec![SectorId::new(2)],
                (2, 3) | (3, 2) => smallvec![SectorId::new(2)],
                (3, 3) => smallvec![SectorId::new(3)],
                _ => smallvec![SectorId::new(0)],
            }
        }
        fn nsymbol(&self, left: SectorId, right: SectorId, coupled: SectorId) -> usize {
            usize::from(self.fusion_channels(left, right).contains(&coupled))
        }
    }
    impl GenericFusionSymbols for Coeff2ConjRule {
        type Scalar = Complex64;
        fn f_symbol_generic(
            &self,
            a: SectorId,
            b: SectorId,
            c: SectorId,
            d: SectorId,
            e: SectorId,
            f: SectorId,
        ) -> GenericFArray<Self::Scalar> {
            let ids = (a.id(), b.id(), c.id(), d.id(), e.id(), f.id());
            match ids {
                // Asymbol(1,2,3) reads F(dual1,1,2,2,0,3) = F(1,1,2,2,0,3): REAL.
                (1, 1, 2, 2, 0, 3) => {
                    GenericFArray::new(vec![Complex64::new(1.0, 0.0)], (1, 1, 1, 1))
                }
                // multi_associator seed for domain [3,3]->3 folded onto b=2: COMPLEX.
                (1, 3, 3, 2, 2, 3) => GenericFArray::new(vec![c2c_w()], (1, 1, 1, 1)),
                _ => {
                    let shape = (
                        self.nsymbol(a, b, e),
                        self.nsymbol(e, c, d),
                        self.nsymbol(b, c, f),
                        self.nsymbol(a, f, d),
                    );
                    if shape == (1, 1, 1, 1) {
                        GenericFArray::new(vec![Complex64::new(1.0, 0.0)], shape)
                    } else {
                        panic!("Coeff2ConjRule: unmodelled non-singleton F{ids:?} shape={shape:?}");
                    }
                }
            }
        }
        fn r_symbol_generic(
            &self,
            _a: SectorId,
            _b: SectorId,
            _c: SectorId,
        ) -> GenericRMatrix<Self::Scalar> {
            GenericRMatrix::new(vec![Complex64::new(1.0, 0.0)], 1, 1)
        }
    }
    impl GenericRigidSymbols for Coeff2ConjRule {
        fn sqrt_dim_scalar(&self, _sector: SectorId) -> Self::Scalar {
            Complex64::new(1.0, 0.0)
        }
        fn inv_sqrt_dim_scalar(&self, _sector: SectorId) -> Self::Scalar {
            Complex64::new(1.0, 0.0)
        }
        fn frobenius_schur_phase_scalar(&self, _sector: SectorId) -> Self::Scalar {
            Complex64::new(1.0, 0.0)
        }
    }

    // Test A: `multi_Fmove_inv` alone returns conj(associator) on complex F.
    #[test]
    fn refute_b2b_multi_fmove_inv_is_conj_associator_complex() {
        let rule = Coeff2ConjRule;
        let s1 = SectorId::new(1);
        let s3 = SectorId::new(3);
        let w = c2c_w();
        // domain tree [3,3] -> 3 (single vertex); leading dual(a)=1, target b=2.
        let domain = FusionTreeKey::new(
            [s3, s3], s3,
            [false, false],
            [],
            [MultiplicityIndex::ONE],
        );
        let terms =
            generic_multi_fmove_inv_tree(&rule, s1, SectorId::new(2), &domain, true).unwrap();
        assert_eq!(terms.len(), 1, "expected a single recoupled candidate");
        let (_, coeff) = &terms[0];
        assert_eq!(coeff.len(), 1, "coeff2 must be length-1 here");
        // Independent TK reading: inv coeff = conj(seed associator) = conj(w).
        assert!(
            (coeff[0] - w.conj()).norm() < 1e-12,
            "multi_Fmove_inv gave {} want conj(w)={} (A2 conj missing?)",
            coeff[0],
            w.conj()
        );
        // Discriminating: conj(w) must differ from w so the check has teeth.
        assert!((w - w.conj()).norm() > 0.1, "w not complex enough");
    }

    // Test B: foldright net observable = raw associator w (the two conjs cancel).
    // A single dropped conj at EITHER site would surface as conj(w).
    #[test]
    fn refute_b2b_foldright_net_is_raw_associator_complex() {
        let rule = Coeff2ConjRule;
        let s1 = SectorId::new(1);
        let s2 = SectorId::new(2);
        let s3 = SectorId::new(3);
        let w = c2c_w();
        // codomain [1,2] -> 3 (coeff1 = unit, A = Asymbol(1,2,3) real = 1),
        // domain [3,3] -> 3 (drives complex coeff2).
        let codomain =
            FusionTreeKey::new([s1, s2], s3, [false, false], [], [MultiplicityIndex::ONE]);
        let domain =
            FusionTreeKey::new([s3, s3], s3, [false, false], [], [MultiplicityIndex::ONE]);
        let pair = FusionTreePairKey::pair(codomain, domain);
        let out = generic_foldright_tree_pair(&rule, &pair).unwrap();
        assert_eq!(out.len(), 1, "expected a single folded term");
        let coeff = out[0].1;
        assert!(
            (coeff - w).norm() < 1e-12,
            "foldright net = {coeff} want raw associator w={w} (odd # of conj slips ⇒ conj(w))"
        );
        // The wrong (single-conj-dropped) answer is conj(w); prove distinguishable.
        assert!(
            (coeff - w.conj()).norm() > 0.1,
            "test cannot discriminate conj(w) from w"
        );
    }

    // ============ Residual (a): real non-diagonal SU(3) B-symbol ============
    //
    // B2a's A4 bend oracle could NOT discriminate a μ↔ν B-matrix transpose:
    // A4Irrep(3)'s Bsymbol is I₂ (its own transpose). It flagged that a
    // *non-diagonal Bsymbol from a real category* (SU(3)) was needed. Extracted
    // from SUNRepresentations.jl v0.4.0 + TensorKitSectors v0.3.9:
    //   Bsymbol((4,2,0),(3,1,0),(3,1,0)) = [[-1/(2√2), -√(7/8)], [√(7/8), -1/(2√2)]]
    //   Bsymbol((3,1,0),(3,2,0),(4,2,0)) = its inverse (real orthogonal, so the
    //   transpose): [[-1/(2√2), √(7/8)], [-√(7/8), -1/(2√2)]]
    // (B_fwd · B_ret = I₂, verified in Julia). These are GENUINELY non-diagonal
    // and non-symmetric, so they discriminate the μ↔ν indexing in the real bend,
    // not just the F→B reshape (which TransposeProbeRule already pins).
    //
    // dim((4,2,0))=27, dim((3,1,0))=dim((3,2,0))=15, all FS phases +1.
    // b_symbol_generic is overridden directly (the full SU(3) F-table is not
    // transcribed), so the bend surgery, coeff₀ = √dim(c)/√dim(a), μ→ν row
    // distribution and round-trip are all exercised against real categorical B.
    #[derive(Clone, Copy, Debug)]
    struct Su3BendRule;
    // ids: 1 = (4,2,0) self-dual, 2 = (3,1,0), 3 = (3,2,0) = dual((3,1,0)).
    impl FusionRule for Su3BendRule {
        fn rule_identity(&self) -> RuleIdentity { RuleIdentity::of_type::<Self>() }
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
            match sector.id() {
                2 => SectorId::new(3),
                3 => SectorId::new(2),
                _ => sector, // 0, 1 self-dual
            }
        }
        // bendright/bendleft never consult these (they use only dual, dims, fs,
        // and b_symbol_generic); provide honest N(a,b,c)=2 for the bent triples.
        fn fusion_channels(&self, left: SectorId, right: SectorId) -> SectorVec {
            match (left.id(), right.id()) {
                (0, x) | (x, 0) => smallvec![SectorId::new(x)],
                (1, 2) | (2, 1) => smallvec![SectorId::new(2)],
                (2, 3) | (3, 2) => smallvec![SectorId::new(1)],
                _ => smallvec![SectorId::new(0)],
            }
        }
        fn nsymbol(&self, left: SectorId, right: SectorId, coupled: SectorId) -> usize {
            match (left.id(), right.id(), coupled.id()) {
                (1, 2, 2) | (2, 3, 1) => 2,
                _ => usize::from(self.fusion_channels(left, right).contains(&coupled)),
            }
        }
    }
    impl GenericFusionSymbols for Su3BendRule {
        type Scalar = f64;
        fn f_symbol_generic(
            &self,
            _a: SectorId,
            _b: SectorId,
            _c: SectorId,
            _d: SectorId,
            _e: SectorId,
            _f: SectorId,
        ) -> GenericFArray<Self::Scalar> {
            unreachable!("b_symbol_generic is overridden; F is never read")
        }
        fn r_symbol_generic(
            &self,
            _a: SectorId,
            _b: SectorId,
            _c: SectorId,
        ) -> GenericRMatrix<Self::Scalar> {
            GenericRMatrix::new(vec![1.0], 1, 1)
        }
    }
    impl GenericRigidSymbols for Su3BendRule {
        fn sqrt_dim_scalar(&self, sector: SectorId) -> Self::Scalar {
            match sector.id() {
                1 => 27.0_f64.sqrt(),
                2 | 3 => 15.0_f64.sqrt(),
                _ => 1.0,
            }
        }
        fn inv_sqrt_dim_scalar(&self, sector: SectorId) -> Self::Scalar {
            1.0 / self.sqrt_dim_scalar(sector)
        }
        fn frobenius_schur_phase_scalar(&self, _sector: SectorId) -> Self::Scalar {
            1.0
        }
        fn b_symbol_generic(
            &self,
            a: SectorId,
            b: SectorId,
            c: SectorId,
        ) -> GenericRMatrix<Self::Scalar> {
            let e = -1.0 / (2.0 * 2.0_f64.sqrt()); // -1/(2√2) = -0.35355339
            let g = (7.0_f64 / 8.0).sqrt(); //  √(7/8)  =  0.93541435
            match (a.id(), b.id(), c.id()) {
                (1, 2, 2) => GenericRMatrix::new(vec![e, -g, g, e], 2, 2), // B_fwd
                (2, 3, 1) => GenericRMatrix::new(vec![e, g, -g, e], 2, 2), // B_ret
                other => panic!("Su3BendRule: unmodelled B{other:?}"),
            }
        }
    }

    fn su3_bfwd() -> [[f64; 2]; 2] {
        let e = -1.0 / (2.0 * 2.0_f64.sqrt());
        let g = (7.0_f64 / 8.0).sqrt();
        [[e, -g], [g, e]]
    }

    // Real-categorical bend oracle: bendright distributes coeff₀·ROW μ of the
    // NON-DIAGONAL SU(3) B to the domain vertices ν. A μ↔ν swap would emit
    // COLUMN μ; B is non-symmetric (B[0,1]≠B[1,0]) so the two are distinct.
    #[test]
    fn b2b_su3_bendright_uses_b_row_not_column() {
        let rule = Su3BendRule;
        let s42 = SectorId::new(1);
        let s31 = SectorId::new(2);
        let b = su3_bfwd();
        let coeff0 = 15.0_f64.sqrt() / 27.0_f64.sqrt(); // √dim(31)/√dim(42)
        for mu in 1..=2usize {
            // cod [42,31]->31 (vtx μ), dom [31]->31.
            let cod = FusionTreeKey::new(
                [s42, s31], s31, [false, false], [], [MultiplicityIndex::new(mu).expect("test multiplicity label is one-based")],
            );
            let dom =
                FusionTreeKey::new([s31], s31, [false], [], []);
            let out = generic_bendright_tree_pair(&rule, &FusionTreePairKey::pair(cod, dom))
                .unwrap();
            let mut got = [0.0f64; 2];
            for (key, coeff) in &out {
                let nu = key.domain_tree().vertices().last().unwrap().get();
                got[nu - 1] = *coeff;
            }
            for nu in 0..2 {
                let want = coeff0 * b[mu - 1][nu]; // ROW μ
                assert!(
                    (got[nu] - want).abs() < 1e-10,
                    "μ={mu} ν={nu}: {} want coeff0·ROW-μ {} (transpose ⇒ column)",
                    got[nu],
                    want
                );
            }
            // Distinguishable from the transposed (column) reading.
            let col = coeff0 * b[if mu == 1 { 1 } else { 0 }][mu - 1];
            assert!((got[0] - col).abs() > 1e-9, "μ={mu}: row/column coincide");
        }
    }

    #[test]
    fn generic_full_key_block_composition_matches_per_source_replay() {
        let rule = Su3BendRule;
        let s42 = SectorId::new(1);
        let s31 = SectorId::new(2);
        let basis = (1..=2)
            .map(|mu| {
                FusionTreePairKey::pair(
                    FusionTreeKey::new(
                        [s42, s31],
                        s31,
                        [false, false],
                        [],
                        [MultiplicityIndex::new(mu)
                            .expect("test multiplicity label is one-based")],
                    ),
                    FusionTreeKey::new([s31], s31, [false], [], []),
                )
            })
            .collect::<Vec<_>>();
        let mut columns = DenseColumns::with_capacity(basis.len(), basis.len());
        for source in 0..basis.len() {
            let row = columns.push_empty_row();
            columns.row_mut(row)[source] = Some(1.0);
        }

        let (dst_basis, dst_columns) =
            compose_generic_block_terms(&rule, &basis, &columns, |rule, key| {
                generic_bendright_tree_pair(rule, key)
            })
            .unwrap();

        let oracle = basis
            .iter()
            .map(|source| {
                compose_generic_tree_pair_terms(
                    &rule,
                    vec![(source.clone(), 1.0)],
                    |rule, key| generic_bendright_tree_pair(rule, key),
                )
                .unwrap()
            })
            .collect::<Vec<_>>();
        let mut expected_order = Vec::<FusionTreePairKey>::new();
        for rows in &oracle {
            for (key, _) in rows {
                if !expected_order.iter().any(|existing| existing == key) {
                    expected_order.push(key.clone());
                }
            }
        }
        assert_eq!(dst_basis, expected_order);
        assert_eq!(dst_columns.num_src, basis.len());
        assert_eq!(dst_columns.num_rows, dst_basis.len());
        for destination_row in 0..dst_basis.len() {
            let domain_vertices = dst_basis[destination_row].domain_tree().vertices();
            assert_eq!(
                domain_vertices.last().map(|index| index.get()),
                Some(destination_row + 1)
            );
            for source in 0..basis.len() {
                let got = dst_columns.row(destination_row)[source].unwrap_or(0.0);
                let want = oracle[source]
                    .iter()
                    .find_map(|(key, coeff)| (key == &dst_basis[destination_row]).then_some(*coeff))
                    .unwrap_or(0.0);
                assert!((got - want).abs() < 1e-12);
            }
        }
    }

    // Round-trip with a real non-diagonal SU(3) B: bendright∘bendleft == id
    // (B_fwd · B_ret = I₂), exercising the non-trivial off-diagonal mixing.
    #[test]
    fn b2b_su3_bend_round_trip_identity() {
        let rule = Su3BendRule;
        let s42 = SectorId::new(1);
        let s31 = SectorId::new(2);
        for mu in 1..=2usize {
            let cod = FusionTreeKey::new(
                [s42, s31], s31, [false, false], [], [MultiplicityIndex::new(mu).expect("test multiplicity label is one-based")],
            );
            let dom =
                FusionTreeKey::new([s31], s31, [false], [], []);
            let pair = FusionTreePairKey::pair(cod, dom);
            let mut totals = std::collections::HashMap::new();
            for (mid, c1) in generic_bendright_tree_pair(&rule, &pair).unwrap() {
                for (out, c2) in generic_bendleft_tree_pair(&rule, &mid).unwrap() {
                    *totals.entry(out).or_insert(0.0) += c1 * c2;
                }
            }
            for (key, coeff) in &totals {
                let want = if key == &pair { 1.0 } else { 0.0 };
                assert!(
                    (coeff - want).abs() < 1e-10,
                    "su3 bend rt μ={mu}: {coeff} want {want}"
                );
            }
            assert!(
                (totals.get(&pair).copied().unwrap_or(0.0) - 1.0).abs() < 1e-10,
                "su3 bend rt μ={mu}: self missing"
            );
        }
    }

    // ==================================================================
    // Stage B2c: generic tree-pair permute / braid / transpose composers.
    // These are thin structural mirrors of the multiplicity-free tree-pair
    // functions, chaining the adversarially-verified B1/B2a/B2b primitives
    // (generic_braid_tree, generic_repartition_tree_pair, generic_cycle_*).
    // The tests below prove the COMPOSITION adds no math: each composer
    // equals the hand-chained primitives it is built from.
    // ==================================================================

    use std::collections::HashMap;

    fn map_terms(terms: Vec<(FusionTreePairKey, f64)>) -> HashMap<FusionTreePairKey, f64> {
        let mut map = HashMap::new();
        for (key, coeff) in terms {
            *map.entry(key).or_insert(0.0) += coeff;
        }
        map
    }

    fn assert_term_maps_eq(
        got: &HashMap<FusionTreePairKey, f64>,
        want: &HashMap<FusionTreePairKey, f64>,
        label: &str,
    ) {
        let mut keys: std::collections::HashSet<&FusionTreePairKey> = got.keys().collect();
        keys.extend(want.keys());
        for key in keys {
            let g = got.get(key).copied().unwrap_or(0.0);
            let w = want.get(key).copied().unwrap_or(0.0);
            assert!((g - w).abs() < 1e-10, "{label}: coeff {g} != {w}");
        }
    }

    fn assert_identity_term_map(
        got: &HashMap<FusionTreePairKey, f64>,
        self_pair: &FusionTreePairKey,
        label: &str,
    ) {
        for (key, coeff) in got {
            let want = if key == self_pair { 1.0 } else { 0.0 };
            assert!((coeff - want).abs() < 1e-10, "{label}: coeff {coeff} != {want}");
        }
        assert!(
            (got.get(self_pair).copied().unwrap_or(0.0) - 1.0).abs() < 1e-10,
            "{label}: self coefficient missing"
        );
    }

    // A4 rank-1/rank-1 pair: cod [3]->3, dom [3]->3 (coupled sector 3).
    fn a4_pair_rank1_1() -> FusionTreePairKey {
        let t = SectorId::new(3);
        let cod = FusionTreeKey::new([t], t, [false], [], []);
        let dom = FusionTreeKey::new([t], t, [false], [], []);
        FusionTreePairKey::pair(cod, dom)
    }

    // A4 rank-2/rank-1 pair: cod [3,3]->3 (vtx μ), dom [3]->3 — an
    // outer-multiplicity tree pair with N(3,3,3)=2.
    fn a4_pair_rank2_1(mu: usize) -> FusionTreePairKey {
        let t = SectorId::new(3);
        let cod = FusionTreeKey::new([t, t], t, [false, false], [], [MultiplicityIndex::new(mu).expect("test multiplicity label is one-based")]);
        let dom = FusionTreeKey::new([t], t, [false], [], []);
        FusionTreePairKey::pair(cod, dom)
    }

    // Gate B2c-1: transpose (planar cyclic permutation) round-trips to the
    // identity. `generic_transpose_tree_pair` chains repartition + the fold/bend
    // A-move (`generic_cycle_*`); applying the swap [1],[0] and then its inverse
    // [1],[0] again must return the original pair with coefficient 1. Run on
    // A4FoldRule, whose A-move is a genuinely non-diagonal outer-multiplicity
    // move, so any coefficient error in the composition breaks the identity.
    #[test]
    fn b2c_generic_transpose_round_trips_to_identity() {
        let rule = A4FoldRule;
        let pair = a4_pair_rank1_1();
        let forward = generic_transpose_tree_pair(&rule, &pair, &[1], &[0]).unwrap();
        let mut totals = HashMap::new();
        for (mid, c1) in forward {
            for (out, c2) in generic_transpose_tree_pair(&rule, &mid, &[1], &[0]).unwrap() {
                *totals.entry(out).or_insert(0.0) += c1 * c2;
            }
        }
        assert_identity_term_map(&totals, &pair, "A4 transpose round-trip");
    }

    // Gate B2c-2: braid with the IDENTITY permutation is the identity map on an
    // outer-multiplicity tree pair. The braid decomposes to zero swaps, so the
    // composer runs repartition-to-all-codomain and back (a verified bend
    // round-trip) around a no-op braid, plus the tree-pair reconstruction
    // closure. Run on A4BendRule (rigid, OM); no braid R-symbol is invoked.
    #[test]
    fn b2c_generic_braid_identity_permutation_is_identity() {
        let rule = A4BendRule;
        for mu in 1..=2 {
            let pair = a4_pair_rank2_1(mu);
            // codomain axes [0,1], domain axis [2]; identity level order.
            let got = map_terms(
                generic_braid_tree_pair(&rule, &pair, &[0, 1], &[2], &[0, 1], &[2]).unwrap(),
            );
            assert_identity_term_map(&got, &pair, &format!("A4 braid-id μ={mu}"));
        }
    }

    // Gate B2c-3: `generic_permute_tree_pair` == `generic_braid_tree_pair` under
    // the identity level order (the definitional relation the mult-free path
    // relies on), and the symmetric-braiding guard is honored. Uses the identity
    // permutation because a non-trivial multi-leg *braid* needs a fully-modeled
    // braiding generic rule (the SU(3) provider, Stage B3); the composition
    // itself is a line-for-line mirror of the fully-tested
    // `multiplicity_free_braid_tree_pair`, and its braid step
    // (`generic_braid_tree`) is independently adversarially verified in B1.
    #[test]
    fn b2c_generic_permute_agrees_with_default_level_braid() {
        let rule = A4BendRule; // Bosonic ⇒ symmetric braiding.
        for mu in 1..=2 {
            let pair = a4_pair_rank2_1(mu);
            let permuted = map_terms(
                generic_permute_tree_pair(&rule, &pair, &[0, 1], &[2]).unwrap(),
            );
            // default levels: codomain [0,1], domain [2].
            let braided = map_terms(
                generic_braid_tree_pair(&rule, &pair, &[0, 1], &[2], &[0, 1], &[2]).unwrap(),
            );
            assert_term_maps_eq(&permuted, &braided, &format!("A4 permute==braid μ={mu}"));
            // And the identity permutation is a genuine no-op.
            assert_identity_term_map(&permuted, &pair, &format!("A4 permute-id μ={mu}"));
        }
    }

    // ===================== Stage B3b: SU(3) table provider ===================
    //
    // Gate 1 (TK oracle): every number below is TensorKit 0.17.0 (jCjQQ) +
    // SUNRepresentations 0.4.0's OWN output — raw SUNRep F/R/dim/dual/FS, and
    // TK's OWN `artin_braid`/`bendright` on a `FusionTreeBlock{SUNIrrep{3}}` —
    // regenerated by `tools/su3-table-gen` + the scratch oracle scripts. The
    // Rust provider (`Su3FusionRule` over the checked-in `su3_table.bin`) must
    // reproduce them to 1e-10. A row/column transpose, a conj-placement slip, or
    // a wrong table byte all break these.
    fn su3() -> Su3FusionRule {
        Su3FusionRule::new()
    }

    fn su3_id(p: u8, q: u8) -> SectorId {
        su3().sector_of(p, q).expect("(p,q) must be in the dim<=27 table")
    }

    // --- Stage B3c-1: group-agnostic reader (SU(4) DATA-ONLY smoke) -------
    // The exact same `TabulatedFusionRule` code, loaded from a *different*
    // group's blob (a small `SU(4)`, dim ≤ 15, table generated by the now
    // N-parametric `tools/sun-table-gen/gen.jl`), must fuse correctly with
    // ZERO Rust changes — the whole point of the generalisation. Values are
    // SUNRepresentations' own SU(4) fusion (id layout in the test).
    static SU4_TABLE_BYTES: &[u8] = include_bytes!("testdata/su4_table.bin");

    fn rehash_table(bytes: &mut [u8]) {
        let hash = fnv1a64(&bytes[20..]).to_le_bytes();
        bytes[12..20].copy_from_slice(&hash);
    }

    fn symbol_record_ranges(bytes: &[u8]) -> (usize, std::ops::Range<usize>, usize, std::ops::Range<usize>) {
        let mut cursor = Cursor { bytes, pos: 4 };
        assert_eq!(cursor.u32().unwrap(), 3);
        let rank = cursor.u32().unwrap() as usize - 1;
        cursor.u64().unwrap();
        let n_irreps = cursor.u32().unwrap() as usize;
        cursor.take(n_irreps * (rank + 6)).unwrap();
        let n_pairs = cursor.u32().unwrap() as usize;
        for _ in 0..n_pairs {
            cursor.take(2).unwrap();
            let n_channels = cursor.u8().unwrap() as usize;
            cursor.take(n_channels * 2).unwrap();
        }
        let r_count_offset = cursor.pos;
        let n_r = cursor.u32().unwrap();
        assert!(n_r > 0);
        let r_start = cursor.pos;
        cursor.take(3).unwrap();
        let rows = cursor.u8().unwrap() as usize;
        let cols = cursor.u8().unwrap() as usize;
        cursor.take(rows * cols * 8).unwrap();
        let r_range = r_start..cursor.pos;
        for _ in 1..n_r {
            cursor.take(3).unwrap();
            let rows = cursor.u8().unwrap() as usize;
            let cols = cursor.u8().unwrap() as usize;
            cursor.take(rows * cols * 8).unwrap();
        }
        let f_count_offset = cursor.pos;
        let n_f = cursor.u32().unwrap();
        assert!(n_f > 0);
        let f_start = cursor.pos;
        cursor.take(6).unwrap();
        let shape = [cursor.u8().unwrap(), cursor.u8().unwrap(), cursor.u8().unwrap(), cursor.u8().unwrap()];
        let len = shape.into_iter().map(usize::from).product::<usize>();
        cursor.take(len * 8).unwrap();
        (r_count_offset, r_range, f_count_offset, f_start..cursor.pos)
    }

    fn remove_record(bytes: &[u8], count_offset: usize, range: std::ops::Range<usize>) -> Vec<u8> {
        let mut mutated = bytes.to_vec();
        let count = u32::from_le_bytes(mutated[count_offset..count_offset + 4].try_into().unwrap());
        mutated[count_offset..count_offset + 4].copy_from_slice(&(count - 1).to_le_bytes());
        mutated.drain(range);
        rehash_table(&mut mutated);
        mutated
    }

    fn first_nonsymmetric_f_shape(bytes: &[u8]) -> usize {
        let (_, _, f_count_offset, _) = symbol_record_ranges(bytes);
        let mut cursor = Cursor { bytes, pos: f_count_offset };
        let n_f = cursor.u32().unwrap();
        for _ in 0..n_f {
            cursor.take(6).unwrap();
            let shape_offset = cursor.pos;
            let shape = [
                cursor.u8().unwrap(),
                cursor.u8().unwrap(),
                cursor.u8().unwrap(),
                cursor.u8().unwrap(),
            ];
            let len = shape.into_iter().map(usize::from).product::<usize>();
            cursor.take(len * 8).unwrap();
            if shape[0] != shape[1] {
                return shape_offset;
            }
        }
        panic!("fixture must contain a non-symmetric F shape")
    }

    #[test]
    fn tabulated_loader_rejects_truncated_and_overflowing_inputs() {
        assert!(matches!(
            TabulatedFusionRule::try_from_bytes(&SU4_TABLE_BYTES[..8], "truncated"),
            Err(TableError::Truncated { .. })
        ));
        let mut overflowing = SU4_TABLE_BYTES.to_vec();
        overflowing[20..24].copy_from_slice(&u32::MAX.to_le_bytes());
        rehash_table(&mut overflowing);
        assert!(matches!(
            TabulatedFusionRule::try_from_bytes(&overflowing, "overflowing"),
            Err(TableError::Invalid { section: "irreps", .. })
        ));
    }

    #[test]
    fn tabulated_loader_rejects_record_counts_above_metadata_budget() {
        let (r_count, _, _, _) = symbol_record_ranges(SU4_TABLE_BYTES);
        let mut excessive = SU4_TABLE_BYTES.to_vec();
        excessive[r_count..r_count + 4]
            .copy_from_slice(&((MAX_METADATA_ENTRIES as u32) + 1).to_le_bytes());
        rehash_table(&mut excessive);
        assert!(matches!(
            TabulatedFusionRule::try_from_bytes(&excessive, "metadata-budget"),
            Err(TableError::Invalid { section: "R", .. })
        ));

        let mut used = MAX_METADATA_ENTRIES - 200;
        consume_metadata_budget("fusion", &mut used, u8::MAX as usize).unwrap_err();
    }

    #[test]
    fn tabulated_loader_rejects_missing_admissible_symbols() {
        let (r_count, r_range, f_count, f_range) = symbol_record_ranges(SU4_TABLE_BYTES);
        let missing_r = remove_record(SU4_TABLE_BYTES, r_count, r_range.clone());
        assert!(matches!(
            TabulatedFusionRule::try_from_bytes(&missing_r, "missing-r"),
            Err(TableError::MissingR(_))
        ));

        let removed = f_range.end - f_range.start;
        let missing_f = remove_record(SU4_TABLE_BYTES, f_count, f_range);
        assert!(matches!(
            TabulatedFusionRule::try_from_bytes(&missing_f, "missing-f"),
            Err(TableError::MissingF(_))
        ));
        assert!(removed > 10);
    }

    #[test]
    fn tabulated_loader_rejects_bad_ids_and_symbol_shapes() {
        let (_, r_range, _, _) = symbol_record_ranges(SU4_TABLE_BYTES);
        let mut bad_id = SU4_TABLE_BYTES.to_vec();
        bad_id[r_range.start] = u8::MAX;
        rehash_table(&mut bad_id);
        assert!(matches!(
            TabulatedFusionRule::try_from_bytes(&bad_id, "bad-id"),
            Err(TableError::Invalid { section: "R", .. })
        ));

        let shape_offset = first_nonsymmetric_f_shape(SU3_TABLE_BYTES);
        let mut bad_shape = SU3_TABLE_BYTES.to_vec();
        bad_shape.swap(shape_offset, shape_offset + 1);
        rehash_table(&mut bad_shape);
        assert!(matches!(
            TabulatedFusionRule::try_from_bytes(&bad_shape, "bad-shape"),
            Err(TableError::Invalid { section: "F", .. })
        ));
    }

    #[test]
    fn tabulated_loader_bounds_symbol_allocation_before_reserving() {
        let (_, _, _, first_f) = symbol_record_ranges(SU4_TABLE_BYTES);
        let shape_offset = first_f.start + 6;
        let mut oversized = SU4_TABLE_BYTES.to_vec();
        oversized[shape_offset..shape_offset + 4].fill(u8::MAX);
        rehash_table(&mut oversized);
        assert!(matches!(
            TabulatedFusionRule::try_from_bytes(&oversized, "oversized-shape"),
            Err(TableError::Invalid { section: "F", .. }) | Err(TableError::Truncated { .. })
        ));
    }

    #[test]
    fn tabulated_loader_rejects_nonunitary_f_associator() {
        let (_, _, _, first_f) = symbol_record_ranges(SU4_TABLE_BYTES);
        let mut corrupted = SU4_TABLE_BYTES.to_vec();
        corrupted[first_f.start + 10..first_f.start + 18].copy_from_slice(&0.0f64.to_le_bytes());
        rehash_table(&mut corrupted);
        assert!(matches!(
            TabulatedFusionRule::try_from_bytes(&corrupted, "nonunitary-f"),
            Err(TableError::Invalid { section: "F", .. })
        ));
    }

    #[test]
    fn tabulated_f_coherence_rejects_excessive_gram_work_before_allocation() {
        let mut nsym = FxHashMap::default();
        nsym.insert((0, 0, 0), 33);
        let mut fsymbols = FxHashMap::default();
        fsymbols.insert(
            [0; 6],
            GenericFArray {
                data: Vec::<f64>::new(),
                shape: (33, 33, 33, 33),
            },
        );
        let mut covered = FxHashMap::default();
        covered.insert((0, 0), smallvec![SectorId::new(0)]);
        assert!(matches!(
            validate_f_unitarity(&nsym, &fsymbols, &covered),
            Err(TableError::Invalid { section: "F", .. })
        ));
    }

    #[test]
    fn tabulated_f_completeness_charges_candidates_with_missing_fourth_nsymbol() {
        let mut nsym = FxHashMap::default();
        for a in 0..101u8 {
            nsym.insert((a, 250, 249), 1);
        }
        for c in 0..101u8 {
            nsym.insert((249, c, 248), 1);
            for f in 100..200u8 {
                nsym.insert((250, c, f), 1);
            }
        }
        assert!(matches!(
            validate_f_completeness(&nsym, &FxHashMap::default()),
            Err(TableError::Invalid { section: "F", .. })
        ));
    }

    #[test]
    fn tabulated_f_completeness_charges_missing_second_pair_lookups() {
        let mut nsym = FxHashMap::default();
        for a in 0..3u8 {
            nsym.insert((a, 250, 249), 1);
        }
        for c in 0..3u8 {
            nsym.insert((249, c, 248), 1);
        }
        assert!(matches!(
            validate_f_completeness_with_limit(&nsym, &FxHashMap::default(), 4),
            Err(TableError::Invalid { section: "F", .. })
        ));
    }

    #[test]
    fn tabulated_symbols_keep_proven_forbidden_tuples_zero() {
        let rule = su3();
        let three = su3_id(1, 0);
        let vacuum = rule.vacuum();
        let r = rule.r_symbol_generic(three, three, vacuum);
        assert_eq!(r.shape(), (0, 0));
        assert!(r.data().is_empty());

        let f = rule.f_symbol_generic(three, three, three, vacuum, vacuum, vacuum);
        assert!(f.shape().0 == 0 || f.shape().1 == 0 || f.shape().2 == 0 || f.shape().3 == 0);
        assert!(f.data().is_empty());
    }

    #[test]
    fn tabulated_symbol_lookup_rejects_sector_ids_that_do_not_fit_the_table() {
        let rule = su3();
        let valid = rule.vacuum();
        for invalid in [SectorId::new(256), SectorId::new(usize::MAX)] {
            assert!(std::panic::catch_unwind(|| rule.r_symbol_generic(invalid, valid, valid)).is_err());
            assert!(
                std::panic::catch_unwind(|| {
                    rule.f_symbol_generic(invalid, valid, valid, valid, valid, valid)
                })
                .is_err()
            );
        }
    }

    #[test]
    fn generic_symbol_shapes_are_checked_in_release_builds() {
        assert_eq!(
            GenericFArray::try_new(vec![0.0; 3], (1, 1, 2, 2)).unwrap_err().expected_len,
            Some(4)
        );
        assert_eq!(
            GenericRMatrix::try_new(vec![0.0; 3], 2, 2).unwrap_err().expected_len,
            Some(4)
        );
        assert_eq!(
            GenericRMatrix::<f64>::try_new(Vec::new(), usize::MAX, 2).unwrap_err().expected_len,
            None
        );
    }

    #[test]
    fn b3c1_su4_table_is_data_only() {
        let rule = TabulatedFusionRule::try_from_bytes(SU4_TABLE_BYTES, "su4_table.bin").unwrap();
        assert_eq!(rule.group_n(), 4);
        // FNV self-check already passed in `from_bytes`; identity is set.
        assert_ne!(rule.provenance(), 0);
        assert_ne!(rule.provenance(), su3().provenance(), "distinct group ⇒ distinct table");
        assert_eq!(rule.fusion_style(), FusionStyleKind::Generic);
        assert!(rule.braiding_style().is_symmetric());
        // id layout (sorted by (dim, label)): 0=1, 1=4̄(0,0,1), 2=4(1,0,0),
        // 3=6(0,1,0), 4=10̄(0,0,2), 5=10(2,0,0), 6=15(1,0,1).
        let four = rule.sector_of_label(&[1, 0, 0]).expect("4 in table");
        let fourbar = rule.sector_of_label(&[0, 0, 1]).expect("4̄ in table");
        let six = rule.sector_of_label(&[0, 1, 0]).expect("6 in table");
        let ten = rule.sector_of_label(&[2, 0, 0]).expect("10 in table");
        let fifteen = rule.sector_of_label(&[1, 0, 1]).expect("15 in table");
        assert_eq!(rule.vacuum(), SectorId::new(0));
        assert_eq!(rule.label(four), &[1, 0, 0]);
        // dual involution + the 3-component label (proves labels are NOT (p,q)).
        assert_eq!(rule.dual(four), fourbar);
        assert_eq!(rule.dual(fourbar), four);
        assert_eq!(rule.dual(six), six); // 6 self-dual
        assert_eq!(rule.dual(fifteen), fifteen); // adjoint self-dual
        // 4 ⊗ 4̄ = 1 ⊕ 15.
        let mut ch = rule.fusion_channels(four, fourbar);
        ch.sort_unstable();
        assert_eq!(ch.to_vec(), vec![rule.vacuum(), fifteen]);
        // 4 ⊗ 4 = 6 ⊕ 10.
        let mut ch2 = rule.fusion_channels(four, four);
        ch2.sort_unstable();
        assert_eq!(ch2.to_vec(), vec![six, ten]);
        assert_eq!(rule.nsymbol(four, fourbar, fifteen), 1);
        // quantum dims from the blob (√dim² round-trips the integer dim).
        let d = |s| (rule.sqrt_dim_scalar(s) * rule.sqrt_dim_scalar(s)).round() as i64;
        assert_eq!([d(four), d(six), d(ten), d(fifteen)], [4, 6, 10, 15]);
    }

    #[test]
    #[ignore = "requires a freshly generated artifact path"]
    fn generated_su4_artifact_passes_the_production_loader() {
        let path = std::env::var("TENET_GENERATED_SU4").expect("TENET_GENERATED_SU4 must be set");
        let bytes = std::fs::read(path).unwrap();
        let rule = TabulatedFusionRule::try_from_bytes(&bytes, "generated-su4").unwrap();
        assert_eq!(rule.group_n(), 4);
        // What: the CI-generated SU(4) smoke artifact uses the current
        // order-preserving, group-agnostic table generator.
        assert_eq!(rule.provenance(), 0xcffd_d18b_bba9_155a);
    }

    // --- table integrity + hard-error boundary ---------------------------
    #[test]
    fn b3b_su3_table_loads_and_reports_identity() {
        let rule = su3();
        assert_ne!(rule.provenance(), 0, "provenance hash must be set");
        assert_eq!(rule.vacuum(), SectorId::new(0));
        assert_eq!(rule.fusion_style(), FusionStyleKind::Generic);
        assert!(rule.braiding_style().is_symmetric()); // Bosonic
        // dim / dual / FS spot-checks vs SUNRepresentations.
        let eight = su3_id(1, 1);
        let three = su3_id(1, 0);
        assert_eq!(rule.dynkin(eight), (1, 1));
        assert_eq!(rule.dual(three), su3_id(0, 1)); // dual(3) = 3̄
        assert_eq!(rule.dual(eight), eight); // 8 self-dual
        assert_eq!(rule.frobenius_schur_phase_scalar(eight), 1.0);
        assert_eq!(rule.frobenius_schur_phase_scalar(three), 1.0);
        assert!((rule.sqrt_dim_scalar(eight) - 8.0_f64.sqrt()).abs() < 1e-12);
        assert!((rule.sqrt_dim_scalar(su3_id(2, 2)) - 27.0_f64.sqrt()).abs() < 1e-12);
    }

    #[test]
    fn b3b_su3_covers_and_nsymbol() {
        let rule = su3();
        let eight = su3_id(1, 1);
        let three = su3_id(1, 0);
        // 8⊗8 closes (∋ 27); 3⊗3̄ closes (∋ 1,8).
        assert!(rule.covers(eight, eight));
        assert!(rule.covers(three, rule.dual(three)));
        // Genuine outer multiplicity: N(8,8,8) = 2.
        assert_eq!(rule.nsymbol(eight, eight, eight), 2);
        assert_eq!(rule.nsymbol(three, three, su3_id(2, 0)), 1); // 3⊗3 ∋ 6, N=1
        // 8⊗10 escapes (∋ 35): not covered.
        assert!(!rule.covers(eight, su3_id(3, 0)));
    }

    #[test]
    fn b3b_su3_channel_order_matches_tensorkit_directproduct() {
        let rule = su3();
        let eight = su3_id(1, 1);
        let channels = rule.fusion_channels(eight, eight);

        // What: the public channel sequence and every sector-keyed
        // multiplicity follow TensorKit/SUNRepresentations' directproduct
        // basis order: 1, 8, 27, 10bar, 10.
        assert_eq!(
            channels.iter().map(|sector| sector.id()).collect::<Vec<_>>(),
            vec![0, 5, 16, 6, 7],
        );
        assert_eq!(
            channels
                .iter()
                .map(|&sector| (sector.id(), rule.nsymbol(eight, eight, sector)))
                .collect::<Vec<_>>(),
            vec![(0, 1), (5, 2), (16, 1), (6, 1), (7, 1)],
        );

        let twenty_seven = su3_id(2, 2);
        assert!(!rule.covers(twenty_seven, eight));
        let in_table = rule.fusion_channels_in_table(twenty_seven, eight);
        // What: an escaping pair retains the order-preserving in-table
        // subsequence of SUN directproduct(27, 8): 8, 27, 10bar, 10.
        assert_eq!(
            in_table
                .iter()
                .map(|sector| sector.id())
                .collect::<Vec<_>>(),
            vec![5, 16, 6, 7],
        );
        assert_eq!(
            in_table
                .iter()
                .map(|&sector| {
                    (
                        sector.id(),
                        rule.nsymbol(twenty_seven, eight, sector),
                    )
                })
                .collect::<Vec<_>>(),
            vec![(5, 1), (16, 2), (6, 1), (7, 1)],
        );

        // What: changing sequence order does not change keyed coefficient
        // lookup or the genuine outer-multiplicity matrix.
        let r888 = rule.r_symbol_generic(eight, eight, eight);
        assert_eq!(r888.shape(), (2, 2));
        assert!((r888.get(0, 0) + 0.2857142857142853).abs() < 1e-10);
        assert!((r888.get(0, 1) - 0.9583148474999088).abs() < 1e-10);
    }

    #[test]
    #[should_panic(expected = "escapes the table")]
    fn b3b_su3_escaping_pair_panics_not_truncates() {
        // 8⊗10 ∋ 35 (out of table): fusion_channels must fail loudly.
        let rule = su3();
        let _ = rule.fusion_channels(su3_id(1, 1), su3_id(3, 0));
    }

    // --- raw symbol pins (validate su3_table.bin bytes + row-major) -------
    #[test]
    fn b3b_su3_raw_symbols_match_sunrepresentations() {
        let rule = su3();
        let eight = su3_id(1, 1);
        // R(8,8,8): the genuine-OM 2×2, row-major. (SUNRep; symmetric ⇒ Hermitian.)
        let r888 = rule.r_symbol_generic(eight, eight, eight);
        assert_eq!(r888.shape(), (2, 2));
        let r_ref = [
            -0.2857142857142853,
            0.9583148474999088,
            0.9583148474999089,
            0.28571428571428553,
        ];
        for (k, &want) in r_ref.iter().enumerate() {
            assert!((r888.data()[k] - want).abs() < 1e-10, "R888[{k}]");
        }
        // R is Hermitian (SU(3) has no non-Hermitian R in the dim<=27 set):
        assert!((r888.data()[1] - r888.data()[2]).abs() < 1e-10);
        // F(8,8,8,8,8,8): rich 2×2×2×2, row-major [μ,ν,κ,λ].
        let f = rule.f_symbol_generic(eight, eight, eight, eight, eight, eight);
        assert_eq!(f.shape(), (2, 2, 2, 2));
        let f_ref = [
            0.857142857142856,
            0.0,
            0.0,
            -0.14285714285714263,
            0.0,
            -0.1428571428571425,
            -0.1428571428571427,
            -0.38332593899996337,
            0.0,
            -0.14285714285714285,
            -0.14285714285714246,
            -0.38332593899996315,
            -0.14285714285714274,
            -0.38332593899996353,
            -0.38332593899996326,
            0.6285714285714274,
        ];
        for (k, &want) in f_ref.iter().enumerate() {
            assert!((f.data()[k] - want).abs() < 1e-10, "F888888[{k}]");
        }
        // A simple N=1 R: R(3,3,6) = 1.
        let r336 = rule.r_symbol_generic(su3_id(1, 0), su3_id(1, 0), su3_id(2, 0));
        assert_eq!(r336.shape(), (1, 1));
        assert!((r336.data()[0] - 1.0).abs() < 1e-10);
        // B/A(8,8,8) derived from F (default trait method) = 2×2 identity (TK).
        let b = rule.b_symbol_generic(eight, eight, eight);
        let a = rule.a_symbol_generic(eight, eight, eight);
        for (m, mat) in [("B", &b), ("A", &a)] {
            assert_eq!(mat.shape(), (2, 2), "{m} shape");
            for i in 0..2 {
                for j in 0..2 {
                    let want = if i == j { 1.0 } else { 0.0 };
                    assert!((mat.get(i, j) - want).abs() < 1e-10, "{m}({i},{j})");
                }
            }
        }
    }

    // --- braid oracle: TK's OWN artin_braid on the genuine N(8,8,8)=2 OM tree.
    // Source tree [8,8,8,8]->vac, innerlines [8,8], vertices [μ,ν,1]; tenet
    // braid index 1 == TK i=2. Each row: (src_μ, src_ν, dst_i0, dst_i1, v0, v1,
    // v2, coeff). dst ids are dense (8=5, 27=16, 10̄=6, 10=7). inv=false (forward
    // R·F̄·R̄) and inv=true (inverse braid) are pinned separately — closing the
    // B1 inverse-braid residual. They coincide to 1e-13 here because every SU(3)
    // R in the table is Hermitian, which is itself the honest B1 finding.
    #[allow(clippy::type_complexity)]
    fn b3b_braid_oracle(inverse: bool) -> Vec<(usize, usize, usize, usize, usize, usize, usize, f64)> {
        if !inverse {
            vec![
                (1, 1, 0, 5, 1, 1, 1, -0.10101525445522083),
                (1, 1, 5, 5, 1, 1, 1, -0.06122448979591838),
                (1, 1, 5, 5, 2, 1, 1, -0.2738042421428303),
                (1, 1, 5, 5, 1, 2, 1, -0.27380424214283056),
                (1, 1, 5, 5, 2, 2, 1, -0.22448979591836649),
                (1, 1, 16, 5, 1, 1, 1, -0.524890659167823),
                (1, 1, 6, 5, 1, 1, 1, 0.3194382824999692),
                (1, 1, 7, 5, 1, 1, 1, -0.6388765649999384),
                (2, 1, 0, 5, 1, 1, 1, 0.3388154635894685),
                (2, 1, 5, 5, 1, 1, 1, -0.2738042421428302),
                (2, 1, 5, 5, 2, 1, 1, -0.22448979591836649),
                (2, 1, 5, 5, 1, 2, 1, 0.7755102040816295),
                (2, 1, 5, 5, 2, 2, 1, -0.10952169685713214),
                (2, 1, 16, 5, 1, 1, 1, 0.11736911946539241),
                (2, 1, 6, 5, 1, 1, 1, -0.07142857142857137),
                (2, 1, 7, 5, 1, 1, 1, -0.3571428571428563),
                (1, 2, 0, 5, 1, 1, 1, 0.3388154635894686),
                (1, 2, 5, 5, 1, 1, 1, -0.2738042421428305),
                (1, 2, 5, 5, 2, 1, 1, 0.7755102040816301),
                (1, 2, 5, 5, 1, 2, 1, -0.22448979591836676),
                (1, 2, 5, 5, 2, 2, 1, -0.10952169685713214),
                (1, 2, 16, 5, 1, 1, 1, 0.11736911946539252),
                (1, 2, 6, 5, 1, 1, 1, -0.07142857142857127),
                (1, 2, 7, 5, 1, 1, 1, -0.35714285714285643),
                (2, 2, 0, 5, 1, 1, 1, 0.101015254455221),
                (2, 2, 5, 5, 1, 1, 1, -0.22448979591836635),
                (2, 2, 5, 5, 2, 1, 1, -0.10952169685713217),
                (2, 2, 5, 5, 1, 2, 1, -0.109521696857132),
                (2, 2, 5, 5, 2, 2, 1, -0.28979591836734603),
                (2, 2, 16, 5, 1, 1, 1, -0.4549052379454467),
                (2, 2, 6, 5, 1, 1, 1, -0.7666518779999266),
                (2, 2, 7, 5, 1, 1, 1, 0.19166296949998135),
            ]
        } else {
            vec![
                (1, 1, 0, 5, 1, 1, 1, -0.10101525445522083),
                (1, 1, 5, 5, 1, 1, 1, -0.061224489795918435),
                (1, 1, 5, 5, 2, 1, 1, -0.2738042421428303),
                (1, 1, 5, 5, 1, 2, 1, -0.2738042421428306),
                (1, 1, 5, 5, 2, 2, 1, -0.22448979591836649),
                (1, 1, 16, 5, 1, 1, 1, -0.524890659167823),
                (1, 1, 6, 5, 1, 1, 1, 0.3194382824999692),
                (1, 1, 7, 5, 1, 1, 1, -0.6388765649999385),
                (2, 1, 0, 5, 1, 1, 1, 0.33881546358946857),
                (2, 1, 5, 5, 1, 1, 1, -0.27380424214283033),
                (2, 1, 5, 5, 2, 1, 1, -0.2244897959183665),
                (2, 1, 5, 5, 1, 2, 1, 0.7755102040816297),
                (2, 1, 5, 5, 2, 2, 1, -0.10952169685713212),
                (2, 1, 16, 5, 1, 1, 1, 0.11736911946539241),
                (2, 1, 6, 5, 1, 1, 1, -0.07142857142857137),
                (2, 1, 7, 5, 1, 1, 1, -0.3571428571428563),
                (1, 2, 0, 5, 1, 1, 1, 0.33881546358946857),
                (1, 2, 5, 5, 1, 1, 1, -0.27380424214283045),
                (1, 2, 5, 5, 2, 1, 1, 0.77551020408163),
                (1, 2, 5, 5, 1, 2, 1, -0.22448979591836674),
                (1, 2, 5, 5, 2, 2, 1, -0.10952169685713213),
                (1, 2, 16, 5, 1, 1, 1, 0.11736911946539247),
                (1, 2, 6, 5, 1, 1, 1, -0.07142857142857124),
                (1, 2, 7, 5, 1, 1, 1, -0.35714285714285643),
                (2, 2, 0, 5, 1, 1, 1, 0.101015254455221),
                (2, 2, 5, 5, 1, 1, 1, -0.22448979591836637),
                (2, 2, 5, 5, 2, 1, 1, -0.10952169685713217),
                (2, 2, 5, 5, 1, 2, 1, -0.10952169685713195),
                (2, 2, 5, 5, 2, 2, 1, -0.289795918367346),
                (2, 2, 16, 5, 1, 1, 1, -0.4549052379454466),
                (2, 2, 6, 5, 1, 1, 1, -0.7666518779999265),
                (2, 2, 7, 5, 1, 1, 1, 0.19166296949998135),
            ]
        }
    }

    #[test]
    fn b3b_su3_generic_braid_matches_tensorkit_om() {
        let rule = su3();
        let eight = su3_id(1, 1);
        let vac = SectorId::new(0);
        for &inverse in &[false, true] {
            let oracle = b3b_braid_oracle(inverse);
            for smu in 1..=2usize {
                for snu in 1..=2usize {
                    let tree = FusionTreeKey::new(
                        [eight, eight, eight, eight], vac,
                        [false, false, false, false],
                        [eight, eight],
                        [MultiplicityIndex::new(smu).expect("test multiplicity label is one-based"), MultiplicityIndex::new(snu).expect("test multiplicity label is one-based"), MultiplicityIndex::ONE],
                    );
                    let out = generic_artin_braid_at_with_inverse(&rule, &tree, 1, inverse).unwrap();
                    // The impl must reproduce every oracle row for this source and
                    // must emit no unexpected nonzero term.
                    let mut matched = vec![false; oracle.len()];
                    for (dst, coeff) in &out {
                        if coeff.abs() < 1e-10 {
                            continue;
                        }
                        let i0 = dst.innerlines()[0].id();
                        let i1 = dst.innerlines()[1].id();
                        let v = dst.vertices();
                        let idx = oracle.iter().position(|&(m, n, o0, o1, w0, w1, w2, val)| {
                            m == smu
                                && n == snu
                                && o0 == i0
                                && o1 == i1
                                && w0 == v[0].get()
                                && w1 == v[1].get()
                                && w2 == v[2].get()
                                && (val - coeff).abs() < 1e-10
                        });
                        match idx {
                            Some(k) => matched[k] = true,
                            None => panic!(
                                "inv={inverse} src=({smu},{snu}) spurious term \
                                 inner=[{i0},{i1}] vtx=[{},{},{}] = {coeff}",
                                v[0].get(),
                                v[1].get(),
                                v[2].get()
                            ),
                        }
                    }
                    for (k, &(m, n, ..)) in oracle.iter().enumerate() {
                        if m == smu && n == snu {
                            assert!(
                                matched[k],
                                "inv={inverse} src=({smu},{snu}) missing oracle row {k}"
                            );
                        }
                    }
                }
            }
        }
    }

    // --- bendright oracle: TK's OWN bendright on FusionTreeBlock{SUNIrrep{3}}.
    // codomain [8,8]->vac, domain []->vac: the single term has coeff
    // 1/√dim(8) · B(8,8,vac) = 1/√8 (TK). Validates the sqrt-dim factor on SU(3).
    #[test]
    fn b3b_su3_bendright_matches_tensorkit() {
        let rule = su3();
        let eight = su3_id(1, 1);
        let vac = SectorId::new(0);
        let cod = FusionTreeKey::new([eight, eight], vac, [false, false], [], [MultiplicityIndex::ONE]);
        let dom = FusionTreeKey::new([], vac, [], [], []);
        let pair = FusionTreePairKey::pair(cod, dom);
        let out = generic_bendright_tree_pair(&rule, &pair).unwrap();
        let total: f64 = out.iter().map(|(_, c)| c.abs()).sum();
        assert!(
            (total - 1.0 / 8.0_f64.sqrt()).abs() < 1e-10,
            "bendright (8,8)->() coeff {total}, want 1/√8"
        );
    }

    // --- Gate 2: in-repo self-consistency (no external oracle). ------------
    #[test]
    fn b3b_su3_braid_inverse_round_trip_is_identity() {
        // forward braid then inverse braid on each output must return the source
        // tree with coefficient 1 (TensorKit `artin_braid` inv-doc invariant).
        let rule = su3();
        let eight = su3_id(1, 1);
        let vac = SectorId::new(0);
        for smu in 1..=2usize {
            for snu in 1..=2usize {
                let tree = FusionTreeKey::new(
                    [eight, eight, eight, eight], vac,
                    [false, false, false, false],
                    [eight, eight],
                    [MultiplicityIndex::new(smu).expect("test multiplicity label is one-based"), MultiplicityIndex::new(snu).expect("test multiplicity label is one-based"), MultiplicityIndex::ONE],
                );
                let mut totals: std::collections::HashMap<FusionTreeKey, f64> =
                    std::collections::HashMap::new();
                for (mid, c1) in generic_artin_braid_at_with_inverse(&rule, &tree, 1, false).unwrap() {
                    for (out, c2) in
                        generic_artin_braid_at_with_inverse(&rule, &mid, 1, true).unwrap()
                    {
                        *totals.entry(out).or_insert(0.0) += c1 * c2;
                    }
                }
                for (key, coeff) in &totals {
                    let want = if key == &tree { 1.0 } else { 0.0 };
                    assert!(
                        (coeff - want).abs() < 1e-10,
                        "round-trip ({smu},{snu}): coeff {coeff} want {want}"
                    );
                }
                assert!((totals.get(&tree).copied().unwrap_or(0.0) - 1.0).abs() < 1e-10);
            }
        }
    }

    #[test]
    fn b3b_su3_bend_round_trip_is_identity() {
        // bendright then bendleft on the OM triple (8,8)->8 with domain [8]->8.
        let rule = su3();
        let eight = su3_id(1, 1);
        assert_eq!(rule.nsymbol(eight, eight, eight), rule.nsymbol(eight, rule.dual(eight), eight));
        for mu in 1..=2 {
            let cod = FusionTreeKey::new(
                [eight, eight], eight,
                [false, false],
                [],
                [MultiplicityIndex::new(mu).expect("test multiplicity label is one-based")],
            );
            let dom = FusionTreeKey::new([eight], eight, [false], [], []);
            let pair = FusionTreePairKey::pair(cod, dom);
            let mut totals: std::collections::HashMap<FusionTreePairKey, f64> =
                std::collections::HashMap::new();
            for (mid, c1) in generic_bendright_tree_pair(&rule, &pair).unwrap() {
                for (out, c2) in generic_bendleft_tree_pair(&rule, &mid).unwrap() {
                    *totals.entry(out).or_insert(0.0) += c1 * c2;
                }
            }
            for (key, coeff) in &totals {
                let want = if key == &pair { 1.0 } else { 0.0 };
                assert!((coeff - want).abs() < 1e-10, "bend round-trip μ={mu}: {coeff} want {want}");
            }
            assert!((totals.get(&pair).copied().unwrap_or(0.0) - 1.0).abs() < 1e-10);
        }
    }

    // ============ REFUTE b3b: enumeration completeness (attack A) ============

    // Dump every codomain tree that fusion_tree_keys_generic enumerates for a
    // given uncoupled list, across ALL coupled sectors. Uses domain == codomain
    // so each coupled group is a codomain×domain cross product; we recover pure
    // codomain trees by taking the unique codomain_tree of each key.
    fn refute_enum_codomain_trees(
        rule: &Su3FusionRule,
        uncoupled: &[SectorId],
    ) -> Vec<(usize, Vec<usize>, Vec<usize>)> {
        let leg = |s: SectorId| SectorLeg::new([(s, 1usize)], false);
        let legs: Vec<_> = uncoupled.iter().map(|&s| leg(s)).collect();
        let hom = FusionTreeHomSpace::new(
            FusionProductSpace::new(legs.clone()),
            FusionProductSpace::new(legs),
        );
        let keys = hom
            .fusion_tree_keys_generic(rule)
            .expect("helper is only used on fully in-table spaces");
        let mut set = std::collections::BTreeSet::new();
        for k in &keys {
            let t = k.codomain_tree();
            let c = t.coupled().id();
            let inner: Vec<usize> = t.innerlines().iter().map(|x| x.id()).collect();
            let vtx: Vec<usize> = t.vertices().iter().map(|x| x.get()).collect();
            set.insert((c, inner, vtx));
        }
        set.into_iter().collect()
    }

    #[test]
    fn refute_a_enum_rank2_88() {
        let rule = su3();
        let eight = su3_id(1, 1);
        let codomain = FusionProductSpace::new([
            SectorLeg::new([(eight, 1)], false),
            SectorLeg::new([(eight, 1)], false),
        ]);
        let domain = FusionProductSpace::new([SectorLeg::new(
            [
                (SectorId::new(0), 1),
                (SectorId::new(5), 1),
                (SectorId::new(6), 1),
                (SectorId::new(7), 1),
                (SectorId::new(16), 1),
            ],
            false,
        )]);
        let keys = FusionTreeHomSpace::new(codomain, domain)
            .fusion_tree_keys_generic(&rule)
            .unwrap();
        let trees = keys
            .iter()
            .map(|key| {
                (
                    key.coupled().id(),
                    Vec::new(),
                    key.codomain_vertices()
                        .iter()
                        .map(|vertex| vertex.get())
                        .collect::<Vec<_>>(),
                )
            })
            .collect::<Vec<_>>();
        // TensorKit in-table oracle for uncoupled (8,8): 6 trees.
        let oracle: Vec<(usize, Vec<usize>, Vec<usize>)> = vec![
            (0, vec![], vec![1]),
            (5, vec![], vec![1]),
            (5, vec![], vec![2]),
            (6, vec![], vec![1]),
            (7, vec![], vec![1]),
            (16, vec![], vec![1]),
        ];
        assert_eq!(trees, oracle, "rank-2 [8,8] enumeration mismatch vs TK");
    }

    #[test]
    fn su3_n888_pair_identity_is_exact_and_ordered() {
        // What: the two outer-multiplicity basis vectors of 8 x 8 -> 8 are
        // distinct pair identities in the production enumerator.
        let rule = su3();
        let eight = su3_id(1, 1);
        let hom = FusionTreeHomSpace::new(
            FusionProductSpace::new([
                SectorLeg::new([(eight, 1)], false),
                SectorLeg::new([(eight, 1)], false),
            ]),
            FusionProductSpace::new([SectorLeg::new([(eight, 1)], false)]),
        );
        let keys = hom
            .fusion_tree_keys_generic_for_coupled(&rule, eight)
            .unwrap();

        assert_eq!(keys.len(), 2);
        assert_eq!(keys[0].codomain_vertices(), &[MultiplicityIndex::ONE]);
        assert_eq!(
            keys[1].codomain_vertices(),
            &[MultiplicityIndex::new(2).unwrap()]
        );
        assert!(keys[0] < keys[1]);
        assert_eq!(
            keys.iter().cloned().collect::<std::collections::HashSet<_>>().len(),
            2
        );
        assert_eq!(
            keys.iter().cloned().collect::<std::collections::BTreeSet<_>>().len(),
            2
        );
    }

    #[test]
    fn refute_a_enum_rank3_333_fundamentals_ok() {
        // Rank-3 codomain that stays fully in-table at every fold: [3,3,3].
        // (3⊗3=3̄+6, 3̄⊗3=1+8, 6⊗3=8+10 — no escaping intermediate.) The
        // enumerator handles this correctly and matches TK exactly. This
        // isolates the rank-3 [8,8,8] failure below to escaping *intermediates*,
        // not a generic rank-3 defect.
        let rule = su3();
        let three = su3_id(1, 0);
        let trees = refute_enum_codomain_trees(&rule, &[three, three, three]);
        let oracle: Vec<(usize, Vec<usize>, Vec<usize>)> = vec![
            (0, vec![1], vec![1, 1]),
            (5, vec![1], vec![1, 1]),
            (5, vec![4], vec![1, 1]),
            (7, vec![4], vec![1, 1]),
        ];
        assert_eq!(trees, oracle, "rank-3 [3,3,3] enumeration mismatch vs TK");
    }

    // FLIPPED refute test (Option A fix): the [8,8,8] full-space enumeration
    // now returns Err (never panics, never truncates) because out-of-table
    // coupled candidates (35, 35̄, 64) exist — while every in-table sector is
    // still constructible per-sector with its exact full-SU(3) tree set (the
    // 24-tree TK pin below).
    #[test]
    fn b3b_fix_enum_rank3_888_full_space_errs_not_panics() {
        let rule = su3();
        let eight = su3_id(1, 1);
        let leg = SectorLeg::new([(eight, 1usize)], false);
        let hom = FusionTreeHomSpace::new(
            FusionProductSpace::new([leg.clone(), leg.clone(), leg.clone()]),
            FusionProductSpace::new([leg]),
        );
        let err = hom.fusion_tree_keys_generic(&rule).unwrap_err();
        let message = err.to_string();
        assert!(
            message.contains("cannot represent this space exactly"),
            "unexpected message: {message}"
        );
        // The escaping sectors are NAMED, never silently dropped: 35=(4,1),
        // 35̄=(1,4), 64=(3,3) are the out-of-table candidates of [8,8,8].
        for label in ["(1,4) dim 35", "(4,1) dim 35", "(3,3) dim 64"] {
            assert!(message.contains(label), "missing {label} in: {message}");
        }
    }

    // Per-sector construction of every in-table coupled sector of [8,8,8],
    // pinned tree-by-tree against TensorKit's own fusiontrees((8,8,8), c)
    // (TK 0.17.0 + SUNRepresentations 0.4.0; ids: 8=5, 10̄=6, 10=7, 27=16).
    // 24 trees total — the complete full-SU(3) set for these sectors, proving
    // "exact, not truncated" for the clean classification.
    #[test]
    fn b3b_fix_enum_rank3_888_per_sector_matches_tensorkit() {
        let rule = su3();
        let eight = su3_id(1, 1);
        // (coupled, [(inner, mu, nu)]) — verbatim TK fusiontrees dump.
        let oracle: [(usize, &[(usize, usize, usize)]); 5] = [
            (0, &[(5, 1, 1), (5, 2, 1)]),
            (
                5,
                &[
                    (0, 1, 1),
                    (5, 1, 1),
                    (5, 2, 1),
                    (5, 1, 2),
                    (5, 2, 2),
                    (16, 1, 1),
                    (6, 1, 1),
                    (7, 1, 1),
                ],
            ),
            (6, &[(5, 1, 1), (5, 2, 1), (16, 1, 1), (6, 1, 1)]),
            (7, &[(5, 1, 1), (5, 2, 1), (16, 1, 1), (7, 1, 1)]),
            (16, &[(5, 1, 1), (5, 2, 1), (16, 1, 1), (16, 1, 2), (6, 1, 1), (7, 1, 1)]),
        ];
        let mut total = 0usize;
        for (coupled, trees) in oracle {
            let c = SectorId::new(coupled);
            let cod_leg = SectorLeg::new([(eight, 1usize)], false);
            let dom_leg = SectorLeg::new([(c, 1usize)], false);
            let hom = FusionTreeHomSpace::new(
                FusionProductSpace::new([cod_leg.clone(), cod_leg.clone(), cod_leg]),
                FusionProductSpace::new([dom_leg]),
            );
            let keys = hom.fusion_tree_keys_generic_for_coupled(&rule, c).unwrap();
            let got: Vec<(usize, usize, usize)> = keys
                .iter()
                .map(|k| {
                    let t = k.codomain_tree();
                    assert_eq!(t.coupled(), c);
                    (
                        t.innerlines()[0].id(),
                        t.vertices()[0].get(),
                        t.vertices()[1].get(),
                    )
                })
                .collect();
            // What: the iterator retains TensorKit's categorical basis order,
            // not merely the same unordered set of tree identities.
            assert_eq!(
                got,
                trees.to_vec(),
                "coupled {coupled}: tree order mismatch vs TK"
            );
            assert_eq!(keys.len(), trees.len(), "coupled {coupled}: multiplicity lost");
            total += keys.len();
        }
        assert_eq!(total, 24, "[8,8,8] has 24 in-table trees (TK oracle)");
    }

    // FLIPPED refute test: rank-2 [27,8] — the pair itself escapes (∋ 35, 35̄,
    // 64), so full-space enumeration is Err; but each in-table coupled sector
    // is clean (escapes happen only at the final fold = coupled candidates) and
    // constructs with the exact TK multiplicities: N(27,8,c) = {8:1, 27:2,
    // 10̄:1, 10:1} (TK directproduct dump).
    #[test]
    fn b3b_fix_enum_rank2_278_per_sector_works_full_space_errs() {
        let rule = su3();
        let t27 = su3_id(2, 2);
        let eight = su3_id(1, 1);
        let cod = || {
            FusionProductSpace::new([
                SectorLeg::new([(t27, 1usize)], false),
                SectorLeg::new([(eight, 1usize)], false),
            ])
        };
        // Full space (domain [8]) errs — never panics, names the escapes.
        let dom_leg = SectorLeg::new([(eight, 1usize)], false);
        let hom = FusionTreeHomSpace::new(cod(), FusionProductSpace::new([dom_leg]));
        let message = hom.fusion_tree_keys_generic(&rule).unwrap_err().to_string();
        assert!(message.contains("cannot represent this space exactly"), "{message}");
        // Per-sector: exact multiplicities.
        for (coupled, n) in [(5usize, 1usize), (16, 2), (6, 1), (7, 1)] {
            let c = SectorId::new(coupled);
            let hom = FusionTreeHomSpace::new(
                cod(),
                FusionProductSpace::new([SectorLeg::new([(c, 1usize)], false)]),
            );
            let keys = hom.fusion_tree_keys_generic_for_coupled(&rule, c).unwrap();
            assert_eq!(keys.len(), n, "N(27,8,{coupled}) mismatch vs TK");
        }
    }

    // Rank-4 one-hop return path: [8,8,8,8] reaches frontier states (35, 35̄,
    // 64) at the intermediate step 3; the final fold consults the v2 one-hop
    // table N(f, 8, c).
    //  * coupled = vacuum: NOT a one-hop return (N(35,8,1)=0 — vac ∉ 35⊗8), so
    //    it stays clean and enumerates exactly 8 trees — pinned against TK's
    //    length(fusiontrees((8,8,8,8), vac)) = 8.
    //  * coupled = 27: IS a one-hop return (N(35,8,27)=1, TK), so its full-SU(3)
    //    tree set includes trees through the out-of-table inner line 35 → the
    //    table cannot enumerate it → Err ("out-of-table intermediates"), never
    //    a truncated block.
    #[test]
    fn b3b_fix_enum_rank4_8888_one_hop_clean_vs_tainted() {
        let rule = su3();
        let eight = su3_id(1, 1);
        let vac = SectorId::new(0);
        let t27 = su3_id(2, 2);
        let leg = || SectorLeg::new([(eight, 1usize)], false);
        let hom = FusionTreeHomSpace::new(
            FusionProductSpace::new([leg(), leg(), leg(), leg()]),
            FusionProductSpace::new(std::iter::empty::<SectorLeg>()),
        );
        // Full space errs (escaped candidates + tainted sectors exist).
        assert!(hom.fusion_tree_keys_generic(&rule).is_err());
        // Clean sector: exact TK tree count.
        let keys = hom.fusion_tree_keys_generic_for_coupled(&rule, vac).unwrap();
        assert_eq!(keys.len(), 8, "TK: length(fusiontrees((8,8,8,8), vac)) == 8");
        // Tainted sector: Err naming the cause, not a truncated enumeration.
        let err = hom
            .fusion_tree_keys_generic_for_coupled(&rule, t27)
            .unwrap_err()
            .to_string();
        assert!(err.contains("out-of-table intermediates"), "{err}");
    }

    // Attack B: independent END-TO-END spot-checks of su3_table.bin through the
    // Rust parser. Each value is fresh SUNRepresentations 0.4.0 output (via
    // /tmp/combenv), row-major flattened the SAME way gen.jl claims to write it.
    // Asymmetric shapes ((2,1,2,1), (1,2,1,1)) + distinct irreps (15, 27) make a
    // transposed axis order or wrong flatten detectable (would break these).
    #[test]
    fn refute_b_table_spot_checks() {
        let rule = su3();
        let sid = |p, q| su3_id(p, q);
        // F(8,8,8; d=27, e=8, f=8) shape (2,1,2,1): OM on μ,κ; involves 27.
        let f1 = rule.f_symbol_generic(
            sid(1, 1),
            sid(1, 1),
            sid(1, 1),
            sid(2, 2),
            sid(1, 1),
            sid(1, 1),
        );
        assert_eq!(f1.shape(), (2, 1, 2, 1));
        for (k, want) in [
            -0.1428571428571427,
            0.2555506259999756,
            0.2555506259999759,
            0.00952380952380997,
        ]
        .iter()
        .enumerate()
        {
            assert!((f1.data()[k] - want).abs() < 1e-10, "F(8,8,8,27,8,8)[{k}]");
        }
        // F(8,8,8; d=8, e=27, f=27) shape (1,1,1,1).
        let f2 = rule.f_symbol_generic(
            sid(1, 1),
            sid(1, 1),
            sid(1, 1),
            sid(1, 1),
            sid(2, 2),
            sid(2, 2),
        );
        assert_eq!(f2.shape(), (1, 1, 1, 1));
        assert!((f2.data()[0] - -0.1749999999999999).abs() < 1e-10);
        // F(8,8,8; d=10, e=10, f=27) shape (1,1,1,1).
        let f3 = rule.f_symbol_generic(
            sid(1, 1),
            sid(1, 1),
            sid(1, 1),
            sid(3, 0),
            sid(3, 0),
            sid(2, 2),
        );
        assert!((f3.data()[0] - 0.3872983346207417).abs() < 1e-10);
        // F(3̄,3,15; d=15, e=8, f=27) shape (1,2,1,1): ν axis length 2, 15+27.
        let f4 = rule.f_symbol_generic(
            sid(0, 1),
            sid(1, 0),
            sid(1, 2),
            sid(1, 2),
            sid(1, 1),
            sid(2, 2),
        );
        assert_eq!(f4.shape(), (1, 2, 1, 1));
        assert!((f4.data()[0] - -0.36018013511259883).abs() < 1e-10, "F4[0]");
        assert!((f4.data()[1] - 0.5198752449100368).abs() < 1e-10, "F4[1]");
        // R(8,8,10) = -1.
        let r = rule.r_symbol_generic(sid(1, 1), sid(1, 1), sid(3, 0));
        assert_eq!(r.shape(), (1, 1));
        assert!((r.data()[0] - -1.0).abs() < 1e-10);
    }

    // Equal hom spaces (by value, even rebuilt independently) intern to one id;
    // a single differing bit — a leg's dual flag, its sector set, its
    // degeneracy, or the codomain/domain rank — must intern to a distinct id.
    fn u1_leg(charge: i32, deg: usize, dual: bool) -> SectorLeg {
        SectorLeg::new([(u1(charge), deg)], dual)
    }

    #[test]
    fn hom_space_id_is_idempotent() {
        let build = || {
            FusionTreeHomSpace::new(
                FusionProductSpace::new([u1_leg(1, 2, false)]),
                FusionProductSpace::new([u1_leg(1, 2, false)]),
            )
        };
        assert_eq!(build().id(), build().id());
    }

    #[test]
    fn hom_space_clone_shares_content_but_not_unpublished_id_state() {
        // What: cloning reuses immutable HomSpace data while each handle keeps
        // its own lazy identity publication snapshot.
        let hom = FusionTreeHomSpace::new(
            FusionProductSpace::new([u1_leg(1, 2, false)]),
            FusionProductSpace::new([u1_leg(-1, 3, true)]),
        );
        let before_id = hom.clone();

        assert!(Arc::ptr_eq(&hom.content, &before_id.content));
        assert!(before_id.existing_id().is_none());

        let id = hom.id();
        assert!(before_id.existing_id().is_none());

        let after_id = hom.clone();
        assert!(Arc::ptr_eq(&hom.content, &after_id.content));
        assert_eq!(after_id.existing_id(), Some(id));
    }

    #[test]
    fn hom_space_id_separates_dual_flip() {
        // Rank-1 duality analog of the #119 regression: flipping one leg's dual
        // bit must not alias, on either the codomain or the domain side.
        let base = FusionTreeHomSpace::new(
            FusionProductSpace::new([u1_leg(1, 2, false)]),
            FusionProductSpace::new([u1_leg(1, 2, false)]),
        );
        let cod_dual = FusionTreeHomSpace::new(
            FusionProductSpace::new([u1_leg(1, 2, true)]),
            FusionProductSpace::new([u1_leg(1, 2, false)]),
        );
        let dom_dual = FusionTreeHomSpace::new(
            FusionProductSpace::new([u1_leg(1, 2, false)]),
            FusionProductSpace::new([u1_leg(1, 2, true)]),
        );
        assert_ne!(base.id(), cod_dual.id());
        assert_ne!(base.id(), dom_dual.id());
    }

    #[test]
    fn hom_space_id_separates_sectors_degeneracy_and_rank() {
        let base = FusionTreeHomSpace::new(
            FusionProductSpace::new([u1_leg(1, 2, false)]),
            FusionProductSpace::new([u1_leg(1, 2, false)]),
        )
        .id();
        let other_charge = FusionTreeHomSpace::new(
            FusionProductSpace::new([u1_leg(3, 2, false)]),
            FusionProductSpace::new([u1_leg(1, 2, false)]),
        )
        .id();
        let other_deg = FusionTreeHomSpace::new(
            FusionProductSpace::new([u1_leg(1, 5, false)]),
            FusionProductSpace::new([u1_leg(1, 2, false)]),
        )
        .id();
        let higher_rank = FusionTreeHomSpace::new(
            FusionProductSpace::new([u1_leg(1, 2, false), u1_leg(0, 2, false)]),
            FusionProductSpace::new([u1_leg(1, 2, false)]),
        )
        .id();
        assert_ne!(base, other_charge);
        assert_ne!(base, other_deg);
        assert_ne!(base, higher_rank);
    }

    #[test]
    fn hom_space_id_remains_semantic_after_intern_eviction() {
        // What: floods the shared hom-space intern table past its cap, which
        // races `concurrent_equal_hom_spaces_share_semantic_identity` (asserts
        // ptr_eq on entries of that same table) if both run concurrently.
        let _guard = test_support::CACHE_TEST_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let build = || {
            FusionTreeHomSpace::new(
                FusionProductSpace::new([u1_leg(17, 2, false)]),
                FusionProductSpace::new([u1_leg(17, 3, true)]),
            )
        };
        let before = build().id();
        for charge in 10_000..10_000 + HOM_SPACE_INTERN_CAP as i32 + 1 {
            let _ = FusionTreeHomSpace::new(
                FusionProductSpace::new([u1_leg(charge, 1, false)]),
                FusionProductSpace::new([u1_leg(charge, 1, false)]),
            )
            .id();
        }
        let after = build().id();
        assert!(!Arc::ptr_eq(&before.key, &after.key));
        assert_eq!(before, after);
        let hash = |id: &HomSpaceId| {
            let mut state = rustc_hash::FxHasher::default();
            id.hash(&mut state);
            std::hash::Hasher::finish(&state)
        };
        assert_eq!(hash(&before), hash(&after));
        assert_eq!(
            hom_space_intern_table().read().unwrap().entries.len(),
            HOM_SPACE_INTERN_CAP
        );
    }

    #[test]
    fn eager_hom_space_derivation_does_not_touch_lazy_id_interner() {
        let rule = U1FusionRule;
        let leg = SectorLeg::new([(u1(-1), 2), (u1(2), 1)], false);
        let hom = FusionTreeHomSpace::new(
            FusionProductSpace::new([leg.clone(), leg.clone()]),
            FusionProductSpace::new([leg.clone(), leg]),
        );

        let selected = hom.select(&rule, &[1, 0], &[3, 2]).unwrap();
        let permuted = hom.permute(&rule, &[1, 0], &[3, 2]).unwrap();
        let composed = FusionTreeHomSpace::compose(&rule, &hom, &hom).unwrap();
        let contracted = FusionTreeHomSpace::tensorcontract_homspace(
            &rule,
            &hom,
            &hom,
            &[2, 3],
            &[0, 1],
            &[0, 1, 2, 3],
            2,
        )
        .unwrap();
        assert!(selected.existing_id().is_none());
        assert!(permuted.existing_id().is_none());
        assert!(composed.existing_id().is_none());
        assert!(contracted.existing_id().is_none());

        let selected_id = selected.id();
        assert_eq!(selected_id, permuted.id());
        assert_eq!(composed.id(), contracted.id());
    }

    #[test]
    fn concurrent_eager_hom_space_derivation_does_not_touch_lazy_id_interner() {
        let rule = U1FusionRule;
        let leg = SectorLeg::new([(u1(-1), 2), (u1(2), 1)], false);
        let hom = FusionTreeHomSpace::new(
            FusionProductSpace::new([leg.clone(), leg.clone()]),
            FusionProductSpace::new([leg.clone(), leg]),
        );
        std::thread::scope(|scope| {
            for _ in 0..8 {
                scope.spawn(|| {
                    for _ in 0..100 {
                        let permuted = hom.permute(&rule, &[1, 0], &[3, 2]).unwrap();
                        let contracted = FusionTreeHomSpace::tensorcontract_homspace(
                            &rule,
                            &hom,
                            &hom,
                            &[2, 3],
                            &[0, 1],
                            &[0, 1, 2, 3],
                            2,
                        )
                        .unwrap();
                        assert!(permuted.existing_id().is_none());
                        assert!(contracted.existing_id().is_none());
                    }
                });
            }
        });
    }

    #[test]
    fn resetting_lazy_hom_space_interner_preserves_semantic_identity() {
        let _guard = test_support::CACHE_TEST_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        reset_hom_space_intern_table();
        let build = || {
            FusionTreeHomSpace::new(
                FusionProductSpace::new([u1_leg(23, 2, false)]),
                FusionProductSpace::new([u1_leg(-23, 3, true)]),
            )
        };
        let before = build().id();
        reset_hom_space_intern_table();
        let after = build().id();
        assert_eq!(before, after);
        assert!(!Arc::ptr_eq(&before.key, &after.key));
    }

    #[test]
    fn concurrent_equal_hom_spaces_share_semantic_identity() {
        // What: asserts ptr_eq across concurrently-built identical hom spaces
        // in the shared intern table; a concurrent flood from
        // `hom_space_id_remains_semantic_after_intern_eviction` could evict an
        // entry mid-build and hand a later thread a fresh (non-aliased) Arc.
        let _guard = test_support::CACHE_TEST_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let ids = std::thread::scope(|scope| {
            (0..8)
                .map(|_| {
                    scope.spawn(|| {
                        FusionTreeHomSpace::new(
                            FusionProductSpace::new([u1_leg(41, 7, false)]),
                            FusionProductSpace::new([u1_leg(41, 9, true)]),
                        )
                        .id()
                    })
                })
                .map(|thread| thread.join().unwrap())
                .collect::<Vec<_>>()
        });
        assert!(ids.windows(2).all(|pair| pair[0] == pair[1]));
        assert!(ids
            .windows(2)
            .all(|pair| Arc::ptr_eq(&pair[0].key, &pair[1].key)));
    }

    // Canary (#153) against silent growth of the hottest recoupling-plan key.
    // A smaller representation is allowed; growth requires re-checking the
    // compact Hash/Eq/Ord and allocation contracts.
    #[test]
    fn fusion_tree_key_size_has_not_silently_grown() {
        assert!(std::mem::size_of::<FusionTreeKey>() <= 264);
        assert_eq!(
            std::mem::size_of::<MultiplicityIndex>(),
            std::mem::size_of::<usize>()
        );
    }

    // Canary (#231) against `CoreError` regrowing past the clippy
    // `result_large_err` threshold: `{Missing,Duplicate}BlockKey` box their
    // `BlockKey` payload precisely to keep every `Result<_, CoreError>` return
    // pointer-cheap on the hot paths that propagate it with `?`.
    #[test]
    fn core_error_size_has_not_silently_grown() {
        assert!(std::mem::size_of::<CoreError>() <= 128);
    }

    #[test]
    fn coupled_sector_regions_describe_canonical_matrix_spans() {
        // What: canonical coupled storage compiles to exact sector ranges and tree extents.
        let rule = Z2FusionRule;
        let leg = || SectorLeg::new([(z2_even(), 2), (z2_odd(), 2)], false);
        let homspace = FusionTreeHomSpace::new(
            FusionProductSpace::new([leg(), leg()]),
            FusionProductSpace::new([leg(), leg()]),
        );
        let keys = homspace.fusion_tree_keys(&rule);
        let space = FusionTensorMapSpace::from_degeneracy_shapes_coupled(
            TensorMapSpace::<2, 2>::from_dims([4, 4], [4, 4]).unwrap(),
            homspace,
            &rule,
            vec![vec![2; 4]; keys.len()],
        )
        .unwrap();

        let structure = space.subblock_structure();
        let cloned_before_query = structure.as_ref().clone();
        assert!(!structure.coupled_region_cache_is_initialized());
        let regions = cloned_before_query
            .coupled_sector_regions(2)
            .unwrap()
            .unwrap();
        assert!(structure.coupled_region_cache_is_initialized());
        let original_regions = structure
            .coupled_sector_regions(2)
            .unwrap()
            .unwrap();
        assert!(Arc::ptr_eq(&regions, &original_regions));

        assert_eq!(regions.len(), 2);
        assert_eq!(regions[0].range().start, 0);
        assert_eq!(regions[0].range().len(), regions[0].rows() * regions[0].cols());
        assert_eq!(regions[1].range().start, regions[0].range().end);
        assert_eq!(regions[1].range().end, space.required_len().unwrap());
        assert!(regions.iter().all(|region| {
            !region.row_trees().is_empty()
                && !region.col_trees().is_empty()
                && region
                    .row_trees()
                    .iter()
                    .all(|tree| tree.extent().unwrap() > 0)
        }));
    }

    #[test]
    fn coupled_sector_regions_reject_noncanonical_and_incomplete_grids() {
        // What: independently packed subblocks and a missing tree pair cannot claim direct spans.
        let rule = Z2FusionRule;
        let leg = || SectorLeg::new([(z2_even(), 1), (z2_odd(), 1)], false);
        let homspace = FusionTreeHomSpace::new(
            FusionProductSpace::new([leg(), leg()]),
            FusionProductSpace::new([leg(), leg()]),
        );
        let keys = homspace.fusion_tree_keys(&rule);
        let mut offset = 0usize;
        let independently_packed = keys
            .iter()
            .map(|key| {
                let block = BlockSpec::column_major_with_key(
                    BlockKey::FusionTree(key.clone()),
                    vec![1; 4],
                    offset,
                )
                .unwrap();
                offset += 1;
                block
            })
            .collect();
        let independently_packed = BlockStructure::from_blocks(independently_packed).unwrap();
        assert_eq!(
            independently_packed.coupled_sector_regions(2).unwrap(),
            None
        );

        let coupled = BlockStructure::coupled_sector_matrix_with_keys(
            &rule,
            2,
            4,
            keys.iter()
                .cloned()
                .map(|key| (key, vec![1; 4]))
                .collect(),
        )
        .unwrap();
        let incomplete = BlockStructure::from_blocks(
            (0..coupled.block_count() - 1)
                .map(|index| {
                    let block = coupled.block(index).unwrap();
                    BlockSpec::with_key(
                        block.key().clone(),
                        block.shape().to_vec(),
                        block.strides().to_vec(),
                        block.offset(),
                    )
                    .unwrap()
                })
                .collect(),
        )
        .unwrap();
        assert_eq!(incomplete.coupled_sector_regions(2).unwrap(), None);
    }

    fn unbound_expert_space<const NOUT: usize, const NIN: usize>(
        blocks: Vec<BlockSpec>,
    ) -> Result<FusionTensorMapSpace<NOUT, NIN>, CoreError> {
        // Storage-only fixtures use opaque ordinals deliberately: categorical
        // key and shape admission is covered independently by try_bind_rule.
        let dense = TensorMapSpace::<NOUT, NIN>::from_dims([1; NOUT], [1; NIN]).unwrap();
        let homspace = FusionTreeHomSpace::from_sector_ids(
            (0..NOUT).map(|_| (0, 1)),
            (0..NIN).map(|_| (0, 1)),
        );
        FusionTensorMapSpace::new_unbound(
            dense,
            homspace,
            BlockStructure::from_blocks_with_rank(NOUT + NIN, blocks).unwrap(),
        )
    }

    #[test]
    fn expert_fusion_space_rejects_self_overlapping_storage() {
        // What: an owning symmetric space cannot assign two logical elements of
        // one block to the same physical element.
        for (block, offset) in [
            (
                BlockSpec::with_key(BlockKey::ordinal(0), vec![2, 2], vec![1, 1], 0)
                    .unwrap(),
                1,
            ),
            (
                BlockSpec::with_key(BlockKey::ordinal(0), vec![2, 1], vec![0, 1], 0)
                    .unwrap(),
                0,
            ),
        ] {
            assert_eq!(
                unbound_expert_space::<1, 1>(vec![block]),
                Err(CoreError::OverlappingBlockStorage {
                    first_block: 0,
                    second_block: 0,
                    offset,
                })
            );
        }
    }

    #[test]
    fn expert_fusion_space_rejects_cross_block_storage_aliases() {
        // What: distinct logical symmetric blocks cannot own the same physical
        // destination element, and diagnostics preserve caller block order.
        let blocks = vec![
            BlockSpec::with_key(BlockKey::ordinal(0), vec![2], vec![2], 0).unwrap(),
            BlockSpec::with_key(BlockKey::ordinal(1), vec![2], vec![1], 1).unwrap(),
        ];
        assert_eq!(
            unbound_expert_space::<1, 0>(blocks),
            Err(CoreError::OverlappingBlockStorage {
                first_block: 0,
                second_block: 1,
                offset: 2,
            })
        );
    }

    #[test]
    fn expert_fusion_space_accepts_exact_non_overlapping_strided_storage() {
        // What: expert admission retains arbitrary legal layouts, including
        // layouts that a conservative sorted-span proof cannot establish.
        let rank_two_cases = [
            vec![BlockSpec::with_key(
                BlockKey::ordinal(0),
                vec![3, 2],
                vec![2, 3],
                0,
            )
            .unwrap()],
            vec![BlockSpec::with_key(
                BlockKey::ordinal(0),
                vec![2, 3],
                vec![3, 1],
                0,
            )
            .unwrap()],
            vec![
                BlockSpec::with_key(BlockKey::ordinal(0), vec![1, 2], vec![0, 1], 0)
                    .unwrap(),
            ],
        ];
        reset_exact_storage_fallback_count();
        for blocks in rank_two_cases {
            unbound_expert_space::<1, 1>(blocks).unwrap();
        }
        assert!(exact_storage_fallback_count() > 0);

        let rank_one_cases = [
            vec![
                BlockSpec::with_key(BlockKey::ordinal(0), vec![4], vec![2], 0).unwrap(),
                BlockSpec::with_key(BlockKey::ordinal(1), vec![4], vec![2], 1).unwrap(),
            ],
            vec![
                BlockSpec::with_key(BlockKey::ordinal(0), vec![2], vec![1], 4).unwrap(),
                BlockSpec::with_key(BlockKey::ordinal(1), vec![2], vec![1], 0).unwrap(),
            ],
            vec![
                BlockSpec::with_key(BlockKey::ordinal(0), vec![0], vec![0], 0).unwrap(),
                BlockSpec::with_key(BlockKey::ordinal(1), vec![1], vec![1], 0).unwrap(),
            ],
        ];
        for blocks in rank_one_cases {
            unbound_expert_space::<1, 0>(blocks).unwrap();
        }

        unbound_expert_space::<0, 0>(vec![
            BlockSpec::with_key(BlockKey::ordinal(0), vec![], vec![], 0).unwrap(),
        ])
        .unwrap();
        unbound_expert_space::<1, 0>(vec![
            BlockSpec::with_key(BlockKey::ordinal(0), vec![0], vec![1], 0).unwrap(),
        ])
        .unwrap();
    }

    #[test]
    fn general_block_structure_retains_aliasing_read_view_contract() {
        // What: storage ownership admission belongs to FusionTensorMapSpace;
        // general block metadata remains usable for intentionally aliased views.
        let structure = BlockStructure::from_blocks(vec![
            BlockSpec::with_key(BlockKey::ordinal(0), vec![2], vec![1], 0).unwrap(),
            BlockSpec::with_key(BlockKey::ordinal(1), vec![2], vec![1], 0).unwrap(),
        ])
        .unwrap();
        assert_eq!(structure.required_len().unwrap(), 2);
    }

    #[test]
    fn fusion_space_adjoint_view_preserves_custom_storage_footprint() {
        // What: adjoint swaps categorical sides and block axes while retaining
        // the already-admitted physical footprint; applying it twice is exact.
        let dense = TensorMapSpace::<1, 1>::from_dims([2], [3]).unwrap();
        let homspace = FusionTreeHomSpace::from_sector_ids([(0, 2)], [(0, 3)]);
        let structure = BlockStructure::from_blocks(vec![
            BlockSpec::with_key(BlockKey::ordinal(0), vec![2, 3], vec![1, 4], 2).unwrap(),
        ])
        .unwrap();
        let source = FusionTensorMapSpace::new_unbound(dense, homspace, structure).unwrap();

        reset_exact_storage_fallback_count();
        let adjoint = source.adjoint_view().unwrap();
        assert_eq!(exact_storage_fallback_count(), 0);
        assert_eq!(adjoint.dense_space().codomain().dims(), &[3]);
        assert_eq!(adjoint.dense_space().domain().dims(), &[2]);
        assert_eq!(adjoint.homspace().codomain(), source.homspace().domain());
        assert_eq!(adjoint.homspace().domain(), source.homspace().codomain());
        let block = adjoint.subblock_structure().block(0).unwrap();
        assert_eq!(block.shape(), &[3, 2]);
        assert_eq!(block.strides(), &[4, 1]);
        assert_eq!(block.offset(), 2);
        assert_eq!(adjoint.adjoint_view().unwrap(), source);
    }

    fn assert_expert_storage_admission_for_rule<R>(rule: &R, sector: SectorId)
    where
        R: MultiplicityFreeFusionRule,
    {
        let homspace = FusionTreeHomSpace::from_sectors([(sector, 2)], [(sector, 3)]);
        let key = homspace.fusion_tree_keys(rule)[0].clone();
        let structure = BlockStructure::from_blocks(vec![
            BlockSpec::with_key(BlockKey::FusionTree(key), vec![2, 3], vec![1, 4], 2).unwrap(),
        ])
        .unwrap();
        FusionTensorMapSpace::new_unbound(
            TensorMapSpace::<1, 1>::from_dims([2], [3]).unwrap(),
            homspace,
            structure,
        )
        .unwrap()
        .try_bind_rule(rule)
        .unwrap();
    }

    #[test]
    fn expert_storage_admission_is_symmetry_independent() {
        // What: the same exact custom-layout admission and categorical binding
        // succeeds for U(1), SU(2), fZ2, and their nested product.
        assert_expert_storage_admission_for_rule(&U1FusionRule, u1(0));
        assert_expert_storage_admission_for_rule(&SU2FusionRule, su2(0));
        assert_expert_storage_admission_for_rule(&FermionParityFusionRule, z2_even());

        type Fz2U1 = ProductFusionRule<FermionParityFusionRule, U1FusionRule>;
        type Triple = ProductFusionRule<Fz2U1, SU2FusionRule>;
        let pair = Fz2U1::new(FermionParityFusionRule, U1FusionRule);
        let pair_vacuum = pair.encode_sector(z2_even(), u1(0));
        let triple = Triple::new(pair, SU2FusionRule);
        let vacuum = triple.encode_sector(pair_vacuum, su2(0));
        assert_expert_storage_admission_for_rule(&triple, vacuum);
    }

    #[test]
    fn canonical_and_factor_bridge_storage_skip_exact_enumeration() {
        // What: hom-space-generated coupled storage and its typed factor-style
        // shared bridge are admitted by structural proof, not element scans.
        let rule = SU2FusionRule;
        let homspace = FusionTreeHomSpace::from_sectors([(su2(1), 2)], [(su2(1), 3)]);
        reset_exact_storage_fallback_count();
        let canonical = FusionTensorMapSpace::from_degeneracy_shapes(
            TensorMapSpace::<1, 1>::from_dims([2], [3]).unwrap(),
            homspace.clone(),
            &rule,
            [vec![2, 3]],
        )
        .unwrap();
        assert_eq!(exact_storage_fallback_count(), 0);

        reset_exact_storage_fallback_count();
        FusionTensorMapSpace::from_shared_subblock_structure(
            TensorMapSpace::<1, 1>::from_dims([2], [3]).unwrap(),
            homspace,
            Arc::clone(canonical.subblock_structure()),
        )
        .unwrap()
        .try_bind_rule(&rule)
        .unwrap();
        assert_eq!(exact_storage_fallback_count(), 0);
    }

    fn local_block_structure_intern_key(index: usize) -> BlockStructureInternKey {
        BlockStructureInternKey {
            rank: 1,
            blocks: Arc::from([BlockStructureContentBlock {
                key: BlockKey::ordinal(index),
                shape: smallvec![1],
                strides: smallvec![1],
                offset: 0,
            }]),
        }
    }

    fn local_block_structure_intern(
        table: &mut BlockStructureInternTable,
        key: BlockStructureInternKey,
        charged_key_bytes: usize,
    ) -> Arc<BlockStructureContent> {
        let rank = key.rank;
        let blocks = Arc::clone(&key.blocks);
        table.intern_with(key, |_| charged_key_bytes, || {
            Arc::new(BlockStructureContent {
                id: BLOCK_STRUCTURE_CONTENT_ID.fetch_add(1, Ordering::Relaxed),
                rank,
                blocks,
                coupled_region_cache: OnceLock::new(),
            })
        })
    }

    #[test]
    fn block_structure_intern_charge_counts_only_spilled_smallvec_storage() {
        // What: inline SmallVec storage adds no heap charge, while spilled
        // storage contributes its full heap capacity in item bytes.
        let inline: SmallVec<[u64; 2]> = smallvec::smallvec![1_u64, 2];
        let spilled: SmallVec<[u64; 2]> = smallvec::smallvec![1_u64, 2, 3];
        assert!(!inline.spilled());
        assert!(spilled.spilled());
        assert_eq!(spilled_smallvec_heap_bytes(&inline), 0);
        assert_eq!(
            spilled_smallvec_heap_bytes(&spilled),
            spilled.capacity() * std::mem::size_of::<u64>()
        );
    }

    #[test]
    fn block_structure_intern_entry_pressure_evicts_oldest() {
        // What: entry pressure removes the oldest admitted key, and a read hit
        // does not promote it in the FIFO order.
        let key0 = local_block_structure_intern_key(0);
        let key1 = local_block_structure_intern_key(1);
        let key2 = local_block_structure_intern_key(2);
        let charge = charged_block_structure_intern_key_bytes(&key0);
        assert_eq!(charged_block_structure_intern_key_bytes(&key1), charge);
        assert_eq!(charged_block_structure_intern_key_bytes(&key2), charge);
        let mut table = BlockStructureInternTable::new(
            2,
            charge.saturating_mul(3),
            charge,
        );

        let _content0 = local_block_structure_intern(&mut table, key0.clone(), charge);
        let _content1 = local_block_structure_intern(&mut table, key1.clone(), charge);
        assert!(table.lookup(&key0).is_some());
        let _content2 = local_block_structure_intern(&mut table, key2.clone(), charge);

        assert!(table.lookup(&key0).is_none());
        assert!(table.lookup(&key1).is_some());
        assert!(table.lookup(&key2).is_some());
        let info = table.info();
        assert_eq!(info.entries(), 2);
        assert_eq!(info.pressure_evictions(), 1);
    }

    #[test]
    fn block_structure_intern_byte_pressure_subtracts_exact_charge() {
        // What: byte pressure subtracts the evicted entry's unequal stored
        // charge exactly before admitting the incoming key.
        let key0 = local_block_structure_intern_key(10);
        let key1 = local_block_structure_intern_key(11);
        let key2 = local_block_structure_intern_key(12);
        let base_charge = charged_block_structure_intern_key_bytes(&key0);
        let charges = [base_charge, base_charge + 1, base_charge + 2];
        let budget = charges[1].saturating_add(charges[2]);
        let mut table = BlockStructureInternTable::new(3, budget, charges[2]);

        let _content0 = local_block_structure_intern(&mut table, key0.clone(), charges[0]);
        let _content1 = local_block_structure_intern(&mut table, key1.clone(), charges[1]);
        assert_eq!(
            table.info().charged_key_bytes(),
            charges[0].saturating_add(charges[1])
        );
        assert_eq!(table.info().pressure_evictions(), 0);
        assert!(table.lookup(&key0).is_some());

        let _content2 = local_block_structure_intern(&mut table, key2.clone(), charges[2]);
        assert!(table.lookup(&key0).is_none());
        assert!(table.lookup(&key1).is_some());
        assert!(table.lookup(&key2).is_some());
        assert_eq!(table.info().charged_key_bytes(), budget);
        assert_eq!(table.info().pressure_evictions(), 1);
    }

    #[test]
    fn block_structure_intern_bypasses_oversized_and_saturated_charges() {
        // What: oversized and saturated charges return complete content but
        // never consume an entry or charged-byte budget.
        let oversized_key = local_block_structure_intern_key(20);
        let saturated_key = local_block_structure_intern_key(21);
        let charge = charged_block_structure_intern_key_bytes(&oversized_key);
        let mut oversized_table = BlockStructureInternTable::new(2, charge, charge - 1);

        let oversized =
            local_block_structure_intern(&mut oversized_table, oversized_key.clone(), charge);
        assert_eq!(oversized.rank(), oversized_key.rank);
        assert_eq!(oversized.blocks(), oversized_key.blocks.as_ref());
        assert!(oversized_table.lookup(&oversized_key).is_none());
        assert_eq!(oversized_table.info().oversized_admission_bypasses(), 1);

        let mut saturated_table = BlockStructureInternTable::new(2, usize::MAX, usize::MAX);
        let saturated = local_block_structure_intern(
            &mut saturated_table,
            saturated_key.clone(),
            usize::MAX,
        );
        assert_eq!(saturated.rank(), saturated_key.rank);
        assert_eq!(saturated.blocks(), saturated_key.blocks.as_ref());
        assert!(saturated_table.lookup(&saturated_key).is_none());

        let info = saturated_table.info();
        assert_eq!(info.entries(), 0);
        assert_eq!(info.charged_key_bytes(), 0);
        assert_eq!(info.oversized_admission_bypasses(), 1);
    }

    #[test]
    fn block_structure_intern_dead_replacement_preserves_fifo_accounting() {
        // What: replacing a dead Weak changes only its content epoch; entry
        // count, charge, counters, and oldest-first eviction order stay fixed.
        let key0 = local_block_structure_intern_key(30);
        let key1 = local_block_structure_intern_key(31);
        let key2 = local_block_structure_intern_key(32);
        let charge = charged_block_structure_intern_key_bytes(&key0);
        let mut table = BlockStructureInternTable::new(
            2,
            charge.saturating_mul(2),
            charge,
        );

        let content0 = local_block_structure_intern(&mut table, key0.clone(), charge);
        let id0 = content0.id();
        let _content1 = local_block_structure_intern(&mut table, key1.clone(), charge);
        let before = table.info();
        drop(content0);
        assert!(table.lookup(&key0).is_none());

        let replacement = local_block_structure_intern(&mut table, key0.clone(), charge);
        assert!(replacement.id() > id0);
        assert_eq!(table.info(), before);

        let _content2 = local_block_structure_intern(&mut table, key2.clone(), charge);
        assert!(table.lookup(&key0).is_none());
        assert!(table.lookup(&key1).is_some());
        assert!(table.lookup(&key2).is_some());
        assert_eq!(table.info().pressure_evictions(), 1);
    }

    #[test]
    fn block_structure_intern_clear_resets_resources_and_counters() {
        // What: clear releases every admitted key and resets byte, eviction,
        // and bypass accounting while preserving configured limits.
        let key0 = local_block_structure_intern_key(40);
        let key1 = local_block_structure_intern_key(41);
        let key2 = local_block_structure_intern_key(42);
        let charge = charged_block_structure_intern_key_bytes(&key0);
        let mut table = BlockStructureInternTable::new(1, charge, charge);
        let _content0 = local_block_structure_intern(&mut table, key0, charge);
        let _content1 = local_block_structure_intern(&mut table, key1, charge);
        let _content2 = local_block_structure_intern(&mut table, key2, usize::MAX);
        assert_eq!(table.info().pressure_evictions(), 1);
        assert_eq!(table.info().oversized_admission_bypasses(), 1);

        table.clear();

        let info = table.info();
        assert_eq!(info.entries(), 0);
        assert_eq!(info.entry_capacity(), 1);
        assert_eq!(info.charged_key_bytes(), 0);
        assert_eq!(info.byte_budget(), charge);
        assert_eq!(info.max_admitted_entry_bytes(), charge);
        assert_eq!(info.pressure_evictions(), 0);
        assert_eq!(info.oversized_admission_bypasses(), 0);
    }

    fn coupled_z2_matrix_structure() -> BlockStructure {
        let rule = Z2FusionRule;
        let leg = || SectorLeg::new([(z2_even(), 2), (z2_odd(), 3)], false);
        let homspace = FusionTreeHomSpace::new(
            FusionProductSpace::new([leg()]),
            FusionProductSpace::new([leg()]),
        );
        let blocks = homspace
            .fusion_tree_keys(&rule)
            .iter()
            .cloned()
            .map(|key| (key, vec![2, 3]))
            .collect();
        BlockStructure::coupled_sector_matrix_with_keys(&rule, 1, 2, blocks).unwrap()
    }

    #[test]
    fn live_then_dead_content_uses_one_weak_canonicalization_epoch() {
        // What: equal live structures and their lazy region result share one
        // content epoch, while rebuilding after every owner dies preserves the
        // complete structure and region semantics under a fresh monotonic id.
        let _guard = test_support::CACHE_TEST_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        reset_core_intern_tables();

        let first = coupled_z2_matrix_structure();
        let second = coupled_z2_matrix_structure();
        let first_content = first.content_key();
        let second_content = second.content_key();
        assert!(Arc::ptr_eq(&first_content, &second_content));
        let id_before = first.content_id();

        let first_regions = first.coupled_sector_regions(1).unwrap().unwrap();
        let second_regions = second.coupled_sector_regions(1).unwrap().unwrap();
        assert!(Arc::ptr_eq(&first_regions, &second_regions));

        let expected_sector = first.sector_structure().clone();
        let expected_degeneracy = first.degeneracy_structure().clone();
        let expected_len = first.required_len().unwrap();
        let expected_regions = first_regions.as_ref().to_vec();
        let weak_content = Arc::downgrade(&first_content);

        drop(first_regions);
        drop(second_regions);
        drop(first_content);
        drop(second_content);
        drop(first);
        drop(second);
        assert!(weak_content.upgrade().is_none());

        let rebuilt = coupled_z2_matrix_structure();
        assert!(rebuilt.content_id() > id_before);
        assert_eq!(rebuilt.sector_structure(), &expected_sector);
        assert_eq!(rebuilt.degeneracy_structure(), &expected_degeneracy);
        assert_eq!(rebuilt.required_len().unwrap(), expected_len);
        assert_eq!(
            rebuilt
                .coupled_sector_regions(1)
                .unwrap()
                .unwrap()
                .as_ref(),
            expected_regions
        );
    }

    #[test]
    fn concurrent_equal_live_content_canonicalizes_once() {
        // What: concurrent equal construction shares one content Arc and id
        // while every returned structure remains live.
        let _guard = test_support::CACHE_TEST_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        reset_core_intern_tables();
        let barrier = std::sync::Barrier::new(8);
        let structures = std::thread::scope(|scope| {
            let barrier = &barrier;
            let threads = (0..8)
                .map(|_| {
                    scope.spawn(move || {
                        barrier.wait();
                        BlockStructure::trivial(&[17, 19]).unwrap()
                    })
                })
                .collect::<Vec<_>>();
            threads
                .into_iter()
                .map(|thread| thread.join().unwrap())
                .collect::<Vec<_>>()
        });
        let content = structures[0].content_key();
        let id = content.id();
        for structure in &structures[1..] {
            let candidate = structure.content_key();
            assert!(Arc::ptr_eq(&content, &candidate));
            assert_eq!(candidate.id(), id);
        }
    }

    #[test]
    fn reset_core_intern_tables_clears_without_reusing_ids() {
        // What: reset preserves a surviving structure and its published region
        // while equal content rebuilt afterward receives a fresh monotonic id.
        let _guard = test_support::CACHE_TEST_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let before = coupled_z2_matrix_structure();
        let id_before = before.content_id();
        let regions_before = before.coupled_sector_regions(1).unwrap().unwrap();

        reset_core_intern_tables();

        let surviving_regions = before.coupled_sector_regions(1).unwrap().unwrap();
        assert!(Arc::ptr_eq(&regions_before, &surviving_regions));
        assert_eq!(before.content_id(), id_before);

        let after = coupled_z2_matrix_structure();
        let id_after = after.content_id();
        assert!(
            id_after > id_before,
            "reset must not reuse content ids, got before={id_before} after={id_after}"
        );
    }

    #[test]
    fn u1_zigzag_roundtrips_native_and_simulated_32_bit_extremes() {
        // What: every i32 charge, including both asymmetric endpoints, has
        // the historical u32 zigzag ID without target-width arithmetic.
        let cases = [
            (i32::MIN, u32::MAX),
            (-1, 1),
            (0, 0),
            (1, 2),
            (i32::MAX, u32::MAX - 1),
        ];
        for (charge, encoded) in cases {
            assert_eq!(u1_charge_to_zigzag_u32(charge), encoded);
            assert_eq!(u1_charge_from_zigzag_u32(encoded), charge);
            let sector = U1Irrep::new(charge).sector_id();
            assert_eq!(sector.id(), encoded as usize);
            assert_eq!(U1Irrep::from_sector_id(sector), Some(U1Irrep::new(charge)));
        }
    }

    #[test]
    fn checked_u1_reports_nonclosure_and_preserves_valid_boundaries() {
        // What: finite-i32 nonclosure is typed, while boundary sums that
        // remain representable are identical to the expert infallible path.
        let rule = U1FusionRule;
        assert_eq!(
            rule.try_dual_sector(u1(i32::MIN)),
            Err(FusionAlgebraError::U1DualOverflow { charge: i32::MIN })
        );
        for (left, right) in [(i32::MAX, 1), (i32::MIN, -1)] {
            assert_eq!(
                rule.try_fusion_channels(u1(left), u1(right)),
                Err(FusionAlgebraError::U1FusionOverflow { left, right })
            );
        }
        for (left, right, expected) in [
            (i32::MAX, 0, i32::MAX),
            (i32::MIN, 0, i32::MIN),
            (i32::MAX, i32::MIN, -1),
        ] {
            let checked = rule.try_fusion_channels(u1(left), u1(right)).unwrap();
            assert_eq!(checked.as_slice(), &[u1(expected)]);
            assert_eq!(checked, rule.fusion_channels(u1(left), u1(right)));
            assert_eq!(
                rule.try_nsymbol(u1(left), u1(right), u1(expected)),
                Ok(rule.nsymbol(u1(left), u1(right), u1(expected)))
            );
        }
        assert_eq!(
            rule.try_dual_sector(u1(i32::MAX)).unwrap(),
            rule.dual(u1(i32::MAX))
        );
    }

    #[test]
    fn checked_su2_distinguishes_invalid_inputs_from_unrepresentable_fusion() {
        // What: valid SU2 inputs whose output exceeds the supported algebra
        // report closure failure, while the exact boundary matches the hot path.
        let rule = SU2FusionRule;
        let boundary = rule
            .try_fusion_channels(su2(127), su2(127))
            .unwrap();
        assert_eq!(boundary, rule.fusion_channels(su2(127), su2(127)));
        assert_eq!(
            rule.try_fusion_channels(su2(128), su2(127)),
            Err(FusionAlgebraError::FusionNotRepresentable {
                left: su2(128),
                right: su2(127),
            })
        );
        assert_eq!(
            rule.try_fusion_channels(SectorId::new(255), su2(0)),
            Err(FusionAlgebraError::InvalidSector {
                sector: SectorId::new(255),
            })
        );
    }

    #[test]
    fn lowered_su2_success_stays_below_the_sector_id_boundary() {
        // What: validated lowered irreps compute channel bounds and
        // multiplicity without revalidating or re-encoding SectorIds.
        reset_su2_id_boundary_observations();
        let rule = SU2FusionRule;
        let left = SU2Irrep::from_twice_spin(2);
        let right = SU2Irrep::from_twice_spin(1);
        let mut channels = Vec::new();
        rule.try_for_each_lowered_channel(left, right, &mut |channel| {
            channels.push(channel.twice_spin());
            Ok(())
        })
        .unwrap();
        assert_eq!(channels, vec![1, 3]);
        assert_eq!(
            rule.try_lowered_nsymbol(left, right, SU2Irrep::from_twice_spin(1)),
            Ok(1)
        );
        assert_eq!(su2_id_boundary_observations(), (0, 0));

        assert!(rule
            .try_fusion_channels(left.sector_id(), right.sector_id())
            .is_ok());
        assert_eq!(su2_id_boundary_observations(), (2, 0));
    }

    #[test]
    fn checked_fibonacci_matches_valid_operations_and_rejects_unknown_sectors() {
        // What: Fibonacci's checked companion preserves every valid operation
        // and rejects IDs outside the two-sector algebra with the exact input.
        let rule = FibonacciFusionRule;
        let vacuum = SectorId::new(0);
        let tau = SectorId::new(1);
        assert_eq!(rule.try_dual_sector(tau), Ok(rule.dual(tau)));
        assert_eq!(
            rule.try_fusion_channels(tau, tau),
            Ok(rule.fusion_channels(tau, tau))
        );
        for coupled in [vacuum, tau] {
            assert_eq!(
                rule.try_nsymbol(tau, tau, coupled),
                Ok(rule.nsymbol(tau, tau, coupled))
            );
        }
        let invalid = SectorId::new(2);
        assert_eq!(
            rule.try_dual_sector(invalid),
            Err(FusionAlgebraError::InvalidSector { sector: invalid })
        );
        assert_eq!(
            rule.try_fusion_channels(tau, invalid),
            Err(FusionAlgebraError::InvalidSector { sector: invalid })
        );
        assert_eq!(
            rule.try_nsymbol(tau, tau, invalid),
            Err(FusionAlgebraError::InvalidSector { sector: invalid })
        );
    }

    #[test]
    fn checked_recursive_product_preserves_channel_order_and_valid_multiplicity() {
        // What: recursive checked products retain the established
        // right-outer/left-inner channel order and valid nsymbol products.
        type Pair = ProductFusionRule<FibonacciFusionRule, FibonacciFusionRule>;
        type Triple = ProductFusionRule<Pair, FibonacciFusionRule>;

        let pair = Pair::new(FibonacciFusionRule, FibonacciFusionRule);
        let pair_tau = pair.try_encode_sector(SectorId::new(1), SectorId::new(1)).unwrap();
        let triple = Triple::new(pair, FibonacciFusionRule);
        let input = triple
            .try_encode_sector(pair_tau, SectorId::new(1))
            .unwrap();
        let checked = triple.try_fusion_channels(input, input).unwrap();
        let infallible = triple.fusion_channels(input, input);
        assert_eq!(checked, infallible);

        let mut expected = SectorVec::new();
        for right in [SectorId::new(0), SectorId::new(1)] {
            for pair_right in [SectorId::new(0), SectorId::new(1)] {
                for pair_left in [SectorId::new(0), SectorId::new(1)] {
                    let pair_channel = triple
                        .left_rule()
                        .try_encode_sector(pair_left, pair_right)
                        .unwrap();
                    expected.push(triple.try_encode_sector(pair_channel, right).unwrap());
                }
            }
        }
        assert_eq!(checked, expected);
        for coupled in checked {
            assert_eq!(
                triple.try_nsymbol(input, input, coupled),
                Ok(triple.nsymbol(input, input, coupled))
            );
        }
    }

    #[derive(Clone, Copy, Debug)]
    struct CheckedMultiplicityRule<const N: usize>;

    impl<const N: usize> FusionRule for CheckedMultiplicityRule<N> {
        fn rule_identity(&self) -> RuleIdentity {
            RuleIdentity::of_type::<Self>()
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

        fn fusion_channels(&self, _left: SectorId, _right: SectorId) -> SectorVec {
            core::iter::once(SectorId::new(0)).collect()
        }

        fn nsymbol(&self, _left: SectorId, _right: SectorId, _coupled: SectorId) -> usize {
            N
        }
    }

    impl<const N: usize> CheckedFusionAlgebra for CheckedMultiplicityRule<N> {
        fn try_dual_sector(&self, sector: SectorId) -> Result<SectorId, FusionAlgebraError> {
            if sector == SectorId::new(0) {
                Ok(sector)
            } else {
                Err(FusionAlgebraError::InvalidSector { sector })
            }
        }

        fn try_fusion_channels(
            &self,
            left: SectorId,
            right: SectorId,
        ) -> Result<SectorVec, FusionAlgebraError> {
            self.try_dual_sector(left)?;
            self.try_dual_sector(right)?;
            Ok(self.fusion_channels(left, right))
        }

        fn try_nsymbol(
            &self,
            left: SectorId,
            right: SectorId,
            coupled: SectorId,
        ) -> Result<usize, FusionAlgebraError> {
            self.try_dual_sector(left)?;
            self.try_dual_sector(right)?;
            self.try_dual_sector(coupled)?;
            Ok(N)
        }
    }

    #[test]
    fn checked_product_reports_multiplicity_overflow_without_panicking() {
        // What: product multiplicities that exceed usize return the exact
        // structured overflow instead of wrapping or entering the hot path.
        type Rule =
            ProductFusionRule<CheckedMultiplicityRule<{ usize::MAX }>, CheckedMultiplicityRule<2>>;
        let rule = Rule::new(CheckedMultiplicityRule, CheckedMultiplicityRule);
        let sector = rule
            .try_encode_sector(SectorId::new(0), SectorId::new(0))
            .unwrap();
        assert_eq!(
            rule.try_nsymbol(sector, sector, sector),
            Err(FusionAlgebraError::MultiplicityOverflow {
                left: sector,
                right: sector,
                coupled: sector,
            })
        );
    }

    #[cfg(target_pointer_width = "64")]
    #[test]
    fn checked_products_preserve_child_u1_errors_and_distinguish_codec_errors() {
        // What: recursive products retain the exact U1 closure cause, while
        // malformed packed IDs remain a distinct codec failure.
        type Fz2U1Codec = PackedProductCodec<Fz2SectorLayout, U1SectorLayout>;
        type Fz2U1Layout = ProductSectorLayout<Fz2SectorLayout, U1SectorLayout>;
        type Fz2U1Rule =
            ProductFusionRule<FermionParityFusionRule, U1FusionRule, Fz2U1Codec>;
        type TripleCodec = PackedProductCodec<Fz2U1Layout, Su2SectorLayout>;
        type TripleLayout = ProductSectorLayout<Fz2U1Layout, Su2SectorLayout>;
        type TripleRule = ProductFusionRule<Fz2U1Rule, SU2FusionRule, TripleCodec>;

        let pair = Fz2U1Rule::new(FermionParityFusionRule, U1FusionRule);
        let pair_min = pair.try_encode_sector(z2_odd(), u1(i32::MIN)).unwrap();
        assert_eq!(
            pair.try_dual_sector(pair_min),
            Err(FusionAlgebraError::U1DualOverflow { charge: i32::MIN })
        );
        let pair_max = pair.try_encode_sector(z2_even(), u1(i32::MAX)).unwrap();
        let pair_one = pair.try_encode_sector(z2_odd(), u1(1)).unwrap();
        assert_eq!(
            pair.try_fusion_channels(pair_max, pair_one),
            Err(FusionAlgebraError::U1FusionOverflow {
                left: i32::MAX,
                right: 1,
            })
        );

        let triple = TripleRule::new(pair, SU2FusionRule);
        let triple_min = triple.try_encode_sector(pair_min, su2(1)).unwrap();
        assert_eq!(
            triple.try_dual_sector(triple_min),
            Err(FusionAlgebraError::U1DualOverflow { charge: i32::MIN })
        );
        let invalid = SectorId::new(1usize << TripleLayout::BITS);
        assert!(matches!(
            triple.try_dual_sector(invalid),
            Err(FusionAlgebraError::ProductCodec(
                ProductSectorCodecError::InvalidHighBits { .. }
            ))
        ));
    }

    #[test]
    fn checked_fusion_algebra_is_object_safe_and_matches_closed_builtins() {
        // What: callers can use checked algebra through one provider object,
        // and closed built-ins retain their infallible results exactly.
        let checked: &dyn CheckedFusionAlgebra = &U1FusionRule;
        assert_eq!(checked.try_dual_sector(u1(7)), Ok(u1(-7)));
        for rule in [
            &Z2FusionRule as &dyn CheckedFusionAlgebra,
            &FermionParityFusionRule,
            &SU2FusionRule,
        ] {
            let left = rule.vacuum();
            let right = rule.vacuum();
            assert_eq!(rule.try_dual_sector(left), Ok(rule.dual(left)));
            assert_eq!(
                rule.try_fusion_channels(left, right),
                Ok(rule.fusion_channels(left, right))
            );
            assert_eq!(
                rule.try_nsymbol(left, right, rule.vacuum()),
                Ok(rule.nsymbol(left, right, rule.vacuum()))
            );
        }
    }

    #[test]
    fn lowered_u1_errors_preserve_the_checked_algebra_cause() {
        // What: direct lowered U1 dual and fusion calls own the exact checked
        // algebra cause instead of retaining only a static classification.
        let dual = U1FusionRule
            .try_lowered_dual(U1Irrep::new(i32::MIN))
            .unwrap_err();
        assert_eq!(
            dual.into_fusion_algebra().unwrap(),
            FusionAlgebraError::U1DualOverflow { charge: i32::MIN }
        );
        let mut emit = |_sector| Ok(());
        let fusion = U1FusionRule
            .try_for_each_lowered_channel(
                U1Irrep::new(i32::MAX),
                U1Irrep::new(1),
                &mut emit,
            )
            .unwrap_err();
        assert_eq!(
            fusion.into_fusion_algebra().unwrap(),
            FusionAlgebraError::U1FusionOverflow {
                left: i32::MAX,
                right: 1,
            }
        );
    }

    #[test]
    fn u1_trivial_a_b_symbols_accept_min_charge_valid_triples() {
        // What: trivial U1 rigidity symbols remain exactly one for valid
        // MIN-containing triples without requiring an unrepresentable dual.
        let rule = U1FusionRule;
        assert_eq!(
            rule.a_symbol_scalar(u1(i32::MIN), u1(0), u1(i32::MIN)),
            1.0
        );
        assert_eq!(
            rule.b_symbol_scalar(u1(0), u1(i32::MIN), u1(i32::MIN)),
            1.0
        );
    }

    #[test]
    fn checked_sector_leg_dual_is_transactional_and_preserves_exact_causes() {
        // What: a checked dual either returns the complete dual leg or the exact
        // algebra failure without changing the source leg.
        let source = SectorLeg::new([(u1(i32::MIN), 2), (u1(1), 3)], false);
        let before = source.clone();
        assert_eq!(
            source.try_dual(&U1FusionRule),
            Err(FusionAlgebraError::U1DualOverflow { charge: i32::MIN })
        );
        assert_eq!(source, before);

        #[cfg(target_pointer_width = "64")]
        {
            type Rule = ProductFusionRule<U1FusionRule, Z2FusionRule, TensorKitProductCodec>;
            let rule = Rule::new(U1FusionRule, Z2FusionRule);
            let min = TensorKitProductCodec::encode(u1(i32::MIN), z2_odd());
            let product = SectorLeg::new([(min, 1)], false);
            assert_eq!(
                product.try_dual(&rule),
                Err(FusionAlgebraError::U1DualOverflow { charge: i32::MIN })
            );
        }
    }

    #[test]
    fn checked_select_and_permute_report_orientation_failure_without_extra_duals() {
        // What: moving a legal MIN codomain leg across the HomSpace boundary
        // reports its algebra error, while same-side selection and identity
        // permutation perform no dual operation.
        let min_leg = SectorLeg::new([(u1(i32::MIN), 2)], false);
        let hom = FusionTreeHomSpace::new(
            FusionProductSpace::new([min_leg]),
            FusionProductSpace::new([]),
        );
        let rule = DualCountingRule::new(U1FusionRule);

        let same_side = hom.try_select_checked(&rule, &[0], &[]).unwrap();
        assert_eq!(same_side, hom);
        assert_eq!(rule.dual_calls(), 0);
        let identity = hom.try_permute_checked(&rule, &[0], &[]).unwrap();
        assert_eq!(identity, hom);
        assert_eq!(rule.dual_calls(), 0);

        assert_eq!(
            hom.try_select_checked(&rule, &[], &[0]),
            Err(CheckedFusionSpaceError::FusionAlgebra(Box::new(
                FusionAlgebraError::U1DualOverflow { charge: i32::MIN }
            )))
        );
        assert_eq!(rule.dual_calls(), 1);

        #[cfg(target_pointer_width = "64")]
        {
            type Rule = ProductFusionRule<U1FusionRule, Z2FusionRule, TensorKitProductCodec>;
            let product_rule = Rule::new(U1FusionRule, Z2FusionRule);
            let product_min = TensorKitProductCodec::encode(u1(i32::MIN), z2_odd());
            let product_hom = FusionTreeHomSpace::new(
                FusionProductSpace::new([SectorLeg::new([(product_min, 1)], false)]),
                FusionProductSpace::new([]),
            );
            assert_eq!(
                product_hom.try_permute_checked(&product_rule, &[], &[0]),
                Err(CheckedFusionSpaceError::FusionAlgebra(Box::new(
                    FusionAlgebraError::U1DualOverflow { charge: i32::MIN }
                )))
            );
        }
    }

    #[test]
    fn checked_orientation_validates_axes_before_dual_arithmetic() {
        // What: malformed axis requests retain Core validation precedence even
        // when the first legal orientation mapping would overflow.
        let min_leg = SectorLeg::new([(u1(i32::MIN), 1)], false);
        let hom = FusionTreeHomSpace::new(
            FusionProductSpace::new([min_leg]),
            FusionProductSpace::new([]),
        );
        let rule = DualCountingRule::new(U1FusionRule);
        assert_eq!(
            hom.try_select_checked(&rule, &[], &[1]),
            Err(CheckedFusionSpaceError::Core(Box::new(
                CoreError::InvalidPermutation {
                    permutation: vec![1],
                    rank: 1,
                }
            )))
        );
        assert_eq!(rule.dual_calls(), 0);
        assert_eq!(
            hom.try_permute_checked(&rule, &[], &[]),
            Err(CheckedFusionSpaceError::Core(Box::new(
                CoreError::InvalidPermutation {
                    permutation: vec![],
                    rank: 1,
                }
            )))
        );
        assert_eq!(rule.dual_calls(), 0);

        let scalar = FusionTreeHomSpace::new(
            FusionProductSpace::new([]),
            FusionProductSpace::new([]),
        );
        assert_eq!(
            FusionTreeHomSpace::try_tensorcontract_homspace_checked(
                &rule,
                &hom,
                &scalar,
                &[],
                &[],
                &[1],
                0,
            ),
            Err(CheckedFusionSpaceError::Core(Box::new(
                CoreError::InvalidPermutation {
                    permutation: vec![1],
                    rank: 1,
                }
            )))
        );
        assert_eq!(rule.dual_calls(), 0);
    }

    #[test]
    fn checked_tensorcontract_separates_pair_validation_and_output_orientation_failures() {
        // What: structural pair mismatches retain Core precedence without
        // touching checked dual arithmetic, while a valid open leg moved to the
        // output domain reports the exact algebra failure.
        let min_leg = || SectorLeg::new([(u1(i32::MIN), 1)], false);
        let scalar = || {
            FusionTreeHomSpace::new(
                FusionProductSpace::new([]),
                FusionProductSpace::new([]),
            )
        };
        let lhs =
            FusionTreeHomSpace::new(FusionProductSpace::new([min_leg()]), FusionProductSpace::new([]));
        let rhs =
            FusionTreeHomSpace::new(FusionProductSpace::new([min_leg()]), FusionProductSpace::new([]));
        let rule = DualCountingRule::new(U1FusionRule);
        assert_eq!(
            FusionTreeHomSpace::try_tensorcontract_homspace_checked(
                &rule,
                &lhs,
                &rhs,
                &[0],
                &[0],
                &[],
                0,
            ),
            Err(CheckedFusionSpaceError::Core(Box::new(
                CoreError::MalformedFusionTree {
                    message: "contracted fusion leg duality flags do not match",
                }
            )))
        );
        assert_eq!(rule.dual_calls(), 0);

        let rhs_count_mismatch = FusionTreeHomSpace::new(
            FusionProductSpace::new([]),
            FusionProductSpace::new([SectorLeg::new(
                [(u1(i32::MIN), 1), (u1(0), 1)],
                false,
            )]),
        );
        assert_eq!(
            FusionTreeHomSpace::try_tensorcontract_homspace_checked(
                &rule,
                &lhs,
                &rhs_count_mismatch,
                &[0],
                &[0],
                &[],
                0,
            ),
            Err(CheckedFusionSpaceError::Core(Box::new(
                CoreError::DimensionMismatch {
                    expected: 1,
                    actual: 2,
                }
            )))
        );
        assert_eq!(rule.dual_calls(), 0);

        assert_eq!(
            FusionTreeHomSpace::try_tensorcontract_homspace_checked(
                &rule,
                &lhs,
                &scalar(),
                &[],
                &[],
                &[0],
                0,
            ),
            Err(CheckedFusionSpaceError::FusionAlgebra(Box::new(
                FusionAlgebraError::U1DualOverflow { charge: i32::MIN }
            )))
        );
        assert_eq!(rule.dual_calls(), 1);
    }

    fn assert_checked_contract_matches_infallible<R>(rule: &R, sector: SectorId)
    where
        R: CheckedFusionAlgebra,
    {
        let leg = SectorLeg::new([(sector, 2)], false);
        let lhs = FusionTreeHomSpace::new(
            FusionProductSpace::new([leg]),
            FusionProductSpace::new([]),
        );
        let rhs = FusionTreeHomSpace::new(
            FusionProductSpace::new([]),
            FusionProductSpace::new([]),
        );
        let expected =
            FusionTreeHomSpace::tensorcontract_homspace(rule, &lhs, &rhs, &[], &[], &[0], 0)
                .unwrap();
        let actual = FusionTreeHomSpace::try_tensorcontract_homspace_checked(
            rule,
            &lhs,
            &rhs,
            &[],
            &[],
            &[0],
            0,
        )
        .unwrap();
        assert_eq!(actual, expected);
    }

    #[test]
    fn checked_tensorcontract_matches_closed_builtin_orientation() {
        // What: checked orientation is semantically identical to the established
        // infallible path for every closed built-in algebra and a product rule.
        assert_checked_contract_matches_infallible(&Z2FusionRule, z2_odd());
        assert_checked_contract_matches_infallible(&FermionParityFusionRule, z2_odd());
        assert_checked_contract_matches_infallible(&U1FusionRule, u1(7));
        assert_checked_contract_matches_infallible(&SU2FusionRule, su2(3));

        #[cfg(target_pointer_width = "64")]
        {
            type Rule = ProductFusionRule<U1FusionRule, Z2FusionRule, TensorKitProductCodec>;
            let rule = Rule::new(U1FusionRule, Z2FusionRule);
            let sector = TensorKitProductCodec::encode(u1(4), z2_odd());
            assert_checked_contract_matches_infallible(&rule, sector);
        }
    }

    #[test]
    fn checked_fusion_space_error_exposes_its_typed_source() {
        // What: callers can inspect either the structural or algebraic source
        // through the standard error chain without parsing display text.
        let core = CheckedFusionSpaceError::from(CoreError::DimensionMismatch {
            expected: 1,
            actual: 2,
        });
        assert!(std::error::Error::source(&core)
            .is_some_and(|source| source.downcast_ref::<CoreError>().is_some()));
        let algebra = CheckedFusionSpaceError::from(FusionAlgebraError::U1DualOverflow {
            charge: i32::MIN,
        });
        assert!(std::error::Error::source(&algebra)
            .is_some_and(|source| source.downcast_ref::<FusionAlgebraError>().is_some()));
    }

    #[test]
    fn checked_homspace_derivation_keeps_semantic_identity_lazy() {
        // What: successful checked metadata derivation does not publish or
        // resolve a process-local HomSpace identity until a caller asks for it.
        let rule = U1FusionRule;
        let leg = SectorLeg::new([(u1(-2), 1), (u1(1), 2)], false);
        let hom = FusionTreeHomSpace::new(
            FusionProductSpace::new([leg.clone(), leg]),
            FusionProductSpace::new([]),
        );
        crate::reset_hom_space_intern_calls();
        let selected = hom.try_select_checked(&rule, &[1, 0], &[]).unwrap();
        let permuted = hom.try_permute_checked(&rule, &[1, 0], &[]).unwrap();
        let scalar = FusionTreeHomSpace::new(
            FusionProductSpace::new([]),
            FusionProductSpace::new([]),
        );
        let contracted = FusionTreeHomSpace::try_tensorcontract_homspace_checked(
            &rule,
            &hom,
            &scalar,
            &[],
            &[],
            &[1, 0],
            2,
        )
        .unwrap();
        assert_eq!(crate::hom_space_intern_calls(), 0);
        assert_eq!(selected, permuted);
        assert_eq!(contracted, permuted);
    }
}
