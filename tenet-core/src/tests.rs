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
            _source: &FusionTreeBlockKey,
            _destination: &FusionTreeBlockKey,
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

    #[derive(Debug, Default)]
    struct SplitOnlyCountingRule {
        f_calls: std::sync::atomic::AtomicUsize,
        r_calls: std::sync::atomic::AtomicUsize,
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
            let tree = FusionTreeKey::from_sector_ids([1, 1], Some(coupled), [false, false], [], [1]);
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
            FusionTreeKey::from_sector_ids([1, 1, 1], Some(1), [false, false, false], [1], [1, 1]);

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

    #[test]
    fn fibonacci_braid_then_inverse_braid_is_identity() {
        // Self-consistency (a): braiding a crossing and then undoing it
        // (reflected levels select the inverse-artin branch) must return the
        // exact original tree with total coefficient 1 — this only holds
        // because R^{ττ}_* is a genuine unit-modulus phase.
        let rule = FibonacciFusionRule;
        for coupled in [0usize, 1usize] {
            let tree =
                FusionTreeKey::from_sector_ids([1, 1], Some(coupled), [false, false], [], [1]);

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
            FusionTreeKey::from_sector_ids([1, 1, 1], Some(1), [false, false, false], [1], [1, 1]);
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
            FusionTreeKey::from_sector_ids([1, 1, 1], Some(1), [false, false, false], [1], [1, 1]);

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

    fn tree_pair_group_fixture(
        codomain: &[usize],
        domain: &[usize],
        coupled: usize,
        codomain_dual: &[bool],
        domain_dual: &[bool],
    ) -> FusionTreeBlockKey {
        FusionTreeBlockKey::pair_from_sector_ids(
            codomain.iter().copied(),
            domain.iter().copied(),
            Some(coupled),
            codomain_dual.iter().copied(),
            domain_dual.iter().copied(),
            [],
            [],
            [],
            [],
        )
    }

    fn assert_mixed_tree_pair_block_group_is_rejected(keys: &[FusionTreeBlockKey]) {
        let expected = CoreError::MalformedFusionTree {
            message: TREE_PAIR_BLOCK_GROUP_ERROR,
        };
        let rule = IdentitySymbolPanicRule;

        // What: identity and non-identity braid entry paths reject before
        // returning a shared coefficient matrix or evaluating rigid symbols.
        assert_eq!(
            multiplicity_free_braid_tree_pair_block(
                &rule,
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
                &rule,
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
            multiplicity_free_permute_tree_pair_block(&rule, keys, &[0, 1], &[2]).unwrap_err(),
            expected
        );
        assert_eq!(
            multiplicity_free_permute_tree_pair_block(&rule, keys, &[1, 0], &[2]).unwrap_err(),
            expected
        );

        // What: planar transpose validates before either its identity return or
        // cyclic repartition path can consume symbols.
        assert_eq!(
            multiplicity_free_transpose_tree_pair_block(&rule, keys, &[0, 1], &[2]).unwrap_err(),
            expected
        );
        assert_eq!(
            multiplicity_free_transpose_tree_pair_block(&rule, keys, &[1, 2], &[0]).unwrap_err(),
            expected
        );
    }

    #[test]
    fn tree_pair_block_apis_reject_mixed_fusion_tree_groups_before_symbols() {
        let base = tree_pair_group_fixture(&[1, 2], &[3], 7, &[false, false], &[false]);
        let mixed = [
            tree_pair_group_fixture(&[1, 2, 4], &[3], 7, &[false, false, false], &[false]),
            tree_pair_group_fixture(&[1, 2], &[3, 4], 7, &[false, false], &[false, false]),
            tree_pair_group_fixture(&[1, 4], &[3], 7, &[false, false], &[false]),
            tree_pair_group_fixture(&[1, 2], &[4], 7, &[false, false], &[false]),
            tree_pair_group_fixture(&[1, 2], &[3], 7, &[false, true], &[false]),
            tree_pair_group_fixture(&[1, 2], &[3], 7, &[false, false], &[true]),
        ];

        for other in mixed {
            let keys = [base.clone(), other];
            let snapshot = keys.clone();
            assert_mixed_tree_pair_block_group_is_rejected(&keys);
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
        let coupled = rule.encode_sector(z2_even(), u1(5)).id();
        let keys = [
            tree_pair_group_fixture(&[sector_a, sector_a], &[sector_a], coupled, &[false; 2], &[false]),
            tree_pair_group_fixture(&[sector_a, sector_b], &[sector_a], coupled, &[false; 2], &[false]),
        ];

        // What: a changed component of an interned product-sector label is a
        // different shared basis group, even when ranks and duality match.
        assert_mixed_tree_pair_block_group_is_rejected(&keys);
    }

    #[test]
    fn tree_pair_block_empty_and_valid_group_identity_policies_are_stable() {
        let rule = IdentitySymbolPanicRule;
        let empty: &[FusionTreeBlockKey] = &[];
        // What: an empty block remains a valid empty coefficient transform.
        assert_eq!(
            multiplicity_free_braid_tree_pair_block(&rule, empty, &[], &[], &[], &[]).unwrap(),
            Vec::<Vec<(FusionTreeBlockKey, f64)>>::new()
        );
        assert_eq!(
            multiplicity_free_permute_tree_pair_block(&rule, empty, &[], &[]).unwrap(),
            Vec::<Vec<(FusionTreeBlockKey, f64)>>::new()
        );
        assert_eq!(
            multiplicity_free_transpose_tree_pair_block(&rule, empty, &[], &[]).unwrap(),
            Vec::<Vec<(FusionTreeBlockKey, f64)>>::new()
        );

        let keys = vec![
            tree_pair_group_fixture(&[1, 2], &[3], 7, &[false, false], &[false]),
            tree_pair_group_fixture(&[1, 2], &[3], 8, &[false, false], &[false]),
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
                &rule,
                &keys,
                &[0, 1],
                &[2],
                &[0, 1],
                &[2],
            )
            .unwrap(),
            expected
        );
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
    fn tree_pair_block_apis_reject_mixed_multiplicity_flags() {
        let codomain = FusionTreeKey::from_sector_ids([1, 2], Some(7), [false, false], [], []);
        let domain = FusionTreeKey::from_sector_ids([3], Some(7), [false], [], []);
        let base = FusionTreeBlockKey::pair(codomain.clone(), domain.clone());
        let generic = FusionTreeBlockKey::pair(codomain.with_has_multiplicity(true), domain);

        // What: equal external labels do not form one block group when their
        // outer-multiplicity semantics differ.
        assert_mixed_tree_pair_block_group_is_rejected(&[base, generic]);
    }

    #[test]
    fn split_only_tree_pair_braid_uses_only_the_required_bend() {
        // What: moving the split from 1|2 to 2|1 with unchanged linearized
        // external-leg order evaluates one bend and no braid symbols, for both
        // the single-source and block entry points.
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
    fn split_only_su2_braid_matches_direct_repartition_in_both_directions() {
        // What: SU(2) 2|2 -> 3|1 and 2|2 -> 1|3 retain the exact tree keys,
        // dual flags, and bend coefficients of the direct repartition oracle.
        let source = FusionTreeBlockKey::pair_from_sector_ids(
            [1, 2],
            [2, 1],
            Some(1),
            [false, true],
            [true, false],
            [],
            [],
            [1],
            [1],
        );

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
                multiplicity_free_repartition_tree_pair(&SU2FusionRule, &source, target_rank)
                    .unwrap();
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
        let all_domain = FusionTreeBlockKey::pair_from_sector_ids(
            [],
            [1, 1],
            Some(0),
            [],
            [false, true],
            [],
            [],
            [],
            [1],
        );
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
    fn split_only_nested_product_braid_matches_direct_repartition() {
        // What: a non-Abelian fZ2 x U(1) x SU(2) tree preserves the product
        // bend phase and duality bookkeeping when only its 1|2 split changes.
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
        let source = FusionTreeBlockKey::pair_from_sector_ids(
            [coupled.id()],
            [domain_left.id(), domain_right.id()],
            Some(coupled.id()),
            [false],
            [false, true],
            [],
            [],
            [],
            [1],
        );

        let actual = multiplicity_free_braid_tree_pair(
            &rule,
            &source,
            &[0, 2],
            &[1],
            &[0],
            &[1, 2],
        )
        .unwrap();
        let expected = multiplicity_free_repartition_tree_pair(&rule, &source, 2).unwrap();
        assert_eq!(actual.len(), expected.len());
        for ((actual_key, actual_coefficient), (expected_key, expected_coefficient)) in
            actual.iter().zip(&expected)
        {
            assert_eq!(actual_key, expected_key);
            assert!((actual_coefficient - expected_coefficient).abs() < 1.0e-12);
        }
    }

    #[test]
    fn nonidentity_tree_pair_braid_does_not_enter_split_only_path() {
        // What: changing the split does not suppress a real external-leg
        // permutation, and malformed axis maps still fail validation.
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
        let tree = FusionTreeKey::from_sector_ids([1], Some(1), [false], [], []);
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
        let source = FusionTreeBlockKey::pair_from_sector_ids(
            [1, 0],
            [1],
            Some(1),
            [false, true],
            [true],
            [],
            [],
            [1],
            [],
        );

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
            FusionTreeBlockKey::pair_from_sector_ids(
                [sector.id()],
                [sector.id()],
                Some(sector.id()),
                [false],
                [false],
                [],
                [],
                [],
                [],
            )
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

        let scalar_source = FusionTreeBlockKey::pair_from_sector_ids(
            Vec::<usize>::new(),
            Vec::<usize>::new(),
            Some(z2_even().id()),
            Vec::<bool>::new(),
            Vec::<bool>::new(),
            Vec::<usize>::new(),
            Vec::<usize>::new(),
            Vec::<usize>::new(),
            Vec::<usize>::new(),
        );
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
    fn compact_lookup_distinguishes_rank1_tree_duality() {
        let nondual = FusionTreeBlockKey::pair_from_sector_ids(
            [1],
            [],
            None,
            [false],
            [],
            [],
            [],
            [],
            [],
        );
        let dual = FusionTreeBlockKey::pair_from_sector_ids(
            [1],
            [],
            None,
            [true],
            [],
            [],
            [],
            [],
            [],
        );
        let structure = SectorStructure::from_keys(1, [BlockKey::from(nondual)]).unwrap();

        assert!(structure.has_compact_lookup());
        assert_eq!(structure.find_index(&BlockKey::from(dual.clone())), None);
        assert_eq!(structure.find_fusion_tree_index(&dual), None);
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
    fn transpose_tree_pair_block_matches_per_source() {
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

        // Canonical planar transpose (`Tensor::transpose`): new codomain is the
        // reversed old domain, new domain the reversed old codomain — a cyclic
        // leg rotation.
        let codomain_permutation = [3usize];
        let domain_permutation = [2usize, 1, 0];
        let mut checked_blocks = 0;
        for src_keys in blocks.values() {
            let batched = multiplicity_free_transpose_tree_pair_block(
                &rule,
                src_keys,
                &codomain_permutation,
                &domain_permutation,
            )
            .unwrap();
            assert_eq!(batched.len(), src_keys.len());
            for (src, batched_rows) in src_keys.iter().zip(&batched) {
                let per_source = multiplicity_free_transpose_tree_pair(
                    &rule,
                    src,
                    &codomain_permutation,
                    &domain_permutation,
                )
                .unwrap();
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

    // --- Stage A: outer-multiplicity (Generic fusion) foundation ----------
    //
    // `ToyOmRule` is purely synthetic (following `AsymmetricAnyonicRule`
    // above as a template): sector 1 ("a") fuses with itself to sector 3
    // ("c") via two independent channels, i.e. N(a,a,c) = 2. It exists only
    // to exercise the `GenericFusionSymbols` wiring and the
    // `FusionTreeKey::has_multiplicity` gate added in this stage.
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
        let first = FusionTreeKey::new([a, a], Some(c), [false, false], [], [SectorId::new(vertices_first)]);
        let second = FusionTreeKey::new([a, a], Some(c), [false, false], [], [SectorId::new(vertices_second)]);
        (first, second)
    }

    #[test]
    fn fusion_tree_key_mult_free_equality_ignores_vertices() {
        // Zero-cost-degeneration gate: with has_multiplicity left at its
        // default (false, as every existing rule in this crate produces),
        // two keys that agree on (uncoupled, coupled, is_dual, innerlines)
        // but disagree on vertices are still `==`/same hash/`Ordering::Equal`
        // — unchanged from before `has_multiplicity` existed.
        let (first, second) = generic_tree_pair(0, 1);
        assert!(!first.has_multiplicity());
        assert!(!second.has_multiplicity());
        assert_eq!(first, second);
        assert_eq!(first.cmp(&second), std::cmp::Ordering::Equal);

        let mut set = std::collections::HashSet::new();
        set.insert(first);
        set.insert(second);
        assert_eq!(set.len(), 1, "mult-free keys differing only in vertices must collapse to one");
    }

    #[test]
    fn fusion_tree_key_generic_distinguishes_vertices() {
        // OM-distinction gate: the same two keys as above, but flagged
        // has_multiplicity=true (as a Generic-fusion tree from ToyOmRule
        // would be) are now distinct: vertices participates in Hash/Eq/Ord.
        let (first, second) = generic_tree_pair(0, 1);
        let first = first.with_has_multiplicity(true);
        let second = second.with_has_multiplicity(true);
        assert!(first.has_multiplicity());
        assert!(second.has_multiplicity());
        assert_ne!(first, second);
        assert_ne!(first.cmp(&second), std::cmp::Ordering::Equal);

        let mut set = std::collections::HashSet::new();
        set.insert(first);
        set.insert(second);
        assert_eq!(set.len(), 2, "Generic keys differing in vertices must stay distinct");
    }

    #[test]
    fn fusion_tree_key_has_multiplicity_is_itself_part_of_identity() {
        // A key with has_multiplicity=false and one with true, but
        // otherwise-identical fields including vertices, must not compare
        // equal either direction (see the Hash impl comment: gating the
        // vertices comparison on only one side's flag would be asymmetric).
        let (mult_free, _) = generic_tree_pair(0, 0);
        let generic = mult_free.clone().with_has_multiplicity(true);
        assert_ne!(mult_free, generic);
        assert_ne!(generic, mult_free);
        assert_ne!(mult_free.cmp(&generic), std::cmp::Ordering::Equal);
        assert_ne!(generic.cmp(&mult_free), std::cmp::Ordering::Equal);
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
        FusionTreeKey::new([a, a], Some(c), [false, false], [], [SectorId::new(vertex)])
            .with_has_multiplicity(true)
    }

    // Rank-3 tree [a, a, a] -> a: fuse a⊗a->c (OM vertex `vertex1`, N=2), then
    // c⊗a->a (vertex2, forced label 1). Innerline [c]. Mixes an OM vertex with
    // a multiplicity-1 vertex.
    fn unitary_rank3_tree(vertex1: usize) -> FusionTreeKey {
        let a = SectorId::new(UnitaryToyOmRule::A);
        let c = SectorId::new(UnitaryToyOmRule::C);
        FusionTreeKey::new(
            [a, a, a],
            Some(a),
            [false, false, false],
            [c],
            [SectorId::new(vertex1), SectorId::new(1)],
        )
        .with_has_multiplicity(true)
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
            .map(|(t, _)| {
                assert!(t.has_multiplicity(), "Generic braid output must be flagged");
                t.vertices()[0].id()
            })
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
                (t.vertices()[0].id(), t.vertices()[1].id())
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
            [a, a, a],
            Some(a),
            [false, false, false],
            [c],
            [SectorId::new(vertex1), SectorId::new(1)],
        )
        .with_has_multiplicity(true)
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
                assert_eq!(out.vertices()[1].id(), 1, "lambda must be 1");
                let sigma = out.vertices()[0].id() - 1;
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
                    let sigma = fin.vertices()[0].id() - 1;
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
                        [three, three, three, three],
                        Some(vac),
                        [false, false, false, false],
                        [three, three],
                        [SectorId::new(mu), SectorId::new(nu), SectorId::new(1)],
                    )
                    .with_has_multiplicity(true);
                    for (out, coeff) in
                        generic_artin_braid_at_with_inverse(&rule, &tree, 1, inverse).unwrap()
                    {
                        assert_eq!(out.innerlines(), &[three, three], "c'=3, e=3 unchanged");
                        assert_eq!(out.vertices()[2].id(), 1);
                        let sigma = out.vertices()[0].id();
                        let lambda = out.vertices()[1].id();
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

    fn a4_three() -> SectorId {
        SectorId::new(3)
    }

    // cod [3,3]->3 (vertex μ), dom [3]->3.
    fn a4_pair_rank2(mu: usize) -> FusionTreeBlockKey {
        let t = a4_three();
        let cod = FusionTreeKey::new([t, t], Some(t), [false, false], [], [SectorId::new(mu)])
            .with_has_multiplicity(true);
        let dom = FusionTreeKey::new([t], Some(t), [false], [], []).with_has_multiplicity(true);
        FusionTreeBlockKey::pair(cod, dom)
    }

    // cod [3,3,3]->3, inner=[x] (vertices v1,v2), dom [3]->3.
    fn a4_pair_rank3(inner: usize, v1: usize, v2: usize) -> FusionTreeBlockKey {
        let t = a4_three();
        let cod = FusionTreeKey::new(
            [t, t, t],
            Some(t),
            [false, false, false],
            [SectorId::new(inner)],
            [SectorId::new(v1), SectorId::new(v2)],
        )
        .with_has_multiplicity(true);
        let dom = FusionTreeKey::new([t], Some(t), [false], [], []).with_has_multiplicity(true);
        FusionTreeBlockKey::pair(cod, dom)
    }

    fn round_trip_bend(
        rule: &A4BendRule,
        pair: &FusionTreeBlockKey,
    ) -> std::collections::HashMap<FusionTreeBlockKey, f64> {
        let mut totals = std::collections::HashMap::new();
        for (mid, c1) in generic_bendright_tree_pair(rule, pair).unwrap() {
            for (out, c2) in generic_bendleft_tree_pair(rule, &mid).unwrap() {
                *totals.entry(out).or_insert(0.0) += c1 * c2;
            }
        }
        totals
    }

    fn assert_identity_map(
        totals: &std::collections::HashMap<FusionTreeBlockKey, f64>,
        expected_self: &FusionTreeBlockKey,
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
                (1, 1) => smallvec![SectorId::new(2)],
                _ => smallvec![SectorId::new(0)],
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
        let c = SectorId::new(2);
        // cod [1,1]->2 (vertex 1); dom []->2 (EMPTY domain ⇒ ν has nowhere to go).
        let cod = FusionTreeKey::new([a, a], Some(c), [false, false], [], [SectorId::new(1)])
            .with_has_multiplicity(true);
        let dom = FusionTreeKey::new([], Some(c), [], [], []).with_has_multiplicity(true);
        let pair = FusionTreeBlockKey::pair(cod, dom);
        let out = generic_bendright_tree_pair(&rule, &pair).unwrap();
        assert_eq!(out.len(), 1, "empty domain collapses ν to one key");
        // coeff0 = √dim(2)·(1/√dim(1)) = 1; keep-last ⇒ B[0,1] = 0.7.
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
            assert_eq!(key.domain_tree().vertices()[0].id(), mu, "rank2 ν == μ (B diagonal)");
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
            let left_coupled = key.codomain_tree().coupled().unwrap().id();
            assert_eq!(left_coupled, inner, "rank3 left_coupled == innerline");
            assert_eq!(key.codomain_tree().vertices()[0].id(), cod_vtx, "rank3 cod vtx");
            assert_eq!(key.domain_tree().vertices()[0].id(), dom_vtx, "rank3 dom vtx");
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
            let cod = FusionTreeKey::new([s, s], Some(s), [false, false], [], [SectorId::new(mu)])
                .with_has_multiplicity(true);
            let dom = FusionTreeKey::new([s], Some(s), [false], [], []).with_has_multiplicity(true);
            let pair = FusionTreeBlockKey::pair(cod, dom);
            let out = generic_bendright_tree_pair(&rule, &pair).unwrap();
            // Collect coeff keyed by output domain vertex label (=ν+1).
            let mut got = [0.0f64; 2];
            for (key, coeff) in &out {
                let nu = key.domain_tree().vertices()[0].id(); // 1-based ν label
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
            [t, t, t],
            Some(t),
            [false, false, false],
            [SectorId::new(inner)],
            [SectorId::new(v1), SectorId::new(v2)],
        )
        .with_has_multiplicity(true)
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
                tr.coupled().unwrap().id() == coupled && tr.vertices()[0].id() == vtx
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
        pair: &FusionTreeBlockKey,
    ) -> std::collections::HashMap<FusionTreeBlockKey, f64> {
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
            let cod = FusionTreeKey::new([t, t], Some(t), [false, false], [], [SectorId::new(mu)])
                .with_has_multiplicity(true);
            let dom = FusionTreeKey::new([t], Some(t), [false], [], []).with_has_multiplicity(true);
            let map = foldright_map(&rule, &FusionTreeBlockKey::pair(cod, dom));
            // TK dst: cod [3]->3, dom [3,3]->3 (isdual=(true,false)) vtx μ, U[μ,μ]=1.
            let exp_cod =
                FusionTreeKey::new([t], Some(t), [false], [], []).with_has_multiplicity(true);
            let exp_dom =
                FusionTreeKey::new([t, t], Some(t), [true, false], [], [SectorId::new(mu)])
                    .with_has_multiplicity(true);
            let exp = FusionTreeBlockKey::pair(exp_cod, exp_dom);
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
        let dom = FusionTreeKey::new([t], Some(t), [false], [], []).with_has_multiplicity(true);
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
            let pair = FusionTreeBlockKey::pair(cod, dom.clone());
            let map = foldright_map(&rule, &pair);
            for (ri, &(cc, cv, dc, dv)) in rows.iter().enumerate() {
                let ex_cod = FusionTreeKey::new(
                    [t, t], Some(SectorId::new(cc)), [false, false], [], [SectorId::new(cv)],
                )
                .with_has_multiplicity(true);
                let ex_dom = FusionTreeKey::new(
                    [t, t], Some(SectorId::new(dc)), [true, false], [], [SectorId::new(dv)],
                )
                .with_has_multiplicity(true);
                let key = FusionTreeBlockKey::pair(ex_cod, ex_dom);
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
        let dom = FusionTreeKey::new([t], Some(t), [false], [], []).with_has_multiplicity(true);
        for (inner, v1, v2) in
            [(0, 1, 1), (1, 1, 1), (2, 1, 1), (3, 1, 1), (3, 2, 1), (3, 1, 2), (3, 2, 2)]
        {
            let pair = FusionTreeBlockKey::pair(a4f_rank3(inner, v1, v2), dom.clone());
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
        let dom = FusionTreeKey::new([t], Some(t), [false], [], []).with_has_multiplicity(true);
        for (inner, v1, v2) in [(0, 1, 1), (3, 1, 1), (3, 2, 1), (3, 1, 2), (3, 2, 2)] {
            let pair = FusionTreeBlockKey::pair(a4f_rank3(inner, v1, v2), dom.clone());
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
                    [t, t], Some(t), [false, false], [], [SectorId::new(cod_mu)],
                )
                .with_has_multiplicity(true);
                let dom = FusionTreeKey::new(
                    [t, t], Some(t), [false, false], [], [SectorId::new(dv)],
                )
                .with_has_multiplicity(true);
                let _ = dom_inner; // rank-2 dom has no innerline; kept for label clarity
                let pair = FusionTreeBlockKey::pair(cod, dom);
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
                    [t, t], Some(t), [false, false], [], [SectorId::new(cod_mu)],
                )
                .with_has_multiplicity(true);
                let dom = FusionTreeKey::new(
                    [t, t], Some(t), [false, false], [], [SectorId::new(dv)],
                )
                .with_has_multiplicity(true);
                let pair = FusionTreeBlockKey::pair(cod, dom);
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
            let cod = FusionTreeKey::new([s, s], Some(s), [false, false], [], [SectorId::new(mu)])
                .with_has_multiplicity(true);
            let dom = FusionTreeKey::new([s], Some(s), [false], [], []).with_has_multiplicity(true);
            let pair = FusionTreeBlockKey::pair(cod, dom);
            let out = generic_foldright_tree_pair(&rule, &pair).unwrap();
            let mut got = [cx(0.0, 0.0); 2];
            for (key, coeff) in &out {
                let nu = key.domain_tree().vertices()[0].id();
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
            let cod = FusionTreeKey::new([s, s], Some(s), [false, false], [], [SectorId::new(mu)])
                .with_has_multiplicity(true);
            let dom = FusionTreeKey::new([s], Some(s), [false], [], []).with_has_multiplicity(true);
            let pair = FusionTreeBlockKey::pair(cod, dom);
            let mut totals: std::collections::HashMap<FusionTreeBlockKey, Complex64> =
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
            let cod = FusionTreeKey::new([s, s], Some(s), [false, false], [], [SectorId::new(mu)])
                .with_has_multiplicity(true);
            let dom = FusionTreeKey::new([s], Some(s), [false], [], []).with_has_multiplicity(true);
            let pair = FusionTreeBlockKey::pair(cod, dom);
            let mut totals: std::collections::HashMap<FusionTreeBlockKey, Complex64> =
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
            [s3, s3],
            Some(s3),
            [false, false],
            [],
            [SectorId::new(1)],
        )
        .with_has_multiplicity(true);
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
            FusionTreeKey::new([s1, s2], Some(s3), [false, false], [], [SectorId::new(1)])
                .with_has_multiplicity(true);
        let domain =
            FusionTreeKey::new([s3, s3], Some(s3), [false, false], [], [SectorId::new(1)])
                .with_has_multiplicity(true);
        let pair = FusionTreeBlockKey::pair(codomain, domain);
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
                [s42, s31], Some(s31), [false, false], [], [SectorId::new(mu)],
            )
            .with_has_multiplicity(true);
            let dom =
                FusionTreeKey::new([s31], Some(s31), [false], [], []).with_has_multiplicity(true);
            let out = generic_bendright_tree_pair(&rule, &FusionTreeBlockKey::pair(cod, dom))
                .unwrap();
            let mut got = [0.0f64; 2];
            for (key, coeff) in &out {
                let nu = key.domain_tree().vertices().last().unwrap().id();
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

    // Round-trip with a real non-diagonal SU(3) B: bendright∘bendleft == id
    // (B_fwd · B_ret = I₂), exercising the non-trivial off-diagonal mixing.
    #[test]
    fn b2b_su3_bend_round_trip_identity() {
        let rule = Su3BendRule;
        let s42 = SectorId::new(1);
        let s31 = SectorId::new(2);
        for mu in 1..=2usize {
            let cod = FusionTreeKey::new(
                [s42, s31], Some(s31), [false, false], [], [SectorId::new(mu)],
            )
            .with_has_multiplicity(true);
            let dom =
                FusionTreeKey::new([s31], Some(s31), [false], [], []).with_has_multiplicity(true);
            let pair = FusionTreeBlockKey::pair(cod, dom);
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

    fn map_terms(terms: Vec<(FusionTreeBlockKey, f64)>) -> HashMap<FusionTreeBlockKey, f64> {
        let mut map = HashMap::new();
        for (key, coeff) in terms {
            *map.entry(key).or_insert(0.0) += coeff;
        }
        map
    }

    fn assert_term_maps_eq(
        got: &HashMap<FusionTreeBlockKey, f64>,
        want: &HashMap<FusionTreeBlockKey, f64>,
        label: &str,
    ) {
        let mut keys: std::collections::HashSet<&FusionTreeBlockKey> = got.keys().collect();
        keys.extend(want.keys());
        for key in keys {
            let g = got.get(key).copied().unwrap_or(0.0);
            let w = want.get(key).copied().unwrap_or(0.0);
            assert!((g - w).abs() < 1e-10, "{label}: coeff {g} != {w}");
        }
    }

    fn assert_identity_term_map(
        got: &HashMap<FusionTreeBlockKey, f64>,
        self_pair: &FusionTreeBlockKey,
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
    fn a4_pair_rank1_1() -> FusionTreeBlockKey {
        let t = SectorId::new(3);
        let cod = FusionTreeKey::new([t], Some(t), [false], [], []).with_has_multiplicity(true);
        let dom = FusionTreeKey::new([t], Some(t), [false], [], []).with_has_multiplicity(true);
        FusionTreeBlockKey::pair(cod, dom)
    }

    // A4 rank-2/rank-1 pair: cod [3,3]->3 (vtx μ), dom [3]->3 — an
    // outer-multiplicity tree pair with N(3,3,3)=2.
    fn a4_pair_rank2_1(mu: usize) -> FusionTreeBlockKey {
        let t = SectorId::new(3);
        let cod = FusionTreeKey::new([t, t], Some(t), [false, false], [], [SectorId::new(mu)])
            .with_has_multiplicity(true);
        let dom = FusionTreeKey::new([t], Some(t), [false], [], []).with_has_multiplicity(true);
        FusionTreeBlockKey::pair(cod, dom)
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
        assert_eq!(rule.provenance(), 0x2afd_b9a5_dcf6_18e6);
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
                        [eight, eight, eight, eight],
                        Some(vac),
                        [false, false, false, false],
                        [eight, eight],
                        [SectorId::new(smu), SectorId::new(snu), SectorId::new(1)],
                    )
                    .with_has_multiplicity(true);
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
                                && w0 == v[0].id()
                                && w1 == v[1].id()
                                && w2 == v[2].id()
                                && (val - coeff).abs() < 1e-10
                        });
                        match idx {
                            Some(k) => matched[k] = true,
                            None => panic!(
                                "inv={inverse} src=({smu},{snu}) spurious term \
                                 inner=[{i0},{i1}] vtx=[{},{},{}] = {coeff}",
                                v[0].id(),
                                v[1].id(),
                                v[2].id()
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
        let cod = FusionTreeKey::new([eight, eight], Some(vac), [false, false], [], [SectorId::new(1)])
            .with_has_multiplicity(true);
        let dom = FusionTreeKey::new([], Some(vac), [], [], []).with_has_multiplicity(true);
        let pair = FusionTreeBlockKey::pair(cod, dom);
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
                    [eight, eight, eight, eight],
                    Some(vac),
                    [false, false, false, false],
                    [eight, eight],
                    [SectorId::new(smu), SectorId::new(snu), SectorId::new(1)],
                )
                .with_has_multiplicity(true);
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
                [eight, eight],
                Some(eight),
                [false, false],
                [],
                [SectorId::new(mu)],
            )
            .with_has_multiplicity(true);
            let dom = FusionTreeKey::new([eight], Some(eight), [false], [], [])
                .with_has_multiplicity(true);
            let pair = FusionTreeBlockKey::pair(cod, dom);
            let mut totals: std::collections::HashMap<FusionTreeBlockKey, f64> =
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
            let c = t.coupled().unwrap().id();
            let inner: Vec<usize> = t.innerlines().iter().map(|x| x.id()).collect();
            let vtx: Vec<usize> = t.vertices().iter().map(|x| x.id()).collect();
            set.insert((c, inner, vtx));
        }
        set.into_iter().collect()
    }

    #[test]
    fn refute_a_enum_rank2_88() {
        let rule = su3();
        let eight = su3_id(1, 1);
        let trees = refute_enum_codomain_trees(&rule, &[eight, eight]);
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
            let got: std::collections::BTreeSet<(usize, usize, usize)> = keys
                .iter()
                .map(|k| {
                    let t = k.codomain_tree();
                    assert_eq!(t.coupled(), Some(c));
                    (
                        t.innerlines()[0].id(),
                        t.vertices()[0].id(),
                        t.vertices()[1].id(),
                    )
                })
                .collect();
            let want: std::collections::BTreeSet<(usize, usize, usize)> =
                trees.iter().copied().collect();
            assert_eq!(got, want, "coupled {coupled}: tree set mismatch vs TK");
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
            );
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
        assert_eq!(hom_space_intern_table().read().unwrap().entries.len(), HOM_SPACE_INTERN_CAP);
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

    // Canary (#153) against silent growth of the hottest recoupling-plan key:
    // pins today's `size_of::<FusionTreeKey>()`. If this fails, a field was
    // added or a `SmallVec` inline capacity changed — re-check the zero-cost
    // mult-free Hash/Eq/Ord contract documented on `FusionTreeKey` before
    // bumping the constant.
    #[test]
    fn fusion_tree_key_size_has_not_silently_grown() {
        assert_eq!(std::mem::size_of::<FusionTreeKey>(), 264);
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

    #[test]
    fn block_structure_intern_tables_plateau_under_distinct_growth() {
        // What: floods the shared block-structure intern/arc tables. Bounded
        // (`<=`) rather than exact-cap assertion below, so this no longer
        // needs the lock to pass — but it takes it anyway (uniform with the
        // other resetters/flooders below) since it's still a large flood of
        // shared state that could otherwise perturb a stricter sibling test.
        let _guard = test_support::CACHE_TEST_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        // Interning far more distinct structures than the cap must leave the
        // capped tables pinned at the cap. Before the LRU cap they grew linearly.
        let overflow = BLOCK_STRUCTURE_INTERN_CAP + 256;
        for i in 0..overflow {
            // Distinct shape per iteration => distinct interned content and a
            // distinct arc-dedup entry (shape dims are metadata, not allocated).
            let _ = BlockStructure::trivial(&[i + 1]).unwrap().into_shared();
        }
        let intern_len = block_structure_intern_table().read().unwrap().len();
        let arc_len = block_structure_arc_table().read().unwrap().len();
        // Other tests share these global tables and may reset or evict
        // concurrently, so exact saturation (== cap) is racy — asserting it
        // made this test flaky in CI. Boundedness alone proves the plateau:
        // uncapped tables would exceed the cap after cap+256 distinct inserts.
        assert!(intern_len <= BLOCK_STRUCTURE_INTERN_CAP);
        assert!(arc_len <= BLOCK_STRUCTURE_INTERN_CAP);
    }

    #[test]
    fn evicted_block_structure_content_reinterns_with_fresh_id() {
        // What: floods the shared block-structure intern table to force an
        // eviction. Takes the lock alongside the table's other flooder/resetter
        // for the same reason as the plateau test above.
        let _guard = test_support::CACHE_TEST_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        // Pinned invariant: an id is NEVER reused. An evicted content that is
        // re-interned gets a strictly greater id (monotonic counter), so a
        // downstream cache keyed by the old id can never be aliased by it.
        let base = 900_000_000usize; // Outside the range other tests intern.
        let id_before = BlockStructure::trivial(&[base]).unwrap().content_id();
        // Flood past the cap with distinct contents to evict the probe entry
        // (never touched again, so it ages to the LRU tail and is dropped).
        for i in 0..(BLOCK_STRUCTURE_INTERN_CAP + 64) {
            let _ = BlockStructure::trivial(&[base + 1 + i]).unwrap();
        }
        let id_after = BlockStructure::trivial(&[base]).unwrap().content_id();
        assert!(
            id_after > id_before,
            "evicted content must re-intern with a strictly greater id, \
             got before={id_before} after={id_after}"
        );
    }

    #[test]
    fn reset_core_intern_tables_clears_without_reusing_ids() {
        // What: calls `reset_core_intern_tables` directly — the resetter half
        // of this species. Takes the shared lock so it can't wipe the table
        // out from under the flood/plateau tests above mid-run.
        let _guard = test_support::CACHE_TEST_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        // Reset coherence: content ids issued after a reset must exceed any
        // issued before it (the counter is not reset), so a tensors-layer key
        // still holding a pre-reset id can never alias post-reset content.
        let base = 800_000_000usize;
        let id_before = BlockStructure::trivial(&[base]).unwrap().content_id();
        reset_core_intern_tables();
        let id_after = BlockStructure::trivial(&[base]).unwrap().content_id();
        assert!(
            id_after > id_before,
            "reset must not reuse content ids, got before={id_before} after={id_after}"
        );
    }
}
