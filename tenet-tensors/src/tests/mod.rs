use super::*;

/// Fixture layout: subblocks packed contiguously in key order. Not a product
/// layout (the only one is the coupled sector matrix); fixtures use it to
/// exercise the arbitrary-strided-view contract of `BlockStructure`.
pub(crate) fn packed_fixture_structure<I, K>(
    rank: usize,
    blocks: I,
) -> Result<BlockStructure, CoreError>
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

use num_complex::{Complex32, Complex64};
use num_traits::{One, Zero};
use std::fmt::Debug;
use std::ops::{Add, Mul};
use tenet_core::{
    multiplicity_free_repartition_tree_pair, unique_permute_tree_pair, BlockKey, BlockSpec,
    BlockStructure, BlockView, BlockViewMut, BraidingStyleKind, CoreError, DegeneracyStructure,
    FermionParityFusionRule, FusionProductSpace, FusionRule, FusionStyleKind, FusionTensorMapSpace,
    FusionTreeBlockKey, FusionTreeGroupKey, FusionTreeHomSpace, FusionTreeKey, HostReadableStorage,
    HostWritableStorage, MultiplicityFreeFusionRule, MultiplicityFreeFusionSymbols,
    MultiplicityFreePivotalSymbols, MultiplicityFreeRigidSymbols, Placement, ProductFusionRule,
    SU2FusionRule, SU2Irrep, SectorId, SectorLeg, SectorStructure, SectorVec, SimilarStorage,
    TensorMap, TensorMapSpace, TensorStorage, Trivial, U1FusionRule, U1Irrep, Z2FusionRule,
};
use tenet_dense::{DenseDotConfig, DenseError, DenseExecutor, DenseRead, DenseWrite};

#[derive(Clone, Debug, PartialEq)]
struct TestHostStorage<T>(Vec<T>);

#[derive(Clone, Debug, PartialEq)]
struct TestHostReadStorage<T>(Vec<T>);

impl<T> TestHostStorage<T> {
    fn new(data: Vec<T>) -> Self {
        Self(data)
    }
}

impl<T> TestHostReadStorage<T> {
    fn new(data: Vec<T>) -> Self {
        Self(data)
    }
}

impl<T> TensorStorage<T> for TestHostStorage<T> {
    fn len(&self) -> usize {
        self.0.len()
    }

    fn placement(&self) -> Placement {
        Placement::Host
    }
}

impl<T> HostReadableStorage<T> for TestHostStorage<T> {
    fn as_slice(&self) -> &[T] {
        &self.0
    }
}

impl<T> HostWritableStorage<T> for TestHostStorage<T> {
    fn as_mut_slice(&mut self) -> &mut [T] {
        &mut self.0
    }
}

impl<T> TensorStorage<T> for TestHostReadStorage<T> {
    fn len(&self) -> usize {
        self.0.len()
    }

    fn placement(&self) -> Placement {
        Placement::Host
    }
}

impl<T> HostReadableStorage<T> for TestHostReadStorage<T> {
    fn as_slice(&self) -> &[T] {
        &self.0
    }
}

fn test_host_tensor_map<T, const NOUT: usize, const NIN: usize>(
    data: Vec<T>,
    space: TensorMapSpace<NOUT, NIN>,
) -> TensorMap<T, NOUT, NIN, Trivial, TestHostStorage<T>> {
    let structure = BlockStructure::trivial(space.dims()).unwrap();
    TensorMap::from_storage_with_structure(TestHostStorage::new(data), space, structure).unwrap()
}

fn test_host_tensor_map_with_structure<T, const NOUT: usize, const NIN: usize>(
    data: Vec<T>,
    space: TensorMapSpace<NOUT, NIN>,
    structure: BlockStructure,
) -> TensorMap<T, NOUT, NIN, Trivial, TestHostStorage<T>> {
    TensorMap::from_storage_with_structure(TestHostStorage::new(data), space, structure).unwrap()
}

fn test_host_fusion_tensor_map<T, const NOUT: usize, const NIN: usize>(
    data: Vec<T>,
    fusion_space: FusionTensorMapSpace<NOUT, NIN>,
) -> TensorMap<T, NOUT, NIN, Trivial, TestHostStorage<T>> {
    TensorMap::from_storage_with_fusion_space(TestHostStorage::new(data), fusion_space).unwrap()
}

fn test_host_read_tensor_map<T, const NOUT: usize, const NIN: usize>(
    data: Vec<T>,
    space: TensorMapSpace<NOUT, NIN>,
) -> TensorMap<T, NOUT, NIN, Trivial, TestHostReadStorage<T>> {
    let structure = BlockStructure::trivial(space.dims()).unwrap();
    TensorMap::from_storage_with_structure(TestHostReadStorage::new(data), space, structure)
        .unwrap()
}

fn test_host_read_tensor_map_with_structure<T, const NOUT: usize, const NIN: usize>(
    data: Vec<T>,
    space: TensorMapSpace<NOUT, NIN>,
    structure: BlockStructure,
) -> TensorMap<T, NOUT, NIN, Trivial, TestHostReadStorage<T>> {
    TensorMap::from_storage_with_structure(TestHostReadStorage::new(data), space, structure)
        .unwrap()
}

fn test_host_read_fusion_tensor_map<T, const NOUT: usize, const NIN: usize>(
    data: Vec<T>,
    fusion_space: FusionTensorMapSpace<NOUT, NIN>,
) -> TensorMap<T, NOUT, NIN, Trivial, TestHostReadStorage<T>> {
    TensorMap::from_storage_with_fusion_space(TestHostReadStorage::new(data), fusion_space).unwrap()
}

fn fusion_tree_test_key<
    const COD: usize,
    const DOM: usize,
    const COD_DUAL: usize,
    const DOM_DUAL: usize,
>(
    codomain: [usize; COD],
    domain: [usize; DOM],
    coupled: usize,
    codomain_is_dual: [bool; COD_DUAL],
    domain_is_dual: [bool; DOM_DUAL],
) -> BlockKey {
    BlockKey::from(FusionTreeBlockKey::pair_from_sector_ids(
        codomain,
        domain,
        Some(coupled),
        codomain_is_dual,
        domain_is_dual,
        [coupled + 100],
        [coupled + 200],
        [coupled + 300],
        [coupled + 400],
    ))
}

fn expect_tree_key(key: &BlockKey) -> FusionTreeBlockKey {
    match key {
        BlockKey::FusionTree(tree) => tree.clone(),
        BlockKey::Dense => panic!("test expected a fusion-tree key"),
    }
}

fn empty_fusion_tree() -> FusionTreeKey {
    empty_fusion_tree_with_coupled(None)
}

fn empty_fusion_tree_with_coupled(coupled: Option<usize>) -> FusionTreeKey {
    FusionTreeKey::try_new_for_rule(
        &Z2FusionRule,
        Vec::<SectorId>::new(),
        coupled.map(SectorId::new),
        Vec::<bool>::new(),
        Vec::<SectorId>::new(),
        Vec::<SectorId>::new(),
    )
    .unwrap()
}

fn all_codomain_fusion_tree_test_key<
    const COD: usize,
    const COD_DUAL: usize,
    const COD_INNER: usize,
    const COD_VERTICES: usize,
>(
    codomain: [usize; COD],
    coupled: Option<usize>,
    codomain_is_dual: [bool; COD_DUAL],
    codomain_innerlines: [usize; COD_INNER],
    codomain_vertices: [usize; COD_VERTICES],
) -> BlockKey {
    BlockKey::from(FusionTreeBlockKey::pair(
        FusionTreeKey::try_from_sector_ids_for_rule(
            &Z2FusionRule,
            codomain,
            coupled,
            codomain_is_dual,
            codomain_innerlines,
            codomain_vertices,
        )
        .unwrap(),
        empty_fusion_tree(),
    ))
}

type FpU1Rule = ProductFusionRule<FermionParityFusionRule, U1FusionRule>;
type FpU1Su2Rule = ProductFusionRule<FpU1Rule, SU2FusionRule>;

fn fz2_u1_su2_tree_pair_fixture() -> (
    FpU1Su2Rule,
    FusionTensorMapSpace<2, 1>,
    FusionTensorMapSpace<2, 1>,
    [SectorId; 2],
) {
    let left_rule = FpU1Rule::default();
    let rule = FpU1Su2Rule::default();
    let even = SectorId::new(0);
    let odd = SectorId::new(1);
    let left_sector =
        |parity, charge| left_rule.encode_sector(parity, U1Irrep::new(charge).sector_id());
    let sector = |parity, charge, twice_spin| {
        rule.encode_sector(
            left_sector(parity, charge),
            SU2Irrep::from_twice_spin(twice_spin).sector_id(),
        )
    };

    let a = sector(odd, 1, 1);
    let b = sector(odd, -1, 1);
    let c0 = sector(even, 0, 0);
    let c1 = sector(even, 0, 2);
    let src_hom = FusionTreeHomSpace::new(
        FusionProductSpace::new([
            SectorLeg::new([(a, 1)], false),
            SectorLeg::new([(b, 1)], false),
        ]),
        FusionProductSpace::new([SectorLeg::new([(c0, 1), (c1, 1)], false)]),
    );
    let dst_hom = FusionTreeHomSpace::new(
        FusionProductSpace::new([
            SectorLeg::new([(b, 1)], false),
            SectorLeg::new([(a, 1)], false),
        ]),
        FusionProductSpace::new([SectorLeg::new([(c0, 1), (c1, 1)], false)]),
    );
    let src_space = FusionTensorMapSpace::from_degeneracy_shapes(
        TensorMapSpace::<2, 1>::from_dims([1, 1], [1]).unwrap(),
        src_hom,
        &rule,
        [vec![1, 1, 1], vec![1, 1, 1]],
    )
    .unwrap();
    let dst_space = FusionTensorMapSpace::from_degeneracy_shapes(
        TensorMapSpace::<2, 1>::from_dims([1, 1], [1]).unwrap(),
        dst_hom,
        &rule,
        [vec![1, 1, 1], vec![1, 1, 1]],
    )
    .unwrap();

    (rule, src_space, dst_space, [c0, c1])
}

fn single_transform_coefficient_for_coupled(
    plan: &TreeTransformGroupPlan<f64>,
    coupled: SectorId,
) -> f64 {
    let mut found = None;
    for spec in plan.specs() {
        assert_eq!(spec.src_keys().len(), 1);
        assert_eq!(spec.dst_keys().len(), 1);
        assert_eq!(spec.recoupling_coefficients_dst_src().len(), 1);
        let dst_coupled = expect_tree_key(&spec.dst_keys()[0]).coupled().unwrap();
        if dst_coupled == coupled {
            assert!(found.is_none(), "duplicate coefficient for {coupled:?}");
            found = Some(spec.recoupling_coefficients_dst_src()[0]);
        }
    }
    found.unwrap_or_else(|| panic!("missing coefficient for {coupled:?}"))
}

fn expected_single_tree_pair_replay(
    plan: &TreeTransformGroupPlan<f64>,
    dst_structure: &BlockStructure,
    src_structure: &BlockStructure,
    initial_dst: &[f64],
    src_data: &[f64],
    alpha: f64,
    beta: f64,
) -> Vec<f64> {
    let mut expected = initial_dst
        .iter()
        .map(|value| beta * value)
        .collect::<Vec<_>>();
    for spec in plan.specs() {
        assert_eq!(spec.src_keys().len(), 1);
        assert_eq!(spec.dst_keys().len(), 1);
        assert_eq!(spec.recoupling_coefficients_dst_src().len(), 1);
        let src_key = &spec.src_keys()[0];
        let dst_key = &spec.dst_keys()[0];
        let src_offset = src_structure.block_by_key(src_key).unwrap().offset();
        let dst_offset = dst_structure.block_by_key(dst_key).unwrap().offset();
        expected[dst_offset] +=
            alpha * spec.recoupling_coefficients_dst_src()[0] * src_data[src_offset];
    }
    expected
}

fn column_major_structure_like(structure: &BlockStructure, shape: Vec<usize>) -> BlockStructure {
    let blocks = (0..structure.block_count())
        .map(|index| (structure.block(index).unwrap().key().clone(), shape.clone()));
    packed_fixture_structure(structure.rank(), blocks).unwrap()
}

#[derive(Clone, Copy, Debug)]
struct UniqueZ2Rule;

impl FusionRule for UniqueZ2Rule {
    fn rule_identity(&self) -> tenet_core::RuleIdentity {
        tenet_core::RuleIdentity::of_type::<Self>()
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
        vec![SectorId::new((left.id() + right.id()) % 2)].into()
    }
}

impl MultiplicityFreeFusionRule for UniqueZ2Rule {}

impl MultiplicityFreeFusionSymbols for UniqueZ2Rule {
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

impl MultiplicityFreePivotalSymbols for UniqueZ2Rule {
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

#[derive(Clone, Copy, Debug)]
struct UniqueAnyonicRule;

impl FusionRule for UniqueAnyonicRule {
    fn rule_identity(&self) -> tenet_core::RuleIdentity {
        tenet_core::RuleIdentity::of_type::<Self>()
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
        vec![SectorId::new((left.id() + right.id()) % 2)].into()
    }
}

impl MultiplicityFreeFusionRule for UniqueAnyonicRule {}

impl MultiplicityFreeFusionSymbols for UniqueAnyonicRule {
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

    fn r_symbol_scalar(&self, left: SectorId, right: SectorId, _coupled: SectorId) -> Self::Scalar {
        if left.id() == 1 && right.id() == 1 {
            -2.0
        } else {
            1.0
        }
    }
}

impl MultiplicityFreePivotalSymbols for UniqueAnyonicRule {
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

impl MultiplicityFreeRigidSymbols for UniqueAnyonicRule {
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
struct UnitaryPhaseAnyonicRule;

impl FusionRule for UnitaryPhaseAnyonicRule {
    fn rule_identity(&self) -> tenet_core::RuleIdentity {
        tenet_core::RuleIdentity::of_type::<Self>()
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

    fn supports_unitary_braid_dagger(&self) -> bool {
        true
    }

    fn fusion_channels(&self, left: SectorId, right: SectorId) -> SectorVec {
        vec![SectorId::new((left.id() + right.id()) % 4)].into()
    }
}

impl MultiplicityFreeFusionRule for UnitaryPhaseAnyonicRule {}

impl MultiplicityFreeFusionSymbols for UnitaryPhaseAnyonicRule {
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

    fn r_symbol_scalar(&self, left: SectorId, right: SectorId, _coupled: SectorId) -> Self::Scalar {
        if matches!((left.id(), right.id()), (1, 1) | (1, 3) | (3, 1)) {
            Complex64::new(0.0, 1.0)
        } else {
            Complex64::new(1.0, 0.0)
        }
    }
}

impl MultiplicityFreePivotalSymbols for UnitaryPhaseAnyonicRule {
    fn bendright_scalar(
        &self,
        _left_coupled: SectorId,
        _bent_sector: SectorId,
        _coupled: SectorId,
        _bent_leg_is_dual: bool,
    ) -> Self::Scalar {
        Complex64::new(1.0, 0.0)
    }

    fn foldright_scalar(
        &self,
        _source: &FusionTreeBlockKey,
        _destination: &FusionTreeBlockKey,
    ) -> Self::Scalar {
        Complex64::new(1.0, 0.0)
    }
}

impl MultiplicityFreeRigidSymbols for UnitaryPhaseAnyonicRule {
    fn dim_scalar(&self, _sector: SectorId) -> Self::Scalar {
        Complex64::new(1.0, 0.0)
    }

    fn inv_dim_scalar(&self, _sector: SectorId) -> Self::Scalar {
        Complex64::new(1.0, 0.0)
    }

    fn sqrt_dim_scalar(&self, _sector: SectorId) -> Self::Scalar {
        Complex64::new(1.0, 0.0)
    }

    fn inv_sqrt_dim_scalar(&self, _sector: SectorId) -> Self::Scalar {
        Complex64::new(1.0, 0.0)
    }

    fn twist_scalar(&self, _sector: SectorId) -> Self::Scalar {
        Complex64::new(1.0, 0.0)
    }

    fn frobenius_schur_phase_scalar(&self, _sector: SectorId) -> Self::Scalar {
        Complex64::new(1.0, 0.0)
    }
}

#[derive(Clone, Copy, Debug)]
struct UniquePlanarRule;

impl FusionRule for UniquePlanarRule {
    fn rule_identity(&self) -> tenet_core::RuleIdentity {
        tenet_core::RuleIdentity::of_type::<Self>()
    }
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
        vec![SectorId::new((left.id() + right.id()) % 2)].into()
    }
}

impl MultiplicityFreeFusionRule for UniquePlanarRule {}

#[derive(Clone, Copy, Debug)]
struct SimpleSu2Rule;

impl FusionRule for SimpleSu2Rule {
    fn rule_identity(&self) -> tenet_core::RuleIdentity {
        tenet_core::RuleIdentity::of_type::<Self>()
    }
    fn fusion_style(&self) -> FusionStyleKind {
        FusionStyleKind::Simple
    }

    fn braiding_style(&self) -> BraidingStyleKind {
        BraidingStyleKind::Bosonic
    }

    fn vacuum(&self) -> SectorId {
        SectorId::new(0)
    }

    fn fusion_channels(&self, left: SectorId, right: SectorId) -> SectorVec {
        let min = left.id().abs_diff(right.id());
        let max = left.id() + right.id();
        (min..=max).step_by(2).map(SectorId::new).collect()
    }
}

impl MultiplicityFreeFusionRule for SimpleSu2Rule {}

#[derive(Clone, Copy, Debug)]
struct GenericMultiplicityRule;

impl FusionRule for GenericMultiplicityRule {
    fn rule_identity(&self) -> tenet_core::RuleIdentity {
        tenet_core::RuleIdentity::of_type::<Self>()
    }
    fn fusion_style(&self) -> FusionStyleKind {
        FusionStyleKind::Generic
    }

    fn braiding_style(&self) -> BraidingStyleKind {
        BraidingStyleKind::Anyonic
    }

    fn vacuum(&self) -> SectorId {
        SectorId::new(0)
    }

    fn fusion_channels(&self, left: SectorId, right: SectorId) -> SectorVec {
        let channels: Vec<SectorId> = match (left.id(), right.id()) {
            (1, 1) => vec![SectorId::new(0), SectorId::new(1)],
            (0, x) | (x, 0) => vec![SectorId::new(x)],
            _ => Vec::new(),
        };
        channels.into()
    }

    fn nsymbol(&self, left: SectorId, right: SectorId, coupled: SectorId) -> usize {
        match (left.id(), right.id(), coupled.id()) {
            (1, 1, 1) => 2,
            _ => usize::from(self.fusion_channels(left, right).contains(&coupled)),
        }
    }
}

fn assert_tensorcopy_dtype<T>(values: Vec<T>, fill: T)
where
    T: Copy + Clone + Debug + PartialEq + strided_kernel::MaybeSendSync,
{
    let space = TensorMapSpace::<2, 0>::from_dims([2, 2], []).unwrap();
    let src = TensorMap::<T, 2, 0>::from_vec(values.clone(), space.clone()).unwrap();
    let mut dst = TensorMap::<T, 2, 0>::filled(fill, space).unwrap();

    tensorcopy_into(&mut dst, &src).unwrap();

    assert_eq!(dst.data(), values.as_slice());
}

fn assert_tensoradd_dtype<T>(
    values: Vec<T>,
    fill: T,
    alpha: T,
    assign_expected: Vec<T>,
    add_expected: Vec<T>,
) where
    T: Copy
        + Clone
        + Debug
        + PartialEq
        + Add<T, Output = T>
        + Mul<T, Output = T>
        + Zero
        + One
        + ConjugateValue
        + strided_kernel::MaybeSendSync,
{
    let space = TensorMapSpace::<2, 0>::from_dims([2, 2], []).unwrap();
    let src = TensorMap::<T, 2, 0>::from_vec(values.clone(), space.clone()).unwrap();

    let mut assign_dst = TensorMap::<T, 2, 0>::filled(fill, space.clone()).unwrap();
    tensoradd_assign_into(&mut assign_dst, &src, alpha).unwrap();
    assert_eq!(assign_dst.data(), assign_expected.as_slice());

    let mut add_dst = TensorMap::<T, 2, 0>::filled(fill, space).unwrap();
    tensoradd_add_into(&mut add_dst, &src, alpha).unwrap();
    assert_eq!(add_dst.data(), add_expected.as_slice());
}

fn assert_tensoradd_general_dtype<T>(values: Vec<T>, fill: T, alpha: T, beta: T, expected: Vec<T>)
where
    T: Copy
        + Clone
        + Debug
        + PartialEq
        + Add<T, Output = T>
        + Mul<T, Output = T>
        + Zero
        + One
        + ConjugateValue
        + strided_kernel::MaybeSendSync,
{
    let space = TensorMapSpace::<2, 0>::from_dims([2, 2], []).unwrap();
    let src = TensorMap::<T, 2, 0>::from_vec(values, space.clone()).unwrap();
    let mut dst = TensorMap::<T, 2, 0>::filled(fill, space).unwrap();

    tensoradd_into(&mut dst, &src, OutputAxisOrder::identity(), alpha, beta).unwrap();

    assert_eq!(dst.data(), expected.as_slice());
}

fn assert_tensoradd_permuted_general_dtype<T>(
    values: Vec<T>,
    fill: T,
    alpha: T,
    beta: T,
    expected: Vec<T>,
) where
    T: Copy
        + Clone
        + Debug
        + PartialEq
        + Add<T, Output = T>
        + Mul<T, Output = T>
        + Zero
        + One
        + ConjugateValue
        + strided_kernel::MaybeSendSync,
{
    let src_space = TensorMapSpace::<2, 0>::from_dims([2, 3], []).unwrap();
    let dst_space = TensorMapSpace::<2, 0>::from_dims([3, 2], []).unwrap();
    let src = TensorMap::<T, 2, 0>::from_vec(values, src_space).unwrap();
    let mut dst = TensorMap::<T, 2, 0>::filled(fill, dst_space).unwrap();

    tensoradd_into(
        &mut dst,
        &src,
        OutputAxisOrder::from_axes(&[1, 0]),
        alpha,
        beta,
    )
    .unwrap();

    assert_eq!(dst.data(), expected.as_slice());
}

fn assert_tree_single_dtype<T>(
    values: Vec<T>,
    fill: T,
    coefficient: T,
    alpha: T,
    beta: T,
    expected: Vec<T>,
) where
    T: Copy
        + Clone
        + Debug
        + PartialEq
        + Add<T, Output = T>
        + Mul<T, Output = T>
        + Zero
        + One
        + ConjugateValue
        + strided_kernel::MaybeSendSync
        + RecouplingCoefficientAction<T>,
{
    let space = TensorMapSpace::<2, 0>::from_dims([2, 2], []).unwrap();
    let src = TensorMap::<T, 2, 0>::from_vec(values, space.clone()).unwrap();
    let mut dst = TensorMap::<T, 2, 0>::filled(fill, space).unwrap();
    let structure = TreeTransformStructure::compile(
        &dst,
        &src,
        &[TreeTransformBlockSpec::single(0, 0, coefficient)],
    )
    .unwrap();
    let mut backend = HostTensorOperations;
    let mut workspace = TreeTransformWorkspace::default();

    tree_transform_execute_with(
        &mut backend,
        &mut workspace,
        &structure,
        &mut dst,
        &src,
        alpha,
        beta,
    )
    .unwrap();

    assert_eq!(dst.data(), expected.as_slice());
}

fn assert_tree_multi_dtype<T>(coefficients: Vec<T>, alpha: T, beta: T, fill: T, expected: Vec<T>)
where
    T: Copy
        + Clone
        + Debug
        + PartialEq
        + Add<T, Output = T>
        + Mul<T, Output = T>
        + Zero
        + One
        + ConjugateValue
        + strided_kernel::MaybeSendSync
        + RecouplingCoefficientAction<T>,
{
    let space = TensorMapSpace::<2, 0>::from_dims([4, 2], []).unwrap();
    let src_structure = BlockStructure::packed_column_major(2, [vec![2, 2], vec![2, 2]]).unwrap();
    let dst_structure = BlockStructure::packed_column_major(2, [vec![4, 1], vec![4, 1]]).unwrap();
    let src = TensorMap::<T, 2, 0>::from_vec_with_structure(
        vec![
            T::one(),
            T::one() + T::one(),
            T::one() + T::one() + T::one(),
            T::one() + T::one() + T::one() + T::one(),
            T::one() + T::one() + T::one() + T::one() + T::one(),
            T::one() + T::one() + T::one() + T::one() + T::one() + T::one(),
            T::one() + T::one() + T::one() + T::one() + T::one() + T::one() + T::one(),
            T::one() + T::one() + T::one() + T::one() + T::one() + T::one() + T::one() + T::one(),
        ],
        space.clone(),
        src_structure,
    )
    .unwrap();
    let mut dst =
        TensorMap::<T, 2, 0>::from_vec_with_structure(vec![fill; 8], space, dst_structure).unwrap();
    let structure = TreeTransformStructure::compile(
        &dst,
        &src,
        &[TreeTransformBlockSpec::multi(
            vec![0, 1],
            vec![0, 1],
            coefficients,
        )],
    )
    .unwrap();
    let mut backend = HostTensorOperations;
    let mut workspace = TreeTransformWorkspace::default();

    tree_transform_execute_with(
        &mut backend,
        &mut workspace,
        &structure,
        &mut dst,
        &src,
        alpha,
        beta,
    )
    .unwrap();

    assert_eq!(dst.data(), expected.as_slice());
    assert_eq!(workspace.source_len(), 8);
    assert_eq!(workspace.destination_len(), 8);
}

fn assert_tree_single_mixed_dtype<D, C>(
    values: Vec<D>,
    fill: D,
    coefficient: C,
    alpha: D,
    beta: D,
    expected: Vec<D>,
) where
    D: TreeTransformScalar + RecouplingCoefficientAction<C> + Clone + Debug,
    C: Copy,
{
    let space = TensorMapSpace::<2, 0>::from_dims([2, 2], []).unwrap();
    let src = TensorMap::<D, 2, 0>::from_vec(values, space.clone()).unwrap();
    let mut dst = TensorMap::<D, 2, 0>::filled(fill, space).unwrap();
    let structure = TreeTransformStructure::compile(
        &dst,
        &src,
        &[TreeTransformBlockSpec::single(0, 0, coefficient)],
    )
    .unwrap();
    let mut backend = HostTensorOperations;
    let mut workspace = TreeTransformWorkspace::default();

    tree_transform_execute_with(
        &mut backend,
        &mut workspace,
        &structure,
        &mut dst,
        &src,
        alpha,
        beta,
    )
    .unwrap();

    assert_eq!(dst.data(), expected.as_slice());
}

fn assert_tree_multi_mixed_dtype<D, C>(
    src_values: Vec<D>,
    recoupling_coefficients_dst_src: Vec<C>,
    alpha: D,
    beta: D,
    fill: D,
    expected: Vec<D>,
) where
    D: TreeTransformScalar + RecouplingCoefficientAction<C> + Clone + Debug,
    C: Copy,
{
    let space = TensorMapSpace::<2, 0>::from_dims([4, 2], []).unwrap();
    let src_structure = BlockStructure::packed_column_major(2, [vec![2, 2], vec![2, 2]]).unwrap();
    let dst_structure = BlockStructure::packed_column_major(2, [vec![4, 1], vec![4, 1]]).unwrap();
    let src =
        TensorMap::<D, 2, 0>::from_vec_with_structure(src_values, space.clone(), src_structure)
            .unwrap();
    let mut dst =
        TensorMap::<D, 2, 0>::from_vec_with_structure(vec![fill; 8], space, dst_structure).unwrap();
    let structure = TreeTransformStructure::compile(
        &dst,
        &src,
        &[TreeTransformBlockSpec::multi(
            vec![0, 1],
            vec![0, 1],
            recoupling_coefficients_dst_src,
        )],
    )
    .unwrap();
    let mut backend = HostTensorOperations;
    let mut workspace = TreeTransformWorkspace::default();

    tree_transform_execute_with(
        &mut backend,
        &mut workspace,
        &structure,
        &mut dst,
        &src,
        alpha,
        beta,
    )
    .unwrap();

    assert_eq!(dst.data(), expected.as_slice());
    assert_eq!(workspace.source_len(), 8);
    assert_eq!(workspace.destination_len(), 8);
}

fn assert_tree_multi_tensorkit_orientation_dtype<T>(
    src_values: Vec<T>,
    recoupling_coefficients_dst_src: Vec<T>,
    alpha: T,
    beta: T,
    fill: T,
    expected: Vec<T>,
) where
    T: Copy
        + Clone
        + Debug
        + PartialEq
        + Add<T, Output = T>
        + Mul<T, Output = T>
        + Zero
        + One
        + ConjugateValue
        + strided_kernel::MaybeSendSync
        + RecouplingCoefficientAction<T>,
{
    let src_space = TensorMapSpace::<2, 0>::from_dims([6, 1], []).unwrap();
    let dst_space = TensorMapSpace::<2, 0>::from_dims([4, 1], []).unwrap();
    let src_structure =
        BlockStructure::packed_column_major(2, [vec![2, 1], vec![2, 1], vec![2, 1]]).unwrap();
    let dst_structure = BlockStructure::packed_column_major(2, [vec![2, 1], vec![2, 1]]).unwrap();
    let src = TensorMap::<T, 2, 0>::from_vec_with_structure(src_values, src_space, src_structure)
        .unwrap();
    let mut dst =
        TensorMap::<T, 2, 0>::from_vec_with_structure(vec![fill; 4], dst_space, dst_structure)
            .unwrap();
    let structure = TreeTransformStructure::compile(
        &dst,
        &src,
        &[TreeTransformBlockSpec::multi(
            vec![0, 1],
            vec![0, 1, 2],
            recoupling_coefficients_dst_src,
        )],
    )
    .unwrap();
    let mut backend = HostTensorOperations;
    let mut workspace = TreeTransformWorkspace::default();

    tree_transform_execute_with(
        &mut backend,
        &mut workspace,
        &structure,
        &mut dst,
        &src,
        alpha,
        beta,
    )
    .unwrap();

    assert_eq!(dst.data(), expected.as_slice());
    assert_eq!(workspace.source_len(), 6);
    assert_eq!(workspace.destination_len(), 4);
}

fn assert_tree_multi_tensorkit_orientation_dense_dtype<T>(
    src_values: Vec<T>,
    recoupling_coefficients_dst_src: Vec<T>,
    alpha: T,
    beta: T,
    fill: T,
    expected: Vec<T>,
) where
    T: DenseRecouplingScalar + Clone + Debug,
{
    let src_space = TensorMapSpace::<2, 0>::from_dims([6, 1], []).unwrap();
    let dst_space = TensorMapSpace::<2, 0>::from_dims([4, 1], []).unwrap();
    let src_structure =
        BlockStructure::packed_column_major(2, [vec![2, 1], vec![2, 1], vec![2, 1]]).unwrap();
    let dst_structure = BlockStructure::packed_column_major(2, [vec![2, 1], vec![2, 1]]).unwrap();
    let src = TensorMap::<T, 2, 0>::from_vec_with_structure(src_values, src_space, src_structure)
        .unwrap();
    let mut dst =
        TensorMap::<T, 2, 0>::from_vec_with_structure(vec![fill; 4], dst_space, dst_structure)
            .unwrap();
    let structure = TreeTransformStructure::compile(
        &dst,
        &src,
        &[TreeTransformBlockSpec::multi(
            vec![0, 1],
            vec![0, 1, 2],
            recoupling_coefficients_dst_src,
        )],
    )
    .unwrap();
    let mut backend = DenseTreeTransformOperations::default();
    let mut workspace = TreeTransformWorkspace::default();

    tree_transform_execute_with(
        &mut backend,
        &mut workspace,
        &structure,
        &mut dst,
        &src,
        alpha,
        beta,
    )
    .unwrap();

    assert_eq!(dst.data(), expected.as_slice());
    assert_eq!(workspace.source_len(), 6);
    assert_eq!(workspace.destination_len(), 4);
}

fn assert_tree_multi_keyed_dtype<T>(
    src_values: Vec<T>,
    recoupling_coefficients_dst_src: Vec<T>,
    expected: Vec<T>,
) where
    T: Copy
        + Clone
        + Debug
        + PartialEq
        + Add<T, Output = T>
        + Mul<T, Output = T>
        + Zero
        + One
        + ConjugateValue
        + strided_kernel::MaybeSendSync
        + RecouplingCoefficientAction<T>,
{
    let key10 = BlockKey::sector_ids([10]);
    let key20 = BlockKey::sector_ids([20]);
    let key100 = BlockKey::sector_ids([100]);
    let key200 = BlockKey::sector_ids([200]);
    let key300 = BlockKey::sector_ids([300]);
    let src_space = TensorMapSpace::<2, 0>::from_dims([6, 1], []).unwrap();
    let dst_space = TensorMapSpace::<2, 0>::from_dims([4, 1], []).unwrap();
    let src_structure = packed_fixture_structure(
        2,
        [
            (key100.clone(), vec![2, 1]),
            (key300.clone(), vec![2, 1]),
            (key200.clone(), vec![2, 1]),
        ],
    )
    .unwrap();
    let dst_structure = packed_fixture_structure(
        2,
        [(key20.clone(), vec![2, 1]), (key10.clone(), vec![2, 1])],
    )
    .unwrap();
    let src = TensorMap::<T, 2, 0>::from_vec_with_structure(src_values, src_space, src_structure)
        .unwrap();
    let mut dst =
        TensorMap::<T, 2, 0>::from_vec_with_structure(vec![T::zero(); 4], dst_space, dst_structure)
            .unwrap();
    let structure = TreeTransformStructure::compile_keyed(
        &dst,
        &src,
        &[TreeTransformKeyBlockSpec::multi(
            vec![key10, key20],
            vec![key100, key200, key300],
            recoupling_coefficients_dst_src,
        )],
    )
    .unwrap();
    let mut backend = HostTensorOperations;
    let mut workspace = TreeTransformWorkspace::default();

    tree_transform_execute_with(
        &mut backend,
        &mut workspace,
        &structure,
        &mut dst,
        &src,
        T::one(),
        T::zero(),
    )
    .unwrap();

    assert_eq!(structure.block_count(), 1);
    assert_eq!(dst.data(), expected.as_slice());
    assert_eq!(workspace.source_len(), 6);
    assert_eq!(workspace.destination_len(), 4);
}

fn assert_tensorcontract_scalar_dtype<D>(lhs_value: D, rhs_value: D, fill: D, expected: D)
where
    D: DenseBlockScalar + RecouplingCoefficientAction<f64> + Clone + Debug,
{
    let lhs_space = TensorMapSpace::<2, 0>::from_dims([1, 1], []).unwrap();
    let rhs_space = TensorMapSpace::<2, 0>::from_dims([1, 1], []).unwrap();
    let dst_space = TensorMapSpace::<2, 0>::from_dims([1, 1], []).unwrap();
    let lhs = TensorMap::<D, 2, 0>::from_vec(vec![lhs_value], lhs_space).unwrap();
    let rhs = TensorMap::<D, 2, 0>::from_vec(vec![rhs_value], rhs_space).unwrap();
    let mut dst = TensorMap::<D, 2, 0>::filled(fill, dst_space).unwrap();

    tensorcontract_into(
        &mut dst,
        &lhs,
        &rhs,
        TensorContractSpec::with_default_output_order(&[1], &[0]),
        D::one() + D::one(),
        D::one() + D::one() + D::one(),
    )
    .unwrap();

    assert_eq!(dst.data(), &[expected]);
}

fn assert_weighted_tensorcontract_scalar_dtype<D>(lhs_value: D, rhs_value: D, fill: D, expected: D)
where
    D: DenseBlockScalar + RecouplingCoefficientAction<f64> + Clone + Debug,
{
    let lhs_space = TensorMapSpace::<2, 0>::from_dims([1, 1], []).unwrap();
    let rhs_space = TensorMapSpace::<2, 0>::from_dims([1, 1], []).unwrap();
    let dst_space = TensorMapSpace::<2, 0>::from_dims([1, 1], []).unwrap();
    let lhs = TensorMap::<D, 2, 0>::from_vec(vec![lhs_value], lhs_space).unwrap();
    let rhs = TensorMap::<D, 2, 0>::from_vec(vec![rhs_value], rhs_space).unwrap();
    let mut dst = TensorMap::<D, 2, 0>::from_vec(vec![fill], dst_space).unwrap();
    let structure = TensorContractStructure::compile_with_block_specs(
        &dst,
        &lhs,
        &rhs,
        TensorContractSpec::with_default_output_order(&[1], &[0]),
        &[TensorContractBlockSpec::with_coefficient(0, 0, 0, 0.5)],
    )
    .unwrap();

    tensorcontract_execute_with(
        &mut DenseTreeTransformOperations::default_executor(),
        &mut TensorContractWorkspace::default(),
        &structure,
        &mut dst,
        &lhs,
        &rhs,
        D::one() + D::one(),
        D::one() + D::one() + D::one(),
    )
    .unwrap();

    assert_eq!(dst.data(), &[expected]);
}

#[derive(Default)]
struct CountingDenseExecutor {
    dot_general_into_calls: usize,
}

impl DenseExecutor for CountingDenseExecutor {
    fn svd(&mut self, _input: DenseRead<'_>) -> Result<Vec<tenet_dense::DenseTensor>, DenseError> {
        unreachable!("tree transform does not call svd")
    }

    fn qr(&mut self, _input: DenseRead<'_>) -> Result<Vec<tenet_dense::DenseTensor>, DenseError> {
        unreachable!("tree transform does not call qr")
    }

    fn eigh(&mut self, _input: DenseRead<'_>) -> Result<Vec<tenet_dense::DenseTensor>, DenseError> {
        unreachable!("tree transform does not call eigh")
    }

    fn dot_general_into(
        &mut self,
        output: DenseWrite<'_>,
        lhs: DenseRead<'_>,
        rhs: DenseRead<'_>,
        config: &tenet_dense::DenseDotConfig,
    ) -> Result<(), DenseError> {
        self.dot_general_into_calls += 1;
        assert_eq!(config, &tenet_dense::DenseDotConfig::matmul());

        // This mock pins the TensorKit-style `mul!` boundary only:
        // `buffer_src :: (blocksize, n_src)` times `U^T :: (n_src, n_dst)`
        // into `buffer_dst :: (blocksize, n_dst)`. Numerical GEMM behavior
        // is covered by the DefaultDenseExecutor test.
        let (mut output, lhs, rhs) = match (output, lhs, rhs) {
            (DenseWrite::F64(output), DenseRead::F64(lhs), DenseRead::F64(rhs)) => {
                (output, lhs, rhs)
            }
            _ => panic!("counting executor only covers f64 recoupling"),
        };

        assert_eq!(lhs.shape(), &[2, 3]);
        assert_eq!(lhs.strides(), &[1, 2]);
        assert_eq!(lhs.offset(), 0);
        assert_eq!(lhs.data(), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);

        assert_eq!(rhs.shape(), &[3, 2]);
        assert_eq!(rhs.strides(), &[1, 3]);
        assert_eq!(rhs.offset(), 0);
        assert_eq!(rhs.data(), &[10.0, 100.0, 1000.0, 20.0, 200.0, 2000.0]);

        assert_eq!(output.shape(), &[2, 2]);
        assert_eq!(output.strides(), &[1, 2]);
        assert_eq!(output.offset(), 0);

        let out_strides = output.strides().to_vec();
        let out_offset = output.offset();
        let out_data = output.data_mut();
        out_data[out_offset] = 5310.0;
        out_data[out_offset + out_strides[0]] = 6420.0;
        out_data[out_offset + out_strides[1]] = 10620.0;
        out_data[out_offset + out_strides[0] + out_strides[1]] = 12840.0;
        Ok(())
    }
}

#[derive(Default)]
struct ContractLayoutCheckingDenseExecutor {
    dot_general_into_calls: usize,
}

impl DenseExecutor for ContractLayoutCheckingDenseExecutor {
    fn svd(&mut self, _input: DenseRead<'_>) -> Result<Vec<tenet_dense::DenseTensor>, DenseError> {
        unreachable!("tensor contraction does not call svd")
    }

    fn qr(&mut self, _input: DenseRead<'_>) -> Result<Vec<tenet_dense::DenseTensor>, DenseError> {
        unreachable!("tensor contraction does not call qr")
    }

    fn eigh(&mut self, _input: DenseRead<'_>) -> Result<Vec<tenet_dense::DenseTensor>, DenseError> {
        unreachable!("tensor contraction does not call eigh")
    }

    fn dot_general_into(
        &mut self,
        output: DenseWrite<'_>,
        lhs: DenseRead<'_>,
        rhs: DenseRead<'_>,
        config: &DenseDotConfig,
    ) -> Result<(), DenseError> {
        self.dot_general_into_calls += 1;
        assert_eq!(
            config,
            &DenseDotConfig::new(vec![1], vec![1], Vec::new(), Vec::new())
        );
        let (mut output, lhs, rhs) = match (output, lhs, rhs) {
            (DenseWrite::F64(output), DenseRead::F64(lhs), DenseRead::F64(rhs)) => {
                (output, lhs, rhs)
            }
            _ => panic!("layout-checking executor only covers f64"),
        };

        if lhs.shape() == [2, 3] {
            assert_eq!(lhs.strides(), &[1, 2]);
            assert_eq!(lhs.offset(), 0);
            assert_eq!(rhs.shape(), &[4, 3]);
            assert_eq!(rhs.strides(), &[1, 4]);
            assert_eq!(rhs.offset(), 0);
            assert_eq!(output.shape(), &[2, 4]);
            assert_eq!(output.strides(), &[1, 2]);
            assert_eq!(output.offset(), 0);
            output
                .data_mut()
                .copy_from_slice(&[115.0, 148.0, 124.0, 160.0, 133.0, 172.0, 142.0, 184.0]);
        } else {
            assert_eq!(lhs.shape(), &[4, 3]);
            assert_eq!(lhs.strides(), &[1, 4]);
            assert_eq!(lhs.offset(), 0);
            assert_eq!(rhs.shape(), &[2, 3]);
            assert_eq!(rhs.strides(), &[1, 2]);
            assert_eq!(rhs.offset(), 0);
            assert_eq!(output.shape(), &[4, 2]);
            assert_eq!(output.strides(), &[1, 4]);
            assert_eq!(output.offset(), 0);
            output
                .data_mut()
                .copy_from_slice(&[115.0, 124.0, 133.0, 142.0, 148.0, 160.0, 172.0, 184.0]);
        }
        Ok(())
    }
}

#[derive(Default)]
struct PanicDenseExecutor;

impl DenseExecutor for PanicDenseExecutor {
    fn svd(&mut self, _input: DenseRead<'_>) -> Result<Vec<tenet_dense::DenseTensor>, DenseError> {
        unreachable!("tensor contraction does not call svd")
    }

    fn qr(&mut self, _input: DenseRead<'_>) -> Result<Vec<tenet_dense::DenseTensor>, DenseError> {
        unreachable!("tensor contraction does not call qr")
    }

    fn eigh(&mut self, _input: DenseRead<'_>) -> Result<Vec<tenet_dense::DenseTensor>, DenseError> {
        unreachable!("tensor contraction does not call eigh")
    }

    fn dot_general_into(
        &mut self,
        _output: DenseWrite<'_>,
        _lhs: DenseRead<'_>,
        _rhs: DenseRead<'_>,
        _config: &DenseDotConfig,
    ) -> Result<(), DenseError> {
        panic!("replay structure mismatch must be rejected before dense execution")
    }
}

mod copy;

mod contract_dense;

mod contract_fusion;

mod tensoradd;

mod tensortrace;

mod tree_transform_exec;

mod tree_transform_plan;
