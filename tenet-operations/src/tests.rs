use super::*;
use num_complex::{Complex32, Complex64};
use std::fmt::Debug;
use tenet_core::{
    BlockSpec, BraidingStyleKind, FermionParityFusionRule, FusionProductSpace,
    FusionTensorMapSpace, FusionTreeHomSpace, FusionTreeKey, MultiplicityFreeFusionRule,
    MultiplicityFreeFusionSymbols, ProductFusionRule, SU2FusionRule, SU2Irrep, SectorId, SectorLeg,
    TensorMapSpace, U1FusionRule, U1Irrep, Z2FusionRule,
};
use tenet_dense::DenseError;

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
    FusionTreeKey::new(
        Vec::<SectorId>::new(),
        coupled.map(SectorId::new),
        Vec::<bool>::new(),
        Vec::<SectorId>::new(),
        Vec::<SectorId>::new(),
    )
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
        FusionTreeKey::from_sector_ids(
            codomain,
            coupled,
            codomain_is_dual,
            codomain_innerlines,
            codomain_vertices,
        ),
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
        FusionProductSpace::new([SectorLeg::new([a], false), SectorLeg::new([b], false)]),
        FusionProductSpace::new([SectorLeg::new([c0, c1], false)]),
    );
    let dst_hom = FusionTreeHomSpace::new(
        FusionProductSpace::new([SectorLeg::new([b], false), SectorLeg::new([a], false)]),
        FusionProductSpace::new([SectorLeg::new([c0, c1], false)]),
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
        assert_eq!(spec.coefficients_src_by_dst().len(), 1);
        let dst_coupled = expect_tree_key(&spec.dst_keys()[0]).coupled().unwrap();
        if dst_coupled == coupled {
            assert!(found.is_none(), "duplicate coefficient for {coupled:?}");
            found = Some(spec.coefficients_src_by_dst()[0]);
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
        assert_eq!(spec.coefficients_src_by_dst().len(), 1);
        let src_key = &spec.src_keys()[0];
        let dst_key = &spec.dst_keys()[0];
        let src_offset = src_structure.block_by_key(src_key).unwrap().offset();
        let dst_offset = dst_structure.block_by_key(dst_key).unwrap().offset();
        expected[dst_offset] += alpha * spec.coefficients_src_by_dst()[0] * src_data[src_offset];
    }
    expected
}

fn column_major_structure_like(structure: &BlockStructure, shape: Vec<usize>) -> BlockStructure {
    let blocks = (0..structure.block_count())
        .map(|index| (structure.block(index).unwrap().key().clone(), shape.clone()));
    BlockStructure::packed_column_major_with_keys(structure.rank(), blocks).unwrap()
}

#[derive(Clone, Copy, Debug)]
struct UniqueZ2Rule;

impl FusionRule for UniqueZ2Rule {
    fn fusion_style(&self) -> FusionStyleKind {
        FusionStyleKind::Unique
    }

    fn braiding_style(&self) -> BraidingStyleKind {
        BraidingStyleKind::Bosonic
    }

    fn vacuum(&self) -> SectorId {
        SectorId::new(0)
    }

    fn fusion_channels(&self, left: SectorId, right: SectorId) -> Vec<SectorId> {
        vec![SectorId::new((left.id() + right.id()) % 2)]
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
    fn fusion_style(&self) -> FusionStyleKind {
        FusionStyleKind::Unique
    }

    fn braiding_style(&self) -> BraidingStyleKind {
        BraidingStyleKind::Anyonic
    }

    fn vacuum(&self) -> SectorId {
        SectorId::new(0)
    }

    fn fusion_channels(&self, left: SectorId, right: SectorId) -> Vec<SectorId> {
        vec![SectorId::new((left.id() + right.id()) % 2)]
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

#[derive(Clone, Copy, Debug)]
struct UniquePlanarRule;

impl FusionRule for UniquePlanarRule {
    fn fusion_style(&self) -> FusionStyleKind {
        FusionStyleKind::Unique
    }

    fn braiding_style(&self) -> BraidingStyleKind {
        BraidingStyleKind::NoBraiding
    }

    fn vacuum(&self) -> SectorId {
        SectorId::new(0)
    }

    fn fusion_channels(&self, left: SectorId, right: SectorId) -> Vec<SectorId> {
        vec![SectorId::new((left.id() + right.id()) % 2)]
    }
}

impl MultiplicityFreeFusionRule for UniquePlanarRule {}

#[derive(Clone, Copy, Debug)]
struct SimpleSu2Rule;

impl FusionRule for SimpleSu2Rule {
    fn fusion_style(&self) -> FusionStyleKind {
        FusionStyleKind::Simple
    }

    fn braiding_style(&self) -> BraidingStyleKind {
        BraidingStyleKind::Bosonic
    }

    fn vacuum(&self) -> SectorId {
        SectorId::new(0)
    }

    fn fusion_channels(&self, left: SectorId, right: SectorId) -> Vec<SectorId> {
        let min = left.id().abs_diff(right.id());
        let max = left.id() + right.id();
        (min..=max).step_by(2).map(SectorId::new).collect()
    }
}

impl MultiplicityFreeFusionRule for SimpleSu2Rule {}

#[derive(Clone, Copy, Debug)]
struct GenericMultiplicityRule;

impl FusionRule for GenericMultiplicityRule {
    fn fusion_style(&self) -> FusionStyleKind {
        FusionStyleKind::Generic
    }

    fn braiding_style(&self) -> BraidingStyleKind {
        BraidingStyleKind::Anyonic
    }

    fn vacuum(&self) -> SectorId {
        SectorId::new(0)
    }

    fn fusion_channels(&self, left: SectorId, right: SectorId) -> Vec<SectorId> {
        match (left.id(), right.id()) {
            (1, 1) => vec![SectorId::new(0), SectorId::new(1)],
            (0, x) | (x, 0) => vec![SectorId::new(x)],
            _ => Vec::new(),
        }
    }

    fn nsymbol(&self, left: SectorId, right: SectorId, coupled: SectorId) -> usize {
        match (left.id(), right.id(), coupled.id()) {
            (1, 1, 1) => 2,
            _ => usize::from(self.fusion_channels(left, right).contains(&coupled)),
        }
    }
}

#[test]
fn copy_into_uses_strided_kernel_for_transposed_views() {
    let src_data = [1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0];
    let src_shape = [3, 2];
    let src_strides = [2, 1];
    let dst_shape = [3, 2];
    let dst_strides = [1, 3];
    let mut dst_data = [0.0_f64; 6];

    let src = BlockView::new(&src_data, &src_shape, &src_strides, 0).unwrap();
    let dst = BlockViewMut::new(&mut dst_data, &dst_shape, &dst_strides, 0).unwrap();
    copy_into(dst, src).unwrap();

    assert_eq!(dst_data, [1.0, 3.0, 5.0, 2.0, 4.0, 6.0]);
}

#[test]
fn scaled_assign_into_uses_strided_kernel() {
    let src_data = [1.0_f64, 2.0, 3.0, 4.0];
    let shape = [2, 2];
    let src_strides = [2, 1];
    let dst_strides = [1, 2];
    let mut dst_data = [0.0_f64; 4];

    let src = BlockView::new(&src_data, &shape, &src_strides, 0).unwrap();
    let dst = BlockViewMut::new(&mut dst_data, &shape, &dst_strides, 0).unwrap();
    scaled_assign_into(dst, src, 2.0).unwrap();

    assert_eq!(dst_data, [2.0, 6.0, 4.0, 8.0]);
}

#[test]
fn scaled_add_into_uses_strided_kernel() {
    let src_data = [1.0_f64, 2.0, 3.0, 4.0];
    let shape = [2, 2];
    let strides = [1, 2];
    let mut dst_data = [10.0_f64, 20.0, 30.0, 40.0];

    let src = BlockView::new(&src_data, &shape, &strides, 0).unwrap();
    let dst = BlockViewMut::new(&mut dst_data, &shape, &strides, 0).unwrap();
    scaled_add_into(dst, src, 3.0).unwrap();

    assert_eq!(dst_data, [13.0, 26.0, 39.0, 52.0]);
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
        + strided_kernel::MaybeSendSync,
{
    let space = TensorMapSpace::<2, 0>::from_dims([2, 2], []).unwrap();
    let src = TensorMap::<T, 2, 0>::from_vec(values, space.clone()).unwrap();
    let mut dst = TensorMap::<T, 2, 0>::filled(fill, space).unwrap();

    tensoradd_into(&mut dst, &src, AxisPermutation::identity(), alpha, beta).unwrap();

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
    coefficients_src_by_dst: Vec<C>,
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
            coefficients_src_by_dst,
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
    coefficients_src_by_dst: Vec<T>,
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
            coefficients_src_by_dst,
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
    coefficients_src_by_dst: Vec<T>,
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
            coefficients_src_by_dst,
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
    coefficients_src_by_dst: Vec<T>,
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
    let src_structure = BlockStructure::packed_column_major_with_keys(
        2,
        [
            (key100.clone(), vec![2, 1]),
            (key300.clone(), vec![2, 1]),
            (key200.clone(), vec![2, 1]),
        ],
    )
    .unwrap();
    let dst_structure = BlockStructure::packed_column_major_with_keys(
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
            coefficients_src_by_dst,
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

#[test]
fn tensorcopy_supports_all_storage_dtypes() {
    assert_tensorcopy_dtype(vec![1.0_f32, 2.0, 3.0, 4.0], 0.0);
    assert_tensorcopy_dtype(vec![1.0_f64, 2.0, 3.0, 4.0], 0.0);
    assert_tensorcopy_dtype(vec![1_i32, 2, 3, 4], 0);
    assert_tensorcopy_dtype(vec![1_i64, 2, 3, 4], 0);
    assert_tensorcopy_dtype(vec![true, false, true, false], false);
    assert_tensorcopy_dtype(
        vec![
            Complex32::new(1.0, 1.0),
            Complex32::new(2.0, -1.0),
            Complex32::new(3.0, 0.5),
            Complex32::new(4.0, -0.5),
        ],
        Complex32::new(0.0, 0.0),
    );
    assert_tensorcopy_dtype(
        vec![
            Complex64::new(1.0, 1.0),
            Complex64::new(2.0, -1.0),
            Complex64::new(3.0, 0.5),
            Complex64::new(4.0, -0.5),
        ],
        Complex64::new(0.0, 0.0),
    );
}

#[test]
fn tensoradd_assign_and_add_support_all_numeric_dtypes() {
    assert_tensoradd_dtype(
        vec![1.0_f32, 2.0, 3.0, 4.0],
        10.0,
        2.0,
        vec![2.0, 4.0, 6.0, 8.0],
        vec![12.0, 14.0, 16.0, 18.0],
    );
    assert_tensoradd_dtype(
        vec![1.0_f64, 2.0, 3.0, 4.0],
        10.0,
        2.0,
        vec![2.0, 4.0, 6.0, 8.0],
        vec![12.0, 14.0, 16.0, 18.0],
    );
    assert_tensoradd_dtype(
        vec![1_i32, 2, 3, 4],
        10,
        2,
        vec![2, 4, 6, 8],
        vec![12, 14, 16, 18],
    );
    assert_tensoradd_dtype(
        vec![1_i64, 2, 3, 4],
        10,
        2,
        vec![2, 4, 6, 8],
        vec![12, 14, 16, 18],
    );
    assert_tensoradd_dtype(
        vec![
            Complex32::new(1.0, 1.0),
            Complex32::new(2.0, -1.0),
            Complex32::new(3.0, 0.5),
            Complex32::new(4.0, -0.5),
        ],
        Complex32::new(10.0, 0.0),
        Complex32::new(2.0, 0.0),
        vec![
            Complex32::new(2.0, 2.0),
            Complex32::new(4.0, -2.0),
            Complex32::new(6.0, 1.0),
            Complex32::new(8.0, -1.0),
        ],
        vec![
            Complex32::new(12.0, 2.0),
            Complex32::new(14.0, -2.0),
            Complex32::new(16.0, 1.0),
            Complex32::new(18.0, -1.0),
        ],
    );
    assert_tensoradd_dtype(
        vec![
            Complex64::new(1.0, 1.0),
            Complex64::new(2.0, -1.0),
            Complex64::new(3.0, 0.5),
            Complex64::new(4.0, -0.5),
        ],
        Complex64::new(10.0, 0.0),
        Complex64::new(2.0, 0.0),
        vec![
            Complex64::new(2.0, 2.0),
            Complex64::new(4.0, -2.0),
            Complex64::new(6.0, 1.0),
            Complex64::new(8.0, -1.0),
        ],
        vec![
            Complex64::new(12.0, 2.0),
            Complex64::new(14.0, -2.0),
            Complex64::new(16.0, 1.0),
            Complex64::new(18.0, -1.0),
        ],
    );
}

#[test]
fn tensoradd_general_beta_supports_all_numeric_dtypes() {
    assert_tensoradd_general_dtype(
        vec![1.0_f32, 2.0, 3.0, 4.0],
        10.0,
        2.0,
        3.0,
        vec![32.0, 34.0, 36.0, 38.0],
    );
    assert_tensoradd_general_dtype(
        vec![1.0_f64, 2.0, 3.0, 4.0],
        10.0,
        2.0,
        3.0,
        vec![32.0, 34.0, 36.0, 38.0],
    );
    assert_tensoradd_general_dtype(vec![1_i32, 2, 3, 4], 10, 2, 3, vec![32, 34, 36, 38]);
    assert_tensoradd_general_dtype(vec![1_i64, 2, 3, 4], 10, 2, 3, vec![32, 34, 36, 38]);
    assert_tensoradd_general_dtype(
        vec![
            Complex32::new(1.0, 1.0),
            Complex32::new(2.0, -1.0),
            Complex32::new(3.0, 0.5),
            Complex32::new(4.0, -0.5),
        ],
        Complex32::new(10.0, 1.0),
        Complex32::new(2.0, 0.0),
        Complex32::new(3.0, 0.0),
        vec![
            Complex32::new(32.0, 5.0),
            Complex32::new(34.0, 1.0),
            Complex32::new(36.0, 4.0),
            Complex32::new(38.0, 2.0),
        ],
    );
    assert_tensoradd_general_dtype(
        vec![
            Complex64::new(1.0, 1.0),
            Complex64::new(2.0, -1.0),
            Complex64::new(3.0, 0.5),
            Complex64::new(4.0, -0.5),
        ],
        Complex64::new(10.0, 1.0),
        Complex64::new(2.0, 0.0),
        Complex64::new(3.0, 0.0),
        vec![
            Complex64::new(32.0, 5.0),
            Complex64::new(34.0, 1.0),
            Complex64::new(36.0, 4.0),
            Complex64::new(38.0, 2.0),
        ],
    );
}

#[test]
fn tree_transform_single_replay_supports_all_numeric_dtypes() {
    assert_tree_single_dtype(
        vec![1.0_f32, 2.0, 3.0, 4.0],
        10.0,
        3.0,
        2.0,
        4.0,
        vec![46.0, 52.0, 58.0, 64.0],
    );
    assert_tree_single_dtype(
        vec![1.0_f64, 2.0, 3.0, 4.0],
        10.0,
        3.0,
        2.0,
        4.0,
        vec![46.0, 52.0, 58.0, 64.0],
    );
    assert_tree_single_dtype(vec![1_i32, 2, 3, 4], 10, 3, 2, 4, vec![46, 52, 58, 64]);
    assert_tree_single_dtype(vec![1_i64, 2, 3, 4], 10, 3, 2, 4, vec![46, 52, 58, 64]);
    assert_tree_single_dtype(
        vec![
            Complex32::new(1.0, 1.0),
            Complex32::new(2.0, -1.0),
            Complex32::new(3.0, 0.5),
            Complex32::new(4.0, -0.5),
        ],
        Complex32::new(10.0, 1.0),
        Complex32::new(3.0, 0.0),
        Complex32::new(2.0, 0.0),
        Complex32::new(4.0, 0.0),
        vec![
            Complex32::new(46.0, 10.0),
            Complex32::new(52.0, -2.0),
            Complex32::new(58.0, 7.0),
            Complex32::new(64.0, 1.0),
        ],
    );
    assert_tree_single_dtype(
        vec![
            Complex64::new(1.0, 1.0),
            Complex64::new(2.0, -1.0),
            Complex64::new(3.0, 0.5),
            Complex64::new(4.0, -0.5),
        ],
        Complex64::new(10.0, 1.0),
        Complex64::new(3.0, 0.0),
        Complex64::new(2.0, 0.0),
        Complex64::new(4.0, 0.0),
        vec![
            Complex64::new(46.0, 10.0),
            Complex64::new(52.0, -2.0),
            Complex64::new(58.0, 7.0),
            Complex64::new(64.0, 1.0),
        ],
    );
}

#[test]
fn tree_transform_single_replay_supports_complex_data_with_real_coefficients() {
    assert_tree_single_mixed_dtype(
        vec![
            Complex32::new(1.0, 1.0),
            Complex32::new(2.0, -1.0),
            Complex32::new(3.0, 0.5),
            Complex32::new(4.0, -0.5),
        ],
        Complex32::new(10.0, 1.0),
        3.0_f64,
        Complex32::new(2.0, 0.0),
        Complex32::new(4.0, 0.0),
        vec![
            Complex32::new(46.0, 10.0),
            Complex32::new(52.0, -2.0),
            Complex32::new(58.0, 7.0),
            Complex32::new(64.0, 1.0),
        ],
    );
    assert_tree_single_mixed_dtype(
        vec![
            Complex64::new(1.0, 1.0),
            Complex64::new(2.0, -1.0),
            Complex64::new(3.0, 0.5),
            Complex64::new(4.0, -0.5),
        ],
        Complex64::new(10.0, 1.0),
        3.0_f64,
        Complex64::new(2.0, 0.0),
        Complex64::new(4.0, 0.0),
        vec![
            Complex64::new(46.0, 10.0),
            Complex64::new(52.0, -2.0),
            Complex64::new(58.0, 7.0),
            Complex64::new(64.0, 1.0),
        ],
    );
}

#[test]
fn tree_transform_multi_pack_gemm_scatter_supports_all_numeric_dtypes() {
    assert_tree_multi_dtype(
        vec![2.0_f32, 3.0, 5.0, 7.0],
        2.0,
        10.0,
        1.0,
        vec![44.0, 54.0, 64.0, 74.0, 90.0, 114.0, 138.0, 162.0],
    );
    assert_tree_multi_dtype(
        vec![2.0_f64, 3.0, 5.0, 7.0],
        2.0,
        10.0,
        1.0,
        vec![44.0, 54.0, 64.0, 74.0, 90.0, 114.0, 138.0, 162.0],
    );
    assert_tree_multi_dtype(
        vec![2_i32, 3, 5, 7],
        2,
        10,
        1,
        vec![44, 54, 64, 74, 90, 114, 138, 162],
    );
    assert_tree_multi_dtype(
        vec![2_i64, 3, 5, 7],
        2,
        10,
        1,
        vec![44, 54, 64, 74, 90, 114, 138, 162],
    );
    assert_tree_multi_dtype(
        vec![
            Complex32::new(2.0, 0.0),
            Complex32::new(3.0, 0.0),
            Complex32::new(5.0, 0.0),
            Complex32::new(7.0, 0.0),
        ],
        Complex32::new(2.0, 0.0),
        Complex32::new(10.0, 0.0),
        Complex32::new(1.0, 1.0),
        vec![
            Complex32::new(44.0, 10.0),
            Complex32::new(54.0, 10.0),
            Complex32::new(64.0, 10.0),
            Complex32::new(74.0, 10.0),
            Complex32::new(90.0, 10.0),
            Complex32::new(114.0, 10.0),
            Complex32::new(138.0, 10.0),
            Complex32::new(162.0, 10.0),
        ],
    );
    assert_tree_multi_dtype(
        vec![
            Complex64::new(2.0, 0.0),
            Complex64::new(3.0, 0.0),
            Complex64::new(5.0, 0.0),
            Complex64::new(7.0, 0.0),
        ],
        Complex64::new(2.0, 0.0),
        Complex64::new(10.0, 0.0),
        Complex64::new(1.0, 1.0),
        vec![
            Complex64::new(44.0, 10.0),
            Complex64::new(54.0, 10.0),
            Complex64::new(64.0, 10.0),
            Complex64::new(74.0, 10.0),
            Complex64::new(90.0, 10.0),
            Complex64::new(114.0, 10.0),
            Complex64::new(138.0, 10.0),
            Complex64::new(162.0, 10.0),
        ],
    );
}

#[test]
fn tree_transform_multi_pack_gemm_scatter_supports_complex_data_with_real_coefficients() {
    assert_tree_multi_mixed_dtype(
        vec![
            Complex32::new(1.0, 0.0),
            Complex32::new(2.0, 0.0),
            Complex32::new(3.0, 0.0),
            Complex32::new(4.0, 0.0),
            Complex32::new(5.0, 0.0),
            Complex32::new(6.0, 0.0),
            Complex32::new(7.0, 0.0),
            Complex32::new(8.0, 0.0),
        ],
        vec![2.0_f64, 3.0, 5.0, 7.0],
        Complex32::new(2.0, 0.0),
        Complex32::new(10.0, 0.0),
        Complex32::new(1.0, 1.0),
        vec![
            Complex32::new(44.0, 10.0),
            Complex32::new(54.0, 10.0),
            Complex32::new(64.0, 10.0),
            Complex32::new(74.0, 10.0),
            Complex32::new(90.0, 10.0),
            Complex32::new(114.0, 10.0),
            Complex32::new(138.0, 10.0),
            Complex32::new(162.0, 10.0),
        ],
    );
    assert_tree_multi_mixed_dtype(
        vec![
            Complex64::new(1.0, 0.0),
            Complex64::new(2.0, 0.0),
            Complex64::new(3.0, 0.0),
            Complex64::new(4.0, 0.0),
            Complex64::new(5.0, 0.0),
            Complex64::new(6.0, 0.0),
            Complex64::new(7.0, 0.0),
            Complex64::new(8.0, 0.0),
        ],
        vec![2.0_f64, 3.0, 5.0, 7.0],
        Complex64::new(2.0, 0.0),
        Complex64::new(10.0, 0.0),
        Complex64::new(1.0, 1.0),
        vec![
            Complex64::new(44.0, 10.0),
            Complex64::new(54.0, 10.0),
            Complex64::new(64.0, 10.0),
            Complex64::new(74.0, 10.0),
            Complex64::new(90.0, 10.0),
            Complex64::new(114.0, 10.0),
            Complex64::new(138.0, 10.0),
            Complex64::new(162.0, 10.0),
        ],
    );
}

#[test]
fn tree_transform_multi_uses_tensorkit_recoupling_orientation_for_all_numeric_dtypes() {
    assert_tree_multi_tensorkit_orientation_dtype(
        vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0],
        vec![10.0, 100.0, 1000.0, 20.0, 200.0, 2000.0],
        2.0,
        3.0,
        1.0,
        vec![10623.0, 12843.0, 21243.0, 25683.0],
    );
    assert_tree_multi_tensorkit_orientation_dtype(
        vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0],
        vec![10.0, 100.0, 1000.0, 20.0, 200.0, 2000.0],
        2.0,
        3.0,
        1.0,
        vec![10623.0, 12843.0, 21243.0, 25683.0],
    );
    assert_tree_multi_tensorkit_orientation_dtype(
        vec![1_i32, 2, 3, 4, 5, 6],
        vec![10, 100, 1000, 20, 200, 2000],
        2,
        3,
        1,
        vec![10623, 12843, 21243, 25683],
    );
    assert_tree_multi_tensorkit_orientation_dtype(
        vec![1_i64, 2, 3, 4, 5, 6],
        vec![10, 100, 1000, 20, 200, 2000],
        2,
        3,
        1,
        vec![10623, 12843, 21243, 25683],
    );
    assert_tree_multi_tensorkit_orientation_dtype(
        vec![
            Complex32::new(1.0, 0.0),
            Complex32::new(2.0, 0.0),
            Complex32::new(3.0, 0.0),
            Complex32::new(4.0, 0.0),
            Complex32::new(5.0, 0.0),
            Complex32::new(6.0, 0.0),
        ],
        vec![
            Complex32::new(10.0, 0.0),
            Complex32::new(100.0, 0.0),
            Complex32::new(1000.0, 0.0),
            Complex32::new(20.0, 0.0),
            Complex32::new(200.0, 0.0),
            Complex32::new(2000.0, 0.0),
        ],
        Complex32::new(2.0, 0.0),
        Complex32::new(3.0, 0.0),
        Complex32::new(1.0, 1.0),
        vec![
            Complex32::new(10623.0, 3.0),
            Complex32::new(12843.0, 3.0),
            Complex32::new(21243.0, 3.0),
            Complex32::new(25683.0, 3.0),
        ],
    );
    assert_tree_multi_tensorkit_orientation_dtype(
        vec![
            Complex64::new(1.0, 0.0),
            Complex64::new(2.0, 0.0),
            Complex64::new(3.0, 0.0),
            Complex64::new(4.0, 0.0),
            Complex64::new(5.0, 0.0),
            Complex64::new(6.0, 0.0),
        ],
        vec![
            Complex64::new(10.0, 0.0),
            Complex64::new(100.0, 0.0),
            Complex64::new(1000.0, 0.0),
            Complex64::new(20.0, 0.0),
            Complex64::new(200.0, 0.0),
            Complex64::new(2000.0, 0.0),
        ],
        Complex64::new(2.0, 0.0),
        Complex64::new(3.0, 0.0),
        Complex64::new(1.0, 1.0),
        vec![
            Complex64::new(10623.0, 3.0),
            Complex64::new(12843.0, 3.0),
            Complex64::new(21243.0, 3.0),
            Complex64::new(25683.0, 3.0),
        ],
    );
}

#[test]
fn tree_transform_dense_backend_matches_tensorkit_recoupling_orientation_for_gemm_dtypes() {
    assert_tree_multi_tensorkit_orientation_dense_dtype(
        vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0],
        vec![10.0, 100.0, 1000.0, 20.0, 200.0, 2000.0],
        2.0,
        3.0,
        1.0,
        vec![10623.0, 12843.0, 21243.0, 25683.0],
    );
    assert_tree_multi_tensorkit_orientation_dense_dtype(
        vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0],
        vec![10.0, 100.0, 1000.0, 20.0, 200.0, 2000.0],
        2.0,
        3.0,
        1.0,
        vec![10623.0, 12843.0, 21243.0, 25683.0],
    );
    assert_tree_multi_tensorkit_orientation_dense_dtype(
        vec![
            Complex32::new(1.0, 0.0),
            Complex32::new(2.0, 0.0),
            Complex32::new(3.0, 0.0),
            Complex32::new(4.0, 0.0),
            Complex32::new(5.0, 0.0),
            Complex32::new(6.0, 0.0),
        ],
        vec![
            Complex32::new(10.0, 0.0),
            Complex32::new(100.0, 0.0),
            Complex32::new(1000.0, 0.0),
            Complex32::new(20.0, 0.0),
            Complex32::new(200.0, 0.0),
            Complex32::new(2000.0, 0.0),
        ],
        Complex32::new(2.0, 0.0),
        Complex32::new(3.0, 0.0),
        Complex32::new(1.0, 1.0),
        vec![
            Complex32::new(10623.0, 3.0),
            Complex32::new(12843.0, 3.0),
            Complex32::new(21243.0, 3.0),
            Complex32::new(25683.0, 3.0),
        ],
    );
    assert_tree_multi_tensorkit_orientation_dense_dtype(
        vec![
            Complex64::new(1.0, 0.0),
            Complex64::new(2.0, 0.0),
            Complex64::new(3.0, 0.0),
            Complex64::new(4.0, 0.0),
            Complex64::new(5.0, 0.0),
            Complex64::new(6.0, 0.0),
        ],
        vec![
            Complex64::new(10.0, 0.0),
            Complex64::new(100.0, 0.0),
            Complex64::new(1000.0, 0.0),
            Complex64::new(20.0, 0.0),
            Complex64::new(200.0, 0.0),
            Complex64::new(2000.0, 0.0),
        ],
        Complex64::new(2.0, 0.0),
        Complex64::new(3.0, 0.0),
        Complex64::new(1.0, 1.0),
        vec![
            Complex64::new(10623.0, 3.0),
            Complex64::new(12843.0, 3.0),
            Complex64::new(21243.0, 3.0),
            Complex64::new(25683.0, 3.0),
        ],
    );
}

#[test]
fn tree_transform_dense_backend_calls_dense_matmul_for_multi_tree_blocks() {
    let src_space = TensorMapSpace::<2, 0>::from_dims([6, 1], []).unwrap();
    let dst_space = TensorMapSpace::<2, 0>::from_dims([4, 1], []).unwrap();
    let src_structure =
        BlockStructure::packed_column_major(2, [vec![2, 1], vec![2, 1], vec![2, 1]]).unwrap();
    let dst_structure = BlockStructure::packed_column_major(2, [vec![2, 1], vec![2, 1]]).unwrap();
    let src = TensorMap::<f64, 2, 0>::from_vec_with_structure(
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        src_space,
        src_structure,
    )
    .unwrap();
    let mut dst =
        TensorMap::<f64, 2, 0>::from_vec_with_structure(vec![1.0; 4], dst_space, dst_structure)
            .unwrap();
    let structure = TreeTransformStructure::compile(
        &dst,
        &src,
        &[TreeTransformBlockSpec::multi(
            vec![0, 1],
            vec![0, 1, 2],
            vec![10.0, 100.0, 1000.0, 20.0, 200.0, 2000.0],
        )],
    )
    .unwrap();
    let mut backend = DenseTreeTransformOperations::new(CountingDenseExecutor::default());
    let mut workspace = TreeTransformWorkspace::default();

    tree_transform_execute_with(
        &mut backend,
        &mut workspace,
        &structure,
        &mut dst,
        &src,
        2.0,
        3.0,
    )
    .unwrap();

    assert_eq!(backend.dense().dot_general_into_calls, 1);
    assert_eq!(dst.data(), &[10623.0, 12843.0, 21243.0, 25683.0]);
}

#[test]
fn tree_transform_compile_keyed_pairs_tree_blocks_by_key_not_index_for_all_numeric_dtypes() {
    assert_tree_multi_keyed_dtype(
        vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0],
        vec![10.0, 100.0, 1000.0, 20.0, 200.0, 2000.0],
        vec![7020.0, 9240.0, 3510.0, 4620.0],
    );
    assert_tree_multi_keyed_dtype(
        vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0],
        vec![10.0, 100.0, 1000.0, 20.0, 200.0, 2000.0],
        vec![7020.0, 9240.0, 3510.0, 4620.0],
    );
    assert_tree_multi_keyed_dtype(
        vec![1_i32, 2, 3, 4, 5, 6],
        vec![10, 100, 1000, 20, 200, 2000],
        vec![7020, 9240, 3510, 4620],
    );
    assert_tree_multi_keyed_dtype(
        vec![1_i64, 2, 3, 4, 5, 6],
        vec![10, 100, 1000, 20, 200, 2000],
        vec![7020, 9240, 3510, 4620],
    );
    assert_tree_multi_keyed_dtype(
        vec![
            Complex32::new(1.0, 0.0),
            Complex32::new(2.0, 0.0),
            Complex32::new(3.0, 0.0),
            Complex32::new(4.0, 0.0),
            Complex32::new(5.0, 0.0),
            Complex32::new(6.0, 0.0),
        ],
        vec![
            Complex32::new(10.0, 0.0),
            Complex32::new(100.0, 0.0),
            Complex32::new(1000.0, 0.0),
            Complex32::new(20.0, 0.0),
            Complex32::new(200.0, 0.0),
            Complex32::new(2000.0, 0.0),
        ],
        vec![
            Complex32::new(7020.0, 0.0),
            Complex32::new(9240.0, 0.0),
            Complex32::new(3510.0, 0.0),
            Complex32::new(4620.0, 0.0),
        ],
    );
    assert_tree_multi_keyed_dtype(
        vec![
            Complex64::new(1.0, 0.0),
            Complex64::new(2.0, 0.0),
            Complex64::new(3.0, 0.0),
            Complex64::new(4.0, 0.0),
            Complex64::new(5.0, 0.0),
            Complex64::new(6.0, 0.0),
        ],
        vec![
            Complex64::new(10.0, 0.0),
            Complex64::new(100.0, 0.0),
            Complex64::new(1000.0, 0.0),
            Complex64::new(20.0, 0.0),
            Complex64::new(200.0, 0.0),
            Complex64::new(2000.0, 0.0),
        ],
        vec![
            Complex64::new(7020.0, 0.0),
            Complex64::new(9240.0, 0.0),
            Complex64::new(3510.0, 0.0),
            Complex64::new(4620.0, 0.0),
        ],
    );
}

#[test]
fn tensoradd_with_backend_allocator_applies_axis_permutation() {
    let src_space = TensorMapSpace::<2, 0>::from_dims([2, 3], []).unwrap();
    let dst_space = TensorMapSpace::<2, 0>::from_dims([3, 2], []).unwrap();
    let src =
        TensorMap::<f64, 2, 0>::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], src_space).unwrap();
    let mut dst = TensorMap::<f64, 2, 0>::filled(10.0, dst_space).unwrap();
    let mut backend = HostTensorOperations;
    let mut allocator = HostAllocator::default();

    tensoradd_into_with(
        &mut backend,
        &mut allocator,
        &mut dst,
        &src,
        AxisPermutation::from_axes(&[1, 0]),
        2.0,
        3.0,
    )
    .unwrap();

    assert_eq!(dst.data(), &[32.0, 36.0, 40.0, 34.0, 38.0, 42.0]);
}

#[test]
fn tensoradd_structure_precomputes_permutation_pairing_and_descriptor() {
    let src_space = TensorMapSpace::<2, 0>::from_dims([2, 3], []).unwrap();
    let dst_space = TensorMapSpace::<2, 0>::from_dims([3, 2], []).unwrap();
    let src = TensorMap::<f64, 2, 0>::from_vec(vec![1.0; 6], src_space).unwrap();
    let dst = TensorMap::<f64, 2, 0>::filled(0.0, dst_space).unwrap();

    let structure =
        TensorAddStructure::compile(&dst, &src, AxisPermutation::from_axes(&[1, 0])).unwrap();

    assert_eq!(structure.rank(), 2);
    assert_eq!(structure.axes(), &[1, 0]);
    assert_eq!(structure.terms().len(), 1);
    assert_eq!(structure.terms()[0].key(), &BlockKey::trivial());
    assert_eq!(structure.terms()[0].dst_block(), 0);
    assert_eq!(structure.terms()[0].src_block(), 0);
}

#[test]
fn tensoradd_structure_replays_without_recompiling() {
    let src_space = TensorMapSpace::<2, 0>::from_dims([2, 3], []).unwrap();
    let dst_space = TensorMapSpace::<2, 0>::from_dims([3, 2], []).unwrap();
    let src =
        TensorMap::<f64, 2, 0>::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], src_space).unwrap();
    let mut dst = TensorMap::<f64, 2, 0>::filled(10.0, dst_space).unwrap();
    let structure =
        TensorAddStructure::compile(&dst, &src, AxisPermutation::from_axes(&[1, 0])).unwrap();
    let mut backend = HostTensorOperations;
    let mut allocator = HostAllocator::default();

    tensoradd_execute_with(
        &mut backend,
        &mut allocator,
        &structure,
        &mut dst,
        &src,
        2.0,
        0.0,
    )
    .unwrap();
    tensoradd_execute_with(
        &mut backend,
        &mut allocator,
        &structure,
        &mut dst,
        &src,
        1.0,
        1.0,
    )
    .unwrap();

    assert_eq!(dst.data(), &[3.0, 9.0, 15.0, 6.0, 12.0, 18.0]);
}

#[test]
fn tensoradd_structure_compiles_concrete_shape_and_replays_it() {
    let src_space = TensorMapSpace::<2, 0>::from_dims([4, 5], []).unwrap();
    let dst_space = TensorMapSpace::<2, 0>::from_dims([5, 4], []).unwrap();
    let src =
        TensorMap::<f64, 2, 0>::from_vec((1..=20).map(|x| x as f64).collect(), src_space).unwrap();
    let mut dst = TensorMap::<f64, 2, 0>::filled(0.0, dst_space).unwrap();
    let structure =
        TensorAddStructure::compile(&dst, &src, AxisPermutation::from_axes(&[1, 0])).unwrap();
    let mut backend = HostTensorOperations;
    let mut allocator = HostAllocator::default();

    tensoradd_execute_with(
        &mut backend,
        &mut allocator,
        &structure,
        &mut dst,
        &src,
        2.0,
        0.0,
    )
    .unwrap();

    assert_eq!(
        dst.data(),
        &[
            2.0, 10.0, 18.0, 26.0, 34.0, 4.0, 12.0, 20.0, 28.0, 36.0, 6.0, 14.0, 22.0, 30.0, 38.0,
            8.0, 16.0, 24.0, 32.0, 40.0,
        ]
    );
}

#[test]
fn tensoradd_structure_replays_multiple_packed_blocks() {
    let src_space = TensorMapSpace::<2, 0>::from_dims([4, 4], []).unwrap();
    let dst_space = TensorMapSpace::<2, 0>::from_dims([4, 4], []).unwrap();
    let src_structure = BlockStructure::packed_column_major(2, [vec![2, 3], vec![1, 4]]).unwrap();
    let dst_structure = BlockStructure::packed_column_major(2, [vec![3, 2], vec![4, 1]]).unwrap();
    let src = TensorMap::<f64, 2, 0>::from_vec_with_structure(
        (1..=10).map(|x| x as f64).collect(),
        src_space,
        src_structure,
    )
    .unwrap();
    let mut dst =
        TensorMap::<f64, 2, 0>::from_vec_with_structure(vec![0.0; 10], dst_space, dst_structure)
            .unwrap();
    let structure =
        TensorAddStructure::compile(&dst, &src, AxisPermutation::from_axes(&[1, 0])).unwrap();
    let mut backend = HostTensorOperations;
    let mut allocator = HostAllocator::default();

    tensoradd_execute_with(
        &mut backend,
        &mut allocator,
        &structure,
        &mut dst,
        &src,
        2.0,
        0.0,
    )
    .unwrap();

    assert_eq!(
        dst.data(),
        &[2.0, 6.0, 10.0, 4.0, 8.0, 12.0, 14.0, 16.0, 18.0, 20.0]
    );
}

#[test]
fn tensoradd_structure_pairs_blocks_by_key_not_index() {
    let src_space = TensorMapSpace::<2, 0>::from_dims([4, 4], []).unwrap();
    let dst_space = TensorMapSpace::<2, 0>::from_dims([4, 4], []).unwrap();
    let src_structure = BlockStructure::packed_column_major_with_keys(
        2,
        [
            (BlockKey::sector_ids([10]), vec![2, 3]),
            (BlockKey::sector_ids([20]), vec![1, 4]),
        ],
    )
    .unwrap();
    let dst_structure = BlockStructure::packed_column_major_with_keys(
        2,
        [
            (BlockKey::sector_ids([20]), vec![4, 1]),
            (BlockKey::sector_ids([10]), vec![3, 2]),
        ],
    )
    .unwrap();
    let src = TensorMap::<f64, 2, 0>::from_vec_with_structure(
        (1..=10).map(|x| x as f64).collect(),
        src_space,
        src_structure,
    )
    .unwrap();
    let mut dst =
        TensorMap::<f64, 2, 0>::from_vec_with_structure(vec![0.0; 10], dst_space, dst_structure)
            .unwrap();
    let structure =
        TensorAddStructure::compile(&dst, &src, AxisPermutation::from_axes(&[1, 0])).unwrap();
    let mut backend = HostTensorOperations;
    let mut allocator = HostAllocator::default();

    assert_eq!(structure.terms()[0].key(), &BlockKey::sector_ids([20]));
    assert_eq!(structure.terms()[0].dst_block(), 0);
    assert_eq!(structure.terms()[0].src_block(), 1);
    assert_eq!(structure.terms()[1].key(), &BlockKey::sector_ids([10]));
    assert_eq!(structure.terms()[1].dst_block(), 1);
    assert_eq!(structure.terms()[1].src_block(), 0);

    tensoradd_execute_with(
        &mut backend,
        &mut allocator,
        &structure,
        &mut dst,
        &src,
        2.0,
        0.0,
    )
    .unwrap();

    assert_eq!(
        dst.data(),
        &[14.0, 16.0, 18.0, 20.0, 2.0, 6.0, 10.0, 4.0, 8.0, 12.0]
    );
}

#[test]
fn tensoradd_structure_rejects_invalid_permutation_at_compile_time() {
    let space = TensorMapSpace::<2, 0>::from_dims([2, 2], []).unwrap();
    let src = TensorMap::<f64, 2, 0>::filled(1.0, space.clone()).unwrap();
    let dst = TensorMap::<f64, 2, 0>::filled(0.0, space).unwrap();

    let err =
        TensorAddStructure::compile(&dst, &src, AxisPermutation::from_axes(&[0, 0])).unwrap_err();

    assert_eq!(
        err,
        OperationError::InvalidPermutation {
            axes: vec![0, 0],
            rank: 2,
        }
    );
}

#[test]
fn tensoradd_structure_rejects_incompatible_shape_at_compile_time() {
    let src_space = TensorMapSpace::<2, 0>::from_dims([4, 5], []).unwrap();
    let dst_space = TensorMapSpace::<2, 0>::from_dims([4, 5], []).unwrap();
    let src = TensorMap::<f64, 2, 0>::filled(1.0, src_space).unwrap();
    let dst = TensorMap::<f64, 2, 0>::filled(0.0, dst_space).unwrap();

    let err =
        TensorAddStructure::compile(&dst, &src, AxisPermutation::from_axes(&[1, 0])).unwrap_err();

    assert_eq!(
        err,
        OperationError::ShapeMismatch {
            dst: vec![4, 5],
            src: vec![5, 4],
        }
    );
}

#[test]
fn tensoradd_structure_rejects_incompatible_replay_structure() {
    let compile_src_space = TensorMapSpace::<2, 0>::from_dims([2, 3], []).unwrap();
    let compile_dst_space = TensorMapSpace::<2, 0>::from_dims([3, 2], []).unwrap();
    let compile_src = TensorMap::<f64, 2, 0>::filled(1.0, compile_src_space).unwrap();
    let compile_dst = TensorMap::<f64, 2, 0>::filled(0.0, compile_dst_space).unwrap();
    let structure = TensorAddStructure::compile(
        &compile_dst,
        &compile_src,
        AxisPermutation::from_axes(&[1, 0]),
    )
    .unwrap();

    let src_space = TensorMapSpace::<2, 0>::from_dims([4, 5], []).unwrap();
    let dst_space = TensorMapSpace::<2, 0>::from_dims([5, 4], []).unwrap();
    let src = TensorMap::<f64, 2, 0>::filled(1.0, src_space).unwrap();
    let mut dst = TensorMap::<f64, 2, 0>::filled(0.0, dst_space).unwrap();
    let mut backend = HostTensorOperations;
    let mut allocator = HostAllocator::default();

    let err = tensoradd_execute_with(
        &mut backend,
        &mut allocator,
        &structure,
        &mut dst,
        &src,
        1.0,
        0.0,
    )
    .unwrap_err();

    assert_eq!(err, OperationError::StructureMismatch { tensor: "dst" });
}

#[test]
fn tensorcontract_structure_precomputes_canonical_dense_descriptor() {
    let lhs_space = TensorMapSpace::<2, 0>::from_dims([2, 3], []).unwrap();
    let rhs_space = TensorMapSpace::<2, 0>::from_dims([3, 2], []).unwrap();
    let dst_space = TensorMapSpace::<2, 0>::from_dims([2, 2], []).unwrap();
    let lhs = TensorMap::<f64, 2, 0>::from_vec(vec![1.0; 6], lhs_space).unwrap();
    let rhs = TensorMap::<f64, 2, 0>::from_vec(vec![1.0; 6], rhs_space).unwrap();
    let dst = TensorMap::<f64, 2, 0>::filled(0.0, dst_space).unwrap();

    let structure = TensorContractStructure::compile(
        &dst,
        &lhs,
        &rhs,
        TensorContractAxisSpec::canonical(&[1], &[0]),
    )
    .unwrap();

    assert_eq!(structure.dst_rank(), 2);
    assert_eq!(structure.lhs_rank(), 2);
    assert_eq!(structure.rhs_rank(), 2);
    assert_eq!(structure.lhs_contracting_axes(), &[1]);
    assert_eq!(structure.rhs_contracting_axes(), &[0]);
    assert_eq!(structure.output_axes(), &[0, 1]);
    assert_eq!(structure.terms().len(), 1);
    assert_eq!(structure.terms()[0].key(), &BlockKey::trivial());
    assert_eq!(structure.terms()[0].dst_block(), 0);
    assert_eq!(structure.terms()[0].lhs_block(), 0);
    assert_eq!(structure.terms()[0].rhs_block(), 0);
}

#[test]
fn tensorcontract_into_uses_dense_backend_for_matmul_and_alpha_beta() {
    let lhs_space = TensorMapSpace::<2, 0>::from_dims([2, 3], []).unwrap();
    let rhs_space = TensorMapSpace::<2, 0>::from_dims([3, 2], []).unwrap();
    let dst_space = TensorMapSpace::<2, 0>::from_dims([2, 2], []).unwrap();
    let lhs =
        TensorMap::<f64, 2, 0>::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], lhs_space).unwrap();
    let rhs =
        TensorMap::<f64, 2, 0>::from_vec(vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0], rhs_space).unwrap();
    let mut dst = TensorMap::<f64, 2, 0>::filled(1.0, dst_space).unwrap();

    tensorcontract_into(
        &mut dst,
        &lhs,
        &rhs,
        TensorContractAxisSpec::canonical(&[1], &[0]),
        2.0,
        3.0,
    )
    .unwrap();

    assert_eq!(dst.data(), &[155.0, 203.0, 209.0, 275.0]);
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

        assert_eq!(lhs.shape(), &[2, 3]);
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
        Ok(())
    }
}

#[test]
fn tensorcontract_structure_honors_output_permutation_with_workspace_scatter() {
    let lhs_space = TensorMapSpace::<2, 0>::from_dims([2, 3], []).unwrap();
    let rhs_space = TensorMapSpace::<2, 0>::from_dims([4, 3], []).unwrap();
    let dst_space = TensorMapSpace::<2, 0>::from_dims([4, 2], []).unwrap();
    let lhs =
        TensorMap::<f64, 2, 0>::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], lhs_space).unwrap();
    let rhs =
        TensorMap::<f64, 2, 0>::from_vec((7..=18).map(|value| value as f64).collect(), rhs_space)
            .unwrap();
    let mut dst = TensorMap::<f64, 2, 0>::filled(0.0, dst_space).unwrap();
    let structure = TensorContractStructure::compile(
        &dst,
        &lhs,
        &rhs,
        TensorContractAxisSpec::new(&[1], &[1], AxisPermutation::from_axes(&[1, 0])),
    )
    .unwrap();
    let mut backend =
        DenseTreeTransformOperations::new(ContractLayoutCheckingDenseExecutor::default());
    let mut workspace = TensorContractWorkspace::default();

    tensorcontract_execute_with(
        &mut backend,
        &mut workspace,
        &structure,
        &mut dst,
        &lhs,
        &rhs,
        1.0,
        0.0,
    )
    .unwrap();

    assert_eq!(backend.dense().dot_general_into_calls, 1);
    assert_eq!(
        dst.data(),
        &[115.0, 124.0, 133.0, 142.0, 148.0, 160.0, 172.0, 184.0]
    );
    assert_eq!(workspace.output_len(), 8);
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
        TensorContractAxisSpec::canonical(&[1], &[0]),
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
        TensorContractAxisSpec::canonical(&[1], &[0]),
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

#[test]
fn tensorcontract_dense_backend_covers_all_gemm_dtypes() {
    assert_tensorcontract_scalar_dtype(2.0_f32, 3.0_f32, 5.0_f32, 27.0_f32);
    assert_tensorcontract_scalar_dtype(2.0_f64, 3.0_f64, 5.0_f64, 27.0_f64);
    assert_tensorcontract_scalar_dtype(
        Complex32::new(2.0, 0.0),
        Complex32::new(3.0, 0.0),
        Complex32::new(5.0, 0.0),
        Complex32::new(27.0, 0.0),
    );
    assert_tensorcontract_scalar_dtype(
        Complex64::new(2.0, 0.0),
        Complex64::new(3.0, 0.0),
        Complex64::new(5.0, 0.0),
        Complex64::new(27.0, 0.0),
    );
}

#[test]
fn tensorcontract_weighted_terms_support_all_gemm_dtypes() {
    assert_weighted_tensorcontract_scalar_dtype(2.0_f32, 3.0_f32, 5.0_f32, 21.0_f32);
    assert_weighted_tensorcontract_scalar_dtype(2.0_f64, 3.0_f64, 5.0_f64, 21.0_f64);
    assert_weighted_tensorcontract_scalar_dtype(
        Complex32::new(2.0, 1.0),
        Complex32::new(3.0, -1.0),
        Complex32::new(5.0, 2.0),
        Complex32::new(22.0, 7.0),
    );
    assert_weighted_tensorcontract_scalar_dtype(
        Complex64::new(2.0, 1.0),
        Complex64::new(3.0, -1.0),
        Complex64::new(5.0, 2.0),
        Complex64::new(22.0, 7.0),
    );
}

#[test]
fn tensorcontract_structure_rejects_invalid_axes_at_compile_time() {
    let lhs_space = TensorMapSpace::<2, 0>::from_dims([2, 3], []).unwrap();
    let rhs_space = TensorMapSpace::<2, 0>::from_dims([3, 2], []).unwrap();
    let dst_space = TensorMapSpace::<2, 0>::from_dims([2, 2], []).unwrap();
    let lhs = TensorMap::<f64, 2, 0>::filled(1.0, lhs_space).unwrap();
    let rhs = TensorMap::<f64, 2, 0>::filled(1.0, rhs_space).unwrap();
    let dst = TensorMap::<f64, 2, 0>::filled(0.0, dst_space).unwrap();

    let duplicate = TensorContractStructure::compile(
        &dst,
        &lhs,
        &rhs,
        TensorContractAxisSpec::canonical(&[1, 1], &[0, 1]),
    )
    .unwrap_err();
    assert_eq!(
        duplicate,
        OperationError::InvalidAxisSet {
            tensor: "lhs",
            axes: vec![1, 1],
            rank: 2,
        }
    );

    let count = TensorContractStructure::compile(
        &dst,
        &lhs,
        &rhs,
        TensorContractAxisSpec::canonical(&[1], &[0, 1]),
    )
    .unwrap_err();
    assert_eq!(
        count,
        OperationError::ContractAxisCountMismatch { lhs: 1, rhs: 2 }
    );
}

#[test]
fn tensorcontract_structure_rejects_dimension_and_output_mismatch_at_compile_time() {
    let lhs_space = TensorMapSpace::<2, 0>::from_dims([2, 3], []).unwrap();
    let rhs_space = TensorMapSpace::<2, 0>::from_dims([4, 2], []).unwrap();
    let dst_space = TensorMapSpace::<2, 0>::from_dims([2, 4], []).unwrap();
    let lhs = TensorMap::<f64, 2, 0>::filled(1.0, lhs_space).unwrap();
    let rhs = TensorMap::<f64, 2, 0>::filled(1.0, rhs_space).unwrap();
    let dst = TensorMap::<f64, 2, 0>::filled(0.0, dst_space).unwrap();

    let contracted_dim = TensorContractStructure::compile(
        &dst,
        &lhs,
        &rhs,
        TensorContractAxisSpec::canonical(&[1], &[1]),
    )
    .unwrap_err();
    assert_eq!(
        contracted_dim,
        OperationError::ShapeMismatch {
            dst: vec![3],
            src: vec![2],
        }
    );

    let wrong_dst_space = TensorMapSpace::<2, 0>::from_dims([4, 2], []).unwrap();
    let wrong_dst = TensorMap::<f64, 2, 0>::filled(0.0, wrong_dst_space).unwrap();
    let valid_rhs_space = TensorMapSpace::<2, 0>::from_dims([3, 2], []).unwrap();
    let valid_rhs = TensorMap::<f64, 2, 0>::filled(1.0, valid_rhs_space).unwrap();
    let output = TensorContractStructure::compile(
        &wrong_dst,
        &lhs,
        &valid_rhs,
        TensorContractAxisSpec::canonical(&[1], &[0]),
    )
    .unwrap_err();
    assert_eq!(
        output,
        OperationError::ShapeMismatch {
            dst: vec![4, 2],
            src: vec![2, 2],
        }
    );
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

#[test]
fn tensorcontract_structure_rejects_incompatible_replay_structure_before_dense_execution() {
    let compile_lhs_space = TensorMapSpace::<2, 0>::from_dims([2, 3], []).unwrap();
    let compile_rhs_space = TensorMapSpace::<2, 0>::from_dims([3, 2], []).unwrap();
    let compile_dst_space = TensorMapSpace::<2, 0>::from_dims([2, 2], []).unwrap();
    let compile_lhs = TensorMap::<f64, 2, 0>::filled(1.0, compile_lhs_space).unwrap();
    let compile_rhs = TensorMap::<f64, 2, 0>::filled(1.0, compile_rhs_space).unwrap();
    let compile_dst = TensorMap::<f64, 2, 0>::filled(0.0, compile_dst_space).unwrap();
    let structure = TensorContractStructure::compile(
        &compile_dst,
        &compile_lhs,
        &compile_rhs,
        TensorContractAxisSpec::canonical(&[1], &[0]),
    )
    .unwrap();

    let lhs_space = TensorMapSpace::<2, 0>::from_dims([4, 3], []).unwrap();
    let rhs_space = TensorMapSpace::<2, 0>::from_dims([3, 2], []).unwrap();
    let dst_space = TensorMapSpace::<2, 0>::from_dims([4, 2], []).unwrap();
    let lhs = TensorMap::<f64, 2, 0>::filled(1.0, lhs_space).unwrap();
    let rhs = TensorMap::<f64, 2, 0>::filled(1.0, rhs_space).unwrap();
    let mut dst = TensorMap::<f64, 2, 0>::filled(0.0, dst_space).unwrap();
    let mut backend = DenseTreeTransformOperations::new(PanicDenseExecutor);
    let mut workspace = TensorContractWorkspace::default();

    let err = tensorcontract_execute_with(
        &mut backend,
        &mut workspace,
        &structure,
        &mut dst,
        &lhs,
        &rhs,
        1.0,
        0.0,
    )
    .unwrap_err();

    assert_eq!(err, OperationError::StructureMismatch { tensor: "dst" });
}

#[test]
fn tensorcontract_structure_rejects_multiblock_until_block_sparse_enumeration_exists() {
    let dense = BlockStructure::trivial(&[2, 2]).unwrap();
    let multiblock = BlockStructure::packed_column_major(2, [vec![1, 2], vec![1, 2]]).unwrap();

    let err = TensorContractStructure::compile_structures(
        &dense,
        &multiblock,
        &dense,
        TensorContractAxisSpec::canonical(&[1], &[0]),
    )
    .unwrap_err();

    assert_eq!(
        err,
        OperationError::UnsupportedTensorContractScope {
            message: "block-sparse contraction enumeration is not implemented yet",
        }
    );
}

#[test]
fn tensorcontract_structure_replays_explicit_block_terms_and_applies_beta_once() {
    let lhs_space = TensorMapSpace::<2, 0>::from_dims([2, 2], []).unwrap();
    let rhs_space = TensorMapSpace::<2, 0>::from_dims([2, 2], []).unwrap();
    let dst_space = TensorMapSpace::<2, 0>::from_dims([1, 1], []).unwrap();
    let lhs_structure = BlockStructure::packed_column_major_with_keys(
        2,
        [
            (BlockKey::sector_ids([10]), vec![1, 2]),
            (BlockKey::sector_ids([20]), vec![1, 2]),
        ],
    )
    .unwrap();
    let rhs_structure = BlockStructure::packed_column_major_with_keys(
        2,
        [
            (BlockKey::sector_ids([30]), vec![2, 1]),
            (BlockKey::sector_ids([40]), vec![2, 1]),
        ],
    )
    .unwrap();
    let dst_structure = BlockStructure::packed_column_major_with_keys(
        2,
        [(BlockKey::sector_ids([99]), vec![1, 1])],
    )
    .unwrap();
    let lhs = TensorMap::<f64, 2, 0>::from_vec_with_structure(
        vec![1.0, 2.0, 3.0, 4.0],
        lhs_space,
        lhs_structure,
    )
    .unwrap();
    let rhs = TensorMap::<f64, 2, 0>::from_vec_with_structure(
        vec![5.0, 6.0, 7.0, 8.0],
        rhs_space,
        rhs_structure,
    )
    .unwrap();
    let mut dst =
        TensorMap::<f64, 2, 0>::from_vec_with_structure(vec![10.0], dst_space, dst_structure)
            .unwrap();
    let structure = TensorContractStructure::compile_with_block_specs(
        &dst,
        &lhs,
        &rhs,
        TensorContractAxisSpec::canonical(&[1], &[0]),
        &[
            TensorContractBlockSpec::with_coefficient(0, 0, 0, 0.5),
            TensorContractBlockSpec::with_coefficient(0, 1, 1, 2.0),
        ],
    )
    .unwrap();

    tensorcontract_execute_with(
        &mut DenseTreeTransformOperations::default_executor(),
        &mut TensorContractWorkspace::default(),
        &structure,
        &mut dst,
        &lhs,
        &rhs,
        2.0,
        3.0,
    )
    .unwrap();

    assert_eq!(structure.terms().len(), 2);
    assert_eq!(dst.data(), &[259.0]);
}

#[test]
fn tensorcontract_structure_rejects_invalid_explicit_block_term_at_compile_time() {
    let dense = BlockStructure::trivial(&[1, 1]).unwrap();

    let err = TensorContractStructure::compile_structures_with_block_specs(
        &dense,
        &dense,
        &dense,
        TensorContractAxisSpec::canonical(&[1], &[0]),
        &[TensorContractBlockSpec::new(0, 1, 0)],
    )
    .unwrap_err();

    assert_eq!(
        err,
        OperationError::BlockIndexOutOfBounds {
            tensor: "lhs",
            index: 1,
            count: 1,
        }
    );
}

#[test]
fn tensorcontract_fusion_structure_enumerates_z2_compose_blocks_and_replays() {
    let rule = Z2FusionRule;
    let leg = || SectorLeg::new([SectorId::new(0), SectorId::new(1)], false);
    let lhs_space = FusionTensorMapSpace::from_degeneracy_shapes(
        TensorMapSpace::<1, 1>::from_dims([1], [1]).unwrap(),
        FusionTreeHomSpace::new(
            FusionProductSpace::new([leg()]),
            FusionProductSpace::new([leg()]),
        ),
        &rule,
        [vec![1, 1], vec![1, 1]],
    )
    .unwrap();
    let rhs_space = FusionTensorMapSpace::from_degeneracy_shapes(
        TensorMapSpace::<1, 1>::from_dims([1], [1]).unwrap(),
        FusionTreeHomSpace::new(
            FusionProductSpace::new([leg()]),
            FusionProductSpace::new([leg()]),
        ),
        &rule,
        [vec![1, 1], vec![1, 1]],
    )
    .unwrap();
    let dst_space = FusionTensorMapSpace::from_degeneracy_shapes(
        TensorMapSpace::<1, 1>::from_dims([1], [1]).unwrap(),
        FusionTreeHomSpace::new(
            FusionProductSpace::new([leg()]),
            FusionProductSpace::new([leg()]),
        ),
        &rule,
        [vec![1, 1], vec![1, 1]],
    )
    .unwrap();
    let lhs =
        TensorMap::<f64, 1, 1>::from_vec_with_fusion_space(vec![2.0, 3.0], lhs_space).unwrap();
    let rhs =
        TensorMap::<f64, 1, 1>::from_vec_with_fusion_space(vec![5.0, 7.0], rhs_space).unwrap();
    let mut dst =
        TensorMap::<f64, 1, 1>::from_vec_with_fusion_space(vec![10.0, 20.0], dst_space).unwrap();

    let specs = tensorcontract_fusion_block_specs(
        &rule,
        dst.fusion_space().unwrap(),
        lhs.fusion_space().unwrap(),
        rhs.fusion_space().unwrap(),
        TensorContractAxisSpec::canonical(&[1], &[0]),
    )
    .unwrap();
    assert_eq!(
        specs,
        vec![
            TensorContractBlockSpec::new(0, 0, 0),
            TensorContractBlockSpec::new(1, 1, 1),
        ]
    );

    tensorcontract_fusion_into(
        &rule,
        &mut dst,
        &lhs,
        &rhs,
        TensorContractAxisSpec::canonical(&[1], &[0]),
        2.0,
        3.0,
    )
    .unwrap();

    assert_eq!(dst.data(), &[50.0, 102.0]);
}

#[test]
fn tensorcontract_fusion_block_specs_enumerates_su2_innerline_blocks_from_homspace() {
    let rule = SU2FusionRule;
    let half = SectorId::new(1);
    let lhs_space = FusionTensorMapSpace::from_degeneracy_shapes(
        TensorMapSpace::<3, 1>::from_dims([1, 1, 1], [1]).unwrap(),
        FusionTreeHomSpace::from_sector_ids([1, 1, 1], [1]),
        &rule,
        [vec![1, 1, 1, 1], vec![1, 1, 1, 1]],
    )
    .unwrap();
    let rhs_space = FusionTensorMapSpace::from_degeneracy_shapes(
        TensorMapSpace::<1, 1>::from_dims([1], [1]).unwrap(),
        FusionTreeHomSpace::from_sector_ids([1], [1]),
        &rule,
        [vec![1, 1]],
    )
    .unwrap();
    let dst_space = FusionTensorMapSpace::from_degeneracy_shapes(
        TensorMapSpace::<3, 1>::from_dims([1, 1, 1], [1]).unwrap(),
        FusionTreeHomSpace::from_sector_ids([1, 1, 1], [1]),
        &rule,
        [vec![1, 1, 1, 1], vec![1, 1, 1, 1]],
    )
    .unwrap();

    let specs = tensorcontract_fusion_block_specs(
        &rule,
        &dst_space,
        &lhs_space,
        &rhs_space,
        TensorContractAxisSpec::canonical(&[3], &[0]),
    )
    .unwrap();

    assert_eq!(
        specs,
        vec![
            TensorContractBlockSpec::new(0, 0, 0),
            TensorContractBlockSpec::new(1, 1, 0),
        ]
    );
    assert_eq!(
        dst_space
            .homspace()
            .fusion_tree_keys_from_external_sectors(&rule, &[half, half, half, half])
            .unwrap()
            .len(),
        2
    );
}

#[test]
fn tensorcontract_fusion_block_specs_rejects_missing_destination_subblock() {
    let rule = Z2FusionRule;
    let leg = || SectorLeg::new([SectorId::new(0), SectorId::new(1)], false);
    let lhs_space = FusionTensorMapSpace::from_degeneracy_shapes(
        TensorMapSpace::<1, 1>::from_dims([1], [1]).unwrap(),
        FusionTreeHomSpace::new(
            FusionProductSpace::new([leg()]),
            FusionProductSpace::new([leg()]),
        ),
        &rule,
        [vec![1, 1], vec![1, 1]],
    )
    .unwrap();
    let rhs_space = FusionTensorMapSpace::from_degeneracy_shapes(
        TensorMapSpace::<1, 1>::from_dims([1], [1]).unwrap(),
        FusionTreeHomSpace::new(
            FusionProductSpace::new([leg()]),
            FusionProductSpace::new([leg()]),
        ),
        &rule,
        [vec![1, 1], vec![1, 1]],
    )
    .unwrap();
    let dst_hom = FusionTreeHomSpace::new(
        FusionProductSpace::new([leg()]),
        FusionProductSpace::new([leg()]),
    );
    let keys = dst_hom.fusion_tree_keys(&rule);
    let dst_structure =
        BlockStructure::packed_column_major_with_keys(2, [(keys[0].clone(), vec![1, 1])]).unwrap();
    let dst_space = FusionTensorMapSpace::new(
        TensorMapSpace::<1, 1>::from_dims([1], [1]).unwrap(),
        dst_hom,
        dst_structure,
    )
    .unwrap();

    let err = tensorcontract_fusion_block_specs(
        &rule,
        &dst_space,
        &lhs_space,
        &rhs_space,
        TensorContractAxisSpec::canonical(&[1], &[0]),
    )
    .unwrap_err();

    assert_eq!(
        err,
        OperationError::MissingBlockKey {
            key: keys[1].clone().into()
        }
    );
}

#[test]
fn tensorcontract_fusion_block_specs_rejects_source_tree_transform_terms() {
    let rule = Z2FusionRule;
    let leg = |is_dual| SectorLeg::new([SectorId::new(0)], is_dual);
    let fusion_space = FusionTensorMapSpace::from_degeneracy_shapes(
        TensorMapSpace::<1, 1>::from_dims([1], [1]).unwrap(),
        FusionTreeHomSpace::new(
            FusionProductSpace::new([leg(false)]),
            FusionProductSpace::new([leg(false)]),
        ),
        &rule,
        [vec![1, 1]],
    )
    .unwrap();
    let transformed_dst_space = FusionTensorMapSpace::from_degeneracy_shapes(
        TensorMapSpace::<1, 1>::from_dims([1], [1]).unwrap(),
        FusionTreeHomSpace::new(
            FusionProductSpace::new([leg(true)]),
            FusionProductSpace::new([leg(true)]),
        ),
        &rule,
        [vec![1, 1]],
    )
    .unwrap();

    let err = tensorcontract_fusion_block_specs(
        &rule,
        &transformed_dst_space,
        &fusion_space,
        &fusion_space,
        TensorContractAxisSpec::canonical(&[0], &[1]),
    )
    .unwrap_err();

    assert_eq!(
            err,
            OperationError::UnsupportedTensorContractScope {
                message: "fusion contraction requiring source tree-pair transforms is not implemented; pre-transform operands explicitly",
            }
        );

    let specs = tensorcontract_fusion_block_specs(
        &rule,
        &transformed_dst_space,
        &fusion_space,
        &fusion_space,
        TensorContractAxisSpec::new(&[1], &[0], AxisPermutation::from_axes(&[1, 0])),
    )
    .unwrap();

    assert_eq!(
        specs,
        vec![TensorContractBlockSpec::with_coefficient(0, 0, 0, 1.0)]
    );
}

#[test]
fn tensorcontract_fusion_into_rejects_source_tree_transform_terms() {
    let rule = Z2FusionRule;
    let leg = |is_dual| SectorLeg::new([SectorId::new(0)], is_dual);
    let src_space = FusionTensorMapSpace::from_degeneracy_shapes(
        TensorMapSpace::<1, 1>::from_dims([1], [1]).unwrap(),
        FusionTreeHomSpace::new(
            FusionProductSpace::new([leg(false)]),
            FusionProductSpace::new([leg(false)]),
        ),
        &rule,
        [vec![1, 1]],
    )
    .unwrap();
    let dst_space = FusionTensorMapSpace::from_degeneracy_shapes(
        TensorMapSpace::<1, 1>::from_dims([1], [1]).unwrap(),
        FusionTreeHomSpace::new(
            FusionProductSpace::new([leg(true)]),
            FusionProductSpace::new([leg(true)]),
        ),
        &rule,
        [vec![1, 1]],
    )
    .unwrap();
    let lhs =
        TensorMap::<f64, 1, 1>::from_vec_with_fusion_space(vec![2.0], src_space.clone()).unwrap();
    let rhs = TensorMap::<f64, 1, 1>::from_vec_with_fusion_space(vec![5.0], src_space).unwrap();
    let mut dst = TensorMap::<f64, 1, 1>::from_vec_with_fusion_space(vec![7.0], dst_space).unwrap();

    let err = tensorcontract_fusion_into(
        &rule,
        &mut dst,
        &lhs,
        &rhs,
        TensorContractAxisSpec::canonical(&[0], &[1]),
        3.0,
        11.0,
    )
    .unwrap_err();

    assert_eq!(
            err,
            OperationError::UnsupportedTensorContractScope {
                message: "fusion contraction requiring source tree-pair transforms is not implemented; pre-transform operands explicitly",
            }
        );
}

#[test]
fn tensorcontract_fusion_output_recoupling_uses_su2_coefficients() {
    let rule = SU2FusionRule;
    let src_key = all_codomain_fusion_tree_test_key(
        [1, 1, 1, 1],
        Some(0),
        [false, false, false, false],
        [0, 1],
        [1, 1, 1],
    );
    let dst_key0 = src_key.clone();
    let dst_key1 = all_codomain_fusion_tree_test_key(
        [1, 1, 1, 1],
        Some(0),
        [false, false, false, false],
        [2, 1],
        [1, 1, 1],
    );
    let scalar_key = BlockKey::from(FusionTreeBlockKey::pair(
        empty_fusion_tree(),
        empty_fusion_tree(),
    ));
    let lhs_space = FusionTensorMapSpace::new(
        TensorMapSpace::<4, 0>::from_dims([1, 1, 1, 1], []).unwrap(),
        FusionTreeHomSpace::from_sector_ids([1, 1, 1, 1], []),
        BlockStructure::packed_column_major_with_keys(4, [(src_key, vec![1, 1, 1, 1])]).unwrap(),
    )
    .unwrap();
    let rhs_space = FusionTensorMapSpace::new(
        TensorMapSpace::<0, 0>::from_dims([], []).unwrap(),
        FusionTreeHomSpace::from_sector_ids([], []),
        BlockStructure::packed_column_major_with_keys(0, [(scalar_key, vec![])]).unwrap(),
    )
    .unwrap();
    let dst_space = FusionTensorMapSpace::new(
        TensorMapSpace::<4, 0>::from_dims([1, 1, 1, 1], []).unwrap(),
        FusionTreeHomSpace::from_sector_ids([1, 1, 1, 1], []),
        BlockStructure::packed_column_major_with_keys(
            4,
            [(dst_key0, vec![1, 1, 1, 1]), (dst_key1, vec![1, 1, 1, 1])],
        )
        .unwrap(),
    )
    .unwrap();
    let lhs = TensorMap::<f64, 4, 0>::from_vec_with_fusion_space(vec![10.0], lhs_space).unwrap();
    let rhs = TensorMap::<f64, 0, 0>::from_vec_with_fusion_space(vec![5.0], rhs_space).unwrap();
    let mut dst =
        TensorMap::<f64, 4, 0>::from_vec_with_fusion_space(vec![1.0, 2.0], dst_space).unwrap();

    let specs = tensorcontract_fusion_block_specs(
        &rule,
        dst.fusion_space().unwrap(),
        lhs.fusion_space().unwrap(),
        rhs.fusion_space().unwrap(),
        TensorContractAxisSpec::new(&[], &[], AxisPermutation::from_axes(&[0, 2, 1, 3])),
    )
    .unwrap();
    assert_eq!(specs.len(), 2);
    assert_eq!(specs[0].dst_block(), 0);
    assert_eq!(specs[1].dst_block(), 1);
    assert!((specs[0].coefficient() - 0.5).abs() < 1.0e-12);
    assert!((specs[1].coefficient() - 0.866_025_403_784_438_6).abs() < 1.0e-12);

    tensorcontract_fusion_into(
        &rule,
        &mut dst,
        &lhs,
        &rhs,
        TensorContractAxisSpec::new(&[], &[], AxisPermutation::from_axes(&[0, 2, 1, 3])),
        2.0,
        3.0,
    )
    .unwrap();

    assert!((dst.data()[0] - 53.0).abs() < 1.0e-12);
    assert!((dst.data()[1] - 92.602_540_378_443_86).abs() < 1.0e-12);
}

#[test]
fn tensorcontract_fusion_explicit_output_transform_materializes_canonical_dst() {
    let rule = SU2FusionRule;
    let lhs_hom = FusionTreeHomSpace::from_sector_ids([1, 1, 1, 1], []);
    let lhs_keys = lhs_hom.fusion_tree_keys(&rule);
    assert_eq!(lhs_keys.len(), 2);
    let src_tree = lhs_keys
        .iter()
        .find(|key| key.codomain_tree().innerlines() == [SectorId::new(0), SectorId::new(1)])
        .expect("SU2 fixture should contain the reference source tree")
        .clone();
    let recoupled_tree = lhs_keys
        .iter()
        .find(|key| **key != src_tree)
        .expect("SU2 fixture should contain the recoupled output tree")
        .clone();
    let src_key = BlockKey::from(src_tree.clone());
    let dst_key0 = BlockKey::from(src_tree);
    let dst_key1 = BlockKey::from(recoupled_tree);
    let lhs_space = FusionTensorMapSpace::new(
        TensorMapSpace::<4, 0>::from_dims([1, 1, 1, 1], []).unwrap(),
        lhs_hom.clone(),
        BlockStructure::packed_column_major_with_keys(4, [(src_key, vec![1, 1, 1, 1])]).unwrap(),
    )
    .unwrap();
    let rhs_space = FusionTensorMapSpace::from_degeneracy_shapes(
        TensorMapSpace::<0, 0>::from_dims([], []).unwrap(),
        FusionTreeHomSpace::from_sector_ids([], []),
        &rule,
        [vec![]],
    )
    .unwrap();
    let dst_space = FusionTensorMapSpace::new(
        TensorMapSpace::<4, 0>::from_dims([1, 1, 1, 1], []).unwrap(),
        lhs_hom,
        BlockStructure::packed_column_major_with_keys(
            4,
            [(dst_key0, vec![1, 1, 1, 1]), (dst_key1, vec![1, 1, 1, 1])],
        )
        .unwrap(),
    )
    .unwrap();
    let lhs_canonical_space = lhs_space.clone();
    let canonical_dst_space = lhs_space.clone();
    let rhs_canonical_space = rhs_space.clone();
    let lhs = TensorMap::<f64, 4, 0>::from_vec_with_fusion_space(vec![10.0], lhs_space).unwrap();
    let rhs = TensorMap::<f64, 0, 0>::from_vec_with_fusion_space(vec![5.0], rhs_space).unwrap();
    let mut expected_dst =
        TensorMap::<f64, 4, 0>::from_vec_with_fusion_space(vec![1.0, 2.0], dst_space.clone())
            .unwrap();
    let mut explicit_dst =
        TensorMap::<f64, 4, 0>::from_vec_with_fusion_space(vec![1.0, 2.0], dst_space.clone())
            .unwrap();
    let mut canonical_dst = TensorMap::<f64, 4, 0>::from_vec_with_fusion_space(
        vec![999.0],
        canonical_dst_space.clone(),
    )
    .unwrap();
    let mut expected_canonical_dst =
        TensorMap::<f64, 4, 0>::from_vec_with_fusion_space(vec![-77.0], canonical_dst_space)
            .unwrap();
    let mut lhs_canonical = TensorMap::<f64, 4, 0>::from_vec_with_fusion_space(
        vec![123.0],
        lhs_canonical_space.clone(),
    )
    .unwrap();
    let mut rhs_canonical = TensorMap::<f64, 0, 0>::from_vec_with_fusion_space(
        vec![456.0],
        rhs_canonical_space.clone(),
    )
    .unwrap();
    let mut expected_lhs_canonical =
        TensorMap::<f64, 4, 0>::from_vec_with_fusion_space(vec![0.0], lhs_canonical_space).unwrap();
    let mut expected_rhs_canonical =
        TensorMap::<f64, 0, 0>::from_vec_with_fusion_space(vec![0.0], rhs_canonical_space).unwrap();
    let axes = TensorContractAxisSpec::new(&[], &[], AxisPermutation::from_axes(&[0, 2, 1, 3]));
    let plan = tensorcontract_fusion_explicit_plan(
        &rule,
        explicit_dst.fusion_space().unwrap(),
        lhs.fusion_space().unwrap(),
        rhs.fusion_space().unwrap(),
        axes,
    )
    .unwrap();
    assert_eq!(plan.canonical_dst_nout(), 4);
    assert_eq!(plan.canonical_dst_nin(), 0);
    assert_eq!(plan.canonical_axes().lhs_contracting_axes(), &[]);
    assert_eq!(plan.canonical_axes().rhs_contracting_axes(), &[]);
    assert_eq!(plan.canonical_axes().output_axes(), &[0, 1, 2, 3]);
    assert_eq!(
        plan.output_transform(),
        &TreeTransformOperationKey::permute([0, 2, 1, 3], Vec::<usize>::new())
    );

    let alpha = 2.0;
    let beta = 3.0;
    let err = tensorcontract_fusion_explicit_plan_into(
        &rule,
        &plan,
        &mut expected_dst,
        &mut expected_lhs_canonical,
        &mut expected_rhs_canonical,
        &lhs,
        &rhs,
        alpha,
        beta,
    )
    .unwrap_err();
    assert_eq!(
        err,
        OperationError::UnsupportedTensorContractScope {
            message: EXPLICIT_OUTPUT_TRANSFORM_REQUIRES_CANONICAL_DST,
        }
    );

    tree_pair_transform_into(
        &rule,
        plan.lhs_transform().clone(),
        &mut expected_lhs_canonical,
        &lhs,
        1.0,
        0.0,
    )
    .unwrap();
    tree_pair_transform_into(
        &rule,
        plan.rhs_transform().clone(),
        &mut expected_rhs_canonical,
        &rhs,
        1.0,
        0.0,
    )
    .unwrap();
    tensorcontract_fusion_into(
        &rule,
        &mut expected_canonical_dst,
        &expected_lhs_canonical,
        &expected_rhs_canonical,
        plan.canonical_axes().as_spec(),
        alpha,
        0.0,
    )
    .unwrap();
    tree_pair_transform_into(
        &rule,
        plan.output_transform().clone(),
        &mut expected_dst,
        &expected_canonical_dst,
        1.0,
        beta,
    )
    .unwrap();

    tensorcontract_fusion_explicit_plan_into_canonical_dst(
        &rule,
        &plan,
        &mut explicit_dst,
        &mut canonical_dst,
        &mut lhs_canonical,
        &mut rhs_canonical,
        &lhs,
        &rhs,
        alpha,
        beta,
    )
    .unwrap();

    assert_eq!(canonical_dst.data(), expected_canonical_dst.data());
    assert_eq!(canonical_dst.data(), &[100.0]);
    for (&actual, &expected) in explicit_dst.data().iter().zip(expected_dst.data()) {
        assert!(
            (actual - expected).abs() < 1.0e-12,
            "actual {actual} expected {expected}"
        );
    }
    assert!((explicit_dst.data()[0] - 53.0).abs() < 1.0e-12);
    assert!((explicit_dst.data()[1] - 92.602_540_378_443_86).abs() < 1.0e-12);
}

#[test]
fn tensorcontract_fusion_su2_keeps_contracted_tree_basis_with_degeneracy() {
    let rule = SU2FusionRule;
    let lhs_hom = FusionTreeHomSpace::from_sector_ids([1], [1, 1, 1]);
    let rhs_hom = FusionTreeHomSpace::from_sector_ids([1, 1, 1], [1]);
    let lhs_keys = lhs_hom.fusion_tree_keys(&rule);
    let rhs_keys = rhs_hom.fusion_tree_keys(&rule);
    assert_eq!(lhs_keys.len(), 2);
    assert_eq!(rhs_keys.len(), 2);
    assert_ne!(
        lhs_keys[0].domain_tree().innerlines()[0],
        lhs_keys[1].domain_tree().innerlines()[0]
    );
    let lhs_space = FusionTensorMapSpace::from_degeneracy_shapes(
        TensorMapSpace::<1, 3>::from_dims([2], [2, 2, 2]).unwrap(),
        lhs_hom,
        &rule,
        [vec![2, 2, 2, 2], vec![2, 2, 2, 2]],
    )
    .unwrap();
    let rhs_space = FusionTensorMapSpace::from_degeneracy_shapes(
        TensorMapSpace::<3, 1>::from_dims([2, 2, 2], [2]).unwrap(),
        rhs_hom,
        &rule,
        [vec![2, 2, 2, 2], vec![2, 2, 2, 2]],
    )
    .unwrap();
    let dst_hom = FusionTreeHomSpace::from_sector_ids([1], [1]);
    let dst_keys = dst_hom.fusion_tree_keys(&rule);
    assert_eq!(dst_keys.len(), 1);
    let dst_space = FusionTensorMapSpace::new(
        TensorMapSpace::<1, 1>::from_dims([2], [2]).unwrap(),
        dst_hom,
        BlockStructure::packed_column_major_with_keys(2, [(dst_keys[0].clone(), vec![2, 2])])
            .unwrap(),
    )
    .unwrap();
    let lhs_data = (0..32).map(|index| 0.25 + index as f64).collect::<Vec<_>>();
    let rhs_data = (0..32)
        .map(|index| 10.0 - 0.5 * index as f64)
        .collect::<Vec<_>>();
    let initial_dst = vec![1.0, -2.0, 3.0, -4.0];
    let lhs = TensorMap::<f64, 1, 3>::from_vec_with_fusion_space(lhs_data, lhs_space).unwrap();
    let rhs = TensorMap::<f64, 3, 1>::from_vec_with_fusion_space(rhs_data, rhs_space).unwrap();
    let mut dst =
        TensorMap::<f64, 1, 1>::from_vec_with_fusion_space(initial_dst.clone(), dst_space).unwrap();
    let axes = TensorContractAxisSpec::canonical(&[1, 2, 3], &[0, 1, 2]);
    let specs = tensorcontract_fusion_block_specs(
        &rule,
        dst.fusion_space().unwrap(),
        lhs.fusion_space().unwrap(),
        rhs.fusion_space().unwrap(),
        axes,
    )
    .unwrap();

    assert_eq!(specs.len(), 2);
    for spec in &specs {
        let lhs_key = match lhs.structure().block(spec.lhs_block()).unwrap().key() {
            BlockKey::FusionTree(key) => key,
            BlockKey::Dense => panic!("expected lhs fusion-tree block"),
        };
        let rhs_key = match rhs.structure().block(spec.rhs_block()).unwrap().key() {
            BlockKey::FusionTree(key) => key,
            BlockKey::Dense => panic!("expected rhs fusion-tree block"),
        };
        assert_eq!(
            lhs_key.domain_tree().innerlines()[0],
            rhs_key.codomain_tree().innerlines()[0],
            "contracted SU2 tree basis must not cross-contract"
        );
    }

    let alpha = 1.25;
    let beta = -0.5;
    let mut expected = initial_dst
        .into_iter()
        .map(|value| beta * value)
        .collect::<Vec<_>>();
    for spec in &specs {
        let lhs_offset = lhs.structure().block(spec.lhs_block()).unwrap().offset();
        let rhs_offset = rhs.structure().block(spec.rhs_block()).unwrap().offset();
        for lhs_open in 0..2 {
            for rhs_open in 0..2 {
                let mut sum = 0.0;
                for a in 0..2 {
                    for b in 0..2 {
                        for c in 0..2 {
                            let lhs_index = lhs_offset + lhs_open + 2 * a + 4 * b + 8 * c;
                            let rhs_index = rhs_offset + a + 2 * b + 4 * c + 8 * rhs_open;
                            sum += lhs.data()[lhs_index] * rhs.data()[rhs_index];
                        }
                    }
                }
                expected[lhs_open + 2 * rhs_open] += alpha * spec.coefficient() * sum;
            }
        }
    }

    tensorcontract_fusion_into(&rule, &mut dst, &lhs, &rhs, axes, alpha, beta).unwrap();

    for (&actual, expected) in dst.data().iter().zip(expected) {
        assert!(
            (actual - expected).abs() < 1.0e-10,
            "actual {actual} expected {expected}"
        );
    }
}

#[test]
fn contracted_fusion_tree_basis_matches_dual_u1_labels_and_flags() {
    let rule = U1FusionRule;
    let plus_two = U1Irrep::new(2).sector_id();
    let minus_two = U1Irrep::new(-2).sector_id();
    let lhs_domain = FusionTreeKey::new(
        [plus_two],
        Some(plus_two),
        [false],
        Vec::<SectorId>::new(),
        Vec::<SectorId>::new(),
    );
    let rhs_codomain = FusionTreeKey::new(
        [minus_two],
        Some(minus_two),
        [false],
        Vec::<SectorId>::new(),
        Vec::<SectorId>::new(),
    );
    assert!(contracted_fusion_tree_basis_matches(
        &rule,
        &lhs_domain,
        &rhs_codomain
    ));

    let raw_rhs_codomain = FusionTreeKey::new(
        [plus_two],
        Some(plus_two),
        [false],
        Vec::<SectorId>::new(),
        Vec::<SectorId>::new(),
    );
    assert!(!contracted_fusion_tree_basis_matches(
        &rule,
        &lhs_domain,
        &raw_rhs_codomain
    ));

    let dual_flag_rhs_codomain = FusionTreeKey::new(
        [minus_two],
        Some(minus_two),
        [true],
        Vec::<SectorId>::new(),
        Vec::<SectorId>::new(),
    );
    assert!(!contracted_fusion_tree_basis_matches(
        &rule,
        &lhs_domain,
        &dual_flag_rhs_codomain
    ));
}

#[test]
fn tensorcontract_fusion_noncanonical_su2_requires_explicit_transform_sequence() {
    let rule = SU2FusionRule;
    let lhs_hom = FusionTreeHomSpace::from_sector_ids([1, 1, 1], [1]);
    let rhs_hom = FusionTreeHomSpace::from_sector_ids([1], [1, 1, 1]);
    let axes = TensorContractAxisSpec::canonical(&[0, 1, 2], &[1, 2, 3]);
    let output_axes = [0, 1];
    let lhs_canonical_hom = lhs_hom
        .permute(&rule, &[3], &[0, 1, 2])
        .expect("valid lhs canonical tree-pair transform");
    let rhs_canonical_hom = rhs_hom
        .permute(&rule, &[1, 2, 3], &[0])
        .expect("valid rhs canonical tree-pair transform");
    let dst_hom = FusionTreeHomSpace::tensorcontract_homspace(
        &rule,
        &lhs_hom,
        &rhs_hom,
        axes.lhs_contracting_axes(),
        axes.rhs_contracting_axes(),
        &output_axes,
        1,
    )
    .unwrap();

    let lhs_space = FusionTensorMapSpace::from_degeneracy_shapes(
        TensorMapSpace::<3, 1>::from_dims([2, 2, 2], [2]).unwrap(),
        lhs_hom,
        &rule,
        [vec![2, 2, 2, 2], vec![2, 2, 2, 2]],
    )
    .unwrap();
    let rhs_space = FusionTensorMapSpace::from_degeneracy_shapes(
        TensorMapSpace::<1, 3>::from_dims([2], [2, 2, 2]).unwrap(),
        rhs_hom,
        &rule,
        [vec![2, 2, 2, 2], vec![2, 2, 2, 2]],
    )
    .unwrap();
    let lhs_canonical_space = FusionTensorMapSpace::from_degeneracy_shapes(
        TensorMapSpace::<1, 3>::from_dims([2], [2, 2, 2]).unwrap(),
        lhs_canonical_hom,
        &rule,
        [vec![2, 2, 2, 2], vec![2, 2, 2, 2]],
    )
    .unwrap();
    let rhs_canonical_space = FusionTensorMapSpace::from_degeneracy_shapes(
        TensorMapSpace::<3, 1>::from_dims([2, 2, 2], [2]).unwrap(),
        rhs_canonical_hom,
        &rule,
        [vec![2, 2, 2, 2], vec![2, 2, 2, 2]],
    )
    .unwrap();
    let dst_space = FusionTensorMapSpace::from_degeneracy_shapes(
        TensorMapSpace::<1, 1>::from_dims([2], [2]).unwrap(),
        dst_hom,
        &rule,
        [vec![2, 2]],
    )
    .unwrap();
    let lhs_data = (0..32)
        .map(|index| 1.0 + 0.125 * index as f64)
        .collect::<Vec<_>>();
    let rhs_data = (0..32)
        .map(|index| -3.0 + 0.25 * index as f64)
        .collect::<Vec<_>>();
    let initial_dst = vec![2.0, -1.0, 4.0, -3.0];
    let initial_dst_for_explicit = initial_dst.clone();
    let lhs = TensorMap::<f64, 3, 1>::from_vec_with_fusion_space(lhs_data, lhs_space).unwrap();
    let rhs = TensorMap::<f64, 1, 3>::from_vec_with_fusion_space(rhs_data, rhs_space).unwrap();
    let mut direct_dst =
        TensorMap::<f64, 1, 1>::from_vec_with_fusion_space(initial_dst.clone(), dst_space.clone())
            .unwrap();
    let mut expected_dst =
        TensorMap::<f64, 1, 1>::from_vec_with_fusion_space(initial_dst, dst_space.clone()).unwrap();
    let mut lhs_canonical = TensorMap::<f64, 1, 3>::from_vec_with_fusion_space(
        vec![0.0; lhs_canonical_space.required_len().unwrap()],
        lhs_canonical_space.clone(),
    )
    .unwrap();
    let mut rhs_canonical = TensorMap::<f64, 3, 1>::from_vec_with_fusion_space(
        vec![0.0; rhs_canonical_space.required_len().unwrap()],
        rhs_canonical_space.clone(),
    )
    .unwrap();
    let plan = tensorcontract_fusion_explicit_plan(
        &rule,
        direct_dst.fusion_space().unwrap(),
        lhs.fusion_space().unwrap(),
        rhs.fusion_space().unwrap(),
        axes,
    )
    .unwrap();
    assert_eq!(
        plan.lhs_transform(),
        &TreeTransformOperationKey::permute([3], [0, 1, 2])
    );
    assert_eq!(
        plan.rhs_transform(),
        &TreeTransformOperationKey::permute([1, 2, 3], [0])
    );
    assert_eq!(plan.canonical_dst_nout(), 1);
    assert_eq!(plan.canonical_dst_nin(), 1);
    assert_eq!(plan.canonical_axes().lhs_contracting_axes(), &[1, 2, 3]);
    assert_eq!(plan.canonical_axes().rhs_contracting_axes(), &[0, 1, 2]);
    assert_eq!(plan.canonical_axes().output_axes(), &[0, 1]);
    assert_eq!(
        plan.output_transform(),
        &TreeTransformOperationKey::permute([0], [1])
    );

    let err = tensorcontract_fusion_block_specs(
        &rule,
        direct_dst.fusion_space().unwrap(),
        lhs.fusion_space().unwrap(),
        rhs.fusion_space().unwrap(),
        axes,
    )
    .unwrap_err();
    assert_eq!(
            err,
            OperationError::UnsupportedTensorContractScope {
                message: "fusion contraction requiring source tree-pair transforms is not implemented; pre-transform operands explicitly",
            }
        );

    tree_pair_transform_into(
        &rule,
        TreeTransformOperationKey::permute([3], [0, 1, 2]),
        &mut lhs_canonical,
        &lhs,
        1.0,
        0.0,
    )
    .unwrap();
    tree_pair_transform_into(
        &rule,
        TreeTransformOperationKey::permute([1, 2, 3], [0]),
        &mut rhs_canonical,
        &rhs,
        1.0,
        0.0,
    )
    .unwrap();
    let canonical_specs = tensorcontract_fusion_block_specs(
        &rule,
        expected_dst.fusion_space().unwrap(),
        lhs_canonical.fusion_space().unwrap(),
        rhs_canonical.fusion_space().unwrap(),
        TensorContractAxisSpec::canonical(&[1, 2, 3], &[0, 1, 2]),
    )
    .unwrap();
    assert_eq!(canonical_specs.len(), 2);

    let alpha = -1.5;
    let beta = 0.25;
    let err = tensorcontract_fusion_into(&rule, &mut direct_dst, &lhs, &rhs, axes, alpha, beta)
        .unwrap_err();
    assert_eq!(
            err,
            OperationError::UnsupportedTensorContractScope {
                message: "fusion contraction requiring source tree-pair transforms is not implemented; pre-transform operands explicitly",
            }
        );
    tensorcontract_fusion_into(
        &rule,
        &mut expected_dst,
        &lhs_canonical,
        &rhs_canonical,
        TensorContractAxisSpec::canonical(&[1, 2, 3], &[0, 1, 2]),
        alpha,
        beta,
    )
    .unwrap();

    assert_eq!(direct_dst.data(), &[2.0, -1.0, 4.0, -3.0]);
    assert_ne!(expected_dst.data(), direct_dst.data());

    let mut explicit_dst =
        TensorMap::<f64, 1, 1>::from_vec_with_fusion_space(initial_dst_for_explicit, dst_space)
            .unwrap();
    tensorcontract_fusion_explicit_plan_into(
        &rule,
        &plan,
        &mut explicit_dst,
        &mut lhs_canonical,
        &mut rhs_canonical,
        &lhs,
        &rhs,
        alpha,
        beta,
    )
    .unwrap();
    for (&actual, &expected) in explicit_dst.data().iter().zip(expected_dst.data()) {
        assert!(
            (actual - expected).abs() < 1.0e-10,
            "actual {actual} expected {expected}"
        );
    }
}

#[test]
fn tensorcontract_fusion_product_noncanonical_requires_explicit_transform() {
    let (rule, src_space, dst_space, _) = fz2_u1_su2_tree_pair_fixture();
    let scalar_key = BlockKey::from(FusionTreeBlockKey::pair(
        empty_fusion_tree(),
        empty_fusion_tree(),
    ));
    let rhs_space = FusionTensorMapSpace::new(
        TensorMapSpace::<0, 0>::from_dims([], []).unwrap(),
        FusionTreeHomSpace::from_sector_ids([], []),
        BlockStructure::packed_column_major_with_keys(0, [(scalar_key, vec![])]).unwrap(),
    )
    .unwrap();
    let lhs = TensorMap::<Complex64, 2, 1>::from_vec_with_fusion_space(
        vec![Complex64::new(1.0, 2.0), Complex64::new(3.0, -1.0)],
        src_space,
    )
    .unwrap();
    let rhs = TensorMap::<Complex64, 0, 0>::from_vec_with_fusion_space(
        vec![Complex64::new(2.0, 0.5)],
        rhs_space,
    )
    .unwrap();
    let mut dst = TensorMap::<Complex64, 2, 1>::from_vec_with_fusion_space(
        vec![Complex64::new(5.0, 1.0), Complex64::new(-2.0, 4.0)],
        dst_space,
    )
    .unwrap();
    let axes = TensorContractAxisSpec::new(&[], &[], AxisPermutation::from_axes(&[1, 0, 2]));
    let err = tensorcontract_fusion_block_specs(
        &rule,
        dst.fusion_space().unwrap(),
        lhs.fusion_space().unwrap(),
        rhs.fusion_space().unwrap(),
        axes,
    )
    .unwrap_err();
    assert_eq!(
            err,
            OperationError::UnsupportedTensorContractScope {
                message: "fusion contraction requiring source tree-pair transforms is not implemented; pre-transform operands explicitly",
            }
        );

    let err = tensorcontract_fusion_into(
        &rule,
        &mut dst,
        &lhs,
        &rhs,
        axes,
        Complex64::new(2.0, 0.0),
        Complex64::new(3.0, 0.0),
    )
    .unwrap_err();

    assert_eq!(
            err,
            OperationError::UnsupportedTensorContractScope {
                message: "fusion contraction requiring source tree-pair transforms is not implemented; pre-transform operands explicitly",
            }
        );
}

#[test]
fn tree_transform_rejects_invalid_block_specs_at_compile_time() {
    let space = TensorMapSpace::<2, 0>::from_dims([4, 2], []).unwrap();
    let structure = BlockStructure::packed_column_major(2, [vec![2, 2], vec![2, 2]]).unwrap();
    let src = TensorMap::<f64, 2, 0>::from_vec_with_structure(
        vec![1.0; 8],
        space.clone(),
        structure.clone(),
    )
    .unwrap();
    let dst =
        TensorMap::<f64, 2, 0>::from_vec_with_structure(vec![0.0; 8], space, structure).unwrap();

    let err = TreeTransformStructure::compile(
        &dst,
        &src,
        &[TreeTransformBlockSpec::multi(
            vec![0, 1],
            vec![0, 1],
            vec![1.0, 2.0],
        )],
    )
    .unwrap_err();
    assert_eq!(
        err,
        OperationError::CoefficientCountMismatch {
            expected: 4,
            actual: 2,
        }
    );

    let err = TreeTransformStructure::compile(
        &dst,
        &src,
        &[
            TreeTransformBlockSpec::single(0, 0, 1.0),
            TreeTransformBlockSpec::single(0, 1, 1.0),
        ],
    )
    .unwrap_err();
    assert_eq!(
        err,
        OperationError::DuplicateTransformDestination { dst_block: 0 }
    );
}

#[test]
fn tree_transform_compile_keyed_rejects_missing_tree_block_key() {
    let src_space = TensorMapSpace::<2, 0>::from_dims([2, 2], []).unwrap();
    let dst_space = TensorMapSpace::<2, 0>::from_dims([2, 2], []).unwrap();
    let key1 = BlockKey::sector_ids([1]);
    let key2 = BlockKey::sector_ids([2]);
    let src_structure =
        BlockStructure::packed_column_major_with_keys(2, [(key1.clone(), vec![2, 2])]).unwrap();
    let dst_structure =
        BlockStructure::packed_column_major_with_keys(2, [(key1.clone(), vec![2, 2])]).unwrap();
    let src =
        TensorMap::<f64, 2, 0>::from_vec_with_structure(vec![1.0; 4], src_space, src_structure)
            .unwrap();
    let dst =
        TensorMap::<f64, 2, 0>::from_vec_with_structure(vec![0.0; 4], dst_space, dst_structure)
            .unwrap();

    let err = TreeTransformStructure::compile_keyed(
        &dst,
        &src,
        &[TreeTransformKeyBlockSpec::single(key2.clone(), key1, 1.0)],
    )
    .unwrap_err();

    assert_eq!(err, OperationError::MissingBlockKey { key: key2 });
}

#[test]
fn tree_transform_group_block_spec_preserves_group_identity_and_ordered_keys() {
    let group_key = FusionTreeGroupKey::from_sector_ids([10, 20], [30], [false, true], [true]);
    let dst_key1 = BlockKey::sector_ids([101, 201]);
    let dst_key2 = BlockKey::sector_ids([102, 202]);
    let src_key = BlockKey::sector_ids([301, 401]);
    let spec = TreeTransformGroupBlockSpec::multi(
        group_key.clone(),
        [dst_key1.clone(), dst_key2.clone()],
        [src_key.clone()],
        vec![2.0_f64, 3.0],
    );

    assert_eq!(spec.group_key(), &group_key);
    assert_eq!(
        spec.group_key()
            .codomain_uncoupled()
            .iter()
            .map(|sector| sector.id())
            .collect::<Vec<_>>(),
        vec![10, 20]
    );
    assert_eq!(
        spec.group_key()
            .domain_uncoupled()
            .iter()
            .map(|sector| sector.id())
            .collect::<Vec<_>>(),
        vec![30]
    );
    assert_eq!(spec.group_key().codomain_is_dual(), &[false, true]);
    assert_eq!(spec.group_key().domain_is_dual(), &[true]);
    assert_eq!(spec.dst_keys(), &[dst_key1, dst_key2]);
    assert_eq!(spec.src_keys(), &[src_key]);
    assert_eq!(spec.coefficients_src_by_dst(), &[2.0, 3.0]);
}

#[test]
fn unique_tree_transform_plan_builder_creates_single_specs_in_source_order() {
    let src_key1 = fusion_tree_test_key([1, 0], [1], 1, [false, false], [false]);
    let src_key2 = fusion_tree_test_key([0, 1], [1], 1, [false, false], [false]);
    let dst_key1 = fusion_tree_test_key([0, 1], [1], 1, [false, false], [false]);
    let dst_key2 = fusion_tree_test_key([1, 0], [1], 1, [false, false], [false]);
    let src_tree1 = expect_tree_key(&src_key1);
    let src_tree2 = expect_tree_key(&src_key2);
    let dst_tree1 = expect_tree_key(&dst_key1);
    let dst_tree2 = expect_tree_key(&dst_key2);
    let src_structure = BlockStructure::packed_column_major_with_keys(
        2,
        [
            (src_key1.clone(), vec![1, 1]),
            (src_key2.clone(), vec![1, 1]),
        ],
    )
    .unwrap();

    let plan = build_unique_tree_transform_group_plan(
        &UniqueZ2Rule,
        TreeTransformOperationKey::transpose([1, 0], [0]),
        &src_structure,
        |src| {
            if src == &src_tree1 {
                Ok((dst_tree1.clone(), 2.0_f64))
            } else if src == &src_tree2 {
                Ok((dst_tree2.clone(), 3.0_f64))
            } else {
                panic!("unexpected source key {src:?}")
            }
        },
    )
    .unwrap();

    assert_eq!(plan.specs().len(), 2);
    assert_eq!(plan.specs()[0].group_key(), &src_tree1.group_key());
    assert_eq!(plan.specs()[0].src_keys(), &[src_key1]);
    assert_eq!(plan.specs()[0].dst_keys(), &[dst_key1]);
    assert_eq!(plan.specs()[0].coefficients_src_by_dst(), &[2.0]);
    assert_eq!(plan.specs()[1].group_key(), &src_tree2.group_key());
    assert_eq!(plan.specs()[1].src_keys(), &[src_key2]);
    assert_eq!(plan.specs()[1].dst_keys(), &[dst_key2]);
    assert_eq!(plan.specs()[1].coefficients_src_by_dst(), &[3.0]);
}

#[test]
fn single_output_unique_tree_transform_helper_rejects_simple_fusion() {
    let src_key = fusion_tree_test_key([1, 1, 1], [1], 1, [false, false, false], [false]);
    let src_structure =
        BlockStructure::packed_column_major_with_keys(4, [(src_key, vec![1, 1, 1, 1])]).unwrap();
    let operation = TreeTransformOperationKey::transpose([2, 1, 0], [0]);

    let err = build_unique_tree_transform_group_plan(
        &SimpleSu2Rule,
        operation.clone(),
        &src_structure,
        |_| -> Result<(FusionTreeBlockKey, f64), OperationError> {
            unreachable!("non-Unique fusion must be rejected before transforming keys")
        },
    )
    .unwrap_err();

    assert_eq!(
        err,
        OperationError::UnsupportedFusionStyle {
            operation,
            style: FusionStyleKind::Simple,
        }
    );
}

#[test]
fn tree_transform_plan_builder_accepts_simple_multi_destination_callback() {
    let src_key0 = all_codomain_fusion_tree_test_key(
        [1, 1, 1, 1],
        Some(0),
        [false, false, false, false],
        [0, 1],
        [1, 1, 1],
    );
    let src_key1 = all_codomain_fusion_tree_test_key(
        [1, 1, 1, 1],
        Some(0),
        [false, false, false, false],
        [2, 1],
        [1, 1, 1],
    );
    let src_tree0 = expect_tree_key(&src_key0);
    let src_tree1 = expect_tree_key(&src_key1);
    let src_structure = BlockStructure::packed_column_major_with_keys(
        4,
        [
            (src_key0.clone(), vec![1, 1, 1, 1]),
            (src_key1.clone(), vec![1, 1, 1, 1]),
        ],
    )
    .unwrap();
    let operation = TreeTransformOperationKey::braid([0, 2, 1, 3], [], [0, 1, 2, 3], []);

    let plan = build_tree_transform_group_plan(&SimpleSu2Rule, operation, &src_structure, |src| {
        if src == &src_tree0 {
            Ok(vec![
                (src_tree0.clone(), 0.5_f64),
                (src_tree1.clone(), 0.866_025_403_784_438_6),
            ])
        } else if src == &src_tree1 {
            Ok(vec![
                (src_tree0.clone(), 0.866_025_403_784_438_6),
                (src_tree1.clone(), -0.5),
            ])
        } else {
            panic!("unexpected source key {src:?}")
        }
    })
    .unwrap();

    assert_eq!(plan.specs().len(), 1);
    let spec = &plan.specs()[0];
    assert_eq!(spec.src_keys(), &[src_key0.clone(), src_key1.clone()]);
    assert_eq!(spec.dst_keys(), &[src_key0, src_key1]);
    assert_eq!(
        spec.coefficients_src_by_dst(),
        &[0.5, 0.866_025_403_784_438_6, 0.866_025_403_784_438_6, -0.5]
    );
}

#[test]
fn multiplicity_free_su2_plan_builder_creates_generic_recoupling_block() {
    let src_key0 = all_codomain_fusion_tree_test_key(
        [1, 1, 1, 1],
        Some(0),
        [false, false, false, false],
        [0, 1],
        [1, 1, 1],
    );
    let src_key1 = all_codomain_fusion_tree_test_key(
        [1, 1, 1, 1],
        Some(0),
        [false, false, false, false],
        [2, 1],
        [1, 1, 1],
    );
    let src_structure = BlockStructure::packed_column_major_with_keys(
        4,
        [
            (src_key0.clone(), vec![1, 1, 1, 1]),
            (src_key1.clone(), vec![1, 1, 1, 1]),
        ],
    )
    .unwrap();
    let operation = TreeTransformOperationKey::braid([0, 2, 1, 3], [], [0, 1, 2, 3], []);

    let plan =
        build_all_codomain_tree_transform_group_plan(&SU2FusionRule, operation, &src_structure)
            .unwrap();

    assert_eq!(plan.specs().len(), 1);
    let spec = &plan.specs()[0];
    assert_eq!(spec.src_keys(), &[src_key0.clone(), src_key1.clone()]);
    assert_eq!(spec.dst_keys(), &[src_key0, src_key1]);
    let expected = [0.5, 0.866_025_403_784_438_6, 0.866_025_403_784_438_6, -0.5];
    assert_eq!(spec.coefficients_src_by_dst().len(), expected.len());
    for (&actual, expected) in spec.coefficients_src_by_dst().iter().zip(expected) {
        assert!(
            (actual - expected).abs() < 1.0e-12,
            "coefficient {actual} != {expected}"
        );
    }

    let src_space = TensorMapSpace::<4, 0>::from_dims([1, 1, 1, 1], []).unwrap();
    let dst_space = TensorMapSpace::<4, 0>::from_dims([1, 1, 1, 1], []).unwrap();
    let src = TensorMap::<f64, 4, 0>::from_vec_with_structure(
        vec![10.0, 20.0],
        src_space,
        src_structure.clone(),
    )
    .unwrap();
    let mut dst = TensorMap::<f64, 4, 0>::from_vec_with_structure(
        vec![0.0, 0.0],
        dst_space,
        src_structure.clone(),
    )
    .unwrap();
    let structure = plan
        .compile_structures(&src_structure, &src_structure)
        .unwrap();
    let mut backend = DenseTreeTransformOperations::default();
    let mut workspace = TreeTransformWorkspace::default();

    tree_transform_execute_with(
        &mut backend,
        &mut workspace,
        &structure,
        &mut dst,
        &src,
        1.0,
        0.0,
    )
    .unwrap();

    assert!(structure.has_pack_gemm_scatter_blocks());
    assert!((dst.data()[0] - 22.320_508_075_688_77).abs() < 1.0e-12);
    assert!((dst.data()[1] + 1.339_745_962_155_612_7).abs() < 1.0e-12);
}

#[test]
fn tree_pair_transform_public_helper_executes_su2_recoupling_block() {
    let src_key0 = all_codomain_fusion_tree_test_key(
        [1, 1, 1, 1],
        Some(0),
        [false, false, false, false],
        [0, 1],
        [1, 1, 1],
    );
    let src_key1 = all_codomain_fusion_tree_test_key(
        [1, 1, 1, 1],
        Some(0),
        [false, false, false, false],
        [2, 1],
        [1, 1, 1],
    );
    let structure = BlockStructure::packed_column_major_with_keys(
        4,
        [
            (src_key0.clone(), vec![1, 1, 1, 1]),
            (src_key1.clone(), vec![1, 1, 1, 1]),
        ],
    )
    .unwrap();
    let src_space = TensorMapSpace::<4, 0>::from_dims([1, 1, 1, 1], []).unwrap();
    let dst_space = TensorMapSpace::<4, 0>::from_dims([1, 1, 1, 1], []).unwrap();
    let src = TensorMap::<f64, 4, 0>::from_vec_with_structure(
        vec![10.0, 20.0],
        src_space,
        structure.clone(),
    )
    .unwrap();
    let mut dst =
        TensorMap::<f64, 4, 0>::from_vec_with_structure(vec![0.0, 0.0], dst_space, structure)
            .unwrap();
    let operation = TreeTransformOperationKey::braid([0, 2, 1, 3], [], [0, 1, 2, 3], []);

    let compiled =
        tree_pair_transform_structure(&SU2FusionRule, operation.clone(), &dst, &src).unwrap();
    assert!(compiled.has_pack_gemm_scatter_blocks());
    tree_pair_transform_into(&SU2FusionRule, operation, &mut dst, &src, 1.0, 0.0).unwrap();

    assert!((dst.data()[0] - 22.320_508_075_688_77).abs() < 1.0e-12);
    assert!((dst.data()[1] + 1.339_745_962_155_612_7).abs() < 1.0e-12);
}

#[test]
fn tree_pair_transform_structure_replays_su2_recoupling_without_recompiling() {
    let src_key0 = all_codomain_fusion_tree_test_key(
        [1, 1, 1, 1],
        Some(0),
        [false, false, false, false],
        [0, 1],
        [1, 1, 1],
    );
    let src_key1 = all_codomain_fusion_tree_test_key(
        [1, 1, 1, 1],
        Some(0),
        [false, false, false, false],
        [2, 1],
        [1, 1, 1],
    );
    let block_structure = BlockStructure::packed_column_major_with_keys(
        4,
        [
            (src_key0.clone(), vec![1, 1, 1, 1]),
            (src_key1.clone(), vec![1, 1, 1, 1]),
        ],
    )
    .unwrap();
    let src_space = TensorMapSpace::<4, 0>::from_dims([1, 1, 1, 1], []).unwrap();
    let dst_space = TensorMapSpace::<4, 0>::from_dims([1, 1, 1, 1], []).unwrap();
    let mut src = TensorMap::<f64, 4, 0>::from_vec_with_structure(
        vec![10.0, 20.0],
        src_space,
        block_structure.clone(),
    )
    .unwrap();
    let mut dst =
        TensorMap::<f64, 4, 0>::from_vec_with_structure(vec![0.0, 0.0], dst_space, block_structure)
            .unwrap();
    let operation = TreeTransformOperationKey::braid([0, 2, 1, 3], [], [0, 1, 2, 3], []);
    let structure = tree_pair_transform_structure(&SU2FusionRule, operation, &dst, &src).unwrap();
    let mut backend = DenseTreeTransformOperations::default();
    let mut workspace = TreeTransformWorkspace::default();
    let expected = |initial: [f64; 2], source: [f64; 2], alpha: f64, beta: f64| {
        let c = 0.866_025_403_784_438_6;
        [
            beta * initial[0] + alpha * (0.5 * source[0] + c * source[1]),
            beta * initial[1] + alpha * (c * source[0] - 0.5 * source[1]),
        ]
    };

    assert!(structure.has_pack_gemm_scatter_blocks());
    tree_transform_execute_with(
        &mut backend,
        &mut workspace,
        &structure,
        &mut dst,
        &src,
        1.0,
        0.0,
    )
    .unwrap();
    let expected_first = expected([0.0, 0.0], [10.0, 20.0], 1.0, 0.0);
    assert!((dst.data()[0] - expected_first[0]).abs() < 1.0e-12);
    assert!((dst.data()[1] - expected_first[1]).abs() < 1.0e-12);
    assert_eq!(workspace.source_len(), 2);
    assert_eq!(workspace.destination_len(), 2);

    src.data_mut().copy_from_slice(&[3.0, -4.0]);
    dst.data_mut().copy_from_slice(&[1.0, 2.0]);
    tree_transform_execute_with(
        &mut backend,
        &mut workspace,
        &structure,
        &mut dst,
        &src,
        2.0,
        -1.0,
    )
    .unwrap();
    let expected_second = expected([1.0, 2.0], [3.0, -4.0], 2.0, -1.0);
    assert!((dst.data()[0] - expected_second[0]).abs() < 1.0e-12);
    assert!((dst.data()[1] - expected_second[1]).abs() < 1.0e-12);
}

#[test]
fn tree_transform_cache_reuses_su2_recoupling_descriptor() {
    let src_key0 = all_codomain_fusion_tree_test_key(
        [1, 1, 1, 1],
        Some(0),
        [false, false, false, false],
        [0, 1],
        [1, 1, 1],
    );
    let src_key1 = all_codomain_fusion_tree_test_key(
        [1, 1, 1, 1],
        Some(0),
        [false, false, false, false],
        [2, 1],
        [1, 1, 1],
    );
    let block_structure = BlockStructure::packed_column_major_with_keys(
        4,
        [
            (src_key0.clone(), vec![1, 1, 1, 1]),
            (src_key1.clone(), vec![1, 1, 1, 1]),
        ],
    )
    .unwrap();
    let src_space = TensorMapSpace::<4, 0>::from_dims([1, 1, 1, 1], []).unwrap();
    let dst_space = TensorMapSpace::<4, 0>::from_dims([1, 1, 1, 1], []).unwrap();
    let src = TensorMap::<f64, 4, 0>::from_vec_with_structure(
        vec![10.0, 20.0],
        src_space,
        block_structure.clone(),
    )
    .unwrap();
    let mut dst =
        TensorMap::<f64, 4, 0>::from_vec_with_structure(vec![0.0, 0.0], dst_space, block_structure)
            .unwrap();
    let operation = TreeTransformOperationKey::braid([0, 2, 1, 3], [], [0, 1, 2, 3], []);
    let mut cache = TreeTransformCache::<f64, TreeTransformBuiltinRuleCacheKey>::new();

    {
        let structure = cache
            .get_or_compile_tree_pair(&SU2FusionRule, operation.clone(), &dst, &src)
            .unwrap();
        assert!(structure.has_pack_gemm_scatter_blocks());
    }
    assert_eq!(cache.plan_len(), 1);
    assert_eq!(cache.structure_len(), 1);

    {
        let structure = cache
            .get_or_compile_tree_pair(&SU2FusionRule, operation.clone(), &dst, &src)
            .unwrap();
        assert!(structure.has_pack_gemm_scatter_blocks());
    }
    assert_eq!(cache.plan_len(), 1);
    assert_eq!(cache.structure_len(), 1);

    let structure = cache
        .get_or_compile_tree_pair(&SU2FusionRule, operation, &dst, &src)
        .unwrap();
    let mut backend = DenseTreeTransformOperations::default();
    let mut workspace = TreeTransformWorkspace::default();
    tree_transform_execute_with(
        &mut backend,
        &mut workspace,
        structure,
        &mut dst,
        &src,
        1.0,
        0.0,
    )
    .unwrap();

    assert!((dst.data()[0] - 22.320_508_075_688_77).abs() < 1.0e-12);
    assert!((dst.data()[1] + 1.339_745_962_155_612_7).abs() < 1.0e-12);
}

#[test]
fn tree_transform_cache_reuses_all_codomain_plan_across_degeneracy_shapes() {
    let src_key0 = all_codomain_fusion_tree_test_key(
        [1, 1, 1, 1],
        Some(0),
        [false, false, false, false],
        [0, 1],
        [1, 1, 1],
    );
    let src_key1 = all_codomain_fusion_tree_test_key(
        [1, 1, 1, 1],
        Some(0),
        [false, false, false, false],
        [2, 1],
        [1, 1, 1],
    );
    let small_structure = BlockStructure::packed_column_major_with_keys(
        4,
        [
            (src_key0.clone(), vec![1, 1, 1, 1]),
            (src_key1.clone(), vec![1, 1, 1, 1]),
        ],
    )
    .unwrap();
    let large_structure = BlockStructure::packed_column_major_with_keys(
        4,
        [(src_key0, vec![2, 1, 1, 1]), (src_key1, vec![2, 1, 1, 1])],
    )
    .unwrap();
    let small_space = TensorMapSpace::<4, 0>::from_dims([1, 1, 1, 1], []).unwrap();
    let large_space = TensorMapSpace::<4, 0>::from_dims([2, 1, 1, 1], []).unwrap();
    let src = TensorMap::<f64, 4, 0>::from_vec_with_structure(
        vec![10.0, 20.0],
        small_space.clone(),
        small_structure.clone(),
    )
    .unwrap();
    let mut dst = TensorMap::<f64, 4, 0>::from_vec_with_structure(
        vec![0.0, 0.0],
        small_space,
        small_structure,
    )
    .unwrap();
    let src_large = TensorMap::<f64, 4, 0>::from_vec_with_structure(
        vec![1.0, 2.0, 3.0, 4.0],
        large_space.clone(),
        large_structure.clone(),
    )
    .unwrap();
    let dst_large = TensorMap::<f64, 4, 0>::from_vec_with_structure(
        vec![0.0, 0.0, 0.0, 0.0],
        large_space,
        large_structure,
    )
    .unwrap();
    let operation = TreeTransformOperationKey::braid([0, 2, 1, 3], [], [0, 1, 2, 3], []);
    let mut cache = TreeTransformCache::<f64, TreeTransformBuiltinRuleCacheKey>::new();

    {
        let structure = cache
            .get_or_compile_all_codomain(&SU2FusionRule, operation.clone(), &dst, &src)
            .unwrap();
        assert!(structure.has_pack_gemm_scatter_blocks());
    }
    assert_eq!(cache.plan_len(), 1);
    assert_eq!(cache.structure_len(), 1);

    {
        let structure = cache
            .get_or_compile_all_codomain(&SU2FusionRule, operation.clone(), &dst, &src)
            .unwrap();
        assert!(structure.has_pack_gemm_scatter_blocks());
    }
    assert_eq!(cache.plan_len(), 1);
    assert_eq!(cache.structure_len(), 1);

    {
        let structure = cache
            .get_or_compile_all_codomain(&SU2FusionRule, operation.clone(), &dst_large, &src_large)
            .unwrap();
        assert!(structure.has_pack_gemm_scatter_blocks());
    }
    assert_eq!(cache.plan_len(), 1);
    assert_eq!(cache.structure_len(), 2);

    let structure = cache
        .get_or_compile_all_codomain(&SU2FusionRule, operation, &dst, &src)
        .unwrap();
    let mut backend = DenseTreeTransformOperations::default();
    let mut workspace = TreeTransformWorkspace::default();
    tree_transform_execute_with(
        &mut backend,
        &mut workspace,
        structure,
        &mut dst,
        &src,
        1.0,
        0.0,
    )
    .unwrap();

    assert!((dst.data()[0] - 22.320_508_075_688_77).abs() < 1.0e-12);
    assert!((dst.data()[1] + 1.339_745_962_155_612_7).abs() < 1.0e-12);
}

#[test]
fn tree_transform_execution_context_reuses_all_codomain_cache() {
    let src_key0 = all_codomain_fusion_tree_test_key(
        [1, 1, 1, 1],
        Some(0),
        [false, false, false, false],
        [0, 1],
        [1, 1, 1],
    );
    let src_key1 = all_codomain_fusion_tree_test_key(
        [1, 1, 1, 1],
        Some(0),
        [false, false, false, false],
        [2, 1],
        [1, 1, 1],
    );
    let block_structure = BlockStructure::packed_column_major_with_keys(
        4,
        [
            (src_key0.clone(), vec![1, 1, 1, 1]),
            (src_key1.clone(), vec![1, 1, 1, 1]),
        ],
    )
    .unwrap();
    let space = TensorMapSpace::<4, 0>::from_dims([1, 1, 1, 1], []).unwrap();
    let mut src = TensorMap::<f64, 4, 0>::from_vec_with_structure(
        vec![10.0, 20.0],
        space.clone(),
        block_structure.clone(),
    )
    .unwrap();
    let mut dst =
        TensorMap::<f64, 4, 0>::from_vec_with_structure(vec![0.0, 0.0], space, block_structure)
            .unwrap();
    let operation = TreeTransformOperationKey::braid([0, 2, 1, 3], [], [0, 1, 2, 3], []);
    let mut context =
        TreeTransformExecutionContext::<f64, TreeTransformBuiltinRuleCacheKey>::default();
    assert_eq!(context.cache().stats(), TreeTransformCacheStats::default());

    all_codomain_tree_transform_into_with_context(
        &mut context,
        &SU2FusionRule,
        operation.clone(),
        &mut dst,
        &src,
        1.0,
        0.0,
    )
    .unwrap();

    assert_eq!(context.cache().plan_len(), 1);
    assert_eq!(context.cache().structure_len(), 1);
    assert_eq!(context.cache().stats().plan_hits(), 0);
    assert_eq!(context.cache().stats().plan_misses(), 1);
    assert_eq!(context.cache().stats().structure_hits(), 0);
    assert_eq!(context.cache().stats().structure_misses(), 1);
    assert!((dst.data()[0] - 22.320_508_075_688_77).abs() < 1.0e-12);
    assert!((dst.data()[1] + 1.339_745_962_155_612_7).abs() < 1.0e-12);

    src.data_mut().copy_from_slice(&[3.0, -4.0]);
    dst.data_mut().copy_from_slice(&[1.0, 2.0]);
    context
        .all_codomain_tree_transform_into(&SU2FusionRule, operation, &mut dst, &src, 2.0, -1.0)
        .unwrap();

    assert_eq!(context.cache().plan_len(), 1);
    assert_eq!(context.cache().structure_len(), 1);
    assert_eq!(context.cache().stats().plan_hits(), 1);
    assert_eq!(context.cache().stats().plan_misses(), 1);
    assert_eq!(context.cache().stats().structure_hits(), 1);
    assert_eq!(context.cache().stats().structure_misses(), 1);
    let c = 0.866_025_403_784_438_6;
    assert!((dst.data()[0] - (-1.0 + 2.0 * (0.5 * 3.0 + c * -4.0))).abs() < 1.0e-12);
    assert!((dst.data()[1] - (-2.0 + 2.0 * (c * 3.0 - 0.5 * -4.0))).abs() < 1.0e-12);
    context.cache_mut().reset_stats();
    assert_eq!(context.cache().stats(), TreeTransformCacheStats::default());
}

#[test]
fn tree_transform_execution_context_separates_tree_pair_and_all_codomain_scopes() {
    let src_key0 = all_codomain_fusion_tree_test_key(
        [1, 1, 1, 1],
        Some(0),
        [false, false, false, false],
        [0, 1],
        [1, 1, 1],
    );
    let src_key1 = all_codomain_fusion_tree_test_key(
        [1, 1, 1, 1],
        Some(0),
        [false, false, false, false],
        [2, 1],
        [1, 1, 1],
    );
    let block_structure = BlockStructure::packed_column_major_with_keys(
        4,
        [
            (src_key0.clone(), vec![1, 1, 1, 1]),
            (src_key1.clone(), vec![1, 1, 1, 1]),
        ],
    )
    .unwrap();
    let space = TensorMapSpace::<4, 0>::from_dims([1, 1, 1, 1], []).unwrap();
    let src = TensorMap::<f64, 4, 0>::from_vec_with_structure(
        vec![10.0, 20.0],
        space.clone(),
        block_structure.clone(),
    )
    .unwrap();
    let mut dst =
        TensorMap::<f64, 4, 0>::from_vec_with_structure(vec![0.0, 0.0], space, block_structure)
            .unwrap();
    let operation = TreeTransformOperationKey::braid([0, 2, 1, 3], [], [0, 1, 2, 3], []);
    let mut context =
        TreeTransformExecutionContext::<f64, TreeTransformBuiltinRuleCacheKey>::default();

    context
        .tree_pair_transform_into(&SU2FusionRule, operation.clone(), &mut dst, &src, 1.0, 0.0)
        .unwrap();
    assert_eq!(context.cache().plan_len(), 1);
    assert_eq!(context.cache().structure_len(), 1);

    dst.data_mut().copy_from_slice(&[0.0, 0.0]);
    context
        .all_codomain_tree_transform_into(&SU2FusionRule, operation, &mut dst, &src, 1.0, 0.0)
        .unwrap();

    assert_eq!(context.cache().plan_len(), 2);
    assert_eq!(context.cache().structure_len(), 2);
    assert!((dst.data()[0] - 22.320_508_075_688_77).abs() < 1.0e-12);
    assert!((dst.data()[1] + 1.339_745_962_155_612_7).abs() < 1.0e-12);
}

#[test]
fn tree_pair_plan_builder_handles_su2_one_by_one_domain_crossing() {
    let src_key = BlockKey::from(FusionTreeBlockKey::pair_from_sector_ids(
        [1],
        [1],
        Some(1),
        [false],
        [false],
        [],
        [],
        [],
        [],
    ));
    let expected_dst_key = BlockKey::from(FusionTreeBlockKey::pair_from_sector_ids(
        [1],
        [1],
        Some(1),
        [true],
        [true],
        [],
        [],
        [],
        [],
    ));
    let src_structure =
        BlockStructure::packed_column_major_with_keys(2, [(src_key.clone(), vec![1, 1])]).unwrap();
    let dst_structure =
        BlockStructure::packed_column_major_with_keys(2, [(expected_dst_key.clone(), vec![1, 1])])
            .unwrap();

    let plan = build_tree_pair_transform_group_plan(
        &SU2FusionRule,
        TreeTransformOperationKey::permute([1], [0]),
        &src_structure,
    )
    .unwrap();

    assert_eq!(plan.specs().len(), 1);
    let spec = &plan.specs()[0];
    assert_eq!(spec.src_keys(), &[src_key]);
    assert_eq!(spec.dst_keys(), &[expected_dst_key]);
    assert_eq!(spec.coefficients_src_by_dst().len(), 1);
    assert!((spec.coefficients_src_by_dst()[0] - 1.0).abs() < 1.0e-12);
    plan.compile_structures(&dst_structure, &src_structure)
        .unwrap();
}

#[test]
fn tree_pair_transform_public_helper_executes_su2_domain_crossing() {
    let src_key = BlockKey::from(FusionTreeBlockKey::pair_from_sector_ids(
        [1],
        [1],
        Some(1),
        [false],
        [false],
        [],
        [],
        [],
        [],
    ));
    let expected_dst_key = BlockKey::from(FusionTreeBlockKey::pair_from_sector_ids(
        [1],
        [1],
        Some(1),
        [true],
        [true],
        [],
        [],
        [],
        [],
    ));
    let src_structure =
        BlockStructure::packed_column_major_with_keys(2, [(src_key, vec![1, 1])]).unwrap();
    let dst_structure =
        BlockStructure::packed_column_major_with_keys(2, [(expected_dst_key.clone(), vec![1, 1])])
            .unwrap();
    let src_space = TensorMapSpace::<1, 1>::from_dims([1], [1]).unwrap();
    let dst_space = TensorMapSpace::<1, 1>::from_dims([1], [1]).unwrap();
    let src = TensorMap::<f64, 1, 1>::from_vec_with_structure(vec![7.0], src_space, src_structure)
        .unwrap();
    let mut dst =
        TensorMap::<f64, 1, 1>::from_vec_with_structure(vec![2.0], dst_space, dst_structure)
            .unwrap();
    let operation = TreeTransformOperationKey::permute([1], [0]);

    tree_pair_transform_into(&SU2FusionRule, operation, &mut dst, &src, 3.0, 5.0).unwrap();

    assert_eq!(dst.structure().block(0).unwrap().key(), &expected_dst_key);
    assert!((dst.data()[0] - 31.0).abs() < 1.0e-12);
}

#[test]
fn tree_pair_transform_public_helper_executes_su2_with_complex_data() {
    let src_key = BlockKey::from(FusionTreeBlockKey::pair_from_sector_ids(
        [1],
        [1],
        Some(1),
        [false],
        [false],
        [],
        [],
        [],
        [],
    ));
    let expected_dst_key = BlockKey::from(FusionTreeBlockKey::pair_from_sector_ids(
        [1],
        [1],
        Some(1),
        [true],
        [true],
        [],
        [],
        [],
        [],
    ));
    let src_structure =
        BlockStructure::packed_column_major_with_keys(2, [(src_key, vec![1, 1])]).unwrap();
    let dst_structure =
        BlockStructure::packed_column_major_with_keys(2, [(expected_dst_key.clone(), vec![1, 1])])
            .unwrap();
    let src_space = TensorMapSpace::<1, 1>::from_dims([1], [1]).unwrap();
    let dst_space = TensorMapSpace::<1, 1>::from_dims([1], [1]).unwrap();
    let src = TensorMap::<Complex64, 1, 1>::from_vec_with_structure(
        vec![Complex64::new(7.0, 1.0)],
        src_space,
        src_structure,
    )
    .unwrap();
    let mut dst = TensorMap::<Complex64, 1, 1>::from_vec_with_structure(
        vec![Complex64::new(2.0, -3.0)],
        dst_space,
        dst_structure,
    )
    .unwrap();
    let operation = TreeTransformOperationKey::permute([1], [0]);

    tree_pair_transform_into(
        &SU2FusionRule,
        operation,
        &mut dst,
        &src,
        Complex64::new(3.0, 0.0),
        Complex64::new(5.0, 0.0),
    )
    .unwrap();

    assert_eq!(dst.structure().block(0).unwrap().key(), &expected_dst_key);
    assert!((dst.data()[0] - Complex64::new(31.0, -12.0)).norm() < 1.0e-12);
}

#[test]
fn tree_pair_operation_key_uses_tensorkit_global_source_axes() {
    let src_key = fusion_tree_test_key([1, 0], [1], 1, [false, false], [false]);
    let src_structure =
        BlockStructure::packed_column_major_with_keys(3, [(src_key, vec![1, 1, 1])]).unwrap();

    let local_domain_identity = build_tree_pair_transform_group_plan(
        &Z2FusionRule,
        TreeTransformOperationKey::permute([1, 0], [0]),
        &src_structure,
    )
    .unwrap_err();
    assert_eq!(
        local_domain_identity,
        OperationError::Core(CoreError::InvalidPermutation {
            permutation: vec![1, 0, 0],
            rank: 3,
        })
    );

    build_tree_pair_transform_group_plan(
        &Z2FusionRule,
        TreeTransformOperationKey::permute([1, 0], [2]),
        &src_structure,
    )
    .unwrap();
}

#[test]
fn tree_pair_transform_public_helper_executes_split_changing_permute() {
    let src_key = BlockKey::from(FusionTreeBlockKey::pair_from_sector_ids(
        [1],
        [0, 1],
        Some(1),
        [false],
        [false, true],
        [],
        [],
        [],
        [1],
    ));
    let src_tree = expect_tree_key(&src_key);
    let operation = TreeTransformOperationKey::permute([0, 2], [1]);
    let (dst_tree, coefficient) =
        unique_permute_tree_pair(&Z2FusionRule, &src_tree, &[0, 2], &[1]).unwrap();
    let dst_key = BlockKey::from(dst_tree);
    let src_structure =
        BlockStructure::packed_column_major_with_keys(3, [(src_key, vec![1, 1, 1])]).unwrap();
    let dst_structure =
        BlockStructure::packed_column_major_with_keys(3, [(dst_key.clone(), vec![1, 1, 1])])
            .unwrap();
    let src_space = TensorMapSpace::<1, 2>::from_dims([1], [1, 1]).unwrap();
    let dst_space = TensorMapSpace::<2, 1>::from_dims([1, 1], [1]).unwrap();
    let src = TensorMap::<f64, 1, 2>::from_vec_with_structure(vec![7.0], src_space, src_structure)
        .unwrap();
    let mut dst =
        TensorMap::<f64, 2, 1>::from_vec_with_structure(vec![2.0], dst_space, dst_structure)
            .unwrap();

    tree_pair_transform_into(&Z2FusionRule, operation, &mut dst, &src, 3.0, 5.0).unwrap();

    assert_eq!(dst.structure().block(0).unwrap().key(), &dst_key);
    assert_eq!(dst.data(), &[3.0 * coefficient * 7.0 + 5.0 * 2.0]);
}

#[test]
fn tree_pair_transform_public_helper_compiles_against_actual_destination_structure() {
    let src_key = BlockKey::from(FusionTreeBlockKey::pair_from_sector_ids(
        [1],
        [0, 1],
        Some(1),
        [false],
        [false, true],
        [],
        [],
        [],
        [1],
    ));
    let src_tree = expect_tree_key(&src_key);
    let operation = TreeTransformOperationKey::permute([0, 2], [1]);
    let (dst_tree, _) = unique_permute_tree_pair(&Z2FusionRule, &src_tree, &[0, 2], &[1]).unwrap();
    let expected_missing = BlockKey::from(dst_tree);
    let src_structure =
        BlockStructure::packed_column_major_with_keys(3, [(src_key.clone(), vec![1, 1, 1])])
            .unwrap();
    let wrong_dst_structure =
        BlockStructure::packed_column_major_with_keys(3, [(src_key, vec![1, 1, 1])]).unwrap();
    let src_space = TensorMapSpace::<1, 2>::from_dims([1], [1, 1]).unwrap();
    let dst_space = TensorMapSpace::<2, 1>::from_dims([1, 1], [1]).unwrap();
    let src = TensorMap::<f64, 1, 2>::from_vec_with_structure(vec![7.0], src_space, src_structure)
        .unwrap();
    let dst =
        TensorMap::<f64, 2, 1>::from_vec_with_structure(vec![0.0], dst_space, wrong_dst_structure)
            .unwrap();

    let err = tree_pair_transform_structure(&Z2FusionRule, operation, &dst, &src).unwrap_err();

    assert_eq!(
        err,
        OperationError::MissingBlockKey {
            key: expected_missing,
        }
    );
}

#[test]
fn multiplicity_free_product_tree_pair_plan_builder_handles_fz2_u1_su2_blocks() {
    let (rule, src_space, dst_space, [c0, c1]) = fz2_u1_su2_tree_pair_fixture();
    let src_structure = src_space.subblock_structure();
    let dst_structure = dst_space.subblock_structure();

    let plan = build_tree_pair_transform_group_plan(
        &rule,
        TreeTransformOperationKey::permute([1, 0], [2]),
        src_structure,
    )
    .unwrap();

    assert_eq!(plan.specs().len(), 2);
    assert!((single_transform_coefficient_for_coupled(&plan, c0) - 1.0).abs() < 1.0e-12);
    assert!((single_transform_coefficient_for_coupled(&plan, c1) + 1.0).abs() < 1.0e-12);
    plan.compile_structures(dst_structure, src_structure)
        .unwrap();
}

#[test]
fn tree_pair_transform_public_helper_executes_product_fz2_u1_su2_blocks() {
    let (rule, src_space, dst_space, [c0, c1]) = fz2_u1_su2_tree_pair_fixture();
    let operation = TreeTransformOperationKey::permute([1, 0], [2]);
    let src =
        TensorMap::<f64, 2, 1>::from_vec_with_fusion_space(vec![10.0, 20.0], src_space.clone())
            .unwrap();
    let mut dst =
        TensorMap::<f64, 2, 1>::from_vec_with_fusion_space(vec![1.0, 2.0], dst_space.clone())
            .unwrap();
    let initial_dst = dst.data().to_vec();
    let plan =
        build_tree_pair_transform_group_plan(&rule, operation.clone(), src.structure()).unwrap();
    assert!((single_transform_coefficient_for_coupled(&plan, c0) - 1.0).abs() < 1.0e-12);
    assert!((single_transform_coefficient_for_coupled(&plan, c1) + 1.0).abs() < 1.0e-12);
    let mut expected = initial_dst
        .iter()
        .map(|value| 3.0 * value)
        .collect::<Vec<_>>();
    for spec in plan.specs() {
        let src_key = &spec.src_keys()[0];
        let dst_key = &spec.dst_keys()[0];
        let src_offset = src.structure().block_by_key(src_key).unwrap().offset();
        let dst_offset = dst.structure().block_by_key(dst_key).unwrap().offset();
        expected[dst_offset] += 2.0 * spec.coefficients_src_by_dst()[0] * src.data()[src_offset];
    }

    tree_pair_transform_into(&rule, operation, &mut dst, &src, 2.0, 3.0).unwrap();

    assert_eq!(dst.structure(), dst_space.subblock_structure());
    for (actual, expected) in dst.data().iter().zip(expected) {
        assert!(
            (actual - expected).abs() < 1.0e-12,
            "actual {actual} != expected {expected}"
        );
    }
}

#[test]
fn tree_pair_transform_public_helper_executes_product_with_complex_data() {
    let (rule, src_space, dst_space, [c0, c1]) = fz2_u1_su2_tree_pair_fixture();
    let operation = TreeTransformOperationKey::permute([1, 0], [2]);
    let src = TensorMap::<Complex64, 2, 1>::from_vec_with_fusion_space(
        vec![Complex64::new(10.0, 1.0), Complex64::new(20.0, -2.0)],
        src_space.clone(),
    )
    .unwrap();
    let mut dst = TensorMap::<Complex64, 2, 1>::from_vec_with_fusion_space(
        vec![Complex64::new(1.0, 3.0), Complex64::new(2.0, -4.0)],
        dst_space.clone(),
    )
    .unwrap();
    let initial_dst = dst.data().to_vec();
    let plan =
        build_tree_pair_transform_group_plan(&rule, operation.clone(), src.structure()).unwrap();
    assert!((single_transform_coefficient_for_coupled(&plan, c0) - 1.0).abs() < 1.0e-12);
    assert!((single_transform_coefficient_for_coupled(&plan, c1) + 1.0).abs() < 1.0e-12);
    let alpha = Complex64::new(2.0, 0.0);
    let beta = Complex64::new(3.0, 0.0);
    let mut expected = initial_dst
        .iter()
        .map(|value| *value * beta)
        .collect::<Vec<_>>();
    for spec in plan.specs() {
        let src_key = &spec.src_keys()[0];
        let dst_key = &spec.dst_keys()[0];
        let src_offset = src.structure().block_by_key(src_key).unwrap().offset();
        let dst_offset = dst.structure().block_by_key(dst_key).unwrap().offset();
        expected[dst_offset] = expected[dst_offset]
            + src.data()[src_offset].scale_by_coefficient(spec.coefficients_src_by_dst()[0])
                * alpha;
    }

    tree_pair_transform_into(&rule, operation, &mut dst, &src, alpha, beta).unwrap();

    assert_eq!(dst.structure(), dst_space.subblock_structure());
    assert_eq!(dst.data(), expected.as_slice());
}

#[test]
fn tree_pair_transform_structure_replays_product_without_recompiling() {
    let (rule, src_space, dst_space, [c0, c1]) = fz2_u1_su2_tree_pair_fixture();
    let operation = TreeTransformOperationKey::permute([1, 0], [2]);
    let mut src =
        TensorMap::<f64, 2, 1>::from_vec_with_fusion_space(vec![10.0, 20.0], src_space.clone())
            .unwrap();
    let mut dst =
        TensorMap::<f64, 2, 1>::from_vec_with_fusion_space(vec![1.0, 2.0], dst_space.clone())
            .unwrap();
    let plan =
        build_tree_pair_transform_group_plan(&rule, operation.clone(), src.structure()).unwrap();
    let structure = tree_pair_transform_structure(&rule, operation, &dst, &src).unwrap();
    let mut backend = DenseTreeTransformOperations::default();
    let mut workspace = TreeTransformWorkspace::default();

    assert!((single_transform_coefficient_for_coupled(&plan, c0) - 1.0).abs() < 1.0e-12);
    assert!((single_transform_coefficient_for_coupled(&plan, c1) + 1.0).abs() < 1.0e-12);
    assert_eq!(structure.block_count(), 2);
    assert!(!structure.has_pack_gemm_scatter_blocks());
    let expected_first = expected_single_tree_pair_replay(
        &plan,
        dst.structure(),
        src.structure(),
        dst.data(),
        src.data(),
        2.0,
        3.0,
    );
    tree_transform_execute_with(
        &mut backend,
        &mut workspace,
        &structure,
        &mut dst,
        &src,
        2.0,
        3.0,
    )
    .unwrap();
    for (actual, expected) in dst.data().iter().zip(expected_first) {
        assert!(
            (actual - expected).abs() < 1.0e-12,
            "actual {actual} != expected {expected}"
        );
    }
    assert_eq!(workspace.source_len(), 0);
    assert_eq!(workspace.destination_len(), 0);

    src.data_mut().copy_from_slice(&[4.0, 5.0]);
    dst.data_mut().copy_from_slice(&[6.0, 7.0]);
    let expected_second = expected_single_tree_pair_replay(
        &plan,
        dst.structure(),
        src.structure(),
        dst.data(),
        src.data(),
        -1.0,
        0.5,
    );
    tree_transform_execute_with(
        &mut backend,
        &mut workspace,
        &structure,
        &mut dst,
        &src,
        -1.0,
        0.5,
    )
    .unwrap();
    for (actual, expected) in dst.data().iter().zip(expected_second) {
        assert!(
            (actual - expected).abs() < 1.0e-12,
            "actual {actual} != expected {expected}"
        );
    }
}

#[test]
fn tree_transform_cache_reuses_product_plan_across_degeneracy_shapes() {
    let (rule, src_space, dst_space, _) = fz2_u1_su2_tree_pair_fixture();
    type RuleKey = <FpU1Su2Rule as TreeTransformRuleCacheKey>::Key;
    let operation = TreeTransformOperationKey::permute([1, 0], [2]);
    let src =
        TensorMap::<f64, 2, 1>::from_vec_with_fusion_space(vec![10.0, 20.0], src_space.clone())
            .unwrap();
    let mut dst =
        TensorMap::<f64, 2, 1>::from_vec_with_fusion_space(vec![1.0, 2.0], dst_space.clone())
            .unwrap();
    let src_large_structure =
        column_major_structure_like(src_space.subblock_structure(), vec![2, 1, 1]);
    let dst_large_structure =
        column_major_structure_like(dst_space.subblock_structure(), vec![2, 1, 1]);
    let large_space = TensorMapSpace::<2, 1>::from_dims([2, 1], [1]).unwrap();
    let src_large = TensorMap::<f64, 2, 1>::from_vec_with_structure(
        vec![1.0, 2.0, 3.0, 4.0],
        large_space.clone(),
        src_large_structure,
    )
    .unwrap();
    let dst_large = TensorMap::<f64, 2, 1>::from_vec_with_structure(
        vec![0.0, 0.0, 0.0, 0.0],
        large_space,
        dst_large_structure,
    )
    .unwrap();
    let mut cache = TreeTransformCache::<f64, RuleKey>::new();

    {
        let structure = cache
            .get_or_compile_tree_pair(&rule, operation.clone(), &dst, &src)
            .unwrap();
        assert_eq!(structure.block_count(), 2);
    }
    assert_eq!(cache.plan_len(), 1);
    assert_eq!(cache.structure_len(), 1);

    {
        let structure = cache
            .get_or_compile_tree_pair(&rule, operation.clone(), &dst, &src)
            .unwrap();
        assert_eq!(structure.block_count(), 2);
    }
    assert_eq!(cache.plan_len(), 1);
    assert_eq!(cache.structure_len(), 1);

    {
        let structure = cache
            .get_or_compile_tree_pair(&rule, operation, &dst_large, &src_large)
            .unwrap();
        assert_eq!(structure.block_count(), 2);
    }
    assert_eq!(cache.plan_len(), 1);
    assert_eq!(cache.structure_len(), 2);

    let structure = cache
        .get_or_compile_tree_pair(
            &rule,
            TreeTransformOperationKey::permute([1, 0], [2]),
            &dst,
            &src,
        )
        .unwrap();
    let plan = build_tree_pair_transform_group_plan(
        &rule,
        TreeTransformOperationKey::permute([1, 0], [2]),
        src.structure(),
    )
    .unwrap();
    let expected = expected_single_tree_pair_replay(
        &plan,
        dst.structure(),
        src.structure(),
        dst.data(),
        src.data(),
        2.0,
        3.0,
    );
    let mut backend = DenseTreeTransformOperations::default();
    let mut workspace = TreeTransformWorkspace::default();
    tree_transform_execute_with(
        &mut backend,
        &mut workspace,
        structure,
        &mut dst,
        &src,
        2.0,
        3.0,
    )
    .unwrap();
    for (actual, expected) in dst.data().iter().zip(expected) {
        assert!(
            (actual - expected).abs() < 1.0e-12,
            "actual {actual} != expected {expected}"
        );
    }
}

#[test]
fn tree_transform_execution_context_reuses_product_tree_pair_cache() {
    let (rule, src_space, dst_space, _) = fz2_u1_su2_tree_pair_fixture();
    type RuleKey = <FpU1Su2Rule as TreeTransformRuleCacheKey>::Key;
    let operation = TreeTransformOperationKey::permute([1, 0], [2]);
    let mut src =
        TensorMap::<f64, 2, 1>::from_vec_with_fusion_space(vec![10.0, 20.0], src_space.clone())
            .unwrap();
    let mut dst =
        TensorMap::<f64, 2, 1>::from_vec_with_fusion_space(vec![1.0, 2.0], dst_space.clone())
            .unwrap();
    let plan =
        build_tree_pair_transform_group_plan(&rule, operation.clone(), src.structure()).unwrap();
    let mut context = TreeTransformExecutionContext::<f64, RuleKey>::default();
    let expected_first = expected_single_tree_pair_replay(
        &plan,
        dst.structure(),
        src.structure(),
        dst.data(),
        src.data(),
        2.0,
        3.0,
    );

    context
        .tree_pair_transform_into(&rule, operation.clone(), &mut dst, &src, 2.0, 3.0)
        .unwrap();

    assert_eq!(context.cache().plan_len(), 1);
    assert_eq!(context.cache().structure_len(), 1);
    for (actual, expected) in dst.data().iter().zip(expected_first) {
        assert!(
            (actual - expected).abs() < 1.0e-12,
            "actual {actual} != expected {expected}"
        );
    }

    src.data_mut().copy_from_slice(&[4.0, 5.0]);
    dst.data_mut().copy_from_slice(&[6.0, 7.0]);
    let expected_second = expected_single_tree_pair_replay(
        &plan,
        dst.structure(),
        src.structure(),
        dst.data(),
        src.data(),
        -1.0,
        0.5,
    );
    tree_pair_transform_into_with_context(
        &mut context,
        &rule,
        operation,
        &mut dst,
        &src,
        -1.0,
        0.5,
    )
    .unwrap();

    assert_eq!(context.cache().plan_len(), 1);
    assert_eq!(context.cache().structure_len(), 1);
    for (actual, expected) in dst.data().iter().zip(expected_second) {
        assert!(
            (actual - expected).abs() < 1.0e-12,
            "actual {actual} != expected {expected}"
        );
    }
}

#[test]
fn tree_transform_execution_context_misses_on_different_tree_pair_operation() {
    let (rule, src_space, dst_space, _) = fz2_u1_su2_tree_pair_fixture();
    type RuleKey = <FpU1Su2Rule as TreeTransformRuleCacheKey>::Key;
    let src =
        TensorMap::<f64, 2, 1>::from_vec_with_fusion_space(vec![10.0, 20.0], src_space.clone())
            .unwrap();
    let mut dst =
        TensorMap::<f64, 2, 1>::from_vec_with_fusion_space(vec![1.0, 2.0], dst_space.clone())
            .unwrap();
    let mut context = TreeTransformExecutionContext::<f64, RuleKey>::default();

    context
        .tree_pair_transform_into(
            &rule,
            TreeTransformOperationKey::permute([1, 0], [2]),
            &mut dst,
            &src,
            1.0,
            0.0,
        )
        .unwrap();
    assert_eq!(context.cache().plan_len(), 1);
    assert_eq!(context.cache().structure_len(), 1);

    dst.data_mut().copy_from_slice(&[1.0, 2.0]);
    context
        .tree_pair_transform_into(
            &rule,
            TreeTransformOperationKey::braid([1, 0], [2], [1, 0], [2]),
            &mut dst,
            &src,
            1.0,
            0.0,
        )
        .unwrap();

    assert_eq!(context.cache().plan_len(), 2);
    assert_eq!(context.cache().structure_len(), 2);
}

#[test]
fn unique_tree_transform_plan_builder_rejects_generic_fusion() {
    let src_key = fusion_tree_test_key([1, 1], [1], 1, [false, false], [false]);
    let src_structure =
        BlockStructure::packed_column_major_with_keys(3, [(src_key, vec![1, 1, 1])]).unwrap();
    let operation = TreeTransformOperationKey::braid([1, 0], [0], [1, 0], [0]);

    let err = build_unique_tree_transform_group_plan(
        &GenericMultiplicityRule,
        operation.clone(),
        &src_structure,
        |_| -> Result<(FusionTreeBlockKey, f64), OperationError> {
            unreachable!("GenericFusion must be rejected before transforming keys")
        },
    )
    .unwrap_err();

    assert_eq!(
        err,
        OperationError::UnsupportedFusionStyle {
            operation,
            style: FusionStyleKind::Generic,
        }
    );
}

#[test]
fn tree_transform_operation_key_distinguishes_permute_from_explicit_braid() {
    assert!(TreeTransformOperationKey::permute([1, 0], [0]).requires_symmetric_braiding());
    assert!(!TreeTransformOperationKey::transpose([1, 0], [0]).requires_symmetric_braiding());
    assert!(
        !TreeTransformOperationKey::braid([1, 0], [0], [1, 0], [0]).requires_symmetric_braiding()
    );
}

#[test]
fn unique_tree_transform_plan_builder_rejects_permute_without_symmetric_braiding() {
    let src_key = fusion_tree_test_key([1, 0], [1], 1, [false, false], [false]);
    let src_structure =
        BlockStructure::packed_column_major_with_keys(3, [(src_key, vec![1, 1, 1])]).unwrap();
    let operation = TreeTransformOperationKey::permute([1, 0], [0]);

    let err = build_unique_tree_transform_group_plan(
        &UniqueAnyonicRule,
        operation.clone(),
        &src_structure,
        |_| -> Result<(FusionTreeBlockKey, f64), OperationError> {
            unreachable!("permutation must reject non-symmetric braiding before key transform")
        },
    )
    .unwrap_err();

    assert_eq!(
        err,
        OperationError::UnsupportedBraidingStyle {
            operation,
            style: BraidingStyleKind::Anyonic,
        }
    );
}

#[test]
fn unique_tree_transform_plan_builder_defers_explicit_no_braiding_to_crossing_logic() {
    let src_key = fusion_tree_test_key([1, 0], [1], 1, [false, false], [false]);
    let src_tree = expect_tree_key(&src_key);
    let src_structure =
        BlockStructure::packed_column_major_with_keys(3, [(src_key.clone(), vec![1, 1, 1])])
            .unwrap();

    let plan = build_unique_tree_transform_group_plan(
        &UniquePlanarRule,
        TreeTransformOperationKey::braid([1, 0], [0], [1, 0], [0]),
        &src_structure,
        |src| Ok((src.clone(), 1.0_f64)),
    )
    .unwrap();

    assert_eq!(plan.specs().len(), 1);
    assert_eq!(plan.specs()[0].group_key(), &src_tree.group_key());
    assert_eq!(plan.specs()[0].src_keys(), &[src_key.clone()]);
    assert_eq!(plan.specs()[0].dst_keys(), &[src_key]);
    assert_eq!(plan.specs()[0].coefficients_src_by_dst(), &[1.0]);
}

#[test]
fn unique_all_codomain_braid_plan_builder_lowers_codomain_single_tree() {
    let src_key = all_codomain_fusion_tree_test_key([1, 1], Some(0), [false, true], [], [1]);
    let expected_dst_key =
        all_codomain_fusion_tree_test_key([1, 1], Some(0), [true, false], [], [1]);
    let src_tree = expect_tree_key(&src_key);
    let src_structure =
        BlockStructure::packed_column_major_with_keys(2, [(src_key.clone(), vec![1, 1])]).unwrap();

    let plan = build_unique_all_codomain_tree_transform_group_plan(
        &UniqueAnyonicRule,
        TreeTransformOperationKey::braid([1, 0], Vec::<usize>::new(), [0, 1], Vec::<usize>::new()),
        &src_structure,
    )
    .unwrap();

    assert_eq!(plan.specs().len(), 1);
    assert_eq!(plan.specs()[0].group_key(), &src_tree.group_key());
    assert_eq!(plan.specs()[0].src_keys(), &[src_key]);
    assert_eq!(plan.specs()[0].dst_keys(), &[expected_dst_key]);
    assert_eq!(plan.specs()[0].coefficients_src_by_dst(), &[-2.0]);
}

#[test]
fn unique_all_codomain_permute_plan_builder_lowers_symmetric_permutation() {
    let src_key = all_codomain_fusion_tree_test_key([1, 0], Some(1), [false, true], [], [1]);
    let expected_dst_key =
        all_codomain_fusion_tree_test_key([0, 1], Some(1), [true, false], [], [1]);
    let src_structure =
        BlockStructure::packed_column_major_with_keys(2, [(src_key.clone(), vec![1, 1])]).unwrap();

    let plan = build_unique_all_codomain_tree_transform_group_plan(
        &UniqueZ2Rule,
        TreeTransformOperationKey::permute([1, 0], Vec::<usize>::new()),
        &src_structure,
    )
    .unwrap();

    assert_eq!(plan.specs().len(), 1);
    assert_eq!(plan.specs()[0].src_keys(), &[src_key]);
    assert_eq!(plan.specs()[0].dst_keys(), &[expected_dst_key]);
    assert_eq!(plan.specs()[0].coefficients_src_by_dst(), &[1.0]);
}

#[test]
fn unique_all_codomain_plan_builder_rejects_domain_operation_scope() {
    let src_key = all_codomain_fusion_tree_test_key([1, 0], Some(1), [false, false], [], [1]);
    let src_structure =
        BlockStructure::packed_column_major_with_keys(2, [(src_key, vec![1, 1])]).unwrap();
    let operation = TreeTransformOperationKey::braid([1, 0], [0], [0, 1], [0]);

    let err = build_unique_all_codomain_tree_transform_group_plan(
        &UniqueZ2Rule,
        operation.clone(),
        &src_structure,
    )
    .unwrap_err();

    assert_eq!(
        err,
        OperationError::UnsupportedTreeTransformScope {
            operation,
            message: "all-codomain UniqueFusion lowering requires an empty domain operation",
        }
    );
}

#[test]
fn unique_all_codomain_plan_builder_accepts_explicit_vacuum_empty_domain() {
    let src_key = BlockKey::from(FusionTreeBlockKey::pair(
        FusionTreeKey::from_sector_ids([1, 1], Some(0), [false, false], [], [1]),
        empty_fusion_tree_with_coupled(Some(0)),
    ));
    let expected_dst_key = BlockKey::from(FusionTreeBlockKey::pair(
        FusionTreeKey::from_sector_ids([1, 1], Some(0), [false, false], [], [1]),
        empty_fusion_tree_with_coupled(Some(0)),
    ));
    let src_structure =
        BlockStructure::packed_column_major_with_keys(2, [(src_key.clone(), vec![1, 1])]).unwrap();

    let plan = build_unique_all_codomain_tree_transform_group_plan(
        &UniqueZ2Rule,
        TreeTransformOperationKey::permute([1, 0], Vec::<usize>::new()),
        &src_structure,
    )
    .unwrap();

    assert_eq!(plan.specs().len(), 1);
    assert_eq!(plan.specs()[0].src_keys(), &[src_key]);
    assert_eq!(plan.specs()[0].dst_keys(), &[expected_dst_key]);
    assert_eq!(plan.specs()[0].coefficients_src_by_dst(), &[1.0]);
}

#[test]
fn unique_all_codomain_plan_builder_rejects_explicit_nonvacuum_empty_domain() {
    let src_key = BlockKey::from(FusionTreeBlockKey::pair(
        FusionTreeKey::from_sector_ids([1, 0], Some(1), [false, false], [], [1]),
        empty_fusion_tree_with_coupled(Some(1)),
    ));
    let src_structure =
        BlockStructure::packed_column_major_with_keys(2, [(src_key, vec![1, 1])]).unwrap();

    let err = build_unique_all_codomain_tree_transform_group_plan(
        &UniqueZ2Rule,
        TreeTransformOperationKey::permute([1, 0], Vec::<usize>::new()),
        &src_structure,
    )
    .unwrap_err();

    assert_eq!(
        err,
        OperationError::ExpectedAllCodomainFusionTree { index: 0 }
    );
}

#[test]
fn unique_all_codomain_plan_builder_rejects_nonempty_domain_tree() {
    let src_key = fusion_tree_test_key([1, 0], [1], 1, [false, false], [false]);
    let src_structure =
        BlockStructure::packed_column_major_with_keys(3, [(src_key, vec![1, 1, 1])]).unwrap();

    let err = build_unique_all_codomain_tree_transform_group_plan(
        &UniqueZ2Rule,
        TreeTransformOperationKey::permute([1, 0], Vec::<usize>::new()),
        &src_structure,
    )
    .unwrap_err();

    assert_eq!(
        err,
        OperationError::ExpectedAllCodomainFusionTree { index: 0 }
    );
}

#[test]
fn unique_all_codomain_permute_plan_builder_rejects_nonsymmetric_braiding() {
    let src_key = all_codomain_fusion_tree_test_key([1, 1], Some(0), [false, false], [], [1]);
    let src_structure =
        BlockStructure::packed_column_major_with_keys(2, [(src_key, vec![1, 1])]).unwrap();
    let operation = TreeTransformOperationKey::permute([1, 0], Vec::<usize>::new());

    let err = build_unique_all_codomain_tree_transform_group_plan(
        &UniqueAnyonicRule,
        operation.clone(),
        &src_structure,
    )
    .unwrap_err();

    assert_eq!(
        err,
        OperationError::UnsupportedBraidingStyle {
            operation,
            style: BraidingStyleKind::Anyonic,
        }
    );
}

#[test]
fn unique_tree_pair_plan_builder_lowers_domain_only_permutation() {
    let src_key = BlockKey::from(FusionTreeBlockKey::pair_from_sector_ids(
        [1],
        [0, 1],
        Some(1),
        [false],
        [false, true],
        [],
        [],
        [],
        [1],
    ));
    let expected_dst_key = BlockKey::from(FusionTreeBlockKey::pair_from_sector_ids(
        [1],
        [1, 0],
        Some(1),
        [false],
        [true, false],
        [],
        [],
        [],
        [1],
    ));
    let src_tree = expect_tree_key(&src_key);
    let src_structure =
        BlockStructure::packed_column_major_with_keys(3, [(src_key.clone(), vec![1, 1, 1])])
            .unwrap();

    let plan = build_unique_tree_pair_transform_group_plan(
        &UniqueZ2Rule,
        TreeTransformOperationKey::permute([0], [2, 1]),
        &src_structure,
    )
    .unwrap();

    assert_eq!(plan.specs().len(), 1);
    assert_eq!(plan.specs()[0].group_key(), &src_tree.group_key());
    assert_eq!(plan.specs()[0].src_keys(), &[src_key]);
    assert_eq!(plan.specs()[0].dst_keys(), &[expected_dst_key]);
    assert_eq!(plan.specs()[0].coefficients_src_by_dst(), &[1.0]);
}

#[test]
fn unique_tree_pair_plan_builder_lowers_codomain_domain_crossing_braid() {
    let src_key = BlockKey::from(FusionTreeBlockKey::pair_from_sector_ids(
        [1],
        [1],
        Some(1),
        [false],
        [true],
        [],
        [],
        [],
        [],
    ));
    let expected_dst_key = BlockKey::from(FusionTreeBlockKey::pair_from_sector_ids(
        [1],
        [1],
        Some(1),
        [false],
        [true],
        [],
        [],
        [],
        [],
    ));
    let src_structure =
        BlockStructure::packed_column_major_with_keys(2, [(src_key.clone(), vec![1, 1])]).unwrap();

    let plan = build_unique_tree_pair_transform_group_plan(
        &UniqueAnyonicRule,
        TreeTransformOperationKey::braid([1], [0], [0], [1]),
        &src_structure,
    )
    .unwrap();

    assert_eq!(plan.specs().len(), 1);
    assert_eq!(plan.specs()[0].src_keys(), &[src_key]);
    assert_eq!(plan.specs()[0].dst_keys(), &[expected_dst_key]);
    assert_eq!(plan.specs()[0].coefficients_src_by_dst(), &[-2.0]);
}

#[test]
fn unique_tree_pair_plan_builder_lowers_cyclic_transpose() {
    let src_key = BlockKey::from(FusionTreeBlockKey::pair_from_sector_ids(
        [1],
        [1],
        Some(1),
        [false],
        [true],
        [],
        [],
        [],
        [],
    ));
    let expected_dst_key = src_key.clone();
    let src_structure =
        BlockStructure::packed_column_major_with_keys(2, [(src_key.clone(), vec![1, 1])]).unwrap();
    let operation = TreeTransformOperationKey::transpose([1], [0]);

    let plan =
        build_unique_tree_pair_transform_group_plan(&UniqueZ2Rule, operation, &src_structure)
            .unwrap();

    assert_eq!(plan.specs().len(), 1);
    assert_eq!(plan.specs()[0].src_keys(), &[src_key]);
    assert_eq!(plan.specs()[0].dst_keys(), &[expected_dst_key]);
    assert_eq!(plan.specs()[0].coefficients_src_by_dst(), &[1.0]);
}

#[test]
fn unique_tree_pair_plan_builder_lowers_rank_four_cyclic_transpose() {
    let src_key = BlockKey::from(FusionTreeBlockKey::pair_from_sector_ids(
        [1, 0],
        [1, 0],
        Some(1),
        [false, false],
        [false, false],
        [],
        [],
        [1],
        [1],
    ));
    let expected_dst_key = BlockKey::from(FusionTreeBlockKey::pair_from_sector_ids(
        [1, 1],
        [0, 0],
        Some(0),
        [true, false],
        [false, true],
        [],
        [],
        [1],
        [1],
    ));
    let src_structure =
        BlockStructure::packed_column_major_with_keys(4, [(src_key.clone(), vec![1, 1, 1, 1])])
            .unwrap();
    let operation = TreeTransformOperationKey::transpose([2, 0], [3, 1]);

    let plan =
        build_unique_tree_pair_transform_group_plan(&UniqueZ2Rule, operation, &src_structure)
            .unwrap();

    assert_eq!(plan.specs().len(), 1);
    assert_eq!(plan.specs()[0].src_keys(), &[src_key]);
    assert_eq!(plan.specs()[0].dst_keys(), &[expected_dst_key]);
    assert_eq!(plan.specs()[0].coefficients_src_by_dst(), &[1.0]);
}

#[test]
fn tree_transform_compile_grouped_lowers_to_replay_ready_structure() {
    let group_key = FusionTreeGroupKey::from_sector_ids([10, 20], [30], [false, true], [true]);
    let key10 = BlockKey::sector_ids([10]);
    let key20 = BlockKey::sector_ids([20]);
    let key100 = BlockKey::sector_ids([100]);
    let key200 = BlockKey::sector_ids([200]);
    let key300 = BlockKey::sector_ids([300]);
    let src_space = TensorMapSpace::<2, 0>::from_dims([6, 1], []).unwrap();
    let dst_space = TensorMapSpace::<2, 0>::from_dims([4, 1], []).unwrap();
    let src_structure = BlockStructure::packed_column_major_with_keys(
        2,
        [
            (key100.clone(), vec![2, 1]),
            (key300.clone(), vec![2, 1]),
            (key200.clone(), vec![2, 1]),
        ],
    )
    .unwrap();
    let dst_structure = BlockStructure::packed_column_major_with_keys(
        2,
        [(key20.clone(), vec![2, 1]), (key10.clone(), vec![2, 1])],
    )
    .unwrap();
    let src = TensorMap::<f64, 2, 0>::from_vec_with_structure(
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        src_space,
        src_structure,
    )
    .unwrap();
    let mut dst =
        TensorMap::<f64, 2, 0>::from_vec_with_structure(vec![0.0; 4], dst_space, dst_structure)
            .unwrap();
    let structure = TreeTransformStructure::compile_grouped(
        &dst,
        &src,
        &[TreeTransformGroupBlockSpec::multi(
            group_key,
            [key10, key20],
            [key100, key200, key300],
            vec![10.0, 100.0, 1000.0, 20.0, 200.0, 2000.0],
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
        1.0,
        0.0,
    )
    .unwrap();

    assert_eq!(structure.block_count(), 1);
    assert_eq!(dst.data(), &[7020.0, 9240.0, 3510.0, 4620.0]);
    assert_eq!(workspace.source_len(), 6);
    assert_eq!(workspace.destination_len(), 4);
}

#[test]
fn tree_transform_compile_grouped_rejects_missing_tree_block_key() {
    let src_space = TensorMapSpace::<2, 0>::from_dims([2, 2], []).unwrap();
    let dst_space = TensorMapSpace::<2, 0>::from_dims([2, 2], []).unwrap();
    let group_key = FusionTreeGroupKey::from_sector_ids([1], [1], [false], [true]);
    let present_key = BlockKey::sector_ids([1]);
    let missing_key = BlockKey::sector_ids([2]);
    let src_structure =
        BlockStructure::packed_column_major_with_keys(2, [(present_key.clone(), vec![2, 2])])
            .unwrap();
    let dst_structure =
        BlockStructure::packed_column_major_with_keys(2, [(present_key.clone(), vec![2, 2])])
            .unwrap();
    let src =
        TensorMap::<f64, 2, 0>::from_vec_with_structure(vec![1.0; 4], src_space, src_structure)
            .unwrap();
    let dst =
        TensorMap::<f64, 2, 0>::from_vec_with_structure(vec![0.0; 4], dst_space, dst_structure)
            .unwrap();

    let err = TreeTransformStructure::compile_grouped(
        &dst,
        &src,
        &[TreeTransformGroupBlockSpec::single(
            group_key,
            missing_key.clone(),
            present_key,
            1.0,
        )],
    )
    .unwrap_err();

    assert_eq!(err, OperationError::MissingBlockKey { key: missing_key });
}

#[test]
fn tree_transform_group_block_spec_from_groups_uses_source_group_and_ordered_keys() {
    let src_key1 = fusion_tree_test_key([10, 20], [30], 5, [false, true], [true]);
    let src_key2 = fusion_tree_test_key([10, 20], [30], 6, [false, true], [true]);
    let dst_key1 = fusion_tree_test_key([20, 10], [30], 7, [true, false], [true]);
    let dst_key2 = fusion_tree_test_key([20, 10], [30], 8, [true, false], [true]);
    let src_structure = BlockStructure::packed_column_major_with_keys(
        2,
        [
            (src_key1.clone(), vec![1, 1]),
            (src_key2.clone(), vec![1, 1]),
        ],
    )
    .unwrap();
    let dst_structure = BlockStructure::packed_column_major_with_keys(
        2,
        [
            (dst_key1.clone(), vec![1, 1]),
            (dst_key2.clone(), vec![1, 1]),
        ],
    )
    .unwrap();
    let src_groups = src_structure.fusion_tree_groups();
    let dst_groups = dst_structure.fusion_tree_groups();

    let spec = TreeTransformGroupBlockSpec::from_block_groups(
        &dst_structure,
        &dst_groups[0],
        &src_structure,
        &src_groups[0],
        vec![1.0_f64, 2.0, 3.0, 4.0],
    )
    .unwrap();

    assert_eq!(spec.group_key(), src_groups[0].group_key());
    assert_ne!(spec.group_key(), dst_groups[0].group_key());
    assert_eq!(spec.src_keys(), &[src_key1, src_key2]);
    assert_eq!(spec.dst_keys(), &[dst_key1, dst_key2]);
    assert_eq!(spec.coefficients_src_by_dst(), &[1.0, 2.0, 3.0, 4.0]);
}

#[test]
fn tree_transform_group_plan_compiles_across_degeneracy_shapes_without_layout_leakage() {
    let src_key1 = fusion_tree_test_key([10, 20], [30], 5, [false, true], [true]);
    let src_key2 = fusion_tree_test_key([10, 20], [30], 6, [false, true], [true]);
    let dst_key1 = fusion_tree_test_key([20, 10], [30], 7, [true, false], [true]);
    let dst_key2 = fusion_tree_test_key([20, 10], [30], 8, [true, false], [true]);
    let src_small = BlockStructure::packed_column_major_with_keys(
        2,
        [
            (src_key1.clone(), vec![2, 1]),
            (src_key2.clone(), vec![2, 1]),
        ],
    )
    .unwrap();
    let dst_small = BlockStructure::packed_column_major_with_keys(
        2,
        [
            (dst_key1.clone(), vec![2, 1]),
            (dst_key2.clone(), vec![2, 1]),
        ],
    )
    .unwrap();
    let src_large = BlockStructure::packed_column_major_with_keys(
        2,
        [(src_key1, vec![3, 1]), (src_key2, vec![3, 1])],
    )
    .unwrap();
    let dst_large = BlockStructure::packed_column_major_with_keys(
        2,
        [(dst_key1, vec![3, 1]), (dst_key2, vec![3, 1])],
    )
    .unwrap();
    let spec = TreeTransformGroupBlockSpec::from_block_groups(
        &dst_small,
        &dst_small.fusion_tree_groups()[0],
        &src_small,
        &src_small.fusion_tree_groups()[0],
        vec![1.0_f64, 0.0, 0.0, 1.0],
    )
    .unwrap();
    let plan = TreeTransformGroupPlan::new(vec![spec]);
    let key = TreeTransformGroupPlanKey::from_plan(
        TreeTransformOperationKey::transpose([1, 0], [0]),
        &plan,
    );
    let large_spec = TreeTransformGroupBlockSpec::from_block_groups(
        &dst_large,
        &dst_large.fusion_tree_groups()[0],
        &src_large,
        &src_large.fusion_tree_groups()[0],
        vec![1.0_f64, 0.0, 0.0, 1.0],
    )
    .unwrap();
    let large_plan = TreeTransformGroupPlan::new(vec![large_spec]);
    let large_key = TreeTransformGroupPlanKey::from_plan(
        TreeTransformOperationKey::transpose([1, 0], [0]),
        &large_plan,
    );
    let mut cache = TreeTransformGroupPlanCache::new();

    cache.insert(key.clone(), plan.clone());

    let small_structure = plan.compile_structures(&dst_small, &src_small).unwrap();
    let cached = cache.get(&large_key).unwrap();
    let large_structure = cached.compile_structures(&dst_large, &src_large).unwrap();

    assert_eq!(key, large_key);
    assert_eq!(cache.len(), 1);
    assert_eq!(plan.specs().len(), 1);
    assert_eq!(small_structure.block_count(), 1);
    assert_eq!(large_structure.block_count(), 1);
    assert_eq!(small_structure.workspace_lens(), (4, 4));
    assert_eq!(large_structure.workspace_lens(), (6, 6));
}

#[test]
fn tree_transform_group_plan_cache_key_tracks_operation_but_not_coefficients() {
    let group_key = FusionTreeGroupKey::from_sector_ids([10, 20], [30], [false, true], [true]);
    let dst_key = fusion_tree_test_key([20, 10], [30], 7, [true, false], [true]);
    let src_key = fusion_tree_test_key([10, 20], [30], 5, [false, true], [true]);
    let plan_a = TreeTransformGroupPlan::new(vec![TreeTransformGroupBlockSpec::single(
        group_key.clone(),
        dst_key.clone(),
        src_key.clone(),
        2.0_f64,
    )]);
    let plan_b = TreeTransformGroupPlan::new(vec![TreeTransformGroupBlockSpec::single(
        group_key, dst_key, src_key, 3.0_f64,
    )]);

    let transpose = TreeTransformGroupPlanKey::from_plan(
        TreeTransformOperationKey::transpose([1, 0], [0]),
        &plan_a,
    );
    let same_operation_different_coefficients = TreeTransformGroupPlanKey::from_plan(
        TreeTransformOperationKey::transpose([1, 0], [0]),
        &plan_b,
    );
    let different_permutation = TreeTransformGroupPlanKey::from_plan(
        TreeTransformOperationKey::transpose([0, 1], [0]),
        &plan_a,
    );
    let braid = TreeTransformGroupPlanKey::from_plan(
        TreeTransformOperationKey::braid([1, 0], [0], [2], [0]),
        &plan_a,
    );

    assert_eq!(transpose, same_operation_different_coefficients);
    assert_ne!(transpose, different_permutation);
    assert_ne!(transpose, braid);
}

#[test]
fn tree_transform_sector_plan_key_is_rule_scope_and_source_sector_only() {
    let src_key1 = fusion_tree_test_key([10, 20], [30], 5, [false, true], [true]);
    let src_key2 = fusion_tree_test_key([10, 20], [30], 6, [false, true], [true]);
    let src_small = BlockStructure::packed_column_major_with_keys(
        2,
        [
            (src_key1.clone(), vec![2, 1]),
            (src_key2.clone(), vec![2, 1]),
        ],
    )
    .unwrap();
    let src_large = BlockStructure::packed_column_major_with_keys(
        2,
        [(src_key1, vec![3, 1]), (src_key2, vec![3, 1])],
    )
    .unwrap();
    let operation = TreeTransformOperationKey::transpose([1, 0], [0]);

    let z2_small =
        TreeTransformSectorPlanKey::tree_pair(&Z2FusionRule, operation.clone(), &src_small)
            .unwrap();
    let z2_large =
        TreeTransformSectorPlanKey::tree_pair(&Z2FusionRule, operation.clone(), &src_large)
            .unwrap();
    let fermion = TreeTransformSectorPlanKey::tree_pair(
        &FermionParityFusionRule,
        operation.clone(),
        &src_small,
    )
    .unwrap();
    let all_codomain =
        TreeTransformSectorPlanKey::all_codomain(&Z2FusionRule, operation, &src_small).unwrap();

    assert_eq!(z2_small, z2_large);
    assert_ne!(z2_small, fermion);
    assert_ne!(z2_small, all_codomain);
}

#[test]
fn tree_transform_structure_cache_key_tracks_concrete_layout() {
    let key = fusion_tree_test_key([10, 20], [30], 5, [false, true], [true]);
    let src = BlockStructure::from_blocks_with_rank(
        2,
        vec![BlockSpec::with_key(key.clone(), vec![2, 3], vec![1, 2], 0).unwrap()],
    )
    .unwrap();
    let shape_changed = BlockStructure::from_blocks_with_rank(
        2,
        vec![BlockSpec::with_key(key.clone(), vec![3, 2], vec![1, 3], 0).unwrap()],
    )
    .unwrap();
    let stride_changed = BlockStructure::from_blocks_with_rank(
        2,
        vec![BlockSpec::with_key(key.clone(), vec![2, 3], vec![2, 1], 0).unwrap()],
    )
    .unwrap();
    let offset_changed = BlockStructure::from_blocks_with_rank(
        2,
        vec![BlockSpec::with_key(key.clone(), vec![2, 3], vec![1, 2], 1).unwrap()],
    )
    .unwrap();
    let plan_key = TreeTransformSectorPlanKey::tree_pair(
        &Z2FusionRule,
        TreeTransformOperationKey::transpose([1, 0], [0]),
        &src,
    )
    .unwrap();
    let base =
        TreeTransformStructureCacheKey::from_structures(plan_key.clone(), &src, &src).unwrap();

    assert_ne!(
        base,
        TreeTransformStructureCacheKey::from_structures(plan_key.clone(), &shape_changed, &src)
            .unwrap()
    );
    assert_ne!(
        base,
        TreeTransformStructureCacheKey::from_structures(plan_key.clone(), &stride_changed, &src)
            .unwrap()
    );
    assert_ne!(
        base,
        TreeTransformStructureCacheKey::from_structures(plan_key, &offset_changed, &src).unwrap()
    );
}

#[test]
fn tree_transform_group_block_spec_rejects_group_structure_mismatch() {
    let src_key = fusion_tree_test_key([10, 20], [30], 5, [false, true], [true]);
    let src_structure =
        BlockStructure::packed_column_major_with_keys(2, [(src_key, vec![1, 1])]).unwrap();
    let dense_structure = BlockStructure::trivial(&[1, 1]).unwrap();
    let src_groups = src_structure.fusion_tree_groups();

    let err = TreeTransformGroupBlockSpec::<f64>::from_block_groups(
        &dense_structure,
        &src_groups[0],
        &src_structure,
        &src_groups[0],
        vec![1.0],
    )
    .unwrap_err();

    assert_eq!(
        err,
        OperationError::FusionTreeGroupMismatch {
            tensor: "dst",
            index: 0,
        }
    );
}

#[test]
fn tree_transform_rejects_incompatible_single_tree_shapes() {
    let src_space = TensorMapSpace::<2, 0>::from_dims([2, 2], []).unwrap();
    let dst_space = TensorMapSpace::<2, 0>::from_dims([4, 1], []).unwrap();
    let src = TensorMap::<f64, 2, 0>::from_vec(vec![1.0; 4], src_space).unwrap();
    let dst = TensorMap::<f64, 2, 0>::filled(0.0, dst_space).unwrap();

    let err =
        TreeTransformStructure::compile(&dst, &src, &[TreeTransformBlockSpec::single(0, 0, 1.0)])
            .unwrap_err();

    assert_eq!(
        err,
        OperationError::ShapeMismatch {
            dst: vec![4, 1],
            src: vec![2, 2],
        }
    );
}

#[test]
fn tree_transform_rejects_mismatched_multi_tree_element_count() {
    let src_space = TensorMapSpace::<2, 0>::from_dims([4, 2], []).unwrap();
    let dst_space = TensorMapSpace::<2, 0>::from_dims([3, 2], []).unwrap();
    let src_structure = BlockStructure::packed_column_major(2, [vec![2, 2], vec![2, 2]]).unwrap();
    let dst_structure = BlockStructure::packed_column_major(2, [vec![3, 1], vec![3, 1]]).unwrap();
    let src =
        TensorMap::<f64, 2, 0>::from_vec_with_structure(vec![1.0; 8], src_space, src_structure)
            .unwrap();
    let dst =
        TensorMap::<f64, 2, 0>::from_vec_with_structure(vec![0.0; 6], dst_space, dst_structure)
            .unwrap();

    let err = TreeTransformStructure::compile(
        &dst,
        &src,
        &[TreeTransformBlockSpec::multi(
            vec![0, 1],
            vec![0, 1],
            vec![1.0, 0.0, 0.0, 1.0],
        )],
    )
    .unwrap_err();

    assert_eq!(
        err,
        OperationError::ElementCountMismatch {
            expected: 3,
            actual: 4,
        }
    );
}
