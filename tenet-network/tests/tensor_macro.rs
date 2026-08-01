use std::cell::Cell;
use std::sync::Arc;

use tenet::core::{
    product_sector, CheckedFusionAlgebra, FermionParityFusionRule, FusionAlgebraError,
    MultiplicityFreeAdmissionMode, MultiplicityFreeRigidSymbols, ProductFusionRuleExt,
    SU2FusionRule, SU2Irrep, SectorCodec, TypedSectorAdmission, U1FusionRule, U1Irrep, Z2Irrep,
};
use tenet::prelude::{Complex64, Error, TensorScalar};
use tenet::typed::{GradedSpace, Runtime, TensorMap, Truncation};
use tenet_network::tensor;

fn space() -> GradedSpace<U1FusionRule> {
    GradedSpace::try_new(Arc::new(U1FusionRule), [(U1Irrep::new(0), 2)], false).unwrap()
}

fn u1_space() -> GradedSpace<U1FusionRule> {
    GradedSpace::try_new(
        Arc::new(U1FusionRule),
        [
            (U1Irrep::new(-1), 2),
            (U1Irrep::new(0), 3),
            (U1Irrep::new(1), 2),
        ],
        false,
    )
    .unwrap()
}

fn su2_space() -> GradedSpace<SU2FusionRule> {
    GradedSpace::try_new(
        Arc::new(SU2FusionRule),
        [
            (SU2Irrep::from_twice_spin(0), 2),
            (SU2Irrep::from_twice_spin(1), 2),
            (SU2Irrep::from_twice_spin(2), 1),
        ],
        false,
    )
    .unwrap()
}

fn assert_close(lhs: &[f64], rhs: &[f64], tol: f64) {
    assert_eq!(lhs.len(), rhs.len(), "data lengths differ");
    for (index, (a, b)) in lhs.iter().zip(rhs).enumerate() {
        assert!(
            (a - b).abs() <= tol * (1.0 + a.abs().max(b.abs())),
            "element {index} differs: {a} vs {b}"
        );
    }
}

fn pair<D: TensorScalar>(
    runtime: &Runtime,
) -> (TensorMap<U1FusionRule, D>, TensorMap<U1FusionRule, D>) {
    let space = space();
    (
        TensorMap::rand_with_seed(runtime, [&space], [&space], 11).unwrap(),
        TensorMap::rand_with_seed(runtime, [&space], [&space], 12).unwrap(),
    )
}

fn assert_pair<R, D>(a: &TensorMap<R, D>, b: &TensorMap<R, D>)
where
    R: TypedSectorAdmission<Error = FusionAlgebraError, Mode = MultiplicityFreeAdmissionMode>
        + MultiplicityFreeRigidSymbols<Scalar = f64>
        + CheckedFusionAlgebra
        + SectorCodec
        + Send,
    D: TensorScalar + Send + Sync + PartialEq + std::fmt::Debug + 'static,
{
    let actual = tensor!([i; k] = a[i; j] * b[j; k]).unwrap();
    let expected = a.contract(b, &[1], &[0], &[0, 1]).unwrap();
    assert_eq!(actual.data(), expected.data());
}

fn assert_pair_case<R, D>(runtime: &Runtime, space: &GradedSpace<R>, seed: u64)
where
    R: TypedSectorAdmission<Error = FusionAlgebraError, Mode = MultiplicityFreeAdmissionMode>
        + MultiplicityFreeRigidSymbols<Scalar = f64>
        + CheckedFusionAlgebra
        + SectorCodec
        + Send,
    D: TensorScalar + Send + Sync + PartialEq + std::fmt::Debug + 'static,
{
    let lhs = TensorMap::<R, D>::rand_with_seed(runtime, [space], [space], seed).unwrap();
    let rhs = TensorMap::<R, D>::rand_with_seed(runtime, [space], [space], seed + 1).unwrap();
    assert_pair(&lhs, &rhs);
}

#[test]
fn typed_host_macro_provider_dtype_matrix_matches_direct_contract() {
    let runtime = Runtime::builder().build().unwrap();
    let u1 = u1_space();
    assert_pair_case::<_, f64>(&runtime, &u1, 750_100);
    assert_pair_case::<_, Complex64>(&runtime, &u1, 750_102);

    let su2 = su2_space();
    assert_pair_case::<_, f64>(&runtime, &su2, 750_110);
    assert_pair_case::<_, Complex64>(&runtime, &su2, 750_112);

    let fz2 = GradedSpace::try_new(
        Arc::new(FermionParityFusionRule),
        [(Z2Irrep::EVEN, 2), (Z2Irrep::ODD, 1)],
        false,
    )
    .unwrap();
    assert_pair_case::<_, f64>(&runtime, &fz2, 750_120);
    assert_pair_case::<_, Complex64>(&runtime, &fz2, 750_122);

    let product = GradedSpace::try_new(
        Arc::new(FermionParityFusionRule.product(U1FusionRule)),
        [
            (product_sector(Z2Irrep::EVEN, U1Irrep::new(0)), 2),
            (product_sector(Z2Irrep::ODD, U1Irrep::new(0)), 1),
        ],
        false,
    )
    .unwrap();
    assert_pair_case::<_, f64>(&runtime, &product, 750_130);
    assert_pair_case::<_, Complex64>(&runtime, &product, 750_132);
}

#[test]
fn owned_operands_infer_the_typed_host_path() {
    let runtime = Runtime::builder().build().unwrap();
    let (a, b) = pair::<f64>(&runtime);
    let actual = tensor!([i; k] = a[i; j] * b[j; k]).unwrap();
    let expected = a.contract(&b, &[1], &[0], &[0, 1]).unwrap();
    assert_eq!(actual.data(), expected.data());
}

#[test]
fn borrowed_first_operand_is_normalized_once() {
    let runtime = Runtime::builder().build().unwrap();
    let (a, b) = pair::<f64>(&runtime);
    let a_ref = &a;
    let actual = tensor!([i; k] = a_ref[i; j] * b[j; k]).unwrap();
    let expected = a.contract(&b, &[1], &[0], &[0, 1]).unwrap();
    assert_eq!(actual.data(), expected.data());
}

#[test]
fn borrowed_later_operand_is_normalized_once() {
    let runtime = Runtime::builder().build().unwrap();
    let (a, b) = pair::<f64>(&runtime);
    let b_ref = &b;
    let actual = tensor!([i; k] = a[i; j] * b_ref[j; k]).unwrap();
    let expected = a.contract(&b, &[1], &[0], &[0, 1]).unwrap();
    assert_eq!(actual.data(), expected.data());
}

#[test]
fn field_operands_are_normalized_without_moving_the_owner() {
    struct Pair {
        lhs: TensorMap<U1FusionRule, f64>,
        rhs: TensorMap<U1FusionRule, f64>,
    }

    let runtime = Runtime::builder().build().unwrap();
    let (lhs, rhs) = pair::<f64>(&runtime);
    let operands = Pair { lhs, rhs };
    let actual = tensor!([i; k] = operands.lhs[i; j] * operands.rhs[j; k]).unwrap();
    let expected = operands
        .lhs
        .contract(&operands.rhs, &[1], &[0], &[0, 1])
        .unwrap();
    assert_eq!(actual.data(), expected.data());
}

#[test]
fn operand_expressions_are_evaluated_exactly_once_in_left_to_right_order() {
    let runtime = Runtime::builder().build().unwrap();
    let (a, b) = pair::<f64>(&runtime);
    let order = Cell::new(0);
    let left = || {
        assert_eq!(order.get(), 0);
        order.set(1);
        a.clone()
    };
    let right = || {
        assert_eq!(order.get(), 1);
        order.set(2);
        b.clone()
    };
    let actual = tensor!([i; k] = (left())[i; j] * (right())[j; k]).unwrap();
    assert_eq!(order.get(), 2);
    let expected = a.contract(&b, &[1], &[0], &[0, 1]).unwrap();
    assert_eq!(actual.data(), expected.data());
}

#[test]
fn parenthesized_temporary_lives_through_execution() {
    let runtime = Runtime::builder().build().unwrap();
    let (a, b) = pair::<f64>(&runtime);
    let actual = tensor!([i; k] = (a.clone())[i; j] * (b.clone())[j; k]).unwrap();
    let expected = a.contract(&b, &[1], &[0], &[0, 1]).unwrap();
    assert_eq!(actual.data(), expected.data());
}

fn assert_high_rank_pairwise<R>(runtime: &Runtime, space: &GradedSpace<R>)
where
    R: TypedSectorAdmission<Error = FusionAlgebraError, Mode = MultiplicityFreeAdmissionMode>
        + MultiplicityFreeRigidSymbols<Scalar = f64>
        + CheckedFusionAlgebra
        + SectorCodec
        + Send,
{
    let lhs =
        TensorMap::<R, f64>::rand_with_seed(runtime, [space, space], [space, space], 101).unwrap();
    let rhs =
        TensorMap::<R, f64>::rand_with_seed(runtime, [space, space], [space, space], 102).unwrap();
    let actual = tensor!([i, j; m, n] = lhs[i, j; k, l] * rhs[k, l; m, n]).unwrap();
    let expected = lhs.contract(&rhs, &[2, 3], &[0, 1], &[0, 1, 2, 3]).unwrap();
    assert_close(actual.data(), expected.data(), 1e-12);
    assert_eq!(actual.codomain_rank(), 2);
    assert_eq!(actual.domain_rank(), 2);
}

#[test]
fn pairwise_macro_matches_direct_high_rank_contract() {
    let runtime = Runtime::builder().build().unwrap();
    assert_high_rank_pairwise(&runtime, &u1_space());
    assert_high_rank_pairwise(&runtime, &su2_space());
}

fn assert_permuted_output<R>(runtime: &Runtime, space: &GradedSpace<R>)
where
    R: TypedSectorAdmission<Error = FusionAlgebraError, Mode = MultiplicityFreeAdmissionMode>
        + MultiplicityFreeRigidSymbols<Scalar = f64>
        + CheckedFusionAlgebra
        + SectorCodec
        + Send,
{
    let lhs =
        TensorMap::<R, f64>::rand_with_seed(runtime, [space, space], [space, space], 111).unwrap();
    let rhs =
        TensorMap::<R, f64>::rand_with_seed(runtime, [space, space], [space, space], 112).unwrap();
    let actual = tensor!([j, i; m, n] = lhs[i, j; k, l] * rhs[k, l; m, n]).unwrap();
    let expected = lhs.contract(&rhs, &[2, 3], &[0, 1], &[1, 0, 2, 3]).unwrap();
    assert_close(actual.data(), expected.data(), 1e-12);
}

#[test]
fn permuted_output_labels_match_ordered_contract() {
    let runtime = Runtime::builder().build().unwrap();
    assert_permuted_output(&runtime, &u1_space());
    assert_permuted_output(&runtime, &su2_space());
}

fn assert_single_permute<R>(runtime: &Runtime, space: &GradedSpace<R>)
where
    R: TypedSectorAdmission<Error = FusionAlgebraError, Mode = MultiplicityFreeAdmissionMode>
        + MultiplicityFreeRigidSymbols<Scalar = f64>
        + CheckedFusionAlgebra
        + SectorCodec
        + Send,
{
    let tensor = TensorMap::<R, f64>::rand_with_seed(runtime, [space], [space], 121).unwrap();
    let actual = tensor!([j; i] = tensor[i; j]).unwrap();
    let expected = tensor.permute(&[1], &[0]).unwrap();
    assert_close(actual.data(), expected.data(), 1e-12);
}

#[test]
fn single_tensor_macro_is_a_permute() {
    let runtime = Runtime::builder().build().unwrap();
    assert_single_permute(&runtime, &u1_space());
    assert_single_permute(&runtime, &su2_space());
}

fn assert_conj_norm<R>(runtime: &Runtime, space: &GradedSpace<R>)
where
    R: TypedSectorAdmission<Error = FusionAlgebraError, Mode = MultiplicityFreeAdmissionMode>
        + MultiplicityFreeRigidSymbols<Scalar = f64>
        + CheckedFusionAlgebra
        + SectorCodec
        + Send,
{
    let tensor =
        TensorMap::<R, f64>::rand_with_seed(runtime, [space, space], [space, space], 131).unwrap();
    let actual = tensor!([] = conj(tensor)[i, j; k, l] * tensor[i, j; k, l])
        .unwrap()
        .scalar()
        .unwrap();
    let norm = tensor.norm().unwrap();
    assert!((actual - norm * norm).abs() <= 1e-10 * (1.0 + norm * norm));
}

#[test]
fn scalar_output_with_conj_matches_norm_squared() {
    let runtime = Runtime::builder().build().unwrap();
    assert_conj_norm(&runtime, &u1_space());
    assert_conj_norm(&runtime, &su2_space());
}

fn assert_three_tensor_chain<R>(runtime: &Runtime, space: &GradedSpace<R>)
where
    R: TypedSectorAdmission<Error = FusionAlgebraError, Mode = MultiplicityFreeAdmissionMode>
        + MultiplicityFreeRigidSymbols<Scalar = f64>
        + CheckedFusionAlgebra
        + SectorCodec
        + Send,
{
    let psi = TensorMap::<R, f64>::rand_with_seed(runtime, [space], [space, space], 141).unwrap();
    let h = TensorMap::<R, f64>::rand_with_seed(runtime, [space], [space], 142).unwrap();
    let actual = tensor!([] = conj(psi)[p; l, r] * h[p; q] * psi[q; l, r])
        .unwrap()
        .scalar()
        .unwrap();
    let h_psi = h.contract(&psi, &[1], &[0], &[0, 1, 2]).unwrap();
    let manual = psi
        .adjoint()
        .unwrap()
        .contract(&h_psi, &[2, 0, 1], &[0, 1, 2], &[])
        .unwrap()
        .scalar()
        .unwrap();
    assert!((actual - manual).abs() <= 1e-10 * (1.0 + manual.abs()));
}

#[test]
fn three_tensor_chain_with_conj_matches_manual_contraction() {
    let runtime = Runtime::builder().build().unwrap();
    assert_three_tensor_chain(&runtime, &u1_space());
    assert_three_tensor_chain(&runtime, &su2_space());
}

#[test]
fn wrong_input_codomain_split_is_rejected() {
    let runtime = Runtime::builder().build().unwrap();
    let space = u1_space();
    let lhs =
        TensorMap::<U1FusionRule, f64>::rand_with_seed(&runtime, [&space], [&space], 161).unwrap();
    let rhs =
        TensorMap::<U1FusionRule, f64>::rand_with_seed(&runtime, [&space], [&space], 162).unwrap();
    let result = tensor!([i; k] = lhs[i, j;] * rhs[j; k]);
    assert!(matches!(result, Err(Error::InvalidArgument(_))));
}

#[test]
fn contracted_leg_degeneracy_mismatch_spells_out_both_legs() {
    let runtime = Runtime::builder().build().unwrap();
    let provider = Arc::new(U1FusionRule);
    let lhs_space = GradedSpace::try_new(
        Arc::clone(&provider),
        [
            (U1Irrep::new(-1), 2),
            (U1Irrep::new(0), 3),
            (U1Irrep::new(1), 2),
        ],
        false,
    )
    .unwrap();
    let rhs_space = GradedSpace::try_new(
        provider,
        [
            (U1Irrep::new(-1), 2),
            (U1Irrep::new(0), 4),
            (U1Irrep::new(1), 2),
        ],
        false,
    )
    .unwrap();
    let lhs =
        TensorMap::<U1FusionRule, f64>::rand_with_seed(&runtime, [&lhs_space], [&lhs_space], 163)
            .unwrap();
    let rhs =
        TensorMap::<U1FusionRule, f64>::rand_with_seed(&runtime, [&rhs_space], [&rhs_space], 164)
            .unwrap();
    let message = tensor!([i; k] = lhs[i; j] * rhs[j; k])
        .unwrap_err()
        .to_string();
    assert!(
        message.contains("space mismatch for contracted label `j`"),
        "{message}"
    );
    assert!(message.contains("operand 0 leg 1"), "{message}");
    assert!(message.contains("operand 1 leg 0"), "{message}");
}

#[test]
fn factorization_fields_and_tuple_fields_contract_without_parentheses() {
    let runtime = Runtime::builder().build().unwrap();
    let space = u1_space();
    let tensor =
        TensorMap::<U1FusionRule, f64>::rand_with_seed(&runtime, [&space, &space], [&space], 401)
            .unwrap();
    let svd = tensor.svd_trunc(&Truncation::Full).unwrap();
    let bare = tensor!([i, j; m] = svd.u[i, j; k] * svd.s[k; l] * svd.vh[l; m]).unwrap();
    let parenthesized =
        tensor!([i, j; m] = (svd.u)[i, j; k] * (svd.s)[k; l] * (svd.vh)[l; m]).unwrap();
    assert_close(bare.data(), parenthesized.data(), 1e-15);
    assert_close(bare.data(), tensor.data(), 1e-10);

    let norm_squared = tensor!([] = conj(svd.u)[i, j; k] * svd.u[i, j; k])
        .unwrap()
        .scalar()
        .unwrap();
    let norm = svd.u.norm().unwrap();
    assert!((norm_squared - norm * norm).abs() <= 1e-10 * (1.0 + norm * norm));

    let qr = tensor.qr_compact().unwrap();
    let recomposed = tensor!([i, j; m] = qr.0[i, j; k] * qr.1[k; m]).unwrap();
    assert_close(recomposed.data(), tensor.data(), 1e-10);
}
