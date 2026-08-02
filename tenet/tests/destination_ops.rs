use tenet::prelude::{
    ContractOverwriteCache, Dtype, Runtime, Scalar, Space, Tensor, TensorExecutionContext,
};

#[test]
fn erased_ordered_overwrite_preserves_contract_error_precedence() {
    let runtime = Runtime::builder().build().unwrap();
    let space = Space::su2([(0, 2), (1, 2), (2, 1)]).unwrap();
    let incompatible = Space::su2([(0, 2), (1, 3), (2, 1)]).unwrap();
    let lhs = Tensor::rand_with_seed(&runtime, Dtype::F64, [&space], [&space], 30_141).unwrap();
    let rhs = Tensor::rand_with_seed(
        &runtime,
        Dtype::F64,
        [&incompatible],
        [&incompatible],
        30_142,
    )
    .unwrap();
    let mut destination = Tensor::zeros(&runtime, Dtype::F64, [&space], [&space]).unwrap();
    let before = destination.data().to_vec();

    let expected = lhs.contract(&rhs, &[1], &[0]).unwrap_err();
    let actual = TensorExecutionContext::for_runtime(&runtime)
        .unwrap()
        .try_contract_ordered_overwrite_into(
            &mut ContractOverwriteCache::default(),
            &mut destination,
            &lhs,
            &rhs,
            &[1],
            &[0],
            &[],
            Scalar::F64(1.0),
        )
        .unwrap_err();

    assert_eq!(actual, expected);
    assert_eq!(destination.data(), before);
}
