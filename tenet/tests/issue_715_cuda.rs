#![cfg(feature = "cuda")]

use tenet::prelude::*;

#[test]
#[ignore]
fn dense_lazy_adjoint_sqrt_keeps_the_existing_cuda_materialization_error() {
    let runtime = Runtime::builder().cuda(0).build().unwrap();
    let space = Space::u1([(0, 2)]);
    let lazy = Tensor::rand_with_seed(&runtime, Dtype::F64, [&space], [&space], 715)
        .unwrap()
        .to_cuda()
        .unwrap()
        .adjoint()
        .unwrap();

    let error = lazy.sqrt().unwrap_err();
    assert!(matches!(error, Error::UnsupportedOnDevice(_)));
    assert!(error
        .to_string()
        .contains("materializing an adjoint device tensor"));
    assert_eq!(lazy.placement(), tenet::core::Placement::Cuda(0));
    assert!(matches!(lazy.try_data(), Err(Error::PlacementMismatch)));
}
