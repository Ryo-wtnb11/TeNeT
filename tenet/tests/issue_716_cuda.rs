#![cfg(feature = "cuda")]

use tenet::prelude::*;

#[test]
#[ignore]
fn dense_lazy_adjoint_eigh_keeps_the_existing_cuda_materialization_error() {
    let runtime = Runtime::builder().cuda(0).build().unwrap();
    let space = Space::u1([(0, 2)]);
    let lazy = Tensor::rand_with_seed(&runtime, Dtype::F64, [&space], [&space], 716)
        .unwrap()
        .to_cuda()
        .unwrap()
        .adjoint()
        .unwrap();

    let expected = Error::UnsupportedOnDevice(
        "materializing an adjoint device tensor has no device implementation yet; move the tensor to the host explicitly with to_host()".to_string(),
    );
    for result in [
        lazy.eigh_vals().map(|_| ()),
        lazy.eigh_full().map(|_| ()),
        lazy.eigh_trunc(&Truncation::rank(1)).map(|_| ()),
    ] {
        assert_eq!(result.unwrap_err(), expected);
    }
    assert_eq!(lazy.placement(), tenet::core::Placement::Cuda(0));
    assert!(matches!(lazy.try_data(), Err(Error::PlacementMismatch)));
}
