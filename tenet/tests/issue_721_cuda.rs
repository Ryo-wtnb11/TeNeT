#![cfg(feature = "cuda")]

use tenet::prelude::*;

#[test]
#[ignore]
fn lazy_decomposition_device_precedence_is_unchanged() {
    let runtime = Runtime::builder().cuda(0).build().unwrap();
    let eight = Space::su3([((1, 1), 1)]).unwrap();
    let lazy = Tensor::rand_with_seed(
        &runtime,
        Dtype::F64,
        [&eight, &eight],
        [&eight, &eight],
        721_004,
    )
    .unwrap()
    .to_cuda()
    .unwrap()
    .adjoint()
    .unwrap();
    let expected = Error::UnsupportedOnDevice(
        "materializing an adjoint device tensor has no device implementation yet; move the tensor to the host explicitly with to_host()".to_string(),
    );

    for result in [
        lazy.svd_compact().map(|_| ()),
        lazy.svd_full().map(|_| ()),
        lazy.svd_trunc(&Truncation::rank(8)).map(|_| ()),
        lazy.qr_compact().map(|_| ()),
    ] {
        assert_eq!(result.unwrap_err(), expected);
    }
    assert_eq!(lazy.placement(), tenet::core::Placement::Cuda(0));
    assert!(matches!(lazy.try_data(), Err(Error::PlacementMismatch)));
}
