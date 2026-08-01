#[test]
fn returning_typed_cuda_destinations_use_native_zero_without_host_transfer() {
    let dense = include_str!("../../tenet-dense/src/cuda_adapter.rs");
    let operations = include_str!("../../tenet-operations/src/cuda.rs");
    let typed = include_str!("../src/typed.rs");

    let dense_zeros = dense
        .split_once("pub fn zeros_f64")
        .unwrap()
        .1
        .split_once("pub fn upload_f64")
        .unwrap()
        .0;
    assert!(dense_zeros.contains(".zeros::<f64>(len)"));
    assert!(dense_zeros.contains(".map(Tensor::F64)"));
    assert!(dense_zeros.contains("cuda_zeros_f64"));
    assert!(!dense_zeros.contains("upload_tensor"));
    assert!(!dense_zeros.contains("vec!["));

    let storage_zeros = operations
        .split_once("pub fn zeros")
        .unwrap()
        .1
        .split_once("pub fn upload")
        .unwrap()
        .0;
    assert!(storage_zeros.contains("CudaDenseStorage::zeros_f64(ctx, len)"));
    assert!(!storage_zeros.contains("upload"));

    assert_eq!(typed.matches("CudaStorage::zeros(cuda,").count(), 2);
    assert!(!typed.contains("vec![0.0; dst_space.space().required_len()?]"));
    assert!(typed.contains("TypedData::Dense(data) => CudaStorage::upload(cuda, data)?"));
}
