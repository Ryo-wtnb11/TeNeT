use super::*;

#[test]
fn dense_view_rejects_out_of_bounds_layout() {
    let data = [0.0; 6];
    let shape = [2, 3];
    let strides = [1, 4];
    let err = DenseView::new(&data, &shape, &strides, 0).unwrap_err();
    assert_eq!(err, DenseError::OutOfBounds);
}
