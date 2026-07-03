use super::*;

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
