#![forbid(unsafe_code)]

//! TensorOperations-style lowering for TeNeT.
//!
//! Public/core tensor code talks in terms of TeNeT-owned block views. This crate
//! lowers those views to strided-rs kernels at the same granularity that
//! TensorKit uses Strided.jl/StridedViews.jl internally.

use core::fmt;
use core::ops::{Add, Mul};

use tenet_core::{BlockLayout, BlockView, BlockViewMut, CoreError};

pub fn copy_into<T>(dst: BlockViewMut<'_, T>, src: BlockView<'_, T>) -> Result<(), OperationError>
where
    T: Copy + strided_kernel::MaybeSendSync,
{
    let mut dst = strided_write(dst)?;
    let src = strided_read(src)?;
    strided_kernel::copy_into(&mut dst, &src).map_err(strided_error)
}

pub fn scaled_assign_into<T>(
    dst: BlockViewMut<'_, T>,
    src: BlockView<'_, T>,
    alpha: T,
) -> Result<(), OperationError>
where
    T: Copy + Mul<T, Output = T> + strided_kernel::MaybeSendSync,
{
    let mut dst = strided_write(dst)?;
    let src = strided_read(src)?;
    strided_kernel::copy_scale(&mut dst, &src, alpha).map_err(strided_error)
}

pub fn scaled_add_into<T>(
    dst: BlockViewMut<'_, T>,
    src: BlockView<'_, T>,
    alpha: T,
) -> Result<(), OperationError>
where
    T: Copy + Add<T, Output = T> + Mul<T, Output = T> + strided_kernel::MaybeSendSync,
{
    let mut dst = strided_write(dst)?;
    let src = strided_read(src)?;
    strided_kernel::axpy(&mut dst, &src, alpha).map_err(strided_error)
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum OperationError {
    Core(CoreError),
    StrideOverflow { value: usize },
    OffsetOverflow { value: usize },
    StridedKernel { message: String },
}

impl fmt::Display for OperationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Core(err) => err.fmt(f),
            Self::StrideOverflow { value } => {
                write!(f, "stride {value} does not fit in strided-rs isize")
            }
            Self::OffsetOverflow { value } => {
                write!(f, "offset {value} does not fit in strided-rs isize")
            }
            Self::StridedKernel { message } => write!(f, "strided kernel error: {message}"),
        }
    }
}

impl std::error::Error for OperationError {}

impl From<CoreError> for OperationError {
    fn from(value: CoreError) -> Self {
        Self::Core(value)
    }
}

fn strided_read<'a, T>(
    view: BlockView<'a, T>,
) -> Result<strided_kernel::StridedView<'a, T>, OperationError> {
    let layout = view.layout();
    let strides = strides_to_isize(layout.strides())?;
    let offset = offset_to_isize(layout.offset())?;
    strided_kernel::StridedView::new(view.data(), layout.shape(), &strides, offset)
        .map_err(strided_error)
}

fn strided_write<'a, T>(
    view: BlockViewMut<'a, T>,
) -> Result<strided_kernel::StridedViewMut<'a, T>, OperationError> {
    let (data, layout) = view.into_parts();
    let strides = strides_to_isize(layout.strides())?;
    let offset = offset_to_isize(layout.offset())?;
    strided_kernel::StridedViewMut::new(data, layout.shape(), &strides, offset)
        .map_err(strided_error)
}

fn strides_to_isize(strides: &[usize]) -> Result<Vec<isize>, OperationError> {
    strides
        .iter()
        .map(|&stride| {
            isize::try_from(stride).map_err(|_| OperationError::StrideOverflow { value: stride })
        })
        .collect()
}

fn offset_to_isize(offset: usize) -> Result<isize, OperationError> {
    isize::try_from(offset).map_err(|_| OperationError::OffsetOverflow { value: offset })
}

fn strided_error(err: strided_kernel::StridedError) -> OperationError {
    OperationError::StridedKernel {
        message: err.to_string(),
    }
}

#[allow(dead_code)]
fn _assert_layout_owned_by_tenet(_layout: BlockLayout<'_>) {}

#[cfg(test)]
mod tests {
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
}
