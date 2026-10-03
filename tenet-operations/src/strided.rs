use tenet_core::{BlockView, BlockViewMut};

use crate::OperationError;

pub fn read<'a, T>(
    view: BlockView<'a, T>,
) -> Result<strided_kernel::StridedView<'a, T>, OperationError> {
    let layout = view.layout();
    let strides = strides_to_isize(layout.strides())?;
    let offset = offset_to_isize(layout.offset())?;
    strided_kernel::StridedView::new(view.data(), layout.shape(), &strides, offset).map_err(error)
}

pub fn write<'a, T>(
    view: BlockViewMut<'a, T>,
) -> Result<strided_kernel::StridedViewMut<'a, T>, OperationError> {
    let (data, layout) = view.into_parts();
    let strides = strides_to_isize(layout.strides())?;
    let offset = offset_to_isize(layout.offset())?;
    strided_kernel::StridedViewMut::new(data, layout.shape(), &strides, offset).map_err(error)
}

pub fn strides_to_isize(strides: &[usize]) -> Result<Vec<isize>, OperationError> {
    strides
        .iter()
        .map(|&stride| {
            isize::try_from(stride).map_err(|_| OperationError::StrideOverflow { value: stride })
        })
        .collect()
}

pub fn offset_to_isize(offset: usize) -> Result<isize, OperationError> {
    isize::try_from(offset).map_err(|_| OperationError::OffsetOverflow { value: offset })
}

pub fn element_count(shape: &[usize]) -> Result<usize, OperationError> {
    tenet_core::checked_product(shape).map_err(|_| OperationError::ElementCountOverflow)
}

pub fn column_major_strides_isize(shape: &[usize]) -> Result<Vec<isize>, OperationError> {
    let mut strides = Vec::with_capacity(shape.len());
    tenet_core::try_for_each_column_major_stride(
        1,
        shape,
        |stride| {
            strides.push(
                isize::try_from(stride)
                    .map_err(|_| OperationError::StrideOverflow { value: stride })?,
            );
            Ok(())
        },
        || OperationError::ElementCountOverflow,
    )?;
    Ok(strides)
}

pub fn column_major_strides_usize(shape: &[usize]) -> Result<Vec<usize>, OperationError> {
    let mut strides = Vec::with_capacity(shape.len());
    tenet_core::try_for_each_column_major_stride(
        1,
        shape,
        |stride| {
            strides.push(stride);
            Ok(())
        },
        || OperationError::ElementCountOverflow,
    )?;
    Ok(strides)
}

pub fn error(err: strided_kernel::StridedError) -> OperationError {
    OperationError::StridedKernel {
        message: err.to_string(),
    }
}
