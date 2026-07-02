use crate::{
    DenseBackend, DenseDType, DenseError, DenseKernelBackend, DenseRead, DenseView, DenseViewMut,
    DenseWrite,
};

use num_traits::{One, Zero};

#[derive(Clone, Debug, Default)]
pub struct StridedKernelBackend;

impl StridedKernelBackend {
    pub fn new() -> Self {
        Self
    }
}

impl DenseKernelBackend for StridedKernelBackend {
    fn backend(&self) -> DenseBackend {
        DenseBackend::Strided
    }

    fn supports_matmul(&self, dtype: DenseDType) -> bool {
        matches!(
            dtype,
            DenseDType::F32 | DenseDType::F64 | DenseDType::C32 | DenseDType::C64
        )
    }

    fn matmul_into(
        &mut self,
        output: DenseWrite<'_>,
        lhs: DenseRead<'_>,
        rhs: DenseRead<'_>,
    ) -> Result<(), DenseError> {
        match (output, lhs, rhs) {
            (DenseWrite::F32(output), DenseRead::F32(lhs), DenseRead::F32(rhs)) => {
                direct_strided_matmul_into(output, lhs, rhs)
            }
            (DenseWrite::F64(output), DenseRead::F64(lhs), DenseRead::F64(rhs)) => {
                direct_strided_matmul_into(output, lhs, rhs)
            }
            (DenseWrite::C32(output), DenseRead::C32(lhs), DenseRead::C32(rhs)) => {
                direct_strided_matmul_into(output, lhs, rhs)
            }
            (DenseWrite::C64(output), DenseRead::C64(lhs), DenseRead::C64(rhs)) => {
                direct_strided_matmul_into(output, lhs, rhs)
            }
            (output, lhs, rhs) => Err(DenseError::Backend {
                backend: DenseBackend::Strided,
                op: "matmul_into",
                message: format!(
                    "unsupported dtype combination output={:?}, lhs={:?}, rhs={:?}",
                    output.dtype(),
                    lhs.dtype(),
                    rhs.dtype()
                ),
            }),
        }
    }
}

fn direct_strided_matmul_into<T>(
    mut output: DenseViewMut<'_, T>,
    lhs: DenseView<'_, T>,
    rhs: DenseView<'_, T>,
) -> Result<(), DenseError>
where
    T: strided_einsum2::Scalar + One + Zero,
{
    let lhs_shape = rank2_shape(lhs.shape(), "lhs")?;
    let rhs_shape = rank2_shape(rhs.shape(), "rhs")?;
    let output_shape = rank2_shape(output.shape(), "output")?;
    if lhs_shape[1] != rhs_shape[0] {
        return Err(shape_error(format!(
            "lhs columns {} do not match rhs rows {}",
            lhs_shape[1], rhs_shape[0]
        )));
    }
    let expected_output = [lhs_shape[0], rhs_shape[1]];
    if output_shape != expected_output {
        return Err(shape_error(format!(
            "output shape {:?} does not match matmul shape {:?}",
            output_shape, expected_output
        )));
    }

    let lhs_strides = rank2_strides_to_isize(lhs.strides(), "lhs")?;
    let rhs_strides = rank2_strides_to_isize(rhs.strides(), "rhs")?;
    let output_strides = rank2_strides_to_isize(output.strides(), "output")?;
    let lhs_offset = offset_to_isize(lhs.offset())?;
    let rhs_offset = offset_to_isize(rhs.offset())?;
    let output_offset = offset_to_isize(output.offset())?;

    let lhs_view =
        strided_einsum2::RawStridedRef::new(lhs.data(), &lhs_shape, &lhs_strides, lhs_offset)
            .map_err(strided_error)?;
    let rhs_view =
        strided_einsum2::RawStridedRef::new(rhs.data(), &rhs_shape, &rhs_strides, rhs_offset)
            .map_err(strided_error)?;
    let output_view = strided_einsum2::RawStridedMut::new(
        output.data_mut(),
        &output_shape,
        &output_strides,
        output_offset,
    )
    .map_err(strided_error)?;

    strided_einsum2::bgemm_raw_strided_into(
        output_view,
        lhs_view,
        rhs_view,
        0,
        1,
        1,
        1,
        T::one(),
        T::zero(),
        false,
        false,
    )
    .map_err(strided_error)
}

fn offset_to_isize(offset: usize) -> Result<isize, DenseError> {
    isize::try_from(offset).map_err(|_| DenseError::OffsetOverflow { value: offset })
}

fn rank2_shape(shape: &[usize], label: &'static str) -> Result<[usize; 2], DenseError> {
    match shape {
        [rows, cols] => Ok([*rows, *cols]),
        _ => Err(shape_error(format!(
            "{label} rank {} is not rank-2",
            shape.len()
        ))),
    }
}

fn rank2_strides_to_isize(
    strides: &[usize],
    label: &'static str,
) -> Result<[isize; 2], DenseError> {
    match strides {
        [row, col] => Ok([
            isize::try_from(*row).map_err(|_| DenseError::StrideOverflow { value: *row })?,
            isize::try_from(*col).map_err(|_| DenseError::StrideOverflow { value: *col })?,
        ]),
        _ => Err(shape_error(format!(
            "{label} stride rank {} is not rank-2",
            strides.len()
        ))),
    }
}

fn shape_error(message: String) -> DenseError {
    DenseError::Backend {
        backend: DenseBackend::Strided,
        op: "matmul_into",
        message,
    }
}

fn strided_error(err: impl std::fmt::Display) -> DenseError {
    DenseError::Backend {
        backend: DenseBackend::Strided,
        op: "matmul_into",
        message: err.to_string(),
    }
}
