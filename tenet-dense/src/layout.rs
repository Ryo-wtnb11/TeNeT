use crate::DenseError;

pub(crate) fn validate_dense_layout(
    len: usize,
    offset: usize,
    shape: &[usize],
    strides: &[usize],
) -> Result<(), DenseError> {
    if shape.len() != strides.len() {
        return Err(DenseError::RankMismatch {
            shape: shape.len(),
            strides: strides.len(),
        });
    }
    if shape.iter().any(|&dim| dim == 0) {
        return if offset <= len {
            Ok(())
        } else {
            Err(DenseError::OutOfBounds)
        };
    }
    if offset >= len {
        return Err(DenseError::OutOfBounds);
    }
    let max_delta = max_offset_delta(shape, strides)?;
    let last = offset
        .checked_add(max_delta)
        .ok_or(DenseError::OffsetOverflow { value: offset })?;
    if last < len {
        Ok(())
    } else {
        Err(DenseError::OutOfBounds)
    }
}

fn max_offset_delta(shape: &[usize], strides: &[usize]) -> Result<usize, DenseError> {
    shape
        .iter()
        .zip(strides)
        .try_fold(0usize, |acc, (&dim, &stride)| {
            let steps = dim.saturating_sub(1);
            let delta = steps
                .checked_mul(stride)
                .ok_or(DenseError::StrideOverflow { value: stride })?;
            acc.checked_add(delta)
                .ok_or(DenseError::ElementCountOverflow)
        })
}

#[cfg(feature = "tenferro")]
pub(crate) fn strides_to_isize(strides: &[usize]) -> Result<Vec<isize>, DenseError> {
    strides
        .iter()
        .map(|&stride| {
            isize::try_from(stride).map_err(|_| DenseError::StrideOverflow { value: stride })
        })
        .collect()
}
