#![forbid(unsafe_code)]

//! Low-level strided execution boundary for TeNeT.
//!
//! This crate owns the adapter contract between TeNeT tensor algorithms and
//! concrete strided kernels such as `strided-rs`, custom CPU kernels, or future
//! device kernels. Higher-level tensor code should pass explicit strided boxes
//! into this crate instead of constructing external view types directly.

use core::fmt;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct StridedLayout<'a> {
    len: usize,
    offset: usize,
    shape: &'a [usize],
    strides: &'a [usize],
}

impl<'a> StridedLayout<'a> {
    pub fn new(
        len: usize,
        offset: usize,
        shape: &'a [usize],
        strides: &'a [usize],
    ) -> Result<Self, StridedError> {
        let layout = Self {
            len,
            offset,
            shape,
            strides,
        };
        validate_layout(layout)?;
        Ok(layout)
    }

    #[inline]
    pub fn len(self) -> usize {
        self.len
    }

    #[inline]
    pub fn offset(self) -> usize {
        self.offset
    }

    #[inline]
    pub fn shape(self) -> &'a [usize] {
        self.shape
    }

    #[inline]
    pub fn strides(self) -> &'a [usize] {
        self.strides
    }

    #[inline]
    pub fn rank(self) -> usize {
        self.shape.len()
    }

    #[inline]
    pub fn is_empty(self) -> bool {
        self.shape.iter().any(|&dim| dim == 0)
    }

    pub fn element_count(self) -> Result<usize, StridedError> {
        checked_element_count(self.shape)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct StridedCopy<'a> {
    src: StridedLayout<'a>,
    dst: StridedLayout<'a>,
}

impl<'a> StridedCopy<'a> {
    pub fn new(src: StridedLayout<'a>, dst: StridedLayout<'a>) -> Result<Self, StridedError> {
        if src.shape != dst.shape {
            return Err(StridedError::ShapeMismatch {
                src: src.shape.to_vec(),
                dst: dst.shape.to_vec(),
            });
        }
        Ok(Self { src, dst })
    }

    #[inline]
    pub fn src(self) -> StridedLayout<'a> {
        self.src
    }

    #[inline]
    pub fn dst(self) -> StridedLayout<'a> {
        self.dst
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum StridedError {
    RankMismatch { shape: usize, strides: usize },
    ShapeMismatch { src: Vec<usize>, dst: Vec<usize> },
    ElementCountOverflow,
    OffsetOverflow,
    OutOfBounds,
}

impl fmt::Display for StridedError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::RankMismatch { shape, strides } => {
                write!(
                    f,
                    "rank mismatch: shape rank {shape}, strides rank {strides}"
                )
            }
            Self::ShapeMismatch { src, dst } => {
                write!(f, "shape mismatch: source {src:?}, destination {dst:?}")
            }
            Self::ElementCountOverflow => write!(f, "strided element count overflow"),
            Self::OffsetOverflow => write!(f, "strided offset overflow"),
            Self::OutOfBounds => write!(f, "strided layout accesses outside the buffer"),
        }
    }
}

impl std::error::Error for StridedError {}

pub fn validate_layout(layout: StridedLayout<'_>) -> Result<(), StridedError> {
    if layout.shape.len() != layout.strides.len() {
        return Err(StridedError::RankMismatch {
            shape: layout.shape.len(),
            strides: layout.strides.len(),
        });
    }
    if layout.is_empty() {
        return if layout.offset <= layout.len {
            Ok(())
        } else {
            Err(StridedError::OutOfBounds)
        };
    }
    if layout.offset >= layout.len {
        return Err(StridedError::OutOfBounds);
    }
    let max_delta = max_offset_delta(layout.shape, layout.strides)?;
    let last = layout
        .offset
        .checked_add(max_delta)
        .ok_or(StridedError::OffsetOverflow)?;
    if last < layout.len {
        Ok(())
    } else {
        Err(StridedError::OutOfBounds)
    }
}

fn checked_element_count(shape: &[usize]) -> Result<usize, StridedError> {
    shape.iter().try_fold(1usize, |acc, &dim| {
        acc.checked_mul(dim)
            .ok_or(StridedError::ElementCountOverflow)
    })
}

fn max_offset_delta(shape: &[usize], strides: &[usize]) -> Result<usize, StridedError> {
    shape
        .iter()
        .zip(strides)
        .try_fold(0usize, |acc, (&dim, &stride)| {
            let steps = dim.saturating_sub(1);
            let delta = steps
                .checked_mul(stride)
                .ok_or(StridedError::OffsetOverflow)?;
            acc.checked_add(delta).ok_or(StridedError::OffsetOverflow)
        })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validates_in_bounds_column_major_layout() {
        let shape = [2, 3, 4];
        let strides = [1, 2, 6];
        let layout = StridedLayout::new(24, 0, &shape, &strides).unwrap();
        assert_eq!(layout.rank(), 3);
        assert_eq!(layout.element_count().unwrap(), 24);
    }

    #[test]
    fn rejects_rank_mismatch() {
        let shape = [2, 3];
        let strides = [1];
        let err = StridedLayout::new(6, 0, &shape, &strides).unwrap_err();
        assert_eq!(
            err,
            StridedError::RankMismatch {
                shape: 2,
                strides: 1
            }
        );
    }

    #[test]
    fn rejects_out_of_bounds_layout() {
        let shape = [2, 3];
        let strides = [1, 4];
        let err = StridedLayout::new(6, 0, &shape, &strides).unwrap_err();
        assert_eq!(err, StridedError::OutOfBounds);
    }

    #[test]
    fn empty_layout_may_point_to_end() {
        let shape = [0, 4];
        let strides = [1, 1];
        let layout = StridedLayout::new(8, 8, &shape, &strides).unwrap();
        assert!(layout.is_empty());
    }

    #[test]
    fn copy_requires_equal_shapes() {
        let src_shape = [2, 3];
        let dst_shape = [3, 2];
        let strides = [1, 2];
        let src = StridedLayout::new(6, 0, &src_shape, &strides).unwrap();
        let dst = StridedLayout::new(6, 0, &dst_shape, &strides).unwrap();
        let err = StridedCopy::new(src, dst).unwrap_err();
        assert_eq!(
            err,
            StridedError::ShapeMismatch {
                src: vec![2, 3],
                dst: vec![3, 2],
            }
        );
    }
}
