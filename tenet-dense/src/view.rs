use num_complex::{Complex32, Complex64};

use crate::layout::validate_dense_layout;
use crate::{DenseDType, DenseError, DensePlacement};

#[derive(Debug)]
pub struct DenseView<'a, T> {
    data: &'a [T],
    shape: &'a [usize],
    strides: &'a [usize],
    offset: usize,
}

impl<'a, T> Clone for DenseView<'a, T> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<'a, T> Copy for DenseView<'a, T> {}

impl<'a, T> DenseView<'a, T> {
    pub fn new(
        data: &'a [T],
        shape: &'a [usize],
        strides: &'a [usize],
        offset: usize,
    ) -> Result<Self, DenseError> {
        validate_dense_layout(data.len(), offset, shape, strides)?;
        Ok(Self {
            data,
            shape,
            strides,
            offset,
        })
    }

    /// Trusted constructor: the caller guarantees the layout was validated
    /// when the owning plan was compiled (replay-side counterpart of the
    /// `*_trusted` kernel entry points). Layout errors are still memory-safe
    /// (worst case an index panic downstream); debug builds re-validate.
    #[inline]
    pub fn new_trusted(
        data: &'a [T],
        shape: &'a [usize],
        strides: &'a [usize],
        offset: usize,
    ) -> Self {
        debug_assert!(validate_dense_layout(data.len(), offset, shape, strides).is_ok());
        Self {
            data,
            shape,
            strides,
            offset,
        }
    }

    #[inline]
    pub fn data(&self) -> &'a [T] {
        self.data
    }

    #[inline]
    pub fn shape(&self) -> &'a [usize] {
        self.shape
    }

    #[inline]
    pub fn strides(&self) -> &'a [usize] {
        self.strides
    }

    #[inline]
    pub fn offset(&self) -> usize {
        self.offset
    }

    #[inline]
    pub fn placement(&self) -> DensePlacement {
        DensePlacement::Host
    }
}

#[derive(Debug)]
pub struct DenseViewMut<'a, T> {
    pub(crate) data: &'a mut [T],
    pub(crate) shape: &'a [usize],
    pub(crate) strides: &'a [usize],
    pub(crate) offset: usize,
}

impl<'a, T> DenseViewMut<'a, T> {
    pub fn new(
        data: &'a mut [T],
        shape: &'a [usize],
        strides: &'a [usize],
        offset: usize,
    ) -> Result<Self, DenseError> {
        validate_dense_layout(data.len(), offset, shape, strides)?;
        Ok(Self {
            data,
            shape,
            strides,
            offset,
        })
    }

    /// Trusted constructor; see [`DenseView::new_trusted`].
    #[inline]
    pub fn new_trusted(
        data: &'a mut [T],
        shape: &'a [usize],
        strides: &'a [usize],
        offset: usize,
    ) -> Self {
        debug_assert!(validate_dense_layout(data.len(), offset, shape, strides).is_ok());
        Self {
            data,
            shape,
            strides,
            offset,
        }
    }

    #[inline]
    pub fn data(&self) -> &[T] {
        self.data
    }

    #[inline]
    pub fn data_mut(&mut self) -> &mut [T] {
        self.data
    }

    #[inline]
    pub fn shape(&self) -> &'a [usize] {
        self.shape
    }

    #[inline]
    pub fn strides(&self) -> &'a [usize] {
        self.strides
    }

    #[inline]
    pub fn offset(&self) -> usize {
        self.offset
    }

    #[inline]
    pub fn placement(&self) -> DensePlacement {
        DensePlacement::Host
    }
}

#[derive(Clone, Copy, Debug)]
pub enum DenseRead<'a> {
    F32(DenseView<'a, f32>),
    F64(DenseView<'a, f64>),
    I32(DenseView<'a, i32>),
    I64(DenseView<'a, i64>),
    Bool(DenseView<'a, bool>),
    C32(DenseView<'a, Complex32>),
    C64(DenseView<'a, Complex64>),
}

impl DenseRead<'_> {
    pub fn dtype(&self) -> DenseDType {
        match self {
            Self::F32(_) => DenseDType::F32,
            Self::F64(_) => DenseDType::F64,
            Self::I32(_) => DenseDType::I32,
            Self::I64(_) => DenseDType::I64,
            Self::Bool(_) => DenseDType::Bool,
            Self::C32(_) => DenseDType::C32,
            Self::C64(_) => DenseDType::C64,
        }
    }

    pub fn shape(&self) -> &[usize] {
        match self {
            Self::F32(view) => view.shape(),
            Self::F64(view) => view.shape(),
            Self::I32(view) => view.shape(),
            Self::I64(view) => view.shape(),
            Self::Bool(view) => view.shape(),
            Self::C32(view) => view.shape(),
            Self::C64(view) => view.shape(),
        }
    }

    pub fn placement(&self) -> DensePlacement {
        match self {
            Self::F32(view) => view.placement(),
            Self::F64(view) => view.placement(),
            Self::I32(view) => view.placement(),
            Self::I64(view) => view.placement(),
            Self::Bool(view) => view.placement(),
            Self::C32(view) => view.placement(),
            Self::C64(view) => view.placement(),
        }
    }
}

#[derive(Debug)]
pub enum DenseWrite<'a> {
    F32(DenseViewMut<'a, f32>),
    F64(DenseViewMut<'a, f64>),
    I32(DenseViewMut<'a, i32>),
    I64(DenseViewMut<'a, i64>),
    Bool(DenseViewMut<'a, bool>),
    C32(DenseViewMut<'a, Complex32>),
    C64(DenseViewMut<'a, Complex64>),
}

impl DenseWrite<'_> {
    pub fn dtype(&self) -> DenseDType {
        match self {
            Self::F32(_) => DenseDType::F32,
            Self::F64(_) => DenseDType::F64,
            Self::I32(_) => DenseDType::I32,
            Self::I64(_) => DenseDType::I64,
            Self::Bool(_) => DenseDType::Bool,
            Self::C32(_) => DenseDType::C32,
            Self::C64(_) => DenseDType::C64,
        }
    }

    pub fn shape(&self) -> &[usize] {
        match self {
            Self::F32(view) => view.shape(),
            Self::F64(view) => view.shape(),
            Self::I32(view) => view.shape(),
            Self::I64(view) => view.shape(),
            Self::Bool(view) => view.shape(),
            Self::C32(view) => view.shape(),
            Self::C64(view) => view.shape(),
        }
    }

    pub fn placement(&self) -> DensePlacement {
        match self {
            Self::F32(view) => view.placement(),
            Self::F64(view) => view.placement(),
            Self::I32(view) => view.placement(),
            Self::I64(view) => view.placement(),
            Self::Bool(view) => view.placement(),
            Self::C32(view) => view.placement(),
            Self::C64(view) => view.placement(),
        }
    }
}
