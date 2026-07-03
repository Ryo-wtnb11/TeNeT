#![forbid(unsafe_code)]

//! Dense block execution boundary for TeNeT.
//!
//! Symmetric tensor algorithms lower to this crate through TeNeT-owned storage
//! views and executors. The storage placement determines the execution path:
//! host views use host kernels, and future device views should use device
//! kernels without exposing concrete runtimes to TensorMap-level code.

use core::fmt;

use num_complex::{Complex32, Complex64};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DenseDType {
    F32,
    F64,
    I32,
    I64,
    Bool,
    C32,
    C64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DenseBackend {
    Tenferro,
    Strided,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DensePlacement {
    Host,
}

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
    data: &'a mut [T],
    shape: &'a [usize],
    strides: &'a [usize],
    offset: usize,
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

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DenseDotConfig {
    lhs_contracting_dims: Vec<usize>,
    rhs_contracting_dims: Vec<usize>,
    lhs_batch_dims: Vec<usize>,
    rhs_batch_dims: Vec<usize>,
}

impl DenseDotConfig {
    pub fn new(
        lhs_contracting_dims: Vec<usize>,
        rhs_contracting_dims: Vec<usize>,
        lhs_batch_dims: Vec<usize>,
        rhs_batch_dims: Vec<usize>,
    ) -> Self {
        Self {
            lhs_contracting_dims,
            rhs_contracting_dims,
            lhs_batch_dims,
            rhs_batch_dims,
        }
    }

    pub fn matmul() -> Self {
        Self::new(vec![1], vec![0], Vec::new(), Vec::new())
    }

    #[inline]
    pub fn lhs_contracting_dims(&self) -> &[usize] {
        &self.lhs_contracting_dims
    }

    #[inline]
    pub fn rhs_contracting_dims(&self) -> &[usize] {
        &self.rhs_contracting_dims
    }

    #[inline]
    pub fn lhs_batch_dims(&self) -> &[usize] {
        &self.lhs_batch_dims
    }

    #[inline]
    pub fn rhs_batch_dims(&self) -> &[usize] {
        &self.rhs_batch_dims
    }
}

#[derive(Clone, Debug)]
pub struct DenseTensor {
    backend: DenseBackend,
    inner: DenseTensorInner,
}

#[derive(Clone, Debug)]
enum DenseTensorInner {
    #[cfg(feature = "tenferro")]
    Tenferro(tenferro_tensor::Tensor),
    #[cfg(not(feature = "tenferro"))]
    #[allow(dead_code)]
    Empty(std::convert::Infallible),
}

impl DenseTensor {
    #[inline]
    pub fn backend(&self) -> DenseBackend {
        self.backend
    }

    pub fn dtype(&self) -> DenseDType {
        match &self.inner {
            #[cfg(feature = "tenferro")]
            DenseTensorInner::Tenferro(tensor) => dense_dtype_from_tenferro(tensor.dtype()),
            #[cfg(not(feature = "tenferro"))]
            DenseTensorInner::Empty(inner) => match *inner {},
        }
    }

    pub fn shape(&self) -> &[usize] {
        match &self.inner {
            #[cfg(feature = "tenferro")]
            DenseTensorInner::Tenferro(tensor) => tensor.shape(),
            #[cfg(not(feature = "tenferro"))]
            DenseTensorInner::Empty(inner) => match *inner {},
        }
    }

    pub fn as_f32_slice(&self) -> Result<&[f32], DenseError> {
        match &self.inner {
            #[cfg(feature = "tenferro")]
            DenseTensorInner::Tenferro(tensor) => tensor
                .as_slice::<f32>()
                .map_err(|err| tenferro_error("DenseTensor::as_f32_slice", err)),
            #[cfg(not(feature = "tenferro"))]
            DenseTensorInner::Empty(inner) => match *inner {},
        }
    }

    pub fn as_f64_slice(&self) -> Result<&[f64], DenseError> {
        match &self.inner {
            #[cfg(feature = "tenferro")]
            DenseTensorInner::Tenferro(tensor) => tensor
                .as_slice::<f64>()
                .map_err(|err| tenferro_error("DenseTensor::as_f64_slice", err)),
            #[cfg(not(feature = "tenferro"))]
            DenseTensorInner::Empty(inner) => match *inner {},
        }
    }

    pub fn as_c32_slice(&self) -> Result<&[Complex32], DenseError> {
        match &self.inner {
            #[cfg(feature = "tenferro")]
            DenseTensorInner::Tenferro(tensor) => tensor
                .as_slice::<Complex32>()
                .map_err(|err| tenferro_error("DenseTensor::as_c32_slice", err)),
            #[cfg(not(feature = "tenferro"))]
            DenseTensorInner::Empty(inner) => match *inner {},
        }
    }

    pub fn as_c64_slice(&self) -> Result<&[Complex64], DenseError> {
        match &self.inner {
            #[cfg(feature = "tenferro")]
            DenseTensorInner::Tenferro(tensor) => tensor
                .as_slice::<Complex64>()
                .map_err(|err| tenferro_error("DenseTensor::as_c64_slice", err)),
            #[cfg(not(feature = "tenferro"))]
            DenseTensorInner::Empty(inner) => match *inner {},
        }
    }

    #[cfg(feature = "tenferro")]
    fn from_tenferro(tensor: tenferro_tensor::Tensor) -> Self {
        Self {
            backend: DenseBackend::Tenferro,
            inner: DenseTensorInner::Tenferro(tensor),
        }
    }
}

pub trait DenseExecutor {
    fn svd(&mut self, input: DenseRead<'_>) -> Result<Vec<DenseTensor>, DenseError>;
    fn qr(&mut self, input: DenseRead<'_>) -> Result<Vec<DenseTensor>, DenseError>;
    fn eigh(&mut self, input: DenseRead<'_>) -> Result<Vec<DenseTensor>, DenseError>;

    /// General (non-Hermitian) eigendecomposition `(values, vectors)`; both
    /// outputs are complex regardless of the input scalar.
    fn eig(&mut self, input: DenseRead<'_>) -> Result<Vec<DenseTensor>, DenseError> {
        let _ = input;
        Err(DenseError::Backend {
            backend: DenseBackend::Tenferro,
            op: "eig",
            message: "executor does not implement the general eigendecomposition".to_string(),
        })
    }

    fn dot_general_into(
        &mut self,
        output: DenseWrite<'_>,
        lhs: DenseRead<'_>,
        rhs: DenseRead<'_>,
        config: &DenseDotConfig,
    ) -> Result<(), DenseError>;

    fn matmul_into(
        &mut self,
        output: DenseWrite<'_>,
        lhs: DenseRead<'_>,
        rhs: DenseRead<'_>,
    ) -> Result<(), DenseError> {
        self.dot_general_into(output, lhs, rhs, &DenseDotConfig::matmul())
    }
}

/// Low-level dense matmul kernel boundary used by dense executors.
///
/// Implementations own the placement-specific rank-2 matmul path. The default
/// host implementation wraps `strided-einsum2`; future BLAS/C++/CUDA kernels
/// should implement this trait without changing TensorMap/fusion code.
pub trait DenseKernelBackend {
    fn supports_matmul(&self, dtype: DenseDType) -> bool;

    fn matmul_into(
        &mut self,
        output: DenseWrite<'_>,
        lhs: DenseRead<'_>,
        rhs: DenseRead<'_>,
    ) -> Result<(), DenseError>;
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum DenseError {
    RankMismatch {
        shape: usize,
        strides: usize,
    },
    ElementCountOverflow,
    StrideOverflow {
        value: usize,
    },
    OffsetOverflow {
        value: usize,
    },
    OutOfBounds,
    Backend {
        backend: DenseBackend,
        op: &'static str,
        message: String,
    },
}

impl fmt::Display for DenseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::RankMismatch { shape, strides } => {
                write!(
                    f,
                    "rank mismatch: shape rank {shape}, strides rank {strides}"
                )
            }
            Self::ElementCountOverflow => write!(f, "dense view element count overflow"),
            Self::StrideOverflow { value } => {
                write!(f, "dense view stride {value} does not fit in isize")
            }
            Self::OffsetOverflow { value } => {
                write!(f, "dense view offset {value} does not fit in isize")
            }
            Self::OutOfBounds => write!(f, "dense view accesses outside the buffer"),
            Self::Backend {
                backend,
                op,
                message,
            } => {
                write!(f, "{backend:?} backend error in {op}: {message}")
            }
        }
    }
}

impl std::error::Error for DenseError {}

fn validate_dense_layout(
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
pub use strided_adapter::StridedKernelBackend;

#[cfg(feature = "tenferro")]
pub use tenferro_adapter::{DefaultDenseExecutor, DenseExecutorWithKernel};

#[cfg(feature = "tenferro")]
mod strided_adapter;

#[cfg(feature = "tenferro")]
mod tenferro_adapter {
    use super::*;

    use super::strided_adapter::StridedKernelBackend;
    use tenferro_cpu::CpuBackend;
    use tenferro_linalg::LinalgBackend;
    use tenferro_tensor::{
        DotGeneralConfig, Tensor, TensorDot, TensorRead, TensorView, TensorViewMut, TensorWrite,
        TypedTensorView, TypedTensorViewMut,
    };

    #[derive(Debug)]
    pub struct DenseExecutorWithKernel<K> {
        backend: CpuBackend,
        kernel: K,
        matmul_config: DotGeneralConfig,
    }

    #[derive(Debug)]
    pub struct DefaultDenseExecutor {
        inner: DenseExecutorWithKernel<StridedKernelBackend>,
    }

    impl DefaultDenseExecutor {
        pub fn new() -> Self {
            Self {
                inner: DenseExecutorWithKernel::with_kernel_backend(StridedKernelBackend::new()),
            }
        }
    }

    impl<K> DenseExecutorWithKernel<K> {
        pub fn with_kernel_backend(kernel: K) -> Self {
            Self {
                backend: CpuBackend::new(),
                kernel,
                matmul_config: DotGeneralConfig {
                    lhs_contracting_dims: vec![1],
                    rhs_contracting_dims: vec![0],
                    lhs_batch_dims: Vec::new(),
                    rhs_batch_dims: Vec::new(),
                },
            }
        }

        #[cfg(test)]
        pub(super) fn kernel_backend(&self) -> &K {
            &self.kernel
        }
    }

    impl Default for DefaultDenseExecutor {
        fn default() -> Self {
            Self::new()
        }
    }

    impl DenseExecutor for DefaultDenseExecutor {
        fn svd(&mut self, input: DenseRead<'_>) -> Result<Vec<DenseTensor>, DenseError> {
            self.inner.svd(input)
        }

        fn qr(&mut self, input: DenseRead<'_>) -> Result<Vec<DenseTensor>, DenseError> {
            self.inner.qr(input)
        }

        fn eigh(&mut self, input: DenseRead<'_>) -> Result<Vec<DenseTensor>, DenseError> {
            self.inner.eigh(input)
        }

        fn eig(&mut self, input: DenseRead<'_>) -> Result<Vec<DenseTensor>, DenseError> {
            self.inner.eig(input)
        }

        fn dot_general_into(
            &mut self,
            output: DenseWrite<'_>,
            lhs: DenseRead<'_>,
            rhs: DenseRead<'_>,
            config: &DenseDotConfig,
        ) -> Result<(), DenseError> {
            self.inner.dot_general_into(output, lhs, rhs, config)
        }

        fn matmul_into(
            &mut self,
            output: DenseWrite<'_>,
            lhs: DenseRead<'_>,
            rhs: DenseRead<'_>,
        ) -> Result<(), DenseError> {
            self.inner.matmul_into(output, lhs, rhs)
        }
    }

    impl<K> DenseExecutor for DenseExecutorWithKernel<K>
    where
        K: DenseKernelBackend,
    {
        fn svd(&mut self, input: DenseRead<'_>) -> Result<Vec<DenseTensor>, DenseError> {
            let input = tenferro_view(input)?;
            self.backend
                .svd_read(input)
                .map(wrap_outputs)
                .map_err(|err| tenferro_error("svd_read", err))
        }

        fn qr(&mut self, input: DenseRead<'_>) -> Result<Vec<DenseTensor>, DenseError> {
            let input = tenferro_view(input)?;
            self.backend
                .qr_read(input)
                .map(wrap_outputs)
                .map_err(|err| tenferro_error("qr_read", err))
        }

        fn eig(&mut self, input: DenseRead<'_>) -> Result<Vec<DenseTensor>, DenseError> {
            let input = tenferro_view(input)?;
            self.backend
                .eig_read(input)
                .map(wrap_outputs)
                .map_err(|err| tenferro_error("eig_read", err))
        }

        fn eigh(&mut self, input: DenseRead<'_>) -> Result<Vec<DenseTensor>, DenseError> {
            let input = tenferro_view(input)?;
            self.backend
                .eigh_read(input)
                .map(wrap_outputs)
                .map_err(|err| tenferro_error("eigh_read", err))
        }

        fn dot_general_into(
            &mut self,
            output: DenseWrite<'_>,
            lhs: DenseRead<'_>,
            rhs: DenseRead<'_>,
            config: &DenseDotConfig,
        ) -> Result<(), DenseError> {
            let lhs = TensorRead::from_view(tenferro_view(lhs)?);
            let rhs = TensorRead::from_view(tenferro_view(rhs)?);
            let output = TensorWrite::from_view(tenferro_view_mut(output)?);
            self.backend
                .dot_general_read_into(lhs, rhs, &tenferro_dot_config(config), output)
                .map_err(|err| tenferro_error("dot_general_read_into", err))
        }

        fn matmul_into(
            &mut self,
            output: DenseWrite<'_>,
            lhs: DenseRead<'_>,
            rhs: DenseRead<'_>,
        ) -> Result<(), DenseError> {
            let dtype = output.dtype();
            if dtype == lhs.dtype() && dtype == rhs.dtype() && self.kernel.supports_matmul(dtype) {
                self.kernel.matmul_into(output, lhs, rhs)
            } else {
                self.tenferro_matmul_into(output, lhs, rhs)
            }
        }
    }

    impl<K> DenseExecutorWithKernel<K>
    where
        K: DenseKernelBackend,
    {
        fn tenferro_matmul_into(
            &mut self,
            output: DenseWrite<'_>,
            lhs: DenseRead<'_>,
            rhs: DenseRead<'_>,
        ) -> Result<(), DenseError> {
            let lhs = TensorRead::from_view(tenferro_view(lhs)?);
            let rhs = TensorRead::from_view(tenferro_view(rhs)?);
            let output = TensorWrite::from_view(tenferro_view_mut(output)?);
            self.backend
                .dot_general_read_into(lhs, rhs, &self.matmul_config, output)
                .map_err(|err| tenferro_error("dot_general_read_into", err))
        }
    }

    fn wrap_outputs(outputs: Vec<Tensor>) -> Vec<DenseTensor> {
        outputs
            .into_iter()
            .map(DenseTensor::from_tenferro)
            .collect()
    }

    fn tenferro_view(input: DenseRead<'_>) -> Result<TensorView<'_>, DenseError> {
        match input {
            DenseRead::F32(view) => typed_tenferro_view(view).map(TensorView::F32),
            DenseRead::F64(view) => typed_tenferro_view(view).map(TensorView::F64),
            DenseRead::I32(view) => typed_tenferro_view(view).map(TensorView::I32),
            DenseRead::I64(view) => typed_tenferro_view(view).map(TensorView::I64),
            DenseRead::Bool(view) => typed_tenferro_view(view).map(TensorView::Bool),
            DenseRead::C32(view) => typed_tenferro_view(view).map(TensorView::C32),
            DenseRead::C64(view) => typed_tenferro_view(view).map(TensorView::C64),
        }
    }

    fn tenferro_view_mut(output: DenseWrite<'_>) -> Result<TensorViewMut<'_>, DenseError> {
        match output {
            DenseWrite::F32(view) => typed_tenferro_view_mut(view).map(TensorViewMut::F32),
            DenseWrite::F64(view) => typed_tenferro_view_mut(view).map(TensorViewMut::F64),
            DenseWrite::I32(view) => typed_tenferro_view_mut(view).map(TensorViewMut::I32),
            DenseWrite::I64(view) => typed_tenferro_view_mut(view).map(TensorViewMut::I64),
            DenseWrite::Bool(view) => typed_tenferro_view_mut(view).map(TensorViewMut::Bool),
            DenseWrite::C32(view) => typed_tenferro_view_mut(view).map(TensorViewMut::C32),
            DenseWrite::C64(view) => typed_tenferro_view_mut(view).map(TensorViewMut::C64),
        }
    }

    fn typed_tenferro_view<'a, T: 'static>(
        view: DenseView<'a, T>,
    ) -> Result<TypedTensorView<'a, T>, DenseError> {
        let strides = strides_to_isize(view.strides())?;
        let offset = isize::try_from(view.offset()).map_err(|_| DenseError::OffsetOverflow {
            value: view.offset(),
        })?;
        TypedTensorView::from_slice(view.shape(), strides, offset, view.data())
            .map_err(|err| tenferro_error("TypedTensorView::from_slice", err))
    }

    fn typed_tenferro_view_mut<'a, T: 'static>(
        view: DenseViewMut<'a, T>,
    ) -> Result<TypedTensorViewMut<'a, T>, DenseError> {
        let DenseViewMut {
            data,
            shape,
            strides,
            offset,
        } = view;
        let strides = strides_to_isize(strides)?;
        let offset =
            isize::try_from(offset).map_err(|_| DenseError::OffsetOverflow { value: offset })?;
        TypedTensorViewMut::from_slice(shape, strides, offset, data)
            .map_err(|err| tenferro_error("TypedTensorViewMut::from_slice", err))
    }

    fn tenferro_dot_config(config: &DenseDotConfig) -> DotGeneralConfig {
        DotGeneralConfig {
            lhs_contracting_dims: config.lhs_contracting_dims().to_vec(),
            rhs_contracting_dims: config.rhs_contracting_dims().to_vec(),
            lhs_batch_dims: config.lhs_batch_dims().to_vec(),
            rhs_batch_dims: config.rhs_batch_dims().to_vec(),
        }
    }
}

#[cfg(feature = "tenferro")]
fn strides_to_isize(strides: &[usize]) -> Result<Vec<isize>, DenseError> {
    strides
        .iter()
        .map(|&stride| {
            isize::try_from(stride).map_err(|_| DenseError::StrideOverflow { value: stride })
        })
        .collect()
}

#[cfg(feature = "tenferro")]
fn dense_dtype_from_tenferro(dtype: tenferro_tensor::DType) -> DenseDType {
    match dtype {
        tenferro_tensor::DType::F32 => DenseDType::F32,
        tenferro_tensor::DType::F64 => DenseDType::F64,
        tenferro_tensor::DType::I32 => DenseDType::I32,
        tenferro_tensor::DType::I64 => DenseDType::I64,
        tenferro_tensor::DType::Bool => DenseDType::Bool,
        tenferro_tensor::DType::C32 => DenseDType::C32,
        tenferro_tensor::DType::C64 => DenseDType::C64,
    }
}

#[cfg(feature = "tenferro")]
fn tenferro_error(op: &'static str, err: tenferro_tensor::Error) -> DenseError {
    DenseError::Backend {
        backend: DenseBackend::Tenferro,
        op,
        message: err.to_string(),
    }
}

#[cfg(test)]
mod tests {
    #![allow(dead_code)]

    use super::*;

    fn assert_f64_close(actual: f64, expected: f64, tol: f64) {
        assert!(
            (actual - expected).abs() <= tol,
            "expected {expected}, got {actual}, tol={tol}"
        );
    }

    fn assert_f32_close(actual: f32, expected: f32, tol: f32) {
        assert!(
            (actual - expected).abs() <= tol,
            "expected {expected}, got {actual}, tol={tol}"
        );
    }

    fn assert_c32_close(actual: Complex32, expected: Complex32, tol: f32) {
        assert_f32_close(actual.re, expected.re, tol);
        assert_f32_close(actual.im, expected.im, tol);
    }

    fn assert_c64_close(actual: Complex64, expected: Complex64, tol: f64) {
        assert_f64_close(actual.re, expected.re, tol);
        assert_f64_close(actual.im, expected.im, tol);
    }

    fn col_major_index(rows: usize, row: usize, col: usize) -> usize {
        row + col * rows
    }

    fn transpose_f32(mat: &[f32], rows: usize, cols: usize) -> Vec<f32> {
        let mut out = vec![0.0; rows * cols];
        for j in 0..cols {
            for i in 0..rows {
                out[col_major_index(cols, j, i)] = mat[col_major_index(rows, i, j)];
            }
        }
        out
    }

    fn transpose_f64(mat: &[f64], rows: usize, cols: usize) -> Vec<f64> {
        let mut out = vec![0.0; rows * cols];
        for j in 0..cols {
            for i in 0..rows {
                out[col_major_index(cols, j, i)] = mat[col_major_index(rows, i, j)];
            }
        }
        out
    }

    fn transpose_c32(mat: &[Complex32], rows: usize, cols: usize) -> Vec<Complex32> {
        let mut out = vec![Complex32::new(0.0, 0.0); rows * cols];
        for j in 0..cols {
            for i in 0..rows {
                out[col_major_index(cols, j, i)] = mat[col_major_index(rows, i, j)];
            }
        }
        out
    }

    fn transpose_c64(mat: &[Complex64], rows: usize, cols: usize) -> Vec<Complex64> {
        let mut out = vec![Complex64::new(0.0, 0.0); rows * cols];
        for j in 0..cols {
            for i in 0..rows {
                out[col_major_index(cols, j, i)] = mat[col_major_index(rows, i, j)];
            }
        }
        out
    }

    fn conjugate_transpose_c32(mat: &[Complex32], rows: usize, cols: usize) -> Vec<Complex32> {
        let mut out = vec![Complex32::new(0.0, 0.0); rows * cols];
        for j in 0..cols {
            for i in 0..rows {
                out[col_major_index(cols, j, i)] = mat[col_major_index(rows, i, j)].conj();
            }
        }
        out
    }

    fn conjugate_transpose_c64(mat: &[Complex64], rows: usize, cols: usize) -> Vec<Complex64> {
        let mut out = vec![Complex64::new(0.0, 0.0); rows * cols];
        for j in 0..cols {
            for i in 0..rows {
                out[col_major_index(cols, j, i)] = mat[col_major_index(rows, i, j)].conj();
            }
        }
        out
    }

    fn matmul_f32(lhs: &[f32], rhs: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
        let mut out = vec![0.0; m * n];
        for j in 0..n {
            for p in 0..k {
                let rhs_pj = rhs[col_major_index(k, p, j)];
                for i in 0..m {
                    out[col_major_index(m, i, j)] += lhs[col_major_index(m, i, p)] * rhs_pj;
                }
            }
        }
        out
    }

    fn matmul_f64(lhs: &[f64], rhs: &[f64], m: usize, k: usize, n: usize) -> Vec<f64> {
        let mut out = vec![0.0; m * n];
        for j in 0..n {
            for p in 0..k {
                let rhs_pj = rhs[col_major_index(k, p, j)];
                for i in 0..m {
                    out[col_major_index(m, i, j)] += lhs[col_major_index(m, i, p)] * rhs_pj;
                }
            }
        }
        out
    }

    fn matmul_c32(
        lhs: &[Complex32],
        rhs: &[Complex32],
        m: usize,
        k: usize,
        n: usize,
    ) -> Vec<Complex32> {
        let mut out = vec![Complex32::new(0.0, 0.0); m * n];
        for j in 0..n {
            for p in 0..k {
                let rhs_pj = rhs[col_major_index(k, p, j)];
                for i in 0..m {
                    out[col_major_index(m, i, j)] += lhs[col_major_index(m, i, p)] * rhs_pj;
                }
            }
        }
        out
    }

    fn matmul_c64(
        lhs: &[Complex64],
        rhs: &[Complex64],
        m: usize,
        k: usize,
        n: usize,
    ) -> Vec<Complex64> {
        let mut out = vec![Complex64::new(0.0, 0.0); m * n];
        for j in 0..n {
            for p in 0..k {
                let rhs_pj = rhs[col_major_index(k, p, j)];
                for i in 0..m {
                    out[col_major_index(m, i, j)] += lhs[col_major_index(m, i, p)] * rhs_pj;
                }
            }
        }
        out
    }

    fn diag_f32(values: &[f32]) -> Vec<f32> {
        let mut out = vec![0.0; values.len() * values.len()];
        for (i, value) in values.iter().enumerate() {
            out[col_major_index(values.len(), i, i)] = *value;
        }
        out
    }

    fn diag_f64(values: &[f64]) -> Vec<f64> {
        let mut out = vec![0.0; values.len() * values.len()];
        for (i, value) in values.iter().enumerate() {
            out[col_major_index(values.len(), i, i)] = *value;
        }
        out
    }

    fn diag_c32_from_real(values: &[f32]) -> Vec<Complex32> {
        let mut out = vec![Complex32::new(0.0, 0.0); values.len() * values.len()];
        for (i, value) in values.iter().enumerate() {
            out[col_major_index(values.len(), i, i)] = Complex32::new(*value, 0.0);
        }
        out
    }

    fn diag_c64_from_real(values: &[f64]) -> Vec<Complex64> {
        let mut out = vec![Complex64::new(0.0, 0.0); values.len() * values.len()];
        for (i, value) in values.iter().enumerate() {
            out[col_major_index(values.len(), i, i)] = Complex64::new(*value, 0.0);
        }
        out
    }

    #[test]
    fn dense_view_rejects_out_of_bounds_layout() {
        let data = [0.0; 6];
        let shape = [2, 3];
        let strides = [1, 4];
        let err = DenseView::new(&data, &shape, &strides, 0).unwrap_err();
        assert_eq!(err, DenseError::OutOfBounds);
    }

    #[cfg(feature = "tenferro")]
    #[derive(Default)]
    struct CountingKernelBackend {
        matmul_calls: usize,
    }

    #[cfg(feature = "tenferro")]
    impl DenseKernelBackend for CountingKernelBackend {
        fn supports_matmul(&self, dtype: DenseDType) -> bool {
            dtype == DenseDType::F64
        }

        fn matmul_into(
            &mut self,
            output: DenseWrite<'_>,
            lhs: DenseRead<'_>,
            rhs: DenseRead<'_>,
        ) -> Result<(), DenseError> {
            self.matmul_calls += 1;
            match (output, lhs, rhs) {
                (DenseWrite::F64(mut output), DenseRead::F64(lhs), DenseRead::F64(rhs)) => {
                    assert_eq!(lhs.shape(), &[2, 2]);
                    assert_eq!(rhs.shape(), &[2, 2]);
                    output.data_mut().fill(7.0);
                    Ok(())
                }
                _ => panic!("CountingKernelBackend should receive only supported F64 matmul"),
            }
        }
    }

    #[cfg(feature = "tenferro")]
    #[test]
    fn default_executor_routes_supported_matmul_to_kernel_backend() {
        let lhs = [1.0, 2.0, 3.0, 4.0];
        let rhs = [5.0, 6.0, 7.0, 8.0];
        let mut output = [0.0; 4];
        let shape = [2, 2];
        let strides = [1, 2];

        let mut executor =
            DenseExecutorWithKernel::with_kernel_backend(CountingKernelBackend::default());
        executor
            .matmul_into(
                DenseWrite::F64(DenseViewMut::new(&mut output, &shape, &strides, 0).unwrap()),
                DenseRead::F64(DenseView::new(&lhs, &shape, &strides, 0).unwrap()),
                DenseRead::F64(DenseView::new(&rhs, &shape, &strides, 0).unwrap()),
            )
            .unwrap();

        assert_eq!(executor.kernel_backend().matmul_calls, 1);
        assert_eq!(output, [7.0; 4]);
    }

    #[cfg(feature = "tenferro")]
    #[test]
    fn default_executor_matmul_into_matches_tensorkit_recoupling_view_for_all_gemm_dtypes() {
        let lhs_shape = [2, 3];
        let lhs_strides = [1, 2];
        let rhs_shape = [3, 2];
        let rhs_strides = [1, 3];
        let out_shape = [2, 2];
        let out_strides = [1, 4];
        let out_offset = 1;

        let mut executor = DefaultDenseExecutor::new();

        let lhs_f32 = vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let u_f32 = vec![10.0_f32, 100.0, 1000.0, 20.0, 200.0, 2000.0];
        let mut out_f32 = vec![-1.0_f32; 8];
        executor
            .matmul_into(
                DenseWrite::F32(
                    DenseViewMut::new(&mut out_f32, &out_shape, &out_strides, out_offset).unwrap(),
                ),
                DenseRead::F32(DenseView::new(&lhs_f32, &lhs_shape, &lhs_strides, 0).unwrap()),
                DenseRead::F32(DenseView::new(&u_f32, &rhs_shape, &rhs_strides, 0).unwrap()),
            )
            .unwrap();
        assert_eq!(
            out_f32,
            vec![-1.0, 5310.0, 6420.0, -1.0, -1.0, 10620.0, 12840.0, -1.0]
        );

        let lhs_f64 = vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0];
        let u_f64 = vec![10.0_f64, 100.0, 1000.0, 20.0, 200.0, 2000.0];
        let mut out_f64 = vec![-1.0_f64; 8];
        executor
            .matmul_into(
                DenseWrite::F64(
                    DenseViewMut::new(&mut out_f64, &out_shape, &out_strides, out_offset).unwrap(),
                ),
                DenseRead::F64(DenseView::new(&lhs_f64, &lhs_shape, &lhs_strides, 0).unwrap()),
                DenseRead::F64(DenseView::new(&u_f64, &rhs_shape, &rhs_strides, 0).unwrap()),
            )
            .unwrap();
        assert_eq!(
            out_f64,
            vec![-1.0, 5310.0, 6420.0, -1.0, -1.0, 10620.0, 12840.0, -1.0]
        );

        let lhs_c32 = lhs_f32
            .iter()
            .map(|&value| Complex32::new(value, 0.0))
            .collect::<Vec<_>>();
        let u_c32 = u_f32
            .iter()
            .map(|&value| Complex32::new(value, 0.0))
            .collect::<Vec<_>>();
        let mut out_c32 = vec![Complex32::new(-1.0, -2.0); 8];
        executor
            .matmul_into(
                DenseWrite::C32(
                    DenseViewMut::new(&mut out_c32, &out_shape, &out_strides, out_offset).unwrap(),
                ),
                DenseRead::C32(DenseView::new(&lhs_c32, &lhs_shape, &lhs_strides, 0).unwrap()),
                DenseRead::C32(DenseView::new(&u_c32, &rhs_shape, &rhs_strides, 0).unwrap()),
            )
            .unwrap();
        assert_eq!(
            out_c32,
            vec![
                Complex32::new(-1.0, -2.0),
                Complex32::new(5310.0, 0.0),
                Complex32::new(6420.0, 0.0),
                Complex32::new(-1.0, -2.0),
                Complex32::new(-1.0, -2.0),
                Complex32::new(10620.0, 0.0),
                Complex32::new(12840.0, 0.0),
                Complex32::new(-1.0, -2.0),
            ]
        );

        let lhs_c64 = lhs_f64
            .iter()
            .map(|&value| Complex64::new(value, 0.0))
            .collect::<Vec<_>>();
        let u_c64 = u_f64
            .iter()
            .map(|&value| Complex64::new(value, 0.0))
            .collect::<Vec<_>>();
        let mut out_c64 = vec![Complex64::new(-1.0, -2.0); 8];
        executor
            .matmul_into(
                DenseWrite::C64(
                    DenseViewMut::new(&mut out_c64, &out_shape, &out_strides, out_offset).unwrap(),
                ),
                DenseRead::C64(DenseView::new(&lhs_c64, &lhs_shape, &lhs_strides, 0).unwrap()),
                DenseRead::C64(DenseView::new(&u_c64, &rhs_shape, &rhs_strides, 0).unwrap()),
            )
            .unwrap();
        assert_eq!(
            out_c64,
            vec![
                Complex64::new(-1.0, -2.0),
                Complex64::new(5310.0, 0.0),
                Complex64::new(6420.0, 0.0),
                Complex64::new(-1.0, -2.0),
                Complex64::new(-1.0, -2.0),
                Complex64::new(10620.0, 0.0),
                Complex64::new(12840.0, 0.0),
                Complex64::new(-1.0, -2.0),
            ]
        );
    }

    #[cfg(feature = "tenferro")]
    #[test]
    fn default_executor_qr_reads_transposed_views_for_all_linalg_dtypes() {
        let f32_data = vec![1.0_f32, -2.0, 3.0, 0.5, -1.0, 4.0];
        let f64_data = vec![1.0_f64, -2.0, 3.0, 0.5, -1.0, 4.0];
        let c32_data = vec![
            Complex32::new(1.0, 0.5),
            Complex32::new(-2.0, 1.0),
            Complex32::new(3.0, -0.25),
            Complex32::new(0.5, -1.0),
            Complex32::new(-1.0, 0.75),
            Complex32::new(4.0, 1.5),
        ];
        let c64_data = vec![
            Complex64::new(1.0, 0.5),
            Complex64::new(-2.0, 1.0),
            Complex64::new(3.0, -0.25),
            Complex64::new(0.5, -1.0),
            Complex64::new(-1.0, 0.75),
            Complex64::new(4.0, 1.5),
        ];
        let shape = [3, 2];
        let strides = [2, 1];
        let mut executor = DefaultDenseExecutor::new();

        let outputs = executor
            .qr(DenseRead::F32(
                DenseView::new(&f32_data, &shape, &strides, 0).unwrap(),
            ))
            .unwrap();
        assert_eq!(outputs[0].dtype(), DenseDType::F32);
        let recon = matmul_f32(
            outputs[0].as_f32_slice().unwrap(),
            outputs[1].as_f32_slice().unwrap(),
            3,
            2,
            2,
        );
        let expected = transpose_f32(&f32_data, 2, 3);
        for (actual, expected) in recon.iter().zip(expected.iter()) {
            assert_f32_close(*actual, *expected, 1.0e-5);
        }

        let outputs = executor
            .qr(DenseRead::F64(
                DenseView::new(&f64_data, &shape, &strides, 0).unwrap(),
            ))
            .unwrap();
        assert_eq!(outputs[0].dtype(), DenseDType::F64);
        let recon = matmul_f64(
            outputs[0].as_f64_slice().unwrap(),
            outputs[1].as_f64_slice().unwrap(),
            3,
            2,
            2,
        );
        let expected = transpose_f64(&f64_data, 2, 3);
        for (actual, expected) in recon.iter().zip(expected.iter()) {
            assert_f64_close(*actual, *expected, 1.0e-9);
        }

        let outputs = executor
            .qr(DenseRead::C32(
                DenseView::new(&c32_data, &shape, &strides, 0).unwrap(),
            ))
            .unwrap();
        assert_eq!(outputs[0].dtype(), DenseDType::C32);
        let recon = matmul_c32(
            outputs[0].as_c32_slice().unwrap(),
            outputs[1].as_c32_slice().unwrap(),
            3,
            2,
            2,
        );
        let expected = transpose_c32(&c32_data, 2, 3);
        for (actual, expected) in recon.iter().zip(expected.iter()) {
            assert_c32_close(*actual, *expected, 1.0e-5);
        }

        let outputs = executor
            .qr(DenseRead::C64(
                DenseView::new(&c64_data, &shape, &strides, 0).unwrap(),
            ))
            .unwrap();
        assert_eq!(outputs[0].dtype(), DenseDType::C64);
        let recon = matmul_c64(
            outputs[0].as_c64_slice().unwrap(),
            outputs[1].as_c64_slice().unwrap(),
            3,
            2,
            2,
        );
        let expected = transpose_c64(&c64_data, 2, 3);
        for (actual, expected) in recon.iter().zip(expected.iter()) {
            assert_c64_close(*actual, *expected, 1.0e-9);
        }
    }

    #[cfg(feature = "tenferro")]
    #[test]
    fn default_executor_eigh_reads_transposed_views_for_all_linalg_dtypes() {
        let f32_data = vec![4.0_f32, 1.0, 1.0, 3.0];
        let f64_data = vec![4.0_f64, 1.0, 1.0, 3.0];
        let c32_data = vec![
            Complex32::new(4.0, 0.0),
            Complex32::new(1.0, -0.5),
            Complex32::new(1.0, 0.5),
            Complex32::new(3.0, 0.0),
        ];
        let c64_data = vec![
            Complex64::new(4.0, 0.0),
            Complex64::new(1.0, -0.5),
            Complex64::new(1.0, 0.5),
            Complex64::new(3.0, 0.0),
        ];
        let shape = [2, 2];
        let strides = [2, 1];
        let mut executor = DefaultDenseExecutor::new();

        let outputs = executor
            .eigh(DenseRead::F32(
                DenseView::new(&f32_data, &shape, &strides, 0).unwrap(),
            ))
            .unwrap();
        assert_eq!(outputs[0].dtype(), DenseDType::F32);
        assert_eq!(outputs[1].dtype(), DenseDType::F32);
        let values = outputs[0].as_f32_slice().unwrap();
        let vectors = outputs[1].as_f32_slice().unwrap();
        let recon = matmul_f32(
            &matmul_f32(vectors, &diag_f32(values), 2, 2, 2),
            &transpose_f32(vectors, 2, 2),
            2,
            2,
            2,
        );
        let expected = transpose_f32(&f32_data, 2, 2);
        for (actual, expected) in recon.iter().zip(expected.iter()) {
            assert_f32_close(*actual, *expected, 1.0e-5);
        }

        let outputs = executor
            .eigh(DenseRead::F64(
                DenseView::new(&f64_data, &shape, &strides, 0).unwrap(),
            ))
            .unwrap();
        assert_eq!(outputs[0].dtype(), DenseDType::F64);
        assert_eq!(outputs[1].dtype(), DenseDType::F64);
        let values = outputs[0].as_f64_slice().unwrap();
        let vectors = outputs[1].as_f64_slice().unwrap();
        let recon = matmul_f64(
            &matmul_f64(vectors, &diag_f64(values), 2, 2, 2),
            &transpose_f64(vectors, 2, 2),
            2,
            2,
            2,
        );
        let expected = transpose_f64(&f64_data, 2, 2);
        for (actual, expected) in recon.iter().zip(expected.iter()) {
            assert_f64_close(*actual, *expected, 1.0e-10);
        }

        let outputs = executor
            .eigh(DenseRead::C32(
                DenseView::new(&c32_data, &shape, &strides, 0).unwrap(),
            ))
            .unwrap();
        assert_eq!(outputs[0].dtype(), DenseDType::F32);
        assert_eq!(outputs[1].dtype(), DenseDType::C32);
        let values = outputs[0].as_f32_slice().unwrap();
        let vectors = outputs[1].as_c32_slice().unwrap();
        let recon = matmul_c32(
            &matmul_c32(vectors, &diag_c32_from_real(values), 2, 2, 2),
            &conjugate_transpose_c32(vectors, 2, 2),
            2,
            2,
            2,
        );
        let expected = transpose_c32(&c32_data, 2, 2);
        for (actual, expected) in recon.iter().zip(expected.iter()) {
            assert_c32_close(*actual, *expected, 1.0e-5);
        }

        let outputs = executor
            .eigh(DenseRead::C64(
                DenseView::new(&c64_data, &shape, &strides, 0).unwrap(),
            ))
            .unwrap();
        assert_eq!(outputs[0].dtype(), DenseDType::F64);
        assert_eq!(outputs[1].dtype(), DenseDType::C64);
        let values = outputs[0].as_f64_slice().unwrap();
        let vectors = outputs[1].as_c64_slice().unwrap();
        let recon = matmul_c64(
            &matmul_c64(vectors, &diag_c64_from_real(values), 2, 2, 2),
            &conjugate_transpose_c64(vectors, 2, 2),
            2,
            2,
            2,
        );
        let expected = transpose_c64(&c64_data, 2, 2);
        for (actual, expected) in recon.iter().zip(expected.iter()) {
            assert_c64_close(*actual, *expected, 1.0e-10);
        }
    }

    #[cfg(feature = "tenferro")]
    #[test]
    fn default_executor_svd_accepts_all_supported_linalg_dtypes() {
        let f32_data = [1.0_f32, -2.0, 0.5, 4.0];
        let f64_data = [1.0_f64, -2.0, 0.5, 4.0];
        let c32_data = [
            Complex32::new(1.0, 0.5),
            Complex32::new(-2.0, 1.0),
            Complex32::new(0.5, -0.25),
            Complex32::new(4.0, 1.5),
        ];
        let c64_data = [
            Complex64::new(1.0, 0.5),
            Complex64::new(-2.0, 1.0),
            Complex64::new(0.5, -0.25),
            Complex64::new(4.0, 1.5),
        ];
        let shape = [2, 2];
        let strides = [2, 1];

        let mut executor = DefaultDenseExecutor::new();
        for (input, dtype) in [
            (
                DenseRead::F32(DenseView::new(&f32_data, &shape, &strides, 0).unwrap()),
                DenseDType::F32,
            ),
            (
                DenseRead::F64(DenseView::new(&f64_data, &shape, &strides, 0).unwrap()),
                DenseDType::F64,
            ),
            (
                DenseRead::C32(DenseView::new(&c32_data, &shape, &strides, 0).unwrap()),
                DenseDType::C32,
            ),
            (
                DenseRead::C64(DenseView::new(&c64_data, &shape, &strides, 0).unwrap()),
                DenseDType::C64,
            ),
        ] {
            let outputs = executor.svd(input).unwrap();
            assert_eq!(outputs[0].dtype(), dtype);
            assert!(matches!(
                (dtype, outputs[1].dtype()),
                (DenseDType::F32, DenseDType::F32)
                    | (DenseDType::F64, DenseDType::F64)
                    | (DenseDType::C32, DenseDType::F32)
                    | (DenseDType::C64, DenseDType::F64)
            ));
            assert_eq!(outputs[2].dtype(), dtype);
        }
    }

    #[cfg(feature = "tenferro")]
    #[test]
    fn default_executor_rejects_integer_linalg_view() {
        let data = [1_i32, 0, 0, 1];
        let shape = [2, 2];
        let strides = [1, 2];
        let view = DenseView::new(&data, &shape, &strides, 0).unwrap();

        let mut executor = DefaultDenseExecutor::new();
        let err = executor.qr(DenseRead::I32(view)).unwrap_err();

        assert!(matches!(
            err,
            DenseError::Backend {
                backend: DenseBackend::Tenferro,
                op: "qr_read",
                ref message,
            } if message.contains("unsupported dtype")
        ));
    }
}
