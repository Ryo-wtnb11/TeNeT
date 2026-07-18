use num_complex::{Complex32, Complex64};

use crate::{DenseBackend, DenseDType, DenseError};

#[cfg(feature = "tenferro")]
use crate::dtype::dense_dtype_from_tenferro;
#[cfg(feature = "tenferro")]
use crate::tenferro_adapter::tenferro_error;

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
    pub(crate) fn from_tenferro(tensor: tenferro_tensor::Tensor) -> Self {
        Self {
            backend: DenseBackend::Tenferro,
            inner: DenseTensorInner::Tenferro(tensor),
        }
    }
}
