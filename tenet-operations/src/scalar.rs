use core::ops::{Add, Mul};

use num_complex::{Complex32, Complex64};
use num_traits::{One, Zero};
use tenet_dense::{DenseRead, DenseView, DenseViewMut, DenseWrite};

pub trait TreeTransformScalar:
    Copy
    + Add<Self, Output = Self>
    + Mul<Self, Output = Self>
    + PartialEq
    + Zero
    + One
    + strided_kernel::MaybeSendSync
{
}

impl<T> TreeTransformScalar for T where
    T: Copy
        + Add<T, Output = T>
        + Mul<T, Output = T>
        + PartialEq
        + Zero
        + One
        + strided_kernel::MaybeSendSync
{
}

/// Action of a categorical recoupling coefficient on tensor storage data.
///
/// TensorKit allows, for example, real SU(2) coefficients to act on complex
/// tensor blocks. Rust needs that conversion boundary to be explicit.
pub trait RecouplingCoefficientAction<C>: Copy {
    fn scale_by_coefficient(self, coefficient: C) -> Self;
    fn coefficient_as_data(coefficient: C) -> Self;
}

macro_rules! impl_same_recoupling_coefficient_action {
    ($($ty:ty),+ $(,)?) => {
        $(
            impl RecouplingCoefficientAction<$ty> for $ty {
                #[inline]
                fn scale_by_coefficient(self, coefficient: $ty) -> Self {
                    self * coefficient
                }

                #[inline]
                fn coefficient_as_data(coefficient: $ty) -> Self {
                    coefficient
                }
            }
        )+
    };
}

impl_same_recoupling_coefficient_action!(f32, f64, i32, i64, Complex32, Complex64);

impl RecouplingCoefficientAction<f64> for f32 {
    #[inline]
    fn scale_by_coefficient(self, coefficient: f64) -> Self {
        self * coefficient as f32
    }

    #[inline]
    fn coefficient_as_data(coefficient: f64) -> Self {
        coefficient as f32
    }
}

impl RecouplingCoefficientAction<f32> for f64 {
    #[inline]
    fn scale_by_coefficient(self, coefficient: f32) -> Self {
        self * f64::from(coefficient)
    }

    #[inline]
    fn coefficient_as_data(coefficient: f32) -> Self {
        f64::from(coefficient)
    }
}

impl RecouplingCoefficientAction<f32> for Complex32 {
    #[inline]
    fn scale_by_coefficient(self, coefficient: f32) -> Self {
        self * coefficient
    }

    #[inline]
    fn coefficient_as_data(coefficient: f32) -> Self {
        Self::new(coefficient, 0.0)
    }
}

impl RecouplingCoefficientAction<f64> for Complex32 {
    #[inline]
    fn scale_by_coefficient(self, coefficient: f64) -> Self {
        self * coefficient as f32
    }

    #[inline]
    fn coefficient_as_data(coefficient: f64) -> Self {
        Self::new(coefficient as f32, 0.0)
    }
}

impl RecouplingCoefficientAction<f32> for Complex64 {
    #[inline]
    fn scale_by_coefficient(self, coefficient: f32) -> Self {
        self * f64::from(coefficient)
    }

    #[inline]
    fn coefficient_as_data(coefficient: f32) -> Self {
        Self::new(f64::from(coefficient), 0.0)
    }
}

impl RecouplingCoefficientAction<f64> for Complex64 {
    #[inline]
    fn scale_by_coefficient(self, coefficient: f64) -> Self {
        self * coefficient
    }

    #[inline]
    fn coefficient_as_data(coefficient: f64) -> Self {
        Self::new(coefficient, 0.0)
    }
}

#[doc(hidden)]
pub trait DenseBlockScalar:
    Copy
    + Add<Self, Output = Self>
    + Mul<Self, Output = Self>
    + PartialEq
    + Zero
    + One
    + strided_kernel::MaybeSendSync
    + 'static
{
    fn dense_read(view: DenseView<'_, Self>) -> DenseRead<'_>;
    fn dense_write(view: DenseViewMut<'_, Self>) -> DenseWrite<'_>;
}

#[doc(hidden)]
pub trait DenseRecouplingScalar: DenseBlockScalar + RecouplingCoefficientAction<Self> {}

impl<T> DenseRecouplingScalar for T where T: DenseBlockScalar + RecouplingCoefficientAction<Self> {}

macro_rules! impl_dense_block_scalar {
    ($ty:ty, $read_variant:ident, $write_variant:ident) => {
        impl DenseBlockScalar for $ty {
            fn dense_read(view: DenseView<'_, Self>) -> DenseRead<'_> {
                DenseRead::$read_variant(view)
            }

            fn dense_write(view: DenseViewMut<'_, Self>) -> DenseWrite<'_> {
                DenseWrite::$write_variant(view)
            }
        }
    };
}

impl_dense_block_scalar!(f32, F32, F32);
impl_dense_block_scalar!(f64, F64, F64);
impl_dense_block_scalar!(Complex32, C32, C32);
impl_dense_block_scalar!(Complex64, C64, C64);
