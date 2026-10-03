use super::*;

#[doc(hidden)]
pub trait TypedTensorInvDispatch<R, D>: TypedTensorModeDispatch<R>
where
    R: TypedSectorAdmission,
    D: AdvancedLinalgScalar,
{
    fn inv(tensor: &TensorMap<R, D>) -> Result<TensorMap<R, D>, Self::FacadeError>;
}

#[doc(hidden)]
pub trait TypedTensorSolveDispatch<R, D>: TypedTensorModeDispatch<R>
where
    R: TypedSectorAdmission,
    D: AdvancedLinalgScalar,
{
    fn solve(
        tensor: &TensorMap<R, D>,
        rhs: &TensorMap<R, D>,
    ) -> Result<TensorMap<R, D>, Self::FacadeError>;
}

#[doc(hidden)]
pub trait TypedTensorPinvDispatch<R, D>: TypedTensorModeDispatch<R>
where
    R: TypedSectorAdmission,
    D: AdvancedLinalgScalar,
{
    fn pinv(tensor: &TensorMap<R, D>, rcond: f64) -> Result<TensorMap<R, D>, Self::FacadeError>;
}

#[doc(hidden)]
pub trait TypedTensorNullDispatch<R, D>: TypedTensorModeDispatch<R>
where
    R: TypedSectorAdmission,
    D: FactorizationScalar,
{
    fn left_null(tensor: &TensorMap<R, D>) -> Result<TensorMap<R, D>, Self::FacadeError>;
    fn right_null(tensor: &TensorMap<R, D>) -> Result<TensorMap<R, D>, Self::FacadeError>;
}

#[doc(hidden)]
pub trait TypedTensorPolarDispatch<R, D>: TypedTensorModeDispatch<R>
where
    R: TypedSectorAdmission,
    D: FactorizationScalar,
{
    fn left_polar(
        tensor: &TensorMap<R, D>,
    ) -> Result<LeftPolar<TensorMap<R, D>>, Self::FacadeError>;
    fn right_polar(
        tensor: &TensorMap<R, D>,
    ) -> Result<RightPolar<TensorMap<R, D>>, Self::FacadeError>;
}

#[doc(hidden)]
pub trait TypedTensorExpDispatch<R, D>: TypedTensorModeDispatch<R>
where
    R: TypedSectorAdmission,
    D: AdvancedLinalgScalar,
{
    fn exp(tensor: &TensorMap<R, D>) -> Result<TensorMap<R, D>, Self::FacadeError>;
}

#[doc(hidden)]
pub trait TypedTensorQrDispatch<R, D>: TypedTensorModeDispatch<R>
where
    R: TypedSectorAdmission,
    D: FactorizationScalar,
{
    fn qr_compact(tensor: &TensorMap<R, D>) -> Result<Qr<TensorMap<R, D>>, Self::FacadeError>;
}

#[doc(hidden)]
pub trait TypedTensorSvdDispatch<R, D>: TypedTensorModeDispatch<R>
where
    R: TypedSectorAdmission,
    D: FactorizationScalar,
{
    fn svd_compact(tensor: &TensorMap<R, D>) -> Result<Svd<TensorMap<R, D>>, Self::FacadeError>;
    fn svd_full(tensor: &TensorMap<R, D>) -> Result<Svd<TensorMap<R, D>>, Self::FacadeError>;
}

#[doc(hidden)]
pub trait TypedTensorLqDispatch<R, D>: TypedTensorModeDispatch<R>
where
    R: TypedSectorAdmission,
    D: FactorizationScalar,
{
    fn lq_compact(tensor: &TensorMap<R, D>) -> Result<Lq<TensorMap<R, D>>, Self::FacadeError>;
}

#[doc(hidden)]
pub trait TypedTensorFullQrDispatch<R, D>: TypedTensorModeDispatch<R>
where
    R: TypedSectorAdmission,
    D: FactorizationScalar,
{
    fn qr_full(tensor: &TensorMap<R, D>) -> Result<Qr<TensorMap<R, D>>, Self::FacadeError>;
}

#[doc(hidden)]
pub trait TypedTensorFullLqDispatch<R, D>: TypedTensorModeDispatch<R>
where
    R: TypedSectorAdmission,
    D: FactorizationScalar,
{
    fn lq_full(tensor: &TensorMap<R, D>) -> Result<Lq<TensorMap<R, D>>, Self::FacadeError>;
}

#[doc(hidden)]
pub trait TypedTensorSvdValsDispatch<R, D>: TypedTensorModeDispatch<R>
where
    R: TypedSectorAdmission,
    D: FactorizationScalar,
{
    fn svd_vals(
        tensor: &TensorMap<R, D>,
    ) -> Result<Vec<SectorSpectrum<<R as TypedSectorAdmission>::Sector, f64>>, Self::FacadeError>;
}

#[doc(hidden)]
pub trait TypedTensorEighValsDispatch<R, D>: TypedTensorModeDispatch<R>
where
    R: TypedSectorAdmission,
    D: FactorizationScalar,
{
    fn eigh_vals(
        tensor: &TensorMap<R, D>,
    ) -> Result<Vec<SectorSpectrum<<R as TypedSectorAdmission>::Sector, f64>>, Self::FacadeError>;
}

#[doc(hidden)]
pub trait TypedTensorEighDispatch<R, D>: TypedTensorModeDispatch<R>
where
    R: TypedSectorAdmission,
    D: FactorizationScalar,
{
    fn eigh_full(tensor: &TensorMap<R, D>) -> Result<Eigh<TensorMap<R, D>>, Self::FacadeError>;
}

#[doc(hidden)]
pub trait TypedTensorEigValsDispatch<R, D>: TypedTensorModeDispatch<R>
where
    R: TypedSectorAdmission,
    D: AdvancedLinalgScalar,
{
    fn eig_vals(
        tensor: &TensorMap<R, D>,
    ) -> Result<
        Vec<SectorSpectrum<<R as TypedSectorAdmission>::Sector, num_complex::Complex64>>,
        Self::FacadeError,
    >;
}

#[doc(hidden)]
pub trait TypedTensorEigDispatch<R, D>: TypedTensorModeDispatch<R>
where
    R: TypedSectorAdmission,
    D: AdvancedLinalgScalar,
{
    fn eig_full(
        tensor: &TensorMap<R, D>,
    ) -> Result<Eig<TensorMap<R, <D as FactorScalar>::Eig>>, Self::FacadeError>;
}
