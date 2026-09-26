//! Expert layer: representation queries on [`crate::prelude::TensorMap`].
//!
//! These answer how a tensor is stored, not what it is mathematically, so
//! they stay out of the user-layer method set. [`crate::prelude::TensorMap::diagview`]
//! is the user-layer reader of a diagonal, whatever the storage.

use crate::typed::{
    Error, SectorSpectrum, TensorMap, TensorScalar, TypedFacadeError, TypedTensorRootDispatch,
};
use tenet_core::{
    CheckedFusionAlgebra, MultiplicityFreeRigidSymbols, SectorCodec, TypedSectorAdmission,
};

/// Returns the compact diagonal spectrum without materializing dense data.
///
/// [`None`] means the representation has no directly stored compact spectrum
/// (it is dense or a lazy adjoint); otherwise it clones only the
/// `O(Σ_c k_c)` compact values in canonical bond-sector order.
#[expect(
    clippy::type_complexity,
    reason = "the compact readback exposes provider-labelled sector spectra"
)]
pub fn diagonal_spectrum<R, D>(
    tensor: &TensorMap<R, D>,
) -> Result<Option<Vec<SectorSpectrum<R::Sector, D>>>, TypedFacadeError<R>>
where
    R: TypedSectorAdmission,
    R::Mode: TypedTensorRootDispatch<R>,
    D: TensorScalar,
{
    tensor.diagonal_spectrum()
}

/// Tests a rank-one map for blockwise diagonality without materializing
/// compact storage.
///
/// Compact storage is diagonal by construction and returns `true` without a
/// scan. This matches TensorKit `isdiag` for finite data at `tol = 0`;
/// positive tolerance uses `max_offdiag <= tol * max(norm(Inf), 1)`.
/// Negative and non-finite tolerances are rejected before every shortcut.
/// Scale `tol` to the payload dtype, as
/// [`FactorizationScalar`](crate::typed::FactorizationScalar) describes.
pub fn is_diagonal<R, D>(tensor: &TensorMap<R, D>, tol: f64) -> Result<bool, Error>
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64> + CheckedFusionAlgebra + SectorCodec,
    D: TensorScalar,
{
    tensor.is_diagonal(tol)
}
