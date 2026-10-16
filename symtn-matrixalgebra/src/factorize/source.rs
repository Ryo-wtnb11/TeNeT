use super::*;

/// The storage a checked factorization reads: dense coupled-sector storage,
/// or a compact diagonal on its bond space. A diagonal is factorized
/// directly (MAK `DiagonalAlgorithm`), so it never needs a dense executor.
pub enum FactorSource<'a, R, D> {
    Dense(BoundDynamicTensorRef<'a, R, D>),
    Diagonal {
        space: &'a BoundDynamicFusionMapSpace<R>,
        spectrum: &'a [SectorSpectrum<D>],
    },
}

/// A factor in the storage its route produced: a compact diagonal on
/// `space`, or dense storage.
pub enum FactorOutput<R, D> {
    Dense(BoundDynFactor<R, D>),
    Diagonal {
        space: BoundDynamicFusionMapSpace<R>,
        values: Vec<SectorSpectrum<D>>,
    },
}

impl<R, D> FactorOutput<R, D> {
    /// The dense factor, which every factor of a dense source is.
    pub fn into_dense(self) -> Option<BoundDynFactor<R, D>> {
        match self {
            Self::Dense(factor) => Some(factor),
            Self::Diagonal { .. } => None,
        }
    }
}

/// The dense executor of a factorization's dense stage, obtained only when
/// that stage runs: a compact diagonal never needs one. A plain `&mut E` is
/// one; the facade's runtime lease is another.
pub trait ExecutorLease {
    type Executor: DenseExecutor + ?Sized;

    fn run<T>(self, stage: impl FnOnce(&mut Self::Executor) -> T) -> T;
}

impl<E: DenseExecutor + ?Sized> ExecutorLease for &mut E {
    type Executor = E;

    fn run<T>(self, stage: impl FnOnce(&mut E) -> T) -> T {
        stage(self)
    }
}

/// Runs the diagonal or the dense stage of `source`; only the dense stage
/// leases an executor.
///
/// `dense_finite` names the family whose shared finite-input stage
/// ([`require_finite_factor_input`], #1986) runs on dense storage here,
/// before the lease, because that family's dense route has no structural
/// admission to run first. `None` leaves the check to a dense route that
/// admits structure first (eig/eigh: endomorphism and stacking; polar: its
/// block shape), so every route reports structure before values, as the
/// diagonal route does.
pub(super) fn factor_from_source<L, E, R, D, T, X>(
    lease: L,
    source: FactorSource<'_, R, D>,
    dense_finite: Option<FactorFamily>,
    diagonal: impl FnOnce(&BoundDynamicFusionMapSpace<R>, &[SectorSpectrum<D>]) -> Result<T, X>,
    dense: impl FnOnce(&mut E, &BoundDynamicTensorRef<'_, R, D>) -> Result<T, X>,
) -> Result<T, X>
where
    L: ExecutorLease<Executor = E>,
    E: DenseExecutor + ?Sized,
    D: FactorScalar,
    X: From<OperationError>,
{
    match source {
        FactorSource::Dense(input) => {
            if let Some(family) = dense_finite {
                require_finite_factor_input(input.data().iter().copied(), family)?;
            }
            lease.run(|executor| dense(executor, &input))
        }
        FactorSource::Diagonal { space, spectrum } => diagonal(space, spectrum),
    }
}
