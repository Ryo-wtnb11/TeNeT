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

/// The storage route that produced a factorization. A diagonal factor of a
/// diagonal input lives on the input bond (`W = V`, TensorKit's diagonal
/// convention), so callers building such a factor need to know the route.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FactorRoute {
    Diagonal,
    Dense,
}

/// A checked factorization result with the route that produced it.
pub type Routed<T, E> = Result<(T, FactorRoute), CheckedGenericFactorPlanError<E>>;

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
pub(super) fn factor_from_source<L, E, R, D, T, X>(
    lease: L,
    source: FactorSource<'_, R, D>,
    diagonal: impl FnOnce(&BoundDynamicFusionMapSpace<R>, &[SectorSpectrum<D>]) -> Result<T, X>,
    dense: impl FnOnce(&mut E, &BoundDynamicTensorRef<'_, R, D>) -> Result<T, X>,
) -> Result<(T, FactorRoute), X>
where
    L: ExecutorLease<Executor = E>,
    E: DenseExecutor + ?Sized,
{
    match source {
        FactorSource::Dense(input) => Ok((
            lease.run(|executor| dense(executor, &input))?,
            FactorRoute::Dense,
        )),
        FactorSource::Diagonal { space, spectrum } => {
            Ok((diagonal(space, spectrum)?, FactorRoute::Diagonal))
        }
    }
}
