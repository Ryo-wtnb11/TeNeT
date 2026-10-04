use super::*;

/// Facade error for checked Generic providers.
#[non_exhaustive]
#[derive(Debug)]
pub enum GenericTensorError<E> {
    /// Ordinary facade validation or payload construction failed.
    Facade(Error),
    /// Checked Generic provider or structural admission failed.
    Structure(CheckedGenericStructureError<E>),
    /// Checked Generic operation-plan construction or replay failed.
    Plan(CheckedGenericPlanError<E>),
    /// Checked Generic tensor-product preparation or execution failed.
    TensorProduct(CheckedGenericTensorProductError<E>),
}

impl<E: core::fmt::Display> core::fmt::Display for GenericTensorError<E> {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Facade(error) => error.fmt(formatter),
            Self::Structure(error) => error.fmt(formatter),
            Self::Plan(error) => error.fmt(formatter),
            Self::TensorProduct(error) => error.fmt(formatter),
        }
    }
}

impl<E: std::error::Error + 'static> std::error::Error for GenericTensorError<E> {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Facade(error) => Some(error),
            Self::Structure(error) => Some(error),
            Self::Plan(error) => Some(error),
            Self::TensorProduct(error) => Some(error),
        }
    }
}

impl<E> From<Error> for GenericTensorError<E> {
    fn from(error: Error) -> Self {
        Self::Facade(error)
    }
}

impl<E> From<tenet_tensors::OperationError> for GenericTensorError<E> {
    fn from(error: tenet_tensors::OperationError) -> Self {
        Self::Facade(error.into())
    }
}

/// Whether an error is an axis misuse, which the facade reports in
/// [`GenericTensorError::Facade`] exactly as the multiplicity-free mode does
/// (#1873): it is an argument error, not a plan failure. Below the facade it
/// has one variant per kind (#1989).
fn is_axis_misuse(error: &OperationError) -> bool {
    matches!(
        error,
        OperationError::InvalidPermutation { .. } | OperationError::InvalidAxisSet { .. }
    )
}

impl<E> From<CheckedGenericStructureError<E>> for GenericTensorError<E> {
    fn from(error: CheckedGenericStructureError<E>) -> Self {
        match error {
            CheckedGenericStructureError::Core(core) => match OperationError::from(core) {
                OperationError::Core(core) => {
                    Self::Structure(CheckedGenericStructureError::Core(core))
                }
                axis => Self::Facade(Error::from(axis)),
            },
            provider => Self::Structure(provider),
        }
    }
}

impl<E> From<CheckedGenericPlanError<E>> for GenericTensorError<E> {
    fn from(error: CheckedGenericPlanError<E>) -> Self {
        match error {
            CheckedGenericPlanError::Operation(operation) if is_axis_misuse(&operation) => {
                Self::Facade(Error::from(operation))
            }
            other => Self::Plan(other),
        }
    }
}

impl<E> From<CheckedGenericFactorPlanError<E>> for GenericTensorError<E> {
    fn from(error: CheckedGenericFactorPlanError<E>) -> Self {
        match error {
            CheckedGenericFactorPlanError::Provider(error) => {
                Self::Plan(CheckedGenericPlanError::Provider(error))
            }
            CheckedGenericFactorPlanError::Operation(error) => {
                Self::from(CheckedGenericPlanError::Operation(error))
            }
        }
    }
}

impl<E> From<CheckedGenericTensorProductError<E>> for GenericTensorError<E> {
    fn from(error: CheckedGenericTensorProductError<E>) -> Self {
        Self::TensorProduct(error)
    }
}
