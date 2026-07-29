use tenet_core::CoreError;

use crate::{BoundDynamicFusionMapSpace, OperationError};

/// Borrowed dynamic tensor input whose provider, complete tree grid, rank, and
/// storage length have been validated before an expert operation can run.
pub struct BoundDynamicTensorRef<'a, R, D> {
    space: &'a BoundDynamicFusionMapSpace<R>,
    data: &'a [D],
}

impl<'a, R, D> BoundDynamicTensorRef<'a, R, D> {
    /// Validates that `data` is exactly the flat storage for `space`.
    pub fn try_new(
        space: &'a BoundDynamicFusionMapSpace<R>,
        data: &'a [D],
    ) -> Result<Self, OperationError> {
        let raw = space.space();
        let hom_rank = raw.homspace().codomain().len() + raw.homspace().domain().len();
        if raw.rank() != hom_rank {
            return Err(OperationError::from_core_preserving_context(
                CoreError::StructureRankMismatch {
                    expected: hom_rank,
                    actual: raw.rank(),
                },
            ));
        }
        if raw.structure().rank() != raw.rank() {
            return Err(OperationError::from_core_preserving_context(
                CoreError::StructureRankMismatch {
                    expected: raw.rank(),
                    actual: raw.structure().rank(),
                },
            ));
        }
        let expected = raw.required_len()?;
        if data.len() != expected {
            return Err(OperationError::from_core_preserving_context(
                CoreError::DimensionMismatch {
                    expected,
                    actual: data.len(),
                },
            ));
        }
        Ok(Self { space, data })
    }

    #[inline]
    pub fn space(&self) -> &BoundDynamicFusionMapSpace<R> {
        self.space
    }

    #[inline]
    pub fn data(&self) -> &'a [D] {
        self.data
    }
}
