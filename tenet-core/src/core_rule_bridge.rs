use core::fmt;
use std::hash::Hash;

use crate::{
    FermionParityFusionRule, FusionAlgebraError, FusionTreePairKey, MultiplicityFreeFusionRule,
    MultiplicityFreeFusionSymbols, ProductSectorCodecError, SectorId, U1FusionRule, U1Irrep,
    Z2FusionRule, Z2Irrep,
};

// Why not tenet-sectors: these traits and errors define FusionTree lowering
// and pivotal operations over core-owned tree keys.
pub(crate) mod lowered_multiplicity_free_sealed {
    pub trait Sealed {}
}

#[derive(Clone, Debug, Eq, PartialEq)]
enum LoweredFusionTreeBuildErrorKind {
    InvalidSector(SectorId),
    Codec(ProductSectorCodecError),
    FusionAlgebra(Box<FusionAlgebraError>),
}

/// Failure while lowering encoded sectors into the built-in multiplicity-free
/// algebra used by the fusion-tree layout builder.
#[doc(hidden)]
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct LoweredFusionTreeBuildError {
    kind: LoweredFusionTreeBuildErrorKind,
}

impl LoweredFusionTreeBuildError {
    pub(crate) fn invalid_sector(sector: SectorId) -> Self {
        Self {
            kind: LoweredFusionTreeBuildErrorKind::InvalidSector(sector),
        }
    }

    pub(crate) fn codec(error: ProductSectorCodecError) -> Self {
        Self {
            kind: LoweredFusionTreeBuildErrorKind::Codec(error),
        }
    }

    pub(crate) fn fusion_algebra(error: FusionAlgebraError) -> Self {
        Self {
            kind: LoweredFusionTreeBuildErrorKind::FusionAlgebra(Box::new(error)),
        }
    }

    /// Extracts an exact finite-algebra cause without string classification.
    #[doc(hidden)]
    pub fn into_fusion_algebra(self) -> Result<FusionAlgebraError, Self> {
        match self.kind {
            LoweredFusionTreeBuildErrorKind::FusionAlgebra(error) => Ok(*error),
            kind => Err(Self { kind }),
        }
    }

    /// Converts every lowered built-in failure into the checked-algebra error
    /// vocabulary without discarding invalid-sector or product-codec details.
    #[doc(hidden)]
    pub fn into_checked_fusion_algebra(self) -> FusionAlgebraError {
        match self.kind {
            LoweredFusionTreeBuildErrorKind::InvalidSector(sector) => {
                FusionAlgebraError::InvalidSector { sector }
            }
            LoweredFusionTreeBuildErrorKind::Codec(error) => {
                FusionAlgebraError::ProductCodec(error)
            }
            LoweredFusionTreeBuildErrorKind::FusionAlgebra(error) => *error,
        }
    }

    #[doc(hidden)]
    pub const fn static_message(&self) -> &'static str {
        match &self.kind {
            LoweredFusionTreeBuildErrorKind::InvalidSector(_) => {
                "built-in fusion-tree layout contains an invalid sector"
            }
            LoweredFusionTreeBuildErrorKind::Codec(_) => {
                "built-in fusion-tree layout contains an invalid product sector"
            }
            LoweredFusionTreeBuildErrorKind::FusionAlgebra(_) => {
                "built-in fusion-tree layout exceeds the representable algebra"
            }
        }
    }
}

impl fmt::Display for LoweredFusionTreeBuildError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.kind {
            LoweredFusionTreeBuildErrorKind::InvalidSector(sector) => {
                write!(formatter, "invalid built-in sector {sector:?}")
            }
            LoweredFusionTreeBuildErrorKind::Codec(error) => error.fmt(formatter),
            LoweredFusionTreeBuildErrorKind::FusionAlgebra(error) => error.fmt(formatter),
        }
    }
}

impl std::error::Error for LoweredFusionTreeBuildError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match &self.kind {
            LoweredFusionTreeBuildErrorKind::FusionAlgebra(error) => Some(error.as_ref()),
            _ => None,
        }
    }
}

/// Typed algebra used only while building built-in multiplicity-free layouts.
///
/// Persistent keys remain encoded as [`SectorId`]; implementations lower a
/// sector once at the miss boundary and operate on components until emission.
#[doc(hidden)]
pub trait LoweredMultiplicityFreeAlgebra:
    MultiplicityFreeFusionRule + lowered_multiplicity_free_sealed::Sealed
{
    type Sector: Copy + Eq + Hash;

    fn try_decode_lowered(
        &self,
        sector: SectorId,
    ) -> Result<Self::Sector, LoweredFusionTreeBuildError>;

    fn try_encode_lowered(
        &self,
        sector: Self::Sector,
    ) -> Result<SectorId, LoweredFusionTreeBuildError>;

    fn try_lowered_vacuum(&self) -> Result<Self::Sector, LoweredFusionTreeBuildError>;

    fn try_lowered_dual(
        &self,
        sector: Self::Sector,
    ) -> Result<Self::Sector, LoweredFusionTreeBuildError>;

    fn try_for_each_lowered_channel<F>(
        &self,
        left: Self::Sector,
        right: Self::Sector,
        emit: &mut F,
    ) -> Result<(), LoweredFusionTreeBuildError>
    where
        F: FnMut(Self::Sector) -> Result<(), LoweredFusionTreeBuildError>;

    fn try_lowered_nsymbol(
        &self,
        left: Self::Sector,
        right: Self::Sector,
        coupled: Self::Sector,
    ) -> Result<usize, LoweredFusionTreeBuildError>;
}

pub trait MultiplicityFreePivotalSymbols: MultiplicityFreeFusionSymbols {
    fn bendright_scalar(
        &self,
        left_coupled: SectorId,
        bent_sector: SectorId,
        coupled: SectorId,
        bent_leg_is_dual: bool,
    ) -> Self::Scalar;

    fn foldright_scalar(
        &self,
        source: &FusionTreePairKey,
        destination: &FusionTreePairKey,
    ) -> Self::Scalar;
}

impl lowered_multiplicity_free_sealed::Sealed for Z2FusionRule {}

impl LoweredMultiplicityFreeAlgebra for Z2FusionRule {
    type Sector = Z2Irrep;

    fn try_decode_lowered(
        &self,
        sector: SectorId,
    ) -> Result<Self::Sector, LoweredFusionTreeBuildError> {
        Z2Irrep::from_sector_id(sector)
            .ok_or_else(|| LoweredFusionTreeBuildError::invalid_sector(sector))
    }

    fn try_encode_lowered(
        &self,
        sector: Self::Sector,
    ) -> Result<SectorId, LoweredFusionTreeBuildError> {
        Ok(sector.into())
    }

    fn try_lowered_vacuum(&self) -> Result<Self::Sector, LoweredFusionTreeBuildError> {
        Ok(Z2Irrep::EVEN)
    }

    fn try_lowered_dual(
        &self,
        sector: Self::Sector,
    ) -> Result<Self::Sector, LoweredFusionTreeBuildError> {
        Ok(sector)
    }

    fn try_for_each_lowered_channel<F>(
        &self,
        left: Self::Sector,
        right: Self::Sector,
        emit: &mut F,
    ) -> Result<(), LoweredFusionTreeBuildError>
    where
        F: FnMut(Self::Sector) -> Result<(), LoweredFusionTreeBuildError>,
    {
        emit(Z2Irrep::new(left.parity() ^ right.parity()))
    }

    fn try_lowered_nsymbol(
        &self,
        left: Self::Sector,
        right: Self::Sector,
        coupled: Self::Sector,
    ) -> Result<usize, LoweredFusionTreeBuildError> {
        Ok(usize::from(
            coupled.parity() == (left.parity() ^ right.parity()),
        ))
    }
}

impl MultiplicityFreePivotalSymbols for Z2FusionRule {
    fn bendright_scalar(
        &self,
        _left_coupled: SectorId,
        _bent_sector: SectorId,
        _coupled: SectorId,
        _bent_leg_is_dual: bool,
    ) -> Self::Scalar {
        1.0
    }

    fn foldright_scalar(
        &self,
        _source: &FusionTreePairKey,
        _destination: &FusionTreePairKey,
    ) -> Self::Scalar {
        1.0
    }
}

impl lowered_multiplicity_free_sealed::Sealed for FermionParityFusionRule {}

impl LoweredMultiplicityFreeAlgebra for FermionParityFusionRule {
    type Sector = Z2Irrep;

    fn try_decode_lowered(
        &self,
        sector: SectorId,
    ) -> Result<Self::Sector, LoweredFusionTreeBuildError> {
        Z2FusionRule.try_decode_lowered(sector)
    }

    fn try_encode_lowered(
        &self,
        sector: Self::Sector,
    ) -> Result<SectorId, LoweredFusionTreeBuildError> {
        Z2FusionRule.try_encode_lowered(sector)
    }

    fn try_lowered_vacuum(&self) -> Result<Self::Sector, LoweredFusionTreeBuildError> {
        Z2FusionRule.try_lowered_vacuum()
    }

    fn try_lowered_dual(
        &self,
        sector: Self::Sector,
    ) -> Result<Self::Sector, LoweredFusionTreeBuildError> {
        Z2FusionRule.try_lowered_dual(sector)
    }

    fn try_for_each_lowered_channel<F>(
        &self,
        left: Self::Sector,
        right: Self::Sector,
        emit: &mut F,
    ) -> Result<(), LoweredFusionTreeBuildError>
    where
        F: FnMut(Self::Sector) -> Result<(), LoweredFusionTreeBuildError>,
    {
        Z2FusionRule.try_for_each_lowered_channel(left, right, emit)
    }

    fn try_lowered_nsymbol(
        &self,
        left: Self::Sector,
        right: Self::Sector,
        coupled: Self::Sector,
    ) -> Result<usize, LoweredFusionTreeBuildError> {
        Z2FusionRule.try_lowered_nsymbol(left, right, coupled)
    }
}

impl MultiplicityFreePivotalSymbols for FermionParityFusionRule {
    fn bendright_scalar(
        &self,
        _left_coupled: SectorId,
        _bent_sector: SectorId,
        _coupled: SectorId,
        _bent_leg_is_dual: bool,
    ) -> Self::Scalar {
        1.0
    }

    fn foldright_scalar(
        &self,
        _source: &FusionTreePairKey,
        _destination: &FusionTreePairKey,
    ) -> Self::Scalar {
        1.0
    }
}

impl lowered_multiplicity_free_sealed::Sealed for U1FusionRule {}

impl LoweredMultiplicityFreeAlgebra for U1FusionRule {
    type Sector = U1Irrep;

    fn try_decode_lowered(
        &self,
        sector: SectorId,
    ) -> Result<Self::Sector, LoweredFusionTreeBuildError> {
        U1Irrep::from_sector_id(sector)
            .ok_or_else(|| LoweredFusionTreeBuildError::invalid_sector(sector))
    }

    fn try_encode_lowered(
        &self,
        sector: Self::Sector,
    ) -> Result<SectorId, LoweredFusionTreeBuildError> {
        Ok(sector.into())
    }

    fn try_lowered_vacuum(&self) -> Result<Self::Sector, LoweredFusionTreeBuildError> {
        Ok(U1Irrep::new(0))
    }

    fn try_lowered_dual(
        &self,
        sector: Self::Sector,
    ) -> Result<Self::Sector, LoweredFusionTreeBuildError> {
        sector
            .checked_dual()
            .map_err(LoweredFusionTreeBuildError::fusion_algebra)
    }

    fn try_for_each_lowered_channel<F>(
        &self,
        left: Self::Sector,
        right: Self::Sector,
        emit: &mut F,
    ) -> Result<(), LoweredFusionTreeBuildError>
    where
        F: FnMut(Self::Sector) -> Result<(), LoweredFusionTreeBuildError>,
    {
        let sector = left
            .checked_fuse(right)
            .map_err(LoweredFusionTreeBuildError::fusion_algebra)?;
        emit(sector)
    }

    fn try_lowered_nsymbol(
        &self,
        left: Self::Sector,
        right: Self::Sector,
        coupled: Self::Sector,
    ) -> Result<usize, LoweredFusionTreeBuildError> {
        left.checked_fuse(right)
            .map(|expected| usize::from(coupled == expected))
            .map_err(LoweredFusionTreeBuildError::fusion_algebra)
    }
}

#[cfg(test)]
mod tests {
    use super::LoweredFusionTreeBuildError;
    use crate::{FusionAlgebraError, ProductSectorCodecError, SectorId};

    #[test]
    fn invalid_sector_and_codec_lowering_errors_preserve_their_exact_categories() {
        let invalid = LoweredFusionTreeBuildError::invalid_sector(SectorId::new(7));
        assert_eq!(
            invalid.static_message(),
            "built-in fusion-tree layout contains an invalid sector"
        );
        assert_eq!(invalid.to_string(), "invalid built-in sector SectorId(7)");
        assert!(std::error::Error::source(&invalid).is_none());
        assert_eq!(
            invalid.clone().into_checked_fusion_algebra(),
            FusionAlgebraError::InvalidSector {
                sector: SectorId::new(7)
            }
        );
        assert_eq!(
            invalid.into_fusion_algebra().unwrap_err().static_message(),
            "built-in fusion-tree layout contains an invalid sector"
        );

        let codec = LoweredFusionTreeBuildError::codec(ProductSectorCodecError::CodecRejected);
        assert_eq!(
            codec.static_message(),
            "built-in fusion-tree layout contains an invalid product sector"
        );
        assert_eq!(codec.to_string(), "product sector codec rejected the value");
        assert!(std::error::Error::source(&codec).is_none());
        assert_eq!(
            codec.clone().into_checked_fusion_algebra(),
            FusionAlgebraError::ProductCodec(ProductSectorCodecError::CodecRejected)
        );
        assert_eq!(
            codec.into_fusion_algebra().unwrap_err().static_message(),
            "built-in fusion-tree layout contains an invalid product sector"
        );
    }

    #[test]
    fn fusion_algebra_lowering_errors_keep_their_source_and_cause() {
        let cause = FusionAlgebraError::U1DualOverflow { charge: i32::MIN };
        let error = LoweredFusionTreeBuildError::fusion_algebra(cause.clone());

        assert_eq!(
            error.static_message(),
            "built-in fusion-tree layout exceeds the representable algebra"
        );
        assert_eq!(error.to_string(), cause.to_string());
        assert_eq!(
            std::error::Error::source(&error)
                .and_then(|source| source.downcast_ref::<FusionAlgebraError>()),
            Some(&cause)
        );
        assert_eq!(error.clone().into_checked_fusion_algebra(), cause);
        assert_eq!(error.into_fusion_algebra(), Ok(cause));
    }
}
