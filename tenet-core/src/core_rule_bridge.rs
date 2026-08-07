use core::fmt;
use std::hash::Hash;

use crate::{
    CU1FusionRule, CU1Irrep, CheckedFusionAlgebra, FermionParityFusionRule, FusionAlgebraError,
    MultiplicityFreeFusionRule, PackedProductCodec, PackedSectorLayout, ProductFusionRule,
    ProductSector, ProductSectorCodec, ProductSectorCodecError, SU2FusionRule, SU2Irrep,
    SectorCodec, SectorId, U1FusionRule, U1Irrep, Z2FusionRule, Z2Irrep, ZNFusionRule, ZNIrrep,
};

// Why not tenet-sectors: this trait and its errors define FusionTree lowering,
// which is core-owned. The pivotal half that also lived here was deleted with
// its only route (#976), so nothing left here speaks about tree keys.
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

impl lowered_multiplicity_free_sealed::Sealed for SU2FusionRule {}

impl LoweredMultiplicityFreeAlgebra for SU2FusionRule {
    type Sector = SU2Irrep;

    fn try_decode_lowered(
        &self,
        sector: SectorId,
    ) -> Result<Self::Sector, LoweredFusionTreeBuildError> {
        SU2Irrep::try_from_sector_id(sector)
            .ok_or_else(|| LoweredFusionTreeBuildError::invalid_sector(sector))
    }

    fn try_encode_lowered(
        &self,
        sector: Self::Sector,
    ) -> Result<SectorId, LoweredFusionTreeBuildError> {
        Ok(sector.into())
    }

    fn try_lowered_vacuum(&self) -> Result<Self::Sector, LoweredFusionTreeBuildError> {
        Ok(SU2Irrep::from_twice_spin(0))
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
        match self
            .try_for_each_representable_channel(left, right, |channel| match emit(channel) {
                Ok(()) => core::ops::ControlFlow::Continue(()),
                Err(error) => core::ops::ControlFlow::Break(error),
            })
            .map_err(LoweredFusionTreeBuildError::fusion_algebra)?
        {
            core::ops::ControlFlow::Continue(()) => Ok(()),
            core::ops::ControlFlow::Break(error) => Err(error),
        }
    }

    fn try_lowered_nsymbol(
        &self,
        left: Self::Sector,
        right: Self::Sector,
        coupled: Self::Sector,
    ) -> Result<usize, LoweredFusionTreeBuildError> {
        let mut multiplicity = 0;
        let _ = self
            .try_for_each_representable_channel(left, right, |channel| {
                if channel == coupled {
                    multiplicity = 1;
                }
                core::ops::ControlFlow::<()>::Continue(())
            })
            .map_err(LoweredFusionTreeBuildError::fusion_algebra)?;
        Ok(multiplicity)
    }
}

impl lowered_multiplicity_free_sealed::Sealed for CU1FusionRule {}

impl LoweredMultiplicityFreeAlgebra for CU1FusionRule {
    type Sector = CU1Irrep;

    fn try_decode_lowered(
        &self,
        sector: SectorId,
    ) -> Result<Self::Sector, LoweredFusionTreeBuildError> {
        self.decode_sector(sector)
            .map_err(LoweredFusionTreeBuildError::fusion_algebra)
    }

    fn try_encode_lowered(
        &self,
        sector: Self::Sector,
    ) -> Result<SectorId, LoweredFusionTreeBuildError> {
        self.encode_sector(&sector)
            .map_err(LoweredFusionTreeBuildError::fusion_algebra)
    }

    fn try_lowered_vacuum(&self) -> Result<Self::Sector, LoweredFusionTreeBuildError> {
        Ok(CU1Irrep::VACUUM)
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
        // The checked provider computes the complete channel list before this
        // loop, so an overflow cannot publish a partial lowered tree.
        for channel in self
            .try_fusion_channels(left.into(), right.into())
            .map_err(LoweredFusionTreeBuildError::fusion_algebra)?
        {
            emit(
                self.decode_sector(channel)
                    .map_err(LoweredFusionTreeBuildError::fusion_algebra)?,
            )?;
        }
        Ok(())
    }

    fn try_lowered_nsymbol(
        &self,
        left: Self::Sector,
        right: Self::Sector,
        coupled: Self::Sector,
    ) -> Result<usize, LoweredFusionTreeBuildError> {
        self.try_nsymbol(left.into(), right.into(), coupled.into())
            .map_err(LoweredFusionTreeBuildError::fusion_algebra)
    }
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

impl lowered_multiplicity_free_sealed::Sealed for ZNFusionRule {}

impl LoweredMultiplicityFreeAlgebra for ZNFusionRule {
    type Sector = ZNIrrep;

    fn try_decode_lowered(
        &self,
        sector: SectorId,
    ) -> Result<Self::Sector, LoweredFusionTreeBuildError> {
        self.decode_sector(sector)
            .map_err(LoweredFusionTreeBuildError::fusion_algebra)
    }
    fn try_encode_lowered(
        &self,
        sector: Self::Sector,
    ) -> Result<SectorId, LoweredFusionTreeBuildError> {
        self.encode_sector(&sector)
            .map_err(LoweredFusionTreeBuildError::fusion_algebra)
    }
    fn try_lowered_vacuum(&self) -> Result<Self::Sector, LoweredFusionTreeBuildError> {
        Ok(self.irrep(0))
    }
    fn try_lowered_dual(
        &self,
        sector: Self::Sector,
    ) -> Result<Self::Sector, LoweredFusionTreeBuildError> {
        Ok(self.irrep(-(sector.charge() as i64)))
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
        emit(self.irrep((left.charge() as u64 + right.charge() as u64) as i64))
    }
    fn try_lowered_nsymbol(
        &self,
        left: Self::Sector,
        right: Self::Sector,
        coupled: Self::Sector,
    ) -> Result<usize, LoweredFusionTreeBuildError> {
        Ok(usize::from(
            coupled == self.irrep((left.charge() as u64 + right.charge() as u64) as i64),
        ))
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
impl<LeftRule, RightRule, LeftLayout, RightLayout> lowered_multiplicity_free_sealed::Sealed
    for ProductFusionRule<LeftRule, RightRule, PackedProductCodec<LeftLayout, RightLayout>>
where
    LeftRule: LoweredMultiplicityFreeAlgebra,
    RightRule: LoweredMultiplicityFreeAlgebra,
    LeftLayout: PackedSectorLayout + 'static,
    RightLayout: PackedSectorLayout + 'static,
{
}

impl<LeftRule, RightRule, LeftLayout, RightLayout> LoweredMultiplicityFreeAlgebra
    for ProductFusionRule<LeftRule, RightRule, PackedProductCodec<LeftLayout, RightLayout>>
where
    LeftRule: LoweredMultiplicityFreeAlgebra,
    RightRule: LoweredMultiplicityFreeAlgebra,
    LeftLayout: PackedSectorLayout + 'static,
    RightLayout: PackedSectorLayout + 'static,
{
    type Sector = ProductSector<LeftRule::Sector, RightRule::Sector>;

    fn try_decode_lowered(
        &self,
        sector: SectorId,
    ) -> Result<Self::Sector, LoweredFusionTreeBuildError> {
        let (left, right) = PackedProductCodec::<LeftLayout, RightLayout>::decode_checked(sector)
            .map_err(LoweredFusionTreeBuildError::codec)?;
        Ok(ProductSector::new(
            self.left_rule().try_decode_lowered(left)?,
            self.right_rule().try_decode_lowered(right)?,
        ))
    }

    fn try_encode_lowered(
        &self,
        sector: Self::Sector,
    ) -> Result<SectorId, LoweredFusionTreeBuildError> {
        let left = self.left_rule().try_encode_lowered(*sector.left())?;
        let right = self.right_rule().try_encode_lowered(*sector.right())?;
        PackedProductCodec::<LeftLayout, RightLayout>::encode_checked(left, right)
            .map_err(LoweredFusionTreeBuildError::codec)
    }

    fn try_lowered_vacuum(&self) -> Result<Self::Sector, LoweredFusionTreeBuildError> {
        Ok(ProductSector::new(
            self.left_rule().try_lowered_vacuum()?,
            self.right_rule().try_lowered_vacuum()?,
        ))
    }

    fn try_lowered_dual(
        &self,
        sector: Self::Sector,
    ) -> Result<Self::Sector, LoweredFusionTreeBuildError> {
        Ok(ProductSector::new(
            self.left_rule().try_lowered_dual(*sector.left())?,
            self.right_rule().try_lowered_dual(*sector.right())?,
        ))
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
        self.right_rule().try_for_each_lowered_channel(
            *left.right(),
            *right.right(),
            &mut |right_channel| {
                self.left_rule().try_for_each_lowered_channel(
                    *left.left(),
                    *right.left(),
                    &mut |left_channel| emit(ProductSector::new(left_channel, right_channel)),
                )
            },
        )
    }

    fn try_lowered_nsymbol(
        &self,
        left: Self::Sector,
        right: Self::Sector,
        coupled: Self::Sector,
    ) -> Result<usize, LoweredFusionTreeBuildError> {
        let left_n =
            self.left_rule()
                .try_lowered_nsymbol(*left.left(), *right.left(), *coupled.left())?;
        let right_n = self.right_rule().try_lowered_nsymbol(
            *left.right(),
            *right.right(),
            *coupled.right(),
        )?;
        match left_n.checked_mul(right_n) {
            Some(multiplicity) => Ok(multiplicity),
            None => {
                // Why not encode every successful call: persistent IDs are
                // needed only to diagnose the exceptional overflow branch.
                let left = self.try_encode_lowered(left)?;
                let right = self.try_encode_lowered(right)?;
                let coupled = self.try_encode_lowered(coupled)?;
                Err(LoweredFusionTreeBuildError::fusion_algebra(
                    FusionAlgebraError::MultiplicityOverflow {
                        left,
                        right,
                        coupled,
                    },
                ))
            }
        }
    }
}
