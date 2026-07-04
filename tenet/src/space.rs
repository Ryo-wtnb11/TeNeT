//! User-layer vector spaces: sector content plus degeneracies for one leg.

use tenet_core::{
    FusionRule, ProductSectorCodec, SU2Irrep, SectorId, SectorLeg, TensorKitProductCodec, U1Irrep,
    Z2Irrep,
};

use crate::error::Error;

/// The fusion rule a [`Space`] (and every [`crate::prelude::Tensor`] built
/// from it) is tagged with. The user layer erases the concrete rule types of
/// the expert layer behind this tag.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub(crate) enum RuleKind {
    U1,
    Z2,
    FZ2,
    SU2,
    U1FZ2,
}

/// Dispatches on a [`RuleKind`], binding `$rule` to a reference to the
/// concrete expert-layer fusion rule inside `$body`.
macro_rules! with_rule {
    ($kind:expr, $rule:ident, $body:expr) => {
        match $kind {
            $crate::space::RuleKind::U1 => {
                let $rule = &tenet_core::U1FusionRule;
                $body
            }
            $crate::space::RuleKind::Z2 => {
                let $rule = &tenet_core::Z2FusionRule;
                $body
            }
            $crate::space::RuleKind::FZ2 => {
                let $rule = &tenet_core::FermionParityFusionRule;
                $body
            }
            $crate::space::RuleKind::SU2 => {
                let $rule = &tenet_core::SU2FusionRule;
                $body
            }
            $crate::space::RuleKind::U1FZ2 => {
                let $rule = &tenet_core::ProductFusionRule::<
                    tenet_core::U1FusionRule,
                    tenet_core::FermionParityFusionRule,
                >::new(
                    tenet_core::U1FusionRule,
                    tenet_core::FermionParityFusionRule,
                );
                $body
            }
        }
    };
}
pub(crate) use with_rule;

/// A graded vector space for one tensor leg: a list of `(sector, degeneracy)`
/// pairs plus a dual flag, tagged with its fusion rule.
///
/// This is the user-layer analog of TensorKit's `U1Space(-1 => 2, 0 => 3,
/// 1 => 2)` style constructors; internally it lowers to a
/// [`tenet_core::SectorLeg`] plus per-sector degeneracies.
///
/// # Examples
///
/// ```
/// use tenet::prelude::Space;
///
/// let v = Space::u1([(-1, 2), (0, 3), (1, 2)]);
/// assert_eq!(v.dim(), 7);
/// let w = v.dual();
/// assert_eq!(w.dim(), 7);
/// assert_eq!(w.dual(), v);
/// ```
#[derive(Clone, Debug, PartialEq)]
pub struct Space {
    pub(crate) rule: RuleKind,
    /// `(sector id, degeneracy)` pairs, stored in the internal sector-id
    /// encoding of the tagged rule.
    pub(crate) sectors: Vec<(SectorId, usize)>,
    pub(crate) dual: bool,
}

impl Space {
    fn new(rule: RuleKind, sectors: Vec<(SectorId, usize)>) -> Self {
        let mut sectors = sectors;
        sectors.sort_by_key(|(sector, _)| *sector);
        Self {
            rule,
            sectors,
            dual: false,
        }
    }

    /// U(1)-graded space from `(charge, degeneracy)` pairs.
    ///
    /// TensorKit equivalent: `U1Space(charge => degeneracy, ...)`.
    pub fn u1<I>(charges: I) -> Self
    where
        I: IntoIterator<Item = (i32, usize)>,
    {
        Self::new(
            RuleKind::U1,
            charges
                .into_iter()
                .map(|(charge, deg)| (U1Irrep::new(charge).sector_id(), deg))
                .collect(),
        )
    }

    /// Z2-graded space from `(parity, degeneracy)` pairs (`0` even, `1` odd).
    ///
    /// TensorKit equivalent: `Z2Space(0 => deg_even, 1 => deg_odd)`.
    pub fn z2<I>(parities: I) -> Self
    where
        I: IntoIterator<Item = (u8, usize)>,
    {
        Self::new(
            RuleKind::Z2,
            parities
                .into_iter()
                .map(|(parity, deg)| (Z2Irrep::new(parity).sector_id(), deg))
                .collect(),
        )
    }

    /// Fermion-parity-graded space from `(parity, degeneracy)` pairs
    /// (`0` even, `1` odd), with fermionic braiding.
    ///
    /// TensorKit equivalent: `Vect[FermionParity](0 => deg_even, 1 => deg_odd)`.
    pub fn fz2<I>(parities: I) -> Self
    where
        I: IntoIterator<Item = (u8, usize)>,
    {
        Self::new(
            RuleKind::FZ2,
            parities
                .into_iter()
                .map(|(parity, deg)| (SectorId::new(usize::from(parity & 1)), deg))
                .collect(),
        )
    }

    /// SU(2)-graded space from `(twice_spin, degeneracy)` pairs, mirroring
    /// [`tenet_core::SU2Irrep::from_twice_spin`] (`twice_spin = 2j`, so `1`
    /// is spin-1/2 and `2` is spin-1).
    ///
    /// TensorKit equivalent: `SU2Space(j => degeneracy, ...)`.
    pub fn su2<I>(spins: I) -> Self
    where
        I: IntoIterator<Item = (usize, usize)>,
    {
        Self::new(
            RuleKind::SU2,
            spins
                .into_iter()
                .map(|(twice_spin, deg)| (SU2Irrep::from_twice_spin(twice_spin).sector_id(), deg))
                .collect(),
        )
    }

    /// U(1) x fermion-parity product space from `((charge, parity),
    /// degeneracy)` pairs, using the TensorKit product-sector encoding
    /// ([`tenet_core::TensorKitProductCodec`]).
    ///
    /// TensorKit equivalent: `Vect[U1Irrep ⊠ FermionParity]`.
    pub fn product<I>(sectors: I) -> Result<Self, Error>
    where
        I: IntoIterator<Item = ((i32, u8), usize)>,
    {
        let sectors = sectors
            .into_iter()
            .map(|((charge, parity), deg)| {
                TensorKitProductCodec::try_encode(
                    U1Irrep::new(charge).sector_id(),
                    SectorId::new(usize::from(parity & 1)),
                )
                .map(|sector| (sector, deg))
                .ok_or_else(|| {
                    Error::InvalidArgument(format!(
                        "product sector ({charge}, {parity}) does not fit the sector-id encoding"
                    ))
                })
            })
            .collect::<Result<Vec<_>, Error>>()?;
        Ok(Self::new(RuleKind::U1FZ2, sectors))
    }

    /// The dual space: every sector is replaced by its dual and the dual
    /// flag is flipped, mirroring TensorKit's `V'`.
    pub fn dual(&self) -> Self {
        let sectors = with_rule!(self.rule, rule, {
            fn dualize<R: FusionRule>(
                rule: &R,
                sectors: &[(SectorId, usize)],
            ) -> Vec<(SectorId, usize)> {
                sectors
                    .iter()
                    .map(|&(sector, deg)| (rule.dual(sector), deg))
                    .collect()
            }
            dualize(rule, &self.sectors)
        });
        let mut sectors = sectors;
        sectors.sort_by_key(|(sector, _)| *sector);
        Self {
            rule: self.rule,
            sectors,
            dual: !self.dual,
        }
    }

    /// Total dimension: the sum of `degeneracy * dim(sector)` over all
    /// sectors (`dim(sector)` is the quantum dimension, e.g. `2j + 1` for
    /// SU(2)).
    pub fn dim(&self) -> usize {
        use tenet_core::MultiplicityFreeRigidSymbols;
        with_rule!(self.rule, rule, {
            self.sectors
                .iter()
                .map(|&(sector, deg)| deg * (rule.dim_scalar(sector).round() as usize))
                .sum()
        })
    }

    /// Lowers this space to the expert-layer [`SectorLeg`].
    pub(crate) fn sector_leg(&self) -> SectorLeg {
        SectorLeg::new(self.sectors.iter().map(|&(sector, _)| sector), self.dual)
    }
}
