//! User-layer vector spaces: sector content plus degeneracies for one leg.

use tenet_core::{
    FusionRule, ProductSectorCodec, SU2Irrep, SectorId, SectorLeg, TensorKitProductCodec, U1Irrep,
    Z2Irrep,
};

use crate::error::Error;

/// A user-facing sector label: the rule-specific charge content of one
/// sector, mirroring the [`Space`] constructors (one variant per
/// constructor). Returned by [`Space::sectors`] and accepted by
/// [`Space::degeneracy`].
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum SectorLabel {
    /// U(1) charge, as passed to [`Space::u1`].
    U1(i32),
    /// Z2 parity (`0` even, `1` odd), as passed to [`Space::z2`].
    Z2(u8),
    /// Fermion parity (`0` even, `1` odd), as passed to [`Space::fz2`].
    FZ2(u8),
    /// SU(2) spin as `twice_spin = 2j`, as passed to [`Space::su2`].
    SU2 { twice_spin: usize },
    /// U(1) x fermion-parity product sector, as passed to [`Space::product`].
    U1FZ2 { charge: i32, parity: u8 },
    /// fZ2 x U(1) x SU(2) triple-product sector, as passed to
    /// [`Space::fz2_u1_su2`].
    FZ2U1SU2 {
        parity: u8,
        charge: i32,
        twice_spin: usize,
    },
}

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
    FZ2U1SU2,
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
            $crate::space::RuleKind::FZ2U1SU2 => {
                // Left-associated (fZ2 ⊠ U1) ⊠ SU2, matching TensorKit's
                // left-associated triple product.
                let $rule = &tenet_core::ProductFusionRule::<
                    tenet_core::ProductFusionRule<
                        tenet_core::FermionParityFusionRule,
                        tenet_core::U1FusionRule,
                    >,
                    tenet_core::SU2FusionRule,
                >::new(
                    tenet_core::ProductFusionRule::new(
                        tenet_core::FermionParityFusionRule,
                        tenet_core::U1FusionRule,
                    ),
                    tenet_core::SU2FusionRule,
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
    /// Normalizes to the TensorKit `GradedSpace.dims` map invariant:
    /// zero-degeneracy sectors are dropped (TensorKit does the same) and a
    /// duplicate sector label panics immediately with a clear message —
    /// mirroring TensorKit's `ArgumentError` at `GradedSpace` construction
    /// (gradedspace.jl:49-56) — instead of surfacing later as an
    /// inconsistent `dim()` or a tensor-construction panic.
    fn new(rule: RuleKind, sectors: Vec<(SectorId, usize)>) -> Self {
        let mut sectors = sectors;
        sectors.retain(|&(_, degeneracy)| degeneracy > 0);
        sectors.sort_by_key(|(sector, _)| *sector);
        for pair in sectors.windows(2) {
            assert!(
                pair[0].0 != pair[1].0,
                "sector {:?} appears multiple times in the space constructor                  (TensorKit rejects duplicate sectors at construction)",
                pair[0].0
            );
        }
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

    /// fZ2 x U(1) x SU(2) triple-product space from `((parity, charge,
    /// twice_spin), degeneracy)` pairs (`parity`: `0` even / `1` odd,
    /// `twice_spin = 2j`), left-associated as `(fZ2 ⊠ U1) ⊠ SU2` with the
    /// TensorKit product-sector encoding applied pairwise.
    ///
    /// TensorKit equivalent: `Vect[FermionParity ⊠ Irrep[U₁] ⊠ Irrep[SU₂]]`.
    pub fn fz2_u1_su2<I>(sectors: I) -> Result<Self, Error>
    where
        I: IntoIterator<Item = ((u8, i32, usize), usize)>,
    {
        let sectors = sectors
            .into_iter()
            .map(|((parity, charge, twice_spin), deg)| {
                TensorKitProductCodec::try_encode(
                    SectorId::new(usize::from(parity & 1)),
                    U1Irrep::new(charge).sector_id(),
                )
                .and_then(|inner| {
                    TensorKitProductCodec::try_encode(
                        inner,
                        SU2Irrep::from_twice_spin(twice_spin).sector_id(),
                    )
                })
                .map(|sector| (sector, deg))
                .ok_or_else(|| {
                    Error::InvalidArgument(format!(
                        "product sector ({parity}, {charge}, {twice_spin}) does not fit the \
                         sector-id encoding"
                    ))
                })
            })
            .collect::<Result<Vec<_>, Error>>()?;
        Ok(Self::new(RuleKind::FZ2U1SU2, sectors))
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

    /// Whether this is a dual space, mirroring TensorKit's `isdual(V)`.
    pub fn is_dual(&self) -> bool {
        self.dual
    }

    /// `true` when `other` carries the same fusion rule, i.e. the two spaces
    /// can appear on the same tensor / be fused.
    pub fn same_rule(&self, other: &Space) -> bool {
        self.rule == other.rule
    }

    /// Decodes an internal sector id into the user-facing label for this
    /// space's rule. Inverse of the encoding done by the constructors.
    fn decode_sector(&self, sector: SectorId) -> SectorLabel {
        match self.rule {
            RuleKind::U1 => SectorLabel::U1(
                U1Irrep::from_sector_id(sector)
                    .expect("invalid U1 sector id")
                    .charge(),
            ),
            RuleKind::Z2 => SectorLabel::Z2(
                Z2Irrep::from_sector_id(sector)
                    .expect("invalid Z2 sector id")
                    .parity(),
            ),
            RuleKind::FZ2 => SectorLabel::FZ2(sector.id() as u8),
            RuleKind::SU2 => SectorLabel::SU2 {
                twice_spin: SU2Irrep::from_sector_id(sector).twice_spin(),
            },
            RuleKind::U1FZ2 => {
                let (u1, fz2) =
                    TensorKitProductCodec::decode(sector).expect("invalid product sector id");
                SectorLabel::U1FZ2 {
                    charge: U1Irrep::from_sector_id(u1)
                        .expect("invalid U1 sector id")
                        .charge(),
                    parity: fz2.id() as u8,
                }
            }
            RuleKind::FZ2U1SU2 => {
                let (inner, su2) =
                    TensorKitProductCodec::decode(sector).expect("invalid product sector id");
                let (fz2, u1) =
                    TensorKitProductCodec::decode(inner).expect("invalid product sector id");
                SectorLabel::FZ2U1SU2 {
                    parity: fz2.id() as u8,
                    charge: U1Irrep::from_sector_id(u1)
                        .expect("invalid U1 sector id")
                        .charge(),
                    twice_spin: SU2Irrep::from_sector_id(su2).twice_spin(),
                }
            }
        }
    }

    /// Encodes a user-facing label into the internal sector id, `None` when
    /// the label's variant does not match this space's rule or the label
    /// does not fit the sector-id encoding.
    fn encode_sector(&self, label: SectorLabel) -> Option<SectorId> {
        match (self.rule, label) {
            (RuleKind::U1, SectorLabel::U1(charge)) => Some(U1Irrep::new(charge).sector_id()),
            (RuleKind::Z2, SectorLabel::Z2(parity)) => Some(Z2Irrep::new(parity).sector_id()),
            (RuleKind::FZ2, SectorLabel::FZ2(parity)) => {
                Some(SectorId::new(usize::from(parity & 1)))
            }
            (RuleKind::SU2, SectorLabel::SU2 { twice_spin }) => {
                Some(SU2Irrep::from_twice_spin(twice_spin).sector_id())
            }
            (RuleKind::U1FZ2, SectorLabel::U1FZ2 { charge, parity }) => {
                TensorKitProductCodec::try_encode(
                    U1Irrep::new(charge).sector_id(),
                    SectorId::new(usize::from(parity & 1)),
                )
            }
            (
                RuleKind::FZ2U1SU2,
                SectorLabel::FZ2U1SU2 {
                    parity,
                    charge,
                    twice_spin,
                },
            ) => TensorKitProductCodec::try_encode(
                SectorId::new(usize::from(parity & 1)),
                U1Irrep::new(charge).sector_id(),
            )
            .and_then(|inner| {
                TensorKitProductCodec::try_encode(
                    inner,
                    SU2Irrep::from_twice_spin(twice_spin).sector_id(),
                )
            }),
            _ => None,
        }
    }

    /// The `(sector, degeneracy)` content of this space in the user-facing
    /// representation, sorted by internal sector id.
    ///
    /// Labels are the *external* (outward-facing) sectors, matching
    /// TensorKit's `sectors(V)`: for a dual space they are the duals of the
    /// sectors the space was constructed with, exactly as [`Self::dual`]
    /// stores them.
    pub fn sectors(&self) -> Vec<(SectorLabel, usize)> {
        self.sectors
            .iter()
            .map(|&(sector, deg)| (self.decode_sector(sector), deg))
            .collect()
    }

    /// Degeneracy of the sector with the given (external) label, `None` when
    /// the sector is absent or the label does not match this space's rule.
    ///
    /// TensorKit equivalent: `dim(V, c)` (named `degeneracy` here because
    /// [`Self::dim`] is the quantum-dimension-weighted total).
    pub fn degeneracy(&self, label: SectorLabel) -> Option<usize> {
        let sector = self.encode_sector(label)?;
        self.sectors
            .iter()
            .find(|&&(s, _)| s == sector)
            .map(|&(_, deg)| deg)
    }

    /// The fused space `V1 ⊗ V2` collapsed to a single leg: every fusion
    /// product of a sector of `self` with a sector of `other`, with the
    /// degeneracy of an outcome `c` given by
    /// `sum over (a, b) with c in a ⊗ b of deg_a * deg_b * N^c_ab`.
    ///
    /// Follows TensorKit's `fuse(V₁, V₂)`
    /// (`spaces/gradedspace.jl:150-158`): inputs enter through their
    /// *external* sector content (dual spaces are not re-dualized — their
    /// stored sectors are already external, see [`Self::dual`]) and the
    /// result is always a non-dual space.
    ///
    /// Errors with [`Error::RuleMismatch`] when the rules differ.
    pub fn fuse(&self, other: &Space) -> Result<Space, Error> {
        if self.rule != other.rule {
            return Err(Error::RuleMismatch);
        }
        let fused = with_rule!(self.rule, rule, {
            fn fuse_sectors<R: FusionRule>(
                rule: &R,
                left: &[(SectorId, usize)],
                right: &[(SectorId, usize)],
            ) -> Vec<(SectorId, usize)> {
                let mut out = std::collections::BTreeMap::<SectorId, usize>::new();
                for &(a, deg_a) in left {
                    for &(b, deg_b) in right {
                        for c in rule.fusion_channels(a, b) {
                            *out.entry(c).or_insert(0) += rule.nsymbol(a, b, c) * deg_a * deg_b;
                        }
                    }
                }
                out.into_iter().collect()
            }
            fuse_sectors(rule, &self.sectors, &other.sectors)
        });
        Ok(Self {
            rule: self.rule,
            sectors: fused,
            dual: false,
        })
    }

    /// N-ary [`Self::fuse`], mirroring TensorKit's variadic
    /// `fuse(V₁, V₂, V₃...)` (`spaces/vectorspaces.jl:269-274`): a single
    /// space folds to its non-dual isomorph (TensorKit's `fuse(V) =
    /// isdual(V) ? flip(V) : V`), more fold pairwise.
    ///
    /// Errors on an empty slice or mismatched rules.
    pub fn fuse_all(spaces: &[&Space]) -> Result<Space, Error> {
        let (first, rest) = spaces
            .split_first()
            .ok_or_else(|| Error::InvalidArgument("fuse_all needs at least one space".into()))?;
        // flip: stored sectors are already external, so dropping the dual
        // flag yields the isomorphic non-dual space.
        let mut fused = Space {
            dual: false,
            ..(*first).clone()
        };
        for space in rest {
            fused = fused.fuse(space)?;
        }
        Ok(fused)
    }

    /// Lowers this space to the expert-layer [`SectorLeg`] (sector,
    /// degeneracy and dual content carried verbatim).
    pub(crate) fn sector_leg(&self) -> SectorLeg {
        SectorLeg::new(self.sectors.iter().copied(), self.dual)
    }

    /// Reconstructs the user-facing space from an expert-layer leg, keyed by
    /// the fusion rule it lives under. Inverse of [`Self::sector_leg`].
    pub(crate) fn from_leg(rule: RuleKind, leg: &SectorLeg) -> Self {
        Self {
            rule,
            sectors: leg.iter().collect(),
            dual: leg.is_dual(),
        }
    }
}
