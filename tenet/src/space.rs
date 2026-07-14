//! User-layer vector spaces: sector content plus degeneracies for one leg.

use std::sync::Arc;

use tenet_core::{
    FermionParityFusionRule, FusionRule, ProductFusionRule, ProductSectorCodec, RuleIdentity,
    SU2FusionRule, SU2Irrep, SectorId, SectorLeg, Su3FusionRule, TensorKitProductCodec,
    U1FusionRule, U1Irrep, Z2FusionRule, Z2Irrep,
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
    // NOTE: no SU(3) variant. Adding one would break every downstream
    // exhaustive `match` on this public enum (a real consumer does exactly
    // that). The SU(3) label read-back is therefore a *separate*, non-breaking
    // accessor — [`Space::su3_sectors`] / [`Space::su3_degeneracy`], returning
    // the concrete `(p, q)` Dynkin labels — instead of an enum variant. The
    // internal `RuleKind::Su3` (pub(crate)) does not leak, so the shared
    // dispatch enum can grow without a downstream break.
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
    /// Stage B3b: SU(3) table provider ([`tenet_core::Su3FusionRule`]). The
    /// first `FusionStyleKind::Generic` (outer-multiplicity) rule reachable from
    /// the user layer. Adding this variant is compile-time zero-cost for the
    /// mult-free path (the χ32 anchor is re-measured to prove it): the
    /// `with_rule!` / `with_rule_ctx!` dispatch macros bind a mult-free rule per
    /// arm, and Su3 is `Generic`, so it takes dedicated non-macro paths
    /// (`*_generic` siblings) for the supported ops (permute/braid/transpose)
    /// and a clear panic for the rest (svd/trace/… are Stage B3c+).
    Su3,
}

pub(crate) type U1Fz2Rule = ProductFusionRule<U1FusionRule, FermionParityFusionRule>;
pub(crate) type Fz2U1Su2Rule =
    ProductFusionRule<ProductFusionRule<FermionParityFusionRule, U1FusionRule>, SU2FusionRule>;

#[derive(Clone, Debug)]
pub(crate) enum UserRuleContext {
    U1(Arc<U1FusionRule>),
    Z2(Arc<Z2FusionRule>),
    FZ2(Arc<FermionParityFusionRule>),
    SU2(Arc<SU2FusionRule>),
    U1FZ2(Arc<U1Fz2Rule>),
    FZ2U1SU2(Arc<Fz2U1Su2Rule>),
    Su3(Arc<Su3FusionRule>),
}

impl UserRuleContext {
    pub(crate) fn kind(&self) -> RuleKind {
        match self {
            Self::U1(_) => RuleKind::U1,
            Self::Z2(_) => RuleKind::Z2,
            Self::FZ2(_) => RuleKind::FZ2,
            Self::SU2(_) => RuleKind::SU2,
            Self::U1FZ2(_) => RuleKind::U1FZ2,
            Self::FZ2U1SU2(_) => RuleKind::FZ2U1SU2,
            Self::Su3(_) => RuleKind::Su3,
        }
    }

    pub(crate) fn identity(&self) -> RuleIdentity {
        match self {
            Self::U1(rule) => rule.rule_identity(),
            Self::Z2(rule) => rule.rule_identity(),
            Self::FZ2(rule) => rule.rule_identity(),
            Self::SU2(rule) => rule.rule_identity(),
            Self::U1FZ2(rule) => rule.rule_identity(),
            Self::FZ2U1SU2(rule) => rule.rule_identity(),
            Self::Su3(rule) => rule.rule_identity(),
        }
    }
}

impl PartialEq for UserRuleContext {
    fn eq(&self, other: &Self) -> bool {
        self.identity() == other.identity()
    }
}

impl Eq for UserRuleContext {}

impl AsRef<UserRuleContext> for UserRuleContext {
    fn as_ref(&self) -> &UserRuleContext {
        self
    }
}

/// Dispatches on a [`RuleKind`], binding `$rule` to a reference to the
/// concrete expert-layer fusion rule inside `$body`.
macro_rules! with_rule {
    ($context:expr, $rule:ident, $body:expr) => {
        match $context {
            $crate::space::UserRuleContext::U1(provider) => {
                let $rule = provider.as_ref();
                $body
            }
            $crate::space::UserRuleContext::Z2(provider) => {
                let $rule = provider.as_ref();
                $body
            }
            $crate::space::UserRuleContext::FZ2(provider) => {
                let $rule = provider.as_ref();
                $body
            }
            $crate::space::UserRuleContext::SU2(provider) => {
                let $rule = provider.as_ref();
                $body
            }
            $crate::space::UserRuleContext::U1FZ2(provider) => {
                let $rule = provider.as_ref();
                $body
            }
            $crate::space::UserRuleContext::FZ2U1SU2(provider) => {
                let $rule = provider.as_ref();
                $body
            }
            // Su3 is `FusionStyleKind::Generic`, so it CANNOT bind through this
            // macro (the shared `$body` calls mult-free-only methods on `$rule`).
            // The supported SU(3) ops (permute/braid/transpose, construction)
            // take dedicated `*_generic` paths that branch on `RuleKind::Su3`
            // BEFORE reaching here; anything else is not yet implemented. This
            // arm is `!`-typed, so it needs no `$body` and keeps every mult-free
            // call site byte-for-byte unchanged (the χ32 zero-cost guarantee).
            $crate::space::UserRuleContext::Su3(_) => {
                unimplemented!(
                    "this operation is not yet supported for SU(3) tensors \
                     (Stage B3b implements permute/braid/transpose; svd/qr/trace/\
                     norm/adjoint/contract are Stage B3c+)"
                )
            }
        }
    };
}
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
#[derive(Clone, Debug)]
pub struct Space {
    pub(crate) context: Arc<UserRuleContext>,
    /// `(sector id, degeneracy)` pairs, stored in the internal sector-id
    /// encoding of the tagged rule.
    pub(crate) sectors: Vec<(SectorId, usize)>,
    pub(crate) dual: bool,
}

impl PartialEq for Space {
    fn eq(&self, other: &Self) -> bool {
        (Arc::ptr_eq(&self.context, &other.context) || self.context == other.context)
            && self.sectors == other.sectors
            && self.dual == other.dual
    }
}

impl Space {
    /// Normalizes to the TensorKit `GradedSpace.dims` map invariant:
    /// zero-degeneracy sectors are dropped (TensorKit does the same) and a
    /// duplicate sector label panics immediately with a clear message —
    /// mirroring TensorKit's `ArgumentError` at `GradedSpace` construction
    /// (gradedspace.jl:49-56) — instead of surfacing later as an
    /// inconsistent `dim()` or a tensor-construction panic.
    fn new(context: Arc<UserRuleContext>, sectors: Vec<(SectorId, usize)>) -> Self {
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
            context,
            sectors,
            dual: false,
        }
    }

    pub(crate) fn rule_kind(&self) -> RuleKind {
        self.context.kind()
    }

    pub(crate) fn rule_context(&self) -> &Arc<UserRuleContext> {
        &self.context
    }

    /// U(1)-graded space from `(charge, degeneracy)` pairs.
    ///
    /// TensorKit equivalent: `U1Space(charge => degeneracy, ...)`.
    ///
    /// # Panics
    ///
    /// Panics on a duplicate sector label — a programming bug in the
    /// constructor call, mirroring TensorKit's `ArgumentError` at
    /// `GradedSpace` construction (`gradedspace.jl:49-56`). The
    /// [`Self::product`] / [`Self::fz2_u1_su2`] constructors return
    /// `Result` instead only because their product-sector *encoding* can
    /// fail on data-dependent capacity, not for duplicates.
    pub fn u1<I>(charges: I) -> Self
    where
        I: IntoIterator<Item = (i32, usize)>,
    {
        Self::new(
            Arc::new(UserRuleContext::U1(Arc::new(U1FusionRule))),
            charges
                .into_iter()
                .map(|(charge, deg)| (U1Irrep::new(charge).sector_id(), deg))
                .collect(),
        )
    }

    /// Z2-graded space from `(parity, degeneracy)` pairs (`0` even, `1` odd).
    ///
    /// TensorKit equivalent: `Z2Space(0 => deg_even, 1 => deg_odd)`.
    ///
    /// # Panics
    ///
    /// Panics on a duplicate sector label (TensorKit `ArgumentError`
    /// parity); see [`Self::u1`].
    pub fn z2<I>(parities: I) -> Self
    where
        I: IntoIterator<Item = (u8, usize)>,
    {
        Self::new(
            Arc::new(UserRuleContext::Z2(Arc::new(Z2FusionRule))),
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
    ///
    /// # Panics
    ///
    /// Panics on a duplicate sector label (TensorKit `ArgumentError`
    /// parity); see [`Self::u1`].
    pub fn fz2<I>(parities: I) -> Self
    where
        I: IntoIterator<Item = (u8, usize)>,
    {
        Self::new(
            Arc::new(UserRuleContext::FZ2(Arc::new(FermionParityFusionRule))),
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
    ///
    /// # Panics
    ///
    /// Panics on a duplicate sector label (TensorKit `ArgumentError`
    /// parity); see [`Self::u1`].
    pub fn su2<I>(spins: I) -> Self
    where
        I: IntoIterator<Item = (usize, usize)>,
    {
        Self::new(
            Arc::new(UserRuleContext::SU2(Arc::new(SU2FusionRule))),
            spins
                .into_iter()
                .map(|(twice_spin, deg)| (SU2Irrep::from_twice_spin(twice_spin).sector_id(), deg))
                .collect(),
        )
    }

    /// SU(3)-graded space from `((p, q), degeneracy)` pairs, where `(p, q)` is
    /// the Dynkin label of an irrep in the Stage B3b `dim ≤ 27` table (e.g.
    /// `(1, 0)` = **3**, `(0, 1)` = **3̄**, `(1, 1)` = **8**, `(2, 2)` = **27**).
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidArgument`] for a `(p, q)` outside the table
    /// (`dim > 27`) — the SU(3) provider is deliberately bounded (Stage B3c
    /// lifts the cut). Panics on a duplicate label, as [`Self::u1`].
    pub fn su3<I>(irreps: I) -> Result<Self, Error>
    where
        I: IntoIterator<Item = ((u8, u8), usize)>,
    {
        let rule = Arc::new(Su3FusionRule::new());
        let sectors = irreps
            .into_iter()
            .map(|((p, q), deg)| {
                rule.sector_of(p, q)
                    .map(|sector| (sector, deg))
                    .ok_or_else(|| {
                        Error::InvalidArgument(format!(
                            "SU(3) irrep ({p},{q}) is outside the dim<=27 table"
                        ))
                    })
            })
            .collect::<Result<Vec<_>, Error>>()?;
        Ok(Self::new(Arc::new(UserRuleContext::Su3(rule)), sectors))
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
        Ok(Self::new(
            Arc::new(UserRuleContext::U1FZ2(Arc::new(U1Fz2Rule::new(
                U1FusionRule,
                FermionParityFusionRule,
            )))),
            sectors,
        ))
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
        Ok(Self::new(
            Arc::new(UserRuleContext::FZ2U1SU2(Arc::new(Fz2U1Su2Rule::new(
                ProductFusionRule::new(FermionParityFusionRule, U1FusionRule),
                SU2FusionRule,
            )))),
            sectors,
        ))
    }

    /// The dual space: every sector is replaced by its dual and the dual
    /// flag is flipped, mirroring TensorKit's `V'`.
    pub fn dual(&self) -> Self {
        // Su3 is Generic, so it cannot ride the mult-free `with_rule!` binding;
        // `dual` needs only `FusionRule::dual`, handled directly.
        if let UserRuleContext::Su3(rule) = self.context.as_ref() {
            let mut sectors: Vec<(SectorId, usize)> = self
                .sectors
                .iter()
                .map(|&(sector, deg)| (rule.dual(sector), deg))
                .collect();
            sectors.sort_by_key(|(sector, _)| *sector);
            return Self {
                context: Arc::clone(&self.context),
                sectors,
                dual: !self.dual,
            };
        }
        let sectors = with_rule!(self.context.as_ref(), rule, {
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
            context: Arc::clone(&self.context),
            sectors,
            dual: !self.dual,
        }
    }

    /// Total dimension: the sum of `degeneracy * dim(sector)` over all
    /// sectors (`dim(sector)` is the quantum dimension, e.g. `2j + 1` for
    /// SU(2)).
    pub fn dim(&self) -> usize {
        use tenet_core::MultiplicityFreeRigidSymbols;
        if let UserRuleContext::Su3(rule) = self.context.as_ref() {
            use tenet_core::GenericRigidSymbols;
            return self
                .sectors
                .iter()
                .map(|&(sector, deg)| {
                    // quantum dim = (sqrt_dim)^2, integer for SU(3).
                    let sqrt = rule.sqrt_dim_scalar(sector);
                    deg * (sqrt * sqrt).round() as usize
                })
                .sum();
        }
        with_rule!(self.context.as_ref(), rule, {
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
        Arc::ptr_eq(&self.context, &other.context) || self.context == other.context
    }

    /// Decodes an internal sector id into the user-facing label for this
    /// space's rule. Inverse of the encoding done by the constructors.
    fn decode_sector(&self, sector: SectorId) -> SectorLabel {
        match self.rule_kind() {
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
            RuleKind::Su3 => unimplemented!(
                "SU(3) sectors do not fit the `SectorLabel` enum without a breaking \
                 `Su3` variant; use the dedicated non-breaking accessors \
                 `Space::su3_sectors` / `Space::su3_degeneracy` instead."
            ),
        }
    }

    /// Encodes a user-facing label into the internal sector id, `None` when
    /// the label's variant does not match this space's rule or the label
    /// does not fit the sector-id encoding.
    fn encode_sector(&self, label: SectorLabel) -> Option<SectorId> {
        match (self.rule_kind(), label) {
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
    ///
    /// # Panics
    ///
    /// Panics on an SU(3) space: its `(p, q)` irreps do not fit the
    /// [`SectorLabel`] enum without a breaking `Su3` variant. Use the fallible
    /// [`Self::try_sectors`] (typed [`Error::UnsupportedForRule`]) to probe, or
    /// [`Self::su3_sectors`] to read the concrete `(p, q)` labels.
    pub fn sectors(&self) -> Vec<(SectorLabel, usize)> {
        self.sectors
            .iter()
            .map(|&(sector, deg)| (self.decode_sector(sector), deg))
            .collect()
    }

    /// Fallible sibling of [`Self::sectors`]: `Ok` with byte-identical content
    /// on every multiplicity-free rule, [`Error::UnsupportedForRule`] on SU(3)
    /// (whose `(p, q)` irreps do not fit [`SectorLabel`]). Read SU(3) sectors
    /// with [`Self::su3_sectors`] instead.
    ///
    /// A separate method rather than changing [`Self::sectors`] to return
    /// `Result`: that signature change breaks every multiplicity-free caller,
    /// a breaking change disproportionate to closing one SU(3) panic surface.
    pub fn try_sectors(&self) -> Result<Vec<(SectorLabel, usize)>, Error> {
        if self.rule_kind() == RuleKind::Su3 {
            return Err(Error::UnsupportedForRule {
                operation: "Space::try_sectors",
                rule: "SU(3)",
            });
        }
        Ok(self.sectors())
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

    /// SU(3) sector read-back: the `((p, q), degeneracy)` content of an SU(3)
    /// space in the same `(p, q)` Dynkin-label form [`Self::su3`] accepts,
    /// sorted by internal sector id (external sectors, as [`Self::sectors`]).
    ///
    /// This is the SU(3) analog of [`Self::sectors`]. It is a *separate*
    /// accessor rather than an `SectorLabel::Su3` variant on purpose: adding a
    /// variant to the public [`SectorLabel`] enum breaks every downstream
    /// exhaustive `match` (a real consumer does exactly that), so the
    /// non-breaking read-back is a dedicated method returning the concrete
    /// `(p, q)` labels. [`Error::RuleMismatch`] on a non-SU(3) space.
    pub fn su3_sectors(&self) -> Result<Vec<((u8, u8), usize)>, Error> {
        if self.rule_kind() != RuleKind::Su3 {
            return Err(Error::RuleMismatch);
        }
        let UserRuleContext::Su3(rule) = self.context.as_ref() else {
            unreachable!("rule kind and provider context are coherent")
        };
        Ok(self
            .sectors
            .iter()
            .map(|&(sector, deg)| (rule.dynkin(sector), deg))
            .collect())
    }

    /// SU(3) sibling of [`Self::degeneracy`]: degeneracy of the `(p, q)` irrep
    /// (external label), `None` when the sector is absent from this space.
    /// [`Error::RuleMismatch`] on a non-SU(3) space, [`Error::InvalidArgument`]
    /// for a `(p, q)` outside the `dim <= 27` table.
    pub fn su3_degeneracy(&self, p: u8, q: u8) -> Result<Option<usize>, Error> {
        if self.rule_kind() != RuleKind::Su3 {
            return Err(Error::RuleMismatch);
        }
        let UserRuleContext::Su3(rule) = self.context.as_ref() else {
            unreachable!("rule kind and provider context are coherent")
        };
        let sector = rule.sector_of(p, q).ok_or_else(|| {
            Error::InvalidArgument(format!(
                "SU(3) irrep ({p},{q}) is outside the dim<=27 table"
            ))
        })?;
        Ok(self
            .sectors
            .iter()
            .find(|&&(s, _)| s == sector)
            .map(|&(_, deg)| deg))
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
        if !self.same_rule(other) {
            return Err(Error::RuleMismatch);
        }
        // SU(3) cannot use the multiplicity-free dispatch below; keep the
        // public Result boundary recoverable until a generic fuse is wired.
        if self.rule_kind() == RuleKind::Su3 {
            return Err(Error::UnsupportedForRule {
                operation: "Space::fuse",
                rule: "SU(3)",
            });
        }
        let fused = with_rule!(self.context.as_ref(), rule, {
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
            context: Arc::clone(&self.context),
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
        if !rest.is_empty()
            && first.rule_kind() == RuleKind::Su3
            && rest.iter().all(|space| first.same_rule(space))
        {
            return Err(Error::UnsupportedForRule {
                operation: "Space::fuse_all",
                rule: "SU(3)",
            });
        }
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
    pub(crate) fn from_leg(context: Arc<UserRuleContext>, leg: &SectorLeg) -> Self {
        Self {
            context,
            sectors: leg.iter().collect(),
            dual: leg.is_dual(),
        }
    }
}

#[cfg(test)]
mod provider_context_tests {
    use super::*;

    #[test]
    fn separately_constructed_builtin_spaces_compare_by_semantic_identity() {
        // What: semantic equality is independent of the provider Arc allocation.
        let first = Space::u1([(0, 2)]);
        let second = Space::u1([(0, 2)]);

        assert!(!Arc::ptr_eq(&first.context, &second.context));
        assert_eq!(first.context.identity(), second.context.identity());
        assert_eq!(first, second);
    }

    #[test]
    fn different_provider_identity_rejects_space_equality_and_fusion() {
        // What: overlapping sector ids cannot erase distinct fusion-rule identities.
        let u1 = Space::u1([(0, 1)]);
        let z2 = Space::z2([(0, 1)]);

        assert_ne!(u1.context.identity(), z2.context.identity());
        assert_ne!(u1, z2);
        assert!(matches!(u1.fuse(&z2), Err(Error::RuleMismatch)));
    }

    #[test]
    fn derived_spaces_inherit_the_actual_context_arc() {
        // What: every structural derivation retains the source provider allocation.
        let source = Space::u1([(0, 2)]);
        let dual = source.dual();
        let rebuilt = Space::from_leg(Arc::clone(&source.context), &source.sector_leg());
        let fused = source.fuse(&rebuilt).unwrap();

        assert!(Arc::ptr_eq(&source.context, &dual.context));
        assert!(Arc::ptr_eq(&source.context, &rebuilt.context));
        assert!(Arc::ptr_eq(&source.context, &fused.context));
    }
}
