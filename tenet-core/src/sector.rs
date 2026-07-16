/// Opaque identifier used by one fusion-rule implementation.
///
/// The numeric value is an internal representation, not a stable wire format
/// or a cross-version sector label. Persist and compare the corresponding
/// semantic irrep/product labels instead. Codec or layout changes may alter
/// numeric IDs, block order, and storage offsets without changing the
/// represented tensor.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct SectorId(usize);

impl SectorId {
    /// Constructs an expert-layer identifier for a specific fusion rule.
    ///
    /// Callers are responsible for using the rule's matching codec.
    pub const fn new(id: usize) -> Self {
        Self(id)
    }

    /// Returns the rule-local opaque numeric representation.
    ///
    /// Do not serialize this value as a semantic sector label.
    #[inline]
    pub const fn id(self) -> usize {
        self.0
    }
}

impl From<usize> for SectorId {
    fn from(value: usize) -> Self {
        Self::new(value)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum FusionStyleKind {
    Unique,
    Simple,
    Generic,
}

impl FusionStyleKind {
    #[inline]
    pub const fn is_multiplicity_free(self) -> bool {
        matches!(self, Self::Unique | Self::Simple)
    }

    #[inline]
    pub const fn has_multiple_outputs(self) -> bool {
        matches!(self, Self::Simple | Self::Generic)
    }

    #[inline]
    pub const fn has_multiplicity(self) -> bool {
        matches!(self, Self::Generic)
    }

    pub const fn combined_with(self, other: Self) -> Self {
        match (self, other) {
            (Self::Generic, _) | (_, Self::Generic) => Self::Generic,
            (Self::Simple, _) | (_, Self::Simple) => Self::Simple,
            (Self::Unique, Self::Unique) => Self::Unique,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum BraidingStyleKind {
    NoBraiding,
    Bosonic,
    Fermionic,
    Anyonic,
}

impl BraidingStyleKind {
    #[inline]
    pub const fn has_braiding(self) -> bool {
        !matches!(self, Self::NoBraiding)
    }

    #[inline]
    pub const fn is_symmetric(self) -> bool {
        matches!(self, Self::Bosonic | Self::Fermionic)
    }

    #[inline]
    pub const fn is_bosonic(self) -> bool {
        matches!(self, Self::Bosonic)
    }

    pub const fn combined_with(self, other: Self) -> Self {
        match (self, other) {
            (Self::NoBraiding, _) | (_, Self::NoBraiding) => Self::NoBraiding,
            (Self::Anyonic, _) | (_, Self::Anyonic) => Self::Anyonic,
            (Self::Fermionic, _) | (_, Self::Fermionic) => Self::Fermionic,
            (Self::Bosonic, Self::Bosonic) => Self::Bosonic,
        }
    }
}

#[derive(Debug, Eq, PartialEq, Hash)]
struct SectorLegData {
    sectors: SectorVec,
    /// Per-sector degeneracy, parallel to `sectors`. The leg is the single
    /// source of truth for the sector -> degeneracy map of one tensor axis
    /// (TensorKit `GradedSpace` parity: the space stores the complete map
    /// independent of which fusion trees are populated).
    degeneracies: DimVec,
}

#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub struct SectorLeg {
    data: Arc<SectorLegData>,
    is_dual: bool,
}

impl SectorLeg {
    /// Builds one external leg from `(sector, degeneracy)` pairs.
    ///
    /// Pairs are stored sorted by sector id; identical duplicate pairs are
    /// removed. Panics when the same sector appears with two different
    /// degeneracies.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenet_core::{SectorLeg, Z2Irrep};
    ///
    /// let leg = SectorLeg::new([(Z2Irrep::ODD, 3), (Z2Irrep::EVEN, 2), (Z2Irrep::ODD, 3)], false);
    /// assert_eq!(leg.sectors().len(), 2);
    /// assert_eq!(leg.degeneracies(), &[2, 3]);
    /// assert!(!leg.is_dual());
    ///
    /// let dual_leg = SectorLeg::new([(Z2Irrep::ODD, 1)], true);
    /// assert!(dual_leg.is_dual());
    /// ```
    pub fn new<Pairs, Sector>(pairs: Pairs, is_dual: bool) -> Self
    where
        Pairs: IntoIterator<Item = (Sector, usize)>,
        Sector: Into<SectorId>,
    {
        let mut pairs = pairs
            .into_iter()
            .map(|(sector, degeneracy)| (sector.into(), degeneracy))
            .collect::<SmallVec<[(SectorId, usize); 8]>>();
        pairs.sort_unstable();
        pairs.dedup();
        for window in pairs.windows(2) {
            assert_ne!(
                window[0].0, window[1].0,
                "sector {:?} listed with conflicting degeneracies {} and {}",
                window[0].0, window[0].1, window[1].1
            );
        }
        let mut sectors = SectorVec::new();
        let mut degeneracies = DimVec::new();
        sectors.reserve(pairs.len());
        degeneracies.reserve(pairs.len());
        for (sector, degeneracy) in pairs {
            sectors.push(sector);
            degeneracies.push(degeneracy);
        }
        Self {
            data: Arc::new(SectorLegData {
                sectors,
                degeneracies,
            }),
            is_dual,
        }
    }

    pub fn from_sector_id(sector: usize, degeneracy: usize) -> Self {
        Self::new([(SectorId::new(sector), degeneracy)], false)
    }

    #[inline]
    pub fn sectors(&self) -> &[SectorId] {
        &self.data.sectors
    }

    /// Per-sector degeneracies, parallel to [`Self::sectors`].
    #[inline]
    pub fn degeneracies(&self) -> &[usize] {
        &self.data.degeneracies
    }

    /// Degeneracy of `sector` on this leg, `None` when the sector is not
    /// part of the leg.
    pub fn degeneracy(&self, sector: SectorId) -> Option<usize> {
        self.data
            .sectors
            .binary_search(&sector)
            .ok()
            .map(|index| self.data.degeneracies[index])
    }

    /// `(sector, degeneracy)` pairs in sorted sector order.
    pub fn iter(&self) -> impl Iterator<Item = (SectorId, usize)> + '_ {
        self.data
            .sectors
            .iter()
            .copied()
            .zip(self.data.degeneracies.iter().copied())
    }

    /// The dual leg: every sector is replaced by its dual (degeneracies
    /// carried along) and the dual flag is flipped.
    pub fn dual<R>(&self, rule: &R) -> Self
    where
        R: FusionRule,
    {
        Self::new(
            self.iter()
                .map(|(sector, degeneracy)| (rule.dual(sector), degeneracy)),
            !self.is_dual,
        )
    }

    #[inline]
    pub const fn is_dual(&self) -> bool {
        self.is_dual
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
struct FusionTreeLeg {
    sector: SectorId,
    is_dual: bool,
}

impl FusionTreeLeg {
    const fn new(sector: SectorId, is_dual: bool) -> Self {
        Self { sector, is_dual }
    }

    const fn sector(self) -> SectorId {
        self.sector
    }

    const fn is_dual(self) -> bool {
        self.is_dual
    }
}
