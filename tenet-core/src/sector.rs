use core::num::NonZeroUsize;

/// One-based outer-multiplicity label at a fusion vertex.
///
/// Unlike [`SectorId`], this value selects a basis vector within a fixed
/// fusion channel. Zero is not a categorical basis label.
#[repr(transparent)]
#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct MultiplicityIndex(NonZeroUsize);

impl MultiplicityIndex {
    /// The unique vertex label of a multiplicity-free fusion channel.
    pub const ONE: Self = Self(NonZeroUsize::MIN);

    /// Constructs a one-based fusion-vertex basis label.
    ///
    /// Returns `None` for zero. Numeric import boundaries that need a typed
    /// error can use [`TryFrom<usize>`], which reports
    /// [`CoreError::InvalidMultiplicityIndex`].
    #[inline]
    pub const fn new(value: usize) -> Option<Self> {
        match NonZeroUsize::new(value) {
            Some(value) => Some(Self(value)),
            None => None,
        }
    }

    /// Returns the one-based fusion-vertex basis label.
    #[inline]
    pub const fn get(self) -> usize {
        self.0.get()
    }
}

impl TryFrom<usize> for MultiplicityIndex {
    type Error = CoreError;

    fn try_from(value: usize) -> Result<Self, Self::Error> {
        Self::new(value).ok_or(CoreError::InvalidMultiplicityIndex { value })
    }
}

impl From<MultiplicityIndex> for usize {
    fn from(value: MultiplicityIndex) -> Self {
        value.get()
    }
}

struct SectorLegData {
    sectors: SectorVec,
    /// Per-sector degeneracy, parallel to `sectors`. The leg is the single
    /// source of truth for the sector -> degeneracy map of one tensor axis
    /// (TensorKit `GradedSpace` parity: the space stores the complete map
    /// independent of which fusion trees are populated).
    degeneracies: DimVec,
    /// The dual of this map, filled by the first dual query and shared by
    /// every leg dualized from this storage afterwards (TensorKit
    /// `dual(V)` shares `V.dims`).
    ///
    /// Why lazy and not at construction: `SectorLeg::new` has no rule, and a
    /// `SectorId`'s dual is rule-relative. Why validated on every query and
    /// not trusted: two rules of one Rust type (Generic providers) can dualize
    /// one id differently, so a query whose rule disagrees with `images`
    /// builds its own storage and leaves this map untouched.
    dual: OnceLock<DualSectorMap>,
}

struct DualSectorMap {
    /// `dual(sectors[i])` under the rule that filled this map.
    images: SectorVec,
    /// The sorted dual map, `None` when it equals the source map.
    moved: Option<MovedSectorMap>,
}

struct MovedSectorMap {
    sectors: SectorVec,
    degeneracies: DimVec,
    /// `dual(sectors[j])` under the filling rule: the source sector that
    /// `sectors[j]` is the dual of.
    images: SectorVec,
}

fn exact_sector_vec(values: impl ExactSizeIterator<Item = SectorId>) -> SectorVec {
    let mut out = SectorVec::with_capacity(values.len());
    out.extend(values);
    out
}

#[non_exhaustive]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SectorLegConstructionError {
    DuplicateSector { sector: SectorId },
}

impl fmt::Display for SectorLegConstructionError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::DuplicateSector { sector } => {
                write!(formatter, "sector {sector:?} appears multiple times")
            }
        }
    }
}

impl std::error::Error for SectorLegConstructionError {}

#[derive(Clone)]
pub struct SectorLeg {
    data: Arc<SectorLegData>,
    is_dual: bool,
    /// Whether this leg's map is `data.dual`'s moved map rather than
    /// `data`'s own. Invariant: set only after `data.dual` holds a moved map;
    /// a `OnceLock` is never cleared.
    moved: bool,
}

impl PartialEq for SectorLeg {
    fn eq(&self, other: &Self) -> bool {
        self.is_dual == other.is_dual
            && ((Arc::ptr_eq(&self.data, &other.data) && self.moved == other.moved)
                || self.map() == other.map())
    }
}

impl Eq for SectorLeg {}

impl Hash for SectorLeg {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        // Same fields in the same order as the former derived hash, so hashes
        // depend on content only, never on which storage a leg shares.
        let (sectors, degeneracies) = self.map();
        sectors.hash(state);
        degeneracies.hash(state);
        self.is_dual.hash(state);
    }
}

impl fmt::Debug for SectorLeg {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        struct Data<'a>(&'a [SectorId], &'a [usize]);
        impl fmt::Debug for Data<'_> {
            fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
                formatter
                    .debug_struct("SectorLegData")
                    .field("sectors", &self.0)
                    .field("degeneracies", &self.1)
                    .finish()
            }
        }
        let (sectors, degeneracies) = self.map();
        formatter
            .debug_struct("SectorLeg")
            .field("data", &Data(sectors, degeneracies))
            .field("is_dual", &self.is_dual)
            .finish()
    }
}

impl SectorLeg {
    fn from_data(sectors: SectorVec, degeneracies: DimVec, is_dual: bool) -> Self {
        Self {
            data: Arc::new(SectorLegData {
                sectors,
                degeneracies,
                dual: OnceLock::new(),
            }),
            is_dual,
            moved: false,
        }
    }

    #[inline]
    fn map(&self) -> (&SectorVec, &DimVec) {
        if self.moved {
            let moved = self
                .data
                .dual
                .get()
                .and_then(|dual| dual.moved.as_ref())
                .expect("a moved leg's storage holds its moved dual map");
            (&moved.sectors, &moved.degeneracies)
        } else {
            (&self.data.sectors, &self.data.degeneracies)
        }
    }

    /// The stored `dual(sector)` of each of this leg's sectors, when a dual
    /// query has filled the shared map.
    fn stored_images(&self) -> Option<&[SectorId]> {
        let dual = self.data.dual.get()?;
        if self.moved {
            dual.moved.as_ref().map(|moved| moved.images.as_slice())
        } else {
            Some(&dual.images)
        }
    }

    /// The dual leg over the same storage; `data.dual` must be filled.
    fn flipped(&self) -> Self {
        let moves = self
            .data
            .dual
            .get()
            .is_some_and(|dual| dual.moved.is_some());
        Self {
            data: Arc::clone(&self.data),
            is_dual: !self.is_dual,
            moved: moves && !self.moved,
        }
    }

    /// Conservative retained bytes for this leg's Arc-backed sector and
    /// degeneracy metadata, excluding the inline `SectorLeg` pointer shell.
    ///
    /// The dual map's heap bytes are charged whether or not a dual query has
    /// filled it yet, so a leg's charge does not grow after a cache admits it.
    #[doc(hidden)]
    pub fn charged_retained_bytes(&self) -> usize {
        let len = self.data.sectors.len();
        let dual_sectors = if len > self.data.sectors.inline_size() {
            len.saturating_mul(3 * std::mem::size_of::<SectorId>())
        } else {
            0
        };
        let dual_degeneracies = if len > self.data.degeneracies.inline_size() {
            len.saturating_mul(std::mem::size_of::<usize>())
        } else {
            0
        };
        std::mem::size_of::<SectorLegData>()
            .saturating_add(2 * std::mem::size_of::<usize>())
            .saturating_add(spilled_smallvec_heap_bytes(&self.data.sectors))
            .saturating_add(spilled_smallvec_heap_bytes(&self.data.degeneracies))
            .saturating_add(dual_sectors)
            .saturating_add(dual_degeneracies)
    }

    /// Builds one external leg from `(sector, degeneracy)` pairs.
    ///
    /// Zero-degeneracy sectors are absent from the resulting leg. Remaining
    /// pairs are stored sorted by sector id.
    ///
    /// # Panics
    ///
    /// Panics when a sector appears more than once. Use [`Self::try_new`] to
    /// handle this error.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenet_core::{SectorLeg, Z2Irrep};
    ///
    /// let leg = SectorLeg::new([(Z2Irrep::ODD, 3), (Z2Irrep::EVEN, 2)], false);
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
        Self::try_new(pairs, is_dual).unwrap_or_else(|error| panic!("{error}"))
    }

    /// Fallible counterpart of [`Self::new`].
    ///
    /// # Errors
    ///
    /// Returns [`SectorLegConstructionError::DuplicateSector`] when any sector
    /// is declared more than once, including zero-degeneracy declarations.
    pub fn try_new<Pairs, Sector>(
        pairs: Pairs,
        is_dual: bool,
    ) -> Result<Self, SectorLegConstructionError>
    where
        Pairs: IntoIterator<Item = (Sector, usize)>,
        Sector: Into<SectorId>,
    {
        let pairs = pairs
            .into_iter()
            .map(|(sector, degeneracy)| (sector.into(), degeneracy))
            .collect::<SmallVec<[(SectorId, usize); 8]>>();
        Self::build(pairs, is_dual, None)
    }

    /// Shared body of [`Self::try_new`] and the dual constructors.
    ///
    /// `share` is the leg the result may share its sector data with when the
    /// two carry the same sector -> degeneracy map: the legs then differ only
    /// in their dual flag, which is the case TensorKit's `dual(V)` covers by
    /// reusing the sector dictionary outright.
    fn build(
        mut pairs: SmallVec<[(SectorId, usize); 8]>,
        is_dual: bool,
        share: Option<&Self>,
    ) -> Result<Self, SectorLegConstructionError> {
        pairs.sort_unstable_by_key(|&(sector, _)| sector);
        // Why not discard zeros first: duplicate validity must not depend on
        // input order or storage representation; TeNeT follows TensorKit's
        // strict tuple invariant, not its SectorDict zero-first corner.
        for window in pairs.windows(2) {
            if window[0].0 == window[1].0 {
                return Err(SectorLegConstructionError::DuplicateSector {
                    sector: window[0].0,
                });
            }
        }
        // Why compare before building: a leg whose map is unchanged by the
        // rule's dual differs from its dual only in the flag, so the dual can
        // borrow the same `Arc` and allocate nothing.
        if let Some(share) = share {
            if share.has_sorted_pairs(&pairs) {
                return Ok(Self {
                    data: Arc::clone(&share.data),
                    is_dual,
                    moved: share.moved,
                });
            }
        }
        let nonzero_count = pairs
            .iter()
            .filter(|(_, degeneracy)| *degeneracy != 0)
            .count();
        let mut sectors = SectorVec::new();
        let mut degeneracies = DimVec::new();
        sectors.reserve(nonzero_count);
        degeneracies.reserve(nonzero_count);
        for (sector, degeneracy) in pairs {
            if degeneracy == 0 {
                continue;
            }
            sectors.push(sector);
            degeneracies.push(degeneracy);
        }
        Ok(Self::from_data(sectors, degeneracies, is_dual))
    }

    pub fn from_sector_id(sector: usize, degeneracy: usize) -> Self {
        Self::new([(SectorId::new(sector), degeneracy)], false)
    }

    #[inline]
    pub fn sectors(&self) -> &[SectorId] {
        self.map().0
    }

    /// Per-sector degeneracies, parallel to [`Self::sectors`].
    #[inline]
    pub fn degeneracies(&self) -> &[usize] {
        self.map().1
    }

    /// Degeneracy of `sector` on this leg, `None` when the sector is not
    /// part of the leg.
    pub fn degeneracy(&self, sector: SectorId) -> Option<usize> {
        let (sectors, degeneracies) = self.map();
        sectors
            .binary_search(&sector)
            .ok()
            .map(|index| degeneracies[index])
    }

    /// `(sector, degeneracy)` pairs in sorted sector order.
    pub fn iter(&self) -> impl Iterator<Item = (SectorId, usize)> + '_ {
        let (sectors, degeneracies) = self.map();
        sectors.iter().copied().zip(degeneracies.iter().copied())
    }

    /// The dual leg: every sector is replaced by its dual (degeneracies
    /// carried along) and the dual flag is flipped.
    ///
    /// After the first query on a leg's storage, the dual shares that
    /// storage and allocates nothing.
    pub fn dual<R>(&self, rule: &R) -> Self
    where
        R: FusionRule,
    {
        self.dual_by(
            |sector| Ok(rule.dual(sector)),
            |sector| SectorLegConstructionError::DuplicateSector { sector },
        )
        .unwrap_or_else(|error| panic!("{error}"))
    }

    /// Shared body of the dual constructors. `dual_of` is queried once per
    /// sector in sector order, so the first rule error is the one returned;
    /// `not_injective` reports the smallest sector two sectors dualize onto.
    fn dual_by<E>(
        &self,
        mut dual_of: impl FnMut(SectorId) -> Result<SectorId, E>,
        not_injective: impl FnOnce(SectorId) -> E,
    ) -> Result<Self, E> {
        let (sectors, degeneracies) = self.map();
        if let Some(images) = self.stored_images() {
            for (index, (&sector, &image)) in sectors.iter().zip(images).enumerate() {
                let dual = dual_of(sector)?;
                if dual != image {
                    // A rule that disagrees with the filling rule: build the
                    // dual as a fresh leg and leave the shared map untouched.
                    let mut pairs = SmallVec::<[(SectorId, usize); 8]>::with_capacity(sectors.len());
                    pairs.extend(images[..index].iter().copied().zip(degeneracies.iter().copied()));
                    pairs.push((dual, degeneracies[index]));
                    for (&sector, &degeneracy) in
                        sectors[index + 1..].iter().zip(&degeneracies[index + 1..])
                    {
                        pairs.push((dual_of(sector)?, degeneracy));
                    }
                    return Self::build(pairs, !self.is_dual, Some(self)).map_err(
                        |SectorLegConstructionError::DuplicateSector { sector }| {
                            not_injective(sector)
                        },
                    );
                }
            }
            return Ok(self.flipped());
        }

        // First dual query on this storage. Only an unmoved leg gets here: a
        // moved leg exists only after its storage's dual map is filled.
        let mut triples = SmallVec::<[(SectorId, usize, SectorId); 8]>::with_capacity(sectors.len());
        for (&sector, &degeneracy) in sectors.iter().zip(degeneracies) {
            triples.push((dual_of(sector)?, degeneracy, sector));
        }
        let images = exact_sector_vec(triples.iter().map(|&(dual, _, _)| dual));
        triples.sort_unstable_by_key(|&(dual, _, _)| dual);
        for window in triples.windows(2) {
            if window[0].0 == window[1].0 {
                return Err(not_injective(window[0].0));
            }
        }
        let unchanged = triples
            .iter()
            .map(|&(dual, degeneracy, _)| (dual, degeneracy))
            .eq(self.iter());
        let moved = (!unchanged).then(|| {
            let mut degeneracies = DimVec::with_capacity(triples.len());
            degeneracies.extend(triples.iter().map(|&(_, degeneracy, _)| degeneracy));
            MovedSectorMap {
                sectors: exact_sector_vec(triples.iter().map(|&(dual, _, _)| dual)),
                degeneracies,
                images: exact_sector_vec(triples.iter().map(|&(_, _, source)| source)),
            }
        });
        match self.data.dual.set(DualSectorMap { images, moved }) {
            Ok(()) => Ok(self.flipped()),
            // Another thread filled the map first; share it only when its
            // rule agrees with this one.
            Err(map) if self.stored_images() == Some(map.images.as_slice()) => Ok(self.flipped()),
            Err(DualSectorMap { moved: None, .. }) => Ok(Self {
                data: Arc::clone(&self.data),
                is_dual: !self.is_dual,
                moved: false,
            }),
            Err(DualSectorMap {
                moved: Some(moved), ..
            }) => Ok(Self::from_data(
                moved.sectors,
                moved.degeneracies,
                !self.is_dual,
            )),
        }
    }

    /// Whether `pairs`, sorted by sector and free of zero degeneracies, is
    /// this leg's own sector -> degeneracy map.
    fn has_sorted_pairs(&self, pairs: &[(SectorId, usize)]) -> bool {
        pairs.len() == self.sectors().len()
            && pairs
                .iter()
                .zip(self.iter())
                .all(|(&pair, own)| pair == own)
    }

    /// Whether `other` borrows this leg's sector and degeneracy storage.
    #[doc(hidden)]
    pub fn shares_sector_data_with(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.data, &other.data)
    }

    /// The dual leg, returning a finite-algebra error when any sector has no
    /// representable dual ([`FusionAlgebraError`] from the rule) or when the
    /// rule's dual collapses two of this leg's sectors onto one
    /// ([`FusionAlgebraError::DualNotInjective`]).
    pub fn try_dual<R>(&self, rule: &R) -> Result<Self, FusionAlgebraError>
    where
        R: CheckedFusionAlgebra,
    {
        // Why not mutate a cloned leg as sectors succeed: a later failure must
        // leave no partially dualized value available to callers.
        //
        // Why an algebra error and not the panicking constructor: a rule whose
        // dual collapses two sectors onto one id breaks the rule's rigidity
        // structure; it is not a caller mistake.
        self.dual_by(
            |sector| rule.try_dual_sector(sector),
            |dual| FusionAlgebraError::DualNotInjective { dual },
        )
    }

    /// Checked Generic sibling of [`Self::try_dual`] that preserves the
    /// provider's concrete error without requiring `CheckedFusionAlgebra`.
    #[doc(hidden)]
    pub fn try_dual_generic<R>(
        &self,
        rule: &R,
    ) -> Result<Self, CheckedGenericStructureError<R::Error>>
    where
        R: CheckedGenericFusion,
    {
        self.dual_by(
            |sector| {
                rule.try_dual(sector)
                    .map_err(CheckedGenericStructureError::Provider)
            },
            |_| {
                CoreError::MalformedFusionTree {
                    message: "checked Generic dual is not injective on one tensor leg",
                }
                .into()
            },
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
