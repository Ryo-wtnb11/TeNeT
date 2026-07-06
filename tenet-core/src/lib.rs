#![forbid(unsafe_code)]

//! Core TensorMap-facing data structures for TeNeT.
//!
//! This crate owns TeNeT's public/core tensor view vocabulary. Lower-level
//! crates may lower these views to concrete strided kernels, but external
//! strided/backend types should not be required by TensorMap users.

use core::fmt;
use core::marker::PhantomData;
use core::ops::{Add, Mul};
use std::collections::hash_map::Entry;
use std::hash::Hash;
use std::sync::{Arc, OnceLock, RwLock, Weak};

use rustc_hash::FxHashMap;
use smallvec::SmallVec;

/// Inline storage for the low layer's small per-rank / per-leg / per-block
/// metadata — the Rust analog of TensorKit's `NTuple` stack fields on
/// `FusionTree`. Structural keys and layouts (sector lists, dims, duals,
/// block indices, strides) stay allocation-free for the common small ranks,
/// so hashing/cloning/comparing them in the cold structure/plan/recoupling
/// caches touches no heap. Inline capacity 8 covers typical tensor ranks and
/// per-leg sector counts; larger cases spill to heap exactly like `Vec`.
pub type SectorVec = SmallVec<[SectorId; 8]>;
/// Inline storage for `usize` metadata (dims, strides, indices, permutations).
pub type DimVec = SmallVec<[usize; 8]>;
/// Inline storage for per-leg duality flags.
pub type DualVec = SmallVec<[bool; 8]>;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Trivial;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Placement {
    Host,
    /// Storage resident on a CUDA device, identified by its ordinal.
    Cuda(usize),
}

pub trait TensorStorage<T> {
    fn len(&self) -> usize;
    fn placement(&self) -> Placement;

    #[inline]
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// Storage-backed scratch allocation matching the source storage placement.
///
/// This is the TensorKit `similar(data, len)` boundary in TeNeT-owned terms.
/// Implementations should allocate temporary storage with the same placement as
/// `self`; host storage returns host buffers, and future device storage should
/// return device-resident buffers.
pub trait SimilarStorage<T>: TensorStorage<T> {
    type Similar: TensorStorage<T>;

    fn similar_filled(&self, len: usize, value: T) -> Self::Similar
    where
        T: Clone;
}

/// Reusable same-placement scratch buffer.
///
/// `reset_filled` returns the buffer to exactly `len` elements equal to
/// `value`, reusing existing capacity where possible so replay paths do not
/// allocate on every call. Device scratch implementations should back this
/// with pooled (stream-ordered) reuse instead of fresh device allocations.
pub trait ScratchStorage<T>: TensorStorage<T> {
    fn reset_filled(&mut self, len: usize, value: T)
    where
        T: Clone;
}

impl<T> ScratchStorage<T> for Vec<T> {
    fn reset_filled(&mut self, len: usize, value: T)
    where
        T: Clone,
    {
        self.clear();
        self.resize(len, value);
    }
}

pub trait HostReadableStorage<T>: TensorStorage<T> {
    fn as_slice(&self) -> &[T];
}

pub trait HostWritableStorage<T>: HostReadableStorage<T> {
    fn as_mut_slice(&mut self) -> &mut [T];
}

pub type HostStorage<T> = Vec<T>;

impl<T> TensorStorage<T> for Vec<T> {
    #[inline]
    fn len(&self) -> usize {
        Vec::len(self)
    }

    #[inline]
    fn placement(&self) -> Placement {
        Placement::Host
    }
}

impl<T> SimilarStorage<T> for Vec<T> {
    type Similar = Vec<T>;

    #[inline]
    fn similar_filled(&self, len: usize, value: T) -> Self::Similar
    where
        T: Clone,
    {
        vec![value; len]
    }
}

impl<T> HostReadableStorage<T> for Vec<T> {
    #[inline]
    fn as_slice(&self) -> &[T] {
        self
    }
}

impl<T> HostWritableStorage<T> for Vec<T> {
    #[inline]
    fn as_mut_slice(&mut self) -> &mut [T] {
        self
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ProductSpace<const N: usize> {
    dims: [usize; N],
    dim: usize,
}

impl<const N: usize> ProductSpace<N> {
    pub fn new(dims: [usize; N]) -> Result<Self, CoreError> {
        let dim = checked_product(&dims)?;
        Ok(Self { dims, dim })
    }

    #[inline]
    pub fn dims(&self) -> &[usize; N] {
        &self.dims
    }

    #[inline]
    pub fn dim(&self) -> usize {
        self.dim
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TensorMapSpace<const NOUT: usize, const NIN: usize> {
    codomain: ProductSpace<NOUT>,
    domain: ProductSpace<NIN>,
    dims: DimVec,
    dense_dim: usize,
}

impl<const NOUT: usize, const NIN: usize> TensorMapSpace<NOUT, NIN> {
    pub fn new(codomain: ProductSpace<NOUT>, domain: ProductSpace<NIN>) -> Result<Self, CoreError> {
        let dense_dim = codomain
            .dim()
            .checked_mul(domain.dim())
            .ok_or(CoreError::ElementCountOverflow)?;
        let mut dims = DimVec::with_capacity(NOUT + NIN);
        dims.extend_from_slice(codomain.dims());
        dims.extend_from_slice(domain.dims());
        Ok(Self {
            codomain,
            domain,
            dims,
            dense_dim,
        })
    }

    /// Builds a dense tensor-map space from codomain and domain dimensions.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenet_core::TensorMapSpace;
    ///
    /// let space = TensorMapSpace::<2, 1>::from_dims([2, 3], [4]).unwrap();
    /// assert_eq!(space.dims(), &[2, 3, 4]);
    /// assert_eq!(space.dense_dim(), 24);
    /// ```
    pub fn from_dims(codomain: [usize; NOUT], domain: [usize; NIN]) -> Result<Self, CoreError> {
        Self::new(ProductSpace::new(codomain)?, ProductSpace::new(domain)?)
    }

    #[inline]
    pub fn codomain(&self) -> &ProductSpace<NOUT> {
        &self.codomain
    }

    #[inline]
    pub fn domain(&self) -> &ProductSpace<NIN> {
        &self.domain
    }

    #[inline]
    pub fn dims(&self) -> &[usize] {
        &self.dims
    }

    #[inline]
    pub fn dense_dim(&self) -> usize {
        self.dense_dim
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct SectorId(usize);

impl SectorId {
    pub const fn new(id: usize) -> Self {
        Self(id)
    }

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

#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub struct SectorLeg {
    sectors: SectorVec,
    /// Per-sector degeneracy, parallel to `sectors`. The leg is the single
    /// source of truth for the sector -> degeneracy map of one tensor axis
    /// (TensorKit `GradedSpace` parity: the space stores the complete map
    /// independent of which fusion trees are populated).
    degeneracies: DimVec,
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
            .collect::<Vec<_>>();
        pairs.sort_unstable();
        pairs.dedup();
        for window in pairs.windows(2) {
            assert_ne!(
                window[0].0, window[1].0,
                "sector {:?} listed with conflicting degeneracies {} and {}",
                window[0].0, window[0].1, window[1].1
            );
        }
        let (sectors, degeneracies) = pairs.into_iter().unzip();
        Self {
            sectors,
            degeneracies,
            is_dual,
        }
    }

    pub fn from_sector_id(sector: usize, degeneracy: usize) -> Self {
        Self::new([(SectorId::new(sector), degeneracy)], false)
    }

    #[inline]
    pub fn sectors(&self) -> &[SectorId] {
        &self.sectors
    }

    /// Per-sector degeneracies, parallel to [`Self::sectors`].
    #[inline]
    pub fn degeneracies(&self) -> &[usize] {
        &self.degeneracies
    }

    /// Degeneracy of `sector` on this leg, `None` when the sector is not
    /// part of the leg.
    pub fn degeneracy(&self, sector: SectorId) -> Option<usize> {
        self.sectors
            .binary_search(&sector)
            .ok()
            .map(|index| self.degeneracies[index])
    }

    /// `(sector, degeneracy)` pairs in sorted sector order.
    pub fn iter(&self) -> impl Iterator<Item = (SectorId, usize)> + '_ {
        self.sectors
            .iter()
            .copied()
            .zip(self.degeneracies.iter().copied())
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

#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub struct FusionProductSpace {
    legs: Vec<SectorLeg>,
}

impl FusionProductSpace {
    /// Builds a product of external fusion legs.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenet_core::{FusionProductSpace, SectorLeg, Z2Irrep};
    ///
    /// let space = FusionProductSpace::new([
    ///     SectorLeg::new([(Z2Irrep::EVEN, 1), (Z2Irrep::ODD, 1)], false),
    ///     SectorLeg::new([(Z2Irrep::EVEN, 1)], true),
    /// ]);
    /// assert_eq!(space.len(), 2);
    /// ```
    pub fn new<Legs>(legs: Legs) -> Self
    where
        Legs: IntoIterator<Item = SectorLeg>,
    {
        Self {
            legs: legs.into_iter().collect(),
        }
    }

    /// Builds a product of single-sector legs from `(sector id, degeneracy)`
    /// pairs.
    pub fn from_sector_ids<Sectors>(sectors: Sectors) -> Self
    where
        Sectors: IntoIterator<Item = (usize, usize)>,
    {
        Self::new(
            sectors
                .into_iter()
                .map(|(sector, degeneracy)| SectorLeg::from_sector_id(sector, degeneracy)),
        )
    }

    #[inline]
    pub fn legs(&self) -> &[SectorLeg] {
        &self.legs
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.legs.len()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.legs.is_empty()
    }

    fn selected_leg_tuples(&self) -> Vec<Vec<FusionTreeLeg>> {
        let mut tuples = Vec::new();
        let mut current = vec![None; self.legs.len()];
        collect_selected_leg_tuples(&self.legs, self.legs.len(), &mut current, &mut tuples);
        tuples
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub struct FusionTreeHomSpace {
    codomain: FusionProductSpace,
    domain: FusionProductSpace,
}

#[derive(Clone, Debug, Eq, PartialEq, Hash)]
struct FusionTreeLegSetSignature {
    sectors: SectorVec,
    is_dual: bool,
}

#[derive(Clone, Debug, Eq, PartialEq, Hash)]
struct FusionTreeHomSpaceCacheKey {
    rule_type: &'static str,
    codomain: Vec<FusionTreeLegSetSignature>,
    domain: Vec<FusionTreeLegSetSignature>,
}

impl FusionTreeHomSpaceCacheKey {
    fn new<R>(homspace: &FusionTreeHomSpace) -> Self
    where
        R: MultiplicityFreeFusionRule,
    {
        Self {
            rule_type: std::any::type_name::<R>(),
            codomain: fusion_product_space_signature(homspace.codomain()),
            domain: fusion_product_space_signature(homspace.domain()),
        }
    }
}

fn fusion_product_space_signature(space: &FusionProductSpace) -> Vec<FusionTreeLegSetSignature> {
    space
        .legs()
        .iter()
        .map(|leg| FusionTreeLegSetSignature {
            sectors: leg.sectors().iter().copied().collect(),
            is_dual: leg.is_dual(),
        })
        .collect()
}

#[derive(Clone, Debug)]
struct FusionTreeBlockLayoutEntry {
    row: usize,
    col: usize,
}

#[derive(Clone, Debug)]
struct FusionTreeCoupledSectorLayout {
    start: usize,
    row_count: usize,
    col_count: usize,
    entries: Vec<FusionTreeBlockLayoutEntry>,
}

#[derive(Clone, Debug)]
struct FusionTreeHomSpaceLayout {
    keys: Arc<[FusionTreeBlockKey]>,
    sectors: Vec<FusionTreeCoupledSectorLayout>,
}

#[derive(Clone, Debug, Eq, PartialEq, Hash)]
struct CoupledBlockStructureCacheKey {
    layout_ptr: usize,
    nout: usize,
    rank: usize,
    shapes: Arc<[DimVec]>,
}

fn fusion_tree_layout_cache(
) -> &'static RwLock<FxHashMap<FusionTreeHomSpaceCacheKey, Arc<FusionTreeHomSpaceLayout>>> {
    static CACHE: OnceLock<
        RwLock<FxHashMap<FusionTreeHomSpaceCacheKey, Arc<FusionTreeHomSpaceLayout>>>,
    > = OnceLock::new();
    CACHE.get_or_init(|| RwLock::new(FxHashMap::default()))
}

fn coupled_block_structure_cache(
) -> &'static RwLock<FxHashMap<CoupledBlockStructureCacheKey, Weak<BlockStructure>>> {
    static CACHE: OnceLock<RwLock<FxHashMap<CoupledBlockStructureCacheKey, Weak<BlockStructure>>>> =
        OnceLock::new();
    CACHE.get_or_init(|| RwLock::new(FxHashMap::default()))
}

fn fusion_tree_layout_from_keys<R>(
    rule: &R,
    keys: Vec<FusionTreeBlockKey>,
) -> FusionTreeHomSpaceLayout
where
    R: MultiplicityFreeFusionRule,
{
    let keys = Arc::<[FusionTreeBlockKey]>::from(keys);
    let mut sectors = Vec::new();
    let mut run_start = 0usize;
    while run_start < keys.len() {
        let coupled = coupled_or_vacuum(rule, keys[run_start].codomain_tree());
        let mut run_end = run_start;
        let mut row_indices = FxHashMap::<FusionTreeKey, usize>::default();
        let mut col_indices = FxHashMap::<FusionTreeKey, usize>::default();
        let mut entries = Vec::new();
        while run_end < keys.len()
            && coupled_or_vacuum(rule, keys[run_end].codomain_tree()) == coupled
        {
            let row = match row_indices.get(keys[run_end].codomain_tree()) {
                Some(&index) => index,
                None => {
                    let index = row_indices.len();
                    row_indices.insert(keys[run_end].codomain_tree().clone(), index);
                    index
                }
            };
            let col = match col_indices.get(keys[run_end].domain_tree()) {
                Some(&index) => index,
                None => {
                    let index = col_indices.len();
                    col_indices.insert(keys[run_end].domain_tree().clone(), index);
                    index
                }
            };
            entries.push(FusionTreeBlockLayoutEntry { row, col });
            run_end += 1;
        }
        sectors.push(FusionTreeCoupledSectorLayout {
            start: run_start,
            row_count: row_indices.len(),
            col_count: col_indices.len(),
            entries,
        });
        run_start = run_end;
    }
    FusionTreeHomSpaceLayout { keys, sectors }
}

impl FusionTreeHomSpace {
    /// Builds a fusion-tree hom space from codomain and domain product spaces.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenet_core::{FusionProductSpace, FusionTreeHomSpace, SectorLeg, Z2Irrep};
    ///
    /// let hom = FusionTreeHomSpace::new(
    ///     FusionProductSpace::new([SectorLeg::new([(Z2Irrep::EVEN, 1), (Z2Irrep::ODD, 1)], false)]),
    ///     FusionProductSpace::new([SectorLeg::new([(Z2Irrep::EVEN, 1)], false)]),
    /// );
    /// assert_eq!(hom.codomain().len(), 1);
    /// assert_eq!(hom.domain().len(), 1);
    /// ```
    pub fn new(codomain: FusionProductSpace, domain: FusionProductSpace) -> Self {
        Self { codomain, domain }
    }

    /// Builds a hom space when each external leg has exactly one sector,
    /// from `(sector, degeneracy)` pairs.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenet_core::{FusionTreeHomSpace, Z2Irrep};
    ///
    /// let hom = FusionTreeHomSpace::from_sectors([(Z2Irrep::EVEN, 1)], [(Z2Irrep::ODD, 1)]);
    /// assert_eq!(hom.codomain().len(), 1);
    /// assert_eq!(hom.domain().len(), 1);
    /// ```
    pub fn from_sectors<Codomain, Domain, CodomainSector, DomainSector>(
        codomain: Codomain,
        domain: Domain,
    ) -> Self
    where
        Codomain: IntoIterator<Item = (CodomainSector, usize)>,
        Domain: IntoIterator<Item = (DomainSector, usize)>,
        CodomainSector: Into<SectorId>,
        DomainSector: Into<SectorId>,
    {
        Self::new(
            FusionProductSpace::new(
                codomain
                    .into_iter()
                    .map(|(sector, degeneracy)| SectorLeg::new([(sector, degeneracy)], false)),
            ),
            FusionProductSpace::new(
                domain
                    .into_iter()
                    .map(|(sector, degeneracy)| SectorLeg::new([(sector, degeneracy)], false)),
            ),
        )
    }

    /// Builds a hom space from raw `(sector id, degeneracy)` pairs.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenet_core::FusionTreeHomSpace;
    ///
    /// let hom = FusionTreeHomSpace::from_sector_ids([(0, 1), (1, 1)], [(1, 1)]);
    /// assert_eq!(hom.codomain().len(), 2);
    /// assert_eq!(hom.domain().len(), 1);
    /// ```
    pub fn from_sector_ids<Codomain, Domain>(codomain: Codomain, domain: Domain) -> Self
    where
        Codomain: IntoIterator<Item = (usize, usize)>,
        Domain: IntoIterator<Item = (usize, usize)>,
    {
        Self::from_sectors(
            codomain
                .into_iter()
                .map(|(sector, degeneracy)| (SectorId::new(sector), degeneracy)),
            domain
                .into_iter()
                .map(|(sector, degeneracy)| (SectorId::new(sector), degeneracy)),
        )
    }

    #[inline]
    pub fn codomain(&self) -> &FusionProductSpace {
        &self.codomain
    }

    #[inline]
    pub fn domain(&self) -> &FusionProductSpace {
        &self.domain
    }

    #[inline]
    pub fn rank(&self) -> usize {
        self.codomain.len() + self.domain.len()
    }

    pub fn select<R>(
        &self,
        rule: &R,
        codomain_axes: &[usize],
        domain_axes: &[usize],
    ) -> Result<Self, CoreError>
    where
        R: FusionRule,
    {
        validate_axis_selection(codomain_axes, domain_axes, self.rank())?;

        let codomain = codomain_axes
            .iter()
            .map(|&axis| self.external_axis_leg(rule, axis))
            .collect::<Vec<_>>();
        let domain = domain_axes
            .iter()
            .map(|&axis| dual_sector_leg(rule, &self.external_axis_leg(rule, axis)))
            .collect::<Vec<_>>();
        Ok(Self::new(
            FusionProductSpace::new(codomain),
            FusionProductSpace::new(domain),
        ))
    }

    pub fn permute<R>(
        &self,
        rule: &R,
        codomain_axes: &[usize],
        domain_axes: &[usize],
    ) -> Result<Self, CoreError>
    where
        R: FusionRule,
    {
        let mut axes = Vec::with_capacity(codomain_axes.len() + domain_axes.len());
        axes.extend_from_slice(codomain_axes);
        axes.extend_from_slice(domain_axes);
        validate_permutation(&axes, self.rank())?;
        self.select(rule, codomain_axes, domain_axes)
    }

    pub fn compose<R>(_rule: &R, lhs: &Self, rhs: &Self) -> Result<Self, CoreError>
    where
        R: FusionRule,
    {
        if lhs.domain.len() != rhs.codomain.len() {
            return Err(CoreError::DimensionMismatch {
                expected: lhs.domain.len(),
                actual: rhs.codomain.len(),
            });
        }
        for (lhs_domain, rhs_codomain) in lhs.domain.legs().iter().zip(rhs.codomain.legs()) {
            validate_composed_leg(lhs_domain, rhs_codomain)?;
        }
        Ok(Self::new(lhs.codomain.clone(), rhs.domain.clone()))
    }

    /// Structural lowering of a general axis contraction, the homspace-level
    /// analog of TensorOperations' `tensorcontract!`: reorders both operands
    /// to `(open, contracted)` x `(contracted, open)`, composes them (see
    /// [`Self::compose`]), and applies the requested output axis order.
    pub fn tensorcontract_homspace<R>(
        rule: &R,
        lhs: &Self,
        rhs: &Self,
        lhs_contracting_axes: &[usize],
        rhs_contracting_axes: &[usize],
        output_axes: &[usize],
        dst_codomain_rank: usize,
    ) -> Result<Self, CoreError>
    where
        R: FusionRule,
    {
        if lhs_contracting_axes.len() != rhs_contracting_axes.len() {
            return Err(CoreError::DimensionMismatch {
                expected: lhs_contracting_axes.len(),
                actual: rhs_contracting_axes.len(),
            });
        }

        let lhs_seen = validate_axis_subset(lhs_contracting_axes, lhs.rank())?;
        let rhs_seen = validate_axis_subset(rhs_contracting_axes, rhs.rank())?;
        let lhs_open_axes = (0..lhs.rank())
            .filter(|&axis| !lhs_seen[axis])
            .collect::<Vec<_>>();
        let rhs_open_axes = (0..rhs.rank())
            .filter(|&axis| !rhs_seen[axis])
            .collect::<Vec<_>>();
        let output_rank = lhs_open_axes.len() + rhs_open_axes.len();
        validate_permutation(output_axes, output_rank)?;
        if dst_codomain_rank > output_rank {
            return Err(CoreError::StructureRankMismatch {
                expected: output_rank,
                actual: dst_codomain_rank,
            });
        }

        let lhs = lhs.permute(rule, &lhs_open_axes, lhs_contracting_axes)?;
        let rhs = rhs.permute(rule, rhs_contracting_axes, &rhs_open_axes)?;
        let composed = Self::compose(rule, &lhs, &rhs)?;
        composed.permute(
            rule,
            &output_axes[..dst_codomain_rank],
            &output_axes[dst_codomain_rank..],
        )
    }

    pub fn fusion_tree_keys<R>(&self, rule: &R) -> Vec<FusionTreeBlockKey>
    where
        R: MultiplicityFreeFusionRule,
    {
        self.cached_fusion_tree_layout(rule).keys.as_ref().to_vec()
    }

    pub fn try_for_each_fusion_tree_key<R, F, E>(&self, rule: &R, mut f: F) -> Result<(), E>
    where
        R: MultiplicityFreeFusionRule,
        F: FnMut(&FusionTreeBlockKey) -> Result<(), E>,
    {
        let layout = self.cached_fusion_tree_layout(rule);
        for key in layout.keys.iter() {
            f(key)?;
        }
        Ok(())
    }

    pub fn try_with_fusion_tree_keys<R, F, T, E>(&self, rule: &R, f: F) -> Result<T, E>
    where
        R: MultiplicityFreeFusionRule,
        F: FnOnce(&[FusionTreeBlockKey]) -> Result<T, E>,
    {
        let layout = self.cached_fusion_tree_layout(rule);
        f(layout.keys.as_ref())
    }

    fn cached_fusion_tree_layout<R>(&self, rule: &R) -> Arc<FusionTreeHomSpaceLayout>
    where
        R: MultiplicityFreeFusionRule,
    {
        let key = FusionTreeHomSpaceCacheKey::new::<R>(self);
        let cache = fusion_tree_layout_cache();
        if let Ok(read) = cache.read() {
            if let Some(layout) = read.get(&key) {
                return Arc::clone(layout);
            }
        }

        let computed = Arc::new(fusion_tree_layout_from_keys(
            rule,
            self.fusion_tree_keys_uncached(rule),
        ));
        if let Ok(mut write) = cache.write() {
            return Arc::clone(write.entry(key).or_insert_with(|| Arc::clone(&computed)));
        }
        computed
    }

    pub fn coupled_subblock_structure<R, Shapes>(
        &self,
        rule: &R,
        nout: usize,
        shapes: Shapes,
    ) -> Result<Arc<BlockStructure>, CoreError>
    where
        R: MultiplicityFreeFusionRule,
        Shapes: IntoIterator,
        Shapes::Item: Into<Vec<usize>>,
    {
        let layout = self.cached_fusion_tree_layout(rule);
        let rank = self.codomain().len() + self.domain().len();
        let shapes = shapes
            .into_iter()
            .map(|shape| shape.into().into_iter().collect::<DimVec>())
            .collect::<Vec<_>>();
        if layout.keys.len() != shapes.len() {
            return Err(CoreError::BlockCountMismatch {
                expected: layout.keys.len(),
                actual: shapes.len(),
            });
        }
        self.validate_degeneracy_shapes(layout.keys.as_ref(), &shapes)?;

        let cache_key = CoupledBlockStructureCacheKey {
            layout_ptr: Arc::as_ptr(&layout) as usize,
            nout,
            rank,
            shapes: Arc::<[DimVec]>::from(shapes),
        };
        let cache = coupled_block_structure_cache();
        if let Ok(read) = cache.read() {
            if let Some(structure) = read.get(&cache_key).and_then(Weak::upgrade) {
                return Ok(structure);
            }
        }

        let specs = coupled_sector_matrix_block_specs_from_layout(
            nout,
            rank,
            &layout,
            cache_key.shapes.as_ref(),
        )?;
        let structure = BlockStructure::from_blocks_with_rank(rank, specs)?.into_shared();

        let mut write = cache
            .write()
            .expect("coupled block structure cache poisoned");
        if let Some(existing) = write.get(&cache_key).and_then(Weak::upgrade) {
            return Ok(existing);
        }
        write.insert(cache_key, Arc::downgrade(&structure));
        Ok(structure)
    }

    fn fusion_tree_keys_uncached<R>(&self, rule: &R) -> Vec<FusionTreeBlockKey>
    where
        R: MultiplicityFreeFusionRule,
    {
        let codomain = fusion_trees_by_coupled_for_space(rule, &self.codomain);
        let domain = fusion_trees_by_coupled_for_space(rule, &self.domain);
        let mut keys = Vec::new();
        let mut codomain_index = 0usize;
        let mut domain_index = 0usize;
        while codomain_index < codomain.len() && domain_index < domain.len() {
            match codomain[codomain_index]
                .coupled
                .cmp(&domain[domain_index].coupled)
            {
                std::cmp::Ordering::Less => codomain_index += 1,
                std::cmp::Ordering::Greater => domain_index += 1,
                std::cmp::Ordering::Equal => {
                    for domain_tree in &domain[domain_index].trees {
                        for codomain_tree in &codomain[codomain_index].trees {
                            keys.push(FusionTreeBlockKey::pair(
                                codomain_tree.clone(),
                                domain_tree.clone(),
                            ));
                        }
                    }
                    codomain_index += 1;
                    domain_index += 1;
                }
            }
        }
        keys
    }

    pub fn sector_structure<R>(&self, rule: &R) -> Result<SectorStructure, CoreError>
    where
        R: MultiplicityFreeFusionRule,
    {
        let rank = self.codomain.len() + self.domain.len();
        SectorStructure::from_keys(rank, self.fusion_tree_keys(rule))
    }

    pub fn unique_fusion_tree_key_from_external_sectors<R>(
        &self,
        rule: &R,
        sectors: &[SectorId],
    ) -> Result<FusionTreeBlockKey, CoreError>
    where
        R: MultiplicityFreeFusionRule,
    {
        let mut keys = self.fusion_tree_keys_from_external_sectors(rule, sectors)?;
        if keys.len() != 1 {
            return Err(CoreError::BlockCountMismatch {
                expected: 1,
                actual: keys.len(),
            });
        }
        Ok(keys.remove(0))
    }

    /// Lowers external per-leg sectors to fusion-tree keys. Domain-side
    /// external sectors are dualized into internal tree sectors here, the
    /// same convention TensorKit applies in `subblock(t, sectors)`.
    pub fn fusion_tree_keys_from_external_sectors<R>(
        &self,
        rule: &R,
        sectors: &[SectorId],
    ) -> Result<Vec<FusionTreeBlockKey>, CoreError>
    where
        R: MultiplicityFreeFusionRule,
    {
        let rank = self.codomain.len() + self.domain.len();
        if sectors.len() != rank {
            return Err(CoreError::DimensionMismatch {
                expected: rank,
                actual: sectors.len(),
            });
        }

        let codomain = fusion_trees_by_coupled_for_selected_space(
            rule,
            &self.codomain,
            &sectors[..self.codomain.len()],
        )?;
        let domain_sectors = sectors[self.codomain.len()..]
            .iter()
            .map(|&sector| rule.dual(sector))
            .collect::<Vec<_>>();
        let domain =
            fusion_trees_by_coupled_for_selected_space(rule, &self.domain, &domain_sectors)?;
        let mut keys = Vec::new();
        let mut codomain_index = 0usize;
        let mut domain_index = 0usize;
        while codomain_index < codomain.len() && domain_index < domain.len() {
            match codomain[codomain_index]
                .coupled
                .cmp(&domain[domain_index].coupled)
            {
                std::cmp::Ordering::Less => codomain_index += 1,
                std::cmp::Ordering::Greater => domain_index += 1,
                std::cmp::Ordering::Equal => {
                    for domain_tree in &domain[domain_index].trees {
                        for codomain_tree in &codomain[codomain_index].trees {
                            keys.push(FusionTreeBlockKey::pair(
                                codomain_tree.clone(),
                                domain_tree.clone(),
                            ));
                        }
                    }
                    codomain_index += 1;
                    domain_index += 1;
                }
            }
        }

        Ok(keys)
    }

    /// Validates per-tree degeneracy shapes against the leg-level
    /// degeneracies (the legs are authoritative): for every tree key,
    /// `shape[axis]` must equal the degeneracy the axis' leg stores for the
    /// tree's uncoupled sector on that axis.
    pub fn validate_degeneracy_shapes<S>(
        &self,
        keys: &[FusionTreeBlockKey],
        shapes: &[S],
    ) -> Result<(), CoreError>
    where
        S: AsRef<[usize]>,
    {
        let legs = self
            .codomain
            .legs()
            .iter()
            .chain(self.domain.legs())
            .collect::<Vec<_>>();
        for (key, shape) in keys.iter().zip(shapes) {
            let shape = shape.as_ref();
            if shape.len() != legs.len() {
                return Err(CoreError::StructureRankMismatch {
                    expected: legs.len(),
                    actual: shape.len(),
                });
            }
            let uncoupled = key
                .codomain_uncoupled()
                .iter()
                .chain(key.domain_uncoupled());
            for ((leg, &sector), &dim) in legs.iter().zip(uncoupled).zip(shape) {
                let expected = leg
                    .degeneracy(sector)
                    .ok_or(CoreError::MalformedFusionTree {
                        message: "fusion tree uses a sector absent from its leg",
                    })?;
                if expected != dim {
                    return Err(CoreError::LegDegeneracyMismatch {
                        sector,
                        expected,
                        actual: dim,
                    });
                }
            }
        }
        Ok(())
    }

    pub fn fusion_tree_groups<R>(&self, rule: &R) -> Result<Vec<FusionTreeBlockGroup>, CoreError>
    where
        R: MultiplicityFreeFusionRule,
    {
        self.sector_structure(rule)
            .map(|structure| structure.fusion_tree_groups())
    }

    /// External leg view of flat axis `axis` (TensorKit's `space(t, i)`
    /// convention, homspace.jl:60-62): codomain legs verbatim, domain legs
    /// dualized. Degeneracies are carried along, keyed by the external
    /// (placement-invariant) sector labels.
    pub fn external_axis_leg<R>(&self, rule: &R, axis: usize) -> SectorLeg
    where
        R: FusionRule,
    {
        if axis < self.codomain.len() {
            self.codomain.legs()[axis].clone()
        } else {
            dual_sector_leg(rule, &self.domain.legs()[axis - self.codomain.len()])
        }
    }
}

fn dual_sector_leg<R>(rule: &R, leg: &SectorLeg) -> SectorLeg
where
    R: FusionRule,
{
    leg.dual(rule)
}

fn validate_axis_subset(axes: &[usize], rank: usize) -> Result<Vec<bool>, CoreError> {
    let mut seen = vec![false; rank];
    for &axis in axes {
        if axis >= rank || seen[axis] {
            return Err(CoreError::InvalidPermutation {
                permutation: axes.to_vec(),
                rank,
            });
        }
        seen[axis] = true;
    }
    Ok(seen)
}

fn validate_axis_selection(
    codomain_axes: &[usize],
    domain_axes: &[usize],
    rank: usize,
) -> Result<(), CoreError> {
    for &axis in codomain_axes.iter().chain(domain_axes) {
        if axis >= rank {
            let mut axes = Vec::with_capacity(codomain_axes.len() + domain_axes.len());
            axes.extend_from_slice(codomain_axes);
            axes.extend_from_slice(domain_axes);
            return Err(CoreError::InvalidPermutation {
                permutation: axes,
                rank,
            });
        }
    }
    Ok(())
}

fn validate_composed_leg(
    lhs_domain: &SectorLeg,
    rhs_codomain: &SectorLeg,
) -> Result<(), CoreError> {
    if lhs_domain.is_dual() != rhs_codomain.is_dual() {
        return Err(CoreError::MalformedFusionTree {
            message: "contracted fusion leg duality flags do not match",
        });
    }
    // TensorKit parity: `A * B` requires `domain(A) == codomain(B)` as
    // spaces, so the stored legs must match verbatim (domain legs store the
    // domain space's own sectors; verified against TensorKit v0.16:
    // `rand(V ← V) * rand(V ← V)` works for V = U1Space(0=>1, 1=>1), a
    // sector set that is not dualization-closed, while `(V ← V) * (? ← V')`
    // is a SpaceMismatch).
    if lhs_domain.sectors().len() != rhs_codomain.sectors().len() {
        return Err(CoreError::DimensionMismatch {
            expected: lhs_domain.sectors().len(),
            actual: rhs_codomain.sectors().len(),
        });
    }
    for ((expected, expected_deg), (actual, actual_deg)) in
        lhs_domain.iter().zip(rhs_codomain.iter())
    {
        if expected != actual {
            return Err(CoreError::SectorMismatch { expected, actual });
        }
        if expected_deg != actual_deg {
            return Err(CoreError::LegDegeneracyMismatch {
                sector: expected,
                expected: expected_deg,
                actual: actual_deg,
            });
        }
    }
    Ok(())
}

/// Computes coupled-sector matrix block specs for fusion-tree subblocks.
///
/// Keys must arrive grouped by coupled sector (the `fusion_tree_keys`
/// enumeration order). Within one coupled sector every codomain tree defines a
/// row block and every domain tree a column block of one column-major sector
/// matrix; the subblock for `(codomain tree, domain tree)` is the strided view
/// at that (row block, column block) position. Full coverage of the
/// `rows × columns` grid is required so the sector matrix has no
/// uninitialized holes.
fn coupled_sector_matrix_block_specs<R, S>(
    rule: &R,
    nout: usize,
    rank: usize,
    keys: &[FusionTreeBlockKey],
    shapes: &[S],
) -> Result<Vec<BlockSpec>, CoreError>
where
    R: FusionRule,
    S: AsRef<[usize]>,
{
    for shape in shapes {
        let shape = shape.as_ref();
        if shape.len() != rank {
            return Err(CoreError::StructureRankMismatch {
                expected: rank,
                actual: shape.len(),
            });
        }
    }

    let mut specs = Vec::with_capacity(keys.len());
    let mut seen_sectors: Vec<SectorId> = Vec::new();
    let mut sector_offset = 0usize;
    let mut run_start = 0usize;
    while run_start < keys.len() {
        let coupled = coupled_or_vacuum(rule, keys[run_start].codomain_tree());
        if seen_sectors.contains(&coupled) {
            return Err(CoreError::MalformedFusionTree {
                message: "coupled sectors must be contiguous in fusion tree key order",
            });
        }
        seen_sectors.push(coupled);
        let mut run_end = run_start;
        while run_end < keys.len()
            && coupled_or_vacuum(rule, keys[run_end].codomain_tree()) == coupled
        {
            if coupled_or_vacuum(rule, keys[run_end].domain_tree()) != coupled {
                return Err(CoreError::MalformedFusionTree {
                    message: "codomain and domain trees must share the coupled sector",
                });
            }
            run_end += 1;
        }

        // Row/column blocks keep first-seen order (offsets are cumulative), with
        // a hash side-index for O(1) tree lookup instead of a linear scan: a run
        // can hold many blocks, so the scan was O(run^1.5).
        let mut row_blocks: Vec<(&FusionTreeKey, usize, usize)> = Vec::new();
        let mut col_blocks: Vec<(&FusionTreeKey, usize, usize)> = Vec::new();
        let mut row_index: FxHashMap<&FusionTreeKey, usize> = FxHashMap::default();
        let mut col_index: FxHashMap<&FusionTreeKey, usize> = FxHashMap::default();
        for index in run_start..run_end {
            let key = &keys[index];
            let shape = shapes[index].as_ref();
            let row_dim = shape[..nout].iter().product::<usize>();
            let col_dim = shape[nout..].iter().product::<usize>();
            match row_index.get(key.codomain_tree()).copied() {
                Some(existing_index) if row_blocks[existing_index].2 != row_dim => {
                    return Err(CoreError::DimensionMismatch {
                        expected: row_blocks[existing_index].2,
                        actual: row_dim,
                    });
                }
                Some(_) => {}
                None => {
                    let offset = row_blocks
                        .last()
                        .map(|(_, start, dim)| start + dim)
                        .unwrap_or(0);
                    row_index.insert(key.codomain_tree(), row_blocks.len());
                    row_blocks.push((key.codomain_tree(), offset, row_dim));
                }
            }
            match col_index.get(key.domain_tree()).copied() {
                Some(existing_index) if col_blocks[existing_index].2 != col_dim => {
                    return Err(CoreError::DimensionMismatch {
                        expected: col_blocks[existing_index].2,
                        actual: col_dim,
                    });
                }
                Some(_) => {}
                None => {
                    let offset = col_blocks
                        .last()
                        .map(|(_, start, dim)| start + dim)
                        .unwrap_or(0);
                    col_index.insert(key.domain_tree(), col_blocks.len());
                    col_blocks.push((key.domain_tree(), offset, col_dim));
                }
            }
        }
        if run_end - run_start != row_blocks.len() * col_blocks.len() {
            return Err(CoreError::BlockCountMismatch {
                expected: row_blocks.len() * col_blocks.len(),
                actual: run_end - run_start,
            });
        }
        let matrix_rows = row_blocks
            .last()
            .map(|(_, start, dim)| start + dim)
            .unwrap_or(0);
        let matrix_cols = col_blocks
            .last()
            .map(|(_, start, dim)| start + dim)
            .unwrap_or(0);

        for index in run_start..run_end {
            let key = &keys[index];
            let shape = shapes[index].as_ref();
            let (_, row_start, _) = row_blocks
                .iter()
                .find(|(tree, _, _)| *tree == key.codomain_tree())
                .expect("row block registered above");
            let (_, col_start, _) = col_blocks
                .iter()
                .find(|(tree, _, _)| *tree == key.domain_tree())
                .expect("column block registered above");
            let mut strides = Vec::with_capacity(rank);
            let mut stride = 1usize;
            for &dim in &shape[..nout] {
                strides.push(stride);
                stride = stride
                    .checked_mul(dim)
                    .ok_or(CoreError::ElementCountOverflow)?;
            }
            let mut stride = matrix_rows;
            for &dim in &shape[nout..] {
                strides.push(stride);
                stride = stride
                    .checked_mul(dim)
                    .ok_or(CoreError::ElementCountOverflow)?;
            }
            let offset = sector_offset
                + row_start
                + matrix_rows
                    .checked_mul(*col_start)
                    .ok_or(CoreError::ElementCountOverflow)?;
            specs.push(BlockSpec::with_key(
                BlockKey::FusionTree(key.clone()),
                shape.to_vec(),
                strides,
                offset,
            )?);
        }

        sector_offset = sector_offset
            .checked_add(
                matrix_rows
                    .checked_mul(matrix_cols)
                    .ok_or(CoreError::ElementCountOverflow)?,
            )
            .ok_or(CoreError::ElementCountOverflow)?;
        run_start = run_end;
    }
    Ok(specs)
}

fn coupled_sector_matrix_block_specs_from_layout<S>(
    nout: usize,
    rank: usize,
    layout: &FusionTreeHomSpaceLayout,
    shapes: &[S],
) -> Result<Vec<BlockSpec>, CoreError>
where
    S: AsRef<[usize]>,
{
    let keys = layout.keys.as_ref();
    if keys.len() != shapes.len() {
        return Err(CoreError::BlockCountMismatch {
            expected: keys.len(),
            actual: shapes.len(),
        });
    }
    for shape in shapes {
        let shape = shape.as_ref();
        if shape.len() != rank {
            return Err(CoreError::StructureRankMismatch {
                expected: rank,
                actual: shape.len(),
            });
        }
    }

    let mut specs = Vec::with_capacity(keys.len());
    let mut sector_offset = 0usize;
    for sector in &layout.sectors {
        if sector.entries.len() != sector.row_count * sector.col_count {
            return Err(CoreError::BlockCountMismatch {
                expected: sector.row_count * sector.col_count,
                actual: sector.entries.len(),
            });
        }

        let mut row_dims = vec![None; sector.row_count];
        let mut col_dims = vec![None; sector.col_count];
        for (local_index, entry) in sector.entries.iter().enumerate() {
            let shape = shapes[sector.start + local_index].as_ref();
            register_layout_dim(&mut row_dims[entry.row], shape[..nout].iter().product())?;
            register_layout_dim(&mut col_dims[entry.col], shape[nout..].iter().product())?;
        }

        let row_dims = row_dims
            .into_iter()
            .map(|dim| {
                dim.ok_or(CoreError::MalformedFusionTree {
                    message: "cached fusion tree layout has an empty row",
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        let col_dims = col_dims
            .into_iter()
            .map(|dim| {
                dim.ok_or(CoreError::MalformedFusionTree {
                    message: "cached fusion tree layout has an empty column",
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        let row_offsets = prefix_offsets(&row_dims)?;
        let col_offsets = prefix_offsets(&col_dims)?;
        let matrix_rows = row_offsets
            .last()
            .zip(row_dims.last())
            .map(|(&offset, &dim)| offset + dim)
            .unwrap_or(0);
        let matrix_cols = col_offsets
            .last()
            .zip(col_dims.last())
            .map(|(&offset, &dim)| offset + dim)
            .unwrap_or(0);

        for (local_index, entry) in sector.entries.iter().enumerate() {
            let index = sector.start + local_index;
            let shape = shapes[index].as_ref();
            let mut strides = Vec::with_capacity(rank);
            let mut stride = 1usize;
            for &dim in &shape[..nout] {
                strides.push(stride);
                stride = stride
                    .checked_mul(dim)
                    .ok_or(CoreError::ElementCountOverflow)?;
            }
            let mut stride = matrix_rows;
            for &dim in &shape[nout..] {
                strides.push(stride);
                stride = stride
                    .checked_mul(dim)
                    .ok_or(CoreError::ElementCountOverflow)?;
            }
            let offset = sector_offset
                + row_offsets[entry.row]
                + matrix_rows
                    .checked_mul(col_offsets[entry.col])
                    .ok_or(CoreError::ElementCountOverflow)?;
            specs.push(BlockSpec::with_key(
                BlockKey::FusionTree(keys[index].clone()),
                shape.to_vec(),
                strides,
                offset,
            )?);
        }

        sector_offset = sector_offset
            .checked_add(
                matrix_rows
                    .checked_mul(matrix_cols)
                    .ok_or(CoreError::ElementCountOverflow)?,
            )
            .ok_or(CoreError::ElementCountOverflow)?;
    }
    Ok(specs)
}

fn register_layout_dim(slot: &mut Option<usize>, dim: usize) -> Result<(), CoreError> {
    match slot {
        Some(existing) if *existing != dim => Err(CoreError::DimensionMismatch {
            expected: *existing,
            actual: dim,
        }),
        Some(_) => Ok(()),
        None => {
            *slot = Some(dim);
            Ok(())
        }
    }
}

fn prefix_offsets(dims: &[usize]) -> Result<Vec<usize>, CoreError> {
    let mut offsets = Vec::with_capacity(dims.len());
    let mut offset = 0usize;
    for &dim in dims {
        offsets.push(offset);
        offset = offset
            .checked_add(dim)
            .ok_or(CoreError::ElementCountOverflow)?;
    }
    Ok(offsets)
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct FusionTensorMapSpace<const NOUT: usize, const NIN: usize> {
    dense_space: TensorMapSpace<NOUT, NIN>,
    homspace: Arc<FusionTreeHomSpace>,
    subblock_structure: Arc<BlockStructure>,
}

impl<const NOUT: usize, const NIN: usize> FusionTensorMapSpace<NOUT, NIN> {
    /// Builds a symmetric tensor-map space from an explicit block structure.
    ///
    /// Prefer [`Self::from_degeneracy_shapes`] for ordinary construction; use
    /// this method when the block structure has already been prepared.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenet_core::{
    ///     BlockStructure, FusionTensorMapSpace, FusionTreeHomSpace, TensorMapSpace,
    /// };
    ///
    /// let dense = TensorMapSpace::<1, 0>::from_dims([2], []).unwrap();
    /// let hom = FusionTreeHomSpace::from_sector_ids([(0, 2)], std::iter::empty::<(usize, usize)>());
    /// let structure = BlockStructure::packed_column_major(1, [vec![2]]).unwrap();
    ///
    /// let space = FusionTensorMapSpace::new(dense, hom, structure).unwrap();
    /// assert_eq!(space.required_len().unwrap(), 2);
    /// ```
    pub fn new(
        dense_space: TensorMapSpace<NOUT, NIN>,
        homspace: FusionTreeHomSpace,
        subblock_structure: BlockStructure,
    ) -> Result<Self, CoreError> {
        Self::from_shared_subblock_structure(
            dense_space,
            homspace,
            subblock_structure.into_shared(),
        )
    }

    pub fn from_shared_subblock_structure(
        dense_space: TensorMapSpace<NOUT, NIN>,
        homspace: FusionTreeHomSpace,
        subblock_structure: Arc<BlockStructure>,
    ) -> Result<Self, CoreError> {
        Self::validate_homspace_rank(&homspace)?;
        let rank = NOUT + NIN;
        if subblock_structure.rank() != rank {
            return Err(CoreError::StructureRankMismatch {
                expected: rank,
                actual: subblock_structure.rank(),
            });
        }
        let subblock_structure = BlockStructure::canonicalize_shared(subblock_structure);
        Ok(Self {
            dense_space,
            homspace: Arc::new(homspace),
            subblock_structure,
        })
    }

    /// Default constructor: TensorKit-equivalent coupled-sector matrix
    /// layout (see [`Self::from_degeneracy_shapes_coupled`]). This is the
    /// only product layout.
    ///
    /// The shapes are given per fusion-tree **subblock** (one entry per
    /// fusion-tree key, in key order), not per coupled-sector matrix block.
    /// This mirrors TensorKit's block/subblock distinction: a coupled-sector
    /// matrix block is assembled from these tree-resolved degeneracy shapes.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenet_core::{
    ///     FusionTensorMapSpace, FusionTreeHomSpace, TensorMapSpace, Z2FusionRule, Z2Irrep,
    /// };
    ///
    /// let rule = Z2FusionRule;
    /// let space = FusionTensorMapSpace::from_degeneracy_shapes(
    ///     TensorMapSpace::<1, 1>::from_dims([1], [1]).unwrap(),
    ///     FusionTreeHomSpace::from_sectors([(Z2Irrep::EVEN, 1)], [(Z2Irrep::EVEN, 1)]),
    ///     &rule,
    ///     [vec![1, 1]],
    /// )
    /// .unwrap();
    /// assert_eq!(space.required_len().unwrap(), 1);
    /// ```
    pub fn from_degeneracy_shapes<R, Shapes>(
        dense_space: TensorMapSpace<NOUT, NIN>,
        homspace: FusionTreeHomSpace,
        rule: &R,
        shapes: Shapes,
    ) -> Result<Self, CoreError>
    where
        R: MultiplicityFreeFusionRule,
        Shapes: IntoIterator,
        Shapes::Item: Into<Vec<usize>>,
    {
        Self::from_degeneracy_shapes_coupled(dense_space, homspace, rule, shapes)
    }

    /// TensorKit-style coupled-sector matrix layout.
    ///
    /// Each coupled sector stores one contiguous column-major matrix whose
    /// rows enumerate (codomain fusion tree × codomain degeneracies) and whose
    /// columns enumerate (domain fusion tree × domain degeneracies). Fusion
    /// tree subblocks are strided views into that matrix, so the canonical
    /// (codomain | domain) matricization needs no packing. Block keys and
    /// their order are identical to [`Self::from_degeneracy_shapes`]; only
    /// strides and offsets differ.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenet_core::{
    ///     FusionProductSpace, FusionTensorMapSpace, FusionTreeHomSpace, SectorLeg,
    ///     TensorMapSpace, Z2FusionRule, Z2Irrep,
    /// };
    ///
    /// let rule = Z2FusionRule;
    /// let leg = || SectorLeg::new([(Z2Irrep::EVEN, 1), (Z2Irrep::ODD, 1)], false);
    /// let space = FusionTensorMapSpace::from_degeneracy_shapes_coupled(
    ///     TensorMapSpace::<1, 1>::from_dims([2], [2]).unwrap(),
    ///     FusionTreeHomSpace::new(
    ///         FusionProductSpace::new([leg()]),
    ///         FusionProductSpace::new([leg()]),
    ///     ),
    ///     &rule,
    ///     [vec![1, 1], vec![1, 1]],
    /// )
    /// .unwrap();
    /// assert_eq!(space.required_len().unwrap(), 2);
    /// ```
    pub fn from_degeneracy_shapes_coupled<R, Shapes>(
        dense_space: TensorMapSpace<NOUT, NIN>,
        homspace: FusionTreeHomSpace,
        rule: &R,
        shapes: Shapes,
    ) -> Result<Self, CoreError>
    where
        R: MultiplicityFreeFusionRule,
        Shapes: IntoIterator,
        Shapes::Item: Into<Vec<usize>>,
    {
        Self::validate_homspace_rank(&homspace)?;
        let subblock_structure = homspace.coupled_subblock_structure(rule, NOUT, shapes)?;
        Self::from_shared_subblock_structure(dense_space, homspace, subblock_structure)
    }

    fn validate_homspace_rank(homspace: &FusionTreeHomSpace) -> Result<(), CoreError> {
        if homspace.codomain().len() != NOUT {
            return Err(CoreError::StructureRankMismatch {
                expected: NOUT,
                actual: homspace.codomain().len(),
            });
        }
        if homspace.domain().len() != NIN {
            return Err(CoreError::StructureRankMismatch {
                expected: NIN,
                actual: homspace.domain().len(),
            });
        }
        Ok(())
    }

    #[inline]
    pub fn dense_space(&self) -> &TensorMapSpace<NOUT, NIN> {
        &self.dense_space
    }

    #[inline]
    pub fn homspace(&self) -> &FusionTreeHomSpace {
        &self.homspace
    }

    /// Shared handle to the hom space; lets replay caches compare spaces by
    /// pointer identity before falling back to structural equality.
    #[inline]
    pub fn homspace_arc(&self) -> &Arc<FusionTreeHomSpace> {
        &self.homspace
    }

    #[inline]
    pub fn subblock_structure(&self) -> &Arc<BlockStructure> {
        &self.subblock_structure
    }

    pub fn find_subblock_index(&self, key: &FusionTreeBlockKey) -> Option<usize> {
        self.subblock_structure
            .find_block_index_by_fusion_tree_key(key)
    }

    pub fn required_len(&self) -> Result<usize, CoreError> {
        self.subblock_structure.required_len()
    }
}

pub trait FusionRule {
    fn fusion_style(&self) -> FusionStyleKind;

    fn braiding_style(&self) -> BraidingStyleKind;

    fn vacuum(&self) -> SectorId;

    fn supports_unitary_braid_dagger(&self) -> bool {
        false
    }

    fn dual(&self, sector: SectorId) -> SectorId {
        sector
    }

    /// Fusion channels of `left ⊗ right`. Returns a `SectorVec` so the common
    /// small result stays stack-inline — this is called millions of times in
    /// the cold recoupling build, and a heap `Vec` per call was ~5% of all
    /// cold-path allocations.
    fn fusion_channels(&self, left: SectorId, right: SectorId) -> SectorVec;

    fn nsymbol(&self, left: SectorId, right: SectorId, coupled: SectorId) -> usize {
        usize::from(self.fusion_channels(left, right).contains(&coupled))
    }
}

pub trait MultiplicityFreeFusionRule: FusionRule {}

pub trait MultiplicityFreeFusionSymbols: MultiplicityFreeFusionRule {
    // Send + Sync because cached recoupling coefficients are shared across
    // tree-transform replay workers (TensorKit sectorscalartype parity: the
    // concrete scalar is a plain number type).
    type Scalar: Clone + Send + Sync;

    fn scalar_one(&self) -> Self::Scalar;

    fn scalar_conj(&self, value: Self::Scalar) -> Self::Scalar;

    fn f_symbol_scalar(
        &self,
        left: SectorId,
        middle: SectorId,
        right: SectorId,
        coupled: SectorId,
        left_coupled: SectorId,
        right_coupled: SectorId,
    ) -> Self::Scalar;

    fn r_symbol_scalar(&self, left: SectorId, right: SectorId, coupled: SectorId) -> Self::Scalar;
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
        source: &FusionTreeBlockKey,
        destination: &FusionTreeBlockKey,
    ) -> Self::Scalar;
}

// `Sync` because the tree-transform plan compile computes recoupling rows
// for independent source trees in parallel, sharing the rule across workers
// (TensorKit threaded transformer construction parity: sector types are
// plain shareable data). All rule implementations are plain data.
pub trait MultiplicityFreeRigidSymbols: MultiplicityFreeFusionSymbols + Sync {
    fn dim_scalar(&self, sector: SectorId) -> Self::Scalar;

    fn inv_dim_scalar(&self, sector: SectorId) -> Self::Scalar;

    fn sqrt_dim_scalar(&self, sector: SectorId) -> Self::Scalar;

    fn inv_sqrt_dim_scalar(&self, sector: SectorId) -> Self::Scalar;

    fn twist_scalar(&self, sector: SectorId) -> Self::Scalar;

    fn frobenius_schur_phase_scalar(&self, sector: SectorId) -> Self::Scalar;

    fn a_symbol_scalar(&self, left: SectorId, right: SectorId, coupled: SectorId) -> Self::Scalar
    where
        Self::Scalar: Mul<Output = Self::Scalar>,
    {
        let factor = self.sqrt_dim_scalar(left)
            * self.sqrt_dim_scalar(right)
            * self.inv_sqrt_dim_scalar(coupled);
        let symbol = self.frobenius_schur_phase_scalar(left)
            * self.f_symbol_scalar(self.dual(left), left, right, right, self.vacuum(), coupled);
        factor * self.scalar_conj(symbol)
    }

    fn b_symbol_scalar(&self, left: SectorId, right: SectorId, coupled: SectorId) -> Self::Scalar
    where
        Self::Scalar: Mul<Output = Self::Scalar>,
    {
        self.sqrt_dim_scalar(left)
            * self.sqrt_dim_scalar(right)
            * self.inv_sqrt_dim_scalar(coupled)
            * self.f_symbol_scalar(left, right, self.dual(right), left, coupled, self.vacuum())
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct ProductSector<Left, Right> {
    left: Left,
    right: Right,
}

impl<Left, Right> ProductSector<Left, Right> {
    pub const fn new(left: Left, right: Right) -> Self {
        Self { left, right }
    }

    #[inline]
    pub const fn left(&self) -> &Left {
        &self.left
    }

    #[inline]
    pub const fn right(&self) -> &Right {
        &self.right
    }

    pub fn sector_id_with<C>(self) -> SectorId
    where
        C: ProductSectorCodec,
        Left: Into<SectorId>,
        Right: Into<SectorId>,
    {
        C::encode(self.left.into(), self.right.into())
    }
}

pub const fn product_sector<Left, Right>(left: Left, right: Right) -> ProductSector<Left, Right> {
    ProductSector::new(left, right)
}

pub trait ProductSectorCodec {
    fn try_encode(left: SectorId, right: SectorId) -> Option<SectorId>;

    fn encode(left: SectorId, right: SectorId) -> SectorId {
        Self::try_encode(left, right).expect("product sector id overflow")
    }

    fn decode(sector: SectorId) -> Option<(SectorId, SectorId)>;
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Hash)]
pub struct TensorKitProductCodec;

impl ProductSectorCodec for TensorKitProductCodec {
    fn try_encode(left: SectorId, right: SectorId) -> Option<SectorId> {
        let left = left.id() as u128;
        let right = right.id() as u128;
        let sum = left.checked_add(right)?;
        let paired = sum
            .checked_mul(sum + 1)
            .and_then(|value| value.checked_div(2))
            .and_then(|value| value.checked_add(left))
            .and_then(|value| usize::try_from(value).ok())?;
        Some(SectorId::new(paired))
    }

    fn decode(sector: SectorId) -> Option<(SectorId, SectorId)> {
        let paired = sector.id() as u128;
        let sum = tensor_kit_product_pairing_sum(paired);
        let triangular = sum.checked_mul(sum + 1)?.checked_div(2)?;
        let left = paired.checked_sub(triangular)?;
        let right = sum.checked_sub(left)?;
        Some((
            SectorId::new(usize::try_from(left).ok()?),
            SectorId::new(usize::try_from(right).ok()?),
        ))
    }
}

fn tensor_kit_product_pairing_sum(paired: u128) -> u128 {
    let mut low = 0u128;
    let mut high = 1u128;
    while triangular_number(high) <= paired {
        high *= 2;
    }
    while low + 1 < high {
        let mid = low + (high - low) / 2;
        if triangular_number(mid) <= paired {
            low = mid;
        } else {
            high = mid;
        }
    }
    low
}

fn triangular_number(value: u128) -> u128 {
    value * (value + 1) / 2
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct ProductFusionRule<LeftRule, RightRule, Codec = TensorKitProductCodec> {
    left: LeftRule,
    right: RightRule,
    _codec: PhantomData<Codec>,
}

impl<LeftRule, RightRule, Codec> ProductFusionRule<LeftRule, RightRule, Codec> {
    pub const fn new(left: LeftRule, right: RightRule) -> Self {
        Self {
            left,
            right,
            _codec: PhantomData,
        }
    }

    #[inline]
    pub const fn left_rule(&self) -> &LeftRule {
        &self.left
    }

    #[inline]
    pub const fn right_rule(&self) -> &RightRule {
        &self.right
    }

    pub fn encode_sector(&self, left: SectorId, right: SectorId) -> SectorId
    where
        Codec: ProductSectorCodec,
    {
        Codec::encode(left, right)
    }

    pub fn decode_sector(&self, sector: SectorId) -> Option<(SectorId, SectorId)>
    where
        Codec: ProductSectorCodec,
    {
        Codec::decode(sector)
    }

    fn decode_sector_or_panic(&self, sector: SectorId) -> (SectorId, SectorId)
    where
        Codec: ProductSectorCodec,
    {
        self.decode_sector(sector)
            .expect("product fusion rule received an invalid product sector")
    }
}

pub const fn product_fusion_rule<LeftRule, RightRule>(
    left: LeftRule,
    right: RightRule,
) -> ProductFusionRule<LeftRule, RightRule> {
    ProductFusionRule::new(left, right)
}

pub const fn product_fusion_rule_with_codec<LeftRule, RightRule, Codec>(
    left: LeftRule,
    right: RightRule,
) -> ProductFusionRule<LeftRule, RightRule, Codec> {
    ProductFusionRule::new(left, right)
}

pub trait ProductFusionRuleExt: FusionRule + Sized {
    fn product<RightRule>(self, right: RightRule) -> ProductFusionRule<Self, RightRule>
    where
        RightRule: FusionRule,
    {
        ProductFusionRule::new(self, right)
    }
}

impl<Rule> ProductFusionRuleExt for Rule where Rule: FusionRule + Sized {}

impl<LeftRule, RightRule, Codec> Default for ProductFusionRule<LeftRule, RightRule, Codec>
where
    LeftRule: Default,
    RightRule: Default,
{
    fn default() -> Self {
        Self::new(LeftRule::default(), RightRule::default())
    }
}

impl<LeftRule, RightRule, Codec> FusionRule for ProductFusionRule<LeftRule, RightRule, Codec>
where
    LeftRule: FusionRule,
    RightRule: FusionRule,
    Codec: ProductSectorCodec,
{
    fn fusion_style(&self) -> FusionStyleKind {
        self.left
            .fusion_style()
            .combined_with(self.right.fusion_style())
    }

    fn braiding_style(&self) -> BraidingStyleKind {
        self.left
            .braiding_style()
            .combined_with(self.right.braiding_style())
    }

    fn vacuum(&self) -> SectorId {
        self.encode_sector(self.left.vacuum(), self.right.vacuum())
    }

    fn supports_unitary_braid_dagger(&self) -> bool {
        self.left.supports_unitary_braid_dagger() && self.right.supports_unitary_braid_dagger()
    }

    fn dual(&self, sector: SectorId) -> SectorId {
        let (left, right) = self.decode_sector_or_panic(sector);
        self.encode_sector(self.left.dual(left), self.right.dual(right))
    }

    fn fusion_channels(&self, left: SectorId, right: SectorId) -> SectorVec {
        let (left_left, left_right) = self.decode_sector_or_panic(left);
        let (right_left, right_right) = self.decode_sector_or_panic(right);
        let left_channels = self.left.fusion_channels(left_left, right_left);
        let right_channels = self.right.fusion_channels(left_right, right_right);
        // Cartesian product of the two sub-rules' channels, matching TensorKit's
        // `⊗(p1,p2) = SectorSet(product(map(⊗, ...)))`. No dedup: each sub-rule
        // is multiplicity-free (distinct channels) and `encode_sector` is the
        // Cantor pairing (a bijection), so distinct (left,right) pairs always
        // encode to distinct ids — the old `channels.contains()` guard was
        // provably dead and made this O(k²) instead of O(k) in k = |L|·|R|.
        let mut channels = SectorVec::with_capacity(left_channels.len() * right_channels.len());
        for right_channel in right_channels {
            for &left_channel in &left_channels {
                channels.push(self.encode_sector(left_channel, right_channel));
            }
        }
        channels
    }

    fn nsymbol(&self, left: SectorId, right: SectorId, coupled: SectorId) -> usize {
        let (left_left, left_right) = self.decode_sector_or_panic(left);
        let (right_left, right_right) = self.decode_sector_or_panic(right);
        let (coupled_left, coupled_right) = self.decode_sector_or_panic(coupled);
        self.left.nsymbol(left_left, right_left, coupled_left)
            * self.right.nsymbol(left_right, right_right, coupled_right)
    }
}

impl<LeftRule, RightRule, Codec> MultiplicityFreeFusionRule
    for ProductFusionRule<LeftRule, RightRule, Codec>
where
    LeftRule: MultiplicityFreeFusionRule,
    RightRule: MultiplicityFreeFusionRule,
    Codec: ProductSectorCodec,
{
}

impl<LeftRule, RightRule, Codec> MultiplicityFreeFusionSymbols
    for ProductFusionRule<LeftRule, RightRule, Codec>
where
    LeftRule: MultiplicityFreeFusionSymbols<Scalar = f64>,
    RightRule: MultiplicityFreeFusionSymbols<Scalar = f64>,
    Codec: ProductSectorCodec,
{
    type Scalar = f64;

    fn scalar_one(&self) -> Self::Scalar {
        1.0
    }

    fn scalar_conj(&self, value: Self::Scalar) -> Self::Scalar {
        value
    }

    fn f_symbol_scalar(
        &self,
        left: SectorId,
        middle: SectorId,
        right: SectorId,
        coupled: SectorId,
        left_coupled: SectorId,
        right_coupled: SectorId,
    ) -> Self::Scalar {
        let (left_l, left_r) = self.decode_sector_or_panic(left);
        let (middle_l, middle_r) = self.decode_sector_or_panic(middle);
        let (right_l, right_r) = self.decode_sector_or_panic(right);
        let (coupled_l, coupled_r) = self.decode_sector_or_panic(coupled);
        let (left_coupled_l, left_coupled_r) = self.decode_sector_or_panic(left_coupled);
        let (right_coupled_l, right_coupled_r) = self.decode_sector_or_panic(right_coupled);
        self.left.f_symbol_scalar(
            left_l,
            middle_l,
            right_l,
            coupled_l,
            left_coupled_l,
            right_coupled_l,
        ) * self.right.f_symbol_scalar(
            left_r,
            middle_r,
            right_r,
            coupled_r,
            left_coupled_r,
            right_coupled_r,
        )
    }

    fn r_symbol_scalar(&self, left: SectorId, right: SectorId, coupled: SectorId) -> Self::Scalar {
        let (left_l, left_r) = self.decode_sector_or_panic(left);
        let (right_l, right_r) = self.decode_sector_or_panic(right);
        let (coupled_l, coupled_r) = self.decode_sector_or_panic(coupled);
        self.left.r_symbol_scalar(left_l, right_l, coupled_l)
            * self.right.r_symbol_scalar(left_r, right_r, coupled_r)
    }
}

impl<LeftRule, RightRule, Codec> MultiplicityFreeRigidSymbols
    for ProductFusionRule<LeftRule, RightRule, Codec>
where
    LeftRule: MultiplicityFreeRigidSymbols<Scalar = f64>,
    RightRule: MultiplicityFreeRigidSymbols<Scalar = f64>,
    // Sync via the trait's supertrait; the codec is a PhantomData marker.
    Codec: ProductSectorCodec + Sync,
{
    fn dim_scalar(&self, sector: SectorId) -> Self::Scalar {
        let (left, right) = self.decode_sector_or_panic(sector);
        self.left.dim_scalar(left) * self.right.dim_scalar(right)
    }

    fn inv_dim_scalar(&self, sector: SectorId) -> Self::Scalar {
        let (left, right) = self.decode_sector_or_panic(sector);
        self.left.inv_dim_scalar(left) * self.right.inv_dim_scalar(right)
    }

    fn sqrt_dim_scalar(&self, sector: SectorId) -> Self::Scalar {
        let (left, right) = self.decode_sector_or_panic(sector);
        self.left.sqrt_dim_scalar(left) * self.right.sqrt_dim_scalar(right)
    }

    fn inv_sqrt_dim_scalar(&self, sector: SectorId) -> Self::Scalar {
        let (left, right) = self.decode_sector_or_panic(sector);
        self.left.inv_sqrt_dim_scalar(left) * self.right.inv_sqrt_dim_scalar(right)
    }

    fn twist_scalar(&self, sector: SectorId) -> Self::Scalar {
        let (left, right) = self.decode_sector_or_panic(sector);
        self.left.twist_scalar(left) * self.right.twist_scalar(right)
    }

    fn frobenius_schur_phase_scalar(&self, sector: SectorId) -> Self::Scalar {
        let (left, right) = self.decode_sector_or_panic(sector);
        self.left.frobenius_schur_phase_scalar(left)
            * self.right.frobenius_schur_phase_scalar(right)
    }

    fn a_symbol_scalar(&self, left: SectorId, right: SectorId, coupled: SectorId) -> Self::Scalar {
        let (left_l, left_r) = self.decode_sector_or_panic(left);
        let (right_l, right_r) = self.decode_sector_or_panic(right);
        let (coupled_l, coupled_r) = self.decode_sector_or_panic(coupled);
        self.left.a_symbol_scalar(left_l, right_l, coupled_l)
            * self.right.a_symbol_scalar(left_r, right_r, coupled_r)
    }

    fn b_symbol_scalar(&self, left: SectorId, right: SectorId, coupled: SectorId) -> Self::Scalar {
        let (left_l, left_r) = self.decode_sector_or_panic(left);
        let (right_l, right_r) = self.decode_sector_or_panic(right);
        let (coupled_l, coupled_r) = self.decode_sector_or_panic(coupled);
        self.left.b_symbol_scalar(left_l, right_l, coupled_l)
            * self.right.b_symbol_scalar(left_r, right_r, coupled_r)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct Z2Irrep {
    parity: u8,
}

impl Z2Irrep {
    pub const EVEN: Self = Self { parity: 0 };
    pub const ODD: Self = Self { parity: 1 };

    pub const fn new(parity: u8) -> Self {
        Self { parity: parity & 1 }
    }

    #[inline]
    pub const fn parity(self) -> u8 {
        self.parity
    }

    #[inline]
    pub const fn sector_id(self) -> SectorId {
        SectorId::new(self.parity as usize)
    }

    pub const fn from_sector_id(sector: SectorId) -> Option<Self> {
        match sector.id() {
            0 => Some(Self::EVEN),
            1 => Some(Self::ODD),
            _ => None,
        }
    }
}

impl From<Z2Irrep> for SectorId {
    fn from(value: Z2Irrep) -> Self {
        value.sector_id()
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Hash)]
pub struct Z2FusionRule;

impl FusionRule for Z2FusionRule {
    fn fusion_style(&self) -> FusionStyleKind {
        FusionStyleKind::Unique
    }

    fn braiding_style(&self) -> BraidingStyleKind {
        BraidingStyleKind::Bosonic
    }

    fn vacuum(&self) -> SectorId {
        Z2Irrep::EVEN.into()
    }

    fn supports_unitary_braid_dagger(&self) -> bool {
        true
    }

    fn fusion_channels(&self, left: SectorId, right: SectorId) -> SectorVec {
        let left = Z2Irrep::from_sector_id(left).expect("Z2 fusion received an invalid sector");
        let right = Z2Irrep::from_sector_id(right).expect("Z2 fusion received an invalid sector");
        core::iter::once(Z2Irrep::new(left.parity() ^ right.parity()).into()).collect()
    }
}

impl MultiplicityFreeFusionRule for Z2FusionRule {}

impl MultiplicityFreeFusionSymbols for Z2FusionRule {
    type Scalar = f64;

    fn scalar_one(&self) -> Self::Scalar {
        1.0
    }

    fn scalar_conj(&self, value: Self::Scalar) -> Self::Scalar {
        value
    }

    fn f_symbol_scalar(
        &self,
        _left: SectorId,
        _middle: SectorId,
        _right: SectorId,
        _coupled: SectorId,
        _left_coupled: SectorId,
        _right_coupled: SectorId,
    ) -> Self::Scalar {
        1.0
    }

    fn r_symbol_scalar(
        &self,
        _left: SectorId,
        _right: SectorId,
        _coupled: SectorId,
    ) -> Self::Scalar {
        1.0
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
        _source: &FusionTreeBlockKey,
        _destination: &FusionTreeBlockKey,
    ) -> Self::Scalar {
        1.0
    }
}

impl MultiplicityFreeRigidSymbols for Z2FusionRule {
    fn dim_scalar(&self, _sector: SectorId) -> Self::Scalar {
        1.0
    }

    fn inv_dim_scalar(&self, _sector: SectorId) -> Self::Scalar {
        1.0
    }

    fn sqrt_dim_scalar(&self, _sector: SectorId) -> Self::Scalar {
        1.0
    }

    fn inv_sqrt_dim_scalar(&self, _sector: SectorId) -> Self::Scalar {
        1.0
    }

    fn twist_scalar(&self, _sector: SectorId) -> Self::Scalar {
        1.0
    }

    fn frobenius_schur_phase_scalar(&self, _sector: SectorId) -> Self::Scalar {
        1.0
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Hash)]
pub struct FermionParityFusionRule;

impl FusionRule for FermionParityFusionRule {
    fn fusion_style(&self) -> FusionStyleKind {
        FusionStyleKind::Unique
    }

    fn braiding_style(&self) -> BraidingStyleKind {
        BraidingStyleKind::Fermionic
    }

    fn vacuum(&self) -> SectorId {
        Z2Irrep::EVEN.into()
    }

    fn supports_unitary_braid_dagger(&self) -> bool {
        true
    }

    fn fusion_channels(&self, left: SectorId, right: SectorId) -> SectorVec {
        Z2FusionRule.fusion_channels(left, right)
    }
}

impl MultiplicityFreeFusionRule for FermionParityFusionRule {}

impl MultiplicityFreeFusionSymbols for FermionParityFusionRule {
    type Scalar = f64;

    fn scalar_one(&self) -> Self::Scalar {
        1.0
    }

    fn scalar_conj(&self, value: Self::Scalar) -> Self::Scalar {
        value
    }

    fn f_symbol_scalar(
        &self,
        _left: SectorId,
        _middle: SectorId,
        _right: SectorId,
        _coupled: SectorId,
        _left_coupled: SectorId,
        _right_coupled: SectorId,
    ) -> Self::Scalar {
        1.0
    }

    fn r_symbol_scalar(&self, left: SectorId, right: SectorId, _coupled: SectorId) -> Self::Scalar {
        if left == Z2Irrep::ODD.into() && right == Z2Irrep::ODD.into() {
            -1.0
        } else {
            1.0
        }
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
        _source: &FusionTreeBlockKey,
        _destination: &FusionTreeBlockKey,
    ) -> Self::Scalar {
        1.0
    }
}

impl MultiplicityFreeRigidSymbols for FermionParityFusionRule {
    fn dim_scalar(&self, _sector: SectorId) -> Self::Scalar {
        1.0
    }

    fn inv_dim_scalar(&self, _sector: SectorId) -> Self::Scalar {
        1.0
    }

    fn sqrt_dim_scalar(&self, _sector: SectorId) -> Self::Scalar {
        1.0
    }

    fn inv_sqrt_dim_scalar(&self, _sector: SectorId) -> Self::Scalar {
        1.0
    }

    fn twist_scalar(&self, sector: SectorId) -> Self::Scalar {
        if sector == Z2Irrep::ODD.into() {
            -1.0
        } else {
            1.0
        }
    }

    fn frobenius_schur_phase_scalar(&self, _sector: SectorId) -> Self::Scalar {
        1.0
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct U1Irrep {
    charge: i32,
}

impl U1Irrep {
    pub const fn new(charge: i32) -> Self {
        Self { charge }
    }

    #[inline]
    pub const fn charge(self) -> i32 {
        self.charge
    }

    pub const fn sector_id(self) -> SectorId {
        let charge = self.charge as i64;
        if charge >= 0 {
            SectorId::new((charge as usize) * 2)
        } else {
            SectorId::new(((-charge) as usize) * 2 - 1)
        }
    }

    pub fn from_sector_id(sector: SectorId) -> Option<Self> {
        let id = sector.id();
        if id > u32::MAX as usize {
            return None;
        }
        let charge = if id % 2 == 0 {
            i64::try_from(id / 2).ok()?
        } else {
            -i64::try_from((id + 1) / 2).ok()?
        };
        i32::try_from(charge).ok().map(Self::new)
    }
}

impl From<U1Irrep> for SectorId {
    fn from(value: U1Irrep) -> Self {
        value.sector_id()
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Hash)]
pub struct U1FusionRule;

impl FusionRule for U1FusionRule {
    fn fusion_style(&self) -> FusionStyleKind {
        FusionStyleKind::Unique
    }

    fn braiding_style(&self) -> BraidingStyleKind {
        BraidingStyleKind::Bosonic
    }

    fn vacuum(&self) -> SectorId {
        U1Irrep::new(0).into()
    }

    fn supports_unitary_braid_dagger(&self) -> bool {
        true
    }

    fn dual(&self, sector: SectorId) -> SectorId {
        let sector = U1Irrep::from_sector_id(sector).expect("U(1) dual received an invalid sector");
        U1Irrep::new(
            sector
                .charge()
                .checked_neg()
                .expect("U(1) dual charge overflow"),
        )
        .into()
    }

    fn fusion_channels(&self, left: SectorId, right: SectorId) -> SectorVec {
        let left = U1Irrep::from_sector_id(left).expect("U(1) fusion received an invalid sector");
        let right = U1Irrep::from_sector_id(right).expect("U(1) fusion received an invalid sector");
        core::iter::once(
            U1Irrep::new(
                left.charge()
                    .checked_add(right.charge())
                    .expect("U(1) fusion charge overflow"),
            )
            .into(),
        )
        .collect()
    }
}

impl MultiplicityFreeFusionRule for U1FusionRule {}

impl MultiplicityFreeFusionSymbols for U1FusionRule {
    type Scalar = f64;

    fn scalar_one(&self) -> Self::Scalar {
        1.0
    }

    fn scalar_conj(&self, value: Self::Scalar) -> Self::Scalar {
        value
    }

    fn f_symbol_scalar(
        &self,
        _left: SectorId,
        _middle: SectorId,
        _right: SectorId,
        _coupled: SectorId,
        _left_coupled: SectorId,
        _right_coupled: SectorId,
    ) -> Self::Scalar {
        1.0
    }

    fn r_symbol_scalar(
        &self,
        _left: SectorId,
        _right: SectorId,
        _coupled: SectorId,
    ) -> Self::Scalar {
        1.0
    }
}

impl MultiplicityFreeRigidSymbols for U1FusionRule {
    fn dim_scalar(&self, _sector: SectorId) -> Self::Scalar {
        1.0
    }

    fn inv_dim_scalar(&self, _sector: SectorId) -> Self::Scalar {
        1.0
    }

    fn sqrt_dim_scalar(&self, _sector: SectorId) -> Self::Scalar {
        1.0
    }

    fn inv_sqrt_dim_scalar(&self, _sector: SectorId) -> Self::Scalar {
        1.0
    }

    fn twist_scalar(&self, _sector: SectorId) -> Self::Scalar {
        1.0
    }

    fn frobenius_schur_phase_scalar(&self, _sector: SectorId) -> Self::Scalar {
        1.0
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct SU2Irrep {
    twice_spin: usize,
}

impl SU2Irrep {
    pub const fn from_twice_spin(twice_spin: usize) -> Self {
        Self { twice_spin }
    }

    #[inline]
    pub const fn twice_spin(self) -> usize {
        self.twice_spin
    }

    #[inline]
    pub const fn sector_id(self) -> SectorId {
        SectorId::new(self.twice_spin)
    }

    pub const fn from_sector_id(sector: SectorId) -> Self {
        Self::from_twice_spin(sector.id())
    }
}

impl From<SU2Irrep> for SectorId {
    fn from(value: SU2Irrep) -> Self {
        value.sector_id()
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Hash)]
pub struct SU2FusionRule;

impl FusionRule for SU2FusionRule {
    fn fusion_style(&self) -> FusionStyleKind {
        FusionStyleKind::Simple
    }

    fn braiding_style(&self) -> BraidingStyleKind {
        BraidingStyleKind::Bosonic
    }

    fn vacuum(&self) -> SectorId {
        SU2Irrep::from_twice_spin(0).into()
    }

    fn supports_unitary_braid_dagger(&self) -> bool {
        true
    }

    fn fusion_channels(&self, left: SectorId, right: SectorId) -> SectorVec {
        let left = SU2Irrep::from_sector_id(left).twice_spin();
        let right = SU2Irrep::from_sector_id(right).twice_spin();
        let min = left.abs_diff(right);
        let max = left + right;
        (min..=max)
            .step_by(2)
            .map(|twice_spin| SU2Irrep::from_twice_spin(twice_spin).into())
            .collect()
    }
}

impl MultiplicityFreeFusionRule for SU2FusionRule {}

impl MultiplicityFreeFusionSymbols for SU2FusionRule {
    type Scalar = f64;

    fn scalar_one(&self) -> Self::Scalar {
        1.0
    }

    fn scalar_conj(&self, value: Self::Scalar) -> Self::Scalar {
        value
    }

    fn f_symbol_scalar(
        &self,
        left: SectorId,
        middle: SectorId,
        right: SectorId,
        coupled: SectorId,
        left_coupled: SectorId,
        right_coupled: SectorId,
    ) -> Self::Scalar {
        let j1 = SU2Irrep::from_sector_id(left).twice_spin();
        let j2 = SU2Irrep::from_sector_id(middle).twice_spin();
        let j3 = SU2Irrep::from_sector_id(right).twice_spin();
        let j4 = SU2Irrep::from_sector_id(coupled).twice_spin();
        let j5 = SU2Irrep::from_sector_id(left_coupled).twice_spin();
        let j6 = SU2Irrep::from_sector_id(right_coupled).twice_spin();
        su2_f_symbol_from_doubled_spins(j1, j2, j3, j4, j5, j6)
    }

    fn r_symbol_scalar(&self, left: SectorId, right: SectorId, coupled: SectorId) -> Self::Scalar {
        if self.nsymbol(left, right, coupled) == 0 {
            return 0.0;
        }
        let left = SU2Irrep::from_sector_id(left).twice_spin();
        let right = SU2Irrep::from_sector_id(right).twice_spin();
        let coupled = SU2Irrep::from_sector_id(coupled).twice_spin();
        let exponent = (left + right - coupled) / 2;
        if exponent % 2 == 0 {
            1.0
        } else {
            -1.0
        }
    }
}

impl MultiplicityFreeRigidSymbols for SU2FusionRule {
    fn dim_scalar(&self, sector: SectorId) -> Self::Scalar {
        (SU2Irrep::from_sector_id(sector).twice_spin() + 1) as f64
    }

    fn inv_dim_scalar(&self, sector: SectorId) -> Self::Scalar {
        1.0 / self.dim_scalar(sector)
    }

    fn sqrt_dim_scalar(&self, sector: SectorId) -> Self::Scalar {
        self.dim_scalar(sector).sqrt()
    }

    fn inv_sqrt_dim_scalar(&self, sector: SectorId) -> Self::Scalar {
        1.0 / self.sqrt_dim_scalar(sector)
    }

    fn twist_scalar(&self, _sector: SectorId) -> Self::Scalar {
        1.0
    }

    fn frobenius_schur_phase_scalar(&self, sector: SectorId) -> Self::Scalar {
        if SU2Irrep::from_sector_id(sector).twice_spin() % 2 == 0 {
            1.0
        } else {
            -1.0
        }
    }
}

fn su2_f_symbol_from_doubled_spins(
    j1: usize,
    j2: usize,
    j3: usize,
    j4: usize,
    j5: usize,
    j6: usize,
) -> f64 {
    if [j1, j2, j3, j4, j5, j6].iter().all(|&j| j == 0) {
        return 1.0;
    }
    let phase_exponent = (j1 + j2 + j3 + j4) / 2;
    let phase = if phase_exponent % 2 == 0 { 1.0 } else { -1.0 };
    let dimension_factor = (((j5 + 1) * (j6 + 1)) as f64).sqrt();
    phase * dimension_factor * wigner_6j_doubled(j1, j2, j5, j3, j4, j6)
}

/// SU(2) 6j symbol (arguments as doubled spins), memoized in a process-global
/// cache keyed by the six doubled spins — the analogue of TensorKit's
/// `WignerSymbols.Wigner6j` LRU. Each distinct symbol's exact summation is
/// evaluated once; every later occurrence (across braids, permutes, and
/// contractions) is a hash lookup. The cached value is bit-identical to the
/// direct computation, so this changes cold compile cost only.
fn wigner_6j_doubled(j1: usize, j2: usize, j3: usize, j4: usize, j5: usize, j6: usize) -> f64 {
    static CACHE: std::sync::OnceLock<std::sync::RwLock<FxHashMap<[usize; 6], f64>>> =
        std::sync::OnceLock::new();
    let cache = CACHE.get_or_init(|| std::sync::RwLock::new(FxHashMap::default()));
    let key = [j1, j2, j3, j4, j5, j6];
    if let Ok(read) = cache.read() {
        if let Some(&value) = read.get(&key) {
            return value;
        }
    }
    let value = wigner_6j_doubled_uncached(j1, j2, j3, j4, j5, j6);
    if let Ok(mut write) = cache.write() {
        write.insert(key, value);
    }
    value
}

fn wigner_6j_doubled_uncached(
    j1: usize,
    j2: usize,
    j3: usize,
    j4: usize,
    j5: usize,
    j6: usize,
) -> f64 {
    let Some(delta_ln) = su2_delta_ln(j1, j2, j3)
        .and_then(|value| su2_delta_ln(j1, j5, j6).map(|next| value + next))
        .and_then(|value| su2_delta_ln(j4, j2, j6).map(|next| value + next))
        .and_then(|value| su2_delta_ln(j4, j5, j3).map(|next| value + next))
    else {
        return 0.0;
    };

    let x = [j1 + j2 + j3, j1 + j5 + j6, j4 + j2 + j6, j4 + j5 + j3];
    let y = [j1 + j2 + j4 + j5, j1 + j3 + j4 + j6, j2 + j3 + j5 + j6];
    let mut z_min = x.into_iter().max().unwrap_or(0);
    let z_max = y.into_iter().min().unwrap_or(0);
    if z_min > z_max {
        return 0.0;
    }
    if z_min % 2 != 0 {
        z_min += 1;
    }

    let mut sum = 0.0;
    let mut z_doubled = z_min;
    while z_doubled <= z_max {
        let z = z_doubled / 2;
        let mut term_ln = ln_factorial(z + 1);
        for value in x {
            term_ln -= ln_factorial((z_doubled - value) / 2);
        }
        for value in y {
            term_ln -= ln_factorial((value - z_doubled) / 2);
        }
        let sign = if z % 2 == 0 { 1.0 } else { -1.0 };
        sum += sign * term_ln.exp();
        z_doubled += 2;
    }

    delta_ln.exp() * sum
}

fn su2_delta_ln(j1: usize, j2: usize, j3: usize) -> Option<f64> {
    if (j1 + j2 + j3) % 2 != 0 {
        return None;
    }
    if j1 + j2 < j3 || j1 + j3 < j2 || j2 + j3 < j1 {
        return None;
    }
    Some(
        0.5 * (ln_factorial((j1 + j2 - j3) / 2)
            + ln_factorial((j1 + j3 - j2) / 2)
            + ln_factorial((j2 + j3 - j1) / 2)
            - ln_factorial((j1 + j2 + j3) / 2 + 1)),
    )
}

/// `ln(n!)`, memoized in a process-global lazily-extended table.
///
/// `ln(n!) = ln((n-1)!) + ln(n)` is monotone, so the table is filled once and
/// every later call is an O(1) lookup. Recoupling-coefficient evaluation
/// (6j symbols) calls this ~7x per summation term, so cold structure compile
/// dominated by the previous naive `(1..=n)` recomputation. The values are
/// identical; this only removes the recomputation.
fn ln_factorial(n: usize) -> f64 {
    static TABLE: std::sync::OnceLock<std::sync::RwLock<Vec<f64>>> = std::sync::OnceLock::new();
    let table = TABLE.get_or_init(|| std::sync::RwLock::new(vec![0.0]));
    if let Ok(read) = table.read() {
        if let Some(&value) = read.get(n) {
            return value;
        }
    }
    let mut write = table.write().expect("ln_factorial table poisoned");
    while write.len() <= n {
        let previous = *write.last().expect("table seeded with ln(0!) = 0");
        let next = write.len();
        write.push(previous + (next as f64).ln());
    }
    write[n]
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct CoupledFusionTrees {
    coupled: SectorId,
    trees: Vec<FusionTreeKey>,
}

fn collect_selected_leg_tuples(
    legs: &[SectorLeg],
    remaining: usize,
    current: &mut [Option<FusionTreeLeg>],
    tuples: &mut Vec<Vec<FusionTreeLeg>>,
) {
    if remaining == 0 {
        tuples.push(
            current
                .iter()
                .map(|leg| leg.expect("fusion tree leg tuple should be fully assigned"))
                .collect(),
        );
        return;
    }

    let index = remaining - 1;
    for &sector in legs[index].sectors() {
        current[index] = Some(FusionTreeLeg::new(sector, legs[index].is_dual()));
        collect_selected_leg_tuples(legs, remaining - 1, current, tuples);
    }
}

fn fusion_trees_by_coupled_for_space<R>(
    rule: &R,
    space: &FusionProductSpace,
) -> Vec<CoupledFusionTrees>
where
    R: MultiplicityFreeFusionRule,
{
    // Group trees by coupled sector via a `coupled -> index` map so the merge
    // is O(1) per (tuple, coupled) pair. The previous `grouped.iter_mut().find`
    // linear scan was O(P·C) (P = tuple×coupled iterations, C = distinct
    // coupled sectors); the map removes the C factor. The final `sort_by_key`
    // still fixes the canonical order, so the map need not preserve it.
    let mut grouped = Vec::<CoupledFusionTrees>::new();
    let mut index: FxHashMap<SectorId, usize> = FxHashMap::default();
    for tuple in space.selected_leg_tuples() {
        let effective = effective_sectors(rule, &tuple);
        let uncoupled: Vec<SectorId> = tuple.iter().map(|leg| leg.sector()).collect();
        let is_dual: Vec<bool> = tuple.iter().map(|leg| leg.is_dual()).collect();
        for coupled in reachable_coupled_sectors(rule, &effective) {
            let trees =
                collect_fusion_trees_for_coupled(rule, &uncoupled, &is_dual, &effective, coupled);
            match index.get(&coupled) {
                Some(&i) => grouped[i].trees.extend(trees),
                None => {
                    index.insert(coupled, grouped.len());
                    grouped.push(CoupledFusionTrees { coupled, trees });
                }
            }
        }
    }
    grouped.sort_by_key(|group| group.coupled);
    grouped
}

fn fusion_trees_by_coupled_for_selected_space<R>(
    rule: &R,
    space: &FusionProductSpace,
    selected: &[SectorId],
) -> Result<Vec<CoupledFusionTrees>, CoreError>
where
    R: MultiplicityFreeFusionRule,
{
    if selected.len() != space.len() {
        return Err(CoreError::DimensionMismatch {
            expected: space.len(),
            actual: selected.len(),
        });
    }
    for (&sector, leg) in selected.iter().zip(space.legs()) {
        // Sectors are stored sorted (SortedVectorDict invariant); binary-search
        // to stay consistent with `SectorLeg::degeneracy`.
        if leg.sectors().binary_search(&sector).is_err() {
            return Err(CoreError::InvalidSector { sector });
        }
    }

    let legs = selected
        .iter()
        .zip(space.legs())
        .map(|(&sector, leg)| FusionTreeLeg::new(sector, leg.is_dual()))
        .collect::<Vec<_>>();
    let effective = effective_sectors(rule, &legs);
    let uncoupled: Vec<SectorId> = legs.iter().map(|leg| leg.sector()).collect();
    let is_dual: Vec<bool> = legs.iter().map(|leg| leg.is_dual()).collect();
    let mut grouped = Vec::new();
    for coupled in reachable_coupled_sectors(rule, &effective) {
        let trees =
            collect_fusion_trees_for_coupled(rule, &uncoupled, &is_dual, &effective, coupled);
        if !trees.is_empty() {
            grouped.push(CoupledFusionTrees { coupled, trees });
        }
    }
    grouped.sort_by_key(|group| group.coupled);
    Ok(grouped)
}

/// Coupled sectors reachable by fusing all legs — TensorKit's `blocksectors`.
/// Computed once per leg tuple (not per enumeration node): the forward fold
/// `⊗` over the legs with dedup. Used only to drive the per-coupled grouping;
/// the tree enumeration itself does not consult it (see below).
fn reachable_coupled_sectors<R>(rule: &R, effective: &[SectorId]) -> Vec<SectorId>
where
    R: MultiplicityFreeFusionRule,
{
    let mut acc: Vec<SectorId> = match effective.first() {
        None => vec![rule.vacuum()],
        Some(&first) => vec![first],
    };
    for &last in effective.iter().skip(1) {
        acc = acc
            .iter()
            .flat_map(|&front| rule.fusion_channels(front, last))
            .collect();
        acc.sort_unstable();
        acc.dedup();
    }
    acc.sort_unstable();
    acc.dedup();
    acc
}

/// Enumerate the fusion trees of `uncoupled` (with `is_dual`) into `coupled`,
/// ported from TensorKit's `_fusiontree_iterate` (fusiontrees/iterator.jl).
/// It walks the inner lines *backward* from `coupled`: peel the last leg `b`,
/// let the adjacent inner line `a` range over `coupled ⊗ dual(b)`, recurse on
/// the front legs fusing to `a`, and prune dead branches by the recursion
/// yielding nothing — no forward `possible_coupled` reachability set, matching
/// TensorKit. Like TensorKit's *lazy* iterator it never materializes an
/// intermediate tree list per recursion level: a single `visit` walk pushes
/// each completed key straight into `out`, threading one reused inner-line
/// stack. Multiplicity-free, so every vertex is the trivial label.
fn collect_fusion_trees_for_coupled<R>(
    rule: &R,
    uncoupled: &[SectorId],
    is_dual: &[bool],
    effective: &[SectorId],
    coupled: SectorId,
) -> Vec<FusionTreeKey>
where
    R: MultiplicityFreeFusionRule,
{
    // Vertices are the trivial label for every multiplicity-free tree; a tree of
    // `n` legs has `n - 1` vertices (and `n - 2` inner lines), or none for n < 2.
    let vertex_count = uncoupled.len().saturating_sub(1);
    let mut out = Vec::new();
    // `inner_rev` accumulates the inner lines outermost-first as the walk
    // descends; the stored key wants innermost-first, so emit reverses it.
    let mut inner_rev: Vec<SectorId> = Vec::new();
    visit_fusion_trees(rule, effective, coupled, &mut inner_rev, &mut |inner_rev| {
        out.push(FusionTreeKey::new(
            uncoupled.iter().copied(),
            Some(coupled),
            is_dual.iter().copied(),
            inner_rev.iter().rev().copied(),
            std::iter::repeat(SectorId::new(1)).take(vertex_count),
        ));
    });
    out
}

fn visit_fusion_trees<R, F>(
    rule: &R,
    effective: &[SectorId],
    coupled: SectorId,
    inner_rev: &mut Vec<SectorId>,
    emit: &mut F,
) where
    R: MultiplicityFreeFusionRule,
    F: FnMut(&[SectorId]),
{
    match effective.len() {
        0 => {
            if coupled == rule.vacuum() {
                emit(inner_rev);
            }
        }
        1 => {
            if effective[0] == coupled {
                emit(inner_rev);
            }
        }
        2 => {
            if rule.nsymbol(effective[0], effective[1], coupled) != 0 {
                emit(inner_rev);
            }
        }
        _ => {
            let last = effective[effective.len() - 1];
            let front_effective = &effective[..effective.len() - 1];
            // Inner line `a` ranges over `coupled ⊗ dual(last)` (TensorKit's
            // `vertexiterN = coupled ⊗ dual(b)`); `Nsymbol(a, last, coupled)` is
            // the last vertex. No forward-reachability filter — dead `a` simply
            // emit nothing from the recursion.
            for front_coupled in rule.fusion_channels(coupled, rule.dual(last)) {
                if rule.nsymbol(front_coupled, last, coupled) == 0 {
                    continue;
                }
                inner_rev.push(front_coupled);
                visit_fusion_trees(rule, front_effective, front_coupled, inner_rev, emit);
                inner_rev.pop();
            }
        }
    }
}

fn effective_sectors<R>(_rule: &R, legs: &[FusionTreeLeg]) -> Vec<SectorId>
where
    R: MultiplicityFreeFusionRule,
{
    legs.iter().map(|leg| leg.sector()).collect()
}

#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct FusionTreeGroupKey {
    codomain_uncoupled: SectorVec,
    domain_uncoupled: SectorVec,
    codomain_is_dual: DualVec,
    domain_is_dual: DualVec,
}

impl FusionTreeGroupKey {
    pub fn new<Codomain, Domain, CodomainDual, DomainDual>(
        codomain_uncoupled: Codomain,
        domain_uncoupled: Domain,
        codomain_is_dual: CodomainDual,
        domain_is_dual: DomainDual,
    ) -> Self
    where
        Codomain: IntoIterator<Item = SectorId>,
        Domain: IntoIterator<Item = SectorId>,
        CodomainDual: IntoIterator<Item = bool>,
        DomainDual: IntoIterator<Item = bool>,
    {
        Self {
            codomain_uncoupled: codomain_uncoupled.into_iter().collect(),
            domain_uncoupled: domain_uncoupled.into_iter().collect(),
            codomain_is_dual: codomain_is_dual.into_iter().collect(),
            domain_is_dual: domain_is_dual.into_iter().collect(),
        }
    }

    pub fn from_sector_ids<Codomain, Domain, CodomainDual, DomainDual>(
        codomain_uncoupled: Codomain,
        domain_uncoupled: Domain,
        codomain_is_dual: CodomainDual,
        domain_is_dual: DomainDual,
    ) -> Self
    where
        Codomain: IntoIterator<Item = usize>,
        Domain: IntoIterator<Item = usize>,
        CodomainDual: IntoIterator<Item = bool>,
        DomainDual: IntoIterator<Item = bool>,
    {
        Self::new(
            codomain_uncoupled.into_iter().map(SectorId::new),
            domain_uncoupled.into_iter().map(SectorId::new),
            codomain_is_dual,
            domain_is_dual,
        )
    }

    #[inline]
    pub fn codomain_uncoupled(&self) -> &[SectorId] {
        &self.codomain_uncoupled
    }

    #[inline]
    pub fn domain_uncoupled(&self) -> &[SectorId] {
        &self.domain_uncoupled
    }

    #[inline]
    pub fn codomain_is_dual(&self) -> &[bool] {
        &self.codomain_is_dual
    }

    #[inline]
    pub fn domain_is_dual(&self) -> &[bool] {
        &self.domain_is_dual
    }
}

#[derive(Clone, Debug)]
pub struct FusionTreeKey {
    uncoupled: SectorVec,
    coupled: Option<SectorId>,
    is_dual: DualVec,
    innerlines: SectorVec,
    vertices: SectorVec,
}

// Identity of a `FusionTreeKey` is `(uncoupled, coupled, is_dual, innerlines)`
// — `vertices` is deliberately excluded from Hash/Eq/Ord. For multiplicity-free
// fusion (every rule in this crate) the vertex labels are functionally
// determined by those four fields (always the trivial vertex), so two keys that
// agree on them agree on `vertices` too: excluding it changes no equivalence
// class or ordering, only the per-op cost. FusionTreeKey comparison/hashing is
// the hottest logic in the cold recoupling-plan build; TensorKit likewise keys
// its `SimpleFusion` fusion trees on the sectors alone. All three impls use the
// SAME four fields so the Hash/Eq and Ord/Eq contracts hold.
impl std::hash::Hash for FusionTreeKey {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.uncoupled.hash(state);
        self.coupled.hash(state);
        self.is_dual.hash(state);
        self.innerlines.hash(state);
    }
}

impl PartialEq for FusionTreeKey {
    fn eq(&self, other: &Self) -> bool {
        self.uncoupled == other.uncoupled
            && self.coupled == other.coupled
            && self.is_dual == other.is_dual
            && self.innerlines == other.innerlines
    }
}

impl Eq for FusionTreeKey {}

impl Ord for FusionTreeKey {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.uncoupled
            .cmp(&other.uncoupled)
            .then_with(|| self.coupled.cmp(&other.coupled))
            .then_with(|| self.is_dual.cmp(&other.is_dual))
            .then_with(|| self.innerlines.cmp(&other.innerlines))
    }
}

impl PartialOrd for FusionTreeKey {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl FusionTreeKey {
    pub fn new<Uncoupled, Dual, Innerlines, Vertices>(
        uncoupled: Uncoupled,
        coupled: Option<SectorId>,
        is_dual: Dual,
        innerlines: Innerlines,
        vertices: Vertices,
    ) -> Self
    where
        Uncoupled: IntoIterator<Item = SectorId>,
        Dual: IntoIterator<Item = bool>,
        Innerlines: IntoIterator<Item = SectorId>,
        Vertices: IntoIterator<Item = SectorId>,
    {
        Self {
            uncoupled: uncoupled.into_iter().collect(),
            coupled,
            is_dual: is_dual.into_iter().collect(),
            innerlines: innerlines.into_iter().collect(),
            vertices: vertices.into_iter().collect(),
        }
    }

    pub fn from_sector_ids<Uncoupled, Dual, Innerlines, Vertices>(
        uncoupled: Uncoupled,
        coupled: Option<usize>,
        is_dual: Dual,
        innerlines: Innerlines,
        vertices: Vertices,
    ) -> Self
    where
        Uncoupled: IntoIterator<Item = usize>,
        Dual: IntoIterator<Item = bool>,
        Innerlines: IntoIterator<Item = usize>,
        Vertices: IntoIterator<Item = usize>,
    {
        Self::new(
            uncoupled.into_iter().map(SectorId::new),
            coupled.map(SectorId::new),
            is_dual,
            innerlines.into_iter().map(SectorId::new),
            vertices.into_iter().map(SectorId::new),
        )
    }

    pub fn from_uncoupled<I>(uncoupled: I) -> Self
    where
        I: IntoIterator<Item = SectorId>,
    {
        let uncoupled = uncoupled.into_iter().collect::<Vec<_>>();
        Self::new(
            uncoupled.clone(),
            None,
            vec![false; uncoupled.len()],
            Vec::new(),
            Vec::new(),
        )
    }

    #[inline]
    pub fn uncoupled(&self) -> &[SectorId] {
        &self.uncoupled
    }

    #[inline]
    pub fn coupled(&self) -> Option<SectorId> {
        self.coupled
    }

    #[inline]
    pub fn is_dual(&self) -> &[bool] {
        &self.is_dual
    }

    #[inline]
    pub fn innerlines(&self) -> &[SectorId] {
        &self.innerlines
    }

    #[inline]
    pub fn vertices(&self) -> &[SectorId] {
        &self.vertices
    }

    fn compact_id(&self) -> Option<usize> {
        if self.uncoupled.len() == 1
            && self.coupled.is_none()
            && self.innerlines.is_empty()
            && self.vertices.is_empty()
        {
            Some(self.uncoupled[0].id())
        } else {
            None
        }
    }
}

/// Split a left-associated fusion tree using TensorKit's `split(f, m)`
/// convention.
///
/// The first output contains the first `front_rank` uncoupled sectors. The
/// second output starts with the intermediate sector between the two pieces and
/// then contains the remaining uncoupled sectors. This is a structural
/// categorical operation: no dense storage is touched and no coefficient is
/// introduced.
pub fn split_fusion_tree<R>(
    rule: &R,
    tree: &FusionTreeKey,
    front_rank: usize,
) -> Result<(FusionTreeKey, FusionTreeKey), CoreError>
where
    R: FusionRule,
{
    let rank = tree.uncoupled().len();
    if front_rank > rank {
        return Err(CoreError::DimensionMismatch {
            expected: rank,
            actual: front_rank,
        });
    }
    validate_fusion_tree_key_shape(tree)?;

    if front_rank == rank {
        let coupled = coupled_or_vacuum(rule, tree);
        let trace_tree = FusionTreeKey::new(
            [coupled],
            Some(coupled),
            [false],
            Vec::<SectorId>::new(),
            Vec::<SectorId>::new(),
        );
        return Ok((tree.clone(), trace_tree));
    }

    if front_rank == 1 {
        let first = tree.uncoupled()[0];
        let front_tree = FusionTreeKey::new(
            [first],
            Some(first),
            [tree.is_dual()[0]],
            Vec::<SectorId>::new(),
            Vec::<SectorId>::new(),
        );
        let mut tail_is_dual = tree.is_dual().to_vec();
        tail_is_dual[0] = false;
        let tail_tree = FusionTreeKey::new(
            tree.uncoupled().to_vec(),
            tree.coupled(),
            tail_is_dual,
            tree.innerlines().to_vec(),
            tree.vertices().to_vec(),
        );
        return Ok((front_tree, tail_tree));
    }

    if front_rank == 0 {
        if rank == 0 {
            return Err(CoreError::MalformedFusionTree {
                message: "split at zero requires a non-empty source fusion tree",
            });
        }
        let unit = rule.vacuum();
        let front_tree = FusionTreeKey::new(
            Vec::<SectorId>::new(),
            Some(unit),
            Vec::<bool>::new(),
            Vec::<SectorId>::new(),
            Vec::<SectorId>::new(),
        );
        let mut tail_uncoupled = Vec::with_capacity(rank + 1);
        tail_uncoupled.push(unit);
        tail_uncoupled.extend_from_slice(tree.uncoupled());
        let mut tail_is_dual = Vec::with_capacity(rank + 1);
        tail_is_dual.push(false);
        tail_is_dual.extend_from_slice(tree.is_dual());
        let mut tail_innerlines = Vec::with_capacity(rank.saturating_sub(1));
        if rank >= 2 {
            tail_innerlines.push(tree.uncoupled()[0]);
            tail_innerlines.extend_from_slice(tree.innerlines());
        }
        let mut tail_vertices = Vec::with_capacity(rank);
        tail_vertices.push(SectorId::new(1));
        tail_vertices.extend_from_slice(tree.vertices());
        let tail_tree = FusionTreeKey::new(
            tail_uncoupled,
            tree.coupled(),
            tail_is_dual,
            tail_innerlines,
            tail_vertices,
        );
        return Ok((front_tree, tail_tree));
    }

    let intermediate =
        *tree
            .innerlines()
            .get(front_rank - 2)
            .ok_or(CoreError::MalformedFusionTree {
                message: "split requires the intermediate innerline",
            })?;
    let front_tree = FusionTreeKey::new(
        tree.uncoupled()[..front_rank].to_vec(),
        Some(intermediate),
        tree.is_dual()[..front_rank].to_vec(),
        tree.innerlines()[..front_rank.saturating_sub(2)].to_vec(),
        tree.vertices()[..front_rank - 1].to_vec(),
    );

    let mut tail_uncoupled = Vec::with_capacity(rank - front_rank + 1);
    tail_uncoupled.push(intermediate);
    tail_uncoupled.extend_from_slice(&tree.uncoupled()[front_rank..]);
    let mut tail_is_dual = Vec::with_capacity(rank - front_rank + 1);
    tail_is_dual.push(false);
    tail_is_dual.extend_from_slice(&tree.is_dual()[front_rank..]);
    let tail_tree = FusionTreeKey::new(
        tail_uncoupled,
        tree.coupled(),
        tail_is_dual,
        tree.innerlines()[front_rank - 1..].to_vec(),
        tree.vertices()[front_rank - 1..].to_vec(),
    );
    Ok((front_tree, tail_tree))
}

fn validate_fusion_tree_key_shape(tree: &FusionTreeKey) -> Result<(), CoreError> {
    let rank = tree.uncoupled().len();
    if tree.is_dual().len() != rank {
        return Err(CoreError::MalformedFusionTree {
            message: "fusion tree sectors and duality flags must have matching length",
        });
    }
    let expected_innerlines = rank.saturating_sub(2);
    if tree.innerlines().len() != expected_innerlines {
        return Err(CoreError::MalformedFusionTree {
            message: "fusion tree has an invalid number of innerlines",
        });
    }
    let expected_vertices = rank.saturating_sub(1);
    if tree.vertices().len() != expected_vertices {
        return Err(CoreError::MalformedFusionTree {
            message: "fusion tree has an invalid number of vertices",
        });
    }
    Ok(())
}

pub fn unique_artin_braid_first<R>(
    rule: &R,
    tree: &FusionTreeKey,
) -> Result<(FusionTreeKey, R::Scalar), CoreError>
where
    R: MultiplicityFreeFusionSymbols,
    R::Scalar: Mul<Output = R::Scalar>,
{
    unique_artin_braid_at(rule, tree, 0)
}

pub fn unique_artin_braid_at<R>(
    rule: &R,
    tree: &FusionTreeKey,
    index: usize,
) -> Result<(FusionTreeKey, R::Scalar), CoreError>
where
    R: MultiplicityFreeFusionSymbols,
    R::Scalar: Mul<Output = R::Scalar>,
{
    unique_artin_braid_at_with_inverse(rule, tree, index, false)
}

pub fn unique_braid_tree<R>(
    rule: &R,
    tree: &FusionTreeKey,
    permutation: &[usize],
    levels: &[usize],
) -> Result<(FusionTreeKey, R::Scalar), CoreError>
where
    R: MultiplicityFreeFusionSymbols,
    R::Scalar: Mul<Output = R::Scalar>,
{
    let rank = tree.uncoupled().len();
    if levels.len() != rank {
        return Err(CoreError::DimensionMismatch {
            expected: rank,
            actual: levels.len(),
        });
    }
    let swaps = permutation_to_adjacent_swaps(permutation, rank)?;
    let mut current = tree.clone();
    let mut coefficient = rule.scalar_one();
    let mut current_levels = levels.to_vec();
    for swap in swaps {
        let inverse = current_levels[swap] > current_levels[swap + 1];
        let (next, step_coefficient) =
            unique_artin_braid_at_with_inverse(rule, &current, swap, inverse)?;
        coefficient = coefficient * step_coefficient;
        current_levels.swap(swap, swap + 1);
        current = next;
    }
    Ok((current, coefficient))
}

pub fn unique_permute_tree<R>(
    rule: &R,
    tree: &FusionTreeKey,
    permutation: &[usize],
) -> Result<(FusionTreeKey, R::Scalar), CoreError>
where
    R: MultiplicityFreeFusionSymbols,
    R::Scalar: Mul<Output = R::Scalar>,
{
    if !rule.braiding_style().is_symmetric() {
        return Err(CoreError::UnsupportedBraidingStyle {
            expected: "symmetric braiding",
            actual: rule.braiding_style(),
        });
    }
    let levels = (0..tree.uncoupled().len()).collect::<Vec<_>>();
    unique_braid_tree(rule, tree, permutation, &levels)
}

pub fn multiplicity_free_artin_braid_at<R>(
    rule: &R,
    tree: &FusionTreeKey,
    index: usize,
) -> Result<Vec<(FusionTreeKey, R::Scalar)>, CoreError>
where
    R: MultiplicityFreeFusionSymbols,
    R::Scalar: Clone + Mul<Output = R::Scalar>,
{
    Ok(
        multiplicity_free_artin_braid_at_with_inverse(rule, tree, index, false)?
            .into_iter()
            .collect(),
    )
}

pub fn multiplicity_free_braid_tree<R>(
    rule: &R,
    tree: &FusionTreeKey,
    permutation: &[usize],
    levels: &[usize],
) -> Result<Vec<(FusionTreeKey, R::Scalar)>, CoreError>
where
    R: MultiplicityFreeFusionSymbols,
    R::Scalar: Clone + Add<Output = R::Scalar> + Mul<Output = R::Scalar>,
{
    if !rule.fusion_style().is_multiplicity_free() {
        return Err(CoreError::UnsupportedFusionStyle {
            expected: FusionStyleKind::Simple,
            actual: rule.fusion_style(),
        });
    }
    let rank = tree.uncoupled().len();
    if levels.len() != rank {
        return Err(CoreError::DimensionMismatch {
            expected: rank,
            actual: levels.len(),
        });
    }
    let swaps = permutation_to_adjacent_swaps(permutation, rank)?;
    let mut current = vec![(tree.clone(), rule.scalar_one())];
    let mut current_levels = levels.to_vec();
    for swap in swaps {
        let inverse = current_levels[swap] > current_levels[swap + 1];
        let mut next_terms = FusionTermAccumulator::new();
        for (tree, coefficient) in current {
            for (next_tree, step_coefficient) in
                multiplicity_free_artin_braid_at_with_inverse(rule, &tree, swap, inverse)?
            {
                next_terms.push(next_tree, coefficient.clone() * step_coefficient);
            }
        }
        current_levels.swap(swap, swap + 1);
        current = next_terms.into_vec();
    }
    Ok(current)
}

pub fn multiplicity_free_permute_tree<R>(
    rule: &R,
    tree: &FusionTreeKey,
    permutation: &[usize],
) -> Result<Vec<(FusionTreeKey, R::Scalar)>, CoreError>
where
    R: MultiplicityFreeFusionSymbols,
    R::Scalar: Clone + Add<Output = R::Scalar> + Mul<Output = R::Scalar>,
{
    if !rule.braiding_style().is_symmetric() {
        return Err(CoreError::UnsupportedBraidingStyle {
            expected: "symmetric braiding",
            actual: rule.braiding_style(),
        });
    }
    let levels = (0..tree.uncoupled().len()).collect::<Vec<_>>();
    multiplicity_free_braid_tree(rule, tree, permutation, &levels)
}

pub fn multiplicity_free_repartition_tree_pair<R>(
    rule: &R,
    tree_pair: &FusionTreeBlockKey,
    target_codomain_rank: usize,
) -> Result<Vec<(FusionTreeBlockKey, R::Scalar)>, CoreError>
where
    R: MultiplicityFreeRigidSymbols,
    R::Scalar: Clone + Add<Output = R::Scalar> + Mul<Output = R::Scalar>,
{
    let total_rank =
        tree_pair.codomain_tree().uncoupled().len() + tree_pair.domain_tree().uncoupled().len();
    if target_codomain_rank > total_rank {
        return Err(CoreError::DimensionMismatch {
            expected: total_rank,
            actual: target_codomain_rank,
        });
    }

    let mut current = vec![(tree_pair.clone(), rule.scalar_one())];
    let mut current_codomain_rank = tree_pair.codomain_tree().uncoupled().len();
    while current_codomain_rank < target_codomain_rank {
        current = compose_tree_pair_terms(rule, current, |rule, key| {
            multiplicity_free_bendleft_tree_pair(rule, key)
        })?;
        current_codomain_rank += 1;
    }
    while current_codomain_rank > target_codomain_rank {
        current = compose_tree_pair_terms(rule, current, |rule, key| {
            multiplicity_free_bendright_tree_pair(rule, key)
        })?;
        current_codomain_rank -= 1;
    }
    Ok(current)
}

pub fn multiplicity_free_braid_tree_pair<R>(
    rule: &R,
    tree_pair: &FusionTreeBlockKey,
    codomain_permutation: &[usize],
    domain_permutation: &[usize],
    codomain_levels: &[usize],
    domain_levels: &[usize],
) -> Result<Vec<(FusionTreeBlockKey, R::Scalar)>, CoreError>
where
    R: MultiplicityFreeRigidSymbols,
    R::Scalar: Clone + Add<Output = R::Scalar> + Mul<Output = R::Scalar>,
{
    let codomain_rank = tree_pair.codomain_tree().uncoupled().len();
    let domain_rank = tree_pair.domain_tree().uncoupled().len();
    if codomain_levels.len() != codomain_rank {
        return Err(CoreError::DimensionMismatch {
            expected: codomain_rank,
            actual: codomain_levels.len(),
        });
    }
    if domain_levels.len() != domain_rank {
        return Err(CoreError::DimensionMismatch {
            expected: domain_rank,
            actual: domain_levels.len(),
        });
    }

    let permutation = linearize_tree_pair_permutation(
        codomain_permutation,
        domain_permutation,
        codomain_rank,
        domain_rank,
    )?;
    let mut levels = Vec::with_capacity(codomain_rank + domain_rank);
    levels.extend_from_slice(codomain_levels);
    levels.extend(domain_levels.iter().rev().copied());

    let all_rank = codomain_rank + domain_rank;
    let mut current = multiplicity_free_repartition_tree_pair(rule, tree_pair, all_rank)?;
    current = compose_tree_pair_terms(rule, current, |rule, key| {
        multiplicity_free_braid_tree(rule, key.codomain_tree(), &permutation, &levels).map(
            |terms| {
                terms
                    .into_iter()
                    .map(|(codomain_tree, coefficient)| {
                        (
                            FusionTreeBlockKey::pair(codomain_tree, key.domain_tree().clone()),
                            coefficient,
                        )
                    })
                    .collect::<Vec<_>>()
            },
        )
    })?;
    multiplicity_free_repartition_terms(rule, current, codomain_permutation.len())
}

pub fn multiplicity_free_permute_tree_pair<R>(
    rule: &R,
    tree_pair: &FusionTreeBlockKey,
    codomain_permutation: &[usize],
    domain_permutation: &[usize],
) -> Result<Vec<(FusionTreeBlockKey, R::Scalar)>, CoreError>
where
    R: MultiplicityFreeRigidSymbols,
    R::Scalar: Clone + Add<Output = R::Scalar> + Mul<Output = R::Scalar>,
{
    if !rule.braiding_style().is_symmetric() {
        return Err(CoreError::UnsupportedBraidingStyle {
            expected: "symmetric braiding",
            actual: rule.braiding_style(),
        });
    }
    let codomain_rank = tree_pair.codomain_tree().uncoupled().len();
    let domain_rank = tree_pair.domain_tree().uncoupled().len();
    let codomain_levels = (0..codomain_rank).collect::<Vec<_>>();
    let domain_levels = (codomain_rank..codomain_rank + domain_rank).collect::<Vec<_>>();
    multiplicity_free_braid_tree_pair(
        rule,
        tree_pair,
        codomain_permutation,
        domain_permutation,
        &codomain_levels,
        &domain_levels,
    )
}

pub fn multiplicity_free_transpose_tree_pair<R>(
    rule: &R,
    tree_pair: &FusionTreeBlockKey,
    codomain_permutation: &[usize],
    domain_permutation: &[usize],
) -> Result<Vec<(FusionTreeBlockKey, R::Scalar)>, CoreError>
where
    R: MultiplicityFreeRigidSymbols,
    R::Scalar: Clone + Add<Output = R::Scalar> + Mul<Output = R::Scalar>,
{
    let codomain_rank = tree_pair.codomain_tree().uncoupled().len();
    let domain_rank = tree_pair.domain_tree().uncoupled().len();
    let permutation = linearize_tree_pair_permutation(
        codomain_permutation,
        domain_permutation,
        codomain_rank,
        domain_rank,
    )?;
    if !is_cyclic_permutation(&permutation) {
        return Err(CoreError::InvalidPermutation {
            permutation,
            rank: codomain_rank + domain_rank,
        });
    }

    let mut position = match permutation.iter().position(|&axis| axis == 0) {
        Some(position) => position,
        None => return Ok(vec![(tree_pair.clone(), rule.scalar_one())]),
    };
    let mut current =
        multiplicity_free_repartition_tree_pair(rule, tree_pair, codomain_permutation.len())?;
    let total_rank = codomain_rank + domain_rank;
    if total_rank == 0 || position == 0 {
        return Ok(current);
    }

    let half_rank = total_rank >> 1;
    while position > 0 && position < half_rank {
        current = compose_tree_pair_terms(rule, current, |rule, key| {
            multiplicity_free_cycle_anticlockwise_tree_pair(rule, key)
        })?;
        position -= 1;
    }
    while position >= half_rank && position > 0 {
        current = compose_tree_pair_terms(rule, current, |rule, key| {
            multiplicity_free_cycle_clockwise_tree_pair(rule, key)
        })?;
        position = (position + 1) % total_rank;
    }

    Ok(current)
}

enum FusionTermAccumulator<K, S> {
    Empty,
    Singleton(K, S),
    Map {
        order: Vec<K>,
        coefficients: FxHashMap<K, S>,
    },
}

impl<K, S> FusionTermAccumulator<K, S>
where
    K: Clone + Eq + Hash,
    S: Clone + Add<Output = S>,
{
    fn new() -> Self {
        Self::Empty
    }

    fn push(&mut self, key: K, coefficient: S) {
        match self {
            Self::Empty => {
                *self = Self::Singleton(key, coefficient);
            }
            Self::Singleton(existing_key, existing) if existing_key == &key => {
                *existing = existing.clone() + coefficient;
            }
            Self::Singleton(_, _) => {
                let previous = std::mem::replace(self, Self::Empty);
                let Self::Singleton(existing_key, existing_coefficient) = previous else {
                    unreachable!("matched singleton state");
                };
                let mut order = Vec::with_capacity(2);
                let mut coefficients = FxHashMap::default();
                Self::push_map_term(
                    &mut order,
                    &mut coefficients,
                    existing_key,
                    existing_coefficient,
                );
                Self::push_map_term(&mut order, &mut coefficients, key, coefficient);
                *self = Self::Map {
                    order,
                    coefficients,
                };
            }
            Self::Map {
                order,
                coefficients,
            } => {
                Self::push_map_term(order, coefficients, key, coefficient);
            }
        }
    }

    fn push_map_term(
        order: &mut Vec<K>,
        coefficients: &mut FxHashMap<K, S>,
        key: K,
        coefficient: S,
    ) {
        match coefficients.entry(key) {
            Entry::Occupied(mut entry) => {
                let existing = entry.get_mut();
                *existing = existing.clone() + coefficient;
            }
            Entry::Vacant(entry) => {
                order.push(entry.key().clone());
                entry.insert(coefficient);
            }
        }
    }

    fn into_vec(self) -> Vec<(K, S)> {
        match self {
            Self::Empty => Vec::new(),
            Self::Singleton(key, coefficient) => vec![(key, coefficient)],
            Self::Map {
                order,
                mut coefficients,
            } => {
                let mut terms = Vec::with_capacity(order.len());
                for key in order {
                    let coefficient = coefficients
                        .remove(&key)
                        .expect("accumulator order only contains inserted keys");
                    terms.push((key, coefficient));
                }
                terms
            }
        }
    }
}

fn compose_tree_pair_terms<R, F, I>(
    rule: &R,
    terms: Vec<(FusionTreeBlockKey, R::Scalar)>,
    mut transform: F,
) -> Result<Vec<(FusionTreeBlockKey, R::Scalar)>, CoreError>
where
    R: MultiplicityFreeRigidSymbols,
    R::Scalar: Clone + Add<Output = R::Scalar> + Mul<Output = R::Scalar>,
    F: FnMut(&R, &FusionTreeBlockKey) -> Result<I, CoreError>,
    I: IntoIterator<Item = (FusionTreeBlockKey, R::Scalar)>,
{
    let mut output = FusionTermAccumulator::new();
    for (key, coefficient) in terms {
        for (next_key, next_coefficient) in transform(rule, &key)? {
            output.push(next_key, coefficient.clone() * next_coefficient);
        }
    }
    Ok(output.into_vec())
}

/// Batched analog of [`compose_tree_pair_terms`]: apply `transform` to every
/// tree-pair of a whole block at once, threading a coefficient *matrix* (a
/// sparse column per original source) instead of re-running the per-source
/// term list. `columns[i]` maps `src index -> coefficient` for `basis[i]`.
///
/// This is the TensorKit 0.17 `artin_braid`/`fsbraid` batching: the elementary
/// step (bend / Artin braid) is walked ONCE for the block and its coefficients
/// are spread across all source columns, so intermediate allocation is
/// O(steps) rather than O(steps × sources) — the term-list style TeNeT used
/// (equivalent to TensorKit ≤0.16's per-tree `FusionTreeDict`) allocated a
/// fresh accumulator and cloned keys per source per step.
fn compose_block_terms<R, F, I>(
    rule: &R,
    basis: &[FusionTreeBlockKey],
    columns: &[Vec<Option<R::Scalar>>],
    num_src: usize,
    mut transform: F,
) -> Result<(Vec<FusionTreeBlockKey>, Vec<Vec<Option<R::Scalar>>>), CoreError>
where
    R: MultiplicityFreeRigidSymbols,
    R::Scalar: Clone + Add<Output = R::Scalar> + Mul<Output = R::Scalar>,
    F: FnMut(&R, &FusionTreeBlockKey) -> Result<I, CoreError>,
    I: IntoIterator<Item = (FusionTreeBlockKey, R::Scalar)>,
{
    let mut index: FxHashMap<FusionTreeBlockKey, usize> = FxHashMap::default();
    let mut next_basis: Vec<FusionTreeBlockKey> = Vec::new();
    let mut next_columns: Vec<Vec<Option<R::Scalar>>> = Vec::new();
    for (source_key, source_column) in basis.iter().zip(columns) {
        for (dst_key, step_coefficient) in transform(rule, source_key)? {
            let row = match index.get(&dst_key) {
                Some(&row) => row,
                None => {
                    let row = next_basis.len();
                    index.insert(dst_key.clone(), row);
                    next_basis.push(dst_key);
                    // Dense coefficient column over the original sources (TK's
                    // `Matrix{E}`): no per-tree hashmap alloc, no usize hashing.
                    next_columns.push(vec![None; num_src]);
                    row
                }
            };
            // dst_column[src] += step_coefficient * source_column[src] for each
            // source that reaches this basis tree.
            let dst_column = &mut next_columns[row];
            for (src, source_coefficient) in source_column.iter().enumerate() {
                let Some(source_coefficient) = source_coefficient else {
                    continue;
                };
                let contribution = step_coefficient.clone() * source_coefficient.clone();
                dst_column[src] = Some(match dst_column[src].take() {
                    Some(existing) => existing + contribution,
                    None => contribution,
                });
            }
        }
    }
    Ok((next_basis, next_columns))
}

/// Batched [`multiplicity_free_braid_tree_pair`] over every source tree-pair of
/// a block (all sharing the same uncoupled sectors / duality). Returns, per
/// source (in `src_keys` order), its `(destination tree-pair, coefficient)`
/// rows — identical content to calling the per-source function on each, but the
/// bend/braid step structure is walked once for the block.
///
/// The floating-point *summation order* of coefficients that reach a
/// destination by several paths differs from the per-source accumulator, so
/// results agree with the per-source version to double-precision rounding, not
/// necessarily bit-for-bit.
pub fn multiplicity_free_braid_tree_pair_block<R>(
    rule: &R,
    src_keys: &[FusionTreeBlockKey],
    codomain_permutation: &[usize],
    domain_permutation: &[usize],
    codomain_levels: &[usize],
    domain_levels: &[usize],
) -> Result<Vec<Vec<(FusionTreeBlockKey, R::Scalar)>>, CoreError>
where
    R: MultiplicityFreeRigidSymbols,
    R::Scalar: Clone + Add<Output = R::Scalar> + Mul<Output = R::Scalar>,
{
    if src_keys.is_empty() {
        return Ok(Vec::new());
    }
    let codomain_rank = src_keys[0].codomain_tree().uncoupled().len();
    let domain_rank = src_keys[0].domain_tree().uncoupled().len();
    if codomain_levels.len() != codomain_rank {
        return Err(CoreError::DimensionMismatch {
            expected: codomain_rank,
            actual: codomain_levels.len(),
        });
    }
    if domain_levels.len() != domain_rank {
        return Err(CoreError::DimensionMismatch {
            expected: domain_rank,
            actual: domain_levels.len(),
        });
    }

    let permutation = linearize_tree_pair_permutation(
        codomain_permutation,
        domain_permutation,
        codomain_rank,
        domain_rank,
    )?;
    let mut levels = Vec::with_capacity(codomain_rank + domain_rank);
    levels.extend_from_slice(codomain_levels);
    levels.extend(domain_levels.iter().rev().copied());
    let all_rank = codomain_rank + domain_rank;

    // Identity matrix: source `i` starts as its own basis tree with coeff one.
    let num_src = src_keys.len();
    let mut basis = src_keys.to_vec();
    let mut columns: Vec<Vec<Option<R::Scalar>>> = (0..num_src)
        .map(|i| {
            let mut column = vec![None; num_src];
            column[i] = Some(rule.scalar_one());
            column
        })
        .collect();

    // Step A: repartition everything into the codomain (bendleft chain).
    let mut current_codomain_rank = codomain_rank;
    while current_codomain_rank < all_rank {
        let (next_basis, next_columns) =
            compose_block_terms(rule, &basis, &columns, num_src, |rule, key| {
                multiplicity_free_bendleft_tree_pair(rule, key)
            })?;
        basis = next_basis;
        columns = next_columns;
        current_codomain_rank += 1;
    }

    // Step B: braid the (now all-codomain) tree ONE adjacent swap at a time,
    // each swap batched across the whole block. This replaces the per-source
    // inner braid (`multiplicity_free_braid_tree`, whose `FusionTermAccumulator`
    // and elementary-swap term lists ran once per source tree) with the shared
    // block matrix walk — the TensorKit 0.17 `artin_braid`-on-a-block scheme.
    let swaps = permutation_to_adjacent_swaps(&permutation, all_rank)?;
    let mut current_levels = levels.clone();
    for swap in swaps {
        let inverse = current_levels[swap] > current_levels[swap + 1];
        let (next_basis, next_columns) =
            compose_block_terms(rule, &basis, &columns, num_src, |rule, key| {
                let domain = key.domain_tree().clone();
                Ok(multiplicity_free_artin_braid_at_with_inverse(
                    rule,
                    key.codomain_tree(),
                    swap,
                    inverse,
                )?
                .into_iter()
                .map(move |(codomain_tree, coefficient)| {
                    (
                        FusionTreeBlockKey::pair(codomain_tree, domain.clone()),
                        coefficient,
                    )
                }))
            })?;
        basis = next_basis;
        columns = next_columns;
        current_levels.swap(swap, swap + 1);
    }

    // Step C: repartition back to the requested codomain rank.
    let target_codomain_rank = codomain_permutation.len();
    while current_codomain_rank > target_codomain_rank {
        let (next_basis, next_columns) =
            compose_block_terms(rule, &basis, &columns, num_src, |rule, key| {
                multiplicity_free_bendright_tree_pair(rule, key)
            })?;
        basis = next_basis;
        columns = next_columns;
        current_codomain_rank -= 1;
    }
    while current_codomain_rank < target_codomain_rank {
        let (next_basis, next_columns) =
            compose_block_terms(rule, &basis, &columns, num_src, |rule, key| {
                multiplicity_free_bendleft_tree_pair(rule, key)
            })?;
        basis = next_basis;
        columns = next_columns;
        current_codomain_rank += 1;
    }

    // Scatter the dense matrix back into per-source row lists. Columns are
    // indexed by source, so iterating in source order needs no sort.
    let mut rows_per_source: Vec<Vec<(FusionTreeBlockKey, R::Scalar)>> = vec![Vec::new(); num_src];
    for (dst_key, column) in basis.iter().zip(&columns) {
        for (src, coefficient) in column.iter().enumerate() {
            if let Some(coefficient) = coefficient {
                rows_per_source[src].push((dst_key.clone(), coefficient.clone()));
            }
        }
    }
    Ok(rows_per_source)
}

/// Batched [`multiplicity_free_permute_tree_pair`] over a block: symmetric
/// braiding with the trivial level ordering.
pub fn multiplicity_free_permute_tree_pair_block<R>(
    rule: &R,
    src_keys: &[FusionTreeBlockKey],
    codomain_permutation: &[usize],
    domain_permutation: &[usize],
) -> Result<Vec<Vec<(FusionTreeBlockKey, R::Scalar)>>, CoreError>
where
    R: MultiplicityFreeRigidSymbols,
    R::Scalar: Clone + Add<Output = R::Scalar> + Mul<Output = R::Scalar>,
{
    if !rule.braiding_style().is_symmetric() {
        return Err(CoreError::UnsupportedBraidingStyle {
            expected: "symmetric braiding",
            actual: rule.braiding_style(),
        });
    }
    if src_keys.is_empty() {
        return Ok(Vec::new());
    }
    let codomain_rank = src_keys[0].codomain_tree().uncoupled().len();
    let domain_rank = src_keys[0].domain_tree().uncoupled().len();
    let codomain_levels = (0..codomain_rank).collect::<Vec<_>>();
    let domain_levels = (codomain_rank..codomain_rank + domain_rank).collect::<Vec<_>>();
    multiplicity_free_braid_tree_pair_block(
        rule,
        src_keys,
        codomain_permutation,
        domain_permutation,
        &codomain_levels,
        &domain_levels,
    )
}

fn multiplicity_free_repartition_terms<R>(
    rule: &R,
    terms: Vec<(FusionTreeBlockKey, R::Scalar)>,
    target_codomain_rank: usize,
) -> Result<Vec<(FusionTreeBlockKey, R::Scalar)>, CoreError>
where
    R: MultiplicityFreeRigidSymbols,
    R::Scalar: Clone + Add<Output = R::Scalar> + Mul<Output = R::Scalar>,
{
    let mut current = terms;
    let Some((first_key, _)) = current.first() else {
        return Ok(current);
    };
    let total_rank =
        first_key.codomain_tree().uncoupled().len() + first_key.domain_tree().uncoupled().len();
    if target_codomain_rank > total_rank {
        return Err(CoreError::DimensionMismatch {
            expected: total_rank,
            actual: target_codomain_rank,
        });
    }
    let mut current_codomain_rank = first_key.codomain_tree().uncoupled().len();
    while current_codomain_rank < target_codomain_rank {
        current = compose_tree_pair_terms(rule, current, |rule, key| {
            multiplicity_free_bendleft_tree_pair(rule, key)
        })?;
        current_codomain_rank += 1;
    }
    while current_codomain_rank > target_codomain_rank {
        current = compose_tree_pair_terms(rule, current, |rule, key| {
            multiplicity_free_bendright_tree_pair(rule, key)
        })?;
        current_codomain_rank -= 1;
    }
    Ok(current)
}

pub fn unique_braid_tree_pair<R>(
    rule: &R,
    tree_pair: &FusionTreeBlockKey,
    codomain_permutation: &[usize],
    domain_permutation: &[usize],
    codomain_levels: &[usize],
    domain_levels: &[usize],
) -> Result<(FusionTreeBlockKey, R::Scalar), CoreError>
where
    R: MultiplicityFreePivotalSymbols,
    R::Scalar: Mul<Output = R::Scalar>,
{
    let codomain_rank = tree_pair.codomain_tree().uncoupled().len();
    let domain_rank = tree_pair.domain_tree().uncoupled().len();
    if codomain_levels.len() != codomain_rank {
        return Err(CoreError::DimensionMismatch {
            expected: codomain_rank,
            actual: codomain_levels.len(),
        });
    }
    if domain_levels.len() != domain_rank {
        return Err(CoreError::DimensionMismatch {
            expected: domain_rank,
            actual: domain_levels.len(),
        });
    }

    let permutation = linearize_tree_pair_permutation(
        codomain_permutation,
        domain_permutation,
        codomain_rank,
        domain_rank,
    )?;
    let mut levels = Vec::with_capacity(codomain_rank + domain_rank);
    levels.extend_from_slice(codomain_levels);
    levels.extend(domain_levels.iter().rev().copied());

    let (all_codomain_pair, repartition_to_all_coeff) =
        unique_repartition_tree_pair(rule, tree_pair, codomain_rank + domain_rank)?;
    let (braided_codomain_tree, braid_coeff) = unique_braid_tree(
        rule,
        all_codomain_pair.codomain_tree(),
        &permutation,
        &levels,
    )?;
    let braided_pair = FusionTreeBlockKey::pair(
        braided_codomain_tree,
        all_codomain_pair.domain_tree().clone(),
    );
    let (dst_pair, repartition_back_coeff) =
        unique_repartition_tree_pair(rule, &braided_pair, codomain_permutation.len())?;

    Ok((
        dst_pair,
        repartition_to_all_coeff * braid_coeff * repartition_back_coeff,
    ))
}

pub fn unique_permute_tree_pair<R>(
    rule: &R,
    tree_pair: &FusionTreeBlockKey,
    codomain_permutation: &[usize],
    domain_permutation: &[usize],
) -> Result<(FusionTreeBlockKey, R::Scalar), CoreError>
where
    R: MultiplicityFreePivotalSymbols,
    R::Scalar: Mul<Output = R::Scalar>,
{
    if !rule.braiding_style().is_symmetric() {
        return Err(CoreError::UnsupportedBraidingStyle {
            expected: "symmetric braiding",
            actual: rule.braiding_style(),
        });
    }
    let codomain_rank = tree_pair.codomain_tree().uncoupled().len();
    let domain_rank = tree_pair.domain_tree().uncoupled().len();
    let codomain_levels = (0..codomain_rank).collect::<Vec<_>>();
    let domain_levels = (codomain_rank..codomain_rank + domain_rank).collect::<Vec<_>>();
    unique_braid_tree_pair(
        rule,
        tree_pair,
        codomain_permutation,
        domain_permutation,
        &codomain_levels,
        &domain_levels,
    )
}

pub fn unique_transpose_tree_pair<R>(
    rule: &R,
    tree_pair: &FusionTreeBlockKey,
    codomain_permutation: &[usize],
    domain_permutation: &[usize],
) -> Result<(FusionTreeBlockKey, R::Scalar), CoreError>
where
    R: MultiplicityFreePivotalSymbols,
    R::Scalar: Mul<Output = R::Scalar>,
{
    let codomain_rank = tree_pair.codomain_tree().uncoupled().len();
    let domain_rank = tree_pair.domain_tree().uncoupled().len();
    let permutation = linearize_tree_pair_permutation(
        codomain_permutation,
        domain_permutation,
        codomain_rank,
        domain_rank,
    )?;
    if !is_cyclic_permutation(&permutation) {
        return Err(CoreError::InvalidPermutation {
            permutation,
            rank: codomain_rank + domain_rank,
        });
    }

    let mut position = match permutation.iter().position(|&axis| axis == 0) {
        Some(position) => position,
        None => return Ok((tree_pair.clone(), rule.scalar_one())),
    };
    let mut current = unique_repartition_tree_pair(rule, tree_pair, codomain_permutation.len())?;
    let total_rank = codomain_rank + domain_rank;
    if total_rank == 0 || position == 0 {
        return Ok(current);
    }

    let half_rank = total_rank >> 1;
    while position > 0 && position < half_rank {
        let (next, coefficient) = unique_cycle_anticlockwise_tree_pair(rule, &current.0)?;
        current = (next, current.1 * coefficient);
        position -= 1;
    }
    while position >= half_rank && position > 0 {
        let (next, coefficient) = unique_cycle_clockwise_tree_pair(rule, &current.0)?;
        current = (next, current.1 * coefficient);
        position = (position + 1) % total_rank;
    }

    Ok(current)
}

pub fn unique_repartition_tree_pair<R>(
    rule: &R,
    tree_pair: &FusionTreeBlockKey,
    target_codomain_rank: usize,
) -> Result<(FusionTreeBlockKey, R::Scalar), CoreError>
where
    R: MultiplicityFreePivotalSymbols,
    R::Scalar: Mul<Output = R::Scalar>,
{
    if rule.fusion_style() != FusionStyleKind::Unique {
        return Err(CoreError::UnsupportedFusionStyle {
            expected: FusionStyleKind::Unique,
            actual: rule.fusion_style(),
        });
    }

    let total_rank =
        tree_pair.codomain_tree().uncoupled().len() + tree_pair.domain_tree().uncoupled().len();
    if target_codomain_rank > total_rank {
        return Err(CoreError::DimensionMismatch {
            expected: total_rank,
            actual: target_codomain_rank,
        });
    }

    let mut current = tree_pair.clone();
    let mut current_codomain_rank = current.codomain_tree().uncoupled().len();
    let mut coefficient = rule.scalar_one();
    while current_codomain_rank < target_codomain_rank {
        let (next, step_coefficient) = unique_bendleft_tree_pair(rule, &current)?;
        coefficient = coefficient * step_coefficient;
        current = next;
        current_codomain_rank += 1;
    }
    while current_codomain_rank > target_codomain_rank {
        let (next, step_coefficient) = unique_bendright_tree_pair(rule, &current)?;
        coefficient = coefficient * step_coefficient;
        current = next;
        current_codomain_rank -= 1;
    }
    Ok((current, coefficient))
}

pub fn linearize_tree_pair_permutation(
    codomain_permutation: &[usize],
    domain_permutation: &[usize],
    codomain_rank: usize,
    domain_rank: usize,
) -> Result<Vec<usize>, CoreError> {
    let total_rank = codomain_rank + domain_rank;
    let mut original_permutation =
        Vec::with_capacity(codomain_permutation.len() + domain_permutation.len());
    original_permutation.extend_from_slice(codomain_permutation);
    original_permutation.extend_from_slice(domain_permutation);
    validate_permutation(&original_permutation, total_rank)?;

    let mut linearized = Vec::with_capacity(total_rank);
    linearized.extend(
        codomain_permutation
            .iter()
            .map(|&axis| linearize_tree_pair_axis(axis, codomain_rank, domain_rank)),
    );
    linearized.extend(
        domain_permutation
            .iter()
            .rev()
            .map(|&axis| linearize_tree_pair_axis(axis, codomain_rank, domain_rank)),
    );
    validate_permutation(&linearized, total_rank)?;
    Ok(linearized)
}

fn unique_artin_braid_at_with_inverse<R>(
    rule: &R,
    tree: &FusionTreeKey,
    index: usize,
    inverse: bool,
) -> Result<(FusionTreeKey, R::Scalar), CoreError>
where
    R: MultiplicityFreeFusionSymbols,
    R::Scalar: Mul<Output = R::Scalar>,
{
    if rule.fusion_style() != FusionStyleKind::Unique {
        return Err(CoreError::UnsupportedFusionStyle {
            expected: FusionStyleKind::Unique,
            actual: rule.fusion_style(),
        });
    }

    let rank = tree.uncoupled().len();
    if index + 1 >= rank {
        return Err(CoreError::InvalidBraidIndex { index, rank });
    }

    let left = tree.uncoupled()[index];
    let right = tree.uncoupled()[index + 1];
    let mut uncoupled = tree.uncoupled().to_vec();
    uncoupled.swap(index, index + 1);
    let mut is_dual = tree.is_dual().to_vec();
    is_dual.swap(index, index + 1);
    let mut innerlines = tree.innerlines().to_vec();
    let mut vertices = tree.vertices().to_vec();

    if left == rule.vacuum() || right == rule.vacuum() {
        if index > 0 {
            let inner_source = if left == rule.vacuum() {
                inner_extended_sector(tree, index + 1)?
            } else {
                inner_extended_sector(tree, index - 1)?
            };
            *innerlines
                .get_mut(index - 1)
                .ok_or(CoreError::MalformedFusionTree {
                    message: "unit braid past the first adjacent pair requires an innerline",
                })? = inner_source;
            if vertices.len() <= index {
                return Err(CoreError::MalformedFusionTree {
                    message: "unit braid past the first adjacent pair requires adjacent vertices",
                });
            }
            vertices.swap(index - 1, index);
        }

        let braided = FusionTreeKey::new(uncoupled, tree.coupled(), is_dual, innerlines, vertices);
        return Ok((braided, rule.scalar_one()));
    }

    if !rule.braiding_style().has_braiding() {
        return Err(CoreError::UnsupportedSectorBraid {
            left,
            right,
            style: rule.braiding_style(),
        });
    }

    if index == 0 {
        let coupled = if rank > 2 {
            tree.innerlines()
                .first()
                .copied()
                .ok_or(CoreError::MalformedFusionTree {
                    message: "first braid of a rank > 2 tree requires the first innerline",
                })?
        } else {
            tree.coupled().ok_or(CoreError::MalformedFusionTree {
                message: "first braid of a rank 2 tree requires a coupled sector",
            })?
        };

        let braided = FusionTreeKey::new(uncoupled, tree.coupled(), is_dual, innerlines, vertices);
        let coefficient = if inverse {
            rule.scalar_conj(rule.r_symbol_scalar(right, left, coupled))
        } else {
            rule.r_symbol_scalar(left, right, coupled)
        };
        return Ok((braided, coefficient));
    }

    let a = inner_extended_sector(tree, index - 1)?;
    let b = left;
    let c = inner_extended_sector(tree, index)?;
    let d = right;
    let e = inner_extended_sector(tree, index + 1)?;
    let c_prime = only_fusion_channel(rule, a, d)?;
    *innerlines
        .get_mut(index - 1)
        .ok_or(CoreError::MalformedFusionTree {
            message: "non-first braid requires an innerline to update",
        })? = c_prime;
    let braided = FusionTreeKey::new(uncoupled, tree.coupled(), is_dual, innerlines, vertices);
    let f_symbol = rule.f_symbol_scalar(d, a, b, e, c_prime, c);
    let coefficient = if inverse {
        let left = rule.r_symbol_scalar(d, c, e);
        let right = rule.r_symbol_scalar(d, a, c_prime);
        rule.scalar_conj(left * f_symbol) * right
    } else {
        let left = rule.r_symbol_scalar(c, d, e);
        let right = rule.r_symbol_scalar(a, d, c_prime);
        left * rule.scalar_conj(f_symbol * right)
    };
    Ok((braided, coefficient))
}

fn multiplicity_free_artin_braid_at_with_inverse<R>(
    rule: &R,
    tree: &FusionTreeKey,
    index: usize,
    inverse: bool,
) -> Result<SmallVec<[(FusionTreeKey, R::Scalar); 2]>, CoreError>
where
    R: MultiplicityFreeFusionSymbols,
    R::Scalar: Clone + Mul<Output = R::Scalar>,
{
    if !rule.fusion_style().is_multiplicity_free() {
        return Err(CoreError::UnsupportedFusionStyle {
            expected: FusionStyleKind::Simple,
            actual: rule.fusion_style(),
        });
    }

    let rank = tree.uncoupled().len();
    if index + 1 >= rank {
        return Err(CoreError::InvalidBraidIndex { index, rank });
    }

    let left = tree.uncoupled()[index];
    let right = tree.uncoupled()[index + 1];
    // Collect into the inline `SmallVec` types (stack-resident for ≤8 legs)
    // rather than heap `Vec`s: this is on the per-swap braid hot path.
    let mut uncoupled: SectorVec = tree.uncoupled().iter().copied().collect();
    uncoupled.swap(index, index + 1);
    let mut is_dual: DualVec = tree.is_dual().iter().copied().collect();
    is_dual.swap(index, index + 1);

    if left == rule.vacuum() || right == rule.vacuum() {
        let mut innerlines = tree.innerlines().to_vec();
        let mut vertices = tree.vertices().to_vec();
        if index > 0 {
            let inner_source = if left == rule.vacuum() {
                inner_extended_sector(tree, index + 1)?
            } else {
                inner_extended_sector(tree, index - 1)?
            };
            *innerlines
                .get_mut(index - 1)
                .ok_or(CoreError::MalformedFusionTree {
                    message: "unit braid past the first adjacent pair requires an innerline",
                })? = inner_source;
            if vertices.len() <= index {
                return Err(CoreError::MalformedFusionTree {
                    message: "unit braid past the first adjacent pair requires adjacent vertices",
                });
            }
            vertices.swap(index - 1, index);
        }
        let mut out = SmallVec::new();
        out.push((
            FusionTreeKey::new(uncoupled, tree.coupled(), is_dual, innerlines, vertices),
            rule.scalar_one(),
        ));
        return Ok(out);
    }

    if !rule.braiding_style().has_braiding() {
        return Err(CoreError::UnsupportedSectorBraid {
            left,
            right,
            style: rule.braiding_style(),
        });
    }

    if index == 0 {
        let coupled = if rank > 2 {
            tree.innerlines()
                .first()
                .copied()
                .ok_or(CoreError::MalformedFusionTree {
                    message: "first braid of a rank > 2 tree requires the first innerline",
                })?
        } else {
            tree.coupled().ok_or(CoreError::MalformedFusionTree {
                message: "first braid of a rank 2 tree requires a coupled sector",
            })?
        };
        let coefficient = if inverse {
            rule.scalar_conj(rule.r_symbol_scalar(right, left, coupled))
        } else {
            rule.r_symbol_scalar(left, right, coupled)
        };
        let mut out = SmallVec::new();
        out.push((
            FusionTreeKey::new(
                uncoupled,
                tree.coupled(),
                is_dual,
                tree.innerlines().iter().copied(),
                tree.vertices().iter().copied(),
            ),
            coefficient,
        ));
        return Ok(out);
    }

    let a = inner_extended_sector(tree, index - 1)?;
    let b = left;
    let c = inner_extended_sector(tree, index)?;
    let d = right;
    let e = inner_extended_sector(tree, index + 1)?;
    let mut terms: SmallVec<[(FusionTreeKey, R::Scalar); 2]> = SmallVec::new();
    for c_prime in rule.fusion_channels(a, d) {
        if rule.nsymbol(c_prime, b, e) == 0 {
            continue;
        }
        let mut innerlines: SectorVec = tree.innerlines().iter().copied().collect();
        *innerlines
            .get_mut(index - 1)
            .ok_or(CoreError::MalformedFusionTree {
                message: "non-first braid requires an innerline to update",
            })? = c_prime;
        let braided = FusionTreeKey::new(
            uncoupled.clone(),
            tree.coupled(),
            is_dual.clone(),
            innerlines,
            tree.vertices().iter().copied(),
        );
        let f_symbol = rule.f_symbol_scalar(d, a, b, e, c_prime, c);
        let coefficient = if inverse {
            let left = rule.r_symbol_scalar(d, c, e);
            let right = rule.r_symbol_scalar(d, a, c_prime);
            rule.scalar_conj(left * f_symbol) * right
        } else {
            let left = rule.r_symbol_scalar(c, d, e);
            let right = rule.r_symbol_scalar(a, d, c_prime);
            left * rule.scalar_conj(f_symbol * right)
        };
        terms.push((braided, coefficient));
    }
    Ok(terms)
}

fn multiplicity_free_bendright_tree_pair<R>(
    rule: &R,
    tree_pair: &FusionTreeBlockKey,
) -> Result<SmallVec<[(FusionTreeBlockKey, R::Scalar); 1]>, CoreError>
where
    R: MultiplicityFreeRigidSymbols,
    R::Scalar: Clone + Mul<Output = R::Scalar>,
{
    let codomain = tree_pair.codomain_tree();
    let domain = tree_pair.domain_tree();
    let codomain_rank = codomain.uncoupled().len();
    if codomain_rank == 0 {
        return Err(CoreError::MalformedFusionTree {
            message: "bendright requires at least one codomain leg",
        });
    }

    let coupled = coupled_or_vacuum(rule, codomain);
    if !domain.uncoupled().is_empty() {
        let domain_coupled = coupled_or_vacuum(rule, domain);
        if domain_coupled != coupled {
            return Err(CoreError::MalformedFusionTree {
                message: "fusion tree pair requires matching coupled sectors",
            });
        }
    }

    let left_coupled = match codomain_rank {
        1 => rule.vacuum(),
        2 => codomain.uncoupled()[0],
        _ => codomain
            .innerlines()
            .last()
            .copied()
            .ok_or(CoreError::MalformedFusionTree {
                message: "bendright requires the last codomain innerline",
            })?,
    };
    let bent_sector = codomain.uncoupled()[codomain_rank - 1];
    let bent_is_dual = codomain.is_dual().get(codomain_rank - 1).copied().ok_or(
        CoreError::MalformedFusionTree {
            message: "codomain tree is missing a duality flag",
        },
    )?;

    // Build the new trees straight from slice iterators: `FusionTreeKey::new`
    // collects into inline `SmallVec`, so passing iterators (not `.to_vec()`)
    // keeps small trees stack-resident — the intermediate heap `Vec`s here were
    // a large share of the cold recoupling-compile malloc traffic.
    let cod_inner = codomain.innerlines();
    let new_codomain_innerlines: &[SectorId] = if codomain_rank > 2 {
        &cod_inner[..cod_inner.len() - 1]
    } else {
        &[]
    };
    let cod_vertices = codomain.vertices();
    let new_codomain_vertices: &[SectorId] = if codomain_rank > 1 {
        &cod_vertices[..cod_vertices.len() - 1]
    } else {
        &[]
    };
    let new_codomain = FusionTreeKey::new(
        codomain.uncoupled()[..codomain_rank - 1].iter().copied(),
        Some(left_coupled),
        codomain.is_dual()[..codomain_rank - 1].iter().copied(),
        new_codomain_innerlines.iter().copied(),
        new_codomain_vertices.iter().copied(),
    );

    let domain_rank = domain.uncoupled().len();
    let new_domain = FusionTreeKey::new(
        domain
            .uncoupled()
            .iter()
            .copied()
            .chain(std::iter::once(rule.dual(bent_sector))),
        Some(left_coupled),
        domain
            .is_dual()
            .iter()
            .copied()
            .chain(std::iter::once(!bent_is_dual)),
        domain
            .innerlines()
            .iter()
            .copied()
            .chain((domain_rank > 1).then_some(coupled)),
        domain
            .vertices()
            .iter()
            .copied()
            .chain((domain_rank > 0).then_some(SectorId::new(1))),
    );

    let mut coefficient = rule.sqrt_dim_scalar(coupled)
        * rule.inv_sqrt_dim_scalar(left_coupled)
        * rule.b_symbol_scalar(left_coupled, bent_sector, coupled);
    if bent_is_dual {
        coefficient = coefficient
            * rule.scalar_conj(rule.frobenius_schur_phase_scalar(rule.dual(bent_sector)));
    }
    let mut out = SmallVec::new();
    out.push((
        FusionTreeBlockKey::pair(new_codomain, new_domain),
        coefficient,
    ));
    Ok(out)
}

fn multiplicity_free_bendleft_tree_pair<R>(
    rule: &R,
    tree_pair: &FusionTreeBlockKey,
) -> Result<SmallVec<[(FusionTreeBlockKey, R::Scalar); 1]>, CoreError>
where
    R: MultiplicityFreeRigidSymbols,
    R::Scalar: Clone + Mul<Output = R::Scalar>,
{
    let swapped = FusionTreeBlockKey::pair(
        tree_pair.domain_tree().clone(),
        tree_pair.codomain_tree().clone(),
    );
    Ok(multiplicity_free_bendright_tree_pair(rule, &swapped)?
        .into_iter()
        .map(|(bent, coefficient)| {
            (
                FusionTreeBlockKey::pair(bent.domain_tree().clone(), bent.codomain_tree().clone()),
                rule.scalar_conj(coefficient),
            )
        })
        .collect())
}

fn multiplicity_free_foldright_tree_pair<R>(
    rule: &R,
    tree_pair: &FusionTreeBlockKey,
) -> Result<Vec<(FusionTreeBlockKey, R::Scalar)>, CoreError>
where
    R: MultiplicityFreeRigidSymbols,
    R::Scalar: Clone + Add<Output = R::Scalar> + Mul<Output = R::Scalar>,
{
    let codomain = tree_pair.codomain_tree();
    let codomain_rank = codomain.uncoupled().len();
    if codomain_rank == 0 {
        return Err(CoreError::MalformedFusionTree {
            message: "foldright requires at least one codomain leg",
        });
    }
    let a = codomain.uncoupled()[0];
    let is_dual_a = codomain
        .is_dual()
        .first()
        .copied()
        .ok_or(CoreError::MalformedFusionTree {
            message: "codomain tree is missing the first duality flag",
        })?;
    let kappa = rule.frobenius_schur_phase_scalar(a);
    let c = coupled_or_vacuum(rule, codomain);

    let mut terms = FusionTermAccumulator::new();
    for (codomain_prime, coeff1) in multiplicity_free_multi_fmove_tree(rule, codomain)? {
        let b = coupled_or_vacuum(rule, &codomain_prime);
        let a_symbol = rule.a_symbol_scalar(a, b, c);
        let coeff0 = rule.sqrt_dim_scalar(c) * rule.inv_sqrt_dim_scalar(b);
        for (domain_prime, coeff2) in multiplicity_free_multi_fmove_inv_tree(
            rule,
            rule.dual(a),
            b,
            tree_pair.domain_tree(),
            !is_dual_a,
        )? {
            let mut coefficient =
                coeff0.clone() * rule.scalar_conj(coeff2) * a_symbol.clone() * coeff1.clone();
            if is_dual_a {
                coefficient = coefficient * kappa.clone();
            }
            terms.push(
                FusionTreeBlockKey::pair(codomain_prime.clone(), domain_prime),
                coefficient,
            );
        }
    }
    Ok(terms.into_vec())
}

fn multiplicity_free_foldleft_tree_pair<R>(
    rule: &R,
    tree_pair: &FusionTreeBlockKey,
) -> Result<Vec<(FusionTreeBlockKey, R::Scalar)>, CoreError>
where
    R: MultiplicityFreeRigidSymbols,
    R::Scalar: Clone + Add<Output = R::Scalar> + Mul<Output = R::Scalar>,
{
    let swapped = FusionTreeBlockKey::pair(
        tree_pair.domain_tree().clone(),
        tree_pair.codomain_tree().clone(),
    );
    Ok(multiplicity_free_foldright_tree_pair(rule, &swapped)?
        .into_iter()
        .map(|(folded, coefficient)| {
            (
                FusionTreeBlockKey::pair(
                    folded.domain_tree().clone(),
                    folded.codomain_tree().clone(),
                ),
                rule.scalar_conj(coefficient),
            )
        })
        .collect())
}

fn multiplicity_free_cycle_clockwise_tree_pair<R>(
    rule: &R,
    tree_pair: &FusionTreeBlockKey,
) -> Result<Vec<(FusionTreeBlockKey, R::Scalar)>, CoreError>
where
    R: MultiplicityFreeRigidSymbols,
    R::Scalar: Clone + Add<Output = R::Scalar> + Mul<Output = R::Scalar>,
{
    let first: Vec<_> = if tree_pair.codomain_tree().uncoupled().is_empty() {
        multiplicity_free_bendleft_tree_pair(rule, tree_pair)?
            .into_iter()
            .collect()
    } else {
        multiplicity_free_foldright_tree_pair(rule, tree_pair)?
    };
    if tree_pair.codomain_tree().uncoupled().is_empty() {
        compose_tree_pair_terms(rule, first, |rule, key| {
            multiplicity_free_foldright_tree_pair(rule, key)
        })
    } else {
        compose_tree_pair_terms(rule, first, |rule, key| {
            multiplicity_free_bendleft_tree_pair(rule, key)
        })
    }
}

fn multiplicity_free_cycle_anticlockwise_tree_pair<R>(
    rule: &R,
    tree_pair: &FusionTreeBlockKey,
) -> Result<Vec<(FusionTreeBlockKey, R::Scalar)>, CoreError>
where
    R: MultiplicityFreeRigidSymbols,
    R::Scalar: Clone + Add<Output = R::Scalar> + Mul<Output = R::Scalar>,
{
    let first: Vec<_> = if tree_pair.domain_tree().uncoupled().is_empty() {
        multiplicity_free_bendright_tree_pair(rule, tree_pair)?
            .into_iter()
            .collect()
    } else {
        multiplicity_free_foldleft_tree_pair(rule, tree_pair)?
    };
    if tree_pair.domain_tree().uncoupled().is_empty() {
        compose_tree_pair_terms(rule, first, |rule, key| {
            multiplicity_free_foldleft_tree_pair(rule, key)
        })
    } else {
        compose_tree_pair_terms(rule, first, |rule, key| {
            multiplicity_free_bendright_tree_pair(rule, key)
        })
    }
}

fn multiplicity_free_multi_fmove_tree<R>(
    rule: &R,
    tree: &FusionTreeKey,
) -> Result<Vec<(FusionTreeKey, R::Scalar)>, CoreError>
where
    R: MultiplicityFreeRigidSymbols,
    R::Scalar: Clone + Mul<Output = R::Scalar>,
{
    let rank = tree.uncoupled().len();
    if rank == 0 {
        return Err(CoreError::MalformedFusionTree {
            message: "multi_Fmove requires at least one uncoupled sector",
        });
    }
    if rank == 1 {
        return Ok(vec![(
            FusionTreeKey::new(
                Vec::<SectorId>::new(),
                Some(rule.vacuum()),
                Vec::<bool>::new(),
                Vec::<SectorId>::new(),
                Vec::<SectorId>::new(),
            ),
            rule.scalar_one(),
        )]);
    }
    if rank == 2 {
        return Ok(vec![(
            FusionTreeKey::new(
                vec![tree.uncoupled()[1]],
                Some(tree.uncoupled()[1]),
                vec![tree.is_dual()[1]],
                Vec::<SectorId>::new(),
                Vec::<SectorId>::new(),
            ),
            rule.scalar_one(),
        )]);
    }

    let first = tree.uncoupled()[0];
    let coupled = coupled_or_vacuum(rule, tree);
    let tail_uncoupled = &tree.uncoupled()[1..];
    let tail_is_dual = &tree.is_dual()[1..];
    let mut terms = Vec::new();
    for tail_coupled in rule.fusion_channels(rule.dual(first), coupled) {
        let tail_effective = effective_sectors_for_uncoupled(rule, tail_uncoupled, tail_is_dual)?;
        for tail_tree in collect_fusion_trees_for_coupled(
            rule,
            tail_uncoupled,
            tail_is_dual,
            &tail_effective,
            tail_coupled,
        ) {
            if let Some(coefficient) =
                multiplicity_free_multi_associator_scalar(rule, tree, &tail_tree)?
            {
                terms.push((tail_tree, coefficient));
            }
        }
    }
    Ok(terms)
}

fn multiplicity_free_multi_fmove_inv_tree<R>(
    rule: &R,
    leading_sector: SectorId,
    coupled: SectorId,
    tree: &FusionTreeKey,
    leading_is_dual: bool,
) -> Result<Vec<(FusionTreeKey, R::Scalar)>, CoreError>
where
    R: MultiplicityFreeRigidSymbols,
    R::Scalar: Clone + Mul<Output = R::Scalar>,
{
    let tree_coupled = coupled_or_vacuum(rule, tree);
    if rule.nsymbol(leading_sector, tree_coupled, coupled) == 0 {
        return Err(CoreError::SectorMismatch {
            expected: coupled,
            actual: tree_coupled,
        });
    }

    let mut uncoupled = Vec::with_capacity(tree.uncoupled().len() + 1);
    uncoupled.push(leading_sector);
    uncoupled.extend_from_slice(tree.uncoupled());
    let mut is_dual = Vec::with_capacity(tree.is_dual().len() + 1);
    is_dual.push(leading_is_dual);
    is_dual.extend_from_slice(tree.is_dual());
    let effective = effective_sectors_for_uncoupled(rule, &uncoupled, &is_dual)?;
    let candidates =
        collect_fusion_trees_for_coupled(rule, &uncoupled, &is_dual, &effective, coupled);

    let mut terms = Vec::new();
    for candidate in candidates {
        for (short_tree, coefficient) in multiplicity_free_multi_fmove_tree(rule, &candidate)? {
            if fusion_tree_keys_match_with_empty_vacuum(rule, &short_tree, tree) {
                terms.push((candidate.clone(), rule.scalar_conj(coefficient)));
            }
        }
    }
    Ok(terms)
}

fn multiplicity_free_multi_associator_scalar<R>(
    rule: &R,
    long: &FusionTreeKey,
    short: &FusionTreeKey,
) -> Result<Option<R::Scalar>, CoreError>
where
    R: MultiplicityFreeFusionSymbols,
    R::Scalar: Mul<Output = R::Scalar>,
{
    let rank = long.uncoupled().len();
    if short.uncoupled().len() + 1 != rank {
        return Ok(None);
    }
    if long.uncoupled()[1..] != *short.uncoupled() || long.is_dual()[1..] != *short.is_dual() {
        return Ok(None);
    }

    let mut coefficient = rule.scalar_one();
    let first = long.uncoupled()[0];
    for tensor_kit_k in 2..rank {
        let right_sector = long.uncoupled()[tensor_kit_k];
        let (middle_left, middle_right) = fusion_tree_vertex_neighbors(long, tensor_kit_k)?;
        let (short_left, short_right) = fusion_tree_vertex_neighbors(short, tensor_kit_k - 1)?;
        coefficient = coefficient
            * rule.f_symbol_scalar(
                first,
                short_left,
                right_sector,
                middle_right,
                middle_left,
                short_right,
            );
    }
    Ok(Some(coefficient))
}

fn fusion_tree_vertex_neighbors(
    tree: &FusionTreeKey,
    leg_index: usize,
) -> Result<(SectorId, SectorId), CoreError> {
    if leg_index == 0 || leg_index >= tree.uncoupled().len() {
        return Err(CoreError::MalformedFusionTree {
            message: "vertex_info requires a non-first uncoupled leg",
        });
    }
    let left = if leg_index == 1 {
        tree.uncoupled()[0]
    } else {
        tree.innerlines()
            .get(leg_index - 2)
            .copied()
            .ok_or(CoreError::MalformedFusionTree {
                message: "fusion tree is missing a left innerline",
            })?
    };
    let right = if leg_index + 1 == tree.uncoupled().len() {
        coupled_or_vacuum_for_tree(tree)?
    } else {
        tree.innerlines()
            .get(leg_index - 1)
            .copied()
            .ok_or(CoreError::MalformedFusionTree {
                message: "fusion tree is missing a right innerline",
            })?
    };
    Ok((left, right))
}

fn coupled_or_vacuum<R>(rule: &R, tree: &FusionTreeKey) -> SectorId
where
    R: FusionRule,
{
    tree.coupled().unwrap_or_else(|| rule.vacuum())
}

fn coupled_or_vacuum_for_tree(tree: &FusionTreeKey) -> Result<SectorId, CoreError> {
    tree.coupled().ok_or(CoreError::MalformedFusionTree {
        message: "non-empty fusion tree requires a coupled sector",
    })
}

fn effective_sectors_for_uncoupled<R>(
    _rule: &R,
    uncoupled: &[SectorId],
    is_dual: &[bool],
) -> Result<Vec<SectorId>, CoreError>
where
    R: FusionRule,
{
    if uncoupled.len() != is_dual.len() {
        return Err(CoreError::MalformedFusionTree {
            message: "fusion tree sectors and duality flags must have matching length",
        });
    }
    Ok(uncoupled.to_vec())
}

fn fusion_tree_keys_match_with_empty_vacuum<R>(
    rule: &R,
    left: &FusionTreeKey,
    right: &FusionTreeKey,
) -> bool
where
    R: FusionRule,
{
    left.uncoupled() == right.uncoupled()
        && left.is_dual() == right.is_dual()
        && left.innerlines() == right.innerlines()
        && left.vertices() == right.vertices()
        && coupled_or_vacuum(rule, left) == coupled_or_vacuum(rule, right)
}

fn permutation_to_adjacent_swaps(
    permutation: &[usize],
    rank: usize,
) -> Result<Vec<usize>, CoreError> {
    if permutation.len() != rank {
        return Err(CoreError::InvalidPermutation {
            permutation: permutation.to_vec(),
            rank,
        });
    }
    let mut seen = vec![false; rank];
    for &axis in permutation {
        if axis >= rank || seen[axis] {
            return Err(CoreError::InvalidPermutation {
                permutation: permutation.to_vec(),
                rank,
            });
        }
        seen[axis] = true;
    }

    let mut work = permutation.to_vec();
    let mut swaps = Vec::new();
    for target in 0..rank.saturating_sub(1) {
        let source = work[target];
        for swap in (target..source).rev() {
            swaps.push(swap);
        }
        for item in work.iter_mut().take(rank).skip(target + 1) {
            if *item < source {
                *item += 1;
            }
        }
        work[target] = target;
    }
    Ok(swaps)
}

fn inner_extended_sector(tree: &FusionTreeKey, index: usize) -> Result<SectorId, CoreError> {
    let rank = tree.uncoupled().len();
    if index == 0 {
        return tree
            .uncoupled()
            .first()
            .copied()
            .ok_or(CoreError::MalformedFusionTree {
                message: "inner-extended tree requires at least one uncoupled sector",
            });
    }
    if index + 1 == rank {
        return tree.coupled().ok_or(CoreError::MalformedFusionTree {
            message: "inner-extended tree requires a coupled sector",
        });
    }
    tree.innerlines()
        .get(index - 1)
        .copied()
        .ok_or(CoreError::MalformedFusionTree {
            message: "inner-extended tree is missing an innerline",
        })
}

fn only_fusion_channel<R>(rule: &R, left: SectorId, right: SectorId) -> Result<SectorId, CoreError>
where
    R: FusionRule,
{
    let channels = rule.fusion_channels(left, right);
    match channels.as_slice() {
        [sector] => Ok(*sector),
        _ => Err(CoreError::FusionChannelCount {
            left,
            right,
            count: channels.len(),
        }),
    }
}

fn unique_bendright_tree_pair<R>(
    rule: &R,
    tree_pair: &FusionTreeBlockKey,
) -> Result<(FusionTreeBlockKey, R::Scalar), CoreError>
where
    R: MultiplicityFreePivotalSymbols,
{
    let codomain = tree_pair.codomain_tree();
    let domain = tree_pair.domain_tree();
    let codomain_rank = codomain.uncoupled().len();
    if codomain_rank == 0 {
        return Err(CoreError::MalformedFusionTree {
            message: "bendright requires at least one codomain leg",
        });
    }

    let coupled = codomain.coupled().ok_or(CoreError::MalformedFusionTree {
        message: "bendright requires a coupled sector on the codomain tree",
    })?;
    if !domain.uncoupled().is_empty() {
        match domain.coupled() {
            Some(domain_coupled) if domain_coupled == coupled => {}
            _ => {
                return Err(CoreError::MalformedFusionTree {
                    message: "fusion tree pair requires matching coupled sectors",
                });
            }
        }
    }

    let bent_sector = codomain.uncoupled()[codomain_rank - 1];
    let bent_is_dual = codomain.is_dual().get(codomain_rank - 1).copied().ok_or(
        CoreError::MalformedFusionTree {
            message: "codomain tree is missing a duality flag",
        },
    )?;
    let left_coupled = match codomain_rank {
        1 => rule.vacuum(),
        2 => codomain.uncoupled()[0],
        _ => codomain
            .innerlines()
            .last()
            .copied()
            .ok_or(CoreError::MalformedFusionTree {
                message: "bendright requires the last codomain innerline",
            })?,
    };

    let new_codomain_innerlines = if codomain_rank > 2 {
        let innerlines = codomain.innerlines();
        innerlines
            .get(..innerlines.len() - 1)
            .ok_or(CoreError::MalformedFusionTree {
                message: "codomain tree has malformed innerlines",
            })?
            .to_vec()
    } else {
        Vec::new()
    };
    let new_codomain_vertices = if codomain_rank > 1 {
        let vertices = codomain.vertices();
        vertices
            .get(..vertices.len() - 1)
            .ok_or(CoreError::MalformedFusionTree {
                message: "codomain tree has malformed vertices",
            })?
            .to_vec()
    } else {
        Vec::new()
    };
    let new_codomain = FusionTreeKey::new(
        codomain.uncoupled()[..codomain_rank - 1].to_vec(),
        Some(left_coupled),
        codomain.is_dual()[..codomain_rank - 1].to_vec(),
        new_codomain_innerlines,
        new_codomain_vertices,
    );

    let domain_rank = domain.uncoupled().len();
    let mut new_domain_uncoupled = domain.uncoupled().to_vec();
    new_domain_uncoupled.push(rule.dual(bent_sector));
    let mut new_domain_is_dual = domain.is_dual().to_vec();
    new_domain_is_dual.push(!bent_is_dual);
    let mut new_domain_innerlines = domain.innerlines().to_vec();
    if domain_rank > 1 {
        new_domain_innerlines.push(coupled);
    }
    let mut new_domain_vertices = domain.vertices().to_vec();
    if domain_rank > 0 {
        new_domain_vertices.push(SectorId::new(1));
    }
    let new_domain = FusionTreeKey::new(
        new_domain_uncoupled,
        Some(left_coupled),
        new_domain_is_dual,
        new_domain_innerlines,
        new_domain_vertices,
    );

    let coefficient = rule.bendright_scalar(left_coupled, bent_sector, coupled, bent_is_dual);
    Ok((
        FusionTreeBlockKey::pair(new_codomain, new_domain),
        coefficient,
    ))
}

fn unique_bendleft_tree_pair<R>(
    rule: &R,
    tree_pair: &FusionTreeBlockKey,
) -> Result<(FusionTreeBlockKey, R::Scalar), CoreError>
where
    R: MultiplicityFreePivotalSymbols,
{
    let swapped = FusionTreeBlockKey::pair(
        tree_pair.domain_tree().clone(),
        tree_pair.codomain_tree().clone(),
    );
    let (bent, coefficient) = unique_bendright_tree_pair(rule, &swapped)?;
    Ok((
        FusionTreeBlockKey::pair(bent.domain_tree().clone(), bent.codomain_tree().clone()),
        rule.scalar_conj(coefficient),
    ))
}

fn unique_foldright_tree_pair<R>(
    rule: &R,
    tree_pair: &FusionTreeBlockKey,
) -> Result<(FusionTreeBlockKey, R::Scalar), CoreError>
where
    R: MultiplicityFreePivotalSymbols,
{
    let codomain = tree_pair.codomain_tree();
    let codomain_rank = codomain.uncoupled().len();
    if codomain_rank == 0 {
        return Err(CoreError::MalformedFusionTree {
            message: "foldright requires at least one codomain leg",
        });
    }
    let first = codomain.uncoupled()[0];
    let first_is_dual =
        codomain
            .is_dual()
            .first()
            .copied()
            .ok_or(CoreError::MalformedFusionTree {
                message: "codomain tree is missing the first duality flag",
            })?;
    let codomain_prime = unique_multi_fmove_tree(rule, codomain)?;
    let recoupled = codomain_prime
        .coupled()
        .ok_or(CoreError::MalformedFusionTree {
            message: "foldright recoupled codomain tree requires a coupled sector",
        })?;
    let domain_prime = unique_multi_fmove_inv_tree(
        rule,
        rule.dual(first),
        recoupled,
        tree_pair.domain_tree(),
        !first_is_dual,
    )?;
    let destination = FusionTreeBlockKey::pair(codomain_prime, domain_prime);
    let coefficient = rule.foldright_scalar(tree_pair, &destination);
    Ok((destination, coefficient))
}

fn unique_foldleft_tree_pair<R>(
    rule: &R,
    tree_pair: &FusionTreeBlockKey,
) -> Result<(FusionTreeBlockKey, R::Scalar), CoreError>
where
    R: MultiplicityFreePivotalSymbols,
{
    let swapped = FusionTreeBlockKey::pair(
        tree_pair.domain_tree().clone(),
        tree_pair.codomain_tree().clone(),
    );
    let (folded, coefficient) = unique_foldright_tree_pair(rule, &swapped)?;
    Ok((
        FusionTreeBlockKey::pair(folded.domain_tree().clone(), folded.codomain_tree().clone()),
        rule.scalar_conj(coefficient),
    ))
}

fn unique_cycle_clockwise_tree_pair<R>(
    rule: &R,
    tree_pair: &FusionTreeBlockKey,
) -> Result<(FusionTreeBlockKey, R::Scalar), CoreError>
where
    R: MultiplicityFreePivotalSymbols,
    R::Scalar: Mul<Output = R::Scalar>,
{
    let (tmp, first_coefficient) = if tree_pair.codomain_tree().uncoupled().is_empty() {
        unique_bendleft_tree_pair(rule, tree_pair)?
    } else {
        unique_foldright_tree_pair(rule, tree_pair)?
    };
    let (dst, second_coefficient) = if tree_pair.codomain_tree().uncoupled().is_empty() {
        unique_foldright_tree_pair(rule, &tmp)?
    } else {
        unique_bendleft_tree_pair(rule, &tmp)?
    };
    Ok((dst, first_coefficient * second_coefficient))
}

fn unique_cycle_anticlockwise_tree_pair<R>(
    rule: &R,
    tree_pair: &FusionTreeBlockKey,
) -> Result<(FusionTreeBlockKey, R::Scalar), CoreError>
where
    R: MultiplicityFreePivotalSymbols,
    R::Scalar: Mul<Output = R::Scalar>,
{
    let (tmp, first_coefficient) = if tree_pair.domain_tree().uncoupled().is_empty() {
        unique_bendright_tree_pair(rule, tree_pair)?
    } else {
        unique_foldleft_tree_pair(rule, tree_pair)?
    };
    let (dst, second_coefficient) = if tree_pair.domain_tree().uncoupled().is_empty() {
        unique_foldleft_tree_pair(rule, &tmp)?
    } else {
        unique_bendright_tree_pair(rule, &tmp)?
    };
    Ok((dst, first_coefficient * second_coefficient))
}

fn unique_multi_fmove_tree<R>(rule: &R, tree: &FusionTreeKey) -> Result<FusionTreeKey, CoreError>
where
    R: MultiplicityFreeFusionRule,
{
    let first = tree
        .uncoupled()
        .first()
        .copied()
        .ok_or(CoreError::MalformedFusionTree {
            message: "multi_Fmove requires at least one uncoupled sector",
        })?;
    let coupled = tree.coupled().ok_or(CoreError::MalformedFusionTree {
        message: "multi_Fmove requires a coupled sector",
    })?;
    let recoupled = only_fusion_channel(rule, rule.dual(first), coupled)?;
    unique_standard_fusion_tree(
        rule,
        &tree.uncoupled()[1..],
        recoupled,
        &tree.is_dual()[1..],
    )
}

fn unique_multi_fmove_inv_tree<R>(
    rule: &R,
    leading_sector: SectorId,
    coupled: SectorId,
    tree: &FusionTreeKey,
    leading_is_dual: bool,
) -> Result<FusionTreeKey, CoreError>
where
    R: MultiplicityFreeFusionRule,
{
    let mut uncoupled = Vec::with_capacity(tree.uncoupled().len() + 1);
    uncoupled.push(leading_sector);
    uncoupled.extend_from_slice(tree.uncoupled());
    let mut is_dual = Vec::with_capacity(tree.is_dual().len() + 1);
    is_dual.push(leading_is_dual);
    is_dual.extend_from_slice(tree.is_dual());
    unique_standard_fusion_tree(rule, &uncoupled, coupled, &is_dual)
}

fn unique_standard_fusion_tree<R>(
    rule: &R,
    uncoupled: &[SectorId],
    coupled: SectorId,
    is_dual: &[bool],
) -> Result<FusionTreeKey, CoreError>
where
    R: MultiplicityFreeFusionRule,
{
    if uncoupled.len() != is_dual.len() {
        return Err(CoreError::MalformedFusionTree {
            message: "fusion tree sectors and duality flags must have matching length",
        });
    }
    let effective = uncoupled.to_vec();
    let trees = collect_fusion_trees_for_coupled(rule, uncoupled, is_dual, &effective, coupled);
    match trees.as_slice() {
        [tree] => Ok(tree.clone()),
        _ => Err(CoreError::FusionChannelCount {
            left: coupled,
            right: coupled,
            count: trees.len(),
        }),
    }
}

fn linearize_tree_pair_axis(axis: usize, codomain_rank: usize, domain_rank: usize) -> usize {
    if axis < codomain_rank {
        axis
    } else {
        domain_rank + 2 * codomain_rank - 1 - axis
    }
}

fn validate_permutation(permutation: &[usize], rank: usize) -> Result<(), CoreError> {
    if permutation.len() != rank {
        return Err(CoreError::InvalidPermutation {
            permutation: permutation.to_vec(),
            rank,
        });
    }
    let mut seen = vec![false; rank];
    for &axis in permutation {
        if axis >= rank || seen[axis] {
            return Err(CoreError::InvalidPermutation {
                permutation: permutation.to_vec(),
                rank,
            });
        }
        seen[axis] = true;
    }
    Ok(())
}

fn is_cyclic_permutation(permutation: &[usize]) -> bool {
    let rank = permutation.len();
    for index in 0..rank {
        if permutation[(index + 1) % rank] != (permutation[index] + 1) % rank {
            return false;
        }
    }
    true
}

#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct FusionTreeBlockKey {
    codomain_tree: FusionTreeKey,
    domain_tree: FusionTreeKey,
}

impl FusionTreeBlockKey {
    pub fn new(
        uncoupled: Vec<SectorId>,
        coupled: Option<SectorId>,
        vertices: Vec<SectorId>,
    ) -> Self {
        let is_dual = vec![false; uncoupled.len()];
        Self::pair(
            FusionTreeKey::new(uncoupled, coupled, is_dual, Vec::new(), vertices),
            FusionTreeKey::new(
                Vec::<SectorId>::new(),
                None,
                Vec::<bool>::new(),
                Vec::<SectorId>::new(),
                Vec::<SectorId>::new(),
            ),
        )
    }

    pub fn pair(codomain_tree: FusionTreeKey, domain_tree: FusionTreeKey) -> Self {
        Self {
            codomain_tree,
            domain_tree,
        }
    }

    pub fn pair_from_sector_ids<
        Codomain,
        Domain,
        CodomainDual,
        DomainDual,
        CodomainInner,
        DomainInner,
        CodomainVertices,
        DomainVertices,
    >(
        codomain_uncoupled: Codomain,
        domain_uncoupled: Domain,
        coupled: Option<usize>,
        codomain_is_dual: CodomainDual,
        domain_is_dual: DomainDual,
        codomain_innerlines: CodomainInner,
        domain_innerlines: DomainInner,
        codomain_vertices: CodomainVertices,
        domain_vertices: DomainVertices,
    ) -> Self
    where
        Codomain: IntoIterator<Item = usize>,
        Domain: IntoIterator<Item = usize>,
        CodomainDual: IntoIterator<Item = bool>,
        DomainDual: IntoIterator<Item = bool>,
        CodomainInner: IntoIterator<Item = usize>,
        DomainInner: IntoIterator<Item = usize>,
        CodomainVertices: IntoIterator<Item = usize>,
        DomainVertices: IntoIterator<Item = usize>,
    {
        Self::pair(
            FusionTreeKey::from_sector_ids(
                codomain_uncoupled,
                coupled,
                codomain_is_dual,
                codomain_innerlines,
                codomain_vertices,
            ),
            FusionTreeKey::from_sector_ids(
                domain_uncoupled,
                coupled,
                domain_is_dual,
                domain_innerlines,
                domain_vertices,
            ),
        )
    }

    pub fn from_uncoupled<I>(uncoupled: I) -> Self
    where
        I: IntoIterator<Item = SectorId>,
    {
        Self::pair(
            FusionTreeKey::from_uncoupled(uncoupled),
            FusionTreeKey::new(
                Vec::<SectorId>::new(),
                None,
                Vec::<bool>::new(),
                Vec::<SectorId>::new(),
                Vec::<SectorId>::new(),
            ),
        )
    }

    #[inline]
    pub fn codomain_tree(&self) -> &FusionTreeKey {
        &self.codomain_tree
    }

    #[inline]
    pub fn domain_tree(&self) -> &FusionTreeKey {
        &self.domain_tree
    }

    #[inline]
    pub fn uncoupled(&self) -> &[SectorId] {
        self.codomain_tree.uncoupled()
    }

    #[inline]
    pub fn codomain_uncoupled(&self) -> &[SectorId] {
        self.codomain_tree.uncoupled()
    }

    #[inline]
    pub fn domain_uncoupled(&self) -> &[SectorId] {
        self.domain_tree.uncoupled()
    }

    #[inline]
    pub fn coupled(&self) -> Option<SectorId> {
        self.codomain_tree.coupled()
    }

    #[inline]
    pub fn vertices(&self) -> &[SectorId] {
        self.codomain_tree.vertices()
    }

    #[inline]
    pub fn codomain_vertices(&self) -> &[SectorId] {
        self.codomain_tree.vertices()
    }

    #[inline]
    pub fn domain_vertices(&self) -> &[SectorId] {
        self.domain_tree.vertices()
    }

    #[inline]
    pub fn codomain_innerlines(&self) -> &[SectorId] {
        self.codomain_tree.innerlines()
    }

    #[inline]
    pub fn domain_innerlines(&self) -> &[SectorId] {
        self.domain_tree.innerlines()
    }

    #[inline]
    pub fn codomain_is_dual(&self) -> &[bool] {
        self.codomain_tree.is_dual()
    }

    #[inline]
    pub fn domain_is_dual(&self) -> &[bool] {
        self.domain_tree.is_dual()
    }

    pub fn external_sectors<R>(&self, rule: &R) -> Vec<SectorId>
    where
        R: FusionRule,
    {
        let mut sectors = Vec::with_capacity(
            self.codomain_tree.uncoupled().len() + self.domain_tree.uncoupled().len(),
        );
        sectors.extend(self.codomain_tree.uncoupled().iter().copied());
        sectors.extend(
            self.domain_tree
                .uncoupled()
                .iter()
                .copied()
                .map(|sector| rule.dual(sector)),
        );
        sectors
    }

    pub fn external_sector<R>(&self, rule: &R, axis: usize) -> Option<SectorId>
    where
        R: FusionRule,
    {
        let codomain_len = self.codomain_tree.uncoupled().len();
        if axis < codomain_len {
            self.codomain_tree.uncoupled().get(axis).copied()
        } else {
            self.domain_tree
                .uncoupled()
                .get(axis.checked_sub(codomain_len)?)
                .copied()
                .map(|sector| rule.dual(sector))
        }
    }

    pub fn external_is_dual(&self) -> Vec<bool> {
        let mut is_dual = Vec::with_capacity(
            self.codomain_tree.is_dual().len() + self.domain_tree.is_dual().len(),
        );
        is_dual.extend(self.codomain_tree.is_dual().iter().copied());
        is_dual.extend(self.domain_tree.is_dual().iter().copied());
        is_dual
    }

    pub fn group_key(&self) -> FusionTreeGroupKey {
        FusionTreeGroupKey::new(
            self.codomain_tree.uncoupled().iter().copied(),
            self.domain_tree.uncoupled().iter().copied(),
            self.codomain_tree.is_dual().iter().copied(),
            self.domain_tree.is_dual().iter().copied(),
        )
    }

    fn compact_id(&self) -> Option<usize> {
        if self.domain_tree.uncoupled().is_empty()
            && self.domain_tree.coupled().is_none()
            && self.domain_tree.innerlines().is_empty()
            && self.domain_tree.vertices().is_empty()
        {
            self.codomain_tree.compact_id()
        } else {
            None
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub enum BlockKey {
    Dense,
    FusionTree(FusionTreeBlockKey),
}

impl BlockKey {
    pub fn trivial() -> Self {
        Self::Dense
    }

    pub fn sectors<I>(sectors: I) -> Self
    where
        I: IntoIterator<Item = SectorId>,
    {
        Self::FusionTree(FusionTreeBlockKey::from_uncoupled(sectors))
    }

    pub fn sector_ids<I>(sector_ids: I) -> Self
    where
        I: IntoIterator<Item = usize>,
    {
        Self::sectors(sector_ids.into_iter().map(SectorId::new))
    }

    pub fn ordinal(index: usize) -> Self {
        Self::sector_ids([index])
    }

    fn compact_id(&self) -> Option<usize> {
        match self {
            Self::Dense => Some(0),
            Self::FusionTree(tree) => tree.compact_id(),
        }
    }

    pub fn fusion_tree_group_key(&self) -> Option<FusionTreeGroupKey> {
        match self {
            Self::Dense => None,
            Self::FusionTree(tree) => Some(tree.group_key()),
        }
    }
}

impl From<FusionTreeBlockKey> for BlockKey {
    fn from(value: FusionTreeBlockKey) -> Self {
        Self::FusionTree(value)
    }
}

impl<const N: usize> From<[SectorId; N]> for BlockKey {
    fn from(value: [SectorId; N]) -> Self {
        Self::sectors(value)
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct BlockSpec {
    key: BlockKey,
    shape: DimVec,
    strides: DimVec,
    offset: usize,
}

impl BlockSpec {
    pub fn new(shape: Vec<usize>, strides: Vec<usize>, offset: usize) -> Result<Self, CoreError> {
        Self::with_key(BlockKey::trivial(), shape, strides, offset)
    }

    pub fn with_key(
        key: BlockKey,
        shape: Vec<usize>,
        strides: Vec<usize>,
        offset: usize,
    ) -> Result<Self, CoreError> {
        if shape.len() != strides.len() {
            return Err(CoreError::RankMismatch {
                shape: shape.len(),
                strides: strides.len(),
            });
        }
        storage_end_exclusive(&shape, &strides, offset)?;
        Ok(Self {
            key,
            shape: shape.into_iter().collect(),
            strides: strides.into_iter().collect(),
            offset,
        })
    }

    pub fn column_major(shape: Vec<usize>, offset: usize) -> Result<Self, CoreError> {
        Self::column_major_with_key(BlockKey::trivial(), shape, offset)
    }

    pub fn column_major_with_key(
        key: BlockKey,
        shape: Vec<usize>,
        offset: usize,
    ) -> Result<Self, CoreError> {
        let strides = column_major_strides(&shape)?;
        Self::with_key(key, shape, strides, offset)
    }

    #[inline]
    pub fn key(&self) -> &BlockKey {
        &self.key
    }

    #[inline]
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    #[inline]
    pub fn strides(&self) -> &[usize] {
        &self.strides
    }

    #[inline]
    pub fn offset(&self) -> usize {
        self.offset
    }

    pub fn element_count(&self) -> Result<usize, CoreError> {
        checked_product(&self.shape)
    }

    pub fn storage_end_exclusive(&self) -> Result<usize, CoreError> {
        storage_end_exclusive(&self.shape, &self.strides, self.offset)
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SectorBlock {
    key: BlockKey,
}

impl SectorBlock {
    pub fn new(key: BlockKey) -> Self {
        Self { key }
    }

    #[inline]
    pub fn key(&self) -> &BlockKey {
        &self.key
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct FusionTreeBlockGroup {
    group_key: FusionTreeGroupKey,
    block_indices: DimVec,
}

impl FusionTreeBlockGroup {
    pub fn new(group_key: FusionTreeGroupKey, block_indices: Vec<usize>) -> Self {
        Self {
            group_key,
            block_indices: block_indices.into_iter().collect(),
        }
    }

    #[inline]
    pub fn group_key(&self) -> &FusionTreeGroupKey {
        &self.group_key
    }

    #[inline]
    pub fn block_indices(&self) -> &[usize] {
        &self.block_indices
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SectorStructure {
    rank: usize,
    blocks: Vec<SectorBlock>,
    sorted_indices: DimVec,
    compact_lookup: Option<CompactBlockLookup>,
}

impl SectorStructure {
    pub fn dense(rank: usize) -> Self {
        Self::from_keys(rank, [BlockKey::trivial()]).expect("dense sector key is unique")
    }

    pub fn empty(rank: usize) -> Self {
        Self {
            rank,
            blocks: Vec::new(),
            sorted_indices: DimVec::new(),
            compact_lookup: None,
        }
    }

    pub fn from_keys<I, K>(rank: usize, keys: I) -> Result<Self, CoreError>
    where
        I: IntoIterator<Item = K>,
        K: Into<BlockKey>,
    {
        let mut blocks = Vec::new();
        for key in keys {
            let key = key.into();
            blocks.push(SectorBlock::new(key));
        }
        let mut sorted_indices = (0..blocks.len()).collect::<DimVec>();
        sorted_indices
            .sort_unstable_by(|&left, &right| blocks[left].key().cmp(blocks[right].key()));
        for pair in sorted_indices.windows(2) {
            let left = blocks[pair[0]].key();
            let right = blocks[pair[1]].key();
            if left == right {
                return Err(CoreError::DuplicateBlockKey { key: left.clone() });
            }
        }
        let compact_lookup = CompactBlockLookup::from_blocks(&blocks);
        Ok(Self {
            rank,
            blocks,
            sorted_indices,
            compact_lookup,
        })
    }

    #[inline]
    pub fn rank(&self) -> usize {
        self.rank
    }

    #[inline]
    pub fn block_count(&self) -> usize {
        self.blocks.len()
    }

    #[inline]
    pub fn blocks(&self) -> &[SectorBlock] {
        &self.blocks
    }

    pub fn fusion_tree_groups(&self) -> Vec<FusionTreeBlockGroup> {
        let mut groups = Vec::<FusionTreeBlockGroup>::new();
        let mut group_indices = FxHashMap::<FusionTreeGroupKey, usize>::default();
        for (index, block) in self.blocks.iter().enumerate() {
            let Some(group_key) = block.key().fusion_tree_group_key() else {
                continue;
            };
            if let Some(&group_index) = group_indices.get(&group_key) {
                groups[group_index].block_indices.push(index);
            } else {
                group_indices.insert(group_key.clone(), groups.len());
                groups.push(FusionTreeBlockGroup::new(group_key, vec![index]));
            }
        }
        groups
    }

    pub fn block(&self, index: usize) -> Result<&SectorBlock, CoreError> {
        self.blocks
            .get(index)
            .ok_or(CoreError::BlockIndexOutOfBounds {
                index,
                count: self.blocks.len(),
            })
    }

    pub fn key(&self, index: usize) -> Result<&BlockKey, CoreError> {
        Ok(self.block(index)?.key())
    }

    pub fn find_index(&self, key: &BlockKey) -> Option<usize> {
        if let (Some(lookup), Some(id)) = (&self.compact_lookup, key.compact_id()) {
            if let Some(index) = lookup.get(id) {
                return Some(index);
            }
        }
        self.sorted_indices
            .binary_search_by(|&index| self.blocks[index].key().cmp(key))
            .ok()
            .map(|position| self.sorted_indices[position])
    }

    pub fn find_fusion_tree_index(&self, key: &FusionTreeBlockKey) -> Option<usize> {
        if let (Some(lookup), Some(id)) = (&self.compact_lookup, key.compact_id()) {
            if let Some(index) = lookup.get(id) {
                return Some(index);
            }
        }
        self.sorted_indices
            .binary_search_by(|&index| match self.blocks[index].key() {
                BlockKey::Dense => std::cmp::Ordering::Less,
                BlockKey::FusionTree(tree) => tree.cmp(key),
            })
            .ok()
            .map(|position| self.sorted_indices[position])
    }

    #[inline]
    pub fn has_compact_lookup(&self) -> bool {
        self.compact_lookup.is_some()
    }

    #[inline]
    pub fn sorted_indices(&self) -> &[usize] {
        &self.sorted_indices
    }

    pub fn pair_indices_from(&self, src: &Self) -> Result<Vec<usize>, CoreError> {
        if self.block_count() != src.block_count() {
            return Err(CoreError::BlockCountMismatch {
                expected: self.block_count(),
                actual: src.block_count(),
            });
        }
        if let Some(src_lookup) = &src.compact_lookup {
            if self
                .blocks
                .iter()
                .all(|block| block.key().compact_id().is_some())
            {
                return self
                    .blocks
                    .iter()
                    .map(|block| {
                        let id = block.key().compact_id().expect("checked above");
                        src_lookup
                            .get(id)
                            .ok_or_else(|| CoreError::MissingBlockKey {
                                key: block.key().clone(),
                            })
                    })
                    .collect();
            }
        }
        self.pair_indices_from_sorted(src)
    }

    fn pair_indices_from_sorted(&self, src: &Self) -> Result<Vec<usize>, CoreError> {
        let mut src_for_dst = vec![usize::MAX; self.block_count()];
        let mut dst_pos = 0usize;
        let mut src_pos = 0usize;
        while dst_pos < self.sorted_indices.len() && src_pos < src.sorted_indices.len() {
            let dst_index = self.sorted_indices[dst_pos];
            let src_index = src.sorted_indices[src_pos];
            let dst_key = self.blocks[dst_index].key();
            let src_key = src.blocks[src_index].key();
            match dst_key.cmp(src_key) {
                std::cmp::Ordering::Less => {
                    return Err(CoreError::MissingBlockKey {
                        key: dst_key.clone(),
                    });
                }
                std::cmp::Ordering::Greater => {
                    return Err(CoreError::MissingBlockKey {
                        key: src_key.clone(),
                    });
                }
                std::cmp::Ordering::Equal => {
                    src_for_dst[dst_index] = src_index;
                    dst_pos += 1;
                    src_pos += 1;
                }
            }
        }
        if dst_pos < self.sorted_indices.len() {
            let dst_index = self.sorted_indices[dst_pos];
            return Err(CoreError::MissingBlockKey {
                key: self.blocks[dst_index].key().clone(),
            });
        }
        if src_pos < src.sorted_indices.len() {
            let src_index = src.sorted_indices[src_pos];
            return Err(CoreError::MissingBlockKey {
                key: src.blocks[src_index].key().clone(),
            });
        }
        Ok(src_for_dst)
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct CompactBlockLookup {
    indices: DimVec,
}

impl CompactBlockLookup {
    const MISSING: usize = usize::MAX;

    fn from_blocks(blocks: &[SectorBlock]) -> Option<Self> {
        let mut max_id = 0usize;
        let mut ids = Vec::with_capacity(blocks.len());
        for block in blocks {
            let id = block.key().compact_id()?;
            max_id = max_id.max(id);
            ids.push(id);
        }
        let len = max_id.checked_add(1)?;
        if len > blocks.len().saturating_mul(4).max(1) {
            return None;
        }
        let mut indices = DimVec::from_elem(Self::MISSING, len);
        for (index, id) in ids.into_iter().enumerate() {
            if indices[id] != Self::MISSING {
                return None;
            }
            indices[id] = index;
        }
        Some(Self { indices })
    }

    fn get(&self, id: usize) -> Option<usize> {
        self.indices
            .get(id)
            .copied()
            .filter(|&index| index != Self::MISSING)
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DegeneracyBlock {
    shape: DimVec,
    strides: DimVec,
    offset: usize,
}

impl DegeneracyBlock {
    pub fn new(shape: DimVec, strides: DimVec, offset: usize) -> Result<Self, CoreError> {
        if shape.len() != strides.len() {
            return Err(CoreError::RankMismatch {
                shape: shape.len(),
                strides: strides.len(),
            });
        }
        storage_end_exclusive(&shape, &strides, offset)?;
        Ok(Self {
            shape,
            strides,
            offset,
        })
    }

    pub fn column_major(shape: Vec<usize>, offset: usize) -> Result<Self, CoreError> {
        let strides = column_major_strides(&shape)?;
        Self::new(
            shape.into_iter().collect(),
            strides.into_iter().collect(),
            offset,
        )
    }

    #[inline]
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    #[inline]
    pub fn strides(&self) -> &[usize] {
        &self.strides
    }

    #[inline]
    pub fn offset(&self) -> usize {
        self.offset
    }

    pub fn element_count(&self) -> Result<usize, CoreError> {
        checked_product(&self.shape)
    }

    pub fn storage_end_exclusive(&self) -> Result<usize, CoreError> {
        storage_end_exclusive(&self.shape, &self.strides, self.offset)
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DegeneracyStructure {
    rank: usize,
    blocks: Vec<DegeneracyBlock>,
}

impl DegeneracyStructure {
    pub fn packed_column_major<I>(rank: usize, shapes: I) -> Result<Self, CoreError>
    where
        I: IntoIterator,
        I::Item: Into<Vec<usize>>,
    {
        let mut offset = 0usize;
        let mut blocks = Vec::new();
        for shape in shapes {
            let shape = shape.into();
            if shape.len() != rank {
                return Err(CoreError::StructureRankMismatch {
                    expected: rank,
                    actual: shape.len(),
                });
            }
            let block = DegeneracyBlock::column_major(shape, offset)?;
            offset = block.storage_end_exclusive()?;
            blocks.push(block);
        }
        Self::from_blocks_with_rank(rank, blocks)
    }

    pub fn from_blocks_with_rank(
        rank: usize,
        blocks: Vec<DegeneracyBlock>,
    ) -> Result<Self, CoreError> {
        for block in &blocks {
            if block.shape().len() != rank {
                return Err(CoreError::StructureRankMismatch {
                    expected: rank,
                    actual: block.shape().len(),
                });
            }
            block.storage_end_exclusive()?;
        }
        Ok(Self { rank, blocks })
    }

    #[inline]
    pub fn rank(&self) -> usize {
        self.rank
    }

    #[inline]
    pub fn block_count(&self) -> usize {
        self.blocks.len()
    }

    #[inline]
    pub fn blocks(&self) -> &[DegeneracyBlock] {
        &self.blocks
    }

    pub fn block(&self, index: usize) -> Result<&DegeneracyBlock, CoreError> {
        self.blocks
            .get(index)
            .ok_or(CoreError::BlockIndexOutOfBounds {
                index,
                count: self.blocks.len(),
            })
    }

    pub fn required_len(&self) -> Result<usize, CoreError> {
        self.blocks.iter().try_fold(0usize, |required, block| {
            Ok(required.max(block.storage_end_exclusive()?))
        })
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct BlockRef<'a> {
    key: &'a BlockKey,
    degeneracy: &'a DegeneracyBlock,
}

impl<'a> BlockRef<'a> {
    #[inline]
    pub fn key(&self) -> &'a BlockKey {
        self.key
    }

    #[inline]
    pub fn shape(&self) -> &'a [usize] {
        self.degeneracy.shape()
    }

    #[inline]
    pub fn strides(&self) -> &'a [usize] {
        self.degeneracy.strides()
    }

    #[inline]
    pub fn offset(&self) -> usize {
        self.degeneracy.offset()
    }

    pub fn element_count(&self) -> Result<usize, CoreError> {
        self.degeneracy.element_count()
    }

    pub fn storage_end_exclusive(&self) -> Result<usize, CoreError> {
        self.degeneracy.storage_end_exclusive()
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub struct BlockStructureContentBlock {
    key: BlockKey,
    shape: DimVec,
    strides: DimVec,
    offset: usize,
}

impl BlockStructureContentBlock {
    #[inline]
    pub fn key(&self) -> &BlockKey {
        &self.key
    }

    #[inline]
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    #[inline]
    pub fn strides(&self) -> &[usize] {
        &self.strides
    }

    #[inline]
    pub fn offset(&self) -> usize {
        self.offset
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct BlockStructureContent {
    id: usize,
    rank: usize,
    blocks: Arc<[BlockStructureContentBlock]>,
}

impl BlockStructureContent {
    #[inline]
    pub fn id(&self) -> usize {
        self.id
    }

    #[inline]
    pub fn rank(&self) -> usize {
        self.rank
    }

    #[inline]
    pub fn blocks(&self) -> &[BlockStructureContentBlock] {
        &self.blocks
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Hash)]
struct BlockStructureInternKey {
    rank: usize,
    blocks: Arc<[BlockStructureContentBlock]>,
}

fn block_structure_intern_table(
) -> &'static RwLock<FxHashMap<BlockStructureInternKey, Arc<BlockStructureContent>>> {
    static TABLE: OnceLock<RwLock<FxHashMap<BlockStructureInternKey, Arc<BlockStructureContent>>>> =
        OnceLock::new();
    TABLE.get_or_init(|| RwLock::new(FxHashMap::default()))
}

fn intern_block_structure_content(
    sector: &SectorStructure,
    degeneracy: &DegeneracyStructure,
) -> Arc<BlockStructureContent> {
    let mut blocks = Vec::with_capacity(sector.block_count());
    for index in 0..sector.block_count() {
        let sector_key = sector
            .key(index)
            .expect("validated block structure sector index");
        let block = degeneracy
            .block(index)
            .expect("validated block structure degeneracy index");
        blocks.push(BlockStructureContentBlock {
            key: sector_key.clone(),
            shape: block.shape().iter().copied().collect(),
            strides: block.strides().iter().copied().collect(),
            offset: block.offset(),
        });
    }

    let blocks = Arc::<[BlockStructureContentBlock]>::from(blocks);
    let key = BlockStructureInternKey {
        rank: sector.rank(),
        blocks: Arc::clone(&blocks),
    };
    let table = block_structure_intern_table();
    if let Ok(read) = table.read() {
        if let Some(content) = read.get(&key) {
            return Arc::clone(content);
        }
    }

    let mut write = table
        .write()
        .expect("block structure intern table poisoned");
    if let Some(content) = write.get(&key) {
        return Arc::clone(content);
    }
    let content = Arc::new(BlockStructureContent {
        id: write.len() + 1,
        rank: sector.rank(),
        blocks,
    });
    write.insert(key, Arc::clone(&content));
    content
}

fn block_structure_arc_table() -> &'static RwLock<FxHashMap<usize, Weak<BlockStructure>>> {
    static TABLE: OnceLock<RwLock<FxHashMap<usize, Weak<BlockStructure>>>> = OnceLock::new();
    TABLE.get_or_init(|| RwLock::new(FxHashMap::default()))
}

fn canonicalize_block_structure_arc(structure: Arc<BlockStructure>) -> Arc<BlockStructure> {
    let id = structure.content_id();
    let table = block_structure_arc_table();
    if let Ok(read) = table.read() {
        if let Some(existing) = read.get(&id).and_then(Weak::upgrade) {
            return existing;
        }
    }

    let mut write = table.write().expect("block structure arc table poisoned");
    if let Some(existing) = write.get(&id).and_then(Weak::upgrade) {
        return existing;
    }
    write.insert(id, Arc::downgrade(&structure));
    structure
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct BlockStructure {
    sector: SectorStructure,
    degeneracy: DegeneracyStructure,
    content: Arc<BlockStructureContent>,
    // Cached at construction; replay validation checks this against storage
    // lengths on every call and must not re-scan all blocks.
    required_len: usize,
}

impl BlockStructure {
    pub fn trivial(shape: &[usize]) -> Result<Self, CoreError> {
        Self::from_parts(
            SectorStructure::dense(shape.len()),
            DegeneracyStructure::packed_column_major(shape.len(), [shape.to_vec()])?,
        )
    }

    pub fn empty(rank: usize) -> Self {
        let sector = SectorStructure::empty(rank);
        let degeneracy = DegeneracyStructure {
            rank,
            blocks: Vec::new(),
        };
        let content = intern_block_structure_content(&sector, &degeneracy);
        Self {
            sector,
            degeneracy,
            content,
            required_len: 0,
        }
    }

    pub fn from_blocks(blocks: Vec<BlockSpec>) -> Result<Self, CoreError> {
        let rank = blocks.first().map(|block| block.shape().len()).unwrap_or(0);
        Self::from_blocks_with_rank(rank, blocks)
    }

    pub fn from_blocks_with_rank(rank: usize, blocks: Vec<BlockSpec>) -> Result<Self, CoreError> {
        let keys = blocks
            .iter()
            .map(|block| block.key().clone())
            .collect::<Vec<_>>();
        let degeneracy_blocks = blocks
            .into_iter()
            .map(|block| DegeneracyBlock::new(block.shape, block.strides, block.offset))
            .collect::<Result<Vec<_>, _>>()?;
        Self::from_parts(
            SectorStructure::from_keys(rank, keys)?,
            DegeneracyStructure::from_blocks_with_rank(rank, degeneracy_blocks)?,
        )
    }

    pub fn from_parts(
        sector: SectorStructure,
        degeneracy: DegeneracyStructure,
    ) -> Result<Self, CoreError> {
        if sector.rank() != degeneracy.rank() {
            return Err(CoreError::StructureRankMismatch {
                expected: sector.rank(),
                actual: degeneracy.rank(),
            });
        }
        if sector.block_count() != degeneracy.block_count() {
            return Err(CoreError::BlockCountMismatch {
                expected: sector.block_count(),
                actual: degeneracy.block_count(),
            });
        }
        let required_len = degeneracy.required_len()?;
        let content = intern_block_structure_content(&sector, &degeneracy);
        Ok(Self {
            sector,
            degeneracy,
            content,
            required_len,
        })
    }

    pub fn into_shared(self) -> Arc<Self> {
        canonicalize_block_structure_arc(Arc::new(self))
    }

    pub fn canonicalize_shared(structure: Arc<Self>) -> Arc<Self> {
        canonicalize_block_structure_arc(structure)
    }

    pub fn packed_column_major<I>(rank: usize, shapes: I) -> Result<Self, CoreError>
    where
        I: IntoIterator,
        I::Item: Into<Vec<usize>>,
    {
        let shapes = shapes
            .into_iter()
            .map(Into::into)
            .collect::<Vec<Vec<usize>>>();
        let sector = SectorStructure::from_keys(rank, (0..shapes.len()).map(BlockKey::ordinal))?;
        let degeneracy = DegeneracyStructure::packed_column_major(rank, shapes)?;
        Self::from_parts(sector, degeneracy)
    }

    /// Coupled-sector matrix layout over fusion-tree block keys.
    ///
    /// Blocks are stable-sorted by coupled sector, then each coupled sector is
    /// laid out as one contiguous column-major matrix with the fusion-tree
    /// subblocks as strided views (see
    /// [`FusionTensorMapSpace::from_degeneracy_shapes_coupled`]). Fails when a
    /// coupled sector does not cover its full codomain-tree x domain-tree
    /// grid, because the sector matrix would contain uninitialized holes.
    pub fn coupled_sector_matrix_with_keys<R>(
        rule: &R,
        nout: usize,
        rank: usize,
        blocks: Vec<(FusionTreeBlockKey, Vec<usize>)>,
    ) -> Result<Self, CoreError>
    where
        R: FusionRule,
    {
        let mut blocks = blocks;
        blocks.sort_by_key(|(key, _)| coupled_or_vacuum(rule, key.codomain_tree()).id());
        let (keys, shapes): (Vec<_>, Vec<_>) = blocks.into_iter().unzip();
        let specs = coupled_sector_matrix_block_specs(rule, nout, rank, &keys, &shapes)?;
        Self::from_blocks_with_rank(rank, specs)
    }

    #[inline]
    pub fn rank(&self) -> usize {
        self.sector.rank()
    }

    #[inline]
    pub fn block_count(&self) -> usize {
        self.sector.block_count()
    }

    #[inline]
    pub fn sector_structure(&self) -> &SectorStructure {
        &self.sector
    }

    #[inline]
    pub fn degeneracy_structure(&self) -> &DegeneracyStructure {
        &self.degeneracy
    }

    #[inline]
    pub fn content_id(&self) -> usize {
        self.content.id()
    }

    #[inline]
    pub fn content_key(&self) -> Arc<BlockStructureContent> {
        Arc::clone(&self.content)
    }

    pub fn fusion_tree_groups(&self) -> Vec<FusionTreeBlockGroup> {
        self.sector.fusion_tree_groups()
    }

    pub fn find_block_index_by_key(&self, key: &BlockKey) -> Option<usize> {
        self.sector.find_index(key)
    }

    pub fn find_block_index_by_fusion_tree_key(&self, key: &FusionTreeBlockKey) -> Option<usize> {
        self.sector.find_fusion_tree_index(key)
    }

    pub fn pair_block_indices_from(&self, src: &BlockStructure) -> Result<Vec<usize>, CoreError> {
        self.sector.pair_indices_from(&src.sector)
    }

    pub fn only_block(&self) -> Result<BlockRef<'_>, CoreError> {
        if self.block_count() == 1 {
            self.block(0)
        } else {
            Err(CoreError::BlockCountMismatch {
                expected: 1,
                actual: self.block_count(),
            })
        }
    }

    pub fn block(&self, index: usize) -> Result<BlockRef<'_>, CoreError> {
        Ok(BlockRef {
            key: self.sector.key(index)?,
            degeneracy: self.degeneracy.block(index)?,
        })
    }

    pub fn block_by_key(&self, key: &BlockKey) -> Result<BlockRef<'_>, CoreError> {
        let index = self
            .find_block_index_by_key(key)
            .ok_or_else(|| CoreError::MissingBlockKey { key: key.clone() })?;
        self.block(index)
    }

    pub fn fusion_tree_block(&self, key: &FusionTreeBlockKey) -> Result<BlockRef<'_>, CoreError> {
        let index = self
            .find_block_index_by_fusion_tree_key(key)
            .ok_or_else(|| CoreError::MissingBlockKey {
                key: BlockKey::FusionTree(key.clone()),
            })?;
        self.block(index)
    }

    pub fn required_len(&self) -> Result<usize, CoreError> {
        Ok(self.required_len)
    }
}

#[derive(Clone, Debug)]
pub struct TensorMap<T, const NOUT: usize, const NIN: usize, S = Trivial, D = Vec<T>> {
    storage: D,
    space: TensorMapSpace<NOUT, NIN>,
    structure: Arc<BlockStructure>,
    fusion_space: Option<Arc<FusionTensorMapSpace<NOUT, NIN>>>,
    _marker: PhantomData<(T, S)>,
}

pub type Tensor<T, const N: usize, S = Trivial> = TensorMap<T, N, 0, S>;

impl<T, const NOUT: usize, const NIN: usize, S> TensorMap<T, NOUT, NIN, S, Vec<T>> {
    /// Builds a dense tensor backed by `Vec<T>` and the default trivial block
    /// structure.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenet_core::{TensorMap, TensorMapSpace};
    ///
    /// let space = TensorMapSpace::<1, 1>::from_dims([2], [2]).unwrap();
    /// let tensor = TensorMap::<f64, 1, 1>::from_vec(
    ///     vec![1.0, 3.0, 2.0, 4.0],
    ///     space,
    /// )
    /// .unwrap();
    /// assert_eq!(tensor.data(), &[1.0, 3.0, 2.0, 4.0]);
    /// ```
    pub fn from_vec(data: Vec<T>, space: TensorMapSpace<NOUT, NIN>) -> Result<Self, CoreError> {
        Self::from_vec_with_structure(data, space.clone(), BlockStructure::trivial(space.dims())?)
    }

    /// Builds a dense tensor backed by `Vec<T>` and an explicit block
    /// structure.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenet_core::{BlockStructure, TensorMap, TensorMapSpace, Trivial};
    ///
    /// let space = TensorMapSpace::<1, 0>::from_dims([2], []).unwrap();
    /// let structure = BlockStructure::packed_column_major(1, [vec![2]]).unwrap();
    /// let tensor = TensorMap::<i32, 1, 0>::from_vec_with_structure(
    ///     vec![10, 20],
    ///     space,
    ///     structure,
    /// )
    /// .unwrap();
    /// assert_eq!(tensor.data(), &[10, 20]);
    /// ```
    pub fn from_vec_with_structure(
        data: Vec<T>,
        space: TensorMapSpace<NOUT, NIN>,
        structure: BlockStructure,
    ) -> Result<Self, CoreError> {
        Self::from_vec_with_shared_structure(data, space, structure.into_shared())
    }

    pub fn from_vec_with_shared_structure(
        data: Vec<T>,
        space: TensorMapSpace<NOUT, NIN>,
        structure: Arc<BlockStructure>,
    ) -> Result<Self, CoreError> {
        Self::from_storage_with_shared_structure(data, space, structure)
    }

    /// Builds a symmetric tensor backed by `Vec<T>`.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenet_core::{
    ///     FusionTensorMapSpace, FusionTreeHomSpace, TensorMap, TensorMapSpace,
    ///     Z2FusionRule, Z2Irrep,
    /// };
    ///
    /// let rule = Z2FusionRule;
    /// let fusion_space = FusionTensorMapSpace::from_degeneracy_shapes(
    ///     TensorMapSpace::<1, 1>::from_dims([1], [1]).unwrap(),
    ///     FusionTreeHomSpace::from_sectors([(Z2Irrep::EVEN, 1)], [(Z2Irrep::EVEN, 1)]),
    ///     &rule,
    ///     [vec![1, 1]],
    /// )
    /// .unwrap();
    /// let tensor = TensorMap::<f64, 1, 1>::from_vec_with_fusion_space(
    ///     vec![3.5],
    ///     fusion_space,
    /// )
    /// .unwrap();
    /// assert_eq!(tensor.data(), &[3.5]);
    /// ```
    pub fn from_vec_with_fusion_space(
        data: Vec<T>,
        fusion_space: FusionTensorMapSpace<NOUT, NIN>,
    ) -> Result<Self, CoreError> {
        Self::from_storage_with_fusion_space(data, fusion_space)
    }

    pub fn from_vec_with_shared_fusion_space(
        data: Vec<T>,
        fusion_space: Arc<FusionTensorMapSpace<NOUT, NIN>>,
    ) -> Result<Self, CoreError> {
        Self::from_storage_with_shared_fusion_space(data, fusion_space)
    }

    /// Builds a tensor by evaluating `fill(key, indices)` for every block
    /// element; positions not covered by any block keep `background`.
    /// Layout-independent: packed and coupled spaces produce identical
    /// tensors from the same `fill`.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenet_core::{
    ///     FusionTensorMapSpace, FusionTreeHomSpace, TensorMap, TensorMapSpace,
    ///     Z2FusionRule, Z2Irrep,
    /// };
    ///
    /// let rule = Z2FusionRule;
    /// let fusion_space = FusionTensorMapSpace::from_degeneracy_shapes(
    ///     TensorMapSpace::<1, 1>::from_dims([1], [1]).unwrap(),
    ///     FusionTreeHomSpace::from_sectors([(Z2Irrep::EVEN, 1)], [(Z2Irrep::EVEN, 1)]),
    ///     &rule,
    ///     [vec![1, 1]],
    /// )
    /// .unwrap();
    /// let tensor = TensorMap::<i32, 1, 1>::from_block_fn_with_fusion_space(
    ///     fusion_space,
    ///     0,
    ///     |_key, indices| 10 + indices[0] as i32,
    /// )
    /// .unwrap();
    /// assert_eq!(tensor.data(), &[10]);
    /// ```
    pub fn from_block_fn_with_fusion_space<F>(
        fusion_space: FusionTensorMapSpace<NOUT, NIN>,
        background: T,
        fill: F,
    ) -> Result<Self, CoreError>
    where
        T: Clone,
        F: FnMut(&BlockKey, &[usize]) -> T,
    {
        let len = fusion_space.required_len()?;
        let mut tensor = Self::from_vec_with_fusion_space(vec![background; len], fusion_space)?;
        tensor.fill_block_elements(fill)?;
        Ok(tensor)
    }
}

impl<T: Clone, const NOUT: usize, const NIN: usize, S> TensorMap<T, NOUT, NIN, S, Vec<T>> {
    /// Builds a dense tensor filled with a single value.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenet_core::{TensorMap, TensorMapSpace};
    ///
    /// let space = TensorMapSpace::<2, 0>::from_dims([2, 3], []).unwrap();
    /// let tensor = TensorMap::<f64, 2, 0>::filled(1.25, space).unwrap();
    /// assert_eq!(tensor.data(), &[1.25; 6]);
    /// ```
    pub fn filled(value: T, space: TensorMapSpace<NOUT, NIN>) -> Result<Self, CoreError> {
        Self::from_vec(vec![value; space.dense_dim()], space)
    }
}

impl<T, const NOUT: usize, const NIN: usize, S, D> TensorMap<T, NOUT, NIN, S, D>
where
    D: TensorStorage<T>,
{
    /// Builds a tensor from caller-provided storage and an explicit block
    /// structure.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenet_core::{BlockStructure, TensorMap, TensorMapSpace, Trivial};
    ///
    /// let space = TensorMapSpace::<1, 0>::from_dims([2], []).unwrap();
    /// let structure = BlockStructure::packed_column_major(1, [vec![2]]).unwrap();
    /// let tensor = TensorMap::<i32, 1, 0, Trivial, Vec<i32>>::from_storage_with_structure(
    ///     vec![1, 2],
    ///     space,
    ///     structure,
    /// )
    /// .unwrap();
    /// assert_eq!(tensor.data(), &[1, 2]);
    /// ```
    pub fn from_storage_with_structure(
        storage: D,
        space: TensorMapSpace<NOUT, NIN>,
        structure: BlockStructure,
    ) -> Result<Self, CoreError> {
        Self::from_storage_with_shared_structure(storage, space, structure.into_shared())
    }

    pub fn from_storage_with_shared_structure(
        storage: D,
        space: TensorMapSpace<NOUT, NIN>,
        structure: Arc<BlockStructure>,
    ) -> Result<Self, CoreError> {
        Self::from_storage_parts(
            storage,
            space,
            BlockStructure::canonicalize_shared(structure),
            None,
        )
    }

    /// Builds a symmetric tensor from caller-provided storage.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenet_core::{
    ///     FusionTensorMapSpace, FusionTreeHomSpace, TensorMap, TensorMapSpace,
    ///     Trivial, Z2FusionRule, Z2Irrep,
    /// };
    ///
    /// let rule = Z2FusionRule;
    /// let fusion_space = FusionTensorMapSpace::from_degeneracy_shapes(
    ///     TensorMapSpace::<1, 1>::from_dims([1], [1]).unwrap(),
    ///     FusionTreeHomSpace::from_sectors([(Z2Irrep::EVEN, 1)], [(Z2Irrep::EVEN, 1)]),
    ///     &rule,
    ///     [vec![1, 1]],
    /// )
    /// .unwrap();
    /// let tensor = TensorMap::<f64, 1, 1, Trivial, Vec<f64>>::from_storage_with_fusion_space(
    ///     vec![2.0],
    ///     fusion_space,
    /// )
    /// .unwrap();
    /// assert_eq!(tensor.data(), &[2.0]);
    /// ```
    pub fn from_storage_with_fusion_space(
        storage: D,
        fusion_space: FusionTensorMapSpace<NOUT, NIN>,
    ) -> Result<Self, CoreError> {
        Self::from_storage_with_shared_fusion_space(storage, Arc::new(fusion_space))
    }

    pub fn from_storage_with_shared_fusion_space(
        storage: D,
        fusion_space: Arc<FusionTensorMapSpace<NOUT, NIN>>,
    ) -> Result<Self, CoreError> {
        Self::from_storage_parts(
            storage,
            fusion_space.dense_space().clone(),
            Arc::clone(fusion_space.subblock_structure()),
            Some(fusion_space),
        )
    }

    fn from_storage_parts(
        storage: D,
        space: TensorMapSpace<NOUT, NIN>,
        structure: Arc<BlockStructure>,
        fusion_space: Option<Arc<FusionTensorMapSpace<NOUT, NIN>>>,
    ) -> Result<Self, CoreError> {
        if structure.rank() != space.dims().len() {
            return Err(CoreError::StructureRankMismatch {
                expected: space.dims().len(),
                actual: structure.rank(),
            });
        }
        let required_len = structure.required_len()?;
        if storage.len() != required_len {
            return Err(CoreError::DimensionMismatch {
                expected: required_len,
                actual: storage.len(),
            });
        }
        Ok(Self {
            storage,
            space,
            structure,
            fusion_space,
            _marker: PhantomData,
        })
    }

    #[inline]
    pub fn storage(&self) -> &D {
        &self.storage
    }

    #[inline]
    pub fn storage_mut(&mut self) -> &mut D {
        &mut self.storage
    }

    #[inline]
    pub fn placement(&self) -> Placement {
        self.storage.placement()
    }

    #[inline]
    pub fn similar_storage_filled(&self, len: usize, value: T) -> D::Similar
    where
        D: SimilarStorage<T>,
        T: Clone,
    {
        self.storage.similar_filled(len, value)
    }

    #[inline]
    pub fn space(&self) -> &TensorMapSpace<NOUT, NIN> {
        &self.space
    }

    #[inline]
    pub fn structure(&self) -> &Arc<BlockStructure> {
        &self.structure
    }

    #[inline]
    pub fn fusion_space(&self) -> Option<&Arc<FusionTensorMapSpace<NOUT, NIN>>> {
        self.fusion_space.as_ref()
    }

    #[inline]
    pub fn dim(&self) -> usize {
        self.storage.len()
    }

    #[inline]
    pub fn storage_dim(&self) -> usize {
        self.storage.len()
    }

    /// Full dense element count obtained by multiplying the uncoupled leg dimensions.
    ///
    /// For block-sparse/symmetric tensors this can be larger than the packed storage
    /// length returned by [`Self::dim`] / [`Self::storage_dim`].
    #[inline]
    pub fn dense_dim(&self) -> usize {
        self.space.dense_dim()
    }

    #[inline]
    pub fn dims(&self) -> &[usize] {
        self.space.dims()
    }
}

impl<T, const NOUT: usize, const NIN: usize, S, D> TensorMap<T, NOUT, NIN, S, D>
where
    D: HostReadableStorage<T>,
{
    #[inline]
    pub fn data(&self) -> &[T] {
        self.storage.as_slice()
    }

    /// Visits every block element as `(key, indices, value)`, independent of
    /// the storage layout.
    pub fn for_each_block_element<F>(&self, mut visit: F) -> Result<(), CoreError>
    where
        F: FnMut(&BlockKey, &[usize], &T),
    {
        let structure = Arc::clone(&self.structure);
        let data = self.storage.as_slice();
        for index in 0..structure.block_count() {
            let block = structure.block(index)?;
            let shape = block.shape();
            let strides = block.strides();
            let offset = block.offset();
            let count: usize = shape.iter().product();
            let mut indices = vec![0usize; shape.len()];
            for _ in 0..count {
                let position = offset
                    + indices
                        .iter()
                        .zip(strides)
                        .map(|(&index, &stride)| index * stride)
                        .sum::<usize>();
                visit(block.key(), &indices, &data[position]);
                for axis in 0..shape.len() {
                    indices[axis] += 1;
                    if indices[axis] < shape[axis] {
                        break;
                    }
                    indices[axis] = 0;
                }
            }
        }
        Ok(())
    }

    pub fn subblock(&self) -> Result<BlockView<'_, T>, CoreError> {
        let block = self.structure.only_block()?;
        BlockView::new(
            self.storage.as_slice(),
            block.shape(),
            block.strides(),
            block.offset(),
        )
    }

    pub fn block(&self, index: usize) -> Result<BlockView<'_, T>, CoreError> {
        let block = self.structure.block(index)?;
        BlockView::new(
            self.storage.as_slice(),
            block.shape(),
            block.strides(),
            block.offset(),
        )
    }

    pub fn block_by_key(&self, key: &BlockKey) -> Result<BlockView<'_, T>, CoreError> {
        let block = self.structure.block_by_key(key)?;
        BlockView::new(
            self.storage.as_slice(),
            block.shape(),
            block.strides(),
            block.offset(),
        )
    }

    pub fn subblock_by_tree(
        &self,
        key: &FusionTreeBlockKey,
    ) -> Result<BlockView<'_, T>, CoreError> {
        let block = self.structure.fusion_tree_block(key)?;
        BlockView::new(
            self.storage.as_slice(),
            block.shape(),
            block.strides(),
            block.offset(),
        )
    }

    pub fn subblock_by_sectors<R>(
        &self,
        rule: &R,
        sectors: &[SectorId],
    ) -> Result<BlockView<'_, T>, CoreError>
    where
        R: MultiplicityFreeFusionRule,
    {
        let mut blocks = self.subblocks_by_sectors(rule, sectors)?;
        if blocks.len() != 1 {
            return Err(CoreError::BlockCountMismatch {
                expected: 1,
                actual: blocks.len(),
            });
        }
        Ok(blocks.remove(0))
    }

    pub fn subblocks_by_sectors<R>(
        &self,
        rule: &R,
        sectors: &[SectorId],
    ) -> Result<Vec<BlockView<'_, T>>, CoreError>
    where
        R: MultiplicityFreeFusionRule,
    {
        let fusion_space = self
            .fusion_space
            .as_ref()
            .ok_or(CoreError::MissingFusionSpace)?;
        let keys = fusion_space
            .homspace()
            .fusion_tree_keys_from_external_sectors(rule, sectors)?;
        let mut blocks = Vec::with_capacity(keys.len());
        for key in keys {
            blocks.push(self.subblock_by_tree(&key)?);
        }
        Ok(blocks)
    }
}

impl<T, const NOUT: usize, const NIN: usize, S, D> TensorMap<T, NOUT, NIN, S, D>
where
    D: HostWritableStorage<T>,
{
    #[inline]
    pub fn data_mut(&mut self) -> &mut [T] {
        self.storage.as_mut_slice()
    }

    /// Fills every block element with `fill(key, indices)`, independent of
    /// the storage layout. The layout-safe way to enter data: constructing
    /// through this (instead of positioning raw values in a flat vector)
    /// gives identical tensors for the packed and coupled layouts.
    pub fn fill_block_elements<F>(&mut self, mut fill: F) -> Result<(), CoreError>
    where
        F: FnMut(&BlockKey, &[usize]) -> T,
    {
        let structure = Arc::clone(&self.structure);
        let data = self.storage.as_mut_slice();
        for index in 0..structure.block_count() {
            let block = structure.block(index)?;
            let shape = block.shape();
            let strides = block.strides();
            let offset = block.offset();
            let count: usize = shape.iter().product();
            let mut indices = vec![0usize; shape.len()];
            for _ in 0..count {
                let position = offset
                    + indices
                        .iter()
                        .zip(strides)
                        .map(|(&index, &stride)| index * stride)
                        .sum::<usize>();
                data[position] = fill(block.key(), &indices);
                for axis in 0..shape.len() {
                    indices[axis] += 1;
                    if indices[axis] < shape[axis] {
                        break;
                    }
                    indices[axis] = 0;
                }
            }
        }
        Ok(())
    }

    pub fn subblock_mut(&mut self) -> Result<BlockViewMut<'_, T>, CoreError> {
        let block = self.structure.only_block()?;
        BlockViewMut::new(
            self.storage.as_mut_slice(),
            block.shape(),
            block.strides(),
            block.offset(),
        )
    }

    pub fn block_mut(&mut self, index: usize) -> Result<BlockViewMut<'_, T>, CoreError> {
        let block = self.structure.block(index)?;
        BlockViewMut::new(
            self.storage.as_mut_slice(),
            block.shape(),
            block.strides(),
            block.offset(),
        )
    }

    pub fn block_mut_by_key(&mut self, key: &BlockKey) -> Result<BlockViewMut<'_, T>, CoreError> {
        let block = self.structure.block_by_key(key)?;
        BlockViewMut::new(
            self.storage.as_mut_slice(),
            block.shape(),
            block.strides(),
            block.offset(),
        )
    }

    pub fn subblock_mut_by_tree(
        &mut self,
        key: &FusionTreeBlockKey,
    ) -> Result<BlockViewMut<'_, T>, CoreError> {
        let block = self.structure.fusion_tree_block(key)?;
        BlockViewMut::new(
            self.storage.as_mut_slice(),
            block.shape(),
            block.strides(),
            block.offset(),
        )
    }

    pub fn subblock_mut_by_sectors<R>(
        &mut self,
        rule: &R,
        sectors: &[SectorId],
    ) -> Result<BlockViewMut<'_, T>, CoreError>
    where
        R: MultiplicityFreeFusionRule,
    {
        let fusion_space = self
            .fusion_space
            .as_ref()
            .ok_or(CoreError::MissingFusionSpace)?;
        let key = fusion_space
            .homspace()
            .unique_fusion_tree_key_from_external_sectors(rule, sectors)?;
        self.subblock_mut_by_tree(&key)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct BlockLayout<'a> {
    len: usize,
    offset: usize,
    shape: &'a [usize],
    strides: &'a [usize],
}

impl<'a> BlockLayout<'a> {
    pub fn new(
        len: usize,
        offset: usize,
        shape: &'a [usize],
        strides: &'a [usize],
    ) -> Result<Self, CoreError> {
        let layout = Self {
            len,
            offset,
            shape,
            strides,
        };
        validate_layout(layout)?;
        Ok(layout)
    }

    #[inline]
    pub fn len(self) -> usize {
        self.len
    }

    #[inline]
    pub fn offset(self) -> usize {
        self.offset
    }

    #[inline]
    pub fn shape(self) -> &'a [usize] {
        self.shape
    }

    #[inline]
    pub fn strides(self) -> &'a [usize] {
        self.strides
    }

    #[inline]
    pub fn rank(self) -> usize {
        self.shape.len()
    }

    #[inline]
    pub fn is_empty(self) -> bool {
        self.shape.iter().any(|&dim| dim == 0)
    }
}

#[derive(Debug)]
pub struct BlockView<'a, T> {
    data: &'a [T],
    layout: BlockLayout<'a>,
}

impl<'a, T> Clone for BlockView<'a, T> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<'a, T> Copy for BlockView<'a, T> {}

impl<'a, T> BlockView<'a, T> {
    pub fn new(
        data: &'a [T],
        shape: &'a [usize],
        strides: &'a [usize],
        offset: usize,
    ) -> Result<Self, CoreError> {
        let layout = BlockLayout::new(data.len(), offset, shape, strides)?;
        Ok(Self { data, layout })
    }

    #[inline]
    pub fn data(&self) -> &'a [T] {
        self.data
    }

    #[inline]
    pub fn layout(&self) -> BlockLayout<'a> {
        self.layout
    }

    #[inline]
    pub fn shape(&self) -> &'a [usize] {
        self.layout.shape()
    }

    #[inline]
    pub fn strides(&self) -> &'a [usize] {
        self.layout.strides()
    }

    #[inline]
    pub fn offset(&self) -> usize {
        self.layout.offset()
    }
}

#[derive(Debug)]
pub struct BlockViewMut<'a, T> {
    data: &'a mut [T],
    layout: BlockLayout<'a>,
}

impl<'a, T> BlockViewMut<'a, T> {
    pub fn new(
        data: &'a mut [T],
        shape: &'a [usize],
        strides: &'a [usize],
        offset: usize,
    ) -> Result<Self, CoreError> {
        let layout = BlockLayout::new(data.len(), offset, shape, strides)?;
        Ok(Self { data, layout })
    }

    #[inline]
    pub fn data(&self) -> &[T] {
        self.data
    }

    #[inline]
    pub fn data_mut(&mut self) -> &mut [T] {
        self.data
    }

    #[inline]
    pub fn layout(&self) -> BlockLayout<'a> {
        self.layout
    }

    #[inline]
    pub fn shape(&self) -> &'a [usize] {
        self.layout.shape()
    }

    #[inline]
    pub fn strides(&self) -> &'a [usize] {
        self.layout.strides()
    }

    #[inline]
    pub fn offset(&self) -> usize {
        self.layout.offset()
    }

    #[inline]
    pub fn into_parts(self) -> (&'a mut [T], BlockLayout<'a>) {
        (self.data, self.layout)
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum CoreError {
    RankMismatch {
        shape: usize,
        strides: usize,
    },
    StructureRankMismatch {
        expected: usize,
        actual: usize,
    },
    DimensionMismatch {
        expected: usize,
        actual: usize,
    },
    InvalidBraidIndex {
        index: usize,
        rank: usize,
    },
    InvalidPermutation {
        permutation: Vec<usize>,
        rank: usize,
    },
    UnsupportedFusionStyle {
        expected: FusionStyleKind,
        actual: FusionStyleKind,
    },
    UnsupportedBraidingStyle {
        expected: &'static str,
        actual: BraidingStyleKind,
    },
    UnsupportedSectorBraid {
        left: SectorId,
        right: SectorId,
        style: BraidingStyleKind,
    },
    InvalidSector {
        sector: SectorId,
    },
    SectorMismatch {
        expected: SectorId,
        actual: SectorId,
    },
    /// A per-sector leg degeneracy disagrees with another authoritative
    /// source (the paired leg of a composition, or a fusion-tree degeneracy
    /// shape validated against its leg).
    LegDegeneracyMismatch {
        sector: SectorId,
        expected: usize,
        actual: usize,
    },
    FusionChannelCount {
        left: SectorId,
        right: SectorId,
        count: usize,
    },
    MalformedFusionTree {
        message: &'static str,
    },
    BlockCountMismatch {
        expected: usize,
        actual: usize,
    },
    BlockIndexOutOfBounds {
        index: usize,
        count: usize,
    },
    DuplicateBlockKey {
        key: BlockKey,
    },
    MissingBlockKey {
        key: BlockKey,
    },
    MissingFusionSpace,
    ElementCountOverflow,
    OffsetOverflow {
        value: usize,
    },
    StrideOverflow {
        value: usize,
    },
    OutOfBounds,
}

impl fmt::Display for CoreError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::RankMismatch { shape, strides } => {
                write!(
                    f,
                    "rank mismatch: shape rank {shape}, strides rank {strides}"
                )
            }
            Self::StructureRankMismatch { expected, actual } => {
                write!(
                    f,
                    "block structure rank mismatch: expected {expected}, got {actual}"
                )
            }
            Self::DimensionMismatch { expected, actual } => {
                write!(f, "dimension mismatch: expected {expected}, got {actual}")
            }
            Self::InvalidBraidIndex { index, rank } => {
                write!(
                    f,
                    "cannot braid adjacent fusion-tree outputs at index {index} for rank {rank}"
                )
            }
            Self::InvalidPermutation { permutation, rank } => {
                write!(f, "invalid permutation {permutation:?} for rank {rank}")
            }
            Self::UnsupportedFusionStyle { expected, actual } => {
                write!(
                    f,
                    "unsupported fusion style {actual:?}; expected {expected:?}"
                )
            }
            Self::UnsupportedBraidingStyle { expected, actual } => {
                write!(
                    f,
                    "unsupported braiding style {actual:?}; expected {expected}"
                )
            }
            Self::UnsupportedSectorBraid { left, right, style } => {
                write!(
                    f,
                    "cannot braid non-unit sectors {left:?} and {right:?} with braiding style {style:?}"
                )
            }
            Self::InvalidSector { sector } => write!(f, "invalid sector {sector:?}"),
            Self::SectorMismatch { expected, actual } => {
                write!(f, "sector mismatch: expected {expected:?}, got {actual:?}")
            }
            Self::LegDegeneracyMismatch {
                sector,
                expected,
                actual,
            } => {
                write!(
                    f,
                    "leg degeneracy mismatch for sector {sector:?}: expected {expected}, got {actual}"
                )
            }
            Self::FusionChannelCount { left, right, count } => {
                write!(
                    f,
                    "expected one fusion channel for {left:?} x {right:?}, got {count}"
                )
            }
            Self::MalformedFusionTree { message } => {
                write!(f, "malformed fusion tree: {message}")
            }
            Self::BlockCountMismatch { expected, actual } => {
                write!(f, "block count mismatch: expected {expected}, got {actual}")
            }
            Self::BlockIndexOutOfBounds { index, count } => {
                write!(f, "block index {index} is out of bounds for {count} blocks")
            }
            Self::DuplicateBlockKey { key } => {
                write!(f, "duplicate block key {key:?}")
            }
            Self::MissingBlockKey { key } => {
                write!(f, "missing matching block for key {key:?}")
            }
            Self::MissingFusionSpace => write!(f, "tensor does not carry a fusion-tree space"),
            Self::ElementCountOverflow => write!(f, "block element count overflow"),
            Self::OffsetOverflow { value } => {
                write!(f, "block offset {value} overflows addressable layout")
            }
            Self::StrideOverflow { value } => {
                write!(f, "block stride {value} overflows addressable layout")
            }
            Self::OutOfBounds => write!(f, "block view accesses outside the buffer"),
        }
    }
}

impl std::error::Error for CoreError {}

pub fn validate_layout(layout: BlockLayout<'_>) -> Result<(), CoreError> {
    if layout.shape.len() != layout.strides.len() {
        return Err(CoreError::RankMismatch {
            shape: layout.shape.len(),
            strides: layout.strides.len(),
        });
    }
    if layout.is_empty() {
        return if layout.offset <= layout.len {
            Ok(())
        } else {
            Err(CoreError::OutOfBounds)
        };
    }
    if layout.offset >= layout.len {
        return Err(CoreError::OutOfBounds);
    }
    let max_delta = max_offset_delta(layout.shape, layout.strides)?;
    let last = layout
        .offset
        .checked_add(max_delta)
        .ok_or(CoreError::OffsetOverflow {
            value: layout.offset,
        })?;
    if last < layout.len {
        Ok(())
    } else {
        Err(CoreError::OutOfBounds)
    }
}

fn max_offset_delta(shape: &[usize], strides: &[usize]) -> Result<usize, CoreError> {
    shape
        .iter()
        .zip(strides)
        .try_fold(0usize, |acc, (&dim, &stride)| {
            let steps = dim.saturating_sub(1);
            let delta = steps
                .checked_mul(stride)
                .ok_or(CoreError::StrideOverflow { value: stride })?;
            acc.checked_add(delta)
                .ok_or(CoreError::ElementCountOverflow)
        })
}

fn storage_end_exclusive(
    shape: &[usize],
    strides: &[usize],
    offset: usize,
) -> Result<usize, CoreError> {
    if shape.len() != strides.len() {
        return Err(CoreError::RankMismatch {
            shape: shape.len(),
            strides: strides.len(),
        });
    }
    if shape.iter().any(|&dim| dim == 0) {
        return Ok(offset);
    }
    let max_delta = max_offset_delta(shape, strides)?;
    offset
        .checked_add(max_delta)
        .and_then(|last| last.checked_add(1))
        .ok_or(CoreError::OffsetOverflow { value: offset })
}

fn checked_product(dims: &[usize]) -> Result<usize, CoreError> {
    dims.iter().try_fold(1usize, |acc, &dim| {
        acc.checked_mul(dim).ok_or(CoreError::ElementCountOverflow)
    })
}

fn column_major_strides(shape: &[usize]) -> Result<Vec<usize>, CoreError> {
    let mut strides = vec![1usize; shape.len()];
    for index in 1..shape.len() {
        strides[index] = strides[index - 1]
            .checked_mul(shape[index - 1])
            .ok_or(CoreError::ElementCountOverflow)?;
    }
    Ok(strides)
}

#[cfg(test)]
mod tests {
    use super::*;
    use smallvec::smallvec;

    /// Fixture layout: subblocks packed contiguously in key order. Not a product
    /// layout (the only one is the coupled sector matrix); fixtures use it to
    /// exercise the arbitrary-strided-view contract of [`BlockStructure`].
    fn packed_fixture_structure<I, K>(rank: usize, blocks: I) -> Result<BlockStructure, CoreError>
    where
        I: IntoIterator<Item = (K, Vec<usize>)>,
        K: Into<BlockKey>,
    {
        let mut keys = Vec::new();
        let mut shapes = Vec::new();
        for (key, shape) in blocks {
            keys.push(key.into());
            shapes.push(shape);
        }
        BlockStructure::from_parts(
            SectorStructure::from_keys(rank, keys)?,
            DegeneracyStructure::packed_column_major(rank, shapes)?,
        )
    }

    #[test]
    fn block_fn_construction_is_layout_independent() {
        let rule = Z2FusionRule;
        let leg = |dual| SectorLeg::new([(z2_even(), 2), (z2_odd(), 2)], dual);
        let homspace = || {
            FusionTreeHomSpace::new(
                FusionProductSpace::new([leg(false), leg(false)]),
                FusionProductSpace::new([leg(false), leg(false)]),
            )
        };
        let shapes = |hom: &FusionTreeHomSpace| {
            hom.fusion_tree_keys(&rule)
                .iter()
                .map(|_| vec![2usize; 4])
                .collect::<Vec<_>>()
        };
        let dense = || TensorMapSpace::<2, 2>::from_dims([4, 4], [4, 4]).unwrap();
        let hom = homspace();
        let packed_structure =
            packed_fixture_structure(4, hom.fusion_tree_keys(&rule).into_iter().zip(shapes(&hom)))
                .unwrap();
        let packed_space =
            FusionTensorMapSpace::<2, 2>::new(dense(), hom.clone(), packed_structure).unwrap();
        let coupled_space = FusionTensorMapSpace::<2, 2>::from_degeneracy_shapes_coupled(
            dense(),
            hom.clone(),
            &rule,
            shapes(&hom),
        )
        .unwrap();

        let fill = |key: &BlockKey, indices: &[usize]| -> f64 {
            let BlockKey::FusionTree(tree) = key else {
                panic!("fusion tree keys expected");
            };
            let mut value = 0.0;
            for (axis, &sector) in tree
                .codomain_tree()
                .uncoupled()
                .iter()
                .chain(tree.domain_tree().uncoupled())
                .enumerate()
            {
                value += (axis as f64 + 1.0) * (sector.id() as f64 + 0.5);
            }
            for (axis, &index) in indices.iter().enumerate() {
                value += (axis as f64 + 2.0) * index as f64;
            }
            value
        };
        let packed =
            TensorMap::<f64, 2, 2>::from_block_fn_with_fusion_space(packed_space, 0.0, fill)
                .unwrap();
        let coupled =
            TensorMap::<f64, 2, 2>::from_block_fn_with_fusion_space(coupled_space, 0.0, fill)
                .unwrap();

        // Raw storage differs between layouts...
        assert_ne!(packed.data(), coupled.data());
        // ...but the logical block content is identical.
        let mut packed_elements = Vec::new();
        packed
            .for_each_block_element(|key, indices, value| {
                packed_elements.push((key.clone(), indices.to_vec(), *value));
            })
            .unwrap();
        let mut cursor = 0;
        coupled
            .for_each_block_element(|key, indices, value| {
                let (expected_key, expected_indices, expected_value) = &packed_elements[cursor];
                assert_eq!(key, expected_key);
                assert_eq!(indices, expected_indices.as_slice());
                assert_eq!(value, expected_value);
                cursor += 1;
            })
            .unwrap();
        assert_eq!(cursor, packed_elements.len());
    }

    #[test]
    fn coupled_layout_embeds_subblocks_into_sector_matrices() {
        let rule = Z2FusionRule;
        let leg = |degeneracy, dual| {
            SectorLeg::new([(z2_even(), degeneracy), (z2_odd(), degeneracy)], dual)
        };
        let homspace = FusionTreeHomSpace::new(
            FusionProductSpace::new([leg(2, false), leg(3, false)]),
            FusionProductSpace::new([leg(2, false), leg(3, false)]),
        );
        let keys = homspace.fusion_tree_keys(&rule);
        let shapes = keys
            .iter()
            .map(|_| vec![2usize, 3, 2, 3])
            .collect::<Vec<_>>();
        let packed = FusionTensorMapSpace::<2, 2>::from_degeneracy_shapes(
            TensorMapSpace::<2, 2>::from_dims([10, 10], [10, 10]).unwrap(),
            homspace.clone(),
            &rule,
            shapes.clone(),
        )
        .unwrap();
        let coupled = FusionTensorMapSpace::<2, 2>::from_degeneracy_shapes_coupled(
            TensorMapSpace::<2, 2>::from_dims([10, 10], [10, 10]).unwrap(),
            homspace,
            &rule,
            shapes,
        )
        .unwrap();

        let packed_structure = packed.subblock_structure();
        let coupled_structure = coupled.subblock_structure();
        assert_eq!(
            packed_structure.block_count(),
            coupled_structure.block_count()
        );
        assert_eq!(
            packed_structure.required_len().unwrap(),
            coupled_structure.required_len().unwrap()
        );

        // Two coupled sectors (even/odd), each with two codomain and two
        // domain trees of subblock row dim 6 and column dim 6: sector matrices
        // are 12 x 12.
        let matrix_rows = 12usize;
        let mut covered = vec![false; coupled_structure.required_len().unwrap()];
        for index in 0..coupled_structure.block_count() {
            let packed_block = packed_structure.block(index).unwrap();
            let coupled_block = coupled_structure.block(index).unwrap();
            assert_eq!(packed_block.key(), coupled_block.key());
            assert_eq!(packed_block.shape(), coupled_block.shape());
            // Codomain legs stay column-major inside the row block; domain
            // legs step whole matrix columns.
            assert_eq!(coupled_block.strides()[0], 1);
            assert_eq!(coupled_block.strides()[1], 2);
            assert_eq!(coupled_block.strides()[2], matrix_rows);
            assert_eq!(coupled_block.strides()[3], matrix_rows * 2);
            for i3 in 0..3 {
                for i2 in 0..2 {
                    for i1 in 0..3 {
                        for i0 in 0..2 {
                            let strides = coupled_block.strides();
                            let position = coupled_block.offset()
                                + i0 * strides[0]
                                + i1 * strides[1]
                                + i2 * strides[2]
                                + i3 * strides[3];
                            assert!(
                                !covered[position],
                                "coupled layout must not overlap between subblocks"
                            );
                            covered[position] = true;
                        }
                    }
                }
            }
        }
        assert!(
            covered.iter().all(|&flag| flag),
            "coupled layout must cover the sector matrices without holes"
        );
    }

    fn u1(charge: i32) -> SectorId {
        U1Irrep::new(charge).sector_id()
    }

    fn z2_even() -> SectorId {
        Z2Irrep::EVEN.sector_id()
    }

    fn z2_odd() -> SectorId {
        Z2Irrep::ODD.sector_id()
    }

    fn su2(twice_spin: usize) -> SectorId {
        SU2Irrep::from_twice_spin(twice_spin).sector_id()
    }

    #[derive(Clone, Copy, Debug)]
    struct BranchingMultiplicityFreeRule;

    impl FusionRule for BranchingMultiplicityFreeRule {
        fn fusion_style(&self) -> FusionStyleKind {
            FusionStyleKind::Simple
        }

        fn braiding_style(&self) -> BraidingStyleKind {
            BraidingStyleKind::Bosonic
        }

        fn vacuum(&self) -> SectorId {
            SectorId::new(0)
        }

        fn dual(&self, sector: SectorId) -> SectorId {
            match sector.id() {
                3 => SectorId::new(1),
                other => SectorId::new(other),
            }
        }

        fn fusion_channels(&self, left: SectorId, right: SectorId) -> SectorVec {
            match (left.id(), right.id()) {
                (0, x) | (x, 0) => smallvec![SectorId::new(x)],
                (1, 1) => smallvec![SectorId::new(0), SectorId::new(2)],
                (1, 2) | (2, 1) => smallvec![SectorId::new(1), SectorId::new(3)],
                (2, 2) => smallvec![SectorId::new(0)],
                _ => SmallVec::new(),
            }
        }
    }

    impl MultiplicityFreeFusionRule for BranchingMultiplicityFreeRule {}

    #[derive(Clone, Copy, Debug)]
    struct UnsortedFusionIteratorOrderRule;

    impl FusionRule for UnsortedFusionIteratorOrderRule {
        fn fusion_style(&self) -> FusionStyleKind {
            FusionStyleKind::Simple
        }

        fn braiding_style(&self) -> BraidingStyleKind {
            BraidingStyleKind::Bosonic
        }

        fn vacuum(&self) -> SectorId {
            SectorId::new(0)
        }

        fn dual(&self, sector: SectorId) -> SectorId {
            sector
        }

        fn fusion_channels(&self, left: SectorId, right: SectorId) -> SectorVec {
            match (left.id(), right.id()) {
                (0, x) | (x, 0) => smallvec![SectorId::new(x)],
                (1, 1) => smallvec![SectorId::new(2), SectorId::new(0)],
                (1, 2) | (2, 1) => smallvec![SectorId::new(1)],
                (2, 2) => smallvec![SectorId::new(0)],
                _ => SmallVec::new(),
            }
        }
    }

    impl MultiplicityFreeFusionRule for UnsortedFusionIteratorOrderRule {}

    #[derive(Clone, Copy, Debug)]
    struct Z4PointedRule;

    impl FusionRule for Z4PointedRule {
        fn fusion_style(&self) -> FusionStyleKind {
            FusionStyleKind::Unique
        }

        fn braiding_style(&self) -> BraidingStyleKind {
            BraidingStyleKind::Bosonic
        }

        fn vacuum(&self) -> SectorId {
            SectorId::new(0)
        }

        fn dual(&self, sector: SectorId) -> SectorId {
            SectorId::new((4 - sector.id() % 4) % 4)
        }

        fn fusion_channels(&self, left: SectorId, right: SectorId) -> SectorVec {
            smallvec![SectorId::new((left.id() + right.id()) % 4)]
        }
    }

    impl MultiplicityFreeFusionRule for Z4PointedRule {}

    #[derive(Clone, Copy, Debug)]
    struct Z2xZ3PointedRule;

    impl Z2xZ3PointedRule {
        const fn encode(z2: usize, z3: usize) -> SectorId {
            SectorId::new((z2 % 2) + 2 * (z3 % 3))
        }

        const fn decode(sector: SectorId) -> (usize, usize) {
            (sector.id() % 2, (sector.id() / 2) % 3)
        }
    }

    impl FusionRule for Z2xZ3PointedRule {
        fn fusion_style(&self) -> FusionStyleKind {
            FusionStyleKind::Unique
        }

        fn braiding_style(&self) -> BraidingStyleKind {
            BraidingStyleKind::Bosonic
        }

        fn vacuum(&self) -> SectorId {
            Self::encode(0, 0)
        }

        fn dual(&self, sector: SectorId) -> SectorId {
            let (z2, z3) = Self::decode(sector);
            Self::encode((2 - z2) % 2, (3 - z3) % 3)
        }

        fn fusion_channels(&self, left: SectorId, right: SectorId) -> SectorVec {
            let (left_z2, left_z3) = Self::decode(left);
            let (right_z2, right_z3) = Self::decode(right);
            smallvec![Self::encode(
                (left_z2 + right_z2) % 2,
                (left_z3 + right_z3) % 3,
            )]
        }
    }

    impl MultiplicityFreeFusionRule for Z2xZ3PointedRule {}

    #[derive(Clone, Copy, Debug)]
    struct PlanarZ2Rule;

    impl FusionRule for PlanarZ2Rule {
        fn fusion_style(&self) -> FusionStyleKind {
            FusionStyleKind::Unique
        }

        fn braiding_style(&self) -> BraidingStyleKind {
            BraidingStyleKind::NoBraiding
        }

        fn vacuum(&self) -> SectorId {
            SectorId::new(0)
        }

        fn fusion_channels(&self, left: SectorId, right: SectorId) -> SectorVec {
            smallvec![SectorId::new((left.id() + right.id()) % 2)]
        }
    }

    impl MultiplicityFreeFusionRule for PlanarZ2Rule {}

    impl MultiplicityFreeFusionSymbols for PlanarZ2Rule {
        type Scalar = f64;

        fn scalar_one(&self) -> Self::Scalar {
            1.0
        }

        fn scalar_conj(&self, value: Self::Scalar) -> Self::Scalar {
            value
        }

        fn f_symbol_scalar(
            &self,
            _left: SectorId,
            _middle: SectorId,
            _right: SectorId,
            _coupled: SectorId,
            _left_coupled: SectorId,
            _right_coupled: SectorId,
        ) -> Self::Scalar {
            1.0
        }

        fn r_symbol_scalar(
            &self,
            _left: SectorId,
            _right: SectorId,
            _coupled: SectorId,
        ) -> Self::Scalar {
            1.0
        }
    }

    #[derive(Clone, Copy, Debug)]
    struct AsymmetricAnyonicRule;

    impl FusionRule for AsymmetricAnyonicRule {
        fn fusion_style(&self) -> FusionStyleKind {
            FusionStyleKind::Unique
        }

        fn braiding_style(&self) -> BraidingStyleKind {
            BraidingStyleKind::Anyonic
        }

        fn vacuum(&self) -> SectorId {
            SectorId::new(0)
        }

        fn fusion_channels(&self, left: SectorId, right: SectorId) -> SectorVec {
            match (left.id(), right.id()) {
                (0, x) | (x, 0) => smallvec![SectorId::new(x)],
                (1, 2) | (2, 1) => smallvec![SectorId::new(3)],
                (3, 1) | (1, 3) => smallvec![SectorId::new(2)],
                (3, 2) | (2, 3) => smallvec![SectorId::new(1)],
                _ => smallvec![SectorId::new((left.id() + right.id()) % 4)],
            }
        }
    }

    impl MultiplicityFreeFusionRule for AsymmetricAnyonicRule {}

    impl MultiplicityFreeFusionSymbols for AsymmetricAnyonicRule {
        type Scalar = f64;

        fn scalar_one(&self) -> Self::Scalar {
            1.0
        }

        fn scalar_conj(&self, value: Self::Scalar) -> Self::Scalar {
            value
        }

        fn f_symbol_scalar(
            &self,
            _left: SectorId,
            _middle: SectorId,
            _right: SectorId,
            _coupled: SectorId,
            _left_coupled: SectorId,
            _right_coupled: SectorId,
        ) -> Self::Scalar {
            11.0
        }

        fn r_symbol_scalar(
            &self,
            left: SectorId,
            right: SectorId,
            _coupled: SectorId,
        ) -> Self::Scalar {
            match (left.id(), right.id()) {
                (1, 2) => 5.0,
                (2, 1) => 7.0,
                (3, 2) => 13.0,
                (2, 3) => 17.0,
                (1, 3) => 19.0,
                (3, 1) => 23.0,
                _ => 1.0,
            }
        }
    }

    impl MultiplicityFreePivotalSymbols for AsymmetricAnyonicRule {
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
            _source: &FusionTreeBlockKey,
            _destination: &FusionTreeBlockKey,
        ) -> Self::Scalar {
            1.0
        }
    }

    fn fusion_tree_pair_order(keys: &[FusionTreeBlockKey]) -> Vec<(Vec<usize>, Vec<usize>, usize)> {
        keys.iter()
            .map(|key| {
                (
                    sector_ids(key.codomain_uncoupled()),
                    sector_ids(key.domain_uncoupled()),
                    key.coupled().expect("test keys have a coupled sector").id(),
                )
            })
            .collect()
    }

    fn sector_ids(sectors: &[SectorId]) -> Vec<usize> {
        sectors.iter().map(|sector| sector.id()).collect()
    }

    #[test]
    fn fusion_style_kind_matches_tensorkit_multiplicity_free_split() {
        assert!(FusionStyleKind::Unique.is_multiplicity_free());
        assert!(FusionStyleKind::Simple.is_multiplicity_free());
        assert!(!FusionStyleKind::Generic.is_multiplicity_free());
        assert!(!FusionStyleKind::Unique.has_multiple_outputs());
        assert!(FusionStyleKind::Simple.has_multiple_outputs());
        assert!(FusionStyleKind::Generic.has_multiple_outputs());
        assert!(!FusionStyleKind::Unique.has_multiplicity());
        assert!(!FusionStyleKind::Simple.has_multiplicity());
        assert!(FusionStyleKind::Generic.has_multiplicity());
    }

    #[test]
    fn braiding_style_kind_matches_tensorkit_hierarchy() {
        assert!(!BraidingStyleKind::NoBraiding.has_braiding());
        assert!(BraidingStyleKind::Bosonic.has_braiding());
        assert!(BraidingStyleKind::Fermionic.has_braiding());
        assert!(BraidingStyleKind::Anyonic.has_braiding());

        assert!(!BraidingStyleKind::NoBraiding.is_symmetric());
        assert!(BraidingStyleKind::Bosonic.is_symmetric());
        assert!(BraidingStyleKind::Fermionic.is_symmetric());
        assert!(!BraidingStyleKind::Anyonic.is_symmetric());

        assert!(BraidingStyleKind::Bosonic.is_bosonic());
        assert!(!BraidingStyleKind::Fermionic.is_bosonic());
        assert_eq!(
            BraidingStyleKind::Bosonic.combined_with(BraidingStyleKind::Fermionic),
            BraidingStyleKind::Fermionic
        );
        assert_eq!(
            BraidingStyleKind::Fermionic.combined_with(BraidingStyleKind::Anyonic),
            BraidingStyleKind::Anyonic
        );
        assert_eq!(
            BraidingStyleKind::Anyonic.combined_with(BraidingStyleKind::NoBraiding),
            BraidingStyleKind::NoBraiding
        );
    }

    #[test]
    fn fusion_rule_exposes_unique_outputs_and_nsymbol_separately() {
        let z2 = Z2FusionRule;
        let su2 = SU2FusionRule;

        assert_eq!(z2.fusion_style(), FusionStyleKind::Unique);
        assert_eq!(
            z2.fusion_channels(SectorId::new(1), SectorId::new(1))
                .to_vec(),
            vec![SectorId::new(0)]
        );
        assert_eq!(
            z2.nsymbol(SectorId::new(1), SectorId::new(1), SectorId::new(0)),
            1
        );
        assert_eq!(
            z2.nsymbol(SectorId::new(1), SectorId::new(1), SectorId::new(1)),
            0
        );

        assert_eq!(su2.fusion_style(), FusionStyleKind::Simple);
        assert_eq!(
            su2.fusion_channels(SectorId::new(1), SectorId::new(1))
                .to_vec(),
            vec![SectorId::new(0), SectorId::new(2)]
        );
        assert_eq!(
            su2.nsymbol(SectorId::new(1), SectorId::new(1), SectorId::new(2)),
            1
        );
    }

    #[test]
    fn multiplicity_free_symbols_are_a_separate_scalar_api() {
        let z2 = Z2FusionRule;

        assert_eq!(z2.scalar_one(), 1.0);
        assert_eq!(
            z2.f_symbol_scalar(
                SectorId::new(1),
                SectorId::new(1),
                SectorId::new(1),
                SectorId::new(1),
                SectorId::new(0),
                SectorId::new(0),
            ),
            1.0
        );
        assert_eq!(
            z2.r_symbol_scalar(SectorId::new(1), SectorId::new(1), SectorId::new(0)),
            1.0
        );
    }

    #[test]
    fn unique_artin_braid_first_allows_unit_crossing_without_braiding() {
        let tree = FusionTreeKey::from_sector_ids([0, 1], Some(1), [false, true], [], [1]);

        let (braided, coefficient) = unique_artin_braid_first(&PlanarZ2Rule, &tree).unwrap();

        assert_eq!(coefficient, 1.0);
        assert_eq!(braided.uncoupled(), &[SectorId::new(1), SectorId::new(0)]);
        assert_eq!(braided.is_dual(), &[true, false]);
        assert_eq!(braided.coupled(), Some(SectorId::new(1)));
        assert!(braided.innerlines().is_empty());
        assert_eq!(braided.vertices(), &[SectorId::new(1)]);
    }

    #[test]
    fn unique_artin_braid_first_rejects_nonunit_crossing_without_braiding() {
        let tree = FusionTreeKey::from_sector_ids([1, 1], Some(0), [false, false], [], [1]);

        let err = unique_artin_braid_first(&PlanarZ2Rule, &tree).unwrap_err();

        assert_eq!(
            err,
            CoreError::UnsupportedSectorBraid {
                left: SectorId::new(1),
                right: SectorId::new(1),
                style: BraidingStyleKind::NoBraiding,
            }
        );
    }

    #[test]
    fn unique_artin_braid_first_uses_r_symbol_for_first_crossing() {
        let tree = FusionTreeKey::from_sector_ids([1, 1], Some(0), [false, true], [], [1]);

        let (braided, coefficient) =
            unique_artin_braid_first(&FermionParityFusionRule, &tree).unwrap();

        assert_eq!(coefficient, -1.0);
        assert_eq!(braided.uncoupled(), &[SectorId::new(1), SectorId::new(1)]);
        assert_eq!(braided.is_dual(), &[true, false]);
        assert_eq!(braided.coupled(), Some(SectorId::new(0)));
        assert!(braided.innerlines().is_empty());
        assert_eq!(braided.vertices(), &[SectorId::new(1)]);
    }

    #[test]
    fn unique_artin_braid_first_uses_first_innerline_for_rank_three() {
        let tree =
            FusionTreeKey::from_sector_ids([1, 1, 1], Some(1), [false, false, false], [0], [1, 1]);

        let (braided, coefficient) =
            unique_artin_braid_first(&FermionParityFusionRule, &tree).unwrap();

        assert_eq!(coefficient, -1.0);
        assert_eq!(
            braided.uncoupled(),
            &[SectorId::new(1), SectorId::new(1), SectorId::new(1)]
        );
        assert_eq!(braided.innerlines(), &[SectorId::new(0)]);
        assert_eq!(braided.vertices(), &[SectorId::new(1), SectorId::new(1)]);
    }

    #[test]
    fn unique_artin_braid_at_updates_innerline_for_later_unit_crossing() {
        let tree =
            FusionTreeKey::from_sector_ids([1, 0, 1], Some(0), [false, false, true], [1], [1, 1]);

        let (braided, coefficient) = unique_artin_braid_at(&PlanarZ2Rule, &tree, 1).unwrap();

        assert_eq!(coefficient, 1.0);
        assert_eq!(
            braided.uncoupled(),
            &[SectorId::new(1), SectorId::new(1), SectorId::new(0)]
        );
        assert_eq!(braided.is_dual(), &[false, true, false]);
        assert_eq!(braided.innerlines(), &[SectorId::new(0)]);
        assert_eq!(braided.vertices(), &[SectorId::new(1), SectorId::new(1)]);
    }

    #[test]
    fn unique_artin_braid_at_uses_f_and_r_symbols_for_later_crossing() {
        let tree =
            FusionTreeKey::from_sector_ids([1, 1, 1], Some(1), [false, true, false], [0], [1, 1]);

        let (braided, coefficient) =
            unique_artin_braid_at(&FermionParityFusionRule, &tree, 1).unwrap();

        assert_eq!(coefficient, -1.0);
        assert_eq!(
            braided.uncoupled(),
            &[SectorId::new(1), SectorId::new(1), SectorId::new(1)]
        );
        assert_eq!(braided.is_dual(), &[false, false, true]);
        assert_eq!(braided.innerlines(), &[SectorId::new(0)]);
        assert_eq!(braided.vertices(), &[SectorId::new(1), SectorId::new(1)]);
    }

    #[test]
    fn unique_artin_braid_at_rejects_out_of_range_index() {
        let tree = FusionTreeKey::from_sector_ids([1, 1], Some(0), [false, false], [], [1]);

        let err = unique_artin_braid_at(&FermionParityFusionRule, &tree, 1).unwrap_err();

        assert_eq!(err, CoreError::InvalidBraidIndex { index: 1, rank: 2 });
    }

    #[test]
    fn permutation_to_adjacent_swaps_matches_tensorkit_order() {
        assert_eq!(
            permutation_to_adjacent_swaps(&[2, 0, 1], 3).unwrap(),
            vec![1, 0]
        );
        assert_eq!(
            permutation_to_adjacent_swaps(&[3, 0, 2, 1], 4).unwrap(),
            vec![2, 1, 0, 2]
        );
    }

    #[test]
    fn unique_braid_tree_replays_tensorkit_swap_order_and_level_updates() {
        let tree =
            FusionTreeKey::from_sector_ids([1, 1, 1], Some(1), [false, false, false], [0], [1, 1]);

        let (braided, coefficient) =
            unique_braid_tree(&FermionParityFusionRule, &tree, &[2, 0, 1], &[0, 1, 2]).unwrap();

        assert_eq!(coefficient, 1.0);
        assert_eq!(
            braided.uncoupled(),
            &[SectorId::new(1), SectorId::new(1), SectorId::new(1)]
        );
        assert_eq!(braided.is_dual(), &[false, false, false]);
        assert_eq!(braided.coupled(), Some(SectorId::new(1)));
        assert_eq!(braided.innerlines(), &[SectorId::new(0)]);
        assert_eq!(braided.vertices(), &[SectorId::new(1), SectorId::new(1)]);
    }

    #[test]
    fn unique_braid_tree_uses_inverse_artin_branch_from_levels() {
        let tree = FusionTreeKey::from_sector_ids([1, 2], Some(3), [false, false], [], [1]);

        let (braided_forward, forward) =
            unique_braid_tree(&AsymmetricAnyonicRule, &tree, &[1, 0], &[0, 1]).unwrap();
        let (braided_inverse, inverse) =
            unique_braid_tree(&AsymmetricAnyonicRule, &tree, &[1, 0], &[1, 0]).unwrap();

        assert_eq!(forward, 5.0);
        assert_eq!(inverse, 7.0);
        assert_eq!(braided_forward, braided_inverse);
        assert_eq!(
            braided_forward.uncoupled(),
            &[SectorId::new(2), SectorId::new(1)]
        );
        assert_eq!(braided_forward.coupled(), Some(SectorId::new(3)));
    }

    #[test]
    fn unique_braid_tree_reflected_levels_select_inverse_artin_branch() {
        let tree = FusionTreeKey::from_sector_ids([1, 2], Some(3), [false, false], [], [1]);
        let levels = [3, 8];
        let min_level = levels.iter().copied().min().unwrap();
        let max_level = levels.iter().copied().max().unwrap();
        let reflected_levels = levels
            .iter()
            .map(|&level| min_level + max_level - level)
            .collect::<Vec<_>>();

        let (forward_tree, forward_coeff) =
            unique_braid_tree(&AsymmetricAnyonicRule, &tree, &[1, 0], &levels).unwrap();
        let (inverse_tree, inverse_coeff) =
            unique_braid_tree(&AsymmetricAnyonicRule, &tree, &[1, 0], &reflected_levels).unwrap();

        assert_eq!(reflected_levels, vec![8, 3]);
        assert_eq!(forward_tree, inverse_tree);
        assert_eq!(forward_coeff, 5.0);
        assert_eq!(inverse_coeff, 7.0);
    }

    #[test]
    fn unique_braid_tree_rejects_invalid_permutation_and_level_count() {
        let tree = FusionTreeKey::from_sector_ids([1, 2], Some(3), [false, false], [], [1]);

        let err = unique_braid_tree(&AsymmetricAnyonicRule, &tree, &[1, 1], &[0, 1]).unwrap_err();
        assert_eq!(
            err,
            CoreError::InvalidPermutation {
                permutation: vec![1, 1],
                rank: 2,
            }
        );

        let err = unique_braid_tree(&AsymmetricAnyonicRule, &tree, &[1, 0], &[0]).unwrap_err();
        assert_eq!(
            err,
            CoreError::DimensionMismatch {
                expected: 2,
                actual: 1,
            }
        );
    }

    #[test]
    fn unique_permute_tree_requires_symmetric_braiding() {
        let tree = FusionTreeKey::from_sector_ids([1, 2], Some(3), [false, false], [], [1]);

        let err = unique_permute_tree(&AsymmetricAnyonicRule, &tree, &[1, 0]).unwrap_err();

        assert_eq!(
            err,
            CoreError::UnsupportedBraidingStyle {
                expected: "symmetric braiding",
                actual: BraidingStyleKind::Anyonic,
            }
        );
    }

    #[test]
    fn linearize_tree_pair_permutation_matches_tensorkit_zero_based_formula() {
        assert_eq!(
            linearize_tree_pair_permutation(&[0, 1], &[2, 3], 2, 2).unwrap(),
            vec![0, 1, 2, 3]
        );
        assert_eq!(
            linearize_tree_pair_permutation(&[3, 0], &[1, 2], 2, 2).unwrap(),
            vec![2, 0, 3, 1]
        );

        let err = linearize_tree_pair_permutation(&[0, 0], &[1, 2], 2, 2).unwrap_err();
        assert_eq!(
            err,
            CoreError::InvalidPermutation {
                permutation: vec![0, 0, 1, 2],
                rank: 4,
            }
        );
    }

    #[test]
    fn unique_repartition_tree_pair_moves_domain_to_reversed_dual_codomain() {
        let source = FusionTreeBlockKey::pair_from_sector_ids(
            [1],
            [0, 1],
            Some(1),
            [false],
            [false, true],
            [],
            [],
            [],
            [1],
        );

        let (all_out, coefficient) =
            unique_repartition_tree_pair(&Z2FusionRule, &source, 3).unwrap();

        assert_eq!(coefficient, 1.0);
        assert_eq!(
            all_out.codomain_uncoupled(),
            &[SectorId::new(1), SectorId::new(1), SectorId::new(0)]
        );
        assert_eq!(all_out.codomain_is_dual(), &[false, false, true]);
        assert_eq!(all_out.codomain_innerlines(), &[SectorId::new(0)]);
        assert_eq!(
            all_out.codomain_vertices(),
            &[SectorId::new(1), SectorId::new(1)]
        );
        assert!(all_out.domain_uncoupled().is_empty());
        assert_eq!(all_out.domain_tree().coupled(), Some(SectorId::new(0)));
    }

    #[test]
    fn unique_braid_tree_pair_matches_single_tree_when_domain_is_empty() {
        let source = FusionTreeBlockKey::pair(
            FusionTreeKey::from_sector_ids([1, 1], Some(0), [false, true], [], [1]),
            FusionTreeKey::new(
                Vec::<SectorId>::new(),
                None,
                Vec::<bool>::new(),
                Vec::<SectorId>::new(),
                Vec::<SectorId>::new(),
            ),
        );

        let (braided, coefficient) = unique_braid_tree_pair(
            &FermionParityFusionRule,
            &source,
            &[1, 0],
            &[],
            &[0, 1],
            &[],
        )
        .unwrap();

        assert_eq!(coefficient, -1.0);
        assert_eq!(
            braided.codomain_uncoupled(),
            &[SectorId::new(1), SectorId::new(1)]
        );
        assert_eq!(braided.codomain_is_dual(), &[true, false]);
        assert!(braided.domain_uncoupled().is_empty());
        assert_eq!(braided.domain_tree().coupled(), None);
    }

    #[test]
    fn unique_permute_tree_pair_handles_domain_only_swap() {
        let source = FusionTreeBlockKey::pair_from_sector_ids(
            [1],
            [0, 1],
            Some(1),
            [false],
            [false, true],
            [],
            [],
            [],
            [1],
        );

        let (permuted, coefficient) =
            unique_permute_tree_pair(&Z2FusionRule, &source, &[0], &[2, 1]).unwrap();

        assert_eq!(coefficient, 1.0);
        assert_eq!(permuted.codomain_uncoupled(), &[SectorId::new(1)]);
        assert_eq!(
            permuted.domain_uncoupled(),
            &[SectorId::new(1), SectorId::new(0)]
        );
        assert_eq!(permuted.domain_is_dual(), &[true, false]);
        assert_eq!(permuted.domain_vertices(), &[SectorId::new(1)]);
    }

    #[test]
    fn unique_permute_tree_pair_includes_codomain_domain_crossing() {
        let source = FusionTreeBlockKey::pair_from_sector_ids(
            [1],
            [1],
            Some(1),
            [false],
            [true],
            [],
            [],
            [],
            [],
        );

        let (permuted, coefficient) =
            unique_permute_tree_pair(&FermionParityFusionRule, &source, &[1], &[0]).unwrap();

        assert_eq!(coefficient, -1.0);
        assert_eq!(permuted.codomain_uncoupled(), &[SectorId::new(1)]);
        assert_eq!(permuted.codomain_is_dual(), &[false]);
        assert_eq!(permuted.domain_uncoupled(), &[SectorId::new(1)]);
        assert_eq!(permuted.domain_is_dual(), &[true]);
    }

    #[test]
    fn unique_transpose_tree_pair_is_cyclic_and_reversible() {
        let source = FusionTreeBlockKey::pair_from_sector_ids(
            [1],
            [1],
            Some(1),
            [false],
            [true],
            [],
            [],
            [],
            [],
        );

        let (transposed, coefficient) =
            unique_transpose_tree_pair(&Z2FusionRule, &source, &[1], &[0]).unwrap();
        let (roundtrip, inverse_coefficient) =
            unique_transpose_tree_pair(&Z2FusionRule, &transposed, &[1], &[0]).unwrap();

        assert_eq!(coefficient, 1.0);
        assert_eq!(inverse_coefficient, 1.0);
        assert_eq!(roundtrip, source);
    }

    #[test]
    fn unique_transpose_tree_pair_matches_tensorkit_clockwise_cycle() {
        let source = FusionTreeBlockKey::pair_from_sector_ids(
            [1, 0],
            [1, 0],
            Some(1),
            [false, false],
            [false, false],
            [],
            [],
            [1],
            [1],
        );
        let expected = FusionTreeBlockKey::pair_from_sector_ids(
            [0, 0],
            [1, 1],
            Some(0),
            [false, true],
            [true, false],
            [],
            [],
            [1],
            [1],
        );

        let (transposed, coefficient) =
            unique_transpose_tree_pair(&Z2FusionRule, &source, &[1, 3], &[0, 2]).unwrap();

        assert_eq!(coefficient, 1.0);
        assert_eq!(transposed, expected);
    }

    #[test]
    fn unique_transpose_tree_pair_matches_tensorkit_anticlockwise_cycle() {
        let source = FusionTreeBlockKey::pair_from_sector_ids(
            [1, 0],
            [1, 0],
            Some(1),
            [false, false],
            [false, false],
            [],
            [],
            [1],
            [1],
        );
        let expected = FusionTreeBlockKey::pair_from_sector_ids(
            [1, 1],
            [0, 0],
            Some(0),
            [true, false],
            [false, true],
            [],
            [],
            [1],
            [1],
        );

        let (transposed, coefficient) =
            unique_transpose_tree_pair(&Z2FusionRule, &source, &[2, 0], &[3, 1]).unwrap();

        assert_eq!(coefficient, 1.0);
        assert_eq!(transposed, expected);
    }

    #[test]
    fn unique_transpose_tree_pair_rejects_noncyclic_permutation() {
        let source = FusionTreeBlockKey::pair_from_sector_ids(
            [1, 0],
            [1],
            Some(1),
            [false, false],
            [false],
            [],
            [],
            [1],
            [],
        );

        let err = unique_transpose_tree_pair(&Z2FusionRule, &source, &[0, 2], &[1]).unwrap_err();

        assert_eq!(
            err,
            CoreError::InvalidPermutation {
                permutation: vec![0, 2, 1],
                rank: 3,
            }
        );
    }

    #[test]
    fn block_view_validates_column_major_layout() {
        let data = [0.0; 6];
        let shape = [2, 3];
        let strides = [1, 2];
        let view = BlockView::new(&data, &shape, &strides, 0).unwrap();
        assert_eq!(view.shape(), &[2, 3]);
        assert_eq!(view.strides(), &[1, 2]);
    }

    #[test]
    fn block_view_rejects_out_of_bounds_layout() {
        let data = [0.0; 6];
        let shape = [2, 3];
        let strides = [1, 4];
        let err = BlockView::new(&data, &shape, &strides, 0).unwrap_err();
        assert_eq!(err, CoreError::OutOfBounds);
    }

    #[test]
    fn trivial_tensormap_exposes_single_column_major_subblock() {
        let space = TensorMapSpace::<2, 1>::from_dims([2, 3], [4]).unwrap();
        let tensor =
            TensorMap::<f64, 2, 1>::from_vec((0..24).map(|x| x as f64).collect(), space).unwrap();

        assert_eq!(tensor.dim(), 24);
        assert_eq!(tensor.dims(), &[2, 3, 4]);
        assert_eq!(tensor.placement(), Placement::Host);
        assert_eq!(tensor.structure().block_count(), 1);

        let block = tensor.subblock().unwrap();
        assert_eq!(
            tensor.structure().block(0).unwrap().key(),
            &BlockKey::trivial()
        );
        assert_eq!(block.shape(), &[2, 3, 4]);
        assert_eq!(block.strides(), &[1, 2, 6]);
        assert_eq!(block.offset(), 0);
        assert_eq!(block.data()[23], 23.0);
    }

    #[test]
    fn block_structure_finds_fusion_tree_subblock_by_key() {
        let first = FusionTreeBlockKey::pair_from_sector_ids(
            [1],
            [1],
            Some(1),
            [false],
            [true],
            [],
            [],
            [],
            [],
        );
        let second = FusionTreeBlockKey::pair_from_sector_ids(
            [0],
            [0],
            Some(0),
            [false],
            [true],
            [],
            [],
            [],
            [],
        );
        let structure = packed_fixture_structure(
            2,
            [
                (BlockKey::from(second.clone()), vec![1, 4]),
                (BlockKey::from(first.clone()), vec![2, 3]),
            ],
        )
        .unwrap();

        let first_block = structure.fusion_tree_block(&first).unwrap();
        let second_block = structure
            .block_by_key(&BlockKey::from(second.clone()))
            .unwrap();

        assert_eq!(first_block.key(), &BlockKey::from(first));
        assert_eq!(first_block.shape(), &[2, 3]);
        assert_eq!(first_block.offset(), 4);
        assert_eq!(second_block.key(), &BlockKey::from(second));
        assert_eq!(second_block.shape(), &[1, 4]);
        assert_eq!(second_block.offset(), 0);
    }

    #[test]
    fn tensormap_subblock_by_tree_returns_matching_view() {
        let first = FusionTreeBlockKey::pair_from_sector_ids(
            [1],
            [1],
            Some(1),
            [false],
            [true],
            [],
            [],
            [],
            [],
        );
        let second = FusionTreeBlockKey::pair_from_sector_ids(
            [0],
            [0],
            Some(0),
            [false],
            [true],
            [],
            [],
            [],
            [],
        );
        let structure = packed_fixture_structure(
            2,
            [
                (BlockKey::from(second.clone()), vec![1, 2]),
                (BlockKey::from(first.clone()), vec![2, 2]),
            ],
        )
        .unwrap();
        let space = TensorMapSpace::<1, 1>::from_dims([3], [3]).unwrap();
        let tensor = TensorMap::<i32, 1, 1>::from_vec_with_structure(
            vec![10, 20, 30, 40, 50, 60],
            space,
            structure,
        )
        .unwrap();

        let first_view = tensor.subblock_by_tree(&first).unwrap();
        let second_view = tensor.block_by_key(&BlockKey::from(second)).unwrap();

        assert_eq!(first_view.shape(), &[2, 2]);
        assert_eq!(first_view.offset(), 2);
        assert_eq!(
            &first_view.data()[first_view.offset()..first_view.offset() + 4],
            &[30, 40, 50, 60]
        );
        assert_eq!(second_view.shape(), &[1, 2]);
        assert_eq!(second_view.offset(), 0);
    }

    #[test]
    fn tensormap_subblock_mut_by_tree_updates_selected_storage() {
        let key = FusionTreeBlockKey::pair_from_sector_ids(
            [1],
            [1],
            Some(1),
            [false],
            [true],
            [],
            [],
            [],
            [],
        );
        let structure = packed_fixture_structure(
            2,
            [
                (BlockKey::sector_ids([0]), vec![1, 2]),
                (BlockKey::from(key.clone()), vec![2, 1]),
            ],
        )
        .unwrap();
        let space = TensorMapSpace::<1, 1>::from_dims([3], [2]).unwrap();
        let mut tensor =
            TensorMap::<i32, 1, 1>::from_vec_with_structure(vec![1, 2, 3, 4], space, structure)
                .unwrap();

        {
            let mut view = tensor.subblock_mut_by_tree(&key).unwrap();
            let offset = view.offset();
            view.data_mut()[offset] = 30;
            view.data_mut()[offset + 1] = 40;
        }

        assert_eq!(tensor.data(), &[1, 2, 30, 40]);
    }

    #[test]
    fn subblock_by_tree_reports_missing_fusion_tree_key() {
        let existing = FusionTreeBlockKey::pair_from_sector_ids(
            [0],
            [0],
            Some(0),
            [false],
            [true],
            [],
            [],
            [],
            [],
        );
        let missing = FusionTreeBlockKey::pair_from_sector_ids(
            [1],
            [1],
            Some(1),
            [false],
            [true],
            [],
            [],
            [],
            [],
        );
        let structure =
            packed_fixture_structure(2, [(BlockKey::from(existing), vec![1, 1])]).unwrap();
        let space = TensorMapSpace::<1, 1>::from_dims([1], [1]).unwrap();
        let tensor =
            TensorMap::<f64, 1, 1>::from_vec_with_structure(vec![1.0], space, structure).unwrap();

        let err = tensor.subblock_by_tree(&missing).unwrap_err();

        assert_eq!(
            err,
            CoreError::MissingBlockKey {
                key: BlockKey::from(missing),
            }
        );
    }

    #[test]
    fn public_u1_irrep_roundtrips_compact_ids_and_fuses() {
        let rule = U1FusionRule;
        let charges = [
            U1Irrep::new(-2),
            U1Irrep::new(-1),
            U1Irrep::new(0),
            U1Irrep::new(1),
            U1Irrep::new(2),
        ];
        let ids = charges.map(SectorId::from);

        assert_eq!(
            ids,
            [
                SectorId::new(3),
                SectorId::new(1),
                SectorId::new(0),
                SectorId::new(2),
                SectorId::new(4),
            ]
        );
        for charge in charges {
            assert_eq!(U1Irrep::from_sector_id(charge.sector_id()), Some(charge));
        }
        assert_eq!(rule.vacuum(), U1Irrep::new(0).sector_id());
        assert_eq!(
            rule.dual(U1Irrep::new(3).sector_id()),
            U1Irrep::new(-3).sector_id()
        );
        assert_eq!(
            rule.fusion_channels(U1Irrep::new(-2).sector_id(), U1Irrep::new(5).sector_id())
                .to_vec(),
            vec![U1Irrep::new(3).sector_id()]
        );
    }

    #[test]
    fn product_sector_codec_uses_tensorkit_diagonal_component_order() {
        let expected = [
            (0, 0),
            (0, 1),
            (1, 0),
            (0, 2),
            (1, 1),
            (2, 0),
            (0, 3),
            (1, 2),
            (2, 1),
            (3, 0),
        ];

        for (id, &(left, right)) in expected.iter().enumerate() {
            let encoded = TensorKitProductCodec::encode(SectorId::new(left), SectorId::new(right));
            assert_eq!(encoded, SectorId::new(id));
            assert_eq!(
                TensorKitProductCodec::decode(encoded),
                Some((SectorId::new(left), SectorId::new(right)))
            );
        }
    }

    #[test]
    fn product_sector_api_exposes_only_generic_composition() {
        let pair = product_sector(z2_odd(), u1(2));
        let encoded = pair.sector_id_with::<TensorKitProductCodec>();
        assert_eq!(encoded, TensorKitProductCodec::encode(z2_odd(), u1(2)));
        assert_eq!(pair.left(), &z2_odd());
        assert_eq!(pair.right(), &u1(2));

        let left_rule = product_fusion_rule(FermionParityFusionRule, U1FusionRule);
        let chained_rule = FermionParityFusionRule
            .product(U1FusionRule)
            .product(SU2FusionRule);
        let left_sector = |parity, charge| left_rule.encode_sector(parity, u1(charge));
        let chained_sector = |parity, charge, twice_spin| {
            chained_rule.encode_sector(left_sector(parity, charge), su2(twice_spin))
        };

        let a = chained_sector(z2_odd(), 1, 1);
        let b = chained_sector(z2_odd(), -1, 1);
        let c0 = chained_sector(z2_even(), 0, 0);
        let c2 = chained_sector(z2_even(), 0, 2);

        assert_eq!(chained_rule.fusion_style(), FusionStyleKind::Simple);
        assert_eq!(chained_rule.braiding_style(), BraidingStyleKind::Fermionic);
        assert_eq!(chained_rule.fusion_channels(a, b).to_vec(), vec![c0, c2]);
    }

    #[test]
    fn product_fusion_rule_combines_fermion_parity_and_u1_componentwise() {
        type FpU1Rule = ProductFusionRule<FermionParityFusionRule, U1FusionRule>;
        let rule = FpU1Rule::default();
        let sector = |parity, charge| rule.encode_sector(parity, u1(charge));
        let odd_two = sector(z2_odd(), 2);
        let odd_minus_five = sector(z2_odd(), -5);
        let even_minus_three = sector(z2_even(), -3);

        assert_eq!(rule.fusion_style(), FusionStyleKind::Unique);
        assert_eq!(rule.braiding_style(), BraidingStyleKind::Fermionic);
        assert_eq!(rule.vacuum(), sector(z2_even(), 0));
        assert_eq!(rule.dual(odd_two), sector(z2_odd(), -2));
        assert_eq!(
            rule.fusion_channels(odd_two, odd_minus_five).to_vec(),
            vec![even_minus_three]
        );
        assert_eq!(rule.nsymbol(odd_two, odd_minus_five, even_minus_three), 1);
        assert_eq!(
            rule.r_symbol_scalar(odd_two, odd_minus_five, even_minus_three),
            -1.0
        );
        assert_eq!(rule.sqrt_dim_scalar(odd_two), 1.0);
    }

    #[test]
    fn product_fusion_rule_nested_fz2_u1_su2_channels_and_symbols_match_tensorkit() {
        type FpU1Rule = ProductFusionRule<FermionParityFusionRule, U1FusionRule>;
        type FpU1Su2Rule = ProductFusionRule<FpU1Rule, SU2FusionRule>;
        let left_rule = FpU1Rule::default();
        let rule = FpU1Su2Rule::default();
        let left_sector = |parity, charge| left_rule.encode_sector(parity, u1(charge));
        let sector = |parity, charge, twice_spin| {
            rule.encode_sector(left_sector(parity, charge), su2(twice_spin))
        };

        let a = sector(z2_odd(), 1, 1);
        let b = sector(z2_odd(), -1, 1);
        let c0 = sector(z2_even(), 0, 0);
        let c2 = sector(z2_even(), 0, 2);

        assert_eq!(rule.fusion_style(), FusionStyleKind::Simple);
        assert_eq!(rule.braiding_style(), BraidingStyleKind::Fermionic);
        assert_eq!(rule.dual(a), sector(z2_odd(), -1, 1));
        assert_eq!(rule.fusion_channels(a, b).to_vec(), vec![c0, c2]);
        assert_eq!(rule.r_symbol_scalar(a, b, c0), 1.0);
        assert_eq!(rule.r_symbol_scalar(a, b, c2), -1.0);
        assert!((rule.sqrt_dim_scalar(c2) - 3.0_f64.sqrt()).abs() < 1.0e-12);

        let vacuum_left = left_sector(z2_even(), 0);
        let spin_half = rule.encode_sector(vacuum_left, su2(1));
        let spin_zero = rule.encode_sector(vacuum_left, su2(0));
        assert!(
            (rule.f_symbol_scalar(
                spin_half, spin_half, spin_half, spin_half, spin_zero, spin_zero,
            ) + 0.5)
                .abs()
                < 1.0e-12
        );
    }

    #[test]
    fn product_fusion_tree_homspace_matches_tensorkit_fz2_u1_su2_fixture() {
        type FpU1Rule = ProductFusionRule<FermionParityFusionRule, U1FusionRule>;
        type FpU1Su2Rule = ProductFusionRule<FpU1Rule, SU2FusionRule>;
        let left_rule = FpU1Rule::default();
        let rule = FpU1Su2Rule::default();
        let left_sector = |parity, charge| left_rule.encode_sector(parity, u1(charge));
        let sector = |parity, charge, twice_spin| {
            rule.encode_sector(left_sector(parity, charge), su2(twice_spin))
        };

        let a = sector(z2_odd(), 1, 1);
        let b = sector(z2_odd(), -1, 1);
        let c0 = sector(z2_even(), 0, 0);
        let c1 = sector(z2_even(), 0, 2);
        assert_eq!(a.id(), 43);
        assert_eq!(b.id(), 19);
        assert_eq!(c0.id(), 0);
        assert_eq!(c1.id(), 3);

        let hom = FusionTreeHomSpace::new(
            FusionProductSpace::new([
                SectorLeg::new([(a, 1)], false),
                SectorLeg::new([(b, 1)], false),
            ]),
            FusionProductSpace::new([SectorLeg::new([(c0, 1), (c1, 1)], false)]),
        );
        let keys = hom.fusion_tree_keys(&rule);

        assert_eq!(keys.len(), 2);
        for (key, coupled) in keys.iter().zip([c0, c1]) {
            assert_eq!(key.coupled(), Some(coupled));
            assert_eq!(key.codomain_uncoupled(), &[a, b]);
            assert_eq!(key.domain_uncoupled(), &[coupled]);
            assert_eq!(key.codomain_is_dual(), &[false, false]);
            assert_eq!(key.domain_is_dual(), &[false]);
            assert_eq!(key.codomain_innerlines(), &[]);
            assert_eq!(key.domain_innerlines(), &[]);
            assert_eq!(key.codomain_vertices(), &[SectorId::new(1)]);
            assert_eq!(key.domain_vertices(), &[]);
        }
    }

    #[test]
    fn product_subblock_by_sectors_handles_simple_fusion_channels_without_manual_tree_keys() {
        type FpU1Rule = ProductFusionRule<FermionParityFusionRule, U1FusionRule>;
        type FpU1Su2Rule = ProductFusionRule<FpU1Rule, SU2FusionRule>;
        let left_rule = FpU1Rule::default();
        let rule = FpU1Su2Rule::default();
        let left_sector = |parity, charge| left_rule.encode_sector(parity, u1(charge));
        let sector = |parity, charge, twice_spin| {
            rule.encode_sector(left_sector(parity, charge), su2(twice_spin))
        };

        let a = sector(z2_odd(), 1, 1);
        let b = sector(z2_odd(), -1, 1);
        let c0 = sector(z2_even(), 0, 0);
        let c1 = sector(z2_even(), 0, 2);
        let dense = TensorMapSpace::<2, 1>::from_dims([1, 1], [1]).unwrap();
        let hom = FusionTreeHomSpace::new(
            FusionProductSpace::new([
                SectorLeg::new([(a, 1)], false),
                SectorLeg::new([(b, 1)], false),
            ]),
            FusionProductSpace::new([SectorLeg::new([(c0, 1), (c1, 1)], false)]),
        );
        let fusion_space = FusionTensorMapSpace::from_degeneracy_shapes(
            dense,
            hom,
            &rule,
            [vec![1, 1, 1], vec![1, 1, 1]],
        )
        .unwrap();
        let tensor =
            TensorMap::<i32, 2, 1>::from_vec_with_fusion_space(vec![100, 200], fusion_space)
                .unwrap();

        let c0_block = tensor.subblock_by_sectors(&rule, &[a, b, c0]).unwrap();
        let c1_block = tensor.subblock_by_sectors(&rule, &[a, b, c1]).unwrap();
        assert_eq!(c0_block.offset(), 0);
        assert_eq!(c0_block.data()[c0_block.offset()], 100);
        assert_eq!(c1_block.offset(), 1);
        assert_eq!(c1_block.data()[c1_block.offset()], 200);

        let all_c0_blocks = tensor.subblocks_by_sectors(&rule, &[a, b, c0]).unwrap();
        assert_eq!(all_c0_blocks.len(), 1);
        assert_eq!(all_c0_blocks[0].offset(), 0);
    }

    #[test]
    fn product_external_domain_sector_is_dualized_componentwise() {
        type FpU1Rule = ProductFusionRule<FermionParityFusionRule, U1FusionRule>;
        let rule = FpU1Rule::default();
        let a = rule.encode_sector(z2_odd(), u1(2));
        let external_domain = rule.dual(a);
        let hom = FusionTreeHomSpace::new(
            FusionProductSpace::new([SectorLeg::new([(a, 1)], false)]),
            FusionProductSpace::new([SectorLeg::new([(a, 1)], false)]),
        );

        let keys = hom
            .fusion_tree_keys_from_external_sectors(&rule, &[a, external_domain])
            .unwrap();
        assert_eq!(keys.len(), 1);
        assert_eq!(keys[0].codomain_uncoupled(), &[a]);
        assert_eq!(keys[0].domain_uncoupled(), &[a]);
        assert_eq!(keys[0].coupled(), Some(a));

        let err = hom
            .fusion_tree_keys_from_external_sectors(&rule, &[a, a])
            .unwrap_err();
        assert_eq!(
            err,
            CoreError::InvalidSector {
                sector: external_domain
            }
        );
    }

    #[test]
    #[should_panic(expected = "Z2 fusion received an invalid sector")]
    fn product_fusion_rule_panics_on_component_invalid_sector_like_existing_rules() {
        type FpU1Rule = ProductFusionRule<FermionParityFusionRule, U1FusionRule>;
        let rule = FpU1Rule::default();
        let invalid_left_component = rule.encode_sector(SectorId::new(2), u1(0));
        let valid = rule.encode_sector(z2_even(), u1(0));

        let _ = rule.fusion_channels(invalid_left_component, valid);
    }

    #[test]
    fn public_su2_irrep_fusion_channels_match_doubled_spin_order() {
        let rule = SU2FusionRule;

        assert_eq!(
            rule.fusion_channels(
                SU2Irrep::from_twice_spin(1).sector_id(),
                SU2Irrep::from_twice_spin(2).sector_id(),
            )
            .to_vec(),
            vec![
                SU2Irrep::from_twice_spin(1).sector_id(),
                SU2Irrep::from_twice_spin(3).sector_id(),
            ]
        );
    }

    #[test]
    fn public_su2_f_and_r_symbols_match_tensorkit_values() {
        let rule = SU2FusionRule;
        let s = |twice_spin| SU2Irrep::from_twice_spin(twice_spin).sector_id();
        let cases = [
            ((1, 1, 1, 1, 0, 0), -0.5),
            ((1, 1, 1, 1, 0, 2), 0.866_025_403_784_438_6),
            ((1, 1, 1, 1, 2, 0), 0.866_025_403_784_438_6),
            ((1, 1, 1, 1, 2, 2), 0.5),
            ((1, 2, 1, 2, 1, 1), -1.0 / 3.0),
            ((2, 2, 2, 2, 0, 2), -0.577_350_269_189_625_7),
            ((2, 2, 2, 2, 2, 2), 0.5),
            ((1, 1, 2, 2, 1, 1), 0.0),
        ];

        for ((a, b, c, d, e, f), expected) in cases {
            let actual = rule.f_symbol_scalar(s(a), s(b), s(c), s(d), s(e), s(f));
            assert!(
                (actual - expected).abs() < 1.0e-12,
                "F({a},{b},{c},{d},{e},{f}) = {actual}, expected {expected}"
            );
        }
        assert_eq!(rule.r_symbol_scalar(s(1), s(1), s(0)), -1.0);
        assert_eq!(rule.r_symbol_scalar(s(1), s(1), s(2)), 1.0);
        assert_eq!(rule.r_symbol_scalar(s(1), s(2), s(0)), 0.0);
    }

    #[derive(Clone, Debug)]
    struct ProbeTreeKey(usize);

    static PROBE_TREE_KEY_EQ_CALLS: std::sync::atomic::AtomicUsize =
        std::sync::atomic::AtomicUsize::new(0);
    static PROBE_TREE_KEY_HASH_CALLS: std::sync::atomic::AtomicUsize =
        std::sync::atomic::AtomicUsize::new(0);

    impl PartialEq for ProbeTreeKey {
        fn eq(&self, other: &Self) -> bool {
            PROBE_TREE_KEY_EQ_CALLS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            self.0 == other.0
        }
    }

    impl Eq for ProbeTreeKey {}

    impl std::hash::Hash for ProbeTreeKey {
        fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
            PROBE_TREE_KEY_HASH_CALLS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            std::hash::Hash::hash(&self.0, state);
        }
    }

    #[test]
    fn fusion_term_accumulator_keeps_singleton_path_and_hashes_multi_terms() {
        PROBE_TREE_KEY_EQ_CALLS.store(0, std::sync::atomic::Ordering::Relaxed);
        PROBE_TREE_KEY_HASH_CALLS.store(0, std::sync::atomic::Ordering::Relaxed);
        let mut singleton = FusionTermAccumulator::new();
        singleton.push(ProbeTreeKey(7), 3usize);
        let singleton_terms = singleton.into_vec();
        assert_eq!(singleton_terms.len(), 1);
        let (singleton_key, singleton_coefficient) = &singleton_terms[0];
        assert_eq!(singleton_key.0, 7);
        assert_eq!(*singleton_coefficient, 3);
        assert_eq!(
            PROBE_TREE_KEY_HASH_CALLS.load(std::sync::atomic::Ordering::Relaxed),
            0
        );

        const DISTINCT: usize = 512;
        const ROUNDS: usize = 4;
        PROBE_TREE_KEY_EQ_CALLS.store(0, std::sync::atomic::Ordering::Relaxed);
        PROBE_TREE_KEY_HASH_CALLS.store(0, std::sync::atomic::Ordering::Relaxed);
        let mut accumulator = FusionTermAccumulator::new();
        for _ in 0..ROUNDS {
            for key in 0..DISTINCT {
                accumulator.push(ProbeTreeKey(key), 1usize);
            }
        }
        let terms = accumulator.into_vec();
        assert_eq!(terms.len(), DISTINCT);
        for (index, (key, coefficient)) in terms.iter().enumerate() {
            assert_eq!(key.0, index);
            assert_eq!(*coefficient, ROUNDS);
        }
        let eq_calls = PROBE_TREE_KEY_EQ_CALLS.load(std::sync::atomic::Ordering::Relaxed);
        assert!(
            eq_calls < DISTINCT * ROUNDS * 8,
            "HashMap-backed accumulation should stay linear; saw {eq_calls} equality checks"
        );
        assert!(
            PROBE_TREE_KEY_HASH_CALLS.load(std::sync::atomic::Ordering::Relaxed) > DISTINCT,
            "multi-term accumulation should use the hash path"
        );
    }

    #[test]
    fn multiplicity_free_su2_braid_expands_innerline_channels() {
        let rule = SU2FusionRule;
        let tree = FusionTreeKey::from_sector_ids(
            [1, 1, 1, 1],
            Some(0),
            [false, false, false, false],
            [0, 1],
            [1, 1, 1],
        );

        let braided =
            multiplicity_free_braid_tree(&rule, &tree, &[0, 2, 1, 3], &[0, 1, 2, 3]).unwrap();

        assert_eq!(braided.len(), 2);
        assert_eq!(braided[0].0.uncoupled(), &[SectorId::new(1); 4]);
        assert_eq!(
            braided[0].0.innerlines(),
            &[SectorId::new(0), SectorId::new(1)]
        );
        assert!((braided[0].1 - 0.5).abs() < 1.0e-12);
        assert_eq!(
            braided[1].0.innerlines(),
            &[SectorId::new(2), SectorId::new(1)]
        );
        assert!((braided[1].1 - 0.866_025_403_784_438_6).abs() < 1.0e-12);
    }

    #[test]
    fn multiplicity_free_su2_repartition_matches_tensorkit_bend_factor() {
        let rule = SU2FusionRule;
        let source = FusionTreeBlockKey::pair_from_sector_ids(
            [1],
            [1],
            Some(1),
            [false],
            [false],
            [],
            [],
            [],
            [],
        );

        let all_codomain = multiplicity_free_repartition_tree_pair(&rule, &source, 2).unwrap();
        assert_eq!(all_codomain.len(), 1);
        assert_eq!(
            all_codomain[0].0.codomain_uncoupled(),
            &[SectorId::new(1); 2]
        );
        assert_eq!(all_codomain[0].0.codomain_is_dual(), &[false, true]);
        assert_eq!(all_codomain[0].0.codomain_innerlines(), &[]);
        assert_eq!(all_codomain[0].0.codomain_vertices(), &[SectorId::new(1)]);
        assert_eq!(
            all_codomain[0].0.codomain_tree().coupled(),
            Some(SectorId::new(0))
        );
        assert_eq!(all_codomain[0].0.domain_uncoupled(), &[]);
        assert_eq!(
            all_codomain[0].0.domain_tree().coupled(),
            Some(SectorId::new(0))
        );
        assert!((all_codomain[0].1 - 2.0_f64.sqrt()).abs() < 1.0e-12);

        let all_domain = multiplicity_free_repartition_tree_pair(&rule, &source, 0).unwrap();
        assert_eq!(all_domain.len(), 1);
        assert_eq!(all_domain[0].0.codomain_uncoupled(), &[]);
        assert_eq!(
            all_domain[0].0.codomain_tree().coupled(),
            Some(SectorId::new(0))
        );
        assert_eq!(all_domain[0].0.domain_uncoupled(), &[SectorId::new(1); 2]);
        assert_eq!(all_domain[0].0.domain_is_dual(), &[false, true]);
        assert!((all_domain[0].1 - 2.0_f64.sqrt()).abs() < 1.0e-12);
    }

    #[test]
    fn multiplicity_free_su2_permute_tree_pair_matches_tensorkit_swap() {
        let rule = SU2FusionRule;
        let source = FusionTreeBlockKey::pair_from_sector_ids(
            [1],
            [1],
            Some(1),
            [false],
            [false],
            [],
            [],
            [],
            [],
        );

        let permuted = multiplicity_free_permute_tree_pair(&rule, &source, &[1], &[0]).unwrap();

        assert_eq!(permuted.len(), 1);
        assert_eq!(permuted[0].0.codomain_uncoupled(), &[SectorId::new(1)]);
        assert_eq!(permuted[0].0.domain_uncoupled(), &[SectorId::new(1)]);
        assert_eq!(permuted[0].0.codomain_is_dual(), &[true]);
        assert_eq!(permuted[0].0.domain_is_dual(), &[true]);
        assert_eq!(
            permuted[0].0.codomain_tree().coupled(),
            Some(SectorId::new(1))
        );
        assert_eq!(
            permuted[0].0.domain_tree().coupled(),
            Some(SectorId::new(1))
        );
        assert!((permuted[0].1 - 1.0).abs() < 1.0e-12);
    }

    fn u1_nonselfdual_tree_pair_fixture() -> FusionTreeBlockKey {
        FusionTreeBlockKey::pair(
            FusionTreeKey::new(
                [u1(1), u1(2)],
                Some(u1(3)),
                [false, false],
                Vec::<SectorId>::new(),
                [SectorId::new(1)],
            ),
            FusionTreeKey::new(
                [u1(3)],
                Some(u1(3)),
                [false],
                Vec::<SectorId>::new(),
                Vec::<SectorId>::new(),
            ),
        )
    }

    #[test]
    fn u1_bendright_dualizes_visible_sector_and_flips_isdual_like_tensorkit() {
        let out = multiplicity_free_bendright_tree_pair(
            &U1FusionRule,
            &u1_nonselfdual_tree_pair_fixture(),
        )
        .unwrap();

        assert_eq!(out.len(), 1);
        assert_eq!(out[0].1, 1.0);
        assert_eq!(out[0].0.codomain_uncoupled(), &[u1(1)]);
        assert_eq!(out[0].0.codomain_tree().coupled(), Some(u1(1)));
        assert_eq!(out[0].0.codomain_is_dual(), &[false]);
        assert_eq!(out[0].0.codomain_innerlines(), &[]);
        assert_eq!(out[0].0.codomain_vertices(), &[]);
        assert_eq!(out[0].0.domain_uncoupled(), &[u1(3), u1(-2)]);
        assert_eq!(out[0].0.domain_tree().coupled(), Some(u1(1)));
        assert_eq!(out[0].0.domain_is_dual(), &[false, true]);
        assert_eq!(out[0].0.domain_innerlines(), &[]);
        assert_eq!(out[0].0.domain_vertices(), &[SectorId::new(1)]);
    }

    #[test]
    fn u1_foldright_dualizes_first_visible_sector_and_flips_isdual_like_tensorkit() {
        let out = multiplicity_free_foldright_tree_pair(
            &U1FusionRule,
            &u1_nonselfdual_tree_pair_fixture(),
        )
        .unwrap();

        assert_eq!(out.len(), 1);
        assert_eq!(out[0].1, 1.0);
        assert_eq!(out[0].0.codomain_uncoupled(), &[u1(2)]);
        assert_eq!(out[0].0.codomain_tree().coupled(), Some(u1(2)));
        assert_eq!(out[0].0.codomain_is_dual(), &[false]);
        assert_eq!(out[0].0.codomain_innerlines(), &[]);
        assert_eq!(out[0].0.codomain_vertices(), &[]);
        assert_eq!(out[0].0.domain_uncoupled(), &[u1(-1), u1(3)]);
        assert_eq!(out[0].0.domain_tree().coupled(), Some(u1(2)));
        assert_eq!(out[0].0.domain_is_dual(), &[true, false]);
        assert_eq!(out[0].0.domain_innerlines(), &[]);
        assert_eq!(out[0].0.domain_vertices(), &[SectorId::new(1)]);
    }

    #[test]
    fn u1_repartition_to_all_domain_matches_tensorkit_nonselfdual_fixture() {
        let out = multiplicity_free_repartition_tree_pair(
            &U1FusionRule,
            &u1_nonselfdual_tree_pair_fixture(),
            0,
        )
        .unwrap();

        assert_eq!(out.len(), 1);
        assert_eq!(out[0].1, 1.0);
        assert_eq!(out[0].0.codomain_uncoupled(), &[]);
        assert_eq!(out[0].0.codomain_tree().coupled(), Some(u1(0)));
        assert_eq!(out[0].0.codomain_is_dual(), &[]);
        assert_eq!(out[0].0.codomain_innerlines(), &[]);
        assert_eq!(out[0].0.codomain_vertices(), &[]);
        assert_eq!(out[0].0.domain_uncoupled(), &[u1(3), u1(-2), u1(-1)]);
        assert_eq!(out[0].0.domain_tree().coupled(), Some(u1(0)));
        assert_eq!(out[0].0.domain_is_dual(), &[false, true, true]);
        assert_eq!(out[0].0.domain_innerlines(), &[u1(1)]);
        assert_eq!(
            out[0].0.domain_vertices(),
            &[SectorId::new(1), SectorId::new(1)]
        );
    }

    #[test]
    fn u1_repartition_to_all_codomain_matches_tensorkit_nonselfdual_fixture() {
        let out = multiplicity_free_repartition_tree_pair(
            &U1FusionRule,
            &u1_nonselfdual_tree_pair_fixture(),
            3,
        )
        .unwrap();

        assert_eq!(out.len(), 1);
        assert_eq!(out[0].1, 1.0);
        assert_eq!(out[0].0.codomain_uncoupled(), &[u1(1), u1(2), u1(-3)]);
        assert_eq!(out[0].0.codomain_tree().coupled(), Some(u1(0)));
        assert_eq!(out[0].0.codomain_is_dual(), &[false, false, true]);
        assert_eq!(out[0].0.codomain_innerlines(), &[u1(3)]);
        assert_eq!(
            out[0].0.codomain_vertices(),
            &[SectorId::new(1), SectorId::new(1)]
        );
        assert_eq!(out[0].0.domain_uncoupled(), &[]);
        assert_eq!(out[0].0.domain_tree().coupled(), Some(u1(0)));
        assert_eq!(out[0].0.domain_is_dual(), &[]);
        assert_eq!(out[0].0.domain_innerlines(), &[]);
        assert_eq!(out[0].0.domain_vertices(), &[]);
    }

    #[test]
    fn u1_transpose_cyclic_23_1_matches_tensorkit_nonselfdual_fixture() {
        let out = multiplicity_free_transpose_tree_pair(
            &U1FusionRule,
            &u1_nonselfdual_tree_pair_fixture(),
            &[1, 2],
            &[0],
        )
        .unwrap();

        assert_eq!(out.len(), 1);
        assert_eq!(out[0].1, 1.0);
        assert_eq!(out[0].0.codomain_uncoupled(), &[u1(2), u1(-3)]);
        assert_eq!(out[0].0.codomain_tree().coupled(), Some(u1(-1)));
        assert_eq!(out[0].0.codomain_is_dual(), &[false, true]);
        assert_eq!(out[0].0.codomain_innerlines(), &[]);
        assert_eq!(out[0].0.codomain_vertices(), &[SectorId::new(1)]);
        assert_eq!(out[0].0.domain_uncoupled(), &[u1(-1)]);
        assert_eq!(out[0].0.domain_tree().coupled(), Some(u1(-1)));
        assert_eq!(out[0].0.domain_is_dual(), &[true]);
        assert_eq!(out[0].0.domain_innerlines(), &[]);
        assert_eq!(out[0].0.domain_vertices(), &[]);
    }

    #[test]
    fn typed_sector_homspace_builds_u1_tree_key() {
        let rule = U1FusionRule;
        let hom = FusionTreeHomSpace::from_sectors([(U1Irrep::new(2), 1)], [(U1Irrep::new(2), 1)]);

        let key = hom
            .unique_fusion_tree_key_from_external_sectors(
                &rule,
                &[U1Irrep::new(2).sector_id(), U1Irrep::new(-2).sector_id()],
            )
            .unwrap();

        assert_eq!(key.codomain_uncoupled(), &[U1Irrep::new(2).sector_id()]);
        assert_eq!(key.domain_uncoupled(), &[U1Irrep::new(2).sector_id()]);
        assert_eq!(key.coupled(), Some(U1Irrep::new(2).sector_id()));
    }

    #[test]
    fn fusion_tensor_space_builds_subblockstructure_from_homspace() {
        let rule = Z2FusionRule;
        let dense = TensorMapSpace::<1, 1>::from_dims([2], [2]).unwrap();
        let hom = FusionTreeHomSpace::new(
            FusionProductSpace::new([SectorLeg::new(
                [(SectorId::new(0), 1), (SectorId::new(1), 3)],
                false,
            )]),
            FusionProductSpace::new([SectorLeg::new(
                [(SectorId::new(0), 2), (SectorId::new(1), 1)],
                false,
            )]),
        );

        let fusion_space = FusionTensorMapSpace::from_degeneracy_shapes(
            dense,
            hom,
            &rule,
            [vec![1, 2], vec![3, 1]],
        )
        .unwrap();

        assert_eq!(fusion_space.subblock_structure().block_count(), 2);
        assert_eq!(fusion_space.required_len().unwrap(), 5);
        assert_eq!(
            fusion_space.subblock_structure().block(0).unwrap().key(),
            &BlockKey::from(FusionTreeBlockKey::pair_from_sector_ids(
                [0],
                [0],
                Some(0),
                [false],
                [false],
                [],
                [],
                [],
                [],
            ))
        );
        assert_eq!(
            fusion_space.subblock_structure().block(1).unwrap().shape(),
            &[3, 1]
        );
    }

    #[test]
    fn fusion_tensor_space_rejects_homspace_rank_mismatch() {
        let rule = Z2FusionRule;
        let dense = TensorMapSpace::<1, 1>::from_dims([2], [2]).unwrap();
        let hom = FusionTreeHomSpace::from_sector_ids([(0, 1), (1, 1)], [(0, 1)]);

        let err = FusionTensorMapSpace::from_degeneracy_shapes(
            dense,
            hom,
            &rule,
            [vec![1, 1], vec![1, 1]],
        )
        .unwrap_err();

        assert_eq!(
            err,
            CoreError::StructureRankMismatch {
                expected: 1,
                actual: 2,
            }
        );
    }

    #[test]
    fn tensormap_subblock_by_sectors_matches_z2_unique() {
        let rule = Z2FusionRule;
        let dense = TensorMapSpace::<1, 1>::from_dims([2], [2]).unwrap();
        let hom = FusionTreeHomSpace::new(
            FusionProductSpace::new([SectorLeg::new(
                [(SectorId::new(0), 1), (SectorId::new(1), 1)],
                false,
            )]),
            FusionProductSpace::new([SectorLeg::new(
                [(SectorId::new(0), 1), (SectorId::new(1), 1)],
                false,
            )]),
        );
        let fusion_space = FusionTensorMapSpace::from_degeneracy_shapes(
            dense,
            hom,
            &rule,
            [vec![1, 1], vec![1, 1]],
        )
        .unwrap();
        let tensor =
            TensorMap::<i32, 1, 1>::from_vec_with_fusion_space(vec![10, 20], fusion_space).unwrap();

        let block = tensor
            .subblock_by_sectors(&rule, &[SectorId::new(1), SectorId::new(1)])
            .unwrap();

        assert_eq!(block.offset(), 1);
        assert_eq!(block.data()[block.offset()], 20);
    }

    #[test]
    fn tensormap_subblock_by_sectors_dualizes_z4_domain_sector() {
        let rule = Z4PointedRule;
        let dense = TensorMapSpace::<1, 1>::from_dims([1], [1]).unwrap();
        let hom = FusionTreeHomSpace::new(
            FusionProductSpace::new([SectorLeg::new([(SectorId::new(1), 1)], false)]),
            FusionProductSpace::new([SectorLeg::new([(SectorId::new(1), 1)], false)]),
        );
        let fusion_space =
            FusionTensorMapSpace::from_degeneracy_shapes(dense, hom, &rule, [vec![1, 1]]).unwrap();
        let tensor =
            TensorMap::<f64, 1, 1>::from_vec_with_fusion_space(vec![3.5], fusion_space).unwrap();

        let block = tensor
            .subblock_by_sectors(&rule, &[SectorId::new(1), SectorId::new(3)])
            .unwrap();

        assert_eq!(block.offset(), 0);
        assert_eq!(block.data()[0], 3.5);
    }

    #[test]
    fn tensormap_subblock_by_sectors_handles_fermionic_z2_key() {
        let rule = FermionParityFusionRule;
        let dense = TensorMapSpace::<1, 1>::from_dims([1], [1]).unwrap();
        let hom = FusionTreeHomSpace::from_sector_ids([(1, 1)], [(1, 1)]);
        let fusion_space =
            FusionTensorMapSpace::from_degeneracy_shapes(dense, hom, &rule, [vec![1, 1]]).unwrap();
        let mut tensor =
            TensorMap::<i32, 1, 1>::from_vec_with_fusion_space(vec![7], fusion_space).unwrap();

        {
            let mut block = tensor
                .subblock_mut_by_sectors(&rule, &[SectorId::new(1), SectorId::new(1)])
                .unwrap();
            let offset = block.offset();
            block.data_mut()[offset] = 11;
        }

        assert_eq!(tensor.data(), &[11]);
    }

    #[test]
    fn tensormap_subblock_by_sectors_handles_product_pointed_rule() {
        let rule = Z2xZ3PointedRule;
        let codomain_sector = Z2xZ3PointedRule::encode(1, 2);
        let domain_tree_sector = rule.dual(Z2xZ3PointedRule::encode(1, 1));
        let dense = TensorMapSpace::<1, 1>::from_dims([1], [1]).unwrap();
        let hom = FusionTreeHomSpace::new(
            FusionProductSpace::new([SectorLeg::new([(codomain_sector, 1)], false)]),
            FusionProductSpace::new([SectorLeg::new([(domain_tree_sector, 1)], false)]),
        );
        let fusion_space =
            FusionTensorMapSpace::from_degeneracy_shapes(dense, hom, &rule, [vec![1, 1]]).unwrap();
        let tensor =
            TensorMap::<i32, 1, 1>::from_vec_with_fusion_space(vec![42], fusion_space).unwrap();

        let block = tensor
            .subblock_by_sectors(&rule, &[codomain_sector, Z2xZ3PointedRule::encode(1, 1)])
            .unwrap();

        assert_eq!(block.data()[block.offset()], 42);
    }

    #[test]
    fn subblock_by_sectors_requires_fusion_tensor_space() {
        let rule = Z2FusionRule;
        let space = TensorMapSpace::<1, 1>::from_dims([1], [1]).unwrap();
        let tensor = TensorMap::<f64, 1, 1>::from_vec(vec![1.0], space).unwrap();

        let err = tensor
            .subblock_by_sectors(&rule, &[SectorId::new(0), SectorId::new(0)])
            .unwrap_err();

        assert_eq!(err, CoreError::MissingFusionSpace);
    }

    #[test]
    fn packed_block_structure_records_rank_offsets_and_required_len() {
        let structure = BlockStructure::packed_column_major(2, [vec![2, 3], vec![1, 4]]).unwrap();

        assert_eq!(structure.rank(), 2);
        assert_eq!(structure.block_count(), 2);
        assert_eq!(structure.sector_structure().block_count(), 2);
        assert_eq!(structure.degeneracy_structure().block_count(), 2);
        let first = structure.block(0).unwrap();
        assert_eq!(first.key(), &BlockKey::ordinal(0));
        assert_eq!(first.shape(), &[2, 3]);
        assert_eq!(first.strides(), &[1, 2]);
        assert_eq!(first.offset(), 0);
        let second = structure.block(1).unwrap();
        assert_eq!(second.key(), &BlockKey::ordinal(1));
        assert_eq!(second.shape(), &[1, 4]);
        assert_eq!(second.strides(), &[1, 1]);
        assert_eq!(second.offset(), 6);
        assert_eq!(structure.required_len().unwrap(), 10);
    }

    #[test]
    fn tensormap_accepts_packed_block_structure() {
        let space = TensorMapSpace::<2, 0>::from_dims([4, 4], []).unwrap();
        let structure = BlockStructure::packed_column_major(2, [vec![2, 3], vec![1, 4]]).unwrap();
        let tensor = TensorMap::<f64, 2, 0>::from_vec_with_structure(
            (0..10).map(|x| x as f64).collect(),
            space,
            structure,
        )
        .unwrap();

        assert_eq!(tensor.data().len(), 10);
        assert_eq!(tensor.dim(), 10);
        assert_eq!(tensor.storage_dim(), 10);
        assert_eq!(tensor.dense_dim(), 16);
        assert_eq!(tensor.structure().rank(), 2);

        let first = tensor.block(0).unwrap();
        assert_eq!(first.shape(), &[2, 3]);
        assert_eq!(first.offset(), 0);

        let second = tensor.block(1).unwrap();
        assert_eq!(second.shape(), &[1, 4]);
        assert_eq!(second.offset(), 6);
    }

    #[test]
    fn block_structure_rejects_duplicate_keys() {
        let first =
            BlockSpec::column_major_with_key(BlockKey::sector_ids([7]), vec![2, 2], 0).unwrap();
        let second =
            BlockSpec::column_major_with_key(BlockKey::sector_ids([7]), vec![1, 3], 4).unwrap();

        let err = BlockStructure::from_blocks_with_rank(2, vec![first, second]).unwrap_err();

        assert_eq!(
            err,
            CoreError::DuplicateBlockKey {
                key: BlockKey::sector_ids([7])
            }
        );
    }

    #[test]
    fn fusion_tree_group_key_records_external_sector_tuples_and_duality() {
        let group = FusionTreeGroupKey::from_sector_ids([2, 3], [5], [false, true], [true]);

        assert_eq!(
            group.codomain_uncoupled(),
            &[SectorId::new(2), SectorId::new(3)]
        );
        assert_eq!(group.domain_uncoupled(), &[SectorId::new(5)]);
        assert_eq!(group.codomain_is_dual(), &[false, true]);
        assert_eq!(group.domain_is_dual(), &[true]);

        let same = FusionTreeGroupKey::new(
            [SectorId::new(2), SectorId::new(3)],
            [SectorId::new(5)],
            [false, true],
            [true],
        );
        assert_eq!(group, same);
    }

    #[test]
    fn fusion_tree_block_key_records_tensorkit_subblock_pair_fields() {
        let key = FusionTreeBlockKey::pair_from_sector_ids(
            [2, 3],
            [5, 7],
            Some(11),
            [false, true],
            [true, false],
            [13],
            [17],
            [19, 23],
            [29, 31],
        );

        assert_eq!(
            key.codomain_uncoupled(),
            &[SectorId::new(2), SectorId::new(3)]
        );
        assert_eq!(
            key.domain_uncoupled(),
            &[SectorId::new(5), SectorId::new(7)]
        );
        assert_eq!(key.coupled(), Some(SectorId::new(11)));
        assert_eq!(key.codomain_is_dual(), &[false, true]);
        assert_eq!(key.domain_is_dual(), &[true, false]);
        assert_eq!(key.codomain_innerlines(), &[SectorId::new(13)]);
        assert_eq!(key.domain_innerlines(), &[SectorId::new(17)]);
        assert_eq!(
            key.codomain_vertices(),
            &[SectorId::new(19), SectorId::new(23)]
        );
        assert_eq!(
            key.domain_vertices(),
            &[SectorId::new(29), SectorId::new(31)]
        );

        let group = key.group_key();
        assert_eq!(group.codomain_uncoupled(), key.codomain_uncoupled());
        assert_eq!(group.domain_uncoupled(), key.domain_uncoupled());
        assert_eq!(group.codomain_is_dual(), key.codomain_is_dual());
        assert_eq!(group.domain_is_dual(), key.domain_is_dual());
    }

    #[test]
    fn fusion_tree_homspace_generates_canonical_coupled_sector_order() {
        let rule = BranchingMultiplicityFreeRule;
        let hom = FusionTreeHomSpace::from_sector_ids([(1, 1), (1, 1)], [(1, 1), (1, 1)]);

        let keys = hom.fusion_tree_keys(&rule);

        assert_eq!(keys.len(), 2);
        assert_eq!(keys[0].coupled(), Some(SectorId::new(0)));
        assert_eq!(keys[1].coupled(), Some(SectorId::new(2)));
        assert_eq!(
            keys[0].codomain_uncoupled(),
            &[SectorId::new(1), SectorId::new(1)]
        );
        assert_eq!(
            keys[0].domain_uncoupled(),
            &[SectorId::new(1), SectorId::new(1)]
        );
        assert!(keys[0].codomain_innerlines().is_empty());
        assert!(keys[0].domain_innerlines().is_empty());
        assert_eq!(keys[0].codomain_vertices(), &[SectorId::new(1)]);
        assert_eq!(keys[0].domain_vertices(), &[SectorId::new(1)]);

        let sector = hom.sector_structure(&rule).unwrap();
        let groups = sector.fusion_tree_groups();
        assert_eq!(groups.len(), 1);
        assert_eq!(groups[0].block_indices(), &[0, 1]);
        assert_eq!(
            groups[0].group_key(),
            &FusionTreeGroupKey::from_sector_ids([1, 1], [1, 1], [false, false], [false, false])
        );
    }

    #[test]
    fn unique_homspace_builds_subblock_key_from_external_sectors() {
        let rule = Z2FusionRule;
        let hom = FusionTreeHomSpace::from_sector_ids([(1, 1)], [(1, 1)]);

        let key = hom
            .unique_fusion_tree_key_from_external_sectors(
                &rule,
                &[SectorId::new(1), SectorId::new(1)],
            )
            .unwrap();

        assert_eq!(key.codomain_uncoupled(), &[SectorId::new(1)]);
        assert_eq!(key.domain_uncoupled(), &[SectorId::new(1)]);
        assert_eq!(key.coupled(), Some(SectorId::new(1)));
        assert_eq!(key.codomain_is_dual(), &[false]);
        assert_eq!(key.domain_is_dual(), &[false]);
    }

    #[test]
    fn unique_homspace_dualizes_domain_external_sectors_like_tensorkit() {
        let rule = Z4PointedRule;
        let hom = FusionTreeHomSpace::new(
            FusionProductSpace::new([SectorLeg::new([(SectorId::new(1), 1)], false)]),
            FusionProductSpace::new([SectorLeg::new([(SectorId::new(1), 1)], false)]),
        );

        let key = hom
            .unique_fusion_tree_key_from_external_sectors(
                &rule,
                &[SectorId::new(1), SectorId::new(3)],
            )
            .unwrap();

        assert_eq!(key.codomain_uncoupled(), &[SectorId::new(1)]);
        assert_eq!(key.domain_uncoupled(), &[SectorId::new(1)]);
        assert_eq!(key.coupled(), Some(SectorId::new(1)));
    }

    #[test]
    fn fusion_tree_block_key_external_sectors_restore_visible_domain_sector() {
        let rule = Z4PointedRule;
        let hom = FusionTreeHomSpace::new(
            FusionProductSpace::new([SectorLeg::new([(SectorId::new(1), 1)], true)]),
            FusionProductSpace::new([SectorLeg::new([(SectorId::new(1), 1)], false)]),
        );
        let key = hom
            .unique_fusion_tree_key_from_external_sectors(
                &rule,
                &[SectorId::new(1), SectorId::new(3)],
            )
            .unwrap();

        assert_eq!(key.codomain_uncoupled(), &[SectorId::new(1)]);
        assert_eq!(key.domain_uncoupled(), &[SectorId::new(1)]);
        assert_eq!(
            key.external_sectors(&rule),
            vec![SectorId::new(1), SectorId::new(3)]
        );
        assert_eq!(key.external_is_dual(), vec![true, false]);
    }

    #[test]
    fn fusion_tree_homspace_compose_matches_nonselfdual_domain_convention() {
        // TensorKit: `A * B` needs `domain(A) == codomain(B)` as spaces, so
        // the stored legs pair verbatim even for non-self-dual sectors
        // (Julia check: `rand(U1Space(0=>1,1=>1) ← same) * itself` works).
        let rule = U1FusionRule;
        let physical = u1(1);
        let lhs = FusionTreeHomSpace::new(
            FusionProductSpace::new([SectorLeg::new([(u1(2), 1)], false)]),
            FusionProductSpace::new([SectorLeg::new([(physical, 1)], false)]),
        );
        let rhs = FusionTreeHomSpace::new(
            FusionProductSpace::new([SectorLeg::new([(physical, 1)], false)]),
            FusionProductSpace::new([SectorLeg::new([(u1(3), 1)], false)]),
        );

        let composed = FusionTreeHomSpace::compose(&rule, &lhs, &rhs).unwrap();

        assert_eq!(composed.codomain().legs()[0].sectors(), &[u1(2)]);
        assert_eq!(composed.domain().legs()[0].sectors(), &[u1(3)]);
    }

    #[test]
    fn fusion_tree_homspace_select_dualizes_axes_like_tensorkit() {
        let rule = U1FusionRule;
        let hom = FusionTreeHomSpace::new(
            FusionProductSpace::new([
                SectorLeg::new([(u1(1), 1)], false),
                SectorLeg::new([(u1(2), 1)], true),
            ]),
            FusionProductSpace::new([
                SectorLeg::new([(u1(3), 1)], false),
                SectorLeg::new([(u1(-5), 1)], true),
            ]),
        );

        let selected = hom.select(&rule, &[2, 0], &[1, 3]).unwrap();

        assert_eq!(selected.codomain().legs()[0].sectors(), &[u1(-3)]);
        assert!(selected.codomain().legs()[0].is_dual());
        assert_eq!(selected.codomain().legs()[1].sectors(), &[u1(1)]);
        assert!(!selected.codomain().legs()[1].is_dual());
        assert_eq!(selected.domain().legs()[0].sectors(), &[u1(-2)]);
        assert!(!selected.domain().legs()[0].is_dual());
        assert_eq!(selected.domain().legs()[1].sectors(), &[u1(-5)]);
        assert!(selected.domain().legs()[1].is_dual());
    }

    #[test]
    fn fusion_tree_homspace_permute_requires_full_axis_permutation() {
        let rule = U1FusionRule;
        let hom = FusionTreeHomSpace::from_sectors([(u1(0), 1), (u1(1), 1)], [(u1(2), 1)]);

        let err = hom.permute(&rule, &[0], &[2]).unwrap_err();
        assert_eq!(
            err,
            CoreError::InvalidPermutation {
                permutation: vec![0, 2],
                rank: 3,
            }
        );

        let err = hom.permute(&rule, &[0, 0], &[2]).unwrap_err();
        assert_eq!(
            err,
            CoreError::InvalidPermutation {
                permutation: vec![0, 0, 2],
                rank: 3,
            }
        );
    }

    #[test]
    fn fusion_tree_homspace_tensorcontract_preserves_canonical_compose() {
        let rule = U1FusionRule;
        let lhs = FusionTreeHomSpace::new(
            FusionProductSpace::new([SectorLeg::new([(u1(2), 1)], false)]),
            FusionProductSpace::new([SectorLeg::new([(u1(1), 1)], false)]),
        );
        let rhs = FusionTreeHomSpace::new(
            FusionProductSpace::new([SectorLeg::new([(u1(1), 1)], false)]),
            FusionProductSpace::new([SectorLeg::new([(u1(3), 1)], false)]),
        );

        let expected = FusionTreeHomSpace::compose(&rule, &lhs, &rhs).unwrap();
        let actual =
            FusionTreeHomSpace::tensorcontract_homspace(&rule, &lhs, &rhs, &[1], &[0], &[0, 1], 1)
                .unwrap();

        assert_eq!(actual, expected);
    }

    #[test]
    fn fusion_tree_homspace_tensorcontract_matches_tensorkit_structural_formula() {
        let rule = U1FusionRule;
        let lhs = FusionTreeHomSpace::new(
            FusionProductSpace::new([SectorLeg::new([(u1(1), 1)], false)]),
            FusionProductSpace::new([SectorLeg::new([(u1(5), 1)], false)]),
        );
        let rhs = FusionTreeHomSpace::new(
            FusionProductSpace::new([SectorLeg::new([(u1(7), 1)], false)]),
            FusionProductSpace::new([SectorLeg::new([(u1(1), 1)], false)]),
        );

        let lhs_permuted = lhs.permute(&rule, &[1], &[0]).unwrap();
        let rhs_permuted = rhs.permute(&rule, &[1], &[0]).unwrap();
        let expected = FusionTreeHomSpace::compose(&rule, &lhs_permuted, &rhs_permuted)
            .unwrap()
            .permute(&rule, &[0], &[1])
            .unwrap();
        let actual =
            FusionTreeHomSpace::tensorcontract_homspace(&rule, &lhs, &rhs, &[0], &[1], &[0, 1], 1)
                .unwrap();

        assert_eq!(actual, expected);
        assert_eq!(actual.codomain().legs()[0].sectors(), &[u1(-5)]);
        assert!(actual.codomain().legs()[0].is_dual());
        assert_eq!(actual.domain().legs()[0].sectors(), &[u1(-7)]);
        assert!(actual.domain().legs()[0].is_dual());
    }

    #[test]
    fn fusion_tree_homspace_tensorcontract_accepts_output_permutation_structurally() {
        let rule = U1FusionRule;
        let lhs = FusionTreeHomSpace::new(
            FusionProductSpace::new([SectorLeg::new([(u1(2), 1)], false)]),
            FusionProductSpace::new([SectorLeg::new([(u1(1), 1)], false)]),
        );
        let rhs = FusionTreeHomSpace::new(
            FusionProductSpace::new([SectorLeg::new([(u1(1), 1)], false)]),
            FusionProductSpace::new([SectorLeg::new([(u1(3), 1)], false)]),
        );

        let composed =
            FusionTreeHomSpace::tensorcontract_homspace(&rule, &lhs, &rhs, &[1], &[0], &[1, 0], 1)
                .unwrap();
        assert_eq!(composed.codomain().len(), 1);
        assert_eq!(composed.domain().len(), 1);
        assert_eq!(composed.codomain().legs()[0].sectors(), &[u1(-3)]);
        assert!(composed.codomain().legs()[0].is_dual());
        assert_eq!(composed.domain().legs()[0].sectors(), &[u1(-2)]);
        assert!(composed.domain().legs()[0].is_dual());
    }

    #[test]
    fn fusion_tree_homspace_compose_rejects_unmatched_contracted_sector() {
        // Pairing a domain leg with the *dual* codomain leg is a
        // SpaceMismatch in TensorKit (`(X ← V) * (V' ← Y)` fails).
        let rule = U1FusionRule;
        let lhs = FusionTreeHomSpace::new(
            FusionProductSpace::new([SectorLeg::new([(u1(0), 1)], false)]),
            FusionProductSpace::new([SectorLeg::new([(u1(1), 1)], false)]),
        );
        let rhs = FusionTreeHomSpace::new(
            FusionProductSpace::new([SectorLeg::new([(rule.dual(u1(1)), 1)], false)]),
            FusionProductSpace::new([SectorLeg::new([(u1(0), 1)], false)]),
        );

        let err = FusionTreeHomSpace::compose(&rule, &lhs, &rhs).unwrap_err();

        assert_eq!(
            err,
            CoreError::SectorMismatch {
                expected: u1(1),
                actual: rule.dual(u1(1)),
            }
        );
    }

    #[test]
    fn unique_homspace_rejects_invalid_external_sector_tuple() {
        let rule = Z4PointedRule;
        let hom = FusionTreeHomSpace::new(
            FusionProductSpace::new([SectorLeg::new([(SectorId::new(1), 1)], false)]),
            FusionProductSpace::new([SectorLeg::new([(SectorId::new(1), 1)], false)]),
        );

        let err = hom
            .unique_fusion_tree_key_from_external_sectors(
                &rule,
                &[SectorId::new(1), SectorId::new(2)],
            )
            .unwrap_err();

        assert_eq!(
            err,
            CoreError::InvalidSector {
                sector: SectorId::new(2),
            }
        );
    }

    #[test]
    fn fusion_tree_homspace_generates_innerline_paths_for_simple_fusion() {
        let rule = BranchingMultiplicityFreeRule;
        let hom = FusionTreeHomSpace::from_sector_ids([(1, 1), (1, 1), (1, 1)], [(1, 1)]);

        let keys = hom.fusion_tree_keys(&rule);

        assert_eq!(keys.len(), 2);
        assert_eq!(keys[0].coupled(), Some(SectorId::new(1)));
        assert_eq!(keys[1].coupled(), Some(SectorId::new(1)));
        assert_eq!(keys[0].codomain_innerlines(), &[SectorId::new(0)]);
        assert_eq!(keys[1].codomain_innerlines(), &[SectorId::new(2)]);
        assert_eq!(
            keys[0].codomain_vertices(),
            &[SectorId::new(1), SectorId::new(1)]
        );
        assert!(keys[0].domain_innerlines().is_empty());
        assert!(keys[0].domain_vertices().is_empty());
        assert_eq!(keys[0].domain_uncoupled(), &[SectorId::new(1)]);

        let groups = hom.fusion_tree_groups(&rule).unwrap();
        assert_eq!(groups.len(), 1);
        assert_eq!(groups[0].block_indices(), &[0, 1]);
    }

    #[test]
    fn fusion_tree_homspace_matches_tensorkit_z2_fusiontreelist_order() {
        let rule = Z2FusionRule;
        let leg = || SectorLeg::new([(SectorId::new(0), 1), (SectorId::new(1), 1)], false);
        let hom = FusionTreeHomSpace::new(
            FusionProductSpace::new([leg(), leg()]),
            FusionProductSpace::new([leg(), leg()]),
        );

        let keys = hom.fusion_tree_keys(&rule);

        // TensorKit.jl 6Camk:
        // V=Vect[Z2Irrep](0=>1,1=>1); W=(V⊗V)←(V⊗V);
        // [(f1.uncoupled, f2.uncoupled, f1.coupled) for (f1,f2) in fusiontrees(W)]
        assert_eq!(
            fusion_tree_pair_order(&keys),
            vec![
                (vec![0, 0], vec![0, 0], 0),
                (vec![1, 1], vec![0, 0], 0),
                (vec![0, 0], vec![1, 1], 0),
                (vec![1, 1], vec![1, 1], 0),
                (vec![1, 0], vec![1, 0], 1),
                (vec![0, 1], vec![1, 0], 1),
                (vec![1, 0], vec![0, 1], 1),
                (vec![0, 1], vec![0, 1], 1),
            ]
        );

        let groups = hom.fusion_tree_groups(&rule).unwrap();
        assert_eq!(groups.len(), keys.len());
        for (index, group) in groups.iter().enumerate() {
            assert_eq!(group.block_indices(), &[index]);
        }
    }

    #[test]
    fn fusion_tree_key_cache_hits_across_degeneracy_and_keeps_dual_signature() {
        let rule = SU2FusionRule;
        let mk_leg = |degeneracy| {
            SectorLeg::new(
                [
                    (SU2Irrep::from_twice_spin(0).sector_id(), degeneracy),
                    (SU2Irrep::from_twice_spin(1).sector_id(), degeneracy + 1),
                ],
                false,
            )
        };
        let hom_small = FusionTreeHomSpace::new(
            FusionProductSpace::new([mk_leg(1), mk_leg(1)]),
            FusionProductSpace::new([mk_leg(1)]),
        );
        let hom_large = FusionTreeHomSpace::new(
            FusionProductSpace::new([mk_leg(4), mk_leg(4)]),
            FusionProductSpace::new([mk_leg(4)]),
        );

        let small_layout = hom_small.cached_fusion_tree_layout(&rule);
        let large_layout = hom_large.cached_fusion_tree_layout(&rule);
        assert!(Arc::ptr_eq(&small_layout, &large_layout));
        assert_eq!(small_layout.keys.as_ref(), large_layout.keys.as_ref());

        let dual_hom = FusionTreeHomSpace::new(
            FusionProductSpace::new([mk_leg(1).dual(&rule), mk_leg(1)]),
            FusionProductSpace::new([mk_leg(1)]),
        );
        let dual_layout = dual_hom.cached_fusion_tree_layout(&rule);
        assert!(!Arc::ptr_eq(&small_layout, &dual_layout));
        assert_ne!(small_layout.keys.as_ref(), dual_layout.keys.as_ref());
    }

    #[test]
    fn fusion_tree_homspace_matches_tensorkit_su2_simple_order() {
        let rule = SU2FusionRule;
        let leg = || {
            SectorLeg::new(
                [
                    (SectorId::new(0), 1),
                    (SectorId::new(1), 1),
                    (SectorId::new(2), 1),
                ],
                false,
            )
        };
        let hom = FusionTreeHomSpace::new(
            FusionProductSpace::new([leg(), leg()]),
            FusionProductSpace::new([leg()]),
        );

        let keys = hom.fusion_tree_keys(&rule);

        // TensorKit.jl 6Camk with sector id = twice spin:
        // V=Vect[SU2Irrep](0=>1,1//2=>1,1=>1); W=(V⊗V)←V;
        // [(2f1.uncoupled, 2f2.uncoupled, 2f1.coupled) for (f1,f2) in fusiontrees(W)]
        assert_eq!(
            fusion_tree_pair_order(&keys),
            vec![
                (vec![0, 0], vec![0], 0),
                (vec![1, 1], vec![0], 0),
                (vec![2, 2], vec![0], 0),
                (vec![1, 0], vec![1], 1),
                (vec![0, 1], vec![1], 1),
                (vec![2, 1], vec![1], 1),
                (vec![1, 2], vec![1], 1),
                (vec![2, 0], vec![2], 2),
                (vec![1, 1], vec![2], 2),
                (vec![0, 2], vec![2], 2),
                (vec![2, 2], vec![2], 2),
            ]
        );
        assert!(keys
            .iter()
            .all(|key| key.codomain_vertices() == [SectorId::new(1)]));
        assert!(keys.iter().all(|key| key.domain_vertices().is_empty()));
    }

    #[test]
    fn braid_tree_pair_block_matches_per_source() {
        use std::collections::BTreeMap;
        let rule = SU2FusionRule;
        let leg = || {
            SectorLeg::new(
                [
                    (SectorId::new(0), 1),
                    (SectorId::new(1), 1),
                    (SectorId::new(2), 1),
                ],
                false,
            )
        };
        // (V⊗V⊗V) ← V spans many uncoupled blocks; test each block.
        let hom = FusionTreeHomSpace::new(
            FusionProductSpace::new([leg(), leg(), leg()]),
            FusionProductSpace::new([leg()]),
        );
        let keys = hom.fusion_tree_keys(&rule);

        // Group source tree-pairs by their uncoupled block (the batching unit).
        let mut blocks: BTreeMap<Vec<usize>, Vec<FusionTreeBlockKey>> = BTreeMap::new();
        for key in &keys {
            let tag: Vec<usize> = key
                .codomain_tree()
                .uncoupled()
                .iter()
                .chain(key.domain_tree().uncoupled())
                .map(|s| s.id())
                .collect();
            blocks.entry(tag).or_default().push(key.clone());
        }

        // Global leg indices: codomain legs 0,1,2 and domain leg 3. Reverse the
        // codomain, keep the domain leg in place.
        let codomain_permutation = [2usize, 1, 0];
        let domain_permutation = [3usize];
        let mut checked_blocks = 0;
        for src_keys in blocks.values() {
            let batched = multiplicity_free_permute_tree_pair_block(
                &rule,
                src_keys,
                &codomain_permutation,
                &domain_permutation,
            )
            .unwrap();
            assert_eq!(batched.len(), src_keys.len());
            for (src, batched_rows) in src_keys.iter().zip(&batched) {
                let per_source = multiplicity_free_permute_tree_pair(
                    &rule,
                    src,
                    &codomain_permutation,
                    &domain_permutation,
                )
                .unwrap();
                // Compare as key -> coefficient maps within double-precision tol.
                let mut want: BTreeMap<FusionTreeBlockKey, f64> = BTreeMap::new();
                for (k, c) in &per_source {
                    *want.entry(k.clone()).or_insert(0.0) += c;
                }
                let mut got: BTreeMap<FusionTreeBlockKey, f64> = BTreeMap::new();
                for (k, c) in batched_rows {
                    *got.entry(k.clone()).or_insert(0.0) += c;
                }
                assert_eq!(
                    want.keys().collect::<Vec<_>>(),
                    got.keys().collect::<Vec<_>>(),
                    "destination trees differ for a source in block"
                );
                for (k, wc) in &want {
                    let gc = got[k];
                    assert!(
                        (wc - gc).abs() <= 1e-12 * (1.0 + wc.abs()),
                        "coefficient mismatch {wc} vs {gc}"
                    );
                }
            }
            checked_blocks += 1;
        }
        assert!(checked_blocks > 0, "expected at least one block");
    }

    #[test]
    fn fusion_tree_homspace_matches_tensorkit_su2_innerline_order() {
        let rule = SU2FusionRule;
        let hom = FusionTreeHomSpace::from_sector_ids([(1, 1), (1, 1), (1, 1)], [(1, 1)]);

        let keys = hom.fusion_tree_keys(&rule);

        // TensorKit.jl 6Camk with sector id = twice spin:
        // V=Vect[SU2Irrep](1//2=>1); W=(V⊗V⊗V)←V;
        // codomain innerlines for fusiontrees(W) are [0], then [2].
        assert_eq!(keys.len(), 2);
        assert_eq!(keys[0].codomain_innerlines(), &[SectorId::new(0)]);
        assert_eq!(keys[1].codomain_innerlines(), &[SectorId::new(2)]);
        assert_eq!(
            fusion_tree_pair_order(&keys),
            vec![(vec![1, 1, 1], vec![1], 1), (vec![1, 1, 1], vec![1], 1),]
        );

        let groups = hom.fusion_tree_groups(&rule).unwrap();
        assert_eq!(groups.len(), 1);
        assert_eq!(groups[0].block_indices(), &[0, 1]);
    }

    #[test]
    fn fusion_tree_homspace_external_sectors_preserve_su2_simple_innerline_order() {
        let rule = SU2FusionRule;
        let half = SectorId::new(1);
        let hom = FusionTreeHomSpace::from_sector_ids([(1, 1), (1, 1), (1, 1)], [(1, 1)]);

        let keys = hom
            .fusion_tree_keys_from_external_sectors(&rule, &[half, half, half, half])
            .unwrap();

        assert_eq!(keys.len(), 2);
        assert_eq!(keys[0].codomain_uncoupled(), &[half, half, half]);
        assert_eq!(keys[0].domain_uncoupled(), &[half]);
        assert_eq!(keys[0].codomain_innerlines(), &[SectorId::new(0)]);
        assert_eq!(keys[1].codomain_innerlines(), &[SectorId::new(2)]);
        assert_eq!(
            fusion_tree_pair_order(&keys),
            vec![(vec![1, 1, 1], vec![1], 1), (vec![1, 1, 1], vec![1], 1),]
        );
    }

    #[test]
    fn tensormap_subblocks_by_sectors_returns_all_su2_simple_innerline_blocks() {
        let rule = SU2FusionRule;
        let half = SectorId::new(1);
        let dense = TensorMapSpace::<3, 1>::from_dims([1, 1, 1], [1]).unwrap();
        let hom = FusionTreeHomSpace::from_sector_ids([(1, 1), (1, 1), (1, 1)], [(1, 1)]);
        let fusion_space = FusionTensorMapSpace::from_degeneracy_shapes(
            dense,
            hom,
            &rule,
            [vec![1, 1, 1, 1], vec![1, 1, 1, 1]],
        )
        .unwrap();
        let tensor =
            TensorMap::<i32, 3, 1>::from_vec_with_fusion_space(vec![11, 22], fusion_space).unwrap();

        let blocks = tensor
            .subblocks_by_sectors(&rule, &[half, half, half, half])
            .unwrap();

        assert_eq!(blocks.len(), 2);
        assert_eq!(blocks[0].offset(), 0);
        assert_eq!(blocks[0].data()[blocks[0].offset()], 11);
        assert_eq!(blocks[1].offset(), 1);
        assert_eq!(blocks[1].data()[blocks[1].offset()], 22);

        let err = tensor
            .subblock_by_sectors(&rule, &[half, half, half, half])
            .unwrap_err();
        assert_eq!(
            err,
            CoreError::BlockCountMismatch {
                expected: 1,
                actual: 2,
            }
        );
    }

    #[test]
    fn fusion_tree_homspace_uses_tensorkit_parent_iterator_order_not_ord_sort() {
        let rule = UnsortedFusionIteratorOrderRule;
        let hom = FusionTreeHomSpace::from_sector_ids([(1, 1), (1, 1), (1, 1)], [(1, 1)]);

        let keys = hom.fusion_tree_keys(&rule);

        // TensorKit rank >= 3 iterator picks the parent line from
        // `coupled ⊗ dual(last)` order. This toy rule returns 1 ⊗ 1 as [2, 0],
        // deliberately opposite to `SectorId` Ord, so an Ord-based replay would
        // produce [0], [2].
        assert_eq!(keys.len(), 2);
        assert_eq!(keys[0].codomain_innerlines(), &[SectorId::new(2)]);
        assert_eq!(keys[1].codomain_innerlines(), &[SectorId::new(0)]);
    }

    #[test]
    fn fusion_tree_homspace_uses_visible_dual_space_sector_label_like_tensorkit() {
        let rule = U1FusionRule;
        let minus_one = U1Irrep::new(-1);
        let hom = FusionTreeHomSpace::new(
            FusionProductSpace::new([SectorLeg::new([(minus_one, 1)], true)]),
            FusionProductSpace::new([SectorLeg::new([(minus_one, 1)], false)]),
        );

        let keys = hom.fusion_tree_keys(&rule);

        // TensorKit:
        // collect(sectors(Vect[U1Irrep](1=>1)')) == [U1Irrep(-1)]
        // fusiontrees((U1Irrep(-1),), U1Irrep(-1), (true,)) keeps uncoupled = -1.
        assert_eq!(keys.len(), 1);
        assert_eq!(keys[0].coupled(), Some(minus_one.into()));
        assert_eq!(keys[0].codomain_uncoupled(), &[minus_one.into()]);
        assert_eq!(keys[0].codomain_is_dual(), &[true]);
        assert_eq!(keys[0].domain_uncoupled(), &[minus_one.into()]);
        assert_eq!(keys[0].domain_is_dual(), &[false]);
    }

    #[test]
    fn fusion_tree_homspace_does_not_dualize_selected_dual_leg_again() {
        let rule = BranchingMultiplicityFreeRule;
        let hom = FusionTreeHomSpace::new(
            FusionProductSpace::new([SectorLeg::new([(SectorId::new(1), 1)], true)]),
            FusionProductSpace::from_sector_ids([(1, 1)]),
        );

        let keys = hom.fusion_tree_keys(&rule);

        assert_eq!(keys.len(), 1);
        assert_eq!(keys[0].coupled(), Some(SectorId::new(1)));
        assert_eq!(keys[0].codomain_uncoupled(), &[SectorId::new(1)]);
        assert_eq!(keys[0].codomain_is_dual(), &[true]);
        assert_eq!(keys[0].domain_uncoupled(), &[SectorId::new(1)]);
        assert_eq!(keys[0].domain_is_dual(), &[false]);
    }

    #[test]
    fn fusion_tree_homspace_fusionblocks_follow_domain_outer_codomain_inner_order() {
        let rule = BranchingMultiplicityFreeRule;
        let hom = FusionTreeHomSpace::new(
            FusionProductSpace::new([
                SectorLeg::new([(SectorId::new(1), 1), (SectorId::new(2), 1)], false),
                SectorLeg::new([(SectorId::new(1), 1)], false),
            ]),
            FusionProductSpace::new([SectorLeg::new(
                [(SectorId::new(1), 1), (SectorId::new(2), 1)],
                false,
            )]),
        );

        let groups = hom.fusion_tree_groups(&rule).unwrap();

        assert_eq!(groups.len(), 2);
        assert_eq!(
            groups[0].group_key(),
            &FusionTreeGroupKey::from_sector_ids([2, 1], [1], [false, false], [false])
        );
        assert_eq!(
            groups[1].group_key(),
            &FusionTreeGroupKey::from_sector_ids([1, 1], [2], [false, false], [false])
        );
    }

    #[test]
    fn fusion_tree_groups_preserve_structure_order_and_ignore_internal_tree_data() {
        let first = BlockKey::from(FusionTreeBlockKey::pair_from_sector_ids(
            [10, 20],
            [30],
            Some(5),
            [false, true],
            [true],
            [101],
            [201],
            [301, 302],
            [401],
        ));
        let second = BlockKey::from(FusionTreeBlockKey::pair_from_sector_ids(
            [1],
            [2, 3],
            Some(4),
            [true],
            [false, true],
            [],
            [202],
            [303],
            [402, 403],
        ));
        let same_group_as_first = BlockKey::from(FusionTreeBlockKey::pair_from_sector_ids(
            [10, 20],
            [30],
            Some(6),
            [false, true],
            [true],
            [102],
            [203],
            [304, 305],
            [404],
        ));

        let keys = vec![first.clone(), second.clone(), same_group_as_first.clone()];
        let sector = SectorStructure::from_keys(2, keys.clone()).unwrap();
        let block_structure =
            packed_fixture_structure(2, keys.into_iter().map(|key| (key, vec![1, 1]))).unwrap();

        let sector_groups = sector.fusion_tree_groups();
        let block_groups = block_structure.fusion_tree_groups();
        assert_eq!(sector_groups, block_groups);
        assert_eq!(sector_groups.len(), 2);
        assert_eq!(sector_groups[0].block_indices(), &[0, 2]);
        assert_eq!(sector_groups[1].block_indices(), &[1]);
        assert_eq!(
            sector_groups[0].group_key(),
            &FusionTreeGroupKey::from_sector_ids([10, 20], [30], [false, true], [true])
        );
        assert_eq!(
            sector_groups[1].group_key(),
            &FusionTreeGroupKey::from_sector_ids([1], [2, 3], [true], [false, true])
        );
    }

    #[test]
    fn fusion_tree_groups_ignore_dense_blocks() {
        let key = BlockKey::from(FusionTreeBlockKey::pair_from_sector_ids(
            [7],
            [8],
            Some(9),
            [false],
            [true],
            [],
            [],
            [],
            [],
        ));
        let sector = SectorStructure::from_keys(2, [BlockKey::trivial(), key]).unwrap();
        let groups = sector.fusion_tree_groups();

        assert_eq!(groups.len(), 1);
        assert_eq!(groups[0].block_indices(), &[1]);
        assert_eq!(
            groups[0].group_key(),
            &FusionTreeGroupKey::from_sector_ids([7], [8], [false], [true])
        );

        let dense = BlockStructure::trivial(&[2, 3]).unwrap();
        assert!(dense.fusion_tree_groups().is_empty());
    }

    #[test]
    fn block_structure_separates_sector_and_degeneracy_data() {
        let sector = SectorStructure::from_keys(
            2,
            [BlockKey::sector_ids([0, 1]), BlockKey::sector_ids([1, 0])],
        )
        .unwrap();
        let degeneracy =
            DegeneracyStructure::packed_column_major(2, [vec![2, 3], vec![3, 2]]).unwrap();
        let structure = BlockStructure::from_parts(sector, degeneracy).unwrap();

        assert_eq!(structure.rank(), 2);
        assert_eq!(
            structure.sector_structure().key(0).unwrap(),
            &BlockKey::sector_ids([0, 1])
        );
        assert_eq!(
            structure.sector_structure().key(1).unwrap(),
            &BlockKey::sector_ids([1, 0])
        );
        assert_eq!(
            structure.degeneracy_structure().block(0).unwrap().shape(),
            &[2, 3]
        );
        assert_eq!(
            structure.degeneracy_structure().block(1).unwrap().offset(),
            6
        );
        assert_eq!(structure.required_len().unwrap(), 12);
    }

    #[test]
    fn sector_structure_pairs_compact_keys_without_map_lookup() {
        let dst = SectorStructure::from_keys(
            2,
            [
                BlockKey::sector_ids([2]),
                BlockKey::sector_ids([0]),
                BlockKey::sector_ids([1]),
            ],
        )
        .unwrap();
        let src = SectorStructure::from_keys(
            2,
            [
                BlockKey::sector_ids([0]),
                BlockKey::sector_ids([1]),
                BlockKey::sector_ids([2]),
            ],
        )
        .unwrap();

        assert!(src.has_compact_lookup());
        assert_eq!(dst.find_index(&BlockKey::sector_ids([0])), Some(1));
        assert_eq!(src.find_index(&BlockKey::sector_ids([2])), Some(2));
        assert_eq!(dst.pair_indices_from(&src).unwrap(), vec![2, 0, 1]);
    }

    #[test]
    fn sector_structure_pairs_general_fusion_keys_by_sorted_merge() {
        let key_a = BlockKey::sectors([SectorId::new(0), SectorId::new(1)]);
        let key_b = BlockKey::sectors([SectorId::new(1), SectorId::new(0)]);
        let dst = SectorStructure::from_keys(2, [key_b.clone(), key_a.clone()]).unwrap();
        let src = SectorStructure::from_keys(2, [key_a.clone(), key_b.clone()]).unwrap();

        assert!(!src.has_compact_lookup());
        assert_eq!(dst.find_index(&key_a), Some(1));
        assert_eq!(src.find_index(&key_b), Some(1));
        assert_eq!(dst.pair_indices_from(&src).unwrap(), vec![1, 0]);
    }

    #[test]
    fn tensormap_rejects_structure_rank_that_does_not_match_space_rank() {
        let space = TensorMapSpace::<2, 0>::from_dims([2, 3], []).unwrap();
        let structure = BlockStructure::packed_column_major(1, [vec![6]]).unwrap();
        let err = TensorMap::<f64, 2, 0>::from_vec_with_structure(vec![0.0; 6], space, structure)
            .unwrap_err();

        assert_eq!(
            err,
            CoreError::StructureRankMismatch {
                expected: 2,
                actual: 1
            }
        );
    }

    #[test]
    fn tensormap_rejects_incorrect_data_length() {
        let space = TensorMapSpace::<1, 1>::from_dims([2], [3]).unwrap();
        let err = TensorMap::<f64, 1, 1>::from_vec(vec![0.0; 5], space).unwrap_err();
        assert_eq!(
            err,
            CoreError::DimensionMismatch {
                expected: 6,
                actual: 5
            }
        );
    }

    #[test]
    fn vec_storage_allocates_similar_host_scratch() {
        let storage = vec![1.0_f64, 2.0];
        let scratch = storage.similar_filled(4, 0.5);

        assert_eq!(scratch, vec![0.5; 4]);
        assert_eq!(scratch.placement(), Placement::Host);
    }

    #[test]
    fn tensormap_allocates_similar_storage_from_backing_storage() {
        let space = TensorMapSpace::<1, 0>::from_dims([2], []).unwrap();
        let tensor = TensorMap::<f64, 1, 0>::from_vec(vec![1.0, 2.0], space).unwrap();
        let scratch = tensor.similar_storage_filled(3, 0.0);

        assert_eq!(scratch, vec![0.0; 3]);
        assert_eq!(scratch.placement(), tensor.placement());
    }

    #[test]
    fn split_fusion_tree_matches_tensorkit_front_tail_convention() {
        let rule = SU2FusionRule;
        let half = SU2Irrep::from_twice_spin(1).sector_id();
        let one = SU2Irrep::from_twice_spin(2).sector_id();
        let tree = FusionTreeKey::new(
            [half, half, one],
            Some(one),
            [false, false, true],
            [SectorId::new(0)],
            [SectorId::new(1), SectorId::new(1)],
        );

        let (front, tail) = split_fusion_tree(&rule, &tree, 2).unwrap();

        assert_eq!(front.uncoupled(), &[half, half]);
        assert_eq!(front.coupled(), Some(SectorId::new(0)));
        assert_eq!(front.is_dual(), &[false, false]);
        assert_eq!(front.innerlines(), &[]);
        assert_eq!(front.vertices(), &[SectorId::new(1)]);
        assert_eq!(tail.uncoupled(), &[SectorId::new(0), one]);
        assert_eq!(tail.coupled(), Some(one));
        assert_eq!(tail.is_dual(), &[false, true]);
        assert_eq!(tail.innerlines(), &[]);
        assert_eq!(tail.vertices(), &[SectorId::new(1)]);
    }

    #[test]
    fn rigid_symbols_separate_twist_from_frobenius_schur_phase() {
        let fermion = FermionParityFusionRule;
        let odd = SectorId::new(1);
        assert_eq!(fermion.dim_scalar(odd), 1.0);
        assert_eq!(fermion.twist_scalar(odd), -1.0);
        assert_eq!(fermion.frobenius_schur_phase_scalar(odd), 1.0);

        let su2 = SU2FusionRule;
        let half = SU2Irrep::from_twice_spin(1).sector_id();
        assert_eq!(su2.dim_scalar(half), 2.0);
        assert_eq!(su2.twist_scalar(half), 1.0);
        assert_eq!(su2.frobenius_schur_phase_scalar(half), -1.0);
    }
}
