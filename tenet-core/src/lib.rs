#![forbid(unsafe_code)]

//! Core TensorMap-facing data structures for TeNeT.
//!
//! This crate owns TeNeT's public/core tensor view vocabulary. Lower-level
//! crates may lower these views to concrete strided kernels, but external
//! strided/backend types should not be required by TensorMap users.

use core::fmt;
use core::marker::PhantomData;
use std::collections::HashMap;
use std::sync::Arc;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Trivial;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Placement {
    Host,
}

pub trait TensorStorage<T> {
    fn len(&self) -> usize;
    fn placement(&self) -> Placement;

    #[inline]
    fn is_empty(&self) -> bool {
        self.len() == 0
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
    dims: Vec<usize>,
    dense_dim: usize,
}

impl<const NOUT: usize, const NIN: usize> TensorMapSpace<NOUT, NIN> {
    pub fn new(codomain: ProductSpace<NOUT>, domain: ProductSpace<NIN>) -> Result<Self, CoreError> {
        let dense_dim = codomain
            .dim()
            .checked_mul(domain.dim())
            .ok_or(CoreError::ElementCountOverflow)?;
        let mut dims = Vec::with_capacity(NOUT + NIN);
        dims.extend_from_slice(codomain.dims());
        dims.extend_from_slice(domain.dims());
        Ok(Self {
            codomain,
            domain,
            dims,
            dense_dim,
        })
    }

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

#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub struct SectorLeg {
    sectors: Vec<SectorId>,
    is_dual: bool,
}

impl SectorLeg {
    pub fn new<Sectors>(sectors: Sectors, is_dual: bool) -> Self
    where
        Sectors: IntoIterator<Item = SectorId>,
    {
        let mut sectors = sectors.into_iter().collect::<Vec<_>>();
        sectors.sort_unstable();
        sectors.dedup();
        Self { sectors, is_dual }
    }

    pub fn from_sector_id(sector: usize) -> Self {
        Self::new([SectorId::new(sector)], false)
    }

    #[inline]
    pub fn sectors(&self) -> &[SectorId] {
        &self.sectors
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
    pub fn new<Legs>(legs: Legs) -> Self
    where
        Legs: IntoIterator<Item = SectorLeg>,
    {
        Self {
            legs: legs.into_iter().collect(),
        }
    }

    pub fn from_sector_ids<Sectors>(sectors: Sectors) -> Self
    where
        Sectors: IntoIterator<Item = usize>,
    {
        Self::new(sectors.into_iter().map(SectorLeg::from_sector_id))
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

impl FusionTreeHomSpace {
    pub fn new(codomain: FusionProductSpace, domain: FusionProductSpace) -> Self {
        Self { codomain, domain }
    }

    pub fn from_sector_ids<Codomain, Domain>(codomain: Codomain, domain: Domain) -> Self
    where
        Codomain: IntoIterator<Item = usize>,
        Domain: IntoIterator<Item = usize>,
    {
        Self::new(
            FusionProductSpace::from_sector_ids(codomain),
            FusionProductSpace::from_sector_ids(domain),
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

    pub fn fusion_tree_keys<R>(&self, rule: &R) -> Vec<FusionTreeBlockKey>
    where
        R: MultiplicityFreeFusionRule,
    {
        let codomain = fusion_trees_by_coupled_for_space(rule, &self.codomain);
        let domain = fusion_trees_by_coupled_for_space(rule, &self.domain);
        let coupled = common_coupled_sectors(&codomain, &domain);
        let mut keys = Vec::new();
        for sector in coupled {
            let domain_trees = trees_for_coupled(&domain, sector);
            let codomain_trees = trees_for_coupled(&codomain, sector);
            for domain_tree in domain_trees {
                for codomain_tree in codomain_trees {
                    keys.push(FusionTreeBlockKey::pair(
                        codomain_tree.clone(),
                        domain_tree.clone(),
                    ));
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

    pub fn fusion_tree_groups<R>(&self, rule: &R) -> Result<Vec<FusionTreeBlockGroup>, CoreError>
    where
        R: MultiplicityFreeFusionRule,
    {
        self.sector_structure(rule)
            .map(|structure| structure.fusion_tree_groups())
    }
}

pub trait MultiplicityFreeFusionRule {
    fn vacuum(&self) -> SectorId;

    fn dual(&self, sector: SectorId) -> SectorId {
        sector
    }

    fn fuse(&self, left: SectorId, right: SectorId) -> Vec<SectorId>;
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
    let mut grouped = Vec::<CoupledFusionTrees>::new();
    for tuple in space.selected_leg_tuples() {
        for coupled in possible_coupled_sectors(rule, &effective_sectors(rule, &tuple)) {
            let trees = fusion_trees_for_coupled(rule, &tuple, coupled);
            if let Some(group) = grouped.iter_mut().find(|group| group.coupled == coupled) {
                group.trees.extend(trees);
            } else {
                grouped.push(CoupledFusionTrees { coupled, trees });
            }
        }
    }
    grouped.sort_by_key(|group| group.coupled);
    grouped
}

fn fusion_trees_for_coupled<R>(
    rule: &R,
    legs: &[FusionTreeLeg],
    coupled: SectorId,
) -> Vec<FusionTreeKey>
where
    R: MultiplicityFreeFusionRule,
{
    let effective = effective_sectors(rule, legs);
    let uncoupled = legs.iter().map(|leg| leg.sector()).collect::<Vec<_>>();
    let is_dual = legs.iter().map(|leg| leg.is_dual()).collect::<Vec<_>>();
    collect_fusion_trees_for_coupled(rule, &uncoupled, &is_dual, &effective, coupled)
}

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
    match effective.len() {
        0 if coupled == rule.vacuum() => vec![FusionTreeKey::new(
            uncoupled.iter().copied(),
            Some(coupled),
            is_dual.iter().copied(),
            Vec::<SectorId>::new(),
            Vec::<SectorId>::new(),
        )],
        0 => Vec::new(),
        1 if effective[0] == coupled => vec![FusionTreeKey::new(
            uncoupled.iter().copied(),
            Some(coupled),
            is_dual.iter().copied(),
            Vec::<SectorId>::new(),
            Vec::<SectorId>::new(),
        )],
        1 => Vec::new(),
        2 => {
            if rule.fuse(effective[0], effective[1]).contains(&coupled) {
                vec![FusionTreeKey::new(
                    uncoupled.iter().copied(),
                    Some(coupled),
                    is_dual.iter().copied(),
                    Vec::<SectorId>::new(),
                    [SectorId::new(1)],
                )]
            } else {
                Vec::new()
            }
        }
        _ => collect_nontrivial_fusion_trees_for_coupled(
            rule, uncoupled, is_dual, effective, coupled,
        ),
    }
}

fn collect_nontrivial_fusion_trees_for_coupled<R>(
    rule: &R,
    uncoupled: &[SectorId],
    is_dual: &[bool],
    effective: &[SectorId],
    coupled: SectorId,
) -> Vec<FusionTreeKey>
where
    R: MultiplicityFreeFusionRule,
{
    let last = effective[effective.len() - 1];
    let front_uncoupled = &uncoupled[..uncoupled.len() - 1];
    let front_is_dual = &is_dual[..is_dual.len() - 1];
    let front_effective = &effective[..effective.len() - 1];
    let mut trees = Vec::new();
    for front_coupled in possible_coupled_sectors(rule, front_effective) {
        if !rule.fuse(front_coupled, last).contains(&coupled) {
            continue;
        }
        for front_tree in collect_fusion_trees_for_coupled(
            rule,
            front_uncoupled,
            front_is_dual,
            front_effective,
            front_coupled,
        ) {
            let mut innerlines = front_tree.innerlines().to_vec();
            innerlines.push(front_coupled);
            let mut vertices = front_tree.vertices().to_vec();
            vertices.push(SectorId::new(1));
            trees.push(FusionTreeKey::new(
                uncoupled.iter().copied(),
                Some(coupled),
                is_dual.iter().copied(),
                innerlines,
                vertices,
            ));
        }
    }
    trees
}

fn possible_coupled_sectors<R>(rule: &R, effective: &[SectorId]) -> Vec<SectorId>
where
    R: MultiplicityFreeFusionRule,
{
    let mut sectors = match effective.len() {
        0 => vec![rule.vacuum()],
        1 => vec![effective[0]],
        _ => {
            let last = effective[effective.len() - 1];
            possible_coupled_sectors(rule, &effective[..effective.len() - 1])
                .into_iter()
                .flat_map(|front| rule.fuse(front, last))
                .collect()
        }
    };
    sectors.sort_unstable();
    sectors.dedup();
    sectors
}

fn effective_sectors<R>(rule: &R, legs: &[FusionTreeLeg]) -> Vec<SectorId>
where
    R: MultiplicityFreeFusionRule,
{
    legs.iter()
        .map(|leg| {
            if leg.is_dual() {
                rule.dual(leg.sector())
            } else {
                leg.sector()
            }
        })
        .collect()
}

fn common_coupled_sectors(
    left: &[CoupledFusionTrees],
    right: &[CoupledFusionTrees],
) -> Vec<SectorId> {
    let mut sectors = left
        .iter()
        .filter(|left_group| {
            right
                .iter()
                .any(|right_group| right_group.coupled == left_group.coupled)
        })
        .map(|group| group.coupled)
        .collect::<Vec<_>>();
    sectors.sort_unstable();
    sectors.dedup();
    sectors
}

fn trees_for_coupled(groups: &[CoupledFusionTrees], coupled: SectorId) -> &[FusionTreeKey] {
    groups
        .iter()
        .find(|group| group.coupled == coupled)
        .map(|group| group.trees.as_slice())
        .unwrap_or(&[])
}

#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct FusionTreeGroupKey {
    codomain_uncoupled: Vec<SectorId>,
    domain_uncoupled: Vec<SectorId>,
    codomain_is_dual: Vec<bool>,
    domain_is_dual: Vec<bool>,
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

#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct FusionTreeKey {
    uncoupled: Vec<SectorId>,
    coupled: Option<SectorId>,
    is_dual: Vec<bool>,
    innerlines: Vec<SectorId>,
    vertices: Vec<SectorId>,
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
    shape: Vec<usize>,
    strides: Vec<usize>,
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
            shape,
            strides,
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
    block_indices: Vec<usize>,
}

impl FusionTreeBlockGroup {
    pub fn new(group_key: FusionTreeGroupKey, block_indices: Vec<usize>) -> Self {
        Self {
            group_key,
            block_indices,
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
    sorted_indices: Vec<usize>,
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
            sorted_indices: Vec::new(),
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
        let mut sorted_indices = (0..blocks.len()).collect::<Vec<_>>();
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
        let mut group_indices = HashMap::<FusionTreeGroupKey, usize>::new();
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
    indices: Vec<usize>,
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
        let mut indices = vec![Self::MISSING; len];
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
    shape: Vec<usize>,
    strides: Vec<usize>,
    offset: usize,
}

impl DegeneracyBlock {
    pub fn new(shape: Vec<usize>, strides: Vec<usize>, offset: usize) -> Result<Self, CoreError> {
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
        Self::new(shape, strides, offset)
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

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct BlockStructure {
    sector: SectorStructure,
    degeneracy: DegeneracyStructure,
}

impl BlockStructure {
    pub fn trivial(shape: &[usize]) -> Result<Self, CoreError> {
        Self::from_parts(
            SectorStructure::dense(shape.len()),
            DegeneracyStructure::packed_column_major(shape.len(), [shape.to_vec()])?,
        )
    }

    pub fn empty(rank: usize) -> Self {
        Self {
            sector: SectorStructure::empty(rank),
            degeneracy: DegeneracyStructure {
                rank,
                blocks: Vec::new(),
            },
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
        Ok(Self { sector, degeneracy })
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

    pub fn packed_column_major_with_keys<I, K>(rank: usize, blocks: I) -> Result<Self, CoreError>
    where
        I: IntoIterator<Item = (K, Vec<usize>)>,
        K: Into<BlockKey>,
    {
        let mut keys = Vec::new();
        let mut shapes = Vec::new();
        for (key, shape) in blocks {
            if shape.len() != rank {
                return Err(CoreError::StructureRankMismatch {
                    expected: rank,
                    actual: shape.len(),
                });
            }
            keys.push(key.into());
            shapes.push(shape);
        }
        Self::from_parts(
            SectorStructure::from_keys(rank, keys)?,
            DegeneracyStructure::packed_column_major(rank, shapes)?,
        )
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

    pub fn fusion_tree_groups(&self) -> Vec<FusionTreeBlockGroup> {
        self.sector.fusion_tree_groups()
    }

    pub fn find_block_index_by_key(&self, key: &BlockKey) -> Option<usize> {
        self.sector.find_index(key)
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

    pub fn required_len(&self) -> Result<usize, CoreError> {
        self.degeneracy.required_len()
    }
}

#[derive(Clone, Debug)]
pub struct TensorMap<T, const NOUT: usize, const NIN: usize, S = Trivial, D = Vec<T>> {
    storage: D,
    space: TensorMapSpace<NOUT, NIN>,
    structure: Arc<BlockStructure>,
    _marker: PhantomData<(T, S)>,
}

pub type Tensor<T, const N: usize, S = Trivial> = TensorMap<T, N, 0, S>;

impl<T, const NOUT: usize, const NIN: usize, S> TensorMap<T, NOUT, NIN, S, Vec<T>> {
    pub fn from_vec(data: Vec<T>, space: TensorMapSpace<NOUT, NIN>) -> Result<Self, CoreError> {
        Self::from_vec_with_structure(data, space.clone(), BlockStructure::trivial(space.dims())?)
    }

    pub fn from_vec_with_structure(
        data: Vec<T>,
        space: TensorMapSpace<NOUT, NIN>,
        structure: BlockStructure,
    ) -> Result<Self, CoreError> {
        Self::from_vec_with_shared_structure(data, space, Arc::new(structure))
    }

    pub fn from_vec_with_shared_structure(
        data: Vec<T>,
        space: TensorMapSpace<NOUT, NIN>,
        structure: Arc<BlockStructure>,
    ) -> Result<Self, CoreError> {
        Self::from_storage_with_shared_structure(data, space, structure)
    }
}

impl<T: Clone, const NOUT: usize, const NIN: usize, S> TensorMap<T, NOUT, NIN, S, Vec<T>> {
    pub fn filled(value: T, space: TensorMapSpace<NOUT, NIN>) -> Result<Self, CoreError> {
        Self::from_vec(vec![value; space.dense_dim()], space)
    }
}

impl<T, const NOUT: usize, const NIN: usize, S, D> TensorMap<T, NOUT, NIN, S, D>
where
    D: TensorStorage<T>,
{
    pub fn from_storage_with_structure(
        storage: D,
        space: TensorMapSpace<NOUT, NIN>,
        structure: BlockStructure,
    ) -> Result<Self, CoreError> {
        Self::from_storage_with_shared_structure(storage, space, Arc::new(structure))
    }

    pub fn from_storage_with_shared_structure(
        storage: D,
        space: TensorMapSpace<NOUT, NIN>,
        structure: Arc<BlockStructure>,
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
    pub fn space(&self) -> &TensorMapSpace<NOUT, NIN> {
        &self.space
    }

    #[inline]
    pub fn structure(&self) -> &Arc<BlockStructure> {
        &self.structure
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
}

impl<T, const NOUT: usize, const NIN: usize, S, D> TensorMap<T, NOUT, NIN, S, D>
where
    D: HostWritableStorage<T>,
{
    #[inline]
    pub fn data_mut(&mut self) -> &mut [T] {
        self.storage.as_mut_slice()
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
    RankMismatch { shape: usize, strides: usize },
    StructureRankMismatch { expected: usize, actual: usize },
    DimensionMismatch { expected: usize, actual: usize },
    BlockCountMismatch { expected: usize, actual: usize },
    BlockIndexOutOfBounds { index: usize, count: usize },
    DuplicateBlockKey { key: BlockKey },
    MissingBlockKey { key: BlockKey },
    ElementCountOverflow,
    OffsetOverflow { value: usize },
    StrideOverflow { value: usize },
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

    #[derive(Clone, Copy, Debug)]
    struct BranchingMultiplicityFreeRule;

    impl MultiplicityFreeFusionRule for BranchingMultiplicityFreeRule {
        fn vacuum(&self) -> SectorId {
            SectorId::new(0)
        }

        fn dual(&self, sector: SectorId) -> SectorId {
            match sector.id() {
                3 => SectorId::new(1),
                other => SectorId::new(other),
            }
        }

        fn fuse(&self, left: SectorId, right: SectorId) -> Vec<SectorId> {
            match (left.id(), right.id()) {
                (0, x) | (x, 0) => vec![SectorId::new(x)],
                (1, 1) => vec![SectorId::new(0), SectorId::new(2)],
                (1, 2) | (2, 1) => vec![SectorId::new(1), SectorId::new(3)],
                (2, 2) => vec![SectorId::new(0)],
                _ => Vec::new(),
            }
        }
    }

    #[derive(Clone, Copy, Debug)]
    struct Z2MultiplicityFreeRule;

    impl MultiplicityFreeFusionRule for Z2MultiplicityFreeRule {
        fn vacuum(&self) -> SectorId {
            SectorId::new(0)
        }

        fn fuse(&self, left: SectorId, right: SectorId) -> Vec<SectorId> {
            vec![SectorId::new((left.id() + right.id()) % 2)]
        }
    }

    #[derive(Clone, Copy, Debug)]
    struct Su2MultiplicityFreeRule;

    impl MultiplicityFreeFusionRule for Su2MultiplicityFreeRule {
        fn vacuum(&self) -> SectorId {
            SectorId::new(0)
        }

        fn fuse(&self, left: SectorId, right: SectorId) -> Vec<SectorId> {
            let min = left.id().abs_diff(right.id());
            let max = left.id() + right.id();
            (min..=max).step_by(2).map(SectorId::new).collect()
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
        let hom = FusionTreeHomSpace::from_sector_ids([1, 1], [1, 1]);

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
    fn fusion_tree_homspace_generates_innerline_paths_for_simple_fusion() {
        let rule = BranchingMultiplicityFreeRule;
        let hom = FusionTreeHomSpace::from_sector_ids([1, 1, 1], [1]);

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
        let rule = Z2MultiplicityFreeRule;
        let leg = || SectorLeg::new([SectorId::new(0), SectorId::new(1)], false);
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
    fn fusion_tree_homspace_matches_tensorkit_su2_simple_order() {
        let rule = Su2MultiplicityFreeRule;
        let leg = || {
            SectorLeg::new(
                [SectorId::new(0), SectorId::new(1), SectorId::new(2)],
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
    fn fusion_tree_homspace_matches_tensorkit_su2_innerline_order() {
        let rule = Su2MultiplicityFreeRule;
        let hom = FusionTreeHomSpace::from_sector_ids([1, 1, 1], [1]);

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
    fn fusion_tree_homspace_uses_dualized_sector_for_matching_but_stores_original_leg() {
        let rule = BranchingMultiplicityFreeRule;
        let hom = FusionTreeHomSpace::new(
            FusionProductSpace::new([SectorLeg::new([SectorId::new(3)], true)]),
            FusionProductSpace::from_sector_ids([1]),
        );

        let keys = hom.fusion_tree_keys(&rule);

        assert_eq!(keys.len(), 1);
        assert_eq!(keys[0].coupled(), Some(SectorId::new(1)));
        assert_eq!(keys[0].codomain_uncoupled(), &[SectorId::new(3)]);
        assert_eq!(keys[0].codomain_is_dual(), &[true]);
        assert_eq!(keys[0].domain_uncoupled(), &[SectorId::new(1)]);
        assert_eq!(keys[0].domain_is_dual(), &[false]);
    }

    #[test]
    fn fusion_tree_homspace_fusionblocks_follow_domain_outer_codomain_inner_order() {
        let rule = BranchingMultiplicityFreeRule;
        let hom = FusionTreeHomSpace::new(
            FusionProductSpace::new([
                SectorLeg::new([SectorId::new(1), SectorId::new(2)], false),
                SectorLeg::new([SectorId::new(1)], false),
            ]),
            FusionProductSpace::new([SectorLeg::new([SectorId::new(1), SectorId::new(2)], false)]),
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
        let block_structure = BlockStructure::packed_column_major_with_keys(
            2,
            keys.into_iter().map(|key| (key, vec![1, 1])),
        )
        .unwrap();

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
}
