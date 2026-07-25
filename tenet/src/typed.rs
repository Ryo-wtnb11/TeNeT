//! Provider-typed facade: spaces and tensor maps that keep the concrete
//! fusion-rule type `R` and speak the provider's own sector labels.
//!
//! The ergonomic [`crate::prelude`] facade erases the rule behind a fixed set
//! of built-ins. This module is its typed sibling: `R` stays concrete through
//! monomorphized construction, so any provider — including one defined
//! downstream — can drive it, and the categorical identity of a tensor comes
//! back as [`SectorCodec::Sector`] labels instead of opaque
//! [`tenet_core::SectorId`] keys. The engine itself never sees a label; the
//! codec is the single boundary where one enters or leaves.
//!
//! The exception is deliberate: [`TensorMap::block`] is the engine-level
//! layout view, and the [`tenet_core::BlockRef`] it returns carries the raw
//! [`tenet_core::BlockKey`]. Labels are what [`TensorMap::block_fusion_trees`]
//! is for.
//!
//! # Phase boundary
//!
//! This is the phase-4 surface of issue #557: construction
//! ([`TensorMap::zeros`], [`TensorMap::from_block_fn`]), inspection
//! ([`TensorMap::codomain`], [`TensorMap::domain`],
//! [`TensorMap::block_fusion_trees`], [`TensorMap::block`],
//! [`TensorMap::block_count`], [`TensorMap::data`], [`TensorMap::runtime`]),
//! and the index-manipulation and contraction
//! operations — [`TensorMap::permute`], [`TensorMap::braid`],
//! [`TensorMap::transpose`], [`TensorMap::transpose_axes`],
//! [`TensorMap::repartition`] and [`TensorMap::contract`].
//!
//! Everything else is deliberately still absent, each for its own reason:
//!
//! - The **decompositions** (`svd_*`, `qr`/`lq`, `left_orth`/`right_orth`, the
//!   null spaces) get their own readiness step: they force a decision on how a
//!   spectrum is labelled and they touch the matrix-algebra result types, so
//!   they are a phase rather than an addition.
//! - The **scalar operations** (`add`, `scale`, `norm`, `inner`, `tr`) are
//!   reachable but raise the operator-overload ergonomics question, which is
//!   reviewed on its own.
//! - **Composition** — TensorKit `A * B`, which unlike [`TensorMap::contract`]
//!   never twists dual legs — is blocked below this layer: fermionic compose
//!   needs a new public seam over `LoweredMultiplicityFreeAlgebra`, which
//!   `tenet-core` seals, and a silently
//!   bosonic-only `compose` would return wrong fermionic signs rather than an
//!   error.
//! - `adjoint` and `conj` are design-gated: only an eager `adjoint` is
//!   reachable here, which would diverge from the lazy erased sibling, and
//!   `conj` has an open correctness question for non-self-dual sectors.
//!
//! Adding any of them ahead of its review would bypass the gate that exists to
//! keep this surface deliberate.
//!
//! Construction consumes only the transactional checked admission path, so a
//! provider that reports an invalid or unrepresentable algebra fails with a
//! typed error and publishes no layout, cache, or admission state.

use std::sync::Arc;

use tenet_core::{
    BlockKey, BlockRef, CheckedFusionAlgebra, FusionProductSpace, FusionTreeHomSpace,
    MultiplicityFreeRigidSymbols, SectorLeg,
};
use tenet_tensors::{
    BoundDynamicFusionMapSpace, BoundDynamicTensorRef, OutputAxisOrder, TreeTransformOperation,
};

pub use tenet_core::SectorCodec;

/// Re-exported so `use tenet::typed::*` is self-sufficient apart from the
/// provider: every fallible method here returns this error.
pub use crate::error::Error;
/// Re-exported for the same reason as [`Error`]: every constructor here takes
/// a runtime. Both types are also in [`crate::prelude`]; re-exporting them
/// here is what lets a caller glob-import this module alone. The module
/// itself stays out of the prelude, because its [`TensorMap`] would collide
/// with the erased [`tenet_core::TensorMap`] already exported there.
pub use crate::runtime::Runtime;

use crate::tensor::{apply_fill, with_planar_axes, Fill, PlanarRequestKind, TensorScalar};
use crate::typed_tensor_core::{
    tensorcontract_owned_multiplicity_free, tree_transform_owned_multiplicity_free,
};

/// One tensor leg: a provider plus the sector-to-degeneracy map of that axis
/// (TensorKit's `GradedSpace`).
///
/// The leg owns the complete map independently of which fusion trees a tensor
/// built on it happens to populate. `is_dual` marks the conjugate space
/// (TensorKit's `V'`).
pub struct GradedSpace<R> {
    provider: Arc<R>,
    leg: SectorLeg,
}

// Why hand-written instead of derived: the derives would demand `R: Clone` and
// `R: Debug`, neither of which a provider needs to satisfy — the provider is
// shared behind an `Arc` and its labels, not the rule itself, are what a
// diagnostic wants to show.
impl<R> Clone for GradedSpace<R> {
    fn clone(&self) -> Self {
        Self {
            provider: Arc::clone(&self.provider),
            leg: self.leg.clone(),
        }
    }
}

impl<R> core::fmt::Debug for GradedSpace<R> {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        formatter
            .debug_struct("GradedSpace")
            .field("leg", &self.leg)
            .finish_non_exhaustive()
    }
}

impl<R> GradedSpace<R>
where
    R: SectorCodec + CheckedFusionAlgebra,
{
    /// Builds a leg from `(label, degeneracy)` pairs.
    ///
    /// Order is irrelevant: the leg stores its sectors in the provider's
    /// [`tenet_core::SectorId`] order. A zero-degeneracy sector is absent from
    /// the result, matching the leg invariant of the erased facade.
    ///
    /// # Errors
    ///
    /// - [`Error::InvalidArgument`] when a label is declared more than once,
    ///   naming the offending label.
    /// - [`Error::InvalidArgument`] when two distinct labels encode to one
    ///   sector id. That is the provider breaking [`SectorCodec`]'s
    ///   injectivity law, and it is reported as such rather than as a caller
    ///   duplicate.
    /// - [`Error::FusionAlgebra`] when the provider cannot represent a label,
    ///   preserving the provider's own encode error.
    pub fn try_new<Pairs>(provider: Arc<R>, pairs: Pairs, is_dual: bool) -> Result<Self, Error>
    where
        Pairs: IntoIterator<Item = (R::Sector, usize)>,
    {
        let pairs: Vec<(R::Sector, usize)> = pairs.into_iter().collect();
        // Why duplicate detection precedes `SectorLeg::try_new`: the leg only
        // ever sees encoded ids, so its own duplicate error can only name a
        // `SectorId`. Diagnosing here lets the caller see the label it wrote,
        // and separates a caller duplicate from a provider whose codec aliases
        // two labels onto one id — the leg cannot tell those apart at all.
        let mut sorted: Vec<&R::Sector> = pairs.iter().map(|(label, _)| label).collect();
        sorted.sort_unstable();
        if let Some(window) = sorted.windows(2).find(|window| window[0] == window[1]) {
            return Err(Error::InvalidArgument(format!(
                "sector label {:?} is declared more than once",
                window[0]
            )));
        }

        let mut encoded = Vec::with_capacity(pairs.len());
        for (label, degeneracy) in &pairs {
            encoded.push((provider.encode_sector(label)?, label, *degeneracy));
        }
        let mut by_id: Vec<_> = encoded.iter().collect();
        by_id.sort_unstable_by_key(|(id, _, _)| *id);
        if let Some(window) = by_id.windows(2).find(|window| window[0].0 == window[1].0) {
            return Err(Error::InvalidArgument(format!(
                "SectorCodec law violation: labels {:?} and {:?} both encode to {:?}",
                window[0].1, window[1].1, window[0].0
            )));
        }

        let leg = SectorLeg::try_new(
            encoded.iter().map(|(id, _, degeneracy)| (*id, *degeneracy)),
            is_dual,
        )
        .map_err(|error| Error::InvalidArgument(error.to_string()))?;
        Ok(Self { provider, leg })
    }

    /// The sector labels carried by this leg, in the provider's
    /// [`tenet_core::SectorId`] order.
    ///
    /// The order is the engine's, deliberately: it is the order of
    /// [`Self::degeneracies`] and of every block layout derived from the leg,
    /// so re-sorting by label here would desynchronize the two. A caller that
    /// wants label order sorts the returned vector.
    ///
    /// # Errors
    ///
    /// Returns [`Error::FusionAlgebra`] when the provider cannot decode an id
    /// it previously produced, which is a violation of [`SectorCodec`]'s
    /// decode-totality law.
    pub fn sectors(&self) -> Result<Vec<R::Sector>, Error> {
        self.leg
            .sectors()
            .iter()
            .map(|&id| self.provider.decode_sector(id).map_err(Error::from))
            .collect()
    }

    /// Per-sector degeneracies, parallel to [`Self::sectors`].
    #[inline]
    pub fn degeneracies(&self) -> &[usize] {
        self.leg.degeneracies()
    }

    /// Whether this is the conjugate space (TensorKit's `V'`).
    #[inline]
    pub fn is_dual(&self) -> bool {
        self.leg.is_dual()
    }

    /// The provider this leg is bound to.
    #[inline]
    pub fn provider(&self) -> &R {
        &self.provider
    }

    /// The conjugate leg: every sector replaced by its dual (degeneracies
    /// carried along) and the dual flag flipped.
    ///
    /// # Errors
    ///
    /// Returns [`Error::FusionAlgebra`] when the provider cannot represent the
    /// dual of some sector, or when its dual collapses two of this leg's
    /// sectors onto one and so is not the involution rigidity requires. No
    /// partially dualized leg is produced in either case.
    pub fn try_dual(&self) -> Result<Self, Error> {
        Ok(Self {
            provider: Arc::clone(&self.provider),
            leg: self.leg.try_dual(self.provider.as_ref())?,
        })
    }
}

impl<R> GradedSpace<R> {
    // Bound-free so the crate-internal accessors stay usable wherever the leg
    // travels, independently of what the caller has to certify.
    pub(crate) fn leg(&self) -> &SectorLeg {
        &self.leg
    }

    pub(crate) fn provider_arc(&self) -> &Arc<R> {
        &self.provider
    }
}

/// The provider-labelled identity of one stored block: the fusion tree on each
/// side of the tensor map, decoded through the codec — the labelled
/// counterpart of [`tenet_core::FusionTreePairKey`], named after TensorKit's
/// `fusiontrees(t)`.
///
/// Why not `BlockSectors` / `block_sectors`: TensorKit's `blocksectors(t)` is
/// the set of coupled sectors of a tensor, which is a strictly smaller thing
/// than a per-block tree pair. Reusing the name for something else would be a
/// false friend for anyone arriving from TensorKit.
///
/// Inner lines are part of the identity, not decoration: from rank three up,
/// two distinct blocks can share their uncoupled and coupled sectors and
/// differ only in how the intermediate fusions ran, so a key without them
/// would not name a block.
///
/// Vertex labels are absent because this facade admits multiplicity-free
/// providers only, where every fusion vertex is the unique one.
#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct BlockFusionTrees<S> {
    coupled: S,
    codomain_uncoupled: Vec<S>,
    codomain_innerlines: Vec<S>,
    domain_uncoupled: Vec<S>,
    domain_innerlines: Vec<S>,
}

impl<S> BlockFusionTrees<S> {
    /// The sector both trees couple to.
    #[inline]
    pub fn coupled(&self) -> &S {
        &self.coupled
    }

    /// Codomain leg sectors, in codomain axis order.
    #[inline]
    pub fn codomain_uncoupled(&self) -> &[S] {
        &self.codomain_uncoupled
    }

    /// Codomain intermediate fusion sectors, from the innermost outwards.
    #[inline]
    pub fn codomain_innerlines(&self) -> &[S] {
        &self.codomain_innerlines
    }

    /// Domain leg sectors, in domain axis order.
    ///
    /// These are the domain spaces' own sectors (TensorKit's `f2.uncoupled`),
    /// not their duals; on both sides the uncoupled sectors fuse to
    /// [`Self::coupled`].
    #[inline]
    pub fn domain_uncoupled(&self) -> &[S] {
        &self.domain_uncoupled
    }

    /// Domain intermediate fusion sectors, from the innermost outwards.
    #[inline]
    pub fn domain_innerlines(&self) -> &[S] {
        &self.domain_innerlines
    }
}

fn decode_sectors<R>(provider: &R, ids: &[tenet_core::SectorId]) -> Result<Vec<R::Sector>, Error>
where
    R: SectorCodec,
{
    ids.iter()
        .map(|&id| provider.decode_sector(id).map_err(Error::from))
        .collect()
}

/// Decodes one block key into provider labels.
///
/// Every id here came out of the engine's own fusion enumeration, so a failure
/// is the provider breaking [`SectorCodec`]'s decode-totality law, and it is
/// surfaced as the codec's own error rather than a panic.
fn decode_block_fusion_trees<R>(
    provider: &R,
    key: &BlockKey,
) -> Result<BlockFusionTrees<R::Sector>, Error>
where
    R: SectorCodec,
{
    let pair = key.as_fusion_tree_pair().ok_or_else(|| {
        Error::InvalidArgument(format!(
            "block key is {}, not a fusion-tree pair",
            key.kind()
        ))
    })?;
    let codomain = pair.codomain_tree();
    let domain = pair.domain_tree();
    Ok(BlockFusionTrees {
        coupled: provider.decode_sector(codomain.coupled())?,
        codomain_uncoupled: decode_sectors(provider, codomain.uncoupled())?,
        codomain_innerlines: decode_sectors(provider, codomain.innerlines())?,
        domain_uncoupled: decode_sectors(provider, domain.uncoupled())?,
        domain_innerlines: decode_sectors(provider, domain.innerlines())?,
    })
}

/// Storage shared by every clone of one typed tensor map: the admitted space
/// and its block payload.
struct TypedTensorBody<R, D> {
    space: BoundDynamicFusionMapSpace<R>,
    data: Vec<D>,
}

/// A block-sparse symmetric tensor map that keeps its provider type.
///
/// `D` is the payload dtype ([`f64`] or [`num_complex::Complex64`]) and is
/// independent of the provider's real categorical coefficient scalar — the
/// same separation TensorKit makes between a tensor's `T` and its sector type.
///
/// Cloning is cheap: the runtime handle and the shared body are both
/// reference-counted.
pub struct TensorMap<R, D> {
    runtime: Runtime,
    body: Arc<TypedTensorBody<R, D>>,
}

// Why hand-written: the derives would demand `R: Clone` and `D: Clone`, and
// neither is needed behind the shared `Arc`.
impl<R, D> Clone for TensorMap<R, D> {
    fn clone(&self) -> Self {
        Self {
            runtime: self.runtime.clone(),
            body: Arc::clone(&self.body),
        }
    }
}

impl<R, D> core::fmt::Debug for TensorMap<R, D> {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        formatter
            .debug_struct("TensorMap")
            .field("elements", &self.body.data.len())
            .finish_non_exhaustive()
    }
}

impl<R, D> TensorMap<R, D>
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64> + CheckedFusionAlgebra + SectorCodec,
    D: TensorScalar,
{
    /// The provider that owns the layout, after proving every leg agrees on
    /// the rule identity.
    fn authority<'a>(legs: &[&'a GradedSpace<R>]) -> Result<&'a Arc<R>, Error> {
        let (authority, rest) = legs.split_first().ok_or_else(|| {
            Error::InvalidArgument(
                "at least one leg is required to infer the fusion provider".to_string(),
            )
        })?;
        // Why compare `RuleIdentity` rather than `Arc::ptr_eq`: separately
        // allocated providers of one rule must interoperate, exactly as they do
        // for the erased facade. Checking here also means a mismatch is
        // reported before any provider algebra or layout staging runs.
        let identity = authority.provider().rule_identity();
        if rest
            .iter()
            .any(|leg| leg.provider().rule_identity() != identity)
        {
            return Err(Error::RuleMismatch);
        }
        Ok(authority.provider_arc())
    }

    fn build(
        runtime: &Runtime,
        provider: Arc<R>,
        codomain: &[&GradedSpace<R>],
        domain: &[&GradedSpace<R>],
        fill: Fill<'_, D>,
    ) -> Result<Self, Error> {
        let hom = FusionTreeHomSpace::new(
            FusionProductSpace::new(codomain.iter().map(|leg| leg.leg().clone())),
            FusionProductSpace::new(domain.iter().map(|leg| leg.leg().clone())),
        );
        // Why only the checked root: the infallible enumeration reaches legacy
        // encoded paths that may panic on an external provider's unrepresentable
        // value. The checked root publishes no layout, cache, or admission state
        // until every fallible stage has passed.
        let space = BoundDynamicFusionMapSpace::from_final_homspace_multiplicity_free_checked(
            provider, hom,
        )?;
        let data = apply_fill(space.space(), fill)?;
        Ok(Self {
            runtime: runtime.clone(),
            body: Arc::new(TypedTensorBody { space, data }),
        })
    }

    /// Zero tensor map on `codomain <- domain` (TensorKit `zeros(T, W <- V)`).
    ///
    /// The payload dtype comes from `D`, so no dtype token is needed. Every
    /// leg must carry a provider with the same
    /// [`tenet_core::FusionRule::rule_identity`]; the first leg's provider allocation
    /// becomes the tensor's authority.
    ///
    /// # Errors
    ///
    /// - [`Error::InvalidArgument`] when no leg is given, since the provider is
    ///   inferred from the legs.
    /// - [`Error::RuleMismatch`] when the legs disagree on the rule identity.
    ///   This is reported before any provider algebra runs.
    /// - [`Error::FusionAlgebra`] / [`Error::Operation`] when the provider
    ///   cannot certify the layout. Nothing is published in that case.
    pub fn zeros<'a, Codomain, Domain>(
        runtime: &Runtime,
        codomain: Codomain,
        domain: Domain,
    ) -> Result<Self, Error>
    where
        Codomain: IntoIterator<Item = &'a GradedSpace<R>>,
        Domain: IntoIterator<Item = &'a GradedSpace<R>>,
        R: 'a,
    {
        let codomain: Vec<&GradedSpace<R>> = codomain.into_iter().collect();
        let domain: Vec<&GradedSpace<R>> = domain.into_iter().collect();
        let legs: Vec<&GradedSpace<R>> = codomain.iter().chain(&domain).copied().collect();
        let provider = Arc::clone(Self::authority(&legs)?);
        Self::build(runtime, provider, &codomain, &domain, Fill::Zeros)
    }

    /// Tensor map whose every symmetry-allowed element is produced by
    /// `fill(sectors, indices)`.
    ///
    /// `sectors` names the block through the provider's own labels; `indices`
    /// are the degeneracy coordinates local to that block, codomain axes
    /// first, first axis fastest. The payload dtype follows `D`.
    ///
    /// The block labels are decoded once per block, not once per element: the
    /// erased odometer underneath reports the same key for every element of a
    /// block, and the decode is memoized against it.
    ///
    /// # Errors
    ///
    /// Everything [`Self::zeros`] reports, plus [`Error::FusionAlgebra`] when
    /// the provider cannot decode a sector its own algebra produced.
    pub fn from_block_fn<'a, Codomain, Domain, F>(
        runtime: &Runtime,
        codomain: Codomain,
        domain: Domain,
        mut fill: F,
    ) -> Result<Self, Error>
    where
        Codomain: IntoIterator<Item = &'a GradedSpace<R>>,
        Domain: IntoIterator<Item = &'a GradedSpace<R>>,
        F: FnMut(&BlockFusionTrees<R::Sector>, &[usize]) -> D,
        R: 'a,
    {
        let codomain: Vec<&GradedSpace<R>> = codomain.into_iter().collect();
        let domain: Vec<&GradedSpace<R>> = domain.into_iter().collect();
        let legs: Vec<&GradedSpace<R>> = codomain.iter().chain(&domain).copied().collect();
        let provider = Arc::clone(Self::authority(&legs)?);

        // Why reuse the erased `Fill::BlockFn` odometer instead of walking the
        // blocks here: the traversal order and the strided element addressing
        // are exactly the erased facade's, and duplicating them would be a
        // second place for the two to drift apart. The adapter below only adds
        // the label decode — memoized on the block key, so the cost stays one
        // decode per block plus one key comparison per element, which the
        // odometer already pays per element anyway.
        let mut decode_failure: Option<Error> = None;
        let mut memo: Option<(BlockKey, BlockFusionTrees<R::Sector>)> = None;
        let mut labelled = |key: &BlockKey, indices: &[usize]| -> D {
            if decode_failure.is_some() {
                return D::from_real(0.0);
            }
            if memo.as_ref().is_none_or(|(cached, _)| cached != key) {
                match decode_block_fusion_trees(provider.as_ref(), key) {
                    Ok(sectors) => memo = Some((key.clone(), sectors)),
                    Err(error) => {
                        decode_failure = Some(error);
                        return D::from_real(0.0);
                    }
                }
            }
            let (_, sectors) = memo.as_ref().expect("memo was just populated");
            fill(sectors, indices)
        };

        let built = Self::build(
            runtime,
            Arc::clone(&provider),
            &codomain,
            &domain,
            Fill::BlockFn(&mut labelled),
        );
        // A decode failure inside the infallible odometer callback is reported
        // here; the partially written buffer never leaves this function.
        if let Some(error) = decode_failure {
            return Err(error);
        }
        built
    }

    /// TensorKit `permute`: re-arranges legs with symmetric braiding.
    ///
    /// `codomain_axes` and `domain_axes` list source axis numbers (`0..rank`,
    /// codomain axes first) for the new codomain and domain — the same
    /// argument shape as the erased [`crate::prelude::Tensor::permute`], so
    /// there is one vocabulary for the operation rather than two.
    ///
    /// # Errors
    ///
    /// [`Error::Operation`] / [`Error::Core`] / [`Error::FusionAlgebra`] when
    /// the axis lists are malformed (out of range, repeated, or not a
    /// partition of `0..rank`) or the provider cannot support the braiding the
    /// requested motion needs. The expert layer's own typed errors are the
    /// contract here: re-validating the axes at this layer would be a second
    /// copy of a rule that already exists one call down, free to drift.
    pub fn permute(&self, codomain_axes: &[usize], domain_axes: &[usize]) -> Result<Self, Error> {
        // Why no identity shortcut (the erased facade shares storage when the
        // axes do not move): the result would be byte-identical either way, so
        // the shortcut is a pure cost question, and adding one without a gate
        // that measures it is speculative. The same reasoning covers every
        // other operation routed through `tree_transform` below.
        self.tree_transform(TreeTransformOperation::permute(
            codomain_axes.iter().copied(),
            domain_axes.iter().copied(),
        ))
    }

    /// TensorKit `braid`: re-arranges legs with an explicit braid, one level
    /// per source axis.
    ///
    /// `codomain_axes` and `domain_axes` name source axes exactly as for
    /// [`Self::permute`]. `levels` is per source *strand*, one entry for every
    /// axis in `0..rank` — codomain axes first — and it is split by the
    /// **source** codomain rank, so entry `i` always describes source axis `i`
    /// regardless of where that axis ends up. The levels decide which strand
    /// crosses above at each transposition; for a symmetric (bosonic) braiding
    /// they cannot change the result, and this is then [`Self::permute`].
    ///
    /// # Errors
    ///
    /// [`Error::InvalidArgument`] when `levels` does not have one entry per
    /// source axis — the one check this facade makes itself, because the axis
    /// lists and the levels are validated by different layers and a
    /// mis-lengthed `levels` would otherwise be split silently.
    ///
    /// Otherwise [`Error::Operation`] / [`Error::Core`] /
    /// [`Error::FusionAlgebra`] straight from the expert layer for malformed
    /// axis lists or a provider that cannot support the requested braiding.
    /// As for [`Self::permute`], those errors are the contract; this layer does
    /// not re-validate axes.
    pub fn braid(
        &self,
        codomain_axes: &[usize],
        domain_axes: &[usize],
        levels: &[usize],
    ) -> Result<Self, Error> {
        // Mirrors the erased pre-check verbatim (`Tensor::transformed`), same
        // message: two facades reporting one mistake two ways is a support
        // burden with no upside.
        let rank = self.rank();
        if levels.len() != rank {
            return Err(Error::InvalidArgument(format!(
                "braid levels must list one level per source axis \
                 (expected {rank}, got {})",
                levels.len()
            )));
        }
        let nout = self.codomain_rank();
        self.tree_transform(TreeTransformOperation::braid(
            codomain_axes.iter().copied(),
            domain_axes.iter().copied(),
            levels[..nout].iter().copied(),
            levels[nout..].iter().copied(),
        ))
    }

    /// TensorKit `transpose`: the planar transpose of `codomain <- domain` to
    /// `domain' <- codomain'`, i.e. a cyclic rotation of the legs round the
    /// planar boundary by the codomain rank, which is what carries every
    /// codomain leg across the boundary and every domain leg back.
    ///
    /// Planar means it **never braids**: legs are bent across the boundary, and
    /// bending conjugates them, so the result's spaces carry flipped dual
    /// flags. Spelling this as a [`Self::permute`] of the same axis order would
    /// be wrong for any provider whose braiding is not symmetric — the two
    /// agree only up to the R-symbols a permute inserts and this does not.
    ///
    /// It is its own inverse: transposing twice restores the source layout.
    ///
    /// # Errors
    ///
    /// [`Error::Operation`] / [`Error::Core`] / [`Error::FusionAlgebra`] from
    /// the expert layer. The generated axis order is planar by construction, so
    /// a failure here means the provider could not carry the bend.
    pub fn transpose(&self) -> Result<Self, Error> {
        self.planar(PlanarRequestKind::FullTranspose)
    }

    /// TensorKit `transpose` with an explicit cyclic axis map.
    ///
    /// The name is Rust-only, disambiguating this from [`Self::transpose`]:
    /// TensorKit has a single `transpose` taking an optional `Index2Tuple`,
    /// which Rust cannot spell as one method. TensorKit's argument-free
    /// `transpose` is [`Self::transpose`]; this is the explicit form.
    ///
    /// `codomain_axes` and `domain_axes` are flat source axis numbers
    /// (`0..rank`, codomain axes first), exactly as for [`Self::permute`], but
    /// together they must describe one **cyclic rotation** of the planar source
    /// order (codomain axes followed by the domain axes reversed). Unlike
    /// [`Self::permute`], this operation never braids.
    ///
    /// # Errors
    ///
    /// [`Error::Operation`] / [`Error::Core`] / [`Error::FusionAlgebra`] when
    /// the axis lists are malformed or are not a cyclic rotation of the planar
    /// order — a re-arrangement that would need a braid is refused rather than
    /// silently braided. As everywhere in this
    /// facade the expert layer owns that validation; it is not repeated here.
    pub fn transpose_axes(
        &self,
        codomain_axes: &[usize],
        domain_axes: &[usize],
    ) -> Result<Self, Error> {
        self.planar(PlanarRequestKind::Explicit {
            codomain_axes,
            domain_axes,
        })
    }

    /// TensorKit `repartition(t, N₁, N₂)`: moves the planar boundary so the
    /// codomain holds `num_codomain` legs and the domain holds the rest.
    ///
    /// The planar order — codomain followed by reversed domain — is preserved;
    /// legs that cross the boundary are bent, and so arrive with their dual
    /// flag flipped and their sectors dualized, without any braid being
    /// introduced.
    ///
    /// # Errors
    ///
    /// [`Error::InvalidArgument`] when `num_codomain` exceeds the rank, and
    /// otherwise [`Error::Operation`] / [`Error::Core`] /
    /// [`Error::FusionAlgebra`] from the expert layer, which owns the
    /// validation this facade passes through.
    pub fn repartition(&self, num_codomain: usize) -> Result<Self, Error> {
        self.planar(PlanarRequestKind::Repartition { num_codomain })
    }

    /// Shared body of the three planar operations: derive the planar axis
    /// order, let the expert layer check it, and run it as a transpose.
    ///
    /// Why the axis derivation is borrowed from the erased layer rather than
    /// rewritten here: it *is* the definition of what "planar" means for each
    /// request kind, and a second copy would be free to drift from the erased
    /// sibling these operations are byte-compared against.
    fn planar(&self, kind: PlanarRequestKind<'_>) -> Result<Self, Error> {
        with_planar_axes(
            self.codomain_rank(),
            self.rank(),
            kind,
            |codomain_axes, domain_axes| {
                // Why `transpose` and not `permute` even when the axes happen
                // to be a plain permutation: domain trees run opposite to the
                // planar boundary, so flattening them into a permute would
                // braid a different leg across it.
                self.tree_transform(TreeTransformOperation::transpose(
                    codomain_axes.iter().copied(),
                    domain_axes.iter().copied(),
                ))
            },
        )
    }

    /// Runs one prepared tree transform on this tensor's own runtime.
    fn tree_transform(&self, operation: TreeTransformOperation) -> Result<Self, Error> {
        // Leasing rather than locking, matching the erased path: independent
        // operations on one runtime must not serialize behind each other.
        let mut lease = self.runtime.lease_context()?;
        let (space, data) = tree_transform_owned_multiplicity_free(
            lease.context().multiplicity_free_lane::<D>(),
            BoundDynamicTensorRef::try_new(&self.body.space, &self.body.data)?,
            operation,
        )?;
        Ok(Self {
            runtime: self.runtime.clone(),
            body: Arc::new(TypedTensorBody { space, data }),
        })
    }

    fn codomain_rank(&self) -> usize {
        self.body.space.space().homspace().codomain().len()
    }

    fn rank(&self) -> usize {
        self.codomain_rank() + self.body.space.space().homspace().domain().len()
    }

    /// Contracts `lhs_axes` of `self` with `rhs_axes` of `other` (pairwise, in
    /// list order) and lays the open axes out in `output_axes`.
    ///
    /// `output_axes` is a permutation of `0..open_rank` over the open axes,
    /// `self`'s ascending first and `other`'s after; passing `0..open_rank`
    /// gives the default order (TensorKit `tensorcontract!` with default
    /// `pAB`). The codomain/domain split of the result follows the
    /// TensorOperations convention the engine implements: **every** open axis
    /// of `self` becomes the result's codomain
    /// (`self.rank() - lhs_axes.len()` axes) and every open axis of `other`
    /// becomes its domain (`other.rank() - rhs_axes.len()` axes), regardless
    /// of which side of either operand those axes came from.
    ///
    /// **Fermionic semantics**: like TensorKit `tensorcontract!` / `@tensor`
    /// (and the erased [`crate::prelude::Tensor::contract`]), this **twists**
    /// dual contracted legs with the fermionic supertrace twist — unlike
    /// composition (TensorKit `A * B` / `mul!`), which never does. Bosonic
    /// rules are unaffected; fermionic rules can differ by signs. There is no
    /// typed `compose` yet, so this is the only contraction semantics the
    /// typed facade offers.
    ///
    /// The result is bound to `self`'s provider allocation, the same
    /// left-authority rule [`Self::zeros`] uses for its first leg: the two
    /// operands must agree on
    /// [`tenet_core::FusionRule::rule_identity`], which makes the choice of
    /// allocation immaterial to the algebra.
    ///
    /// # Errors
    ///
    /// - [`Error::RuntimeMismatch`] when the operands belong to different
    ///   runtimes.
    /// - [`Error::Operation`] / [`Error::Core`] / [`Error::FusionAlgebra`] for
    ///   malformed axis lists, an output order that is not a permutation of
    ///   the open axes, mismatched contracted legs, or operands whose
    ///   providers report different rule identities. Those all come back from
    ///   the expert layer, which owns the rules; re-checking them here would
    ///   be a second copy free to drift.
    pub fn contract(
        &self,
        other: &Self,
        lhs_axes: &[usize],
        rhs_axes: &[usize],
        output_axes: &[usize],
    ) -> Result<Self, Error> {
        // The one check the expert layer cannot make: it never sees the two
        // runtimes, and mixing execution state across them is a trust-boundary
        // violation rather than an algebra error. Mirrors the erased facade's
        // `check_same_world`. Dtype and placement need no arm here — `D` is a
        // type parameter and the typed facade is host-only.
        if !self.runtime.same_runtime(&other.runtime) {
            return Err(Error::RuntimeMismatch);
        }
        let mut lease = self.runtime.lease_context()?;
        let (space, data) = tensorcontract_owned_multiplicity_free(
            lease.context().multiplicity_free_lane::<D>(),
            BoundDynamicTensorRef::try_new(&self.body.space, &self.body.data)?,
            BoundDynamicTensorRef::try_new(&other.body.space, &other.body.data)?,
            lhs_axes,
            rhs_axes,
            // Why `OutputAxisOrder` stays out of the signature: it is an
            // expert-layer borrow type, and a `&[usize]` says the same thing
            // at the facade without a second public vocabulary.
            OutputAxisOrder::from_axes(output_axes),
        )?;
        Ok(Self {
            runtime: self.runtime.clone(),
            body: Arc::new(TypedTensorBody { space, data }),
        })
    }

    /// The codomain legs, in axis order.
    pub fn codomain(&self) -> Vec<GradedSpace<R>> {
        self.legs(self.body.space.space().homspace().codomain())
    }

    /// The domain legs, in axis order.
    pub fn domain(&self) -> Vec<GradedSpace<R>> {
        self.legs(self.body.space.space().homspace().domain())
    }

    fn legs(&self, product: &FusionProductSpace) -> Vec<GradedSpace<R>> {
        product
            .legs()
            .iter()
            .map(|leg| GradedSpace {
                provider: Arc::clone(self.body.space.provider_arc()),
                leg: leg.clone(),
            })
            .collect()
    }

    /// The provider-labelled fusion trees of the block at `index`.
    ///
    /// # Errors
    ///
    /// [`Error::Core`] when `index` is out of range, [`Error::FusionAlgebra`]
    /// when the provider cannot decode one of its own sectors.
    pub fn block_fusion_trees(&self, index: usize) -> Result<BlockFusionTrees<R::Sector>, Error> {
        let block = self.body.space.space().structure().block(index)?;
        decode_block_fusion_trees(self.body.space.provider(), block.key())
    }

    /// Engine-level layout view of the block at `index`: its shape, strides
    /// and offset addressing into [`Self::data`].
    ///
    /// This is the one accessor here that speaks the engine's vocabulary
    /// rather than the provider's — the returned [`BlockRef`] exposes the raw
    /// [`tenet_core::BlockKey`], whose sectors are [`tenet_core::SectorId`]s.
    /// It is not wrapped because a layout view has no use for labels; for the
    /// block's categorical identity use [`Self::block_fusion_trees`], which
    /// reports the same block through the codec.
    ///
    /// # Errors
    ///
    /// [`Error::Core`] when `index` is out of range.
    pub fn block(&self, index: usize) -> Result<BlockRef<'_>, Error> {
        self.body
            .space
            .space()
            .structure()
            .block(index)
            .map_err(Error::from)
    }

    /// The runtime this tensor map is bound to.
    #[inline]
    pub fn runtime(&self) -> &Runtime {
        &self.runtime
    }

    /// Number of stored symmetry-allowed blocks.
    #[inline]
    pub fn block_count(&self) -> usize {
        self.body.space.space().structure().block_count()
    }

    /// The whole block payload in storage order.
    ///
    /// Individual blocks address this buffer through their own offset and
    /// strides.
    #[inline]
    pub fn data(&self) -> &[D] {
        &self.body.data
    }
}
