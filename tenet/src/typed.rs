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
//! [`TensorMap::repartition`] and [`TensorMap::contract`] — plus the
//! decompositions of issue #567: [`TensorMap::svd_compact`],
//! [`TensorMap::svd_full`], [`TensorMap::svd_trunc`], [`TensorMap::svd_vals`],
//! [`TensorMap::qr_compact`], [`TensorMap::qr_full`],
//! [`TensorMap::lq_compact`], [`TensorMap::lq_full`],
//! [`TensorMap::left_orth`], [`TensorMap::right_orth`],
//! [`TensorMap::left_null`] and [`TensorMap::right_null`] — plus the scalar
//! operations of issue #568: [`TensorMap::add`], [`TensorMap::scale`],
//! [`TensorMap::norm`], [`TensorMap::norm_inf`], [`TensorMap::normalize`],
//! [`TensorMap::inner`], [`TensorMap::dot`], [`TensorMap::tr`],
//! [`TensorMap::trace_pairs`] and [`TensorMap::adjoint`].
//!
//! Everything else is deliberately still absent, each for its own reason:
//!
//! - The **eigendecompositions** (`eigh_*`, `eig_*`) ride with the typed
//!   diagonal-storage question of issue #570: `eigh_full`'s `d` factor has no
//!   seam and would have to instantiate that question rather than inherit it,
//!   and `eig_*` additionally needs a per-method `D::Eig` bound. Shipping part
//!   of the family would leave a broken parity row.
//! - The **operator overloads** (`impl Add`, `impl Mul`) are out on purpose.
//!   `&a * &b` means [`crate::prelude::Tensor::compose`] in the erased facade
//!   and this facade has no `compose`, so a typed `Mul` meaning scalar
//!   multiplication would be a cross-facade false friend. And an operator
//!   cannot return a `Result`: the erased `Mul` precedent panics, and a
//!   panicking `+` as the only spelling of addition contradicts this facade's
//!   passthrough-error contract. Adding them later is not a breaking change.
//! - The **`is_hermitian` / `project_*` family** is blocked: four of its seven
//!   members need `compose`, `id` or `eigh_vals`, none of which exist here, and
//!   shipping the reachable three would leave the broken parity row this doc
//!   refuses everywhere else.
//! - **Composition** — TensorKit `A * B`, which unlike [`TensorMap::contract`]
//!   never twists dual legs — is blocked below this layer: fermionic compose
//!   needs a new public seam over `LoweredMultiplicityFreeAlgebra`, which
//!   `tenet-core` seals, and a silently
//!   bosonic-only `compose` would return wrong fermionic signs rather than an
//!   error.
//! - `conj` stays design-gated on its open correctness question for
//!   non-self-dual sectors. [`TensorMap::adjoint`] is now in, eagerly: see its
//!   own documentation for why that is TensorKit's `adjoint!` rather than a
//!   divergence from the erased facade's lazy view.
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
/// Re-exported for the same reason as [`Error`] and [`Runtime`]:
/// [`TensorMap::svd_trunc`] takes one, so `use tenet::typed::*` would not be
/// self-sufficient without it.
pub use tenet_matrixalgebra::Truncation;

use tenet_matrixalgebra::BoundDynFactor;

use crate::tensor::{
    apply_fill, weighted_inner, weighted_trace, with_planar_axes, Fill, PlanarRequestKind,
    TensorScalar,
};
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

/// One coupled sector's factorization spectrum, labelled through the provider:
/// the typed counterpart of [`tenet_matrixalgebra::SectorSpectrum`], whose
/// `sector` is a raw [`tenet_core::SectorId`].
///
/// Why decode rather than extend the raw-id exception that [`TensorMap::block`]
/// carries: that exception is scoped to engine layout views, and a spectrum is
/// caller-facing physics — [`TensorMap::svd_vals`]'s entire return would
/// otherwise be raw ids.
///
/// `values` is descending by magnitude, as the seam guarantees.
#[derive(Clone, Debug, PartialEq)]
pub struct SectorSpectrum<S> {
    /// The coupled sector, in the provider's own labels.
    pub sector: S,
    /// That sector's values, descending by magnitude.
    pub values: Vec<f64>,
}

/// Result of [`TensorMap::svd_trunc`]: `t ~ u * s * vh` with the truncated
/// bond (TensorKit 0.17 `svd_trunc`, which returns `(U, S, Vᴴ, ϵ)`).
// The `SectorCodec` bound is the field types' own: `singular_values` is
// labelled, so the struct cannot be spelled without it.
pub struct SvdTrunc<R: SectorCodec, D> {
    /// Left isometry `u : codomain <- bond`.
    pub u: TensorMap<R, D>,
    /// Singular-value factor `s : bond <- bond`. Dense — see the order-parity
    /// gap on [`TensorMap::svd_compact`] (issue #570).
    pub s: TensorMap<R, D>,
    /// Right isometry `vh : bond <- domain`.
    pub vh: TensorMap<R, D>,
    /// Kept singular values per coupled sector, sorted by provider label.
    pub singular_values: Vec<SectorSpectrum<R::Sector>>,
    /// Quantum-dimension-weighted 2-norm of everything discarded.
    pub error: f64,
}

// Why hand-written, as for `TensorMap` itself: the derives would demand
// `R: Clone + Debug`, and neither is needed — the provider lives behind an
// `Arc` and its labels, not the rule, are what a diagnostic shows.
impl<R, D> Clone for SvdTrunc<R, D>
where
    R: SectorCodec,
{
    fn clone(&self) -> Self {
        Self {
            u: self.u.clone(),
            s: self.s.clone(),
            vh: self.vh.clone(),
            singular_values: self.singular_values.clone(),
            error: self.error,
        }
    }
}

impl<R, D> core::fmt::Debug for SvdTrunc<R, D>
where
    R: SectorCodec,
{
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        // Every field is shown: `TensorMap`'s own `Debug` is bound-free, so
        // there is nothing about the factors this impl cannot print, and the
        // erased `SvdTrunc` shows its three tensors too.
        formatter
            .debug_struct("SvdTrunc")
            .field("u", &self.u)
            .field("s", &self.s)
            .field("vh", &self.vh)
            .field("singular_values", &self.singular_values)
            .field("error", &self.error)
            .finish()
    }
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

    /// Wraps one factor the matrix-algebra seam produced into a typed tensor
    /// map. `BoundDynFactor::into_parts` hands back exactly the pair
    /// [`TypedTensorBody`] stores, so there is nothing to validate here — the
    /// seam already certified the space against its own data.
    fn wrap_bound_factor(&self, factor: BoundDynFactor<R, D>) -> Self {
        let (space, data) = factor.into_parts();
        Self {
            runtime: self.runtime.clone(),
            body: Arc::new(TypedTensorBody { space, data }),
        }
    }

    /// Decodes a seam spectrum into provider labels and sorts it by label.
    ///
    /// Every id here came out of the engine's own coupled-sector enumeration,
    /// so a decode failure is the provider breaking [`SectorCodec`]'s
    /// decode-totality law — same contract as [`decode_block_fusion_trees`].
    fn decode_spectrum(
        &self,
        raw: Vec<tenet_matrixalgebra::SectorSpectrum>,
    ) -> Result<Vec<SectorSpectrum<R::Sector>>, Error> {
        let provider = self.body.space.provider();
        let mut decoded: Vec<SectorSpectrum<R::Sector>> = raw
            .into_iter()
            .map(|entry| {
                Ok(SectorSpectrum {
                    sector: provider.decode_sector(entry.sector)?,
                    values: entry.values,
                })
            })
            .collect::<Result<_, Error>>()?;
        // Label order, not the engine's id order: see the type's own rustdoc
        // for why this facade sorts and the erased one does not.
        decoded.sort_by(|left, right| left.sector.cmp(&right.sector));
        Ok(decoded)
    }

    /// Borrowed seam view of this tensor map.
    fn bound_ref(&self) -> Result<BoundDynamicTensorRef<'_, R, D>, Error> {
        BoundDynamicTensorRef::try_new(&self.body.space, &self.body.data).map_err(Error::from)
    }

    /// TensorKit 0.17 / MatrixAlgebraKit `svd_compact`: `t = u * s * vh` with
    /// the bond `min(rows, cols)` per coupled sector.
    ///
    /// Returns `(u, s, vh)` with `u : codomain <- bond`, `s : bond <- bond`
    /// and `vh : bond <- domain`.
    ///
    /// # Order-parity gap (issue #570)
    ///
    /// TensorKit's `svd_compact` returns `s` as a `DiagonalTensorMap`, and the
    /// erased [`crate::prelude::Tensor`] matches it with diagonal storage. This
    /// facade has only dense block storage, so `s` costs `Σ_c k_c²` instead of
    /// `Σ_c k_c`, and a downstream `u * s * vh` runs the dense GEMM path rather
    /// than the O(d·n) block scaling. Interim guidance: a caller that only
    /// needs the spectrum should use [`Self::svd_vals`], which never
    /// materializes `s` at all. When typed diagonal storage lands the signature
    /// does not change — only the storage behind `s`.
    ///
    /// # Errors
    ///
    /// [`Error::Operation`] / [`Error::Core`] / [`Error::FusionAlgebra`]
    /// straight from the matrix-algebra seam. As everywhere in this facade
    /// there are no pre-checks here: the seam owns the rules, and a second copy
    /// would be free to drift.
    pub fn svd_compact(&self) -> Result<(Self, Self, Self), Error> {
        // Dense lease only, matching the erased sibling: a factorization runs
        // entirely on the dense-executor boundary, so leasing the (scarcer)
        // recoupling context here would serialize unrelated work for nothing.
        let mut dense = self.runtime.lease_dense();
        let out = tenet_matrixalgebra::svd_compact_dyn(dense.dense(), &self.bound_ref()?)?;
        let (u, s, vh, _) = out.into_parts();
        Ok((
            self.wrap_bound_factor(u),
            self.wrap_bound_factor(s),
            self.wrap_bound_factor(vh),
        ))
    }

    /// TensorKit 0.17 / MatrixAlgebraKit `svd_full`: `t = u * s * vh` with
    /// square unitaries and a rectangular `s` per coupled sector.
    ///
    /// Returns `(u, s, vh)` with `u : codomain <- W`, `s : W <- W'` and
    /// `vh : W' <- domain`.
    ///
    /// Unlike [`Self::svd_compact`] this carries no order-parity gap: TensorKit's
    /// own `svd_full` builds `s` as a dense rectangular tensor
    /// (`similar(t, real(scalartype(t)), V_cod <- V_dom)`), so a dense `s` here
    /// is TK-exact.
    ///
    /// # Errors
    ///
    /// As [`Self::svd_compact`]: the seam's own errors, unfiltered.
    pub fn svd_full(&self) -> Result<(Self, Self, Self), Error> {
        let mut dense = self.runtime.lease_dense();
        let out = tenet_matrixalgebra::svd_full_dyn(dense.dense(), &self.bound_ref()?)?;
        let (u, s, vh, _) = out.into_parts();
        Ok((
            self.wrap_bound_factor(u),
            self.wrap_bound_factor(s),
            self.wrap_bound_factor(vh),
        ))
    }

    /// TensorKit 0.17 / MatrixAlgebraKit `svd_trunc`: `t ~ u * s * vh` with the
    /// bond truncated by `truncation`; see [`SvdTrunc`].
    ///
    /// TensorKit returns the four-tuple `(U, S, Vᴴ, ϵ)`; this returns them as a
    /// named struct, following the same rule the erased facade uses — tuples up
    /// to three, a struct beyond.
    ///
    /// The dense-`s` order-parity gap of [`Self::svd_compact`] (issue #570)
    /// applies here verbatim, with the same interim guidance.
    ///
    /// # Errors
    ///
    /// [`Error::Operation`] / [`Error::Core`] / [`Error::FusionAlgebra`] from
    /// the seam, including a malformed `truncation` — the truncation policy is
    /// validated where it is applied, not here.
    pub fn svd_trunc(&self, truncation: &Truncation) -> Result<SvdTrunc<R, D>, Error> {
        let mut dense = self.runtime.lease_dense();
        let out =
            tenet_matrixalgebra::svd_trunc_dyn(dense.dense(), &self.bound_ref()?, truncation)?;
        let (u, s, vh, singular_values, error) = out.into_parts();
        Ok(SvdTrunc {
            u: self.wrap_bound_factor(u),
            s: self.wrap_bound_factor(s),
            vh: self.wrap_bound_factor(vh),
            singular_values: self.decode_spectrum(singular_values)?,
            error,
        })
    }

    /// TensorKit 0.17 / MatrixAlgebraKit `svd_vals`: the singular values per
    /// coupled sector, and nothing else.
    ///
    /// No factor tensor is built, so this is also the way around the dense-`s`
    /// ceiling documented on [`Self::svd_compact`].
    ///
    /// # Errors
    ///
    /// [`Error::Operation`] / [`Error::Core`] from the seam, plus
    /// [`Error::FusionAlgebra`] when the provider cannot decode a coupled
    /// sector its own algebra produced.
    pub fn svd_vals(&self) -> Result<Vec<SectorSpectrum<R::Sector>>, Error> {
        let mut dense = self.runtime.lease_dense();
        let raw = tenet_matrixalgebra::svd_vals_dyn(dense.dense(), &self.bound_ref()?)?;
        self.decode_spectrum(raw)
    }

    /// TensorKit 0.17 / MatrixAlgebraKit `qr_compact`: `t = q * r` with `q`
    /// carrying orthonormal columns per coupled sector.
    ///
    /// # Errors
    ///
    /// As [`Self::svd_compact`]: the seam's own errors, unfiltered.
    pub fn qr_compact(&self) -> Result<(Self, Self), Error> {
        let mut dense = self.runtime.lease_dense();
        let (q, r) = tenet_matrixalgebra::qr_compact_dyn(dense.dense(), &self.bound_ref()?)?;
        Ok((self.wrap_bound_factor(q), self.wrap_bound_factor(r)))
    }

    /// TensorKit 0.17 / MatrixAlgebraKit `qr_full`: `t = q * r` with a square
    /// `q` per coupled sector.
    ///
    /// # Errors
    ///
    /// As [`Self::svd_compact`]: the seam's own errors, unfiltered.
    pub fn qr_full(&self) -> Result<(Self, Self), Error> {
        let mut dense = self.runtime.lease_dense();
        let (q, r) = tenet_matrixalgebra::qr_full_dyn(dense.dense(), &self.bound_ref()?)?;
        Ok((self.wrap_bound_factor(q), self.wrap_bound_factor(r)))
    }

    /// TensorKit 0.17 / MatrixAlgebraKit `lq_compact`: `t = l * q` with `q`
    /// carrying orthonormal rows per coupled sector.
    ///
    /// # Errors
    ///
    /// As [`Self::svd_compact`]: the seam's own errors, unfiltered.
    pub fn lq_compact(&self) -> Result<(Self, Self), Error> {
        let mut dense = self.runtime.lease_dense();
        let (l, q) = tenet_matrixalgebra::lq_compact_dyn(dense.dense(), &self.bound_ref()?)?;
        Ok((self.wrap_bound_factor(l), self.wrap_bound_factor(q)))
    }

    /// TensorKit 0.17 / MatrixAlgebraKit `lq_full`: `t = l * q` with a square
    /// `q` per coupled sector.
    ///
    /// # Errors
    ///
    /// As [`Self::svd_compact`]: the seam's own errors, unfiltered.
    pub fn lq_full(&self) -> Result<(Self, Self), Error> {
        let mut dense = self.runtime.lease_dense();
        let (l, q) = tenet_matrixalgebra::lq_full_dyn(dense.dense(), &self.bound_ref()?)?;
        Ok((self.wrap_bound_factor(l), self.wrap_bound_factor(q)))
    }

    /// TensorKit 0.17 `left_orth`: the left isometry factorization
    /// `t = v * c`, `v` isometric and `c` the corestriction.
    ///
    /// TensorKit's default `kind` is `:qr`, so this is [`Self::qr_compact`] —
    /// the same one-line delegation the erased facade makes, deliberately, so
    /// the two names cannot come to mean different things.
    ///
    /// # Errors
    ///
    /// Exactly [`Self::qr_compact`]'s.
    pub fn left_orth(&self) -> Result<(Self, Self), Error> {
        self.qr_compact()
    }

    /// TensorKit 0.17 `right_orth`: the right isometry factorization
    /// `t = c * vh`, `vh` carrying orthonormal rows.
    ///
    /// TensorKit's default `kind` is `:lq`, so this is [`Self::lq_compact`];
    /// see [`Self::left_orth`] for why it is a delegation.
    ///
    /// # Errors
    ///
    /// Exactly [`Self::lq_compact`]'s.
    pub fn right_orth(&self) -> Result<(Self, Self), Error> {
        self.lq_compact()
    }

    /// TensorKit 0.17 / MatrixAlgebraKit `left_null`: `n : codomain <- W` with
    /// `n^H * t = 0`.
    ///
    /// # Errors
    ///
    /// As [`Self::svd_compact`]: the seam's own errors, unfiltered.
    pub fn left_null(&self) -> Result<Self, Error> {
        let mut dense = self.runtime.lease_dense();
        let out = tenet_matrixalgebra::left_null_dyn(dense.dense(), &self.bound_ref()?)?;
        Ok(self.wrap_bound_factor(out))
    }

    /// TensorKit 0.17 / MatrixAlgebraKit `right_null`: `n : W <- domain` with
    /// `t * n^H = 0`.
    ///
    /// # Errors
    ///
    /// As [`Self::svd_compact`]: the seam's own errors, unfiltered.
    pub fn right_null(&self) -> Result<Self, Error> {
        let mut dense = self.runtime.lease_dense();
        let out = tenet_matrixalgebra::right_null_dyn(dense.dense(), &self.bound_ref()?)?;
        Ok(self.wrap_bound_factor(out))
    }

    /// Builds a sibling on this tensor's own space and runtime from a fresh
    /// buffer. Every element-wise scalar operation below produces exactly
    /// that: the space is unchanged and only the payload is new, so the shared
    /// [`BoundDynamicFusionMapSpace`] is cloned rather than re-derived — it
    /// carries a checked admission proof this kind of operation cannot
    /// invalidate.
    fn with_data(&self, data: Vec<D>) -> Self {
        Self {
            runtime: self.runtime.clone(),
            body: Arc::new(TypedTensorBody {
                space: self.body.space.clone(),
                data,
            }),
        }
    }

    /// The linear combination `alpha * self + beta * other`, mirroring the
    /// erased [`crate::prelude::Tensor::add`].
    ///
    /// Both operands must live on the same runtime and on the same space —
    /// identical hom space and block layout — since the combination is
    /// element-wise on the shared storage order.
    ///
    /// # False friend
    ///
    /// VectorInterface's `add(y, x, α, β)` is `y * β + x * α`: its **first**
    /// coefficient belongs to its **second** argument. Here `alpha` belongs to
    /// `self` and `beta` to `other`, matching the erased facade. Callers
    /// arriving from Julia should go by the argument order, not by the
    /// coefficient names.
    ///
    /// # Errors
    ///
    /// - [`Error::RuntimeMismatch`] when the operands belong to different
    ///   runtimes, as for [`Self::contract`].
    /// - [`Error::InvalidArgument`] when they do not live on the same space,
    ///   with the erased facade's own message.
    pub fn add(&self, other: &Self, alpha: D, beta: D) -> Result<Self, Error> {
        // Runtime first, exactly as `contract` does: crossing runtimes is a
        // trust-boundary violation rather than an algebra error, and the
        // erased facade's `check_same_space` checks it first too.
        if !self.runtime.same_runtime(&other.runtime) {
            return Err(Error::RuntimeMismatch);
        }
        // `DynamicFusionMapSpace: PartialEq` covers the hom space, the
        // codomain/domain split and the block structure, which is exactly what
        // makes the zipped element-wise combination below meaningful. Message
        // verbatim from the erased `check_same_space`: one mistake reported two
        // ways across the two facades is a support burden with no upside.
        if self.body.space.space() != other.body.space.space() {
            return Err(Error::InvalidArgument(
                "tensors live on different spaces or block layouts".to_string(),
            ));
        }
        Ok(self.with_data(
            self.body
                .data
                .iter()
                .zip(&other.body.data)
                .map(|(&x, &y)| x * alpha + y * beta)
                .collect(),
        ))
    }

    /// `factor * self` (TensorKit `scale`).
    ///
    /// Infallible, unlike the erased [`crate::prelude::Tensor::scale`]: that
    /// one returns a `Result` because it must reconcile a runtime dtype and a
    /// possible device or diagonal storage first, none of which exist here —
    /// `D` is a type parameter and the payload is always a host buffer. The
    /// erased `scale`/`scale_c64` split has the same origin and likewise
    /// collapses: `factor` is simply a `D`.
    pub fn scale(&self, factor: D) -> Self {
        self.with_data(self.body.data.iter().map(|&value| value * factor).collect())
    }

    /// Partial trace over pairs of mutually dual legs (TensorKit
    /// `tensortrace!` / TensorOperations `@tensor a[i, i; j]`).
    ///
    /// Each `(lhs, rhs)` pair of flat axis numbers (`0..rank`, codomain axes
    /// first) is traced away; the remaining legs keep their order and their
    /// codomain/domain side. Tracing nothing returns the source.
    ///
    /// This is the **tensor-contraction** trace: it applies the categorical
    /// trace coefficients, including a fermionic rule's twists, so it is the
    /// supertrace there. [`Self::tr`] is TensorKit's positive trace instead,
    /// and the two genuinely disagree for a fermionic provider.
    ///
    /// The `&[(usize, usize)]` shape mirrors the erased facade; TensorKit's
    /// native parallel-list `Index2Tuple` is what the seam takes internally.
    /// One cross-facade vocabulary wins over matching the seam's.
    ///
    /// # Errors
    ///
    /// [`Error::InvalidArgument`] when the pair list is malformed — an axis out
    /// of range, or one named twice — with the erased facade's own message.
    /// Otherwise [`Error::Operation`] / [`Error::Core`] /
    /// [`Error::FusionAlgebra`] from the seam, which owns the rest of the
    /// validation (legs that are not mutually dual, above all).
    pub fn trace_pairs(&self, pairs: &[(usize, usize)]) -> Result<Self, Error> {
        // Why this validation is kept rather than left to the seam, unlike
        // everywhere else in this facade: `seen` is not a check, it is the
        // derivation of `output_axes` below — the seam cannot supply it, and a
        // malformed list would otherwise produce a silently wrong output order
        // rather than an error. Same precedent as `braid`'s levels pre-check.
        let rank = self.rank();
        let mut seen = vec![false; rank];
        for &(lhs, rhs) in pairs {
            for axis in [lhs, rhs] {
                if axis >= rank || seen[axis] {
                    return Err(Error::InvalidArgument(format!(
                        "invalid trace pair list {pairs:?} for rank {rank} \
                         (axes must be in range and distinct)"
                    )));
                }
                seen[axis] = true;
            }
        }
        if pairs.is_empty() {
            return Ok(self.clone());
        }
        let output_axes: Vec<usize> = (0..rank).filter(|&axis| !seen[axis]).collect();
        let destination_codomain_rank = output_axes
            .iter()
            .filter(|&&axis| axis < self.codomain_rank())
            .count();
        let trace_lhs: Vec<usize> = pairs.iter().map(|&(lhs, _)| lhs).collect();
        let trace_rhs: Vec<usize> = pairs.iter().map(|&(_, rhs)| rhs).collect();
        let axes = tenet_tensors::TensorTraceAxisSpec::new(&output_axes, &trace_lhs, &trace_rhs);
        // Preflight first, exactly as the erased facade does: the checked
        // homspace selection must fail before any destination layout is
        // derived, so a rejected trace publishes no state.
        let homspace = tenet_tensors::tensortrace_fusion_dyn_selected_homspace_checked(
            &self.body.space,
            axes,
            destination_codomain_rank,
        )?;
        let space = self.body.space.derive_from_final_homspace(homspace)?;
        let data = tenet_tensors::tensortrace_fusion_dyn_owned_checked(
            &space,
            &self.body.space,
            &self.body.data,
            axes,
            D::from_real(1.0),
        )?;
        Ok(Self {
            runtime: self.runtime.clone(),
            body: Arc::new(TypedTensorBody { space, data }),
        })
    }

    /// TensorKit `adjoint` (dagger): swaps codomain and domain and
    /// conjugate-transposes every block. Real payloads are transposed only;
    /// c64 entries are conjugated as well.
    ///
    /// Eager, into a fresh destination — TensorKit's own `adjoint!`
    /// (`linalg.jl:218`), so this is a TK-sanctioned form rather than a
    /// divergence. The erased [`crate::prelude::Tensor::adjoint`] is instead
    /// the analogue of TensorKit's lazy `AdjointTensorMap` view: same result,
    /// different point at which the work is paid. Only the eager seam is
    /// reachable from this facade.
    ///
    /// # Errors
    ///
    /// [`Error::Operation`] / [`Error::Core`] / [`Error::FusionAlgebra`]
    /// straight from the seam, which owns the bend the dagger performs.
    pub fn adjoint(&self) -> Result<Self, Error> {
        let (space, data) = tenet_tensors::adjoint_bound_dyn(&self.body.space, &self.body.data)
            .map_err(Error::from)?;
        Ok(Self {
            runtime: self.runtime.clone(),
            body: Arc::new(TypedTensorBody { space, data }),
        })
    }

    /// TensorKit `norm`: the Frobenius norm weighted by the coupled sectors'
    /// quantum dimensions, `norm(t)^2 = Σ_c dim(c) * |block_c|^2`.
    ///
    /// Always real, for both payload dtypes. For an abelian rule every
    /// `dim(c)` is one and this is the plain Frobenius norm; for a non-abelian
    /// one it is not.
    ///
    /// # Errors
    ///
    /// [`Error::Core`] when the block structure cannot be walked, which is an
    /// engine-internal invariant rather than a caller mistake.
    pub fn norm(&self) -> Result<f64, Error> {
        // Same weighted reduction the erased `norm` runs, on the same helper:
        // a second copy would be free to drift from the sibling this is
        // byte-compared against.
        Ok(self.weighted_self_inner()?.re.sqrt())
    }

    /// TensorKit `norm(t, Inf)`: the largest absolute stored entry.
    ///
    /// Julia's `norm(array, Inf)` is the maximum absolute element (for
    /// matrices too), and TensorKit applies it per block, so on coupled
    /// storage it is the maximum over the whole payload. Unlike [`Self::norm`]
    /// this is **not** quantum-dimension weighted.
    ///
    /// # Errors
    ///
    /// None today; the `Result` keeps the shape of [`Self::norm`], which the
    /// two are usually reached through together.
    pub fn norm_inf(&self) -> Result<f64, Error> {
        // Why `widen_complex().norm()` rather than an f64/c64 match: the erased
        // facade needs the match because its dtype is a runtime property. Here
        // the widening is exact and `Complex64::new(x, 0.0).norm()` is exactly
        // `|x|`, so one expression covers both instantiations.
        Ok(self
            .body
            .data
            .iter()
            .map(|&value| value.widen_complex().norm())
            .fold(0.0, f64::max))
    }

    /// TensorKit `normalize`: `self / norm(self)`, the unit-norm tensor
    /// pointing the same way. The norm is [`Self::norm`]'s, so the result
    /// satisfies `t.normalize()?.norm()? == 1`.
    ///
    /// Like TensorKit, a zero-norm tensor is not special-cased: normalizing it
    /// divides by zero and yields non-finite entries. Guard the caller if that
    /// input is reachable.
    ///
    /// # Errors
    ///
    /// Exactly [`Self::norm`]'s.
    pub fn normalize(&self) -> Result<Self, Error> {
        Ok(self.scale(D::from_real(1.0 / self.norm()?)))
    }

    /// The dimension-weighted inner product of this tensor with itself, the
    /// body of [`Self::norm`].
    fn weighted_self_inner(&self) -> Result<num_complex::Complex64, Error> {
        weighted_inner(
            self.body.space.provider(),
            self.body.space.space().structure(),
            self.body.space.space().nout(),
            &self.body.data,
            &self.body.data,
        )
    }

    /// TensorKit `dot(x, y)`: the quantum-dimension-weighted Frobenius inner
    /// product `Σ_c dim(c) * <a_c, b_c>` with **`self` conjugated** — the
    /// product is conjugate-linear in its first argument.
    ///
    /// `t.inner(&t)?` is `t.norm()?²` up to floating point, and for `D = f64`
    /// the result is exactly real: the erased sibling returns
    /// `Scalar::F64(value.re)` there, and the narrowing here is the same `.re`.
    ///
    /// # Errors
    ///
    /// Exactly [`Self::add`]'s: the operands must share a runtime and a space.
    pub fn inner(&self, other: &Self) -> Result<D, Error> {
        if !self.runtime.same_runtime(&other.runtime) {
            return Err(Error::RuntimeMismatch);
        }
        if self.body.space.space() != other.body.space.space() {
            return Err(Error::InvalidArgument(
                "tensors live on different spaces or block layouts".to_string(),
            ));
        }
        // `D::from_complex64` is `.re` for the real scalar and the identity for
        // the complex one, so this is bit-identical to the erased facade's
        // `Scalar::F64(v.re)` / `Scalar::C64(v)` dispatch, without the enum.
        Ok(D::from_complex64(weighted_inner(
            self.body.space.provider(),
            self.body.space.space().structure(),
            self.body.space.space().nout(),
            &self.body.data,
            &other.body.data,
        )?))
    }

    /// `LinearAlgebra.dot` / TensorKit `dot(x, y)` — an alias for
    /// [`Self::inner`], for callers who reach for that name. The erased facade
    /// makes the same alias, deliberately, so the two names cannot come to
    /// mean different things.
    ///
    /// # Errors
    ///
    /// Exactly [`Self::inner`]'s.
    pub fn dot(&self, other: &Self) -> Result<D, Error> {
        self.inner(other)
    }

    /// TensorKit `tr`: the full trace of an endomorphism
    /// (`domain == codomain`), pairing codomain leg `i` with domain leg `i`.
    ///
    /// This is TensorKit's **positive** trace, quantum-dimension weighted:
    /// `Σ_c dim(c) * tr(b_c)`. It is *not* the supertrace — a fermionic rule's
    /// twists belong to tensor contraction, and [`Self::trace_pairs`] is where
    /// they appear. The two therefore disagree for a fermionic provider, by
    /// design and as in the erased facade.
    ///
    /// # Errors
    ///
    /// [`Error::InvalidArgument`] when the tensor is not an endomorphism, with
    /// the erased facade's own message, and [`Error::Core`] when the block
    /// structure cannot be walked.
    pub fn tr(&self) -> Result<D, Error> {
        let hom = self.body.space.space().homspace();
        // Mirrors the erased pre-check verbatim, message included: the weighted
        // trace below indexes codomain axis `i` together with domain axis
        // `nout + i` and would be meaningless without it.
        if hom.codomain().legs() != hom.domain().legs() {
            return Err(Error::InvalidArgument(
                "tr() requires an endomorphism (domain == codomain)".to_string(),
            ));
        }
        Ok(D::from_complex64(weighted_trace(
            self.body.space.provider(),
            self.body.space.space().structure(),
            self.body.space.space().nout(),
            &self.body.data,
        )?))
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
