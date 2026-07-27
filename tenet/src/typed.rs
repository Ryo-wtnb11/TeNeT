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
//! # Product symmetries
//!
//! A product symmetry needs no new constructor here, and no new type in the
//! engine. A product of providers *is* a provider — build it with
//! [`tenet_core::ProductFusionRuleExt::product`] and label it with
//! [`tenet_core::product_sector`] — so this facade drives `fZ2 ⊠ U(1)`, or any
//! other ordered product of admitted components, through the same
//! [`GradedSpace::try_new`] and [`TensorMap::zeros`] as a single symmetry:
//!
//! ```
//! use std::sync::Arc;
//!
//! use tenet::core::{
//!     product_sector, FermionParityFusionRule, ProductFusionRuleExt, U1FusionRule, U1Irrep,
//!     Z2Irrep,
//! };
//! use tenet::typed::{Error, GradedSpace, Runtime, TensorMap};
//!
//! # fn main() -> Result<(), Error> {
//! let runtime = Runtime::builder().build()?;
//! let rule = Arc::new(FermionParityFusionRule.product(U1FusionRule));
//!
//! let even = product_sector(Z2Irrep::EVEN, U1Irrep::new(0));
//! let odd = product_sector(Z2Irrep::ODD, U1Irrep::new(1));
//! let v = GradedSpace::try_new(Arc::clone(&rule), [(even, 2), (odd, 1)], false)?;
//!
//! let t: TensorMap<_, f64> = TensorMap::zeros(&runtime, [&v], [&v])?;
//! assert_eq!(t.block_count(), 2);
//! assert_eq!(t.block_fusion_trees(0)?.coupled(), &even);
//! # Ok(())
//! # }
//! ```
//!
//! Three or more factors are the same call again, because the product is
//! itself a provider. The spelling
//!
//! ```text
//! FermionParityFusionRule.product(U1FusionRule).product(SU2FusionRule)
//! ```
//!
//! is the left-associated `(fZ2 ⊠ U(1)) ⊠ SU(2)`, whose labels are
//! `product_sector(product_sector(parity, charge), spin)` — the label nests in
//! step with the provider. **Factor order and association are structure, not
//! an equivalence.** `U(1) ⊠ fZ2` and `fZ2 ⊠ U(1)` are both legal, are
//! different Rust types with different [`tenet_core::RuleIdentity`]s, and
//! converting between them is an explicit component swap; nothing here permutes
//! factors for you. Association is where TeNeT diverges from TensorKit rather
//! than mirrors it: TK's `⊠` flattens nested products into one
//! `ProductSector{Tuple{…}}`, so association is unobservable there, while
//! TeNeT keeps the nesting in the Rust type — see
//! [`tenet_core::ProductFusionRule`] for the full statement.
//!
//! The scope of that claim is this facade's current admission:
//! `MultiplicityFreeRigidSymbols<Scalar = f64> + CheckedFusionAlgebra +
//! SectorCodec`. Complex categorical scalars and anyonic admission are issue
//! #539; multiplicity-bearing providers (SU(3) and any other `Generic` rule)
//! need their own admission path, as noted below.
//!
//! **`ProductSector` is not `ProductSpace`.** [`tenet_core::ProductSector`] is
//! a *sector label*: one irrep of a Deligne product category, TensorKit's
//! `ProductSector` (`TensorKitSectors` 0.3.4, `src/product.jl:245-294`).
//! TensorKit's `ProductSpace` is the unrelated leg-level notion — an ordered
//! list of vector spaces forming a tensor's codomain or domain — which this
//! facade spells as the `[&v, &w]` leg slices passed to every constructor, not
//! as a public container type.
//!
//! The erased [`crate::prelude::Space::product`] and
//! [`crate::prelude::Space::fz2_u1_su2`] constructors cover two fixed products
//! for compatibility. They are not the extension mechanism: the route above is.
//!
//! # Phase boundary
//!
//! This is the phase-6 surface of issue #557: construction
//! ([`TensorMap::zeros`], [`TensorMap::from_block_fn`], [`TensorMap::id`]),
//! inspection ([`TensorMap::codomain`], [`TensorMap::domain`],
//! [`TensorMap::block_fusion_trees`], [`TensorMap::block`],
//! [`TensorMap::block_count`], [`TensorMap::data`], [`TensorMap::runtime`]),
//! the index-manipulation and contraction operations
//! ([`TensorMap::permute`], [`TensorMap::braid`], [`TensorMap::transpose`],
//! [`TensorMap::transpose_axes`], [`TensorMap::repartition`],
//! [`TensorMap::contract`], [`TensorMap::compose`]), the scalar operations
//! ([`TensorMap::add`], [`TensorMap::scale`], [`TensorMap::norm`],
//! [`TensorMap::norm_inf`], [`TensorMap::normalize`], [`TensorMap::inner`],
//! [`TensorMap::dot`], [`TensorMap::tr`], [`TensorMap::trace_pairs`],
//! [`TensorMap::adjoint`]), the factorizations ([`TensorMap::svd_compact`],
//! [`TensorMap::svd_full`], [`TensorMap::svd_trunc`], [`TensorMap::svd_vals`],
//! [`TensorMap::qr_compact`], [`TensorMap::qr_full`],
//! [`TensorMap::lq_compact`], [`TensorMap::lq_full`],
//! [`TensorMap::left_orth`], [`TensorMap::right_orth`],
//! [`TensorMap::left_null`], [`TensorMap::right_null`]) and — with issue #570
//! — the **eigendecompositions** ([`TensorMap::eigh_full`],
//! [`TensorMap::eigh_trunc`], [`TensorMap::eigh_vals`],
//! [`TensorMap::eig_full`], [`TensorMap::eig_trunc`], [`TensorMap::eig_vals`])
//! and the **`is_hermitian` / `project_*` family** ([`TensorMap::is_hermitian`],
//! [`TensorMap::is_antihermitian`], [`TensorMap::is_isometric`],
//! [`TensorMap::is_unitary`], [`TensorMap::is_posdef`],
//! [`TensorMap::project_hermitian`], [`TensorMap::project_antihermitian`]) and
//! — with issue #576 — the **matrix functions** ([`TensorMap::exp`],
//! [`TensorMap::inv`], [`TensorMap::pinv`], [`TensorMap::sqrt`]).
//!
//! Issue #570 also gave the facade **compact diagonal storage**: a spectrum
//! factor — `svd_compact`'s and `svd_trunc`'s `s`, `eigh`/`eig`'s `d` — holds
//! `Σ_c k_c` values rather than the `Σ_c k_c²` block-diagonal buffer they would
//! fill, which is what TensorKit's `DiagonalTensorMap` is. It is a storage
//! property and not a type: no signature mentions it, [`TensorMap::data`] still
//! reports the dense buffer (materialized once, on demand, shared by every
//! clone), and the operations that can exploit it — [`TensorMap::compose`],
//! [`TensorMap::scale`], [`TensorMap::add`], [`TensorMap::adjoint`] and the
//! reductions — do so silently. The ones that cannot say so in their own
//! documentation: [`TensorMap::permute`] and its family, and
//! [`TensorMap::contract`].
//!
//! [`TensorMap::compose`] was previously documented here as blocked below this
//! layer, on a public seam sealed by `LoweredMultiplicityFreeAlgebra`. That
//! diagnosis was wrong: the composition path never decoded a typed sector, and
//! the lowered bound was inherited from one inner call in a bosonic
//! short-circuit that already had a non-lowered twin. Swapping that one call
//! opened the seam for every provider, fermionic signs included.
//!
//! What is still absent — among what remains, the entries below are the ones
//! with a decision behind them rather than a queue position. This facade is
//! deliberately narrower than the erased [`crate::prelude::Tensor`], which also
//! carries `left_polar`/`right_polar`, `twist`, `flip`, the unit-leg
//! insert/remove pair, `absorb`, the structural
//! `isomorphism`/`isometry`/`unitary` constructors and `rand`/`rand_with_seed`.
//! Those are ports waiting on nothing but their turn; the ones below are not:
//!
//! - The **rest of the matrix-function family** — the trigonometric and
//!   hyperbolic members, `log`, `sylvester`, the `\` and `/` solves and integer
//!   `^` — is out by decision, not by queue position (issue #576). Every one of
//!   them is a spectral function or a solve over the same seams, so adding them
//!   is mechanical; what is missing is a reason to. The four that landed are
//!   the ones the tensor-network algorithms in this repository actually call.
//!   One capability gap still stands behind that line: general endomorphism
//!   **`sqrt`** needs a Schur seam ([`TensorMap::sqrt`] is the diagonal-bond
//!   idiom only), and that seam does not exist below this facade. The one that
//!   used to stand beside it is closed — [`TensorMap::exp`] accepts any
//!   endomorphism since issue #577, through a blockwise Padé arm. The erased
//!   [`crate::prelude::Tensor::exp`] carried a complexity-parity gap against
//!   this one — it densified a diagonal payload where this facade has an
//!   O(rank) arm — until issue #578 gave it the same arm.
//! - **Outer multiplicity** (SU(3) and any other `Generic` provider) is out at
//!   the admission boundary, not at this layer: every constructor here consumes
//!   the multiplicity-free checked root. A `Generic` provider needs its own
//!   admission path before any of this surface can accept one.
//! - **Device placement** is absent for the same structural reason: the payload
//!   is a `Vec<D>` host buffer by construction, and there is no dtype or
//!   placement token to reconcile because `D` is a type parameter. Adding a
//!   device would change what the body holds, not what a method promises.
//! - The **operator overloads** (`impl Add`, `impl Mul`) are out on the
//!   `Result` argument alone. An operator cannot return one: the erased
//!   `Mul` precedent panics, and a panicking `*` or `+` as the only spelling
//!   of an operation contradicts this facade's passthrough-error contract. The
//!   cross-facade false-friend argument that used to stand beside it expired
//!   with [`TensorMap::compose`]: `&a * &b` means composition in the erased
//!   facade, and composition is what this facade would spell it as. Adding
//!   them later is not a breaking change.
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
    BoundDynamicFusionMapSpace, BoundDynamicTensorRef, DynamicFusionMapSpace, OutputAxisOrder,
    TreeTransformOperation,
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
pub use tenet_matrixalgebra::{Truncation, TruncationSpace};

use tenet_matrixalgebra::{BoundDynFactor, FactorScalar};

use crate::tensor::{
    apply_fill, coupled_region_pow_sum, sector_regions, validate_norm_p, weighted_inner,
    weighted_trace, with_planar_axes, Fill, PlanarRequestKind, TensorScalar,
};
use crate::typed_tensor_core::{
    tensorcompose_owned_multiplicity_free, tensorcontract_owned_multiplicity_free,
    tree_transform_owned_multiplicity_free,
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

    /// This leg read as a fixed truncation target: TensorKit `truncspace(V)`
    /// (`src/factorizations/truncation.jl:261-269`).
    ///
    /// The resulting [`Truncation::space`] policy keeps exactly this leg's
    /// degeneracy for every coupled sector it carries, and drops any sector it
    /// does not — TensorKit reads the same `dim(V, c)`, which is zero for an
    /// absent sector. A request longer than the spectrum offers is clamped to
    /// it. The dual flag is ignored: a truncation target names sector content,
    /// not orientation.
    ///
    /// The profile records this leg's provider identity, so handing it to a
    /// factorization of a tensor built on a different rule is a typed error
    /// rather than a silent truncation to nothing.
    ///
    /// # Complexity
    ///
    /// `O(k log k)` in the number of sectors (one `BTreeMap` build); no
    /// spectrum or payload is touched.
    pub fn truncspace(&self) -> TruncationSpace {
        TruncationSpace::new(
            self.provider.rule_identity(),
            self.leg
                .sectors()
                .iter()
                .copied()
                .zip(self.leg.degeneracies().iter().copied()),
        )
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
///
/// `V` is the value type and defaults to [`f64`], the singular/Hermitian-
/// eigenvalue case, so `SectorSpectrum<S>` keeps its meaning. The general
/// eigendecompositions spell it `SectorSpectrum<S, Complex64>`. The default is
/// what [`tenet_matrixalgebra::SectorSpectrum`] already does, for the same
/// reason: a real spectrum is by far the common one, and a caller who never
/// touches `eig_*` should never have to name the parameter.
#[derive(Clone, Debug, PartialEq)]
pub struct SectorSpectrum<S, V = f64> {
    /// The coupled sector, in the provider's own labels.
    pub sector: S,
    /// That sector's values, descending by magnitude.
    pub values: Vec<V>,
}

/// Result of [`TensorMap::svd_trunc`]: `t ~ u * s * vh` with the truncated
/// bond (TensorKit 0.17 `svd_trunc`, which returns `(U, S, Vᴴ, ϵ)`).
// The `SectorCodec` bound is the field types' own: `singular_values` is
// labelled, so the struct cannot be spelled without it.
pub struct SvdTrunc<R: SectorCodec, D> {
    /// Left isometry `u : codomain <- bond`.
    pub u: TensorMap<R, D>,
    /// Singular-value factor `s : bond <- bond`, in compact diagonal storage
    /// (TensorKit's `DiagonalTensorMap`); see [`TensorMap::svd_compact`].
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

/// The two block payload representations one typed tensor map can carry.
///
/// The erased [`crate::tensor::Data`] needs three diagonal variants to record a
/// spectrum's dtype and whether it must widen on materialization; here `D` is a
/// type parameter, so the whole question collapses to one arm holding values of
/// exactly the payload type.
enum TypedData<D> {
    /// The dense coupled-sector buffer every operation can read.
    Dense(Vec<D>),
    /// Compact O(Σ_c k_c) storage for a spectrum factor (SVD `s`, `eigh`/`eig`
    /// `d`): only the per-sector diagonal values, keyed by the engine's raw
    /// [`tenet_core::SectorId`] — a stored payload never leaves this module, so
    /// there is nothing here for the codec to label.
    Diagonal(Vec<tenet_matrixalgebra::SectorSpectrum<D>>),
}

/// Applies a scalar function to every stored value of a compact spectrum,
/// leaving the sector keys and the per-sector lengths untouched.
///
/// This is the whole of the O(rank) arm shared by [`TensorMap::exp`],
/// [`TensorMap::inv`], [`TensorMap::pinv`] and [`TensorMap::sqrt`]: a spectral
/// function acts on eigenvalues, so it never moves weight between sectors and
/// never changes a bond dimension, which is exactly why the result can stay on
/// the space it was called on.
fn map_spectrum<D: Copy>(
    spectrum: &[tenet_matrixalgebra::SectorSpectrum<D>],
    mut value_of: impl FnMut(D) -> Result<D, Error>,
) -> Result<Vec<tenet_matrixalgebra::SectorSpectrum<D>>, Error> {
    spectrum
        .iter()
        .map(|entry| {
            Ok(tenet_matrixalgebra::SectorSpectrum {
                sector: entry.sector,
                values: entry
                    .values
                    .iter()
                    .map(|&value| value_of(value))
                    .collect::<Result<_, Error>>()?,
            })
        })
        .collect()
}

/// Two compact spectra that live on one bond space must agree sector for
/// sector and length for length; when they do not, the space and the payload
/// have gone out of step, which is an engine invariant break rather than
/// anything a caller did.
fn spectra_disagree() -> Error {
    Error::InvalidArgument("equal bond spaces carry incompatible compact spectra".to_string())
}

/// Result of [`TensorMap::eig_trunc`]: `t ~ v * d * v^-1` with the eigenbasis
/// truncated (MatrixAlgebraKit `eig_trunc`).
///
/// The factors are `D::Eig`-payloaded, not `D`-payloaded: a real matrix has
/// complex eigenpairs, so both are complex for either input dtype — TensorKit's
/// `eigen`, whose `D` and `V` are `ComplexF64` even for a real argument.
// `D: TensorScalar` rather than a bare parameter because the field types are
// spelled through `D::Eig`, which is `FactorScalar`'s associated type.
pub struct EigTrunc<R: SectorCodec, D: TensorScalar> {
    /// Eigenvalue factor `d : bond <- bond`, in compact diagonal storage.
    pub d: TensorMap<R, <D as FactorScalar>::Eig>,
    /// Eigenbasis `v : codomain <- bond`.
    pub v: TensorMap<R, <D as FactorScalar>::Eig>,
    /// Kept eigenvalues per coupled sector, sorted by provider label.
    pub eigenvalues: Vec<SectorSpectrum<R::Sector, num_complex::Complex64>>,
    /// Quantum-dimension-weighted 2-norm of the discarded `|eigenvalue|`s.
    pub error: f64,
}

// Hand-written for the reason [`SvdTrunc`]'s are.
impl<R, D> Clone for EigTrunc<R, D>
where
    R: SectorCodec,
    D: TensorScalar,
{
    fn clone(&self) -> Self {
        Self {
            d: self.d.clone(),
            v: self.v.clone(),
            eigenvalues: self.eigenvalues.clone(),
            error: self.error,
        }
    }
}

impl<R, D> core::fmt::Debug for EigTrunc<R, D>
where
    R: SectorCodec,
    D: TensorScalar,
{
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        formatter
            .debug_struct("EigTrunc")
            .field("d", &self.d)
            .field("v", &self.v)
            .field("eigenvalues", &self.eigenvalues)
            .field("error", &self.error)
            .finish()
    }
}

/// [`TensorMap::diagonal_factor`]'s body, as a free function so the
/// `eig_*` family can build a `TensorMap<R, D::Eig>` from a `TensorMap<R, D>`.
/// The payload type of a factor need not be the payload type of the tensor it
/// came from, and an inherent method cannot say that.
fn diagonal_factor_on<R, E, V>(
    runtime: &Runtime,
    authority: &BoundDynamicFusionMapSpace<R>,
    mut spectrum: Vec<tenet_matrixalgebra::SectorSpectrum<V>>,
    to_scalar: impl Fn(V) -> E,
) -> Result<TensorMap<R, E>, Error>
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64>,
{
    spectrum.sort_unstable_by_key(|entry| entry.sector);
    let space = tenet_matrixalgebra::diagonal_bond_bound_space_like(authority, &spectrum)?;
    let data = spectrum
        .into_iter()
        .map(|entry| tenet_matrixalgebra::SectorSpectrum {
            sector: entry.sector,
            values: entry.values.into_iter().map(&to_scalar).collect(),
        })
        .collect();
    Ok(TensorMap {
        runtime: runtime.clone(),
        body: Arc::new(TypedTensorBody {
            space,
            data: TypedData::Diagonal(data),
            dense_cache: std::sync::OnceLock::new(),
        }),
    })
}

/// [`TensorMap::wrap_bound_factor`]'s body, free for the same reason as
/// [`diagonal_factor_on`].
fn wrap_factor_on<R, E>(runtime: &Runtime, factor: BoundDynFactor<R, E>) -> TensorMap<R, E>
where
    R: tenet_core::FusionRule,
{
    let (space, data) = factor.into_parts();
    TensorMap {
        runtime: runtime.clone(),
        body: Arc::new(TypedTensorBody::dense(space, data)),
    }
}

/// `dense_factor * dense + diagonal_factor * spectrum`, laid out per `space`.
///
/// The dense operand is scaled into a fresh owned buffer and the spectrum is
/// added onto that buffer's per-block diagonal, which is the only place a bond
/// space is non-zero. Same block addressing as
/// [`tenet_matrixalgebra::diagonal_bond_data`], which is what put the values
/// there in the first place.
fn scatter_spectrum<D>(
    space: &DynamicFusionMapSpace,
    dense: &[D],
    dense_factor: D,
    spectrum: &[tenet_matrixalgebra::SectorSpectrum<D>],
    diagonal_factor: D,
) -> Result<Vec<D>, Error>
where
    D: TensorScalar,
{
    let mut data: Vec<D> = dense.iter().map(|&value| value * dense_factor).collect();
    let structure = space.structure();
    for index in 0..structure.block_count() {
        let block = structure.block(index)?;
        let Some(pair) = block.key().as_fusion_tree_pair() else {
            continue;
        };
        let sector = pair.codomain_tree().coupled();
        // O(k) per block, so O(k²) over the walk. Fine at the sizes a bond
        // space reaches; index it if a spectrum ever spans many sectors.
        let Some(entry) = spectrum.iter().find(|entry| entry.sector == sector) else {
            // Both operands live on one space, and a compact payload's space is
            // built from its own spectrum, so every block's coupled sector has
            // an entry. Skipping is the safe behaviour if that ever breaks —
            // the block keeps the dense operand's contribution — but it is a
            // silent wrong answer, so say so loudly in a debug build.
            debug_assert!(
                false,
                "no spectrum entry for coupled sector {sector:?} on its own bond space"
            );
            continue;
        };
        let strides = block.strides();
        let stride = strides[0] + strides[1];
        let offset = block.offset();
        // The three lengths agree by construction, for the same reason.
        debug_assert_eq!(block.shape()[0], block.shape()[1]);
        debug_assert_eq!(block.shape()[0], entry.values.len());
        let count = block.shape()[0]
            .min(block.shape()[1])
            .min(entry.values.len());
        for (i, &value) in entry.values[..count].iter().enumerate() {
            let position = offset + i * stride;
            data[position] = data[position] + value * diagonal_factor;
        }
    }
    Ok(data)
}

/// Whether `space` is a bond space: rank one on each side, with the same leg on
/// both — the shape a compact spectrum can address, and the only shape whose
/// dense form is block-diagonal.
///
/// Verbatim from the erased `Tensor::is_diagonal_bond_space`, deliberately: it
/// is the guard TensorKit's `DiagonalTensorMap` gets for free from its type, and
/// two facades disagreeing about which destinations may stay compact would be a
/// silent divergence rather than a visible one.
///
/// Applied either to the *destination* of an operation — an operand's storage
/// says what it holds, only the destination says whether a compact result is
/// representable — or, in [`TensorMap::sqrt`], to the receiver, because there
/// the bond shape is the operation's own domain restriction rather than a
/// storage question.
///
/// # Reachability
///
/// Only [`TensorMap::sqrt`] can make this answer `false`, and does: a general
/// tensor is a legal argument to write and an illegal one to accept, so the
/// guard is killable there. At the compact-*destination* call sites it still
/// cannot fail — every [`TypedData::Diagonal`] payload this module can produce
/// sits on a space built by [`diagonal_factor_on`], i.e. by
/// [`tenet_matrixalgebra::diagonal_bond_bound_space_like`], which is a bond
/// space by construction, and the operations that preserve the payload
/// ([`TensorMap::scale`], [`TensorMap::add`], [`TensorMap::adjoint`], the
/// `D * D` arm) all keep that space. It stays at those sites because the next
/// constructor of a compact payload — a diagonal-aware `contract`, say — would
/// be the first one able to aim at a destination that is not a bond space, and
/// should find the
/// check already in place rather than have to notice it is missing.
fn is_diagonal_bond_space(space: &DynamicFusionMapSpace) -> bool {
    let homspace = space.homspace();
    space.nout() == 1 && space.nin() == 1 && homspace.codomain().legs() == homspace.domain().legs()
}

/// Result of [`TensorMap::eigh_trunc`]: `t ~ v * d * v^H` with the eigenbasis
/// truncated (MatrixAlgebraKit `eigh_trunc`).
///
/// Field order is `d` then `v`, matching [`TensorMap::eigh_full`]'s tuple and
/// MatrixAlgebraKit's own `initialize_output`, so the two cannot be read the
/// wrong way round against each other.
// The `SectorCodec` bound is the field types' own, exactly as for [`SvdTrunc`].
pub struct EighTrunc<R: SectorCodec, D> {
    /// Eigenvalue factor `d : bond <- bond`, in compact diagonal storage.
    pub d: TensorMap<R, D>,
    /// Eigenvector isometry `v : codomain <- bond`.
    pub v: TensorMap<R, D>,
    /// Kept eigenvalues per coupled sector, sorted by provider label. Real for
    /// both payload dtypes, as TensorKit's Hermitian `D` is.
    pub eigenvalues: Vec<SectorSpectrum<R::Sector>>,
    /// Quantum-dimension-weighted 2-norm of everything discarded.
    pub error: f64,
}

// Hand-written for the reason [`SvdTrunc`]'s are.
impl<R, D> Clone for EighTrunc<R, D>
where
    R: SectorCodec,
{
    fn clone(&self) -> Self {
        Self {
            d: self.d.clone(),
            v: self.v.clone(),
            eigenvalues: self.eigenvalues.clone(),
            error: self.error,
        }
    }
}

impl<R, D> core::fmt::Debug for EighTrunc<R, D>
where
    R: SectorCodec,
{
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        formatter
            .debug_struct("EighTrunc")
            .field("d", &self.d)
            .field("v", &self.v)
            .field("eigenvalues", &self.eigenvalues)
            .field("error", &self.error)
            .finish()
    }
}

/// Storage shared by every clone of one typed tensor map: the admitted space
/// and its block payload.
struct TypedTensorBody<R, D> {
    space: BoundDynamicFusionMapSpace<R>,
    data: TypedData<D>,
    /// Materialization of a [`TypedData::Diagonal`] payload into the dense
    /// coupled layout, computed at most once and shared by every clone of this
    /// body — the erased sibling's `compact_dense` cache, without its hand-copy
    /// on each `Tensor` value. Never populated for a dense payload.
    dense_cache: std::sync::OnceLock<Vec<D>>,
}

impl<R, D> TypedTensorBody<R, D> {
    /// A body holding an already-dense payload.
    fn dense(space: BoundDynamicFusionMapSpace<R>, data: Vec<D>) -> Self {
        Self {
            space,
            data: TypedData::Dense(data),
            dense_cache: std::sync::OnceLock::new(),
        }
    }
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
            // Storage-shaped, deliberately: a compact spectrum payload reports
            // the values it stores, and forcing its dense materialization for a
            // `{:?}` would make a diagnostic the most expensive call on the type.
            .field(
                "elements",
                &match &self.body.data {
                    TypedData::Dense(data) => data.len(),
                    TypedData::Diagonal(spectrum) => {
                        spectrum.iter().map(|entry| entry.values.len()).sum()
                    }
                },
            )
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
            body: Arc::new(TypedTensorBody::dense(space, data)),
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

    /// The identity endomorphism on `spaces <- spaces` (TensorKit `id(V)`):
    /// every coupled-sector block is the identity matrix.
    ///
    /// TensorKit's `one`/`id` for the same object. The erased
    /// [`crate::prelude::Tensor::id`] takes a [`crate::prelude::Dtype`] token;
    /// here the payload dtype is `D`, so there is nothing to pass — otherwise
    /// the argument shape is the erased one, a single leg list used for both
    /// sides.
    ///
    /// Square by construction: the codomain *is* the domain, so the
    /// isomorphism precondition the erased structural constructors check
    /// (`isomorphism`, `isometry`) holds trivially and is not re-checked. The
    /// legs may still be heterogeneous — different sector content and
    /// different degeneracies per leg — since only the fused content matters
    /// and it is identical on both sides by definition.
    ///
    /// # Errors
    ///
    /// Everything [`Self::zeros`] reports, plus [`Error::Core`] when the
    /// admitted layout is not the canonical coupled-sector matrix one, which
    /// is an engine invariant rather than a caller mistake.
    #[doc(alias = "one")]
    pub fn id<'a, S>(runtime: &Runtime, spaces: S) -> Result<Self, Error>
    where
        S: IntoIterator<Item = &'a GradedSpace<R>>,
        R: 'a,
    {
        let spaces: Vec<&GradedSpace<R>> = spaces.into_iter().collect();
        let provider = Arc::clone(Self::authority(&spaces)?);
        let mut identity = Self::build(runtime, provider, &spaces, &spaces, Fill::Zeros)?;
        // The zero fill is written into, not replaced: `build` has just
        // allocated the payload and nothing else holds the body yet, so the
        // diagonal goes in place rather than into a second buffer.
        let body =
            Arc::get_mut(&mut identity.body).expect("a freshly built body has no other owner");
        // Same coupled-sector region walk and same diagonal addressing as the
        // erased `Tensor::structural`, on the shared helper: the two build the
        // same tensor, and a second copy of the offset arithmetic would be free
        // to drift from the sibling this is byte-compared against.
        let regions = {
            let space = body.space.space();
            sector_regions(space.structure(), space.nout())?
        };
        let TypedData::Dense(data) = &mut body.data else {
            unreachable!("`build` always produces a dense payload");
        };
        for region in regions.iter() {
            for i in 0..region.rows().min(region.cols()) {
                data[region.range().start + i * (region.rows() + 1)] = D::from_real(1.0);
            }
        }
        Ok(identity)
    }

    /// TensorKit `permute`: re-arranges legs with symmetric braiding.
    ///
    /// `codomain_axes` and `domain_axes` list source axis numbers (`0..rank`,
    /// codomain axes first) for the new codomain and domain — the same
    /// argument shape as the erased [`crate::prelude::Tensor::permute`], so
    /// there is one vocabulary for the operation rather than two.
    ///
    /// # Compact storage
    ///
    /// A factor in compact diagonal storage ([`Self::svd_compact`]'s `s`,
    /// [`Self::eigh_full`]'s `d`) is **materialized** here, and so by
    /// [`Self::braid`], [`Self::transpose`], [`Self::transpose_axes`] and
    /// [`Self::repartition`] as well: the result is a dense `Σ_c k_c²` buffer.
    /// TensorKit draws the line in the same place — its `DiagonalTensorMap`
    /// implements only the two permutations that leave a diagonal diagonal —
    /// and the general case genuinely is not diagonal, so this is a missing
    /// specialization for two axis orders rather than a missing operation.
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
        // Compact arm: a rank-(1,1) leg swap of a spectrum factor is a
        // per-sector rescaling of the stored diagonal, so it never touches the
        // `Σ_c k_c²` materialization. Every other geometry — an explicit braid,
        // any higher rank, the identity partitions `repartition` produces —
        // falls through to the dense route below; the shared guard documents
        // why. TensorKit 0.17 `src/tensors/diagonal.jl:215-242` makes the same
        // split.
        if let Some(spectrum) = self.spectrum() {
            if crate::typed_tensor_core::is_rank_one_diagonal_swap(
                self.codomain_rank(),
                self.rank() - self.codomain_rank(),
                &operation,
            ) {
                let destination = self.body.space.transformed_multiplicity_free(&operation)?;
                let transformed = crate::typed_tensor_core::transform_rank_one_diagonal_spectrum(
                    self.body.space.provider(),
                    self.body.space.space(),
                    destination.space(),
                    &operation,
                    spectrum,
                )?;
                return Ok(self.with_spectrum_on(destination, transformed));
            }
        }
        // Leasing rather than locking, matching the erased path: independent
        // operations on one runtime must not serialize behind each other.
        let mut lease = self.runtime.lease_context()?;
        let (space, data) = tree_transform_owned_multiplicity_free(
            lease.context().multiplicity_free_lane::<D>(),
            BoundDynamicTensorRef::try_new(&self.body.space, self.dense_data())?,
            operation,
        )?;
        Ok(Self {
            runtime: self.runtime.clone(),
            body: Arc::new(TypedTensorBody::dense(space, data)),
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
    /// rules are unaffected; fermionic rules can differ by signs.
    /// [`Self::compose`] is the other semantics, and its documentation states
    /// the exact relation between the two.
    ///
    /// # Compact fast paths
    ///
    /// Contracting **one** leg against a factor in compact diagonal storage —
    /// an `s` from [`Self::svd_compact`], a `d` from [`Self::eigh_full`] — is
    /// a per-leg bond scaling and is run as one: the other operand's contracted
    /// leg is multiplied by the spectrum in place of a GEMM, and the result is
    /// laid out with a single [`Self::permute`]. Mathematically it is the same
    /// tensor the dense route computes, so this is a cost question only, and any
    /// pattern that does not fit falls through to the dense route rather than
    /// being refused.
    ///
    /// **Complexity.** In `docs/complexity_parity_policy.md`'s parameters — `d`
    /// the per-sector bond degeneracy, `n` the other operand's *open*-leg size,
    /// so its blocks hold `d·n` entries — the dense route materializes the
    /// spectrum as a `Σ_c d_c²` block-diagonal buffer and multiplies it in, at
    /// O(d²) storage and O(d²·n) work. The scaling route touches each of those
    /// `d·n` entries once, at O(d) storage and O(d·n) work, which is the order
    /// that policy's row requires. `D · D` multiplies the two spectra
    /// elementwise and stays compact, at O(d).
    ///
    /// **TensorKit correspondence.** This is what TensorKit's
    /// `DiagonalTensorMap` gets from its type: `block(D, c)` is a `Diagonal`, so
    /// LinearAlgebra dispatches the multiplication to `lmul!`/`rmul!` scaling
    /// (`diagonal.jl`), with no braiding or recoupling of its own.
    ///
    /// **Which patterns.** Exactly the two geometries that are a composition on
    /// the contracted leg, in either order — the contracted leg of the compact
    /// operand is its bond, and the other operand's is a leg on the side that
    /// faces it (`t`'s domain against `D`'s codomain, or `D`'s domain against a
    /// codomain leg of `t`, at any position). A leg on the far side, more than
    /// one contracted leg, or an output order that would move the surviving
    /// bond of a `D · D` product across the codomain/domain split all take the
    /// dense route: the first two are not proved geometries, and the last is not
    /// equivalent to rebinding the product spectrum (checked in #453). A
    /// supertrace twist on a dual contracted leg of `other` would also decline,
    /// and cannot currently arise — see `try_contract_diagonal`.
    ///
    /// The erased [`crate::prelude::Tensor::contract`] took the same arm in #75,
    /// and the two facades are byte-compared across it.
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
        if let Some(compact) = self.try_contract_diagonal(other, lhs_axes, rhs_axes, output_axes)? {
            return Ok(compact);
        }
        let mut lease = self.runtime.lease_context()?;
        let (space, data) = tensorcontract_owned_multiplicity_free(
            lease.context().multiplicity_free_lane::<D>(),
            BoundDynamicTensorRef::try_new(&self.body.space, self.dense_data())?,
            BoundDynamicTensorRef::try_new(&other.body.space, other.dense_data())?,
            lhs_axes,
            rhs_axes,
            // Why `OutputAxisOrder` stays out of the signature: it is an
            // expert-layer borrow type, and a `&[usize]` says the same thing
            // at the facade without a second public vocabulary.
            OutputAxisOrder::from_axes(output_axes),
        )?;
        Ok(Self {
            runtime: self.runtime.clone(),
            body: Arc::new(TypedTensorBody::dense(space, data)),
        })
    }

    /// Categorical composition of two tensor maps, TensorKit `A * B` / `mul!`:
    /// `self`'s whole domain is contracted against `other`'s whole codomain,
    /// leaving `self.codomain() <- other.domain()`.
    ///
    /// **Fermionic semantics**: unlike [`Self::contract`] (TensorKit
    /// `tensorcontract!` / `@tensor`), composition never twists dual
    /// contracted legs — there is no supertrace here. Bosonic rules cannot
    /// tell the two apart; a fermionic one differs by a sign on every dual
    /// contracted leg carrying an odd sector, so the exact relation is
    /// `self.compose(other) == self.contract(twist(other, other's dual
    /// codomain legs), ..)`. Reach for `compose` when you mean operator
    /// multiplication of tensor maps, and for `contract` when you mean
    /// index-notation contraction.
    ///
    /// The axes are not arguments, deliberately: composition is defined by the
    /// codomain/domain split itself, and both TensorKit's `*` and the erased
    /// [`crate::prelude::Tensor::compose`] take none.
    ///
    /// The result is bound to `self`'s provider allocation — the same
    /// left-authority rule as [`Self::contract`] and [`Self::zeros`] — with one
    /// exemption: the `D * t` compact arm below returns `t`'s own space and
    /// runtime handle, because that space *is* the destination and rebuilding
    /// it under the left allocation would be a copy for nothing. The two
    /// allocations must already agree on
    /// [`tenet_core::FusionRule::rule_identity`] for the composition to be
    /// legal at all, so the choice is immaterial to the algebra. The erased
    /// [`crate::prelude::Tensor::compose`] takes the same exemption on the same
    /// arm, and the two facades are byte-compared across it.
    ///
    /// # Compact fast paths
    ///
    /// When either operand carries compact diagonal storage — an `s` from
    /// [`Self::svd_compact`], a `d` from [`Self::eigh_full`] — and the
    /// destination is representable, this takes TensorKit's
    /// `DiagonalTensorMap` route instead of a GEMM: `t * D` and `D * t` scale
    /// one bond axis per block (`rmul!` / `lmul!`), and `D * D` multiplies the
    /// two spectra elementwise and stays compact. Verified twist-free against
    /// TK's `diagonal.jl`: `block(D, c)` is a `Diagonal`, so LinearAlgebra
    /// dispatches to scaling, with no braiding or recoupling. The result is the
    /// same tensor the dense route computes, so this is a cost question only,
    /// and any operand or destination that does not fit falls through to the
    /// dense path rather than being refused.
    ///
    /// # Errors
    ///
    /// - [`Error::RuntimeMismatch`] when the operands belong to different
    ///   runtimes, as for [`Self::contract`].
    /// - [`Error::Operation`] / [`Error::Core`] / [`Error::FusionAlgebra`]
    ///   when the two are not composable — mismatched ranks, legs that are not
    ///   mutually dual, or providers reporting different rule identities.
    ///   Those come back from the expert layer, which owns the rules.
    #[doc(alias = "mul")]
    pub fn compose(&self, other: &Self) -> Result<Self, Error> {
        // Runtime first, exactly as `contract`: crossing runtimes is a
        // trust-boundary violation rather than an algebra error, and the
        // expert layer never sees the two runtimes.
        if !self.runtime.same_runtime(&other.runtime) {
            return Err(Error::RuntimeMismatch);
        }
        if let Some(compact) = self.compose_compact(other)? {
            return Ok(compact);
        }
        let lhs_axes: Vec<usize> = (self.codomain_rank()..self.rank()).collect();
        let rhs_axes: Vec<usize> = (0..other.codomain_rank()).collect();
        let mut lease = self.runtime.lease_context()?;
        let (space, data) = tensorcompose_owned_multiplicity_free(
            lease.context().multiplicity_free_lane::<D>(),
            BoundDynamicTensorRef::try_new(&self.body.space, self.dense_data())?,
            BoundDynamicTensorRef::try_new(&other.body.space, other.dense_data())?,
            &lhs_axes,
            &rhs_axes,
        )?;
        Ok(Self {
            runtime: self.runtime.clone(),
            body: Arc::new(TypedTensorBody::dense(space, data)),
        })
    }

    /// The compact arms of [`Self::compose`], or `None` when the operands or
    /// the destination cannot support one and the dense route must run.
    ///
    /// Each arm proves its destination rather than deriving it. Composition
    /// glues `self.codomain <- other.domain`, so:
    ///
    /// - `D * D` — both operands are bond spaces (`codomain == domain`), so
    ///   when the two spaces are equal the destination *is* that space, and
    ///   [`is_diagonal_bond_space`] certifies it can hold a compact result.
    /// - `t * D` — the destination is `t.codomain <- D.domain`; `D` is a bond
    ///   space, so `D.domain == D.codomain`, and requiring that to equal
    ///   `t.domain` makes the destination `t`'s own space. `t`'s payload is
    ///   then `t`'s data with each block's trailing axis scaled.
    /// - `D * t` — the mirror image, scaling `t`'s leading axis.
    ///
    /// Without those equalities the destination is a different space (a dual
    /// leg on the contracted side is the reachable case) and reusing an
    /// operand's would silently produce a tensor on the wrong space, so the
    /// arm declines and the expert layer decides — including by rejecting a
    /// composition that is not one at all.
    fn compose_compact(&self, other: &Self) -> Result<Option<Self>, Error> {
        if !self.same_rule(other) {
            return Ok(None);
        }
        let (left, right) = (self.body.space.space(), other.body.space.space());
        match (self.spectrum(), other.spectrum()) {
            (Some(lhs), Some(rhs)) => {
                // Both clauses are unreachable today and stay for the reason
                // [`is_diagonal_bond_space`] gives. `left != right` is the
                // weaker one: two compact payloads on unequal bond spaces
                // necessarily carry spectra that differ in their sectors or
                // their lengths, so the elementwise product below would refuse
                // them anyway — just with `spectra_disagree`'s message instead
                // of the expert layer's. Removing it would change which error a
                // caller sees, not whether one is reported.
                if left != right || !is_diagonal_bond_space(left) {
                    return Ok(None);
                }
                if lhs.len() != rhs.len() {
                    return Err(spectra_disagree());
                }
                let product = lhs
                    .iter()
                    .zip(rhs)
                    .map(|(left, right)| {
                        if left.sector != right.sector || left.values.len() != right.values.len() {
                            return Err(spectra_disagree());
                        }
                        Ok(tenet_matrixalgebra::SectorSpectrum {
                            sector: left.sector,
                            values: left
                                .values
                                .iter()
                                .zip(&right.values)
                                .map(|(&a, &b)| a * b)
                                .collect(),
                        })
                    })
                    .collect::<Result<Vec<_>, Error>>()?;
                Ok(Some(self.with_spectrum(product)))
            }
            // `t * D`: scale each block's trailing axis (TensorKit `rmul!`).
            (None, Some(spectrum)) => {
                if !is_diagonal_bond_space(right)
                    || left.homspace().domain().legs() != right.homspace().codomain().legs()
                {
                    return Ok(None);
                }
                self.scaled_axis(None, spectrum).map(Some)
            }
            // `D * t`: scale each block's leading axis (TensorKit `lmul!`).
            (Some(spectrum), None) => {
                if !is_diagonal_bond_space(left)
                    || right.homspace().codomain().legs() != left.homspace().domain().legs()
                {
                    return Ok(None);
                }
                other.scaled_axis(Some(0), spectrum).map(Some)
            }
            (None, None) => Ok(None),
        }
    }

    /// The compact arm of [`Self::contract`] (issue #584), or `None` when the
    /// operands, the axis pattern or the output order do not fit one and the
    /// dense route must run.
    ///
    /// A one-axis contraction against a compact operand is a bond scaling, so
    /// the spectrum multiplies the *other* operand's contracted leg — O(d·n)
    /// against the dense route's O(d²·n) GEMM on a materialized `Σ_c d_c²`
    /// buffer — and one [`Self::permute`] lays the result out. The permute is
    /// what carries every recoupling and bend, so this adds no mathematics of
    /// its own; it is the same scale-plus-one-permute structure the erased
    /// `Tensor::try_contract_diagonal_fast_path` runs, and the two are
    /// byte-compared in `tests/typed_facade.rs`.
    ///
    /// # Which patterns, and why only those
    ///
    /// The engine admits a contracted pair only when the two legs agree on
    /// their raw duality flag on the compose-shaped pairing (one operand's
    /// domain leg against the other's codomain leg), and a compact operand's
    /// leg *is* its bond on both sides. So each arm requires exactly that
    /// pairing and compares the two legs itself: raw equality is the engine's
    /// admissibility condition here, so a mismatch is a contraction the dense
    /// route must reject rather than one this arm may answer, and the arm
    /// declines so the expert layer reports it in its own words. `D · D` is
    /// handed to [`Self::compose_compact`], which is the same product and
    /// already proves its destination.
    ///
    /// # The twist, and why it is not folded
    ///
    /// [`Self::contract`] applies the fermionic supertrace twist to a **dual**
    /// contracted leg of `other`, where [`Self::compose`] does not. The erased
    /// sibling folds that `θ = ±1` into the spectrum values. Here the case
    /// cannot arise, so the arm declines instead of carrying arithmetic no test
    /// could reach: a compact payload's bond leg is built non-dual
    /// (`diagonal_bond_bound_space_like`), the arms pair it with a *codomain*
    /// leg of `other` whose external duality is exactly its raw flag, and
    /// admissibility forces that flag to equal the bond's. The guard stays
    /// because the first constructor of a compact payload on a dual bond leg —
    /// or of an arm pairing a domain leg of `other` — should find a decline
    /// rather than a silent sign error, and the erased fold is what to port
    /// then.
    fn try_contract_diagonal(
        &self,
        other: &Self,
        lhs_axes: &[usize],
        rhs_axes: &[usize],
        output_axes: &[usize],
    ) -> Result<Option<Self>, Error> {
        if lhs_axes.len() != 1 || rhs_axes.len() != 1 || !self.same_rule(other) {
            return Ok(None);
        }
        let (lhs_axis, rhs_axis) = (lhs_axes[0], rhs_axes[0]);
        if lhs_axis >= self.rank() || rhs_axis >= other.rank() {
            return Ok(None);
        }
        // Why the provider rather than a stored flag: `braiding_style` is the
        // rule's own answer, and `R` is concrete here.
        let fermionic =
            self.body.space.provider().braiding_style() == tenet_core::BraidingStyleKind::Fermionic;
        if fermionic
            && other
                .body
                .space
                .space()
                .homspace()
                .external_axis_is_dual(rhs_axis)
                != Some(false)
        {
            return Ok(None);
        }
        let (left, right) = (self.body.space.space(), other.body.space.space());
        let (left_home, right_home) = (left.homspace(), right.homspace());
        match (self.spectrum(), other.spectrum()) {
            // `D · D`: the same product as `D * D`, which already knows how to
            // stay compact and which destinations may hold the result.
            (Some(_), Some(_)) => {
                if lhs_axis != 1 || rhs_axis != 0 || output_axes.iter().copied().ne(0..2) {
                    // Why not a reordered output: `pAB` can move the surviving
                    // bond across the codomain/domain split, and rebinding the
                    // product spectrum there is not equivalent to a permute
                    // (#453).
                    return Ok(None);
                }
                self.compose_compact(other)
            }
            // `t · D` (TensorKit `rmul!`): scale `t`'s contracted domain leg,
            // then move it to where the contraction's output order wants it.
            (None, Some(spectrum)) => {
                if rhs_axis != 0 || lhs_axis < self.codomain_rank() {
                    return Ok(None);
                }
                if left_home.domain().legs()[lhs_axis - self.codomain_rank()]
                    != right_home.codomain().legs()[0]
                {
                    return Ok(None);
                }
                let mut source: Vec<usize> = (0..self.rank()).filter(|&a| a != lhs_axis).collect();
                source.push(lhs_axis);
                self.scaled_axis(Some(lhs_axis), spectrum)?
                    .permuted_to_output(&source, output_axes, self.rank() - 1)
            }
            // `D · t` (TensorKit `lmul!`): the mirror image, scaling the
            // contracted codomain leg of `t` at whatever position it sits.
            (Some(spectrum), None) => {
                if lhs_axis != 1 || rhs_axis >= other.codomain_rank() {
                    return Ok(None);
                }
                if left_home.domain().legs()[0] != right_home.codomain().legs()[rhs_axis] {
                    return Ok(None);
                }
                let mut source = vec![rhs_axis];
                source.extend((0..other.rank()).filter(|&a| a != rhs_axis));
                other
                    .scaled_axis(Some(rhs_axis), spectrum)?
                    // One open axis of `self` survives, so the destination's
                    // codomain rank is one — the contraction convention puts
                    // every open axis of the left operand there.
                    .permuted_to_output(&source, output_axes, 1)
            }
            (None, None) => Ok(None),
        }
    }

    /// This tensor's axes, listed in `source[output_axes[..]]` order and split
    /// at `codomain_rank`, or `None` when `output_axes` is not a permutation of
    /// `0..source.len()`.
    ///
    /// `source` is the contraction's default output order expressed as axes of
    /// the scaled operand, so this is the fast path's counterpart of the erased
    /// `output_source_axes_for_order`. An `output_axes` that is not a
    /// permutation declines rather than errors: the dense route validates it
    /// and reports it, and one error message beats two.
    fn permuted_to_output(
        &self,
        source: &[usize],
        output_axes: &[usize],
        codomain_rank: usize,
    ) -> Result<Option<Self>, Error> {
        let mut sorted = output_axes.to_vec();
        sorted.sort_unstable();
        if sorted.iter().copied().ne(0..source.len()) {
            return Ok(None);
        }
        let ordered: Vec<usize> = output_axes.iter().map(|&axis| source[axis]).collect();
        self.permute(&ordered[..codomain_rank], &ordered[codomain_rank..])
            .map(Some)
    }

    /// This tensor with one bond axis of every block scaled by `spectrum`,
    /// on its own space. `axis = None` scales the trailing axis, `Some(0)` the
    /// leading one, exactly as the seam names them.
    fn scaled_axis(
        &self,
        axis: Option<usize>,
        spectrum: &[tenet_matrixalgebra::SectorSpectrum<D>],
    ) -> Result<Self, Error> {
        let mut data = self.dense_data().to_vec();
        tenet_matrixalgebra::scale_axis_by_spectrum_mapped(
            self.body.space.space(),
            &mut data,
            axis,
            spectrum,
            |value| value,
        )?;
        Ok(self.with_data(data))
    }

    /// Wraps one factor the matrix-algebra seam produced into a typed tensor
    /// map. `BoundDynFactor::into_parts` hands back exactly the pair
    /// [`TypedTensorBody`] stores, so there is nothing to validate here — the
    /// seam already certified the space against its own data.
    fn wrap_bound_factor(&self, factor: BoundDynFactor<R, D>) -> Self {
        wrap_factor_on(&self.runtime, factor)
    }

    /// Wraps a seam spectrum as a factor in compact diagonal storage: the bond
    /// space is derived from the spectrum itself, but the payload stays the
    /// `Σ_c k_c` values rather than the `Σ_c k_c²` block-diagonal buffer they
    /// would fill (TensorKit's `DiagonalTensorMap`).
    ///
    /// The spectrum is stored raw — engine [`tenet_core::SectorId`]s, values in
    /// the payload dtype `D`. Decoding belongs to the caller-facing spectrum
    /// fields, not to storage; a stored payload never leaves this module.
    ///
    /// Sorted by sector id first, matching the erased
    /// `Tensor::from_diagonal_real_spectrum`: the bond leg is built from this
    /// order, so the two facades' factors are only byte-comparable if both sort.
    fn diagonal_factor<V>(
        &self,
        spectrum: Vec<tenet_matrixalgebra::SectorSpectrum<V>>,
        to_scalar: impl Fn(V) -> D,
    ) -> Result<Self, Error> {
        diagonal_factor_on(&self.runtime, &self.body.space, spectrum, to_scalar)
    }

    /// Decodes a seam spectrum into provider labels and sorts it by label.
    ///
    /// Every id here came out of the engine's own coupled-sector enumeration,
    /// so a decode failure is the provider breaking [`SectorCodec`]'s
    /// decode-totality law — same contract as [`decode_block_fusion_trees`].
    fn decode_spectrum<V>(
        &self,
        raw: Vec<tenet_matrixalgebra::SectorSpectrum<V>>,
    ) -> Result<Vec<SectorSpectrum<R::Sector, V>>, Error> {
        let provider = self.body.space.provider();
        let mut decoded: Vec<SectorSpectrum<R::Sector, V>> = raw
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
        BoundDynamicTensorRef::try_new(&self.body.space, self.dense_data()).map_err(Error::from)
    }

    /// TensorKit 0.17 / MatrixAlgebraKit `svd_compact`: `t = u * s * vh` with
    /// the bond `min(rows, cols)` per coupled sector.
    ///
    /// Returns `(u, s, vh)` with `u : codomain <- bond`, `s : bond <- bond`
    /// and `vh : bond <- domain`.
    ///
    /// # Storage
    ///
    /// `s` is held in compact diagonal storage — `Σ_c k_c` values, not the
    /// `Σ_c k_c²` block-diagonal buffer — matching the `DiagonalTensorMap`
    /// TensorKit's own `svd_compact` returns. A downstream `u.compose(&s)` or
    /// `s.compose(&vh)` takes the O(d·n) bond-scaling path rather than a dense
    /// GEMM. [`Self::data`] still reports the dense buffer, materializing it
    /// once on demand; a caller who only needs the values should reach for
    /// [`Self::svd_vals`], which builds no factor at all.
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
        // Why the `_factors_` seam rather than `svd_compact_dyn`: the latter
        // builds the dense block-diagonal `s` itself, so taking it and throwing
        // it away would pay the very `Σ_c k_c²` allocation this storage avoids.
        let (u, vh, spectrum) =
            tenet_matrixalgebra::svd_compact_factors_dyn(dense.dense(), &self.bound_ref()?)?;
        Ok((
            self.wrap_bound_factor(u),
            self.diagonal_factor(spectrum, D::from_real)?,
            self.wrap_bound_factor(vh),
        ))
    }

    /// TensorKit 0.17 / MatrixAlgebraKit `svd_full`: `t = u * s * vh` with
    /// square unitaries and a rectangular `s` per coupled sector.
    ///
    /// Returns `(u, s, vh)` with `u : codomain <- W`, `s : W <- W'` and
    /// `vh : W' <- domain`.
    ///
    /// `s` is dense here where [`Self::svd_compact`]'s is diagonal, and that is
    /// TK-exact rather than a residual gap: TensorKit's own `svd_full` builds
    /// `s` as a dense rectangular tensor
    /// (`similar(t, real(scalartype(t)), V_cod <- V_dom)`). TensorKit's
    /// diagonal-`S` `svd_full!` applies to diagonal *inputs*, which is a
    /// different operation.
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
    /// `s` is in compact diagonal storage, exactly as [`Self::svd_compact`]'s.
    ///
    /// # Errors
    ///
    /// [`Error::Operation`] / [`Error::Core`] / [`Error::FusionAlgebra`] from
    /// the seam, including a malformed `truncation` — the truncation policy is
    /// validated where it is applied, not here.
    pub fn svd_trunc(&self, truncation: &Truncation) -> Result<SvdTrunc<R, D>, Error> {
        let mut dense = self.runtime.lease_dense();
        // The `_factors_` seam, for the reason `svd_compact` gives.
        let (u, vh, singular_values, error) = tenet_matrixalgebra::svd_trunc_factors_dyn(
            dense.dense(),
            &self.bound_ref()?,
            truncation,
        )?;
        Ok(SvdTrunc {
            u: self.wrap_bound_factor(u),
            s: self.diagonal_factor(singular_values.clone(), D::from_real)?,
            vh: self.wrap_bound_factor(vh),
            singular_values: self.decode_spectrum(singular_values)?,
            error,
        })
    }

    /// TensorKit 0.17 / MatrixAlgebraKit `svd_vals`: the singular values per
    /// coupled sector, and nothing else.
    ///
    /// No factor tensor and no bond space is built at all, so this is cheaper
    /// still than reading [`Self::svd_compact`]'s compact `s`.
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

    /// TensorKit 0.17 / MatrixAlgebraKit `eigh_full`: the Hermitian
    /// eigendecomposition `t = v * d * v^H` of an endomorphism, returned as
    /// `(d, v)`.
    ///
    /// `d : bond <- bond` carries the eigenvalues in compact diagonal storage
    /// (TensorKit's `DiagonalTensorMap`), so `v.compose(&d)` takes the
    /// bond-scaling path; `v : codomain <- bond` is the eigenbasis. The
    /// eigenvalues are real for both payload dtypes — TensorKit's Hermitian `D`
    /// is real too — but `d` keeps the payload dtype `D` so it composes with
    /// `v` directly.
    ///
    /// The `(d, v)` order is MatrixAlgebraKit's `initialize_output` order and
    /// the erased [`crate::prelude::Tensor::eigh_full`]'s, not the `v, d`
    /// reading order of the formula. It is deliberate on both facades.
    ///
    /// # Errors
    ///
    /// [`Error::Operation`] when the tensor is not an endomorphism or its
    /// coupled blocks are not Hermitian, and otherwise
    /// [`Error::Core`] / [`Error::FusionAlgebra`] from the seam — which owns
    /// those rules, so they are not re-checked here.
    pub fn eigh_full(&self) -> Result<(Self, Self), Error> {
        let mut dense = self.runtime.lease_dense();
        let out = tenet_matrixalgebra::eigh_full_dyn(dense.dense(), &self.bound_ref()?)?;
        let (v, eigenvalues) = out.into_parts();
        Ok((
            self.diagonal_factor(eigenvalues, D::from_real)?,
            self.wrap_bound_factor(v),
        ))
    }

    /// TensorKit 0.17 / MatrixAlgebraKit `eigh_trunc`: [`Self::eigh_full`] with
    /// the eigenbasis truncated by `truncation`; see [`EighTrunc`].
    ///
    /// Returned as a named struct rather than a four-tuple, the same rule
    /// [`Self::svd_trunc`] follows.
    ///
    /// # Errors
    ///
    /// Exactly [`Self::eigh_full`]'s, plus a malformed `truncation` — validated
    /// where it is applied, not here.
    pub fn eigh_trunc(&self, truncation: &Truncation) -> Result<EighTrunc<R, D>, Error> {
        let mut dense = self.runtime.lease_dense();
        let out =
            tenet_matrixalgebra::eigh_trunc_dyn(dense.dense(), &self.bound_ref()?, truncation)?;
        let (v, eigenvalues, error) = out.into_parts();
        Ok(EighTrunc {
            d: self.diagonal_factor(eigenvalues.clone(), D::from_real)?,
            v: self.wrap_bound_factor(v),
            eigenvalues: self.decode_spectrum(eigenvalues)?,
            error,
        })
    }

    /// TensorKit 0.17 / MatrixAlgebraKit `eigh_vals`: the Hermitian eigenvalues
    /// per coupled sector, and nothing else.
    ///
    /// No factor and no bond space is built, so this is the cheap way to ask
    /// about a spectrum — the [`Self::svd_vals`] of the eigendecompositions.
    ///
    /// # Errors
    ///
    /// [`Self::eigh_full`]'s, plus [`Error::FusionAlgebra`] when the provider
    /// cannot decode a coupled sector its own algebra produced.
    pub fn eigh_vals(&self) -> Result<Vec<SectorSpectrum<R::Sector>>, Error> {
        let mut dense = self.runtime.lease_dense();
        let raw = tenet_matrixalgebra::eigh_vals_dyn(dense.dense(), &self.bound_ref()?)?;
        self.decode_spectrum(raw)
    }

    /// TensorKit 0.17 / MatrixAlgebraKit `eig_full`: the general
    /// (non-Hermitian) eigendecomposition `t = v * d * v^-1` of an
    /// endomorphism, returned as `(d, v)` — [`Self::eigh_full`]'s order, for
    /// the same reason.
    ///
    /// Both factors are complex whatever `D` is: a real matrix's eigenpairs are
    /// complex in general, and TensorKit's `eigen` likewise returns
    /// `ComplexF64` `D` and `V` for a real argument. `d` carries the spectrum
    /// in compact diagonal storage.
    ///
    /// # The `D::Eig` bound
    ///
    /// The `where` clause is vacuous for the two payload types this facade
    /// admits — `f64` and `Complex64` both have `Eig = Complex64`, which is a
    /// [`TensorScalar`]. It is written out because
    /// [`tenet_matrixalgebra::FactorScalar::Eig`] is the wider seam's associated
    /// type and is not constrained to this facade's scalars, so without it the
    /// factors could not be `TensorMap`s at all. Per-method rather than on the
    /// impl block, so nothing outside the `eig_*` row pays for it.
    ///
    /// # Errors
    ///
    /// [`Error::Operation`] when the tensor is not an endomorphism, and
    /// otherwise [`Error::Core`] / [`Error::FusionAlgebra`] from the seam.
    #[allow(clippy::type_complexity)]
    pub fn eig_full(
        &self,
    ) -> Result<
        (
            TensorMap<R, <D as FactorScalar>::Eig>,
            TensorMap<R, <D as FactorScalar>::Eig>,
        ),
        Error,
    >
    where
        <D as FactorScalar>::Eig: TensorScalar,
    {
        let mut dense = self.runtime.lease_dense();
        let out = tenet_matrixalgebra::eig_full_dyn(dense.dense(), &self.bound_ref()?)?;
        let (v, eigenvalues) = out.into_parts();
        Ok((
            diagonal_factor_on(
                &self.runtime,
                &self.body.space,
                eigenvalues,
                <<D as FactorScalar>::Eig as FactorScalar>::from_complex64,
            )?,
            wrap_factor_on(&self.runtime, v),
        ))
    }

    /// TensorKit 0.17 / MatrixAlgebraKit `eig_trunc`: [`Self::eig_full`] with
    /// the eigenbasis truncated by descending `|eigenvalue|`; see [`EigTrunc`].
    ///
    /// # Errors
    ///
    /// Exactly [`Self::eig_full`]'s, plus a malformed `truncation`.
    pub fn eig_trunc(&self, truncation: &Truncation) -> Result<EigTrunc<R, D>, Error>
    where
        // See [`Self::eig_full`] for why this bound is per-method.
        <D as FactorScalar>::Eig: TensorScalar,
    {
        let mut dense = self.runtime.lease_dense();
        let out =
            tenet_matrixalgebra::eig_trunc_dyn(dense.dense(), &self.bound_ref()?, truncation)?;
        let (v, eigenvalues, error) = out.into_parts();
        Ok(EigTrunc {
            d: diagonal_factor_on(
                &self.runtime,
                &self.body.space,
                eigenvalues.clone(),
                <<D as FactorScalar>::Eig as FactorScalar>::from_complex64,
            )?,
            v: wrap_factor_on(&self.runtime, v),
            eigenvalues: self.decode_spectrum(eigenvalues)?,
            error,
        })
    }

    /// TensorKit 0.17 / MatrixAlgebraKit `eig_vals`: the general eigenvalues
    /// per coupled sector, and nothing else. Complex for both payload dtypes.
    ///
    /// # Errors
    ///
    /// [`Self::eig_full`]'s, plus [`Error::FusionAlgebra`] when the provider
    /// cannot decode a coupled sector its own algebra produced.
    pub fn eig_vals(&self) -> Result<Vec<SectorSpectrum<R::Sector, num_complex::Complex64>>, Error>
    where
        // Carried across the whole row even though this member builds no
        // factor: the three are one API surface, and a caller who can spell two
        // of them but not the third would be reading an accident.
        <D as FactorScalar>::Eig: TensorScalar,
    {
        let mut dense = self.runtime.lease_dense();
        let raw = tenet_matrixalgebra::eig_vals_dyn(dense.dense(), &self.bound_ref()?)?;
        self.decode_spectrum(raw)
    }

    /// The matrix exponential `exp(t) = Σ_k t^k / k!`, evaluated per coupled
    /// sector — TensorKit's `exp` (`linalg.jl:44`), which copies and calls
    /// `exp!` (`linalg.jl:420-428`): check `domain == codomain`, then
    /// exponentiate every block.
    ///
    /// # Domain
    ///
    /// Any endomorphism, of any dtype. Since issue #577 there are two dense
    /// routes and the input picks one:
    ///
    /// - **Hermitian blocks** take the spectral function `V exp(D) Vᴴ` of the
    ///   Hermitian eigendecomposition. Exact, and the cheaper of the two.
    /// - **Everything else** takes blockwise scaling-and-squaring Padé [13/13]
    ///   (Higham 2005) — the algorithm behind the `LinearAlgebra.exp!` that
    ///   TensorKit's own `exp!` calls. Non-normal, defective and complex
    ///   non-Hermitian blocks are all in domain; nothing is symmetrized.
    ///
    /// The **compact** arm is TensorKit's `exp(::DiagonalTensorMap)`:
    /// unconditionally elementwise, with no hermiticity gate. Storage therefore
    /// decides how `exp` is computed, no longer whether it is defined.
    ///
    /// # Errors
    ///
    /// - [`Error::Operation`] when the input is not an endomorphism
    ///   (`codomain != domain`), when a general block holds a nonfinite entry,
    ///   when a general block's column 1-norm overflows to infinity although
    ///   every entry of it is finite, or when the backend fails — including an
    ///   executor that supplies no
    ///   dense solve, which the Padé route needs and which surfaces as
    ///   `DenseError::Unsupported`. Nothing is published unless every coupled
    ///   sector succeeded.
    /// - [`Error::Core`] / [`Error::FusionAlgebra`] from the composition that
    ///   reassembles `V exp(D) Vᴴ` on the Hermitian route.
    ///
    /// # Complexity
    ///
    /// Dense input: `O(Σ_c n_c³)` on both routes — one Hermitian
    /// eigendecomposition per coupled sector plus one composition, with
    /// `exp(D)` folded into a column scaling of `V` rather than materialized;
    /// or six GEMMs, one solve and `s = max(0, ceil(log2(||A_c||_1 / theta_13)))`
    /// squarings per sector — over the balanced block, so a badly scaled one
    /// pays for its true magnitude and not its scaling — with an
    /// `O(max_c n_c²)` Padé workspace reused across sectors. That workspace is
    /// the whole of the scratch on the canonical layout; a payload whose
    /// coupled sectors are not laid out in contiguous regions takes a fallback
    /// that matricizes them all first, adding `O(Σ_c n_c²)`. Neither route
    /// couples sectors. Compact input (TensorKit's
    /// `DiagonalTensorMap`): the **O(rank) elementwise arm**, `exp(s_i)` over
    /// the `Σ_c k_c` stored values, staying compact. The erased
    /// [`crate::prelude::Tensor::exp`] has the same arm since issue #578 — it
    /// used to materialize a diagonal payload and eigendecompose the
    /// block-diagonal buffer — so the two facades agree on complexity and on
    /// what each storage accepts.
    pub fn exp(&self) -> Result<Self, Error> {
        if let Some(spectrum) = self.spectrum() {
            // Why the spectrum is exponentiated unconditionally while the dense
            // arm asks about hermiticity: the dense question picks an algorithm
            // (spectral or Padé), not a domain, and a diagonal is already in its
            // eigenbasis so neither answer would change what happens here.
            // TensorKit splits the same way (#576, #578).
            return Ok(self.with_spectrum(map_spectrum(spectrum, |value| Ok(value.exp_value()))?));
        }
        let mut dense = self.runtime.lease_dense();
        let mut lease = self.runtime.lease_context()?;
        let out = tenet_matrixalgebra::exp_dyn(
            dense.dense(),
            lease.context().multiplicity_free_lane::<D>(),
            &BoundDynamicTensorRef::try_new(&self.body.space, self.dense_data())?,
        )?;
        Ok(self.wrap_bound_factor(out))
    }

    /// TensorKit 0.17 / MatrixAlgebraKit `inv`: the true inverse `t^-1` of a
    /// nonsingular map, defined by `t * t^-1 = id` on the codomain and
    /// `t^-1 * t = id` on the domain. Computed per coupled sector as the exact
    /// dense solve `t_c X_c = 1`, not as a spectral function — there is no
    /// truncation policy to apply and no factor tensor to build.
    ///
    /// # Domain
    ///
    /// TensorKit asks for `codomain ≅ domain` — **isomorphic, not equal** —
    /// and returns a map `domain <- codomain`. This facade's seam agrees: a
    /// rank-one codomain and a rank-two domain with the same coupled-sector
    /// dimensions are accepted, and the result carries the two spaces swapped.
    /// The pin is `inv_accepts_isomorphic_but_unequal_codomain_and_domain`.
    ///
    /// # Errors
    ///
    /// - [`Error::Operation`] when the two sides are not isomorphic, and when a
    ///   coupled-sector block is singular — the dense solve is where that
    ///   surfaces, so it comes back as an execution error rather than an
    ///   argument one. Never a panic.
    /// - [`Error::InvalidArgument`] from the compact arm below, whose zero
    ///   entry is visible before any solve runs and is therefore reported as
    ///   the caller mistake it is. The two storages of one singular tensor
    ///   consequently report different variants; both are pinned by
    ///   `inv_reports_a_singular_input_as_a_typed_error`.
    ///
    /// # Complexity
    ///
    /// Dense input: `O(Σ_c n_c³)`, one LU solve per coupled sector. Compact
    /// input (a spectrum factor, TensorKit's `DiagonalTensorMap`): the
    /// **O(rank) elementwise-reciprocal arm**, `1/s_i` over the `Σ_c k_c`
    /// stored values, and the result stays compact — matching TensorKit's
    /// `inv(::DiagonalTensorMap)`, which is `inv.(d.data)`. Nothing dense is
    /// built on either side of that arm.
    pub fn inv(&self) -> Result<Self, Error> {
        if let Some(spectrum) = self.spectrum() {
            // Why `== 0` and not a tolerance: the dense arm has none either
            // (the solve either fails or it does not), and a compact arm that
            // refused near-zero entries would let storage change the answer.
            // Same comparison as the erased facade's `try_recip`.
            return Ok(self.with_spectrum(map_spectrum(spectrum, |value| {
                if value.abs_value() == 0.0 {
                    Err(Error::InvalidArgument(
                        "inv of a singular diagonal (zero entry)".to_string(),
                    ))
                } else {
                    Ok(value.recip_value())
                }
            })?));
        }
        let mut dense = self.runtime.lease_dense();
        let out = tenet_matrixalgebra::inv_direct_dyn(dense.dense(), &self.bound_ref()?)?;
        Ok(self.wrap_bound_factor(out))
    }

    /// TensorKit 0.17 / MatrixAlgebraKit `pinv`: the Moore-Penrose
    /// pseudo-inverse `t⁺ = V S⁺ Uᴴ`, where `t = U S Vᴴ` is the compact SVD and
    /// `S⁺` inverts every singular value above the cutoff and sends the rest to
    /// zero. `t⁺` satisfies `t t⁺ t = t` and reduces to [`Self::inv`] when `t`
    /// is nonsingular and `rcond` is small enough to keep every singular value.
    ///
    /// # Tolerance, and the divergence from TensorKit
    ///
    /// The cutoff is `rcond * σ_max` with **one global `σ_max` taken across all
    /// coupled sectors**, and the comparison is strict: a singular value
    /// sitting exactly on the cutoff is discarded. TensorKit instead takes
    /// per-block `atol`/`rtol` keywords, so its relative tolerance is measured
    /// against each block's own largest singular value. That is a deliberate
    /// divergence, not a gap: a per-block relative tolerance cannot cut
    /// anything in a one-dimensional sector however small that sector's
    /// contribution to the tensor is, and TensorKit's own source carries a TODO
    /// saying the tolerance should be relative to the total norm — which is
    /// what this facade already does. TensorKit's `DiagonalTensorMap` branch is
    /// deliberately **not** mirrored either: there `rtol` is ignored whenever
    /// `atol` is nonzero, the default is no cutoff at all, and its comparison
    /// (`abs(x) < tol` discards) *keeps* a value sitting exactly on the cutoff
    /// — the opposite of the strict `>` above. On that last point TensorKit
    /// contradicts itself: its general `pinv` goes through Julia's, which keeps
    /// only `sigma > tol`, and that is the boundary this facade matches on both
    /// storages.
    ///
    /// # Errors
    ///
    /// - [`Error::InvalidArgument`] when `rcond` is not finite or is negative.
    ///   Checked before any work on both storages.
    /// - [`Error::Operation`] / [`Error::Core`] from the SVD, on the dense arm.
    ///
    /// There is no singular-input failure: sending the offending directions to
    /// zero is what a pseudo-inverse is for.
    ///
    /// # Complexity
    ///
    /// Dense input: one compact SVD, `O(Σ_c n_c³)`, plus a bond scaling and one
    /// composition; `S⁺` is folded into a column scaling rather than
    /// materialized. Compact input (TensorKit's `DiagonalTensorMap`): the
    /// **O(rank) elementwise cutoff-and-reciprocal arm** over the `Σ_c k_c`
    /// stored values — the singular values of a diagonal are its `|entry|`s, so
    /// no SVD is needed — and the result stays compact.
    pub fn pinv(&self, rcond: f64) -> Result<Self, Error> {
        // Ahead of the storage split, so both arms answer alike: the seam
        // repeats this check for its own callers, but the compact arm never
        // reaches the seam.
        if !rcond.is_finite() || rcond < 0.0 {
            return Err(Error::InvalidArgument(
                "pinv rcond must be finite and non-negative".to_string(),
            ));
        }
        if let Some(spectrum) = self.spectrum() {
            let cutoff = rcond
                * spectrum
                    .iter()
                    .flat_map(|entry| entry.values.iter())
                    .fold(0.0f64, |largest, &value| largest.max(value.abs_value()));
            // Strict `>`, matching the dense fold and the erased facade: a
            // value exactly on the cutoff is cut. Changing it to `>=` is what
            // `pinv_cuts_a_singular_value_sitting_exactly_on_the_cutoff` kills.
            return Ok(self.with_spectrum(map_spectrum(spectrum, |value| {
                Ok(if value.abs_value() > cutoff {
                    value.recip_value()
                } else {
                    D::from_real(0.0)
                })
            })?));
        }
        let mut dense = self.runtime.lease_dense();
        let mut lease = self.runtime.lease_context()?;
        let out = tenet_matrixalgebra::pinv_dyn(
            dense.dense(),
            lease.context().multiplicity_free_lane::<D>(),
            &BoundDynamicTensorRef::try_new(&self.body.space, self.dense_data())?,
            rcond,
        )?;
        Ok(self.wrap_bound_factor(out))
    }

    /// TensorKit 0.17 `sqrt(::DiagonalTensorMap)`: the elementwise principal
    /// square root of a diagonal bond tensor, `√s_i` on each diagonal entry, so
    /// that `√t · √t = t`. This is the idiom that splits singular values in
    /// Vidal-gauge and gate-application updates.
    ///
    /// # Domain
    ///
    /// The receiver must be a **diagonal bond tensor** `[v] <- [v]`: one
    /// codomain leg equal to the one domain leg, and every stored block
    /// diagonal, with off-diagonal entries exactly zero. That is the shape the
    /// factorizations produce ([`Self::svd_compact`]'s and [`Self::svd_trunc`]'s
    /// `s`, [`Self::eigh_full`]'s `d`), and it is the receiver type TensorKit's
    /// own diagonal `sqrt` demands.
    ///
    /// General endomorphism `sqrt` is deliberately out of scope. TensorKit does
    /// have one (`sqrt(::AbstractTensorMap)`, Schur-based, always returning a
    /// complex tensor), but no Schur seam exists below this facade, and its
    /// value-independent complexification is not expressible in a typed
    /// signature. A wider `sqrt` is a separate phase, not an omission here.
    ///
    /// # Errors
    ///
    /// [`Error::InvalidArgument`] in every failure case:
    ///
    /// - the receiver is not shaped `[v] <- [v]`;
    /// - a stored block has a nonzero off-diagonal entry (dense arm only — a
    ///   compact payload has none by construction);
    /// - the payload is `f64` and a diagonal entry is negative. The message
    ///   points at the complex payload, matching the erased facade and
    ///   TensorKit's diagonal-path `DomainError`. TensorKit's *dense* path
    ///   instead complexifies silently, which contradicts its own diagonal path
    ///   and is not mirrored here.
    ///
    /// A [`num_complex::Complex64`] payload never fails on a value: it takes the
    /// principal branch (`√(-1) = +i`).
    ///
    /// # Complexity
    ///
    /// Compact input (TensorKit's `DiagonalTensorMap`, what the factorizations
    /// hand back): the **O(rank) elementwise arm** over the `Σ_c k_c` stored
    /// values, staying compact — so `s.sqrt()` and the two `compose`s around it
    /// are all bond scalings. Dense input: `O(Σ_c n_c²)`, one walk over the
    /// block-diagonal buffer, which is what the off-diagonal check costs; the
    /// root itself is still only `Σ_c n_c` square roots.
    pub fn sqrt(&self) -> Result<Self, Error> {
        // Same guard as the erased facade's, and the same one
        // [`is_diagonal_bond_space`] applies to compact *destinations*: here it
        // is asked of the receiver, which is what makes it reachable.
        if !is_diagonal_bond_space(self.body.space.space()) {
            return Err(Error::InvalidArgument(
                "sqrt requires a diagonal bond tensor `[v] <- [v]` (equal single \
                 codomain and domain legs), like the `s` factor of svd_trunc"
                    .to_string(),
            ));
        }
        if let Some(spectrum) = self.spectrum() {
            return Ok(self.with_spectrum(map_spectrum(spectrum, D::sqrt_value)?));
        }
        // Dense payload on a bond space: block-diagonal by the space's shape,
        // but only by convention — the buffer is free to hold anything, so the
        // off-diagonal entries are checked rather than assumed. Skipping the
        // check would silently drop them.
        let data = self.dense_data();
        let zero = num_complex::Complex64::new(0.0, 0.0);
        let mut out = vec![D::from_real(0.0); data.len()];
        let structure = self.body.space.space().structure();
        for index in 0..structure.block_count() {
            let block = structure.block(index)?;
            let (shape, strides, offset) = (block.shape(), block.strides(), block.offset());
            for row in 0..shape[0] {
                for col in 0..shape[1] {
                    let position = offset + row * strides[0] + col * strides[1];
                    if row == col {
                        out[position] = data[position].sqrt_value()?;
                    } else if data[position].widen_complex() != zero {
                        return Err(Error::InvalidArgument(format!(
                            "sqrt requires a diagonal bond tensor, but block {:?} has a \
                             nonzero off-diagonal entry at ({row}, {col})",
                            block.key()
                        )));
                    }
                }
            }
        }
        Ok(self.with_data(out))
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
            body: Arc::new(TypedTensorBody::dense(self.body.space.clone(), data)),
        }
    }

    /// A sibling on this tensor's own space carrying a new compact spectrum.
    /// Every operation that reaches this keeps the bond space it was called on,
    /// so the checked admission proof carries over exactly as for
    /// [`Self::with_data`].
    fn with_spectrum(&self, spectrum: Vec<tenet_matrixalgebra::SectorSpectrum<D>>) -> Self {
        Self {
            runtime: self.runtime.clone(),
            body: Arc::new(TypedTensorBody {
                space: self.body.space.clone(),
                data: TypedData::Diagonal(spectrum),
                dense_cache: std::sync::OnceLock::new(),
            }),
        }
    }

    /// A sibling on a **different** space carrying a new compact spectrum —
    /// [`Self::with_spectrum`] for the one operation that moves the bond space
    /// rather than keeping it. The space must be one the expert layer derived
    /// from this tensor's own, so the checked admission proof carries over the
    /// same way.
    fn with_spectrum_on(
        &self,
        space: BoundDynamicFusionMapSpace<R>,
        spectrum: Vec<tenet_matrixalgebra::SectorSpectrum<D>>,
    ) -> Self {
        Self {
            runtime: self.runtime.clone(),
            body: Arc::new(TypedTensorBody {
                space,
                data: TypedData::Diagonal(spectrum),
                dense_cache: std::sync::OnceLock::new(),
            }),
        }
    }

    /// The compact payload, when this tensor has one.
    fn spectrum(&self) -> Option<&[tenet_matrixalgebra::SectorSpectrum<D>]> {
        match &self.body.data {
            TypedData::Diagonal(spectrum) => Some(spectrum),
            TypedData::Dense(_) => None,
        }
    }

    /// Whether two operands' providers are the same rule. The compact paths
    /// below skip the expert layer, which is where a mismatch would otherwise
    /// be caught, so they have to ask themselves.
    fn same_rule(&self, other: &Self) -> bool {
        self.body.space.provider().rule_identity() == other.body.space.provider().rule_identity()
    }

    /// The dimension-weighted inner product of two compact spectra,
    /// `Σ_c dim(c) * Σ_i conj(a_i) b_i` — [`weighted_inner`]'s reduction with
    /// the zeros left out, since a bond space's dense form is zero off the
    /// per-sector diagonal.
    fn compact_inner(
        lhs: &[tenet_matrixalgebra::SectorSpectrum<D>],
        rhs: &[tenet_matrixalgebra::SectorSpectrum<D>],
        provider: &R,
    ) -> Result<num_complex::Complex64, Error> {
        if lhs.len() != rhs.len() {
            return Err(spectra_disagree());
        }
        let mut total = num_complex::Complex64::new(0.0, 0.0);
        for (left, right) in lhs.iter().zip(rhs) {
            if left.sector != right.sector || left.values.len() != right.values.len() {
                return Err(spectra_disagree());
            }
            let mut partial = D::from_real(0.0);
            for (&a, &b) in left.values.iter().zip(&right.values) {
                partial = partial + FactorScalar::adjoint(a) * b;
            }
            total += partial.widen_complex() * provider.dim_scalar(left.sector);
        }
        Ok(total)
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
    ///   with the erased facade's own message. Operands whose providers report
    ///   different rule identities land here too, rather than in
    ///   [`Error::RuleMismatch`] as the erased `check_same_world` would report
    ///   them: the space comparison already covers rule identity, so a
    ///   separate check would only re-report the same disagreement.
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
        match (self.spectrum(), other.spectrum()) {
            // Two spectra on one bond space: the sum is diagonal too.
            (Some(lhs), Some(rhs)) => {
                if lhs.len() != rhs.len() {
                    return Err(spectra_disagree());
                }
                let sum = lhs
                    .iter()
                    .zip(rhs)
                    .map(|(left, right)| {
                        if left.sector != right.sector || left.values.len() != right.values.len() {
                            return Err(spectra_disagree());
                        }
                        Ok(tenet_matrixalgebra::SectorSpectrum {
                            sector: left.sector,
                            values: left
                                .values
                                .iter()
                                .zip(&right.values)
                                .map(|(&x, &y)| x * alpha + y * beta)
                                .collect(),
                        })
                    })
                    .collect::<Result<Vec<_>, Error>>()?;
                return Ok(self.with_spectrum(sum));
            }
            // Mixed: the result is dense, but the *diagonal operand* is never
            // materialized to get there — the one O(n²) buffer this needs is
            // the owned result, which scatters the spectrum onto its own
            // diagonal. Materializing first would allocate a second.
            (Some(diagonal), None) => {
                return Ok(self.with_data(scatter_spectrum(
                    self.body.space.space(),
                    other.dense_data(),
                    beta,
                    diagonal,
                    alpha,
                )?))
            }
            (None, Some(diagonal)) => {
                return Ok(self.with_data(scatter_spectrum(
                    self.body.space.space(),
                    self.dense_data(),
                    alpha,
                    diagonal,
                    beta,
                )?))
            }
            (None, None) => {}
        }
        Ok(self.with_data(
            self.dense_data()
                .iter()
                .zip(other.dense_data())
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
    ///
    /// Compact diagonal storage is preserved: scaling a spectrum factor stays
    /// `Σ_c k_c` values rather than densifying.
    pub fn scale(&self, factor: D) -> Self {
        if let Some(spectrum) = self.spectrum() {
            return self.with_spectrum(
                spectrum
                    .iter()
                    .map(|entry| tenet_matrixalgebra::SectorSpectrum {
                        sector: entry.sector,
                        values: entry.values.iter().map(|&value| value * factor).collect(),
                    })
                    .collect(),
            );
        }
        self.with_data(
            self.dense_data()
                .iter()
                .map(|&value| value * factor)
                .collect(),
        )
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
            self.dense_data(),
            axes,
            D::from_real(1.0),
        )?;
        Ok(Self {
            runtime: self.runtime.clone(),
            body: Arc::new(TypedTensorBody::dense(space, data)),
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
        if let Some(spectrum) = self.spectrum() {
            // A bond space is its own adjoint (`codomain == domain`), and the
            // dagger of a diagonal is the conjugated diagonal — so this is
            // O(Σ_c k_c) with no dense buffer and no bend. For a real payload
            // `FactorScalar::adjoint` is the identity, which is why there is no
            // separate real arm: the erased facade's `RealF64`/`RealC64`/`C64`
            // split exists only because its dtype is a runtime property.
            return Ok(self.with_spectrum(
                spectrum
                    .iter()
                    .map(|entry| tenet_matrixalgebra::SectorSpectrum {
                        sector: entry.sector,
                        values: entry
                            .values
                            .iter()
                            .map(|&value| FactorScalar::adjoint(value))
                            .collect(),
                    })
                    .collect(),
            ));
        }
        let (space, data) = tenet_tensors::adjoint_bound_dyn(&self.body.space, self.dense_data())
            .map_err(Error::from)?;
        Ok(Self {
            runtime: self.runtime.clone(),
            body: Arc::new(TypedTensorBody::dense(space, data)),
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
        if let Some(spectrum) = self.spectrum() {
            let provider = self.body.space.provider();
            return Ok(Self::compact_inner(spectrum, spectrum, provider)?.re.sqrt());
        }
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
        if let Some(spectrum) = self.spectrum() {
            return Ok(spectrum
                .iter()
                .flat_map(|entry| entry.values.iter())
                .map(|&value| value.widen_complex().norm())
                .fold(0.0, f64::max));
        }
        Ok(self
            .dense_data()
            .iter()
            .map(|&value| value.widen_complex().norm())
            .fold(0.0, f64::max))
    }

    /// TensorKit `norm(t, p)` for a general exponent
    /// (`src/tensors/linalg.jl:257-275`):
    ///
    /// ```text
    /// p == Inf     -> maximum entry magnitude over blocks(t)
    /// finite p > 0 -> (Σ_c dim(c) * norm(block_c, p)^p)^(1/p)
    /// ```
    ///
    /// `norm(block, p)` is Julia's *entrywise* p-norm — matrices included — so
    /// this is never an operator norm. Only `p == 2` is the quantum-dimension
    /// weighted Frobenius norm of [`Self::norm`]; every other exponent weights
    /// the same `dim(c)` against a different power sum.
    ///
    /// A separate method rather than an optional argument because Rust has no
    /// overloading. `p == 2.0` and `p == f64::INFINITY` delegate to
    /// [`Self::norm`] and [`Self::norm_inf`], so the three cannot drift apart.
    ///
    /// # Complexity
    ///
    /// One pass over the payload: `O(N)` for a dense payload of `N` scalars,
    /// `O(Σ_c k_c)` on compact diagonal storage. The compact arm reads the
    /// stored spectra and never materializes — the `k_c² − k_c` off-diagonal
    /// zeros contribute nothing to any `p > 0`.
    ///
    /// # Errors
    ///
    /// - [`Error::InvalidArgument`] when `p` is NaN, zero, negative, or `-inf`;
    ///   TensorKit throws `ArgumentError` over the same domain.
    /// - [`Error::Core`] when the block structure cannot be walked, exactly as
    ///   [`Self::norm`].
    pub fn norm_p(&self, p: f64) -> Result<f64, Error> {
        // Checked before any dispatch so an invalid `p` is rejected the same
        // way on compact and dense storage.
        validate_norm_p(p)?;
        if p == 2.0 {
            return self.norm();
        }
        if p.is_infinite() {
            return self.norm_inf();
        }
        let provider = self.body.space.provider();
        if let Some(spectrum) = self.spectrum() {
            let total: f64 = spectrum
                .iter()
                .map(|entry| {
                    provider.dim_scalar(entry.sector)
                        * entry
                            .values
                            .iter()
                            .map(|&value| value.widen_complex().norm().powf(p))
                            .sum::<f64>()
                })
                .sum();
            return Ok(total.powf(p.recip()));
        }
        coupled_region_pow_sum(
            self.body.space.space().structure(),
            self.body.space.space().nout(),
            self.dense_data(),
            p,
            |coupled| provider.dim_scalar(coupled),
        )
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
            self.dense_data(),
            self.dense_data(),
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
    /// Exactly [`Self::add`]'s — the operands must share a runtime and a space
    /// — plus [`Error::Core`] from the block-structure walk, as for
    /// [`Self::norm`].
    pub fn inner(&self, other: &Self) -> Result<D, Error> {
        if !self.runtime.same_runtime(&other.runtime) {
            return Err(Error::RuntimeMismatch);
        }
        if self.body.space.space() != other.body.space.space() {
            return Err(Error::InvalidArgument(
                "tensors live on different spaces or block layouts".to_string(),
            ));
        }
        // Two compact spectra reduce without either being materialized. A mixed
        // pair does not: the dense operand has to be read at its diagonal
        // positions anyway, so it goes through the shared materialization
        // cache like every other dense consumer.
        if let (Some(lhs), Some(rhs)) = (self.spectrum(), other.spectrum()) {
            let provider = self.body.space.provider();
            return Ok(D::from_complex64(Self::compact_inner(lhs, rhs, provider)?));
        }
        // `D::from_complex64` is `.re` for the real scalar and the identity for
        // the complex one, so this is bit-identical to the erased facade's
        // `Scalar::F64(v.re)` / `Scalar::C64(v)` dispatch, without the enum.
        Ok(D::from_complex64(weighted_inner(
            self.body.space.provider(),
            self.body.space.space().structure(),
            self.body.space.space().nout(),
            self.dense_data(),
            other.dense_data(),
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
        if let Some(spectrum) = self.spectrum() {
            // `Σ_c dim(c) * Σ_i d_i`: TensorKit's positive trace on a
            // `DiagonalTensorMap`, read straight off the stored values.
            let provider = self.body.space.provider();
            let mut total = num_complex::Complex64::new(0.0, 0.0);
            for entry in spectrum {
                let mut partial = D::from_real(0.0);
                for &value in &entry.values {
                    partial = partial + value;
                }
                total += partial.widen_complex() * provider.dim_scalar(entry.sector);
            }
            return Ok(D::from_complex64(total));
        }
        Ok(D::from_complex64(weighted_trace(
            self.body.space.provider(),
            self.body.space.space().structure(),
            self.body.space.space().nout(),
            self.dense_data(),
        )?))
    }

    /// Whether the tensor equals its own adjoint within `tol`, relative to its
    /// norm (TensorKit `ishermitian`).
    ///
    /// A non-endomorphism is never Hermitian and comes back `false` rather than
    /// as an error, which is where both this facade and the erased one differ
    /// from TensorKit — it throws. A predicate that can only be called after
    /// another predicate is not one.
    ///
    /// # Errors
    ///
    /// [`Self::add`]'s and [`Self::adjoint`]'s, which is where the work happens.
    pub fn is_hermitian(&self, tol: f64) -> Result<bool, Error> {
        if !self.is_endomorphism() {
            return Ok(false);
        }
        let difference = self.add(&self.adjoint()?, D::from_real(1.0), D::from_real(-1.0))?;
        Ok(difference.norm()? <= tol * self.norm()?.max(1.0))
    }

    /// Whether the tensor equals minus its own adjoint within `tol`
    /// (TensorKit `isantihermitian`). A non-endomorphism is `false`, as for
    /// [`Self::is_hermitian`].
    ///
    /// # Errors
    ///
    /// Exactly [`Self::is_hermitian`]'s.
    pub fn is_antihermitian(&self, tol: f64) -> Result<bool, Error> {
        if !self.is_endomorphism() {
            return Ok(false);
        }
        let sum = self.add(&self.adjoint()?, D::from_real(1.0), D::from_real(1.0))?;
        Ok(sum.norm()? <= tol * self.norm()?.max(1.0))
    }

    /// Whether `t† ∘ t` is the identity on the domain within `tol`
    /// (TensorKit `isisometric`): the columns are orthonormal. Defined for any
    /// shape, not only square ones.
    ///
    /// # Errors
    ///
    /// [`Self::adjoint`]'s, [`Self::compose`]'s and [`Self::id`]'s.
    pub fn is_isometric(&self, tol: f64) -> Result<bool, Error> {
        let gram = self.adjoint()?.compose(self)?;
        let identity = Self::id(&self.runtime, &self.domain())?;
        let difference = gram.add(&identity, D::from_real(1.0), D::from_real(-1.0))?;
        Ok(difference.norm()? <= tol * gram.norm()?.max(1.0))
    }

    /// Whether the tensor is unitary within `tol` (TensorKit `isunitary`):
    /// isometric in both directions.
    ///
    /// # Errors
    ///
    /// Exactly [`Self::is_isometric`]'s.
    pub fn is_unitary(&self, tol: f64) -> Result<bool, Error> {
        Ok(self.is_isometric(tol)? && self.adjoint()?.is_isometric(tol)?)
    }

    /// Whether the tensor is Hermitian and positive definite (TensorKit
    /// `isposdef`): every Hermitian eigenvalue exceeds `tol * max(norm, 1)`.
    ///
    /// Strict, like TensorKit's Cholesky-based test: a positive *semi*definite
    /// spectrum — an eigenvalue at zero — is `false`. With `tol = 0.0` this is
    /// exact strict positivity up to floating point.
    ///
    /// # Errors
    ///
    /// [`Self::is_hermitian`]'s and [`Self::eigh_vals`]'s.
    pub fn is_posdef(&self, tol: f64) -> Result<bool, Error> {
        if !self.is_hermitian(tol)? {
            return Ok(false);
        }
        let threshold = tol * self.norm()?.max(1.0);
        // Compact arm: a spectrum factor's stored values *are* its Hermitian
        // eigenvalues, so there is nothing to factorize and nothing to
        // materialize (#585). The gate and the threshold above are already
        // compact — `is_hermitian` and `norm` both read the spectrum — so this
        // is the last step that reached `dense_data()`.
        if let Some(spectrum) = self.spectrum() {
            return Ok(crate::typed_tensor_core::compact_is_posdef(
                spectrum, threshold,
            ));
        }
        Ok(self
            .eigh_vals()?
            .iter()
            .flat_map(|spectrum| spectrum.values.iter())
            .all(|&eigenvalue| eigenvalue > threshold))
    }

    /// The Hermitian part `(t + t†)/2` (TensorKit `project_hermitian`), the
    /// nearest Hermitian tensor.
    ///
    /// # Errors
    ///
    /// [`Self::add`]'s — including [`Error::InvalidArgument`] when the tensor is
    /// not an endomorphism, since then it and its adjoint live on different
    /// spaces. Unlike [`Self::is_hermitian`] there is no `false` to return here.
    pub fn project_hermitian(&self) -> Result<Self, Error> {
        self.add(&self.adjoint()?, D::from_real(0.5), D::from_real(0.5))
    }

    /// The anti-Hermitian part `(t - t†)/2` (TensorKit
    /// `project_antihermitian`).
    ///
    /// # Errors
    ///
    /// Exactly [`Self::project_hermitian`]'s.
    pub fn project_antihermitian(&self) -> Result<Self, Error> {
        self.add(&self.adjoint()?, D::from_real(0.5), D::from_real(-0.5))
    }

    /// Whether codomain and domain are the same product space — the
    /// precondition every member of the family above tests against, and the
    /// same comparison [`Self::tr`] makes.
    fn is_endomorphism(&self) -> bool {
        let hom = self.body.space.space().homspace();
        hom.codomain().legs() == hom.domain().legs()
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
        self.dense_data()
    }

    /// The dense coupled-sector payload, materializing a compact spectrum on
    /// first demand into the body's shared cache.
    ///
    /// Why infallible: the fill is total on a bond space this module built
    /// itself from that same spectrum, so a failure would be an engine
    /// invariant break rather than a caller mistake — the same judgement, and
    /// the same `expect`, as the erased `Tensor::materialize_diagonal`.
    fn dense_data(&self) -> &[D] {
        match &self.body.data {
            TypedData::Dense(data) => data,
            TypedData::Diagonal(spectrum) => self.body.dense_cache.get_or_init(|| {
                tenet_matrixalgebra::diagonal_bond_data(
                    self.body.space.space(),
                    spectrum,
                    &|value| value,
                )
                .expect("diagonal fill is total on the stored bond space")
            }),
        }
    }
}
