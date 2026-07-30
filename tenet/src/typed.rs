//! Provider-typed facade: spaces and tensor maps that keep the concrete
//! fusion-rule type `R` and speak the provider's own sector labels.
//!
//! The ergonomic [`crate::prelude`] facade erases the rule behind a fixed set
//! of built-ins. This module is its typed sibling: `R` stays concrete through
//! monomorphized construction, so any provider — including one defined
//! downstream — can drive it, and the categorical identity of a tensor comes
//! back as [`TypedSectorAdmission::Sector`] labels instead of opaque
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
//! The scope of that product-provider claim is:
//! `MultiplicityFreeRigidSymbols<Scalar = f64> + CheckedFusionAlgebra +
//! SectorCodec`. Complex categorical scalars and anyonic product operations
//! are issue #539.
//!
//! **`ProductSector` is not `ProductSpace`.** [`tenet_core::ProductSector`] is
//! a *sector label*: one irrep of a Deligne product category, TensorKit's
//! `ProductSector` (from `TensorKitSectors`).
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
//! ([`TensorMap::zeros`], [`TensorMap::from_block_fn`], [`TensorMap::id`],
//! [`TensorMap::rand`], [`TensorMap::rand_with_seed`],
//! [`TensorMap::isomorphism`], [`TensorMap::unitary`],
//! [`TensorMap::isometry`]),
//! inspection ([`TensorMap::codomain`], [`TensorMap::domain`],
//! [`TensorMap::block_fusion_trees`], [`TensorMap::block`],
//! [`TensorMap::block_count`], [`TensorMap::data`], [`TensorMap::runtime`]),
//! the index-manipulation and contraction operations
//! ([`TensorMap::permute`], [`TensorMap::braid`], [`TensorMap::transpose`],
//! [`TensorMap::transpose_axes`], [`TensorMap::repartition`],
//! [`TensorMap::contract`], [`TensorMap::contract_ordered`],
//! [`TensorMap::compose`]), the scalar operations
//! ([`TensorMap::add`], [`TensorMap::scale`], [`TensorMap::norm`],
//! [`TensorMap::norm_inf`], [`TensorMap::norm_p`], [`TensorMap::normalize`],
//! [`TensorMap::inner`],
//! [`TensorMap::dot`], [`TensorMap::tr`], [`TensorMap::trace_pairs`],
//! [`TensorMap::adjoint`]), the factorizations ([`TensorMap::svd_compact`],
//! [`TensorMap::svd_full`], [`TensorMap::svd_trunc`], [`TensorMap::svd_vals`],
//! [`TensorMap::qr_compact`], [`TensorMap::qr_full`],
//! [`TensorMap::lq_compact`], [`TensorMap::lq_full`],
//! [`TensorMap::left_polar`], [`TensorMap::right_polar`],
//! [`TensorMap::left_orth`], [`TensorMap::right_orth`],
//! [`TensorMap::left_null`], [`TensorMap::right_null`], with
//! [`GradedSpace::truncspace`] naming a fixed truncation target) and — with
//! issue #570
//! — the **eigendecompositions** ([`TensorMap::eigh_full`],
//! [`TensorMap::eigh_trunc`], [`TensorMap::eigh_vals`],
//! [`TensorMap::eig_full`], [`TensorMap::eig_trunc`], [`TensorMap::eig_vals`])
//! and the **`is_hermitian` / `project_*` family** ([`TensorMap::is_hermitian`],
//! [`TensorMap::is_antihermitian`], [`TensorMap::is_isometric`],
//! [`TensorMap::is_unitary`], [`TensorMap::is_posdef`],
//! [`TensorMap::project_hermitian`], [`TensorMap::project_antihermitian`]) and
//! — with issue #576 — the **matrix functions** ([`TensorMap::exp`],
//! [`TensorMap::inv`], [`TensorMap::pinv`], [`TensorMap::sqrt`]) and — with
//! issue #580 — the **typed inspection, scalar and conversion group**
//! ([`TensorMap::rank`], [`TensorMap::codomain_rank`],
//! [`TensorMap::domain_rank`] and their TensorKit aliases `numout` / `numin` /
//! `numind`, [`TensorMap::leg_dims`], [`TensorMap::leg_dim`],
//! [`TensorMap::codomain_spaces`], [`TensorMap::domain_spaces`],
//! [`TensorMap::scalar`], [`TensorMap::zeros_like`], [`TensorMap::to_c64`],
//! [`TensorMap::re`], [`TensorMap::im`]) and the **concatenation/absorb
//! group** ([`TensorMap::catdomain`], [`TensorMap::catcodomain`],
//! [`TensorMap::absorb`]) and the **index-unit group** ([`TensorMap::twist`],
//! [`TensorMap::flip`], [`TensorMap::insert_left_unit`],
//! [`TensorMap::insert_right_unit`], [`TensorMap::remove_unit`]).
//!
//! Issue #570 also gave the facade **compact diagonal storage**: a spectrum
//! factor — `svd_compact`'s and `svd_trunc`'s `s`, `eigh`/`eig`'s `d` — holds
//! `Σ_c k_c` values rather than the `Σ_c k_c²` block-diagonal buffer they would
//! fill, which is what TensorKit's `DiagonalTensorMap` is. It is a storage
//! property and not a type: no signature mentions it, [`TensorMap::data`] still
//! reports the dense buffer (materialized once, on demand, shared by every
//! clone), and the operations that can exploit it — [`TensorMap::compose`],
//! [`TensorMap::scale`], [`TensorMap::add`], [`TensorMap::adjoint`],
//! [`TensorMap::trace_pairs`] on its full-pair arm, and the reductions — do so
//! silently. The ones that cannot say so in their own
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
//! with a decision behind them rather than a queue position:
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
//! - **Outer multiplicity contractions and factorizations** remain outside this
//!   leaf. Checked `Generic` providers use the ordinary `permute`, `braid`, and
//!   `repartition` methods through their retained provider authority, but
//!   planar `transpose`, contractions, and factorizations still retain their
//!   multiplicity-free bounds.
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

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use tenet_core::{
    validate_unit_layout_correspondence_checked, BlockKey, BlockRef, CanonicalUnitFusionRule,
    CheckedFusionAlgebra, CheckedGenericAdmissionMode, CheckedGenericStructureError,
    FusionAlgebraError, FusionProductSpace, FusionTreeHomSpace, MultiplicityFreeAdmissionMode,
    MultiplicityFreeRigidSymbols, MultiplicityIndex, ProductFusionRule, ProductSector,
    ProductSectorCodec, SectorLeg, TypedSectorAdmission, UnitLegInsertion,
};
use tenet_core::{CheckedGenericFusion, CheckedGenericRigidSymbols};
use tenet_tensors::{
    tree_transform_dyn_owned_checked_generic, BoundDynamicFusionMapSpace, BoundDynamicTensorRef,
    DynamicFusionMapSpace, OutputAxisOrder, TreeTransformOperation,
};

pub use tenet_core::SectorCodec;
#[cfg(feature = "racah-generated")]
pub use tenet_core::{SUNFusionRule, SUNFusionRuleError};
pub use tenet_tensors::CheckedGenericPlanError;

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
    absorb_mapped, apply_fill, cat_homspace, check_flip_layout_identity, compile_cat_plan,
    coupled_region_pow_sum, flip_block_factor, flip_toggled_homspace, internal_layout_error,
    map_checked_unit_layout_error, reject_unbraided_nonunit_legs, scale_blocks_impl,
    sector_regions, twist_block_factor, twist_is_identity_over_blocks, validate_norm_p,
    weighted_inner, weighted_trace, with_planar_axes, CatOperandLayout, CatSide, Fill,
    PlanarRequestKind, TensorScalar,
};
use crate::tensor_core::{
    pow_by_squaring, tensorcompose_owned_multiplicity_free, tensorcontract_owned_multiplicity_free,
    tensorproduct_owned_multiplicity_free, tree_transform_owned_multiplicity_free,
};

/// Facade error for checked Generic providers.
#[non_exhaustive]
#[derive(Debug)]
pub enum GenericTensorError<E> {
    /// Ordinary facade validation or payload construction failed.
    Facade(Error),
    /// Checked Generic provider or structural admission failed.
    Structure(CheckedGenericStructureError<E>),
    /// Checked Generic operation-plan construction or replay failed.
    Plan(CheckedGenericPlanError<E>),
}

impl<E: core::fmt::Display> core::fmt::Display for GenericTensorError<E> {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Facade(error) => error.fmt(formatter),
            Self::Structure(error) => error.fmt(formatter),
            Self::Plan(error) => error.fmt(formatter),
        }
    }
}

impl<E: std::error::Error + 'static> std::error::Error for GenericTensorError<E> {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Facade(error) => Some(error),
            Self::Structure(error) => Some(error),
            Self::Plan(error) => Some(error),
        }
    }
}

impl<E> From<Error> for GenericTensorError<E> {
    fn from(error: Error) -> Self {
        Self::Facade(error)
    }
}

impl<E> From<CheckedGenericStructureError<E>> for GenericTensorError<E> {
    fn from(error: CheckedGenericStructureError<E>) -> Self {
        Self::Structure(error)
    }
}

impl<E> From<CheckedGenericPlanError<E>> for GenericTensorError<E> {
    fn from(error: CheckedGenericPlanError<E>) -> Self {
        Self::Plan(error)
    }
}

/// Tensor-side layout admission selected by a provider-owned mode.
#[doc(hidden)]
pub trait TypedTensorModeDispatch<R>: typed_admission_private::Sealed
where
    R: TypedSectorAdmission,
{
    /// Error returned by the ordinary typed facade.
    type FacadeError: std::error::Error + From<Error>;

    /// Preserves a provider-side admission error.
    fn map_provider_error(error: R::Error) -> Self::FacadeError;
}

/// Tensor-side root construction selected by a provider-owned mode.
#[doc(hidden)]
pub trait TypedTensorRootDispatch<R>: TypedTensorModeDispatch<R>
where
    R: TypedSectorAdmission,
{
    /// Builds one fully admitted root while retaining `provider`.
    fn build_root(
        provider: Arc<R>,
        homspace: FusionTreeHomSpace,
    ) -> Result<BoundDynamicFusionMapSpace<R>, Self::FacadeError>;
}

/// Tensor-side tree-transform execution selected by a provider-owned mode.
///
/// ```
/// use tenet::core::{
///     CheckedGenericAdmissionMode, CheckedGenericRigidSymbols, TypedSectorAdmission,
/// };
/// use tenet::typed::TensorMap;
///
/// fn checked_transpose<R>(tensor: &TensorMap<R, f64>)
/// where
///     R: TypedSectorAdmission<
///             Error = <R as tenet::core::CheckedGenericFusion>::Error,
///             Mode = CheckedGenericAdmissionMode,
///         > + CheckedGenericRigidSymbols<Scalar = f64>,
/// {
///     let _ = tensor.transpose();
/// }
/// ```
#[doc(hidden)]
pub trait TypedTensorTransformDispatch<R, D>: TypedTensorModeDispatch<R>
where
    R: TypedSectorAdmission,
    D: TensorScalar,
{
    /// Executes one admitted permutation, braid, or planar transpose.
    fn tree_transform(
        tensor: &TensorMap<R, D>,
        operation: TreeTransformOperation,
    ) -> Result<TensorMap<R, D>, Self::FacadeError>;
}

mod typed_admission_private {
    use super::{CheckedGenericAdmissionMode, MultiplicityFreeAdmissionMode};

    pub trait Sealed {}

    impl Sealed for MultiplicityFreeAdmissionMode {}
    impl Sealed for CheckedGenericAdmissionMode {}
}

impl<R> TypedTensorModeDispatch<R> for MultiplicityFreeAdmissionMode
where
    R: TypedSectorAdmission<Error = FusionAlgebraError, Mode = MultiplicityFreeAdmissionMode>,
{
    type FacadeError = Error;

    fn map_provider_error(error: <R as TypedSectorAdmission>::Error) -> Self::FacadeError {
        error.into()
    }
}

impl<R> TypedTensorRootDispatch<R> for MultiplicityFreeAdmissionMode
where
    R: TypedSectorAdmission<Error = FusionAlgebraError, Mode = MultiplicityFreeAdmissionMode>
        + MultiplicityFreeRigidSymbols<Scalar = f64>
        + CheckedFusionAlgebra
        + SectorCodec,
{
    fn build_root(
        provider: Arc<R>,
        homspace: FusionTreeHomSpace,
    ) -> Result<BoundDynamicFusionMapSpace<R>, Self::FacadeError> {
        BoundDynamicFusionMapSpace::from_final_homspace_multiplicity_free_checked(
            provider, homspace,
        )
        .map_err(Into::into)
    }
}

impl<R> TypedTensorModeDispatch<R> for CheckedGenericAdmissionMode
where
    R: TypedSectorAdmission<
            Error = <R as CheckedGenericFusion>::Error,
            Mode = CheckedGenericAdmissionMode,
        > + CheckedGenericFusion,
{
    type FacadeError = GenericTensorError<<R as CheckedGenericFusion>::Error>;

    fn map_provider_error(error: <R as TypedSectorAdmission>::Error) -> Self::FacadeError {
        GenericTensorError::Structure(CheckedGenericStructureError::Provider(error))
    }
}

impl<R> TypedTensorRootDispatch<R> for CheckedGenericAdmissionMode
where
    R: TypedSectorAdmission<
            Error = <R as CheckedGenericFusion>::Error,
            Mode = CheckedGenericAdmissionMode,
        > + CheckedGenericFusion,
{
    fn build_root(
        provider: Arc<R>,
        homspace: FusionTreeHomSpace,
    ) -> Result<BoundDynamicFusionMapSpace<R>, Self::FacadeError> {
        BoundDynamicFusionMapSpace::from_final_homspace_generic_checked(provider, homspace)
            .map_err(Into::into)
    }
}

impl<R, D> TypedTensorTransformDispatch<R, D> for MultiplicityFreeAdmissionMode
where
    R: TypedSectorAdmission<Error = FusionAlgebraError, Mode = MultiplicityFreeAdmissionMode>
        + MultiplicityFreeRigidSymbols<Scalar = f64>
        + CheckedFusionAlgebra
        + SectorCodec,
    D: TensorScalar,
{
    fn tree_transform(
        tensor: &TensorMap<R, D>,
        operation: TreeTransformOperation,
    ) -> Result<TensorMap<R, D>, Self::FacadeError> {
        tensor.tree_transform_multiplicity_free(operation)
    }
}

impl<R, D> TypedTensorTransformDispatch<R, D> for CheckedGenericAdmissionMode
where
    R: TypedSectorAdmission<
            Error = <R as CheckedGenericFusion>::Error,
            Mode = CheckedGenericAdmissionMode,
        > + CheckedGenericRigidSymbols<Scalar = f64>,
    D: TensorScalar,
{
    fn tree_transform(
        tensor: &TensorMap<R, D>,
        operation: TreeTransformOperation,
    ) -> Result<TensorMap<R, D>, Self::FacadeError> {
        let (space, data) = tree_transform_dyn_owned_checked_generic(
            operation,
            &tensor.body.space,
            tensor.dense_data(),
            D::from_real(1.0),
        )?;
        Ok(TensorMap {
            runtime: tensor.runtime.clone(),
            body: Arc::new(TypedTensorBody::dense(space, data)),
        })
    }
}

type TypedFacadeError<R> =
    <<R as TypedSectorAdmission>::Mode as TypedTensorModeDispatch<R>>::FacadeError;

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
    R: TypedSectorAdmission,
    R::Mode: TypedTensorModeDispatch<R>,
{
    /// Builds a leg from `(label, degeneracy)` pairs — TensorKit's
    /// `GradedSpace` / `Vect[I](c => d, ...; dual)` constructor family.
    ///
    /// **Dual-leg convention.** Labels are stored exactly as given —
    /// `is_dual` only marks the orientation, it never dualizes them. On a
    /// dual leg that is *not* what TensorKit's constructor-plus-read
    /// composition reports: `sectors(Vect[U1](1 => 1; dual = true))` is `-1`
    /// (TK dualizes stored keys on read), while
    /// `try_new(.., [(1, 1)], true)?.sectors()?` is `1`. A dual leg meant to
    /// agree with TensorKit must be built from pre-dualized labels, or via
    /// [`Self::try_dual`], which dualizes at construction.
    ///
    /// Order is irrelevant: the leg stores its sectors in the provider's
    /// [`tenet_core::SectorId`] order. A zero-degeneracy sector is absent from
    /// the result, matching the leg invariant of the erased facade.
    ///
    /// # Complexity
    ///
    /// `O(k log k)` in the number of pairs: one encode per label plus the two
    /// duplicate-detection sorts below; no payload is touched.
    ///
    /// # Errors
    ///
    /// Facade validation rejects duplicate labels and non-injective encodings;
    /// provider encode failures remain in the mode's facade error. Legacy
    /// multiplicity-free providers use [`Error`], while checked Generic
    /// providers retain their typed structure/provider error.
    pub fn try_new<Pairs>(
        provider: Arc<R>,
        pairs: Pairs,
        is_dual: bool,
    ) -> Result<Self, TypedFacadeError<R>>
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
            ))
            .into());
        }

        let mut encoded = Vec::with_capacity(pairs.len());
        for (label, degeneracy) in &pairs {
            encoded.push((
                TypedSectorAdmission::try_encode_label(provider.as_ref(), label)
                    .map_err(<R::Mode as TypedTensorModeDispatch<R>>::map_provider_error)?,
                label,
                *degeneracy,
            ));
        }
        let mut by_id: Vec<_> = encoded.iter().collect();
        by_id.sort_unstable_by_key(|(id, _, _)| *id);
        if let Some(window) = by_id.windows(2).find(|window| window[0].0 == window[1].0) {
            return Err(Error::InvalidArgument(format!(
                "SectorCodec law violation: labels {:?} and {:?} both encode to {:?}",
                window[0].1, window[1].1, window[0].0
            ))
            .into());
        }

        let leg = SectorLeg::try_new(
            encoded.iter().map(|(id, _, degeneracy)| (*id, *degeneracy)),
            is_dual,
        )
        .map_err(|error| TypedFacadeError::<R>::from(Error::InvalidArgument(error.to_string())))?;
        Ok(Self { provider, leg })
    }

    /// The sector labels carried by this leg, in the provider's
    /// [`tenet_core::SectorId`] order — TensorKit `sectors(V)`, with one convention difference:
    /// the stored labels are returned as-is, never dualized on read, while
    /// TensorKit dualizes stored keys when `isdual(V)`. A leg from
    /// [`Self::try_dual`] already stores dual labels, so there the two
    /// surfaces agree; a leg built by [`Self::try_new`] with `is_dual =
    /// true` from non-pre-dualized labels does not (see the dual-leg
    /// convention there). One decode per sector, one `Vec` allocation per
    /// call.
    ///
    /// The order is the engine's, deliberately: it is the order of
    /// [`Self::degeneracies`] and of every block layout derived from the leg,
    /// so re-sorting by label here would desynchronize the two. A caller that
    /// wants label order sorts the returned vector.
    ///
    /// # Errors
    ///
    /// Provider decode failures remain in the mode's facade error; decoding an id
    /// previously produced by the same provider is required to be total.
    pub fn sectors(&self) -> Result<Vec<R::Sector>, TypedFacadeError<R>> {
        self.leg
            .sectors()
            .iter()
            .map(|&id| {
                TypedSectorAdmission::try_decode_label(self.provider.as_ref(), id)
                    .map_err(<R::Mode as TypedTensorModeDispatch<R>>::map_provider_error)
            })
            .collect()
    }

    /// The conjugate leg: every sector replaced by its dual (degeneracies
    /// carried along) and the dual flag flipped — TensorKit `dual(V)` / `V'`,
    /// which must satisfy the `dual(dual(V)) == V` contract of TensorKit's
    /// `dual(::VectorSpace)`. TensorKit only flips the flag and
    /// dualizes labels lazily on read; this leg rewrites its stored sector
    /// table eagerly — `O(k log k)`, one provider dual per sector plus the
    /// leg constructor's re-sort — and [`Self::sectors`] then reports the
    /// dual labels just as TK's `sectors(V')` does, provided the source
    /// leg's stored labels were its external content (see
    /// [`Self::try_new`]'s dual-leg convention).
    ///
    /// # Errors
    ///
    /// Provider dual failures remain in the mode's facade error. A dual that
    /// collapses two sectors is reported as facade validation failure. No
    /// partially dualized leg is produced in either case.
    pub fn try_dual(&self) -> Result<Self, TypedFacadeError<R>> {
        let sectors = self
            .leg
            .sectors()
            .iter()
            .copied()
            .zip(self.leg.degeneracies().iter().copied())
            .map(|(sector, degeneracy)| {
                TypedSectorAdmission::try_dual_id(self.provider.as_ref(), sector)
                    .map_err(<R::Mode as TypedTensorModeDispatch<R>>::map_provider_error)
                    .map(|dual| (dual, degeneracy))
            })
            .collect::<Result<Vec<_>, _>>()?;
        let mut duals: Vec<_> = sectors.iter().map(|(sector, _)| *sector).collect();
        duals.sort_unstable();
        if let Some(duplicate) = duals.windows(2).find(|pair| pair[0] == pair[1]) {
            return Err(Error::InvalidArgument(format!(
                "dual map is not injective: sector {:?} appears multiple times",
                duplicate[0]
            ))
            .into());
        }
        let leg = SectorLeg::try_new(sectors, !self.leg.is_dual()).map_err(|error| {
            TypedFacadeError::<R>::from(Error::InvalidArgument(error.to_string()))
        })?;
        Ok(Self {
            provider: Arc::clone(&self.provider),
            leg,
        })
    }

    /// This leg read as a fixed truncation target: TensorKit `truncspace(V)`.
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
            TypedSectorAdmission::typed_rule_identity(self.provider.as_ref()),
            self.leg
                .sectors()
                .iter()
                .copied()
                .zip(self.leg.degeneracies().iter().copied()),
        )
    }
}

impl<R> GradedSpace<R> {
    /// Per-sector degeneracies, parallel to semantic sector readback.
    #[inline]
    pub fn degeneracies(&self) -> &[usize] {
        self.leg.degeneracies()
    }

    /// Whether this is the conjugate space (TensorKit's `V'`).
    #[inline]
    pub fn is_dual(&self) -> bool {
        self.leg.is_dual()
    }

    /// The provider bound to this leg.
    #[inline]
    pub fn provider(&self) -> &R {
        self.provider.as_ref()
    }

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
/// Vertex labels are part of the identity for Generic providers: two trees
/// can have identical sector labels and differ only by outer multiplicity.
#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct BlockFusionTrees<S> {
    coupled: S,
    codomain_uncoupled: Vec<S>,
    codomain_innerlines: Vec<S>,
    codomain_vertices: Vec<MultiplicityIndex>,
    domain_uncoupled: Vec<S>,
    domain_innerlines: Vec<S>,
    domain_vertices: Vec<MultiplicityIndex>,
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

    /// Codomain outer-multiplicity labels, in fusion-vertex order.
    #[inline]
    pub fn codomain_vertices(&self) -> &[MultiplicityIndex] {
        &self.codomain_vertices
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

    /// Domain outer-multiplicity labels, in fusion-vertex order.
    #[inline]
    pub fn domain_vertices(&self) -> &[MultiplicityIndex] {
        &self.domain_vertices
    }
}

fn decode_sectors<R>(
    provider: &R,
    ids: &[tenet_core::SectorId],
) -> Result<Vec<R::Sector>, TypedFacadeError<R>>
where
    R: TypedSectorAdmission,
    R::Mode: TypedTensorModeDispatch<R>,
{
    ids.iter()
        .map(|&id| {
            TypedSectorAdmission::try_decode_label(provider, id)
                .map_err(<R::Mode as TypedTensorModeDispatch<R>>::map_provider_error)
        })
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
) -> Result<BlockFusionTrees<R::Sector>, TypedFacadeError<R>>
where
    R: TypedSectorAdmission,
    R::Mode: TypedTensorModeDispatch<R>,
{
    let pair = key.as_fusion_tree_pair().ok_or_else(|| {
        TypedFacadeError::<R>::from(Error::InvalidArgument(format!(
            "block key is {}, not a fusion-tree pair",
            key.kind()
        )))
    })?;
    let codomain = pair.codomain_tree();
    let domain = pair.domain_tree();
    Ok(BlockFusionTrees {
        coupled: TypedSectorAdmission::try_decode_label(provider, codomain.coupled())
            .map_err(<R::Mode as TypedTensorModeDispatch<R>>::map_provider_error)?,
        codomain_uncoupled: decode_sectors(provider, codomain.uncoupled())?,
        codomain_innerlines: decode_sectors(provider, codomain.innerlines())?,
        codomain_vertices: codomain.vertices().to_vec(),
        domain_uncoupled: decode_sectors(provider, domain.uncoupled())?,
        domain_innerlines: decode_sectors(provider, domain.innerlines())?,
        domain_vertices: domain.vertices().to_vec(),
    })
}

fn map_block_fusion_trees<A, B>(
    source: &BlockFusionTrees<A>,
    map: impl Fn(&A) -> B,
) -> BlockFusionTrees<B> {
    BlockFusionTrees {
        coupled: map(&source.coupled),
        codomain_uncoupled: source.codomain_uncoupled.iter().map(&map).collect(),
        codomain_innerlines: source.codomain_innerlines.iter().map(&map).collect(),
        codomain_vertices: source.codomain_vertices.clone(),
        domain_uncoupled: source.domain_uncoupled.iter().map(&map).collect(),
        domain_innerlines: source.domain_innerlines.iter().map(map).collect(),
        domain_vertices: source.domain_vertices.clone(),
    }
}

struct PreparedProductOperand<'a, S, P, D>
where
    P: SectorCodec,
{
    source: &'a TensorMap<S, D>,
    codomain: Vec<GradedSpace<P>>,
    domain: Vec<GradedSpace<P>>,
    blocks: HashMap<BlockFusionTrees<P::Sector>, (usize, Vec<usize>, Vec<usize>)>,
}

fn prepare_product_operand<S, P, D>(
    source: &TensorMap<S, D>,
    provider: Arc<P>,
    embed: impl Fn(S::Sector) -> P::Sector,
    project: impl Fn(&P::Sector) -> S::Sector,
) -> Result<PreparedProductOperand<'_, S, P, D>, Error>
where
    S: MultiplicityFreeRigidSymbols<Scalar = f64>
        + CheckedFusionAlgebra
        + SectorCodec
        + CanonicalUnitFusionRule,
    P: MultiplicityFreeRigidSymbols<Scalar = f64>
        + CheckedFusionAlgebra
        + SectorCodec
        + CanonicalUnitFusionRule,
    D: TensorScalar,
{
    let embed_legs = |legs: Vec<GradedSpace<S>>| -> Result<Vec<GradedSpace<P>>, Error> {
        legs.into_iter()
            .map(|leg| {
                let sectors = leg.sectors()?;
                GradedSpace::try_new(
                    Arc::clone(&provider),
                    sectors
                        .into_iter()
                        .zip(leg.degeneracies().iter().copied())
                        .map(|(sector, degeneracy)| (embed(sector), degeneracy)),
                    leg.is_dual(),
                )
            })
            .collect()
    };
    let codomain = embed_legs(source.codomain())?;
    let domain = embed_legs(source.domain())?;
    let homspace = FusionTreeHomSpace::new(
        FusionProductSpace::new(codomain.iter().map(|leg| leg.leg().clone())),
        FusionProductSpace::new(domain.iter().map(|leg| leg.leg().clone())),
    );
    let mut blocks = HashMap::with_capacity(source.block_count());
    for index in 0..source.block_count() {
        let key = source.block_fusion_trees(index)?;
        let block = source.block(index)?;
        blocks.insert(
            key,
            (
                block.offset(),
                block.shape().to_vec(),
                block.strides().to_vec(),
            ),
        );
    }
    // Stage (do not publish) the product layout and prove that canonical-unit
    // projection is a bijection onto the source blocks. The callback below can
    // therefore not discover `missing_block` only after target admission.
    let prepared = homspace.prepare_fusion_tree_layout_checked(provider.as_ref())?;
    let mut projected = HashSet::with_capacity(prepared.keys().len());
    let mut target_blocks = HashMap::with_capacity(prepared.keys().len());
    for key in prepared.keys() {
        let labelled =
            decode_block_fusion_trees(provider.as_ref(), &BlockKey::FusionTree(key.clone()))?;
        let source_key = map_block_fusion_trees(&labelled, &project);
        let Some(layout) = blocks.get(&source_key).cloned() else {
            return Err(Error::InvalidArgument(
                "canonical-unit product embedding did not preserve source blocks".to_string(),
            ));
        };
        if !projected.insert(source_key) || target_blocks.insert(labelled, layout).is_some() {
            return Err(Error::InvalidArgument(
                "canonical-unit product embedding did not preserve source blocks".to_string(),
            ));
        }
    }
    if projected.len() != blocks.len() {
        return Err(Error::InvalidArgument(
            "canonical-unit product embedding did not preserve source blocks".to_string(),
        ));
    }

    Ok(PreparedProductOperand {
        source,
        codomain,
        domain,
        blocks: target_blocks,
    })
}

impl<S, P, D> PreparedProductOperand<'_, S, P, D>
where
    S: MultiplicityFreeRigidSymbols<Scalar = f64> + CheckedFusionAlgebra + SectorCodec,
    P: MultiplicityFreeRigidSymbols<Scalar = f64>
        + CheckedFusionAlgebra
        + SectorCodec
        + CanonicalUnitFusionRule,
    D: TensorScalar,
{
    fn commit(self) -> Result<TensorMap<P, D>, Error> {
        let data = self.source.dense_data();
        let blocks = self.blocks;
        let source = self.source;
        let codomain = self.codomain;
        let domain = self.domain;
        let mut missing_block = false;
        let built = TensorMap::from_block_fn(
            source.runtime(),
            codomain.iter(),
            domain.iter(),
            |key, indices| {
                let Some((offset, shape, strides)) = blocks.get(key) else {
                    missing_block = true;
                    return D::from_real(0.0);
                };
                if indices.len() != shape.len()
                    || indices.iter().zip(shape).any(|(&index, &dim)| index >= dim)
                {
                    missing_block = true;
                    return D::from_real(0.0);
                }
                let position = indices
                    .iter()
                    .zip(strides)
                    .fold(*offset, |position, (&index, &stride)| {
                        position + index * stride
                    });
                data[position]
            },
        )?;
        if missing_block {
            return Err(Error::InvalidArgument(
                "canonical-unit product embedding did not preserve a source block".to_string(),
            ));
        }
        Ok(built)
    }
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
    /// That sector's values. Public diagonal construction preserves this order;
    /// factorization outputs separately use descending magnitude order.
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
        body: Arc::new(TypedTensorBody::diagonal(space, data)),
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
    /// Why the payload carries its own reference count rather than sitting
    /// inline in the body: an operation that rewrites only the *space* and
    /// leaves every stored value where it is — inserting or removing a unit
    /// leg, whose trivial sector adds no block and reorders nothing — must be
    /// an O(1) metadata edit. Inline, such an operation would have to copy the
    /// whole `Vec<D>` to build a body with the new space; behind this `Arc` it
    /// clones a pointer. That O(1)-reuse argument holds for **dense payloads
    /// only**: `TypedData::Diagonal` is a bond-space-only representation, and
    /// reusing one under a rewritten non-bond space would leave `spectrum()`
    /// returning `Some` and send `exp`/`inv`/`pinv`/`scale` down their compact
    /// elementwise arms on a tensor that is no longer an endomorphism. The
    /// Group 4 (#580) contract is therefore: materialize a compact payload to
    /// dense *before* changing the space, exactly as the references do —
    /// TensorKit 0.17 shares `t.data` only for ordinary `TensorMap` and routes
    /// `DiagonalTensorMap` through the generic similar+block-copy branch
    /// (`src/tensors/indexmanipulations.jl:124-136,158-195`), and the erased
    /// facade materializes `Data::Diagonal` first and only then shares the
    /// resulting `Arc<Data>` (`tenet/src/tensor.rs`
    /// `materialized_dense_data_arc`). The Group 4 slice (#580 PR 5) holds
    /// that contract in [`TensorMap::shareable_dense_payload`]: a dense
    /// payload is shared at pointer cost, a compact one is materialized into
    /// a *fresh* dense payload (one copy) — never by sharing the body-local
    /// `dense_cache`, which only lends a borrowed slice tied to this body's
    /// space/payload pairing. And `TypedData::Diagonal` must not be
    /// broadened to non-bond spaces without separately proving every compact
    /// fast path (`spectrum`/`exp`/`inv`/`pinv`/`scale`).
    /// Clone-then-modify keeps the same property from the
    /// other side: two *bodies* can share one payload until one of them
    /// writes (the unit-leg operations build new bodies over old dense
    /// payloads; every write route publishes a new payload instead of
    /// reaching through the `Arc`). This is also parity, not invention: the erased sibling's
    /// `tensor.rs` `TensorBody { space: Arc<..>, data: Arc<Data> }` has had
    /// exactly this two-`Arc` layout all along, so typed converges onto the
    /// established in-repo shape.
    data: Arc<TypedData<D>>,
    /// Materialization of a [`TypedData::Diagonal`] payload into the dense
    /// coupled layout, computed at most once and shared by every clone of this
    /// body — the erased sibling's `compact_dense` cache, without its hand-copy
    /// on each `Tensor` value. Never populated for a dense payload.
    ///
    /// Deliberately *not* inside the payload `Arc`: the materialized buffer is
    /// a function of the payload **and** the space it is laid out on, so it
    /// belongs to the body that owns that pairing — any body sharing the
    /// payload starts from a cold cache and materializes for itself. (Reusing
    /// a `Diagonal` payload under a *different* space is not a scenario this
    /// placement serves: that reuse is forbidden outright — see the `data`
    /// field rationale on the Group 4 contract.)
    dense_cache: std::sync::OnceLock<Vec<D>>,
}

impl<R, D> TypedTensorBody<R, D> {
    /// A body holding an already-dense payload.
    fn dense(space: BoundDynamicFusionMapSpace<R>, data: Vec<D>) -> Self {
        Self::new(space, TypedData::Dense(data))
    }

    /// A body holding a compact spectrum payload.
    fn diagonal(
        space: BoundDynamicFusionMapSpace<R>,
        spectrum: Vec<tenet_matrixalgebra::SectorSpectrum<D>>,
    ) -> Self {
        Self::new(space, TypedData::Diagonal(spectrum))
    }

    fn new(space: BoundDynamicFusionMapSpace<R>, data: TypedData<D>) -> Self {
        Self {
            space,
            data: Arc::new(data),
            dense_cache: std::sync::OnceLock::new(),
        }
    }

    /// A body installing an already-shared payload under a (usually
    /// rewritten) space — the unit-leg operations' O(1) dense reuse
    /// (#580 PR 5). The cache starts cold on purpose: it belongs to the
    /// body's own space/payload pairing (see the `dense_cache` rationale).
    fn with_shared_payload(space: BoundDynamicFusionMapSpace<R>, data: Arc<TypedData<D>>) -> Self {
        Self {
            space,
            data,
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
                &match &*self.body.data {
                    TypedData::Dense(data) => data.len(),
                    TypedData::Diagonal(spectrum) => {
                        spectrum.iter().map(|entry| entry.values.len()).sum()
                    }
                },
            )
            .finish_non_exhaustive()
    }
}

impl<R, D> TensorMap<R, D> {
    /// The provider allocation that owns this tensor's categorical layout.
    #[inline]
    pub fn provider(&self) -> &R {
        self.body.space.provider()
    }

    /// Number of codomain legs.
    #[inline]
    pub fn codomain_rank(&self) -> usize {
        self.body.space.space().homspace().codomain().len()
    }

    /// Number of domain legs.
    #[inline]
    pub fn domain_rank(&self) -> usize {
        self.body.space.space().homspace().domain().len()
    }

    /// Total number of legs.
    #[inline]
    pub fn rank(&self) -> usize {
        self.codomain_rank() + self.domain_rank()
    }

    /// TensorKit-compatible alias for [`Self::codomain_rank`].
    #[inline]
    pub fn numout(&self) -> usize {
        self.codomain_rank()
    }

    /// TensorKit-compatible alias for [`Self::domain_rank`].
    #[inline]
    pub fn numin(&self) -> usize {
        self.domain_rank()
    }

    /// TensorKit-compatible alias for [`Self::rank`].
    #[inline]
    pub fn numind(&self) -> usize {
        self.rank()
    }

    /// Engine-level layout view of one stored block.
    pub fn block(&self, index: usize) -> Result<BlockRef<'_>, Error> {
        self.body
            .space
            .space()
            .structure()
            .block(index)
            .map_err(Error::from)
    }

    /// Runtime bound to this tensor map.
    #[inline]
    pub fn runtime(&self) -> &Runtime {
        &self.runtime
    }

    /// Number of stored fusion-tree blocks.
    #[inline]
    pub fn block_count(&self) -> usize {
        self.body.space.space().structure().block_count()
    }
}

impl<R, D> TensorMap<R, D>
where
    D: TensorScalar,
{
    /// Whole dense payload in storage order.
    #[inline]
    pub fn data(&self) -> &[D] {
        self.dense_data()
    }

    fn dense_data(&self) -> &[D] {
        match &*self.body.data {
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

impl<R, D> TensorMap<R, D>
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64> + CheckedFusionAlgebra + SectorCodec,
    D: TensorScalar,
{
    /// Builds a compact diagonal map `bond <- bond` from labelled sector values.
    ///
    /// TeNeT's typed counterpart of TensorKit `DiagonalTensorMap` / `diagm`
    /// stores `O(Σ_c k_c)` values. Input labels may be permuted, but must name
    /// every nonzero bond sector exactly once; output is canonicalized to the
    /// bond's engine-sector order. Each vector must equal that sector's
    /// degeneracy. All validation precedes checked layout admission, and the
    /// supplied dual flag is preserved.
    pub fn diagonal<I>(runtime: &Runtime, bond: &GradedSpace<R>, spectra: I) -> Result<Self, Error>
    where
        I: IntoIterator<Item = SectorSpectrum<R::Sector, D>>,
    {
        let mut supplied = HashMap::new();
        for entry in spectra {
            let sector = bond.provider().encode_sector(&entry.sector)?;
            if supplied.insert(sector, entry.values).is_some() {
                return Err(Error::InvalidArgument(
                    "diagonal spectrum has a duplicate sector".into(),
                ));
            }
        }
        let mut spectrum = Vec::with_capacity(bond.degeneracies().len());
        for (&sector, &degeneracy) in bond.leg().sectors().iter().zip(bond.degeneracies()) {
            let values = supplied.remove(&sector).ok_or_else(|| {
                Error::InvalidArgument("diagonal spectrum is missing a bond sector".into())
            })?;
            if values.len() != degeneracy {
                return Err(Error::InvalidArgument(
                    "diagonal spectrum length does not match bond degeneracy".into(),
                ));
            }
            spectrum.push(tenet_matrixalgebra::SectorSpectrum { sector, values });
        }
        if !supplied.is_empty() {
            return Err(Error::InvalidArgument(
                "diagonal spectrum contains an unknown bond sector".into(),
            ));
        }
        let space = Self::build_space(Arc::clone(bond.provider_arc()), &[bond], &[bond])?;
        Ok(Self {
            runtime: runtime.clone(),
            body: Arc::new(TypedTensorBody::diagonal(space, spectrum)),
        })
    }

    /// Returns the compact diagonal spectrum without materializing dense data.
    ///
    /// This is typed `diag` readback: [`None`] means the tensor is dense;
    /// otherwise it clones only the `O(Σ_c k_c)` compact values in canonical
    /// bond-sector order.
    pub fn diagonal_spectrum(&self) -> Result<Option<Vec<SectorSpectrum<R::Sector, D>>>, Error> {
        self.spectrum()
            .map(|spectrum| {
                spectrum
                    .iter()
                    .map(|entry| {
                        Ok(SectorSpectrum {
                            sector: self.body.space.provider().decode_sector(entry.sector)?,
                            values: entry.values.clone(),
                        })
                    })
                    .collect()
            })
            .transpose()
    }

    /// Tests a rank-one map for blockwise diagonality without materializing compact storage.
    ///
    /// This matches TensorKit `isdiag` for finite data at `tol = 0`; positive
    /// tolerance uses `max_offdiag <= tol * max(norm_inf, 1)`. Negative and
    /// non-finite tolerances are rejected before every shortcut.
    pub fn is_diagonal(&self, tol: f64) -> Result<bool, Error> {
        if !tol.is_finite() || tol < 0.0 {
            return Err(Error::InvalidArgument(
                "diagonal tolerance must be finite and nonnegative".into(),
            ));
        }
        if self.spectrum().is_some() {
            return Ok(true);
        }
        if self.rank() != 2 || self.numout() != 1 || self.numin() != 1 {
            return Ok(false);
        }
        let data = self.dense_data();
        let mut norm = 0.0_f64;
        let mut offdiag = 0.0_f64;
        for index in 0..self.body.space.space().structure().block_count() {
            let block = self.body.space.space().structure().block(index)?;
            for row in 0..block.shape()[0] {
                for col in 0..block.shape()[1] {
                    let value = data
                        [block.offset() + row * block.strides()[0] + col * block.strides()[1]]
                        .widen_complex()
                        .norm();
                    norm = norm.max(value);
                    if row != col {
                        if !value.is_finite() {
                            return Ok(false);
                        }
                        offdiag = offdiag.max(value);
                    }
                }
            }
        }
        Ok(offdiag <= tol * norm.max(1.0))
    }
}

impl<R, D> TensorMap<R, D>
where
    R: TypedSectorAdmission,
    R::Mode: TypedTensorRootDispatch<R>,
    D: TensorScalar,
{
    /// Returns the first leg's provider allocation after proving that every
    /// leg has the same categorical identity.
    fn authority<'a>(legs: &[&'a GradedSpace<R>]) -> Result<&'a Arc<R>, TypedFacadeError<R>> {
        let (first, rest) = legs.split_first().ok_or_else(|| {
            TypedFacadeError::<R>::from(Error::InvalidArgument(
                "at least one leg is required to infer the fusion provider".into(),
            ))
        })?;
        // Equal identities, rather than pointer equality, let separately
        // allocated providers interoperate. This guard precedes every provider
        // query and layout-admission stage.
        let expected_identity = TypedSectorAdmission::typed_rule_identity(first.provider());
        for leg in rest {
            let actual_identity = TypedSectorAdmission::typed_rule_identity(leg.provider());
            if actual_identity != expected_identity {
                return Err(TypedFacadeError::<R>::from(Error::RuleMismatch));
            }
        }
        Ok(first.provider_arc())
    }

    /// Validation half of [`Self::build`]: admits the complete bound layout
    /// without touching payload or runtime RNG state.
    fn build_space(
        provider: Arc<R>,
        codomain: &[&GradedSpace<R>],
        domain: &[&GradedSpace<R>],
    ) -> Result<BoundDynamicFusionMapSpace<R>, TypedFacadeError<R>> {
        let homspace = FusionTreeHomSpace::new(
            FusionProductSpace::new(codomain.iter().map(|leg| leg.leg().clone())),
            FusionProductSpace::new(domain.iter().map(|leg| leg.leg().clone())),
        );
        <R::Mode as TypedTensorRootDispatch<R>>::build_root(provider, homspace)
    }

    /// Payload half of [`Self::build`]: fills only a fully admitted layout and
    /// validates its final data length before publication.
    fn fill_space(
        runtime: &Runtime,
        space: BoundDynamicFusionMapSpace<R>,
        fill: Fill<'_, D>,
    ) -> Result<Self, TypedFacadeError<R>> {
        let data = apply_fill(space.space(), fill).map_err(TypedFacadeError::<R>::from)?;
        BoundDynamicTensorRef::try_new(&space, &data)
            .map_err(Error::from)
            .map_err(TypedFacadeError::<R>::from)?;
        Ok(Self {
            runtime: runtime.clone(),
            body: Arc::new(TypedTensorBody::dense(space, data)),
        })
    }

    fn build(
        runtime: &Runtime,
        provider: Arc<R>,
        codomain: &[&GradedSpace<R>],
        domain: &[&GradedSpace<R>],
        fill: Fill<'_, D>,
    ) -> Result<Self, TypedFacadeError<R>> {
        let space = Self::build_space(provider, codomain, domain)?;
        Self::fill_space(runtime, space, fill)
    }

    /// Zero tensor map on `codomain <- domain` (TensorKit `zeros(T, W <- V)`).
    ///
    /// Every leg must have the same rule identity; the first leg's exact
    /// provider allocation becomes the tensor's authority. Identity mismatch
    /// is reported before provider algebra is queried. Layout admission and
    /// payload validation are transactional: failure publishes no tensor.
    ///
    /// # Complexity
    ///
    /// One fusion-tree layout admission plus one `O(stored_len)` zeroed
    /// payload allocation.
    pub fn zeros<'a, Codomain, Domain>(
        runtime: &Runtime,
        codomain: Codomain,
        domain: Domain,
    ) -> Result<Self, TypedFacadeError<R>>
    where
        Codomain: IntoIterator<Item = &'a GradedSpace<R>>,
        Domain: IntoIterator<Item = &'a GradedSpace<R>>,
        R: 'a,
    {
        let codomain: Vec<_> = codomain.into_iter().collect();
        let domain: Vec<_> = domain.into_iter().collect();
        let legs: Vec<_> = codomain.iter().chain(&domain).copied().collect();
        let provider = Arc::clone(Self::authority(&legs)?);
        Self::build(runtime, provider, &codomain, &domain, Fill::Zeros)
    }

    /// Tensor map whose every symmetry-allowed element is produced by
    /// `fill(sectors, indices)`.
    ///
    /// All block keys are decoded exactly once before the first callback.
    /// Therefore a late decode failure invokes `fill` zero times and publishes
    /// neither a partial payload nor a tensor.
    ///
    /// **No TensorKit counterpart.** TensorKit builds an uninitialized,
    /// zeroed, or random tensor and then mutates its blocks; this labelled
    /// callback is TeNeT's semantic-fixture constructor.
    ///
    /// # Complexity
    ///
    /// One layout admission, one decode per stored block, and one callback per
    /// stored element.
    pub fn from_block_fn<'a, Codomain, Domain, F>(
        runtime: &Runtime,
        codomain: Codomain,
        domain: Domain,
        mut fill: F,
    ) -> Result<Self, TypedFacadeError<R>>
    where
        Codomain: IntoIterator<Item = &'a GradedSpace<R>>,
        Domain: IntoIterator<Item = &'a GradedSpace<R>>,
        F: FnMut(&BlockFusionTrees<R::Sector>, &[usize]) -> D,
        R: 'a,
    {
        let codomain: Vec<_> = codomain.into_iter().collect();
        let domain: Vec<_> = domain.into_iter().collect();
        let legs: Vec<_> = codomain.iter().chain(&domain).copied().collect();
        let provider = Arc::clone(Self::authority(&legs)?);
        let space = Self::build_space(Arc::clone(&provider), &codomain, &domain)?;
        let structure = space.space().structure();
        let mut labelled = HashMap::with_capacity(structure.block_count());
        for index in 0..structure.block_count() {
            let key = structure
                .block(index)
                .map_err(Error::from)
                .map_err(TypedFacadeError::<R>::from)?
                .key()
                .clone();
            let decoded = decode_block_fusion_trees(provider.as_ref(), &key)?;
            labelled.insert(key, decoded);
        }
        let mut callback = |key: &BlockKey, indices: &[usize]| {
            fill(
                labelled
                    .get(key)
                    .expect("all admitted block keys were decoded before payload fill"),
                indices,
            )
        };
        Self::fill_space(runtime, space, Fill::BlockFn(&mut callback))
    }

    /// Random tensor map using the runtime's deterministic splitmix64 stream.
    ///
    /// The stream position is drawn only after every fallible identity,
    /// provider, shape, and layout-admission stage succeeds. A failed call
    /// therefore does not advance the runtime RNG or shift later seedless
    /// results.
    ///
    /// # Complexity
    ///
    /// One layout admission and one `O(stored_len)` payload allocation/fill.
    pub fn rand<'a, Codomain, Domain>(
        runtime: &Runtime,
        codomain: Codomain,
        domain: Domain,
    ) -> Result<Self, TypedFacadeError<R>>
    where
        Codomain: IntoIterator<Item = &'a GradedSpace<R>>,
        Domain: IntoIterator<Item = &'a GradedSpace<R>>,
        R: 'a,
    {
        let codomain: Vec<_> = codomain.into_iter().collect();
        let domain: Vec<_> = domain.into_iter().collect();
        let legs: Vec<_> = codomain.iter().chain(&domain).copied().collect();
        let provider = Arc::clone(Self::authority(&legs)?);
        let space = Self::build_space(provider, &codomain, &domain)?;
        Self::fill_space(runtime, space, Fill::Rand(runtime.next_rand_seed()))
    }

    /// Random tensor map using an explicit deterministic splitmix64 seed.
    ///
    /// Reproducibility is defined for the same TeNeT version and layout;
    /// semantic cross-version fixtures should use [`Self::from_block_fn`].
    ///
    /// # Complexity
    ///
    /// One layout admission and one `O(stored_len)` payload allocation/fill.
    pub fn rand_with_seed<'a, Codomain, Domain>(
        runtime: &Runtime,
        codomain: Codomain,
        domain: Domain,
        seed: u64,
    ) -> Result<Self, TypedFacadeError<R>>
    where
        Codomain: IntoIterator<Item = &'a GradedSpace<R>>,
        Domain: IntoIterator<Item = &'a GradedSpace<R>>,
        R: 'a,
    {
        let codomain: Vec<_> = codomain.into_iter().collect();
        let domain: Vec<_> = domain.into_iter().collect();
        let legs: Vec<_> = codomain.iter().chain(&domain).copied().collect();
        let provider = Arc::clone(Self::authority(&legs)?);
        Self::build(runtime, provider, &codomain, &domain, Fill::Rand(seed))
    }

    /// Provider-labelled fusion trees for one stored block.
    pub fn block_fusion_trees(
        &self,
        index: usize,
    ) -> Result<BlockFusionTrees<R::Sector>, TypedFacadeError<R>> {
        let block = self
            .body
            .space
            .space()
            .structure()
            .block(index)
            .map_err(Error::from)
            .map_err(TypedFacadeError::<R>::from)?;
        decode_block_fusion_trees(self.body.space.provider(), block.key())
    }
}

impl<R, D> TensorMap<R, D>
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64> + CheckedFusionAlgebra + SectorCodec,
    D: TensorScalar,
{
    /// The fused external sector content of one side of a structural
    /// constructor — the typed counterpart of the erased `Space::fuse_all`
    /// (TensorKit `fuse`, `spaces/gradedspace.jl:150-158`), on the one shared
    /// provider-generic fold. Duality is dropped exactly as there: stored
    /// sector content is already external.
    ///
    /// No SU(3) `UnsupportedForRule` guard, unlike the erased `Space::fuse`:
    /// the typed facade's `R` bound is `MultiplicityFreeRigidSymbols`, which
    /// the SU(3) provider does not implement, so an SU(3) rule cannot reach
    /// this fold from the typed surface at all — the guard would be dead
    /// code here. A multiplicity-carrying *external* provider is likewise
    /// excluded by the same bound; the fold's `N`-symbol weighting is still
    /// correct for it, so nothing depends on the exclusion.
    fn fused_content(
        provider: &R,
        legs: &[&GradedSpace<R>],
    ) -> Result<Vec<(tenet_core::SectorId, usize)>, Error> {
        let (first, rest) = legs.split_first().ok_or_else(|| {
            // Same class as the erased `Space::fuse_all` on an empty side.
            Error::InvalidArgument("fuse_all needs at least one space".into())
        })?;
        let mut fused: Vec<(tenet_core::SectorId, usize)> = first.leg().iter().collect();
        for leg in rest {
            let pairs: Vec<(tenet_core::SectorId, usize)> = leg.leg().iter().collect();
            fused = crate::space::fuse_sector_content(provider, &fused, &pairs)?;
        }
        Ok(fused)
    }

    /// Shared body of the structural constructors: checks the fused fit,
    /// builds zeros and writes the (partial) identity into every
    /// coupled-sector matrix — the same route as [`Self::id`], which is the
    /// same route as the erased `Tensor::structural`.
    fn structural<'a, C, M>(
        runtime: &Runtime,
        codomain: C,
        domain: M,
        embed: bool,
        what: &str,
    ) -> Result<Self, Error>
    where
        C: IntoIterator<Item = &'a GradedSpace<R>>,
        M: IntoIterator<Item = &'a GradedSpace<R>>,
        R: 'a,
    {
        let codomain: Vec<&GradedSpace<R>> = codomain.into_iter().collect();
        let domain: Vec<&GradedSpace<R>> = domain.into_iter().collect();
        let legs: Vec<&GradedSpace<R>> = codomain.iter().chain(&domain).copied().collect();
        let provider = Arc::clone(Self::authority(&legs)?);
        let fused_codomain = Self::fused_content(&provider, &codomain)?;
        let fused_domain = Self::fused_content(&provider, &domain)?;
        let fits = if embed {
            // TensorKit `domain ≾ codomain`: sectorwise embeddable.
            fused_domain
                .iter()
                .all(|&(sector, deg)| fused_codomain.iter().any(|&(s, d)| s == sector && d >= deg))
        } else {
            // TensorKit `domain ≅ codomain`: identical fused sector content
            // (both sides are SectorId-sorted, so slice equality is content
            // equality).
            fused_codomain == fused_domain
        };
        if !fits {
            // Same message shape as the erased `Tensor::structural`, so the
            // two facades stay diagnosable side by side.
            return Err(Error::InvalidArgument(format!(
                "{what}: codomain and domain are not {} (fused sector content differs)",
                if embed {
                    "isometrically embeddable"
                } else {
                    "isomorphic"
                }
            )));
        }
        let mut tensor = Self::build(runtime, provider, &codomain, &domain, Fill::Zeros)?;
        Self::write_identity_blocks(&mut tensor)?;
        Ok(tensor)
    }

    /// The canonical structural isomorphism `codomain <- domain` (TensorKit
    /// `isomorphism(W ← V)`): every
    /// coupled-sector block is the identity matrix, which requires the fused
    /// codomain and domain to carry identical sector content.
    ///
    /// # Errors
    ///
    /// Everything [`Self::zeros`] reports, plus [`Error::InvalidArgument`]
    /// when the fused codomain and domain differ in sector content
    /// (TensorKit's `SpaceMismatch` on `domain ≅ codomain`).
    ///
    /// # Complexity
    ///
    /// One fused-content fold over the legs plus one `O(stored_len)` payload.
    pub fn isomorphism<'a, C, M>(runtime: &Runtime, codomain: C, domain: M) -> Result<Self, Error>
    where
        C: IntoIterator<Item = &'a GradedSpace<R>>,
        M: IntoIterator<Item = &'a GradedSpace<R>>,
        R: 'a,
    {
        Self::structural(runtime, codomain, domain, false, "isomorphism")
    }

    /// TensorKit `unitary(W ← V)`: identical
    /// to [`Self::isomorphism`] — TensorKit only adds a Euclidean
    /// inner-product check, which every tenet fusion rule satisfies.
    ///
    /// # Errors and complexity
    ///
    /// Exactly [`Self::isomorphism`]'s.
    pub fn unitary<'a, C, M>(runtime: &Runtime, codomain: C, domain: M) -> Result<Self, Error>
    where
        C: IntoIterator<Item = &'a GradedSpace<R>>,
        M: IntoIterator<Item = &'a GradedSpace<R>>,
        R: 'a,
    {
        Self::structural(runtime, codomain, domain, false, "unitary")
    }

    /// The canonical isometry `codomain <- domain` (TensorKit
    /// `isometry(W ← V)`): each
    /// coupled-sector block is the partial identity (the first `cols` columns
    /// of the identity), so `t† ∘ t = id(domain)`. Requires the domain to
    /// embed isometrically in the codomain (sectorwise
    /// `deg_domain <= deg_codomain` on the fused content).
    ///
    /// # Errors
    ///
    /// Everything [`Self::zeros`] reports, plus [`Error::InvalidArgument`]
    /// when the fused domain does not embed sectorwise into the fused
    /// codomain (TensorKit's `SpaceMismatch` on `domain ≾ codomain`).
    ///
    /// # Complexity
    ///
    /// One fused-content fold over the legs plus one `O(stored_len)` payload.
    pub fn isometry<'a, C, M>(runtime: &Runtime, codomain: C, domain: M) -> Result<Self, Error>
    where
        C: IntoIterator<Item = &'a GradedSpace<R>>,
        M: IntoIterator<Item = &'a GradedSpace<R>>,
        R: 'a,
    {
        Self::structural(runtime, codomain, domain, true, "isometry")
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
    ///
    /// # Complexity
    ///
    /// As [`Self::zeros`] — one layout admission and one `O(stored_len)`
    /// payload — plus one pass writing each coupled-sector diagonal in place.
    #[doc(alias = "one")]
    pub fn id<'a, S>(runtime: &Runtime, spaces: S) -> Result<Self, Error>
    where
        S: IntoIterator<Item = &'a GradedSpace<R>>,
        R: 'a,
    {
        let spaces: Vec<&GradedSpace<R>> = spaces.into_iter().collect();
        let provider = Arc::clone(Self::authority(&spaces)?);
        let mut identity = Self::build(runtime, provider, &spaces, &spaces, Fill::Zeros)?;
        Self::write_identity_blocks(&mut identity)?;
        Ok(identity)
    }

    /// Writes the (partial) identity into every coupled-sector matrix of a
    /// freshly built zero tensor — TensorKit's `one!` per block
    /// (`tensors/linalg.jl:102-158`), shared by [`Self::id`] and the
    /// structural constructors.
    fn write_identity_blocks(tensor: &mut Self) -> Result<(), Error> {
        // The zero fill is written into, not replaced: `build` has just
        // allocated the payload and nothing else holds the body yet, so the
        // diagonal goes in place rather than into a second buffer.
        let body = Arc::get_mut(&mut tensor.body).expect("a freshly built body has no other owner");
        // Same coupled-sector region walk and same diagonal addressing as the
        // erased `Tensor::structural`, on the shared helper: the two build the
        // same tensor, and a second copy of the offset arithmetic would be free
        // to drift from the sibling this is byte-compared against.
        let regions = {
            let space = body.space.space();
            sector_regions(space.structure(), space.nout())?
        };
        // The payload `Arc` is unique for the same reason the body one is: this
        // tensor was built two statements ago and never handed out.
        let payload =
            Arc::get_mut(&mut body.data).expect("a freshly built payload has no other owner");
        let TypedData::Dense(data) = payload else {
            unreachable!("`build` always produces a dense payload");
        };
        for region in regions.iter() {
            for i in 0..region.rows().min(region.cols()) {
                data[region.range().start + i * (region.rows() + 1)] = D::from_real(1.0);
            }
        }
        Ok(())
    }
}

impl<R, D> TensorMap<R, D>
where
    R: TypedSectorAdmission,
    R::Mode: TypedTensorTransformDispatch<R, D>,
    D: TensorScalar,
{
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
    /// Checked Generic providers return [`GenericTensorError::Plan`] with the
    /// concrete provider error preserved as its source.
    ///
    /// ```
    /// use std::sync::Arc;
    ///
    /// use tenet::core::{U1FusionRule, U1Irrep};
    /// use tenet::typed::{GradedSpace, Runtime, TensorMap};
    ///
    /// let runtime = Runtime::builder().build()?;
    /// let rule = Arc::new(U1FusionRule);
    /// let v = GradedSpace::try_new(
    ///     Arc::clone(&rule),
    ///     [(U1Irrep::new(0), 1), (U1Irrep::new(1), 2)],
    ///     false,
    /// )?;
    /// let w = GradedSpace::try_new(
    ///     Arc::clone(&rule),
    ///     [(U1Irrep::new(0), 1), (U1Irrep::new(1), 1)],
    ///     false,
    /// )?;
    /// let t: TensorMap<_, f64> = TensorMap::rand(&runtime, [&v], [&w])?;
    /// assert_eq!(t.leg_dims()?, [3, 2]);
    ///
    /// let swapped = t.permute(&[1], &[0])?;
    /// assert_eq!(swapped.leg_dims()?, [2, 3]);
    /// // A bosonic two-leg swap is an involution: swapping back restores the
    /// // payload exactly.
    /// assert_eq!(swapped.permute(&[1], &[0])?.data(), t.data());
    /// # Ok::<(), tenet::typed::Error>(())
    /// ```
    pub fn permute(
        &self,
        codomain_axes: &[usize],
        domain_axes: &[usize],
    ) -> Result<Self, TypedFacadeError<R>> {
        // Why no identity shortcut (the erased facade shares storage when the
        // axes do not move): the result would be byte-identical either way, so
        // the shortcut is a pure cost question, and adding one without a gate
        // that measures it is speculative. The same reasoning covers every
        // other operation routed through `tree_transform` below.
        <R::Mode as TypedTensorTransformDispatch<R, D>>::tree_transform(
            self,
            TreeTransformOperation::permute(
                codomain_axes.iter().copied(),
                domain_axes.iter().copied(),
            ),
        )
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
    /// Checked Generic failures use [`GenericTensorError::Plan`].
    pub fn braid(
        &self,
        codomain_axes: &[usize],
        domain_axes: &[usize],
        levels: &[usize],
    ) -> Result<Self, TypedFacadeError<R>> {
        // Mirrors the erased pre-check verbatim (`Tensor::transformed`), same
        // message: two facades reporting one mistake two ways is a support
        // burden with no upside.
        let rank = self.rank();
        if levels.len() != rank {
            return Err(Error::InvalidArgument(format!(
                "braid levels must list one level per source axis \
                 (expected {rank}, got {})",
                levels.len()
            ))
            .into());
        }
        let nout = self.codomain_rank();
        <R::Mode as TypedTensorTransformDispatch<R, D>>::tree_transform(
            self,
            TreeTransformOperation::braid(
                codomain_axes.iter().copied(),
                domain_axes.iter().copied(),
                levels[..nout].iter().copied(),
                levels[nout..].iter().copied(),
            ),
        )
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
    /// validation this facade passes through. Checked Generic failures use
    /// [`GenericTensorError::Plan`].
    pub fn repartition(&self, num_codomain: usize) -> Result<Self, TypedFacadeError<R>> {
        self.planar(PlanarRequestKind::Repartition { num_codomain })
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
    pub fn transpose(&self) -> Result<Self, TypedFacadeError<R>> {
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
    ) -> Result<Self, TypedFacadeError<R>> {
        self.planar(PlanarRequestKind::Explicit {
            codomain_axes,
            domain_axes,
        })
    }

    /// Shared body of the three planar operations: derive the planar axis
    /// order, let the expert layer check it, and run it as a transpose.
    ///
    /// Why the axis derivation is borrowed from the erased layer rather than
    /// rewritten here: it *is* the definition of what "planar" means for each
    /// request kind, and a second copy would be free to drift from the erased
    /// sibling these operations are byte-compared against.
    fn planar(&self, kind: PlanarRequestKind<'_>) -> Result<Self, TypedFacadeError<R>> {
        let operation = with_planar_axes(
            self.codomain_rank(),
            self.rank(),
            kind,
            |codomain_axes, domain_axes| {
                // Why `transpose` and not `permute` even when the axes happen
                // to be a plain permutation: domain trees run opposite to the
                // planar boundary, so flattening them into a permute would
                // braid a different leg across it.
                Ok(TreeTransformOperation::transpose(
                    codomain_axes.iter().copied(),
                    domain_axes.iter().copied(),
                ))
            },
        )
        .map_err(TypedFacadeError::<R>::from)?;
        <R::Mode as TypedTensorTransformDispatch<R, D>>::tree_transform(self, operation)
    }
}

impl<R, D> TensorMap<R, D>
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64> + CheckedFusionAlgebra + SectorCodec,
    D: TensorScalar,
{
    /// Runs one prepared tree transform on this tensor's own runtime.
    fn tree_transform_multiplicity_free(
        &self,
        operation: TreeTransformOperation,
    ) -> Result<Self, Error> {
        // Compact arm: a rank-(1,1) leg swap of a spectrum factor is a
        // per-sector rescaling of the stored diagonal, so it never touches the
        // `Σ_c k_c²` materialization. Every other geometry — an explicit braid,
        // any higher rank, the identity partitions `repartition` produces —
        // falls through to the dense route below; the shared guard documents
        // why. TensorKit 0.17 `src/tensors/diagonal.jl:215-242` makes the same
        // split.
        if let Some(spectrum) = self.spectrum() {
            if crate::tensor_core::is_rank_one_diagonal_swap(
                self.codomain_rank(),
                self.rank() - self.codomain_rank(),
                &operation,
            ) {
                let destination = self.body.space.transformed_multiplicity_free(&operation)?;
                let transformed = crate::tensor_core::transform_rank_one_diagonal_spectrum(
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
    ///
    /// ```
    /// use std::sync::Arc;
    ///
    /// use tenet::core::{U1FusionRule, U1Irrep};
    /// use tenet::typed::{GradedSpace, Runtime, TensorMap};
    ///
    /// let runtime = Runtime::builder().build()?;
    /// let rule = Arc::new(U1FusionRule);
    /// let v = GradedSpace::try_new(
    ///     Arc::clone(&rule),
    ///     [(U1Irrep::new(0), 2), (U1Irrep::new(1), 1)],
    ///     false,
    /// )?;
    /// let t: TensorMap<_, f64> = TensorMap::rand(&runtime, [&v], [&v])?;
    /// let id = TensorMap::id(&runtime, [&v])?;
    ///
    /// // Contracting the domain leg with the identity's codomain leg is a
    /// // no-op on the payload; `[0, 1]` keeps the open axes in place.
    /// let out = t.contract(&id, &[1], &[0], &[0, 1])?;
    /// assert_eq!(out.data(), t.data());
    /// # Ok::<(), tenet::typed::Error>(())
    /// ```
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

    /// Documented alias of [`Self::contract`]: same arguments, same
    /// semantics, same compact fast paths and complexity, same errors — the
    /// delegation is total, so everything is stated there once.
    ///
    /// It exists for cross-facade name parity. The erased facade spells the
    /// operation as a pair — [`crate::prelude::Tensor::contract`] (implicit
    /// identity output order) and [`crate::prelude::Tensor::contract_ordered`]
    /// (explicit `output_axes`) — while this facade's [`Self::contract`]
    /// always takes the order explicitly, so it *is* the ordered route and
    /// this alias only lets the erased pair's name resolve here too.
    ///
    /// **TensorKit correspondence.** TensorKit has no `contract_ordered`
    /// entry point either: the counterpart of `output_axes` is
    /// `tensorcontract!`'s `pAB` output permutation (`TO.tensorcontract!`;
    /// the destination structure is `permute(compose(sA, sB), pAB)`, per
    /// `tensorcontract_structure`). The
    /// erased [`crate::prelude::Tensor::contract_ordered`] is the sibling
    /// reference for the erased-side semantics.
    ///
    /// # Errors
    ///
    /// Exactly [`Self::contract`]'s. One deliberate divergence from the
    /// erased sibling on inputs carrying *two* defects — mismatched
    /// contracted legs *and* a bad output order: the erased
    /// `contract_ordered` validates the contracted spaces before inspecting
    /// the output order (its documented "why not report `pAB` first" choice),
    /// while this facade delegates all validation to the expert layer, which
    /// reports the output-order defect first. The precedence is pinned in
    /// tests as it stands; re-checking the spaces here to force a match would
    /// be a second copy of the rules, free to drift.
    #[inline]
    pub fn contract_ordered(
        &self,
        other: &Self,
        lhs_axes: &[usize],
        rhs_axes: &[usize],
        output_axes: &[usize],
    ) -> Result<Self, Error> {
        self.contract(other, lhs_axes, rhs_axes, output_axes)
    }

    /// Tensor product in one category, ordered as
    /// `codomain(self), codomain(other); domain(self), domain(other)`.
    ///
    /// The two codomain trees and the two domain trees are merged
    /// independently with F moves. No legs cross and no R symbol is needed,
    /// including for a `NoBraiding` provider.
    pub fn otimes(&self, other: &Self) -> Result<Self, Error>
    where
        R: CanonicalUnitFusionRule,
    {
        if !self.runtime.same_runtime(&other.runtime) {
            return Err(Error::RuntimeMismatch);
        }
        let (space, data) = tensorproduct_owned_multiplicity_free(
            BoundDynamicTensorRef::try_new(&self.body.space, self.dense_data())?,
            BoundDynamicTensorRef::try_new(&other.body.space, other.dense_data())?,
        )?;
        Ok(Self {
            runtime: self.runtime.clone(),
            body: Arc::new(TypedTensorBody::dense(space, data)),
        })
    }

    /// TensorKit `deligneproduct`: embeds `self` as `(a, 𝟙)` and `other` as
    /// `(𝟙, b)` in the supplied ordered product category, then combines the
    /// embedded tensors with the F-only [`Self::otimes`] route.
    ///
    /// This operation is typed-only and keeps the payload type `D` unchanged.
    /// The caller supplies the exact [`ProductFusionRule`], including its
    /// component providers and codec; both component [`tenet_core::RuleIdentity`] values
    /// must match the operands, and the codec participates in the product
    /// identity. [`CanonicalUnitFusionRule`] is required for both components
    /// because TeNeT stores no separate unitor data. Factor order and nested
    /// association are preserved exactly rather than reassociated or swapped.
    ///
    /// Validation reports [`Error::RuntimeMismatch`] before component
    /// [`Error::RuleMismatch`]. Both vacuum embeddings, codec decodes, and
    /// source/target fusion-tree bijections are prepared before either
    /// embedded `TensorMap` or layout is published. After that transaction
    /// succeeds, the operation builds the two embedded tensors by copying
    /// their dense data (materializing a compact operand when necessary).
    pub fn deligne_product<R2, C>(
        &self,
        other: &TensorMap<R2, D>,
        product: Arc<ProductFusionRule<R, R2, C>>,
    ) -> Result<TensorMap<ProductFusionRule<R, R2, C>, D>, Error>
    where
        R: CanonicalUnitFusionRule,
        R2: MultiplicityFreeRigidSymbols<Scalar = f64>
            + CheckedFusionAlgebra
            + SectorCodec
            + CanonicalUnitFusionRule,
        C: ProductSectorCodec + Sync + 'static,
    {
        if !self.runtime.same_runtime(&other.runtime) {
            return Err(Error::RuntimeMismatch);
        }
        if product.left_rule().rule_identity() != self.body.space.provider().rule_identity()
            || product.right_rule().rule_identity() != other.body.space.provider().rule_identity()
        {
            return Err(Error::RuleMismatch);
        }
        let left_vacuum = product
            .left_rule()
            .decode_sector(product.left_rule().vacuum())?;
        let right_vacuum = product
            .right_rule()
            .decode_sector(product.right_rule().vacuum())?;
        let left = prepare_product_operand(
            self,
            Arc::clone(&product),
            |sector| ProductSector::new(sector, right_vacuum.clone()),
            |sector| sector.left().clone(),
        )?;
        let right = prepare_product_operand(
            other,
            product,
            |sector| ProductSector::new(left_vacuum.clone(), sector),
            |sector| sector.right().clone(),
        )?;
        let left = left.commit()?;
        let right = right.commit()?;
        left.otimes(&right)
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

    /// Integer tensor-map power (TensorKit `t ^ p`), using `O(log |p|)`
    /// compositions. Zero returns the multiplicative identity (staying compact
    /// for compact input); negative powers invert once.
    ///
    /// Returns [`Error::InvalidArgument`] unless this is an endomorphism.
    pub fn powi(&self, exponent: i32) -> Result<Self, Error> {
        if !self.is_endomorphism() {
            return Err(Error::InvalidArgument(
                "powi() requires an endomorphism (domain == codomain)".to_string(),
            ));
        }
        if exponent == 0 {
            if let Some(spectrum) = self.spectrum() {
                return Ok(self.with_spectrum(map_spectrum(spectrum, |_| Ok(D::from_real(1.0)))?));
            }
            return Self::id(&self.runtime, &self.domain());
        }

        let power = if exponent < 0 {
            self.inv()?
        } else {
            self.clone()
        };
        pow_by_squaring(power, exponent.unsigned_abs(), Self::compose)
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
    ///
    /// ```
    /// use std::sync::Arc;
    ///
    /// use tenet::core::{U1FusionRule, U1Irrep};
    /// use tenet::typed::{GradedSpace, Runtime, TensorMap};
    ///
    /// let runtime = Runtime::builder().build()?;
    /// let rule = Arc::new(U1FusionRule);
    /// let v = GradedSpace::try_new(
    ///     Arc::clone(&rule),
    ///     [(U1Irrep::new(0), 2), (U1Irrep::new(1), 2)],
    ///     false,
    /// )?;
    /// let t: TensorMap<_, f64> = TensorMap::rand(&runtime, [&v], [&v])?;
    ///
    /// let (u, s, vh) = t.svd_compact()?;
    /// assert!(u.is_isometric(1e-12)?);
    /// let rebuilt = u.compose(&s)?.compose(&vh)?;
    /// let max_err = rebuilt
    ///     .data()
    ///     .iter()
    ///     .zip(t.data())
    ///     .map(|(a, b)| (a - b).abs())
    ///     .fold(0.0f64, f64::max);
    /// assert!(max_err < 1e-12);
    /// # Ok::<(), tenet::typed::Error>(())
    /// ```
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
    ///
    /// # Complexity
    ///
    /// `O(Σ_c n_c³)` — sectorwise cubic, no global materialization; the seam
    /// runs one dense QR per coupled-sector matrix. A compact-diagonal
    /// payload (TensorKit's `DiagonalTensorMap`) is materialized into the
    /// dense coupled buffer first, through the same [`Self::data`] route as
    /// [`Self::left_polar`]. TensorKit 0.17 *does* keep a diagonal QR compact
    /// (MatrixAlgebraKit's `DiagonalAlgorithm`); that fast path is not
    /// adopted here — the issue #613 Group 4 contract requires every compact
    /// fast path to be re-proven individually, the same deferral the polars
    /// record.
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
    ///
    /// # Complexity
    ///
    /// As [`Self::qr_compact`]: sectorwise cubic, with a compact-diagonal
    /// payload materialized dense first (TensorKit's `DiagonalAlgorithm`
    /// covers `qr_full!` too — same non-adoption, same #613 Group 4
    /// deferral).
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
    ///
    /// # Complexity
    ///
    /// As [`Self::qr_compact`]: sectorwise cubic, with a compact-diagonal
    /// payload materialized dense first (TensorKit's `DiagonalAlgorithm`
    /// covers the LQ pair as well — same non-adoption, same #613 Group 4
    /// deferral).
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
    ///
    /// # Complexity
    ///
    /// As [`Self::lq_compact`]: sectorwise cubic, compact-diagonal payload
    /// materialized dense first.
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
    /// # Null bond
    ///
    /// `W` is a fresh non-dual single-leg bond space carrying, per coupled
    /// sector `c`, the `rows_c − rank_c` null directions; `rank_c` is the
    /// numerical rank the seam takes from that sector's compact SVD, counting
    /// `σ > ε(dtype) · max(rows_c, cols_c) · σ_max,c` as nonzero. A sector
    /// with no null directions is absent from `W`, so `W` is empty for a
    /// numerically full-rank tensor. Note this is *not*
    /// TensorKit/MatrixAlgebraKit's default `left_null`, which without a
    /// truncation argument is QR-based and counts only the structural nullity
    /// `rows_c − min(rows_c, cols_c)` (MatrixAlgebraKit
    /// `interface/orthnull.jl`, the `alg::Nothing` mode); the seam's behavior
    /// corresponds to their SVD mode with a tolerance.
    ///
    /// # Errors
    ///
    /// As [`Self::svd_compact`]: the seam's own errors, unfiltered.
    ///
    /// # Complexity
    ///
    /// Sectorwise cubic — one compact SVD per coupled sector plus an
    /// orthonormal completion of the sectors that keep null directions; a
    /// compact-diagonal payload is materialized dense first, as for
    /// [`Self::qr_compact`].
    pub fn left_null(&self) -> Result<Self, Error> {
        let mut dense = self.runtime.lease_dense();
        let out = tenet_matrixalgebra::left_null_dyn(dense.dense(), &self.bound_ref()?)?;
        Ok(self.wrap_bound_factor(out))
    }

    /// TensorKit 0.17 / MatrixAlgebraKit `right_null`: `n : W <- domain` with
    /// `t * n^H = 0`.
    ///
    /// # Null bond
    ///
    /// As [`Self::left_null`], mirrored: `W` is a fresh non-dual single-leg
    /// bond space with `cols_c − rank_c` directions per coupled sector under
    /// the same SVD numerical-rank cutoff, sectors with none absent — and the
    /// same divergence from TensorKit/MatrixAlgebraKit's QR-based default
    /// applies.
    ///
    /// # Errors
    ///
    /// As [`Self::svd_compact`]: the seam's own errors, unfiltered.
    ///
    /// # Complexity
    ///
    /// As [`Self::left_null`]: sectorwise cubic, compact-diagonal payload
    /// materialized dense first.
    pub fn right_null(&self) -> Result<Self, Error> {
        let mut dense = self.runtime.lease_dense();
        let out = tenet_matrixalgebra::right_null_dyn(dense.dense(), &self.bound_ref()?)?;
        Ok(self.wrap_bound_factor(out))
    }

    /// TensorKit 0.17 / MatrixAlgebraKit `left_polar`: the polar decomposition
    /// `t = w ∘ p`, returned as `(w, p)` — `w` isometric (`w† ∘ w = id` on the
    /// domain) and `p` Hermitian positive semidefinite.
    ///
    /// Factor spaces per TensorKit 0.17: `w` lives on the input's
    /// own space `codomain <- domain`, `p` on `domain <- domain`. TensorKit
    /// also exposes algorithm kinds for the polars; neither tenet facade does —
    /// a deliberate narrowing, in parity with the erased
    /// [`crate::prelude::Tensor::left_polar`]. The erased facade materializes
    /// an adjoint view before decomposing; this facade has no adjoint views, so
    /// there is no counterpart to that step here.
    ///
    /// # Errors
    ///
    /// As [`Self::svd_compact`]: the seam's own errors, unfiltered — in
    /// particular [`Error::Operation`] when some coupled-sector matrix has
    /// fewer rows than columns (the left polar needs every sector at least as
    /// tall as it is wide).
    ///
    /// # Complexity
    ///
    /// `O(Σ_c n_c³)` — sectorwise cubic, no global materialization; the seam
    /// factorizes each coupled sector on its own. A compact-diagonal payload
    /// (TensorKit's `DiagonalTensorMap`) materializes through the same
    /// [`Self::data`] route as [`Self::qr_compact`] first: TensorKit 0.17 has
    /// no diagonal polar specialization either (its `DiagonalAlgorithm`
    /// table gives `DiagonalTensorMap` only `copy_input` for the polars, so
    /// it dispatches dense per block), and the
    /// issue #613 Group 4 contract requires any compact fast path to be
    /// individually re-proven — out of scope here.
    pub fn left_polar(&self) -> Result<(Self, Self), Error> {
        // Dense lease before the context lease — the polar seam recouples
        // internally, so unlike QR/LQ/null it takes the context lane; the
        // lease order matches every existing site on both facades.
        let mut dense = self.runtime.lease_dense();
        let mut lease = self.runtime.lease_context()?;
        let (w, p) = tenet_matrixalgebra::left_polar_dyn(
            dense.dense(),
            lease.context().multiplicity_free_lane::<D>(),
            &self.bound_ref()?,
        )?;
        Ok((self.wrap_bound_factor(w), self.wrap_bound_factor(p)))
    }

    /// TensorKit 0.17 / MatrixAlgebraKit `right_polar`: the polar
    /// decomposition `t = p ∘ w`, returned as `(p, w)` — `p` Hermitian
    /// positive semidefinite and `w` a coisometry (`w ∘ w† = id` on the
    /// codomain).
    ///
    /// Factor spaces per TensorKit 0.17: `p` on
    /// `codomain <- codomain`, `w` on the input's own space
    /// `codomain <- domain`. Everything [`Self::left_polar`] says about
    /// algorithm kinds, adjoint views and the compact-diagonal route holds
    /// here unchanged.
    ///
    /// # Errors
    ///
    /// As [`Self::left_polar`], mirrored: [`Error::Operation`] when some
    /// coupled-sector matrix has fewer columns than rows.
    ///
    /// # Complexity
    ///
    /// As [`Self::left_polar`]: `O(Σ_c n_c³)`, sectorwise, with a
    /// compact-diagonal payload materialized first.
    pub fn right_polar(&self) -> Result<(Self, Self), Error> {
        // See `left_polar` for the lease order rationale.
        let mut dense = self.runtime.lease_dense();
        let mut lease = self.runtime.lease_context()?;
        let (p, w) = tenet_matrixalgebra::right_polar_dyn(
            dense.dense(),
            lease.context().multiplicity_free_lane::<D>(),
            &self.bound_ref()?,
        )?;
        Ok((self.wrap_bound_factor(p), self.wrap_bound_factor(w)))
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
    /// sector — TensorKit's `exp`, which copies and calls
    /// `exp!`: check `domain == codomain`, then
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
    ///
    /// ```
    /// use std::sync::Arc;
    ///
    /// use tenet::core::{U1FusionRule, U1Irrep};
    /// use tenet::typed::{GradedSpace, Runtime, TensorMap};
    ///
    /// let runtime = Runtime::builder().build()?;
    /// let rule = Arc::new(U1FusionRule);
    /// let v = GradedSpace::try_new(
    ///     Arc::clone(&rule),
    ///     [(U1Irrep::new(0), 2), (U1Irrep::new(1), 1)],
    ///     false,
    /// )?;
    ///
    /// // exp of the zero endomorphism is the identity.
    /// let zero: TensorMap<_, f64> = TensorMap::zeros(&runtime, [&v], [&v])?;
    /// let id: TensorMap<_, f64> = TensorMap::id(&runtime, [&v])?;
    /// let max_err = zero
    ///     .exp()?
    ///     .data()
    ///     .iter()
    ///     .zip(id.data())
    ///     .map(|(a, b)| (a - b).abs())
    ///     .fold(0.0f64, f64::max);
    /// assert!(max_err < 1e-15);
    /// # Ok::<(), tenet::typed::Error>(())
    /// ```
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
            body: Arc::new(TypedTensorBody::diagonal(self.body.space.clone(), spectrum)),
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
            body: Arc::new(TypedTensorBody::diagonal(space, spectrum)),
        }
    }

    /// The compact payload, when this tensor has one.
    fn spectrum(&self) -> Option<&[tenet_matrixalgebra::SectorSpectrum<D>]> {
        match &*self.body.data {
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
    ///
    /// ```
    /// use std::sync::Arc;
    ///
    /// use tenet::core::{FermionParityFusionRule, Z2Irrep};
    /// use tenet::typed::{GradedSpace, Runtime, TensorMap};
    ///
    /// let runtime = Runtime::builder().build()?;
    /// let rule = Arc::new(FermionParityFusionRule);
    /// let v = GradedSpace::try_new(
    ///     Arc::clone(&rule),
    ///     [(Z2Irrep::EVEN, 1), (Z2Irrep::ODD, 1)],
    ///     false,
    /// )?;
    /// let t: TensorMap<_, f64> = TensorMap::rand(&runtime, [&v], [&v])?;
    ///
    /// // `1.0 * t + 1.0 * t` doubles every entry, so the norm doubles too.
    /// let doubled = t.add(&t, 1.0, 1.0)?;
    /// assert!((doubled.norm()? - 2.0 * t.norm()?).abs() < 1e-12);
    /// # Ok::<(), tenet::typed::Error>(())
    /// ```
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
    /// # Complexity
    ///
    /// Dense storage runs the partial-trace engine over the whole payload. A
    /// compact spectrum factor traced over its only pair reduces the stored
    /// spectrum in `O(Σ_c k_c)` without materializing (#604) — the typed twin
    /// of the erased facade's #585 arm, with the same deliberately narrow
    /// guard: one pair on a rank-(1,1) source, where the destination tree is
    /// empty and the coefficient collapses to a per-sector scalar,
    /// `dim(c) · θ(c)` on a direct traced codomain leg and `dim(c)` on a dual
    /// one. That twist is what makes this the supertrace and not [`Self::tr`];
    /// the coefficient is pinned byte-for-byte against the erased compact arm
    /// and numerically (the engine's summation order differs) against the
    /// engine route by the oracle sweeps in `tests/typed_facade.rs` and
    /// `tenet/src/tensor/compact_diagonal_tests.rs`.
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
        // Compact arm (#604): the full trace of a rank-(1,1) spectrum factor
        // over its only pair is a reduction of the stored spectrum, so there
        // is nothing to materialize — the typed twin of the erased #585 arm in
        // `tensor.rs`, which owns the long-form rationale. In brief: this is
        // the *categorical* trace, not `tr()`'s — the engine's
        // `trace_channel_factor` carries the quantum dimension of the traced
        // channel and, exactly where the traced leg is *not* dual, its
        // fermionic twist, which is what makes this the supertrace for a
        // fermionic rule and the coefficient `dim(c) · θ(c)` rather than
        // `tr()`'s unconditional `dim(c)`. The guard is this narrow because
        // with one pair and rank two the destination is the empty tree, so the
        // traced channel is a single uncoupled sector and the coefficient
        // collapses to a per-sector scalar; any wider geometry leaves an open
        // destination tree whose recoupling is not a per-sector scaling.
        // Today the geometric conditions are implied by the Group 4 contract
        // (`TypedData::Diagonal` lives on bond spaces only), so they are
        // defensive parity with the erased guard, not a reachable branch. No
        // adjoint-view exclusion, unlike erased: this facade's `adjoint` is
        // eager and keeps compact storage compact, so there is no lazy view to
        // exclude. The coefficient is not derivable here — it is pinned
        // against the erased arm and the engine route by the oracle sweeps in
        // `tests/typed_facade.rs` (`compact_full_trace_*`) and, on the erased
        // side, `tensor/compact_diagonal_tests.rs`.
        if let Some(spectrum) = self.spectrum() {
            if rank == 2 && self.codomain_rank() == 1 && pairs.len() == 1 {
                let traced_leg_is_dual: bool =
                    self.body.space.space().homspace().codomain().legs()[0].is_dual();
                let provider: &R = self.body.space.provider();
                // Accumulated in `Complex64` and narrowed once through the
                // #568 `UserScalar` surface, mirroring both the compact `tr`
                // above and the erased arm's `ordinary_trace_with` — same
                // per-sector reduction order, so the two facades agree
                // byte-for-byte. The dtype story is simpler than erased's
                // three-way `DiagonalData` split: the typed spectrum already
                // stores `SectorSpectrum<D>`, and the coefficient is the
                // provider's real scalar, so the result is a plain `D`.
                let mut total: num_complex::Complex64 = num_complex::Complex64::new(0.0, 0.0);
                for entry in spectrum {
                    let coefficient: f64 = if traced_leg_is_dual {
                        provider.dim_scalar(entry.sector)
                    } else {
                        provider.dim_scalar(entry.sector) * provider.twist_scalar(entry.sector)
                    };
                    let mut partial: D = D::from_real(0.0);
                    for &value in &entry.values {
                        partial = partial + value;
                    }
                    total += partial.widen_complex() * coefficient;
                }
                // The erased arm's internal check, on the shared derived
                // space: a fully traced rank-(1,1) destination is one scalar.
                if space.space().required_len()? != 1 {
                    return Err(internal_layout_error(
                        "a fully traced rank-one destination is not a single scalar",
                    ));
                }
                let value: D = D::from_complex64(total);
                return Ok(Self {
                    runtime: self.runtime.clone(),
                    body: Arc::new(TypedTensorBody::dense(space, vec![value])),
                });
            }
        }
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
    /// Eager, into a fresh destination — TensorKit's own `adjoint!`, so this is a TK-sanctioned form rather than a
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

    /// TensorKit `norm(t, p)` for a general exponent:
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
    /// satisfies `t.normalize()?.norm()? == 1` up to floating point.
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
            return Ok(crate::tensor_core::compact_is_posdef(spectrum, threshold));
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
    ///
    /// Allocates: each call builds a fresh `Vec` and clones every leg's
    /// sector table (the provider travels by `Arc` bump). Hold the result
    /// rather than re-calling in a loop.
    pub fn codomain(&self) -> Vec<GradedSpace<R>> {
        self.legs(self.body.space.space().homspace().codomain())
    }

    /// The domain legs, in axis order.
    ///
    /// Allocates per call, exactly as [`Self::codomain`].
    pub fn domain(&self) -> Vec<GradedSpace<R>> {
        self.legs(self.body.space.space().homspace().domain())
    }

    /// The codomain legs, in axis order (TensorKit `codomain(t)`).
    /// Documented alias of
    /// [`Self::codomain`], carried for cross-facade name parity with the
    /// erased [`crate::prelude::Tensor::codomain_spaces`].
    #[inline]
    pub fn codomain_spaces(&self) -> Vec<GradedSpace<R>> {
        self.codomain()
    }

    /// The domain legs, in axis order (TensorKit `domain(t)`) — the
    /// spaces as written, i.e.
    /// *not* dualized. Documented alias of [`Self::domain`], carried for
    /// cross-facade name parity with the erased
    /// [`crate::prelude::Tensor::domain_spaces`].
    #[inline]
    pub fn domain_spaces(&self) -> Vec<GradedSpace<R>> {
        self.domain()
    }

    /// Quantum-dimension-weighted total dimension of every leg, in flat order
    /// (codomain legs first, then domain legs) — TensorKit's `dim(space(t,
    /// i))` per leg.
    /// Contraction planners use it as a size/FLOP proxy, exactly as they use
    /// the erased [`crate::prelude::Tensor::leg_dims`].
    ///
    /// Same rounding formula as the erased facade:
    /// `Σ_sector round(degeneracy * dim(sector))` per leg. The erased sibling
    /// needs a dedicated SU(3) branch because its rule set is a closed enum
    /// whose SU(3) arm speaks a different symbol trait; here the provider
    /// abstraction carries `dim_scalar` uniformly, so there is deliberately
    /// **no special case** — fewer branches, identical semantics.
    ///
    /// # Complexity
    ///
    /// `O(Σ_leg sectors)`; allocates the returned `Vec<usize>` only, never a
    /// payload.
    ///
    /// # Errors
    ///
    /// None today; the `Result` keeps the erased signature's shape so the two
    /// facades stay drop-in for each other.
    pub fn leg_dims(&self) -> Result<Vec<usize>, Error> {
        let hom = self.body.space.space().homspace();
        let provider = self.body.space.provider();
        Ok(hom
            .codomain()
            .legs()
            .iter()
            .chain(hom.domain().legs())
            .map(|leg| Self::weighted_leg_dim(provider, leg))
            .collect())
    }

    /// Quantum-dimension-weighted size of one flat leg — one entry of
    /// [`Self::leg_dims`] without building the whole vector.
    ///
    /// # Errors
    ///
    /// [`Error::InvalidArgument`] when `axis >= rank()`, with the erased
    /// facade's own message.
    pub fn leg_dim(&self, axis: usize) -> Result<usize, Error> {
        let hom = self.body.space.space().homspace();
        let nout = hom.codomain().len();
        let leg = if axis < nout {
            &hom.codomain().legs()[axis]
        } else if axis < self.rank() {
            &hom.domain().legs()[axis - nout]
        } else {
            return Err(Error::InvalidArgument(format!(
                "axis {axis} out of range for rank {}",
                self.rank()
            )));
        };
        Ok(Self::weighted_leg_dim(self.body.space.provider(), leg))
    }

    /// The erased facade's per-leg reduction, verbatim: quantum dimensions are
    /// generally irrational (SU(2) `sqrt` products, anyonic golden ratios), so
    /// the per-sector weight is computed in `f64` and rounded once.
    fn weighted_leg_dim(provider: &R, leg: &SectorLeg) -> usize {
        leg.sectors()
            .iter()
            .zip(leg.degeneracies())
            .map(|(&sector, &degeneracy)| {
                (degeneracy as f64 * provider.dim_scalar(sector)).round() as usize
            })
            .sum()
    }

    /// The single element of a rank-0 (scalar) tensor, e.g. the result of
    /// contracting every leg — TensorKit `scalar` (an empty payload reads
    /// as zero there too).
    ///
    /// Returns `D` directly: the static-dtype counterpart of the erased
    /// [`crate::prelude::Tensor::scalar`], whose `Scalar` enum exists only
    /// because its dtype is a runtime property. Not a semantic difference —
    /// the value is the same sum of the coupled payload.
    ///
    /// # Errors
    ///
    /// [`Error::InvalidArgument`] on a tensor with legs, with the erased
    /// facade's own message.
    pub fn scalar(&self) -> Result<D, Error> {
        if self.rank() != 0 {
            return Err(Error::InvalidArgument(format!(
                "scalar() requires a rank-0 tensor, got rank {}",
                self.rank()
            )));
        }
        // A rank-0 payload holds at most one element; summing matches the
        // erased facade and gives the empty payload its zero for free.
        Ok(self
            .dense_data()
            .iter()
            .fold(D::from_real(0.0), |acc, &value| acc + value))
    }

    /// TensorKit `catdomain(t1, t2)`:
    /// concatenate two `N₁ <- 1` tensor maps along their sole domain leg. The
    /// codomain product spaces must match exactly; the two domain legs must
    /// share duality and are combined by direct sum `V = V1 ⊕ V2`; reduced
    /// data is copied into adjacent column slabs per coupled sector, `self`
    /// first.
    ///
    /// Rust uses a method (`t1.catdomain(&t2)`) because binary tensor
    /// operations in this API are methods; the name and operand order match
    /// TensorKit's free function, exactly as the erased
    /// [`crate::prelude::Tensor::catdomain`] does.
    ///
    /// Narrowings against the erased sibling, all statically enforced rather
    /// than semantic differences: both operands share one `D`, so the erased
    /// mixed-dtype widening arm (`execute_c64`) is unrepresentable — widen
    /// with [`Self::to_c64`] first; there are no device placements to check;
    /// and the erased lazy-adjoint fast path has no counterpart because the
    /// typed [`Self::adjoint`] is eager. A compact diagonal operand is
    /// materialized dense first (once, into the body cache), as the erased
    /// route does through its coupled-data materialization.
    ///
    /// # Complexity
    ///
    /// One output allocation and a single `O(len(self) + len(other))` copy
    /// pass over the compiled per-sector slab plan — the same plan object the
    /// erased facade executes.
    ///
    /// # Errors
    ///
    /// [`Error::RuleMismatch`] on differing provider identities and
    /// [`Error::RuntimeMismatch`] on differing runtimes, in that order (the
    /// erased `check_same_execution_world`); then
    /// [`Error::InvalidArgument`] — with the erased facade's own messages —
    /// for a multi-leg domain, mismatched codomain product spaces, or changed
    /// legs of opposite duality.
    pub fn catdomain(&self, other: &Self) -> Result<Self, Error> {
        self.cat(other, CatSide::Domain)
    }

    /// TensorKit `catcodomain(t1, t2)`:
    /// concatenate two `1 <- N₂` tensor maps along their sole codomain leg.
    /// The domain product spaces must match exactly; the two codomain legs
    /// must share duality and are combined by direct sum; reduced data is
    /// copied into adjacent row slabs per coupled sector, `self` first.
    ///
    /// Method-vs-free-function note, narrowings, complexity and error
    /// classes: exactly as [`Self::catdomain`], with the codomain and domain
    /// roles swapped.
    pub fn catcodomain(&self, other: &Self) -> Result<Self, Error> {
        self.cat(other, CatSide::Codomain)
    }

    /// Shared route of [`Self::catdomain`] / [`Self::catcodomain`]: the same
    /// validation core, copy-plan compiler and executor the erased facade
    /// uses (`cat_homspace` / `compile_cat_plan` — #580 PR 4), fed with the
    /// typed bound space's structure and executed on the dense payloads.
    fn cat(&self, other: &Self, side: CatSide) -> Result<Self, Error> {
        // Rule identity before runtime: the erased
        // `check_same_execution_world` order, minus its placement arm (no
        // devices here). Same rationale as `authority()`: separately
        // allocated providers of one rule interoperate; different identities
        // are rejected before any layout work.
        if self.body.space.provider().rule_identity() != other.body.space.provider().rule_identity()
        {
            return Err(Error::RuleMismatch);
        }
        if !self.runtime.same_runtime(&other.runtime) {
            return Err(Error::RuntimeMismatch);
        }
        let lhs = self.body.space.space().homspace();
        let rhs = other.body.space.space().homspace();
        let (axis, homspace) = cat_homspace(
            lhs.codomain(),
            lhs.domain(),
            rhs.codomain(),
            rhs.domain(),
            side,
        )?;
        // The same derivation route the erased `from_homspace` takes
        // (`build_bound_space_like` is `derive_from_final_homspace` on the
        // authority space); no new space-construction logic.
        let space = self.body.space.derive_from_final_homspace(homspace)?;
        let plan = compile_cat_plan(
            space.space().structure(),
            space.space().nout(),
            [
                CatOperandLayout::owned(
                    self.body.space.space().structure(),
                    self.body.space.space().nout(),
                    self.body.space.space().nin(),
                )?,
                CatOperandLayout::owned(
                    other.body.space.space().structure(),
                    other.body.space.space().nout(),
                    other.body.space.space().nin(),
                )?,
            ],
            axis,
            side,
        )?
        // The erased owned route carries the same expectation: an all-owned
        // pair never declines its layout.
        .ok_or_else(|| {
            internal_layout_error("owned concatenation unexpectedly declined its layout")
        })?;
        let data = plan.execute(self.dense_data(), other.dense_data())?;
        Ok(Self {
            runtime: self.runtime.clone(),
            body: Arc::new(TypedTensorBody::dense(space, data)),
        })
    }

    /// TensorKit `absorb(tdst, tsrc)` (which copies and delegates to
    /// `absorb!`): copies the common per-axis prefix of every shared
    /// fusion-tree block of `source` into a deep copy of `self` (TK takes
    /// the `min` of the two block shapes per axis). Blocks whose key the source
    /// does not carry are untouched, so the caller owns the initialization of
    /// the non-shared region — TK documents the same contract.
    ///
    /// The result keeps `self`'s spaces and dtype. Equal `D` is required by
    /// the signature, so the erased facade's f64→c64 widening and its
    /// `InexactScalarConversion` narrowing arm are statically
    /// unrepresentable — widen with [`Self::to_c64`] first. No device
    /// placements exist here; a compact diagonal payload (on either side) is
    /// materialized dense first, exactly once, as the erased route does.
    ///
    /// # Complexity
    ///
    /// One output allocation (the destination clone) plus `O(min-prefix)`
    /// overwrites per shared block, walked by the same merge-join over sorted
    /// block keys the erased facade executes (`absorb_mapped`).
    ///
    /// # Errors
    ///
    /// In the erased facade's order: [`Error::InvalidArgument`] on unequal
    /// codomain/domain ranks (TK throws its `DimensionError` for the same),
    /// [`Error::RuleMismatch`] on differing provider identities,
    /// [`Error::RuntimeMismatch`] on differing runtimes, and
    /// [`Error::InvalidArgument`] when corresponding legs differ in duality.
    /// The messages name `TensorMap::absorb` where the erased ones name
    /// `Tensor::absorb` — same shape, honest receiver.
    pub fn absorb(&self, source: &Self) -> Result<Self, Error> {
        let destination_space = self.body.space.space();
        let source_space = source.body.space.space();
        if destination_space.nout() != source_space.nout()
            || destination_space.nin() != source_space.nin()
        {
            return Err(Error::InvalidArgument(format!(
                "TensorMap::absorb requires equal codomain/domain ranks, got {}|{} and {}|{}",
                destination_space.nout(),
                destination_space.nin(),
                source_space.nout(),
                source_space.nin()
            )));
        }
        if self.body.space.provider().rule_identity()
            != source.body.space.provider().rule_identity()
        {
            return Err(Error::RuleMismatch);
        }
        if !self.runtime.same_runtime(&source.runtime) {
            return Err(Error::RuntimeMismatch);
        }
        for (destination_leg, source_leg) in destination_space
            .homspace()
            .codomain()
            .legs()
            .iter()
            .chain(destination_space.homspace().domain().legs())
            .zip(
                source_space
                    .homspace()
                    .codomain()
                    .legs()
                    .iter()
                    .chain(source_space.homspace().domain().legs()),
            )
        {
            if destination_leg.is_dual() != source_leg.is_dual() {
                return Err(Error::InvalidArgument(
                    "TensorMap::absorb requires corresponding legs to have equal duality"
                        .to_string(),
                ));
            }
        }
        let destination_data = self.dense_data();
        let source_data = source.dense_data();
        // The erased `validate_absorb_layout` internal guard, minus its
        // dtype/device arms (unrepresentable here): the dense payloads must
        // cover their structures before any block walk trusts the offsets.
        if destination_space.structure().required_len()? != destination_data.len()
            || source_space.structure().required_len()? != source_data.len()
        {
            return Err(internal_layout_error(
                "absorb block layout does not cover scalar storage",
            ));
        }
        let mut output = destination_data.to_vec();
        absorb_mapped(
            destination_space.structure(),
            &mut output,
            source_space.structure(),
            source_data,
            Ok,
        )?;
        Ok(Self {
            runtime: self.runtime.clone(),
            body: Arc::new(TypedTensorBody::dense(self.body.space.clone(), output)),
        })
    }

    /// TensorKit `twist(t, inds)` (and its in-place `twist!`): multiplies
    /// each fusion-tree block by the product over
    /// `legs` (flat leg indices, codomain first) of the ribbon-twist
    /// eigenvalue θ of that leg's uncoupled sector. θ = −1 for odd fermionic
    /// sectors and +1 for every bosonic sector, so this is a no-op on purely
    /// bosonic legs and an involution (θ² = 1) on fermionic ones.
    ///
    /// When the twist is the identity on every stored block — TensorKit
    /// `has_shared_twist`: a bosonic
    /// provider, or no requested leg touching a twisted sector — the result
    /// is a body-sharing clone, O(1), exactly as TensorKit's `copy = false`
    /// default shares `t`. A compact
    /// spectrum factor scales spectrum-per-sector and **stays compact**,
    /// O(Σ_c k_c) — parity with the erased `Data::Diagonal` fast path, and
    /// with TensorKit, whose `DiagonalTensorMap` twist stays diagonal
    /// because `similar` preserves the diagonal storage
    /// and `twist!` only scales blocks.
    /// Otherwise: one scaled copy of the dense payload, O(len), through the
    /// same per-block walk as the erased facade (`scale_blocks_impl`).
    ///
    /// No adjoint arm (this facade's `adjoint` is eager, so there is no lazy
    /// view to materialize first) and no device arm (the payload is a host
    /// `Vec<D>` by construction) — deliberate narrowings, not semantic
    /// differences. The erased route's SU(3) rejection is dead here: the
    /// multiplicity-free admission bound keeps a `Generic` provider out at
    /// construction.
    ///
    /// # Errors
    ///
    /// [`Error::InvalidArgument`] when a leg is out of range, with the
    /// erased facade's own message — reported before the empty-list
    /// short-circuit, matching the erased validation order. An empty `legs`
    /// returns an identical clone.
    pub fn twist(&self, legs: &[usize]) -> Result<Self, Error> {
        let rank = self.rank();
        if let Some(&leg) = legs.iter().find(|&&leg| leg >= rank) {
            return Err(Error::InvalidArgument(format!(
                "twist leg {leg} out of range for rank {rank}"
            )));
        }
        if legs.is_empty() {
            return Ok(self.clone());
        }
        let provider = self.body.space.provider();
        // NoBraiding preflight (PR #620 review): before the compact arm and
        // before any θ evaluation — see `reject_unbraided_nonunit_legs`.
        reject_unbraided_nonunit_legs(
            provider,
            self.body.space.space().homspace(),
            legs,
            "twist",
            true,
        )?;
        let nout = self.codomain_rank();
        if let TypedData::Diagonal(spectrum) = &*self.body.data {
            // Compact arm, mirroring the erased `scaled_by_sector` route: a
            // bond space's two legs both carry the block's coupled sector, so
            // the per-block factor collapses to θ(sector)^|legs|. The space
            // is unchanged, so the payload may stay compact.
            let sector_factor = |sector: tenet_core::SectorId| -> f64 {
                legs.iter().map(|_| provider.twist_scalar(sector)).product()
            };
            if spectrum
                .iter()
                .all(|entry| sector_factor(entry.sector) == 1.0)
            {
                return Ok(self.clone());
            }
            let scaled = spectrum
                .iter()
                .map(|entry| {
                    let factor = D::from_real(sector_factor(entry.sector));
                    tenet_matrixalgebra::SectorSpectrum {
                        sector: entry.sector,
                        values: entry.values.iter().map(|&value| value * factor).collect(),
                    }
                })
                .collect();
            return Ok(self.with_spectrum_on(self.body.space.clone(), scaled));
        }
        if twist_is_identity_over_blocks(provider, self.body.space.space().structure(), nout, legs)?
        {
            return Ok(self.clone());
        }
        let mut data = self.dense_data().to_vec();
        scale_blocks_impl(self.body.space.space(), &mut data, &|key| match key {
            BlockKey::FusionTree(key) => twist_block_factor(provider, key, nout, legs),
            _ => 1.0,
        })?;
        Ok(Self {
            runtime: self.runtime.clone(),
            body: Arc::new(TypedTensorBody::dense(self.body.space.clone(), data)),
        })
    }

    /// TensorKit `flip(t, I)`:
    /// return a tensor isomorphic to `self` where the duality flag of each
    /// leg in `legs` (flat indices, codomain first; a leg listed twice is
    /// flipped twice, sequentially) is toggled,
    /// `space(t', i) = flip(space(t, i))`. The stored sectors and the block
    /// layout are unchanged; each fusion-tree block picks up the
    /// Z-isomorphism phase of TensorKit's fusion-tree `flip`
    /// per flipped leg with uncoupled sector `a` and pre-flip duality `d`
    /// (χ = Frobenius–Schur phase, θ = ribbon twist; both real for every
    /// rule in scope): codomain leg → `d ? χ·θ : 1`; domain leg →
    /// `d ? χ : θ`.
    ///
    /// Like TensorKit's, this `flip` is *not* an involution: flipping the
    /// same leg twice returns to the original spaces but can scale odd
    /// blocks (e.g. by θ = −1 on fermionic legs); only `flip⁴ = id` in
    /// general.
    ///
    /// One scaled copy of the dense payload into a fresh body, O(len); a
    /// compact spectrum factor materializes first (the flipped space is no
    /// longer a bond space, so the result cannot stay compact). The same
    /// facade narrowings as [`Self::twist`] apply: no adjoint arm, no device
    /// arm, SU(3) dead at the admission bound.
    ///
    /// # Errors
    ///
    /// [`Error::InvalidArgument`] when a leg is out of range (erased message,
    /// erased validation order — before the empty-list short-circuit; empty
    /// `legs` returns an identical clone). Otherwise [`Error::Operation`] /
    /// [`Error::Core`] from the layout derivation of the toggled hom space.
    pub fn flip(&self, legs: &[usize]) -> Result<Self, Error> {
        let rank = self.rank();
        if let Some(&leg) = legs.iter().find(|&&leg| leg >= rank) {
            return Err(Error::InvalidArgument(format!(
                "flip leg {leg} out of range for rank {rank}"
            )));
        }
        if legs.is_empty() {
            return Ok(self.clone());
        }
        let hom = self.body.space.space().homspace();
        // NoBraiding preflight (PR #620 review): flip's coefficients are
        // built from the same θ/χ — see `reject_unbraided_nonunit_legs`.
        reject_unbraided_nonunit_legs(self.body.space.provider(), hom, legs, "flip", false)?;
        let nout = hom.codomain().len();
        // Sequential semantics for repeated legs, from the helper shared
        // with the erased facade (#580 PR 5).
        let (new_hom, occurrences) = flip_toggled_homspace(hom, legs);
        let space = self.body.space.derive_from_final_homspace(new_hom)?;
        check_flip_layout_identity(
            self.body.space.space().structure(),
            space.space().structure(),
        )?;
        let provider = self.body.space.provider();
        let mut data = self.dense_data().to_vec();
        scale_blocks_impl(space.space(), &mut data, &|key| match key {
            BlockKey::FusionTree(key) => flip_block_factor(provider, key, nout, &occurrences),
            _ => 1.0,
        })?;
        Ok(Self {
            runtime: self.runtime.clone(),
            body: Arc::new(TypedTensorBody::dense(space, data)),
        })
    }

    /// TensorKit `insertleftunit(t, i; dual)`: inserts the canonical unit
    /// leg — the vacuum with degeneracy one, or its dual — at zero-based
    /// external slot `position`, following TensorKit's left seam convention
    /// (the codomain/domain seam belongs to the domain side). The trivial
    /// sector adds no block and reorders nothing, so the stored values are
    /// untouched.
    ///
    /// O(1) for a dense payload: the new body shares the payload allocation,
    /// exactly as TensorKit's `copy = false` default shares `t.data` for an
    /// ordinary `TensorMap`. A compact
    /// spectrum factor materializes into a fresh dense payload first (one
    /// copy) — the #613 Group 4 contract; TensorKit routes its
    /// `DiagonalTensorMap` through the generic similar+block-copy branch
    /// for the same reason.
    ///
    /// The `where R: CanonicalUnitFusionRule` bound is the provider's
    /// certification that its vacuum obeys the canonical unit laws — the
    /// hom-space transform and the layout validator both demand it, so an
    /// external provider opts in with one marker impl. No adjoint/device
    /// arms; SU(3) dead at the admission bound (see [`Self::twist`]).
    ///
    /// # Errors
    ///
    /// [`Error::InvalidArgument`] when `position` exceeds the rank (erased
    /// message shape, named for this receiver). Otherwise the layout
    /// derivation's and unit-correspondence validator's own classes,
    /// mapped exactly as the erased facade maps them.
    pub fn insert_left_unit(&self, position: usize, dual: bool) -> Result<Self, Error>
    where
        R: CanonicalUnitFusionRule,
    {
        self.insert_unit(
            UnitLegInsertion::Left { position, dual },
            "TensorMap::insert_left_unit",
        )
    }

    /// TensorKit `insertrightunit(t, i; dual)`: inserts the canonical unit
    /// leg at zero-based external slot `position`, following TensorKit's
    /// right seam convention (the codomain/domain seam belongs to the
    /// codomain side). Everything else — sharing, compact materialization,
    /// bounds, errors — exactly as [`Self::insert_left_unit`].
    pub fn insert_right_unit(&self, position: usize, dual: bool) -> Result<Self, Error>
    where
        R: CanonicalUnitFusionRule,
    {
        self.insert_unit(
            UnitLegInsertion::Right { position, dual },
            "TensorMap::insert_right_unit",
        )
    }

    /// Shared route of [`Self::insert_left_unit`] /
    /// [`Self::insert_right_unit`]: the tenet-core hom-space transform, the
    /// same checked layout-correspondence validator the erased facade runs,
    /// then a new body over the shared (or once-materialized) payload.
    fn insert_unit(&self, insertion: UnitLegInsertion, operation: &str) -> Result<Self, Error>
    where
        R: CanonicalUnitFusionRule,
    {
        let (UnitLegInsertion::Left { position, .. } | UnitLegInsertion::Right { position, .. }) =
            insertion;
        if position > self.rank() {
            return Err(Error::InvalidArgument(format!(
                "{operation}: position {position} exceeds rank {}",
                self.rank()
            )));
        }
        let provider = self.body.space.provider();
        let source_hom = self.body.space.space().homspace();
        let homspace = match insertion {
            UnitLegInsertion::Left { position, dual } => {
                source_hom.insert_left_unit(provider, position, dual)?
            }
            UnitLegInsertion::Right { position, dual } => {
                source_hom.insert_right_unit(provider, position, dual)?
            }
        };
        let destination = self.body.space.derive_from_final_homspace(homspace)?;
        validate_unit_layout_correspondence_checked(
            provider,
            (source_hom, self.body.space.space().structure()),
            (
                destination.space().homspace(),
                destination.space().structure(),
            ),
            insertion,
        )
        .map_err(map_checked_unit_layout_error)?;
        let data = self.shareable_dense_payload();
        Ok(Self {
            runtime: self.runtime.clone(),
            body: Arc::new(TypedTensorBody::with_shared_payload(destination, data)),
        })
    }

    /// TensorKit `removeunit(t, i)`: removes the canonical unit
    /// leg at flat external axis `axis`. The selected leg must contain
    /// exactly the vacuum sector with degeneracy one. This undoes
    /// [`Self::insert_left_unit`] / [`Self::insert_right_unit`]; sharing and
    /// compact materialization exactly as there — a dense insert→remove
    /// round trip returns to the original spaces on the original payload
    /// allocation.
    ///
    /// # Errors
    ///
    /// [`Error::InvalidArgument`] when `axis` is out of range or the leg is
    /// not a canonical unit leg (erased message shapes, named for this
    /// receiver). Otherwise the layout derivation's and validator's own
    /// classes, mapped exactly as the erased facade maps them.
    pub fn remove_unit(&self, axis: usize) -> Result<Self, Error>
    where
        R: CanonicalUnitFusionRule,
    {
        if axis >= self.rank() {
            return Err(Error::InvalidArgument(format!(
                "TensorMap::remove_unit: axis {axis} is out of range for rank {}",
                self.rank()
            )));
        }
        let provider = self.body.space.provider();
        let source_hom = self.body.space.space().homspace();
        let nout = source_hom.codomain().len();
        let leg = if axis < nout {
            &source_hom.codomain().legs()[axis]
        } else {
            &source_hom.domain().legs()[axis - nout]
        };
        if leg.sectors() != [provider.vacuum()] || leg.degeneracy(provider.vacuum()) != Some(1) {
            return Err(Error::InvalidArgument(format!(
                "TensorMap::remove_unit: axis {axis} is not a canonical unit leg"
            )));
        }
        // The insertion that this removal undoes, for the correspondence
        // validator: a codomain leg is the right seam's insertion, a domain
        // leg the left seam's — same reconstruction as the erased facade.
        let insertion = if axis < nout {
            UnitLegInsertion::Right {
                position: axis,
                dual: leg.is_dual(),
            }
        } else {
            UnitLegInsertion::Left {
                position: axis,
                dual: leg.is_dual(),
            }
        };
        let homspace = source_hom.remove_unit(provider, axis)?;
        let destination = self.body.space.derive_from_final_homspace(homspace)?;
        validate_unit_layout_correspondence_checked(
            provider,
            (
                destination.space().homspace(),
                destination.space().structure(),
            ),
            (source_hom, self.body.space.space().structure()),
            insertion,
        )
        .map_err(map_checked_unit_layout_error)?;
        let data = self.shareable_dense_payload();
        Ok(Self {
            runtime: self.runtime.clone(),
            body: Arc::new(TypedTensorBody::with_shared_payload(destination, data)),
        })
    }

    /// The payload a space-only rewrite (unit-leg insert/remove) may install
    /// in its new body: a dense payload is shared at pointer cost; a compact
    /// spectrum is materialized into a **fresh** dense payload first (one
    /// copy) — the #613 Group 4 contract. Never the body-local
    /// `dense_cache`: that buffer belongs to this body's space/payload
    /// pairing and only lends a borrowed slice (see the
    /// [`TypedTensorBody::data`] rationale).
    ///
    /// Infallible for the reason [`Self::dense_data`] is: the diagonal fill
    /// is total on a bond space this module built from that same spectrum.
    fn shareable_dense_payload(&self) -> Arc<TypedData<D>> {
        match &*self.body.data {
            TypedData::Dense(_) => Arc::clone(&self.body.data),
            TypedData::Diagonal(spectrum) => Arc::new(TypedData::Dense(
                tenet_matrixalgebra::diagonal_bond_data(
                    self.body.space.space(),
                    spectrum,
                    &|value| value,
                )
                .expect("diagonal fill is total on the stored bond space"),
            )),
        }
    }

    /// A zero tensor on the same spaces and dtype as `self` (TensorKit
    /// `zerovector`). Cheapest same-shape
    /// constructor: scales the storage by zero rather than re-deriving the
    /// block structure — exactly the erased
    /// [`crate::prelude::Tensor::zeros_like`], but infallible because the
    /// typed [`Self::scale`] is.
    ///
    /// Compact diagonal storage is preserved, as it is for every scaling.
    pub fn zeros_like(&self) -> Self {
        self.scale(D::from_real(0.0))
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
}

// Bound-free, like the accessor impl on `GradedSpace<R>`: an element-wise
// dtype conversion touches the stored payload only — no provider algebra, no
// layout derivation — so demanding the construction bounds here would be
// certification without a certificate to check.
impl<R> TensorMap<R, f64> {
    /// Widens to a c64 tensor map, imaginary parts zero (TensorKit
    /// `Base.complex`).
    ///
    /// Element-wise on the stored payload: a dense payload is widened in
    /// place-order, a compact spectrum factor maps spectrum-to-spectrum and
    /// **stays compact** (same as the erased `to_c64_storage` route). One
    /// `O(stored_len)` output allocation; the space is shared, not re-derived.
    ///
    /// Infallible, unlike the erased pair [`crate::prelude::Tensor::to_c64`] /
    /// `try_to_c64`: that split exists only for device residency, which the
    /// typed facade does not have — a deliberate narrowing, not a semantic
    /// difference.
    ///
    /// ```
    /// use std::sync::Arc;
    ///
    /// use tenet::core::{U1FusionRule, U1Irrep};
    /// use tenet::typed::{GradedSpace, Runtime, TensorMap};
    ///
    /// let runtime = Runtime::builder().build()?;
    /// let rule = Arc::new(U1FusionRule);
    /// let v = GradedSpace::try_new(
    ///     Arc::clone(&rule),
    ///     [(U1Irrep::new(0), 1), (U1Irrep::new(1), 2)],
    ///     false,
    /// )?;
    /// let t: TensorMap<_, f64> = TensorMap::rand(&runtime, [&v], [&v])?;
    ///
    /// let widened = t.to_c64();
    /// // The real part round-trips exactly; the imaginary part is zero.
    /// assert_eq!(widened.re().data(), t.data());
    /// assert_eq!(widened.im().norm()?, 0.0);
    /// # Ok::<(), tenet::typed::Error>(())
    /// ```
    pub fn to_c64(&self) -> TensorMap<R, num_complex::Complex64> {
        let widen = |&value: &f64| num_complex::Complex64::new(value, 0.0);
        let body = match &*self.body.data {
            TypedData::Dense(data) => {
                TypedTensorBody::dense(self.body.space.clone(), data.iter().map(widen).collect())
            }
            TypedData::Diagonal(spectrum) => TypedTensorBody::diagonal(
                self.body.space.clone(),
                map_spectrum_dtype(spectrum, |value| num_complex::Complex64::new(value, 0.0)),
            ),
        };
        TensorMap {
            runtime: self.runtime.clone(),
            body: Arc::new(body),
        }
    }
}

// Typed-first: the erased facade has no `re`/`im` counterpart, so there is no
// route to extract and cross-facade parity is impossible — the gates are law
// checks (`re(t) + i·im(t)` rebuilds `t`) instead. TensorKit's real-input
// branches (`real(t) = t`, `imag(t) = zerovector(t)` for a real scalartype)
// are statically unrepresentable here: these methods exist on the `Complex64`
// impl only, and `to_c64().re()` covers the round trip.
impl<R> TensorMap<R, num_complex::Complex64> {
    /// The element-wise real part, as an f64 tensor map on the same spaces
    /// (TensorKit `Base.real`: blockwise element-wise, result scalartype
    /// real).
    ///
    /// A compact spectrum factor maps spectrum-to-spectrum and stays compact.
    /// One `O(stored_len)` output allocation; the space is shared.
    pub fn re(&self) -> TensorMap<R, f64> {
        self.map_parts(|value| value.re)
    }

    /// The element-wise imaginary part, as an f64 tensor map on the same
    /// spaces (TensorKit `Base.imag`).
    ///
    /// A compact spectrum factor maps spectrum-to-spectrum and stays compact.
    /// One `O(stored_len)` output allocation; the space is shared.
    pub fn im(&self) -> TensorMap<R, f64> {
        self.map_parts(|value| value.im)
    }

    /// The shared body of [`Self::re`] / [`Self::im`]: one element-wise
    /// component map over whichever payload representation is stored.
    fn map_parts(&self, part: impl Fn(num_complex::Complex64) -> f64) -> TensorMap<R, f64> {
        let body = match &*self.body.data {
            TypedData::Dense(data) => TypedTensorBody::dense(
                self.body.space.clone(),
                data.iter().map(|&value| part(value)).collect(),
            ),
            TypedData::Diagonal(spectrum) => TypedTensorBody::diagonal(
                self.body.space.clone(),
                map_spectrum_dtype(spectrum, part),
            ),
        };
        TensorMap {
            runtime: self.runtime.clone(),
            body: Arc::new(body),
        }
    }
}

/// [`map_spectrum`]'s cross-dtype sibling: the same sector-and-length
/// preserving value map, for the conversions whose output dtype differs from
/// the input's — which is exactly why the two cannot share one signature.
fn map_spectrum_dtype<A: Copy, B>(
    spectrum: &[tenet_matrixalgebra::SectorSpectrum<A>],
    value_of: impl Fn(A) -> B,
) -> Vec<tenet_matrixalgebra::SectorSpectrum<B>> {
    spectrum
        .iter()
        .map(|entry| tenet_matrixalgebra::SectorSpectrum {
            sector: entry.sector,
            values: entry.values.iter().map(|&value| value_of(value)).collect(),
        })
        .collect()
}

/// Representation gates for [`TypedTensorBody`] (#580 PR 0).
///
/// These live inside the module on purpose: the properties under test are the
/// private layout — which `Arc` holds what — and asserting them from
/// `tests/` would mean publishing accessors the facade does not otherwise
/// need. Byte-level neutrality of this layout is *not* re-asserted here; it is
/// already pinned by the typed-versus-erased oracles in `tests/typed_facade.rs`
/// across U(1), SU(2), fZ2, Z2, an external provider and both `ProductFusionRule`
/// orders, and the dense-cache behavior by `tests/typed_diagonal_allocations.rs`.
#[cfg(test)]
mod representation_gates {
    use super::*;
    use tenet_core::{Z2FusionRule, Z2Irrep};

    fn fixture() -> TensorMap<Z2FusionRule, f64> {
        let runtime = Runtime::builder().dense_threads(1).build().unwrap();
        let leg =
            GradedSpace::try_new(Arc::new(Z2FusionRule), [(Z2Irrep::EVEN, 8)], false).unwrap();
        let mut state = 0x5eed_0580u64;
        TensorMap::from_block_fn(&runtime, [&leg], [&leg], move |_, _| {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            ((state >> 33) as f64) / (u32::MAX as f64) - 0.5
        })
        .unwrap()
    }

    #[test]
    fn clone_copies_no_payload_bytes() {
        // What: `clone` is O(1) in the payload. Measured structurally rather
        // than by an allocator, which cannot distinguish "no copy" from "a
        // copy the size of a warm cache line".
        let tensor = fixture();
        let twin = tensor.clone();
        assert!(Arc::ptr_eq(&tensor.body, &twin.body));
        assert!(Arc::ptr_eq(&tensor.body.data, &twin.body.data));
        assert_eq!(tensor.data().as_ptr(), twin.data().as_ptr());
        // One payload, however many handles reach it.
        assert_eq!(Arc::strong_count(&tensor.body.data), 1);
        assert_eq!(Arc::strong_count(&tensor.body), 2);
    }

    /// A small fermionic fixture whose codomain leg 0 carries only the even
    /// sector (θ = 1 everywhere on it) while leg 1 and the domain leg carry
    /// the odd sector too — so one tensor exposes both twist short-circuit
    /// answers.
    fn fz2_fixture() -> TensorMap<tenet_core::FermionParityFusionRule, f64> {
        let runtime = Runtime::builder().dense_threads(1).build().unwrap();
        let provider = Arc::new(tenet_core::FermionParityFusionRule);
        let even_only =
            GradedSpace::try_new(Arc::clone(&provider), [(Z2Irrep::EVEN, 2)], false).unwrap();
        let mixed = GradedSpace::try_new(
            Arc::clone(&provider),
            [(Z2Irrep::EVEN, 1), (Z2Irrep::ODD, 2)],
            false,
        )
        .unwrap();
        let mut state = 0x5eed_0613u64;
        TensorMap::from_block_fn(&runtime, [&even_only, &mixed], [&mixed], move |_, _| {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            ((state >> 33) as f64) / (u32::MAX as f64) - 0.5
        })
        .unwrap()
    }

    #[test]
    fn unit_insert_and_remove_share_the_dense_payload_arc() {
        // What (#580 PR 5, gate 5): the O(1) property the PR 0 gate
        // `a_body_on_a_different_space_reuses_the_payload_allocation` proved
        // by hand-constructing a body, now proved through the real
        // operations it anticipated — a dense payload's `Arc` is shared
        // unchanged through an insert→remove round trip, and the fresh
        // bodies start with cold caches. Supersedes that PR 0 gate: this one
        // checks everything it did (payload reuse at pointer cost under a
        // rewritten space) minus the hand-built struct shape, which the real
        // operations now compile against anyway.
        let tensor = fixture();
        let inserted = tensor.insert_left_unit(1, false).unwrap();
        assert!(!Arc::ptr_eq(&tensor.body, &inserted.body));
        assert!(Arc::ptr_eq(&tensor.body.data, &inserted.body.data));
        assert!(inserted.body.dense_cache.get().is_none());
        let removed = inserted.remove_unit(1).unwrap();
        assert!(Arc::ptr_eq(&tensor.body.data, &removed.body.data));
        // One payload allocation, three bodies holding it.
        assert_eq!(Arc::strong_count(&tensor.body.data), 3);
        assert_eq!(tensor.data().as_ptr(), removed.data().as_ptr());
    }

    #[test]
    fn a_compact_payload_materializes_exactly_once_for_the_unit_ops() {
        // What (#580 PR 5, gate 5): the compact half of the #613 Group 4
        // contract — a `Diagonal` payload is materialized into a *fresh*
        // dense payload (one copy, never the body-local `dense_cache`
        // buffer), and the follow-up remove shares that dense `Arc` rather
        // than copying again.
        let s = fixture().svd_compact().unwrap().1;
        let warmed = s.data().as_ptr(); // warm the body-local cache first
        let inserted = s.insert_left_unit(0, false).unwrap();
        assert!(!Arc::ptr_eq(&s.body.data, &inserted.body.data));
        assert!(matches!(&*inserted.body.data, TypedData::Dense(_)));
        // Fresh buffer, not the cache the warm-up populated.
        assert_ne!(inserted.data().as_ptr(), warmed);
        let removed = inserted.remove_unit(0).unwrap();
        assert!(Arc::ptr_eq(&inserted.body.data, &removed.body.data));
    }

    #[test]
    fn twist_identity_short_circuit_shares_the_whole_body() {
        // What (#580 PR 5, gate 5): both identity answers allocate nothing —
        // the bosonic O(1) arm (Z2) and the fermionic per-block scan when no
        // requested leg touches a twisted sector (fZ2, even-only leg 0) both
        // return a body-sharing clone; a leg that does touch the odd sector
        // publishes a new body.
        let tensor = fixture();
        let twisted = tensor.twist(&[0, 1]).unwrap();
        assert!(Arc::ptr_eq(&tensor.body, &twisted.body));

        let fermionic = fz2_fixture();
        let untouched = fermionic.twist(&[0]).unwrap();
        assert!(Arc::ptr_eq(&fermionic.body, &untouched.body));
        let touched = fermionic.twist(&[1]).unwrap();
        assert!(!Arc::ptr_eq(&fermionic.body, &touched.body));
    }

    #[test]
    fn twist_on_a_compact_spectrum_stays_compact() {
        // What (#580 PR 5, gate 5): the compact twist arm scales
        // spectrum-per-sector and keeps `TypedData::Diagonal` — the space is
        // unchanged, so O(Σ_c k_c) storage survives — and its own identity
        // answer (θ ≡ 1 across the spectrum's sectors) is a body-sharing
        // clone.
        let s = fz2_fixture().svd_compact().unwrap().1;
        let twisted = s.twist(&[0]).unwrap();
        assert!(matches!(&*twisted.body.data, TypedData::Diagonal(_)));
        assert!(!Arc::ptr_eq(&s.body.data, &twisted.body.data));

        let bosonic_s = fixture().svd_compact().unwrap().1;
        let untouched = bosonic_s.twist(&[0]).unwrap();
        assert!(Arc::ptr_eq(&bosonic_s.body, &untouched.body));
    }

    #[test]
    fn a_written_payload_leaves_the_shared_one_untouched() {
        // What: clone-then-modify. Sharing is only sound if a write on one
        // handle cannot be seen through the other — every write route in this
        // module publishes a new payload rather than reaching through the `Arc`.
        let tensor = fixture();
        let twin = tensor.clone();
        let before: Vec<f64> = tensor.data().to_vec();

        let scaled = twin.scale(2.0);

        assert_eq!(tensor.data(), before.as_slice());
        assert_eq!(twin.data(), before.as_slice());
        assert_ne!(scaled.data().as_ptr(), tensor.data().as_ptr());
        assert!(!Arc::ptr_eq(&scaled.body.data, &tensor.body.data));
    }

    #[test]
    fn the_dense_cache_lives_per_body_not_in_the_payload_arc() {
        // What: cache placement. `dense_cache` sits in the body, outside the
        // payload `Arc`, so a body that shares a payload starts with a cold
        // cache and materializes for itself, yielding a distinct buffer. This
        // is a same-space check on purpose — it makes no unit-leg claim: a
        // `Diagonal` payload may never be reused under a rewritten space at
        // all (see the `data` field rationale on the Group 4 contract). It
        // does *not* gate the struct shape — the fresh `OnceLock` below is
        // hand-supplied, so any layout keeping the field compiles and passes.
        let s = fixture().svd_compact().unwrap().1;
        assert!(s.body.dense_cache.get().is_none(), "cache warm at birth");
        let materialized = s.data().as_ptr();
        assert!(s.body.dense_cache.get().is_some());

        // Hand-constructed body, same caveat as the gate above: shape changes
        // become compile errors.
        let reused = TensorMap {
            runtime: s.runtime.clone(),
            body: Arc::new(TypedTensorBody {
                space: s.body.space.clone(),
                data: Arc::clone(&s.body.data),
                dense_cache: std::sync::OnceLock::new(),
            }),
        };
        assert!(reused.body.dense_cache.get().is_none());
        assert_ne!(reused.data().as_ptr(), materialized);
    }
}
