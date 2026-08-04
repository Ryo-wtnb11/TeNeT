//! Provider-typed facade: spaces and tensor maps that keep the concrete
//! fusion-rule type `R` and speak the provider's own sector labels.
//!
//! This is the canonical user facade re-exported by [`crate::prelude`]. `R`
//! stays concrete through monomorphized construction, so any provider —
//! including one defined downstream — can drive it, and the categorical
//! identity of a tensor comes back as [`TypedSectorAdmission::Sector`] labels
//! instead of opaque [`tenet_core::SectorId`] keys. The engine itself never
//! sees a label; the codec is the single boundary where one enters or leaves.
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
//!   endomorphism since issue #577, through a blockwise Padé arm.
//! - **Outer multiplicity factorizations** remain outside this leaf. Checked
//!   `Generic` providers use the ordinary tree transforms, tensor product, and
//!   direct-owned `contract` / `compose` routes through retained provider
//!   authority. Lazy adjoint construction and factorizations still retain
//!   their multiplicity-free bounds.
//! - **Device execution** is absent, not device representation: the body can
//!   carry a non-host `S` through [`TensorMap<R, D, S>`], while public
//!   construction and arithmetic deliberately remain on the default `Vec<D>`
//!   storage. Non-host operations wait for an explicit, [`Runtime`]-dependent
//!   transfer/device leaf. [`tenet_core::Placement`] is diagnostic metadata;
//!   no operation dispatches on it.
//! - The **operator overloads** (`impl Add`, `impl Mul`) are out because they
//!   cannot return `Result`; panicking operators would contradict this
//!   facade's passthrough-error contract. Adding them later is not a breaking
//!   change.
//! - `conj` stays design-gated on its open correctness question for
//!   non-self-dual sectors. [`TensorMap::adjoint`] is the TensorKit-style lazy
//!   parent view for dense storage; compact diagonal storage keeps its direct
//!   `O(Σ_c k_c)` conjugation path.
//!
//! Adding any of them ahead of its review would bypass the gate that exists to
//! keep this surface deliberate.
//!
//! Construction consumes only the transactional checked admission path, so a
//! provider that reports an invalid or unrepresentable algebra fails with a
//! typed error and publishes no layout, cache, or admission state.

#![cfg_attr(
    not(feature = "cuda"),
    doc = "```compile_fail\nuse tenet::typed::CudaStorage;\n```"
)]
#![cfg_attr(
    not(feature = "cuda"),
    doc = "```compile_fail\nuse tenet::typed::TensorMap;\nfn no_cuda<R>(tensor: &TensorMap<R, f64>) { let _ = tensor.to_cuda(); }\n```"
)]

use std::collections::{HashMap, HashSet};
use std::hash::Hash;
use std::sync::{Arc, OnceLock};

use tenet_core::{
    validate_unit_layout_correspondence_checked, BlockKey, BlockRef, CanonicalUnitFusionRule,
    CheckedFusionAlgebra, CheckedGenericAdmissionMode, CheckedGenericStructureError,
    FusionAlgebraError, FusionProductSpace, FusionTreeHomSpace, MultiplicityFreeAdmissionMode,
    MultiplicityFreeRigidSymbols, MultiplicityIndex, ProductFusionRule, ProductSector,
    ProductSectorCodec, SectorId, SectorLeg, TypedSectorAdmission, UnitLegInsertion,
};
use tenet_core::{
    CheckedGenericFusion, CheckedGenericRigidSymbols, HostReadableStorage, Placement, TensorStorage,
};
#[cfg(feature = "cuda")]
use tenet_core::{CoupledSectorRegion, CoupledTreeExtent};
#[cfg(feature = "cuda")]
use tenet_dense::{cuda_gemm_region_into, CudaDenseContext, CudaDenseStorage};
#[cfg(feature = "cuda")]
use tenet_operations::StorageGemm;
use tenet_tensors::{
    tensorcontract_owned_checked_generic, tree_transform_dyn_owned_checked_generic_in_context,
    BoundDynamicFusionMapSpace, BoundDynamicTensorRef, DynamicFusionMapSpace, OutputAxisOrder,
    TensorContractSpec, TreeTransformOperation, TreeTransformOperationKind,
    ValidatedDynamicFusionLayout,
};

pub use tenet_core::SectorCodec;
#[cfg(feature = "racah-generated")]
pub use tenet_core::{SUNFusionRule, SUNFusionRuleError};
/// Flat f64 CUDA storage used by explicit typed ownership transfer.
#[cfg(feature = "cuda")]
pub use tenet_tensors::cuda::CudaStorage;
#[cfg(feature = "cuda")]
use tenet_tensors::cuda::CudaStorageGemm;
pub use tenet_tensors::CheckedGenericPlanError;

/// Re-exported so `use tenet::typed::*` is self-sufficient apart from the
/// provider: every fallible method here returns this error.
pub use crate::error::Error;
/// Re-exported for the same reason as [`Error`]: every constructor here takes
/// a runtime. Both types are also in [`crate::prelude`]; re-exporting them
/// here is what lets a caller glob-import this module alone. The canonical
/// [`TensorMap`] and [`GradedSpace`] are also re-exported by [`crate::prelude`].
pub use crate::runtime::Runtime;
/// Re-exported for the same reason as [`Error`] and [`Runtime`]:
/// [`TensorMap::svd_trunc`] takes one, so `use tenet::typed::*` would not be
/// self-sufficient without it.
pub use tenet_matrixalgebra::{Truncation, TruncationSpace};

use tenet_matrixalgebra::{BoundDynFactor, FactorScalar};

use crate::runtime::{Ctx, Ctxs};
use crate::tensor::{
    absorb_mapped, apply_fill, cat_homspace, check_flip_layout_identity, compile_cat_plan,
    coupled_region_pow_sum, flip_block_factor, flip_toggled_homspace, internal_layout_error,
    logical_adjoint_axes_to_parent, lower_adjoint_tree_transform_operation,
    map_checked_unit_layout_error, reject_unbraided_nonunit_legs, scale_blocks_impl,
    sector_regions, twist_block_factor, twist_factor_with_inverse, twist_is_identity_over_blocks,
    validate_norm_p, weighted_inner, weighted_trace, with_planar_axes, CatOperandLayout, CatSide,
    Fill, PlanarRequestKind,
};
#[cfg(feature = "cuda")]
use crate::tensor::{
    assemble_left_factor, assemble_right_factor, cuda_is_hermitian_region, cuda_qr_region,
    cuda_svd_region, decide_kept, fill_diagonal_values, typed_cuda_eigh_region, upload_selector,
};
pub use crate::tensor_core::CheckedGenericTensorProductError;
use crate::tensor_core::{
    pow_by_squaring, tensorcompose_owned_multiplicity_free,
    tensorcontract_oriented_multiplicity_free, tensorcontract_owned_multiplicity_free,
    tensorproduct_owned_checked_generic, tensorproduct_owned_multiplicity_free,
    tree_transform_owned_multiplicity_free, OrientedContractionKind,
};
use crate::RuntimeIdentity;

/// Scalar payloads supported by [`TensorMap`].
///
/// This trait is sealed; the supported scalar types are `f64` and
/// [`num_complex::Complex64`].
#[allow(private_bounds)]
pub trait TensorScalar: ScalarOps {}

impl TensorScalar for f64 {}
impl TensorScalar for num_complex::Complex64 {}

/// Internal scalar operations shared by typed tensor execution.
pub(crate) trait ScalarOps:
    FactorScalar + tenet_tensors::RecouplingCoefficientAction<f64>
{
    fn ctx_of<Key: Clone + Eq + Hash + Send + Sync + 'static>(
        ctxs: &mut Ctxs<Key>,
    ) -> &mut Ctx<Self, Key>;
    fn rand_unit(state: &mut u64) -> Self;
    fn abs_value(self) -> f64;
    fn exp_value(self) -> Self;
    fn recip_value(self) -> Self;
    fn sqrt_value(self) -> Result<Self, Error>;
}

impl ScalarOps for f64 {
    fn ctx_of<Key: Clone + Eq + Hash + Send + Sync + 'static>(
        ctxs: &mut Ctxs<Key>,
    ) -> &mut Ctx<Self, Key> {
        &mut ctxs.f64
    }

    fn rand_unit(state: &mut u64) -> Self {
        random_unit(state)
    }

    fn abs_value(self) -> f64 {
        self.abs()
    }

    fn exp_value(self) -> Self {
        self.exp()
    }

    fn recip_value(self) -> Self {
        1.0 / self
    }

    fn sqrt_value(self) -> Result<Self, Error> {
        if self < 0.0 {
            Err(Error::InvalidArgument(format!(
                "sqrt of a negative diagonal entry {self}; convert to c64 \
                 with to_c64() for the complex square root"
            )))
        } else {
            Ok(self.sqrt())
        }
    }
}

impl ScalarOps for num_complex::Complex64 {
    fn ctx_of<Key: Clone + Eq + Hash + Send + Sync + 'static>(
        ctxs: &mut Ctxs<Key>,
    ) -> &mut Ctx<Self, Key> {
        &mut ctxs.c64
    }

    fn rand_unit(state: &mut u64) -> Self {
        Self::new(random_unit(state), random_unit(state))
    }

    fn abs_value(self) -> f64 {
        self.norm()
    }

    fn exp_value(self) -> Self {
        self.exp()
    }

    fn recip_value(self) -> Self {
        Self::new(1.0, 0.0) / self
    }

    fn sqrt_value(self) -> Result<Self, Error> {
        Ok(self.sqrt())
    }
}

fn random_unit(state: &mut u64) -> f64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut value = *state;
    value = (value ^ (value >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    value ^= value >> 31;
    ((value >> 11) as f64) / ((1_u64 << 52) as f64) - 1.0
}

/// Direct-sums two sector legs by adding matching degeneracies.
pub(crate) fn oplus_sector_legs(lhs: &SectorLeg, rhs: &SectorLeg) -> Result<SectorLeg, Error> {
    if lhs.is_dual() != rhs.is_dual() {
        return Err(Error::InvalidArgument(
            "oplus: cannot direct-sum spaces of opposite duality (dualize one first)".into(),
        ));
    }
    let mut sectors: Vec<(SectorId, usize)> = lhs.iter().collect();
    for (sector, deg) in rhs.iter() {
        match sectors.iter_mut().find(|(s, _)| *s == sector) {
            Some(entry) => {
                entry.1 = entry.1.checked_add(deg).ok_or_else(|| {
                    Error::InvalidArgument(format!(
                        "oplus: degeneracy overflow for sector {sector:?}"
                    ))
                })?;
            }
            None => sectors.push((sector, deg)),
        }
    }
    sectors.retain(|&(_, deg)| deg > 0);
    sectors.sort_by_key(|(sector, _)| *sector);
    Ok(SectorLeg::new(sectors, lhs.is_dual()))
}

/// Fuses sector content, including fusion multiplicities, in sector-id order.
pub(crate) fn fuse_sector_content<R: CheckedFusionAlgebra + ?Sized>(
    rule: &R,
    left: &[(SectorId, usize)],
    right: &[(SectorId, usize)],
) -> Result<Vec<(SectorId, usize)>, tenet_core::FusionAlgebraError> {
    let mut out = std::collections::BTreeMap::<SectorId, usize>::new();
    for &(a, deg_a) in left {
        for &(b, deg_b) in right {
            for c in rule.try_fusion_channels(a, b)? {
                *out.entry(c).or_insert(0) += rule.try_nsymbol(a, b, c)? * deg_a * deg_b;
            }
        }
    }
    Ok(out.into_iter().collect())
}

#[cfg(all(test, feature = "cuda"))]
thread_local! {
    /// `(download_calls, device_partials_len, host_partials_len)`.
    static CUDA_REDUCTION_BUFFER_OBSERVATION:
        std::cell::Cell<Option<(usize, usize, usize)>> = const {
            std::cell::Cell::new(None)
        };
    /// `(payload_zero_uploads, coefficient_uploads, kernels)`.
    static CUDA_ARITHMETIC_OBSERVATION:
        std::cell::Cell<Option<(usize, usize, usize)>> = const {
            std::cell::Cell::new(None)
        };
    /// `(qr_calls, diagonal_values, selector_uploads, output_uploads,
    /// assembly_gemms, live_route_scratch, peak_route_scratch)`.
    static CUDA_QR_OBSERVATION: std::cell::Cell<Option<CudaQrObservation>> = const {
            std::cell::Cell::new(None)
        };
    /// `(successful_results, spectrum_scalars, final_storage_creations,
    /// live_route_scratch, peak_route_scratch)`.
    static CUDA_SVD_OBSERVATION: std::cell::Cell<Option<CudaSvdObservation>> = const {
            std::cell::Cell::new(None)
        };
    /// `(successful_results, spectrum_scalars, final_storage_creations,
    /// live_raw_factors, peak_raw_factors, live_raw_bytes, peak_raw_bytes)`.
    static CUDA_SVD_TRUNC_OBSERVATION: std::cell::Cell<Option<CudaSvdTruncObservation>> = const {
            std::cell::Cell::new(None)
        };
    static CUDA_SVD_TRUNC_EVENTS: std::cell::RefCell<Option<Vec<(&'static str, usize)>>> = const {
        std::cell::RefCell::new(None)
    };
    static CUDA_SVD_TRUNC_LOCK_DEPTH: std::cell::Cell<usize> = const {
        std::cell::Cell::new(0)
    };
    static CUDA_SVD_TRUNC_FINAL_EXTENTS: std::cell::RefCell<Option<Vec<usize>>> = const {
        std::cell::RefCell::new(None)
    };
    static CUDA_SVD_TRUNC_ALLOCATIONS: std::cell::RefCell<Option<Vec<(&'static str, usize)>>> = const {
        std::cell::RefCell::new(None)
    };
    static CUDA_SVD_TRUNC_RELEASES: std::cell::RefCell<Option<Vec<(usize, usize)>>> = const {
        std::cell::RefCell::new(None)
    };
    /// `(stage, one-based ordinal)` for operation-local failure injection.
    static CUDA_SVD_TRUNC_FAILURE: std::cell::Cell<Option<(&'static str, usize)>> = const {
        std::cell::Cell::new(None)
    };
    static CUDA_EIGH_FAILURE: std::cell::Cell<Option<(&'static str, usize)>> = const {
        std::cell::Cell::new(None)
    };
}

#[cfg(all(test, feature = "cuda"))]
type CudaQrObservation = (usize, usize, usize, usize, usize, usize, usize);
#[cfg(all(test, feature = "cuda"))]
type CudaSvdObservation = (usize, usize, usize, usize, usize);
#[cfg(all(test, feature = "cuda"))]
type CudaSvdTruncObservation = (usize, usize, usize, usize, usize, usize, usize);

#[cfg(all(test, feature = "cuda"))]
fn update_cuda_svd_trunc_observation(
    update: impl FnOnce(CudaSvdTruncObservation) -> CudaSvdTruncObservation,
) {
    CUDA_SVD_TRUNC_OBSERVATION.with(|observation| {
        if let Some(current) = observation.get() {
            observation.set(Some(update(current)));
        }
    });
}

#[cfg(all(test, feature = "cuda"))]
fn observe_cuda_svd_trunc_decomposition(values: usize) {
    update_cuda_svd_trunc_observation(
        |(results, total, creations, live, peak, bytes, peak_bytes)| {
            (
                results + 1,
                total + values,
                creations,
                live,
                peak,
                bytes,
                peak_bytes,
            )
        },
    );
}

#[cfg(all(test, feature = "cuda"))]
fn observe_cuda_svd_trunc_final_storage_creation() {
    update_cuda_svd_trunc_observation(
        |(results, total, creations, live, peak, bytes, peak_bytes)| {
            (results, total, creations + 1, live, peak, bytes, peak_bytes)
        },
    );
}

#[cfg(all(test, feature = "cuda"))]
pub(crate) fn observe_cuda_svd_trunc_allocation(kind: &'static str, extent: usize) {
    CUDA_SVD_TRUNC_ALLOCATIONS.with(|allocations| {
        if let Some(allocations) = allocations.borrow_mut().as_mut() {
            allocations.push((kind, extent));
        }
    });
}

#[cfg(all(test, feature = "cuda"))]
fn observe_cuda_svd_trunc_release() {
    CUDA_SVD_TRUNC_OBSERVATION.with(|observation| {
        let Some((_, _, _, live, _, bytes, _)) = observation.get() else {
            return;
        };
        CUDA_SVD_TRUNC_RELEASES.with(|releases| {
            if let Some(releases) = releases.borrow_mut().as_mut() {
                releases.push((live, bytes));
            }
        });
    });
}

#[cfg(all(test, feature = "cuda"))]
fn observe_cuda_svd_trunc_event(event: &'static str) {
    let depth = CUDA_SVD_TRUNC_LOCK_DEPTH.with(|depth| depth.get());
    CUDA_SVD_TRUNC_EVENTS.with(|events| {
        if let Some(events) = events.borrow_mut().as_mut() {
            events.push((event, depth));
        }
    });
}

#[cfg(all(test, feature = "cuda"))]
struct CudaSvdTruncLockObservationGuard;

#[cfg(all(test, feature = "cuda"))]
impl CudaSvdTruncLockObservationGuard {
    fn new() -> Self {
        CUDA_SVD_TRUNC_LOCK_DEPTH.with(|depth| depth.set(depth.get() + 1));
        Self
    }
}

#[cfg(all(test, feature = "cuda"))]
impl Drop for CudaSvdTruncLockObservationGuard {
    fn drop(&mut self) {
        CUDA_SVD_TRUNC_LOCK_DEPTH.with(|depth| depth.set(depth.get() - 1));
    }
}

#[cfg(all(test, feature = "cuda"))]
fn update_cuda_svd_observation(update: impl FnOnce(CudaSvdObservation) -> CudaSvdObservation) {
    CUDA_SVD_OBSERVATION.with(|observation| {
        if let Some(current) = observation.get() {
            observation.set(Some(update(current)));
        }
    });
}

#[cfg(all(test, feature = "cuda"))]
pub(crate) fn observe_cuda_svd_decomposition(values: usize) {
    update_cuda_svd_observation(|(results, total, creations, live, peak)| {
        (results + 1, total + values, creations, live, peak)
    });
}

#[cfg(all(test, feature = "cuda"))]
fn observe_cuda_svd_final_storage_creation() {
    update_cuda_svd_observation(|(results, total, creations, live, peak)| {
        (results, total, creations + 1, live, peak)
    });
}

#[cfg(all(test, feature = "cuda"))]
fn observe_cuda_arithmetic(zero_uploads: usize, coefficient_uploads: usize, kernels: usize) {
    CUDA_ARITHMETIC_OBSERVATION.with(|observation| {
        if let Some((zeros, coefficients, calls)) = observation.get() {
            observation.set(Some((
                zeros + zero_uploads,
                coefficients + coefficient_uploads,
                calls + kernels,
            )));
        }
    });
}

#[cfg(all(test, feature = "cuda"))]
fn update_cuda_qr_observation(update: impl FnOnce(CudaQrObservation) -> CudaQrObservation) {
    CUDA_QR_OBSERVATION.with(|observation| {
        if let Some(current) = observation.get() {
            observation.set(Some(update(current)));
        }
    });
}

#[cfg(all(test, feature = "cuda"))]
pub(crate) fn observe_cuda_qr_decomposition(diagonal_values: usize) {
    update_cuda_qr_observation(|(qr, diagonal, selectors, outputs, gemms, live, peak)| {
        (
            qr + 1,
            diagonal + diagonal_values,
            selectors,
            outputs,
            gemms,
            live,
            peak,
        )
    });
}

#[cfg(all(test, feature = "cuda"))]
pub(crate) fn observe_cuda_qr_selector_upload() {
    update_cuda_qr_observation(|(qr, diagonal, selectors, outputs, gemms, live, peak)| {
        (qr, diagonal, selectors + 1, outputs, gemms, live, peak)
    });
}

#[cfg(all(test, feature = "cuda"))]
pub(crate) fn observe_cuda_qr_assembly_gemm() {
    update_cuda_qr_observation(|(qr, diagonal, selectors, outputs, gemms, live, peak)| {
        (qr, diagonal, selectors, outputs, gemms + 1, live, peak)
    });
}

#[cfg(all(test, feature = "cuda"))]
fn observe_cuda_qr_output_upload() {
    update_cuda_qr_observation(|(qr, diagonal, selectors, outputs, gemms, live, peak)| {
        (qr, diagonal, selectors, outputs + 1, gemms, live, peak)
    });
}

#[cfg(feature = "cuda")]
struct TypedCudaQrScratch {
    left: CudaDenseStorage,
    right: CudaDenseStorage,
    selector: CudaStorage,
}

#[cfg(feature = "cuda")]
impl TypedCudaQrScratch {
    fn new(left: CudaDenseStorage, right: CudaDenseStorage, selector: CudaStorage) -> Self {
        #[cfg(test)]
        update_cuda_qr_observation(|(qr, diagonal, selectors, outputs, gemms, live, peak)| {
            let live = live + 1;
            (
                qr,
                diagonal,
                selectors,
                outputs,
                gemms,
                live,
                peak.max(live),
            )
        });
        Self {
            left,
            right,
            selector,
        }
    }
}

#[cfg(all(test, feature = "cuda"))]
impl Drop for TypedCudaQrScratch {
    fn drop(&mut self) {
        update_cuda_qr_observation(|(qr, diagonal, selectors, outputs, gemms, live, peak)| {
            (qr, diagonal, selectors, outputs, gemms, live - 1, peak)
        });
    }
}

#[cfg(feature = "cuda")]
struct TypedCudaSvdScratch {
    left: CudaDenseStorage,
    right: CudaDenseStorage,
    selector: CudaStorage,
}

#[cfg(feature = "cuda")]
impl TypedCudaSvdScratch {
    fn new(left: CudaDenseStorage, right: CudaDenseStorage, selector: CudaStorage) -> Self {
        #[cfg(test)]
        update_cuda_svd_observation(|(results, total, creations, live, peak)| {
            let live = live + 1;
            (results, total, creations, live, peak.max(live))
        });
        Self {
            left,
            right,
            selector,
        }
    }
}

#[cfg(all(test, feature = "cuda"))]
impl Drop for TypedCudaSvdScratch {
    fn drop(&mut self) {
        update_cuda_svd_observation(|(results, total, creations, live, peak)| {
            (results, total, creations, live - 1, peak)
        });
    }
}

#[cfg(feature = "cuda")]
struct TypedCudaSvdRetainedFactors {
    left: CudaDenseStorage,
    right: CudaDenseStorage,
    #[cfg(test)]
    bytes: usize,
}

#[cfg(feature = "cuda")]
impl TypedCudaSvdRetainedFactors {
    fn new(
        left: CudaDenseStorage,
        right: CudaDenseStorage,
        rows: usize,
        cols: usize,
        rank: usize,
    ) -> Result<Self, Error> {
        #[cfg(not(test))]
        let _ = (rows, cols, rank);
        #[cfg(test)]
        let bytes = rows
            .checked_add(cols)
            .and_then(|sum| sum.checked_mul(rank))
            .and_then(|elements| elements.checked_mul(std::mem::size_of::<f64>()))
            .ok_or_else(|| internal_layout_error("retained CUDA SVD factor bytes overflow"))?;
        #[cfg(test)]
        update_cuda_svd_trunc_observation(
            |(results, total, creations, live, peak, live_bytes, peak_bytes)| {
                let live = live + 1;
                let live_bytes = live_bytes + bytes;
                (
                    results,
                    total,
                    creations,
                    live,
                    peak.max(live),
                    live_bytes,
                    peak_bytes.max(live_bytes),
                )
            },
        );
        Ok(Self {
            left,
            right,
            #[cfg(test)]
            bytes,
        })
    }
}

#[cfg(all(test, feature = "cuda"))]
impl Drop for TypedCudaSvdRetainedFactors {
    fn drop(&mut self) {
        update_cuda_svd_trunc_observation(
            |(results, total, creations, live, peak, bytes, peak_bytes)| {
                (
                    results,
                    total,
                    creations,
                    live - 1,
                    peak,
                    bytes - self.bytes,
                    peak_bytes,
                )
            },
        );
    }
}

#[cfg(feature = "cuda")]
#[derive(Clone, Copy)]
struct TypedCudaQrRoute {
    source: usize,
    left: usize,
    right: usize,
    rank: usize,
}

#[cfg(feature = "cuda")]
struct TypedCudaQrPlan<R> {
    left_space: BoundDynamicFusionMapSpace<R>,
    right_space: BoundDynamicFusionMapSpace<R>,
    source_regions: Arc<[CoupledSectorRegion]>,
    left_regions: Arc<[CoupledSectorRegion]>,
    right_regions: Arc<[CoupledSectorRegion]>,
    routes: Vec<TypedCudaQrRoute>,
}

#[cfg(feature = "cuda")]
#[derive(Clone, Copy)]
struct TypedCudaSvdTruncRoute {
    source: usize,
    left: usize,
    right: usize,
    full_rank: usize,
    kept: usize,
}

#[cfg(feature = "cuda")]
struct TypedCudaSvdTruncPlan<R> {
    left_space: BoundDynamicFusionMapSpace<R>,
    middle_space: BoundDynamicFusionMapSpace<R>,
    right_space: BoundDynamicFusionMapSpace<R>,
    source_regions: Arc<[CoupledSectorRegion]>,
    left_regions: Arc<[CoupledSectorRegion]>,
    right_regions: Arc<[CoupledSectorRegion]>,
    routes: Vec<TypedCudaSvdTruncRoute>,
}

#[cfg(feature = "cuda")]
fn cuda_qr_tree_extents_match(
    source: &[CoupledTreeExtent],
    factor: &[CoupledTreeExtent],
) -> Result<bool, Error> {
    if source.len() != factor.len() {
        return Ok(false);
    }
    let mut matched = vec![false; factor.len()];
    for source_tree in source {
        let source_extent = source_tree.extent()?;
        let mut match_index = None;
        for (index, factor_tree) in factor.iter().enumerate() {
            if !matched[index]
                && source_tree.tree() == factor_tree.tree()
                && source_extent == factor_tree.extent()?
            {
                match_index = Some(index);
                break;
            }
        }
        let Some(index) = match_index else {
            return Ok(false);
        };
        matched[index] = true;
    }
    Ok(matched.into_iter().all(|is_matched| is_matched))
}

#[cfg(feature = "cuda")]
fn validate_cuda_svd_middle_regions<R>(
    plan: &TypedCudaQrPlan<R>,
    middle_regions: &[CoupledSectorRegion],
) -> Result<(), Error> {
    if middle_regions.len() != plan.routes.len() {
        return Err(internal_layout_error(
            "compact SVD diagonal factor does not have exactly one region per route",
        ));
    }
    let mut by_sector = HashMap::with_capacity(middle_regions.len());
    for region in middle_regions {
        if by_sector.insert(region.coupled(), region).is_some() {
            return Err(internal_layout_error(
                "compact SVD diagonal factor contains a duplicate coupled sector",
            ));
        }
    }
    for route in &plan.routes {
        let source = &plan.source_regions[route.source];
        let middle = by_sector.get(&source.coupled()).ok_or_else(|| {
            internal_layout_error("compact SVD diagonal factor is missing a source sector")
        })?;
        let range_len = middle
            .range()
            .end
            .checked_sub(middle.range().start)
            .ok_or_else(|| {
                internal_layout_error("compact SVD diagonal factor has an invalid region range")
            })?;
        let expected_len = route.rank.checked_mul(route.rank).ok_or_else(|| {
            internal_layout_error("compact SVD diagonal factor region length overflows")
        })?;
        if (middle.rows(), middle.cols()) != (route.rank, route.rank)
            || range_len != expected_len
            || !cuda_qr_tree_extents_match(middle.row_trees(), middle.col_trees())?
        {
            return Err(internal_layout_error(
                "compact SVD diagonal factor region does not match its source route",
            ));
        }
    }
    Ok(())
}

#[cfg(feature = "cuda")]
fn cuda_qr_diagonal_sign(value: f64) -> f64 {
    if value < 0.0 {
        -1.0
    } else {
        1.0
    }
}

#[cfg(feature = "cuda")]
fn validate_cuda_reduction_placement(
    expected: Placement,
    lhs: Placement,
    rhs: Placement,
) -> Result<(), Error> {
    if lhs != expected || rhs != expected {
        return Err(Error::PlacementMismatch);
    }
    Ok(())
}

#[cfg(feature = "cuda")]
fn download_cuda_reduction_partials(
    partials: &CudaStorage,
    cuda: &CudaDenseContext,
) -> Result<Vec<f64>, Error> {
    #[cfg(test)]
    let device_len = partials.len();
    #[cfg(test)]
    let observed_call = CUDA_REDUCTION_BUFFER_OBSERVATION.with(|observation| {
        observation.get().map(|(calls, _, _)| {
            let calls = calls + 1;
            observation.set(Some((calls, device_len, 0)));
            calls
        })
    });
    let values = partials.download(cuda)?;
    #[cfg(test)]
    if let Some(calls) = observed_call {
        CUDA_REDUCTION_BUFFER_OBSERVATION.with(|observation| {
            observation.set(Some((calls, device_len, values.len())));
        });
    }
    Ok(values)
}

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
    /// Checked Generic tensor-product preparation or execution failed.
    TensorProduct(CheckedGenericTensorProductError<E>),
}

impl<E: core::fmt::Display> core::fmt::Display for GenericTensorError<E> {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Facade(error) => error.fmt(formatter),
            Self::Structure(error) => error.fmt(formatter),
            Self::Plan(error) => error.fmt(formatter),
            Self::TensorProduct(error) => error.fmt(formatter),
        }
    }
}

impl<E: std::error::Error + 'static> std::error::Error for GenericTensorError<E> {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Facade(error) => Some(error),
            Self::Structure(error) => Some(error),
            Self::Plan(error) => Some(error),
            Self::TensorProduct(error) => Some(error),
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

impl<E> From<CheckedGenericTensorProductError<E>> for GenericTensorError<E> {
    fn from(error: CheckedGenericTensorProductError<E>) -> Self {
        Self::TensorProduct(error)
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

/// Tensor-product execution selected by a provider-owned mode.
#[doc(hidden)]
pub trait TypedTensorProductDispatch<R, D>: TypedTensorModeDispatch<R>
where
    R: TypedSectorAdmission,
    D: TensorScalar,
{
    /// Executes the F-only product while preserving the left provider.
    fn tensor_product(
        lhs: &TensorMap<R, D>,
        rhs: &TensorMap<R, D>,
    ) -> Result<TensorMap<R, D>, Self::FacadeError>;
}

/// Contraction execution selected by a provider-owned mode.
#[doc(hidden)]
pub trait TypedTensorContractDispatch<R, D>: TypedTensorModeDispatch<R>
where
    R: TypedSectorAdmission,
    D: TensorScalar,
{
    fn contract(
        lhs: &TensorMap<R, D>,
        rhs: &TensorMap<R, D>,
        lhs_axes: &[usize],
        rhs_axes: &[usize],
        output_axes: &[usize],
    ) -> Result<TensorMap<R, D>, Self::FacadeError>;

    fn compose(
        lhs: &TensorMap<R, D>,
        rhs: &TensorMap<R, D>,
    ) -> Result<TensorMap<R, D>, Self::FacadeError>;
}

mod typed_admission_private {
    use super::{CheckedGenericAdmissionMode, MultiplicityFreeAdmissionMode};

    pub trait Sealed {}

    impl Sealed for MultiplicityFreeAdmissionMode {}
    impl Sealed for CheckedGenericAdmissionMode {}
}

#[doc(hidden)]
pub trait TypedSpaceModeDispatch<R>: TypedTensorModeDispatch<R>
where
    R: TypedSectorAdmission,
{
    fn vacuum(provider: &R) -> SectorId;
    fn fusion_channels(
        provider: &R,
        left: SectorId,
        right: SectorId,
    ) -> Result<Vec<SectorId>, <Self as TypedTensorModeDispatch<R>>::FacadeError>;
    fn nsymbol(
        provider: &R,
        left: SectorId,
        right: SectorId,
        coupled: SectorId,
    ) -> Result<usize, <Self as TypedTensorModeDispatch<R>>::FacadeError>;
    fn dim(
        provider: &R,
        sector: SectorId,
    ) -> Result<f64, <Self as TypedTensorModeDispatch<R>>::FacadeError>;
}

impl<R> TypedSpaceModeDispatch<R> for MultiplicityFreeAdmissionMode
where
    R: TypedSectorAdmission<Error = FusionAlgebraError, Mode = MultiplicityFreeAdmissionMode>
        + MultiplicityFreeRigidSymbols<Scalar = f64>
        + CheckedFusionAlgebra
        + SectorCodec,
{
    fn vacuum(provider: &R) -> SectorId {
        provider.vacuum()
    }

    fn fusion_channels(
        provider: &R,
        left: SectorId,
        right: SectorId,
    ) -> Result<Vec<SectorId>, TypedFacadeError<R>> {
        provider
            .try_fusion_channels(left, right)
            .map(|channels| channels.into_iter().collect())
            .map_err(Into::into)
    }

    fn nsymbol(
        provider: &R,
        left: SectorId,
        right: SectorId,
        coupled: SectorId,
    ) -> Result<usize, TypedFacadeError<R>> {
        provider
            .try_nsymbol(left, right, coupled)
            .map_err(Into::into)
    }

    fn dim(provider: &R, sector: SectorId) -> Result<f64, TypedFacadeError<R>> {
        Ok(provider.dim_scalar(sector))
    }
}

impl<R> TypedSpaceModeDispatch<R> for CheckedGenericAdmissionMode
where
    R: TypedSectorAdmission<
            Error = <R as CheckedGenericFusion>::Error,
            Mode = CheckedGenericAdmissionMode,
        > + CheckedGenericRigidSymbols<Scalar = f64>,
{
    fn vacuum(provider: &R) -> SectorId {
        CheckedGenericFusion::vacuum(provider)
    }

    fn fusion_channels(
        provider: &R,
        left: SectorId,
        right: SectorId,
    ) -> Result<Vec<SectorId>, TypedFacadeError<R>> {
        provider
            .try_fusion_channels(left, right)
            .map(|channels| channels.into_iter().collect())
            .map_err(<Self as TypedTensorModeDispatch<R>>::map_provider_error)
    }

    fn nsymbol(
        provider: &R,
        left: SectorId,
        right: SectorId,
        coupled: SectorId,
    ) -> Result<usize, TypedFacadeError<R>> {
        provider
            .try_nsymbol(left, right, coupled)
            .map_err(<Self as TypedTensorModeDispatch<R>>::map_provider_error)
    }

    fn dim(provider: &R, sector: SectorId) -> Result<f64, TypedFacadeError<R>> {
        provider
            .try_sqrt_dim_scalar(sector)
            .map(|sqrt_dim| sqrt_dim * sqrt_dim)
            .map_err(<Self as TypedTensorModeDispatch<R>>::map_provider_error)
    }
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
        let materialized = tensor.materialized_tensor_uncached()?;
        let body = materialized
            .owned_body()
            .expect("uncached materialization is owned");
        let mut lease = tensor.runtime.lease_context()?;
        let (space, data) = tree_transform_dyn_owned_checked_generic_in_context(
            lease.context().generic_lane::<D>().tree_context_mut(),
            operation,
            &body.space,
            body.materialized_dense_data(),
            D::from_real(1.0),
        )?;
        Ok(TensorMap {
            runtime: tensor.runtime.clone(),
            repr: owned_repr(TypedTensorBody::dense(space, data)),
        })
    }
}

impl<R, D> TypedTensorProductDispatch<R, D> for MultiplicityFreeAdmissionMode
where
    R: TypedSectorAdmission<Error = FusionAlgebraError, Mode = MultiplicityFreeAdmissionMode>
        + MultiplicityFreeRigidSymbols<Scalar = f64>
        + CheckedFusionAlgebra
        + SectorCodec
        + CanonicalUnitFusionRule,
    D: TensorScalar,
{
    fn tensor_product(
        lhs: &TensorMap<R, D>,
        rhs: &TensorMap<R, D>,
    ) -> Result<TensorMap<R, D>, Self::FacadeError> {
        let lhs_owned = lhs.materialized_tensor_uncached()?;
        let rhs_owned = rhs.materialized_tensor_uncached()?;
        let lhs_body = lhs_owned
            .owned_body()
            .expect("uncached materialization is owned");
        let rhs_body = rhs_owned
            .owned_body()
            .expect("uncached materialization is owned");
        let (space, data) = tensorproduct_owned_multiplicity_free(
            BoundDynamicTensorRef::try_new(&lhs_body.space, lhs_body.materialized_dense_data())?,
            BoundDynamicTensorRef::try_new(&rhs_body.space, rhs_body.materialized_dense_data())?,
        )?;
        Ok(TensorMap {
            runtime: lhs.runtime.clone(),
            repr: owned_repr(TypedTensorBody::dense(space, data)),
        })
    }
}

impl<R, D> TypedTensorProductDispatch<R, D> for CheckedGenericAdmissionMode
where
    R: TypedSectorAdmission<
            Error = <R as CheckedGenericFusion>::Error,
            Mode = CheckedGenericAdmissionMode,
        > + CheckedGenericRigidSymbols<Scalar = f64>,
    D: TensorScalar,
{
    fn tensor_product(
        lhs: &TensorMap<R, D>,
        rhs: &TensorMap<R, D>,
    ) -> Result<TensorMap<R, D>, Self::FacadeError> {
        let lhs_owned = lhs.materialized_tensor_uncached()?;
        let rhs_owned = rhs.materialized_tensor_uncached()?;
        let lhs_body = lhs_owned
            .owned_body()
            .expect("uncached materialization is owned");
        let rhs_body = rhs_owned
            .owned_body()
            .expect("uncached materialization is owned");
        let (space, data) = tensorproduct_owned_checked_generic(
            &lhs_body.space,
            lhs_body.materialized_dense_data(),
            &rhs_body.space,
            rhs_body.materialized_dense_data(),
        )?;
        Ok(TensorMap {
            runtime: lhs.runtime.clone(),
            repr: owned_repr(TypedTensorBody::dense(space, data)),
        })
    }
}

impl<R, D> TypedTensorContractDispatch<R, D> for MultiplicityFreeAdmissionMode
where
    R: TypedSectorAdmission<Error = FusionAlgebraError, Mode = MultiplicityFreeAdmissionMode>
        + MultiplicityFreeRigidSymbols<Scalar = f64>
        + CheckedFusionAlgebra
        + SectorCodec,
    D: TensorScalar,
{
    fn contract(
        lhs: &TensorMap<R, D>,
        rhs: &TensorMap<R, D>,
        lhs_axes: &[usize],
        rhs_axes: &[usize],
        output_axes: &[usize],
    ) -> Result<TensorMap<R, D>, Self::FacadeError> {
        contract_multiplicity_free(lhs, rhs, lhs_axes, rhs_axes, output_axes)
    }

    fn compose(
        lhs: &TensorMap<R, D>,
        rhs: &TensorMap<R, D>,
    ) -> Result<TensorMap<R, D>, Self::FacadeError> {
        compose_multiplicity_free(lhs, rhs)
    }
}

impl<R, D> TypedTensorContractDispatch<R, D> for CheckedGenericAdmissionMode
where
    R: TypedSectorAdmission<
            Error = <R as CheckedGenericFusion>::Error,
            Mode = CheckedGenericAdmissionMode,
        > + CheckedGenericRigidSymbols<Scalar = f64>,
    D: TensorScalar,
{
    fn contract(
        lhs: &TensorMap<R, D>,
        rhs: &TensorMap<R, D>,
        lhs_axes: &[usize],
        rhs_axes: &[usize],
        output_axes: &[usize],
    ) -> Result<TensorMap<R, D>, Self::FacadeError> {
        let (TypedTensorRepr::Owned(lhs_body), TypedTensorRepr::Owned(rhs_body)) =
            (&lhs.repr, &rhs.repr)
        else {
            return Err(Error::InvalidArgument(
                "checked Generic contraction currently requires direct owned tensors".to_string(),
            )
            .into());
        };
        let (space, data) = tensorcontract_owned_checked_generic(
            &lhs_body.space,
            lhs_body.materialized_dense_data(),
            &rhs_body.space,
            rhs_body.materialized_dense_data(),
            TensorContractSpec::new(lhs_axes, rhs_axes, OutputAxisOrder::from_axes(output_axes)),
        )?;
        Ok(TensorMap {
            runtime: lhs.runtime.clone(),
            repr: owned_repr(TypedTensorBody::dense(space, data)),
        })
    }

    fn compose(
        lhs: &TensorMap<R, D>,
        rhs: &TensorMap<R, D>,
    ) -> Result<TensorMap<R, D>, Self::FacadeError> {
        let lhs_axes = (lhs.codomain_rank()..lhs.rank()).collect::<Vec<_>>();
        let rhs_axes = (0..rhs.codomain_rank()).collect::<Vec<_>>();
        Self::contract(
            lhs,
            rhs,
            &lhs_axes,
            &rhs_axes,
            &(0..lhs.codomain_rank() + rhs.domain_rank()).collect::<Vec<_>>(),
        )
    }
}

fn contract_multiplicity_free<R, D>(
    lhs: &TensorMap<R, D>,
    rhs: &TensorMap<R, D>,
    lhs_axes: &[usize],
    rhs_axes: &[usize],
    output_axes: &[usize],
) -> Result<TensorMap<R, D>, Error>
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64> + CheckedFusionAlgebra + SectorCodec,
    D: TensorScalar,
{
    if let Some(compact) = lhs.try_contract_diagonal(rhs, lhs_axes, rhs_axes, output_axes)? {
        return Ok(compact);
    }
    let mut lease = lhs.runtime.lease_context()?;
    let output_order = OutputAxisOrder::from_axes(output_axes);
    let (space, data) =
        if let (TypedTensorRepr::Owned(lhs_body), TypedTensorRepr::Owned(rhs_body)) =
            (&lhs.repr, &rhs.repr)
        {
            tensorcontract_owned_multiplicity_free(
                lease.context().multiplicity_free_lane::<D>(),
                BoundDynamicTensorRef::try_new(
                    &lhs_body.space,
                    lhs_body.materialized_dense_data(),
                )?,
                BoundDynamicTensorRef::try_new(
                    &rhs_body.space,
                    rhs_body.materialized_dense_data(),
                )?,
                lhs_axes,
                rhs_axes,
                output_order,
            )?
        } else {
            let (lhs_operand, lhs_data) = lhs.fusion_operand_and_data();
            let (rhs_operand, rhs_data) = rhs.fusion_operand_and_data();
            tensorcontract_oriented_multiplicity_free(
                lease.context().multiplicity_free_lane::<D>(),
                lhs.logical_space(),
                lhs_operand,
                lhs_data,
                rhs.logical_space(),
                rhs_operand,
                rhs_data,
                lhs_axes,
                rhs_axes,
                output_order,
                OrientedContractionKind::Contract,
            )?
        };
    Ok(TensorMap {
        runtime: lhs.runtime.clone(),
        repr: owned_repr(TypedTensorBody::dense(space, data)),
    })
}

fn compose_multiplicity_free<R, D>(
    lhs: &TensorMap<R, D>,
    rhs: &TensorMap<R, D>,
) -> Result<TensorMap<R, D>, Error>
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64> + CheckedFusionAlgebra + SectorCodec,
    D: TensorScalar,
{
    if let Some(compact) = lhs.compose_compact(rhs)? {
        return Ok(compact);
    }
    let lhs_axes = (lhs.codomain_rank()..lhs.rank()).collect::<Vec<_>>();
    let rhs_axes = (0..rhs.codomain_rank()).collect::<Vec<_>>();
    let mut lease = lhs.runtime.lease_context()?;
    let (space, data) =
        if let (TypedTensorRepr::Owned(lhs_body), TypedTensorRepr::Owned(rhs_body)) =
            (&lhs.repr, &rhs.repr)
        {
            tensorcompose_owned_multiplicity_free(
                lease.context().multiplicity_free_lane::<D>(),
                BoundDynamicTensorRef::try_new(
                    &lhs_body.space,
                    lhs_body.materialized_dense_data(),
                )?,
                BoundDynamicTensorRef::try_new(
                    &rhs_body.space,
                    rhs_body.materialized_dense_data(),
                )?,
                &lhs_axes,
                &rhs_axes,
            )?
        } else {
            let (lhs_operand, lhs_data) = lhs.fusion_operand_and_data();
            let (rhs_operand, rhs_data) = rhs.fusion_operand_and_data();
            tensorcontract_oriented_multiplicity_free(
                lease.context().multiplicity_free_lane::<D>(),
                lhs.logical_space(),
                lhs_operand,
                lhs_data,
                rhs.logical_space(),
                rhs_operand,
                rhs_data,
                &lhs_axes,
                &rhs_axes,
                OutputAxisOrder::identity(),
                OrientedContractionKind::Compose,
            )?
        };
    Ok(TensorMap {
        runtime: lhs.runtime.clone(),
        repr: owned_repr(TypedTensorBody::dense(space, data)),
    })
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

impl<R> PartialEq for GradedSpace<R>
where
    R: TypedSectorAdmission,
{
    fn eq(&self, other: &Self) -> bool {
        TypedSectorAdmission::typed_rule_identity(self.provider())
            == TypedSectorAdmission::typed_rule_identity(other.provider())
            && self.leg == other.leg
    }
}

impl<R> Eq for GradedSpace<R> where R: TypedSectorAdmission {}

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
    /// **Dual-leg convention.** TensorKit interprets constructor labels
    /// through the orientation: `sectors(Vect[U1](1 => 1; dual = true))` is
    /// `-1`. TeNeT stores external sector content in [`SectorLeg`], so a dual
    /// constructor eagerly stores each label's provider dual. Readback and
    /// every tensor layout therefore see the same external labels.
    ///
    /// Order is irrelevant: the leg stores its sectors in the provider's
    /// [`tenet_core::SectorId`] order. A zero-degeneracy sector is absent from
    /// the result.
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
        Self::try_new_shared(provider, pairs, is_dual)
    }

    /// Owned-provider sibling of [`Self::try_new`]. The provider is placed in
    /// one [`Arc`] at entry, then follows the identical transactional
    /// validation and normalization path.
    pub fn try_new_owned<Pairs>(
        provider: R,
        pairs: Pairs,
        is_dual: bool,
    ) -> Result<Self, TypedFacadeError<R>>
    where
        Pairs: IntoIterator<Item = (R::Sector, usize)>,
    {
        Self::try_new_shared(Arc::new(provider), pairs, is_dual)
    }

    fn try_new_shared<Pairs>(
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

        if is_dual {
            for (id, _, _) in &mut encoded {
                *id = TypedSectorAdmission::try_dual_id(provider.as_ref(), *id)
                    .map_err(<R::Mode as TypedTensorModeDispatch<R>>::map_provider_error)?;
            }
            let mut duals: Vec<_> = encoded.iter().map(|(id, _, _)| *id).collect();
            duals.sort_unstable();
            if let Some(duplicate) = duals.windows(2).find(|pair| pair[0] == pair[1]) {
                return Err(Error::InvalidArgument(format!(
                    "dual map is not injective: sector {:?} appears multiple times",
                    duplicate[0]
                ))
                .into());
            }
        }

        let leg = SectorLeg::try_new(
            encoded.iter().map(|(id, _, degeneracy)| (*id, *degeneracy)),
            is_dual,
        )
        .map_err(|error| TypedFacadeError::<R>::from(Error::InvalidArgument(error.to_string())))?;
        Ok(Self { provider, leg })
    }

    /// The sector labels carried by this leg, in the provider's
    /// [`tenet_core::SectorId`] order — TensorKit `sectors(V)`. Constructor
    /// normalization and [`Self::try_dual`] both keep the stored ids equal to
    /// the leg's external sector content. One decode per sector, one `Vec`
    /// allocation per call.
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

    /// Degeneracy of one external sector label — TensorKit `dim(V, c)`.
    /// A representable label absent from the space has degeneracy zero.
    pub fn degeneracy(&self, sector: &R::Sector) -> Result<usize, TypedFacadeError<R>> {
        let id = TypedSectorAdmission::try_encode_label(self.provider.as_ref(), sector)
            .map_err(<R::Mode as TypedTensorModeDispatch<R>>::map_provider_error)?;
        Ok(self.leg.degeneracy(id).unwrap_or(0))
    }

    /// Whether this space carries `sector` with nonzero degeneracy.
    pub fn has_sector(&self, sector: &R::Sector) -> Result<bool, TypedFacadeError<R>> {
        self.degeneracy(sector).map(|degeneracy| degeneracy != 0)
    }

    /// The conjugate leg: every sector replaced by its dual (degeneracies
    /// carried along) and the dual flag flipped — TensorKit `dual(V)` / `V'`,
    /// which must satisfy the `dual(dual(V)) == V` contract of TensorKit's
    /// `dual(::VectorSpace)`. TensorKit only flips the flag and
    /// dualizes labels lazily on read; this leg rewrites its stored sector
    /// table eagerly — `O(k log k)`, one provider dual per sector plus the
    /// leg constructor's re-sort — and [`Self::sectors`] then reports the
    /// dual labels just as TK's `sectors(V')` does.
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

impl<R> GradedSpace<R>
where
    R: TypedSectorAdmission,
    R::Mode: TypedSpaceModeDispatch<R>,
{
    /// Exact quantum-dimension-weighted total dimension,
    /// `sum_c degeneracy(c) * dim(c)`, without integer rounding.
    pub fn dim(&self) -> Result<f64, TypedFacadeError<R>> {
        let mut total = 0.0;
        for (&sector, &degeneracy) in self.leg.sectors().iter().zip(self.leg.degeneracies()) {
            total += degeneracy as f64
                * <R::Mode as TypedSpaceModeDispatch<R>>::dim(self.provider(), sector)?;
        }
        Ok(total)
    }

    /// Unit space for this provider: one nondual vacuum sector of degeneracy
    /// one, retaining this space's exact provider allocation.
    pub fn unitspace(&self) -> Result<Self, TypedFacadeError<R>> {
        let vacuum = <R::Mode as TypedSpaceModeDispatch<R>>::vacuum(self.provider());
        TypedSectorAdmission::try_decode_label(self.provider(), vacuum)
            .map_err(<R::Mode as TypedTensorModeDispatch<R>>::map_provider_error)?;
        let leg = SectorLeg::try_new([(vacuum, 1)], false).map_err(|error| {
            TypedFacadeError::<R>::from(Error::InvalidArgument(error.to_string()))
        })?;
        Ok(Self {
            provider: Arc::clone(self.provider_arc()),
            leg,
        })
    }

    /// TensorKit `fuse(self, other)`: fuse external sector content and return
    /// a nondual space. Provider identity is checked before algebra queries.
    pub fn fuse(&self, other: &Self) -> Result<Self, TypedFacadeError<R>> {
        self.require_same_identity(other)?;
        let mut fused = std::collections::BTreeMap::<SectorId, usize>::new();
        for (left, left_deg) in self.leg.iter() {
            for (right, right_deg) in other.leg.iter() {
                let pair_deg = left_deg.checked_mul(right_deg).ok_or_else(|| {
                    TypedFacadeError::<R>::from(Error::InvalidArgument(
                        "fuse: degeneracy multiplication overflow".into(),
                    ))
                })?;
                for coupled in <R::Mode as TypedSpaceModeDispatch<R>>::fusion_channels(
                    self.provider(),
                    left,
                    right,
                )? {
                    let multiplicity = <R::Mode as TypedSpaceModeDispatch<R>>::nsymbol(
                        self.provider(),
                        left,
                        right,
                        coupled,
                    )?;
                    let contribution = pair_deg.checked_mul(multiplicity).ok_or_else(|| {
                        TypedFacadeError::<R>::from(Error::InvalidArgument(
                            "fuse: degeneracy multiplication overflow".into(),
                        ))
                    })?;
                    let entry = fused.entry(coupled).or_insert(0);
                    *entry = entry.checked_add(contribution).ok_or_else(|| {
                        TypedFacadeError::<R>::from(Error::InvalidArgument(format!(
                            "fuse: degeneracy overflow for sector {coupled:?}"
                        )))
                    })?;
                }
            }
        }
        fused.retain(|_, degeneracy| *degeneracy != 0);
        for &sector in fused.keys() {
            TypedSectorAdmission::try_decode_label(self.provider(), sector)
                .map_err(<R::Mode as TypedTensorModeDispatch<R>>::map_provider_error)?;
        }
        let leg = SectorLeg::try_new(fused, false).map_err(|error| {
            TypedFacadeError::<R>::from(Error::InvalidArgument(error.to_string()))
        })?;
        Ok(Self {
            provider: Arc::clone(self.provider_arc()),
            leg,
        })
    }

    /// TensorKit direct sum. Equal provider identity and orientation are
    /// required; degeneracy addition is checked.
    pub fn oplus(&self, other: &Self) -> Result<Self, TypedFacadeError<R>> {
        self.require_same_identity(other)?;
        let leg = oplus_sector_legs(&self.leg, &other.leg).map_err(TypedFacadeError::<R>::from)?;
        Ok(Self {
            provider: Arc::clone(self.provider_arc()),
            leg,
        })
    }

    fn require_same_identity(&self, other: &Self) -> Result<(), TypedFacadeError<R>> {
        if TypedSectorAdmission::typed_rule_identity(self.provider())
            != TypedSectorAdmission::typed_rule_identity(other.provider())
        {
            return Err(TypedFacadeError::<R>::from(Error::RuleMismatch));
        }
        Ok(())
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

    /// Raw logical leg snapshot used by typed network replay admission.
    #[doc(hidden)]
    pub fn network_sector_leg(&self) -> &SectorLeg {
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
        let source = self.source;
        let materialized = source.materialized_tensor_uncached()?;
        let data = materialized
            .owned_body()
            .expect("uncached materialization is owned")
            .materialized_dense_data();
        let blocks = self.blocks;
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
pub struct SvdTrunc<R: SectorCodec, D, S = Vec<D>> {
    /// Left isometry `u : codomain <- bond`.
    pub u: TensorMap<R, D, S>,
    /// Singular-value factor `s : bond <- bond`. Host storage keeps TensorKit's
    /// compact `DiagonalTensorMap` representation; CUDA storage returns the
    /// same factor as a dense device block diagonal because CUDA diagonal
    /// storage is not part of the current typed contract.
    pub s: TensorMap<R, D, S>,
    /// Right isometry `vh : bond <- domain`.
    pub vh: TensorMap<R, D, S>,
    /// Kept singular values per coupled sector, sorted by provider label.
    pub singular_values: Vec<SectorSpectrum<R::Sector>>,
    /// Quantum-dimension-weighted 2-norm of everything discarded.
    pub error: f64,
}

// Why hand-written, as for `TensorMap` itself: the derives would demand
// `R: Clone + Debug`, and neither is needed — the provider lives behind an
// `Arc` and its labels, not the rule, are what a diagnostic shows.
impl<R, D, S> Clone for SvdTrunc<R, D, S>
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

impl<R, D, S> core::fmt::Debug for SvdTrunc<R, D, S>
where
    R: SectorCodec,
    S: TensorStorage<D>,
{
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        // Every field is shown; the storage bound is exactly the one needed by
        // `TensorMap` to report its stored element count.
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
/// `D` is a type parameter, so one diagonal arm holds values of exactly the
/// payload type.
enum TypedData<D, S = Vec<D>> {
    /// The dense coupled-sector buffer every operation can read.
    Dense(S),
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
pub struct EigTrunc<R: SectorCodec, D: TensorScalar, S = Vec<<D as FactorScalar>::Eig>> {
    /// Eigenvalue factor `d : bond <- bond`, in compact diagonal storage.
    pub d: TensorMap<R, <D as FactorScalar>::Eig, S>,
    /// Eigenbasis `v : codomain <- bond`.
    pub v: TensorMap<R, <D as FactorScalar>::Eig, S>,
    /// Kept eigenvalues per coupled sector, sorted by provider label.
    pub eigenvalues: Vec<SectorSpectrum<R::Sector, num_complex::Complex64>>,
    /// Quantum-dimension-weighted 2-norm of the discarded `|eigenvalue|`s.
    pub error: f64,
}

// Hand-written for the reason [`SvdTrunc`]'s are.
impl<R, D, S> Clone for EigTrunc<R, D, S>
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

impl<R, D, S> core::fmt::Debug for EigTrunc<R, D, S>
where
    R: SectorCodec,
    D: TensorScalar,
    S: TensorStorage<<D as FactorScalar>::Eig>,
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
        repr: owned_repr(TypedTensorBody::diagonal(space, data)),
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
        repr: owned_repr(TypedTensorBody::dense(space, data)),
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
    add_spectrum_into(space, &mut data, spectrum, diagonal_factor)?;
    Ok(data)
}

fn add_spectrum_into<D>(
    space: &DynamicFusionMapSpace,
    data: &mut [D],
    spectrum: &[tenet_matrixalgebra::SectorSpectrum<D>],
    diagonal_factor: D,
) -> Result<(), Error>
where
    D: TensorScalar,
{
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
    Ok(())
}

/// Whether `space` is a bond space: rank one on each side, with the same leg on
/// both — the shape a compact spectrum can address, and the only shape whose
/// dense form is block-diagonal.
///
/// This is the guard TensorKit's `DiagonalTensorMap` gets for free from its
/// type.
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
pub struct EighTrunc<R: SectorCodec, D, S = Vec<D>> {
    /// Eigenvalue factor `d : bond <- bond`, in compact diagonal storage.
    pub d: TensorMap<R, D, S>,
    /// Eigenvector isometry `v : codomain <- bond`.
    pub v: TensorMap<R, D, S>,
    /// Kept eigenvalues per coupled sector, sorted by provider label. Real for
    /// both payload dtypes, as TensorKit's Hermitian `D` is.
    pub eigenvalues: Vec<SectorSpectrum<R::Sector>>,
    /// Quantum-dimension-weighted 2-norm of everything discarded.
    pub error: f64,
}

// Hand-written for the reason [`SvdTrunc`]'s are.
impl<R, D, S> Clone for EighTrunc<R, D, S>
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

impl<R, D, S> core::fmt::Debug for EighTrunc<R, D, S>
where
    R: SectorCodec,
    S: TensorStorage<D>,
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
struct TypedTensorBody<R, D, S = Vec<D>> {
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
    /// (`src/tensors/indexmanipulations.jl:124-136,158-195`). The Group 4
    /// slice (#580 PR 5) holds that contract in
    /// [`TensorMap::shareable_dense_payload`]: a dense
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
    /// reaching through the `Arc`).
    data: Arc<TypedData<D, S>>,
    /// Materialization of a [`TypedData::Diagonal`] payload into the dense
    /// coupled layout, computed at most once and shared by every clone of this
    /// body. Never populated for a dense payload.
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

impl<R, D, S> TypedTensorBody<R, D, S> {
    /// A body holding an already-dense payload.
    fn dense(space: BoundDynamicFusionMapSpace<R>, data: S) -> Self {
        Self::new(space, TypedData::Dense(data))
    }

    fn new(space: BoundDynamicFusionMapSpace<R>, data: TypedData<D, S>) -> Self {
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
    fn with_shared_payload(
        space: BoundDynamicFusionMapSpace<R>,
        data: Arc<TypedData<D, S>>,
    ) -> Self {
        Self {
            space,
            data,
            dense_cache: std::sync::OnceLock::new(),
        }
    }
}

impl<R, D> TypedTensorBody<R, D> {
    /// A body holding a compact spectrum payload.
    fn diagonal(
        space: BoundDynamicFusionMapSpace<R>,
        spectrum: Vec<tenet_matrixalgebra::SectorSpectrum<D>>,
    ) -> Self {
        Self::new(space, TypedData::Diagonal(spectrum))
    }
}

struct TypedAdjointView<R, D, S = Vec<D>> {
    parent: Arc<TypedTensorBody<R, D, S>>,
    logical_space: BoundDynamicFusionMapSpace<R>,
    // Lazy adjoint materialization is deliberately host-allocated. `S` names
    // the canonical parent payload; it is not a promise that arbitrary storage
    // can allocate a same-storage result.
    materialized: OnceLock<Arc<TypedTensorBody<R, D>>>,
    #[cfg(test)]
    materialized_body_builds: std::sync::atomic::AtomicUsize,
}

enum TypedTensorRepr<R, D, S = Vec<D>> {
    Owned(Arc<TypedTensorBody<R, D, S>>),
    Adjoint(Arc<TypedAdjointView<R, D, S>>),
}

fn owned_repr<R, D, S>(body: TypedTensorBody<R, D, S>) -> TypedTensorRepr<R, D, S> {
    TypedTensorRepr::Owned(Arc::new(body))
}

/// A block-sparse symmetric tensor map that keeps its provider type.
///
/// `D` is the payload dtype ([`f64`] or [`num_complex::Complex64`]) and is
/// independent of the provider's real categorical coefficient scalar — the
/// same separation TensorKit makes between a tensor's `T` and its sector type.
///
/// `S` is the owned payload storage and defaults to [`Vec<D>`]. Runtime
/// placement is diagnostic metadata from [`TensorStorage::placement`], not an
/// operation-dispatch mechanism; the current arithmetic impls remain on the
/// default host storage.
///
/// Host readback exists only when the storage implements
/// [`HostReadableStorage`]:
///
/// ```compile_fail
/// use tenet::core::{Placement, TensorStorage};
/// use tenet::typed::TensorMap;
///
/// struct DeviceStorage(usize);
/// impl TensorStorage<f64> for DeviceStorage {
///     fn len(&self) -> usize { self.0 }
///     fn placement(&self) -> Placement { Placement::Cuda(0) }
/// }
///
/// fn cannot_read<R>(tensor: &TensorMap<R, f64, DeviceStorage>) {
///     let _ = tensor.data();
/// }
/// ```
///
/// Naming a storage type and cloning its handle do not require the storage
/// itself to implement [`Clone`]; decomposition result types preserve that
/// storage parameter too:
///
/// ```
/// use tenet::core::{Placement, TensorStorage, U1FusionRule};
/// use tenet::typed::{EighTrunc, EigTrunc, SvdTrunc, TensorMap};
///
/// struct OpaqueStorage;
/// impl TensorStorage<f64> for OpaqueStorage {
///     fn len(&self) -> usize { 0 }
///     fn placement(&self) -> Placement { Placement::Cuda(0) }
/// }
///
/// fn clone_handle(tensor: &TensorMap<U1FusionRule, f64, OpaqueStorage>) {
///     let _: TensorMap<U1FusionRule, f64, OpaqueStorage> = tensor.clone();
/// }
///
/// fn name_results(
///     _: Option<SvdTrunc<U1FusionRule, f64, OpaqueStorage>>,
///     _: Option<EighTrunc<U1FusionRule, f64, OpaqueStorage>>,
///     _: Option<EigTrunc<U1FusionRule, f64, OpaqueStorage>>,
/// ) {}
/// ```
///
/// Cloning is cheap: the runtime handle and the shared body are both
/// reference-counted, and cloning does not require `S: Clone`.
pub struct TensorMap<R, D, S = Vec<D>> {
    runtime: Runtime,
    repr: TypedTensorRepr<R, D, S>,
}

impl<R, D, S> AsRef<Self> for TensorMap<R, D, S> {
    fn as_ref(&self) -> &Self {
        self
    }
}

/// Host tensor authority parked without retaining its [`Runtime`].
///
/// This is an internal ownership seam for the Runtime-owned `tensor!` workspace
/// pool. It retains only a validated provider-neutral layout and an owned dense
/// payload; detach/attach never copies or materializes tensor data, and attach
/// binds the layout to the current execution authority's exact provider.
#[doc(hidden)]
pub struct RuntimeDetachedTensorMap<D> {
    runtime: RuntimeIdentity,
    layout: ValidatedDynamicFusionLayout,
    data: Arc<TypedData<D, Vec<D>>>,
}

/// Storage route used by typed network replay admission.
#[doc(hidden)]
#[derive(Clone, Copy, Eq, PartialEq)]
pub enum NetworkReuseClass {
    OwnedDense,
    Compact,
    LazyAdjoint,
}

// Why hand-written: the derive would demand `R: Clone`, `D: Clone`, and
// `S: Clone`; none is needed because the representation sits behind `Arc`.
impl<R, D, S> Clone for TensorMap<R, D, S> {
    fn clone(&self) -> Self {
        Self {
            runtime: self.runtime.clone(),
            repr: match &self.repr {
                TypedTensorRepr::Owned(body) => TypedTensorRepr::Owned(Arc::clone(body)),
                TypedTensorRepr::Adjoint(view) => TypedTensorRepr::Adjoint(Arc::clone(view)),
            },
        }
    }
}

impl<R, D> TensorMap<R, D>
where
    R: tenet_core::FusionRule,
{
    /// Removes Runtime and provider ownership from an ordinary dense Host
    /// destination while preserving its validated layout and payload allocation.
    #[doc(hidden)]
    pub fn detach_runtime(self) -> Option<RuntimeDetachedTensorMap<D>> {
        let TensorMap { runtime, repr } = self;
        let TypedTensorRepr::Owned(body) = repr else {
            return None;
        };
        let body = Arc::try_unwrap(body).ok()?;
        if Arc::strong_count(&body.data) != 1 || !matches!(body.data.as_ref(), TypedData::Dense(_))
        {
            return None;
        }
        Some(RuntimeDetachedTensorMap {
            runtime: runtime.identity(),
            layout: body.space.validated_layout(),
            data: body.data,
        })
    }
}

impl<D> RuntimeDetachedTensorMap<D> {
    /// Dense allocation capacity and conservative dependent-layout bytes
    /// retained by this parked destination.
    ///
    /// Shared layout descendants are charged per parked destination. This can
    /// over-count, but it keeps the Runtime budget a true ceiling even when the
    /// workspace is the last owner of an Arc-backed layout allocation.
    #[doc(hidden)]
    pub fn retained_dense_capacity_bytes(&self) -> usize {
        let payload_capacity = match self.data.as_ref() {
            TypedData::Dense(data) => data.capacity().saturating_mul(std::mem::size_of::<D>()),
            TypedData::Diagonal(_) => {
                debug_assert!(false, "runtime-detached destinations are always dense");
                0
            }
        };
        payload_capacity
            .saturating_add(std::mem::size_of::<TypedData<D, Vec<D>>>())
            .saturating_add(2 * std::mem::size_of::<usize>())
            .saturating_add(self.layout.charged_retained_bytes())
    }

    /// Whether this parked tensor belongs to `runtime`.
    #[doc(hidden)]
    pub fn matches_runtime(&self, runtime: &Runtime) -> bool {
        self.runtime.matches(runtime)
    }

    /// Validates Runtime identity and layout rebinding against the current
    /// authority without consuming this parked destination.
    #[doc(hidden)]
    pub fn can_attach<R>(&self, runtime: &Runtime, authority: &TensorMap<R, D>) -> Result<(), Error>
    where
        R: tenet_core::FusionRule,
    {
        if !self.matches_runtime(runtime) {
            return Err(Error::RuntimeMismatch);
        }
        authority.logical_space().rebind_validated(&self.layout)?;
        Ok(())
    }

    /// Rebinds this provider-neutral destination to the current authority's
    /// exact provider allocation after [`Self::can_attach`] succeeds.
    #[doc(hidden)]
    pub fn attach_runtime<R>(
        self,
        runtime: &Runtime,
        authority: &TensorMap<R, D>,
    ) -> Result<TensorMap<R, D>, Error>
    where
        R: tenet_core::FusionRule,
    {
        if !self.matches_runtime(runtime) {
            return Err(Error::RuntimeMismatch);
        }
        let space = authority.logical_space().rebind_validated(&self.layout)?;
        Ok(TensorMap {
            runtime: runtime.clone(),
            repr: owned_repr(TypedTensorBody::with_shared_payload(space, self.data)),
        })
    }
}

impl<R, D, S> core::fmt::Debug for TensorMap<R, D, S>
where
    S: TensorStorage<D>,
{
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let stored = &self.storage_body().data;
        formatter
            .debug_struct("TensorMap")
            // Storage-shaped, deliberately: a compact spectrum payload reports
            // the values it stores, and forcing its dense materialization for a
            // `{:?}` would make a diagnostic the most expensive call on the type.
            .field(
                "elements",
                &match stored.as_ref() {
                    TypedData::Dense(data) => data.len(),
                    TypedData::Diagonal(spectrum) => {
                        spectrum.iter().map(|entry| entry.values.len()).sum()
                    }
                },
            )
            .finish_non_exhaustive()
    }
}

impl<R, D, S> TensorMap<R, D, S> {
    /// Classifies the representation produced after an optional adjoint.
    #[doc(hidden)]
    pub fn network_reuse_class(&self, adjoint: bool) -> NetworkReuseClass {
        match &self.repr {
            TypedTensorRepr::Owned(body) => match body.data.as_ref() {
                TypedData::Diagonal(_) => NetworkReuseClass::Compact,
                TypedData::Dense(_) if adjoint => NetworkReuseClass::LazyAdjoint,
                TypedData::Dense(_) => NetworkReuseClass::OwnedDense,
            },
            TypedTensorRepr::Adjoint(_) if adjoint => NetworkReuseClass::OwnedDense,
            TypedTensorRepr::Adjoint(_) => NetworkReuseClass::LazyAdjoint,
        }
    }

    /// Matches the metadata produced after an optional adjoint without allocation.
    #[doc(hidden)]
    pub fn network_input_metadata_matches(
        &self,
        adjoint: bool,
        expected_legs: &[SectorLeg],
        expected_class: NetworkReuseClass,
    ) -> bool
    where
        R: TypedSectorAdmission,
    {
        if self.network_reuse_class(adjoint) != expected_class || self.rank() != expected_legs.len()
        {
            return false;
        }
        (0..self.rank()).all(|axis| {
            let source_axis = if adjoint {
                (axis + self.codomain_rank()) % self.rank()
            } else {
                axis
            };
            let Some(actual) = self.network_source_leg(source_axis) else {
                return false;
            };
            actual == &expected_legs[axis]
        })
    }

    fn network_source_leg(&self, axis: usize) -> Option<&SectorLeg> {
        let homspace = self.logical_space().space().homspace();
        if axis < self.codomain_rank() {
            homspace.codomain().legs().get(axis)
        } else {
            homspace.domain().legs().get(axis - self.codomain_rank())
        }
    }

    fn storage_body(&self) -> &Arc<TypedTensorBody<R, D, S>> {
        match &self.repr {
            TypedTensorRepr::Owned(body) => body,
            TypedTensorRepr::Adjoint(view) => &view.parent,
        }
    }

    fn logical_space(&self) -> &BoundDynamicFusionMapSpace<R> {
        match &self.repr {
            TypedTensorRepr::Owned(body) => &body.space,
            TypedTensorRepr::Adjoint(view) => &view.logical_space,
        }
    }

    fn owned_body(&self) -> Option<&Arc<TypedTensorBody<R, D, S>>> {
        match &self.repr {
            TypedTensorRepr::Owned(body) => Some(body),
            TypedTensorRepr::Adjoint(_) => None,
        }
    }

    fn dense_adjoint_view(&self) -> Result<Self, Error>
    where
        R: MultiplicityFreeRigidSymbols<Scalar = f64>,
    {
        Ok(match &self.repr {
            TypedTensorRepr::Owned(parent) => {
                let logical_space = tenet_tensors::adjoint_bound_space_dyn(&parent.space)?;
                debug_assert!(Arc::ptr_eq(
                    parent.space.provider_arc(),
                    logical_space.provider_arc()
                ));
                Self {
                    runtime: self.runtime.clone(),
                    repr: TypedTensorRepr::Adjoint(Arc::new(TypedAdjointView {
                        parent: Arc::clone(parent),
                        logical_space,
                        materialized: OnceLock::new(),
                        #[cfg(test)]
                        materialized_body_builds: std::sync::atomic::AtomicUsize::new(0),
                    })),
                }
            }
            TypedTensorRepr::Adjoint(view) => Self {
                runtime: self.runtime.clone(),
                repr: TypedTensorRepr::Owned(Arc::clone(&view.parent)),
            },
        })
    }

    /// The provider allocation that owns this tensor's categorical layout.
    #[inline]
    pub fn provider(&self) -> &R {
        self.logical_space().provider()
    }

    /// Number of codomain legs.
    #[inline]
    pub fn codomain_rank(&self) -> usize {
        self.logical_space().space().homspace().codomain().len()
    }

    /// Number of domain legs.
    #[inline]
    pub fn domain_rank(&self) -> usize {
        self.logical_space().space().homspace().domain().len()
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

    /// One block in the tensor's logical coupled layout.
    ///
    /// Metadata only: reading an adjoint block does not materialize its data.
    pub fn block(&self, index: usize) -> Result<BlockRef<'_>, Error> {
        self.logical_space()
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

    /// The codomain legs, in axis order.
    ///
    /// Allocates: each call builds a fresh `Vec` and clones every leg's
    /// sector table (the provider travels by `Arc` bump). Hold the result
    /// rather than re-calling in a loop.
    pub fn codomain(&self) -> Vec<GradedSpace<R>> {
        self.legs(self.logical_space().space().homspace().codomain())
    }

    /// The domain legs, in axis order.
    ///
    /// Allocates per call, exactly as [`Self::codomain`].
    pub fn domain(&self) -> Vec<GradedSpace<R>> {
        self.legs(self.logical_space().space().homspace().domain())
    }

    /// The codomain legs, in axis order (TensorKit `codomain(t)`).
    /// Documented alias of
    /// [`Self::codomain`], using TensorKit's accessor name.
    #[inline]
    pub fn codomain_spaces(&self) -> Vec<GradedSpace<R>> {
        self.codomain()
    }

    /// The domain legs, in axis order (TensorKit `domain(t)`) — the
    /// spaces as written, i.e.
    /// *not* dualized. Documented alias of [`Self::domain`], using
    /// TensorKit's accessor name.
    #[inline]
    pub fn domain_spaces(&self) -> Vec<GradedSpace<R>> {
        self.domain()
    }

    fn legs(&self, product: &FusionProductSpace) -> Vec<GradedSpace<R>> {
        product
            .legs()
            .iter()
            .map(|leg| GradedSpace {
                provider: Arc::clone(self.logical_space().provider_arc()),
                leg: leg.clone(),
            })
            .collect()
    }

    /// Number of stored fusion-tree blocks.
    #[inline]
    pub fn block_count(&self) -> usize {
        self.logical_space().space().structure().block_count()
    }
}

impl<R, D, S> TensorMap<R, D, S>
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64>,
{
    /// Quantum-dimension-weighted total dimension of every leg, in flat order
    /// (codomain legs first, then domain legs) — TensorKit's `dim(space(t,
    /// i))` per leg.
    /// Contraction planners use it as a size/FLOP proxy.
    ///
    /// The rounding formula is
    /// `Σ_sector round(degeneracy * dim(sector))` per leg. The provider
    /// abstraction carries `dim_scalar` uniformly, so there is deliberately
    /// no group-specific branch.
    ///
    /// # Complexity
    ///
    /// `O(Σ_leg sectors)`; allocates the returned `Vec<usize>` only, never a
    /// payload.
    ///
    /// # Errors
    ///
    /// None today; the `Result` leaves room for future fallible dimension
    /// providers without changing the method shape.
    pub fn leg_dims(&self) -> Result<Vec<usize>, Error> {
        let hom = self.logical_space().space().homspace();
        let provider = self.logical_space().provider();
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
    /// [`Error::InvalidArgument`] when `axis >= rank()`.
    pub fn leg_dim(&self, axis: usize) -> Result<usize, Error> {
        let hom = self.logical_space().space().homspace();
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
        Ok(Self::weighted_leg_dim(self.logical_space().provider(), leg))
    }

    /// Quantum dimensions are generally irrational (SU(2) `sqrt` products,
    /// anyonic golden ratios), so the per-sector weight is computed in `f64`
    /// and rounded once.
    fn weighted_leg_dim(provider: &R, leg: &SectorLeg) -> usize {
        leg.sectors()
            .iter()
            .zip(leg.degeneracies())
            .map(|(&sector, &degeneracy)| {
                (degeneracy as f64 * provider.dim_scalar(sector)).round() as usize
            })
            .sum()
    }
}

impl<R, D, S> TensorMap<R, D, S>
where
    S: TensorStorage<D>,
{
    /// Reports where the canonical payload lives.
    ///
    /// This is diagnostic metadata only. Arithmetic never dispatches on
    /// placement, and transfers are always explicit and type-changing.
    pub fn placement(&self) -> Placement {
        match self.storage_body().data.as_ref() {
            TypedData::Dense(storage) => storage.placement(),
            TypedData::Diagonal(_) => Placement::Host,
        }
    }
}

#[cfg(feature = "cuda")]
impl<R> TensorMap<R, f64> {
    /// Uploads host f64 ownership to this tensor's Runtime CUDA context.
    ///
    /// Dense storage uploads directly. Compact diagonal storage is expanded
    /// operation-locally and becomes dense on device; the source's reusable
    /// dense cache stays cold, and a roundtrip remains dense rather than
    /// recovering compactness. A lazy adjoint transfers only its canonical
    /// parent and rebuilds a cold lazy view over the device parent.
    ///
    /// CUDA transfer is deliberately unavailable for c64 payloads:
    ///
    /// ```compile_fail
    /// use tenet::core::U1FusionRule;
    /// use tenet::typed::TensorMap;
    ///
    /// fn no_c64_upload(tensor: &TensorMap<U1FusionRule, num_complex::Complex64>) {
    ///     let _ = tensor.to_cuda();
    /// }
    /// ```
    pub fn to_cuda(&self) -> Result<TensorMap<R, f64, CudaStorage>, Error> {
        let state = self.runtime.lock();
        let cuda = state.cuda.as_ref().ok_or_else(|| {
            Error::InvalidArgument(
                "this runtime was built without a CUDA device; use \
                 Runtime::builder().cuda(device)"
                    .to_string(),
            )
        })?;
        let upload = |body: &Arc<TypedTensorBody<R, f64>>| {
            let storage = match body.data.as_ref() {
                TypedData::Dense(data) => CudaStorage::upload(cuda, data)?,
                TypedData::Diagonal(spectrum) => {
                    let dense = tenet_matrixalgebra::diagonal_bond_data(
                        body.space.space(),
                        spectrum,
                        &|value| value,
                    )?;
                    CudaStorage::upload(cuda, &dense)?
                }
            };
            Ok::<_, Error>(Arc::new(TypedTensorBody::dense(
                body.space.clone(),
                storage,
            )))
        };

        let repr = match &self.repr {
            TypedTensorRepr::Owned(body) => TypedTensorRepr::Owned(upload(body)?),
            TypedTensorRepr::Adjoint(view) => {
                TypedTensorRepr::Adjoint(Arc::new(TypedAdjointView {
                    parent: upload(&view.parent)?,
                    logical_space: view.logical_space.clone(),
                    materialized: OnceLock::new(),
                    #[cfg(test)]
                    materialized_body_builds: std::sync::atomic::AtomicUsize::new(0),
                }))
            }
        };
        Ok(TensorMap {
            runtime: self.runtime.clone(),
            repr,
        })
    }
}

#[cfg(feature = "cuda")]
impl<R> TensorMap<R, f64, CudaStorage> {
    /// Downloads device f64 ownership into one final dense host buffer.
    ///
    /// A lazy adjoint downloads only its canonical parent and rebuilds a cold
    /// host lazy view. No receiver-sized logical adjoint is materialized.
    /// Device storage is never implicitly host-readable:
    ///
    /// ```compile_fail
    /// use tenet::core::U1FusionRule;
    /// use tenet::typed::{CudaStorage, TensorMap};
    ///
    /// fn no_device_slice(tensor: &TensorMap<U1FusionRule, f64, CudaStorage>) {
    ///     let _ = tensor.data();
    /// }
    /// ```
    pub fn to_host(&self) -> Result<TensorMap<R, f64>, Error> {
        let state = self.runtime.lock();
        let cuda = state.cuda.as_ref().ok_or_else(|| {
            Error::InvalidArgument("this runtime was built without a CUDA device".to_string())
        })?;
        let download = |body: &Arc<TypedTensorBody<R, f64, CudaStorage>>| {
            let TypedData::Dense(storage) = body.data.as_ref() else {
                unreachable!("typed CUDA transfer never produces compact storage")
            };
            let data = storage.download(cuda)?;
            Ok::<_, Error>(Arc::new(TypedTensorBody::dense(body.space.clone(), data)))
        };

        let repr = match &self.repr {
            TypedTensorRepr::Owned(body) => TypedTensorRepr::Owned(download(body)?),
            TypedTensorRepr::Adjoint(view) => {
                TypedTensorRepr::Adjoint(Arc::new(TypedAdjointView {
                    parent: download(&view.parent)?,
                    logical_space: view.logical_space.clone(),
                    materialized: OnceLock::new(),
                    #[cfg(test)]
                    materialized_body_builds: std::sync::atomic::AtomicUsize::new(0),
                }))
            }
        };
        Ok(TensorMap {
            runtime: self.runtime.clone(),
            repr,
        })
    }
}

#[cfg(feature = "cuda")]
/// Checked Generic providers deliberately have no device execution methods in
/// this leaf:
///
/// ```compile_fail
/// use tenet::core::U1FusionRule;
/// use tenet::typed::{CudaStorage, TensorMap};
///
/// fn no_checked_generic_cuda_operations(
///     lhs: &TensorMap<U1FusionRule, f64, CudaStorage>,
///     rhs: &TensorMap<U1FusionRule, f64, CudaStorage>,
/// ) {
///     let _ = lhs.norm();
///     let _ = lhs.inner(rhs);
///     let _ = lhs.dot(rhs);
///     let _ = lhs.scale(2.0);
///     let _ = lhs.add(rhs, 2.0, -3.0);
///     let _ = lhs.zeros_like();
///     let _ = lhs.normalize();
///     let _ = lhs.qr_compact();
///     let _ = lhs.svd_compact();
/// }
/// ```
///
/// Complex CUDA reductions are likewise absent: CUDA storage currently owns
/// f64 payloads only.
///
/// ```compile_fail
/// use num_complex::Complex64;
/// use tenet::core::U1FusionRule;
/// use tenet::typed::{CudaStorage, TensorMap};
///
/// fn no_c64_cuda_operations(
///     lhs: &TensorMap<U1FusionRule, Complex64, CudaStorage>,
///     rhs: &TensorMap<U1FusionRule, Complex64, CudaStorage>,
/// ) {
///     let _ = lhs.norm();
///     let _ = lhs.inner(rhs);
///     let _ = lhs.dot(rhs);
///     let _ = lhs.scale(Complex64::new(2.0, 0.0));
///     let _ = lhs.add(
///         rhs,
///         Complex64::new(2.0, 0.0),
///         Complex64::new(-3.0, 0.0),
///     );
///     let _ = lhs.zeros_like();
///     let _ = lhs.normalize();
///     let _ = lhs.qr_compact();
///     let _ = lhs.svd_compact();
/// }
/// ```
///
/// Compact QR remains unavailable for checked-Generic and complex CUDA
/// tensors independently of the reduction/arithmetic surface above:
///
/// ```compile_fail
/// use tenet::core::U1FusionRule;
/// use tenet::typed::{CudaStorage, TensorMap};
///
/// fn no_checked_generic_cuda_qr(tensor: &TensorMap<U1FusionRule, f64, CudaStorage>) {
///     let _ = tensor.qr_compact();
/// }
/// ```
///
/// ```compile_fail
/// use num_complex::Complex64;
/// use tenet::core::U1FusionRule;
/// use tenet::typed::{CudaStorage, TensorMap};
///
/// fn no_complex_cuda_qr(tensor: &TensorMap<U1FusionRule, Complex64, CudaStorage>) {
///     let _ = tensor.qr_compact();
/// }
/// ```
///
/// Compact SVD has the same deliberately narrow typed CUDA surface:
///
/// ```compile_fail
/// use tenet::core::U1FusionRule;
/// use tenet::typed::{CudaStorage, TensorMap};
///
/// fn no_checked_generic_cuda_svd(tensor: &TensorMap<U1FusionRule, f64, CudaStorage>) {
///     let _ = tensor.svd_compact();
/// }
/// ```
///
/// ```compile_fail
/// use num_complex::Complex64;
/// use tenet::core::U1FusionRule;
/// use tenet::typed::{CudaStorage, TensorMap};
///
/// fn no_complex_cuda_svd(tensor: &TensorMap<U1FusionRule, Complex64, CudaStorage>) {
///     let _ = tensor.svd_compact();
/// }
/// ```
///
/// Truncated SVD is absent at the same checked-Generic/complex boundaries:
///
/// ```compile_fail
/// use tenet::core::U1FusionRule;
/// use tenet::typed::{CudaStorage, TensorMap, Truncation};
///
/// fn no_checked_generic_cuda_svd_trunc(tensor: &TensorMap<U1FusionRule, f64, CudaStorage>) {
///     let _ = tensor.svd_trunc(&Truncation::Full);
/// }
/// ```
///
/// ```compile_fail
/// use num_complex::Complex64;
/// use tenet::core::U1FusionRule;
/// use tenet::typed::{CudaStorage, TensorMap, Truncation};
///
/// fn no_complex_cuda_svd_trunc(
///     tensor: &TensorMap<U1FusionRule, Complex64, CudaStorage>,
/// ) {
///     let _ = tensor.svd_trunc(&Truncation::Full);
/// }
/// ```
impl<R> TensorMap<R, f64, CudaStorage>
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64> + CheckedFusionAlgebra + SectorCodec,
{
    /// Lazy categorical adjoint over the same parent device allocation.
    pub fn adjoint(&self) -> Result<Self, Error> {
        self.dense_adjoint_view()
    }

    fn direct_cuda_storage(&self, operation: &'static str) -> Result<&CudaStorage, Error> {
        match &self.repr {
            TypedTensorRepr::Owned(body) => match body.data.as_ref() {
                TypedData::Dense(storage) => Ok(storage),
                TypedData::Diagonal(_) => Err(Error::UnsupportedOnDevice(format!(
                    "{operation} requires dense CUDA storage"
                ))),
            },
            TypedTensorRepr::Adjoint(_) => Err(Error::UnsupportedOnDevice(format!(
                "{operation} does not support lazy adjoint CUDA operands"
            ))),
        }
    }

    fn validate_cuda_owned_metadata(
        expected: Placement,
        actual: Placement,
        required_len: usize,
        actual_len: usize,
    ) -> Result<(), Error> {
        if actual != expected {
            return Err(Error::PlacementMismatch);
        }
        if actual_len != required_len {
            return Err(internal_layout_error(
                "CUDA payload length does not match its admitted tensor space",
            ));
        }
        Ok(())
    }

    fn compile_cuda_qr_plan(
        &self,
        source_regions: Arc<[CoupledSectorRegion]>,
    ) -> Result<TypedCudaQrPlan<R>, Error> {
        let source_space = self.logical_space().space();
        let hom = source_space.homspace();
        let bond = SectorLeg::new(
            source_regions.iter().filter_map(|region| {
                let rank = region.rows().min(region.cols());
                (rank != 0).then_some((region.coupled(), rank))
            }),
            false,
        );
        let left_space =
            self.logical_space()
                .derive_from_final_homspace(FusionTreeHomSpace::new(
                    FusionProductSpace::new(hom.codomain().legs().iter().cloned()),
                    FusionProductSpace::new([bond.clone()]),
                ))?;
        let right_space =
            self.logical_space()
                .derive_from_final_homspace(FusionTreeHomSpace::new(
                    FusionProductSpace::new([bond]),
                    FusionProductSpace::new(hom.domain().legs().iter().cloned()),
                ))?;
        let left_regions =
            sector_regions(left_space.space().structure(), left_space.space().nout())?;
        let right_regions =
            sector_regions(right_space.space().structure(), right_space.space().nout())?;
        let index_by_sector = |regions: &[CoupledSectorRegion]| {
            let mut indices = HashMap::with_capacity(regions.len());
            for (index, region) in regions.iter().enumerate() {
                if indices.insert(region.coupled(), index).is_some() {
                    return Err(internal_layout_error(
                        "compact QR factor contains a duplicate coupled sector",
                    ));
                }
            }
            Ok(indices)
        };
        let left_by_sector = index_by_sector(&left_regions)?;
        let right_by_sector = index_by_sector(&right_regions)?;
        let mut routes = Vec::with_capacity(source_regions.len());
        for (source, region) in source_regions.iter().enumerate() {
            let rank = region.rows().min(region.cols());
            if rank == 0 {
                continue;
            }
            let left = *left_by_sector.get(&region.coupled()).ok_or_else(|| {
                internal_layout_error("compact QR left factor is missing a source sector")
            })?;
            let right = *right_by_sector.get(&region.coupled()).ok_or_else(|| {
                internal_layout_error("compact QR right factor is missing a source sector")
            })?;
            let left_region = &left_regions[left];
            let right_region = &right_regions[right];
            if (left_region.rows(), left_region.cols()) != (region.rows(), rank)
                || (right_region.rows(), right_region.cols()) != (rank, region.cols())
                || !cuda_qr_tree_extents_match(region.row_trees(), left_region.row_trees())?
                || !cuda_qr_tree_extents_match(region.col_trees(), right_region.col_trees())?
            {
                return Err(internal_layout_error(
                    "compact QR factor region does not match its source route",
                ));
            }
            routes.push(TypedCudaQrRoute {
                source,
                left,
                right,
                rank,
            });
        }
        if routes.len() != left_regions.len() || routes.len() != right_regions.len() {
            return Err(internal_layout_error(
                "compact QR factor contains an unrouted coupled sector",
            ));
        }
        Ok(TypedCudaQrPlan {
            left_space,
            right_space,
            source_regions,
            left_regions,
            right_regions,
            routes,
        })
    }

    fn compile_cuda_svd_trunc_plan(
        &self,
        source_regions: Arc<[CoupledSectorRegion]>,
        kept_spectra: &[tenet_matrixalgebra::SectorSpectrum<f64>],
    ) -> Result<TypedCudaSvdTruncPlan<R>, Error> {
        let source_space = self.logical_space().space();
        let hom = source_space.homspace();
        let bond = SectorLeg::new(
            kept_spectra
                .iter()
                .map(|entry| (entry.sector, entry.values.len())),
            false,
        );
        let left_space =
            self.logical_space()
                .derive_from_final_homspace(FusionTreeHomSpace::new(
                    FusionProductSpace::new(hom.codomain().legs().iter().cloned()),
                    FusionProductSpace::new([bond.clone()]),
                ))?;
        #[cfg(test)]
        {
            observe_cuda_svd_trunc_event("admission_left");
            if CUDA_SVD_TRUNC_FAILURE.with(|failure| failure.get()) == Some(("admission", 1)) {
                let invalid_bond = SectorLeg::new([(SectorId::new(usize::MAX), 1)], false);
                self.logical_space()
                    .derive_from_final_homspace(FusionTreeHomSpace::new(
                        FusionProductSpace::new(hom.codomain().legs().iter().cloned()),
                        FusionProductSpace::new([invalid_bond]),
                    ))?;
                return Err(internal_layout_error(
                    "provider accepted an invalid final-admission sector",
                ));
            }
        }
        let middle_space =
            self.logical_space()
                .derive_from_final_homspace(FusionTreeHomSpace::new(
                    FusionProductSpace::new([bond.clone()]),
                    FusionProductSpace::new([bond.clone()]),
                ))?;
        let right_space =
            self.logical_space()
                .derive_from_final_homspace(FusionTreeHomSpace::new(
                    FusionProductSpace::new([bond]),
                    FusionProductSpace::new(hom.domain().legs().iter().cloned()),
                ))?;
        let left_regions =
            sector_regions(left_space.space().structure(), left_space.space().nout())?;
        let middle_regions = sector_regions(
            middle_space.space().structure(),
            middle_space.space().nout(),
        )?;
        let right_regions =
            sector_regions(right_space.space().structure(), right_space.space().nout())?;
        let index_by_sector = |regions: &[CoupledSectorRegion]| {
            let mut indices = HashMap::with_capacity(regions.len());
            for (index, region) in regions.iter().enumerate() {
                if indices.insert(region.coupled(), index).is_some() {
                    return Err(internal_layout_error(
                        "compact SVD truncated factor contains a duplicate coupled sector",
                    ));
                }
            }
            Ok(indices)
        };
        let source_by_sector = index_by_sector(&source_regions)?;
        let left_by_sector = index_by_sector(&left_regions)?;
        let middle_by_sector = index_by_sector(&middle_regions)?;
        let right_by_sector = index_by_sector(&right_regions)?;
        let mut routes = Vec::with_capacity(kept_spectra.len());
        for spectrum in kept_spectra {
            let kept = spectrum.values.len();
            if kept == 0 {
                return Err(internal_layout_error(
                    "compact SVD truncated plan retained an empty sector",
                ));
            }
            let source = *source_by_sector.get(&spectrum.sector).ok_or_else(|| {
                internal_layout_error("compact SVD truncated factor is missing a source sector")
            })?;
            let source_region = &source_regions[source];
            let full_rank = source_region.rows().min(source_region.cols());
            if kept > full_rank {
                return Err(internal_layout_error(
                    "compact SVD truncated rank exceeds its source route",
                ));
            }
            let left = *left_by_sector.get(&spectrum.sector).ok_or_else(|| {
                internal_layout_error("compact SVD truncated left factor is missing a sector")
            })?;
            let middle = *middle_by_sector.get(&spectrum.sector).ok_or_else(|| {
                internal_layout_error("compact SVD truncated diagonal factor is missing a sector")
            })?;
            let right = *right_by_sector.get(&spectrum.sector).ok_or_else(|| {
                internal_layout_error("compact SVD truncated right factor is missing a sector")
            })?;
            let left_region = &left_regions[left];
            let middle_region = &middle_regions[middle];
            let right_region = &right_regions[right];
            let middle_len = middle_region
                .range()
                .end
                .checked_sub(middle_region.range().start)
                .ok_or_else(|| internal_layout_error("truncated SVD middle range is invalid"))?;
            if (left_region.rows(), left_region.cols()) != (source_region.rows(), kept)
                || (middle_region.rows(), middle_region.cols()) != (kept, kept)
                || middle_len
                    != kept.checked_mul(kept).ok_or_else(|| {
                        internal_layout_error("truncated SVD middle length overflows")
                    })?
                || (right_region.rows(), right_region.cols()) != (kept, source_region.cols())
                || !cuda_qr_tree_extents_match(source_region.row_trees(), left_region.row_trees())?
                || !cuda_qr_tree_extents_match(
                    middle_region.row_trees(),
                    middle_region.col_trees(),
                )?
                || !cuda_qr_tree_extents_match(source_region.col_trees(), right_region.col_trees())?
            {
                return Err(internal_layout_error(
                    "compact SVD truncated factor region does not match its source route",
                ));
            }
            routes.push(TypedCudaSvdTruncRoute {
                source,
                left,
                right,
                full_rank,
                kept,
            });
        }
        if routes.len() != left_regions.len()
            || routes.len() != middle_regions.len()
            || routes.len() != right_regions.len()
        {
            return Err(internal_layout_error(
                "compact SVD truncated factor contains an unrouted coupled sector",
            ));
        }
        Ok(TypedCudaSvdTruncPlan {
            left_space,
            middle_space,
            right_space,
            source_regions,
            left_regions,
            right_regions,
            routes,
        })
    }

    /// Streamed compact QR of owned dense CUDA storage. Each nonempty coupled
    /// sector is gauge-fixed on device; only its compact R diagonal is read
    /// back for the exact sign decision.
    pub fn qr_compact(&self) -> Result<(Self, Self), Error> {
        let source = self.direct_cuda_storage("qr_compact")?;
        let source_space = self.logical_space().space();
        let required_len = source_space.required_len()?;
        let source_regions = sector_regions(source_space.structure(), source_space.nout())?;

        {
            let mut state = self.runtime.lock();
            let cuda = state.cuda.as_mut().ok_or_else(|| {
                Error::InvalidArgument(
                    "this runtime was built without a CUDA device; use \
                     Runtime::builder().cuda(device)"
                        .to_string(),
                )
            })?;
            Self::validate_cuda_owned_metadata(
                Placement::Cuda(cuda.device()),
                source.placement(),
                required_len,
                source.len(),
            )?;
        }

        // Provider queries and final HomSpace admission belong outside the
        // execution lock; the plan owns every source-to-factor route.
        let plan = self.compile_cuda_qr_plan(source_regions)?;
        let left_len = plan.left_space.space().required_len()?;
        let right_len = plan.right_space.space().required_len()?;

        let (left_data, right_data) = {
            let mut state = self.runtime.lock();
            let cuda = state.cuda.as_mut().ok_or_else(|| {
                Error::InvalidArgument(
                    "this runtime was built without a CUDA device; use \
                     Runtime::builder().cuda(device)"
                        .to_string(),
                )
            })?;
            let mut left_data = CudaStorage::upload(cuda, &vec![0.0; left_len])?;
            #[cfg(test)]
            observe_cuda_qr_output_upload();
            let mut right_data = CudaStorage::upload(cuda, &vec![0.0; right_len])?;
            #[cfg(test)]
            observe_cuda_qr_output_upload();
            for route in &plan.routes {
                let source_region = &plan.source_regions[route.source];
                let left_region = &plan.left_regions[route.left];
                let right_region = &plan.right_regions[route.right];
                let (raw_left, raw_right, diagonal) = cuda_qr_region(
                    cuda,
                    &source.0,
                    source_region.range().start,
                    source_region.rows(),
                    source_region.cols(),
                )?;
                let signs = diagonal.iter().copied().map(cuda_qr_diagonal_sign);
                let selector = upload_selector(
                    cuda,
                    route.rank,
                    route.rank,
                    signs.enumerate().map(|(index, sign)| (index, index, sign)),
                )?;
                let scratch = TypedCudaQrScratch::new(raw_left, raw_right, selector);
                assemble_left_factor(
                    cuda,
                    &mut left_data,
                    left_region,
                    source_region,
                    &scratch.left,
                    route.rank,
                    &scratch.selector,
                    route.rank,
                )?;
                assemble_right_factor(
                    cuda,
                    &mut right_data,
                    right_region,
                    source_region,
                    &scratch.selector,
                    route.rank,
                    route.rank,
                    &scratch.right,
                )?;
            }
            (left_data, right_data)
        };

        Ok((
            Self {
                runtime: self.runtime.clone(),
                repr: owned_repr(TypedTensorBody::dense(plan.left_space, left_data)),
            },
            Self {
                runtime: self.runtime.clone(),
                repr: owned_repr(TypedTensorBody::dense(plan.right_space, right_data)),
            },
        ))
    }

    /// Streamed compact SVD of owned dense CUDA storage.
    ///
    /// Each nonempty coupled-sector route is decomposed and assembled before
    /// its raw device factors are dropped. Singular values are the only
    /// numerical tensor payload downloaded; the backend additionally reads
    /// O(1) solver-status metadata per route. The returned `s` is deliberately
    /// a dense CUDA tensor because
    /// CUDA diagonal storage is not part of the typed storage contract.
    /// `u` and `vh` retain the raw CUDA backend gauge; unlike the Host method,
    /// this method does not impose TensorKit's largest-pivot sign gauge.
    pub fn svd_compact(&self) -> Result<(Self, Self, Self), Error> {
        let source = self.direct_cuda_storage("svd_compact")?;
        let source_space = self.logical_space().space();
        let required_len = source_space.required_len()?;
        let source_regions = sector_regions(source_space.structure(), source_space.nout())?;

        {
            let mut state = self.runtime.lock();
            let cuda = state.cuda.as_mut().ok_or_else(|| {
                Error::InvalidArgument(
                    "this runtime was built without a CUDA device; use \
                     Runtime::builder().cuda(device)"
                        .to_string(),
                )
            })?;
            Self::validate_cuda_owned_metadata(
                Placement::Cuda(cuda.device()),
                source.placement(),
                required_len,
                source.len(),
            )?;
        }
        // As for typed CUDA QR, all provider work and final-space admission
        // complete before the execution lock and before any output exists.
        let plan = self.compile_cuda_qr_plan(source_regions)?;
        let bond = plan.left_space.space().homspace().domain().legs()[0].clone();
        let middle_space =
            self.logical_space()
                .derive_from_final_homspace(FusionTreeHomSpace::new(
                    FusionProductSpace::new([bond.clone()]),
                    FusionProductSpace::new([bond]),
                ))?;
        let middle_regions = sector_regions(
            middle_space.space().structure(),
            middle_space.space().nout(),
        )?;
        validate_cuda_svd_middle_regions(&plan, &middle_regions)?;
        let left_len = plan.left_space.space().required_len()?;
        let middle_len = middle_space.space().required_len()?;
        let right_len = plan.right_space.space().required_len()?;
        let (left_data, middle_data, right_data) = {
            let mut state = self.runtime.lock();
            let cuda = state.cuda.as_mut().ok_or_else(|| {
                Error::InvalidArgument(
                    "this runtime was built without a CUDA device; use \
                     Runtime::builder().cuda(device)"
                        .to_string(),
                )
            })?;
            let mut left_data = CudaStorage::upload(cuda, &vec![0.0; left_len])?;
            #[cfg(test)]
            observe_cuda_svd_final_storage_creation();
            let mut right_data = CudaStorage::upload(cuda, &vec![0.0; right_len])?;
            #[cfg(test)]
            observe_cuda_svd_final_storage_creation();
            let mut spectra = Vec::with_capacity(plan.routes.len());

            for route in &plan.routes {
                let source_region = &plan.source_regions[route.source];
                let left_region = &plan.left_regions[route.left];
                let right_region = &plan.right_regions[route.right];
                let (raw_left, values, raw_right) = cuda_svd_region(
                    cuda,
                    &source.0,
                    source_region.range().start,
                    source_region.rows(),
                    source_region.cols(),
                )?;
                if values.len() != route.rank {
                    return Err(internal_layout_error(
                        "compact SVD spectrum length does not match its source route",
                    ));
                }
                let selector = upload_selector(
                    cuda,
                    route.rank,
                    route.rank,
                    (0..route.rank).map(|index| (index, index, 1.0)),
                )?;
                // The scratch owns all route-local allocations. It is dropped
                // at the end of this iteration, bounding peak raw-factor
                // storage independently of the number of sectors.
                let scratch = TypedCudaSvdScratch::new(raw_left, raw_right, selector);
                assemble_left_factor(
                    cuda,
                    &mut left_data,
                    left_region,
                    source_region,
                    &scratch.left,
                    route.rank,
                    &scratch.selector,
                    route.rank,
                )?;
                assemble_right_factor(
                    cuda,
                    &mut right_data,
                    right_region,
                    source_region,
                    &scratch.selector,
                    route.rank,
                    route.rank,
                    &scratch.right,
                )?;
                spectra.push(tenet_matrixalgebra::SectorSpectrum {
                    sector: source_region.coupled(),
                    values,
                });
            }

            let mut middle_host = vec![0.0; middle_len];
            fill_diagonal_values(middle_space.space().structure(), &mut middle_host, &spectra)?;
            let middle_data = CudaStorage::upload(cuda, &middle_host)?;
            #[cfg(test)]
            observe_cuda_svd_final_storage_creation();
            (left_data, middle_data, right_data)
        };

        Ok((
            Self {
                runtime: self.runtime.clone(),
                repr: owned_repr(TypedTensorBody::dense(plan.left_space, left_data)),
            },
            Self {
                runtime: self.runtime.clone(),
                repr: owned_repr(TypedTensorBody::dense(middle_space, middle_data)),
            },
            Self {
                runtime: self.runtime.clone(),
                repr: owned_repr(TypedTensorBody::dense(plan.right_space, right_data)),
            },
        ))
    }

    fn upload_cuda_svd_trunc_final(
        cuda: &CudaDenseContext,
        values: &[f64],
    ) -> Result<CudaStorage, Error> {
        #[cfg(test)]
        {
            observe_cuda_svd_trunc_event("final_storage");
            let ordinal = CUDA_SVD_TRUNC_FINAL_EXTENTS.with(|extents| {
                let mut extents = extents.borrow_mut();
                let Some(extents) = extents.as_mut() else {
                    return 0;
                };
                extents.push(values.len());
                extents.len()
            });
            if CUDA_SVD_TRUNC_FAILURE.with(|failure| failure.get()) == Some(("final", ordinal)) {
                return Err(Error::InvalidArgument(
                    "injected truncated SVD final storage failure".to_string(),
                ));
            }
        }
        let storage = CudaStorage::upload(cuda, values)?;
        #[cfg(test)]
        {
            observe_cuda_svd_trunc_final_storage_creation();
            observe_cuda_svd_trunc_allocation("final", values.len());
        }
        Ok(storage)
    }

    #[cfg(test)]
    fn inject_cuda_svd_trunc_assembly_failure(ordinal: usize) -> Result<(), Error> {
        observe_cuda_svd_trunc_event("assembly");
        if CUDA_SVD_TRUNC_FAILURE.with(|failure| failure.get()) == Some(("assembly", ordinal)) {
            return Err(Error::InvalidArgument(
                "injected truncated SVD assembly failure".to_string(),
            ));
        }
        Ok(())
    }

    #[cfg(test)]
    fn inject_cuda_svd_trunc_failure(stage: &'static str, ordinal: usize) -> Result<(), Error> {
        if CUDA_SVD_TRUNC_FAILURE.with(|failure| failure.get()) == Some((stage, ordinal)) {
            return Err(Error::InvalidArgument(format!(
                "injected truncated SVD {stage} failure"
            )));
        }
        Ok(())
    }

    /// One-pass truncated SVD of owned dense CUDA storage.
    ///
    /// Raw per-sector U/Vh factors remain device-resident until the global,
    /// quantum-dimension-weighted truncation decision is known. The returned
    /// `s` is dense CUDA storage; U/Vh retain the raw CUDA backend gauge.
    pub fn svd_trunc(
        &self,
        truncation: &Truncation,
    ) -> Result<SvdTrunc<R, f64, CudaStorage>, Error> {
        let source = self.direct_cuda_storage("svd_trunc")?;
        let source_space = self.logical_space().space();
        let required_len = source_space.required_len()?;
        let source_regions = sector_regions(source_space.structure(), source_space.nout())?;

        {
            let mut state = self.runtime.lock();
            let cuda = state.cuda.as_mut().ok_or_else(|| {
                Error::InvalidArgument(
                    "this runtime was built without a CUDA device; use \
                     Runtime::builder().cuda(device)"
                        .to_string(),
                )
            })?;
            Self::validate_cuda_owned_metadata(
                Placement::Cuda(cuda.device()),
                source.placement(),
                required_len,
                source.len(),
            )?;
        }

        // The shared selector remains the sole truncation-policy authority.
        // An empty spectrum validates the policy and every TruncationSpace
        // identity without invoking dimension queries or CUDA work.
        #[cfg(test)]
        observe_cuda_svd_trunc_event("policy_identity");
        decide_kept(self.logical_space().provider(), &[], Some(truncation))?;

        // This performs the same complete source/factor route preflight as
        // compact SVD before the first decomposition. Its full-rank factor
        // spaces are not published; the kept spaces are admitted below.
        let source_plan = self.compile_cuda_qr_plan(source_regions)?;
        let (raw_spectra, mut retained) = {
            let mut state = self.runtime.lock();
            #[cfg(test)]
            let _lock_observation = CudaSvdTruncLockObservationGuard::new();
            let cuda = state.cuda.as_mut().ok_or_else(|| {
                Error::InvalidArgument(
                    "this runtime was built without a CUDA device; use \
                     Runtime::builder().cuda(device)"
                        .to_string(),
                )
            })?;
            let mut spectra = Vec::with_capacity(source_plan.source_regions.len());
            let mut retained = Vec::with_capacity(source_plan.source_regions.len());
            #[cfg(test)]
            let mut decomposition_ordinal = 0;
            for region in source_plan.source_regions.iter() {
                #[cfg(test)]
                {
                    decomposition_ordinal += 1;
                }
                let rank = region.rows().min(region.cols());
                if rank == 0 {
                    spectra.push(tenet_matrixalgebra::SectorSpectrum {
                        sector: region.coupled(),
                        values: Vec::new(),
                    });
                    retained.push(None);
                    continue;
                }
                #[cfg(test)]
                observe_cuda_svd_trunc_event("decomposition");
                let (raw_left, values, raw_right) = cuda_svd_region(
                    cuda,
                    &source.0,
                    region.range().start,
                    region.rows(),
                    region.cols(),
                )?;
                #[cfg(test)]
                observe_cuda_svd_trunc_decomposition(values.len());
                if values.len() != rank {
                    return Err(internal_layout_error(
                        "truncated SVD spectrum length does not match its source route",
                    ));
                }
                let factors = TypedCudaSvdRetainedFactors::new(
                    raw_left,
                    raw_right,
                    region.rows(),
                    region.cols(),
                    rank,
                )?;
                #[cfg(test)]
                Self::inject_cuda_svd_trunc_failure("decomposition", decomposition_ordinal)?;
                spectra.push(tenet_matrixalgebra::SectorSpectrum {
                    sector: region.coupled(),
                    values,
                });
                retained.push(Some(factors));
            }
            (spectra, retained)
        };

        // Provider dimension queries, global selection, label decoding, and
        // every kept-space admission are intentionally outside the CUDA lock.
        #[cfg(test)]
        {
            observe_cuda_svd_trunc_event("selection");
            Self::inject_cuda_svd_trunc_failure("selection", 1)?;
        }
        let (kept_spectra, error) = decide_kept(
            self.logical_space().provider(),
            &raw_spectra,
            Some(truncation),
        )?;
        let kept_sectors: HashSet<_> = kept_spectra.iter().map(|entry| entry.sector).collect();
        for (index, region) in source_plan.source_regions.iter().enumerate() {
            if !kept_sectors.contains(&region.coupled()) {
                retained[index] = None;
            }
        }
        #[cfg(test)]
        observe_cuda_svd_trunc_event("decode");
        let mut singular_values: Vec<SectorSpectrum<R::Sector>> = kept_spectra
            .iter()
            .map(|entry| {
                Ok(SectorSpectrum {
                    sector: self
                        .logical_space()
                        .provider()
                        .decode_sector(entry.sector)?,
                    values: entry.values.clone(),
                })
            })
            .collect::<Result<_, Error>>()?;
        singular_values.sort_by(|left, right| left.sector.cmp(&right.sector));
        #[cfg(test)]
        observe_cuda_svd_trunc_event("final_admission");
        let plan = self
            .compile_cuda_svd_trunc_plan(Arc::clone(&source_plan.source_regions), &kept_spectra)?;
        let left_len = plan.left_space.space().required_len()?;
        let middle_len = plan.middle_space.space().required_len()?;
        let right_len = plan.right_space.space().required_len()?;
        let mut middle_host = vec![0.0; middle_len];
        fill_diagonal_values(
            plan.middle_space.space().structure(),
            &mut middle_host,
            &kept_spectra,
        )?;

        let (left_data, middle_data, right_data) = {
            let mut state = self.runtime.lock();
            #[cfg(test)]
            let _lock_observation = CudaSvdTruncLockObservationGuard::new();
            let cuda = state.cuda.as_mut().ok_or_else(|| {
                Error::InvalidArgument(
                    "this runtime was built without a CUDA device; use \
                     Runtime::builder().cuda(device)"
                        .to_string(),
                )
            })?;
            let mut left_data = Self::upload_cuda_svd_trunc_final(cuda, &vec![0.0; left_len])?;
            let middle_data = Self::upload_cuda_svd_trunc_final(cuda, &middle_host)?;
            let mut right_data = Self::upload_cuda_svd_trunc_final(cuda, &vec![0.0; right_len])?;
            #[cfg(test)]
            let mut assembly_ordinal = 0;
            for route in plan.routes.iter() {
                #[cfg(test)]
                {
                    assembly_ordinal += 1;
                }
                let source_region = &plan.source_regions[route.source];
                let factors = retained[route.source].take().ok_or_else(|| {
                    internal_layout_error("kept truncated SVD route has no retained factors")
                })?;
                #[cfg(test)]
                Self::inject_cuda_svd_trunc_assembly_failure(assembly_ordinal)?;
                let left_selector = upload_selector(
                    cuda,
                    route.full_rank,
                    route.kept,
                    (0..route.kept).map(|index| (index, index, 1.0)),
                )?;
                assemble_left_factor(
                    cuda,
                    &mut left_data,
                    &plan.left_regions[route.left],
                    source_region,
                    &factors.left,
                    route.full_rank,
                    &left_selector,
                    route.kept,
                )?;
                #[cfg(test)]
                Self::inject_cuda_svd_trunc_failure("right_assembly", assembly_ordinal)?;
                let right_selector = upload_selector(
                    cuda,
                    route.kept,
                    route.full_rank,
                    (0..route.kept).map(|index| (index, index, 1.0)),
                )?;
                assemble_right_factor(
                    cuda,
                    &mut right_data,
                    &plan.right_regions[route.right],
                    source_region,
                    &right_selector,
                    route.kept,
                    route.full_rank,
                    &factors.right,
                )?;
                drop(factors);
                #[cfg(test)]
                observe_cuda_svd_trunc_release();
            }
            (left_data, middle_data, right_data)
        };

        #[cfg(test)]
        observe_cuda_svd_trunc_event("publication");

        Ok(SvdTrunc {
            u: Self {
                runtime: self.runtime.clone(),
                repr: owned_repr(TypedTensorBody::dense(plan.left_space, left_data)),
            },
            s: Self {
                runtime: self.runtime.clone(),
                repr: owned_repr(TypedTensorBody::dense(plan.middle_space, middle_data)),
            },
            vh: Self {
                runtime: self.runtime.clone(),
                repr: owned_repr(TypedTensorBody::dense(plan.right_space, right_data)),
            },
            singular_values,
            error,
        })
    }

    /// Hermitian eigendecomposition of an owned dense CUDA endomorphism.
    ///
    /// Returns `(d, v)` with `self = v * d * v.adjoint()`. Both factors remain
    /// on the source device. A lazy-adjoint receiver is rejected explicitly;
    /// no receiver-sized payload is downloaded or materialized.
    pub fn eigh_full(&self) -> Result<(Self, Self), Error> {
        let out = self.eigh_trunc(&Truncation::Full)?;
        Ok((out.d, out.v))
    }

    #[cfg(test)]
    fn inject_cuda_eigh_failure(stage: &'static str, ordinal: usize) -> Result<(), Error> {
        if CUDA_EIGH_FAILURE.with(|failure| failure.get()) == Some((stage, ordinal)) {
            return Err(Error::InvalidArgument(format!(
                "injected CUDA EIGH {stage} failure"
            )));
        }
        Ok(())
    }

    /// Truncated Hermitian eigendecomposition of an owned dense CUDA tensor.
    ///
    /// Hermiticity is checked sectorwise on device before the first cuSOLVER
    /// call. Only scalar residual metadata and eigenvalues cross to the host;
    /// eigenvectors and both returned factors remain device-resident.
    pub fn eigh_trunc(
        &self,
        truncation: &Truncation,
    ) -> Result<EighTrunc<R, f64, CudaStorage>, Error> {
        let source = self.direct_cuda_storage("eigh_trunc")?;
        let source_space = self.logical_space().space();
        if source_space.homspace().codomain() != source_space.homspace().domain() {
            return Err(
                tenet_tensors::OperationError::UnsupportedTensorContractScope {
                    message: "eigh requires an endomorphism (codomain == domain)",
                }
                .into(),
            );
        }
        let required_len = source_space.required_len()?;
        let source_regions = sector_regions(source_space.structure(), source_space.nout())?;
        if source_regions
            .iter()
            .any(|region| region.rows() != region.cols())
        {
            return Err(internal_layout_error(
                "CUDA EIGH source contains a non-square coupled-sector region",
            ));
        }

        {
            let mut state = self.runtime.lock();
            let cuda = state.cuda.as_mut().ok_or_else(|| {
                Error::InvalidArgument(
                    "this runtime was built without a CUDA device; use \
                     Runtime::builder().cuda(device)"
                        .to_string(),
                )
            })?;
            Self::validate_cuda_owned_metadata(
                Placement::Cuda(cuda.device()),
                source.placement(),
                required_len,
                source.len(),
            )?;
        }

        // Validate the policy and every provider identity before device work.
        decide_kept(self.logical_space().provider(), &[], Some(truncation))?;
        // The existing compact factor plan is the canonical source -> left
        // factor route. EIGH needs that left route only; no new plan hierarchy.
        let source_plan = self.compile_cuda_qr_plan(source_regions)?;

        // Admission is complete. Validate every block before the first EIGH so
        // a late non-Hermitian sector cannot trigger partial numerical work.
        {
            let mut state = self.runtime.lock();
            let cuda = state.cuda.as_mut().ok_or_else(|| {
                Error::InvalidArgument(
                    "this runtime was built without a CUDA device; use \
                     Runtime::builder().cuda(device)"
                        .to_string(),
                )
            })?;
            for region in source_plan.source_regions.iter() {
                if !cuda_is_hermitian_region(cuda, &source.0, region.range().start, region.rows())?
                {
                    return Err(
                        tenet_tensors::OperationError::UnsupportedTensorContractScope {
                            message: "eigh requires every coupled-sector block to be Hermitian",
                        }
                        .into(),
                    );
                }
            }
        }

        let (raw_spectra, mut raw_vectors, orders) = {
            let mut state = self.runtime.lock();
            let cuda = state.cuda.as_mut().ok_or_else(|| {
                Error::InvalidArgument(
                    "this runtime was built without a CUDA device; use \
                     Runtime::builder().cuda(device)"
                        .to_string(),
                )
            })?;
            let mut spectra = Vec::with_capacity(source_plan.source_regions.len());
            let mut vectors = Vec::with_capacity(source_plan.source_regions.len());
            let mut orders = Vec::with_capacity(source_plan.source_regions.len());
            #[cfg(test)]
            let mut decomposition_ordinal = 0;
            for region in source_plan.source_regions.iter() {
                let n = region.rows();
                if n == 0 {
                    spectra.push(tenet_matrixalgebra::SectorSpectrum {
                        sector: region.coupled(),
                        values: Vec::new(),
                    });
                    vectors.push(None);
                    orders.push(Vec::new());
                    continue;
                }
                let (values, vector) =
                    typed_cuda_eigh_region(cuda, &source.0, region.range().start, n)?;
                #[cfg(test)]
                {
                    decomposition_ordinal += 1;
                    Self::inject_cuda_eigh_failure("decomposition", decomposition_ordinal)?;
                }
                if values.iter().any(|value| !value.is_finite()) {
                    return Err(internal_layout_error(
                        "CUDA EIGH returned a non-finite eigenvalue",
                    ));
                }
                let mut order: Vec<_> = (0..n).collect();
                order.sort_by(|&left, &right| {
                    values[right]
                        .abs()
                        .total_cmp(&values[left].abs())
                        .then(left.cmp(&right))
                });
                let sorted = order.iter().map(|&index| values[index]).collect();
                spectra.push(tenet_matrixalgebra::SectorSpectrum {
                    sector: region.coupled(),
                    values: sorted,
                });
                vectors.push(Some(vector));
                orders.push(order);
            }
            (spectra, vectors, orders)
        };

        let (kept_spectra, error) = decide_kept(
            self.logical_space().provider(),
            &raw_spectra,
            Some(truncation),
        )?;
        let mut eigenvalues: Vec<SectorSpectrum<R::Sector>> = kept_spectra
            .iter()
            .map(|entry| {
                Ok(SectorSpectrum {
                    sector: self
                        .logical_space()
                        .provider()
                        .decode_sector(entry.sector)?,
                    values: entry.values.clone(),
                })
            })
            .collect::<Result<_, Error>>()?;
        eigenvalues.sort_by(|left, right| left.sector.cmp(&right.sector));

        // Reuse the existing kept-rank space/route proof. Its right-factor
        // fields are intentionally unused; measured need, not EIGH alone,
        // would justify splitting another private plan type.
        let plan = self
            .compile_cuda_svd_trunc_plan(Arc::clone(&source_plan.source_regions), &kept_spectra)?;
        let vector_len = plan.left_space.space().required_len()?;
        let diagonal_len = plan.middle_space.space().required_len()?;
        let mut diagonal_host = vec![0.0; diagonal_len];
        fill_diagonal_values(
            plan.middle_space.space().structure(),
            &mut diagonal_host,
            &kept_spectra,
        )?;

        let (diagonal_data, vector_data) = {
            let mut state = self.runtime.lock();
            let cuda = state.cuda.as_mut().ok_or_else(|| {
                Error::InvalidArgument(
                    "this runtime was built without a CUDA device; use \
                     Runtime::builder().cuda(device)"
                        .to_string(),
                )
            })?;
            let diagonal_data = CudaStorage::upload(cuda, &diagonal_host)?;
            let mut vector_data = CudaStorage::upload(cuda, &vec![0.0; vector_len])?;
            #[cfg(test)]
            let mut assembly_ordinal = 0;
            for route in plan.routes.iter() {
                #[cfg(test)]
                {
                    assembly_ordinal += 1;
                    Self::inject_cuda_eigh_failure("assembly", assembly_ordinal)?;
                }
                let source_region = &plan.source_regions[route.source];
                let raw = raw_vectors[route.source].take().ok_or_else(|| {
                    internal_layout_error("kept CUDA EIGH route has no eigenvectors")
                })?;
                let order = &orders[route.source];
                if route.kept > order.len() {
                    return Err(internal_layout_error(
                        "kept CUDA EIGH rank exceeds its eigenvector order",
                    ));
                }
                let selector = upload_selector(
                    cuda,
                    route.full_rank,
                    route.kept,
                    order[..route.kept]
                        .iter()
                        .enumerate()
                        .map(|(column, &row)| (row, column, 1.0)),
                )?;
                assemble_left_factor(
                    cuda,
                    &mut vector_data,
                    &plan.left_regions[route.left],
                    source_region,
                    &raw,
                    route.full_rank,
                    &selector,
                    route.kept,
                )?;
            }
            (diagonal_data, vector_data)
        };

        Ok(EighTrunc {
            d: Self {
                runtime: self.runtime.clone(),
                repr: owned_repr(TypedTensorBody::dense(plan.middle_space, diagonal_data)),
            },
            v: Self {
                runtime: self.runtime.clone(),
                repr: owned_repr(TypedTensorBody::dense(plan.left_space, vector_data)),
            },
            eigenvalues,
            error,
        })
    }

    fn cuda_axpby_owned(
        &self,
        required_len: usize,
        lhs: (&CudaStorage, f64),
        rhs: Option<(&CudaStorage, f64)>,
    ) -> Result<CudaStorage, Error> {
        let (lhs, alpha) = lhs;
        let mut state = self.runtime.lock();
        let cuda = state.cuda.as_mut().ok_or_else(|| {
            Error::InvalidArgument(
                "this runtime was built without a CUDA device; use \
                 Runtime::builder().cuda(device)"
                    .to_string(),
            )
        })?;
        let expected = Placement::Cuda(cuda.device());
        Self::validate_cuda_owned_metadata(expected, lhs.placement(), required_len, lhs.len())?;
        if let Some((rhs, _)) = rhs {
            Self::validate_cuda_owned_metadata(expected, rhs.placement(), required_len, rhs.len())?;
        }

        // ponytail: #740 keeps these proven Host uploads until native device
        // allocation publishes cross-stream writes correctly and wins a bench.
        let coefficient_values = match rhs {
            Some((_, beta)) => vec![alpha, beta],
            None => vec![alpha],
        };
        // Keep coefficients as data operands: descriptor alpha == 0 permits
        // CUDA to skip source reads and erase NaN/Inf propagation. Arithmetic
        // does not promise signed-zero bit parity across storage backends.
        let coefficients = CudaStorage::upload(cuda, &coefficient_values)?;
        #[cfg(test)]
        observe_cuda_arithmetic(0, 1, 0);
        let mut output = CudaStorage::upload(cuda, &vec![0.0; required_len])?;
        #[cfg(test)]
        observe_cuda_arithmetic(1, 0, 0);
        if required_len != 0 {
            cuda_gemm_region_into(
                cuda,
                &mut output.0,
                0,
                required_len,
                &lhs.0,
                0,
                required_len,
                &coefficients.0,
                0,
                1,
                required_len,
                1,
                1,
                1.0,
                0.0,
            )
            .map_err(|err| Error::from(tenet_tensors::OperationError::Dense(err)))?;
            #[cfg(test)]
            observe_cuda_arithmetic(0, 0, 1);
            if let Some((rhs, _)) = rhs {
                cuda_gemm_region_into(
                    cuda,
                    &mut output.0,
                    0,
                    required_len,
                    &rhs.0,
                    0,
                    required_len,
                    &coefficients.0,
                    1,
                    1,
                    required_len,
                    1,
                    1,
                    1.0,
                    1.0,
                )
                .map_err(|err| Error::from(tenet_tensors::OperationError::Dense(err)))?;
                #[cfg(test)]
                observe_cuda_arithmetic(0, 0, 1);
            }
        }
        Ok(output)
    }

    fn cuda_zeros_owned(
        &self,
        required_len: usize,
        source: &CudaStorage,
    ) -> Result<CudaStorage, Error> {
        let mut state = self.runtime.lock();
        let cuda = state.cuda.as_mut().ok_or_else(|| {
            Error::InvalidArgument(
                "this runtime was built without a CUDA device; use \
                 Runtime::builder().cuda(device)"
                    .to_string(),
            )
        })?;
        Self::validate_cuda_owned_metadata(
            Placement::Cuda(cuda.device()),
            source.placement(),
            required_len,
            source.len(),
        )?;
        let output = CudaStorage::upload(cuda, &vec![0.0; required_len])?;
        #[cfg(test)]
        observe_cuda_arithmetic(1, 0, 0);
        Ok(output)
    }

    fn preflight_owned_cuda_arithmetic(
        &self,
        required_len: usize,
        storage: &CudaStorage,
    ) -> Result<(), Error> {
        let mut state = self.runtime.lock();
        let cuda = state.cuda.as_mut().ok_or_else(|| {
            Error::InvalidArgument(
                "this runtime was built without a CUDA device; use \
                 Runtime::builder().cuda(device)"
                    .to_string(),
            )
        })?;
        Self::validate_cuda_owned_metadata(
            Placement::Cuda(cuda.device()),
            storage.placement(),
            required_len,
            storage.len(),
        )
    }

    fn with_owned_cuda_storage(&self, storage: CudaStorage) -> Self {
        let body = self
            .owned_body()
            .expect("CUDA arithmetic output authority must be owned");
        Self {
            runtime: self.runtime.clone(),
            repr: owned_repr(TypedTensorBody::dense(body.space.clone(), storage)),
        }
    }

    /// Fresh device result `factor * self` for owned storage; a lazy adjoint
    /// redirects algebraically through its canonical parent. Zero factors
    /// preserve nonfinite propagation, but signed-zero bits are backend-local.
    pub fn scale(&self, factor: f64) -> Result<Self, Error> {
        let required_len = self.logical_space().space().required_len()?;
        if let TypedTensorRepr::Adjoint(view) = &self.repr {
            let parent = Self {
                runtime: self.runtime.clone(),
                repr: TypedTensorRepr::Owned(Arc::clone(&view.parent)),
            };
            return parent.scale(factor)?.adjoint();
        }
        let source = self.direct_cuda_storage("scale")?;
        let output = self.cuda_axpby_owned(required_len, (source, factor), None)?;
        Ok(self.with_owned_cuda_storage(output))
    }

    /// Fresh device result `alpha * self + beta * other`. Zero coefficients
    /// preserve nonfinite propagation, but signed-zero bits are backend-local.
    pub fn add(&self, other: &Self, alpha: f64, beta: f64) -> Result<Self, Error> {
        let required_len = self.logical_space().space().required_len()?;
        if !self.runtime.same_runtime(&other.runtime) {
            return Err(Error::RuntimeMismatch);
        }
        if self.logical_space().space() != other.logical_space().space() {
            return Err(Error::InvalidArgument(
                "tensors live on different spaces or block layouts".to_string(),
            ));
        }
        match (&self.repr, &other.repr) {
            (TypedTensorRepr::Adjoint(lhs), TypedTensorRepr::Adjoint(rhs)) => {
                let lhs = Self {
                    runtime: self.runtime.clone(),
                    repr: TypedTensorRepr::Owned(Arc::clone(&lhs.parent)),
                };
                let rhs = Self {
                    runtime: other.runtime.clone(),
                    repr: TypedTensorRepr::Owned(Arc::clone(&rhs.parent)),
                };
                return lhs.add(&rhs, alpha, beta)?.adjoint();
            }
            (TypedTensorRepr::Adjoint(_), TypedTensorRepr::Owned(_))
            | (TypedTensorRepr::Owned(_), TypedTensorRepr::Adjoint(_)) => {
                return Err(Error::UnsupportedOnDevice(
                    "add does not support mixed owned/lazy CUDA operands".to_string(),
                ));
            }
            (TypedTensorRepr::Owned(_), TypedTensorRepr::Owned(_)) => {}
        }
        let lhs = self.direct_cuda_storage("add")?;
        let rhs = other.direct_cuda_storage("add")?;
        let output = self.cuda_axpby_owned(required_len, (lhs, alpha), Some((rhs, beta)))?;
        Ok(self.with_owned_cuda_storage(output))
    }

    /// Exact positive-zero device tensor, independent of source values.
    pub fn zeros_like(&self) -> Result<Self, Error> {
        let required_len = self.logical_space().space().required_len()?;
        if let TypedTensorRepr::Adjoint(view) = &self.repr {
            let parent = Self {
                runtime: self.runtime.clone(),
                repr: TypedTensorRepr::Owned(Arc::clone(&view.parent)),
            };
            return parent.zeros_like()?.adjoint();
        }
        let source = self.direct_cuda_storage("zeros_like")?;
        let output = self.cuda_zeros_owned(required_len, source)?;
        Ok(self.with_owned_cuda_storage(output))
    }

    /// Dimension-weighted unit normalization. Zero norm deliberately follows
    /// Host IEEE behavior and produces non-finite stored entries.
    pub fn normalize(&self) -> Result<Self, Error> {
        let required_len = self.logical_space().space().required_len()?;
        if let TypedTensorRepr::Adjoint(view) = &self.repr {
            let parent = Self {
                runtime: self.runtime.clone(),
                repr: TypedTensorRepr::Owned(Arc::clone(&view.parent)),
            };
            return parent.normalize()?.adjoint();
        }
        let storage = self.direct_cuda_storage("normalize")?;
        self.preflight_owned_cuda_arithmetic(required_len, storage)?;
        self.scale(1.0 / self.norm()?)
    }

    /// Quantum-dimension-weighted Frobenius reduction over owned CUDA storage.
    ///
    /// The device returns one scalar per coupled sector. Category weights stay
    /// with the tensor and are applied only after releasing the Runtime lock.
    fn weighted_inner_cuda(&self, lhs: &CudaStorage, rhs: &CudaStorage) -> Result<f64, Error> {
        let space = self.logical_space().space();
        let regions = sector_regions(space.structure(), space.nout())?;
        let mut state = self.runtime.lock();
        let cuda = state.cuda.as_mut().ok_or_else(|| {
            Error::InvalidArgument(
                "this runtime was built without a CUDA device; use \
                 Runtime::builder().cuda(device)"
                    .to_string(),
            )
        })?;
        validate_cuda_reduction_placement(
            Placement::Cuda(cuda.device()),
            lhs.placement(),
            rhs.placement(),
        )?;
        // ponytail: #740 keeps the proven host-zero upload until a native
        // allocation has correct cross-stream publication and measured value.
        let mut partials = CudaStorage::upload(cuda, &vec![0.0; regions.len().max(1)])?;
        {
            let mut gemm = CudaStorageGemm::new(cuda);
            for (index, region) in regions.iter().enumerate() {
                let len = region.rows() * region.cols();
                if len == 0 {
                    continue;
                }
                gemm.matmul_range_into(
                    &mut partials,
                    index,
                    lhs,
                    region.range().start,
                    rhs,
                    region.range().start,
                    1,
                    len,
                    1,
                )?;
            }
        }
        let values = download_cuda_reduction_partials(&partials, cuda)?;
        drop(state);

        Ok(regions
            .iter()
            .zip(values)
            .map(|(region, value)| {
                value * self.logical_space().provider().dim_scalar(region.coupled())
            })
            .sum())
    }

    /// Quantum-dimension-weighted Frobenius norm of a device tensor.
    ///
    /// A lazy adjoint delegates to its canonical parent because this norm is
    /// adjoint invariant; no logical-adjoint payload is materialized.
    pub fn norm(&self) -> Result<f64, Error> {
        if let TypedTensorRepr::Adjoint(view) = &self.repr {
            return Self {
                runtime: self.runtime.clone(),
                repr: TypedTensorRepr::Owned(Arc::clone(&view.parent)),
            }
            .norm();
        }
        let storage = self.direct_cuda_storage("norm")?;
        Ok(self.weighted_inner_cuda(storage, storage)?.sqrt())
    }

    /// Quantum-dimension-weighted Frobenius inner product of owned f64 device
    /// tensors. Lazy adjoints remain an explicit unsupported device scope.
    pub fn inner(&self, other: &Self) -> Result<f64, Error> {
        if !self.runtime.same_runtime(&other.runtime) {
            return Err(Error::RuntimeMismatch);
        }
        if self.logical_space().space() != other.logical_space().space() {
            return Err(Error::InvalidArgument(
                "tensors live on different spaces or block layouts".to_string(),
            ));
        }
        let lhs = self.direct_cuda_storage("inner")?;
        let rhs = other.direct_cuda_storage("inner")?;
        self.weighted_inner_cuda(lhs, rhs)
    }

    /// Total alias of [`Self::inner`].
    #[inline]
    pub fn dot(&self, other: &Self) -> Result<f64, Error> {
        self.inner(other)
    }

    fn cuda_fusion_operand(
        &self,
        operation: &'static str,
    ) -> Result<
        (
            &BoundDynamicFusionMapSpace<R>,
            tenet_tensors::FusionOperand<'_>,
            &CudaStorage,
        ),
        Error,
    > {
        match &self.repr {
            TypedTensorRepr::Owned(body) => match body.data.as_ref() {
                TypedData::Dense(storage) => Ok((
                    &body.space,
                    tenet_tensors::FusionOperand::direct(body.space.space()),
                    storage,
                )),
                TypedData::Diagonal(_) => Err(Error::UnsupportedOnDevice(format!(
                    "{operation} requires dense CUDA storage"
                ))),
            },
            TypedTensorRepr::Adjoint(view) => match view.parent.data.as_ref() {
                TypedData::Dense(storage) => Ok((
                    &view.logical_space,
                    tenet_tensors::FusionOperand::adjoint(view.parent.space.space()),
                    storage,
                )),
                TypedData::Diagonal(_) => Err(Error::UnsupportedOnDevice(format!(
                    "{operation} requires dense CUDA storage"
                ))),
            },
        }
    }

    fn unsupported_direct_contract() -> Error {
        tenet_tensors::OperationError::UnsupportedTensorContractScope {
            message: "typed CUDA contraction supports only whole-domain/whole-codomain axes in \
                      canonical order with identity output order",
        }
        .into()
    }

    /// Contracts owned or lazy-adjoint device tensors through the canonical fully-direct
    /// coupled-block route. Other layouts are explicit unsupported errors;
    /// device data is never downloaded or materialized on host.
    pub fn contract(
        &self,
        other: &Self,
        lhs_axes: &[usize],
        rhs_axes: &[usize],
        output_axes: &[usize],
    ) -> Result<Self, Error> {
        if !self.runtime.same_runtime(&other.runtime) {
            return Err(Error::RuntimeMismatch);
        }
        let (lhs_space, lhs_operand, lhs_storage) = self.cuda_fusion_operand("contract")?;
        let (rhs_space, rhs_operand, rhs_storage) = other.cuda_fusion_operand("contract")?;
        if !lhs_axes
            .iter()
            .copied()
            .eq(self.codomain_rank()..self.rank())
            || !rhs_axes.iter().copied().eq(0..other.codomain_rank())
        {
            return Err(Self::unsupported_direct_contract());
        }
        let output_rank = self.codomain_rank() + other.domain_rank();
        if !output_axes.iter().copied().eq(0..output_rank) {
            return Err(Self::unsupported_direct_contract());
        }
        let dst_space = BoundDynamicFusionMapSpace::contracted_multiplicity_free_ordered(
            lhs_space,
            rhs_space,
            lhs_axes,
            rhs_axes,
            OutputAxisOrder::identity(),
        )?;
        let mut state = self.runtime.lock();
        let crate::runtime::RuntimeState { mf, cuda, .. } = &mut *state;
        let cuda = cuda.as_mut().ok_or_else(|| {
            Error::InvalidArgument(
                "this runtime was built without a CUDA device; use \
                 Runtime::builder().cuda(device)"
                    .to_string(),
            )
        })?;
        let expected_placement = Placement::Cuda(cuda.device());
        if lhs_storage.placement() != expected_placement
            || rhs_storage.placement() != expected_placement
        {
            return Err(Error::PlacementMismatch);
        }
        // ponytail: the existing device seam initializes by uploading zeros;
        // replace this only with a measured native allocation/memset leaf.
        let mut dst = CudaStorage::upload(cuda, &vec![0.0; dst_space.space().required_len()?])?;
        mf.f64
            .tensorcontract_fusion_dyn_prelowered_direct_on_storage(
                &mut CudaStorageGemm::new(cuda),
                &dst_space,
                &mut dst,
                lhs_operand,
                lhs_storage,
                rhs_operand,
                rhs_storage,
                tenet_tensors::TensorContractSpec::new_with_conjugation(
                    lhs_axes,
                    rhs_axes,
                    OutputAxisOrder::identity(),
                    lhs_operand.storage_conjugate(),
                    rhs_operand.storage_conjugate(),
                ),
            )?;
        drop(state);
        Ok(Self {
            runtime: self.runtime.clone(),
            repr: owned_repr(TypedTensorBody::dense(dst_space, dst)),
        })
    }

    /// Total alias of [`Self::contract`] with the same device capability and
    /// error behavior.
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

    /// Tensor-map composition on owned or lazy-adjoint device tensors. This uses the
    /// twist-free composition compiler and therefore remains distinct from
    /// [`Self::contract`] for fermionic providers.
    #[doc(alias = "mul")]
    pub fn compose(&self, other: &Self) -> Result<Self, Error> {
        if !self.runtime.same_runtime(&other.runtime) {
            return Err(Error::RuntimeMismatch);
        }
        let (lhs_space, lhs_operand, lhs_storage) = self.cuda_fusion_operand("compose")?;
        let (rhs_space, rhs_operand, rhs_storage) = other.cuda_fusion_operand("compose")?;
        let lhs_axes: Vec<_> = (self.codomain_rank()..self.rank()).collect();
        let rhs_axes: Vec<_> = (0..other.codomain_rank()).collect();
        let dst_space = BoundDynamicFusionMapSpace::contracted_multiplicity_free(
            lhs_space, rhs_space, &lhs_axes, &rhs_axes,
        )?;
        let mut state = self.runtime.lock();
        let crate::runtime::RuntimeState { mf, cuda, .. } = &mut *state;
        let cuda = cuda.as_mut().ok_or_else(|| {
            Error::InvalidArgument(
                "this runtime was built without a CUDA device; use \
                 Runtime::builder().cuda(device)"
                    .to_string(),
            )
        })?;
        let expected_placement = Placement::Cuda(cuda.device());
        if lhs_storage.placement() != expected_placement
            || rhs_storage.placement() != expected_placement
        {
            return Err(Error::PlacementMismatch);
        }
        let mut dst = CudaStorage::upload(cuda, &vec![0.0; dst_space.space().required_len()?])?;
        mf.f64
            .tensorcompose_fusion_dyn_prelowered_direct_on_storage(
                &mut CudaStorageGemm::new(cuda),
                &dst_space,
                &mut dst,
                lhs_operand,
                lhs_storage,
                rhs_operand,
                rhs_storage,
                &lhs_axes,
                &rhs_axes,
            )?;
        drop(state);
        Ok(Self {
            runtime: self.runtime.clone(),
            repr: owned_repr(TypedTensorBody::dense(dst_space, dst)),
        })
    }
}

impl<R, D> TensorMap<R, D>
where
    D: TensorScalar,
{
    fn fusion_operand_and_data(&self) -> (tenet_tensors::FusionOperand<'_>, &[D]) {
        match &self.repr {
            TypedTensorRepr::Owned(body) => (
                tenet_tensors::FusionOperand::direct(body.space.space()),
                body.materialized_dense_data(),
            ),
            TypedTensorRepr::Adjoint(view) => (
                tenet_tensors::FusionOperand::adjoint(view.parent.space.space()),
                view.parent.materialized_dense_data(),
            ),
        }
    }

    fn cat_operand(&self) -> Result<(CatOperandLayout<'_>, &[D]), Error> {
        match &self.repr {
            TypedTensorRepr::Owned(body) => Ok((
                CatOperandLayout::owned(
                    body.space.space().structure(),
                    body.space.space().nout(),
                    body.space.space().nin(),
                )?,
                body.materialized_dense_data(),
            )),
            TypedTensorRepr::Adjoint(view) => Ok((
                CatOperandLayout::adjoint(
                    view.parent.space.space().structure(),
                    view.parent.space.space().nout(),
                    view.parent.space.space().nin(),
                )?,
                view.parent.materialized_dense_data(),
            )),
        }
    }

    /// Builds an operation-local logical tensor without publishing the
    /// receiver's reusable materialization cache, but still constructs a full
    /// receiver-sized logical payload. Prefer an oriented kernel or algebraic
    /// redirect when one implements the same semantics.
    fn materialized_tensor_uncached(&self) -> Result<Self, Error> {
        let TypedTensorRepr::Adjoint(view) = &self.repr else {
            return Ok(self.clone());
        };
        let data = tenet_tensors::materialize_adjoint_data_dyn(
            view.parent.space.space(),
            view.logical_space.space(),
            view.parent.materialized_dense_data(),
        )?;
        Ok(Self {
            runtime: self.runtime.clone(),
            repr: owned_repr(TypedTensorBody::dense(view.logical_space.clone(), data)),
        })
    }
}

impl<R, D, S> TensorMap<R, D, S>
where
    D: TensorScalar,
    S: HostReadableStorage<D>,
{
    /// Whole dense payload in the tensor's logical coupled-layout order.
    ///
    /// A lazy adjoint is materialized into host storage at most once across
    /// all clones. The canonical parent payload remains in `S`.
    #[inline]
    pub fn data(&self) -> &[D] {
        match &self.repr {
            TypedTensorRepr::Owned(body) => body.materialized_dense_data(),
            TypedTensorRepr::Adjoint(view) => view
                .materialized
                .get_or_init(|| {
                    let data = tenet_tensors::materialize_adjoint_data_dyn(
                        view.parent.space.space(),
                        view.logical_space.space(),
                        view.parent.materialized_dense_data(),
                    )
                    .expect("a pre-admitted typed adjoint must materialize");
                    #[cfg(test)]
                    view.materialized_body_builds
                        .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    Arc::new(TypedTensorBody::dense(view.logical_space.clone(), data))
                })
                .materialized_dense_data(),
        }
    }
}

impl<R, D, S> TypedTensorBody<R, D, S>
where
    D: TensorScalar,
    S: HostReadableStorage<D>,
{
    fn materialized_dense_data(&self) -> &[D] {
        match &*self.data {
            TypedData::Dense(data) => data.as_slice(),
            TypedData::Diagonal(spectrum) => self.dense_cache.get_or_init(|| {
                tenet_matrixalgebra::diagonal_bond_data(self.space.space(), spectrum, &|value| {
                    value
                })
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
            repr: owned_repr(TypedTensorBody::diagonal(space, spectrum)),
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
                            sector: self
                                .logical_space()
                                .provider()
                                .decode_sector(entry.sector)?,
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
        let materialized = self.materialized_tensor_uncached()?;
        let data = materialized
            .owned_body()
            .expect("uncached materialization is owned")
            .materialized_dense_data();
        let mut norm = 0.0_f64;
        let mut offdiag = 0.0_f64;
        for index in 0..self.logical_space().space().structure().block_count() {
            let block = self.logical_space().space().structure().block(index)?;
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
            repr: owned_repr(TypedTensorBody::dense(space, data)),
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
            .logical_space()
            .space()
            .structure()
            .block(index)
            .map_err(Error::from)
            .map_err(TypedFacadeError::<R>::from)?;
        decode_block_fusion_trees(self.logical_space().provider(), block.key())
    }
}

impl<R, D> TensorMap<R, D>
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64> + CheckedFusionAlgebra + SectorCodec,
    D: TensorScalar,
{
    /// The fused external sector content of one side of a structural
    /// constructor (TensorKit `fuse`, `spaces/gradedspace.jl:150-158`), using
    /// the shared provider-generic fold. Stored sector content is already
    /// external, so duality is dropped.
    ///
    /// The typed facade's `R` bound is `MultiplicityFreeRigidSymbols`, so a
    /// Generic-fusion provider cannot reach
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
            fused = fuse_sector_content(provider, &fused, &pairs)?;
        }
        Ok(fused)
    }

    /// Shared body of the structural constructors: checks the fused fit,
    /// builds zeros and writes the (partial) identity into every
    /// coupled-sector matrix — the same route as [`Self::id`].
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
    /// TensorKit's `one`/`id` for the same object. The payload dtype is `D`, so
    /// no runtime dtype token is needed; one leg list is used for both sides.
    ///
    /// Square by construction: the codomain *is* the domain, so the
    /// isomorphism precondition holds trivially and is not re-checked. The
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
        let TypedTensorRepr::Owned(body) = &mut tensor.repr else {
            unreachable!("a freshly built tensor is owned")
        };
        let body = Arc::get_mut(body).expect("a freshly built body has no other owner");
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

fn tree_operation_matches_axes(
    operation: &TreeTransformOperation,
    kind: TreeTransformOperationKind,
    codomain_axes: &[usize],
    domain_axes: &[usize],
) -> bool {
    operation.kind() == kind
        && operation.codomain_permutation() == codomain_axes
        && operation.domain_permutation() == domain_axes
}

impl<R, D> TensorMap<R, D>
where
    R: TypedSectorAdmission<Error = FusionAlgebraError, Mode = MultiplicityFreeAdmissionMode>
        + MultiplicityFreeRigidSymbols<Scalar = f64>
        + CheckedFusionAlgebra
        + SectorCodec,
    D: TensorScalar,
{
    /// Overwrites `destination` with `alpha * self.permute(...)` without
    /// replacing its provider, space, body, or dense host allocation.
    ///
    /// # Errors
    ///
    /// Validation and plan-construction failures leave `destination`
    /// unchanged. A backend error after replay begins may leave it partially
    /// overwritten.
    pub fn permute_overwrite_into(
        &self,
        destination: &mut Self,
        codomain_axes: &[usize],
        domain_axes: &[usize],
        alpha: D,
    ) -> Result<(), Error> {
        self.overwrite_tree_transform(
            destination,
            alpha,
            |_, _| {
                Ok(TreeTransformOperation::permute(
                    codomain_axes.iter().copied(),
                    domain_axes.iter().copied(),
                ))
            },
            |operation| {
                tree_operation_matches_axes(
                    operation,
                    TreeTransformOperationKind::Permute,
                    codomain_axes,
                    domain_axes,
                )
            },
        )
    }

    /// Overwrites `destination` with `alpha * self.transpose()` while
    /// preserving the destination's identities and allocation.
    /// Validation and failure behavior matches
    /// [`Self::permute_overwrite_into`].
    pub fn transpose_overwrite_into(&self, destination: &mut Self, alpha: D) -> Result<(), Error> {
        let source_codomain_rank = self.codomain_rank();
        let source_rank = self.rank();
        self.overwrite_tree_transform(
            destination,
            alpha,
            |source, _| {
                with_planar_axes(
                    source.codomain_rank(),
                    source.rank(),
                    PlanarRequestKind::FullTranspose,
                    |codomain_axes, domain_axes| {
                        Ok(TreeTransformOperation::transpose(
                            codomain_axes.iter().copied(),
                            domain_axes.iter().copied(),
                        ))
                    },
                )
            },
            |operation| {
                operation.kind() == TreeTransformOperationKind::Transpose
                    && operation
                        .codomain_permutation()
                        .iter()
                        .copied()
                        .eq((source_codomain_rank..source_rank).rev())
                    && operation
                        .domain_permutation()
                        .iter()
                        .copied()
                        .eq((0..source_codomain_rank).rev())
            },
        )
    }

    /// Overwrites `destination` with
    /// `alpha * self.transpose_axes(codomain_axes, domain_axes)`.
    /// Validation and failure behavior matches
    /// [`Self::permute_overwrite_into`].
    pub fn transpose_axes_overwrite_into(
        &self,
        destination: &mut Self,
        codomain_axes: &[usize],
        domain_axes: &[usize],
        alpha: D,
    ) -> Result<(), Error> {
        self.overwrite_tree_transform(
            destination,
            alpha,
            |source, _| {
                with_planar_axes(
                    source.codomain_rank(),
                    source.rank(),
                    PlanarRequestKind::Explicit {
                        codomain_axes,
                        domain_axes,
                    },
                    |codomain_axes, domain_axes| {
                        Ok(TreeTransformOperation::transpose(
                            codomain_axes.iter().copied(),
                            domain_axes.iter().copied(),
                        ))
                    },
                )
            },
            |operation| {
                tree_operation_matches_axes(
                    operation,
                    TreeTransformOperationKind::Transpose,
                    codomain_axes,
                    domain_axes,
                )
            },
        )
    }

    /// Overwrites `destination` with
    /// `alpha * self.repartition(destination.codomain_rank())`.
    /// Validation and failure behavior matches
    /// [`Self::permute_overwrite_into`].
    pub fn repartition_overwrite_into(
        &self,
        destination: &mut Self,
        alpha: D,
    ) -> Result<(), Error> {
        let source_codomain_rank = self.codomain_rank();
        let source_rank = self.rank();
        let destination_codomain_rank = destination.codomain_rank();
        self.overwrite_tree_transform(
            destination,
            alpha,
            |source, destination| {
                if destination.rank() != source.rank() {
                    return Err(Error::InvalidArgument(format!(
                        "repartition destination rank {} does not match source rank {}",
                        destination.rank(),
                        source.rank()
                    )));
                }
                with_planar_axes(
                    source.codomain_rank(),
                    source.rank(),
                    PlanarRequestKind::Repartition {
                        num_codomain: destination.codomain_rank(),
                    },
                    |codomain_axes, domain_axes| {
                        Ok(TreeTransformOperation::transpose(
                            codomain_axes.iter().copied(),
                            domain_axes.iter().copied(),
                        ))
                    },
                )
            },
            |operation| {
                let planar_axis = |position: usize| {
                    if position < source_codomain_rank {
                        position
                    } else {
                        source_rank - 1 - (position - source_codomain_rank)
                    }
                };
                operation.kind() == TreeTransformOperationKind::Transpose
                    && operation
                        .codomain_permutation()
                        .iter()
                        .copied()
                        .eq((0..destination_codomain_rank).map(planar_axis))
                    && operation
                        .domain_permutation()
                        .iter()
                        .copied()
                        .eq((destination_codomain_rank..source_rank)
                            .rev()
                            .map(planar_axis))
            },
        )
    }

    /// One admission and replay boundary for every typed Host overwrite.
    fn overwrite_tree_transform(
        &self,
        destination: &mut Self,
        alpha: D,
        operation: impl FnOnce(&Self, &Self) -> Result<TreeTransformOperation, Error>,
        admitted_operation_matches: impl FnMut(&TreeTransformOperation) -> bool,
    ) -> Result<(), Error> {
        if !self.runtime.same_runtime(&destination.runtime) {
            return Err(Error::RuntimeMismatch);
        }
        let identity = TypedSectorAdmission::typed_rule_identity(self.provider());
        if identity != TypedSectorAdmission::typed_rule_identity(destination.provider()) {
            return Err(Error::RuleMismatch);
        }

        let source_body = match &self.repr {
            TypedTensorRepr::Owned(body) if matches!(body.data.as_ref(), TypedData::Dense(_)) => {
                body
            }
            _ => {
                return Err(Error::InvalidArgument(
                    "typed destination tree transform requires an ordinary dense host source"
                        .to_string(),
                ))
            }
        };
        let destination_body = match &destination.repr {
            TypedTensorRepr::Owned(body) if matches!(body.data.as_ref(), TypedData::Dense(_)) => {
                body
            }
            _ => {
                return Err(Error::InvalidArgument(
                    "destination must use ordinary dense host storage".to_string(),
                ))
            }
        };
        if Arc::ptr_eq(&source_body.data, &destination_body.data) {
            return Err(Error::InvalidArgument(
                "destination storage must not alias an input".to_string(),
            ));
        }

        let admitted_operation = self.runtime.admitted_tree_pair_operation(
            &identity,
            &source_body.space,
            &destination_body.space,
            admitted_operation_matches,
        );
        let exact_layout_admitted = admitted_operation.is_some();
        let operation = match admitted_operation {
            Some(operation) => operation,
            None => operation(self, destination)?,
        };
        if !exact_layout_admitted {
            let expected = source_body
                .space
                .transformed_multiplicity_free(&operation)?;
            if destination_body.space.space() != expected.space() {
                return Err(Error::InvalidArgument(
                    "destination fusion space or block layout does not match the operation result"
                        .to_string(),
                ));
            }
        }
        let required = destination_body.space.space().required_len()?;
        let actual = match destination_body.data.as_ref() {
            TypedData::Dense(data) => data.len(),
            TypedData::Diagonal(_) => unreachable!("dense destination checked above"),
        };
        if actual != required {
            return Err(Error::InvalidArgument(format!(
                "destination storage length {actual} does not match required length {required}"
            )));
        }
        if Arc::strong_count(destination_body) != 1
            || Arc::strong_count(&destination_body.data) != 1
        {
            return Err(Error::InvalidArgument(
                "destination storage must be uniquely owned".to_string(),
            ));
        }

        let source_structure = source_body.space.space().structure();
        let source_data = match source_body.data.as_ref() {
            TypedData::Dense(data) => data.as_slice(),
            TypedData::Diagonal(_) => unreachable!("dense source checked above"),
        };
        {
            let mut lease = self.runtime.lease_context()?;
            let context = lease
                .context()
                .multiplicity_free_lane::<D>()
                .tree_context_mut();
            let TypedTensorRepr::Owned(destination_body) = &mut destination.repr else {
                unreachable!("ordinary destination checked above")
            };
            let destination_body =
                Arc::get_mut(destination_body).expect("unique destination body checked above");
            let destination_provider = destination_body.space.provider();
            let destination_structure = destination_body.space.space().structure();
            let destination_data = Arc::get_mut(&mut destination_body.data)
                .expect("unique destination payload checked above");
            let TypedData::Dense(destination_data) = destination_data else {
                unreachable!("dense destination checked above")
            };
            context.tree_transform_dyn_overwrite_into_ref(
                destination_provider,
                &operation,
                destination_structure,
                source_structure,
                destination_data.as_mut_slice(),
                source_data,
                alpha,
            )?;
        }
        if !exact_layout_admitted {
            self.runtime.admit_exact_tree_pair_layout(
                identity,
                &operation,
                &source_body.space,
                destination.logical_space(),
            );
        }
        Ok(())
    }

    /// Overwrites `destination` with
    /// `alpha * self.contract(other, lhs_axes, rhs_axes, output_axes)` while
    /// preserving the destination's provider, space, body, and dense Host
    /// allocation.
    ///
    /// # Errors
    ///
    /// Admission failures through runtime-context leasing leave `destination`
    /// unchanged. The destination is cleared immediately before shared-engine
    /// compilation/replay, so a later engine error may leave it zeroed or
    /// partially overwritten.
    #[allow(clippy::too_many_arguments)]
    pub fn contract_overwrite_into(
        &self,
        other: &Self,
        destination: &mut Self,
        lhs_axes: &[usize],
        rhs_axes: &[usize],
        output_axes: &[usize],
        alpha: D,
    ) -> Result<(), Error> {
        if !self.runtime.same_runtime(&other.runtime)
            || !self.runtime.same_runtime(&destination.runtime)
        {
            return Err(Error::RuntimeMismatch);
        }
        let identity = TypedSectorAdmission::typed_rule_identity(self.provider());
        if identity != TypedSectorAdmission::typed_rule_identity(other.provider())
            || identity != TypedSectorAdmission::typed_rule_identity(destination.provider())
        {
            return Err(Error::RuleMismatch);
        }

        let destination_body = match &destination.repr {
            TypedTensorRepr::Owned(body) if matches!(body.data.as_ref(), TypedData::Dense(_)) => {
                body
            }
            _ => {
                return Err(Error::InvalidArgument(
                    "contraction destination must use ordinary dense host storage".to_string(),
                ))
            }
        };
        if Arc::ptr_eq(&destination_body.data, &self.storage_body().data)
            || Arc::ptr_eq(&destination_body.data, &other.storage_body().data)
        {
            return Err(Error::InvalidArgument(
                "destination storage must not alias an input".to_string(),
            ));
        }

        let output_order = OutputAxisOrder::from_axes(output_axes);
        let expected = BoundDynamicFusionMapSpace::contracted_multiplicity_free_ordered(
            self.logical_space(),
            other.logical_space(),
            lhs_axes,
            rhs_axes,
            output_order,
        )?;
        if destination_body.space.space() != expected.space() {
            return Err(Error::InvalidArgument(
                "destination fusion space or block layout does not match the contraction result"
                    .to_string(),
            ));
        }
        let execution_destination = self
            .logical_space()
            .rebind_validated(&destination_body.space.validated_layout())?;

        let (lhs, lhs_data) = self.fusion_operand_and_data();
        let (rhs, rhs_data) = other.fusion_operand_and_data();
        let required_destination = destination_body.space.space().required_len()?;
        let actual_destination = match destination_body.data.as_ref() {
            TypedData::Dense(data) => data.len(),
            TypedData::Diagonal(_) => unreachable!("dense destination checked above"),
        };
        for (tensor, actual, required) in [
            ("lhs", lhs_data.len(), lhs.storage_space().required_len()?),
            ("rhs", rhs_data.len(), rhs.storage_space().required_len()?),
            ("destination", actual_destination, required_destination),
        ] {
            if actual != required {
                return Err(Error::InvalidArgument(format!(
                    "{tensor} storage length {actual} does not match required length {required}"
                )));
            }
        }
        if Arc::strong_count(destination_body) != 1
            || Arc::strong_count(&destination_body.data) != 1
        {
            return Err(Error::InvalidArgument(
                "destination storage must be uniquely owned".to_string(),
            ));
        }

        let mut lease = self.runtime.lease_context()?;
        let context = lease.context().multiplicity_free_lane::<D>();
        let TypedTensorRepr::Owned(destination_body) = &mut destination.repr else {
            unreachable!("ordinary destination checked above")
        };
        let destination_body =
            Arc::get_mut(destination_body).expect("unique destination body checked above");
        let destination_data = Arc::get_mut(&mut destination_body.data)
            .expect("unique destination payload checked above");
        let TypedData::Dense(destination_data) = destination_data else {
            unreachable!("dense destination checked above")
        };
        let zero = D::from_real(0.0);
        destination_data.fill(zero);
        context.tensorcontract_fusion_dyn_prelowered_into(
            &execution_destination,
            destination_data,
            lhs,
            lhs_data,
            rhs,
            rhs_data,
            TensorContractSpec::new_with_conjugation(
                lhs_axes,
                rhs_axes,
                output_order,
                lhs.storage_conjugate(),
                rhs.storage_conjugate(),
            ),
            alpha,
            zero,
        )?;
        Ok(())
    }

    /// Total alias of [`Self::contract_overwrite_into`].
    #[inline]
    #[allow(clippy::too_many_arguments)]
    pub fn contract_ordered_overwrite_into(
        &self,
        other: &Self,
        destination: &mut Self,
        lhs_axes: &[usize],
        rhs_axes: &[usize],
        output_axes: &[usize],
        alpha: D,
    ) -> Result<(), Error> {
        self.contract_overwrite_into(other, destination, lhs_axes, rhs_axes, output_axes, alpha)
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
    /// codomain axes first) for the new codomain and domain.
    ///
    /// # Compact storage
    ///
    /// A non-identity transform of a factor in compact diagonal storage
    /// ([`Self::svd_compact`]'s `s`, [`Self::eigh_full`]'s `d`) is
    /// **materialized** here, and so by [`Self::braid`], [`Self::transpose`],
    /// [`Self::transpose_axes`] and [`Self::repartition`] as well: the result
    /// is a dense `Σ_c k_c²` buffer. An exact identity returns the source body
    /// unchanged and preserves compact storage.
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
        if self.axes_are_identity(codomain_axes, domain_axes) {
            return Ok(self.clone());
        }
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
        if self.axes_are_identity(codomain_axes, domain_axes) {
            return Ok(self.clone());
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
        if num_codomain == self.codomain_rank() {
            return Ok(self.clone());
        }
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
        if self.rank() == 0 {
            return Ok(self.clone());
        }
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
        if self.axes_are_identity(codomain_axes, domain_axes) {
            return Ok(self.clone());
        }
        self.planar(PlanarRequestKind::Explicit {
            codomain_axes,
            domain_axes,
        })
    }

    /// Exact current split and axis order, checked without constructing a
    /// transform operation (whose inline axis storage can spill at high rank).
    #[inline]
    fn axes_are_identity(&self, codomain_axes: &[usize], domain_axes: &[usize]) -> bool {
        let codomain_rank = self.codomain_rank();
        codomain_axes.iter().copied().eq(0..codomain_rank)
            && domain_axes.iter().copied().eq(codomain_rank..self.rank())
    }

    /// Shared body of the three planar operations: derive the planar axis
    /// order, let the expert layer check it, and run it as a transpose.
    ///
    /// The shared axis derivation defines what "planar" means for each request
    /// kind; duplicating it here would allow the definitions to drift.
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
    R: TypedSectorAdmission,
    R::Mode: TypedTensorProductDispatch<R, D>,
    D: TensorScalar,
{
    /// Tensor product in one category, ordered as
    /// `codomain(self), codomain(other); domain(self), domain(other)`.
    ///
    /// The two codomain trees and the two domain trees are merged
    /// independently with F moves. No legs cross and no R symbol is needed,
    /// including for a `NoBraiding` provider.
    ///
    /// Equal provider identities are sufficient; the two tensors may own
    /// different `Arc` allocations. The output always retains `self`'s exact
    /// provider allocation.
    ///
    /// # Errors
    ///
    /// [`Error::RuntimeMismatch`] is reported before provider work. Checked
    /// Generic providers preserve algebra and malformed-F failures in
    /// [`GenericTensorError::TensorProduct`].
    pub fn otimes(&self, other: &Self) -> Result<Self, TypedFacadeError<R>> {
        if !self.runtime.same_runtime(&other.runtime) {
            return Err(Error::RuntimeMismatch.into());
        }
        <R::Mode as TypedTensorProductDispatch<R, D>>::tensor_product(self, other)
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
                let destination = self
                    .logical_space()
                    .transformed_multiplicity_free(&operation)?;
                let transformed = crate::tensor_core::transform_rank_one_diagonal_spectrum(
                    self.logical_space().provider(),
                    self.logical_space().space(),
                    destination.space(),
                    &operation,
                    spectrum,
                )?;
                return Ok(self.with_spectrum_on(destination, transformed));
            }
        }
        if let TypedTensorRepr::Adjoint(view) = &self.repr {
            let parent_space = view.parent.space.space();
            let lowered = lower_adjoint_tree_transform_operation(
                parent_space.nout(),
                parent_space.nin(),
                &operation,
            )?;
            let parent = Self {
                runtime: self.runtime.clone(),
                repr: TypedTensorRepr::Owned(Arc::clone(&view.parent)),
            };
            return parent.tree_transform_multiplicity_free(lowered)?.adjoint();
        }
        // Leasing rather than locking, matching the erased path: independent
        // operations on one runtime must not serialize behind each other.
        let mut lease = self.runtime.lease_context()?;
        let body = self.owned_body().expect("owned tree transform input");
        let (space, data) = tree_transform_owned_multiplicity_free(
            lease.context().multiplicity_free_lane::<D>(),
            BoundDynamicTensorRef::try_new(&body.space, body.materialized_dense_data())?,
            operation,
        )?;
        Ok(Self {
            runtime: self.runtime.clone(),
            repr: owned_repr(TypedTensorBody::dense(space, data)),
        })
    }
}

impl<R, D> TensorMap<R, D>
where
    R: TypedSectorAdmission,
    R::Mode: TypedTensorContractDispatch<R, D>,
    D: TensorScalar,
{
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
    /// **Fermionic semantics**: like TensorKit `tensorcontract!` / `@tensor`,
    /// this **twists**
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
    ///   Checked Generic providers preserve provider and replay failures in
    ///   [`GenericTensorError::Plan`] and currently accept direct-owned inputs.
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
    ) -> Result<Self, TypedFacadeError<R>> {
        // The one check the expert layer cannot make: it never sees the two
        // runtimes, and mixing execution state across them is a trust-boundary
        // violation rather than an algebra error. Mirrors the erased facade's
        // `check_same_world`. Dtype and placement need no arm here — `D` is a
        // type parameter and the typed facade is host-only.
        if !self.runtime.same_runtime(&other.runtime) {
            return Err(Error::RuntimeMismatch.into());
        }
        <R::Mode as TypedTensorContractDispatch<R, D>>::contract(
            self,
            other,
            lhs_axes,
            rhs_axes,
            output_axes,
        )
    }

    /// Documented alias of [`Self::contract`]: same arguments, same
    /// semantics, same compact fast paths and complexity, same errors — the
    /// delegation is total, so everything is stated there once.
    ///
    /// [`Self::contract`] always takes the order explicitly, so this alias
    /// exists only to make that intent visible at the call site.
    ///
    /// **TensorKit correspondence.** TensorKit has no `contract_ordered`
    /// entry point either: the counterpart of `output_axes` is
    /// `tensorcontract!`'s `pAB` output permutation (`TO.tensorcontract!`;
    /// the destination structure is `permute(compose(sA, sB), pAB)`, per
    /// `tensorcontract_structure`).
    ///
    /// # Errors
    ///
    /// Exactly [`Self::contract`]'s.
    #[inline]
    pub fn contract_ordered(
        &self,
        other: &Self,
        lhs_axes: &[usize],
        rhs_axes: &[usize],
        output_axes: &[usize],
    ) -> Result<Self, TypedFacadeError<R>> {
        self.contract(other, lhs_axes, rhs_axes, output_axes)
    }
}

impl<R, D> TensorMap<R, D>
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64> + CheckedFusionAlgebra + SectorCodec,
    D: TensorScalar,
{
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
        if product.left_rule().rule_identity() != self.logical_space().provider().rule_identity()
            || product.right_rule().rule_identity()
                != other.logical_space().provider().rule_identity()
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
}

impl<R, D> TensorMap<R, D>
where
    R: TypedSectorAdmission,
    R::Mode: TypedTensorContractDispatch<R, D>,
    D: TensorScalar,
{
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
    /// codomain/domain split itself, and TensorKit's `*` takes none.
    ///
    /// The result is bound to `self`'s provider allocation — the same
    /// left-authority rule as [`Self::contract`] and [`Self::zeros`] — with one
    /// exemption: the `D * t` compact arm below returns `t`'s own space and
    /// runtime handle, because that space *is* the destination and rebuilding
    /// it under the left allocation would be a copy for nothing. The two
    /// allocations must already agree on
    /// [`tenet_core::FusionRule::rule_identity`] for the composition to be
    /// legal at all, so the choice is immaterial to the algebra.
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
    ///   Checked Generic failures use [`GenericTensorError::Plan`].
    #[doc(alias = "mul")]
    pub fn compose(&self, other: &Self) -> Result<Self, TypedFacadeError<R>> {
        // Runtime first, exactly as `contract`: crossing runtimes is a
        // trust-boundary violation rather than an algebra error, and the
        // expert layer never sees the two runtimes.
        if !self.runtime.same_runtime(&other.runtime) {
            return Err(Error::RuntimeMismatch.into());
        }
        <R::Mode as TypedTensorContractDispatch<R, D>>::compose(self, other)
    }
}

impl<R, D> TensorMap<R, D>
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64> + CheckedFusionAlgebra + SectorCodec,
    D: TensorScalar,
{
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
        let (left, right) = (self.logical_space().space(), other.logical_space().space());
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
    /// its own; it is a scale followed by one permutation.
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
    /// contracted leg of `other`, where [`Self::compose`] does not. Here the case
    /// cannot arise, so the arm declines instead of carrying arithmetic no test
    /// could reach: a compact payload's bond leg is built non-dual
    /// (`diagonal_bond_bound_space_like`), the arms pair it with a *codomain*
    /// leg of `other` whose external duality is exactly its raw flag, and
    /// admissibility forces that flag to equal the bond's. The guard stays
    /// because the first constructor of a compact payload on a dual bond leg —
    /// or of an arm pairing a domain leg of `other` — should decline rather
    /// than silently return a wrong sign.
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
        let fermionic = self.logical_space().provider().braiding_style()
            == tenet_core::BraidingStyleKind::Fermionic;
        if fermionic
            && other
                .logical_space()
                .space()
                .homspace()
                .external_axis_is_dual(rhs_axis)
                != Some(false)
        {
            return Ok(None);
        }
        let (left, right) = (self.logical_space().space(), other.logical_space().space());
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
                let Some(completed) = other
                    .scaled_axis(Some(rhs_axis), spectrum)?
                    // One open axis of `self` survives, so the destination's
                    // codomain rank is one — the contraction convention puts
                    // every open axis of the left operand there.
                    .permuted_to_output(&source, output_axes, 1)?
                else {
                    return Ok(None);
                };
                // Scaling and permutation deliberately run on `other`; only
                // after both succeed do we rebind their validated owned-dense
                // result to the public contract's exact left authority.
                let TensorMap { repr, .. } = completed;
                let TypedTensorRepr::Owned(body) = repr else {
                    return Ok(None);
                };
                if !matches!(body.data.as_ref(), TypedData::Dense(_)) {
                    return Ok(None);
                }
                let space = self
                    .logical_space()
                    .rebind_validated(&body.space.validated_layout())?;
                Ok(Some(Self {
                    runtime: self.runtime.clone(),
                    repr: owned_repr(TypedTensorBody::with_shared_payload(
                        space,
                        Arc::clone(&body.data),
                    )),
                }))
            }
            (None, None) => Ok(None),
        }
    }

    /// This tensor's axes, listed in `source[output_axes[..]]` order and split
    /// at `codomain_rank`, or `None` when `output_axes` is not a permutation of
    /// `0..source.len()`.
    ///
    /// `source` is the contraction's default output order expressed as axes of
    /// the scaled operand. An `output_axes` that is not a
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
        let mut data = if matches!(&self.repr, TypedTensorRepr::Adjoint(_)) {
            let (operand, source) = self.fusion_operand_and_data();
            let mut data = vec![D::from_real(0.0); self.logical_space().space().required_len()?];
            tenet_tensors::oriented_fusion_add_into(
                self.logical_space().space().structure(),
                &mut data,
                operand,
                source,
                operand,
                source,
                D::from_real(1.0),
                D::from_real(0.0),
            )?;
            data
        } else {
            self.owned_body()
                .expect("owned scaled-axis input")
                .materialized_dense_data()
                .to_vec()
        };
        tenet_matrixalgebra::scale_axis_by_spectrum_mapped(
            self.logical_space().space(),
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
    /// Sorted by sector id first because the bond leg is built from this order.
    fn diagonal_factor<V>(
        &self,
        spectrum: Vec<tenet_matrixalgebra::SectorSpectrum<V>>,
        to_scalar: impl Fn(V) -> D,
    ) -> Result<Self, Error> {
        diagonal_factor_on(&self.runtime, self.logical_space(), spectrum, to_scalar)
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
        let provider = self.logical_space().provider();
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
        let body = self.owned_body().ok_or_else(|| {
            internal_layout_error("factorization input must be owned after adjoint dispatch")
        })?;
        BoundDynamicTensorRef::try_new(&body.space, body.materialized_dense_data())
            .map_err(Error::from)
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
        let (u, vh, spectrum) = match &self.repr {
            TypedTensorRepr::Adjoint(view) => tenet_matrixalgebra::svd_compact_adjoint_factors_dyn(
                dense.dense(),
                &BoundDynamicTensorRef::try_new(
                    &view.parent.space,
                    view.parent.materialized_dense_data(),
                )?,
            )?,
            TypedTensorRepr::Owned(_) => {
                tenet_matrixalgebra::svd_compact_factors_dyn(dense.dense(), &self.bound_ref()?)?
            }
        };
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
        let out = match &self.repr {
            TypedTensorRepr::Adjoint(view) => tenet_matrixalgebra::svd_full_adjoint_dyn(
                dense.dense(),
                &BoundDynamicTensorRef::try_new(
                    &view.parent.space,
                    view.parent.materialized_dense_data(),
                )?,
            )?,
            TypedTensorRepr::Owned(_) => {
                tenet_matrixalgebra::svd_full_dyn(dense.dense(), &self.bound_ref()?)?
            }
        };
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
    /// named struct.
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
        let (u, vh, singular_values, error) = match &self.repr {
            TypedTensorRepr::Adjoint(view) => tenet_matrixalgebra::svd_trunc_adjoint_factors_dyn(
                dense.dense(),
                &BoundDynamicTensorRef::try_new(
                    &view.parent.space,
                    view.parent.materialized_dense_data(),
                )?,
                truncation,
            )?,
            TypedTensorRepr::Owned(_) => tenet_matrixalgebra::svd_trunc_factors_dyn(
                dense.dense(),
                &self.bound_ref()?,
                truncation,
            )?,
        };
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
        // Singular values and coupled-sector ids are invariant under adjoint,
        // so an oriented input or logical-payload copy cannot change this output.
        let raw = match &self.repr {
            TypedTensorRepr::Adjoint(view) => tenet_matrixalgebra::svd_vals_dyn(
                dense.dense(),
                &BoundDynamicTensorRef::try_new(
                    &view.parent.space,
                    view.parent.materialized_dense_data(),
                )?,
            )?,
            TypedTensorRepr::Owned(_) => {
                tenet_matrixalgebra::svd_vals_dyn(dense.dense(), &self.bound_ref()?)?
            }
        };
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
    /// `O(Σ_c n_c³)` — sectorwise cubic; the seam runs one dense QR per
    /// coupled-sector matrix. A lazy adjoint first allocates its whole logical
    /// dense payload as an operation-local owned tensor. That allocation is
    /// not published in the receiver's reusable materialization cache, and
    /// the returned factors are owned. A compact-diagonal payload
    /// (TensorKit's `DiagonalTensorMap`) is materialized into the dense coupled
    /// buffer first, through the same [`Self::data`] route as
    /// [`Self::left_polar`]. TensorKit 0.17 *does* keep a diagonal QR compact
    /// (MatrixAlgebraKit's `DiagonalAlgorithm`); that fast path is not adopted
    /// here — the issue #613 Group 4 contract requires every compact fast path
    /// to be re-proven individually, the same deferral the polars record.
    pub fn qr_compact(&self) -> Result<(Self, Self), Error> {
        if matches!(&self.repr, TypedTensorRepr::Adjoint(_)) {
            return self.materialized_tensor_uncached()?.qr_compact();
        }
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
    /// As [`Self::qr_compact`]: sectorwise cubic. This includes the uncached
    /// whole-logical-payload allocation for a lazy adjoint. A compact-diagonal
    /// payload is materialized dense first (TensorKit's `DiagonalAlgorithm`
    /// covers `qr_full!` too — same non-adoption, same #613 Group 4 deferral).
    pub fn qr_full(&self) -> Result<(Self, Self), Error> {
        if matches!(&self.repr, TypedTensorRepr::Adjoint(_)) {
            return self.materialized_tensor_uncached()?.qr_full();
        }
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
    /// Sectorwise cubic. A lazy adjoint runs compact QR on its owned parent,
    /// reverses and adjoints the factors, then materializes both outputs into
    /// detached owned tensors. This publishes no receiver cache and retains
    /// neither parent factor buffer. A compact-diagonal payload is materialized
    /// dense first (TensorKit's `DiagonalAlgorithm` covers the LQ pair as well
    /// — same non-adoption, same #613 Group 4 deferral).
    pub fn lq_compact(&self) -> Result<(Self, Self), Error> {
        if matches!(&self.repr, TypedTensorRepr::Adjoint(_)) {
            let (q, r) = self.adjoint()?.qr_compact()?;
            return Ok((
                r.adjoint()?.materialized_tensor_uncached()?,
                q.adjoint()?.materialized_tensor_uncached()?,
            ));
        }
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
    /// As [`Self::lq_compact`]: sectorwise cubic, including the lazy-adjoint
    /// parent-QR route and two detached owned output payloads. A compact-diagonal
    /// payload is materialized dense first.
    pub fn lq_full(&self) -> Result<(Self, Self), Error> {
        if matches!(&self.repr, TypedTensorRepr::Adjoint(_)) {
            let (q, r) = self.adjoint()?.qr_full()?;
            return Ok((
                r.adjoint()?.materialized_tensor_uncached()?,
                q.adjoint()?.materialized_tensor_uncached()?,
            ));
        }
        let mut dense = self.runtime.lease_dense();
        let (l, q) = tenet_matrixalgebra::lq_full_dyn(dense.dense(), &self.bound_ref()?)?;
        Ok((self.wrap_bound_factor(l), self.wrap_bound_factor(q)))
    }

    /// TensorKit 0.17 `left_orth`: the left isometry factorization
    /// `t = v * c`, `v` isometric and `c` the corestriction.
    ///
    /// TensorKit's default `kind` is `:qr`, so this delegates directly to
    /// [`Self::qr_compact`].
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
    /// [`Self::qr_compact`]. A lazy adjoint runs the owned parent's
    /// [`Self::right_null`] and returns its detached adjoint, without
    /// materializing the receiver.
    pub fn left_null(&self) -> Result<Self, Error> {
        if matches!(&self.repr, TypedTensorRepr::Adjoint(_)) {
            return self
                .adjoint()?
                .right_null()?
                .adjoint()?
                .materialized_tensor_uncached();
        }
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
    /// materialized dense first. A lazy adjoint mirrors the parent redirect
    /// described there.
    pub fn right_null(&self) -> Result<Self, Error> {
        if matches!(&self.repr, TypedTensorRepr::Adjoint(_)) {
            return self
                .adjoint()?
                .left_null()?
                .adjoint()?
                .materialized_tensor_uncached();
        }
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
    /// also exposes algorithm kinds for the polars; TeNeT deliberately does
    /// not. A lazy typed adjoint executes
    /// the opposite polar on its exact owned parent, keeps the already-owned
    /// positive factor, and returns an owned adjoint of the isometry without
    /// publishing the receiver's materialization cache.
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
        if let TypedTensorRepr::Adjoint(view) = &self.repr {
            let mut dense = self.runtime.lease_dense();
            let mut lease = self.runtime.lease_context()?;
            let (w, p) = tenet_matrixalgebra::left_polar_adjoint_parent_dyn(
                dense.dense(),
                lease.context().multiplicity_free_lane::<D>(),
                &BoundDynamicTensorRef::try_new(
                    &view.parent.space,
                    view.parent.materialized_dense_data(),
                )?,
            )?;
            return Ok((self.wrap_bound_factor(w), self.wrap_bound_factor(p)));
        }
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
        if let TypedTensorRepr::Adjoint(view) = &self.repr {
            let mut dense = self.runtime.lease_dense();
            let mut lease = self.runtime.lease_context()?;
            let (p, w) = tenet_matrixalgebra::right_polar_adjoint_parent_dyn(
                dense.dense(),
                lease.context().multiplicity_free_lane::<D>(),
                &BoundDynamicTensorRef::try_new(
                    &view.parent.space,
                    view.parent.materialized_dense_data(),
                )?,
            )?;
            return Ok((self.wrap_bound_factor(p), self.wrap_bound_factor(w)));
        }
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
    /// The `(d, v)` order is MatrixAlgebraKit's `initialize_output` order, not
    /// the `v, d` reading order of the formula.
    ///
    /// # Errors
    ///
    /// [`Error::Operation`] when the tensor is not an endomorphism or its
    /// coupled blocks are not Hermitian, and otherwise
    /// [`Error::Core`] / [`Error::FusionAlgebra`] from the seam — which owns
    /// those rules, so they are not re-checked here.
    pub fn eigh_full(&self) -> Result<(Self, Self), Error> {
        if matches!(&self.repr, TypedTensorRepr::Adjoint(_)) {
            return self.materialized_tensor_uncached()?.eigh_full();
        }
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
        if matches!(&self.repr, TypedTensorRepr::Adjoint(_)) {
            return self.materialized_tensor_uncached()?.eigh_trunc(truncation);
        }
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
        if matches!(&self.repr, TypedTensorRepr::Adjoint(_)) {
            return self.materialized_tensor_uncached()?.eigh_vals();
        }
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
        if matches!(&self.repr, TypedTensorRepr::Adjoint(_)) {
            return self.materialized_tensor_uncached()?.eig_full();
        }
        let mut dense = self.runtime.lease_dense();
        let out = tenet_matrixalgebra::eig_full_dyn(dense.dense(), &self.bound_ref()?)?;
        let (v, eigenvalues) = out.into_parts();
        Ok((
            diagonal_factor_on(
                &self.runtime,
                self.logical_space(),
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
        if matches!(&self.repr, TypedTensorRepr::Adjoint(_)) {
            return self.materialized_tensor_uncached()?.eig_trunc(truncation);
        }
        let mut dense = self.runtime.lease_dense();
        let out =
            tenet_matrixalgebra::eig_trunc_dyn(dense.dense(), &self.bound_ref()?, truncation)?;
        let (v, eigenvalues, error) = out.into_parts();
        Ok(EigTrunc {
            d: diagonal_factor_on(
                &self.runtime,
                self.logical_space(),
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
        if matches!(&self.repr, TypedTensorRepr::Adjoint(_)) {
            return self.materialized_tensor_uncached()?.eig_vals();
        }
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
    /// couples sectors. A dense lazy adjoint builds one operation-local logical
    /// payload per call without publishing its reusable receiver cache. Compact
    /// input (TensorKit's `DiagonalTensorMap`): the **O(rank) elementwise
    /// arm**, `exp(s_i)` over the `Σ_c k_c` stored values, staying compact.
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
        let local = matches!(&self.repr, TypedTensorRepr::Adjoint(_))
            .then(|| self.materialized_tensor_uncached())
            .transpose()?;
        let body = local
            .as_ref()
            .and_then(Self::owned_body)
            .unwrap_or_else(|| self.owned_body().expect("owned representation"));
        let out = tenet_matrixalgebra::exp_dyn(
            dense.dense(),
            lease.context().multiplicity_free_lane::<D>(),
            &BoundDynamicTensorRef::try_new(&body.space, body.materialized_dense_data())?,
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
    /// built on either side of that arm. A host lazy adjoint solves its owned
    /// parent and returns a detached owned adjoint of that inverse; it does not
    /// allocate or publish a separate receiver-materialization payload.
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
        if matches!(&self.repr, TypedTensorRepr::Adjoint(_)) {
            // (A†)^-1 = (A^-1)†. Keep the receiver's lazy cache cold by
            // solving the owned parent, then detach the final adjoint so the
            // result retains neither the parent inverse nor its payload.
            return self
                .adjoint()?
                .inv()?
                .adjoint()?
                .materialized_tensor_uncached();
        }
        let mut dense = self.runtime.lease_dense();
        let out = tenet_matrixalgebra::inv_direct_dyn(dense.dense(), &self.bound_ref()?)?;
        Ok(self.wrap_bound_factor(out))
    }

    /// TensorKit 0.17 / MatrixAlgebraKit `pinv`: the Moore-Penrose
    /// thresholded pseudo-inverse `t⁺ = V S⁺ Uᴴ`, where `t = U S Vᴴ` is the compact SVD and
    /// `S⁺` inverts every singular value above the cutoff and sends the rest to
    /// zero. This is the exact Moore-Penrose inverse of the hard-thresholded
    /// effective-rank tensor `t_r`. It is the Moore-Penrose inverse of `t`
    /// itself only when no genuinely nonzero singular value is discarded, and
    /// then satisfies `t t⁺ t = t`. It reduces to [`Self::inv`] when `t` is
    /// nonsingular and `rcond` keeps every singular value.
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
        let out = match &self.repr {
            TypedTensorRepr::Adjoint(view) => tenet_matrixalgebra::pinv_adjoint_parent_dyn(
                dense.dense(),
                lease.context().multiplicity_free_lane::<D>(),
                &BoundDynamicTensorRef::try_new(
                    &view.parent.space,
                    view.parent.materialized_dense_data(),
                )?,
                rcond,
            )?,
            TypedTensorRepr::Owned(_) => tenet_matrixalgebra::pinv_dyn(
                dense.dense(),
                lease.context().multiplicity_free_lane::<D>(),
                &self.bound_ref()?,
                rcond,
            )?,
        };
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
    ///   points at the complex payload, matching TensorKit's diagonal-path
    ///   `DomainError`. TensorKit's *dense* path
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
    /// root itself is still only `Σ_c n_c` square roots. A dense lazy adjoint
    /// builds one operation-local logical payload without publishing its
    /// reusable receiver cache.
    pub fn sqrt(&self) -> Result<Self, Error> {
        // Same guard as the erased facade's, and the same one
        // [`is_diagonal_bond_space`] applies to compact *destinations*: here it
        // is asked of the receiver, which is what makes it reachable.
        if !is_diagonal_bond_space(self.logical_space().space()) {
            return Err(Error::InvalidArgument(
                "sqrt requires a diagonal bond tensor `[v] <- [v]` (equal single \
                 codomain and domain legs), like the `s` factor of svd_trunc"
                    .to_string(),
            ));
        }
        if let Some(spectrum) = self.spectrum() {
            return Ok(self.with_spectrum(map_spectrum(spectrum, D::sqrt_value)?));
        }
        if matches!(&self.repr, TypedTensorRepr::Adjoint(_)) {
            return self.materialized_tensor_uncached()?.sqrt();
        }
        // Dense payload on a bond space: block-diagonal by the space's shape,
        // but only by convention — the buffer is free to hold anything, so the
        // off-diagonal entries are checked rather than assumed. Skipping the
        // check would silently drop them.
        let data = self
            .owned_body()
            .expect("owned square-root input")
            .materialized_dense_data();
        let zero = num_complex::Complex64::new(0.0, 0.0);
        let mut out = vec![D::from_real(0.0); data.len()];
        let structure = self.logical_space().space().structure();
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
            repr: owned_repr(TypedTensorBody::dense(self.logical_space().clone(), data)),
        }
    }

    /// A sibling on this tensor's own space carrying a new compact spectrum.
    /// Every operation that reaches this keeps the bond space it was called on,
    /// so the checked admission proof carries over exactly as for
    /// [`Self::with_data`].
    fn with_spectrum(&self, spectrum: Vec<tenet_matrixalgebra::SectorSpectrum<D>>) -> Self {
        Self {
            runtime: self.runtime.clone(),
            repr: owned_repr(TypedTensorBody::diagonal(
                self.logical_space().clone(),
                spectrum,
            )),
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
            repr: owned_repr(TypedTensorBody::diagonal(space, spectrum)),
        }
    }

    /// The compact payload, when this tensor has one.
    fn spectrum(&self) -> Option<&[tenet_matrixalgebra::SectorSpectrum<D>]> {
        let body = self.owned_body()?;
        match body.data.as_ref() {
            TypedData::Diagonal(spectrum) => Some(spectrum),
            TypedData::Dense(_) => None,
        }
    }

    /// Whether two operands' providers are the same rule. The compact paths
    /// below skip the expert layer, which is where a mismatch would otherwise
    /// be caught, so they have to ask themselves.
    fn same_rule(&self, other: &Self) -> bool {
        self.logical_space().provider().rule_identity()
            == other.logical_space().provider().rule_identity()
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

    /// The linear combination `alpha * self + beta * other`.
    ///
    /// Both operands must live on the same runtime and on the same space —
    /// identical hom space and block layout — since the combination is
    /// element-wise on the shared storage order.
    ///
    /// # False friend
    ///
    /// VectorInterface's `add(y, x, α, β)` is `y * β + x * α`: its **first**
    /// coefficient belongs to its **second** argument. Here `alpha` belongs to
    /// `self` and `beta` to `other`. Callers
    /// arriving from Julia should go by the argument order, not by the
    /// coefficient names.
    ///
    /// # Errors
    ///
    /// - [`Error::RuntimeMismatch`] when the operands belong to different
    ///   runtimes, as for [`Self::contract`].
    /// - [`Error::InvalidArgument`] when they do not live on the same space.
    ///   The space comparison already covers rule identity, so a separate
    ///   check would only re-report the same disagreement.
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
        if self.logical_space().space() != other.logical_space().space() {
            return Err(Error::InvalidArgument(
                "tensors live on different spaces or block layouts".to_string(),
            ));
        }
        if matches!(&self.repr, TypedTensorRepr::Adjoint(_))
            || matches!(&other.repr, TypedTensorRepr::Adjoint(_))
        {
            if let (Some(spectrum), TypedTensorRepr::Adjoint(_)) = (self.spectrum(), &other.repr) {
                let (operand, dense) = other.fusion_operand_and_data();
                let mut data =
                    vec![D::from_real(0.0); self.logical_space().space().required_len()?];
                tenet_tensors::oriented_fusion_add_into(
                    self.logical_space().space().structure(),
                    &mut data,
                    operand,
                    dense,
                    operand,
                    dense,
                    beta,
                    D::from_real(0.0),
                )?;
                add_spectrum_into(self.logical_space().space(), &mut data, spectrum, alpha)?;
                return Ok(self.with_data(data));
            }
            if let (TypedTensorRepr::Adjoint(_), Some(spectrum)) = (&self.repr, other.spectrum()) {
                let (operand, dense) = self.fusion_operand_and_data();
                let mut data =
                    vec![D::from_real(0.0); self.logical_space().space().required_len()?];
                tenet_tensors::oriented_fusion_add_into(
                    self.logical_space().space().structure(),
                    &mut data,
                    operand,
                    dense,
                    operand,
                    dense,
                    alpha,
                    D::from_real(0.0),
                )?;
                add_spectrum_into(self.logical_space().space(), &mut data, spectrum, beta)?;
                return Ok(self.with_data(data));
            }
            let (lhs, lhs_data) = self.fusion_operand_and_data();
            let (rhs, rhs_data) = other.fusion_operand_and_data();
            let mut data = vec![D::from_real(0.0); self.logical_space().space().required_len()?];
            tenet_tensors::oriented_fusion_add_into(
                self.logical_space().space().structure(),
                &mut data,
                lhs,
                lhs_data,
                rhs,
                rhs_data,
                alpha,
                beta,
            )?;
            return Ok(self.with_data(data));
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
                    self.logical_space().space(),
                    other
                        .owned_body()
                        .expect("owned add input")
                        .materialized_dense_data(),
                    beta,
                    diagonal,
                    alpha,
                )?))
            }
            (None, Some(diagonal)) => {
                return Ok(self.with_data(scatter_spectrum(
                    self.logical_space().space(),
                    self.owned_body()
                        .expect("owned add input")
                        .materialized_dense_data(),
                    alpha,
                    diagonal,
                    beta,
                )?))
            }
            (None, None) => {}
        }
        Ok(self.with_data(
            self.owned_body()
                .expect("owned add input")
                .materialized_dense_data()
                .iter()
                .zip(
                    other
                        .owned_body()
                        .expect("owned add input")
                        .materialized_dense_data(),
                )
                .map(|(&x, &y)| x * alpha + y * beta)
                .collect(),
        ))
    }

    /// `factor * self` (TensorKit `scale`).
    ///
    /// Infallible because the host payload dtype is the type parameter `D`;
    /// `factor` is simply another `D`.
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
        if let TypedTensorRepr::Adjoint(view) = &self.repr {
            let parent = Self {
                runtime: self.runtime.clone(),
                repr: TypedTensorRepr::Owned(Arc::clone(&view.parent)),
            };
            return parent
                .scale(FactorScalar::adjoint(factor))
                .adjoint()
                .expect("scaling a pre-admitted adjoint must preserve its layout");
        }
        self.with_data(
            self.owned_body()
                .expect("owned scale input")
                .materialized_dense_data()
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
    /// TensorKit's native parallel-list `Index2Tuple` is what the seam takes
    /// internally; the Rust API uses `&[(usize, usize)]`.
    ///
    /// # Complexity
    ///
    /// Dense storage runs the partial-trace engine over the whole payload. A
    /// compact spectrum factor traced over its only pair reduces the stored
    /// spectrum in `O(Σ_c k_c)` without materializing (#604), with a
    /// deliberately narrow guard: one pair on a rank-(1,1) source, where the
    /// destination tree is empty and the coefficient collapses to a per-sector
    /// scalar,
    /// `dim(c) · θ(c)` on a direct traced codomain leg and `dim(c)` on a dual
    /// one. That twist is what makes this the supertrace and not [`Self::tr`];
    /// the coefficient is checked numerically against the engine route by the
    /// oracle sweeps in `tests/typed_facade.rs` and
    /// `tenet/src/tensor/compact_diagonal_tests.rs`.
    ///
    /// # Errors
    ///
    /// [`Error::InvalidArgument`] when the pair list is malformed — an axis out
    /// of range, or one named twice.
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
        let mapped_output_axes;
        let mapped_trace_lhs;
        let mapped_trace_rhs;
        let (source_space, source_data, axes) = match &self.repr {
            TypedTensorRepr::Owned(body) => (
                &body.space,
                match &*body.data {
                    TypedData::Dense(data) => Some(data.as_slice()),
                    TypedData::Diagonal(_) => None,
                },
                tenet_tensors::TensorTraceAxisSpec::new(&output_axes, &trace_lhs, &trace_rhs),
            ),
            TypedTensorRepr::Adjoint(view) => {
                let parent = view.parent.space.space();
                mapped_output_axes =
                    logical_adjoint_axes_to_parent(parent.nout(), parent.nin(), &output_axes);
                mapped_trace_lhs =
                    logical_adjoint_axes_to_parent(parent.nout(), parent.nin(), &trace_lhs);
                mapped_trace_rhs =
                    logical_adjoint_axes_to_parent(parent.nout(), parent.nin(), &trace_rhs);
                (
                    &view.parent.space,
                    Some(view.parent.materialized_dense_data()),
                    tenet_tensors::TensorTraceAxisSpec::new_with_conjugation(
                        &mapped_output_axes,
                        &mapped_trace_lhs,
                        &mapped_trace_rhs,
                        true,
                    ),
                )
            }
        };
        // Preflight first, exactly as the erased facade does: the checked
        // homspace selection must fail before any destination layout is
        // derived, so a rejected trace publishes no state.
        let homspace = tenet_tensors::tensortrace_fusion_dyn_selected_homspace_checked(
            source_space,
            axes,
            destination_codomain_rank,
        )?;
        let space = source_space.derive_from_final_homspace(homspace)?;
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
        // defensive parity with the erased guard, not a reachable branch. A
        // lazy dense adjoint has no compact spectrum and therefore goes through
        // the parent-oriented trace seam; a compact adjoint remains an owned
        // compact tensor. The coefficient is not derivable here — it is pinned
        // against the erased arm and the engine route by the oracle sweeps in
        // `tests/typed_facade.rs` (`compact_full_trace_*`) and, on the erased
        // side, `tensor/compact_diagonal_tests.rs`.
        if let Some(spectrum) = self.spectrum() {
            if rank == 2 && self.codomain_rank() == 1 && pairs.len() == 1 {
                let traced_leg_is_dual: bool =
                    self.logical_space().space().homspace().codomain().legs()[0].is_dual();
                let provider: &R = self.logical_space().provider();
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
                    repr: owned_repr(TypedTensorBody::dense(space, vec![value])),
                });
            }
        }
        let source_data = source_data.unwrap_or_else(|| {
            self.owned_body()
                .expect("owned trace input")
                .materialized_dense_data()
        });
        let data = tenet_tensors::tensortrace_fusion_dyn_owned_checked(
            &space,
            source_space,
            source_data,
            axes,
            D::from_real(1.0),
        )?;
        Ok(Self {
            runtime: self.runtime.clone(),
            repr: owned_repr(TypedTensorBody::dense(space, data)),
        })
    }

    /// TensorKit `adjoint` (dagger): swaps codomain and domain and
    /// conjugate-transposes every block. Real payloads are transposed only;
    /// c64 entries are conjugated as well.
    ///
    /// Dense storage is a lazy parent-backed view, matching TensorKit's
    /// `AdjointTensorMap`: metadata swaps immediately, and only [`Self::data`]
    /// publishes a deferred whole-payload materialization across clones.
    /// Compact diagonal storage keeps its established `O(Σ_c k_c)` owned
    /// conjugation path and never enters the general lazy cell.
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
        self.dense_adjoint_view()
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
            let provider = self.logical_space().provider();
            return Ok(Self::compact_inner(spectrum, spectrum, provider)?.re.sqrt());
        }
        if let TypedTensorRepr::Adjoint(view) = &self.repr {
            let parent = Self {
                runtime: self.runtime.clone(),
                repr: TypedTensorRepr::Owned(Arc::clone(&view.parent)),
            };
            return parent.norm();
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
        if let TypedTensorRepr::Adjoint(view) = &self.repr {
            return Ok(view
                .parent
                .materialized_dense_data()
                .iter()
                .map(|&value| value.widen_complex().norm())
                .fold(0.0, f64::max));
        }
        Ok(self
            .owned_body()
            .expect("owned norm input")
            .materialized_dense_data()
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
        if let TypedTensorRepr::Adjoint(view) = &self.repr {
            let parent = Self {
                runtime: self.runtime.clone(),
                repr: TypedTensorRepr::Owned(Arc::clone(&view.parent)),
            };
            return parent.norm_p(p);
        }
        let provider = self.logical_space().provider();
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
            self.logical_space().space().structure(),
            self.logical_space().space().nout(),
            self.owned_body()
                .expect("owned norm input")
                .materialized_dense_data(),
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
            self.logical_space().provider(),
            self.logical_space().space().structure(),
            self.logical_space().space().nout(),
            self.owned_body()
                .expect("owned norm input")
                .materialized_dense_data(),
            self.owned_body()
                .expect("owned norm input")
                .materialized_dense_data(),
        )
    }

    /// TensorKit `dot(x, y)`: the quantum-dimension-weighted Frobenius inner
    /// product `Σ_c dim(c) * <a_c, b_c>` with **`self` conjugated** — the
    /// product is conjugate-linear in its first argument.
    ///
    /// `t.inner(&t)?` is `t.norm()?²` up to floating point, and for `D = f64`
    /// the result is exactly real.
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
        if self.logical_space().space() != other.logical_space().space() {
            return Err(Error::InvalidArgument(
                "tensors live on different spaces or block layouts".to_string(),
            ));
        }
        // Two compact spectra reduce without either being materialized. A
        // compact/dense pair needs the compact factor's dense cache, while a
        // lazy dense operand remains parent-oriented.
        if let (Some(lhs), Some(rhs)) = (self.spectrum(), other.spectrum()) {
            let provider = self.logical_space().provider();
            return Ok(D::from_complex64(Self::compact_inner(lhs, rhs, provider)?));
        }
        if matches!(&self.repr, TypedTensorRepr::Adjoint(_))
            || matches!(&other.repr, TypedTensorRepr::Adjoint(_))
        {
            let provider = self.logical_space().provider();
            let (lhs_operand, lhs_data) = self.fusion_operand_and_data();
            let (rhs_operand, rhs_data) = other.fusion_operand_and_data();
            let value = match (&self.repr, &other.repr) {
                (TypedTensorRepr::Adjoint(lhs), TypedTensorRepr::Adjoint(rhs)) => {
                    tenet_tensors::oriented_fusion_inner(
                        lhs.parent.space.space().structure(),
                        tenet_tensors::FusionOperand::direct(rhs.parent.space.space()),
                        rhs.parent.materialized_dense_data(),
                        tenet_tensors::FusionOperand::direct(lhs.parent.space.space()),
                        lhs.parent.materialized_dense_data(),
                        |sector| D::from_real(provider.dim_scalar(sector)),
                    )?
                }
                _ => tenet_tensors::oriented_fusion_inner(
                    self.logical_space().space().structure(),
                    lhs_operand,
                    lhs_data,
                    rhs_operand,
                    rhs_data,
                    |sector| D::from_real(provider.dim_scalar(sector)),
                )?,
            };
            return Ok(value);
        }
        // `D::from_complex64` is `.re` for the real scalar and the identity for
        // the complex one, so this is bit-identical to the erased facade's
        // `Scalar::F64(v.re)` / `Scalar::C64(v)` dispatch, without the enum.
        Ok(D::from_complex64(weighted_inner(
            self.logical_space().provider(),
            self.logical_space().space().structure(),
            self.logical_space().space().nout(),
            self.owned_body()
                .expect("owned inner input")
                .materialized_dense_data(),
            other
                .owned_body()
                .expect("owned inner input")
                .materialized_dense_data(),
        )?))
    }

    /// `LinearAlgebra.dot` / TensorKit `dot(x, y)` — an alias for
    /// [`Self::inner`].
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
    /// they appear. The two therefore disagree for a fermionic provider by
    /// design.
    ///
    /// # Errors
    ///
    /// [`Error::InvalidArgument`] when the tensor is not an endomorphism, and
    /// [`Error::Core`] when the block structure cannot be walked.
    pub fn tr(&self) -> Result<D, Error> {
        let hom = self.logical_space().space().homspace();
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
            let provider = self.logical_space().provider();
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
        if let TypedTensorRepr::Adjoint(view) = &self.repr {
            let parent = Self {
                runtime: self.runtime.clone(),
                repr: TypedTensorRepr::Owned(Arc::clone(&view.parent)),
            };
            return Ok(FactorScalar::adjoint(parent.tr()?));
        }
        Ok(D::from_complex64(weighted_trace(
            self.logical_space().provider(),
            self.logical_space().space().structure(),
            self.logical_space().space().nout(),
            self.owned_body()
                .expect("owned trace input")
                .materialized_dense_data(),
        )?))
    }

    /// Whether the tensor equals its own adjoint within `tol`, relative to its
    /// norm (TensorKit `ishermitian`).
    ///
    /// A non-endomorphism is never Hermitian and comes back `false` rather than
    /// as an error; TensorKit throws. A predicate that can only be called after
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
        // is the last step that reached `materialized_dense_data()`.
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
        let hom = self.logical_space().space().homspace();
        hom.codomain().legs() == hom.domain().legs()
    }

    /// The single element of a rank-0 (scalar) tensor, e.g. the result of
    /// contracting every leg — TensorKit `scalar` (an empty payload reads
    /// as zero there too).
    ///
    /// Returns `D` directly: the value is the sum of the coupled payload.
    ///
    /// # Errors
    ///
    /// [`Error::InvalidArgument`] on a tensor with legs.
    pub fn scalar(&self) -> Result<D, Error> {
        if self.rank() != 0 {
            return Err(Error::InvalidArgument(format!(
                "scalar() requires a rank-0 tensor, got rank {}",
                self.rank()
            )));
        }
        // A rank-0 payload holds at most one element; summing matches the
        // erased facade and gives the empty payload its zero for free.
        let materialized = self.materialized_tensor_uncached()?;
        Ok(materialized
            .owned_body()
            .expect("uncached materialization is owned")
            .materialized_dense_data()
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
    /// TensorKit's free function.
    ///
    /// Both operands share one `D`, so mixed-dtype widening is statically
    /// unrepresentable — widen with [`Self::to_c64`] first.
    /// A lazy adjoint is read from parent storage through the oriented copy
    /// plan without publishing a receiver-sized materialization. A compact
    /// diagonal operand is materialized dense once on demand.
    ///
    /// # Complexity
    ///
    /// One output allocation and a single `O(len(self) + len(other))` copy
    /// pass over the compiled per-sector slab plan. If an oriented geometry is
    /// conservatively declined, correctness falls back to operation-local
    /// uncached materialization before retrying the owned plan.
    ///
    /// # Errors
    ///
    /// [`Error::RuleMismatch`] on differing provider identities and
    /// [`Error::RuntimeMismatch`] on differing runtimes, in that order; then
    /// [`Error::InvalidArgument`] for a multi-leg domain, mismatched codomain
    /// product spaces, or changed
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

    /// Shared route of [`Self::catdomain`] / [`Self::catcodomain`], using
    /// `cat_homspace` and `compile_cat_plan` over the typed bound space.
    fn cat(&self, other: &Self, side: CatSide) -> Result<Self, Error> {
        // Rule identity before runtime: the erased
        // `check_same_execution_world` order, minus its placement arm (no
        // devices here). Same rationale as `authority()`: separately
        // allocated providers of one rule interoperate; different identities
        // are rejected before any layout work.
        if self.logical_space().provider().rule_identity()
            != other.logical_space().provider().rule_identity()
        {
            return Err(Error::RuleMismatch);
        }
        if !self.runtime.same_runtime(&other.runtime) {
            return Err(Error::RuntimeMismatch);
        }
        let lhs = self.logical_space().space().homspace();
        let rhs = other.logical_space().space().homspace();
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
        let space = self.logical_space().derive_from_final_homspace(homspace)?;
        let (lhs_layout, lhs_data) = self.cat_operand()?;
        let (rhs_layout, rhs_data) = other.cat_operand()?;
        let Some(plan) = compile_cat_plan(
            space.space().structure(),
            space.space().nout(),
            [lhs_layout, rhs_layout],
            axis,
            side,
        )?
        else {
            let lhs = self.materialized_tensor_uncached()?;
            let rhs = other.materialized_tensor_uncached()?;
            return lhs.cat(&rhs, side);
        };
        let data = plan.execute(lhs_data, rhs_data)?;
        Ok(Self {
            runtime: self.runtime.clone(),
            repr: owned_repr(TypedTensorBody::dense(space, data)),
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
    /// the signature; widen with [`Self::to_c64`] first. A compact diagonal
    /// payload (on either side) is materialized dense first, exactly once.
    ///
    /// # Complexity
    ///
    /// One output allocation for owned dense inputs plus `O(min-prefix)`
    /// overwrites per shared block. A lazy input currently adds one
    /// operation-local logical payload; it is not published in the receiver's
    /// reusable materialization cache.
    ///
    /// # Errors
    ///
    /// [`Error::InvalidArgument`] on unequal codomain/domain ranks (TK throws
    /// its `DimensionError` for the same), [`Error::RuleMismatch`] on differing
    /// provider identities,
    /// [`Error::RuntimeMismatch`] on differing runtimes, and
    /// [`Error::InvalidArgument`] when corresponding legs differ in duality.
    pub fn absorb(&self, source: &Self) -> Result<Self, Error> {
        let destination_space = self.logical_space().space();
        let source_space = source.logical_space().space();
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
        if self.logical_space().provider().rule_identity()
            != source.logical_space().provider().rule_identity()
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
        // Lazy inputs still need a logical dense payload until absorb gains an
        // oriented copy plan, but an ordinary operation must not publish that
        // receiver-sized compatibility cache. Keep both copies request-local.
        let destination = self.materialized_tensor_uncached()?;
        let source = source.materialized_tensor_uncached()?;
        let destination_data = destination
            .owned_body()
            .expect("uncached materialization is owned")
            .materialized_dense_data();
        let source_data = source
            .owned_body()
            .expect("uncached materialization is owned")
            .materialized_dense_data();
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
            repr: owned_repr(TypedTensorBody::dense(self.logical_space().clone(), output)),
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
    /// O(Σ_c k_c), like TensorKit's `DiagonalTensorMap` twist,
    /// because `similar` preserves the diagonal storage
    /// and `twist!` only scales blocks.
    /// Otherwise: one scaled copy of the dense payload, O(len), through
    /// `scale_blocks_impl`.
    ///
    /// A lazy dense adjoint redirects through the parent with the inverse
    /// categorical phase, leaving its receiver cache cold; compact adjoints
    /// remain compact and use the spectrum arm above. There is no device arm
    /// (the payload is a host `Vec<D>` by construction). The multiplicity-free
    /// admission bound excludes `Generic` providers.
    ///
    /// # Errors
    ///
    /// [`Error::InvalidArgument`] when a leg is out of range, reported before
    /// the empty-list short-circuit. An empty `legs` returns an identical clone.
    pub fn twist(&self, legs: &[usize]) -> Result<Self, Error> {
        self.twist_with_inverse(legs, false)
    }

    /// Applies the inverse TensorKit ribbon twist on the selected legs.
    pub fn twist_inverse(&self, legs: &[usize]) -> Result<Self, Error> {
        self.twist_with_inverse(legs, true)
    }

    fn twist_with_inverse(&self, legs: &[usize], inverse: bool) -> Result<Self, Error> {
        let rank = self.rank();
        let name = if inverse { "twist_inverse" } else { "twist" };
        if let Some(&leg) = legs.iter().find(|&&leg| leg >= rank) {
            return Err(Error::InvalidArgument(format!(
                "{name} leg {leg} out of range for rank {rank}"
            )));
        }
        if legs.is_empty() {
            return Ok(self.clone());
        }
        let provider = self.logical_space().provider();
        // NoBraiding preflight (PR #620 review): before the compact arm and
        // before any θ evaluation — see `reject_unbraided_nonunit_legs`.
        reject_unbraided_nonunit_legs(
            provider,
            self.logical_space().space().homspace(),
            legs,
            name,
            true,
        )?;
        if let TypedTensorRepr::Adjoint(view) = &self.repr {
            let parent = Self {
                runtime: self.runtime.clone(),
                repr: TypedTensorRepr::Owned(Arc::clone(&view.parent)),
            };
            let axes = logical_adjoint_axes_to_parent(
                view.parent.space.space().nout(),
                view.parent.space.space().nin(),
                legs,
            );
            return parent.twist_with_inverse(&axes, !inverse)?.adjoint();
        }
        let nout = self.codomain_rank();
        if let Some(spectrum) = self.spectrum() {
            // Compact arm, mirroring the erased `scaled_by_sector` route: a
            // bond space's two legs both carry the block's coupled sector, so
            // the per-block factor collapses to θ(sector)^|legs|. The space
            // is unchanged, so the payload may stay compact.
            let sector_factor = |sector: tenet_core::SectorId| -> f64 {
                let factor = legs.iter().map(|_| provider.twist_scalar(sector)).product();
                twist_factor_with_inverse(provider, factor, inverse)
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
            return Ok(self.with_spectrum_on(self.logical_space().clone(), scaled));
        }
        if twist_is_identity_over_blocks(
            provider,
            self.logical_space().space().structure(),
            nout,
            legs,
        )? {
            return Ok(self.clone());
        }
        let mut data = self
            .owned_body()
            .expect("owned twist input")
            .materialized_dense_data()
            .to_vec();
        scale_blocks_impl(self.logical_space().space(), &mut data, &|key| match key {
            BlockKey::FusionTree(key) => twist_block_factor(provider, key, nout, legs, inverse),
            _ => 1.0,
        })?;
        Ok(Self {
            runtime: self.runtime.clone(),
            repr: owned_repr(TypedTensorBody::dense(self.logical_space().clone(), data)),
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
    /// facade narrowings as [`Self::twist`] apply: a lazy dense adjoint
    /// redirects through the parent with the inverse categorical map and
    /// stays cold; there is no device arm, and Generic fusion is dead at the
    /// admission bound.
    ///
    /// # Errors
    ///
    /// [`Error::InvalidArgument`] when a leg is out of range, before the
    /// empty-list short-circuit; empty `legs` returns an identical clone.
    /// Otherwise [`Error::Operation`] /
    /// [`Error::Core`] from the layout derivation of the toggled hom space.
    pub fn flip(&self, legs: &[usize]) -> Result<Self, Error> {
        self.flip_with_inverse(legs, false)
    }

    /// Applies the inverse TensorKit Z-isomorphism on the selected legs.
    pub fn flip_inverse(&self, legs: &[usize]) -> Result<Self, Error> {
        self.flip_with_inverse(legs, true)
    }

    fn flip_with_inverse(&self, legs: &[usize], inverse: bool) -> Result<Self, Error> {
        let rank = self.rank();
        let name = if inverse { "flip_inverse" } else { "flip" };
        if let Some(&leg) = legs.iter().find(|&&leg| leg >= rank) {
            return Err(Error::InvalidArgument(format!(
                "{name} leg {leg} out of range for rank {rank}"
            )));
        }
        if legs.is_empty() {
            return Ok(self.clone());
        }
        let hom = self.logical_space().space().homspace();
        // NoBraiding preflight (PR #620 review): flip's coefficients are
        // built from the same θ/χ — see `reject_unbraided_nonunit_legs`.
        reject_unbraided_nonunit_legs(self.logical_space().provider(), hom, legs, name, false)?;
        if let TypedTensorRepr::Adjoint(view) = &self.repr {
            let parent = Self {
                runtime: self.runtime.clone(),
                repr: TypedTensorRepr::Owned(Arc::clone(&view.parent)),
            };
            let axes = logical_adjoint_axes_to_parent(
                view.parent.space.space().nout(),
                view.parent.space.space().nin(),
                legs,
            );
            return parent.flip_with_inverse(&axes, !inverse)?.adjoint();
        }
        let nout = hom.codomain().len();
        // Sequential semantics for repeated legs, from the helper shared
        // with the erased facade (#580 PR 5).
        let (new_hom, occurrences) = flip_toggled_homspace(hom, legs);
        let space = self.logical_space().derive_from_final_homspace(new_hom)?;
        check_flip_layout_identity(
            self.logical_space().space().structure(),
            space.space().structure(),
        )?;
        let provider = self.logical_space().provider();
        let mut data = self
            .owned_body()
            .expect("owned flip input")
            .materialized_dense_data()
            .to_vec();
        scale_blocks_impl(space.space(), &mut data, &|key| match key {
            BlockKey::FusionTree(key) => {
                flip_block_factor(provider, key, nout, &occurrences, inverse)
            }
            _ => 1.0,
        })?;
        Ok(Self {
            runtime: self.runtime.clone(),
            repr: owned_repr(TypedTensorBody::dense(space, data)),
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
    /// arms; Generic fusion is dead at the admission bound (see [`Self::twist`]).
    ///
    /// # Errors
    ///
    /// [`Error::InvalidArgument`] when `position` exceeds the rank. Otherwise
    /// the layout derivation's and unit-correspondence validator's own classes.
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
    /// [`Self::insert_right_unit`]: the tenet-core hom-space transform, checked
    /// layout correspondence, then a new body over the shared (or
    /// once-materialized) payload.
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
        let provider = self.logical_space().provider();
        let source_hom = self.logical_space().space().homspace();
        let homspace = match insertion {
            UnitLegInsertion::Left { position, dual } => {
                source_hom.insert_left_unit(provider, position, dual)?
            }
            UnitLegInsertion::Right { position, dual } => {
                source_hom.insert_right_unit(provider, position, dual)?
            }
        };
        let destination = self.logical_space().derive_from_final_homspace(homspace)?;
        validate_unit_layout_correspondence_checked(
            provider,
            (source_hom, self.logical_space().space().structure()),
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
            repr: owned_repr(TypedTensorBody::with_shared_payload(destination, data)),
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
    /// not a canonical unit leg. Otherwise the layout derivation's and
    /// validator's own classes.
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
        let provider = self.logical_space().provider();
        let source_hom = self.logical_space().space().homspace();
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
        let destination = self.logical_space().derive_from_final_homspace(homspace)?;
        validate_unit_layout_correspondence_checked(
            provider,
            (
                destination.space().homspace(),
                destination.space().structure(),
            ),
            (source_hom, self.logical_space().space().structure()),
            insertion,
        )
        .map_err(map_checked_unit_layout_error)?;
        let data = self.shareable_dense_payload();
        Ok(Self {
            runtime: self.runtime.clone(),
            repr: owned_repr(TypedTensorBody::with_shared_payload(destination, data)),
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
    /// Infallible for the reason [`TypedTensorBody::materialized_dense_data`]
    /// is: the diagonal fill
    /// is total on a bond space this module built from that same spectrum.
    fn shareable_dense_payload(&self) -> Arc<TypedData<D>> {
        let materialized = self
            .materialized_tensor_uncached()
            .expect("a pre-admitted typed adjoint must materialize");
        let body = materialized
            .owned_body()
            .expect("uncached materialization is owned");
        match body.data.as_ref() {
            TypedData::Dense(_) => Arc::clone(&body.data),
            TypedData::Diagonal(spectrum) => Arc::new(TypedData::Dense(
                tenet_matrixalgebra::diagonal_bond_data(body.space.space(), spectrum, &|value| {
                    value
                })
                .expect("diagonal fill is total on the stored bond space"),
            )),
        }
    }

    /// A zero tensor on the same spaces and dtype as `self` (TensorKit
    /// `zerovector`). Dense and compact payloads are freshly initialized to
    /// exact positive zero, independently of non-finite source values. A lazy
    /// adjoint zeros its canonical parent and stays a cold lazy adjoint.
    pub fn zeros_like(&self) -> Self {
        if let TypedTensorRepr::Adjoint(view) = &self.repr {
            let parent = Self {
                runtime: self.runtime.clone(),
                repr: TypedTensorRepr::Owned(Arc::clone(&view.parent)),
            };
            return parent
                .zeros_like()
                .adjoint()
                .expect("zeroing a pre-admitted adjoint must preserve its layout");
        }
        if let Some(spectrum) = self.spectrum() {
            return self.with_spectrum(
                spectrum
                    .iter()
                    .map(|entry| tenet_matrixalgebra::SectorSpectrum {
                        sector: entry.sector,
                        values: vec![D::from_real(0.0); entry.values.len()],
                    })
                    .collect(),
            );
        }
        self.with_data(vec![
            D::from_real(0.0);
            self.owned_body()
                .expect("owned zero input")
                .materialized_dense_data()
                .len()
        ])
    }
}

// Bound-free, like the accessor impl on `GradedSpace<R>`: dtype conversion
// needs no provider algebra or new layout admission.
impl<R> TensorMap<R, f64> {
    /// Widens to a c64 tensor map, imaginary parts zero (TensorKit
    /// `Base.complex`).
    ///
    /// Element-wise on an owned payload: a dense payload is widened in
    /// place-order, while a compact spectrum maps spectrum-to-spectrum and
    /// **stays compact**. A cold lazy adjoint uses an operation-local payload
    /// before the output allocation; an already-owned input needs only its
    /// `O(stored_len)` output. The logical space is shared, not re-derived.
    ///
    /// Infallible for host storage; device transfer and conversion use their
    /// storage-specific APIs.
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
        let materialized = self
            .materialized_tensor_uncached()
            .expect("a pre-admitted typed adjoint must materialize");
        let source = materialized
            .owned_body()
            .expect("uncached materialization is owned");
        let body = match source.data.as_ref() {
            TypedData::Dense(data) => {
                TypedTensorBody::dense(source.space.clone(), data.iter().map(widen).collect())
            }
            TypedData::Diagonal(spectrum) => TypedTensorBody::diagonal(
                source.space.clone(),
                map_spectrum_dtype(spectrum, |value| num_complex::Complex64::new(value, 0.0)),
            ),
        };
        TensorMap {
            runtime: self.runtime.clone(),
            repr: owned_repr(body),
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
    /// A compact spectrum maps spectrum-to-spectrum and stays compact. A cold
    /// lazy adjoint uses an operation-local payload for the `O(stored_len)`
    /// output; the logical space is shared.
    pub fn re(&self) -> TensorMap<R, f64> {
        self.map_parts(|value| value.re)
    }

    /// The element-wise imaginary part, as an f64 tensor map on the same
    /// spaces (TensorKit `Base.imag`).
    ///
    /// A compact spectrum maps spectrum-to-spectrum and stays compact. A cold
    /// lazy adjoint uses an operation-local payload for the `O(stored_len)`
    /// output; the logical space is shared.
    pub fn im(&self) -> TensorMap<R, f64> {
        self.map_parts(|value| value.im)
    }

    /// The shared owned-input route of [`Self::re`] / [`Self::im`].
    fn map_parts(&self, part: impl Fn(num_complex::Complex64) -> f64) -> TensorMap<R, f64> {
        let materialized = self
            .materialized_tensor_uncached()
            .expect("a pre-admitted typed adjoint must materialize");
        let source = materialized
            .owned_body()
            .expect("uncached materialization is owned");
        let body = match source.data.as_ref() {
            TypedData::Dense(data) => TypedTensorBody::dense(
                source.space.clone(),
                data.iter().map(|&value| part(value)).collect(),
            ),
            TypedData::Diagonal(spectrum) => {
                TypedTensorBody::diagonal(source.space.clone(), map_spectrum_dtype(spectrum, part))
            }
        };
        TensorMap {
            runtime: self.runtime.clone(),
            repr: owned_repr(body),
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
/// need. Public semantic oracles live in `tests/typed_facade.rs`; dense-cache
/// behavior lives in `tests/typed_diagonal_allocations.rs`.
#[cfg(test)]
mod representation_gates {
    use super::*;
    use tenet_core::{product_sector, ProductFusionRuleExt};
    use tenet_core::{
        CU1FusionRule, CU1Irrep, FermionParityFusionRule, SU2FusionRule, SU2Irrep, U1FusionRule,
        U1Irrep, Z2FusionRule, Z2Irrep, ZNFusionRule,
    };
    use tenet_dense::{
        DefaultDenseExecutor, DenseBackend, DenseDotConfig, DenseError, DenseExecutor, DenseRead,
        DenseTensor, DenseWrite,
    };

    struct NonCloneHost(Vec<f64>);

    impl TensorStorage<f64> for NonCloneHost {
        fn len(&self) -> usize {
            self.0.len()
        }

        fn placement(&self) -> tenet_core::Placement {
            tenet_core::Placement::Host
        }
    }

    impl HostReadableStorage<f64> for NonCloneHost {
        fn as_slice(&self) -> &[f64] {
            &self.0
        }
    }

    #[derive(Default)]
    struct FailSecondSvd {
        inner: DefaultDenseExecutor,
        calls: usize,
    }

    #[derive(Default)]
    struct FailSecondQr {
        inner: DefaultDenseExecutor,
        calls: usize,
    }

    impl DenseExecutor for FailSecondSvd {
        fn svd(&mut self, _: DenseRead<'_>) -> Result<Vec<DenseTensor>, DenseError> {
            panic!("full SVD must use the destination API")
        }

        fn svd_into(
            &mut self,
            input: DenseRead<'_>,
            u: DenseWrite<'_>,
            s: DenseWrite<'_>,
            vt: DenseWrite<'_>,
        ) -> Result<(), DenseError> {
            self.calls += 1;
            if self.calls == 2 {
                return Err(DenseError::Backend {
                    backend: DenseBackend::Tenferro,
                    op: "svd_into",
                    message: "injected second-sector failure".to_string(),
                });
            }
            self.inner.svd_into(input, u, s, vt)
        }

        fn qr(&mut self, _: DenseRead<'_>) -> Result<Vec<DenseTensor>, DenseError> {
            panic!("test only exercises SVD")
        }

        fn eigh(&mut self, _: DenseRead<'_>) -> Result<Vec<DenseTensor>, DenseError> {
            panic!("test only exercises SVD")
        }

        fn dot_general_into(
            &mut self,
            _: DenseWrite<'_>,
            _: DenseRead<'_>,
            _: DenseRead<'_>,
            _: &DenseDotConfig,
        ) -> Result<(), DenseError> {
            panic!("test only exercises SVD")
        }
    }

    impl DenseExecutor for FailSecondQr {
        fn svd(&mut self, _: DenseRead<'_>) -> Result<Vec<DenseTensor>, DenseError> {
            panic!("test only exercises QR")
        }

        fn qr(&mut self, _: DenseRead<'_>) -> Result<Vec<DenseTensor>, DenseError> {
            panic!("QR must use the destination API")
        }

        fn qr_into(
            &mut self,
            input: DenseRead<'_>,
            q: DenseWrite<'_>,
            r: DenseWrite<'_>,
        ) -> Result<(), DenseError> {
            self.calls += 1;
            if self.calls == 2 {
                return Err(DenseError::Backend {
                    backend: DenseBackend::Tenferro,
                    op: "qr_into",
                    message: "injected second-sector failure".to_string(),
                });
            }
            self.inner.qr_into(input, q, r)
        }

        fn eigh(&mut self, _: DenseRead<'_>) -> Result<Vec<DenseTensor>, DenseError> {
            panic!("test only exercises QR")
        }

        fn dot_general_into(
            &mut self,
            _: DenseWrite<'_>,
            _: DenseRead<'_>,
            _: DenseRead<'_>,
            _: &DenseDotConfig,
        ) -> Result<(), DenseError> {
            panic!("test only exercises QR")
        }
    }

    fn owned<R, D, S>(tensor: &TensorMap<R, D, S>) -> &Arc<TypedTensorBody<R, D, S>> {
        tensor.owned_body().expect("test fixture must be owned")
    }

    fn materialized_adjoint_builds<R, D, S>(tensor: &TensorMap<R, D, S>) -> usize {
        let TypedTensorRepr::Adjoint(view) = &tensor.repr else {
            return 0;
        };
        view.materialized_body_builds
            .load(std::sync::atomic::Ordering::Relaxed)
    }

    #[cfg(feature = "cuda")]
    fn assert_cuda_tensor_matches_host<R>(
        actual: &TensorMap<R, f64>,
        expected: &TensorMap<R, f64>,
        provider: *const R,
        runtime: &crate::runtime::RuntimeIdentity,
    ) where
        R: MultiplicityFreeRigidSymbols<Scalar = f64> + CheckedFusionAlgebra + SectorCodec,
    {
        assert!(std::ptr::eq(actual.provider(), provider));
        assert!(runtime.matches(actual.runtime()));
        assert_eq!(
            actual.logical_space().space(),
            expected.logical_space().space()
        );
        assert_eq!(actual.data(), expected.data());
        assert_eq!(actual.block_count(), expected.block_count());
        for index in 0..actual.block_count() {
            let actual_block = actual.block(index).unwrap();
            let expected_block = expected.block(index).unwrap();
            assert_eq!(actual_block.key(), expected_block.key());
            assert_eq!(actual_block.offset(), expected_block.offset());
            assert_eq!(actual_block.shape(), expected_block.shape());
            assert_eq!(actual_block.strides(), expected_block.strides());
            assert_eq!(
                actual.block_fusion_trees(index).unwrap(),
                expected.block_fusion_trees(index).unwrap()
            );
        }
    }

    #[cfg(feature = "cuda")]
    fn assert_cuda_lazy_contract_orientations<R>(lhs: &TensorMap<R, f64>, rhs: &TensorMap<R, f64>)
    where
        R: MultiplicityFreeRigidSymbols<Scalar = f64> + CheckedFusionAlgebra + SectorCodec,
    {
        let lhs_axes: Vec<_> = (lhs.codomain_rank()..lhs.rank()).collect();
        let rhs_axes: Vec<_> = (0..rhs.codomain_rank()).collect();
        let output_axes: Vec<_> = (0..lhs.codomain_rank() + rhs.domain_rank()).collect();
        let expected_contract = lhs
            .contract(rhs, &lhs_axes, &rhs_axes, &output_axes)
            .unwrap();
        let expected_compose = lhs.compose(rhs).unwrap();
        let provider = lhs.provider() as *const R;
        let runtime = lhs.runtime().identity();

        for (lhs_adjoint, rhs_adjoint) in
            [(false, false), (true, false), (false, true), (true, true)]
        {
            let device_operand = |logical: &TensorMap<R, f64>, adjoint: bool| {
                if adjoint {
                    eager_adjoint_oracle(logical)
                        .to_cuda()
                        .unwrap()
                        .adjoint()
                        .unwrap()
                } else {
                    logical.to_cuda().unwrap()
                }
            };
            let lhs_device = device_operand(lhs, lhs_adjoint);
            let rhs_device = device_operand(rhs, rhs_adjoint);
            let contract = lhs_device
                .contract(&rhs_device, &lhs_axes, &rhs_axes, &output_axes)
                .unwrap()
                .to_host()
                .unwrap();
            let compose = lhs_device.compose(&rhs_device).unwrap().to_host().unwrap();

            assert_cuda_tensor_matches_host(&contract, &expected_contract, provider, &runtime);
            assert_cuda_tensor_matches_host(&compose, &expected_compose, provider, &runtime);
            assert_eq!(materialized_adjoint_builds(&lhs_device), 0);
            assert_eq!(materialized_adjoint_builds(&rhs_device), 0);
        }
    }

    fn u1_lazy_fixture() -> TensorMap<U1FusionRule, f64> {
        let runtime = Runtime::builder().dense_threads(1).build().unwrap();
        let provider = Arc::new(U1FusionRule);
        let left = GradedSpace::try_new(
            Arc::clone(&provider),
            [(U1Irrep::new(-1), 1), (U1Irrep::new(0), 2)],
            false,
        )
        .unwrap();
        let right = GradedSpace::try_new(
            Arc::clone(&provider),
            [(U1Irrep::new(0), 3), (U1Irrep::new(1), 1)],
            true,
        )
        .unwrap();
        let domain = GradedSpace::try_new(
            provider,
            [
                (U1Irrep::new(-1), 2),
                (U1Irrep::new(0), 1),
                (U1Irrep::new(1), 2),
            ],
            false,
        )
        .unwrap();
        TensorMap::from_block_fn(&runtime, [&left, &right], [&domain], |_, indices| {
            indices.iter().sum::<usize>() as f64 + 1.0
        })
        .unwrap()
    }

    #[test]
    fn storage_parameter_clone_shares_non_clone_payload() {
        let source = u1_lazy_fixture();
        let tensor: TensorMap<_, _, NonCloneHost> = TensorMap {
            runtime: source.runtime.clone(),
            repr: owned_repr(TypedTensorBody::dense(
                source.logical_space().clone(),
                NonCloneHost(source.data().to_vec()),
            )),
        };

        let twin = tensor.clone();

        assert!(Arc::ptr_eq(owned(&tensor), owned(&twin)));
        assert!(std::ptr::eq(tensor.provider(), twin.provider()));
        assert_eq!(tensor.data(), twin.data());
    }

    #[test]
    fn typed_placement_is_diagnostic_for_dense_compact_and_lazy_host_storage() {
        let source = u1_lazy_fixture();
        let diagonal = source.svd_compact().unwrap().1;
        let lazy = source.adjoint().unwrap();

        assert_eq!(source.placement(), Placement::Host);
        assert_eq!(diagonal.placement(), Placement::Host);
        assert_eq!(lazy.placement(), Placement::Host);
        assert!(owned(&diagonal).dense_cache.get().is_none());
        assert_eq!(materialized_adjoint_builds(&lazy), 0);
    }

    #[test]
    fn typed_zeros_like_is_exact_and_representation_preserving() {
        let source = u1_lazy_fixture();
        let values = [f64::NAN, f64::INFINITY, f64::NEG_INFINITY, 0.0, -0.0];
        let index = std::cell::Cell::new(0usize);
        let dense = TensorMap::from_block_fn(
            source.runtime(),
            &source.codomain(),
            &source.domain(),
            |_, _| {
                let i = index.get();
                index.set(i + 1);
                values[i % values.len()]
            },
        )
        .unwrap();
        let source_bits: Vec<_> = dense.data().iter().map(|value| value.to_bits()).collect();
        let provider = dense.provider() as *const _;
        let zero = dense.zeros_like();
        assert!(zero.data().iter().all(|value| value.to_bits() == 0));
        assert_eq!(
            dense
                .data()
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            source_bits
        );
        assert!(std::ptr::eq(zero.provider(), provider));
        assert!(zero.runtime().same_runtime(dense.runtime()));
        assert_eq!(zero.logical_space().space(), dense.logical_space().space());

        let complex = dense.to_c64();
        let complex = complex.with_data(
            complex
                .data()
                .iter()
                .enumerate()
                .map(|(i, _)| {
                    num_complex::Complex64::new(
                        values[i % values.len()],
                        values[(i + 1) % values.len()],
                    )
                })
                .collect(),
        );
        let complex_zero = complex.zeros_like();
        assert!(complex_zero
            .data()
            .iter()
            .all(|value| value.re.to_bits() == 0 && value.im.to_bits() == 0));

        let compact = source.svd_compact().unwrap().1;
        let compact = compact.with_spectrum(
            compact
                .spectrum()
                .unwrap()
                .iter()
                .map(|entry| tenet_matrixalgebra::SectorSpectrum {
                    sector: entry.sector,
                    values: (0..entry.values.len())
                        .map(|i| values[i % values.len()])
                        .collect(),
                })
                .collect(),
        );
        let compact_zero = compact.zeros_like();
        assert!(matches!(
            owned(&compact_zero).data.as_ref(),
            TypedData::Diagonal(_)
        ));
        assert!(compact_zero
            .spectrum()
            .unwrap()
            .iter()
            .flat_map(|entry| &entry.values)
            .all(|value| value.to_bits() == 0));
        assert!(owned(&compact).dense_cache.get().is_none());
        assert!(owned(&compact_zero).dense_cache.get().is_none());

        let lazy = dense.adjoint().unwrap();
        let lazy_zero = lazy.zeros_like();
        assert!(matches!(lazy_zero.repr, TypedTensorRepr::Adjoint(_)));
        assert_eq!(materialized_adjoint_builds(&lazy), 0);
        assert_eq!(materialized_adjoint_builds(&lazy_zero), 0);
        assert!(std::ptr::eq(lazy_zero.provider(), provider));

        let empty_leg =
            GradedSpace::try_new(Arc::new(U1FusionRule), [(U1Irrep::new(0), 0)], false).unwrap();
        let empty =
            TensorMap::from_block_fn(source.runtime(), [&empty_leg], [&empty_leg], |_, _| {
                f64::NAN
            })
            .unwrap();
        assert!(empty.data().is_empty());
        assert!(empty.zeros_like().data().is_empty());
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn typed_cuda_owned_metadata_validation_orders_ordinal_before_length() {
        type DeviceTensor = TensorMap<U1FusionRule, f64, CudaStorage>;
        assert!(DeviceTensor::validate_cuda_owned_metadata(
            Placement::Cuda(0),
            Placement::Cuda(0),
            7,
            7
        )
        .is_ok());
        assert_eq!(
            DeviceTensor::validate_cuda_owned_metadata(
                Placement::Cuda(1),
                Placement::Cuda(0),
                7,
                6
            )
            .unwrap_err(),
            Error::PlacementMismatch
        );
        assert!(matches!(
            DeviceTensor::validate_cuda_owned_metadata(
                Placement::Cuda(0),
                Placement::Cuda(0),
                7,
                6
            ),
            Err(Error::InvalidArgument(message)) if message.contains("payload length")
        ));
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn typed_cuda_qr_tree_route_validation_is_order_independent_and_bijective() {
        let source = u1_lazy_fixture();
        let regions = sector_regions(
            source.logical_space().space().structure(),
            source.logical_space().space().nout(),
        )
        .unwrap();
        let trees = regions
            .iter()
            .flat_map(|region| [region.row_trees(), region.col_trees()])
            .find(|trees| trees.len() > 1)
            .expect("fixture must contain a multi-tree coupled sector");
        let mut reordered = trees.to_vec();
        reordered.reverse();
        assert!(cuda_qr_tree_extents_match(trees, &reordered).unwrap());
        reordered.pop();
        assert!(!cuda_qr_tree_extents_match(trees, &reordered).unwrap());
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn typed_cuda_qr_sign_keeps_exact_zero_and_flips_tiny_negative_pivots() {
        assert_eq!(cuda_qr_diagonal_sign(0.0), 1.0);
        assert_eq!(cuda_qr_diagonal_sign(-0.0), 1.0);
        assert_eq!(cuda_qr_diagonal_sign(-1.0e-300), -1.0);
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn typed_cuda_factorizations_reject_compact_and_lazy_before_runtime_work() {
        let diagonal = u1_lazy_fixture().svd_compact().unwrap().1;
        let TypedData::Diagonal(spectrum) = owned(&diagonal).data.as_ref() else {
            unreachable!("SVD factor is compact")
        };
        let device_diagonal: TensorMap<_, f64, CudaStorage> = TensorMap {
            runtime: diagonal.runtime.clone(),
            repr: owned_repr(TypedTensorBody::new(
                diagonal.logical_space().clone(),
                TypedData::<f64, CudaStorage>::Diagonal(spectrum.clone()),
            )),
        };
        assert!(matches!(
            device_diagonal.qr_compact(),
            Err(Error::UnsupportedOnDevice(message)) if message.contains("dense CUDA storage")
        ));
        assert!(matches!(
            device_diagonal.svd_compact(),
            Err(Error::UnsupportedOnDevice(message)) if message.contains("dense CUDA storage")
        ));
        assert!(matches!(
            device_diagonal.svd_trunc(&Truncation::Full),
            Err(Error::UnsupportedOnDevice(message)) if message.contains("dense CUDA storage")
        ));
        assert!(matches!(
            device_diagonal.eigh_full(),
            Err(Error::UnsupportedOnDevice(message)) if message.contains("dense CUDA storage")
        ));
        let lazy = device_diagonal.adjoint().unwrap();
        assert!(matches!(
            lazy.qr_compact(),
            Err(Error::UnsupportedOnDevice(message)) if message.contains("lazy adjoint")
        ));
        assert!(matches!(
            lazy.svd_compact(),
            Err(Error::UnsupportedOnDevice(message)) if message.contains("lazy adjoint")
        ));
        assert!(matches!(
            lazy.svd_trunc(&Truncation::Full),
            Err(Error::UnsupportedOnDevice(message)) if message.contains("lazy adjoint")
        ));
        assert!(matches!(
            lazy.eigh_trunc(&Truncation::Full),
            Err(Error::UnsupportedOnDevice(message)) if message.contains("lazy adjoint")
        ));
    }

    #[cfg(feature = "cuda")]
    #[test]
    #[ignore = "requires a real CUDA device"]
    fn typed_cuda_eigh_full_and_trunc_match_host_without_hidden_materialization() {
        let runtime = Runtime::builder().cuda(0).dense_threads(1).build().unwrap();
        let provider = Arc::new(U1FusionRule);
        let leg = GradedSpace::try_new(
            Arc::clone(&provider),
            [(U1Irrep::new(-1), 2), (U1Irrep::new(0), 3)],
            false,
        )
        .unwrap();
        let source = TensorMap::from_block_fn(&runtime, [&leg], [&leg], |_, indices| {
            let (row, col) = (indices[0], indices[1]);
            if row == col {
                row as f64 + 1.0
            } else if row.abs_diff(col) == 1 {
                0.125
            } else {
                0.0
            }
        })
        .unwrap();
        let device = source.to_cuda().unwrap();

        let expected_full = source.eigh_full().unwrap();
        let (d_device, v_device) = device.eigh_full().unwrap();
        assert_eq!(d_device.placement(), Placement::Cuda(0));
        assert_eq!(v_device.placement(), Placement::Cuda(0));
        assert!(Arc::ptr_eq(
            v_device.logical_space().provider_arc(),
            source.logical_space().provider_arc()
        ));
        let d = d_device.to_host().unwrap();
        let v = v_device.to_host().unwrap();
        assert_typed_map_close(&d, &expected_full.0, 1.0e-10);
        assert_typed_map_close(
            &source.compose(&v).unwrap(),
            &v.compose(&d).unwrap(),
            1.0e-10,
        );

        let truncation = Truncation::rank(3);
        let expected_trunc = source.eigh_trunc(&truncation).unwrap();
        let actual_trunc = device.eigh_trunc(&truncation).unwrap();
        assert_eq!(
            actual_trunc.eigenvalues.len(),
            expected_trunc.eigenvalues.len()
        );
        for (actual, expected) in actual_trunc
            .eigenvalues
            .iter()
            .zip(&expected_trunc.eigenvalues)
        {
            assert_eq!(actual.sector, expected.sector);
            assert_eq!(actual.values.len(), expected.values.len());
            assert!(actual
                .values
                .iter()
                .zip(&expected.values)
                .all(|(actual, expected)| (actual - expected).abs() < 1.0e-10));
        }
        assert!((actual_trunc.error - expected_trunc.error).abs() < 1.0e-12);
        let d = actual_trunc.d.to_host().unwrap();
        let v = actual_trunc.v.to_host().unwrap();
        assert_eq!(
            v.logical_space().space(),
            expected_trunc.v.logical_space().space()
        );
        assert_typed_map_close(&d, &expected_trunc.d, 1.0e-10);
        assert_typed_map_close(
            &source.compose(&v).unwrap(),
            &v.compose(&d).unwrap(),
            1.0e-10,
        );
        assert_eq!(materialized_adjoint_builds(&device), 0);

        let su2_provider = Arc::new(SU2FusionRule);
        let su2_leg = GradedSpace::try_new(
            Arc::clone(&su2_provider),
            [
                (SU2Irrep::from_twice_spin(0), 2),
                (SU2Irrep::from_twice_spin(1), 2),
            ],
            false,
        )
        .unwrap();
        let su2_source =
            TensorMap::from_block_fn(&runtime, [&su2_leg], [&su2_leg], |_, indices| {
                if indices[0] == indices[1] {
                    indices[0] as f64 + 1.0
                } else {
                    0.25
                }
            })
            .unwrap();
        assert!(su2_source.block_count() >= 2);
        let su2_device = su2_source.to_cuda().unwrap();
        let (su2_d, su2_v) = su2_device.eigh_full().unwrap();
        assert!(Arc::ptr_eq(
            su2_v.logical_space().provider_arc(),
            su2_source.logical_space().provider_arc()
        ));
        let su2_d = su2_d.to_host().unwrap();
        let su2_v = su2_v.to_host().unwrap();
        assert_typed_map_close(
            &su2_source.compose(&su2_v).unwrap(),
            &su2_v.compose(&su2_d).unwrap(),
            1.0e-10,
        );

        let input_before_failure = device.to_host().unwrap();
        for failure in [("decomposition", 2), ("assembly", 2)] {
            CUDA_EIGH_FAILURE.with(|injected| injected.set(Some(failure)));
            assert!(device.eigh_full().is_err());
            CUDA_EIGH_FAILURE.with(|injected| injected.set(None));
            assert_typed_map_close(
                &device.to_host().unwrap(),
                &input_before_failure,
                f64::EPSILON,
            );
        }

        let nonhermitian =
            TensorMap::from_block_fn(&runtime, [&leg], [&leg], |_, indices| {
                match (indices[0], indices[1]) {
                    (0, 1) => 1.0,
                    _ => 0.0,
                }
            })
            .unwrap()
            .to_cuda()
            .unwrap();
        assert!(matches!(
            nonhermitian.eigh_full(),
            Err(Error::Operation(error))
                if matches!(
                    error.as_ref(),
                    tenet_tensors::OperationError::UnsupportedTensorContractScope { .. }
                )
        ));
    }

    #[cfg(feature = "cuda")]
    #[test]
    #[ignore = "requires a real CUDA device"]
    fn typed_cuda_qr_work_and_preflight_are_streamed_and_transactional() {
        let runtime = Runtime::builder().cuda(0).dense_threads(1).build().unwrap();
        let provider = Arc::new(SU2FusionRule);
        let leg = GradedSpace::try_new(
            Arc::clone(&provider),
            [
                (SU2Irrep::from_twice_spin(0), 2),
                (SU2Irrep::from_twice_spin(1), 2),
            ],
            false,
        )
        .unwrap();
        let source = TensorMap::from_block_fn(&runtime, [&leg, &leg], [&leg], |_, indices| {
            indices.iter().sum::<usize>() as f64 + 1.0
        })
        .unwrap();
        let regions = sector_regions(
            source.logical_space().space().structure(),
            source.logical_space().space().nout(),
        )
        .unwrap();
        let nonempty = regions
            .iter()
            .filter(|region| region.rows() != 0 && region.cols() != 0)
            .count();
        let diagonal_values = regions
            .iter()
            .map(|region| region.rows().min(region.cols()))
            .sum();
        let assembly_gemms = regions
            .iter()
            .map(|region| {
                region
                    .row_trees()
                    .iter()
                    .filter(|tree| tree.extent().is_ok_and(|extent| extent != 0))
                    .count()
                    + region
                        .col_trees()
                        .iter()
                        .filter(|tree| tree.extent().is_ok_and(|extent| extent != 0))
                        .count()
            })
            .sum();
        let source_device = source.to_cuda().unwrap();

        CUDA_QR_OBSERVATION.with(|observation| observation.set(Some((0, 0, 0, 0, 0, 0, 0))));
        source_device.qr_compact().unwrap();
        CUDA_QR_OBSERVATION.with(|observation| {
            assert_eq!(
                observation.get(),
                Some((
                    nonempty,
                    diagonal_values,
                    nonempty,
                    2,
                    assembly_gemms,
                    0,
                    usize::from(nonempty != 0),
                ))
            );
            observation.set(None);
        });

        let malformed_storage = {
            let state = runtime.lock();
            CudaStorage::upload(state.cuda.as_ref().unwrap(), &[]).unwrap()
        };
        let malformed = TensorMap {
            runtime: runtime.clone(),
            repr: owned_repr(TypedTensorBody::dense(
                source.logical_space().clone(),
                malformed_storage,
            )),
        };
        let sentinel = (usize::MAX, 0, 0, 0, 0, 0, 0);
        CUDA_QR_OBSERVATION.with(|observation| observation.set(Some(sentinel)));
        assert!(matches!(
            malformed.qr_compact(),
            Err(Error::InvalidArgument(message)) if message.contains("payload length")
        ));
        CUDA_QR_OBSERVATION.with(|observation| {
            assert_eq!(observation.get(), Some(sentinel));
            observation.set(None);
        });

        let stranded_storage = {
            let state = runtime.lock();
            CudaStorage::upload(state.cuda.as_ref().unwrap(), source.data()).unwrap()
        };
        let stranded = TensorMap {
            runtime: Runtime::builder().build().unwrap(),
            repr: owned_repr(TypedTensorBody::dense(
                source.logical_space().clone(),
                stranded_storage,
            )),
        };
        CUDA_QR_OBSERVATION.with(|observation| observation.set(Some(sentinel)));
        assert!(matches!(
            stranded.qr_compact(),
            Err(Error::InvalidArgument(message)) if message.contains("without a CUDA device")
        ));
        CUDA_QR_OBSERVATION.with(|observation| {
            assert_eq!(observation.get(), Some(sentinel));
            observation.set(None);
        });

        let zn3 = Arc::new(ZNFusionRule::new(3).unwrap());
        let charge0 = GradedSpace::try_new(Arc::clone(&zn3), [(zn3.irrep(0), 1)], false).unwrap();
        let charge1 = GradedSpace::try_new(Arc::clone(&zn3), [(zn3.irrep(1), 1)], false).unwrap();
        let empty: TensorMap<_, f64> =
            TensorMap::from_block_fn(&runtime, [&charge0], [&charge1], |_, _| 1.0).unwrap();
        CUDA_QR_OBSERVATION.with(|observation| observation.set(Some((0, 0, 0, 0, 0, 0, 0))));
        empty.to_cuda().unwrap().qr_compact().unwrap();
        CUDA_QR_OBSERVATION.with(|observation| {
            assert_eq!(observation.get(), Some((0, 0, 0, 2, 0, 0, 0)));
            observation.set(None);
        });
    }

    #[cfg(feature = "cuda")]
    #[test]
    #[ignore = "requires a real CUDA device"]
    fn typed_cuda_svd_work_is_streamed_and_preflight_is_transactional() {
        let runtime = Runtime::builder().cuda(0).dense_threads(1).build().unwrap();
        let provider = Arc::new(SU2FusionRule);
        let leg = GradedSpace::try_new(
            Arc::clone(&provider),
            [
                (SU2Irrep::from_twice_spin(0), 2),
                (SU2Irrep::from_twice_spin(1), 2),
            ],
            false,
        )
        .unwrap();
        let source = TensorMap::from_block_fn(&runtime, [&leg, &leg], [&leg], |_, indices| {
            indices.iter().sum::<usize>() as f64 + 1.0
        })
        .unwrap();
        let regions = sector_regions(
            source.logical_space().space().structure(),
            source.logical_space().space().nout(),
        )
        .unwrap();
        let nonempty = regions
            .iter()
            .filter(|region| region.rows() != 0 && region.cols() != 0)
            .count();
        let singular_values = regions
            .iter()
            .map(|region| region.rows().min(region.cols()))
            .sum();
        let source_device = source.to_cuda().unwrap();
        CUDA_SVD_OBSERVATION.with(|observation| observation.set(Some((0, 0, 0, 0, 0))));
        source_device.svd_compact().unwrap();
        CUDA_SVD_OBSERVATION.with(|observation| {
            assert_eq!(
                observation.get(),
                Some((nonempty, singular_values, 3, 0, usize::from(nonempty != 0),))
            );
            observation.set(None);
        });

        let malformed_storage = {
            let state = runtime.lock();
            CudaStorage::upload(state.cuda.as_ref().unwrap(), &[]).unwrap()
        };
        let malformed = TensorMap {
            runtime: runtime.clone(),
            repr: owned_repr(TypedTensorBody::dense(
                source.logical_space().clone(),
                malformed_storage,
            )),
        };
        for lazy in [false, true] {
            CUDA_SVD_OBSERVATION.with(|observation| observation.set(Some((0, 0, 0, 0, 0))));
            let rejected = if lazy {
                source_device.adjoint().unwrap().svd_compact()
            } else {
                malformed.svd_compact()
            };
            assert!(rejected.is_err());
            CUDA_SVD_OBSERVATION.with(|observation| {
                assert_eq!(observation.get(), Some((0, 0, 0, 0, 0)));
                observation.set(None);
            });
        }

        let stranded_storage = {
            let state = runtime.lock();
            CudaStorage::upload(state.cuda.as_ref().unwrap(), source.data()).unwrap()
        };
        let stranded = TensorMap {
            runtime: Runtime::builder().build().unwrap(),
            repr: owned_repr(TypedTensorBody::dense(
                source.logical_space().clone(),
                stranded_storage,
            )),
        };
        CUDA_SVD_OBSERVATION.with(|observation| observation.set(Some((0, 0, 0, 0, 0))));
        assert!(stranded.svd_compact().is_err());
        CUDA_SVD_OBSERVATION.with(|observation| {
            assert_eq!(observation.get(), Some((0, 0, 0, 0, 0)));
            observation.set(None);
        });
    }

    #[cfg(feature = "cuda")]
    #[test]
    #[ignore = "requires a real CUDA device"]
    fn typed_cuda_svd_trunc_has_two_lock_phases_and_transactional_cleanup() {
        // What: one-pass lifetime accounting, lock ordering, exact final-body
        // allocation, rank-zero semantics, and every late failure stay atomic.
        let runtime = Runtime::builder().cuda(0).dense_threads(1).build().unwrap();
        let provider = Arc::new(SU2FusionRule);
        let leg = GradedSpace::try_new(
            Arc::clone(&provider),
            [
                (SU2Irrep::from_twice_spin(0), 2),
                (SU2Irrep::from_twice_spin(1), 2),
            ],
            false,
        )
        .unwrap();
        let source = TensorMap::from_block_fn(&runtime, [&leg, &leg], [&leg], |_, indices| {
            indices.iter().sum::<usize>() as f64 + 1.0
        })
        .unwrap();
        let regions = sector_regions(
            source.logical_space().space().structure(),
            source.logical_space().space().nout(),
        )
        .unwrap();
        let nonempty = regions
            .iter()
            .filter(|region| region.rows() != 0 && region.cols() != 0)
            .count();
        let spectrum_scalars = regions
            .iter()
            .map(|region| region.rows().min(region.cols()))
            .sum();
        let raw_bytes = regions
            .iter()
            .map(|region| {
                (region.rows() + region.cols())
                    * region.rows().min(region.cols())
                    * std::mem::size_of::<f64>()
            })
            .sum();
        let route_bytes: Vec<_> = regions
            .iter()
            .filter(|region| region.rows() != 0 && region.cols() != 0)
            .map(|region| {
                (region.rows() + region.cols())
                    * region.rows().min(region.cols())
                    * std::mem::size_of::<f64>()
            })
            .collect();
        assert!(nonempty >= 2, "fixture must exercise later-route failures");
        let device = source.to_cuda().unwrap();

        CUDA_SVD_TRUNC_OBSERVATION.with(|observation| observation.set(Some((0, 0, 0, 0, 0, 0, 0))));
        CUDA_SVD_TRUNC_EVENTS.with(|events| *events.borrow_mut() = Some(Vec::new()));
        CUDA_SVD_TRUNC_FINAL_EXTENTS.with(|extents| *extents.borrow_mut() = Some(Vec::new()));
        CUDA_SVD_TRUNC_ALLOCATIONS.with(|allocations| *allocations.borrow_mut() = Some(Vec::new()));
        CUDA_SVD_TRUNC_RELEASES.with(|releases| *releases.borrow_mut() = Some(Vec::new()));
        let actual = device.svd_trunc(&Truncation::Full).unwrap();
        let extents =
            CUDA_SVD_TRUNC_FINAL_EXTENTS.with(|extents| extents.borrow().clone().unwrap());
        assert_eq!(
            extents,
            vec![
                actual.u.to_host().unwrap().data().len(),
                actual.s.to_host().unwrap().data().len(),
                actual.vh.to_host().unwrap().data().len(),
            ]
        );
        let allocations =
            CUDA_SVD_TRUNC_ALLOCATIONS.with(|allocations| allocations.borrow().clone().unwrap());
        assert_eq!(
            allocations
                .iter()
                .filter(|(kind, _)| *kind == "final")
                .map(|(_, extent)| *extent)
                .collect::<Vec<_>>(),
            extents
        );
        assert_eq!(allocations.len(), 3 + 2 * nonempty);
        assert!(allocations
            .iter()
            .all(|(kind, _)| matches!(*kind, "final" | "selector")));
        assert_eq!(
            allocations
                .iter()
                .filter(|(_, extent)| extents.contains(extent))
                .count(),
            3,
            "an operation-local scratch allocation reused a final-output extent"
        );
        let mut remaining_bytes = raw_bytes;
        let expected_releases = route_bytes
            .iter()
            .enumerate()
            .map(|(index, bytes)| {
                remaining_bytes -= bytes;
                (nonempty - index - 1, remaining_bytes)
            })
            .collect::<Vec<_>>();
        assert_eq!(
            CUDA_SVD_TRUNC_RELEASES.with(|releases| releases.borrow().clone().unwrap()),
            expected_releases
        );
        CUDA_SVD_TRUNC_OBSERVATION.with(|observation| {
            assert_eq!(
                observation.get(),
                Some((nonempty, spectrum_scalars, 3, 0, nonempty, 0, raw_bytes,))
            );
        });
        CUDA_SVD_TRUNC_EVENTS.with(|events| {
            let events = events.borrow();
            let events = events.as_ref().unwrap();
            assert!(events.iter().all(|(name, depth)| match *name {
                "decomposition" | "final_storage" | "assembly" => *depth == 1,
                _ => *depth == 0,
            }));
            let admission = events
                .iter()
                .position(|(name, _)| *name == "final_admission")
                .unwrap();
            let allocation = events
                .iter()
                .position(|(name, _)| *name == "final_storage")
                .unwrap();
            let publication = events
                .iter()
                .position(|(name, _)| *name == "publication")
                .unwrap();
            assert!(admission < allocation && allocation < publication);
        });

        CUDA_SVD_TRUNC_OBSERVATION.with(|observation| observation.set(Some((0, 0, 0, 0, 0, 0, 0))));
        CUDA_SVD_TRUNC_FINAL_EXTENTS.with(|extents| *extents.borrow_mut() = Some(Vec::new()));
        let rank_zero = device.svd_trunc(&Truncation::rank(0)).unwrap();
        assert!(rank_zero.singular_values.is_empty());
        assert_eq!(
            rank_zero.error,
            source.svd_trunc(&Truncation::rank(0)).unwrap().error
        );
        assert_eq!(
            CUDA_SVD_TRUNC_FINAL_EXTENTS.with(|extents| extents.borrow().clone().unwrap()),
            vec![0, 0, 0]
        );
        CUDA_SVD_TRUNC_OBSERVATION.with(|observation| {
            let observed = observation.get().unwrap();
            assert_eq!(
                (observed.0, observed.1, observed.2),
                (nonempty, spectrum_scalars, 3)
            );
            assert_eq!((observed.3, observed.5), (0, 0));
        });

        let second_nonempty = regions
            .iter()
            .enumerate()
            .filter(|(_, region)| region.rows() != 0 && region.cols() != 0)
            .nth(1)
            .map(|(index, _)| index)
            .unwrap()
            + 1;
        let failures = [
            ("decomposition", second_nonempty),
            ("selection", 1),
            ("admission", 1),
            ("final", 1),
            ("final", 2),
            ("final", 3),
            ("assembly", 2),
            ("right_assembly", 2),
        ];
        for failure in failures {
            CUDA_SVD_TRUNC_OBSERVATION
                .with(|observation| observation.set(Some((0, 0, 0, 0, 0, 0, 0))));
            CUDA_SVD_TRUNC_EVENTS.with(|events| *events.borrow_mut() = Some(Vec::new()));
            CUDA_SVD_TRUNC_FINAL_EXTENTS.with(|extents| *extents.borrow_mut() = Some(Vec::new()));
            CUDA_SVD_TRUNC_ALLOCATIONS
                .with(|allocations| *allocations.borrow_mut() = Some(Vec::new()));
            CUDA_SVD_TRUNC_RELEASES.with(|releases| *releases.borrow_mut() = Some(Vec::new()));
            CUDA_SVD_TRUNC_FAILURE.with(|injected| injected.set(Some(failure)));
            assert!(device.svd_trunc(&Truncation::Full).is_err());
            CUDA_SVD_TRUNC_FAILURE.with(|injected| injected.set(None));
            CUDA_SVD_TRUNC_OBSERVATION.with(|observation| {
                let observed = observation.get().unwrap();
                assert_eq!((observed.3, observed.5), (0, 0));
                if failure.0 == "decomposition" {
                    assert_eq!((observed.0, observed.4), (2, 2));
                }
            });
            assert!(CUDA_SVD_TRUNC_EVENTS.with(|events| events
                .borrow()
                .as_ref()
                .unwrap()
                .iter()
                .all(|(name, _)| *name != "publication")));
            if failure.0 == "final" {
                let attempted =
                    CUDA_SVD_TRUNC_FINAL_EXTENTS.with(|extents| extents.borrow().clone().unwrap());
                assert_eq!(
                    CUDA_SVD_TRUNC_ALLOCATIONS
                        .with(|allocations| allocations.borrow().clone().unwrap()),
                    attempted[..failure.1 - 1]
                        .iter()
                        .map(|&extent| ("final", extent))
                        .collect::<Vec<_>>()
                );
            }
            if matches!(failure.0, "assembly" | "right_assembly") {
                assert_eq!(
                    CUDA_SVD_TRUNC_RELEASES.with(|releases| releases.borrow().clone().unwrap()),
                    vec![expected_releases[0]]
                );
            }
            if failure.0 == "admission" {
                assert!(CUDA_SVD_TRUNC_EVENTS.with(|events| events
                    .borrow()
                    .as_ref()
                    .unwrap()
                    .iter()
                    .any(|(name, depth)| *name == "admission_left" && *depth == 0)));
                assert!(CUDA_SVD_TRUNC_ALLOCATIONS.with(|allocations| allocations
                    .borrow()
                    .as_ref()
                    .unwrap()
                    .is_empty()));
            }
            assert!(device.svd_trunc(&Truncation::Full).is_ok());
        }

        CUDA_SVD_TRUNC_OBSERVATION.with(|observation| observation.set(Some((0, 0, 0, 0, 0, 0, 0))));
        CUDA_SVD_TRUNC_FINAL_EXTENTS.with(|extents| *extents.borrow_mut() = Some(Vec::new()));
        let foreign_provider = Arc::new(U1FusionRule);
        let foreign_space =
            GradedSpace::try_new(foreign_provider, [(U1Irrep::new(0), 1)], false).unwrap();
        assert!(device
            .svd_trunc(&Truncation::space(foreign_space.truncspace()))
            .is_err());
        assert_eq!(
            CUDA_SVD_TRUNC_OBSERVATION.with(|observation| observation.get()),
            Some((0, 0, 0, 0, 0, 0, 0))
        );
        assert!(CUDA_SVD_TRUNC_FINAL_EXTENTS.with(|extents| extents
            .borrow()
            .as_ref()
            .unwrap()
            .is_empty()));

        CUDA_SVD_TRUNC_OBSERVATION.with(|observation| observation.set(None));
        CUDA_SVD_TRUNC_EVENTS.with(|events| *events.borrow_mut() = None);
        CUDA_SVD_TRUNC_FINAL_EXTENTS.with(|extents| *extents.borrow_mut() = None);
        CUDA_SVD_TRUNC_ALLOCATIONS.with(|allocations| *allocations.borrow_mut() = None);
        CUDA_SVD_TRUNC_RELEASES.with(|releases| *releases.borrow_mut() = None);
    }

    #[cfg(feature = "cuda")]
    #[test]
    #[ignore = "requires a real CUDA device"]
    fn typed_cuda_svd_trunc_observes_empty_mixed_and_discard_all_routes() {
        // What: structural emptiness and policy-selected emptiness create no
        // selectors, while mixed inputs decompose only their matched route.
        let runtime = Runtime::builder().cuda(0).dense_threads(1).build().unwrap();
        let provider = Arc::new(U1FusionRule);
        let mixed_rows = GradedSpace::try_new(
            Arc::clone(&provider),
            [(U1Irrep::new(0), 2), (U1Irrep::new(1), 2)],
            false,
        )
        .unwrap();
        let mixed_cols = GradedSpace::try_new(
            Arc::clone(&provider),
            [(U1Irrep::new(0), 3), (U1Irrep::new(2), 1)],
            false,
        )
        .unwrap();
        let mixed =
            TensorMap::from_block_fn(&runtime, [&mixed_rows], [&mixed_cols], |_, indices| {
                (1 + indices[0] + 2 * indices[1]) as f64
            })
            .unwrap();
        let mixed_regions = sector_regions(
            mixed.logical_space().space().structure(),
            mixed.logical_space().space().nout(),
        )
        .unwrap();
        let mixed_nonempty = mixed_regions
            .iter()
            .filter(|region| region.rows() != 0 && region.cols() != 0)
            .count();
        let mixed_scalars = mixed_regions
            .iter()
            .map(|region| region.rows().min(region.cols()))
            .sum();
        assert_eq!(mixed_nonempty, 1);
        let mixed_device = mixed.to_cuda().unwrap();

        CUDA_SVD_TRUNC_OBSERVATION.with(|observation| observation.set(Some((0, 0, 0, 0, 0, 0, 0))));
        CUDA_SVD_TRUNC_FINAL_EXTENTS.with(|extents| *extents.borrow_mut() = Some(Vec::new()));
        CUDA_SVD_TRUNC_ALLOCATIONS.with(|allocations| *allocations.borrow_mut() = Some(Vec::new()));
        let mixed_result = mixed_device.svd_trunc(&Truncation::Full).unwrap();
        let mixed_extents = vec![
            mixed_result.u.to_host().unwrap().data().len(),
            mixed_result.s.to_host().unwrap().data().len(),
            mixed_result.vh.to_host().unwrap().data().len(),
        ];
        assert_eq!(
            CUDA_SVD_TRUNC_FINAL_EXTENTS.with(|extents| extents.borrow().clone().unwrap()),
            mixed_extents
        );
        CUDA_SVD_TRUNC_OBSERVATION.with(|observation| {
            let observed = observation.get().unwrap();
            assert_eq!((observed.0, observed.1, observed.2), (1, mixed_scalars, 3));
            assert_eq!((observed.3, observed.5), (0, 0));
        });
        assert_eq!(
            CUDA_SVD_TRUNC_ALLOCATIONS.with(|allocations| allocations
                .borrow()
                .as_ref()
                .unwrap()
                .len()),
            3 + 2 * mixed_nonempty
        );

        let empty_rows =
            GradedSpace::try_new(Arc::clone(&provider), [(U1Irrep::new(4), 2)], false).unwrap();
        let empty: TensorMap<_, f64> =
            TensorMap::from_block_fn(&runtime, [&empty_rows], [&mixed_cols], |_, _| 1.0).unwrap();
        assert!(empty.data().is_empty());
        CUDA_SVD_TRUNC_OBSERVATION.with(|observation| observation.set(Some((0, 0, 0, 0, 0, 0, 0))));
        CUDA_SVD_TRUNC_FINAL_EXTENTS.with(|extents| *extents.borrow_mut() = Some(Vec::new()));
        CUDA_SVD_TRUNC_ALLOCATIONS.with(|allocations| *allocations.borrow_mut() = Some(Vec::new()));
        empty
            .to_cuda()
            .unwrap()
            .svd_trunc(&Truncation::Full)
            .unwrap();
        assert_eq!(
            CUDA_SVD_TRUNC_OBSERVATION.with(|observation| observation.get()),
            Some((0, 0, 3, 0, 0, 0, 0))
        );
        assert_eq!(
            CUDA_SVD_TRUNC_FINAL_EXTENTS.with(|extents| extents.borrow().clone().unwrap()),
            vec![0, 0, 0]
        );
        assert_eq!(
            CUDA_SVD_TRUNC_ALLOCATIONS.with(|allocations| allocations.borrow().clone().unwrap()),
            vec![("final", 0), ("final", 0), ("final", 0)]
        );

        let absent = GradedSpace::try_new(Arc::new(U1FusionRule), [(U1Irrep::new(9), 1)], false)
            .unwrap()
            .truncspace();
        for policy in [
            Truncation::space(absent.clone()),
            Truncation::rank(usize::MAX).and(Truncation::space(absent.clone())),
        ] {
            CUDA_SVD_TRUNC_OBSERVATION
                .with(|observation| observation.set(Some((0, 0, 0, 0, 0, 0, 0))));
            CUDA_SVD_TRUNC_FINAL_EXTENTS.with(|extents| *extents.borrow_mut() = Some(Vec::new()));
            CUDA_SVD_TRUNC_ALLOCATIONS
                .with(|allocations| *allocations.borrow_mut() = Some(Vec::new()));
            let result = mixed_device.svd_trunc(&policy).unwrap();
            assert!(result.singular_values.is_empty());
            CUDA_SVD_TRUNC_OBSERVATION.with(|observation| {
                let observed = observation.get().unwrap();
                assert_eq!((observed.0, observed.1, observed.2), (1, mixed_scalars, 3));
                assert_eq!((observed.3, observed.5), (0, 0));
            });
            assert_eq!(
                CUDA_SVD_TRUNC_FINAL_EXTENTS.with(|extents| extents.borrow().clone().unwrap()),
                vec![0, 0, 0]
            );
            assert_eq!(
                CUDA_SVD_TRUNC_ALLOCATIONS
                    .with(|allocations| allocations.borrow().clone().unwrap()),
                vec![("final", 0), ("final", 0), ("final", 0)]
            );
        }

        CUDA_SVD_TRUNC_OBSERVATION.with(|observation| observation.set(None));
        CUDA_SVD_TRUNC_FINAL_EXTENTS.with(|extents| *extents.borrow_mut() = None);
        CUDA_SVD_TRUNC_ALLOCATIONS.with(|allocations| *allocations.borrow_mut() = None);
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn missing_cuda_context_precedes_compact_expansion_and_lazy_materialization() {
        let source = u1_lazy_fixture();
        let diagonal = source.svd_compact().unwrap().1;
        let lazy = source.adjoint().unwrap();
        let TypedData::Diagonal(spectrum) = owned(&diagonal).data.as_ref() else {
            unreachable!("SVD factor is compact")
        };
        let mut malformed_spectrum = spectrum.clone();
        for entry in &mut malformed_spectrum {
            entry.values.clear();
        }
        let malformed_expansion = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            tenet_matrixalgebra::diagonal_bond_data(
                diagonal.logical_space().space(),
                &malformed_spectrum,
                &|value| value,
            )
        }));
        assert!(
            malformed_expansion.is_err() || matches!(malformed_expansion, Ok(Err(_))),
            "the fixture must fail if compact expansion runs"
        );
        let malformed = TensorMap {
            runtime: diagonal.runtime.clone(),
            repr: owned_repr(TypedTensorBody::diagonal(
                diagonal.logical_space().clone(),
                malformed_spectrum,
            )),
        };
        let missing_context = Error::InvalidArgument(
            "this runtime was built without a CUDA device; use Runtime::builder().cuda(device)"
                .to_string(),
        );

        assert_eq!(malformed.to_cuda().unwrap_err(), missing_context);
        assert!(matches!(lazy.to_cuda(), Err(error) if error == missing_context));
        assert!(owned(&malformed).dense_cache.get().is_none());
        assert!(owned(&diagonal).dense_cache.get().is_none());
        assert_eq!(materialized_adjoint_builds(&lazy), 0);
    }

    #[cfg(feature = "cuda")]
    #[test]
    #[ignore]
    fn typed_cuda_compact_and_lazy_roundtrips_keep_source_caches_cold() {
        let runtime = Runtime::builder().cuda(0).dense_threads(1).build().unwrap();
        let provider = Arc::new(U1FusionRule);
        let leg = GradedSpace::try_new(
            Arc::clone(&provider),
            [(U1Irrep::new(0), 2), (U1Irrep::new(1), 1)],
            false,
        )
        .unwrap();
        let source: TensorMap<_, f64> =
            TensorMap::from_block_fn(&runtime, [&leg], [&leg], |_, indices| {
                indices.iter().sum::<usize>() as f64 + 1.0
            })
            .unwrap();

        let diagonal = source.svd_compact().unwrap().1;
        let TypedData::Diagonal(spectrum) = owned(&diagonal).data.as_ref() else {
            unreachable!("SVD factor is compact")
        };
        let expected_diagonal = tenet_matrixalgebra::diagonal_bond_data(
            diagonal.logical_space().space(),
            spectrum,
            &|value| value,
        )
        .unwrap();
        let diagonal_device = diagonal.to_cuda().unwrap();
        assert_eq!(diagonal_device.placement(), Placement::Cuda(0));
        assert!(owned(&diagonal).dense_cache.get().is_none());
        let diagonal_host = diagonal_device.to_host().unwrap();
        assert!(matches!(
            owned(&diagonal_host).data.as_ref(),
            TypedData::Dense(_)
        ));
        assert_eq!(diagonal_host.data(), expected_diagonal);
        assert!(owned(&diagonal).dense_cache.get().is_none());

        let lazy = source.adjoint().unwrap();
        let expected_lazy = tenet_tensors::materialize_adjoint_data_dyn(
            source.logical_space().space(),
            lazy.logical_space().space(),
            source.data(),
        )
        .unwrap();
        let lazy_device = lazy.to_cuda().unwrap();
        assert_eq!(materialized_adjoint_builds(&lazy), 0);
        let TypedTensorRepr::Adjoint(device_view) = &lazy_device.repr else {
            unreachable!("transfer preserves the lazy view")
        };
        assert!(device_view.materialized.get().is_none());
        let expected_norm = source.norm().unwrap();
        CUDA_REDUCTION_BUFFER_OBSERVATION.with(|observation| observation.set(Some((0, 0, 0))));
        assert!(
            (lazy_device.norm().unwrap() - expected_norm).abs() <= 1e-12 * (1.0 + expected_norm)
        );
        let observed = CUDA_REDUCTION_BUFFER_OBSERVATION.with(|observation| {
            let observed = observation.get().unwrap();
            observation.set(None);
            observed
        });
        let sector_count = sector_regions(
            source.logical_space().space().structure(),
            source.logical_space().space().nout(),
        )
        .unwrap()
        .len();
        assert_eq!(observed, (1, sector_count.max(1), sector_count.max(1)));
        assert!(source.data().len() > sector_count.max(1));

        macro_rules! observed_arithmetic {
            ($expression:expr, $arithmetic:expr, $reduction:expr) => {{
                CUDA_ARITHMETIC_OBSERVATION.with(|observation| observation.set(Some((0, 0, 0))));
                CUDA_REDUCTION_BUFFER_OBSERVATION
                    .with(|observation| observation.set(Some((0, 0, 0))));
                let result = $expression;
                CUDA_ARITHMETIC_OBSERVATION.with(|observation| {
                    assert_eq!(observation.get(), Some($arithmetic));
                    observation.set(None);
                });
                CUDA_REDUCTION_BUFFER_OBSERVATION.with(|observation| {
                    assert_eq!(observation.get(), Some($reduction));
                    observation.set(None);
                });
                result
            }};
        }

        let source_device = source.to_cuda().unwrap();
        let empty_storage = {
            let state = runtime.lock();
            CudaStorage::upload(state.cuda.as_ref().unwrap(), &[]).unwrap()
        };
        let malformed_length = TensorMap {
            runtime: runtime.clone(),
            repr: owned_repr(TypedTensorBody::dense(
                source.logical_space().clone(),
                empty_storage,
            )),
        };
        let work_sentinel = (usize::MAX, usize::MAX, usize::MAX);
        CUDA_ARITHMETIC_OBSERVATION.with(|observation| observation.set(Some(work_sentinel)));
        assert!(matches!(
            malformed_length.scale(2.0),
            Err(Error::InvalidArgument(message)) if message.contains("payload length")
        ));
        CUDA_ARITHMETIC_OBSERVATION.with(|observation| {
            assert_eq!(observation.get(), Some(work_sentinel));
            observation.set(None);
        });

        observed_arithmetic!(source_device.scale(-2.0), (1, 1, 1), (0, 0, 0)).unwrap();
        observed_arithmetic!(
            source_device.add(&source_device, 2.0, -3.0),
            (1, 1, 2),
            (0, 0, 0)
        )
        .unwrap();
        observed_arithmetic!(source_device.zeros_like(), (1, 0, 0), (0, 0, 0)).unwrap();
        observed_arithmetic!(
            source_device.normalize(),
            (1, 1, 1),
            (1, sector_count.max(1), sector_count.max(1))
        )
        .unwrap();

        let lazy_scale =
            observed_arithmetic!(lazy_device.scale(-2.0), (1, 1, 1), (0, 0, 0)).unwrap();
        let lazy_add = observed_arithmetic!(
            lazy_device.add(&lazy_device, 2.0, -3.0),
            (1, 1, 2),
            (0, 0, 0)
        )
        .unwrap();
        let lazy_zero =
            observed_arithmetic!(lazy_device.zeros_like(), (1, 0, 0), (0, 0, 0)).unwrap();
        let lazy_normalized = observed_arithmetic!(
            lazy_device.normalize(),
            (1, 1, 1),
            (1, sector_count.max(1), sector_count.max(1))
        )
        .unwrap();
        for result in [&lazy_scale, &lazy_add, &lazy_zero, &lazy_normalized] {
            assert!(matches!(result.repr, TypedTensorRepr::Adjoint(_)));
            assert_eq!(materialized_adjoint_builds(result), 0);
        }
        assert_eq!(materialized_adjoint_builds(&lazy_device), 0);
        assert!(matches!(
            observed_arithmetic!(
                lazy_device.add(&source_device, 2.0, -3.0),
                (0, 0, 0),
                (0, 0, 0)
            ),
            Err(Error::UnsupportedOnDevice(_))
        ));
        assert_eq!(materialized_adjoint_builds(&lazy_device), 0);

        assert!(matches!(
            lazy_device.inner(&lazy_device),
            Err(Error::UnsupportedOnDevice(_))
        ));
        assert!(matches!(
            lazy_device.dot(&lazy_device),
            Err(Error::UnsupportedOnDevice(_))
        ));
        assert_eq!(materialized_adjoint_builds(&lazy_device), 0);
        assert!(device_view.materialized.get().is_none());

        let mut missing_context = lazy_device.clone();
        missing_context.runtime = Runtime::builder().build().unwrap();
        let preflight_sentinel = (usize::MAX, usize::MAX, usize::MAX);
        CUDA_REDUCTION_BUFFER_OBSERVATION
            .with(|observation| observation.set(Some(preflight_sentinel)));
        assert!(matches!(
            missing_context.norm(),
            Err(Error::InvalidArgument(message)) if message.contains("without a CUDA device")
        ));
        CUDA_REDUCTION_BUFFER_OBSERVATION.with(|observation| {
            assert_eq!(observation.get(), Some(preflight_sentinel));
            observation.set(None);
        });
        assert_eq!(materialized_adjoint_builds(&missing_context), 0);
        CUDA_ARITHMETIC_OBSERVATION.with(|observation| observation.set(Some(preflight_sentinel)));
        assert!(matches!(
            missing_context.zeros_like(),
            Err(Error::InvalidArgument(message)) if message.contains("without a CUDA device")
        ));
        CUDA_ARITHMETIC_OBSERVATION.with(|observation| {
            assert_eq!(observation.get(), Some(preflight_sentinel));
            observation.set(None);
        });
        let device_clone = lazy_device.clone();
        let TypedTensorRepr::Adjoint(clone_view) = &device_clone.repr else {
            unreachable!("clone preserves the lazy view")
        };
        assert!(Arc::ptr_eq(device_view, clone_view));

        let lazy_host = device_clone.to_host().unwrap();
        let TypedTensorRepr::Adjoint(host_view) = &lazy_host.repr else {
            unreachable!("roundtrip preserves the lazy view")
        };
        assert!(host_view.materialized.get().is_none());
        assert_eq!(materialized_adjoint_builds(&lazy), 0);
        assert_eq!(lazy_host.data(), expected_lazy);
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn typed_cuda_reduction_placement_validation_is_exact() {
        assert!(validate_cuda_reduction_placement(
            Placement::Cuda(0),
            Placement::Cuda(0),
            Placement::Cuda(0)
        )
        .is_ok());
        assert_eq!(
            validate_cuda_reduction_placement(
                Placement::Cuda(1),
                Placement::Cuda(0),
                Placement::Cuda(0)
            )
            .unwrap_err(),
            Error::PlacementMismatch
        );
    }

    #[cfg(feature = "cuda")]
    #[test]
    #[ignore = "requires a real CUDA device"]
    fn typed_cuda_lazy_adjoint_contract_and_compose_match_rectangular_host_oracles() {
        let runtime = Runtime::builder().cuda(0).dense_threads(1).build().unwrap();
        let provider = Arc::new(U1FusionRule);
        let leg = |degeneracy| {
            GradedSpace::try_new(
                Arc::clone(&provider),
                [(U1Irrep::new(0), degeneracy)],
                false,
            )
            .unwrap()
        };
        let (m, k, n) = (2, 3, 4);
        let lhs = TensorMap::from_block_fn(&runtime, [&leg(m)], [&leg(k)], |_, indices| {
            (indices[0] + m * indices[1]) as f64 + 1.0
        })
        .unwrap();
        let rhs = TensorMap::from_block_fn(&runtime, [&leg(k)], [&leg(n)], |_, indices| {
            (2 * indices[0] + indices[1]) as f64 + 1.0
        })
        .unwrap();
        let expected_contract = lhs.contract(&rhs, &[1], &[0], &[0, 1]).unwrap();
        let expected_compose = lhs.compose(&rhs).unwrap();

        for upload_parent_first in [false, true] {
            for (lhs_adjoint, rhs_adjoint) in
                [(false, false), (true, false), (false, true), (true, true)]
            {
                let device_operand = |logical: &TensorMap<U1FusionRule, f64>, adjoint: bool| {
                    if !adjoint {
                        return logical.to_cuda().unwrap();
                    }
                    let parent = eager_adjoint_oracle(logical);
                    if upload_parent_first {
                        let device = parent.to_cuda().unwrap().adjoint().unwrap();
                        assert_eq!(materialized_adjoint_builds(&device), 0);
                        device
                    } else {
                        let lazy = parent.adjoint().unwrap();
                        let device = lazy.to_cuda().unwrap();
                        assert_eq!(materialized_adjoint_builds(&lazy), 0);
                        device
                    }
                };
                let lhs_device = device_operand(&lhs, lhs_adjoint);
                let rhs_device = device_operand(&rhs, rhs_adjoint);
                let contracted = if upload_parent_first {
                    lhs_device
                        .contract_ordered(&rhs_device, &[1], &[0], &[0, 1])
                        .unwrap()
                } else {
                    lhs_device
                        .contract(&rhs_device, &[1], &[0], &[0, 1])
                        .unwrap()
                };
                let composed = lhs_device.compose(&rhs_device).unwrap();
                let contracted = contracted.to_host().unwrap();
                let composed = composed.to_host().unwrap();

                assert_eq!(
                    contracted.logical_space().space(),
                    expected_contract.logical_space().space()
                );
                assert_eq!(
                    composed.logical_space().space(),
                    expected_compose.logical_space().space()
                );
                assert_eq!(contracted.data(), expected_contract.data());
                assert_eq!(composed.data(), expected_compose.data());
                assert!(Arc::ptr_eq(
                    contracted.logical_space().provider_arc(),
                    lhs.logical_space().provider_arc()
                ));
                assert_eq!(materialized_adjoint_builds(&lhs_device), 0);
                assert_eq!(materialized_adjoint_builds(&rhs_device), 0);
            }
        }
    }

    #[cfg(feature = "cuda")]
    #[test]
    #[ignore = "requires a real CUDA device"]
    fn typed_cuda_lazy_adjoint_preserves_fermionic_contract_sign() {
        let runtime = Runtime::builder().cuda(0).dense_threads(1).build().unwrap();
        let provider = Arc::new(FermionParityFusionRule);
        let odd = |is_dual| {
            GradedSpace::try_new(Arc::clone(&provider), [(Z2Irrep::ODD, 1)], is_dual).unwrap()
        };
        let lhs =
            TensorMap::from_block_fn(&runtime, [&odd(false)], [&odd(true)], |_, _| 2.0).unwrap();
        let rhs =
            TensorMap::from_block_fn(&runtime, [&odd(true)], [&odd(false)], |_, _| 3.0).unwrap();

        for (lhs_adjoint, rhs_adjoint) in
            [(false, false), (true, false), (false, true), (true, true)]
        {
            let device_operand = |logical: &TensorMap<FermionParityFusionRule, f64>,
                                  adjoint: bool| {
                if adjoint {
                    eager_adjoint_oracle(logical)
                        .to_cuda()
                        .unwrap()
                        .adjoint()
                        .unwrap()
                } else {
                    logical.to_cuda().unwrap()
                }
            };
            let lhs_device = device_operand(&lhs, lhs_adjoint);
            let rhs_device = device_operand(&rhs, rhs_adjoint);
            let contract = lhs_device
                .contract(&rhs_device, &[1], &[0], &[0, 1])
                .unwrap()
                .to_host()
                .unwrap();
            let compose = lhs_device.compose(&rhs_device).unwrap().to_host().unwrap();

            assert_eq!(contract.data(), &[-6.0]);
            assert_eq!(compose.data(), &[6.0]);
            assert_eq!(materialized_adjoint_builds(&lhs_device), 0);
            assert_eq!(materialized_adjoint_builds(&rhs_device), 0);
        }
    }

    #[cfg(feature = "cuda")]
    #[test]
    #[ignore = "requires a real CUDA device"]
    fn typed_cuda_lazy_adjoint_covers_su2_rank_five_and_simple_product() {
        let runtime = Runtime::builder().cuda(0).dense_threads(1).build().unwrap();

        let su2_provider = Arc::new(SU2FusionRule);
        let su2 = GradedSpace::try_new(
            Arc::clone(&su2_provider),
            [
                (SU2Irrep::from_twice_spin(0), 1),
                (SU2Irrep::from_twice_spin(1), 2),
            ],
            false,
        )
        .unwrap();
        let su2_lhs =
            TensorMap::from_block_fn(&runtime, [&su2, &su2, &su2], [&su2, &su2], |_, indices| {
                indices.iter().sum::<usize>() as f64 + 1.0
            })
            .unwrap();
        let su2_rhs =
            TensorMap::from_block_fn(&runtime, [&su2, &su2], [&su2, &su2, &su2], |_, indices| {
                indices.iter().sum::<usize>() as f64 + 3.0
            })
            .unwrap();
        assert_cuda_lazy_contract_orientations(&su2_lhs, &su2_rhs);

        let product_provider = Arc::new(U1FusionRule.product(FermionParityFusionRule));
        let product = GradedSpace::try_new(
            Arc::clone(&product_provider),
            [
                (product_sector(U1Irrep::new(0), Z2Irrep::EVEN), 2),
                (product_sector(U1Irrep::new(1), Z2Irrep::ODD), 1),
            ],
            false,
        )
        .unwrap();
        let product_lhs =
            TensorMap::from_block_fn(&runtime, [&product], [&product], |_, indices| {
                indices.iter().sum::<usize>() as f64 + 1.0
            })
            .unwrap();
        let product_rhs =
            TensorMap::from_block_fn(&runtime, [&product], [&product], |_, indices| {
                indices.iter().sum::<usize>() as f64 + 4.0
            })
            .unwrap();
        assert_cuda_lazy_contract_orientations(&product_lhs, &product_rhs);
    }

    #[test]
    fn generic_lazy_adjoint_keeps_parent_storage_and_caches_a_host_body() {
        let source = u1_lazy_fixture();
        let parent: Arc<TypedTensorBody<_, _, NonCloneHost>> = Arc::new(TypedTensorBody::dense(
            source.logical_space().clone(),
            NonCloneHost(source.data().to_vec()),
        ));
        let logical_space = tenet_tensors::adjoint_bound_space_dyn(&parent.space).unwrap();
        let lazy = TensorMap {
            runtime: source.runtime.clone(),
            repr: TypedTensorRepr::Adjoint(Arc::new(TypedAdjointView {
                parent: Arc::clone(&parent),
                logical_space,
                materialized: OnceLock::new(),
                materialized_body_builds: std::sync::atomic::AtomicUsize::new(0),
            })),
        };

        let expected = source.adjoint().unwrap();
        assert_eq!(lazy.data(), expected.data());
        let TypedTensorRepr::Adjoint(view) = &lazy.repr else {
            unreachable!("fixture is a lazy adjoint")
        };
        let _: &OnceLock<Arc<TypedTensorBody<U1FusionRule, f64, Vec<f64>>>> = &view.materialized;
        assert!(view.materialized.get().is_some());
        assert!(Arc::ptr_eq(&parent, &view.parent));
    }

    fn su2_lazy_fixture() -> TensorMap<SU2FusionRule, f64> {
        let runtime = Runtime::builder().dense_threads(1).build().unwrap();
        let provider = Arc::new(SU2FusionRule);
        let leg = GradedSpace::try_new(
            provider,
            [
                (SU2Irrep::from_twice_spin(0), 2),
                (SU2Irrep::from_twice_spin(1), 1),
            ],
            false,
        )
        .unwrap();
        TensorMap::from_block_fn(&runtime, [&leg, &leg], [&leg], |_, indices| {
            indices.iter().sum::<usize>() as f64 + 1.0
        })
        .unwrap()
    }

    fn u1_matrix_fixture(
        codomain: impl IntoIterator<Item = (i32, usize)>,
        domain: impl IntoIterator<Item = (i32, usize)>,
    ) -> TensorMap<U1FusionRule, f64> {
        let runtime = Runtime::builder().dense_threads(1).build().unwrap();
        let provider = Arc::new(U1FusionRule);
        let codomain = GradedSpace::try_new(
            Arc::clone(&provider),
            codomain
                .into_iter()
                .map(|(charge, degeneracy)| (U1Irrep::new(charge), degeneracy)),
            false,
        )
        .unwrap();
        let domain = GradedSpace::try_new(
            provider,
            domain
                .into_iter()
                .map(|(charge, degeneracy)| (U1Irrep::new(charge), degeneracy)),
            false,
        )
        .unwrap();
        TensorMap::from_block_fn(&runtime, [&codomain], [&domain], |_, indices| {
            (indices.iter().sum::<usize>() + 1) as f64
        })
        .unwrap()
    }

    fn genuinely_complex<R>(source: &TensorMap<R, f64>) -> TensorMap<R, num_complex::Complex64> {
        TensorMap {
            runtime: source.runtime.clone(),
            repr: owned_repr(TypedTensorBody::dense(
                source.logical_space().clone(),
                source
                    .data()
                    .iter()
                    .enumerate()
                    .map(|(index, &value)| {
                        num_complex::Complex64::new(value, (index + 1) as f64 / 7.0)
                    })
                    .collect(),
            )),
        }
    }

    fn assert_lazy_involution<R, D>(source: &TensorMap<R, D>)
    where
        R: MultiplicityFreeRigidSymbols<Scalar = f64> + CheckedFusionAlgebra + SectorCodec,
        D: TensorScalar,
    {
        let adjoint = source.adjoint().unwrap();
        let TypedTensorRepr::Adjoint(view) = &adjoint.repr else {
            panic!("dense adjoint must be lazy");
        };
        assert!(Arc::ptr_eq(&view.parent, owned(source)));
        assert!(Arc::ptr_eq(
            view.parent.space.provider_arc(),
            view.logical_space.provider_arc()
        ));
        assert!(std::ptr::eq(adjoint.provider(), source.provider()));
        assert_eq!(materialized_adjoint_builds(&adjoint), 0);
        assert!(view.materialized.get().is_none());

        let clone = adjoint.clone();
        let TypedTensorRepr::Adjoint(clone_view) = &clone.repr else {
            unreachable!()
        };
        assert!(Arc::ptr_eq(view, clone_view));

        let restored = adjoint.adjoint().unwrap();
        assert!(Arc::ptr_eq(owned(source), owned(&restored)));
        assert_eq!(source.data().as_ptr(), restored.data().as_ptr());
        assert_eq!(materialized_adjoint_builds(&adjoint), 0);
    }

    #[test]
    fn lazy_adjoint_representation_and_involution_cover_unique_simple_and_both_dtypes() {
        let u1_f64 = u1_lazy_fixture();
        let u1_c64 = u1_f64.to_c64();
        let su2_f64 = su2_lazy_fixture();
        let su2_c64 = su2_f64.to_c64();

        assert_lazy_involution(&u1_f64);
        assert_lazy_involution(&u1_c64);
        assert_lazy_involution(&su2_f64);
        assert_lazy_involution(&su2_c64);
    }

    #[test]
    fn lazy_adjoint_metadata_is_logical_and_cold() {
        let source = u1_lazy_fixture();
        let adjoint = source.adjoint().unwrap();
        assert_eq!((source.codomain_rank(), source.domain_rank()), (2, 1));
        assert_eq!((adjoint.codomain_rank(), adjoint.domain_rank()), (1, 2));
        assert_eq!(adjoint.rank(), 3);
        let signature = |space: &GradedSpace<U1FusionRule>| {
            (
                space.sectors().unwrap(),
                space.degeneracies().to_vec(),
                space.is_dual(),
            )
        };
        let source_codomain = source.codomain_spaces();
        let source_domain = source.domain_spaces();
        let adjoint_codomain = adjoint.codomain_spaces();
        let adjoint_domain = adjoint.domain_spaces();
        assert_eq!(
            signature(&adjoint_codomain[0]),
            signature(&source_domain[0])
        );
        assert_eq!(
            signature(&adjoint_domain[0]),
            signature(&source_codomain[0])
        );
        assert_eq!(
            signature(&adjoint_domain[1]),
            signature(&source_codomain[1])
        );
        let source_dims = source.leg_dims().unwrap();
        assert_eq!(
            adjoint.leg_dims().unwrap(),
            [source_dims[2], source_dims[0], source_dims[1]]
        );
        assert_eq!(adjoint.block_count(), source.block_count());
        let expected = tenet_tensors::adjoint_bound_space_dyn(source.logical_space()).unwrap();
        for index in 0..adjoint.block_count() {
            let actual = adjoint.block(index).unwrap();
            let expected_block = expected.space().structure().block(index).unwrap();
            assert_eq!(actual.key(), expected_block.key());
            assert_eq!(actual.shape(), expected_block.shape());
            assert_eq!(actual.strides(), expected_block.strides());
            assert_eq!(actual.offset(), expected_block.offset());
            assert_eq!(
                adjoint.block_fusion_trees(index).unwrap(),
                decode_block_fusion_trees(adjoint.provider(), expected_block.key()).unwrap()
            );
        }
        assert_eq!(adjoint.logical_space().space(), expected.space());
        assert!(!format!("{adjoint:?}").is_empty());
        assert_eq!(materialized_adjoint_builds(&adjoint), 0);
        let TypedTensorRepr::Adjoint(view) = &adjoint.repr else {
            unreachable!()
        };
        assert!(view.materialized.get().is_none());
    }

    #[test]
    fn cloned_adjoint_materializes_once_across_threads() {
        let source = u1_lazy_fixture().to_c64();
        let (_, expected) =
            tenet_tensors::adjoint_bound_dyn(source.logical_space(), source.data()).unwrap();
        let adjoint = source.adjoint().unwrap();
        let expected_len = adjoint.logical_space().space().required_len().unwrap();
        let barrier = Arc::new(std::sync::Barrier::new(8));
        let threads: Vec<_> = (0..8)
            .map(|_| {
                let adjoint = adjoint.clone();
                let barrier = Arc::clone(&barrier);
                std::thread::spawn(move || {
                    barrier.wait();
                    let data = adjoint.data();
                    (data.as_ptr() as usize, data.to_vec())
                })
            })
            .collect();
        let outputs: Vec<_> = threads
            .into_iter()
            .map(|thread| thread.join().unwrap())
            .collect();
        assert!(outputs
            .iter()
            .all(|(pointer, data)| *pointer == outputs[0].0 && data.len() == expected_len));
        assert_eq!(outputs[0].1, expected);
        assert_eq!(materialized_adjoint_builds(&adjoint), 1);
    }

    #[test]
    fn svd_vals_reads_the_parent_without_materializing_the_adjoint() {
        // What: values-only SVD preserves typed sector spectra across cold,
        // repeated, cloned, and concurrent lazy-adjoint reads.
        macro_rules! assert_fixture {
            ($source:expr) => {{
                let source = $source;
                let expected = source.svd_vals().unwrap();
                let lazy = source.adjoint().unwrap();
                assert_eq!(lazy.svd_vals().unwrap(), expected);
                assert_eq!(lazy.svd_vals().unwrap(), expected);
                assert_eq!(lazy.clone().svd_vals().unwrap(), expected);
                assert_eq!(materialized_adjoint_builds(&lazy), 0);
                let TypedTensorRepr::Adjoint(view) = &lazy.repr else {
                    unreachable!()
                };
                assert!(view.materialized.get().is_none());
                assert!(Arc::ptr_eq(
                    view.logical_space.provider_arc(),
                    source.logical_space().provider_arc()
                ));
            }};
        }
        assert_fixture!(u1_lazy_fixture());
        assert_fixture!(u1_lazy_fixture().to_c64());
        assert_fixture!(su2_lazy_fixture());
        assert_fixture!(su2_lazy_fixture().to_c64());

        let source = u1_lazy_fixture().to_c64();
        let expected = source.svd_vals().unwrap();
        let lazy = source.adjoint().unwrap();
        let threads: Vec<_> = (0..4)
            .map(|_| {
                let lazy = lazy.clone();
                std::thread::spawn(move || lazy.svd_vals().unwrap())
            })
            .collect();
        for thread in threads {
            assert_eq!(thread.join().unwrap(), expected);
        }
        assert_eq!(materialized_adjoint_builds(&lazy), 0);
        let TypedTensorRepr::Adjoint(view) = &lazy.repr else {
            unreachable!()
        };
        assert!(view.materialized.get().is_none());
    }

    fn assert_compact_svd_reads_parent<R, D>(source: &TensorMap<R, D>)
    where
        R: MultiplicityFreeRigidSymbols<Scalar = f64> + CheckedFusionAlgebra + SectorCodec,
        D: TensorScalar + core::fmt::Debug,
    {
        let eager = eager_adjoint_oracle(source);
        let lazy = source.adjoint().unwrap();
        let actual = lazy.svd_compact().unwrap();
        let expected = eager.svd_compact().unwrap();

        assert_eq!(materialized_adjoint_builds(&lazy), 0);
        let TypedTensorRepr::Adjoint(view) = &lazy.repr else {
            unreachable!()
        };
        assert!(view.materialized.get().is_none());
        for (actual, expected) in [
            (&actual.0, &expected.0),
            (&actual.1, &expected.1),
            (&actual.2, &expected.2),
        ] {
            assert_eq!(
                actual.logical_space().space(),
                expected.logical_space().space()
            );
            assert!(Arc::ptr_eq(
                actual.logical_space().provider_arc(),
                source.logical_space().provider_arc()
            ));
            assert!(actual
                .data()
                .iter()
                .zip(expected.data())
                .all(|(&left, &right)| {
                    (left.widen_complex() - right.widen_complex()).norm() < 1e-12
                }));
        }
        assert!(actual.0.is_isometric(1e-12).unwrap());
        let rebuilt = actual
            .0
            .compose(&actual.1)
            .unwrap()
            .compose(&actual.2)
            .unwrap();
        assert!(rebuilt
            .data()
            .iter()
            .zip(eager.data())
            .all(|(&left, &right)| {
                (left.widen_complex() - right.widen_complex()).norm() < 1e-12
            }));
        assert_eq!(materialized_adjoint_builds(&lazy), 0);
        assert!(view.materialized.get().is_none());
    }

    #[test]
    fn compact_svd_reads_the_parent_without_materializing_the_adjoint() {
        // What: typed compact factors keep the eager logical-adjoint semantics,
        // provider authority, final gauge, and reconstruction without an input copy.
        let u1 = u1_lazy_fixture();
        let su2 = su2_lazy_fixture();
        assert_compact_svd_reads_parent(&u1);
        assert_compact_svd_reads_parent(&genuinely_complex(&u1));
        assert_compact_svd_reads_parent(&su2);
        assert_compact_svd_reads_parent(&genuinely_complex(&su2));
    }

    fn assert_full_svd_reads_parent<R, D>(source: &TensorMap<R, D>, compare_factor_bytes: bool)
    where
        R: MultiplicityFreeRigidSymbols<Scalar = f64> + CheckedFusionAlgebra + SectorCodec,
        D: TensorScalar + core::fmt::Debug,
    {
        let eager = eager_adjoint_oracle(source);
        let lazy = source.adjoint().unwrap();
        let actual = lazy.svd_full().unwrap();
        let expected = eager.svd_full().unwrap();

        let TypedTensorRepr::Adjoint(view) = &lazy.repr else {
            unreachable!()
        };
        for (actual, expected) in [
            (&actual.0, &expected.0),
            (&actual.1, &expected.1),
            (&actual.2, &expected.2),
        ] {
            assert_eq!(
                actual.logical_space().space(),
                expected.logical_space().space()
            );
            assert!(Arc::ptr_eq(
                actual.logical_space().provider_arc(),
                source.logical_space().provider_arc()
            ));
        }
        assert!(actual
            .1
            .data()
            .iter()
            .zip(expected.1.data())
            .all(|(&left, &right)| {
                (left.widen_complex() - right.widen_complex()).norm() < 1e-12
            }));
        if compare_factor_bytes {
            for (actual, expected) in [(&actual.0, &expected.0), (&actual.2, &expected.2)] {
                assert!(actual
                    .data()
                    .iter()
                    .zip(expected.data())
                    .all(|(&left, &right)| {
                        (left.widen_complex() - right.widen_complex()).norm() < 1e-12
                    }));
            }
        }
        assert!(actual.0.is_isometric(1e-12).unwrap());
        assert!(actual.2.is_isometric(1e-12).unwrap());
        let rebuilt = actual
            .0
            .compose(&actual.1)
            .unwrap()
            .compose(&actual.2)
            .unwrap();
        assert!(rebuilt
            .data()
            .iter()
            .zip(eager.data())
            .all(|(&left, &right)| {
                (left.widen_complex() - right.widen_complex()).norm() < 1e-12
            }));
        assert_eq!(materialized_adjoint_builds(&lazy), 0);
        assert!(view.materialized.get().is_none());
    }

    #[test]
    fn full_svd_adjoint_rectangular_matched_matches_materialized_oracle() {
        let matched = u1_matrix_fixture([(0, 2)], [(0, 3)]);
        assert_full_svd_reads_parent(&matched, true);
        assert_full_svd_reads_parent(&genuinely_complex(&matched), true);
    }

    #[test]
    fn full_svd_adjoint_unmatched_row_only_matches_materialized_oracle() {
        let source = u1_matrix_fixture([(0, 2), (1, 1)], [(0, 3)]);
        assert_full_svd_reads_parent(&source, false);
    }

    #[test]
    fn full_svd_adjoint_unmatched_column_only_matches_materialized_oracle() {
        let source = u1_matrix_fixture([(0, 2)], [(0, 3), (1, 1)]);
        assert_full_svd_reads_parent(&source, false);
    }

    #[test]
    fn full_svd_adjoint_disjoint_matches_materialized_oracle() {
        let source = u1_matrix_fixture([(1, 2)], [(0, 3)]);
        assert_full_svd_reads_parent(&source, false);
    }

    #[test]
    fn full_svd_adjoint_multitree_matches_materialized_oracle() {
        let multitree = su2_lazy_fixture();
        assert_full_svd_reads_parent(&multitree, false);
        assert_full_svd_reads_parent(&genuinely_complex(&multitree), false);
    }

    #[test]
    fn full_svd_late_failure_does_not_publish_the_adjoint_cache() {
        let runtime = Runtime::builder()
            .with_dense_executor(Box::new(FailSecondSvd::default()))
            .build()
            .unwrap();
        let provider = Arc::new(U1FusionRule);
        let leg = GradedSpace::try_new(
            provider,
            [(U1Irrep::new(0), 2), (U1Irrep::new(1), 2)],
            false,
        )
        .unwrap();
        let source = TensorMap::from_block_fn(&runtime, [&leg], [&leg], |_, indices| {
            (indices.iter().sum::<usize>() + 1) as f64
        })
        .unwrap();
        let before = source.data().to_vec();
        let lazy = source.adjoint().unwrap();

        assert!(matches!(lazy.svd_full(), Err(Error::Operation(_))));
        assert_eq!(source.data(), before);
        assert_eq!(materialized_adjoint_builds(&lazy), 0);
        let TypedTensorRepr::Adjoint(view) = &lazy.repr else {
            unreachable!()
        };
        assert!(view.materialized.get().is_none());
    }

    fn assert_null_redirect<R, D>(source: &TensorMap<R, D>)
    where
        R: MultiplicityFreeRigidSymbols<Scalar = f64> + CheckedFusionAlgebra + SectorCodec,
        D: TensorScalar + core::fmt::Debug,
    {
        let target = eager_adjoint_oracle(source);
        let lazy = source.adjoint().unwrap();
        for (actual, expected, left) in [
            (lazy.left_null().unwrap(), target.left_null().unwrap(), true),
            (
                lazy.right_null().unwrap(),
                target.right_null().unwrap(),
                false,
            ),
        ] {
            assert!(actual.owned_body().is_some());
            assert_eq!(
                actual.logical_space().space(),
                expected.logical_space().space()
            );
            assert!(Arc::ptr_eq(
                actual.logical_space().provider_arc(),
                source.logical_space().provider_arc()
            ));
            let actual_projector = if left {
                actual.compose(&actual.adjoint().unwrap()).unwrap()
            } else {
                actual.adjoint().unwrap().compose(&actual).unwrap()
            };
            let expected_projector = if left {
                expected.compose(&expected.adjoint().unwrap()).unwrap()
            } else {
                expected.adjoint().unwrap().compose(&expected).unwrap()
            };
            assert!(actual_projector
                .data()
                .iter()
                .zip(expected_projector.data())
                .all(|(&actual, &expected)| {
                    (actual.widen_complex() - expected.widen_complex()).norm() < 1e-11
                }));
            let residual = if left {
                actual.adjoint().unwrap().compose(&target).unwrap()
            } else {
                target.compose(&actual.adjoint().unwrap()).unwrap()
            };
            assert!(residual.norm().unwrap() < 1e-10 * (1.0 + target.norm().unwrap()));
            assert!(if left {
                actual.is_isometric(1e-11).unwrap()
            } else {
                actual.adjoint().unwrap().is_isometric(1e-11).unwrap()
            });
            let _ = actual.data();
        }
        assert_eq!(materialized_adjoint_builds(&lazy), 0);
        let TypedTensorRepr::Adjoint(view) = &lazy.repr else {
            unreachable!()
        };
        assert!(view.materialized.get().is_none());
    }

    #[test]
    fn null_spaces_redirect_through_the_parent_without_materializing_the_adjoint() {
        let fixtures = [
            u1_matrix_fixture([(0, 3)], [(0, 2)]),
            u1_matrix_fixture([(0, 3), (1, 2)], [(0, 2)]),
            u1_matrix_fixture([(0, 3)], [(0, 2), (1, 2)]),
            u1_matrix_fixture([(1, 2)], [(0, 3)]),
            u1_matrix_fixture([(0, 3)], [(0, 3)]),
        ];
        for source in &fixtures {
            assert_null_redirect(source);
            assert_null_redirect(&genuinely_complex(source));
        }
        let multitree = su2_lazy_fixture();
        assert_null_redirect(&multitree);
        assert_null_redirect(&genuinely_complex(&multitree));
    }

    fn assert_null_late_failure(left: bool) {
        let runtime = Runtime::builder()
            .with_dense_executor(Box::new(FailSecondSvd::default()))
            .build()
            .unwrap();
        let provider = Arc::new(U1FusionRule);
        let leg = GradedSpace::try_new(
            provider,
            [(U1Irrep::new(0), 2), (U1Irrep::new(1), 2)],
            false,
        )
        .unwrap();
        let source = TensorMap::from_block_fn(&runtime, [&leg], [&leg], |_, indices| {
            (indices.iter().sum::<usize>() + 1) as f64
        })
        .unwrap();
        let before = source.data().to_vec();
        let lazy = source.adjoint().unwrap();

        let result = if left {
            lazy.left_null()
        } else {
            lazy.right_null()
        };
        assert!(matches!(result, Err(Error::Operation(_))));
        assert_eq!(source.data(), before);
        assert_eq!(materialized_adjoint_builds(&lazy), 0);
        let TypedTensorRepr::Adjoint(view) = &lazy.repr else {
            unreachable!()
        };
        assert!(view.materialized.get().is_none());
    }

    #[test]
    fn null_space_late_failure_leaves_parent_and_adjoint_cache_unchanged() {
        assert_null_late_failure(true);
        assert_null_late_failure(false);
    }

    fn assert_polar_redirect<R, D>(source: &TensorMap<R, D>, left: bool)
    where
        R: MultiplicityFreeRigidSymbols<Scalar = f64> + CheckedFusionAlgebra + SectorCodec,
        D: TensorScalar + core::fmt::Debug,
    {
        let target = eager_adjoint_oracle(source);
        let lazy = source.adjoint().unwrap();
        let actual = if left {
            lazy.left_polar().unwrap()
        } else {
            lazy.right_polar().unwrap()
        };
        let expected = if left {
            target.left_polar().unwrap()
        } else {
            target.right_polar().unwrap()
        };
        assert_polar_factors(source, &target, &actual, &expected, left);
        assert_eq!(materialized_adjoint_builds(&lazy), 0);
        let TypedTensorRepr::Adjoint(view) = &lazy.repr else {
            unreachable!()
        };
        assert!(view.materialized.get().is_none());
    }

    fn assert_typed_map_close<R, D>(
        actual: &TensorMap<R, D>,
        expected: &TensorMap<R, D>,
        tolerance: f64,
    ) where
        R: MultiplicityFreeRigidSymbols<Scalar = f64> + CheckedFusionAlgebra + SectorCodec,
        D: TensorScalar,
    {
        assert_eq!(
            actual.logical_space().space(),
            expected.logical_space().space()
        );
        assert!(actual
            .data()
            .iter()
            .zip(expected.data())
            .all(|(&actual, &expected)| {
                (actual.widen_complex() - expected.widen_complex()).norm() < tolerance
            }));
    }

    fn assert_eigh_uses_a_cold_logical_copy(source: &TensorMap<U1FusionRule, f64>) {
        let eager = eager_adjoint_oracle(source);
        let expected_vals = eager.eigh_vals().unwrap();
        let expected_full = eager.eigh_full().unwrap();
        let expected_trunc = eager.eigh_trunc(&Truncation::rank(1)).unwrap();
        let parent_body = Arc::clone(owned(source));
        let parent_data = Arc::clone(&parent_body.data);
        let lazy = source.adjoint().unwrap();

        for _ in 0..2 {
            assert_eq!(lazy.clone().eigh_vals().unwrap(), expected_vals);
            let full = lazy.clone().eigh_full().unwrap();
            assert_eq!(full.0.data(), expected_full.0.data());
            assert_eq!(full.1.data(), expected_full.1.data());
            let trunc = lazy.clone().eigh_trunc(&Truncation::rank(1)).unwrap();
            assert_eq!(trunc.eigenvalues, expected_trunc.eigenvalues);
            assert_eq!(trunc.error, expected_trunc.error);
            assert_eq!(trunc.d.data(), expected_trunc.d.data());
            assert_eq!(trunc.v.data(), expected_trunc.v.data());
            for output in [&full.0, &full.1, &trunc.d, &trunc.v] {
                assert!(output.owned_body().is_some());
                assert!(Arc::ptr_eq(
                    output.logical_space().provider_arc(),
                    source.logical_space().provider_arc()
                ));
            }
        }

        let calls = (0..4)
            .map(|_| {
                let clone = lazy.clone();
                std::thread::spawn(move || {
                    let vals = clone.eigh_vals().unwrap();
                    let full = clone.eigh_full().unwrap();
                    let trunc = clone.eigh_trunc(&Truncation::rank(1)).unwrap();
                    (vals, full.0.data().to_vec(), trunc.error)
                })
            })
            .collect::<Vec<_>>();
        for call in calls {
            let (vals, diagonal, error) = call.join().unwrap();
            assert_eq!(vals, expected_vals);
            assert_eq!(diagonal, expected_full.0.data());
            assert_eq!(error, expected_trunc.error);
        }
        assert!(Arc::ptr_eq(owned(source), &parent_body));
        assert!(Arc::ptr_eq(&owned(source).data, &parent_data));
        assert_eq!(materialized_adjoint_builds(&lazy), 0);
        let TypedTensorRepr::Adjoint(view) = &lazy.repr else {
            unreachable!()
        };
        assert!(view.materialized.get().is_none());
    }

    #[test]
    fn eigh_dense_lazy_near_hermitian_uses_logical_triangle_and_stays_cold() {
        let runtime = Runtime::builder().dense_threads(1).build().unwrap();
        let provider = Arc::new(U1FusionRule);
        let leg = GradedSpace::try_new(provider, [(U1Irrep::new(0), 2)], false).unwrap();
        let source = TensorMap::from_block_fn(&runtime, [&leg], [&leg], |_, indices| {
            match (indices[0], indices[1]) {
                (0, 0) | (1, 1) => 1.0,
                (0, 1) => 4.0e-15,
                _ => 0.0,
            }
        })
        .unwrap();
        let logical = eager_adjoint_oracle(&source);
        let logical_vals = logical.eigh_vals().unwrap();
        let parent_vals = source.eigh_vals().unwrap();
        assert!(logical_vals[0]
            .values
            .iter()
            .zip(&parent_vals[0].values)
            .any(|(logical, parent)| (logical - parent).abs() > 1.0e-15));
        let logical_trunc = logical.eigh_trunc(&Truncation::rank(1)).unwrap();
        let parent_trunc = source.eigh_trunc(&Truncation::rank(1)).unwrap();
        assert_ne!(logical_trunc.eigenvalues, parent_trunc.eigenvalues);
        assert_ne!(logical_trunc.error, parent_trunc.error);

        assert_eigh_uses_a_cold_logical_copy(&source);
    }

    #[test]
    fn eigh_dense_lazy_complex_orientation_and_failures_match_logical_oracles() {
        let runtime = Runtime::builder().dense_threads(1).build().unwrap();
        let provider = Arc::new(U1FusionRule);
        let leg = GradedSpace::try_new(provider, [(U1Irrep::new(0), 2)], false).unwrap();
        let hermitian = TensorMap::from_block_fn(&runtime, [&leg], [&leg], |_, indices| {
            match (indices[0], indices[1]) {
                (0, 0) => num_complex::Complex64::new(2.0, 0.0),
                (1, 1) => num_complex::Complex64::new(3.0, 0.0),
                (0, 1) => num_complex::Complex64::new(0.0, 1.0),
                (1, 0) => num_complex::Complex64::new(0.0, -1.0),
                _ => unreachable!(),
            }
        })
        .unwrap();
        let eager = eager_adjoint_oracle(&hermitian);
        let expected = eager.eigh_full().unwrap();
        let lazy = hermitian.adjoint().unwrap();
        let actual = lazy.eigh_full().unwrap();
        assert_eq!(actual.0.data(), expected.0.data());
        assert_eq!(actual.1.data(), expected.1.data());
        let reconstructed = actual
            .1
            .compose(&actual.0)
            .unwrap()
            .compose(&actual.1.adjoint().unwrap())
            .unwrap();
        assert_typed_map_close(&reconstructed, &eager, 1.0e-12);
        assert_eq!(materialized_adjoint_builds(&lazy), 0);
        let TypedTensorRepr::Adjoint(view) = &lazy.repr else {
            unreachable!()
        };
        assert!(view.materialized.get().is_none());

        let nonhermitian =
            TensorMap::from_block_fn(&runtime, [&leg], [&leg], |_, indices| {
                match (indices[0], indices[1]) {
                    (0, 0) => 1.0,
                    (1, 1) => 2.0,
                    (0, 1) => 1.0,
                    _ => 0.0,
                }
            })
            .unwrap();
        let eager = eager_adjoint_oracle(&nonhermitian);
        let expected = [
            eager.eigh_vals().unwrap_err().to_string(),
            eager.eigh_full().unwrap_err().to_string(),
            eager
                .eigh_trunc(&Truncation::rank(1))
                .unwrap_err()
                .to_string(),
        ];
        let lazy = nonhermitian.adjoint().unwrap();
        for _ in 0..2 {
            assert_eq!(lazy.eigh_vals().unwrap_err().to_string(), expected[0]);
            assert_eq!(lazy.eigh_full().unwrap_err().to_string(), expected[1]);
            assert_eq!(
                lazy.eigh_trunc(&Truncation::rank(1))
                    .unwrap_err()
                    .to_string(),
                expected[2]
            );
        }
        assert_eq!(materialized_adjoint_builds(&lazy), 0);
        let TypedTensorRepr::Adjoint(view) = &lazy.repr else {
            unreachable!()
        };
        assert!(view.materialized.get().is_none());
    }

    fn assert_eig_uses_a_cold_logical_copy(source: &TensorMap<U1FusionRule, f64>) {
        let eager = eager_adjoint_oracle(source);
        let expected_vals = eager.eig_vals().unwrap();
        let expected_full = eager.eig_full().unwrap();
        let expected_trunc = eager.eig_trunc(&Truncation::rank(1)).unwrap();
        let parent_body = Arc::clone(owned(source));
        let parent_data = Arc::clone(&parent_body.data);
        let lazy = source.adjoint().unwrap();

        for _ in 0..2 {
            assert_eq!(lazy.clone().eig_vals().unwrap(), expected_vals);
            let full = lazy.clone().eig_full().unwrap();
            assert_eq!(full.0.data(), expected_full.0.data());
            assert_eq!(full.1.data(), expected_full.1.data());
            let trunc = lazy.clone().eig_trunc(&Truncation::rank(1)).unwrap();
            assert_eq!(trunc.eigenvalues, expected_trunc.eigenvalues);
            assert_eq!(trunc.error, expected_trunc.error);
            assert_eq!(trunc.d.data(), expected_trunc.d.data());
            assert_eq!(trunc.v.data(), expected_trunc.v.data());
            for output in [&full.0, &full.1, &trunc.d, &trunc.v] {
                assert!(output.owned_body().is_some());
                assert!(Arc::ptr_eq(
                    output.logical_space().provider_arc(),
                    source.logical_space().provider_arc()
                ));
            }
        }

        let calls = (0..4)
            .map(|_| {
                let clone = lazy.clone();
                std::thread::spawn(move || {
                    let vals = clone.eig_vals().unwrap();
                    let full = clone.eig_full().unwrap();
                    let trunc = clone.eig_trunc(&Truncation::rank(1)).unwrap();
                    (vals, full.0.data().to_vec(), trunc.error)
                })
            })
            .collect::<Vec<_>>();
        for call in calls {
            let (vals, diagonal, error) = call.join().unwrap();
            assert_eq!(vals, expected_vals);
            assert_eq!(diagonal, expected_full.0.data());
            assert_eq!(error, expected_trunc.error);
        }
        assert!(Arc::ptr_eq(owned(source), &parent_body));
        assert!(Arc::ptr_eq(&owned(source).data, &parent_data));
        assert_eq!(materialized_adjoint_builds(&lazy), 0);
        let TypedTensorRepr::Adjoint(view) = &lazy.repr else {
            unreachable!()
        };
        assert!(view.materialized.get().is_none());

        let _ = lazy.data();
        assert_eq!(materialized_adjoint_builds(&lazy), 1);
        assert!(view.materialized.get().is_some());
    }

    #[test]
    fn eig_dense_lazy_nonnormal_is_logical_owned_repeatable_and_cold() {
        let runtime = Runtime::builder().dense_threads(1).build().unwrap();
        let provider = Arc::new(U1FusionRule);
        let leg = GradedSpace::try_new(provider, [(U1Irrep::new(0), 2)], false).unwrap();
        let source = TensorMap::from_block_fn(&runtime, [&leg], [&leg], |_, indices| {
            match (indices[0], indices[1]) {
                (0, 0) => 1.0,
                (1, 1) => 2.0,
                (0, 1) => 1.0,
                _ => 0.0,
            }
        })
        .unwrap();
        let logical = eager_adjoint_oracle(&source);
        let (d, v) = source.adjoint().unwrap().eig_full().unwrap();
        let lhs = logical.to_c64().compose(&v).unwrap();
        let rhs = v.compose(&d).unwrap();
        assert_typed_map_close(&lhs, &rhs, 1.0e-12);

        assert_eig_uses_a_cold_logical_copy(&source);
    }

    #[test]
    fn eig_dense_lazy_real_order_signed_zero_and_defective_cases_match_logical_oracles() {
        let runtime = Runtime::builder().dense_threads(1).build().unwrap();
        let provider = Arc::new(U1FusionRule);
        let scalar_leg =
            GradedSpace::try_new(Arc::clone(&provider), [(U1Irrep::new(0), 1)], false).unwrap();
        let negative =
            TensorMap::from_block_fn(&runtime, [&scalar_leg], [&scalar_leg], |_, _| -2.0).unwrap();
        let lazy = negative.adjoint().unwrap();
        let value = lazy.eig_vals().unwrap()[0].values[0];
        assert_eq!(value, num_complex::Complex64::new(-2.0, 0.0));
        assert_eq!(value.im.to_bits(), 0.0f64.to_bits());
        assert_eq!(materialized_adjoint_builds(&lazy), 0);

        let leg = GradedSpace::try_new(provider, [(U1Irrep::new(0), 2)], false).unwrap();
        let rotation = TensorMap::from_block_fn(&runtime, [&leg], [&leg], |_, indices| {
            match (indices[0], indices[1]) {
                (0, 1) => -1.0,
                (1, 0) => 1.0,
                _ => 0.0,
            }
        })
        .unwrap();
        let eager = eager_adjoint_oracle(&rotation);
        let expected = eager.eig_vals().unwrap();
        let lazy = rotation.adjoint().unwrap();
        assert_eq!(lazy.eig_vals().unwrap(), expected);
        assert_eq!(expected[0].values[0].im, 1.0);
        assert_eq!(expected[0].values[1].im, -1.0);
        let trunc = lazy.eig_trunc(&Truncation::rank(1)).unwrap();
        let expected_trunc = eager.eig_trunc(&Truncation::rank(1)).unwrap();
        assert_eq!(trunc.eigenvalues, expected_trunc.eigenvalues);
        assert_eq!(trunc.d.data(), expected_trunc.d.data());
        assert_eq!(trunc.v.data(), expected_trunc.v.data());
        assert_eq!(trunc.error, expected_trunc.error);
        assert_eq!(materialized_adjoint_builds(&lazy), 0);

        for epsilon in [0.0, 1.0e-12] {
            let jordan = TensorMap::from_block_fn(&runtime, [&leg], [&leg], |_, indices| {
                match (indices[0], indices[1]) {
                    (0, 0) | (1, 1) => 1.0,
                    (0, 1) => 1.0,
                    (1, 0) => epsilon,
                    _ => unreachable!(),
                }
            })
            .unwrap();
            let eager = eager_adjoint_oracle(&jordan);
            let lazy = jordan.adjoint().unwrap();
            assert_eq!(lazy.eig_vals().unwrap(), eager.eig_vals().unwrap());
            let actual = lazy.eig_full().unwrap();
            let expected = eager.eig_full().unwrap();
            assert_eq!(actual.0.data(), expected.0.data());
            assert_eq!(actual.1.data(), expected.1.data());
            assert_eq!(materialized_adjoint_builds(&lazy), 0);
        }
    }

    #[test]
    fn eig_dense_lazy_failures_match_logical_oracle_and_stay_cold() {
        let runtime = Runtime::builder().dense_threads(1).build().unwrap();
        let provider = Arc::new(U1FusionRule);
        let left =
            GradedSpace::try_new(Arc::clone(&provider), [(U1Irrep::new(0), 2)], false).unwrap();
        let right = GradedSpace::try_new(provider, [(U1Irrep::new(0), 3)], false).unwrap();
        let source = TensorMap::from_block_fn(&runtime, [&left], [&right], |_, indices| {
            (indices[0] + indices[1]) as f64
        })
        .unwrap();
        let eager = eager_adjoint_oracle(&source);
        let expected = [
            eager.eig_vals().unwrap_err().to_string(),
            eager.eig_full().unwrap_err().to_string(),
            eager
                .eig_trunc(&Truncation::rank(1))
                .unwrap_err()
                .to_string(),
        ];
        let lazy = source.adjoint().unwrap();
        for _ in 0..2 {
            assert_eq!(lazy.eig_vals().unwrap_err().to_string(), expected[0]);
            assert_eq!(lazy.eig_full().unwrap_err().to_string(), expected[1]);
            assert_eq!(
                lazy.eig_trunc(&Truncation::rank(1))
                    .unwrap_err()
                    .to_string(),
                expected[2]
            );
        }
        assert_eq!(materialized_adjoint_builds(&lazy), 0);
        let TypedTensorRepr::Adjoint(view) = &lazy.repr else {
            unreachable!()
        };
        assert!(view.materialized.get().is_none());
    }

    #[test]
    fn exp_of_a_near_hermitian_adjoint_uses_the_logical_orientation_and_stays_cold() {
        // What: the fixed approximate-Hermitian dispatch must see logical A^H,
        // whose lower triangle is the conjugated parent upper triangle. A
        // parent-exp redirect feeds EIGH the other triangle and changes values.
        let runtime = Runtime::builder().dense_threads(1).build().unwrap();
        let provider = Arc::new(U1FusionRule);
        let leg = GradedSpace::try_new(provider, [(U1Irrep::new(0), 2)], false).unwrap();
        let delta = 4.0e-15;
        let parent = TensorMap::from_block_fn(&runtime, [&leg], [&leg], |_, indices| {
            match (indices[0], indices[1]) {
                (0, 0) => num_complex::Complex64::new(0.25, 0.0),
                (1, 1) => num_complex::Complex64::new(-0.5, 0.0),
                (0, 1) => num_complex::Complex64::new(delta, 0.0),
                _ => num_complex::Complex64::new(0.0, 0.0),
            }
        })
        .unwrap();
        let eager = eager_adjoint_oracle(&parent);
        let expected = eager.exp().unwrap();
        let parent_redirect = parent
            .exp()
            .unwrap()
            .adjoint()
            .unwrap()
            .materialized_tensor_uncached()
            .unwrap();
        let lazy = parent.adjoint().unwrap();
        let actual = lazy.exp().unwrap();

        assert_typed_map_close(&actual, &expected, 1.0e-20);
        assert!(parent_redirect
            .data()
            .iter()
            .zip(expected.data())
            .any(|(&actual, &expected)| {
                (actual.widen_complex() - expected.widen_complex()).norm() > 1.0e-16
            }));
        assert_eq!(materialized_adjoint_builds(&lazy), 0);
        let TypedTensorRepr::Adjoint(view) = &lazy.repr else {
            unreachable!()
        };
        assert!(view.materialized.get().is_none());
    }

    fn assert_exp_uses_a_cold_logical_copy<R, D>(source: &TensorMap<R, D>)
    where
        R: MultiplicityFreeRigidSymbols<Scalar = f64>
            + CheckedFusionAlgebra
            + SectorCodec
            + Send
            + Sync
            + 'static,
        D: TensorScalar + core::fmt::Debug + Send + Sync + 'static,
    {
        let eager = eager_adjoint_oracle(source);
        let expected = eager.exp().unwrap();
        let parent_body = Arc::clone(owned(source));
        let parent_data = Arc::clone(&parent_body.data);
        let lazy = source.adjoint().unwrap();

        for _ in 0..2 {
            let actual = lazy.clone().exp().unwrap();
            assert_typed_map_close(&actual, &expected, 1.0e-9);
            assert!(actual.owned_body().is_some());
            assert!(Arc::ptr_eq(
                actual.logical_space().provider_arc(),
                source.logical_space().provider_arc()
            ));
            assert!(!Arc::ptr_eq(&owned(&actual).data, &parent_data));
            let _ = actual.data();
        }
        let calls = (0..4)
            .map(|_| {
                let clone = lazy.clone();
                std::thread::spawn(move || clone.exp().unwrap())
            })
            .collect::<Vec<_>>();
        for call in calls {
            assert_typed_map_close(&call.join().unwrap(), &expected, 1.0e-9);
        }
        assert!(Arc::ptr_eq(owned(source), &parent_body));
        assert!(Arc::ptr_eq(&owned(source).data, &parent_data));
        assert_eq!(materialized_adjoint_builds(&lazy), 0);
        let TypedTensorRepr::Adjoint(view) = &lazy.repr else {
            unreachable!()
        };
        assert!(view.materialized.get().is_none());
    }

    #[test]
    fn exp_uses_owned_provider_native_outputs_without_warming_lazy_receivers() {
        // What: real/complex non-self-dual U(1) and a genuine SU(2) multitree
        // remain deterministic across repeats, clones, and concurrent calls.
        let runtime = Runtime::builder().dense_threads(1).build().unwrap();
        let provider = Arc::new(U1FusionRule);
        let leg = GradedSpace::try_new(
            provider,
            [
                (U1Irrep::new(-1), 2),
                (U1Irrep::new(0), 3),
                (U1Irrep::new(2), 1),
            ],
            false,
        )
        .unwrap();
        let u1 = TensorMap::from_block_fn(&runtime, [&leg], [&leg], |_, indices| {
            if indices[0] == indices[1] {
                0.25 + indices[0] as f64 / 10.0
            } else {
                (indices[0] + 2 * indices[1] + 1) as f64 / 100.0
            }
        })
        .unwrap();
        assert_exp_uses_a_cold_logical_copy(&u1);
        assert_exp_uses_a_cold_logical_copy(&genuinely_complex(&u1));

        let provider = Arc::new(SU2FusionRule);
        let half =
            GradedSpace::try_new(provider, [(SU2Irrep::from_twice_spin(1), 1)], false).unwrap();
        let su2 = TensorMap::from_block_fn(
            &runtime,
            [&half, &half, &half],
            [&half, &half, &half],
            |_, indices| (indices.iter().sum::<usize>() + 1) as f64 / 20.0,
        )
        .unwrap();
        assert!(su2.logical_space().space().structure().block_count() > 1);
        assert_exp_uses_a_cold_logical_copy(&genuinely_complex(&su2));
    }

    #[test]
    fn exp_failure_leaves_the_lazy_receiver_and_parent_untouched() {
        let runtime = Runtime::builder().dense_threads(1).build().unwrap();
        let provider = Arc::new(U1FusionRule);
        let leg = GradedSpace::try_new(provider, [(U1Irrep::new(0), 2)], false).unwrap();
        let source = TensorMap::from_block_fn(&runtime, [&leg], [&leg], |_, indices| {
            if indices == [0, 1] {
                num_complex::Complex64::new(f64::NAN, 0.0)
            } else {
                num_complex::Complex64::new((indices[0] + indices[1] + 1) as f64, 0.0)
            }
        })
        .unwrap();
        let before = source.data().to_vec();
        let parent = Arc::clone(owned(&source));
        let data = Arc::clone(&parent.data);
        let lazy = source.adjoint().unwrap();

        assert!(matches!(lazy.exp(), Err(Error::Operation(_))));
        assert!(source.data().iter().zip(&before).all(|(actual, expected)| {
            actual.re.to_bits() == expected.re.to_bits()
                && actual.im.to_bits() == expected.im.to_bits()
        }));
        assert!(Arc::ptr_eq(owned(&source), &parent));
        assert!(Arc::ptr_eq(&owned(&source).data, &data));
        assert_eq!(materialized_adjoint_builds(&lazy), 0);
        let TypedTensorRepr::Adjoint(view) = &lazy.repr else {
            unreachable!()
        };
        assert!(view.materialized.get().is_none());
    }

    fn assert_sqrt_uses_a_cold_logical_copy<R, D>(source: &TensorMap<R, D>)
    where
        R: MultiplicityFreeRigidSymbols<Scalar = f64>
            + CheckedFusionAlgebra
            + SectorCodec
            + Send
            + Sync
            + 'static,
        D: TensorScalar + core::fmt::Debug + Send + Sync + 'static,
    {
        let expected = eager_adjoint_oracle(source).sqrt().unwrap();
        let parent = Arc::clone(owned(source));
        let parent_data = Arc::clone(&parent.data);
        let lazy = source.adjoint().unwrap();

        for _ in 0..2 {
            let actual = lazy.clone().sqrt().unwrap();
            assert_typed_map_close(&actual, &expected, f64::EPSILON);
            assert!(actual.owned_body().is_some());
            assert!(Arc::ptr_eq(
                actual.logical_space().provider_arc(),
                source.logical_space().provider_arc()
            ));
            assert!(!Arc::ptr_eq(&owned(&actual).data, &parent_data));
        }
        let calls = (0..4)
            .map(|_| {
                let clone = lazy.clone();
                std::thread::spawn(move || clone.sqrt().unwrap())
            })
            .collect::<Vec<_>>();
        for call in calls {
            assert_typed_map_close(&call.join().unwrap(), &expected, f64::EPSILON);
        }
        assert!(Arc::ptr_eq(owned(source), &parent));
        assert!(Arc::ptr_eq(&owned(source).data, &parent_data));
        assert_eq!(materialized_adjoint_builds(&lazy), 0);
        let TypedTensorRepr::Adjoint(view) = &lazy.repr else {
            unreachable!()
        };
        assert!(view.materialized.get().is_none());

        let _ = lazy.data();
        assert_eq!(materialized_adjoint_builds(&lazy), 1);
        assert!(view.materialized.get().is_some());
    }

    #[test]
    fn sqrt_dense_lazy_success_is_owned_provider_native_repeatable_and_cold() {
        let runtime = Runtime::builder().dense_threads(1).build().unwrap();
        let provider = Arc::new(U1FusionRule);
        let leg = GradedSpace::try_new(
            provider,
            [(U1Irrep::new(-1), 2), (U1Irrep::new(2), 4)],
            false,
        )
        .unwrap();
        let f64_source = TensorMap::from_block_fn(&runtime, [&leg], [&leg], |_, indices| {
            if indices[0] == indices[1] {
                (indices[0] + 1) as f64
            } else {
                0.0
            }
        })
        .unwrap();
        assert_sqrt_uses_a_cold_logical_copy(&f64_source);

        let c64_source = TensorMap::from_block_fn(&runtime, [&leg], [&leg], |_, indices| {
            if indices[0] == indices[1] {
                match indices[0] % 4 {
                    0 => num_complex::Complex64::new(-4.0, 0.0),
                    1 => num_complex::Complex64::new(-4.0, -0.0),
                    2 => num_complex::Complex64::new(-4.0, 1.0e-300),
                    _ => num_complex::Complex64::new(-4.0, -1.0e-300),
                }
            } else {
                num_complex::Complex64::new(0.0, -0.0)
            }
        })
        .unwrap();
        let expected = eager_adjoint_oracle(&c64_source).sqrt().unwrap();
        let lazy = c64_source.adjoint().unwrap();
        let actual = lazy.sqrt().unwrap();
        assert!(actual
            .data()
            .iter()
            .zip(expected.data())
            .all(|(actual, expected)| {
                actual.re.to_bits() == expected.re.to_bits()
                    && actual.im.to_bits() == expected.im.to_bits()
            }));
        assert_eq!(materialized_adjoint_builds(&lazy), 0);
        assert_sqrt_uses_a_cold_logical_copy(&c64_source);

        let provider = Arc::new(SU2FusionRule);
        let leg = GradedSpace::try_new(
            provider,
            [
                (SU2Irrep::from_twice_spin(0), 2),
                (SU2Irrep::from_twice_spin(1), 4),
            ],
            false,
        )
        .unwrap();
        let su2 = TensorMap::from_block_fn(&runtime, [&leg], [&leg], |_, indices| {
            if indices[0] == indices[1] {
                num_complex::Complex64::new(-4.0, (indices[0] as f64 - 1.0) / 10.0)
            } else {
                num_complex::Complex64::new(0.0, 0.0)
            }
        })
        .unwrap();
        assert_sqrt_uses_a_cold_logical_copy(&su2);
    }

    #[test]
    fn sqrt_dense_lazy_failures_preserve_logical_order_and_stay_cold() {
        let runtime = Runtime::builder().dense_threads(1).build().unwrap();
        let provider = Arc::new(U1FusionRule);
        let leg = GradedSpace::try_new(provider, [(U1Irrep::new(0), 3)], false).unwrap();

        let offdiag = TensorMap::from_block_fn(&runtime, [&leg], [&leg], |_, indices| {
            if indices == [0, 1] {
                1.0
            } else {
                0.0
            }
        })
        .unwrap();
        let lazy = offdiag.adjoint().unwrap();
        let error = lazy.sqrt().unwrap_err().to_string();
        assert!(error.contains("(1, 0)"), "{error}");
        assert_eq!(materialized_adjoint_builds(&lazy), 0);

        let mixed = TensorMap::from_block_fn(&runtime, [&leg], [&leg], |_, indices| {
            match (indices[0], indices[1]) {
                (1, 1) => -1.0,
                (0, 2) => 1.0,
                _ => 0.0,
            }
        })
        .unwrap();
        let eager_error = eager_adjoint_oracle(&mixed).sqrt().unwrap_err().to_string();
        let lazy = mixed.adjoint().unwrap();
        assert_eq!(lazy.sqrt().unwrap_err().to_string(), eager_error);
        assert!(eager_error.contains("negative"), "{eager_error}");
        assert_eq!(materialized_adjoint_builds(&lazy), 0);

        let negative = TensorMap::from_block_fn(&runtime, [&leg], [&leg], |_, indices| {
            if indices[0] == indices[1] {
                -1.0
            } else {
                0.0
            }
        })
        .unwrap();
        let eager_error = eager_adjoint_oracle(&negative)
            .sqrt()
            .unwrap_err()
            .to_string();
        let lazy = negative.adjoint().unwrap();
        for _ in 0..2 {
            assert_eq!(lazy.clone().sqrt().unwrap_err().to_string(), eager_error);
        }
        assert_eq!(materialized_adjoint_builds(&lazy), 0);
        let TypedTensorRepr::Adjoint(view) = &lazy.repr else {
            unreachable!()
        };
        assert!(view.materialized.get().is_none());
    }

    fn assert_polar_factors<R, D>(
        source: &TensorMap<R, D>,
        target: &TensorMap<R, D>,
        actual: &(TensorMap<R, D>, TensorMap<R, D>),
        expected: &(TensorMap<R, D>, TensorMap<R, D>),
        left: bool,
    ) where
        R: MultiplicityFreeRigidSymbols<Scalar = f64> + CheckedFusionAlgebra + SectorCodec,
        D: TensorScalar + core::fmt::Debug,
    {
        let reconstructed = actual.0.compose(&actual.1).unwrap();
        assert_typed_map_close(&reconstructed, target, 1e-10);
        let (positive, isometry) = if left {
            (&actual.1, &actual.0)
        } else {
            (&actual.0, &actual.1)
        };
        assert!(if left {
            isometry.is_isometric(1e-11).unwrap()
        } else {
            isometry.adjoint().unwrap().is_isometric(1e-11).unwrap()
        });
        assert!(positive.is_hermitian(1e-11).unwrap());
        assert!(positive
            .eigh_vals()
            .unwrap()
            .iter()
            .all(|entry| entry.values.iter().all(|&value| value >= -1e-11)));
        for factor in [&actual.0, &actual.1] {
            assert!(factor.owned_body().is_some());
            assert!(Arc::ptr_eq(
                factor.logical_space().provider_arc(),
                source.logical_space().provider_arc()
            ));
            let _ = factor.data();
        }
        assert_eq!(
            actual.0.logical_space().space(),
            expected.0.logical_space().space()
        );
        assert_eq!(
            actual.1.logical_space().space(),
            expected.1.logical_space().space()
        );
    }

    fn assert_rank_deficient_polar_support<R, D>(source: &TensorMap<R, D>, left: bool)
    where
        R: MultiplicityFreeRigidSymbols<Scalar = f64> + CheckedFusionAlgebra + SectorCodec,
        D: TensorScalar + core::fmt::Debug,
    {
        let target = eager_adjoint_oracle(source);
        let target_pinv = target.pinv(1e-10).unwrap();
        let target_codomain = target.compose(&target_pinv).unwrap();
        let target_domain = target_pinv.compose(&target).unwrap();
        let lazy = source.adjoint().unwrap();
        let factors = if left {
            lazy.left_polar().unwrap()
        } else {
            lazy.right_polar().unwrap()
        };
        let (positive, isometry) = if left {
            (&factors.1, &factors.0)
        } else {
            (&factors.0, &factors.1)
        };
        let positive_pinv = positive.pinv(1e-10).unwrap();
        if left {
            let support = positive_pinv.compose(positive).unwrap();
            assert_typed_map_close(&support, &target_domain, 1e-9);
            let image = isometry
                .compose(&support)
                .unwrap()
                .compose(&isometry.adjoint().unwrap())
                .unwrap();
            assert_typed_map_close(&image, &target_codomain, 1e-9);
        } else {
            let support = positive.compose(&positive_pinv).unwrap();
            assert_typed_map_close(&support, &target_codomain, 1e-9);
            let image = isometry
                .adjoint()
                .unwrap()
                .compose(&support)
                .unwrap()
                .compose(isometry)
                .unwrap();
            assert_typed_map_close(&image, &target_domain, 1e-9);
        }
        assert_eq!(materialized_adjoint_builds(&lazy), 0);
        let TypedTensorRepr::Adjoint(view) = &lazy.repr else {
            unreachable!()
        };
        assert!(view.materialized.get().is_none());
    }

    fn assert_inverse_redirect<R, D>(source: &TensorMap<R, D>)
    where
        R: MultiplicityFreeRigidSymbols<Scalar = f64>
            + CheckedFusionAlgebra
            + SectorCodec
            + Send
            + Sync
            + 'static,
        D: TensorScalar + core::fmt::Debug + Send + Sync + 'static,
    {
        let eager = eager_adjoint_oracle(source);
        let expected = eager.inv().unwrap();
        let parent_body = Arc::clone(owned(source));
        let parent_data = Arc::clone(&parent_body.data);
        let lazy = source.adjoint().unwrap();

        for _ in 0..2 {
            let actual = lazy.clone().inv().unwrap();
            assert_typed_map_close(&actual, &expected, 1e-10);
            let codomain = eager.codomain();
            let domain = eager.domain();
            assert_typed_map_close(
                &eager.compose(&actual).unwrap(),
                &TensorMap::id(source.runtime(), codomain.iter()).unwrap(),
                1e-9,
            );
            assert_typed_map_close(
                &actual.compose(&eager).unwrap(),
                &TensorMap::id(source.runtime(), domain.iter()).unwrap(),
                1e-9,
            );
            assert!(actual.owned_body().is_some());
            assert!(Arc::ptr_eq(
                actual.logical_space().provider_arc(),
                source.logical_space().provider_arc()
            ));
            assert!(!Arc::ptr_eq(&owned(&actual).data, &parent_data));
            let _ = actual.data();
        }

        let calls = (0..4)
            .map(|_| {
                let clone = lazy.clone();
                std::thread::spawn(move || clone.inv().unwrap())
            })
            .collect::<Vec<_>>();
        for call in calls {
            assert_typed_map_close(&call.join().unwrap(), &expected, 1e-10);
        }

        assert!(Arc::ptr_eq(owned(source), &parent_body));
        assert!(Arc::ptr_eq(&owned(source).data, &parent_data));
        assert_eq!(materialized_adjoint_builds(&lazy), 0);
        let TypedTensorRepr::Adjoint(view) = &lazy.repr else {
            unreachable!()
        };
        assert!(view.materialized.get().is_none());
    }

    #[test]
    fn inverse_redirect_is_owned_provider_native_repeatable_and_cold() {
        // What: U(1) complex blocks and a genuine SU(2) multitree use the
        // inverse identity without warming the lazy receiver, including
        // cloned and concurrent calls, and return detached provider-native data.
        let runtime = Runtime::builder().dense_threads(1).build().unwrap();
        let provider = Arc::new(U1FusionRule);
        let leg = GradedSpace::try_new(
            provider,
            [
                (U1Irrep::new(-1), 2),
                (U1Irrep::new(0), 3),
                (U1Irrep::new(2), 1),
            ],
            false,
        )
        .unwrap();
        let u1 = TensorMap::from_block_fn(&runtime, [&leg], [&leg], |_, indices| {
            let re = if indices[0] == indices[1] {
                20.0 + indices[0] as f64
            } else {
                (indices[0] + 2 * indices[1] + 1) as f64 / 100.0
            };
            num_complex::Complex64::new(re, (indices[0] + indices[1] + 1) as f64 / 200.0)
        })
        .unwrap();
        let identity = TensorMap::<_, num_complex::Complex64>::id(&runtime, [&leg]).unwrap();
        let u1 = u1
            .add(
                &identity,
                num_complex::Complex64::new(1.0, 0.0),
                num_complex::Complex64::new(100.0, 0.0),
            )
            .unwrap();
        assert_inverse_redirect(&u1);

        let provider = Arc::new(U1FusionRule);
        let wide =
            GradedSpace::try_new(Arc::clone(&provider), [(U1Irrep::new(0), 4)], false).unwrap();
        let narrow = GradedSpace::try_new(provider, [(U1Irrep::new(0), 2)], false).unwrap();
        let unequal =
            TensorMap::from_block_fn(&runtime, [&wide], [&narrow, &narrow], |_, indices| {
                let column = 2 * indices[1] + indices[2];
                if indices[0] == column {
                    10.0 + indices[0] as f64
                } else {
                    (indices[0] + column + 1) as f64 / 100.0
                }
            })
            .unwrap();
        assert_ne!(unequal.codomain_rank(), unequal.domain_rank());
        assert_inverse_redirect(&unequal);

        let provider = Arc::new(SU2FusionRule);
        let half =
            GradedSpace::try_new(provider, [(SU2Irrep::from_twice_spin(1), 1)], false).unwrap();
        let su2 = TensorMap::from_block_fn(
            &runtime,
            [&half, &half, &half],
            [&half, &half, &half],
            |_, indices| {
                if indices[..3] == indices[3..] {
                    20.0 + indices.iter().sum::<usize>() as f64
                } else {
                    (indices.iter().sum::<usize>() + 1) as f64 / 100.0
                }
            },
        )
        .unwrap();
        let identity = TensorMap::<_, f64>::id(&runtime, [&half, &half, &half]).unwrap();
        let su2 = su2.add(&identity, 1.0, 100.0).unwrap();
        assert!(su2.logical_space().space().structure().block_count() > 1);
        assert_inverse_redirect(&su2);
    }

    #[test]
    fn inverse_redirect_failure_and_negative_powi_leave_the_receiver_cold() {
        // What: negative powers inherit the inverse redirect, while a singular
        // solve changes neither parent Arc/bytes nor the lazy receiver cache.
        let runtime = Runtime::builder().dense_threads(1).build().unwrap();
        let provider = Arc::new(U1FusionRule);
        let leg = GradedSpace::try_new(provider, [(U1Irrep::new(0), 3)], false).unwrap();
        let source = TensorMap::from_block_fn(&runtime, [&leg], [&leg], |_, indices| {
            if indices[0] == indices[1] {
                4.0 + indices[0] as f64
            } else {
                (indices[0] + indices[1] + 1) as f64 / 20.0
            }
        })
        .unwrap();
        let lazy = source.adjoint().unwrap();
        let eager = eager_adjoint_oracle(&source);
        assert_typed_map_close(&lazy.powi(-3).unwrap(), &eager.powi(-3).unwrap(), 1e-10);
        assert_eq!(materialized_adjoint_builds(&lazy), 0);

        let singular = source.scale(0.0);
        let before = singular.data().to_vec();
        let body = Arc::clone(owned(&singular));
        let data = Arc::clone(&body.data);
        let cold = singular.adjoint().unwrap();
        assert!(matches!(cold.inv(), Err(Error::Operation(_))));
        assert_eq!(singular.data(), before);
        assert!(Arc::ptr_eq(owned(&singular), &body));
        assert!(Arc::ptr_eq(&owned(&singular).data, &data));
        assert_eq!(materialized_adjoint_builds(&cold), 0);
        let TypedTensorRepr::Adjoint(view) = &cold.repr else {
            unreachable!()
        };
        assert!(view.materialized.get().is_none());

        // The first U(1) sector solves before the second singular sector
        // fails, pinning atomicity after partial backend progress.
        let provider = Arc::new(U1FusionRule);
        let leg = GradedSpace::try_new(
            provider,
            [(U1Irrep::new(0), 1), (U1Irrep::new(1), 2)],
            false,
        )
        .unwrap();
        let late = TensorMap::from_block_fn(&runtime, [&leg], [&leg], |trees, indices| {
            if trees.codomain_uncoupled[0] == U1Irrep::new(0) && indices[0] == indices[1] {
                2.0
            } else {
                0.0
            }
        })
        .unwrap();
        let before = late.data().to_vec();
        let data = Arc::clone(&owned(&late).data);
        let cold = late.adjoint().unwrap();
        assert!(matches!(cold.inv(), Err(Error::Operation(_))));
        assert_eq!(late.data(), before);
        assert!(Arc::ptr_eq(&owned(&late).data, &data));
        assert_eq!(materialized_adjoint_builds(&cold), 0);
        let TypedTensorRepr::Adjoint(view) = &cold.repr else {
            unreachable!()
        };
        assert!(view.materialized.get().is_none());
    }

    fn assert_pinv_redirect<R, D>(source: &TensorMap<R, D>, rcond: f64, exact_original: bool)
    where
        R: MultiplicityFreeRigidSymbols<Scalar = f64>
            + CheckedFusionAlgebra
            + SectorCodec
            + Send
            + Sync
            + 'static,
        D: TensorScalar + core::fmt::Debug + Send + Sync + 'static,
    {
        let eager = eager_adjoint_oracle(source);
        let expected = eager.pinv(rcond).unwrap();
        let parent_body = Arc::clone(owned(source));
        let parent_data = Arc::clone(&parent_body.data);
        let lazy = source.adjoint().unwrap();

        for _ in 0..2 {
            let actual = lazy.clone().pinv(rcond).unwrap();
            assert_typed_map_close(&actual, &expected, 1e-9);
            let pap = actual.compose(&eager).unwrap().compose(&actual).unwrap();
            assert_typed_map_close(&pap, &actual, 1e-8);
            assert!(eager.compose(&actual).unwrap().is_hermitian(1e-9).unwrap());
            assert!(actual.compose(&eager).unwrap().is_hermitian(1e-9).unwrap());
            if exact_original {
                let apa = eager.compose(&actual).unwrap().compose(&eager).unwrap();
                assert_typed_map_close(&apa, &eager, 1e-8);
            }
            assert!(actual.owned_body().is_some());
            assert!(Arc::ptr_eq(
                actual.logical_space().provider_arc(),
                source.logical_space().provider_arc()
            ));
            assert!(!Arc::ptr_eq(&owned(&actual).data, &parent_data));
            let _ = actual.data();
        }

        let calls = (0..4)
            .map(|_| {
                let clone = lazy.clone();
                std::thread::spawn(move || clone.pinv(rcond).unwrap())
            })
            .collect::<Vec<_>>();
        for call in calls {
            assert_typed_map_close(&call.join().unwrap(), &expected, 1e-9);
        }
        assert!(Arc::ptr_eq(owned(source), &parent_body));
        assert!(Arc::ptr_eq(&owned(source).data, &parent_data));
        assert_eq!(materialized_adjoint_builds(&lazy), 0);
        let TypedTensorRepr::Adjoint(view) = &lazy.repr else {
            unreachable!()
        };
        assert!(view.materialized.get().is_none());
    }

    #[test]
    fn pinv_redirect_preserves_semantics_ownership_and_cold_concurrency() {
        // What: full-rank, rectangular/rank-deficient, non-self-dual U(1),
        // complex data, empty support, and a genuine SU(2) multitree all use
        // the parent-factor seam and return detached provider-native outputs.
        let runtime = Runtime::builder().dense_threads(1).build().unwrap();
        let provider = Arc::new(U1FusionRule);
        let leg = GradedSpace::try_new(
            provider,
            [(U1Irrep::new(-1), 2), (U1Irrep::new(0), 3)],
            false,
        )
        .unwrap();
        let full = TensorMap::from_block_fn(&runtime, [&leg], [&leg], |_, indices| {
            let re = if indices[0] == indices[1] {
                30.0 + indices[0] as f64
            } else {
                (indices[0] + indices[1] + 1) as f64 / 100.0
            };
            num_complex::Complex64::new(re, (indices[0] + 2 * indices[1] + 1) as f64 / 200.0)
        })
        .unwrap();
        assert_pinv_redirect(&full, 1e-12, true);

        assert_pinv_redirect(
            &genuinely_complex(&u1_matrix_fixture([(0, 3), (1, 2)], [(0, 2)])),
            1e-10,
            false,
        );
        assert_pinv_redirect(&u1_matrix_fixture([(1, 2)], [(0, 3)]), 1e-10, false);
        let su2 = genuinely_complex(&su2_lazy_fixture());
        assert!(su2.logical_space().space().structure().block_count() > 1);
        assert_pinv_redirect(&su2, 1e-10, false);
    }

    #[test]
    fn pinv_redirect_late_svd_failure_keeps_parent_and_receiver_cold() {
        // What: a successful first-sector SVD cannot publish factors, mutate
        // parent bytes, or initialize the lazy receiver when sector two fails.
        let runtime = Runtime::builder()
            .with_dense_executor(Box::new(FailSecondSvd::default()))
            .build()
            .unwrap();
        let provider = Arc::new(U1FusionRule);
        let leg = GradedSpace::try_new(
            provider,
            [(U1Irrep::new(0), 2), (U1Irrep::new(1), 2)],
            false,
        )
        .unwrap();
        let source = TensorMap::from_block_fn(&runtime, [&leg], [&leg], |_, indices| {
            (indices.iter().sum::<usize>() + 1) as f64
        })
        .unwrap();
        let before = source.data().to_vec();
        let data = Arc::clone(&owned(&source).data);
        let lazy = source.adjoint().unwrap();
        assert!(matches!(lazy.pinv(0.0), Err(Error::Operation(_))));
        assert_eq!(source.data(), before);
        assert!(Arc::ptr_eq(&owned(&source).data, &data));
        assert_eq!(materialized_adjoint_builds(&lazy), 0);
        let TypedTensorRepr::Adjoint(view) = &lazy.repr else {
            unreachable!()
        };
        assert!(view.materialized.get().is_none());
    }

    #[test]
    fn polar_redirects_through_parent_with_owned_psd_factors_and_a_cold_receiver() {
        let tall = u1_matrix_fixture([(0, 3)], [(0, 2)]);
        assert_polar_redirect(&tall, false);
        assert_polar_redirect(&genuinely_complex(&tall), false);

        let wide = u1_matrix_fixture([(0, 2)], [(0, 3)]);
        assert_polar_redirect(&wide, true);

        let codomain_only = u1_matrix_fixture([(0, 2), (1, 2)], [(0, 2)]);
        assert_polar_redirect(&codomain_only, false);
        let domain_only = u1_matrix_fixture([(0, 2)], [(0, 2), (1, 2)]);
        assert_polar_redirect(&domain_only, true);

        let provider = Arc::new(SU2FusionRule);
        let half =
            GradedSpace::try_new(provider, [(SU2Irrep::from_twice_spin(1), 1)], false).unwrap();
        let multitree = TensorMap::from_block_fn(
            &Runtime::builder().dense_threads(1).build().unwrap(),
            [&half, &half, &half],
            [&half],
            |_, indices| (indices.iter().sum::<usize>() + 1) as f64,
        )
        .unwrap();
        assert_eq!(
            multitree.logical_space().space().structure().block_count(),
            2
        );
        let multitree = genuinely_complex(&multitree);
        assert_polar_redirect(&multitree, false);

        let runtime = Runtime::builder().dense_threads(1).build().unwrap();
        let provider = Arc::new(U1FusionRule);
        let leg = GradedSpace::try_new(provider, [(U1Irrep::new(0), 3)], false).unwrap();
        let rank_deficient = TensorMap::from_block_fn(&runtime, [&leg], [&leg], |_, indices| {
            let value = ((indices[0] + 1) * (indices[1] + 1)) as f64;
            num_complex::Complex64::new(value, value / 3.0)
        })
        .unwrap();
        assert_polar_redirect(&rank_deficient, true);
        assert_polar_redirect(&rank_deficient, false);
        assert_rank_deficient_polar_support(&rank_deficient, true);
        assert_rank_deficient_polar_support(&rank_deficient, false);
    }

    #[test]
    fn polar_redirect_repeats_clones_and_runs_concurrently_without_warming_receiver() {
        let source = genuinely_complex(&u1_matrix_fixture([(0, 3)], [(0, 3)]));
        let target = eager_adjoint_oracle(&source);
        let lazy = source.adjoint().unwrap();
        for left in [true, false] {
            let expected = if left {
                target.left_polar().unwrap()
            } else {
                target.right_polar().unwrap()
            };
            for _ in 0..2 {
                let actual = if left {
                    lazy.clone().left_polar().unwrap()
                } else {
                    lazy.clone().right_polar().unwrap()
                };
                assert_polar_factors(&source, &target, &actual, &expected, left);
            }
            let calls = (0..4)
                .map(|_| {
                    let clone = lazy.clone();
                    std::thread::spawn(move || {
                        if left {
                            clone.left_polar().unwrap()
                        } else {
                            clone.right_polar().unwrap()
                        }
                    })
                })
                .collect::<Vec<_>>();
            for call in calls {
                let actual = call.join().unwrap();
                assert_polar_factors(&source, &target, &actual, &expected, left);
            }
        }
        assert_eq!(materialized_adjoint_builds(&lazy), 0);
        let TypedTensorRepr::Adjoint(view) = &lazy.repr else {
            unreachable!()
        };
        assert!(view.materialized.get().is_none());
    }

    #[test]
    fn polar_redirect_wrong_direction_keeps_requested_name_and_receiver_cold() {
        let source = u1_matrix_fixture([(0, 3)], [(0, 2)]);
        let lazy = source.adjoint().unwrap();
        let error = lazy.left_polar().unwrap_err();
        assert!(matches!(
            error,
            Error::Operation(error)
                if matches!(
                    error.as_ref(),
                    tenet_tensors::OperationError::InvalidArgument { message }
                        if *message == "left_polar requires rows >= columns in every coupled-sector matrix"
                )
        ));
        assert_eq!(materialized_adjoint_builds(&lazy), 0);

        let source = u1_matrix_fixture([(0, 2)], [(0, 3)]);
        let lazy = source.adjoint().unwrap();
        let error = lazy.right_polar().unwrap_err();
        assert!(matches!(
            error,
            Error::Operation(error)
                if matches!(
                    error.as_ref(),
                    tenet_tensors::OperationError::InvalidArgument { message }
                        if *message == "right_polar requires columns >= rows in every coupled-sector matrix"
                )
        ));
        assert_eq!(materialized_adjoint_builds(&lazy), 0);
    }

    #[test]
    fn polar_redirect_late_failure_leaves_parent_and_receiver_unchanged() {
        for left in [true, false] {
            let runtime = Runtime::builder()
                .with_dense_executor(Box::new(FailSecondSvd::default()))
                .build()
                .unwrap();
            let provider = Arc::new(U1FusionRule);
            let leg = GradedSpace::try_new(
                provider,
                [(U1Irrep::new(0), 2), (U1Irrep::new(1), 2)],
                false,
            )
            .unwrap();
            let source = TensorMap::from_block_fn(&runtime, [&leg], [&leg], |_, indices| {
                (indices.iter().sum::<usize>() + 1) as f64
            })
            .unwrap();
            let before = source.data().to_vec();
            let lazy = source.adjoint().unwrap();
            let result = if left {
                lazy.left_polar()
            } else {
                lazy.right_polar()
            };
            assert!(matches!(result, Err(Error::Operation(_))));
            assert_eq!(source.data(), before);
            assert_eq!(materialized_adjoint_builds(&lazy), 0);
            let TypedTensorRepr::Adjoint(view) = &lazy.repr else {
                unreachable!()
            };
            assert!(view.materialized.get().is_none());
        }
    }

    fn assert_qr_lq_factors<R, D>(
        source: &TensorMap<R, D>,
        target: &TensorMap<R, D>,
        actual: &(TensorMap<R, D>, TensorMap<R, D>),
        expected: &(TensorMap<R, D>, TensorMap<R, D>),
        qr: bool,
        compare_gauge: bool,
    ) where
        R: MultiplicityFreeRigidSymbols<Scalar = f64> + CheckedFusionAlgebra + SectorCodec,
        D: TensorScalar + core::fmt::Debug,
    {
        for (actual, expected) in [(&actual.0, &expected.0), (&actual.1, &expected.1)] {
            assert!(actual.owned_body().is_some());
            assert_eq!(
                actual.logical_space().space(),
                expected.logical_space().space()
            );
            assert!(Arc::ptr_eq(
                actual.logical_space().provider_arc(),
                source.logical_space().provider_arc()
            ));
            if compare_gauge {
                assert!(actual
                    .data()
                    .iter()
                    .zip(expected.data())
                    .all(|(&left, &right)| {
                        (left.widen_complex() - right.widen_complex()).norm() < 1e-12
                    }));
            }
        }
        let isometry = if qr {
            actual.0.is_isometric(1e-12).unwrap()
        } else {
            actual.1.adjoint().unwrap().is_isometric(1e-12).unwrap()
        };
        assert!(isometry);
        let rebuilt = actual.0.compose(&actual.1).unwrap();
        assert!(rebuilt
            .data()
            .iter()
            .zip(target.data())
            .all(|(&left, &right)| {
                (left.widen_complex() - right.widen_complex()).norm() < 1e-12
            }));
    }

    fn assert_qr_lq_keeps_input_cache_cold<R, D>(source: &TensorMap<R, D>)
    where
        R: MultiplicityFreeRigidSymbols<Scalar = f64> + CheckedFusionAlgebra + SectorCodec,
        D: TensorScalar + core::fmt::Debug,
    {
        let target = eager_adjoint_oracle(source);
        let lazy = source.adjoint().unwrap();

        let actual = lazy.qr_compact().unwrap();
        assert_qr_lq_factors(
            source,
            &target,
            &actual,
            &target.qr_compact().unwrap(),
            true,
            true,
        );
        let actual = lazy.lq_compact().unwrap();
        assert_qr_lq_factors(
            source,
            &target,
            &actual,
            &target.lq_compact().unwrap(),
            false,
            true,
        );
        let actual = lazy.qr_full().unwrap();
        assert_qr_lq_factors(
            source,
            &target,
            &actual,
            &target.qr_full().unwrap(),
            true,
            false,
        );
        let actual = lazy.lq_full().unwrap();
        assert_qr_lq_factors(
            source,
            &target,
            &actual,
            &target.lq_full().unwrap(),
            false,
            false,
        );

        assert_eq!(materialized_adjoint_builds(&lazy), 0);
        let TypedTensorRepr::Adjoint(view) = &lazy.repr else {
            unreachable!()
        };
        assert!(view.materialized.get().is_none());
    }

    #[test]
    fn qr_lq_adjoint_dispatch_covers_unique_simple_dtypes_and_rectangles() {
        // What: adjoint QR uses an operation-local logical copy while LQ uses
        // the parent QR, preserving compact gauge, full semantics, and provider authority.
        let tall = u1_matrix_fixture([(-2, 1), (0, 3)], [(-2, 1), (0, 1)]);
        let wide = u1_matrix_fixture([(-2, 1), (0, 1)], [(-2, 1), (0, 3)]);
        assert_qr_lq_keeps_input_cache_cold(&tall);
        assert_qr_lq_keeps_input_cache_cold(&genuinely_complex(&tall));
        assert_qr_lq_keeps_input_cache_cold(&wide);
        assert_qr_lq_keeps_input_cache_cold(&genuinely_complex(&wide));

        let multitree = su2_lazy_fixture();
        assert_qr_lq_keeps_input_cache_cold(&multitree);
        assert_qr_lq_keeps_input_cache_cold(&genuinely_complex(&multitree));
    }

    #[test]
    fn full_qr_lq_adjoint_dispatch_handles_unmatched_and_disjoint_sectors() {
        for source in [
            u1_matrix_fixture([(0, 2), (1, 1)], [(0, 3)]),
            u1_matrix_fixture([(0, 2)], [(0, 3), (1, 1)]),
        ] {
            let target = eager_adjoint_oracle(&source);
            let lazy = source.adjoint().unwrap();
            let qr = lazy.qr_full().unwrap();
            assert_qr_lq_factors(
                &source,
                &target,
                &qr,
                &target.qr_full().unwrap(),
                true,
                false,
            );
            let lq = lazy.lq_full().unwrap();
            assert_qr_lq_factors(
                &source,
                &target,
                &lq,
                &target.lq_full().unwrap(),
                false,
                false,
            );
            assert_eq!(materialized_adjoint_builds(&lazy), 0);
            let TypedTensorRepr::Adjoint(view) = &lazy.repr else {
                unreachable!()
            };
            assert!(view.materialized.get().is_none());
        }
    }

    #[test]
    fn qr_lq_adjoint_dispatch_handles_an_empty_homspace() {
        let source = u1_matrix_fixture([(1, 2)], [(0, 3)]);
        assert!(source.data().is_empty());
        assert_qr_lq_keeps_input_cache_cold(&source);
    }

    #[test]
    fn qr_lq_uncached_owned_outputs_repeat_clone_and_run_concurrently() {
        let source = genuinely_complex(&su2_lazy_fixture());
        let target = eager_adjoint_oracle(&source);
        let lazy = source.adjoint().unwrap();
        let expected_qr = lazy.qr_compact().unwrap();
        let expected_lq = lazy.lq_full().unwrap();
        for _ in 0..2 {
            let qr = lazy.clone().qr_compact().unwrap();
            let lq = lazy.clone().lq_full().unwrap();
            assert_qr_lq_factors(&source, &target, &qr, &expected_qr, true, true);
            assert_qr_lq_factors(&source, &target, &lq, &expected_lq, false, false);
        }
        std::thread::scope(|scope| {
            let calls: Vec<_> = (0..4)
                .map(|_| {
                    let lazy = lazy.clone();
                    scope.spawn(move || {
                        let qr = lazy.qr_compact().unwrap();
                        let lq = lazy.lq_full().unwrap();
                        (qr, lq)
                    })
                })
                .collect();
            for call in calls {
                let (qr, lq) = call.join().unwrap();
                assert_qr_lq_factors(&source, &target, &qr, &expected_qr, true, true);
                assert_qr_lq_factors(&source, &target, &lq, &expected_lq, false, false);
            }
        });
        assert_eq!(materialized_adjoint_builds(&lazy), 0);
        let TypedTensorRepr::Adjoint(view) = &lazy.repr else {
            unreachable!()
        };
        assert!(view.materialized.get().is_none());
    }

    fn assert_full_qr_lq_late_failure(qr: bool) {
        let runtime = Runtime::builder()
            .with_dense_executor(Box::new(FailSecondQr::default()))
            .build()
            .unwrap();
        let provider = Arc::new(U1FusionRule);
        let leg = GradedSpace::try_new(
            provider,
            [(U1Irrep::new(0), 2), (U1Irrep::new(1), 2)],
            false,
        )
        .unwrap();
        let source = TensorMap::from_block_fn(&runtime, [&leg], [&leg], |_, indices| {
            (indices.iter().sum::<usize>() + 1) as f64
        })
        .unwrap();
        let before = source.data().to_vec();
        let lazy = source.adjoint().unwrap();

        let result = if qr { lazy.qr_full() } else { lazy.lq_full() };
        assert!(matches!(result, Err(Error::Operation(_))));
        assert_eq!(source.data(), before);
        assert_eq!(materialized_adjoint_builds(&lazy), 0);
        let TypedTensorRepr::Adjoint(view) = &lazy.repr else {
            unreachable!()
        };
        assert!(view.materialized.get().is_none());
    }

    #[test]
    fn full_qr_lq_late_failure_leaves_parent_and_adjoint_cache_unchanged() {
        assert_full_qr_lq_late_failure(true);
        assert_full_qr_lq_late_failure(false);
    }

    fn assert_truncated_svd_reads_parent<R, D>(source: &TensorMap<R, D>)
    where
        R: MultiplicityFreeRigidSymbols<Scalar = f64> + CheckedFusionAlgebra + SectorCodec,
        D: TensorScalar + core::fmt::Debug,
    {
        let eager = eager_adjoint_oracle(source);
        let lazy = source.adjoint().unwrap();
        let truncation = Truncation::rank(1);
        let actual = lazy.svd_trunc(&truncation).unwrap();
        let expected = eager.svd_trunc(&truncation).unwrap();

        assert_eq!(actual.singular_values, expected.singular_values);
        assert_eq!(actual.error, expected.error);
        assert_eq!(materialized_adjoint_builds(&lazy), 0);
        let TypedTensorRepr::Adjoint(view) = &lazy.repr else {
            unreachable!()
        };
        assert!(view.materialized.get().is_none());
        for (actual, expected) in [
            (&actual.u, &expected.u),
            (&actual.s, &expected.s),
            (&actual.vh, &expected.vh),
        ] {
            assert_eq!(
                actual.logical_space().space(),
                expected.logical_space().space()
            );
            assert!(Arc::ptr_eq(
                actual.logical_space().provider_arc(),
                source.logical_space().provider_arc()
            ));
            assert!(actual
                .data()
                .iter()
                .zip(expected.data())
                .all(|(&left, &right)| {
                    (left.widen_complex() - right.widen_complex()).norm() < 1e-12
                }));
        }
        assert!(actual.u.is_isometric(1e-12).unwrap());
        assert_eq!(materialized_adjoint_builds(&lazy), 0);
        assert!(view.materialized.get().is_none());
    }

    #[test]
    fn truncated_svd_reads_the_parent_without_materializing_the_adjoint() {
        // What: truncation selection, error, factor gauge, and typed provider
        // authority match an eager logical-adjoint oracle without an input copy.
        let u1 = u1_lazy_fixture();
        let su2 = su2_lazy_fixture();
        assert_truncated_svd_reads_parent(&u1);
        assert_truncated_svd_reads_parent(&genuinely_complex(&u1));
        assert_truncated_svd_reads_parent(&su2);
        assert_truncated_svd_reads_parent(&genuinely_complex(&su2));
    }

    #[test]
    fn rejected_truncation_does_not_materialize_the_adjoint() {
        // What: the parent-native path preserves typed truncation errors
        // without publishing the logical-adjoint payload first.
        let source = u1_lazy_fixture();
        let foreign = GradedSpace::try_new(
            Arc::new(SU2FusionRule),
            [(SU2Irrep::from_twice_spin(0), 1)],
            false,
        )
        .unwrap();
        let lazy = source.adjoint().unwrap();
        assert!(matches!(
            lazy.svd_trunc(&Truncation::space(foreign.truncspace())),
            Err(Error::Operation(_))
        ));
        assert_eq!(materialized_adjoint_builds(&lazy), 0);
        let TypedTensorRepr::Adjoint(view) = &lazy.repr else {
            unreachable!()
        };
        assert!(view.materialized.get().is_none());
    }

    fn eager_adjoint_oracle<R, D>(source: &TensorMap<R, D>) -> TensorMap<R, D>
    where
        R: MultiplicityFreeRigidSymbols<Scalar = f64> + CheckedFusionAlgebra + SectorCodec,
        D: TensorScalar,
    {
        let (space, data) =
            tenet_tensors::adjoint_bound_dyn(source.logical_space(), source.data()).unwrap();
        TensorMap {
            runtime: source.runtime.clone(),
            repr: owned_repr(TypedTensorBody::dense(space, data)),
        }
    }

    #[test]
    fn inverse_twist_and_flip_keep_the_lazy_adjoint_materialization_boundary() {
        let source = fz2_fixture();
        let eager = eager_adjoint_oracle(&source);

        let lazy_twist = source.adjoint().unwrap();
        assert_eq!(
            lazy_twist.twist_inverse(&[0]).unwrap().data(),
            eager.twist_inverse(&[0]).unwrap().data()
        );
        assert_eq!(materialized_adjoint_builds(&lazy_twist), 0);

        let lazy_flip = source.adjoint().unwrap();
        let actual = lazy_flip.flip_inverse(&[1]).unwrap();
        let expected = eager.flip_inverse(&[1]).unwrap();
        assert_eq!(actual.data(), expected.data());
        assert_eq!(
            actual.logical_space().space(),
            expected.logical_space().space()
        );
        assert_eq!(materialized_adjoint_builds(&lazy_flip), 0);
    }

    #[test]
    fn simple_lazy_observers_and_owned_outputs_do_not_publish() {
        let source = u1_matrix_fixture([(0, 2)], [(0, 2)]);
        let eager = eager_adjoint_oracle(&source);
        let lazy = source.adjoint().unwrap();

        assert_eq!(lazy.is_diagonal(0.0), eager.is_diagonal(0.0));
        assert!(lazy.is_diagonal(-1.0).is_err());
        assert_eq!(lazy.to_c64().data(), eager.to_c64().data());
        assert_eq!(lazy.to_c64().re().data(), eager.to_c64().re().data());
        assert_eq!(lazy.to_c64().im().data(), eager.to_c64().im().data());
        assert_eq!(
            lazy.insert_left_unit(0, false).unwrap().data(),
            eager.insert_left_unit(0, false).unwrap().data()
        );
        assert_eq!(materialized_adjoint_builds(&lazy), 0);

        let scalar = source.trace_pairs(&[(0, 1)]).unwrap();
        let scalar_eager = eager_adjoint_oracle(&scalar);
        let scalar_lazy = scalar.adjoint().unwrap();
        assert_eq!(
            scalar_lazy.scalar().unwrap(),
            scalar_eager.scalar().unwrap()
        );
        assert_eq!(materialized_adjoint_builds(&scalar_lazy), 0);

        let complex = genuinely_complex(&source);
        let eager_complex = eager_adjoint_oracle(&complex);
        let lazy_complex = complex.adjoint().unwrap();
        assert_eq!(lazy_complex.re().data(), eager_complex.re().data());
        assert_eq!(lazy_complex.im().data(), eager_complex.im().data());
        assert_eq!(materialized_adjoint_builds(&lazy_complex), 0);

        let clone = lazy.clone();
        assert_eq!(lazy.data(), eager.data());
        assert_eq!(clone.data(), eager.data());
        assert_eq!(materialized_adjoint_builds(&lazy), 1);
    }

    #[test]
    fn lazy_otimes_orientations_and_deligne_inputs_stay_cold() {
        let lhs = u1_lazy_fixture();
        let rhs = lhs.scale(2.0);
        let eager_lhs = eager_adjoint_oracle(&lhs);
        let eager_rhs = eager_adjoint_oracle(&rhs);
        let lazy_lhs = lhs.adjoint().unwrap();
        let lazy_rhs = rhs.adjoint().unwrap();

        for (actual, expected) in [
            (lhs.otimes(&rhs).unwrap(), lhs.otimes(&rhs).unwrap()),
            (
                lhs.otimes(&lazy_rhs).unwrap(),
                lhs.otimes(&eager_rhs).unwrap(),
            ),
            (
                lazy_lhs.otimes(&rhs).unwrap(),
                eager_lhs.otimes(&rhs).unwrap(),
            ),
            (
                lazy_lhs.otimes(&lazy_rhs).unwrap(),
                eager_lhs.otimes(&eager_rhs).unwrap(),
            ),
        ] {
            assert_eq!(actual.data(), expected.data());
            assert!(Arc::ptr_eq(
                actual.logical_space().provider_arc(),
                lhs.logical_space().provider_arc()
            ));
        }
        assert_eq!(materialized_adjoint_builds(&lazy_lhs), 0);
        assert_eq!(materialized_adjoint_builds(&lazy_rhs), 0);

        let deligne_lhs = u1_matrix_fixture([(0, 2)], [(0, 2)]);
        let deligne_rhs = deligne_lhs.scale(3.0);
        let eager_deligne_lhs = eager_adjoint_oracle(&deligne_lhs);
        let eager_deligne_rhs = eager_adjoint_oracle(&deligne_rhs);
        let lazy_deligne_lhs = deligne_lhs.adjoint().unwrap();
        let lazy_deligne_rhs = deligne_rhs.adjoint().unwrap();
        let product = Arc::new(U1FusionRule.product(U1FusionRule));
        let actual = lazy_deligne_lhs
            .deligne_product(&lazy_deligne_rhs, Arc::clone(&product))
            .unwrap();
        let expected = eager_deligne_lhs
            .deligne_product(&eager_deligne_rhs, product)
            .unwrap();
        assert_eq!(actual.data(), expected.data());
        assert_eq!(materialized_adjoint_builds(&lazy_deligne_lhs), 0);
        assert_eq!(materialized_adjoint_builds(&lazy_deligne_rhs), 0);
    }

    #[test]
    fn absorb_uses_operation_local_logical_payloads_without_warming_lazy_inputs() {
        let destination_parent = genuinely_complex(&u1_lazy_fixture());
        let source_parent = destination_parent.scale(num_complex::Complex64::new(2.0, -1.0));
        let eager_destination = eager_adjoint_oracle(&destination_parent);
        let eager_source = eager_adjoint_oracle(&source_parent);
        let expected = eager_destination.absorb(&eager_source).unwrap();
        let lazy_destination = destination_parent.adjoint().unwrap();
        let lazy_source = source_parent.adjoint().unwrap();

        let actual = lazy_destination.absorb(&lazy_source).unwrap();
        assert_typed_map_close(&actual, &expected, 1e-12);
        assert_eq!(materialized_adjoint_builds(&lazy_destination), 0);
        assert_eq!(materialized_adjoint_builds(&lazy_source), 0);
    }

    fn assert_parent_native_transform<R, D>(
        source: &TensorMap<R, D>,
        operation: impl Fn(&TensorMap<R, D>) -> Result<TensorMap<R, D>, Error>,
    ) where
        R: MultiplicityFreeRigidSymbols<Scalar = f64> + CheckedFusionAlgebra + SectorCodec,
        D: TensorScalar + core::fmt::Debug,
    {
        let lazy = source.adjoint().unwrap();
        let eager = eager_adjoint_oracle(source);
        let actual = operation(&lazy).unwrap();
        let expected = operation(&eager).unwrap();
        assert_eq!(materialized_adjoint_builds(&lazy), 0);
        let TypedTensorRepr::Adjoint(view) = &actual.repr else {
            panic!("a transformed lazy adjoint must remain parent-backed");
        };
        assert!(view.materialized.get().is_none());
        assert_eq!(
            actual.logical_space().space(),
            expected.logical_space().space()
        );
        assert!(Arc::ptr_eq(
            actual.logical_space().provider_arc(),
            source.logical_space().provider_arc()
        ));
        assert_eq!(actual.data().len(), expected.data().len());
        assert!(actual
            .data()
            .iter()
            .zip(expected.data())
            .all(|(&actual, &expected)| {
                (actual.widen_complex() - expected.widen_complex()).norm() < 1e-12
            }));
        assert_eq!(materialized_adjoint_builds(&lazy), 0);
    }

    fn assert_parent_native_transform_suite<R, D>(source: &TensorMap<R, D>)
    where
        R: MultiplicityFreeRigidSymbols<Scalar = f64> + CheckedFusionAlgebra + SectorCodec,
        D: TensorScalar + core::fmt::Debug,
    {
        assert_parent_native_transform(source, |tensor| tensor.permute(&[2, 0], &[1]));
        assert_parent_native_transform(source, |tensor| tensor.braid(&[2, 0], &[1], &[17, 3, 11]));
        assert_parent_native_transform(source, TensorMap::transpose);
        assert_parent_native_transform(source, |tensor| tensor.transpose_axes(&[2, 1], &[0]));
        assert_parent_native_transform(source, |tensor| tensor.repartition(2));
    }

    #[test]
    fn nonidentity_adjoint_transforms_stay_parent_native_for_unique_simple_and_both_dtypes() {
        let u1 = u1_lazy_fixture();
        let su2 = su2_lazy_fixture();
        let u1_c64 = u1.to_c64().scale(num_complex::Complex64::new(1.0, 2.0));
        let su2_c64 = su2.to_c64().scale(num_complex::Complex64::new(1.0, 2.0));
        assert_parent_native_transform_suite(&u1);
        assert_parent_native_transform_suite(&u1_c64);
        assert_parent_native_transform_suite(&su2);
        assert_parent_native_transform_suite(&su2_c64);
    }

    fn assert_close<D: TensorScalar>(actual: D, expected: D) {
        assert!((actual.widen_complex() - expected.widen_complex()).norm() < 1e-12);
    }

    fn assert_parent_native_elementwise<R, D>(source: &TensorMap<R, D>, alpha: D, beta: D)
    where
        R: MultiplicityFreeRigidSymbols<Scalar = f64> + CheckedFusionAlgebra + SectorCodec,
        D: TensorScalar + core::fmt::Debug,
    {
        let lazy = source.adjoint().unwrap();
        let eager = eager_adjoint_oracle(source);

        let add = lazy.add(&eager, alpha, beta).unwrap();
        let expected_add = eager.add(&eager, alpha, beta).unwrap();
        assert!(add
            .data()
            .iter()
            .zip(expected_add.data())
            .all(|(&actual, &expected)| {
                (actual.widen_complex() - expected.widen_complex()).norm() < 1e-12
            }));
        let add_both = lazy.add(&lazy, alpha, beta).unwrap();
        assert!(add_both
            .data()
            .iter()
            .zip(expected_add.data())
            .all(|(&actual, &expected)| {
                (actual.widen_complex() - expected.widen_complex()).norm() < 1e-12
            }));
        let add_rhs = eager.add(&lazy, alpha, beta).unwrap();
        assert!(add_rhs
            .data()
            .iter()
            .zip(expected_add.data())
            .all(|(&actual, &expected)| {
                (actual.widen_complex() - expected.widen_complex()).norm() < 1e-12
            }));

        let scaled = lazy.scale(alpha);
        let expected_scaled = eager.scale(alpha);
        let TypedTensorRepr::Adjoint(scaled_view) = &scaled.repr else {
            panic!("scaling a lazy adjoint must remain parent-backed");
        };
        assert!(scaled_view.materialized.get().is_none());
        assert!(scaled
            .data()
            .iter()
            .zip(expected_scaled.data())
            .all(|(&actual, &expected)| {
                (actual.widen_complex() - expected.widen_complex()).norm() < 1e-12
            }));

        assert_close(lazy.inner(&eager).unwrap(), eager.inner(&eager).unwrap());
        assert_close(eager.inner(&lazy).unwrap(), eager.inner(&eager).unwrap());
        assert_close(lazy.inner(&lazy).unwrap(), eager.inner(&eager).unwrap());
        assert!((lazy.norm().unwrap() - eager.norm().unwrap()).abs() < 1e-12);
        assert!((lazy.norm_inf().unwrap() - eager.norm_inf().unwrap()).abs() < 1e-12);
        assert!((lazy.norm_p(1.5).unwrap() - eager.norm_p(1.5).unwrap()).abs() < 1e-12);

        let normalized = lazy.normalize().unwrap();
        let TypedTensorRepr::Adjoint(normalized_view) = &normalized.repr else {
            panic!("normalizing a lazy adjoint must remain parent-backed");
        };
        assert!(normalized_view.materialized.get().is_none());
        assert!((normalized.norm().unwrap() - 1.0).abs() < 1e-12);
        assert_eq!(materialized_adjoint_builds(&lazy), 0);
    }

    #[test]
    fn adjoint_elementwise_and_reductions_stay_parent_native() {
        let u1 = u1_lazy_fixture();
        let su2 = su2_lazy_fixture();
        let u1_c64 = u1.to_c64().scale(num_complex::Complex64::new(1.0, 2.0));
        let su2_c64 = su2.to_c64().scale(num_complex::Complex64::new(1.0, 2.0));
        assert_parent_native_elementwise(&u1, 2.0, -0.5);
        assert_parent_native_elementwise(&su2, 2.0, -0.5);
        assert_parent_native_elementwise(
            &u1_c64,
            num_complex::Complex64::new(0.5, 1.0),
            num_complex::Complex64::new(-0.25, 0.75),
        );
        assert_parent_native_elementwise(
            &su2_c64,
            num_complex::Complex64::new(0.5, 1.0),
            num_complex::Complex64::new(-0.25, 0.75),
        );
    }

    fn assert_parent_native_trace_pairs<R, D>(source: &TensorMap<R, D>)
    where
        R: MultiplicityFreeRigidSymbols<Scalar = f64> + CheckedFusionAlgebra + SectorCodec,
        D: TensorScalar + core::fmt::Debug,
    {
        let lazy = source.adjoint().unwrap();
        let eager = eager_adjoint_oracle(source);
        let expected = eager.trace_pairs(&[(0, 1)]).unwrap();
        let actual = lazy.trace_pairs(&[(0, 1)]).unwrap();
        assert_eq!(
            actual.logical_space().space(),
            expected.logical_space().space()
        );
        assert!(Arc::ptr_eq(
            actual.logical_space().provider_arc(),
            source.logical_space().provider_arc()
        ));
        assert!(actual
            .data()
            .iter()
            .zip(expected.data())
            .all(|(&actual, &expected)| {
                (actual.widen_complex() - expected.widen_complex()).norm() < 1e-12
            }));
        assert!(lazy.trace_pairs(&[(0, 3)]).is_err());
        assert!(lazy.trace_pairs(&[(0, 0)]).is_err());
        assert_eq!(materialized_adjoint_builds(&lazy), 0);
    }

    #[test]
    fn adjoint_trace_pairs_stays_parent_native() {
        let runtime = Runtime::builder().dense_threads(1).build().unwrap();
        let u1_provider = Arc::new(U1FusionRule);
        let u1_traced = GradedSpace::try_new(
            Arc::clone(&u1_provider),
            [(U1Irrep::new(0), 2), (U1Irrep::new(1), 1)],
            false,
        )
        .unwrap();
        let u1_open = GradedSpace::try_new(
            u1_provider,
            [(U1Irrep::new(-1), 1), (U1Irrep::new(0), 1)],
            true,
        )
        .unwrap();
        let u1 = TensorMap::from_block_fn(
            &runtime,
            [&u1_traced, &u1_open],
            [&u1_traced],
            |_, indices| indices.iter().sum::<usize>() as f64 + 1.0,
        )
        .unwrap();
        let su2_provider = Arc::new(SU2FusionRule);
        let su2_traced = GradedSpace::try_new(
            Arc::clone(&su2_provider),
            [
                (SU2Irrep::from_twice_spin(0), 2),
                (SU2Irrep::from_twice_spin(1), 1),
            ],
            false,
        )
        .unwrap();
        let su2_open = GradedSpace::try_new(
            su2_provider,
            [
                (SU2Irrep::from_twice_spin(0), 1),
                (SU2Irrep::from_twice_spin(1), 1),
            ],
            true,
        )
        .unwrap();
        let su2 = TensorMap::from_block_fn(
            &runtime,
            [&su2_traced, &su2_open, &su2_open],
            [&su2_traced],
            |_, indices| indices.iter().sum::<usize>() as f64 + 1.0,
        )
        .unwrap();
        assert!(su2.block_count() > 1);
        assert!(
            eager_adjoint_oracle(&su2)
                .trace_pairs(&[(0, 1)])
                .unwrap()
                .block_count()
                > 1
        );
        let u1_c64 = u1.to_c64().scale(num_complex::Complex64::new(1.0, 2.0));
        let su2_c64 = su2.to_c64().scale(num_complex::Complex64::new(1.0, 2.0));
        assert_parent_native_trace_pairs(&u1);
        assert_parent_native_trace_pairs(&u1_c64);
        assert_parent_native_trace_pairs(&su2);
        assert_parent_native_trace_pairs(&su2_c64);
    }

    fn assert_parent_native_tr<R, D>(source: &TensorMap<R, D>)
    where
        R: MultiplicityFreeRigidSymbols<Scalar = f64> + CheckedFusionAlgebra + SectorCodec,
        D: TensorScalar + core::fmt::Debug,
    {
        let lazy = source.adjoint().unwrap();
        let eager = eager_adjoint_oracle(source);
        assert_close(lazy.tr().unwrap(), eager.tr().unwrap());
        assert_eq!(materialized_adjoint_builds(&lazy), 0);
    }

    #[test]
    fn adjoint_positive_trace_conjugates_the_parent_without_materializing() {
        let u1_source = u1_lazy_fixture();
        let u1_leg = u1_source.codomain_spaces().remove(0);
        let u1 =
            TensorMap::from_block_fn(u1_source.runtime(), [&u1_leg], [&u1_leg], |_, indices| {
                (indices[0] + 2 * indices[1]) as f64 + 1.0
            })
            .unwrap();
        let su2_source = su2_lazy_fixture();
        let su2_leg = su2_source.codomain_spaces().remove(0);
        let su2 = TensorMap::from_block_fn(
            su2_source.runtime(),
            [&su2_leg],
            [&su2_leg],
            |_, indices| (indices[0] + 2 * indices[1]) as f64 + 1.0,
        )
        .unwrap();
        let u1_c64 = u1.to_c64().scale(num_complex::Complex64::new(1.0, 2.0));
        let su2_c64 = su2.to_c64().scale(num_complex::Complex64::new(1.0, 2.0));
        assert_parent_native_tr(&u1);
        assert_parent_native_tr(&u1_c64);
        assert_parent_native_tr(&su2);
        assert_parent_native_tr(&su2_c64);
    }

    fn assert_parent_native_contract_and_compose<R, D>(source: &TensorMap<R, D>)
    where
        R: MultiplicityFreeRigidSymbols<Scalar = f64> + CheckedFusionAlgebra + SectorCodec,
        D: TensorScalar + core::fmt::Debug,
    {
        let lazy = source.adjoint().unwrap();
        let eager = eager_adjoint_oracle(source);
        for (lhs, rhs, expected_lhs, expected_rhs) in [
            (&lazy, &eager, &eager, &eager),
            (&eager, &lazy, &eager, &eager),
            (&lazy, &lazy, &eager, &eager),
        ] {
            let actual = lhs.contract(rhs, &[1], &[0], &[1, 0]).unwrap();
            let expected = expected_lhs
                .contract(expected_rhs, &[1], &[0], &[1, 0])
                .unwrap();
            assert_eq!(
                actual.logical_space().space(),
                expected.logical_space().space()
            );
            assert!(Arc::ptr_eq(
                actual.logical_space().provider_arc(),
                lhs.logical_space().provider_arc()
            ));
            assert!(actual
                .data()
                .iter()
                .zip(expected.data())
                .all(|(&actual, &expected)| {
                    (actual.widen_complex() - expected.widen_complex()).norm() < 1e-12
                }));

            let actual = lhs.compose(rhs).unwrap();
            let expected = expected_lhs.compose(expected_rhs).unwrap();
            assert_eq!(
                actual.logical_space().space(),
                expected.logical_space().space()
            );
            assert!(Arc::ptr_eq(
                actual.logical_space().provider_arc(),
                lhs.logical_space().provider_arc()
            ));
            assert!(actual
                .data()
                .iter()
                .zip(expected.data())
                .all(|(&actual, &expected)| {
                    (actual.widen_complex() - expected.widen_complex()).norm() < 1e-12
                }));
        }
        assert!(lazy.contract(&eager, &[2], &[0], &[0, 1]).is_err());
        assert_eq!(materialized_adjoint_builds(&lazy), 0);
    }

    #[test]
    fn adjoint_contract_and_compose_stay_parent_native() {
        let u1_source = u1_lazy_fixture();
        let u1_leg = u1_source.codomain_spaces().remove(0);
        let u1 =
            TensorMap::from_block_fn(u1_source.runtime(), [&u1_leg], [&u1_leg], |_, indices| {
                (indices[0] + 2 * indices[1]) as f64 + 1.0
            })
            .unwrap();
        let su2_source = su2_lazy_fixture();
        let su2_leg = su2_source.codomain_spaces().remove(0);
        let su2 = TensorMap::from_block_fn(
            su2_source.runtime(),
            [&su2_leg],
            [&su2_leg],
            |_, indices| (indices[0] + 2 * indices[1]) as f64 + 1.0,
        )
        .unwrap();
        let u1_c64 = u1.to_c64().scale(num_complex::Complex64::new(1.0, 2.0));
        let su2_c64 = su2.to_c64().scale(num_complex::Complex64::new(1.0, 2.0));
        assert_parent_native_contract_and_compose(&u1);
        assert_parent_native_contract_and_compose(&u1_c64);
        assert_parent_native_contract_and_compose(&su2);
        assert_parent_native_contract_and_compose(&su2_c64);
    }

    fn assert_rank_three_su2_contract_and_compose<D>(source: &TensorMap<SU2FusionRule, D>)
    where
        D: TensorScalar + core::fmt::Debug,
    {
        assert!(source.block_count() > 1);
        let lazy = source.adjoint().unwrap();
        let eager = eager_adjoint_oracle(source);

        let actual = lazy.contract(source, &[2, 1], &[1, 0], &[1, 0]).unwrap();
        let expected = eager.contract(source, &[2, 1], &[1, 0], &[1, 0]).unwrap();
        assert!(actual.block_count() > 1);
        assert_eq!(
            actual.logical_space().space(),
            expected.logical_space().space()
        );
        assert!(Arc::ptr_eq(
            actual.logical_space().provider_arc(),
            source.logical_space().provider_arc()
        ));
        assert!(actual
            .data()
            .iter()
            .zip(expected.data())
            .all(|(&actual, &expected)| {
                (actual.widen_complex() - expected.widen_complex()).norm() < 1e-12
            }));

        let actual = lazy.compose(source).unwrap();
        let expected = eager.compose(source).unwrap();
        assert!(actual.block_count() > 1);
        assert_eq!(
            actual.logical_space().space(),
            expected.logical_space().space()
        );
        assert!(Arc::ptr_eq(
            actual.logical_space().provider_arc(),
            source.logical_space().provider_arc()
        ));
        assert!(actual
            .data()
            .iter()
            .zip(expected.data())
            .all(|(&actual, &expected)| {
                (actual.widen_complex() - expected.widen_complex()).norm() < 1e-12
            }));
        assert_eq!(materialized_adjoint_builds(&lazy), 0);
    }

    #[test]
    fn rank_three_su2_adjoint_contract_and_compose_use_oriented_recoupling() {
        let source = su2_lazy_fixture();
        let complex = source.to_c64().scale(num_complex::Complex64::new(1.0, 2.0));
        assert_rank_three_su2_contract_and_compose(&source);
        assert_rank_three_su2_contract_and_compose(&complex);
    }

    fn assert_fermionic_contract_and_compose_semantics<D>(
        source: &TensorMap<FermionParityFusionRule, D>,
    ) where
        D: TensorScalar + core::fmt::Debug,
    {
        let lazy = source.adjoint().unwrap();
        let eager = eager_adjoint_oracle(source);
        let contract = lazy.contract(source, &[1], &[0], &[0, 1]).unwrap();
        let expected_contract = eager.contract(source, &[1], &[0], &[0, 1]).unwrap();
        let compose = lazy.compose(source).unwrap();
        let expected_compose = eager.compose(source).unwrap();
        assert!(contract.data().iter().zip(expected_contract.data()).all(
            |(&actual, &expected)| {
                (actual.widen_complex() - expected.widen_complex()).norm() < 1e-12
            }
        ));
        assert!(compose
            .data()
            .iter()
            .zip(expected_compose.data())
            .all(|(&actual, &expected)| {
                (actual.widen_complex() - expected.widen_complex()).norm() < 1e-12
            }));
        assert!(contract
            .data()
            .iter()
            .zip(compose.data())
            .any(|(&contract, &compose)| {
                (contract.widen_complex() - compose.widen_complex()).norm() > 1e-12
            }));
        assert!(Arc::ptr_eq(
            contract.logical_space().provider_arc(),
            source.logical_space().provider_arc()
        ));
        assert!(Arc::ptr_eq(
            compose.logical_space().provider_arc(),
            source.logical_space().provider_arc()
        ));
        assert_eq!(materialized_adjoint_builds(&lazy), 0);
    }

    #[test]
    fn fermionic_lazy_contract_keeps_the_supertrace_distinct_from_compose() {
        let runtime = Runtime::builder().dense_threads(1).build().unwrap();
        let provider = Arc::new(FermionParityFusionRule);
        let leg =
            GradedSpace::try_new(provider, [(Z2Irrep::EVEN, 2), (Z2Irrep::ODD, 2)], true).unwrap();
        let source = TensorMap::from_block_fn(&runtime, [&leg], [&leg], |trees, indices| {
            let sector = trees.codomain_uncoupled[0];
            let parity_weight = if sector == Z2Irrep::ODD { 3.0 } else { 1.0 };
            parity_weight * (indices[0] + 2 * indices[1] + 1) as f64
        })
        .unwrap();
        let complex = source.to_c64().scale(num_complex::Complex64::new(1.0, 2.0));
        assert_fermionic_contract_and_compose_semantics(&source);
        assert_fermionic_contract_and_compose_semantics(&complex);
    }

    fn assert_same_error(actual: Error, expected: Error) {
        assert_eq!(
            core::mem::discriminant(&actual),
            core::mem::discriminant(&expected)
        );
        assert_eq!(actual.to_string(), expected.to_string());
    }

    #[test]
    fn lazy_contract_preserves_validation_precedence_without_materializing() {
        let source = {
            let fixture = u1_lazy_fixture();
            let leg = fixture.codomain_spaces().remove(0);
            TensorMap::from_block_fn(fixture.runtime(), [&leg], [&leg], |_, indices| {
                (indices[0] + 2 * indices[1] + 1) as f64
            })
            .unwrap()
        };
        let eager = eager_adjoint_oracle(&source);
        for (lhs_axes, rhs_axes, output_axes) in [
            (&[2][..], &[0][..], &[0, 1][..]),
            (&[1, 1][..], &[0, 0][..], &[][..]),
        ] {
            let lazy = source.adjoint().unwrap();
            assert_same_error(
                lazy.contract(&eager, lhs_axes, rhs_axes, output_axes)
                    .unwrap_err(),
                eager
                    .contract(&eager, lhs_axes, rhs_axes, output_axes)
                    .unwrap_err(),
            );
            assert_eq!(materialized_adjoint_builds(&lazy), 0);
        }

        let bad_leg = GradedSpace::try_new(
            Arc::clone(source.logical_space().provider_arc()),
            [(U1Irrep::new(7), 1)],
            false,
        )
        .unwrap();
        let bad =
            TensorMap::from_block_fn(source.runtime(), [&bad_leg], [&bad_leg], |_, _| 1.0).unwrap();
        let lazy = source.adjoint().unwrap();
        assert_same_error(
            lazy.contract(&bad, &[1], &[0], &[0, 0]).unwrap_err(),
            eager.contract(&bad, &[1], &[0], &[0, 0]).unwrap_err(),
        );
        assert_eq!(materialized_adjoint_builds(&lazy), 0);

        let other_runtime = Runtime::builder().dense_threads(1).build().unwrap();
        let source_leg = source.codomain_spaces().remove(0);
        let other =
            TensorMap::from_block_fn(&other_runtime, [&source_leg], [&source_leg], |_, _| 1.0)
                .unwrap();
        let lazy = source.adjoint().unwrap();
        assert_same_error(
            lazy.contract(&other, &[1], &[0], &[0, 1]).unwrap_err(),
            eager.contract(&other, &[1], &[0], &[0, 1]).unwrap_err(),
        );
        assert_eq!(materialized_adjoint_builds(&lazy), 0);

        let runtime = Runtime::builder().dense_threads(1).build().unwrap();
        let z2 = Arc::new(ZNFusionRule::new(2).unwrap());
        let z3 = Arc::new(ZNFusionRule::new(3).unwrap());
        let z2_leg = GradedSpace::try_new(Arc::clone(&z2), [(z2.irrep(0), 2)], false).unwrap();
        let z3_leg = GradedSpace::try_new(Arc::clone(&z3), [(z3.irrep(0), 2)], false).unwrap();
        let z2_tensor =
            TensorMap::from_block_fn(&runtime, [&z2_leg], [&z2_leg], |_, _| 1.0).unwrap();
        let z3_tensor =
            TensorMap::from_block_fn(&runtime, [&z3_leg], [&z3_leg], |_, _| 1.0).unwrap();
        let eager = eager_adjoint_oracle(&z2_tensor);
        let lazy = z2_tensor.adjoint().unwrap();
        assert_same_error(
            lazy.contract(&z3_tensor, &[1], &[0], &[0, 1]).unwrap_err(),
            eager.contract(&z3_tensor, &[1], &[0], &[0, 1]).unwrap_err(),
        );
        assert_eq!(materialized_adjoint_builds(&lazy), 0);
    }

    #[test]
    fn lazy_binary_outputs_keep_the_lhs_provider_allocation() {
        let runtime = Runtime::builder().dense_threads(1).build().unwrap();
        let lhs_provider = Arc::new(U1FusionRule);
        let rhs_provider = Arc::new(U1FusionRule);
        assert!(!Arc::ptr_eq(&lhs_provider, &rhs_provider));
        let lhs_leg = GradedSpace::try_new(
            Arc::clone(&lhs_provider),
            [(U1Irrep::new(0), 2), (U1Irrep::new(1), 1)],
            false,
        )
        .unwrap();
        let rhs_leg = GradedSpace::try_new(
            Arc::clone(&rhs_provider),
            [(U1Irrep::new(0), 2), (U1Irrep::new(1), 1)],
            false,
        )
        .unwrap();
        let lhs = TensorMap::from_block_fn(&runtime, [&lhs_leg], [&lhs_leg], |_, indices| {
            (indices[0] + 2 * indices[1] + 1) as f64
        })
        .unwrap();
        let rhs = TensorMap::from_block_fn(&runtime, [&rhs_leg], [&rhs_leg], |_, indices| {
            (2 * indices[0] + indices[1] + 1) as f64
        })
        .unwrap();
        let lazy = lhs.adjoint().unwrap();
        let eager = eager_adjoint_oracle(&lhs);
        let rhs_lazy = rhs.adjoint().unwrap();
        let rhs_eager = eager_adjoint_oracle(&rhs);
        for (actual, expected) in [
            (
                lazy.contract(&rhs_lazy, &[1], &[0], &[0, 1]).unwrap(),
                eager.contract(&rhs_eager, &[1], &[0], &[0, 1]).unwrap(),
            ),
            (
                lazy.compose(&rhs_lazy).unwrap(),
                eager.compose(&rhs_eager).unwrap(),
            ),
        ] {
            assert_eq!(actual.data(), expected.data());
            assert!(Arc::ptr_eq(
                actual.logical_space().provider_arc(),
                &lhs_provider
            ));
            assert!(!Arc::ptr_eq(
                actual.logical_space().provider_arc(),
                &rhs_provider
            ));
        }
        assert_eq!(materialized_adjoint_builds(&lazy), 0);
        assert_eq!(materialized_adjoint_builds(&rhs_lazy), 0);
    }

    #[test]
    fn mixed_compact_add_does_not_materialize_the_lazy_operand() {
        let runtime = Runtime::builder().dense_threads(1).build().unwrap();
        let provider = Arc::new(U1FusionRule);
        let bond = GradedSpace::try_new(provider, [(U1Irrep::new(0), 2)], false).unwrap();
        let dense = TensorMap::from_block_fn(&runtime, [&bond], [&bond], |_, indices| {
            (indices[0] + 2 * indices[1]) as f64
        })
        .unwrap();
        let lazy = dense.adjoint().unwrap();
        let eager = eager_adjoint_oracle(&dense);
        let diagonal = TensorMap::diagonal(
            &runtime,
            &bond,
            [SectorSpectrum {
                sector: U1Irrep::new(0),
                values: vec![2.0, 3.0],
            }],
        )
        .unwrap();
        let actual = diagonal.add(&lazy, 0.5, -2.0).unwrap();
        let expected = diagonal.add(&eager, 0.5, -2.0).unwrap();
        assert_eq!(actual.data(), expected.data());
        let reverse = lazy.add(&diagonal, -2.0, 0.5).unwrap();
        let expected_reverse = eager.add(&diagonal, -2.0, 0.5).unwrap();
        assert_eq!(reverse.data(), expected_reverse.data());
        assert_eq!(materialized_adjoint_builds(&lazy), 0);
        assert!(owned(&diagonal).dense_cache.get().is_none());

        for (lhs, rhs, eager_lhs, eager_rhs) in [
            (&diagonal, &lazy, &diagonal, &eager),
            (&lazy, &diagonal, &eager, &diagonal),
        ] {
            let actual = lhs.compose(rhs).unwrap();
            let expected = eager_lhs.compose(eager_rhs).unwrap();
            assert_eq!(actual.data(), expected.data());
            let actual = lhs.contract(rhs, &[1], &[0], &[0, 1]).unwrap();
            let expected = eager_lhs.contract(eager_rhs, &[1], &[0], &[0, 1]).unwrap();
            assert_eq!(actual.data(), expected.data());
        }
        assert_eq!(materialized_adjoint_builds(&lazy), 0);
        assert!(owned(&diagonal).dense_cache.get().is_none());

        assert_close(
            diagonal.inner(&lazy).unwrap(),
            diagonal.inner(&eager).unwrap(),
        );
        assert_close(
            lazy.inner(&diagonal).unwrap(),
            eager.inner(&diagonal).unwrap(),
        );
        assert_eq!(materialized_adjoint_builds(&lazy), 0);
        assert!(owned(&diagonal).dense_cache.get().is_some());
    }

    #[test]
    fn identity_transforms_preserve_the_cold_lazy_view() {
        let adjoint = u1_lazy_fixture().adjoint().unwrap();
        let TypedTensorRepr::Adjoint(view) = &adjoint.repr else {
            unreachable!()
        };

        let outputs = [
            adjoint.permute(&[0], &[1, 2]).unwrap(),
            adjoint.braid(&[0], &[1, 2], &[0, 1, 2]).unwrap(),
            adjoint.transpose_axes(&[0], &[1, 2]).unwrap(),
            adjoint.repartition(1).unwrap(),
        ];
        for output in &outputs {
            let TypedTensorRepr::Adjoint(output_view) = &output.repr else {
                panic!("identity transform must preserve the lazy representation");
            };
            assert!(Arc::ptr_eq(view, output_view));
        }
        assert!(adjoint.braid(&[0], &[1, 2], &[]).is_err());
        assert!(adjoint.permute(&[0, 0], &[1]).is_err());
        assert!(adjoint.braid(&[0, 0], &[1], &[0, 1, 2]).is_err());
        assert!(adjoint.transpose_axes(&[0, 0], &[1]).is_err());
        assert!(adjoint.repartition(4).is_err());
        assert_eq!(materialized_adjoint_builds(&adjoint), 0);
        assert!(view.materialized.get().is_none());

        let scalar = fixture().trace_pairs(&[(0, 1)]).unwrap();
        let scalar_adjoint = scalar.adjoint().unwrap();
        let TypedTensorRepr::Adjoint(scalar_view) = &scalar_adjoint.repr else {
            unreachable!()
        };
        let scalar_transpose = scalar_adjoint.transpose().unwrap();
        let TypedTensorRepr::Adjoint(transpose_view) = &scalar_transpose.repr else {
            panic!("rank-zero transpose must preserve the lazy representation");
        };
        assert!(Arc::ptr_eq(scalar_view, transpose_view));
        assert!(scalar_view.materialized.get().is_none());
    }

    #[test]
    fn compact_adjoint_never_enters_the_lazy_representation() {
        let runtime = Runtime::builder().dense_threads(1).build().unwrap();
        let provider = Arc::new(U1FusionRule);
        let bond = GradedSpace::try_new(provider, [(U1Irrep::new(0), 2)], false).unwrap();
        let real = TensorMap::diagonal(
            &runtime,
            &bond,
            [SectorSpectrum {
                sector: U1Irrep::new(0),
                values: vec![1.0, 2.0],
            }],
        )
        .unwrap();
        let complex = TensorMap::diagonal(
            &runtime,
            &bond,
            [SectorSpectrum {
                sector: U1Irrep::new(0),
                values: vec![
                    num_complex::Complex64::new(1.0, 2.0),
                    num_complex::Complex64::new(3.0, -4.0),
                ],
            }],
        )
        .unwrap();

        let real_adjoint = real.adjoint().unwrap();
        let complex_adjoint = complex.adjoint().unwrap();
        assert!(matches!(&real_adjoint.repr, TypedTensorRepr::Owned(_)));
        assert!(matches!(&complex_adjoint.repr, TypedTensorRepr::Owned(_)));
        assert!(matches!(
            owned(&real_adjoint).data.as_ref(),
            TypedData::Diagonal(_)
        ));
        assert_eq!(real_adjoint.spectrum().unwrap()[0].values, [1.0, 2.0]);
        assert!(owned(&real_adjoint).dense_cache.get().is_none());
        let spectrum = complex_adjoint.spectrum().unwrap();
        assert_eq!(
            spectrum[0].values,
            [
                num_complex::Complex64::new(1.0, -2.0),
                num_complex::Complex64::new(3.0, 4.0)
            ]
        );
        assert!(owned(&complex_adjoint).dense_cache.get().is_none());

        let complex_restored = complex_adjoint.adjoint().unwrap();
        assert!(matches!(&complex_restored.repr, TypedTensorRepr::Owned(_)));
        assert!(matches!(
            owned(&complex_restored).data.as_ref(),
            TypedData::Diagonal(_)
        ));
        assert_eq!(
            complex_restored.spectrum().unwrap()[0].values,
            [
                num_complex::Complex64::new(1.0, 2.0),
                num_complex::Complex64::new(3.0, -4.0)
            ]
        );
        assert!(owned(&complex_restored).dense_cache.get().is_none());
    }

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
        assert!(Arc::ptr_eq(owned(&tensor), owned(&twin)));
        assert!(Arc::ptr_eq(&owned(&tensor).data, &owned(&twin).data));
        assert_eq!(tensor.data().as_ptr(), twin.data().as_ptr());
        // One payload, however many handles reach it.
        assert_eq!(Arc::strong_count(&owned(&tensor).data), 1);
        assert_eq!(Arc::strong_count(owned(&tensor)), 2);
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
        assert!(!Arc::ptr_eq(owned(&tensor), owned(&inserted)));
        assert!(Arc::ptr_eq(&owned(&tensor).data, &owned(&inserted).data));
        assert!(owned(&inserted).dense_cache.get().is_none());
        let removed = inserted.remove_unit(1).unwrap();
        assert!(Arc::ptr_eq(&owned(&tensor).data, &owned(&removed).data));
        // One payload allocation, three bodies holding it.
        assert_eq!(Arc::strong_count(&owned(&tensor).data), 3);
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
        assert!(!Arc::ptr_eq(&owned(&s).data, &owned(&inserted).data));
        assert!(matches!(&*owned(&inserted).data, TypedData::Dense(_)));
        // Fresh buffer, not the cache the warm-up populated.
        assert_ne!(inserted.data().as_ptr(), warmed);
        let removed = inserted.remove_unit(0).unwrap();
        assert!(Arc::ptr_eq(&owned(&inserted).data, &owned(&removed).data));
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
        assert!(Arc::ptr_eq(owned(&tensor), owned(&twisted)));

        let fermionic = fz2_fixture();
        let untouched = fermionic.twist(&[0]).unwrap();
        assert!(Arc::ptr_eq(owned(&fermionic), owned(&untouched)));
        let touched = fermionic.twist(&[1]).unwrap();
        assert!(!Arc::ptr_eq(owned(&fermionic), owned(&touched)));
    }

    fn transform_seam_calls<T>(operation: impl FnOnce() -> T) -> usize {
        crate::tensor_core::TREE_TRANSFORM_SEAM_CALLS.with(|observation| observation.set(Some(0)));
        let _output = operation();
        crate::tensor_core::TREE_TRANSFORM_SEAM_CALLS
            .with(|observation| observation.replace(None))
            .unwrap()
    }

    fn poison_destination<R, D>(destination: &mut TensorMap<R, D>)
    where
        D: TensorScalar,
    {
        let TypedTensorRepr::Owned(body) = &mut destination.repr else {
            panic!("overwrite destination fixture must be owned")
        };
        let body = Arc::get_mut(body).expect("overwrite destination body must be unique");
        let data = Arc::get_mut(&mut body.data).expect("overwrite payload must be unique");
        let TypedData::Dense(data) = data else {
            panic!("overwrite destination fixture must be dense")
        };
        data.fill(D::from_real(f64::NAN));
    }

    fn assert_overwrite_matches<R, D>(
        source: &TensorMap<R, D>,
        expected: TensorMap<R, D>,
        alpha: D,
        overwrite: impl FnOnce(&mut TensorMap<R, D>) -> Result<(), Error>,
    ) where
        R: MultiplicityFreeRigidSymbols<Scalar = f64> + CheckedFusionAlgebra + SectorCodec,
        D: TensorScalar + core::fmt::Debug,
    {
        let source_before = source.data().to_vec();
        let mut destination = expected.zeros_like();
        poison_destination(&mut destination);
        let provider = Arc::as_ptr(destination.logical_space().provider_arc());
        let body = Arc::as_ptr(owned(&destination));
        let space = destination.logical_space().space() as *const DynamicFusionMapSpace;
        let storage = destination.data().as_ptr();

        overwrite(&mut destination).unwrap();

        let scaled = expected.scale(alpha);
        assert_eq!(destination.data(), scaled.data());
        assert_eq!(source.data(), source_before);
        assert_eq!(
            Arc::as_ptr(destination.logical_space().provider_arc()),
            provider
        );
        assert_eq!(Arc::as_ptr(owned(&destination)), body);
        assert_eq!(
            destination.logical_space().space() as *const DynamicFusionMapSpace,
            space
        );
        assert_eq!(destination.data().as_ptr(), storage);
    }

    #[allow(clippy::too_many_arguments)]
    fn assert_contract_overwrite_matches<R, D>(
        label: &str,
        lhs: &TensorMap<R, D>,
        rhs: &TensorMap<R, D>,
        lhs_axes: &[usize],
        rhs_axes: &[usize],
        output_axes: &[usize],
        alpha: D,
        ordered_alias: bool,
    ) where
        R: TypedSectorAdmission<Error = FusionAlgebraError, Mode = MultiplicityFreeAdmissionMode>
            + MultiplicityFreeRigidSymbols<Scalar = f64>
            + CheckedFusionAlgebra
            + SectorCodec,
        D: TensorScalar + core::fmt::Debug,
    {
        let expected = lhs
            .contract(rhs, lhs_axes, rhs_axes, output_axes)
            .unwrap_or_else(|error| panic!("{label} returning oracle failed: {error:?}"));
        let lhs_before = lhs.data().to_vec();
        let rhs_before = rhs.data().to_vec();
        let mut destination = expected.zeros_like();
        poison_destination(&mut destination);
        let provider = Arc::as_ptr(destination.logical_space().provider_arc());
        let body = Arc::as_ptr(owned(&destination));
        let space = destination.logical_space().space() as *const DynamicFusionMapSpace;
        let storage = destination.data().as_ptr();

        if ordered_alias {
            lhs.contract_ordered_overwrite_into(
                rhs,
                &mut destination,
                lhs_axes,
                rhs_axes,
                output_axes,
                alpha,
            )
            .unwrap_or_else(|error| panic!("{label} ordered overwrite failed: {error:?}"));
        } else {
            lhs.contract_overwrite_into(
                rhs,
                &mut destination,
                lhs_axes,
                rhs_axes,
                output_axes,
                alpha,
            )
            .unwrap_or_else(|error| panic!("{label} overwrite failed: {error:?}"));
        }

        assert_eq!(destination.data(), expected.scale(alpha).data());
        assert_eq!(lhs.data(), lhs_before);
        assert_eq!(rhs.data(), rhs_before);
        assert_eq!(
            Arc::as_ptr(destination.logical_space().provider_arc()),
            provider
        );
        assert_eq!(Arc::as_ptr(owned(&destination)), body);
        assert_eq!(
            destination.logical_space().space() as *const DynamicFusionMapSpace,
            space
        );
        assert_eq!(destination.data().as_ptr(), storage);
    }

    #[test]
    fn typed_tree_overwrite_matches_owned_provider_and_scalar_matrix() {
        let runtime = Runtime::builder().dense_threads(1).build().unwrap();

        let u1_provider = Arc::new(U1FusionRule);
        let u1_leg = GradedSpace::try_new(
            Arc::clone(&u1_provider),
            [
                (U1Irrep::new(-1), 1),
                (U1Irrep::new(0), 2),
                (U1Irrep::new(1), 1),
            ],
            false,
        )
        .unwrap();
        let u1 = TensorMap::from_block_fn(&runtime, [&u1_leg, &u1_leg], [&u1_leg], |_, indices| {
            indices.iter().sum::<usize>() as f64 + 1.0
        })
        .unwrap();
        let u1_permuted = u1.permute(&[1], &[2, 0]).unwrap();
        assert_overwrite_matches(&u1, u1_permuted, -1.5, |destination| {
            u1.permute_overwrite_into(destination, &[1], &[2, 0], -1.5)
        });
        let u1_identity = u1.zeros_like();
        assert_overwrite_matches(&u1, u1_identity, 0.0, |destination| {
            u1.permute_overwrite_into(destination, &[0, 1], &[2], 0.0)
        });

        let independent_leg = GradedSpace::try_new(
            Arc::new(U1FusionRule),
            [
                (U1Irrep::new(-1), 1),
                (U1Irrep::new(0), 2),
                (U1Irrep::new(1), 1),
            ],
            false,
        )
        .unwrap();
        let mut independent_destination = TensorMap::from_block_fn(
            &runtime,
            [&independent_leg, &independent_leg],
            [&independent_leg],
            |_, _| f64::NAN,
        )
        .unwrap();
        assert!(!Arc::ptr_eq(
            u1.logical_space().provider_arc(),
            independent_destination.logical_space().provider_arc()
        ));
        let destination_provider =
            Arc::as_ptr(independent_destination.logical_space().provider_arc());
        u1.permute_overwrite_into(&mut independent_destination, &[0, 1], &[2], 2.0)
            .unwrap();
        assert_eq!(independent_destination.data(), u1.scale(2.0).data());
        assert_eq!(
            Arc::as_ptr(independent_destination.logical_space().provider_arc()),
            destination_provider
        );

        let su2_provider = Arc::new(SU2FusionRule);
        let su2_leg = GradedSpace::try_new(
            su2_provider,
            [
                (SU2Irrep::from_twice_spin(0), 2),
                (SU2Irrep::from_twice_spin(1), 1),
            ],
            false,
        )
        .unwrap();
        let su2 =
            TensorMap::from_block_fn(&runtime, [&su2_leg, &su2_leg], [&su2_leg], |_, indices| {
                indices.iter().sum::<usize>() as f64 + 1.0
            })
            .unwrap()
            .to_c64();
        let alpha = num_complex::Complex64::new(0.75, -0.25);
        let su2_permuted = su2.permute(&[1], &[2, 0]).unwrap();
        assert_overwrite_matches(&su2, su2_permuted, alpha, |destination| {
            su2.permute_overwrite_into(destination, &[1], &[2, 0], alpha)
        });

        let product_provider = Arc::new(U1FusionRule.product(FermionParityFusionRule));
        let product_leg = GradedSpace::try_new(
            product_provider,
            [
                (product_sector(U1Irrep::new(0), Z2Irrep::EVEN), 2),
                (product_sector(U1Irrep::new(1), Z2Irrep::ODD), 1),
            ],
            false,
        )
        .unwrap();
        let product = TensorMap::from_block_fn(
            &runtime,
            [&product_leg, &product_leg],
            [&product_leg],
            |_, indices| indices.iter().sum::<usize>() as f64 + 1.0,
        )
        .unwrap();
        let product_permuted = product.permute(&[1], &[2, 0]).unwrap();
        assert_overwrite_matches(&product, product_permuted, 2.0, |destination| {
            product.permute_overwrite_into(destination, &[1], &[2, 0], 2.0)
        });
    }

    #[test]
    fn typed_planar_overwrite_matches_fermionic_owned_routes() {
        let runtime = Runtime::builder().dense_threads(1).build().unwrap();
        let provider = Arc::new(FermionParityFusionRule);
        let odd = GradedSpace::try_new(provider, [(Z2Irrep::ODD, 2)], false).unwrap();
        let source =
            TensorMap::from_block_fn(&runtime, [&odd, &odd], [&odd, &odd], |_, indices| {
                indices.iter().sum::<usize>() as f64 + 1.0
            })
            .unwrap();
        let alpha = -1.25;

        assert_overwrite_matches(&source, source.transpose().unwrap(), alpha, |destination| {
            source.transpose_overwrite_into(destination, alpha)
        });
        assert_overwrite_matches(
            &source,
            source.transpose_axes(&[1, 3], &[0, 2]).unwrap(),
            alpha,
            |destination| {
                source.transpose_axes_overwrite_into(destination, &[1, 3], &[0, 2], alpha)
            },
        );
        let right = source.repartition(3).unwrap();
        assert_overwrite_matches(&source, right, alpha, |destination| {
            source.repartition_overwrite_into(destination, alpha)
        });
        let left = source.repartition(1).unwrap();
        assert_overwrite_matches(&source, left, alpha, |destination| {
            source.repartition_overwrite_into(destination, alpha)
        });
    }

    fn f64_bits<R>(tensor: &TensorMap<R, f64>) -> Vec<u64> {
        tensor.data().iter().map(|value| value.to_bits()).collect()
    }

    fn f64_destination_state<R>(tensor: &TensorMap<R, f64>) -> (Vec<u64>, [usize; 4]) {
        (
            f64_bits(tensor),
            [
                Arc::as_ptr(tensor.logical_space().provider_arc()) as usize,
                Arc::as_ptr(owned(tensor)) as usize,
                tensor.logical_space().space() as *const DynamicFusionMapSpace as usize,
                tensor.data().as_ptr() as usize,
            ],
        )
    }

    fn pop_dense_element<R, D>(tensor: &mut TensorMap<R, D>) {
        let TypedTensorRepr::Owned(body) = &mut tensor.repr else {
            panic!("malformed-storage fixture must be owned")
        };
        let body = Arc::get_mut(body).expect("malformed-storage body must be unique");
        let data = Arc::get_mut(&mut body.data).expect("malformed-storage payload must be unique");
        let TypedData::Dense(data) = data else {
            panic!("malformed-storage fixture must be dense")
        };
        data.pop();
    }

    #[test]
    fn typed_tree_overwrite_rejections_leave_destination_unchanged() {
        let runtime = Runtime::builder().dense_threads(1).build().unwrap();
        let provider = Arc::new(U1FusionRule);
        let leg = GradedSpace::try_new(
            provider,
            [(U1Irrep::new(0), 2), (U1Irrep::new(1), 1)],
            false,
        )
        .unwrap();
        let source = TensorMap::from_block_fn(&runtime, [&leg, &leg], [&leg], |_, indices| {
            indices.iter().sum::<usize>() as f64 + 1.0
        })
        .unwrap();
        let expected = source.permute(&[1], &[2, 0]).unwrap();

        let assert_unchanged = |destination: &TensorMap<U1FusionRule, f64>, before: &[u64]| {
            assert_eq!(f64_bits(destination), before);
        };

        let mut wrong_layout = source.zeros_like();
        poison_destination(&mut wrong_layout);
        let before = f64_bits(&wrong_layout);
        assert!(source
            .permute_overwrite_into(&mut wrong_layout, &[1], &[2, 0], 1.0)
            .is_err());
        assert_unchanged(&wrong_layout, &before);

        for (codomain_axes, domain_axes) in [(&[1, 1][..], &[2][..]), (&[1][..], &[2, 3][..])] {
            let mut destination = expected.zeros_like();
            poison_destination(&mut destination);
            let before = f64_bits(&destination);
            assert!(source
                .permute_overwrite_into(&mut destination, codomain_axes, domain_axes, 1.0,)
                .is_err());
            assert_unchanged(&destination, &before);
        }

        let mut nonplanar = source.transpose().unwrap().zeros_like();
        poison_destination(&mut nonplanar);
        let before = f64_bits(&nonplanar);
        assert!(source
            .transpose_axes_overwrite_into(&mut nonplanar, &[0, 2], &[1], 1.0)
            .is_err());
        assert_unchanged(&nonplanar, &before);

        let other_runtime = Runtime::builder().dense_threads(1).build().unwrap();
        let mut foreign = expected.zeros_like();
        foreign.runtime = other_runtime;
        poison_destination(&mut foreign);
        let before = f64_bits(&foreign);
        assert_eq!(
            source
                .permute_overwrite_into(&mut foreign, &[1], &[2, 0], 1.0)
                .unwrap_err(),
            Error::RuntimeMismatch
        );
        assert_unchanged(&foreign, &before);

        let mut shared_body = expected.zeros_like();
        poison_destination(&mut shared_body);
        let before = f64_bits(&shared_body);
        let shared_body_handle = shared_body.clone();
        assert!(source
            .permute_overwrite_into(&mut shared_body, &[1], &[2, 0], 1.0)
            .is_err());
        assert_unchanged(&shared_body, &before);
        drop(shared_body_handle);

        let mut shared_payload = expected.zeros_like();
        poison_destination(&mut shared_payload);
        let before = f64_bits(&shared_payload);
        let payload_handle = shared_payload.insert_left_unit(0, false).unwrap();
        assert!(source
            .permute_overwrite_into(&mut shared_payload, &[1], &[2, 0], 1.0)
            .is_err());
        assert_unchanged(&shared_payload, &before);
        drop(payload_handle);

        let mut alias = source
            .insert_left_unit(0, false)
            .unwrap()
            .remove_unit(0)
            .unwrap();
        let before = f64_bits(&alias);
        assert!(source
            .permute_overwrite_into(&mut alias, &[0, 1], &[2], 1.0)
            .is_err());
        assert_unchanged(&alias, &before);

        let mut bad_len = expected.zeros_like();
        poison_destination(&mut bad_len);
        let before = {
            let TypedTensorRepr::Owned(body) = &mut bad_len.repr else {
                unreachable!()
            };
            let body = Arc::get_mut(body).unwrap();
            let data = Arc::get_mut(&mut body.data).unwrap();
            let TypedData::Dense(data) = data else {
                unreachable!()
            };
            data.pop();
            f64_bits(&bad_len)
        };
        assert!(source
            .permute_overwrite_into(&mut bad_len, &[1], &[2, 0], 1.0)
            .is_err());
        assert_unchanged(&bad_len, &before);

        let mut lazy_destination = expected.adjoint().unwrap();
        let before = f64_bits(&lazy_destination);
        assert!(source
            .permute_overwrite_into(&mut lazy_destination, &[1], &[2, 0], 1.0)
            .is_err());
        assert_eq!(f64_bits(&lazy_destination), before);

        let lazy_source = source.adjoint().unwrap();
        let mut destination = expected.zeros_like();
        poison_destination(&mut destination);
        let before = f64_bits(&destination);
        assert!(lazy_source
            .permute_overwrite_into(&mut destination, &[1], &[2, 0], 1.0)
            .is_err());
        assert_unchanged(&destination, &before);

        let square = TensorMap::from_block_fn(&runtime, [&leg], [&leg], |_, indices| {
            indices.iter().sum::<usize>() as f64 + 1.0
        })
        .unwrap();
        let compact = square.svd_compact().unwrap().1;
        let mut compact_destination = compact.zeros_like();
        let before = compact_destination.data().to_vec();
        assert!(square
            .permute_overwrite_into(&mut compact_destination, &[0], &[1], 1.0)
            .is_err());
        assert_eq!(compact_destination.data(), before);

        let z2 = Arc::new(ZNFusionRule::new(2).unwrap());
        let z3 = Arc::new(ZNFusionRule::new(3).unwrap());
        let z2_leg = GradedSpace::try_new(Arc::clone(&z2), [(z2.irrep(0), 1)], false).unwrap();
        let z3_leg = GradedSpace::try_new(Arc::clone(&z3), [(z3.irrep(0), 1)], false).unwrap();
        let z2_source =
            TensorMap::from_block_fn(&runtime, [&z2_leg], [&z2_leg], |_, _| 2.0).unwrap();
        let mut z3_destination =
            TensorMap::from_block_fn(&runtime, [&z3_leg], [&z3_leg], |_, _| f64::NAN).unwrap();
        let before = f64_bits(&z3_destination);
        assert_eq!(
            z2_source
                .permute_overwrite_into(&mut z3_destination, &[0], &[1], 1.0)
                .unwrap_err(),
            Error::RuleMismatch
        );
        assert_eq!(f64_bits(&z3_destination), before);
    }

    #[test]
    fn typed_tree_overwrite_covers_boundary_ranks_and_runtime_cache_reuse() {
        let runtime = Runtime::builder().dense_threads(1).build().unwrap();
        let provider = Arc::new(U1FusionRule);
        let leg = GradedSpace::try_new(provider, [(U1Irrep::new(0), 1)], false).unwrap();

        let one_sided = TensorMap::from_block_fn(
            &runtime,
            [&leg],
            std::iter::empty::<&GradedSpace<U1FusionRule>>(),
            |_, _| 3.0,
        )
        .unwrap();
        let moved = one_sided.repartition(0).unwrap();
        assert_overwrite_matches(&one_sided, moved, 2.0, |destination| {
            one_sided.repartition_overwrite_into(destination, 2.0)
        });

        let square = TensorMap::from_block_fn(&runtime, [&leg], [&leg], |_, _| 4.0).unwrap();
        let scalar = square.trace_pairs(&[(0, 1)]).unwrap();
        let scalar_destination = scalar.transpose().unwrap();
        assert_overwrite_matches(&scalar, scalar_destination, -0.5, |destination| {
            scalar.transpose_overwrite_into(destination, -0.5)
        });

        let high_rank = TensorMap::from_block_fn(
            &runtime,
            (0..9).map(|_| &leg),
            (0..8).map(|_| &leg),
            |_, _| 1.0,
        )
        .unwrap();
        let mut high_rank_destination = high_rank.zeros_like();
        poison_destination(&mut high_rank_destination);
        let before = f64_bits(&high_rank_destination);
        assert!(high_rank
            .permute_overwrite_into(
                &mut high_rank_destination,
                &[0, 1, 2, 3, 4, 5, 6, 7, 17],
                &[8, 9, 10, 11, 12, 13, 14, 15],
                1.0,
            )
            .is_err());
        assert_eq!(f64_bits(&high_rank_destination), before);

        runtime.clear_tree_transform_cache();
        let source = TensorMap::from_block_fn(&runtime, [&leg, &leg], [&leg], |_, indices| {
            indices.iter().sum::<usize>() as f64 + 1.0
        })
        .unwrap();
        let expected = source.permute(&[1], &[2, 0]).unwrap();
        runtime.clear_tree_transform_cache();
        let mut first = expected.zeros_like();
        source
            .permute_overwrite_into(&mut first, &[1], &[2, 0], 1.0)
            .unwrap();
        let cold = runtime.tree_transform_cache_info();
        let mut second = expected.zeros_like();
        source
            .permute_overwrite_into(&mut second, &[1], &[2, 0], 1.0)
            .unwrap();
        let warm = runtime.tree_transform_cache_info();
        assert_eq!(warm.entries(), cold.entries());
        assert!(warm.hits() > cold.hits());
        assert_eq!(first.data(), second.data());
    }

    #[test]
    fn typed_tree_overwrite_shared_runtime_is_concurrent_and_deterministic() {
        // What: exact-layout admission and completed replay share one Runtime
        // without serializing execution or changing results across callers.
        let runtime = Runtime::builder().dense_threads(1).build().unwrap();
        let leg = GradedSpace::try_new(
            Arc::new(U1FusionRule),
            [
                (U1Irrep::new(-1), 2),
                (U1Irrep::new(0), 3),
                (U1Irrep::new(1), 2),
            ],
            false,
        )
        .unwrap();
        let source = TensorMap::from_block_fn(&runtime, [&leg, &leg], [&leg], |_, indices| {
            indices.iter().sum::<usize>() as f64 + 1.0
        })
        .unwrap();
        let expected = source.permute(&[1], &[2, 0]).unwrap();
        let mut warm = expected.zeros_like();
        source
            .permute_overwrite_into(&mut warm, &[1], &[2, 0], 1.0)
            .unwrap();

        std::thread::scope(|scope| {
            let handles: Vec<_> = (0..4)
                .map(|_| {
                    scope.spawn(|| {
                        let mut destination = expected.zeros_like();
                        source
                            .permute_overwrite_into(&mut destination, &[1], &[2, 0], 1.0)
                            .unwrap();
                        destination
                    })
                })
                .collect();
            for handle in handles {
                assert_eq!(handle.join().unwrap().data(), expected.data());
            }
        });
    }

    #[test]
    fn typed_contract_overwrite_matches_provider_scalar_and_order_matrix() {
        let runtime = Runtime::builder().dense_threads(1).build().unwrap();

        let u1_leg = GradedSpace::try_new(
            Arc::new(U1FusionRule),
            [
                (U1Irrep::new(-1), 1),
                (U1Irrep::new(0), 2),
                (U1Irrep::new(1), 1),
            ],
            false,
        )
        .unwrap();
        let u1 = TensorMap::from_block_fn(&runtime, [&u1_leg], [&u1_leg], |_, indices| {
            indices.iter().sum::<usize>() as f64 + 1.0
        })
        .unwrap();
        assert_contract_overwrite_matches("u1", &u1, &u1, &[1], &[0], &[0, 1], -1.5, false);

        let su2_leg = GradedSpace::try_new(
            Arc::new(SU2FusionRule),
            [
                (SU2Irrep::from_twice_spin(0), 2),
                (SU2Irrep::from_twice_spin(1), 1),
            ],
            false,
        )
        .unwrap();
        let su2 = TensorMap::from_block_fn(&runtime, [&su2_leg], [&su2_leg], |_, indices| {
            indices.iter().sum::<usize>() as f64 + 1.0
        })
        .unwrap()
        .to_c64();
        assert_contract_overwrite_matches(
            "su2",
            &su2,
            &su2,
            &[1],
            &[0],
            &[1, 0],
            num_complex::Complex64::new(0.75, -0.25),
            true,
        );

        let odd = GradedSpace::try_new(
            Arc::new(FermionParityFusionRule),
            [(Z2Irrep::ODD, 2)],
            false,
        )
        .unwrap();
        let fermionic = TensorMap::from_block_fn(&runtime, [&odd], [&odd], |_, indices| {
            indices.iter().sum::<usize>() as f64 + 1.0
        })
        .unwrap();
        assert_contract_overwrite_matches(
            "fz2",
            &fermionic,
            &fermionic,
            &[0, 1],
            &[1, 0],
            &[],
            0.0,
            false,
        );

        let product_leg = GradedSpace::try_new(
            Arc::new(U1FusionRule.product(FermionParityFusionRule)),
            [
                (product_sector(U1Irrep::new(0), Z2Irrep::EVEN), 2),
                (product_sector(U1Irrep::new(1), Z2Irrep::ODD), 1),
            ],
            false,
        )
        .unwrap();
        let product =
            TensorMap::from_block_fn(&runtime, [&product_leg], [&product_leg], |_, indices| {
                indices.iter().sum::<usize>() as f64 + 1.0
            })
            .unwrap();
        assert_contract_overwrite_matches(
            "product",
            &product,
            &product,
            &[1],
            &[0],
            &[0, 1],
            2.0,
            false,
        );

        let q = GradedSpace::try_new(
            Arc::new(CU1FusionRule),
            [(CU1Irrep::from_twice_charge(1), 1)],
            false,
        )
        .unwrap();
        let cu1 = TensorMap::from_block_fn(&runtime, [&q, &q, &q], [&q], |_, _| 1.0).unwrap();
        let cu1_expected = cu1.contract(&cu1, &[3], &[0], &[5, 1, 3, 0, 4, 2]).unwrap();
        assert!(cu1_expected.data().contains(&0.0));
        assert_contract_overwrite_matches(
            "cu1",
            &cu1,
            &cu1,
            &[3],
            &[0],
            &[5, 1, 3, 0, 4, 2],
            1.0,
            false,
        );
    }

    #[test]
    fn typed_contract_overwrite_keeps_distinct_destination_provider_authority() {
        let runtime = Runtime::builder().dense_threads(1).build().unwrap();
        let build = |provider: Arc<U1FusionRule>, offset: f64| {
            let leg = GradedSpace::try_new(
                provider,
                [
                    (U1Irrep::new(-1), 1),
                    (U1Irrep::new(0), 2),
                    (U1Irrep::new(1), 1),
                ],
                false,
            )
            .unwrap();
            TensorMap::from_block_fn(&runtime, [&leg], [&leg], |_, indices| {
                offset + indices.iter().sum::<usize>() as f64
            })
            .unwrap()
        };

        let lhs = build(Arc::new(U1FusionRule), 1.0);
        let rhs = build(Arc::new(U1FusionRule), 10.0);
        let destination_provider = Arc::new(U1FusionRule);
        let destination_lhs = build(Arc::clone(&destination_provider), 0.0);
        let destination_rhs = build(destination_provider, 0.0);
        let expected = lhs.contract(&rhs, &[1], &[0], &[0, 1]).unwrap();
        let mut destination = destination_lhs
            .contract(&destination_rhs, &[1], &[0], &[0, 1])
            .unwrap()
            .zeros_like();
        poison_destination(&mut destination);

        assert!(!Arc::ptr_eq(
            lhs.logical_space().provider_arc(),
            rhs.logical_space().provider_arc()
        ));
        assert!(!Arc::ptr_eq(
            lhs.logical_space().provider_arc(),
            destination.logical_space().provider_arc()
        ));
        let provider = Arc::as_ptr(destination.logical_space().provider_arc());
        let body = Arc::as_ptr(owned(&destination));
        let space = destination.logical_space().space() as *const DynamicFusionMapSpace;
        let storage = destination.data().as_ptr();

        lhs.contract_overwrite_into(&rhs, &mut destination, &[1], &[0], &[0, 1], 1.0)
            .unwrap();

        assert_eq!(destination.data(), expected.data());
        assert_eq!(
            Arc::as_ptr(destination.logical_space().provider_arc()),
            provider
        );
        assert_eq!(Arc::as_ptr(owned(&destination)), body);
        assert_eq!(
            destination.logical_space().space() as *const DynamicFusionMapSpace,
            space
        );
        assert_eq!(destination.data().as_ptr(), storage);
    }

    #[test]
    fn typed_contract_overwrite_accepts_lazy_and_compact_inputs_without_warming_adjoint() {
        let runtime = Runtime::builder().dense_threads(1).build().unwrap();
        let leg = GradedSpace::try_new(
            Arc::new(U1FusionRule),
            [(U1Irrep::new(0), 3), (U1Irrep::new(1), 1)],
            false,
        )
        .unwrap();
        let lhs = TensorMap::from_block_fn(&runtime, [&leg], [&leg], |_, indices| {
            (indices[0] + 2 * indices[1] + 1) as f64
        })
        .unwrap();
        let rhs = TensorMap::from_block_fn(&runtime, [&leg], [&leg], |_, indices| {
            (2 * indices[0] + indices[1] + 1) as f64
        })
        .unwrap();
        let lazy_lhs = lhs.adjoint().unwrap();
        let lazy_rhs = rhs.adjoint().unwrap();
        let expected = lazy_lhs.contract(&lazy_rhs, &[1], &[0], &[0, 1]).unwrap();
        let mut destination = expected.zeros_like();
        poison_destination(&mut destination);
        lazy_lhs
            .contract_overwrite_into(&lazy_rhs, &mut destination, &[1], &[0], &[0, 1], 1.0)
            .unwrap();
        assert_eq!(destination.data(), expected.data());
        assert_eq!(materialized_adjoint_builds(&lazy_lhs), 0);
        assert_eq!(materialized_adjoint_builds(&lazy_rhs), 0);
        for lazy in [&lazy_lhs, &lazy_rhs] {
            let TypedTensorRepr::Adjoint(view) = &lazy.repr else {
                unreachable!()
            };
            assert!(view.materialized.get().is_none());
        }

        let (u, s, _) = lhs.svd_compact().unwrap();
        assert!(owned(&s).dense_cache.get().is_none());
        let expected = u.contract(&s, &[1], &[0], &[0, 1]).unwrap();
        let mut destination = expected.zeros_like();
        poison_destination(&mut destination);
        u.contract_overwrite_into(&s, &mut destination, &[1], &[0], &[0, 1], 1.0)
            .unwrap();
        assert_eq!(destination.data(), expected.data());
        assert!(owned(&s).dense_cache.get().is_some());
    }

    #[test]
    fn typed_contract_overwrite_rejections_are_preclear_and_atomic() {
        let runtime = Runtime::builder().dense_threads(1).build().unwrap();
        let provider = Arc::new(U1FusionRule);
        let leg = GradedSpace::try_new(
            provider,
            [(U1Irrep::new(0), 2), (U1Irrep::new(1), 1)],
            false,
        )
        .unwrap();
        let lhs = TensorMap::from_block_fn(&runtime, [&leg], [&leg], |_, indices| {
            (indices[0] + 2 * indices[1] + 1) as f64
        })
        .unwrap();
        let rhs = TensorMap::from_block_fn(&runtime, [&leg], [&leg], |_, indices| {
            (2 * indices[0] + indices[1] + 1) as f64
        })
        .unwrap();
        let expected = lhs.contract(&rhs, &[1], &[0], &[0, 1]).unwrap();
        let destination = || {
            let mut destination = expected.zeros_like();
            poison_destination(&mut destination);
            destination
        };

        let mut foreign = destination();
        foreign.runtime = Runtime::builder().dense_threads(1).build().unwrap();
        let before = f64_destination_state(&foreign);
        assert_eq!(
            lhs.contract_overwrite_into(&rhs, &mut foreign, &[9], &[0], &[0, 1], 1.0,)
                .unwrap_err(),
            Error::RuntimeMismatch
        );
        assert_eq!(f64_destination_state(&foreign), before);

        let other_runtime = Runtime::builder().dense_threads(1).build().unwrap();
        let foreign_rhs = TensorMap::from_block_fn(&other_runtime, [&leg], [&leg], |_, indices| {
            (indices[0] + indices[1] + 1) as f64
        })
        .unwrap();
        let mut rejected = destination();
        let before = f64_destination_state(&rejected);
        assert_eq!(
            lhs.contract_overwrite_into(&foreign_rhs, &mut rejected, &[9], &[0], &[0, 1], 1.0,)
                .unwrap_err(),
            Error::RuntimeMismatch
        );
        assert_eq!(f64_destination_state(&rejected), before);

        for (lhs_axes, rhs_axes, output_axes) in [
            (&[1, 1][..], &[0, 0][..], &[][..]),
            (&[2][..], &[0][..], &[0, 1][..]),
            (&[1][..], &[0][..], &[0, 0][..]),
        ] {
            let mut rejected = destination();
            let before = f64_destination_state(&rejected);
            assert!(lhs
                .contract_overwrite_into(&rhs, &mut rejected, lhs_axes, rhs_axes, output_axes, 1.0,)
                .is_err());
            assert_eq!(f64_destination_state(&rejected), before);
        }

        let bad_leg = GradedSpace::try_new(
            Arc::clone(lhs.logical_space().provider_arc()),
            [(U1Irrep::new(7), 1)],
            false,
        )
        .unwrap();
        let bad_rhs =
            TensorMap::from_block_fn(&runtime, [&bad_leg], [&bad_leg], |_, _| 1.0).unwrap();
        let mut rejected = destination();
        let before = f64_destination_state(&rejected);
        assert!(lhs
            .contract_overwrite_into(&bad_rhs, &mut rejected, &[1], &[0], &[0, 1], 1.0,)
            .is_err());
        assert_eq!(f64_destination_state(&rejected), before);

        let mut wrong_layout = lhs.insert_left_unit(0, false).unwrap().zeros_like();
        poison_destination(&mut wrong_layout);
        let before = f64_destination_state(&wrong_layout);
        assert!(lhs
            .contract_overwrite_into(&rhs, &mut wrong_layout, &[1], &[0], &[0, 1], 1.0,)
            .is_err());
        assert_eq!(f64_destination_state(&wrong_layout), before);

        for malformed in ["lhs", "rhs", "destination"] {
            let mut bad_lhs = lhs.scale(1.0);
            let mut bad_rhs = rhs.scale(1.0);
            let mut rejected = destination();
            match malformed {
                "lhs" => pop_dense_element(&mut bad_lhs),
                "rhs" => pop_dense_element(&mut bad_rhs),
                "destination" => pop_dense_element(&mut rejected),
                _ => unreachable!(),
            }
            let before = f64_destination_state(&rejected);
            assert!(bad_lhs
                .contract_overwrite_into(&bad_rhs, &mut rejected, &[1], &[0], &[0, 1], 1.0,)
                .is_err());
            assert_eq!(f64_destination_state(&rejected), before);
        }

        let mut lhs_alias = lhs
            .insert_left_unit(0, false)
            .unwrap()
            .remove_unit(0)
            .unwrap();
        let before = f64_destination_state(&lhs_alias);
        assert!(lhs
            .contract_overwrite_into(&rhs, &mut lhs_alias, &[1], &[0], &[0, 1], 1.0,)
            .is_err());
        assert_eq!(f64_destination_state(&lhs_alias), before);

        let mut rhs_alias = rhs
            .insert_left_unit(0, false)
            .unwrap()
            .remove_unit(0)
            .unwrap();
        let before = f64_destination_state(&rhs_alias);
        assert!(lhs
            .contract_overwrite_into(&rhs, &mut rhs_alias, &[1], &[0], &[0, 1], 1.0,)
            .is_err());
        assert_eq!(f64_destination_state(&rhs_alias), before);

        let mut shared_body = destination();
        let shared_body_handle = shared_body.clone();
        let before = f64_destination_state(&shared_body);
        assert!(lhs
            .contract_overwrite_into(&rhs, &mut shared_body, &[1], &[0], &[0, 1], 1.0,)
            .is_err());
        assert_eq!(f64_destination_state(&shared_body), before);
        drop(shared_body_handle);

        let mut shared_payload = destination();
        let shared_payload_handle = shared_payload.insert_left_unit(0, false).unwrap();
        let before = f64_destination_state(&shared_payload);
        assert!(lhs
            .contract_overwrite_into(&rhs, &mut shared_payload, &[1], &[0], &[0, 1], 1.0,)
            .is_err());
        assert_eq!(f64_destination_state(&shared_payload), before);
        drop(shared_payload_handle);

        let mut lazy_destination = expected.adjoint().unwrap();
        let TypedTensorRepr::Adjoint(view) = &lazy_destination.repr else {
            unreachable!()
        };
        let view = Arc::as_ptr(view);
        assert!(lhs
            .contract_overwrite_into(&rhs, &mut lazy_destination, &[1], &[0], &[0, 1], 1.0,)
            .is_err());
        let TypedTensorRepr::Adjoint(after) = &lazy_destination.repr else {
            unreachable!()
        };
        assert_eq!(Arc::as_ptr(after), view);
        assert!(after.materialized.get().is_none());

        let mut compact_destination = lhs.svd_compact().unwrap().1;
        let payload = Arc::clone(&owned(&compact_destination).data);
        assert!(owned(&compact_destination).dense_cache.get().is_none());
        assert!(lhs
            .contract_overwrite_into(&rhs, &mut compact_destination, &[1], &[0], &[0, 1], 1.0,)
            .is_err());
        assert!(Arc::ptr_eq(&owned(&compact_destination).data, &payload));
        assert!(owned(&compact_destination).dense_cache.get().is_none());

        let z2 = Arc::new(ZNFusionRule::new(2).unwrap());
        let z3 = Arc::new(ZNFusionRule::new(3).unwrap());
        let z2_leg = GradedSpace::try_new(Arc::clone(&z2), [(z2.irrep(0), 1)], false).unwrap();
        let z3_leg = GradedSpace::try_new(Arc::clone(&z3), [(z3.irrep(0), 1)], false).unwrap();
        let z2_lhs = TensorMap::from_block_fn(&runtime, [&z2_leg], [&z2_leg], |_, _| 1.0).unwrap();
        let z2_rhs = TensorMap::from_block_fn(&runtime, [&z2_leg], [&z2_leg], |_, _| 2.0).unwrap();
        let z3_rhs = TensorMap::from_block_fn(&runtime, [&z3_leg], [&z3_leg], |_, _| 2.0).unwrap();
        let mut rejected = z2_lhs
            .contract(&z2_rhs, &[1], &[0], &[0, 1])
            .unwrap()
            .zeros_like();
        poison_destination(&mut rejected);
        let before = f64_destination_state(&rejected);
        assert_eq!(
            z2_lhs
                .contract_overwrite_into(&z3_rhs, &mut rejected, &[1], &[0], &[0, 1], 1.0,)
                .unwrap_err(),
            Error::RuleMismatch
        );
        assert_eq!(f64_destination_state(&rejected), before);

        let z3_lhs = TensorMap::from_block_fn(&runtime, [&z3_leg], [&z3_leg], |_, _| 1.0).unwrap();
        let mut z3_destination = z3_lhs
            .contract(&z3_rhs, &[1], &[0], &[0, 1])
            .unwrap()
            .zeros_like();
        poison_destination(&mut z3_destination);
        let before = f64_destination_state(&z3_destination);
        assert_eq!(
            z2_lhs
                .contract_overwrite_into(&z2_rhs, &mut z3_destination, &[1], &[0], &[0, 1], 1.0,)
                .unwrap_err(),
            Error::RuleMismatch
        );
        assert_eq!(f64_destination_state(&z3_destination), before);
    }

    #[test]
    fn typed_contract_overwrite_handles_unmatched_sectors_and_reuses_runtime_cache() {
        let runtime = Runtime::builder().dense_threads(1).build().unwrap();
        let provider = Arc::new(U1FusionRule);
        let bond =
            GradedSpace::try_new(Arc::clone(&provider), [(U1Irrep::new(0), 2)], false).unwrap();
        let left = GradedSpace::try_new(
            Arc::clone(&provider),
            [(U1Irrep::new(0), 1), (U1Irrep::new(1), 1)],
            false,
        )
        .unwrap();
        let partly_disjoint = GradedSpace::try_new(
            Arc::clone(&provider),
            [(U1Irrep::new(0), 1), (U1Irrep::new(2), 1)],
            false,
        )
        .unwrap();
        let disjoint = GradedSpace::try_new(provider, [(U1Irrep::new(3), 1)], false).unwrap();
        let lhs = TensorMap::from_block_fn(&runtime, [&left], [&bond], |_, indices| {
            indices.iter().sum::<usize>() as f64 + 1.0
        })
        .unwrap();
        for open in [&partly_disjoint, &disjoint] {
            let rhs = TensorMap::from_block_fn(&runtime, [&bond], [open], |_, indices| {
                indices.iter().sum::<usize>() as f64 + 2.0
            })
            .unwrap();
            assert_contract_overwrite_matches(
                "unmatched sectors",
                &lhs,
                &rhs,
                &[1],
                &[0],
                &[0, 1],
                1.0,
                false,
            );
        }

        let provider = Arc::new(CU1FusionRule);
        let q =
            GradedSpace::try_new(provider, [(CU1Irrep::from_twice_charge(1), 1)], false).unwrap();
        let source = TensorMap::from_block_fn(&runtime, [&q, &q, &q], [&q], |_, _| 1.0).unwrap();
        let axes = [5, 1, 3, 0, 4, 2];
        let expected = source.contract(&source, &[3], &[0], &axes).unwrap();
        runtime.clear_tree_transform_cache();
        let mut first = expected.zeros_like();
        poison_destination(&mut first);
        source
            .contract_overwrite_into(&source, &mut first, &[3], &[0], &axes, 1.0)
            .unwrap();
        let cold = runtime.tree_transform_cache_info();
        let mut second = expected.zeros_like();
        poison_destination(&mut second);
        source
            .contract_overwrite_into(&source, &mut second, &[3], &[0], &axes, 1.0)
            .unwrap();
        let warm = runtime.tree_transform_cache_info();
        assert_eq!(first.data(), second.data());
        assert_eq!(warm.entries(), cold.entries());
        assert!(warm.hits() > cold.hits());
    }

    #[test]
    fn exact_identity_transforms_share_unique_and_simple_bodies() {
        // What (#689 PR A): exact identity permute/braid/transpose/repartition
        // never reach the transform seam and return the same body allocation.
        // Body identity also pins zero payload copies more directly than an
        // allocator byte count can.
        let runtime = Runtime::builder().dense_threads(1).build().unwrap();

        let u1_provider = Arc::new(U1FusionRule);
        let u1_leg = GradedSpace::try_new(
            Arc::clone(&u1_provider),
            [
                (U1Irrep::new(-1), 1),
                (U1Irrep::new(0), 2),
                (U1Irrep::new(1), 1),
            ],
            false,
        )
        .unwrap();
        let u1_f64: TensorMap<U1FusionRule, f64> =
            TensorMap::from_block_fn(&runtime, [&u1_leg, &u1_leg], [&u1_leg], |_, indices| {
                indices.iter().sum::<usize>() as f64 + 1.0
            })
            .unwrap();
        let u1_c64 = u1_f64.to_c64();

        let su2_provider = Arc::new(SU2FusionRule);
        let su2_leg = GradedSpace::try_new(
            Arc::clone(&su2_provider),
            [
                (SU2Irrep::from_twice_spin(0), 2),
                (SU2Irrep::from_twice_spin(1), 1),
            ],
            false,
        )
        .unwrap();
        let su2_f64: TensorMap<SU2FusionRule, f64> =
            TensorMap::from_block_fn(&runtime, [&su2_leg, &su2_leg], [&su2_leg], |_, indices| {
                indices.iter().sum::<usize>() as f64 + 1.0
            })
            .unwrap();
        let su2_c64 = su2_f64.to_c64();

        macro_rules! assert_identity_ops {
            ($tensor:expr) => {{
                let tensor = $tensor;
                let calls = transform_seam_calls(|| {
                    for output in [
                        tensor.permute(&[0, 1], &[2]).unwrap(),
                        tensor.braid(&[0, 1], &[2], &[5, 3, 1]).unwrap(),
                        tensor.transpose_axes(&[0, 1], &[2]).unwrap(),
                        tensor.repartition(2).unwrap(),
                    ] {
                        assert!(Arc::ptr_eq(owned(tensor), owned(&output)));
                        assert!(Arc::ptr_eq(&owned(tensor).data, &owned(&output).data));
                        assert_eq!(tensor.data().as_ptr(), output.data().as_ptr());
                    }
                });
                assert_eq!(calls, 0);
            }};
        }

        assert_identity_ops!(&u1_f64);
        assert_identity_ops!(&u1_c64);
        assert_identity_ops!(&su2_f64);
        assert_identity_ops!(&su2_c64);

        // Validation still precedes the braid shortcut.
        let calls = transform_seam_calls(|| {
            assert!(u1_f64.braid(&[0, 1], &[2], &[0, 0]).is_err());
        });
        assert_eq!(calls, 0);
        assert!(u1_f64.permute(&[0, 1], &[2, 3]).is_err());
        assert!(u1_f64.transpose_axes(&[0, 1], &[2, 3]).is_err());
        assert!(u1_f64.repartition(4).is_err());

        // Negative control: the counter observes a real transform.
        let calls = transform_seam_calls(|| {
            let moved = u1_f64.permute(&[1, 0], &[2]).unwrap();
            assert!(!Arc::ptr_eq(owned(&u1_f64), owned(&moved)));
        });
        assert_eq!(calls, 1);
    }

    #[test]
    fn high_rank_identity_has_no_inline_capacity_boundary() {
        let runtime = Runtime::builder().dense_threads(1).build().unwrap();
        let provider = Arc::new(U1FusionRule);
        let leg = GradedSpace::try_new(provider, [(U1Irrep::new(0), 1)], false).unwrap();
        let tensor: TensorMap<U1FusionRule, f64> =
            TensorMap::from_block_fn(&runtime, [&leg; 10], [&leg; 9], |_, _| 1.0).unwrap();
        let codomain_axes: Vec<_> = (0..10).collect();
        let domain_axes: Vec<_> = (10..19).collect();
        let levels = vec![0; 19];

        let calls = transform_seam_calls(|| {
            for output in [
                tensor.permute(&codomain_axes, &domain_axes).unwrap(),
                tensor.braid(&codomain_axes, &domain_axes, &levels).unwrap(),
                tensor.transpose_axes(&codomain_axes, &domain_axes).unwrap(),
            ] {
                assert!(Arc::ptr_eq(owned(&tensor), owned(&output)));
            }
        });
        assert_eq!(calls, 0);
    }

    #[test]
    fn compact_identity_transforms_do_not_materialize() {
        let factor = fixture().svd_compact().unwrap().1;
        assert!(matches!(&*owned(&factor).data, TypedData::Diagonal(_)));
        assert!(owned(&factor).dense_cache.get().is_none());

        let calls = transform_seam_calls(|| {
            for output in [
                factor.permute(&[0], &[1]).unwrap(),
                factor.braid(&[0], &[1], &[2, 1]).unwrap(),
                factor.transpose_axes(&[0], &[1]).unwrap(),
                factor.repartition(1).unwrap(),
            ] {
                assert!(Arc::ptr_eq(owned(&factor), owned(&output)));
                assert!(matches!(&*owned(&output).data, TypedData::Diagonal(_)));
                assert!(owned(&output).dense_cache.get().is_none());
            }
        });
        assert_eq!(calls, 0);
        assert!(owned(&factor).dense_cache.get().is_none());
    }

    #[test]
    fn scalar_transpose_shares_the_body() {
        let scalar = fixture().trace_pairs(&[(0, 1)]).unwrap();
        assert_eq!(scalar.rank(), 0);
        let calls = transform_seam_calls(|| {
            let transposed = scalar.transpose().unwrap();
            assert!(Arc::ptr_eq(owned(&scalar), owned(&transposed)));
        });
        assert_eq!(calls, 0);
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
        assert!(matches!(&*owned(&twisted).data, TypedData::Diagonal(_)));
        assert!(!Arc::ptr_eq(&owned(&s).data, &owned(&twisted).data));
        let inverse = s.twist_inverse(&[0]).unwrap();
        assert!(matches!(&*owned(&inverse).data, TypedData::Diagonal(_)));
        assert!(!Arc::ptr_eq(&owned(&s).data, &owned(&inverse).data));

        let bosonic_s = fixture().svd_compact().unwrap().1;
        let untouched = bosonic_s.twist(&[0]).unwrap();
        assert!(Arc::ptr_eq(owned(&bosonic_s), owned(&untouched)));
        let untouched_inverse = bosonic_s.twist_inverse(&[0]).unwrap();
        assert!(Arc::ptr_eq(owned(&bosonic_s), owned(&untouched_inverse)));
    }

    #[test]
    fn lazy_cat_reads_parent_storage_without_publishing_adjoint_caches() {
        let runtime = Runtime::builder().build().unwrap();
        let provider = Arc::new(U1FusionRule);
        let leg = |degeneracy| {
            GradedSpace::try_new(
                Arc::clone(&provider),
                [(U1Irrep::new(0), degeneracy)],
                false,
            )
            .unwrap()
        };
        let common = leg(3);
        let left = leg(2);
        let right = leg(4);
        let lhs: TensorMap<U1FusionRule, num_complex::Complex64> =
            TensorMap::rand_with_seed(&runtime, [&left], [&common], 773_101)
                .unwrap()
                .adjoint()
                .unwrap();
        let rhs: TensorMap<U1FusionRule, num_complex::Complex64> =
            TensorMap::rand_with_seed(&runtime, [&right], [&common], 773_102)
                .unwrap()
                .adjoint()
                .unwrap();

        assert_eq!(materialized_adjoint_builds(&lhs), 0);
        assert_eq!(materialized_adjoint_builds(&rhs), 0);
        let _ = lhs.catdomain(&rhs).unwrap();
        assert_eq!(materialized_adjoint_builds(&lhs), 0);
        assert_eq!(materialized_adjoint_builds(&rhs), 0);

        let upper: TensorMap<U1FusionRule, num_complex::Complex64> =
            TensorMap::rand_with_seed(&runtime, [&common], [&left], 773_103)
                .unwrap()
                .adjoint()
                .unwrap();
        let lower: TensorMap<U1FusionRule, num_complex::Complex64> =
            TensorMap::rand_with_seed(&runtime, [&common], [&right], 773_104)
                .unwrap()
                .adjoint()
                .unwrap();
        upper.catcodomain(&lower).unwrap();
        assert_eq!(materialized_adjoint_builds(&upper), 0);
        assert_eq!(materialized_adjoint_builds(&lower), 0);
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
        assert!(!Arc::ptr_eq(&owned(&scaled).data, &owned(&tensor).data));
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
        assert!(owned(&s).dense_cache.get().is_none(), "cache warm at birth");
        let materialized = s.data().as_ptr();
        assert!(owned(&s).dense_cache.get().is_some());

        // Hand-constructed body, same caveat as the gate above: shape changes
        // become compile errors.
        let reused = TensorMap {
            runtime: s.runtime.clone(),
            repr: owned_repr(TypedTensorBody {
                space: owned(&s).space.clone(),
                data: Arc::clone(&owned(&s).data),
                dense_cache: std::sync::OnceLock::new(),
            }),
        };
        assert!(owned(&reused).dense_cache.get().is_none());
        assert_ne!(reused.data().as_ptr(), materialized);
    }
}
