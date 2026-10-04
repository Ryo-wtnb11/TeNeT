//! Provider-typed facade: spaces and tensor maps that keep the concrete
//! fusion-rule type `R` and speak the provider's own sector labels.
//!
//! This is the canonical user facade. `R`
//! stays concrete through monomorphized construction, so any provider —
//! including one defined downstream that implements the required admission and
//! operation capability traits — can drive it, and the categorical
//! identity of a tensor comes back as [`TypedSectorAdmission::Sector`] labels
//! instead of opaque [`crate::sector::SectorId`] keys. The engine itself never
//! sees a label; the codec is the single boundary where one enters or leaves.
//!
//! The exception is deliberate: [`TensorMap::subblock`] is the engine-level
//! layout view, and the [`crate::expert::BlockRef`] it returns carries the raw
//! [`crate::typed::BlockKey`]. Labels are what [`TensorMap::subblock_fusion_trees`]
//! is for.
//!
//! # Product symmetries
//!
//! A product symmetry needs no new constructor here, and no new type in the
//! engine. A product of providers *is* a provider — build it with
//! [`crate::sector::ProductFusionRuleExt::product`] and label it with
//! [`crate::sector::product_sector`] — so this facade drives `fZ2 ⊠ U(1)`, or any
//! other ordered product of admitted components, through the same
//! [`GradedSpace::try_new`] and [`TensorMap::zeros`] as a single symmetry:
//!
//! ```
//! use std::sync::Arc;
//! use tenet::sector::{
//!     product_sector, FermionParityFusionRule, ProductFusionRuleExt, U1FusionRule, U1Irrep, Z2Irrep,
//! };
//! use tenet::typed::{Error, GradedSpace, Runtime, TensorMap};
//!
//! # fn main() -> Result<(), Error> {
//! let runtime = Runtime::builder().build()?;
//! let rule = FermionParityFusionRule.product(U1FusionRule);
//!
//! let even = product_sector(Z2Irrep::EVEN, U1Irrep::new(0));
//! let odd = product_sector(Z2Irrep::ODD, U1Irrep::new(1));
//! let v = GradedSpace::try_new(Arc::new(rule), [(even, 2), (odd, 1)])?;
//!
//! let t: TensorMap<_, f64> = TensorMap::zeros(&runtime, [&v], [&v])?;
//! assert_eq!(t.subblock_count(), 2);
//! assert_eq!(t.subblock_fusion_trees(0)?.coupled(), &even);
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
//! different Rust types with different [`crate::sector::RuleIdentity`]s, and
//! converting between them is an explicit component swap; nothing here permutes
//! factors for you. Association is where TeNeT diverges from TensorKit rather
//! than mirrors it: TK's `⊠` flattens nested products into one
//! `ProductSector{Tuple{…}}`, so association is unobservable there, while
//! TeNeT keeps the nesting in the Rust type — see
//! [`crate::sector::ProductFusionRule`] for the full statement.
//!
//! The product-provider claim is subject to the capability bounds of the
//! operation being called. Host paths admit complex categorical scalars where
//! the corresponding categorical-scalar and recoupling bounds are present;
//! some decompositions, network paths, and CUDA remain narrower.
//!
//! **`ProductSector` is not `ProductSpace`.** [`crate::sector::ProductSector`] is
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
//! ([`TensorMap::zeros`], [`TensorMap::from_subblock_fn`],
//! [`TensorMap::rand_with_seed`], [`TensorMap::isomorphism`],
//! [`TensorMap::isometry`]),
//! inspection ([`TensorMap::codomain`], [`TensorMap::domain`],
//! [`TensorMap::subblock_fusion_trees`], [`TensorMap::subblock`],
//! [`TensorMap::subblock_count`], [`TensorMap::block`], [`TensorMap::blocks`],
//! [`TensorMap::dense_data`], [`TensorMap::runtime`]),
//! the index-manipulation and contraction operations
//! ([`TensorMap::permute`], [`TensorMap::braid`], [`TensorMap::transpose`],
//! [`TensorMap::repartition`],
//! [`TensorMap::contract`],
//! [`TensorMap::compose`]), the scalar operations
//! ([`TensorMap::axpby`], [`TensorMap::scale`], [`TensorMap::norm`],
//! [`TensorMap::inner`], [`TensorMap::tr`], [`TensorMap::trace_pairs`],
//! [`TensorMap::adjoint`]), the factorizations ([`TensorMap::svd_compact`],
//! [`TensorMap::svd_full`], [`TensorMap::svd_vals`],
//! [`TensorMap::qr_compact`], [`TensorMap::qr_full`],
//! [`TensorMap::lq_compact`], [`TensorMap::lq_full`],
//! [`TensorMap::left_polar`], [`TensorMap::right_polar`],
//! [`TensorMap::left_null`], [`TensorMap::right_null`]) and the **truncation
//! primitives** a truncated factorization is composed from
//! ([`TensorMap::diagview`], [`GradedSpace::find_truncated`],
//! [`GradedSpace::truncspace`], [`TensorMap::restrict_leg`]; see the
//! tutorial) and — with
//! issue #570
//! — the **eigendecompositions** ([`TensorMap::eigh_full`],
//! [`TensorMap::eigh_vals`], [`TensorMap::eig_full`], [`TensorMap::eig_vals`])
//! and
//! — with issue #576 — the **matrix functions** ([`TensorMap::exp`],
//! [`TensorMap::inv`], [`TensorMap::pinv`], and the elementwise
//! [`TensorMap::map_diagonal`] on compact diagonals) and — with
//! issue #580 — the **typed inspection, scalar and conversion group**
//! ([`TensorMap::rank`], [`TensorMap::codomain_rank`],
//! [`TensorMap::domain_rank`], [`TensorMap::rank`], [`TensorMap::leg_dims`],
//! [`TensorMap::codomain`], [`TensorMap::domain`],
//! [`TensorMap::scalar`], [`TensorMap::zeros_like`], [`TensorMap::convert`],
//! [`TensorMap::re`], [`TensorMap::im`]) and the **concatenation/absorb
//! group** ([`TensorMap::cat`], [`TensorMap::absorb`]) and the **index-unit
//! group** ([`TensorMap::twist`], [`TensorMap::flip`],
//! [`TensorMap::insert_unit`], [`TensorMap::remove_unit`]) and — with
//! issue #1323 — the **explicit payload precision conversion**
//! [`TensorMap::convert`] (exact widenings and the two lossy narrowings), with
//! no implicit conversion anywhere.
//!
//! Issue #570 also gave the facade **compact diagonal storage**. For
//! Host providers, the `s` factor from `svd_compact` is compact (and stays
//! compact through `restrict_leg` on both legs). Checked `Generic` EIGH/EIG
//! likewise store their `d` factor compactly.
//! A compact factor holds `Σ_c k_c` values rather than the `Σ_c k_c²`
//! block-diagonal buffer it would fill, which is what TensorKit's
//! `DiagonalTensorMap` is. It is a storage property and not a type: no signature
//! mentions it, [`TensorMap::dense_data`] refuses it (call
//! [`TensorMap::materialize`] for the dense buffer), and the operations that can exploit
//! it — [`TensorMap::compose`], admitted bond-scaling [`TensorMap::contract`],
//! [`TensorMap::cat`], [`TensorMap::absorb`], [`TensorMap::scale`],
//! [`TensorMap::axpby`], [`TensorMap::adjoint`],
//! [`TensorMap::trace_pairs`] on its full-pair arm, and the reductions — do so
//! silently. Rank-(1,1) [`TensorMap::permute`] and [`TensorMap::transpose`]
//! swaps keep the compact result; an admitted [`TensorMap::braid`] reads the
//! compact source directly and publishes a dense result. Other transform and
//! contraction geometries use the documented dense route.
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
//!   hyperbolic members, `log`, `sylvester` and a general `sqrt` — is out by
//!   decision, not by queue position (issue #576). Every one of them is a
//!   spectral function or a solve over the same seams, so adding them is
//!   mechanical; what is missing is a reason to. Right solves and integer
//!   powers are compositions of [`TensorMap::adjoint`], [`TensorMap::solve`],
//!   [`TensorMap::inv`] and [`TensorMap::compose`] (the zeroth power is
//!   [`TensorMap::isomorphism`] of the domain onto itself), and elementwise
//!   maps of a compact spectrum go through [`TensorMap::map_diagonal`].
//!   One capability gap still stands behind that line: general endomorphism
//!   **`sqrt`** needs a Schur seam, and that seam does not exist below this
//!   facade. The one that used to stand beside it is closed —
//!   [`TensorMap::exp`] accepts any endomorphism since issue #577, through a
//!   blockwise Padé arm.
//! - Some **outer multiplicity factorization** leaves remain outside this
//!   facade. Checked `Generic` providers have provider-neutral SVD/QR/LQ,
//!   numerical null spaces, and the admitted matrix-function subset; each leaf
//!   documents its own lazy-adjoint and compact-storage boundary.
//! - **Device execution** is absent, not device representation: the body can
//!   carry a non-host `S` through [`TensorMap<R, D, S>`], while public
//!   construction and arithmetic deliberately remain on the default `Vec<D>`
//!   storage. Non-host operations wait for an explicit, [`Runtime`]-dependent
//!   transfer/device leaf. [`crate::expert::Placement`] is diagnostic metadata;
//!   no operation dispatches on it.
//! - The **operator overloads** (`impl Add`, `impl Mul`) are out because they
//!   cannot return `Result`; panicking operators would contradict this
//!   facade's passthrough-error contract. Adding them later is not a breaking
//!   change.
//! - `conj` stays design-gated on its open correctness question for
//!   non-self-dual sectors. [`TensorMap::adjoint`] is the TensorKit-style lazy
//!   parent view for dense storage. A compact diagonal, in either admission
//!   mode, keeps its direct `O(Σ_c k_c)` conjugation path.
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
use std::sync::Arc;

use smallvec::SmallVec;
use tenet_core::{
    validate_unit_layout_correspondence_checked,
    validate_unit_layout_correspondence_generic_checked, BlockRef, BlockStructure, BlockView,
    CoupledSectorRegion, CoupledTreeExtent, FusionProductSpace, FusionTreeHomSpace,
    PreparedTreePairOperation, ProductFusionRule, ProductSector, SectorId, SectorLeg,
    UnitLegInsertion,
};
use tenet_core::{HostReadableStorage, Placement, TensorStorage};
#[cfg(feature = "cuda")]
use tenet_dense::{
    cuda_copy_region_into, cuda_eigh_region, cuda_gemm_region_into,
    cuda_hermitian_regions as dense_cuda_hermitian_regions, cuda_qr_region as dense_cuda_qr_region,
    cuda_svd_gauge_phases, cuda_svd_region as dense_cuda_svd_region, CudaDenseContext,
    CudaDenseStorage, CudaSvdGaugeWeights,
};
use tenet_operations::scale_value;
#[cfg(feature = "cuda")]
use tenet_operations::{CudaTreeTransformDestination, StorageGemm};
use tenet_tensors::{
    expand_physical_host, project_physical_host, tensorcompose_owned_checked_generic_in_context,
    tensorcontract_owned_checked_generic_in_context,
    tree_transform_dyn_owned_checked_generic_input_in_context, BoundDynamicFusionMapSpace,
    BoundDynamicTensorRef, CheckedTreeTransformInput, DynamicFusionMapSpace, OutputAxisOrder,
    OwnedCatCopy, OwnedCatSide, TensorContractSpec, TreeTransformOperation,
    TreeTransformOperationKind, ValidatedDynamicFusionLayout,
};

use crate::sector::{
    CanonicalUnitFusionRule, CategoricalScalar, CheckedCanonicalUnitFusionRule,
    CheckedFusionAlgebra, CheckedGenericAdmissionMode, CheckedGenericFusion, CheckedGenericPivotal,
    CheckedGenericRigidSymbols, MultiplicityFreeAdmissionMode, MultiplicityFreeFusionSymbols,
    MultiplicityFreeRigidSymbols, PhysicalFusionBasis, ProductSectorCodec, SectorCodec,
    TypedSectorAdmission,
};
/// Causes carried by [`Error`] and [`GenericTensorError`] variants.
pub use tenet_core::{CheckedGenericStructureError, CoreError, FusionAlgebraError};
/// Flat f64 CUDA storage used by explicit typed ownership transfer.
#[cfg(feature = "cuda")]
pub use tenet_tensors::cuda::CudaStorage;
#[cfg(feature = "cuda")]
use tenet_tensors::cuda::CudaStorageGemm;
pub use tenet_tensors::CheckedGenericPlanError;
pub use tenet_tensors::OperationError;

/// Error returned by physical-basis expansion and projection.
pub type PhysicalDenseError<E> = tenet_tensors::PhysicalConversionError<E>;

/// Every fallible method here returns this error.
pub use crate::error::{Alternative, Error};
#[cfg(feature = "cuda")]
pub use crate::runtime::CudaTreeTransformStats;
/// The runtime every constructor here takes, its builder and configuration,
/// and its cache observability.
pub use crate::runtime::{
    LinalgBackend, Runtime, RuntimeBuilder, RuntimeConfigError, RuntimeTreeTransformCacheInfo,
    TreeTransformCacheInfo,
};
/// The complex payload scalars.
pub use num_complex::{Complex32, Complex64};
/// Block and fusion-tree keys and vertex labels read from a [`TensorMap`],
/// and the error of a checked fusion-space query.
pub use tenet_core::{
    BlockKey, BlockKeyKind, CheckedFusionSpaceError, FusionTreeGroupKey, FusionTreeKey,
    FusionTreePairKey, MultiplicityIndex, OpaqueBlockKey,
};
/// The device-scalar supertrait of [`CudaPayload`] and its real counterpart.
#[cfg(feature = "cuda")]
pub use tenet_dense::{CudaRealScalar, CudaScalar};
/// The dense-scalar supertrait of the payload traits; its `Eig` is the
/// payload of [`TensorMap::eig_full`].
pub use tenet_matrixalgebra::FactorScalar;
/// The spectrum-magnitude bound of [`GradedSpace::find_truncated`]. Concrete
/// `f64`/`Complex64` callers never name it, but a caller generic over the
/// payload must, so it is re-exported here rather than left unnameable
/// outside the crate.
pub use tenet_matrixalgebra::SpectrumMagnitude;
/// Named factor sets returned by the factorization methods of [`TensorMap`].
/// They are defined next to the expert factorizations in `tenet-matrixalgebra`,
/// which return the same types, and re-exported here.
pub use tenet_matrixalgebra::{Eig, Eigh, LeftPolar, Lq, Qr, RightPolar, Svd};
/// [`GradedSpace::find_truncated`] and the truncated factorizations take
/// these.
pub use tenet_matrixalgebra::{Truncation, TruncationError, TruncationSpace};
/// The coefficient action a payload scalar supports; physical-basis
/// expansion and projection name it as a bound.
pub use tenet_tensors::RecouplingCoefficientAction;

use tenet_matrixalgebra::seam::{rescaled_power_norm, CheckedGenericFactorPlanError};
use tenet_matrixalgebra::BoundDynFactor;

use crate::runtime::RuntimeIdentity;
use crate::runtime::{Ctx, Ctxs};
pub use crate::tensor_core::CheckedGenericTensorProductError;
use crate::tensor_core::{
    internal_layout_error, oriented_contract_destination, tensorcompose_owned_multiplicity_free,
    tensorcontract_oriented_multiplicity_free,
    tensorcontract_oriented_multiplicity_free_into_slice,
    tensorcontract_owned_multiplicity_free_into_slice, tensorproduct_owned_checked_generic,
    tensorproduct_owned_multiplicity_free, tree_transform_owned_multiplicity_free,
    tree_transform_owned_multiplicity_free_into, OrientedContractionKind,
};

mod batched;
#[allow(deprecated)]
pub use batched::{
    BatchError, BatchMemberRepresentation, ComposePlan, ComposeWorkspace, ContractPlan,
    ContractWorkspace, EighFullPlan, EighFullWorkspace, EighStackOutput, MemberFault,
    PreparedCompose, PreparedEighFull, SignatureField, StackedTensorMap, StructureSignature,
};

#[cfg(test)]
mod contract_stacking_tests;
mod serialization;
#[cfg(test)]
mod view_tests;
pub use serialization::{
    DecodeError, DecodeLimits, EncodeError, PersistedScalar, TypedPersistenceCodec,
};

// --- split leaf #1587: generated module wiring below ---
mod scalar;
use linear_ops::{host_axpby_into, require_destination_space, unique_dense_destination};
pub(crate) use scalar::ScalarOps;
pub use scalar::{AdvancedLinalgScalar, FactorizationScalar, TensorScalar};
#[cfg(feature = "cuda")]
pub use scalar::{CudaFactorizationPayload, CudaPayload};
mod numeric;
pub(crate) use numeric::{
    absorb_compact_source, absorb_mapped, coupled_region_inner, coupled_region_weighted_sum,
    multiplicity_free_dim, sector_regions, validate_norm_p, weighted_trace,
};
use numeric::{max_abs, pinv_seam_error, scaled_power};
mod block_layout;
#[cfg(feature = "cuda")]
use block_layout::for_each_block_element;
#[cfg(feature = "cuda")]
use block_layout::logical_adjoint_axis_to_parent;
#[cfg(test)]
pub(crate) use block_layout::CAT_PLAN_DECLINES_ORIENTED;
pub(crate) use block_layout::{
    apply_fill, cat_homspace, check_flip_layout_identity, compile_cat_plan, flip_block_factor,
    flip_toggled_homspace, logical_adjoint_axes_to_parent, lower_adjoint_tree_transform_operation,
    map_checked_unit_layout_error, oplus_sector_legs, reject_unbraided_nonunit_legs,
    scale_blocks_impl, twist_block_factor, twist_factor_with_inverse,
    twist_is_identity_over_blocks, with_planar_axes, CatOperandData, CatOperandLayout, Fill,
    PlanarRequestKind,
};
use block_layout::{fuse_sector_content, uncoupled_sector_of_leg};
pub use block_layout::{Direction, Duality, Side};
#[cfg(feature = "cuda")]
mod cuda_factor;
#[cfg(feature = "cuda")]
pub(crate) use cuda_factor::{
    assemble_aligned_left_factor, assemble_left_factor, assemble_right_factor, copy_whole_factor,
    cuda_download_spectra, cuda_hermitian_regions, cuda_qr_region, cuda_svd_region, dense_err,
    fill_diagonal_values, typed_cuda_eigh_region, upload_selector,
};
#[cfg(feature = "cuda")]
use cuda_factor::{
    compile_cuda_eigh_plan, compile_cuda_qr_plan, download_cuda_reduction_partials,
    validate_cuda_reduction_placement, validate_cuda_svd_middle_regions, TypedCudaEighPlan,
    TypedCudaEighRoute, TypedCudaQrPlan, TypedCudaQrRoute, TypedCudaQrScratch, TypedCudaSvdScratch,
};
#[cfg(all(test, feature = "cuda"))]
use cuda_factor::{
    cuda_factor_layout_is_aligned, cuda_qr_tree_extents_match, observe_cuda_arithmetic,
    observe_cuda_qr_output_upload, observe_cuda_svd_final_storage_creation,
    CUDA_ARITHMETIC_OBSERVATION, CUDA_EIGH_FAILURE, CUDA_EIGH_SELECTOR_UPLOADS, CUDA_EIGH_TREEWISE,
    CUDA_QR_OBSERVATION, CUDA_REDUCTION_BUFFER_OBSERVATION, CUDA_SVD_OBSERVATION,
    CUDA_SVD_TREEWISE,
};
mod generic_error;
pub use generic_error::GenericTensorError;
mod dispatch;
use dispatch::{MultiplicityFreeContractExecution, MultiplicityFreeTransformExecution};
pub use dispatch::{
    TypedAdjointSpace, TypedSpaceModeDispatch, TypedTensorConstructionDispatch,
    TypedTensorContractDispatch, TypedTensorFlipDispatch, TypedTensorModeDispatch,
    TypedTensorProductDispatch, TypedTensorRootDispatch, TypedTensorTraceDispatch,
    TypedTensorTransformDispatch, TypedTensorTwistDispatch, TypedTruncationDispatch,
};
mod factorize;
pub use factorize::{
    FusionMode, TypedTensorEigDispatch, TypedTensorEighDispatch, TypedTensorExpDispatch,
    TypedTensorInvDispatch, TypedTensorPinvDispatch, TypedTensorSolveDispatch,
};
mod checked_generic_contract;
mod mode_dispatch;
pub(crate) use checked_generic_contract::TypedFacadeError;
use checked_generic_contract::{trace_source, write_identity_blocks_generic};
use mode_dispatch::checked_compact_spectrum_layout;
#[doc(hidden)]
pub use tenet_tensors::{reject_non_symmetric_contraction, NON_SYMMETRIC_CONTRACTION_UNSUPPORTED};
mod space;
use space::{
    require_restriction_set, require_selected_leg_of, restricted_space, restriction_starts,
    space_with_replaced_legs,
};
pub use space::{GradedSpace, LegSelection, TruncatedSelection};
mod fusion_tree;
#[cfg(test)]
use fusion_tree::full_svd_spectrum_matches_bonds;
use fusion_tree::{
    add_spectrum_into, decode_block_fusion_trees, diagonal_factor_on, diagonal_factor_on_bound,
    diagonal_factor_on_checked, diagonal_factor_on_source_checked, exp_spectrum,
    full_svd_compact_bond, full_svd_compact_layout, inv_spectrum, is_diagonal_bond_space,
    map_spectrum, prepare_product_operand, reject_singular_compact_divisor, scatter_spectrum,
    spectra_disagree, wrap_factor_on, TypedData,
};
pub use fusion_tree::{
    BlockFusionTrees, CoupledBlock, CoupledBlockPayload, FusionTreeLabels, SectorSpectrum,
};
mod tensor_repr;
#[cfg(test)]
pub(crate) use tensor_repr::ADJOINT_MATERIALIZATIONS;
use tensor_repr::{
    borrowed_view_unsupported, owned_repr, TypedAdjointView, TypedTensorBody, TypedTensorRepr,
};
#[cfg(test)]
use tensor_repr::{
    observe_adjoint_materialization, DIAGONAL_MATERIALIZATIONS, UNCACHED_ADJOINT_MATERIALIZATIONS,
};
pub use tensor_repr::{PayloadConversion, PhysicalDense, TensorMap, TensorRef};
mod cat;
mod construction;
mod contract_ops;
pub use contract_ops::ContractSpec;
/// The seam `tenet-network` drives; not part of the facade.
#[doc(hidden)]
pub mod __network;
#[cfg(feature = "cuda")]
mod cuda_contract;
#[cfg(feature = "cuda")]
mod cuda_ops;
#[cfg(feature = "cuda")]
mod cuda_scalar;
#[cfg(feature = "cuda")]
mod cuda_transfer;
#[cfg(feature = "cuda")]
mod cuda_transform;
mod inspection;
mod linear_ops;
mod network_seam;
#[cfg(test)]
use network_seam::NetworkDegeneracyRestriction;
mod reduction_ops;
mod restrict;
mod transform_ops;
mod twist_flip;
use transform_ops::map_spectrum_dtype;
pub use transform_ops::TypedTensorUnitDispatch;
#[cfg(feature = "cuda")]
use transform_ops::{braid_operation, tree_operation_matches_axes};

/// In-module gates on the typed facade's private state.
///
/// These live inside the module on purpose: they assert what `tests/` cannot
/// see without accessors the facade does not otherwise need — which `Arc`
/// holds what ([`TypedTensorBody`], #580 PR 0), dense-kernel and solver call
/// counts, cache warmth, and device call counts. The module itself keeps the
/// representation gates and the shared fixtures; its children group the
/// remaining gates by operation family: multiplicity-free and checked
/// factorizations, lazy-adjoint redirects, CUDA, `*_into` overwrite, compact
/// transforms/cat/absorb, contraction and network restriction. Public semantic
/// oracles live in `tests/typed_facade.rs`; dense-cache behavior lives in
/// `tests/typed_diagonal_allocations.rs`.
#[cfg(test)]
mod representation_gates;
