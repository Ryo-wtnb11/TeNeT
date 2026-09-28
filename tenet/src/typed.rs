//! Provider-typed facade: spaces and tensor maps that keep the concrete
//! fusion-rule type `R` and speak the provider's own sector labels.
//!
//! This is the canonical user facade re-exported by [`crate::prelude`]. `R`
//! stays concrete through monomorphized construction, so any provider —
//! including one defined downstream that implements the required admission and
//! operation capability traits — can drive it, and the categorical
//! identity of a tensor comes back as [`TypedSectorAdmission::Sector`] labels
//! instead of opaque [`tenet_core::SectorId`] keys. The engine itself never
//! sees a label; the codec is the single boundary where one enters or leaves.
//!
//! The exception is deliberate: [`TensorMap::subblock`] is the engine-level
//! layout view, and the [`tenet_core::BlockRef`] it returns carries the raw
//! [`tenet_core::BlockKey`]. Labels are what [`TensorMap::subblock_fusion_trees`]
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
//! use tenet::core::{
//!     product_sector, FermionParityFusionRule, ProductFusionRuleExt, U1FusionRule, U1Irrep,
//!     Z2Irrep,
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
//! different Rust types with different [`tenet_core::RuleIdentity`]s, and
//! converting between them is an explicit component swap; nothing here permutes
//! factors for you. Association is where TeNeT diverges from TensorKit rather
//! than mirrors it: TK's `⊠` flattens nested products into one
//! `ProductSector{Tuple{…}}`, so association is unobservable there, while
//! TeNeT keeps the nesting in the Rust type — see
//! [`tenet_core::ProductFusionRule`] for the full statement.
//!
//! The product-provider claim is subject to the capability bounds of the
//! operation being called. Host paths admit complex categorical scalars where
//! the corresponding categorical-scalar and recoupling bounds are present;
//! some decompositions, network paths, and CUDA remain narrower.
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
//! multiplicity-free providers, the `s` factor from `svd_compact` is compact
//! (and stays compact through `restrict_leg` on both legs); for checked `Generic` providers, only the `d` factor
//! from EIGH/EIG is compact, while checked SVD publishes its `s` factor densely.
//! A compact factor holds `Σ_c k_c` values rather than the `Σ_c k_c²`
//! block-diagonal buffer it would fill, which is what TensorKit's
//! `DiagonalTensorMap` is. It is a storage property and not a type: no signature
//! mentions it, [`TensorMap::dense_data`] refuses it (call
//! [`TensorMap::materialize`] for the dense buffer), and the operations that can exploit
//! it — [`TensorMap::compose`],
//! [`TensorMap::scale`], [`TensorMap::axpby`], [`TensorMap::adjoint`],
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
//!   transfer/device leaf. [`tenet_core::Placement`] is diagnostic metadata;
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

use num_complex::{Complex32, Complex64};
use smallvec::SmallVec;
use tenet_core::{
    validate_unit_layout_correspondence_checked,
    validate_unit_layout_correspondence_generic_checked, BlockKey, BlockRef, BlockStructure,
    BlockView, CanonicalUnitFusionRule, CategoricalScalar, CheckedCanonicalUnitFusionRule,
    CheckedFusionAlgebra, CheckedGenericAdmissionMode, CheckedGenericStructureError,
    CoupledSectorRegion, CoupledTreeExtent, FusionAlgebraError, FusionProductSpace,
    FusionTreeHomSpace, FusionTreePairKey, MultiplicityFreeAdmissionMode,
    MultiplicityFreeFusionSymbols, MultiplicityFreeRigidSymbols, MultiplicityIndex,
    PhysicalFusionBasis, PreparedTreePairOperation, ProductFusionRule, ProductSector,
    ProductSectorCodec, SectorId, SectorLeg, TypedSectorAdmission, UnitLegInsertion,
};
use tenet_core::{
    CheckedGenericFusion, CheckedGenericPivotal, CheckedGenericRigidSymbols, HostReadableStorage,
    Placement, TensorStorage,
};
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

pub use tenet_core::SectorCodec;
#[cfg(feature = "racah-generated")]
pub use tenet_core::{SUNFusionRule, SUNFusionRuleError};
/// Flat f64 CUDA storage used by explicit typed ownership transfer.
#[cfg(feature = "cuda")]
pub use tenet_tensors::cuda::CudaStorage;
#[cfg(feature = "cuda")]
use tenet_tensors::cuda::CudaStorageGemm;
pub use tenet_tensors::CheckedGenericPlanError;

/// Error returned by physical-basis expansion and projection.
pub type PhysicalDenseError<E> = tenet_tensors::PhysicalConversionError<E>;

/// Re-exported so `use tenet::typed::*` is self-sufficient apart from the
/// provider: every fallible method here returns this error.
pub use crate::error::{Alternative, Error};
/// Re-exported for the same reason as [`Error`]: every constructor here takes
/// a runtime. Both types are also in [`crate::prelude`]; re-exporting them
/// here is what lets a caller glob-import this module alone. The canonical
/// [`TensorMap`] and [`GradedSpace`] are also re-exported by [`crate::prelude`].
pub use crate::runtime::Runtime;
/// The spectrum-magnitude bound of [`GradedSpace::find_truncated`]. Concrete
/// `f64`/`Complex64` callers never name it, but a caller generic over the
/// payload must, so it is re-exported here rather than left unnameable
/// outside the crate.
pub use tenet_matrixalgebra::SpectrumMagnitude;
/// Named factor sets returned by the factorization methods of [`TensorMap`].
/// They are defined next to the expert factorizations in `tenet-matrixalgebra`,
/// which return the same types, and re-exported here and in [`crate::prelude`].
pub use tenet_matrixalgebra::{Eig, Eigh, LeftPolar, Lq, Qr, RightPolar, Svd};
/// Re-exported for the same reason as [`Error`] and [`Runtime`]:
/// [`GradedSpace::find_truncated`] takes one, so `use tenet::typed::*` would
/// not be self-sufficient without it.
pub use tenet_matrixalgebra::{Truncation, TruncationSpace};

use tenet_matrixalgebra::{
    rescaled_power_norm, BoundDynFactor, CheckedGenericFactorPlanError, FactorScalar,
};

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
use crate::RuntimeIdentity;

mod batched;
pub use batched::{
    BatchError, BatchMemberRepresentation, EighStackOutput, MemberFault, PreparedCompose,
    PreparedEighFull, SignatureField, StackedTensorMap, StructureSignature,
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
#[allow(unused_imports)]
pub(crate) use scalar::ScalarOps;
#[allow(unused_imports)]
use scalar::{host_add_impl, host_axpby_into, host_scale_impl, CheckedGenericSpectrumResult};
pub use scalar::{
    AdvancedLinalgScalar, FactorizationScalar, NetworkDegeneracyRestriction, TensorScalar,
};
#[cfg(feature = "cuda")]
pub use scalar::{CudaFactorizationPayload, CudaPayload};
mod numeric;
#[allow(unused_imports)]
pub(crate) use numeric::{
    absorb_mapped, coupled_region_inner, coupled_region_weighted_sum, sector_regions,
    validate_norm_p, weighted_inner, weighted_trace,
};
#[allow(unused_imports)]
use numeric::{
    copy_absorb_prefix, julia_complex32_reciprocal_wide, julia_complex64_reciprocal, max_abs,
    max_propagating_nan, pinv_seam_error, random_unit, random_unit_f32, robust_cinv, scaled_power,
    splitmix64,
};
mod block_layout;
#[cfg(feature = "cuda")]
#[allow(unused_imports)]
use block_layout::for_each_block_element;
#[cfg(test)]
#[allow(unused_imports)]
use block_layout::observe_cat_result_layout_build;
#[allow(unused_imports)]
pub(crate) use block_layout::{
    apply_fill, cat_homspace, cat_logical_block_key, check_flip_layout_identity, compile_cat_plan,
    flip_block_factor, flip_toggled_homspace, logical_adjoint_axes_to_parent,
    lower_adjoint_tree_transform_operation, map_checked_unit_layout_error, oplus_sector_legs,
    reject_unbraided_nonunit_legs, scale_blocks_impl, twist_block_factor,
    twist_factor_with_inverse, twist_is_identity_over_blocks, validate_axis_permutation,
    validate_contracted_axes, with_planar_axes, CatCopyPlan, CatOperandLayout, Fill,
    PlanarRequestKind, TensorOrientation,
};
#[allow(unused_imports)]
use block_layout::{
    cat_region_tree_orders_match, cat_source_blocks, cat_source_regions,
    cat_source_regions_if_monotone, cat_storage_axis, fill_block_elements, fuse_sector_content,
    logical_adjoint_axis_to_parent, scale_strided_block, uncoupled_sector_of_leg,
};
pub use block_layout::{Direction, Duality, Side};
#[cfg(test)]
#[allow(unused_imports)]
pub(crate) use block_layout::{CAT_PLAN_DECLINES_ORIENTED, CAT_RESULT_LAYOUT_BUILDS};
mod cuda_factor;
#[cfg(feature = "cuda")]
#[allow(unused_imports)]
pub(crate) use cuda_factor::{
    assemble_aligned_left_factor, assemble_left_factor, assemble_right_factor, copy_whole_factor,
    cuda_download_spectra, cuda_hermitian_regions, cuda_qr_region, cuda_svd_region, dense_err,
    fill_diagonal_values, typed_cuda_eigh_region, upload_selector,
};
#[cfg(feature = "cuda")]
#[allow(unused_imports)]
use cuda_factor::{
    compile_cuda_eigh_plan, compile_cuda_qr_plan, cuda_factor_layout_is_aligned,
    cuda_qr_tree_extents_match, download_cuda_reduction_partials,
    validate_cuda_reduction_placement, validate_cuda_svd_middle_regions, TypedCudaEighPlan,
    TypedCudaEighRoute, TypedCudaQrPlan, TypedCudaQrRoute, TypedCudaQrScratch, TypedCudaSvdScratch,
};
#[cfg(all(test, feature = "cuda"))]
#[allow(unused_imports)]
use cuda_factor::{
    observe_cuda_arithmetic, observe_cuda_qr_output_upload,
    observe_cuda_svd_final_storage_creation, update_cuda_qr_observation,
    update_cuda_svd_observation, CudaQrObservation, CudaSvdObservation,
    CUDA_ARITHMETIC_OBSERVATION, CUDA_EIGH_FAILURE, CUDA_EIGH_SELECTOR_UPLOADS, CUDA_EIGH_TREEWISE,
    CUDA_QR_OBSERVATION, CUDA_REDUCTION_BUFFER_OBSERVATION, CUDA_SVD_OBSERVATION,
    CUDA_SVD_TREEWISE,
};
#[cfg(all(test, feature = "cuda"))]
#[allow(unused_imports)]
pub(crate) use cuda_factor::{
    observe_cuda_factor_assembly_gemm, observe_cuda_factor_copy, observe_cuda_qr_decomposition,
    observe_cuda_qr_selector_upload, observe_cuda_svd_decomposition,
};
mod generic_error;
pub use generic_error::GenericTensorError;
mod dispatch;
#[allow(unused_imports)]
use dispatch::{MultiplicityFreeContractExecution, MultiplicityFreeTransformExecution};
pub use dispatch::{
    TypedSpaceModeDispatch, TypedTensorAddScaleDispatch, TypedTensorAdjointDispatch,
    TypedTensorConstructionDispatch, TypedTensorContractDispatch, TypedTensorEigDispatch,
    TypedTensorEigValsDispatch, TypedTensorEighDispatch, TypedTensorEighValsDispatch,
    TypedTensorExpDispatch, TypedTensorFlipDispatch, TypedTensorFullLqDispatch,
    TypedTensorFullQrDispatch, TypedTensorInvDispatch, TypedTensorLqDispatch,
    TypedTensorModeDispatch, TypedTensorNullDispatch, TypedTensorPinvDispatch,
    TypedTensorPolarDispatch, TypedTensorProductDispatch, TypedTensorQrDispatch,
    TypedTensorReductionDispatch, TypedTensorRootDispatch, TypedTensorSolveDispatch,
    TypedTensorSvdDispatch, TypedTensorSvdValsDispatch, TypedTensorTraceDispatch,
    TypedTensorTransformDispatch, TypedTensorTwistDispatch, TypedTruncationDispatch,
};
mod mode_dispatch;
#[allow(unused_imports)]
use mode_dispatch::{
    checked_generic_flip_destination, checked_generic_owned_bodies, checked_generic_solve_into,
    contract_checked_generic, flip_checked_generic, flip_checked_generic_owned,
    twist_checked_generic, twist_checked_generic_owned,
};
mod checked_generic_contract;
#[allow(unused_imports)]
pub(crate) use checked_generic_contract::TypedFacadeError;
#[allow(unused_imports)]
use checked_generic_contract::{
    compose_multiplicity_free, compose_multiplicity_free_with_lane, contract_destination,
    contract_multiplicity_free, contract_multiplicity_free_into, trace_pair_axes,
    trace_pairs_checked_generic, write_dense_identity_blocks, write_identity_blocks_generic,
    TracePairAxes,
};
pub use checked_generic_contract::{
    reject_non_symmetric_contraction, NON_SYMMETRIC_CONTRACTION_UNSUPPORTED,
};
mod space;
#[allow(unused_imports)]
use space::{
    require_restriction_set, require_selected_leg_of, restricted_space, restriction_starts,
    space_with_replaced_legs,
};
pub use space::{GradedSpace, LegSelection, TruncatedSelection};
mod fusion_tree;
#[allow(unused_imports)]
use fusion_tree::{
    add_spectrum_into, decode_block_fusion_trees, decode_sectors, diagonal_factor_on,
    diagonal_factor_on_checked, is_diagonal_bond_space, map_block_fusion_trees, map_spectrum,
    prepare_product_operand, scatter_spectrum, spectra_disagree, wrap_factor_on,
    PreparedProductOperand, TreeExtents, TypedData,
};
pub use fusion_tree::{
    BlockFusionTrees, CoupledBlock, CoupledBlockPayload, FusionTreeLabels, SectorSpectrum,
};
mod tensor_repr;
#[cfg(test)]
#[allow(unused_imports)]
pub(crate) use tensor_repr::ADJOINT_MATERIALIZATIONS;
#[allow(unused_imports)]
use tensor_repr::{
    borrowed_view_unsupported, owned_repr, TypedAdjointView, TypedTensorBody, TypedTensorRepr,
    ViewAdjoint,
};
#[cfg(test)]
#[allow(unused_imports)]
use tensor_repr::{
    observe_adjoint_materialization, DIAGONAL_MATERIALIZATIONS, UNCACHED_ADJOINT_MATERIALIZATIONS,
};
pub use tensor_repr::{
    NetworkPayloadStorage, NetworkReuseClass, PayloadConversion, PhysicalDense,
    RuntimeDetachedTensorMap, TensorMap, TensorRef,
};
mod cuda_ops;
#[cfg(feature = "cuda")]
pub use cuda_ops::CudaTracePairs;
mod transform_ops;
#[allow(unused_imports)]
use transform_ops::{
    braid_operation, generic_insert_unit, generic_remove_unit, map_spectrum_dtype,
    tree_operation_matches_axes,
};
pub use transform_ops::{ContractSpec, TypedTensorUnitDispatch};

/// Representation gates for [`TypedTensorBody`] (#580 PR 0).
///
/// These live inside the module on purpose: the properties under test are the
/// private layout — which `Arc` holds what — and asserting them from
/// `tests/` would mean publishing accessors the facade does not otherwise
/// need. Public semantic oracles live in `tests/typed_facade.rs`; dense-cache
/// behavior lives in `tests/typed_diagonal_allocations.rs`.
#[cfg(test)]
mod representation_gates;
