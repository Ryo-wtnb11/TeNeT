//! Symmetry contract: everything a user needs to define a fusion rule or to
//! bound generic tensor code by one.
//!
//! A rule defined outside this workspace implements these traits and then
//! uses the same [`crate::typed::TensorMap`] API as the built-in rules
//! ([`U1FusionRule`], [`SU2FusionRule`], ...) exported here. This module is
//! the only path to each item it lists.

/// Built-in fusion rules, their sector labels, and product constructors.
pub use tenet_core::{
    product_fusion_rule, product_fusion_rule_with_codec, product_sector, CU1FusionRule, CU1Irrep,
    FermionParityFusionRule, FibonacciFusionRule, FibonacciSector, ProductFusionRule,
    ProductFusionRuleExt, ProductSector, SU2FusionRule, SU2Irrep, SectorId, U1FusionRule, U1Irrep,
    Z2FusionRule, Z2Irrep, ZNFusionRule, ZNIrrep,
};
pub use tenet_core::{
    BraidingStyleKind, FusionStyleKind, GenericFArray, GenericRMatrix, RuleIdentity,
    SectorOrderKey, SectorVec,
};
/// Rule traits: the fusion, symbol and admission interfaces a rule
/// implements, and the bounds generic code names.
pub use tenet_core::{
    CanonicalUnitFusionRule, CategoricalScalar, CheckedCanonicalUnitFusionRule,
    CheckedFusionAlgebra, CheckedGenericFusion, CheckedGenericPivotal, CheckedGenericRigidSymbols,
    CoupledSectorFold, FusionRule, MultiplicityFreeFusionRule, MultiplicityFreeFusionSymbols,
    MultiplicityFreeRigidSymbols, PhysicalFusionBasis, SectorCodec, TypedSectorAdmission,
};
/// Sealed admission modes: the values of `TypedSectorAdmission::Mode`.
#[doc(hidden)]
pub use tenet_core::{CheckedGenericAdmissionMode, MultiplicityFreeAdmissionMode};
/// Label codecs of a [`ProductFusionRule`], its third type
/// parameter, and the per-factor layouts they pack.
pub use tenet_core::{
    Fz2SectorLayout, PackedProductCodec, PackedSectorLayout, ProductSectorCodec,
    ProductSectorCodecError, ProductSectorLayout, Su2SectorLayout, TensorKitProductCodec,
    U1SectorLayout,
};
/// Vocabulary the rule traits' signatures use.
pub use tenet_core::{
    PhysicalBasisError, ProductSectorComponent, PromoteCoefficientScalar, SymbolShapeError,
};
#[cfg(feature = "racah-generated")]
pub use tenet_core::{SUNFusionRule, SUNFusionRuleError};
