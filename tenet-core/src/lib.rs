#![forbid(unsafe_code)]

//! Core TensorMap-facing data structures for TeNeT.
//!
//! This crate owns TeNeT's public/core tensor view vocabulary. Lower-level
//! crates may lower these views to concrete strided kernels, but external
//! strided/backend types should not be required by TensorMap users.

use core::fmt;
use core::marker::PhantomData;
use core::ops::{Add, Mul};
use std::collections::{hash_map::Entry, BTreeMap};
use std::hash::Hash;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, OnceLock, RwLock, Weak};

#[cfg(test)]
use num_complex::Complex64;
use rustc_hash::FxHashMap;
use smallvec::SmallVec;
pub use tenet_sectors::CheckedGenericAdmissionMode;
pub use tenet_sectors::{
    product_fusion_rule, product_fusion_rule_with_codec, product_sector, BraidingStyleKind,
    CU1FusionRule, CU1Irrep, CanonicalUnitFusionRule, CategoricalScalar,
    CheckedCanonicalUnitFusionRule, CheckedFusionAlgebra, CheckedGenericFusion,
    CheckedGenericPivotal, CheckedGenericRigidSymbols, CoupledSectorFold, CoupledSectorFoldBuilder,
    FermionParityFusionRule, FibonacciFusionRule, FibonacciSector, FusionAlgebraError, FusionRule,
    FusionStyleKind, Fz2SectorLayout, GenericFArray, GenericRMatrix, InfallibleGeneric,
    MultiplicityFreeAdmissionMode, MultiplicityFreeFusionRule, MultiplicityFreeFusionSymbols,
    MultiplicityFreeRigidSymbols, PackedProductCodec, PackedSectorLayout, PhysicalBasisError,
    PhysicalFusionBasis, ProductFusionRule, ProductFusionRuleExt, ProductSector,
    ProductSectorCodec, ProductSectorCodecError, ProductSectorComponent, ProductSectorLayout,
    PromoteCoefficientScalar, RuleIdentity, SU2CoefficientError, SU2FusionRule, SU2Irrep,
    SectorCodec, SectorId, SectorOrderKey, SectorVec, Su2SectorLayout, SymbolShapeError,
    TensorKitProductCodec, TypedSectorAdmission, U1FusionRule, U1Irrep, U1SectorLayout,
    Z2FusionRule, Z2Irrep, ZNFusionRule, ZNIrrep, CU1_MAX_TWICE_CHARGE, SU2_MAX_DOUBLED_SPIN,
};
#[cfg(any(test, feature = "testing"))]
pub use tenet_sectors::{GenericFusionSymbols, GenericRigidSymbols};
#[cfg(feature = "racah-generated")]
pub use tenet_sectors::{SUNFusionRule, SUNFusionRuleError, SUNSymbolError};

pub mod axes;
mod storage;
pub use storage::*;
mod space;
pub use space::*;
mod sector;
pub use sector::*;
mod fusion_space;
pub use fusion_space::*;
mod fusion_tree;
pub use fusion_tree::*;
mod block_structure;
pub use block_structure::*;
mod cache;
pub use cache::{
    set_structure_cache_byte_budget, structure_cache_info, structure_cache_infos,
    StructureCacheInfo, StructureCacheKind,
};
mod tensor_map;
pub use tensor_map::*;
mod error;
pub use error::*;
#[cfg(feature = "testing")]
#[doc(hidden)]
pub mod testing;

#[cfg(test)]
mod tests;
#[cfg(test)]
pub(crate) use tests::test_support;
