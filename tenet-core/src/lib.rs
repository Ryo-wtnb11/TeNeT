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
    FermionParityFusionRule, FibonacciFusionRule, FusionAlgebraError, FusionRule, FusionStyleKind,
    Fz2SectorLayout, GenericFArray, GenericFusionSymbols, GenericRMatrix, GenericRigidSymbols,
    InfallibleGeneric, MultiplicityFreeAdmissionMode, MultiplicityFreeFusionRule,
    MultiplicityFreeFusionSymbols, MultiplicityFreeRigidSymbols, PackedProductCodec,
    PackedSectorLayout, ProductFusionRule, ProductFusionRuleExt, ProductSector, ProductSectorCodec,
    ProductSectorCodecError, ProductSectorComponent, ProductSectorLayout, PromoteCoefficientScalar,
    RuleIdentity, SU2FusionRule, SU2Irrep, SectorCodec, SectorId, SectorVec, Su2SectorLayout,
    SymbolShapeError, TensorKitProductCodec, TypedSectorAdmission, U1FusionRule, U1Irrep,
    U1SectorLayout, Z2FusionRule, Z2Irrep, ZNFusionRule, ZNIrrep, CU1_MAX_TWICE_CHARGE,
    SU2_MAX_DOUBLED_SPIN,
};
#[cfg(feature = "racah-generated")]
pub use tenet_sectors::{SUNFusionRule, SUNFusionRuleError};

mod core_rule_bridge;
pub use core_rule_bridge::{LoweredFusionTreeBuildError, LoweredMultiplicityFreeAlgebra};

include!("storage.rs");
include!("space.rs");
include!("sector.rs");
include!("fusion_space.rs");
include!("fusion_tree.rs");
include!("block_structure.rs");
include!("tensor_map.rs");
include!("error.rs");

#[cfg(test)]
include!("tests.rs");
