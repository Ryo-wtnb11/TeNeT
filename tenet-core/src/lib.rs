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
    FusionStyleKind, Fz2SectorLayout, GenericFArray, GenericFusionSymbols, GenericRMatrix,
    GenericRigidSymbols, InfallibleGeneric, MultiplicityFreeAdmissionMode,
    MultiplicityFreeFusionRule, MultiplicityFreeFusionSymbols, MultiplicityFreeRigidSymbols,
    PackedProductCodec, PackedSectorLayout, PhysicalBasisError, PhysicalFusionBasis,
    ProductFusionRule, ProductFusionRuleExt, ProductSector, ProductSectorCodec,
    ProductSectorCodecError, ProductSectorComponent, ProductSectorLayout, PromoteCoefficientScalar,
    RuleIdentity, SU2FusionRule, SU2Irrep, SectorCodec, SectorId, SectorOrderKey, SectorVec,
    Su2SectorLayout, SymbolShapeError, TensorKitProductCodec, TypedSectorAdmission, U1FusionRule,
    U1Irrep, U1SectorLayout, Z2FusionRule, Z2Irrep, ZNFusionRule, ZNIrrep, CU1_MAX_TWICE_CHARGE,
    SU2_MAX_DOUBLED_SPIN,
};
#[cfg(feature = "racah-generated")]
pub use tenet_sectors::{SUNFusionRule, SUNFusionRuleError};

#[rustfmt::skip] // #1817 formats the former include! fragments in a separate formatting-only PR.
mod storage;
pub use storage::*;
#[rustfmt::skip] // #1817 formats the former include! fragments in a separate formatting-only PR.
mod space;
pub use space::*;
#[rustfmt::skip] // #1817 formats the former include! fragments in a separate formatting-only PR.
mod sector;
pub use sector::*;
#[rustfmt::skip] // #1817 formats the former include! fragments in a separate formatting-only PR.
mod fusion_space;
pub use fusion_space::*;
#[rustfmt::skip] // #1817 formats the former include! fragments in a separate formatting-only PR.
mod fusion_tree;
pub use fusion_tree::*;
#[rustfmt::skip] // #1817 formats the former include! fragments in a separate formatting-only PR.
mod block_structure;
pub use block_structure::*;
#[rustfmt::skip] // #1817 formats the former include! fragments in a separate formatting-only PR.
mod tensor_map;
pub use tensor_map::*;
#[rustfmt::skip] // #1817 formats the former include! fragments in a separate formatting-only PR.
mod error;
pub use error::*;

#[cfg(test)]
#[rustfmt::skip] // #1817 formats the former include! fragments in a separate formatting-only PR.
mod tests;
#[cfg(test)]
pub(crate) use tests::test_support;
