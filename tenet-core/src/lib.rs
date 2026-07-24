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

use num_complex::Complex64;
use rustc_hash::FxHashMap;
use smallvec::{smallvec, SmallVec};
pub use tenet_sectors::{
    product_sector, BraidingStyleKind, CheckedFusionAlgebra, CoupledSectorFold,
    FermionParityFusionRule, FusionAlgebraError, FusionRule, FusionStyleKind, Fz2SectorLayout,
    GenericBraidScalar, GenericFArray, GenericFusionSymbols, GenericRMatrix, GenericRigidSymbols,
    MultiplicityFreeFusionRule, MultiplicityFreeFusionSymbols, MultiplicityFreeRigidSymbols,
    PackedProductCodec, PackedSectorLayout, ProductSector, ProductSectorCodec,
    ProductSectorCodecError, ProductSectorComponent, ProductSectorLayout, RuleIdentity, SectorId,
    SectorVec, Su2SectorLayout, SymbolShapeError, TensorKitProductCodec, U1FusionRule, U1Irrep,
    U1SectorLayout, Z2FusionRule, Z2Irrep,
};

mod core_rule_bridge;
pub(crate) use core_rule_bridge::lowered_multiplicity_free_sealed;
pub use core_rule_bridge::{
    LoweredFusionTreeBuildError, LoweredMultiplicityFreeAlgebra, MultiplicityFreePivotalSymbols,
};

include!("storage.rs");
include!("space.rs");
include!("sector.rs");
include!("fusion_space.rs");
mod su2_exact;
include!("fusion_rule.rs");
include!("su3.rs");
include!("fusion_tree.rs");
include!("block_structure.rs");
include!("tensor_map.rs");
include!("error.rs");

#[cfg(test)]
include!("tests.rs");
