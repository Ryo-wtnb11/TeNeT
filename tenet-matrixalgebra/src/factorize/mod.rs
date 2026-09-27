use std::collections::BTreeMap;

use rustc_hash::{FxHashMap, FxHashSet};
use std::fmt;
use std::sync::Arc;

#[cfg(test)]
use std::cell::{Cell, RefCell};

use num_complex::Complex64;
use num_traits::{Float, Zero};
use tenet_core::{
    BlockKey, BlockRef, BlockStructure, CheckedGenericFusion, CheckedGenericRigidSymbols,
    CheckedGenericStructureError, CoreError, CoupledSectorRegion, CoupledTreeExtent,
    FusionProductSpace, FusionRule, FusionTensorMapSpace, FusionTreeHomSpace, FusionTreeKey,
    FusionTreePairKey, InfallibleGeneric, MultiplicityFreeRigidSymbols, SectorId, SectorLeg,
    SectorStructure, TensorMap, TensorMapSpace,
};
use tenet_dense::{
    DenseBackend, DenseDotConfig, DenseError, DenseExecutor, DenseFactorization, DenseOwned,
    DenseTensor, DenseView, DenseViewMut,
};

pub use tenet_tensors::BoundDynamicTensorRef;
use tenet_tensors::{
    BoundDynamicFusionMapSpace, DenseBlockScalar, DenseRecouplingScalar, DynamicFusionMapSpace,
    PreparedCheckedGenericDynamicSpace, ValidatedDynamicFusionLayout,
};

use crate::results::{LeftPolar, Lq, Qr, RightPolar, Svd};
use crate::truncation::{select_truncation, Truncation, WeightedSpectrum};
use tenet_tensors::OperationError;

mod eig;
mod inverse;
mod null_space;
mod polar;
mod qr_lq;
mod region;
mod svd;

#[cfg(test)]
mod numerical_null_tests;
#[cfg(test)]
mod hermitian_scale_tests;
#[cfg(test)]
mod sector_matricization_tests;

// Blanket re-exports so every existing `crate::factorize::<name>` path
// (used by `compose.rs`, `matrix_functions.rs`, and `lib.rs`'s own
// `pub use factorize::{...}`) keeps resolving unchanged after the move.
// Each child's own `pub`/`pub(crate)` items pass straight through; each
// child's `pub(super)` items (bumped from fully-private during the move)
// become reachable here and, by re-export, throughout the crate.
pub(crate) use eig::*;
pub(crate) use inverse::*;
pub(crate) use null_space::*;
pub(crate) use polar::*;
pub(crate) use qr_lq::*;
pub(crate) use region::*;
pub(crate) use svd::*;
