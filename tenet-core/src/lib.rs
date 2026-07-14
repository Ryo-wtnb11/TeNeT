#![forbid(unsafe_code)]

//! Core TensorMap-facing data structures for TeNeT.
//!
//! This crate owns TeNeT's public/core tensor view vocabulary. Lower-level
//! crates may lower these views to concrete strided kernels, but external
//! strided/backend types should not be required by TensorMap users.

use core::fmt;
use core::marker::PhantomData;
use core::ops::{Add, Mul};
use std::collections::hash_map::Entry;
use std::hash::Hash;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, OnceLock, RwLock, Weak};

use num_complex::Complex64;
use rustc_hash::FxHashMap;
use smallvec::{smallvec, SmallVec};

include!("storage.rs");
include!("space.rs");
include!("sector.rs");
include!("fusion_space.rs");
include!("fusion_rule.rs");
include!("su3.rs");
include!("fusion_tree.rs");
include!("block_structure.rs");
include!("tensor_map.rs");
include!("error.rs");

#[cfg(test)]
include!("tests.rs");
