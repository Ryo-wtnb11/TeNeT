use super::*;
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Arc,
};

use tenet_core::{
    generic_braid_tree_pair, generic_permute_tree_pair, generic_transpose_tree_pair,
    CheckedGenericFusion, CheckedGenericRigidSymbols, GenericFArray, GenericFusionSymbols,
    GenericRMatrix, GenericRigidSymbols,
};

/// Test view of the logical coefficient payload (an explicit copy).
trait GatheredCoefficients<T> {
    fn gathered_coefficients(&self) -> Vec<T>;
}

impl<T: Copy> GatheredCoefficients<T> for tenet_operations::TreeTransformStructure<T> {
    fn gathered_coefficients(&self) -> Vec<T> {
        let mut coefficients = Vec::new();
        self.gather_recoupling_coefficients_into(&mut coefficients);
        coefficients
    }
}

mod admission;
mod checked_generic;
mod context_cache;
mod generic;
mod generic_facade;
mod grouped_storage;
mod plan_builder;
mod tree_pair;
mod unique;

use admission::*;
use generic::*;
use plan_builder::*;
