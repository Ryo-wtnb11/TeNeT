use super::*;
use crate::test_numerics::numerics;
use std::sync::Arc;

mod axis_selection;
mod block_specs_output;
mod compose;
mod coupled_layout;
mod facade;
mod fermion_twist;
mod granular_caches;
mod inactive_beta;
mod lazy_adjoint;
mod non_core_form;
mod oriented_storage;
mod parallel_replay;
mod prelowered;
mod prepared;

use compose::*;
use facade::*;
use non_core_form::*;
