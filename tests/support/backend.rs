//! The one test predicate for "the unset default CPU provider is faer".
//!
//! The authority is the dense layer's compiled default
//! (`cpu-blas-core` selects BLAS, otherwise faer), not a hand-written list of
//! `blas-*` features. Gates that only hold on faer (allocation budgets, bit
//! pins) branch on this at run time.

use tenet::expert::CpuBackendKind;

pub fn faer_is_default_backend() -> bool {
    CpuBackendKind::default_compiled() == CpuBackendKind::Faer
}
