//! The `cuda` feature is additive: every typed method unified over storage
//! (#1756) has one definition per name, so its path form resolves with and
//! without `cuda`. CI's `cuda-check` compiles this file under `cuda`, where a
//! Host/CUDA twin pair would be `E0034`-ambiguous.
//!
//! Each #1756 leaf adds the names it unifies.

use std::sync::Arc;

use tenet::sector::{U1FusionRule, U1Irrep};
use tenet::typed::{GradedSpace, Runtime, TensorMap};

#[test]
fn unified_methods_resolve_in_path_form() {
    let runtime = Runtime::builder().build().unwrap();
    let v = GradedSpace::try_new(Arc::new(U1FusionRule), [(U1Irrep::new(0), 2)]).unwrap();
    let t: TensorMap<_, f64> = TensorMap::isomorphism(&runtime, [&v], [&v]).unwrap();

    // L1 reductions.
    assert_eq!(TensorMap::norm(&t, f64::INFINITY).unwrap(), 1.0);
    assert_eq!(TensorMap::inner(&t, &t).unwrap(), 2.0);
}
