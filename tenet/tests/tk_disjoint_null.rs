//! Disjoint U(1) zero-map null-space completion against TensorKit `f87ca7f`
//! (Project 0.17.1, Julia 1.11.6), oracle section 6.

use std::sync::Arc;

use tenet::core::{U1FusionRule, U1Irrep};
use tenet::typed::{GradedSpace, Runtime, TensorMap};

#[test]
fn direct_and_lazy_adjoint_disjoint_null_spaces_match_tensorkit() {
    let runtime = Runtime::builder().build().unwrap();
    let rule = Arc::new(U1FusionRule);
    let codomain = GradedSpace::try_new_shared(Arc::clone(&rule), [(U1Irrep::new(0), 2)]).unwrap();
    let domain = GradedSpace::try_new_shared(rule, [(U1Irrep::new(1), 3)]).unwrap();
    let source: TensorMap<_, f64> = TensorMap::zeros(&runtime, [&codomain], [&domain]).unwrap();

    assert!(source.data().is_empty());
    for (tensor, left_dim, right_dim) in [(source.clone(), 2, 3), (source.adjoint().unwrap(), 3, 2)]
    {
        let left = tensor.left_null().unwrap();
        let right = tensor.right_null().unwrap();

        assert_eq!(left.data().len(), left_dim * left_dim);
        assert_eq!(right.data().len(), right_dim * right_dim);
        assert!((left.norm().unwrap() - (left_dim as f64).sqrt()).abs() <= 1e-12);
        assert!((right.norm().unwrap() - (right_dim as f64).sqrt()).abs() <= 1e-12);
        assert!(left.is_isometric(1e-12).unwrap());
        assert!(right.adjoint().unwrap().is_isometric(1e-12).unwrap());
        assert!(left
            .adjoint()
            .unwrap()
            .compose(&tensor)
            .unwrap()
            .data()
            .is_empty());
        assert!(tensor
            .compose(&right.adjoint().unwrap())
            .unwrap()
            .data()
            .is_empty());
    }
}
