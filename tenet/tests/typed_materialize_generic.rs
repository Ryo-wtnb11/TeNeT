//! `materialize` (#1514) on checked Generic providers with fusion
//! multiplicity. Independent oracle (reviewer, #1522): every adjoint block is
//! located by swapping the parent's codomain and domain trees (uncoupled
//! sectors, inner lines and vertex labels), and every entry is the conjugate
//! of the parent entry at the swapped axes, addressed through the parent's own
//! strides.

#![cfg(feature = "racah-generated")]

use std::sync::Arc;

use tenet::prelude::{Complex64, Runtime};
use tenet::typed::{GradedSpace, NetworkReuseClass, SUNFusionRule, TensorMap};

fn check(n: usize, label: Vec<i64>) {
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let provider = Arc::new(SUNFusionRule::new(n).unwrap());
    let a = GradedSpace::try_new_with_arc(Arc::clone(&provider), [(label.clone(), 2)]).unwrap();
    let b = GradedSpace::try_new_with_arc(Arc::clone(&provider), [(label, 3)]).unwrap();
    let b_dual = b.try_dual().unwrap();
    let tensor: TensorMap<_, Complex64> =
        TensorMap::from_block_fn(&runtime, [&a, &b], [&b_dual], |trees, index| {
            let vertex = trees
                .codomain_vertices()
                .iter()
                .map(|m| m.get())
                .sum::<usize>() as f64;
            let spread = index
                .iter()
                .enumerate()
                .map(|(axis, i)| (i * (axis + 1) * 3) as f64)
                .sum::<f64>();
            Complex64::new(1.0 + vertex * 10.0 + spread, 0.5 + index[0] as f64 - vertex)
        })
        .unwrap();
    let lazy = tensor.adjoint().unwrap();
    assert!(lazy.network_reuse_class(false) == NetworkReuseClass::LazyAdjoint);
    let owned = lazy.materialize().unwrap();
    assert!(owned.network_reuse_class(false) == NetworkReuseClass::OwnedDense);
    assert_eq!(owned.block_count(), tensor.block_count());
    let (parent_nout, adjoint_nout) = (tensor.codomain_rank(), owned.codomain_rank());
    let mut saw_multiplicity = false;
    for adjoint_index in 0..owned.block_count() {
        let adjoint_trees = owned.block_fusion_trees(adjoint_index).unwrap();
        if adjoint_trees.domain_vertices().iter().any(|v| v.get() == 2) {
            saw_multiplicity = true;
        }
        let parent_index = (0..tensor.block_count())
            .find(|&i| {
                let parent_trees = tensor.block_fusion_trees(i).unwrap();
                parent_trees.codomain_uncoupled() == adjoint_trees.domain_uncoupled()
                    && parent_trees.domain_uncoupled() == adjoint_trees.codomain_uncoupled()
                    && parent_trees.codomain_innerlines() == adjoint_trees.domain_innerlines()
                    && parent_trees.domain_innerlines() == adjoint_trees.codomain_innerlines()
                    && parent_trees.codomain_vertices() == adjoint_trees.domain_vertices()
                    && parent_trees.domain_vertices() == adjoint_trees.codomain_vertices()
                    && parent_trees.coupled() == adjoint_trees.coupled()
            })
            .unwrap();
        let adjoint_block = owned.block(adjoint_index).unwrap();
        let parent_block = tensor.block(parent_index).unwrap();
        let shape = adjoint_block.shape().to_vec();
        let rank = shape.len();
        for linear in 0..shape.iter().product::<usize>() {
            let mut rest = linear;
            let mut index = vec![0; rank];
            for axis in 0..rank {
                index[axis] = rest % shape[axis];
                rest /= shape[axis];
            }
            let adjoint_offset = adjoint_block.offset()
                + (0..rank)
                    .map(|axis| index[axis] * adjoint_block.strides()[axis])
                    .sum::<usize>();
            let parent_offset = parent_block.offset()
                + (0..rank)
                    .map(|axis| {
                        let parent_axis = if axis < adjoint_nout {
                            parent_nout + axis
                        } else {
                            axis - adjoint_nout
                        };
                        index[axis] * parent_block.strides()[parent_axis]
                    })
                    .sum::<usize>();
            assert_eq!(
                owned.data()[adjoint_offset],
                tensor.data()[parent_offset].conj()
            );
        }
    }
    assert!(
        saw_multiplicity,
        "fixture must carry a multiplicity-2 vertex"
    );
}

#[test]
fn materialize_matches_swapped_trees_with_multiplicity() {
    check(3, vec![2, 2]);
    check(4, vec![2, 0, 2]);
}
