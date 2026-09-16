//! Literal lazy-adjoint oracles shared by the multiplicity-free and checked
//! Generic integration tests.
//!
//! Everything here works on public snapshots (fusion trees, block geometry and
//! the flat payload) so the expected values never pass through the production
//! adjoint kernel under test.

#![allow(dead_code)]

use tenet::typed::BlockFusionTrees;

pub struct BlockGeometry {
    pub shape: Vec<usize>,
    pub strides: Vec<usize>,
    pub offset: usize,
}

pub struct Snapshot<S, D> {
    pub nout: usize,
    pub blocks: Vec<(BlockFusionTrees<S>, BlockGeometry)>,
    pub data: Vec<D>,
}

/// Snapshots a `TensorMap` through its public block API.
#[macro_export]
macro_rules! snapshot {
    ($tensor:expr) => {{
        let tensor = &$tensor;
        $crate::common::Snapshot {
            nout: tensor.codomain_rank(),
            blocks: (0..tensor.block_count())
                .map(|index| {
                    let block = tensor.block(index).unwrap();
                    (
                        tensor.block_fusion_trees(index).unwrap(),
                        $crate::common::BlockGeometry {
                            shape: block.shape().to_vec(),
                            strides: block.strides().to_vec(),
                            offset: block.offset(),
                        },
                    )
                })
                .collect(),
            data: tensor.data().to_vec(),
        }
    }};
}

/// Explicit odometer over a dense rectangle, innermost axis first.
pub fn for_each_index(shape: &[usize], mut visit: impl FnMut(&[usize])) {
    if shape.contains(&0) {
        return;
    }
    let mut index = vec![0usize; shape.len()];
    loop {
        visit(&index);
        let mut axis = 0;
        loop {
            if axis == shape.len() {
                return;
            }
            index[axis] += 1;
            if index[axis] < shape[axis] {
                break;
            }
            index[axis] = 0;
            axis += 1;
        }
    }
}

pub fn linear(geometry: &BlockGeometry, index: &[usize]) -> usize {
    geometry.offset
        + index
            .iter()
            .zip(&geometry.strides)
            .map(|(&i, &s)| i * s)
            .sum::<usize>()
}

/// `parent` is the dagger of `lazy`'s block: codomain and domain trees swap.
pub fn is_adjoint_pair<S: PartialEq>(
    lazy: &BlockFusionTrees<S>,
    parent: &BlockFusionTrees<S>,
) -> bool {
    lazy.coupled() == parent.coupled()
        && lazy.codomain_uncoupled() == parent.domain_uncoupled()
        && lazy.codomain_innerlines() == parent.domain_innerlines()
        && lazy.codomain_vertices() == parent.domain_vertices()
        && lazy.domain_uncoupled() == parent.codomain_uncoupled()
        && lazy.domain_innerlines() == parent.codomain_innerlines()
        && lazy.domain_vertices() == parent.codomain_vertices()
}

/// The pre-#1201 adjoint kernel as a literal formula:
/// `L[b, x] = conj(P[s(b), pi(x)])` where `s` swaps the codomain/domain trees
/// and `pi` swaps the dense axis halves (`x[..nin]` addresses the parent's
/// domain axes, `x[nin..]` its codomain axes).
pub fn literal_adjoint_payload<S: PartialEq, D: Copy + Default>(
    parent: &Snapshot<S, D>,
    lazy: &Snapshot<S, D>,
    conj: impl Fn(D) -> D,
) -> Vec<D> {
    // The lazy codomain is the parent domain: the split of `x` is at the
    // parent's `nin == lazy.nout`.
    let parent_nout = parent.nout;
    let split = lazy.nout;
    let mut result = vec![D::default(); lazy.data.len()];
    for (trees, geometry) in &lazy.blocks {
        let (_, source) = parent
            .blocks
            .iter()
            .find(|(candidate, _)| is_adjoint_pair(trees, candidate))
            .expect("every lazy block has exactly one swapped parent block");
        for_each_index(&geometry.shape, |x| {
            let source_position = source.offset
                + x[split..]
                    .iter()
                    .zip(&source.strides[..parent_nout])
                    .map(|(&i, &s)| i * s)
                    .sum::<usize>()
                + x[..split]
                    .iter()
                    .zip(&source.strides[parent_nout..])
                    .map(|(&i, &s)| i * s)
                    .sum::<usize>();
            result[linear(geometry, x)] = conj(parent.data[source_position]);
        });
    }
    result
}

/// Blockwise equality by fusion trees, independent of block order or layout.
pub fn assert_same_tensor<S: PartialEq + std::fmt::Debug, D: Copy + std::fmt::Debug>(
    actual: &Snapshot<S, D>,
    expected: &Snapshot<S, D>,
    close: impl Fn(D, D) -> bool,
) {
    assert_eq!(actual.nout, expected.nout);
    assert_eq!(actual.blocks.len(), expected.blocks.len());
    for (trees, geometry) in &actual.blocks {
        let (_, other) = expected
            .blocks
            .iter()
            .find(|(candidate, _)| candidate == trees)
            .unwrap_or_else(|| panic!("missing block {trees:?}"));
        assert_eq!(geometry.shape, other.shape, "block {trees:?}");
        for_each_index(&geometry.shape, |x| {
            let a = actual.data[linear(geometry, x)];
            let e = expected.data[linear(other, x)];
            assert!(close(a, e), "block {trees:?} at {x:?}: {a:?} != {e:?}");
        });
    }
}

/// TensorKit `tr`: `sum_c dim(c) * tr(b_c)` over the blocks whose codomain and
/// domain trees coincide, pairing codomain axis `i` with domain axis `nout+i`.
pub fn literal_weighted_trace<S: PartialEq, D: Copy>(
    tensor: &Snapshot<S, D>,
    dim: impl Fn(&S) -> f64,
    mut accumulate: impl FnMut(D, f64),
) {
    for (trees, geometry) in &tensor.blocks {
        if trees.codomain_uncoupled() != trees.domain_uncoupled()
            || trees.codomain_innerlines() != trees.domain_innerlines()
            || trees.codomain_vertices() != trees.domain_vertices()
        {
            continue;
        }
        let weight = dim(trees.coupled());
        for_each_index(&geometry.shape[..tensor.nout], |i| {
            let position = geometry.offset
                + i.iter()
                    .enumerate()
                    .map(|(axis, &c)| {
                        c * (geometry.strides[axis] + geometry.strides[tensor.nout + axis])
                    })
                    .sum::<usize>();
            accumulate(tensor.data[position], weight);
        });
    }
}
