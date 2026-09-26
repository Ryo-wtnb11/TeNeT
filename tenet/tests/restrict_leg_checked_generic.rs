//! `restrict_leg` / `embed_leg` on a checked Generic provider (#1285).
//!
//! SU(3) with the adjoint irrep gives fusion trees that differ only by their
//! outer-multiplicity vertex, which is the part of a block key a
//! degeneracy-only operation must leave alone. `otimes`/`id` are
//! multiplicity-free only, so the oracle here is a literal per-block gather
//! driven by the public fusion-tree keys and block geometry, which shares no
//! code with the restriction kernel.

#![cfg(feature = "racah-generated")]

use std::sync::Arc;

use tenet::prelude::Runtime;
use tenet::typed::{GradedSpace, LegSelection, SUNFusionRule, TensorMap};

#[test]
fn restrict_and_embed_preserve_su3_multiplicity_vertices() {
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let provider = Arc::new(SUNFusionRule::new(3).unwrap());
    let adjoint = vec![2i64, 2];
    let leg = GradedSpace::try_new_with_arc(Arc::clone(&provider), [(adjoint.clone(), 3)]).unwrap();
    let spectator =
        GradedSpace::try_new_with_arc(Arc::clone(&provider), [(adjoint.clone(), 2)]).unwrap();
    let source: TensorMap<_, f64> = TensorMap::from_block_fn(
        &runtime,
        [&leg, &spectator, &spectator],
        [],
        |trees, indices| {
            (100 * trees.codomain_vertices()[0].get() + indices.iter().sum::<usize>()) as f64
        },
    )
    .unwrap();
    assert!(
        (0..source.subblock_count()).any(|index| source
            .subblock_fusion_trees(index)
            .unwrap()
            .codomain_vertices()
            .iter()
            .any(|vertex| vertex.get() == 2)),
        "fixture must carry a Generic vertex key mu = 2"
    );

    let selection = LegSelection::try_new(&leg, [(adjoint.clone(), 1..3)]).unwrap();
    let restricted = source.restrict_leg(0, &selection).unwrap();
    assert_eq!(restricted.codomain()[0], *selection.subspace());
    assert_eq!(restricted.subblock_count(), source.subblock_count());

    // Oracle: a literal per-block gather driven by the public fusion-tree
    // keys and block geometry, sharing no code with the restriction kernel.
    let expected = literal_slice(&source, &restricted, 0, 1);
    assert_eq!(restricted.data(), expected);

    // Embedding back writes the same rectangle into a zero payload and leaves
    // every vertex key in place.
    let embedded = restricted.embed_leg(0, &selection).unwrap();
    assert_eq!(embedded.codomain()[0], leg);
    let expected = literal_scatter(&restricted, &embedded, 0, 1);
    assert_eq!(embedded.data(), expected);
    assert_eq!(
        embedded.restrict_leg(0, &selection).unwrap().data(),
        restricted.data()
    );
}

/// The payload `destination` must hold when each of its blocks is the
/// rectangle of `source`'s block with the same fusion trees, offset by `start`
/// on `axis`.
fn literal_slice(
    source: &TensorMap<SUNFusionRule, f64>,
    destination: &TensorMap<SUNFusionRule, f64>,
    axis: usize,
    start: usize,
) -> Vec<f64> {
    let mut payload = vec![f64::NAN; destination.data().len()];
    for index in 0..destination.subblock_count() {
        let trees = destination.subblock_fusion_trees(index).unwrap();
        let block = destination.subblock(index).unwrap();
        let matched = (0..source.subblock_count())
            .find(|&candidate| source.subblock_fusion_trees(candidate).unwrap() == trees)
            .expect("every destination tree pair survives from the source");
        let from = source.subblock(matched).unwrap();
        for_each_index(block.shape(), |position| {
            let to_offset: usize = block.offset()
                + position
                    .iter()
                    .zip(block.strides())
                    .map(|(&index, &stride)| index * stride)
                    .sum::<usize>();
            let from_offset: usize = from.offset()
                + position
                    .iter()
                    .enumerate()
                    .zip(from.strides())
                    .map(|((dimension, &index), &stride)| {
                        (index + if dimension == axis { start } else { 0 }) * stride
                    })
                    .sum::<usize>();
            payload[to_offset] = source.data()[from_offset];
        });
    }
    payload
}

/// The payload `destination` must hold when each `source` block is written
/// into its rectangle at `start` on `axis` and everything else stays zero.
fn literal_scatter(
    source: &TensorMap<SUNFusionRule, f64>,
    destination: &TensorMap<SUNFusionRule, f64>,
    axis: usize,
    start: usize,
) -> Vec<f64> {
    let mut payload = vec![0.0; destination.data().len()];
    for index in 0..source.subblock_count() {
        let trees = source.subblock_fusion_trees(index).unwrap();
        let block = source.subblock(index).unwrap();
        let matched = (0..destination.subblock_count())
            .find(|&candidate| destination.subblock_fusion_trees(candidate).unwrap() == trees)
            .expect("every source tree pair exists in the larger space");
        let into = destination.subblock(matched).unwrap();
        for_each_index(block.shape(), |position| {
            let from_offset: usize = block.offset()
                + position
                    .iter()
                    .zip(block.strides())
                    .map(|(&index, &stride)| index * stride)
                    .sum::<usize>();
            let to_offset: usize = into.offset()
                + position
                    .iter()
                    .enumerate()
                    .zip(into.strides())
                    .map(|((dimension, &index), &stride)| {
                        (index + if dimension == axis { start } else { 0 }) * stride
                    })
                    .sum::<usize>();
            payload[to_offset] = source.data()[from_offset];
        });
    }
    payload
}

fn for_each_index(shape: &[usize], mut visit: impl FnMut(&[usize])) {
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
