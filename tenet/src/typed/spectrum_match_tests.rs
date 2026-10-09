//! #2081: `add_spectrum_into` pairs blocks with spectrum entries in O(G).
//!
//! The operation counter is the number of blocks that missed the cursor and
//! scanned the spectrum; it is zero when both sides run in sector order, so
//! the walk costs one comparison per block regardless of G.

use std::sync::Arc;

use super::fusion_tree::add_spectrum_counting_misses;
use super::{GradedSpace, SectorSpectrum, TensorMap};
use crate::runtime::Runtime;
use tenet_core::{U1FusionRule, U1Irrep};

fn misses_and_data(groups: usize) -> (usize, Vec<f64>) {
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let sectors: Vec<_> = (0..groups as i32)
        .map(|k| U1Irrep::new(31 * k - 15_000))
        .collect();
    let bond =
        GradedSpace::try_new(Arc::new(U1FusionRule), sectors.iter().map(|&s| (s, 1))).unwrap();
    let compact = TensorMap::<U1FusionRule, f64>::diagonal(
        &runtime,
        &bond,
        sectors
            .iter()
            .enumerate()
            .map(|(i, &sector)| SectorSpectrum {
                sector,
                values: vec![1.0 + i as f64],
            }),
    )
    .unwrap();
    let mut data = vec![0.0; groups];
    let misses = add_spectrum_counting_misses(
        compact.logical_space().space(),
        &mut data,
        compact.spectrum().unwrap(),
        2.0,
    )
    .unwrap();
    (misses, data)
}

#[test]
fn spectrum_walk_scans_nothing_and_scatters_every_sector() {
    for groups in [1000, 2000] {
        let (misses, data) = misses_and_data(groups);
        assert_eq!(misses, 0, "G = {groups}");
        let mut sorted = data.clone();
        sorted.sort_by(f64::total_cmp);
        let expected: Vec<f64> = (0..groups).map(|i| 2.0 * (1.0 + i as f64)).collect();
        assert_eq!(sorted, expected, "G = {groups}");
    }
}
