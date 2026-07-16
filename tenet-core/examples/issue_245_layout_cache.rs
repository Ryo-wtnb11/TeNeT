use std::hint::black_box;
use std::time::Instant;

use tenet_core::{FusionProductSpace, FusionTreeHomSpace, SectorLeg, U1FusionRule, U1Irrep};

fn main() {
    println!("sector_count,ns_per_warm_tree_keys,ns_per_warm_coupled_layout");
    for sector_count in [1, 8, 64] {
        let sectors = (0..sector_count)
            .map(|charge| (U1Irrep::new(charge), 1))
            .collect::<Vec<_>>();
        let homspace = FusionTreeHomSpace::new(
            FusionProductSpace::new([SectorLeg::new(sectors.clone(), false)]),
            FusionProductSpace::new([SectorLeg::new(sectors, false)]),
        );
        let rule = U1FusionRule;
        let shapes = vec![vec![1, 1]; sector_count as usize];
        black_box(homspace.fusion_tree_keys(&rule));
        black_box(
            homspace
                .coupled_subblock_structure(&rule, 1, shapes.iter().cloned())
                .unwrap(),
        );
        let iterations = 200_000 / (sector_count as usize).max(1);
        let started = Instant::now();
        for _ in 0..iterations {
            black_box(homspace.fusion_tree_keys(&rule));
        }
        let tree_keys_ns = started.elapsed().as_nanos() / iterations as u128;
        let started = Instant::now();
        for _ in 0..iterations {
            black_box(
                homspace
                    .coupled_subblock_structure(&rule, 1, shapes.iter().cloned())
                    .unwrap(),
            );
        }
        println!(
            "{sector_count},{tree_keys_ns},{}",
            started.elapsed().as_nanos() / iterations as u128
        );
    }
}
