use std::hint::black_box;
use std::time::{Duration, Instant};

use tenet_core::{
    FusionProductSpace, FusionTensorMapSpace, FusionTreeHomSpace, SectorLeg, TensorMap,
    TensorMapSpace, U1FusionRule, U1Irrep,
};

const CHARGE_COUNTS: &[usize] = &[1, 8, 64, 512, 4096];

fn main() {
    println!("# U(1) pointed UniqueFusion finite-charge hot path");
    println!(
        "charge_count,block_count,tree_keys_first_ns,tree_keys_repeat_ns,build_ns,tree_lookup_ns,sector_lookup_ns"
    );
    for &charge_count in CHARGE_COUNTS {
        run_u1_case(charge_count);
    }
}

fn run_u1_case(charge_count: usize) {
    let rule = U1FusionRule;
    let charges = charges(charge_count);
    let hom = homspace(&charges);
    let tree_keys_first_elapsed = elapsed_once(|| {
        let keys = hom.fusion_tree_keys(&rule);
        black_box(keys.len());
    });
    let tree_keys_repeat_iters = iterations(charge_count, 100_000);
    let tree_keys_repeat_elapsed = elapsed_per_iter(tree_keys_repeat_iters, || {
        let keys = hom.fusion_tree_keys(&rule);
        black_box(keys.len());
    });

    let compile_iters = iterations(charge_count, 200_000);
    let build_elapsed = elapsed_per_iter(compile_iters, || {
        let fusion_space = fusion_space(&rule, &charges);
        black_box(fusion_space.subblock_structure().block_count());
    });

    let fusion_space = fusion_space(&rule, &charges);
    let block_count = fusion_space.subblock_structure().block_count();
    let keys = fusion_space.homspace().fusion_tree_keys(&rule);
    let data = vec![0.0_f64; fusion_space.required_len().unwrap()];
    let tensor = TensorMap::<f64, 1, 1>::from_vec_with_fusion_space(data, fusion_space).unwrap();

    let lookup_iters = iterations(charge_count, 100_000);
    let tree_elapsed = elapsed_per_op(lookup_iters, charge_count, || {
        let mut checksum = 0usize;
        for key in &keys {
            let block = tensor.subblock_by_tree(key).unwrap();
            checksum ^= block.offset();
        }
        black_box(checksum);
    });
    let sector_elapsed = elapsed_per_op(lookup_iters, charge_count, || {
        let mut checksum = 0usize;
        for charge in &charges {
            let sector = U1Irrep::new(*charge).sector_id();
            let external_domain = U1Irrep::new(-*charge).sector_id();
            let block = tensor
                .subblock_by_sectors(&rule, &[sector, external_domain])
                .unwrap();
            checksum ^= block.offset();
        }
        black_box(checksum);
    });

    println!(
        "{},{},{},{},{},{},{}",
        charge_count,
        block_count,
        tree_keys_first_elapsed.as_nanos(),
        tree_keys_repeat_elapsed.as_nanos(),
        build_elapsed.as_nanos(),
        tree_elapsed.as_nanos(),
        sector_elapsed.as_nanos()
    );
}

fn charges(charge_count: usize) -> Vec<i32> {
    assert!(charge_count <= i32::MAX as usize);
    (0..charge_count)
        .map(|charge| i32::try_from(charge).unwrap())
        .collect()
}

fn fusion_space(rule: &U1FusionRule, charges: &[i32]) -> FusionTensorMapSpace<1, 1> {
    let dense = TensorMapSpace::<1, 1>::from_dims([charges.len()], [charges.len()]).unwrap();
    FusionTensorMapSpace::from_degeneracy_shapes(
        dense,
        homspace(charges),
        rule,
        (0..charges.len()).map(|_| vec![1, 1]),
    )
    .unwrap()
}

fn homspace(charges: &[i32]) -> FusionTreeHomSpace {
    let sectors = charges
        .iter()
        .map(|&charge| (U1Irrep::new(charge), 1))
        .collect::<Vec<_>>();
    FusionTreeHomSpace::new(
        FusionProductSpace::new([SectorLeg::new(sectors.clone(), false)]),
        FusionProductSpace::new([SectorLeg::new(sectors, false)]),
    )
}

fn iterations(item_count: usize, budget: usize) -> usize {
    (budget / item_count.max(1)).clamp(10, 20_000)
}

fn elapsed_per_iter(iters: usize, mut f: impl FnMut()) -> Duration {
    for _ in 0..iters.min(100) {
        f();
    }
    let start = Instant::now();
    for _ in 0..iters {
        f();
    }
    let elapsed = start.elapsed();
    Duration::from_nanos((elapsed.as_nanos() / iters.max(1) as u128) as u64)
}

fn elapsed_once(mut f: impl FnMut()) -> Duration {
    let start = Instant::now();
    f();
    start.elapsed()
}

fn elapsed_per_op(iters: usize, op_count: usize, mut f: impl FnMut()) -> Duration {
    for _ in 0..iters.min(100) {
        f();
    }
    let start = Instant::now();
    for _ in 0..iters {
        f();
    }
    let elapsed = start.elapsed();
    let total_ops = iters.max(1) * op_count.max(1);
    Duration::from_nanos((elapsed.as_nanos() / total_ops as u128) as u64)
}
