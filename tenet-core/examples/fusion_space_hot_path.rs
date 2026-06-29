use std::hint::black_box;
use std::time::{Duration, Instant};

use tenet_core::{
    BraidingStyleKind, FusionProductSpace, FusionRule, FusionStyleKind, FusionTensorMapSpace,
    FusionTreeHomSpace, MultiplicityFreeFusionRule, SectorId, SectorLeg, TensorMap, TensorMapSpace,
};

const CHARGE_COUNTS: &[usize] = &[1, 8, 64, 512, 4096];

fn main() {
    println!("# U(1) pointed UniqueFusion finite-charge hot path");
    println!("charge_count,block_count,build_ns,tree_lookup_ns,sector_lookup_ns");
    for &charge_count in CHARGE_COUNTS {
        run_u1_case(charge_count);
    }
}

#[derive(Clone, Copy, Debug)]
struct U1PointedRule;

impl U1PointedRule {
    const OFFSET: i32 = 1_000_000;

    fn encode(charge: i32) -> SectorId {
        let shifted = charge
            .checked_add(Self::OFFSET)
            .expect("benchmark U(1) charge should stay in encoded range");
        SectorId::new(usize::try_from(shifted).expect("encoded charge should be non-negative"))
    }

    fn decode(sector: SectorId) -> i32 {
        i32::try_from(sector.id()).expect("benchmark sector id should fit i32") - Self::OFFSET
    }
}

impl FusionRule for U1PointedRule {
    fn fusion_style(&self) -> FusionStyleKind {
        FusionStyleKind::Unique
    }

    fn braiding_style(&self) -> BraidingStyleKind {
        BraidingStyleKind::Bosonic
    }

    fn vacuum(&self) -> SectorId {
        Self::encode(0)
    }

    fn dual(&self, sector: SectorId) -> SectorId {
        Self::encode(-Self::decode(sector))
    }

    fn fusion_channels(&self, left: SectorId, right: SectorId) -> Vec<SectorId> {
        vec![Self::encode(Self::decode(left) + Self::decode(right))]
    }
}

impl MultiplicityFreeFusionRule for U1PointedRule {}

fn run_u1_case(charge_count: usize) {
    let rule = U1PointedRule;
    let charges = charges(charge_count);
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
            let sector = U1PointedRule::encode(*charge);
            let external_domain = U1PointedRule::encode(-*charge);
            let block = tensor
                .subblock_by_sectors(&rule, &[sector, external_domain])
                .unwrap();
            checksum ^= block.offset();
        }
        black_box(checksum);
    });

    println!(
        "{},{},{},{},{}",
        charge_count,
        block_count,
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

fn fusion_space(rule: &U1PointedRule, charges: &[i32]) -> FusionTensorMapSpace<1, 1> {
    let sectors = charges
        .iter()
        .map(|&charge| U1PointedRule::encode(charge))
        .collect::<Vec<_>>();
    let dense = TensorMapSpace::<1, 1>::from_dims([charges.len()], [charges.len()]).unwrap();
    let hom = FusionTreeHomSpace::new(
        FusionProductSpace::new([SectorLeg::new(sectors.clone(), false)]),
        FusionProductSpace::new([SectorLeg::new(sectors, false)]),
    );
    FusionTensorMapSpace::from_degeneracy_shapes(
        dense,
        hom,
        rule,
        (0..charges.len()).map(|_| vec![1, 1]),
    )
    .unwrap()
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
