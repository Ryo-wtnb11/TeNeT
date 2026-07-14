//! Warm-cache contraction microbenchmark across symmetries.
//!
//! Mirrors `benchmarks/tensorkit_microbench.jl`: rank-4 tensors `A, B` in
//! `V ⊗ V ← V ⊗ V` with a uniform degeneracy per sector, three workloads:
//!
//! - `compose`:  C[a b; g h] = A[a b; c d] * B[c d; g h]  (core route)
//! - `swap`:     C[a b; g h] = A[a b; c d] * B[d c; g h]  (source transforms)
//! - `swap+out`: C[b a; g h] = A[a b; c d] * B[d c; g h]  (plus output transform)
//!
//! Usage: `cargo run --release --example microbench_fusion [degeneracy] [min_ms]`

use std::time::Instant;

use tenet_core::{
    FermionParityFusionRule, FusionProductSpace, FusionTensorMapSpace, FusionTreeHomSpace,
    MultiplicityFreeRigidSymbols, ProductFusionRule, ProductSectorCodec, SU2Irrep, SectorId,
    SectorLeg, TensorKitProductCodec, TensorMap, TensorMapSpace, U1FusionRule, U1Irrep,
};
use tenet_tensors::{
    OutputAxisOrder, TensorContractFusionExecutionContext, TensorContractSpec,
    TreeTransformRuleCacheKey,
};

struct Workload {
    name: &'static str,
    lhs_axes: [usize; 2],
    rhs_axes: [usize; 2],
    output_axes: [usize; 4],
}

const WORKLOADS: [Workload; 3] = [
    Workload {
        name: "compose",
        lhs_axes: [2, 3],
        rhs_axes: [0, 1],
        output_axes: [0, 1, 2, 3],
    },
    Workload {
        name: "swap",
        lhs_axes: [3, 2],
        rhs_axes: [0, 1],
        output_axes: [0, 1, 2, 3],
    },
    Workload {
        name: "swap+out",
        lhs_axes: [3, 2],
        rhs_axes: [0, 1],
        output_axes: [1, 0, 2, 3],
    },
];

fn main() {
    let mut args = std::env::args().skip(1);
    let degeneracy: usize = args
        .next()
        .map(|value| value.parse().expect("degeneracy must be an integer"))
        .unwrap_or(8);
    let min_ms: u64 = args
        .next()
        .map(|value| value.parse().expect("min_ms must be an integer"))
        .unwrap_or(300);

    println!("# tenet microbench: degeneracy={degeneracy} min_ms={min_ms}");
    println!("# symmetry\tworkload\tus_per_iter\titers\tchecksum");

    run_case("U1", &U1FusionRule, &u1_sectors(), degeneracy, min_ms);
    run_case(
        "fZ2",
        &FermionParityFusionRule,
        &[SectorId::new(0), SectorId::new(1)],
        degeneracy,
        min_ms,
    );
    run_case(
        "SU2",
        &tenet_core::SU2FusionRule,
        &[
            SU2Irrep::from_twice_spin(0).sector_id(),
            SU2Irrep::from_twice_spin(1).sector_id(),
            SU2Irrep::from_twice_spin(2).sector_id(),
        ],
        degeneracy,
        min_ms,
    );
    run_case(
        "U1xfZ2",
        &ProductFusionRule::<U1FusionRule, FermionParityFusionRule>::new(
            U1FusionRule,
            FermionParityFusionRule,
        ),
        &u1_fparity_sectors(),
        degeneracy,
        min_ms,
    );
}

fn u1_sectors() -> Vec<SectorId> {
    [-1, 0, 1]
        .into_iter()
        .map(|charge| U1Irrep::new(charge).sector_id())
        .collect()
}

fn u1_fparity_sectors() -> Vec<SectorId> {
    [(-1, 1), (0, 0), (1, 1)]
        .into_iter()
        .map(|(charge, parity)| {
            TensorKitProductCodec::try_encode(
                U1Irrep::new(charge).sector_id(),
                SectorId::new(parity),
            )
            .expect("product sector encoding")
        })
        .collect()
}

fn run_case<R>(name: &str, rule: &R, sectors: &[SectorId], degeneracy: usize, min_ms: u64)
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64> + TreeTransformRuleCacheKey,
    R::Key: Clone + Eq + std::hash::Hash,
{
    let leg = || SectorLeg::new(sectors.iter().map(|&sector| (sector, degeneracy)), false);
    let leg_dim = sectors.len() * degeneracy;
    let homspace = || {
        FusionTreeHomSpace::new(
            FusionProductSpace::new([leg(), leg()]),
            FusionProductSpace::new([leg(), leg()]),
        )
    };
    let space = |hom: FusionTreeHomSpace| {
        let key_count = hom.fusion_tree_keys(rule).len();
        let dense =
            TensorMapSpace::<2, 2>::from_dims([leg_dim, leg_dim], [leg_dim, leg_dim]).unwrap();
        let shapes = vec![vec![degeneracy; 4]; key_count];
        FusionTensorMapSpace::from_degeneracy_shapes(dense, hom, rule, shapes).unwrap()
    };

    let lhs_space = space(homspace());
    let rhs_space = space(homspace());
    let lhs_len = lhs_space.required_len().unwrap();
    let rhs_len = rhs_space.required_len().unwrap();
    let lhs = TensorMap::<f64, 2, 2>::from_vec_with_fusion_space(
        (0..lhs_len)
            .map(|index| (index % 17) as f64 * 0.25 - 2.0)
            .collect(),
        lhs_space,
    )
    .unwrap();
    let rhs = TensorMap::<f64, 2, 2>::from_vec_with_fusion_space(
        (0..rhs_len)
            .map(|index| (index % 13) as f64 * 0.5 - 3.0)
            .collect(),
        rhs_space,
    )
    .unwrap();

    for workload in &WORKLOADS {
        let dst_hom = FusionTreeHomSpace::tensorcontract_homspace(
            rule,
            lhs.fusion_space().unwrap().homspace(),
            rhs.fusion_space().unwrap().homspace(),
            &workload.lhs_axes,
            &workload.rhs_axes,
            &workload.output_axes,
            2,
        )
        .unwrap();
        let dst_space = space(dst_hom);
        let dst_len = dst_space.required_len().unwrap();
        let mut dst =
            TensorMap::<f64, 2, 2>::from_vec_with_fusion_space(vec![0.0; dst_len], dst_space)
                .unwrap();

        let mut context = TensorContractFusionExecutionContext::<f64, R::Key>::default();
        // Bench-only override of the transform-replay worker count; the size
        // gate is dropped so small-degeneracy workloads exercise the parallel
        // path too (production keeps the gate).
        if let Ok(threads) = std::env::var("MICROBENCH_RECOUPLING_THREADS") {
            let threads: usize = threads.parse().expect("MICROBENCH_RECOUPLING_THREADS");
            let backend = context.tree_context_mut().backend_mut();
            backend.set_recoupling_threads(threads);
            backend.set_transform_parallel_min_len(0);
        }
        let axes = || {
            TensorContractSpec::new(
                &workload.lhs_axes,
                &workload.rhs_axes,
                OutputAxisOrder::from_axes(&workload.output_axes),
            )
        };

        // First call compiles every structure cache; report it as cold time.
        let cold_start = Instant::now();
        context
            .tensorcontract_fusion_into(rule, &mut dst, &lhs, &rhs, axes(), 1.0, 0.0)
            .unwrap();
        let cold_us = cold_start.elapsed().as_secs_f64() * 1e6;
        for _ in 0..2 {
            context
                .tensorcontract_fusion_into(rule, &mut dst, &lhs, &rhs, axes(), 1.0, 0.0)
                .unwrap();
        }

        let min_duration = std::time::Duration::from_millis(min_ms);
        let prepared = std::env::var("MICROBENCH_PREPARED").is_ok().then(|| {
            context
                .prepare_tensorcontract_fusion(rule, &dst, &lhs, &rhs, axes())
                .unwrap()
        });
        let mut iters = 0u64;
        let start = Instant::now();
        while start.elapsed() < min_duration {
            match &prepared {
                Some(plan) => context
                    .execute_prepared_tensorcontract_fusion(
                        plan, rule, &mut dst, &lhs, &rhs, 1.0, 0.0,
                    )
                    .unwrap(),
                None => context
                    .tensorcontract_fusion_into(rule, &mut dst, &lhs, &rhs, axes(), 1.0, 0.0)
                    .unwrap(),
            }
            iters += 1;
        }
        let elapsed = start.elapsed();
        let us_per_iter = elapsed.as_secs_f64() * 1e6 / iters as f64;
        let checksum: f64 = dst.data().iter().copied().sum();
        println!(
            "{name}\t{workload}\t{us_per_iter:.2}\t{iters}\t{checksum:.6}\tcold={cold_us:.0}",
            workload = workload.name,
        );

        if std::env::var("MICROBENCH_PROFILE").is_ok() {
            let mut profile = tenet_tensors::TensorContractFusionProfile::default();
            let profile_iters = iters.clamp(1, 200);
            for _ in 0..profile_iters {
                let mut step = tenet_tensors::TensorContractFusionProfile::default();
                context
                    .tensorcontract_fusion_into_profiled(
                        rule,
                        &mut dst,
                        &lhs,
                        &rhs,
                        axes(),
                        1.0,
                        0.0,
                        &mut step,
                    )
                    .unwrap();
                profile.accumulate(&step);
            }
            let us =
                |duration: std::time::Duration| duration.as_secs_f64() * 1e6 / profile_iters as f64;
            println!(
                "  route={route:?} total={total:.1} plan_lookups={lookups:.1} \
                 src_transforms={src:.1} scratch_prepare={scratch:.1} \
                 pack={pack:.1} matmul={matmul:.1} scatter={scatter:.1} \
                 scale+validate={scale:.1} out_transform={out:.1} groups={groups}",
                route = profile.route,
                total = us(profile.total),
                lookups = us(profile.prepared_plan
                    + profile.source_space_lookup
                    + profile.core_dst_space_lookup
                    + profile.fusion_block_plan_lookup
                    + profile.core_route_check
                    + profile.typed_space_setup),
                src = us(profile.lhs_transform + profile.rhs_transform),
                scratch = us(profile.lhs_scratch_prepare
                    + profile.rhs_scratch_prepare
                    + profile.dst_scratch_prepare
                    + profile.core_workspace_prepare),
                pack = us(profile.core_pack_lhs + profile.core_pack_rhs),
                matmul = us(profile.core_matmul),
                scatter = us(profile.core_scatter),
                scale = us(profile.core_scale + profile.core_validate),
                out = us(profile.output_transform),
                groups = profile.core_contract_groups / profile_iters as usize,
            );
            let tree = &profile.tree_replay;
            println!(
                "  tree_replay: total={total:.1} inactive_scale={inactive_scale:.1} \
                 single={single:.1}({sb}) \
                 pack={pack:.1}({pc}) recoupling={rec:.1} matmul={mm:.1} \
                 scatter={scatter:.1}({sc}) prepare={prep:.1} multi_blocks={mb}",
                total = us(tree.total),
                inactive_scale = us(tree.inactive_scale),
                single = us(tree.single_total),
                sb = tree.single_blocks / profile_iters as usize,
                pack = us(tree.multi_pack),
                pc = tree.packed_columns / profile_iters as usize,
                rec = us(tree.multi_scalar_recoupling),
                mm = us(tree.multi_matmul_total),
                scatter = us(tree.multi_scatter),
                sc = tree.scattered_columns / profile_iters as usize,
                prep = us(tree.multi_workspace_prepare),
                mb = tree.multi_blocks / profile_iters as usize,
            );
        }
    }
}
