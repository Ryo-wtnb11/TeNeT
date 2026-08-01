use std::hint::black_box;
use std::sync::Arc;
use std::time::Instant;

use tenet::core::{U1FusionRule, U1Irrep};
use tenet::typed::{GradedSpace, Runtime, TensorMap};
use tenet_network::{plan_cache_stats, tensor};

fn main() {
    let runtime = Runtime::builder().build().expect("runtime");
    let space = GradedSpace::try_new(
        Arc::new(U1FusionRule),
        [
            (U1Irrep::new(-1), 8),
            (U1Irrep::new(0), 16),
            (U1Irrep::new(1), 8),
        ],
        false,
    )
    .expect("space");
    let a =
        TensorMap::<_, f64>::rand_with_seed(&runtime, [&space, &space], [&space, &space], 12401)
            .expect("lhs");
    let b =
        TensorMap::<_, f64>::rand_with_seed(&runtime, [&space, &space], [&space, &space], 12402)
            .expect("rhs");

    let cold_start = Instant::now();
    black_box(tensor!([i, j; m, n] = a[i, j; k, l] * b[k, l; m, n]).expect("cold"));
    let cold = cold_start.elapsed();

    let iterations = 20;
    let warm_start = Instant::now();
    for _ in 0..iterations {
        black_box(tensor!([i, j; m, n] = a[i, j; k, l] * b[k, l; m, n]).expect("warm"));
    }
    let warm = warm_start.elapsed() / iterations;
    let stats = plan_cache_stats(&runtime);
    println!(
        "cold={cold:?} warm_mean={warm:?} hits={} topology_materializations={} \
         workspaces_created={} workspace_reuses={} workspace_slot_grows={}",
        stats.hits,
        stats.topology_materializations,
        stats.workspaces_created,
        stats.workspace_reuses,
        stats.workspace_slot_grows,
    );
}
