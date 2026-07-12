use std::hint::black_box;
use std::time::Instant;

use tenet::prelude::{Dtype, Runtime, Space, Tensor};
use tenet_network::{plan_cache_stats, tensor};

fn main() {
    let runtime = Runtime::builder().build().expect("runtime");
    let space = Space::u1([(-1, 8), (0, 16), (1, 8)]);
    let a = Tensor::rand_with_seed(
        &runtime,
        Dtype::F64,
        [&space, &space],
        [&space, &space],
        12401,
    )
    .expect("lhs");
    let b = Tensor::rand_with_seed(
        &runtime,
        Dtype::F64,
        [&space, &space],
        [&space, &space],
        12402,
    )
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
