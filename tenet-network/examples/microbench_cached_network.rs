mod u1 {
    use std::{hint::black_box, sync::Arc, time::Instant};
    use tenet::core::{U1FusionRule, U1Irrep};
    use tenet::typed::{GradedSpace, Runtime, TensorMap};
    use tenet_network::{configure_plan_cache, plan_cache_stats, tensor, PlanCacheConfig};

    fn fixture(runtime: &Runtime) -> (TensorMap<U1FusionRule, f64>, TensorMap<U1FusionRule, f64>) {
        let space = GradedSpace::try_new_shared(
            Arc::new(U1FusionRule),
            [
                (U1Irrep::new(-1), 8),
                (U1Irrep::new(0), 16),
                (U1Irrep::new(1), 8),
            ],
        )
        .unwrap();
        (
            TensorMap::rand_with_seed(runtime, [&space, &space], [&space, &space], 12401).unwrap(),
            TensorMap::rand_with_seed(runtime, [&space, &space], [&space, &space], 12402).unwrap(),
        )
    }
    fn row(cache: &str, phase: &str, iterations: usize, elapsed: f64, runtime: &Runtime) {
        let s = plan_cache_stats(runtime);
        println!(
            "U1,{cache},{phase},{iterations},{:.3},{},{},{},{},{},{},{},{},{},{},{}",
            elapsed / iterations as f64,
            s.entries,
            s.hits,
            s.misses,
            s.workspaces_created,
            s.workspace_reuses,
            s.idle_workspaces,
            s.retained_workspace_bytes,
            s.peak_retained_workspace_bytes,
            s.workspace_byte_admissions,
            s.workspace_byte_rejections,
            s.workspace_byte_evictions
        );
    }
    pub fn run_case(cache: &str) {
        let runtime = Runtime::builder().build().unwrap();
        let enabled = cache == "enabled";
        if !enabled {
            configure_plan_cache(
                &runtime,
                PlanCacheConfig {
                    enabled: false,
                    ..Default::default()
                },
            );
        }
        let (a, b) = fixture(&runtime);
        let start = Instant::now();
        let cold = black_box(tensor!([i,j;m,n] = a[i,j;k,l] * b[k,l;m,n]).unwrap());
        let cold_elapsed = start.elapsed().as_secs_f64() * 1e6;
        row(cache, "cold", 1, cold_elapsed, &runtime);
        let cold_stats = plan_cache_stats(&runtime);
        if enabled {
            assert_eq!(
                (cold_stats.misses, cold_stats.hits, cold_stats.entries),
                (1, 0, 1)
            );
        } else {
            assert_eq!(cold_stats, Default::default());
        }
        for _ in 0..2 {
            black_box(tensor!([i,j;m,n] = a[i,j;k,l] * b[k,l;m,n]).unwrap());
        }
        let start = Instant::now();
        let mut warm = None;
        for _ in 0..20 {
            warm = Some(black_box(
                tensor!([i,j;m,n] = a[i,j;k,l] * b[k,l;m,n]).unwrap(),
            ));
        }
        let warm_elapsed = start.elapsed().as_secs_f64() * 1e6;
        row(cache, "warm", 20, warm_elapsed, &runtime);
        let stats = plan_cache_stats(&runtime);
        if enabled {
            assert!(stats.workspace_reuses > 0);
        } else {
            assert_eq!(stats, Default::default());
        }
        let oracle_runtime = Runtime::builder().build().unwrap();
        let (oa, ob) = fixture(&oracle_runtime);
        let oracle = oa.contract(&ob, &[2, 3], &[0, 1], &[0, 1, 2, 3]).unwrap();
        for actual in [&cold, warm.as_ref().unwrap()] {
            assert_eq!(actual.data(), oracle.data());
            assert_eq!(actual.codomain(), oracle.codomain());
            assert_eq!(actual.domain(), oracle.domain());
            assert_eq!(actual.block_count(), oracle.block_count());
            for index in 0..actual.block_count() {
                assert_eq!(actual.block(index).unwrap(), oracle.block(index).unwrap());
                assert_eq!(
                    actual.block_fusion_trees(index).unwrap(),
                    oracle.block_fusion_trees(index).unwrap()
                );
            }
        }
    }
}

#[cfg(not(feature = "racah-generated"))]
fn run_u1_medians() {
    let mut rows = std::collections::BTreeMap::<(String, String), Vec<(f64, Vec<String>)>>::new();
    for cache in ["enabled", "disabled"] {
        for sample in 0..3 {
            let output = std::process::Command::new(std::env::current_exe().unwrap())
                .env("U1_NETWORK_CHILD", cache)
                .output()
                .expect("U1 child");
            assert!(
                output.status.success(),
                "U1 child failed: {}",
                String::from_utf8_lossy(&output.stderr)
            );
            for line in String::from_utf8(output.stdout)
                .unwrap()
                .lines()
                .filter(|line| line.starts_with("U1,"))
            {
                println!("# raw_sample={sample},{line}");
                let fields = line.split(',').map(str::to_owned).collect::<Vec<_>>();
                rows.entry((fields[1].clone(), fields[2].clone()))
                    .or_default()
                    .push((fields[4].parse().unwrap(), fields));
            }
        }
    }
    for (_, mut samples) in rows {
        samples.sort_by(|lhs, rhs| lhs.0.total_cmp(&rhs.0));
        println!("{}", samples.remove(samples.len() / 2).1.join(","));
    }
}

#[cfg(feature = "racah-generated")]
mod checked_generic {
    use std::hint::black_box;
    use std::process::Command;
    use std::sync::Arc;
    use std::time::Instant;

    use tenet::typed::{GradedSpace, Runtime, SUNFusionRule, TensorMap};
    use tenet_network::{configure_plan_cache, plan_cache_stats, tensor, PlanCacheConfig};

    const ITERATIONS: usize = 20;

    fn fixture(
        runtime: &Runtime,
        provider: Arc<SUNFusionRule>,
        label: Vec<i64>,
    ) -> (TensorMap<SUNFusionRule, f64>, TensorMap<SUNFusionRule, f64>) {
        let space = GradedSpace::try_new_shared(provider, [(label, 2)]).expect("space");
        (
            TensorMap::rand_with_seed(runtime, [&space, &space], [&space, &space], 12401)
                .expect("lhs"),
            TensorMap::rand_with_seed(runtime, [&space, &space], [&space, &space], 12402)
                .expect("rhs"),
        )
    }

    fn assert_matches_oracle(
        actual: &TensorMap<SUNFusionRule, f64>,
        oracle: &TensorMap<SUNFusionRule, f64>,
        authority: &TensorMap<SUNFusionRule, f64>,
    ) {
        assert!(std::ptr::eq(actual.provider(), authority.provider()));
        assert_eq!(actual.codomain(), oracle.codomain());
        assert_eq!(actual.domain(), oracle.domain());
        assert_eq!(actual.block_count(), oracle.block_count());
        for index in 0..actual.block_count() {
            assert_eq!(actual.block(index).unwrap(), oracle.block(index).unwrap());
            assert_eq!(
                actual.block_fusion_trees(index).unwrap(),
                oracle.block_fusion_trees(index).unwrap()
            );
        }
        assert_eq!(actual.data(), oracle.data());
    }

    fn print_row(
        symmetry: &str,
        cache: &str,
        phase: &str,
        iterations: usize,
        elapsed_us: f64,
        runtime: &Runtime,
    ) {
        let stats = plan_cache_stats(runtime);
        println!(
            "{symmetry},{cache},{phase},{iterations},{:.3},{},{},{},{},{},{},{},{},{},{},{}",
            elapsed_us / iterations as f64,
            stats.entries,
            stats.hits,
            stats.misses,
            stats.workspaces_created,
            stats.workspace_reuses,
            stats.idle_workspaces,
            stats.retained_workspace_bytes,
            stats.peak_retained_workspace_bytes,
            stats.workspace_byte_admissions,
            stats.workspace_byte_rejections,
            stats.workspace_byte_evictions,
        );
    }

    fn run_case(
        symmetry: &str,
        provider: Arc<SUNFusionRule>,
        label: Vec<i64>,
        cache: &str,
        enabled: bool,
    ) {
        let runtime = Runtime::builder().build().expect("runtime");
        if !enabled {
            configure_plan_cache(
                &runtime,
                PlanCacheConfig {
                    enabled: false,
                    ..Default::default()
                },
            );
        }
        let (a, b) = fixture(&runtime, Arc::clone(&provider), label.clone());

        let cold_start = Instant::now();
        let cold = black_box(tensor!([i, j; m, n] = a[i, j; k, l] * b[k, l; m, n]).expect("cold"));
        let cold_us = cold_start.elapsed().as_secs_f64() * 1e6;
        print_row(symmetry, cache, "cold", 1, cold_us, &runtime);
        let cold_stats = plan_cache_stats(&runtime);
        if enabled {
            assert_eq!(
                (cold_stats.misses, cold_stats.hits, cold_stats.entries),
                (1, 0, 1)
            );
            assert!(cold_stats.retained_workspace_bytes > 0);
        } else {
            assert_eq!(cold_stats, Default::default());
        }

        for _ in 0..2 {
            black_box(tensor!([i, j; m, n] = a[i, j; k, l] * b[k, l; m, n]).expect("warmup"));
        }
        let warm_start = Instant::now();
        let mut warm_output = None;
        for _ in 0..ITERATIONS {
            warm_output = Some(black_box(
                tensor!([i, j; m, n] = a[i, j; k, l] * b[k, l; m, n]).expect("warm"),
            ));
        }
        let warm_us = warm_start.elapsed().as_secs_f64() * 1e6;
        print_row(symmetry, cache, "warm", ITERATIONS, warm_us, &runtime);
        let stats = plan_cache_stats(&runtime);
        if enabled {
            assert!(stats.retained_workspace_bytes > 0);
            assert!(stats.workspace_reuses > 0);
        } else {
            assert_eq!(stats, Default::default());
        }
        // The independent ordinary-contract oracle is deliberately post-timing:
        // no provider work is allowed to warm the measured cold call.
        let oracle_runtime = Runtime::builder().build().expect("oracle runtime");
        let (oracle_a, oracle_b) = fixture(&oracle_runtime, provider, label);
        let oracle = oracle_a
            .contract(&oracle_b, &[2, 3], &[0, 1], &[0, 1, 2, 3])
            .expect("ordinary-contract oracle");
        assert_matches_oracle(&cold, &oracle, &a);
        assert_matches_oracle(warm_output.as_ref().expect("warm output"), &oracle, &a);
    }

    fn run_fixture(symmetry: &str, n: usize, label: Vec<i64>, cache: &str) {
        let provider = Arc::new(SUNFusionRule::new(n).expect("provider"));
        run_case(symmetry, provider, label, cache, cache == "enabled");
    }

    fn run_median_fixture(symmetry: &str, cache: &str) {
        let mut rows =
            std::collections::BTreeMap::<(String, String), Vec<(f64, Vec<String>)>>::new();
        for sample in 0..3 {
            let output = Command::new(std::env::current_exe().expect("example executable"))
                .env("CHECKED_GENERIC_NETWORK_CHILD", symmetry)
                .env("CHECKED_GENERIC_NETWORK_CACHE", cache)
                .output()
                .expect("checked-Generic child");
            assert!(
                output.status.success(),
                "checked-Generic child failed: {}",
                String::from_utf8_lossy(&output.stderr)
            );
            for line in String::from_utf8(output.stdout)
                .expect("child UTF-8")
                .lines()
                .filter(|line| line.starts_with(symmetry))
            {
                println!("# raw_sample={sample},{line}");
                let fields = line.split(',').map(str::to_owned).collect::<Vec<_>>();
                rows.entry((fields[1].clone(), fields[2].clone()))
                    .or_default()
                    .push((fields[4].parse().expect("timing"), fields));
            }
        }
        for (_, mut samples) in rows {
            samples.sort_by(|lhs, rhs| lhs.0.total_cmp(&rhs.0));
            println!("{}", samples.remove(samples.len() / 2).1.join(","));
        }
    }

    pub fn main() {
        println!("symmetry,cache,phase,iterations,us_per_iter,entries,hits,misses,workspaces_created,workspace_reuses,idle_workspaces,retained_workspace_bytes,peak_retained_workspace_bytes,workspace_byte_admissions,workspace_byte_rejections,workspace_byte_evictions");
        if let Ok(symmetry) = std::env::var("CHECKED_GENERIC_NETWORK_CHILD") {
            let cache = std::env::var("CHECKED_GENERIC_NETWORK_CACHE").expect("child cache");
            match symmetry.as_str() {
                "U1" => super::u1::run_case(&cache),
                "SU3[1;1]" => run_fixture("SU3[1;1]", 3, vec![1, 1], &cache),
                "SU4[1;0;1]" => run_fixture("SU4[1;0;1]", 4, vec![1, 0, 1], &cache),
                _ => panic!("unknown checked-Generic child fixture"),
            }
            return;
        }
        println!("# checked_generic_cold_scope=fresh child with no prior operation, oracle, or cache-case run; timer starts after mandatory provider/space/tensor fixture construction, so construction-time Racah work is excluded; timings are medians of three child samples");
        for cache in ["enabled", "disabled"] {
            run_median_fixture("U1", cache);
            run_median_fixture("SU3[1;1]", cache);
            run_median_fixture("SU4[1;0;1]", cache);
        }
    }
}

#[cfg(feature = "racah-generated")]
fn main() {
    checked_generic::main();
}

#[cfg(not(feature = "racah-generated"))]
fn main() {
    println!("symmetry,cache,phase,iterations,us_per_iter,entries,hits,misses,workspaces_created,workspace_reuses,idle_workspaces,retained_workspace_bytes,peak_retained_workspace_bytes,workspace_byte_admissions,workspace_byte_rejections,workspace_byte_evictions");
    if let Ok(cache) = std::env::var("U1_NETWORK_CHILD") {
        u1::run_case(&cache);
    } else {
        println!("# U1_cold_scope=fresh child process after mandatory fixture construction; timings are medians of three child samples");
        run_u1_medians();
    }
}
