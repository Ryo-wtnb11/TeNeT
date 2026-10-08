//! Release-mode concurrency gates of the process-global completed-transformer
//! cache (#2014-3 §5): no single-flight, so no caller parks on another's
//! build, even when work stealing re-enters a caller's Rayon region. Each
//! case runs in its own process (the counters are process-global) under a
//! watchdog, issues one cold transform from many tasks, and requires every
//! task to finish, one admission, and byte-identical results.
//!
//! Run in release: `cargo test --release -p tenet-rs --lib cache_concurrency`.

use std::sync::mpsc;
use std::sync::Arc;
use std::time::Duration;

use rayon::prelude::*;
use tenet_core::{SU2FusionRule, SU2Irrep};
use tenet_tensors::{TreeTransformOperation, TreeTransformStructure};

use crate::runtime::Runtime;
use crate::typed::{GradedSpace, TensorMap};

const TASKS: usize = 32;

/// Runs `body` on a fresh thread and fails if it does not finish: a parked
/// thread is a deadlock, not a slow test.
fn watchdog(body: impl FnOnce() + Send + 'static) {
    let (done, finished) = mpsc::channel();
    std::thread::spawn(move || {
        body();
        done.send(()).unwrap();
    });
    finished
        .recv_timeout(Duration::from_secs(300))
        .expect("concurrent cold misses did not finish: a thread parked on a build");
}

/// An SU(2) `V^2 <- V^2` tensor (non-Unique fusion, so the build runs
/// recoupling regions) and its permute.
fn fixture(runtime: &Runtime) -> TensorMap<SU2FusionRule, f64> {
    let space = GradedSpace::try_new(
        Arc::new(SU2FusionRule),
        [
            (SU2Irrep::from_twice_spin(0), 2),
            (SU2Irrep::from_twice_spin(1), 3),
            (SU2Irrep::from_twice_spin(2), 2),
            (SU2Irrep::from_twice_spin(3), 1),
        ],
    )
    .unwrap();
    TensorMap::rand_with_seed(runtime, [&space, &space], [&space, &space], 20_143).unwrap()
}

fn operation() -> TreeTransformOperation {
    TreeTransformOperation::permute([2, 0], [3, 1])
}

/// Resolves the fixture's completed transformer through one leased Runtime
/// context (planning only: the dense backend is not entered on a pool
/// worker).
fn resolve(
    runtime: &Runtime,
    source: &TensorMap<SU2FusionRule, f64>,
) -> TreeTransformStructure<f64> {
    let source = source.test_bound_space();
    let destination = source.transformed_multiplicity_free(&operation()).unwrap();
    let mut lease = runtime.lease_context().unwrap();
    lease
        .context()
        .multiplicity_free_lane::<f64>()
        .unwrap()
        .tree_context_mut()
        .compile_tree_pair_structure(
            source.provider(),
            &operation(),
            destination.space().structure(),
            source.space().structure(),
        )
        .unwrap()
}

fn admissions() -> u64 {
    crate::test_cache::completed().admissions()
}

/// The resolved transformers agree with each other and with a sequential
/// rebuild, field for field; exactly `expected` admissions happened.
fn assert_converged(built: &[TreeTransformStructure<f64>], before: u64, expected: Option<u64>) {
    assert_eq!(built.len(), TASKS);
    for structure in built {
        assert_eq!(structure, &built[0]);
    }
    if let Some(expected) = expected {
        assert_eq!(admissions() - before, expected);
    }
}

fn runtime() -> Runtime {
    Runtime::builder()
        .dense_threads(4)
        .recoupling_threads(4)
        .build()
        .unwrap()
}

#[test]
fn cache_concurrency_user_par_iter_on_the_runtime_pool() {
    if crate::test_cache::run_isolated_or_return(
        "TENET_CACHE_CONCURRENCY_RUNTIME_POOL",
        "cache_concurrency_tests::cache_concurrency_user_par_iter_on_the_runtime_pool",
    ) {
        return;
    }
    watchdog(|| {
        let runtime = runtime();
        let source = fixture(&runtime);
        crate::cache::clear();
        let before = admissions();
        let shared = runtime.execution_config().shared_ctx.clone();
        let built = shared.install(|| {
            (0..TASKS)
                .into_par_iter()
                .map(|_| resolve(&runtime, &source))
                .collect::<Vec<_>>()
        });
        assert_converged(&built, before, Some(1));
    });
}

#[test]
fn cache_concurrency_user_par_iter_on_a_foreign_pool() {
    if crate::test_cache::run_isolated_or_return(
        "TENET_CACHE_CONCURRENCY_FOREIGN_POOL",
        "cache_concurrency_tests::cache_concurrency_user_par_iter_on_a_foreign_pool",
    ) {
        return;
    }
    watchdog(|| {
        let runtime = runtime();
        let source = fixture(&runtime);
        crate::cache::clear();
        let before = admissions();
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(4)
            .build()
            .unwrap();
        let built = pool.install(|| {
            (0..TASKS)
                .into_par_iter()
                .map(|_| resolve(&runtime, &source))
                .collect::<Vec<_>>()
        });
        assert_converged(&built, before, Some(1));
    });
}

#[test]
fn cache_concurrency_ambient_pool_without_a_runtime() {
    if crate::test_cache::run_isolated_or_return(
        "TENET_CACHE_CONCURRENCY_AMBIENT_POOL",
        "cache_concurrency_tests::cache_concurrency_ambient_pool_without_a_runtime",
    ) {
        return;
    }
    watchdog(|| {
        let runtime = runtime();
        let source = fixture(&runtime);
        let source = source.test_bound_space().clone();
        let destination = source.transformed_multiplicity_free(&operation()).unwrap();
        crate::cache::clear();
        let before = admissions();
        // Standalone contexts on Rayon's global pool: no Runtime pool,
        // recoupling regions on the ambient pool.
        let built = (0..TASKS)
            .into_par_iter()
            .map(|_| {
                let mut context = tenet_tensors::TreeTransformExecutionContext::<
                    f64,
                    tenet_core::RuleIdentity,
                >::default();
                context
                    .backend_mut()
                    .set_recoupling_threads(std::num::NonZeroUsize::new(4).unwrap());
                context
                    .compile_tree_pair_structure(
                        source.provider(),
                        &operation(),
                        destination.space().structure(),
                        source.space().structure(),
                    )
                    .unwrap()
            })
            .collect::<Vec<_>>();
        assert_converged(&built, before, Some(1));
    });
}

#[test]
fn cache_concurrency_with_a_racing_clear() {
    if crate::test_cache::run_isolated_or_return(
        "TENET_CACHE_CONCURRENCY_RACING_CLEAR",
        "cache_concurrency_tests::cache_concurrency_with_a_racing_clear",
    ) {
        return;
    }
    watchdog(|| {
        let runtime = runtime();
        let source = fixture(&runtime);
        crate::cache::clear();
        let before = admissions();
        let stop = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let clearer = {
            let stop = Arc::clone(&stop);
            std::thread::spawn(move || {
                while !stop.load(std::sync::atomic::Ordering::Relaxed) {
                    crate::cache::clear();
                    std::thread::yield_now();
                }
            })
        };
        let shared = runtime.execution_config().shared_ctx.clone();
        let built = shared.install(|| {
            (0..TASKS)
                .into_par_iter()
                .map(|_| resolve(&runtime, &source))
                .collect::<Vec<_>>()
        });
        stop.store(true, std::sync::atomic::Ordering::Relaxed);
        clearer.join().unwrap();
        // Clears zero the counters, so only convergence is checked here.
        let _ = before;
        assert_converged(&built, 0, None);
    });
}

#[test]
fn global_clear_takes_no_runtime_or_device_lock() {
    // What: `tenet::cache::clear` is semantic only. With this Runtime's state
    // lock and a leased execution context held, a clear still completes.
    let runtime = runtime();
    let state = runtime.lock();
    let lease = runtime.lease_context().unwrap();
    let (done, finished) = mpsc::channel();
    std::thread::spawn(move || {
        crate::cache::clear();
        done.send(()).unwrap();
    });
    finished
        .recv_timeout(Duration::from_secs(60))
        .expect("the global clear waited on a Runtime lock");
    drop(lease);
    drop(state);
}
