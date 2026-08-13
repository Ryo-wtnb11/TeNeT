//! Issue #1124 measurement of the existing borrowed tree-transform admission seam.
//! Tenferro 0.3 exposes no cache statistics here, so fresh versus reused
//! `DenseTreeTransformOperations` is the recorded GEMM-analysis proxy.

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::mem::size_of;
use std::sync::Arc;
use std::time::Duration;

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use tenet_core::BlockStructure;
use tenet_operations::{
    DenseTreeTransformOperations, TreeTransformBackend, TreeTransformBlockSpec,
    TreeTransformReplayProfile, TreeTransformStructure, TreeTransformWorkspace,
};

struct CountingAllocator;

thread_local! {
    static COUNTING: Cell<bool> = const { Cell::new(false) };
    static ALLOCATIONS: Cell<usize> = const { Cell::new(0) };
    static ALLOCATED_BYTES: Cell<usize> = const { Cell::new(0) };
}

unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let pointer = unsafe { System.alloc(layout) };
        if !pointer.is_null() && COUNTING.get() {
            ALLOCATIONS.set(ALLOCATIONS.get() + 1);
            ALLOCATED_BYTES.set(ALLOCATED_BYTES.get() + layout.size());
        }
        pointer
    }

    unsafe fn dealloc(&self, pointer: *mut u8, layout: Layout) {
        unsafe { System.dealloc(pointer, layout) }
    }

    unsafe fn realloc(&self, pointer: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        let pointer = unsafe { System.realloc(pointer, layout, new_size) };
        if !pointer.is_null() && COUNTING.get() {
            ALLOCATIONS.set(ALLOCATIONS.get() + 1);
            ALLOCATED_BYTES.set(ALLOCATED_BYTES.get() + new_size);
        }
        pointer
    }
}

#[global_allocator]
static ALLOCATOR: CountingAllocator = CountingAllocator;

struct Fixture {
    name: &'static str,
    structure: Arc<BlockStructure>,
    a: TreeTransformStructure<f64>,
    b: TreeTransformStructure<f64>,
    src: Vec<f64>,
    dst: Vec<f64>,
}

impl Fixture {
    fn new(name: &'static str, groups: usize, width: usize, elements: usize) -> Self {
        let blocks = groups * width;
        let structure = Arc::new(
            BlockStructure::packed_column_major(1, (0..blocks).map(|_| vec![elements])).unwrap(),
        );
        let specs = |diagonal: f64| {
            (0..groups)
                .map(|group| {
                    let indices = (group * width..(group + 1) * width).collect::<Vec<_>>();
                    let coefficients = (0..width * width)
                        .map(|index| {
                            if index / width == index % width {
                                diagonal
                            } else {
                                0.125
                            }
                        })
                        .collect();
                    TreeTransformBlockSpec::multi(indices.clone(), indices, coefficients)
                })
                .collect::<Vec<_>>()
        };
        let a = TreeTransformStructure::compile_structures(&structure, &structure, &specs(1.0))
            .unwrap();
        let b = TreeTransformStructure::compile_structures(&structure, &structure, &specs(0.75))
            .unwrap();
        let len = blocks * elements;
        Self {
            name,
            structure,
            a,
            b,
            src: (0..len).map(|index| (index % 17) as f64).collect(),
            dst: vec![0.0; len],
        }
    }

    fn describe(&self) {
        let plan = self.a.recoupling_plan();
        let workspace_payload_lower_bound_bytes =
            (plan.source_len() + plan.destination_len() + plan.coefficient_len()) * size_of::<f64>();
        println!(
            "fixture={} tasks={} jobs={} runs={} structure_charged_bytes={} workspace_payload_lower_bound_bytes={}",
            self.name,
            self.a.block_count(),
            plan.jobs().len(),
            plan.runs().len(),
            self.a.charged_payload_bytes(),
            workspace_payload_lower_bound_bytes,
        );
    }
}

fn replay(
    backend: &mut DenseTreeTransformOperations,
    workspace: &mut TreeTransformWorkspace<f64>,
    transform: &TreeTransformStructure<f64>,
    structure: &Arc<BlockStructure>,
    dst: &mut [f64],
    src: &[f64],
) {
    backend
        .tree_transform_structure_overwrite_into_raw(
            workspace, transform, structure, structure, dst, src, 1.0,
        )
        .unwrap();
}

fn profiled_replay(
    backend: &mut DenseTreeTransformOperations,
    workspace: &mut TreeTransformWorkspace<f64>,
    transform: &TreeTransformStructure<f64>,
    structure: &Arc<BlockStructure>,
    dst: &mut [f64],
    src: &[f64],
) -> TreeTransformReplayProfile {
    let mut profile = TreeTransformReplayProfile::default();
    backend
        .tree_transform_structure_overwrite_into_raw_profiled(
            workspace,
            transform,
            structure,
            structure,
            dst,
            src,
            1.0,
            &mut profile,
        )
        .unwrap();
    profile
}

fn measured<T>(operation: impl FnOnce() -> T) -> (T, usize, usize) {
    ALLOCATIONS.set(0);
    ALLOCATED_BYTES.set(0);
    COUNTING.set(true);
    let value = operation();
    COUNTING.set(false);
    (value, ALLOCATIONS.get(), ALLOCATED_BYTES.get())
}

fn print_sample(label: &str, profile: TreeTransformReplayProfile, calls: usize, bytes: usize) {
    let replay_phases = profile.single_total
        + profile.multi_workspace_prepare
        + profile.multi_pack
        + profile.multi_coefficient_prepare
        + profile.multi_matmul_total
        + profile.multi_scatter
        + profile.strided_kernel;
    println!(
        "phase={} total_ns={} admission_ns={} coefficient_prepare_ns={} numerical_replay_phases_ns={} caller_allocations={} caller_requested_bytes={}",
        label,
        profile.total.as_nanos(),
        profile.validate.as_nanos(),
        profile.multi_coefficient_prepare.as_nanos(),
        replay_phases.saturating_sub(profile.multi_coefficient_prepare).as_nanos(),
        calls,
        bytes,
    );
}

fn report_first_and_warm(fixture: &mut Fixture) {
    fixture.describe();
    let ((profile, _backend, _workspace), calls, bytes) = measured(|| {
        let mut backend = DenseTreeTransformOperations::default();
        let mut workspace = TreeTransformWorkspace::default();
        let profile = profiled_replay(
            &mut backend,
            &mut workspace,
            &fixture.a,
            &fixture.structure,
            &mut fixture.dst,
            &fixture.src,
        );
        (profile, backend, workspace)
    });
    print_sample(
        "cold_fresh_workspace_fresh_executor_A",
        profile,
        calls,
        bytes,
    );

    let mut backend = DenseTreeTransformOperations::default();
    let mut workspace = TreeTransformWorkspace::default();
    replay(
        &mut backend,
        &mut workspace,
        &fixture.a,
        &fixture.structure,
        &mut fixture.dst,
        &fixture.src,
    );
    let (profile, calls, bytes) = measured(|| {
        profiled_replay(
            &mut backend,
            &mut workspace,
            &fixture.a,
            &fixture.structure,
            &mut fixture.dst,
            &fixture.src,
        )
    });
    print_sample(
        "warm_reused_workspace_reused_executor_AA",
        profile,
        calls,
        bytes,
    );

    let (profile, calls, bytes) = measured(|| {
        profiled_replay(
            &mut backend,
            &mut workspace,
            &fixture.b,
            &fixture.structure,
            &mut fixture.dst,
            &fixture.src,
        )
    });
    print_sample(
        "warm_reused_workspace_reused_executor_AB",
        profile,
        calls,
        bytes,
    );
}

fn bench_fixture(c: &mut Criterion, mut fixture: Fixture) {
    report_first_and_warm(&mut fixture);
    let mut group = c.benchmark_group(fixture.name);
    group.measurement_time(Duration::from_secs(3));
    group.sample_size(30);

    group.bench_function("fresh_workspace_fresh_executor_A", |bencher| {
        bencher.iter(|| {
            let mut backend = DenseTreeTransformOperations::default();
            let mut workspace = TreeTransformWorkspace::default();
            replay(
                &mut backend,
                &mut workspace,
                &fixture.a,
                &fixture.structure,
                &mut fixture.dst,
                &fixture.src,
            );
            black_box((backend, workspace));
        });
    });

    let mut workspace = TreeTransformWorkspace::default();
    group.bench_function("reused_workspace_fresh_executor_A", |bencher| {
        bencher.iter(|| {
            let mut backend = DenseTreeTransformOperations::default();
            replay(
                &mut backend,
                &mut workspace,
                &fixture.a,
                &fixture.structure,
                &mut fixture.dst,
                &fixture.src,
            );
            black_box(backend);
        });
    });

    let mut backend = DenseTreeTransformOperations::default();
    let mut workspace = TreeTransformWorkspace::default();
    group.bench_function("reused_workspace_reused_executor_AA", |bencher| {
        bencher.iter(|| {
            replay(
                &mut backend,
                &mut workspace,
                &fixture.a,
                &fixture.structure,
                &mut fixture.dst,
                &fixture.src,
            );
        });
    });

    let mut backend = DenseTreeTransformOperations::default();
    let mut workspace = TreeTransformWorkspace::default();
    let mut use_a = false;
    group.bench_function("reused_workspace_reused_executor_AB", |bencher| {
        bencher.iter(|| {
            use_a = !use_a;
            let transform = if use_a { &fixture.a } else { &fixture.b };
            replay(
                &mut backend,
                &mut workspace,
                transform,
                &fixture.structure,
                &mut fixture.dst,
                &fixture.src,
            );
        });
    });
    group.finish();
}

fn bench_tree_transform_lowering(c: &mut Criterion) {
    bench_fixture(c, Fixture::new("many_small", 32, 2, 8));
    bench_fixture(c, Fixture::new("few_large", 2, 8, 256));
}

criterion_group!(benches, bench_tree_transform_lowering);
criterion_main!(benches);
