use std::alloc::{GlobalAlloc, Layout, System};
use std::hint::black_box;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Instant;

use tenet_core::{
    FusionProductSpace, FusionTensorMapSpace, FusionTreeHomSpace, SU2FusionRule, SU2Irrep,
    SectorId, SectorLeg, TensorMap, TensorMapSpace,
};
use tenet_matrixalgebra::{
    eigh_full_dyn, qr_compact_dyn, sector_matricization_diagnostic, svd_compact_dyn,
    BoundDynamicTensorRef,
};
use tenet_tensors::{BoundDynamicFusionMapSpace, DynamicFusionMapSpace};

struct CountingAllocator;

static ALLOCATIONS: AtomicUsize = AtomicUsize::new(0);

unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
        System.alloc(layout)
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
        System.alloc_zeroed(layout)
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
        System.realloc(ptr, layout, new_size)
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        System.dealloc(ptr, layout);
    }
}

#[global_allocator]
static GLOBAL: CountingAllocator = CountingAllocator;

fn main() {
    let mat_iters = env_usize("ISSUE19_MAT_ITERS", 200);
    let svd_iters = env_usize("ISSUE19_SVD_ITERS", 5);
    let qr_iters = env_usize("ISSUE19_QR_ITERS", svd_iters);
    let eigh_iters = env_usize("ISSUE19_EIGH_ITERS", svd_iters);
    let rule = SU2FusionRule;
    let tensor = synthetic_su2_tensor();
    let dyn_space =
        DynamicFusionMapSpace::from_typed(tensor.fusion_space().expect("fusion tensor"));
    let bound_space =
        BoundDynamicFusionMapSpace::bind_multiplicity_free(dyn_space.clone(), Arc::new(rule))
            .unwrap();
    let input = BoundDynamicTensorRef::try_new(&bound_space, tensor.data()).unwrap();

    let mut summaries = sector_matricization_diagnostic(&input).unwrap();
    summaries.sort_by_key(|summary| summary.sector.id());
    println!(
        "synthetic_su2 storage_len={} block_count={}",
        tensor.data().len(),
        tensor.structure().block_count()
    );
    for summary in &summaries {
        println!(
            "sector j2={} rows={} cols={} elements={}",
            SU2Irrep::from_sector_id(summary.sector).twice_spin(),
            summary.rows,
            summary.cols,
            summary.elements
        );
    }

    for _ in 0..10 {
        black_box(sector_matricization_diagnostic(&input).unwrap());
    }
    let start = Instant::now();
    for _ in 0..mat_iters {
        black_box(sector_matricization_diagnostic(&input).unwrap());
    }
    let elapsed = start.elapsed();
    println!(
        "sector_matricizations iters={} total_ms={:.3} avg_us={:.3}",
        mat_iters,
        elapsed.as_secs_f64() * 1.0e3,
        elapsed.as_secs_f64() * 1.0e6 / mat_iters as f64
    );

    let mut dense = tenet_dense::DefaultDenseExecutor::new();
    black_box(svd_compact_dyn(&mut dense, &input).unwrap());
    ALLOCATIONS.store(0, Ordering::Relaxed);
    let start = Instant::now();
    for _ in 0..svd_iters {
        black_box(svd_compact_dyn(&mut dense, &input).unwrap());
    }
    let elapsed = start.elapsed();
    let allocations = ALLOCATIONS.load(Ordering::Relaxed);
    println!(
        "svd_compact_dyn iters={} total_ms={:.3} avg_us={:.3} allocations={} allocs_per_iter={:.2}",
        svd_iters,
        elapsed.as_secs_f64() * 1.0e3,
        elapsed.as_secs_f64() * 1.0e6 / svd_iters as f64,
        allocations,
        allocations as f64 / svd_iters as f64
    );

    black_box(qr_compact_dyn(&mut dense, &input).unwrap());
    ALLOCATIONS.store(0, Ordering::Relaxed);
    let start = Instant::now();
    for _ in 0..qr_iters {
        black_box(qr_compact_dyn(&mut dense, &input).unwrap());
    }
    let elapsed = start.elapsed();
    let allocations = ALLOCATIONS.load(Ordering::Relaxed);
    println!(
        "qr_compact_dyn iters={} total_ms={:.3} avg_us={:.3} allocations={} allocs_per_iter={:.2}",
        qr_iters,
        elapsed.as_secs_f64() * 1.0e3,
        elapsed.as_secs_f64() * 1.0e6 / qr_iters as f64,
        allocations,
        allocations as f64 / qr_iters as f64
    );

    black_box(eigh_full_dyn(&mut dense, &input).unwrap());
    ALLOCATIONS.store(0, Ordering::Relaxed);
    let start = Instant::now();
    for _ in 0..eigh_iters {
        black_box(eigh_full_dyn(&mut dense, &input).unwrap());
    }
    let elapsed = start.elapsed();
    let allocations = ALLOCATIONS.load(Ordering::Relaxed);
    println!(
        "eigh_full_dyn iters={} total_ms={:.3} avg_us={:.3} allocations={} allocs_per_iter={:.2}",
        eigh_iters,
        elapsed.as_secs_f64() * 1.0e3,
        elapsed.as_secs_f64() * 1.0e6 / eigh_iters as f64,
        allocations,
        allocations as f64 / eigh_iters as f64
    );
}

fn env_usize(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(default)
}

fn synthetic_su2_tensor() -> TensorMap<f64, 2, 2> {
    let sectors = [(0usize, 2usize), (1usize, 2usize), (2usize, 3usize)];
    let leg = || {
        SectorLeg::new(
            sectors.iter().map(|&(twice_spin, degeneracy)| {
                (
                    SU2Irrep::from_twice_spin(twice_spin).sector_id(),
                    degeneracy,
                )
            }),
            false,
        )
    };
    let degeneracy_of = |sector: SectorId| -> usize {
        let twice_spin = SU2Irrep::from_sector_id(sector).twice_spin();
        sectors
            .iter()
            .find(|&&(candidate, _)| candidate == twice_spin)
            .map(|&(_, degeneracy)| degeneracy)
            .expect("sector in synthetic leg")
    };
    let leg_dim = sectors
        .iter()
        .map(|&(_, degeneracy)| degeneracy)
        .sum::<usize>();
    let homspace = FusionTreeHomSpace::new(
        FusionProductSpace::new([leg(), leg()]),
        FusionProductSpace::new([leg(), leg()]),
    );
    let shapes = homspace
        .fusion_tree_keys(&SU2FusionRule)
        .iter()
        .map(|key| {
            key.codomain_tree()
                .uncoupled()
                .iter()
                .chain(key.domain_tree().uncoupled())
                .map(|&sector| degeneracy_of(sector))
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let space = FusionTensorMapSpace::from_degeneracy_shapes_coupled(
        TensorMapSpace::<2, 2>::from_dims([leg_dim, leg_dim], [leg_dim, leg_dim]).unwrap(),
        homspace,
        &SU2FusionRule,
        shapes,
    )
    .unwrap();
    let len = space.required_len().unwrap();
    TensorMap::<f64, 2, 2>::from_vec_with_fusion_space(
        (0..len)
            .map(|index| ((index * 17 + 11) % 97) as f64 / 13.0 - 3.0)
            .collect(),
        space,
    )
    .unwrap()
}
