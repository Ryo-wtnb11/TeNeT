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
    validate_hermitian_regions, BoundDynamicTensorRef,
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
    let rule = Arc::new(SU2FusionRule);
    let general = synthetic_su2_tensor();
    let general_space = BoundDynamicFusionMapSpace::bind_multiplicity_free(
        DynamicFusionMapSpace::from_typed(general.fusion_space().expect("fusion tensor")),
        Arc::clone(&rule),
    )
    .unwrap();
    let general_input = BoundDynamicTensorRef::try_new(&general_space, general.data()).unwrap();

    let hermitian = synthetic_hermitian_su2_tensor();
    let hermitian_space = BoundDynamicFusionMapSpace::bind_multiplicity_free(
        DynamicFusionMapSpace::from_typed(hermitian.fusion_space().expect("fusion tensor")),
        rule,
    )
    .unwrap();
    let hermitian_input =
        BoundDynamicTensorRef::try_new(&hermitian_space, hermitian.data()).unwrap();
    let regions = hermitian_space
        .space()
        .structure()
        .coupled_sector_regions(hermitian_space.space().nout())
        .unwrap()
        .expect("synthetic Hermitian layout must expose coupled-sector regions");
    validate_hermitian_regions(hermitian_input.data(), &regions)
        .expect("synthetic factorization input must be Hermitian");

    let mut summaries = sector_matricization_diagnostic(&general_input).unwrap();
    summaries.sort_by_key(|summary| summary.sector.id());
    println!(
        "general_su2 storage_len={} block_count={}",
        general.data().len(),
        general.structure().block_count()
    );
    println!(
        "hermitian_su2 storage_len={} block_count={}",
        hermitian.data().len(),
        hermitian.structure().block_count()
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
        black_box(sector_matricization_diagnostic(&general_input).unwrap());
    }
    let start = Instant::now();
    for _ in 0..mat_iters {
        black_box(sector_matricization_diagnostic(&general_input).unwrap());
    }
    let elapsed = start.elapsed();
    println!(
        "sector_matricizations iters={} total_ms={:.3} avg_us={:.3}",
        mat_iters,
        elapsed.as_secs_f64() * 1.0e3,
        elapsed.as_secs_f64() * 1.0e6 / mat_iters as f64
    );

    let mut dense = tenet_dense::DefaultDenseExecutor::new();
    black_box(svd_compact_dyn(&mut dense, &general_input).unwrap());
    ALLOCATIONS.store(0, Ordering::Relaxed);
    let start = Instant::now();
    for _ in 0..svd_iters {
        black_box(svd_compact_dyn(&mut dense, &general_input).unwrap());
    }
    let elapsed = start.elapsed();
    let allocation_calls = ALLOCATIONS.load(Ordering::Relaxed);
    println!(
        "svd_compact_dyn fixture=general iters={} total_ms={:.3} avg_us={:.3} allocation_calls={} allocation_calls_per_iter={:.2}",
        svd_iters,
        elapsed.as_secs_f64() * 1.0e3,
        elapsed.as_secs_f64() * 1.0e6 / svd_iters as f64,
        allocation_calls,
        allocation_calls as f64 / svd_iters as f64
    );

    black_box(qr_compact_dyn(&mut dense, &general_input).unwrap());
    ALLOCATIONS.store(0, Ordering::Relaxed);
    let start = Instant::now();
    for _ in 0..qr_iters {
        black_box(qr_compact_dyn(&mut dense, &general_input).unwrap());
    }
    let elapsed = start.elapsed();
    let allocation_calls = ALLOCATIONS.load(Ordering::Relaxed);
    println!(
        "qr_compact_dyn fixture=general iters={} total_ms={:.3} avg_us={:.3} allocation_calls={} allocation_calls_per_iter={:.2}",
        qr_iters,
        elapsed.as_secs_f64() * 1.0e3,
        elapsed.as_secs_f64() * 1.0e6 / qr_iters as f64,
        allocation_calls,
        allocation_calls as f64 / qr_iters as f64
    );

    black_box(eigh_full_dyn(&mut dense, &hermitian_input).unwrap());
    ALLOCATIONS.store(0, Ordering::Relaxed);
    let start = Instant::now();
    for _ in 0..eigh_iters {
        black_box(eigh_full_dyn(&mut dense, &hermitian_input).unwrap());
    }
    let elapsed = start.elapsed();
    let allocation_calls = ALLOCATIONS.load(Ordering::Relaxed);
    println!(
        "eigh_full_dyn fixture=hermitian iters={} total_ms={:.3} avg_us={:.3} allocation_calls={} allocation_calls_per_iter={:.2}",
        eigh_iters,
        elapsed.as_secs_f64() * 1.0e3,
        elapsed.as_secs_f64() * 1.0e6 / eigh_iters as f64,
        allocation_calls,
        allocation_calls as f64 / eigh_iters as f64
    );
}

fn env_usize(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(default)
}

fn synthetic_su2_space() -> FusionTensorMapSpace<2, 2> {
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
    FusionTensorMapSpace::from_degeneracy_shapes_coupled(
        TensorMapSpace::<2, 2>::from_dims([leg_dim, leg_dim], [leg_dim, leg_dim]).unwrap(),
        homspace,
        &SU2FusionRule,
        shapes,
    )
    .unwrap()
}

fn synthetic_su2_tensor() -> TensorMap<f64, 2, 2> {
    let space = synthetic_su2_space();
    let len = space.required_len().unwrap();
    TensorMap::<f64, 2, 2>::from_vec_with_fusion_space(
        (0..len)
            .map(|index| ((index * 17 + 11) % 97) as f64 / 13.0 - 3.0)
            .collect(),
        space,
    )
    .unwrap()
}

fn synthetic_hermitian_su2_tensor() -> TensorMap<f64, 2, 2> {
    let space = synthetic_su2_space();
    let len = space.required_len().unwrap();
    let regions = space
        .subblock_structure()
        .coupled_sector_regions(2)
        .unwrap()
        .expect("synthetic Hermitian layout must expose coupled-sector regions");
    let mut data = vec![0.0; len];
    for region in regions.iter() {
        assert_eq!(region.rows(), region.cols());
        let start = region.range().start;
        let n = region.rows();
        for col in 0..n {
            for row in 0..n {
                let low = row.min(col);
                let high = row.max(col);
                data[start + row + n * col] =
                    ((region.coupled().id() + 3 * low + 5 * high) % 19) as f64 * 0.5 - 4.0;
            }
        }
    }
    TensorMap::<f64, 2, 2>::from_vec_with_fusion_space(data, space).unwrap()
}
