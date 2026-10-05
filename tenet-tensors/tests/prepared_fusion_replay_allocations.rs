//! A warmed prepared rank-4 fusion replay allocates only the BLAS backend's
//! per-submission vectors. Counted on every thread, not only the caller: the
//! replay's dense GEMMs may run on the provider's or the context's worker
//! threads, and a caller-thread counter would miss them (#2013).

use tenet_core::{
    FermionParityFusionRule, FusionProductSpace, FusionTensorMapSpace, FusionTreeHomSpace,
    MultiplicityFreeRigidSymbols, ProductFusionRule, ProductSectorCodec, SU2Irrep, SectorId,
    SectorLeg, TensorKitProductCodec, TensorMap, TensorMapSpace, U1FusionRule, U1Irrep,
};
use tenet_tensors::{
    DenseTreeTransformOperations, OutputAxisOrder, TensorContractFusionExecutionContext,
    TensorContractSpec, TreeTransformRuleCacheKey,
};

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

static COUNTING: AtomicBool = AtomicBool::new(false);
static CALLS: AtomicU64 = AtomicU64::new(0);
static BYTES: AtomicU64 = AtomicU64::new(0);

struct ProcessWide;

fn count(size: usize) {
    if COUNTING.load(Ordering::Relaxed) {
        CALLS.fetch_add(1, Ordering::Relaxed);
        BYTES.fetch_add(size as u64, Ordering::Relaxed);
    }
}

// SAFETY: forwards every call to `System` unchanged; the counters are atomics.
unsafe impl GlobalAlloc for ProcessWide {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        count(layout.size());
        unsafe { System.alloc(layout) }
    }
    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        count(layout.size());
        unsafe { System.alloc_zeroed(layout) }
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        count(new_size);
        unsafe { System.realloc(ptr, layout, new_size) }
    }
}

#[global_allocator]
static ALLOCATOR: ProcessWide = ProcessWide;

/// Upper bound on a warmed replay's allocations, all the BLAS backend's per
/// grouped submission (#2013): Tenferro's grouped-GEMM validation
/// (`tenferro-tensor 0.7.1 src/backend.rs:670`, `validate_grouped_gemm`) and
/// the BLAS grouped dispatch's batch list (`tenferro-cpu 0.7.1
/// src/gemm/mod.rs:1630`, `grouped_gemm_blas_typed`). TeNeT itself allocates
/// nothing.
const WARM_REPLAY_CALL_BUDGET: u64 = 2;

/// Allocation calls and bytes made by `f` on all threads.
fn process_wide<T>(f: impl FnOnce() -> T) -> (T, u64, u64) {
    CALLS.store(0, Ordering::SeqCst);
    BYTES.store(0, Ordering::SeqCst);
    COUNTING.store(true, Ordering::SeqCst);
    let out = f();
    COUNTING.store(false, Ordering::SeqCst);
    (
        out,
        CALLS.load(Ordering::SeqCst),
        BYTES.load(Ordering::SeqCst),
    )
}

const WORKLOADS: [([usize; 2], [usize; 2], [usize; 4]); 3] = [
    ([2, 3], [0, 1], [0, 1, 2, 3]),
    ([3, 2], [0, 1], [0, 1, 2, 3]),
    ([3, 2], [0, 1], [1, 0, 2, 3]),
];

fn u1_sectors() -> Vec<SectorId> {
    [-1, 0, 1]
        .into_iter()
        .map(|charge| U1Irrep::new(charge).sector_id())
        .collect()
}

fn product_sectors() -> Vec<SectorId> {
    [(-1, 1), (0, 0), (1, 1)]
        .into_iter()
        .map(|(charge, parity)| {
            TensorKitProductCodec::try_encode(
                U1Irrep::new(charge).sector_id(),
                SectorId::new(parity),
            )
            .unwrap()
        })
        .collect()
}

fn assert_replay_allocates_only_the_backend_dispatch<R>(rule: &R, sectors: &[SectorId])
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64> + TreeTransformRuleCacheKey,
    R::Key: Clone + Eq + std::hash::Hash,
{
    let leg = || SectorLeg::new(sectors.iter().map(|&sector| (sector, 1)), false);
    let homspace = || {
        FusionTreeHomSpace::new(
            FusionProductSpace::new([leg(), leg()]),
            FusionProductSpace::new([leg(), leg()]),
        )
    };
    let space = |hom: FusionTreeHomSpace| {
        let key_count = hom.fusion_tree_keys(rule).len();
        FusionTensorMapSpace::from_degeneracy_shapes_coupled(
            TensorMapSpace::<2, 2>::from_dims(
                [sectors.len(), sectors.len()],
                [sectors.len(), sectors.len()],
            )
            .unwrap(),
            hom,
            rule,
            vec![vec![1; 4]; key_count],
        )
        .unwrap()
    };
    let lhs_space = space(homspace());
    let rhs_space = space(homspace());
    let lhs = TensorMap::<f64, 2, 2>::from_vec_with_fusion_space(
        (0..lhs_space.required_len().unwrap())
            .map(|index| index as f64 * 0.25 - 2.0)
            .collect(),
        lhs_space,
    )
    .unwrap();
    let rhs = TensorMap::<f64, 2, 2>::from_vec_with_fusion_space(
        (0..rhs_space.required_len().unwrap())
            .map(|index| index as f64 * 0.5 - 3.0)
            .collect(),
        rhs_space,
    )
    .unwrap();

    for (lhs_axes, rhs_axes, output_axes) in WORKLOADS {
        let axes = || {
            TensorContractSpec::new(
                &lhs_axes,
                &rhs_axes,
                OutputAxisOrder::from_axes(&output_axes),
            )
        };
        let dst_hom = FusionTreeHomSpace::tensorcontract_homspace(
            rule,
            lhs.fusion_space().unwrap().homspace(),
            rhs.fusion_space().unwrap().homspace(),
            &lhs_axes,
            &rhs_axes,
            &output_axes,
            2,
        )
        .unwrap();
        let dst_space = space(dst_hom);
        let mut expected = TensorMap::<f64, 2, 2>::from_vec_with_fusion_space(
            vec![0.0; dst_space.required_len().unwrap()],
            dst_space,
        )
        .unwrap();
        // One dense thread: a multi-threaded pool's workers allocate while
        // starting up, racing the process-wide count.
        let mut context = TensorContractFusionExecutionContext::<f64, R::Key>::new(
            DenseTreeTransformOperations::with_threads(1).unwrap(),
            DenseTreeTransformOperations::with_threads(1).unwrap(),
        );
        context
            .tensorcontract_fusion_into(rule, &mut expected, &lhs, &rhs, axes(), 1.0, 0.0)
            .unwrap();
        let mut actual = expected.clone();
        let prepared = context
            .prepare_tensorcontract_fusion(rule, &actual, &lhs, &rhs, axes())
            .unwrap();
        for _ in 0..2 {
            context
                .execute_prepared_tensorcontract_fusion(
                    &prepared,
                    rule,
                    &mut actual,
                    &lhs,
                    &rhs,
                    1.0,
                    0.0,
                )
                .unwrap();
        }
        let (result, calls, bytes) = process_wide(|| {
            context.execute_prepared_tensorcontract_fusion(
                &prepared,
                rule,
                &mut actual,
                &lhs,
                &rhs,
                1.0,
                0.0,
            )
        });
        result.unwrap();

        assert_eq!(actual.data(), expected.data());
        assert!(
            calls <= WARM_REPLAY_CALL_BUDGET,
            "axes={lhs_axes:?}/{output_axes:?}: {calls} calls, {bytes} bytes"
        );
    }
}

#[test]
fn warmed_prepared_rank4_fusion_replay_allocates_only_the_backend_dispatch() {
    assert_replay_allocates_only_the_backend_dispatch(&U1FusionRule, &u1_sectors());
    assert_replay_allocates_only_the_backend_dispatch(
        &FermionParityFusionRule,
        &[SectorId::new(0), SectorId::new(1)],
    );
    assert_replay_allocates_only_the_backend_dispatch(
        &tenet_core::SU2FusionRule,
        &[
            SU2Irrep::from_twice_spin(0).sector_id(),
            SU2Irrep::from_twice_spin(1).sector_id(),
            SU2Irrep::from_twice_spin(2).sector_id(),
        ],
    );
    assert_replay_allocates_only_the_backend_dispatch(
        &ProductFusionRule::<U1FusionRule, FermionParityFusionRule>::new(
            U1FusionRule,
            FermionParityFusionRule,
        ),
        &product_sectors(),
    );
}
