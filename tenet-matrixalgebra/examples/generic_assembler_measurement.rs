//! Narrow diagnostic for the checked Generic factorization path up to its first dense call.
//!
//! The provider is synthetic but structurally admitted through the checked public API. It is not
//! a physical category, uses no Racah coefficients, and says nothing about dense numerical
//! performance. Fixture construction and admission stay outside measurement. Each timed call
//! includes `BoundDynamicTensorRef` construction, checked routing and matrix assembly, error
//! propagation from the rejecting dense executor, and destruction of the returned error.
//! Set `TENET_GENERIC_VALUES_BORROW_CASES=1` to compare matched canonical and padded layouts.

use std::alloc::{GlobalAlloc, Layout, System};
use std::convert::Infallible;
use std::hint::black_box;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Instant;

use tenet_core::{
    BlockKey, BlockSpec, BlockStructure, BraidingStyleKind, CheckedGenericFusion,
    FusionProductSpace, FusionRule, FusionStyleKind, FusionTensorMapSpace, FusionTreeHomSpace,
    RuleIdentity, SectorId, SectorLeg, SectorVec, TensorMapSpace,
};
use tenet_dense::{DenseDotConfig, DenseError, DenseExecutor, DenseRead, DenseTensor, DenseWrite};
use tenet_matrixalgebra::{
    svd_vals_dyn_checked_generic, BoundDynamicTensorRef, CheckedGenericFactorPlanError,
};
use tenet_tensors::{BoundDynamicFusionMapSpace, DynamicFusionMapSpace};

struct CountingAllocator;

static MEASURE_ALLOCATIONS: AtomicBool = AtomicBool::new(false);
static ALLOCATION_CALLS: AtomicUsize = AtomicUsize::new(0);
static ALLOCATION_BYTES: AtomicUsize = AtomicUsize::new(0);

// SAFETY: every operation is forwarded unchanged to `System`; relaxed atomics
// count allocation requests while the measurement gate is on.
unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        if MEASURE_ALLOCATIONS.load(Ordering::Relaxed) {
            ALLOCATION_CALLS.fetch_add(1, Ordering::Relaxed);
            ALLOCATION_BYTES.fetch_add(layout.size(), Ordering::Relaxed);
        }
        System.alloc(layout)
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        if MEASURE_ALLOCATIONS.load(Ordering::Relaxed) {
            ALLOCATION_CALLS.fetch_add(1, Ordering::Relaxed);
            ALLOCATION_BYTES.fetch_add(layout.size(), Ordering::Relaxed);
        }
        System.alloc_zeroed(layout)
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        if MEASURE_ALLOCATIONS.load(Ordering::Relaxed) {
            ALLOCATION_CALLS.fetch_add(1, Ordering::Relaxed);
            ALLOCATION_BYTES.fetch_add(new_size, Ordering::Relaxed);
        }
        System.realloc(ptr, layout, new_size)
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        System.dealloc(ptr, layout);
    }
}

#[global_allocator]
static GLOBAL: CountingAllocator = CountingAllocator;

#[derive(Clone, Copy)]
struct MeasurementRule;

fn multiplicity(sector: SectorId) -> usize {
    match sector.id() {
        1 => 1,
        2 => 2,
        4 => 4,
        8 | 9 => 8,
        16 => 16,
        _ => 0,
    }
}

impl FusionRule for MeasurementRule {
    fn rule_identity(&self) -> RuleIdentity {
        RuleIdentity::of_type::<Self>()
    }

    fn fusion_style(&self) -> FusionStyleKind {
        FusionStyleKind::Generic
    }

    fn braiding_style(&self) -> BraidingStyleKind {
        BraidingStyleKind::Bosonic
    }

    fn vacuum(&self) -> SectorId {
        SectorId::new(0)
    }

    fn dual(&self, sector: SectorId) -> SectorId {
        sector
    }

    fn fusion_channels(&self, left: SectorId, right: SectorId) -> SectorVec {
        if left == FusionRule::vacuum(self) {
            [right].into_iter().collect()
        } else if right == FusionRule::vacuum(self) || left == right && multiplicity(left) > 0 {
            [left].into_iter().collect()
        } else {
            SectorVec::new()
        }
    }

    fn nsymbol(&self, left: SectorId, right: SectorId, coupled: SectorId) -> usize {
        if left == FusionRule::vacuum(self) && right == coupled
            || right == FusionRule::vacuum(self) && left == coupled
        {
            1
        } else if left == right && right == coupled {
            multiplicity(coupled)
        } else {
            0
        }
    }
}

impl CheckedGenericFusion for MeasurementRule {
    type Error = Infallible;

    fn rule_identity(&self) -> RuleIdentity {
        FusionRule::rule_identity(self)
    }

    fn fusion_style(&self) -> FusionStyleKind {
        FusionRule::fusion_style(self)
    }

    fn braiding_style(&self) -> BraidingStyleKind {
        FusionRule::braiding_style(self)
    }

    fn vacuum(&self) -> SectorId {
        FusionRule::vacuum(self)
    }

    fn try_dual(&self, sector: SectorId) -> Result<SectorId, Self::Error> {
        Ok(FusionRule::dual(self, sector))
    }

    fn try_fusion_channels(
        &self,
        left: SectorId,
        right: SectorId,
    ) -> Result<SectorVec, Self::Error> {
        Ok(FusionRule::fusion_channels(self, left, right))
    }

    fn try_fusion_channels_in_table(
        &self,
        left: SectorId,
        right: SectorId,
    ) -> Result<SectorVec, Self::Error> {
        self.try_fusion_channels(left, right)
    }

    fn try_nsymbol(
        &self,
        left: SectorId,
        right: SectorId,
        coupled: SectorId,
    ) -> Result<usize, Self::Error> {
        Ok(FusionRule::nsymbol(self, left, right, coupled))
    }
}

struct RejectDense {
    calls: usize,
}

fn rejected() -> DenseError {
    DenseError::RankMismatch {
        shape: 0,
        strides: 0,
    }
}

impl DenseExecutor for RejectDense {
    fn svd(&mut self, input: DenseRead<'_>) -> Result<Vec<DenseTensor>, DenseError> {
        black_box(input);
        self.calls += 1;
        Err(rejected())
    }

    fn qr(&mut self, _: DenseRead<'_>) -> Result<Vec<DenseTensor>, DenseError> {
        Err(rejected())
    }

    fn eigh(&mut self, _: DenseRead<'_>) -> Result<Vec<DenseTensor>, DenseError> {
        Err(rejected())
    }

    fn dot_general_into(
        &mut self,
        _: DenseWrite<'_>,
        _: DenseRead<'_>,
        _: DenseRead<'_>,
        _: &DenseDotConfig,
    ) -> Result<(), DenseError> {
        Err(rejected())
    }
}

struct Fixture {
    space: BoundDynamicFusionMapSpace<MeasurementRule>,
    data: Vec<f64>,
}

fn canonical_fixture(labels: &[usize], degeneracy: usize) -> Fixture {
    let provider = Arc::new(MeasurementRule);
    let leg = SectorLeg::new(
        labels
            .iter()
            .copied()
            .map(|label| (SectorId::new(label), degeneracy)),
        false,
    );
    let homspace = FusionTreeHomSpace::new(
        FusionProductSpace::new([leg.clone(), leg.clone()]),
        FusionProductSpace::new([leg.clone(), leg]),
    );
    let canonical =
        BoundDynamicFusionMapSpace::from_final_homspace_generic_checked(provider, homspace)
            .unwrap();
    let data = vec![1.0; canonical.space().required_len().unwrap()];
    Fixture {
        space: canonical,
        data,
    }
}

fn fixture(labels: &[usize], degeneracy: usize, interleave: bool) -> Fixture {
    let canonical = canonical_fixture(labels, degeneracy);
    let provider = Arc::clone(canonical.space.provider_arc());
    let homspace = canonical.space.space().homspace().clone();
    let source = canonical.space.space().structure();
    let mut indices = (0..source.block_count()).collect::<Vec<_>>();
    if interleave {
        indices.sort_by_key(|&index| {
            let block = source.block(index).unwrap();
            let BlockKey::FusionTree(key) = block.key() else {
                unreachable!()
            };
            let sector = key.codomain_tree().coupled().id();
            let same_sector_before = (0..index)
                .filter(|&before| {
                    let before = source.block(before).unwrap();
                    matches!(before.key(), BlockKey::FusionTree(candidate) if candidate.codomain_tree().coupled().id() == sector)
                })
                .count();
            (same_sector_before, sector)
        });
    }
    let mut offset = 1usize;
    let blocks = indices
        .into_iter()
        .map(|index| {
            let block = source.block(index).unwrap();
            let spec = BlockSpec::column_major_with_key(
                block.key().clone(),
                block.shape().to_vec(),
                offset,
            )
            .unwrap();
            offset += block.shape().iter().product::<usize>() + 1;
            spec
        })
        .collect();
    let structure = BlockStructure::from_blocks_with_rank(4, blocks).unwrap();
    let physical = labels.len() * degeneracy;
    let typed = FusionTensorMapSpace::new_unbound(
        TensorMapSpace::<2, 2>::from_dims([physical; 2], [physical; 2]).unwrap(),
        homspace,
        structure,
    )
    .unwrap()
    .try_bind_rule(provider.as_ref())
    .unwrap();
    let dynamic = DynamicFusionMapSpace::from_typed(&typed);
    let space = BoundDynamicFusionMapSpace::bind_generic(dynamic, provider).unwrap();
    let data = vec![1.0; space.space().required_len().unwrap()];
    Fixture { space, data }
}

fn run_once(fixture: &Fixture) -> (CheckedGenericFactorPlanError<Infallible>, usize) {
    let input = BoundDynamicTensorRef::try_new(&fixture.space, &fixture.data).unwrap();
    let mut dense = RejectDense { calls: 0 };
    let error = svd_vals_dyn_checked_generic(&mut dense, &input).unwrap_err();
    (error, dense.calls)
}

fn validate_run(fixture: &Fixture) {
    let (error, calls) = run_once(fixture);
    assert!(matches!(
        error,
        CheckedGenericFactorPlanError::Operation(tenet_tensors::OperationError::Dense(
            DenseError::RankMismatch {
                shape: 0,
                strides: 0
            }
        ))
    ));
    assert_eq!(calls, 1);
}

fn measure(name: &str, fixture: &Fixture, iterations: usize, samples: usize) {
    validate_run(fixture);
    for _ in 0..16 {
        drop(black_box(run_once(fixture)));
    }
    for sample in 0..samples {
        validate_run(fixture);
        let start = Instant::now();
        for _ in 0..iterations {
            drop(black_box(run_once(fixture)));
        }
        let elapsed = start.elapsed();
        validate_run(fixture);

        ALLOCATION_CALLS.store(0, Ordering::Relaxed);
        ALLOCATION_BYTES.store(0, Ordering::Relaxed);
        MEASURE_ALLOCATIONS.store(true, Ordering::Relaxed);
        for _ in 0..iterations {
            drop(black_box(run_once(fixture)));
        }
        MEASURE_ALLOCATIONS.store(false, Ordering::Relaxed);
        validate_run(fixture);
        println!(
            "{name},{sample},{iterations},{:.3},{:.3},{:.3}",
            elapsed.as_nanos() as f64 / iterations as f64,
            ALLOCATION_CALLS.load(Ordering::Relaxed) as f64 / iterations as f64,
            ALLOCATION_BYTES.load(Ordering::Relaxed) as f64 / iterations as f64,
        );
    }
}

fn env_usize(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(default)
}

fn main() {
    let iterations = env_usize("TENET_GENERIC_ASSEMBLER_ITERS", 2_000);
    let samples = env_usize("TENET_GENERIC_ASSEMBLER_SAMPLES", 7);
    assert!(iterations > 0, "iterations must be nonzero");
    assert!(samples > 0, "samples must be nonzero");
    println!("case,sample,iterations,ns_per_iter,alloc_calls_per_iter,alloc_bytes_per_iter");
    if std::env::var("TENET_GENERIC_VALUES_BORROW_CASES").as_deref() == Ok("1") {
        for (trees, degeneracy) in [(2, 3), (8, 1)] {
            measure(
                &format!("canonical_one_sector_t{trees}_deg{degeneracy}"),
                &canonical_fixture(&[trees], degeneracy),
                iterations,
                samples,
            );
            measure(
                &format!("padded_one_sector_t{trees}_deg{degeneracy}"),
                &fixture(&[trees], degeneracy, false),
                iterations,
                samples,
            );
        }
        return;
    }
    for trees in [1, 2, 4, 8, 16] {
        measure(
            &format!("one_sector_t{trees}_deg1"),
            &fixture(&[trees], 1, false),
            iterations,
            samples,
        );
    }
    measure(
        "one_sector_t2_deg3",
        &fixture(&[2], 3, false),
        iterations,
        samples,
    );
    measure(
        "interleaved_two_sector_t8_deg1",
        &fixture(&[8, 9], 1, true),
        iterations,
        samples,
    );
    measure(
        "few_large_t1_deg4",
        &fixture(&[1], 4, false),
        iterations,
        samples,
    );
}
