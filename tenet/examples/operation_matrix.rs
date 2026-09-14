//! Same-process cold/warm measurements for public basic tensor operations.

use std::{
    alloc::{GlobalAlloc, Layout, System},
    cell::Cell,
    convert::Infallible,
    hint::black_box,
    sync::Arc,
    time::{Duration, Instant},
};

use tenet::prelude::*;

use tenet::core::{
    complete_hom_space_structure_cache_info, fusion_tree_layout_cache_info, BlockRef, BlockSpec,
    BlockStructure, BraidingStyleKind, CheckedGenericFusion, CompleteHomSpaceStructureCacheInfo,
    FusionProductSpace, FusionRule, FusionStyleKind, FusionTensorMapSpace, FusionTreeHomSpace,
    FusionTreeLayoutCacheInfo, RuleIdentity, SectorId, SectorLeg, SectorVec, TensorMapSpace,
};
use tenet::dense::DefaultDenseExecutor;
use tenet_matrixalgebra::{
    lq_compact_dyn_checked_generic, lq_full_dyn_checked_generic, qr_compact_dyn_checked_generic,
    qr_compact_dyn_generic, qr_full_dyn_checked_generic, svd_compact_dyn_checked_generic,
    BoundDynFactor, CheckedGenericFactorPlanError, FactorScalar,
};
use tenet_tensors::{BoundDynamicFusionMapSpace, BoundDynamicTensorRef, DynamicFusionMapSpace};

struct CountingAllocator;

thread_local! {
    static COUNTING: Cell<bool> = const { Cell::new(false) };
    static ALLOCATION_CALLS: Cell<usize> = const { Cell::new(0) };
    static REQUESTED_BYTES: Cell<usize> = const { Cell::new(0) };
}

unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let pointer = unsafe { System.alloc(layout) };
        if !pointer.is_null() && COUNTING.get() {
            ALLOCATION_CALLS.set(ALLOCATION_CALLS.get() + 1);
            REQUESTED_BYTES.set(REQUESTED_BYTES.get() + layout.size());
        }
        pointer
    }

    unsafe fn dealloc(&self, pointer: *mut u8, layout: Layout) {
        unsafe { System.dealloc(pointer, layout) };
    }

    unsafe fn realloc(&self, pointer: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        let pointer = unsafe { System.realloc(pointer, layout, new_size) };
        if !pointer.is_null() && COUNTING.get() {
            ALLOCATION_CALLS.set(ALLOCATION_CALLS.get() + 1);
            REQUESTED_BYTES.set(REQUESTED_BYTES.get() + new_size);
        }
        pointer
    }
}

#[global_allocator]
static ALLOCATOR: CountingAllocator = CountingAllocator;

#[derive(Clone, Copy)]
struct Allocations {
    calls: usize,
    requested_bytes: usize,
}

fn measure_allocations<T, E>(
    operation: impl FnOnce() -> Result<T, E>,
) -> Result<(T, Allocations), E> {
    ALLOCATION_CALLS.set(0);
    REQUESTED_BYTES.set(0);
    COUNTING.set(true);
    let result = operation();
    COUNTING.set(false);
    result.map(|value| {
        (
            value,
            Allocations {
                calls: ALLOCATION_CALLS.get(),
                requested_bytes: REQUESTED_BYTES.get(),
            },
        )
    })
}

#[derive(Clone, Copy)]
struct Counters {
    runtime: RuntimeTreeTransformCacheInfo,
    fusion_layout: FusionTreeLayoutCacheInfo,
    complete_hom: CompleteHomSpaceStructureCacheInfo,
}

fn counters(runtime: &Runtime) -> Counters {
    Counters {
        runtime: runtime.tree_transform_cache_info(),
        fusion_layout: fusion_tree_layout_cache_info(),
        complete_hom: complete_hom_space_structure_cache_info(),
    }
}

fn delta(after: usize, before: usize) -> isize {
    after as isize - before as isize
}

#[expect(
    clippy::too_many_arguments,
    reason = "the example printer exposes one local benchmark row without introducing a reporting abstraction"
)]
fn print_sample(
    symmetry: &str,
    operation: &str,
    form: &str,
    phase: &str,
    iterations: u64,
    elapsed: Duration,
    allocations: Allocations,
    before: Counters,
    after: Counters,
) {
    let tree_before = before.runtime;
    let tree_after = after.runtime;
    let layout_before = before.fusion_layout;
    let layout_after = after.fusion_layout;
    let hom_before = before.complete_hom;
    let hom_after = after.complete_hom;
    println!(
        "{symmetry},{operation},{form},{phase},{iterations},{us:.3},\
         {tree_hits},{tree_misses},{tree_evictions},{tree_bypasses},{tree_entries_delta},\
         {tree_bytes_before},{tree_bytes_after},{tree_bytes_delta},\
         {layout_misses},{layout_evictions},{layout_bypasses},{layout_entries_delta},\
         {layout_bytes_before},{layout_bytes_after},{layout_bytes_delta},\
         {hom_hits},{hom_misses},{hom_admissions},{hom_evictions},{hom_bypasses},\
         {hom_entries_delta},{hom_bytes_before},{hom_bytes_after},{hom_bytes_delta},\
         NA,{allocation_calls},{requested_bytes},NA,NA,NA,NA,NA",
        us = elapsed.as_secs_f64() * 1e6 / iterations as f64,
        tree_hits = tree_after.hits() - tree_before.hits(),
        tree_misses = tree_after.misses() - tree_before.misses(),
        tree_evictions = tree_after.evictions() - tree_before.evictions(),
        tree_bypasses = tree_after.admission_bypasses() - tree_before.admission_bypasses(),
        tree_entries_delta = delta(tree_after.entries(), tree_before.entries()),
        tree_bytes_before = tree_before.charged_payload_bytes(),
        tree_bytes_after = tree_after.charged_payload_bytes(),
        tree_bytes_delta = delta(
            tree_after.charged_payload_bytes(),
            tree_before.charged_payload_bytes(),
        ),
        layout_misses = layout_after.misses() - layout_before.misses(),
        layout_evictions = layout_after.evictions() - layout_before.evictions(),
        layout_bypasses = layout_after.admission_bypasses() - layout_before.admission_bypasses(),
        layout_entries_delta = delta(layout_after.entries(), layout_before.entries()),
        layout_bytes_before = layout_before.charged_payload_bytes(),
        layout_bytes_after = layout_after.charged_payload_bytes(),
        layout_bytes_delta = delta(
            layout_after.charged_payload_bytes(),
            layout_before.charged_payload_bytes(),
        ),
        hom_hits = hom_after.hits() - hom_before.hits(),
        hom_misses = hom_after.misses() - hom_before.misses(),
        hom_admissions = hom_after.admissions() - hom_before.admissions(),
        hom_evictions = hom_after.evictions() - hom_before.evictions(),
        hom_bypasses = hom_after.bypasses() - hom_before.bypasses(),
        hom_entries_delta = delta(hom_after.entries(), hom_before.entries()),
        hom_bytes_before = hom_before.charged_bytes(),
        hom_bytes_after = hom_after.charged_bytes(),
        hom_bytes_delta = delta(hom_after.charged_bytes(), hom_before.charged_bytes()),
        allocation_calls = allocations.calls,
        requested_bytes = allocations.requested_bytes,
    );
}

#[expect(
    clippy::too_many_arguments,
    reason = "the example harness keeps local labels, phases, timing policy, and measured closure explicit"
)]
fn bench<T, E>(
    runtime: &Runtime,
    symmetry: &str,
    operation: &str,
    form: &str,
    first_phase: &str,
    repeated_phase: &str,
    min_time: Duration,
    mut operation_fn: impl FnMut() -> Result<T, E>,
) -> Result<T, E> {
    let cold_before = counters(runtime);
    let cold_start = Instant::now();
    let (cold_output, cold_allocations) = measure_allocations(&mut operation_fn)?;
    black_box(&cold_output);
    let cold_elapsed = cold_start.elapsed();
    let cold_after = counters(runtime);
    print_sample(
        symmetry,
        operation,
        form,
        first_phase,
        1,
        cold_elapsed,
        cold_allocations,
        cold_before,
        cold_after,
    );

    black_box(operation_fn()?);
    black_box(operation_fn()?);
    let warm_before = counters(runtime);
    let warm_start = Instant::now();
    let mut iterations = 0;
    let (_, warm_allocations) = measure_allocations(|| {
        while iterations < 2 || warm_start.elapsed() < min_time {
            black_box(operation_fn()?);
            iterations += 1;
        }
        Ok(())
    })?;
    let warm_elapsed = warm_start.elapsed();
    let warm_after = counters(runtime);
    print_sample(
        symmetry,
        operation,
        form,
        repeated_phase,
        iterations,
        warm_elapsed,
        warm_allocations,
        warm_before,
        warm_after,
    );
    if let Ok(milliseconds) = std::env::var("OP_MATRIX_PROFILE_PAUSE_MS") {
        std::thread::sleep(Duration::from_millis(
            milliseconds
                .parse()
                .expect("OP_MATRIX_PROFILE_PAUSE_MS must be an integer"),
        ));
    }
    Ok(cold_output)
}

fn benchmark_runtime() -> Result<Runtime, Error> {
    let backend = match std::env::var("OP_MATRIX_GEMM_BACKEND").as_deref() {
        Ok("blas") => LinalgBackend::Blas,
        Ok("faer") | Err(_) => LinalgBackend::Faer,
        Ok(other) => {
            return Err(Error::InvalidArgument(format!(
                "OP_MATRIX_GEMM_BACKEND must be `faer` or `blas`, got `{other}`"
            )))
        }
    };
    let mut builder = Runtime::builder().dense_threads(1).gemm_backend(backend);
    if std::env::var("OP_MATRIX_CACHE").as_deref() == Ok("disabled") {
        builder = builder.tree_transform_cache_byte_budget(0);
    }
    builder.build()
}

fn operation_enabled(operation: &str) -> bool {
    std::env::var("OP_MATRIX_OPERATION").map_or(true, |selected| selected == operation)
}

fn form_enabled(form: &str) -> bool {
    std::env::var("OP_MATRIX_FORM").map_or(true, |selected| selected == form)
}

fn assert_f64_payload_close(actual: &[f64], expected: &[f64]) {
    const ABS_TOLERANCE: f64 = 64.0 * f64::EPSILON;
    const REL_TOLERANCE: f64 = 256.0 * f64::EPSILON;

    assert_eq!(actual.len(), expected.len());
    for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
        assert!(
            actual.is_finite() && expected.is_finite(),
            "non-finite payload at index {index}: actual={actual}, expected={expected}"
        );
        let error = (actual - expected).abs();
        let tolerance = ABS_TOLERANCE + REL_TOLERANCE * actual.abs().max(expected.abs());
        assert!(
            error <= tolerance,
            "payload mismatch at index {index}: actual={actual}, expected={expected}, error={error}, tolerance={tolerance}"
        );
    }
}

macro_rules! assert_same_tensor {
    ($actual:expr, $expected:expr, $authority:expr) => {{
        assert!(std::ptr::eq($actual.provider(), $authority.provider()));
        assert_eq!($actual.codomain(), $expected.codomain());
        assert_eq!($actual.domain(), $expected.domain());
        assert_eq!($actual.block_count(), $expected.block_count());
        for index in 0..$actual.block_count() {
            assert_eq!($actual.block(index)?, $expected.block(index)?);
            assert_eq!(
                $actual.block_fusion_trees(index)?,
                $expected.block_fusion_trees(index)?
            );
        }
        assert_f64_payload_close($actual.data(), $expected.data());
    }};
}

#[derive(Clone, Copy)]
struct LayoutGenericRule;

impl FusionRule for LayoutGenericRule {
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
        if left.id() < 16 && right.id() < 16 {
            [SectorId::new(left.id() ^ right.id())]
                .into_iter()
                .collect()
        } else {
            SectorVec::new()
        }
    }
}

impl CheckedGenericFusion for LayoutGenericRule {
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
        Ok(FusionRule::fusion_channels(self, left, right))
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

trait HarnessScalar: FactorScalar + Copy {
    const NAME: &'static str;

    fn from_parts(real: f64, imaginary: f64) -> Self;
    fn as_complex(self) -> Complex64;
}

impl HarnessScalar for f64 {
    const NAME: &'static str = "f64";

    fn from_parts(real: f64, _imaginary: f64) -> Self {
        real
    }

    fn as_complex(self) -> Complex64 {
        Complex64::new(self, 0.0)
    }
}

impl HarnessScalar for Complex64 {
    const NAME: &'static str = "c64";

    fn from_parts(real: f64, imaginary: f64) -> Self {
        Complex64::new(real, imaginary)
    }

    fn as_complex(self) -> Complex64 {
        self
    }
}

fn layout_generic_value(sector: SectorId, row: usize, column: usize) -> f64 {
    let diagonal = f64::from(row == column) * (2.0 + sector.id() as f64);
    diagonal + ((row + 1) * 3 + (column + 1) * 5 + sector.id()) as f64 / 32.0
}

fn block_value(data: &[f64], block: BlockRef<'_>, row: usize, column: usize) -> f64 {
    data[block.offset() + row * block.strides()[0] + column * block.strides()[1]]
}

fn fill_layout_generic_data(space: &BoundDynamicFusionMapSpace<LayoutGenericRule>) -> Vec<f64> {
    let structure = space.space().structure();
    let mut data = vec![0.0; space.space().required_len().unwrap()];
    for index in 0..structure.block_count() {
        let block = structure.block(index).unwrap();
        let sector = block.key().as_fusion_tree_pair().unwrap().coupled();
        for column in 0..block.shape()[1] {
            for row in 0..block.shape()[0] {
                let offset =
                    block.offset() + row * block.strides()[0] + column * block.strides()[1];
                data[offset] = layout_generic_value(sector, row, column);
            }
        }
    }
    data
}

struct LayoutGenericInput {
    space: BoundDynamicFusionMapSpace<LayoutGenericRule>,
    data: Vec<f64>,
}

fn layout_generic_qr_fixture(
    degeneracy: usize,
) -> Result<(LayoutGenericInput, LayoutGenericInput), Box<dyn std::error::Error>> {
    let provider = Arc::new(LayoutGenericRule);
    let vacuum = SectorId::new(0);
    let charge = SectorId::new(1);
    let homspace = FusionTreeHomSpace::new(
        FusionProductSpace::new([SectorLeg::new(
            [(vacuum, degeneracy), (charge, 2 * degeneracy)],
            false,
        )]),
        FusionProductSpace::new([SectorLeg::new(
            [(vacuum, 2 * degeneracy), (charge, degeneracy)],
            false,
        )]),
    );
    let ordinary = BoundDynamicFusionMapSpace::from_final_homspace_generic(
        Arc::clone(&provider),
        homspace.clone(),
    )?;
    let expert = |reverse: bool| {
        let ordinary_structure = ordinary.space().structure();
        let mut offset = 1;
        let mut blocks = Vec::with_capacity(ordinary_structure.block_count());
        let mut indices = (0..ordinary_structure.block_count()).collect::<Vec<_>>();
        if reverse {
            indices.reverse();
        }
        for index in indices {
            let block = ordinary_structure.block(index)?;
            blocks.push(BlockSpec::column_major_with_key(
                block.key().clone(),
                block.shape().to_vec(),
                offset,
            )?);
            offset += block.element_count()? + 1;
        }
        let structure = BlockStructure::from_blocks_with_rank(2, blocks)?;
        let dense_dim = 3 * degeneracy;
        let typed = FusionTensorMapSpace::new_unbound(
            TensorMapSpace::<1, 1>::from_dims([dense_dim], [dense_dim])?,
            homspace.clone(),
            structure,
        )?
        .try_bind_rule(provider.as_ref())?;
        let dynamic = DynamicFusionMapSpace::from_typed(&typed);
        let bound = BoundDynamicFusionMapSpace::bind_generic(dynamic, Arc::clone(&provider))?;
        let data = fill_layout_generic_data(&bound);
        Ok::<_, Box<dyn std::error::Error>>(LayoutGenericInput { space: bound, data })
    };
    Ok((expert(false)?, expert(true)?))
}

fn assert_layout_generic_source(
    space: &BoundDynamicFusionMapSpace<LayoutGenericRule>,
    data: &[f64],
) {
    let structure = space.space().structure();
    assert_eq!(structure.block_count(), 2);
    for index in 0..structure.block_count() {
        let block = structure.block(index).unwrap();
        let sector = block.key().as_fusion_tree_pair().unwrap().coupled();
        for column in 0..block.shape()[1] {
            for row in 0..block.shape()[0] {
                assert_eq!(
                    block_value(data, block, row, column),
                    layout_generic_value(sector, row, column)
                );
            }
        }
    }
}

fn factor_block_for_sector<'a>(
    factor: &'a BoundDynFactor<LayoutGenericRule, f64>,
    sector: SectorId,
) -> BlockRef<'a> {
    let structure = factor.space().space().structure();
    (0..structure.block_count())
        .map(|index| structure.block(index).unwrap())
        .find(|block| block.key().as_fusion_tree_pair().unwrap().coupled() == sector)
        .expect("the compact factor retains every source sector")
}

fn assert_layout_generic_factors_equal(
    actual: &BoundDynFactor<LayoutGenericRule, f64>,
    expected: &BoundDynFactor<LayoutGenericRule, f64>,
) {
    assert_eq!(
        actual.space().space().homspace(),
        expected.space().space().homspace()
    );
    let actual_structure = actual.space().space().structure();
    let expected_structure = expected.space().space().structure();
    assert_eq!(
        actual_structure.block_count(),
        expected_structure.block_count()
    );
    for index in 0..expected_structure.block_count() {
        let expected_block = expected_structure.block(index).unwrap();
        let actual_block = actual_structure.block_by_key(expected_block.key()).unwrap();
        assert_eq!(actual_block.shape(), expected_block.shape());
        for column in 0..expected_block.shape()[1] {
            for row in 0..expected_block.shape()[0] {
                let actual_value = block_value(actual.data(), actual_block, row, column);
                let expected_value = block_value(expected.data(), expected_block, row, column);
                let tolerance = 512.0 * f64::EPSILON * expected_value.abs().max(1.0);
                assert!((actual_value - expected_value).abs() <= tolerance);
            }
        }
    }
}

fn assert_layout_generic_qr_reconstructs(
    left: &BoundDynFactor<LayoutGenericRule, f64>,
    right: &BoundDynFactor<LayoutGenericRule, f64>,
) {
    for sector in [SectorId::new(0), SectorId::new(1)] {
        let left_block = factor_block_for_sector(left, sector);
        let right_block = factor_block_for_sector(right, sector);
        assert_eq!(left_block.shape()[1], right_block.shape()[0]);
        for column in 0..right_block.shape()[1] {
            for row in 0..left_block.shape()[0] {
                let actual = (0..left_block.shape()[1])
                    .map(|inner| {
                        block_value(left.data(), left_block, row, inner)
                            * block_value(right.data(), right_block, inner, column)
                    })
                    .sum::<f64>();
                let expected = layout_generic_value(sector, row, column);
                let tolerance = 2048.0 * f64::EPSILON * expected.abs().max(1.0);
                assert!((actual - expected).abs() <= tolerance);
            }
        }
    }
}

fn run_layout_generic_qr(
    degeneracy: usize,
    min_time: Duration,
) -> Result<(), Box<dyn std::error::Error>> {
    if !operation_enabled("qr_compact_generic_layout") || !form_enabled("owned") {
        return Ok(());
    }
    let (ordered, reordered) = layout_generic_qr_fixture(degeneracy)?;
    assert_layout_generic_source(&ordered.space, &ordered.data);
    assert_layout_generic_source(&reordered.space, &reordered.data);
    let ordered_structure = ordered.space.space().structure();
    let reordered_structure = reordered.space.space().structure();
    assert_eq!(
        ordered_structure.block(0)?.key(),
        reordered_structure.block(1)?.key()
    );
    assert_eq!(
        ordered_structure.block(1)?.key(),
        reordered_structure.block(0)?.key()
    );

    let ordered_input = BoundDynamicTensorRef::try_new(&ordered.space, &ordered.data)?;
    let reordered_input = BoundDynamicTensorRef::try_new(&reordered.space, &reordered.data)?;
    let mut preflight_dense = DefaultDenseExecutor::new();
    let ordered_expected = qr_compact_dyn_generic(&mut preflight_dense, &ordered_input)?;
    let reordered_expected = qr_compact_dyn_generic(&mut preflight_dense, &reordered_input)?;
    assert_layout_generic_factors_equal(&reordered_expected.0, &ordered_expected.0);
    assert_layout_generic_factors_equal(&reordered_expected.1, &ordered_expected.1);
    assert_layout_generic_qr_reconstructs(&ordered_expected.0, &ordered_expected.1);
    assert_layout_generic_qr_reconstructs(&reordered_expected.0, &reordered_expected.1);

    println!(
        "# GenericLayout: qr_fixture_matrices=2 row_trees=2 col_trees=2 source_blocks=2 matrix_shapes={}x{},{}x{}",
        degeneracy,
        2 * degeneracy,
        2 * degeneracy,
        degeneracy
    );
    for (symmetry, input) in [
        ("GenericLayout-ordered", &ordered_input),
        ("GenericLayout-reordered", &reordered_input),
    ] {
        let runtime = benchmark_runtime()?;
        let mut dense = DefaultDenseExecutor::new();
        let (left, right) = bench(
            &runtime,
            symmetry,
            "qr_compact_generic_layout",
            "owned",
            "first_after_setup",
            "warm_after_setup",
            min_time,
            || qr_compact_dyn_generic(&mut dense, input),
        )?;
        assert_layout_generic_qr_reconstructs(&left, &right);
    }
    Ok(())
}

struct CheckedLayoutInput<D> {
    space: BoundDynamicFusionMapSpace<LayoutGenericRule>,
    data: Vec<D>,
}

#[derive(Clone, Copy)]
enum CheckedMatrixAspect {
    Alternating,
    Square,
    Wide,
    Tall,
}

fn checked_layout_value<D: HarnessScalar>(sector: SectorId, row: usize, column: usize) -> D {
    let real = layout_generic_value(sector, row, column);
    let imaginary = if D::NAME == "c64" {
        ((row + 2) * 7 + (column + 1) * 11 + sector.id() * 3) as f64 / 37.0
    } else {
        0.0
    };
    D::from_parts(real, imaginary)
}

fn fill_checked_layout_data<D: HarnessScalar>(
    space: &BoundDynamicFusionMapSpace<LayoutGenericRule>,
) -> Vec<D> {
    let structure = space.space().structure();
    let mut data = vec![D::zero(); space.space().required_len().unwrap()];
    for index in 0..structure.block_count() {
        let block = structure.block(index).unwrap();
        let sector = block.key().as_fusion_tree_pair().unwrap().coupled();
        for column in 0..block.shape()[1] {
            for row in 0..block.shape()[0] {
                let offset =
                    block.offset() + row * block.strides()[0] + column * block.strides()[1];
                data[offset] = checked_layout_value(sector, row, column);
            }
        }
    }
    data
}

fn checked_layout_fixture<D: HarnessScalar>(
    degeneracy: usize,
    sector_count: usize,
) -> Result<(CheckedLayoutInput<D>, CheckedLayoutInput<D>), Box<dyn std::error::Error>> {
    checked_layout_fixture_with_aspect(degeneracy, sector_count, CheckedMatrixAspect::Alternating)
}

fn checked_layout_fixture_with_aspect<D: HarnessScalar>(
    degeneracy: usize,
    sector_count: usize,
    aspect: CheckedMatrixAspect,
) -> Result<(CheckedLayoutInput<D>, CheckedLayoutInput<D>), Box<dyn std::error::Error>> {
    let provider = Arc::new(LayoutGenericRule);
    let codomain = (0..sector_count)
        .map(|sector| {
            (
                SectorId::new(sector),
                match aspect {
                    CheckedMatrixAspect::Alternating if sector % 2 != 0 => 2 * degeneracy,
                    CheckedMatrixAspect::Tall => 2 * degeneracy,
                    _ => degeneracy,
                },
            )
        })
        .collect::<Vec<_>>();
    let domain = (0..sector_count)
        .map(|sector| {
            (
                SectorId::new(sector),
                match aspect {
                    CheckedMatrixAspect::Alternating if sector % 2 == 0 => 2 * degeneracy,
                    CheckedMatrixAspect::Wide => 2 * degeneracy,
                    _ => degeneracy,
                },
            )
        })
        .collect::<Vec<_>>();
    let codomain_dim = codomain.iter().map(|(_, dim)| dim).sum();
    let domain_dim = domain.iter().map(|(_, dim)| dim).sum();
    let homspace = FusionTreeHomSpace::new(
        FusionProductSpace::new([SectorLeg::new(codomain, false)]),
        FusionProductSpace::new([SectorLeg::new(domain, false)]),
    );
    let canonical = BoundDynamicFusionMapSpace::from_final_homspace_generic_checked(
        Arc::clone(&provider),
        homspace.clone(),
    )?;
    let canonical_data = fill_checked_layout_data(&canonical);

    let canonical_structure = canonical.space().structure();
    let mut offset = 1;
    let mut blocks = Vec::with_capacity(canonical_structure.block_count());
    for index in (0..canonical_structure.block_count()).rev() {
        let block = canonical_structure.block(index)?;
        blocks.push(BlockSpec::column_major_with_key(
            block.key().clone(),
            block.shape().to_vec(),
            offset,
        )?);
        offset += block.element_count()? + 1;
    }
    let structure = BlockStructure::from_blocks_with_rank(2, blocks)?;
    let typed = FusionTensorMapSpace::new_unbound(
        TensorMapSpace::<1, 1>::from_dims([codomain_dim], [domain_dim])?,
        homspace,
        structure,
    )?
    .try_bind_rule(provider.as_ref())?;
    let fallback = BoundDynamicFusionMapSpace::bind_generic(
        DynamicFusionMapSpace::from_typed(&typed),
        provider,
    )?;
    let fallback_data = fill_checked_layout_data(&fallback);
    Ok((
        CheckedLayoutInput {
            space: canonical,
            data: canonical_data,
        },
        CheckedLayoutInput {
            space: fallback,
            data: fallback_data,
        },
    ))
}

fn factor_block<'a, D>(
    factor: &'a BoundDynFactor<LayoutGenericRule, D>,
    sector: SectorId,
) -> BlockRef<'a> {
    let structure = factor.space().space().structure();
    (0..structure.block_count())
        .map(|index| structure.block(index).unwrap())
        .find(|block| block.key().as_fusion_tree_pair().unwrap().coupled() == sector)
        .expect("checked compact factor retains every source sector")
}

fn checked_block_value<D: HarnessScalar>(
    data: &[D],
    block: BlockRef<'_>,
    row: usize,
    column: usize,
) -> Complex64 {
    data[block.offset() + row * block.strides()[0] + column * block.strides()[1]].as_complex()
}

fn assert_checked_source_unchanged<D: HarnessScalar>(actual: &[D], expected: &[D]) {
    assert_eq!(actual.len(), expected.len());
    for (&actual, &expected) in actual.iter().zip(expected) {
        assert_eq!(actual.as_complex(), expected.as_complex());
    }
}

fn assert_checked_pair_reconstructs<D: HarnessScalar>(
    left: &BoundDynFactor<LayoutGenericRule, D>,
    right: &BoundDynFactor<LayoutGenericRule, D>,
    sector_count: usize,
) {
    assert_eq!(left.space().space().structure().block_count(), sector_count);
    assert_eq!(
        right.space().space().structure().block_count(),
        sector_count
    );
    for sector in (0..sector_count).map(SectorId::new) {
        let left_block = factor_block(left, sector);
        let right_block = factor_block(right, sector);
        let rows = left_block.shape()[0];
        let kept = left_block.shape()[1];
        let columns = right_block.shape()[1];
        assert_eq!(right_block.shape()[0], kept);
        for column in 0..columns {
            for row in 0..rows {
                let actual = (0..kept).fold(Complex64::new(0.0, 0.0), |sum, inner| {
                    sum + checked_block_value(left.data(), left_block, row, inner)
                        * checked_block_value(right.data(), right_block, inner, column)
                });
                let expected = checked_layout_value::<D>(sector, row, column).as_complex();
                assert!((actual - expected).norm() <= 2.0e-10 * expected.norm().max(1.0));
            }
        }
    }
}

fn assert_checked_svd_reconstructs<D: HarnessScalar>(
    u: &BoundDynFactor<LayoutGenericRule, D>,
    s: &BoundDynFactor<LayoutGenericRule, D>,
    vh: &BoundDynFactor<LayoutGenericRule, D>,
    sector_count: usize,
) {
    assert_eq!(u.space().space().structure().block_count(), sector_count);
    assert_eq!(s.space().space().structure().block_count(), sector_count);
    assert_eq!(vh.space().space().structure().block_count(), sector_count);
    for sector in (0..sector_count).map(SectorId::new) {
        let u_block = factor_block(u, sector);
        let s_block = factor_block(s, sector);
        let vh_block = factor_block(vh, sector);
        let rows = u_block.shape()[0];
        let kept = u_block.shape()[1];
        let columns = vh_block.shape()[1];
        assert_eq!(s_block.shape(), [kept, kept]);
        assert_eq!(vh_block.shape()[0], kept);
        let singular_values = (0..kept)
            .map(|index| checked_block_value(s.data(), s_block, index, index))
            .collect::<Vec<_>>();
        for value in &singular_values {
            assert!(value.im.abs() <= 2.0e-12 && value.re >= 0.0);
        }
        for adjacent in singular_values.windows(2) {
            assert!(adjacent[0].re >= adjacent[1].re);
        }
        for column in 0..columns {
            for row in 0..rows {
                let actual = (0..kept).fold(Complex64::new(0.0, 0.0), |sum, inner| {
                    sum + checked_block_value(u.data(), u_block, row, inner)
                        * checked_block_value(s.data(), s_block, inner, inner)
                        * checked_block_value(vh.data(), vh_block, inner, column)
                });
                let expected = checked_layout_value::<D>(sector, row, column).as_complex();
                assert!((actual - expected).norm() <= 4.0e-10 * expected.norm().max(1.0));
            }
        }
    }
}

fn assert_columns_orthonormal<D: HarnessScalar>(
    factor: &BoundDynFactor<LayoutGenericRule, D>,
    sector_count: usize,
) {
    for sector in (0..sector_count).map(SectorId::new) {
        let block = factor_block(factor, sector);
        for right in 0..block.shape()[1] {
            for left in 0..block.shape()[1] {
                let actual = (0..block.shape()[0]).fold(Complex64::new(0.0, 0.0), |sum, row| {
                    sum + checked_block_value(factor.data(), block, row, left).conj()
                        * checked_block_value(factor.data(), block, row, right)
                });
                let expected = Complex64::new(f64::from(left == right), 0.0);
                assert!((actual - expected).norm() <= 2.0e-10);
            }
        }
    }
}

fn assert_rows_orthonormal<D: HarnessScalar>(
    factor: &BoundDynFactor<LayoutGenericRule, D>,
    sector_count: usize,
) {
    for sector in (0..sector_count).map(SectorId::new) {
        let block = factor_block(factor, sector);
        for lower in 0..block.shape()[0] {
            for upper in 0..block.shape()[0] {
                let actual = (0..block.shape()[1]).fold(Complex64::new(0.0, 0.0), |sum, column| {
                    sum + checked_block_value(factor.data(), block, upper, column)
                        * checked_block_value(factor.data(), block, lower, column).conj()
                });
                let expected = Complex64::new(f64::from(upper == lower), 0.0);
                assert!((actual - expected).norm() <= 2.0e-10);
            }
        }
    }
}

fn checked_compact_example_error(error: CheckedGenericFactorPlanError<Infallible>) -> Error {
    Error::InvalidArgument(format!("checked compact input fixture failed: {error:?}"))
}

fn preflight_checked_compact_input<D: HarnessScalar>(
    input: &BoundDynamicTensorRef<'_, LayoutGenericRule, D>,
    sector_count: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut dense = DefaultDenseExecutor::new();
    let qr =
        qr_compact_dyn_checked_generic(&mut dense, input).map_err(checked_compact_example_error)?;
    assert_checked_pair_reconstructs(&qr.0, &qr.1, sector_count);
    assert_columns_orthonormal(&qr.0, sector_count);
    drop(qr);
    let svd = svd_compact_dyn_checked_generic(&mut dense, input)
        .map_err(checked_compact_example_error)?;
    assert_checked_svd_reconstructs(&svd.0, &svd.1, &svd.2, sector_count);
    assert_columns_orthonormal(&svd.0, sector_count);
    assert_rows_orthonormal(&svd.2, sector_count);
    drop(svd);
    let lq =
        lq_compact_dyn_checked_generic(&mut dense, input).map_err(checked_compact_example_error)?;
    assert_checked_pair_reconstructs(&lq.0, &lq.1, sector_count);
    assert_rows_orthonormal(&lq.1, sector_count);
    Ok(())
}

fn run_checked_compact_operation<D: HarnessScalar>(
    operation: &str,
    dense: &mut DefaultDenseExecutor,
    input: &BoundDynamicTensorRef<'_, LayoutGenericRule, D>,
) -> Result<(), Box<dyn std::error::Error>> {
    match operation {
        "qr" => drop(black_box(
            qr_compact_dyn_checked_generic(dense, input).map_err(checked_compact_example_error)?,
        )),
        "svd" => drop(black_box(
            svd_compact_dyn_checked_generic(dense, input).map_err(checked_compact_example_error)?,
        )),
        "lq" => drop(black_box(
            lq_compact_dyn_checked_generic(dense, input).map_err(checked_compact_example_error)?,
        )),
        _ => unreachable!("fixed checked compact operation table"),
    }
    Ok(())
}

fn run_checked_compact_input_fixture<D: HarnessScalar>(
    workload: &str,
    degeneracy: usize,
    sector_count: usize,
    min_time: Duration,
) -> Result<(), Box<dyn std::error::Error>> {
    let setup_runtime = benchmark_runtime()?;
    let setup_symmetry = format!("GenericCheckedInput-{workload}-{}", D::NAME);
    drop(bench(
        &setup_runtime,
        &setup_symmetry,
        "checked_compact_input_fixture",
        "setup",
        "fixture_first",
        "fixture_repeat",
        min_time,
        || checked_layout_fixture::<D>(degeneracy, sector_count),
    )?);
    let (canonical, fallback) = checked_layout_fixture::<D>(degeneracy, sector_count)?;
    let (canonical_changed, fallback_changed) =
        checked_layout_fixture::<D>(degeneracy + 1, sector_count)?;
    let canonical_structure = canonical.space.space().structure();
    assert_eq!(canonical_structure.block_count(), sector_count);
    assert_eq!(
        fallback.space.space().structure().block_count(),
        sector_count
    );
    let mut matrix_shapes = Vec::with_capacity(sector_count);
    for sector in 0..sector_count {
        let block = canonical_structure.block(sector)?;
        assert_eq!(
            block.key().as_fusion_tree_pair().unwrap().coupled(),
            SectorId::new(sector)
        );
        matrix_shapes.push(format!("{}x{}", block.shape()[0], block.shape()[1]));
    }
    println!(
        "# GenericCheckedInput: workload={} dtype={} labels=0..{} G={} row_trees={} col_trees={} source_blocks={} matrix_shapes={} layouts=canonical,padded_reordered setup=fixture+binding+literal_preflight_outside_timer measured=public_owned_factorization+fresh_return+drop caller_allocations=requested_only peak_native_worker=NA",
        workload,
        D::NAME,
        sector_count - 1,
        canonical_structure.block_count(),
        canonical_structure.block_count(),
        canonical_structure.block_count(),
        canonical_structure.block_count(),
        matrix_shapes.join(";")
    );
    println!(
        "# GenericCheckedInput: shape_change=alternating_preconstructed_d{}_d{} fixture_construction=excluded region_tables=preinitialized_by_literal_preflight executor=reused_per_operation returned_factors=dropped_inside_timed_closure",
        degeneracy,
        degeneracy + 1
    );
    for (layout, fixture, changed_fixture) in [
        ("canonical", canonical, canonical_changed),
        ("fallback", fallback, fallback_changed),
    ] {
        let input = BoundDynamicTensorRef::try_new(&fixture.space, &fixture.data)?;
        let changed_input =
            BoundDynamicTensorRef::try_new(&changed_fixture.space, &changed_fixture.data)?;
        let original = fixture.data.clone();
        let changed_original = changed_fixture.data.clone();
        preflight_checked_compact_input(&input, sector_count)?;
        preflight_checked_compact_input(&changed_input, sector_count)?;
        assert_checked_source_unchanged(&fixture.data, &original);
        assert_checked_source_unchanged(&changed_fixture.data, &changed_original);

        let symmetry = format!("GenericCheckedInput-{workload}-{layout}-{}", D::NAME);
        let runtime = benchmark_runtime()?;
        let mut dense = DefaultDenseExecutor::new();
        let qr = bench(
            &runtime,
            &symmetry,
            "checked_compact_input_qr",
            "owned",
            "first_after_setup",
            "warm_after_setup",
            min_time,
            || {
                qr_compact_dyn_checked_generic(&mut dense, &input)
                    .map_err(checked_compact_example_error)
            },
        )?;
        assert_checked_pair_reconstructs(&qr.0, &qr.1, sector_count);
        drop(qr);
        let svd = bench(
            &runtime,
            &symmetry,
            "checked_compact_input_svd",
            "owned",
            "first_after_setup",
            "warm_after_setup",
            min_time,
            || {
                svd_compact_dyn_checked_generic(&mut dense, &input)
                    .map_err(checked_compact_example_error)
            },
        )?;
        assert_checked_svd_reconstructs(&svd.0, &svd.1, &svd.2, sector_count);
        drop(svd);
        let lq = bench(
            &runtime,
            &symmetry,
            "checked_compact_input_lq",
            "owned",
            "first_after_setup",
            "warm_after_setup",
            min_time,
            || {
                lq_compact_dyn_checked_generic(&mut dense, &input)
                    .map_err(checked_compact_example_error)
            },
        )?;
        assert_checked_pair_reconstructs(&lq.0, &lq.1, sector_count);
        drop(lq);
        assert_checked_source_unchanged(&fixture.data, &original);

        for operation in ["qr", "svd", "lq"] {
            let runtime = benchmark_runtime()?;
            let mut dense = DefaultDenseExecutor::new();
            let mut changed = false;
            bench(
                &runtime,
                &symmetry,
                &format!("checked_compact_input_{operation}_shape_alternating"),
                "owned",
                "first_after_inputs_and_regions_setup",
                "warm_shape_alternating",
                min_time,
                || {
                    let selected = if changed { &changed_input } else { &input };
                    changed = !changed;
                    run_checked_compact_operation(operation, &mut dense, selected)
                },
            )?;
        }
        assert_checked_source_unchanged(&fixture.data, &original);
        assert_checked_source_unchanged(&changed_fixture.data, &changed_original);
    }
    Ok(())
}

fn run_checked_compact_input_for<D: HarnessScalar>(
    degeneracy: usize,
    min_time: Duration,
) -> Result<(), Box<dyn std::error::Error>> {
    run_checked_compact_input_fixture::<D>("few-large", degeneracy, 2, min_time)?;
    run_checked_compact_input_fixture::<D>("many-small", (degeneracy / 16).max(1), 16, min_time)
}

fn run_checked_compact_input(
    degeneracy: usize,
    min_time: Duration,
) -> Result<(), Box<dyn std::error::Error>> {
    if !operation_enabled("checked_compact_input") || !form_enabled("owned") {
        return Ok(());
    }
    run_checked_compact_input_for::<f64>(degeneracy, min_time)?;
    run_checked_compact_input_for::<Complex64>(degeneracy, min_time)
}

fn assert_full_qr_lq<D: HarnessScalar>(
    input: &BoundDynamicTensorRef<'_, LayoutGenericRule, D>,
    left: &BoundDynFactor<LayoutGenericRule, D>,
    right: &BoundDynFactor<LayoutGenericRule, D>,
    operation: &str,
    sector_count: usize,
) {
    assert!(Arc::ptr_eq(
        input.space().provider_arc(),
        left.space().provider_arc()
    ));
    assert!(Arc::ptr_eq(
        input.space().provider_arc(),
        right.space().provider_arc()
    ));
    assert_checked_pair_reconstructs(left, right, sector_count);
    for sector in (0..sector_count).map(SectorId::new) {
        let source = (0..input.space().space().structure().block_count())
            .map(|index| input.space().space().structure().block(index).unwrap())
            .find(|block| block.key().as_fusion_tree_pair().unwrap().coupled() == sector)
            .unwrap();
        let left_block = factor_block(left, sector);
        let right_block = factor_block(right, sector);
        let (rows, columns) = (source.shape()[0], source.shape()[1]);
        let gauge_factor = if operation == "qr" { right } else { left };
        let gauge_block = factor_block(gauge_factor, sector);
        if operation == "qr" {
            assert_eq!(left_block.shape(), [rows, rows]);
            assert_eq!(right_block.shape(), [rows, columns]);
        } else {
            assert_eq!(left_block.shape(), [rows, columns]);
            assert_eq!(right_block.shape(), [columns, columns]);
        }
        for diagonal in 0..rows.min(columns) {
            let value = checked_block_value(gauge_factor.data(), gauge_block, diagonal, diagonal);
            assert!(value.im.abs() <= 2.0e-10 && value.re >= 0.0);
        }
    }
    if operation == "qr" {
        assert_columns_orthonormal(left, sector_count);
    } else {
        assert_rows_orthonormal(right, sector_count);
    }
}

fn run_checked_full_qr_lq<D: HarnessScalar>(
    operation: &str,
    dense: &mut DefaultDenseExecutor,
    input: &BoundDynamicTensorRef<'_, LayoutGenericRule, D>,
) -> Result<(), Error> {
    if operation == "qr" {
        drop(black_box(
            qr_full_dyn_checked_generic(dense, input).map_err(checked_compact_example_error)?,
        ));
    } else {
        drop(black_box(
            lq_full_dyn_checked_generic(dense, input).map_err(checked_compact_example_error)?,
        ));
    }
    Ok(())
}

fn run_full_qr_fixture<D: HarnessScalar>(
    workload: &str,
    degeneracy: usize,
    sector_count: usize,
    shape_name: &str,
    aspect: CheckedMatrixAspect,
    min_time: Duration,
) -> Result<(), Box<dyn std::error::Error>> {
    let setup_runtime = benchmark_runtime()?;
    let setup_symmetry = format!("FullQr-{workload}-{shape_name}-{}", D::NAME);
    drop(bench(
        &setup_runtime,
        &setup_symmetry,
        "full_qr_fixture",
        "setup",
        "fixture_first",
        "fixture_repeat",
        min_time,
        || checked_layout_fixture_with_aspect::<D>(degeneracy, sector_count, aspect),
    )?);
    let (canonical, padded) =
        checked_layout_fixture_with_aspect::<D>(degeneracy, sector_count, aspect)?;
    let (canonical_changed, padded_changed) =
        checked_layout_fixture_with_aspect::<D>(degeneracy + 1, sector_count, aspect)?;
    for (layout, fixture, changed_fixture) in [
        ("canonical", canonical, canonical_changed),
        ("padded", padded, padded_changed),
    ] {
        let input = BoundDynamicTensorRef::try_new(&fixture.space, &fixture.data)?;
        let changed =
            BoundDynamicTensorRef::try_new(&changed_fixture.space, &changed_fixture.data)?;
        let original = fixture.data.clone();
        let changed_original = changed_fixture.data.clone();
        let mut preflight_dense = DefaultDenseExecutor::new();
        for selected in [&input, &changed] {
            let qr = qr_full_dyn_checked_generic(&mut preflight_dense, selected)
                .map_err(checked_compact_example_error)?;
            assert_full_qr_lq(selected, &qr.0, &qr.1, "qr", sector_count);
            drop(qr);
            let lq = lq_full_dyn_checked_generic(&mut preflight_dense, selected)
                .map_err(checked_compact_example_error)?;
            assert_full_qr_lq(selected, &lq.0, &lq.1, "lq", sector_count);
            drop(lq);
        }
        drop(preflight_dense);
        let symmetry = format!("FullQr-{workload}-{shape_name}-{layout}-{}", D::NAME);
        for operation in ["qr", "lq"] {
            let runtime = benchmark_runtime()?;
            let mut dense = DefaultDenseExecutor::new();
            bench(
                &runtime,
                &symmetry,
                &format!("checked_full_{operation}"),
                "owned",
                "first_after_setup",
                "warm_after_setup",
                min_time,
                || run_checked_full_qr_lq(operation, &mut dense, &input),
            )?;
            let mut alternate = false;
            bench(
                &runtime,
                &symmetry,
                &format!("checked_full_{operation}_shape_alternating"),
                "owned",
                "first_after_inputs_setup",
                "warm_shape_alternating",
                min_time,
                || {
                    let selected = if alternate { &changed } else { &input };
                    alternate = !alternate;
                    run_checked_full_qr_lq(operation, &mut dense, selected)
                },
            )?;
        }
        assert_checked_source_unchanged(&fixture.data, &original);
        assert_checked_source_unchanged(&changed_fixture.data, &changed_original);
    }
    Ok(())
}

fn run_full_qr_lowering(min_time: Duration) -> Result<(), Box<dyn std::error::Error>> {
    if std::env::var("OP_MATRIX_OPERATION").as_deref() != Ok("full_qr_lowering")
        || !form_enabled("owned")
    {
        return Ok(());
    }
    let min_time = min_time.max(Duration::from_millis(100));
    println!("# FullQr: fixed_global_D=32 geometries=few-large:G2:d32,many-small:G16:d2 shapes=square,wide,tall layouts=canonical,padded dtypes=f64,c64 fixture_and_literal_preflight_outside_factor_timers dense_executor=DefaultDenseExecutor factorization_provider=compiled_cpu_features OP_MATRIX_GEMM_BACKEND_does_not_select_factorization_provider threads=wrapper_RAYON_BLAS_1");
    for (workload, degeneracy, sectors) in [("few-large", 32, 2), ("many-small", 2, 16)] {
        for (shape_name, aspect) in [
            ("square", CheckedMatrixAspect::Square),
            ("wide", CheckedMatrixAspect::Wide),
            ("tall", CheckedMatrixAspect::Tall),
        ] {
            run_full_qr_fixture::<f64>(
                workload, degeneracy, sectors, shape_name, aspect, min_time,
            )?;
            run_full_qr_fixture::<Complex64>(
                workload, degeneracy, sectors, shape_name, aspect, min_time,
            )?;
        }
    }
    Ok(())
}

macro_rules! run_provider {
    ($symmetry:literal, $rule:ty, $space:expr, $min_time:expr) => {{
        let space = $space;
        for operation in ["permute", "transpose", "repartition"] {
            if !operation_enabled(operation) {
                continue;
            }
            for form in ["owned", "destination"] {
                if !form_enabled(form) {
                    continue;
                }
                let runtime = benchmark_runtime()?;
                let source = TensorMap::<$rule, f64>::rand_with_seed(
                    &runtime,
                    [&space, &space],
                    [&space],
                    724,
                )?;
                if form == "owned" {
                    let cold = bench(
                        &runtime,
                        $symmetry,
                        operation,
                        form,
                        "cold",
                        "warm",
                        $min_time,
                        || match operation {
                            "permute" => source.permute(&[1], &[2, 0]),
                            "transpose" => source.transpose(),
                            "repartition" => source.repartition(1),
                            _ => unreachable!("fixed tree-operation table"),
                        },
                    )?;
                    assert!(cold.norm()?.is_finite());
                    let expected = match operation {
                        "permute" => source.permute(&[1], &[2, 0])?,
                        "transpose" => source.transpose()?,
                        "repartition" => source.repartition(1)?,
                        _ => unreachable!("fixed tree-operation table"),
                    };
                    assert_same_tensor!(cold, expected, source);
                } else {
                    let expected = match operation {
                        "permute" => source.permute(&[1], &[2, 0])?,
                        "transpose" => source.transpose()?,
                        "repartition" => source.repartition(1)?,
                        _ => unreachable!("fixed tree-operation table"),
                    };
                    let mut destination = expected.zeros_like();
                    bench(
                        &runtime,
                        $symmetry,
                        operation,
                        form,
                        "first_after_setup",
                        "warm_after_setup",
                        $min_time,
                        || match operation {
                            "permute" => {
                                source.permute_overwrite_into(&mut destination, &[1], &[2, 0], 1.0)
                            }
                            "transpose" => source.transpose_overwrite_into(&mut destination, 1.0),
                            "repartition" => {
                                source.repartition_overwrite_into(&mut destination, 1.0)
                            }
                            _ => unreachable!("fixed tree-operation table"),
                        },
                    )?;
                    assert_same_tensor!(destination, expected, source);
                }
            }
        }

        for operation in ["trace", "trace_adjoint"] {
            if !operation_enabled(operation) || !form_enabled("owned") {
                continue;
            }
            let runtime = benchmark_runtime()?;
            let source = TensorMap::<$rule, f64>::rand_with_seed(
                &runtime,
                [&space, &space],
                [&space, &space],
                727,
            )?;
            let logical_source = match operation {
                "trace" => source,
                "trace_adjoint" => source.adjoint()?,
                _ => unreachable!("fixed trace-operation table"),
            };
            let traced = bench(
                &runtime,
                $symmetry,
                operation,
                "owned",
                "cold",
                "warm",
                $min_time,
                || logical_source.trace_pairs(&[(1, 2)]),
            )?;
            assert!(traced.norm()?.is_finite());
            let expected = logical_source.trace_pairs(&[(1, 2)])?;
            assert_same_tensor!(traced, expected, logical_source);
        }

        let runtime = benchmark_runtime()?;
        let lhs = TensorMap::<$rule, f64>::rand_with_seed(
            &runtime,
            [&space, &space],
            [&space, &space],
            728,
        )?;
        let rhs = TensorMap::<$rule, f64>::rand_with_seed(
            &runtime,
            [&space, &space],
            [&space, &space],
            729,
        )?;
        let lhs_adjoint = lhs.adjoint()?;
        let rhs_adjoint = rhs.adjoint()?;
        for (suffix, left, right) in [("", &lhs, &rhs), ("_adjoint", &lhs_adjoint, &rhs_adjoint)] {
            let scale_name = format!("scale{suffix}");
            if operation_enabled(&scale_name) && form_enabled("owned") {
                let scaled = bench(
                    &runtime,
                    $symmetry,
                    &scale_name,
                    "owned",
                    "cold",
                    "warm",
                    $min_time,
                    || Ok::<_, Error>(left.scale(0.5)),
                )?;
                let error = (scaled.norm()? - 0.5 * left.norm()?).abs();
                assert!(error <= 256.0 * f64::EPSILON * left.norm()?.max(1.0));
            }

            let add_name = format!("add{suffix}");
            if operation_enabled(&add_name) && form_enabled("owned") {
                let alpha = 0.75;
                let beta = -0.25;
                let added = bench(
                    &runtime,
                    $symmetry,
                    &add_name,
                    "owned",
                    "cold",
                    "warm",
                    $min_time,
                    || left.add(right, alpha, beta),
                )?;
                let expected_norm_squared = alpha * alpha * left.norm()?.powi(2)
                    + beta * beta * right.norm()?.powi(2)
                    + 2.0 * alpha * beta * left.inner(right)?;
                let error = (added.norm()?.powi(2) - expected_norm_squared).abs();
                assert!(error <= 1024.0 * f64::EPSILON * expected_norm_squared.abs().max(1.0));
            }

            let norm_name = format!("norm{suffix}");
            if operation_enabled(&norm_name) && form_enabled("owned") {
                let value = bench(
                    &runtime,
                    $symmetry,
                    &norm_name,
                    "owned",
                    "cold",
                    "warm",
                    $min_time,
                    || left.norm(),
                )?;
                assert!(value.is_finite());
            }

            let inner_name = format!("inner{suffix}");
            if operation_enabled(&inner_name) && form_enabled("owned") {
                let value = bench(
                    &runtime,
                    $symmetry,
                    &inner_name,
                    "owned",
                    "cold",
                    "warm",
                    $min_time,
                    || left.inner(right),
                )?;
                assert!(value.is_finite());
            }
        }

        if operation_enabled("compose") && form_enabled("owned") {
            let runtime = benchmark_runtime()?;
            let lhs = TensorMap::<$rule, f64>::rand_with_seed(
                &runtime,
                [&space, &space],
                [&space, &space],
                725,
            )?;
            let rhs = TensorMap::<$rule, f64>::rand_with_seed(
                &runtime,
                [&space, &space],
                [&space, &space],
                726,
            )?;
            let composed = bench(
                &runtime,
                $symmetry,
                "compose",
                "owned",
                "cold",
                "warm",
                $min_time,
                || lhs.compose(&rhs),
            )?;
            let contracted = lhs.contract(&rhs, &[2, 3], &[0, 1], &[0, 1, 2, 3])?;
            assert_same_tensor!(composed, contracted, lhs);
        }

        for (operation, lhs_axes, rhs_axes, output_axes) in [
            (
                "contract_identity",
                &[2, 3][..],
                &[0, 1][..],
                &[0, 1, 2, 3][..],
            ),
            (
                "contract_input_swap",
                &[3, 2][..],
                &[0, 1][..],
                &[0, 1, 2, 3][..],
            ),
            (
                "contract_input_output_swap",
                &[3, 2][..],
                &[0, 1][..],
                &[1, 0, 2, 3][..],
            ),
        ] {
            if !operation_enabled(operation) {
                continue;
            }
            for form in ["owned", "destination"] {
                if !form_enabled(form) {
                    continue;
                }
                let runtime = benchmark_runtime()?;
                let lhs = TensorMap::<$rule, f64>::rand_with_seed(
                    &runtime,
                    [&space, &space],
                    [&space, &space],
                    725,
                )?;
                let rhs = TensorMap::<$rule, f64>::rand_with_seed(
                    &runtime,
                    [&space, &space],
                    [&space, &space],
                    726,
                )?;
                if form == "owned" {
                    let cold = bench(
                        &runtime,
                        $symmetry,
                        operation,
                        form,
                        "cold",
                        "warm",
                        $min_time,
                        || lhs.contract(&rhs, lhs_axes, rhs_axes, output_axes),
                    )?;
                    assert!(cold.norm()?.is_finite());
                    let expected = lhs.contract(&rhs, lhs_axes, rhs_axes, output_axes)?;
                    assert_same_tensor!(cold, expected, lhs);
                } else {
                    let expected = lhs.contract(&rhs, lhs_axes, rhs_axes, output_axes)?;
                    let mut destination = expected.zeros_like();
                    bench(
                        &runtime,
                        $symmetry,
                        operation,
                        form,
                        "first_after_setup",
                        "warm_after_setup",
                        $min_time,
                        || {
                            lhs.contract_overwrite_into(
                                &rhs,
                                &mut destination,
                                lhs_axes,
                                rhs_axes,
                                output_axes,
                                1.0,
                            )
                        },
                    )?;
                    assert_same_tensor!(destination, expected, lhs);
                }
            }
        }
    }};
}

#[cfg(feature = "racah-generated")]
fn run_checked_sun(
    symmetry: &str,
    provider: std::sync::Arc<tenet::typed::SUNFusionRule>,
    label: Vec<i64>,
    degeneracy: usize,
    min_time: Duration,
    qr_only: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    use tenet::typed::SUNFusionRule;

    let space = GradedSpace::try_new_with_arc(provider, [(label, degeneracy)])?;
    if form_enabled("destination") {
        println!("# {symmetry}: destination rows excluded: the public destination methods retain multiplicity-free dispatch bounds, so the exact SUN fixtures cannot call them");
    }
    for operation in ["permute", "transpose", "repartition"] {
        if qr_only || !operation_enabled(operation) || !form_enabled("owned") {
            continue;
        }
        let runtime = benchmark_runtime()?;
        let source = TensorMap::<SUNFusionRule, f64>::rand_with_seed(
            &runtime,
            [&space, &space],
            [&space],
            724,
        )?;
        let cold = bench(
            &runtime,
            symmetry,
            operation,
            "owned",
            "cold",
            "warm",
            min_time,
            || match operation {
                "permute" => source.permute(&[1], &[2, 0]),
                "transpose" => source.transpose(),
                "repartition" => source.repartition(1),
                _ => unreachable!("fixed tree-operation table"),
            },
        )?;
        let expected = match operation {
            "permute" => source.permute(&[1], &[2, 0])?,
            "transpose" => source.transpose()?,
            "repartition" => source.repartition(1)?,
            _ => unreachable!("fixed tree-operation table"),
        };
        assert_same_tensor!(cold, expected, source);
    }

    if !qr_only && (operation_enabled("trace") || operation_enabled("trace_adjoint")) {
        println!("# {symmetry}: trace rows excluded: checked-Generic trace dispatch exists, but SUNFusionRule lacks the required SectorCodec");
    }

    let runtime = benchmark_runtime()?;
    let lhs = TensorMap::<SUNFusionRule, f64>::rand_with_seed(
        &runtime,
        [&space, &space],
        [&space, &space],
        725,
    )?;
    let rhs = TensorMap::<SUNFusionRule, f64>::rand_with_seed(
        &runtime,
        [&space, &space],
        [&space, &space],
        726,
    )?;
    if operation_enabled("qr_compact") && form_enabled("owned") {
        let mut sectors = Vec::new();
        let mut row_trees = Vec::new();
        let mut col_trees = Vec::new();
        for index in 0..lhs.block_count() {
            let trees = lhs.block_fusion_trees(index)?;
            if !sectors.contains(trees.coupled()) {
                sectors.push(trees.coupled().clone());
            }
            let row = (
                trees.coupled().clone(),
                trees.codomain_uncoupled().to_vec(),
                trees.codomain_innerlines().to_vec(),
                trees.codomain_vertices().to_vec(),
            );
            if !row_trees.contains(&row) {
                row_trees.push(row);
            }
            let column = (
                trees.coupled().clone(),
                trees.domain_uncoupled().to_vec(),
                trees.domain_innerlines().to_vec(),
                trees.domain_vertices().to_vec(),
            );
            if !col_trees.contains(&column) {
                col_trees.push(column);
            }
        }
        println!(
            "# {symmetry}: qr_fixture_matrices={} row_trees={} col_trees={} source_blocks={}",
            sectors.len(),
            row_trees.len(),
            col_trees.len(),
            lhs.block_count()
        );
    }
    for (operation, action) in [
        ("scale", 0),
        ("add", 1),
        ("norm", 2),
        ("inner", 3),
        ("compose", 4),
        ("contract_identity", 5),
        ("contract_input_swap", 6),
        ("contract_input_output_swap", 7),
        ("qr_compact", 8),
    ] {
        if (qr_only && action != 8) || !operation_enabled(operation) || !form_enabled("owned") {
            continue;
        }
        match action {
            0 => {
                let scaled = bench(
                    &runtime,
                    symmetry,
                    operation,
                    "owned",
                    "cold",
                    "warm",
                    min_time,
                    || {
                        Ok::<_, tenet::typed::GenericTensorError<tenet::typed::SUNFusionRuleError>>(
                            lhs.scale(0.5),
                        )
                    },
                )?;
                for (&actual, &input) in scaled.data().iter().zip(lhs.data()) {
                    assert_eq!(actual, 0.5 * input);
                }
                assert_same_tensor!(scaled, lhs.scale(0.5), lhs);
            }
            1 => {
                let added = bench(
                    &runtime,
                    symmetry,
                    operation,
                    "owned",
                    "cold",
                    "warm",
                    min_time,
                    || lhs.add(&rhs, 0.75, -0.25),
                )?;
                for ((&actual, &left), &right) in
                    added.data().iter().zip(lhs.data()).zip(rhs.data())
                {
                    assert_eq!(actual, 0.75 * left - 0.25 * right);
                }
                assert_same_tensor!(added, lhs.add(&rhs, 0.75, -0.25)?, lhs);
            }
            2 => {
                let norm = bench(
                    &runtime,
                    symmetry,
                    operation,
                    "owned",
                    "cold",
                    "warm",
                    min_time,
                    || lhs.norm(),
                )?;
                let inner_self = lhs.inner(&lhs)?;
                assert!(
                    (norm * norm - inner_self).abs()
                        <= 256.0 * f64::EPSILON * inner_self.abs().max(1.0)
                );
            }
            3 => {
                let inner = bench(
                    &runtime,
                    symmetry,
                    operation,
                    "owned",
                    "cold",
                    "warm",
                    min_time,
                    || lhs.inner(&rhs),
                )?;
                let reverse = rhs.inner(&lhs)?;
                assert!((inner - reverse).abs() <= 256.0 * f64::EPSILON * inner.abs().max(1.0));
            }
            4 => {
                let composed = bench(
                    &runtime,
                    symmetry,
                    operation,
                    "owned",
                    "cold",
                    "warm",
                    min_time,
                    || lhs.compose(&rhs),
                )?;
                assert_same_tensor!(
                    composed,
                    lhs.contract(&rhs, &[2, 3], &[0, 1], &[0, 1, 2, 3])?,
                    lhs
                );
            }
            8 => {
                let (q, r) = bench(
                    &runtime,
                    symmetry,
                    operation,
                    "owned",
                    "cold",
                    "warm",
                    min_time,
                    || lhs.qr_compact(),
                )?;
                assert_same_tensor!(q.compose(&r)?, lhs, lhs);
            }
            _ => {
                let (lhs_axes, output_axes) = match action {
                    5 => (&[2, 3][..], &[0, 1, 2, 3][..]),
                    6 => (&[3, 2][..], &[0, 1, 2, 3][..]),
                    7 => (&[3, 2][..], &[1, 0, 2, 3][..]),
                    _ => unreachable!(),
                };
                let output = bench(
                    &runtime,
                    symmetry,
                    operation,
                    "owned",
                    "cold",
                    "warm",
                    min_time,
                    || lhs.contract(&rhs, lhs_axes, &[0, 1], output_axes),
                )?;
                assert_same_tensor!(
                    output,
                    lhs.contract(&rhs, lhs_axes, &[0, 1], output_axes)?,
                    lhs
                );
            }
        }
    }
    Ok(())
}

#[cfg(feature = "racah-generated")]
fn lazy_transform_source<D: HarnessScalar + TensorScalar>(
    runtime: &Runtime,
    provider: Arc<tenet::typed::SUNFusionRule>,
    label: Vec<i64>,
    rank: usize,
    extent: usize,
) -> Result<TensorMap<tenet::typed::SUNFusionRule, D>, Box<dyn std::error::Error>> {
    let legs = (0..rank)
        .map(|axis| {
            GradedSpace::try_new_with_arc(
                Arc::clone(&provider),
                [(label.clone(), extent + axis % 2)],
            )
        })
        .collect::<Result<Vec<_>, _>>()?;
    let nout = rank - 1;
    Ok(TensorMap::from_block_fn(
        runtime,
        legs[..nout].iter(),
        legs[nout..].iter(),
        |trees, indices| {
            let weighted_labels = |labels: &[Vec<i64>], factor| {
                labels
                    .iter()
                    .flatten()
                    .enumerate()
                    .map(|(axis, value)| (axis + 1) * factor * value.unsigned_abs() as usize)
                    .sum::<usize>()
            };
            let label_marker = trees
                .coupled()
                .iter()
                .enumerate()
                .map(|(axis, value)| (axis + 1) * 97 * value.unsigned_abs() as usize)
                .sum::<usize>()
                + weighted_labels(trees.codomain_uncoupled(), 13)
                + weighted_labels(trees.domain_uncoupled(), 37)
                + weighted_labels(trees.codomain_innerlines(), 53)
                + weighted_labels(trees.domain_innerlines(), 71);
            let vertex_marker = trees
                .codomain_vertices()
                .iter()
                .enumerate()
                .map(|(axis, vertex)| (axis + 1) * 19 * vertex.get())
                .chain(
                    trees
                        .domain_vertices()
                        .iter()
                        .enumerate()
                        .map(|(axis, vertex)| (axis + 1) * 43 * vertex.get()),
                )
                .sum::<usize>();
            let index_marker = indices
                .iter()
                .enumerate()
                .map(|(axis, &index)| (axis + 2) * (axis + 5) * index)
                .sum::<usize>();
            let marker = label_marker + vertex_marker + index_marker;
            D::from_parts(1.0 + marker as f64 / 17.0, 0.25 + marker as f64 / 29.0)
        },
    )?)
}

#[cfg(feature = "racah-generated")]
fn apply_lazy_transform<D: HarnessScalar + TensorScalar>(
    source: &TensorMap<tenet::typed::SUNFusionRule, D>,
    operation: &str,
) -> Result<TensorMap<tenet::typed::SUNFusionRule, D>, Error> {
    let rank = source.rank();
    let nout = source.codomain().len();
    match (operation, rank) {
        ("permute", 3) => source.permute(&[1], &[2, 0]),
        ("braid", 3) => source.braid(&[1], &[2, 0], &[5, 2, 7]),
        ("transpose", _) => source.transpose(),
        ("transpose_axes", _) => {
            let mut planar = (0..nout).chain((nout..rank).rev()).collect::<Vec<_>>();
            planar.rotate_left(1);
            let target_nout = rank - nout;
            let codomain = planar[..target_nout].to_vec();
            let domain = planar[target_nout..]
                .iter()
                .rev()
                .copied()
                .collect::<Vec<_>>();
            source.transpose_axes(&codomain, &domain)
        }
        ("repartition", 3) => source.repartition(rank - nout),
        ("permute", 4) => source.permute(&[2, 0], &[3, 1]),
        ("braid", 4) => source.braid(&[2, 0], &[3, 1], &[7, 1, 6, 3]),
        ("repartition", 4) => source.repartition(rank - nout),
        _ => unreachable!("fixed lazy transform table"),
    }
}

#[cfg(feature = "racah-generated")]
fn assert_lazy_transform_same<D: HarnessScalar + TensorScalar>(
    actual: &TensorMap<tenet::typed::SUNFusionRule, D>,
    expected: &TensorMap<tenet::typed::SUNFusionRule, D>,
) -> Result<(), Error> {
    assert_eq!(actual.codomain(), expected.codomain());
    assert_eq!(actual.domain(), expected.domain());
    assert_eq!(actual.block_count(), expected.block_count());
    for index in 0..actual.block_count() {
        let actual_block = actual.block(index)?;
        let expected_block = expected.block(index)?;
        assert_eq!(actual_block.shape(), expected_block.shape());
        assert_eq!(actual_block.strides(), expected_block.strides());
        assert_eq!(actual_block.offset(), expected_block.offset());
        assert_eq!(
            actual.block_fusion_trees(index)?,
            expected.block_fusion_trees(index)?
        );
    }
    assert_eq!(actual.data().len(), expected.data().len());
    assert!(actual
        .data()
        .iter()
        .zip(expected.data())
        .all(|(&a, &b)| (a.as_complex() - b.as_complex()).norm()
            <= 1.0e-11 * (1.0 + b.as_complex().norm())));
    Ok(())
}

#[cfg(feature = "racah-generated")]
fn preflight_lazy_transform<D: HarnessScalar + TensorScalar>(
    n: usize,
    label: Vec<i64>,
    rank: usize,
    extent: usize,
) -> Result<(usize, usize, usize), Box<dyn std::error::Error>> {
    use tenet::typed::SUNFusionRule;
    let oracle_runtime = benchmark_runtime()?;
    let oracle_provider = Arc::new(SUNFusionRule::new(n)?);
    let oracle_parent = lazy_transform_source::<D>(
        &oracle_runtime,
        oracle_provider,
        label.clone(),
        rank,
        extent,
    )?;
    let oracle_lazy = oracle_parent.adjoint()?;
    let oracle_data = oracle_lazy.data();
    let expected_source = TensorMap::from_block_fn(
        &oracle_runtime,
        oracle_lazy.codomain().iter(),
        oracle_lazy.domain().iter(),
        |trees, indices| {
            let index = (0..oracle_lazy.block_count())
                .find(|&index| oracle_lazy.block_fusion_trees(index).unwrap() == trees.clone())
                .expect("oracle logical block exists");
            let block = oracle_lazy.block(index).unwrap();
            oracle_data[block.offset()
                + indices
                    .iter()
                    .zip(block.strides())
                    .map(|(&coordinate, &stride)| coordinate * stride)
                    .sum::<usize>()]
        },
    )?;

    let runtime = benchmark_runtime()?;
    let provider = Arc::new(SUNFusionRule::new(n)?);
    let source = lazy_transform_source::<D>(&runtime, provider, label, rank, extent)?;
    let lazy = source.adjoint()?;
    for operation in [
        "permute",
        "braid",
        "transpose",
        "transpose_axes",
        "repartition",
    ] {
        let actual = apply_lazy_transform(&lazy, operation)?;
        let expected = apply_lazy_transform(&expected_source, operation)?;
        assert!(std::ptr::eq(actual.provider(), source.provider()));
        assert_lazy_transform_same(&actual, &expected)?;
    }
    let mut coupled = Vec::new();
    for index in 0..source.block_count() {
        let sector = source.block_fusion_trees(index)?.coupled().clone();
        if !coupled.contains(&sector) {
            coupled.push(sector);
        }
    }
    Ok((source.block_count(), coupled.len(), source.data().len()))
}

#[cfg(feature = "racah-generated")]
fn run_lazy_transform_row<D: HarnessScalar + TensorScalar>(
    symmetry: &str,
    n: usize,
    label: Vec<i64>,
    rank: usize,
    operation: &str,
    form: &str,
    extent: usize,
    alternating: bool,
    min_time: Duration,
) -> Result<(), Box<dyn std::error::Error>> {
    use tenet::typed::SUNFusionRule;
    let runtime = benchmark_runtime()?;
    let provider = Arc::new(SUNFusionRule::new(n)?);
    let source =
        lazy_transform_source::<D>(&runtime, Arc::clone(&provider), label.clone(), rank, extent)?;
    let changed_extent = if extent == 1 { 3 } else { 1 };
    let changed = lazy_transform_source::<D>(&runtime, provider, label, rank, changed_extent)?;
    let source = if form == "lazy" {
        source.adjoint()?
    } else {
        source
    };
    let changed = if form == "lazy" {
        changed.adjoint()?
    } else {
        changed
    };
    let first_phase = if alternating {
        format!("first_after_setup_extent{extent}_alternating")
    } else {
        format!("first_after_setup_extent{extent}_fixed")
    };
    let repeated_phase = if alternating {
        format!("warm_extent{extent}_{changed_extent}_alternating")
    } else {
        format!("warm_extent{extent}_fixed")
    };
    let mut alternate = false;
    bench(
        &runtime,
        symmetry,
        &format!("lazy_tree_transform_{operation}_{}", D::NAME),
        form,
        &first_phase,
        &repeated_phase,
        min_time,
        || {
            let input = if alternating && alternate {
                &changed
            } else {
                &source
            };
            alternate = !alternate;
            let output = apply_lazy_transform(input, operation)?;
            black_box(output.data().first());
            drop(black_box(output));
            Ok::<(), Error>(())
        },
    )?;
    Ok(())
}

#[cfg(feature = "racah-generated")]
fn run_lazy_tree_transform(min_time: Duration) -> Result<(), Box<dyn std::error::Error>> {
    if std::env::var("OP_MATRIX_OPERATION").as_deref() != Ok("lazy_tree_transform") {
        return Ok(());
    }
    if !form_enabled("owned") {
        println!("# LazyTreeTransform: destination rows excluded: checked-Generic lazy transforms expose the public owned-result form only");
        return Ok(());
    }
    println!("# LazyTreeTransform: public_owned_output full_output_and_first_data_observation_and_drop_inside_timer oracle=separate_runtime_owned_logical_fixture extents=1,3 direct_source_split=rank-1 lazy_source_split=1 planar_target_split=opposite direct_control=true peak_and_copy_counters=NA");
    for (symmetry, n, label, rank) in [
        ("SU3[1;1]-rank3", 3, vec![1, 1], 3),
        ("SU4[1;0;1]-rank4", 4, vec![1, 0, 1], 4),
    ] {
        for extent in [1, 3] {
            let (blocks_f64, sectors_f64, stored_f64) =
                preflight_lazy_transform::<f64>(n, label.clone(), rank, extent)?;
            let (blocks_c64, sectors_c64, stored_c64) =
                preflight_lazy_transform::<Complex64>(n, label.clone(), rank, extent)?;
            println!("# LazyTreeTransformFixture symmetry={symmetry} rank={rank} extent={extent} coupled_sectors_f64={sectors_f64} blocks_f64={blocks_f64} stored_f64={stored_f64} coupled_sectors_c64={sectors_c64} blocks_c64={blocks_c64} stored_c64={stored_c64}");
        }
        for operation in [
            "permute",
            "braid",
            "transpose",
            "transpose_axes",
            "repartition",
        ] {
            for form in ["direct", "lazy"] {
                for extent in [1, 3] {
                    run_lazy_transform_row::<f64>(
                        symmetry,
                        n,
                        label.clone(),
                        rank,
                        operation,
                        form,
                        extent,
                        false,
                        min_time,
                    )?;
                    run_lazy_transform_row::<Complex64>(
                        symmetry,
                        n,
                        label.clone(),
                        rank,
                        operation,
                        form,
                        extent,
                        false,
                        min_time,
                    )?;
                }
                run_lazy_transform_row::<f64>(
                    symmetry,
                    n,
                    label.clone(),
                    rank,
                    operation,
                    form,
                    1,
                    true,
                    min_time,
                )?;
                run_lazy_transform_row::<Complex64>(
                    symmetry,
                    n,
                    label.clone(),
                    rank,
                    operation,
                    form,
                    1,
                    true,
                    min_time,
                )?;
            }
        }
    }
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    if let Ok(operation) = std::env::var("OP_MATRIX_OPERATION") {
        if !matches!(
            operation.as_str(),
            "permute"
                | "transpose"
                | "repartition"
                | "trace"
                | "trace_adjoint"
                | "scale"
                | "scale_adjoint"
                | "add"
                | "add_adjoint"
                | "norm"
                | "norm_adjoint"
                | "inner"
                | "inner_adjoint"
                | "compose"
                | "contract_identity"
                | "contract_input_swap"
                | "contract_input_output_swap"
                | "qr_compact"
                | "qr_compact_generic_layout"
                | "checked_compact_input"
                | "full_qr_lowering"
                | "lazy_tree_transform"
        ) {
            return Err(Box::new(Error::InvalidArgument(format!(
                "unknown OP_MATRIX_OPERATION `{operation}`"
            ))));
        }
    }
    if let Ok(form) = std::env::var("OP_MATRIX_FORM") {
        if !matches!(form.as_str(), "owned" | "destination") {
            return Err(Box::new(Error::InvalidArgument(format!(
                "unknown OP_MATRIX_FORM `{form}`"
            ))));
        }
    }
    #[cfg(not(feature = "racah-generated"))]
    if matches!(
        std::env::var("OP_MATRIX_OPERATION").as_deref(),
        Ok("qr_compact" | "lazy_tree_transform")
    ) {
        return Err(Box::new(Error::InvalidArgument(
            "the selected operation-matrix row requires the racah-generated feature".into(),
        )));
    }
    let min_ms = std::env::var("OP_MATRIX_MIN_MS")
        .ok()
        .map(|value| value.parse().expect("OP_MATRIX_MIN_MS must be an integer"))
        .unwrap_or(20);
    let degeneracy = std::env::var("OP_MATRIX_DEGENERACY")
        .ok()
        .map(|value| {
            value
                .parse()
                .expect("OP_MATRIX_DEGENERACY must be an integer")
        })
        .unwrap_or(8);
    println!(
        "# tenet_authority={}",
        std::env::var("TENET_AUTHORITY").unwrap_or_else(|_| "unknown".into())
    );
    println!(
        "# tenferro_authority={}",
        std::env::var("TENFERRO_AUTHORITY").unwrap_or_else(|_| "unknown".into())
    );
    println!(
        "# features=cpu-faer:{} blas-provider:{} cuda:{} gemm_backend={}",
        cfg!(feature = "cpu-faer"),
        cfg!(any(
            feature = "cpu-blas",
            feature = "blas-accelerate",
            feature = "blas-openblas",
            feature = "blas-mkl"
        )),
        cfg!(feature = "cuda"),
        std::env::var("OP_MATRIX_GEMM_BACKEND").unwrap_or_else(|_| "faer".into())
    );
    println!("# degeneracy={degeneracy}");
    println!("# threads=RAYON_NUM_THREADS:{} OPENBLAS_NUM_THREADS:{} OMP_NUM_THREADS:{} MKL_NUM_THREADS:{}", env_or_unset("RAYON_NUM_THREADS"), env_or_unset("OPENBLAS_NUM_THREADS"), env_or_unset("OMP_NUM_THREADS"), env_or_unset("MKL_NUM_THREADS"));
    println!("# cold_scope=fresh Runtime tree-transform store; process-global interned structures may already be warm");
    println!(
        "# cache_mode={}",
        std::env::var("OP_MATRIX_CACHE").unwrap_or_else(|_| "enabled".into())
    );
    println!("# allocation_scope=caller-thread Rust allocation calls and requested bytes during the measured phase; excludes worker threads, native BLAS allocation, frees, and peak/live bytes");
    println!("# unavailable_counters=exact_layout_admission,operation_local_scratch_bytes,provider_queries,transform_passes,gemm_calls,host_device_transfers");
    println!("symmetry,operation,form,phase,iterations,us_per_iter,tree_hits,tree_misses,tree_evictions,tree_bypasses,tree_entries_delta,tree_charged_payload_bytes_before,tree_charged_payload_bytes_after,tree_charged_payload_bytes_delta,fusion_layout_misses,fusion_layout_evictions,fusion_layout_bypasses,fusion_layout_entries_delta,fusion_layout_charged_payload_bytes_before,fusion_layout_charged_payload_bytes_after,fusion_layout_charged_payload_bytes_delta,complete_hom_hits,complete_hom_misses,complete_hom_admissions,complete_hom_evictions,complete_hom_bypasses,complete_hom_entries_delta,complete_hom_charged_bytes_before,complete_hom_charged_bytes_after,complete_hom_charged_bytes_delta,exact_layout_admission,caller_allocation_calls,caller_requested_allocation_bytes,operation_local_scratch_bytes,provider_queries,transform_passes,gemm_calls,host_device_transfers");

    let min_time = Duration::from_millis(min_ms);
    run_layout_generic_qr(degeneracy, min_time)?;
    run_checked_compact_input(degeneracy, min_time)?;
    run_full_qr_lowering(min_time)?;
    #[cfg(feature = "racah-generated")]
    run_lazy_tree_transform(min_time)?;
    run_provider!(
        "U1",
        U1FusionRule,
        GradedSpace::try_new(
            U1FusionRule,
            [
                (U1Irrep::new(-1), degeneracy),
                (U1Irrep::new(0), degeneracy),
                (U1Irrep::new(1), degeneracy),
            ]
        )?,
        min_time
    );
    run_provider!(
        "SU2",
        SU2FusionRule,
        GradedSpace::try_new(
            SU2FusionRule,
            [
                (SU2Irrep::from_twice_spin(0), degeneracy),
                (SU2Irrep::from_twice_spin(1), degeneracy),
                (SU2Irrep::from_twice_spin(2), degeneracy),
            ]
        )?,
        min_time
    );
    #[cfg(feature = "racah-generated")]
    {
        use std::sync::Arc;
        use tenet::typed::SUNFusionRule;

        run_checked_sun(
            "SU3[0;0]",
            Arc::new(SUNFusionRule::new(3)?),
            vec![0, 0],
            degeneracy,
            min_time,
            true,
        )?;
        run_checked_sun(
            "SU3[1;1]",
            Arc::new(SUNFusionRule::new(3)?),
            vec![1, 1],
            degeneracy,
            min_time,
            false,
        )?;
        run_checked_sun(
            "SU4[1;0;1]",
            Arc::new(SUNFusionRule::new(4)?),
            vec![1, 0, 1],
            degeneracy,
            min_time,
            false,
        )?;
    }
    Ok(())
}

fn env_or_unset(name: &str) -> String {
    std::env::var(name).unwrap_or_else(|_| "unset".into())
}
