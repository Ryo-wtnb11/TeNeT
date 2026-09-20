//! Revision-pinned device performance baseline for the existing CUDA path.
//!
//! Mirrors the Host protocol of `tenet/examples/operation_matrix.rs`: a fresh
//! `Runtime` per row, the fixture built before the timer, one cold call then a
//! warm-up then `--iterations` measured calls, correctness checked after the
//! timer, environment header lines, a CSV table, and caller-thread allocation
//! counters. It adds the device columns this leaf exists for: host/device
//! transfer calls and bytes, device buffer allocations, GEMM submissions and
//! cuSOLVER calls, read from `tenet::dense::cuda_transfer_stats`.
//!
//! This is a validation fixture. It never gates CI on wall clock, and no
//! measured value feeds a dispatch decision.
//!
//! Run through `benchmarks/cuda_operation_matrix.sh`.

use std::{
    alloc::{GlobalAlloc, Layout, System},
    cell::Cell,
};

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

#[cfg(not(feature = "cuda"))]
fn main() {
    eprintln!(
        "cuda_operation_matrix measures the CUDA device path and is unsupported without the \
         `cuda` feature; rebuild with `--no-default-features --features cuda,cpu-faer`"
    );
    std::process::exit(2);
}

#[cfg(feature = "cuda")]
fn main() {
    device::run();
}

#[cfg(feature = "cuda")]
mod device {
    use super::{ALLOCATION_CALLS, COUNTING, REQUESTED_BYTES};
    use std::{hint::black_box, sync::Arc, time::Instant};

    use tenet::core::{
        product_sector, CheckedFusionAlgebra, FermionParityFusionRule, FusionAlgebraError,
        MultiplicityFreeAdmissionMode, MultiplicityFreeRigidSymbols, ProductFusionRuleExt,
        SU2FusionRule, SU2Irrep, SectorCodec, TypedSectorAdmission, U1FusionRule, U1Irrep, Z2Irrep,
    };
    use tenet::dense::{cuda_transfer_stats, CudaTransferStats};
    use tenet::prelude::Complex64;
    use tenet::typed::{CudaStorage, GradedSpace, Runtime, TensorMap, Truncation};
    use tenet_network::tensor;

    /// Fixture families. The parameters are explicit CLI inputs; nothing in
    /// TeNeT or in this harness dispatches on them.
    const FAMILIES: [&str; 2] = ["many-small", "few-large"];

    pub(super) struct Config {
        /// Coupled block count per family, in `FAMILIES` order.
        pub blocks: [usize; 2],
        /// Per-sector degeneracy per family, in `FAMILIES` order.
        pub degeneracy: [usize; 2],
        pub iterations: usize,
        pub warmup: usize,
        pub device: usize,
    }

    fn parse_pair(name: &str, raw: &str) -> [usize; 2] {
        let mut parts = raw.split(',');
        let first = parts.next().unwrap_or_default();
        let second = parts.next().unwrap_or(first);
        if parts.next().is_some() {
            panic!("{name} takes at most two comma-separated values (many-small,few-large)");
        }
        [
            first
                .trim()
                .parse()
                .unwrap_or_else(|_| panic!("{name} many-small value must be an integer")),
            second
                .trim()
                .parse()
                .unwrap_or_else(|_| panic!("{name} few-large value must be an integer")),
        ]
    }

    fn parse_config() -> Config {
        let mut config = Config {
            blocks: [64, 4],
            degeneracy: [4, 64],
            iterations: 20,
            warmup: 3,
            device: 0,
        };
        let mut args = std::env::args().skip(1);
        while let Some(flag) = args.next() {
            let value = args
                .next()
                .unwrap_or_else(|| panic!("{flag} requires a value"));
            match flag.as_str() {
                "--blocks" => config.blocks = parse_pair("--blocks", &value),
                "--degeneracy" => config.degeneracy = parse_pair("--degeneracy", &value),
                "--iterations" => {
                    config.iterations = value.parse().expect("--iterations must be an integer")
                }
                "--warmup" => config.warmup = value.parse().expect("--warmup must be an integer"),
                "--device" => config.device = value.parse().expect("--device must be an integer"),
                other => panic!(
                    "unknown flag `{other}`; supported: --blocks, --degeneracy, --iterations, \
                     --warmup, --device"
                ),
            }
        }
        assert!(config.iterations > 0, "--iterations must be positive");
        config
    }

    /// Payload dtypes this baseline covers. Both are real device payloads;
    /// the fixture values differ only in carrying an imaginary part.
    pub(super) trait HarnessScalar:
        tenet::typed::FactorizationScalar + tenet::typed::CudaPayload
    {
        const NAME: &'static str;
        fn entry(real: f64, imaginary: f64) -> Self;
        fn distance(self, other: Self) -> f64;
        fn magnitude(self) -> f64;
    }

    impl HarnessScalar for f64 {
        const NAME: &'static str = "f64";
        fn entry(real: f64, _imaginary: f64) -> Self {
            real
        }
        fn distance(self, other: Self) -> f64 {
            (self - other).abs()
        }
        fn magnitude(self) -> f64 {
            self.abs()
        }
    }

    impl HarnessScalar for Complex64 {
        const NAME: &'static str = "c64";
        fn entry(real: f64, imaginary: f64) -> Self {
            Complex64::new(real, imaginary)
        }
        fn distance(self, other: Self) -> f64 {
            (self - other).norm()
        }
        fn magnitude(self) -> f64 {
            self.norm()
        }
    }

    #[derive(Clone, Copy, Default)]
    struct Sample {
        nanos: u64,
        alloc_calls: u64,
        alloc_bytes: u64,
        device: CudaTransferStats,
        barrier_d2h_calls: u64,
        barrier_d2h_bytes: u64,
    }

    fn stats_delta(after: CudaTransferStats, before: CudaTransferStats) -> CudaTransferStats {
        CudaTransferStats {
            h2d_calls: after.h2d_calls - before.h2d_calls,
            h2d_bytes: after.h2d_bytes - before.h2d_bytes,
            d2h_calls: after.d2h_calls - before.d2h_calls,
            d2h_bytes: after.d2h_bytes - before.d2h_bytes,
            device_allocs: after.device_allocs - before.device_allocs,
            gemm_calls: after.gemm_calls - before.gemm_calls,
            solver_calls: after.solver_calls - before.solver_calls,
            copy_calls: after.copy_calls - before.copy_calls,
        }
    }

    /// One measured call. Setup stays outside the timed region, the owned
    /// result is handed back rather than dropped inside it, and the completion
    /// barrier runs after the clock stops with its own device traffic recorded
    /// in separate columns.
    fn measure<T>(mut operation: impl FnMut() -> T, mut barrier: impl FnMut()) -> (T, Sample) {
        let before = cuda_transfer_stats();
        ALLOCATION_CALLS.set(0);
        REQUESTED_BYTES.set(0);
        COUNTING.set(true);
        let start = Instant::now();
        let output = operation();
        let nanos = start.elapsed().as_nanos() as u64;
        COUNTING.set(false);
        black_box(&output);
        let after = cuda_transfer_stats();
        barrier();
        let after_barrier = cuda_transfer_stats();
        (
            output,
            Sample {
                nanos,
                alloc_calls: ALLOCATION_CALLS.get() as u64,
                alloc_bytes: REQUESTED_BYTES.get() as u64,
                device: stats_delta(after, before),
                barrier_d2h_calls: after_barrier.d2h_calls - after.d2h_calls,
                barrier_d2h_bytes: after_barrier.d2h_bytes - after.d2h_bytes,
            },
        )
    }

    fn median(values: &mut [u64]) -> u64 {
        values.sort_unstable();
        values[values.len() / 2]
    }

    fn summarize(samples: &[Sample]) -> (Sample, u64) {
        let field = |get: fn(&Sample) -> u64| {
            let mut values: Vec<u64> = samples.iter().map(get).collect();
            median(&mut values)
        };
        let minimum = samples.iter().map(|sample| sample.nanos).min().unwrap_or(0);
        (
            Sample {
                nanos: field(|sample| sample.nanos),
                alloc_calls: field(|sample| sample.alloc_calls),
                alloc_bytes: field(|sample| sample.alloc_bytes),
                device: CudaTransferStats {
                    h2d_calls: field(|sample| sample.device.h2d_calls),
                    h2d_bytes: field(|sample| sample.device.h2d_bytes),
                    d2h_calls: field(|sample| sample.device.d2h_calls),
                    d2h_bytes: field(|sample| sample.device.d2h_bytes),
                    device_allocs: field(|sample| sample.device.device_allocs),
                    gemm_calls: field(|sample| sample.device.gemm_calls),
                    solver_calls: field(|sample| sample.device.solver_calls),
                    copy_calls: field(|sample| sample.device.copy_calls),
                },
                barrier_d2h_calls: field(|sample| sample.barrier_d2h_calls),
                barrier_d2h_bytes: field(|sample| sample.barrier_d2h_bytes),
            },
            minimum,
        )
    }

    #[derive(Clone, Copy)]
    struct Labels<'a> {
        provider: &'a str,
        dtype: &'a str,
        family: &'a str,
        blocks: usize,
        degeneracy: usize,
        operation: &'a str,
    }

    /// One printable phase of a row. Rows are buffered rather than printed as
    /// they are measured, because the `check` column is only known once both
    /// phases have run.
    struct PhaseRow {
        phase: &'static str,
        iterations: usize,
        sample: Sample,
        minimum: u64,
    }

    /// Host arms cannot fail: their `Result` exists only so both arms share one
    /// `bench` signature.
    type Never = std::convert::Infallible;

    #[allow(clippy::too_many_arguments)]
    fn print_row(
        labels: Labels<'_>,
        target: &str,
        phase: &str,
        iterations: usize,
        sample: &Sample,
        minimum: u64,
        check: &str,
    ) {
        println!(
            "{provider},{dtype},{family},{blocks},{degeneracy},{operation},{target},{phase},\
             {iterations},{median},{minimum},{alloc_calls},{alloc_bytes},{h2d_calls},{h2d_bytes},\
             {d2h_calls},{d2h_bytes},{device_allocs},{gemm_calls},{solver_calls},{copy_calls},\
             {barrier_calls},{barrier_bytes},{check}",
            provider = labels.provider,
            dtype = labels.dtype,
            family = labels.family,
            blocks = labels.blocks,
            degeneracy = labels.degeneracy,
            operation = labels.operation,
            median = sample.nanos,
            alloc_calls = sample.alloc_calls,
            alloc_bytes = sample.alloc_bytes,
            h2d_calls = sample.device.h2d_calls,
            h2d_bytes = sample.device.h2d_bytes,
            d2h_calls = sample.device.d2h_calls,
            d2h_bytes = sample.device.d2h_bytes,
            device_allocs = sample.device.device_allocs,
            gemm_calls = sample.device.gemm_calls,
            solver_calls = sample.device.solver_calls,
            copy_calls = sample.device.copy_calls,
            barrier_calls = sample.barrier_d2h_calls,
            barrier_bytes = sample.barrier_d2h_bytes,
        );
    }

    fn print_rows(labels: Labels<'_>, target: &str, rows: &[PhaseRow], check: &str) {
        for row in rows {
            print_row(
                labels,
                target,
                row.phase,
                row.iterations,
                &row.sample,
                row.minimum,
                check,
            );
        }
    }

    fn print_skip(labels: Labels<'_>, target: &str, reason: &str) {
        print_row(
            labels,
            target,
            "skipped",
            0,
            &Sample::default(),
            0,
            &sanitize(reason),
        );
    }

    fn sanitize(text: &str) -> String {
        text.replace([',', '\n', '\r'], "; ")
    }

    /// Runs the first call, the warm-up, and the measured warm iterations of
    /// one operation, and hands back the first call's output together with the
    /// unprinted phase rows. Nothing is validated here: the caller checks the
    /// returned output after both phases have been measured.
    fn bench<T, E: std::fmt::Display>(
        config: &Config,
        first_phase: &'static str,
        mut operation: impl FnMut() -> Result<T, E>,
        mut barrier: impl FnMut(),
    ) -> Result<(T, [PhaseRow; 2]), String> {
        let (first, cold) = measure(&mut operation, &mut barrier);
        let first = first.map_err(|error| sanitize(&error.to_string()))?;
        for _ in 0..config.warmup {
            black_box(operation().map_err(|error| sanitize(&error.to_string()))?);
        }
        let mut samples = Vec::with_capacity(config.iterations);
        for _ in 0..config.iterations {
            let (output, sample) = measure(&mut operation, &mut barrier);
            drop(output.map_err(|error| sanitize(&error.to_string()))?);
            samples.push(sample);
        }
        let (warm, minimum) = summarize(&samples);
        Ok((
            first,
            [
                PhaseRow {
                    phase: first_phase,
                    iterations: 1,
                    sample: cold,
                    minimum: cold.nanos,
                },
                PhaseRow {
                    phase: "warm",
                    iterations: config.iterations,
                    sample: warm,
                    minimum,
                },
            ],
        ))
    }

    fn payload_close<D: HarnessScalar + Copy>(
        actual: &[D],
        expected: &[D],
        tolerance: f64,
    ) -> bool {
        actual.len() == expected.len()
            && actual.iter().zip(expected).all(|(&actual, &expected)| {
                actual.distance(expected) <= tolerance * (1.0 + expected.magnitude())
            })
    }

    fn verdict(ok: bool, name: &str) -> String {
        if ok {
            format!("ok:{name}")
        } else {
            format!("MISMATCH:{name}")
        }
    }

    pub(super) fn run() {
        let config = parse_config();
        print_header(&config);
        run_all(&config);
    }

    fn print_header(config: &Config) {
        println!(
            "# tenet_authority={}",
            std::env::var("TENET_AUTHORITY").unwrap_or_else(|_| "unknown".into())
        );
        println!(
            "# tenferro_authority={}",
            std::env::var("TENFERRO_AUTHORITY").unwrap_or_else(|_| "unknown".into())
        );
        println!(
            "# cuda_env TENFERRO_CUTENSOR_PATH={} CUDA_VISIBLE_DEVICES={} device_ordinal={}",
            std::env::var("TENFERRO_CUTENSOR_PATH").unwrap_or_else(|_| "unset".into()),
            std::env::var("CUDA_VISIBLE_DEVICES").unwrap_or_else(|_| "unset".into()),
            config.device
        );
        println!(
            "# fixtures families={FAMILIES:?} blocks={:?} degeneracy={:?} rank=2 \
             shape=endomorphism[V;V]",
            config.blocks, config.degeneracy
        );
        println!(
            "# protocol fresh_Runtime_per_row fixture_before_timer first_call warmup={} \
             warm_iterations={} statistic=per_iteration_median(+ns_min) \
             correctness_checked_after_both_timed_phases",
            config.warmup, config.iterations
        );
        println!(
            "# cold_scope=cold is the first call of the measured operation on a fresh Runtime; \
             the fixture's own upload has already run, so process-global CUDA state and \
             interned space structures may be warm. Rows whose measured operation is itself \
             performed during fixture setup report first_after_setup instead of cold"
        );
        println!(
            "# sampling=one process per invocation; unlike the Host operation_matrix wrapper \
             there is no three-process median, so a row carries this process's per-iteration \
             median only"
        );
        println!(
            "# allocation_scope=caller-thread Rust allocation calls and requested bytes during \
             the timed region; excludes worker threads, frees, and device memory"
        );
        println!(
            "# device_counter_scope=tenet_dense::cuda_transfer_stats deltas over the timed \
             region; device_allocs counts uploads plus tenferro tensors wrapped as \
             CudaDenseStorage and excludes tenferro-internal solver workspaces"
        );
        println!(
            "# peak_device_memory=NA: neither TeNeT nor tenferro exposes a device allocation \
             high-water mark, and a per-row nvidia-smi sample would observe the whole device \
             rather than this process's buffers"
        );
        println!(
            "# synchronization=NA: no device synchronization is reachable through TeNeT's \
             public API, and tenferro's with_cubecl success path does not synchronize, so a \
             timed region measures submission unless the operation itself downloads. The \
             completion barrier after each timed call is norm() on one fixture input, which \
             submits one GEMM per coupled block and downloads the per-block partials; only its \
             downloads are recorded, in the barrier_* columns, and none of its traffic is \
             included in the phase columns"
        );
        // #1278 moved the tenferro backend library initialization
        // (cuTENSOR, cuSOLVER/cuBLAS handles) from the first submission to
        // Runtime construction, so the cost that used to sit in every `cold`
        // cell is reported here instead of disappearing. `first` is this
        // process's first CUDA Runtime and also pays CUDA context creation;
        // `second` is a second fresh Runtime in the same process, where the
        // context is warm but the backend instance, and therefore the
        // warm-up, is new. Both are one observation, not a median.
        let build_start = Instant::now();
        let first_runtime = Runtime::builder()
            .cuda(config.device)
            .dense_threads(1)
            .build()
            .expect("CUDA Runtime for the construction measurement");
        let first_build_ns = build_start.elapsed().as_nanos();
        drop(first_runtime);
        let build_start = Instant::now();
        let second_runtime = Runtime::builder()
            .cuda(config.device)
            .dense_threads(1)
            .build()
            .expect("CUDA Runtime for the construction measurement");
        let second_build_ns = build_start.elapsed().as_nanos();
        drop(second_runtime);
        println!(
            "# runtime_build_ns first={first_build_ns} second={second_build_ns} \
             scope=one Runtime::builder().cuda(device).dense_threads(1).build() each, one \
             observation, first includes CUDA context creation, both include the #1278 \
             backend warm-up"
        );
        println!(
            "# tensorkit_cuda_comparison=absent: neither TensorKit nor QSpace has a matched \
             CUDA fixture for these rows at this revision"
        );
        println!(
            "provider,dtype,family,blocks,degeneracy,operation,target,phase,iterations,\
             ns_median,ns_min,alloc_calls,alloc_bytes,h2d_calls,h2d_bytes,d2h_calls,d2h_bytes,\
             device_allocs,gemm_calls,solver_calls,copy_calls,barrier_d2h_calls,\
             barrier_d2h_bytes,check"
        );
    }

    struct Fixture<R, D>
    where
        R: SectorCodec,
        D: HarnessScalar,
    {
        /// Kept so the row's tensors outlive their Runtime borrow; the field
        /// itself is never read.
        #[allow(dead_code)]
        runtime: Runtime,
        host: Vec<TensorMap<R, D>>,
        device: Vec<TensorMap<R, D, CudaStorage<D>>>,
        hermitian_host: Option<TensorMap<R, D>>,
        hermitian_device: Option<TensorMap<R, D, CudaStorage<D>>>,
    }

    /// Builds one row's Runtime, Host fixture, and device upload. Everything
    /// here happens before any timer starts.
    fn fixture<R, D>(config: &Config, space: &GradedSpace<R>, count: usize) -> Fixture<R, D>
    where
        R: TypedSectorAdmission<Error = FusionAlgebraError, Mode = MultiplicityFreeAdmissionMode>
            + MultiplicityFreeRigidSymbols<Scalar = f64>
            + CheckedFusionAlgebra
            + SectorCodec
            + Send
            + Sync
            + 'static,
        D: HarnessScalar,
    {
        let runtime = Runtime::builder()
            .cuda(config.device)
            .dense_threads(1)
            .build()
            .expect("CUDA Runtime for the measured row");
        let host: Vec<TensorMap<R, D>> = (0..count)
            .map(|index| {
                let shift = index as f64;
                TensorMap::from_block_fn(&runtime, [space], [space], move |_, indices| {
                    let row = indices[0] as f64;
                    let column = indices[1] as f64;
                    D::entry(
                        1.0 + shift + 0.5 * row + 0.25 * column,
                        0.125 * row - 0.0625 * column - 0.03125 * shift,
                    )
                })
                .expect("Host fixture tensor")
            })
            .collect();
        let device = host
            .iter()
            .map(|tensor| tensor.to_cuda().expect("fixture upload"))
            .collect();
        let hermitian_host =
            TensorMap::<R, D>::from_block_fn(&runtime, [space], [space], |_, indices| {
                let (row, column) = (indices[0], indices[1]);
                if row == column {
                    D::entry(row as f64 + 1.0, 0.0)
                } else {
                    let magnitude = 0.25 / (1.0 + row.abs_diff(column) as f64);
                    D::entry(magnitude, if row < column { 0.125 } else { -0.125 })
                }
            })
            .ok();
        let hermitian_device = hermitian_host
            .as_ref()
            .and_then(|tensor| tensor.to_cuda().ok());
        Fixture {
            runtime,
            host,
            device,
            hermitian_host,
            hermitian_device,
        }
    }

    /// Prints both arms of a row whose device probe failed, so an unsupported
    /// boundary stays visible in the table instead of vanishing.
    fn skip_row(labels: Labels<'_>, reason: &str) {
        print_skip(labels, "cuda", reason);
        print_skip(labels, "host", reason);
    }

    fn run_dtype<R, D>(
        config: &Config,
        provider: &str,
        family: &str,
        blocks: usize,
        degeneracy: usize,
        space: &GradedSpace<R>,
    ) where
        R: TypedSectorAdmission<Error = FusionAlgebraError, Mode = MultiplicityFreeAdmissionMode>
            + MultiplicityFreeRigidSymbols<Scalar = f64>
            + CheckedFusionAlgebra
            + SectorCodec
            + Send
            + Sync
            + 'static,
        D: HarnessScalar,
    {
        let label = |operation: &'static str| Labels {
            provider,
            dtype: D::NAME,
            family,
            blocks,
            degeneracy,
            operation,
        };
        let tolerance = 1.0e-9;
        let alpha = D::entry(1.5, 0.25);
        let beta = D::entry(-0.5, 0.125);
        let factor = D::entry(2.0, -0.5);

        // transfer roundtrip; the measured operations also run during fixture
        // setup, so their first phase is reported as first_after_setup
        {
            let fixture = fixture::<R, D>(config, space, 1);
            let source = &fixture.host[0];
            let uploaded = &fixture.device[0];
            let barrier = || {
                let _ = uploaded.norm();
            };
            match bench(config, "first_after_setup", || source.to_cuda(), barrier) {
                Err(reason) => skip_row(label("to_cuda"), &reason),
                Ok((first, rows)) => {
                    let check = verdict(
                        payload_close(
                            first.to_host().expect("roundtrip download").data(),
                            source.data(),
                            0.0,
                        ),
                        "roundtrip_bit_equal",
                    );
                    print_rows(label("to_cuda"), "cuda", &rows, &check);
                    print_skip(label("to_cuda"), "host", "no Host counterpart");
                }
            }
            match bench(config, "first_after_setup", || uploaded.to_host(), barrier) {
                Err(reason) => skip_row(label("to_host"), &reason),
                Ok((first, rows)) => {
                    let check = verdict(
                        payload_close(first.data(), source.data(), 0.0),
                        "roundtrip_bit_equal",
                    );
                    print_rows(label("to_host"), "cuda", &rows, &check);
                    print_skip(label("to_host"), "host", "no Host counterpart");
                }
            }
        }

        // contract, canonical direct
        {
            let fixture = fixture::<R, D>(config, space, 2);
            let (lhs, rhs) = (&fixture.host[0], &fixture.host[1]);
            let (lhs_device, rhs_device) = (&fixture.device[0], &fixture.device[1]);
            let barrier = || {
                let _ = lhs_device.norm();
            };
            match bench(
                config,
                "cold",
                || lhs_device.contract(rhs_device, &[1], &[0], &[0, 1]),
                barrier,
            ) {
                Err(reason) => skip_row(label("contract_direct"), &reason),
                Ok((device_first, device_rows)) => {
                    let (host_first, host_rows) = bench(
                        config,
                        "cold",
                        || {
                            Ok::<_, Never>(
                                lhs.contract(rhs, &[1], &[0], &[0, 1])
                                    .expect("Host contract"),
                            )
                        },
                        || {},
                    )
                    .expect("Host contract arm");
                    let check = verdict(
                        payload_close(
                            device_first.to_host().expect("download").data(),
                            host_first.data(),
                            tolerance,
                        ),
                        "host_value_equality",
                    );
                    print_rows(label("contract_direct"), "cuda", &device_rows, &check);
                    print_rows(label("contract_direct"), "host", &host_rows, &check);
                }
            }
        }

        // contract with a lazy-adjoint left operand; the lazy views are built
        // once as operands, outside every timed region
        {
            let fixture = fixture::<R, D>(config, space, 2);
            let (lhs, rhs) = (&fixture.host[0], &fixture.host[1]);
            let (lhs_device, rhs_device) = (&fixture.device[0], &fixture.device[1]);
            let barrier = || {
                let _ = lhs_device.norm();
            };
            match lhs_device.adjoint() {
                Err(error) => skip_row(label("contract_lazy_adjoint_lhs"), &error.to_string()),
                Ok(lhs_adjoint) => {
                    let host_adjoint = lhs.adjoint().expect("Host adjoint");
                    match bench(
                        config,
                        "cold",
                        || lhs_adjoint.contract(rhs_device, &[1], &[0], &[0, 1]),
                        barrier,
                    ) {
                        Err(reason) => skip_row(label("contract_lazy_adjoint_lhs"), &reason),
                        Ok((device_first, device_rows)) => {
                            let (host_first, host_rows) = bench(
                                config,
                                "cold",
                                || {
                                    Ok::<_, Never>(
                                        host_adjoint
                                            .contract(rhs, &[1], &[0], &[0, 1])
                                            .expect("Host adjoint contract"),
                                    )
                                },
                                || {},
                            )
                            .expect("Host adjoint contract arm");
                            let check = verdict(
                                payload_close(
                                    device_first.to_host().expect("download").data(),
                                    host_first.data(),
                                    tolerance,
                                ),
                                "host_value_equality",
                            );
                            print_rows(
                                label("contract_lazy_adjoint_lhs"),
                                "cuda",
                                &device_rows,
                                &check,
                            );
                            print_rows(
                                label("contract_lazy_adjoint_lhs"),
                                "host",
                                &host_rows,
                                &check,
                            );
                        }
                    }
                }
            }
        }

        // compose
        {
            let fixture = fixture::<R, D>(config, space, 2);
            let (lhs, rhs) = (&fixture.host[0], &fixture.host[1]);
            let (lhs_device, rhs_device) = (&fixture.device[0], &fixture.device[1]);
            let barrier = || {
                let _ = lhs_device.norm();
            };
            match bench(config, "cold", || lhs_device.compose(rhs_device), barrier) {
                Err(reason) => skip_row(label("compose"), &reason),
                Ok((device_first, device_rows)) => {
                    let (host_first, host_rows) = bench(
                        config,
                        "cold",
                        || Ok::<_, Never>(lhs.compose(rhs).expect("Host compose")),
                        || {},
                    )
                    .expect("Host compose arm");
                    let check = verdict(
                        payload_close(
                            device_first.to_host().expect("download").data(),
                            host_first.data(),
                            tolerance,
                        ),
                        "host_value_equality",
                    );
                    print_rows(label("compose"), "cuda", &device_rows, &check);
                    print_rows(label("compose"), "host", &host_rows, &check);
                }
            }
        }

        // scale
        {
            let fixture = fixture::<R, D>(config, space, 1);
            let source = &fixture.host[0];
            let source_device = &fixture.device[0];
            let barrier = || {
                let _ = source_device.norm();
            };
            match bench(config, "cold", || source_device.scale(factor), barrier) {
                Err(reason) => skip_row(label("scale"), &reason),
                Ok((device_first, device_rows)) => {
                    let (host_first, host_rows) = bench(
                        config,
                        "cold",
                        || Ok::<_, Never>(source.scale(factor)),
                        || {},
                    )
                    .expect("Host scale arm");
                    let check = verdict(
                        payload_close(
                            device_first.to_host().expect("download").data(),
                            host_first.data(),
                            tolerance,
                        ),
                        "host_value_equality",
                    );
                    print_rows(label("scale"), "cuda", &device_rows, &check);
                    print_rows(label("scale"), "host", &host_rows, &check);
                }
            }
        }

        // add, owned operands
        {
            let fixture = fixture::<R, D>(config, space, 2);
            let (lhs, rhs) = (&fixture.host[0], &fixture.host[1]);
            let (lhs_device, rhs_device) = (&fixture.device[0], &fixture.device[1]);
            let barrier = || {
                let _ = lhs_device.norm();
            };
            match bench(
                config,
                "cold",
                || lhs_device.add(rhs_device, alpha, beta),
                barrier,
            ) {
                Err(reason) => skip_row(label("add_owned"), &reason),
                Ok((device_first, device_rows)) => {
                    let (host_first, host_rows) = bench(
                        config,
                        "cold",
                        || Ok::<_, Never>(lhs.add(rhs, alpha, beta).expect("Host add")),
                        || {},
                    )
                    .expect("Host add arm");
                    let check = verdict(
                        payload_close(
                            device_first.to_host().expect("download").data(),
                            host_first.data(),
                            tolerance,
                        ),
                        "host_value_equality",
                    );
                    print_rows(label("add_owned"), "cuda", &device_rows, &check);
                    print_rows(label("add_owned"), "host", &host_rows, &check);
                }
            }
        }

        // add, lazy-adjoint fold on both operands
        {
            let fixture = fixture::<R, D>(config, space, 2);
            let (lhs, rhs) = (&fixture.host[0], &fixture.host[1]);
            let (lhs_device, rhs_device) = (&fixture.device[0], &fixture.device[1]);
            let barrier = || {
                let _ = lhs_device.norm();
            };
            match (lhs_device.adjoint(), rhs_device.adjoint()) {
                (Err(error), _) | (_, Err(error)) => {
                    skip_row(label("add_lazy_fold"), &error.to_string())
                }
                (Ok(lhs_adjoint), Ok(rhs_adjoint)) => {
                    let host_lhs = lhs.adjoint().expect("Host adjoint");
                    let host_rhs = rhs.adjoint().expect("Host adjoint");
                    match bench(
                        config,
                        "cold",
                        || lhs_adjoint.add(&rhs_adjoint, alpha, beta),
                        barrier,
                    ) {
                        Err(reason) => skip_row(label("add_lazy_fold"), &reason),
                        Ok((device_first, device_rows)) => {
                            let (host_first, host_rows) = bench(
                                config,
                                "cold",
                                || {
                                    Ok::<_, Never>(
                                        host_lhs
                                            .add(&host_rhs, alpha, beta)
                                            .expect("Host lazy add"),
                                    )
                                },
                                || {},
                            )
                            .expect("Host lazy add arm");
                            let check = verdict(
                                payload_close(
                                    device_first.to_host().expect("download").data(),
                                    host_first.data(),
                                    tolerance,
                                ),
                                "host_value_equality",
                            );
                            print_rows(label("add_lazy_fold"), "cuda", &device_rows, &check);
                            print_rows(label("add_lazy_fold"), "host", &host_rows, &check);
                        }
                    }
                }
            }
        }

        // norm
        {
            let fixture = fixture::<R, D>(config, space, 1);
            let source = &fixture.host[0];
            let source_device = &fixture.device[0];
            match bench(config, "cold", || source_device.norm(), || {}) {
                Err(reason) => skip_row(label("norm"), &reason),
                Ok((device_first, device_rows)) => {
                    let (host_first, host_rows) = bench(
                        config,
                        "cold",
                        || Ok::<_, Never>(source.norm().expect("Host norm")),
                        || {},
                    )
                    .expect("Host norm arm");
                    let check = verdict(
                        (device_first - host_first).abs() <= tolerance * (1.0 + host_first.abs()),
                        "host_scalar_equality",
                    );
                    print_rows(label("norm"), "cuda", &device_rows, &check);
                    print_rows(label("norm"), "host", &host_rows, &check);
                }
            }
        }

        // inner
        {
            let fixture = fixture::<R, D>(config, space, 2);
            let (lhs, rhs) = (&fixture.host[0], &fixture.host[1]);
            let (lhs_device, rhs_device) = (&fixture.device[0], &fixture.device[1]);
            match bench(config, "cold", || lhs_device.inner(rhs_device), || {}) {
                Err(reason) => skip_row(label("inner"), &reason),
                Ok((device_first, device_rows)) => {
                    let (host_first, host_rows) = bench(
                        config,
                        "cold",
                        || Ok::<_, Never>(lhs.inner(rhs).expect("Host inner")),
                        || {},
                    )
                    .expect("Host inner arm");
                    let check = verdict(
                        device_first.distance(host_first)
                            <= tolerance * (1.0 + host_first.magnitude()),
                        "host_scalar_equality",
                    );
                    print_rows(label("inner"), "cuda", &device_rows, &check);
                    print_rows(label("inner"), "host", &host_rows, &check);
                }
            }
        }

        // svd_compact
        {
            let fixture = fixture::<R, D>(config, space, 1);
            let source = &fixture.host[0];
            let source_device = &fixture.device[0];
            let barrier = || {
                let _ = source_device.norm();
            };
            match bench(config, "cold", || source_device.svd_compact(), barrier) {
                Err(reason) => skip_row(label("svd_compact"), &reason),
                Ok(((u, s, vh), device_rows)) => {
                    let (_, host_rows) = bench(
                        config,
                        "cold",
                        || Ok::<_, Never>(source.svd_compact().expect("Host svd_compact")),
                        || {},
                    )
                    .expect("Host svd_compact arm");
                    let rebuilt = u
                        .compose(&s)
                        .and_then(|left| left.compose(&vh))
                        .expect("device SVD reconstruction")
                        .to_host()
                        .expect("download");
                    let check = verdict(
                        payload_close(rebuilt.data(), source.data(), tolerance),
                        "device_reconstruction",
                    );
                    print_rows(label("svd_compact"), "cuda", &device_rows, &check);
                    print_rows(label("svd_compact"), "host", &host_rows, &check);
                }
            }
        }

        // svd_trunc, one policy
        {
            let fixture = fixture::<R, D>(config, space, 1);
            let source = &fixture.host[0];
            let source_device = &fixture.device[0];
            let truncation = Truncation::rank(degeneracy.max(1));
            let barrier = || {
                let _ = source_device.norm();
            };
            match bench(
                config,
                "cold",
                || source_device.svd_trunc(&truncation),
                barrier,
            ) {
                Err(reason) => skip_row(label("svd_trunc_rank"), &reason),
                Ok((device_first, device_rows)) => {
                    let (host_first, host_rows) = bench(
                        config,
                        "cold",
                        || Ok::<_, Never>(source.svd_trunc(&truncation).expect("Host svd_trunc")),
                        || {},
                    )
                    .expect("Host svd_trunc arm");
                    let kept_match = device_first.singular_values.len()
                        == host_first.singular_values.len()
                        && device_first
                            .singular_values
                            .iter()
                            .zip(&host_first.singular_values)
                            .all(|(actual, expected)| {
                                actual.values.len() == expected.values.len()
                                    && actual.values.iter().zip(&expected.values).all(
                                        |(actual, expected)| {
                                            (actual - expected).abs()
                                                <= tolerance * (1.0 + expected.abs())
                                        },
                                    )
                            });
                    let check = verdict(
                        kept_match
                            && (device_first.error - host_first.error).abs()
                                <= tolerance * (1.0 + host_first.error.abs()),
                        "host_spectrum_and_discard_weight",
                    );
                    print_rows(label("svd_trunc_rank"), "cuda", &device_rows, &check);
                    print_rows(label("svd_trunc_rank"), "host", &host_rows, &check);
                }
            }
        }

        // eigh_full, on the Hermitian fixture
        {
            let fixture = fixture::<R, D>(config, space, 1);
            match (&fixture.hermitian_host, &fixture.hermitian_device) {
                (Some(source), Some(source_device)) => {
                    let barrier = || {
                        let _ = source_device.norm();
                    };
                    match bench(config, "cold", || source_device.eigh_full(), barrier) {
                        Err(reason) => skip_row(label("eigh_full"), &reason),
                        Ok(((d, v), device_rows)) => {
                            let (_, host_rows) = bench(
                                config,
                                "cold",
                                || Ok::<_, Never>(source.eigh_full().expect("Host eigh_full")),
                                || {},
                            )
                            .expect("Host eigh_full arm");
                            let rebuilt = v
                                .compose(&d)
                                .and_then(|left| {
                                    v.adjoint().and_then(|adjoint| left.compose(&adjoint))
                                })
                                .expect("device EIGH reconstruction")
                                .to_host()
                                .expect("download");
                            let check = verdict(
                                payload_close(rebuilt.data(), source.data(), tolerance),
                                "device_reconstruction",
                            );
                            print_rows(label("eigh_full"), "cuda", &device_rows, &check);
                            print_rows(label("eigh_full"), "host", &host_rows, &check);
                        }
                    }
                }
                _ => skip_row(label("eigh_full"), "Hermitian fixture unavailable"),
            }
        }

        // canonical three-tensor `tensor!` chain
        {
            let fixture = fixture::<R, D>(config, space, 3);
            let (a, b, c) = (&fixture.host[0], &fixture.host[1], &fixture.host[2]);
            let (ad, bd, cd) = (&fixture.device[0], &fixture.device[1], &fixture.device[2]);
            let barrier = || {
                let _ = ad.norm();
            };
            match bench(
                config,
                "cold",
                || tensor!([p; s] = ad[p; q] * bd[q; r] * cd[r; s]),
                barrier,
            ) {
                Err(reason) => skip_row(label("network_chain3"), &reason),
                Ok((device_first, device_rows)) => {
                    let (host_first, host_rows) = bench(
                        config,
                        "cold",
                        || {
                            Ok::<_, Never>(
                                tensor!([p; s] = a[p; q] * b[q; r] * c[r; s]).expect("Host chain"),
                            )
                        },
                        || {},
                    )
                    .expect("Host chain arm");
                    let check = verdict(
                        payload_close(
                            device_first.to_host().expect("download").data(),
                            host_first.data(),
                            tolerance,
                        ),
                        "host_value_equality",
                    );
                    print_rows(label("network_chain3"), "cuda", &device_rows, &check);
                    print_rows(label("network_chain3"), "host", &host_rows, &check);
                }
            }
        }
    }

    /// `qr_compact` has an `f64`-only device payload, so it is its own row set.
    fn run_qr<R>(
        config: &Config,
        provider: &str,
        family: &str,
        blocks: usize,
        degeneracy: usize,
        space: &GradedSpace<R>,
    ) where
        R: TypedSectorAdmission<Error = FusionAlgebraError, Mode = MultiplicityFreeAdmissionMode>
            + MultiplicityFreeRigidSymbols<Scalar = f64>
            + CheckedFusionAlgebra
            + SectorCodec
            + Send
            + Sync
            + 'static,
    {
        let labels = Labels {
            provider,
            dtype: <f64 as HarnessScalar>::NAME,
            family,
            blocks,
            degeneracy,
            operation: "qr_compact",
        };
        let fixture = fixture::<R, f64>(config, space, 1);
        let source = &fixture.host[0];
        let source_device = &fixture.device[0];
        let barrier = || {
            let _ = source_device.norm();
        };
        match bench(config, "cold", || source_device.qr_compact(), barrier) {
            Err(reason) => skip_row(labels, &reason),
            Ok(((q, r), device_rows)) => {
                let (_, host_rows) = bench(
                    config,
                    "cold",
                    || Ok::<_, Never>(source.qr_compact().expect("Host qr_compact")),
                    || {},
                )
                .expect("Host qr_compact arm");
                let rebuilt = q
                    .compose(&r)
                    .expect("device QR reconstruction")
                    .to_host()
                    .expect("download");
                let check = verdict(
                    payload_close(rebuilt.data(), source.data(), 1.0e-9),
                    "device_reconstruction",
                );
                print_rows(labels, "cuda", &device_rows, &check);
                print_rows(labels, "host", &host_rows, &check);
            }
        }
    }

    macro_rules! provider_rows {
        ($config:expr, $name:literal, $build:expr) => {{
            for (index, family) in FAMILIES.iter().enumerate() {
                let blocks = $config.blocks[index];
                let degeneracy = $config.degeneracy[index];
                let space = ($build)(blocks, degeneracy);
                run_dtype::<_, f64>($config, $name, family, blocks, degeneracy, &space);
                run_qr::<_>($config, $name, family, blocks, degeneracy, &space);
                run_dtype::<_, Complex64>($config, $name, family, blocks, degeneracy, &space);
            }
        }};
    }

    fn run_all(config: &Config) {
        provider_rows!(config, "U1", |blocks: usize, degeneracy: usize| {
            GradedSpace::try_new_with_arc(
                Arc::new(U1FusionRule),
                (0..blocks).map(|index| (U1Irrep::new(index as i32), degeneracy)),
            )
            .expect("U(1) fixture leg")
        });
        provider_rows!(config, "fZ2", |_blocks: usize, degeneracy: usize| {
            // Fermion parity has exactly two irreps, so the realized block
            // count of this provider is 2 in both families. That is a provider
            // capability, not a size decision made by the harness.
            GradedSpace::try_new_with_arc(
                Arc::new(FermionParityFusionRule),
                [(Z2Irrep::EVEN, degeneracy), (Z2Irrep::ODD, degeneracy)],
            )
            .expect("fZ2 fixture leg")
        });
        provider_rows!(config, "SU2", |blocks: usize, degeneracy: usize| {
            GradedSpace::try_new_with_arc(
                Arc::new(SU2FusionRule),
                (0..blocks).map(|index| (SU2Irrep::from_twice_spin(index), degeneracy)),
            )
            .expect("SU(2) fixture leg")
        });
        provider_rows!(config, "U1xfZ2", |blocks: usize, degeneracy: usize| {
            GradedSpace::try_new_with_arc(
                Arc::new(FermionParityFusionRule.product(U1FusionRule)),
                (0..blocks).map(|index| {
                    (
                        product_sector(
                            if index % 2 == 0 {
                                Z2Irrep::EVEN
                            } else {
                                Z2Irrep::ODD
                            },
                            U1Irrep::new((index / 2) as i32),
                        ),
                        degeneracy,
                    )
                }),
            )
            .expect("fZ2 x U(1) fixture leg")
        });
    }
}
