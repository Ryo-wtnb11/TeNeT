//! `DenseExecutor::matmul_batch_real_rhs_into` (#1407): a real matrix acting
//! on complex operands is the same real matrix acting on the real and on the
//! imaginary component separately.
//!
//! Oracles, none of which reads the code under test:
//! - the same executor's real batch GEMM run on the separated components (the
//!   componentwise definition itself), by the non-finite class and within
//!   `docs/testing_numerics.md`, and bit for bit on the faer kernel;
//! - a naive per-entry sum, compared by the non-finite class and, for finite
//!   entries, within `docs/testing_numerics.md`;
//! - for finite data, the promoted complex GEMM the transform used before,
//!   within the same tolerance: the linear map is unchanged, only the
//!   arithmetic differs.
//!
//! The payloads mix `±inf`, NaN payloads, `-0` and subnormals into ordinary
//! values, where the promoted product `(u + 0i)(x + yi)` is not componentwise.

#![cfg(feature = "tenferro")]

use num_complex::{Complex, Complex32, Complex64};
use tenet_dense::{
    strided_batch_runs, DefaultDenseExecutor, DenseDotConfig, DenseError, DenseExecutor,
    DenseGemmBatchJob, DenseRead, DenseScalar, DenseTensor, DenseView, DenseViewMut, DenseWrite,
};

#[path = "../../tests/support/numerics.rs"]
mod numerics;

trait Lane: Copy + Default + std::fmt::Debug + PartialEq + 'static + numerics::Numeric
where
    Complex<Self>: numerics::Numeric,
{
    const SPECIAL: [Self; 6];
    const ONE: DenseScalar;
    const ZERO: DenseScalar;
    const COMPLEX_ONE: DenseScalar;
    const COMPLEX_ZERO: DenseScalar;
    fn from_f64(value: f64) -> Self;
    fn to_f64(self) -> f64;
    fn bits(self) -> u64;
    fn read(view: DenseView<'_, Self>) -> DenseRead<'_>;
    fn write(view: DenseViewMut<'_, Self>) -> DenseWrite<'_>;
    fn read_complex(view: DenseView<'_, Complex<Self>>) -> DenseRead<'_>;
    fn write_complex(view: DenseViewMut<'_, Complex<Self>>) -> DenseWrite<'_>;
}

macro_rules! lane {
    ($ty:ty, $variant:ident, $complex:ident, [$($special:expr),*]) => {
        impl Lane for $ty {
            const SPECIAL: [Self; 6] = [$(<$ty>::from_bits($special)),*];
            const ONE: DenseScalar = DenseScalar::$variant(1.0);
            const ZERO: DenseScalar = DenseScalar::$variant(0.0);
            const COMPLEX_ONE: DenseScalar = DenseScalar::$complex(Complex::new(1.0, 0.0));
            const COMPLEX_ZERO: DenseScalar = DenseScalar::$complex(Complex::new(0.0, 0.0));
            fn from_f64(value: f64) -> Self {
                value as $ty
            }
            fn to_f64(self) -> f64 {
                self.into()
            }
            fn bits(self) -> u64 {
                u64::from(self.to_bits())
            }
            fn read(view: DenseView<'_, Self>) -> DenseRead<'_> {
                DenseRead::$variant(view)
            }
            fn write(view: DenseViewMut<'_, Self>) -> DenseWrite<'_> {
                DenseWrite::$variant(view)
            }
            fn read_complex(view: DenseView<'_, Complex<Self>>) -> DenseRead<'_> {
                DenseRead::$complex(view)
            }
            fn write_complex(view: DenseViewMut<'_, Complex<Self>>) -> DenseWrite<'_> {
                DenseWrite::$complex(view)
            }
        }
    };
}

lane!(
    f64,
    F64,
    C64,
    [
        0x8000_0000_0000_0000, // -0
        0x7ff0_0000_0000_0000, // +inf
        0xfff0_0000_0000_0000, // -inf
        0x7ff8_0000_0000_1407, // NaN with a payload
        0x0000_0000_0000_0001, // smallest subnormal
        0x000f_ffff_ffff_ffff  // largest subnormal
    ]
);
lane!(
    f32,
    F32,
    C32,
    [
        0x8000_0000,
        0x7f80_0000,
        0xff80_0000,
        0x7fc0_1407,
        0x0000_0001,
        0x007f_ffff
    ]
);

/// Deterministic values; every fourth one is a special value when requested.
fn values<T: Lane>(len: usize, seed: u64, special: bool) -> Vec<T>
where
    Complex<T>: numerics::Numeric,
{
    let mut state = seed.wrapping_mul(0x9e37_79b9_7f4a_7c15) | 1;
    (0..len)
        .map(|index| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            if special && index % 4 == 1 {
                return T::SPECIAL[(state % 6) as usize];
            }
            let magnitude = 0.25 + (state >> 11) as f64 / (1u64 << 53) as f64;
            let sign = if state & 1 == 0 { 1.0 } else { -1.0 };
            T::from_f64(sign * magnitude * 2f64.powi(((state >> 1) % 8) as i32 - 4))
        })
        .collect()
}

struct Case {
    name: &'static str,
    jobs: Vec<DenseGemmBatchJob>,
    view_offset: usize,
}

fn job(
    dst: usize,
    lhs: usize,
    rhs: usize,
    rows: usize,
    k: usize,
    cols: usize,
) -> DenseGemmBatchJob {
    DenseGemmBatchJob {
        dst_offset: dst,
        lhs_offset: lhs,
        rhs_offset: rhs,
        rows,
        contracted: k,
        cols,
    }
}

fn cases() -> Vec<Case> {
    vec![
        Case {
            name: "single job",
            jobs: vec![job(0, 0, 0, 5, 3, 4)],
            view_offset: 0,
        },
        Case {
            name: "one-element job at a view offset",
            jobs: vec![job(2, 1, 3, 1, 1, 1)],
            view_offset: 3,
        },
        // Uniform steps with disjoint destinations: one strided-batch run.
        Case {
            name: "uniform run",
            jobs: (0..3).map(|i| job(i * 12, i * 8, i * 6, 4, 2, 3)).collect(),
            view_offset: 1,
        },
        // Mixed shapes and gaps: the grouped route.
        Case {
            name: "irregular jobs",
            jobs: vec![
                job(1, 0, 0, 3, 4, 2),
                job(9, 14, 8, 6, 1, 5),
                job(40, 21, 13, 2, 7, 3),
                job(47, 36, 34, 7, 2, 2),
            ],
            view_offset: 2,
        },
    ]
}

fn extent(jobs: &[DenseGemmBatchJob], view_offset: usize) -> (usize, usize, usize) {
    let end = |f: fn(&DenseGemmBatchJob) -> usize| {
        view_offset + jobs.iter().map(f).max().unwrap_or(0) + 1
    };
    (
        end(|j| j.dst_offset + j.rows * j.cols),
        end(|j| j.lhs_offset + j.rows * j.contracted),
        end(|j| j.rhs_offset + j.contracted * j.cols),
    )
}

struct Buffers<T> {
    out: Vec<Complex<T>>,
    lhs: Vec<Complex<T>>,
    rhs: Vec<T>,
}

fn buffers<T: Lane>(case: &Case, special: bool) -> Buffers<T>
where
    Complex<T>: numerics::Numeric,
{
    let (out_len, lhs_len, rhs_len) = extent(&case.jobs, case.view_offset);
    let re = values::<T>(lhs_len, 1, special);
    let im = values::<T>(lhs_len, 2, special);
    // A sentinel shows which destination entries the batch left untouched.
    let sentinel = Complex::new(T::from_f64(7.0), T::from_f64(-7.0));
    Buffers {
        out: vec![sentinel; out_len],
        lhs: re
            .into_iter()
            .zip(im)
            .map(|(r, i)| Complex::new(r, i))
            .collect(),
        rhs: values::<T>(rhs_len, 3, special),
    }
}

fn run_real_rhs<T: Lane, E: DenseExecutor>(
    executor: &mut E,
    case: &Case,
    b: &mut Buffers<T>,
) -> Result<(), DenseError>
where
    Complex<T>: numerics::Numeric,
{
    let (out_shape, lhs_shape, rhs_shape) = (
        [b.out.len() - case.view_offset],
        [b.lhs.len() - case.view_offset],
        [b.rhs.len() - case.view_offset],
    );
    let unit = [1];
    let runs = strided_batch_runs(&case.jobs);
    executor.matmul_batch_real_rhs_into(
        T::write_complex(
            DenseViewMut::new(&mut b.out, &out_shape, &unit, case.view_offset).unwrap(),
        ),
        T::read_complex(DenseView::new(&b.lhs, &lhs_shape, &unit, case.view_offset).unwrap()),
        T::read(DenseView::new(&b.rhs, &rhs_shape, &unit, case.view_offset).unwrap()),
        &case.jobs,
        &runs,
    )
}

/// The real batch GEMM of one component, from the same sentinel.
fn real_component<T: Lane>(
    executor: &mut DefaultDenseExecutor,
    case: &Case,
    b: &Buffers<T>,
    part: fn(Complex<T>) -> T,
) -> Vec<T>
where
    Complex<T>: numerics::Numeric,
{
    let mut out: Vec<T> = b.out.iter().copied().map(part).collect();
    let lhs: Vec<T> = b.lhs.iter().copied().map(part).collect();
    let (out_shape, lhs_shape, rhs_shape) = (
        [out.len() - case.view_offset],
        [lhs.len() - case.view_offset],
        [b.rhs.len() - case.view_offset],
    );
    let unit = [1];
    executor
        .matmul_batch_axpby_into(
            T::write(DenseViewMut::new(&mut out, &out_shape, &unit, case.view_offset).unwrap()),
            T::read(DenseView::new(&lhs, &lhs_shape, &unit, case.view_offset).unwrap()),
            T::read(DenseView::new(&b.rhs, &rhs_shape, &unit, case.view_offset).unwrap()),
            &case.jobs,
            &strided_batch_runs(&case.jobs),
            T::ONE,
            T::ZERO,
        )
        .unwrap();
    out
}

/// Naive per-entry sums of each component, in `f64`.
fn naive<T: Lane>(case: &Case, b: &Buffers<T>) -> Vec<(usize, f64, f64)>
where
    Complex<T>: numerics::Numeric,
{
    let o = case.view_offset;
    let mut entries = Vec::new();
    for j in &case.jobs {
        for col in 0..j.cols {
            for row in 0..j.rows {
                let (mut re, mut im) = (0.0, 0.0);
                for k in 0..j.contracted {
                    let x = b.lhs[o + j.lhs_offset + row + k * j.rows];
                    let u = b.rhs[o + j.rhs_offset + k + col * j.contracted].to_f64();
                    re += x.re.to_f64() * u;
                    im += x.im.to_f64() * u;
                }
                entries.push((o + j.dst_offset + row + col * j.rows, re, im));
            }
        }
    }
    entries
}

#[track_caller]
fn assert_component_close<T: Lane>(what: &str, got: T, want: f64, terms: usize)
where
    Complex<T>: numerics::Numeric,
{
    let got = got.to_f64();
    if want.is_nan() || got.is_nan() {
        assert!(
            want.is_nan() && got.is_nan(),
            "{what}: {got} vs naive {want}"
        );
    } else if want.is_infinite() {
        assert_eq!(got, want, "{what}");
    } else {
        let bound = numerics::tolerance::<T>(terms, want.abs());
        assert!(
            (got - want).abs() <= bound,
            "{what}: {got} vs naive {want} (> {bound:e})"
        );
    }
}

fn differential<T: Lane>()
where
    Complex<T>: numerics::Numeric,
{
    let mut executor = DefaultDenseExecutor::new();
    let mut non_finite = false;
    for case in cases() {
        let mut b = buffers::<T>(&case, true);
        let untouched = b.out.clone();
        let want_re = real_component(&mut executor, &case, &b, |z| z.re);
        let want_im = real_component(&mut executor, &case, &b, |z| z.im);
        run_real_rhs(&mut executor, &case, &mut b).unwrap();

        let terms = case.jobs.iter().map(|j| j.contracted).max().unwrap();
        for (i, z) in b.out.iter().enumerate() {
            // Faer's real kernel reduces each entry independently of the row
            // count, so the interleaved GEMM reproduces the separated one bit
            // for bit; that is what makes TeNeT's bitwise transform oracle
            // hold. Accelerate's does not (last-ulp and NaN-payload
            // differences), so the portable contract is the value class and
            // the tolerance.
            if cfg!(all(feature = "cpu-faer", not(feature = "cpu-blas-core"))) {
                assert_eq!(
                    (z.re.bits(), z.im.bits()),
                    (want_re[i].bits(), want_im[i].bits()),
                    "{}: entry {i} is not the componentwise real GEMM",
                    case.name
                );
            }
            assert_component_close(case.name, z.re, want_re[i].to_f64(), terms);
            assert_component_close(case.name, z.im, want_im[i].to_f64(), terms);
        }
        let written = naive(&case, &b);
        for &(i, re, im) in &written {
            assert_component_close(case.name, b.out[i].re, re, terms);
            assert_component_close(case.name, b.out[i].im, im, terms);
        }
        for (i, z) in b.out.iter().enumerate() {
            if written.iter().all(|&(w, _, _)| w != i) {
                assert_eq!(
                    *z, untouched[i],
                    "{}: entry {i} outside every job",
                    case.name
                );
            }
        }
        non_finite |= b
            .out
            .iter()
            .any(|z| !z.re.to_f64().is_finite() || !z.im.to_f64().is_finite());
    }
    assert!(non_finite, "the fixtures reach no non-finite output");
}

#[test]
fn real_rhs_batch_is_the_componentwise_real_gemm_c64() {
    differential::<f64>();
}

#[test]
fn real_rhs_batch_is_the_componentwise_real_gemm_c32() {
    differential::<f32>();
}

/// On finite data the real-rhs product and the promoted complex GEMM are the
/// same linear map; they may differ in the last bits, since one runs a real
/// GEMM and the other a complex product.
fn matches_promoted_complex_gemm<T: Lane>()
where
    Complex<T>: numerics::Numeric,
{
    let mut executor = DefaultDenseExecutor::new();
    for case in cases() {
        let mut b = buffers::<T>(&case, false);
        let mut promoted = b.out.clone();
        let rhs: Vec<Complex<T>> = b
            .rhs
            .iter()
            .map(|&u| Complex::new(u, T::default()))
            .collect();
        let o = case.view_offset;
        let (out_shape, lhs_shape, rhs_shape) =
            ([promoted.len() - o], [b.lhs.len() - o], [rhs.len() - o]);
        let unit = [1];
        executor
            .matmul_batch_axpby_into(
                T::write_complex(DenseViewMut::new(&mut promoted, &out_shape, &unit, o).unwrap()),
                T::read_complex(DenseView::new(&b.lhs, &lhs_shape, &unit, o).unwrap()),
                T::read_complex(DenseView::new(&rhs, &rhs_shape, &unit, o).unwrap()),
                &case.jobs,
                &strided_batch_runs(&case.jobs),
                T::COMPLEX_ONE,
                T::COMPLEX_ZERO,
            )
            .unwrap();
        run_real_rhs(&mut executor, &case, &mut b).unwrap();
        // Each entry sums `contracted` products of one lhs row with one rhs column.
        let terms = case.jobs.iter().map(|j| j.contracted).max().unwrap();
        for (got, want) in b.out.iter().zip(&promoted) {
            numerics::assert_close(case.name, *got, *want, terms);
        }
    }
}

#[test]
fn real_rhs_batch_matches_promoted_complex_gemm_on_finite_data() {
    matches_promoted_complex_gemm::<f64>();
    matches_promoted_complex_gemm::<f32>();
}

/// Only the trait's required methods: the real-rhs batch runs the default
/// componentwise loop.
struct DefaultsOnly;

impl DenseExecutor for DefaultsOnly {
    fn svd(&mut self, _input: DenseRead<'_>) -> Result<Vec<DenseTensor>, DenseError> {
        unreachable!()
    }
    fn qr(&mut self, _input: DenseRead<'_>) -> Result<Vec<DenseTensor>, DenseError> {
        unreachable!()
    }
    fn eigh(&mut self, _input: DenseRead<'_>) -> Result<Vec<DenseTensor>, DenseError> {
        unreachable!()
    }
    fn dot_general_into(
        &mut self,
        _output: DenseWrite<'_>,
        _lhs: DenseRead<'_>,
        _rhs: DenseRead<'_>,
        _config: &DenseDotConfig,
    ) -> Result<(), DenseError> {
        unreachable!("complex operands never reach a dot")
    }
}

fn default_loop<T: Lane>()
where
    Complex<T>: numerics::Numeric,
{
    for case in cases() {
        let mut b = buffers::<T>(&case, true);
        run_real_rhs(&mut DefaultsOnly, &case, &mut b).unwrap();
        let terms = case.jobs.iter().map(|j| j.contracted).max().unwrap();
        for (i, re, im) in naive(&case, &b) {
            assert_component_close(case.name, b.out[i].re, re, terms);
            assert_component_close(case.name, b.out[i].im, im, terms);
        }
    }
}

#[test]
fn default_real_rhs_batch_is_componentwise() {
    default_loop::<f64>();
    default_loop::<f32>();
}

#[test]
fn real_rhs_batch_rejects_mismatched_dtypes() {
    let mut out = vec![Complex64::default(); 1];
    let lhs = vec![Complex64::default(); 1];
    let rhs = vec![0.0f32; 1];
    let shape = [1];
    let unit = [1];
    let error = DefaultDenseExecutor::new()
        .matmul_batch_real_rhs_into(
            DenseWrite::C64(DenseViewMut::new(&mut out, &shape, &unit, 0).unwrap()),
            DenseRead::C64(DenseView::new(&lhs, &shape, &unit, 0).unwrap()),
            DenseRead::F32(DenseView::new(&rhs, &shape, &unit, 0).unwrap()),
            &[job(0, 0, 0, 1, 1, 1)],
            &[1],
        )
        .unwrap_err();
    assert!(matches!(
        error,
        DenseError::Unsupported {
            op: "matmul_batch_real_rhs_into",
            ..
        }
    ));
}
