#[allow(unused_imports)]
use super::*;

include!("../../tests/common/predicate_chains.rs");

impl<T: ScalarOps> ChainCoefficient for T {
    fn real(value: f64) -> Self {
        T::from_real(value)
    }
}

/// The pre-#1541 tuple order of a two-factor result, for the helpers
/// below that treat QR/LQ and left/right polar factors uniformly.
trait FactorPair<T> {
    fn pair(self) -> (T, T);
}
impl<T> FactorPair<T> for Qr<T> {
    fn pair(self) -> (T, T) {
        (self.q, self.r)
    }
}
impl<T> FactorPair<T> for Lq<T> {
    fn pair(self) -> (T, T) {
        (self.l, self.q)
    }
}
impl<T> FactorPair<T> for LeftPolar<T> {
    fn pair(self) -> (T, T) {
        (self.w, self.p)
    }
}
impl<T> FactorPair<T> for RightPolar<T> {
    fn pair(self) -> (T, T) {
        (self.p, self.wh)
    }
}
use tenet_core::{product_sector, ProductFusionRuleExt};
use tenet_core::{
    BlockKey, BlockSpec, BlockStructure, CU1FusionRule, CU1Irrep, FermionParityFusionRule,
    FusionTreeKey, FusionTreePairKey, SU2FusionRule, SU2Irrep, U1FusionRule, U1Irrep, Z2FusionRule,
    Z2Irrep, ZNFusionRule,
};
use tenet_dense::{
    DefaultDenseExecutor, DenseBackend, DenseDotConfig, DenseError, DenseExecutor, DenseRead,
    DenseTensor, DenseWrite,
};

#[derive(Default)]
struct CountPolarKernels {
    inner: DefaultDenseExecutor,
    svd_calls: Arc<std::sync::atomic::AtomicUsize>,
    gemm_calls: Arc<std::sync::atomic::AtomicUsize>,
}

impl DenseExecutor for CountPolarKernels {
    fn svd(&mut self, input: DenseRead<'_>) -> Result<Vec<DenseTensor>, DenseError> {
        self.svd_calls
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        self.inner.svd(input)
    }
    fn svd_into(
        &mut self,
        input: DenseRead<'_>,
        u: DenseWrite<'_>,
        s: DenseWrite<'_>,
        vt: DenseWrite<'_>,
    ) -> Result<(), DenseError> {
        self.svd_calls
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        self.inner.svd_into(input, u, s, vt)
    }
    fn qr(&mut self, _: DenseRead<'_>) -> Result<Vec<DenseTensor>, DenseError> {
        panic!("test only exercises polar")
    }
    fn eigh(&mut self, _: DenseRead<'_>) -> Result<Vec<DenseTensor>, DenseError> {
        panic!("test only exercises polar")
    }
    fn dot_general_into(
        &mut self,
        output: DenseWrite<'_>,
        lhs: DenseRead<'_>,
        rhs: DenseRead<'_>,
        config: &DenseDotConfig,
    ) -> Result<(), DenseError> {
        self.gemm_calls
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        self.inner.dot_general_into(output, lhs, rhs, config)
    }
}

#[test]
fn compact_diagonal_qr_lq_preserves_compact_factors_without_dense_kernels() {
    let runtime = Runtime::builder()
        .with_dense_executor(Box::<CountPolarKernels>::default())
        .build()
        .unwrap();
    let leg = GradedSpace::try_new(Arc::new(U1FusionRule), [(U1Irrep::new(1), 3)])
        .unwrap()
        .try_dual()
        .unwrap();
    let input: TensorMap<_, f64> = TensorMap::diagonal(
        &runtime,
        &leg,
        [SectorSpectrum {
            sector: U1Irrep::new(-1),
            values: vec![-2.0, 0.0, 3.0],
        }],
    )
    .unwrap();
    for operation in 0..4 {
        DIAGONAL_MATERIALIZATIONS.set(0);
        let (left, right) = match operation {
            0 => input.qr_compact(&[0], &[1]).unwrap().pair(),
            1 => input.qr_full(&[0], &[1]).unwrap().pair(),
            2 => input.lq_compact(&[0], &[1]).unwrap().pair(),
            _ => input.lq_full(&[0], &[1]).unwrap().pair(),
        };
        assert_eq!(DIAGONAL_MATERIALIZATIONS.get(), 0);
        for factor in [&left, &right] {
            assert_eq!(factor.codomain(), input.codomain());
            assert_eq!(factor.domain(), input.domain());
            assert!(factor.diagview().is_ok());
            assert!(factor.dense_data().is_err());
            assert_eq!(
                factor.map_diagonal(|x| x).unwrap().diagview().unwrap(),
                factor.diagview().unwrap()
            );
        }
    }
}

#[test]
fn compact_diagonal_qr_lq_rejects_inconsistent_spectra_and_nonbond_spaces() {
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let leg = GradedSpace::try_new(
        Arc::new(U1FusionRule),
        [(U1Irrep::new(0), 2), (U1Irrep::new(1), 1)],
    )
    .unwrap();
    let input: TensorMap<_, f64> = TensorMap::diagonal(
        &runtime,
        &leg,
        [
            SectorSpectrum {
                sector: U1Irrep::new(1),
                values: vec![2.0],
            },
            SectorSpectrum {
                sector: U1Irrep::new(0),
                values: vec![-1.0, 0.0],
            },
        ],
    )
    .unwrap();
    let spectrum = input.spectrum().unwrap();
    let admit = |values: &[tenet_matrixalgebra::SectorSpectrum<f64>]| {
        tenet_matrixalgebra::qr_diagonal_dyn(input.logical_space(), values)
    };
    assert!(admit(spectrum).is_some());
    let mut reversed = spectrum.to_vec();
    reversed.reverse();
    assert!(admit(&reversed).is_some());
    assert!(admit(&spectrum[..1]).is_none());
    let mut duplicate = spectrum.to_vec();
    duplicate[1].sector = duplicate[0].sector;
    assert!(admit(&duplicate).is_none());
    let mut wrong = spectrum.to_vec();
    wrong[0].values.push(1.0);
    assert!(admit(&wrong).is_none());
    wrong = spectrum.to_vec();
    wrong[0].values[0] = f64::NAN;
    assert!(admit(&wrong).is_none());
    let multileg: TensorMap<_, f64> =
        TensorMap::rand_with_seed(&runtime, [&leg, &leg], [&leg, &leg], 1).unwrap();
    assert!(tenet_matrixalgebra::qr_diagonal_dyn(multileg.logical_space(), spectrum).is_none());
    let dual = leg.try_dual().unwrap();
    let nonendo: TensorMap<_, f64> =
        TensorMap::rand_with_seed(&runtime, [&leg], [&dual], 1).unwrap();
    assert!(tenet_matrixalgebra::qr_diagonal_dyn(nonendo.logical_space(), spectrum).is_none());
}

#[test]
fn compact_diagonal_polar_skips_input_materialization_svd_and_gemm() {
    let svd_calls = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let gemm_calls = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let runtime = Runtime::builder()
        .with_dense_executor(Box::new(CountPolarKernels {
            inner: DefaultDenseExecutor::default(),
            svd_calls: Arc::clone(&svd_calls),
            gemm_calls: Arc::clone(&gemm_calls),
        }))
        .build()
        .unwrap();
    let leg = GradedSpace::try_new(Arc::new(U1FusionRule), [(U1Irrep::new(0), 3)]).unwrap();
    let input: TensorMap<_, Complex64> = TensorMap::diagonal(
        &runtime,
        &leg,
        [SectorSpectrum {
            sector: U1Irrep::new(0),
            values: vec![
                Complex64::new(-2.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 3.0),
            ],
        }],
    )
    .unwrap();
    DIAGONAL_MATERIALIZATIONS.set(0);
    let left = input.left_polar(&[0], &[1]).unwrap();
    assert!(left.w.diagview().is_ok());
    assert!(left.p.diagview().is_ok());
    assert!(left.w.dense_data().is_err());
    assert!(left.p.dense_data().is_err());
    assert_eq!(DIAGONAL_MATERIALIZATIONS.get(), 0);
    assert_eq!(svd_calls.load(std::sync::atomic::Ordering::Relaxed), 0);
    assert_eq!(gemm_calls.load(std::sync::atomic::Ordering::Relaxed), 0);
    let old_dense = tenet_matrixalgebra::left_polar_diagonal_dyn(
        input.logical_space(),
        input.spectrum().unwrap(),
    )
    .unwrap()
    .unwrap();
    assert_eq!(
        old_dense.w.data(),
        left.w.materialize().unwrap().dense_data().unwrap()
    );
    assert_eq!(
        old_dense.p.data(),
        left.p.materialize().unwrap().dense_data().unwrap()
    );
    DIAGONAL_MATERIALIZATIONS.set(0);
    let right = input.right_polar(&[0], &[1]).unwrap();
    assert!(right.wh.diagview().is_ok());
    assert!(right.p.diagview().is_ok());
    assert!(right.wh.dense_data().is_err());
    assert!(right.p.dense_data().is_err());
    assert_eq!(DIAGONAL_MATERIALIZATIONS.get(), 0);
    assert_eq!(svd_calls.load(std::sync::atomic::Ordering::Relaxed), 0);
    assert_eq!(gemm_calls.load(std::sync::atomic::Ordering::Relaxed), 0);
}

#[test]
fn compact_diagonal_qr_lq_matches_hand_phase_and_absolute_value_all_scalars() {
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let leg = GradedSpace::try_new(Arc::new(U1FusionRule), [(U1Irrep::new(0), 4)]).unwrap();
    macro_rules! check {
        ($ty:ty, $values:expr, $phase:expr, $magnitude:expr, $tol:expr) => {{
            let input: TensorMap<_, $ty> = TensorMap::diagonal(
                &runtime,
                &leg,
                [SectorSpectrum {
                    sector: U1Irrep::new(0),
                    values: $values,
                }],
            )
            .unwrap();
            let saved = input.diagview().unwrap();
            for operation in 0..4 {
                let (w, p) = match operation {
                    0 => input.qr_compact(&[0], &[1]).unwrap().pair(),
                    1 => input.qr_full(&[0], &[1]).unwrap().pair(),
                    2 => {
                        let Lq { l, q } = input.lq_compact(&[0], &[1]).unwrap();
                        (q, l)
                    }
                    _ => {
                        let Lq { l, q } = input.lq_full(&[0], &[1]).unwrap();
                        (q, l)
                    }
                };
                assert_eq!(w.codomain(), input.codomain());
                assert_eq!(w.domain(), input.domain());
                assert_eq!(p.codomain(), input.domain());
                assert_eq!(p.domain(), p.codomain());
                assert!(w.dense_data().is_err());
                assert!(p.dense_data().is_err());
                assert!(w.diagview().is_ok());
                assert!(p.diagview().is_ok());
                let w = w.materialize().unwrap();
                let p = p.materialize().unwrap();
                for (i, (&actual_w, &actual_p)) in w
                    .dense_data()
                    .unwrap()
                    .iter()
                    .zip(p.dense_data().unwrap())
                    .enumerate()
                {
                    let row = i % 4;
                    let col = i / 4;
                    let expected_w = if row == col {
                        $phase[row]
                    } else {
                        Complex64::new(0.0, 0.0)
                    };
                    let expected_p = if row == col { $magnitude[row] } else { 0.0 };
                    assert!((actual_w.widen_complex() - expected_w).norm() <= $tol);
                    assert!((actual_p.widen_complex().re - expected_p).abs() <= $tol);
                }
                assert_typed_map_close(
                    &w.compose(&p).unwrap(),
                    &input.materialize().unwrap(),
                    $tol,
                );
                let unit = w.adjoint().unwrap().compose(&w).unwrap();
                for (i, &value) in unit.dense_data().unwrap().iter().enumerate() {
                    let expected = if i % 4 == i / 4 { 1.0 } else { 0.0 };
                    assert!((value.widen_complex() - Complex64::new(expected, 0.0)).norm() <= $tol);
                }
            }
            assert_eq!(input.diagview().unwrap(), saved);
        }};
    }
    let phase = [
        Complex64::new(-1.0, 0.0),
        Complex64::new(0.6, 0.8),
        Complex64::new(1.0, 0.0),
        Complex64::new(1.0, 0.0),
    ];
    let magnitude = [2.0, 5.0, 0.0, 1.0];
    check!(
        f32,
        vec![-2.0, 5.0, -0.0, 1.0],
        [phase[0], Complex64::new(1.0, 0.0), phase[2], phase[3]],
        magnitude,
        2e-6
    );
    check!(
        f64,
        vec![-2.0, 5.0, -0.0, 1.0],
        [phase[0], Complex64::new(1.0, 0.0), phase[2], phase[3]],
        magnitude,
        1e-12
    );
    check!(
        num_complex::Complex32,
        vec![
            num_complex::Complex32::new(-2.0, 0.0),
            num_complex::Complex32::new(3.0, 4.0),
            num_complex::Complex32::new(-0.0, 0.0),
            num_complex::Complex32::new(1.0, 0.0)
        ],
        phase,
        magnitude,
        2e-6
    );
    check!(
        Complex64,
        vec![
            Complex64::new(-2.0, 0.0),
            Complex64::new(3.0, 4.0),
            Complex64::new(-0.0, 0.0),
            Complex64::new(1.0, 0.0)
        ],
        phase,
        magnitude,
        1e-12
    );
}

#[test]
fn compact_diagonal_qr_lq_preserves_sectors_and_changed_roles() {
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    macro_rules! check {
        ($leg:expr, $spectra:expr) => {{
            let input: TensorMap<_, f64> = TensorMap::diagonal(&runtime, &$leg, $spectra).unwrap();
            for (rows, cols) in [([0], [1]), ([1], [0])] {
                let permuted = input.permute(&rows, &cols).unwrap();
                let dense = permuted.materialize().unwrap();
                for operation in 0..4 {
                    let (a, b) = match operation {
                        0 => input.qr_compact(&rows, &cols).unwrap().pair(),
                        1 => input.qr_full(&rows, &cols).unwrap().pair(),
                        2 => input.lq_compact(&rows, &cols).unwrap().pair(),
                        _ => input.lq_full(&rows, &cols).unwrap().pair(),
                    };
                    assert_eq!(a.codomain(), permuted.codomain());
                    assert_eq!(a.domain(), permuted.domain());
                    assert_eq!(b.codomain(), permuted.codomain());
                    assert_eq!(b.domain(), permuted.domain());
                    assert!(a.diagview().is_ok());
                    assert!(b.diagview().is_ok());
                    let reconstructed = a.compose(&b).unwrap().materialize().unwrap();
                    assert_typed_map_close(&reconstructed, &dense, 1e-12);
                    let q = if operation < 2 { &a } else { &b };
                    let gram = q.adjoint().unwrap().compose(q).unwrap();
                    let expected =
                        TensorMap::isomorphism(&runtime, &q.domain(), &q.domain()).unwrap();
                    assert_typed_map_close(
                        &gram.materialize().unwrap(),
                        &expected.materialize().unwrap(),
                        1e-12,
                    );
                }
            }
        }};
    }
    let u1 = GradedSpace::try_new(
        Arc::new(U1FusionRule),
        [(U1Irrep::new(0), 3), (U1Irrep::new(1), 2)],
    )
    .unwrap()
    .try_dual()
    .unwrap();
    check!(
        u1,
        [
            SectorSpectrum {
                sector: U1Irrep::new(-1),
                values: vec![-2.0, 0.0]
            },
            SectorSpectrum {
                sector: U1Irrep::new(0),
                values: vec![1.0, -3.0, 2.0]
            },
        ]
    );
    let spin0 = SU2Irrep::from_twice_spin(0);
    let half = SU2Irrep::from_twice_spin(1);
    let su2 = GradedSpace::try_new(Arc::new(SU2FusionRule), [(spin0, 2), (half, 1)]).unwrap();
    check!(
        su2,
        [
            SectorSpectrum {
                sector: spin0,
                values: vec![2.0, -2.0]
            },
            SectorSpectrum {
                sector: half,
                values: vec![0.0]
            },
        ]
    );
    let even = product_sector(U1Irrep::new(0), Z2Irrep::EVEN);
    let odd = product_sector(U1Irrep::new(1), Z2Irrep::ODD);
    let fermion = GradedSpace::try_new(
        Arc::new(U1FusionRule.product(FermionParityFusionRule)),
        [(even, 2), (odd, 1)],
    )
    .unwrap();
    check!(
        fermion,
        [
            SectorSpectrum {
                sector: even,
                values: vec![-2.0, 0.0]
            },
            SectorSpectrum {
                sector: odd,
                values: vec![3.0]
            },
        ]
    );
    let zero = GradedSpace::try_new(
        Arc::new(U1FusionRule),
        [(U1Irrep::new(0), 0), (U1Irrep::new(1), 2)],
    )
    .unwrap();
    check!(
        zero,
        [SectorSpectrum {
            sector: U1Irrep::new(1),
            values: vec![0.0, -3.0],
        }]
    );
    let empty = GradedSpace::try_new(Arc::new(U1FusionRule), []).unwrap();
    check!(empty, Vec::<SectorSpectrum<_, f64>>::new());
}

#[test]
fn compact_diagonal_qr_lq_scales_complex_subnormals_before_normalizing() {
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let leg = GradedSpace::try_new(Arc::new(U1FusionRule), [(U1Irrep::new(0), 1)]).unwrap();
    macro_rules! check {
        ($ty:ty, $tiny:expr, $tol:expr) => {{
            let tiny = $tiny;
            let input: TensorMap<_, $ty> = TensorMap::diagonal(
                &runtime,
                &leg,
                [SectorSpectrum {
                    sector: U1Irrep::new(0),
                    values: vec![<$ty>::new(tiny, tiny)],
                }],
            )
            .unwrap();
            for operation in 0..4 {
                let (w, p) = match operation {
                    0 => input.qr_compact(&[0], &[1]).unwrap().pair(),
                    1 => input.qr_full(&[0], &[1]).unwrap().pair(),
                    2 => {
                        let Lq { l, q } = input.lq_compact(&[0], &[1]).unwrap();
                        (q, l)
                    }
                    _ => {
                        let Lq { l, q } = input.lq_full(&[0], &[1]).unwrap();
                        (q, l)
                    }
                };
                let phase = w.diagview().unwrap()[0].values[0].widen_complex();
                let magnitude = p.diagview().unwrap()[0].values[0].widen_complex().re;
                assert!((phase.re - std::f64::consts::FRAC_1_SQRT_2).abs() < $tol);
                assert!((phase.im - std::f64::consts::FRAC_1_SQRT_2).abs() < $tol);
                assert!(magnitude.is_finite() && magnitude > 0.0);
                assert!((phase.norm() - 1.0).abs() <= $tol);
                let rebuilt = phase * magnitude;
                let original = input.diagview().unwrap()[0].values[0].widen_complex();
                // The minimum subnormal rounds its magnitude to one ULP.
                assert!((rebuilt.re - original.re).abs() <= tiny as f64);
                assert!((rebuilt.im - original.im).abs() <= tiny as f64);
            }
        }};
    }
    check!(num_complex::Complex32, f32::from_bits(1), 2e-6);
    check!(Complex64, f64::from_bits(1), 1e-12);
}

#[test]
fn compact_diagonal_qr_lq_nonfinite_and_unrepresentable_values_use_existing_route() {
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let leg = GradedSpace::try_new(Arc::new(U1FusionRule), [(U1Irrep::new(0), 1)]).unwrap();
    for value in [
        Complex64::new(f64::NAN, 0.0),
        Complex64::new(f64::INFINITY, 1.0),
        Complex64::new(f64::MAX, f64::MAX),
    ] {
        let input: TensorMap<_, Complex64> = TensorMap::diagonal(
            &runtime,
            &leg,
            [SectorSpectrum {
                sector: U1Irrep::new(0),
                values: vec![value],
            }],
        )
        .unwrap();
        for operation in 0..4 {
            DIAGONAL_MATERIALIZATIONS.set(0);
            match operation {
                0 => {
                    let _ = input.qr_compact(&[0], &[1]);
                }
                1 => {
                    let _ = input.qr_full(&[0], &[1]);
                }
                2 => {
                    let _ = input.lq_compact(&[0], &[1]);
                }
                _ => {
                    let _ = input.lq_full(&[0], &[1]);
                }
            }
            assert_eq!(DIAGONAL_MATERIALIZATIONS.get(), 1);
        }
    }
    let input: TensorMap<_, num_complex::Complex32> = TensorMap::diagonal(
        &runtime,
        &leg,
        [SectorSpectrum {
            sector: U1Irrep::new(0),
            values: vec![num_complex::Complex32::new(f32::MAX, f32::MAX)],
        }],
    )
    .unwrap();
    for operation in 0..4 {
        DIAGONAL_MATERIALIZATIONS.set(0);
        match operation {
            0 => {
                let _ = input.qr_compact(&[0], &[1]);
            }
            1 => {
                let _ = input.qr_full(&[0], &[1]);
            }
            2 => {
                let _ = input.lq_compact(&[0], &[1]);
            }
            _ => {
                let _ = input.lq_full(&[0], &[1]);
            }
        }
        assert_eq!(DIAGONAL_MATERIALIZATIONS.get(), 1);
    }
}

#[test]
fn compact_diagonal_polar_matches_hand_phase_and_absolute_value_all_scalars() {
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let leg = GradedSpace::try_new(Arc::new(U1FusionRule), [(U1Irrep::new(0), 4)]).unwrap();
    macro_rules! check {
        ($ty:ty, $values:expr, $phase:expr, $magnitude:expr, $tol:expr) => {{
            let input: TensorMap<_, $ty> = TensorMap::diagonal(
                &runtime,
                &leg,
                [SectorSpectrum {
                    sector: U1Irrep::new(0),
                    values: $values,
                }],
            )
            .unwrap();
            let saved = input.diagview().unwrap();
            for left in [true, false] {
                let (w, p) = if left {
                    let LeftPolar { w, p } = input.left_polar(&[0], &[1]).unwrap();
                    (w, p)
                } else {
                    let RightPolar { p, wh } = input.right_polar(&[0], &[1]).unwrap();
                    (wh, p)
                };
                assert_eq!(w.codomain(), input.codomain());
                assert_eq!(w.domain(), input.domain());
                assert_eq!(
                    p.codomain(),
                    if left {
                        input.domain()
                    } else {
                        input.codomain()
                    }
                );
                assert_eq!(p.domain(), p.codomain());
                assert!(w.dense_data().is_err());
                assert!(p.dense_data().is_err());
                assert!(w.diagview().is_ok());
                assert!(p.diagview().is_ok());
                let w = w.materialize().unwrap();
                let p = p.materialize().unwrap();
                for (i, (&actual_w, &actual_p)) in w
                    .dense_data()
                    .unwrap()
                    .iter()
                    .zip(p.dense_data().unwrap())
                    .enumerate()
                {
                    let row = i % 4;
                    let col = i / 4;
                    let expected_w = if row == col {
                        $phase[row]
                    } else {
                        Complex64::new(0.0, 0.0)
                    };
                    let expected_p = if row == col { $magnitude[row] } else { 0.0 };
                    assert!((actual_w.widen_complex() - expected_w).norm() <= $tol);
                    assert!((actual_p.widen_complex().re - expected_p).abs() <= $tol);
                }
                assert_typed_map_close(
                    &w.compose(&p).unwrap(),
                    &input.materialize().unwrap(),
                    $tol,
                );
                let unit = w.adjoint().unwrap().compose(&w).unwrap();
                for (i, &value) in unit
                    .materialize()
                    .unwrap()
                    .dense_data()
                    .unwrap()
                    .iter()
                    .enumerate()
                {
                    let expected = if i % 4 == i / 4 { 1.0 } else { 0.0 };
                    assert!((value.widen_complex() - Complex64::new(expected, 0.0)).norm() <= $tol);
                }
            }
            assert_eq!(input.diagview().unwrap(), saved);
        }};
    }
    let phase = [
        Complex64::new(-1.0, 0.0),
        Complex64::new(0.6, 0.8),
        Complex64::new(1.0, 0.0),
        Complex64::new(1.0, 0.0),
    ];
    let magnitude = [2.0, 5.0, 0.0, 1.0];
    check!(
        f32,
        vec![-2.0, 5.0, -0.0, 1.0],
        [phase[0], Complex64::new(1.0, 0.0), phase[2], phase[3]],
        magnitude,
        2e-6
    );
    check!(
        f64,
        vec![-2.0, 5.0, -0.0, 1.0],
        [phase[0], Complex64::new(1.0, 0.0), phase[2], phase[3]],
        magnitude,
        1e-12
    );
    check!(
        num_complex::Complex32,
        vec![
            num_complex::Complex32::new(-2.0, 0.0),
            num_complex::Complex32::new(3.0, 4.0),
            num_complex::Complex32::new(-0.0, 0.0),
            num_complex::Complex32::new(1.0, 0.0)
        ],
        phase,
        magnitude,
        2e-6
    );
    check!(
        Complex64,
        vec![
            Complex64::new(-2.0, 0.0),
            Complex64::new(3.0, 4.0),
            Complex64::new(-0.0, 0.0),
            Complex64::new(1.0, 0.0)
        ],
        phase,
        magnitude,
        1e-12
    );
}

#[test]
fn compact_diagonal_polar_preserves_sectors_and_changed_roles() {
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    macro_rules! check {
        ($leg:expr, $spectra:expr) => {{
            let input: TensorMap<_, f64> = TensorMap::diagonal(&runtime, &$leg, $spectra).unwrap();
            for (rows, cols) in [([0], [1]), ([1], [0])] {
                let permuted = input.permute(&rows, &cols).unwrap();
                let dense = permuted.materialize().unwrap();
                let expected_left = dense.left_polar(&[0], &[1]).unwrap();
                let expected_right = dense.right_polar(&[0], &[1]).unwrap();
                DIAGONAL_MATERIALIZATIONS.set(0);
                let left = input.left_polar(&rows, &cols).unwrap();
                let right = input.right_polar(&rows, &cols).unwrap();
                assert_eq!(DIAGONAL_MATERIALIZATIONS.get(), 0);
                if permuted.spectrum().is_some() {
                    for factor in [&left.w, &left.p, &right.p, &right.wh] {
                        assert!(factor.diagview().is_ok());
                        assert!(factor.dense_data().is_err());
                        assert_eq!(factor.codomain(), permuted.codomain());
                        assert_eq!(factor.domain(), permuted.domain());
                    }
                }
                assert_polar_factors(
                    &input,
                    &dense,
                    &(left.w, left.p),
                    &(expected_left.w, expected_left.p),
                    true,
                );
                assert_polar_factors(
                    &input,
                    &dense,
                    &(right.p, right.wh),
                    &(expected_right.p, expected_right.wh),
                    false,
                );
            }
        }};
    }
    let u1 = GradedSpace::try_new(
        Arc::new(U1FusionRule),
        [(U1Irrep::new(0), 3), (U1Irrep::new(1), 2)],
    )
    .unwrap()
    .try_dual()
    .unwrap();
    check!(
        u1,
        [
            SectorSpectrum {
                sector: U1Irrep::new(-1),
                values: vec![-2.0, 0.0]
            },
            SectorSpectrum {
                sector: U1Irrep::new(0),
                values: vec![1.0, -3.0, 2.0]
            },
        ]
    );
    let spin0 = SU2Irrep::from_twice_spin(0);
    let half = SU2Irrep::from_twice_spin(1);
    let su2 = GradedSpace::try_new(Arc::new(SU2FusionRule), [(spin0, 2), (half, 1)]).unwrap();
    check!(
        su2,
        [
            SectorSpectrum {
                sector: spin0,
                values: vec![2.0, -2.0]
            },
            SectorSpectrum {
                sector: half,
                values: vec![0.0]
            },
        ]
    );
    let even = product_sector(U1Irrep::new(0), Z2Irrep::EVEN);
    let odd = product_sector(U1Irrep::new(1), Z2Irrep::ODD);
    let fermion = GradedSpace::try_new(
        Arc::new(U1FusionRule.product(FermionParityFusionRule)),
        [(even, 2), (odd, 1)],
    )
    .unwrap();
    check!(
        fermion,
        [
            SectorSpectrum {
                sector: even,
                values: vec![-2.0, 0.0]
            },
            SectorSpectrum {
                sector: odd,
                values: vec![3.0]
            },
        ]
    );
    let zero = GradedSpace::try_new(
        Arc::new(U1FusionRule),
        [(U1Irrep::new(0), 0), (U1Irrep::new(1), 2)],
    )
    .unwrap();
    check!(
        zero,
        [SectorSpectrum {
            sector: U1Irrep::new(1),
            values: vec![0.0, -3.0],
        }]
    );
    let empty = GradedSpace::try_new(Arc::new(U1FusionRule), []).unwrap();
    check!(empty, Vec::<SectorSpectrum<_, f64>>::new());
}

#[test]
fn compact_diagonal_polar_scales_complex_subnormals_before_normalizing() {
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let leg = GradedSpace::try_new(Arc::new(U1FusionRule), [(U1Irrep::new(0), 1)]).unwrap();
    macro_rules! check {
        ($ty:ty, $tiny:expr, $tol:expr) => {{
            let tiny = $tiny;
            let input: TensorMap<_, $ty> = TensorMap::diagonal(
                &runtime,
                &leg,
                [SectorSpectrum {
                    sector: U1Irrep::new(0),
                    values: vec![<$ty>::new(tiny, tiny)],
                }],
            )
            .unwrap();
            for left in [true, false] {
                let (w, p) = if left {
                    let LeftPolar { w, p } = input.left_polar(&[0], &[1]).unwrap();
                    (w, p)
                } else {
                    let RightPolar { p, wh } = input.right_polar(&[0], &[1]).unwrap();
                    (wh, p)
                };
                let phase = w.diagview().unwrap()[0].values[0].widen_complex();
                let magnitude = p.diagview().unwrap()[0].values[0].widen_complex().re;
                assert!((phase.re - std::f64::consts::FRAC_1_SQRT_2).abs() < $tol);
                assert!((phase.im - std::f64::consts::FRAC_1_SQRT_2).abs() < $tol);
                assert!(magnitude.is_finite() && magnitude > 0.0);
            }
        }};
    }
    check!(num_complex::Complex32, f32::from_bits(1), 2e-6);
    check!(Complex64, f64::from_bits(1), 1e-12);
}

#[test]
fn compact_diagonal_polar_nonfinite_and_unrepresentable_values_use_existing_route() {
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let leg = GradedSpace::try_new(Arc::new(U1FusionRule), [(U1Irrep::new(0), 1)]).unwrap();
    for value in [
        Complex64::new(f64::NAN, 0.0),
        Complex64::new(f64::INFINITY, 1.0),
    ] {
        let input: TensorMap<_, Complex64> = TensorMap::diagonal(
            &runtime,
            &leg,
            [SectorSpectrum {
                sector: U1Irrep::new(0),
                values: vec![value],
            }],
        )
        .unwrap();
        for left in [true, false] {
            DIAGONAL_MATERIALIZATIONS.set(0);
            if left {
                let _ = input.left_polar(&[0], &[1]);
            } else {
                let _ = input.right_polar(&[0], &[1]);
            }
            assert_eq!(DIAGONAL_MATERIALIZATIONS.get(), 1);
        }
    }
    let input: TensorMap<_, num_complex::Complex32> = TensorMap::diagonal(
        &runtime,
        &leg,
        [SectorSpectrum {
            sector: U1Irrep::new(0),
            values: vec![num_complex::Complex32::new(f32::MAX, f32::MAX)],
        }],
    )
    .unwrap();
    for left in [true, false] {
        DIAGONAL_MATERIALIZATIONS.set(0);
        if left {
            let _ = input.left_polar(&[0], &[1]);
        } else {
            let _ = input.right_polar(&[0], &[1]);
        }
        assert_eq!(DIAGONAL_MATERIALIZATIONS.get(), 1);
    }
}

#[derive(Default)]
struct CountEighVals {
    inner: DefaultDenseExecutor,
    calls: Arc<std::sync::atomic::AtomicUsize>,
}

impl DenseExecutor for CountEighVals {
    fn svd(&mut self, _: DenseRead<'_>) -> Result<Vec<DenseTensor>, DenseError> {
        panic!("test only exercises eigh_vals")
    }

    fn qr(&mut self, _: DenseRead<'_>) -> Result<Vec<DenseTensor>, DenseError> {
        panic!("test only exercises eigh_vals")
    }

    fn eigh(&mut self, _: DenseRead<'_>) -> Result<Vec<DenseTensor>, DenseError> {
        panic!("test only exercises eigh_vals")
    }

    fn eigh_vals(&mut self, input: DenseRead<'_>) -> Result<DenseTensor, DenseError> {
        self.calls
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        self.inner.eigh_vals(input)
    }

    fn dot_general_into(
        &mut self,
        _: DenseWrite<'_>,
        _: DenseRead<'_>,
        _: DenseRead<'_>,
        _: &DenseDotConfig,
    ) -> Result<(), DenseError> {
        panic!("test only exercises eigh_vals")
    }
}

#[derive(Default)]
struct CountEighFull {
    inner: DefaultDenseExecutor,
    calls: Arc<std::sync::atomic::AtomicUsize>,
}

impl DenseExecutor for CountEighFull {
    fn svd(&mut self, _: DenseRead<'_>) -> Result<Vec<DenseTensor>, DenseError> {
        panic!("test only exercises eigh_full")
    }
    fn qr(&mut self, _: DenseRead<'_>) -> Result<Vec<DenseTensor>, DenseError> {
        panic!("test only exercises eigh_full")
    }
    fn eigh(&mut self, input: DenseRead<'_>) -> Result<Vec<DenseTensor>, DenseError> {
        self.calls
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        self.inner.eigh(input)
    }
    fn eigh_into(
        &mut self,
        input: DenseRead<'_>,
        values: DenseWrite<'_>,
        vectors: DenseWrite<'_>,
    ) -> Result<(), DenseError> {
        self.calls
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        self.inner.eigh_into(input, values, vectors)
    }
    fn dot_general_into(
        &mut self,
        _: DenseWrite<'_>,
        _: DenseRead<'_>,
        _: DenseRead<'_>,
        _: &DenseDotConfig,
    ) -> Result<(), DenseError> {
        panic!("test only exercises eigh_full")
    }
}

#[test]
fn compact_diagonal_eigh_full_skips_dense_input_and_solver() {
    let calls = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let runtime = Runtime::builder()
        .with_dense_executor(Box::new(CountEighFull {
            inner: DefaultDenseExecutor::default(),
            calls: Arc::clone(&calls),
        }))
        .build()
        .unwrap();
    let leg = GradedSpace::try_new(Arc::new(U1FusionRule), [(U1Irrep::new(0), 3)]).unwrap();
    let input: TensorMap<_, Complex64> = TensorMap::diagonal(
        &runtime,
        &leg,
        [SectorSpectrum {
            sector: U1Irrep::new(0),
            values: vec![
                Complex64::new(2.0, 0.0),
                Complex64::new(-2.0, 0.0),
                Complex64::new(1.0, 0.0),
            ],
        }],
    )
    .unwrap();
    DIAGONAL_MATERIALIZATIONS.set(0);
    let Eigh { d, v } = input.eigh_full(&[0], &[1]).unwrap();
    assert_eq!(
        d.diagview().unwrap()[0].values,
        vec![
            Complex64::new(-2.0, 0.0),
            Complex64::new(2.0, 0.0),
            Complex64::new(1.0, 0.0)
        ]
    );
    assert_eq!(v.dense_data().unwrap().len(), 9);
    assert_eq!(DIAGONAL_MATERIALIZATIONS.get(), 0);
    assert_eq!(calls.load(std::sync::atomic::Ordering::Relaxed), 0);
    DIAGONAL_MATERIALIZATIONS.set(0);
    let _ = input.eigh_full(&[1], &[0]).unwrap();
    assert_eq!(DIAGONAL_MATERIALIZATIONS.get(), 0);
    assert_eq!(calls.load(std::sync::atomic::Ordering::Relaxed), 0);
}

#[test]
fn compact_diagonal_eigh_full_hand_permutation_all_scalars() {
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let leg = GradedSpace::try_new(Arc::new(U1FusionRule), [(U1Irrep::new(0), 3)]).unwrap();
    macro_rules! check {
        ($dtype:ty, $values:expr, $sorted:expr, $zero:expr, $one:expr) => {{
            let input: TensorMap<_, $dtype> = TensorMap::diagonal(
                &runtime,
                &leg,
                [SectorSpectrum {
                    sector: U1Irrep::new(0),
                    values: $values,
                }],
            )
            .unwrap();
            let saved = input.diagview().unwrap();
            let dense = input.materialize().unwrap().eigh_full(&[0], &[1]).unwrap();
            DIAGONAL_MATERIALIZATIONS.set(0);
            let Eigh { d, v } = input.eigh_full(&[0], &[1]).unwrap();
            assert_eq!(DIAGONAL_MATERIALIZATIONS.get(), 0);
            assert_eq!(input.diagview().unwrap(), saved);
            assert_eq!(d.diagview().unwrap()[0].values, $sorted);
            assert_eq!(d.codomain(), v.domain());
            assert_eq!(d.domain(), v.domain());
            assert_eq!(v.codomain(), input.codomain());
            assert_eq!(v.domain(), dense.v.domain());
            assert!(!v.domain()[0].is_dual());
            let z: $dtype = $zero;
            let o: $dtype = $one;
            assert_eq!(v.dense_data().unwrap(), [z, o, z, o, z, z, z, z, o]);
            assert_eq!(
                input.compose(&v).unwrap().dense_data().unwrap(),
                v.compose(&d).unwrap().dense_data().unwrap()
            );
            assert_eq!(
                v.compose(&d)
                    .unwrap()
                    .compose(&v.adjoint().unwrap())
                    .unwrap()
                    .dense_data()
                    .unwrap(),
                input.materialize().unwrap().dense_data().unwrap()
            );
            assert_eq!(
                v.adjoint()
                    .unwrap()
                    .compose(&v)
                    .unwrap()
                    .dense_data()
                    .unwrap(),
                [o, z, z, z, o, z, z, z, o]
            );
        }};
    }
    check!(
        f32,
        vec![2.0_f32, -2.0, 1.0],
        vec![-2.0_f32, 2.0, 1.0],
        0.0_f32,
        1.0_f32
    );
    check!(
        f64,
        vec![2.0_f64, -2.0, 1.0],
        vec![-2.0_f64, 2.0, 1.0],
        0.0_f64,
        1.0_f64
    );
    check!(
        num_complex::Complex32,
        vec![
            num_complex::Complex32::new(2.0, 0.0),
            num_complex::Complex32::new(-2.0, 0.0),
            num_complex::Complex32::new(1.0, 0.0)
        ],
        vec![
            num_complex::Complex32::new(-2.0, 0.0),
            num_complex::Complex32::new(2.0, 0.0),
            num_complex::Complex32::new(1.0, 0.0)
        ],
        num_complex::Complex32::new(0.0, 0.0),
        num_complex::Complex32::new(1.0, 0.0)
    );
    check!(
        Complex64,
        vec![
            Complex64::new(2.0, 0.0),
            Complex64::new(-2.0, 0.0),
            Complex64::new(1.0, 0.0)
        ],
        vec![
            Complex64::new(-2.0, 0.0),
            Complex64::new(2.0, 0.0),
            Complex64::new(1.0, 0.0)
        ],
        Complex64::new(0.0, 0.0),
        Complex64::new(1.0, 0.0)
    );
}

#[test]
fn compact_diagonal_eigh_full_signed_zeros_remain_a_valid_eigenbasis() {
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let leg = GradedSpace::try_new(Arc::new(U1FusionRule), [(U1Irrep::new(0), 2)]).unwrap();
    let input: TensorMap<_, f64> = TensorMap::diagonal(
        &runtime,
        &leg,
        [SectorSpectrum {
            sector: U1Irrep::new(0),
            values: vec![0.0, -0.0],
        }],
    )
    .unwrap();
    let dense = input.materialize().unwrap().eigh_full(&[0], &[1]).unwrap();
    DIAGONAL_MATERIALIZATIONS.set(0);
    let Eigh { d, v } = input.eigh_full(&[0], &[1]).unwrap();
    assert_eq!(DIAGONAL_MATERIALIZATIONS.get(), 0);
    assert_eq!(
        d.diagview().unwrap()[0].values,
        dense.d.diagview().unwrap()[0].values
    );
    let av = input.compose(&v).unwrap();
    let vd = v.compose(&d).unwrap();
    assert_eq!(av.dense_data().unwrap(), vd.dense_data().unwrap());
    assert_eq!(
        vd.compose(&v.adjoint().unwrap())
            .unwrap()
            .dense_data()
            .unwrap(),
        input.materialize().unwrap().dense_data().unwrap()
    );
    assert_eq!(
        v.adjoint()
            .unwrap()
            .compose(&v)
            .unwrap()
            .dense_data()
            .unwrap(),
        [1.0, 0.0, 0.0, 1.0]
    );
}

#[test]
fn compact_diagonal_eigh_full_preserves_sector_spaces_and_zero_regions() {
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    macro_rules! check {
        ($leg:expr, $spectra:expr, $expected:expr) => {{
            let input: TensorMap<_, f64> = TensorMap::diagonal(&runtime, &$leg, $spectra).unwrap();
            let saved = input.diagview().unwrap();
            let dense = input.materialize().unwrap().eigh_full(&[0], &[1]).unwrap();
            DIAGONAL_MATERIALIZATIONS.set(0);
            let Eigh { d, v } = input.eigh_full(&[0], &[1]).unwrap();
            assert_eq!(DIAGONAL_MATERIALIZATIONS.get(), 0);
            assert_eq!(input.diagview().unwrap(), saved);
            assert_eq!(d.diagview().unwrap(), $expected);
            assert_eq!(v.codomain(), input.codomain());
            assert_eq!(v.domain(), dense.v.domain());
            assert!(!v.domain()[0].is_dual());
            assert_eq!(d.codomain(), v.domain());
            assert_eq!(d.domain(), v.domain());
            let av = input.compose(&v).unwrap();
            let vd = v.compose(&d).unwrap();
            for (&a, &b) in av
                .dense_data()
                .unwrap()
                .iter()
                .zip(vd.dense_data().unwrap())
            {
                assert!((a - b).abs() <= 1e-12);
            }
            let recon = vd.compose(&v.adjoint().unwrap()).unwrap();
            for (&a, &b) in recon
                .dense_data()
                .unwrap()
                .iter()
                .zip(input.materialize().unwrap().dense_data().unwrap())
            {
                assert!((a - b).abs() <= 1e-12);
            }
            let unit = v.adjoint().unwrap().compose(&v).unwrap();
            let bond = v.domain();
            let expected_unit = TensorMap::<_, f64>::from_subblock_fn(
                &runtime,
                [&bond[0]],
                [&bond[0]],
                |_, indices| f64::from(indices[0] == indices[1]),
            )
            .unwrap();
            for (&a, &b) in unit
                .dense_data()
                .unwrap()
                .iter()
                .zip(expected_unit.dense_data().unwrap())
            {
                assert!((a - b).abs() <= 1e-12);
            }
        }};
    }
    let u1 = GradedSpace::try_new(
        Arc::new(U1FusionRule),
        [(U1Irrep::new(0), 3), (U1Irrep::new(1), 2)],
    )
    .unwrap();
    let dual = u1.try_dual().unwrap();
    check!(
        dual,
        [
            SectorSpectrum {
                sector: U1Irrep::new(-1),
                values: vec![2.0, -2.0]
            },
            SectorSpectrum {
                sector: U1Irrep::new(0),
                values: vec![1.0, -4.0, 0.0]
            },
        ],
        vec![
            SectorSpectrum {
                sector: U1Irrep::new(0),
                values: vec![-4.0, 1.0, 0.0]
            },
            SectorSpectrum {
                sector: U1Irrep::new(-1),
                values: vec![-2.0, 2.0]
            },
        ]
    );
    let spin0 = SU2Irrep::from_twice_spin(0);
    let spin_half = SU2Irrep::from_twice_spin(1);
    let su2 = GradedSpace::try_new(Arc::new(SU2FusionRule), [(spin0, 2), (spin_half, 1)]).unwrap();
    check!(
        su2,
        [
            SectorSpectrum {
                sector: spin_half,
                values: vec![-3.0]
            },
            SectorSpectrum {
                sector: spin0,
                values: vec![2.0, -2.0]
            },
        ],
        vec![
            SectorSpectrum {
                sector: spin0,
                values: vec![-2.0, 2.0]
            },
            SectorSpectrum {
                sector: spin_half,
                values: vec![-3.0]
            },
        ]
    );
    let even = product_sector(U1Irrep::new(0), Z2Irrep::EVEN);
    let odd = product_sector(U1Irrep::new(1), Z2Irrep::ODD);
    let fermion = GradedSpace::try_new(
        Arc::new(U1FusionRule.product(FermionParityFusionRule)),
        [(even, 2), (odd, 1)],
    )
    .unwrap();
    check!(
        fermion,
        [
            SectorSpectrum {
                sector: odd,
                values: vec![-3.0]
            },
            SectorSpectrum {
                sector: even,
                values: vec![2.0, -2.0]
            },
        ],
        vec![
            SectorSpectrum {
                sector: even,
                values: vec![-2.0, 2.0]
            },
            SectorSpectrum {
                sector: odd,
                values: vec![-3.0]
            },
        ]
    );
    let zero = GradedSpace::try_new(
        Arc::new(U1FusionRule),
        [(U1Irrep::new(0), 0), (U1Irrep::new(1), 2)],
    )
    .unwrap();
    check!(
        zero,
        [SectorSpectrum {
            sector: U1Irrep::new(1),
            values: vec![2.0, -3.0]
        }],
        vec![SectorSpectrum {
            sector: U1Irrep::new(1),
            values: vec![-3.0, 2.0]
        },]
    );
    let empty = GradedSpace::try_new(Arc::new(U1FusionRule), []).unwrap();
    check!(
        empty,
        Vec::<SectorSpectrum<_, f64>>::new(),
        Vec::<SectorSpectrum<_, f64>>::new()
    );
}

#[test]
fn compact_diagonal_eigh_full_retains_near_hermitian_and_nonfinite_routes() {
    let calls = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let runtime = Runtime::builder()
        .with_dense_executor(Box::new(CountEighFull {
            inner: DefaultDenseExecutor::default(),
            calls: Arc::clone(&calls),
        }))
        .build()
        .unwrap();
    let leg = GradedSpace::try_new(Arc::new(U1FusionRule), [(U1Irrep::new(0), 1)]).unwrap();
    let near: TensorMap<_, Complex64> = TensorMap::diagonal(
        &runtime,
        &leg,
        [SectorSpectrum {
            sector: U1Irrep::new(0),
            values: vec![Complex64::new(1.0, 1e-15)],
        }],
    )
    .unwrap();
    DIAGONAL_MATERIALIZATIONS.set(0);
    let Eigh { d, .. } = near.eigh_full(&[0], &[1]).unwrap();
    assert_eq!(d.diagview().unwrap()[0].values, [Complex64::new(1.0, 0.0)]);
    assert_eq!(DIAGONAL_MATERIALIZATIONS.get(), 1);
    assert_eq!(calls.load(std::sync::atomic::Ordering::Relaxed), 1);

    for value in [f64::NAN, f64::INFINITY] {
        let input: TensorMap<_, f64> = TensorMap::diagonal(
            &runtime,
            &leg,
            [SectorSpectrum {
                sector: U1Irrep::new(0),
                values: vec![value],
            }],
        )
        .unwrap();
        DIAGONAL_MATERIALIZATIONS.set(0);
        let actual = input.eigh_full(&[0], &[1]);
        assert_eq!(DIAGONAL_MATERIALIZATIONS.get(), 1);
        let dense = input.materialize().unwrap().eigh_full(&[0], &[1]);
        assert_eq!(actual.is_err(), dense.is_err());
    }
    let extreme: TensorMap<_, f32> = TensorMap::diagonal(
        &runtime,
        &leg,
        [SectorSpectrum {
            sector: U1Irrep::new(0),
            values: vec![f32::MAX],
        }],
    )
    .unwrap();
    let calls_before = calls.load(std::sync::atomic::Ordering::Relaxed);
    DIAGONAL_MATERIALIZATIONS.set(0);
    assert_eq!(
        extreme.eigh_full(&[0], &[1]).unwrap().d.diagview().unwrap()[0].values,
        [f32::MAX]
    );
    assert_eq!(DIAGONAL_MATERIALIZATIONS.get(), 0);
    assert_eq!(
        calls.load(std::sync::atomic::Ordering::Relaxed),
        calls_before
    );
    let input: TensorMap<_, f64> = TensorMap::diagonal(
        &runtime,
        &leg,
        [SectorSpectrum {
            sector: U1Irrep::new(0),
            values: vec![1.0],
        }],
    )
    .unwrap();
    DIAGONAL_MATERIALIZATIONS.set(0);
    assert!(matches!(
        input.eigh_full(&[0, 1], &[]),
        Err(Error::Operation(error))
            if matches!(error.as_ref(), tenet_tensors::OperationError::UnsupportedTensorContractScope {
                message: "eigh requires an endomorphism (codomain == domain)"
            })
    ));
    assert_eq!(DIAGONAL_MATERIALIZATIONS.get(), 1);
}

#[test]
fn compact_diagonal_eigh_full_changed_roles_match_explicit_dense_permute() {
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let leg = GradedSpace::try_new(
        Arc::new(U1FusionRule),
        [(U1Irrep::new(0), 1), (U1Irrep::new(1), 2)],
    )
    .unwrap()
    .try_dual()
    .unwrap();
    let input: TensorMap<_, f64> = TensorMap::diagonal(
        &runtime,
        &leg,
        [
            SectorSpectrum {
                sector: U1Irrep::new(-1),
                values: vec![-2.0, 3.0],
            },
            SectorSpectrum {
                sector: U1Irrep::new(0),
                values: vec![1.0],
            },
        ],
    )
    .unwrap();
    let swapped = input.permute(&[1], &[0]).unwrap();
    assert_ne!(swapped.diagview().unwrap(), input.diagview().unwrap());
    let reference = swapped
        .materialize()
        .unwrap()
        .eigh_full(&[0], &[1])
        .unwrap();
    DIAGONAL_MATERIALIZATIONS.set(0);
    let Eigh { d, v } = input.eigh_full(&[1], &[0]).unwrap();
    assert_eq!(DIAGONAL_MATERIALIZATIONS.get(), 0);
    assert_eq!(d.diagview().unwrap(), reference.d.diagview().unwrap());
    assert_eq!(d.codomain(), reference.d.codomain());
    assert_eq!(d.domain(), reference.d.domain());
    assert_eq!(v.codomain(), reference.v.codomain());
    assert_eq!(v.domain(), reference.v.domain());
    assert!(!v.domain()[0].is_dual());
    let av = swapped.compose(&v).unwrap();
    let vd = v.compose(&d).unwrap();
    for (&a, &b) in av
        .dense_data()
        .unwrap()
        .iter()
        .zip(vd.dense_data().unwrap())
    {
        assert!((a - b).abs() <= 1e-12);
    }
    let recon = vd.compose(&v.adjoint().unwrap()).unwrap();
    for (&a, &b) in recon
        .dense_data()
        .unwrap()
        .iter()
        .zip(swapped.materialize().unwrap().dense_data().unwrap())
    {
        assert!((a - b).abs() <= 1e-12);
    }
}

#[derive(Default)]
struct CountEigFull {
    inner: DefaultDenseExecutor,
    calls: Arc<std::sync::atomic::AtomicUsize>,
}

impl DenseExecutor for CountEigFull {
    fn svd(&mut self, _: DenseRead<'_>) -> Result<Vec<DenseTensor>, DenseError> {
        panic!("test only exercises eig_full")
    }
    fn qr(&mut self, _: DenseRead<'_>) -> Result<Vec<DenseTensor>, DenseError> {
        panic!("test only exercises eig_full")
    }
    fn eigh(&mut self, _: DenseRead<'_>) -> Result<Vec<DenseTensor>, DenseError> {
        panic!("test only exercises eig_full")
    }
    fn eig(&mut self, input: DenseRead<'_>) -> Result<Vec<DenseTensor>, DenseError> {
        self.calls
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        self.inner.eig(input)
    }
    fn dot_general_into(
        &mut self,
        _: DenseWrite<'_>,
        _: DenseRead<'_>,
        _: DenseRead<'_>,
        _: &DenseDotConfig,
    ) -> Result<(), DenseError> {
        panic!("test only exercises eig_full")
    }
}

#[test]
fn compact_diagonal_eig_full_skips_dense_input_and_solver() {
    let calls = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let runtime = Runtime::builder()
        .with_dense_executor(Box::new(CountEigFull {
            inner: DefaultDenseExecutor::default(),
            calls: Arc::clone(&calls),
        }))
        .build()
        .unwrap();
    let leg = GradedSpace::try_new(Arc::new(U1FusionRule), [(U1Irrep::new(0), 3)]).unwrap();
    let input: TensorMap<_, Complex64> = TensorMap::diagonal(
        &runtime,
        &leg,
        [SectorSpectrum {
            sector: U1Irrep::new(0),
            values: vec![
                Complex64::new(1.0, 1.0),
                Complex64::new(-4.0, 0.0),
                Complex64::new(0.0, 0.0),
            ],
        }],
    )
    .unwrap();
    DIAGONAL_MATERIALIZATIONS.set(0);
    let Eig { d, v } = input.eig_full(&[0], &[1]).unwrap();
    assert_eq!(
        d.diagview().unwrap()[0].values,
        vec![
            Complex64::new(-4.0, 0.0),
            Complex64::new(1.0, 1.0),
            Complex64::new(0.0, 0.0)
        ]
    );
    assert_eq!(v.dense_data().unwrap().len(), 9);
    assert_eq!(DIAGONAL_MATERIALIZATIONS.get(), 0);
    assert_eq!(calls.load(std::sync::atomic::Ordering::Relaxed), 0);
    DIAGONAL_MATERIALIZATIONS.set(0);
    let _ = input.eig_full(&[1], &[0]).unwrap();
    assert_eq!(DIAGONAL_MATERIALIZATIONS.get(), 0);
    assert_eq!(calls.load(std::sync::atomic::Ordering::Relaxed), 0);
}

#[test]
fn compact_diagonal_eig_full_hand_permutation_all_scalars() {
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let leg = GradedSpace::try_new(Arc::new(U1FusionRule), [(U1Irrep::new(0), 3)]).unwrap();
    macro_rules! check {
        ($dtype:ty, $eig:ty, $values:expr, $expected:expr, $zero:expr, $one:expr, $tol:expr, $promote:expr) => {{
            let input: TensorMap<_, $dtype> = TensorMap::diagonal(
                &runtime,
                &leg,
                [SectorSpectrum {
                    sector: U1Irrep::new(0),
                    values: $values,
                }],
            )
            .unwrap();
            let original = input.diagview().unwrap();
            DIAGONAL_MATERIALIZATIONS.set(0);
            let Eig { d, v } = input.eig_full(&[0], &[1]).unwrap();
            assert_eq!(DIAGONAL_MATERIALIZATIONS.get(), 0);
            assert_eq!(input.diagview().unwrap(), original);
            assert_eq!(d.diagview().unwrap()[0].values, $expected);
            assert_eq!(d.codomain(), v.domain());
            assert_eq!(d.domain(), v.domain());
            assert_eq!(v.codomain(), input.codomain());
            assert!(!v.domain()[0].is_dual());
            let z: $eig = $zero;
            let o: $eig = $one;
            assert_eq!(v.dense_data().unwrap(), [z, o, z, o, z, z, z, z, o]);
            let promoted = ($promote)(&input);
            let av = promoted.compose(&v).unwrap();
            let vd = v.compose(&d).unwrap();
            assert_typed_map_close(&av, &vd, $tol);
            assert_typed_map_close(
                &vd.compose(&v.adjoint().unwrap()).unwrap(),
                &promoted.materialize().unwrap(),
                $tol,
            );
            let unit = v.adjoint().unwrap().compose(&v).unwrap();
            assert_eq!(unit.dense_data().unwrap(), [o, z, z, z, o, z, z, z, o]);
        }};
    }
    check!(
        f32,
        num_complex::Complex32,
        vec![1.0_f32, -4.0, 0.25],
        vec![
            num_complex::Complex32::new(-4.0, 0.0),
            num_complex::Complex32::new(1.0, 0.0),
            num_complex::Complex32::new(0.25, 0.0)
        ],
        num_complex::Complex32::new(0.0, 0.0),
        num_complex::Complex32::new(1.0, 0.0),
        1e-5,
        |input: &TensorMap<U1FusionRule, f32>| input.convert::<num_complex::Complex32>()
    );
    check!(
        f64,
        Complex64,
        vec![1.0_f64, -4.0, 0.25],
        vec![
            Complex64::new(-4.0, 0.0),
            Complex64::new(1.0, 0.0),
            Complex64::new(0.25, 0.0)
        ],
        Complex64::new(0.0, 0.0),
        Complex64::new(1.0, 0.0),
        1e-12,
        |input: &TensorMap<U1FusionRule, f64>| input.convert::<Complex64>()
    );
    check!(
        num_complex::Complex32,
        num_complex::Complex32,
        vec![
            num_complex::Complex32::new(1.0, 1.0),
            num_complex::Complex32::new(-4.0, 0.0),
            num_complex::Complex32::new(0.0, 0.25)
        ],
        vec![
            num_complex::Complex32::new(-4.0, 0.0),
            num_complex::Complex32::new(1.0, 1.0),
            num_complex::Complex32::new(0.0, 0.25)
        ],
        num_complex::Complex32::new(0.0, 0.0),
        num_complex::Complex32::new(1.0, 0.0),
        1e-5,
        |input: &TensorMap<U1FusionRule, num_complex::Complex32>| input.clone()
    );
    check!(
        Complex64,
        Complex64,
        vec![
            Complex64::new(1.0, 1.0),
            Complex64::new(-4.0, 0.0),
            Complex64::new(0.0, 0.25)
        ],
        vec![
            Complex64::new(-4.0, 0.0),
            Complex64::new(1.0, 1.0),
            Complex64::new(0.0, 0.25)
        ],
        Complex64::new(0.0, 0.0),
        Complex64::new(1.0, 0.0),
        1e-12,
        |input: &TensorMap<U1FusionRule, Complex64>| input.clone()
    );
}

#[test]
fn compact_diagonal_eig_full_preserves_sector_spaces_and_zero_regions() {
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    macro_rules! check {
        ($leg:expr, $spectra:expr, $expected:expr) => {{
            let input: TensorMap<_, Complex64> =
                TensorMap::diagonal(&runtime, &$leg, $spectra).unwrap();
            let original = input.diagview().unwrap();
            let dense = input.materialize().unwrap().eig_full(&[0], &[1]).unwrap();
            DIAGONAL_MATERIALIZATIONS.set(0);
            let Eig { d, v } = input.eig_full(&[0], &[1]).unwrap();
            assert_eq!(DIAGONAL_MATERIALIZATIONS.get(), 0);
            assert_eq!(input.diagview().unwrap(), original);
            assert_eq!(d.diagview().unwrap(), $expected);
            assert_eq!(d.codomain(), v.domain());
            assert_eq!(d.domain(), v.domain());
            assert_eq!(v.codomain(), input.codomain());
            assert_eq!(v.domain(), dense.v.domain());
            assert!(!v.domain()[0].is_dual());
            let av = input.compose(&v).unwrap();
            let vd = v.compose(&d).unwrap();
            assert_typed_map_close(&av, &vd, 1e-12);
            assert_typed_map_close(
                &vd.compose(&v.adjoint().unwrap()).unwrap(),
                &input.materialize().unwrap(),
                1e-12,
            );
            let unit = v.adjoint().unwrap().compose(&v).unwrap();
            let bond = v.domain();
            let expected_unit = TensorMap::<_, Complex64>::from_subblock_fn(
                &runtime,
                [&bond[0]],
                [&bond[0]],
                |_, indices| Complex64::new(f64::from(indices[0] == indices[1]), 0.0),
            )
            .unwrap();
            assert_typed_map_close(&unit, &expected_unit, 1e-12);
        }};
    }
    let u1 = GradedSpace::try_new(
        Arc::new(U1FusionRule),
        [(U1Irrep::new(0), 3), (U1Irrep::new(1), 2)],
    )
    .unwrap()
    .try_dual()
    .unwrap();
    check!(
        u1,
        [
            SectorSpectrum {
                sector: U1Irrep::new(-1),
                values: vec![Complex64::new(0.0, 2.0), Complex64::new(3.0, 0.0)],
            },
            SectorSpectrum {
                sector: U1Irrep::new(0),
                values: vec![
                    Complex64::new(1.0, 0.0),
                    Complex64::new(0.0, -4.0),
                    Complex64::new(0.0, 0.0)
                ],
            },
        ],
        vec![
            SectorSpectrum {
                sector: U1Irrep::new(0),
                values: vec![
                    Complex64::new(0.0, -4.0),
                    Complex64::new(1.0, 0.0),
                    Complex64::new(0.0, 0.0)
                ],
            },
            SectorSpectrum {
                sector: U1Irrep::new(-1),
                values: vec![Complex64::new(3.0, 0.0), Complex64::new(0.0, 2.0)],
            },
        ]
    );
    let spin0 = SU2Irrep::from_twice_spin(0);
    let spin_half = SU2Irrep::from_twice_spin(1);
    let su2 = GradedSpace::try_new(Arc::new(SU2FusionRule), [(spin0, 2), (spin_half, 1)]).unwrap();
    check!(
        su2,
        [
            SectorSpectrum {
                sector: spin_half,
                values: vec![Complex64::new(0.0, -3.0)]
            },
            SectorSpectrum {
                sector: spin0,
                values: vec![Complex64::new(1.0, 0.0), Complex64::new(2.0, 1.0)]
            },
        ],
        vec![
            SectorSpectrum {
                sector: spin0,
                values: vec![Complex64::new(2.0, 1.0), Complex64::new(1.0, 0.0)]
            },
            SectorSpectrum {
                sector: spin_half,
                values: vec![Complex64::new(0.0, -3.0)]
            },
        ]
    );
    let even = product_sector(U1Irrep::new(0), Z2Irrep::EVEN);
    let odd = product_sector(U1Irrep::new(1), Z2Irrep::ODD);
    let fermion = GradedSpace::try_new(
        Arc::new(U1FusionRule.product(FermionParityFusionRule)),
        [(even, 2), (odd, 1)],
    )
    .unwrap();
    check!(
        fermion,
        [
            SectorSpectrum {
                sector: odd,
                values: vec![Complex64::new(0.0, -3.0)]
            },
            SectorSpectrum {
                sector: even,
                values: vec![Complex64::new(1.0, 0.0), Complex64::new(2.0, 1.0)]
            },
        ],
        vec![
            SectorSpectrum {
                sector: even,
                values: vec![Complex64::new(2.0, 1.0), Complex64::new(1.0, 0.0)]
            },
            SectorSpectrum {
                sector: odd,
                values: vec![Complex64::new(0.0, -3.0)]
            },
        ]
    );
    let zero = GradedSpace::try_new(
        Arc::new(U1FusionRule),
        [(U1Irrep::new(0), 0), (U1Irrep::new(1), 2)],
    )
    .unwrap();
    check!(
        zero,
        [SectorSpectrum {
            sector: U1Irrep::new(1),
            values: vec![Complex64::new(1.0, 0.0), Complex64::new(0.0, -3.0)]
        }],
        vec![SectorSpectrum {
            sector: U1Irrep::new(1),
            values: vec![Complex64::new(0.0, -3.0), Complex64::new(1.0, 0.0)]
        }]
    );
    let empty = GradedSpace::try_new(Arc::new(U1FusionRule), []).unwrap();
    check!(
        empty,
        Vec::<SectorSpectrum<_, Complex64>>::new(),
        Vec::<SectorSpectrum<_, Complex64>>::new()
    );
}

#[test]
fn compact_diagonal_eig_full_changed_roles_match_explicit_dense_permute() {
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let leg = GradedSpace::try_new(
        Arc::new(U1FusionRule),
        [(U1Irrep::new(0), 1), (U1Irrep::new(1), 2)],
    )
    .unwrap()
    .try_dual()
    .unwrap();
    let input: TensorMap<_, Complex64> = TensorMap::diagonal(
        &runtime,
        &leg,
        [
            SectorSpectrum {
                sector: U1Irrep::new(-1),
                values: vec![Complex64::new(-2.0, 0.5), Complex64::new(0.0, 3.0)],
            },
            SectorSpectrum {
                sector: U1Irrep::new(0),
                values: vec![Complex64::new(1.0, 0.0)],
            },
        ],
    )
    .unwrap();
    let swapped = input.permute(&[1], &[0]).unwrap();
    assert_ne!(swapped.diagview().unwrap(), input.diagview().unwrap());
    let reference = swapped.materialize().unwrap().eig_full(&[0], &[1]).unwrap();
    DIAGONAL_MATERIALIZATIONS.set(0);
    let Eig { d, v } = input.eig_full(&[1], &[0]).unwrap();
    assert_eq!(DIAGONAL_MATERIALIZATIONS.get(), 0);
    assert_eq!(d.diagview().unwrap(), reference.d.diagview().unwrap());
    assert_eq!(d.codomain(), reference.d.codomain());
    assert_eq!(d.domain(), reference.d.domain());
    assert_eq!(v.codomain(), reference.v.codomain());
    assert_eq!(v.domain(), reference.v.domain());
    assert!(!v.domain()[0].is_dual());
    let vd = v.compose(&d).unwrap();
    assert_typed_map_close(&swapped.compose(&v).unwrap(), &vd, 1e-12);
    assert_typed_map_close(
        &vd.compose(&v.adjoint().unwrap()).unwrap(),
        &swapped.materialize().unwrap(),
        1e-12,
    );
}

#[test]
fn compact_diagonal_eig_full_ties_and_signed_zeros_are_valid() {
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let leg = GradedSpace::try_new(Arc::new(U1FusionRule), [(U1Irrep::new(0), 4)]).unwrap();
    let input: TensorMap<_, Complex64> = TensorMap::diagonal(
        &runtime,
        &leg,
        [SectorSpectrum {
            sector: U1Irrep::new(0),
            values: vec![
                Complex64::new(0.0, 2.0),
                Complex64::new(0.0, -2.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(-0.0, 0.0),
            ],
        }],
    )
    .unwrap();
    DIAGONAL_MATERIALIZATIONS.set(0);
    let Eig { d, v } = input.eig_full(&[0], &[1]).unwrap();
    assert_eq!(DIAGONAL_MATERIALIZATIONS.get(), 0);
    let values = &d.diagview().unwrap()[0].values;
    assert_eq!(values.len(), 4);
    assert!(values
        .windows(2)
        .all(|pair| pair[0].norm() >= pair[1].norm()));
    assert_eq!(
        values
            .iter()
            .filter(|&&value| value == Complex64::new(0.0, 0.0))
            .count(),
        2
    );
    assert!(values.contains(&Complex64::new(0.0, 2.0)));
    assert!(values.contains(&Complex64::new(0.0, -2.0)));
    let vd = v.compose(&d).unwrap();
    assert_typed_map_close(&input.compose(&v).unwrap(), &vd, 1e-12);
    assert_typed_map_close(
        &vd.compose(&v.adjoint().unwrap()).unwrap(),
        &input.materialize().unwrap(),
        1e-12,
    );
    let unit = v.adjoint().unwrap().compose(&v).unwrap();
    assert_eq!(
        unit.dense_data()
            .unwrap()
            .iter()
            .filter(|&&x| x == Complex64::new(1.0, 0.0))
            .count(),
        4
    );
    assert_eq!(
        unit.dense_data()
            .unwrap()
            .iter()
            .filter(|&&x| x == Complex64::new(0.0, 0.0))
            .count(),
        12
    );
}

#[test]
fn compact_diagonal_eig_full_retains_nonfinite_and_widened_norm_fallbacks() {
    let calls = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let runtime = Runtime::builder()
        .with_dense_executor(Box::new(CountEigFull {
            inner: DefaultDenseExecutor::default(),
            calls: Arc::clone(&calls),
        }))
        .build()
        .unwrap();
    let leg = GradedSpace::try_new(Arc::new(U1FusionRule), [(U1Irrep::new(0), 1)]).unwrap();
    for value in [
        Complex64::new(f64::NAN, 0.0),
        Complex64::new(f64::INFINITY, 0.0),
        Complex64::new(f64::MAX, f64::MAX),
    ] {
        let input: TensorMap<_, Complex64> = TensorMap::diagonal(
            &runtime,
            &leg,
            [SectorSpectrum {
                sector: U1Irrep::new(0),
                values: vec![value],
            }],
        )
        .unwrap();
        let before = calls.load(std::sync::atomic::Ordering::Relaxed);
        DIAGONAL_MATERIALIZATIONS.set(0);
        let _ = input.eig_full(&[0], &[1]);
        assert_eq!(DIAGONAL_MATERIALIZATIONS.get(), 1);
        assert_eq!(calls.load(std::sync::atomic::Ordering::Relaxed), before + 1);
    }
    let input: TensorMap<_, f32> = TensorMap::diagonal(
        &runtime,
        &leg,
        [SectorSpectrum {
            sector: U1Irrep::new(0),
            values: vec![f32::MAX],
        }],
    )
    .unwrap();
    let before = calls.load(std::sync::atomic::Ordering::Relaxed);
    DIAGONAL_MATERIALIZATIONS.set(0);
    assert_eq!(
        input.eig_full(&[0], &[1]).unwrap().d.diagview().unwrap()[0].values,
        [num_complex::Complex32::new(f32::MAX, 0.0)]
    );
    assert_eq!(DIAGONAL_MATERIALIZATIONS.get(), 0);
    assert_eq!(calls.load(std::sync::atomic::Ordering::Relaxed), before);
    DIAGONAL_MATERIALIZATIONS.set(0);
    assert!(matches!(
        input.eig_full(&[0, 1], &[]),
        Err(Error::Operation(error))
            if matches!(error.as_ref(), tenet_tensors::OperationError::UnsupportedTensorContractScope {
                message: "eig requires an endomorphism (codomain == domain)"
            })
    ));
    assert_eq!(DIAGONAL_MATERIALIZATIONS.get(), 1);
}

#[derive(Default)]
struct CountEigVals {
    inner: DefaultDenseExecutor,
    calls: Arc<std::sync::atomic::AtomicUsize>,
}

impl DenseExecutor for CountEigVals {
    fn svd(&mut self, _: DenseRead<'_>) -> Result<Vec<DenseTensor>, DenseError> {
        panic!("test only exercises eig_vals")
    }
    fn qr(&mut self, _: DenseRead<'_>) -> Result<Vec<DenseTensor>, DenseError> {
        panic!("test only exercises eig_vals")
    }
    fn eigh(&mut self, _: DenseRead<'_>) -> Result<Vec<DenseTensor>, DenseError> {
        panic!("test only exercises eig_vals")
    }
    fn eig_vals(&mut self, input: DenseRead<'_>) -> Result<DenseTensor, DenseError> {
        self.calls
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        self.inner.eig_vals(input)
    }
    fn dot_general_into(
        &mut self,
        _: DenseWrite<'_>,
        _: DenseRead<'_>,
        _: DenseRead<'_>,
        _: &DenseDotConfig,
    ) -> Result<(), DenseError> {
        panic!("test only exercises eig_vals")
    }
}

#[test]
fn compact_diagonal_eig_vals_skips_dense_input_and_solver() {
    let calls = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let runtime = Runtime::builder()
        .with_dense_executor(Box::new(CountEigVals {
            inner: DefaultDenseExecutor::default(),
            calls: Arc::clone(&calls),
        }))
        .build()
        .unwrap();
    let leg = GradedSpace::try_new(
        Arc::new(U1FusionRule),
        [(U1Irrep::new(0), 3), (U1Irrep::new(1), 2)],
    )
    .unwrap();
    let input: TensorMap<_, Complex64> = TensorMap::diagonal(
        &runtime,
        &leg,
        [
            SectorSpectrum {
                sector: U1Irrep::new(1),
                values: vec![Complex64::new(0.0, 2.0), Complex64::new(-2.0, 0.0)],
            },
            SectorSpectrum {
                sector: U1Irrep::new(0),
                values: vec![
                    Complex64::new(1.0, 1.0),
                    Complex64::new(-4.0, 0.0),
                    Complex64::new(0.0, 0.0),
                ],
            },
        ],
    )
    .unwrap();
    let saved = input.diagview().unwrap();
    DIAGONAL_MATERIALIZATIONS.set(0);
    let result = input.eig_vals(&[0], &[1]).unwrap();
    assert_eq!(
        result[0].values,
        vec![
            Complex64::new(-4.0, 0.0),
            Complex64::new(1.0, 1.0),
            Complex64::new(0.0, 0.0)
        ]
    );
    assert_eq!(
        result[1]
            .values
            .iter()
            .map(|v| v.norm())
            .collect::<Vec<_>>(),
        vec![2.0, 2.0]
    );
    assert_eq!(DIAGONAL_MATERIALIZATIONS.get(), 0);
    assert_eq!(calls.load(std::sync::atomic::Ordering::Relaxed), 0);
    assert_eq!(input.diagview().unwrap(), saved);
}

#[test]
fn compact_diagonal_eig_vals_match_hand_spectra_across_scalars_and_sectors() {
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    macro_rules! check {
        ($leg:expr, $spectra:expr, $dtype:ty, $expected:expr, $tol:expr) => {{
            let input: TensorMap<_, $dtype> =
                TensorMap::diagonal(&runtime, &$leg, $spectra).unwrap();
            let saved = input.diagview().unwrap();
            let dense_values = input.materialize().unwrap().eig_vals(&[0], &[1]).unwrap();
            DIAGONAL_MATERIALIZATIONS.set(0);
            let actual = input.eig_vals(&[0], &[1]).unwrap();
            assert_eq!(DIAGONAL_MATERIALIZATIONS.get(), 0);
            assert_eq!(input.diagview().unwrap(), saved);
            let expected: Vec<Vec<Complex64>> = $expected;
            assert_eq!(actual.len(), expected.len());
            for ((actual, dense), expected) in actual.iter().zip(&dense_values).zip(expected) {
                assert_eq!(actual.sector, dense.sector);
                assert_eq!(actual.values.len(), expected.len());
                let mut actual_values = actual.values.clone();
                let mut dense_values = dense.values.clone();
                let mut expected_values = expected;
                let order = |a: &Complex64, b: &Complex64| {
                    a.re.total_cmp(&b.re).then(a.im.total_cmp(&b.im))
                };
                actual_values.sort_by(order);
                dense_values.sort_by(order);
                expected_values.sort_by(order);
                assert_eq!(actual_values, expected_values);
                for ((actual, dense), expected) in actual_values
                    .iter()
                    .zip(&dense_values)
                    .zip(&expected_values)
                {
                    assert_eq!(actual, expected);
                    assert!((*actual - *dense).norm() <= $tol);
                }
                assert!(actual
                    .values
                    .windows(2)
                    .all(|pair| pair[0].norm() >= pair[1].norm()));
            }
        }};
    }
    let u1 = GradedSpace::try_new(
        Arc::new(U1FusionRule),
        [(U1Irrep::new(0), 3), (U1Irrep::new(1), 2)],
    )
    .unwrap();
    let real32 = [1.0000001_f32, -3.25, 0.3];
    check!(
        u1,
        [
            SectorSpectrum {
                sector: U1Irrep::new(1),
                values: vec![2.0_f32, -2.0]
            },
            SectorSpectrum {
                sector: U1Irrep::new(0),
                values: real32.to_vec()
            },
        ],
        f32,
        vec![
            real32.map(|v| Complex64::new(v as f64, 0.0)).to_vec(),
            vec![Complex64::new(2.0, 0.0), Complex64::new(-2.0, 0.0)],
        ],
        1e-5
    );
    let complex32 = [
        num_complex::Complex32::new(1.0000001, 0.25),
        num_complex::Complex32::new(-3.25, 0.0),
        num_complex::Complex32::new(0.0, -0.3),
    ];
    check!(
        u1,
        [
            SectorSpectrum {
                sector: U1Irrep::new(1),
                values: vec![
                    num_complex::Complex32::new(0.0, 2.0),
                    num_complex::Complex32::new(-2.0, 0.0)
                ]
            },
            SectorSpectrum {
                sector: U1Irrep::new(0),
                values: complex32.to_vec()
            },
        ],
        num_complex::Complex32,
        vec![
            complex32
                .map(|v| Complex64::new(v.re as f64, v.im as f64))
                .to_vec(),
            vec![Complex64::new(0.0, 2.0), Complex64::new(-2.0, 0.0)],
        ],
        1e-5
    );
    let dual = u1.try_dual().unwrap();
    check!(
        dual,
        [
            SectorSpectrum {
                sector: U1Irrep::new(-1),
                values: vec![2.0_f64, -2.0]
            },
            SectorSpectrum {
                sector: U1Irrep::new(0),
                values: vec![1.5, -4.0, 0.0]
            },
        ],
        f64,
        vec![
            vec![Complex64::new(2.0, 0.0), Complex64::new(-2.0, 0.0)],
            vec![
                Complex64::new(1.5, 0.0),
                Complex64::new(-4.0, 0.0),
                Complex64::new(0.0, 0.0)
            ],
        ],
        1e-12
    );
    let spin0 = SU2Irrep::from_twice_spin(0);
    let spin_half = SU2Irrep::from_twice_spin(1);
    let su2 = GradedSpace::try_new(Arc::new(SU2FusionRule), [(spin0, 2), (spin_half, 1)]).unwrap();
    check!(
        su2,
        [
            SectorSpectrum {
                sector: spin_half,
                values: vec![Complex64::new(0.0, -3.0)]
            },
            SectorSpectrum {
                sector: spin0,
                values: vec![Complex64::new(2.0, 1.0), Complex64::new(-2.0, -1.0)]
            },
        ],
        Complex64,
        vec![
            vec![Complex64::new(2.0, 1.0), Complex64::new(-2.0, -1.0)],
            vec![Complex64::new(0.0, -3.0)],
        ],
        1e-12
    );
    let even = product_sector(U1Irrep::new(0), Z2Irrep::EVEN);
    let odd = product_sector(U1Irrep::new(1), Z2Irrep::ODD);
    let fermion = GradedSpace::try_new(
        Arc::new(U1FusionRule.product(FermionParityFusionRule)),
        [(even, 2), (odd, 1)],
    )
    .unwrap();
    check!(
        fermion,
        [
            SectorSpectrum {
                sector: odd,
                values: vec![Complex64::new(0.0, -3.0)]
            },
            SectorSpectrum {
                sector: even,
                values: vec![Complex64::new(2.0, 1.0), Complex64::new(-2.0, -1.0)]
            },
        ],
        Complex64,
        vec![
            vec![Complex64::new(2.0, 1.0), Complex64::new(-2.0, -1.0)],
            vec![Complex64::new(0.0, -3.0)],
        ],
        1e-12
    );
    let zero = GradedSpace::try_new(
        Arc::new(U1FusionRule),
        [(U1Irrep::new(0), 0), (U1Irrep::new(1), 2)],
    )
    .unwrap();
    check!(
        zero,
        [SectorSpectrum {
            sector: U1Irrep::new(1),
            values: vec![2.0_f64, -3.0]
        }],
        f64,
        vec![vec![Complex64::new(2.0, 0.0), Complex64::new(-3.0, 0.0)]],
        1e-12
    );
    let empty = GradedSpace::try_new(Arc::new(U1FusionRule), []).unwrap();
    let input: TensorMap<_, f64> =
        TensorMap::diagonal(&runtime, &empty, Vec::<SectorSpectrum<_, f64>>::new()).unwrap();
    DIAGONAL_MATERIALIZATIONS.set(0);
    assert!(input.eig_vals(&[0], &[1]).unwrap().is_empty());
    assert_eq!(DIAGONAL_MATERIALIZATIONS.get(), 0);
}

#[test]
fn compact_diagonal_eig_vals_retains_dense_fallback() {
    let calls = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let runtime = Runtime::builder()
        .with_dense_executor(Box::new(CountEigVals {
            inner: DefaultDenseExecutor::default(),
            calls: Arc::clone(&calls),
        }))
        .build()
        .unwrap();
    let leg = GradedSpace::try_new(Arc::new(U1FusionRule), [(U1Irrep::new(0), 1)]).unwrap();
    for value in [
        Complex64::new(f64::NAN, 0.0),
        Complex64::new(f64::INFINITY, 0.0),
        Complex64::new(f64::MAX, f64::MAX),
    ] {
        let input: TensorMap<_, Complex64> = TensorMap::diagonal(
            &runtime,
            &leg,
            [SectorSpectrum {
                sector: U1Irrep::new(0),
                values: vec![value],
            }],
        )
        .unwrap();
        DIAGONAL_MATERIALIZATIONS.set(0);
        let _ = input.eig_vals(&[0], &[1]);
        assert_eq!(DIAGONAL_MATERIALIZATIONS.get(), 1);
    }
    assert_eq!(calls.load(std::sync::atomic::Ordering::Relaxed), 3);
    let input: TensorMap<_, f64> = TensorMap::diagonal(
        &runtime,
        &leg,
        [SectorSpectrum {
            sector: U1Irrep::new(0),
            values: vec![1.0],
        }],
    )
    .unwrap();
    DIAGONAL_MATERIALIZATIONS.set(0);
    let error = input.eig_vals(&[0, 1], &[]).unwrap_err();
    assert!(matches!(
        error,
        Error::Operation(error)
            if matches!(
                error.as_ref(),
                tenet_tensors::OperationError::UnsupportedTensorContractScope {
                    message: "eig requires an endomorphism (codomain == domain)"
                }
            )
    ));
    assert_eq!(DIAGONAL_MATERIALIZATIONS.get(), 1);
}

#[test]
fn compact_diagonal_eigh_vals_skips_dense_input_and_solver() {
    let calls = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let runtime = Runtime::builder()
        .with_dense_executor(Box::new(CountEighVals {
            inner: DefaultDenseExecutor::default(),
            calls: Arc::clone(&calls),
        }))
        .build()
        .unwrap();
    let leg = GradedSpace::try_new(
        Arc::new(U1FusionRule),
        [(U1Irrep::new(0), 3), (U1Irrep::new(1), 2)],
    )
    .unwrap();
    let source: TensorMap<_, f64> = TensorMap::diagonal(
        &runtime,
        &leg,
        [
            SectorSpectrum {
                sector: U1Irrep::new(1),
                values: vec![2.0, -2.0],
            },
            SectorSpectrum {
                sector: U1Irrep::new(0),
                values: vec![1.0, -4.0, 0.0],
            },
        ],
    )
    .unwrap();
    DIAGONAL_MATERIALIZATIONS.set(0);
    let result = source.eigh_vals(&[0], &[1]).unwrap();
    assert_eq!(result[0].values, vec![-4.0, 1.0, 0.0]);
    assert_eq!(result[1].values, vec![-2.0, 2.0]);
    assert_eq!(DIAGONAL_MATERIALIZATIONS.get(), 0);
    assert_eq!(calls.load(std::sync::atomic::Ordering::Relaxed), 0);
}

#[test]
fn compact_diagonal_eigh_vals_match_hand_spectra_across_scalars_and_sectors() {
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    macro_rules! check {
        ($leg:expr, $spectra:expr, $dtype:ty, $expected:expr, $tol:expr) => {{
            let input: TensorMap<_, $dtype> =
                TensorMap::diagonal(&runtime, &$leg, $spectra).unwrap();
            let saved = input.diagview().unwrap();
            let dense = input.materialize().unwrap();
            let dense_values = dense.eigh_vals(&[0], &[1]).unwrap();
            DIAGONAL_MATERIALIZATIONS.set(0);
            let actual = input.eigh_vals(&[0], &[1]).unwrap();
            assert_eq!(DIAGONAL_MATERIALIZATIONS.get(), 0);
            assert_eq!(input.diagview().unwrap(), saved);
            let expected: Vec<Vec<f64>> = $expected;
            assert_eq!(actual.len(), expected.len());
            for ((actual, dense), expected) in actual.iter().zip(&dense_values).zip(expected) {
                assert_eq!(actual.sector, dense.sector);
                assert_eq!(actual.values.len(), expected.len());
                for ((&actual, &dense), expected) in
                    actual.values.iter().zip(&dense.values).zip(expected)
                {
                    assert!((actual - expected).abs() <= $tol);
                    assert!((actual - dense).abs() <= $tol);
                }
            }
        }};
    }
    let u1 = GradedSpace::try_new(
        Arc::new(U1FusionRule),
        [(U1Irrep::new(0), 3), (U1Irrep::new(1), 2)],
    )
    .unwrap();
    check!(
        u1,
        [
            SectorSpectrum {
                sector: U1Irrep::new(1),
                values: vec![2.0_f32, -2.0]
            },
            SectorSpectrum {
                sector: U1Irrep::new(0),
                values: vec![1.0, -4.0, 0.0]
            },
        ],
        f32,
        vec![vec![-4.0, 1.0, 0.0], vec![-2.0, 2.0]],
        1e-5
    );
    check!(
        u1,
        [
            SectorSpectrum {
                sector: U1Irrep::new(1),
                values: vec![
                    num_complex::Complex32::new(2.0, 0.0),
                    num_complex::Complex32::new(-2.0, 0.0)
                ]
            },
            SectorSpectrum {
                sector: U1Irrep::new(0),
                values: vec![
                    num_complex::Complex32::new(1.0, 0.0),
                    num_complex::Complex32::new(-4.0, 0.0),
                    num_complex::Complex32::new(0.0, 0.0)
                ]
            },
        ],
        num_complex::Complex32,
        vec![vec![-4.0, 1.0, 0.0], vec![-2.0, 2.0]],
        1e-5
    );
    let dual = u1.try_dual().unwrap();
    check!(
        dual,
        [
            SectorSpectrum {
                sector: U1Irrep::new(-1),
                values: vec![2.0_f64, -2.0]
            },
            SectorSpectrum {
                sector: U1Irrep::new(0),
                values: vec![1.0, -4.0, 0.0]
            },
        ],
        f64,
        vec![vec![-2.0, 2.0], vec![-4.0, 1.0, 0.0]],
        1e-12
    );
    let spin0 = SU2Irrep::from_twice_spin(0);
    let spin_half = SU2Irrep::from_twice_spin(1);
    let su2 = GradedSpace::try_new(Arc::new(SU2FusionRule), [(spin0, 2), (spin_half, 1)]).unwrap();
    check!(
        su2,
        [
            SectorSpectrum {
                sector: spin_half,
                values: vec![Complex64::new(-3.0, 0.0)]
            },
            SectorSpectrum {
                sector: spin0,
                values: vec![Complex64::new(2.0, 0.0), Complex64::new(-2.0, 0.0)]
            },
        ],
        Complex64,
        vec![vec![-2.0, 2.0], vec![-3.0]],
        1e-12
    );
    let even = product_sector(U1Irrep::new(0), Z2Irrep::EVEN);
    let odd = product_sector(U1Irrep::new(1), Z2Irrep::ODD);
    let fermion = GradedSpace::try_new(
        Arc::new(U1FusionRule.product(FermionParityFusionRule)),
        [(even, 2), (odd, 1)],
    )
    .unwrap();
    check!(
        fermion,
        [
            SectorSpectrum {
                sector: odd,
                values: vec![-3.0_f64]
            },
            SectorSpectrum {
                sector: even,
                values: vec![2.0, -2.0]
            },
        ],
        f64,
        vec![vec![-2.0, 2.0], vec![-3.0]],
        1e-12
    );
    let zero_sector = GradedSpace::try_new(
        Arc::new(U1FusionRule),
        [(U1Irrep::new(0), 0), (U1Irrep::new(1), 2)],
    )
    .unwrap();
    check!(
        zero_sector,
        [SectorSpectrum {
            sector: U1Irrep::new(1),
            values: vec![2.0_f64, -3.0]
        }],
        f64,
        vec![vec![-3.0, 2.0]],
        1e-12
    );
    let empty = GradedSpace::try_new(Arc::new(U1FusionRule), []).unwrap();
    let input: TensorMap<_, f64> =
        TensorMap::diagonal(&runtime, &empty, Vec::<SectorSpectrum<_, f64>>::new()).unwrap();
    DIAGONAL_MATERIALIZATIONS.set(0);
    assert!(input.eigh_vals(&[0], &[1]).unwrap().is_empty());
    assert_eq!(DIAGONAL_MATERIALIZATIONS.get(), 0);
}

#[test]
fn compact_diagonal_eigh_vals_preserves_complex_dense_fallback() {
    let calls = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let runtime = Runtime::builder()
        .with_dense_executor(Box::new(CountEighVals {
            inner: DefaultDenseExecutor::default(),
            calls: Arc::clone(&calls),
        }))
        .build()
        .unwrap();
    let leg = GradedSpace::try_new(Arc::new(U1FusionRule), [(U1Irrep::new(0), 1)]).unwrap();
    let near: TensorMap<_, Complex64> = TensorMap::diagonal(
        &runtime,
        &leg,
        [SectorSpectrum {
            sector: U1Irrep::new(0),
            values: vec![Complex64::new(1.0, 1e-15)],
        }],
    )
    .unwrap();
    DIAGONAL_MATERIALIZATIONS.set(0);
    assert_eq!(near.eigh_vals(&[0], &[1]).unwrap()[0].values, [1.0]);
    assert_eq!(DIAGONAL_MATERIALIZATIONS.get(), 1);
    assert_eq!(calls.load(std::sync::atomic::Ordering::Relaxed), 1);

    let nonfinite: TensorMap<_, f64> = TensorMap::diagonal(
        &runtime,
        &leg,
        [SectorSpectrum {
            sector: U1Irrep::new(0),
            values: vec![f64::NAN],
        }],
    )
    .unwrap();
    DIAGONAL_MATERIALIZATIONS.set(0);
    assert!(nonfinite.eigh_vals(&[0], &[1]).is_err());
    assert_eq!(DIAGONAL_MATERIALIZATIONS.get(), 1);
}

#[test]
fn compact_diagonal_eigh_vals_widens_stored_single_precision_values() {
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let leg = GradedSpace::try_new(Arc::new(U1FusionRule), [(U1Irrep::new(0), 3)]).unwrap();
    let values = [1.0000001_f32, -1.0000002_f32, 0.3_f32];
    let expected = vec![values[1] as f64, values[0] as f64, values[2] as f64];
    let real: TensorMap<_, f32> = TensorMap::diagonal(
        &runtime,
        &leg,
        [SectorSpectrum {
            sector: U1Irrep::new(0),
            values: values.to_vec(),
        }],
    )
    .unwrap();
    let complex: TensorMap<_, num_complex::Complex32> = TensorMap::diagonal(
        &runtime,
        &leg,
        [SectorSpectrum {
            sector: U1Irrep::new(0),
            values: values
                .map(|value| num_complex::Complex32::new(value, 0.0))
                .to_vec(),
        }],
    )
    .unwrap();
    DIAGONAL_MATERIALIZATIONS.set(0);
    for actual in [real.eigh_vals(&[0], &[1]), complex.eigh_vals(&[0], &[1])] {
        assert_eq!(actual.unwrap()[0].values, expected);
    }
    assert_eq!(DIAGONAL_MATERIALIZATIONS.get(), 0);
}

#[test]
fn compact_diagonal_svd_full_preserves_spaces_without_materialization_or_solver() {
    let calls = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let runtime = Runtime::builder()
        .dense_threads(1)
        .with_dense_executor(Box::new(CountPolarKernels {
            svd_calls: Arc::clone(&calls),
            ..Default::default()
        }))
        .build()
        .unwrap();
    let leg = GradedSpace::try_new(Arc::new(U1FusionRule), [(U1Irrep::new(2), 3)])
        .unwrap()
        .try_dual()
        .unwrap();
    let input: TensorMap<_, f64> = TensorMap::diagonal(
        &runtime,
        &leg,
        [SectorSpectrum {
            sector: U1Irrep::new(-2),
            values: vec![0.0, -4.0, 2.0],
        }],
    )
    .unwrap();
    let w = GradedSpace::try_new(Arc::new(U1FusionRule), [(U1Irrep::new(-2), 3)]).unwrap();
    let dense = input.materialize().unwrap();
    let old = dense.svd_full(&[0], &[1]).unwrap();
    assert!(calls.load(std::sync::atomic::Ordering::Relaxed) > 0);
    calls.store(0, std::sync::atomic::Ordering::Relaxed);
    DIAGONAL_MATERIALIZATIONS.set(0);
    let out = input.svd_full(&[0], &[1]).unwrap();
    assert_eq!(DIAGONAL_MATERIALIZATIONS.get(), 0);
    assert_eq!(calls.load(std::sync::atomic::Ordering::Relaxed), 0);
    assert_eq!(out.u.codomain(), input.codomain());
    assert_eq!(out.vh.domain(), input.domain());
    assert_eq!(out.u.domain(), vec![w.clone()]);
    assert_eq!(out.s.codomain(), vec![w.clone()]);
    assert_eq!(out.s.domain(), vec![w.clone()]);
    assert_eq!(out.vh.codomain(), vec![w]);
    for (actual, expected) in [(&out.u, &old.u), (&out.s, &old.s), (&out.vh, &old.vh)] {
        assert_eq!(actual.codomain(), expected.codomain());
        assert_eq!(actual.domain(), expected.domain());
    }
    assert!(out.u.dense_data().is_ok());
    assert!(out.vh.dense_data().is_ok());
    assert!(out.s.dense_data().is_err());
    assert_eq!(out.s.diagview().unwrap()[0].values, vec![4.0, 2.0, 0.0]);
    assert_eq!(
        out.s.diagonal_spectrum().unwrap().unwrap(),
        out.s.diagview().unwrap()
    );
    let rebuilt = out.u.compose(&out.s).unwrap().compose(&out.vh).unwrap();
    assert_eq!(rebuilt.dense_data().unwrap(), dense.dense_data().unwrap());
    for (gram, identity) in [
        (
            out.u.adjoint().unwrap().compose(&out.u).unwrap(),
            TensorMap::isomorphism(&runtime, &out.u.domain(), &out.u.domain()).unwrap(),
        ),
        (
            out.vh.compose(&out.vh.adjoint().unwrap()).unwrap(),
            TensorMap::isomorphism(&runtime, &out.vh.codomain(), &out.vh.codomain()).unwrap(),
        ),
    ] {
        assert!(gram.axpby(1.0, &identity, -1.0).unwrap().norm(2.0).unwrap() < 1e-12);
    }
    assert_eq!(
        out.s.materialize().unwrap().dense_data().unwrap(),
        &[4.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0]
    );
}

#[test]
fn full_svd_compact_bond_requires_complete_exact_nondual_sector_legs() {
    let a = tenet_core::SectorId::new(1);
    let b = tenet_core::SectorId::new(2);
    let row = tenet_core::SectorLeg::new([(a, 2), (b, 3)], false);
    let exact = vec![
        tenet_matrixalgebra::SectorSpectrum {
            sector: a,
            values: vec![4.0, 1.0],
        },
        tenet_matrixalgebra::SectorSpectrum {
            sector: b,
            values: vec![3.0, 0.0, 0.0],
        },
    ];
    assert!(full_svd_spectrum_matches_bonds(&row, &row, &exact));
    let same_total = tenet_core::SectorLeg::new([(a, 3), (b, 2)], false);
    assert!(!full_svd_spectrum_matches_bonds(&row, &same_total, &exact));
    assert!(!full_svd_spectrum_matches_bonds(&row, &row, &exact[..1]));
    let dual = tenet_core::SectorLeg::new([(a, 2), (b, 3)], true);
    assert!(!full_svd_spectrum_matches_bonds(&dual, &dual, &exact));
    let duplicate = [exact[0].clone(), exact[0].clone()];
    assert!(!full_svd_spectrum_matches_bonds(&row, &row, &duplicate));
}

#[test]
fn dense_multisector_full_svd_publishes_compact_s_for_real_and_complex() {
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let leg = GradedSpace::try_new(
        Arc::new(U1FusionRule),
        [(U1Irrep::new(0), 2), (U1Irrep::new(1), 3)],
    )
    .unwrap();
    let input: TensorMap<_, f64> =
        TensorMap::from_subblock_fn(&runtime, [&leg], [&leg], |_, indices| {
            if indices[0] == indices[1] {
                (indices[0] + 2) as f64
            } else {
                0.25
            }
        })
        .unwrap();
    let Svd { u, s, vh } = input.svd_full(&[0], &[1]).unwrap();
    assert!(s.dense_data().is_err());
    let values = s.diagview().unwrap();
    assert_eq!(values.len(), 2);
    assert_eq!(values[0].sector, U1Irrep::new(0));
    assert_eq!(values[1].sector, U1Irrep::new(1));
    assert_eq!(values[0].values.len(), 2);
    assert_eq!(values[1].values.len(), 3);
    assert!(values
        .iter()
        .all(|entry| entry.values.windows(2).all(|pair| pair[0] >= pair[1])));
    assert_eq!(
        s.materialize().unwrap().diagview().unwrap(),
        s.diagview().unwrap()
    );
    let rebuilt = u.compose(&s).unwrap().compose(&vh).unwrap();
    assert!(rebuilt
        .dense_data()
        .unwrap()
        .iter()
        .zip(input.dense_data().unwrap())
        .all(|(actual, expected)| (actual - expected).abs() < 1.0e-12));
    assert!(input.svd_full(&[1], &[0]).unwrap().s.dense_data().is_err());

    let complex = input.convert::<Complex64>().scale(Complex64::new(1.0, 0.5));
    let Svd { u, s, vh } = complex.svd_full(&[0], &[1]).unwrap();
    assert!(s.dense_data().is_err());
    assert_eq!(s.diagview().unwrap().len(), 2);
    let rebuilt = u.compose(&s).unwrap().compose(&vh).unwrap();
    assert!(rebuilt
        .dense_data()
        .unwrap()
        .iter()
        .zip(complex.dense_data().unwrap())
        .all(|(actual, expected)| (*actual - *expected).norm() < 1.0e-12));
}

#[test]
fn compact_diagonal_null_uses_coordinate_kernel_without_dense_solver() {
    let calls = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let runtime = Runtime::builder()
        .with_dense_executor(Box::new(CountPolarKernels {
            svd_calls: Arc::clone(&calls),
            ..Default::default()
        }))
        .build()
        .unwrap();
    let leg = GradedSpace::try_new(
        Arc::new(U1FusionRule),
        [(U1Irrep::new(0), 3), (U1Irrep::new(1), 2)],
    )
    .unwrap();
    let input: TensorMap<_, f64> = TensorMap::diagonal(
        &runtime,
        &leg,
        [
            SectorSpectrum {
                sector: U1Irrep::new(0),
                values: vec![3.0, 0.0, -2.0],
            },
            SectorSpectrum {
                sector: U1Irrep::new(1),
                values: vec![0.0, 0.0],
            },
        ],
    )
    .unwrap();
    DIAGONAL_MATERIALIZATIONS.set(0);
    let left = input.left_null(&[0], &[1]).unwrap();
    let right = input.right_null(&[0], &[1]).unwrap();
    assert_eq!(DIAGONAL_MATERIALIZATIONS.get(), 0);
    assert_eq!(calls.load(std::sync::atomic::Ordering::Relaxed), 0);
    assert_eq!(left.codomain(), input.codomain());
    assert_eq!(right.domain(), input.domain());
    assert_eq!(left.domain()[0].sectors().unwrap().len(), 2);
    assert_eq!(right.codomain()[0].sectors().unwrap().len(), 2);
    assert_eq!(left.domain()[0].degeneracy(&U1Irrep::new(0)).unwrap(), 1);
    assert_eq!(left.domain()[0].degeneracy(&U1Irrep::new(1)).unwrap(), 2);
    assert_eq!(right.codomain()[0].degeneracy(&U1Irrep::new(0)).unwrap(), 1);
    assert_eq!(right.codomain()[0].degeneracy(&U1Irrep::new(1)).unwrap(), 2);
    let coordinates = [0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0];
    assert_eq!(left.dense_data().unwrap(), coordinates);
    assert_eq!(right.dense_data().unwrap(), coordinates);
    assert!(
        left.adjoint()
            .unwrap()
            .compose(&input)
            .unwrap()
            .norm(2.0)
            .unwrap()
            < 1e-12
    );
    assert!(
        input
            .compose(&right.adjoint().unwrap())
            .unwrap()
            .norm(2.0)
            .unwrap()
            < 1e-12
    );
}

#[test]
fn compact_diagonal_null_su2_sectors_use_reduced_coordinate_basis() {
    let calls = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let runtime = Runtime::builder()
        .with_dense_executor(Box::new(CountPolarKernels {
            svd_calls: Arc::clone(&calls),
            ..Default::default()
        }))
        .build()
        .unwrap();
    let spin0 = SU2Irrep::from_twice_spin(0);
    let half = SU2Irrep::from_twice_spin(1);
    let leg = GradedSpace::try_new(Arc::new(SU2FusionRule), [(spin0, 3), (half, 2)]).unwrap();
    let input: TensorMap<_, f64> = TensorMap::diagonal(
        &runtime,
        &leg,
        [
            SectorSpectrum {
                sector: spin0,
                values: vec![3.0, 0.0, -2.0],
            },
            SectorSpectrum {
                sector: half,
                values: vec![0.0, 4.0],
            },
        ],
    )
    .unwrap();
    DIAGONAL_MATERIALIZATIONS.set(0);
    let left = input.left_null(&[0], &[1]).unwrap();
    let right = input.right_null(&[0], &[1]).unwrap();
    assert_eq!(DIAGONAL_MATERIALIZATIONS.get(), 0);
    assert_eq!(calls.load(std::sync::atomic::Ordering::Relaxed), 0);
    assert_eq!(left.codomain(), input.codomain());
    assert_eq!(right.domain(), input.domain());
    for bond in [&left.domain()[0], &right.codomain()[0]] {
        assert!(!bond.is_dual());
        assert_eq!(bond.sectors().unwrap().len(), 2);
        assert_eq!(bond.degeneracy(&spin0).unwrap(), 1);
        assert_eq!(bond.degeneracy(&half).unwrap(), 1);
    }
    let coordinates = [0.0, 1.0, 0.0, 1.0, 0.0];
    assert_eq!(left.dense_data().unwrap(), coordinates);
    assert_eq!(right.dense_data().unwrap(), coordinates);
    assert!(
        left.adjoint()
            .unwrap()
            .compose(&input)
            .unwrap()
            .norm(2.0)
            .unwrap()
            < 1e-12
    );
    assert!(
        input
            .compose(&right.adjoint().unwrap())
            .unwrap()
            .norm(2.0)
            .unwrap()
            < 1e-12
    );
    let left_gram = left.adjoint().unwrap().compose(&left).unwrap();
    let right_gram = right.compose(&right.adjoint().unwrap()).unwrap();
    let left_eye = TensorMap::isomorphism(&runtime, &left.domain(), &left.domain()).unwrap();
    let right_eye = TensorMap::isomorphism(&runtime, &right.codomain(), &right.codomain()).unwrap();
    assert!(
        left_gram
            .axpby(1.0, &left_eye, -1.0)
            .unwrap()
            .norm(2.0)
            .unwrap()
            < 1e-12
    );
    assert!(
        right_gram
            .axpby(1.0, &right_eye, -1.0)
            .unwrap()
            .norm(2.0)
            .unwrap()
            < 1e-12
    );
}

#[test]
fn compact_diagonal_null_and_cutoff_fallback_cover_all_scalars() {
    let calls = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let runtime = Runtime::builder()
        .with_dense_executor(Box::new(CountPolarKernels {
            svd_calls: Arc::clone(&calls),
            ..Default::default()
        }))
        .build()
        .unwrap();
    let near_runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let sector = U1Irrep::new(0);
    let leg = GradedSpace::try_new(Arc::new(U1FusionRule), [(sector, 3)]).unwrap();
    macro_rules! check_scalar {
        ($scalar:ty, $convert:expr, $eps:expr) => {{
            let input: TensorMap<_, $scalar> = TensorMap::diagonal(
                &runtime,
                &leg,
                [SectorSpectrum {
                    sector,
                    values: vec![($convert)(0.0), ($convert)(3.0), ($convert)(0.0)],
                }],
            )
            .unwrap();
            calls.store(0, std::sync::atomic::Ordering::Relaxed);
            DIAGONAL_MATERIALIZATIONS.set(0);
            let left = input.left_null(&[0], &[1]).unwrap();
            let right = input.right_null(&[0], &[1]).unwrap();
            assert_eq!(calls.load(std::sync::atomic::Ordering::Relaxed), 0);
            assert_eq!(DIAGONAL_MATERIALIZATIONS.get(), 0);
            let coordinates = [
                ($convert)(1.0),
                ($convert)(0.0),
                ($convert)(0.0),
                ($convert)(0.0),
                ($convert)(0.0),
                ($convert)(1.0),
            ];
            assert_eq!(left.dense_data().unwrap(), coordinates);
            assert_eq!(right.dense_data().unwrap(), coordinates);
            assert_eq!(left.domain()[0].degeneracy(&sector).unwrap(), 2);
            assert_eq!(right.codomain()[0].degeneracy(&sector).unwrap(), 2);

            let full: TensorMap<_, $scalar> = TensorMap::diagonal(
                &runtime,
                &leg,
                [SectorSpectrum {
                    sector,
                    values: vec![($convert)(1.0), ($convert)(2.0), ($convert)(3.0)],
                }],
            )
            .unwrap();
            calls.store(0, std::sync::atomic::Ordering::Relaxed);
            DIAGONAL_MATERIALIZATIONS.set(0);
            let full_left = full.left_null(&[0], &[1]).unwrap();
            let full_right = full.right_null(&[0], &[1]).unwrap();
            assert_eq!(calls.load(std::sync::atomic::Ordering::Relaxed), 0);
            assert_eq!(DIAGONAL_MATERIALIZATIONS.get(), 0);
            assert!(full_left.dense_data().unwrap().is_empty());
            assert!(full_right.dense_data().unwrap().is_empty());
            assert!(full_left.domain()[0].sectors().unwrap().is_empty());
            assert!(full_right.codomain()[0].sectors().unwrap().is_empty());

            let cutoff = ($eps as f64) * 2.0;
            let small_leg = GradedSpace::try_new(Arc::new(U1FusionRule), [(sector, 2)]).unwrap();
            for (small, expected_len) in [
                (0.5 * cutoff, Some(2)),
                (cutoff, None),
                (2.0 * cutoff, Some(0)),
            ] {
                let near: TensorMap<_, $scalar> = TensorMap::diagonal(
                    &near_runtime,
                    &small_leg,
                    [SectorSpectrum {
                        sector,
                        values: vec![($convert)(1.0), ($convert)(small)],
                    }],
                )
                .unwrap();
                DIAGONAL_MATERIALIZATIONS.set(0);
                let left = near.left_null(&[0], &[1]).unwrap();
                let right = near.right_null(&[0], &[1]).unwrap();
                assert_eq!(DIAGONAL_MATERIALIZATIONS.get(), 2);
                if let Some(expected_len) = expected_len {
                    assert_eq!(left.dense_data().unwrap().len(), expected_len);
                    assert_eq!(right.dense_data().unwrap().len(), expected_len);
                } else {
                    assert_eq!(left.codomain(), near.codomain());
                    assert_eq!(right.domain(), near.domain());
                    assert!(left.dense_data().unwrap().len() <= 2);
                    assert!(right.dense_data().unwrap().len() <= 2);
                }
            }
        }};
    }
    check_scalar!(f32, |x: f64| x as f32, f32::EPSILON);
    check_scalar!(f64, |x: f64| x, f64::EPSILON);
    check_scalar!(
        num_complex::Complex32,
        |x: f64| num_complex::Complex32::new(x as f32, 0.0),
        f32::EPSILON
    );
    check_scalar!(
        num_complex::Complex64,
        |x: f64| num_complex::Complex64::new(x, 0.0),
        f64::EPSILON
    );

    let tiny: TensorMap<_, f32> = TensorMap::diagonal(
        &near_runtime,
        &GradedSpace::try_new(Arc::new(U1FusionRule), [(sector, 1)]).unwrap(),
        [SectorSpectrum {
            sector,
            values: vec![f32::from_bits(1)],
        }],
    )
    .unwrap();
    DIAGONAL_MATERIALIZATIONS.set(0);
    let _ = tiny.left_null(&[0], &[1]);
    assert_eq!(DIAGONAL_MATERIALIZATIONS.get(), 1);

    let unrepresentable: TensorMap<_, num_complex::Complex32> = TensorMap::diagonal(
        &near_runtime,
        &GradedSpace::try_new(Arc::new(U1FusionRule), [(sector, 1)]).unwrap(),
        [SectorSpectrum {
            sector,
            values: vec![num_complex::Complex32::new(f32::MAX, f32::MAX)],
        }],
    )
    .unwrap();
    DIAGONAL_MATERIALIZATIONS.set(0);
    let _ = unrepresentable.left_null(&[0], &[1]);
    assert_eq!(DIAGONAL_MATERIALIZATIONS.get(), 1);
}

#[test]
fn compact_diagonal_null_preserves_dual_space_and_lazy_adjoint_semantics() {
    use num_complex::Complex64;

    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let leg = GradedSpace::try_new(Arc::new(U1FusionRule), [(U1Irrep::new(1), 2)])
        .unwrap()
        .try_dual()
        .unwrap();
    let input: TensorMap<_, Complex64> = TensorMap::diagonal(
        &runtime,
        &leg,
        [SectorSpectrum {
            sector: U1Irrep::new(-1),
            values: vec![Complex64::new(0.0, 0.0), Complex64::new(2.0, 3.0)],
        }],
    )
    .unwrap();
    let dense = input.materialize().unwrap();
    let lazy = dense.adjoint().unwrap();
    let left = input.left_null(&[0], &[1]).unwrap();
    let right = input.right_null(&[0], &[1]).unwrap();
    assert_eq!(left.codomain(), input.codomain());
    assert_eq!(right.domain(), input.domain());
    assert_eq!(left.domain()[0].degeneracy(&U1Irrep::new(-1)).unwrap(), 1);
    assert!(!left.domain()[0].is_dual());
    assert!(!right.codomain()[0].is_dual());
    assert_eq!(
        left.dense_data().unwrap(),
        [Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)]
    );
    assert_eq!(
        right.dense_data().unwrap(),
        [Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)]
    );
    let lazy_left = lazy.left_null(&[0], &[1]).unwrap();
    let lazy_right = lazy.right_null(&[0], &[1]).unwrap();
    assert_eq!(lazy_left.codomain(), lazy.codomain());
    assert_eq!(lazy_right.domain(), lazy.domain());
    assert_eq!(
        lazy_left.domain()[0].degeneracy(&U1Irrep::new(-1)).unwrap(),
        1
    );
    assert_eq!(
        lazy_right.codomain()[0]
            .degeneracy(&U1Irrep::new(-1))
            .unwrap(),
        1
    );
    assert!(
        lazy_left
            .adjoint()
            .unwrap()
            .compose(&lazy)
            .unwrap()
            .norm(2.0)
            .unwrap()
            < 1e-12
    );
    assert!(
        lazy.compose(&lazy_right.adjoint().unwrap())
            .unwrap()
            .norm(2.0)
            .unwrap()
            < 1e-12
    );
}

#[test]
fn compact_diagonal_null_product_sectors_roles_and_nonfinite_fallback() {
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let provider = Arc::new(U1FusionRule.product(FermionParityFusionRule));
    let even = product_sector(U1Irrep::new(0), Z2Irrep::EVEN);
    let odd = product_sector(U1Irrep::new(1), Z2Irrep::ODD);
    let leg = GradedSpace::try_new(Arc::clone(&provider), [(even, 2), (odd, 2)]).unwrap();
    let input: TensorMap<_, f64> = TensorMap::diagonal(
        &runtime,
        &leg,
        [
            SectorSpectrum {
                sector: even,
                values: vec![0.0, -2.0],
            },
            SectorSpectrum {
                sector: odd,
                values: vec![3.0, 0.0],
            },
        ],
    )
    .unwrap();
    DIAGONAL_MATERIALIZATIONS.set(0);
    let left = input.left_null(&[0], &[1]).unwrap();
    let right = input.right_null(&[0], &[1]).unwrap();
    assert_eq!(DIAGONAL_MATERIALIZATIONS.get(), 0);
    assert_eq!(left.domain()[0].degeneracy(&even).unwrap(), 1);
    assert_eq!(left.domain()[0].degeneracy(&odd).unwrap(), 1);
    assert_eq!(right.codomain()[0].degeneracy(&even).unwrap(), 1);
    assert_eq!(right.codomain()[0].degeneracy(&odd).unwrap(), 1);
    assert_eq!(left.dense_data().unwrap(), [1.0, 0.0, 0.0, 1.0]);
    assert_eq!(right.dense_data().unwrap(), [1.0, 0.0, 0.0, 1.0]);
    let permuted = input.permute(&[1], &[0]).unwrap();
    let changed_left = input.left_null(&[1], &[0]).unwrap();
    let changed_right = input.right_null(&[1], &[0]).unwrap();
    assert_eq!(changed_left.codomain(), permuted.codomain());
    assert_eq!(changed_right.domain(), permuted.domain());
    assert!(
        changed_left
            .adjoint()
            .unwrap()
            .compose(&permuted)
            .unwrap()
            .norm(2.0)
            .unwrap()
            < 1e-12
    );
    assert!(
        permuted
            .compose(&changed_right.adjoint().unwrap())
            .unwrap()
            .norm(2.0)
            .unwrap()
            < 1e-12
    );

    let nonfinite: TensorMap<_, f64> = TensorMap::diagonal(
        &runtime,
        &leg,
        [
            SectorSpectrum {
                sector: even,
                values: vec![f64::NAN, 0.0],
            },
            SectorSpectrum {
                sector: odd,
                values: vec![3.0, 0.0],
            },
        ],
    )
    .unwrap();
    DIAGONAL_MATERIALIZATIONS.set(0);
    assert!(nonfinite.left_null(&[0], &[1]).is_err());
    assert_eq!(DIAGONAL_MATERIALIZATIONS.get(), 1);
}

#[test]
fn compact_diagonal_svd_full_sorted_hand_oracle_all_scalars() {
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let leg = GradedSpace::try_new(Arc::new(U1FusionRule), [(U1Irrep::new(0), 4)]).unwrap();
    macro_rules! check {
        ($ty:ty, $second:expr, $phase:expr, $tol:expr) => {{
            let input: TensorMap<_, $ty> = TensorMap::diagonal(
                &runtime,
                &leg,
                [SectorSpectrum {
                    sector: U1Irrep::new(0),
                    values: vec![
                        <$ty as FactorScalar>::from_real(0.0),
                        $second,
                        <$ty as FactorScalar>::from_real(2.0),
                        <$ty as FactorScalar>::from_real(-4.0),
                    ],
                }],
            )
            .unwrap();
            let out = input.svd_full(&[0], &[1]).unwrap();
            assert_eq!(
                out.s.diagview().unwrap()[0].values,
                [4.0, 2.0, 2.0, 0.0].map(<$ty as FactorScalar>::from_real)
            );
            // Hand order: magnitude 4, the two 2s in source order, then zero.
            let permutation = [3, 1, 2, 0];
            let phases = [
                Complex64::new(-1.0, 0.0),
                $phase,
                Complex64::new(1.0, 0.0),
                Complex64::new(1.0, 0.0),
            ];
            for col in 0..4 {
                for row in 0..4 {
                    let u = Complex64::new(if row == permutation[col] { 1.0 } else { 0.0 }, 0.0);
                    let vh = if col == permutation[row] {
                        phases[row]
                    } else {
                        Complex64::new(0.0, 0.0)
                    };
                    assert!(
                        (out.u.dense_data().unwrap()[row + 4 * col].widen_complex() - u).norm()
                            < $tol
                    );
                    assert!(
                        (out.vh.dense_data().unwrap()[row + 4 * col].widen_complex() - vh).norm()
                            < $tol
                    );
                }
            }
        }};
    }
    check!(f32, -2.0, Complex64::new(-1.0, 0.0), 1e-6);
    check!(f64, -2.0, Complex64::new(-1.0, 0.0), 1e-12);
    check!(
        num_complex::Complex32,
        num_complex::Complex32::new(0.0, -2.0),
        Complex64::new(0.0, -1.0),
        1e-6
    );
    check!(
        Complex64,
        Complex64::new(0.0, -2.0),
        Complex64::new(0.0, -1.0),
        1e-12
    );
}

#[test]
fn compact_diagonal_svd_full_subnormal_phase_and_nonfinite_fallback() {
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let leg = GradedSpace::try_new(Arc::new(U1FusionRule), [(U1Irrep::new(0), 1)]).unwrap();
    macro_rules! check {
        ($ty:ty, $value:expr, $phase:expr, $tol:expr, $bad:expr) => {{
            let make = |value| {
                TensorMap::<_, $ty>::diagonal(
                    &runtime,
                    &leg,
                    [SectorSpectrum {
                        sector: U1Irrep::new(0),
                        values: vec![value],
                    }],
                )
                .unwrap()
            };
            let input = make($value);
            DIAGONAL_MATERIALIZATIONS.set(0);
            let out = input.svd_full(&[0], &[1]).unwrap();
            assert_eq!(DIAGONAL_MATERIALIZATIONS.get(), 0);
            assert!(out.s.diagview().unwrap()[0].values[0].widen_complex().re > 0.0);
            assert!(
                (out.u.dense_data().unwrap()[0].widen_complex() - Complex64::new(1.0, 0.0)).norm()
                    < $tol
            );
            assert!((out.vh.dense_data().unwrap()[0].widen_complex() - $phase).norm() < $tol);
            assert_eq!(
                out.u
                    .compose(&out.s)
                    .unwrap()
                    .compose(&out.vh)
                    .unwrap()
                    .dense_data()
                    .unwrap(),
                &[$value]
            );
            for value in $bad {
                let input = make(value);
                let dense_result = input.materialize().unwrap().svd_full(&[0], &[1]);
                DIAGONAL_MATERIALIZATIONS.set(0);
                let result = input.svd_full(&[0], &[1]);
                assert_eq!(DIAGONAL_MATERIALIZATIONS.get(), 1);
                match (result, dense_result) {
                    (Err(a), Err(b)) => assert_eq!(a.to_string(), b.to_string()),
                    (Ok(a), Ok(_)) => assert!(a.s.dense_data().is_ok()),
                    _ => panic!("diagonal fallback changed dense acceptance"),
                }
            }
        }};
    }
    check!(
        f32,
        -f32::from_bits(1),
        Complex64::new(-1.0, 0.0),
        1e-6,
        [f32::NAN, f32::INFINITY]
    );
    check!(
        f64,
        -f64::from_bits(1),
        Complex64::new(-1.0, 0.0),
        1e-12,
        [f64::NAN, f64::INFINITY]
    );
    let phase = Complex64::new(0.5_f64.sqrt(), 0.5_f64.sqrt());
    check!(
        num_complex::Complex32,
        num_complex::Complex32::new(f32::from_bits(1), f32::from_bits(1)),
        phase,
        1e-6,
        [
            num_complex::Complex32::new(f32::NAN, 0.0),
            num_complex::Complex32::new(f32::INFINITY, 0.0),
            num_complex::Complex32::new(f32::MAX, f32::MAX)
        ]
    );
    check!(
        Complex64,
        Complex64::new(f64::from_bits(1), f64::from_bits(1)),
        phase,
        1e-12,
        [
            Complex64::new(f64::NAN, 0.0),
            Complex64::new(f64::INFINITY, 0.0),
            Complex64::new(f64::MAX, f64::MAX)
        ]
    );
}

#[test]
fn compact_diagonal_svd_uses_spectrum_without_dense_input() {
    for full in [false, true] {
        let runtime = Runtime::builder().dense_threads(1).build().unwrap();
        let leg = GradedSpace::try_new(
            Arc::new(U1FusionRule),
            [(U1Irrep::new(0), 3), (U1Irrep::new(1), 2)],
        )
        .unwrap();
        let source: TensorMap<_, f64> = TensorMap::diagonal(
            &runtime,
            &leg,
            [
                SectorSpectrum {
                    sector: U1Irrep::new(1),
                    values: vec![3.0, -2.0],
                },
                SectorSpectrum {
                    sector: U1Irrep::new(0),
                    values: vec![1.0, -4.0, 0.0],
                },
            ],
        )
        .unwrap();
        let input = source.materialize().unwrap().dense_data().unwrap().to_vec();
        DIAGONAL_MATERIALIZATIONS.set(0);
        let Svd { u, s, vh } = (if full {
            source.svd_full(&[0], &[1])
        } else {
            source.svd_compact(&[0], &[1])
        })
        .unwrap();
        assert_eq!(DIAGONAL_MATERIALIZATIONS.get(), 0);
        assert_eq!(
            s.diagview()
                .unwrap()
                .iter()
                .map(|e| e.values.clone())
                .collect::<Vec<_>>(),
            vec![vec![4.0, 1.0, 0.0], vec![3.0, 2.0]]
        );
        let rebuilt = u.compose(&s).unwrap().compose(&vh).unwrap();
        assert!(rebuilt
            .dense_data()
            .unwrap()
            .iter()
            .zip(input)
            .all(|(a, b)| (a - b).abs() < 1e-12));
        assert_eq!(source.diagview().unwrap()[0].values, vec![1.0, -4.0, 0.0]);
    }
}

#[test]
fn compact_diagonal_svd_vals_uses_only_the_stored_spectrum() {
    let solver_calls = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let runtime = Runtime::builder()
        .with_dense_executor(Box::new(FailSecondSvd {
            record: Some(Arc::clone(&solver_calls)),
            ..Default::default()
        }))
        .build()
        .unwrap();
    let leg = GradedSpace::try_new(
        Arc::new(U1FusionRule),
        [(U1Irrep::new(0), 3), (U1Irrep::new(1), 2)],
    )
    .unwrap();
    let input: TensorMap<_, f64> = TensorMap::diagonal(
        &runtime,
        &leg,
        [
            SectorSpectrum {
                sector: U1Irrep::new(1),
                values: vec![-2.0, 3.0],
            },
            SectorSpectrum {
                sector: U1Irrep::new(0),
                values: vec![-4.0, 0.0, 1.0],
            },
        ],
    )
    .unwrap();
    let before = input.diagview().unwrap();
    DIAGONAL_MATERIALIZATIONS.set(0);
    assert_eq!(
        input.svd_vals(&[0], &[1]).unwrap(),
        vec![
            SectorSpectrum {
                sector: U1Irrep::new(0),
                values: vec![4.0, 1.0, 0.0]
            },
            SectorSpectrum {
                sector: U1Irrep::new(1),
                values: vec![3.0, 2.0]
            },
        ]
    );
    assert_eq!(DIAGONAL_MATERIALIZATIONS.get(), 0);
    assert_eq!(input.diagview().unwrap(), before);

    let dual_bond = GradedSpace::try_new(Arc::new(U1FusionRule), [(U1Irrep::new(1), 2)])
        .unwrap()
        .try_dual()
        .unwrap();
    let dual: TensorMap<_, f64> = TensorMap::diagonal(
        &runtime,
        &dual_bond,
        [SectorSpectrum {
            sector: U1Irrep::new(-1),
            values: vec![-2.0, 1.0],
        }],
    )
    .unwrap();
    DIAGONAL_MATERIALIZATIONS.set(0);
    assert_eq!(dual.svd_vals(&[0], &[1]).unwrap()[0].values, [2.0, 1.0]);
    assert_eq!(DIAGONAL_MATERIALIZATIONS.get(), 0);
    assert_eq!(solver_calls.load(std::sync::atomic::Ordering::Relaxed), 0);
}

#[cfg(feature = "racah-generated")]
#[test]
fn checked_compact_diagonal_svd_vals_skips_materialization_and_solver() {
    use tenet_core::SUNFusionRule;

    let solver_calls = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let runtime = Runtime::builder()
        .with_dense_executor(Box::new(FailSecondSvd {
            record: Some(Arc::clone(&solver_calls)),
            ..Default::default()
        }))
        .build()
        .unwrap();
    let provider = Arc::new(SUNFusionRule::new(3).unwrap());
    let leg =
        GradedSpace::try_new(Arc::clone(&provider), [(vec![0, 0], 2), (vec![1, 0], 3)]).unwrap();
    let real: TensorMap<_, f64> = TensorMap::diagonal(
        &runtime,
        &leg,
        [
            SectorSpectrum {
                sector: vec![1, 0],
                values: vec![3.0, -2.0, 1.0],
            },
            SectorSpectrum {
                sector: vec![0, 0],
                values: vec![-4.0, 0.0],
            },
        ],
    )
    .unwrap();
    DIAGONAL_MATERIALIZATIONS.set(0);
    assert_eq!(
        real.svd_vals(&[0], &[1]).unwrap(),
        vec![
            SectorSpectrum {
                sector: vec![0, 0],
                values: vec![4.0, 0.0],
            },
            SectorSpectrum {
                sector: vec![1, 0],
                values: vec![3.0, 2.0, 1.0],
            },
        ]
    );
    assert_eq!(DIAGONAL_MATERIALIZATIONS.get(), 0);
    assert_eq!(solver_calls.load(std::sync::atomic::Ordering::Relaxed), 0);

    let stored = real.spectrum().unwrap();
    let mut missing = stored.to_vec();
    missing.pop();
    assert!(
        tenet_matrixalgebra::svd_vals_compact_diagonal_dyn(&owned(&real).space, &missing)
            .unwrap()
            .is_none()
    );
    let mut duplicate = stored.to_vec();
    duplicate[1].sector = duplicate[0].sector;
    assert!(
        tenet_matrixalgebra::svd_vals_compact_diagonal_dyn(&owned(&real).space, &duplicate)
            .unwrap()
            .is_none()
    );
    let mut short = stored.to_vec();
    short[0].values.pop();
    assert!(
        tenet_matrixalgebra::svd_vals_compact_diagonal_dyn(&owned(&real).space, &short)
            .unwrap()
            .is_none()
    );

    let complex: TensorMap<_, Complex64> = TensorMap::diagonal(
        &runtime,
        &leg,
        [
            SectorSpectrum {
                sector: vec![1, 0],
                values: vec![
                    Complex64::new(-3.0, 4.0),
                    Complex64::new(0.0, 0.0),
                    Complex64::new(1.0, 1.0),
                ],
            },
            SectorSpectrum {
                sector: vec![0, 0],
                values: vec![Complex64::new(-4.0, 0.0), Complex64::new(0.0, 0.0)],
            },
        ],
    )
    .unwrap();
    DIAGONAL_MATERIALIZATIONS.set(0);
    let got = complex.svd_vals(&[0], &[1]).unwrap();
    assert_eq!(got[0].sector, vec![0, 0]);
    assert_eq!(got[0].values, [4.0, 0.0]);
    assert_eq!(got[1].sector, vec![1, 0]);
    assert_eq!(got[1].values[0], 5.0);
    assert!((got[1].values[1] - 2.0_f64.sqrt()).abs() < 1e-12);
    assert_eq!(got[1].values[2], 0.0);
    assert_eq!(DIAGONAL_MATERIALIZATIONS.get(), 0);
    assert_eq!(solver_calls.load(std::sync::atomic::Ordering::Relaxed), 0);

    let dual_leg = GradedSpace::try_new(Arc::clone(&provider), [(vec![1, 0], 2)])
        .unwrap()
        .try_dual()
        .unwrap();
    let dual_sector = dual_leg.sectors().unwrap().remove(0);
    let dual: TensorMap<_, f64> = TensorMap::diagonal(
        &runtime,
        &dual_leg,
        [SectorSpectrum {
            sector: dual_sector.clone(),
            values: vec![-2.0, 0.0],
        }],
    )
    .unwrap();
    DIAGONAL_MATERIALIZATIONS.set(0);
    assert_eq!(
        dual.svd_vals(&[0], &[1]).unwrap(),
        vec![SectorSpectrum {
            sector: dual_sector,
            values: vec![2.0, 0.0]
        }]
    );
    assert_eq!(DIAGONAL_MATERIALIZATIONS.get(), 0);
    assert_eq!(solver_calls.load(std::sync::atomic::Ordering::Relaxed), 0);
}

#[cfg(feature = "racah-generated")]
#[test]
fn checked_compact_diagonal_eigh_vals_reads_stored_real_spectrum() {
    use tenet_core::SUNFusionRule;

    let calls = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let runtime = Runtime::builder()
        .with_dense_executor(Box::new(CountEighVals {
            calls: Arc::clone(&calls),
            ..Default::default()
        }))
        .build()
        .unwrap();
    let provider = Arc::new(SUNFusionRule::new(3).unwrap());
    let leg =
        GradedSpace::try_new(Arc::clone(&provider), [(vec![0, 0], 2), (vec![1, 0], 3)]).unwrap();
    let input: TensorMap<_, Complex64> = TensorMap::diagonal(
        &runtime,
        &leg,
        [
            SectorSpectrum {
                sector: vec![1, 0],
                values: vec![
                    Complex64::new(2.0, 0.0),
                    Complex64::new(-2.0, 0.0),
                    Complex64::new(1.0, 0.0),
                ],
            },
            SectorSpectrum {
                sector: vec![0, 0],
                values: vec![Complex64::new(-4.0, 0.0), Complex64::new(0.0, 0.0)],
            },
        ],
    )
    .unwrap();
    DIAGONAL_MATERIALIZATIONS.set(0);
    let got = input.eigh_vals(&[0], &[1]).unwrap();
    assert_eq!(
        got,
        vec![
            SectorSpectrum {
                sector: vec![0, 0],
                values: vec![-4.0, 0.0],
            },
            SectorSpectrum {
                sector: vec![1, 0],
                values: vec![-2.0, 2.0, 1.0],
            },
        ]
    );
    assert_eq!(DIAGONAL_MATERIALIZATIONS.get(), 0);
    assert_eq!(calls.load(std::sync::atomic::Ordering::Relaxed), 0);

    let swapped = input.eigh_vals(&[1], &[0]).unwrap();
    assert_eq!(calls.load(std::sync::atomic::Ordering::Relaxed), 2);
    let swapped_dense = input.permute(&[1], &[0]).unwrap().materialize().unwrap();
    assert_eq!(swapped, swapped_dense.eigh_vals(&[0], &[1]).unwrap());
    let dense = input.materialize().unwrap();
    assert_eq!(got, dense.eigh_vals(&[0], &[1]).unwrap());

    let real: TensorMap<_, f64> = TensorMap::diagonal(
        &runtime,
        &leg,
        [
            SectorSpectrum {
                sector: vec![1, 0],
                values: vec![2.0, -2.0, 1.0],
            },
            SectorSpectrum {
                sector: vec![0, 0],
                values: vec![-4.0, 0.0],
            },
        ],
    )
    .unwrap();
    let dense_real = real.materialize().unwrap();
    DIAGONAL_MATERIALIZATIONS.set(0);
    assert_eq!(
        real.eigh_vals(&[0], &[1]).unwrap(),
        dense_real.eigh_vals(&[0], &[1]).unwrap()
    );
    assert_eq!(DIAGONAL_MATERIALIZATIONS.get(), 0);

    let narrow: TensorMap<_, num_complex::Complex32> = TensorMap::diagonal(
        &runtime,
        &leg,
        [
            SectorSpectrum {
                sector: vec![1, 0],
                values: vec![
                    num_complex::Complex32::new(2.0, 0.0),
                    num_complex::Complex32::new(-2.0, 0.0),
                    num_complex::Complex32::new(1.0, 0.0),
                ],
            },
            SectorSpectrum {
                sector: vec![0, 0],
                values: vec![
                    num_complex::Complex32::new(-4.0, 0.0),
                    num_complex::Complex32::new(0.0, 0.0),
                ],
            },
        ],
    )
    .unwrap();
    let dense_narrow = narrow.materialize().unwrap();
    DIAGONAL_MATERIALIZATIONS.set(0);
    assert_eq!(
        narrow.eigh_vals(&[0], &[1]).unwrap(),
        dense_narrow.eigh_vals(&[0], &[1]).unwrap()
    );
    assert_eq!(DIAGONAL_MATERIALIZATIONS.get(), 0);
    calls.store(0, std::sync::atomic::Ordering::Relaxed);

    let dual_leg = GradedSpace::try_new(Arc::clone(&provider), [(vec![1, 0], 2)])
        .unwrap()
        .try_dual()
        .unwrap();
    let dual_sector = dual_leg.sectors().unwrap().remove(0);
    let dual: TensorMap<_, f32> = TensorMap::diagonal(
        &runtime,
        &dual_leg,
        [SectorSpectrum {
            sector: dual_sector.clone(),
            values: vec![-2.0, 2.0],
        }],
    )
    .unwrap();
    DIAGONAL_MATERIALIZATIONS.set(0);
    assert_eq!(
        dual.eigh_vals(&[0], &[1]).unwrap(),
        vec![SectorSpectrum {
            sector: dual_sector,
            values: vec![-2.0, 2.0],
        }]
    );
    assert_eq!(DIAGONAL_MATERIALIZATIONS.get(), 0);
    assert_eq!(calls.load(std::sync::atomic::Ordering::Relaxed), 0);
}

#[cfg(feature = "racah-generated")]
#[test]
fn checked_compact_diagonal_eigh_vals_rejects_inconsistent_spectrum_admission() {
    use tenet_core::SUNFusionRule;

    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let provider = Arc::new(SUNFusionRule::new(3).unwrap());
    let leg = GradedSpace::try_new(provider, [(vec![0, 0], 2), (vec![1, 0], 1)]).unwrap();
    let input: TensorMap<_, f64> = TensorMap::diagonal(
        &runtime,
        &leg,
        [
            SectorSpectrum {
                sector: vec![0, 0],
                values: vec![1.0, -1.0],
            },
            SectorSpectrum {
                sector: vec![1, 0],
                values: vec![2.0],
            },
        ],
    )
    .unwrap();
    let spectrum = input.spectrum().unwrap();
    let admit = |entries: &[tenet_matrixalgebra::SectorSpectrum<f64>]| {
        tenet_matrixalgebra::eigh_vals_diagonal_dyn(input.logical_space(), entries).unwrap()
    };
    assert!(admit(spectrum).is_some());
    let mut reversed = spectrum.to_vec();
    reversed.reverse();
    assert!(admit(&reversed).is_some());
    assert!(admit(&spectrum[..1]).is_none());
    let mut duplicate = spectrum.to_vec();
    duplicate[1].sector = duplicate[0].sector;
    assert!(admit(&duplicate).is_none());
    let mut wrong_size = spectrum.to_vec();
    wrong_size[0].values.push(0.0);
    assert!(admit(&wrong_size).is_none());
}

#[cfg(feature = "racah-generated")]
#[test]
fn checked_compact_diagonal_eigh_vals_keeps_dense_hermiticity_boundary() {
    use tenet_core::SUNFusionRule;

    let calls = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let runtime = Runtime::builder()
        .with_dense_executor(Box::new(CountEighVals {
            calls: Arc::clone(&calls),
            ..Default::default()
        }))
        .build()
        .unwrap();
    let provider = Arc::new(SUNFusionRule::new(3).unwrap());
    let leg = GradedSpace::try_new(provider, [(vec![0, 0], 1)]).unwrap();
    let near: TensorMap<_, Complex64> = TensorMap::diagonal(
        &runtime,
        &leg,
        [SectorSpectrum {
            sector: vec![0, 0],
            values: vec![Complex64::new(1.0, 1e-15)],
        }],
    )
    .unwrap();
    DIAGONAL_MATERIALIZATIONS.set(0);
    assert_eq!(near.eigh_vals(&[0], &[1]).unwrap()[0].values, [1.0]);
    assert_eq!(DIAGONAL_MATERIALIZATIONS.get(), 1);
    assert_eq!(calls.load(std::sync::atomic::Ordering::Relaxed), 1);

    let nonfinite: TensorMap<_, f64> = TensorMap::diagonal(
        &runtime,
        &leg,
        [SectorSpectrum {
            sector: vec![0, 0],
            values: vec![f64::NAN],
        }],
    )
    .unwrap();
    DIAGONAL_MATERIALIZATIONS.set(0);
    let compact_error = nonfinite.eigh_vals(&[0], &[1]).unwrap_err();
    assert_eq!(DIAGONAL_MATERIALIZATIONS.get(), 1);
    let dense_error = nonfinite
        .materialize()
        .unwrap()
        .eigh_vals(&[0], &[1])
        .unwrap_err();
    assert_eq!(compact_error.to_string(), dense_error.to_string());
}

#[cfg(feature = "racah-generated")]
#[test]
fn checked_compact_diagonal_svd_vals_rounds_at_payload_precision() {
    use tenet_core::SUNFusionRule;

    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let provider = Arc::new(SUNFusionRule::new(3).unwrap());
    let leg = GradedSpace::try_new(provider, [(vec![1, 0], 2)]).unwrap();
    let real: TensorMap<_, f32> = TensorMap::diagonal(
        &runtime,
        &leg,
        [SectorSpectrum {
            sector: vec![1, 0],
            values: vec![-1.000_000_1, 0.0],
        }],
    )
    .unwrap();
    DIAGONAL_MATERIALIZATIONS.set(0);
    assert_eq!(
        real.svd_vals(&[0], &[1]).unwrap()[0].values,
        [1.000_000_1_f32 as f64, 0.0]
    );
    assert_eq!(DIAGONAL_MATERIALIZATIONS.get(), 0);

    let complex: TensorMap<_, num_complex::Complex32> = TensorMap::diagonal(
        &runtime,
        &leg,
        [SectorSpectrum {
            sector: vec![1, 0],
            values: vec![
                num_complex::Complex32::new(1.0, 1.0),
                num_complex::Complex32::new(-3.0, 4.0),
            ],
        }],
    )
    .unwrap();
    DIAGONAL_MATERIALIZATIONS.set(0);
    assert_eq!(
        complex.svd_vals(&[0], &[1]).unwrap()[0].values,
        [5.0, (2.0_f64.sqrt() as f32) as f64]
    );
    assert_eq!(DIAGONAL_MATERIALIZATIONS.get(), 0);

    for value in [
        num_complex::Complex32::new(f32::NAN, 0.0),
        num_complex::Complex32::new(f32::INFINITY, 0.0),
        num_complex::Complex32::new(f32::MAX * 0.75, f32::MAX * 0.75),
    ] {
        let input: TensorMap<_, num_complex::Complex32> = TensorMap::diagonal(
            &runtime,
            &leg,
            [SectorSpectrum {
                sector: vec![1, 0],
                values: vec![value, num_complex::Complex32::new(0.0, 0.0)],
            }],
        )
        .unwrap();
        DIAGONAL_MATERIALIZATIONS.set(0);
        let compact = input.svd_vals(&[0], &[1]);
        assert_eq!(DIAGONAL_MATERIALIZATIONS.get(), 1);
        let dense = input.materialize().unwrap().svd_vals(&[0], &[1]);
        assert_eq!(compact.is_err(), dense.is_err());
        if let (Err(compact_error), Err(dense_error)) = (compact, dense) {
            assert_eq!(compact_error.to_string(), dense_error.to_string());
        }
    }
}

#[cfg(feature = "racah-generated")]
#[test]
fn checked_compact_diagonal_pinv_does_not_materialize_the_input() {
    use tenet_core::SUNFusionRule;

    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let provider = Arc::new(SUNFusionRule::new(3).unwrap());
    let bond =
        GradedSpace::try_new(Arc::clone(&provider), [(vec![0, 0], 2), (vec![1, 0], 3)]).unwrap();
    let input: TensorMap<_, f64> = TensorMap::diagonal(
        &runtime,
        &bond,
        [
            SectorSpectrum {
                sector: vec![0, 0],
                values: vec![8.0, -4.0],
            },
            SectorSpectrum {
                sector: vec![1, 0],
                values: vec![1.0, 2.0, 0.0],
            },
        ],
    )
    .unwrap();
    DIAGONAL_MATERIALIZATIONS.set(0);
    let output = input.pinv(&[0], &[1], 0.25).unwrap();
    assert_eq!(DIAGONAL_MATERIALIZATIONS.get(), 0);
    assert!(output.dense_data().is_err());
    assert_eq!(output.diagview().unwrap()[0].values, [0.125, -0.25]);

    let TypedData::Diagonal(spectrum) = owned(&input).data.as_ref() else {
        panic!("diagonal constructor must keep compact storage");
    };
    let source = &owned(&input).space;
    let destination = &owned(&output).space;
    assert!(super::mode_dispatch::checked_compact_pinv_layout(
        source,
        destination,
        spectrum
    ));
    let mut missing = spectrum.clone();
    missing.pop();
    assert!(!super::mode_dispatch::checked_compact_pinv_layout(
        source,
        destination,
        &missing
    ));
    let mut duplicate = spectrum.clone();
    duplicate[1].sector = duplicate[0].sector;
    assert!(!super::mode_dispatch::checked_compact_pinv_layout(
        source,
        destination,
        &duplicate
    ));
    let mut short = spectrum.clone();
    short[0].values.pop();
    assert!(!super::mode_dispatch::checked_compact_pinv_layout(
        source,
        destination,
        &short
    ));
    let wrong_bond = GradedSpace::try_new(Arc::clone(&provider), [(vec![0, 0], 2)]).unwrap();
    let wrong: TensorMap<_, f64> = TensorMap::diagonal(
        &runtime,
        &wrong_bond,
        [SectorSpectrum {
            sector: vec![0, 0],
            values: vec![1.0, 1.0],
        }],
    )
    .unwrap();
    assert!(!super::mode_dispatch::checked_compact_pinv_layout(
        source,
        &owned(&wrong).space,
        spectrum
    ));
}

#[test]
fn compact_diagonal_svd_vals_rounds_at_payload_precision() {
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let leg = GradedSpace::try_new(Arc::new(U1FusionRule), [(U1Irrep::new(0), 2)]).unwrap();
    let real: TensorMap<_, f32> = TensorMap::diagonal(
        &runtime,
        &leg,
        [SectorSpectrum {
            sector: U1Irrep::new(0),
            values: vec![-1.000_000_1, 0.0],
        }],
    )
    .unwrap();
    assert_eq!(
        real.svd_vals(&[0], &[1]).unwrap()[0].values,
        [1.000_000_1_f32 as f64, 0.0]
    );

    let complex: TensorMap<_, num_complex::Complex32> = TensorMap::diagonal(
        &runtime,
        &leg,
        [SectorSpectrum {
            sector: U1Irrep::new(0),
            values: vec![
                num_complex::Complex32::new(1.0, 1.0),
                num_complex::Complex32::new(-3.0, 4.0),
            ],
        }],
    )
    .unwrap();
    assert_eq!(
        complex.svd_vals(&[0], &[1]).unwrap()[0].values,
        [5.0, (2.0_f64.sqrt() as f32) as f64]
    );
}

#[test]
fn compact_diagonal_svd_vals_retains_dense_rejection_edges() {
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let leg = GradedSpace::try_new(Arc::new(U1FusionRule), [(U1Irrep::new(0), 1)]).unwrap();
    for value in [
        num_complex::Complex32::new(f32::NAN, 0.0),
        num_complex::Complex32::new(f32::INFINITY, 0.0),
        num_complex::Complex32::new(f32::MAX * 0.75, f32::MAX * 0.75),
    ] {
        let input: TensorMap<_, num_complex::Complex32> = TensorMap::diagonal(
            &runtime,
            &leg,
            [SectorSpectrum {
                sector: U1Irrep::new(0),
                values: vec![value],
            }],
        )
        .unwrap();
        DIAGONAL_MATERIALIZATIONS.set(0);
        let compact = input.svd_vals(&[0], &[1]);
        assert_eq!(DIAGONAL_MATERIALIZATIONS.get(), 1);
        let dense = input.materialize().unwrap().svd_vals(&[0], &[1]);
        assert_eq!(compact.is_err(), dense.is_err());
    }
}

#[test]
fn compact_diagonal_svd_complex_permutation_and_phase() {
    for full in [false, true] {
        let runtime = Runtime::builder().dense_threads(1).build().unwrap();
        let leg = GradedSpace::try_new(
            Arc::new(U1FusionRule),
            [(U1Irrep::new(0), 3), (U1Irrep::new(1), 2)],
        )
        .unwrap();
        let z = Complex64::new;
        let input: TensorMap<_, Complex64> = TensorMap::diagonal(
            &runtime,
            &leg,
            [
                SectorSpectrum {
                    sector: U1Irrep::new(1),
                    values: vec![z(-2.0, 0.0), z(4.0, 0.0)],
                },
                SectorSpectrum {
                    sector: U1Irrep::new(0),
                    values: vec![z(0.0, 0.0), z(0.0, -3.0), z(1.0, 1.0)],
                },
            ],
        )
        .unwrap();
        DIAGONAL_MATERIALIZATIONS.set(0);
        let Svd { u, s, vh } = (if full {
            input.svd_full(&[0], &[1])
        } else {
            input.svd_compact(&[0], &[1])
        })
        .unwrap();
        assert_eq!(DIAGONAL_MATERIALIZATIONS.get(), 0);
        let root2 = 2.0_f64.sqrt();
        assert_eq!(u.dense_data().unwrap().len(), 13);
        assert_eq!(vh.dense_data().unwrap().len(), 13);
        for (actual, expected) in u.dense_data().unwrap().iter().zip([
            z(0.0, 0.0),
            z(1.0, 0.0),
            z(0.0, 0.0),
            z(0.0, 0.0),
            z(0.0, 0.0),
            z(1.0, 0.0),
            z(1.0, 0.0),
            z(0.0, 0.0),
            z(0.0, 0.0),
            z(0.0, 0.0),
            z(1.0, 0.0),
            z(1.0, 0.0),
            z(0.0, 0.0),
        ]) {
            assert!((*actual - expected).norm() < 1e-12);
        }
        for (actual, expected) in vh.dense_data().unwrap().iter().zip([
            z(0.0, 0.0),
            z(0.0, 0.0),
            z(1.0, 0.0),
            z(0.0, -1.0),
            z(0.0, 0.0),
            z(0.0, 0.0),
            z(0.0, 0.0),
            z(1.0 / root2, 1.0 / root2),
            z(0.0, 0.0),
            z(0.0, 0.0),
            z(-1.0, 0.0),
            z(1.0, 0.0),
            z(0.0, 0.0),
        ]) {
            assert!((*actual - expected).norm() < 1e-12);
        }
        assert_eq!(
            s.diagview()
                .unwrap()
                .iter()
                .map(|e| e.values.clone())
                .collect::<Vec<_>>(),
            vec![
                vec![z(3.0, 0.0), z(root2, 0.0), z(0.0, 0.0)],
                vec![z(4.0, 0.0), z(2.0, 0.0)]
            ]
        );
        let rebuilt = u.compose(&s).unwrap().compose(&vh).unwrap();
        assert!(rebuilt
            .dense_data()
            .unwrap()
            .iter()
            .zip(input.materialize().unwrap().dense_data().unwrap())
            .all(|(a, b)| (*a - *b).norm() < 1e-12));
    }
}

#[test]
fn compact_diagonal_svd_other_host_scalars_and_sectors() {
    for full in [false, true] {
        let runtime = Runtime::builder().dense_threads(1).build().unwrap();
        macro_rules! check {
            ($rule:expr, $legs:expr, $spectra:expr, $dtype:ty, $tol:expr) => {{
                let leg = GradedSpace::try_new(Arc::new($rule), $legs).unwrap();
                let input: TensorMap<_, $dtype> =
                    TensorMap::diagonal(&runtime, &leg, $spectra).unwrap();
                let before = input.materialize().unwrap().dense_data().unwrap().to_vec();
                let saved = input.diagview().unwrap();
                let dense_values = input.materialize().unwrap().svd_vals(&[0], &[1]).unwrap();
                DIAGONAL_MATERIALIZATIONS.set(0);
                let compact_values = input.svd_vals(&[0], &[1]).unwrap();
                assert_eq!(DIAGONAL_MATERIALIZATIONS.get(), 0);
                assert_eq!(compact_values.len(), dense_values.len());
                for (actual, expected) in compact_values.iter().zip(&dense_values) {
                    assert_eq!(actual.sector, expected.sector);
                    assert_eq!(actual.values.len(), expected.values.len());
                    for (&a, &b) in actual.values.iter().zip(&expected.values) {
                        assert!((a - b).abs() <= $tol * b.max(1.0));
                    }
                }
                DIAGONAL_MATERIALIZATIONS.set(0);
                let Svd { u, s, vh } = (if full {
                    input.svd_full(&[0], &[1])
                } else {
                    input.svd_compact(&[0], &[1])
                })
                .unwrap();
                assert_eq!(DIAGONAL_MATERIALIZATIONS.get(), 0);
                let rebuilt = u.compose(&s).unwrap().compose(&vh).unwrap();
                assert!(rebuilt
                    .dense_data()
                    .unwrap()
                    .iter()
                    .zip(before)
                    .all(|(a, b)| FactorScalar::widen_complex(*a - b).norm() <= $tol));
                assert_eq!(input.diagview().unwrap(), saved);
            }};
        }
        check!(
            U1FusionRule,
            [(U1Irrep::new(0), 3)],
            [SectorSpectrum {
                sector: U1Irrep::new(0),
                values: vec![-0.0_f32, -3.0, 1.0]
            }],
            f32,
            1e-5
        );
        check!(
            U1FusionRule,
            [(U1Irrep::new(0), 2)],
            [SectorSpectrum {
                sector: U1Irrep::new(0),
                values: vec![
                    num_complex::Complex32::new(1.0, 2.0),
                    num_complex::Complex32::new(-3.0, 1.0)
                ]
            }],
            num_complex::Complex32,
            1e-5
        );
        let spin0 = SU2Irrep::from_twice_spin(0);
        let spin_half = SU2Irrep::from_twice_spin(1);
        check!(
            SU2FusionRule,
            [(spin0, 2), (spin_half, 1)],
            [
                SectorSpectrum {
                    sector: spin_half,
                    values: vec![-4.0_f64]
                },
                SectorSpectrum {
                    sector: spin0,
                    values: vec![2.0, -3.0]
                },
            ],
            f64,
            1e-12
        );
        let even = product_sector(U1Irrep::new(0), Z2Irrep::EVEN);
        let odd = product_sector(U1Irrep::new(1), Z2Irrep::ODD);
        check!(
            U1FusionRule.product(FermionParityFusionRule),
            [(even, 2), (odd, 1)],
            [
                SectorSpectrum {
                    sector: odd,
                    values: vec![-4.0_f64]
                },
                SectorSpectrum {
                    sector: even,
                    values: vec![2.0, -3.0]
                },
            ],
            f64,
            1e-12
        );
        check!(
            U1FusionRule,
            [(U1Irrep::new(0), 0), (U1Irrep::new(1), 2)],
            [SectorSpectrum {
                sector: U1Irrep::new(1),
                values: vec![2.0_f64, -3.0]
            }],
            f64,
            1e-12
        );
        let empty = GradedSpace::try_new(Arc::new(U1FusionRule), []).unwrap();
        let input: TensorMap<_, f64> =
            TensorMap::diagonal(&runtime, &empty, Vec::<SectorSpectrum<_, f64>>::new()).unwrap();
        DIAGONAL_MATERIALIZATIONS.set(0);
        assert!(input.svd_vals(&[0], &[1]).unwrap().is_empty());
        assert_eq!(DIAGONAL_MATERIALIZATIONS.get(), 0);
        let Svd { u, s, vh } = (if full {
            input.svd_full(&[0], &[1])
        } else {
            input.svd_compact(&[0], &[1])
        })
        .unwrap();
        assert_eq!(DIAGONAL_MATERIALIZATIONS.get(), 0);
        assert_eq!(
            u.dense_data().unwrap().len() + vh.dense_data().unwrap().len(),
            0
        );
        assert!(s.diagview().unwrap().is_empty());
    }
}

#[test]
fn compact_diagonal_svd_preserves_dense_failure_at_c32_range_edge() {
    for full in [false, true] {
        let runtime = Runtime::builder().dense_threads(1).build().unwrap();
        let leg = GradedSpace::try_new(Arc::new(U1FusionRule), [(U1Irrep::new(0), 1)]).unwrap();
        let large = f32::MAX * 0.75;
        let input: TensorMap<_, num_complex::Complex32> = TensorMap::diagonal(
            &runtime,
            &leg,
            [SectorSpectrum {
                sector: U1Irrep::new(0),
                values: vec![num_complex::Complex32::new(large, large)],
            }],
        )
        .unwrap();
        let dense_result = if full {
            input.materialize().unwrap().svd_full(&[0], &[1])
        } else {
            input.materialize().unwrap().svd_compact(&[0], &[1])
        };
        let compact_result = if full {
            input.svd_full(&[0], &[1])
        } else {
            input.svd_compact(&[0], &[1])
        };
        assert_eq!(compact_result.is_err(), dense_result.is_err());
    }
}

#[test]
fn compact_diagonal_svd_subnormal_complex_phase_is_unit() {
    for full in [false, true] {
        let runtime = Runtime::builder().dense_threads(1).build().unwrap();
        let leg = GradedSpace::try_new(Arc::new(U1FusionRule), [(U1Irrep::new(0), 1)]).unwrap();
        let tiny = f64::from_bits(1);
        let input: TensorMap<_, Complex64> = TensorMap::diagonal(
            &runtime,
            &leg,
            [SectorSpectrum {
                sector: U1Irrep::new(0),
                values: vec![Complex64::new(tiny, tiny)],
            }],
        )
        .unwrap();
        let Svd { u, s, vh } = (if full {
            input.svd_full(&[0], &[1])
        } else {
            input.svd_compact(&[0], &[1])
        })
        .unwrap();
        assert!((vh.dense_data().unwrap()[0].norm() - 1.0).abs() < 1e-12);
        assert!((u.dense_data().unwrap()[0].norm() - 1.0).abs() < 1e-12);
        let rebuilt = u.compose(&s).unwrap().compose(&vh).unwrap();
        assert_eq!(rebuilt.dense_data().unwrap()[0], Complex64::new(tiny, tiny));
    }
}

struct NonCloneHost(Vec<f64>);

impl TensorStorage<f64> for NonCloneHost {
    fn len(&self) -> usize {
        self.0.len()
    }

    fn placement(&self) -> tenet_core::Placement {
        tenet_core::Placement::Host
    }
}

impl HostReadableStorage<f64> for NonCloneHost {
    fn as_slice(&self) -> &[f64] {
        &self.0
    }
}

#[derive(Default)]
struct FailSecondSvd {
    inner: DefaultDenseExecutor,
    calls: usize,
    record: Option<Arc<std::sync::atomic::AtomicUsize>>,
}

#[derive(Default)]
struct FailSecondQr {
    inner: DefaultDenseExecutor,
    calls: usize,
}

#[cfg(feature = "racah-generated")]
struct CountingSolve {
    inner: DefaultDenseExecutor,
    calls: Arc<std::sync::atomic::AtomicUsize>,
    failure: Option<&'static str>,
}

#[cfg(feature = "racah-generated")]
impl DenseExecutor for CountingSolve {
    fn svd(&mut self, _: DenseRead<'_>) -> Result<Vec<DenseTensor>, DenseError> {
        panic!("test only exercises solve")
    }

    fn qr(&mut self, _: DenseRead<'_>) -> Result<Vec<DenseTensor>, DenseError> {
        panic!("test only exercises solve")
    }

    fn eigh(&mut self, _: DenseRead<'_>) -> Result<Vec<DenseTensor>, DenseError> {
        panic!("test only exercises solve")
    }

    fn solve_into(
        &mut self,
        a: DenseRead<'_>,
        b: DenseRead<'_>,
        x: DenseWrite<'_>,
    ) -> Result<(), DenseError> {
        self.calls
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        if let Some(message) = self.failure {
            return Err(DenseError::Backend {
                backend: DenseBackend::Tenferro,
                op: "solve_into",
                message: message.to_string(),
            });
        }
        self.inner.solve_into(a, b, x)
    }

    fn dot_general_into(
        &mut self,
        _: DenseWrite<'_>,
        _: DenseRead<'_>,
        _: DenseRead<'_>,
        _: &DenseDotConfig,
    ) -> Result<(), DenseError> {
        panic!("test only exercises solve")
    }
}

impl DenseExecutor for FailSecondSvd {
    fn svd(&mut self, input: DenseRead<'_>) -> Result<Vec<DenseTensor>, DenseError> {
        self.calls += 1;
        if let Some(record) = &self.record {
            record.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }
        if self.calls == 2 {
            return Err(DenseError::Backend {
                backend: DenseBackend::Tenferro,
                op: "svd_into",
                message: "injected second-sector failure".to_string(),
            });
        }
        self.inner.svd(input)
    }

    fn svd_into(
        &mut self,
        input: DenseRead<'_>,
        u: DenseWrite<'_>,
        s: DenseWrite<'_>,
        vt: DenseWrite<'_>,
    ) -> Result<(), DenseError> {
        self.calls += 1;
        if let Some(record) = &self.record {
            record.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }
        if self.calls == 2 {
            return Err(DenseError::Backend {
                backend: DenseBackend::Tenferro,
                op: "svd_into",
                message: "injected second-sector failure".to_string(),
            });
        }
        self.inner.svd_into(input, u, s, vt)
    }

    fn qr(&mut self, _: DenseRead<'_>) -> Result<Vec<DenseTensor>, DenseError> {
        panic!("test only exercises SVD")
    }

    fn eigh(&mut self, _: DenseRead<'_>) -> Result<Vec<DenseTensor>, DenseError> {
        panic!("test only exercises SVD")
    }

    fn dot_general_into(
        &mut self,
        _: DenseWrite<'_>,
        _: DenseRead<'_>,
        _: DenseRead<'_>,
        _: &DenseDotConfig,
    ) -> Result<(), DenseError> {
        panic!("test only exercises SVD")
    }
}

impl DenseExecutor for FailSecondQr {
    fn svd(&mut self, _: DenseRead<'_>) -> Result<Vec<DenseTensor>, DenseError> {
        panic!("test only exercises QR")
    }

    fn qr(&mut self, _: DenseRead<'_>) -> Result<Vec<DenseTensor>, DenseError> {
        panic!("QR must use the destination API")
    }

    fn qr_into(
        &mut self,
        input: DenseRead<'_>,
        q: DenseWrite<'_>,
        r: DenseWrite<'_>,
    ) -> Result<(), DenseError> {
        self.calls += 1;
        if self.calls == 2 {
            return Err(DenseError::Backend {
                backend: DenseBackend::Tenferro,
                op: "qr_into",
                message: "injected second-sector failure".to_string(),
            });
        }
        self.inner.qr_into(input, q, r)
    }

    fn eigh(&mut self, _: DenseRead<'_>) -> Result<Vec<DenseTensor>, DenseError> {
        panic!("test only exercises QR")
    }

    fn dot_general_into(
        &mut self,
        _: DenseWrite<'_>,
        _: DenseRead<'_>,
        _: DenseRead<'_>,
        _: &DenseDotConfig,
    ) -> Result<(), DenseError> {
        panic!("test only exercises QR")
    }
}

fn owned<R, D, S>(tensor: &TensorMap<R, D, S>) -> &Arc<TypedTensorBody<R, D, S>> {
    tensor.owned_body().expect("test fixture must be owned")
}

#[cfg(feature = "racah-generated")]
#[test]
fn checked_generic_lazy_transforms_do_not_materialize_uncached_input() {
    use tenet_core::SUNFusionRule;

    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let provider = Arc::new(SUNFusionRule::new(3).unwrap());
    let fundamental = GradedSpace::try_new(Arc::clone(&provider), [(vec![1, 0], 2)]).unwrap();
    let antifundamental = GradedSpace::try_new(Arc::clone(&provider), [(vec![0, 1], 3)]).unwrap();
    let source: TensorMap<_, Complex64> = TensorMap::from_subblock_fn(
        &runtime,
        [&fundamental, &fundamental],
        [&antifundamental],
        |trees, indices| {
            Complex64::new(
                indices.iter().sum::<usize>() as f64,
                trees.coupled().iter().sum::<i64>() as f64 + 0.5,
            )
        },
    )
    .unwrap();
    let lazy = source.adjoint().unwrap();
    let eager = lazy.materialized_tensor_uncached().unwrap();
    UNCACHED_ADJOINT_MATERIALIZATIONS.set(0);

    let outputs = [
        (
            lazy.permute(&[0, 2], &[1]).unwrap(),
            eager.permute(&[0, 2], &[1]).unwrap(),
        ),
        (
            lazy.braid(&[0, 2], &[1], &[0, 1, 2]).unwrap(),
            eager.braid(&[0, 2], &[1], &[0, 1, 2]).unwrap(),
        ),
        (lazy.repartition(2).unwrap(), eager.repartition(2).unwrap()),
        (
            lazy.transpose(&[2, 1], &[0]).unwrap(),
            eager.transpose(&[2, 1], &[0]).unwrap(),
        ),
        (
            lazy.transpose(&[0, 2], &[1]).unwrap(),
            eager.transpose(&[0, 2], &[1]).unwrap(),
        ),
    ];
    let TypedTensorRepr::Adjoint(_) = &lazy.repr else {
        unreachable!()
    };
    for (actual, expected) in outputs {
        assert!(matches!(&actual.repr, TypedTensorRepr::Owned(_)));
        assert_eq!(
            actual.logical_space().space(),
            expected.logical_space().space()
        );
        assert_eq!(
            actual.dense_data().unwrap().len(),
            expected.dense_data().unwrap().len()
        );
        assert!(actual
            .dense_data()
            .unwrap()
            .iter()
            .zip(expected.dense_data().unwrap())
            .all(|(&actual, &expected)| (actual - expected).norm() < 1.0e-10));
        assert!(actual.norm(2.0).unwrap().is_finite());
        assert!(actual.qr_compact(&[0, 1], &[2]).is_ok());
    }
    assert_eq!(UNCACHED_ADJOINT_MATERIALIZATIONS.get(), 0);
}

#[test]
fn physical_projection_publishes_only_after_success_on_receiver_authority() {
    let runtime = Runtime::builder().build().unwrap();
    let provider = Arc::new(SU2FusionRule);
    let half = SU2Irrep::from_twice_spin(1);
    let leg = GradedSpace::try_new(Arc::clone(&provider), [(half, 1)]).unwrap();
    let source = TensorMap::from_subblock_fn(&runtime, [&leg, &leg, &leg], [&leg], |trees, _| {
        if trees.codomain_innerlines()[0].twice_spin() == 0 {
            1.25
        } else {
            -0.75
        }
    })
    .unwrap();
    let target =
        TensorMap::from_subblock_fn(&runtime, [&leg, &leg, &leg], [&leg], |_, _| f64::NAN).unwrap();
    let target_body = Arc::clone(owned(&target));
    let target_data = Arc::clone(&target_body.data);
    let physical = source.to_physical_dense().unwrap();
    let projected = target.project_physical_dense(&physical).unwrap();

    assert!(projected.runtime.same_runtime(&target.runtime));
    assert!(Arc::ptr_eq(
        projected.logical_space().provider_arc(),
        target.logical_space().provider_arc()
    ));
    assert_eq!(
        projected.logical_space().space(),
        target.logical_space().space()
    );
    assert!(projected
        .dense_data()
        .unwrap()
        .iter()
        .zip(source.dense_data().unwrap())
        .all(|(&actual, &expected)| (actual - expected).abs() < 2.0e-12));
    assert!(Arc::ptr_eq(owned(&target), &target_body));
    assert!(Arc::ptr_eq(&owned(&target).data, &target_data));
    assert!(target
        .dense_data()
        .unwrap()
        .iter()
        .all(|value| value.is_nan()));

    let failure = target.project_physical_dense(&PhysicalDense {
        shape: vec![2, 2],
        data: vec![0.0; 4],
    });
    assert!(failure.is_err());
    assert!(Arc::ptr_eq(owned(&target), &target_body));
    assert!(Arc::ptr_eq(&owned(&target).data, &target_data));
}

#[cfg(feature = "racah-generated")]
#[test]
fn checked_generic_contract_reuses_one_runtime_generic_lane() {
    use tenet_core::SUNFusionRule;

    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let provider = Arc::new(SUNFusionRule::new(3).unwrap());
    let leg = GradedSpace::try_new(Arc::clone(&provider), [(vec![1_i64, 1_i64], 1)]).unwrap();
    let tensor: TensorMap<_, f64> =
        TensorMap::from_subblock_fn(&runtime, [&leg, &leg], [&leg], |trees, _| {
            trees.codomain_vertices()[0].get() as f64
        })
        .unwrap();
    let identity: TensorMap<_, f64> =
        TensorMap::from_subblock_fn(&runtime, [&leg], [&leg], |_, _| 1.0).unwrap();

    for _ in 0..2 {
        let output = tensor
            .contract(
                &identity,
                &ContractSpec {
                    lhs: &[2],
                    rhs: &[0],
                    codomain: &[0, 1],
                    domain: &[2],
                },
            )
            .unwrap();
        assert_eq!(output.dense_data().unwrap(), tensor.dense_data().unwrap());
        assert!(std::ptr::eq(output.provider(), provider.as_ref()));
    }

    let mut lease = runtime.lease_context().unwrap();
    let context = lease.context();
    assert_eq!(context.generic_lane_uses(), 2);
}

fn scalar_structure(keys: &[FusionTreePairKey]) -> BlockStructure {
    BlockStructure::from_blocks(
        keys.iter()
            .cloned()
            .enumerate()
            .map(|(offset, key)| {
                let rank = key.codomain_uncoupled().len() + key.domain_uncoupled().len();
                BlockSpec::column_major_with_key(key.into(), vec![1; rank], offset).unwrap()
            })
            .collect(),
    )
    .unwrap()
}

#[test]
fn absorb_merge_matches_only_complete_asymmetric_interleaved_keys() {
    let rule = SU2FusionRule;
    let pair = |coupled, innerline| {
        let tree = FusionTreeKey::try_from_sector_ids_for_rule(
            &rule,
            [1, 1, 1],
            coupled,
            [false; 3],
            [innerline],
            [1, 1],
        )
        .unwrap();
        FusionTreePairKey::pair(tree.clone(), tree)
    };
    let inner_zero = pair(1, 0);
    let inner_two = pair(1, 2);
    let coupled_three = pair(3, 2);
    let destination_keys = vec![inner_zero.clone(), inner_two.clone(), coupled_three.clone()];

    for source_keys in [vec![inner_two], vec![inner_zero, coupled_three]] {
        let mut destination_keys = destination_keys.clone();
        destination_keys.sort();
        let mut source_keys = source_keys;
        source_keys.sort();
        let destination_structure = scalar_structure(&destination_keys);
        let source_structure = scalar_structure(&source_keys);
        let mut destination = (0..destination_keys.len())
            .map(|index| 100.0 + index as f64)
            .collect::<Vec<_>>();
        let before = destination.clone();
        let source = (0..source_keys.len())
            .map(|index| 10.0 + index as f64)
            .collect::<Vec<_>>();
        let source_values = source_keys
            .iter()
            .cloned()
            .zip(source.iter().copied())
            .collect::<HashMap<_, _>>();

        absorb_mapped(
            &destination_structure,
            &mut destination,
            &source_structure,
            &source,
            Ok,
        )
        .unwrap();

        for index in 0..destination_structure.block_count() {
            let block = destination_structure.block(index).unwrap();
            let BlockKey::FusionTree(key) = block.key() else {
                unreachable!()
            };
            assert_eq!(
                destination[block.offset()],
                source_values
                    .get(key)
                    .copied()
                    .unwrap_or(before[block.offset()])
            );
        }
    }
}

#[test]
fn coupled_region_inner_rejects_malformed_scalar_range() {
    let tensor = su2_lazy_fixture();
    let structure = owned(&tensor).space.space().structure();
    let data = tensor.dense_data().unwrap();
    let error = coupled_region_inner(
        structure,
        owned(&tensor).space.space().nout(),
        &data[..data.len() - 1],
        data,
        |_| Ok::<_, Error>(1.0),
    )
    .unwrap_err();
    assert!(matches!(error, Error::InvalidArgument(message) if
        message.contains("internal coupled-layout invariant violated")));
}

#[test]
fn coupled_region_inner_keeps_empty_and_non_fusion_boundaries() {
    let empty = BlockStructure::empty(3);
    assert_eq!(
        coupled_region_inner::<f64, _, Error>(&empty, 1, &[], &[], |_| Ok(7.0)).unwrap(),
        Complex64::new(0.0, 0.0)
    );

    let trivial = BlockStructure::trivial(&[2, 2]).unwrap();
    let error = coupled_region_inner(&trivial, 1, &[1.0; 4], &[1.0; 4], |_| Ok::<_, Error>(1.0))
        .unwrap_err();
    assert!(matches!(error, Error::InvalidArgument(message) if
        message.contains("non-packed coupled-sector layout")));
}

#[cfg(feature = "cuda")]
fn assert_cuda_tensor_matches_host<R>(
    actual: &TensorMap<R, f64>,
    expected: &TensorMap<R, f64>,
    provider: *const R,
    runtime: &crate::runtime::RuntimeIdentity,
) where
    R: MultiplicityFreeRigidSymbols<Scalar = f64> + CheckedFusionAlgebra + SectorCodec,
{
    assert!(std::ptr::eq(actual.provider(), provider));
    assert!(runtime.matches(actual.runtime()));
    assert_eq!(
        actual.logical_space().space(),
        expected.logical_space().space()
    );
    assert_eq!(actual.dense_data().unwrap(), expected.dense_data().unwrap());
    assert_eq!(actual.subblock_count(), expected.subblock_count());
    for index in 0..actual.subblock_count() {
        let actual_block = actual.subblock(index).unwrap();
        let expected_block = expected.subblock(index).unwrap();
        assert_eq!(actual_block.key(), expected_block.key());
        assert_eq!(actual_block.offset(), expected_block.offset());
        assert_eq!(actual_block.shape(), expected_block.shape());
        assert_eq!(actual_block.strides(), expected_block.strides());
        assert_eq!(
            actual.subblock_fusion_trees(index).unwrap(),
            expected.subblock_fusion_trees(index).unwrap()
        );
    }
}

#[cfg(feature = "cuda")]
fn assert_cuda_lazy_contract_orientations<R>(lhs: &TensorMap<R, f64>, rhs: &TensorMap<R, f64>)
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64> + CheckedFusionAlgebra + SectorCodec,
{
    let lhs_axes: Vec<_> = (lhs.codomain_rank()..lhs.rank()).collect();
    let rhs_axes: Vec<_> = (0..rhs.codomain_rank()).collect();
    let output_axes: Vec<_> = (0..lhs.codomain_rank() + rhs.domain_rank()).collect();
    let (codomain, domain) = output_axes.split_at(lhs.codomain_rank());
    let spec = ContractSpec {
        lhs: &lhs_axes,
        rhs: &rhs_axes,
        codomain,
        domain,
    };
    let expected_contract = lhs.contract(rhs, &spec).unwrap();
    let expected_compose = lhs.compose(rhs).unwrap();
    let provider = lhs.provider() as *const R;
    let runtime = lhs.runtime().identity();

    for (lhs_adjoint, rhs_adjoint) in [(false, false), (true, false), (false, true), (true, true)] {
        let device_operand = |logical: &TensorMap<R, f64>, adjoint: bool| {
            if adjoint {
                eager_adjoint_oracle(logical)
                    .to_cuda()
                    .unwrap()
                    .adjoint()
                    .unwrap()
            } else {
                logical.to_cuda().unwrap()
            }
        };
        let lhs_device = device_operand(lhs, lhs_adjoint);
        let rhs_device = device_operand(rhs, rhs_adjoint);
        let contract = lhs_device
            .contract(&rhs_device, &spec)
            .unwrap()
            .to_host()
            .unwrap();
        let compose = lhs_device.compose(&rhs_device).unwrap().to_host().unwrap();

        assert_cuda_tensor_matches_host(&contract, &expected_contract, provider, &runtime);
        assert_cuda_tensor_matches_host(&compose, &expected_compose, provider, &runtime);
    }
}

fn u1_lazy_fixture() -> TensorMap<U1FusionRule, f64> {
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let provider = Arc::new(U1FusionRule);
    let left = GradedSpace::try_new(
        Arc::clone(&provider),
        [(U1Irrep::new(-1), 1), (U1Irrep::new(0), 2)],
    )
    .unwrap();
    let right = GradedSpace::try_new(
        Arc::clone(&provider),
        [(U1Irrep::new(0), 3), (U1Irrep::new(1), 1)],
    )
    .and_then(|space| space.try_dual())
    .unwrap();
    let domain = GradedSpace::try_new(
        provider,
        [
            (U1Irrep::new(-1), 2),
            (U1Irrep::new(0), 1),
            (U1Irrep::new(1), 2),
        ],
    )
    .unwrap();
    TensorMap::from_subblock_fn(&runtime, [&left, &right], [&domain], |_, indices| {
        indices.iter().sum::<usize>() as f64 + 1.0
    })
    .unwrap()
}

#[test]
fn network_degeneracy_restriction_copies_nonprefix_rectangles_and_lazy_adjoint() {
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let provider = Arc::new(U1FusionRule);
    let zero = U1Irrep::new(0);
    let rows = GradedSpace::try_new(Arc::clone(&provider), [(zero, 3)]).unwrap();
    let columns = GradedSpace::try_new(Arc::clone(&provider), [(zero, 4)]).unwrap();
    let source = TensorMap::from_subblock_fn(&runtime, [&rows], [&columns], |_, indices| {
        (indices[0] + 10 * indices[1]) as f64
    })
    .unwrap();
    let zero_id = TypedSectorAdmission::try_encode_label(provider.as_ref(), &zero).unwrap();

    let direct = source
        .network_restrict_degeneracies(
            false,
            &[
                NetworkDegeneracyRestriction {
                    effective_axis: 0,
                    authority_sector: zero_id,
                    range: 1..3,
                    partner: false,
                },
                NetworkDegeneracyRestriction {
                    effective_axis: 1,
                    authority_sector: zero_id,
                    range: 2..4,
                    partner: false,
                },
            ],
        )
        .unwrap();
    assert_eq!(direct.dense_data().unwrap(), &[21.0, 22.0, 31.0, 32.0]);
    assert!(Arc::ptr_eq(
        direct.logical_space().provider_arc(),
        &provider
    ));

    let lazy = source.adjoint().unwrap();
    let restricted = lazy
        .network_restrict_degeneracies(
            false,
            &[
                NetworkDegeneracyRestriction {
                    effective_axis: 0,
                    authority_sector: zero_id,
                    range: 1..3,
                    partner: false,
                },
                NetworkDegeneracyRestriction {
                    effective_axis: 1,
                    authority_sector: zero_id,
                    range: 1..3,
                    partner: false,
                },
            ],
        )
        .unwrap();
    assert_eq!(restricted.dense_data().unwrap(), &[11.0, 21.0, 12.0, 22.0]);
}

#[test]
fn coupled_block_reads_do_not_materialize_a_lazy_adjoint() {
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let provider = Arc::new(U1FusionRule);
    let leg = GradedSpace::try_new(
        Arc::clone(&provider),
        [(U1Irrep::new(0), 2), (U1Irrep::new(1), 3)],
    )
    .unwrap();
    let source: TensorMap<_, num_complex::Complex64> =
        TensorMap::from_subblock_fn(&runtime, [&leg, &leg], [&leg], |_, index| {
            num_complex::Complex64::new(index[0] as f64, index[2] as f64 + 1.0)
        })
        .unwrap();
    let lazy = source.adjoint().unwrap();
    let clone = lazy.clone();
    let mut entries = 0;
    for (sector, block) in lazy.blocks().unwrap() {
        let again = clone.block(&sector).unwrap();
        assert_eq!((again.rows(), again.cols()), (block.rows(), block.cols()));
        for row in 0..block.rows() {
            for col in 0..block.cols() {
                assert_eq!(block.get(row, col), again.get(row, col));
                entries += 1;
            }
        }
    }
    assert!(entries > 0);
}

#[test]
fn network_degeneracy_restriction_keeps_its_validation_order_after_the_shared_kernel() {
    // The shared per-sector kernel must not move any of this path's own
    // checks: axis/duplicate, then empty range, then sector presence, then
    // the degeneracy bound, all before a destination exists.
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let provider = Arc::new(U1FusionRule);
    let zero = U1Irrep::new(0);
    let leg = GradedSpace::try_new(Arc::clone(&provider), [(zero, 3)]).unwrap();
    let source: TensorMap<_, f64> =
        TensorMap::from_subblock_fn(&runtime, [&leg], [&leg], |_, indices| indices[0] as f64)
            .unwrap();
    let zero_id = TypedSectorAdmission::try_encode_label(provider.as_ref(), &zero).unwrap();
    let absent_id =
        TypedSectorAdmission::try_encode_label(provider.as_ref(), &U1Irrep::new(4)).unwrap();
    let message = |restrictions: &[NetworkDegeneracyRestriction]| {
        let Err(error) = source.network_restrict_degeneracies(false, restrictions) else {
            panic!("request must be rejected");
        };
        error.to_string()
    };
    let restriction = |effective_axis, authority_sector, range: std::ops::Range<usize>| {
        NetworkDegeneracyRestriction {
            effective_axis,
            authority_sector,
            range,
            partner: false,
        }
    };

    // Out-of-range axis outranks an empty range and an absent sector.
    assert!(message(&[restriction(2, absent_id, 1..1)])
        .contains("invalid or duplicate effective restriction axis"));
    // A duplicate axis outranks everything that follows it.
    assert!(
        message(&[restriction(0, zero_id, 0..1), restriction(0, zero_id, 0..1)])
            .contains("invalid or duplicate effective restriction axis")
    );
    // An empty range outranks an absent sector.
    assert!(message(&[restriction(0, absent_id, 1..1)]).contains("must be nonempty"));
    // An absent sector outranks the degeneracy bound.
    assert!(message(&[restriction(0, absent_id, 0..9)]).contains("is absent from effective axis"));
    assert!(message(&[restriction(0, zero_id, 0..9)]).contains("exceeds axis"));
    // The tensor itself is untouched and a valid request still works.
    assert_eq!(
        source.dense_data().unwrap(),
        &[0.0, 1.0, 2.0, 0.0, 1.0, 2.0, 0.0, 1.0, 2.0]
    );
    assert_eq!(
        source
            .network_restrict_degeneracies(false, &[restriction(0, zero_id, 1..3)])
            .unwrap()
            .dense_data()
            .unwrap(),
        &[1.0, 2.0, 1.0, 2.0, 1.0, 2.0]
    );
}

#[test]
fn network_degeneracy_restriction_maps_effective_nonselfdual_domain_sector() {
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let provider = Arc::new(U1FusionRule);
    let plus = U1Irrep::new(1);
    let space = GradedSpace::try_new(Arc::clone(&provider), [(plus, 3)]).unwrap();
    let source = TensorMap::from_subblock_fn(&runtime, [&space], [&space], |_, indices| {
        (indices[0] + 10 * indices[1]) as f64
    })
    .unwrap();
    let plus_id = TypedSectorAdmission::try_encode_label(provider.as_ref(), &plus).unwrap();
    let restricted = source
        .network_restrict_degeneracies(
            false,
            &[
                NetworkDegeneracyRestriction {
                    effective_axis: 0,
                    authority_sector: plus_id,
                    range: 1..3,
                    partner: false,
                },
                NetworkDegeneracyRestriction {
                    effective_axis: 1,
                    authority_sector: plus_id,
                    range: 1..3,
                    partner: true,
                },
            ],
        )
        .unwrap();
    assert_eq!(restricted.dense_data().unwrap(), &[11.0, 12.0, 21.0, 22.0]);
}

#[test]
fn network_degeneracy_restriction_conjugates_complex_lazy_adjoint() {
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let provider = Arc::new(U1FusionRule);
    let zero = U1Irrep::new(0);
    let rows = GradedSpace::try_new(Arc::clone(&provider), [(zero, 2)]).unwrap();
    let columns = GradedSpace::try_new(Arc::clone(&provider), [(zero, 3)]).unwrap();
    let source = TensorMap::from_subblock_fn(&runtime, [&rows], [&columns], |_, indices| {
        Complex64::new(
            indices[0] as f64 + 10.0 * indices[1] as f64,
            indices[1] as f64 + 1.0,
        )
    })
    .unwrap();
    let sector = TypedSectorAdmission::try_encode_label(provider.as_ref(), &zero).unwrap();
    let lazy = source.adjoint().unwrap();
    let restricted = lazy
        .network_restrict_degeneracies(
            false,
            &[
                NetworkDegeneracyRestriction {
                    effective_axis: 0,
                    authority_sector: sector,
                    range: 1..3,
                    partner: false,
                },
                NetworkDegeneracyRestriction {
                    effective_axis: 1,
                    authority_sector: sector,
                    range: 0..2,
                    partner: false,
                },
            ],
        )
        .unwrap();
    assert_eq!(
        restricted.dense_data().unwrap(),
        &[
            Complex64::new(10.0, -2.0),
            Complex64::new(20.0, -3.0),
            Complex64::new(11.0, -2.0),
            Complex64::new(21.0, -3.0),
        ]
    );
}

#[test]
fn network_scatter_seals_authority_split_and_zero_block_legs_before_mutation() {
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let provider = Arc::new(U1FusionRule);
    let zero = U1Irrep::new(0);
    let full = GradedSpace::try_new(Arc::clone(&provider), [(zero, 2)]).unwrap();
    let mut destination =
        TensorMap::<_, f64>::from_subblock_fn(&runtime, [&full], [&full], |_, ij| {
            (1 + ij[0] + 2 * ij[1]) as f64
        })
        .unwrap();
    let before = destination.dense_data().unwrap().to_vec();

    let other_provider = Arc::new(U1FusionRule);
    let other = GradedSpace::try_new(other_provider, [(zero, 2)]).unwrap();
    let wrong_authority = TensorMap::<_, f64>::zeros(&runtime, [&other], [&other]).unwrap();
    assert!(destination
        .network_scatter_add_assign(&wrong_authority, &[None, None])
        .is_err());
    assert_eq!(destination.dense_data().unwrap(), before);

    let wrong_split = TensorMap::<_, f64>::zeros(&runtime, [&full, &full], []).unwrap();
    assert!(destination
        .network_scatter_add_assign(&wrong_split, &[None, None])
        .is_err());
    assert_eq!(destination.dense_data().unwrap(), before);

    // A non-vacuum rank-one map has no admissible blocks, so only logical
    // leg validation can reject malformed scatter metadata.
    let charged_full = GradedSpace::try_new(Arc::clone(&provider), [(U1Irrep::new(1), 2)]).unwrap();
    let charged_piece =
        GradedSpace::try_new(Arc::clone(&provider), [(U1Irrep::new(1), 1)]).unwrap();
    let mut empty_destination = TensorMap::<_, f64>::zeros(&runtime, [&charged_full], []).unwrap();
    let empty_piece = TensorMap::<_, f64>::zeros(&runtime, [&charged_piece], []).unwrap();
    assert_eq!(empty_destination.subblock_count(), 0);
    empty_destination
        .network_scatter_add_assign(&empty_piece, &[Some(1..2)])
        .unwrap();
    assert!(empty_destination
        .network_scatter_add_assign(&empty_piece, &[Some(2..3)])
        .is_err());

    let two_sectors =
        GradedSpace::try_new(provider, [(U1Irrep::new(1), 1), (U1Irrep::new(2), 1)]).unwrap();
    let empty_two = TensorMap::<_, f64>::zeros(&runtime, [&two_sectors], []).unwrap();
    assert_eq!(empty_two.subblock_count(), 0);
    assert!(empty_destination
        .network_scatter_add_assign(&empty_two, &[Some(0..1)])
        .is_err());
}

#[test]
fn network_scatter_reads_complex_lazy_adjoint_parent_without_materializing() {
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let provider = Arc::new(U1FusionRule);
    let plus = U1Irrep::new(1);
    let rows = GradedSpace::try_new(Arc::clone(&provider), [(plus, 2)]).unwrap();
    let columns = GradedSpace::try_new(provider, [(plus, 3)]).unwrap();
    let source = TensorMap::from_subblock_fn(&runtime, [&rows], [&columns], |_, ij| {
        Complex64::new((ij[0] + 2 * ij[1]) as f64, (1 + ij[0] + ij[1]) as f64)
    })
    .unwrap();
    let lazy = source.adjoint().unwrap();
    let codomain = lazy.codomain();
    let domain = lazy.domain();
    let mut destination = TensorMap::zeros(&runtime, codomain.iter(), domain.iter()).unwrap();
    destination
        .network_scatter_add_assign(&lazy, &[None, None])
        .unwrap();
    let expected = (0..2)
        .flat_map(|column| {
            (0..3).map(move |row| {
                Complex64::new((column + 2 * row) as f64, -((1 + column + row) as f64))
            })
        })
        .collect::<Vec<_>>();
    assert_eq!(destination.dense_data().unwrap(), expected);
    assert_eq!(DIAGONAL_MATERIALIZATIONS.get(), 0);
}

#[test]
fn storage_parameter_clone_shares_non_clone_payload() {
    let source = u1_lazy_fixture();
    let tensor: TensorMap<_, _, NonCloneHost> = TensorMap {
        runtime: source.runtime.clone(),
        repr: owned_repr(TypedTensorBody::dense(
            source.logical_space().clone(),
            NonCloneHost(source.dense_data().unwrap().to_vec()),
        )),
    };

    let twin = tensor.clone();

    assert!(Arc::ptr_eq(owned(&tensor), owned(&twin)));
    assert!(std::ptr::eq(tensor.provider(), twin.provider()));
    assert_eq!(tensor.dense_data().unwrap(), twin.dense_data().unwrap());
}

#[test]
fn typed_placement_is_diagnostic_for_dense_compact_and_lazy_host_storage() {
    DIAGONAL_MATERIALIZATIONS.set(0);
    let source = u1_lazy_fixture();
    let diagonal = source.svd_compact(&[0, 1], &[2]).unwrap().s;
    let lazy = source.adjoint().unwrap();

    assert_eq!(source.placement(), Placement::Host);
    assert_eq!(diagonal.placement(), Placement::Host);
    assert_eq!(lazy.placement(), Placement::Host);
    assert_eq!(DIAGONAL_MATERIALIZATIONS.get(), 0);
}

#[test]
fn typed_zeros_like_is_exact_and_representation_preserving() {
    DIAGONAL_MATERIALIZATIONS.set(0);
    let source = u1_lazy_fixture();
    let values = [f64::NAN, f64::INFINITY, f64::NEG_INFINITY, 0.0, -0.0];
    let index = std::cell::Cell::new(0usize);
    let dense = TensorMap::from_subblock_fn(
        source.runtime(),
        &source.codomain(),
        &source.domain(),
        |_, _| {
            let i = index.get();
            index.set(i + 1);
            values[i % values.len()]
        },
    )
    .unwrap();
    let source_bits: Vec<_> = dense
        .dense_data()
        .unwrap()
        .iter()
        .map(|value| value.to_bits())
        .collect();
    let provider = dense.provider() as *const _;
    let zero = dense.zeros_like();
    assert!(zero
        .dense_data()
        .unwrap()
        .iter()
        .all(|value| value.to_bits() == 0));
    assert_eq!(
        dense
            .dense_data()
            .unwrap()
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>(),
        source_bits
    );
    assert!(std::ptr::eq(zero.provider(), provider));
    assert!(zero.runtime().same_runtime(dense.runtime()));
    assert_eq!(zero.logical_space().space(), dense.logical_space().space());

    let complex = dense.convert::<Complex64>();
    let complex = complex.with_data(
        complex
            .dense_data()
            .unwrap()
            .iter()
            .enumerate()
            .map(|(i, _)| {
                num_complex::Complex64::new(
                    values[i % values.len()],
                    values[(i + 1) % values.len()],
                )
            })
            .collect(),
    );
    let complex_zero = complex.zeros_like();
    assert!(complex_zero
        .dense_data()
        .unwrap()
        .iter()
        .all(|value| value.re.to_bits() == 0 && value.im.to_bits() == 0));

    let compact = source.svd_compact(&[0, 1], &[2]).unwrap().s;
    let compact = compact.with_spectrum(
        compact
            .spectrum()
            .unwrap()
            .iter()
            .map(|entry| tenet_matrixalgebra::SectorSpectrum {
                sector: entry.sector,
                values: (0..entry.values.len())
                    .map(|i| values[i % values.len()])
                    .collect(),
            })
            .collect(),
    );
    let compact_zero = compact.zeros_like();
    assert!(matches!(
        owned(&compact_zero).data.as_ref(),
        TypedData::Diagonal(_)
    ));
    assert!(compact_zero
        .spectrum()
        .unwrap()
        .iter()
        .flat_map(|entry| &entry.values)
        .all(|value| value.to_bits() == 0));
    assert_eq!(DIAGONAL_MATERIALIZATIONS.get(), 0);

    let lazy = dense.adjoint().unwrap();
    let lazy_zero = lazy.zeros_like();
    assert!(matches!(lazy_zero.repr, TypedTensorRepr::Adjoint(_)));
    assert!(std::ptr::eq(lazy_zero.provider(), provider));

    let empty_leg = GradedSpace::try_new(Arc::new(U1FusionRule), [(U1Irrep::new(0), 0)]).unwrap();
    let empty =
        TensorMap::from_subblock_fn(source.runtime(), [&empty_leg], [&empty_leg], |_, _| {
            f64::NAN
        })
        .unwrap();
    assert!(empty.dense_data().unwrap().is_empty());
    assert!(empty.zeros_like().dense_data().unwrap().is_empty());
}

#[cfg(feature = "cuda")]
#[test]
fn typed_cuda_owned_metadata_validation_orders_ordinal_before_length() {
    type DeviceTensor = TensorMap<U1FusionRule, f64, CudaStorage>;
    assert!(DeviceTensor::validate_cuda_owned_metadata(
        Placement::Cuda(0),
        Placement::Cuda(0),
        7,
        7
    )
    .is_ok());
    assert_eq!(
        DeviceTensor::validate_cuda_owned_metadata(Placement::Cuda(1), Placement::Cuda(0), 7, 6)
            .unwrap_err(),
        Error::PlacementMismatch
    );
    assert!(matches!(
        DeviceTensor::validate_cuda_owned_metadata(
            Placement::Cuda(0),
            Placement::Cuda(0),
            7,
            6
        ),
        Err(Error::InvalidArgument(message)) if message.contains("payload length")
    ));
}

/// `(factor_copies, selector_uploads, assembly_gemms)` a compact QR or
/// compact SVD assembly must perform for this plan: an aligned side is one
/// whole-factor copy and no selector upload, a non-aligned side is one GEMM
/// per nonempty target tree.
#[cfg(feature = "cuda")]
fn cuda_route_assembly_counts<R>(plan: &TypedCudaQrPlan<R>) -> (usize, usize, usize) {
    let nonempty_trees = |trees: &[CoupledTreeExtent]| {
        trees
            .iter()
            .filter(|tree| tree.extent().is_ok_and(|extent| extent != 0))
            .count()
    };
    let mut counts = (0, 0, 0);
    for route in &plan.routes {
        if route.aligned_left {
            counts.0 += 1;
        } else {
            counts.2 += nonempty_trees(plan.left_regions[route.left].row_trees());
        }
        if route.aligned_right {
            counts.0 += 1;
        } else {
            counts.2 += nonempty_trees(plan.right_regions[route.right].col_trees());
        }
        if !(route.aligned_left && route.aligned_right) {
            counts.1 += 1;
        }
    }
    counts
}

#[cfg(feature = "cuda")]
#[test]
fn typed_cuda_qr_tree_route_validation_is_order_independent_and_bijective() {
    let source = u1_lazy_fixture();
    let regions = sector_regions(
        source.logical_space().space().structure(),
        source.logical_space().space().nout(),
    )
    .unwrap();
    let trees = regions
        .iter()
        .flat_map(|region| [region.row_trees(), region.col_trees()])
        .find(|trees| trees.len() > 1)
        .expect("fixture must contain a multi-tree coupled sector");
    let mut reordered = trees.to_vec();
    reordered.reverse();
    assert!(cuda_qr_tree_extents_match(trees, &reordered).unwrap());
    reordered.pop();
    assert!(!cuda_qr_tree_extents_match(trees, &reordered).unwrap());

    // The aligned-copy dispatch is the stricter, order-sensitive predicate:
    // a permuted tree sequence carries the same blocks but a different
    // layout, so it must fall back to the per-tree GEMM.
    let extent: usize = trees.iter().map(|tree| tree.extent().unwrap()).sum();
    assert!(cuda_factor_layout_is_aligned(trees, trees, extent).unwrap());
    let mut permuted = trees.to_vec();
    permuted.reverse();
    assert!(!cuda_factor_layout_is_aligned(trees, &permuted, extent).unwrap());
    assert!(
        !cuda_factor_layout_is_aligned(trees, trees, extent + 1).unwrap(),
        "trees that do not tile the region are never aligned"
    );
}

#[cfg(feature = "cuda")]
#[test]
fn typed_cuda_factorizations_reject_compact_lazy_and_truncation_before_runtime_work() {
    let diagonal = u1_lazy_fixture().svd_compact(&[0, 1], &[2]).unwrap().s;
    let TypedData::Diagonal(spectrum) = owned(&diagonal).data.as_ref() else {
        unreachable!("SVD factor is compact")
    };
    let device_diagonal: TensorMap<_, f64, CudaStorage> = TensorMap {
        runtime: diagonal.runtime.clone(),
        repr: owned_repr(TypedTensorBody::new(
            diagonal.logical_space().clone(),
            TypedData::<f64, CudaStorage>::Diagonal(spectrum.clone()),
        )),
    };
    assert!(matches!(
        device_diagonal.qr_compact(&codomain_axes(&device_diagonal), &domain_axes(&device_diagonal)),
        Err(Error::UnsupportedOnDevice(message)) if message.contains("dense CUDA storage")
    ));
    assert!(matches!(
        device_diagonal.svd_compact(&codomain_axes(&device_diagonal), &domain_axes(&device_diagonal)),
        Err(Error::UnsupportedOnDevice(message)) if message.contains("dense CUDA storage")
    ));
    assert!(matches!(
        device_diagonal.eigh_full(&codomain_axes(&device_diagonal), &domain_axes(&device_diagonal)),
        Err(Error::UnsupportedOnDevice(message)) if message.contains("dense CUDA storage")
    ));
    // #1452: the adjoint of a compact diagonal is the owned (conjugated)
    // diagonal on every storage, never a lazy view, so it is rejected as
    // compact storage. The lazy-operand rejection needs a dense device
    // parent and is covered by
    // `typed_cuda_factorizations_reject_lazy_adjoint_before_runtime_work`.
    let adjoint = device_diagonal.adjoint().unwrap();
    assert!(Arc::ptr_eq(owned(&adjoint), owned(&device_diagonal)));
    assert!(matches!(
        adjoint.qr_compact(&codomain_axes(&adjoint), &domain_axes(&adjoint)),
        Err(Error::UnsupportedOnDevice(message)) if message.contains("dense CUDA storage")
    ));
    assert!(matches!(
        adjoint.svd_compact(&codomain_axes(&adjoint), &domain_axes(&adjoint)),
        Err(Error::UnsupportedOnDevice(message)) if message.contains("dense CUDA storage")
    ));
    assert!(matches!(
        adjoint.eigh_full(&codomain_axes(&adjoint), &domain_axes(&adjoint)),
        Err(Error::UnsupportedOnDevice(message)) if message.contains("dense CUDA storage")
    ));

    let complex_spectrum: Vec<_> = spectrum
        .iter()
        .map(|entry| tenet_matrixalgebra::SectorSpectrum {
            sector: entry.sector,
            values: entry
                .values
                .iter()
                .map(|&value| num_complex::Complex64::new(value, value + 1.0))
                .collect(),
        })
        .collect();
    let complex_diagonal: TensorMap<_, num_complex::Complex64, CudaStorage<_>> = TensorMap {
        runtime: diagonal.runtime.clone(),
        repr: owned_repr(TypedTensorBody::diagonal(
            diagonal.logical_space().clone(),
            complex_spectrum.clone(),
        )),
    };
    let complex_adjoint = complex_diagonal.adjoint().unwrap();
    let TypedData::Diagonal(conjugated) = owned(&complex_adjoint).data.as_ref() else {
        unreachable!("the adjoint of a compact diagonal is compact")
    };
    for (actual, source) in conjugated.iter().zip(&complex_spectrum) {
        assert_eq!(actual.sector, source.sector);
        let expected: Vec<_> = source.values.iter().map(|value| value.conj()).collect();
        assert_eq!(actual.values, expected);
    }
}

#[cfg(feature = "cuda")]
#[test]
#[ignore = "requires a real CUDA device"]
fn typed_cuda_factorizations_reject_lazy_adjoint_before_runtime_work() {
    let runtime = Runtime::builder().cuda(0).dense_threads(1).build().unwrap();
    let leg = GradedSpace::try_new(
        Arc::new(U1FusionRule),
        [(U1Irrep::new(0), 2), (U1Irrep::new(1), 1)],
    )
    .unwrap();
    let source: TensorMap<_, f64> =
        TensorMap::from_subblock_fn(&runtime, [&leg], [&leg], |_, indices| {
            indices.iter().sum::<usize>() as f64 + 1.0
        })
        .unwrap();
    let lazy = source.to_cuda().unwrap().adjoint().unwrap();
    assert!(matches!(&lazy.repr, TypedTensorRepr::Adjoint(_)));
    assert!(matches!(
        lazy.qr_compact(&codomain_axes(&lazy), &domain_axes(&lazy)),
        Err(Error::UnsupportedOnDevice(message)) if message.contains("lazy adjoint")
    ));
    assert!(matches!(
        lazy.svd_compact(&codomain_axes(&lazy), &domain_axes(&lazy)),
        Err(Error::UnsupportedOnDevice(message)) if message.contains("lazy adjoint")
    ));
    assert!(matches!(
        lazy.eigh_full(&codomain_axes(&lazy), &domain_axes(&lazy)),
        Err(Error::UnsupportedOnDevice(message)) if message.contains("lazy adjoint")
    ));
}

#[cfg(feature = "cuda")]
#[test]
#[ignore = "requires a real CUDA device"]
fn typed_cuda_eigh_full_matches_host_without_hidden_materialization() {
    let runtime = Runtime::builder().cuda(0).dense_threads(1).build().unwrap();
    let provider = Arc::new(U1FusionRule);
    let leg = GradedSpace::try_new(
        Arc::clone(&provider),
        [(U1Irrep::new(-1), 2), (U1Irrep::new(0), 3)],
    )
    .unwrap();
    let source = TensorMap::from_subblock_fn(&runtime, [&leg], [&leg], |_, indices| {
        let (row, col) = (indices[0], indices[1]);
        if row == col {
            row as f64 + 1.0
        } else if row.abs_diff(col) == 1 {
            0.125
        } else {
            0.0
        }
    })
    .unwrap();
    let device = source.to_cuda().unwrap();

    let expected_full = source
        .eigh_full(&codomain_axes(&source), &domain_axes(&source))
        .unwrap();
    let Eigh {
        d: d_device,
        v: v_device,
    } = device
        .eigh_full(&codomain_axes(&device), &domain_axes(&device))
        .unwrap();
    assert_eq!(d_device.placement(), Placement::Cuda(0));
    assert_eq!(v_device.placement(), Placement::Cuda(0));
    assert!(Arc::ptr_eq(
        v_device.logical_space().provider_arc(),
        source.logical_space().provider_arc()
    ));
    let d = d_device.to_host().unwrap();
    let v = v_device.to_host().unwrap();
    assert_typed_map_close(&d, &expected_full.d, 1.0e-10);
    assert_typed_map_close(
        &source.compose(&v).unwrap(),
        &v.compose(&d).unwrap(),
        1.0e-10,
    );

    let su2_provider = Arc::new(SU2FusionRule);
    let su2_leg = GradedSpace::try_new(
        Arc::clone(&su2_provider),
        [
            (SU2Irrep::from_twice_spin(0), 2),
            (SU2Irrep::from_twice_spin(1), 2),
        ],
    )
    .unwrap();
    let su2_source = TensorMap::from_subblock_fn(&runtime, [&su2_leg], [&su2_leg], |_, indices| {
        if indices[0] == indices[1] {
            indices[0] as f64 + 1.0
        } else {
            0.25
        }
    })
    .unwrap();
    assert!(su2_source.subblock_count() >= 2);
    let su2_device = su2_source.to_cuda().unwrap();
    let Eigh { d: su2_d, v: su2_v } = su2_device
        .eigh_full(&codomain_axes(&su2_device), &domain_axes(&su2_device))
        .unwrap();
    assert!(Arc::ptr_eq(
        su2_v.logical_space().provider_arc(),
        su2_source.logical_space().provider_arc()
    ));
    let su2_d = su2_d.to_host().unwrap();
    let su2_v = su2_v.to_host().unwrap();
    assert_typed_map_close(
        &su2_source.compose(&su2_v).unwrap(),
        &su2_v.compose(&su2_d).unwrap(),
        1.0e-10,
    );

    let input_before_failure = device.to_host().unwrap();
    for failure in [("decomposition", 2), ("assembly", 2)] {
        CUDA_EIGH_FAILURE.with(|injected| injected.set(Some(failure)));
        assert!(device
            .eigh_full(&codomain_axes(&device), &domain_axes(&device))
            .is_err());
        CUDA_EIGH_FAILURE.with(|injected| injected.set(None));
        assert_typed_map_close(
            &device.to_host().unwrap(),
            &input_before_failure,
            f64::EPSILON,
        );
    }

    let nonhermitian = TensorMap::from_subblock_fn(&runtime, [&leg], [&leg], |_, indices| {
        match (indices[0], indices[1]) {
            (0, 1) => 1.0,
            _ => 0.0,
        }
    })
    .unwrap()
    .to_cuda()
    .unwrap();
    assert!(matches!(
        nonhermitian.eigh_full(&codomain_axes(&nonhermitian), &domain_axes(&nonhermitian)),
        Err(Error::Operation(error))
            if matches!(
                error.as_ref(),
                tenet_tensors::OperationError::UnsupportedTensorContractScope { .. }
            )
    ));
}

/// Z2 endomorphism whose canonical sector blocks are `[[2,1],[1,2]]`
/// (spectrum {3, 1}), stored with rows stacked by ascending codomain tree
/// and columns by descending domain tree, so each block reads
/// `[[1,2],[2,1]]`: still Hermitian, but with spectrum {3, -1}.
fn mis_stacked_hermitian_z2(runtime: &Runtime) -> TensorMap<Z2FusionRule, f64> {
    let rule = Z2FusionRule;
    let leg = || {
        SectorLeg::new(
            [
                (Z2Irrep::new(0).sector_id(), 1),
                (Z2Irrep::new(1).sector_id(), 1),
            ],
            false,
        )
    };
    let homspace = FusionTreeHomSpace::new(
        FusionProductSpace::new([leg(), leg()]),
        FusionProductSpace::new([leg(), leg()]),
    );
    let mut blocks: Vec<(FusionTreePairKey, Vec<usize>)> = homspace
        .fusion_tree_keys(&rule)
        .iter()
        .map(|key| (key.clone(), vec![1; 4]))
        .collect();
    blocks.sort_by(|(a, _), (b, _)| {
        a.codomain_tree()
            .cmp(b.codomain_tree())
            .then(b.domain_tree().cmp(a.domain_tree()))
    });
    let structure = BlockStructure::coupled_sector_matrix_with_keys(&rule, 2, 4, blocks).unwrap();
    let regions = structure.coupled_sector_regions(2).unwrap().unwrap();
    assert!(regions
        .iter()
        .all(|region| region.row_trees() != region.col_trees()));
    let space = tenet_core::FusionTensorMapSpace::new_unbound(
        tenet_core::TensorMapSpace::<2, 2>::from_dims([2, 2], [2, 2]).unwrap(),
        homspace,
        structure,
    )
    .unwrap()
    .try_bind_rule(&rule)
    .unwrap();
    let core = tenet_core::TensorMap::<f64, 2, 2>::from_block_fn_with_fusion_space(
        space,
        0.0,
        |key, _| {
            let BlockKey::FusionTree(tree) = key else {
                unreachable!("fusion-tree blocks")
            };
            if tree.codomain_tree() == tree.domain_tree() {
                2.0
            } else {
                1.0
            }
        },
    )
    .unwrap();
    let space = BoundDynamicFusionMapSpace::bind_multiplicity_free(
        tenet_tensors::DynamicFusionMapSpace::from_typed(core.fusion_space().unwrap()),
        Arc::new(rule),
    )
    .unwrap();
    TensorMap {
        runtime: runtime.clone(),
        repr: owned_repr(TypedTensorBody {
            space,
            data: Arc::new(TypedData::Dense(core.data().to_vec())),
        }),
    }
}

#[test]
fn host_eigh_refuses_a_mis_stacked_block_that_stays_hermitian() {
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let tensor = mis_stacked_hermitian_z2(&runtime);
    let error = format!("{:?}", tensor.eigh_full(&[0, 1], &[2, 3]).err());
    assert!(error.contains("eigh_full requires identical"), "{error}");
}

#[cfg(feature = "cuda")]
#[test]
#[ignore = "requires a real CUDA device"]
fn typed_cuda_eigh_refuses_a_mis_stacked_block_that_stays_hermitian() {
    // What: the device path reads the same tiling as the host and must
    // refuse it too, instead of returning the {3, -1} spectrum of the
    // column-permuted block.
    let runtime = Runtime::builder().cuda(0).dense_threads(1).build().unwrap();
    let device = mis_stacked_hermitian_z2(&runtime).to_cuda().unwrap();
    assert!(matches!(
        device.eigh_full(&codomain_axes(&device), &domain_axes(&device)),
        Err(Error::Operation(error))
            if matches!(
                error.as_ref(),
                tenet_tensors::OperationError::UnsupportedTensorContractScope { message }
                    if message.starts_with("eigh_full requires identical")
            )
    ));
}

#[cfg(feature = "cuda")]
#[test]
#[ignore = "requires a real CUDA device"]
fn typed_cuda_eigh_aligned_assembly_matches_the_per_tree_path_bitwise() {
    // What: an aligned route permutes its eigenvectors with one GEMM, the
    // general path with one per codomain tree; both read one selector
    // uploaded per call and, being exact data movement, publish the same
    // bytes.
    let runtime = Runtime::builder().cuda(0).dense_threads(1).build().unwrap();
    let leg = GradedSpace::try_new(
        Arc::new(U1FusionRule),
        [
            (U1Irrep::new(-1), 2),
            (U1Irrep::new(0), 1),
            (U1Irrep::new(1), 2),
        ],
    )
    .unwrap();
    let source =
        TensorMap::<U1FusionRule, f64>::rand_with_seed(&runtime, [&leg, &leg], [&leg, &leg], 11)
            .unwrap();
    let source = source.axpby(1.0, &source.adjoint().unwrap(), 1.0).unwrap();
    let device = source.to_cuda().unwrap();
    let sectors = sector_regions(
        device.logical_space().space().structure(),
        device.logical_space().space().nout(),
    )
    .unwrap();
    let trees: usize = sectors.iter().map(|region| region.row_trees().len()).sum();
    assert!(
        trees > sectors.len(),
        "the fixture needs multi-tree sectors"
    );

    let mut outputs = Vec::new();
    for (treewise, gemms) in [(false, sectors.len()), (true, trees)] {
        CUDA_EIGH_TREEWISE.with(|flag| flag.set(treewise));
        CUDA_QR_OBSERVATION.with(|observation| observation.set(Some((0, 0, 0, 0, 0, 0, 0))));
        CUDA_EIGH_SELECTOR_UPLOADS.with(|uploads| uploads.set(Some(0)));
        let Eigh { d, v } = device
            .eigh_full(&codomain_axes(&device), &domain_axes(&device))
            .unwrap();
        CUDA_EIGH_TREEWISE.with(|flag| flag.set(false));
        assert_eq!(
            CUDA_EIGH_SELECTOR_UPLOADS.with(|uploads| uploads.replace(None)),
            Some(1)
        );
        CUDA_QR_OBSERVATION.with(|observation| {
            let (_, _, _, _, assembly_gemms, _, _) = observation.get().unwrap();
            assert_eq!(assembly_gemms, gemms);
            observation.set(None);
        });
        outputs.push((d.to_host().unwrap(), v.to_host().unwrap()));
    }
    let bits = |map: &TensorMap<U1FusionRule, f64>| {
        map.dense_data()
            .unwrap()
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>()
    };
    assert_eq!(bits(&outputs[0].0), bits(&outputs[1].0));
    assert_eq!(bits(&outputs[0].1), bits(&outputs[1].1));
    let (d, v) = &outputs[0];
    assert_typed_map_close(&source.compose(v).unwrap(), &v.compose(d).unwrap(), 1.0e-10);
}

#[cfg(feature = "cuda")]
#[test]
#[ignore = "requires a real CUDA device"]
fn typed_cuda_qr_work_and_preflight_are_streamed_and_transactional() {
    let runtime = Runtime::builder().cuda(0).dense_threads(1).build().unwrap();
    let provider = Arc::new(SU2FusionRule);
    let leg = GradedSpace::try_new(
        Arc::clone(&provider),
        [
            (SU2Irrep::from_twice_spin(0), 2),
            (SU2Irrep::from_twice_spin(1), 2),
        ],
    )
    .unwrap();
    let source = TensorMap::from_subblock_fn(&runtime, [&leg, &leg], [&leg], |_, indices| {
        indices.iter().sum::<usize>() as f64 + 1.0
    })
    .unwrap();
    let regions = sector_regions(
        source.logical_space().space().structure(),
        source.logical_space().space().nout(),
    )
    .unwrap();
    let nonempty = regions
        .iter()
        .filter(|region| region.rows() != 0 && region.cols() != 0)
        .count();
    let source_device = source.to_cuda().unwrap();
    // Per-route transfer and kernel counts follow the proved layout flag.
    let plan = source_device
        .compile_cuda_qr_plan(Arc::clone(&regions))
        .unwrap();
    let (factor_copies, selector_uploads, assembly_gemms) = cuda_route_assembly_counts(&plan);
    assert_eq!(plan.routes.len(), nonempty);
    // Both factor spaces of this fixture reproduce the source tree layout,
    // so every route takes the whole-factor copy and the assembly uploads
    // and downloads nothing. The non-aligned fallback is a layout
    // property, not a workload one, and is covered by
    // `typed_cuda_qr_tree_route_validation_is_order_independent_and_bijective`.
    assert!(
        plan.routes
            .iter()
            .all(|route| route.aligned_left && route.aligned_right),
        "expected an all-aligned route mix, got {:?}",
        plan.routes
            .iter()
            .map(|route| (route.aligned_left, route.aligned_right))
            .collect::<Vec<_>>()
    );
    assert_eq!(
        (factor_copies, selector_uploads, assembly_gemms),
        (2 * plan.routes.len(), 0, 0)
    );

    CUDA_QR_OBSERVATION.with(|observation| observation.set(Some((0, 0, 0, 0, 0, 0, 0))));
    source_device
        .qr_compact(&codomain_axes(&source_device), &domain_axes(&source_device))
        .unwrap()
        .pair();
    CUDA_QR_OBSERVATION.with(|observation| {
        assert_eq!(
            observation.get(),
            Some((
                nonempty,
                factor_copies,
                selector_uploads,
                2,
                assembly_gemms,
                0,
                usize::from(nonempty != 0),
            ))
        );
        observation.set(None);
    });

    let malformed_storage = {
        let lease = runtime.lease_cuda().unwrap();
        CudaStorage::<f64>::upload(&lease, &[]).unwrap()
    };
    let malformed = TensorMap {
        runtime: runtime.clone(),
        repr: owned_repr(TypedTensorBody::dense(
            source.logical_space().clone(),
            malformed_storage,
        )),
    };
    let sentinel = (usize::MAX, 0, 0, 0, 0, 0, 0);
    CUDA_QR_OBSERVATION.with(|observation| observation.set(Some(sentinel)));
    assert!(matches!(
        malformed.qr_compact(&codomain_axes(&malformed), &domain_axes(&malformed)),
        Err(Error::InvalidArgument(message)) if message.contains("payload length")
    ));
    CUDA_QR_OBSERVATION.with(|observation| {
        assert_eq!(observation.get(), Some(sentinel));
        observation.set(None);
    });

    let stranded_storage = {
        let lease = runtime.lease_cuda().unwrap();
        CudaStorage::upload(&lease, source.dense_data().unwrap()).unwrap()
    };
    let stranded = TensorMap {
        runtime: Runtime::builder().build().unwrap(),
        repr: owned_repr(TypedTensorBody::dense(
            source.logical_space().clone(),
            stranded_storage,
        )),
    };
    CUDA_QR_OBSERVATION.with(|observation| observation.set(Some(sentinel)));
    assert!(matches!(
        stranded.qr_compact(&codomain_axes(&stranded), &domain_axes(&stranded)),
        Err(Error::InvalidArgument(message)) if message.contains("without a CUDA device")
    ));
    CUDA_QR_OBSERVATION.with(|observation| {
        assert_eq!(observation.get(), Some(sentinel));
        observation.set(None);
    });

    let zn3 = Arc::new(ZNFusionRule::new(3).unwrap());
    let charge0 = GradedSpace::try_new(Arc::clone(&zn3), [(zn3.irrep(0), 1)]).unwrap();
    let charge1 = GradedSpace::try_new(Arc::clone(&zn3), [(zn3.irrep(1), 1)]).unwrap();
    let empty: TensorMap<_, f64> =
        TensorMap::from_subblock_fn(&runtime, [&charge0], [&charge1], |_, _| 1.0).unwrap();
    CUDA_QR_OBSERVATION.with(|observation| observation.set(Some((0, 0, 0, 0, 0, 0, 0))));
    empty
        .to_cuda()
        .unwrap()
        .qr_compact(&codomain_axes(&empty), &domain_axes(&empty))
        .unwrap()
        .pair();
    CUDA_QR_OBSERVATION.with(|observation| {
        assert_eq!(observation.get(), Some((0, 0, 0, 2, 0, 0, 0)));
        observation.set(None);
    });
}

#[cfg(feature = "cuda")]
#[test]
#[ignore = "requires a real CUDA device"]
fn typed_cuda_svd_work_is_streamed_and_preflight_is_transactional() {
    let runtime = Runtime::builder().cuda(0).dense_threads(1).build().unwrap();
    let provider = Arc::new(SU2FusionRule);
    let leg = GradedSpace::try_new(
        Arc::clone(&provider),
        [
            (SU2Irrep::from_twice_spin(0), 2),
            (SU2Irrep::from_twice_spin(1), 2),
        ],
    )
    .unwrap();
    let source = TensorMap::from_subblock_fn(&runtime, [&leg, &leg], [&leg], |_, indices| {
        indices.iter().sum::<usize>() as f64 + 1.0
    })
    .unwrap();
    let regions = sector_regions(
        source.logical_space().space().structure(),
        source.logical_space().space().nout(),
    )
    .unwrap();
    let nonempty = regions
        .iter()
        .filter(|region| region.rows() != 0 && region.cols() != 0)
        .count();
    let singular_values = regions
        .iter()
        .map(|region| region.rows().min(region.cols()))
        .sum();
    let source_device = source.to_cuda().unwrap();
    // Compact SVD assembles through the same aligned-copy dispatch as QR
    // and shares its copy/selector/GEMM observation.
    let plan = source_device
        .compile_cuda_qr_plan(Arc::clone(&regions))
        .unwrap();
    let (factor_copies, _, assembly_gemms) = cuda_route_assembly_counts(&plan);
    CUDA_SVD_OBSERVATION.with(|observation| observation.set(Some((0, 0, 0, 0, 0))));
    CUDA_QR_OBSERVATION.with(|observation| observation.set(Some((0, 0, 0, 0, 0, 0, 0))));
    source_device
        .svd_compact(&codomain_axes(&source_device), &domain_axes(&source_device))
        .unwrap();
    CUDA_SVD_OBSERVATION.with(|observation| {
        assert_eq!(
            observation.get(),
            Some((nonempty, singular_values, 3, 0, usize::from(nonempty != 0),))
        );
        observation.set(None);
    });
    CUDA_QR_OBSERVATION.with(|observation| {
        // No device QR and no QR output upload happen here; the shared
        // slots record only this assembly's copies and GEMMs. The SVD
        // builds its gauge selectors on the device, so it uploads none.
        assert_eq!(
            observation.get(),
            Some((0, factor_copies, 0, 0, assembly_gemms, 0, 0))
        );
        observation.set(None);
    });

    let malformed_storage = {
        let lease = runtime.lease_cuda().unwrap();
        CudaStorage::<f64>::upload(&lease, &[]).unwrap()
    };
    let malformed = TensorMap {
        runtime: runtime.clone(),
        repr: owned_repr(TypedTensorBody::dense(
            source.logical_space().clone(),
            malformed_storage,
        )),
    };
    for lazy in [false, true] {
        CUDA_SVD_OBSERVATION.with(|observation| observation.set(Some((0, 0, 0, 0, 0))));
        let rejected = if lazy {
            source_device.adjoint().unwrap().svd_compact(&[0], &[1, 2])
        } else {
            malformed.svd_compact(&codomain_axes(&malformed), &domain_axes(&malformed))
        };
        assert!(rejected.is_err());
        CUDA_SVD_OBSERVATION.with(|observation| {
            assert_eq!(observation.get(), Some((0, 0, 0, 0, 0)));
            observation.set(None);
        });
    }

    let stranded_storage = {
        let lease = runtime.lease_cuda().unwrap();
        CudaStorage::upload(&lease, source.dense_data().unwrap()).unwrap()
    };
    let stranded = TensorMap {
        runtime: Runtime::builder().build().unwrap(),
        repr: owned_repr(TypedTensorBody::dense(
            source.logical_space().clone(),
            stranded_storage,
        )),
    };
    CUDA_SVD_OBSERVATION.with(|observation| observation.set(Some((0, 0, 0, 0, 0))));
    assert!(stranded
        .svd_compact(&codomain_axes(&stranded), &domain_axes(&stranded))
        .is_err());
    CUDA_SVD_OBSERVATION.with(|observation| {
        assert_eq!(observation.get(), Some((0, 0, 0, 0, 0)));
        observation.set(None);
    });
}

#[cfg(feature = "cuda")]
#[test]
#[ignore = "requires a real CUDA device"]
fn typed_cuda_svd_non_aligned_routes_upload_no_selector_and_the_same_diagonal() {
    // What: the rustdoc's upload count, three zero-initialized factors
    // and no selector upload even on non-aligned routes (the gauge
    // selector is built on the device), and a diagonal that does not
    // depend on the assembly path.
    let runtime = Runtime::builder().cuda(0).dense_threads(1).build().unwrap();
    let leg = GradedSpace::try_new(
        Arc::new(U1FusionRule),
        [
            (U1Irrep::new(-1), 2),
            (U1Irrep::new(0), 1),
            (U1Irrep::new(1), 2),
        ],
    )
    .unwrap();
    let source =
        TensorMap::<U1FusionRule, f64>::rand_with_seed(&runtime, [&leg, &leg], [&leg], 13).unwrap();
    let device = source.to_cuda().unwrap();
    let routes = sector_regions(
        source.logical_space().space().structure(),
        source.logical_space().space().nout(),
    )
    .unwrap()
    .iter()
    .filter(|region| region.rows() != 0 && region.cols() != 0)
    .count();
    let mut diagonals = Vec::new();
    for (treewise, selectors) in [(false, 0), (true, 0)] {
        CUDA_SVD_TREEWISE.with(|flag| flag.set(treewise));
        CUDA_SVD_OBSERVATION.with(|observation| observation.set(Some((0, 0, 0, 0, 0))));
        CUDA_QR_OBSERVATION.with(|observation| observation.set(Some((0, 0, 0, 0, 0, 0, 0))));
        let Svd { s, .. } = device
            .svd_compact(&codomain_axes(&device), &domain_axes(&device))
            .unwrap();
        CUDA_SVD_TREEWISE.with(|flag| flag.set(false));
        let (_, _, creations, _, _) = CUDA_SVD_OBSERVATION
            .with(|observation| observation.replace(None))
            .unwrap();
        let (_, _, selector_uploads, _, _, _, _) = CUDA_QR_OBSERVATION
            .with(|observation| observation.replace(None))
            .unwrap();
        assert_eq!(
            (creations, selector_uploads),
            (3, selectors),
            "treewise {treewise}"
        );
        diagonals.push(
            s.to_host()
                .unwrap()
                .dense_data()
                .unwrap()
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
        );
    }
    assert!(routes > 1);
    assert_eq!(diagonals[0], diagonals[1]);
}

/// Each nonempty coupled sector of a host tensor as `(sector, rows, cols,
/// column-major widened values)`.
#[cfg(feature = "cuda")]
fn gauge_sector_matrices<R, D>(
    tensor: &TensorMap<R, D>,
) -> Vec<(SectorId, usize, usize, Vec<Complex64>)>
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64> + CheckedFusionAlgebra + SectorCodec,
    D: CudaFactorizationPayload,
{
    let tensor = tensor.materialize().unwrap();
    let data = tensor.dense_data().unwrap();
    let space = tensor.logical_space().space();
    sector_regions(space.structure(), space.nout())
        .unwrap()
        .iter()
        .filter(|region| region.rows() != 0 && region.cols() != 0)
        .map(|region| {
            let values = data[region.range()]
                .iter()
                .map(|value| value.widen_complex())
                .collect();
            (region.coupled(), region.rows(), region.cols(), values)
        })
        .collect()
}

/// `A = U diag(g) Vh` from the Host SVD of `host`, with every sector's
/// spectrum replaced by `g(i) = 2` for `i < 2` and `1 / (2 + i)` after:
/// an exactly degenerate top pair in every sector of dimension >= 2.
#[cfg(feature = "cuda")]
fn with_degenerate_spectrum<R, D>(host: &TensorMap<R, D>) -> TensorMap<R, D>
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64> + CheckedFusionAlgebra + SectorCodec,
    D: CudaFactorizationPayload,
{
    let Svd { u, s, vh } = host
        .svd_compact(&codomain_axes(host), &domain_axes(host))
        .unwrap();
    let bond = s.domain();
    let spectrum = TensorMap::<R, D>::from_subblock_fn(host.runtime(), &bond, &bond, |_, index| {
        if index[0] != index[1] {
            D::zero()
        } else if index[0] < 2 {
            D::from_real(2.0)
        } else {
            D::from_real(1.0 / (2 + index[0]) as f64)
        }
    })
    .unwrap();
    u.compose(&spectrum).unwrap().compose(&vh).unwrap()
}

/// Consistency contract of #1552, Host SVD as the oracle: (1) singular
/// values agree; (2) a vector of a non-degenerate singular value with a
/// clear pivot agrees entrywise (same sign/phase gauge); (3) a degenerate
/// group, or a vector whose pivot is ambiguous at solver rounding, agrees
/// by its projector. Also: the pivot of every device `u` column is real
/// and non-negative. Returns how many vectors were compared entrywise.
#[cfg(feature = "cuda")]
fn assert_device_svd_gauge_matches_host<R, D>(host: &TensorMap<R, D>) -> usize
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64> + CheckedFusionAlgebra + SectorCodec,
    D: CudaFactorizationPayload,
{
    let tolerance = 1.0e3 * D::epsilon();
    let expected = host
        .svd_compact(&codomain_axes(host), &domain_axes(host))
        .unwrap();
    let actual = host
        .to_cuda()
        .unwrap()
        .svd_compact(&codomain_axes(host), &domain_axes(host))
        .unwrap();
    let actual = Svd {
        u: actual.u.to_host().unwrap(),
        s: actual.s.to_host().unwrap(),
        vh: actual.vh.to_host().unwrap(),
    };
    let matrices = |svd: &Svd<TensorMap<R, D>>| {
        (
            gauge_sector_matrices(&svd.u),
            gauge_sector_matrices(&svd.s),
            gauge_sector_matrices(&svd.vh),
        )
    };
    let (host_u, host_s, host_vh) = matrices(&expected);
    let (device_u, device_s, device_vh) = matrices(&actual);
    assert_eq!(host_u.len(), device_u.len());
    let find = |list: &[(SectorId, usize, usize, Vec<Complex64>)], sector: SectorId| {
        list.iter()
            .find(|entry| entry.0 == sector)
            .cloned()
            .expect("every sector has all three factors")
    };
    let mut entrywise = 0;
    for (sector, rows, k, u_host) in &host_u {
        let (rows, k) = (*rows, *k);
        let (_, _, _, u_device) = find(&device_u, *sector);
        let (_, _, cols, vh_host) = find(&host_vh, *sector);
        let (_, _, _, vh_device) = find(&device_vh, *sector);
        let s_host = find(&host_s, *sector).3;
        let s_device = find(&device_s, *sector).3;
        let singular = |s: &[Complex64], i: usize| s[i * (k + 1)].re;
        // (1) singular values.
        for i in 0..k {
            let (d, h) = (singular(&s_device, i), singular(&s_host, i));
            assert!(
                (d - h).abs() <= tolerance * h.abs().max(1.0),
                "s: {d} vs {h}"
            );
        }
        let u_col = |u: &[Complex64], j: usize| u[j * rows..(j + 1) * rows].to_vec();
        let vh_row =
            |vh: &[Complex64], j: usize| (0..cols).map(|c| vh[j + k * c]).collect::<Vec<_>>();
        // Device pivot: the first largest-|u| entry is real and >= 0.
        for j in 0..k {
            let column = u_col(&u_device, j);
            let top = column.iter().map(|x| x.norm()).fold(0.0, f64::max);
            assert!(
                column
                    .iter()
                    .any(|x| x.norm() >= top - tolerance && x.im.abs() <= tolerance && x.re >= 0.0),
                "device u column {j} of {sector:?} has no real non-negative pivot: {column:?}"
            );
        }
        // Group singular values into degenerate runs.
        let mut start = 0;
        while start < k {
            let mut end = start + 1;
            while end < k
                && (singular(&s_host, end - 1) - singular(&s_host, end)).abs()
                    <= 1.0e-6 * singular(&s_host, start).max(1.0)
            {
                end += 1;
            }
            let clear_pivot = |j: usize| {
                let mut magnitudes: Vec<f64> = u_col(u_host, j).iter().map(|x| x.norm()).collect();
                magnitudes.sort_by(|a, b| b.partial_cmp(a).unwrap());
                magnitudes.len() < 2 || magnitudes[0] - magnitudes[1] > 1.0e-6
            };
            if end - start == 1 && clear_pivot(start) {
                // (2) same gauge, entrywise.
                for (d, h) in u_col(&u_device, start).iter().zip(u_col(u_host, start)) {
                    assert!(
                        (d - h).norm() <= tolerance * 10.0,
                        "u {sector:?}[{start}]: {d} vs {h}"
                    );
                }
                for (d, h) in vh_row(&vh_device, start)
                    .iter()
                    .zip(vh_row(&vh_host, start))
                {
                    assert!(
                        (d - h).norm() <= tolerance * 10.0,
                        "vh {sector:?}[{start}]: {d} vs {h}"
                    );
                }
                entrywise += 1;
            } else {
                // (3) projectors of the run.
                let projector = |vectors: Vec<Vec<Complex64>>| {
                    let n = vectors[0].len();
                    let mut p = vec![Complex64::new(0.0, 0.0); n * n];
                    for v in &vectors {
                        for a in 0..n {
                            for b in 0..n {
                                p[a + n * b] += v[a] * v[b].conj();
                            }
                        }
                    }
                    p
                };
                let run = start..end;
                for (host_vectors, device_vectors) in [
                    (
                        run.clone().map(|j| u_col(u_host, j)).collect::<Vec<_>>(),
                        run.clone().map(|j| u_col(&u_device, j)).collect::<Vec<_>>(),
                    ),
                    (
                        run.clone().map(|j| vh_row(&vh_host, j)).collect(),
                        run.clone().map(|j| vh_row(&vh_device, j)).collect(),
                    ),
                ] {
                    let (p, q) = (projector(host_vectors), projector(device_vectors));
                    let difference = p
                        .iter()
                        .zip(&q)
                        .map(|(a, b)| (a - b).norm_sqr())
                        .sum::<f64>()
                        .sqrt();
                    assert!(
                        difference <= tolerance * 10.0,
                        "projector {sector:?}[{run:?}]: {difference:e}"
                    );
                }
            }
            start = end;
        }
    }
    entrywise
}

#[cfg(feature = "cuda")]
#[test]
#[ignore = "requires a real CUDA device"]
fn typed_cuda_svd_follows_the_host_gauge_with_and_without_degenerate_spectra() {
    // What: #1552's three-part consistency contract against the Host SVD
    // over U(1) and SU(2), f64 and c64, generic and exactly degenerate
    // spectra, and aligned as well as forced per-tree (non-aligned)
    // assembly, where the gauge rides the selector GEMM instead.
    let runtime = Runtime::builder().cuda(0).dense_threads(1).build().unwrap();
    let u1 = GradedSpace::try_new(
        Arc::new(U1FusionRule),
        [
            (U1Irrep::new(-1), 2),
            (U1Irrep::new(0), 3),
            (U1Irrep::new(1), 2),
        ],
    )
    .unwrap();
    let su2 = GradedSpace::try_new(
        Arc::new(SU2FusionRule),
        [
            (SU2Irrep::from_twice_spin(0), 2),
            (SU2Irrep::from_twice_spin(1), 2),
        ],
    )
    .unwrap();
    fn cases<R>(runtime: &Runtime, leg: &GradedSpace<R>) -> usize
    where
        R: MultiplicityFreeRigidSymbols<Scalar = f64> + CheckedFusionAlgebra + SectorCodec,
    {
        let real = TensorMap::<R, f64>::rand_with_seed(runtime, [leg, leg], [leg], 21).unwrap();
        let complex =
            TensorMap::<R, Complex64>::rand_with_seed(runtime, [leg, leg], [leg], 22).unwrap();
        let mut entrywise = 0;
        for treewise in [false, true] {
            CUDA_SVD_TREEWISE.with(|flag| flag.set(treewise));
            entrywise += assert_device_svd_gauge_matches_host(&real);
            entrywise += assert_device_svd_gauge_matches_host(&complex);
            assert_device_svd_gauge_matches_host(&with_degenerate_spectrum(&real));
            assert_device_svd_gauge_matches_host(&with_degenerate_spectrum(&complex));
            CUDA_SVD_TREEWISE.with(|flag| flag.set(false));
        }
        entrywise
    }
    // The entrywise (sign/phase) statement must not be vacuous.
    assert!(cases(&runtime, &u1) > 8);
    assert!(cases(&runtime, &su2) > 8);
}

#[cfg(feature = "cuda")]
#[test]
#[ignore = "requires a real CUDA device"]
fn typed_cuda_svd_of_a_near_tie_follows_the_first_largest_entry() {
    // What: A = [[1, -1], [-1, 1]] has singular vectors whose entries tie
    // in exact arithmetic. cuSOLVER returns them 1 ulp apart
    // (0.7071067811865475 vs ...476, observed on the A100), so the rule
    // is checked exactly on the device's own output: in every column the
    // first entry of largest magnitude is positive. The ±1 scaling is
    // exact, so these magnitudes are cuSOLVER's. No column or `vh` row may
    // be zeroed (a tie-break that summed tied entries would do that).
    // The exact-tie contract (first row wins) is pinned on hand-built
    // factors in `tenet-dense/tests/cuda_svd_gauge.rs`.
    let runtime = Runtime::builder().cuda(0).dense_threads(1).build().unwrap();
    let leg = GradedSpace::try_new(Arc::new(U1FusionRule), [(U1Irrep::new(0), 2)]).unwrap();
    let host =
        TensorMap::<U1FusionRule, f64>::from_subblock_fn(&runtime, [&leg], [&leg], |_, index| {
            if index[0] == index[1] {
                1.0
            } else {
                -1.0
            }
        })
        .unwrap();
    let Svd { u, s, vh } = host
        .to_cuda()
        .unwrap()
        .svd_compact(&codomain_axes(&host), &domain_axes(&host))
        .unwrap();
    let (u, s, vh) = (
        u.to_host().unwrap(),
        s.to_host().unwrap(),
        vh.to_host().unwrap(),
    );
    let half = std::f64::consts::FRAC_1_SQRT_2;
    let u = u.dense_data().unwrap();
    for column in u.chunks(2) {
        let pivot = if column[1].abs() > column[0].abs() {
            column[1]
        } else {
            column[0]
        };
        assert!(pivot > 0.0, "u = {u:?}");
    }
    let s = s.dense_data().unwrap();
    assert!((s[0] - 2.0).abs() <= 1e-12 && s[3].abs() <= 1e-12);
    // Column 0 spans [1, -1], column 1 spans [1, 1], none zeroed.
    assert!(
        (u[0] + u[1]).abs() <= 1e-12 && (u[2] - u[3]).abs() <= 1e-12,
        "u = {u:?}"
    );
    let vh = vh.dense_data().unwrap();
    for value in u.iter().chain(vh) {
        assert!(
            (value.abs() - half).abs() <= 1e-12,
            "u = {u:?}, vh = {vh:?}"
        );
    }
    // vh row 0 takes u column 0's phase: A = 2 u0 vh0 with vh0 = u0.
    assert!(
        (vh[0] - u[0]).abs() <= 1e-12 && (vh[2] - u[1]).abs() <= 1e-12,
        "vh = {vh:?}"
    );
}

#[cfg(feature = "cuda")]
#[test]
#[ignore = "requires a real CUDA device"]
fn typed_cuda_svd_gauge_costs_the_documented_ops_and_no_download() {
    // What: the rustdoc's cost of the gauge. Per nonempty route, 13 ops for
    // the phases, 2 per aligned side (broadcast, mul) or 1 per non-aligned
    // side (the selector's embed_diagonal), and 1 conj for a complex left
    // side. Per call, one gauge-weight upload replaces the old per-route
    // identity selectors; nothing is downloaded.
    let runtime = Runtime::builder().cuda(0).dense_threads(1).build().unwrap();
    let leg = GradedSpace::try_new(
        Arc::new(U1FusionRule),
        [
            (U1Irrep::new(-1), 2),
            (U1Irrep::new(0), 1),
            (U1Irrep::new(1), 2),
        ],
    )
    .unwrap();
    fn check<D: CudaFactorizationPayload>(
        runtime: &Runtime,
        leg: &GradedSpace<U1FusionRule>,
        seed: u64,
    ) {
        let host =
            TensorMap::<U1FusionRule, D>::rand_with_seed(runtime, [leg, leg], [leg], seed).unwrap();
        let device = host.to_cuda().unwrap();
        let regions = sector_regions(
            host.logical_space().space().structure(),
            host.logical_space().space().nout(),
        )
        .unwrap();
        let plan = device.compile_cuda_qr_plan(Arc::clone(&regions)).unwrap();
        let max_rows = plan
            .routes
            .iter()
            .map(|route| regions[route.source].rows())
            .max()
            .unwrap();
        for treewise in [false, true] {
            let expected_ops: u64 = plan
                .routes
                .iter()
                .map(|route| {
                    let side = |aligned: bool| if aligned && !treewise { 2 } else { 1 };
                    13 + side(route.aligned_left)
                        + side(route.aligned_right)
                        + u64::from(<D as tenet_dense::CudaScalar>::IS_COMPLEX)
                })
                .sum();
            CUDA_SVD_TREEWISE.with(|flag| flag.set(treewise));
            CUDA_QR_OBSERVATION.with(|observation| observation.set(Some((0, 0, 0, 0, 0, 0, 0))));
            let before = tenet_dense::cuda_transfer_stats();
            let Svd { u, s, vh } = device
                .svd_compact(&codomain_axes(&device), &domain_axes(&device))
                .unwrap();
            let after = tenet_dense::cuda_transfer_stats();
            CUDA_SVD_TREEWISE.with(|flag| flag.set(false));
            let (_, _, selector_uploads, _, _, _, _) = CUDA_QR_OBSERVATION
                .with(|observation| observation.replace(None))
                .unwrap();
            assert_eq!(selector_uploads, 0, "treewise {treewise}");
            assert_eq!(
                after.gauge_ops - before.gauge_ops,
                expected_ops,
                "treewise {treewise}"
            );
            assert_eq!(after.d2h_calls - before.d2h_calls, 0, "treewise {treewise}");
            assert_eq!(after.h2d_calls - before.h2d_calls, 4, "treewise {treewise}");
            let len = |t: &TensorMap<U1FusionRule, D, CudaStorage<D>>| {
                t.logical_space().space().required_len().unwrap()
            };
            let payload = |len: usize| (len * std::mem::size_of::<D>()) as u64;
            assert_eq!(
                after.h2d_bytes - before.h2d_bytes,
                payload(len(&u) + len(&s) + len(&vh))
                    + (max_rows * std::mem::size_of::<i64>()) as u64,
                "treewise {treewise}"
            );
        }
    }
    check::<f64>(&runtime, &leg, 31);
    check::<Complex64>(&runtime, &leg, 32);
}

/// `s` exactly as `svd_compact` built it before #1536: each nonempty
/// sector's cuSOLVER spectrum, downloaded and placed on a host zero
/// buffer by [`fill_diagonal_values`]. Returned as widened bits.
#[cfg(feature = "cuda")]
fn downloaded_svd_diagonal_bits<R, D>(
    device: &TensorMap<R, D, CudaStorage<D>>,
    s_len: usize,
    s_structure: &BlockStructure,
) -> Vec<(u64, u64)>
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64> + CheckedFusionAlgebra + SectorCodec,
    D: CudaFactorizationPayload,
{
    let source = device.direct_cuda_storage("test").unwrap();
    let space = device.logical_space().space();
    let regions = sector_regions(space.structure(), space.nout()).unwrap();
    let mut lease = device.runtime.lease_cuda().unwrap();
    let cuda = &mut *lease;
    let mut sectors = Vec::new();
    let mut spectra = Vec::new();
    for region in regions.iter() {
        if region.rows() == 0 || region.cols() == 0 {
            continue;
        }
        let (_, spectrum, _) = cuda_svd_region::<D>(
            cuda,
            &source.0,
            region.range().start,
            region.rows(),
            region.cols(),
        )
        .unwrap();
        sectors.push(region.coupled());
        spectra.push(spectrum);
    }
    let spectra: Vec<_> = sectors
        .into_iter()
        .zip(cuda_download_spectra::<D>(cuda, &spectra).unwrap())
        .map(|(sector, values)| tenet_matrixalgebra::SectorSpectrum { sector, values })
        .collect();
    let mut host = vec![<D as tenet_dense::CudaScalar>::ZERO; s_len];
    fill_diagonal_values(s_structure, &mut host, &spectra).unwrap();
    host.into_iter().map(widened_bits).collect()
}

#[cfg(feature = "cuda")]
fn widened_bits<D: FactorScalar>(value: D) -> (u64, u64) {
    let value = value.widen_complex();
    (value.re.to_bits(), value.im.to_bits())
}

#[cfg(feature = "cuda")]
fn assert_device_svd_diagonal_matches_the_downloaded_one<R, D>(host: &TensorMap<R, D>)
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64> + CheckedFusionAlgebra + SectorCodec,
    D: CudaFactorizationPayload,
{
    let regions = sector_regions(
        host.logical_space().space().structure(),
        host.logical_space().space().nout(),
    )
    .unwrap();
    let nonempty = regions
        .iter()
        .filter(|region| region.rows() != 0 && region.cols() != 0)
        .count();
    assert!(nonempty > 1, "the fixture needs several blocks");
    let device = host.to_cuda().unwrap();
    let Svd { s, .. } = device
        .svd_compact(&codomain_axes(&device), &domain_axes(&device))
        .unwrap();
    let s = s.to_host().unwrap();
    let expected = downloaded_svd_diagonal_bits(
        &device,
        s.materialize().unwrap().dense_data().unwrap().len(),
        s.logical_space().space().structure(),
    );
    let actual: Vec<_> = s
        .materialize()
        .unwrap()
        .dense_data()
        .unwrap()
        .iter()
        .copied()
        .map(widened_bits)
        .collect();
    assert_eq!(actual, expected);
    // The spectra themselves agree with the Host SVD to dtype tolerance.
    let Svd { s: host_s, .. } = host
        .svd_compact(&codomain_axes(host), &domain_axes(host))
        .unwrap();
    let tolerance = 1.0e3 * D::epsilon();
    for (device, host) in s
        .materialize()
        .unwrap()
        .dense_data()
        .unwrap()
        .iter()
        .zip(host_s.materialize().unwrap().dense_data().unwrap())
    {
        let (device, host) = (device.widen_complex(), host.widen_complex());
        assert!(
            (device - host).norm() <= tolerance * host.norm().max(1.0),
            "{device} vs {host}"
        );
    }
}

#[cfg(feature = "cuda")]
/// Every fixture below has a coupled sector on one side only: it has no
/// block, no route and no diagonal region, and must not shift the others.
fn assert_device_svd_diagonal_every_dtype<R>(
    runtime: &Runtime,
    codomain: &[&GradedSpace<R>],
    domain: &[&GradedSpace<R>],
) where
    R: MultiplicityFreeRigidSymbols<Scalar = f64> + CheckedFusionAlgebra + SectorCodec,
{
    let fused = |legs: &[&GradedSpace<R>]| {
        legs[1..]
            .iter()
            .fold(legs[0].clone(), |acc, leg| acc.fuse(leg).unwrap())
            .sectors()
            .unwrap()
    };
    let (coupled_codomain, coupled_domain) = (fused(codomain), fused(domain));
    assert!(
        coupled_codomain
            .iter()
            .any(|sector| !coupled_domain.contains(sector))
            || coupled_domain
                .iter()
                .any(|sector| !coupled_codomain.contains(sector)),
        "the fixture needs a coupled sector on one side only"
    );
    let host = |seed| {
        TensorMap::<R, f64>::rand_with_seed(
            runtime,
            codomain.iter().copied(),
            domain.iter().copied(),
            seed,
        )
        .unwrap()
    };
    assert_device_svd_diagonal_matches_the_downloaded_one(&host(3));
    assert_device_svd_diagonal_matches_the_downloaded_one(
        &TensorMap::<R, Complex64>::rand_with_seed(
            runtime,
            codomain.iter().copied(),
            domain.iter().copied(),
            5,
        )
        .unwrap(),
    );
    assert_device_svd_diagonal_matches_the_downloaded_one(
        &TensorMap::<R, f32>::rand_with_seed(
            runtime,
            codomain.iter().copied(),
            domain.iter().copied(),
            7,
        )
        .unwrap(),
    );
    assert_device_svd_diagonal_matches_the_downloaded_one(
        &TensorMap::<R, num_complex::Complex32>::rand_with_seed(
            runtime,
            codomain.iter().copied(),
            domain.iter().copied(),
            11,
        )
        .unwrap(),
    );
}

#[cfg(feature = "cuda")]
#[test]
#[ignore = "requires a real CUDA device"]
fn typed_cuda_svd_diagonal_written_on_device_equals_the_downloaded_diagonal_bitwise() {
    // What: `s` is now written by a device strided copy of each sector's
    // spectrum; it must equal, bit for bit, the host-filled `s` built from
    // the same solver's downloaded spectra, across symmetries, dtypes,
    // multi-tree (non-aligned) sectors and an empty coupled sector.
    let runtime = Runtime::builder().cuda(0).dense_threads(1).build().unwrap();

    let u1 = Arc::new(U1FusionRule);
    let u1_leg = |charges: &[(i32, usize)]| {
        GradedSpace::try_new(
            Arc::clone(&u1),
            charges
                .iter()
                .map(|&(charge, dim)| (U1Irrep::new(charge), dim)),
        )
        .unwrap()
    };
    let small = u1_leg(&[(-1, 2), (0, 1), (1, 2)]);
    // Coupled sector 3 exists in the domain only: an empty route.
    let wide = u1_leg(&[(-1, 3), (0, 4), (1, 2), (3, 2)]);
    assert_device_svd_diagonal_every_dtype(&runtime, &[&small, &small], &[&wide]);

    let su2 = Arc::new(SU2FusionRule);
    let su2_leg = GradedSpace::try_new(
        Arc::clone(&su2),
        [
            (SU2Irrep::from_twice_spin(0), 2),
            (SU2Irrep::from_twice_spin(1), 2),
        ],
    )
    .unwrap();
    // Spin 3/2 occurs in the codomain only.
    assert_device_svd_diagonal_every_dtype(&runtime, &[&su2_leg, &su2_leg, &su2_leg], &[&su2_leg]);

    let fermion = Arc::new(U1FusionRule.product(FermionParityFusionRule));
    let fermion_leg = GradedSpace::try_new(
        Arc::clone(&fermion),
        [
            (product_sector(U1Irrep::new(0), Z2Irrep::EVEN), 2),
            (product_sector(U1Irrep::new(1), Z2Irrep::ODD), 2),
            (product_sector(U1Irrep::new(-1), Z2Irrep::ODD), 1),
        ],
    )
    .unwrap();
    assert_device_svd_diagonal_every_dtype(
        &runtime,
        &[&fermion_leg, &fermion_leg],
        &[&fermion_leg],
    );
}

#[cfg(feature = "cuda")]
#[test]
fn missing_cuda_context_precedes_compact_expansion_and_lazy_materialization() {
    DIAGONAL_MATERIALIZATIONS.set(0);
    let source = u1_lazy_fixture();
    let diagonal = source
        .svd_compact(&codomain_axes(&source), &domain_axes(&source))
        .unwrap()
        .s;
    let lazy = source.adjoint().unwrap();
    let TypedData::Diagonal(spectrum) = owned(&diagonal).data.as_ref() else {
        unreachable!("SVD factor is compact")
    };
    let mut malformed_spectrum = spectrum.clone();
    for entry in &mut malformed_spectrum {
        entry.values.clear();
    }
    let malformed_expansion = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        tenet_matrixalgebra::diagonal_bond_data(
            diagonal.logical_space().space(),
            &malformed_spectrum,
            &|value| value,
        )
    }));
    assert!(
        malformed_expansion.is_err() || matches!(malformed_expansion, Ok(Err(_))),
        "the fixture must fail if compact expansion runs"
    );
    let malformed = TensorMap {
        runtime: diagonal.runtime.clone(),
        repr: owned_repr(TypedTensorBody::diagonal(
            diagonal.logical_space().clone(),
            malformed_spectrum,
        )),
    };
    let missing_context = Error::InvalidArgument(
        "this runtime was built without a CUDA device; use Runtime::builder().cuda(device)"
            .to_string(),
    );

    assert_eq!(malformed.to_cuda().unwrap_err(), missing_context);
    assert!(matches!(lazy.to_cuda(), Err(error) if error == missing_context));
    assert_eq!(DIAGONAL_MATERIALIZATIONS.get(), 0);
}

#[cfg(feature = "cuda")]
#[test]
#[ignore]
fn typed_cuda_compact_and_lazy_roundtrips_keep_source_caches_cold() {
    DIAGONAL_MATERIALIZATIONS.set(0);
    let runtime = Runtime::builder().cuda(0).dense_threads(1).build().unwrap();
    let provider = Arc::new(U1FusionRule);
    let leg = GradedSpace::try_new(
        Arc::clone(&provider),
        [(U1Irrep::new(0), 2), (U1Irrep::new(1), 1)],
    )
    .unwrap();
    let source: TensorMap<_, f64> =
        TensorMap::from_subblock_fn(&runtime, [&leg], [&leg], |_, indices| {
            indices.iter().sum::<usize>() as f64 + 1.0
        })
        .unwrap();

    let diagonal = source
        .svd_compact(&codomain_axes(&source), &domain_axes(&source))
        .unwrap()
        .s;
    let TypedData::Diagonal(spectrum) = owned(&diagonal).data.as_ref() else {
        unreachable!("SVD factor is compact")
    };
    let expected_diagonal = tenet_matrixalgebra::diagonal_bond_data(
        diagonal.logical_space().space(),
        spectrum,
        &|value| value,
    )
    .unwrap();
    let diagonal_device = diagonal.to_cuda().unwrap();
    assert_eq!(diagonal_device.placement(), Placement::Cuda(0));
    assert_eq!(DIAGONAL_MATERIALIZATIONS.get(), 0);
    let diagonal_host = diagonal_device.to_host().unwrap();
    assert!(matches!(
        owned(&diagonal_host).data.as_ref(),
        TypedData::Dense(_)
    ));
    assert_eq!(diagonal_host.dense_data().unwrap(), expected_diagonal);
    assert_eq!(DIAGONAL_MATERIALIZATIONS.get(), 0);

    let lazy = source.adjoint().unwrap();
    let expected_lazy = tenet_tensors::materialize_adjoint_data_dyn(
        source.logical_space().space(),
        lazy.logical_space().space(),
        source.dense_data().unwrap(),
    )
    .unwrap();
    let lazy_device = lazy.to_cuda().unwrap();
    let TypedTensorRepr::Adjoint(device_view) = &lazy_device.repr else {
        unreachable!("transfer preserves the lazy view")
    };
    let expected_norm = source.norm(2.0).unwrap();
    CUDA_REDUCTION_BUFFER_OBSERVATION.with(|observation| observation.set(Some((0, 0, 0))));
    assert!(
        (lazy_device.norm(2.0).unwrap() - expected_norm).abs() <= 1e-12 * (1.0 + expected_norm)
    );
    let observed = CUDA_REDUCTION_BUFFER_OBSERVATION.with(|observation| {
        let observed = observation.get().unwrap();
        observation.set(None);
        observed
    });
    let sector_count = sector_regions(
        source.logical_space().space().structure(),
        source.logical_space().space().nout(),
    )
    .unwrap()
    .len();
    assert_eq!(observed, (1, sector_count.max(1), sector_count.max(1)));
    assert!(source.dense_data().unwrap().len() > sector_count.max(1));

    macro_rules! observed_arithmetic {
        ($expression:expr, $arithmetic:expr, $reduction:expr) => {{
            CUDA_ARITHMETIC_OBSERVATION.with(|observation| observation.set(Some((0, 0, 0))));
            CUDA_REDUCTION_BUFFER_OBSERVATION.with(|observation| observation.set(Some((0, 0, 0))));
            let result = $expression;
            CUDA_ARITHMETIC_OBSERVATION.with(|observation| {
                assert_eq!(observation.get(), Some($arithmetic));
                observation.set(None);
            });
            CUDA_REDUCTION_BUFFER_OBSERVATION.with(|observation| {
                assert_eq!(observation.get(), Some($reduction));
                observation.set(None);
            });
            result
        }};
    }

    let source_device = source.to_cuda().unwrap();
    let empty_storage = {
        let lease = runtime.lease_cuda().unwrap();
        CudaStorage::<f64>::upload(&lease, &[]).unwrap()
    };
    let malformed_length = TensorMap {
        runtime: runtime.clone(),
        repr: owned_repr(TypedTensorBody::dense(
            source.logical_space().clone(),
            empty_storage,
        )),
    };
    let work_sentinel = (usize::MAX, usize::MAX, usize::MAX);
    CUDA_ARITHMETIC_OBSERVATION.with(|observation| observation.set(Some(work_sentinel)));
    assert!(matches!(
        malformed_length.scale(2.0),
        Err(Error::InvalidArgument(message)) if message.contains("payload length")
    ));
    CUDA_ARITHMETIC_OBSERVATION.with(|observation| {
        assert_eq!(observation.get(), Some(work_sentinel));
        observation.set(None);
    });

    observed_arithmetic!(source_device.scale(-2.0), (1, 1, 1), (0, 0, 0)).unwrap();
    observed_arithmetic!(
        source_device.axpby(2.0, &source_device, -3.0),
        (1, 1, 2),
        (0, 0, 0)
    )
    .unwrap();
    observed_arithmetic!(source_device.zeros_like(), (1, 0, 0), (0, 0, 0)).unwrap();

    let lazy_scale = observed_arithmetic!(lazy_device.scale(-2.0), (1, 1, 1), (0, 0, 0)).unwrap();
    let lazy_add = observed_arithmetic!(
        lazy_device.axpby(2.0, &lazy_device, -3.0),
        (1, 1, 2),
        (0, 0, 0)
    )
    .unwrap();
    let lazy_zero = observed_arithmetic!(lazy_device.zeros_like(), (1, 0, 0), (0, 0, 0)).unwrap();
    for result in [&lazy_scale, &lazy_add, &lazy_zero] {
        assert!(matches!(result.repr, TypedTensorRepr::Adjoint(_)));
    }
    assert!(matches!(
        observed_arithmetic!(
            lazy_device.axpby(2.0, &source_device, -3.0),
            (0, 0, 0),
            (0, 0, 0)
        ),
        Err(Error::UnsupportedOnDevice(_))
    ));

    assert!(matches!(
        lazy_device.inner(&lazy_device),
        Err(Error::UnsupportedOnDevice(_))
    ));
    assert!(matches!(
        lazy_device.inner(&lazy_device),
        Err(Error::UnsupportedOnDevice(_))
    ));

    let mut missing_context = lazy_device.clone();
    missing_context.runtime = Runtime::builder().build().unwrap();
    let preflight_sentinel = (usize::MAX, usize::MAX, usize::MAX);
    CUDA_REDUCTION_BUFFER_OBSERVATION.with(|observation| observation.set(Some(preflight_sentinel)));
    assert!(matches!(
        missing_context.norm(2.0),
        Err(Error::InvalidArgument(message)) if message.contains("without a CUDA device")
    ));
    CUDA_REDUCTION_BUFFER_OBSERVATION.with(|observation| {
        assert_eq!(observation.get(), Some(preflight_sentinel));
        observation.set(None);
    });
    CUDA_ARITHMETIC_OBSERVATION.with(|observation| observation.set(Some(preflight_sentinel)));
    assert!(matches!(
        missing_context.zeros_like(),
        Err(Error::InvalidArgument(message)) if message.contains("without a CUDA device")
    ));
    CUDA_ARITHMETIC_OBSERVATION.with(|observation| {
        assert_eq!(observation.get(), Some(preflight_sentinel));
        observation.set(None);
    });
    let device_clone = lazy_device.clone();
    let TypedTensorRepr::Adjoint(clone_view) = &device_clone.repr else {
        unreachable!("clone preserves the lazy view")
    };
    assert!(Arc::ptr_eq(device_view, clone_view));

    let lazy_host = device_clone.to_host().unwrap();
    let TypedTensorRepr::Adjoint(_) = &lazy_host.repr else {
        unreachable!("roundtrip preserves the lazy view")
    };
    assert_eq!(
        lazy_host.materialize().unwrap().dense_data().unwrap(),
        expected_lazy
    );
}

/// #1268: a `Complex64` device payload must perform exactly the same
/// number of device allocations, coefficient uploads, kernels, and
/// reduction downloads as the `f64` payload on the same structure. Only
/// the bytes per element change (covered in `tenet-dense`).
#[cfg(feature = "cuda")]
#[test]
#[ignore = "requires a real CUDA device"]
fn typed_cuda_complex_payload_costs_the_same_device_calls_as_f64() {
    let runtime = Runtime::builder().cuda(0).dense_threads(1).build().unwrap();
    let provider = Arc::new(U1FusionRule);
    let leg = GradedSpace::try_new(
        Arc::clone(&provider),
        [(U1Irrep::new(0), 2), (U1Irrep::new(1), 1)],
    )
    .unwrap();
    let real: TensorMap<_, f64> =
        TensorMap::from_subblock_fn(&runtime, [&leg], [&leg], |_, indices| {
            indices.iter().sum::<usize>() as f64 + 1.0
        })
        .unwrap();
    let complex: TensorMap<_, Complex64> =
        TensorMap::from_subblock_fn(&runtime, [&leg], [&leg], |_, indices| {
            let ramp = indices.iter().sum::<usize>() as f64;
            Complex64::new(ramp + 1.0, -(ramp + 1.75))
        })
        .unwrap();

    fn observe<T>(run: impl FnOnce() -> T) -> ((usize, usize, usize), (usize, usize, usize)) {
        CUDA_ARITHMETIC_OBSERVATION.with(|observation| observation.set(Some((0, 0, 0))));
        CUDA_REDUCTION_BUFFER_OBSERVATION.with(|observation| observation.set(Some((0, 0, 0))));
        drop(run());
        let arithmetic = CUDA_ARITHMETIC_OBSERVATION.with(|observation| {
            let observed = observation.get().unwrap();
            observation.set(None);
            observed
        });
        let reduction = CUDA_REDUCTION_BUFFER_OBSERVATION.with(|observation| {
            let observed = observation.get().unwrap();
            observation.set(None);
            observed
        });
        (arithmetic, reduction)
    }

    let real_device = real.to_cuda().unwrap();
    let complex_device = complex.to_cuda().unwrap();
    assert_eq!(
        observe(|| real_device.scale(-2.0).unwrap()),
        observe(|| complex_device.scale(Complex64::new(-2.0, 0.5)).unwrap())
    );
    assert_eq!(
        observe(|| real_device.axpby(2.0, &real_device, -3.0).unwrap()),
        observe(|| complex_device
            .axpby(
                Complex64::new(2.0, 1.0),
                &complex_device,
                Complex64::new(-3.0, 0.25)
            )
            .unwrap())
    );
    assert_eq!(
        observe(|| real_device.zeros_like().unwrap()),
        observe(|| complex_device.zeros_like().unwrap())
    );
    assert_eq!(
        observe(|| real_device.inner(&real_device).unwrap()),
        observe(|| complex_device.inner(&complex_device).unwrap())
    );
    // Contraction allocates its destination and runs its kernels inside
    // the replay seam, which has no arithmetic/reduction hooks: both
    // dtypes must leave those counters untouched rather than falling back
    // to the axpby or reduction paths.
    let untouched = ((0, 0, 0), (0, 0, 0));
    assert_eq!(
        observe(|| real_device
            .contract(
                &real_device,
                &ContractSpec {
                    lhs: &[1],
                    rhs: &[0],
                    codomain: &[0],
                    domain: &[1]
                }
            )
            .unwrap()),
        untouched
    );
    assert_eq!(
        observe(|| complex_device
            .contract(
                &complex_device,
                &ContractSpec {
                    lhs: &[1],
                    rhs: &[0],
                    codomain: &[0],
                    domain: &[1]
                }
            )
            .unwrap()),
        untouched
    );
}

#[cfg(feature = "cuda")]
#[test]
fn typed_cuda_reduction_placement_validation_is_exact() {
    assert!(validate_cuda_reduction_placement(
        Placement::Cuda(0),
        Placement::Cuda(0),
        Placement::Cuda(0)
    )
    .is_ok());
    assert_eq!(
        validate_cuda_reduction_placement(
            Placement::Cuda(1),
            Placement::Cuda(0),
            Placement::Cuda(0)
        )
        .unwrap_err(),
        Error::PlacementMismatch
    );
}

#[cfg(feature = "cuda")]
#[test]
#[ignore = "requires a real CUDA device"]
fn typed_cuda_lazy_adjoint_contract_and_compose_match_rectangular_host_oracles() {
    let runtime = Runtime::builder().cuda(0).dense_threads(1).build().unwrap();
    let provider = Arc::new(U1FusionRule);
    let leg = |degeneracy| {
        GradedSpace::try_new(Arc::clone(&provider), [(U1Irrep::new(0), degeneracy)]).unwrap()
    };
    let (m, k, n) = (2, 3, 4);
    let lhs = TensorMap::from_subblock_fn(&runtime, [&leg(m)], [&leg(k)], |_, indices| {
        (indices[0] + m * indices[1]) as f64 + 1.0
    })
    .unwrap();
    let rhs = TensorMap::from_subblock_fn(&runtime, [&leg(k)], [&leg(n)], |_, indices| {
        (2 * indices[0] + indices[1]) as f64 + 1.0
    })
    .unwrap();
    let expected_contract = lhs
        .contract(
            &rhs,
            &ContractSpec {
                lhs: &[1],
                rhs: &[0],
                codomain: &[0],
                domain: &[1],
            },
        )
        .unwrap();
    let expected_compose = lhs.compose(&rhs).unwrap();

    for upload_parent_first in [false, true] {
        for (lhs_adjoint, rhs_adjoint) in
            [(false, false), (true, false), (false, true), (true, true)]
        {
            let device_operand = |logical: &TensorMap<U1FusionRule, f64>, adjoint: bool| {
                if !adjoint {
                    return logical.to_cuda().unwrap();
                }
                let parent = eager_adjoint_oracle(logical);
                if upload_parent_first {
                    parent.to_cuda().unwrap().adjoint().unwrap()
                } else {
                    parent.adjoint().unwrap().to_cuda().unwrap()
                }
            };
            let lhs_device = device_operand(&lhs, lhs_adjoint);
            let rhs_device = device_operand(&rhs, rhs_adjoint);
            let contracted = lhs_device
                .contract(
                    &rhs_device,
                    &ContractSpec {
                        lhs: &[1],
                        rhs: &[0],
                        codomain: &[0],
                        domain: &[1],
                    },
                )
                .unwrap();
            let composed = lhs_device.compose(&rhs_device).unwrap();
            let contracted = contracted.to_host().unwrap();
            let composed = composed.to_host().unwrap();

            assert_eq!(
                contracted.logical_space().space(),
                expected_contract.logical_space().space()
            );
            assert_eq!(
                composed.logical_space().space(),
                expected_compose.logical_space().space()
            );
            assert_eq!(
                contracted.dense_data().unwrap(),
                expected_contract.dense_data().unwrap()
            );
            assert_eq!(
                composed.dense_data().unwrap(),
                expected_compose.dense_data().unwrap()
            );
            assert!(Arc::ptr_eq(
                contracted.logical_space().provider_arc(),
                lhs.logical_space().provider_arc()
            ));
        }
    }
}

#[cfg(feature = "cuda")]
#[test]
#[ignore = "requires a real CUDA device"]
fn typed_cuda_lazy_adjoint_preserves_fermionic_contract_sign() {
    let runtime = Runtime::builder().cuda(0).dense_threads(1).build().unwrap();
    let provider = Arc::new(FermionParityFusionRule);
    let odd = |is_dual| {
        GradedSpace::try_new(Arc::clone(&provider), [(Z2Irrep::ODD, 1)])
            .and_then(|space| if is_dual { space.try_dual() } else { Ok(space) })
            .unwrap()
    };
    let lhs =
        TensorMap::from_subblock_fn(&runtime, [&odd(false)], [&odd(true)], |_, _| 2.0).unwrap();
    let rhs =
        TensorMap::from_subblock_fn(&runtime, [&odd(true)], [&odd(false)], |_, _| 3.0).unwrap();

    for (lhs_adjoint, rhs_adjoint) in [(false, false), (true, false), (false, true), (true, true)] {
        let device_operand = |logical: &TensorMap<FermionParityFusionRule, f64>, adjoint: bool| {
            if adjoint {
                eager_adjoint_oracle(logical)
                    .to_cuda()
                    .unwrap()
                    .adjoint()
                    .unwrap()
            } else {
                logical.to_cuda().unwrap()
            }
        };
        let lhs_device = device_operand(&lhs, lhs_adjoint);
        let rhs_device = device_operand(&rhs, rhs_adjoint);
        let contract = lhs_device
            .contract(
                &rhs_device,
                &ContractSpec {
                    lhs: &[1],
                    rhs: &[0],
                    codomain: &[0],
                    domain: &[1],
                },
            )
            .unwrap()
            .to_host()
            .unwrap();
        let compose = lhs_device.compose(&rhs_device).unwrap().to_host().unwrap();

        assert_eq!(contract.dense_data().unwrap(), &[-6.0]);
        assert_eq!(compose.dense_data().unwrap(), &[6.0]);
    }
}

#[cfg(feature = "cuda")]
#[test]
#[ignore = "requires a real CUDA device"]
fn typed_cuda_lazy_adjoint_covers_su2_rank_five_and_simple_product() {
    let runtime = Runtime::builder().cuda(0).dense_threads(1).build().unwrap();

    let su2_provider = Arc::new(SU2FusionRule);
    let su2 = GradedSpace::try_new(
        Arc::clone(&su2_provider),
        [
            (SU2Irrep::from_twice_spin(0), 1),
            (SU2Irrep::from_twice_spin(1), 2),
        ],
    )
    .unwrap();
    let su2_lhs =
        TensorMap::from_subblock_fn(&runtime, [&su2, &su2, &su2], [&su2, &su2], |_, indices| {
            indices.iter().sum::<usize>() as f64 + 1.0
        })
        .unwrap();
    let su2_rhs =
        TensorMap::from_subblock_fn(&runtime, [&su2, &su2], [&su2, &su2, &su2], |_, indices| {
            indices.iter().sum::<usize>() as f64 + 3.0
        })
        .unwrap();
    assert_cuda_lazy_contract_orientations(&su2_lhs, &su2_rhs);

    let product_provider = Arc::new(U1FusionRule.product(FermionParityFusionRule));
    let product = GradedSpace::try_new(
        Arc::clone(&product_provider),
        [
            (product_sector(U1Irrep::new(0), Z2Irrep::EVEN), 2),
            (product_sector(U1Irrep::new(1), Z2Irrep::ODD), 1),
        ],
    )
    .unwrap();
    let product_lhs =
        TensorMap::from_subblock_fn(&runtime, [&product], [&product], |_, indices| {
            indices.iter().sum::<usize>() as f64 + 1.0
        })
        .unwrap();
    let product_rhs =
        TensorMap::from_subblock_fn(&runtime, [&product], [&product], |_, indices| {
            indices.iter().sum::<usize>() as f64 + 4.0
        })
        .unwrap();
    assert_cuda_lazy_contract_orientations(&product_lhs, &product_rhs);
}

#[test]
fn generic_lazy_adjoint_keeps_parent_storage_and_refuses_a_dense_borrow() {
    let source = u1_lazy_fixture();
    let parent: Arc<TypedTensorBody<_, _, NonCloneHost>> = Arc::new(TypedTensorBody::dense(
        source.logical_space().clone(),
        NonCloneHost(source.dense_data().unwrap().to_vec()),
    ));
    let logical_space = tenet_tensors::adjoint_bound_space_dyn(&parent.space).unwrap();
    let lazy = TensorMap {
        runtime: source.runtime.clone(),
        repr: TypedTensorRepr::Adjoint(Arc::new(TypedAdjointView::new(
            Arc::clone(&parent),
            logical_space,
        ))),
    };

    assert!(matches!(
        lazy.dense_data(),
        Err(Error::Unsupported {
            alternative: crate::error::Alternative::Materialize,
            ..
        })
    ));
    let TypedTensorRepr::Adjoint(view) = &lazy.repr else {
        unreachable!("fixture is a lazy adjoint")
    };
    assert!(Arc::ptr_eq(&parent, &view.parent));
}

fn su2_lazy_fixture() -> TensorMap<SU2FusionRule, f64> {
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let provider = Arc::new(SU2FusionRule);
    let leg = GradedSpace::try_new(
        provider,
        [
            (SU2Irrep::from_twice_spin(0), 2),
            (SU2Irrep::from_twice_spin(1), 1),
        ],
    )
    .unwrap();
    TensorMap::from_subblock_fn(&runtime, [&leg, &leg], [&leg], |_, indices| {
        indices.iter().sum::<usize>() as f64 + 1.0
    })
    .unwrap()
}

fn u1_matrix_fixture(
    codomain: impl IntoIterator<Item = (i32, usize)>,
    domain: impl IntoIterator<Item = (i32, usize)>,
) -> TensorMap<U1FusionRule, f64> {
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let provider = Arc::new(U1FusionRule);
    let codomain = GradedSpace::try_new(
        Arc::clone(&provider),
        codomain
            .into_iter()
            .map(|(charge, degeneracy)| (U1Irrep::new(charge), degeneracy)),
    )
    .unwrap();
    let domain = GradedSpace::try_new(
        provider,
        domain
            .into_iter()
            .map(|(charge, degeneracy)| (U1Irrep::new(charge), degeneracy)),
    )
    .unwrap();
    TensorMap::from_subblock_fn(&runtime, [&codomain], [&domain], |_, indices| {
        (indices.iter().sum::<usize>() + 1) as f64
    })
    .unwrap()
}

fn genuinely_complex<R>(source: &TensorMap<R, f64>) -> TensorMap<R, num_complex::Complex64> {
    TensorMap {
        runtime: source.runtime.clone(),
        repr: owned_repr(TypedTensorBody::dense(
            source.logical_space().clone(),
            source
                .dense_data()
                .unwrap()
                .iter()
                .enumerate()
                .map(|(index, &value)| num_complex::Complex64::new(value, (index + 1) as f64 / 7.0))
                .collect(),
        )),
    }
}

fn assert_lazy_involution<R, D>(source: &TensorMap<R, D>)
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64> + CheckedFusionAlgebra + SectorCodec,
    D: TensorScalar,
{
    let adjoint = source.adjoint().unwrap();
    let TypedTensorRepr::Adjoint(view) = &adjoint.repr else {
        panic!("dense adjoint must be lazy");
    };
    assert!(Arc::ptr_eq(&view.parent, owned(source)));
    assert!(Arc::ptr_eq(
        view.parent.space.provider_arc(),
        view.logical_space.provider_arc()
    ));
    assert!(std::ptr::eq(adjoint.provider(), source.provider()));

    let clone = adjoint.clone();
    let TypedTensorRepr::Adjoint(clone_view) = &clone.repr else {
        unreachable!()
    };
    assert!(Arc::ptr_eq(view, clone_view));

    let restored = adjoint.adjoint().unwrap();
    assert!(Arc::ptr_eq(owned(source), owned(&restored)));
    assert_eq!(
        source.dense_data().unwrap().as_ptr(),
        restored.dense_data().unwrap().as_ptr()
    );
}

#[test]
fn lazy_adjoint_representation_and_involution_cover_unique_simple_and_both_dtypes() {
    let u1_f64 = u1_lazy_fixture();
    let u1_c64 = u1_f64.convert::<Complex64>();
    let su2_f64 = su2_lazy_fixture();
    let su2_c64 = su2_f64.convert::<Complex64>();

    assert_lazy_involution(&u1_f64);
    assert_lazy_involution(&u1_c64);
    assert_lazy_involution(&su2_f64);
    assert_lazy_involution(&su2_c64);
}

#[test]
fn lazy_adjoint_metadata_is_logical_and_cold() {
    let source = u1_lazy_fixture();
    let adjoint = source.adjoint().unwrap();
    assert_eq!((source.codomain_rank(), source.domain_rank()), (2, 1));
    assert_eq!((adjoint.codomain_rank(), adjoint.domain_rank()), (1, 2));
    assert_eq!(adjoint.rank(), 3);
    let signature = |space: &GradedSpace<U1FusionRule>| {
        (
            space.sectors().unwrap(),
            space.degeneracies().to_vec(),
            space.is_dual(),
        )
    };
    let source_codomain = source.codomain();
    let source_domain = source.domain();
    let adjoint_codomain = adjoint.codomain();
    let adjoint_domain = adjoint.domain();
    assert_eq!(
        signature(&adjoint_codomain[0]),
        signature(&source_domain[0])
    );
    assert_eq!(
        signature(&adjoint_domain[0]),
        signature(&source_codomain[0])
    );
    assert_eq!(
        signature(&adjoint_domain[1]),
        signature(&source_codomain[1])
    );
    let source_dims = source.leg_dims().unwrap();
    assert_eq!(
        adjoint.leg_dims().unwrap(),
        [source_dims[2], source_dims[0], source_dims[1]]
    );
    assert_eq!(adjoint.subblock_count(), source.subblock_count());
    let expected = tenet_tensors::adjoint_bound_space_dyn(source.logical_space()).unwrap();
    for index in 0..adjoint.subblock_count() {
        let actual = adjoint.subblock(index).unwrap();
        let expected_block = expected.space().structure().block(index).unwrap();
        assert_eq!(actual.key(), expected_block.key());
        assert_eq!(actual.shape(), expected_block.shape());
        assert_eq!(actual.strides(), expected_block.strides());
        assert_eq!(actual.offset(), expected_block.offset());
        assert_eq!(
            adjoint.subblock_fusion_trees(index).unwrap(),
            decode_block_fusion_trees(adjoint.provider(), expected_block.key()).unwrap()
        );
    }
    assert_eq!(adjoint.logical_space().space(), expected.space());
    assert!(!format!("{adjoint:?}").is_empty());
    let TypedTensorRepr::Adjoint(_) = &adjoint.repr else {
        unreachable!()
    };
}

#[cfg(feature = "racah-generated")]
#[test]
fn checked_generic_inv_lazy_is_detached_and_keeps_receiver_cold() {
    use tenet_core::SUNFusionRule;

    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let provider = Arc::new(SUNFusionRule::new(3).unwrap());
    let leg = GradedSpace::try_new(Arc::clone(&provider), [(vec![1, 1], 1)]).unwrap();
    let source: TensorMap<_, f64> =
        TensorMap::from_subblock_fn(&runtime, [&leg], [&leg], |_, _| 2.0).unwrap();
    let lazy = source.adjoint().unwrap();
    let inverse = lazy.inv(&[0], &[1]).unwrap();

    let TypedTensorRepr::Adjoint(_) = &lazy.repr else {
        unreachable!()
    };
    assert!(matches!(inverse.repr, TypedTensorRepr::Owned(_)));
    assert!(!std::ptr::eq(
        inverse.dense_data().unwrap().as_ptr(),
        source.dense_data().unwrap().as_ptr()
    ));
    assert!(inverse
        .dense_data()
        .unwrap()
        .iter()
        .all(|value| (*value - 0.5).abs() < 1.0e-12));
}

#[cfg(feature = "racah-generated")]
#[test]
fn checked_generic_null_lazy_redirects_are_owned_and_keep_receiver_cold() {
    // What: both null directions use the opposite operation on the owned
    // parent, detach the final adjoint, and never publish the lazy cache.
    use tenet_core::SUNFusionRule;

    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let provider = Arc::new(SUNFusionRule::new(3).unwrap());
    let leg = GradedSpace::try_new(Arc::clone(&provider), [(vec![1, 1], 2)]).unwrap();
    let source: TensorMap<_, f64> =
        TensorMap::from_subblock_fn(&runtime, [&leg, &leg], [&leg, &leg], |trees, indices| {
            let row = indices[0] + 2 * indices[1];
            let column = indices[2] + 2 * indices[3];
            f64::from(row == column)
                * trees.codomain_vertices()[0].get() as f64
                * trees.domain_vertices()[0].get() as f64
        })
        .unwrap();
    let body = Arc::clone(owned(&source));
    let payload = Arc::clone(&body.data);
    let lazy = source.adjoint().unwrap();
    let expected_left = source
        .right_null(&[0, 1], &[2, 3])
        .unwrap()
        .adjoint()
        .unwrap()
        .materialized_tensor_uncached()
        .unwrap();
    let expected_right = source
        .left_null(&[0, 1], &[2, 3])
        .unwrap()
        .adjoint()
        .unwrap()
        .materialized_tensor_uncached()
        .unwrap();

    for (actual, expected) in [
        (lazy.left_null(&[0, 1], &[2, 3]).unwrap(), expected_left),
        (lazy.right_null(&[0, 1], &[2, 3]).unwrap(), expected_right),
    ] {
        assert!(matches!(&actual.repr, TypedTensorRepr::Owned(_)));
        assert_eq!(
            actual.logical_space().space(),
            expected.logical_space().space()
        );
        assert!(actual
            .dense_data()
            .unwrap()
            .iter()
            .zip(expected.dense_data().unwrap())
            .all(|(&actual, &expected)| (actual - expected).abs() < 1e-10));
        assert!(std::ptr::eq(actual.provider(), provider.as_ref()));
    }
    assert!(Arc::ptr_eq(owned(&source), &body));
    assert!(Arc::ptr_eq(&owned(&source).data, &payload));
    let TypedTensorRepr::Adjoint(_) = &lazy.repr else {
        unreachable!()
    };
}

#[cfg(feature = "racah-generated")]
#[test]
fn checked_generic_compact_qr_lq_reject_lazy_adjoint_without_materializing() {
    use tenet_core::SUNFusionRule;

    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let provider = Arc::new(SUNFusionRule::new(3).unwrap());
    let leg = GradedSpace::try_new(Arc::clone(&provider), [(vec![1, 1], 1)]).unwrap();
    let source: TensorMap<_, f64> =
        TensorMap::from_subblock_fn(&runtime, [&leg, &leg], [&leg], |trees, _| {
            trees.codomain_vertices()[0].get() as f64
        })
        .unwrap();

    for qr in [true, false] {
        let lazy = source.adjoint().unwrap();
        let result = if qr {
            lazy.qr_compact(&[0], &[1, 2]).map(drop)
        } else {
            lazy.lq_compact(&[0], &[1, 2]).map(drop)
        };
        assert!(
            matches!(
                result,
                Err(GenericTensorError::Facade(Error::InvalidArgument(_)))
            ),
            "qr = {qr}: {result:?}"
        );
    }
}

#[cfg(feature = "racah-generated")]
#[test]
fn checked_generic_exp_lazy_is_owned_and_stays_cold() {
    use tenet_core::SUNFusionRule;

    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let provider = Arc::new(SUNFusionRule::new(3).unwrap());
    let leg = GradedSpace::try_new(Arc::clone(&provider), [(vec![1, 1], 1)]).unwrap();
    let source: TensorMap<_, f64> =
        TensorMap::from_subblock_fn(&runtime, [&leg], [&leg], |_, ij| f64::from(ij == [0, 1]))
            .unwrap();
    let lazy = source.adjoint().unwrap();
    let actual = lazy.exp(&[0], &[1]).unwrap();
    let expected = source.exp(&[0], &[1]).unwrap().adjoint().unwrap();
    assert!(matches!(actual.repr, TypedTensorRepr::Owned(_)));
    assert_eq!(
        actual.dense_data().unwrap(),
        expected.materialize().unwrap().dense_data().unwrap()
    );
    assert!(std::ptr::eq(actual.provider(), provider.as_ref()));
    assert!(actual.runtime().shares_state_with(source.runtime()));
    assert_eq!(actual.codomain(), lazy.codomain());
    assert_eq!(actual.domain(), lazy.domain());
}

#[cfg(feature = "racah-generated")]
#[test]
fn checked_generic_lazy_flip_stays_cold_and_keeps_logical_duality() {
    use tenet_core::SUNFusionRule;

    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let provider = Arc::new(SUNFusionRule::new(3).unwrap());
    let leg = GradedSpace::try_new(Arc::clone(&provider), [(vec![2, 2], 1)]).unwrap();
    let source: TensorMap<_, f64> =
        TensorMap::from_subblock_fn(&runtime, [&leg, &leg], [&leg], |trees, _| {
            trees.codomain_vertices()[0].get() as f64
        })
        .unwrap();
    let lazy = source.adjoint().unwrap();

    let flipped = lazy.flip(&[1], Direction::Forward).unwrap();
    assert_eq!(flipped.domain()[0].is_dual(), !lazy.domain()[0].is_dual());

    let expected = source
        .flip(&[0], Direction::Inverse)
        .unwrap()
        .adjoint()
        .unwrap();
    assert_eq!(
        flipped.materialize().unwrap().dense_data().unwrap(),
        expected.materialize().unwrap().dense_data().unwrap()
    );
    assert_eq!(flipped.codomain(), expected.codomain());
    assert_eq!(flipped.domain(), expected.domain());
    assert!(std::ptr::eq(flipped.provider(), provider.as_ref()));
    assert!(flipped.runtime().shares_state_with(expected.runtime()));
}

#[test]
fn svd_vals_reads_the_parent_without_materializing_the_adjoint() {
    // What: values-only SVD preserves typed sector spectra across cold,
    // repeated, cloned, and concurrent lazy-adjoint reads.
    macro_rules! assert_fixture {
        ($source:expr) => {{
            let source = $source;
            let expected = source.svd_vals(&[0, 1], &[2]).unwrap();
            let lazy = source.adjoint().unwrap();
            assert_eq!(lazy.svd_vals(&[0], &[1, 2]).unwrap(), expected);
            assert_eq!(lazy.svd_vals(&[0], &[1, 2]).unwrap(), expected);
            assert_eq!(lazy.clone().svd_vals(&[0], &[1, 2]).unwrap(), expected);
            let TypedTensorRepr::Adjoint(view) = &lazy.repr else {
                unreachable!()
            };
            assert!(Arc::ptr_eq(
                view.logical_space.provider_arc(),
                source.logical_space().provider_arc()
            ));
        }};
    }
    assert_fixture!(u1_lazy_fixture());
    assert_fixture!(u1_lazy_fixture().convert::<Complex64>());
    assert_fixture!(su2_lazy_fixture());
    assert_fixture!(su2_lazy_fixture().convert::<Complex64>());

    let source = u1_lazy_fixture().convert::<Complex64>();
    let expected = source.svd_vals(&[0, 1], &[2]).unwrap();
    let lazy = source.adjoint().unwrap();
    let threads: Vec<_> = (0..4)
        .map(|_| {
            let lazy = lazy.clone();
            std::thread::spawn(move || lazy.svd_vals(&[0], &[1, 2]).unwrap())
        })
        .collect();
    for thread in threads {
        assert_eq!(thread.join().unwrap(), expected);
    }
    let TypedTensorRepr::Adjoint(_) = &lazy.repr else {
        unreachable!()
    };
}

fn assert_compact_svd_reads_parent<R, D>(source: &TensorMap<R, D>)
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64> + CheckedFusionAlgebra + SectorCodec,
    D: FactorizationScalar + core::fmt::Debug,
{
    let eager = eager_adjoint_oracle(source);
    let lazy = source.adjoint().unwrap();
    let actual = lazy.svd_compact(&[0], &[1, 2]).unwrap();
    let expected = eager.svd_compact(&[0], &[1, 2]).unwrap();

    let TypedTensorRepr::Adjoint(_) = &lazy.repr else {
        unreachable!()
    };
    for (actual, expected) in [
        (&actual.u, &expected.u),
        (&actual.s, &expected.s),
        (&actual.vh, &expected.vh),
    ] {
        assert_eq!(
            actual.logical_space().space(),
            expected.logical_space().space()
        );
        assert!(Arc::ptr_eq(
            actual.logical_space().provider_arc(),
            source.logical_space().provider_arc()
        ));
        assert!(actual
            .materialize()
            .unwrap()
            .dense_data()
            .unwrap()
            .iter()
            .zip(expected.materialize().unwrap().dense_data().unwrap())
            .all(|(&left, &right)| {
                (left.widen_complex() - right.widen_complex()).norm() < 1e-12
            }));
    }
    assert!(is_isometric!(actual.u, 1e-12));
    let rebuilt = actual
        .u
        .compose(&actual.s)
        .unwrap()
        .compose(&actual.vh)
        .unwrap();
    assert!(rebuilt
        .dense_data()
        .unwrap()
        .iter()
        .zip(eager.dense_data().unwrap())
        .all(|(&left, &right)| { (left.widen_complex() - right.widen_complex()).norm() < 1e-12 }));
}

#[test]
fn compact_svd_reads_the_parent_without_materializing_the_adjoint() {
    // What: typed compact factors keep the eager logical-adjoint semantics,
    // provider authority, final gauge, and reconstruction without an input copy.
    let u1 = u1_lazy_fixture();
    let su2 = su2_lazy_fixture();
    assert_compact_svd_reads_parent(&u1);
    assert_compact_svd_reads_parent(&genuinely_complex(&u1));
    assert_compact_svd_reads_parent(&su2);
    assert_compact_svd_reads_parent(&genuinely_complex(&su2));
}

fn assert_full_svd_reads_parent<R, D>(source: &TensorMap<R, D>, compare_factor_bytes: bool)
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64> + CheckedFusionAlgebra + SectorCodec,
    D: FactorizationScalar + core::fmt::Debug,
{
    let eager = eager_adjoint_oracle(source);
    let lazy = source.adjoint().unwrap();
    let actual = lazy
        .svd_full(&codomain_axes(&lazy), &domain_axes(&lazy))
        .unwrap();
    let expected = eager
        .svd_full(&codomain_axes(&eager), &domain_axes(&eager))
        .unwrap();

    let TypedTensorRepr::Adjoint(_) = &lazy.repr else {
        unreachable!()
    };
    for (actual, expected) in [
        (&actual.u, &expected.u),
        (&actual.s, &expected.s),
        (&actual.vh, &expected.vh),
    ] {
        assert_eq!(
            actual.logical_space().space(),
            expected.logical_space().space()
        );
        assert!(Arc::ptr_eq(
            actual.logical_space().provider_arc(),
            source.logical_space().provider_arc()
        ));
    }
    assert!(actual
        .s
        .dense_data()
        .unwrap()
        .iter()
        .zip(expected.s.dense_data().unwrap())
        .all(|(&left, &right)| { (left.widen_complex() - right.widen_complex()).norm() < 1e-12 }));
    if compare_factor_bytes {
        for (actual, expected) in [(&actual.u, &expected.u), (&actual.vh, &expected.vh)] {
            assert!(actual
                .dense_data()
                .unwrap()
                .iter()
                .zip(expected.dense_data().unwrap())
                .all(|(&left, &right)| {
                    (left.widen_complex() - right.widen_complex()).norm() < 1e-12
                }));
        }
    }
    assert!(is_isometric!(actual.u, 1e-12));
    assert!(is_isometric!(actual.vh, 1e-12));
    let rebuilt = actual
        .u
        .compose(&actual.s)
        .unwrap()
        .compose(&actual.vh)
        .unwrap();
    assert!(rebuilt
        .dense_data()
        .unwrap()
        .iter()
        .zip(eager.dense_data().unwrap())
        .all(|(&left, &right)| { (left.widen_complex() - right.widen_complex()).norm() < 1e-12 }));
}

#[test]
fn full_svd_adjoint_rectangular_matched_matches_materialized_oracle() {
    let matched = u1_matrix_fixture([(0, 2)], [(0, 3)]);
    assert_full_svd_reads_parent(&matched, true);
    assert_full_svd_reads_parent(&genuinely_complex(&matched), true);
}

#[test]
fn full_svd_adjoint_unmatched_row_only_matches_materialized_oracle() {
    let source = u1_matrix_fixture([(0, 2), (1, 1)], [(0, 3)]);
    assert_full_svd_reads_parent(&source, false);
}

#[test]
fn full_svd_adjoint_unmatched_column_only_matches_materialized_oracle() {
    let source = u1_matrix_fixture([(0, 2)], [(0, 3), (1, 1)]);
    assert_full_svd_reads_parent(&source, false);
}

#[test]
fn full_svd_adjoint_disjoint_matches_materialized_oracle() {
    let source = u1_matrix_fixture([(1, 2)], [(0, 3)]);
    assert_full_svd_reads_parent(&source, false);
}

#[test]
fn full_svd_adjoint_multitree_matches_materialized_oracle() {
    let multitree = su2_lazy_fixture();
    assert_full_svd_reads_parent(&multitree, false);
    assert_full_svd_reads_parent(&genuinely_complex(&multitree), false);
}

#[test]
fn full_svd_late_failure_does_not_publish_the_adjoint_cache() {
    let runtime = Runtime::builder()
        .with_dense_executor(Box::new(FailSecondSvd::default()))
        .build()
        .unwrap();
    let provider = Arc::new(U1FusionRule);
    let leg = GradedSpace::try_new(provider, [(U1Irrep::new(0), 2), (U1Irrep::new(1), 2)]).unwrap();
    let source = TensorMap::from_subblock_fn(&runtime, [&leg], [&leg], |_, indices| {
        (indices.iter().sum::<usize>() + 1) as f64
    })
    .unwrap();
    let before = source.dense_data().unwrap().to_vec();
    let lazy = source.adjoint().unwrap();

    assert!(matches!(
        lazy.svd_full(&[0], &[1]),
        Err(Error::Operation(_))
    ));
    assert_eq!(source.dense_data().unwrap(), before);
    let TypedTensorRepr::Adjoint(_) = &lazy.repr else {
        unreachable!()
    };
}

fn assert_null_redirect<R, D>(source: &TensorMap<R, D>)
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64> + CheckedFusionAlgebra + SectorCodec,
    D: FactorizationScalar + core::fmt::Debug,
{
    let target = eager_adjoint_oracle(source);
    let lazy = source.adjoint().unwrap();
    for (actual, expected, left) in [
        (
            lazy.left_null(&codomain_axes(&lazy), &domain_axes(&lazy))
                .unwrap(),
            target
                .left_null(&codomain_axes(&target), &domain_axes(&target))
                .unwrap(),
            true,
        ),
        (
            lazy.right_null(&codomain_axes(&lazy), &domain_axes(&lazy))
                .unwrap(),
            target
                .right_null(&codomain_axes(&target), &domain_axes(&target))
                .unwrap(),
            false,
        ),
    ] {
        assert!(actual.owned_body().is_some());
        assert_eq!(
            actual.logical_space().space(),
            expected.logical_space().space()
        );
        assert!(Arc::ptr_eq(
            actual.logical_space().provider_arc(),
            source.logical_space().provider_arc()
        ));
        let actual_projector = if left {
            actual.compose(&actual.adjoint().unwrap()).unwrap()
        } else {
            actual.adjoint().unwrap().compose(&actual).unwrap()
        };
        let expected_projector = if left {
            expected.compose(&expected.adjoint().unwrap()).unwrap()
        } else {
            expected.adjoint().unwrap().compose(&expected).unwrap()
        };
        assert!(actual_projector
            .dense_data()
            .unwrap()
            .iter()
            .zip(expected_projector.dense_data().unwrap())
            .all(|(&actual, &expected)| {
                (actual.widen_complex() - expected.widen_complex()).norm() < 1e-11
            }));
        let residual = if left {
            actual.adjoint().unwrap().compose(&target).unwrap()
        } else {
            target.compose(&actual.adjoint().unwrap()).unwrap()
        };
        assert!(residual.norm(2.0).unwrap() < 1e-10 * (1.0 + target.norm(2.0).unwrap()));
        assert!(if left {
            is_isometric!(actual, 1e-11)
        } else {
            is_isometric!(actual.adjoint().unwrap(), 1e-11)
        });
        let _ = actual.dense_data().unwrap();
    }
    let TypedTensorRepr::Adjoint(_) = &lazy.repr else {
        unreachable!()
    };
}

#[test]
fn null_spaces_redirect_through_the_parent_without_materializing_the_adjoint() {
    let fixtures = [
        u1_matrix_fixture([(0, 3)], [(0, 2)]),
        u1_matrix_fixture([(0, 3), (1, 2)], [(0, 2)]),
        u1_matrix_fixture([(0, 3)], [(0, 2), (1, 2)]),
        u1_matrix_fixture([(1, 2)], [(0, 3)]),
        u1_matrix_fixture([(0, 3)], [(0, 3)]),
    ];
    for source in &fixtures {
        assert_null_redirect(source);
        assert_null_redirect(&genuinely_complex(source));
    }
    let multitree = su2_lazy_fixture();
    assert_null_redirect(&multitree);
    assert_null_redirect(&genuinely_complex(&multitree));
}

fn assert_null_late_failure(left: bool) {
    let runtime = Runtime::builder()
        .with_dense_executor(Box::new(FailSecondSvd::default()))
        .build()
        .unwrap();
    let provider = Arc::new(U1FusionRule);
    let leg = GradedSpace::try_new(provider, [(U1Irrep::new(0), 2), (U1Irrep::new(1), 2)]).unwrap();
    let source = TensorMap::from_subblock_fn(&runtime, [&leg], [&leg], |_, indices| {
        (indices.iter().sum::<usize>() + 1) as f64
    })
    .unwrap();
    let before = source.dense_data().unwrap().to_vec();
    let lazy = source.adjoint().unwrap();

    let result = if left {
        lazy.left_null(&[0], &[1])
    } else {
        lazy.right_null(&[0], &[1])
    };
    assert!(matches!(result, Err(Error::Operation(_))));
    assert_eq!(source.dense_data().unwrap(), before);
    let TypedTensorRepr::Adjoint(_) = &lazy.repr else {
        unreachable!()
    };
}

#[test]
fn null_space_late_failure_leaves_parent_and_adjoint_cache_unchanged() {
    assert_null_late_failure(true);
    assert_null_late_failure(false);
}

fn assert_polar_redirect<R, D>(source: &TensorMap<R, D>, left: bool)
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64> + CheckedFusionAlgebra + SectorCodec,
    D: FactorizationScalar + core::fmt::Debug,
{
    let target = eager_adjoint_oracle(source);
    let lazy = source.adjoint().unwrap();
    let actual = if left {
        lazy.left_polar(&[0], &[1]).unwrap().pair()
    } else {
        lazy.right_polar(&codomain_axes(&lazy), &domain_axes(&lazy))
            .unwrap()
            .pair()
    };
    let expected = if left {
        target.left_polar(&[0], &[1]).unwrap().pair()
    } else {
        target
            .right_polar(&codomain_axes(&target), &domain_axes(&target))
            .unwrap()
            .pair()
    };
    assert_polar_factors(source, &target, &actual, &expected, left);
    let TypedTensorRepr::Adjoint(_) = &lazy.repr else {
        unreachable!()
    };
}

fn assert_typed_map_close<R, D>(
    actual: &TensorMap<R, D>,
    expected: &TensorMap<R, D>,
    tolerance: f64,
) where
    R: MultiplicityFreeRigidSymbols<Scalar = f64> + CheckedFusionAlgebra + SectorCodec,
    D: TensorScalar,
{
    assert_eq!(
        actual.logical_space().space(),
        expected.logical_space().space()
    );
    assert!(actual
        .materialize()
        .unwrap()
        .dense_data()
        .unwrap()
        .iter()
        .zip(expected.materialize().unwrap().dense_data().unwrap())
        .all(|(&actual, &expected)| {
            (actual.widen_complex() - expected.widen_complex()).norm() < tolerance
        }));
}

fn assert_eigh_uses_a_cold_logical_copy(source: &TensorMap<U1FusionRule, f64>) {
    let eager = eager_adjoint_oracle(source);
    let expected_vals = eager.eigh_vals(&[0], &[1]).unwrap();
    let expected_full = eager.eigh_full(&[0], &[1]).unwrap();
    let parent_body = Arc::clone(owned(source));
    let parent_data = Arc::clone(&parent_body.data);
    let lazy = source.adjoint().unwrap();

    for _ in 0..2 {
        assert_eq!(lazy.clone().eigh_vals(&[0], &[1]).unwrap(), expected_vals);
        let full = lazy.clone().eigh_full(&[0], &[1]).unwrap();
        assert_eq!(
            full.d.materialize().unwrap().dense_data().unwrap(),
            expected_full.d.materialize().unwrap().dense_data().unwrap()
        );
        assert_eq!(
            full.v.dense_data().unwrap(),
            expected_full.v.dense_data().unwrap()
        );
        for output in [&full.d, &full.v] {
            assert!(output.owned_body().is_some());
            assert!(Arc::ptr_eq(
                output.logical_space().provider_arc(),
                source.logical_space().provider_arc()
            ));
        }
    }

    let calls = (0..4)
        .map(|_| {
            let clone = lazy.clone();
            std::thread::spawn(move || {
                let vals = clone.eigh_vals(&[0], &[1]).unwrap();
                let full = clone.eigh_full(&[0], &[1]).unwrap();
                (
                    vals,
                    full.d.materialize().unwrap().dense_data().unwrap().to_vec(),
                )
            })
        })
        .collect::<Vec<_>>();
    for call in calls {
        let (vals, diagonal) = call.join().unwrap();
        assert_eq!(vals, expected_vals);
        assert_eq!(
            diagonal,
            expected_full.d.materialize().unwrap().dense_data().unwrap()
        );
    }
    assert!(Arc::ptr_eq(owned(source), &parent_body));
    assert!(Arc::ptr_eq(&owned(source).data, &parent_data));
    let TypedTensorRepr::Adjoint(_) = &lazy.repr else {
        unreachable!()
    };
}

#[test]
fn eigh_dense_lazy_near_hermitian_uses_logical_triangle_and_stays_cold() {
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let provider = Arc::new(U1FusionRule);
    let leg = GradedSpace::try_new(provider, [(U1Irrep::new(0), 2)]).unwrap();
    let source = TensorMap::from_subblock_fn(&runtime, [&leg], [&leg], |_, indices| {
        match (indices[0], indices[1]) {
            (0, 0) | (1, 1) => 1.0,
            (0, 1) => 4.0e-15,
            _ => 0.0,
        }
    })
    .unwrap();
    let logical = eager_adjoint_oracle(&source);
    let logical_vals = logical.eigh_vals(&[0], &[1]).unwrap();
    let parent_vals = source.eigh_vals(&[0], &[1]).unwrap();
    assert!(logical_vals[0]
        .values
        .iter()
        .zip(&parent_vals[0].values)
        .any(|(logical, parent)| (logical - parent).abs() > 1.0e-15));

    assert_eigh_uses_a_cold_logical_copy(&source);
}

#[test]
fn eigh_dense_lazy_complex_orientation_and_failures_match_logical_oracles() {
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let provider = Arc::new(U1FusionRule);
    let leg = GradedSpace::try_new(provider, [(U1Irrep::new(0), 2)]).unwrap();
    let hermitian = TensorMap::from_subblock_fn(&runtime, [&leg], [&leg], |_, indices| {
        match (indices[0], indices[1]) {
            (0, 0) => num_complex::Complex64::new(2.0, 0.0),
            (1, 1) => num_complex::Complex64::new(3.0, 0.0),
            (0, 1) => num_complex::Complex64::new(0.0, 1.0),
            (1, 0) => num_complex::Complex64::new(0.0, -1.0),
            _ => unreachable!(),
        }
    })
    .unwrap();
    let eager = eager_adjoint_oracle(&hermitian);
    let expected = eager.eigh_full(&[0], &[1]).unwrap();
    let lazy = hermitian.adjoint().unwrap();
    let actual = lazy.eigh_full(&[0], &[1]).unwrap();
    assert_eq!(
        actual.d.materialize().unwrap().dense_data().unwrap(),
        expected.d.materialize().unwrap().dense_data().unwrap()
    );
    assert_eq!(
        actual.v.dense_data().unwrap(),
        expected.v.dense_data().unwrap()
    );
    let reconstructed = actual
        .v
        .compose(&actual.d)
        .unwrap()
        .compose(&actual.v.adjoint().unwrap())
        .unwrap();
    assert_typed_map_close(&reconstructed, &eager, 1.0e-12);
    let TypedTensorRepr::Adjoint(_) = &lazy.repr else {
        unreachable!()
    };

    let nonhermitian = TensorMap::from_subblock_fn(&runtime, [&leg], [&leg], |_, indices| {
        match (indices[0], indices[1]) {
            (0, 0) => 1.0,
            (1, 1) => 2.0,
            (0, 1) => 1.0,
            _ => 0.0,
        }
    })
    .unwrap();
    let eager = eager_adjoint_oracle(&nonhermitian);
    let expected = [
        eager.eigh_vals(&[0], &[1]).unwrap_err().to_string(),
        eager.eigh_full(&[0], &[1]).unwrap_err().to_string(),
    ];
    let lazy = nonhermitian.adjoint().unwrap();
    for _ in 0..2 {
        assert_eq!(
            lazy.eigh_vals(&[0], &[1]).unwrap_err().to_string(),
            expected[0]
        );
        assert_eq!(
            lazy.eigh_full(&[0], &[1]).unwrap_err().to_string(),
            expected[1]
        );
    }
    let TypedTensorRepr::Adjoint(_) = &lazy.repr else {
        unreachable!()
    };
}

fn assert_eig_uses_a_cold_logical_copy(source: &TensorMap<U1FusionRule, f64>) {
    let eager = eager_adjoint_oracle(source);
    let expected_vals = eager.eig_vals(&[0], &[1]).unwrap();
    let expected_full = eager.eig_full(&[0], &[1]).unwrap();
    let parent_body = Arc::clone(owned(source));
    let parent_data = Arc::clone(&parent_body.data);
    let lazy = source.adjoint().unwrap();

    for _ in 0..2 {
        assert_eq!(lazy.clone().eig_vals(&[0], &[1]).unwrap(), expected_vals);
        let full = lazy.clone().eig_full(&[0], &[1]).unwrap();
        assert_eq!(
            full.d.materialize().unwrap().dense_data().unwrap(),
            expected_full.d.materialize().unwrap().dense_data().unwrap()
        );
        assert_eq!(
            full.v.dense_data().unwrap(),
            expected_full.v.dense_data().unwrap()
        );
        for output in [&full.d, &full.v] {
            assert!(output.owned_body().is_some());
            assert!(Arc::ptr_eq(
                output.logical_space().provider_arc(),
                source.logical_space().provider_arc()
            ));
        }
    }

    let calls = (0..4)
        .map(|_| {
            let clone = lazy.clone();
            std::thread::spawn(move || {
                let vals = clone.eig_vals(&[0], &[1]).unwrap();
                let full = clone.eig_full(&[0], &[1]).unwrap();
                (
                    vals,
                    full.d.materialize().unwrap().dense_data().unwrap().to_vec(),
                )
            })
        })
        .collect::<Vec<_>>();
    for call in calls {
        let (vals, diagonal) = call.join().unwrap();
        assert_eq!(vals, expected_vals);
        assert_eq!(
            diagonal,
            expected_full.d.materialize().unwrap().dense_data().unwrap()
        );
    }
    assert!(Arc::ptr_eq(owned(source), &parent_body));
    assert!(Arc::ptr_eq(&owned(source).data, &parent_data));
    let TypedTensorRepr::Adjoint(_) = &lazy.repr else {
        unreachable!()
    };

    // Positive control: the probe sees an implicit materialization.
    let _ = lazy.materialized_tensor_uncached().unwrap();
}

#[test]
fn eig_dense_lazy_nonnormal_is_logical_owned_repeatable_and_cold() {
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let provider = Arc::new(U1FusionRule);
    let leg = GradedSpace::try_new(provider, [(U1Irrep::new(0), 2)]).unwrap();
    let source = TensorMap::from_subblock_fn(&runtime, [&leg], [&leg], |_, indices| {
        match (indices[0], indices[1]) {
            (0, 0) => 1.0_f64,
            (1, 1) => 2.0,
            (0, 1) => 1.0,
            _ => 0.0,
        }
    })
    .unwrap();
    let logical = eager_adjoint_oracle(&source);
    let Eig { d, v } = source.adjoint().unwrap().eig_full(&[0], &[1]).unwrap();
    let lhs = logical.convert::<Complex64>().compose(&v).unwrap();
    let rhs = v.compose(&d).unwrap();
    assert_typed_map_close(&lhs, &rhs, 1.0e-12);

    assert_eig_uses_a_cold_logical_copy(&source);
}

#[test]
fn eig_dense_lazy_real_order_signed_zero_and_defective_cases_match_logical_oracles() {
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let provider = Arc::new(U1FusionRule);
    let scalar_leg = GradedSpace::try_new(Arc::clone(&provider), [(U1Irrep::new(0), 1)]).unwrap();
    let negative =
        TensorMap::from_subblock_fn(&runtime, [&scalar_leg], [&scalar_leg], |_, _| -2.0).unwrap();
    let lazy = negative.adjoint().unwrap();
    let value = lazy.eig_vals(&[0], &[1]).unwrap()[0].values[0];
    assert_eq!(value, num_complex::Complex64::new(-2.0, 0.0));
    assert_eq!(value.im.to_bits(), 0.0f64.to_bits());

    let leg = GradedSpace::try_new(provider, [(U1Irrep::new(0), 2)]).unwrap();
    let rotation = TensorMap::from_subblock_fn(&runtime, [&leg], [&leg], |_, indices| {
        match (indices[0], indices[1]) {
            (0, 1) => -1.0,
            (1, 0) => 1.0,
            _ => 0.0,
        }
    })
    .unwrap();
    let eager = eager_adjoint_oracle(&rotation);
    let expected = eager.eig_vals(&[0], &[1]).unwrap();
    let lazy = rotation.adjoint().unwrap();
    assert_eq!(lazy.eig_vals(&[0], &[1]).unwrap(), expected);
    assert_eq!(expected[0].values[0].im, 1.0);
    assert_eq!(expected[0].values[1].im, -1.0);

    for epsilon in [0.0, 1.0e-12] {
        let jordan = TensorMap::from_subblock_fn(&runtime, [&leg], [&leg], |_, indices| {
            match (indices[0], indices[1]) {
                (0, 0) | (1, 1) => 1.0,
                (0, 1) => 1.0,
                (1, 0) => epsilon,
                _ => unreachable!(),
            }
        })
        .unwrap();
        let eager = eager_adjoint_oracle(&jordan);
        let lazy = jordan.adjoint().unwrap();
        assert_eq!(
            lazy.eig_vals(&[0], &[1]).unwrap(),
            eager.eig_vals(&[0], &[1]).unwrap()
        );
        let actual = lazy.eig_full(&[0], &[1]).unwrap();
        let expected = eager.eig_full(&[0], &[1]).unwrap();
        assert_eq!(
            actual.d.materialize().unwrap().dense_data().unwrap(),
            expected.d.materialize().unwrap().dense_data().unwrap()
        );
        assert_eq!(
            actual.v.dense_data().unwrap(),
            expected.v.dense_data().unwrap()
        );
    }
}

#[test]
fn eig_dense_lazy_failures_match_logical_oracle_and_stay_cold() {
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let provider = Arc::new(U1FusionRule);
    let left = GradedSpace::try_new(Arc::clone(&provider), [(U1Irrep::new(0), 2)]).unwrap();
    let right = GradedSpace::try_new(provider, [(U1Irrep::new(0), 3)]).unwrap();
    let source = TensorMap::from_subblock_fn(&runtime, [&left], [&right], |_, indices| {
        (indices[0] + indices[1]) as f64
    })
    .unwrap();
    let eager = eager_adjoint_oracle(&source);
    let expected = [
        eager.eig_vals(&[0], &[1]).unwrap_err().to_string(),
        eager.eig_full(&[0], &[1]).unwrap_err().to_string(),
    ];
    let lazy = source.adjoint().unwrap();
    for _ in 0..2 {
        assert_eq!(
            lazy.eig_vals(&[0], &[1]).unwrap_err().to_string(),
            expected[0]
        );
        assert_eq!(
            lazy.eig_full(&[0], &[1]).unwrap_err().to_string(),
            expected[1]
        );
    }
    let TypedTensorRepr::Adjoint(_) = &lazy.repr else {
        unreachable!()
    };
}

#[test]
fn exp_of_a_near_hermitian_adjoint_uses_the_logical_orientation_and_stays_cold() {
    // What: the fixed approximate-Hermitian dispatch must see logical A^H,
    // whose lower triangle is the conjugated parent upper triangle. A
    // parent-exp redirect feeds EIGH the other triangle and changes values.
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let provider = Arc::new(U1FusionRule);
    let leg = GradedSpace::try_new(provider, [(U1Irrep::new(0), 2)]).unwrap();
    let delta = 4.0e-15;
    let parent = TensorMap::from_subblock_fn(&runtime, [&leg], [&leg], |_, indices| {
        match (indices[0], indices[1]) {
            (0, 0) => num_complex::Complex64::new(0.25, 0.0),
            (1, 1) => num_complex::Complex64::new(-0.5, 0.0),
            (0, 1) => num_complex::Complex64::new(delta, 0.0),
            _ => num_complex::Complex64::new(0.0, 0.0),
        }
    })
    .unwrap();
    let eager = eager_adjoint_oracle(&parent);
    let expected = eager.exp(&[0], &[1]).unwrap();
    let parent_redirect = parent
        .exp(&[0], &[1])
        .unwrap()
        .adjoint()
        .unwrap()
        .materialized_tensor_uncached()
        .unwrap();
    let lazy = parent.adjoint().unwrap();
    let actual = lazy.exp(&[0], &[1]).unwrap();

    assert_typed_map_close(&actual, &expected, 1.0e-20);
    assert!(parent_redirect
        .dense_data()
        .unwrap()
        .iter()
        .zip(expected.dense_data().unwrap())
        .any(|(&actual, &expected)| {
            (actual.widen_complex() - expected.widen_complex()).norm() > 1.0e-16
        }));
    let TypedTensorRepr::Adjoint(_) = &lazy.repr else {
        unreachable!()
    };
}

fn assert_exp_uses_a_cold_logical_copy<R, D>(source: &TensorMap<R, D>)
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64>
        + CheckedFusionAlgebra
        + SectorCodec
        + Send
        + Sync
        + 'static,
    D: AdvancedLinalgScalar + core::fmt::Debug + Send + Sync + 'static,
{
    let eager = eager_adjoint_oracle(source);
    let expected = eager
        .exp(&codomain_axes(&eager), &domain_axes(&eager))
        .unwrap();
    let parent_body = Arc::clone(owned(source));
    let parent_data = Arc::clone(&parent_body.data);
    let lazy = source.adjoint().unwrap();

    for _ in 0..2 {
        let actual = lazy
            .clone()
            .exp(&codomain_axes(&lazy), &domain_axes(&lazy))
            .unwrap();
        assert_typed_map_close(&actual, &expected, 1.0e-9);
        assert!(actual.owned_body().is_some());
        assert!(Arc::ptr_eq(
            actual.logical_space().provider_arc(),
            source.logical_space().provider_arc()
        ));
        assert!(!Arc::ptr_eq(&owned(&actual).data, &parent_data));
        let _ = actual.dense_data().unwrap();
    }
    let calls = (0..4)
        .map(|_| {
            let clone = lazy.clone();
            std::thread::spawn(move || {
                clone
                    .exp(&codomain_axes(&clone), &domain_axes(&clone))
                    .unwrap()
            })
        })
        .collect::<Vec<_>>();
    for call in calls {
        assert_typed_map_close(&call.join().unwrap(), &expected, 1.0e-9);
    }
    assert!(Arc::ptr_eq(owned(source), &parent_body));
    assert!(Arc::ptr_eq(&owned(source).data, &parent_data));
    let TypedTensorRepr::Adjoint(_) = &lazy.repr else {
        unreachable!()
    };
}

#[test]
fn exp_uses_owned_provider_native_outputs_without_warming_lazy_receivers() {
    // What: real/complex non-self-dual U(1) and a genuine SU(2) multitree
    // remain deterministic across repeats, clones, and concurrent calls.
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let provider = Arc::new(U1FusionRule);
    let leg = GradedSpace::try_new(
        provider,
        [
            (U1Irrep::new(-1), 2),
            (U1Irrep::new(0), 3),
            (U1Irrep::new(2), 1),
        ],
    )
    .unwrap();
    let u1 = TensorMap::from_subblock_fn(&runtime, [&leg], [&leg], |_, indices| {
        if indices[0] == indices[1] {
            0.25 + indices[0] as f64 / 10.0
        } else {
            (indices[0] + 2 * indices[1] + 1) as f64 / 100.0
        }
    })
    .unwrap();
    assert_exp_uses_a_cold_logical_copy(&u1);
    assert_exp_uses_a_cold_logical_copy(&genuinely_complex(&u1));

    let provider = Arc::new(SU2FusionRule);
    let half = GradedSpace::try_new(provider, [(SU2Irrep::from_twice_spin(1), 1)]).unwrap();
    let su2 = TensorMap::from_subblock_fn(
        &runtime,
        [&half, &half, &half],
        [&half, &half, &half],
        |_, indices| (indices.iter().sum::<usize>() + 1) as f64 / 20.0,
    )
    .unwrap();
    assert!(su2.logical_space().space().structure().block_count() > 1);
    assert_exp_uses_a_cold_logical_copy(&genuinely_complex(&su2));
}

#[test]
fn exp_failure_leaves_the_lazy_receiver_and_parent_untouched() {
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let provider = Arc::new(U1FusionRule);
    let leg = GradedSpace::try_new(provider, [(U1Irrep::new(0), 2)]).unwrap();
    let source = TensorMap::from_subblock_fn(&runtime, [&leg], [&leg], |_, indices| {
        if indices == [0, 1] {
            num_complex::Complex64::new(f64::NAN, 0.0)
        } else {
            num_complex::Complex64::new((indices[0] + indices[1] + 1) as f64, 0.0)
        }
    })
    .unwrap();
    let before = source.dense_data().unwrap().to_vec();
    let parent = Arc::clone(owned(&source));
    let data = Arc::clone(&parent.data);
    let lazy = source.adjoint().unwrap();

    assert!(matches!(lazy.exp(&[0], &[1]), Err(Error::Operation(_))));
    assert!(source
        .dense_data()
        .unwrap()
        .iter()
        .zip(&before)
        .all(|(actual, expected)| {
            actual.re.to_bits() == expected.re.to_bits()
                && actual.im.to_bits() == expected.im.to_bits()
        }));
    assert!(Arc::ptr_eq(owned(&source), &parent));
    assert!(Arc::ptr_eq(&owned(&source).data, &data));
    let TypedTensorRepr::Adjoint(_) = &lazy.repr else {
        unreachable!()
    };
}

fn assert_polar_factors<R, D>(
    source: &TensorMap<R, D>,
    target: &TensorMap<R, D>,
    actual: &(TensorMap<R, D>, TensorMap<R, D>),
    expected: &(TensorMap<R, D>, TensorMap<R, D>),
    left: bool,
) where
    R: MultiplicityFreeRigidSymbols<Scalar = f64> + CheckedFusionAlgebra + SectorCodec,
    D: FactorizationScalar + core::fmt::Debug,
{
    let reconstructed = actual.0.compose(&actual.1).unwrap();
    assert_typed_map_close(&reconstructed, target, 1e-10);
    let (positive, isometry) = if left {
        (&actual.1, &actual.0)
    } else {
        (&actual.0, &actual.1)
    };
    assert!(if left {
        is_isometric!(isometry, 1e-11)
    } else {
        is_isometric!(isometry.adjoint().unwrap(), 1e-11)
    });
    assert!(is_hermitian!(positive, 1e-11));
    assert!(positive
        .eigh_vals(&[0], &[1])
        .unwrap()
        .iter()
        .all(|entry| entry.values.iter().all(|&value| value >= -1e-11)));
    for factor in [&actual.0, &actual.1] {
        assert!(factor.owned_body().is_some());
        assert!(Arc::ptr_eq(
            factor.logical_space().provider_arc(),
            source.logical_space().provider_arc()
        ));
        let _ = factor.materialize().unwrap().dense_data().unwrap();
    }
    assert_eq!(
        actual.0.logical_space().space(),
        expected.0.logical_space().space()
    );
    assert_eq!(
        actual.1.logical_space().space(),
        expected.1.logical_space().space()
    );
}

fn assert_rank_deficient_polar_support<R, D>(source: &TensorMap<R, D>, left: bool)
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64> + CheckedFusionAlgebra + SectorCodec,
    D: AdvancedLinalgScalar + core::fmt::Debug,
{
    let target = eager_adjoint_oracle(source);
    let target_pinv = target.pinv(&[0], &[1], 1e-10).unwrap();
    let target_codomain = target.compose(&target_pinv).unwrap();
    let target_domain = target_pinv.compose(&target).unwrap();
    let lazy = source.adjoint().unwrap();
    let factors = if left {
        lazy.left_polar(&[0], &[1]).unwrap().pair()
    } else {
        lazy.right_polar(&[0], &[1]).unwrap().pair()
    };
    let (positive, isometry) = if left {
        (&factors.1, &factors.0)
    } else {
        (&factors.0, &factors.1)
    };
    let positive_pinv = positive.pinv(&[0], &[1], 1e-10).unwrap();
    if left {
        let support = positive_pinv.compose(positive).unwrap();
        assert_typed_map_close(&support, &target_domain, 1e-9);
        let image = isometry
            .compose(&support)
            .unwrap()
            .compose(&isometry.adjoint().unwrap())
            .unwrap();
        assert_typed_map_close(&image, &target_codomain, 1e-9);
    } else {
        let support = positive.compose(&positive_pinv).unwrap();
        assert_typed_map_close(&support, &target_codomain, 1e-9);
        let image = isometry
            .adjoint()
            .unwrap()
            .compose(&support)
            .unwrap()
            .compose(isometry)
            .unwrap();
        assert_typed_map_close(&image, &target_domain, 1e-9);
    }
    let TypedTensorRepr::Adjoint(_) = &lazy.repr else {
        unreachable!()
    };
}

fn assert_inverse_redirect<R, D>(source: &TensorMap<R, D>)
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64>
        + CheckedFusionAlgebra
        + SectorCodec
        + Send
        + Sync
        + 'static,
    D: AdvancedLinalgScalar + core::fmt::Debug + Send + Sync + 'static,
{
    let eager = eager_adjoint_oracle(source);
    let expected = eager
        .inv(&codomain_axes(&eager), &domain_axes(&eager))
        .unwrap();
    let parent_body = Arc::clone(owned(source));
    let parent_data = Arc::clone(&parent_body.data);
    let lazy = source.adjoint().unwrap();

    for _ in 0..2 {
        let actual = lazy
            .clone()
            .inv(&codomain_axes(&lazy), &domain_axes(&lazy))
            .unwrap();
        assert_typed_map_close(&actual, &expected, 1e-10);
        let codomain = eager.codomain();
        let domain = eager.domain();
        assert_typed_map_close(
            &eager.compose(&actual).unwrap(),
            &TensorMap::isomorphism(source.runtime(), codomain.iter(), codomain.iter()).unwrap(),
            1e-9,
        );
        assert_typed_map_close(
            &actual.compose(&eager).unwrap(),
            &TensorMap::isomorphism(source.runtime(), domain.iter(), domain.iter()).unwrap(),
            1e-9,
        );
        assert!(actual.owned_body().is_some());
        assert!(Arc::ptr_eq(
            actual.logical_space().provider_arc(),
            source.logical_space().provider_arc()
        ));
        assert!(!Arc::ptr_eq(&owned(&actual).data, &parent_data));
        let _ = actual.dense_data().unwrap();
    }

    let calls = (0..4)
        .map(|_| {
            let clone = lazy.clone();
            std::thread::spawn(move || {
                clone
                    .inv(&codomain_axes(&clone), &domain_axes(&clone))
                    .unwrap()
            })
        })
        .collect::<Vec<_>>();
    for call in calls {
        assert_typed_map_close(&call.join().unwrap(), &expected, 1e-10);
    }

    assert!(Arc::ptr_eq(owned(source), &parent_body));
    assert!(Arc::ptr_eq(&owned(source).data, &parent_data));
    let TypedTensorRepr::Adjoint(_) = &lazy.repr else {
        unreachable!()
    };
}

#[test]
fn inverse_redirect_is_owned_provider_native_repeatable_and_cold() {
    // What: U(1) complex blocks and a genuine SU(2) multitree use the
    // inverse identity without materializing the lazy receiver, including
    // cloned and concurrent calls, and return detached provider-native data.
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let provider = Arc::new(U1FusionRule);
    let leg = GradedSpace::try_new(
        provider,
        [
            (U1Irrep::new(-1), 2),
            (U1Irrep::new(0), 3),
            (U1Irrep::new(2), 1),
        ],
    )
    .unwrap();
    let u1 = TensorMap::from_subblock_fn(&runtime, [&leg], [&leg], |_, indices| {
        let re = if indices[0] == indices[1] {
            20.0 + indices[0] as f64
        } else {
            (indices[0] + 2 * indices[1] + 1) as f64 / 100.0
        };
        num_complex::Complex64::new(re, (indices[0] + indices[1] + 1) as f64 / 200.0)
    })
    .unwrap();
    let identity =
        TensorMap::<_, num_complex::Complex64>::isomorphism(&runtime, [&leg], [&leg]).unwrap();
    let u1 = u1
        .axpby(
            num_complex::Complex64::new(1.0, 0.0),
            &identity,
            num_complex::Complex64::new(100.0, 0.0),
        )
        .unwrap();
    assert_inverse_redirect(&u1);

    let provider = Arc::new(U1FusionRule);
    let wide = GradedSpace::try_new(Arc::clone(&provider), [(U1Irrep::new(0), 4)]).unwrap();
    let narrow = GradedSpace::try_new(provider, [(U1Irrep::new(0), 2)]).unwrap();
    let unequal =
        TensorMap::from_subblock_fn(&runtime, [&wide], [&narrow, &narrow], |_, indices| {
            let column = 2 * indices[1] + indices[2];
            if indices[0] == column {
                10.0 + indices[0] as f64
            } else {
                (indices[0] + column + 1) as f64 / 100.0
            }
        })
        .unwrap();
    assert_ne!(unequal.codomain_rank(), unequal.domain_rank());
    assert_inverse_redirect(&unequal);

    let provider = Arc::new(SU2FusionRule);
    let half = GradedSpace::try_new(provider, [(SU2Irrep::from_twice_spin(1), 1)]).unwrap();
    let su2 = TensorMap::from_subblock_fn(
        &runtime,
        [&half, &half, &half],
        [&half, &half, &half],
        |_, indices| {
            if indices[..3] == indices[3..] {
                20.0 + indices.iter().sum::<usize>() as f64
            } else {
                (indices.iter().sum::<usize>() + 1) as f64 / 100.0
            }
        },
    )
    .unwrap();
    let identity =
        TensorMap::<_, f64>::isomorphism(&runtime, [&half, &half, &half], [&half, &half, &half])
            .unwrap();
    let su2 = su2.axpby(1.0, &identity, 100.0).unwrap();
    assert!(su2.logical_space().space().structure().block_count() > 1);
    assert_inverse_redirect(&su2);
}

#[test]
fn inverse_redirect_failure_leaves_the_receiver_cold() {
    // What: a singular solve changes neither parent Arc/bytes nor the lazy
    // receiver.
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let provider = Arc::new(U1FusionRule);
    let leg = GradedSpace::try_new(provider, [(U1Irrep::new(0), 3)]).unwrap();
    let source = TensorMap::from_subblock_fn(&runtime, [&leg], [&leg], |_, indices| {
        if indices[0] == indices[1] {
            4.0 + indices[0] as f64
        } else {
            (indices[0] + indices[1] + 1) as f64 / 20.0
        }
    })
    .unwrap();
    let singular = source.scale(0.0);
    let before = singular.dense_data().unwrap().to_vec();
    let body = Arc::clone(owned(&singular));
    let data = Arc::clone(&body.data);
    let cold = singular.adjoint().unwrap();
    assert!(matches!(cold.inv(&[0], &[1]), Err(Error::Operation(_))));
    assert_eq!(singular.dense_data().unwrap(), before);
    assert!(Arc::ptr_eq(owned(&singular), &body));
    assert!(Arc::ptr_eq(&owned(&singular).data, &data));
    let TypedTensorRepr::Adjoint(_) = &cold.repr else {
        unreachable!()
    };

    // The first U(1) sector solves before the second singular sector
    // fails, pinning atomicity after partial backend progress.
    let provider = Arc::new(U1FusionRule);
    let leg = GradedSpace::try_new(provider, [(U1Irrep::new(0), 1), (U1Irrep::new(1), 2)]).unwrap();
    let late = TensorMap::from_subblock_fn(&runtime, [&leg], [&leg], |trees, indices| {
        if trees.codomain_uncoupled[0] == U1Irrep::new(0) && indices[0] == indices[1] {
            2.0
        } else {
            0.0
        }
    })
    .unwrap();
    let before = late.dense_data().unwrap().to_vec();
    let data = Arc::clone(&owned(&late).data);
    let cold = late.adjoint().unwrap();
    assert!(matches!(cold.inv(&[0], &[1]), Err(Error::Operation(_))));
    assert_eq!(late.dense_data().unwrap(), before);
    assert!(Arc::ptr_eq(&owned(&late).data, &data));
    let TypedTensorRepr::Adjoint(_) = &cold.repr else {
        unreachable!()
    };
}

#[test]
fn solve_is_transactional_provider_native_and_cache_cold() {
    DIAGONAL_MATERIALIZATIONS.set(0);
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let lhs_provider = Arc::new(U1FusionRule);
    let rhs_provider = Arc::new(U1FusionRule);
    let lhs_leg = GradedSpace::try_new(
        Arc::clone(&lhs_provider),
        [(U1Irrep::new(0), 2), (U1Irrep::new(1), 1)],
    )
    .unwrap();
    let rhs_codomain = GradedSpace::try_new(
        Arc::clone(&rhs_provider),
        [(U1Irrep::new(0), 2), (U1Irrep::new(1), 1)],
    )
    .unwrap();
    let rhs_domain =
        GradedSpace::try_new(rhs_provider, [(U1Irrep::new(0), 3), (U1Irrep::new(1), 2)]).unwrap();
    let divisor = TensorMap::from_subblock_fn(&runtime, [&lhs_leg], [&lhs_leg], |_, indices| {
        if indices[0] == indices[1] {
            4.0 + indices[0] as f64
        } else {
            0.25
        }
    })
    .unwrap();
    let rhs =
        TensorMap::from_subblock_fn(&runtime, [&rhs_codomain], [&rhs_domain], |_, indices| {
            (indices[0] + 2 * indices[1] + 1) as f64
        })
        .unwrap();

    let solution = divisor.solve(&[0], &[1], &rhs, &[0], &[1]).unwrap();
    assert_typed_map_close(&divisor.compose(&solution).unwrap(), &rhs, 1e-11);
    assert!(Arc::ptr_eq(
        solution.logical_space().provider_arc(),
        &lhs_provider
    ));

    let complex_divisor = divisor.convert::<Complex64>();
    let complex_rhs = rhs
        .convert::<Complex64>()
        .scale(num_complex::Complex64::new(1.0, 0.25));
    let complex_solution = complex_divisor
        .solve(&[0], &[1], &complex_rhs, &[0], &[1])
        .unwrap();
    assert_typed_map_close(
        &complex_divisor.compose(&complex_solution).unwrap(),
        &complex_rhs,
        1e-11,
    );
    assert!(Arc::ptr_eq(
        complex_solution.logical_space().provider_arc(),
        &lhs_provider
    ));

    let lazy = divisor.adjoint().unwrap();
    let expected = eager_adjoint_oracle(&divisor)
        .solve(&[0], &[1], &rhs, &[0], &[1])
        .unwrap();
    assert_typed_map_close(
        &lazy.solve(&[0], &[1], &rhs, &[0], &[1]).unwrap(),
        &expected,
        1e-11,
    );

    let square_rhs =
        TensorMap::from_subblock_fn(&runtime, [&rhs_codomain], [&rhs_codomain], |_, indices| {
            (2 * indices[0] + indices[1] + 1) as f64
        })
        .unwrap();
    let lazy_rhs = square_rhs.adjoint().unwrap();
    let expected = divisor
        .solve(&[0], &[1], &eager_adjoint_oracle(&square_rhs), &[0], &[1])
        .unwrap();
    assert_typed_map_close(
        &divisor.solve(&[0], &[1], &lazy_rhs, &[0], &[1]).unwrap(),
        &expected,
        1e-11,
    );

    let bad_leg = GradedSpace::try_new(Arc::clone(&lhs_provider), [(U1Irrep::new(7), 1)]).unwrap();
    let bad = TensorMap::from_subblock_fn(&runtime, [&bad_leg], [&bad_leg], |_, _| 1.0)
        .unwrap()
        .adjoint()
        .unwrap();
    assert!(matches!(
        lazy.solve(&[0], &[1], &bad, &[0], &[1]),
        Err(Error::InvalidArgument(_))
    ));

    let singular_dense = divisor.scale(0.0).adjoint().unwrap();
    assert!(matches!(
        singular_dense.solve(&[0], &[1], &lazy_rhs, &[0], &[1]),
        Err(Error::Operation(_))
    ));

    let compact_rhs = TensorMap::diagonal(
        &runtime,
        &lhs_leg,
        [
            SectorSpectrum {
                sector: U1Irrep::new(0),
                values: vec![2.0, 3.0],
            },
            SectorSpectrum {
                sector: U1Irrep::new(1),
                values: vec![5.0],
            },
        ],
    )
    .unwrap();
    let compact_solution = divisor.solve(&[0], &[1], &compact_rhs, &[0], &[1]).unwrap();
    // A dense divisor densifies the compact RHS into its solve buffer once.
    assert_eq!(DIAGONAL_MATERIALIZATIONS.get(), 1);
    DIAGONAL_MATERIALIZATIONS.set(0);

    let compact_divisor = TensorMap::diagonal(
        &runtime,
        &lhs_leg,
        [
            SectorSpectrum {
                sector: U1Irrep::new(0),
                values: vec![2.0, 4.0],
            },
            SectorSpectrum {
                sector: U1Irrep::new(1),
                values: vec![8.0],
            },
        ],
    )
    .unwrap();
    let compact_compact = compact_divisor
        .solve(&[0], &[1], &compact_rhs, &[0], &[1])
        .unwrap();
    assert!(compact_compact.spectrum().is_some());
    assert_eq!(compact_compact.spectrum().unwrap()[0].values, [1.0, 0.75]);
    assert_eq!(compact_compact.spectrum().unwrap()[1].values, [0.625]);
    assert_eq!(DIAGONAL_MATERIALIZATIONS.get(), 0);
    assert_typed_map_close(
        &divisor.compose(&compact_solution).unwrap(),
        &compact_rhs,
        1e-11,
    );

    let scaled = compact_divisor.solve(&[0], &[1], &rhs, &[0], &[1]).unwrap();
    assert_typed_map_close(&compact_divisor.compose(&scaled).unwrap(), &rhs, 1e-11);
    assert!(Arc::ptr_eq(
        scaled.logical_space().provider_arc(),
        compact_divisor.logical_space().provider_arc()
    ));
    assert_eq!(DIAGONAL_MATERIALIZATIONS.get(), 0);

    let singular = TensorMap::diagonal(
        &runtime,
        &lhs_leg,
        [
            SectorSpectrum {
                sector: U1Irrep::new(0),
                values: vec![2.0, 0.0],
            },
            SectorSpectrum {
                sector: U1Irrep::new(1),
                values: vec![8.0],
            },
        ],
    )
    .unwrap();
    assert!(matches!(
        singular.solve(&[0], &[1], &rhs, &[0], &[1]),
        Err(Error::Operation(error))
            if matches!(*error, tenet_tensors::OperationError::Dense(
                tenet_dense::DenseError::NumericalFailure { op: "solve_into", .. }
            ))
    ));
    assert_eq!(DIAGONAL_MATERIALIZATIONS.get(), 0);
}

#[cfg(feature = "racah-generated")]
fn assert_checked_generic_solve_acceptance<D>()
where
    D: AdvancedLinalgScalar + core::fmt::Debug,
{
    use tenet_core::SUNFusionRule;

    let calls = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let runtime = Runtime::builder()
        .with_dense_executor(Box::new(CountingSolve {
            inner: DefaultDenseExecutor::default(),
            calls: Arc::clone(&calls),
            failure: None,
        }))
        .build()
        .unwrap();
    let provider = Arc::new(SUNFusionRule::new(3).unwrap());
    let leg = GradedSpace::try_new(Arc::clone(&provider), [(vec![1, 1], 2)]).unwrap();
    let divisor: TensorMap<_, D> =
        TensorMap::from_subblock_fn(&runtime, [&leg, &leg], [&leg, &leg], |trees, ij| {
            let row = ij[0] + 2 * ij[1];
            let col = ij[2] + 2 * ij[3];
            D::from_real(if row == col {
                7.0 + trees.codomain_vertices()[0].get() as f64
            } else {
                0.125 * (1 + trees.domain_vertices()[0].get()) as f64
            })
        })
        .unwrap();
    let rhs: TensorMap<_, D> =
        TensorMap::from_subblock_fn(&runtime, [&leg, &leg], [&leg, &leg], |trees, ij| {
            D::from_real(
                (1 + ij.iter().sum::<usize>() + 3 * trees.domain_vertices()[0].get()) as f64,
            )
        })
        .unwrap();
    assert!((0..divisor.subblock_count()).any(|i| {
        divisor
            .subblock_fusion_trees(i)
            .unwrap()
            .codomain_vertices()[0]
            .get()
            == 2
    }));

    for (lazy_lhs, lazy_rhs) in [(false, false), (true, false), (false, true), (true, true)] {
        let lhs = if lazy_lhs {
            divisor.adjoint().unwrap()
        } else {
            divisor.clone()
        };
        let right = if lazy_rhs {
            rhs.adjoint().unwrap()
        } else {
            rhs.clone()
        };
        let lhs_before = divisor.dense_data().unwrap().to_vec();
        let rhs_before = rhs.dense_data().unwrap().to_vec();
        calls.store(0, std::sync::atomic::Ordering::Relaxed);
        let solution = lhs
            .solve(&[0, 1], &[2, 3], &right, &[0, 1], &[2, 3])
            .unwrap();
        assert!(matches!(solution.repr, TypedTensorRepr::Owned(_)));
        assert_eq!(
            calls.load(std::sync::atomic::Ordering::Relaxed),
            5,
            "the SU(3) μ=2 fixture has five nonempty coupled-sector solve routes"
        );
        let lhs_oracle = lhs.materialized_tensor_uncached().unwrap();
        let rhs_oracle = right.materialized_tensor_uncached().unwrap();
        let reconstructed = lhs_oracle.compose(&solution).unwrap();
        assert!(reconstructed
            .dense_data()
            .unwrap()
            .iter()
            .zip(rhs_oracle.dense_data().unwrap())
            .all(|(&actual, &expected)| {
                (actual.widen_complex() - expected.widen_complex()).norm() < 2e-10
            }));
        for i in 0..solution.subblock_count() {
            assert_eq!(
                reconstructed.subblock_fusion_trees(i).unwrap(),
                rhs_oracle.subblock_fusion_trees(i).unwrap(),
            );
        }
        assert_eq!(divisor.dense_data().unwrap(), lhs_before.as_slice());
        assert_eq!(rhs.dense_data().unwrap(), rhs_before.as_slice());
    }
}

#[cfg(feature = "racah-generated")]
#[test]
fn checked_generic_solves_are_owned_uncached_and_one_call_per_route() {
    // What: the solve keeps all four lazy input pairs cold on
    // nondegenerate real and complex cross-multiplicity matrices.
    assert_checked_generic_solve_acceptance::<f64>();
    assert_checked_generic_solve_acceptance::<num_complex::Complex64>();
}

#[cfg(feature = "racah-generated")]
#[test]
fn checked_generic_left_solve_preserves_injected_backend_provenance() {
    use tenet_core::SUNFusionRule;

    let calls = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let runtime = Runtime::builder()
        .with_dense_executor(Box::new(CountingSolve {
            inner: DefaultDenseExecutor::default(),
            calls: Arc::clone(&calls),
            failure: Some("injected checked solve failure"),
        }))
        .build()
        .unwrap();
    let provider = Arc::new(SUNFusionRule::new(3).unwrap());
    let leg = GradedSpace::try_new(Arc::clone(&provider), [(vec![1, 1], 1)]).unwrap();
    let lhs: TensorMap<_, f64> =
        TensorMap::from_subblock_fn(&runtime, [&leg], [&leg], |_, ij| f64::from(ij[0] == ij[1]))
            .unwrap();
    let rhs = lhs.scale(2.0);
    let before_lhs = lhs.dense_data().unwrap().to_vec();
    let before_rhs = rhs.dense_data().unwrap().to_vec();
    assert!(matches!(
        lhs.solve(&[0], &[1], &rhs, &[0], &[1]),
        Err(GenericTensorError::Facade(Error::Operation(error)))
            if matches!(*error, tenet_tensors::OperationError::Dense(
                DenseError::Backend { op: "solve_into", ref message, .. }
            ) if message == "injected checked solve failure")
    ));
    assert_eq!(calls.load(std::sync::atomic::Ordering::Relaxed), 1);
    assert_eq!(lhs.dense_data().unwrap(), before_lhs.as_slice());
    assert_eq!(rhs.dense_data().unwrap(), before_rhs.as_slice());
}

fn assert_pinv_redirect<R, D>(source: &TensorMap<R, D>, rcond: f64, exact_original: bool)
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64>
        + CheckedFusionAlgebra
        + SectorCodec
        + Send
        + Sync
        + 'static,
    D: AdvancedLinalgScalar + core::fmt::Debug + Send + Sync + 'static,
{
    let eager = eager_adjoint_oracle(source);
    let expected = eager
        .pinv(&codomain_axes(&eager), &domain_axes(&eager), rcond)
        .unwrap();
    let parent_body = Arc::clone(owned(source));
    let parent_data = Arc::clone(&parent_body.data);
    let lazy = source.adjoint().unwrap();

    for _ in 0..2 {
        let actual = lazy
            .clone()
            .pinv(&codomain_axes(&lazy), &domain_axes(&lazy), rcond)
            .unwrap();
        assert_typed_map_close(&actual, &expected, 1e-9);
        let pap = actual.compose(&eager).unwrap().compose(&actual).unwrap();
        assert_typed_map_close(&pap, &actual, 1e-8);
        assert!(is_hermitian!(eager.compose(&actual).unwrap(), 1e-9));
        assert!(is_hermitian!(actual.compose(&eager).unwrap(), 1e-9));
        if exact_original {
            let apa = eager.compose(&actual).unwrap().compose(&eager).unwrap();
            assert_typed_map_close(&apa, &eager, 1e-8);
        }
        assert!(actual.owned_body().is_some());
        assert!(Arc::ptr_eq(
            actual.logical_space().provider_arc(),
            source.logical_space().provider_arc()
        ));
        assert!(!Arc::ptr_eq(&owned(&actual).data, &parent_data));
        let _ = actual.dense_data().unwrap();
    }

    let calls = (0..4)
        .map(|_| {
            let clone = lazy.clone();
            std::thread::spawn(move || {
                clone
                    .pinv(&codomain_axes(&clone), &domain_axes(&clone), rcond)
                    .unwrap()
            })
        })
        .collect::<Vec<_>>();
    for call in calls {
        assert_typed_map_close(&call.join().unwrap(), &expected, 1e-9);
    }
    assert!(Arc::ptr_eq(owned(source), &parent_body));
    assert!(Arc::ptr_eq(&owned(source).data, &parent_data));
    let TypedTensorRepr::Adjoint(_) = &lazy.repr else {
        unreachable!()
    };
}

#[test]
fn pinv_redirect_preserves_semantics_ownership_and_cold_concurrency() {
    // What: full-rank, rectangular/rank-deficient, non-self-dual U(1),
    // complex data, empty support, and a genuine SU(2) multitree all use
    // the parent-factor seam and return detached provider-native outputs.
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let provider = Arc::new(U1FusionRule);
    let leg =
        GradedSpace::try_new(provider, [(U1Irrep::new(-1), 2), (U1Irrep::new(0), 3)]).unwrap();
    let full = TensorMap::from_subblock_fn(&runtime, [&leg], [&leg], |_, indices| {
        let re = if indices[0] == indices[1] {
            30.0 + indices[0] as f64
        } else {
            (indices[0] + indices[1] + 1) as f64 / 100.0
        };
        num_complex::Complex64::new(re, (indices[0] + 2 * indices[1] + 1) as f64 / 200.0)
    })
    .unwrap();
    assert_pinv_redirect(&full, 1e-12, true);

    assert_pinv_redirect(
        &genuinely_complex(&u1_matrix_fixture([(0, 3), (1, 2)], [(0, 2)])),
        1e-10,
        false,
    );
    assert_pinv_redirect(&u1_matrix_fixture([(1, 2)], [(0, 3)]), 1e-10, false);
    let su2 = genuinely_complex(&su2_lazy_fixture());
    assert!(su2.logical_space().space().structure().block_count() > 1);
    assert_pinv_redirect(&su2, 1e-10, false);
}

#[test]
fn pinv_redirect_late_svd_failure_keeps_parent_and_receiver_cold() {
    // What: a successful first-sector SVD cannot publish factors, mutate
    // parent bytes, or initialize the lazy receiver when sector two fails.
    let runtime = Runtime::builder()
        .with_dense_executor(Box::new(FailSecondSvd::default()))
        .build()
        .unwrap();
    let provider = Arc::new(U1FusionRule);
    let leg = GradedSpace::try_new(provider, [(U1Irrep::new(0), 2), (U1Irrep::new(1), 2)]).unwrap();
    let source = TensorMap::from_subblock_fn(&runtime, [&leg], [&leg], |_, indices| {
        (indices.iter().sum::<usize>() + 1) as f64
    })
    .unwrap();
    let before = source.dense_data().unwrap().to_vec();
    let data = Arc::clone(&owned(&source).data);
    let lazy = source.adjoint().unwrap();
    assert!(matches!(
        lazy.pinv(&[0], &[1], 0.0),
        Err(Error::Operation(_))
    ));
    assert_eq!(source.dense_data().unwrap(), before);
    assert!(Arc::ptr_eq(&owned(&source).data, &data));
    let TypedTensorRepr::Adjoint(_) = &lazy.repr else {
        unreachable!()
    };
}

#[test]
fn polar_redirects_through_parent_with_owned_psd_factors_and_a_cold_receiver() {
    let tall = u1_matrix_fixture([(0, 3)], [(0, 2)]);
    assert_polar_redirect(&tall, false);
    assert_polar_redirect(&genuinely_complex(&tall), false);

    let wide = u1_matrix_fixture([(0, 2)], [(0, 3)]);
    assert_polar_redirect(&wide, true);

    let codomain_only = u1_matrix_fixture([(0, 2), (1, 2)], [(0, 2)]);
    assert_polar_redirect(&codomain_only, false);
    let domain_only = u1_matrix_fixture([(0, 2)], [(0, 2), (1, 2)]);
    assert_polar_redirect(&domain_only, true);

    let provider = Arc::new(SU2FusionRule);
    let half = GradedSpace::try_new(provider, [(SU2Irrep::from_twice_spin(1), 1)]).unwrap();
    let multitree = TensorMap::from_subblock_fn(
        &Runtime::builder().dense_threads(1).build().unwrap(),
        [&half, &half, &half],
        [&half],
        |_, indices| (indices.iter().sum::<usize>() + 1) as f64,
    )
    .unwrap();
    assert_eq!(
        multitree.logical_space().space().structure().block_count(),
        2
    );
    let multitree = genuinely_complex(&multitree);
    assert_polar_redirect(&multitree, false);

    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let provider = Arc::new(U1FusionRule);
    let leg = GradedSpace::try_new(provider, [(U1Irrep::new(0), 3)]).unwrap();
    let rank_deficient = TensorMap::from_subblock_fn(&runtime, [&leg], [&leg], |_, indices| {
        let value = ((indices[0] + 1) * (indices[1] + 1)) as f64;
        num_complex::Complex64::new(value, value / 3.0)
    })
    .unwrap();
    assert_polar_redirect(&rank_deficient, true);
    assert_polar_redirect(&rank_deficient, false);
    assert_rank_deficient_polar_support(&rank_deficient, true);
    assert_rank_deficient_polar_support(&rank_deficient, false);
}

#[test]
fn polar_redirect_repeats_clones_and_runs_concurrently_without_warming_receiver() {
    let source = genuinely_complex(&u1_matrix_fixture([(0, 3)], [(0, 3)]));
    let target = eager_adjoint_oracle(&source);
    let lazy = source.adjoint().unwrap();
    for left in [true, false] {
        let expected = if left {
            target.left_polar(&[0], &[1]).unwrap().pair()
        } else {
            target.right_polar(&[0], &[1]).unwrap().pair()
        };
        for _ in 0..2 {
            let actual = if left {
                lazy.clone().left_polar(&[0], &[1]).unwrap().pair()
            } else {
                lazy.clone().right_polar(&[0], &[1]).unwrap().pair()
            };
            assert_polar_factors(&source, &target, &actual, &expected, left);
        }
        let calls = (0..4)
            .map(|_| {
                let clone = lazy.clone();
                std::thread::spawn(move || {
                    if left {
                        clone.left_polar(&[0], &[1]).unwrap().pair()
                    } else {
                        clone.right_polar(&[0], &[1]).unwrap().pair()
                    }
                })
            })
            .collect::<Vec<_>>();
        for call in calls {
            let actual = call.join().unwrap();
            assert_polar_factors(&source, &target, &actual, &expected, left);
        }
    }
    let TypedTensorRepr::Adjoint(_) = &lazy.repr else {
        unreachable!()
    };
}

#[test]
fn polar_redirect_wrong_direction_keeps_requested_name_and_receiver_cold() {
    let source = u1_matrix_fixture([(0, 3)], [(0, 2)]);
    let lazy = source.adjoint().unwrap();
    let error = lazy.left_polar(&[0], &[1]).unwrap_err();
    assert!(matches!(
        error,
        Error::Operation(error)
            if matches!(
                error.as_ref(),
                tenet_tensors::OperationError::InvalidArgument { message }
                    if *message == "left_polar requires rows >= columns in every coupled-sector matrix"
            )
    ));

    let source = u1_matrix_fixture([(0, 2)], [(0, 3)]);
    let lazy = source.adjoint().unwrap();
    let error = lazy.right_polar(&[0], &[1]).unwrap_err();
    assert!(matches!(
        error,
        Error::Operation(error)
            if matches!(
                error.as_ref(),
                tenet_tensors::OperationError::InvalidArgument { message }
                    if *message == "right_polar requires columns >= rows in every coupled-sector matrix"
            )
    ));
}

#[test]
fn polar_redirect_late_failure_leaves_parent_and_receiver_unchanged() {
    for left in [true, false] {
        let runtime = Runtime::builder()
            .with_dense_executor(Box::new(FailSecondSvd::default()))
            .build()
            .unwrap();
        let provider = Arc::new(U1FusionRule);
        let leg =
            GradedSpace::try_new(provider, [(U1Irrep::new(0), 2), (U1Irrep::new(1), 2)]).unwrap();
        let source = TensorMap::from_subblock_fn(&runtime, [&leg], [&leg], |_, indices| {
            (indices.iter().sum::<usize>() + 1) as f64
        })
        .unwrap();
        let before = source.dense_data().unwrap().to_vec();
        let lazy = source.adjoint().unwrap();
        let result = if left {
            lazy.left_polar(&[0], &[1]).map(drop)
        } else {
            lazy.right_polar(&[0], &[1]).map(drop)
        };
        assert!(matches!(result, Err(Error::Operation(_))));
        assert_eq!(source.dense_data().unwrap(), before);
        let TypedTensorRepr::Adjoint(_) = &lazy.repr else {
            unreachable!()
        };
    }
}

fn assert_qr_lq_factors<R, D>(
    source: &TensorMap<R, D>,
    target: &TensorMap<R, D>,
    actual: &(TensorMap<R, D>, TensorMap<R, D>),
    expected: &(TensorMap<R, D>, TensorMap<R, D>),
    qr: bool,
    compare_gauge: bool,
) where
    R: MultiplicityFreeRigidSymbols<Scalar = f64> + CheckedFusionAlgebra + SectorCodec,
    D: TensorScalar + core::fmt::Debug,
{
    for (actual, expected) in [(&actual.0, &expected.0), (&actual.1, &expected.1)] {
        assert!(actual.owned_body().is_some());
        assert_eq!(
            actual.logical_space().space(),
            expected.logical_space().space()
        );
        assert!(Arc::ptr_eq(
            actual.logical_space().provider_arc(),
            source.logical_space().provider_arc()
        ));
        if compare_gauge {
            assert!(actual
                .dense_data()
                .unwrap()
                .iter()
                .zip(expected.dense_data().unwrap())
                .all(|(&left, &right)| {
                    (left.widen_complex() - right.widen_complex()).norm() < 1e-12
                }));
        }
    }
    let isometry = if qr {
        is_isometric!(actual.0, 1e-12)
    } else {
        is_isometric!(actual.1.adjoint().unwrap(), 1e-12)
    };
    assert!(isometry);
    let rebuilt = actual.0.compose(&actual.1).unwrap();
    assert!(rebuilt
        .dense_data()
        .unwrap()
        .iter()
        .zip(target.dense_data().unwrap())
        .all(|(&left, &right)| { (left.widen_complex() - right.widen_complex()).norm() < 1e-12 }));
}

fn assert_qr_lq_keeps_input_cache_cold<R, D>(source: &TensorMap<R, D>)
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64> + CheckedFusionAlgebra + SectorCodec,
    D: FactorizationScalar + core::fmt::Debug,
{
    let target = eager_adjoint_oracle(source);
    let lazy = source.adjoint().unwrap();

    let actual = lazy
        .qr_compact(&codomain_axes(&lazy), &domain_axes(&lazy))
        .unwrap()
        .pair();
    assert_qr_lq_factors(
        source,
        &target,
        &actual,
        &target
            .qr_compact(&codomain_axes(&target), &domain_axes(&target))
            .unwrap()
            .pair(),
        true,
        true,
    );
    let actual = lazy
        .lq_compact(&codomain_axes(&lazy), &domain_axes(&lazy))
        .unwrap()
        .pair();
    assert_qr_lq_factors(
        source,
        &target,
        &actual,
        &target
            .lq_compact(&codomain_axes(&target), &domain_axes(&target))
            .unwrap()
            .pair(),
        false,
        true,
    );
    let actual = lazy
        .qr_full(&codomain_axes(&lazy), &domain_axes(&lazy))
        .unwrap()
        .pair();
    assert_qr_lq_factors(
        source,
        &target,
        &actual,
        &target
            .qr_full(&codomain_axes(&target), &domain_axes(&target))
            .unwrap()
            .pair(),
        true,
        false,
    );
    let actual = lazy
        .lq_full(&codomain_axes(&lazy), &domain_axes(&lazy))
        .unwrap()
        .pair();
    assert_qr_lq_factors(
        source,
        &target,
        &actual,
        &target
            .lq_full(&codomain_axes(&target), &domain_axes(&target))
            .unwrap()
            .pair(),
        false,
        false,
    );

    let TypedTensorRepr::Adjoint(_) = &lazy.repr else {
        unreachable!()
    };
}

#[test]
fn qr_lq_adjoint_dispatch_covers_unique_simple_dtypes_and_rectangles() {
    // What: adjoint QR uses an operation-local logical copy while LQ uses
    // the parent QR, preserving compact gauge, full semantics, and provider authority.
    let tall = u1_matrix_fixture([(-2, 1), (0, 3)], [(-2, 1), (0, 1)]);
    let wide = u1_matrix_fixture([(-2, 1), (0, 1)], [(-2, 1), (0, 3)]);
    assert_qr_lq_keeps_input_cache_cold(&tall);
    assert_qr_lq_keeps_input_cache_cold(&genuinely_complex(&tall));
    assert_qr_lq_keeps_input_cache_cold(&wide);
    assert_qr_lq_keeps_input_cache_cold(&genuinely_complex(&wide));

    let multitree = su2_lazy_fixture();
    assert_qr_lq_keeps_input_cache_cold(&multitree);
    assert_qr_lq_keeps_input_cache_cold(&genuinely_complex(&multitree));
}

#[test]
fn full_qr_lq_adjoint_dispatch_handles_unmatched_and_disjoint_sectors() {
    for source in [
        u1_matrix_fixture([(0, 2), (1, 1)], [(0, 3)]),
        u1_matrix_fixture([(0, 2)], [(0, 3), (1, 1)]),
    ] {
        let target = eager_adjoint_oracle(&source);
        let lazy = source.adjoint().unwrap();
        let qr = lazy.qr_full(&[0], &[1]).unwrap().pair();
        assert_qr_lq_factors(
            &source,
            &target,
            &qr,
            &target.qr_full(&[0], &[1]).unwrap().pair(),
            true,
            false,
        );
        let lq = lazy.lq_full(&[0], &[1]).unwrap().pair();
        assert_qr_lq_factors(
            &source,
            &target,
            &lq,
            &target.lq_full(&[0], &[1]).unwrap().pair(),
            false,
            false,
        );
        let TypedTensorRepr::Adjoint(_) = &lazy.repr else {
            unreachable!()
        };
    }
}

#[test]
fn qr_lq_adjoint_dispatch_handles_an_empty_homspace() {
    let source = u1_matrix_fixture([(1, 2)], [(0, 3)]);
    assert!(source.dense_data().unwrap().is_empty());
    assert_qr_lq_keeps_input_cache_cold(&source);
}

#[test]
fn qr_lq_uncached_owned_outputs_repeat_clone_and_run_concurrently() {
    let source = genuinely_complex(&su2_lazy_fixture());
    let target = eager_adjoint_oracle(&source);
    let lazy = source.adjoint().unwrap();
    let expected_qr = lazy.qr_compact(&[0], &[1, 2]).unwrap().pair();
    let expected_lq = lazy.lq_full(&[0], &[1, 2]).unwrap().pair();
    for _ in 0..2 {
        let qr = lazy.clone().qr_compact(&[0], &[1, 2]).unwrap().pair();
        let lq = lazy.clone().lq_full(&[0], &[1, 2]).unwrap().pair();
        assert_qr_lq_factors(&source, &target, &qr, &expected_qr, true, true);
        assert_qr_lq_factors(&source, &target, &lq, &expected_lq, false, false);
    }
    std::thread::scope(|scope| {
        let calls: Vec<_> = (0..4)
            .map(|_| {
                let lazy = lazy.clone();
                scope.spawn(move || {
                    let qr = lazy.qr_compact(&[0], &[1, 2]).unwrap().pair();
                    let lq = lazy.lq_full(&[0], &[1, 2]).unwrap().pair();
                    (qr, lq)
                })
            })
            .collect();
        for call in calls {
            let (qr, lq) = call.join().unwrap();
            assert_qr_lq_factors(&source, &target, &qr, &expected_qr, true, true);
            assert_qr_lq_factors(&source, &target, &lq, &expected_lq, false, false);
        }
    });
    let TypedTensorRepr::Adjoint(_) = &lazy.repr else {
        unreachable!()
    };
}

fn assert_full_qr_lq_late_failure(qr: bool) {
    let runtime = Runtime::builder()
        .with_dense_executor(Box::new(FailSecondQr::default()))
        .build()
        .unwrap();
    let provider = Arc::new(U1FusionRule);
    let leg = GradedSpace::try_new(provider, [(U1Irrep::new(0), 2), (U1Irrep::new(1), 2)]).unwrap();
    let source = TensorMap::from_subblock_fn(&runtime, [&leg], [&leg], |_, indices| {
        (indices.iter().sum::<usize>() + 1) as f64
    })
    .unwrap();
    let before = source.dense_data().unwrap().to_vec();
    let lazy = source.adjoint().unwrap();

    let result = if qr {
        lazy.qr_full(&[0], &[1]).map(drop)
    } else {
        lazy.lq_full(&[0], &[1]).map(drop)
    };
    assert!(matches!(result, Err(Error::Operation(_))));
    assert_eq!(source.dense_data().unwrap(), before);
    let TypedTensorRepr::Adjoint(_) = &lazy.repr else {
        unreachable!()
    };
}

#[test]
fn full_qr_lq_late_failure_leaves_parent_and_adjoint_cache_unchanged() {
    assert_full_qr_lq_late_failure(true);
    assert_full_qr_lq_late_failure(false);
}

fn assert_truncated_svd_reads_parent<R, D>(source: &TensorMap<R, D>)
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64> + CheckedFusionAlgebra + SectorCodec,
    D: FactorizationScalar + SpectrumMagnitude + core::fmt::Debug,
{
    // The truncated SVD is the composition `svd_compact` -> `diagview` ->
    // `find_truncated` -> `restrict_*`.
    struct Truncated<R: SectorCodec, D> {
        u: TensorMap<R, D>,
        s: TensorMap<R, D>,
        vh: TensorMap<R, D>,
        singular_values: Vec<SectorSpectrum<R::Sector, f64>>,
        error: f64,
    }
    fn truncated<R, D>(tensor: &TensorMap<R, D>, truncation: &Truncation) -> Truncated<R, D>
    where
        R: MultiplicityFreeRigidSymbols<Scalar = f64> + CheckedFusionAlgebra + SectorCodec,
        D: FactorizationScalar + SpectrumMagnitude,
    {
        let Svd { u, s, vh } = tensor.svd_compact(&[0], &[1, 2]).unwrap();
        let found = s.domain()[0]
            .find_truncated(&s.diagview().unwrap(), truncation)
            .unwrap();
        let s = s
            .restrict_leg(&[(0, &found.selection), (1, &found.selection)])
            .unwrap();
        Truncated {
            u: u.restrict_leg(&[(u.codomain_rank(), &found.selection)])
                .unwrap(),
            singular_values: s
                .diagview()
                .unwrap()
                .into_iter()
                .map(|entry| SectorSpectrum {
                    sector: entry.sector,
                    values: entry.values.iter().map(|v| v.magnitude()).collect(),
                })
                .collect(),
            s,
            vh: vh.restrict_leg(&[(0, &found.selection)]).unwrap(),
            error: found.error,
        }
    }
    let eager = eager_adjoint_oracle(source);
    let lazy = source.adjoint().unwrap();
    let truncation = Truncation::rank(1);
    let actual = truncated(&lazy, &truncation);
    let expected = truncated(&eager, &truncation);

    // The lazy route factors the parent while the oracle factors the
    // materialized adjoint, so kept sectors and counts match exactly and
    // values under the tolerance rule; every payload entry can reach a
    // singular value, so `terms` is the payload length.
    let terms = source.dense_data().unwrap().len();
    assert_eq!(actual.singular_values.len(), expected.singular_values.len());
    for (actual, expected) in actual.singular_values.iter().zip(&expected.singular_values) {
        assert_eq!(actual.sector, expected.sector);
        crate::test_numerics::numerics::assert_slices_close(
            "kept singular values",
            &actual.values,
            &expected.values,
            terms,
        );
    }
    crate::test_numerics::numerics::assert_close(
        "truncation error",
        actual.error,
        expected.error,
        terms,
    );
    let TypedTensorRepr::Adjoint(_) = &lazy.repr else {
        unreachable!()
    };
    for (actual, expected) in [
        (&actual.u, &expected.u),
        (&actual.s, &expected.s),
        (&actual.vh, &expected.vh),
    ] {
        assert_eq!(
            actual.logical_space().space(),
            expected.logical_space().space()
        );
        assert!(Arc::ptr_eq(
            actual.logical_space().provider_arc(),
            source.logical_space().provider_arc()
        ));
        assert!(actual
            .materialize()
            .unwrap()
            .dense_data()
            .unwrap()
            .iter()
            .zip(expected.materialize().unwrap().dense_data().unwrap())
            .all(|(&left, &right)| {
                (left.widen_complex() - right.widen_complex()).norm() < 1e-12
            }));
    }
    assert!(is_isometric!(actual.u, 1e-12));
}

#[test]
fn truncated_svd_reads_the_parent_without_materializing_the_adjoint() {
    // What: truncation selection, error, factor gauge, and typed provider
    // authority match an eager logical-adjoint oracle without an input copy.
    let u1 = u1_lazy_fixture();
    let su2 = su2_lazy_fixture();
    assert_truncated_svd_reads_parent(&u1);
    assert_truncated_svd_reads_parent(&genuinely_complex(&u1));
    assert_truncated_svd_reads_parent(&su2);
    assert_truncated_svd_reads_parent(&genuinely_complex(&su2));
}

#[test]
fn rejected_truncation_does_not_materialize_the_adjoint() {
    // What: the parent-native path preserves typed truncation errors
    // without publishing the logical-adjoint payload first.
    let source = u1_lazy_fixture();
    let foreign =
        GradedSpace::try_new(Arc::new(SU2FusionRule), [(SU2Irrep::from_twice_spin(0), 1)]).unwrap();
    let lazy = source.adjoint().unwrap();
    let Svd { s, .. } = lazy.svd_compact(&[0], &[1, 2]).unwrap();
    assert!(s.domain()[0]
        .find_truncated(
            &s.diagview().unwrap(),
            &Truncation::space(foreign.truncspace())
        )
        .is_err());
    let TypedTensorRepr::Adjoint(_) = &lazy.repr else {
        unreachable!()
    };
}

fn eager_adjoint_oracle<R, D>(source: &TensorMap<R, D>) -> TensorMap<R, D>
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64> + CheckedFusionAlgebra + SectorCodec,
    D: TensorScalar,
{
    let (space, data) =
        tenet_tensors::adjoint_bound_dyn(source.logical_space(), source.dense_data().unwrap())
            .unwrap();
    TensorMap {
        runtime: source.runtime.clone(),
        repr: owned_repr(TypedTensorBody::dense(space, data)),
    }
}

#[test]
fn inverse_twist_and_flip_keep_the_lazy_adjoint_materialization_boundary() {
    let source = fz2_fixture();
    let eager = eager_adjoint_oracle(&source);

    let lazy_twist = source.adjoint().unwrap();
    assert_eq!(
        lazy_twist
            .twist(&[0], Direction::Inverse)
            .unwrap()
            .materialize()
            .unwrap()
            .dense_data()
            .unwrap(),
        eager
            .twist(&[0], Direction::Inverse)
            .unwrap()
            .dense_data()
            .unwrap()
    );

    let lazy_flip = source.adjoint().unwrap();
    let actual = lazy_flip.flip(&[1], Direction::Inverse).unwrap();
    let expected = eager.flip(&[1], Direction::Inverse).unwrap();
    assert_eq!(
        actual.materialize().unwrap().dense_data().unwrap(),
        expected.dense_data().unwrap()
    );
    assert_eq!(
        actual.logical_space().space(),
        expected.logical_space().space()
    );
}

#[test]
fn simple_lazy_observers_and_owned_outputs_do_not_publish() {
    let source = u1_matrix_fixture([(0, 2)], [(0, 2)]);
    let eager = eager_adjoint_oracle(&source);
    let lazy = source.adjoint().unwrap();

    assert_eq!(lazy.is_diagonal(0.0), eager.is_diagonal(0.0));
    assert!(lazy.is_diagonal(-1.0).is_err());
    assert_eq!(
        lazy.convert::<Complex64>()
            .materialize()
            .unwrap()
            .dense_data()
            .unwrap()
            .to_vec(),
        eager
            .convert::<Complex64>()
            .materialize()
            .unwrap()
            .dense_data()
            .unwrap()
            .to_vec()
    );
    assert_eq!(
        lazy.convert::<Complex64>()
            .re()
            .materialize()
            .unwrap()
            .dense_data()
            .unwrap()
            .to_vec(),
        eager
            .convert::<Complex64>()
            .re()
            .materialize()
            .unwrap()
            .dense_data()
            .unwrap()
            .to_vec()
    );
    assert_eq!(
        lazy.convert::<Complex64>()
            .im()
            .materialize()
            .unwrap()
            .dense_data()
            .unwrap()
            .to_vec(),
        eager
            .convert::<Complex64>()
            .im()
            .materialize()
            .unwrap()
            .dense_data()
            .unwrap()
            .to_vec()
    );
    assert_eq!(
        lazy.insert_unit(0, Side::Domain, Duality::Plain)
            .unwrap()
            .materialize()
            .unwrap()
            .dense_data()
            .unwrap()
            .to_vec(),
        eager
            .insert_unit(0, Side::Domain, Duality::Plain)
            .unwrap()
            .materialize()
            .unwrap()
            .dense_data()
            .unwrap()
            .to_vec()
    );

    let scalar = source.trace_pairs(&[(0, 1)]).unwrap();
    let scalar_eager = eager_adjoint_oracle(&scalar);
    let scalar_lazy = scalar.adjoint().unwrap();
    assert_eq!(
        scalar_lazy.scalar().unwrap(),
        scalar_eager.scalar().unwrap()
    );

    let complex = genuinely_complex(&source);
    let eager_complex = eager_adjoint_oracle(&complex);
    let lazy_complex = complex.adjoint().unwrap();
    assert_eq!(
        lazy_complex
            .re()
            .materialize()
            .unwrap()
            .dense_data()
            .unwrap()
            .to_vec(),
        eager_complex
            .re()
            .materialize()
            .unwrap()
            .dense_data()
            .unwrap()
            .to_vec()
    );
    assert_eq!(
        lazy_complex
            .im()
            .materialize()
            .unwrap()
            .dense_data()
            .unwrap()
            .to_vec(),
        eager_complex
            .im()
            .materialize()
            .unwrap()
            .dense_data()
            .unwrap()
            .to_vec()
    );

    let clone = lazy.clone();
    assert_eq!(
        lazy.materialize().unwrap().dense_data().unwrap().to_vec(),
        eager.materialize().unwrap().dense_data().unwrap().to_vec()
    );
    assert_eq!(
        clone.materialize().unwrap().dense_data().unwrap().to_vec(),
        eager.materialize().unwrap().dense_data().unwrap().to_vec()
    );
    // Explicit materialization is not an implicit build.
}

#[test]
fn lazy_otimes_orientations_and_deligne_inputs_stay_cold() {
    let lhs = u1_lazy_fixture();
    let rhs = lhs.scale(2.0);
    let eager_lhs = eager_adjoint_oracle(&lhs);
    let eager_rhs = eager_adjoint_oracle(&rhs);
    let lazy_lhs = lhs.adjoint().unwrap();
    let lazy_rhs = rhs.adjoint().unwrap();

    for (actual, expected) in [
        (lhs.otimes(&rhs).unwrap(), lhs.otimes(&rhs).unwrap()),
        (
            lhs.otimes(&lazy_rhs).unwrap(),
            lhs.otimes(&eager_rhs).unwrap(),
        ),
        (
            lazy_lhs.otimes(&rhs).unwrap(),
            eager_lhs.otimes(&rhs).unwrap(),
        ),
        (
            lazy_lhs.otimes(&lazy_rhs).unwrap(),
            eager_lhs.otimes(&eager_rhs).unwrap(),
        ),
    ] {
        assert_eq!(actual.dense_data().unwrap(), expected.dense_data().unwrap());
        assert!(Arc::ptr_eq(
            actual.logical_space().provider_arc(),
            lhs.logical_space().provider_arc()
        ));
    }

    let deligne_lhs = u1_matrix_fixture([(0, 2)], [(0, 2)]);
    let deligne_rhs = deligne_lhs.scale(3.0);
    let eager_deligne_lhs = eager_adjoint_oracle(&deligne_lhs);
    let eager_deligne_rhs = eager_adjoint_oracle(&deligne_rhs);
    let lazy_deligne_lhs = deligne_lhs.adjoint().unwrap();
    let lazy_deligne_rhs = deligne_rhs.adjoint().unwrap();
    let product = Arc::new(U1FusionRule.product(U1FusionRule));
    let actual = lazy_deligne_lhs
        .deligne_product(&lazy_deligne_rhs, Arc::clone(&product))
        .unwrap();
    let expected = eager_deligne_lhs
        .deligne_product(&eager_deligne_rhs, product)
        .unwrap();
    assert_eq!(actual.dense_data().unwrap(), expected.dense_data().unwrap());
}

#[test]
fn absorb_uses_operation_local_logical_payloads_without_warming_lazy_inputs() {
    let destination_parent = genuinely_complex(&u1_lazy_fixture());
    let source_parent = destination_parent.scale(num_complex::Complex64::new(2.0, -1.0));
    let eager_destination = eager_adjoint_oracle(&destination_parent);
    let eager_source = eager_adjoint_oracle(&source_parent);
    let expected = eager_destination.absorb(&eager_source).unwrap();
    let lazy_destination = destination_parent.adjoint().unwrap();
    let lazy_source = source_parent.adjoint().unwrap();

    let actual = lazy_destination.absorb(&lazy_source).unwrap();
    assert_typed_map_close(&actual, &expected, 1e-12);
}

fn assert_parent_native_transform<R, D>(
    source: &TensorMap<R, D>,
    operation: impl Fn(&TensorMap<R, D>) -> Result<TensorMap<R, D>, Error>,
) where
    R: MultiplicityFreeRigidSymbols<Scalar = f64> + CheckedFusionAlgebra + SectorCodec,
    D: TensorScalar + core::fmt::Debug,
{
    let lazy = source.adjoint().unwrap();
    let eager = eager_adjoint_oracle(source);
    let actual = operation(&lazy).unwrap();
    let expected = operation(&eager).unwrap();
    let TypedTensorRepr::Adjoint(_) = &actual.repr else {
        panic!("a transformed lazy adjoint must remain parent-backed");
    };
    assert_eq!(
        actual.logical_space().space(),
        expected.logical_space().space()
    );
    assert!(Arc::ptr_eq(
        actual.logical_space().provider_arc(),
        source.logical_space().provider_arc()
    ));
    assert_eq!(
        actual.materialize().unwrap().dense_data().unwrap().len(),
        expected.dense_data().unwrap().len()
    );
    assert!(actual
        .materialize()
        .unwrap()
        .dense_data()
        .unwrap()
        .iter()
        .zip(expected.dense_data().unwrap())
        .all(|(&actual, &expected)| {
            (actual.widen_complex() - expected.widen_complex()).norm() < 1e-12
        }));
}

fn assert_parent_native_transform_suite<R, D>(source: &TensorMap<R, D>)
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64> + CheckedFusionAlgebra + SectorCodec,
    D: TensorScalar + core::fmt::Debug,
{
    assert_parent_native_transform(source, |tensor| tensor.permute(&[2, 0], &[1]));
    assert_parent_native_transform(source, |tensor| tensor.braid(&[2, 0], &[1], &[17, 3, 11]));
    assert_parent_native_transform(source, |tensor| tensor.transpose(&[2, 1], &[0]));
    assert_parent_native_transform(source, |tensor| tensor.repartition(2));
}

#[test]
fn nonidentity_adjoint_transforms_stay_parent_native_for_unique_simple_and_both_dtypes() {
    let u1 = u1_lazy_fixture();
    let su2 = su2_lazy_fixture();
    let u1_c64 = u1
        .convert::<Complex64>()
        .scale(num_complex::Complex64::new(1.0, 2.0));
    let su2_c64 = su2
        .convert::<Complex64>()
        .scale(num_complex::Complex64::new(1.0, 2.0));
    assert_parent_native_transform_suite(&u1);
    assert_parent_native_transform_suite(&u1_c64);
    assert_parent_native_transform_suite(&su2);
    assert_parent_native_transform_suite(&su2_c64);
}

fn assert_close<D: TensorScalar>(actual: D, expected: D) {
    assert!((actual.widen_complex() - expected.widen_complex()).norm() < 1e-12);
}

fn assert_parent_native_elementwise<R, D>(source: &TensorMap<R, D>, alpha: D, beta: D)
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64> + CheckedFusionAlgebra + SectorCodec,
    D: TensorScalar + core::fmt::Debug,
{
    let lazy = source.adjoint().unwrap();
    let eager = eager_adjoint_oracle(source);

    let add = lazy.axpby(alpha, &eager, beta).unwrap();
    let expected_add = eager.axpby(alpha, &eager, beta).unwrap();
    assert!(add
        .dense_data()
        .unwrap()
        .iter()
        .zip(expected_add.dense_data().unwrap())
        .all(|(&actual, &expected)| {
            (actual.widen_complex() - expected.widen_complex()).norm() < 1e-12
        }));
    let add_both = lazy.axpby(alpha, &lazy, beta).unwrap();
    assert!(add_both
        .dense_data()
        .unwrap()
        .iter()
        .zip(expected_add.dense_data().unwrap())
        .all(|(&actual, &expected)| {
            (actual.widen_complex() - expected.widen_complex()).norm() < 1e-12
        }));
    let add_rhs = eager.axpby(alpha, &lazy, beta).unwrap();
    assert!(add_rhs
        .dense_data()
        .unwrap()
        .iter()
        .zip(expected_add.dense_data().unwrap())
        .all(|(&actual, &expected)| {
            (actual.widen_complex() - expected.widen_complex()).norm() < 1e-12
        }));

    let scaled = lazy.scale(alpha);
    let expected_scaled = eager.scale(alpha);
    let TypedTensorRepr::Adjoint(_) = &scaled.repr else {
        panic!("scaling a lazy adjoint must remain parent-backed");
    };
    assert!(scaled
        .materialize()
        .unwrap()
        .dense_data()
        .unwrap()
        .iter()
        .zip(expected_scaled.dense_data().unwrap())
        .all(|(&actual, &expected)| {
            (actual.widen_complex() - expected.widen_complex()).norm() < 1e-12
        }));

    assert_close(lazy.inner(&eager).unwrap(), eager.inner(&eager).unwrap());
    assert_close(eager.inner(&lazy).unwrap(), eager.inner(&eager).unwrap());
    assert_close(lazy.inner(&lazy).unwrap(), eager.inner(&eager).unwrap());
    assert!((lazy.norm(2.0).unwrap() - eager.norm(2.0).unwrap()).abs() < 1e-12);
    assert!((lazy.norm(f64::INFINITY).unwrap() - eager.norm(f64::INFINITY).unwrap()).abs() < 1e-12);
    assert!((lazy.norm(1.5).unwrap() - eager.norm(1.5).unwrap()).abs() < 1e-12);

    let normalized = lazy.scale(D::from_real(1.0 / lazy.norm(2.0).unwrap()));
    let TypedTensorRepr::Adjoint(_) = &normalized.repr else {
        panic!("normalizing a lazy adjoint must remain parent-backed");
    };
    assert!((normalized.norm(2.0).unwrap() - 1.0).abs() < 1e-12);
}

#[test]
fn adjoint_elementwise_and_reductions_stay_parent_native() {
    let u1 = u1_lazy_fixture();
    let su2 = su2_lazy_fixture();
    let u1_c64 = u1
        .convert::<Complex64>()
        .scale(num_complex::Complex64::new(1.0, 2.0));
    let su2_c64 = su2
        .convert::<Complex64>()
        .scale(num_complex::Complex64::new(1.0, 2.0));
    assert_parent_native_elementwise(&u1, 2.0, -0.5);
    assert_parent_native_elementwise(&su2, 2.0, -0.5);
    assert_parent_native_elementwise(
        &u1_c64,
        num_complex::Complex64::new(0.5, 1.0),
        num_complex::Complex64::new(-0.25, 0.75),
    );
    assert_parent_native_elementwise(
        &su2_c64,
        num_complex::Complex64::new(0.5, 1.0),
        num_complex::Complex64::new(-0.25, 0.75),
    );
}

fn assert_parent_native_trace_pairs<R, D>(source: &TensorMap<R, D>)
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64> + CheckedFusionAlgebra + SectorCodec,
    D: TensorScalar + core::fmt::Debug,
{
    let lazy = source.adjoint().unwrap();
    let eager = eager_adjoint_oracle(source);
    let expected = eager.trace_pairs(&[(0, 1)]).unwrap();
    let actual = lazy.trace_pairs(&[(0, 1)]).unwrap();
    assert_eq!(
        actual.logical_space().space(),
        expected.logical_space().space()
    );
    assert!(Arc::ptr_eq(
        actual.logical_space().provider_arc(),
        source.logical_space().provider_arc()
    ));
    assert!(actual
        .dense_data()
        .unwrap()
        .iter()
        .zip(expected.dense_data().unwrap())
        .all(|(&actual, &expected)| {
            (actual.widen_complex() - expected.widen_complex()).norm() < 1e-12
        }));
    assert!(lazy.trace_pairs(&[(0, 3)]).is_err());
    assert!(lazy.trace_pairs(&[(0, 0)]).is_err());
}

#[test]
fn adjoint_trace_pairs_stays_parent_native() {
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let u1_provider = Arc::new(U1FusionRule);
    let u1_traced = GradedSpace::try_new(
        Arc::clone(&u1_provider),
        [(U1Irrep::new(0), 2), (U1Irrep::new(1), 1)],
    )
    .unwrap();
    let u1_open = GradedSpace::try_new(u1_provider, [(U1Irrep::new(-1), 1), (U1Irrep::new(0), 1)])
        .and_then(|space| space.try_dual())
        .unwrap();
    let u1 = TensorMap::from_subblock_fn(
        &runtime,
        [&u1_traced, &u1_open],
        [&u1_traced],
        |_, indices| indices.iter().sum::<usize>() as f64 + 1.0,
    )
    .unwrap();
    let su2_provider = Arc::new(SU2FusionRule);
    let su2_traced = GradedSpace::try_new(
        Arc::clone(&su2_provider),
        [
            (SU2Irrep::from_twice_spin(0), 2),
            (SU2Irrep::from_twice_spin(1), 1),
        ],
    )
    .unwrap();
    let su2_open = GradedSpace::try_new(
        su2_provider,
        [
            (SU2Irrep::from_twice_spin(0), 1),
            (SU2Irrep::from_twice_spin(1), 1),
        ],
    )
    .and_then(|space| space.try_dual())
    .unwrap();
    let su2 = TensorMap::from_subblock_fn(
        &runtime,
        [&su2_traced, &su2_open, &su2_open],
        [&su2_traced],
        |_, indices| indices.iter().sum::<usize>() as f64 + 1.0,
    )
    .unwrap();
    assert!(su2.subblock_count() > 1);
    assert!(
        eager_adjoint_oracle(&su2)
            .trace_pairs(&[(0, 1)])
            .unwrap()
            .subblock_count()
            > 1
    );
    let u1_c64 = u1
        .convert::<Complex64>()
        .scale(num_complex::Complex64::new(1.0, 2.0));
    let su2_c64 = su2
        .convert::<Complex64>()
        .scale(num_complex::Complex64::new(1.0, 2.0));
    assert_parent_native_trace_pairs(&u1);
    assert_parent_native_trace_pairs(&u1_c64);
    assert_parent_native_trace_pairs(&su2);
    assert_parent_native_trace_pairs(&su2_c64);
}

fn assert_parent_native_tr<R, D>(source: &TensorMap<R, D>)
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64> + CheckedFusionAlgebra + SectorCodec,
    D: TensorScalar + core::fmt::Debug,
{
    let lazy = source.adjoint().unwrap();
    let eager = eager_adjoint_oracle(source);
    assert_close(lazy.tr().unwrap(), eager.tr().unwrap());
}

#[test]
fn adjoint_positive_trace_conjugates_the_parent_without_materializing() {
    let u1_source = u1_lazy_fixture();
    let u1_leg = u1_source.codomain().remove(0);
    let u1 =
        TensorMap::from_subblock_fn(u1_source.runtime(), [&u1_leg], [&u1_leg], |_, indices| {
            (indices[0] + 2 * indices[1]) as f64 + 1.0
        })
        .unwrap();
    let su2_source = su2_lazy_fixture();
    let su2_leg = su2_source.codomain().remove(0);
    let su2 = TensorMap::from_subblock_fn(
        su2_source.runtime(),
        [&su2_leg],
        [&su2_leg],
        |_, indices| (indices[0] + 2 * indices[1]) as f64 + 1.0,
    )
    .unwrap();
    let u1_c64 = u1
        .convert::<Complex64>()
        .scale(num_complex::Complex64::new(1.0, 2.0));
    let su2_c64 = su2
        .convert::<Complex64>()
        .scale(num_complex::Complex64::new(1.0, 2.0));
    assert_parent_native_tr(&u1);
    assert_parent_native_tr(&u1_c64);
    assert_parent_native_tr(&su2);
    assert_parent_native_tr(&su2_c64);
}

fn assert_parent_native_contract_and_compose<R, D>(source: &TensorMap<R, D>)
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64> + CheckedFusionAlgebra + SectorCodec,
    D: TensorScalar + core::fmt::Debug,
{
    let lazy = source.adjoint().unwrap();
    let eager = eager_adjoint_oracle(source);
    for (lhs, rhs, expected_lhs, expected_rhs) in [
        (&lazy, &eager, &eager, &eager),
        (&eager, &lazy, &eager, &eager),
        (&lazy, &lazy, &eager, &eager),
    ] {
        let actual = lhs
            .contract(
                rhs,
                &ContractSpec {
                    lhs: &[1],
                    rhs: &[0],
                    codomain: &[1],
                    domain: &[0],
                },
            )
            .unwrap();
        let expected = expected_lhs
            .contract(
                expected_rhs,
                &ContractSpec {
                    lhs: &[1],
                    rhs: &[0],
                    codomain: &[1],
                    domain: &[0],
                },
            )
            .unwrap();
        assert_eq!(
            actual.logical_space().space(),
            expected.logical_space().space()
        );
        assert!(Arc::ptr_eq(
            actual.logical_space().provider_arc(),
            lhs.logical_space().provider_arc()
        ));
        assert!(actual
            .dense_data()
            .unwrap()
            .iter()
            .zip(expected.dense_data().unwrap())
            .all(|(&actual, &expected)| {
                (actual.widen_complex() - expected.widen_complex()).norm() < 1e-12
            }));

        let actual = lhs.compose(rhs).unwrap();
        let expected = expected_lhs.compose(expected_rhs).unwrap();
        assert_eq!(
            actual.logical_space().space(),
            expected.logical_space().space()
        );
        assert!(Arc::ptr_eq(
            actual.logical_space().provider_arc(),
            lhs.logical_space().provider_arc()
        ));
        assert!(actual
            .dense_data()
            .unwrap()
            .iter()
            .zip(expected.dense_data().unwrap())
            .all(|(&actual, &expected)| {
                (actual.widen_complex() - expected.widen_complex()).norm() < 1e-12
            }));
    }
    assert!(lazy
        .contract(
            &eager,
            &ContractSpec {
                lhs: &[2],
                rhs: &[0],
                codomain: &[0],
                domain: &[1]
            }
        )
        .is_err());
}

#[test]
fn adjoint_contract_and_compose_stay_parent_native() {
    let u1_source = u1_lazy_fixture();
    let u1_leg = u1_source.codomain().remove(0);
    let u1 =
        TensorMap::from_subblock_fn(u1_source.runtime(), [&u1_leg], [&u1_leg], |_, indices| {
            (indices[0] + 2 * indices[1]) as f64 + 1.0
        })
        .unwrap();
    let su2_source = su2_lazy_fixture();
    let su2_leg = su2_source.codomain().remove(0);
    let su2 = TensorMap::from_subblock_fn(
        su2_source.runtime(),
        [&su2_leg],
        [&su2_leg],
        |_, indices| (indices[0] + 2 * indices[1]) as f64 + 1.0,
    )
    .unwrap();
    let u1_c64 = u1
        .convert::<Complex64>()
        .scale(num_complex::Complex64::new(1.0, 2.0));
    let su2_c64 = su2
        .convert::<Complex64>()
        .scale(num_complex::Complex64::new(1.0, 2.0));
    assert_parent_native_contract_and_compose(&u1);
    assert_parent_native_contract_and_compose(&u1_c64);
    assert_parent_native_contract_and_compose(&su2);
    assert_parent_native_contract_and_compose(&su2_c64);
}

fn assert_rank_three_su2_contract_and_compose<D>(source: &TensorMap<SU2FusionRule, D>)
where
    D: TensorScalar + core::fmt::Debug,
{
    assert!(source.subblock_count() > 1);
    let lazy = source.adjoint().unwrap();
    let eager = eager_adjoint_oracle(source);

    let actual = lazy
        .contract(
            source,
            &ContractSpec {
                lhs: &[2, 1],
                rhs: &[1, 0],
                codomain: &[1],
                domain: &[0],
            },
        )
        .unwrap();
    let expected = eager
        .contract(
            source,
            &ContractSpec {
                lhs: &[2, 1],
                rhs: &[1, 0],
                codomain: &[1],
                domain: &[0],
            },
        )
        .unwrap();
    assert!(actual.subblock_count() > 1);
    assert_eq!(
        actual.logical_space().space(),
        expected.logical_space().space()
    );
    assert!(Arc::ptr_eq(
        actual.logical_space().provider_arc(),
        source.logical_space().provider_arc()
    ));
    assert!(actual
        .dense_data()
        .unwrap()
        .iter()
        .zip(expected.dense_data().unwrap())
        .all(|(&actual, &expected)| {
            (actual.widen_complex() - expected.widen_complex()).norm() < 1e-12
        }));

    let actual = lazy.compose(source).unwrap();
    let expected = eager.compose(source).unwrap();
    assert!(actual.subblock_count() > 1);
    assert_eq!(
        actual.logical_space().space(),
        expected.logical_space().space()
    );
    assert!(Arc::ptr_eq(
        actual.logical_space().provider_arc(),
        source.logical_space().provider_arc()
    ));
    assert!(actual
        .dense_data()
        .unwrap()
        .iter()
        .zip(expected.dense_data().unwrap())
        .all(|(&actual, &expected)| {
            (actual.widen_complex() - expected.widen_complex()).norm() < 1e-12
        }));
}

#[test]
fn rank_three_su2_adjoint_contract_and_compose_use_oriented_recoupling() {
    let source = su2_lazy_fixture();
    let complex = source
        .convert::<Complex64>()
        .scale(num_complex::Complex64::new(1.0, 2.0));
    assert_rank_three_su2_contract_and_compose(&source);
    assert_rank_three_su2_contract_and_compose(&complex);
}

fn assert_fermionic_contract_and_compose_semantics<D>(
    source: &TensorMap<FermionParityFusionRule, D>,
) where
    D: TensorScalar + core::fmt::Debug,
{
    let lazy = source.adjoint().unwrap();
    let eager = eager_adjoint_oracle(source);
    let contract = lazy
        .contract(
            source,
            &ContractSpec {
                lhs: &[1],
                rhs: &[0],
                codomain: &[0],
                domain: &[1],
            },
        )
        .unwrap();
    let expected_contract = eager
        .contract(
            source,
            &ContractSpec {
                lhs: &[1],
                rhs: &[0],
                codomain: &[0],
                domain: &[1],
            },
        )
        .unwrap();
    let compose = lazy.compose(source).unwrap();
    let expected_compose = eager.compose(source).unwrap();
    assert!(contract
        .dense_data()
        .unwrap()
        .iter()
        .zip(expected_contract.dense_data().unwrap())
        .all(|(&actual, &expected)| {
            (actual.widen_complex() - expected.widen_complex()).norm() < 1e-12
        }));
    assert!(compose
        .dense_data()
        .unwrap()
        .iter()
        .zip(expected_compose.dense_data().unwrap())
        .all(|(&actual, &expected)| {
            (actual.widen_complex() - expected.widen_complex()).norm() < 1e-12
        }));
    assert!(contract
        .dense_data()
        .unwrap()
        .iter()
        .zip(compose.dense_data().unwrap())
        .any(|(&contract, &compose)| {
            (contract.widen_complex() - compose.widen_complex()).norm() > 1e-12
        }));
    assert!(Arc::ptr_eq(
        contract.logical_space().provider_arc(),
        source.logical_space().provider_arc()
    ));
    assert!(Arc::ptr_eq(
        compose.logical_space().provider_arc(),
        source.logical_space().provider_arc()
    ));
}

#[test]
fn fermionic_lazy_contract_keeps_the_supertrace_distinct_from_compose() {
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let provider = Arc::new(FermionParityFusionRule);
    let leg = GradedSpace::try_new(provider, [(Z2Irrep::EVEN, 2), (Z2Irrep::ODD, 2)])
        .and_then(|space| space.try_dual())
        .unwrap();
    let source = TensorMap::from_subblock_fn(&runtime, [&leg], [&leg], |trees, indices| {
        let sector = trees.codomain_uncoupled[0];
        let parity_weight = if sector == Z2Irrep::ODD { 3.0 } else { 1.0 };
        parity_weight * (indices[0] + 2 * indices[1] + 1) as f64
    })
    .unwrap();
    let complex = source
        .convert::<Complex64>()
        .scale(num_complex::Complex64::new(1.0, 2.0));
    assert_fermionic_contract_and_compose_semantics(&source);
    assert_fermionic_contract_and_compose_semantics(&complex);
}

/// A rank-(1, 1) domain leg against a rank-(1, 1) codomain leg, default split.
const RANK_TWO_COMPOSE: ContractSpec<'static> = ContractSpec {
    lhs: &[1],
    rhs: &[0],
    codomain: &[0],
    domain: &[1],
};

fn assert_same_error(actual: Error, expected: Error) {
    assert_eq!(
        core::mem::discriminant(&actual),
        core::mem::discriminant(&expected)
    );
    assert_eq!(actual.to_string(), expected.to_string());
}

#[test]
fn lazy_contract_preserves_validation_precedence_without_materializing() {
    let source = {
        let fixture = u1_lazy_fixture();
        let leg = fixture.codomain().remove(0);
        TensorMap::from_subblock_fn(fixture.runtime(), [&leg], [&leg], |_, indices| {
            (indices[0] + 2 * indices[1] + 1) as f64
        })
        .unwrap()
    };
    let eager = eager_adjoint_oracle(&source);
    for spec in [
        ContractSpec {
            lhs: &[2],
            rhs: &[0],
            codomain: &[0],
            domain: &[1],
        },
        ContractSpec {
            lhs: &[1, 1],
            rhs: &[0, 0],
            codomain: &[],
            domain: &[],
        },
    ] {
        let lazy = source.adjoint().unwrap();
        assert_same_error(
            lazy.contract(&eager, &spec).unwrap_err(),
            eager.contract(&eager, &spec).unwrap_err(),
        );
    }

    let bad_leg = GradedSpace::try_new(
        Arc::clone(source.logical_space().provider_arc()),
        [(U1Irrep::new(7), 1)],
    )
    .unwrap();
    let bad =
        TensorMap::from_subblock_fn(source.runtime(), [&bad_leg], [&bad_leg], |_, _| 1.0).unwrap();
    let lazy = source.adjoint().unwrap();
    assert_same_error(
        lazy.contract(
            &bad,
            &ContractSpec {
                lhs: &[1],
                rhs: &[0],
                codomain: &[0],
                domain: &[0],
            },
        )
        .unwrap_err(),
        eager
            .contract(
                &bad,
                &ContractSpec {
                    lhs: &[1],
                    rhs: &[0],
                    codomain: &[0],
                    domain: &[0],
                },
            )
            .unwrap_err(),
    );

    let other_runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let source_leg = source.codomain().remove(0);
    let other =
        TensorMap::from_subblock_fn(&other_runtime, [&source_leg], [&source_leg], |_, _| 1.0)
            .unwrap();
    let lazy = source.adjoint().unwrap();
    assert_same_error(
        lazy.contract(
            &other,
            &ContractSpec {
                lhs: &[1],
                rhs: &[0],
                codomain: &[0],
                domain: &[1],
            },
        )
        .unwrap_err(),
        eager
            .contract(
                &other,
                &ContractSpec {
                    lhs: &[1],
                    rhs: &[0],
                    codomain: &[0],
                    domain: &[1],
                },
            )
            .unwrap_err(),
    );

    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let z2 = Arc::new(ZNFusionRule::new(2).unwrap());
    let z3 = Arc::new(ZNFusionRule::new(3).unwrap());
    let z2_leg = GradedSpace::try_new(Arc::clone(&z2), [(z2.irrep(0), 2)]).unwrap();
    let z3_leg = GradedSpace::try_new(Arc::clone(&z3), [(z3.irrep(0), 2)]).unwrap();
    let z2_tensor =
        TensorMap::from_subblock_fn(&runtime, [&z2_leg], [&z2_leg], |_, _| 1.0).unwrap();
    let z3_tensor =
        TensorMap::from_subblock_fn(&runtime, [&z3_leg], [&z3_leg], |_, _| 1.0).unwrap();
    let eager = eager_adjoint_oracle(&z2_tensor);
    let lazy = z2_tensor.adjoint().unwrap();
    assert_same_error(
        lazy.contract(&z3_tensor, &RANK_TWO_COMPOSE).unwrap_err(),
        eager.contract(&z3_tensor, &RANK_TWO_COMPOSE).unwrap_err(),
    );
}

#[test]
fn lazy_binary_outputs_keep_the_lhs_provider_allocation() {
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let lhs_provider = Arc::new(U1FusionRule);
    let rhs_provider = Arc::new(U1FusionRule);
    assert!(!Arc::ptr_eq(&lhs_provider, &rhs_provider));
    let lhs_leg = GradedSpace::try_new(
        Arc::clone(&lhs_provider),
        [(U1Irrep::new(0), 2), (U1Irrep::new(1), 1)],
    )
    .unwrap();
    let rhs_leg = GradedSpace::try_new(
        Arc::clone(&rhs_provider),
        [(U1Irrep::new(0), 2), (U1Irrep::new(1), 1)],
    )
    .unwrap();
    let lhs = TensorMap::from_subblock_fn(&runtime, [&lhs_leg], [&lhs_leg], |_, indices| {
        (indices[0] + 2 * indices[1] + 1) as f64
    })
    .unwrap();
    let rhs = TensorMap::from_subblock_fn(&runtime, [&rhs_leg], [&rhs_leg], |_, indices| {
        (2 * indices[0] + indices[1] + 1) as f64
    })
    .unwrap();
    let lazy = lhs.adjoint().unwrap();
    let eager = eager_adjoint_oracle(&lhs);
    let rhs_lazy = rhs.adjoint().unwrap();
    let rhs_eager = eager_adjoint_oracle(&rhs);
    for (actual, expected) in [
        (
            lazy.contract(
                &rhs_lazy,
                &ContractSpec {
                    lhs: &[1],
                    rhs: &[0],
                    codomain: &[0],
                    domain: &[1],
                },
            )
            .unwrap(),
            eager.contract(&rhs_eager, &RANK_TWO_COMPOSE).unwrap(),
        ),
        (
            lazy.compose(&rhs_lazy).unwrap(),
            eager.compose(&rhs_eager).unwrap(),
        ),
    ] {
        assert_eq!(actual.dense_data().unwrap(), expected.dense_data().unwrap());
        assert!(Arc::ptr_eq(
            actual.logical_space().provider_arc(),
            &lhs_provider
        ));
        assert!(!Arc::ptr_eq(
            actual.logical_space().provider_arc(),
            &rhs_provider
        ));
    }
}

#[test]
fn mixed_compact_add_does_not_materialize_the_lazy_operand() {
    DIAGONAL_MATERIALIZATIONS.set(0);
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let provider = Arc::new(U1FusionRule);
    let bond = GradedSpace::try_new(provider, [(U1Irrep::new(0), 2)]).unwrap();
    let dense = TensorMap::from_subblock_fn(&runtime, [&bond], [&bond], |_, indices| {
        (indices[0] + 2 * indices[1]) as f64
    })
    .unwrap();
    let lazy = dense.adjoint().unwrap();
    let eager = eager_adjoint_oracle(&dense);
    let diagonal = TensorMap::diagonal(
        &runtime,
        &bond,
        [SectorSpectrum {
            sector: U1Irrep::new(0),
            values: vec![2.0, 3.0],
        }],
    )
    .unwrap();
    let actual = diagonal.axpby(0.5, &lazy, -2.0).unwrap();
    let expected = diagonal.axpby(0.5, &eager, -2.0).unwrap();
    assert_eq!(actual.dense_data().unwrap(), expected.dense_data().unwrap());
    let reverse = lazy.axpby(-2.0, &diagonal, 0.5).unwrap();
    let expected_reverse = eager.axpby(-2.0, &diagonal, 0.5).unwrap();
    assert_eq!(
        reverse.dense_data().unwrap(),
        expected_reverse.dense_data().unwrap()
    );
    assert_eq!(DIAGONAL_MATERIALIZATIONS.get(), 0);

    for (lhs, rhs, eager_lhs, eager_rhs) in [
        (&diagonal, &lazy, &diagonal, &eager),
        (&lazy, &diagonal, &eager, &diagonal),
    ] {
        let actual = lhs.compose(rhs).unwrap();
        let expected = eager_lhs.compose(eager_rhs).unwrap();
        assert_eq!(actual.dense_data().unwrap(), expected.dense_data().unwrap());
        let actual = lhs
            .contract(
                rhs,
                &ContractSpec {
                    lhs: &[1],
                    rhs: &[0],
                    codomain: &[0],
                    domain: &[1],
                },
            )
            .unwrap();
        let expected = eager_lhs
            .contract(
                eager_rhs,
                &ContractSpec {
                    lhs: &[1],
                    rhs: &[0],
                    codomain: &[0],
                    domain: &[1],
                },
            )
            .unwrap();
        assert_eq!(actual.dense_data().unwrap(), expected.dense_data().unwrap());
    }
    assert_eq!(DIAGONAL_MATERIALIZATIONS.get(), 0);

    assert_close(
        diagonal.inner(&lazy).unwrap(),
        diagonal.inner(&eager).unwrap(),
    );
    assert_close(
        lazy.inner(&diagonal).unwrap(),
        eager.inner(&diagonal).unwrap(),
    );
    assert_eq!(DIAGONAL_MATERIALIZATIONS.get(), 0);
}

#[test]
fn mixed_compact_dense_inner_does_not_materialize_the_diagonal() {
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let provider = Arc::new(U1FusionRule);
    let bond = GradedSpace::try_new(provider, [(U1Irrep::new(0), 2)]).unwrap();
    let dense = TensorMap::from_subblock_fn(&runtime, [&bond], [&bond], |_, indices| {
        match (indices[0], indices[1]) {
            (0, 0) => Complex64::new(1.0, 2.0),
            (1, 1) => Complex64::new(4.0, -1.0),
            _ => Complex64::new(6.0, 7.0),
        }
    })
    .unwrap();
    let lazy = dense.adjoint().unwrap();
    let diagonal = TensorMap::diagonal(
        &runtime,
        &bond,
        [SectorSpectrum {
            sector: U1Irrep::new(0),
            values: vec![Complex64::new(2.0, 3.0), Complex64::new(-1.0, 0.5)],
        }],
    )
    .unwrap();

    DIAGONAL_MATERIALIZATIONS.set(0);
    assert_close(diagonal.inner(&dense).unwrap(), Complex64::new(3.5, 0.0));
    assert_close(dense.inner(&diagonal).unwrap(), Complex64::new(3.5, 0.0));
    assert_close(diagonal.inner(&lazy).unwrap(), Complex64::new(-7.5, -10.0));
    assert_close(lazy.inner(&diagonal).unwrap(), Complex64::new(-7.5, 10.0));
    assert_eq!(DIAGONAL_MATERIALIZATIONS.get(), 0);
}

#[test]
fn mixed_compact_dense_inner_weights_sectors_and_skips_structural_zeros() {
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let spin0 = SU2Irrep::from_twice_spin(0);
    let spin_half = SU2Irrep::from_twice_spin(1);
    let bond = GradedSpace::try_new(Arc::new(SU2FusionRule), [(spin0, 2), (spin_half, 1)]).unwrap();
    let diagonal = TensorMap::diagonal(
        &runtime,
        &bond,
        [
            SectorSpectrum {
                sector: spin0,
                values: vec![1.0, 2.0],
            },
            SectorSpectrum {
                sector: spin_half,
                values: vec![3.0],
            },
        ],
    )
    .unwrap();
    let dense = TensorMap::from_subblock_fn(&runtime, [&bond], [&bond], |trees, indices| {
        if indices[0] != indices[1] {
            if indices[0] == 0 {
                f64::NAN
            } else {
                f64::INFINITY
            }
        } else if *trees.coupled() == spin0 {
            [4.0, 5.0][indices[0]]
        } else {
            6.0
        }
    })
    .unwrap();
    let lazy = dense.adjoint().unwrap();

    DIAGONAL_MATERIALIZATIONS.set(0);
    for dense in [&dense, &lazy] {
        assert_eq!(diagonal.inner(dense).unwrap(), 50.0);
        assert_eq!(dense.inner(&diagonal).unwrap(), 50.0);
    }
    assert_eq!(DIAGONAL_MATERIALIZATIONS.get(), 0);

    let nonfinite_diagonal =
        TensorMap::from_subblock_fn(&runtime, [&bond], [&bond], |trees, indices| {
            if indices == [0, 0] && *trees.coupled() == spin0 {
                f64::NAN
            } else {
                1.0
            }
        })
        .unwrap();
    assert!(diagonal.inner(&nonfinite_diagonal).unwrap().is_nan());
    assert!(nonfinite_diagonal.inner(&diagonal).unwrap().is_nan());

    let infinite_diagonal =
        TensorMap::from_subblock_fn(&runtime, [&bond], [&bond], |trees, indices| {
            if indices == [0, 0] && *trees.coupled() == spin0 {
                f64::INFINITY
            } else {
                1.0
            }
        })
        .unwrap();
    assert!(diagonal.inner(&infinite_diagonal).unwrap().is_infinite());
    assert!(infinite_diagonal.inner(&diagonal).unwrap().is_infinite());
}

#[test]
fn mixed_compact_dense_inner_accepts_an_empty_bond() {
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let bond = GradedSpace::try_new(Arc::new(U1FusionRule), []).unwrap();
    let diagonal: TensorMap<_, f64> =
        TensorMap::diagonal(&runtime, &bond, Vec::<SectorSpectrum<_, f64>>::new()).unwrap();
    let dense: TensorMap<_, f64> =
        TensorMap::from_subblock_fn(&runtime, [&bond], [&bond], |_, _| unreachable!()).unwrap();

    DIAGONAL_MATERIALIZATIONS.set(0);
    assert_eq!(diagonal.inner(&dense).unwrap(), 0.0);
    assert_eq!(dense.inner(&diagonal).unwrap(), 0.0);
    assert_eq!(DIAGONAL_MATERIALIZATIONS.get(), 0);
}

#[test]
fn mixed_compact_lazy_inner_maps_non_self_dual_parent_blocks() {
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let bond = GradedSpace::try_new(
        Arc::new(U1FusionRule),
        [
            (U1Irrep::new(-1), 1),
            (U1Irrep::new(0), 2),
            (U1Irrep::new(2), 1),
        ],
    )
    .unwrap();
    let diagonal = TensorMap::diagonal(
        &runtime,
        &bond,
        [
            SectorSpectrum {
                sector: U1Irrep::new(2),
                values: vec![6.0],
            },
            SectorSpectrum {
                sector: U1Irrep::new(-1),
                values: vec![5.0],
            },
            SectorSpectrum {
                sector: U1Irrep::new(0),
                values: vec![3.0, 4.0],
            },
        ],
    )
    .unwrap();
    let dense = TensorMap::from_subblock_fn(&runtime, [&bond], [&bond], |trees, indices| {
        if indices[0] != indices[1] {
            -1000.0
        } else {
            match trees.coupled().charge() {
                -1 => 30.0,
                0 => [10.0, 20.0][indices[0]],
                2 => 40.0,
                charge => panic!("unexpected charge {charge}"),
            }
        }
    })
    .unwrap();
    let lazy = dense.adjoint().unwrap();

    DIAGONAL_MATERIALIZATIONS.set(0);
    assert_eq!(diagonal.inner(&lazy).unwrap(), 500.0);
    assert_eq!(lazy.inner(&diagonal).unwrap(), 500.0);
    assert_eq!(DIAGONAL_MATERIALIZATIONS.get(), 0);
}

#[test]
fn identity_transforms_preserve_the_cold_lazy_view() {
    let adjoint = u1_lazy_fixture().adjoint().unwrap();
    let TypedTensorRepr::Adjoint(view) = &adjoint.repr else {
        unreachable!()
    };

    let outputs = [
        adjoint.permute(&[0], &[1, 2]).unwrap(),
        adjoint.braid(&[0], &[1, 2], &[0, 1, 2]).unwrap(),
        adjoint.transpose(&[0], &[1, 2]).unwrap(),
        adjoint.repartition(1).unwrap(),
    ];
    for output in &outputs {
        let TypedTensorRepr::Adjoint(output_view) = &output.repr else {
            panic!("identity transform must preserve the lazy representation");
        };
        assert!(Arc::ptr_eq(view, output_view));
    }
    assert!(adjoint.braid(&[0], &[1, 2], &[]).is_err());
    assert!(adjoint.permute(&[0, 0], &[1]).is_err());
    assert!(adjoint.braid(&[0, 0], &[1], &[0, 1, 2]).is_err());
    assert!(adjoint.transpose(&[0, 0], &[1]).is_err());
    assert!(adjoint.repartition(4).is_err());

    let scalar = fixture().trace_pairs(&[(0, 1)]).unwrap();
    let scalar_adjoint = scalar.adjoint().unwrap();
    let TypedTensorRepr::Adjoint(scalar_view) = &scalar_adjoint.repr else {
        unreachable!()
    };
    let scalar_transpose = scalar_adjoint.transpose(&[], &[]).unwrap();
    let TypedTensorRepr::Adjoint(transpose_view) = &scalar_transpose.repr else {
        panic!("rank-zero transpose must preserve the lazy representation");
    };
    assert!(Arc::ptr_eq(scalar_view, transpose_view));
}

#[test]
#[cfg(debug_assertions)]
#[should_panic(expected = "a lazy adjoint never holds a compact diagonal parent")]
fn lazy_adjoint_constructor_rejects_a_compact_diagonal_parent() {
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let bond = GradedSpace::try_new(Arc::new(U1FusionRule), [(U1Irrep::new(0), 2)]).unwrap();
    let diagonal = TensorMap::diagonal(
        &runtime,
        &bond,
        [SectorSpectrum {
            sector: U1Irrep::new(0),
            values: vec![1.0, 2.0],
        }],
    )
    .unwrap();
    let _ = TypedAdjointView::new(
        Arc::clone(owned(&diagonal)),
        diagonal.logical_space().clone(),
    );
}

#[test]
fn compact_adjoint_never_enters_the_lazy_representation() {
    DIAGONAL_MATERIALIZATIONS.set(0);
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let provider = Arc::new(U1FusionRule);
    let bond = GradedSpace::try_new(provider, [(U1Irrep::new(0), 2)]).unwrap();
    let real = TensorMap::diagonal(
        &runtime,
        &bond,
        [SectorSpectrum {
            sector: U1Irrep::new(0),
            values: vec![1.0, 2.0],
        }],
    )
    .unwrap();
    let complex = TensorMap::diagonal(
        &runtime,
        &bond,
        [SectorSpectrum {
            sector: U1Irrep::new(0),
            values: vec![
                num_complex::Complex64::new(1.0, 2.0),
                num_complex::Complex64::new(3.0, -4.0),
            ],
        }],
    )
    .unwrap();

    let real_adjoint = real.adjoint().unwrap();
    let complex_adjoint = complex.adjoint().unwrap();
    assert!(matches!(&real_adjoint.repr, TypedTensorRepr::Owned(_)));
    assert!(matches!(&complex_adjoint.repr, TypedTensorRepr::Owned(_)));
    assert!(matches!(
        owned(&real_adjoint).data.as_ref(),
        TypedData::Diagonal(_)
    ));
    assert_eq!(real_adjoint.spectrum().unwrap()[0].values, [1.0, 2.0]);
    // TensorKit's real `adjoint(d) = d`: the body is shared, not copied.
    assert!(Arc::ptr_eq(owned(&real_adjoint), owned(&real)));
    assert_eq!(DIAGONAL_MATERIALIZATIONS.get(), 0);
    let spectrum = complex_adjoint.spectrum().unwrap();
    assert_eq!(
        spectrum[0].values,
        [
            num_complex::Complex64::new(1.0, -2.0),
            num_complex::Complex64::new(3.0, 4.0)
        ]
    );
    assert_eq!(DIAGONAL_MATERIALIZATIONS.get(), 0);

    let complex_restored = complex_adjoint.adjoint().unwrap();
    assert!(matches!(&complex_restored.repr, TypedTensorRepr::Owned(_)));
    assert!(matches!(
        owned(&complex_restored).data.as_ref(),
        TypedData::Diagonal(_)
    ));
    assert_eq!(
        complex_restored.spectrum().unwrap()[0].values,
        [
            num_complex::Complex64::new(1.0, 2.0),
            num_complex::Complex64::new(3.0, -4.0)
        ]
    );
    assert_eq!(DIAGONAL_MATERIALIZATIONS.get(), 0);
}

fn fixture() -> TensorMap<Z2FusionRule, f64> {
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let leg = GradedSpace::try_new(Arc::new(Z2FusionRule), [(Z2Irrep::EVEN, 8)]).unwrap();
    let mut state = 0x5eed_0580u64;
    TensorMap::from_subblock_fn(&runtime, [&leg], [&leg], move |_, _| {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
        ((state >> 33) as f64) / (u32::MAX as f64) - 0.5
    })
    .unwrap()
}

#[test]
fn clone_copies_no_payload_bytes() {
    // What: `clone` is O(1) in the payload. Measured structurally rather
    // than by an allocator, which cannot distinguish "no copy" from "a
    // copy the size of a warm cache line".
    let tensor = fixture();
    let twin = tensor.clone();
    assert!(Arc::ptr_eq(owned(&tensor), owned(&twin)));
    assert!(Arc::ptr_eq(&owned(&tensor).data, &owned(&twin).data));
    assert_eq!(
        tensor.dense_data().unwrap().as_ptr(),
        twin.dense_data().unwrap().as_ptr()
    );
    // One payload, however many handles reach it.
    assert_eq!(Arc::strong_count(&owned(&tensor).data), 1);
    assert_eq!(Arc::strong_count(owned(&tensor)), 2);
}

/// A small fermionic fixture whose codomain leg 0 carries only the even
/// sector (θ = 1 everywhere on it) while leg 1 and the domain leg carry
/// the odd sector too — so one tensor exposes both twist short-circuit
/// answers.
fn fz2_fixture() -> TensorMap<tenet_core::FermionParityFusionRule, f64> {
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let provider = Arc::new(tenet_core::FermionParityFusionRule);
    let even_only = GradedSpace::try_new(Arc::clone(&provider), [(Z2Irrep::EVEN, 2)]).unwrap();
    let mixed = GradedSpace::try_new(
        Arc::clone(&provider),
        [(Z2Irrep::EVEN, 1), (Z2Irrep::ODD, 2)],
    )
    .unwrap();
    let mut state = 0x5eed_0613u64;
    TensorMap::from_subblock_fn(&runtime, [&even_only, &mixed], [&mixed], move |_, _| {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
        ((state >> 33) as f64) / (u32::MAX as f64) - 0.5
    })
    .unwrap()
}

#[test]
fn unit_insert_and_remove_share_the_dense_payload_arc() {
    DIAGONAL_MATERIALIZATIONS.set(0);
    // What (#580 PR 5, gate 5): the O(1) property the PR 0 gate
    // `a_body_on_a_different_space_reuses_the_payload_allocation` proved
    // by hand-constructing a body, now proved through the real
    // operations it anticipated — a dense payload's `Arc` is shared
    // unchanged through an insert→remove round trip, and the fresh
    // bodies start with cold caches. Supersedes that PR 0 gate: this one
    // checks everything it did (payload reuse at pointer cost under a
    // rewritten space) minus the hand-built struct shape, which the real
    // operations now compile against anyway.
    let tensor = fixture();
    let inserted = tensor.insert_unit(1, Side::Domain, Duality::Plain).unwrap();
    assert!(!Arc::ptr_eq(owned(&tensor), owned(&inserted)));
    assert!(Arc::ptr_eq(&owned(&tensor).data, &owned(&inserted).data));
    assert_eq!(DIAGONAL_MATERIALIZATIONS.get(), 0);
    let removed = inserted.remove_unit(1).unwrap();
    assert!(Arc::ptr_eq(&owned(&tensor).data, &owned(&removed).data));
    // One payload allocation, three bodies holding it.
    assert_eq!(Arc::strong_count(&owned(&tensor).data), 3);
    assert_eq!(
        tensor.dense_data().unwrap().as_ptr(),
        removed.dense_data().unwrap().as_ptr()
    );
}

#[test]
fn a_compact_payload_materializes_exactly_once_for_the_unit_ops() {
    // What (#580 PR 5, gate 5): the compact half of the #613 Group 4
    // contract — a `Diagonal` payload is materialized into a *fresh*
    // dense payload (one copy), and the follow-up remove shares that
    // dense `Arc` rather than copying again.
    let s = fixture().svd_compact(&[0], &[1]).unwrap().s;
    let inserted = s.insert_unit(0, Side::Domain, Duality::Plain).unwrap();
    assert!(!Arc::ptr_eq(&owned(&s).data, &owned(&inserted).data));
    assert!(matches!(&*owned(&inserted).data, TypedData::Dense(_)));
    assert!(matches!(&*owned(&s).data, TypedData::Diagonal(_)));
    let removed = inserted.remove_unit(0).unwrap();
    assert!(Arc::ptr_eq(&owned(&inserted).data, &owned(&removed).data));
}

#[test]
fn twist_identity_short_circuit_shares_the_whole_body() {
    // What (#580 PR 5, gate 5): both identity answers allocate nothing —
    // the bosonic O(1) arm (Z2) and the fermionic per-block scan when no
    // requested leg touches a twisted sector (fZ2, even-only leg 0) both
    // return a body-sharing clone; a leg that does touch the odd sector
    // publishes a new body.
    let tensor = fixture();
    let twisted = tensor.twist(&[0, 1], Direction::Forward).unwrap();
    assert!(Arc::ptr_eq(owned(&tensor), owned(&twisted)));

    let fermionic = fz2_fixture();
    let untouched = fermionic.twist(&[0], Direction::Forward).unwrap();
    assert!(Arc::ptr_eq(owned(&fermionic), owned(&untouched)));
    let touched = fermionic.twist(&[1], Direction::Forward).unwrap();
    assert!(!Arc::ptr_eq(owned(&fermionic), owned(&touched)));
}

fn transform_seam_calls<T>(operation: impl FnOnce() -> T) -> usize {
    crate::tensor_core::TREE_TRANSFORM_SEAM_CALLS.with(|observation| observation.set(Some(0)));
    let _output = operation();
    crate::tensor_core::TREE_TRANSFORM_SEAM_CALLS
        .with(|observation| observation.replace(None))
        .unwrap()
}

fn poison_destination<R, D>(destination: &mut TensorMap<R, D>)
where
    D: TensorScalar,
{
    let TypedTensorRepr::Owned(body) = &mut destination.repr else {
        panic!("overwrite destination fixture must be owned")
    };
    let body = Arc::get_mut(body).expect("overwrite destination body must be unique");
    let data = Arc::get_mut(&mut body.data).expect("overwrite payload must be unique");
    let TypedData::Dense(data) = data else {
        panic!("overwrite destination fixture must be dense")
    };
    data.fill(D::from_real(f64::NAN));
}

fn assert_overwrite_matches<R, D>(
    source: &TensorMap<R, D>,
    expected: TensorMap<R, D>,
    alpha: D,
    overwrite: impl FnOnce(&mut TensorMap<R, D>) -> Result<(), Error>,
) where
    R: MultiplicityFreeRigidSymbols<Scalar = f64> + CheckedFusionAlgebra + SectorCodec,
    D: TensorScalar + core::fmt::Debug + crate::test_numerics::numerics::Numeric,
{
    let source_before = source.dense_data().unwrap().to_vec();
    let mut destination = expected.zeros_like();
    poison_destination(&mut destination);
    let provider = Arc::as_ptr(destination.logical_space().provider_arc());
    let body = Arc::as_ptr(owned(&destination));
    let space = destination.logical_space().space() as *const DynamicFusionMapSpace;
    let storage = destination.dense_data().unwrap().as_ptr();

    overwrite(&mut destination).unwrap();

    // The overwrite folds `alpha` into the recoupling while the oracle
    // scales afterwards; a recoupled entry sums at most one term per
    // source block.
    let scaled = expected.scale(alpha);
    crate::test_numerics::numerics::assert_slices_close(
        "overwrite against the scaled owned route",
        destination.dense_data().unwrap(),
        scaled.dense_data().unwrap(),
        source.subblock_count(),
    );
    assert_eq!(source.dense_data().unwrap(), source_before);
    assert_eq!(
        Arc::as_ptr(destination.logical_space().provider_arc()),
        provider
    );
    assert_eq!(Arc::as_ptr(owned(&destination)), body);
    assert_eq!(
        destination.logical_space().space() as *const DynamicFusionMapSpace,
        space
    );
    assert_eq!(destination.dense_data().unwrap().as_ptr(), storage);
}

#[allow(clippy::too_many_arguments)]
#[allow(deprecated)]
fn assert_contract_overwrite_matches<R, D>(
    label: &str,
    lhs: &TensorMap<R, D>,
    rhs: &TensorMap<R, D>,
    lhs_axes: &[usize],
    rhs_axes: &[usize],
    output_axes: &[usize],
    alpha: D,
    ordered_alias: bool,
) where
    R: TypedSectorAdmission<Error = FusionAlgebraError, Mode = MultiplicityFreeAdmissionMode>
        + MultiplicityFreeRigidSymbols<Scalar = f64>
        + CheckedFusionAlgebra
        + SectorCodec,
    D: TensorScalar + core::fmt::Debug + crate::test_numerics::numerics::Numeric,
{
    let (codomain, domain) = output_axes.split_at(lhs.rank() - lhs_axes.len());
    let spec = ContractSpec {
        lhs: lhs_axes,
        rhs: rhs_axes,
        codomain,
        domain,
    };
    let expected = lhs
        .contract(rhs, &spec)
        .unwrap_or_else(|error| panic!("{label} returning oracle failed: {error:?}"));
    let lhs_before = lhs.dense_data().unwrap().to_vec();
    let rhs_before = rhs.dense_data().unwrap().to_vec();
    let mut destination = expected.zeros_like();
    poison_destination(&mut destination);
    let provider = Arc::as_ptr(destination.logical_space().provider_arc());
    let body = Arc::as_ptr(owned(&destination));
    let space = destination.logical_space().space() as *const DynamicFusionMapSpace;
    let storage = destination.dense_data().unwrap().as_ptr();

    if ordered_alias {
        lhs.contract_into(rhs, &spec, &mut destination, alpha, D::from_real(0.0))
            .unwrap_or_else(|error| panic!("{label} ordered overwrite failed: {error:?}"));
    } else {
        lhs.contract_into(rhs, &spec, &mut destination, alpha, D::from_real(0.0))
            .unwrap_or_else(|error| panic!("{label} overwrite failed: {error:?}"));
    }

    // Owned and destination routes of one contraction. An entry is
    // bilinear in the operands, so `len(lhs) * len(rhs)` bounds its
    // terms, recoupled fusion trees (the cu1 output order) included.
    let terms = lhs.dense_data().unwrap().len() * rhs.dense_data().unwrap().len();
    crate::test_numerics::numerics::assert_slices_close(
        label,
        destination.dense_data().unwrap(),
        expected.scale(alpha).dense_data().unwrap(),
        terms,
    );
    assert_eq!(lhs.dense_data().unwrap(), lhs_before);
    assert_eq!(rhs.dense_data().unwrap(), rhs_before);
    assert_eq!(
        Arc::as_ptr(destination.logical_space().provider_arc()),
        provider
    );
    assert_eq!(Arc::as_ptr(owned(&destination)), body);
    assert_eq!(
        destination.logical_space().space() as *const DynamicFusionMapSpace,
        space
    );
    assert_eq!(destination.dense_data().unwrap().as_ptr(), storage);
}

#[test]
fn typed_tree_overwrite_matches_owned_provider_and_scalar_matrix() {
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();

    let u1_provider = Arc::new(U1FusionRule);
    let u1_leg = GradedSpace::try_new(
        Arc::clone(&u1_provider),
        [
            (U1Irrep::new(-1), 1),
            (U1Irrep::new(0), 2),
            (U1Irrep::new(1), 1),
        ],
    )
    .unwrap();
    let u1 = TensorMap::from_subblock_fn(&runtime, [&u1_leg, &u1_leg], [&u1_leg], |_, indices| {
        indices.iter().sum::<usize>() as f64 + 1.0
    })
    .unwrap();
    let u1_permuted = u1.permute(&[1], &[2, 0]).unwrap();
    assert_overwrite_matches(&u1, u1_permuted, -1.5, |destination| {
        u1.permute_into(&[1], &[2, 0], destination, -1.5, 0.0)
    });
    let u1_identity = u1.zeros_like();
    assert_overwrite_matches(&u1, u1_identity, 0.0, |destination| {
        u1.permute_into(&[0, 1], &[2], destination, 0.0, 0.0)
    });

    let independent_leg = GradedSpace::try_new(
        Arc::new(U1FusionRule),
        [
            (U1Irrep::new(-1), 1),
            (U1Irrep::new(0), 2),
            (U1Irrep::new(1), 1),
        ],
    )
    .unwrap();
    let mut independent_destination = TensorMap::from_subblock_fn(
        &runtime,
        [&independent_leg, &independent_leg],
        [&independent_leg],
        |_, _| f64::NAN,
    )
    .unwrap();
    assert!(!Arc::ptr_eq(
        u1.logical_space().provider_arc(),
        independent_destination.logical_space().provider_arc()
    ));
    let destination_provider = Arc::as_ptr(independent_destination.logical_space().provider_arc());
    u1.permute_into(&[0, 1], &[2], &mut independent_destination, 2.0, 0.0)
        .unwrap();
    assert_eq!(
        independent_destination.dense_data().unwrap(),
        u1.scale(2.0).dense_data().unwrap()
    );
    assert_eq!(
        Arc::as_ptr(independent_destination.logical_space().provider_arc()),
        destination_provider
    );

    let su2_provider = Arc::new(SU2FusionRule);
    let su2_leg = GradedSpace::try_new(
        su2_provider,
        [
            (SU2Irrep::from_twice_spin(0), 2),
            (SU2Irrep::from_twice_spin(1), 1),
        ],
    )
    .unwrap();
    let su2 =
        TensorMap::from_subblock_fn(&runtime, [&su2_leg, &su2_leg], [&su2_leg], |_, indices| {
            indices.iter().sum::<usize>() as f64 + 1.0
        })
        .unwrap()
        .convert::<Complex64>();
    let alpha = num_complex::Complex64::new(0.75, -0.25);
    let su2_permuted = su2.permute(&[1], &[2, 0]).unwrap();
    assert_overwrite_matches(&su2, su2_permuted, alpha, |destination| {
        su2.permute_into(&[1], &[2, 0], destination, alpha, alpha * 0.0)
    });

    let product_provider = Arc::new(U1FusionRule.product(FermionParityFusionRule));
    let product_leg = GradedSpace::try_new(
        product_provider,
        [
            (product_sector(U1Irrep::new(0), Z2Irrep::EVEN), 2),
            (product_sector(U1Irrep::new(1), Z2Irrep::ODD), 1),
        ],
    )
    .unwrap();
    let product = TensorMap::from_subblock_fn(
        &runtime,
        [&product_leg, &product_leg],
        [&product_leg],
        |_, indices| indices.iter().sum::<usize>() as f64 + 1.0,
    )
    .unwrap();
    let product_permuted = product.permute(&[1], &[2, 0]).unwrap();
    assert_overwrite_matches(&product, product_permuted, 2.0, |destination| {
        product.permute_into(&[1], &[2, 0], destination, 2.0, 0.0)
    });
}

#[test]
fn typed_planar_overwrite_matches_fermionic_owned_routes() {
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let provider = Arc::new(FermionParityFusionRule);
    let odd = GradedSpace::try_new(provider, [(Z2Irrep::ODD, 2)]).unwrap();
    let source = TensorMap::from_subblock_fn(&runtime, [&odd, &odd], [&odd, &odd], |_, indices| {
        indices.iter().sum::<usize>() as f64 + 1.0
    })
    .unwrap();
    let alpha = -1.25;

    assert_overwrite_matches(
        &source,
        source.transpose(&[3, 2], &[1, 0]).unwrap(),
        alpha,
        |destination| source.transpose_into(&[3, 2], &[1, 0], destination, alpha, alpha * 0.0),
    );
    assert_overwrite_matches(
        &source,
        source.transpose(&[1, 3], &[0, 2]).unwrap(),
        alpha,
        |destination| source.transpose_into(&[1, 3], &[0, 2], destination, alpha, alpha * 0.0),
    );
    let right = source.repartition(3).unwrap();
    assert_overwrite_matches(&source, right, alpha, |destination| {
        source.repartition_into(destination, alpha, alpha * 0.0)
    });
    let left = source.repartition(1).unwrap();
    assert_overwrite_matches(&source, left, alpha, |destination| {
        source.repartition_into(destination, alpha, alpha * 0.0)
    });
}

fn f64_bits<R>(tensor: &TensorMap<R, f64>) -> Vec<u64> {
    tensor
        .materialize()
        .unwrap()
        .dense_data()
        .unwrap()
        .iter()
        .map(|value| value.to_bits())
        .collect()
}

fn f64_destination_state<R>(tensor: &TensorMap<R, f64>) -> (Vec<u64>, [usize; 4]) {
    (
        f64_bits(tensor),
        [
            Arc::as_ptr(tensor.logical_space().provider_arc()) as usize,
            Arc::as_ptr(owned(tensor)) as usize,
            tensor.logical_space().space() as *const DynamicFusionMapSpace as usize,
            tensor.dense_data().unwrap().as_ptr() as usize,
        ],
    )
}

fn pop_dense_element<R, D>(tensor: &mut TensorMap<R, D>) {
    let TypedTensorRepr::Owned(body) = &mut tensor.repr else {
        panic!("malformed-storage fixture must be owned")
    };
    let body = Arc::get_mut(body).expect("malformed-storage body must be unique");
    let data = Arc::get_mut(&mut body.data).expect("malformed-storage payload must be unique");
    let TypedData::Dense(data) = data else {
        panic!("malformed-storage fixture must be dense")
    };
    data.pop();
}

#[test]
fn typed_tree_overwrite_rejections_leave_destination_unchanged() {
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let provider = Arc::new(U1FusionRule);
    let leg = GradedSpace::try_new(provider, [(U1Irrep::new(0), 2), (U1Irrep::new(1), 1)]).unwrap();
    let source = TensorMap::from_subblock_fn(&runtime, [&leg, &leg], [&leg], |_, indices| {
        indices.iter().sum::<usize>() as f64 + 1.0
    })
    .unwrap();
    let expected = source.permute(&[1], &[2, 0]).unwrap();

    let assert_unchanged = |destination: &TensorMap<U1FusionRule, f64>, before: &[u64]| {
        assert_eq!(f64_bits(destination), before);
    };

    let mut wrong_layout = source.zeros_like();
    poison_destination(&mut wrong_layout);
    let before = f64_bits(&wrong_layout);
    assert!(source
        .permute_into(&[1], &[2, 0], &mut wrong_layout, 1.0, 0.0)
        .is_err());
    assert_unchanged(&wrong_layout, &before);

    for (codomain_axes, domain_axes) in [(&[1, 1][..], &[2][..]), (&[1][..], &[2, 3][..])] {
        let mut destination = expected.zeros_like();
        poison_destination(&mut destination);
        let before = f64_bits(&destination);
        assert!(source
            .permute_into(codomain_axes, domain_axes, &mut destination, 1.0, 0.0)
            .is_err());
        assert_unchanged(&destination, &before);
    }

    let mut nonplanar = source.transpose(&[2], &[1, 0]).unwrap().zeros_like();
    poison_destination(&mut nonplanar);
    let before = f64_bits(&nonplanar);
    assert!(source
        .transpose_into(&[0, 2], &[1], &mut nonplanar, 1.0, 0.0)
        .is_err());
    assert_unchanged(&nonplanar, &before);

    let other_runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let mut foreign = expected.zeros_like();
    foreign.runtime = other_runtime;
    poison_destination(&mut foreign);
    let before = f64_bits(&foreign);
    assert_eq!(
        source
            .permute_into(&[1], &[2, 0], &mut foreign, 1.0, 0.0)
            .unwrap_err(),
        Error::RuntimeMismatch
    );
    assert_unchanged(&foreign, &before);

    let mut shared_body = expected.zeros_like();
    poison_destination(&mut shared_body);
    let before = f64_bits(&shared_body);
    let shared_body_handle = shared_body.clone();
    assert!(source
        .permute_into(&[1], &[2, 0], &mut shared_body, 1.0, 0.0)
        .is_err());
    assert_unchanged(&shared_body, &before);
    drop(shared_body_handle);

    let mut shared_payload = expected.zeros_like();
    poison_destination(&mut shared_payload);
    let before = f64_bits(&shared_payload);
    let payload_handle = shared_payload
        .insert_unit(0, Side::Domain, Duality::Plain)
        .unwrap();
    assert!(source
        .permute_into(&[1], &[2, 0], &mut shared_payload, 1.0, 0.0)
        .is_err());
    assert_unchanged(&shared_payload, &before);
    drop(payload_handle);

    let mut alias = source
        .insert_unit(0, Side::Domain, Duality::Plain)
        .unwrap()
        .remove_unit(0)
        .unwrap();
    let before = f64_bits(&alias);
    assert!(source
        .permute_into(&[0, 1], &[2], &mut alias, 1.0, 0.0)
        .is_err());
    assert_unchanged(&alias, &before);

    let mut bad_len = expected.zeros_like();
    poison_destination(&mut bad_len);
    let before = {
        let TypedTensorRepr::Owned(body) = &mut bad_len.repr else {
            unreachable!()
        };
        let body = Arc::get_mut(body).unwrap();
        let data = Arc::get_mut(&mut body.data).unwrap();
        let TypedData::Dense(data) = data else {
            unreachable!()
        };
        data.pop();
        f64_bits(&bad_len)
    };
    assert!(source
        .permute_into(&[1], &[2, 0], &mut bad_len, 1.0, 0.0)
        .is_err());
    assert_unchanged(&bad_len, &before);

    let mut lazy_destination = expected.adjoint().unwrap();
    let before = f64_bits(&lazy_destination);
    assert!(source
        .permute_into(&[1], &[2, 0], &mut lazy_destination, 1.0, 0.0)
        .is_err());
    assert_eq!(f64_bits(&lazy_destination), before);

    let lazy_source = source.adjoint().unwrap();
    let mut destination = expected.zeros_like();
    poison_destination(&mut destination);
    let before = f64_bits(&destination);
    assert!(lazy_source
        .permute_into(&[1], &[2, 0], &mut destination, 1.0, 0.0)
        .is_err());
    assert_unchanged(&destination, &before);

    let square = TensorMap::from_subblock_fn(&runtime, [&leg], [&leg], |_, indices| {
        indices.iter().sum::<usize>() as f64 + 1.0
    })
    .unwrap();
    let compact = square.svd_compact(&[0], &[1]).unwrap().s;
    let mut compact_destination = compact.zeros_like();
    let before = compact_destination
        .materialize()
        .unwrap()
        .dense_data()
        .unwrap()
        .to_vec();
    assert!(square
        .permute_into(&[0], &[1], &mut compact_destination, 1.0, 0.0)
        .is_err());
    assert_eq!(
        compact_destination
            .materialize()
            .unwrap()
            .dense_data()
            .unwrap(),
        before
    );

    let z2 = Arc::new(ZNFusionRule::new(2).unwrap());
    let z3 = Arc::new(ZNFusionRule::new(3).unwrap());
    let z2_leg = GradedSpace::try_new(Arc::clone(&z2), [(z2.irrep(0), 1)]).unwrap();
    let z3_leg = GradedSpace::try_new(Arc::clone(&z3), [(z3.irrep(0), 1)]).unwrap();
    let z2_source =
        TensorMap::from_subblock_fn(&runtime, [&z2_leg], [&z2_leg], |_, _| 2.0).unwrap();
    let mut z3_destination =
        TensorMap::from_subblock_fn(&runtime, [&z3_leg], [&z3_leg], |_, _| f64::NAN).unwrap();
    let before = f64_bits(&z3_destination);
    assert_eq!(
        z2_source
            .permute_into(&[0], &[1], &mut z3_destination, 1.0, 0.0)
            .unwrap_err(),
        Error::RuleMismatch
    );
    assert_eq!(f64_bits(&z3_destination), before);
}

#[test]
fn typed_tree_overwrite_covers_boundary_ranks_and_runtime_cache_reuse() {
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let provider = Arc::new(U1FusionRule);
    let leg = GradedSpace::try_new(provider, [(U1Irrep::new(0), 1)]).unwrap();

    let one_sided = TensorMap::from_subblock_fn(
        &runtime,
        [&leg],
        std::iter::empty::<&GradedSpace<U1FusionRule>>(),
        |_, _| 3.0,
    )
    .unwrap();
    let moved = one_sided.repartition(0).unwrap();
    assert_overwrite_matches(&one_sided, moved, 2.0, |destination| {
        one_sided.repartition_into(destination, 2.0, 0.0)
    });

    let square = TensorMap::from_subblock_fn(&runtime, [&leg], [&leg], |_, _| 4.0).unwrap();
    let scalar = square.trace_pairs(&[(0, 1)]).unwrap();
    let scalar_destination = scalar.transpose(&[], &[]).unwrap();
    assert_overwrite_matches(&scalar, scalar_destination, -0.5, |destination| {
        scalar.transpose_into(&[], &[], destination, -0.5, 0.0)
    });

    let high_rank = TensorMap::from_subblock_fn(
        &runtime,
        (0..9).map(|_| &leg),
        (0..8).map(|_| &leg),
        |_, _| 1.0,
    )
    .unwrap();
    let mut high_rank_destination = high_rank.zeros_like();
    poison_destination(&mut high_rank_destination);
    let before = f64_bits(&high_rank_destination);
    assert!(high_rank
        .permute_into(
            &[0, 1, 2, 3, 4, 5, 6, 7, 17],
            &[8, 9, 10, 11, 12, 13, 14, 15],
            &mut high_rank_destination,
            1.0,
            0.0,
        )
        .is_err());
    assert_eq!(f64_bits(&high_rank_destination), before);

    runtime.clear_tree_transform_cache();
    let source = TensorMap::from_subblock_fn(&runtime, [&leg, &leg], [&leg], |_, indices| {
        indices.iter().sum::<usize>() as f64 + 1.0
    })
    .unwrap();
    let expected = source.permute(&[1], &[2, 0]).unwrap();
    runtime.clear_tree_transform_cache();
    let mut first = expected.zeros_like();
    source
        .permute_into(&[1], &[2, 0], &mut first, 1.0, 0.0)
        .unwrap();
    let cold = runtime.tree_transform_cache_info().structures;
    let mut second = expected.zeros_like();
    source
        .permute_into(&[1], &[2, 0], &mut second, 1.0, 0.0)
        .unwrap();
    let warm = runtime.tree_transform_cache_info().structures;
    assert_eq!(warm.entries(), cold.entries());
    assert!(warm.hits() > cold.hits());
    assert_eq!(first.dense_data().unwrap(), second.dense_data().unwrap());
}

#[test]
fn typed_tree_overwrite_shared_runtime_is_concurrent_and_deterministic() {
    // What: exact-layout admission and completed replay share one Runtime
    // without serializing execution or changing results across callers.
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let leg = GradedSpace::try_new(
        Arc::new(U1FusionRule),
        [
            (U1Irrep::new(-1), 2),
            (U1Irrep::new(0), 3),
            (U1Irrep::new(1), 2),
        ],
    )
    .unwrap();
    let source = TensorMap::from_subblock_fn(&runtime, [&leg, &leg], [&leg], |_, indices| {
        indices.iter().sum::<usize>() as f64 + 1.0
    })
    .unwrap();
    let expected = source.permute(&[1], &[2, 0]).unwrap();
    let mut warm = expected.zeros_like();
    source
        .permute_into(&[1], &[2, 0], &mut warm, 1.0, 0.0)
        .unwrap();

    std::thread::scope(|scope| {
        let handles: Vec<_> = (0..4)
            .map(|_| {
                scope.spawn(|| {
                    let mut destination = expected.zeros_like();
                    source
                        .permute_into(&[1], &[2, 0], &mut destination, 1.0, 0.0)
                        .unwrap();
                    destination
                })
            })
            .collect();
        for handle in handles {
            assert_eq!(
                handle.join().unwrap().dense_data().unwrap(),
                expected.dense_data().unwrap()
            );
        }
    });
}

#[test]
fn typed_contract_overwrite_matches_provider_scalar_and_order_matrix() {
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();

    let u1_leg = GradedSpace::try_new(
        Arc::new(U1FusionRule),
        [
            (U1Irrep::new(-1), 1),
            (U1Irrep::new(0), 2),
            (U1Irrep::new(1), 1),
        ],
    )
    .unwrap();
    let u1 = TensorMap::from_subblock_fn(&runtime, [&u1_leg], [&u1_leg], |_, indices| {
        indices.iter().sum::<usize>() as f64 + 1.0
    })
    .unwrap();
    assert_contract_overwrite_matches("u1", &u1, &u1, &[1], &[0], &[0, 1], -1.5, false);

    let su2_leg = GradedSpace::try_new(
        Arc::new(SU2FusionRule),
        [
            (SU2Irrep::from_twice_spin(0), 2),
            (SU2Irrep::from_twice_spin(1), 1),
        ],
    )
    .unwrap();
    let su2 = TensorMap::from_subblock_fn(&runtime, [&su2_leg], [&su2_leg], |_, indices| {
        indices.iter().sum::<usize>() as f64 + 1.0
    })
    .unwrap()
    .convert::<Complex64>();
    assert_contract_overwrite_matches(
        "su2",
        &su2,
        &su2,
        &[1],
        &[0],
        &[1, 0],
        num_complex::Complex64::new(0.75, -0.25),
        true,
    );

    let odd = GradedSpace::try_new(Arc::new(FermionParityFusionRule), [(Z2Irrep::ODD, 2)]).unwrap();
    let fermionic = TensorMap::from_subblock_fn(&runtime, [&odd], [&odd], |_, indices| {
        indices.iter().sum::<usize>() as f64 + 1.0
    })
    .unwrap();
    assert_contract_overwrite_matches(
        "fz2",
        &fermionic,
        &fermionic,
        &[0, 1],
        &[1, 0],
        &[],
        0.0,
        false,
    );

    let product_leg = GradedSpace::try_new(
        Arc::new(U1FusionRule.product(FermionParityFusionRule)),
        [
            (product_sector(U1Irrep::new(0), Z2Irrep::EVEN), 2),
            (product_sector(U1Irrep::new(1), Z2Irrep::ODD), 1),
        ],
    )
    .unwrap();
    let product =
        TensorMap::from_subblock_fn(&runtime, [&product_leg], [&product_leg], |_, indices| {
            indices.iter().sum::<usize>() as f64 + 1.0
        })
        .unwrap();
    assert_contract_overwrite_matches(
        "product",
        &product,
        &product,
        &[1],
        &[0],
        &[0, 1],
        2.0,
        false,
    );

    let q = GradedSpace::try_new(
        Arc::new(CU1FusionRule),
        [(CU1Irrep::from_twice_charge(1), 1)],
    )
    .unwrap();
    let cu1 = TensorMap::from_subblock_fn(&runtime, [&q, &q, &q], [&q], |_, _| 1.0).unwrap();
    let cu1_expected = cu1
        .contract(
            &cu1,
            &ContractSpec {
                lhs: &[3],
                rhs: &[0],
                codomain: &[5, 1, 3],
                domain: &[0, 4, 2],
            },
        )
        .unwrap();
    assert!(cu1_expected.dense_data().unwrap().contains(&0.0));
    assert_contract_overwrite_matches(
        "cu1",
        &cu1,
        &cu1,
        &[3],
        &[0],
        &[5, 1, 3, 0, 4, 2],
        1.0,
        false,
    );
}

#[test]
fn typed_contract_overwrite_keeps_distinct_destination_provider_authority() {
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let build = |provider: Arc<U1FusionRule>, offset: f64| {
        let leg = GradedSpace::try_new(
            provider,
            [
                (U1Irrep::new(-1), 1),
                (U1Irrep::new(0), 2),
                (U1Irrep::new(1), 1),
            ],
        )
        .unwrap();
        TensorMap::from_subblock_fn(&runtime, [&leg], [&leg], |_, indices| {
            offset + indices.iter().sum::<usize>() as f64
        })
        .unwrap()
    };

    let lhs = build(Arc::new(U1FusionRule), 1.0);
    let rhs = build(Arc::new(U1FusionRule), 10.0);
    let destination_provider = Arc::new(U1FusionRule);
    let destination_lhs = build(Arc::clone(&destination_provider), 0.0);
    let destination_rhs = build(destination_provider, 0.0);
    let expected = lhs
        .contract(
            &rhs,
            &ContractSpec {
                lhs: &[1],
                rhs: &[0],
                codomain: &[0],
                domain: &[1],
            },
        )
        .unwrap();
    let mut destination = destination_lhs
        .contract(
            &destination_rhs,
            &ContractSpec {
                lhs: &[1],
                rhs: &[0],
                codomain: &[0],
                domain: &[1],
            },
        )
        .unwrap()
        .zeros_like();
    poison_destination(&mut destination);

    assert!(!Arc::ptr_eq(
        lhs.logical_space().provider_arc(),
        rhs.logical_space().provider_arc()
    ));
    assert!(!Arc::ptr_eq(
        lhs.logical_space().provider_arc(),
        destination.logical_space().provider_arc()
    ));
    let provider = Arc::as_ptr(destination.logical_space().provider_arc());
    let body = Arc::as_ptr(owned(&destination));
    let space = destination.logical_space().space() as *const DynamicFusionMapSpace;
    let storage = destination.dense_data().unwrap().as_ptr();

    lhs.contract_into(&rhs, &RANK_TWO_COMPOSE, &mut destination, 1.0, 0.0)
        .unwrap();

    assert_eq!(
        destination.dense_data().unwrap(),
        expected.dense_data().unwrap()
    );
    assert_eq!(
        Arc::as_ptr(destination.logical_space().provider_arc()),
        provider
    );
    assert_eq!(Arc::as_ptr(owned(&destination)), body);
    assert_eq!(
        destination.logical_space().space() as *const DynamicFusionMapSpace,
        space
    );
    assert_eq!(destination.dense_data().unwrap().as_ptr(), storage);
}

#[test]
fn typed_contract_overwrite_accepts_lazy_and_compact_inputs_without_warming_adjoint() {
    DIAGONAL_MATERIALIZATIONS.set(0);
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let leg = GradedSpace::try_new(
        Arc::new(U1FusionRule),
        [(U1Irrep::new(0), 3), (U1Irrep::new(1), 1)],
    )
    .unwrap();
    let lhs = TensorMap::from_subblock_fn(&runtime, [&leg], [&leg], |_, indices| {
        (indices[0] + 2 * indices[1] + 1) as f64
    })
    .unwrap();
    let rhs = TensorMap::from_subblock_fn(&runtime, [&leg], [&leg], |_, indices| {
        (2 * indices[0] + indices[1] + 1) as f64
    })
    .unwrap();
    let lazy_lhs = lhs.adjoint().unwrap();
    let lazy_rhs = rhs.adjoint().unwrap();
    let expected = lazy_lhs
        .contract(
            &lazy_rhs,
            &ContractSpec {
                lhs: &[1],
                rhs: &[0],
                codomain: &[0],
                domain: &[1],
            },
        )
        .unwrap();
    let mut destination = expected.zeros_like();
    poison_destination(&mut destination);
    lazy_lhs
        .contract_into(&lazy_rhs, &RANK_TWO_COMPOSE, &mut destination, 1.0, 0.0)
        .unwrap();
    assert_eq!(
        destination.dense_data().unwrap(),
        expected.dense_data().unwrap()
    );
    for lazy in [&lazy_lhs, &lazy_rhs] {
        let TypedTensorRepr::Adjoint(_) = &lazy.repr else {
            unreachable!()
        };
    }

    let Svd { u, s, .. } = lhs.svd_compact(&[0], &[1]).unwrap();
    assert_eq!(DIAGONAL_MATERIALIZATIONS.get(), 0);
    let expected = u
        .contract(
            &s,
            &ContractSpec {
                lhs: &[1],
                rhs: &[0],
                codomain: &[0],
                domain: &[1],
            },
        )
        .unwrap();
    let mut destination = expected.zeros_like();
    poison_destination(&mut destination);
    u.contract_into(&s, &RANK_TWO_COMPOSE, &mut destination, 1.0, 0.0)
        .unwrap();
    assert_eq!(
        destination.dense_data().unwrap(),
        expected.dense_data().unwrap()
    );
    // Positive control: the compact operand is densified operation-locally.
    assert!(DIAGONAL_MATERIALIZATIONS.get() > 0);
}

#[test]
fn typed_contract_overwrite_rejections_are_preclear_and_atomic() {
    DIAGONAL_MATERIALIZATIONS.set(0);
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let provider = Arc::new(U1FusionRule);
    let leg = GradedSpace::try_new(provider, [(U1Irrep::new(0), 2), (U1Irrep::new(1), 1)]).unwrap();
    let lhs = TensorMap::from_subblock_fn(&runtime, [&leg], [&leg], |_, indices| {
        (indices[0] + 2 * indices[1] + 1) as f64
    })
    .unwrap();
    let rhs = TensorMap::from_subblock_fn(&runtime, [&leg], [&leg], |_, indices| {
        (2 * indices[0] + indices[1] + 1) as f64
    })
    .unwrap();
    let expected = lhs
        .contract(
            &rhs,
            &ContractSpec {
                lhs: &[1],
                rhs: &[0],
                codomain: &[0],
                domain: &[1],
            },
        )
        .unwrap();
    let destination = || {
        let mut destination = expected.zeros_like();
        poison_destination(&mut destination);
        destination
    };

    let mut foreign = destination();
    foreign.runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let before = f64_destination_state(&foreign);
    assert_eq!(
        lhs.contract_into(
            &rhs,
            &ContractSpec {
                lhs: &[9],
                ..RANK_TWO_COMPOSE
            },
            &mut foreign,
            1.0,
            0.0,
        )
        .unwrap_err(),
        Error::RuntimeMismatch
    );
    assert_eq!(f64_destination_state(&foreign), before);

    let other_runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let foreign_rhs = TensorMap::from_subblock_fn(&other_runtime, [&leg], [&leg], |_, indices| {
        (indices[0] + indices[1] + 1) as f64
    })
    .unwrap();
    let mut rejected = destination();
    let before = f64_destination_state(&rejected);
    assert_eq!(
        lhs.contract_into(
            &foreign_rhs,
            &ContractSpec {
                lhs: &[9],
                ..RANK_TWO_COMPOSE
            },
            &mut rejected,
            1.0,
            0.0,
        )
        .unwrap_err(),
        Error::RuntimeMismatch
    );
    assert_eq!(f64_destination_state(&rejected), before);

    for spec in [
        ContractSpec {
            lhs: &[1, 1],
            rhs: &[0, 0],
            codomain: &[],
            domain: &[],
        },
        ContractSpec {
            lhs: &[2],
            ..RANK_TWO_COMPOSE
        },
        ContractSpec {
            domain: &[0],
            ..RANK_TWO_COMPOSE
        },
    ] {
        let mut rejected = destination();
        let before = f64_destination_state(&rejected);
        assert!(lhs
            .contract_into(&rhs, &spec, &mut rejected, 1.0, 0.0)
            .is_err());
        assert_eq!(f64_destination_state(&rejected), before);
    }

    let bad_leg = GradedSpace::try_new(
        Arc::clone(lhs.logical_space().provider_arc()),
        [(U1Irrep::new(7), 1)],
    )
    .unwrap();
    let bad_rhs =
        TensorMap::from_subblock_fn(&runtime, [&bad_leg], [&bad_leg], |_, _| 1.0).unwrap();
    let mut rejected = destination();
    let before = f64_destination_state(&rejected);
    assert!(lhs
        .contract_into(&bad_rhs, &RANK_TWO_COMPOSE, &mut rejected, 1.0, 0.0)
        .is_err());
    assert_eq!(f64_destination_state(&rejected), before);

    let mut wrong_layout = lhs
        .insert_unit(0, Side::Domain, Duality::Plain)
        .unwrap()
        .zeros_like();
    poison_destination(&mut wrong_layout);
    let before = f64_destination_state(&wrong_layout);
    assert!(lhs
        .contract_into(&rhs, &RANK_TWO_COMPOSE, &mut wrong_layout, 1.0, 0.0)
        .is_err());
    assert_eq!(f64_destination_state(&wrong_layout), before);

    for malformed in ["lhs", "rhs", "destination"] {
        let mut bad_lhs = lhs.scale(1.0);
        let mut bad_rhs = rhs.scale(1.0);
        let mut rejected = destination();
        match malformed {
            "lhs" => pop_dense_element(&mut bad_lhs),
            "rhs" => pop_dense_element(&mut bad_rhs),
            "destination" => pop_dense_element(&mut rejected),
            _ => unreachable!(),
        }
        let before = f64_destination_state(&rejected);
        assert!(bad_lhs
            .contract_into(&bad_rhs, &RANK_TWO_COMPOSE, &mut rejected, 1.0, 0.0)
            .is_err());
        assert_eq!(f64_destination_state(&rejected), before);
    }

    let mut lhs_alias = lhs
        .insert_unit(0, Side::Domain, Duality::Plain)
        .unwrap()
        .remove_unit(0)
        .unwrap();
    let before = f64_destination_state(&lhs_alias);
    assert!(lhs
        .contract_into(&rhs, &RANK_TWO_COMPOSE, &mut lhs_alias, 1.0, 0.0)
        .is_err());
    assert_eq!(f64_destination_state(&lhs_alias), before);

    let mut rhs_alias = rhs
        .insert_unit(0, Side::Domain, Duality::Plain)
        .unwrap()
        .remove_unit(0)
        .unwrap();
    let before = f64_destination_state(&rhs_alias);
    assert!(lhs
        .contract_into(&rhs, &RANK_TWO_COMPOSE, &mut rhs_alias, 1.0, 0.0)
        .is_err());
    assert_eq!(f64_destination_state(&rhs_alias), before);

    let mut shared_body = destination();
    let shared_body_handle = shared_body.clone();
    let before = f64_destination_state(&shared_body);
    assert!(lhs
        .contract_into(&rhs, &RANK_TWO_COMPOSE, &mut shared_body, 1.0, 0.0)
        .is_err());
    assert_eq!(f64_destination_state(&shared_body), before);
    drop(shared_body_handle);

    let mut shared_payload = destination();
    let shared_payload_handle = shared_payload
        .insert_unit(0, Side::Domain, Duality::Plain)
        .unwrap();
    let before = f64_destination_state(&shared_payload);
    assert!(lhs
        .contract_into(&rhs, &RANK_TWO_COMPOSE, &mut shared_payload, 1.0, 0.0)
        .is_err());
    assert_eq!(f64_destination_state(&shared_payload), before);
    drop(shared_payload_handle);

    let mut lazy_destination = expected.adjoint().unwrap();
    let TypedTensorRepr::Adjoint(view) = &lazy_destination.repr else {
        unreachable!()
    };
    let view = Arc::as_ptr(view);
    assert!(lhs
        .contract_into(&rhs, &RANK_TWO_COMPOSE, &mut lazy_destination, 1.0, 0.0)
        .is_err());
    let TypedTensorRepr::Adjoint(after) = &lazy_destination.repr else {
        unreachable!()
    };
    assert_eq!(Arc::as_ptr(after), view);

    let mut compact_destination = lhs.svd_compact(&[0], &[1]).unwrap().s;
    let payload = Arc::clone(&owned(&compact_destination).data);
    assert_eq!(DIAGONAL_MATERIALIZATIONS.get(), 0);
    assert!(lhs
        .contract_into(&rhs, &RANK_TWO_COMPOSE, &mut compact_destination, 1.0, 0.0)
        .is_err());
    assert!(Arc::ptr_eq(&owned(&compact_destination).data, &payload));
    assert_eq!(DIAGONAL_MATERIALIZATIONS.get(), 0);

    let z2 = Arc::new(ZNFusionRule::new(2).unwrap());
    let z3 = Arc::new(ZNFusionRule::new(3).unwrap());
    let z2_leg = GradedSpace::try_new(Arc::clone(&z2), [(z2.irrep(0), 1)]).unwrap();
    let z3_leg = GradedSpace::try_new(Arc::clone(&z3), [(z3.irrep(0), 1)]).unwrap();
    let z2_lhs = TensorMap::from_subblock_fn(&runtime, [&z2_leg], [&z2_leg], |_, _| 1.0).unwrap();
    let z2_rhs = TensorMap::from_subblock_fn(&runtime, [&z2_leg], [&z2_leg], |_, _| 2.0).unwrap();
    let z3_rhs = TensorMap::from_subblock_fn(&runtime, [&z3_leg], [&z3_leg], |_, _| 2.0).unwrap();
    let mut rejected = z2_lhs
        .contract(
            &z2_rhs,
            &ContractSpec {
                lhs: &[1],
                rhs: &[0],
                codomain: &[0],
                domain: &[1],
            },
        )
        .unwrap()
        .zeros_like();
    poison_destination(&mut rejected);
    let before = f64_destination_state(&rejected);
    assert_eq!(
        z2_lhs
            .contract_into(&z3_rhs, &RANK_TWO_COMPOSE, &mut rejected, 1.0, 0.0)
            .unwrap_err(),
        Error::RuleMismatch
    );
    assert_eq!(f64_destination_state(&rejected), before);

    let z3_lhs = TensorMap::from_subblock_fn(&runtime, [&z3_leg], [&z3_leg], |_, _| 1.0).unwrap();
    let mut z3_destination = z3_lhs
        .contract(
            &z3_rhs,
            &ContractSpec {
                lhs: &[1],
                rhs: &[0],
                codomain: &[0],
                domain: &[1],
            },
        )
        .unwrap()
        .zeros_like();
    poison_destination(&mut z3_destination);
    let before = f64_destination_state(&z3_destination);
    assert_eq!(
        z2_lhs
            .contract_into(&z2_rhs, &RANK_TWO_COMPOSE, &mut z3_destination, 1.0, 0.0)
            .unwrap_err(),
        Error::RuleMismatch
    );
    assert_eq!(f64_destination_state(&z3_destination), before);
}

#[test]
fn typed_contract_overwrite_handles_unmatched_sectors_and_reuses_runtime_cache() {
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let provider = Arc::new(U1FusionRule);
    let bond = GradedSpace::try_new(Arc::clone(&provider), [(U1Irrep::new(0), 2)]).unwrap();
    let left = GradedSpace::try_new(
        Arc::clone(&provider),
        [(U1Irrep::new(0), 1), (U1Irrep::new(1), 1)],
    )
    .unwrap();
    let partly_disjoint = GradedSpace::try_new(
        Arc::clone(&provider),
        [(U1Irrep::new(0), 1), (U1Irrep::new(2), 1)],
    )
    .unwrap();
    let disjoint = GradedSpace::try_new(provider, [(U1Irrep::new(3), 1)]).unwrap();
    let lhs = TensorMap::from_subblock_fn(&runtime, [&left], [&bond], |_, indices| {
        indices.iter().sum::<usize>() as f64 + 1.0
    })
    .unwrap();
    for open in [&partly_disjoint, &disjoint] {
        let rhs = TensorMap::from_subblock_fn(&runtime, [&bond], [open], |_, indices| {
            indices.iter().sum::<usize>() as f64 + 2.0
        })
        .unwrap();
        assert_contract_overwrite_matches(
            "unmatched sectors",
            &lhs,
            &rhs,
            &[1],
            &[0],
            &[0, 1],
            1.0,
            false,
        );
    }

    let provider = Arc::new(CU1FusionRule);
    let q = GradedSpace::try_new(provider, [(CU1Irrep::from_twice_charge(1), 1)]).unwrap();
    let source = TensorMap::from_subblock_fn(&runtime, [&q, &q, &q], [&q], |_, _| 1.0).unwrap();
    let axes = [5, 1, 3, 0, 4, 2];
    let expected = source
        .contract(
            &source,
            &ContractSpec {
                lhs: &[3],
                rhs: &[0],
                codomain: &axes[..3],
                domain: &axes[3..],
            },
        )
        .unwrap();
    runtime.clear_tree_transform_cache();
    let mut first = expected.zeros_like();
    poison_destination(&mut first);
    source
        .contract_into(
            &source,
            &ContractSpec {
                lhs: &[3],
                rhs: &[0],
                codomain: &axes[..3],
                domain: &axes[3..],
            },
            &mut first,
            1.0,
            0.0,
        )
        .unwrap();
    let cold = runtime.tree_transform_cache_info().structures;
    let mut second = expected.zeros_like();
    poison_destination(&mut second);
    source
        .contract_into(
            &source,
            &ContractSpec {
                lhs: &[3],
                rhs: &[0],
                codomain: &axes[..3],
                domain: &axes[3..],
            },
            &mut second,
            1.0,
            0.0,
        )
        .unwrap();
    let warm = runtime.tree_transform_cache_info().structures;
    assert_eq!(first.dense_data().unwrap(), second.dense_data().unwrap());
    assert_eq!(warm.entries(), cold.entries());
    assert!(warm.hits() > cold.hits());
}

#[test]
fn exact_identity_transforms_share_unique_and_simple_bodies() {
    // What (#689 PR A): exact identity permute/braid/transpose/repartition
    // never reach the transform seam and return the same body allocation.
    // Body identity also pins zero payload copies more directly than an
    // allocator byte count can.
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();

    let u1_provider = Arc::new(U1FusionRule);
    let u1_leg = GradedSpace::try_new(
        Arc::clone(&u1_provider),
        [
            (U1Irrep::new(-1), 1),
            (U1Irrep::new(0), 2),
            (U1Irrep::new(1), 1),
        ],
    )
    .unwrap();
    let u1_f64: TensorMap<U1FusionRule, f64> =
        TensorMap::from_subblock_fn(&runtime, [&u1_leg, &u1_leg], [&u1_leg], |_, indices| {
            indices.iter().sum::<usize>() as f64 + 1.0
        })
        .unwrap();
    let u1_c64 = u1_f64.convert::<Complex64>();

    let su2_provider = Arc::new(SU2FusionRule);
    let su2_leg = GradedSpace::try_new(
        Arc::clone(&su2_provider),
        [
            (SU2Irrep::from_twice_spin(0), 2),
            (SU2Irrep::from_twice_spin(1), 1),
        ],
    )
    .unwrap();
    let su2_f64: TensorMap<SU2FusionRule, f64> =
        TensorMap::from_subblock_fn(&runtime, [&su2_leg, &su2_leg], [&su2_leg], |_, indices| {
            indices.iter().sum::<usize>() as f64 + 1.0
        })
        .unwrap();
    let su2_c64 = su2_f64.convert::<Complex64>();

    macro_rules! assert_identity_ops {
        ($tensor:expr) => {{
            let tensor = $tensor;
            let calls = transform_seam_calls(|| {
                for output in [
                    tensor.permute(&[0, 1], &[2]).unwrap(),
                    tensor.braid(&[0, 1], &[2], &[5, 3, 1]).unwrap(),
                    tensor.transpose(&[0, 1], &[2]).unwrap(),
                    tensor.repartition(2).unwrap(),
                ] {
                    assert!(Arc::ptr_eq(owned(tensor), owned(&output)));
                    assert!(Arc::ptr_eq(&owned(tensor).data, &owned(&output).data));
                    assert_eq!(
                        tensor.dense_data().unwrap().as_ptr(),
                        output.dense_data().unwrap().as_ptr()
                    );
                }
            });
            assert_eq!(calls, 0);
        }};
    }

    assert_identity_ops!(&u1_f64);
    assert_identity_ops!(&u1_c64);
    assert_identity_ops!(&su2_f64);
    assert_identity_ops!(&su2_c64);

    // Validation still precedes the braid shortcut.
    let calls = transform_seam_calls(|| {
        assert!(u1_f64.braid(&[0, 1], &[2], &[0, 0]).is_err());
    });
    assert_eq!(calls, 0);
    assert!(u1_f64.permute(&[0, 1], &[2, 3]).is_err());
    assert!(u1_f64.transpose(&[0, 1], &[2, 3]).is_err());
    assert!(u1_f64.repartition(4).is_err());

    // Negative control: the counter observes a real transform.
    let calls = transform_seam_calls(|| {
        let moved = u1_f64.permute(&[1, 0], &[2]).unwrap();
        assert!(!Arc::ptr_eq(owned(&u1_f64), owned(&moved)));
    });
    assert_eq!(calls, 1);
}

#[test]
fn high_rank_identity_has_no_inline_capacity_boundary() {
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let provider = Arc::new(U1FusionRule);
    let leg = GradedSpace::try_new(provider, [(U1Irrep::new(0), 1)]).unwrap();
    let tensor: TensorMap<U1FusionRule, f64> =
        TensorMap::from_subblock_fn(&runtime, [&leg; 10], [&leg; 9], |_, _| 1.0).unwrap();
    let codomain_axes: Vec<_> = (0..10).collect();
    let domain_axes: Vec<_> = (10..19).collect();
    let levels = vec![0; 19];

    let calls = transform_seam_calls(|| {
        for output in [
            tensor.permute(&codomain_axes, &domain_axes).unwrap(),
            tensor.braid(&codomain_axes, &domain_axes, &levels).unwrap(),
            tensor.transpose(&codomain_axes, &domain_axes).unwrap(),
        ] {
            assert!(Arc::ptr_eq(owned(&tensor), owned(&output)));
        }
    });
    assert_eq!(calls, 0);
}

#[test]
fn compact_identity_transforms_do_not_materialize() {
    DIAGONAL_MATERIALIZATIONS.set(0);
    let factor = fixture().svd_compact(&[0], &[1]).unwrap().s;
    assert!(matches!(&*owned(&factor).data, TypedData::Diagonal(_)));
    assert_eq!(DIAGONAL_MATERIALIZATIONS.get(), 0);

    let calls = transform_seam_calls(|| {
        for output in [
            factor.permute(&[0], &[1]).unwrap(),
            factor.braid(&[0], &[1], &[2, 1]).unwrap(),
            factor.transpose(&[0], &[1]).unwrap(),
            factor.repartition(1).unwrap(),
        ] {
            assert!(Arc::ptr_eq(owned(&factor), owned(&output)));
            assert!(matches!(&*owned(&output).data, TypedData::Diagonal(_)));
            assert_eq!(DIAGONAL_MATERIALIZATIONS.get(), 0);
        }
    });
    assert_eq!(calls, 0);
    assert_eq!(DIAGONAL_MATERIALIZATIONS.get(), 0);
}

#[test]
fn scalar_transpose_shares_the_body() {
    let scalar = fixture().trace_pairs(&[(0, 1)]).unwrap();
    assert_eq!(scalar.rank(), 0);
    let calls = transform_seam_calls(|| {
        let transposed = scalar.transpose(&[], &[]).unwrap();
        assert!(Arc::ptr_eq(owned(&scalar), owned(&transposed)));
    });
    assert_eq!(calls, 0);
}

#[test]
fn twist_on_a_compact_spectrum_stays_compact() {
    // What (#580 PR 5, gate 5): the compact twist arm scales
    // spectrum-per-sector and keeps `TypedData::Diagonal` — the space is
    // unchanged, so O(Σ_c k_c) storage survives — and its own identity
    // answer (θ ≡ 1 across the spectrum's sectors) is a body-sharing
    // clone.
    let s = fz2_fixture().svd_compact(&[0, 1], &[2]).unwrap().s;
    let twisted = s.twist(&[0], Direction::Forward).unwrap();
    assert!(matches!(&*owned(&twisted).data, TypedData::Diagonal(_)));
    assert!(!Arc::ptr_eq(&owned(&s).data, &owned(&twisted).data));
    let inverse = s.twist(&[0], Direction::Inverse).unwrap();
    assert!(matches!(&*owned(&inverse).data, TypedData::Diagonal(_)));
    assert!(!Arc::ptr_eq(&owned(&s).data, &owned(&inverse).data));

    let bosonic_s = fixture().svd_compact(&[0], &[1]).unwrap().s;
    let untouched = bosonic_s.twist(&[0], Direction::Forward).unwrap();
    assert!(Arc::ptr_eq(owned(&bosonic_s), owned(&untouched)));
    let untouched_inverse = bosonic_s.twist(&[0], Direction::Inverse).unwrap();
    assert!(Arc::ptr_eq(owned(&bosonic_s), owned(&untouched_inverse)));
}

#[test]
fn lazy_cat_reads_parent_storage_without_publishing_adjoint_caches() {
    let runtime = Runtime::builder().build().unwrap();
    let provider = Arc::new(U1FusionRule);
    let leg = |degeneracy| {
        GradedSpace::try_new(Arc::clone(&provider), [(U1Irrep::new(0), degeneracy)]).unwrap()
    };
    let common = leg(3);
    let left = leg(2);
    let right = leg(4);
    let lhs: TensorMap<U1FusionRule, num_complex::Complex64> =
        TensorMap::rand_with_seed(&runtime, [&left], [&common], 773_101)
            .unwrap()
            .adjoint()
            .unwrap();
    let rhs: TensorMap<U1FusionRule, num_complex::Complex64> =
        TensorMap::rand_with_seed(&runtime, [&right], [&common], 773_102)
            .unwrap()
            .adjoint()
            .unwrap();

    let _ = lhs.cat(&rhs, Side::Domain).unwrap();

    let upper: TensorMap<U1FusionRule, num_complex::Complex64> =
        TensorMap::rand_with_seed(&runtime, [&common], [&left], 773_103)
            .unwrap()
            .adjoint()
            .unwrap();
    let lower: TensorMap<U1FusionRule, num_complex::Complex64> =
        TensorMap::rand_with_seed(&runtime, [&common], [&right], 773_104)
            .unwrap()
            .adjoint()
            .unwrap();
    upper.cat(&lower, Side::Codomain).unwrap();
}

#[test]
fn a_written_payload_leaves_the_shared_one_untouched() {
    // What: clone-then-modify. Sharing is only sound if a write on one
    // handle cannot be seen through the other — every write route in this
    // module publishes a new payload rather than reaching through the `Arc`.
    let tensor = fixture();
    let twin = tensor.clone();
    let before: Vec<f64> = tensor.dense_data().unwrap().to_vec();

    let scaled = twin.scale(2.0);

    assert_eq!(tensor.dense_data().unwrap(), before.as_slice());
    assert_eq!(twin.dense_data().unwrap(), before.as_slice());
    assert_ne!(
        scaled.dense_data().unwrap().as_ptr(),
        tensor.dense_data().unwrap().as_ptr()
    );
    assert!(!Arc::ptr_eq(&owned(&scaled).data, &owned(&tensor).data));
}

#[test]
fn weighted_trace_keeps_misaligned_and_nonpacked_layouts_on_the_literal_walk() {
    let tree = |dual| {
        FusionTreeKey::try_from_sector_ids_for_rule(&Z2FusionRule, [0], 0, [dual], [], []).unwrap()
    };
    let (a, b) = (tree(false), tree(true));
    let block = |row: &FusionTreeKey, col: &FusionTreeKey, offset| {
        BlockSpec::with_key(
            BlockKey::FusionTree(FusionTreePairKey::pair(row.clone(), col.clone())),
            vec![1, 1],
            vec![1, 2],
            offset,
        )
        .unwrap()
    };
    // The packed matrix's logical row order is [a, b] while its columns
    // are [b, a]; its two literal diagonal blocks occur b then a.
    let misaligned = BlockStructure::from_blocks(vec![
        block(&a, &b, 0),
        block(&b, &b, 1),
        block(&a, &a, 2),
        block(&b, &a, 3),
    ])
    .unwrap();
    assert!(!misaligned.coupled_sector_regions(1).unwrap().unwrap()[0].has_aligned_diagonal());
    let mut weights = Vec::new();
    let value = weighted_trace(&misaligned, 1, &[10.0, 20.0, 30.0, 40.0], |sector| {
        weights.push(sector);
        Ok::<_, Error>(2.0)
    })
    .unwrap();
    assert_eq!(value, Complex64::new(100.0, 0.0));
    assert_eq!(weights, vec![SectorId::new(0), SectorId::new(0)]);

    let nonpacked = BlockStructure::from_blocks(vec![BlockSpec::with_key(
        BlockKey::FusionTree(FusionTreePairKey::pair(a.clone(), a)),
        vec![2, 2],
        vec![2, 4],
        0,
    )
    .unwrap()])
    .unwrap();
    assert_eq!(nonpacked.coupled_sector_regions(1).unwrap(), None);
    let value = weighted_trace(&nonpacked, 1, &[3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 5.0], |_| {
        Ok::<_, Error>(2.0)
    })
    .unwrap();
    assert_eq!(value, Complex64::new(16.0, 0.0));
}

/// GL-3 (#1281): a device operation must not need the coarse Runtime
/// state mutex. The reverse direction is the observable one: Host
/// standalone contraction never takes `state`, so only a parked holder of
/// that lock can prove a device operation is independent of it.
#[cfg(feature = "cuda")]
#[test]
#[ignore = "requires a real CUDA device"]
fn device_work_completes_while_another_thread_holds_the_runtime_state_lock() {
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::mpsc;
    use std::sync::Barrier;
    use std::time::Duration;

    let runtime = Runtime::builder().cuda(0).dense_threads(1).build().unwrap();
    let provider = Arc::new(U1FusionRule);
    let leg = GradedSpace::try_new(
        Arc::clone(&provider),
        [(U1Irrep::new(0), 2), (U1Irrep::new(1), 2)],
    )
    .unwrap();
    let lhs = TensorMap::from_subblock_fn(&runtime, [&leg], [&leg], |_, indices| {
        indices[0] as f64 + 1.0
    })
    .unwrap();
    let rhs = TensorMap::from_subblock_fn(&runtime, [&leg], [&leg], |_, indices| {
        indices[1] as f64 + 2.0
    })
    .unwrap();

    // Independent oracle for the device result, computed on Host before
    // the state lock is parked.
    let host_expected = TensorMap::from_subblock_fn(&runtime, [&leg], [&leg], |_, indices| {
        indices[0] as f64 + 1.0
    })
    .unwrap()
    .contract(
        &TensorMap::from_subblock_fn(&runtime, [&leg], [&leg], |_, indices| {
            indices[1] as f64 + 2.0
        })
        .unwrap(),
        &ContractSpec {
            lhs: &[1],
            rhs: &[0],
            codomain: &[0],
            domain: &[1],
        },
    )
    .unwrap();

    let holding = Arc::new(Barrier::new(2));
    let release = Arc::new(AtomicBool::new(false));
    let (sender, receiver) = mpsc::channel();

    let values = std::thread::scope(|scope| {
        let holder_runtime = runtime.clone();
        let holder_barrier = Arc::clone(&holding);
        let holder_release = Arc::clone(&release);
        scope.spawn(move || {
            let _state = holder_runtime.lock();
            holder_barrier.wait();
            while !holder_release.load(Ordering::SeqCst) {
                std::thread::yield_now();
            }
        });

        holding.wait();
        scope.spawn(move || {
            let device = lhs
                .to_cuda()
                .and_then(|lhs| Ok((lhs, rhs.to_cuda()?)))
                .and_then(|(lhs, rhs)| {
                    lhs.contract(
                        &rhs,
                        &ContractSpec {
                            lhs: &[1],
                            rhs: &[0],
                            codomain: &[0],
                            domain: &[1],
                        },
                    )
                })
                .and_then(|out| out.to_host());
            let _ = sender.send(device.map(|out| out.dense_data().unwrap().to_vec()));
        });

        let outcome = receiver.recv_timeout(Duration::from_secs(30));
        release.store(true, Ordering::SeqCst);
        outcome
            .expect("device transfer and contraction blocked on the Runtime state lock")
            .expect("device contraction failed")
    });

    // The device result under the parked state lock is the Host result.
    assert_eq!(values.len(), host_expected.dense_data().unwrap().len());
    assert!(values.iter().any(|value| *value != 0.0));
    for (actual, expected) in values.iter().zip(host_expected.dense_data().unwrap()) {
        assert!(
            (actual - expected).abs() < 1.0e-12,
            "device value {actual} differs from the Host oracle {expected}"
        );
    }
}

/// GL-3 (#1281): the device lock nests with the CPU leases in either
/// order without deadlocking, because it is a separate mutex.
#[cfg(feature = "cuda")]
#[test]
#[ignore = "requires a real CUDA device"]
fn device_lease_nests_with_cpu_leases_in_both_orders() {
    let runtime = Runtime::builder().cuda(0).dense_threads(1).build().unwrap();

    {
        let cuda = runtime.lease_cuda().unwrap();
        let context = runtime.lease_context().unwrap();
        let dense = runtime.lease_dense();
        drop(dense);
        // A CPU lease released under the device guard must not deadlock.
        drop(context);
        drop(cuda);
    }

    let context = runtime.lease_context().unwrap();
    let cuda = runtime.lease_cuda().unwrap();
    drop(context);
    drop(cuda);
}

/// GL-3 (#1281): device lowering touches no execution context, so a
/// device contraction leaves the Runtime tree-transform cache untouched.
#[cfg(feature = "cuda")]
#[test]
#[ignore = "requires a real CUDA device"]
fn device_contraction_leaves_the_tree_transform_cache_unchanged() {
    let runtime = Runtime::builder().cuda(0).dense_threads(1).build().unwrap();
    let provider = Arc::new(U1FusionRule);
    let leg = GradedSpace::try_new(
        Arc::clone(&provider),
        [(U1Irrep::new(0), 2), (U1Irrep::new(1), 2)],
    )
    .unwrap();
    let lhs = TensorMap::from_subblock_fn(&runtime, [&leg], [&leg], |_, indices| {
        indices[0] as f64 + 1.0
    })
    .unwrap()
    .to_cuda()
    .unwrap();
    let rhs = TensorMap::from_subblock_fn(&runtime, [&leg], [&leg], |_, indices| {
        indices[1] as f64 + 2.0
    })
    .unwrap()
    .to_cuda()
    .unwrap();

    let before = runtime.tree_transform_cache_info().structures;
    let product = lhs
        .contract(
            &rhs,
            &ContractSpec {
                lhs: &[1],
                rhs: &[0],
                codomain: &[0],
                domain: &[1],
            },
        )
        .unwrap();
    assert_eq!(product.placement(), Placement::Cuda(0));
    assert_eq!(runtime.tree_transform_cache_info().structures, before);
}

/// #1337: filling the compact payload from a borrowed slice instead of
/// consuming the spectrum must move exactly the same bits, in the same
/// order, at every payload dtype.
///
/// The expected values are written out rather than derived from
/// `FactorScalar::from_real`, so the assertion is independent of the
/// conversion under test. The fixture carries a zero, a negative value, an
/// `f64` subnormal (which `f32` must flush to `+0.0`, not to a NaN or a
/// denormal of its own) and an `f32` subnormal (which must survive), and
/// declares its sectors out of engine order so a factor that kept the
/// caller's order or reordered values inside a sector fails.
#[test]
fn diagonal_spectrum_factor_converts_every_value_bitwise() {
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let provider = Arc::new(U1FusionRule);
    let zero = U1Irrep::new(0);
    let one = U1Irrep::new(1);
    let leg = GradedSpace::try_new(Arc::clone(&provider), [(zero, 3), (one, 2)]).unwrap();
    let authority: TensorMap<_, f64> =
        TensorMap::from_subblock_fn(&runtime, [&leg], [&leg], |_, _| 0.0).unwrap();
    let id = |label| TypedSectorAdmission::try_encode_label(provider.as_ref(), &label).unwrap();

    let tiny = f64::from(f32::MIN_POSITIVE) / 2.0;
    let spectrum = || {
        vec![
            tenet_matrixalgebra::SectorSpectrum {
                sector: id(one),
                values: vec![0.0_f64, -1.5],
            },
            tenet_matrixalgebra::SectorSpectrum {
                sector: id(zero),
                values: vec![1.0_f64, f64::MIN_POSITIVE / 2.0, tiny],
            },
        ]
    };

    let mut wide = spectrum();
    let wide_factor: TensorMap<_, f64> = diagonal_factor_on(
        &runtime,
        authority.logical_space(),
        &mut wide,
        <f64 as FactorScalar>::from_real,
    )
    .unwrap();
    let seen = wide_factor.diagview().unwrap();
    let bits: Vec<(U1Irrep, Vec<u64>)> = seen
        .iter()
        .map(|entry| {
            (
                entry.sector,
                entry.values.iter().map(|value| value.to_bits()).collect(),
            )
        })
        .collect();
    assert_eq!(
        bits,
        vec![
            (
                zero,
                vec![
                    1.0_f64.to_bits(),
                    (f64::MIN_POSITIVE / 2.0).to_bits(),
                    tiny.to_bits()
                ]
            ),
            (one, vec![0.0_f64.to_bits(), (-1.5_f64).to_bits()]),
        ]
    );

    let mut narrow = spectrum();
    let narrow_factor: TensorMap<_, f32> = diagonal_factor_on(
        &runtime,
        authority.logical_space(),
        &mut narrow,
        <f32 as FactorScalar>::from_real,
    )
    .unwrap();
    let seen = narrow_factor.diagview().unwrap();
    let bits: Vec<(U1Irrep, Vec<u32>)> = seen
        .iter()
        .map(|entry| {
            (
                entry.sector,
                entry.values.iter().map(|value| value.to_bits()).collect(),
            )
        })
        .collect();
    assert_eq!(
        bits,
        vec![
            (
                zero,
                vec![
                    1.0_f32.to_bits(),
                    0.0_f32.to_bits(),
                    (f32::MIN_POSITIVE / 2.0).to_bits()
                ]
            ),
            (one, vec![0.0_f32.to_bits(), (-1.5_f32).to_bits()]),
        ]
    );
}

#[test]
fn compact_arms_never_densify_their_spectrum_operand() {
    // What (#1548): the operations with a compact-diagonal arm read the stored
    // spectrum and never enter the operation-local densification. The probe
    // counts entries directly, which the integration byte ceilings cannot
    // tell apart from a scaled copy.
    let tensor = fixture();
    let Svd { u, s: d, vh } = tensor.svd_compact(&[0], &[1]).unwrap();
    let complex_d = tensor
        .convert::<Complex64>()
        .svd_compact(&[0], &[1])
        .unwrap()
        .s;
    DIAGONAL_MATERIALIZATIONS.set(0);

    let _ = d.scale(0.5);
    let _ = d.adjoint().unwrap();
    let _ = d.axpby(0.75, &d, -0.5).unwrap();
    let _ = d.axpby(0.75, &tensor, -0.5).unwrap();
    let _ = tensor.axpby(0.75, &d, -0.5).unwrap();
    let _ = d.compose(&d).unwrap();
    let _ = u.compose(&d).unwrap();
    let _ = d.compose(&vh).unwrap();
    for p in [2.0, f64::INFINITY, 3.0] {
        let _ = d.norm(p).unwrap();
    }
    let _ = d.tr().unwrap();
    let _ = d.inner(&d).unwrap();
    let _ = d.exp(&[0], &[1]).unwrap();
    let _ = d.inv(&[0], &[1]).unwrap();
    let _ = d.pinv(&[0], &[1], 1e-12).unwrap();
    let _ = d.map_diagonal(|x| x.abs().sqrt()).unwrap();
    let _ = tensor
        .contract(
            &d,
            &ContractSpec {
                lhs: &[1],
                rhs: &[0],
                codomain: &[0],
                domain: &[1],
            },
        )
        .unwrap();
    let _ = tensor
        .contract(
            &d,
            &ContractSpec {
                lhs: &[1],
                rhs: &[0],
                codomain: &[1],
                domain: &[0],
            },
        )
        .unwrap();
    let _ = d
        .contract(
            &tensor,
            &ContractSpec {
                lhs: &[1],
                rhs: &[0],
                codomain: &[0],
                domain: &[1],
            },
        )
        .unwrap();
    let _ = d
        .contract(
            &d,
            &ContractSpec {
                lhs: &[1],
                rhs: &[0],
                codomain: &[0],
                domain: &[1],
            },
        )
        .unwrap();
    let _ = d.permute(&[1], &[0]).unwrap();
    let _ = d.transpose(&[1], &[0]).unwrap();
    let _ = d.repartition(1).unwrap();
    let _ = d.zeros_like();
    let _ = d.convert::<Complex64>();
    let _ = complex_d.re();
    let _ = complex_d.im();
    let _ = d.trace_pairs(&[(0, 1)]).unwrap();
    let _ = d.diagview().unwrap();
    assert_eq!(DIAGONAL_MATERIALIZATIONS.get(), 0);

    // A one-term rank-(1,1) braid reads the compact source directly.
    let _ = d.braid(&[1], &[0], &[0, 1]).unwrap();
    assert_eq!(DIAGONAL_MATERIALIZATIONS.get(), 0);
}

#[test]
fn compact_braid_off_diagonal_zeros_are_numerically_zero() {
    // This finite witness exercises the one-term admission; structural zeros
    // have no prescribed sign under a fermionic braid.
    use tenet_core::PreparedTreePairOperation;
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    macro_rules! probe {
        ($rule:expr, $sectors:expr, $negative_zeros:expr) => {{
            let leg = GradedSpace::try_new(Arc::new($rule), $sectors).unwrap();
            let tensor: TensorMap<_, f64> =
                TensorMap::rand_with_seed(&runtime, [&leg], [&leg], 1617).unwrap();
            let diagonal = tensor.svd_compact(&[0], &[1]).unwrap().s;
            let source = diagonal.logical_space().space().structure();
            let prepared = PreparedTreePairOperation::prepare_braid(
                diagonal.provider(),
                1,
                1,
                &[1],
                &[0],
                &[0],
                &[1],
            )
            .unwrap();
            let mut counts = Vec::new();
            let destination = diagonal
                .logical_space()
                .transformed_multiplicity_free(&TreeTransformOperation::braid([1], [0], [0], [1]))
                .unwrap();
            let destination_structure = destination.space().structure();
            let mut covered = vec![false; destination_structure.block_count()];
            for i in 0..source.block_count() {
                let source_block = source.block(i).unwrap();
                let pair = source_block.key().as_fusion_tree_pair().unwrap();
                let rows = prepared
                    .execute_multiplicity_free(diagonal.provider(), pair)
                    .unwrap();
                counts.push(rows.len());
                assert_eq!(rows.len(), 1);
                let (destination_pair, coefficient) = &rows[0];
                assert!(coefficient.is_finite() && *coefficient != 0.0);
                let destination_index = (0..destination_structure.block_count())
                    .find(|&j| {
                        destination_structure.block(j).unwrap().key()
                            == &tenet_core::BlockKey::FusionTree(destination_pair.clone())
                    })
                    .expect("braid destination exists");
                assert!(!std::mem::replace(&mut covered[destination_index], true));
                let destination_block = destination_structure.block(destination_index).unwrap();
                assert_eq!(source_block.shape(), destination_block.shape());
                assert_eq!(destination_block.shape()[0], destination_block.shape()[1]);
            }
            assert!(covered.iter().all(|&seen| seen));
            DIAGONAL_MATERIALIZATIONS.set(0);
            let result = diagonal.braid(&[1], &[0], &[0, 1]).unwrap();
            assert_eq!(DIAGONAL_MATERIALIZATIONS.get(), 0);
            let dense_oracle = diagonal
                .materialize()
                .unwrap()
                .braid(&[1], &[0], &[0, 1])
                .unwrap();
            for (&actual, &expected) in result
                .dense_data()
                .unwrap()
                .iter()
                .zip(dense_oracle.dense_data().unwrap())
            {
                assert!((actual - expected).abs() <= 32.0 * f64::EPSILON * expected.abs().max(1.0));
            }
            assert!(diagonal
                .diagview()
                .unwrap()
                .iter()
                .flat_map(|entry| &entry.values)
                .all(|&value| value != 0.0));
            let output = result.dense_data().unwrap();
            let structure = result.logical_space().space().structure();
            let mut negative_zeros = 0;
            for block_index in 0..structure.block_count() {
                let block = structure.block(block_index).unwrap();
                assert_eq!(block.shape().len(), 2);
                assert_eq!(block.shape()[0], block.shape()[1]);
                for column in 0..block.shape()[1] {
                    for row in 0..block.shape()[0] {
                        if row != column {
                            let offset = block.offset()
                                + row * block.strides()[0]
                                + column * block.strides()[1];
                            assert_eq!(output[offset], 0.0);
                            negative_zeros +=
                                usize::from(output[offset].to_bits() == (-0.0_f64).to_bits());
                        }
                    }
                }
            }
            assert!(counts.iter().all(|&count| count == 1));
            assert_eq!(negative_zeros, $negative_zeros);
        }};
    }
    probe!(
        U1FusionRule,
        [(U1Irrep::new(0), 3), (U1Irrep::new(1), 2)],
        0
    );
    probe!(
        SU2FusionRule,
        [
            (SU2Irrep::from_twice_spin(0), 3),
            (SU2Irrep::from_twice_spin(1), 2)
        ],
        0
    );
    probe!(
        FermionParityFusionRule,
        [(Z2Irrep::EVEN, 3), (Z2Irrep::ODD, 2)],
        0
    );
    probe!(
        FermionParityFusionRule.product(U1FusionRule),
        [
            (product_sector(Z2Irrep::EVEN, U1Irrep::new(0)), 3),
            (product_sector(Z2Irrep::ODD, U1Irrep::new(1)), 2),
        ],
        0
    );
}

#[test]
fn compact_fermionic_braid_matches_hand_diagonal_and_complex_sign() {
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let bond = GradedSpace::try_new(
        Arc::new(FermionParityFusionRule),
        [(Z2Irrep::EVEN, 2), (Z2Irrep::ODD, 3)],
    )
    .unwrap();
    let real: TensorMap<_, f64> = TensorMap::diagonal(
        &runtime,
        &bond,
        [
            SectorSpectrum {
                sector: Z2Irrep::EVEN,
                values: vec![2.0, -3.0],
            },
            SectorSpectrum {
                sector: Z2Irrep::ODD,
                values: vec![5.0, -7.0, 11.0],
            },
        ],
    )
    .unwrap();
    // Fibonacci has complex categorical coefficients, but its public compact
    // constructor is unavailable: TypedTensorRootDispatch requires R::Scalar=f64.
    // This Complex64 payload still exercises the supported real-coefficient lane.
    let complex = real
        .convert::<Complex64>()
        .map_diagonal(|value| Complex64::new(value.re, value.re / 4.0))
        .unwrap();
    let real_output = real.braid(&[1], &[0], &[0, 1]).unwrap();
    let complex_output = complex.braid(&[1], &[0], &[0, 1]).unwrap();
    let real_spectra = real_output.diagview().unwrap();
    let complex_spectra = complex_output.diagview().unwrap();
    let source_spectra = real.diagview().unwrap();
    assert_eq!(real_spectra.len(), source_spectra.len());
    assert_eq!(complex_spectra.len(), source_spectra.len());
    for (real_entry, complex_entry) in real_spectra.iter().zip(&complex_spectra) {
        assert_eq!(real_entry.sector, complex_entry.sector);
        let source = source_spectra
            .iter()
            .find(|entry| entry.sector == real_entry.sector)
            .unwrap();
        assert_eq!(real_entry.values.len(), source.values.len());
        assert_eq!(complex_entry.values.len(), source.values.len());
        let sign = if real_entry.sector == Z2Irrep::ODD {
            -1.0
        } else {
            1.0
        };
        for ((&real_value, &complex_value), &source_value) in real_entry
            .values
            .iter()
            .zip(&complex_entry.values)
            .zip(&source.values)
        {
            assert_eq!(real_value, sign * source_value);
            assert_eq!(
                complex_value,
                Complex64::new(sign * source_value, sign * source_value / 4.0)
            );
        }
    }
    let structure = real_output.logical_space().space().structure();
    let real_data = real_output.dense_data().unwrap();
    let complex_data = complex_output.dense_data().unwrap();
    for block_index in 0..structure.block_count() {
        let block = structure.block(block_index).unwrap();
        for column in 0..block.shape()[1] {
            for row in 0..block.shape()[0] {
                if row != column {
                    let offset =
                        block.offset() + row * block.strides()[0] + column * block.strides()[1];
                    assert_eq!(real_data[offset], 0.0);
                    assert_eq!(complex_data[offset], Complex64::new(0.0, 0.0));
                }
            }
        }
    }
}

#[test]
fn compact_cat_braid_absorb_avoid_dense_source_materializations() {
    let diagonal = fixture().svd_compact(&[0], &[1]).unwrap().s;
    DIAGONAL_MATERIALIZATIONS.set(0);
    let _ = diagonal.cat(&diagonal, Side::Domain).unwrap();
    let cat = DIAGONAL_MATERIALIZATIONS.get();
    DIAGONAL_MATERIALIZATIONS.set(0);
    let _ = diagonal.braid(&[1], &[0], &[0, 1]).unwrap();
    let braid = DIAGONAL_MATERIALIZATIONS.get();
    DIAGONAL_MATERIALIZATIONS.set(0);
    let _ = diagonal.absorb(&diagonal).unwrap();
    let absorb = DIAGONAL_MATERIALIZATIONS.get();
    assert_eq!([cat, braid, absorb], [0, 0, 0]);
}

#[test]
fn absorb_compact_source_zeros_shared_off_diagonal_and_preserves_outer_region() {
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let provider = Arc::new(Z2FusionRule);
    let wide = GradedSpace::try_new(Arc::clone(&provider), [(Z2Irrep::EVEN, 4)]).unwrap();
    let narrow = GradedSpace::try_new(provider, [(Z2Irrep::EVEN, 2)]).unwrap();
    let receiver: TensorMap<_, f64> =
        TensorMap::from_subblock_fn(&runtime, [&wide], [&wide], |_, index| {
            1.0 + index[0] as f64 + 10.0 * index[1] as f64
        })
        .unwrap();
    let source: TensorMap<_, f64> = TensorMap::diagonal(
        &runtime,
        &narrow,
        [SectorSpectrum {
            sector: Z2Irrep::EVEN,
            values: vec![2.0, -3.0],
        }],
    )
    .unwrap();
    let result = receiver.absorb(&source).unwrap();
    let data = result.dense_data().unwrap();
    let block = result.logical_space().space().structure().block(0).unwrap();
    assert_eq!(block.shape(), &[4, 4]);
    for column in 0..4 {
        for row in 0..4 {
            let expected = if row < 2 && column < 2 {
                if row == column {
                    [2.0, -3.0][row]
                } else {
                    0.0
                }
            } else {
                1.0 + row as f64 + 10.0 * column as f64
            };
            let offset = block.offset() + row * block.strides()[0] + column * block.strides()[1];
            assert_eq!(data[offset], expected, "row={row}, column={column}");
        }
    }
}

#[test]
fn compact_absorb_handles_strided_prefixes_and_interleaved_sectors() {
    let key = |sector| {
        let tree = FusionTreeKey::try_from_sector_ids_for_rule(
            &Z2FusionRule,
            [sector],
            sector,
            [false],
            [],
            [],
        )
        .unwrap();
        BlockKey::FusionTree(FusionTreePairKey::pair(tree.clone(), tree))
    };
    let source = BlockStructure::from_blocks(vec![
        BlockSpec::column_major_with_key(key(0), vec![1, 1], 0).unwrap(),
        BlockSpec::column_major_with_key(key(1), vec![2, 2], 1).unwrap(),
    ])
    .unwrap();
    let destination =
        BlockStructure::from_blocks(vec![
            BlockSpec::with_key(key(1), vec![2, 2], vec![2, 5], 0).unwrap()
        ])
        .unwrap();
    let spectrum = [
        tenet_matrixalgebra::SectorSpectrum {
            sector: SectorId::new(0),
            values: vec![5.0],
        },
        tenet_matrixalgebra::SectorSpectrum {
            sector: SectorId::new(1),
            values: vec![7.0, 9.0],
        },
    ];
    let mut values = vec![42.0; 8];
    absorb_compact_source(&destination, &mut values, &source, &spectrum).unwrap();
    assert_eq!(values, [7.0, 42.0, 0.0, 42.0, 42.0, 0.0, 42.0, 9.0]);

    // A destination-only lower sector is likewise untouched before the
    // shared strided block; this exercises both sides of the sector merge.
    let destination = BlockStructure::from_blocks(vec![
        BlockSpec::column_major_with_key(key(0), vec![1, 1], 0).unwrap(),
        BlockSpec::with_key(key(1), vec![2, 2], vec![2, 5], 1).unwrap(),
    ])
    .unwrap();
    let source = BlockStructure::from_blocks(vec![BlockSpec::column_major_with_key(
        key(1),
        vec![2, 2],
        0,
    )
    .unwrap()])
    .unwrap();
    let mut values = vec![42.0; 9];
    absorb_compact_source(&destination, &mut values, &source, &spectrum[1..]).unwrap();
    assert_eq!(values, [42.0, 7.0, 42.0, 0.0, 42.0, 42.0, 0.0, 42.0, 9.0]);

    let duplicate = [spectrum[1].clone(), spectrum[1].clone()];
    assert!(absorb_compact_source(&destination, &mut values, &source, &duplicate).is_err());
    assert!(absorb_compact_source(&destination, &mut values, &source, &[]).is_err());
    let wrong_shape = [tenet_matrixalgebra::SectorSpectrum {
        sector: SectorId::new(1),
        values: vec![7.0],
    }];
    assert!(absorb_compact_source(&destination, &mut values, &source, &wrong_shape).is_err());
}

#[test]
fn compact_cat_and_absorb_preserve_stored_bits_and_zero_structural_cells() {
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let bond = GradedSpace::try_new(Arc::new(Z2FusionRule), [(Z2Irrep::EVEN, 2)]).unwrap();
    let nan = f64::from_bits(0x7ff8_0000_0000_1617);
    let compact: TensorMap<_, f64> = TensorMap::diagonal(
        &runtime,
        &bond,
        [SectorSpectrum {
            sector: Z2Irrep::EVEN,
            values: vec![-0.0, nan],
        }],
    )
    .unwrap();
    let dense = compact.materialize().unwrap();
    let compact_cat = compact.cat(&compact, Side::Domain).unwrap();
    let dense_cat = dense.cat(&dense, Side::Domain).unwrap();
    assert_eq!(
        compact_cat
            .dense_data()
            .unwrap()
            .iter()
            .map(|x| x.to_bits())
            .collect::<Vec<_>>(),
        dense_cat
            .dense_data()
            .unwrap()
            .iter()
            .map(|x| x.to_bits())
            .collect::<Vec<_>>()
    );
    assert!(compact_cat
        .dense_data()
        .unwrap()
        .iter()
        .any(|x| x.to_bits() == (-0.0f64).to_bits()));
    assert!(compact_cat
        .dense_data()
        .unwrap()
        .iter()
        .any(|x| x.to_bits() == nan.to_bits()));
    assert!(compact_cat
        .dense_data()
        .unwrap()
        .iter()
        .any(|x| x.to_bits() == 0.0f64.to_bits()));

    let receiver: TensorMap<_, f64> =
        TensorMap::from_subblock_fn(&runtime, [&bond], [&bond], |_, index| {
            10.0 + index[0] as f64 + index[1] as f64
        })
        .unwrap();
    let compact_absorb = receiver.absorb(&compact).unwrap();
    let dense_absorb = receiver.absorb(&dense).unwrap();
    assert_eq!(
        compact_absorb
            .dense_data()
            .unwrap()
            .iter()
            .map(|x| x.to_bits())
            .collect::<Vec<_>>(),
        dense_absorb
            .dense_data()
            .unwrap()
            .iter()
            .map(|x| x.to_bits())
            .collect::<Vec<_>>()
    );
    let block = compact_absorb
        .logical_space()
        .space()
        .structure()
        .block(0)
        .unwrap();
    let values = compact_absorb.dense_data().unwrap();
    let offset =
        |row, column| block.offset() + row * block.strides()[0] + column * block.strides()[1];
    assert_eq!(values[offset(0, 0)].to_bits(), (-0.0f64).to_bits());
    assert_eq!(values[offset(1, 1)].to_bits(), nan.to_bits());
    assert_eq!(values[offset(0, 1)].to_bits(), 0.0f64.to_bits());
    assert_eq!(values[offset(1, 0)].to_bits(), 0.0f64.to_bits());
}

/// A destination whose device is not the Runtime's is rejected before any
/// device work, its bytes untouched (#1551). The public API cannot build
/// one — a tensor's storage lives on its own Runtime's device, and the
/// Runtime check comes first — so this in-crate gate forges it from a
/// second device's tensor.
#[cfg(feature = "cuda")]
#[test]
#[ignore = "requires two real CUDA devices"]
fn typed_cuda_into_rejects_a_foreign_device_destination_untouched() {
    let rt0 = Runtime::builder().cuda(0).dense_threads(1).build().unwrap();
    let rt1 = Runtime::builder().cuda(1).dense_threads(1).build().unwrap();
    let v = GradedSpace::try_new(
        Arc::new(U1FusionRule),
        [
            (U1Irrep::new(0), 2),
            (U1Irrep::new(1), 1),
            (U1Irrep::new(-1), 2),
        ],
    )
    .unwrap();
    let w = v.try_dual().unwrap();
    let host_t = TensorMap::<_, f64>::rand_with_seed(&rt0, [&v, &w], [&v, &w], 1).unwrap();
    let host_rhs = TensorMap::<_, f64>::rand_with_seed(&rt0, [&v, &w], [&v], 2).unwrap();
    let (t, rhs) = (host_t.to_cuda().unwrap(), host_rhs.to_cuda().unwrap());
    let spec = ContractSpec {
        lhs: &[2, 3],
        rhs: &[0, 1],
        codomain: &[2, 0],
        domain: &[1],
    };
    let foreign = |like: TensorMap<U1FusionRule, f64>| {
        let (codomain, domain) = (like.codomain(), like.domain());
        let mut next = 0u64;
        let mut device = TensorMap::<_, f64>::from_subblock_fn(&rt1, &codomain, &domain, |_, _| {
            next += 1;
            std::f64::consts::PI * next as f64
        })
        .unwrap()
        .to_cuda()
        .unwrap();
        let before: Vec<u64> = device
            .to_host()
            .unwrap()
            .dense_data()
            .unwrap()
            .iter()
            .map(|x| x.to_bits())
            .collect();
        device.runtime = rt0.clone();
        (device, before)
    };
    type Device = TensorMap<U1FusionRule, f64, CudaStorage<f64>>;
    let check = |what: &str,
                 (mut device, before): (Device, Vec<u64>),
                 call: &dyn Fn(&mut Device) -> Result<(), Error>| {
        assert_eq!(call(&mut device), Err(Error::PlacementMismatch), "{what}");
        device.runtime = rt1.clone();
        let after: Vec<u64> = device
            .to_host()
            .unwrap()
            .dense_data()
            .unwrap()
            .iter()
            .map(|x| x.to_bits())
            .collect();
        assert_eq!(after, before, "{what}: destination changed");
    };
    check(
        "permute_into",
        foreign(host_t.permute(&[2, 0], &[1, 3]).unwrap()),
        &|d| t.permute_into(&[2, 0], &[1, 3], d, 1.0, 0.5),
    );
    check(
        "braid_into",
        foreign(host_t.braid(&[1, 0], &[3, 2], &[0, 1, 2, 3]).unwrap()),
        &|d| t.braid_into(&[1, 0], &[3, 2], &[0, 1, 2, 3], d, 1.0, 0.5),
    );
    check(
        "transpose_into",
        foreign(host_t.transpose(&[1, 3], &[0, 2]).unwrap()),
        &|d| t.transpose_into(&[1, 3], &[0, 2], d, 1.0, 0.5),
    );
    check(
        "repartition_into",
        foreign(host_t.repartition(1).unwrap()),
        &|d| t.repartition_into(d, 1.0, 0.5),
    );
    check(
        "trace_pairs_into",
        foreign(host_t.trace_pairs(&[(0, 2)]).unwrap()),
        &|d| t.trace_pairs_into(&[(0, 2)], d, 1.0, 0.5),
    );
    check(
        "contract_into",
        foreign(host_t.contract(&host_rhs, &spec).unwrap()),
        &|d| t.contract_into(&rhs, &spec, d, 1.0, 0.5),
    );
    check("axpby_into", foreign(host_t.clone()), &|d| {
        t.axpby_into(d, 1.0, 0.5)
    });
}
