//! `RuntimeBuilder::with_dense_executor` lets a caller select the CPU
//! linear-algebra backend by injecting a `DenseExecutor` (issue #64). This
//! checks the runtime actually drives the injected executor and that doing so
//! is numerically identical to the faer default.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use tenet::dense::{
    DefaultDenseExecutor, DenseDotConfig, DenseError, DenseExecutor, DenseGemmBatchJob, DenseRead,
    DenseScalar, DenseTensor, DenseWrite, MatrixOp,
};
use tenet::prelude::{GradedSpace, Runtime, TensorMap, U1FusionRule, U1Irrep};

fn u1_space(entries: [(i32, usize); 3]) -> GradedSpace<U1FusionRule> {
    GradedSpace::try_new(
        Arc::new(U1FusionRule),
        entries.map(|(charge, degeneracy)| (U1Irrep::new(charge), degeneracy)),
        false,
    )
    .unwrap()
}

/// Per-kernel call counts, so a test can prove both that the injected backend
/// is the one the runtime drives and that a storage-local route never reaches
/// it. Every entry point is counted where it enters the executor, not where it
/// bottoms out: the default `DenseExecutor` funnels some of the GEMM family
/// into `dot_general_into`, but a backend is free to override each one.
#[derive(Default)]
struct SpyCounts {
    svd: AtomicUsize,
    eigh: AtomicUsize,
    gemm: AtomicUsize,
    solve: AtomicUsize,
}

impl SpyCounts {
    fn read(&self) -> (usize, usize, usize, usize) {
        (
            self.svd.load(Ordering::Relaxed),
            self.eigh.load(Ordering::Relaxed),
            self.gemm.load(Ordering::Relaxed),
            self.solve.load(Ordering::Relaxed),
        )
    }
}

/// Delegates every dense op to the faer default, counting the calls so the
/// tests can see which kernels a public operation drives.
struct SpyExecutor {
    inner: DefaultDenseExecutor,
    counts: Arc<SpyCounts>,
}

impl DenseExecutor for SpyExecutor {
    fn svd(&mut self, input: DenseRead<'_>) -> Result<Vec<DenseTensor>, DenseError> {
        self.counts.svd.fetch_add(1, Ordering::Relaxed);
        self.inner.svd(input)
    }
    fn qr(&mut self, input: DenseRead<'_>) -> Result<Vec<DenseTensor>, DenseError> {
        self.inner.qr(input)
    }
    fn eigh(&mut self, input: DenseRead<'_>) -> Result<Vec<DenseTensor>, DenseError> {
        self.counts.eigh.fetch_add(1, Ordering::Relaxed);
        self.inner.eigh(input)
    }
    fn eigh_into(
        &mut self,
        input: DenseRead<'_>,
        values: DenseWrite<'_>,
        vectors: DenseWrite<'_>,
    ) -> Result<(), DenseError> {
        self.counts.eigh.fetch_add(1, Ordering::Relaxed);
        self.inner.eigh_into(input, values, vectors)
    }
    fn eigh_vals(&mut self, input: DenseRead<'_>) -> Result<DenseTensor, DenseError> {
        self.counts.eigh.fetch_add(1, Ordering::Relaxed);
        self.inner.eigh_vals(input)
    }
    fn solve_into(
        &mut self,
        a: DenseRead<'_>,
        b: DenseRead<'_>,
        x: DenseWrite<'_>,
    ) -> Result<(), DenseError> {
        self.counts.solve.fetch_add(1, Ordering::Relaxed);
        self.inner.solve_into(a, b, x)
    }
    fn dot_general_into(
        &mut self,
        output: DenseWrite<'_>,
        lhs: DenseRead<'_>,
        rhs: DenseRead<'_>,
        config: &DenseDotConfig,
    ) -> Result<(), DenseError> {
        self.counts.gemm.fetch_add(1, Ordering::Relaxed);
        self.inner.dot_general_into(output, lhs, rhs, config)
    }
    fn matmul_into(
        &mut self,
        output: DenseWrite<'_>,
        lhs: DenseRead<'_>,
        rhs: DenseRead<'_>,
    ) -> Result<(), DenseError> {
        self.counts.gemm.fetch_add(1, Ordering::Relaxed);
        self.inner.matmul_into(output, lhs, rhs)
    }
    fn matmul_axpby_into(
        &mut self,
        output: DenseWrite<'_>,
        lhs: DenseRead<'_>,
        rhs: DenseRead<'_>,
        alpha: DenseScalar,
        beta: DenseScalar,
    ) -> Result<(), DenseError> {
        self.counts.gemm.fetch_add(1, Ordering::Relaxed);
        self.inner.matmul_axpby_into(output, lhs, rhs, alpha, beta)
    }
    #[allow(clippy::too_many_arguments)]
    fn matmul_batch_axpby_into(
        &mut self,
        output: DenseWrite<'_>,
        lhs: DenseRead<'_>,
        rhs: DenseRead<'_>,
        jobs: &[DenseGemmBatchJob],
        runs: &[usize],
        alpha: DenseScalar,
        beta: DenseScalar,
    ) -> Result<(), DenseError> {
        self.counts.gemm.fetch_add(1, Ordering::Relaxed);
        self.inner
            .matmul_batch_axpby_into(output, lhs, rhs, jobs, runs, alpha, beta)
    }
    #[allow(clippy::too_many_arguments)]
    fn matmul_batch_axpby_with_ops_into(
        &mut self,
        output: DenseWrite<'_>,
        lhs: DenseRead<'_>,
        rhs: DenseRead<'_>,
        jobs: &[DenseGemmBatchJob],
        runs: &[usize],
        lhs_op: MatrixOp,
        rhs_op: MatrixOp,
        alpha: DenseScalar,
        beta: DenseScalar,
    ) -> Result<(), DenseError> {
        self.counts.gemm.fetch_add(1, Ordering::Relaxed);
        self.inner.matmul_batch_axpby_with_ops_into(
            output, lhs, rhs, jobs, runs, lhs_op, rhs_op, alpha, beta,
        )
    }
}

#[test]
fn injected_dense_executor_is_used_and_preserves_results() {
    let counts = Arc::new(SpyCounts::default());
    let spy = SpyExecutor {
        inner: DefaultDenseExecutor::default(),
        counts: Arc::clone(&counts),
    };
    let rt = Runtime::builder()
        .with_dense_executor(Box::new(spy))
        .build()
        .unwrap();

    let v = u1_space([(-1, 2), (0, 2), (1, 1)]);
    let t = TensorMap::<U1FusionRule, f64>::rand_with_seed(&rt, [&v, &v], [&v, &v], 99).unwrap();
    let (_, s, _) = t.svd_compact().unwrap();

    assert!(
        counts.svd.load(Ordering::Relaxed) > 0,
        "the injected executor's svd was never called — the runtime is not \
         driving the injected backend"
    );

    // No behavior change: the same seeded tensor on the faer default runtime
    // yields identical singular values.
    let rt_default = Runtime::builder().build().unwrap();
    let t_default =
        TensorMap::<U1FusionRule, f64>::rand_with_seed(&rt_default, [&v, &v], [&v, &v], 99)
            .unwrap();
    let (_, s_default, _) = t_default.svd_compact().unwrap();
    assert_eq!(s.data().len(), s_default.data().len());
    for (a, b) in s.data().iter().zip(s_default.data()) {
        assert!(
            (a - b).abs() <= 1e-12 * (1.0 + a.abs()),
            "singular value differs from the default backend: {a} vs {b}"
        );
    }
}

#[test]
fn compact_diagonal_exp_drives_no_dense_kernel() {
    // Issue #578: `exp` on compact diagonal storage is elementwise on the
    // stored spectrum, so the injected backend sees nothing at all — not the
    // Hermitian eigendecomposition the dense route runs, not the GEMM that
    // reassembles `V exp(D) V^H`, not a solve. The spy is the direct
    // observation of that; the allocation gates only see its cost.
    let counts = Arc::new(SpyCounts::default());
    let rt = Runtime::builder()
        .with_dense_executor(Box::new(SpyExecutor {
            inner: DefaultDenseExecutor::default(),
            counts: Arc::clone(&counts),
        }))
        .build()
        .unwrap();

    let v = u1_space([(-1, 3), (0, 4), (1, 3)]);
    let t = TensorMap::<U1FusionRule, f64>::rand_with_seed(&rt, [&v], [&v], 578).unwrap();
    let s = t.svd_compact().unwrap().1;
    let (svd, ..) = counts.read();
    assert!(svd > 0, "the fixture never reached the injected backend");

    let before = counts.read();
    let image = s.exp().unwrap();
    let (_, eigh, gemm, solve) = counts.read();
    assert_eq!(
        (eigh, gemm, solve),
        (before.1, before.2, before.3),
        "compact exp drove a dense kernel"
    );

    // And it computed the right thing: every singular value exponentiated.
    // Compared per sector, since `svd_vals` sorts within a sector, and the
    // exponential is monotone so the order carries over.
    let source_values = s.svd_vals().unwrap();
    let image_values = image.svd_vals().unwrap();
    assert_eq!(source_values.len(), image_values.len());
    for (source, image) in source_values.iter().zip(&image_values) {
        assert_eq!(source.sector, image.sector);
        assert_eq!(source.values.len(), image.values.len());
        for (source, image) in source.values.iter().zip(&image.values) {
            let expected = source.exp();
            assert!(
                (expected - image).abs() <= 1e-12 * (1.0 + expected.abs()),
                "exp({source}) is {expected}, got {image}"
            );
        }
    }
}
