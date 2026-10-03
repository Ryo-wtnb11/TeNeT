// The one `DenseExecutor` test double (#1825): every kernel entry forwards to
// the compiled `DefaultDenseExecutor` and is counted where it enters the
// executor; selected kernels can fail on their n-th call, and a strict spy
// panics on any kernel outside the set a test allows. Included with
// `include!` so the integration tests and the crate's own unit tests share it;
// the includer brings the dense names into scope (`tenet::expert::*` in
// integration tests, `tenet_dense::*` in the crate):
// `DefaultDenseExecutor, DenseBackend, DenseDotConfig, DenseError,
// DenseExecutor, DenseGemmBatchJob, DenseRead, DenseScalar, DenseTensor,
// DenseWrite, MatrixOp`.
//
// Why the capability entries stay at the trait default instead of forwarding:
// `supports_svd_full` stays false and `with_linalg_scope`/`factorize_batch`
// run their bodies through this spy, so every kernel the operation reaches is
// observed here rather than inside the wrapped backend.

/// One counted executor entry point.
#[allow(dead_code)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Kernel {
    Svd,
    SvdInto,
    SvdVals,
    Qr,
    QrInto,
    Eigh,
    EighInto,
    EighVals,
    Eig,
    EigVals,
    Solve,
    DotGeneral,
    Matmul,
    MatmulAxpby,
    MatmulBatch,
    MatmulBatchOps,
}

#[allow(dead_code)]
impl Kernel {
    const ALL: [Kernel; 16] = [
        Kernel::Svd,
        Kernel::SvdInto,
        Kernel::SvdVals,
        Kernel::Qr,
        Kernel::QrInto,
        Kernel::Eigh,
        Kernel::EighInto,
        Kernel::EighVals,
        Kernel::Eig,
        Kernel::EigVals,
        Kernel::Solve,
        Kernel::DotGeneral,
        Kernel::Matmul,
        Kernel::MatmulAxpby,
        Kernel::MatmulBatch,
        Kernel::MatmulBatchOps,
    ];
    /// The full SVD entries (owned and destination form).
    const SVD: &'static [Kernel] = &[Kernel::Svd, Kernel::SvdInto];
    const QR: &'static [Kernel] = &[Kernel::Qr, Kernel::QrInto];
    const EIGH: &'static [Kernel] = &[Kernel::Eigh, Kernel::EighInto];
    /// Every GEMM entry.
    const GEMM: &'static [Kernel] = &[
        Kernel::DotGeneral,
        Kernel::Matmul,
        Kernel::MatmulAxpby,
        Kernel::MatmulBatch,
        Kernel::MatmulBatchOps,
    ];

    /// The executor method name, used as the `op` of an injected error.
    fn op(self) -> &'static str {
        match self {
            Kernel::Svd => "svd",
            Kernel::SvdInto => "svd_into",
            Kernel::SvdVals => "svd_vals",
            Kernel::Qr => "qr",
            Kernel::QrInto => "qr_into",
            Kernel::Eigh => "eigh",
            Kernel::EighInto => "eigh_into",
            Kernel::EighVals => "eigh_vals",
            Kernel::Eig => "eig",
            Kernel::EigVals => "eig_vals",
            Kernel::Solve => "solve_into",
            Kernel::DotGeneral => "dot_general_into",
            Kernel::Matmul => "matmul_into",
            Kernel::MatmulAxpby => "matmul_axpby_into",
            Kernel::MatmulBatch => "matmul_batch_axpby_into",
            Kernel::MatmulBatchOps => "matmul_batch_axpby_with_ops_into",
        }
    }
}

/// Per-entry call counts, shared with the test through an `Arc`.
#[allow(dead_code)]
#[derive(Default)]
struct SpyCounts {
    calls: [std::sync::atomic::AtomicUsize; 16],
    /// Jobs submitted through the batched GEMM entries.
    batch_jobs: std::sync::atomic::AtomicUsize,
}

#[allow(dead_code)]
impl SpyCounts {
    fn get(&self, kernel: Kernel) -> usize {
        self.calls[kernel as usize].load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Calls summed over `kernels`.
    fn of(&self, kernels: &[Kernel]) -> usize {
        kernels.iter().map(|&kernel| self.get(kernel)).sum()
    }

    /// Calls of every entry not in `kernels`: zero proves an operation reached
    /// only the listed kernels.
    fn others(&self, kernels: &[Kernel]) -> usize {
        Kernel::ALL
            .iter()
            .filter(|kernel| !kernels.contains(kernel))
            .map(|&kernel| self.get(kernel))
            .sum()
    }

    /// Calls of every entry.
    fn total(&self) -> usize {
        self.others(&[])
    }

    /// Zeroes every count, for a test that measures several phases.
    fn reset(&self) {
        for calls in &self.calls {
            calls.store(0, std::sync::atomic::Ordering::Relaxed);
        }
        self.batch_jobs
            .store(0, std::sync::atomic::Ordering::Relaxed);
    }

    fn batch_jobs(&self) -> usize {
        self.batch_jobs.load(std::sync::atomic::Ordering::Relaxed)
    }
}

/// Fails the `nth` call (1-based, counted over `kernels` together), or every
/// call when `nth` is `None`: with `DenseError::Backend { op, message }`, or
/// with `DenseError::Unsupported` when `unsupported` (a missing capability).
#[allow(dead_code)]
struct SpyFault {
    kernels: &'static [Kernel],
    nth: Option<usize>,
    message: &'static str,
    unsupported: bool,
}

#[allow(dead_code)]
#[derive(Default)]
struct SpyExecutor {
    inner: DefaultDenseExecutor,
    counts: std::sync::Arc<SpyCounts>,
    faults: Vec<SpyFault>,
    /// `(allowed kernels, reason)`: any other entry panics with `reason`.
    only: Option<(&'static [Kernel], &'static str)>,
}

#[allow(dead_code)]
impl SpyExecutor {
    /// A spy counting into `counts`.
    fn counting(counts: &std::sync::Arc<SpyCounts>) -> Self {
        Self {
            counts: std::sync::Arc::clone(counts),
            ..Self::default()
        }
    }

    /// Panics with `reason` on any entry outside `kernels`, so a test proves
    /// the operation reaches no other dense kernel.
    fn only(mut self, kernels: &'static [Kernel], reason: &'static str) -> Self {
        self.only = Some((kernels, reason));
        self
    }

    /// Adds a fault; see [`SpyFault`].
    fn failing(
        mut self,
        kernels: &'static [Kernel],
        nth: Option<usize>,
        message: &'static str,
    ) -> Self {
        self.faults.push(SpyFault {
            kernels,
            nth,
            message,
            unsupported: false,
        });
        self
    }

    /// Makes `kernels` report `DenseError::Unsupported`, as an executor that
    /// leaves them at the trait default does.
    fn without(mut self, kernels: &'static [Kernel], message: &'static str) -> Self {
        self.faults.push(SpyFault {
            kernels,
            nth: None,
            message,
            unsupported: true,
        });
        self
    }

    fn enter(&self, kernel: Kernel) -> Result<(), DenseError> {
        if let Some((allowed, reason)) = self.only {
            assert!(allowed.contains(&kernel), "{reason}: unexpected {kernel:?}");
        }
        self.counts.calls[kernel as usize].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        for fault in &self.faults {
            if fault.kernels.contains(&kernel)
                && fault
                    .nth
                    .is_none_or(|nth| self.counts.of(fault.kernels) == nth)
            {
                let (op, message) = (kernel.op(), fault.message.to_string());
                return Err(if fault.unsupported {
                    DenseError::Unsupported { op, message }
                } else {
                    DenseError::Backend {
                        backend: DenseBackend::Tenferro,
                        op,
                        message,
                    }
                });
            }
        }
        Ok(())
    }
}

impl DenseExecutor for SpyExecutor {
    fn svd(&mut self, input: DenseRead<'_>) -> Result<Vec<DenseTensor>, DenseError> {
        self.enter(Kernel::Svd)?;
        self.inner.svd(input)
    }
    fn svd_into(
        &mut self,
        input: DenseRead<'_>,
        u: DenseWrite<'_>,
        s: DenseWrite<'_>,
        vt: DenseWrite<'_>,
    ) -> Result<(), DenseError> {
        self.enter(Kernel::SvdInto)?;
        self.inner.svd_into(input, u, s, vt)
    }
    fn svd_vals(&mut self, input: DenseRead<'_>) -> Result<DenseTensor, DenseError> {
        self.enter(Kernel::SvdVals)?;
        self.inner.svd_vals(input)
    }
    fn qr(&mut self, input: DenseRead<'_>) -> Result<Vec<DenseTensor>, DenseError> {
        self.enter(Kernel::Qr)?;
        self.inner.qr(input)
    }
    fn qr_into(
        &mut self,
        input: DenseRead<'_>,
        q: DenseWrite<'_>,
        r: DenseWrite<'_>,
    ) -> Result<(), DenseError> {
        self.enter(Kernel::QrInto)?;
        self.inner.qr_into(input, q, r)
    }
    fn eigh(&mut self, input: DenseRead<'_>) -> Result<Vec<DenseTensor>, DenseError> {
        self.enter(Kernel::Eigh)?;
        self.inner.eigh(input)
    }
    fn eigh_into(
        &mut self,
        input: DenseRead<'_>,
        values: DenseWrite<'_>,
        vectors: DenseWrite<'_>,
    ) -> Result<(), DenseError> {
        self.enter(Kernel::EighInto)?;
        self.inner.eigh_into(input, values, vectors)
    }
    fn eigh_vals(&mut self, input: DenseRead<'_>) -> Result<DenseTensor, DenseError> {
        self.enter(Kernel::EighVals)?;
        self.inner.eigh_vals(input)
    }
    fn eig(&mut self, input: DenseRead<'_>) -> Result<Vec<DenseTensor>, DenseError> {
        self.enter(Kernel::Eig)?;
        self.inner.eig(input)
    }
    fn eig_vals(&mut self, input: DenseRead<'_>) -> Result<DenseTensor, DenseError> {
        self.enter(Kernel::EigVals)?;
        self.inner.eig_vals(input)
    }
    fn solve_into(
        &mut self,
        a: DenseRead<'_>,
        b: DenseRead<'_>,
        x: DenseWrite<'_>,
    ) -> Result<(), DenseError> {
        self.enter(Kernel::Solve)?;
        self.inner.solve_into(a, b, x)
    }
    fn dot_general_into(
        &mut self,
        output: DenseWrite<'_>,
        lhs: DenseRead<'_>,
        rhs: DenseRead<'_>,
        config: &DenseDotConfig,
    ) -> Result<(), DenseError> {
        self.enter(Kernel::DotGeneral)?;
        self.inner.dot_general_into(output, lhs, rhs, config)
    }
    fn matmul_into(
        &mut self,
        output: DenseWrite<'_>,
        lhs: DenseRead<'_>,
        rhs: DenseRead<'_>,
    ) -> Result<(), DenseError> {
        self.enter(Kernel::Matmul)?;
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
        self.enter(Kernel::MatmulAxpby)?;
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
        self.enter(Kernel::MatmulBatch)?;
        self.counts
            .batch_jobs
            .fetch_add(jobs.len(), std::sync::atomic::Ordering::Relaxed);
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
        self.enter(Kernel::MatmulBatchOps)?;
        self.counts
            .batch_jobs
            .fetch_add(jobs.len(), std::sync::atomic::Ordering::Relaxed);
        self.inner.matmul_batch_axpby_with_ops_into(
            output, lhs, rhs, jobs, runs, lhs_op, rhs_op, alpha, beta,
        )
    }
}
