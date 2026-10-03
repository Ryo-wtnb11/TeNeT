use super::*;

pub(super) struct RejectExecutorCalls;

pub(super) struct FailComposition;

#[derive(Default)]
pub(super) struct SvdCallSpy {
    pub(super) inner: tenet_dense::DefaultDenseExecutor,
    pub(super) svd_calls: usize,
}

#[derive(Default)]
pub(super) struct RejectSvdInto {
    pub(super) inner: tenet_dense::DefaultDenseExecutor,
    pub(super) svd_calls: usize,
    pub(super) svd_into_calls: usize,
    pub(super) output_ptrs: Vec<(usize, usize)>,
    pub(super) gemm_ptrs: Vec<(usize, usize)>,
    pub(super) gemm_views: Vec<(DenseReadView, DenseReadView, bool, bool)>,
}

#[derive(Debug)]
pub(super) struct DenseReadView {
    pub(super) shape: Vec<usize>,
    pub(super) strides: Vec<usize>,
    pub(super) offset: usize,
}

#[derive(Default)]
pub(super) struct RejectEighInto {
    pub(super) inner: tenet_dense::DefaultDenseExecutor,
    pub(super) eigh_calls: usize,
    pub(super) eigh_into_calls: usize,
    pub(super) vector_ptrs: Vec<usize>,
}

fn dense_tensor_pointer(tensor: &DenseTensor) -> usize {
    if let Ok(data) = tensor.as_f32_slice() {
        return data.as_ptr() as usize;
    }
    if let Ok(data) = tensor.as_f64_slice() {
        return data.as_ptr() as usize;
    }
    if let Ok(data) = tensor.as_c32_slice() {
        return data.as_ptr() as usize;
    }
    if let Ok(data) = tensor.as_c64_slice() {
        return data.as_ptr() as usize;
    }
    panic!("compact SVD fixture must return a supported host dtype")
}

fn dense_read_pointer(read: &DenseRead<'_>) -> usize {
    match read {
        DenseRead::F32(view) => view.data().as_ptr() as usize,
        DenseRead::F64(view) => view.data().as_ptr() as usize,
        DenseRead::I32(view) => view.data().as_ptr() as usize,
        DenseRead::I64(view) => view.data().as_ptr() as usize,
        DenseRead::Bool(view) => view.data().as_ptr() as usize,
        DenseRead::C32(view) => view.data().as_ptr() as usize,
        DenseRead::C64(view) => view.data().as_ptr() as usize,
    }
}

fn dense_read_view(read: &DenseRead<'_>) -> DenseReadView {
    let (shape, strides, offset) = match read {
        DenseRead::F32(view) => (view.shape(), view.strides(), view.offset()),
        DenseRead::F64(view) => (view.shape(), view.strides(), view.offset()),
        DenseRead::I32(view) => (view.shape(), view.strides(), view.offset()),
        DenseRead::I64(view) => (view.shape(), view.strides(), view.offset()),
        DenseRead::Bool(view) => (view.shape(), view.strides(), view.offset()),
        DenseRead::C32(view) => (view.shape(), view.strides(), view.offset()),
        DenseRead::C64(view) => (view.shape(), view.strides(), view.offset()),
    };
    DenseReadView {
        shape: shape.to_vec(),
        strides: strides.to_vec(),
        offset,
    }
}

#[derive(Default)]
pub(super) struct SolveCallSpy {
    pub(super) inner: tenet_dense::DefaultDenseExecutor,
    pub(super) solve_calls: usize,
    pub(super) destination_ptrs: Vec<usize>,
}

#[derive(Default)]
pub(super) struct FailSecondSolve {
    pub(super) inner: tenet_dense::DefaultDenseExecutor,
    pub(super) solve_calls: usize,
}

#[derive(Default)]
pub(super) struct FailSecondSvd {
    pub(super) inner: tenet_dense::DefaultDenseExecutor,
    pub(super) calls: usize,
}

#[derive(Default)]
pub(super) struct FailAfterObservingSvdInput {
    pub(super) observed: Vec<Vec<f64>>,
    pub(super) outputs: Option<Vec<DenseTensor>>,
}

#[derive(Default)]
pub(super) struct FailAfterObservingQrInput {
    pub(super) inner: tenet_dense::DefaultDenseExecutor,
    pub(super) observed: Vec<Vec<f64>>,
    pub(super) qr_succeeds: bool,
    pub(super) outputs: Option<Vec<DenseTensor>>,
}

#[derive(Default)]
pub(super) struct FailAfterSvdQr {
    pub(super) inner: tenet_dense::DefaultDenseExecutor,
    pub(super) svd_calls: usize,
    pub(super) qr_calls: usize,
}

#[derive(Default)]
pub(super) struct FailAfterObservingEighInput {
    pub(super) observed: Vec<Vec<f64>>,
    pub(super) outputs: Option<Vec<DenseTensor>>,
}

#[derive(Default)]
pub(super) struct EighCallSpy {
    pub(super) calls: usize,
}

pub(super) struct NativeFullSvdSpy {
    pub(super) inner: tenet_dense::DefaultDenseExecutor,
    pub(super) full_calls: usize,
}

pub(super) struct FailSecondOwnedFullSvd {
    pub(super) inner: tenet_dense::DefaultDenseExecutor,
    pub(super) calls: usize,
}

/// The compiled default provider, so every feature set exercises the direct
/// full-SVD path it actually ships; `None` only where no provider has one.
fn native_full_svd_executor() -> Option<tenet_dense::DefaultDenseExecutor> {
    let mut executor = tenet_dense::DefaultDenseExecutor::new();
    if executor.supports_svd_full() {
        return Some(executor);
    }
    assert!(matches!(
        executor.svd_full_owned(DenseOwned::F64(vec![1.0]), 1, 1),
        Err(DenseError::Unsupported {
            op: "svd_full_owned",
            ..
        })
    ));
    None
}

impl NativeFullSvdSpy {
    pub(super) fn new() -> Option<Self> {
        native_full_svd_executor().map(|inner| Self {
            inner,
            full_calls: 0,
        })
    }
}

impl FailSecondOwnedFullSvd {
    pub(super) fn new() -> Option<Self> {
        native_full_svd_executor().map(|inner| Self { inner, calls: 0 })
    }
}

#[derive(Default)]
pub(super) struct RecordingEigh {
    pub(super) inner: tenet_dense::DefaultDenseExecutor,
    pub(super) raw_values: Vec<Vec<f64>>,
}

impl DenseExecutor for RejectExecutorCalls {
    fn svd(&mut self, _: DenseRead<'_>) -> Result<Vec<DenseTensor>, DenseError> {
        panic!("validation must reject the input before SVD execution")
    }

    fn qr(&mut self, _: DenseRead<'_>) -> Result<Vec<DenseTensor>, DenseError> {
        panic!("validation must reject the input before QR execution")
    }

    fn eigh(&mut self, _: DenseRead<'_>) -> Result<Vec<DenseTensor>, DenseError> {
        panic!("validation must reject the input before EIGH execution")
    }

    fn dot_general_into(
        &mut self,
        _: DenseWrite<'_>,
        _: DenseRead<'_>,
        _: DenseRead<'_>,
        _: &DenseDotConfig,
    ) -> Result<(), DenseError> {
        panic!("validation must reject the input before dense execution")
    }
}

impl DenseExecutor for FailComposition {
    fn svd(&mut self, _: DenseRead<'_>) -> Result<Vec<DenseTensor>, DenseError> {
        panic!("composition backend must not run SVD")
    }

    fn qr(&mut self, _: DenseRead<'_>) -> Result<Vec<DenseTensor>, DenseError> {
        panic!("composition backend must not run QR")
    }

    fn eigh(&mut self, _: DenseRead<'_>) -> Result<Vec<DenseTensor>, DenseError> {
        panic!("composition backend must not run EIGH")
    }

    fn dot_general_into(
        &mut self,
        _: DenseWrite<'_>,
        _: DenseRead<'_>,
        _: DenseRead<'_>,
        _: &DenseDotConfig,
    ) -> Result<(), DenseError> {
        Err(DenseError::Backend {
            backend: DenseBackend::Tenferro,
            op: "dot_general_into",
            message: "injected recomposition failure".to_string(),
        })
    }
}

impl DenseExecutor for SvdCallSpy {
    fn svd(&mut self, input: DenseRead<'_>) -> Result<Vec<DenseTensor>, DenseError> {
        self.svd_calls += 1;
        self.inner.svd(input)
    }

    fn qr(&mut self, input: DenseRead<'_>) -> Result<Vec<DenseTensor>, DenseError> {
        self.inner.qr(input)
    }

    fn eigh(&mut self, input: DenseRead<'_>) -> Result<Vec<DenseTensor>, DenseError> {
        self.inner.eigh(input)
    }

    fn dot_general_into(
        &mut self,
        output: DenseWrite<'_>,
        lhs: DenseRead<'_>,
        rhs: DenseRead<'_>,
        config: &DenseDotConfig,
    ) -> Result<(), DenseError> {
        self.inner.dot_general_into(output, lhs, rhs, config)
    }
}

impl DenseExecutor for NativeFullSvdSpy {
    fn svd(&mut self, _: DenseRead<'_>) -> Result<Vec<DenseTensor>, DenseError> {
        panic!("native full SVD must not use the legacy SVD route")
    }

    fn qr(&mut self, _: DenseRead<'_>) -> Result<Vec<DenseTensor>, DenseError> {
        panic!("native full SVD must not use orthonormal completion")
    }

    fn eigh(&mut self, _: DenseRead<'_>) -> Result<Vec<DenseTensor>, DenseError> {
        panic!("test only exercises full SVD")
    }

    fn supports_svd_full(&self) -> bool {
        true
    }

    fn svd_full_owned(
        &mut self,
        input: DenseOwned,
        rows: usize,
        cols: usize,
    ) -> Result<Vec<DenseTensor>, DenseError> {
        self.full_calls += 1;
        self.inner.svd_full_owned(input, rows, cols)
    }

    fn dot_general_into(
        &mut self,
        _: DenseWrite<'_>,
        _: DenseRead<'_>,
        _: DenseRead<'_>,
        _: &DenseDotConfig,
    ) -> Result<(), DenseError> {
        panic!("test only exercises full SVD")
    }
}

impl DenseExecutor for FailSecondOwnedFullSvd {
    fn svd(&mut self, _: DenseRead<'_>) -> Result<Vec<DenseTensor>, DenseError> {
        panic!("claimed native full SVD must not retry the legacy route")
    }
    fn qr(&mut self, _: DenseRead<'_>) -> Result<Vec<DenseTensor>, DenseError> {
        panic!("claimed native full SVD must not use completion")
    }
    fn eigh(&mut self, _: DenseRead<'_>) -> Result<Vec<DenseTensor>, DenseError> {
        panic!("test only exercises full SVD")
    }
    fn supports_svd_full(&self) -> bool {
        true
    }
    fn svd_full_owned(
        &mut self,
        input: DenseOwned,
        rows: usize,
        cols: usize,
    ) -> Result<Vec<DenseTensor>, DenseError> {
        self.calls += 1;
        if self.calls == 2 {
            return Err(DenseError::Backend {
                backend: DenseBackend::Tenferro,
                op: "svd_full_owned",
                message: "injected second-sector failure".to_string(),
            });
        }
        self.inner.svd_full_owned(input, rows, cols)
    }
    fn dot_general_into(
        &mut self,
        _: DenseWrite<'_>,
        _: DenseRead<'_>,
        _: DenseRead<'_>,
        _: &DenseDotConfig,
    ) -> Result<(), DenseError> {
        panic!("test only exercises full SVD")
    }
}

impl DenseExecutor for RejectSvdInto {
    fn svd(&mut self, input: DenseRead<'_>) -> Result<Vec<DenseTensor>, DenseError> {
        self.svd_calls += 1;
        let outputs = self.inner.svd(input)?;
        self.output_ptrs.push((
            dense_tensor_pointer(&outputs[0]),
            dense_tensor_pointer(&outputs[2]),
        ));
        Ok(outputs)
    }

    fn svd_into(
        &mut self,
        _: DenseRead<'_>,
        _: DenseWrite<'_>,
        _: DenseWrite<'_>,
        _: DenseWrite<'_>,
    ) -> Result<(), DenseError> {
        self.svd_into_calls += 1;
        Err(DenseError::Backend {
            backend: DenseBackend::Tenferro,
            op: "svd_into",
            message: "direct compact SVD must not use svd_into".to_string(),
        })
    }

    fn qr(&mut self, input: DenseRead<'_>) -> Result<Vec<DenseTensor>, DenseError> {
        self.inner.qr(input)
    }

    fn eigh(&mut self, input: DenseRead<'_>) -> Result<Vec<DenseTensor>, DenseError> {
        self.inner.eigh(input)
    }

    fn dot_general_into(
        &mut self,
        output: DenseWrite<'_>,
        lhs: DenseRead<'_>,
        rhs: DenseRead<'_>,
        config: &DenseDotConfig,
    ) -> Result<(), DenseError> {
        self.gemm_ptrs
            .push((dense_read_pointer(&lhs), dense_read_pointer(&rhs)));
        self.gemm_views.push((
            dense_read_view(&lhs),
            dense_read_view(&rhs),
            config.lhs_conj(),
            config.rhs_conj(),
        ));
        self.inner.dot_general_into(output, lhs, rhs, config)
    }
}

impl DenseExecutor for RejectEighInto {
    fn svd(&mut self, input: DenseRead<'_>) -> Result<Vec<DenseTensor>, DenseError> {
        self.inner.svd(input)
    }

    fn qr(&mut self, input: DenseRead<'_>) -> Result<Vec<DenseTensor>, DenseError> {
        self.inner.qr(input)
    }

    fn eigh(&mut self, input: DenseRead<'_>) -> Result<Vec<DenseTensor>, DenseError> {
        self.eigh_calls += 1;
        let outputs = self.inner.eigh(input)?;
        self.vector_ptrs.push(dense_tensor_pointer(&outputs[1]));
        Ok(outputs)
    }

    fn eigh_into(
        &mut self,
        _: DenseRead<'_>,
        _: DenseWrite<'_>,
        _: DenseWrite<'_>,
    ) -> Result<(), DenseError> {
        self.eigh_into_calls += 1;
        Err(DenseError::Backend {
            backend: DenseBackend::Tenferro,
            op: "eigh_into",
            message: "direct compact EIGH must not use eigh_into".to_string(),
        })
    }

    fn dot_general_into(
        &mut self,
        output: DenseWrite<'_>,
        lhs: DenseRead<'_>,
        rhs: DenseRead<'_>,
        config: &DenseDotConfig,
    ) -> Result<(), DenseError> {
        self.inner.dot_general_into(output, lhs, rhs, config)
    }
}

impl DenseExecutor for SolveCallSpy {
    fn svd(&mut self, _: DenseRead<'_>) -> Result<Vec<DenseTensor>, DenseError> {
        panic!("inverse must not execute an SVD")
    }

    fn qr(&mut self, input: DenseRead<'_>) -> Result<Vec<DenseTensor>, DenseError> {
        self.inner.qr(input)
    }

    fn eigh(&mut self, input: DenseRead<'_>) -> Result<Vec<DenseTensor>, DenseError> {
        self.inner.eigh(input)
    }

    fn solve_into(
        &mut self,
        a: DenseRead<'_>,
        b: DenseRead<'_>,
        x: DenseWrite<'_>,
    ) -> Result<(), DenseError> {
        self.solve_calls += 1;
        self.destination_ptrs.push(match &x {
            DenseWrite::F32(view) => view.data().as_ptr() as usize,
            DenseWrite::F64(view) => view.data().as_ptr() as usize,
            DenseWrite::I32(view) => view.data().as_ptr() as usize,
            DenseWrite::I64(view) => view.data().as_ptr() as usize,
            DenseWrite::Bool(view) => view.data().as_ptr() as usize,
            DenseWrite::C32(view) => view.data().as_ptr() as usize,
            DenseWrite::C64(view) => view.data().as_ptr() as usize,
        });
        self.inner.solve_into(a, b, x)
    }

    fn dot_general_into(
        &mut self,
        _: DenseWrite<'_>,
        _: DenseRead<'_>,
        _: DenseRead<'_>,
        _: &DenseDotConfig,
    ) -> Result<(), DenseError> {
        panic!("inverse must not recompose factors")
    }
}

impl DenseExecutor for FailSecondSolve {
    fn svd(&mut self, _: DenseRead<'_>) -> Result<Vec<DenseTensor>, DenseError> {
        panic!("inverse must not execute an SVD")
    }

    fn qr(&mut self, input: DenseRead<'_>) -> Result<Vec<DenseTensor>, DenseError> {
        self.inner.qr(input)
    }

    fn eigh(&mut self, input: DenseRead<'_>) -> Result<Vec<DenseTensor>, DenseError> {
        self.inner.eigh(input)
    }

    fn solve_into(
        &mut self,
        a: DenseRead<'_>,
        b: DenseRead<'_>,
        x: DenseWrite<'_>,
    ) -> Result<(), DenseError> {
        self.solve_calls += 1;
        if self.solve_calls == 2 {
            return Err(DenseError::Backend {
                backend: DenseBackend::Tenferro,
                op: "solve_into",
                message: "injected second-sector failure".to_string(),
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
        panic!("inverse must not recompose factors")
    }
}

impl DenseExecutor for FailAfterObservingSvdInput {
    fn svd(&mut self, input: DenseRead<'_>) -> Result<Vec<DenseTensor>, DenseError> {
        if let Some(outputs) = self.outputs.take() {
            return Ok(outputs);
        }
        let DenseRead::F64(input) = input else {
            panic!("test input must be f64")
        };
        self.observed.push(input.data().to_vec());
        if let Some(outputs) = self.outputs.take() {
            return Ok(outputs);
        }
        Err(DenseError::Backend {
            backend: DenseBackend::Tenferro,
            op: "svd_into",
            message: "injected failure".to_string(),
        })
    }

    fn svd_into(
        &mut self,
        input: DenseRead<'_>,
        u: DenseWrite<'_>,
        s: DenseWrite<'_>,
        vt: DenseWrite<'_>,
    ) -> Result<(), DenseError> {
        let DenseRead::F64(input) = input else {
            panic!("test input must be f64")
        };
        self.observed.push(input.data().to_vec());
        let DenseWrite::F64(u) = u else {
            panic!("test U must be f64")
        };
        let DenseWrite::F64(s) = s else {
            panic!("test singular values must be f64")
        };
        let DenseWrite::F64(vt) = vt else {
            panic!("test Vh must be f64")
        };
        assert!(u.data().iter().all(|&value| value == 0.0));
        assert!(s.data().iter().all(|&value| value == 0.0));
        assert!(vt.data().iter().all(|&value| value == 0.0));
        Err(DenseError::Backend {
            backend: DenseBackend::Tenferro,
            op: "svd_into",
            message: "injected failure".to_string(),
        })
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

impl DenseExecutor for FailSecondSvd {
    fn svd(&mut self, input: DenseRead<'_>) -> Result<Vec<DenseTensor>, DenseError> {
        self.calls += 1;
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
        if self.calls == 2 {
            return Err(DenseError::Backend {
                backend: DenseBackend::Tenferro,
                op: "svd_into",
                message: "injected second-sector failure".to_string(),
            });
        }
        self.inner.svd_into(input, u, s, vt)
    }

    fn qr(&mut self, input: DenseRead<'_>) -> Result<Vec<DenseTensor>, DenseError> {
        self.inner.qr(input)
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

impl DenseExecutor for FailAfterObservingQrInput {
    fn svd(&mut self, _: DenseRead<'_>) -> Result<Vec<DenseTensor>, DenseError> {
        panic!("test only exercises QR")
    }

    fn qr(&mut self, input: DenseRead<'_>) -> Result<Vec<DenseTensor>, DenseError> {
        if let Some(outputs) = self.outputs.take() {
            return Ok(outputs);
        }
        let DenseRead::F64(input) = input else {
            panic!("test input must be f64")
        };
        self.observed.push(input.data().to_vec());
        if self.qr_succeeds {
            self.inner.qr(DenseRead::F64(input))
        } else {
            Err(DenseError::Backend {
                backend: DenseBackend::Tenferro,
                op: "qr",
                message: "injected failure".to_string(),
            })
        }
    }

    fn qr_into(
        &mut self,
        input: DenseRead<'_>,
        q: DenseWrite<'_>,
        r: DenseWrite<'_>,
    ) -> Result<(), DenseError> {
        let _ = (input, q, r);
        Err(DenseError::Backend {
            backend: DenseBackend::Tenferro,
            op: "qr_into",
            message: "injected failure".to_string(),
        })
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

impl DenseExecutor for FailAfterSvdQr {
    fn svd(&mut self, input: DenseRead<'_>) -> Result<Vec<DenseTensor>, DenseError> {
        self.svd_calls += 1;
        self.inner.svd(input)
    }

    fn svd_into(
        &mut self,
        _: DenseRead<'_>,
        _: DenseWrite<'_>,
        _: DenseWrite<'_>,
        _: DenseWrite<'_>,
    ) -> Result<(), DenseError> {
        panic!("numerical null completion must use owned SVD outputs")
    }

    fn qr(&mut self, _: DenseRead<'_>) -> Result<Vec<DenseTensor>, DenseError> {
        self.qr_calls += 1;
        Err(DenseError::Backend {
            backend: DenseBackend::Tenferro,
            op: "qr",
            message: "injected completion failure".to_string(),
        })
    }

    fn eigh(&mut self, _: DenseRead<'_>) -> Result<Vec<DenseTensor>, DenseError> {
        panic!("test only exercises numerical null completion")
    }

    fn dot_general_into(
        &mut self,
        _: DenseWrite<'_>,
        _: DenseRead<'_>,
        _: DenseRead<'_>,
        _: &DenseDotConfig,
    ) -> Result<(), DenseError> {
        panic!("test only exercises numerical null completion")
    }
}

impl DenseExecutor for FailAfterObservingEighInput {
    fn svd(&mut self, _: DenseRead<'_>) -> Result<Vec<DenseTensor>, DenseError> {
        panic!("test only exercises EIGH")
    }

    fn qr(&mut self, _: DenseRead<'_>) -> Result<Vec<DenseTensor>, DenseError> {
        panic!("test only exercises EIGH")
    }

    fn eigh(&mut self, input: DenseRead<'_>) -> Result<Vec<DenseTensor>, DenseError> {
        let DenseRead::F64(input) = input else {
            panic!("test input must be f64")
        };
        self.observed.push(input.data().to_vec());
        if let Some(outputs) = self.outputs.take() {
            return Ok(outputs);
        }
        Err(DenseError::Backend {
            backend: DenseBackend::Tenferro,
            op: "eigh_into",
            message: "injected failure".to_string(),
        })
    }

    fn eigh_into(
        &mut self,
        input: DenseRead<'_>,
        values: DenseWrite<'_>,
        vectors: DenseWrite<'_>,
    ) -> Result<(), DenseError> {
        let DenseRead::F64(input) = input else {
            panic!("test input must be f64")
        };
        self.observed.push(input.data().to_vec());
        let DenseWrite::F64(values) = values else {
            panic!("test eigenvalues must be f64")
        };
        let DenseWrite::F64(vectors) = vectors else {
            panic!("test eigenvectors must be f64")
        };
        assert!(values.data().iter().all(|&value| value == 0.0));
        assert!(vectors.data().iter().all(|&value| value == 0.0));
        Err(DenseError::Backend {
            backend: DenseBackend::Tenferro,
            op: "eigh_into",
            message: "injected failure".to_string(),
        })
    }

    fn dot_general_into(
        &mut self,
        _: DenseWrite<'_>,
        _: DenseRead<'_>,
        _: DenseRead<'_>,
        _: &DenseDotConfig,
    ) -> Result<(), DenseError> {
        panic!("test only exercises EIGH")
    }
}

impl DenseExecutor for EighCallSpy {
    fn svd(&mut self, _: DenseRead<'_>) -> Result<Vec<DenseTensor>, DenseError> {
        panic!("test only exercises EIGH")
    }

    fn qr(&mut self, _: DenseRead<'_>) -> Result<Vec<DenseTensor>, DenseError> {
        panic!("test only exercises EIGH")
    }

    fn eigh(&mut self, _: DenseRead<'_>) -> Result<Vec<DenseTensor>, DenseError> {
        self.calls += 1;
        Err(DenseError::Backend {
            backend: DenseBackend::Tenferro,
            op: "eigh_into",
            message: "injected failure".to_string(),
        })
    }

    fn eigh_into(
        &mut self,
        _: DenseRead<'_>,
        _: DenseWrite<'_>,
        _: DenseWrite<'_>,
    ) -> Result<(), DenseError> {
        self.calls += 1;
        Err(DenseError::Backend {
            backend: DenseBackend::Tenferro,
            op: "eigh_into",
            message: "injected failure".to_string(),
        })
    }

    fn eigh_vals(&mut self, _: DenseRead<'_>) -> Result<DenseTensor, DenseError> {
        self.calls += 1;
        Err(DenseError::Backend {
            backend: DenseBackend::Tenferro,
            op: "eigh_vals",
            message: "injected failure".to_string(),
        })
    }

    fn dot_general_into(
        &mut self,
        _: DenseWrite<'_>,
        _: DenseRead<'_>,
        _: DenseRead<'_>,
        _: &DenseDotConfig,
    ) -> Result<(), DenseError> {
        panic!("test only exercises EIGH")
    }
}

impl DenseExecutor for RecordingEigh {
    fn svd(&mut self, _: DenseRead<'_>) -> Result<Vec<DenseTensor>, DenseError> {
        panic!("test only exercises EIGH")
    }

    fn qr(&mut self, _: DenseRead<'_>) -> Result<Vec<DenseTensor>, DenseError> {
        panic!("test only exercises EIGH")
    }

    fn eigh(&mut self, input: DenseRead<'_>) -> Result<Vec<DenseTensor>, DenseError> {
        let outputs = self.inner.eigh(input)?;
        self.raw_values.push(outputs[0].as_f64_slice()?.to_vec());
        Ok(outputs)
    }

    fn dot_general_into(
        &mut self,
        _: DenseWrite<'_>,
        _: DenseRead<'_>,
        _: DenseRead<'_>,
        _: &DenseDotConfig,
    ) -> Result<(), DenseError> {
        panic!("test only exercises EIGH")
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct LateGenericError(pub(super) usize);

impl fmt::Display for LateGenericError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "late Generic provider failure at call {}",
            self.0
        )
    }
}

impl std::error::Error for LateGenericError {}

pub(super) struct LateGenericSpy {
    pub(super) rule: FactorGenericRule,
    pub(super) fail_at: usize,
    pub(super) calls: Cell<usize>,
}

pub(super) struct CountingDense {
    pub(super) inner: tenet_dense::DefaultDenseExecutor,
    pub(super) svd_calls: usize,
    pub(super) svd_into_calls: usize,
    pub(super) svd_vals_calls: usize,
    pub(super) qr_calls: usize,
    pub(super) eig_calls: usize,
    pub(super) eigh_calls: usize,
}

#[derive(Debug)]
pub(super) struct FullQrObservation {
    pub(super) input_shape: Vec<usize>,
    pub(super) q_shape: Vec<usize>,
    pub(super) r_shape: Vec<usize>,
    pub(super) values: Vec<Complex64>,
}

pub(super) struct FullQrInputSpy {
    pub(super) inner: tenet_dense::DefaultDenseExecutor,
    pub(super) observations: Vec<FullQrObservation>,
}

impl Default for FullQrInputSpy {
    fn default() -> Self {
        Self {
            inner: tenet_dense::DefaultDenseExecutor::new(),
            observations: Vec::new(),
        }
    }
}

impl DenseExecutor for FullQrInputSpy {
    fn svd(&mut self, _: DenseRead<'_>) -> Result<Vec<DenseTensor>, DenseError> {
        panic!("test only exercises full QR/LQ")
    }

    fn qr(&mut self, _: DenseRead<'_>) -> Result<Vec<DenseTensor>, DenseError> {
        panic!("full QR/LQ must use the destination API")
    }

    fn qr_into(
        &mut self,
        input: DenseRead<'_>,
        q: DenseWrite<'_>,
        r: DenseWrite<'_>,
    ) -> Result<(), DenseError> {
        let input_shape = input.shape().to_vec();
        let (strides, offset, data_len, values) = match input {
            DenseRead::F64(view) => (
                view.strides().to_vec(),
                view.offset(),
                view.data().len(),
                view.data()
                    .iter()
                    .map(|&value| Complex64::new(value, 0.0))
                    .collect(),
            ),
            DenseRead::C64(view) => (
                view.strides().to_vec(),
                view.offset(),
                view.data().len(),
                view.data().to_vec(),
            ),
            _ => panic!("full QR/LQ fixture must be f64 or c64"),
        };
        assert_eq!(offset, 0);
        assert_eq!(strides, [1, input_shape[0]]);
        assert_eq!(data_len, input_shape.iter().product::<usize>());
        self.observations.push(FullQrObservation {
            input_shape,
            q_shape: q.shape().to_vec(),
            r_shape: r.shape().to_vec(),
            values,
        });
        self.inner.qr_into(input, q, r)
    }

    fn eigh(&mut self, _: DenseRead<'_>) -> Result<Vec<DenseTensor>, DenseError> {
        panic!("test only exercises full QR/LQ")
    }

    fn dot_general_into(
        &mut self,
        _: DenseWrite<'_>,
        _: DenseRead<'_>,
        _: DenseRead<'_>,
        _: &DenseDotConfig,
    ) -> Result<(), DenseError> {
        panic!("test only exercises full QR/LQ")
    }
}

impl Default for CountingDense {
    fn default() -> Self {
        Self {
            inner: tenet_dense::DefaultDenseExecutor::new(),
            svd_calls: 0,
            svd_into_calls: 0,
            svd_vals_calls: 0,
            qr_calls: 0,
            eig_calls: 0,
            eigh_calls: 0,
        }
    }
}

impl DenseExecutor for CountingDense {
    fn svd(&mut self, input: DenseRead<'_>) -> Result<Vec<DenseTensor>, DenseError> {
        self.svd_calls += 1;
        self.inner.svd(input)
    }

    fn svd_vals(&mut self, input: DenseRead<'_>) -> Result<DenseTensor, DenseError> {
        self.svd_vals_calls += 1;
        self.inner.svd_vals(input)
    }

    fn svd_into(
        &mut self,
        input: DenseRead<'_>,
        u: DenseWrite<'_>,
        s: DenseWrite<'_>,
        vt: DenseWrite<'_>,
    ) -> Result<(), DenseError> {
        self.svd_into_calls += 1;
        self.svd_calls += 1;
        self.inner.svd_into(input, u, s, vt)
    }

    fn qr(&mut self, input: DenseRead<'_>) -> Result<Vec<DenseTensor>, DenseError> {
        self.qr_calls += 1;
        self.inner.qr(input)
    }

    fn eigh(&mut self, input: DenseRead<'_>) -> Result<Vec<DenseTensor>, DenseError> {
        self.eigh_calls += 1;
        self.inner.eigh(input)
    }

    fn eig(&mut self, input: DenseRead<'_>) -> Result<Vec<DenseTensor>, DenseError> {
        self.eig_calls += 1;
        self.inner.eig(input)
    }

    fn eigh_into(
        &mut self,
        input: DenseRead<'_>,
        values: DenseWrite<'_>,
        vectors: DenseWrite<'_>,
    ) -> Result<(), DenseError> {
        self.eigh_calls += 1;
        self.inner.eigh_into(input, values, vectors)
    }

    fn dot_general_into(
        &mut self,
        output: DenseWrite<'_>,
        lhs: DenseRead<'_>,
        rhs: DenseRead<'_>,
        config: &DenseDotConfig,
    ) -> Result<(), DenseError> {
        self.inner.dot_general_into(output, lhs, rhs, config)
    }
}

impl FusionRule for LateGenericSpy {
    fn rule_identity(&self) -> RuleIdentity {
        self.rule.rule_identity()
    }
    fn fusion_style(&self) -> FusionStyleKind {
        self.rule.fusion_style()
    }
    fn braiding_style(&self) -> BraidingStyleKind {
        self.rule.braiding_style()
    }
    fn vacuum(&self) -> SectorId {
        self.rule.vacuum()
    }
    fn dual(&self, sector: SectorId) -> SectorId {
        self.rule.dual(sector)
    }
    fn fusion_channels(&self, left: SectorId, right: SectorId) -> SectorVec {
        self.rule.fusion_channels(left, right)
    }
    fn nsymbol(&self, left: SectorId, right: SectorId, coupled: SectorId) -> usize {
        self.rule.nsymbol(left, right, coupled)
    }
}

impl LateGenericSpy {
    pub(super) fn call<T>(&self, value: impl FnOnce() -> T) -> Result<T, LateGenericError> {
        let call = self.calls.get() + 1;
        self.calls.set(call);
        if call == self.fail_at {
            Err(LateGenericError(call))
        } else {
            Ok(value())
        }
    }
}

impl CheckedGenericFusion for LateGenericSpy {
    type Error = LateGenericError;

    fn rule_identity(&self) -> RuleIdentity {
        self.rule.rule_identity()
    }

    fn fusion_style(&self) -> FusionStyleKind {
        self.rule.fusion_style()
    }

    fn braiding_style(&self) -> BraidingStyleKind {
        self.rule.braiding_style()
    }

    fn vacuum(&self) -> SectorId {
        self.rule.vacuum()
    }

    fn try_dual(&self, sector: SectorId) -> Result<SectorId, Self::Error> {
        self.call(|| self.rule.dual(sector))
    }

    fn try_fusion_channels(
        &self,
        left: SectorId,
        right: SectorId,
    ) -> Result<SectorVec, Self::Error> {
        self.call(|| self.rule.fusion_channels(left, right))
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
        self.call(|| self.rule.nsymbol(left, right, coupled))
    }
}

impl CheckedGenericRigidSymbols for LateGenericSpy {
    type Scalar = f64;

    fn try_sqrt_dim_scalar(&self, sector: SectorId) -> Result<Self::Scalar, Self::Error> {
        self.call(|| self.rule.sqrt_dim_scalar(sector))
    }

    fn try_inv_sqrt_dim_scalar(&self, sector: SectorId) -> Result<Self::Scalar, Self::Error> {
        self.call(|| self.rule.inv_sqrt_dim_scalar(sector))
    }

    fn try_frobenius_schur_phase_scalar(
        &self,
        sector: SectorId,
    ) -> Result<Self::Scalar, Self::Error> {
        self.call(|| self.rule.frobenius_schur_phase_scalar(sector))
    }

    fn try_f_symbol_generic(
        &self,
        a: SectorId,
        b: SectorId,
        c: SectorId,
        d: SectorId,
        e: SectorId,
        f: SectorId,
    ) -> Result<GenericFArray<Self::Scalar>, Self::Error> {
        self.call(|| self.rule.f_symbol_generic(a, b, c, d, e, f))
    }

    fn try_r_symbol_generic(
        &self,
        a: SectorId,
        b: SectorId,
        coupled: SectorId,
    ) -> Result<GenericRMatrix<Self::Scalar>, Self::Error> {
        self.call(|| self.rule.r_symbol_generic(a, b, coupled))
    }
}

pub(super) fn late_spy_calls(run: &dyn Fn(&LateGenericSpy)) -> usize {
    let probe = LateGenericSpy {
        rule: FactorGenericRule,
        fail_at: usize::MAX,
        calls: Cell::new(0),
    };
    run(&probe);
    probe.calls.get()
}

#[derive(Default)]
pub(super) struct MatrixFunctionCallSpy {
    pub(super) inner: tenet_dense::DefaultDenseExecutor,
    pub(super) eigh_calls: usize,
    pub(super) solve_calls: usize,
    pub(super) matmul_calls: usize,
    /// Ordinal of a solve that must fail, for the failure-atomicity gate.
    pub(super) fail_solve_number: Option<usize>,
}

impl DenseExecutor for MatrixFunctionCallSpy {
    fn svd(&mut self, input: DenseRead<'_>) -> Result<Vec<DenseTensor>, DenseError> {
        self.inner.svd(input)
    }

    fn qr(&mut self, input: DenseRead<'_>) -> Result<Vec<DenseTensor>, DenseError> {
        self.inner.qr(input)
    }

    fn eigh(&mut self, input: DenseRead<'_>) -> Result<Vec<DenseTensor>, DenseError> {
        self.eigh_calls += 1;
        self.inner.eigh(input)
    }

    fn eigh_into(
        &mut self,
        input: DenseRead<'_>,
        values: DenseWrite<'_>,
        vectors: DenseWrite<'_>,
    ) -> Result<(), DenseError> {
        self.eigh_calls += 1;
        self.inner.eigh_into(input, values, vectors)
    }

    fn solve_into(
        &mut self,
        a: DenseRead<'_>,
        b: DenseRead<'_>,
        x: DenseWrite<'_>,
    ) -> Result<(), DenseError> {
        self.solve_calls += 1;
        if self.fail_solve_number == Some(self.solve_calls) {
            return Err(DenseError::Backend {
                backend: DenseBackend::Tenferro,
                op: "solve_into",
                message: "injected sector failure".to_string(),
            });
        }
        self.inner.solve_into(a, b, x)
    }

    fn dot_general_into(
        &mut self,
        output: DenseWrite<'_>,
        lhs: DenseRead<'_>,
        rhs: DenseRead<'_>,
        config: &DenseDotConfig,
    ) -> Result<(), DenseError> {
        self.matmul_calls += 1;
        self.inner.dot_general_into(output, lhs, rhs, config)
    }
}
