//! Shared fixtures, spy executors and imports for tenet-matrixalgebra's
//! test suite, split by responsibility into sibling files (#1596).

use tenet_core::{
    product_fusion_rule, BlockKey, BlockSpec, BlockStructure, BraidingStyleKind,
    CheckedGenericFusion, CheckedGenericRigidSymbols, CoreError, CoupledSectorFold,
    FermionParityFusionRule, FusionProductSpace, FusionRule, FusionStyleKind, FusionTensorMapSpace,
    FusionTreeHomSpace, FusionTreeKey, GenericFArray, GenericFusionSymbols, GenericRMatrix,
    GenericRigidSymbols, InfallibleGeneric, MultiplicityFreeFusionRule,
    MultiplicityFreeFusionSymbols, MultiplicityFreeRigidSymbols, RuleIdentity, SU2FusionRule,
    SU2Irrep, SectorId, SectorLeg, SectorVec, TensorMap, TensorMapSpace, U1FusionRule, U1Irrep,
    Z2FusionRule,
};
use tenet_tensors::{
    BoundDynamicFusionMapSpace, DenseTreeTransformOperations, DynamicFusionMapSpace,
    OperationError, OutputAxisOrder, TensorContractFusionExecutionContext, TensorContractSpec,
    TreeTransformRuleCacheKey,
};

use crate::factorize::{
    dyn_space_of, map_square_sectors_dyn_into, pinv_cutoff, typed_from_bound_factor,
    typed_from_dyn, validate_eigenvector_singular_values, validate_inverse_region_routes_for_test,
    BoundTensorMap,
};
use crate::test_numerics::numerics;
use crate::*;
use crate::{LeftPolar, Lq, Qr, RightPolar, Svd};
use num_complex::{Complex32, Complex64};
use num_traits::Zero;
use std::{cell::Cell, convert::Infallible, fmt, sync::Arc};
use tenet_dense::{
    DenseBackend, DenseDotConfig, DenseError, DenseExecutor, DenseOwned, DenseRead, DenseTensor,
    DenseWrite,
};

struct RejectExecutorCalls;

struct FailComposition;

#[derive(Default)]
struct SvdCallSpy {
    inner: tenet_dense::DefaultDenseExecutor,
    svd_calls: usize,
}

#[derive(Default)]
struct RejectSvdInto {
    inner: tenet_dense::DefaultDenseExecutor,
    svd_calls: usize,
    svd_into_calls: usize,
    output_ptrs: Vec<(usize, usize)>,
    gemm_ptrs: Vec<(usize, usize)>,
    gemm_views: Vec<(DenseReadView, DenseReadView, bool, bool)>,
}

#[derive(Debug)]
struct DenseReadView {
    shape: Vec<usize>,
    strides: Vec<usize>,
    offset: usize,
}

#[derive(Default)]
struct RejectEighInto {
    inner: tenet_dense::DefaultDenseExecutor,
    eigh_calls: usize,
    eigh_into_calls: usize,
    vector_ptrs: Vec<usize>,
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
struct SolveCallSpy {
    inner: tenet_dense::DefaultDenseExecutor,
    solve_calls: usize,
    destination_ptrs: Vec<usize>,
}

#[derive(Default)]
struct FailSecondSolve {
    inner: tenet_dense::DefaultDenseExecutor,
    solve_calls: usize,
}

#[derive(Default)]
struct FailSecondSvd {
    inner: tenet_dense::DefaultDenseExecutor,
    calls: usize,
}

#[derive(Default)]
struct FailAfterObservingSvdInput {
    observed: Vec<Vec<f64>>,
    outputs: Option<Vec<DenseTensor>>,
}

#[derive(Default)]
struct FailAfterObservingQrInput {
    inner: tenet_dense::DefaultDenseExecutor,
    observed: Vec<Vec<f64>>,
    qr_succeeds: bool,
    outputs: Option<Vec<DenseTensor>>,
}

#[derive(Default)]
struct FailAfterSvdQr {
    inner: tenet_dense::DefaultDenseExecutor,
    svd_calls: usize,
    qr_calls: usize,
}

#[derive(Default)]
struct FailAfterObservingEighInput {
    observed: Vec<Vec<f64>>,
    outputs: Option<Vec<DenseTensor>>,
}

#[derive(Default)]
struct EighCallSpy {
    calls: usize,
}

struct NativeFullSvdSpy {
    inner: tenet_dense::DefaultDenseExecutor,
    full_calls: usize,
}

struct FailSecondOwnedFullSvd {
    inner: tenet_dense::DefaultDenseExecutor,
    calls: usize,
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
    fn new() -> Option<Self> {
        native_full_svd_executor().map(|inner| Self {
            inner,
            full_calls: 0,
        })
    }
}

impl FailSecondOwnedFullSvd {
    fn new() -> Option<Self> {
        native_full_svd_executor().map(|inner| Self { inner, calls: 0 })
    }
}

#[derive(Default)]
struct RecordingEigh {
    inner: tenet_dense::DefaultDenseExecutor,
    raw_values: Vec<Vec<f64>>,
}

#[derive(Clone)]
struct IdentityQdimRule {
    identity: RuleIdentity,
    qdim: f64,
}

impl IdentityQdimRule {
    fn new(qdim: f64) -> Self {
        Self {
            identity: RuleIdentity::new_unique::<Self>(),
            qdim,
        }
    }
}

impl FusionRule for IdentityQdimRule {
    fn rule_identity(&self) -> RuleIdentity {
        self.identity.clone()
    }
    fn fusion_style(&self) -> FusionStyleKind {
        FusionStyleKind::Unique
    }
    fn braiding_style(&self) -> BraidingStyleKind {
        BraidingStyleKind::Bosonic
    }
    fn vacuum(&self) -> SectorId {
        SectorId::new(0)
    }
    fn fusion_channels(&self, _: SectorId, _: SectorId) -> SectorVec {
        [SectorId::new(0)].into_iter().collect()
    }
}

impl MultiplicityFreeFusionRule for IdentityQdimRule {}

impl MultiplicityFreeFusionSymbols for IdentityQdimRule {
    type Scalar = f64;
    fn f_symbol_scalar(
        &self,
        _: SectorId,
        _: SectorId,
        _: SectorId,
        _: SectorId,
        _: SectorId,
        _: SectorId,
    ) -> f64 {
        1.0
    }
    fn r_symbol_scalar(&self, _: SectorId, _: SectorId, _: SectorId) -> f64 {
        1.0
    }
}

impl MultiplicityFreeRigidSymbols for IdentityQdimRule {
    fn dim_scalar(&self, _: SectorId) -> f64 {
        self.qdim
    }
    fn inv_dim_scalar(&self, _: SectorId) -> f64 {
        self.qdim.recip()
    }
    fn sqrt_dim_scalar(&self, _: SectorId) -> f64 {
        self.qdim.sqrt()
    }
    fn inv_sqrt_dim_scalar(&self, _: SectorId) -> f64 {
        self.qdim.sqrt().recip()
    }
    fn twist_scalar(&self, _: SectorId) -> f64 {
        1.0
    }
    fn frobenius_schur_phase_scalar(&self, _: SectorId) -> f64 {
        1.0
    }
}

fn bound_tensor<R, D, const NOUT: usize, const NIN: usize>(
    provider: Arc<R>,
    tensor: &TensorMap<D, NOUT, NIN>,
) -> BoundTensorMap<R, D, NOUT, NIN>
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64>,
    D: Clone,
{
    BoundTensorMap::try_new(provider, tensor.clone()).unwrap()
}

macro_rules! bound_tensor_ref {
    ($provider:expr, $tensor:expr) => {
        bound_tensor($provider, $tensor).as_ref()
    };
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

fn assert_svd_blocks_match<const NOUT: usize, const NIN: usize>(
    lhs: &TensorMap<f64, NOUT, NIN>,
    rhs: &TensorMap<f64, NOUT, NIN>,
) {
    let lhs_structure = std::sync::Arc::clone(lhs.structure());
    let rhs_structure = std::sync::Arc::clone(rhs.structure());
    assert_eq!(lhs_structure.block_count(), rhs_structure.block_count());
    for index in 0..lhs_structure.block_count() {
        let lhs_block = lhs_structure.block(index).unwrap();
        let rhs_block = rhs_structure.block(index).unwrap();
        assert_eq!(lhs_block.key(), rhs_block.key());
        assert_eq!(lhs_block.shape(), rhs_block.shape());
        let shape = lhs_block.shape().to_vec();
        let count = shape.iter().product::<usize>();
        let mut multi_index = vec![0usize; shape.len()];
        for _ in 0..count {
            let lhs_position = lhs_block.offset()
                + multi_index
                    .iter()
                    .zip(lhs_block.strides())
                    .map(|(&i, &s)| i * s)
                    .sum::<usize>();
            let rhs_position = rhs_block.offset()
                + multi_index
                    .iter()
                    .zip(rhs_block.strides())
                    .map(|(&i, &s)| i * s)
                    .sum::<usize>();
            let lhs_value = lhs.data()[lhs_position];
            let rhs_value = rhs.data()[rhs_position];
            assert!(
                (lhs_value - rhs_value).abs() < 1e-10,
                "block {index} element {multi_index:?}: {lhs_value} != {rhs_value}"
            );
            for axis in 0..shape.len() {
                multi_index[axis] += 1;
                if multi_index[axis] < shape[axis] {
                    break;
                }
                multi_index[axis] = 0;
            }
        }
    }
}

fn assert_factor_layout_matches_legacy_shapes<R>(actual: &BoundDynamicFusionMapSpace<R>)
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64>,
{
    // What: canonical factor construction has the exact block layout produced
    // by the former per-tree shape authority.
    let provider = Arc::clone(actual.provider_arc());
    let homspace = actual.space().homspace().clone();
    let shapes = homspace
        .fusion_tree_keys(provider.as_ref())
        .iter()
        .map(|key| {
            homspace
                .codomain()
                .legs()
                .iter()
                .zip(key.codomain_tree().uncoupled())
                .chain(
                    homspace
                        .domain()
                        .legs()
                        .iter()
                        .zip(key.domain_tree().uncoupled()),
                )
                .map(|(leg, &sector)| {
                    leg.degeneracy(sector)
                        .expect("factor tree sector must belong to its final leg")
                })
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let legacy =
        BoundDynamicFusionMapSpace::from_degeneracy_shapes(provider, homspace, shapes).unwrap();
    let actual_space = actual.space();
    let legacy_space = legacy.space();
    assert_eq!(actual_space.nout(), legacy_space.nout());
    assert_eq!(actual_space.nin(), legacy_space.nin());
    assert_eq!(
        actual_space.required_len().unwrap(),
        legacy_space.required_len().unwrap()
    );
    assert_eq!(
        actual_space.structure().block_count(),
        legacy_space.structure().block_count()
    );
    for index in 0..actual_space.structure().block_count() {
        let actual_block = actual_space.structure().block(index).unwrap();
        let legacy_block = legacy_space.structure().block(index).unwrap();
        assert_eq!(actual_block.key(), legacy_block.key());
        assert_eq!(actual_block.shape(), legacy_block.shape());
        assert_eq!(actual_block.strides(), legacy_block.strides());
        assert_eq!(actual_block.offset(), legacy_block.offset());
    }
}

#[derive(Clone, Copy)]
struct FactorGenericRule;

impl FusionRule for FactorGenericRule {
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
        match (left.id(), right.id()) {
            (0, x) | (x, 0) => [SectorId::new(x)].into_iter().collect(),
            (1, 1) => [SectorId::new(0), SectorId::new(1)].into_iter().collect(),
            _ => SectorVec::new(),
        }
    }

    fn nsymbol(&self, left: SectorId, right: SectorId, coupled: SectorId) -> usize {
        if (left.id(), right.id(), coupled.id()) == (1, 1, 1) {
            2
        } else {
            usize::from(self.fusion_channels(left, right).contains(&coupled))
        }
    }
}

impl GenericFusionSymbols for FactorGenericRule {
    type Scalar = f64;

    fn f_symbol_generic(
        &self,
        a: SectorId,
        b: SectorId,
        c: SectorId,
        d: SectorId,
        e: SectorId,
        f: SectorId,
    ) -> GenericFArray<Self::Scalar> {
        let shape = (
            self.nsymbol(a, b, e),
            self.nsymbol(e, c, d),
            self.nsymbol(b, c, f),
            self.nsymbol(a, f, d),
        );
        let rows = shape.0 * shape.1;
        let cols = shape.2 * shape.3;
        let mut data = vec![0.0; rows * cols];
        for index in 0..rows.min(cols) {
            data[index * cols + index] = 1.0;
        }
        GenericFArray::new(data, shape)
    }

    fn r_symbol_generic(
        &self,
        _a: SectorId,
        _b: SectorId,
        coupled: SectorId,
    ) -> GenericRMatrix<Self::Scalar> {
        let size = if coupled == SectorId::new(1) { 2 } else { 1 };
        let mut data = vec![0.0; size * size];
        for index in 0..size {
            data[index * size + index] = 1.0;
        }
        GenericRMatrix::new(data, size, size)
    }
}

impl GenericRigidSymbols for FactorGenericRule {
    fn sqrt_dim_scalar(&self, sector: SectorId) -> Self::Scalar {
        if sector == SectorId::new(1) {
            (1.0 + 2.0_f64.sqrt()).sqrt()
        } else {
            1.0
        }
    }

    fn inv_sqrt_dim_scalar(&self, sector: SectorId) -> Self::Scalar {
        self.sqrt_dim_scalar(sector).recip()
    }

    fn frobenius_schur_phase_scalar(&self, _sector: SectorId) -> Self::Scalar {
        1.0
    }
}

fn generic_factorization_input() -> (BoundDynamicFusionMapSpace<FactorGenericRule>, Vec<f64>) {
    let provider = Arc::new(FactorGenericRule);
    let x = SectorId::new(1);
    let left = SectorLeg::new([(x, 2)], false);
    let unit = SectorLeg::new([(x, 1)], false);
    let homspace = FusionTreeHomSpace::new(
        FusionProductSpace::new([left, unit.clone()]),
        FusionProductSpace::new([unit.clone(), unit]),
    );
    let space =
        BoundDynamicFusionMapSpace::from_final_homspace_generic(provider, homspace).unwrap();
    let data = (0..space.space().required_len().unwrap())
        .map(|index| 1.0 + index as f64 / 8.0)
        .collect();
    (space, data)
}

fn padded_generic_factorization_input(
    source: &BoundDynamicFusionMapSpace<FactorGenericRule>,
    source_data: &[f64],
) -> (BoundDynamicFusionMapSpace<FactorGenericRule>, Vec<f64>) {
    expert_generic_factorization_input(source, source_data, false)
}

fn expert_generic_factorization_input(
    source: &BoundDynamicFusionMapSpace<FactorGenericRule>,
    source_data: &[f64],
    reverse_blocks: bool,
) -> (BoundDynamicFusionMapSpace<FactorGenericRule>, Vec<f64>) {
    let source_structure = source.space().structure();
    let mut offset = 1usize;
    let mut blocks = Vec::with_capacity(source_structure.block_count());
    let mut indices = (0..source_structure.block_count()).collect::<Vec<_>>();
    if reverse_blocks {
        indices.reverse();
    }
    for index in indices {
        let block = source_structure.block(index).unwrap();
        blocks.push(
            BlockSpec::column_major_with_key(block.key().clone(), block.shape().to_vec(), offset)
                .unwrap(),
        );
        offset += block.shape().iter().product::<usize>() + 1;
    }
    let structure = BlockStructure::from_blocks_with_rank(source.space().rank(), blocks).unwrap();
    let typed_space = FusionTensorMapSpace::new_unbound(
        TensorMapSpace::<2, 2>::from_dims([2, 1], [1, 1]).unwrap(),
        source.space().homspace().clone(),
        structure,
    )
    .unwrap()
    .try_bind_rule(source.provider())
    .unwrap();
    let tensor = TensorMap::<f64, 2, 2>::from_block_fn_with_fusion_space(
        typed_space,
        0.0,
        |key, indices| {
            let block = source_structure
                .block(
                    source_structure
                        .find_block_index_by_key(key)
                        .expect("copy preserves every key"),
                )
                .unwrap();
            source_data[block.offset()
                + indices
                    .iter()
                    .zip(block.strides())
                    .map(|(&index, &stride)| index * stride)
                    .sum::<usize>()]
        },
    )
    .unwrap();
    let dynamic = DynamicFusionMapSpace::from_typed(tensor.fusion_space().unwrap());
    let bound =
        BoundDynamicFusionMapSpace::bind_generic(dynamic, Arc::clone(source.provider_arc()))
            .unwrap();
    (bound, tensor.data().to_vec())
}

fn flattened_block_value<D: FactorScalar>(
    data: &[D],
    block: tenet_core::BlockRef<'_>,
    axes: std::ops::Range<usize>,
    mut flat_index: usize,
) -> (usize, D) {
    let mut offset = block.offset();
    for axis in axes {
        let coordinate = flat_index % block.shape()[axis];
        flat_index /= block.shape()[axis];
        offset += coordinate * block.strides()[axis];
    }
    (offset, data[offset])
}

/// The diagonal `S` of a generic compact SVD, built from its spectrum on
/// `u`'s provider.
fn generic_diagonal_factor<R, D>(
    u: &BoundDynFactor<R, D>,
    spectrum: &[SectorSpectrum],
) -> BoundDynFactor<R, D>
where
    R: FusionRule,
    D: FactorScalar,
{
    let space =
        diagonal_bond_bound_space_generic(Arc::clone(u.space().provider_arc()), spectrum).unwrap();
    let data = diagonal_bond_data(space.space(), spectrum, &D::from_real).unwrap();
    BoundDynFactor::from_bound(space, data, 1, 1).unwrap()
}

fn assert_compact_factors_reconstruct_input<R, D>(
    input: &BoundDynamicTensorRef<'_, R, D>,
    left: &BoundDynFactor<R, D>,
    diagonal: Option<&BoundDynFactor<R, D>>,
    right: &BoundDynFactor<R, D>,
) where
    D: FactorScalar,
{
    let source_structure = input.space().space().structure();
    let left_structure = left.space().space().structure();
    let right_structure = right.space().space().structure();
    for source_index in 0..source_structure.block_count() {
        let source = source_structure.block(source_index).unwrap();
        let BlockKey::FusionTree(source_key) = source.key() else {
            panic!("factorization fixture must use fusion-tree blocks")
        };
        let left_block = (0..left_structure.block_count())
            .map(|index| left_structure.block(index).unwrap())
            .find(|block| {
                block
                    .key()
                    .as_fusion_tree_pair()
                    .is_some_and(|key| key.codomain_tree() == source_key.codomain_tree())
            })
            .unwrap();
        let right_block = (0..right_structure.block_count())
            .map(|index| right_structure.block(index).unwrap())
            .find(|block| {
                block
                    .key()
                    .as_fusion_tree_pair()
                    .is_some_and(|key| key.domain_tree() == source_key.domain_tree())
            })
            .unwrap();
        let rows = source.shape()[..input.space().space().nout()]
            .iter()
            .product::<usize>();
        let cols = source.shape()[input.space().space().nout()..]
            .iter()
            .product::<usize>();
        let rank = left_block.shape()[left_block.shape().len() - 1];
        assert_eq!(right_block.shape()[0], rank);
        let diagonal_block = diagonal.map(|factor| {
            let structure = factor.space().space().structure();
            (0..structure.block_count())
                .map(|index| structure.block(index).unwrap())
                .find(|block| {
                    block
                        .key()
                        .as_fusion_tree_pair()
                        .is_some_and(|key| key.coupled() == source_key.coupled())
                })
                .unwrap()
        });
        for column in 0..cols {
            for row in 0..rows {
                let mut reconstructed = D::zero();
                for bond in 0..rank {
                    let (left_offset, _) = flattened_block_value(
                        left.data(),
                        left_block,
                        0..left_block.shape().len() - 1,
                        row,
                    );
                    let left_value = left.data()
                        [left_offset + bond * left_block.strides()[left_block.shape().len() - 1]];
                    let (right_offset, _) = flattened_block_value(
                        right.data(),
                        right_block,
                        1..right_block.shape().len(),
                        column,
                    );
                    let right_value = right.data()[right_offset + bond * right_block.strides()[0]];
                    let scale = diagonal_block.map_or_else(D::one, |block| {
                        diagonal.unwrap().data()
                            [block.offset() + bond * block.strides()[0] + bond * block.strides()[1]]
                    });
                    reconstructed = reconstructed + left_value * scale * right_value;
                }
                let (_, expected) = flattened_block_value(
                    input.data(),
                    source,
                    0..source.shape().len(),
                    row + rows * column,
                );
                assert!(
                    (reconstructed.widen_complex() - expected.widen_complex()).norm() < 1.0e-10
                );
            }
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct LateGenericError(usize);

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

struct LateGenericSpy {
    rule: FactorGenericRule,
    fail_at: usize,
    calls: Cell<usize>,
}

struct CountingDense {
    inner: tenet_dense::DefaultDenseExecutor,
    svd_calls: usize,
    svd_into_calls: usize,
    svd_vals_calls: usize,
    qr_calls: usize,
    eig_calls: usize,
    eigh_calls: usize,
}

#[derive(Debug)]
struct FullQrObservation {
    input_shape: Vec<usize>,
    q_shape: Vec<usize>,
    r_shape: Vec<usize>,
    values: Vec<Complex64>,
}

struct FullQrInputSpy {
    inner: tenet_dense::DefaultDenseExecutor,
    observations: Vec<FullQrObservation>,
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

struct CheckedOnlyFactorRule {
    calls: Cell<usize>,
}

impl CheckedOnlyFactorRule {
    fn call<T>(&self, value: impl FnOnce(&FactorGenericRule) -> T) -> Result<T, Infallible> {
        self.calls.set(self.calls.get() + 1);
        Ok(value(&FactorGenericRule))
    }
}

impl CheckedGenericFusion for CheckedOnlyFactorRule {
    type Error = Infallible;

    fn rule_identity(&self) -> RuleIdentity {
        FactorGenericRule.rule_identity()
    }

    fn fusion_style(&self) -> FusionStyleKind {
        FactorGenericRule.fusion_style()
    }

    fn braiding_style(&self) -> BraidingStyleKind {
        FactorGenericRule.braiding_style()
    }

    fn vacuum(&self) -> SectorId {
        FactorGenericRule.vacuum()
    }

    fn try_dual(&self, sector: SectorId) -> Result<SectorId, Self::Error> {
        self.call(|rule| rule.dual(sector))
    }

    fn try_fusion_channels(
        &self,
        left: SectorId,
        right: SectorId,
    ) -> Result<SectorVec, Self::Error> {
        self.call(|rule| rule.fusion_channels(left, right))
    }

    fn try_fusion_channels_in_table(
        &self,
        left: SectorId,
        right: SectorId,
    ) -> Result<SectorVec, Self::Error> {
        self.call(|rule| rule.fusion_channels(left, right))
    }

    fn try_nsymbol(
        &self,
        left: SectorId,
        right: SectorId,
        coupled: SectorId,
    ) -> Result<usize, Self::Error> {
        self.call(|rule| rule.nsymbol(left, right, coupled))
    }
}

impl LateGenericSpy {
    fn call<T>(&self, value: impl FnOnce() -> T) -> Result<T, LateGenericError> {
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

fn checked_svd_matrix(sector: SectorId, complex: bool) -> (usize, usize, Vec<Complex64>) {
    match sector.id() {
        0 => (
            2,
            2,
            vec![
                Complex64::new(0.0, 0.0),
                Complex64::new(1.0, 0.0),
                if complex {
                    Complex64::new(0.0, 4.0)
                } else {
                    Complex64::new(4.0, 0.0)
                },
                Complex64::new(0.0, 0.0),
            ],
        ),
        1 => {
            let a = 3.0 / 2.0_f64.sqrt();
            (
                3,
                2,
                vec![
                    Complex64::new(a, 0.0),
                    if complex {
                        Complex64::new(0.0, a)
                    } else {
                        Complex64::new(a, 0.0)
                    },
                    Complex64::new(0.0, 0.0),
                    Complex64::new(0.0, 0.0),
                    Complex64::new(0.0, 0.0),
                    if complex {
                        Complex64::new(0.0, -2.0)
                    } else {
                        Complex64::new(2.0, 0.0)
                    },
                ],
            )
        }
        id => panic!("unexpected checked-SVD fixture sector {id}"),
    }
}

fn generic_svd_truncation_input<D>(
    complex: bool,
) -> (BoundDynamicFusionMapSpace<FactorGenericRule>, Vec<D>)
where
    D: FactorScalar,
{
    let vacuum = SectorId::new(0);
    let x = SectorId::new(1);
    let homspace = FusionTreeHomSpace::new(
        FusionProductSpace::new([SectorLeg::new([(vacuum, 2), (x, 3)], false)]),
        FusionProductSpace::new([SectorLeg::new([(vacuum, 2), (x, 2)], false)]),
    );
    let space = BoundDynamicFusionMapSpace::from_final_homspace_generic(
        Arc::new(FactorGenericRule),
        homspace,
    )
    .unwrap();
    let mut data = vec![D::zero(); space.space().required_len().unwrap()];
    for index in 0..space.space().structure().block_count() {
        let block = space.space().structure().block(index).unwrap();
        let BlockKey::FusionTree(key) = block.key() else {
            panic!("Generic SVD fixture must use fusion-tree blocks")
        };
        let (rows, cols, matrix) = checked_svd_matrix(key.codomain_tree().coupled(), complex);
        assert_eq!(block.shape(), [rows, cols]);
        for col in 0..cols {
            for row in 0..rows {
                let source_index = row + rows * col;
                let destination =
                    block.offset() + row * block.strides()[0] + col * block.strides()[1];
                data[destination] = D::from_complex64(matrix[source_index]);
            }
        }
    }
    (space, data)
}

fn padded_generic_svd_truncation_input<D>(
    source: &BoundDynamicFusionMapSpace<FactorGenericRule>,
    source_data: &[D],
) -> (BoundDynamicFusionMapSpace<FactorGenericRule>, Vec<D>)
where
    D: FactorScalar,
{
    let source_structure = source.space().structure();
    let mut offset = 1usize;
    let mut blocks = Vec::with_capacity(source_structure.block_count());
    for index in 0..source_structure.block_count() {
        let block = source_structure.block(index).unwrap();
        blocks.push(
            BlockSpec::column_major_with_key(block.key().clone(), block.shape().to_vec(), offset)
                .unwrap(),
        );
        offset += block.shape().iter().product::<usize>() + 1;
    }
    let structure = BlockStructure::from_blocks_with_rank(source.space().rank(), blocks).unwrap();
    let typed_space = FusionTensorMapSpace::new_unbound(
        TensorMapSpace::<1, 1>::from_dims([1], [1]).unwrap(),
        source.space().homspace().clone(),
        structure,
    )
    .unwrap()
    .try_bind_rule(source.provider())
    .unwrap();
    let tensor = TensorMap::<D, 1, 1>::from_block_fn_with_fusion_space(
        typed_space,
        D::zero(),
        |key, indices| {
            let block = source_structure
                .block(
                    source_structure
                        .find_block_index_by_key(key)
                        .expect("copy preserves every key"),
                )
                .unwrap();
            source_data[block.offset()
                + indices
                    .iter()
                    .zip(block.strides())
                    .map(|(&index, &stride)| index * stride)
                    .sum::<usize>()]
        },
    )
    .unwrap();
    let dynamic = DynamicFusionMapSpace::from_typed(tensor.fusion_space().unwrap());
    let bound =
        BoundDynamicFusionMapSpace::bind_generic(dynamic, Arc::clone(source.provider_arc()))
            .unwrap();
    (bound, tensor.data().to_vec())
}

fn assert_generic_complex_factor_close(
    actual: &BoundDynFactor<FactorGenericRule, Complex64>,
    expected: &BoundDynFactor<FactorGenericRule, Complex64>,
) {
    assert_eq!(
        actual.space().space().homspace(),
        expected.space().space().homspace()
    );
    assert_eq!(
        actual.space().space().structure(),
        expected.space().space().structure()
    );
    assert_eq!(actual.data().len(), expected.data().len());
    for (&actual, &expected) in actual.data().iter().zip(expected.data()) {
        assert!((actual - expected).norm() < 1.0e-12);
    }
}

#[expect(
    clippy::arc_with_non_send_sync,
    reason = "the checked Generic API requires Arc identity while Cell is a single-threaded call spy"
)]
fn checked_svd_truncation_input<D>(
    complex: bool,
) -> (
    Arc<LateGenericSpy>,
    BoundDynamicFusionMapSpace<LateGenericSpy>,
    Vec<D>,
)
where
    D: FactorScalar,
{
    let vacuum = SectorId::new(0);
    let x = SectorId::new(1);
    let homspace = FusionTreeHomSpace::new(
        FusionProductSpace::new([SectorLeg::new([(vacuum, 2), (x, 3)], false)]),
        FusionProductSpace::new([SectorLeg::new([(vacuum, 2), (x, 2)], false)]),
    );
    let source = BoundDynamicFusionMapSpace::from_final_homspace_generic(
        Arc::new(FactorGenericRule),
        homspace,
    )
    .unwrap();
    let provider = Arc::new(LateGenericSpy {
        rule: FactorGenericRule,
        fail_at: usize::MAX,
        calls: Cell::new(0),
    });
    let checked =
        BoundDynamicFusionMapSpace::bind_generic(source.space().clone(), Arc::clone(&provider))
            .unwrap();
    let mut data = vec![D::zero(); checked.space().required_len().unwrap()];
    let structure = checked.space().structure();
    for index in 0..structure.block_count() {
        let block = structure.block(index).unwrap();
        let BlockKey::FusionTree(key) = block.key() else {
            panic!("checked Generic fixture must use fusion-tree blocks")
        };
        let sector = key.codomain_tree().coupled();
        let (rows, cols, matrix) = checked_svd_matrix(sector, complex);
        assert_eq!(block.shape(), [rows, cols]);
        for col in 0..cols {
            for row in 0..rows {
                let source_index = row + rows * col;
                let destination =
                    block.offset() + row * block.strides()[0] + col * block.strides()[1];
                data[destination] = D::from_complex64(matrix[source_index]);
            }
        }
    }
    (provider, checked, data)
}

#[expect(
    clippy::arc_with_non_send_sync,
    reason = "the checked Generic API requires Arc identity while Cell is a single-threaded call spy"
)]
fn checked_svd_wide_input<D>() -> (
    Arc<LateGenericSpy>,
    BoundDynamicFusionMapSpace<LateGenericSpy>,
    Vec<D>,
)
where
    D: FactorScalar,
{
    let vacuum = SectorId::new(0);
    let x = SectorId::new(1);
    let homspace = FusionTreeHomSpace::new(
        FusionProductSpace::new([SectorLeg::new([(vacuum, 2), (x, 2)], false)]),
        FusionProductSpace::new([SectorLeg::new([(vacuum, 2), (x, 3)], false)]),
    );
    let source = BoundDynamicFusionMapSpace::from_final_homspace_generic(
        Arc::new(FactorGenericRule),
        homspace,
    )
    .unwrap();
    let provider = Arc::new(LateGenericSpy {
        rule: FactorGenericRule,
        fail_at: usize::MAX,
        calls: Cell::new(0),
    });
    let checked =
        BoundDynamicFusionMapSpace::bind_generic(source.space().clone(), Arc::clone(&provider))
            .unwrap();
    let mut data = vec![D::zero(); checked.space().required_len().unwrap()];
    let structure = checked.space().structure();
    for index in 0..structure.block_count() {
        let block = structure.block(index).unwrap();
        let BlockKey::FusionTree(key) = block.key() else {
            panic!("checked Generic fixture must use fusion-tree blocks")
        };
        let (rows, cols, matrix) = checked_svd_matrix(key.coupled(), true);
        assert_eq!(block.shape(), [cols, rows]);
        for column in 0..rows {
            for row in 0..cols {
                let source = matrix[column + rows * row].conj();
                let destination =
                    block.offset() + row * block.strides()[0] + column * block.strides()[1];
                data[destination] = D::from_complex64(source);
            }
        }
    }
    (provider, checked, data)
}

#[expect(
    clippy::arc_with_non_send_sync,
    reason = "the checked Generic API requires Arc identity while Cell is a single-threaded call spy"
)]
fn bind_checked_only(
    space: &BoundDynamicFusionMapSpace<impl FusionRule>,
) -> (
    Arc<CheckedOnlyFactorRule>,
    BoundDynamicFusionMapSpace<CheckedOnlyFactorRule>,
) {
    let provider = Arc::new(CheckedOnlyFactorRule {
        calls: Cell::new(0),
    });
    let checked = BoundDynamicFusionMapSpace::from_final_homspace_generic_checked(
        Arc::clone(&provider),
        space.space().homspace().clone(),
    )
    .unwrap();
    (provider, checked)
}

fn generic_values_endomorphism_input() -> (
    BoundDynamicFusionMapSpace<FactorGenericRule>,
    Vec<Complex64>,
    Vec<Complex64>,
) {
    let x = SectorId::new(1);
    let leg = SectorLeg::new([(x, 1)], false);
    let homspace = FusionTreeHomSpace::new(
        FusionProductSpace::new([leg.clone(), leg.clone()]),
        FusionProductSpace::new([leg.clone(), leg]),
    );
    let space = BoundDynamicFusionMapSpace::from_final_homspace_generic(
        Arc::new(FactorGenericRule),
        homspace,
    )
    .unwrap();
    let regions = space
        .space()
        .structure()
        .coupled_sector_regions(2)
        .unwrap()
        .unwrap();
    let mut hermitian = vec![Complex64::zero(); space.space().required_len().unwrap()];
    let mut general = hermitian.clone();
    for region in regions.iter() {
        let (hermitian_matrix, general_matrix): (&[Complex64], &[Complex64]) =
            match region.coupled().id() {
                0 => (&[Complex64::new(-4.0, 0.0)], &[Complex64::new(2.0, -1.0)]),
                1 => (
                    &[
                        Complex64::new(2.0, 0.0),
                        Complex64::new(0.0, -1.0),
                        Complex64::new(0.0, 1.0),
                        Complex64::new(2.0, 0.0),
                    ],
                    &[
                        Complex64::new(1.0, 1.0),
                        Complex64::new(0.0, 0.0),
                        Complex64::new(2.0, 0.0),
                        Complex64::new(3.0, -1.0),
                    ],
                ),
                sector => panic!("unexpected Generic values sector {sector}"),
            };
        assert_eq!(region.range().len(), hermitian_matrix.len());
        hermitian[region.range()].copy_from_slice(hermitian_matrix);
        general[region.range()].copy_from_slice(general_matrix);
    }
    (space, hermitian, general)
}

fn padded_reordered_generic_endomorphism_input(
    source: &BoundDynamicFusionMapSpace<FactorGenericRule>,
    source_data: &[Complex64],
) -> (
    BoundDynamicFusionMapSpace<FactorGenericRule>,
    Vec<Complex64>,
) {
    expert_generic_endomorphism_input(
        source,
        source_data,
        (0..source.space().structure().block_count()).rev(),
    )
}

fn expert_generic_endomorphism_input(
    source: &BoundDynamicFusionMapSpace<FactorGenericRule>,
    source_data: &[Complex64],
    indices: impl IntoIterator<Item = usize>,
) -> (
    BoundDynamicFusionMapSpace<FactorGenericRule>,
    Vec<Complex64>,
) {
    let source_structure = source.space().structure();
    let mut offset = 1usize;
    let mut blocks = Vec::with_capacity(source_structure.block_count());
    for index in indices {
        let block = source_structure.block(index).unwrap();
        blocks.push(
            BlockSpec::column_major_with_key(block.key().clone(), block.shape().to_vec(), offset)
                .unwrap(),
        );
        offset += block.shape().iter().product::<usize>() + 1;
    }
    let structure = BlockStructure::from_blocks_with_rank(4, blocks).unwrap();
    let typed_space = FusionTensorMapSpace::new_unbound(
        TensorMapSpace::<2, 2>::from_dims([1, 1], [1, 1]).unwrap(),
        source.space().homspace().clone(),
        structure,
    )
    .unwrap()
    .try_bind_rule(source.provider())
    .unwrap();
    let tensor = TensorMap::<Complex64, 2, 2>::from_block_fn_with_fusion_space(
        typed_space,
        Complex64::zero(),
        |key, indices| {
            let block = source_structure
                .block(source_structure.find_block_index_by_key(key).unwrap())
                .unwrap();
            source_data[block.offset()
                + indices
                    .iter()
                    .zip(block.strides())
                    .map(|(&index, &stride)| index * stride)
                    .sum::<usize>()]
        },
    )
    .unwrap();
    let dynamic = DynamicFusionMapSpace::from_typed(tensor.fusion_space().unwrap());
    let bound =
        BoundDynamicFusionMapSpace::bind_generic(dynamic, Arc::clone(source.provider_arc()))
            .unwrap();
    (bound, tensor.data().to_vec())
}

fn interleaved_generic_endomorphism_input(
    source: &BoundDynamicFusionMapSpace<FactorGenericRule>,
    source_data: &[Complex64],
) -> (
    BoundDynamicFusionMapSpace<FactorGenericRule>,
    Vec<Complex64>,
) {
    let structure = source.space().structure();
    let regions = structure.coupled_sector_regions(2).unwrap().unwrap();
    let scalar = regions
        .iter()
        .find(|region| region.coupled() == SectorId::new(0))
        .unwrap();
    let matrix = regions
        .iter()
        .find(|region| region.coupled() == SectorId::new(1))
        .unwrap();
    assert_eq!((matrix.row_trees().len(), matrix.col_trees().len()), (2, 2));
    let find = |row: usize, col: usize| {
        (0..structure.block_count())
            .find(|&index| {
                structure
                    .block(index)
                    .unwrap()
                    .key()
                    .as_fusion_tree_pair()
                    .is_some_and(|key| {
                        key.codomain_tree() == matrix.row_trees()[row].tree()
                            && key.domain_tree() == matrix.col_trees()[col].tree()
                    })
            })
            .unwrap()
    };
    let scalar_index = (0..structure.block_count())
        .find(|&index| {
            structure
                .block(index)
                .unwrap()
                .key()
                .as_fusion_tree_pair()
                .is_some_and(|key| key.coupled() == scalar.coupled())
        })
        .unwrap();
    expert_generic_endomorphism_input(
        source,
        source_data,
        [find(1, 1), scalar_index, find(0, 1), find(1, 0), find(0, 0)],
    )
}

fn late_spy_calls(run: &dyn Fn(&LateGenericSpy)) -> usize {
    let probe = LateGenericSpy {
        rule: FactorGenericRule,
        fail_at: usize::MAX,
        calls: Cell::new(0),
    };
    run(&probe);
    probe.calls.get()
}

fn checked_enumeration_calls<D: FactorScalar>(factor: &BoundDynFactor<LateGenericSpy, D>) -> usize {
    let homspace = factor.space().space().homspace().clone();
    late_spy_calls(&|probe| {
        homspace
            .prepare_coupled_subblock_structure_from_leg_degeneracies_generic_checked(probe)
            .unwrap();
    })
}

fn f64_qr_outputs(rows: usize, cols: usize) -> Vec<DenseTensor> {
    let mut dense = tenet_dense::DefaultDenseExecutor::new();
    let data = vec![1.0; rows * cols];
    dense
        .qr(DenseRead::F64(
            tenet_dense::DenseView::new(&data, &[rows, cols], &[1, rows], 0).unwrap(),
        ))
        .unwrap()
}

fn c64_qr_outputs(rows: usize, cols: usize) -> Vec<DenseTensor> {
    let mut dense = tenet_dense::DefaultDenseExecutor::new();
    let data = vec![Complex64::new(1.0, 1.0); rows * cols];
    dense
        .qr(DenseRead::C64(
            tenet_dense::DenseView::new(&data, &[rows, cols], &[1, rows], 0).unwrap(),
        ))
        .unwrap()
}

fn f64_eigh_outputs(order: usize) -> Vec<DenseTensor> {
    let mut dense = tenet_dense::DefaultDenseExecutor::new();
    let data = vec![1.0; order * order];
    let shape = [order, order];
    let strides = [1, order];
    dense
        .eigh(DenseRead::F64(
            tenet_dense::DenseView::new(&data, &shape, &strides, 0).unwrap(),
        ))
        .unwrap()
}

fn c64_eigh_outputs(order: usize) -> Vec<DenseTensor> {
    let mut dense = tenet_dense::DefaultDenseExecutor::new();
    let data = vec![Complex64::new(1.0, 1.0); order * order];
    let shape = [order, order];
    let strides = [1, order];
    dense
        .eigh(DenseRead::C64(
            tenet_dense::DenseView::new(&data, &shape, &strides, 0).unwrap(),
        ))
        .unwrap()
}

fn rectangular_svd_tensor(rows: usize, cols: usize) -> TensorMap<f64, 1, 1> {
    let rule = Z2FusionRule;
    let even = SectorId::new(0);
    let homspace = FusionTreeHomSpace::new(
        FusionProductSpace::new([SectorLeg::new([(even, rows)], false)]),
        FusionProductSpace::new([SectorLeg::new([(even, cols)], false)]),
    );
    let shapes = homspace
        .fusion_tree_keys(&rule)
        .iter()
        .map(|_| vec![rows, cols])
        .collect::<Vec<_>>();
    let space = FusionTensorMapSpace::from_degeneracy_shapes_coupled(
        TensorMapSpace::<1, 1>::from_dims([rows], [cols]).unwrap(),
        homspace,
        &rule,
        shapes,
    )
    .unwrap();
    TensorMap::from_vec_with_fusion_space(
        (0..rows * cols)
            .map(|index| ((index * 11 + 2) % 19) as f64 - 7.0)
            .collect(),
        space,
    )
    .unwrap()
}

fn mixed_rectangular_tensor(
    even_shape: (usize, usize),
    odd_shape: (usize, usize),
) -> TensorMap<f64, 1, 1> {
    let rule = Z2FusionRule;
    let even = SectorId::new(0);
    let odd = SectorId::new(1);
    let homspace = FusionTreeHomSpace::new(
        FusionProductSpace::new([SectorLeg::new(
            [(even, even_shape.0), (odd, odd_shape.0)],
            false,
        )]),
        FusionProductSpace::new([SectorLeg::new(
            [(even, even_shape.1), (odd, odd_shape.1)],
            false,
        )]),
    );
    let shapes = homspace
        .fusion_tree_keys(&rule)
        .iter()
        .map(|key| match key.codomain_tree().coupled() {
            sector if sector == even => vec![even_shape.0, even_shape.1],
            sector if sector == odd => vec![odd_shape.0, odd_shape.1],
            sector => panic!("unexpected Z2 sector {sector:?}"),
        })
        .collect::<Vec<_>>();
    let space = FusionTensorMapSpace::from_degeneracy_shapes_coupled(
        TensorMapSpace::<1, 1>::from_dims(
            [even_shape.0 + odd_shape.0],
            [even_shape.1 + odd_shape.1],
        )
        .unwrap(),
        homspace,
        &rule,
        shapes,
    )
    .unwrap();
    TensorMap::from_vec_with_fusion_space(
        (0..space.required_len().unwrap())
            .map(|index| ((index * 7 + 3) % 17) as f64 - 6.0)
            .collect(),
        space,
    )
    .unwrap()
}

fn transposed_rectangular_tensor(
    tensor: &TensorMap<f64, 1, 1>,
    rows: usize,
    cols: usize,
) -> TensorMap<f64, 1, 1> {
    let mut data = vec![0.0; rows * cols];
    for col in 0..cols {
        for row in 0..rows {
            data[col + cols * row] = tensor.data()[row + rows * col];
        }
    }
    TensorMap::from_vec_with_fusion_space(
        data,
        rectangular_svd_tensor(cols, rows)
            .fusion_space()
            .unwrap()
            .as_ref()
            .clone(),
    )
    .unwrap()
}

fn mixed_rectangular_c32_tensor() -> TensorMap<Complex32, 1, 1> {
    let rule = Z2FusionRule;
    let even = SectorId::new(0);
    let odd = SectorId::new(1);
    let homspace = FusionTreeHomSpace::new(
        FusionProductSpace::new([SectorLeg::new([(even, 5), (odd, 2)], false)]),
        FusionProductSpace::new([SectorLeg::new([(even, 3), (odd, 4)], false)]),
    );
    let shapes = homspace
        .fusion_tree_keys(&rule)
        .iter()
        .map(|key| match key.codomain_tree().coupled() {
            sector if sector == even => vec![5, 3],
            sector if sector == odd => vec![2, 4],
            sector => panic!("unexpected Z2 sector {sector:?}"),
        })
        .collect::<Vec<_>>();
    let space = FusionTensorMapSpace::from_degeneracy_shapes_coupled(
        TensorMapSpace::<1, 1>::from_dims([7], [7]).unwrap(),
        homspace,
        &rule,
        shapes,
    )
    .unwrap();
    TensorMap::from_vec_with_fusion_space(
        (0..space.required_len().unwrap())
            .map(|index| {
                Complex32::new(
                    ((index * 7 + 2) % 17) as f32 - 6.0,
                    ((index * 5 + 3) % 13) as f32 * 0.25 - 1.0,
                )
            })
            .collect(),
        space,
    )
    .unwrap()
}

fn tsvd_test_tensor<R>(rule: &R, sectors: &[SectorId]) -> TensorMap<f64, 2, 2>
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64>,
{
    let degeneracy = 2usize;
    let leg = || SectorLeg::new(sectors.iter().map(|&sector| (sector, degeneracy)), false);
    let leg_dim = sectors.len() * degeneracy;
    let homspace = FusionTreeHomSpace::new(
        FusionProductSpace::new([leg(), leg()]),
        FusionProductSpace::new([leg(), leg()]),
    );
    let key_count = homspace.fusion_tree_keys(rule).len();
    let space = FusionTensorMapSpace::from_degeneracy_shapes_coupled(
        TensorMapSpace::<2, 2>::from_dims([leg_dim, leg_dim], [leg_dim, leg_dim]).unwrap(),
        homspace,
        rule,
        vec![vec![degeneracy; 4]; key_count],
    )
    .unwrap();
    let len = space.required_len().unwrap();
    TensorMap::<f64, 2, 2>::from_vec_with_fusion_space(
        (0..len)
            .map(|index| ((index * 11 + 5) % 29) as f64 * 0.25 - 3.0)
            .collect(),
        space,
    )
    .unwrap()
}

fn contract_pair<R>(
    rule: &R,
    template: &TensorMap<f64, 2, 2>,
    left: &TensorMap<f64, 2, 1>,
    right: &TensorMap<f64, 1, 2>,
) -> TensorMap<f64, 2, 2>
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64> + TreeTransformRuleCacheKey,
{
    let mut reconstructed = TensorMap::<f64, 2, 2>::from_vec_with_fusion_space(
        vec![0.0; template.data().len()],
        template.fusion_space().unwrap().as_ref().clone(),
    )
    .unwrap();
    let mut context = TensorContractFusionExecutionContext::<f64, R::Key>::default();
    context
        .tensorcontract_fusion_into(
            rule,
            &mut reconstructed,
            left,
            right,
            TensorContractSpec::new(&[2], &[0], OutputAxisOrder::from_axes(&[0, 1, 2, 3])),
            1.0,
            0.0,
        )
        .unwrap();
    reconstructed
}

fn hermitian_test_tensor<R>(rule: &R, sectors: &[SectorId]) -> TensorMap<f64, 2, 2>
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64>,
{
    let degeneracy = 2usize;
    let leg = || SectorLeg::new(sectors.iter().map(|&sector| (sector, degeneracy)), false);
    let leg_dim = sectors.len() * degeneracy;
    let homspace = FusionTreeHomSpace::new(
        FusionProductSpace::new([leg(), leg()]),
        FusionProductSpace::new([leg(), leg()]),
    );
    let key_count = homspace.fusion_tree_keys(rule).len();
    let space = FusionTensorMapSpace::from_degeneracy_shapes_coupled(
        TensorMapSpace::<2, 2>::from_dims([leg_dim, leg_dim], [leg_dim, leg_dim]).unwrap(),
        homspace,
        rule,
        vec![vec![degeneracy; 4]; key_count],
    )
    .unwrap();
    // Symmetric under swapping the (codomain tree, row indices) and
    // (domain tree, column indices) labels, so every coupled sector matrix is
    // symmetric (real Hermitian).
    let side_label = |tree: &FusionTreeKey, indices: &[usize]| -> u64 {
        let mut label = 17u64;
        for &sector in tree.uncoupled() {
            label = label.wrapping_mul(31).wrapping_add(sector.id() as u64 + 1);
        }
        for &index in indices {
            label = label.wrapping_mul(37).wrapping_add(index as u64 + 1);
        }
        label
    };
    TensorMap::<f64, 2, 2>::from_block_fn_with_fusion_space(space, 0.0, |key, indices| {
        let BlockKey::FusionTree(tree) = key else {
            return 0.0;
        };
        let row = side_label(tree.codomain_tree(), &indices[..2]);
        let col = side_label(tree.domain_tree(), &indices[2..]);
        let (low, high) = if row <= col { (row, col) } else { (col, row) };
        let hash = low
            .wrapping_mul(6364136223846793005)
            .wrapping_add(high.wrapping_mul(1442695040888963407));
        ((hash >> 33) % 19) as f64 * 0.5 - 4.0
    })
    .unwrap()
}

fn assert_eigen_equation<R>(
    rule: &R,
    tensor: &TensorMap<f64, 2, 2>,
    v: &TensorMap<f64, 2, 1>,
    d: &TensorMap<f64, 1, 1>,
) where
    R: MultiplicityFreeRigidSymbols<Scalar = f64> + TreeTransformRuleCacheKey,
{
    let mut context = TensorContractFusionExecutionContext::<f64, R::Key>::default();
    // t . V
    let mut tv = TensorMap::<f64, 2, 1>::from_vec_with_fusion_space(
        vec![0.0; v.data().len()],
        v.fusion_space().unwrap().as_ref().clone(),
    )
    .unwrap();
    context
        .tensorcontract_fusion_into(
            rule,
            &mut tv,
            tensor,
            v,
            TensorContractSpec::new(&[2, 3], &[0, 1], OutputAxisOrder::from_axes(&[0, 1, 2])),
            1.0,
            0.0,
        )
        .unwrap();
    // V . D
    let mut vd = TensorMap::<f64, 2, 1>::from_vec_with_fusion_space(
        vec![0.0; v.data().len()],
        v.fusion_space().unwrap().as_ref().clone(),
    )
    .unwrap();
    context
        .tensorcontract_fusion_into(
            rule,
            &mut vd,
            v,
            d,
            TensorContractSpec::new(&[2], &[0], OutputAxisOrder::from_axes(&[0, 1, 2])),
            1.0,
            0.0,
        )
        .unwrap();

    for (index, (lhs, rhs)) in tv.data().iter().zip(vd.data()).enumerate() {
        assert!(
            (lhs - rhs).abs() < 1e-9,
            "eigen equation violated at raw position {index}: {lhs} != {rhs}"
        );
    }
}

fn dense_sector_matrices<const A: usize, const B: usize>(
    tensor_nout: usize,
    t: &TensorMap<f64, A, B>,
) -> Vec<(SectorId, usize, usize, Vec<f64>)> {
    // Matricize per coupled sector (rows = codomain trees x degeneracy,
    // cols = domain trees x degeneracy) for dense checks in tests.
    struct SectorAccumulator {
        sector: SectorId,
        rows: usize,
        cols: usize,
        row_trees: Vec<(FusionTreeKey, usize)>,
        col_trees: Vec<(FusionTreeKey, usize)>,
        entries: Vec<(usize, usize, f64)>,
    }
    let structure = std::sync::Arc::clone(t.structure());
    let mut sectors: Vec<SectorAccumulator> = Vec::new();
    for index in 0..structure.block_count() {
        let block = structure.block(index).unwrap();
        let BlockKey::FusionTree(key) = block.key() else {
            continue;
        };
        let sector = key.codomain_tree().coupled();
        let entry = match sectors.iter_mut().find(|entry| entry.sector == sector) {
            Some(entry) => entry,
            None => {
                sectors.push(SectorAccumulator {
                    sector,
                    rows: 0,
                    cols: 0,
                    row_trees: Vec::new(),
                    col_trees: Vec::new(),
                    entries: Vec::new(),
                });
                sectors.last_mut().unwrap()
            }
        };
        let shape = block.shape().to_vec();
        let row_dim: usize = shape[..tensor_nout].iter().product();
        let col_dim: usize = shape[tensor_nout..].iter().product();
        let row_offset = match entry
            .row_trees
            .iter()
            .find(|(tree, _)| tree == key.codomain_tree())
        {
            Some((_, offset)) => *offset,
            None => {
                let offset = entry.rows;
                entry.row_trees.push((key.codomain_tree().clone(), offset));
                entry.rows += row_dim;
                offset
            }
        };
        let col_offset = match entry
            .col_trees
            .iter()
            .find(|(tree, _)| tree == key.domain_tree())
        {
            Some((_, offset)) => *offset,
            None => {
                let offset = entry.cols;
                entry.col_trees.push((key.domain_tree().clone(), offset));
                entry.cols += col_dim;
                offset
            }
        };
        let strides = block.strides().to_vec();
        let offset = block.offset();
        let mut indices = vec![0usize; shape.len()];
        for _ in 0..shape.iter().product::<usize>() {
            let position = offset
                + indices
                    .iter()
                    .zip(&strides)
                    .map(|(&i, &s)| i * s)
                    .sum::<usize>();
            let mut row = 0;
            let mut stride = 1;
            for axis in 0..tensor_nout {
                row += indices[axis] * stride;
                stride *= shape[axis];
            }
            let mut col = 0;
            let mut col_stride = 1;
            for axis in tensor_nout..shape.len() {
                col += indices[axis] * col_stride;
                col_stride *= shape[axis];
            }
            entry
                .entries
                .push((row_offset + row, col_offset + col, t.data()[position]));
            for axis in 0..shape.len() {
                indices[axis] += 1;
                if indices[axis] < shape[axis] {
                    break;
                }
                indices[axis] = 0;
            }
        }
    }
    sectors
        .into_iter()
        .map(|entry| {
            let mut matrix = vec![0.0; entry.rows * entry.cols];
            for (row, col, value) in entry.entries {
                matrix[row + entry.rows * col] = value;
            }
            (entry.sector, entry.rows, entry.cols, matrix)
        })
        .collect()
}

fn assert_orthonormal_columns(matrices: &[(SectorId, usize, usize, Vec<f64>)]) {
    for (sector, rows, cols, matrix) in matrices {
        for left in 0..*cols {
            for right in 0..*cols {
                let mut dot = 0.0;
                for row in 0..*rows {
                    dot += matrix[row + rows * left] * matrix[row + rows * right];
                }
                let expected = if left == right { 1.0 } else { 0.0 };
                assert!(
                    (dot - expected).abs() < 1e-9,
                    "sector {sector:?}: column dot ({left},{right}) = {dot}"
                );
            }
        }
    }
}

fn one_sector_matrix<D: Clone>(data: Vec<D>) -> TensorMap<D, 1, 1> {
    one_sector_rectangular_matrix(data, 2, 2)
}

fn assert_eigh_preflight<D: FactorScalar + std::fmt::Debug>(
    tensor: &TensorMap<D, 1, 1>,
    accepted: bool,
) {
    let mut dense = EighCallSpy::default();
    let error = eigh_full(
        &mut dense,
        &bound_tensor_ref!(Arc::new(Z2FusionRule), tensor),
    )
    .unwrap_err();

    if accepted {
        assert!(matches!(error, OperationError::Dense(_)));
        assert_eq!(dense.calls, 1);
    } else {
        assert!(matches!(error, OperationError::InvalidArgument { .. }));
        assert_eq!(dense.calls, 0);
    }
}

fn one_sector_rectangular_matrix<D: Clone>(
    data: Vec<D>,
    rows: usize,
    cols: usize,
) -> TensorMap<D, 1, 1> {
    let rule = Z2FusionRule;
    let codomain = SectorLeg::new([(SectorId::new(0), rows)], false);
    let domain = SectorLeg::new([(SectorId::new(0), cols)], false);
    let homspace = FusionTreeHomSpace::new(
        FusionProductSpace::new([codomain]),
        FusionProductSpace::new([domain]),
    );
    let space = FusionTensorMapSpace::from_degeneracy_shapes_coupled(
        TensorMapSpace::<1, 1>::from_dims([rows], [cols]).unwrap(),
        homspace,
        &rule,
        vec![vec![rows, cols]],
    )
    .unwrap();
    TensorMap::from_vec_with_fusion_space(data, space).unwrap()
}

/// Path agreement of two spectra: sectors and kept counts exact, values under
/// the workspace rule with the sector's value count as `terms`.
fn assert_spectra_agree(what: &str, lhs: &[SectorSpectrum], rhs: &[SectorSpectrum]) {
    assert_eq!(lhs.len(), rhs.len(), "{what}");
    for (lhs, rhs) in lhs.iter().zip(rhs) {
        assert_eq!(lhs.sector, rhs.sector, "{what}");
        numerics::assert_slices_close(what, &lhs.values, &rhs.values, lhs.values.len());
    }
}

fn assert_real_spectra_close(lhs: &[SectorSpectrum], rhs: &[SectorSpectrum]) {
    assert_eq!(lhs.len(), rhs.len());
    for (lhs, rhs) in lhs.iter().zip(rhs) {
        assert_eq!(lhs.sector, rhs.sector);
        assert_eq!(lhs.values.len(), rhs.values.len());
        for (&lhs, &rhs) in lhs.values.iter().zip(&rhs.values) {
            assert!((lhs - rhs).abs() <= 1e-10, "{lhs} vs {rhs}");
        }
    }
}

fn padded_copy<R, D, const NOUT: usize, const NIN: usize>(
    rule: &R,
    source: &TensorMap<D, NOUT, NIN>,
) -> TensorMap<D, NOUT, NIN>
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64>,
    D: FactorScalar,
{
    let mut offset = 1usize;
    let mut blocks = Vec::with_capacity(source.structure().block_count());
    for index in 0..source.structure().block_count() {
        let block = source.structure().block(index).unwrap();
        blocks.push(
            BlockSpec::column_major_with_key(block.key().clone(), block.shape().to_vec(), offset)
                .unwrap(),
        );
        offset += block.shape().iter().product::<usize>() + 1;
    }
    let source_space = source.fusion_space().unwrap();
    let structure = BlockStructure::from_blocks_with_rank(NOUT + NIN, blocks).unwrap();
    let padded_space = FusionTensorMapSpace::new_unbound(
        source_space.dense_space().clone(),
        source_space.homspace().clone(),
        structure,
    )
    .unwrap()
    .try_bind_rule(rule)
    .unwrap();
    TensorMap::from_block_fn_with_fusion_space(padded_space, D::zero(), |key, indices| {
        let block = source.block_by_key(key).unwrap();
        let position = block.offset()
            + indices
                .iter()
                .zip(block.strides())
                .map(|(&index, &stride)| index * stride)
                .sum::<usize>();
        block.data()[position]
    })
    .unwrap()
}

fn reversed_complete_grid_copy<R, D, const NOUT: usize, const NIN: usize>(
    rule: &R,
    source: &TensorMap<D, NOUT, NIN>,
) -> TensorMap<D, NOUT, NIN>
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64>,
    D: FactorScalar,
{
    let mut offset = 0usize;
    let mut blocks = Vec::with_capacity(source.structure().block_count());
    for index in (0..source.structure().block_count()).rev() {
        let block = source.structure().block(index).unwrap();
        blocks.push(
            BlockSpec::column_major_with_key(block.key().clone(), block.shape().to_vec(), offset)
                .unwrap(),
        );
        offset += block.shape().iter().product::<usize>();
    }
    let source_space = source.fusion_space().unwrap();
    let structure = BlockStructure::from_blocks_with_rank(NOUT + NIN, blocks).unwrap();
    let reordered_space = FusionTensorMapSpace::new_unbound(
        source_space.dense_space().clone(),
        source_space.homspace().clone(),
        structure,
    )
    .unwrap()
    .try_bind_rule(rule)
    .unwrap();
    TensorMap::from_block_fn_with_fusion_space(reordered_space, D::zero(), |key, indices| {
        let block = source.block_by_key(key).unwrap();
        let position = block.offset()
            + indices
                .iter()
                .zip(block.strides())
                .map(|(&index, &stride)| index * stride)
                .sum::<usize>();
        block.data()[position]
    })
    .unwrap()
}

fn reversed_coupled_tree_basis_copy<R, D, const NOUT: usize, const NIN: usize>(
    rule: &R,
    source: &TensorMap<D, NOUT, NIN>,
) -> TensorMap<D, NOUT, NIN>
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64>,
    D: FactorScalar,
{
    let blocks = (0..source.structure().block_count())
        .rev()
        .map(|index| {
            let block = source.structure().block(index).unwrap();
            let BlockKey::FusionTree(key) = block.key() else {
                unreachable!("source has fusion-tree blocks")
            };
            (key.clone(), block.shape().to_vec())
        })
        .collect();
    let structure =
        BlockStructure::coupled_sector_matrix_with_keys(rule, NOUT, NOUT + NIN, blocks).unwrap();
    let source_space = source.fusion_space().unwrap();
    let space = FusionTensorMapSpace::new_unbound(
        source_space.dense_space().clone(),
        source_space.homspace().clone(),
        structure,
    )
    .unwrap()
    .try_bind_rule(rule)
    .unwrap();
    TensorMap::from_block_fn_with_fusion_space(space, D::zero(), |key, indices| {
        let block = source.block_by_key(key).unwrap();
        block.data()[block.offset()
            + indices
                .iter()
                .zip(block.strides())
                .map(|(&index, &stride)| index * stride)
                .sum::<usize>()]
    })
    .unwrap()
}

fn assert_identity_matrices(matrices: &[(SectorId, usize, usize, Vec<f64>)]) {
    assert!(!matrices.is_empty());
    for (sector, rows, cols, matrix) in matrices {
        assert_eq!(rows, cols, "identity block must be square in {sector:?}");
        for col in 0..*cols {
            for row in 0..*rows {
                let expected = if row == col { 1.0 } else { 0.0 };
                let value = matrix[row + rows * col];
                assert!(
                    (value - expected).abs() < 1e-9,
                    "sector {sector:?} ({row},{col}): {value}"
                );
            }
        }
    }
}

fn default_context() -> TensorContractFusionExecutionContext<f64, RuleIdentity> {
    TensorContractFusionExecutionContext::<f64, RuleIdentity>::default()
}

fn u1_cross_space_map<D: FactorScalar>(
    codomain: &[(i32, usize)],
    domain: &[(i32, usize)],
) -> TensorMap<D, 1, 1> {
    let codomain_leg = SectorLeg::new(
        codomain
            .iter()
            .map(|&(charge, degeneracy)| (U1Irrep::new(charge).sector_id(), degeneracy)),
        false,
    );
    let domain_leg = SectorLeg::new(
        domain
            .iter()
            .map(|&(charge, degeneracy)| (U1Irrep::new(charge).sector_id(), degeneracy)),
        false,
    );
    let homspace = FusionTreeHomSpace::new(
        FusionProductSpace::new([codomain_leg.clone()]),
        FusionProductSpace::new([domain_leg.clone()]),
    );
    let shapes = homspace
        .fusion_tree_keys(&U1FusionRule)
        .iter()
        .map(|key| {
            let coupled = key.codomain_tree().coupled();
            vec![
                codomain_leg.degeneracy(coupled).unwrap(),
                domain_leg.degeneracy(coupled).unwrap(),
            ]
        })
        .collect::<Vec<_>>();
    let space = FusionTensorMapSpace::from_degeneracy_shapes_coupled(
        TensorMapSpace::<1, 1>::from_dims(
            [codomain.iter().map(|(_, degeneracy)| degeneracy).sum()],
            [domain.iter().map(|(_, degeneracy)| degeneracy).sum()],
        )
        .unwrap(),
        homspace,
        &U1FusionRule,
        shapes,
    )
    .unwrap();
    TensorMap::from_block_fn_with_fusion_space(space, D::zero(), |_, indices| {
        if indices[0] == indices[1] {
            D::one()
        } else {
            D::zero()
        }
    })
    .unwrap()
}

fn u1_block_endomorphism<D>(blocks: &[(i32, usize, Vec<D>)]) -> TensorMap<D, 1, 1>
where
    D: Copy + Zero,
{
    let blocks = blocks
        .iter()
        .map(|(charge, dimension, data)| {
            (U1Irrep::new(*charge).sector_id(), *dimension, data.clone())
        })
        .collect::<Vec<_>>();
    block_endomorphism(&U1FusionRule, &blocks)
}

/// `1 <- 1` endomorphism with one fusion tree per coupled sector, on any rule.
fn block_endomorphism<R, D>(rule: &R, blocks: &[(SectorId, usize, Vec<D>)]) -> TensorMap<D, 1, 1>
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64>,
    D: Copy + Zero,
{
    let sectors = blocks
        .iter()
        .map(|(sector, dimension, _)| (*sector, *dimension))
        .collect::<Vec<_>>();
    let leg = SectorLeg::new(sectors.iter().copied(), false);
    let total_dimension = sectors.iter().map(|(_, dimension)| dimension).sum();
    let homspace = FusionTreeHomSpace::new(
        FusionProductSpace::new([leg.clone()]),
        FusionProductSpace::new([leg]),
    );
    let shapes = homspace
        .fusion_tree_keys(rule)
        .iter()
        .map(|key| {
            let coupled = key.codomain_tree().coupled();
            let (_, dimension, data) = blocks
                .iter()
                .find(|(sector, _, _)| *sector == coupled)
                .unwrap();
            assert_eq!(data.len(), dimension * dimension);
            vec![*dimension, *dimension]
        })
        .collect::<Vec<_>>();
    let space = FusionTensorMapSpace::from_degeneracy_shapes_coupled(
        TensorMapSpace::<1, 1>::from_dims([total_dimension], [total_dimension]).unwrap(),
        homspace,
        rule,
        shapes,
    )
    .unwrap();
    TensorMap::from_block_fn_with_fusion_space(space, D::zero(), |key, indices| {
        let BlockKey::FusionTree(tree) = key else {
            return D::zero();
        };
        let coupled = tree.codomain_tree().coupled();
        let (_, dimension, data) = blocks
            .iter()
            .find(|(sector, _, _)| *sector == coupled)
            .unwrap();
        data[indices[0] + dimension * indices[1]]
    })
    .unwrap()
}

fn scalar_u1_block<D: Copy>(tensor: &TensorMap<D, 1, 1>, charge: i32) -> D {
    scalar_block(tensor, U1Irrep::new(charge).sector_id())
}

fn scalar_block<D: Copy>(tensor: &TensorMap<D, 1, 1>, sector: SectorId) -> D {
    let structure = tensor.structure();
    let block = (0..structure.block_count())
        .map(|index| structure.block(index).unwrap())
        .find(|block| {
            let BlockKey::FusionTree(key) = block.key() else {
                return false;
            };
            key.codomain_tree().coupled() == sector
        })
        .unwrap();
    assert_eq!(block.shape(), &[1, 1]);
    tensor.data()[block.offset()]
}

fn assert_identity_sector_matrices(matrices: &[(SectorId, usize, usize, Vec<f64>)]) {
    for (sector, rows, cols, matrix) in matrices {
        assert_eq!(rows, cols, "sector {sector:?}: expected square factor");
        for col in 0..*cols {
            for row in 0..*rows {
                let expected = if row == col { 1.0 } else { 0.0 };
                let value = matrix[row + rows * col];
                assert!(
                    (value - expected).abs() < 1e-9,
                    "sector {sector:?}: entry ({row},{col}) = {value}"
                );
            }
        }
    }
}

/// The `V = U1Space(0=>3, 1=>2)` oracle fill, in 0-based degeneracy indices.
fn exp_oracle_fill(charge: i32, row: usize, column: usize, scale: f64) -> f64 {
    scale * (0.5 + 0.25 * row as f64 - 0.75 * column as f64 + 0.125 * charge as f64)
}

fn exp_oracle_fill_imaginary(row: usize, column: usize, scale: f64) -> f64 {
    scale * (0.125 * row as f64 + 0.375 * column as f64 - 0.25)
}

fn exp_oracle_block<D: FactorScalar>(charge: i32, order: usize, scale: f64) -> Vec<D> {
    let mut data = vec![D::zero(); order * order];
    for column in 0..order {
        for row in 0..order {
            let real = exp_oracle_fill(charge, row, column, scale);
            let imaginary = if D::epsilon() == f64::EPSILON && size_of::<D>() == size_of::<f64>() {
                0.0
            } else {
                exp_oracle_fill_imaginary(row, column, scale)
            };
            data[row + order * column] = D::from_complex64(Complex64::new(real, imaginary));
        }
    }
    data
}

fn exp_oracle_tensor<D: FactorScalar>(scale: f64) -> TensorMap<D, 1, 1> {
    u1_block_endomorphism(&[
        (0, 3, exp_oracle_block::<D>(0, 3, scale)),
        (1, 2, exp_oracle_block::<D>(1, 2, scale)),
    ])
}

#[derive(Default)]
struct MatrixFunctionCallSpy {
    inner: tenet_dense::DefaultDenseExecutor,
    eigh_calls: usize,
    solve_calls: usize,
    matmul_calls: usize,
    /// Ordinal of a solve that must fail, for the failure-atomicity gate.
    fail_solve_number: Option<usize>,
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

/// Copy of `source` whose coupled sectors stack rows by ascending codomain
/// tree and columns by descending domain tree: the same operator, but row `i`
/// and column `i` of a multi-tree sector name different tree states.
fn mis_stacked_endomorphism_copy<R, D>(rule: &R, source: &TensorMap<D, 2, 2>) -> TensorMap<D, 2, 2>
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64>,
    D: FactorScalar,
{
    let mut blocks: Vec<(tenet_core::FusionTreePairKey, Vec<usize>)> =
        (0..source.structure().block_count())
            .map(|index| {
                let block = source.structure().block(index).unwrap();
                let BlockKey::FusionTree(key) = block.key() else {
                    unreachable!("source has fusion-tree blocks")
                };
                (key.clone(), block.shape().to_vec())
            })
            .collect();
    blocks.sort_by(|(a, _), (b, _)| {
        a.codomain_tree()
            .cmp(b.codomain_tree())
            .then(b.domain_tree().cmp(a.domain_tree()))
    });
    let structure = BlockStructure::coupled_sector_matrix_with_keys(rule, 2, 4, blocks).unwrap();
    let regions = structure.coupled_sector_regions(2).unwrap().unwrap();
    assert!(
        regions
            .iter()
            .any(|region| region.row_trees() != region.col_trees()),
        "the fixture must mis-stack at least one coupled sector"
    );
    let source_space = source.fusion_space().unwrap();
    let space = FusionTensorMapSpace::new_unbound(
        source_space.dense_space().clone(),
        source_space.homspace().clone(),
        structure,
    )
    .unwrap()
    .try_bind_rule(rule)
    .unwrap();
    TensorMap::from_block_fn_with_fusion_space(space, D::zero(), |key, indices| {
        let block = source.block_by_key(key).unwrap();
        block.data()[block.offset()
            + indices
                .iter()
                .zip(block.strides())
                .map(|(&index, &stride)| index * stride)
                .sum::<usize>()]
    })
    .unwrap()
}

fn assert_stacking_refusal<T: fmt::Debug>(result: Result<T, OperationError>, operation: &str) {
    match result {
        Err(OperationError::UnsupportedTensorContractScope { message })
            if message.starts_with(operation) && message.contains("stacking") => {}
        other => panic!("{operation}: expected a stacking refusal, got {other:?}"),
    }
}

mod qr_lq;
mod svd;
mod eigh_eig;
mod polar;
mod null_space;
mod matrix_functions;
mod dispatch;
