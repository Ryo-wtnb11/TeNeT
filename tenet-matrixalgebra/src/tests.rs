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
    dyn_space_of, map_square_sectors_dyn_into, truncate_svd, typed_from_bound_factor,
    typed_from_dyn, validate_eigenvector_singular_values, validate_inverse_region_routes_for_test,
    BoundTensorMap,
};
use crate::*;
use num_complex::{Complex32, Complex64};
use num_traits::Zero;
use std::{cell::Cell, convert::Infallible, fmt, sync::Arc};
use tenet_dense::{
    DenseBackend, DenseDotConfig, DenseError, DenseExecutor, DenseRead, DenseTensor, DenseWrite,
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
struct FailAfterObservingEighInput {
    observed: Vec<Vec<f64>>,
    outputs: Option<Vec<DenseTensor>>,
}

#[derive(Default)]
struct EighCallSpy {
    calls: usize,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ValuesOperation {
    Svd,
    Eigh,
    Eig,
}

struct FailSecondValues {
    inner: tenet_dense::DefaultDenseExecutor,
    operation: ValuesOperation,
    calls: usize,
}

impl FailSecondValues {
    fn new(operation: ValuesOperation) -> Self {
        Self {
            inner: tenet_dense::DefaultDenseExecutor::new(),
            operation,
            calls: 0,
        }
    }

    fn fail(&mut self, operation: ValuesOperation) -> Result<(), DenseError> {
        assert_eq!(self.operation, operation);
        self.calls += 1;
        if self.calls == 2 {
            Err(DenseError::Backend {
                backend: DenseBackend::Tenferro,
                op: "values",
                message: "injected second-sector failure".to_string(),
            })
        } else {
            Ok(())
        }
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

impl DenseExecutor for FailSecondValues {
    fn svd(&mut self, _: DenseRead<'_>) -> Result<Vec<DenseTensor>, DenseError> {
        panic!("test only exercises values-only operations")
    }

    fn qr(&mut self, _: DenseRead<'_>) -> Result<Vec<DenseTensor>, DenseError> {
        panic!("test only exercises values-only operations")
    }

    fn eigh(&mut self, _: DenseRead<'_>) -> Result<Vec<DenseTensor>, DenseError> {
        panic!("test only exercises values-only operations")
    }

    fn svd_vals(&mut self, input: DenseRead<'_>) -> Result<DenseTensor, DenseError> {
        self.fail(ValuesOperation::Svd)?;
        self.inner.svd_vals(input)
    }

    fn eigh_vals(&mut self, input: DenseRead<'_>) -> Result<DenseTensor, DenseError> {
        self.fail(ValuesOperation::Eigh)?;
        self.inner.eigh_vals(input)
    }

    fn eig_vals(&mut self, input: DenseRead<'_>) -> Result<DenseTensor, DenseError> {
        self.fail(ValuesOperation::Eig)?;
        self.inner.eig_vals(input)
    }

    fn dot_general_into(
        &mut self,
        _: DenseWrite<'_>,
        _: DenseRead<'_>,
        _: DenseRead<'_>,
        _: &DenseDotConfig,
    ) -> Result<(), DenseError> {
        panic!("test only exercises values-only operations")
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

fn scale_vt_rows_by_singular_values<const NIN: usize>(
    vt: &mut TensorMap<f64, 1, NIN>,
    singular_values: &[SectorSpectrum],
) {
    let structure = std::sync::Arc::clone(vt.structure());
    for index in 0..structure.block_count() {
        let block = structure.block(index).unwrap();
        let BlockKey::FusionTree(key) = block.key() else {
            continue;
        };
        let sector = key.codomain_tree().coupled();
        let values = &singular_values
            .iter()
            .find(|entry| entry.sector == sector)
            .expect("singular values for every Vt sector")
            .values;
        let shape = block.shape().to_vec();
        let count = shape.iter().product::<usize>();
        let mut multi_index = vec![0usize; shape.len()];
        for _ in 0..count {
            let position = block.offset()
                + multi_index
                    .iter()
                    .zip(block.strides())
                    .map(|(&i, &s)| i * s)
                    .sum::<usize>();
            vt.data_mut()[position] *= values[multi_index[0]];
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

fn run_tsvd_reconstruction_case<R>(rule: &R, sectors: &[SectorId], coupled_layout: bool)
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64> + TreeTransformRuleCacheKey + Clone,
{
    let degeneracy = 2usize;
    let leg = || SectorLeg::new(sectors.iter().map(|&sector| (sector, degeneracy)), false);
    let leg_dim = sectors.len() * degeneracy;
    let homspace = FusionTreeHomSpace::new(
        FusionProductSpace::new([leg(), leg()]),
        FusionProductSpace::new([leg(), leg()]),
    );
    let key_count = homspace.fusion_tree_keys(rule).len();
    let dense = TensorMapSpace::<2, 2>::from_dims([leg_dim, leg_dim], [leg_dim, leg_dim]).unwrap();
    let shapes = vec![vec![degeneracy; 4]; key_count];
    let space = if coupled_layout {
        FusionTensorMapSpace::from_degeneracy_shapes_coupled(dense, homspace, rule, shapes).unwrap()
    } else {
        FusionTensorMapSpace::from_degeneracy_shapes(dense, homspace, rule, shapes).unwrap()
    };
    let len = space.required_len().unwrap();
    let tensor = TensorMap::<f64, 2, 2>::from_vec_with_fusion_space(
        (0..len)
            .map(|index| ((index * 7 + 3) % 23) as f64 * 0.5 - 5.0)
            .collect(),
        space,
    )
    .unwrap();

    let mut dense_executor = tenet_dense::DefaultDenseExecutor::new();
    let svd = svd_trunc(
        &mut dense_executor,
        &bound_tensor_ref!(Arc::new(rule.clone()), &tensor),
        &Truncation::Full,
    )
    .unwrap();
    assert_factor_layout_matches_legacy_shapes(svd.u.space());
    assert_factor_layout_matches_legacy_shapes(svd.s.space());
    assert_factor_layout_matches_legacy_shapes(svd.vh.space());

    for entry in &svd.singular_values {
        for pair in entry.values.windows(2) {
            assert!(
                pair[0] >= pair[1] - 1e-12,
                "singular values must be descending in sector {:?}",
                entry.sector
            );
        }
        assert!(entry.values.iter().all(|&value| value >= -1e-12));
    }

    let mut scaled_vt = svd.vh.tensor().clone();
    scale_vt_rows_by_singular_values(&mut scaled_vt, &svd.singular_values);

    let mut reconstructed = TensorMap::<f64, 2, 2>::from_vec_with_fusion_space(
        vec![0.0; len],
        tensor.fusion_space().unwrap().as_ref().clone(),
    )
    .unwrap();
    let mut context = TensorContractFusionExecutionContext::<f64, R::Key>::default();
    context
        .tensorcontract_fusion_into(
            rule,
            &mut reconstructed,
            &svd.u,
            &scaled_vt,
            TensorContractSpec::new(&[2], &[0], OutputAxisOrder::from_axes(&[0, 1, 2, 3])),
            1.0,
            0.0,
        )
        .unwrap();

    assert_svd_blocks_match(&tensor, &reconstructed);
}

#[test]
fn tsvd_fusion_reconstructs_z2_tensor_packed_layout() {
    run_tsvd_reconstruction_case(&Z2FusionRule, &[SectorId::new(0), SectorId::new(1)], false);
}

#[test]
fn tsvd_fusion_reconstructs_z2_tensor_coupled_layout() {
    run_tsvd_reconstruction_case(&Z2FusionRule, &[SectorId::new(0), SectorId::new(1)], true);
}

fn assert_compact_svd_direct_copy_probe() {
    let probe = crate::factorize::compact_svd_copy_probe();
    assert_eq!(probe.input_pack_calls, 0);
    assert_eq!(probe.input_pack_bytes, 0);
    assert_eq!(probe.output_scatter_calls, 0);
    assert_eq!(probe.output_scatter_bytes, 0);
}

#[test]
fn compact_svd_canonical_layout_skips_input_pack_and_factor_scatter() {
    // What: canonical coupled storage reaches final factor destinations without numerical copies.
    let rule = Z2FusionRule;
    let tensor = tsvd_test_tensor(&rule, &[SectorId::new(0), SectorId::new(1)]);
    let mut dense = tenet_dense::DefaultDenseExecutor::new();

    crate::factorize::reset_compact_svd_copy_probe();
    svd_compact(&mut dense, &bound_tensor_ref!(Arc::new(rule), &tensor)).unwrap();

    assert_compact_svd_direct_copy_probe();
}

#[test]
fn compact_svd_noncanonical_layout_uses_copy_fallback() {
    // What: an expert noncanonical view retains the general pack-and-scatter implementation.
    let rule = Z2FusionRule;
    let tensor = tsvd_test_tensor(&rule, &[SectorId::new(0), SectorId::new(1)]);
    let bound = bound_tensor(Arc::new(rule), &tensor);
    let adjoint_space = bound.space().adjoint_view().unwrap();
    let input = BoundDynamicTensorRef::try_new(&adjoint_space, bound.data()).unwrap();
    let mut dense = tenet_dense::DefaultDenseExecutor::new();

    crate::factorize::reset_compact_svd_copy_probe();
    svd_compact_dyn(&mut dense, &input).unwrap();
    let probe = crate::factorize::compact_svd_copy_probe();

    assert!(probe.input_pack_calls > 0);
    assert!(probe.input_pack_bytes > 0);
    assert!(probe.output_scatter_calls > 0);
    assert!(probe.output_scatter_bytes > 0);
}

#[test]
fn compact_qr_canonical_layout_skips_input_pack_and_factor_scatter() {
    // What: canonical compact QR reads source regions and writes final factor regions directly.
    let rule = Z2FusionRule;
    let tensor = tsvd_test_tensor(&rule, &[SectorId::new(0), SectorId::new(1)]);
    let mut dense = tenet_dense::DefaultDenseExecutor::new();

    crate::factorize::reset_compact_qr_copy_probe();
    qr_compact(&mut dense, &bound_tensor_ref!(Arc::new(rule), &tensor)).unwrap();

    let probe = crate::factorize::compact_qr_copy_probe();
    assert_eq!(probe.input_pack_bytes, 0);
    assert_eq!(probe.output_scatter_bytes, 0);
}

#[test]
fn compact_qr_noncanonical_layout_uses_copy_fallback() {
    // What: expert noncanonical compact QR retains positive pack-and-scatter copy evidence.
    let rule = Z2FusionRule;
    let tensor = tsvd_test_tensor(&rule, &[SectorId::new(0), SectorId::new(1)]);
    let bound = bound_tensor(Arc::new(rule), &tensor);
    let adjoint_space = bound.space().adjoint_view().unwrap();
    let input = BoundDynamicTensorRef::try_new(&adjoint_space, bound.data()).unwrap();
    let mut dense = tenet_dense::DefaultDenseExecutor::new();

    crate::factorize::reset_compact_qr_copy_probe();
    qr_compact_dyn(&mut dense, &input).unwrap();
    let probe = crate::factorize::compact_qr_copy_probe();

    assert!(probe.input_pack_bytes > 0);
    assert!(probe.output_scatter_bytes > 0);
}

#[test]
fn compact_qr_lq_noncanonical_layout_does_not_call_qr_into() {
    // The fallback still consumes `qr` outputs; its scatter is TeNeT-owned.
    let rule = Z2FusionRule;
    let tensor = tsvd_test_tensor(&rule, &[SectorId::new(0), SectorId::new(1)]);
    let bound = bound_tensor(Arc::new(rule), &tensor);
    let adjoint_space = bound.space().adjoint_view().unwrap();
    let input = BoundDynamicTensorRef::try_new(&adjoint_space, bound.data()).unwrap();
    let mut dense = FailAfterObservingQrInput {
        qr_succeeds: true,
        ..Default::default()
    };

    qr_compact_dyn(&mut dense, &input).unwrap();
    lq_compact_dyn(&mut dense, &input).unwrap();

    assert!(!dense.observed.is_empty());
}

#[test]
fn eigh_canonical_layout_skips_input_pack_and_vector_scatter() {
    // What: canonical EIGH reads source regions and writes final eigenvector regions directly.
    let rule = Z2FusionRule;
    let tensor = hermitian_test_tensor(&rule, &[SectorId::new(0), SectorId::new(1)]);
    let mut dense = tenet_dense::DefaultDenseExecutor::new();

    crate::factorize::reset_eigh_copy_probe();
    eigh_full(&mut dense, &bound_tensor_ref!(Arc::new(rule), &tensor)).unwrap();

    assert_eq!(
        crate::factorize::eigh_copy_probe(),
        crate::factorize::EighCopyProbe::default()
    );
}

#[test]
fn compact_lq_canonical_layout_uses_only_bounded_adjoint_copies() {
    // What: canonical compact LQ avoids general pack/scatter while accounting for its three reusable scratch buffers and required adjoint copies.
    let rule = Z2FusionRule;
    let tensor = tsvd_test_tensor(&rule, &[SectorId::new(0), SectorId::new(1)]);
    let mut dense = tenet_dense::DefaultDenseExecutor::new();

    crate::factorize::reset_compact_lq_copy_probe();
    let (left, right) =
        lq_compact(&mut dense, &bound_tensor_ref!(Arc::new(rule), &tensor)).unwrap();
    let probe = crate::factorize::compact_lq_copy_probe();

    assert_eq!(probe.input_pack_calls, 0);
    assert_eq!(probe.input_pack_bytes, 0);
    assert_eq!(probe.output_scatter_calls, 0);
    assert_eq!(probe.output_scatter_bytes, 0);
    assert_eq!(probe.scratch_buffer_count, 1);
    assert!(probe.scratch_capacity_bytes > 0);
    assert!(probe.adjoint_scratch_fill_calls > 0);
    assert_eq!(
        probe.adjoint_scratch_fill_bytes,
        std::mem::size_of_val(tensor.data())
    );
    assert!(probe.final_adjoint_copy_calls > 0);
    assert_eq!(
        probe.final_adjoint_copy_bytes,
        (left.data().len() + right.data().len()) * std::mem::size_of::<f64>()
    );
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

fn one_sector_generic_factorization_input(
) -> (BoundDynamicFusionMapSpace<FactorGenericRule>, Vec<f64>) {
    let provider = Arc::new(FactorGenericRule);
    let vacuum = SectorId::new(0);
    let x = SectorId::new(1);
    let homspace = FusionTreeHomSpace::new(
        FusionProductSpace::new([
            SectorLeg::new([(x, 2)], false),
            SectorLeg::new([(x, 1)], false),
        ]),
        FusionProductSpace::new([
            SectorLeg::new([(x, 3)], false),
            SectorLeg::new([(vacuum, 1)], false),
        ]),
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

fn assert_generic_factor_close(
    actual: &BoundDynFactor<FactorGenericRule, f64>,
    expected: &BoundDynFactor<FactorGenericRule, f64>,
) {
    assert_eq!(
        actual.space().space().homspace(),
        expected.space().space().homspace()
    );
    assert_eq!(actual.data().len(), expected.data().len());
    for (&actual, &expected) in actual.data().iter().zip(expected.data()) {
        assert!((actual - expected).abs() < 1.0e-12);
    }
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

#[test]
fn provider_neutral_generic_compact_factorizations_remain_covered() {
    let (space, data) = generic_factorization_input();
    let input = BoundDynamicTensorRef::try_new(&space, &data).unwrap();
    let mut dense = tenet_dense::DefaultDenseExecutor::new();

    assert!(!svd_vals_dyn_generic(&mut dense, &input).unwrap().is_empty());
    let (u, vh, values) = svd_compact_factors_dyn_generic(&mut dense, &input).unwrap();
    assert!(!values.is_empty());
    assert!(Arc::ptr_eq(u.space().provider_arc(), space.provider_arc()));
    assert!(Arc::ptr_eq(vh.space().provider_arc(), space.provider_arc()));
    qr_compact_dyn_generic(&mut dense, &input).unwrap();
    lq_compact_dyn_generic(&mut dense, &input).unwrap();
}

#[test]
fn direct_compact_svd_uses_owned_executor_outputs_only() {
    let tensor = rectangular_svd_tensor(3, 2);
    let mut direct = RejectSvdInto::default();
    svd_compact(
        &mut direct,
        &bound_tensor_ref!(Arc::new(Z2FusionRule), &tensor),
    )
    .unwrap();
    assert_eq!(direct.svd_calls, 1);
    assert_eq!(direct.svd_into_calls, 0);

    let (space, data) = generic_factorization_input();
    let input = BoundDynamicTensorRef::try_new(&space, &data).unwrap();
    let mut generic = RejectSvdInto::default();
    svd_compact_factors_dyn_generic(&mut generic, &input).unwrap();
    assert!(generic.svd_calls > 0);
    assert_eq!(generic.svd_into_calls, 0);

    let bound = bound_tensor(Arc::new(Z2FusionRule), &tensor);
    let mut polar = RejectSvdInto::default();
    let mut context = default_context();
    left_polar(&mut polar, &mut context, &bound.as_ref()).unwrap();
    assert_eq!(polar.svd_calls, 1);
    assert_eq!(polar.svd_into_calls, 0);

    let fallback_space = bound.space().adjoint_view().unwrap();
    let fallback = BoundDynamicTensorRef::try_new(&fallback_space, bound.data()).unwrap();
    let mut legacy = RejectSvdInto::default();
    assert!(matches!(
        svd_compact_dyn(&mut legacy, &fallback),
        Err(OperationError::Dense(DenseError::Backend {
            op: "svd_into",
            ..
        }))
    ));
    assert_eq!(legacy.svd_calls, 0);
    assert_eq!(legacy.svd_into_calls, 1);
}

#[test]
fn generic_compact_qr_lq_paths_do_not_call_qr_into() {
    let (canonical_space, canonical_data) = generic_factorization_input();
    let canonical = BoundDynamicTensorRef::try_new(&canonical_space, &canonical_data).unwrap();
    let (fallback_space, fallback_data) =
        expert_generic_factorization_input(&canonical_space, &canonical_data, true);
    let fallback = BoundDynamicTensorRef::try_new(&fallback_space, &fallback_data).unwrap();

    let mut direct_dense = FailAfterObservingQrInput {
        qr_succeeds: true,
        ..Default::default()
    };
    qr_compact_dyn_generic(&mut direct_dense, &canonical).unwrap();
    lq_compact_dyn_generic(&mut direct_dense, &canonical).unwrap();
    assert!(!direct_dense.observed.is_empty());

    let mut fallback_dense = FailAfterObservingQrInput {
        qr_succeeds: true,
        ..Default::default()
    };
    qr_compact_dyn_generic(&mut fallback_dense, &fallback).unwrap();
    lq_compact_dyn_generic(&mut fallback_dense, &fallback).unwrap();
    assert!(!fallback_dense.observed.is_empty());
}

#[test]
fn provider_neutral_generic_factorizations_keep_the_strided_fallback() {
    let (canonical_space, canonical_data) = generic_factorization_input();
    let (padded_space, padded_data) =
        padded_generic_factorization_input(&canonical_space, &canonical_data);
    let canonical = BoundDynamicTensorRef::try_new(&canonical_space, &canonical_data).unwrap();
    let padded = BoundDynamicTensorRef::try_new(&padded_space, &padded_data).unwrap();
    let mut dense = tenet_dense::DefaultDenseExecutor::new();

    let canonical_values = svd_vals_dyn_generic(&mut dense, &canonical).unwrap();
    let padded_values = svd_vals_dyn_generic(&mut dense, &padded).unwrap();
    assert_real_spectra_close(&padded_values, &canonical_values);

    let canonical_svd = svd_compact_factors_dyn_generic(&mut dense, &canonical).unwrap();
    let padded_svd = svd_compact_factors_dyn_generic(&mut dense, &padded).unwrap();
    assert_generic_factor_close(&padded_svd.0, &canonical_svd.0);
    assert_generic_factor_close(&padded_svd.1, &canonical_svd.1);
    assert_real_spectra_close(&padded_svd.2, &canonical_svd.2);

    let canonical_qr = qr_compact_dyn_generic(&mut dense, &canonical).unwrap();
    let padded_qr = qr_compact_dyn_generic(&mut dense, &padded).unwrap();
    assert_generic_factor_close(&padded_qr.0, &canonical_qr.0);
    assert_generic_factor_close(&padded_qr.1, &canonical_qr.1);

    let canonical_lq = lq_compact_dyn_generic(&mut dense, &canonical).unwrap();
    let padded_lq = lq_compact_dyn_generic(&mut dense, &padded).unwrap();
    assert_generic_factor_close(&padded_lq.0, &canonical_lq.0);
    assert_generic_factor_close(&padded_lq.1, &canonical_lq.1);
}

#[test]
fn generic_pair_publication_keeps_reordered_tree_scatter_fallback() {
    let (canonical_space, canonical_data) = generic_factorization_input();
    let (reordered_space, reordered_data) =
        expert_generic_factorization_input(&canonical_space, &canonical_data, true);
    let reordered = BoundDynamicTensorRef::try_new(&reordered_space, &reordered_data).unwrap();
    let mut dense = tenet_dense::DefaultDenseExecutor::new();

    crate::factorize::reset_generic_pair_publication_probe();
    crate::factorize::reset_compact_qr_copy_probe();
    let actual_qr = qr_compact_dyn_generic(&mut dense, &reordered).unwrap();
    assert!(!actual_qr.0.data().is_empty());
    assert!(!actual_qr.1.data().is_empty());
    let probe = crate::factorize::generic_pair_publication_probe();
    assert_eq!(
        (probe.canonical_publications, probe.fallback_publications),
        (0, 1)
    );
    assert!(probe.left_scattered_elements > 0);
    assert!(probe.right_scattered_elements > 0);
    let qr_copy = crate::factorize::compact_qr_copy_probe();
    assert_eq!(
        qr_copy.output_scatter_calls,
        probe.left_scatter_calls + probe.right_scatter_calls
    );
    assert_eq!(
        qr_copy.output_scatter_bytes,
        (actual_qr.0.data().len() + actual_qr.1.data().len()) * std::mem::size_of::<f64>()
    );

    crate::factorize::reset_generic_pair_publication_probe();
    crate::factorize::reset_compact_lq_copy_probe();
    let actual_lq = lq_compact_dyn_generic(&mut dense, &reordered).unwrap();
    assert!(!actual_lq.0.data().is_empty());
    assert!(!actual_lq.1.data().is_empty());
    let probe = crate::factorize::generic_pair_publication_probe();
    assert_eq!(
        (probe.canonical_publications, probe.fallback_publications),
        (0, 1)
    );
    assert!(probe.left_scattered_elements > 0);
    assert!(probe.right_scattered_elements > 0);
    let lq_copy = crate::factorize::compact_lq_copy_probe();
    assert_eq!(
        lq_copy.output_scatter_calls,
        probe.left_scatter_calls + probe.right_scatter_calls
    );
    assert_eq!(
        lq_copy.output_scatter_bytes,
        (actual_lq.0.data().len() + actual_lq.1.data().len()) * std::mem::size_of::<f64>()
    );
}

#[test]
fn checked_generic_pair_publication_reuses_one_sector_owners() {
    let (source, data) = one_sector_generic_factorization_input();
    let (provider, checked) = bind_checked_only(&source);
    let input = BoundDynamicTensorRef::try_new(&checked, &data).unwrap();
    let mut dense = tenet_dense::DefaultDenseExecutor::new();
    crate::factorize::reset_generic_pair_publication_probe();

    let (left, right) = qr_compact_dyn_checked_generic(&mut dense, &input).unwrap();

    let probe = crate::factorize::generic_pair_publication_probe();
    let blocks = left.space().space().structure().block_count()
        + right.space().space().structure().block_count();
    assert_eq!(probe.canonical_publications, 1);
    assert_eq!(probe.fallback_publications, 0);
    assert_eq!((probe.left_owner_reused, probe.right_owner_reused), (1, 1));
    assert_eq!(
        (probe.left_appended_elements, probe.right_appended_elements),
        (0, 0)
    );
    assert_eq!(
        (
            probe.left_scattered_elements,
            probe.right_scattered_elements
        ),
        (0, 0)
    );
    // The checked route validates each staged key once (one cursor event per
    // block) and commits the structure it validated, so the admitted-key
    // equality of a two-source construction (formerly one visit and one more
    // event per block: `blocks` visits, `2 * blocks` events) no longer runs.
    assert_eq!(probe.output_blocks_visited, 0);
    assert_eq!(probe.ordered_key_validation_events, blocks);
    assert_eq!(
        (probe.fallback_row_lookups, probe.fallback_col_lookups),
        (0, 0)
    );
    assert!(Arc::ptr_eq(left.space().provider_arc(), &provider));
    assert!(Arc::ptr_eq(right.space().provider_arc(), &provider));
}

fn assert_pair_reconstructs_checked_literal<R, D>(
    left: &BoundDynFactor<R, D>,
    right: &BoundDynFactor<R, D>,
    complex: bool,
) where
    D: FactorScalar,
{
    let left_regions = left
        .space()
        .space()
        .structure()
        .coupled_sector_regions(left.space().space().nout())
        .unwrap()
        .unwrap();
    let right_regions = right
        .space()
        .space()
        .structure()
        .coupled_sector_regions(right.space().space().nout())
        .unwrap()
        .unwrap();
    for sector in [SectorId::new(0), SectorId::new(1)] {
        let left_region = left_regions
            .iter()
            .find(|region| region.coupled() == sector)
            .unwrap();
        let right_region = right_regions
            .iter()
            .find(|region| region.coupled() == sector)
            .unwrap();
        let (rows, cols, expected) = checked_svd_matrix(sector, complex);
        assert_eq!((left_region.rows(), right_region.cols()), (rows, cols));
        assert_eq!(left_region.cols(), right_region.rows());
        let kept = left_region.cols();
        for col in 0..cols {
            for row in 0..rows {
                let actual = (0..kept).fold(D::zero(), |sum, bond| {
                    sum + left.data()[left_region.range().start + row + rows * bond]
                        * right.data()[right_region.range().start + bond + kept * col]
                });
                assert!((actual.widen_complex() - expected[row + rows * col]).norm() < 1.0e-10);
            }
        }
    }
}

#[test]
#[expect(
    clippy::arc_with_non_send_sync,
    reason = "the checked Generic API requires Arc identity while Cell is a single-threaded call spy"
)]
fn staged_generic_pair_callers_publish_canonical_owned_payloads() {
    let (source, data) = one_sector_generic_factorization_input();
    let (padded_space, padded_data) = padded_generic_factorization_input(&source, &data);
    let padded = BoundDynamicTensorRef::try_new(&padded_space, &padded_data).unwrap();
    let provider = Arc::new(LateGenericSpy {
        rule: FactorGenericRule,
        fail_at: usize::MAX,
        calls: Cell::new(0),
    });
    let checked =
        BoundDynamicFusionMapSpace::bind_generic(source.space().clone(), Arc::clone(&provider))
            .unwrap();
    let checked_input = BoundDynamicTensorRef::try_new(&checked, &data).unwrap();
    let mut dense = tenet_dense::DefaultDenseExecutor::new();
    crate::factorize::reset_generic_pair_publication_probe();

    qr_compact_dyn_generic(&mut dense, &padded).unwrap();
    lq_compact_dyn_generic(&mut dense, &padded).unwrap();
    qr_compact_dyn_checked_generic(&mut dense, &checked_input).unwrap();
    svd_compact_dyn_checked_generic(&mut dense, &checked_input).unwrap();
    lq_compact_dyn_checked_generic(&mut dense, &checked_input).unwrap();
    qr_full_dyn_checked_generic(&mut dense, &checked_input).unwrap();
    lq_full_dyn_checked_generic(&mut dense, &checked_input).unwrap();

    let probe = crate::factorize::generic_pair_publication_probe();
    assert_eq!(probe.canonical_publications, 7);
    assert_eq!(probe.fallback_publications, 0);
    assert_eq!((probe.left_owner_reused, probe.right_owner_reused), (7, 7));
    assert_eq!(
        (probe.left_appended_elements, probe.right_appended_elements),
        (0, 0)
    );
    assert_eq!(
        (
            probe.left_scattered_elements,
            probe.right_scattered_elements
        ),
        (0, 0)
    );
}

fn assert_checked_generic_pair_publication_appends_multiple_literal_sectors<D>(complex: bool)
where
    D: FactorScalar,
{
    let (provider, space, data) = checked_svd_truncation_input::<D>(complex);
    let input = BoundDynamicTensorRef::try_new(&space, &data).unwrap();
    let mut dense = tenet_dense::DefaultDenseExecutor::new();
    crate::factorize::reset_generic_pair_publication_probe();

    let qr = qr_compact_dyn_checked_generic(&mut dense, &input).unwrap();
    assert_pair_reconstructs_checked_literal(&qr.0, &qr.1, complex);
    let qr_probe = crate::factorize::generic_pair_publication_probe();
    let qr_blocks = qr.0.space().space().structure().block_count()
        + qr.1.space().space().structure().block_count();
    // Formerly `qr_blocks` visits and `2 * qr_blocks` events: the checked
    // route no longer cross-checks a separate key list against its structure.
    assert_eq!(qr_probe.output_blocks_visited, 0);
    assert_eq!(qr_probe.ordered_key_validation_events, qr_blocks);
    assert_eq!(
        (qr_probe.fallback_row_lookups, qr_probe.fallback_col_lookups),
        (0, 0)
    );
    svd_compact_dyn_checked_generic(&mut dense, &input).unwrap();
    let lq = lq_compact_dyn_checked_generic(&mut dense, &input).unwrap();
    assert_pair_reconstructs_checked_literal(&lq.0, &lq.1, complex);
    let full_qr = qr_full_dyn_checked_generic(&mut dense, &input).unwrap();
    assert_pair_reconstructs_checked_literal(&full_qr.0, &full_qr.1, complex);
    let full_lq = lq_full_dyn_checked_generic(&mut dense, &input).unwrap();
    assert_pair_reconstructs_checked_literal(&full_lq.0, &full_lq.1, complex);

    let probe = crate::factorize::generic_pair_publication_probe();
    assert_eq!(probe.canonical_publications, 5);
    assert_eq!(probe.fallback_publications, 0);
    assert!(probe.left_appended_elements > 0);
    assert!(probe.right_appended_elements > 0);
    assert_eq!(
        (
            probe.left_scattered_elements,
            probe.right_scattered_elements
        ),
        (0, 0)
    );
    assert!(Arc::ptr_eq(qr.0.space().provider_arc(), &provider));
    assert!(Arc::ptr_eq(qr.1.space().provider_arc(), &provider));
}

#[test]
fn checked_generic_pair_publication_appends_multiple_literal_sectors() {
    assert_checked_generic_pair_publication_appends_multiple_literal_sectors::<f64>(false);
    assert_checked_generic_pair_publication_appends_multiple_literal_sectors::<Complex64>(true);
}

#[test]
fn square_matrix_function_rejects_noncanonical_admitted_output_before_kernel() {
    let (canonical_space, canonical_data) = generic_factorization_input();
    let (padded_space, _) = padded_generic_factorization_input(&canonical_space, &canonical_data);
    let input = BoundDynamicTensorRef::try_new(&canonical_space, &canonical_data).unwrap();
    let called = Cell::new(false);
    let result = map_square_sectors_dyn_into(
        &input,
        padded_space,
        |_| -> Result<(), OperationError> {
            called.set(true);
            unreachable!("layout rejection precedes kernel initialization")
        },
        |_, _, _, _, _| unreachable!("layout rejection precedes dense work"),
    );
    assert!(matches!(
        result,
        Err(OperationError::UnsupportedTensorContractScope { .. })
    ));
    assert!(!called.get());
}

#[test]
fn generic_exp_direct_reuses_the_exact_input_provider_and_layout() {
    let provider = Arc::new(FactorGenericRule);
    let x = SectorId::new(1);
    let leg = SectorLeg::new([(x, 1)], false);
    let homspace = FusionTreeHomSpace::new(
        FusionProductSpace::new([leg.clone(), leg.clone()]),
        FusionProductSpace::new([leg.clone(), leg]),
    );
    let space =
        BoundDynamicFusionMapSpace::from_final_homspace_generic(provider, homspace).unwrap();
    let data = vec![0.0; space.space().required_len().unwrap()];
    let input = BoundDynamicTensorRef::try_new(&space, &data).unwrap();
    let mut dense = tenet_dense::DefaultDenseExecutor::new();
    let output = exp_pade13_direct_into_dyn(&mut dense, &input).unwrap();

    assert!(Arc::ptr_eq(
        output.space().provider_arc(),
        space.provider_arc()
    ));
    assert_eq!(
        output.space().space().structure(),
        space.space().structure()
    );
    assert_eq!(output.space().space().homspace(), space.space().homspace());
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
struct ValuesInputObservation {
    operation: ValuesOperation,
    pointer: usize,
    shape: Vec<usize>,
    strides: Vec<usize>,
    offset: usize,
    values: Vec<Complex64>,
}

struct ValuesInputSpy {
    inner: tenet_dense::DefaultDenseExecutor,
    observations: Vec<ValuesInputObservation>,
}

#[derive(Debug)]
struct CompactInputObservation {
    pointer: usize,
    shape: Vec<usize>,
    strides: Vec<usize>,
    offset: usize,
    values: Vec<Complex64>,
}

struct CompactInputSpy {
    inner: tenet_dense::DefaultDenseExecutor,
    operation: crate::factorize::CheckedCompactOperation,
    observations: Vec<CompactInputObservation>,
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

impl CompactInputSpy {
    fn new(operation: crate::factorize::CheckedCompactOperation) -> Self {
        Self {
            inner: tenet_dense::DefaultDenseExecutor::new(),
            operation,
            observations: Vec::new(),
        }
    }

    fn observe(&mut self, input: DenseRead<'_>) {
        let (pointer, strides, offset, values) = match input {
            DenseRead::F64(view) => (
                view.data().as_ptr() as usize,
                view.strides().to_vec(),
                view.offset(),
                view.data()
                    .iter()
                    .map(|&value| Complex64::new(value, 0.0))
                    .collect(),
            ),
            DenseRead::C64(view) => (
                view.data().as_ptr() as usize,
                view.strides().to_vec(),
                view.offset(),
                view.data().to_vec(),
            ),
            _ => panic!("checked Generic compact fixture must be f64 or c64"),
        };
        self.observations.push(CompactInputObservation {
            pointer,
            shape: input.shape().to_vec(),
            strides,
            offset,
            values,
        });
    }
}

impl DenseExecutor for CompactInputSpy {
    fn svd(&mut self, _: DenseRead<'_>) -> Result<Vec<DenseTensor>, DenseError> {
        panic!("compact SVD must use the destination API")
    }

    fn svd_into(
        &mut self,
        input: DenseRead<'_>,
        u: DenseWrite<'_>,
        s: DenseWrite<'_>,
        vt: DenseWrite<'_>,
    ) -> Result<(), DenseError> {
        assert_eq!(
            self.operation,
            crate::factorize::CheckedCompactOperation::Svd
        );
        self.observe(input);
        self.inner.svd_into(input, u, s, vt)
    }

    fn qr(&mut self, input: DenseRead<'_>) -> Result<Vec<DenseTensor>, DenseError> {
        assert!(matches!(
            self.operation,
            crate::factorize::CheckedCompactOperation::Qr
                | crate::factorize::CheckedCompactOperation::Lq
        ));
        self.observe(input);
        self.inner.qr(input)
    }

    fn qr_into(
        &mut self,
        input: DenseRead<'_>,
        q: DenseWrite<'_>,
        r: DenseWrite<'_>,
    ) -> Result<(), DenseError> {
        let _ = (input, q, r);
        panic!("compact QR/LQ must not use qr_into")
    }

    fn eigh(&mut self, _: DenseRead<'_>) -> Result<Vec<DenseTensor>, DenseError> {
        panic!("test only exercises compact QR/SVD/LQ")
    }

    fn dot_general_into(
        &mut self,
        _: DenseWrite<'_>,
        _: DenseRead<'_>,
        _: DenseRead<'_>,
        _: &DenseDotConfig,
    ) -> Result<(), DenseError> {
        panic!("test only exercises compact QR/SVD/LQ")
    }
}

impl Default for ValuesInputSpy {
    fn default() -> Self {
        Self {
            inner: tenet_dense::DefaultDenseExecutor::new(),
            observations: Vec::new(),
        }
    }
}

impl ValuesInputSpy {
    fn observe(&mut self, operation: ValuesOperation, input: DenseRead<'_>) {
        let (pointer, strides, offset, values) = match input {
            DenseRead::F64(view) => (
                view.data().as_ptr() as usize,
                view.strides().to_vec(),
                view.offset(),
                view.data()
                    .iter()
                    .map(|&value| Complex64::new(value, 0.0))
                    .collect(),
            ),
            DenseRead::C64(view) => (
                view.data().as_ptr() as usize,
                view.strides().to_vec(),
                view.offset(),
                view.data().to_vec(),
            ),
            _ => panic!("checked Generic values fixture must be f64 or c64"),
        };
        self.observations.push(ValuesInputObservation {
            operation,
            pointer,
            shape: input.shape().to_vec(),
            strides,
            offset,
            values,
        });
    }
}

impl DenseExecutor for ValuesInputSpy {
    fn svd(&mut self, _: DenseRead<'_>) -> Result<Vec<DenseTensor>, DenseError> {
        panic!("test only exercises values-only operations")
    }

    fn qr(&mut self, _: DenseRead<'_>) -> Result<Vec<DenseTensor>, DenseError> {
        panic!("test only exercises values-only operations")
    }

    fn eigh(&mut self, _: DenseRead<'_>) -> Result<Vec<DenseTensor>, DenseError> {
        panic!("test only exercises values-only operations")
    }

    fn svd_vals(&mut self, input: DenseRead<'_>) -> Result<DenseTensor, DenseError> {
        self.observe(ValuesOperation::Svd, input);
        self.inner.svd_vals(input)
    }

    fn eigh_vals(&mut self, input: DenseRead<'_>) -> Result<DenseTensor, DenseError> {
        self.observe(ValuesOperation::Eigh, input);
        self.inner.eigh_vals(input)
    }

    fn eig_vals(&mut self, input: DenseRead<'_>) -> Result<DenseTensor, DenseError> {
        self.observe(ValuesOperation::Eig, input);
        self.inner.eig_vals(input)
    }

    fn dot_general_into(
        &mut self,
        _: DenseWrite<'_>,
        _: DenseRead<'_>,
        _: DenseRead<'_>,
        _: &DenseDotConfig,
    ) -> Result<(), DenseError> {
        panic!("test only exercises values-only operations")
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

struct FailSingleLegFold {
    rule: FactorGenericRule,
    fail_at: usize,
    single_leg_folds: Cell<usize>,
}

impl FusionRule for FailSingleLegFold {
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

impl CheckedGenericFusion for FailSingleLegFold {
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
        Ok(self.rule.dual(sector))
    }
    fn try_fusion_channels(
        &self,
        left: SectorId,
        right: SectorId,
    ) -> Result<SectorVec, Self::Error> {
        Ok(self.rule.fusion_channels(left, right))
    }
    fn try_fusion_channels_in_table(
        &self,
        left: SectorId,
        right: SectorId,
    ) -> Result<SectorVec, Self::Error> {
        self.try_fusion_channels(left, right)
    }
    fn try_coupled_sector_fold(
        &self,
        effective: &[SectorId],
    ) -> Result<CoupledSectorFold, Self::Error> {
        if effective.len() == 1 {
            let call = self.single_leg_folds.get() + 1;
            self.single_leg_folds.set(call);
            if call == self.fail_at {
                return Err(LateGenericError(call));
            }
        }
        Ok(InfallibleGeneric::new(&self.rule)
            .try_coupled_sector_fold(effective)
            .unwrap())
    }
    fn try_nsymbol(
        &self,
        left: SectorId,
        right: SectorId,
        coupled: SectorId,
    ) -> Result<usize, Self::Error> {
        Ok(self.rule.nsymbol(left, right, coupled))
    }
}

impl CheckedGenericRigidSymbols for FailSingleLegFold {
    type Scalar = f64;

    fn try_sqrt_dim_scalar(&self, sector: SectorId) -> Result<Self::Scalar, Self::Error> {
        Ok(self.rule.sqrt_dim_scalar(sector))
    }
    fn try_inv_sqrt_dim_scalar(&self, sector: SectorId) -> Result<Self::Scalar, Self::Error> {
        Ok(self.rule.inv_sqrt_dim_scalar(sector))
    }
    fn try_frobenius_schur_phase_scalar(
        &self,
        sector: SectorId,
    ) -> Result<Self::Scalar, Self::Error> {
        Ok(self.rule.frobenius_schur_phase_scalar(sector))
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
        Ok(self.rule.f_symbol_generic(a, b, c, d, e, f))
    }
    fn try_r_symbol_generic(
        &self,
        a: SectorId,
        b: SectorId,
        coupled: SectorId,
    ) -> Result<GenericRMatrix<Self::Scalar>, Self::Error> {
        Ok(self.rule.r_symbol_generic(a, b, coupled))
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

fn assert_borrowed_values_inputs<D: FactorScalar>(
    observations: &[ValuesInputObservation],
    operation: ValuesOperation,
    data: &[D],
    regions: &[tenet_core::CoupledSectorRegion],
) {
    assert_eq!(observations.len(), regions.len());
    for (observation, region) in observations.iter().zip(regions) {
        assert_eq!(observation.operation, operation);
        assert_eq!(observation.shape, [region.rows(), region.cols()]);
        assert_eq!(observation.strides, [1, region.rows()]);
        assert_eq!(observation.offset, 0);
        assert_eq!(
            observation.pointer,
            data.as_ptr() as usize + region.range().start * std::mem::size_of::<D>()
        );
        let expected = data[region.range()]
            .iter()
            .map(|&value| value.widen_complex())
            .collect::<Vec<_>>();
        assert_eq!(observation.values, expected);
    }
}

fn assert_compact_input_observations<D: FactorScalar>(
    operation: crate::factorize::CheckedCompactOperation,
    dense: &[CompactInputObservation],
    data: &[D],
    regions: &[tenet_core::CoupledSectorRegion],
) {
    let lowering = crate::factorize::checked_compact_input_observations();
    assert_eq!(lowering.len(), regions.len());
    assert_eq!(dense.len(), regions.len());
    for ((lowering, dense), region) in lowering.iter().zip(dense).zip(regions) {
        let source_pointer =
            data.as_ptr() as usize + region.range().start * std::mem::size_of::<D>();
        assert_eq!(lowering.operation, operation);
        assert_eq!(lowering.input_pointer, data.as_ptr() as usize);
        assert_eq!(lowering.matrix_pointer, source_pointer);
        assert_eq!(lowering.elements, region.range().len());
        assert_eq!(dense.offset, 0);
        match operation {
            crate::factorize::CheckedCompactOperation::Qr
            | crate::factorize::CheckedCompactOperation::Svd => {
                assert_eq!(lowering.adjoint_pointer, None);
                assert_eq!(dense.pointer, source_pointer);
                assert_eq!(dense.shape, [region.rows(), region.cols()]);
                assert_eq!(dense.strides, [1, region.rows()]);
                assert_eq!(
                    dense.values,
                    data[region.range()]
                        .iter()
                        .map(|&value| value.widen_complex())
                        .collect::<Vec<_>>()
                );
            }
            crate::factorize::CheckedCompactOperation::Lq => {
                assert_eq!(lowering.adjoint_pointer, Some(dense.pointer));
                assert_ne!(dense.pointer, source_pointer);
                assert_eq!(dense.shape, [region.cols(), region.rows()]);
                assert_eq!(dense.strides, [1, region.cols()]);
                let source = &data[region.range()];
                let expected = (0..region.rows())
                    .flat_map(|column| {
                        (0..region.cols()).map(move |row| {
                            source[column + region.rows() * row].widen_complex().conj()
                        })
                    })
                    .collect::<Vec<_>>();
                assert_eq!(dense.values, expected);
            }
        }
    }
}

fn assert_checked_compact_input_borrowing<D: FactorScalar>(
    space: BoundDynamicFusionMapSpace<LateGenericSpy>,
    data: Vec<D>,
) {
    let regions = space
        .space()
        .structure()
        .coupled_sector_regions(space.space().nout())
        .unwrap()
        .unwrap();
    let input = BoundDynamicTensorRef::try_new(&space, &data).unwrap();
    let before = data.clone();

    crate::factorize::reset_compact_qr_copy_probe();
    crate::factorize::reset_checked_compact_input_observations();
    let mut qr = CompactInputSpy::new(crate::factorize::CheckedCompactOperation::Qr);
    let factors = qr_compact_dyn_checked_generic(&mut qr, &input).unwrap();
    assert_compact_factors_reconstruct_input(&input, &factors.0, None, &factors.1);
    assert_compact_input_observations(
        crate::factorize::CheckedCompactOperation::Qr,
        &qr.observations,
        &data,
        &regions,
    );
    assert_eq!(
        crate::factorize::compact_qr_copy_probe().input_pack_calls,
        0
    );

    crate::factorize::reset_compact_svd_copy_probe();
    crate::factorize::reset_checked_compact_input_observations();
    let mut svd = CompactInputSpy::new(crate::factorize::CheckedCompactOperation::Svd);
    let factors = svd_compact_dyn_checked_generic(&mut svd, &input).unwrap();
    assert_compact_factors_reconstruct_input(&input, &factors.0, Some(&factors.1), &factors.2);
    assert_compact_input_observations(
        crate::factorize::CheckedCompactOperation::Svd,
        &svd.observations,
        &data,
        &regions,
    );
    assert_eq!(
        crate::factorize::compact_svd_copy_probe().input_pack_calls,
        0
    );

    crate::factorize::reset_compact_lq_copy_probe();
    crate::factorize::reset_checked_compact_input_observations();
    let mut lq = CompactInputSpy::new(crate::factorize::CheckedCompactOperation::Lq);
    let factors = lq_compact_dyn_checked_generic(&mut lq, &input).unwrap();
    assert_compact_factors_reconstruct_input(&input, &factors.0, None, &factors.1);
    assert_compact_input_observations(
        crate::factorize::CheckedCompactOperation::Lq,
        &lq.observations,
        &data,
        &regions,
    );
    let probe = crate::factorize::compact_lq_copy_probe();
    assert_eq!(probe.input_pack_calls, 0);
    assert_eq!(probe.adjoint_scratch_fill_calls, regions.len());
    assert_eq!(
        probe.adjoint_scratch_fill_bytes,
        data.len() * std::mem::size_of::<D>()
    );
    assert!(data == before);
}

#[test]
fn checked_generic_compact_factors_borrow_real_and_complex_canonical_inputs() {
    let (_, real_space, real_data) = checked_svd_truncation_input::<f64>(false);
    assert_checked_compact_input_borrowing(real_space, real_data);
    let (_, complex_space, complex_data) = checked_svd_truncation_input::<Complex64>(true);
    assert_checked_compact_input_borrowing(complex_space, complex_data);
    let (_, wide_space, wide_data) = checked_svd_wide_input::<Complex64>();
    assert_checked_compact_input_borrowing(wide_space, wide_data);
}

#[test]
fn checked_generic_compact_geometry_preserves_complete_multi_tree_identity() {
    let (source, _, data) = generic_values_endomorphism_input();
    let (_, space) = bind_checked_only(&source);
    let input = BoundDynamicTensorRef::try_new(&space, &data).unwrap();
    let regions = space
        .space()
        .structure()
        .coupled_sector_regions(space.space().nout())
        .unwrap()
        .unwrap();
    let multiplicity_region = regions
        .iter()
        .find(|region| region.coupled() == SectorId::new(1))
        .unwrap();
    assert_eq!(multiplicity_region.row_trees().len(), 2);
    let first = multiplicity_region.row_trees()[0].tree();
    let second = multiplicity_region.row_trees()[1].tree();
    assert_eq!(first.uncoupled(), second.uncoupled());
    assert_eq!(first.is_dual(), second.is_dual());
    assert_eq!(first.innerlines(), second.innerlines());
    assert_ne!(first.vertices(), second.vertices());

    crate::factorize::reset_checked_compact_input_observations();
    let mut qr = CompactInputSpy::new(crate::factorize::CheckedCompactOperation::Qr);
    let qr_factors = qr_compact_dyn_checked_generic(&mut qr, &input).unwrap();
    assert_compact_input_observations(
        crate::factorize::CheckedCompactOperation::Qr,
        &qr.observations,
        &data,
        &regions,
    );
    assert_compact_factors_reconstruct_input(&input, &qr_factors.0, None, &qr_factors.1);

    crate::factorize::reset_checked_compact_input_observations();
    let mut svd = CompactInputSpy::new(crate::factorize::CheckedCompactOperation::Svd);
    let svd_factors = svd_compact_dyn_checked_generic(&mut svd, &input).unwrap();
    assert_compact_input_observations(
        crate::factorize::CheckedCompactOperation::Svd,
        &svd.observations,
        &data,
        &regions,
    );
    assert_compact_factors_reconstruct_input(
        &input,
        &svd_factors.0,
        Some(&svd_factors.1),
        &svd_factors.2,
    );

    crate::factorize::reset_checked_compact_input_observations();
    let mut lq = CompactInputSpy::new(crate::factorize::CheckedCompactOperation::Lq);
    let lq_factors = lq_compact_dyn_checked_generic(&mut lq, &input).unwrap();
    assert_compact_input_observations(
        crate::factorize::CheckedCompactOperation::Lq,
        &lq.observations,
        &data,
        &regions,
    );
    assert_compact_factors_reconstruct_input(&input, &lq_factors.0, None, &lq_factors.1);
}

#[test]
#[expect(
    clippy::arc_with_non_send_sync,
    reason = "the checked Generic API requires Arc identity while Cell is a single-threaded call spy"
)]
fn checked_generic_compact_factors_keep_padded_reordered_input_pack() {
    let (canonical_space, canonical_data) = generic_factorization_input();
    let (expert_space, expert_data) =
        expert_generic_factorization_input(&canonical_space, &canonical_data, true);
    let provider = Arc::new(LateGenericSpy {
        rule: FactorGenericRule,
        fail_at: usize::MAX,
        calls: Cell::new(0),
    });
    let expert_space = BoundDynamicFusionMapSpace::bind_generic(
        expert_space.space().clone(),
        Arc::clone(&provider),
    )
    .unwrap();
    let expert = BoundDynamicTensorRef::try_new(&expert_space, &expert_data).unwrap();
    let canonical_before = canonical_data.clone();
    let expert_before = expert_data.clone();
    let mut dense = tenet_dense::DefaultDenseExecutor::new();

    crate::factorize::reset_compact_qr_copy_probe();
    crate::factorize::reset_checked_compact_input_observations();
    let actual_qr = qr_compact_dyn_checked_generic(&mut dense, &expert).unwrap();
    assert_compact_factors_reconstruct_input(&expert, &actual_qr.0, None, &actual_qr.1);
    assert!(crate::factorize::compact_qr_copy_probe().input_pack_calls > 0);

    crate::factorize::reset_compact_svd_copy_probe();
    crate::factorize::reset_checked_compact_input_observations();
    let actual_svd = svd_compact_dyn_checked_generic(&mut dense, &expert).unwrap();
    assert_compact_factors_reconstruct_input(
        &expert,
        &actual_svd.0,
        Some(&actual_svd.1),
        &actual_svd.2,
    );
    assert!(crate::factorize::compact_svd_copy_probe().input_pack_calls > 0);

    crate::factorize::reset_compact_lq_copy_probe();
    crate::factorize::reset_checked_compact_input_observations();
    let actual_lq = lq_compact_dyn_checked_generic(&mut dense, &expert).unwrap();
    assert_compact_factors_reconstruct_input(&expert, &actual_lq.0, None, &actual_lq.1);
    let lq_probe = crate::factorize::compact_lq_copy_probe();
    assert!(lq_probe.input_pack_calls > 0);
    assert!(lq_probe.adjoint_scratch_fill_calls > 0);

    let start = expert_data.as_ptr() as usize;
    let end = start + expert_data.len() * std::mem::size_of::<f64>();
    for observation in crate::factorize::checked_compact_input_observations() {
        assert_eq!(observation.input_pointer, start);
        assert!(observation.matrix_pointer < start || observation.matrix_pointer >= end);
    }
    assert!(canonical_data == canonical_before);
    assert!(expert_data == expert_before);
    assert!(Arc::ptr_eq(actual_qr.0.space().provider_arc(), &provider));
    assert!(Arc::ptr_eq(actual_svd.1.space().provider_arc(), &provider));
    assert!(Arc::ptr_eq(actual_lq.1.space().provider_arc(), &provider));
}

#[test]
#[expect(
    clippy::arc_with_non_send_sync,
    reason = "the checked Generic API requires Arc identity while Cell is a single-threaded call spy"
)]
fn checked_generic_compact_interleaved_fallback_keeps_literal_matrix_order() {
    let (canonical, _, canonical_data) = generic_values_endomorphism_input();
    let (expert_space, expert_data) =
        interleaved_generic_endomorphism_input(&canonical, &canonical_data);
    let provider = Arc::new(LateGenericSpy {
        rule: FactorGenericRule,
        fail_at: usize::MAX,
        calls: Cell::new(0),
    });
    let expert_space = BoundDynamicFusionMapSpace::bind_generic(
        expert_space.space().clone(),
        Arc::clone(&provider),
    )
    .unwrap();
    let expert = BoundDynamicTensorRef::try_new(&expert_space, &expert_data).unwrap();
    let before = expert_data.clone();
    let direct = [
        vec![
            Complex64::new(3.0, -1.0),
            Complex64::new(2.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(1.0, 1.0),
        ],
        vec![Complex64::new(2.0, -1.0)],
    ];
    let adjoint = [
        vec![
            Complex64::new(3.0, 1.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(2.0, 0.0),
            Complex64::new(1.0, -1.0),
        ],
        vec![Complex64::new(2.0, 1.0)],
    ];

    crate::factorize::reset_compact_qr_copy_probe();
    let mut qr = CompactInputSpy::new(crate::factorize::CheckedCompactOperation::Qr);
    let factors = qr_compact_dyn_checked_generic(&mut qr, &expert).unwrap();
    assert_eq!(
        qr.observations
            .iter()
            .map(|observation| observation.values.clone())
            .collect::<Vec<_>>(),
        direct
    );
    assert_compact_factors_reconstruct_input(&expert, &factors.0, None, &factors.1);
    assert!(crate::factorize::compact_qr_copy_probe().input_pack_calls > 0);

    crate::factorize::reset_compact_svd_copy_probe();
    let mut svd = CompactInputSpy::new(crate::factorize::CheckedCompactOperation::Svd);
    let factors = svd_compact_dyn_checked_generic(&mut svd, &expert).unwrap();
    assert_eq!(
        svd.observations
            .iter()
            .map(|observation| observation.values.clone())
            .collect::<Vec<_>>(),
        direct
    );
    assert_compact_factors_reconstruct_input(&expert, &factors.0, Some(&factors.1), &factors.2);
    assert!(crate::factorize::compact_svd_copy_probe().input_pack_calls > 0);

    crate::factorize::reset_compact_lq_copy_probe();
    let mut lq = CompactInputSpy::new(crate::factorize::CheckedCompactOperation::Lq);
    let factors = lq_compact_dyn_checked_generic(&mut lq, &expert).unwrap();
    assert_eq!(
        lq.observations
            .iter()
            .map(|observation| observation.values.clone())
            .collect::<Vec<_>>(),
        adjoint
    );
    assert_compact_factors_reconstruct_input(&expert, &factors.0, None, &factors.1);
    let probe = crate::factorize::compact_lq_copy_probe();
    assert!(probe.input_pack_calls > 0);
    assert!(probe.adjoint_scratch_fill_calls > 0);
    assert_eq!(expert_data, before);
}

fn assert_real_spectra_by_sector_close(actual: &[SectorSpectrum], expected: &[SectorSpectrum]) {
    assert_eq!(actual.len(), expected.len());
    for expected in expected {
        let actual = actual
            .iter()
            .find(|actual| actual.sector == expected.sector)
            .unwrap();
        assert_eq!(actual.values.len(), expected.values.len());
        for (&actual, &expected) in actual.values.iter().zip(&expected.values) {
            assert!((actual - expected).abs() < 1.0e-10);
        }
    }
}

fn assert_complex_spectra_by_sector_close(
    actual: &[SectorSpectrum<Complex64>],
    expected: &[SectorSpectrum<Complex64>],
) {
    assert_eq!(actual.len(), expected.len());
    for expected in expected {
        let actual = actual
            .iter()
            .find(|actual| actual.sector == expected.sector)
            .unwrap();
        assert_eq!(actual.values.len(), expected.values.len());
        for (&actual, &expected) in actual.values.iter().zip(&expected.values) {
            assert!((actual - expected).norm() < 1.0e-10);
        }
    }
}

#[test]
fn checked_only_generic_values_borrow_canonical_input_regions() {
    let (_, rectangular_space, rectangular_data) = checked_svd_truncation_input::<f64>(false);
    let (rectangular_provider, rectangular_space) = bind_checked_only(&rectangular_space);
    let rectangular_calls = rectangular_provider.calls.get();
    let rectangular_before = rectangular_data.clone();
    let rectangular_regions = rectangular_space
        .space()
        .structure()
        .coupled_sector_regions(1)
        .unwrap()
        .unwrap();
    let mut spy = ValuesInputSpy::default();
    crate::factorize::reset_values_matricization_fallbacks();
    let spectra = svd_vals_dyn_checked_generic(
        &mut spy,
        &BoundDynamicTensorRef::try_new(&rectangular_space, &rectangular_data).unwrap(),
    )
    .unwrap();
    assert_borrowed_values_inputs(
        &spy.observations,
        ValuesOperation::Svd,
        &rectangular_data,
        &rectangular_regions,
    );
    assert_real_spectra_by_sector_close(
        &spectra,
        &[
            SectorSpectrum {
                sector: SectorId::new(0),
                values: vec![4.0, 1.0],
            },
            SectorSpectrum {
                sector: SectorId::new(1),
                values: vec![3.0, 2.0],
            },
        ],
    );
    assert_eq!(rectangular_provider.calls.get(), rectangular_calls);
    assert_eq!(rectangular_data, rectangular_before);

    let (endomorphism_space, hermitian, general) = generic_values_endomorphism_input();
    let (endomorphism_provider, endomorphism_space) = bind_checked_only(&endomorphism_space);
    let provider_calls = endomorphism_provider.calls.get();
    let regions = endomorphism_space
        .space()
        .structure()
        .coupled_sector_regions(2)
        .unwrap()
        .unwrap();
    let multiplicity_region = regions
        .iter()
        .find(|region| region.coupled() == SectorId::new(1))
        .unwrap();
    assert_eq!(multiplicity_region.row_trees().len(), 2);
    assert_eq!(multiplicity_region.col_trees().len(), 2);
    let hermitian_before = hermitian.clone();
    let general_before = general.clone();
    spy.observations.clear();
    let eigh = eigh_vals_dyn_checked_generic(
        &mut spy,
        &BoundDynamicTensorRef::try_new(&endomorphism_space, &hermitian).unwrap(),
    )
    .unwrap();
    assert_borrowed_values_inputs(
        &spy.observations,
        ValuesOperation::Eigh,
        &hermitian,
        &regions,
    );
    assert_real_spectra_by_sector_close(
        &eigh,
        &[
            SectorSpectrum {
                sector: SectorId::new(0),
                values: vec![-4.0],
            },
            SectorSpectrum {
                sector: SectorId::new(1),
                values: vec![3.0, 1.0],
            },
        ],
    );
    spy.observations.clear();
    let eig = eig_vals_dyn_checked_generic(
        &mut spy,
        &BoundDynamicTensorRef::try_new(&endomorphism_space, &general).unwrap(),
    )
    .unwrap();
    assert_borrowed_values_inputs(&spy.observations, ValuesOperation::Eig, &general, &regions);
    assert_complex_spectra_by_sector_close(
        &eig,
        &[
            SectorSpectrum {
                sector: SectorId::new(0),
                values: vec![Complex64::new(2.0, -1.0)],
            },
            SectorSpectrum {
                sector: SectorId::new(1),
                values: vec![Complex64::new(3.0, -1.0), Complex64::new(1.0, 1.0)],
            },
        ],
    );
    assert_eq!(crate::factorize::values_matricization_fallbacks(), 0);
    assert_eq!(endomorphism_provider.calls.get(), provider_calls);
    assert_eq!(hermitian, hermitian_before);
    assert_eq!(general, general_before);
}

#[test]
#[expect(
    clippy::arc_with_non_send_sync,
    reason = "the checked Generic API requires Arc identity while Cell is a single-threaded call spy"
)]
fn checked_generic_values_keep_padded_reordered_fallback() {
    let (canonical_space, hermitian, general) = generic_values_endomorphism_input();
    let (padded_space, padded_hermitian) =
        padded_reordered_generic_endomorphism_input(&canonical_space, &hermitian);
    let (_, padded_general) =
        padded_reordered_generic_endomorphism_input(&canonical_space, &general);
    let provider = Arc::new(LateGenericSpy {
        rule: FactorGenericRule,
        fail_at: usize::MAX,
        calls: Cell::new(0),
    });
    let canonical_space = BoundDynamicFusionMapSpace::bind_generic(
        canonical_space.space().clone(),
        Arc::clone(&provider),
    )
    .unwrap();
    let padded_space = BoundDynamicFusionMapSpace::bind_generic(
        padded_space.space().clone(),
        Arc::clone(&provider),
    )
    .unwrap();
    assert!(padded_space
        .space()
        .structure()
        .coupled_sector_regions(2)
        .unwrap()
        .is_none());
    let provider_calls = provider.calls.get();
    let canonical_general = BoundDynamicTensorRef::try_new(&canonical_space, &general).unwrap();
    let canonical_hermitian = BoundDynamicTensorRef::try_new(&canonical_space, &hermitian).unwrap();
    let padded_general_input =
        BoundDynamicTensorRef::try_new(&padded_space, &padded_general).unwrap();
    let padded_hermitian_input =
        BoundDynamicTensorRef::try_new(&padded_space, &padded_hermitian).unwrap();
    let mut dense = tenet_dense::DefaultDenseExecutor::new();
    let expected_svd = svd_vals_dyn_checked_generic(&mut dense, &canonical_general).unwrap();
    let expected_eigh = eigh_vals_dyn_checked_generic(&mut dense, &canonical_hermitian).unwrap();
    let expected_eig = eig_vals_dyn_checked_generic(&mut dense, &canonical_general).unwrap();

    let general_before = padded_general.clone();
    let hermitian_before = padded_hermitian.clone();
    let mut spy = ValuesInputSpy::default();
    crate::factorize::reset_values_matricization_fallbacks();
    let actual_svd = svd_vals_dyn_checked_generic(&mut spy, &padded_general_input).unwrap();
    let actual_eigh = eigh_vals_dyn_checked_generic(&mut spy, &padded_hermitian_input).unwrap();
    let actual_eig = eig_vals_dyn_checked_generic(&mut spy, &padded_general_input).unwrap();
    assert_eq!(crate::factorize::values_matricization_fallbacks(), 3);
    assert_real_spectra_by_sector_close(&actual_svd, &expected_svd);
    assert_real_spectra_by_sector_close(&actual_eigh, &expected_eigh);
    assert_complex_spectra_by_sector_close(&actual_eig, &expected_eig);
    assert_eq!(provider.calls.get(), provider_calls);
    assert_eq!(padded_general, general_before);
    assert_eq!(padded_hermitian, hermitian_before);

    let general_start = padded_general.as_ptr() as usize;
    let general_end = general_start + std::mem::size_of_val(padded_general.as_slice());
    let hermitian_start = padded_hermitian.as_ptr() as usize;
    let hermitian_end = hermitian_start + std::mem::size_of_val(padded_hermitian.as_slice());
    for observation in &spy.observations {
        let (start, end) = if observation.operation == ValuesOperation::Eigh {
            (hermitian_start, hermitian_end)
        } else {
            (general_start, general_end)
        };
        assert!(observation.pointer < start || observation.pointer >= end);
    }
}

#[test]
fn checked_only_generic_eigh_validates_every_region_before_dense_work() {
    let (space, mut hermitian, _) = generic_values_endomorphism_input();
    let (provider, space) = bind_checked_only(&space);
    let regions = space
        .space()
        .structure()
        .coupled_sector_regions(2)
        .unwrap()
        .unwrap();
    let later = regions.last().unwrap();
    hermitian[later.range().start + 1] += Complex64::new(1.0, 0.0);
    let before = hermitian.clone();
    let provider_calls = provider.calls.get();
    let mut dense = EighCallSpy::default();
    crate::factorize::reset_values_matricization_fallbacks();

    let error = eigh_vals_dyn_checked_generic(
        &mut dense,
        &BoundDynamicTensorRef::try_new(&space, &hermitian).unwrap(),
    )
    .unwrap_err();

    assert!(matches!(
        error,
        CheckedGenericFactorPlanError::Operation(OperationError::InvalidArgument {
            message: "eigh requires Hermitian coupled-sector blocks",
        })
    ));
    assert_eq!(dense.calls, 0);
    assert_eq!(crate::factorize::values_matricization_fallbacks(), 0);
    assert_eq!(provider.calls.get(), provider_calls);
    assert_eq!(hermitian, before);
}

#[test]
#[expect(
    clippy::arc_with_non_send_sync,
    reason = "the checked Generic API requires Arc identity while Cell is a single-threaded call spy"
)]
fn checked_only_generic_values_preserve_empty_scalar_and_shape_boundaries() {
    let x = SectorId::new(1);
    let empty_leg = SectorLeg::new([(x, 0)], false);
    let empty_homspace = FusionTreeHomSpace::new(
        FusionProductSpace::new([empty_leg.clone()]),
        FusionProductSpace::new([empty_leg]),
    );
    let empty_provider = Arc::new(CheckedOnlyFactorRule {
        calls: Cell::new(0),
    });
    let empty_space = BoundDynamicFusionMapSpace::from_final_homspace_generic_checked(
        Arc::clone(&empty_provider),
        empty_homspace,
    )
    .unwrap();
    let empty_calls = empty_provider.calls.get();
    let empty_data: [f64; 0] = [];
    let empty = BoundDynamicTensorRef::try_new(&empty_space, &empty_data).unwrap();
    let mut reject = RejectExecutorCalls;
    assert!(svd_vals_dyn_checked_generic(&mut reject, &empty)
        .unwrap()
        .is_empty());
    assert!(eigh_vals_dyn_checked_generic(&mut reject, &empty)
        .unwrap()
        .is_empty());
    assert!(eig_vals_dyn_checked_generic(&mut reject, &empty)
        .unwrap()
        .is_empty());
    assert_eq!(empty_provider.calls.get(), empty_calls);

    let scalar_provider = Arc::new(CheckedOnlyFactorRule {
        calls: Cell::new(0),
    });
    let scalar_space = BoundDynamicFusionMapSpace::from_final_homspace_generic_checked(
        Arc::clone(&scalar_provider),
        FusionTreeHomSpace::new(FusionProductSpace::new([]), FusionProductSpace::new([])),
    )
    .unwrap();
    let scalar_calls = scalar_provider.calls.get();
    let scalar_data = [-3.0];
    let scalar = BoundDynamicTensorRef::try_new(&scalar_space, &scalar_data).unwrap();
    let mut dense = tenet_dense::DefaultDenseExecutor::new();
    assert_eq!(
        svd_vals_dyn_checked_generic(&mut dense, &scalar).unwrap()[0].values,
        [3.0]
    );
    assert_eq!(
        eigh_vals_dyn_checked_generic(&mut dense, &scalar).unwrap()[0].values,
        [-3.0]
    );
    assert_eq!(
        eig_vals_dyn_checked_generic(&mut dense, &scalar).unwrap()[0].values,
        [Complex64::new(-3.0, 0.0)]
    );
    assert_eq!(scalar_provider.calls.get(), scalar_calls);

    let (_, rectangular, data) = checked_svd_truncation_input::<f64>(false);
    let (rectangular_provider, rectangular) = bind_checked_only(&rectangular);
    let rectangular_calls = rectangular_provider.calls.get();
    let rectangular = BoundDynamicTensorRef::try_new(&rectangular, &data).unwrap();
    assert!(matches!(
        eigh_vals_dyn_checked_generic(&mut reject, &rectangular),
        Err(CheckedGenericFactorPlanError::Operation(
            OperationError::UnsupportedTensorContractScope { .. }
        ))
    ));
    assert!(matches!(
        eig_vals_dyn_checked_generic(&mut reject, &rectangular),
        Err(CheckedGenericFactorPlanError::Operation(
            OperationError::UnsupportedTensorContractScope { .. }
        ))
    ));
    assert_eq!(rectangular_provider.calls.get(), rectangular_calls);
}

fn assert_checked_svd_truncation<D>(complex: bool, truncation: &Truncation, kept: usize)
where
    D: FactorScalar,
{
    let (provider, space, data) = checked_svd_truncation_input::<D>(complex);
    let input = BoundDynamicTensorRef::try_new(&space, &data).unwrap();
    let mut dense = CountingDense::default();
    let (u, vh, spectra, error) =
        svd_trunc_factors_dyn_checked_generic(&mut dense, &input, truncation).unwrap();

    assert_eq!(dense.svd_into_calls, 2);
    assert_eq!(dense.svd_vals_calls, 0);
    assert!(Arc::ptr_eq(u.space().provider_arc(), &provider));
    assert!(Arc::ptr_eq(vh.space().provider_arc(), &provider));

    let expected_spectra = [
        (SectorId::new(0), [4.0, 1.0]),
        (SectorId::new(1), [3.0, 2.0]),
    ];
    let mut residual_squared = 0.0;
    for (sector, expected) in expected_spectra {
        let spectrum = spectra.iter().find(|entry| entry.sector == sector).unwrap();
        assert_eq!(spectrum.values.len(), kept);
        for (&actual, &expected) in spectrum.values.iter().zip(&expected[..kept]) {
            assert!((actual - expected).abs() < 1.0e-10);
        }

        let u_block = (0..u.space().space().structure().block_count())
            .map(|index| u.space().space().structure().block(index).unwrap())
            .find(|block| {
                matches!(
                    block.key(),
                    BlockKey::FusionTree(key) if key.codomain_tree().coupled() == sector
                )
            })
            .unwrap();
        let vh_block = (0..vh.space().space().structure().block_count())
            .map(|index| vh.space().space().structure().block(index).unwrap())
            .find(|block| {
                matches!(
                    block.key(),
                    BlockKey::FusionTree(key) if key.domain_tree().coupled() == sector
                )
            })
            .unwrap();
        let (rows, cols, matrix) = checked_svd_matrix(sector, complex);
        assert_eq!(u_block.shape(), [rows, kept]);
        assert_eq!(vh_block.shape(), [kept, cols]);

        let u_value = |row: usize, col: usize| {
            u.data()[u_block.offset() + row * u_block.strides()[0] + col * u_block.strides()[1]]
                .widen_complex()
        };
        let vh_value = |row: usize, col: usize| {
            vh.data()[vh_block.offset() + row * vh_block.strides()[0] + col * vh_block.strides()[1]]
                .widen_complex()
        };
        for left in 0..kept {
            for right in 0..kept {
                let u_inner = (0..rows)
                    .map(|row| u_value(row, left).conj() * u_value(row, right))
                    .sum::<Complex64>();
                let vh_inner = (0..cols)
                    .map(|col| vh_value(left, col) * vh_value(right, col).conj())
                    .sum::<Complex64>();
                let expected = if left == right { 1.0 } else { 0.0 };
                assert!((u_inner - expected).norm() < 1.0e-10);
                assert!((vh_inner - expected).norm() < 1.0e-10);
            }
        }
        for col in 0..cols {
            for row in 0..rows {
                let reconstructed = (0..kept)
                    .map(|bond| u_value(row, bond) * spectrum.values[bond] * vh_value(bond, col))
                    .sum::<Complex64>();
                residual_squared += if sector == SectorId::new(1) {
                    (1.0 + 2.0_f64.sqrt()) * (reconstructed - matrix[row + rows * col]).norm_sqr()
                } else {
                    (reconstructed - matrix[row + rows * col]).norm_sqr()
                };
            }
        }
    }
    let expected_error = if kept == 2 {
        0.0
    } else {
        (1.0 + 4.0 * (1.0 + 2.0_f64.sqrt())).sqrt()
    };
    assert!((error - expected_error).abs() < 1.0e-10);
    assert!((residual_squared.sqrt() - error).abs() < 1.0e-10);
}

#[test]
fn checked_generic_svd_trunc_uses_each_real_compact_decomposition_once() {
    assert_checked_svd_truncation::<f64>(false, &Truncation::Full, 2);
    assert_checked_svd_truncation::<f64>(false, &Truncation::absolute_cutoff(2.5).unwrap(), 1);
}

#[test]
fn checked_generic_svd_trunc_uses_each_complex_compact_decomposition_once() {
    assert_checked_svd_truncation::<Complex64>(true, &Truncation::Full, 2);
    assert_checked_svd_truncation::<Complex64>(true, &Truncation::absolute_cutoff(2.5).unwrap(), 1);
}

#[test]
#[expect(
    clippy::arc_with_non_send_sync,
    reason = "the checked Generic API requires Arc identity while Cell is a single-threaded call spy"
)]
fn checked_generic_svd_trunc_empty_input_skips_dense_execution() {
    let vacuum = SectorId::new(0);
    let x = SectorId::new(1);
    let homspace = FusionTreeHomSpace::new(
        FusionProductSpace::new([SectorLeg::new([(vacuum, 1)], false)]),
        FusionProductSpace::new([SectorLeg::new([(x, 1)], false)]),
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
    let data = Vec::<f64>::new();
    let input = BoundDynamicTensorRef::try_new(&checked, &data).unwrap();
    let mut dense = CountingDense::default();
    let (u, vh, spectra, error) =
        svd_trunc_factors_dyn_checked_generic(&mut dense, &input, &Truncation::Full).unwrap();

    assert_eq!(dense.svd_into_calls, 0);
    assert_eq!(dense.svd_vals_calls, 0);
    assert!(spectra.is_empty());
    assert_eq!(error, 0.0);
    assert!(u.data().is_empty());
    assert!(vh.data().is_empty());
    assert!(Arc::ptr_eq(u.space().provider_arc(), &provider));
    assert!(Arc::ptr_eq(vh.space().provider_arc(), &provider));
}

#[test]
fn checked_generic_svd_trunc_dense_failure_precedes_output_provider_admission() {
    let (provider, space, data) = checked_svd_truncation_input::<f64>(false);
    let input = BoundDynamicTensorRef::try_new(&space, &data).unwrap();
    let before = input.data().to_vec();
    let mut dense = FailAfterObservingSvdInput::default();
    let result = svd_trunc_factors_dyn_checked_generic(&mut dense, &input, &Truncation::Full);

    assert!(matches!(
        result,
        Err(CheckedGenericFactorPlanError::Operation(
            OperationError::Dense(DenseError::Backend { op: "svd_into", .. })
        ))
    ));
    assert_eq!(provider.calls.get(), 0);
    assert_eq!(dense.observed.len(), 1);
    assert_eq!(input.data(), before);
}

#[test]
#[expect(
    clippy::arc_with_non_send_sync,
    reason = "the checked Generic API requires Arc identity while Cell is a single-threaded call spy"
)]
fn checked_generic_svd_trunc_preserves_full_s_fold_failure() {
    let (source, data) = generic_factorization_input();
    // U and Vh construction each enumerate their layout once and fold both
    // one-leg bond sectors (U 1-2, Vh 3-4), making the first fold for the
    // full diagonal S the fifth one. Formerly each side enumerated twice
    // (keys, then space: U 1-4, Vh 5-8) and S's first fold was the ninth.
    const FIRST_FULL_S_FOLD: usize = 5;
    let failing_provider = Arc::new(FailSingleLegFold {
        rule: FactorGenericRule,
        fail_at: FIRST_FULL_S_FOLD,
        single_leg_folds: Cell::new(0),
    });
    let failing_space = BoundDynamicFusionMapSpace::bind_generic(
        source.space().clone(),
        Arc::clone(&failing_provider),
    )
    .unwrap();
    let input = BoundDynamicTensorRef::try_new(&failing_space, &data).unwrap();
    let before = input.data().to_vec();
    let mut dense = CountingDense::default();
    let cutoff = Truncation::absolute_cutoff(1.0e100).unwrap();
    let result = svd_trunc_factors_dyn_checked_generic(&mut dense, &input, &cutoff);

    assert!(matches!(
        result,
        Err(CheckedGenericFactorPlanError::Provider(LateGenericError(call)))
            if call == FIRST_FULL_S_FOLD
    ));
    assert_eq!(failing_provider.single_leg_folds.get(), FIRST_FULL_S_FOLD);
    assert_eq!(dense.svd_into_calls, 2);
    assert_eq!(dense.svd_vals_calls, 0);
    assert_eq!(input.data(), before);
}

#[test]
#[expect(
    clippy::arc_with_non_send_sync,
    reason = "the checked Generic API requires Arc identity while Cell is a single-threaded call spy"
)]
fn checked_generic_polar_covers_scalar_maps() {
    // What: a rank-zero tensor map is a one-by-one vacuum-sector matrix for
    // both left and right polar decomposition.
    let provider = Arc::new(LateGenericSpy {
        rule: FactorGenericRule,
        fail_at: usize::MAX,
        calls: Cell::new(0),
    });
    let empty = FusionProductSpace::new(std::iter::empty::<SectorLeg>());
    let homspace = FusionTreeHomSpace::new(empty.clone(), empty);
    let space = BoundDynamicFusionMapSpace::from_final_homspace_generic_checked(
        Arc::clone(&provider),
        homspace,
    )
    .unwrap();
    let data: [f64; 1] = [-3.0];
    let input = BoundDynamicTensorRef::try_new(&space, &data).unwrap();
    let mut dense = tenet_dense::DefaultDenseExecutor::new();

    let (left_w, left_p) = left_polar_dyn_checked_generic(&mut dense, &input).unwrap();
    let (right_p, right_w) = right_polar_dyn_checked_generic(&mut dense, &input).unwrap();
    assert!(Arc::ptr_eq(left_w.space().provider_arc(), &provider));
    assert!(Arc::ptr_eq(left_p.space().provider_arc(), &provider));
    assert!((left_w.data()[0].abs() - 1.0).abs() < 1.0e-12);
    assert!((right_w.data()[0].abs() - 1.0).abs() < 1.0e-12);
    assert!((left_p.data()[0] - 3.0).abs() < 1.0e-12);
    assert!((right_p.data()[0] - 3.0).abs() < 1.0e-12);
    assert!((left_w.data()[0] * left_p.data()[0] - data[0]).abs() < 1.0e-12);
    assert!((right_p.data()[0] * right_w.data()[0] - data[0]).abs() < 1.0e-12);
}

#[test]
fn checked_generic_factor_plan_late_failure_precedes_commit() {
    let (space, _data) = generic_factorization_input();
    let complete = LateGenericSpy {
        rule: FactorGenericRule,
        fail_at: usize::MAX,
        calls: Cell::new(0),
    };
    let prepared =
        crate::factorize::prepare_compact_factor_plan_generic_checked_for_test(&space, &complete)
            .unwrap()
            .expect("canonical checked plan");
    let final_call = complete.calls.get();
    assert!(final_call > 1);
    crate::factorize::finish_compact_factor_plan_generic_for_test(&space, prepared).unwrap();
    assert_eq!(complete.calls.get(), final_call);

    let failing = LateGenericSpy {
        rule: FactorGenericRule,
        fail_at: final_call,
        calls: Cell::new(0),
    };
    crate::factorize::reset_generic_factor_plan_finish_calls();
    let error = match crate::factorize::prepare_compact_factor_plan_generic_checked_for_test(
        &space, &failing,
    ) {
        Err(error) => error,
        Ok(_) => panic!("late provider failure must abort checked preparation"),
    };
    assert!(matches!(
        error,
        crate::factorize::CheckedGenericFactorPlanError::Provider(LateGenericError(call))
            if call == final_call
    ));
    assert_eq!(crate::factorize::generic_factor_plan_finish_calls(), 0);
}

#[test]
#[expect(
    clippy::arc_with_non_send_sync,
    reason = "the checked Generic API requires Arc identity while Cell is a single-threaded call spy"
)]
fn checked_generic_full_svd_preserves_provider_and_completes_unmatched_rows() {
    let rule = FactorGenericRule;
    let x = SectorId::new(1);
    let vacuum = SectorId::new(0);
    let homspace = FusionTreeHomSpace::new(
        FusionProductSpace::new([SectorLeg::new([(x, 1), (vacuum, 1)], false)]),
        FusionProductSpace::new([SectorLeg::new([(x, 1)], false)]),
    );
    let source =
        BoundDynamicFusionMapSpace::from_final_homspace_generic(Arc::new(rule), homspace).unwrap();
    let data = vec![1.0; source.space().required_len().unwrap()];
    let checked_provider = Arc::new(LateGenericSpy {
        rule: FactorGenericRule,
        fail_at: usize::MAX,
        calls: Cell::new(0),
    });
    let checked = BoundDynamicFusionMapSpace::bind_generic(
        source.space().clone(),
        Arc::clone(&checked_provider),
    )
    .unwrap();
    let input = BoundDynamicTensorRef::try_new(&checked, &data).unwrap();
    let mut dense = tenet_dense::DefaultDenseExecutor::new();
    let full = svd_full_dyn_checked_generic(&mut dense, &input).unwrap();
    assert!(Arc::ptr_eq(
        full.u().space().provider_arc(),
        &checked_provider
    ));
    assert!(Arc::ptr_eq(
        full.s().space().provider_arc(),
        &checked_provider
    ));
    assert!(Arc::ptr_eq(
        full.vh().space().provider_arc(),
        &checked_provider
    ));
    let structure = full.u().space().space().structure();
    assert!((0..structure.block_count()).any(|index| {
        matches!(
            structure.block(index).unwrap().key(),
            BlockKey::FusionTree(key) if key.codomain_tree().coupled() == vacuum
        )
    }));
}

#[test]
#[expect(
    clippy::arc_with_non_send_sync,
    reason = "the checked Generic API requires Arc identity while Cell is a single-threaded call spy"
)]
fn checked_generic_full_svd_failure_publishes_no_factors() {
    let (source, data) = generic_factorization_input();
    let failing = Arc::new(LateGenericSpy {
        rule: FactorGenericRule,
        fail_at: 2,
        calls: Cell::new(0),
    });
    let checked =
        BoundDynamicFusionMapSpace::bind_generic(source.space().clone(), failing).unwrap();
    let input = BoundDynamicTensorRef::try_new(&checked, &data).unwrap();
    let mut dense = tenet_dense::DefaultDenseExecutor::new();
    let result = svd_full_dyn_checked_generic(&mut dense, &input);
    assert!(matches!(
        result,
        Err(crate::CheckedGenericFactorPlanError::Provider(_))
    ));
}

#[test]
#[expect(
    clippy::arc_with_non_send_sync,
    reason = "the checked Generic API requires Arc identity while Cell is a single-threaded call spy"
)]
fn checked_generic_eigh_stages_dense_work_before_checked_factor_admission() {
    let x = SectorId::new(1);
    let leg = SectorLeg::new([(x, 1)], false);
    let homspace = FusionTreeHomSpace::new(
        FusionProductSpace::new([leg.clone(), leg.clone()]),
        FusionProductSpace::new([leg.clone(), leg]),
    );
    let source = BoundDynamicFusionMapSpace::from_final_homspace_generic(
        Arc::new(FactorGenericRule),
        homspace,
    )
    .unwrap();
    let data = vec![0.0; source.space().required_len().unwrap()];

    let failing = Arc::new(LateGenericSpy {
        rule: FactorGenericRule,
        fail_at: 1,
        calls: Cell::new(0),
    });
    let checked =
        BoundDynamicFusionMapSpace::bind_generic(source.space().clone(), failing).unwrap();
    let input = BoundDynamicTensorRef::try_new(&checked, &data).unwrap();
    let mut dense = CountingDense::default();
    assert!(matches!(
        eigh_full_dyn_checked_generic(&mut dense, &input),
        Err(CheckedGenericFactorPlanError::Provider(LateGenericError(1)))
    ));
    assert_eq!(dense.eigh_calls, 2);
    assert_eq!(input.data(), data);

    let complete = Arc::new(LateGenericSpy {
        rule: FactorGenericRule,
        fail_at: usize::MAX,
        calls: Cell::new(0),
    });
    let checked =
        BoundDynamicFusionMapSpace::bind_generic(source.space().clone(), Arc::clone(&complete))
            .unwrap();
    let input = BoundDynamicTensorRef::try_new(&checked, &data).unwrap();
    let mut dense = CountingDense::default();
    let full = eigh_full_dyn_checked_generic(&mut dense, &input).unwrap();
    assert_eq!(dense.eigh_calls, 2);
    assert!(Arc::ptr_eq(full.v().space().provider_arc(), &complete));
    assert!(full
        .eigenvalues()
        .iter()
        .flat_map(|entry| &entry.values)
        .all(|value| *value == 0.0));
}

#[test]
#[expect(
    clippy::arc_with_non_send_sync,
    reason = "the checked Generic API requires Arc identity while Cell is a single-threaded call spy"
)]
fn checked_generic_eigh_uses_owned_dense_output() {
    let x = SectorId::new(1);
    let leg = SectorLeg::new([(x, 1)], false);
    let homspace = FusionTreeHomSpace::new(
        FusionProductSpace::new([leg.clone(), leg.clone()]),
        FusionProductSpace::new([leg.clone(), leg]),
    );
    let source = BoundDynamicFusionMapSpace::from_final_homspace_generic(
        Arc::new(FactorGenericRule),
        homspace,
    )
    .unwrap();
    let data = vec![0.0; source.space().required_len().unwrap()];
    let provider = Arc::new(LateGenericSpy {
        rule: FactorGenericRule,
        fail_at: usize::MAX,
        calls: Cell::new(0),
    });
    let checked =
        BoundDynamicFusionMapSpace::bind_generic(source.space().clone(), Arc::clone(&provider))
            .unwrap();
    let input = BoundDynamicTensorRef::try_new(&checked, &data).unwrap();
    let mut dense = RejectEighInto::default();

    let full = eigh_full_dyn_checked_generic(&mut dense, &input).unwrap();

    assert_eq!(dense.eigh_calls, 2);
    assert_eq!(dense.eigh_into_calls, 0);
    assert!(Arc::ptr_eq(full.v().space().provider_arc(), &provider));
}

#[test]
#[expect(
    clippy::arc_with_non_send_sync,
    reason = "the checked Generic API requires Arc identity while Cell is a single-threaded call spy"
)]
fn checked_generic_eigh_keeps_owned_vectors_in_live_pairs_before_publication() {
    let x = SectorId::new(1);
    let leg = SectorLeg::new([(x, 1)], false);
    let homspace = FusionTreeHomSpace::new(
        FusionProductSpace::new([leg.clone(), leg.clone()]),
        FusionProductSpace::new([leg.clone(), leg]),
    );
    let source = BoundDynamicFusionMapSpace::from_final_homspace_generic(
        Arc::new(FactorGenericRule),
        homspace,
    )
    .unwrap();
    let data = vec![0.0; source.space().required_len().unwrap()];
    let provider = Arc::new(LateGenericSpy {
        rule: FactorGenericRule,
        fail_at: usize::MAX,
        calls: Cell::new(0),
    });
    let checked =
        BoundDynamicFusionMapSpace::bind_generic(source.space().clone(), Arc::clone(&provider))
            .unwrap();
    let input = BoundDynamicTensorRef::try_new(&checked, &data).unwrap();
    let mut dense = RejectEighInto::default();
    crate::factorize::reset_checked_eigh_pair_pointers();

    let full = eigh_full_dyn_checked_generic(&mut dense, &input).unwrap();
    let before_publication = crate::factorize::checked_eigh_pair_pointers();

    assert_eq!(dense.eigh_into_calls, 0);
    assert_eq!(dense.vector_ptrs, before_publication);
    assert_eq!(dense.vector_ptrs.len(), full.eigenvalues().len());
}

fn assert_checked_generic_eigh_live_pair_owners<D: crate::factorize::FactorScalar>() {
    let x = SectorId::new(1);
    let leg = SectorLeg::new([(x, 1)], false);
    let homspace = FusionTreeHomSpace::new(
        FusionProductSpace::new([leg.clone(), leg.clone()]),
        FusionProductSpace::new([leg.clone(), leg]),
    );
    let source = BoundDynamicFusionMapSpace::from_final_homspace_generic(
        Arc::new(FactorGenericRule),
        homspace,
    )
    .unwrap();
    let data = vec![D::zero(); source.space().required_len().unwrap()];
    let provider = Arc::new(LateGenericSpy {
        rule: FactorGenericRule,
        fail_at: usize::MAX,
        calls: Cell::new(0),
    });
    let checked =
        BoundDynamicFusionMapSpace::bind_generic(source.space().clone(), Arc::clone(&provider))
            .unwrap();
    let input = BoundDynamicTensorRef::try_new(&checked, &data).unwrap();
    let mut dense = RejectEighInto::default();
    crate::factorize::reset_checked_eigh_pair_pointers();

    let full = eigh_full_dyn_checked_generic(&mut dense, &input).unwrap();

    assert_eq!(dense.eigh_into_calls, 0);
    assert_eq!(
        dense.vector_ptrs,
        crate::factorize::checked_eigh_pair_pointers()
    );
    assert_eq!(dense.vector_ptrs.len(), full.eigenvalues().len());
    assert!(Arc::ptr_eq(full.v().space().provider_arc(), &provider));
}

#[test]
#[expect(
    clippy::arc_with_non_send_sync,
    reason = "the checked Generic API requires Arc identity while Cell is a single-threaded call spy"
)]
fn checked_generic_eigh_keeps_live_pair_owners_for_every_dtype() {
    assert_checked_generic_eigh_live_pair_owners::<f64>();
    assert_checked_generic_eigh_live_pair_owners::<f32>();
    assert_checked_generic_eigh_live_pair_owners::<Complex32>();
    assert_checked_generic_eigh_live_pair_owners::<Complex64>();
}

#[test]
#[expect(
    clippy::arc_with_non_send_sync,
    reason = "the checked Generic API requires Arc identity while Cell is a single-threaded call spy"
)]
fn checked_generic_eigh_reconstructs_complex_unequal_multi_tree_sectors() {
    let (source, hermitian, _) = generic_values_endomorphism_input();
    let source_regions = source
        .space()
        .structure()
        .coupled_sector_regions(2)
        .unwrap()
        .unwrap();
    assert_eq!(
        source_regions.iter().map(|region| region.rows()).collect::<Vec<_>>(),
        [1, 2]
    );
    let multi_tree = source_regions
        .iter()
        .find(|region| region.rows() == 2)
        .unwrap();
    assert_eq!(
        (multi_tree.row_trees().len(), multi_tree.col_trees().len()),
        (2, 2)
    );
    assert!(hermitian.iter().any(|value| value.im != 0.0));

    let (provider, checked) = bind_checked_only(&source);
    let input = BoundDynamicTensorRef::try_new(&checked, &hermitian).unwrap();
    let full =
        eigh_full_dyn_checked_generic(&mut tenet_dense::DefaultDenseExecutor::new(), &input)
            .unwrap();
    let vector_regions = full
        .v()
        .space()
        .space()
        .structure()
        .coupled_sector_regions(2)
        .unwrap()
        .unwrap();

    for source_region in source_regions.iter() {
        let vectors = vector_regions
            .iter()
            .find(|region| region.coupled() == source_region.coupled())
            .unwrap();
        let values = &full
            .eigenvalues()
            .iter()
            .find(|spectrum| spectrum.sector == source_region.coupled())
            .unwrap()
            .values;
        let n = source_region.rows();
        assert_eq!((vectors.rows(), vectors.cols(), values.len()), (n, n, n));
        for column in 0..n {
            for row in 0..n {
                let reconstructed = (0..n)
                    .map(|bond| {
                        full.v().data()[vectors.range().start + row + n * bond]
                            * values[bond]
                            * full.v().data()[vectors.range().start + column + n * bond].conj()
                    })
                    .sum::<Complex64>();
                let expected = hermitian[source_region.range().start + row + n * column];
                assert!((reconstructed - expected).norm() < 1.0e-10);
            }
        }
    }
    assert!(Arc::ptr_eq(full.v().space().provider_arc(), &provider));
}

#[test]
#[expect(
    clippy::arc_with_non_send_sync,
    reason = "the checked Generic API requires Arc identity while Cell is a single-threaded call spy"
)]
fn checked_generic_eigh_stably_keeps_raw_exact_signed_ties() {
    let (source, mut hermitian, _) = generic_values_endomorphism_input();
    let regions = source
        .space()
        .structure()
        .coupled_sector_regions(2)
        .unwrap()
        .unwrap();
    for region in regions.iter() {
        let values = match region.rows() {
            1 => &[Complex64::new(1.0, 0.0)][..],
            2 => &[
                Complex64::new(-2.0, 0.0),
                Complex64::zero(),
                Complex64::zero(),
                Complex64::new(2.0, 0.0),
            ],
            rows => panic!("unexpected checked EIGH tie fixture size {rows}"),
        };
        hermitian[region.range()].copy_from_slice(values);
    }
    let (_, checked) = bind_checked_only(&source);
    let input = BoundDynamicTensorRef::try_new(&checked, &hermitian).unwrap();
    let mut dense = RecordingEigh::default();

    let full = eigh_full_dyn_checked_generic(&mut dense, &input).unwrap();

    let tied_sector = full
        .eigenvalues()
        .iter()
        .find(|spectrum| spectrum.values.len() == 2)
        .unwrap();
    let raw_tied = dense
        .raw_values
        .iter()
        .find(|values| values.len() == 2)
        .unwrap()
        .iter()
        .copied()
        .filter(|value| value.abs() == 2.0)
        .collect::<Vec<_>>();
    let published_tied = tied_sector
        .values
        .iter()
        .copied()
        .filter(|value| value.abs() == 2.0)
        .collect::<Vec<_>>();
    assert_eq!(raw_tied.len(), 2);
    assert!(raw_tied.iter().any(|value| *value < 0.0));
    assert!(raw_tied.iter().any(|value| *value > 0.0));
    assert_eq!(published_tied, raw_tied);
}

#[test]
fn checked_generic_eig_uses_the_existing_numerical_rank_boundary() {
    let epsilon = f64::EPSILON;
    assert!(validate_eigenvector_singular_values(&[1.0, 2.0 * epsilon], 2, epsilon).is_err());
    assert!(validate_eigenvector_singular_values(&[1.0, 2.0 * epsilon * 1.01], 2, epsilon).is_ok());
    assert!(validate_eigenvector_singular_values(&[1.0, f64::NAN], 2, epsilon).is_err());
}

#[test]
#[expect(
    clippy::arc_with_non_send_sync,
    reason = "the checked Generic API requires Arc identity while Cell is a single-threaded call spy"
)]
fn checked_generic_eig_stages_dense_work_before_checked_factor_admission() {
    let x = SectorId::new(1);
    let leg = SectorLeg::new([(x, 1)], false);
    let homspace = FusionTreeHomSpace::new(
        FusionProductSpace::new([leg.clone(), leg.clone()]),
        FusionProductSpace::new([leg.clone(), leg]),
    );
    let source = BoundDynamicFusionMapSpace::from_final_homspace_generic(
        Arc::new(FactorGenericRule),
        homspace,
    )
    .unwrap();
    let data = vec![0.0; source.space().required_len().unwrap()];
    let failing = Arc::new(LateGenericSpy {
        rule: FactorGenericRule,
        fail_at: 1,
        calls: Cell::new(0),
    });
    let checked =
        BoundDynamicFusionMapSpace::bind_generic(source.space().clone(), failing).unwrap();
    let input = BoundDynamicTensorRef::try_new(&checked, &data).unwrap();
    let mut dense = CountingDense::default();
    assert!(matches!(
        eig_full_dyn_checked_generic(&mut dense, &input),
        Err(CheckedGenericFactorPlanError::Provider(LateGenericError(1)))
    ));
    assert_eq!(dense.eig_calls, 2);
    assert_eq!(dense.svd_vals_calls, 2);
    assert_eq!(input.data(), data);
}

#[test]
#[expect(
    clippy::arc_with_non_send_sync,
    reason = "the checked Generic API requires Arc identity while Cell is a single-threaded call spy"
)]
fn checked_generic_full_svd_completes_unmatched_columns_and_disjoint_space() {
    let rule = FactorGenericRule;
    let x = SectorId::new(1);
    let vacuum = SectorId::new(0);
    let homspace = FusionTreeHomSpace::new(
        FusionProductSpace::new([SectorLeg::new([(x, 1)], false)]),
        FusionProductSpace::new([SectorLeg::new([(x, 1), (vacuum, 1)], false)]),
    );
    let source =
        BoundDynamicFusionMapSpace::from_final_homspace_generic(Arc::new(rule), homspace).unwrap();
    let data = vec![1.0; source.space().required_len().unwrap()];
    let checked = BoundDynamicFusionMapSpace::bind_generic(
        source.space().clone(),
        Arc::new(LateGenericSpy {
            rule: FactorGenericRule,
            fail_at: usize::MAX,
            calls: Cell::new(0),
        }),
    )
    .unwrap();
    let input = BoundDynamicTensorRef::try_new(&checked, &data).unwrap();
    let mut dense = tenet_dense::DefaultDenseExecutor::new();
    let full = svd_full_dyn_checked_generic(&mut dense, &input).unwrap();
    let structure = full.vh().space().space().structure();
    assert!((0..structure.block_count()).any(|index| {
        matches!(
            structure.block(index).unwrap().key(),
            BlockKey::FusionTree(key) if key.domain_tree().coupled() == vacuum
        )
    }));

    let disjoint = FusionTreeHomSpace::new(
        FusionProductSpace::new([SectorLeg::new([(vacuum, 1)], false)]),
        FusionProductSpace::new([SectorLeg::new([(x, 1)], false)]),
    );
    let source =
        BoundDynamicFusionMapSpace::from_final_homspace_generic(Arc::new(rule), disjoint).unwrap();
    let checked = BoundDynamicFusionMapSpace::bind_generic(
        source.space().clone(),
        Arc::new(LateGenericSpy {
            rule: FactorGenericRule,
            fail_at: usize::MAX,
            calls: Cell::new(0),
        }),
    )
    .unwrap();
    let data = vec![1.0; source.space().required_len().unwrap()];
    let input = BoundDynamicTensorRef::try_new(&checked, &data).unwrap();
    let full = svd_full_dyn_checked_generic(&mut dense, &input).unwrap();
    assert!(full.singular_values().is_empty());
}

#[expect(
    clippy::arc_with_non_send_sync,
    reason = "the checked Generic API requires Arc identity while Cell is a single-threaded call spy"
)]
fn assert_checked_full_svd_builder_failure(fail_at: usize) {
    let (source, data) = generic_factorization_input();
    let provider = Arc::new(LateGenericSpy {
        rule: FactorGenericRule,
        fail_at,
        calls: Cell::new(0),
    });
    let checked =
        BoundDynamicFusionMapSpace::bind_generic(source.space().clone(), Arc::clone(&provider))
            .unwrap();
    let input = BoundDynamicTensorRef::try_new(&checked, &data).unwrap();
    let before = input.data().to_vec();
    let mut dense = CountingDense::default();
    let result = svd_full_dyn_checked_generic(&mut dense, &input);

    assert!(matches!(
        result,
        Err(crate::CheckedGenericFactorPlanError::Provider(LateGenericError(call)))
            if call == fail_at
    ));
    assert_eq!(input.data(), before);
    assert!(Arc::ptr_eq(input.space().provider_arc(), &provider));
    assert_eq!(dense.svd_calls, 2);
    assert_eq!(dense.qr_calls, 2);
}

// Checked full-SVD provider sequence for `generic_factorization_input`: ten
// calls of the two multiplicity-aware dimension preflights, then the single
// U output-layout enumeration (calls 11-13), then the single Vh enumeration
// (calls 14-16); the diagonal S space issues no provider query. Before the
// one-sided owner enumerated its layout once, each output enumerated twice
// (U 11-16, Vh 17-22) and these tests pinned calls 15, 19 and 22.
const FULL_SVD_DIMENSION_PREFLIGHT_CALLS: usize = 10;
const FULL_SVD_U_FIRST_CALL: usize = FULL_SVD_DIMENSION_PREFLIGHT_CALLS + 1;
const FULL_SVD_U_LAST_CALL: usize = FULL_SVD_DIMENSION_PREFLIGHT_CALLS + 3;
const FULL_SVD_VH_FIRST_CALL: usize = FULL_SVD_U_LAST_CALL + 1;
const FULL_SVD_VH_LAST_CALL: usize = FULL_SVD_U_LAST_CALL + 3;

#[test]
fn checked_generic_full_svd_u_builder_failure_preserves_provider_context() {
    // What: the first and last post-dense checked-provider calls of U-space
    // construction propagate their exact provider error.
    assert_checked_full_svd_builder_failure(FULL_SVD_U_FIRST_CALL);
    assert_checked_full_svd_builder_failure(FULL_SVD_U_LAST_CALL);
}

#[test]
fn checked_generic_full_svd_vh_builder_failure_preserves_provider_context() {
    // What: Vh-space construction propagates its exact provider error without
    // publishing U, from its first call to its last.
    assert_checked_full_svd_builder_failure(FULL_SVD_VH_FIRST_CALL);
    assert_checked_full_svd_builder_failure(FULL_SVD_VH_LAST_CALL);
}

#[test]
#[expect(
    clippy::arc_with_non_send_sync,
    reason = "the checked Generic API requires Arc identity while Cell is a single-threaded call spy"
)]
fn checked_generic_full_svd_enumerates_each_output_layout_once() {
    // What: the provider sees the dimension preflights plus exactly one
    // layout enumeration per one-sided output; the diagonal S publication
    // adds none. Formerly 22 calls (each output enumerated twice), now 16.
    let (source, data) = generic_factorization_input();
    let provider = Arc::new(LateGenericSpy {
        rule: FactorGenericRule,
        fail_at: usize::MAX,
        calls: Cell::new(0),
    });
    let checked =
        BoundDynamicFusionMapSpace::bind_generic(source.space().clone(), Arc::clone(&provider))
            .unwrap();
    let input = BoundDynamicTensorRef::try_new(&checked, &data).unwrap();
    let mut dense = CountingDense::default();
    let output = svd_full_dyn_checked_generic(&mut dense, &input).unwrap();
    assert_eq!(provider.calls.get(), FULL_SVD_VH_LAST_CALL);

    let probe_calls = |run: &dyn Fn(&LateGenericSpy)| {
        let probe = LateGenericSpy {
            rule: FactorGenericRule,
            fail_at: usize::MAX,
            calls: Cell::new(0),
        };
        run(&probe);
        probe.calls.get()
    };
    let homspace = source.space().homspace();
    let preflight = probe_calls(&|probe| {
        coupled_sector_block_dimensions_generic_checked(homspace.codomain(), probe).unwrap();
        coupled_sector_block_dimensions_generic_checked(homspace.domain(), probe).unwrap();
    });
    assert_eq!(preflight, FULL_SVD_DIMENSION_PREFLIGHT_CALLS);
    let enumeration = |factor: &BoundDynFactor<LateGenericSpy, f64>| {
        let homspace = factor.space().space().homspace().clone();
        probe_calls(&|probe| {
            homspace
                .prepare_coupled_subblock_structure_from_leg_degeneracies_generic_checked(probe)
                .unwrap();
        })
    };
    assert_eq!(
        enumeration(output.u()),
        FULL_SVD_U_LAST_CALL - FULL_SVD_U_FIRST_CALL + 1
    );
    assert_eq!(
        enumeration(output.vh()),
        FULL_SVD_VH_LAST_CALL - FULL_SVD_VH_FIRST_CALL + 1
    );
    assert_eq!(enumeration(output.s()), 0);
    assert!(Arc::ptr_eq(output.u().space().provider_arc(), &provider));
    assert!(Arc::ptr_eq(output.vh().space().provider_arc(), &provider));
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

// Checked compact-QR provider sequence for `generic_factorization_input`: the
// compact plan issues no preflight query, the left (Q) space enumerates once
// (calls 1-3), then the right (R) space enumerates once (calls 4-6). Before
// the paired builder enumerated each side once, each side enumerated twice
// (left keys 1-3, left space 4-6, right keys 7-9, right space 10-12) and the
// first right-side call was the seventh.
const COMPACT_PAIR_LEFT_LAST_CALL: usize = 3;
const COMPACT_PAIR_RIGHT_FIRST_CALL: usize = 4;
const COMPACT_PAIR_RIGHT_LAST_CALL: usize = 6;

#[test]
#[expect(
    clippy::arc_with_non_send_sync,
    reason = "the checked Generic API requires Arc identity while Cell is a single-threaded call spy"
)]
fn checked_generic_compact_pair_builder_failure_preserves_provider_context() {
    // What: the last call of the left enumeration and the first call of the
    // right enumeration propagate their exact provider error without
    // publishing either factor or touching the input.
    let (source, data) = generic_factorization_input();
    for fail_at in [COMPACT_PAIR_LEFT_LAST_CALL, COMPACT_PAIR_RIGHT_FIRST_CALL] {
        let provider = Arc::new(LateGenericSpy {
            rule: FactorGenericRule,
            fail_at,
            calls: Cell::new(0),
        });
        let checked =
            BoundDynamicFusionMapSpace::bind_generic(source.space().clone(), Arc::clone(&provider))
                .unwrap();
        let input = BoundDynamicTensorRef::try_new(&checked, &data).unwrap();
        let mut dense = CountingDense::default();
        let result = qr_compact_dyn_checked_generic(&mut dense, &input);
        assert!(matches!(
            result,
            Err(CheckedGenericFactorPlanError::Provider(LateGenericError(call))) if call == fail_at
        ));
        assert_eq!(provider.calls.get(), fail_at);
        assert_eq!(dense.qr_calls, 2);
        assert_eq!(input.data(), data);
    }

    let provider = Arc::new(LateGenericSpy {
        rule: FactorGenericRule,
        fail_at: usize::MAX,
        calls: Cell::new(0),
    });
    let checked =
        BoundDynamicFusionMapSpace::bind_generic(source.space().clone(), Arc::clone(&provider))
            .unwrap();
    let input = BoundDynamicTensorRef::try_new(&checked, &data).unwrap();
    let (q, r) = qr_compact_dyn_checked_generic(&mut CountingDense::default(), &input).unwrap();
    assert_eq!(provider.calls.get(), COMPACT_PAIR_RIGHT_LAST_CALL);
    assert_eq!(checked_enumeration_calls(&q), COMPACT_PAIR_LEFT_LAST_CALL);
    assert_eq!(
        checked_enumeration_calls(&r),
        COMPACT_PAIR_RIGHT_LAST_CALL - COMPACT_PAIR_LEFT_LAST_CALL
    );
}

#[test]
#[expect(
    clippy::arc_with_non_send_sync,
    reason = "the checked Generic API requires Arc identity while Cell is a single-threaded call spy"
)]
fn checked_generic_svd_trunc_enumerates_each_factor_layout_once() {
    // What: a truncating checked SVD queries the provider for exactly one
    // enumeration of each of the compact U and Vh spaces, the diagonal S
    // space, and the sliced U and Vh spaces, plus one dimension weight per
    // bond sector for the truncation decision. On
    // `generic_factorization_input`: compact U 3 + Vh 3, S 0, weights 2,
    // sliced U 3 + Vh 3 = 14 calls. Formerly the compact pair and both
    // sliced spaces enumerated twice: 2 * (3 + 3) + 0 + 2 + 2 * (3 + 3) = 26.
    const TRUNC_SVD_CALLS: usize = 14;
    let (source, data) = generic_factorization_input();
    let provider = Arc::new(LateGenericSpy {
        rule: FactorGenericRule,
        fail_at: usize::MAX,
        calls: Cell::new(0),
    });
    let checked =
        BoundDynamicFusionMapSpace::bind_generic(source.space().clone(), Arc::clone(&provider))
            .unwrap();
    let input = BoundDynamicTensorRef::try_new(&checked, &data).unwrap();
    let mut dense = CountingDense::default();
    let (u_compact, s, vh_compact) = svd_compact_dyn_checked_generic(&mut dense, &input).unwrap();
    let compact_calls = provider.calls.get();
    assert!(s.data().len() > 1);

    let (u, vh, truncated, _) =
        svd_trunc_factors_dyn_checked_generic(&mut dense, &input, &Truncation::rank(4)).unwrap();
    let trunc_calls = provider.calls.get() - compact_calls;
    // Weighted rank 4 keeps one value in each of the two bond sectors and
    // drops the second value of sector 1, so both sliced spaces are strict
    // sub-spaces that still carry every sector.
    assert_eq!(
        truncated
            .iter()
            .map(|entry| entry.values.len())
            .collect::<Vec<_>>(),
        [1, 1]
    );
    assert_eq!(
        compact_calls,
        checked_enumeration_calls(&u_compact)
            + checked_enumeration_calls(&s)
            + checked_enumeration_calls(&vh_compact)
    );
    let weight_calls = late_spy_calls(&|probe| {
        for entry in &truncated {
            probe.try_sqrt_dim_scalar(entry.sector).unwrap();
        }
    });
    assert_eq!(weight_calls, 2);
    assert_eq!(
        trunc_calls,
        compact_calls
            + weight_calls
            + checked_enumeration_calls(&u)
            + checked_enumeration_calls(&vh)
    );
    assert_eq!(trunc_calls, TRUNC_SVD_CALLS);
    assert!(Arc::ptr_eq(u.space().provider_arc(), &provider));
    assert!(Arc::ptr_eq(vh.space().provider_arc(), &provider));
}

#[test]
#[expect(
    clippy::arc_with_non_send_sync,
    reason = "the checked Generic API requires Arc identity while Cell is a single-threaded call spy"
)]
fn checked_generic_full_svd_local_shape_error_precedes_provider_query() {
    // What: checked tensor admission reports the local storage mismatch before
    // any provider-backed factorization work can run.
    let (source, data) = generic_factorization_input();
    let provider = Arc::new(LateGenericSpy {
        rule: FactorGenericRule,
        fail_at: 1,
        calls: Cell::new(0),
    });
    let checked =
        BoundDynamicFusionMapSpace::bind_generic(source.space().clone(), Arc::clone(&provider))
            .unwrap();
    let error = match BoundDynamicTensorRef::try_new(&checked, &data[..data.len() - 1]) {
        Ok(_) => panic!("short storage must be rejected"),
        Err(error) => error,
    };
    assert!(matches!(
        error,
        OperationError::Core(CoreError::DimensionMismatch { .. })
    ));
    assert_eq!(provider.calls.get(), 0);
}

fn checked_spy_null(
    left: bool,
    dense: &mut impl DenseExecutor,
    input: &BoundDynamicTensorRef<'_, LateGenericSpy, f64>,
) -> Result<BoundDynFactor<LateGenericSpy, f64>, CheckedGenericFactorPlanError<LateGenericError>> {
    if left {
        left_null_dyn_checked_generic(dense, input)
    } else {
        right_null_dyn_checked_generic(dense, input)
    }
}

#[test]
#[expect(
    clippy::arc_with_non_send_sync,
    reason = "the checked Generic API requires Arc identity while Cell is a single-threaded call spy"
)]
fn checked_generic_null_admits_only_after_all_dense_work_and_keeps_exact_authority() {
    // What: both null sides finish every local SVD/completion before their
    // data-dependent output admission, whose late error publishes no factor.
    let (source, _) = generic_factorization_input();
    let data = vec![0.0; source.space().required_len().unwrap()];
    for left in [true, false] {
        let complete_provider = Arc::new(LateGenericSpy {
            rule: FactorGenericRule,
            fail_at: usize::MAX,
            calls: Cell::new(0),
        });
        let complete_space = BoundDynamicFusionMapSpace::bind_generic(
            source.space().clone(),
            Arc::clone(&complete_provider),
        )
        .unwrap();
        let complete_input = BoundDynamicTensorRef::try_new(&complete_space, &data).unwrap();
        let mut complete_dense = CountingDense::default();
        let factor = checked_spy_null(left, &mut complete_dense, &complete_input).unwrap();
        assert!(Arc::ptr_eq(
            factor.space().provider_arc(),
            &complete_provider
        ));
        let final_call = complete_provider.calls.get();
        assert!(final_call > 1);
        assert!(complete_dense.svd_calls > 1);

        let failing_provider = Arc::new(LateGenericSpy {
            rule: FactorGenericRule,
            fail_at: final_call,
            calls: Cell::new(0),
        });
        let failing_space = BoundDynamicFusionMapSpace::bind_generic(
            source.space().clone(),
            Arc::clone(&failing_provider),
        )
        .unwrap();
        let failing_input = BoundDynamicTensorRef::try_new(&failing_space, &data).unwrap();
        let before = failing_input.data().to_vec();
        let mut failing_dense = CountingDense::default();
        assert!(matches!(
            checked_spy_null(left, &mut failing_dense, &failing_input),
            Err(CheckedGenericFactorPlanError::Provider(LateGenericError(call)))
                if call == final_call
        ));
        assert_eq!(failing_dense.svd_calls, complete_dense.svd_calls);
        assert_eq!(failing_dense.qr_calls, complete_dense.qr_calls);
        assert_eq!(failing_input.data(), before);
        assert!(Arc::ptr_eq(
            failing_input.space().provider_arc(),
            &failing_provider
        ));
    }
}

#[test]
#[expect(
    clippy::arc_with_non_send_sync,
    reason = "the checked Generic API requires Arc identity while Cell is a single-threaded call spy"
)]
fn checked_generic_null_dense_failure_never_reaches_output_admission() {
    // What: a later sector SVD failure stops after structural preflight and
    // before the checked output builder or any scatter can run.
    let (source, _) = generic_factorization_input();
    let data = vec![0.0; source.space().required_len().unwrap()];
    for left in [true, false] {
        let provider = Arc::new(LateGenericSpy {
            rule: FactorGenericRule,
            fail_at: usize::MAX,
            calls: Cell::new(0),
        });
        let checked =
            BoundDynamicFusionMapSpace::bind_generic(source.space().clone(), Arc::clone(&provider))
                .unwrap();
        let input = BoundDynamicTensorRef::try_new(&checked, &data).unwrap();
        let before = input.data().to_vec();
        let expected_preflight = {
            let probe = LateGenericSpy {
                rule: FactorGenericRule,
                fail_at: usize::MAX,
                calls: Cell::new(0),
            };
            let side = if left {
                source.space().homspace().codomain()
            } else {
                source.space().homspace().domain()
            };
            coupled_sector_block_dimensions_generic_checked(side, &probe).unwrap();
            probe.calls.get()
        };
        assert!(matches!(
            checked_spy_null(left, &mut FailSecondSvd::default(), &input),
            Err(CheckedGenericFactorPlanError::Operation(
                OperationError::Dense(_)
            ))
        ));
        assert_eq!(provider.calls.get(), expected_preflight);
        assert_eq!(input.data(), before);
    }
}

#[test]
#[expect(
    clippy::arc_with_non_send_sync,
    reason = "the checked Generic API requires Arc identity while Cell is a single-threaded call spy"
)]
fn checked_generic_disjoint_null_is_identity_without_dense_calls() {
    // What: disjoint support returns complete identity bases for both sides
    // and never enters SVD or completion.
    let x = SectorId::new(1);
    let vacuum = SectorId::new(0);
    let homspace = FusionTreeHomSpace::new(
        FusionProductSpace::new([SectorLeg::new([(x, 2)], false)]),
        FusionProductSpace::new([SectorLeg::new([(vacuum, 3)], false)]),
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
    let data = vec![0.0; checked.space().required_len().unwrap()];
    let input = BoundDynamicTensorRef::try_new(&checked, &data).unwrap();
    let left = left_null_dyn_checked_generic(&mut RejectExecutorCalls, &input).unwrap();
    let right = right_null_dyn_checked_generic(&mut RejectExecutorCalls, &input).unwrap();
    assert_eq!(left.data(), &[1.0, 0.0, 0.0, 1.0]);
    assert_eq!(right.data().len(), 9);
    for column in 0..3 {
        for row in 0..3 {
            assert_eq!(right.data()[row + 3 * column], f64::from(row == column));
        }
    }
    assert!(Arc::ptr_eq(left.space().provider_arc(), &provider));
    assert!(Arc::ptr_eq(right.space().provider_arc(), &provider));
}

#[test]
fn eigh_noncanonical_layout_uses_copy_fallback() {
    // What: expert noncanonical EIGH retains positive pack-and-vector-scatter copy evidence.
    let rule = Z2FusionRule;
    let tensor = hermitian_test_tensor(&rule, &[SectorId::new(0), SectorId::new(1)]);
    let bound = bound_tensor(Arc::new(rule), &tensor);
    let adjoint_space = bound.space().adjoint_view().unwrap();
    let input = BoundDynamicTensorRef::try_new(&adjoint_space, bound.data()).unwrap();
    let mut dense = tenet_dense::DefaultDenseExecutor::new();

    crate::factorize::reset_eigh_copy_probe();
    eigh_full_dyn(&mut dense, &input).unwrap();
    let probe = crate::factorize::eigh_copy_probe();

    assert!(probe.input_pack_bytes > 0);
    assert!(probe.output_scatter_bytes > 0);
}

#[test]
fn eigh_direct_rejects_a_later_nonhermitian_sector_before_any_dense_call() {
    // What: canonical EIGH validates every coupled sector without packing before any driver call.
    let rule = Z2FusionRule;
    let tensor = hermitian_test_tensor(&rule, &[SectorId::new(0), SectorId::new(1)]);
    let regions = tensor
        .structure()
        .coupled_sector_regions(2)
        .unwrap()
        .unwrap();
    let later = regions.last().unwrap();
    let mut data = tensor.data().to_vec();
    data[later.range().start + 1] += 1.0;
    let nonhermitian = TensorMap::<f64, 2, 2>::from_vec_with_fusion_space(
        data,
        tensor.fusion_space().unwrap().as_ref().clone(),
    )
    .unwrap();
    let mut dense = EighCallSpy::default();

    crate::factorize::reset_eigh_copy_probe();
    let error = eigh_full(
        &mut dense,
        &bound_tensor_ref!(Arc::new(rule), &nonhermitian),
    )
    .unwrap_err();

    assert_eq!(
        error,
        OperationError::InvalidArgument {
            message: "eigh requires Hermitian coupled-sector blocks",
        }
    );
    assert_eq!(dense.calls, 0);
    assert_eq!(
        crate::factorize::eigh_copy_probe(),
        crate::factorize::EighCopyProbe::default()
    );
}

#[test]
fn eigh_fallback_rejects_nonhermitian_complex_input_before_dense_execution() {
    // What: a valid noncanonical layout receives the same complex Hermitian preflight after packing.
    let rule = Z2FusionRule;
    let real = hermitian_test_tensor(&rule, &[SectorId::new(0), SectorId::new(1)]);
    let regions = real.structure().coupled_sector_regions(2).unwrap().unwrap();
    let later = regions.last().unwrap();
    let mut data = real
        .data()
        .iter()
        .map(|&value| Complex64::new(value, 0.0))
        .collect::<Vec<_>>();
    data[later.range().start + 1] += Complex64::new(1.0, 2.0);
    let tensor = TensorMap::<Complex64, 2, 2>::from_vec_with_fusion_space(
        data,
        real.fusion_space().unwrap().as_ref().clone(),
    )
    .unwrap();
    let bound = bound_tensor(Arc::new(rule), &tensor);
    let adjoint_space = bound.space().adjoint_view().unwrap();
    let input = BoundDynamicTensorRef::try_new(&adjoint_space, bound.data()).unwrap();
    let mut dense = EighCallSpy::default();

    crate::factorize::reset_eigh_copy_probe();
    let error = eigh_full_dyn(&mut dense, &input).unwrap_err();

    assert_eq!(
        error,
        OperationError::InvalidArgument {
            message: "eigh requires Hermitian coupled-sector blocks",
        }
    );
    assert_eq!(dense.calls, 0);
    assert!(crate::factorize::eigh_copy_probe().input_pack_bytes > 0);
}

#[test]
fn eigh_vals_rejects_a_later_nonhermitian_sector_before_any_dense_call() {
    // What: values-only EIGH validates every borrowed sector before its first no-vector driver.
    let rule = Z2FusionRule;
    let tensor = hermitian_test_tensor(&rule, &[SectorId::new(0), SectorId::new(1)]);
    let regions = tensor
        .structure()
        .coupled_sector_regions(2)
        .unwrap()
        .unwrap();
    let later = regions.last().unwrap();
    let mut data = tensor.data().to_vec();
    data[later.range().start + 1] += 1.0;
    let nonhermitian = TensorMap::<f64, 2, 2>::from_vec_with_fusion_space(
        data,
        tensor.fusion_space().unwrap().as_ref().clone(),
    )
    .unwrap();
    let before = nonhermitian.data().to_vec();
    let mut dense = EighCallSpy::default();

    let error = eigh_vals(
        &mut dense,
        &bound_tensor_ref!(Arc::new(rule), &nonhermitian),
    )
    .unwrap_err();

    assert_eq!(
        error,
        OperationError::InvalidArgument {
            message: "eigh requires Hermitian coupled-sector blocks",
        }
    );
    assert_eq!(dense.calls, 0);
    assert_eq!(nonhermitian.data(), before);
}

#[test]
fn eigh_uses_64_epsilon_relative_tolerance_for_every_factor_dtype() {
    // What: normalized residuals below 64 eps pass and those above it fail for every dtype.
    let within_f32_delta = 62.0 * f32::EPSILON * 10.0_f32.sqrt();
    let outside_f32_delta = 66.0 * f32::EPSILON * 10.0_f32.sqrt();
    let within_f64_delta = 62.0 * f64::EPSILON * 10.0_f64.sqrt();
    let outside_f64_delta = 66.0 * f64::EPSILON * 10.0_f64.sqrt();
    let within_f32 = one_sector_matrix(vec![1.0_f32, within_f32_delta, 0.0, 2.0]);
    let outside_f32 = one_sector_matrix(vec![1.0_f32, outside_f32_delta, 0.0, 2.0]);
    let within_c32 = one_sector_matrix(vec![
        Complex32::new(1.0, 0.0),
        Complex32::new(within_f32_delta, 0.0),
        Complex32::new(0.0, 0.0),
        Complex32::new(2.0, 0.0),
    ]);
    let outside_c32 = one_sector_matrix(vec![
        Complex32::new(1.0, 0.0),
        Complex32::new(outside_f32_delta, 0.0),
        Complex32::new(0.0, 0.0),
        Complex32::new(2.0, 0.0),
    ]);
    let within_f64 = one_sector_matrix(vec![1.0_f64, within_f64_delta, 0.0, 2.0]);
    let outside_f64 = one_sector_matrix(vec![1.0_f64, outside_f64_delta, 0.0, 2.0]);
    let within_c64 = one_sector_matrix(vec![
        Complex64::new(1.0, 0.0),
        Complex64::new(within_f64_delta, 0.0),
        Complex64::new(0.0, 0.0),
        Complex64::new(2.0, 0.0),
    ]);
    let outside_c64 = one_sector_matrix(vec![
        Complex64::new(1.0, 0.0),
        Complex64::new(outside_f64_delta, 0.0),
        Complex64::new(0.0, 0.0),
        Complex64::new(2.0, 0.0),
    ]);

    assert_eigh_preflight(&within_f32, true);
    assert_eigh_preflight(&outside_f32, false);
    assert_eigh_preflight(&within_c32, true);
    assert_eigh_preflight(&outside_c32, false);
    assert_eigh_preflight(&within_f64, true);
    assert_eigh_preflight(&outside_f64, false);
    assert_eigh_preflight(&within_c64, true);
    assert_eigh_preflight(&outside_c64, false);
}

#[test]
fn eigh_hermitian_preflight_is_invariant_under_finite_rescaling() {
    // What: multiplying a block cannot change a fixed relative perturbation's classification.
    for scale in [1.0e-200, 1.0, 1.0e200] {
        let accepted = one_sector_matrix(vec![
            scale,
            62.0 * f64::EPSILON * 10.0_f64.sqrt() * scale,
            0.0,
            2.0 * scale,
        ]);
        let rejected = one_sector_matrix(vec![
            scale,
            66.0 * f64::EPSILON * 10.0_f64.sqrt() * scale,
            0.0,
            2.0 * scale,
        ]);
        assert_eigh_preflight(&accepted, true);
        assert_eigh_preflight(&rejected, false);
    }
    for scale in [1.0e-30_f32, 1.0, 1.0e30] {
        let accepted = one_sector_matrix(vec![
            scale,
            62.0 * f32::EPSILON * 10.0_f32.sqrt() * scale,
            0.0,
            2.0 * scale,
        ]);
        let rejected = one_sector_matrix(vec![
            scale,
            66.0 * f32::EPSILON * 10.0_f32.sqrt() * scale,
            0.0,
            2.0 * scale,
        ]);
        assert_eigh_preflight(&accepted, true);
        assert_eigh_preflight(&rejected, false);
    }
}

#[test]
fn eigh_hermitian_preflight_preserves_subnormal_relative_defects() {
    // What: normalization precedes subtraction, so dividing by two cannot erase a minimum subnormal defect.
    let s32 = f32::from_bits(1);
    let s64 = f64::from_bits(1);
    assert_eigh_preflight(&one_sector_matrix(vec![s32, s32, s32, 0.0]), true);
    assert_eigh_preflight(&one_sector_matrix(vec![s64, s64, s64, 0.0]), true);
    assert_eigh_preflight(&one_sector_matrix(vec![s32, 0.0, s32, 0.0]), false);
    assert_eigh_preflight(&one_sector_matrix(vec![s64, 0.0, s64, 0.0]), false);
}

#[test]
fn eigh_accepts_exact_hermitian_max_magnitude_inputs() {
    // What: stable Frobenius scaling accepts exact Hermitian matrices at finite maxima.
    let max_f32 = one_sector_matrix(vec![f32::MAX, 0.0, 0.0, f32::MAX]);
    let max_f64 = one_sector_matrix(vec![f64::MAX, 0.0, 0.0, f64::MAX]);
    let max_c32 = one_sector_matrix(vec![
        Complex32::new(f32::MAX, 0.0),
        Complex32::new(0.0, 0.0),
        Complex32::new(0.0, 0.0),
        Complex32::new(f32::MAX, 0.0),
    ]);
    let max_c64 = one_sector_matrix(vec![
        Complex64::new(f64::MAX, 0.0),
        Complex64::new(0.0, 0.0),
        Complex64::new(0.0, 0.0),
        Complex64::new(f64::MAX, 0.0),
    ]);

    assert_eigh_preflight(&max_f32, true);
    assert_eigh_preflight(&max_f64, true);
    assert_eigh_preflight(&max_c32, true);
    assert_eigh_preflight(&max_c64, true);
}

#[test]
fn eigh_rejects_a_large_nonhermitian_input() {
    // What: overflow-safe tolerance comparison still rejects large finite asymmetry.
    let tensor = one_sector_matrix(vec![f64::MAX, f64::MAX, 0.0, f64::MAX]);

    assert_eigh_preflight(&tensor, false);
}

#[test]
fn eigh_relative_hermitian_preflight_is_block_size_independent() {
    // What: repeating the same relative diagonal defect cannot change acceptance with block size.
    let rtol = 64.0 * f64::EPSILON;
    for n in [2, 64] {
        for (delta, accepted) in [(rtol / 2.0, true), (2.0 * rtol, false)] {
            let mut data = vec![Complex64::new(0.0, 0.0); n * n];
            for diagonal in 0..n {
                data[diagonal + n * diagonal] = Complex64::new(1.0, delta);
            }
            assert_eigh_preflight(&one_sector_rectangular_matrix(data, n, n), accepted);
        }
    }
}

#[test]
fn eigh_relative_hermitian_preflight_counts_cross_block_pairs_twice() {
    // What: a defect spanning the 32x32 traversal boundary contributes both conjugate positions.
    const N: usize = 33;
    let rtol = 64.0 * f64::EPSILON;
    for (factor, accepted) in [(1.3, true), (1.6, false)] {
        let mut data = vec![0.0; N * N];
        for diagonal in 0..N {
            data[diagonal + N * diagonal] = 1.0;
        }
        data[N * 32] = factor * rtol * (N as f64).sqrt();
        assert_eigh_preflight(&one_sector_rectangular_matrix(data, N, N), accepted);
    }
}

#[test]
fn eigh_rejects_a_nonreal_complex_diagonal_before_dense_execution() {
    // What: complex Hermitian validation checks diagonal reality as well as off-diagonal conjugacy.
    let tensor = one_sector_matrix(vec![
        Complex64::new(1.0, 1.0),
        Complex64::new(0.0, 0.0),
        Complex64::new(0.0, 0.0),
        Complex64::new(2.0, 0.0),
    ]);
    let mut dense = EighCallSpy::default();

    let error = eigh_full(
        &mut dense,
        &bound_tensor_ref!(Arc::new(Z2FusionRule), &tensor),
    )
    .unwrap_err();

    assert!(matches!(error, OperationError::InvalidArgument { .. }));
    assert_eq!(dense.calls, 0);
}

#[test]
fn eigh_rejects_nonfinite_input_before_dense_execution() {
    // What: NaN and infinity cannot satisfy the Hermitian EIGH input contract.
    for value in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        let tensor = one_sector_matrix(vec![value, 0.0, 0.0, 2.0]);
        let mut dense = EighCallSpy::default();
        let error = eigh_full(
            &mut dense,
            &bound_tensor_ref!(Arc::new(Z2FusionRule), &tensor),
        )
        .unwrap_err();
        assert!(matches!(error, OperationError::InvalidArgument { .. }));
        assert_eq!(dense.calls, 0);
    }
}

#[test]
fn eigh_preserves_endomorphism_error_precedence() {
    // What: a non-endomorphism retains its structural error before numeric Hermitian inspection.
    let tensor = one_sector_rectangular_matrix(vec![f64::NAN; 6], 2, 3);
    let mut dense = EighCallSpy::default();

    let error = eigh_full(
        &mut dense,
        &bound_tensor_ref!(Arc::new(Z2FusionRule), &tensor),
    )
    .unwrap_err();

    assert_eq!(
        error,
        OperationError::UnsupportedTensorContractScope {
            message: "eigh requires an endomorphism (codomain == domain)",
        }
    );
    assert_eq!(dense.calls, 0);
}

#[test]
fn hermitian_region_validation_rejects_short_storage_without_panicking() {
    // What: the cross-crate region validator reports malformed storage as a typed structural error.
    let tensor = one_sector_matrix(vec![1.0_f64, 0.0, 0.0, 2.0]);
    let regions = tensor
        .structure()
        .coupled_sector_regions(1)
        .unwrap()
        .unwrap();

    let error = validate_hermitian_regions(&tensor.data()[..3], &regions).unwrap_err();

    assert_eq!(
        error,
        OperationError::ElementCountMismatch {
            expected: 4,
            actual: 3,
        }
    );
}

#[test]
fn compact_lq_noncanonical_layout_uses_copy_fallback() {
    // What: expert noncanonical compact LQ retains positive general pack-and-scatter evidence without direct-region scratch accounting.
    let rule = Z2FusionRule;
    let tensor = tsvd_test_tensor(&rule, &[SectorId::new(0), SectorId::new(1)]);
    let bound = bound_tensor(Arc::new(rule), &tensor);
    let adjoint_space = bound.space().adjoint_view().unwrap();
    let input = BoundDynamicTensorRef::try_new(&adjoint_space, bound.data()).unwrap();
    let mut dense = tenet_dense::DefaultDenseExecutor::new();

    crate::factorize::reset_compact_lq_copy_probe();
    lq_compact_dyn(&mut dense, &input).unwrap();
    let probe = crate::factorize::compact_lq_copy_probe();

    assert!(probe.input_pack_bytes > 0);
    assert!(probe.output_scatter_bytes > 0);
    assert_eq!(probe.scratch_buffer_count, 0);
    assert_eq!(probe.adjoint_scratch_fill_bytes, 0);
    assert_eq!(probe.final_adjoint_copy_bytes, 0);
}

#[test]
fn eigh_error_preserves_borrowed_input_and_publishes_no_output() {
    // What: an EIGH backend failure leaves borrowed storage unchanged and returns no vectors.
    let rule = Arc::new(Z2FusionRule);
    let canonical = hermitian_test_tensor(rule.as_ref(), &[SectorId::new(0), SectorId::new(1)]);
    let padded = padded_copy(rule.as_ref(), &canonical);
    for (tensor, is_fallback) in [(&canonical, false), (&padded, true)] {
        let before = tensor.data().to_vec();
        let mut dense = FailAfterObservingEighInput::default();

        crate::factorize::reset_eigh_copy_probe();
        let result = eigh_full(&mut dense, &bound_tensor_ref!(Arc::clone(&rule), tensor));

        assert!(matches!(result, Err(OperationError::Dense(_))));
        assert_eq!(tensor.data(), before);
        assert!(!dense.observed.is_empty());
        if is_fallback {
            assert!(crate::factorize::eigh_copy_probe().input_pack_bytes > 0);
        } else {
            assert!(dense
                .observed
                .iter()
                .all(|sector| before.windows(sector.len()).any(|window| window == sector)));
        }
    }
}

#[test]
fn compact_svd_adjoint_error_preserves_borrowed_input_and_publishes_no_factors() {
    // What: an SVD backend failure leaves the parent storage unchanged and
    // returns no partially constructed adjoint factors.
    let rule = Z2FusionRule;
    let tensor = hermitian_test_tensor(&rule, &[SectorId::new(0), SectorId::new(1)]);
    let before = tensor.data().to_vec();
    let bound = bound_tensor(Arc::new(rule), &tensor);
    let mut adjoint_dense = FailAfterObservingSvdInput::default();
    let result = svd_compact_adjoint_factors_dyn(&mut adjoint_dense, &bound.as_ref().dynamic());
    assert!(matches!(result, Err(OperationError::Dense(_))));
    assert_eq!(tensor.data(), before);
    assert!(!adjoint_dense.observed.is_empty());
}

#[test]
fn truncated_svd_adjoint_error_preserves_borrowed_input_and_publishes_no_factors() {
    let rule = Z2FusionRule;
    let tensor = hermitian_test_tensor(&rule, &[SectorId::new(0), SectorId::new(1)]);
    let before = tensor.data().to_vec();
    let bound = bound_tensor(Arc::new(rule), &tensor);
    let mut dense = FailSecondSvd::default();

    let result =
        svd_trunc_adjoint_factors_dyn(&mut dense, &bound.as_ref().dynamic(), &Truncation::rank(1));

    assert!(matches!(result, Err(OperationError::Dense(_))));
    assert_eq!(tensor.data(), before);
    assert_eq!(dense.calls, 2);
}

#[test]
fn full_svd_late_error_preserves_input_and_publishes_no_factors() {
    // What: the adjoint-oriented full-SVD engine finishes every sector before
    // allocating any returned factor.
    let rule = Z2FusionRule;
    let tensor = hermitian_test_tensor(&rule, &[SectorId::new(0), SectorId::new(1)]);
    let before = tensor.data().to_vec();
    let bound = bound_tensor(Arc::new(rule), &tensor);
    let mut dense = FailSecondSvd::default();

    crate::factorize::reset_factor_buffer_build_counts_for_test();
    let result = svd_full_adjoint_dyn(&mut dense, &bound.as_ref().dynamic());

    assert!(matches!(result, Err(OperationError::Dense(_))));
    assert_eq!(tensor.data(), before);
    assert_eq!(dense.calls, 2);
    assert_eq!(
        crate::factorize::factor_buffer_build_counts_for_test(),
        (0, 0)
    );
}

#[test]
fn full_svd_publishes_owned_factors_without_scatter_in_both_orientations() {
    // What: the production full SVD (direct and adjoint engines) admits U and
    // Vh layouts that the staged per-sector factors already occupy, so both
    // factors are transferred instead of zero-filled and scattered.
    let rule = Z2FusionRule;
    let tensor = tsvd_test_tensor(&rule, &[SectorId::new(0), SectorId::new(1)]);
    let bound = bound_tensor(Arc::new(rule), &tensor);
    let mut dense = tenet_dense::DefaultDenseExecutor::new();

    crate::factorize::reset_one_sided_publication_probe();
    let direct = svd_full_dyn(&mut dense, &bound.as_ref().dynamic()).unwrap();
    let probe = crate::factorize::one_sided_publication_probe();
    assert_eq!(
        (probe.canonical_publications, probe.fallback_publications),
        (2, 0)
    );
    assert!(probe.appended_elements > 0);

    crate::factorize::reset_one_sided_publication_probe();
    let adjoint = svd_full_adjoint_dyn(&mut dense, &bound.as_ref().dynamic()).unwrap();
    let probe = crate::factorize::one_sided_publication_probe();
    assert_eq!(
        (probe.canonical_publications, probe.fallback_publications),
        (2, 0)
    );
    assert_eq!(
        direct.u().space().space().required_len().unwrap(),
        adjoint.u().space().space().required_len().unwrap()
    );
}

#[test]
fn full_svd_adjoint_builds_only_the_final_factor_buffers() {
    let tensor = one_sector_rectangular_matrix(vec![1.0, 2.0, 3.0, 4.0, 5.0, 7.0], 2, 3);
    let bound = bound_tensor(Arc::new(Z2FusionRule), &tensor);
    let mut dense = tenet_dense::DefaultDenseExecutor::new();

    crate::factorize::reset_factor_buffer_build_counts_for_test();
    let output = svd_full_adjoint_dyn(&mut dense, &bound.as_ref().dynamic()).unwrap();

    assert_eq!(
        crate::factorize::factor_buffer_build_counts_for_test(),
        (1, 1)
    );
    assert_eq!(output.u().space().space().required_len().unwrap(), 9);
    assert_eq!(output.s().space().space().required_len().unwrap(), 6);
    assert_eq!(output.vh().space().space().required_len().unwrap(), 4);
}

fn unequal_fallback_eigh_fixtures() -> (TensorMap<f64, 1, 1>, TensorMap<Complex64, 1, 1>) {
    let source = mixed_rectangular_tensor((3, 3), (2, 2));
    let source_regions = source
        .structure()
        .coupled_sector_regions(1)
        .unwrap()
        .unwrap();
    assert_eq!(
        source_regions
            .iter()
            .map(|region| region.rows())
            .collect::<Vec<_>>(),
        vec![3, 2]
    );
    let mut data = vec![0.0; source.data().len()];
    for region in source_regions.iter() {
        for diagonal in 0..region.rows() {
            data[region.range().start + diagonal + region.rows() * diagonal] = match diagonal {
                0 => -2.0,
                1 => 2.0,
                _ => 1.0,
            };
        }
    }
    let real = TensorMap::<f64, 1, 1>::from_vec_with_fusion_space(
        data,
        source.fusion_space().unwrap().as_ref().clone(),
    )
    .unwrap();
    let complex = TensorMap::<Complex64, 1, 1>::from_vec_with_fusion_space(
        real.data()
            .iter()
            .map(|&value| Complex64::new(value, 0.0))
            .collect(),
        real.fusion_space().unwrap().as_ref().clone(),
    )
    .unwrap();
    let mut complex_data = complex.data().to_vec();
    for region in source_regions.iter() {
        complex_data[region.range().start + 1] = Complex64::new(1.2, 1.6);
        complex_data[region.range().start + region.rows()] = Complex64::new(1.2, -1.6);
    }
    let complex = TensorMap::<Complex64, 1, 1>::from_vec_with_fusion_space(
        complex_data,
        complex.fusion_space().unwrap().as_ref().clone(),
    )
    .unwrap();
    (real, complex)
}

#[test]
fn eigh_fallback_stably_orders_equal_magnitudes() {
    // What: the noncanonical fallback preserves an exact real backend tie.
    let rule = Arc::new(Z2FusionRule);
    let (source, _) = unequal_fallback_eigh_fixtures();
    let source_regions = source
        .structure()
        .coupled_sector_regions(1)
        .unwrap()
        .unwrap();
    let padded = padded_copy(rule.as_ref(), &source);
    assert!(padded
        .structure()
        .coupled_sector_regions(1)
        .unwrap()
        .is_none());
    let mut dense = RecordingEigh::default();

    let eigh = eigh_full(&mut dense, &bound_tensor_ref!(Arc::clone(&rule), &padded)).unwrap();

    assert_eq!(dense.raw_values.len(), source_regions.len());
    assert_eq!(
        eigh.eigenvalues
            .iter()
            .map(|spectrum| spectrum.sector)
            .collect::<Vec<_>>(),
        source_regions
            .iter()
            .map(|region| region.coupled())
            .collect::<Vec<_>>(),
    );
    for (spectrum, raw) in eigh.eigenvalues.iter().zip(&dense.raw_values) {
        assert!(spectrum
            .values
            .windows(2)
            .all(|pair| pair[0].abs() >= pair[1].abs()));
        let raw_tied = raw
            .iter()
            .copied()
            .filter(|value| value.abs() == 2.0)
            .collect::<Vec<_>>();
        let published_tied = spectrum
            .values
            .iter()
            .copied()
            .filter(|value| value.abs() == 2.0)
            .collect::<Vec<_>>();
        assert_eq!(raw_tied.len(), 2);
        assert!(raw_tied.iter().any(|value| *value < 0.0));
        assert!(raw_tied.iter().any(|value| *value > 0.0));
        assert_eq!(published_tied, raw_tied);
    }
}

#[test]
fn eigh_fallback_reconstructs_complex_unequal_sectors() {
    // What: the padded fallback scatters each complex owned V into the public
    // factor structure without changing any sector's V D V^H reconstruction.
    let rule = Arc::new(Z2FusionRule);
    let (_, source) = unequal_fallback_eigh_fixtures();
    let source_regions = source
        .structure()
        .coupled_sector_regions(1)
        .unwrap()
        .unwrap();
    let padded = padded_copy(rule.as_ref(), &source);
    assert!(padded
        .structure()
        .coupled_sector_regions(1)
        .unwrap()
        .is_none());
    let mut dense = tenet_dense::DefaultDenseExecutor::new();
    let eigh = eigh_full(&mut dense, &bound_tensor_ref!(Arc::clone(&rule), &padded)).unwrap();
    let vector_regions = eigh
        .v
        .structure()
        .coupled_sector_regions(1)
        .unwrap()
        .unwrap();
    for source_region in source_regions.iter() {
        let vector_region = vector_regions
            .iter()
            .find(|region| region.coupled() == source_region.coupled())
            .unwrap();
        let values = &eigh
            .eigenvalues
            .iter()
            .find(|spectrum| spectrum.sector == source_region.coupled())
            .unwrap()
            .values;
        let n = source_region.rows();
        assert_eq!(vector_region.rows(), n);
        for column in 0..n {
            for row in 0..n {
                let reconstructed = (0..n)
                    .map(|bond| {
                        eigh.v.data()[vector_region.range().start + row + n * bond]
                            * values[bond]
                            * eigh.v.data()[vector_region.range().start + column + n * bond].conj()
                    })
                    .sum::<Complex64>();
                let expected = source.data()[source_region.range().start + row + n * column];
                assert!((reconstructed - expected).norm() < 1e-9);
            }
        }
    }
}

#[test]
fn eigh_direct_column_reorder_preserves_the_literal_three_cycle() {
    // What: direct owned vectors reorder in place by destination columns [1, 2, 0].
    let mut vectors = vec![
        10.0, 11.0, 12.0, // column 0
        20.0, 21.0, 22.0, // column 1
        30.0, 31.0, 32.0, // column 2
    ];
    let mut visited = vec![false; 3];
    let mut scratch = vec![0.0; 3];
    crate::factorize::reorder_columns_in_place_for_test(
        &mut vectors,
        3,
        &[1, 2, 0],
        &mut visited,
        &mut scratch,
    );
    assert_eq!(
        vectors,
        vec![20.0, 21.0, 22.0, 30.0, 31.0, 32.0, 10.0, 11.0, 12.0]
    );
}

#[test]
fn eigh_rejects_non_finite_owned_eigenvalues_before_sorting() {
    // What: the private owned-output validator rejects malformed spectra before sorting.
    for value in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        assert_eq!(
            crate::factorize::validate_real_eigenvalues_for_test(&[value]),
            Err(OperationError::InvalidArgument {
                message: "eigenvalues must be finite",
            })
        );
    }
}

#[test]
fn eigh_vectors_retain_each_callers_exact_provider_arc() {
    // What: per-call EIGH factor construction preserves each caller's provider allocation.
    let tensor = hermitian_test_tensor(&Z2FusionRule, &[SectorId::new(0), SectorId::new(1)]);
    let first_provider = Arc::new(Z2FusionRule);
    let second_provider = Arc::new(Z2FusionRule);
    let first = bound_tensor(Arc::clone(&first_provider), &tensor);
    let second = bound_tensor(Arc::clone(&second_provider), &tensor);
    let mut dense = tenet_dense::DefaultDenseExecutor::new();

    let first_eigh = eigh_full(&mut dense, &first.as_ref()).unwrap();
    let second_eigh = eigh_full(&mut dense, &second.as_ref()).unwrap();

    assert!(Arc::ptr_eq(
        first_eigh.v.space().provider_arc(),
        &first_provider
    ));
    assert!(Arc::ptr_eq(
        second_eigh.v.space().provider_arc(),
        &second_provider
    ));
}

#[test]
fn eigh_direct_outputs_keep_executor_vector_owner() {
    // What: a one-region direct EIGH publishes the executor-returned V buffer.
    fn check<D: crate::factorize::FactorScalar>(tensor: &TensorMap<D, 1, 1>) {
        let mut dense = RejectEighInto::default();
        let eigh = eigh_full(
            &mut dense,
            &bound_tensor_ref!(Arc::new(Z2FusionRule), tensor),
        )
        .unwrap();
        assert_eq!(dense.eigh_calls, 1);
        assert_eq!(dense.eigh_into_calls, 0);
        assert_eq!(dense.vector_ptrs, vec![eigh.v.data().as_ptr() as usize]);
    }

    let tensor = one_sector_matrix(vec![2.0_f64, 0.0, 0.0, 3.0]);
    let space = tensor.fusion_space().unwrap().as_ref().clone();
    check(&tensor);
    check(
        &TensorMap::<f32, 1, 1>::from_vec_with_fusion_space(
            tensor.data().iter().map(|&value| value as f32).collect(),
            space.clone(),
        )
        .unwrap(),
    );
    check(
        &TensorMap::<Complex32, 1, 1>::from_vec_with_fusion_space(
            tensor
                .data()
                .iter()
                .map(|&value| Complex32::new(value as f32, 0.0))
                .collect(),
            space.clone(),
        )
        .unwrap(),
    );
    check(
        &TensorMap::<Complex64, 1, 1>::from_vec_with_fusion_space(
            tensor
                .data()
                .iter()
                .map(|&value| Complex64::new(value, 0.0))
                .collect(),
            space,
        )
        .unwrap(),
    );
}

#[test]
fn eigh_fallback_keeps_owned_vectors_until_the_final_scatter_for_every_dtype() {
    // What: the fallback's executor V is the same allocation immediately before
    // TeNeT's required structural scatter, not the final factor allocation.
    fn check<D: crate::factorize::FactorScalar>(tensor: &TensorMap<D, 1, 1>) {
        let rule = Arc::new(Z2FusionRule);
        let bound = bound_tensor(Arc::clone(&rule), tensor);
        let adjoint = bound.space().adjoint_view().unwrap();
        assert!(crate::factorize::compact_factor_plan_for_test(&adjoint)
            .unwrap()
            .is_none());
        let input = BoundDynamicTensorRef::try_new(&adjoint, bound.data()).unwrap();
        let mut dense = RejectEighInto::default();
        crate::factorize::reset_eigh_owned_vector_pointers();

        let eigh = eigh_full_dyn(&mut dense, &input).unwrap();
        let before_scatter = crate::factorize::eigh_owned_vector_pointers();

        assert!(!before_scatter.is_empty());
        assert_eq!(dense.eigh_into_calls, 0);
        assert_eq!(dense.vector_ptrs, before_scatter);
        assert_eq!(dense.eigh_calls, before_scatter.len());
        assert!(Arc::ptr_eq(
            input.space().provider_arc(),
            eigh.v().space().provider_arc()
        ));
        assert!(before_scatter
            .iter()
            .all(|&pointer| pointer != eigh.v().data().as_ptr() as usize));
    }

    let (real, complex) = unequal_fallback_eigh_fixtures();
    check(&real);
    let space = real.fusion_space().unwrap().as_ref().clone();
    check(
        &TensorMap::<f32, 1, 1>::from_vec_with_fusion_space(
            real.data().iter().map(|&value| value as f32).collect(),
            space.clone(),
        )
        .unwrap(),
    );
    check(
        &TensorMap::<Complex32, 1, 1>::from_vec_with_fusion_space(
            complex
                .data()
                .iter()
                .map(|value| Complex32::new(value.re as f32, value.im as f32))
                .collect(),
            complex.fusion_space().unwrap().as_ref().clone(),
        )
        .unwrap(),
    );
    check(&complex);
}

#[test]
fn eigh_zero_only_input_normalizes_to_an_empty_factorization_result() {
    // What: a zero-only endomorphism has no phantom output sector or spectrum
    // entry and does not invoke the dense executor.
    let tensor = rectangular_svd_tensor(0, 0);
    let mut dense = RejectExecutorCalls;

    let eigh = eigh_full(
        &mut dense,
        &bound_tensor_ref!(Arc::new(Z2FusionRule), &tensor),
    )
    .unwrap();

    assert!(eigh.v.data().is_empty());
    assert!(eigh.d.data().is_empty());
    assert!(eigh.eigenvalues.is_empty());
    assert!(eigh.v.space().space().homspace().domain().legs()[0]
        .sectors()
        .is_empty());
    assert!(eigh.d.space().space().homspace().codomain().legs()[0]
        .sectors()
        .is_empty());
}

#[test]
fn compact_qr_error_preserves_borrowed_input_and_publishes_no_factors() {
    // What: a QR backend failure leaves borrowed storage unchanged and returns no factor pair.
    let rule = Z2FusionRule;
    let tensor = tsvd_test_tensor(&rule, &[SectorId::new(0), SectorId::new(1)]);
    let before = tensor.data().to_vec();
    let mut dense = FailAfterObservingQrInput::default();

    let result = qr_compact(&mut dense, &bound_tensor_ref!(Arc::new(rule), &tensor));

    assert!(matches!(result, Err(OperationError::Dense(_))));
    assert_eq!(tensor.data(), before);
    assert!(!dense.observed.is_empty());
    assert!(dense
        .observed
        .iter()
        .all(|sector| before.windows(sector.len()).any(|window| window == sector)));
}

#[test]
fn compact_qr_uses_owned_executor_outputs_not_qr_into() {
    let rule = Z2FusionRule;
    let tensor = tsvd_test_tensor(&rule, &[SectorId::new(0), SectorId::new(1)]);
    let mut dense = FailAfterObservingQrInput {
        qr_succeeds: true,
        ..Default::default()
    };
    let input = bound_tensor(Arc::new(rule), &tensor);

    let (q, r) = qr_compact(&mut dense, &input.as_ref()).unwrap();

    assert!(!dense.observed.is_empty());
    assert!(!q.data().is_empty());
    assert!(!r.data().is_empty());
}

fn f64_svd_outputs(rows: usize, cols: usize) -> Vec<DenseTensor> {
    let mut dense = tenet_dense::DefaultDenseExecutor::new();
    let data = vec![1.0; rows * cols];
    dense
        .svd(DenseRead::F64(
            tenet_dense::DenseView::new(&data, &[rows, cols], &[1, rows], 0).unwrap(),
        ))
        .unwrap()
}

fn c64_svd_outputs(rows: usize, cols: usize) -> Vec<DenseTensor> {
    let mut dense = tenet_dense::DefaultDenseExecutor::new();
    let data = vec![Complex64::new(1.0, 1.0); rows * cols];
    dense
        .svd(DenseRead::C64(
            tenet_dense::DenseView::new(&data, &[rows, cols], &[1, rows], 0).unwrap(),
        ))
        .unwrap()
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

#[test]
fn compact_owned_svd_preserves_svd_into_output_precedence() {
    let tensor = rectangular_svd_tensor(2, 2);
    let bound = bound_tensor(Arc::new(Z2FusionRule), &tensor);
    let input = bound.as_ref();
    let check = |outputs: Vec<DenseTensor>, expected: &str| {
        let mut dense = FailAfterObservingSvdInput {
            outputs: Some(outputs),
            ..Default::default()
        };
        let error = svd_compact(&mut dense, &input).unwrap_err();
        assert!(format!("{error}").contains(expected), "{error:?}");
    };

    check(vec![f64_svd_outputs(2, 2).remove(0)], "exactly (U, S, Vt)");
    check(
        f64_svd_outputs(1, 1),
        "output shape mismatch: source [1, 1], destination [2, 2]",
    );
    let mut outputs = f64_svd_outputs(2, 2);
    outputs[1] = f64_svd_outputs(1, 1).remove(1);
    check(
        outputs,
        "output shape mismatch: source [1], destination [2]",
    );
    let mut outputs = f64_svd_outputs(2, 2);
    outputs[2] = f64_svd_outputs(1, 1).remove(2);
    check(
        outputs,
        "output shape mismatch: source [1, 1], destination [2, 2]",
    );
    let outputs = c64_svd_outputs(2, 2);
    let expected = outputs[0].as_f64_slice().unwrap_err();
    let mut dense = FailAfterObservingSvdInput {
        outputs: Some(outputs),
        ..Default::default()
    };
    let error = svd_compact(&mut dense, &input).unwrap_err();
    assert!(matches!(error, OperationError::Dense(actual) if actual == expected));

    let mut outputs = f64_svd_outputs(2, 2);
    outputs[1] = c64_svd_outputs(2, 2).remove(0);
    let expected = outputs[1].as_f64_slice().unwrap_err();
    let mut dense = FailAfterObservingSvdInput {
        outputs: Some(outputs),
        ..Default::default()
    };
    let error = svd_compact(&mut dense, &input).unwrap_err();
    assert!(matches!(error, OperationError::Dense(actual) if actual == expected));

    let mut outputs = f64_svd_outputs(2, 2);
    outputs[2] = c64_svd_outputs(2, 2).remove(0);
    let expected = outputs[2].as_f64_slice().unwrap_err();
    let mut dense = FailAfterObservingSvdInput {
        outputs: Some(outputs),
        ..Default::default()
    };
    let error = svd_compact(&mut dense, &input).unwrap_err();
    assert!(matches!(error, OperationError::Dense(actual) if actual == expected));
}

#[test]
fn compact_owned_qr_preserves_qr_into_output_precedence() {
    let tensor = rectangular_svd_tensor(2, 2);
    let bound = bound_tensor(Arc::new(Z2FusionRule), &tensor);
    let input = bound.as_ref();
    let check = |outputs: Vec<DenseTensor>, expected: &str| {
        let mut dense = FailAfterObservingQrInput {
            outputs: Some(outputs),
            ..Default::default()
        };
        let error = qr_compact(&mut dense, &input).unwrap_err();
        assert!(format!("{error}").contains(expected), "{error:?}");
    };

    check(vec![f64_qr_outputs(2, 2).remove(0)], "exactly (Q, R)");
    check(
        f64_qr_outputs(1, 1),
        "output shape mismatch: source [1, 1], destination [2, 2]",
    );
    let mut outputs = f64_qr_outputs(2, 2);
    outputs[1] = f64_qr_outputs(1, 1).remove(1);
    check(
        outputs,
        "output shape mismatch: source [1, 1], destination [2, 2]",
    );
    let outputs = c64_qr_outputs(2, 2);
    let expected = outputs[0].as_f64_slice().unwrap_err();
    let mut dense = FailAfterObservingQrInput {
        outputs: Some(outputs),
        ..Default::default()
    };
    let error = qr_compact(&mut dense, &input).unwrap_err();
    assert!(matches!(error, OperationError::Dense(actual) if actual == expected));
}

#[test]
fn compact_owned_eigh_preserves_eigh_into_output_precedence() {
    let tensor = one_sector_matrix(vec![2.0, 0.0, 0.0, 3.0]);
    let bound = bound_tensor(Arc::new(Z2FusionRule), &tensor);
    let input = bound.as_ref();
    let check = |outputs: Vec<DenseTensor>, expected: &str| {
        let mut dense = FailAfterObservingEighInput {
            outputs: Some(outputs),
            ..Default::default()
        };
        let error = eigh_full(&mut dense, &input).unwrap_err();
        assert!(format!("{error}").contains(expected), "{error:?}");
    };

    check(
        vec![f64_eigh_outputs(2).remove(0)],
        "dense EIGH must return exactly (values, vectors)",
    );
    check(
        f64_eigh_outputs(1),
        "output shape mismatch: source [1], destination [2]",
    );
    let mut outputs = f64_eigh_outputs(2);
    outputs[1] = f64_eigh_outputs(1).remove(1);
    check(
        outputs,
        "output shape mismatch: source [1, 1], destination [2, 2]",
    );
    let mut outputs = f64_eigh_outputs(2);
    outputs[0] = c64_eigh_outputs(2).remove(1);
    let expected = outputs[0].as_f64_slice().unwrap_err();
    let mut dense = FailAfterObservingEighInput {
        outputs: Some(outputs),
        ..Default::default()
    };
    let error = eigh_full(&mut dense, &input).unwrap_err();
    assert!(matches!(error, OperationError::Dense(actual) if actual == expected));

    let mut outputs = f64_eigh_outputs(2);
    outputs[1] = c64_eigh_outputs(2).remove(1);
    let expected = outputs[1].as_f64_slice().unwrap_err();
    let mut dense = FailAfterObservingEighInput {
        outputs: Some(outputs),
        ..Default::default()
    };
    let error = eigh_full(&mut dense, &input).unwrap_err();
    assert!(matches!(error, OperationError::Dense(actual) if actual == expected));
}

#[test]
fn compact_qr_factors_retain_each_callers_exact_provider_arc() {
    // What: per-call QR factor construction preserves each caller's provider allocation.
    let tensor = rectangular_svd_tensor(7, 5);
    let first_provider = Arc::new(Z2FusionRule);
    let second_provider = Arc::new(Z2FusionRule);
    let first = bound_tensor(Arc::clone(&first_provider), &tensor);
    let second = bound_tensor(Arc::clone(&second_provider), &tensor);
    let mut dense = tenet_dense::DefaultDenseExecutor::new();

    let (first_q, first_r) = qr_compact(&mut dense, &first.as_ref()).unwrap();
    let (second_q, second_r) = qr_compact(&mut dense, &second.as_ref()).unwrap();

    for factor in [&first_q, &first_r] {
        assert!(Arc::ptr_eq(factor.space().provider_arc(), &first_provider));
    }
    for factor in [&second_q, &second_r] {
        assert!(Arc::ptr_eq(factor.space().provider_arc(), &second_provider));
    }
}

#[test]
fn compact_lq_error_preserves_borrowed_input_and_publishes_no_factors() {
    // What: an LQ backend failure leaves borrowed storage unchanged and returns no factor pair.
    let rule = Z2FusionRule;
    let tensor = tsvd_test_tensor(&rule, &[SectorId::new(0), SectorId::new(1)]);
    let before = tensor.data().to_vec();
    let mut dense = FailAfterObservingQrInput::default();

    let result = lq_compact(&mut dense, &bound_tensor_ref!(Arc::new(rule), &tensor));

    assert!(matches!(result, Err(OperationError::Dense(_))));
    assert_eq!(tensor.data(), before);
    assert!(!dense.observed.is_empty());
}

#[test]
fn compact_lq_factors_retain_each_callers_exact_provider_arc() {
    // What: per-call LQ factor construction preserves each caller's provider allocation.
    let tensor = rectangular_svd_tensor(7, 5);
    let first_provider = Arc::new(Z2FusionRule);
    let second_provider = Arc::new(Z2FusionRule);
    let first = bound_tensor(Arc::clone(&first_provider), &tensor);
    let second = bound_tensor(Arc::clone(&second_provider), &tensor);
    let mut dense = tenet_dense::DefaultDenseExecutor::new();

    let (first_l, first_q) = lq_compact(&mut dense, &first.as_ref()).unwrap();
    let (second_l, second_q) = lq_compact(&mut dense, &second.as_ref()).unwrap();

    for factor in [&first_l, &first_q] {
        assert!(Arc::ptr_eq(factor.space().provider_arc(), &first_provider));
    }
    for factor in [&second_l, &second_q] {
        assert!(Arc::ptr_eq(factor.space().provider_arc(), &second_provider));
    }
}

#[test]
fn compact_svd_error_preserves_borrowed_input_and_publishes_no_factors() {
    // What: a provider failure cannot mutate borrowed tensor storage or return partial factors.
    let rule = Z2FusionRule;
    let tensor = tsvd_test_tensor(&rule, &[SectorId::new(0), SectorId::new(1)]);
    let before = tensor.data().to_vec();
    let mut dense = FailAfterObservingSvdInput::default();

    let result = svd_compact(&mut dense, &bound_tensor_ref!(Arc::new(rule), &tensor));

    assert!(matches!(result, Err(OperationError::Dense(_))));
    assert_eq!(tensor.data(), before);
    assert!(!dense.observed.is_empty());
    assert!(dense
        .observed
        .iter()
        .all(|sector| before.windows(sector.len()).any(|window| window == sector)));
}

#[test]
fn compact_factor_plan_does_not_retain_provider() {
    // What: plan construction does not retain the input space's provider.
    let tensor = rectangular_svd_tensor(19, 11);
    let provider = Arc::new(Z2FusionRule);
    let weak = Arc::downgrade(&provider);
    let bound = bound_tensor(Arc::clone(&provider), &tensor);
    let plan = crate::factorize::compact_factor_plan_for_test(bound.space())
        .unwrap()
        .unwrap();

    drop(bound);
    drop(provider);

    assert!(weak.upgrade().is_none());
    drop(plan);
}

#[test]
fn compact_factor_plan_is_identical_across_calls_on_one_space() {
    // What: rebuilding the per-call plan on the same bound space yields the
    // same routes and the same region tables (shared `Arc`s), with the first
    // plan still alive.
    let charges =
        [U1Irrep::new(-1), U1Irrep::new(0), U1Irrep::new(1)].map(|charge| charge.sector_id());
    let tensor = tsvd_test_tensor(&U1FusionRule, &charges);
    let bound = bound_tensor(Arc::new(U1FusionRule), &tensor);
    let first = crate::factorize::compact_factor_plan_for_test(bound.space())
        .unwrap()
        .unwrap();
    let second = crate::factorize::compact_factor_plan_for_test(bound.space())
        .unwrap()
        .unwrap();

    let (first_source, first_u, first_vh) =
        crate::factorize::compact_factor_plan_regions_for_test(&first);
    let (second_source, second_u, second_vh) =
        crate::factorize::compact_factor_plan_regions_for_test(&second);
    assert!(first_source.len() >= 3);
    assert!(Arc::ptr_eq(&first_source, &second_source));
    assert!(Arc::ptr_eq(&first_u, &second_u));
    assert!(Arc::ptr_eq(&first_vh, &second_vh));
    assert_eq!(
        crate::factorize::compact_factor_plan_routes_for_test(&first),
        crate::factorize::compact_factor_plan_routes_for_test(&second)
    );
}

#[test]
fn compact_factor_routes_agree_between_sorted_and_unsorted_region_tables() {
    // What: canonical factor regions are strictly sorted by coupled sector and
    // route without a map; an expert (unsorted) region table routes through
    // the map path to the same regions in the same source order.
    let charges =
        [U1Irrep::new(-1), U1Irrep::new(0), U1Irrep::new(1)].map(|charge| charge.sector_id());
    let tensor = tsvd_test_tensor(&U1FusionRule, &charges);
    let bound = bound_tensor(Arc::new(U1FusionRule), &tensor);
    let plan = crate::factorize::compact_factor_plan_for_test(bound.space())
        .unwrap()
        .unwrap();
    let (source, u, vh) = crate::factorize::compact_factor_plan_regions_for_test(&plan);
    assert!(u
        .windows(2)
        .all(|pair| pair[0].coupled() < pair[1].coupled()));
    assert!(vh
        .windows(2)
        .all(|pair| pair[0].coupled() < pair[1].coupled()));

    let mut shuffled_u = u.to_vec();
    let mut shuffled_vh = vh.to_vec();
    shuffled_u.rotate_left(2);
    shuffled_vh.reverse();
    assert!(!shuffled_u
        .windows(2)
        .all(|pair| pair[0].coupled() < pair[1].coupled()));
    let sorted =
        crate::factorize::validate_compact_factor_routes_for_test(&source, &u, &vh).unwrap();
    let unsorted = crate::factorize::validate_compact_factor_routes_for_test(
        &source,
        &shuffled_u,
        &shuffled_vh,
    )
    .unwrap();
    assert_eq!(
        sorted,
        crate::factorize::compact_factor_plan_routes_for_test(&plan)
    );
    assert_eq!(sorted.len(), unsorted.len());
    for (sorted_route, unsorted_route) in sorted.iter().zip(&unsorted) {
        let (sorted_source, sorted_left, sorted_right) = sorted_route.factor_regions_for_test();
        let (unsorted_source, unsorted_left, unsorted_right) =
            unsorted_route.factor_regions_for_test();
        assert_eq!(sorted_source, unsorted_source);
        assert_eq!(
            sorted_left.map(|index| &u[index]),
            unsorted_left.map(|index| &shuffled_u[index])
        );
        assert_eq!(
            sorted_right.map(|index| &vh[index]),
            unsorted_right.map(|index| &shuffled_vh[index])
        );
    }
}

#[test]
fn compact_svd_qr_lq_direct_regions_follow_factor_order_for_reversed_sector_spans() {
    let rule = Z2FusionRule;
    let source = mixed_rectangular_tensor((3, 2), (2, 4));
    let tensor = reversed_complete_grid_copy(&rule, &source);
    let bound = bound_tensor(Arc::new(rule), &tensor);
    assert!(bound
        .space()
        .space()
        .structure()
        .coupled_sector_regions(1)
        .unwrap()
        .is_some());
    let plan = crate::factorize::compact_factor_plan_for_test(bound.space())
        .unwrap()
        .unwrap();
    assert!(crate::factorize::compact_factor_plan_routes_for_test(&plan)
        .iter()
        .any(|route| {
            let (source, left, right) = route.factor_regions_for_test();
            left.is_some_and(|left| left != source) || right.is_some_and(|right| right != source)
        }));

    let input = bound.as_ref();
    let input = input.dynamic();
    let mut dense = tenet_dense::DefaultDenseExecutor::new();
    let svd = svd_compact_dyn(&mut dense, &input).unwrap();
    assert_compact_factors_reconstruct_input(&input, svd.u(), Some(svd.s()), svd.vh());
    assert_eq!(
        svd.singular_values()
            .iter()
            .map(|entry| entry.sector)
            .collect::<Vec<_>>(),
        tensor
            .structure()
            .coupled_sector_regions(1)
            .unwrap()
            .unwrap()
            .iter()
            .map(|region| region.coupled())
            .collect::<Vec<_>>()
    );
    let (q, r) = qr_compact_dyn(&mut dense, &input).unwrap();
    assert_compact_factors_reconstruct_input(&input, &q, None, &r);
    let (l, q) = lq_compact_dyn(&mut dense, &input).unwrap();
    assert_compact_factors_reconstruct_input(&input, &l, None, &q);

    let complex = TensorMap::<Complex64, 1, 1>::from_vec_with_fusion_space(
        tensor
            .data()
            .iter()
            .map(|&value| Complex64::new(value, value * 0.25))
            .collect(),
        tensor.fusion_space().unwrap().as_ref().clone(),
    )
    .unwrap();
    let complex_bound = bound_tensor(Arc::new(Z2FusionRule), &complex);
    let complex_ref = complex_bound.as_ref();
    let complex_input = complex_ref.dynamic();
    let svd = svd_compact_dyn(&mut dense, &complex_input).unwrap();
    assert_compact_factors_reconstruct_input(&complex_input, svd.u(), Some(svd.s()), svd.vh());
    assert_eq!(
        svd.singular_values()
            .iter()
            .map(|entry| entry.sector)
            .collect::<Vec<_>>(),
        complex
            .structure()
            .coupled_sector_regions(1)
            .unwrap()
            .unwrap()
            .iter()
            .map(|region| region.coupled())
            .collect::<Vec<_>>()
    );
}

#[test]
fn compact_factor_plan_rejects_duplicate_missing_mismatched_and_extra_routes() {
    // What: every nonzero source sector has one shape-correct left/right route and no extras.
    let rule = Z2FusionRule;
    let tensor = rectangular_svd_tensor(17, 13);
    let bound = bound_tensor(Arc::new(rule), &tensor);
    let plan = crate::factorize::compact_factor_plan_for_test(bound.space())
        .unwrap()
        .unwrap();
    let (source, u, vh) = crate::factorize::compact_factor_plan_regions_for_test(&plan);

    let mut duplicate = u.to_vec();
    duplicate.push(u[0].clone());
    assert!(
        crate::factorize::validate_compact_factor_routes_for_test(&source, &duplicate, &vh,)
            .is_err()
    );
    assert!(crate::factorize::validate_compact_factor_routes_for_test(&source, &[], &vh,).is_err());
    assert!(crate::factorize::validate_compact_factor_routes_for_test(&source, &vh, &vh,).is_err());

    let multi = tsvd_test_tensor(&rule, &[SectorId::new(0), SectorId::new(1)]);
    let multi_bound = bound_tensor(Arc::new(rule), &multi);
    let multi_plan = crate::factorize::compact_factor_plan_for_test(multi_bound.space())
        .unwrap()
        .unwrap();
    let (multi_source, multi_u, multi_vh) =
        crate::factorize::compact_factor_plan_regions_for_test(&multi_plan);
    let mut reversed_u = multi_u.to_vec();
    let mut reversed_vh = multi_vh.to_vec();
    reversed_u.reverse();
    reversed_vh.reverse();
    crate::factorize::validate_compact_factor_routes_for_test(
        &multi_source,
        &reversed_u,
        &reversed_vh,
    )
    .unwrap();
    let mut extra = u.to_vec();
    extra.push(
        multi_u
            .iter()
            .find(|region| region.coupled() == SectorId::new(1))
            .unwrap()
            .clone(),
    );
    assert!(
        crate::factorize::validate_compact_factor_routes_for_test(&source, &extra, &vh,).is_err()
    );
}

#[test]
fn tsvd_fusion_reconstructs_su2_tensor() {
    run_tsvd_reconstruction_case(
        &SU2FusionRule,
        &[
            SU2Irrep::from_twice_spin(0).sector_id(),
            SU2Irrep::from_twice_spin(1).sector_id(),
        ],
        true,
    );
}

#[test]
fn tsvd_fusion_reconstructs_u1_tensor() {
    run_tsvd_reconstruction_case(
        &U1FusionRule,
        &[
            U1Irrep::new(-1).sector_id(),
            U1Irrep::new(0).sector_id(),
            U1Irrep::new(1).sector_id(),
        ],
        false,
    );
}

#[test]
fn tsvd_fusion_reconstructs_fermion_parity_tensor() {
    // What: the canonical direct SVD preserves both fermion-parity sectors.
    run_tsvd_reconstruction_case(
        &FermionParityFusionRule,
        &[SectorId::new(0), SectorId::new(1)],
        true,
    );
}

#[test]
fn tsvd_fusion_reconstructs_product_rule_tensor() {
    // What: direct sector spans are keyed by the encoded product SectorId.
    let rule = product_fusion_rule(FermionParityFusionRule, U1FusionRule);
    let sectors = [
        rule.encode_sector(SectorId::new(0), U1Irrep::new(0).sector_id()),
        rule.encode_sector(SectorId::new(1), U1Irrep::new(1).sector_id()),
    ];
    run_tsvd_reconstruction_case(&rule, &sectors, true);
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

fn assert_rectangular_direct_svd(rows: usize, cols: usize) {
    let rule = Z2FusionRule;
    let tensor = rectangular_svd_tensor(rows, cols);
    let mut dense = tenet_dense::DefaultDenseExecutor::new();
    let bound = bound_tensor(Arc::new(rule), &tensor);
    crate::factorize::reset_compact_svd_copy_probe();
    let svd = svd_compact(&mut dense, &bound.as_ref()).unwrap();
    assert_factor_layout_matches_legacy_shapes(svd.u.space());
    assert_factor_layout_matches_legacy_shapes(svd.s.space());
    assert_factor_layout_matches_legacy_shapes(svd.vh.space());
    assert_compact_svd_direct_copy_probe();
    let rank = rows.min(cols);
    if rank == 0 {
        assert!(svd.u.space().space().homspace().domain().legs()[0]
            .sectors()
            .is_empty());
        assert!(svd.vh.space().space().homspace().codomain().legs()[0]
            .sectors()
            .is_empty());
    }
    let singular = svd
        .singular_values
        .first()
        .map(|entry| entry.values.as_slice())
        .unwrap_or_default();
    assert_eq!(singular.len(), rank);
    for col in 0..cols {
        for row in 0..rows {
            let reconstructed = (0..rank)
                .map(|bond| {
                    svd.u.data()[row + rows * bond]
                        * singular[bond]
                        * svd.vh.data()[bond + rank * col]
                })
                .sum::<f64>();
            assert!((reconstructed - tensor.data()[row + rows * col]).abs() < 1e-10);
        }
    }

    let adjoint = svd_compact_adjoint_factors_dyn(&mut dense, &bound.as_ref().dynamic()).unwrap();
    let adjoint_singular = adjoint
        .2
        .first()
        .map(|entry| entry.values.as_slice())
        .unwrap_or_default();
    assert_eq!(adjoint_singular.len(), rank);
    for col in 0..rows {
        for row in 0..cols {
            let reconstructed = (0..rank)
                .map(|bond| {
                    adjoint.0.data()[row + cols * bond]
                        * adjoint_singular[bond]
                        * adjoint.1.data()[bond + rank * col]
                })
                .sum::<f64>();
            assert!((reconstructed - tensor.data()[col + rows * row]).abs() < 1e-10);
        }
    }
}

#[test]
fn compact_svd_direct_spans_reconstruct_tall_and_wide_matrices() {
    // What: exact final-factor spans work for both compact rectangular shapes.
    assert_rectangular_direct_svd(5, 3);
    assert_rectangular_direct_svd(3, 5);
}

#[test]
fn compact_svd_direct_outputs_keep_executor_factor_owners() {
    fn check<D: crate::factorize::FactorScalar>(tensor: &TensorMap<D, 1, 1>) {
        let mut dense = RejectSvdInto::default();
        let bound = bound_tensor(Arc::new(Z2FusionRule), tensor);
        crate::factorize::reset_compact_svd_copy_probe();
        let svd = svd_compact(&mut dense, &bound.as_ref()).unwrap();
        assert_eq!(dense.output_ptrs.len(), 1);
        let (u, vh) = dense.output_ptrs[0];
        assert_eq!(u, svd.u.data().as_ptr() as usize);
        assert_eq!(vh, svd.vh.data().as_ptr() as usize);
        let probe = crate::factorize::compact_svd_copy_probe();
        assert_compact_svd_direct_copy_probe();
        assert_eq!(probe.owned_output_publications, 2);
        assert_eq!(probe.owned_output_owner_reused, 2);
    }

    let tensor = rectangular_svd_tensor(3, 2);
    check(&tensor);
    let space = tensor.fusion_space().unwrap().as_ref().clone();
    check(
        &TensorMap::<f32, 1, 1>::from_vec_with_fusion_space(
            tensor.data().iter().map(|&value| value as f32).collect(),
            space.clone(),
        )
        .unwrap(),
    );
    check(
        &TensorMap::<Complex32, 1, 1>::from_vec_with_fusion_space(
            tensor
                .data()
                .iter()
                .map(|&value| Complex32::new(value as f32, value as f32 * 0.25))
                .collect(),
            space.clone(),
        )
        .unwrap(),
    );
    check(
        &TensorMap::<Complex64, 1, 1>::from_vec_with_fusion_space(
            tensor
                .data()
                .iter()
                .map(|&value| Complex64::new(value, value * 0.25))
                .collect(),
            space,
        )
        .unwrap(),
    );
}

#[test]
fn compact_svd_zero_only_input_normalizes_to_an_empty_factorization_result() {
    // What: a zero-only row or column produces empty factors and no phantom
    // spectrum entry or factor route.
    assert_rectangular_direct_svd(0, 3);
    assert_rectangular_direct_svd(3, 0);
}

fn assert_rectangular_direct_qr(rows: usize, cols: usize) {
    let rule = Z2FusionRule;
    let tensor = rectangular_svd_tensor(rows, cols);
    let mut dense = tenet_dense::DefaultDenseExecutor::new();
    crate::factorize::reset_compact_qr_copy_probe();
    let (q, r) = qr_compact(&mut dense, &bound_tensor_ref!(Arc::new(rule), &tensor)).unwrap();
    assert_factor_layout_matches_legacy_shapes(q.space());
    assert_factor_layout_matches_legacy_shapes(r.space());
    let probe = crate::factorize::compact_qr_copy_probe();
    assert_eq!(probe.input_pack_bytes, 0);
    assert_eq!(probe.output_scatter_bytes, 0);
    let rank = rows.min(cols);
    if rank == 0 {
        assert!(q.space().space().homspace().domain().legs()[0]
            .sectors()
            .is_empty());
        assert!(r.space().space().homspace().codomain().legs()[0]
            .sectors()
            .is_empty());
    }
    for col in 0..cols {
        for row in 0..rows {
            let reconstructed = (0..rank)
                .map(|bond| q.data()[row + rows * bond] * r.data()[bond + rank * col])
                .sum::<f64>();
            assert!((reconstructed - tensor.data()[row + rows * col]).abs() < 1e-10);
        }
    }
}

#[test]
fn compact_qr_direct_spans_reconstruct_tall_and_wide_matrices() {
    // What: exact final Q/R spans reconstruct both compact rectangular orientations.
    assert_rectangular_direct_qr(5, 3);
    assert_rectangular_direct_qr(3, 5);
}

#[test]
fn compact_qr_direct_single_sector_keeps_executor_factor_owners() {
    let tensor = rectangular_svd_tensor(3, 2);
    let mut dense = tenet_dense::DefaultDenseExecutor::new();
    crate::factorize::reset_compact_qr_copy_probe();
    qr_compact(
        &mut dense,
        &bound_tensor_ref!(Arc::new(Z2FusionRule), &tensor),
    )
    .unwrap();
    let probe = crate::factorize::compact_qr_copy_probe();
    assert_eq!(probe.input_pack_bytes, 0);
    assert_eq!(probe.output_scatter_bytes, 0);
    assert_eq!(probe.owned_output_publications, 2);
    assert_eq!(probe.owned_output_owner_reused, 2);
}

#[test]
fn compact_qr_zero_only_input_normalizes_to_an_empty_factorization_result() {
    // What: a zero-only row or column produces empty Q/R spaces without
    // calling an invalid factor route.
    assert_rectangular_direct_qr(0, 3);
    assert_rectangular_direct_qr(3, 0);
}

fn assert_rectangular_direct_lq(rows: usize, cols: usize) {
    let rule = Z2FusionRule;
    let tensor = rectangular_svd_tensor(rows, cols);
    let mut dense = tenet_dense::DefaultDenseExecutor::new();
    crate::factorize::reset_compact_lq_copy_probe();
    let (left, right) =
        lq_compact(&mut dense, &bound_tensor_ref!(Arc::new(rule), &tensor)).unwrap();
    assert_factor_layout_matches_legacy_shapes(left.space());
    assert_factor_layout_matches_legacy_shapes(right.space());
    let probe = crate::factorize::compact_lq_copy_probe();
    assert_eq!(probe.input_pack_bytes, 0);
    assert_eq!(probe.output_scatter_bytes, 0);
    assert_eq!(probe.scratch_buffer_count, 1);
    let rank = rows.min(cols);
    if rank == 0 {
        assert!(left.space().space().homspace().domain().legs()[0]
            .sectors()
            .is_empty());
        assert!(right.space().space().homspace().codomain().legs()[0]
            .sectors()
            .is_empty());
    }
    for col in 0..cols {
        for row in 0..rows {
            let reconstructed = (0..rank)
                .map(|bond| left.data()[row + rows * bond] * right.data()[bond + rank * col])
                .sum::<f64>();
            assert!((reconstructed - tensor.data()[row + rows * col]).abs() < 1e-10);
        }
    }
    assert_eq!(probe.adjoint_scratch_fill_calls, usize::from(rank > 0));
    assert_eq!(probe.final_adjoint_copy_calls, usize::from(rank > 0) * 2);
}

#[test]
fn compact_lq_direct_spans_reconstruct_zero_unit_tall_wide_and_square() {
    // What: direct LQ spans reconstruct every rectangular edge orientation without general pack/scatter.
    for (rows, cols) in [(0, 3), (3, 0), (1, 1), (5, 3), (3, 5), (4, 4)] {
        assert_rectangular_direct_lq(rows, cols);
    }
}

#[test]
fn compact_svd_direct_and_fallback_apply_the_same_gauge() {
    // What: direct writes do not change the canonical phase chosen by the fallback.
    let rule = Z2FusionRule;
    let tensor = rectangular_svd_tensor(3, 3);
    let mut transposed_data = vec![0.0; 9];
    for col in 0..3 {
        for row in 0..3 {
            transposed_data[row + 3 * col] = tensor.data()[col + 3 * row];
        }
    }
    let transposed = TensorMap::from_vec_with_fusion_space(
        transposed_data,
        tensor.fusion_space().unwrap().as_ref().clone(),
    )
    .unwrap();
    let mut dense = tenet_dense::DefaultDenseExecutor::new();
    let direct = svd_compact(&mut dense, &bound_tensor_ref!(Arc::new(rule), &transposed)).unwrap();
    let bound = bound_tensor(Arc::new(rule), &tensor);
    let adjoint_space = bound.space().adjoint_view().unwrap();
    let fallback_input = BoundDynamicTensorRef::try_new(&adjoint_space, bound.data()).unwrap();
    let fallback = svd_compact_dyn(&mut dense, &fallback_input).unwrap();

    for (left, right) in direct.u.data().iter().zip(fallback.u().data()) {
        assert!((left - right).abs() < 1e-12);
    }
    for (left, right) in direct.vh.data().iter().zip(fallback.vh().data()) {
        assert!((left - right).abs() < 1e-12);
    }
    assert_eq!(direct.singular_values, fallback.singular_values());

    let adjoint_fallback = svd_compact_adjoint_factors_dyn(&mut dense, &fallback_input).unwrap();
    let expected = svd_compact_factors_dyn(&mut dense, &bound.as_ref().dynamic()).unwrap();
    for (actual, expected) in adjoint_fallback.0.data().iter().zip(expected.0.data()) {
        assert!((actual - expected).abs() < 1e-12);
    }
    for (actual, expected) in adjoint_fallback.1.data().iter().zip(expected.1.data()) {
        assert!((actual - expected).abs() < 1e-12);
    }
    for (actual, expected) in adjoint_fallback.2.iter().zip(&expected.2) {
        assert_eq!(actual.sector, expected.sector);
        for (actual, expected) in actual.values.iter().zip(&expected.values) {
            assert!((actual - expected).abs() < 1e-12);
        }
    }
}

#[test]
fn compact_svd_adjoint_accepts_padded_parent_layout() {
    // What: the optimized adjoint path uses the existing packed fallback for
    // custom offsets instead of materializing a canonical adjoint input.
    let rule = Z2FusionRule;
    let parent = padded_copy(&rule, &rectangular_svd_tensor(5, 3));
    let bound = bound_tensor(Arc::new(rule), &parent);
    let mut dense = tenet_dense::DefaultDenseExecutor::new();
    let actual = svd_compact_adjoint_factors_dyn(&mut dense, &bound.as_ref().dynamic()).unwrap();
    let singular = &actual.2[0].values;
    let source = parent.structure().block(0).unwrap();
    for col in 0..5 {
        for row in 0..3 {
            let reconstructed = (0..3)
                .map(|bond| {
                    actual.0.data()[row + 3 * bond]
                        * singular[bond]
                        * actual.1.data()[bond + 3 * col]
                })
                .sum::<f64>();
            let expected = parent.data()
                [source.offset() + col * source.strides()[0] + row * source.strides()[1]];
            assert!((reconstructed - expected).abs() < 1e-10);
        }
    }
}

#[test]
fn compact_svd_c64_reconstructs_mixed_tall_and_wide_sectors_without_copies() {
    use num_complex::Complex64;

    // What: one call reconstructs mixed rectangular complex sectors directly in final storage.
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
    let tensor = TensorMap::<Complex64, 1, 1>::from_vec_with_fusion_space(
        (0..space.required_len().unwrap())
            .map(|index| {
                Complex64::new(
                    ((index * 7 + 2) % 17) as f64 - 6.0,
                    ((index * 5 + 3) % 13) as f64 * 0.25 - 1.0,
                )
            })
            .collect(),
        space,
    )
    .unwrap();
    let mut dense = tenet_dense::DefaultDenseExecutor::new();
    crate::factorize::reset_compact_svd_copy_probe();

    let svd = svd_compact(&mut dense, &bound_tensor_ref!(Arc::new(rule), &tensor)).unwrap();

    assert_compact_svd_direct_copy_probe();
    let input_regions = tensor
        .structure()
        .coupled_sector_regions(1)
        .unwrap()
        .unwrap();
    let u_regions = svd
        .u
        .tensor()
        .structure()
        .coupled_sector_regions(1)
        .unwrap()
        .unwrap();
    let vh_regions = svd
        .vh
        .tensor()
        .structure()
        .coupled_sector_regions(1)
        .unwrap()
        .unwrap();
    for input_region in input_regions.iter() {
        let sector = input_region.coupled();
        let u_region = u_regions
            .iter()
            .find(|region| region.coupled() == sector)
            .unwrap();
        let vh_region = vh_regions
            .iter()
            .find(|region| region.coupled() == sector)
            .unwrap();
        let singular = &svd
            .singular_values
            .iter()
            .find(|values| values.sector == sector)
            .unwrap()
            .values;
        let rows = input_region.rows();
        let cols = input_region.cols();
        let rank = rows.min(cols);
        for col in 0..cols {
            for row in 0..rows {
                let reconstructed = (0..rank)
                    .map(|bond| {
                        svd.u.data()[u_region.range().start + row + rows * bond]
                            * singular[bond]
                            * svd.vh.data()[vh_region.range().start + bond + rank * col]
                    })
                    .sum::<Complex64>();
                let expected = tensor.data()[input_region.range().start + row + rows * col];
                assert!((reconstructed - expected).norm() < 1e-10);
            }
        }
    }
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

#[test]
fn compact_qr_c64_reconstructs_mixed_tall_and_wide_sectors_without_copies() {
    use num_complex::Complex64;

    // What: one complex QR call reconstructs mixed rectangular sectors in final storage.
    let rule = Z2FusionRule;
    let source = mixed_rectangular_c32_tensor();
    let tensor = TensorMap::<Complex64, 1, 1>::from_vec_with_fusion_space(
        source
            .data()
            .iter()
            .map(|value| Complex64::new(value.re as f64, value.im as f64))
            .collect(),
        source.fusion_space().unwrap().as_ref().clone(),
    )
    .unwrap();
    let mut dense = tenet_dense::DefaultDenseExecutor::new();
    crate::factorize::reset_compact_qr_copy_probe();

    let (q, r) = qr_compact(&mut dense, &bound_tensor_ref!(Arc::new(rule), &tensor)).unwrap();

    let probe = crate::factorize::compact_qr_copy_probe();
    assert_eq!(probe.input_pack_bytes, 0);
    assert_eq!(probe.output_scatter_bytes, 0);
    let input_regions = tensor
        .structure()
        .coupled_sector_regions(1)
        .unwrap()
        .unwrap();
    let q_regions = q.structure().coupled_sector_regions(1).unwrap().unwrap();
    let r_regions = r.structure().coupled_sector_regions(1).unwrap().unwrap();
    for input_region in input_regions.iter() {
        let sector = input_region.coupled();
        let q_region = q_regions
            .iter()
            .find(|region| region.coupled() == sector)
            .unwrap();
        let r_region = r_regions
            .iter()
            .find(|region| region.coupled() == sector)
            .unwrap();
        let rows = input_region.rows();
        let cols = input_region.cols();
        let rank = rows.min(cols);
        for col in 0..cols {
            for row in 0..rows {
                let reconstructed = (0..rank)
                    .map(|bond| {
                        q.data()[q_region.range().start + row + rows * bond]
                            * r.data()[r_region.range().start + bond + rank * col]
                    })
                    .sum::<Complex64>();
                let expected = tensor.data()[input_region.range().start + row + rows * col];
                assert!((reconstructed - expected).norm() < 1e-10);
            }
        }
    }
}

#[test]
fn compact_lq_c64_reconstructs_mixed_tall_and_wide_sectors_with_bounded_scratch() {
    use num_complex::Complex64;

    // What: one complex LQ call reconstructs mixed rectangular sectors using bounded adjoint scratch and final regions.
    let rule = Z2FusionRule;
    let source = mixed_rectangular_c32_tensor();
    let tensor = TensorMap::<Complex64, 1, 1>::from_vec_with_fusion_space(
        source
            .data()
            .iter()
            .map(|value| Complex64::new(value.re as f64, value.im as f64))
            .collect(),
        source.fusion_space().unwrap().as_ref().clone(),
    )
    .unwrap();
    let mut dense = tenet_dense::DefaultDenseExecutor::new();
    crate::factorize::reset_compact_lq_copy_probe();

    let (left, right) =
        lq_compact(&mut dense, &bound_tensor_ref!(Arc::new(rule), &tensor)).unwrap();

    let probe = crate::factorize::compact_lq_copy_probe();
    assert_eq!(probe.input_pack_bytes, 0);
    assert_eq!(probe.output_scatter_bytes, 0);
    assert_eq!(probe.scratch_buffer_count, 1);
    assert!(probe.adjoint_scratch_fill_bytes > 0);
    assert!(probe.final_adjoint_copy_bytes > 0);
    let input_regions = tensor
        .structure()
        .coupled_sector_regions(1)
        .unwrap()
        .unwrap();
    let left_regions = left.structure().coupled_sector_regions(1).unwrap().unwrap();
    let right_regions = right
        .structure()
        .coupled_sector_regions(1)
        .unwrap()
        .unwrap();
    for input_region in input_regions.iter() {
        let sector = input_region.coupled();
        let left_region = left_regions
            .iter()
            .find(|region| region.coupled() == sector)
            .unwrap();
        let right_region = right_regions
            .iter()
            .find(|region| region.coupled() == sector)
            .unwrap();
        let rows = input_region.rows();
        let cols = input_region.cols();
        let rank = rows.min(cols);
        for col in 0..cols {
            for row in 0..rows {
                let reconstructed = (0..rank)
                    .map(|bond| {
                        left.data()[left_region.range().start + row + rows * bond]
                            * right.data()[right_region.range().start + bond + rank * col]
                    })
                    .sum::<Complex64>();
                let expected = tensor.data()[input_region.range().start + row + rows * col];
                assert!((reconstructed - expected).norm() < 1e-10);
            }
        }
    }
}

#[test]
fn compact_svd_c32_reconstructs_mixed_tall_and_wide_sectors_without_copies() {
    // What: single-precision complex direct spans reconstruct both rectangular orientations.
    let rule = Z2FusionRule;
    let tensor = mixed_rectangular_c32_tensor();
    let mut dense = tenet_dense::DefaultDenseExecutor::new();
    crate::factorize::reset_compact_svd_copy_probe();

    let svd = svd_compact(&mut dense, &bound_tensor_ref!(Arc::new(rule), &tensor)).unwrap();

    assert_compact_svd_direct_copy_probe();
    let input_regions = tensor
        .structure()
        .coupled_sector_regions(1)
        .unwrap()
        .unwrap();
    let u_regions = svd
        .u
        .tensor()
        .structure()
        .coupled_sector_regions(1)
        .unwrap()
        .unwrap();
    let vh_regions = svd
        .vh
        .tensor()
        .structure()
        .coupled_sector_regions(1)
        .unwrap()
        .unwrap();
    for input_region in input_regions.iter() {
        let sector = input_region.coupled();
        let u_region = u_regions
            .iter()
            .find(|region| region.coupled() == sector)
            .unwrap();
        let vh_region = vh_regions
            .iter()
            .find(|region| region.coupled() == sector)
            .unwrap();
        let singular = &svd
            .singular_values
            .iter()
            .find(|values| values.sector == sector)
            .unwrap()
            .values;
        let rows = input_region.rows();
        let cols = input_region.cols();
        let rank = rows.min(cols);
        for col in 0..cols {
            for row in 0..rows {
                let reconstructed = (0..rank)
                    .map(|bond| {
                        svd.u.data()[u_region.range().start + row + rows * bond]
                            * singular[bond] as f32
                            * svd.vh.data()[vh_region.range().start + bond + rank * col]
                    })
                    .sum::<Complex32>();
                let expected = tensor.data()[input_region.range().start + row + rows * col];
                assert!((reconstructed - expected).norm() < 2e-4);
            }
        }
    }
}

#[test]
fn compact_svd_c32_direct_and_fallback_apply_the_same_gauge() {
    // What: the single-precision direct path preserves the fallback's canonical complex phase.
    let rule = Z2FusionRule;
    let real = rectangular_svd_tensor(3, 3);
    let data = real
        .data()
        .iter()
        .enumerate()
        .map(|(index, &value)| Complex32::new(value as f32, (index as f32 - 3.0) * 0.25))
        .collect::<Vec<_>>();
    let tensor =
        TensorMap::from_vec_with_fusion_space(data, real.fusion_space().unwrap().as_ref().clone())
            .unwrap();
    let mut transposed_data = vec![Complex32::new(0.0, 0.0); 9];
    for col in 0..3 {
        for row in 0..3 {
            transposed_data[row + 3 * col] = tensor.data()[col + 3 * row];
        }
    }
    let transposed = TensorMap::from_vec_with_fusion_space(
        transposed_data,
        tensor.fusion_space().unwrap().as_ref().clone(),
    )
    .unwrap();
    let mut dense = tenet_dense::DefaultDenseExecutor::new();
    crate::factorize::reset_compact_svd_copy_probe();
    let direct = svd_compact(&mut dense, &bound_tensor_ref!(Arc::new(rule), &transposed)).unwrap();
    assert_compact_svd_direct_copy_probe();
    let bound = bound_tensor(Arc::new(rule), &tensor);
    let adjoint_space = bound.space().adjoint_view().unwrap();
    let fallback_input = BoundDynamicTensorRef::try_new(&adjoint_space, bound.data()).unwrap();
    crate::factorize::reset_compact_svd_copy_probe();
    let fallback = svd_compact_dyn(&mut dense, &fallback_input).unwrap();
    let fallback_probe = crate::factorize::compact_svd_copy_probe();
    assert!(fallback_probe.input_pack_calls > 0);
    assert!(fallback_probe.output_scatter_calls > 0);
    for entry in &direct.singular_values {
        assert!(entry.values.last().is_some_and(|value| *value > 1e-3));
        assert!(entry
            .values
            .windows(2)
            .all(|pair| (pair[0] - pair[1]).abs() > 1e-3));
    }

    assert_eq!(direct.u.data().len(), fallback.u().data().len());
    for (left, right) in direct.u.data().iter().zip(fallback.u().data()) {
        assert!((*left - *right).norm() < 2e-5);
    }
    assert_eq!(direct.vh.data().len(), fallback.vh().data().len());
    for (left, right) in direct.vh.data().iter().zip(fallback.vh().data()) {
        assert!((*left - *right).norm() < 2e-5);
    }
    assert_eq!(
        direct.singular_values.len(),
        fallback.singular_values().len()
    );
    for (left_entry, right_entry) in direct
        .singular_values
        .iter()
        .zip(fallback.singular_values())
    {
        assert_eq!(left_entry.sector, right_entry.sector);
        assert_eq!(left_entry.values.len(), right_entry.values.len());
        for (left, right) in left_entry.values.iter().zip(&right_entry.values) {
            assert!((left - right).abs() < 1e-5);
        }
    }
}

#[test]
fn svd_trunc_c32_reports_the_discarded_reconstruction_error() {
    // What: Complex32 spectrum buffering and truncation preserve the reported discarded norm.
    let rule = Z2FusionRule;
    let tensor = mixed_rectangular_c32_tensor();
    let mut dense = tenet_dense::DefaultDenseExecutor::new();
    crate::factorize::reset_compact_svd_copy_probe();

    let svd = svd_trunc(
        &mut dense,
        &bound_tensor_ref!(Arc::new(rule), &tensor),
        &Truncation::rank(4),
    )
    .unwrap();

    assert_compact_svd_direct_copy_probe();
    assert_eq!(
        svd.singular_values
            .iter()
            .map(|entry| entry.values.len())
            .sum::<usize>(),
        4
    );
    let input_regions = tensor
        .structure()
        .coupled_sector_regions(1)
        .unwrap()
        .unwrap();
    let u_regions = svd
        .u
        .tensor()
        .structure()
        .coupled_sector_regions(1)
        .unwrap()
        .unwrap();
    let vh_regions = svd
        .vh
        .tensor()
        .structure()
        .coupled_sector_regions(1)
        .unwrap()
        .unwrap();
    let mut distance_squared = 0.0f64;
    for input_region in input_regions.iter() {
        let sector = input_region.coupled();
        let singular = &svd
            .singular_values
            .iter()
            .find(|values| values.sector == sector)
            .unwrap()
            .values;
        let u_region = u_regions.iter().find(|region| region.coupled() == sector);
        let vh_region = vh_regions.iter().find(|region| region.coupled() == sector);
        let rows = input_region.rows();
        let cols = input_region.cols();
        for col in 0..cols {
            for row in 0..rows {
                let reconstructed = match (u_region, vh_region) {
                    (Some(u_region), Some(vh_region)) => (0..singular.len())
                        .map(|bond| {
                            svd.u.data()[u_region.range().start + row + rows * bond]
                                * singular[bond] as f32
                                * svd.vh.data()
                                    [vh_region.range().start + bond + singular.len() * col]
                        })
                        .sum::<Complex32>(),
                    _ => Complex32::new(0.0, 0.0),
                };
                let expected = tensor.data()[input_region.range().start + row + rows * col];
                distance_squared += (reconstructed - expected).norm_sqr() as f64;
            }
        }
    }
    let distance = distance_squared.sqrt();
    assert!(svd.error > 0.0);
    assert!(
        (distance - svd.error).abs() < 2e-3,
        "Complex32 distance {distance} != error {}",
        svd.error
    );
}

fn weighted_norm_squared_of_difference<R>(
    rule: &R,
    lhs: &TensorMap<f64, 2, 2>,
    rhs: &TensorMap<f64, 2, 2>,
) -> f64
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64>,
{
    let lhs_structure = std::sync::Arc::clone(lhs.structure());
    let rhs_structure = std::sync::Arc::clone(rhs.structure());
    assert_eq!(lhs_structure.block_count(), rhs_structure.block_count());
    let mut total = 0.0;
    for index in 0..lhs_structure.block_count() {
        let lhs_block = lhs_structure.block(index).unwrap();
        let rhs_block = rhs_structure.block(index).unwrap();
        assert_eq!(lhs_block.key(), rhs_block.key());
        let BlockKey::FusionTree(key) = lhs_block.key() else {
            continue;
        };
        let weight = rule.dim_scalar(key.codomain_tree().coupled());
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
            let difference = lhs.data()[lhs_position] - rhs.data()[rhs_position];
            total += weight * difference * difference;
            for axis in 0..shape.len() {
                multi_index[axis] += 1;
                if multi_index[axis] < shape[axis] {
                    break;
                }
                multi_index[axis] = 0;
            }
        }
    }
    total
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

#[test]
fn svd_compact_factor_dims_include_sectors_without_populated_trees() {
    // What: public compact SVD retains the complete original leg space,
    // including a sector absent from every populated fusion block.
    let rule = U1FusionRule;
    let neutral = U1Irrep::new(0).sector_id();
    let positive = U1Irrep::new(1).sector_id();
    let homspace = FusionTreeHomSpace::new(
        FusionProductSpace::new([SectorLeg::new([(neutral, 2), (positive, 3)], false)]),
        FusionProductSpace::new([SectorLeg::new([(neutral, 2)], false)]),
    );
    let space = FusionTensorMapSpace::from_degeneracy_shapes_coupled(
        TensorMapSpace::<1, 1>::from_dims([5], [2]).unwrap(),
        homspace,
        &rule,
        [vec![2, 2]],
    )
    .unwrap();
    let tensor = TensorMap::from_vec_with_fusion_space(vec![1.0, 2.0, 3.0, 4.0], space).unwrap();
    let original_homspace = tensor.fusion_space().unwrap().homspace().clone();
    let mut dense = tenet_dense::DefaultDenseExecutor::new();

    let result = svd_compact(&mut dense, &bound_tensor_ref!(Arc::new(rule), &tensor)).unwrap();

    assert_factor_layout_matches_legacy_shapes(result.u.space());
    assert_factor_layout_matches_legacy_shapes(result.s.space());
    assert_factor_layout_matches_legacy_shapes(result.vh.space());
    assert_eq!(result.u.tensor().space().dims(), &[5, 2]);
    assert_eq!(result.vh.tensor().space().dims(), &[2, 2]);
    assert_eq!(
        result
            .u
            .tensor()
            .fusion_space()
            .unwrap()
            .homspace()
            .codomain(),
        original_homspace.codomain()
    );
    assert_eq!(
        result
            .vh
            .tensor()
            .fusion_space()
            .unwrap()
            .homspace()
            .domain(),
        original_homspace.domain()
    );
}

#[test]
fn svd_compact_preserves_asymmetric_non_self_dual_u1_factor_layouts() {
    // What: canonical factors retain unequal degeneracies and the dual U(1)
    // domain convention while reconstructing both coupled sectors.
    let rule = U1FusionRule;
    let neutral = U1Irrep::new(0).sector_id();
    let positive = U1Irrep::new(1).sector_id();
    let negative = U1Irrep::new(-1).sector_id();
    let codomain = SectorLeg::new([(neutral, 3), (positive, 2)], false);
    let domain = SectorLeg::new([(neutral, 1), (negative, 4)], true);
    let homspace = FusionTreeHomSpace::new(
        FusionProductSpace::new([codomain]),
        FusionProductSpace::new([domain]),
    );
    let shapes = homspace
        .fusion_tree_keys(&rule)
        .iter()
        .map(|key| match key.codomain_tree().coupled() {
            sector if sector == neutral => vec![3, 1],
            sector if sector == positive => vec![2, 4],
            sector => panic!("unexpected U(1) sector {sector:?}"),
        })
        .collect::<Vec<_>>();
    let space = FusionTensorMapSpace::from_degeneracy_shapes_coupled(
        TensorMapSpace::<1, 1>::from_dims([5], [5]).unwrap(),
        homspace,
        &rule,
        shapes,
    )
    .unwrap();
    let tensor = TensorMap::<f64, 1, 1>::from_vec_with_fusion_space(
        (0..space.required_len().unwrap())
            .map(|index| ((index * 7 + 3) % 17) as f64 - 6.0)
            .collect(),
        space,
    )
    .unwrap();
    let mut dense = tenet_dense::DefaultDenseExecutor::new();

    let svd = svd_compact(&mut dense, &bound_tensor_ref!(Arc::new(rule), &tensor)).unwrap();

    assert_factor_layout_matches_legacy_shapes(svd.u.space());
    assert_factor_layout_matches_legacy_shapes(svd.s.space());
    assert_factor_layout_matches_legacy_shapes(svd.vh.space());
    let input_regions = tensor
        .structure()
        .coupled_sector_regions(1)
        .unwrap()
        .unwrap();
    let u_regions = svd
        .u
        .structure()
        .coupled_sector_regions(1)
        .unwrap()
        .unwrap();
    let vh_regions = svd
        .vh
        .structure()
        .coupled_sector_regions(1)
        .unwrap()
        .unwrap();
    for input_region in input_regions.iter() {
        let sector = input_region.coupled();
        let u_region = u_regions
            .iter()
            .find(|region| region.coupled() == sector)
            .unwrap();
        let vh_region = vh_regions
            .iter()
            .find(|region| region.coupled() == sector)
            .unwrap();
        let singular = &svd
            .singular_values
            .iter()
            .find(|entry| entry.sector == sector)
            .unwrap()
            .values;
        for col in 0..input_region.cols() {
            for row in 0..input_region.rows() {
                let reconstructed = (0..singular.len())
                    .map(|bond| {
                        svd.u.data()[u_region.range().start + row + input_region.rows() * bond]
                            * singular[bond]
                            * svd.vh.data()[vh_region.range().start + bond + singular.len() * col]
                    })
                    .sum::<f64>();
                let expected =
                    tensor.data()[input_region.range().start + row + input_region.rows() * col];
                assert!((reconstructed - expected).abs() < 1.0e-10);
            }
        }
    }
}

#[test]
fn typed_factor_axis_sum_overflow_is_exact_without_storage_materialization() {
    // What: an axis whose structural-zero degeneracies exceed usize reports
    // the exact checked error without allocating storage for those dimensions.
    let rule = U1FusionRule;
    let homspace = FusionTreeHomSpace::new(
        FusionProductSpace::new([SectorLeg::new(
            [
                (U1Irrep::new(1).sector_id(), usize::MAX),
                (U1Irrep::new(2).sector_id(), 1),
            ],
            false,
        )]),
        FusionProductSpace::new([]),
    );
    let space = FusionTensorMapSpace::from_degeneracy_shapes_coupled(
        TensorMapSpace::<1, 0>::from_dims([1], []).unwrap(),
        homspace,
        &rule,
        Vec::<Vec<usize>>::new(),
    )
    .unwrap();

    let error = typed_from_dyn::<_, f64, 1, 0>(
        &rule,
        (
            tenet_tensors::DynamicFusionMapSpace::from_typed(&space),
            Vec::new(),
        ),
    )
    .unwrap_err();

    assert_eq!(error, OperationError::Core(CoreError::ElementCountOverflow));
}

fn u1_lowest_label_matrix(rows: usize, cols: usize) -> TensorMap<f64, 1, 1> {
    let rule = U1FusionRule;
    let minimum = U1Irrep::new(i32::MIN + 1).sector_id();
    let homspace = FusionTreeHomSpace::new(
        FusionProductSpace::new([SectorLeg::new([(minimum, rows)], false)]),
        FusionProductSpace::new([SectorLeg::new([(minimum, cols)], false)]),
    );
    let space = FusionTensorMapSpace::from_degeneracy_shapes_coupled(
        TensorMapSpace::<1, 1>::from_dims([rows], [cols]).unwrap(),
        homspace,
        &rule,
        [vec![rows, cols]],
    )
    .unwrap();
    TensorMap::from_vec_with_fusion_space(
        (0..rows * cols)
            .map(|index| ((index * 7 + 3) % 17) as f64 - 5.0)
            .collect(),
        space,
    )
    .unwrap()
}

fn assert_matrix_product(
    expected: &[f64],
    rows: usize,
    inner: usize,
    cols: usize,
    left: &[f64],
    right: &[f64],
) {
    for col in 0..cols {
        for row in 0..rows {
            let actual = (0..inner)
                .map(|index| left[row + rows * index] * right[index + inner * col])
                .sum::<f64>();
            assert!(
                (actual - expected[row + rows * col]).abs() < 1e-10,
                "matrix product differs at ({row}, {col}): {actual} != {}",
                expected[row + rows * col]
            );
        }
    }
}

#[test]
fn compact_factors_do_not_relabel_the_lowest_u1_sector() {
    // What: compact SVD, QR, and LQ return correctly oriented factors at the lowest U(1) label.
    let rule = U1FusionRule;
    let tensor = u1_lowest_label_matrix(3, 2);
    let mut dense = tenet_dense::DefaultDenseExecutor::new();

    let svd = svd_compact(&mut dense, &bound_tensor_ref!(Arc::new(rule), &tensor)).unwrap();
    assert_eq!(svd.u.tensor().space().dims(), &[3, 2]);
    assert_eq!(svd.s.tensor().space().dims(), &[2, 2]);
    assert_eq!(svd.vh.tensor().space().dims(), &[2, 2]);
    let singular = &svd.singular_values[0].values;
    let mut scaled_vh = svd.vh.data().to_vec();
    for col in 0..2 {
        for row in 0..2 {
            scaled_vh[row + 2 * col] *= singular[row];
        }
    }
    assert_matrix_product(tensor.data(), 3, 2, 2, svd.u.data(), &scaled_vh);

    let (q, r) = qr_compact(&mut dense, &bound_tensor_ref!(Arc::new(rule), &tensor)).unwrap();
    assert_eq!(q.tensor().space().dims(), &[3, 2]);
    assert_eq!(r.tensor().space().dims(), &[2, 2]);
    assert_matrix_product(tensor.data(), 3, 2, 2, q.data(), r.data());

    let (l, q) = lq_compact(&mut dense, &bound_tensor_ref!(Arc::new(rule), &tensor)).unwrap();
    assert_eq!(l.tensor().space().dims(), &[3, 2]);
    assert_eq!(q.tensor().space().dims(), &[2, 2]);
    assert_matrix_product(tensor.data(), 3, 2, 2, l.data(), q.data());
}

#[test]
fn eigh_full_does_not_relabel_the_lowest_u1_sector() {
    // What: full EIGH preserves the eigen equation and factor orientation at the lowest U(1) label.
    let rule = U1FusionRule;
    let minimum = U1Irrep::new(i32::MIN + 1).sector_id();
    let homspace = FusionTreeHomSpace::new(
        FusionProductSpace::new([SectorLeg::new([(minimum, 2)], false)]),
        FusionProductSpace::new([SectorLeg::new([(minimum, 2)], false)]),
    );
    let space = FusionTensorMapSpace::from_degeneracy_shapes_coupled(
        TensorMapSpace::<1, 1>::from_dims([2], [2]).unwrap(),
        homspace,
        &rule,
        [vec![2, 2]],
    )
    .unwrap();
    let tensor = TensorMap::from_vec_with_fusion_space(vec![4.0, 1.0, 1.0, 3.0], space).unwrap();
    let mut dense = tenet_dense::DefaultDenseExecutor::new();

    let result = eigh_full(&mut dense, &bound_tensor_ref!(Arc::new(rule), &tensor)).unwrap();

    assert_eq!(result.v.tensor().space().dims(), &[2, 2]);
    assert_eq!(result.d.tensor().space().dims(), &[2, 2]);
    let values = &result.eigenvalues[0].values;
    assert_eq!(values.len(), 2);
    for (col, &value) in values.iter().enumerate() {
        for row in 0..2 {
            let lhs = (0..2)
                .map(|index| tensor.data()[row + 2 * index] * result.v.data()[index + 2 * col])
                .sum::<f64>();
            let rhs = result.v.data()[row + 2 * col] * value;
            assert!((lhs - rhs).abs() < 1e-10);
        }
    }
}

#[cfg(target_pointer_width = "64")]
#[test]
fn compact_factorizations_do_not_relabel_product_lowest_u1_sectors() {
    // What: product-sector factors inherit the no-relabel contract for compact SVD, QR, LQ, and full EIGH.
    let rule = product_fusion_rule(FermionParityFusionRule, U1FusionRule);
    let minimum = rule
        .try_encode_sector(SectorId::new(1), U1Irrep::new(i32::MIN + 1).sector_id())
        .unwrap();
    let matrix = |rows: usize, cols: usize, data: Vec<f64>| {
        let homspace = FusionTreeHomSpace::new(
            FusionProductSpace::new([SectorLeg::new([(minimum, rows)], false)]),
            FusionProductSpace::new([SectorLeg::new([(minimum, cols)], false)]),
        );
        let space = FusionTensorMapSpace::from_degeneracy_shapes_coupled(
            TensorMapSpace::<1, 1>::from_dims([rows], [cols]).unwrap(),
            homspace,
            &rule,
            [vec![rows, cols]],
        )
        .unwrap();
        TensorMap::from_vec_with_fusion_space(data, space).unwrap()
    };
    let rectangular = matrix(3, 2, vec![-2.0, 5.0, 1.0, 4.0, -3.0, 2.0]);
    let mut dense = tenet_dense::DefaultDenseExecutor::new();

    let svd = svd_compact(
        &mut dense,
        &bound_tensor_ref!(Arc::new(rule.clone()), &rectangular),
    )
    .unwrap();
    assert_eq!(svd.u.tensor().space().dims(), &[3, 2]);
    assert_eq!(svd.s.tensor().space().dims(), &[2, 2]);
    assert_eq!(svd.vh.tensor().space().dims(), &[2, 2]);

    let (q, r) = qr_compact(
        &mut dense,
        &bound_tensor_ref!(Arc::new(rule.clone()), &rectangular),
    )
    .unwrap();
    assert_eq!(q.tensor().space().dims(), &[3, 2]);
    assert_eq!(r.tensor().space().dims(), &[2, 2]);
    assert_matrix_product(rectangular.data(), 3, 2, 2, q.data(), r.data());

    let (l, q) = lq_compact(
        &mut dense,
        &bound_tensor_ref!(Arc::new(rule.clone()), &rectangular),
    )
    .unwrap();
    assert_eq!(l.tensor().space().dims(), &[3, 2]);
    assert_eq!(q.tensor().space().dims(), &[2, 2]);
    assert_matrix_product(rectangular.data(), 3, 2, 2, l.data(), q.data());

    let hermitian = matrix(2, 2, vec![4.0, 1.0, 1.0, 3.0]);
    let result = eigh_full(&mut dense, &bound_tensor_ref!(Arc::new(rule), &hermitian)).unwrap();
    assert_eq!(result.v.tensor().space().dims(), &[2, 2]);
    assert_eq!(result.d.tensor().space().dims(), &[2, 2]);
    let values = &result.eigenvalues[0].values;
    assert_eq!(values.len(), 2);
    for (col, &value) in values.iter().enumerate() {
        for row in 0..2 {
            let lhs = (0..2)
                .map(|index| hermitian.data()[row + 2 * index] * result.v.data()[index + 2 * col])
                .sum::<f64>();
            let rhs = result.v.data()[row + 2 * col] * value;
            assert!((lhs - rhs).abs() < 1e-10);
        }
    }
}

#[test]
fn svd_rejects_a_different_provider_before_dense_execution() {
    let tensor = hermitian_test_tensor(&Z2FusionRule, &[SectorId::new(0), SectorId::new(1)]);

    let _backend = RejectExecutorCalls;
    let error = match BoundTensorMap::try_new(Arc::new(U1FusionRule), tensor) {
        Ok(_) => panic!("mismatched provider must not produce an authority"),
        Err(error) => error,
    };

    assert!(matches!(
        error,
        OperationError::Core(CoreError::FusionRuleMismatch { .. })
    ));
}

#[test]
fn svd_full_rejects_a_different_provider_before_dense_execution() {
    let tensor = tsvd_test_tensor(&Z2FusionRule, &[SectorId::new(0), SectorId::new(1)]);

    let _backend = RejectExecutorCalls;
    let error = match BoundTensorMap::try_new(Arc::new(U1FusionRule), tensor) {
        Ok(_) => panic!("mismatched provider must not produce an authority"),
        Err(error) => error,
    };

    assert!(matches!(
        error,
        OperationError::Core(CoreError::FusionRuleMismatch { .. })
    ));
}

#[test]
fn public_svd_authority_rejects_same_type_with_different_identity_and_qdim() {
    // What: provider provenance, not the Rust type or sector ids, owns qdim.
    let source_rule = IdentityQdimRule::new(1.0);
    let other_rule = IdentityQdimRule::new((1.0 + 5.0_f64.sqrt()) / 2.0);
    let tensor = tsvd_test_tensor(&source_rule, &[SectorId::new(0)]);

    let _backend = RejectExecutorCalls;
    let error = match BoundTensorMap::try_new(Arc::new(other_rule), tensor) {
        Ok(_) => panic!("different provider identity must not produce an authority"),
        Err(error) => error,
    };

    assert!(matches!(
        error,
        OperationError::Core(CoreError::FusionRuleMismatch { .. })
    ));
}

#[test]
fn svd_input_rejects_short_storage_before_dense_execution() {
    let tensor = tsvd_test_tensor(&Z2FusionRule, &[SectorId::new(0), SectorId::new(1)]);
    let bound = BoundDynamicFusionMapSpace::bind_multiplicity_free(
        dyn_space_of(&tensor).unwrap(),
        Arc::new(Z2FusionRule),
    )
    .unwrap();
    let short = &tensor.data()[..tensor.data().len() - 1];

    let error = match BoundDynamicTensorRef::try_new(&bound, short) {
        Ok(_) => panic!("short storage must be rejected"),
        Err(error) => error,
    };

    assert!(matches!(
        error,
        OperationError::Core(CoreError::DimensionMismatch { .. })
    ));
}

#[test]
fn pinv_rejects_invalid_rcond_before_dense_execution() {
    // What: invalid cutoff policy is rejected without entering the factorization backend.
    let rule = Z2FusionRule;
    let tensor = hermitian_test_tensor(&rule, &[SectorId::new(0), SectorId::new(1)]);
    let input = bound_tensor(Arc::new(rule), &tensor);
    for rcond in [-1.0, f64::NAN, f64::INFINITY] {
        let mut dense = RejectExecutorCalls;
        let mut context = default_context();
        let error = pinv(&mut dense, &mut context, &input.as_ref(), rcond).unwrap_err();
        assert!(matches!(error, OperationError::InvalidArgument { .. }));
    }
}

#[test]
fn spectral_outputs_retain_the_exact_input_provider_allocation() {
    // What: scalar promotion and spectral recomposition preserve provider authority by Arc identity.
    let rule = Z2FusionRule;
    let provider = Arc::new(rule);
    let hermitian = hermitian_test_tensor(&rule, &[SectorId::new(0), SectorId::new(1)]);
    let hermitian_space = BoundDynamicFusionMapSpace::bind_multiplicity_free(
        dyn_space_of(&hermitian).unwrap(),
        Arc::clone(&provider),
    )
    .unwrap();
    let hermitian_input =
        BoundDynamicTensorRef::try_new(&hermitian_space, hermitian.data()).unwrap();
    let general = tsvd_test_tensor(&rule, &[SectorId::new(0), SectorId::new(1)]);
    let general_space = BoundDynamicFusionMapSpace::bind_multiplicity_free(
        dyn_space_of(&general).unwrap(),
        Arc::clone(&provider),
    )
    .unwrap();
    let general_input = BoundDynamicTensorRef::try_new(&general_space, general.data()).unwrap();
    let mut dense = tenet_dense::DefaultDenseExecutor::new();

    let eigh = eigh_full_dyn(&mut dense, &hermitian_input).unwrap();
    assert!(Arc::ptr_eq(&provider, eigh.v().space().provider_arc()));

    let eig = eig_full_dyn(&mut dense, &general_input).unwrap();
    assert!(Arc::ptr_eq(&provider, eig.v().space().provider_arc()));

    let mut context = default_context();
    let exponential = exp_dyn(&mut dense, &mut context, &hermitian_input).unwrap();
    assert!(Arc::ptr_eq(&provider, exponential.space().provider_arc()));
}

#[test]
fn typed_svd_borrows_input_authority_and_retains_its_exact_allocation() {
    // What: borrowed typed input creates no replacement authority, and every SVD factor inherits it.
    let rule = Z2FusionRule;
    let provider = Arc::new(rule);
    let tensor = tsvd_test_tensor(&rule, &[SectorId::new(0), SectorId::new(1)]);
    let input = bound_tensor(Arc::clone(&provider), &tensor);
    let first = input.as_ref();
    let second = input.as_ref();

    assert!(std::ptr::eq(first.space(), input.space()));
    assert!(std::ptr::eq(second.space(), input.space()));
    assert!(std::ptr::eq(first.tensor(), input.tensor()));
    assert!(std::ptr::eq(second.tensor(), input.tensor()));

    let mut dense = tenet_dense::DefaultDenseExecutor::new();
    let factors = svd_compact(&mut dense, &first).unwrap();
    assert!(Arc::ptr_eq(&provider, factors.u.space().provider_arc()));
    assert!(Arc::ptr_eq(&provider, factors.s.space().provider_arc()));
    assert!(Arc::ptr_eq(&provider, factors.vh.space().provider_arc()));
}

fn reconstruct_from_svd<R>(
    rule: &R,
    template: &TensorMap<f64, 2, 2>,
    svd: &SvdTrunc<R, f64, 2, 2>,
) -> TensorMap<f64, 2, 2>
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64> + TreeTransformRuleCacheKey<Key = RuleIdentity>,
{
    let mut scaled_vt = svd.vh.tensor().clone();
    scale_vt_rows_by_singular_values(&mut scaled_vt, &svd.singular_values);
    let mut reconstructed = TensorMap::<f64, 2, 2>::from_vec_with_fusion_space(
        vec![0.0; template.data().len()],
        template.fusion_space().unwrap().as_ref().clone(),
    )
    .unwrap();
    let mut context = TensorContractFusionExecutionContext::<f64, RuleIdentity>::default();
    context
        .tensorcontract_fusion_into(
            rule,
            &mut reconstructed,
            &svd.u,
            &scaled_vt,
            TensorContractSpec::new(&[2], &[0], OutputAxisOrder::from_axes(&[0, 1, 2, 3])),
            1.0,
            0.0,
        )
        .unwrap();
    reconstructed
}

#[test]
fn tsvd_truncdim_bounds_weighted_dimension_and_reports_error_su2() {
    let rule = SU2FusionRule;
    let sectors = [
        SU2Irrep::from_twice_spin(0).sector_id(),
        SU2Irrep::from_twice_spin(1).sector_id(),
    ];
    let tensor = tsvd_test_tensor(&rule, &sectors);
    let mut dense_executor = tenet_dense::DefaultDenseExecutor::new();

    let max_dim = 10usize;
    let svd = svd_trunc(
        &mut dense_executor,
        &bound_tensor_ref!(Arc::new(rule), &tensor),
        &Truncation::rank(max_dim),
    )
    .unwrap();
    let error = svd.error;

    let weighted_dim: f64 = svd
        .singular_values
        .iter()
        .map(|entry| rule.dim_scalar(entry.sector) * entry.values.len() as f64)
        .sum();
    assert!(
        weighted_dim <= max_dim as f64 + 1e-9,
        "weighted dimension {weighted_dim} exceeds bound {max_dim}"
    );
    assert!(error > 0.0, "this cut must discard weight");

    let reconstructed = reconstruct_from_svd(&rule, &tensor, &svd);
    let distance = weighted_norm_squared_of_difference(&rule, &tensor, &reconstructed).sqrt();
    assert!(
        (distance - error).abs() < 1e-8,
        "reconstruction distance {distance} != reported truncation error {error}"
    );
}

#[test]
fn tsvd_truncbelow_drops_exactly_the_small_values() {
    let rule = Z2FusionRule;
    let sectors = [SectorId::new(0), SectorId::new(1)];
    let tensor = tsvd_test_tensor(&rule, &sectors);
    let mut dense_executor = tenet_dense::DefaultDenseExecutor::new();

    let full = svd_trunc(
        &mut dense_executor,
        &bound_tensor_ref!(Arc::new(rule), &tensor),
        &Truncation::Full,
    )
    .unwrap();
    let threshold = {
        let mut all: Vec<f64> = full
            .singular_values
            .iter()
            .flat_map(|entry| entry.values.iter().copied())
            .collect();
        all.sort_by(|a, b| b.partial_cmp(a).unwrap());
        (all[all.len() / 2] + all[all.len() / 2 - 1]) / 2.0
    };

    let truncation = Truncation::absolute_cutoff(threshold).unwrap();
    let svd = svd_trunc(
        &mut dense_executor,
        &bound_tensor_ref!(Arc::new(rule), &tensor),
        &truncation,
    )
    .unwrap();
    let error = svd.error;

    for entry in &svd.singular_values {
        assert!(entry.values.iter().all(|&value| value >= threshold));
    }
    let kept: usize = svd
        .singular_values
        .iter()
        .map(|entry| entry.values.len())
        .sum();
    let full_count: usize = full
        .singular_values
        .iter()
        .map(|entry| entry.values.len())
        .sum();
    assert!(kept < full_count);
    assert!(error > 0.0);

    let reconstructed = reconstruct_from_svd(&rule, &tensor, &svd);
    let distance = weighted_norm_squared_of_difference(&rule, &tensor, &reconstructed).sqrt();
    assert!((distance - error).abs() < 1e-8);
}

#[test]
fn tsvd_truncerr_respects_relative_tolerance() {
    let rule = U1FusionRule;
    let sectors = [
        U1Irrep::new(-1).sector_id(),
        U1Irrep::new(0).sector_id(),
        U1Irrep::new(1).sector_id(),
    ];
    let tensor = tsvd_test_tensor(&rule, &sectors);
    let mut dense_executor = tenet_dense::DefaultDenseExecutor::new();

    let tolerance = 0.2;
    let truncation = Truncation::relative_error(tolerance).unwrap();
    let svd = svd_trunc(
        &mut dense_executor,
        &bound_tensor_ref!(Arc::new(rule), &tensor),
        &truncation,
    )
    .unwrap();
    let error = svd.error;

    let norm = weighted_norm_squared_of_difference(
        &rule,
        &tensor,
        &TensorMap::<f64, 2, 2>::from_vec_with_fusion_space(
            vec![0.0; tensor.data().len()],
            tensor.fusion_space().unwrap().as_ref().clone(),
        )
        .unwrap(),
    )
    .sqrt();
    assert!(
        error <= tolerance * norm + 1e-9,
        "truncation error {error} exceeds tolerance {tolerance} * norm {norm}"
    );
    assert!(error > 0.0, "tolerance 0.2 must discard something here");

    let reconstructed = reconstruct_from_svd(&rule, &tensor, &svd);
    let distance = weighted_norm_squared_of_difference(&rule, &tensor, &reconstructed).sqrt();
    assert!((distance - error).abs() < 1e-8);
}

#[test]
fn leftorth_fusion_reconstructs_z2_and_su2_tensors() {
    for (rule_case, sectors) in [
        (0usize, vec![SectorId::new(0), SectorId::new(1)]),
        (
            1usize,
            vec![
                SU2Irrep::from_twice_spin(0).sector_id(),
                SU2Irrep::from_twice_spin(1).sector_id(),
            ],
        ),
    ] {
        if rule_case == 0 {
            let rule = Z2FusionRule;
            let tensor = tsvd_test_tensor(&rule, &sectors);
            let mut dense_executor = tenet_dense::DefaultDenseExecutor::new();
            let (q, r) = qr_compact(
                &mut dense_executor,
                &bound_tensor_ref!(Arc::new(rule), &tensor),
            )
            .unwrap();
            let reconstructed = contract_pair(&rule, &tensor, &q, &r);
            assert_svd_blocks_match(&tensor, &reconstructed);
        } else {
            let rule = SU2FusionRule;
            let tensor = tsvd_test_tensor(&rule, &sectors);
            let mut dense_executor = tenet_dense::DefaultDenseExecutor::new();
            let (q, r) = qr_compact(
                &mut dense_executor,
                &bound_tensor_ref!(Arc::new(rule), &tensor),
            )
            .unwrap();
            let reconstructed = contract_pair(&rule, &tensor, &q, &r);
            assert_svd_blocks_match(&tensor, &reconstructed);
        }
    }
}

fn assert_compact_qr_reconstructs_rule<R>(rule: &R, sectors: &[SectorId])
where
    R: Clone + MultiplicityFreeRigidSymbols<Scalar = f64> + TreeTransformRuleCacheKey,
{
    let tensor = tsvd_test_tensor(rule, sectors);
    let mut dense = tenet_dense::DefaultDenseExecutor::new();
    let (q, r) = qr_compact(
        &mut dense,
        &bound_tensor_ref!(Arc::new((*rule).clone()), &tensor),
    )
    .unwrap();
    assert_factor_layout_matches_legacy_shapes(q.space());
    assert_factor_layout_matches_legacy_shapes(r.space());
    let reconstructed = contract_pair(rule, &tensor, &q, &r);
    assert_svd_blocks_match(&tensor, &reconstructed);
}

#[test]
fn compact_qr_reconstructs_u1_fermion_parity_and_product_rules() {
    // What: direct Q/R routes preserve abelian, fermionic, and encoded product sector labels.
    assert_compact_qr_reconstructs_rule(
        &U1FusionRule,
        &[
            U1Irrep::new(-1).sector_id(),
            U1Irrep::new(0).sector_id(),
            U1Irrep::new(1).sector_id(),
        ],
    );
    assert_compact_qr_reconstructs_rule(
        &FermionParityFusionRule,
        &[SectorId::new(0), SectorId::new(1)],
    );
    let product = product_fusion_rule(FermionParityFusionRule, U1FusionRule);
    let product_sectors = [
        product.encode_sector(SectorId::new(0), U1Irrep::new(0).sector_id()),
        product.encode_sector(SectorId::new(1), U1Irrep::new(1).sector_id()),
    ];
    assert_compact_qr_reconstructs_rule(&product, &product_sectors);

    let nested = product_fusion_rule(product, SU2FusionRule);
    let nested_sectors = [
        nested.encode_sector(product_sectors[0], SU2Irrep::from_twice_spin(0).sector_id()),
        nested.encode_sector(product_sectors[1], SU2Irrep::from_twice_spin(1).sector_id()),
    ];
    crate::factorize::reset_compact_qr_copy_probe();
    assert_compact_qr_reconstructs_rule(&nested, &nested_sectors);
    let probe = crate::factorize::compact_qr_copy_probe();
    assert_eq!(probe.input_pack_bytes, 0);
    assert_eq!(probe.output_scatter_bytes, 0);
}

fn assert_compact_lq_reconstructs_rule<R>(rule: &R, sectors: &[SectorId])
where
    R: Clone + MultiplicityFreeRigidSymbols<Scalar = f64> + TreeTransformRuleCacheKey,
{
    let tensor = tsvd_test_tensor(rule, sectors);
    let mut dense = tenet_dense::DefaultDenseExecutor::new();
    let (left, right) = lq_compact(
        &mut dense,
        &bound_tensor_ref!(Arc::new((*rule).clone()), &tensor),
    )
    .unwrap();
    assert_factor_layout_matches_legacy_shapes(left.space());
    assert_factor_layout_matches_legacy_shapes(right.space());
    let reconstructed = contract_pair(rule, &tensor, &left, &right);
    assert_svd_blocks_match(&tensor, &reconstructed);
}

#[test]
fn compact_lq_reconstructs_u1_fermion_parity_and_product_rules() {
    // What: direct LQ routes preserve non-Abelian, abelian, fermionic, and nested product sector labels.
    assert_compact_lq_reconstructs_rule(
        &U1FusionRule,
        &[
            U1Irrep::new(-1).sector_id(),
            U1Irrep::new(0).sector_id(),
            U1Irrep::new(1).sector_id(),
        ],
    );
    assert_compact_lq_reconstructs_rule(
        &SU2FusionRule,
        &[
            SU2Irrep::from_twice_spin(0).sector_id(),
            SU2Irrep::from_twice_spin(1).sector_id(),
        ],
    );
    assert_compact_lq_reconstructs_rule(
        &FermionParityFusionRule,
        &[SectorId::new(0), SectorId::new(1)],
    );
    let product = product_fusion_rule(FermionParityFusionRule, U1FusionRule);
    let product_sectors = [
        product.encode_sector(SectorId::new(0), U1Irrep::new(0).sector_id()),
        product.encode_sector(SectorId::new(1), U1Irrep::new(1).sector_id()),
    ];
    assert_compact_lq_reconstructs_rule(&product, &product_sectors);

    let nested = product_fusion_rule(product, SU2FusionRule);
    let nested_sectors = [
        nested.encode_sector(product_sectors[0], SU2Irrep::from_twice_spin(0).sector_id()),
        nested.encode_sector(product_sectors[1], SU2Irrep::from_twice_spin(1).sector_id()),
    ];
    crate::factorize::reset_compact_lq_copy_probe();
    assert_compact_lq_reconstructs_rule(&nested, &nested_sectors);
    let probe = crate::factorize::compact_lq_copy_probe();
    assert_eq!(probe.input_pack_bytes, 0);
    assert_eq!(probe.output_scatter_bytes, 0);
    assert_eq!(probe.scratch_buffer_count, 1);
}

#[test]
fn rightorth_fusion_reconstructs_z2_and_su2_tensors() {
    {
        let rule = Z2FusionRule;
        let tensor = tsvd_test_tensor(&rule, &[SectorId::new(0), SectorId::new(1)]);
        let mut dense_executor = tenet_dense::DefaultDenseExecutor::new();
        let (l, q) = lq_compact(
            &mut dense_executor,
            &bound_tensor_ref!(Arc::new(rule), &tensor),
        )
        .unwrap();
        let reconstructed = contract_pair(&rule, &tensor, &l, &q);
        assert_svd_blocks_match(&tensor, &reconstructed);
    }
    {
        let rule = SU2FusionRule;
        let tensor = tsvd_test_tensor(
            &rule,
            &[
                SU2Irrep::from_twice_spin(0).sector_id(),
                SU2Irrep::from_twice_spin(1).sector_id(),
            ],
        );
        let mut dense_executor = tenet_dense::DefaultDenseExecutor::new();
        let (l, q) = lq_compact(
            &mut dense_executor,
            &bound_tensor_ref!(Arc::new(rule), &tensor),
        )
        .unwrap();
        let reconstructed = contract_pair(&rule, &tensor, &l, &q);
        assert_svd_blocks_match(&tensor, &reconstructed);
    }
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

#[test]
fn tsvd_singular_tensor_composes_u_s_vt() {
    let rule = SU2FusionRule;
    let tensor = tsvd_test_tensor(
        &rule,
        &[
            SU2Irrep::from_twice_spin(0).sector_id(),
            SU2Irrep::from_twice_spin(1).sector_id(),
        ],
    );
    let mut dense_executor = tenet_dense::DefaultDenseExecutor::new();
    let svd = svd_trunc(
        &mut dense_executor,
        &bound_tensor_ref!(Arc::new(rule), &tensor),
        &Truncation::Full,
    )
    .unwrap();
    let s_tensor = svd.s.clone();

    let mut context = TensorContractFusionExecutionContext::<f64, RuleIdentity>::default();
    let mut u_s = TensorMap::<f64, 2, 1>::from_vec_with_fusion_space(
        vec![0.0; svd.u.data().len()],
        svd.u.fusion_space().unwrap().as_ref().clone(),
    )
    .unwrap();
    context
        .tensorcontract_fusion_into(
            &rule,
            &mut u_s,
            &svd.u,
            &s_tensor,
            TensorContractSpec::new(&[2], &[0], OutputAxisOrder::from_axes(&[0, 1, 2])),
            1.0,
            0.0,
        )
        .unwrap();

    let mut reconstructed = TensorMap::<f64, 2, 2>::from_vec_with_fusion_space(
        vec![0.0; tensor.data().len()],
        tensor.fusion_space().unwrap().as_ref().clone(),
    )
    .unwrap();
    context
        .tensorcontract_fusion_into(
            &rule,
            &mut reconstructed,
            &u_s,
            &svd.vh,
            TensorContractSpec::new(&[2], &[0], OutputAxisOrder::from_axes(&[0, 1, 2, 3])),
            1.0,
            0.0,
        )
        .unwrap();

    assert_svd_blocks_match(&tensor, &reconstructed);
}

#[test]
fn svd_trunc_is_svd_compact_plus_host_truncation() {
    let rule = SU2FusionRule;
    let tensor = tsvd_test_tensor(
        &rule,
        &[
            SU2Irrep::from_twice_spin(0).sector_id(),
            SU2Irrep::from_twice_spin(1).sector_id(),
        ],
    );
    let truncation = Truncation::rank(9).and(Truncation::absolute_cutoff(1e-12).unwrap());

    let mut dense_executor = tenet_dense::DefaultDenseExecutor::new();
    let composed = {
        let compact = svd_compact(
            &mut dense_executor,
            &bound_tensor_ref!(Arc::new(rule), &tensor),
        )
        .unwrap();
        truncate_svd(compact, &truncation).unwrap()
    };
    let direct = svd_trunc(
        &mut dense_executor,
        &bound_tensor_ref!(Arc::new(rule), &tensor),
        &truncation,
    )
    .unwrap();

    assert_eq!(composed.singular_values, direct.singular_values);
    assert!((composed.error - direct.error).abs() < 1e-15);
    assert_eq!(composed.u.data(), direct.u.data());
    assert_eq!(composed.s.data(), direct.s.data());
    assert_eq!(composed.vh.data(), direct.vh.data());
}

#[test]
fn truncate_svd_full_reuses_the_prebuilt_diagonal_factor() {
    // What: composed compact-then-full truncation moves its existing S without rebuilding it.
    let rule = Z2FusionRule;
    let tensor = tsvd_test_tensor(&rule, &[SectorId::new(0), SectorId::new(1)]);
    let mut dense = tenet_dense::DefaultDenseExecutor::new();
    let compact = svd_compact(&mut dense, &bound_tensor_ref!(Arc::new(rule), &tensor)).unwrap();

    crate::factorize::reset_diagonal_bond_build_probe();
    let result = truncate_svd(compact, &Truncation::Full).unwrap();

    assert_eq!(result.error, 0.0);
    assert_eq!(
        crate::factorize::diagonal_bond_build_probe(),
        crate::factorize::DiagonalBondBuildProbe::default()
    );
}

#[test]
fn svd_trunc_builds_only_the_returned_diagonal_factor() {
    // What: partial and full truncation each materialize S once at the final returned rank.
    let rule = SU2FusionRule;
    let tensor = tsvd_test_tensor(
        &rule,
        &[
            SU2Irrep::from_twice_spin(0).sector_id(),
            SU2Irrep::from_twice_spin(1).sector_id(),
        ],
    );
    let input = bound_tensor(Arc::new(rule), &tensor);
    let mut dense = tenet_dense::DefaultDenseExecutor::new();
    let full_rank = svd_vals_dyn(&mut dense, &input.as_ref().dynamic())
        .unwrap()
        .iter()
        .map(|entry| entry.values.len())
        .sum::<usize>();

    crate::factorize::reset_diagonal_bond_build_probe();
    let partial =
        svd_trunc_dyn(&mut dense, &input.as_ref().dynamic(), &Truncation::rank(5)).unwrap();
    let partial_rank = partial
        .singular_values()
        .iter()
        .map(|entry| entry.values.len())
        .sum();
    for factor in [partial.u(), partial.s(), partial.vh()] {
        assert_factor_layout_matches_legacy_shapes(factor.space());
    }
    assert!(partial_rank < full_rank);
    assert!(partial.error() > 0.0);
    assert_eq!(
        crate::factorize::diagonal_bond_build_probe(),
        crate::factorize::DiagonalBondBuildProbe {
            calls: 1,
            values: partial_rank,
        }
    );

    crate::factorize::reset_diagonal_bond_build_probe();
    let full = svd_trunc_dyn(&mut dense, &input.as_ref().dynamic(), &Truncation::Full).unwrap();
    let returned_full_rank = full
        .singular_values()
        .iter()
        .map(|entry| entry.values.len())
        .sum::<usize>();
    for factor in [full.u(), full.s(), full.vh()] {
        assert_factor_layout_matches_legacy_shapes(factor.space());
    }
    assert_eq!(returned_full_rank, full_rank);
    assert_eq!(full.error(), 0.0);
    assert_eq!(
        crate::factorize::diagonal_bond_build_probe(),
        crate::factorize::DiagonalBondBuildProbe {
            calls: 1,
            values: full_rank,
        }
    );
}

#[test]
fn svd_trunc_factor_only_core_skips_dense_s_and_dense_contract_wraps_once() {
    // What: the factor-only entry returns the same truncated U/Vh, spectrum,
    // and error without building S; the existing dense-S API wraps it once.
    let rule = SU2FusionRule;
    let tensor = tsvd_test_tensor(
        &rule,
        &[
            SU2Irrep::from_twice_spin(0).sector_id(),
            SU2Irrep::from_twice_spin(1).sector_id(),
        ],
    );
    let input = bound_tensor(Arc::new(rule), &tensor);
    let truncation = Truncation::rank(5);
    let mut dense = tenet_dense::DefaultDenseExecutor::new();

    crate::factorize::reset_diagonal_bond_build_probe();
    let (u, vh, singular_values, error) =
        svd_trunc_factors_dyn(&mut dense, &input.as_ref().dynamic(), &truncation).unwrap();
    assert_eq!(
        crate::factorize::diagonal_bond_build_probe(),
        crate::factorize::DiagonalBondBuildProbe::default()
    );

    crate::factorize::reset_diagonal_bond_build_probe();
    let wrapped = svd_trunc_dyn(&mut dense, &input.as_ref().dynamic(), &truncation).unwrap();
    assert_eq!(u.data(), wrapped.u().data());
    assert_eq!(vh.data(), wrapped.vh().data());
    assert_eq!(singular_values, wrapped.singular_values());
    assert_eq!(error, wrapped.error());
    assert_eq!(
        crate::factorize::diagonal_bond_build_probe(),
        crate::factorize::DiagonalBondBuildProbe {
            calls: 1,
            values: singular_values.iter().map(|entry| entry.values.len()).sum(),
        }
    );
}

#[test]
fn svd_trunc_zero_rank_returns_empty_factors_and_the_full_error() {
    // What: an all-discard decision publishes rank-zero factors and reports the entire weighted norm.
    let rule = SU2FusionRule;
    let provider = Arc::new(rule);
    let tensor = tsvd_test_tensor(
        &rule,
        &[
            SU2Irrep::from_twice_spin(0).sector_id(),
            SU2Irrep::from_twice_spin(1).sector_id(),
        ],
    );
    let input = bound_tensor(Arc::clone(&provider), &tensor);
    let mut dense = tenet_dense::DefaultDenseExecutor::new();
    let full_spectrum = svd_vals_dyn(&mut dense, &input.as_ref().dynamic()).unwrap();
    let expected_error = full_spectrum
        .iter()
        .map(|entry| {
            rule.dim_scalar(entry.sector)
                * entry.values.iter().map(|value| value * value).sum::<f64>()
        })
        .sum::<f64>()
        .sqrt();

    crate::factorize::reset_diagonal_bond_build_probe();
    let result =
        svd_trunc_dyn(&mut dense, &input.as_ref().dynamic(), &Truncation::rank(0)).unwrap();

    assert!(result.singular_values().is_empty());
    assert!(result.u().data().is_empty());
    assert!(result.s().data().is_empty());
    assert!(result.vh().data().is_empty());
    for factor in [result.u(), result.s(), result.vh()] {
        assert_eq!(factor.space().space().structure().block_count(), 0);
        assert_factor_layout_matches_legacy_shapes(factor.space());
    }
    assert!((result.error() - expected_error).abs() < 1e-12);
    assert_eq!(
        crate::factorize::diagonal_bond_build_probe(),
        crate::factorize::DiagonalBondBuildProbe {
            calls: 1,
            values: 0,
        }
    );
    for factor in [result.u(), result.s(), result.vh()] {
        assert!(Arc::ptr_eq(factor.space().provider_arc(), &provider));
    }
}

#[test]
fn svd_trunc_dense_failure_preserves_input_and_builds_no_diagonal_factor() {
    // What: a failed dense SVD leaves borrowed input unchanged and cannot publish or build factors.
    let rule = Z2FusionRule;
    let tensor = tsvd_test_tensor(&rule, &[SectorId::new(0), SectorId::new(1)]);
    let before = tensor.data().to_vec();
    let mut dense = FailAfterObservingSvdInput::default();

    crate::factorize::reset_diagonal_bond_build_probe();
    let result = svd_trunc(
        &mut dense,
        &bound_tensor_ref!(Arc::new(rule), &tensor),
        &Truncation::rank(1),
    );

    assert!(matches!(result, Err(OperationError::Dense(_))));
    assert_eq!(tensor.data(), before);
    assert_eq!(
        crate::factorize::diagonal_bond_build_probe(),
        crate::factorize::DiagonalBondBuildProbe::default()
    );
}

fn assert_zero_axis_svd_trunc(rows: usize, cols: usize) {
    let rule = Z2FusionRule;
    let tensor = rectangular_svd_tensor(rows, cols);
    let input = bound_tensor(Arc::new(rule), &tensor);
    let mut dense = tenet_dense::DefaultDenseExecutor::new();

    for truncation in [Truncation::Full, Truncation::rank(1)] {
        crate::factorize::reset_diagonal_bond_build_probe();
        let result = svd_trunc_dyn(&mut dense, &input.as_ref().dynamic(), &truncation).unwrap();
        assert_eq!(
            result
                .singular_values()
                .iter()
                .map(|entry| entry.values.len())
                .sum::<usize>(),
            0
        );
        assert!(result.u().data().is_empty());
        assert!(result.s().data().is_empty());
        assert!(result.vh().data().is_empty());
        for factor in [result.u(), result.s(), result.vh()] {
            assert_eq!(factor.space().space().structure().block_count(), 0);
            assert_factor_layout_matches_legacy_shapes(factor.space());
        }
        assert_eq!(result.error(), 0.0);
        assert_eq!(
            crate::factorize::diagonal_bond_build_probe(),
            crate::factorize::DiagonalBondBuildProbe {
                calls: 1,
                values: 0,
            }
        );
    }
}

#[test]
fn svd_trunc_zero_only_input_normalizes_to_an_empty_factorization_result() {
    // What: full and partial truncation expose no phantom sector when either
    // side of the zero-only input is absent.
    assert_zero_axis_svd_trunc(0, 3);
    assert_zero_axis_svd_trunc(3, 0);
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

#[test]
fn eigh_full_satisfies_the_eigen_equation() {
    let rule = SU2FusionRule;
    let tensor = hermitian_test_tensor(
        &rule,
        &[
            SU2Irrep::from_twice_spin(0).sector_id(),
            SU2Irrep::from_twice_spin(1).sector_id(),
        ],
    );
    let mut dense_executor = tenet_dense::DefaultDenseExecutor::new();
    let eigh = eigh_full(
        &mut dense_executor,
        &bound_tensor_ref!(Arc::new(rule), &tensor),
    )
    .unwrap();

    for entry in &eigh.eigenvalues {
        for pair in entry.values.windows(2) {
            assert!(
                pair[0].abs() >= pair[1].abs() - 1e-12,
                "eigenvalues must be stored descending by magnitude"
            );
        }
    }
    assert_eigen_equation(&rule, &tensor, &eigh.v, &eigh.d);
}

fn assert_eigh_reconstructs_rule<R>(rule: &R, sectors: &[SectorId])
where
    R: Clone + MultiplicityFreeRigidSymbols<Scalar = f64> + TreeTransformRuleCacheKey,
{
    let tensor = hermitian_test_tensor(rule, sectors);
    let mut dense = tenet_dense::DefaultDenseExecutor::new();
    let eigh = eigh_full(
        &mut dense,
        &bound_tensor_ref!(Arc::new((*rule).clone()), &tensor),
    )
    .unwrap();
    assert_factor_layout_matches_legacy_shapes(eigh.v.space());
    assert_factor_layout_matches_legacy_shapes(eigh.d.space());
    for entry in &eigh.eigenvalues {
        for pair in entry.values.windows(2) {
            assert!(pair[0].abs() >= pair[1].abs() - 1e-12);
        }
    }
    assert_eigen_equation(rule, &tensor, &eigh.v, &eigh.d);
}

#[test]
fn eigh_reconstructs_u1_fermion_parity_and_product_rules() {
    // What: direct EIGH preserves abelian, fermionic, product, and nested sector identities.
    assert_eigh_reconstructs_rule(
        &U1FusionRule,
        &[
            U1Irrep::new(-1).sector_id(),
            U1Irrep::new(0).sector_id(),
            U1Irrep::new(1).sector_id(),
        ],
    );
    assert_eigh_reconstructs_rule(
        &FermionParityFusionRule,
        &[SectorId::new(0), SectorId::new(1)],
    );
    let product = product_fusion_rule(FermionParityFusionRule, U1FusionRule);
    let product_sectors = [
        product.encode_sector(SectorId::new(0), U1Irrep::new(0).sector_id()),
        product.encode_sector(SectorId::new(1), U1Irrep::new(1).sector_id()),
    ];
    assert_eigh_reconstructs_rule(&product, &product_sectors);

    let nested = product_fusion_rule(product, SU2FusionRule);
    let nested_sectors = [
        nested.encode_sector(product_sectors[0], SU2Irrep::from_twice_spin(0).sector_id()),
        nested.encode_sector(product_sectors[1], SU2Irrep::from_twice_spin(1).sector_id()),
    ];
    crate::factorize::reset_eigh_copy_probe();
    assert_eigh_reconstructs_rule(&nested, &nested_sectors);
    assert_eq!(
        crate::factorize::eigh_copy_probe(),
        crate::factorize::EighCopyProbe::default()
    );
}

#[test]
fn eigh_c64_reconstructs_multi_sector_hermitian_input_and_fixes_gauge() {
    use num_complex::Complex64;

    // What: complex direct vectors reconstruct every sector and use the canonical phase gauge.
    let rule = Z2FusionRule;
    let real = hermitian_test_tensor(&rule, &[SectorId::new(0), SectorId::new(1)]);
    let tensor = TensorMap::<Complex64, 2, 2>::from_vec_with_fusion_space(
        real.data()
            .iter()
            .map(|&value| Complex64::new(value, 0.0))
            .collect(),
        real.fusion_space().unwrap().as_ref().clone(),
    )
    .unwrap();
    let mut dense = tenet_dense::DefaultDenseExecutor::new();
    let eigh = eigh_full(&mut dense, &bound_tensor_ref!(Arc::new(rule), &tensor)).unwrap();
    let input_regions = tensor
        .structure()
        .coupled_sector_regions(2)
        .unwrap()
        .unwrap();
    let vector_regions = eigh
        .v
        .structure()
        .coupled_sector_regions(2)
        .unwrap()
        .unwrap();
    for input_region in input_regions.iter() {
        let sector = input_region.coupled();
        let vector_region = vector_regions
            .iter()
            .find(|region| region.coupled() == sector)
            .unwrap();
        let values = &eigh
            .eigenvalues
            .iter()
            .find(|entry| entry.sector == sector)
            .unwrap()
            .values;
        let n = input_region.rows();
        for bond in 0..n {
            let column = &eigh.v.data()[vector_region.range().start + bond * n
                ..vector_region.range().start + (bond + 1) * n];
            let pivot = column
                .iter()
                .max_by(|a, b| a.norm_sqr().partial_cmp(&b.norm_sqr()).unwrap())
                .unwrap();
            assert!(pivot.im.abs() < 1e-12);
            assert!(pivot.re >= 0.0);
        }
        for col in 0..n {
            for row in 0..n {
                let reconstructed = (0..n)
                    .map(|bond| {
                        eigh.v.data()[vector_region.range().start + row + n * bond]
                            * values[bond]
                            * eigh.v.data()[vector_region.range().start + col + n * bond].conj()
                    })
                    .sum::<Complex64>();
                let expected = tensor.data()[input_region.range().start + row + n * col];
                assert!((reconstructed - expected).norm() < 1e-9);
            }
        }
    }
}

#[test]
fn eigh_direct_regions_publish_vectors_by_factor_order_and_spectra_by_source_order() {
    // What: reversed source spans leave spectra in source traversal while V is
    // interpreted through the published factor's sector regions.
    fn check<D: crate::factorize::FactorScalar>(tensor: &TensorMap<D, 1, 1>) {
        let rule = Arc::new(Z2FusionRule);
        let bound = bound_tensor(Arc::clone(&rule), tensor);
        let plan = crate::factorize::compact_factor_plan_for_test(bound.space())
            .unwrap()
            .unwrap();
        assert!(crate::factorize::compact_factor_plan_routes_for_test(&plan)
            .iter()
            .any(|route| {
                let (source, left, _) = route.factor_regions_for_test();
                left.is_some_and(|left| left != source)
            }));

        let mut dense = tenet_dense::DefaultDenseExecutor::new();
        let eigh = eigh_full(&mut dense, &bound.as_ref()).unwrap();
        let source_regions = tensor
            .structure()
            .coupled_sector_regions(1)
            .unwrap()
            .unwrap();
        assert_eq!(
            eigh.eigenvalues
                .iter()
                .map(|entry| entry.sector)
                .collect::<Vec<_>>(),
            source_regions
                .iter()
                .map(|region| region.coupled())
                .collect::<Vec<_>>()
        );
        let vector_regions = eigh
            .v
            .structure()
            .coupled_sector_regions(1)
            .unwrap()
            .unwrap();
        for source in source_regions.iter() {
            let sector = source.coupled();
            let vector = vector_regions
                .iter()
                .find(|region| region.coupled() == sector)
                .unwrap();
            let values = &eigh
                .eigenvalues
                .iter()
                .find(|entry| entry.sector == sector)
                .unwrap()
                .values;
            let n = source.rows();
            for col in 0..n {
                for row in 0..n {
                    let reconstructed = (0..n)
                        .map(|bond| {
                            eigh.v.data()[vector.range().start + row + n * bond].widen_complex()
                                * values[bond]
                                * eigh.v.data()[vector.range().start + col + n * bond]
                                    .widen_complex()
                                    .conj()
                        })
                        .sum::<Complex64>();
                    let expected =
                        tensor.data()[source.range().start + row + n * col].widen_complex();
                    assert!(
                        (reconstructed - expected).norm() < 2e-5,
                        "sector {sector:?}, entry ({row}, {col})"
                    );
                }
            }
        }
    }

    let rule = Z2FusionRule;
    let source = mixed_rectangular_tensor((3, 3), (2, 2));
    let mut data = source.data().to_vec();
    for region in source
        .structure()
        .coupled_sector_regions(1)
        .unwrap()
        .unwrap()
        .iter()
    {
        let n = region.rows();
        for col in 0..n {
            for row in 0..n {
                data[region.range().start + row + n * col] = if row == col {
                    (row + 2) as f64
                } else {
                    (row + col + 1) as f64 * 0.25
                };
            }
        }
    }
    let real = reversed_complete_grid_copy(
        &rule,
        &TensorMap::<f64, 1, 1>::from_vec_with_fusion_space(
            data,
            source.fusion_space().unwrap().as_ref().clone(),
        )
        .unwrap(),
    );
    check(&real);
    check(
        &TensorMap::<Complex64, 1, 1>::from_vec_with_fusion_space(
            real.data()
                .iter()
                .map(|&value| Complex64::new(value, 0.0))
                .collect(),
            real.fusion_space().unwrap().as_ref().clone(),
        )
        .unwrap(),
    );
}

#[test]
fn eigh_trunc_truncates_by_magnitude_and_keeps_eigen_equation() {
    let rule = Z2FusionRule;
    let tensor = hermitian_test_tensor(&rule, &[SectorId::new(0), SectorId::new(1)]);
    let mut dense_executor = tenet_dense::DefaultDenseExecutor::new();

    let full = eigh_full(
        &mut dense_executor,
        &bound_tensor_ref!(Arc::new(rule), &tensor),
    )
    .unwrap();
    let full_count: usize = full
        .eigenvalues
        .iter()
        .map(|entry| entry.values.len())
        .sum();
    let max_dim = full_count / 2;
    let eigh = eigh_trunc(
        &mut dense_executor,
        &bound_tensor_ref!(Arc::new(rule), &tensor),
        &Truncation::rank(max_dim),
    )
    .unwrap();

    let kept: usize = eigh
        .eigenvalues
        .iter()
        .map(|entry| entry.values.len())
        .sum();
    assert!(kept <= max_dim);
    assert!(eigh.error > 0.0);
    // Truncated eigenvectors still satisfy t . V = V . D exactly.
    assert_eigen_equation(&rule, &tensor, &eigh.v, &eigh.d);
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

fn bound_factor_matrices<R>(
    factor: &BoundDynFactor<R, f64>,
) -> Vec<(SectorId, usize, usize, Vec<f64>)> {
    factor
        .space()
        .space()
        .structure()
        .coupled_sector_regions(factor.space().space().nout())
        .unwrap()
        .unwrap()
        .iter()
        .map(|region| {
            assert_eq!(region.range().len(), region.rows() * region.cols());
            (
                region.coupled(),
                region.rows(),
                region.cols(),
                factor.data()[region.range()].to_vec(),
            )
        })
        .collect()
}

fn assert_orthonormal_rows(matrices: &[(SectorId, usize, usize, Vec<f64>)]) {
    for (sector, rows, cols, matrix) in matrices {
        for upper in 0..*rows {
            for lower in 0..*rows {
                let dot = (0..*cols)
                    .map(|col| matrix[upper + rows * col] * matrix[lower + rows * col])
                    .sum::<f64>();
                let expected = if upper == lower { 1.0 } else { 0.0 };
                assert!(
                    (dot - expected).abs() < 1.0e-9,
                    "sector {sector:?}: row dot ({upper},{lower}) = {dot}"
                );
            }
        }
    }
}

fn assert_nonnegative_diagonal(matrices: &[(SectorId, usize, usize, Vec<f64>)]) {
    for (sector, rows, cols, matrix) in matrices {
        for index in 0..(*rows).min(*cols) {
            assert!(
                matrix[index + rows * index] >= 0.0,
                "sector {sector:?}: diagonal {index} is negative"
            );
        }
    }
}

fn assert_full_qr_observation(
    observation: &FullQrObservation,
    input: &[Complex64],
    rows: usize,
    cols: usize,
) {
    let dense_cols = if rows <= cols { cols } else { cols + rows };
    assert_eq!(observation.input_shape, [rows, dense_cols]);
    assert_eq!(observation.q_shape, [rows, rows]);
    assert_eq!(observation.r_shape, [rows, dense_cols]);
    let mut expected = vec![Complex64::new(0.0, 0.0); rows * dense_cols];
    expected[..rows * cols].copy_from_slice(input);
    if rows > cols {
        for row in 0..rows {
            expected[rows * cols + row * rows + row] = Complex64::new(1.0, 0.0);
        }
    }
    assert_eq!(observation.values, expected);
}

fn adjoint_complex(input: &[Complex64], rows: usize, cols: usize) -> Vec<Complex64> {
    let mut output = vec![Complex64::new(0.0, 0.0); input.len()];
    for col in 0..cols {
        for row in 0..rows {
            output[col + cols * row] = input[row + rows * col].conj();
        }
    }
    output
}

#[test]
fn full_qr_and_lq_use_original_input_only_when_economy_q_is_full() {
    let rule = Z2FusionRule;
    let tensor = mixed_rectangular_tensor((2, 4), (3, 1));
    let matrices = dense_sector_matrices(1, &tensor);
    let input = bound_tensor(Arc::new(rule), &tensor);
    let input_ref = input.as_ref();
    let input = input_ref.dynamic();

    let mut qr_dense = FullQrInputSpy::default();
    let (q, r) = qr_full_dyn(&mut qr_dense, &input).unwrap();
    assert_eq!(qr_dense.observations.len(), matrices.len());
    for (observation, (_, rows, cols, matrix)) in qr_dense.observations.iter().zip(matrices.iter())
    {
        let matrix = matrix
            .iter()
            .map(|&value| Complex64::new(value, 0.0))
            .collect::<Vec<_>>();
        assert_full_qr_observation(observation, &matrix, *rows, *cols);
    }
    assert_orthonormal_columns(&bound_factor_matrices(&q));
    assert_nonnegative_diagonal(&bound_factor_matrices(&r));
    assert_compact_factors_reconstruct_input(&input, &q, None, &r);

    let mut lq_dense = FullQrInputSpy::default();
    let (l, q) = lq_full_dyn(&mut lq_dense, &input).unwrap();
    assert_eq!(lq_dense.observations.len(), matrices.len());
    for (observation, (_, rows, cols, matrix)) in lq_dense.observations.iter().zip(matrices.iter())
    {
        let matrix = matrix
            .iter()
            .map(|&value| Complex64::new(value, 0.0))
            .collect::<Vec<_>>();
        let adjoint = adjoint_complex(&matrix, *rows, *cols);
        assert_full_qr_observation(observation, &adjoint, *cols, *rows);
    }
    assert_nonnegative_diagonal(&bound_factor_matrices(&l));
    assert_orthonormal_rows(&bound_factor_matrices(&q));
    assert_compact_factors_reconstruct_input(&input, &l, None, &q);
}

fn checked_fixture_matrices(
    space: &BoundDynamicFusionMapSpace<LateGenericSpy>,
    data: &[Complex64],
) -> Vec<(usize, usize, Vec<Complex64>)> {
    (0..space.space().structure().block_count())
        .map(|index| {
            let block = space.space().structure().block(index).unwrap();
            let (rows, cols) = (block.shape()[0], block.shape()[1]);
            let mut matrix = vec![Complex64::new(0.0, 0.0); rows * cols];
            for col in 0..cols {
                for row in 0..rows {
                    matrix[row + rows * col] =
                        data[block.offset() + row * block.strides()[0] + col * block.strides()[1]];
                }
            }
            (rows, cols, matrix)
        })
        .collect()
}

fn assert_checked_full_qr_lq_inputs(
    provider: Arc<LateGenericSpy>,
    space: BoundDynamicFusionMapSpace<LateGenericSpy>,
    data: Vec<Complex64>,
) {
    let matrices = checked_fixture_matrices(&space, &data);
    let input = BoundDynamicTensorRef::try_new(&space, &data).unwrap();

    let mut qr_dense = FullQrInputSpy::default();
    let (q, r) = qr_full_dyn_checked_generic(&mut qr_dense, &input).unwrap();
    assert_eq!(qr_dense.observations.len(), matrices.len());
    for (observation, (rows, cols, matrix)) in qr_dense.observations.iter().zip(matrices.iter()) {
        assert_full_qr_observation(observation, matrix, *rows, *cols);
    }
    assert_compact_factors_reconstruct_input(&input, &q, None, &r);
    assert!(Arc::ptr_eq(q.space().provider_arc(), &provider));
    assert!(Arc::ptr_eq(r.space().provider_arc(), &provider));

    let mut lq_dense = FullQrInputSpy::default();
    let (l, q) = lq_full_dyn_checked_generic(&mut lq_dense, &input).unwrap();
    assert_eq!(lq_dense.observations.len(), matrices.len());
    for (observation, (rows, cols, matrix)) in lq_dense.observations.iter().zip(matrices.iter()) {
        let adjoint = adjoint_complex(matrix, *rows, *cols);
        assert_full_qr_observation(observation, &adjoint, *cols, *rows);
    }
    assert_compact_factors_reconstruct_input(&input, &l, None, &q);
    assert!(Arc::ptr_eq(l.space().provider_arc(), &provider));
    assert!(Arc::ptr_eq(q.space().provider_arc(), &provider));
}

#[test]
fn checked_full_qr_and_lq_preserve_complex_inputs_and_provider_identity() {
    let (provider, space, data) = checked_svd_truncation_input::<Complex64>(true);
    assert_checked_full_qr_lq_inputs(provider, space, data);
    let (provider, space, data) = checked_svd_wide_input::<Complex64>();
    assert_checked_full_qr_lq_inputs(provider, space, data);
}

#[test]
fn full_and_compact_qr_lq_match_for_rank_deficient_no_completion_shapes() {
    let rule = Z2FusionRule;
    let wide_space = rectangular_svd_tensor(2, 3)
        .fusion_space()
        .unwrap()
        .as_ref()
        .clone();
    let wide =
        TensorMap::from_vec_with_fusion_space(vec![1.0, 2.0, 2.0, 4.0, 3.0, 6.0], wide_space)
            .unwrap();
    let wide_input = bound_tensor(Arc::new(rule), &wide);
    let wide_input_ref = wide_input.as_ref();
    let wide_input = wide_input_ref.dynamic();
    let mut dense = tenet_dense::DefaultDenseExecutor::new();
    let compact = qr_compact_dyn(&mut dense, &wide_input).unwrap();
    let full = qr_full_dyn(&mut dense, &wide_input).unwrap();
    assert_eq!(full.0.space().space(), compact.0.space().space());
    assert_eq!(full.1.space().space(), compact.1.space().space());
    assert_eq!(full.0.data(), compact.0.data());
    assert_eq!(full.1.data(), compact.1.data());
    assert_orthonormal_columns(&bound_factor_matrices(&full.0));
    assert_nonnegative_diagonal(&bound_factor_matrices(&full.1));
    assert_compact_factors_reconstruct_input(&wide_input, &full.0, None, &full.1);

    let tall = transposed_rectangular_tensor(&wide, 2, 3);
    let tall_input = bound_tensor(Arc::new(rule), &tall);
    let tall_input_ref = tall_input.as_ref();
    let tall_input = tall_input_ref.dynamic();
    let compact = lq_compact_dyn(&mut dense, &tall_input).unwrap();
    let full = lq_full_dyn(&mut dense, &tall_input).unwrap();
    assert_eq!(full.0.space().space(), compact.0.space().space());
    assert_eq!(full.1.space().space(), compact.1.space().space());
    assert_eq!(full.0.data(), compact.0.data());
    assert_eq!(full.1.data(), compact.1.data());
    assert_nonnegative_diagonal(&bound_factor_matrices(&full.0));
    assert_orthonormal_rows(&bound_factor_matrices(&full.1));
    assert_compact_factors_reconstruct_input(&tall_input, &full.0, None, &full.1);
}

#[test]
fn qr_full_gives_square_unitary_and_reconstructs() {
    let rule = SU2FusionRule;
    let tensor = tsvd_test_tensor(
        &rule,
        &[
            SU2Irrep::from_twice_spin(0).sector_id(),
            SU2Irrep::from_twice_spin(1).sector_id(),
        ],
    );
    let mut dense_executor = tenet_dense::DefaultDenseExecutor::new();
    let (q, r) = qr_full(
        &mut dense_executor,
        &bound_tensor_ref!(Arc::new(rule), &tensor),
    )
    .unwrap();

    let matrices = dense_sector_matrices(2, &q);
    for (_, rows, cols, _) in &matrices {
        assert_eq!(rows, cols, "full Q must be square per sector");
    }
    assert_orthonormal_columns(&matrices);

    let reconstructed = contract_pair(&rule, &tensor, &q, &r);
    assert_svd_blocks_match(&tensor, &reconstructed);
}

#[test]
fn lq_full_reconstructs() {
    let rule = Z2FusionRule;
    let tensor = tsvd_test_tensor(&rule, &[SectorId::new(0), SectorId::new(1)]);
    let mut dense_executor = tenet_dense::DefaultDenseExecutor::new();
    let (l, q) = lq_full(
        &mut dense_executor,
        &bound_tensor_ref!(Arc::new(rule), &tensor),
    )
    .unwrap();
    let reconstructed = contract_pair(&rule, &tensor, &l, &q);
    assert_svd_blocks_match(&tensor, &reconstructed);
}

#[test]
fn svd_full_gives_square_unitaries_and_reconstructs() {
    let rule = Z2FusionRule;
    let tensor = tsvd_test_tensor(&rule, &[SectorId::new(0), SectorId::new(1)]);
    let mut dense_executor = tenet_dense::DefaultDenseExecutor::new();
    let full = svd_full(
        &mut dense_executor,
        &bound_tensor_ref!(Arc::new(rule), &tensor),
    )
    .unwrap();

    let matrices = dense_sector_matrices(2, &full.u);
    for (_, rows, cols, _) in &matrices {
        assert_eq!(rows, cols, "full U must be square per sector");
    }
    assert_orthonormal_columns(&matrices);

    // U . S has U's codomain and S's (column) bond as domain; build its space
    // from the contraction homspace and per-tree shapes.
    let us_hom = FusionTreeHomSpace::tensorcontract_homspace(
        &rule,
        full.u.fusion_space().unwrap().homspace(),
        full.s.fusion_space().unwrap().homspace(),
        &[2],
        &[0],
        &[0, 1, 2],
        2,
    )
    .unwrap();
    let u_structure = std::sync::Arc::clone(full.u.structure());
    let s_structure = std::sync::Arc::clone(full.s.structure());
    let shapes = us_hom
        .fusion_tree_keys(&rule)
        .iter()
        .map(|key| {
            let sector = key.domain_tree().coupled();
            let mut shape = None;
            for index in 0..u_structure.block_count() {
                let block = u_structure.block(index).unwrap();
                let BlockKey::FusionTree(u_key) = block.key() else {
                    continue;
                };
                if u_key.codomain_tree() == key.codomain_tree() {
                    shape = Some(block.shape()[..2].to_vec());
                    break;
                }
            }
            let mut shape = shape.expect("U tree present");
            let mut s_cols = 0;
            for index in 0..s_structure.block_count() {
                let block = s_structure.block(index).unwrap();
                let BlockKey::FusionTree(s_key) = block.key() else {
                    continue;
                };
                let s_sector = s_key.domain_tree().coupled();
                if s_sector == sector {
                    s_cols = block.shape()[1];
                    break;
                }
            }
            shape.push(s_cols);
            shape
        })
        .collect::<Vec<_>>();
    let dims = full.u.tensor().space().dims();
    let us_space = FusionTensorMapSpace::<2, 1>::from_degeneracy_shapes_coupled(
        TensorMapSpace::<2, 1>::from_dims([dims[0], dims[1]], [full.s.tensor().space().dims()[1]])
            .unwrap(),
        us_hom,
        &rule,
        shapes,
    )
    .unwrap();
    let mut us = TensorMap::<f64, 2, 1>::from_vec_with_fusion_space(
        vec![0.0; us_space.required_len().unwrap()],
        us_space,
    )
    .unwrap();
    let mut context = TensorContractFusionExecutionContext::<f64, RuleIdentity>::default();
    context
        .tensorcontract_fusion_into(
            &rule,
            &mut us,
            &full.u,
            &full.s,
            TensorContractSpec::new(&[2], &[0], OutputAxisOrder::from_axes(&[0, 1, 2])),
            1.0,
            0.0,
        )
        .unwrap();
    let reconstructed = contract_pair(&rule, &tensor, &us, &full.vh);
    assert_svd_blocks_match(&tensor, &reconstructed);
}

#[test]
fn full_factorizations_preserve_compact_bytes_on_matching_square_support() {
    let rule = U1FusionRule;
    let neutral = U1Irrep::new(0).sector_id();
    let homspace = FusionTreeHomSpace::new(
        FusionProductSpace::new([SectorLeg::new([(neutral, 2)], false)]),
        FusionProductSpace::new([SectorLeg::new([(neutral, 2)], false)]),
    );
    let space = FusionTensorMapSpace::from_degeneracy_shapes_coupled(
        TensorMapSpace::<1, 1>::from_dims([2], [2]).unwrap(),
        homspace,
        &rule,
        [vec![2, 2]],
    )
    .unwrap();
    let tensor = TensorMap::from_vec_with_fusion_space(vec![-1.0, 3.0, 2.0, 4.0], space).unwrap();
    let input = bound_tensor(Arc::new(rule), &tensor);
    let mut dense = tenet_dense::DefaultDenseExecutor::new();

    let compact = svd_compact(&mut dense, &input.as_ref()).unwrap();
    crate::factorize::reset_factor_buffer_build_counts_for_test();
    let full = svd_full(&mut dense, &input.as_ref()).unwrap();
    assert_eq!(
        crate::factorize::factor_buffer_build_counts_for_test(),
        (1, 1),
        "full SVD must build exactly its returned U and Vh buffers"
    );
    assert_eq!(full.u.data(), compact.u.data());
    assert_eq!(full.s.data(), compact.s.data());
    assert_eq!(full.vh.data(), compact.vh.data());

    let (q_compact, r_compact) = qr_compact(&mut dense, &input.as_ref()).unwrap();
    let (q_full, r_full) = qr_full(&mut dense, &input.as_ref()).unwrap();
    assert_eq!(q_full.data(), q_compact.data());
    assert_eq!(r_full.data(), r_compact.data());

    let (l_compact, q_compact) = lq_compact(&mut dense, &input.as_ref()).unwrap();
    let (l_full, q_full) = lq_full(&mut dense, &input.as_ref()).unwrap();
    assert_eq!(l_full.data(), l_compact.data());
    assert_eq!(q_full.data(), q_compact.data());
}

#[test]
fn full_factorizations_skip_dense_backend_for_disjoint_support() {
    let rule = U1FusionRule;
    let positive = U1Irrep::new(1).sector_id();
    let neutral = U1Irrep::new(0).sector_id();
    let homspace = FusionTreeHomSpace::new(
        FusionProductSpace::new([SectorLeg::new([(positive, 2)], false)]),
        FusionProductSpace::new([SectorLeg::new([(neutral, 3)], false)]),
    );
    let space = FusionTensorMapSpace::from_degeneracy_shapes_coupled(
        TensorMapSpace::<1, 1>::from_dims([2], [3]).unwrap(),
        homspace,
        &rule,
        Vec::<Vec<usize>>::new(),
    )
    .unwrap();
    let tensor = TensorMap::from_vec_with_fusion_space(Vec::<f64>::new(), space).unwrap();
    let input = bound_tensor(Arc::new(rule), &tensor);

    svd_full(&mut RejectExecutorCalls, &input.as_ref()).unwrap();
    qr_full(&mut RejectExecutorCalls, &input.as_ref()).unwrap();
    lq_full(&mut RejectExecutorCalls, &input.as_ref()).unwrap();
}

#[test]
fn svd_trunc_c64_reconstruction_distance_matches_error() {
    use num_complex::Complex64;
    let rule = Z2FusionRule;
    let sectors = [SectorId::new(0), SectorId::new(1)];
    let degeneracy = 2usize;
    let leg = || SectorLeg::new(sectors.iter().map(|&sector| (sector, degeneracy)), false);
    let leg_dim = sectors.len() * degeneracy;
    let homspace = FusionTreeHomSpace::new(
        FusionProductSpace::new([leg(), leg()]),
        FusionProductSpace::new([leg(), leg()]),
    );
    let key_count = homspace.fusion_tree_keys(&rule).len();
    let space = FusionTensorMapSpace::from_degeneracy_shapes_coupled(
        TensorMapSpace::<2, 2>::from_dims([leg_dim, leg_dim], [leg_dim, leg_dim]).unwrap(),
        homspace,
        &rule,
        vec![vec![degeneracy; 4]; key_count],
    )
    .unwrap();
    let len = space.required_len().unwrap();
    let tensor = TensorMap::<Complex64, 2, 2>::from_vec_with_fusion_space(
        (0..len)
            .map(|i| {
                Complex64::new(
                    ((i * 7 + 3) % 23) as f64 * 0.5 - 5.0,
                    ((i * 5 + 1) % 17) as f64 * 0.25 - 2.0,
                )
            })
            .collect(),
        space,
    )
    .unwrap();

    let mut dense_executor = tenet_dense::DefaultDenseExecutor::new();
    crate::factorize::reset_compact_svd_copy_probe();
    let svd = svd_trunc(
        &mut dense_executor,
        &bound_tensor_ref!(Arc::new(rule), &tensor),
        &Truncation::rank(8),
    )
    .unwrap();
    assert_compact_svd_direct_copy_probe();
    assert!(svd.error > 0.0);
    for entry in &svd.singular_values {
        for pair in entry.values.windows(2) {
            assert!(pair[0] >= pair[1] - 1e-12);
        }
    }

    // Scale Vh rows by the (real) singular values.
    let mut scaled_vh = svd.vh.tensor().clone();
    {
        let structure = std::sync::Arc::clone(scaled_vh.structure());
        for index in 0..structure.block_count() {
            let block = structure.block(index).unwrap();
            let BlockKey::FusionTree(key) = block.key() else {
                continue;
            };
            let sector = key.codomain_tree().coupled();
            let values = &svd
                .singular_values
                .iter()
                .find(|entry| entry.sector == sector)
                .unwrap()
                .values;
            let shape = block.shape().to_vec();
            let strides = block.strides().to_vec();
            let offset = block.offset();
            let count = shape.iter().product::<usize>();
            let mut indices = vec![0usize; shape.len()];
            for _ in 0..count {
                let position = offset
                    + indices
                        .iter()
                        .zip(&strides)
                        .map(|(&i, &s)| i * s)
                        .sum::<usize>();
                scaled_vh.data_mut()[position] *= values[indices[0]];
                for axis in 0..shape.len() {
                    indices[axis] += 1;
                    if indices[axis] < shape[axis] {
                        break;
                    }
                    indices[axis] = 0;
                }
            }
        }
    }

    let mut reconstructed = TensorMap::<Complex64, 2, 2>::from_vec_with_fusion_space(
        vec![Complex64::new(0.0, 0.0); len],
        tensor.fusion_space().unwrap().as_ref().clone(),
    )
    .unwrap();
    let mut context = TensorContractFusionExecutionContext::<Complex64, RuleIdentity>::default();
    context
        .tensorcontract_fusion_into(
            &rule,
            &mut reconstructed,
            &svd.u,
            &scaled_vh,
            TensorContractSpec::new(&[2], &[0], OutputAxisOrder::from_axes(&[0, 1, 2, 3])),
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 0.0),
        )
        .unwrap();

    // Weighted 2-norm of the difference equals the reported error (Z2 has
    // quantum dimension 1 everywhere).
    let distance = tensor
        .data()
        .iter()
        .zip(reconstructed.data())
        .map(|(lhs, rhs)| (lhs - rhs).norm_sqr())
        .sum::<f64>()
        .sqrt();
    assert!(
        (distance - svd.error).abs() < 1e-8,
        "distance {distance} != error {}",
        svd.error
    );
}

#[test]
fn eig_full_satisfies_the_eigen_equation_for_real_input() {
    use num_complex::Complex64;
    let rule = Z2FusionRule;
    // Non-symmetric endomorphism.
    let tensor = tsvd_test_tensor(&rule, &[SectorId::new(0), SectorId::new(1)]);
    let mut dense_executor = tenet_dense::DefaultDenseExecutor::new();
    let eig = eig_full(
        &mut dense_executor,
        &bound_tensor_ref!(Arc::new(rule), &tensor),
    )
    .unwrap();

    for entry in &eig.eigenvalues {
        for pair in entry.values.windows(2) {
            assert!(pair[0].norm() >= pair[1].norm() - 1e-12);
        }
    }

    // Promote t to complex (same space => same layout => elementwise cast).
    let tensor_c = TensorMap::<Complex64, 2, 2>::from_vec_with_fusion_space(
        tensor
            .data()
            .iter()
            .map(|&value| Complex64::new(value, 0.0))
            .collect(),
        tensor.fusion_space().unwrap().as_ref().clone(),
    )
    .unwrap();

    let mut context = TensorContractFusionExecutionContext::<Complex64, RuleIdentity>::default();
    let mut tv = TensorMap::<Complex64, 2, 1>::from_vec_with_fusion_space(
        vec![Complex64::new(0.0, 0.0); eig.v.data().len()],
        eig.v.fusion_space().unwrap().as_ref().clone(),
    )
    .unwrap();
    context
        .tensorcontract_fusion_into(
            &rule,
            &mut tv,
            &tensor_c,
            &eig.v,
            TensorContractSpec::new(&[2, 3], &[0, 1], OutputAxisOrder::from_axes(&[0, 1, 2])),
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 0.0),
        )
        .unwrap();
    let mut vd = TensorMap::<Complex64, 2, 1>::from_vec_with_fusion_space(
        vec![Complex64::new(0.0, 0.0); eig.v.data().len()],
        eig.v.fusion_space().unwrap().as_ref().clone(),
    )
    .unwrap();
    context
        .tensorcontract_fusion_into(
            &rule,
            &mut vd,
            &eig.v,
            &eig.d,
            TensorContractSpec::new(&[2], &[0], OutputAxisOrder::from_axes(&[0, 1, 2])),
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 0.0),
        )
        .unwrap();
    for (index, (lhs, rhs)) in tv.data().iter().zip(vd.data()).enumerate() {
        assert!(
            (lhs - rhs).norm() < 1e-8,
            "eigen equation violated at raw position {index}: {lhs} != {rhs}"
        );
    }
}

#[test]
fn null_spaces_are_orthonormal_and_annihilate_the_tensor() {
    let rule = Z2FusionRule;
    let sectors = [SectorId::new(0), SectorId::new(1)];
    let degeneracy = 2usize;
    let leg = || SectorLeg::new(sectors.iter().map(|&sector| (sector, degeneracy)), false);
    let leg_dim = sectors.len() * degeneracy;

    // Tall map (2 codomain legs, 1 domain leg): nontrivial left null space.
    let tall_hom = FusionTreeHomSpace::new(
        FusionProductSpace::new([leg(), leg()]),
        FusionProductSpace::new([leg()]),
    );
    let key_count = tall_hom.fusion_tree_keys(&rule).len();
    let tall_space = FusionTensorMapSpace::from_degeneracy_shapes_coupled(
        TensorMapSpace::<2, 1>::from_dims([leg_dim, leg_dim], [leg_dim]).unwrap(),
        tall_hom,
        &rule,
        vec![vec![degeneracy; 3]; key_count],
    )
    .unwrap();
    let len = tall_space.required_len().unwrap();
    let tall = TensorMap::<f64, 2, 1>::from_vec_with_fusion_space(
        (0..len).map(|i| ((i * 3 + 1) % 13) as f64 - 6.0).collect(),
        tall_space,
    )
    .unwrap();
    let mut dense_executor = tenet_dense::DefaultDenseExecutor::new();
    let null = left_null(
        &mut dense_executor,
        &bound_tensor_ref!(Arc::new(rule), &tall),
    )
    .unwrap();

    let null_matrices = dense_sector_matrices(2, &null);
    assert!(!null_matrices.is_empty());
    assert_orthonormal_columns(&null_matrices);
    let tensor_matrices = dense_sector_matrices(2, &tall);
    for (sector, n_rows, n_cols, n) in &null_matrices {
        let (_, a_rows, a_cols, a) = tensor_matrices
            .iter()
            .find(|(candidate, ..)| candidate == sector)
            .expect("tensor sector present");
        assert_eq!(n_rows, a_rows);
        assert_eq!(*n_cols, a_rows - (*a_rows).min(*a_cols));
        // N^T A = 0.
        for null_col in 0..*n_cols {
            for a_col in 0..*a_cols {
                let mut dot = 0.0;
                for row in 0..*a_rows {
                    dot += n[row + n_rows * null_col] * a[row + a_rows * a_col];
                }
                assert!(dot.abs() < 1e-9, "left null failed: {dot}");
            }
        }
    }

    // Wide map (1 codomain leg, 2 domain legs): nontrivial right null space.
    let wide_hom = FusionTreeHomSpace::new(
        FusionProductSpace::new([leg()]),
        FusionProductSpace::new([leg(), leg()]),
    );
    let key_count = wide_hom.fusion_tree_keys(&rule).len();
    let wide_space = FusionTensorMapSpace::from_degeneracy_shapes_coupled(
        TensorMapSpace::<1, 2>::from_dims([leg_dim], [leg_dim, leg_dim]).unwrap(),
        wide_hom,
        &rule,
        vec![vec![degeneracy; 3]; key_count],
    )
    .unwrap();
    let len = wide_space.required_len().unwrap();
    let wide = TensorMap::<f64, 1, 2>::from_vec_with_fusion_space(
        (0..len).map(|i| ((i * 5 + 2) % 11) as f64 - 5.0).collect(),
        wide_space,
    )
    .unwrap();
    let null = right_null(
        &mut dense_executor,
        &bound_tensor_ref!(Arc::new(rule), &wide),
    )
    .unwrap();

    let null_matrices = dense_sector_matrices(1, &null);
    assert!(!null_matrices.is_empty());
    let tensor_matrices = dense_sector_matrices(1, &wide);
    for (sector, n_rows, n_cols, n) in &null_matrices {
        let (_, a_rows, a_cols, a) = tensor_matrices
            .iter()
            .find(|(candidate, ..)| candidate == sector)
            .expect("tensor sector present");
        assert_eq!(n_cols, a_cols);
        assert_eq!(*n_rows, a_cols - (*a_cols).min(*a_rows));
        // Rows of N are orthonormal: N N^T = I.
        for left in 0..*n_rows {
            for right in 0..*n_rows {
                let mut dot = 0.0;
                for col in 0..*n_cols {
                    dot += n[left + n_rows * col] * n[right + n_rows * col];
                }
                let expected = if left == right { 1.0 } else { 0.0 };
                assert!((dot - expected).abs() < 1e-9);
            }
        }
        // A N^T = 0 (rows of N span the kernel).
        for a_row in 0..*a_rows {
            for null_row in 0..*n_rows {
                let mut dot = 0.0;
                for col in 0..*a_cols {
                    dot += a[a_row + a_rows * col] * n[null_row + n_rows * col];
                }
                assert!(dot.abs() < 1e-9, "right null failed: {dot}");
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

#[test]
fn rectangular_full_svd_has_square_outer_factors_and_reconstructs() {
    // What: full SVD returns U(m,m), S(m,n), Vh(n,n) and recomposes tall and wide inputs.
    let rule = Z2FusionRule;
    let mut dense = tenet_dense::DefaultDenseExecutor::new();
    for (rows, cols) in [(2, 3), (3, 2)] {
        let matrix = one_sector_rectangular_matrix(
            (0..rows * cols)
                .map(|index| ((index * 5 + 1) % 11) as f64 - 4.0)
                .collect(),
            rows,
            cols,
        );
        let input = bound_tensor(Arc::new(rule), &matrix);
        let full = svd_full(&mut dense, &input.as_ref()).unwrap();
        assert_factor_layout_matches_legacy_shapes(full.u.space());
        assert_factor_layout_matches_legacy_shapes(full.s.space());
        assert_factor_layout_matches_legacy_shapes(full.vh.space());
        assert_eq!(full.u.structure().block(0).unwrap().shape(), &[rows, rows]);
        assert_eq!(full.s.structure().block(0).unwrap().shape(), &[rows, cols]);
        assert_eq!(full.vh.structure().block(0).unwrap().shape(), &[cols, cols]);

        let mut us = vec![0.0; rows * cols];
        for col in 0..cols {
            for inner in 0..rows {
                for row in 0..rows {
                    us[row + rows * col] +=
                        full.u.data()[row + rows * inner] * full.s.data()[inner + rows * col];
                }
            }
        }
        let mut reconstructed = vec![0.0; rows * cols];
        for col in 0..cols {
            for inner in 0..cols {
                for row in 0..rows {
                    reconstructed[row + rows * col] +=
                        us[row + rows * inner] * full.vh.data()[inner + cols * col];
                }
            }
        }
        for (actual, expected) in reconstructed.iter().zip(matrix.data()) {
            assert!((actual - expected).abs() < 1.0e-9);
        }
    }
}

#[test]
fn rank_deficient_real_null_spaces_include_zero_and_duplicate_directions() {
    // What: numerical nullity, not the rectangular shape deficit, determines both null spaces.
    let rule = Z2FusionRule;
    let mut dense = tenet_dense::DefaultDenseExecutor::new();
    for (matrix, expected_nullity) in [
        (one_sector_matrix(vec![0.0; 4]), 2),
        (one_sector_matrix(vec![1.0, 2.0, 1.0, 2.0]), 1),
        (one_sector_matrix(vec![1.0, 1.0, 2.0, 2.0]), 1),
    ] {
        let input = bound_tensor(Arc::new(rule), &matrix);
        let left = left_null(&mut dense, &input.as_ref()).unwrap();
        let right = right_null(&mut dense, &input.as_ref()).unwrap();
        let left_shape = left.structure().block(0).unwrap().shape();
        let right_shape = right.structure().block(0).unwrap().shape();
        assert_eq!(left_shape, &[2, expected_nullity]);
        assert_eq!(right_shape, &[expected_nullity, 2]);

        for null_col in 0..expected_nullity {
            for matrix_col in 0..2 {
                let dot = (0..2)
                    .map(|row| {
                        left.data()[row + 2 * null_col] * matrix.data()[row + 2 * matrix_col]
                    })
                    .sum::<f64>();
                assert!(dot.abs() < 1.0e-10);
            }
        }
        for matrix_row in 0..2 {
            for null_row in 0..expected_nullity {
                let dot = (0..2)
                    .map(|col| {
                        matrix.data()[matrix_row + 2 * col]
                            * right.data()[null_row + expected_nullity * col]
                    })
                    .sum::<f64>();
                assert!(dot.abs() < 1.0e-10);
            }
        }
    }
}

#[test]
fn numerical_null_rank_uses_the_documented_f64_threshold() {
    // What: singular values immediately below and above
    // epsilon(f64) * max(m, n) * sigma_max fall on opposite rank decisions.
    let rule = Z2FusionRule;
    let tolerance = f64::EPSILON * 2.0;
    let mut dense = tenet_dense::DefaultDenseExecutor::new();
    for (small, expected_nullity) in [(0.5 * tolerance, 1), (2.0 * tolerance, 0)] {
        let matrix = one_sector_matrix(vec![1.0, 0.0, 0.0, small]);
        let input = bound_tensor(Arc::new(rule), &matrix);
        let left = left_null(&mut dense, &input.as_ref()).unwrap();
        let right = right_null(&mut dense, &input.as_ref()).unwrap();
        if expected_nullity == 0 {
            assert!(left.data().is_empty());
            assert!(right.data().is_empty());
        } else {
            assert_eq!(left.structure().block(0).unwrap().shape(), &[2, 1]);
            assert_eq!(right.structure().block(0).unwrap().shape(), &[1, 2]);
        }
    }
}

#[test]
fn numerical_null_rank_uses_the_documented_f32_threshold() {
    // What: the rank contract follows the input dtype rather than silently
    // applying the f64 machine epsilon to f32 sectors.
    let rule = Z2FusionRule;
    let tolerance = f32::EPSILON * 2.0;
    let mut dense = tenet_dense::DefaultDenseExecutor::new();
    for (small, expected_nullity) in [(0.5 * tolerance, 1), (2.0 * tolerance, 0)] {
        let matrix = one_sector_matrix(vec![1.0_f32, 0.0, 0.0, small]);
        let input = bound_tensor(Arc::new(rule), &matrix);
        let left = left_null(&mut dense, &input.as_ref()).unwrap();
        let right = right_null(&mut dense, &input.as_ref()).unwrap();
        if expected_nullity == 0 {
            assert!(left.data().is_empty());
            assert!(right.data().is_empty());
        } else {
            assert_eq!(left.structure().block(0).unwrap().shape(), &[2, 1]);
            assert_eq!(right.structure().block(0).unwrap().shape(), &[1, 2]);
        }
    }
}

#[test]
fn rectangular_rank_deficient_null_spaces_include_shape_and_rank_deficits() {
    // What: tall and wide sectors include both the rectangular shape deficit
    // and additional null directions caused by numerical rank deficiency.
    let rule = Z2FusionRule;
    let mut dense = tenet_dense::DefaultDenseExecutor::new();
    for (rows, cols, data, left_nullity, right_nullity) in [
        (3, 2, vec![1.0, 2.0, 3.0, 2.0, 4.0, 6.0], 2, 1),
        (2, 3, vec![1.0, 2.0, 2.0, 4.0, 3.0, 6.0], 1, 2),
    ] {
        let matrix = one_sector_rectangular_matrix(data, rows, cols);
        let input = bound_tensor(Arc::new(rule), &matrix);
        let left = left_null(&mut dense, &input.as_ref()).unwrap();
        let right = right_null(&mut dense, &input.as_ref()).unwrap();
        assert_eq!(
            left.structure().block(0).unwrap().shape(),
            &[rows, left_nullity]
        );
        assert_eq!(
            right.structure().block(0).unwrap().shape(),
            &[right_nullity, cols]
        );

        for null_col in 0..left_nullity {
            for matrix_col in 0..cols {
                let dot = (0..rows)
                    .map(|row| {
                        left.data()[row + rows * null_col] * matrix.data()[row + rows * matrix_col]
                    })
                    .sum::<f64>();
                assert!(dot.abs() < 1.0e-9);
            }
        }
        for matrix_row in 0..rows {
            for null_row in 0..right_nullity {
                let dot = (0..cols)
                    .map(|col| {
                        matrix.data()[matrix_row + rows * col]
                            * right.data()[null_row + right_nullity * col]
                    })
                    .sum::<f64>();
                assert!(dot.abs() < 1.0e-9);
            }
        }
    }
}

#[test]
fn rank_deficient_complex_null_spaces_include_zero_and_duplicate_directions() {
    // What: complex conjugation and numerical-rank detection preserve the full left/right kernels.
    use num_complex::Complex64;

    let rule = Z2FusionRule;
    let zero = Complex64::new(0.0, 0.0);
    let duplicate = one_sector_matrix(vec![
        Complex64::new(1.0, 1.0),
        Complex64::new(2.0, -1.0),
        Complex64::new(1.0, 1.0),
        Complex64::new(2.0, -1.0),
    ]);
    let mut dense = tenet_dense::DefaultDenseExecutor::new();
    for (matrix, expected_nullity) in [(one_sector_matrix(vec![zero; 4]), 2), (duplicate, 1)] {
        let input = bound_tensor(Arc::new(rule), &matrix);
        let left = left_null(&mut dense, &input.as_ref()).unwrap();
        let right = right_null(&mut dense, &input.as_ref()).unwrap();
        assert_eq!(
            left.structure().block(0).unwrap().shape(),
            &[2, expected_nullity]
        );
        assert_eq!(
            right.structure().block(0).unwrap().shape(),
            &[expected_nullity, 2]
        );

        for null_col in 0..expected_nullity {
            for matrix_col in 0..2 {
                let dot = (0..2)
                    .map(|row| {
                        left.data()[row + 2 * null_col].conj() * matrix.data()[row + 2 * matrix_col]
                    })
                    .sum::<Complex64>();
                assert!(dot.norm() < 1.0e-10);
            }
        }
        for matrix_row in 0..2 {
            for null_row in 0..expected_nullity {
                let dot = (0..2)
                    .map(|col| {
                        matrix.data()[matrix_row + 2 * col]
                            * right.data()[null_row + expected_nullity * col].conj()
                    })
                    .sum::<Complex64>();
                assert!(dot.norm() < 1.0e-10);
            }
        }
    }
}

#[test]
fn spectrum_only_entry_points_return_descending_magnitudes() {
    let rule = Z2FusionRule;
    let hermitian = hermitian_test_tensor(&rule, &[SectorId::new(0), SectorId::new(1)]);
    let general = tsvd_test_tensor(&rule, &[SectorId::new(0), SectorId::new(1)]);
    let mut dense_executor = tenet_dense::DefaultDenseExecutor::new();

    let svd = svd_vals(
        &mut dense_executor,
        &bound_tensor_ref!(Arc::new(rule), &general),
    )
    .unwrap();
    assert!(!svd.is_empty());
    for entry in &svd {
        for pair in entry.values.windows(2) {
            assert!(pair[0] >= pair[1] - 1e-12);
        }
    }
    let eigh = eigh_vals(
        &mut dense_executor,
        &bound_tensor_ref!(Arc::new(rule), &hermitian),
    )
    .unwrap();
    assert!(!eigh.is_empty());
    for entry in &eigh {
        for pair in entry.values.windows(2) {
            assert!(pair[0].abs() >= pair[1].abs() - 1e-12);
        }
    }
    let eig = eig_vals(
        &mut dense_executor,
        &bound_tensor_ref!(Arc::new(rule), &general),
    )
    .unwrap();
    assert!(!eig.is_empty());
    for entry in &eig {
        for pair in entry.values.windows(2) {
            assert!(pair[0].norm() >= pair[1].norm() - 1e-12);
        }
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

fn assert_complex_spectra_close(
    lhs: &[SectorSpectrum<Complex64>],
    rhs: &[SectorSpectrum<Complex64>],
) {
    assert_eq!(lhs.len(), rhs.len());
    for (lhs, rhs) in lhs.iter().zip(rhs) {
        assert_eq!(lhs.sector, rhs.sector);
        assert_eq!(lhs.values.len(), rhs.values.len());
        for (&lhs, &rhs) in lhs.values.iter().zip(&rhs.values) {
            assert!((lhs - rhs).norm() <= 1e-10, "{lhs} vs {rhs}");
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

fn assert_value_region_paths_match<R, D>(
    rule: Arc<R>,
    general: &TensorMap<D, 2, 2>,
    hermitian: &TensorMap<D, 2, 2>,
) where
    R: MultiplicityFreeRigidSymbols<Scalar = f64>,
    D: FactorScalar,
{
    let general_padded = padded_copy(rule.as_ref(), general);
    let hermitian_padded = padded_copy(rule.as_ref(), hermitian);
    let general_bound = bound_tensor(Arc::clone(&rule), general);
    let hermitian_bound = bound_tensor(Arc::clone(&rule), hermitian);
    let general_fallback = bound_tensor(Arc::clone(&rule), &general_padded);
    let hermitian_fallback = bound_tensor(rule, &hermitian_padded);
    assert!(general_bound
        .space()
        .space()
        .structure()
        .coupled_sector_regions(2)
        .unwrap()
        .is_some());
    assert!(general_fallback
        .space()
        .space()
        .structure()
        .coupled_sector_regions(2)
        .unwrap()
        .is_none());
    assert!(hermitian_fallback
        .space()
        .space()
        .structure()
        .coupled_sector_regions(2)
        .unwrap()
        .is_none());
    let general_before = general_bound.data().to_vec();
    let hermitian_before = hermitian_bound.data().to_vec();
    let general_fallback_before = general_fallback.data().to_vec();
    let hermitian_fallback_before = hermitian_fallback.data().to_vec();
    let mut dense = tenet_dense::DefaultDenseExecutor::new();

    crate::factorize::reset_values_matricization_fallbacks();
    let direct_svd = svd_vals_dyn(&mut dense, &general_bound.as_ref().dynamic()).unwrap();
    let direct_eigh = eigh_vals_dyn(&mut dense, &hermitian_bound.as_ref().dynamic()).unwrap();
    let direct_eig = eig_vals_dyn(&mut dense, &general_bound.as_ref().dynamic()).unwrap();
    assert_eq!(crate::factorize::values_matricization_fallbacks(), 0);

    crate::factorize::reset_values_matricization_fallbacks();
    let packed_svd = svd_vals_dyn(&mut dense, &general_fallback.as_ref().dynamic()).unwrap();
    let packed_eigh = eigh_vals_dyn(&mut dense, &hermitian_fallback.as_ref().dynamic()).unwrap();
    let packed_eig = eig_vals_dyn(&mut dense, &general_fallback.as_ref().dynamic()).unwrap();
    assert_eq!(crate::factorize::values_matricization_fallbacks(), 3);

    assert_real_spectra_close(&direct_svd, &packed_svd);
    assert_real_spectra_close(&direct_eigh, &packed_eigh);
    assert_complex_spectra_close(&direct_eig, &packed_eig);
    assert!(general_bound.data() == general_before);
    assert!(hermitian_bound.data() == hermitian_before);
    assert!(general_fallback.data() == general_fallback_before);
    assert!(hermitian_fallback.data() == hermitian_fallback_before);
}

#[test]
fn value_region_paths_match_packed_oracles_across_supported_rules() {
    // What: canonical region borrowing and noncanonical packing return the same
    // ordered spectra for Abelian, non-Abelian, fermionic, and product rules.
    let u1 = [
        U1Irrep::new(-1).sector_id(),
        U1Irrep::new(0).sector_id(),
        U1Irrep::new(1).sector_id(),
    ];
    assert_value_region_paths_match(
        Arc::new(U1FusionRule),
        &tsvd_test_tensor(&U1FusionRule, &u1),
        &hermitian_test_tensor(&U1FusionRule, &u1),
    );

    let su2 = [
        SU2Irrep::from_twice_spin(0).sector_id(),
        SU2Irrep::from_twice_spin(1).sector_id(),
    ];
    assert_value_region_paths_match(
        Arc::new(SU2FusionRule),
        &tsvd_test_tensor(&SU2FusionRule, &su2),
        &hermitian_test_tensor(&SU2FusionRule, &su2),
    );

    let fz2 = [SectorId::new(0), SectorId::new(1)];
    assert_value_region_paths_match(
        Arc::new(FermionParityFusionRule),
        &tsvd_test_tensor(&FermionParityFusionRule, &fz2),
        &hermitian_test_tensor(&FermionParityFusionRule, &fz2),
    );

    let product = product_fusion_rule(FermionParityFusionRule, U1FusionRule);
    let product_sectors = [
        product.encode_sector(SectorId::new(0), U1Irrep::new(0).sector_id()),
        product.encode_sector(SectorId::new(1), U1Irrep::new(1).sector_id()),
    ];
    assert_value_region_paths_match(
        Arc::new(product.clone()),
        &tsvd_test_tensor(&product, &product_sectors),
        &hermitian_test_tensor(&product, &product_sectors),
    );
}

#[test]
fn value_region_paths_match_packed_oracles_for_complex64() {
    // What: borrowed C64 spans preserve nonreal general matrices and conjugate
    // off-diagonal Hermitian matrices across all three values-only operations.
    let rule = Z2FusionRule;
    let sectors = [SectorId::new(0), SectorId::new(1)];
    let general = tsvd_test_tensor(&rule, &sectors);
    let hermitian = hermitian_test_tensor(&rule, &sectors);
    let general = TensorMap::<Complex64, 2, 2>::from_vec_with_fusion_space(
        general
            .data()
            .iter()
            .enumerate()
            .map(|(index, &value)| Complex64::new(value, (index % 7) as f64 * 0.125 - 0.25))
            .collect(),
        general.fusion_space().unwrap().as_ref().clone(),
    )
    .unwrap();
    let regions = hermitian
        .structure()
        .coupled_sector_regions(2)
        .unwrap()
        .unwrap();
    let mut hermitian_data = hermitian
        .data()
        .iter()
        .map(|&value| Complex64::new(value, 0.0))
        .collect::<Vec<_>>();
    for region in regions.iter().filter(|region| region.rows() >= 2) {
        let start = region.range().start;
        hermitian_data[start + 1].im = -0.75;
        hermitian_data[start + region.rows()].im = 0.75;
    }
    let hermitian = TensorMap::<Complex64, 2, 2>::from_vec_with_fusion_space(
        hermitian_data,
        hermitian.fusion_space().unwrap().as_ref().clone(),
    )
    .unwrap();

    assert_value_region_paths_match(Arc::new(rule), &general, &hermitian);
}

#[test]
fn values_only_public_boundaries_distinguish_empty_sectors_from_a_scalar() {
    // What: zero degeneracies remove the sector entirely, while a rank-zero
    // scalar remains one vacuum-sector 1x1 matrix for every values operation.
    let empty = rectangular_svd_tensor(0, 0);
    assert_eq!(empty.structure().block_count(), 0);
    assert!(empty.data().is_empty());
    let mut reject = RejectExecutorCalls;
    crate::factorize::reset_values_matricization_fallbacks();
    assert!(svd_vals(
        &mut reject,
        &bound_tensor_ref!(Arc::new(Z2FusionRule), &empty)
    )
    .unwrap()
    .is_empty());
    assert!(eigh_vals(
        &mut reject,
        &bound_tensor_ref!(Arc::new(Z2FusionRule), &empty)
    )
    .unwrap()
    .is_empty());
    assert!(eig_vals(
        &mut reject,
        &bound_tensor_ref!(Arc::new(Z2FusionRule), &empty)
    )
    .unwrap()
    .is_empty());
    assert_eq!(crate::factorize::values_matricization_fallbacks(), 0);

    let rule = Z2FusionRule;
    let homspace =
        FusionTreeHomSpace::new(FusionProductSpace::new([]), FusionProductSpace::new([]));
    let shapes = vec![Vec::new(); homspace.fusion_tree_keys(&rule).len()];
    let space = FusionTensorMapSpace::from_degeneracy_shapes_coupled(
        TensorMapSpace::<0, 0>::from_dims([], []).unwrap(),
        homspace,
        &rule,
        shapes,
    )
    .unwrap();
    let scalar = TensorMap::<f64, 0, 0>::from_vec_with_fusion_space(vec![-3.0], space).unwrap();
    let mut dense = tenet_dense::DefaultDenseExecutor::new();
    let svd = svd_vals(&mut dense, &bound_tensor_ref!(Arc::new(rule), &scalar)).unwrap();
    let eigh = eigh_vals(&mut dense, &bound_tensor_ref!(Arc::new(rule), &scalar)).unwrap();
    let eig = eig_vals(&mut dense, &bound_tensor_ref!(Arc::new(rule), &scalar)).unwrap();
    assert_eq!(svd[0].sector, rule.vacuum());
    assert_eq!(svd[0].values, vec![3.0]);
    assert_eq!(eigh[0].sector, rule.vacuum());
    assert_eq!(eigh[0].values, vec![-3.0]);
    assert_eq!(eig[0].sector, rule.vacuum());
    assert_eq!(eig[0].values, vec![Complex64::new(-3.0, 0.0)]);
}

#[test]
fn values_only_second_sector_failures_publish_no_partial_spectrum() {
    // What: after one successful sector, each dense values failure returns Err
    // without exposing the accumulated prefix or mutating borrowed input.
    let sectors = [SectorId::new(0), SectorId::new(1)];
    let general = tsvd_test_tensor(&Z2FusionRule, &sectors);
    let hermitian = hermitian_test_tensor(&Z2FusionRule, &sectors);
    for operation in [
        ValuesOperation::Svd,
        ValuesOperation::Eigh,
        ValuesOperation::Eig,
    ] {
        let tensor = if operation == ValuesOperation::Eigh {
            &hermitian
        } else {
            &general
        };
        let input = bound_tensor(Arc::new(Z2FusionRule), tensor);
        let before = input.data().to_vec();
        let mut dense = FailSecondValues::new(operation);
        let result = match operation {
            ValuesOperation::Svd => svd_vals(&mut dense, &input.as_ref()).map(|_| ()),
            ValuesOperation::Eigh => eigh_vals(&mut dense, &input.as_ref()).map(|_| ()),
            ValuesOperation::Eig => eig_vals(&mut dense, &input.as_ref()).map(|_| ()),
        };

        assert!(matches!(result, Err(OperationError::Dense(_))));
        assert_eq!(dense.calls, 2);
        assert_eq!(input.data(), before);
    }
}

#[test]
fn values_only_stable_ties_match_provider_order_on_direct_and_padded_layouts() {
    // What: equal singular values and equal-magnitude eigenvalues retain the
    // dense provider's order on both the borrowed and packed sector paths.
    let rule = Z2FusionRule;
    let svd_input =
        one_sector_rectangular_matrix(vec![2.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 1.0], 3, 3);
    let eigh_input =
        one_sector_rectangular_matrix(vec![1.0, 0.0, 0.0, 0.0, -2.0, 0.0, 0.0, 0.0, 2.0], 3, 3);
    let eig_input =
        one_sector_rectangular_matrix(vec![0.0, 1.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 1.0], 3, 3);
    let svd_padded = padded_copy(&rule, &svd_input);
    let eigh_padded = padded_copy(&rule, &eigh_input);
    let eig_padded = padded_copy(&rule, &eig_input);
    let shape = [3, 3];
    let strides = [1, 3];
    let mut dense = tenet_dense::DefaultDenseExecutor::new();

    let raw_svd = dense
        .svd_vals(DenseRead::F64(
            tenet_dense::DenseView::new(svd_input.data(), &shape, &strides, 0).unwrap(),
        ))
        .unwrap()
        .as_f64_slice()
        .unwrap()
        .to_vec();
    let mut raw_eigh = dense
        .eigh_vals(DenseRead::F64(
            tenet_dense::DenseView::new(eigh_input.data(), &shape, &strides, 0).unwrap(),
        ))
        .unwrap()
        .as_f64_slice()
        .unwrap()
        .to_vec();
    raw_eigh.sort_by(|a, b| b.abs().partial_cmp(&a.abs()).unwrap());
    let mut raw_eig = dense
        .eig_vals(DenseRead::F64(
            tenet_dense::DenseView::new(eig_input.data(), &shape, &strides, 0).unwrap(),
        ))
        .unwrap()
        .as_c64_slice()
        .unwrap()
        .to_vec();
    raw_eig.sort_by(|a, b| b.norm().partial_cmp(&a.norm()).unwrap());

    let direct_svd = svd_vals(&mut dense, &bound_tensor_ref!(Arc::new(rule), &svd_input)).unwrap();
    let padded_svd = svd_vals(&mut dense, &bound_tensor_ref!(Arc::new(rule), &svd_padded)).unwrap();
    let direct_eigh =
        eigh_vals(&mut dense, &bound_tensor_ref!(Arc::new(rule), &eigh_input)).unwrap();
    let padded_eigh =
        eigh_vals(&mut dense, &bound_tensor_ref!(Arc::new(rule), &eigh_padded)).unwrap();
    let direct_eig = eig_vals(&mut dense, &bound_tensor_ref!(Arc::new(rule), &eig_input)).unwrap();
    let padded_eig = eig_vals(&mut dense, &bound_tensor_ref!(Arc::new(rule), &eig_padded)).unwrap();

    assert_eq!(direct_svd[0].values, raw_svd);
    assert_eq!(direct_eigh[0].values, raw_eigh);
    assert_eq!(direct_eig[0].values, raw_eig);
    assert_real_spectra_close(&direct_svd, &padded_svd);
    assert_real_spectra_close(&direct_eigh, &padded_eigh);
    assert_complex_spectra_close(&direct_eig, &padded_eig);
}

#[test]
fn values_only_entry_points_match_untruncated_decomposition_spectra() {
    // The `_vals` paths call LAPACK `job='N'` (no vectors) and must reproduce
    // the untruncated decomposition's spectrum. This is a numerical-agreement check,
    // not bit-for-bit: LAPACK backends may route the vectors-vs-no-vectors
    // cases through different routines (e.g. `gesdd` divide-and-conquer for the
    // full SVD vs `gesvd` QR for values-only), which differ in the last ULPs.
    let rule = Z2FusionRule;
    let hermitian = hermitian_test_tensor(&rule, &[SectorId::new(0), SectorId::new(1)]);
    let general = tsvd_test_tensor(&rule, &[SectorId::new(0), SectorId::new(1)]);
    let mut dense_executor = tenet_dense::DefaultDenseExecutor::new();

    let tol = 1e-10;
    let assert_real_close = |vals: &[SectorSpectrum], full: &[SectorSpectrum]| {
        assert_eq!(vals.len(), full.len());
        for (a, b) in vals.iter().zip(full) {
            assert_eq!(a.sector, b.sector);
            assert_eq!(a.values.len(), b.values.len());
            for (x, y) in a.values.iter().zip(&b.values) {
                assert!((x - y).abs() <= tol, "{x} vs {y}");
            }
        }
    };

    let svd_vals_spectra = svd_vals(
        &mut dense_executor,
        &bound_tensor_ref!(Arc::new(rule), &general),
    )
    .unwrap();
    let svd_compact_spectra = svd_compact(
        &mut dense_executor,
        &bound_tensor_ref!(Arc::new(rule), &general),
    )
    .unwrap()
    .singular_values;
    assert_real_close(&svd_vals_spectra, &svd_compact_spectra);

    let eigh_vals_spectra = eigh_vals(
        &mut dense_executor,
        &bound_tensor_ref!(Arc::new(rule), &hermitian),
    )
    .unwrap();
    let eigh_full_spectra = eigh_full(
        &mut dense_executor,
        &bound_tensor_ref!(Arc::new(rule), &hermitian),
    )
    .unwrap()
    .eigenvalues;
    assert_real_close(&eigh_vals_spectra, &eigh_full_spectra);

    let eig_vals_spectra = eig_vals(
        &mut dense_executor,
        &bound_tensor_ref!(Arc::new(rule), &general),
    )
    .unwrap();
    let eig_full_spectra = eig_full(
        &mut dense_executor,
        &bound_tensor_ref!(Arc::new(rule), &general),
    )
    .unwrap()
    .eigenvalues;
    assert_eq!(eig_vals_spectra.len(), eig_full_spectra.len());
    for (a, b) in eig_vals_spectra.iter().zip(&eig_full_spectra) {
        assert_eq!(a.sector, b.sector);
        assert_eq!(a.values.len(), b.values.len());
        for (x, y) in a.values.iter().zip(&b.values) {
            assert!((x - y).norm() <= tol, "{x} vs {y}");
        }
    }
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

fn lowered_z2_binding<const NOUT: usize, const NIN: usize>(
    tensor: &TensorMap<f64, NOUT, NIN>,
) -> BoundDynamicFusionMapSpace<Z2FusionRule> {
    let provider = Arc::new(Z2FusionRule);
    let raw = dyn_space_of(tensor).unwrap();
    let hom = raw.homspace().clone();
    // Why not caller-supplied per-tree shapes: the #586 sweep narrowed the
    // shape-admission bridge to tenet-tensors; the kept public installer
    // derives the identical blocks from the final homspace's leg
    // degeneracies for these dense-leg fixtures.
    BoundDynamicFusionMapSpace::from_final_homspace_multiplicity_free_lowered(provider, hom)
        .unwrap()
}

#[test]
fn ordinary_factorizations_and_composition_inherit_lowered_layout_strategy() {
    // What: cold compact SVD, compact QR, full EIGH, adjoint, and factor
    // composition all retain the ordinary built-in layout-build strategy.
    let tensor = hermitian_test_tensor(&Z2FusionRule, &[SectorId::new(0), SectorId::new(1)]);
    let bound = lowered_z2_binding(&tensor);
    let expert = BoundDynamicFusionMapSpace::bind_multiplicity_free(
        dyn_space_of(&tensor).unwrap(),
        Arc::new(Z2FusionRule),
    )
    .unwrap();
    let malformed = FusionTreeHomSpace::new(
        FusionProductSpace::new([SectorLeg::new([(SectorId::new(99), 1)], false)]),
        FusionProductSpace::new([]),
    );
    assert!(expert.prime_derived_homspace(&malformed).is_ok());
    let input = BoundDynamicTensorRef::try_new(&bound, tensor.data()).unwrap();
    let mut dense = tenet_dense::DefaultDenseExecutor::new();

    let svd = svd_compact_dyn(&mut dense, &input).unwrap();
    for factor in [svd.u(), svd.s(), svd.vh()] {
        assert!(factor.space().prime_derived_homspace(&malformed).is_err());
    }
    let (q, r) = qr_compact_dyn(&mut dense, &input).unwrap();
    assert!(q.space().prime_derived_homspace(&malformed).is_err());
    assert!(r.space().prime_derived_homspace(&malformed).is_err());
    let eigh = eigh_full_dyn(&mut dense, &input).unwrap();
    assert!(eigh.v().space().prime_derived_homspace(&malformed).is_err());

    let adjoint = crate::factorize::adjoint_bound_factor(svd.u()).unwrap();
    assert!(adjoint.space().prime_derived_homspace(&malformed).is_err());
    let mut context = default_context();
    let composed = crate::compose::compose_bound_dyn(&mut context, svd.u(), svd.s()).unwrap();
    assert!(composed.space().prime_derived_homspace(&malformed).is_err());
}

#[test]
fn derived_matrix_functions_inherit_the_exact_provider_arc() {
    // What: every migrated owned result retains the input authority allocation.
    let tensor = hermitian_test_tensor(&Z2FusionRule, &[SectorId::new(0), SectorId::new(1)]);
    let provider = Arc::new(Z2FusionRule);
    let bound = BoundDynamicFusionMapSpace::bind_multiplicity_free(
        dyn_space_of(&tensor).unwrap(),
        Arc::clone(&provider),
    )
    .unwrap();
    let input = BoundDynamicTensorRef::try_new(&bound, tensor.data()).unwrap();
    let mut dense = tenet_dense::DefaultDenseExecutor::new();
    let mut context = default_context();

    let (w_left, p_left) = left_polar_dyn(&mut dense, &mut context, &input).unwrap();
    let (p_right, w_right) = right_polar_dyn(&mut dense, &mut context, &input).unwrap();
    let inverse = inv_dyn(&mut dense, &mut context, &input).unwrap();
    let pseudo_inverse = pinv_dyn(&mut dense, &mut context, &input, 1.0e-13).unwrap();

    for factor in [
        &w_left,
        &p_left,
        &p_right,
        &w_right,
        &inverse,
        &pseudo_inverse,
    ] {
        assert!(Arc::ptr_eq(factor.space().provider_arc(), &provider));
    }
}

#[test]
fn adjoint_composition_gives_the_identity_on_the_bond() {
    let rule = SU2FusionRule;
    let tensor = tsvd_test_tensor(
        &rule,
        &[
            SU2Irrep::from_twice_spin(0).sector_id(),
            SU2Irrep::from_twice_spin(1).sector_id(),
        ],
    );
    let mut dense_executor = tenet_dense::DefaultDenseExecutor::new();
    let (q, _) = qr_compact(
        &mut dense_executor,
        &bound_tensor_ref!(Arc::new(rule), &tensor),
    )
    .unwrap();
    let qh = tenet_tensors::adjoint(&rule, &q).unwrap();
    let mut context = default_context();
    let identity = crate::compose::compose(&mut context, &rule, &qh, &q).unwrap();
    assert_identity_matrices(&dense_sector_matrices(1, &identity));
}

#[test]
fn exp_of_a_hermitian_tensor_inverts_under_negation() {
    let rule = Z2FusionRule;
    let raw = hermitian_test_tensor(&rule, &[SectorId::new(0), SectorId::new(1)]);
    // Keep the spectrum modest so exp(t) exp(-t) stays well conditioned.
    let tensor = TensorMap::<f64, 2, 2>::from_vec_with_fusion_space(
        raw.data().iter().map(|value| 0.1 * value).collect(),
        raw.fusion_space().unwrap().as_ref().clone(),
    )
    .unwrap();
    let negated = TensorMap::<f64, 2, 2>::from_vec_with_fusion_space(
        tensor.data().iter().map(|value| -value).collect(),
        tensor.fusion_space().unwrap().as_ref().clone(),
    )
    .unwrap();
    let mut dense_executor = tenet_dense::DefaultDenseExecutor::new();
    let mut context = default_context();

    let forward = exp(
        &mut dense_executor,
        &mut context,
        &bound_tensor_ref!(Arc::new(rule), &tensor),
    )
    .unwrap();
    let backward = exp(
        &mut dense_executor,
        &mut context,
        &bound_tensor_ref!(Arc::new(rule), &negated),
    )
    .unwrap();
    let identity = crate::compose::compose(&mut context, &rule, &forward, &backward).unwrap();
    assert_identity_matrices(&dense_sector_matrices(2, &identity));
}

#[test]
fn pinv_satisfies_the_moore_penrose_identity() {
    let rule = Z2FusionRule;
    let sectors = [SectorId::new(0), SectorId::new(1)];
    let degeneracy = 2usize;
    let leg = || SectorLeg::new(sectors.iter().map(|&sector| (sector, degeneracy)), false);
    let leg_dim = sectors.len() * degeneracy;
    let hom = FusionTreeHomSpace::new(
        FusionProductSpace::new([leg(), leg()]),
        FusionProductSpace::new([leg()]),
    );
    let key_count = hom.fusion_tree_keys(&rule).len();
    let space = FusionTensorMapSpace::from_degeneracy_shapes_coupled(
        TensorMapSpace::<2, 1>::from_dims([leg_dim, leg_dim], [leg_dim]).unwrap(),
        hom,
        &rule,
        vec![vec![degeneracy; 3]; key_count],
    )
    .unwrap();
    let len = space.required_len().unwrap();
    let tensor = TensorMap::<f64, 2, 1>::from_vec_with_fusion_space(
        (0..len).map(|i| ((i * 3 + 2) % 11) as f64 - 5.0).collect(),
        space,
    )
    .unwrap();
    let mut dense_executor = tenet_dense::DefaultDenseExecutor::new();
    let mut context = default_context();

    let plus = pinv(
        &mut dense_executor,
        &mut context,
        &bound_tensor_ref!(Arc::new(rule), &tensor),
        1e-12,
    )
    .unwrap();
    let tp = crate::compose::compose(&mut context, &rule, &tensor, &plus).unwrap();
    let tpt = crate::compose::compose(&mut context, &rule, &tp, &tensor).unwrap();
    for (index, (lhs, rhs)) in tpt.data().iter().zip(tensor.data()).enumerate() {
        assert!(
            (lhs - rhs).abs() < 1e-8,
            "Moore-Penrose violated at raw position {index}: {lhs} != {rhs}"
        );
    }
}

#[test]
fn inv_composes_to_the_identity() {
    let rule = Z2FusionRule;
    let tensor = hermitian_test_tensor(&rule, &[SectorId::new(0), SectorId::new(1)]);
    assert!(tensor
        .structure()
        .coupled_sector_regions(2)
        .unwrap()
        .is_some());
    let expected_sectors = dense_sector_matrices(2, &tensor).len();
    let mut dense_executor = SolveCallSpy::default();
    let mut context = default_context();
    let inverse = inv(
        &mut dense_executor,
        &mut context,
        &bound_tensor_ref!(Arc::new(rule), &tensor),
    )
    .unwrap();
    assert_eq!(dense_executor.solve_calls, expected_sectors);
    let identity = crate::compose::compose(&mut context, &rule, &tensor, &inverse).unwrap();
    assert_identity_matrices(&dense_sector_matrices(2, &identity));
}

#[test]
fn solve_left_uses_one_direct_solve_per_sector_for_rectangular_rhs() {
    // What: A \ B accepts multiple RHS columns and never forms an inverse.
    let divisor = u1_block_endomorphism(&[(0, 2, vec![2.0_f64, 0.0, 1.0, 3.0]), (1, 1, vec![4.0])]);
    let rhs = u1_block_map(&[
        (0, 2, 3, vec![4.0_f64, 6.0, 10.0, 12.0, 16.0, 18.0]),
        (1, 1, 2, vec![28.0, 32.0]),
    ]);
    let divisor_provider = Arc::new(U1FusionRule);
    let rhs_provider = Arc::new(U1FusionRule);
    let divisor = bound_tensor(Arc::clone(&divisor_provider), &divisor);
    let rhs = bound_tensor(Arc::clone(&rhs_provider), &rhs);
    let mut dense = SolveCallSpy::default();

    let solved = solve_left_direct_dyn(
        &mut dense,
        &divisor.as_ref().dynamic(),
        &rhs.as_ref().dynamic(),
    )
    .unwrap();

    assert!(Arc::ptr_eq(
        solved.space().provider_arc(),
        &divisor_provider
    ));
    assert!(!Arc::ptr_eq(solved.space().provider_arc(), &rhs_provider));
    let solved: BoundTensorMap<_, _, 1, 1> = typed_from_bound_factor(solved).unwrap();
    let sectors = dense_sector_matrices(1, solved.tensor());
    let even = sectors
        .iter()
        .find(|(sector, _, _, _)| *sector == U1Irrep::new(0).sector_id())
        .unwrap();
    let odd = sectors
        .iter()
        .find(|(sector, _, _, _)| *sector == U1Irrep::new(1).sector_id())
        .unwrap();
    assert_eq!((even.1, even.2), (2, 3));
    assert_eq!(even.3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    assert_eq!((odd.1, odd.2), (1, 2));
    assert_eq!(odd.3, vec![7.0, 8.0]);
    assert_eq!(dense.solve_calls, 2);
    // What: the backend wrote the first final sector in the returned payload;
    // no owned solution twin followed by a full-result copy can satisfy this
    // pointer identity.
    assert_eq!(dense.destination_ptrs[0], solved.data().as_ptr() as usize);
}

#[test]
fn solve_left_preserves_complex_values_without_adjointing() {
    // What: A \ B is the ordinary complex linear solve, not a Hermitian route.
    let divisor_data = [
        Complex64::new(2.0, 1.0),
        Complex64::new(0.0, 0.5),
        Complex64::new(1.0, -1.0),
        Complex64::new(3.0, 0.0),
    ];
    let expected = [
        Complex64::new(1.0, 2.0),
        Complex64::new(-1.0, 0.5),
        Complex64::new(0.25, -0.75),
        Complex64::new(2.0, 1.0),
    ];
    let mut rhs_data = vec![Complex64::zero(); 4];
    for col in 0..2 {
        for row in 0..2 {
            rhs_data[row + 2 * col] = divisor_data[row] * expected[2 * col]
                + divisor_data[row + 2] * expected[1 + 2 * col];
        }
    }
    let divisor = u1_block_endomorphism(&[(0, 2, divisor_data.to_vec())]);
    let rhs = u1_block_map(&[(0, 2, 2, rhs_data)]);
    let divisor = bound_tensor(Arc::new(U1FusionRule), &divisor);
    let rhs = bound_tensor(Arc::new(U1FusionRule), &rhs);
    let mut dense = SolveCallSpy::default();

    let solved = solve_left_direct_dyn(
        &mut dense,
        &divisor.as_ref().dynamic(),
        &rhs.as_ref().dynamic(),
    )
    .unwrap();

    for (&actual, expected) in solved.data().iter().zip(expected) {
        assert!((actual - expected).norm() < 1.0e-12);
    }
    assert_eq!(dense.solve_calls, 1);
}

#[test]
fn solve_left_discards_an_output_when_a_later_sector_fails() {
    // What: an earlier successful sector cannot publish a partial solution.
    let divisor = u1_block_endomorphism(&[(0, 1, vec![2.0_f64]), (1, 1, vec![3.0])]);
    let rhs = u1_block_map(&[(0, 1, 1, vec![4.0_f64]), (1, 1, 1, vec![9.0_f64])]);
    let divisor = bound_tensor(Arc::new(U1FusionRule), &divisor);
    let rhs = bound_tensor(Arc::new(U1FusionRule), &rhs);
    let mut dense = FailSecondSolve::default();

    let error = solve_left_direct_dyn(
        &mut dense,
        &divisor.as_ref().dynamic(),
        &rhs.as_ref().dynamic(),
    )
    .unwrap_err();

    assert!(matches!(
        error,
        OperationError::Dense(DenseError::Backend {
            op: "solve_into",
            ..
        })
    ));
    assert_eq!(dense.solve_calls, 2);
}

#[test]
fn solve_left_validates_spaces_before_backend_execution() {
    // What: codomain mismatch and a rectangular divisor are structural
    // failures, not backend calls.
    let divisor = u1_block_endomorphism(&[(0, 2, vec![1.0_f64, 0.0, 0.0, 1.0])]);
    let wrong_codomain = u1_cross_space_map::<f64>(&[(0, 3)], &[(0, 1)]);
    let divisor = bound_tensor(Arc::new(U1FusionRule), &divisor);
    let wrong_codomain = bound_tensor(Arc::new(U1FusionRule), &wrong_codomain);
    let mut dense = RejectExecutorCalls;
    let error = solve_left_direct_dyn(
        &mut dense,
        &divisor.as_ref().dynamic(),
        &wrong_codomain.as_ref().dynamic(),
    )
    .unwrap_err();
    assert!(matches!(
        error,
        OperationError::UnsupportedTensorContractScope {
            message: "solve requires equal divisor and right-hand-side codomains"
        }
    ));

    let rectangular = u1_cross_space_map::<f64>(&[(0, 2)], &[(0, 3)]);
    let rhs = u1_cross_space_map::<f64>(&[(0, 2)], &[(0, 1)]);
    let rectangular = bound_tensor(Arc::new(U1FusionRule), &rectangular);
    let rhs = bound_tensor(Arc::new(U1FusionRule), &rhs);
    let error = solve_left_direct_dyn(
        &mut dense,
        &rectangular.as_ref().dynamic(),
        &rhs.as_ref().dynamic(),
    )
    .unwrap_err();
    assert!(matches!(
        error,
        OperationError::UnsupportedTensorContractScope {
            message: "solve requires an isomorphic divisor codomain and domain"
        }
    ));

    let incomplete = u1_cross_space_map::<f64>(&[(0, 1), (1, 1)], &[(0, 1)]);
    let rhs = u1_cross_space_map::<f64>(&[(0, 1), (1, 1)], &[(0, 1)]);
    let incomplete = bound_tensor(Arc::new(U1FusionRule), &incomplete);
    let rhs = bound_tensor(Arc::new(U1FusionRule), &rhs);
    let error = solve_left_direct_dyn(
        &mut dense,
        &incomplete.as_ref().dynamic(),
        &rhs.as_ref().dynamic(),
    )
    .unwrap_err();
    assert!(matches!(
        error,
        OperationError::UnsupportedTensorContractScope {
            message: "solve requires an isomorphic divisor codomain and domain"
        }
    ));
}

#[test]
fn solve_left_preserves_dense_singularity_and_capability_errors() {
    // What: solve reports the backend's stable error class without falling
    // back to SVD, inverse formation, or a pseudoinverse.
    let divisor = u1_block_endomorphism(&[(0, 2, vec![1.0_f64, 0.0, 0.0, 0.0])]);
    let rhs = u1_block_map(&[(0, 2, 1, vec![1.0_f64, 1.0])]);
    let divisor = bound_tensor(Arc::new(U1FusionRule), &divisor);
    let rhs = bound_tensor(Arc::new(U1FusionRule), &rhs);
    let error = solve_left_direct_dyn(
        &mut tenet_dense::DefaultDenseExecutor::new(),
        &divisor.as_ref().dynamic(),
        &rhs.as_ref().dynamic(),
    )
    .unwrap_err();
    assert!(matches!(
        error,
        OperationError::Dense(DenseError::NumericalFailure {
            op: "solve_into",
            ..
        })
    ));

    let divisor = u1_block_endomorphism(&[(0, 1, vec![2.0_f64])]);
    let rhs = u1_block_map(&[(0, 1, 1, vec![1.0_f64])]);
    let divisor = bound_tensor(Arc::new(U1FusionRule), &divisor);
    let rhs = bound_tensor(Arc::new(U1FusionRule), &rhs);
    let error = solve_left_direct_dyn(
        &mut RejectExecutorCalls,
        &divisor.as_ref().dynamic(),
        &rhs.as_ref().dynamic(),
    )
    .unwrap_err();
    assert!(matches!(
        error,
        OperationError::Dense(DenseError::Unsupported {
            op: "solve_into",
            ..
        })
    ));
}

#[test]
fn solve_left_direct_into_rejects_foreign_authority_and_wrong_output_before_execution() {
    // What: caller-admitted output metadata is an input contract, not a hint.
    let divisor = u1_block_endomorphism(&[(0, 1, vec![2.0_f64])]);
    let rhs = u1_block_map(&[(0, 1, 1, vec![1.0_f64])]);
    let provider = Arc::new(U1FusionRule);
    let divisor = bound_tensor(Arc::clone(&provider), &divisor);
    let rhs = bound_tensor(Arc::new(U1FusionRule), &rhs);
    let expected = FusionTreeHomSpace::new(
        divisor.space().space().homspace().domain().clone(),
        rhs.space().space().homspace().domain().clone(),
    );

    let foreign = BoundDynamicFusionMapSpace::from_final_homspace_multiplicity_free(
        Arc::new(U1FusionRule),
        expected.clone(),
    )
    .unwrap();
    let error = solve_left_direct_into_dyn(
        &mut RejectExecutorCalls,
        &divisor.as_ref().dynamic(),
        &rhs.as_ref().dynamic(),
        foreign,
    )
    .unwrap_err();
    assert!(matches!(error, OperationError::StructureMismatch { .. }));

    let wrong = BoundDynamicFusionMapSpace::from_final_homspace_multiplicity_free(
        provider,
        FusionTreeHomSpace::new(
            divisor.space().space().homspace().domain().clone(),
            FusionProductSpace::new([
                SectorLeg::new([(U1Irrep::new(0).sector_id(), 1)], false),
                SectorLeg::new([(U1Irrep::new(0).sector_id(), 1)], false),
            ]),
        ),
    )
    .unwrap();
    let error = solve_left_direct_into_dyn(
        &mut RejectExecutorCalls,
        &divisor.as_ref().dynamic(),
        &rhs.as_ref().dynamic(),
        wrong,
    )
    .unwrap_err();
    assert!(matches!(error, OperationError::StructureMismatch { .. }));
}

#[test]
#[expect(
    clippy::arc_with_non_send_sync,
    reason = "the checked Generic API requires Arc identity while Cell is a single-threaded call spy"
)]
fn pinv_direct_into_rejects_foreign_authority_and_wrong_output_before_execution() {
    // What: the checked-Generic pinv seam treats its admitted destination as
    // a sealed input, before it allocates staging or invokes SVD/GEMM.
    let (base, data) = generic_factorization_input();
    let provider = Arc::new(LateGenericSpy {
        rule: FactorGenericRule,
        fail_at: usize::MAX,
        calls: Cell::new(0),
    });
    let source =
        BoundDynamicFusionMapSpace::bind_generic(base.space().clone(), Arc::clone(&provider))
            .unwrap();
    let input = BoundDynamicTensorRef::try_new(&source, &data).unwrap();
    let expected = FusionTreeHomSpace::new(
        source.space().homspace().domain().clone(),
        source.space().homspace().codomain().clone(),
    );

    let foreign = BoundDynamicFusionMapSpace::from_final_homspace_generic_checked(
        Arc::new(LateGenericSpy {
            rule: FactorGenericRule,
            fail_at: usize::MAX,
            calls: Cell::new(0),
        }),
        expected.clone(),
    )
    .unwrap();
    let error = pinv_direct_into_dyn(&mut RejectExecutorCalls, &input, foreign, 0.0).unwrap_err();
    assert!(matches!(error, OperationError::StructureMismatch { .. }));

    let wrong = BoundDynamicFusionMapSpace::from_final_homspace_generic_checked(
        provider,
        FusionTreeHomSpace::new(
            source.space().homspace().domain().clone(),
            FusionProductSpace::new([
                SectorLeg::new([(SectorId::new(1), 1)], false),
                SectorLeg::new([(SectorId::new(1), 1)], false),
            ]),
        ),
    )
    .unwrap();
    let error = pinv_direct_into_dyn(&mut RejectExecutorCalls, &input, wrong, 0.0).unwrap_err();
    assert!(matches!(error, OperationError::StructureMismatch { .. }));
}

#[test]
fn solve_left_direct_into_rejects_late_tree_route_before_execution() {
    // What: an admitted output with the right identity and HomSpace still
    // cannot reinterpret a different coupled-tree basis.
    let source = hermitian_test_tensor(&Z2FusionRule, &[SectorId::new(0), SectorId::new(1)]);
    let output = reversed_coupled_tree_basis_copy(&Z2FusionRule, &source);
    let provider = Arc::new(Z2FusionRule);
    let divisor = bound_tensor(Arc::clone(&provider), &source);
    let rhs = bound_tensor(Arc::clone(&provider), &source);
    let output_space = BoundDynamicFusionMapSpace::bind_multiplicity_free(
        dyn_space_of(&output).unwrap(),
        Arc::clone(&provider),
    )
    .unwrap();
    let mut dense = SolveCallSpy::default();

    let error = solve_left_direct_into_dyn(
        &mut dense,
        &divisor.as_ref().dynamic(),
        &rhs.as_ref().dynamic(),
        output_space,
    )
    .unwrap_err();

    assert!(
        matches!(
            error,
            OperationError::UnsupportedTensorContractScope {
                message: "solve coupled-sector tree bases are incompatible"
            }
        ),
        "{error:?}"
    );
    assert_eq!(dense.solve_calls, 0);
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

fn assert_disjoint_null_spaces_keep_structural_directions<D: FactorScalar>() {
    let assert_identity = |data: &[D], order: usize| {
        for col in 0..order {
            for row in 0..order {
                let expected = if row == col { 1.0 } else { 0.0 };
                assert!((data[row + order * col].widen_complex().re - expected).abs() < 1e-12);
                assert!(data[row + order * col].widen_complex().im.abs() < 1e-12);
            }
        }
    };
    let provider = Arc::new(U1FusionRule);
    let tensor = u1_cross_space_map::<D>(&[(1, 2)], &[(0, 3)]);
    let input = bound_tensor(Arc::clone(&provider), &tensor);

    crate::factorize::reset_factor_buffer_build_counts_for_test();
    let left = left_null(&mut RejectExecutorCalls, &input.as_ref()).unwrap();
    assert_eq!(
        crate::factorize::factor_buffer_build_counts_for_test(),
        (1, 0)
    );
    assert_eq!(left.structure().block_count(), 1);
    assert_eq!(left.structure().block(0).unwrap().shape(), &[2, 2]);
    assert_eq!(left.data().len(), 4);
    assert_identity(left.data(), 2);
    assert!(Arc::ptr_eq(left.space().provider_arc(), &provider));

    crate::factorize::reset_factor_buffer_build_counts_for_test();
    let right = right_null(&mut RejectExecutorCalls, &input.as_ref()).unwrap();
    assert_eq!(
        crate::factorize::factor_buffer_build_counts_for_test(),
        (0, 1)
    );
    assert_eq!(right.structure().block_count(), 1);
    assert_eq!(right.structure().block(0).unwrap().shape(), &[3, 3]);
    assert_eq!(right.data().len(), 9);
    assert_identity(right.data(), 3);
    assert!(Arc::ptr_eq(right.space().provider_arc(), &provider));
}

#[test]
fn disjoint_null_spaces_keep_all_structural_directions_without_dense_work() {
    // What: a zero map between disjoint supports has the whole codomain/domain
    // as its left/right null space and builds only the requested factor.
    assert_disjoint_null_spaces_keep_structural_directions::<f64>();
    assert_disjoint_null_spaces_keep_structural_directions::<Complex64>();
}

#[test]
fn unmatched_null_sectors_coexist_with_a_full_rank_matched_sector() {
    // What: matched full-rank directions disappear while side-only sectors
    // survive as identity bases.
    let provider = Arc::new(U1FusionRule);
    let tensor = u1_cross_space_map::<f64>(&[(0, 1), (1, 2)], &[(0, 1), (2, 3)]);
    let input = bound_tensor(Arc::clone(&provider), &tensor);

    let mut dense = SvdCallSpy::default();
    let left = left_null(&mut dense, &input.as_ref()).unwrap();
    assert_eq!(dense.svd_calls, 1);
    assert_eq!(left.structure().block_count(), 1);
    assert_eq!(left.structure().block(0).unwrap().shape(), &[2, 2]);

    let mut dense = SvdCallSpy::default();
    let right = right_null(&mut dense, &input.as_ref()).unwrap();
    assert_eq!(dense.svd_calls, 1);
    assert_eq!(right.structure().block_count(), 1);
    assert_eq!(right.structure().block(0).unwrap().shape(), &[3, 3]);
}

#[test]
fn null_space_second_sector_failure_builds_no_factor() {
    // What: all dense work finishes before the one requested factor is built.
    let provider = Arc::new(U1FusionRule);
    let tensor = u1_cross_space_map::<f64>(&[(0, 2), (1, 2)], &[(0, 2), (1, 2)]);
    let before = tensor.data().to_vec();
    let input = bound_tensor(provider, &tensor);

    crate::factorize::reset_factor_buffer_build_counts_for_test();
    assert!(matches!(
        left_null(&mut FailSecondSvd::default(), &input.as_ref()),
        Err(OperationError::Dense(_))
    ));
    assert_eq!(
        crate::factorize::factor_buffer_build_counts_for_test(),
        (0, 0)
    );

    crate::factorize::reset_factor_buffer_build_counts_for_test();
    assert!(matches!(
        right_null(&mut FailSecondSvd::default(), &input.as_ref()),
        Err(OperationError::Dense(_))
    ));
    assert_eq!(
        crate::factorize::factor_buffer_build_counts_for_test(),
        (0, 0)
    );
    assert_eq!(tensor.data(), before);
}

#[test]
#[expect(
    clippy::type_complexity,
    reason = "the test table pairs codomain and domain sector fixtures directly"
)]
fn inv_rejects_nonisomorphic_spaces_before_dense_execution() {
    // What: neither a square stored-sector intersection nor equal total
    // dimension substitutes for complete coupled-sector isomorphism.
    let cases: &[(&[(i32, usize)], &[(i32, usize)])] = &[
        (&[(0, 1), (1, 1)], &[(0, 1)]),
        (&[(0, 1), (1, 1)], &[(0, 1), (2, 1)]),
    ];
    for &(codomain, domain) in cases {
        let tensor = u1_cross_space_map::<f64>(codomain, domain);
        let mut dense = RejectExecutorCalls;
        let mut context = TensorContractFusionExecutionContext::<f64, RuleIdentity>::default();
        let error = inv(
            &mut dense,
            &mut context,
            &bound_tensor_ref!(Arc::new(U1FusionRule), &tensor),
        )
        .unwrap_err();
        assert!(matches!(
            error,
            OperationError::UnsupportedTensorContractScope {
                message: "inv requires isomorphic codomain and domain"
            }
        ));
    }
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

fn u1_block_map<D>(blocks: &[(i32, usize, usize, Vec<D>)]) -> TensorMap<D, 1, 1>
where
    D: Copy + Zero,
{
    let codomain = SectorLeg::new(
        blocks
            .iter()
            .map(|(charge, rows, _, _)| (U1Irrep::new(*charge).sector_id(), *rows)),
        false,
    );
    let domain = SectorLeg::new(
        blocks
            .iter()
            .map(|(charge, _, cols, _)| (U1Irrep::new(*charge).sector_id(), *cols)),
        false,
    );
    let homspace = FusionTreeHomSpace::new(
        FusionProductSpace::new([codomain]),
        FusionProductSpace::new([domain]),
    );
    let rows = blocks.iter().map(|(_, rows, _, _)| rows).sum();
    let cols = blocks.iter().map(|(_, _, cols, _)| cols).sum();
    let shapes = homspace
        .fusion_tree_keys(&U1FusionRule)
        .iter()
        .map(|key| {
            let coupled = key.codomain_tree().coupled();
            let (_, rows, cols, data) = blocks
                .iter()
                .find(|(charge, _, _, _)| U1Irrep::new(*charge).sector_id() == coupled)
                .unwrap();
            assert_eq!(data.len(), rows * cols);
            vec![*rows, *cols]
        })
        .collect::<Vec<_>>();
    let space = FusionTensorMapSpace::from_degeneracy_shapes_coupled(
        TensorMapSpace::<1, 1>::from_dims([rows], [cols]).unwrap(),
        homspace,
        &U1FusionRule,
        shapes,
    )
    .unwrap();
    TensorMap::from_block_fn_with_fusion_space(space, D::zero(), |key, indices| {
        let BlockKey::FusionTree(tree) = key else {
            return D::zero();
        };
        let coupled = tree.codomain_tree().coupled();
        let (_, rows, _, data) = blocks
            .iter()
            .find(|(charge, _, _, _)| U1Irrep::new(*charge).sector_id() == coupled)
            .unwrap();
        data[indices[0] + rows * indices[1]]
    })
    .unwrap()
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

fn scalar_block_endomorphism<R>(rule: &R, blocks: &[(SectorId, f64)]) -> TensorMap<f64, 1, 1>
where
    R: MultiplicityFreeFusionRule,
{
    let leg = SectorLeg::new(blocks.iter().map(|&(sector, _)| (sector, 1)), false);
    let homspace = FusionTreeHomSpace::new(
        FusionProductSpace::new([leg.clone()]),
        FusionProductSpace::new([leg]),
    );
    let shapes = vec![vec![1, 1]; homspace.fusion_tree_keys(rule).len()];
    let space = FusionTensorMapSpace::from_degeneracy_shapes_coupled(
        TensorMapSpace::<1, 1>::from_dims([blocks.len()], [blocks.len()]).unwrap(),
        homspace,
        rule,
        shapes,
    )
    .unwrap();
    TensorMap::from_block_fn_with_fusion_space(space, 0.0, |key, _| {
        let BlockKey::FusionTree(tree) = key else {
            return 0.0;
        };
        let sector = tree.codomain_tree().coupled();
        blocks
            .iter()
            .find_map(|&(candidate, value)| (candidate == sector).then_some(value))
            .unwrap()
    })
    .unwrap()
}

fn assert_scale_separated_inverse<R>(rule: R, sectors: [SectorId; 2])
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64> + TreeTransformRuleCacheKey,
{
    let values = [3.0, 1e-14];
    let tensor =
        scalar_block_endomorphism(&rule, &[(sectors[0], values[0]), (sectors[1], values[1])]);
    let mut dense = tenet_dense::DefaultDenseExecutor::new();
    let mut context = TensorContractFusionExecutionContext::<f64, R::Key>::default();
    let inverse = inv(
        &mut dense,
        &mut context,
        &bound_tensor_ref!(Arc::new(rule), &tensor),
    )
    .unwrap();

    for (sector, value) in sectors.into_iter().zip(values) {
        assert!((value * scalar_block(inverse.tensor(), sector) - 1.0).abs() < 1e-12);
    }
}

#[test]
fn inv_uses_each_u1_sector_scale_for_f64_rank_and_value() {
    // What: an invertible scalar sector remains invertible regardless of another sector's scale.
    for dominant in [1.0, 1e12] {
        let tensor = u1_block_endomorphism(&[(0, 1, vec![dominant]), (1, 1, vec![1e-14_f64])]);
        let mut dense = tenet_dense::DefaultDenseExecutor::new();
        let mut context = TensorContractFusionExecutionContext::<f64, RuleIdentity>::default();

        let inverse = inv(
            &mut dense,
            &mut context,
            &bound_tensor_ref!(Arc::new(U1FusionRule), &tensor),
        )
        .unwrap();

        let dominant_inverse = scalar_u1_block(inverse.tensor(), 0);
        let small_inverse = scalar_u1_block(inverse.tensor(), 1);
        assert!((dominant * dominant_inverse - 1.0).abs() < 1e-12);
        assert!((1e-14 * small_inverse - 1.0).abs() < 1e-12);
        assert!((small_inverse / 1e14 - 1.0).abs() < 1e-12);
    }
}

#[test]
fn inv_uses_sector_local_scale_for_su2_fz2_and_product_rules() {
    // What: sector-local rank and inversion apply uniformly to non-Abelian,
    // fermionic, and nested product rules rather than only the U1 fixture.
    assert_scale_separated_inverse(
        SU2FusionRule,
        [
            SU2Irrep::from_twice_spin(0).sector_id(),
            SU2Irrep::from_twice_spin(1).sector_id(),
        ],
    );
    assert_scale_separated_inverse(
        FermionParityFusionRule,
        [SectorId::new(0), SectorId::new(1)],
    );

    let fz2_u1 = product_fusion_rule(FermionParityFusionRule, U1FusionRule);
    let product_sectors = [
        fz2_u1.encode_sector(SectorId::new(0), U1Irrep::new(0).sector_id()),
        fz2_u1.encode_sector(SectorId::new(1), U1Irrep::new(1).sector_id()),
    ];
    let fz2_u1_su2 = product_fusion_rule(fz2_u1, SU2FusionRule);
    let nested_sectors = [
        fz2_u1_su2.encode_sector(product_sectors[0], SU2Irrep::from_twice_spin(0).sector_id()),
        fz2_u1_su2.encode_sector(product_sectors[1], SU2Irrep::from_twice_spin(1).sector_id()),
    ];
    assert_scale_separated_inverse(fz2_u1_su2, nested_sectors);
}

#[test]
fn inv_uses_each_u1_sector_scale_for_phased_c64_values() {
    // What: complex phases do not couple numerical-rank decisions across sectors.
    let large = Complex64::from_polar(1e8, 0.37);
    let small = Complex64::from_polar(1e-14, -0.91);
    let tensor = u1_block_endomorphism(&[(0, 1, vec![large]), (1, 1, vec![small])]);
    let mut dense = tenet_dense::DefaultDenseExecutor::new();
    let mut context = TensorContractFusionExecutionContext::<Complex64, RuleIdentity>::default();

    let inverse = inv(
        &mut dense,
        &mut context,
        &bound_tensor_ref!(Arc::new(U1FusionRule), &tensor),
    )
    .unwrap();

    assert!(
        (large * scalar_u1_block(inverse.tensor(), 0) - Complex64::new(1.0, 0.0)).norm() < 1e-12
    );
    assert!(
        (small * scalar_u1_block(inverse.tensor(), 1) - Complex64::new(1.0, 0.0)).norm() < 1e-12
    );
}

#[test]
fn inv_solves_padded_u1_sectors_and_matches_the_dense_oracle() {
    // What: a legal expert layout uses the same sector solves and restores each
    // dense inverse to the destination block layout.
    let tensor = u1_block_endomorphism(&[
        (0, 2, vec![2.0_f64, 1.0, 3.0, 4.0]),
        (1, 2, vec![1.0_f64, 2.0, 0.0, 3.0]),
    ]);
    let padded = padded_copy(&U1FusionRule, &tensor);
    assert!(padded
        .structure()
        .coupled_sector_regions(1)
        .unwrap()
        .is_none());
    let mut dense = SolveCallSpy::default();
    let mut context = default_context();

    let inverse = inv(
        &mut dense,
        &mut context,
        &bound_tensor_ref!(Arc::new(U1FusionRule), &padded),
    )
    .unwrap();

    let expected = [
        (U1Irrep::new(0).sector_id(), vec![0.8, -0.2, -0.6, 0.4]),
        (
            U1Irrep::new(1).sector_id(),
            vec![1.0, -2.0 / 3.0, 0.0, 1.0 / 3.0],
        ),
    ];
    for (sector, _, _, values) in dense_sector_matrices(1, inverse.tensor()) {
        let oracle = expected
            .iter()
            .find_map(|(candidate, values)| (*candidate == sector).then_some(values))
            .unwrap();
        for (&actual, &expected) in values.iter().zip(oracle) {
            assert!((actual - expected).abs() < 1.0e-12);
        }
    }
    assert_eq!(dense.solve_calls, 2);
}

#[test]
fn inv_reorders_a_complete_expert_tree_grid_by_key() {
    // What: an expert layout's block order cannot transpose matrix axes or
    // replace fusion-tree identity as the inverse routing authority.
    let rule = Z2FusionRule;
    let leg = || SectorLeg::new([(SectorId::new(0), 2), (SectorId::new(1), 2)], false);
    let homspace = FusionTreeHomSpace::new(
        FusionProductSpace::new([leg(), leg()]),
        FusionProductSpace::new([leg(), leg()]),
    );
    let keys = homspace.fusion_tree_keys(&rule);
    let space = FusionTensorMapSpace::from_degeneracy_shapes_coupled(
        TensorMapSpace::<2, 2>::from_dims([4, 4], [4, 4]).unwrap(),
        homspace,
        &rule,
        vec![vec![2; 4]; keys.len()],
    )
    .unwrap();
    let tree_code = |tree: &FusionTreeKey| {
        tree.uncoupled()
            .iter()
            .fold(0usize, |value, sector| value * 2 + sector.id())
    };
    let diagonal = |tree: &FusionTreeKey, row: usize| 2.0 + tree_code(tree) as f64 + row as f64;
    let upper = |tree: &FusionTreeKey| 0.25 + tree_code(tree) as f64 * 0.125;
    let canonical = TensorMap::from_block_fn_with_fusion_space(space, 0.0, |key, indices| {
        let BlockKey::FusionTree(key) = key else {
            return 0.0;
        };
        if key.codomain_tree() != key.domain_tree() {
            return 0.0;
        }
        let row = indices[0] + 2 * indices[1];
        let col = indices[2] + 2 * indices[3];
        if row == col {
            diagonal(key.codomain_tree(), row)
        } else if row == 0 && col == 1 {
            upper(key.codomain_tree())
        } else {
            0.0
        }
    })
    .unwrap();
    let reordered = reversed_complete_grid_copy(&rule, &canonical);
    assert!(reordered
        .structure()
        .coupled_sector_regions(2)
        .unwrap()
        .is_none());
    let mut dense = SolveCallSpy::default();
    let mut context = default_context();

    let inverse = inv(
        &mut dense,
        &mut context,
        &bound_tensor_ref!(Arc::new(rule), &reordered),
    )
    .unwrap();

    for index in 0..inverse.structure().block_count() {
        let block = inverse.structure().block(index).unwrap();
        let BlockKey::FusionTree(key) = block.key() else {
            panic!("inverse output must retain fusion-tree blocks")
        };
        for row in 0..4 {
            for col in 0..4 {
                let position = block.offset()
                    + (row % 2) * block.strides()[0]
                    + (row / 2) * block.strides()[1]
                    + (col % 2) * block.strides()[2]
                    + (col / 2) * block.strides()[3];
                let expected = if key.codomain_tree() != key.domain_tree() {
                    0.0
                } else if row == col {
                    diagonal(key.codomain_tree(), row).recip()
                } else if row == 0 && col == 1 {
                    -upper(key.codomain_tree())
                        / (diagonal(key.codomain_tree(), 0) * diagonal(key.codomain_tree(), 1))
                } else {
                    0.0
                };
                assert!(
                    (inverse.data()[position] - expected).abs() < 1.0e-12,
                    "tree={key:?} row={row} col={col}"
                );
            }
        }
    }
    assert_eq!(dense.solve_calls, 2);
}

#[test]
fn inv_preserves_genuinely_complex_nonhermitian_sector_values() {
    // What: inverse is an ordinary solve, not an adjoint or Hermitian spectral operation.
    let a = Complex64::new(2.0, 1.0);
    let b = Complex64::new(3.0, -2.0);
    let c = Complex64::new(1.0, 4.0);
    let d = Complex64::new(5.0, -1.0);
    let tensor = u1_block_endomorphism(&[(0, 2, vec![a, b, c, d])]);
    let mut dense = SolveCallSpy::default();
    let bound = bound_tensor(Arc::new(U1FusionRule), &tensor);

    let inverse = inv_direct_dyn(&mut dense, &bound.as_ref().dynamic()).unwrap();

    let determinant = a * d - b * c;
    let oracle = [
        d / determinant,
        -b / determinant,
        -c / determinant,
        a / determinant,
    ];
    for (&actual, expected) in inverse.data().iter().zip(oracle) {
        assert!((actual - expected).norm() < 1.0e-12);
    }
    assert_eq!(dense.solve_calls, 1);
}

#[test]
fn inv_dyn_reverses_isomorphic_spaces_with_different_tree_ranks() {
    // What: sector identity routes an inverse between isomorphic reduced spaces
    // even when their external tree ranks differ.
    let charge = U1Irrep::new(0).sector_id();
    let homspace = FusionTreeHomSpace::new(
        FusionProductSpace::new([
            SectorLeg::new([(charge, 2)], false),
            SectorLeg::new([(charge, 3)], false),
        ]),
        FusionProductSpace::new([SectorLeg::new([(charge, 6)], false)]),
    );
    let space = FusionTensorMapSpace::from_degeneracy_shapes_coupled(
        TensorMapSpace::<2, 1>::from_dims([2, 3], [6]).unwrap(),
        homspace,
        &U1FusionRule,
        [vec![2, 3, 6]],
    )
    .unwrap();
    let mut data = vec![0.0_f64; 36];
    for index in 0..6 {
        data[index + 6 * index] = index as f64 + 1.0;
    }
    let tensor = TensorMap::<f64, 2, 1>::from_vec_with_fusion_space(data, space).unwrap();
    let bound = bound_tensor(Arc::new(U1FusionRule), &tensor);
    let mut dense = SolveCallSpy::default();

    let inverse = inv_direct_dyn(&mut dense, &bound.as_ref().dynamic()).unwrap();
    let inverse: BoundTensorMap<_, _, 1, 2> = typed_from_bound_factor(inverse).unwrap();

    assert_eq!(
        inverse
            .tensor()
            .fusion_space()
            .unwrap()
            .homspace()
            .codomain(),
        tensor.fusion_space().unwrap().homspace().domain()
    );
    assert_eq!(
        inverse.tensor().fusion_space().unwrap().homspace().domain(),
        tensor.fusion_space().unwrap().homspace().codomain()
    );
    for row in 0..6 {
        for col in 0..6 {
            let expected = if row == col {
                1.0 / (row as f64 + 1.0)
            } else {
                0.0
            };
            assert!((inverse.data()[row + 6 * col] - expected).abs() < 1.0e-12);
        }
    }
    assert_eq!(dense.solve_calls, 1);
}

#[test]
fn inv_route_preflight_rejects_a_missing_later_sector_before_execution() {
    // What: every sector route is checked before the first dense solve.
    let source = u1_block_endomorphism(&[(0, 1, vec![1.0_f64]), (1, 1, vec![2.0])]);
    let incomplete_output = u1_block_endomorphism(&[(0, 1, vec![1.0_f64])]);
    let source_regions = source
        .structure()
        .coupled_sector_regions(1)
        .unwrap()
        .unwrap();
    let output_regions = incomplete_output
        .structure()
        .coupled_sector_regions(1)
        .unwrap()
        .unwrap();

    let error =
        validate_inverse_region_routes_for_test(&source_regions, &output_regions).unwrap_err();

    assert!(matches!(
        error,
        OperationError::UnsupportedTensorContractScope {
            message: "inverse output is missing a source coupled sector"
        }
    ));
}

#[test]
fn inv_accepts_tiny_nonzero_pivots_for_all_factor_dtypes() {
    // What: ordinary inverse follows provider LU singularity semantics rather
    // than rejecting a nonzero pivot using a dtype-relative SVD cutoff.
    let mut dense = tenet_dense::DefaultDenseExecutor::new();

    let tensor = u1_block_endomorphism(&[(0, 2, vec![1.0_f32, 0.0, 0.0, 1.0e-8])]);
    let bound = bound_tensor(Arc::new(U1FusionRule), &tensor);
    let inverse = inv_direct_dyn(&mut dense, &bound.as_ref().dynamic()).unwrap();
    assert!((inverse.data()[0] - 1.0).abs() < 1.0e-6);
    assert!((inverse.data()[3] * 1.0e-8 - 1.0).abs() < 1.0e-5);

    let tensor = u1_block_endomorphism(&[(0, 2, vec![1.0_f64, 0.0, 0.0, 1.0e-16])]);
    let bound = bound_tensor(Arc::new(U1FusionRule), &tensor);
    let inverse = inv_direct_dyn(&mut dense, &bound.as_ref().dynamic()).unwrap();
    assert!((inverse.data()[0] - 1.0).abs() < 1.0e-12);
    assert!((inverse.data()[3] * 1.0e-16 - 1.0).abs() < 1.0e-12);

    let phase = Complex32::from_polar(1.0e-8, -0.41);
    let tensor = u1_block_endomorphism(&[(
        0,
        2,
        vec![
            Complex32::new(1.0, 0.0),
            Complex32::zero(),
            Complex32::zero(),
            phase,
        ],
    )]);
    let bound = bound_tensor(Arc::new(U1FusionRule), &tensor);
    let inverse = inv_direct_dyn(&mut dense, &bound.as_ref().dynamic()).unwrap();
    assert!((inverse.data()[3] * phase - Complex32::new(1.0, 0.0)).norm() < 1.0e-5);

    let phase = Complex64::from_polar(1.0e-16, 0.23);
    let tensor = u1_block_endomorphism(&[(
        0,
        2,
        vec![
            Complex64::new(1.0, 0.0),
            Complex64::zero(),
            Complex64::zero(),
            phase,
        ],
    )]);
    let bound = bound_tensor(Arc::new(U1FusionRule), &tensor);
    let inverse = inv_direct_dyn(&mut dense, &bound.as_ref().dynamic()).unwrap();
    assert!((inverse.data()[3] * phase - Complex64::new(1.0, 0.0)).norm() < 1.0e-12);
}

#[test]
fn inv_rejects_a_genuinely_singular_sector_without_an_output() {
    // What: the dense solve reports an exact singular direction as a typed numerical failure.
    let tensor = u1_block_endomorphism(&[(0, 2, vec![1.0_f64, 0.0, 0.0, 0.0])]);
    let mut dense = tenet_dense::DefaultDenseExecutor::new();
    let mut context = TensorContractFusionExecutionContext::<f64, RuleIdentity>::default();

    let error = inv(
        &mut dense,
        &mut context,
        &bound_tensor_ref!(Arc::new(U1FusionRule), &tensor),
    )
    .unwrap_err();

    assert!(matches!(
        error,
        OperationError::Dense(DenseError::NumericalFailure {
            op: "solve_into",
            ..
        })
    ));
    assert_eq!(context.tree_context().cache().structure_len(), 0);
    assert_eq!(context.dynamic_fusion_space_cache_len(), 0);
}

#[test]
fn inv_discards_unpublished_output_when_a_later_sector_fails() {
    // What: success in an earlier sector cannot publish a partial inverse when
    // a later backend solve fails.
    let tensor = u1_block_endomorphism(&[(0, 1, vec![2.0_f64]), (1, 1, vec![3.0])]);
    let bound = bound_tensor(Arc::new(U1FusionRule), &tensor);
    let mut dense = FailSecondSolve::default();

    let error = inv_direct_dyn(&mut dense, &bound.as_ref().dynamic()).unwrap_err();

    assert!(matches!(
        error,
        OperationError::Dense(DenseError::Backend {
            op: "solve_into",
            ..
        })
    ));
    assert_eq!(dense.solve_calls, 2);
}

#[test]
fn inv_propagates_unsupported_solve_without_svd_fallback() {
    // What: a dense executor without solve support returns its typed capability
    // error instead of silently changing inverse algorithms.
    let tensor = u1_block_endomorphism(&[(0, 1, vec![2.0_f64])]);
    let mut dense = RejectExecutorCalls;
    let bound = bound_tensor(Arc::new(U1FusionRule), &tensor);

    let error = inv_direct_dyn(&mut dense, &bound.as_ref().dynamic()).unwrap_err();

    assert!(matches!(
        error,
        OperationError::Dense(DenseError::Unsupported {
            op: "solve_into",
            ..
        })
    ));
}

#[test]
fn inv_accepts_a_zero_dimensional_endomorphism_without_dense_execution() {
    // What: the inverse of the legal empty endomorphism is the empty endomorphism.
    let tensor = rectangular_svd_tensor(0, 0);
    let mut dense = RejectExecutorCalls;
    let mut context = TensorContractFusionExecutionContext::<f64, RuleIdentity>::default();

    let inverse = inv(
        &mut dense,
        &mut context,
        &bound_tensor_ref!(Arc::new(Z2FusionRule), &tensor),
    )
    .unwrap();

    assert!(inverse.data().is_empty());
    assert_eq!(inverse.tensor().fusion_space(), tensor.fusion_space());
}

#[test]
fn inv_solves_the_rank_zero_scalar_sector() {
    // What: a rank-zero scalar is one 1x1 vacuum-sector solve, not an empty tensor.
    let rule = Z2FusionRule;
    let homspace =
        FusionTreeHomSpace::new(FusionProductSpace::new([]), FusionProductSpace::new([]));
    let shapes = vec![Vec::new(); homspace.fusion_tree_keys(&rule).len()];
    let space = FusionTensorMapSpace::from_degeneracy_shapes_coupled(
        TensorMapSpace::<0, 0>::from_dims([], []).unwrap(),
        homspace,
        &rule,
        shapes,
    )
    .unwrap();
    let scalar = TensorMap::<f64, 0, 0>::from_vec_with_fusion_space(vec![-4.0], space).unwrap();
    let mut dense = SolveCallSpy::default();
    let mut context = default_context();

    let inverse = inv(
        &mut dense,
        &mut context,
        &bound_tensor_ref!(Arc::new(rule), &scalar),
    )
    .unwrap();

    assert_eq!(inverse.data(), &[-0.25]);
    assert_eq!(dense.solve_calls, 1);
}

#[test]
fn pinv_keeps_its_global_rcond_cutoff() {
    // What: public pinv still drops singular values relative to the global maximum.
    let tensor = u1_block_endomorphism(&[(0, 1, vec![1.0_f64]), (1, 1, vec![1e-14])]);
    let mut dense = tenet_dense::DefaultDenseExecutor::new();
    let mut context = TensorContractFusionExecutionContext::<f64, RuleIdentity>::default();

    let inverse = pinv(
        &mut dense,
        &mut context,
        &bound_tensor_ref!(Arc::new(U1FusionRule), &tensor),
        1e-12,
    )
    .unwrap();

    assert!((scalar_u1_block(inverse.tensor(), 0) - 1.0).abs() < 1e-12);
    assert_eq!(scalar_u1_block(inverse.tensor(), 1), 0.0);
}

#[test]
fn pinv_adjoint_parent_uses_one_parent_svd_and_the_shared_global_cutoff() {
    // What: the largest singular value is in the later sector, and the first
    // sector sits exactly on the strict global cutoff. The parent-native seam
    // runs one SVD per stored sector, preserves provider identity, and emits
    // the final logical-adjoint orientation directly.
    let tensor = u1_block_endomorphism(&[(0, 1, vec![0.5_f64]), (1, 1, vec![1.0])]);
    let provider = Arc::new(U1FusionRule);
    let bound = bound_tensor(Arc::clone(&provider), &tensor);
    let mut dense = SvdCallSpy::default();
    let mut context = TensorContractFusionExecutionContext::<f64, RuleIdentity>::default();

    let output =
        pinv_adjoint_parent_dyn(&mut dense, &mut context, &bound.as_ref().dynamic(), 0.5).unwrap();
    assert_eq!(dense.svd_calls, 2);
    assert!(Arc::ptr_eq(output.space().provider_arc(), &provider));
    let output: BoundTensorMap<_, _, 1, 1> = typed_from_bound_factor(output).unwrap();
    assert_eq!(scalar_u1_block(output.tensor(), 0), 0.0);
    assert!((scalar_u1_block(output.tensor(), 1) - 1.0).abs() < 1e-12);
}

#[test]
fn pinv_adjoint_parent_rejects_invalid_rcond_before_svd() {
    // What: the hidden seam owns the same validation precedence as ordinary
    // pinv, independently of either facade.
    let tensor = u1_block_endomorphism(&[(0, 1, vec![1.0_f64])]);
    let bound = bound_tensor(Arc::new(U1FusionRule), &tensor);
    for rcond in [-1.0, f64::NAN, f64::INFINITY] {
        let mut dense = RejectExecutorCalls;
        let mut context = TensorContractFusionExecutionContext::<f64, RuleIdentity>::default();
        assert!(matches!(
            pinv_adjoint_parent_dyn(&mut dense, &mut context, &bound.as_ref().dynamic(), rcond,),
            Err(OperationError::InvalidArgument { .. })
        ));
    }
}

#[test]
fn pinv_adjoint_parent_discards_unpublished_factors_on_late_svd_failure() {
    // What: a successful first sector cannot publish factors or an output when
    // the second sector's SVD fails.
    let tensor = u1_block_endomorphism(&[(0, 1, vec![2.0_f64]), (1, 1, vec![3.0])]);
    let bound = bound_tensor(Arc::new(U1FusionRule), &tensor);
    let mut dense = FailSecondSvd::default();
    let mut context = TensorContractFusionExecutionContext::<f64, RuleIdentity>::default();

    assert!(matches!(
        pinv_adjoint_parent_dyn(&mut dense, &mut context, &bound.as_ref().dynamic(), 0.0,),
        Err(OperationError::Dense(DenseError::Backend {
            op: "svd_into",
            ..
        }))
    ));
    assert_eq!(dense.calls, 2);
}

#[test]
fn pinv_adjoint_parent_discards_unpublished_output_on_recomposition_failure() {
    // What: after a successful parent SVD and scaling, a failed final compose
    // returns no partial output.
    let tensor = u1_block_endomorphism(&[(0, 2, vec![2.0, 0.0, 0.0, 3.0])]);
    let bound = bound_tensor(Arc::new(U1FusionRule), &tensor);
    let mut dense = tenet_dense::DefaultDenseExecutor::new();
    let mut context: TensorContractFusionExecutionContext<
        f64,
        RuleIdentity,
        DenseTreeTransformOperations,
        DenseTreeTransformOperations<FailComposition>,
    > = TensorContractFusionExecutionContext::new(
        DenseTreeTransformOperations::default(),
        DenseTreeTransformOperations::new(FailComposition),
    );

    assert!(matches!(
        pinv_adjoint_parent_dyn(&mut dense, &mut context, &bound.as_ref().dynamic(), 0.0,),
        Err(OperationError::Dense(DenseError::Backend {
            op: "dot_general_into",
            ..
        }))
    ));
}

#[test]
fn polar_decompositions_reconstruct_with_isometric_factors() {
    let rule = SU2FusionRule;
    let tensor = tsvd_test_tensor(
        &rule,
        &[
            SU2Irrep::from_twice_spin(0).sector_id(),
            SU2Irrep::from_twice_spin(1).sector_id(),
        ],
    );
    let mut dense_executor = tenet_dense::DefaultDenseExecutor::new();
    let mut context = default_context();

    let (isometry, positive) = left_polar(
        &mut dense_executor,
        &mut context,
        &bound_tensor_ref!(Arc::new(rule), &tensor),
    )
    .unwrap();
    let reconstructed = crate::compose::compose(&mut context, &rule, &isometry, &positive).unwrap();
    assert_svd_blocks_match(&tensor, &reconstructed);
    let wh = tenet_tensors::adjoint(&rule, &isometry).unwrap();
    let unit = crate::compose::compose(&mut context, &rule, &wh, &isometry).unwrap();
    assert_identity_matrices(&dense_sector_matrices(2, &unit));

    let (positive, isometry) = right_polar(
        &mut dense_executor,
        &mut context,
        &bound_tensor_ref!(Arc::new(rule), &tensor),
    )
    .unwrap();
    let reconstructed = crate::compose::compose(&mut context, &rule, &positive, &isometry).unwrap();
    assert_svd_blocks_match(&tensor, &reconstructed);
}

#[test]
fn polar_rejects_wrong_rectangular_direction_before_dense_execution() {
    // What: invalid left/right directions are rejected before any sector SVD starts.
    let rule = Z2FusionRule;
    for (operation, rows, cols) in [("left_polar", 2, 3), ("right_polar", 3, 2)] {
        let tensor = rectangular_svd_tensor(rows, cols);
        let mut dense = RejectExecutorCalls;
        let mut context = default_context();
        let result = if operation == "left_polar" {
            left_polar(
                &mut dense,
                &mut context,
                &bound_tensor_ref!(Arc::new(rule), &tensor),
            )
        } else {
            right_polar(
                &mut dense,
                &mut context,
                &bound_tensor_ref!(Arc::new(rule), &tensor),
            )
        };

        assert!(matches!(
            result,
            Err(OperationError::InvalidArgument { message })
                if message.contains(operation)
                    && message.contains("coupled-sector")
        ));
    }
}

fn assert_polar_direction_error_before_dense(tensor: &TensorMap<f64, 1, 1>, left: bool) {
    let before = tensor.data().to_vec();
    let mut dense = RejectExecutorCalls;
    let mut context = default_context();
    let input = bound_tensor(Arc::new(U1FusionRule), tensor);
    let error = if left {
        left_polar(&mut dense, &mut context, &input.as_ref()).unwrap_err()
    } else {
        right_polar(&mut dense, &mut context, &input.as_ref()).unwrap_err()
    };
    let operation = if left { "left_polar" } else { "right_polar" };
    assert!(matches!(
        error,
        OperationError::InvalidArgument { message }
            if message.contains(operation) && message.contains("coupled-sector")
    ));
    assert_eq!(tensor.data(), before);
}

fn assert_valid_unmatched_left_polar(tensor: &TensorMap<f64, 1, 1>) {
    let provider = Arc::new(U1FusionRule);
    let input = bound_tensor(Arc::clone(&provider), tensor);
    let mut dense = tenet_dense::DefaultDenseExecutor::new();
    let mut context = default_context();
    let (isometry, positive) = left_polar(&mut dense, &mut context, &input.as_ref()).unwrap();

    assert!(Arc::ptr_eq(isometry.space().provider_arc(), &provider));
    assert!(Arc::ptr_eq(positive.space().provider_arc(), &provider));
    let reconstructed =
        crate::compose::compose(&mut context, provider.as_ref(), &isometry, &positive).unwrap();
    assert_svd_blocks_match(tensor, &reconstructed);
    let adjoint = tenet_tensors::adjoint(provider.as_ref(), &isometry).unwrap();
    let gram =
        crate::compose::compose(&mut context, provider.as_ref(), &adjoint, &isometry).unwrap();
    assert_identity_matrices(&dense_sector_matrices(1, &gram));
    assert_identity_matrices(&dense_sector_matrices(1, &positive));
}

fn assert_valid_unmatched_right_polar(tensor: &TensorMap<f64, 1, 1>) {
    let provider = Arc::new(U1FusionRule);
    let input = bound_tensor(Arc::clone(&provider), tensor);
    let mut dense = tenet_dense::DefaultDenseExecutor::new();
    let mut context = default_context();
    let (positive, isometry) = right_polar(&mut dense, &mut context, &input.as_ref()).unwrap();

    assert!(Arc::ptr_eq(positive.space().provider_arc(), &provider));
    assert!(Arc::ptr_eq(isometry.space().provider_arc(), &provider));
    let reconstructed =
        crate::compose::compose(&mut context, provider.as_ref(), &positive, &isometry).unwrap();
    assert_svd_blocks_match(tensor, &reconstructed);
    let adjoint = tenet_tensors::adjoint(provider.as_ref(), &isometry).unwrap();
    let gram =
        crate::compose::compose(&mut context, provider.as_ref(), &isometry, &adjoint).unwrap();
    assert_identity_matrices(&dense_sector_matrices(1, &gram));
    assert_identity_matrices(&dense_sector_matrices(1, &positive));
}

#[test]
fn polar_complete_dimension_preflight_handles_unmatched_and_disjoint_support() {
    // What: side-only sectors participate as rows x 0 or 0 x columns, so only
    // the direction whose isometry law is structurally possible is accepted.
    let codomain_only = u1_cross_space_map::<f64>(&[(0, 2), (1, 3)], &[(0, 2)]);
    assert_valid_unmatched_left_polar(&codomain_only);
    assert_polar_direction_error_before_dense(&codomain_only, false);

    let domain_only = u1_cross_space_map::<f64>(&[(0, 2)], &[(0, 2), (1, 3)]);
    assert_valid_unmatched_right_polar(&domain_only);
    assert_polar_direction_error_before_dense(&domain_only, true);

    let disjoint = u1_cross_space_map::<f64>(&[(1, 2)], &[(0, 3)]);
    assert_polar_direction_error_before_dense(&disjoint, true);
    assert_polar_direction_error_before_dense(&disjoint, false);
}

#[test]
fn polar_complete_dimension_preflight_handles_empty_sides_and_empty_products() {
    // What: an empty smaller side remains a valid vacuous isometry, while an
    // empty larger side is rejected; rank-zero products still carry vacuum.
    let empty_codomain = u1_cross_space_map::<f64>(&[], &[(0, 2)]);
    assert_polar_direction_error_before_dense(&empty_codomain, true);
    let mut dense = RejectExecutorCalls;
    let mut context = default_context();
    right_polar(
        &mut dense,
        &mut context,
        &bound_tensor_ref!(Arc::new(U1FusionRule), &empty_codomain),
    )
    .unwrap();

    let empty_domain = u1_cross_space_map::<f64>(&[(0, 2)], &[]);
    assert_polar_direction_error_before_dense(&empty_domain, false);
    let mut dense = RejectExecutorCalls;
    let mut context = default_context();
    left_polar(
        &mut dense,
        &mut context,
        &bound_tensor_ref!(Arc::new(U1FusionRule), &empty_domain),
    )
    .unwrap();

    let empty = u1_cross_space_map::<f64>(&[], &[]);
    let mut dense = RejectExecutorCalls;
    let mut context = default_context();
    let input = bound_tensor(Arc::new(U1FusionRule), &empty);
    left_polar(&mut dense, &mut context, &input.as_ref()).unwrap();
    right_polar(&mut dense, &mut context, &input.as_ref()).unwrap();

    let rule = U1FusionRule;
    let homspace =
        FusionTreeHomSpace::new(FusionProductSpace::new([]), FusionProductSpace::new([]));
    let shapes = vec![Vec::new(); homspace.fusion_tree_keys(&rule).len()];
    let space = FusionTensorMapSpace::from_degeneracy_shapes_coupled(
        TensorMapSpace::<0, 0>::from_dims([], []).unwrap(),
        homspace,
        &rule,
        shapes,
    )
    .unwrap();
    let scalar = TensorMap::<f64, 0, 0>::from_vec_with_fusion_space(vec![2.0], space).unwrap();
    let mut dense = tenet_dense::DefaultDenseExecutor::new();
    let mut context = default_context();
    let input = bound_tensor(Arc::new(rule), &scalar);
    let (w, p) = left_polar(&mut dense, &mut context, &input.as_ref()).unwrap();
    assert_eq!(w.data(), &[1.0]);
    assert_eq!(p.data(), &[2.0]);
    let (p, w) = right_polar(&mut dense, &mut context, &input.as_ref()).unwrap();
    assert_eq!(p.data(), &[2.0]);
    assert_eq!(w.data(), &[1.0]);
}

#[test]
fn polar_second_sector_failure_leaves_the_source_unchanged() {
    // What: a later dense failure publishes no factors and cannot mutate the
    // borrowed source, in either direction.
    let tensor = u1_cross_space_map::<f64>(&[(0, 2), (1, 2)], &[(0, 2), (1, 2)]);
    let before = tensor.data().to_vec();
    let input = bound_tensor(Arc::new(U1FusionRule), &tensor);
    for left in [true, false] {
        let mut dense = FailSecondSvd::default();
        let mut context = default_context();
        let result = if left {
            left_polar(&mut dense, &mut context, &input.as_ref())
        } else {
            right_polar(&mut dense, &mut context, &input.as_ref())
        };
        assert!(matches!(result, Err(OperationError::Dense(_))));
        assert_eq!(dense.calls, 2);
        assert_eq!(tensor.data(), before);
    }
}

#[test]
fn polar_validates_every_sector_before_direct_or_fallback_svd_execution() {
    // What: a later invalid sector prevents SVD of an earlier valid sector on both layouts.
    let rule = Z2FusionRule;
    let direct = mixed_rectangular_tensor((4, 2), (1, 3));
    let direct_bound = bound_tensor(Arc::new(rule), &direct);
    assert!(
        crate::factorize::compact_factor_plan_for_test(direct_bound.space())
            .unwrap()
            .is_some()
    );
    let mut dense = SvdCallSpy::default();
    let mut context = default_context();
    let direct_error = left_polar(&mut dense, &mut context, &direct_bound.as_ref()).unwrap_err();
    assert!(matches!(
        direct_error,
        OperationError::InvalidArgument { message }
            if message.contains("left_polar")
                && message.contains("coupled-sector")
    ));
    assert_eq!(dense.svd_calls, 0);

    let fallback_source = mixed_rectangular_tensor((2, 4), (3, 1));
    let fallback_bound = bound_tensor(Arc::new(rule), &fallback_source);
    let fallback_space = fallback_bound.space().adjoint_view().unwrap();
    assert!(
        crate::factorize::compact_factor_plan_for_test(&fallback_space)
            .unwrap()
            .is_none()
    );
    let fallback_input =
        BoundDynamicTensorRef::try_new(&fallback_space, fallback_bound.data()).unwrap();
    let mut dense = SvdCallSpy::default();
    let mut context = default_context();
    let fallback_error = left_polar_dyn(&mut dense, &mut context, &fallback_input).unwrap_err();
    assert!(matches!(
        fallback_error,
        OperationError::InvalidArgument { message }
            if message.contains("left_polar")
                && message.contains("coupled-sector")
    ));
    assert_eq!(dense.svd_calls, 0);
}

#[test]
fn polar_valid_direct_and_fallback_layouts_agree() {
    // What: valid fallback matricizations preserve the direct polar factors in both directions.
    let rule = Z2FusionRule;
    for (operation, source_rows, source_cols) in [("left_polar", 2, 3), ("right_polar", 3, 2)] {
        let source = rectangular_svd_tensor(source_rows, source_cols);
        let transposed = transposed_rectangular_tensor(&source, source_rows, source_cols);
        let source_bound = bound_tensor(Arc::new(rule), &source);
        let direct_bound = bound_tensor(Arc::new(rule), &transposed);
        let fallback_space = source_bound.space().adjoint_view().unwrap();
        let fallback_input =
            BoundDynamicTensorRef::try_new(&fallback_space, source_bound.data()).unwrap();
        assert!(
            crate::factorize::compact_factor_plan_for_test(direct_bound.space())
                .unwrap()
                .is_some()
        );
        assert!(
            crate::factorize::compact_factor_plan_for_test(&fallback_space)
                .unwrap()
                .is_none()
        );
        let mut direct_dense = tenet_dense::DefaultDenseExecutor::new();
        let mut direct_context = default_context();
        let mut fallback_dense = tenet_dense::DefaultDenseExecutor::new();
        let mut fallback_context = default_context();

        let (direct_first, direct_second, fallback_first, fallback_second) =
            if operation == "left_polar" {
                let (direct_first, direct_second) = left_polar(
                    &mut direct_dense,
                    &mut direct_context,
                    &direct_bound.as_ref(),
                )
                .unwrap();
                let (fallback_first, fallback_second) =
                    left_polar_dyn(&mut fallback_dense, &mut fallback_context, &fallback_input)
                        .unwrap();
                (
                    direct_first.data().to_vec(),
                    direct_second.data().to_vec(),
                    fallback_first.data().to_vec(),
                    fallback_second.data().to_vec(),
                )
            } else {
                let (direct_first, direct_second) = right_polar(
                    &mut direct_dense,
                    &mut direct_context,
                    &direct_bound.as_ref(),
                )
                .unwrap();
                let (fallback_first, fallback_second) =
                    right_polar_dyn(&mut fallback_dense, &mut fallback_context, &fallback_input)
                        .unwrap();
                (
                    direct_first.data().to_vec(),
                    direct_second.data().to_vec(),
                    fallback_first.data().to_vec(),
                    fallback_second.data().to_vec(),
                )
            };

        assert_eq!(direct_first.len(), fallback_first.len());
        assert_eq!(direct_second.len(), fallback_second.len());
        for (direct, fallback) in direct_first.iter().zip(&fallback_first) {
            assert!((direct - fallback).abs() < 1e-10);
        }
        for (direct, fallback) in direct_second.iter().zip(&fallback_second) {
            assert!((direct - fallback).abs() < 1e-10);
        }
    }
}

#[test]
fn single_precision_svd_and_eig_work_end_to_end() {
    use num_complex::Complex32;
    let rule = Z2FusionRule;
    let sectors = [SectorId::new(0), SectorId::new(1)];
    let degeneracy = 2usize;
    let leg = || SectorLeg::new(sectors.iter().map(|&sector| (sector, degeneracy)), false);
    let leg_dim = sectors.len() * degeneracy;
    let homspace = || {
        FusionTreeHomSpace::new(
            FusionProductSpace::new([leg(), leg()]),
            FusionProductSpace::new([leg(), leg()]),
        )
    };
    let space = || {
        let hom = homspace();
        let key_count = hom.fusion_tree_keys(&rule).len();
        FusionTensorMapSpace::from_degeneracy_shapes_coupled(
            TensorMapSpace::<2, 2>::from_dims([leg_dim, leg_dim], [leg_dim, leg_dim]).unwrap(),
            hom,
            &rule,
            vec![vec![degeneracy; 4]; key_count],
        )
        .unwrap()
    };
    let f32_space = space();
    let len = f32_space.required_len().unwrap();
    let tensor_f32 = TensorMap::<f32, 2, 2>::from_vec_with_fusion_space(
        (0..len)
            .map(|i| ((i * 7 + 3) % 23) as f32 * 0.5 - 5.0)
            .collect(),
        f32_space,
    )
    .unwrap();

    let mut dense_executor = tenet_dense::DefaultDenseExecutor::new();
    let svd = svd_trunc(
        &mut dense_executor,
        &bound_tensor_ref!(Arc::new(rule), &tensor_f32),
        &Truncation::rank(8),
    )
    .unwrap();
    assert!(svd.error > 0.0);

    // Reconstruct through an f32 contraction and compare against the
    // truncation error at single precision.
    let mut scaled_vh = svd.vh.tensor().clone();
    {
        let structure = std::sync::Arc::clone(scaled_vh.structure());
        for index in 0..structure.block_count() {
            let block = structure.block(index).unwrap();
            let BlockKey::FusionTree(key) = block.key() else {
                continue;
            };
            let sector = key.codomain_tree().coupled();
            let values = &svd
                .singular_values
                .iter()
                .find(|entry| entry.sector == sector)
                .unwrap()
                .values;
            let shape = block.shape().to_vec();
            let strides = block.strides().to_vec();
            let offset = block.offset();
            let count = shape.iter().product::<usize>();
            let mut indices = vec![0usize; shape.len()];
            for _ in 0..count {
                let position = offset
                    + indices
                        .iter()
                        .zip(&strides)
                        .map(|(&i, &s)| i * s)
                        .sum::<usize>();
                scaled_vh.data_mut()[position] *= values[indices[0]] as f32;
                for axis in 0..shape.len() {
                    indices[axis] += 1;
                    if indices[axis] < shape[axis] {
                        break;
                    }
                    indices[axis] = 0;
                }
            }
        }
    }
    let mut context = TensorContractFusionExecutionContext::<f32, RuleIdentity>::default();
    let reconstructed = crate::compose::compose(&mut context, &rule, &svd.u, &scaled_vh).unwrap();
    let distance = tensor_f32
        .data()
        .iter()
        .zip(reconstructed.data())
        .map(|(lhs, rhs)| ((lhs - rhs) as f64).powi(2))
        .sum::<f64>()
        .sqrt();
    assert!(
        (distance - svd.error).abs() < 1e-3,
        "f32 distance {distance} != error {}",
        svd.error
    );

    // Complex32 general eigendecomposition returns Complex32 factors.
    let c32_space = space();
    let len = c32_space.required_len().unwrap();
    let tensor_c32 = TensorMap::<Complex32, 2, 2>::from_vec_with_fusion_space(
        (0..len)
            .map(|i| {
                Complex32::new(
                    ((i * 3 + 1) % 13) as f32 - 6.0,
                    ((i * 5 + 2) % 11) as f32 - 5.0,
                )
            })
            .collect(),
        c32_space,
    )
    .unwrap();
    let eig = eig_full(
        &mut dense_executor,
        &bound_tensor_ref!(Arc::new(rule), &tensor_c32),
    )
    .unwrap();
    assert!(!eig.eigenvalues.is_empty());
    for entry in &eig.eigenvalues {
        for pair in entry.values.windows(2) {
            assert!(pair[0].norm() >= pair[1].norm() - 1e-6);
        }
    }
    let _: &TensorMap<Complex32, 2, 1> = &eig.v;
}

#[test]
fn positive_diagonal_gauge_matches_tensorkit_qr_reference() {
    // TensorKit 0.17.0 / MatrixAlgebraKit 0.6.8 crosscheck:
    //   A = [-1 2; 3 4; 5 -6]; Q, R = MatrixAlgebraKit.qr_compact(A)
    // (default `positive = true` since MAK 0.6.8). Column-major reference:
    let q_ref = [
        -0.16903085094570325,
        0.50709255283711,
        0.8451542547285166,
        0.21398024625545642,
        0.8559209850218259,
        -0.4707565417620042,
    ];
    let r_ref = [
        5.916079783099615,
        0.0,
        -3.380617018914066,
        6.676183683170241,
    ];
    // Start from the equally valid un-gauged QR with both diagonal signs
    // flipped (Q -> -Q, R -> -R); the gauge must restore the reference.
    let mut q: Vec<f64> = q_ref.iter().map(|v| -v).collect();
    let mut r: Vec<f64> = r_ref.iter().map(|v| -v).collect();
    crate::factorize::positive_diagonal_gauge(&mut q, 3, &mut r, 2, 2);
    for (value, reference) in q.iter().zip(&q_ref) {
        assert!(
            (value - reference).abs() < 1e-14,
            "Q {value} != {reference}"
        );
    }
    for (value, reference) in r.iter().zip(&r_ref) {
        assert!(
            (value - reference).abs() < 1e-14,
            "R {value} != {reference}"
        );
    }
}

#[test]
fn positive_diagonal_gauge_complex_phase_and_zero_diagonal() {
    use num_complex::Complex64;
    let c = Complex64::new;
    // q: 3 x 3, r: 3 x 3 upper triangular with complex diagonal phases and a
    // zero diagonal entry (row 1), column-major.
    let q: Vec<Complex64> = (0..9)
        .map(|i| c((i as f64 * 0.7 - 2.0).sin(), (i as f64 * 1.3 + 0.5).cos()))
        .collect();
    let r = vec![
        c(-3.0, 4.0),
        c(0.0, 0.0),
        c(0.0, 0.0),
        c(1.0, -2.0),
        c(0.0, 0.0),
        c(0.0, 0.0),
        c(0.5, 0.25),
        c(2.0, 1.0),
        c(0.0, -7.0),
    ];
    let product = |q: &[Complex64], r: &[Complex64]| -> Vec<Complex64> {
        let mut out = vec![c(0.0, 0.0); 9];
        for col in 0..3 {
            for row in 0..3 {
                for k in 0..3 {
                    out[row + 3 * col] += q[row + 3 * k] * r[k + 3 * col];
                }
            }
        }
        out
    };
    let before = product(&q, &r);
    let mut q_gauged = q.clone();
    let mut r_gauged = r.clone();
    crate::factorize::positive_diagonal_gauge(&mut q_gauged, 3, &mut r_gauged, 3, 3);
    // Diagonal of R is real non-negative; the zero entry keeps phase 1.
    for j in 0..3 {
        let diagonal = r_gauged[j + 3 * j];
        assert!(
            diagonal.im.abs() < 1e-14,
            "R[{j},{j}] = {diagonal} not real"
        );
        assert!(diagonal.re >= 0.0, "R[{j},{j}] = {diagonal} negative");
    }
    let zero_diagonal_row = 1;
    let leading_dimension = 3;
    let zero_diagonal_index = zero_diagonal_row + leading_dimension * zero_diagonal_row;
    assert_eq!(r_gauged[zero_diagonal_index], c(0.0, 0.0));
    assert_eq!(q_gauged[3], q[3], "zero diagonal must not rescale Q column");
    // Q * R is unchanged.
    let after = product(&q_gauged, &r_gauged);
    for (lhs, rhs) in after.iter().zip(&before) {
        assert!(
            (lhs - rhs).norm() < 1e-13,
            "product changed: {lhs} vs {rhs}"
        );
    }
}

#[test]
fn svd_compact_gauge_matches_matrixalgebrakit_phase_rule() {
    use num_complex::Complex64;
    let c = Complex64::new;
    let mut u = vec![
        c(3.0, 4.0),
        c(1.0, -1.0),
        c(-2.0, 0.5),
        c(0.25, -0.5),
        c(-4.0, 0.0),
        c(1.0, 2.0),
    ];
    let mut vh = vec![
        c(0.5, -1.0),
        c(-0.25, 0.75),
        c(1.0, 0.0),
        c(0.0, -2.0),
        c(-1.5, 0.25),
        c(0.75, -0.5),
    ];
    let sigma = [2.0, 0.75];
    let product = |u: &[Complex64], vh: &[Complex64]| -> Vec<Complex64> {
        let mut out = vec![c(0.0, 0.0); 9];
        for col in 0..3 {
            for row in 0..3 {
                for k in 0..2 {
                    out[row + 3 * col] += u[row + 3 * k] * sigma[k] * vh[k + 2 * col];
                }
            }
        }
        out
    };
    let before = product(&u, &vh);
    crate::factorize::svd_compact_gauge(&mut u, 3, 3, &mut vh, 2, 3, 2);
    for &(row, col) in &[(0, 0), (1, 1)] {
        let pivot = u[row + 3 * col];
        assert!(pivot.im.abs() < 1e-14, "pivot {pivot} not real");
        assert!(pivot.re >= 0.0, "pivot {pivot} negative");
    }
    let after = product(&u, &vh);
    for (lhs, rhs) in after.iter().zip(&before) {
        assert!(
            (lhs - rhs).norm() < 1e-13,
            "product changed: {lhs} vs {rhs}"
        );
    }
}

#[test]
fn svd_compact_adjoint_gauge_fixes_final_left_factor() {
    use num_complex::Complex64;
    let c = Complex64::new;
    let mut u = vec![
        c(3.0, 4.0),
        c(1.0, -1.0),
        c(-2.0, 0.5),
        c(0.25, -0.5),
        c(-4.0, 0.0),
        c(1.0, 2.0),
    ];
    let mut vh = vec![
        c(0.5, -1.0),
        c(-0.25, 0.75),
        c(1.0, 0.0),
        c(0.0, -2.0),
        c(-0.5, 1.0),
        c(0.75, -0.5),
    ];
    let sigma = [2.0, 0.75];
    let product = |u: &[Complex64], vh: &[Complex64]| -> Vec<Complex64> {
        let mut out = vec![c(0.0, 0.0); 9];
        for col in 0..3 {
            for row in 0..3 {
                for k in 0..2 {
                    out[row + 3 * col] += u[row + 3 * k] * sigma[k] * vh[k + 2 * col];
                }
            }
        }
        out
    };
    let before = product(&u, &vh);
    crate::factorize::svd_compact_adjoint_gauge(&mut u, 3, 3, &mut vh, 2, 3, 2);

    // These rows become the columns of final U = V after adjointing Vh.
    for &(row, col) in &[(0, 0), (1, 1)] {
        let pivot = vh[row + 2 * col];
        assert!(pivot.im.abs() < 1e-14, "pivot {pivot} not real");
        assert!(pivot.re >= 0.0, "pivot {pivot} negative");
    }
    let after = product(&u, &vh);
    for (lhs, rhs) in after.iter().zip(&before) {
        assert!(
            (lhs - rhs).norm() < 1e-13,
            "product changed: {lhs} vs {rhs}"
        );
    }
}

#[test]
fn eigenvector_gauge_matches_matrixalgebrakit_phase_rule() {
    use num_complex::Complex64;
    let c = Complex64::new;
    let mut vectors = vec![
        c(3.0, 4.0),
        c(1.0, -1.0),
        c(-2.0, 0.5),
        c(0.25, -0.5),
        c(-4.0, 0.0),
        c(1.0, 2.0),
    ];

    crate::factorize::eigenvector_gauge(&mut vectors, 3, 3, 2);

    for &(row, col) in &[(0, 0), (1, 1)] {
        let pivot = vectors[row + 3 * col];
        assert!(pivot.im.abs() < 1e-14, "pivot {pivot} not real");
        assert!(pivot.re >= 0.0, "pivot {pivot} negative");
    }
}

#[test]
fn svd_full_gauge_fixes_extra_vh_rows_without_changing_product() {
    use num_complex::Complex64;
    let c = Complex64::new;
    let mut u = vec![c(0.0, -2.0), c(0.25, 0.5), c(1.0, -1.0), c(-3.0, 0.0)];
    let mut vh = vec![
        c(1.0, 0.5),
        c(-0.25, 0.75),
        c(1.0, -1.0),
        c(0.5, -0.5),
        c(2.0, 0.0),
        c(-0.5, 0.25),
        c(-1.0, 0.75),
        c(0.0, -1.5),
        c(0.25, 0.0),
    ];
    let sigma = [1.5, 0.7];
    let product = |u: &[Complex64], vh: &[Complex64]| -> Vec<Complex64> {
        let mut out = vec![c(0.0, 0.0); 6];
        for col in 0..3 {
            for row in 0..2 {
                for k in 0..2 {
                    out[row + 2 * col] += u[row + 2 * k] * sigma[k] * vh[k + 3 * col];
                }
            }
        }
        out
    };
    let before = product(&u, &vh);
    crate::factorize::svd_full_gauge(&mut u, 2, 2, &mut vh, 3, 3);
    for &(row, col) in &[(0, 0), (1, 1)] {
        let pivot = u[row + 2 * col];
        assert!(pivot.im.abs() < 1e-14, "U pivot {pivot} not real");
        assert!(pivot.re >= 0.0, "U pivot {pivot} negative");
    }
    let extra_pivot = vh[2]; // row 2, col 0 (row + 3 * col)
    assert!(
        extra_pivot.im.abs() < 1e-14,
        "Vh pivot {extra_pivot} not real"
    );
    assert!(extra_pivot.re >= 0.0, "Vh pivot {extra_pivot} negative");
    let after = product(&u, &vh);
    for (lhs, rhs) in after.iter().zip(&before) {
        assert!(
            (lhs - rhs).norm() < 1e-13,
            "product changed: {lhs} vs {rhs}"
        );
    }
}

#[test]
fn qr_compact_positive_gauge_idempotent_on_isometry() {
    for rule_case in [0usize, 1usize] {
        if rule_case == 0 {
            let rule = Z2FusionRule;
            let tensor = tsvd_test_tensor(&rule, &[SectorId::new(0), SectorId::new(1)]);
            let mut dense_executor = tenet_dense::DefaultDenseExecutor::new();
            let input = bound_tensor(Arc::new(rule), &tensor);
            let (q, _) = qr_compact(&mut dense_executor, &input.as_ref()).unwrap();
            let (q2, r2) = qr_compact(&mut dense_executor, &q.as_ref()).unwrap();
            assert_svd_blocks_match(&q, &q2);
            assert_identity_sector_matrices(&dense_sector_matrices(1, &r2));
        } else {
            let rule = SU2FusionRule;
            let tensor = tsvd_test_tensor(
                &rule,
                &[
                    SU2Irrep::from_twice_spin(0).sector_id(),
                    SU2Irrep::from_twice_spin(1).sector_id(),
                ],
            );
            let mut dense_executor = tenet_dense::DefaultDenseExecutor::new();
            let input = bound_tensor(Arc::new(rule), &tensor);
            let (q, _) = qr_compact(&mut dense_executor, &input.as_ref()).unwrap();
            let (q2, r2) = qr_compact(&mut dense_executor, &q.as_ref()).unwrap();
            assert_svd_blocks_match(&q, &q2);
            assert_identity_sector_matrices(&dense_sector_matrices(1, &r2));
        }
    }
}

#[test]
fn lq_compact_positive_gauge_idempotent_on_isometry() {
    let rule = Z2FusionRule;
    let tensor = tsvd_test_tensor(&rule, &[SectorId::new(0), SectorId::new(1)]);
    let mut dense_executor = tenet_dense::DefaultDenseExecutor::new();
    let input = bound_tensor(Arc::new(rule), &tensor);
    let (_, q) = lq_compact(&mut dense_executor, &input.as_ref()).unwrap();
    let (l2, q2) = lq_compact(&mut dense_executor, &q.as_ref()).unwrap();
    assert_svd_blocks_match(&q, &q2);
    assert_identity_sector_matrices(&dense_sector_matrices(1, &l2));
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

// ============================================================================
// Issue #577: general (non-Hermitian) matrix exponential.
//
// Oracle provenance. Every `*_matches_the_tensorkit_oracle` constant below is
// TensorKit output for the same tensor, from
// `~/.julia/packages/TensorKit/6Camk` (0.16.2) on Julia 1.11.6:
//
// ```julia
// V = U1Space(0=>3, 1=>2)
// t = zeros(T, V <- V)
// for (c, b) in blocks(t), j in axes(b,2), i in axes(b,1)
//     re = 0.5 + 0.25*(i-1) - 0.75*(j-1) + 0.125*convert(Int, c.charge)
//     b[i,j] = (T <: Complex) ? complex(re, 0.125*(i-1) + 0.375*(j-1) - 0.25) : re
// end
// b .*= scale
// exp(t)
// ```
//
// The pinned 0.17.0 tree (`~/.julia/packages/TensorKit/jCjQQ`) was not the
// resolved version, which does not weaken the oracle: `exp!`
// (`src/tensors/linalg.jl:420-428`) is character-identical in the two trees and
// contains no arithmetic — it checks `domain == codomain` and hands every block
// to `LinearAlgebra.exp!`, so the numbers are Julia stdlib v1.11's, not
// TensorKit's, in either tree.
//
// Fixture certification: the fill is a closed formula in the coupled charge and
// the two degeneracy indices, and `u1_block_endomorphism` places one fusion
// tree per coupled sector, so `u1_block_matrix` reads back exactly the matrix
// Julia's `blocks(t)` iterated over. `exp_fixture_blocks_reproduce_the_oracle_input`
// pins that correspondence through the pre-existing reader before any exponential
// is taken.
// ============================================================================

/// Relative Frobenius agreement with the TensorKit oracle, per the #577 design.
const EXP_ORACLE_RTOL: f64 = 1e-12;

/// Residual of `exp(A) exp(-A) - 1`, per the #577 design.
const EXP_INVERSE_RTOL: f64 = 1e-11;

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

/// Column-major coupled-sector matrix of a single-tree `1 <- 1` U(1) tensor.
fn u1_block_matrix<D: Copy + Zero>(tensor: &TensorMap<D, 1, 1>, charge: i32) -> Vec<D> {
    let sector = U1Irrep::new(charge).sector_id();
    let structure = tensor.structure();
    let block = (0..structure.block_count())
        .map(|index| structure.block(index).unwrap())
        .find(|block| match block.key() {
            BlockKey::FusionTree(key) => key.codomain_tree().coupled() == sector,
            _ => false,
        })
        .unwrap();
    let order = block.shape()[0];
    assert_eq!(block.shape(), &[order, order]);
    let mut matrix = vec![D::zero(); order * order];
    for column in 0..order {
        for row in 0..order {
            matrix[row + order * column] = tensor.data()
                [block.offset() + row * block.strides()[0] + column * block.strides()[1]];
        }
    }
    matrix
}

/// Relative Frobenius comparison against a row-major `(re, im)` oracle block.
fn assert_sector_matrix_matches<D: FactorScalar>(
    actual: &[D],
    expected_rows: &[&[(f64, f64)]],
    rtol: f64,
    what: &str,
) {
    let order = expected_rows.len();
    assert_eq!(actual.len(), order * order, "{what}: unexpected block size");
    let mut residual = 0.0_f64;
    let mut reference = 0.0_f64;
    for (row, entries) in expected_rows.iter().enumerate() {
        assert_eq!(entries.len(), order, "{what}: oracle block is not square");
        for (column, &(real, imaginary)) in entries.iter().enumerate() {
            let expected = Complex64::new(real, imaginary);
            let got = actual[row + order * column].widen_complex();
            residual += (got - expected).norm_sqr();
            reference += expected.norm_sqr();
        }
    }
    let relative = residual.sqrt() / reference.sqrt();
    assert!(
        relative <= rtol,
        "{what}: relative Frobenius error {relative:e} exceeds {rtol:e}"
    );
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

/// An executor whose selected backend supplies GEMM but no dense solve — the
/// trait default for `solve_into` is what must reach the caller.
#[derive(Default)]
struct SolvelessExecutor {
    inner: tenet_dense::DefaultDenseExecutor,
}

impl DenseExecutor for SolvelessExecutor {
    fn svd(&mut self, input: DenseRead<'_>) -> Result<Vec<DenseTensor>, DenseError> {
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

#[test]
fn exp_fixture_blocks_reproduce_the_oracle_input() {
    // What: fixture certification. Before any exponential is taken, the blocks
    // TeNeT stores are the blocks Julia's `blocks(t)` filled — read back with
    // the pre-existing block reader, not with anything #577 introduced.
    let tensor = exp_oracle_tensor::<f64>(4.0);
    for (charge, order) in [(0, 3usize), (1, 2usize)] {
        let block = u1_block_matrix(&tensor, charge);
        for column in 0..order {
            for row in 0..order {
                assert_eq!(
                    block[row + order * column],
                    exp_oracle_fill(charge, row, column, 4.0),
                    "fixture entry ({row}, {column}) of charge {charge}"
                );
            }
        }
    }
    // Julia `opnorm(b, 1)` of the same blocks: 9.0 and 6.0, both above
    // theta_13, so the fixture exercises the scaling-and-squaring phase.
    for (charge, order, expected) in [(0, 3usize, 9.0), (1, 2usize, 6.0)] {
        let block = u1_block_matrix(&tensor, charge);
        let norm1 = (0..order)
            .map(|column| {
                (0..order)
                    .map(|row| block[row + order * column].abs())
                    .sum::<f64>()
            })
            .fold(0.0_f64, f64::max);
        assert_eq!(norm1, expected, "one-norm of charge {charge}");
    }
}

#[test]
fn exp_of_a_nilpotent_jordan_block_is_the_terminating_series() {
    // What: a nonnormal, non-diagonalizable block the spectral route cannot
    // touch. exp(J) = 1 + J + J^2/2 + J^3/6 exactly, so the approximant, the
    // scaling choice and the solve are all pinned against closed form.
    let mut jordan = vec![0.0_f64; 16];
    for row in 0..3 {
        jordan[row + 4 * (row + 1)] = 1.0;
    }
    let tensor = u1_block_endomorphism(&[(0, 4, jordan)]);
    let mut dense = tenet_dense::DefaultDenseExecutor::new();
    let mut context = default_context();

    let exponential = exp(
        &mut dense,
        &mut context,
        &bound_tensor_ref!(Arc::new(U1FusionRule), &tensor),
    )
    .unwrap();

    let expected = [
        [(1.0, 0.0), (1.0, 0.0), (0.5, 0.0), (1.0 / 6.0, 0.0)],
        [(0.0, 0.0), (1.0, 0.0), (1.0, 0.0), (0.5, 0.0)],
        [(0.0, 0.0), (0.0, 0.0), (1.0, 0.0), (1.0, 0.0)],
        [(0.0, 0.0), (0.0, 0.0), (0.0, 0.0), (1.0, 0.0)],
    ];
    assert_sector_matrix_matches(
        &u1_block_matrix(exponential.tensor(), 0),
        &expected.iter().map(|row| &row[..]).collect::<Vec<_>>(),
        1e-14,
        "nilpotent Jordan block",
    );
}

#[test]
fn exp_of_a_real_skew_symmetric_block_is_the_analytic_rotation() {
    // What: skew-symmetric is anti-Hermitian, so the eigh gate refuses it while
    // its exponential is the exact rotation by the same angle — at any angle.
    // The large one is the value-level gate on scaling and squaring: at
    // ||A||_1 = 20 the [13/13] approximant is far outside its accuracy range,
    // so an unscaled evaluation misses the rotation by orders of magnitude.
    for angle in [0.7_f64, 20.0] {
        let tensor = u1_block_endomorphism(&[(0, 2, vec![0.0_f64, angle, -angle, 0.0])]);
        let mut dense = tenet_dense::DefaultDenseExecutor::new();
        let mut context = default_context();

        let exponential = exp(
            &mut dense,
            &mut context,
            &bound_tensor_ref!(Arc::new(U1FusionRule), &tensor),
        )
        .unwrap();

        let expected = [
            [(angle.cos(), 0.0), (-angle.sin(), 0.0)],
            [(angle.sin(), 0.0), (angle.cos(), 0.0)],
        ];
        assert_sector_matrix_matches(
            &u1_block_matrix(exponential.tensor(), 0),
            &expected.iter().map(|row| &row[..]).collect::<Vec<_>>(),
            1e-13,
            &format!("skew-symmetric rotation by {angle}"),
        );
    }
}

#[test]
// Verbatim `%.17g` TensorKit output. Trimming the literals to the shortest
// round-tripping form would obscure that provenance for no gain.
#[allow(clippy::excessive_precision)]
fn exp_of_a_multisector_u1_endomorphism_matches_the_tensorkit_oracle() {
    // What: two coupled sectors of different order, at a scale below and above
    // theta_13, against frozen TensorKit values.
    for (scale, charge_zero, charge_one) in [
        (
            1.0,
            &[
                [
                    (0.9849644284568706, 0.0),
                    (-0.37626016105405391, 0.0),
                    (-0.73748475056497831, 0.0),
                ],
                [
                    (0.44650857769439511, 0.0),
                    (0.82943202363305835, 0.0),
                    (-0.78764453042827864, 0.0),
                ],
                [
                    (0.90805272693191974, 0.0),
                    (0.035124208320170609, 0.0),
                    (0.16219568970842124, 0.0),
                ],
            ],
            &[
                [(1.7819357803578582, 0.0), (-0.18045636327066489, 0.0)],
                [(1.2631945428946545, 0.0), (1.0601103272751986, 0.0)],
            ],
        ),
        (
            4.0,
            &[
                [
                    (-0.6308945936141368, 0.0),
                    (-0.27404909616219597, 0.0),
                    (1.0827964012897446, 0.0),
                ],
                [
                    (-1.1147351879032152, 0.0),
                    (0.51577938090254927, 0.0),
                    (0.14629394970831344, 0.0),
                ],
                [
                    (-0.59857578219229368, 0.0),
                    (-0.69439214203270549, 0.0),
                    (0.20979149812688241, 0.0),
                ],
            ],
            &[
                [(6.8456187388272456, 0.0), (-1.9710572969428048, 0.0)],
                [(13.797401078599625, 0.0), (-1.0386104489439727, 0.0)],
            ],
        ),
    ] {
        let tensor = exp_oracle_tensor::<f64>(scale);
        let mut dense = tenet_dense::DefaultDenseExecutor::new();
        let mut context = default_context();

        let exponential = exp(
            &mut dense,
            &mut context,
            &bound_tensor_ref!(Arc::new(U1FusionRule), &tensor),
        )
        .unwrap();

        assert_sector_matrix_matches(
            &u1_block_matrix(exponential.tensor(), 0),
            &charge_zero.iter().map(|row| &row[..]).collect::<Vec<_>>(),
            EXP_ORACLE_RTOL,
            &format!("f64 scale {scale} charge 0"),
        );
        assert_sector_matrix_matches(
            &u1_block_matrix(exponential.tensor(), 1),
            &charge_one.iter().map(|row| &row[..]).collect::<Vec<_>>(),
            EXP_ORACLE_RTOL,
            &format!("f64 scale {scale} charge 1"),
        );
    }
}

#[test]
// Verbatim `%.17g` TensorKit output. Trimming the literals to the shortest
// round-tripping form would obscure that provenance for no gain.
#[allow(clippy::excessive_precision)]
fn exp_of_a_complex_nonnormal_u1_endomorphism_matches_the_tensorkit_oracle() {
    // What: c64, nonnormal and non-Hermitian in both the real and imaginary
    // parts — the arm where a real-arithmetic slip in the approximant shows up.
    let tensor = exp_oracle_tensor::<Complex64>(1.0);
    let mut dense = tenet_dense::DefaultDenseExecutor::new();
    let mut context = TensorContractFusionExecutionContext::<Complex64, RuleIdentity>::default();

    let exponential = exp(
        &mut dense,
        &mut context,
        &bound_tensor_ref!(Arc::new(U1FusionRule), &tensor),
    )
    .unwrap();

    let charge_zero = [
        [
            (0.92690089541443432, -0.1610299101898417),
            (-0.41081187668012081, -0.030497613725301645),
            (-0.74852464877467562, 0.10003468273923843),
        ],
        [
            (0.36740563761195039, 0.071261345504936915),
            (0.73145191286879296, 0.12778526275599536),
            (-0.9045018118743644, 0.18430918000705374),
        ],
        [
            (0.80791037980946623, 0.30355260119971544),
            (-0.12628429758229334, 0.28606813923729235),
            (-0.060478974974053017, 0.26858367727486893),
        ],
    ];
    let charge_one = [
        [
            (1.7454107392461176, -0.35809087573900988),
            (-0.17904543786950494, 0.179045437869505),
        ],
        [
            (1.2533180650865348, -0.17904543786950497),
            (1.0292289877680976, 0.35809087573900988),
        ],
    ];
    assert_sector_matrix_matches(
        &u1_block_matrix(exponential.tensor(), 0),
        &charge_zero.iter().map(|row| &row[..]).collect::<Vec<_>>(),
        EXP_ORACLE_RTOL,
        "c64 charge 0",
    );
    assert_sector_matrix_matches(
        &u1_block_matrix(exponential.tensor(), 1),
        &charge_one.iter().map(|row| &row[..]).collect::<Vec<_>>(),
        EXP_ORACLE_RTOL,
        "c64 charge 1",
    );
}

#[test]
fn exp_agrees_between_the_direct_region_and_packed_layouts() {
    // What: the same blocks reached through the canonical coupled-sector
    // regions and through the packed matricization fall-back produce the same
    // tensor — the kernel sees identical matrices on both routes.
    let tensor = exp_oracle_tensor::<f64>(4.0);
    let packed = padded_copy(&U1FusionRule, &tensor);
    assert!(tensor
        .structure()
        .coupled_sector_regions(1)
        .unwrap()
        .is_some());
    assert!(packed
        .structure()
        .coupled_sector_regions(1)
        .unwrap()
        .is_none());
    let mut dense = tenet_dense::DefaultDenseExecutor::new();
    let mut context = default_context();

    let direct = exp(
        &mut dense,
        &mut context,
        &bound_tensor_ref!(Arc::new(U1FusionRule), &tensor),
    )
    .unwrap();
    let fallback = exp(
        &mut dense,
        &mut context,
        &bound_tensor_ref!(Arc::new(U1FusionRule), &packed),
    )
    .unwrap();

    for charge in [0, 1] {
        assert_eq!(
            u1_block_matrix(direct.tensor(), charge),
            u1_block_matrix(fallback.tensor(), charge),
            "packed layout changed the exponential of charge {charge}"
        );
    }
}

// ============================================================================
// Multi-tree coupled sectors.
//
// Every fixture above places exactly one fusion tree per coupled sector, so a
// permutation of whole tree-aligned row blocks inside a sector is invisible to
// them: the sector matrix has only one block to permute. `tsvd_test_tensor` at
// rank 2 <- 2 with three U(1) charges and degeneracy 2 gives coupled sectors
// carrying one, two and three trees (orders 4, 8, 12), so the tree layout
// inside a sector matrix has something to get wrong. The oracle below is the
// defining series, which shares no scaling rule, no Padé table, no solve and no
// squaring loop with the implementation, and is read back through the
// pre-existing `coupled_sector_regions` reader.
// ============================================================================

/// Scale of the multi-tree fixture. Chosen so the largest sector 1-norm sits
/// just above `theta_13` — the squaring phase runs, and the series oracle is
/// still deep inside its convergent range at 60 terms.
const EXP_MULTITREE_SCALE: f64 = 0.3;

/// Entrywise agreement with the series oracle, relative to the largest entry of
/// the sector. Loose next to the Padé error because the series is summed in the
/// fixture's own (non-normal) basis; still orders of magnitude below any
/// misplaced tree block, which moves entries by O(1).
const EXP_MULTITREE_TOL: f64 = 1e-11;

/// `exp(A)` by its defining series `Σ_k A^k / k!`, summed to `terms`.
///
/// Deliberately shares nothing with the implementation: no scaling, no
/// squaring, no Padé coefficients, no dense solve — only GEMM by hand.
fn taylor_exp(matrix: &[f64], order: usize, terms: usize) -> Vec<f64> {
    let mut result = vec![0.0_f64; order * order];
    let mut term = vec![0.0_f64; order * order];
    for index in 0..order {
        term[index + order * index] = 1.0;
        result[index + order * index] = 1.0;
    }
    let mut next = vec![0.0_f64; order * order];
    for step in 1..=terms {
        for column in 0..order {
            for row in 0..order {
                let mut sum = 0.0_f64;
                for inner in 0..order {
                    sum += term[row + order * inner] * matrix[inner + order * column];
                }
                next[row + order * column] = sum / step as f64;
            }
        }
        term.copy_from_slice(&next);
        for (accumulated, &value) in result.iter_mut().zip(term.iter()) {
            *accumulated += value;
        }
    }
    result
}

/// Column-major coupled-sector matrices of a canonical layout, read through the
/// pre-existing region reader.
fn coupled_sector_matrices<const NOUT: usize, const NIN: usize>(
    tensor: &TensorMap<f64, NOUT, NIN>,
) -> Vec<(SectorId, usize, Vec<f64>)> {
    tensor
        .structure()
        .coupled_sector_regions(NOUT)
        .unwrap()
        .expect("canonical coupled-sector storage")
        .iter()
        .map(|region| {
            assert_eq!(
                region.rows(),
                region.cols(),
                "endomorphism sector must be square"
            );
            (
                region.coupled(),
                region.rows(),
                tensor.data()[region.range()].to_vec(),
            )
        })
        .collect()
}

/// Rank 2 <- 2 U(1) endomorphism whose coupled sectors carry one, two and three
/// fusion trees, scaled into the oracle's comfortable range.
fn exp_multitree_fixture() -> TensorMap<f64, 2, 2> {
    let sectors: Vec<SectorId> = (0..3).map(|c| U1Irrep::new(c).sector_id()).collect();
    let tensor = tsvd_test_tensor(&U1FusionRule, &sectors);
    TensorMap::<f64, 2, 2>::from_vec_with_fusion_space(
        tensor
            .data()
            .iter()
            .map(|value| value * EXP_MULTITREE_SCALE)
            .collect(),
        tensor.fusion_space().unwrap().as_ref().clone(),
    )
    .unwrap()
}

/// Entrywise comparison of a whole tensor's sectors against the series oracle.
fn assert_multitree_exp_matches_the_series(
    exponential: &TensorMap<f64, 2, 2>,
    sources: &[(SectorId, usize, Vec<f64>)],
    what: &str,
) {
    let images = coupled_sector_matrices(exponential);
    assert_eq!(images.len(), sources.len(), "{what}: sector count changed");
    for ((sector, order, source), (image_sector, image_order, image)) in
        sources.iter().zip(images.iter())
    {
        assert_eq!(
            (sector, order),
            (image_sector, image_order),
            "{what}: layout"
        );
        let expected = taylor_exp(source, *order, 60);
        let tolerance = EXP_MULTITREE_TOL * expected.iter().fold(1.0_f64, |m, v| m.max(v.abs()));
        for column in 0..*order {
            for row in 0..*order {
                let index = row + order * column;
                let residual = (image[index] - expected[index]).abs();
                assert!(
                    residual <= tolerance,
                    "{what}: sector {sector:?} entry ({row}, {column}) is {} against series {}, \
                     residual {residual:e}",
                    image[index],
                    expected[index]
                );
            }
        }
    }
}

#[test]
fn exp_of_a_multi_tree_sector_matches_the_series_entrywise() {
    // What: the tree layout *inside* a coupled sector. Permuting whole
    // tree-aligned row blocks of a multi-tree sector preserves every norm, the
    // sector count, the solve count and the round trip through exp(-A), so only
    // an entrywise comparison against an independently computed exponential of
    // the *same* source matrix can see it.
    let tensor = exp_multitree_fixture();
    let sources = coupled_sector_matrices(&tensor);
    // Certify the fixture actually is multi-tree, and that it exercises the
    // general arm at a scale where squaring happens.
    assert_eq!(
        sources
            .iter()
            .map(|(_, order, _)| *order)
            .collect::<Vec<_>>(),
        vec![4, 8, 12, 8, 4],
        "fixture no longer has one-, two- and three-tree coupled sectors"
    );
    let widest = sources
        .iter()
        .map(|(_, order, matrix)| {
            (0..*order)
                .map(|column| {
                    (0..*order)
                        .map(|row| matrix[row + order * column].abs())
                        .sum::<f64>()
                })
                .fold(0.0_f64, f64::max)
        })
        .fold(0.0_f64, f64::max);
    assert!(
        // theta_13 = 5.3719..., so this window means s >= 1.
        widest > 5.372 && widest < 12.0,
        "fixture 1-norm {widest} is outside the scaling-and-squaring window"
    );

    let mut spy = MatrixFunctionCallSpy::default();
    let mut context = default_context();
    let exponential = exp(
        &mut spy,
        &mut context,
        &bound_tensor_ref!(Arc::new(U1FusionRule), &tensor),
    )
    .unwrap();
    assert_eq!(spy.eigh_calls, 0, "the fixture must reach the general arm");
    assert_eq!(spy.solve_calls, sources.len(), "one solve per sector");

    assert_multitree_exp_matches_the_series(exponential.tensor(), &sources, "direct regions");
}

#[test]
fn exp_of_a_multi_tree_sector_matches_the_series_through_the_packed_layout() {
    // What: the same gate on the matricization fall-back, where the sector
    // matrix is assembled tree by tree and written back through
    // `reorder_inverse_solution` — the route that has to rebuild the tree
    // layout rather than inherit it.
    let tensor = exp_multitree_fixture();
    let packed = padded_copy(&U1FusionRule, &tensor);
    assert!(
        packed
            .structure()
            .coupled_sector_regions(2)
            .unwrap()
            .is_none(),
        "the padded copy must take the matricization fall-back"
    );
    let sources = coupled_sector_matrices(&tensor);

    let mut dense = tenet_dense::DefaultDenseExecutor::new();
    let mut context = default_context();
    let exponential = exp(
        &mut dense,
        &mut context,
        &bound_tensor_ref!(Arc::new(U1FusionRule), &packed),
    )
    .unwrap();

    // The output layout is always canonical, so the same reader applies.
    assert_multitree_exp_matches_the_series(exponential.tensor(), &sources, "packed layout");
}

#[test]
fn exp_of_a_general_endomorphism_inverts_under_negation() {
    // What: exp(A) exp(-A) = 1 for a non-Hermitian A, the check that catches a
    // sign or a transposition the oracle blocks would also have to agree with.
    let tensor = exp_oracle_tensor::<f64>(1.0);
    let negated = TensorMap::<f64, 1, 1>::from_vec_with_fusion_space(
        tensor.data().iter().map(|value| -value).collect(),
        tensor.fusion_space().unwrap().as_ref().clone(),
    )
    .unwrap();
    let rule = U1FusionRule;
    let mut dense = tenet_dense::DefaultDenseExecutor::new();
    let mut context = default_context();

    let forward = exp(
        &mut dense,
        &mut context,
        &bound_tensor_ref!(Arc::new(rule), &tensor),
    )
    .unwrap();
    let backward = exp(
        &mut dense,
        &mut context,
        &bound_tensor_ref!(Arc::new(rule), &negated),
    )
    .unwrap();

    let identity =
        crate::compose::compose(&mut context, &rule, forward.tensor(), backward.tensor()).unwrap();
    for (charge, order) in [(0, 3usize), (1, 2usize)] {
        let block = u1_block_matrix(&identity, charge);
        for column in 0..order {
            for row in 0..order {
                let expected = if row == column { 1.0 } else { 0.0 };
                let residual = (block[row + order * column] - expected).abs();
                assert!(
                    residual <= EXP_INVERSE_RTOL,
                    "charge {charge} entry ({row}, {column}) residual {residual:e}"
                );
            }
        }
    }
}

fn hermitian_exp_fixture() -> TensorMap<f64, 1, 1> {
    u1_block_endomorphism(&[
        (
            0,
            3,
            vec![0.5, 0.25, -0.125, 0.25, -0.75, 0.375, -0.125, 0.375, 1.25],
        ),
        (1, 2, vec![0.25, 0.5, 0.5, -0.125]),
    ])
}

#[test]
fn exp_of_a_hermitian_endomorphism_is_the_spectral_route_bit_for_bit() {
    // What: the retained route, pinned two ways — the dispatch (one EIGH per
    // sector, no solve, no GEMM) and the published values, byte for byte
    // against `v exp(d) v^H` computed here on the same backend.
    //
    // The reference is computed rather than frozen because a frozen one is a
    // pin on the platform's LAPACK: the constants this test used to carry were
    // right on macOS and a few ULP off on Linux, so CI failed on values that
    // were never the point. Byte-identity against the spectral route is, and it
    // still catches a reroute onto Pade, whose approximant does not land on the
    // eigendecomposition's last bits.
    let tensor = hermitian_exp_fixture();
    let mut spy = MatrixFunctionCallSpy::default();
    let mut context = default_context();

    let exponential = exp(
        &mut spy,
        &mut context,
        &bound_tensor_ref!(Arc::new(U1FusionRule), &tensor),
    )
    .unwrap();

    assert_eq!(
        spy.eigh_calls, 2,
        "one eigendecomposition per coupled sector"
    );
    assert_eq!(spy.solve_calls, 0, "the Hermitian route must not solve");
    assert_eq!(spy.matmul_calls, 0, "the Hermitian route must not GEMM");

    let mut dense = tenet_dense::DefaultDenseExecutor::new();
    let bound = bound_tensor(Arc::new(U1FusionRule), &tensor);
    let spectral = crate::matrix_functions::spectral_function_dyn(
        &mut dense,
        &mut context,
        &bound.as_ref().dynamic(),
        &f64::exp,
    )
    .unwrap();
    let spectral: BoundTensorMap<_, _, 1, 1> = typed_from_bound_factor(spectral).unwrap();
    assert_eq!(exponential.tensor().data(), spectral.tensor().data());
}

#[test]
fn exp_of_a_hermitian_c64_endomorphism_takes_the_spectral_route() {
    // What: the *dispatch*, not the values. The retained Hermitian route is
    // pinned bit for bit only on U(1)/f64 above, so a hermiticity predicate that
    // misclassified complex input would silently reroute it onto Pade with
    // nothing failing. A solve on Hermitian input is the observable.
    let half = Complex64::new(0.5, 0.0);
    let charge_zero = vec![
        half,
        Complex64::new(0.25, -0.125),
        Complex64::new(-0.125, 0.375),
        Complex64::new(0.25, 0.125),
        Complex64::new(-0.75, 0.0),
        Complex64::new(0.375, -0.25),
        Complex64::new(-0.125, -0.375),
        Complex64::new(0.375, 0.25),
        Complex64::new(1.25, 0.0),
    ];
    let charge_one = vec![
        Complex64::new(1.0, 0.0),
        Complex64::new(0.25, 0.5),
        Complex64::new(0.25, -0.5),
        Complex64::new(-0.75, 0.0),
    ];
    let tensor = u1_block_endomorphism(&[(0, 3, charge_zero), (1, 2, charge_one)]);
    let mut spy = MatrixFunctionCallSpy::default();
    let mut context = TensorContractFusionExecutionContext::<Complex64, RuleIdentity>::default();

    exp(
        &mut spy,
        &mut context,
        &bound_tensor_ref!(Arc::new(U1FusionRule), &tensor),
    )
    .unwrap();

    assert_eq!(
        spy.eigh_calls, 2,
        "one eigendecomposition per coupled sector"
    );
    assert_eq!(
        spy.solve_calls, 0,
        "a Hermitian c64 input must not reach the Pade solve"
    );
}

#[test]
fn exp_of_a_hermitian_su2_endomorphism_takes_the_spectral_route() {
    // What: the same dispatch gate on a non-abelian rule, where the coupled
    // sector matrix carries the recoupling structure the predicate reads.
    let rule = SU2FusionRule;
    let symmetric = |a: f64, b: f64, c: f64| vec![a, b, b, c];
    let blocks = [
        (
            SU2Irrep::from_twice_spin(0).sector_id(),
            2usize,
            symmetric(0.5, 0.25, -0.75),
        ),
        (
            SU2Irrep::from_twice_spin(1).sector_id(),
            2usize,
            symmetric(1.25, -0.375, 0.625),
        ),
        (
            SU2Irrep::from_twice_spin(2).sector_id(),
            2usize,
            symmetric(-0.25, 0.5, 0.125),
        ),
    ];
    let tensor = block_endomorphism(&rule, &blocks);
    let mut spy = MatrixFunctionCallSpy::default();
    let mut context = default_context();

    exp(
        &mut spy,
        &mut context,
        &bound_tensor_ref!(Arc::new(rule), &tensor),
    )
    .unwrap();

    assert_eq!(
        spy.eigh_calls, 3,
        "one eigendecomposition per coupled sector"
    );
    assert_eq!(
        spy.solve_calls, 0,
        "a Hermitian SU(2) input must not reach the Pade solve"
    );
}

#[test]
fn exp_of_a_general_endomorphism_runs_one_solve_per_sector_and_no_eigh() {
    // What: the general arm's per-sector budget — six GEMMs and one solve for a
    // block that needs no squaring, and never an eigendecomposition.
    let tensor = exp_oracle_tensor::<f64>(1.0);
    let mut spy = MatrixFunctionCallSpy::default();
    let mut context = default_context();

    exp(
        &mut spy,
        &mut context,
        &bound_tensor_ref!(Arc::new(U1FusionRule), &tensor),
    )
    .unwrap();

    assert_eq!(spy.eigh_calls, 0, "the general arm must not eigendecompose");
    assert_eq!(spy.solve_calls, 2, "one solve per nonempty coupled sector");
    assert_eq!(spy.matmul_calls, 12, "six GEMMs per sector at s = 0");
}

#[test]
fn exp_sector_work_scales_with_the_sector_count_and_the_scaling_count() {
    // What: complexity parity. Doubling the number of equal-size sectors
    // doubles the block work instead of coupling them into one dense cube, and
    // a block above theta_13 pays exactly its squarings.
    let block = |charge| exp_oracle_block::<f64>(charge, 2, 1.0);
    let two = u1_block_endomorphism(&[(0, 2, block(0)), (1, 2, block(1))]);
    let four = u1_block_endomorphism(&[
        (0, 2, block(0)),
        (1, 2, block(1)),
        (2, 2, block(2)),
        (3, 2, block(3)),
    ]);
    let mut context = default_context();

    let mut two_spy = MatrixFunctionCallSpy::default();
    exp(
        &mut two_spy,
        &mut context,
        &bound_tensor_ref!(Arc::new(U1FusionRule), &two),
    )
    .unwrap();
    let mut four_spy = MatrixFunctionCallSpy::default();
    exp(
        &mut four_spy,
        &mut context,
        &bound_tensor_ref!(Arc::new(U1FusionRule), &four),
    )
    .unwrap();

    assert_eq!(four_spy.solve_calls, 2 * two_spy.solve_calls);
    assert_eq!(four_spy.matmul_calls, 2 * two_spy.matmul_calls);

    // ||A||_1 = 36 for this block, so s = ceil(log2(36 / theta_13)) = 3 and the
    // squaring loop adds exactly three GEMMs on top of the six.
    let scaled = u1_block_endomorphism(&[(0, 3, exp_oracle_block::<f64>(0, 3, 16.0))]);
    let mut scaled_spy = MatrixFunctionCallSpy::default();
    exp(
        &mut scaled_spy,
        &mut context,
        &bound_tensor_ref!(Arc::new(U1FusionRule), &scaled),
    )
    .unwrap();
    assert_eq!(scaled_spy.matmul_calls, 9);
    assert_eq!(scaled_spy.solve_calls, 1);
}

#[test]
fn exp_balances_a_badly_scaled_block_before_the_pade_evaluation() {
    // What: the balancing step Julia's `exp!` runs before the approximant
    // (`LAPACK.gebal!('B', A)`, stdlib v1.11 `dense.jl:684`) and undoes after
    // it, pinned two ways on `A = [0 1e16; 1e-16 0]`, whose square is the
    // identity and whose exponential is therefore `cosh(1) I + sinh(1) A`.
    // Unbalanced the block has `||A||_1 = 1e16` and pays 51 squarings, each one
    // squaring the approximant's error along with it; balanced it has norm
    // ~1.11, below theta_13, so the *dispatch* — six GEMMs and no squaring —
    // is the sharpest observable there is.
    let tensor = u1_block_endomorphism(&[(0, 2, vec![0.0_f64, 1e-16, 1e16, 0.0])]);
    let mut spy = MatrixFunctionCallSpy::default();
    let mut context = default_context();

    let exponential = exp(
        &mut spy,
        &mut context,
        &bound_tensor_ref!(Arc::new(U1FusionRule), &tensor),
    )
    .unwrap();

    assert_eq!(
        spy.matmul_calls, 6,
        "the balanced block is below theta_13: six GEMMs, no squaring"
    );
    let cosh = 1.0_f64.cosh();
    let sinh = 1.0_f64.sinh();
    let expected = [
        [(cosh, 0.0), (sinh * 1e16, 0.0)],
        [(sinh * 1e-16, 0.0), (cosh, 0.0)],
    ];
    assert_sector_matrix_matches(
        &u1_block_matrix(exponential.tensor(), 0),
        &expected.iter().map(|row| &row[..]).collect::<Vec<_>>(),
        1e-14,
        "balanced [0 1e16; 1e-16 0]",
    );
}

#[test]
// `%.17g` of a 400-bit BigFloat oracle; see the comment below.
#[allow(clippy::excessive_precision)]
fn exp_undoes_the_balancing_permutation_and_scaling_like_julia() {
    // What: the *undo* half of balancing, on a block that exercises both halves
    // of `gebal('B')` at once — a column that isolates an eigenvalue, so the
    // permutation moves it to the front, and a badly scaled remainder, so the
    // diagonal similarity acts on the window behind it. The regression fixture
    // above needs no permutation and so cannot see an undo applied in the wrong
    // order; this one can.
    //
    // That both halves fired is asserted directly on `balance_in_place`, against
    // Julia 1.11.6:
    //
    // ```julia
    // d = [1e8, 1.0, 1e-8]
    // B = [0.0 0.0 1.0; 2.0 3.0 5.0; 1.0 0.0 0.0]
    // A = [B[i, j] * d[i] / d[j] for i in 1:3, j in 1:3]
    // LinearAlgebra.LAPACK.gebal!('B', copy(A))
    // # (2, 3, [2.0, 9.007199254740992e15, 1.0])       # 9.007...e15 == 2^53
    // ```
    //
    // `ilo, ihi = 2, 3` is the permutation, `scale[2] = 2^53` the scaling.
    //
    // The *values* are checked against an exact oracle rather than against a
    // recorded `exp(A)`: `A = D B D^-1`, so `exp(A) = D exp(B) D^-1` entry by
    // entry, and `exp(B)` is available in closed form — its `{1,3}` corner is
    // `[cosh 1, sinh 1; sinh 1, cosh 1]`, its `(2,2)` entry `e^3`, and the two
    // remaining entries are the integrals `e^3 (7 (1 - e^-2) -/+ 1.5 (1 - e^-4)) / 4`.
    // The constants below are that oracle to 17 digits, cross-checked against a
    // 400-bit BigFloat scaling-and-squaring Taylor evaluation of `exp(B)` in
    // Julia. Recording a `Float64` `exp(A)` instead is what broke CI: the
    // fixture's own conditioning makes the answer platform-dependent in the
    // ninth digit, so no recorded double is portable at a tight tolerance.
    //
    // The tolerance is that conditioning, made explicit: balancing equalizes the
    // window's norms without seeing the isolated column, so the balanced block
    // still has `||A||_1 ~ 7e8` and takes 27 squarings, each of which roughly
    // doubles the relative error. `2^27 * eps ~ 3.0e-8` bounds it; `1e-7` is
    // that bound with a factor of three of room. Julia's own answer sits at
    // 2.24e-8 on macOS and 2.15e-8 on Linux, i.e. the *whole* macOS/Linux spread
    // is 1.1e-9, two orders inside the gate. The structural zeros stay EXACT.
    let diagonal = [1e8_f64, 1.0, 1e-8];
    let rows = [[0.0_f64, 0.0, 1.0], [2.0, 3.0, 5.0], [1.0, 0.0, 0.0]];
    let mut block = vec![0.0_f64; 9];
    for (row, entries) in rows.iter().enumerate() {
        for (column, &entry) in entries.iter().enumerate() {
            block[row + 3 * column] = entry * diagonal[row] / diagonal[column];
        }
    }
    assert_gebal_matches(
        &block,
        3,
        (1, 2),
        &[2.0, 2.0_f64.powi(53), 1.0],
        "the permuting-and-scaling fixture",
    );

    let tensor = u1_block_endomorphism(&[(0, 3, block)]);
    let mut dense = tenet_dense::DefaultDenseExecutor::new();
    let mut context = default_context();

    let exponential = exp(
        &mut dense,
        &mut context,
        &bound_tensor_ref!(Arc::new(U1FusionRule), &tensor),
    )
    .unwrap();

    let expected = [
        [1.5430806348152437_f64, 0.0, 1.1752011936438014e16],
        [
            2.2998574860019004e-07,
            20.085536923187668,
            3778681797.1531172,
        ],
        [1.1752011936438015e-16, 0.0, 1.5430806348152437],
    ];
    let actual = u1_block_matrix(exponential.tensor(), 0);
    for (row, entries) in expected.iter().enumerate() {
        for (column, &want) in entries.iter().enumerate() {
            let got = actual[row + 3 * column];
            // The two structural zeros are the permutation's own signature: the
            // isolated column stays exactly isolated, to the last bit. Every
            // other entry gets the squaring-error budget derived above.
            let tolerance = if want == 0.0 { 0.0 } else { 1e-7 * want.abs() };
            assert!(
                (got - want).abs() <= tolerance,
                "entry ({row}, {column}): {got:.17e} differs from the oracle's {want:.17e}"
            );
        }
    }
}

/// Column-major `order x order` block from row-major rows.
fn column_major<D: Copy + Zero>(rows: &[&[D]]) -> Vec<D> {
    let order = rows.len();
    let mut block = vec![D::zero(); order * order];
    for (row, entries) in rows.iter().enumerate() {
        assert_eq!(entries.len(), order);
        for (column, &entry) in entries.iter().enumerate() {
            block[row + order * column] = entry;
        }
    }
    block
}

/// Asserts `balance_in_place` reproduces `LAPACK.gebal!('B', A)`, whose output
/// is quoted 0-based here: LAPACK's `ilo`/`ihi` are 1-based and inclusive,
/// TeNeT's are 0-based and inclusive.
fn assert_gebal_matches<D: FactorScalar>(
    block: &[D],
    order: usize,
    expected_window: (usize, usize),
    expected_scale: &[f64],
    what: &str,
) {
    let mut matrix = block.to_vec();
    let mut scale = vec![0.0_f64; order];
    let window = crate::matrix_functions::balance_in_place(&mut matrix, order, &mut scale);
    assert_eq!(window, expected_window, "{what}: window");
    assert_eq!(scale, expected_scale, "{what}: scale");
}

#[test]
fn balancing_measures_rows_and_columns_with_the_euclidean_norm_like_gebal() {
    // What: `gebal` measures its rows and columns with `DNRM2`
    // (`dgebal.f:341-342`), not with a sum of `abs1`. The fixture is chosen so
    // the two disagree — the `abs1` sum reaches `scale = [1, 1/2, 1]` here,
    // where LAPACK reaches `[2, 1, 1]` — so it is the norm itself under test
    // and not merely the loop around it.
    //
    // Oracle, Julia 1.11.6:
    //
    // ```julia
    // LinearAlgebra.LAPACK.gebal!('B', Float64[0 4 0; 1 0 1; 1 1 0])
    // # (1, 3, [2.0, 1.0, 1.0])
    // ```
    //
    // `Float32` and `ComplexF64` of the same matrix give the same triple, so
    // all three are pinned to the one certified answer.
    let rows: [&[f64]; 3] = [&[0.0, 4.0, 0.0], &[1.0, 0.0, 1.0], &[1.0, 1.0, 0.0]];
    let block = column_major(&rows);
    assert_gebal_matches(
        &block,
        3,
        (0, 2),
        &[2.0, 1.0, 1.0],
        "f64 [0 4 0; 1 0 1; 1 1 0]",
    );

    let single = block.iter().map(|&entry| entry as f32).collect::<Vec<_>>();
    assert_gebal_matches(&single, 3, (0, 2), &[2.0, 1.0, 1.0], "f32 of the same");

    let complex = block
        .iter()
        .map(|&entry| Complex64::new(entry, 0.0))
        .collect::<Vec<_>>();
    assert_gebal_matches(&complex, 3, (0, 2), &[2.0, 1.0, 1.0], "c64 of the same");

    // And the span is the whole window *including* the diagonal:
    // `DNRM2(L-K+1, A(K,I), 1)` is contiguous, it does not skip `A(I,I)`.
    // Dropping the diagonal reaches `[4, 1, 1]` on this second fixture.
    //
    // ```julia
    // LinearAlgebra.LAPACK.gebal!('B', Float64[-4 -1 -9; -1 1 -7; 0 -9 -7])
    // # (1, 3, [2.0, 1.0, 1.0])
    // ```
    let with_diagonal: [&[f64]; 3] = [&[-4.0, -1.0, -9.0], &[-1.0, 1.0, -7.0], &[0.0, -9.0, -7.0]];
    assert_gebal_matches(
        &column_major(&with_diagonal),
        3,
        (0, 2),
        &[2.0, 1.0, 1.0],
        "f64 [-4 -1 -9; -1 1 -7; 0 -9 -7]",
    );
}

#[test]
fn balancing_sizes_a_complex_iamax_element_by_its_modulus() {
    // What: `zgebal` *selects* `CA`/`RA` with `IZAMAX`, which compares
    // `|Re| + |Im|`, but then takes the **modulus** of the element it selected
    // (`zgebal.f:348-351`). The two differ by up to `sqrt(2)`, most of a radix
    // step, so they can stop the scaling loop one factor of two apart.
    //
    // The fixture makes that visible: column 1 isolates an eigenvalue, so the
    // window starts at row 2 while `CA` still scans row 1, letting the entry
    // `(3 + 4i) * 1e271` — `abs1` 7e271, modulus 5e271 — dominate `CA` and land
    // between `sfmax2 / 2` and `sfmax2` after the loop's doublings.
    //
    // Oracle, Julia 1.11.6:
    //
    // ```julia
    // A = ComplexF64[7 (3+4im)*1e271 0; 0 0 1e41; 0 1 0]
    // LinearAlgebra.LAPACK.gebal!('B', copy(A))
    // # (2, 3, [1.0, 1.4757395258967641e20, 0.5])   # 1.4757...e20 == 2^67
    // ```
    //
    // Sizing `CA`/`RA` by `abs1` instead gives `[1, 2^66, 0.25]`.
    let huge = Complex64::new(3.0, 4.0) * 1e271;
    let zero = Complex64::new(0.0, 0.0);
    let rows: [&[Complex64]; 3] = [
        &[Complex64::new(7.0, 0.0), huge, zero],
        &[zero, zero, Complex64::new(1e41, 0.0)],
        &[zero, Complex64::new(1.0, 0.0), zero],
    ];
    assert_gebal_matches(
        &column_major(&rows),
        3,
        (1, 2),
        &[1.0, 2.0_f64.powi(67), 0.5],
        "c64 modulus-vs-abs1 fixture",
    );
}

#[test]
fn balancing_takes_its_machine_bounds_from_the_single_precision_component() {
    // What: `sgebal`/`cgebal` derive `SFMIN1` and friends from `SLAMCH`
    // (`sgebal.f:330`), `dgebal`/`zgebal` from `DLAMCH` (`dgebal.f:330`). Under
    // the double bounds this fixture's radix loop runs to a factor around
    // `2^970`, which is `inf` in `f32`; under the single bounds it stops at
    // `2^102`, which is not.
    //
    // Oracle, Julia 1.11.6:
    //
    // ```julia
    // LinearAlgebra.LAPACK.gebal!('B', Float32[0 3f38; 1f-45 0])
    // # (1, 2, Float32[5.0706024f30, 1.4551915f-11])  # 2^102 and 2^-36
    // ```
    //
    // The same triple certifies `ComplexF32`.
    let rows: [&[f32]; 2] = [&[0.0, 3e38], &[1e-45, 0.0]];
    let expected = [2.0_f64.powi(102), 2.0_f64.powi(-36)];
    assert_gebal_matches(&column_major(&rows), 2, (0, 1), &expected, "f32 edge block");

    let complex = column_major(&rows)
        .iter()
        .map(|&entry| Complex32::new(entry, 0.0))
        .collect::<Vec<_>>();
    assert_gebal_matches(&complex, 2, (0, 1), &expected, "c32 edge block");
}

#[test]
fn exp_of_a_single_precision_block_spanning_the_whole_f32_range() {
    // What: the P1 regression, through the public facade. Every entry of
    // `A = [0 3e38; 1e-45 0]` and of its exponential is a finite `f32`, but
    // balancing with the *double* machine bounds produces a factor that is
    // `inf` in `f32`, and the Pade evaluation behind it went to all-NaN.
    //
    // Oracle, Julia 1.11.6: `exp(Float32[0 3f38; 1f-45 0])` is
    // `Float32[1.0000002 3.0000004f38; 1.0f-45 1.0000002]`, and the `ComplexF32`
    // matrix gives the same with zero imaginary parts. The tolerance is a
    // single-precision one: the oracle is quoted to the eight digits `f32`
    // carries.
    let rows: [&[f32]; 2] = [&[0.0, 3e38], &[1e-45, 0.0]];
    let expected: [&[(f64, f64)]; 2] = [
        &[(1.0000002, 0.0), (3.0000004e38, 0.0)],
        &[(1e-45, 0.0), (1.0000002, 0.0)],
    ];
    let mut dense = tenet_dense::DefaultDenseExecutor::new();

    let tensor = u1_block_endomorphism(&[(0, 2, column_major(&rows))]);
    let exponential = exp(
        &mut dense,
        &mut TensorContractFusionExecutionContext::<f32, RuleIdentity>::default(),
        &bound_tensor_ref!(Arc::new(U1FusionRule), &tensor),
    )
    .unwrap();
    assert_sector_matrix_matches(
        &u1_block_matrix(exponential.tensor(), 0),
        &expected,
        1e-6,
        "f32 [0 3e38; 1e-45 0]",
    );

    let complex_block = column_major(&rows)
        .iter()
        .map(|&entry| Complex32::new(entry, 0.0))
        .collect::<Vec<_>>();
    let tensor = u1_block_endomorphism(&[(0, 2, complex_block)]);
    let exponential = exp(
        &mut dense,
        &mut TensorContractFusionExecutionContext::<Complex32, RuleIdentity>::default(),
        &bound_tensor_ref!(Arc::new(U1FusionRule), &tensor),
    )
    .unwrap();
    assert_sector_matrix_matches(
        &u1_block_matrix(exponential.tensor(), 0),
        &expected,
        1e-6,
        "c32 [0 3e38; 1e-45 0]",
    );
}

#[test]
fn exp_reports_unsupported_when_the_executor_cannot_solve() {
    // What: device storage whose backend supplies GEMM but no solve is refused
    // in the backend's own words, not silently routed onto the host.
    let tensor = exp_oracle_tensor::<f64>(1.0);
    let mut dense = SolvelessExecutor::default();
    let mut context = default_context();

    let error = exp(
        &mut dense,
        &mut context,
        &bound_tensor_ref!(Arc::new(U1FusionRule), &tensor),
    )
    .unwrap_err();

    assert!(
        matches!(
            error,
            OperationError::Dense(DenseError::Unsupported {
                op: "solve_into",
                ..
            })
        ),
        "unexpected error {error:?}"
    );
}

#[test]
fn exp_rejects_a_non_endomorphism() {
    // What: TensorKit's own precondition (`domain == codomain`) still holds —
    // #577 widens which endomorphisms are accepted, not which maps are.
    let rule = U1FusionRule;
    let codomain = SectorLeg::new([(U1Irrep::new(0).sector_id(), 3)], false);
    let domain = SectorLeg::new([(U1Irrep::new(0).sector_id(), 2)], false);
    let homspace = FusionTreeHomSpace::new(
        FusionProductSpace::new([codomain]),
        FusionProductSpace::new([domain]),
    );
    let shapes = vec![vec![3, 2]; homspace.fusion_tree_keys(&rule).len()];
    let space = FusionTensorMapSpace::from_degeneracy_shapes_coupled(
        TensorMapSpace::<1, 1>::from_dims([3], [2]).unwrap(),
        homspace,
        &rule,
        shapes,
    )
    .unwrap();
    let tensor = TensorMap::<f64, 1, 1>::from_vec_with_fusion_space(
        (0..space.required_len().unwrap())
            .map(|index| index as f64)
            .collect(),
        space,
    )
    .unwrap();
    let mut dense = tenet_dense::DefaultDenseExecutor::new();
    let mut context = default_context();

    let error = exp(
        &mut dense,
        &mut context,
        &bound_tensor_ref!(Arc::new(rule), &tensor),
    )
    .unwrap_err();

    assert!(
        matches!(
            error,
            OperationError::UnsupportedTensorContractScope {
                message: "exp requires an endomorphism (codomain == domain)"
            }
        ),
        "an exp caller must be refused in exp's words, not eigh's: {error:?}"
    );
}

#[test]
fn exp_rejects_a_nonfinite_general_block() {
    // What: a NaN block is not Hermitian to MatrixAlgebraKit's predicate, so it
    // arrives at the general arm; it must be named there rather than handed to
    // the backend as a silent NaN.
    let tensor = u1_block_endomorphism(&[(0, 2, vec![1.0_f64, 0.5, f64::NAN, 2.0])]);
    let mut dense = tenet_dense::DefaultDenseExecutor::new();
    let mut context = default_context();

    let error = exp(
        &mut dense,
        &mut context,
        &bound_tensor_ref!(Arc::new(U1FusionRule), &tensor),
    )
    .unwrap_err();

    assert!(
        matches!(
            error,
            OperationError::InvalidArgument {
                message: "exp requires finite coupled-sector blocks"
            }
        ),
        "unexpected error {error:?}"
    );
}

#[test]
fn exp_rejects_a_block_whose_column_norm_overflows() {
    // What: every entry is finite but the column 1-norm is not. The squaring
    // count is `ceil(log2(inf / theta_13))` cast to `u32`, which saturates to
    // `u32::MAX`; read back as `i32` that is -1, so the block would be scaled
    // *up* and then squared ~4.3e9 times — a finite input that never returns.
    // The overflowing norm has to be refused where it is computed.
    //
    // The norm in question is the *balanced* one, since balancing runs first,
    // so the fixture is one balancing cannot rescue: every entry is within a
    // factor of five of `f64::MAX`, which puts LAPACK's overflow guards (`ca`
    // and `ra` against `sfmax2`) in the way of any radix step, and the column
    // sum stays infinite. A block whose imbalance balancing *can* undo —
    // `[1e308 1; 1e308 2]`, this fixture before #577's balancing existed — is
    // no longer refused and is not meant to be: Julia reaches the same finite
    // balanced norm on it and exponentiates it.
    let tensor = u1_block_endomorphism(&[(0, 2, vec![1e308_f64, 1e308, 2e307, 1e308])]);
    let mut dense = tenet_dense::DefaultDenseExecutor::new();
    let mut context = default_context();

    let error = exp(
        &mut dense,
        &mut context,
        &bound_tensor_ref!(Arc::new(U1FusionRule), &tensor),
    )
    .unwrap_err();

    assert!(
        matches!(
            error,
            OperationError::InvalidArgument {
                message: "exp requires coupled-sector blocks with a finite 1-norm"
            }
        ),
        "unexpected error {error:?}"
    );
}

#[test]
fn exp_publishes_nothing_when_a_later_sector_fails() {
    // What: failure atomicity. A backend failure on the second sector leaves no
    // tensor and does not touch the input's storage.
    let tensor = exp_oracle_tensor::<f64>(1.0);
    let before = tensor.data().to_vec();
    let mut spy = MatrixFunctionCallSpy {
        fail_solve_number: Some(2),
        ..MatrixFunctionCallSpy::default()
    };
    let mut context = default_context();

    let error = exp(
        &mut spy,
        &mut context,
        &bound_tensor_ref!(Arc::new(U1FusionRule), &tensor),
    )
    .unwrap_err();

    assert!(
        matches!(
            error,
            OperationError::Dense(DenseError::Backend {
                op: "solve_into",
                ..
            })
        ),
        "unexpected error {error:?}"
    );
    assert_eq!(
        spy.solve_calls, 2,
        "the failing sector must have been tried"
    );
    assert_eq!(tensor.data(), &before[..], "input storage was mutated");
}

#[test]
fn noncanonical_mf_svd_and_eigh_scatter_each_output_block_once() {
    // What: on a four-sector noncanonical MF input, the compact SVD and EIGH
    // fallbacks group each factor side once and iterate only the scattered
    // blocks (F = B = 16 per side), instead of G_s * B = 64 visits per side.
    use crate::factorize::{reset_scatter_visit_probe, scatter_visit_probe, ScatterVisitProbe};
    let sectors = (0..4).map(SectorId::new).collect::<Vec<_>>();
    let rule = || tenet_core::ZNFusionRule::new(4).unwrap();
    let mut dense = tenet_dense::DefaultDenseExecutor::new();

    let tensor = tsvd_test_tensor(&rule(), &sectors);
    let bound = bound_tensor(Arc::new(rule()), &tensor);
    let adjoint_space = bound.space().adjoint_view().unwrap();
    let input = BoundDynamicTensorRef::try_new(&adjoint_space, bound.data()).unwrap();
    reset_scatter_visit_probe();
    let (u, vh, _) = svd_compact_factors_dyn(&mut dense, &input).unwrap();
    let b_u = u.space().space().structure().block_count();
    let b_vh = vh.space().space().structure().block_count();
    assert_eq!((b_u, b_vh), (16, 16));
    let probe = scatter_visit_probe();
    assert_eq!(
        probe,
        ScatterVisitProbe {
            left_grouped: b_u,
            right_grouped: b_vh,
            left_groups_built: 1,
            right_groups_built: 1,
            left_visits: b_u,
            right_visits: b_vh,
        }
    );
    assert!(probe.left_grouped + probe.left_visits < 4 * b_u);
    assert!(probe.right_grouped + probe.right_visits < 4 * b_vh);

    let tensor = hermitian_test_tensor(&rule(), &sectors);
    let bound = bound_tensor(Arc::new(rule()), &tensor);
    let adjoint_space = bound.space().adjoint_view().unwrap();
    let input = BoundDynamicTensorRef::try_new(&adjoint_space, bound.data()).unwrap();
    reset_scatter_visit_probe();
    let full = eigh_full_dyn(&mut dense, &input).unwrap();
    let b_v = full.v().space().space().structure().block_count();
    assert_eq!(b_v, 16);
    let probe = scatter_visit_probe();
    assert_eq!(
        probe,
        ScatterVisitProbe {
            left_grouped: b_v,
            left_groups_built: 1,
            left_visits: b_v,
            ..ScatterVisitProbe::default()
        }
    );
    assert!(probe.left_grouped + probe.left_visits < 4 * b_v);
}
