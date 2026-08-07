use tenet_core::{
    product_fusion_rule, BlockKey, BlockSpec, BlockStructure, BraidingStyleKind,
    CheckedGenericFusion, CoreError, FermionParityFusionRule, FusionProductSpace, FusionRule,
    FusionStyleKind, FusionTensorMapSpace, FusionTreeHomSpace, FusionTreeKey, GenericFArray,
    GenericFusionSymbols, GenericRMatrix, GenericRigidSymbols, MultiplicityFreeFusionRule,
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
    typed_from_dyn, validate_inverse_region_routes_for_test, BoundTensorMap,
};
use crate::*;
use num_complex::{Complex32, Complex64};
use num_traits::Zero;
use std::{cell::Cell, fmt, sync::Arc};
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
}

#[derive(Default)]
struct FailAfterObservingQrInput {
    observed: Vec<Vec<f64>>,
}

#[derive(Default)]
struct FailAfterObservingEighInput {
    observed: Vec<Vec<f64>>,
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

struct EqualMagnitudeEigh;

struct NanEigh;

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

impl DenseExecutor for FailAfterObservingQrInput {
    fn svd(&mut self, _: DenseRead<'_>) -> Result<Vec<DenseTensor>, DenseError> {
        panic!("test only exercises QR")
    }

    fn qr(&mut self, _: DenseRead<'_>) -> Result<Vec<DenseTensor>, DenseError> {
        panic!("compact QR must use the destination API")
    }

    fn qr_into(
        &mut self,
        input: DenseRead<'_>,
        q: DenseWrite<'_>,
        r: DenseWrite<'_>,
    ) -> Result<(), DenseError> {
        let DenseRead::F64(input) = input else {
            panic!("test input must be f64")
        };
        self.observed.push(input.data().to_vec());
        let DenseWrite::F64(q) = q else {
            panic!("test Q must be f64")
        };
        let DenseWrite::F64(r) = r else {
            panic!("test R must be f64")
        };
        assert!(q.data().iter().all(|&value| value == 0.0));
        assert!(r.data().iter().all(|&value| value == 0.0));
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

    fn eigh(&mut self, _: DenseRead<'_>) -> Result<Vec<DenseTensor>, DenseError> {
        panic!("canonical EIGH must use the destination API")
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
        panic!("canonical EIGH must use the destination API")
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

impl DenseExecutor for EqualMagnitudeEigh {
    fn svd(&mut self, _: DenseRead<'_>) -> Result<Vec<DenseTensor>, DenseError> {
        panic!("test only exercises EIGH")
    }

    fn qr(&mut self, _: DenseRead<'_>) -> Result<Vec<DenseTensor>, DenseError> {
        panic!("test only exercises EIGH")
    }

    fn eigh(&mut self, _: DenseRead<'_>) -> Result<Vec<DenseTensor>, DenseError> {
        panic!("canonical EIGH must use the destination API")
    }

    fn eigh_into(
        &mut self,
        _: DenseRead<'_>,
        values: DenseWrite<'_>,
        vectors: DenseWrite<'_>,
    ) -> Result<(), DenseError> {
        let DenseWrite::F64(mut values) = values else {
            panic!("test eigenvalues must be f64")
        };
        let DenseWrite::F64(mut vectors) = vectors else {
            panic!("test eigenvectors must be f64")
        };
        assert_eq!(values.data().len(), 3);
        values.data_mut().copy_from_slice(&[1.0, -2.0, 2.0]);
        vectors.data_mut().copy_from_slice(&[
            1.0, 0.0, 0.0, // first backend column
            0.0, 1.0, 0.0, // second backend column
            0.0, 0.0, 1.0, // third backend column
        ]);
        Ok(())
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

impl DenseExecutor for NanEigh {
    fn svd(&mut self, _: DenseRead<'_>) -> Result<Vec<DenseTensor>, DenseError> {
        panic!("test only exercises EIGH")
    }

    fn qr(&mut self, _: DenseRead<'_>) -> Result<Vec<DenseTensor>, DenseError> {
        panic!("test only exercises EIGH")
    }

    fn eigh(&mut self, _: DenseRead<'_>) -> Result<Vec<DenseTensor>, DenseError> {
        panic!("canonical EIGH must use the destination API")
    }

    fn eigh_into(
        &mut self,
        _: DenseRead<'_>,
        values: DenseWrite<'_>,
        vectors: DenseWrite<'_>,
    ) -> Result<(), DenseError> {
        let DenseWrite::F64(mut values) = values else {
            panic!("test eigenvalues must be f64")
        };
        let DenseWrite::F64(mut vectors) = vectors else {
            panic!("test eigenvectors must be f64")
        };
        values.data_mut().fill(f64::NAN);
        vectors.data_mut().fill(0.0);
        Ok(())
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

#[test]
fn compact_svd_canonical_layout_skips_input_pack_and_factor_scatter() {
    // What: canonical coupled storage reaches final factor destinations without numerical copies.
    let rule = Z2FusionRule;
    let tensor = tsvd_test_tensor(&rule, &[SectorId::new(0), SectorId::new(1)]);
    let mut dense = tenet_dense::DefaultDenseExecutor::new();

    crate::factorize::reset_compact_svd_copy_probe();
    svd_compact(&mut dense, &bound_tensor_ref!(Arc::new(rule), &tensor)).unwrap();

    assert_eq!(
        crate::factorize::compact_svd_copy_probe(),
        crate::factorize::CompactSvdCopyProbe::default()
    );
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

    assert_eq!(
        crate::factorize::compact_qr_copy_probe(),
        crate::factorize::CompactQrCopyProbe::default()
    );
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
    assert_eq!(probe.scratch_buffer_count, 3);
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

fn padded_generic_factorization_input(
    source: &BoundDynamicFusionMapSpace<FactorGenericRule>,
    source_data: &[f64],
) -> (BoundDynamicFusionMapSpace<FactorGenericRule>, Vec<f64>) {
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
    qr_calls: usize,
}

impl Default for CountingDense {
    fn default() -> Self {
        Self {
            inner: tenet_dense::DefaultDenseExecutor::new(),
            svd_calls: 0,
            qr_calls: 0,
        }
    }
}

impl DenseExecutor for CountingDense {
    fn svd(&mut self, input: DenseRead<'_>) -> Result<Vec<DenseTensor>, DenseError> {
        self.svd_calls += 1;
        self.inner.svd(input)
    }

    fn qr(&mut self, input: DenseRead<'_>) -> Result<Vec<DenseTensor>, DenseError> {
        self.qr_calls += 1;
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

#[test]
fn checked_generic_full_svd_u_builder_failure_preserves_provider_context() {
    // What: after the two multiplicity-aware dimension DPs (ten checked calls),
    // the first post-dense checked-provider call belongs to U-space construction.
    assert_checked_full_svd_builder_failure(15);
}

#[test]
fn checked_generic_full_svd_vh_builder_failure_preserves_provider_context() {
    // What: Vh-space construction propagates its exact provider error without publishing U.
    assert_checked_full_svd_builder_failure(19);
}

#[test]
fn checked_generic_full_svd_s_builder_failure_preserves_provider_context() {
    // What: S-space construction propagates its exact provider error after U/Vh staging;
    // its final checked admission call follows the ten-call dimension preflight at 22.
    assert_checked_full_svd_builder_failure(22);
}

#[test]
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
    let rule = Z2FusionRule;
    let tensor = hermitian_test_tensor(&rule, &[SectorId::new(0), SectorId::new(1)]);
    let before = tensor.data().to_vec();
    let mut dense = FailAfterObservingEighInput::default();

    let result = eigh_full(&mut dense, &bound_tensor_ref!(Arc::new(rule), &tensor));

    assert!(matches!(result, Err(OperationError::Dense(_))));
    assert_eq!(tensor.data(), before);
    assert!(!dense.observed.is_empty());
    assert!(dense
        .observed
        .iter()
        .all(|sector| before.windows(sector.len()).any(|window| window == sector)));
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

#[test]
fn eigh_stably_orders_equal_magnitudes_and_reorders_vectors_in_place() {
    // What: equal magnitudes retain backend order while larger-magnitude columns move together.
    let tensor =
        one_sector_rectangular_matrix(vec![1.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 3.0], 3, 3);
    let mut dense = EqualMagnitudeEigh;

    let eigh = eigh_full(
        &mut dense,
        &bound_tensor_ref!(Arc::new(Z2FusionRule), &tensor),
    )
    .unwrap();

    assert_eq!(eigh.eigenvalues[0].values, vec![-2.0, 2.0, 1.0]);
    assert_eq!(
        eigh.v.data(),
        &[0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0]
    );
}

#[test]
fn eigh_rejects_non_finite_backend_eigenvalues_before_sorting() {
    // What: backend non-finite spectra become a typed operation error, not a comparator panic.
    let tensor =
        one_sector_rectangular_matrix(vec![1.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 3.0], 3, 3);
    let mut dense = NanEigh;

    let error = eigh_full(
        &mut dense,
        &bound_tensor_ref!(Arc::new(Z2FusionRule), &tensor),
    )
    .unwrap_err();

    assert_eq!(
        error,
        OperationError::InvalidArgument {
            message: "eigenvalues must be finite",
        }
    );
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
    assert!(Arc::strong_count(&plan) >= 1);
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
    assert_eq!(
        crate::factorize::compact_svd_copy_probe(),
        crate::factorize::CompactSvdCopyProbe::default()
    );
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
    assert_eq!(
        crate::factorize::compact_qr_copy_probe(),
        crate::factorize::CompactQrCopyProbe::default()
    );
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
    assert_eq!(probe.scratch_buffer_count, 3);
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

    assert_eq!(
        crate::factorize::compact_svd_copy_probe(),
        crate::factorize::CompactSvdCopyProbe::default()
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

    assert_eq!(
        crate::factorize::compact_qr_copy_probe(),
        crate::factorize::CompactQrCopyProbe::default()
    );
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
    assert_eq!(probe.scratch_buffer_count, 3);
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

    assert_eq!(
        crate::factorize::compact_svd_copy_probe(),
        crate::factorize::CompactSvdCopyProbe::default()
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
    assert_eq!(
        crate::factorize::compact_svd_copy_probe(),
        crate::factorize::CompactSvdCopyProbe::default()
    );
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

    assert_eq!(
        crate::factorize::compact_svd_copy_probe(),
        crate::factorize::CompactSvdCopyProbe::default()
    );
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

fn u1_minimum_matrix(rows: usize, cols: usize) -> TensorMap<f64, 1, 1> {
    let rule = U1FusionRule;
    let minimum = U1Irrep::new(i32::MIN).sector_id();
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
fn compact_factors_do_not_relabel_u1_minimum_domain_sectors() {
    // What: compact SVD, QR, and LQ return correctly oriented factors at the finite U(1) dual boundary.
    let rule = U1FusionRule;
    let tensor = u1_minimum_matrix(3, 2);
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
fn eigh_full_does_not_relabel_u1_minimum_domain_sectors() {
    // What: full EIGH preserves the eigen equation and factor orientation at the finite U(1) dual boundary.
    let rule = U1FusionRule;
    let minimum = U1Irrep::new(i32::MIN).sector_id();
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
    for col in 0..2 {
        for row in 0..2 {
            let lhs = (0..2)
                .map(|index| tensor.data()[row + 2 * index] * result.v.data()[index + 2 * col])
                .sum::<f64>();
            let rhs = result.v.data()[row + 2 * col] * values[col];
            assert!((lhs - rhs).abs() < 1e-10);
        }
    }
}

#[cfg(target_pointer_width = "64")]
#[test]
fn compact_factorizations_do_not_relabel_product_minimum_domain_sectors() {
    // What: product-sector factors inherit the no-relabel contract for compact SVD, QR, LQ, and full EIGH.
    let rule = product_fusion_rule(FermionParityFusionRule, U1FusionRule);
    let minimum = rule
        .try_encode_sector(SectorId::new(1), U1Irrep::new(i32::MIN).sector_id())
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
    for col in 0..2 {
        for row in 0..2 {
            let lhs = (0..2)
                .map(|index| hermitian.data()[row + 2 * index] * result.v.data()[index + 2 * col])
                .sum::<f64>();
            let rhs = result.v.data()[row + 2 * col] * values[col];
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
    assert_eq!(
        crate::factorize::compact_qr_copy_probe(),
        crate::factorize::CompactQrCopyProbe::default()
    );
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
    assert_eq!(probe.scratch_buffer_count, 3);
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
    assert_eq!(
        crate::factorize::compact_svd_copy_probe(),
        crate::factorize::CompactSvdCopyProbe::default()
    );
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
    assert_eq!(r_gauged[1 + 3 * 1], c(0.0, 0.0));
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
