use tenet_core::{
    product_fusion_rule, BlockKey, BraidingStyleKind, CoreError, FermionParityFusionRule,
    FusionProductSpace, FusionRule, FusionStyleKind, FusionTensorMapSpace, FusionTreeHomSpace,
    FusionTreeKey, MultiplicityFreeFusionRule, MultiplicityFreeFusionSymbols,
    MultiplicityFreeRigidSymbols, RuleIdentity, SU2FusionRule, SU2Irrep, SectorId, SectorLeg,
    SectorVec, TensorMap, TensorMapSpace, U1FusionRule, U1Irrep, Z2FusionRule,
};
use tenet_tensors::{
    BoundDynamicFusionMapSpace, OperationError, OutputAxisOrder,
    TensorContractFusionExecutionContext, TensorContractSpec, TreeTransformBuiltinRuleCacheKey,
    TreeTransformRuleCacheKey,
};

use crate::factorize::{dyn_space_of, truncate_svd, typed_from_dyn, BoundTensorMap};
use crate::*;
use num_complex::{Complex32, Complex64};
use num_traits::Zero;
use std::sync::{Arc, Mutex};
use tenet_dense::{
    DenseBackend, DenseDotConfig, DenseError, DenseExecutor, DenseRead, DenseTensor, DenseWrite,
};

static COMPACT_FACTOR_PLAN_IDENTITY_TEST_LOCK: Mutex<()> = Mutex::new(());

struct RejectExecutorCalls;

#[derive(Default)]
struct SvdCallSpy {
    inner: tenet_dense::DefaultDenseExecutor,
    svd_calls: usize,
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

struct NonFiniteSvdSpectrum {
    singular_value: f64,
}

struct EqualMagnitudeEigh;

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
    fn scalar_one(&self) -> f64 {
        1.0
    }
    fn scalar_conj(&self, value: f64) -> f64 {
        value
    }
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

impl DenseExecutor for NonFiniteSvdSpectrum {
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
        assert_eq!(input.shape(), &[1, 1]);
        let DenseWrite::F64(mut u) = u else {
            panic!("test U must be f64")
        };
        let DenseWrite::F64(mut s) = s else {
            panic!("test singular values must be f64")
        };
        let DenseWrite::F64(mut vt) = vt else {
            panic!("test Vh must be f64")
        };
        assert_eq!(u.shape(), &[1, 1]);
        assert_eq!(s.shape(), &[1]);
        assert_eq!(vt.shape(), &[1, 1]);
        u.data_mut()[0] = 1.0;
        s.data_mut()[0] = self.singular_value;
        vt.data_mut()[0] = 1.0;
        Ok(())
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
    // What: values-only EIGH validates all packed sectors before its first no-vector driver.
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
}

#[test]
fn eigh_uses_matrixalgebrakit_tolerance_for_f32_c32_and_f64() {
    // What: expert EIGH uses each real dtype's eps(maxabs)^(3/4) boundary.
    let within_f32 = one_sector_matrix(vec![1.0_f32, 1.0e-7, 0.0, 2.0]);
    let outside_f32 = one_sector_matrix(vec![1.0_f32, 1.0e-3, 0.0, 2.0]);
    let within_c32 = one_sector_matrix(vec![
        Complex32::new(1.0, 0.0),
        Complex32::new(1.0e-7, 0.0),
        Complex32::new(0.0, 0.0),
        Complex32::new(2.0, 0.0),
    ]);
    let outside_c32 = one_sector_matrix(vec![
        Complex32::new(1.0, 0.0),
        Complex32::new(1.0e-3, 0.0),
        Complex32::new(0.0, 0.0),
        Complex32::new(2.0, 0.0),
    ]);
    let within_f64 = one_sector_matrix(vec![1.0_f64, 1.0e-13, 0.0, 2.0]);
    let outside_f64 = one_sector_matrix(vec![1.0_f64, 1.0e-8, 0.0, 2.0]);

    assert_eigh_preflight(&within_f32, true);
    assert_eigh_preflight(&outside_f32, false);
    assert_eigh_preflight(&within_c32, true);
    assert_eigh_preflight(&outside_c32, false);
    assert_eigh_preflight(&within_f64, true);
    assert_eigh_preflight(&outside_f64, false);
}

#[test]
fn eigh_accepts_exact_hermitian_max_magnitude_inputs() {
    // What: tolerance squaring cannot reject exact Hermitian f32 or f64 matrices at finite maxima.
    let max_f32 = one_sector_matrix(vec![f32::MAX, 0.0, 0.0, f32::MAX]);
    let max_f64 = one_sector_matrix(vec![f64::MAX, 0.0, 0.0, f64::MAX]);

    assert_eigh_preflight(&max_f32, true);
    assert_eigh_preflight(&max_f64, true);
}

#[test]
fn eigh_rejects_a_large_nonhermitian_input() {
    // What: overflow-safe tolerance comparison still rejects large finite asymmetry.
    let tensor = one_sector_matrix(vec![f64::MAX, f64::MAX, 0.0, f64::MAX]);

    assert_eigh_preflight(&tensor, false);
}

#[test]
fn eigh_matches_the_blocked_matrixalgebrakit_hermitian_oracle() {
    // What: cross-block residuals use projection halves, pair multiplicity, and a global sum.
    const N: usize = 33;
    const MAK_TOL: f64 = 3.059_163_337_652_406e-12;
    let accepted_delta = 1.2 * MAK_TOL;
    assert!(accepted_delta / 2.0_f64.sqrt() < MAK_TOL);

    let mut accepted = vec![0.0; N * N];
    accepted[0] = 2.0;
    accepted[N * 32] = accepted_delta;
    assert_eigh_preflight(&one_sector_rectangular_matrix(accepted, N, N), true);

    let rejected_delta = 1.1 * MAK_TOL;
    assert!(rejected_delta / 2.0_f64.sqrt() < MAK_TOL);
    assert!((2.0 * rejected_delta.powi(2) / 2.0).sqrt() > MAK_TOL);

    let mut rejected = vec![0.0; N * N];
    rejected[0] = 2.0;
    rejected[N * 32] = rejected_delta;
    rejected[1 + N * 32] = rejected_delta;
    assert_eigh_preflight(&one_sector_rectangular_matrix(rejected, N, N), false);
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
fn eigh_vectors_retain_each_callers_exact_provider_arc() {
    // What: a shared geometry plan rebinds EIGH vectors to each caller's provider allocation.
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
    // What: a shared geometry plan rebinds both QR factors to each caller's provider allocation.
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
    // What: a shared geometry plan rebinds both LQ factors to each caller's provider allocation.
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
fn compact_svd_qr_eigh_and_lq_preserve_one_factor_plan_generation() {
    // What: SVD, QR, EIGH, and LQ keep one factor-plan generation alive across all operations.
    let _guard = COMPACT_FACTOR_PLAN_IDENTITY_TEST_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let tensor = hermitian_test_tensor(&Z2FusionRule, &[SectorId::new(0), SectorId::new(1)]);
    let bound = bound_tensor(Arc::new(Z2FusionRule), &tensor);
    let mut dense = tenet_dense::DefaultDenseExecutor::new();

    tenet_tensors::reset_global_operation_caches();
    let before = crate::factorize::compact_factor_plan_for_test(bound.space())
        .unwrap()
        .unwrap();
    svd_compact(&mut dense, &bound.as_ref()).unwrap();
    qr_compact(&mut dense, &bound.as_ref()).unwrap();
    eigh_full(&mut dense, &bound.as_ref()).unwrap();
    lq_compact(&mut dense, &bound.as_ref()).unwrap();
    let after = crate::factorize::compact_factor_plan_for_test(bound.space())
        .unwrap()
        .unwrap();

    assert!(Arc::ptr_eq(&before, &after));
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
fn compact_factor_plan_reuses_clone_before_init_and_concurrent_first_use() {
    // What: one semantic compact-factor plan serves clones made before and during first use.
    let _guard = COMPACT_FACTOR_PLAN_IDENTITY_TEST_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let tensor = rectangular_svd_tensor(23, 17);
    let bound = bound_tensor(Arc::new(Z2FusionRule), &tensor);
    let before_init = bound.space().clone();
    let spaces = (0..8).map(|_| before_init.clone()).collect::<Vec<_>>();
    let plans = std::thread::scope(|scope| {
        spaces
            .iter()
            .map(|space| {
                scope.spawn(|| {
                    crate::factorize::compact_factor_plan_for_test(space)
                        .unwrap()
                        .unwrap()
                })
            })
            .collect::<Vec<_>>()
            .into_iter()
            .map(|handle| handle.join().unwrap())
            .collect::<Vec<_>>()
    });
    let after_init = bound.space().clone();
    let after = crate::factorize::compact_factor_plan_for_test(&after_init)
        .unwrap()
        .unwrap();

    assert!(plans.iter().all(|plan| Arc::ptr_eq(plan, &plans[0])));
    assert!(Arc::ptr_eq(&after, &plans[0]));
}

#[test]
fn compact_svd_shared_plan_rebinds_every_factor_to_each_caller() {
    // What: one semantic plan serves distinct provider Arcs while U/S/Vh inherit each caller.
    let _guard = COMPACT_FACTOR_PLAN_IDENTITY_TEST_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let tensor = rectangular_svd_tensor(7, 5);
    let first_provider = Arc::new(Z2FusionRule);
    let second_provider = Arc::new(Z2FusionRule);
    let first = bound_tensor(Arc::clone(&first_provider), &tensor);
    let second = bound_tensor(Arc::clone(&second_provider), &tensor);
    let first_plan = crate::factorize::compact_factor_plan_for_test(first.space())
        .unwrap()
        .unwrap();
    let second_plan = crate::factorize::compact_factor_plan_for_test(second.space())
        .unwrap()
        .unwrap();
    assert!(Arc::ptr_eq(&first_plan, &second_plan));

    let first_input = BoundDynamicTensorRef::try_new(first.space(), tensor.data()).unwrap();
    let second_input = BoundDynamicTensorRef::try_new(second.space(), tensor.data()).unwrap();
    let mut dense = tenet_dense::DefaultDenseExecutor::new();
    let first_svd = svd_compact_dyn(&mut dense, &first_input).unwrap();
    let second_svd = svd_compact_dyn(&mut dense, &second_input).unwrap();

    for factor in [first_svd.u(), first_svd.s(), first_svd.vh()] {
        assert!(Arc::ptr_eq(factor.space().provider_arc(), &first_provider));
    }
    for factor in [second_svd.u(), second_svd.s(), second_svd.vh()] {
        assert!(Arc::ptr_eq(factor.space().provider_arc(), &second_provider));
    }
}

#[test]
fn global_operation_reset_replaces_compact_factor_plan_generation() {
    // What: a completed global reset invalidates both the shared plan and this thread's front.
    let _guard = COMPACT_FACTOR_PLAN_IDENTITY_TEST_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let tensor = rectangular_svd_tensor(29, 23);
    let bound = bound_tensor(Arc::new(Z2FusionRule), &tensor);
    let before = crate::factorize::compact_factor_plan_for_test(bound.space())
        .unwrap()
        .unwrap();

    tenet_tensors::reset_global_operation_caches();

    let after = crate::factorize::compact_factor_plan_for_test(bound.space())
        .unwrap()
        .unwrap();
    assert!(!Arc::ptr_eq(&before, &after));
}

#[test]
fn compact_factor_cached_plan_does_not_retain_first_provider() {
    // What: a cached semantic plan never owns the provider that first built it.
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
    crate::factorize::reset_compact_svd_copy_probe();
    let svd = svd_compact(&mut dense, &bound_tensor_ref!(Arc::new(rule), &tensor)).unwrap();
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
    R: MultiplicityFreeRigidSymbols<Scalar = f64>
        + TreeTransformRuleCacheKey<Key = TreeTransformBuiltinRuleCacheKey>,
{
    let mut scaled_vt = svd.vh.tensor().clone();
    scale_vt_rows_by_singular_values(&mut scaled_vt, &svd.singular_values);
    let mut reconstructed = TensorMap::<f64, 2, 2>::from_vec_with_fusion_space(
        vec![0.0; template.data().len()],
        template.fusion_space().unwrap().as_ref().clone(),
    )
    .unwrap();
    let mut context =
        TensorContractFusionExecutionContext::<f64, TreeTransformBuiltinRuleCacheKey>::default();
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

    let svd = svd_trunc(
        &mut dense_executor,
        &bound_tensor_ref!(Arc::new(rule), &tensor),
        &Truncation::absolute_cutoff(threshold),
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
    let svd = svd_trunc(
        &mut dense_executor,
        &bound_tensor_ref!(Arc::new(rule), &tensor),
        &Truncation::relative_error(tolerance),
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

    let mut context =
        TensorContractFusionExecutionContext::<f64, TreeTransformBuiltinRuleCacheKey>::default();
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
    let truncation = Truncation::rank(9).and(Truncation::absolute_cutoff(1e-12));

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
    let mut context =
        TensorContractFusionExecutionContext::<f64, TreeTransformBuiltinRuleCacheKey>::default();
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
    let mut context = TensorContractFusionExecutionContext::<
        Complex64,
        TreeTransformBuiltinRuleCacheKey,
    >::default();
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

    let mut context = TensorContractFusionExecutionContext::<
        Complex64,
        TreeTransformBuiltinRuleCacheKey,
    >::default();
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

fn default_context() -> TensorContractFusionExecutionContext<f64, TreeTransformBuiltinRuleCacheKey>
{
    TensorContractFusionExecutionContext::<f64, TreeTransformBuiltinRuleCacheKey>::default()
}

fn lowered_z2_binding<const NOUT: usize, const NIN: usize>(
    tensor: &TensorMap<f64, NOUT, NIN>,
) -> BoundDynamicFusionMapSpace<Z2FusionRule> {
    let provider = Arc::new(Z2FusionRule);
    let raw = dyn_space_of(tensor).unwrap();
    let hom = raw.homspace().clone();
    hom.try_fusion_tree_keys_lowered(provider.as_ref()).unwrap();
    let shapes = hom
        .fusion_tree_keys(provider.as_ref())
        .iter()
        .map(|key| {
            let index = raw
                .structure()
                .find_block_index_by_key(&BlockKey::FusionTree(key.clone()))
                .unwrap();
            raw.structure().block(index).unwrap().shape().to_vec()
        })
        .collect::<Vec<_>>();
    BoundDynamicFusionMapSpace::from_degeneracy_shapes_lowered(provider, hom, shapes).unwrap()
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
    let mut dense_executor = tenet_dense::DefaultDenseExecutor::new();
    let mut context = default_context();
    let inverse = inv(
        &mut dense_executor,
        &mut context,
        &bound_tensor_ref!(Arc::new(rule), &tensor),
    )
    .unwrap();
    let identity = crate::compose::compose(&mut context, &rule, &tensor, &inverse).unwrap();
    assert_identity_matrices(&dense_sector_matrices(2, &identity));
}

fn u1_cross_space_map(codomain: &[(i32, usize)], domain: &[(i32, usize)]) -> TensorMap<f64, 1, 1> {
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
    TensorMap::from_block_fn_with_fusion_space(space, 0.0, |_, indices| {
        if indices[0] == indices[1] {
            1.0
        } else {
            0.0
        }
    })
    .unwrap()
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
        let tensor = u1_cross_space_map(codomain, domain);
        let mut dense = RejectExecutorCalls;
        let mut context =
            TensorContractFusionExecutionContext::<f64, TreeTransformBuiltinRuleCacheKey>::default(
            );
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
    let sectors = blocks
        .iter()
        .map(|(charge, dimension, _)| (U1Irrep::new(*charge).sector_id(), *dimension))
        .collect::<Vec<_>>();
    let leg = SectorLeg::new(sectors.iter().copied(), false);
    let total_dimension = sectors.iter().map(|(_, dimension)| dimension).sum();
    let homspace = FusionTreeHomSpace::new(
        FusionProductSpace::new([leg.clone()]),
        FusionProductSpace::new([leg]),
    );
    let shapes = homspace
        .fusion_tree_keys(&U1FusionRule)
        .iter()
        .map(|key| {
            let sector = key.codomain_tree().coupled();
            let (_, dimension, data) = blocks
                .iter()
                .find(|(charge, _, _)| U1Irrep::new(*charge).sector_id() == sector)
                .unwrap();
            assert_eq!(data.len(), dimension * dimension);
            vec![*dimension, *dimension]
        })
        .collect::<Vec<_>>();
    let space = FusionTensorMapSpace::from_degeneracy_shapes_coupled(
        TensorMapSpace::<1, 1>::from_dims([total_dimension], [total_dimension]).unwrap(),
        homspace,
        &U1FusionRule,
        shapes,
    )
    .unwrap();
    TensorMap::from_block_fn_with_fusion_space(space, D::zero(), |key, indices| {
        let BlockKey::FusionTree(tree) = key else {
            return D::zero();
        };
        let sector = tree.codomain_tree().coupled();
        let (_, dimension, data) = blocks
            .iter()
            .find(|(charge, _, _)| U1Irrep::new(*charge).sector_id() == sector)
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
        let mut context =
            TensorContractFusionExecutionContext::<f64, TreeTransformBuiltinRuleCacheKey>::default(
            );

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
    let mut context = TensorContractFusionExecutionContext::<
        Complex64,
        TreeTransformBuiltinRuleCacheKey,
    >::default();

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
fn inv_numerical_rank_uses_the_factor_dtype_epsilon() {
    // What: the same condition ratio is full-rank in double precision and rank-deficient in single precision.
    let matrix_f64 = vec![1.0, 0.0, 0.0, 1e-8];
    let tensor_f64 = u1_block_endomorphism(&[(0, 2, matrix_f64)]);
    let mut dense = tenet_dense::DefaultDenseExecutor::new();
    let mut f64_context =
        TensorContractFusionExecutionContext::<f64, TreeTransformBuiltinRuleCacheKey>::default();
    inv(
        &mut dense,
        &mut f64_context,
        &bound_tensor_ref!(Arc::new(U1FusionRule), &tensor_f64),
    )
    .unwrap();

    let matrix_f32 = vec![1.0_f32, 0.0, 0.0, 1e-8];
    let tensor_f32 = u1_block_endomorphism(&[(0, 2, matrix_f32)]);
    let mut f32_context =
        TensorContractFusionExecutionContext::<f32, TreeTransformBuiltinRuleCacheKey>::default();
    let f32_error = inv(
        &mut dense,
        &mut f32_context,
        &bound_tensor_ref!(Arc::new(U1FusionRule), &tensor_f32),
    )
    .unwrap_err();
    assert!(matches!(
        f32_error,
        OperationError::UnsupportedTensorContractScope {
            message: "inv requires full-rank blocks"
        }
    ));

    let large_c64 = Complex64::from_polar(1.0, 0.23);
    let small_c64 = Complex64::from_polar(1e-8, -0.41);
    let tensor_c64 = u1_block_endomorphism(&[(
        0,
        2,
        vec![large_c64, Complex64::zero(), Complex64::zero(), small_c64],
    )]);
    let mut c64_context = TensorContractFusionExecutionContext::<
        Complex64,
        TreeTransformBuiltinRuleCacheKey,
    >::default();
    inv(
        &mut dense,
        &mut c64_context,
        &bound_tensor_ref!(Arc::new(U1FusionRule), &tensor_c64),
    )
    .unwrap();

    let large_c32 = Complex32::from_polar(1.0, 0.23);
    let small_c32 = Complex32::from_polar(1e-8, -0.41);
    let tensor_c32 = u1_block_endomorphism(&[(
        0,
        2,
        vec![large_c32, Complex32::zero(), Complex32::zero(), small_c32],
    )]);
    let mut c32_context = TensorContractFusionExecutionContext::<
        Complex32,
        TreeTransformBuiltinRuleCacheKey,
    >::default();
    let c32_error = inv(
        &mut dense,
        &mut c32_context,
        &bound_tensor_ref!(Arc::new(U1FusionRule), &tensor_c32),
    )
    .unwrap_err();
    assert!(matches!(
        c32_error,
        OperationError::UnsupportedTensorContractScope {
            message: "inv requires full-rank blocks"
        }
    ));
}

#[test]
fn inv_rejects_a_genuinely_singular_sector_without_an_output() {
    // What: one exact zero singular direction returns the typed full-rank error.
    let tensor = u1_block_endomorphism(&[(0, 2, vec![1.0_f64, 0.0, 0.0, 0.0])]);
    let mut dense = tenet_dense::DefaultDenseExecutor::new();
    let mut context =
        TensorContractFusionExecutionContext::<f64, TreeTransformBuiltinRuleCacheKey>::default();

    let error = inv(
        &mut dense,
        &mut context,
        &bound_tensor_ref!(Arc::new(U1FusionRule), &tensor),
    )
    .unwrap_err();

    assert!(matches!(
        error,
        OperationError::UnsupportedTensorContractScope {
            message: "inv requires full-rank blocks"
        }
    ));
    assert_eq!(context.tree_context().cache().plan_len(), 0);
    assert_eq!(context.tree_context().cache().structure_len(), 0);
    assert_eq!(context.dynamic_fusion_space_cache_len(), 0);
    assert_eq!(context.contraction_resolution_cache_len(), 0);
}

#[test]
fn inv_rejects_nonfinite_backend_spectra_before_recomposition() {
    // What: backend NaN and infinity spectra return the typed error without publishing an inverse.
    let tensor = u1_block_endomorphism(&[(0, 1, vec![1.0_f64])]);
    for singular_value in [f64::NAN, f64::INFINITY] {
        let mut dense = NonFiniteSvdSpectrum { singular_value };
        let mut context =
            TensorContractFusionExecutionContext::<f64, TreeTransformBuiltinRuleCacheKey>::default(
            );

        let error = inv(
            &mut dense,
            &mut context,
            &bound_tensor_ref!(Arc::new(U1FusionRule), &tensor),
        )
        .unwrap_err();

        assert!(matches!(
            error,
            OperationError::UnsupportedTensorContractScope {
                message: "inv requires full-rank blocks"
            }
        ));
        assert_eq!(context.tree_context().cache().plan_len(), 0);
        assert_eq!(context.tree_context().cache().structure_len(), 0);
        assert_eq!(context.dynamic_fusion_space_cache_len(), 0);
        assert_eq!(context.contraction_resolution_cache_len(), 0);
    }
}

#[test]
fn inv_accepts_a_zero_dimensional_endomorphism_without_dense_execution() {
    // What: the inverse of the legal empty endomorphism is the empty endomorphism.
    let tensor = rectangular_svd_tensor(0, 0);
    let mut dense = RejectExecutorCalls;
    let mut context =
        TensorContractFusionExecutionContext::<f64, TreeTransformBuiltinRuleCacheKey>::default();

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
fn pinv_keeps_its_global_rcond_cutoff() {
    // What: public pinv still drops singular values relative to the global maximum.
    let tensor = u1_block_endomorphism(&[(0, 1, vec![1.0_f64]), (1, 1, vec![1e-14])]);
    let mut dense = tenet_dense::DefaultDenseExecutor::new();
    let mut context =
        TensorContractFusionExecutionContext::<f64, TreeTransformBuiltinRuleCacheKey>::default();

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
    let mut context =
        TensorContractFusionExecutionContext::<f32, TreeTransformBuiltinRuleCacheKey>::default();
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
