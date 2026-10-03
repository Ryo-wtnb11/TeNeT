#[path = "../common/mod.rs"]
mod common;
#[path = "../../../tests/support/numerics.rs"]
mod numerics;

use std::fmt;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;
use tenet::typed::ContractSpec;
use tenet::typed::Side;

use tenet::expert::DenseBackend;
use tenet::expert::{
    DefaultDenseExecutor, DenseDotConfig, DenseError, DenseExecutor, DenseGemmBatchJob, DenseRead,
    DenseScalar, DenseTensor, DenseWrite, MatrixOp,
};
use tenet::sector::SectorId;
use tenet::sector::{
    BraidingStyleKind, CheckedGenericAdmissionMode, CheckedGenericFusion,
    CheckedGenericRigidSymbols, FusionStyleKind, GenericFArray, GenericRMatrix, RuleIdentity,
    SectorVec, TypedSectorAdmission,
};
use tenet::typed::__network::NetworkReuseClass;
use tenet::typed::CheckedGenericStructureError;
use tenet::typed::{
    CheckedGenericTensorProductError, Eig, Eigh, GradedSpace, LeftPolar, Lq, Qr, RightPolar, Svd,
    TensorMap, Truncation, TypedTensorConstructionDispatch,
};
use tenet::typed::{Complex32, Complex64, GenericTensorError, Runtime, SectorSpectrum};

/// The receiver's own split as leg roles: `rows = 0..nout`.
#[cfg(feature = "racah-generated")]
fn codomain_axes<R, D, S>(t: &tenet::typed::TensorMap<R, D, S>) -> Vec<usize> {
    (0..t.codomain_rank()).collect()
}

/// The receiver's own split as leg roles: `cols = nout..rank`.
#[cfg(feature = "racah-generated")]
fn domain_axes<R, D, S>(t: &tenet::typed::TensorMap<R, D, S>) -> Vec<usize> {
    (t.codomain_rank()..t.rank()).collect()
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
enum Label {
    Vacuum,
    One,
    Two,
    X,
    AliasX,
    Invalid,
}

/// `terms` of the tolerance rule for a product of endomorphisms: an
/// endomorphism payload holds `sum_c n_c^2` entries, so its square root
/// bounds every coupled block's contracted length `n_c`.
fn endomorphism_terms(payload_len: usize) -> usize {
    (payload_len as f64).sqrt().ceil() as usize
}

#[cfg(feature = "racah-generated")]
trait SunEigInput: tenet::typed::AdvancedLinalgScalar<Eig = Complex64> + fmt::Debug {
    fn to_complex(
        source: &TensorMap<tenet::sector::SUNFusionRule, Self>,
    ) -> TensorMap<tenet::sector::SUNFusionRule, Complex64>;
}

#[cfg(feature = "racah-generated")]
impl SunEigInput for f64 {
    fn to_complex(
        source: &TensorMap<tenet::sector::SUNFusionRule, Self>,
    ) -> TensorMap<tenet::sector::SUNFusionRule, Complex64> {
        source.convert::<Complex64>()
    }
}

#[cfg(feature = "racah-generated")]
impl SunEigInput for Complex64 {
    fn to_complex(
        source: &TensorMap<tenet::sector::SUNFusionRule, Self>,
    ) -> TensorMap<tenet::sector::SUNFusionRule, Complex64> {
        source.clone()
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
enum ToyError {
    InvalidSector,
    Decode,
    Algebra,
}

impl fmt::Display for ToyError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{self:?}")
    }
}

impl std::error::Error for ToyError {}

struct CheckedOnlyToy {
    identity_tag: u8,
    fail_algebra: AtomicBool,
    fail_dim: AtomicBool,
    fail_decode: AtomicBool,
    fail_decode_on_query: AtomicUsize,
    algebra_queries: AtomicUsize,
    coefficient_queries: AtomicUsize,
    f_queries: AtomicUsize,
    r_queries: AtomicUsize,
    malformed_f: AtomicBool,
    invalid_style: AtomicBool,
    extra_vacuum_channel: AtomicBool,
    use_product_probe: bool,
    fractional_dim: bool,
    fail_f_on_query: AtomicUsize,
    identity_queries: AtomicUsize,
    style_queries: AtomicUsize,
    commit_identity_seen: AtomicBool,
    committed: AtomicBool,
    commit_count: AtomicUsize,
    postcommit_queries: AtomicUsize,
    commit_after_queries: AtomicUsize,
    queries_since_reset: AtomicUsize,
}

impl CheckedOnlyToy {
    fn new(identity_tag: u8) -> Self {
        Self {
            identity_tag,
            fail_algebra: AtomicBool::new(false),
            fail_dim: AtomicBool::new(false),
            fail_decode: AtomicBool::new(false),
            fail_decode_on_query: AtomicUsize::new(0),
            algebra_queries: AtomicUsize::new(0),
            coefficient_queries: AtomicUsize::new(0),
            f_queries: AtomicUsize::new(0),
            r_queries: AtomicUsize::new(0),
            malformed_f: AtomicBool::new(false),
            invalid_style: AtomicBool::new(false),
            extra_vacuum_channel: AtomicBool::new(false),
            use_product_probe: false,
            fractional_dim: false,
            fail_f_on_query: AtomicUsize::new(0),
            identity_queries: AtomicUsize::new(0),
            style_queries: AtomicUsize::new(0),
            commit_identity_seen: AtomicBool::new(false),
            committed: AtomicBool::new(false),
            commit_count: AtomicUsize::new(0),
            postcommit_queries: AtomicUsize::new(0),
            commit_after_queries: AtomicUsize::new(0),
            queries_since_reset: AtomicUsize::new(0),
        }
    }

    fn new_product_probe(identity_tag: u8) -> Self {
        Self {
            use_product_probe: true,
            ..Self::new(identity_tag)
        }
    }

    fn new_space_probe(identity_tag: u8) -> Self {
        Self {
            use_product_probe: true,
            fractional_dim: true,
            ..Self::new(identity_tag)
        }
    }

    fn x(&self) -> SectorId {
        SectorId::new(3)
    }

    fn probe_fusion_channels(left: SectorId, right: SectorId) -> SectorVec {
        let ids: &[usize] = match (left.id(), right.id()) {
            (0, x) | (x, 0) => return [SectorId::new(x)].into_iter().collect(),
            (3, 3) | (3, 1) | (1, 3) => &[3],
            (1, 1) => &[1],
            _ => &[],
        };
        ids.iter().copied().map(SectorId::new).collect()
    }

    fn probe_nsymbol(left: SectorId, right: SectorId, coupled: SectorId) -> usize {
        if (left.id(), right.id(), coupled.id()) == (3, 3, 3) {
            2
        } else {
            usize::from(Self::probe_fusion_channels(left, right).contains(&coupled))
        }
    }

    fn fusion_channels(&self, left: SectorId, right: SectorId) -> SectorVec {
        let mut channels = if self.use_product_probe {
            Self::probe_fusion_channels(left, right)
        } else {
            match (left.id(), right.id()) {
                (0, x) | (x, 0) => [SectorId::new(x)].into_iter().collect(),
                (3, 3) => [SectorId::new(0), SectorId::new(3)].into_iter().collect(),
                _ => SectorVec::new(),
            }
        };
        if self.extra_vacuum_channel.load(Ordering::Relaxed) && left.id() == 0 && right == self.x()
        {
            channels.push(SectorId::new(0));
        }
        channels
    }

    fn nsymbol(&self, left: SectorId, right: SectorId, coupled: SectorId) -> usize {
        // The injected vacuum channel carries a real multiplicity: a listed
        // channel with N = 0 violates the provider contract.
        if self.extra_vacuum_channel.load(Ordering::Relaxed)
            && left.id() == 0
            && right == self.x()
            && coupled.id() == 0
        {
            return 1;
        }
        if self.use_product_probe {
            Self::probe_nsymbol(left, right, coupled)
        } else if (left.id(), right.id(), coupled.id()) == (3, 3, 3) {
            2
        } else {
            usize::from(self.fusion_channels(left, right).contains(&coupled))
        }
    }

    fn reset_commit_spy(&self) {
        self.commit_identity_seen.store(false, Ordering::Relaxed);
        self.committed.store(false, Ordering::Relaxed);
        self.commit_count.store(0, Ordering::Relaxed);
        self.postcommit_queries.store(0, Ordering::Relaxed);
        self.commit_after_queries.store(0, Ordering::Relaxed);
        self.queries_since_reset.store(0, Ordering::Relaxed);
    }

    fn arm_commit_spy_after_queries(&self, query_count: usize) {
        self.reset_commit_spy();
        assert!(query_count > 0);
        self.commit_after_queries
            .store(query_count, Ordering::Relaxed);
    }

    fn record_query(&self) -> usize {
        let query = self.queries_since_reset.fetch_add(1, Ordering::Relaxed) + 1;
        let commit_after = self.commit_after_queries.load(Ordering::Relaxed);
        if commit_after != 0 && query == commit_after {
            if !self.committed.swap(true, Ordering::Relaxed) {
                self.commit_count.fetch_add(1, Ordering::Relaxed);
            }
        } else if self.committed.load(Ordering::Relaxed) {
            self.postcommit_queries.fetch_add(1, Ordering::Relaxed);
        }
        query
    }
}

fn reset_provider_queries(provider: &CheckedOnlyToy) {
    for counter in [
        &provider.algebra_queries,
        &provider.coefficient_queries,
        &provider.f_queries,
        &provider.r_queries,
        &provider.identity_queries,
        &provider.style_queries,
        &provider.queries_since_reset,
    ] {
        counter.store(0, Ordering::Relaxed);
    }
}

fn assert_no_provider_queries(provider: &CheckedOnlyToy) {
    for counter in [
        &provider.algebra_queries,
        &provider.coefficient_queries,
        &provider.f_queries,
        &provider.r_queries,
        &provider.identity_queries,
        &provider.style_queries,
    ] {
        assert_eq!(counter.load(Ordering::Relaxed), 0);
    }
    assert_eq!(provider.queries_since_reset.load(Ordering::Relaxed), 0);
}

impl CheckedGenericFusion for CheckedOnlyToy {
    type Error = ToyError;

    fn rule_identity(&self) -> RuleIdentity {
        self.identity_queries.fetch_add(1, Ordering::Relaxed);
        self.record_query();
        if self.commit_after_queries.load(Ordering::Relaxed) == 0
            && self.f_queries.load(Ordering::Relaxed) > 0
        {
            self.commit_identity_seen.store(true, Ordering::Relaxed);
        }
        RuleIdentity::from_canonical_bytes::<Self>(
            0x677,
            Arc::<[u8]>::from([
                self.identity_tag,
                u8::from(self.use_product_probe),
                u8::from(self.fractional_dim),
            ]),
        )
    }

    fn fusion_style(&self) -> FusionStyleKind {
        self.style_queries.fetch_add(1, Ordering::Relaxed);
        self.record_query();
        if self.commit_after_queries.load(Ordering::Relaxed) == 0
            && self.commit_identity_seen.load(Ordering::Relaxed)
            && !self.committed.swap(true, Ordering::Relaxed)
        {
            self.commit_count.fetch_add(1, Ordering::Relaxed);
        }
        if self.invalid_style.load(Ordering::Relaxed) {
            FusionStyleKind::Unique
        } else {
            FusionStyleKind::Generic
        }
    }

    fn braiding_style(&self) -> BraidingStyleKind {
        self.record_query();
        BraidingStyleKind::Bosonic
    }

    fn vacuum(&self) -> SectorId {
        self.record_query();
        SectorId::new(0)
    }

    fn try_dual(&self, sector: SectorId) -> Result<SectorId, Self::Error> {
        self.record_query();
        self.algebra_queries.fetch_add(1, Ordering::Relaxed);
        Ok(sector)
    }

    fn try_fusion_channels(
        &self,
        left: SectorId,
        right: SectorId,
    ) -> Result<SectorVec, Self::Error> {
        self.record_query();
        self.algebra_queries.fetch_add(1, Ordering::Relaxed);
        if self.fail_algebra.load(Ordering::Relaxed) {
            return Err(ToyError::Algebra);
        }
        Ok(self.fusion_channels(left, right))
    }

    fn try_fusion_channels_in_table(
        &self,
        left: SectorId,
        right: SectorId,
    ) -> Result<SectorVec, Self::Error> {
        self.record_query();
        self.algebra_queries.fetch_add(1, Ordering::Relaxed);
        Ok(self.fusion_channels(left, right))
    }

    fn try_nsymbol(
        &self,
        left: SectorId,
        right: SectorId,
        coupled: SectorId,
    ) -> Result<usize, Self::Error> {
        self.record_query();
        self.algebra_queries.fetch_add(1, Ordering::Relaxed);
        Ok(self.nsymbol(left, right, coupled))
    }
}

impl CheckedGenericRigidSymbols for CheckedOnlyToy {
    type Scalar = f64;

    fn try_sqrt_dim_scalar(&self, sector: SectorId) -> Result<f64, Self::Error> {
        self.record_query();
        self.coefficient_queries.fetch_add(1, Ordering::Relaxed);
        if self.fail_algebra.load(Ordering::Relaxed) || self.fail_dim.load(Ordering::Relaxed) {
            return Err(ToyError::Algebra);
        }
        Ok(if sector.id() == 3 {
            if self.fractional_dim {
                2.5_f64
            } else {
                1.0 + 2.0_f64.sqrt()
            }
            .sqrt()
        } else {
            1.0
        })
    }

    fn try_inv_sqrt_dim_scalar(&self, sector: SectorId) -> Result<f64, Self::Error> {
        self.record_query();
        self.coefficient_queries.fetch_add(1, Ordering::Relaxed);
        Ok(if sector.id() == 3 {
            1.0 / if self.fractional_dim {
                2.5_f64
            } else {
                1.0 + 2.0_f64.sqrt()
            }
            .sqrt()
        } else {
            1.0
        })
    }

    fn try_frobenius_schur_phase_scalar(&self, sector: SectorId) -> Result<f64, Self::Error> {
        self.record_query();
        self.coefficient_queries.fetch_add(1, Ordering::Relaxed);
        let _ = sector;
        Ok(1.0)
    }

    fn try_f_symbol_generic(
        &self,
        a: SectorId,
        b: SectorId,
        c: SectorId,
        d: SectorId,
        e: SectorId,
        f: SectorId,
    ) -> Result<GenericFArray<f64>, Self::Error> {
        self.record_query();
        self.coefficient_queries.fetch_add(1, Ordering::Relaxed);
        let query = self.f_queries.fetch_add(1, Ordering::Relaxed) + 1;
        if self.fail_algebra.load(Ordering::Relaxed)
            || self.fail_f_on_query.load(Ordering::Relaxed) == query
        {
            return Err(ToyError::Algebra);
        }
        let shape = (
            self.nsymbol(a, b, e),
            self.nsymbol(e, c, d),
            self.nsymbol(b, c, f),
            self.nsymbol(a, f, d),
        );
        let len = shape.0 * shape.1 * shape.2 * shape.3;
        let symbol = if self.use_product_probe {
            let data = (0..len)
                .map(|index| {
                    let magnitude = (index + 1) as f64;
                    if index % 2 == 0 {
                        magnitude
                    } else {
                        -magnitude
                    }
                })
                .collect();
            GenericFArray::new(data, shape)
        } else if e == f {
            let cols = shape.2 * shape.3;
            GenericFArray::new(
                (0..len)
                    .map(|index| f64::from(index / cols == index % cols))
                    .collect(),
                shape,
            )
        } else {
            GenericFArray::new(vec![0.0; len], shape)
        };
        if self.malformed_f.load(Ordering::Relaxed) {
            Ok(GenericFArray::new(
                symbol.data().to_vec(),
                (1, 1, symbol.data().len(), 1),
            ))
        } else {
            Ok(symbol)
        }
    }

    fn try_r_symbol_generic(
        &self,
        a: SectorId,
        b: SectorId,
        c: SectorId,
    ) -> Result<GenericRMatrix<f64>, Self::Error> {
        self.record_query();
        self.coefficient_queries.fetch_add(1, Ordering::Relaxed);
        self.r_queries.fetch_add(1, Ordering::Relaxed);
        if self.fail_algebra.load(Ordering::Relaxed) {
            return Err(ToyError::Algebra);
        }
        let rows = self.nsymbol(a, b, c);
        Ok(GenericRMatrix::new(
            (0..rows * rows)
                .map(|index| f64::from(index / rows == index % rows))
                .collect(),
            rows,
            rows,
        ))
    }
}

impl TypedSectorAdmission for CheckedOnlyToy {
    type Sector = Label;
    type Error = ToyError;
    type Mode = CheckedGenericAdmissionMode;

    fn typed_rule_identity(&self) -> RuleIdentity {
        CheckedGenericFusion::rule_identity(self)
    }

    fn try_encode_label(&self, sector: &Self::Sector) -> Result<SectorId, Self::Error> {
        self.record_query();
        match sector {
            Label::Vacuum => Ok(self.vacuum()),
            Label::One if self.use_product_probe => Ok(SectorId::new(1)),
            Label::Two if self.use_product_probe => Ok(SectorId::new(2)),
            Label::X => Ok(self.x()),
            Label::AliasX => Ok(self.x()),
            Label::One | Label::Two | Label::Invalid => Err(ToyError::InvalidSector),
        }
    }

    fn try_decode_label(&self, sector: SectorId) -> Result<Self::Sector, Self::Error> {
        let query = self.record_query();
        if self.fail_decode.load(Ordering::Relaxed)
            || self.fail_decode_on_query.load(Ordering::Relaxed) == query
        {
            return Err(ToyError::Decode);
        }
        if sector == self.vacuum() {
            Ok(Label::Vacuum)
        } else if self.use_product_probe && sector == SectorId::new(1) {
            Ok(Label::One)
        } else if self.use_product_probe && sector == SectorId::new(2) {
            Ok(Label::Two)
        } else if sector == self.x() {
            Ok(Label::X)
        } else {
            Err(ToyError::InvalidSector)
        }
    }

    fn try_dual_id(&self, sector: SectorId) -> Result<SectorId, Self::Error> {
        CheckedGenericFusion::try_dual(self, sector)
    }
}

fn complex_spectra<S: Clone>(
    sectors: [S; 2],
    values: [&[(f64, f64)]; 2],
) -> Vec<SectorSpectrum<S, Complex64>> {
    sectors
        .into_iter()
        .zip(values)
        .map(|(sector, values)| SectorSpectrum {
            sector,
            values: values
                .iter()
                .map(|&(re, im)| Complex64::new(re, im))
                .collect(),
        })
        .collect()
}

fn assert_same_checked_generic_layout_and_close<R, D>(
    actual: &TensorMap<R, D>,
    expected: &TensorMap<R, D>,
    close: impl Fn(D, D) -> f64,
) where
    R: TypedSectorAdmission,
    D: tenet::typed::TensorScalar + fmt::Debug,
{
    assert_eq!(actual.subblock_count(), expected.subblock_count());
    for index in 0..actual.subblock_count() {
        let actual_block = actual.subblock(index).unwrap();
        let expected_block = expected.subblock(index).unwrap();
        assert_eq!(actual_block.key(), expected_block.key());
        assert_eq!(actual_block.shape(), expected_block.shape());
        assert_eq!(actual_block.strides(), expected_block.strides());
    }
    assert_eq!(
        actual.dense_data().unwrap().len(),
        expected.materialize().unwrap().dense_data().unwrap().len()
    );
    for (&actual, &expected) in actual
        .dense_data()
        .unwrap()
        .iter()
        .zip(expected.materialize().unwrap().dense_data().unwrap())
    {
        assert!(
            close(actual, expected) < 1e-10,
            "{actual:?} != {expected:?}"
        );
    }
}

include!("../common/spy_executor.rs");

/// The full and values-only SVD entries a pinv/polar call may reach.
const PINV_SVD: &[Kernel] = &[Kernel::Svd, Kernel::SvdInto, Kernel::SvdVals];

/// Counts SVD (`PINV_SVD`) and GEMM (`Kernel::GEMM`) calls; the `fail_svd`-th
/// SVD or the `fail_gemm`-th GEMM call fails.
fn pinv_spy(
    counts: &Arc<SpyCounts>,
    fail_svd: Option<usize>,
    fail_gemm: Option<usize>,
) -> SpyExecutor {
    let mut spy = SpyExecutor::counting(counts);
    if let Some(nth) = fail_svd {
        spy = spy.failing(PINV_SVD, Some(nth), "injected pinv SVD failure");
    }
    if let Some(nth) = fail_gemm {
        spy = spy.failing(Kernel::GEMM, Some(nth), "injected pinv GEMM failure");
    }
    spy
}

// ---------------------------------------------------------------------------
// #1201: lazy-adjoint reads through FusionOperand + the shared strided owner,
// and checked `tr` collapsed onto `weighted_trace`.
// ---------------------------------------------------------------------------

fn lazy_oracle_value(
    trees: &tenet::typed::BlockFusionTrees<Label>,
    indices: &[usize],
) -> Complex64 {
    let tree = (format!("{trees:?}").bytes().fold(0u32, |acc, byte| {
        acc.wrapping_mul(31).wrapping_add(byte as u32)
    }) % 97) as f64;
    let mut re = tree;
    let mut im = -0.5 * tree + 0.25;
    for (axis, &index) in indices.iter().enumerate() {
        re += (index as f64 + 1.0) * (axis as f64 + 1.0);
        im += (index as f64 + 1.0) * (axis as f64 + 1.0) * (axis as f64 + 1.0) * 0.5;
    }
    Complex64::new(re, im)
}

fn lazy_close_c64(a: Complex64, b: Complex64) -> bool {
    (a - b).norm() <= 1e-12 * (1.0 + b.norm())
}

fn lazy_close_f64(a: f64, b: f64) -> bool {
    (a - b).abs() <= 1e-12 * (1.0 + b.abs())
}

mod construction;
mod eig;
mod matrix_fn;
mod null_polar;
mod pinv;
mod qr_lq;
mod reductions;
mod svd;
mod transforms;
