use std::fmt;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;

use tenet::core::{
    BraidingStyleKind, CheckedGenericAdmissionMode, CheckedGenericFusion,
    CheckedGenericRigidSymbols, CheckedGenericStructureError, FusionRule, FusionStyleKind,
    GenericFArray, GenericFusionSymbols, GenericRMatrix, GenericRigidSymbols, RuleIdentity,
    SectorId, SectorVec, Su3FusionRule, TypedSectorAdmission,
};
use tenet::prelude::{Complex64, Runtime};
use tenet::typed::{GenericTensorError, GradedSpace, TensorMap};

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
enum Label {
    Vacuum,
    X,
    Invalid,
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
    rule: Su3FusionRule,
    identity_tag: u8,
    fail_algebra: AtomicBool,
    fail_decode: AtomicBool,
    algebra_queries: AtomicUsize,
    coefficient_queries: AtomicUsize,
}

impl CheckedOnlyToy {
    fn new(identity_tag: u8) -> Self {
        Self {
            rule: Su3FusionRule::new(),
            identity_tag,
            fail_algebra: AtomicBool::new(false),
            fail_decode: AtomicBool::new(false),
            algebra_queries: AtomicUsize::new(0),
            coefficient_queries: AtomicUsize::new(0),
        }
    }

    fn x(&self) -> SectorId {
        self.rule.sector_of(1, 1).unwrap()
    }
}

impl CheckedGenericFusion for CheckedOnlyToy {
    type Error = ToyError;

    fn rule_identity(&self) -> RuleIdentity {
        RuleIdentity::from_canonical_bytes::<Self>(0x677, Arc::<[u8]>::from([self.identity_tag]))
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
        self.algebra_queries.fetch_add(1, Ordering::Relaxed);
        Ok(self.rule.dual(sector))
    }

    fn try_fusion_channels(
        &self,
        left: SectorId,
        right: SectorId,
    ) -> Result<SectorVec, Self::Error> {
        self.algebra_queries.fetch_add(1, Ordering::Relaxed);
        if self.fail_algebra.load(Ordering::Relaxed) {
            return Err(ToyError::Algebra);
        }
        Ok(self.rule.fusion_channels(left, right))
    }

    fn try_fusion_channels_in_table(
        &self,
        left: SectorId,
        right: SectorId,
    ) -> Result<SectorVec, Self::Error> {
        self.algebra_queries.fetch_add(1, Ordering::Relaxed);
        Ok(self.rule.fusion_channels_in_table(left, right))
    }

    fn try_nsymbol(
        &self,
        left: SectorId,
        right: SectorId,
        coupled: SectorId,
    ) -> Result<usize, Self::Error> {
        self.algebra_queries.fetch_add(1, Ordering::Relaxed);
        Ok(self.rule.nsymbol(left, right, coupled))
    }
}

impl CheckedGenericRigidSymbols for CheckedOnlyToy {
    type Scalar = f64;

    fn try_sqrt_dim_scalar(&self, sector: SectorId) -> Result<f64, Self::Error> {
        self.coefficient_queries.fetch_add(1, Ordering::Relaxed);
        Ok(self.rule.sqrt_dim_scalar(sector))
    }

    fn try_inv_sqrt_dim_scalar(&self, sector: SectorId) -> Result<f64, Self::Error> {
        self.coefficient_queries.fetch_add(1, Ordering::Relaxed);
        Ok(self.rule.inv_sqrt_dim_scalar(sector))
    }

    fn try_frobenius_schur_phase_scalar(&self, sector: SectorId) -> Result<f64, Self::Error> {
        self.coefficient_queries.fetch_add(1, Ordering::Relaxed);
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
    ) -> Result<GenericFArray<f64>, Self::Error> {
        self.coefficient_queries.fetch_add(1, Ordering::Relaxed);
        if self.fail_algebra.load(Ordering::Relaxed) {
            return Err(ToyError::Algebra);
        }
        Ok(self.rule.f_symbol_generic(a, b, c, d, e, f))
    }

    fn try_r_symbol_generic(
        &self,
        a: SectorId,
        b: SectorId,
        c: SectorId,
    ) -> Result<GenericRMatrix<f64>, Self::Error> {
        self.coefficient_queries.fetch_add(1, Ordering::Relaxed);
        if self.fail_algebra.load(Ordering::Relaxed) {
            return Err(ToyError::Algebra);
        }
        Ok(self.rule.r_symbol_generic(a, b, c))
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
        match sector {
            Label::Vacuum => Ok(self.rule.vacuum()),
            Label::X => Ok(self.x()),
            Label::Invalid => Err(ToyError::InvalidSector),
        }
    }

    fn try_decode_label(&self, sector: SectorId) -> Result<Self::Sector, Self::Error> {
        if self.fail_decode.load(Ordering::Relaxed) {
            return Err(ToyError::Decode);
        }
        if sector == self.rule.vacuum() {
            Ok(Label::Vacuum)
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

#[test]
fn checked_only_provider_uses_ordinary_typed_ownership_and_vertices() {
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let first = Arc::new(CheckedOnlyToy::new(0));
    let second = Arc::new(CheckedOnlyToy::new(0));
    let left = GradedSpace::try_new(Arc::clone(&first), [(Label::X, 2)], false).unwrap();
    let right = GradedSpace::try_new(Arc::clone(&second), [(Label::X, 2)], false).unwrap();

    let tensor: TensorMap<_, f64> = TensorMap::zeros(&runtime, [&left, &right], [&right]).unwrap();
    assert!(std::ptr::eq(tensor.provider(), first.as_ref()));
    assert_eq!(tensor.rank(), 3);
    assert_eq!(tensor.block_count(), 2);
    let vertices: Vec<_> = (0..tensor.block_count())
        .map(|index| {
            let trees = tensor.block_fusion_trees(index).unwrap();
            assert_eq!(trees.coupled(), &Label::X);
            assert_eq!(trees.codomain_uncoupled(), &[Label::X, Label::X]);
            assert!(trees.codomain_innerlines().is_empty());
            assert_eq!(trees.domain_uncoupled(), &[Label::X]);
            assert!(trees.domain_vertices().is_empty());
            trees.codomain_vertices()[0].get()
        })
        .collect();
    assert_eq!(vertices, [1, 2]);
    assert_eq!(left.sectors().unwrap(), [Label::X]);

    let clone = tensor.clone();
    assert!(std::ptr::eq(clone.provider(), first.as_ref()));
    assert_eq!(clone.data().as_ptr(), tensor.data().as_ptr());
}

#[test]
fn checked_only_multiplicity_two_transforms_keep_the_source_authority() {
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let provider = Arc::new(CheckedOnlyToy::new(0));
    let leg = GradedSpace::try_new(Arc::clone(&provider), [(Label::X, 1)], false).unwrap();
    let source: TensorMap<_, f64> =
        TensorMap::from_block_fn(&runtime, [&leg, &leg], [&leg], |trees, _| {
            trees.codomain_vertices()[0].get() as f64
        })
        .unwrap();
    let snapshot = |tensor: &TensorMap<CheckedOnlyToy, f64>| {
        (0..tensor.block_count())
            .map(|index| tensor.block_fusion_trees(index).unwrap())
            .collect::<Vec<_>>()
    };
    let source_snapshot = snapshot(&source);
    provider.coefficient_queries.store(0, Ordering::Relaxed);
    let error = source.braid(&[1, 0], &[2], &[0, 1]).unwrap_err();
    assert!(matches!(error, GenericTensorError::Facade(_)));
    assert_eq!(provider.coefficient_queries.load(Ordering::Relaxed), 0);

    let permuted = source.permute(&[1, 0], &[2]).unwrap();
    assert!(std::ptr::eq(permuted.provider(), provider.as_ref()));
    let restored = permuted.permute(&[1, 0], &[2]).unwrap();
    assert_eq!(snapshot(&restored), source_snapshot);
    for (actual, expected) in restored.data().iter().zip(source.data()) {
        assert!((actual - expected).abs() <= 1e-12);
    }

    let braided = source.braid(&[1, 0], &[2], &[0, 1, 2]).unwrap();
    assert!(std::ptr::eq(braided.provider(), provider.as_ref()));

    let repartitioned = source.repartition(1).unwrap();
    assert!(std::ptr::eq(repartitioned.provider(), provider.as_ref()));
    let restored = repartitioned.repartition(2).unwrap();
    assert_eq!(snapshot(&restored), source_snapshot);
    for (actual, expected) in restored.data().iter().zip(source.data()) {
        assert!((actual - expected).abs() <= 1e-12);
    }

    provider.fail_algebra.store(true, Ordering::Relaxed);
    let error = source.permute(&[1, 0], &[2]).unwrap_err();
    assert!(matches!(
        error,
        GenericTensorError::Plan(tenet::typed::CheckedGenericPlanError::Provider(
            ToyError::Algebra
        ))
    ));
}

#[test]
fn checked_errors_stay_typed_and_callback_waits_for_all_decodes() {
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let provider = Arc::new(CheckedOnlyToy::new(0));
    let error =
        GradedSpace::try_new(Arc::clone(&provider), [(Label::Invalid, 1)], false).unwrap_err();
    assert!(matches!(
        error,
        GenericTensorError::Structure(CheckedGenericStructureError::Provider(
            ToyError::InvalidSector
        ))
    ));

    let leg = GradedSpace::try_new(Arc::clone(&provider), [(Label::X, 1)], false).unwrap();
    provider.fail_decode.store(true, Ordering::Relaxed);
    let callbacks = AtomicUsize::new(0);
    let error = TensorMap::<_, f64>::from_block_fn(&runtime, [&leg, &leg], [&leg], |_, _| {
        callbacks.fetch_add(1, Ordering::Relaxed);
        1.0
    })
    .unwrap_err();
    assert!(matches!(
        error,
        GenericTensorError::Structure(CheckedGenericStructureError::Provider(ToyError::Decode))
    ));
    assert_eq!(callbacks.load(Ordering::Relaxed), 0);
}

#[test]
fn identity_mismatch_precedes_algebra_queries_and_both_dtypes_fill() {
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let first = Arc::new(CheckedOnlyToy::new(0));
    let other = Arc::new(CheckedOnlyToy::new(1));
    let left = GradedSpace::try_new(Arc::clone(&first), [(Label::X, 1)], false).unwrap();
    let right = GradedSpace::try_new(Arc::clone(&other), [(Label::X, 1)], false).unwrap();
    let error = TensorMap::<_, f64>::zeros(&runtime, [&left], [&right]).unwrap_err();
    assert!(matches!(error, GenericTensorError::Facade(_)));
    assert_eq!(first.algebra_queries.load(Ordering::Relaxed), 0);
    assert_eq!(other.algebra_queries.load(Ordering::Relaxed), 0);

    let real: TensorMap<_, f64> =
        TensorMap::rand_with_seed(&runtime, [&left, &left], [&left], 7).unwrap();
    let complex: TensorMap<_, Complex64> =
        TensorMap::rand_with_seed(&runtime, [&left, &left], [&left], 7).unwrap();
    assert_eq!(real.data().len(), complex.data().len());
    assert!(complex.data().iter().any(|value| value.im != 0.0));
}

#[test]
fn failed_checked_admission_does_not_advance_the_runtime_stream() {
    let runtime_a = Runtime::builder().dense_threads(1).build().unwrap();
    let runtime_b = Runtime::builder().dense_threads(1).build().unwrap();
    let provider = Arc::new(CheckedOnlyToy::new(7));
    let leg = GradedSpace::try_new(Arc::clone(&provider), [(Label::X, 1)], false).unwrap();

    provider.fail_algebra.store(true, Ordering::Relaxed);
    assert!(TensorMap::<_, f64>::rand(&runtime_a, [&leg, &leg], [&leg]).is_err());
    provider.fail_algebra.store(false, Ordering::Relaxed);

    let after_failure = TensorMap::<_, f64>::rand(&runtime_a, [&leg, &leg], [&leg]).unwrap();
    let control = TensorMap::<_, f64>::rand(&runtime_b, [&leg, &leg], [&leg]).unwrap();
    assert_eq!(after_failure.data(), control.data());
}

#[cfg(feature = "racah-generated")]
#[test]
fn sun_adjoint_multiplicity_transforms_round_trip_labels_vertices_and_payload() {
    use tenet::typed::SUNFusionRule;

    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    for (n, adjoint) in [(3, vec![1, 1]), (4, vec![1, 0, 1])] {
        let provider = Arc::new(SUNFusionRule::new(n).unwrap());
        let leg =
            GradedSpace::try_new(Arc::clone(&provider), [(adjoint.clone(), 1)], false).unwrap();
        let tensor: TensorMap<_, f64> =
            TensorMap::from_block_fn(&runtime, [&leg, &leg], [&leg], |trees, _| {
                trees.codomain_vertices()[0].get() as f64
            })
            .unwrap();
        assert_eq!(tensor.block_count(), 2);
        for index in 0..2 {
            let trees = tensor.block_fusion_trees(index).unwrap();
            assert_eq!(trees.coupled(), &adjoint);
            assert_eq!(
                trees.codomain_uncoupled(),
                &[adjoint.clone(), adjoint.clone()]
            );
            assert_eq!(trees.domain_uncoupled(), &[adjoint.clone()]);
            assert_eq!(trees.codomain_vertices()[0].get(), index + 1);
            assert_eq!(tensor.block(index).unwrap().shape(), &[1, 1, 1]);
        }
        assert_eq!(tensor.data(), &[1.0, 2.0]);

        let snapshot = |tensor: &TensorMap<SUNFusionRule, f64>| {
            (0..tensor.block_count())
                .map(|index| tensor.block_fusion_trees(index).unwrap())
                .collect::<Vec<_>>()
        };
        let source_snapshot = snapshot(&tensor);
        for restored in [
            tensor
                .permute(&[1, 0], &[2])
                .unwrap()
                .permute(&[1, 0], &[2])
                .unwrap(),
            tensor
                .braid(&[0, 2], &[1], &[0, 1, 2])
                .unwrap()
                .braid(&[0, 2], &[1], &[0, 1, 2])
                .unwrap(),
            tensor.repartition(1).unwrap().repartition(2).unwrap(),
        ] {
            assert!(std::ptr::eq(restored.provider(), provider.as_ref()));
            assert_eq!(snapshot(&restored), source_snapshot);
            for (actual, expected) in restored.data().iter().zip(tensor.data()) {
                assert!((actual - expected).abs() <= 1e-10);
            }
        }
    }
}
