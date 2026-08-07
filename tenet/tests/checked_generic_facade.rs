use std::fmt;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;

use tenet::core::{
    BraidingStyleKind, CheckedGenericAdmissionMode, CheckedGenericFusion,
    CheckedGenericRigidSymbols, CheckedGenericStructureError, FusionStyleKind, GenericFArray,
    GenericRMatrix, RuleIdentity, SectorId, SectorVec, TypedSectorAdmission,
};
use tenet::prelude::{Complex64, Runtime};
use tenet::typed::{
    CheckedGenericTensorProductError, GenericTensorError, GradedSpace, TensorMap, Truncation,
};

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
enum Label {
    Vacuum,
    One,
    Two,
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
    identity_tag: u8,
    fail_algebra: AtomicBool,
    fail_decode: AtomicBool,
    algebra_queries: AtomicUsize,
    coefficient_queries: AtomicUsize,
    f_queries: AtomicUsize,
    r_queries: AtomicUsize,
    malformed_f: AtomicBool,
    invalid_style: AtomicBool,
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
            fail_decode: AtomicBool::new(false),
            algebra_queries: AtomicUsize::new(0),
            coefficient_queries: AtomicUsize::new(0),
            f_queries: AtomicUsize::new(0),
            r_queries: AtomicUsize::new(0),
            malformed_f: AtomicBool::new(false),
            invalid_style: AtomicBool::new(false),
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
        if self.use_product_probe {
            Self::probe_fusion_channels(left, right)
        } else {
            match (left.id(), right.id()) {
                (0, x) | (x, 0) => [SectorId::new(x)].into_iter().collect(),
                (3, 3) => [SectorId::new(0), SectorId::new(3)].into_iter().collect(),
                _ => SectorVec::new(),
            }
        }
    }

    fn nsymbol(&self, left: SectorId, right: SectorId, coupled: SectorId) -> usize {
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

    fn record_query(&self) {
        let query = self.queries_since_reset.fetch_add(1, Ordering::Relaxed) + 1;
        let commit_after = self.commit_after_queries.load(Ordering::Relaxed);
        if commit_after != 0 && query == commit_after {
            if !self.committed.swap(true, Ordering::Relaxed) {
                self.commit_count.fetch_add(1, Ordering::Relaxed);
            }
        } else if self.committed.load(Ordering::Relaxed) {
            self.postcommit_queries.fetch_add(1, Ordering::Relaxed);
        }
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
        if self.fail_algebra.load(Ordering::Relaxed) {
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
            Label::One | Label::Two | Label::Invalid => Err(ToyError::InvalidSector),
        }
    }

    fn try_decode_label(&self, sector: SectorId) -> Result<Self::Sector, Self::Error> {
        self.record_query();
        if self.fail_decode.load(Ordering::Relaxed) {
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

#[test]
fn checked_generic_space_algebra_keeps_multiplicity_dimensions_and_failures_typed() {
    let provider = Arc::new(CheckedOnlyToy::new_space_probe(7));
    let rhs_provider = Arc::new(CheckedOnlyToy::new_space_probe(7));
    let left = GradedSpace::try_new(Arc::clone(&provider), [(Label::X, 2)], false).unwrap();
    let right = GradedSpace::try_new(Arc::clone(&rhs_provider), [(Label::X, 3)], false).unwrap();

    let dim = left.dim().unwrap();
    assert!((dim - 5.0).abs() < 1.0e-12);
    assert_ne!(dim, 6.0);
    let fused = left.fuse(&right).unwrap();
    assert_eq!(fused.degeneracy(&Label::X).unwrap(), 12);
    assert!(std::ptr::eq(fused.provider(), provider.as_ref()));
    assert!(!std::ptr::eq(fused.provider(), rhs_provider.as_ref()));
    let summed = left.oplus(&right).unwrap();
    assert_eq!(summed.degeneracy(&Label::X).unwrap(), 5);
    assert!(std::ptr::eq(summed.provider(), provider.as_ref()));
    assert!(!std::ptr::eq(summed.provider(), rhs_provider.as_ref()));
    let unit = left.unitspace().unwrap();
    assert!(std::ptr::eq(unit.provider(), provider.as_ref()));
    assert_eq!(unit.degeneracy(&Label::Vacuum).unwrap(), 1);

    let foreign_provider = Arc::new(CheckedOnlyToy::new_space_probe(8));
    let foreign =
        GradedSpace::try_new(Arc::clone(&foreign_provider), [(Label::X, 1)], false).unwrap();
    let before = provider.algebra_queries.load(Ordering::Relaxed)
        + foreign_provider.algebra_queries.load(Ordering::Relaxed);
    assert!(matches!(
        left.oplus(&foreign),
        Err(GenericTensorError::Facade(
            tenet::prelude::Error::RuleMismatch
        ))
    ));
    assert!(matches!(
        left.fuse(&foreign),
        Err(GenericTensorError::Facade(
            tenet::prelude::Error::RuleMismatch
        ))
    ));
    assert_eq!(
        provider.algebra_queries.load(Ordering::Relaxed)
            + foreign_provider.algebra_queries.load(Ordering::Relaxed),
        before
    );

    provider.fail_algebra.store(true, Ordering::Relaxed);
    assert!(matches!(
        left.fuse(&right),
        Err(GenericTensorError::Structure(
            CheckedGenericStructureError::Provider(ToyError::Algebra)
        ))
    ));
    assert_eq!(left.degeneracy(&Label::X).unwrap(), 2);
    assert_eq!(right.degeneracy(&Label::X).unwrap(), 3);
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

#[cfg(feature = "cuda")]
#[test]
#[ignore]
fn checked_only_provider_roundtrips_through_typed_cuda_without_algebra_dispatch() {
    let runtime = Runtime::builder().cuda(0).build().unwrap();
    let provider = Arc::new(CheckedOnlyToy::new(0));
    let codomain = GradedSpace::try_new(
        Arc::clone(&provider),
        [(Label::Vacuum, 1), (Label::X, 2)],
        false,
    )
    .unwrap();
    let domain = GradedSpace::try_new(
        Arc::clone(&provider),
        [(Label::Vacuum, 1), (Label::X, 3)],
        true,
    )
    .unwrap();
    let source: TensorMap<_, f64> =
        TensorMap::from_block_fn(&runtime, [&codomain], [&domain], |_, indices| {
            indices.iter().sum::<usize>() as f64 + 1.0
        })
        .unwrap();
    let vertex_leg = GradedSpace::try_new(Arc::clone(&provider), [(Label::X, 2)], false).unwrap();
    let vertex_source: TensorMap<_, f64> = TensorMap::from_block_fn(
        &runtime,
        [&vertex_leg, &vertex_leg],
        [&vertex_leg],
        |trees, indices| {
            trees.codomain_vertices()[0].get() as f64 + indices.iter().sum::<usize>() as f64
        },
    )
    .unwrap();
    let block_structure = |tensor: &TensorMap<CheckedOnlyToy, f64>| {
        (0..tensor.block_count())
            .map(|index| {
                let block = tensor.block(index).unwrap();
                (
                    block.key().clone(),
                    tensor.block_fusion_trees(index).unwrap(),
                    block.offset(),
                    block.shape().to_vec(),
                    block.strides().to_vec(),
                )
            })
            .collect::<Vec<_>>()
    };
    let structure = |tensor: &TensorMap<CheckedOnlyToy, f64>| {
        let mut codomain_legs = Vec::new();
        let mut domain_legs = Vec::new();
        for index in 0..tensor.block_count() {
            let block = tensor.block(index).unwrap();
            let trees = tensor.block_fusion_trees(index).unwrap();
            let tenet::core::BlockKey::FusionTree(raw_trees) = block.key() else {
                panic!("checked Generic tensors use fusion-tree block keys")
            };
            assert_eq!(trees.codomain_uncoupled().len(), 1);
            assert_eq!(trees.domain_uncoupled().len(), 1);
            assert_eq!(block.shape().len(), 2);
            codomain_legs.push((
                trees.codomain_uncoupled()[0],
                block.shape()[0],
                raw_trees.codomain_tree().is_dual()[0],
            ));
            domain_legs.push((
                trees.domain_uncoupled()[0],
                block.shape()[1],
                raw_trees.domain_tree().is_dual()[0],
            ));
        }
        codomain_legs.sort_unstable();
        codomain_legs.dedup();
        domain_legs.sort_unstable();
        domain_legs.dedup();
        (codomain_legs, domain_legs, block_structure(tensor))
    };
    let expected_structure = structure(&source);
    let expected_vertex_structure = block_structure(&vertex_source);
    assert_eq!(
        expected_structure.0,
        [(Label::Vacuum, 1, false), (Label::X, 2, false)]
    );
    assert_eq!(
        expected_structure.1,
        [(Label::Vacuum, 1, true), (Label::X, 3, true)]
    );
    let expected = source.data().to_vec();
    let expected_vertex_data = vertex_source.data().to_vec();
    provider.algebra_queries.store(0, Ordering::Relaxed);
    provider.coefficient_queries.store(0, Ordering::Relaxed);

    let device = source.to_cuda().unwrap();
    let restored = device.to_host().unwrap();
    let vertex_device = vertex_source.to_cuda().unwrap();
    let vertex_restored = vertex_device.to_host().unwrap();

    assert!(std::ptr::eq(restored.provider(), provider.as_ref()));
    assert_eq!(restored.data(), expected);
    assert_eq!(structure(&restored), expected_structure);
    assert_eq!(vertex_restored.data(), expected_vertex_data);
    assert_eq!(block_structure(&vertex_restored), expected_vertex_structure);
    assert_eq!(provider.algebra_queries.load(Ordering::Relaxed), 0);
    assert_eq!(provider.coefficient_queries.load(Ordering::Relaxed), 0);
}

#[test]
fn checked_only_multiplicity_two_transforms_keep_the_source_authority() {
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let provider = Arc::new(CheckedOnlyToy::new(0));
    let leg = GradedSpace::try_new(Arc::clone(&provider), [(Label::X, 1)], false).unwrap();
    let source: TensorMap<_, f64> =
        TensorMap::from_block_fn(&runtime, [&leg, &leg, &leg], [], |trees, _| {
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
    let error = source.braid(&[1, 0, 2], &[], &[0, 1]).unwrap_err();
    assert!(matches!(error, GenericTensorError::Facade(_)));
    assert_eq!(provider.coefficient_queries.load(Ordering::Relaxed), 0);

    let permuted = source.permute(&[1, 0, 2], &[]).unwrap();
    assert!(std::ptr::eq(permuted.provider(), provider.as_ref()));
    let restored = permuted.permute(&[1, 0, 2], &[]).unwrap();
    assert_eq!(snapshot(&restored), source_snapshot);
    for (actual, expected) in restored.data().iter().zip(source.data()) {
        assert!((actual - expected).abs() <= 1e-12);
    }

    let braided = source.braid(&[1, 0, 2], &[], &[0, 1, 2]).unwrap();
    assert!(std::ptr::eq(braided.provider(), provider.as_ref()));

    provider.fail_algebra.store(true, Ordering::Relaxed);
    let error = source.permute(&[1, 0, 2], &[]).unwrap_err();
    assert!(matches!(
        error,
        GenericTensorError::Plan(tenet::typed::CheckedGenericPlanError::Provider(
            ToyError::Algebra
        ))
    ));
}

#[test]
fn checked_generic_reductions_cover_real_complex_dense_payloads() {
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let provider = Arc::new(CheckedOnlyToy::new(0));
    let leg = GradedSpace::try_new(Arc::clone(&provider), [(Label::X, 2)], false).unwrap();
    let source: TensorMap<_, f64> =
        TensorMap::from_block_fn(&runtime, [&leg], [&leg], |_, indices| {
            (indices.iter().sum::<usize>() + 1) as f64
        })
        .unwrap();
    let inner = source.inner(&source).unwrap();
    assert!(inner.is_finite());
    assert!((source.norm().unwrap() * source.norm().unwrap() - inner).abs() < 1e-12);
    assert!(source.tr().unwrap().is_finite());
    let complex = source.to_c64();
    assert!(complex.inner(&complex).unwrap().re.is_finite());
    assert!(complex.norm().unwrap().is_finite());
    assert!(complex.tr().unwrap().re.is_finite());
    assert!(provider.coefficient_queries.load(Ordering::Relaxed) > 0);
}

#[test]
fn checked_generic_host_add_scale_cover_real_and_complex_payloads() {
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let provider = Arc::new(CheckedOnlyToy::new(0));
    let leg = GradedSpace::try_new(Arc::clone(&provider), [(Label::X, 2)], false).unwrap();
    let source: TensorMap<_, f64> =
        TensorMap::from_block_fn(&runtime, [&leg], [&leg], |_, indices| {
            (indices.iter().sum::<usize>() + 1) as f64
        })
        .unwrap();
    let added = source.add(&source, 2.0, -1.0).unwrap();
    assert_eq!(added.data(), source.data());
    let scaled = source.scale(3.0);
    assert!(scaled
        .data()
        .iter()
        .zip(source.data())
        .all(|(a, b)| (*a - 3.0 * *b).abs() < 1e-12));

    let complex = source.to_c64();
    let added = complex
        .add(
            &complex,
            Complex64::new(2.0, 0.0),
            Complex64::new(-1.0, 0.0),
        )
        .unwrap();
    assert_eq!(added.data(), complex.data());
    let scaled = complex.scale(Complex64::new(0.5, -1.0));
    assert!(scaled
        .data()
        .iter()
        .zip(complex.data())
        .all(|(a, b)| (*a - *b * Complex64::new(0.5, -1.0)).norm() < 1e-12));
}

#[test]
fn checked_generic_add_rejects_runtime_before_layout_without_queries() {
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let foreign_runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let provider = Arc::new(CheckedOnlyToy::new(0));
    let narrow = GradedSpace::try_new(Arc::clone(&provider), [(Label::X, 1)], false).unwrap();
    let wide = GradedSpace::try_new(Arc::clone(&provider), [(Label::X, 2)], false).unwrap();
    let left: TensorMap<_, f64> =
        TensorMap::from_block_fn(&runtime, [&narrow], [&narrow], |_, _| 1.0).unwrap();
    let right: TensorMap<_, f64> =
        TensorMap::from_block_fn(&foreign_runtime, [&wide], [&wide], |_, _| 2.0).unwrap();
    reset_provider_queries(&provider);
    let before = left.data().to_vec();
    let error = left.add(&right, 1.0, 1.0).unwrap_err();
    assert!(matches!(
        error,
        GenericTensorError::Facade(tenet::prelude::Error::RuntimeMismatch)
    ));
    assert_eq!(left.data(), before.as_slice());
    assert_no_provider_queries(&provider);
}

#[test]
fn checked_generic_add_rejects_layout_mismatch_without_queries() {
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let provider = Arc::new(CheckedOnlyToy::new(0));
    let narrow = GradedSpace::try_new(Arc::clone(&provider), [(Label::X, 1)], false).unwrap();
    let wide = GradedSpace::try_new(Arc::clone(&provider), [(Label::X, 2)], false).unwrap();
    let left: TensorMap<_, f64> =
        TensorMap::from_block_fn(&runtime, [&narrow], [&narrow], |_, _| 1.0).unwrap();
    let right: TensorMap<_, f64> =
        TensorMap::from_block_fn(&runtime, [&wide], [&wide], |_, _| 2.0).unwrap();
    reset_provider_queries(&provider);
    let before = left.data().to_vec();
    let error = left.add(&right, 1.0, 1.0).unwrap_err();
    assert!(matches!(
        error,
        GenericTensorError::Facade(tenet::prelude::Error::InvalidArgument(_))
    ));
    assert_eq!(left.data(), before.as_slice());
    assert_no_provider_queries(&provider);
}

#[test]
fn checked_generic_add_assign_rejects_runtime_before_layout_and_preserves_receiver() {
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let foreign_runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let provider = Arc::new(CheckedOnlyToy::new(0));
    let narrow = GradedSpace::try_new(Arc::clone(&provider), [(Label::X, 1)], false).unwrap();
    let wide = GradedSpace::try_new(Arc::clone(&provider), [(Label::X, 2)], false).unwrap();
    let mut left: TensorMap<_, f64> =
        TensorMap::from_block_fn(&runtime, [&narrow], [&narrow], |_, _| 1.0).unwrap();
    let right: TensorMap<_, f64> =
        TensorMap::from_block_fn(&foreign_runtime, [&wide], [&wide], |_, _| 2.0).unwrap();
    let before_data = left.data().to_vec();
    let before_trees = (0..left.block_count())
        .map(|index| left.block_fusion_trees(index).unwrap())
        .collect::<Vec<_>>();
    reset_provider_queries(&provider);
    let error = left.add_assign(&right, 1.0, 1.0).unwrap_err();
    assert!(matches!(
        error,
        GenericTensorError::Facade(tenet::prelude::Error::RuntimeMismatch)
    ));
    assert_eq!(left.data(), before_data.as_slice());
    assert_eq!(
        (0..left.block_count())
            .map(|index| left.block_fusion_trees(index).unwrap())
            .collect::<Vec<_>>(),
        before_trees
    );
    assert_no_provider_queries(&provider);
}

#[test]
fn checked_generic_add_assign_rejects_layout_mismatch_and_preserves_receiver() {
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let provider = Arc::new(CheckedOnlyToy::new(0));
    let narrow = GradedSpace::try_new(Arc::clone(&provider), [(Label::X, 1)], false).unwrap();
    let wide = GradedSpace::try_new(Arc::clone(&provider), [(Label::X, 2)], false).unwrap();
    let mut left: TensorMap<_, f64> =
        TensorMap::from_block_fn(&runtime, [&narrow], [&narrow], |_, _| 1.0).unwrap();
    let right: TensorMap<_, f64> =
        TensorMap::from_block_fn(&runtime, [&wide], [&wide], |_, _| 2.0).unwrap();
    let before_data = left.data().to_vec();
    let before_trees = (0..left.block_count())
        .map(|index| left.block_fusion_trees(index).unwrap())
        .collect::<Vec<_>>();
    reset_provider_queries(&provider);
    let error = left.add_assign(&right, 1.0, 1.0).unwrap_err();
    assert!(matches!(
        error,
        GenericTensorError::Facade(tenet::prelude::Error::InvalidArgument(_))
    ));
    assert_eq!(left.data(), before_data.as_slice());
    assert_eq!(
        (0..left.block_count())
            .map(|index| left.block_fusion_trees(index).unwrap())
            .collect::<Vec<_>>(),
        before_trees
    );
    assert_no_provider_queries(&provider);
}

#[cfg(feature = "racah-generated")]
#[test]
fn sun_checked_generic_unit_insert_remove_preserves_authority_and_payload() {
    use tenet::prelude::GenericUnitTensorMapExt;
    use tenet::typed::SUNFusionRule;

    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    for n in [3, 4] {
        let provider = Arc::new(SUNFusionRule::new(n).unwrap());
        let label = if n == 3 { vec![1, 1] } else { vec![1, 0, 1] };
        let leg = GradedSpace::try_new(Arc::clone(&provider), [(label, 1)], false).unwrap();
        let source: TensorMap<_, f64> =
            TensorMap::from_block_fn(&runtime, [&leg], [&leg], |_, _| 2.0).unwrap();

        assert!(source.remove_unit(0).is_err());
        let inserted = source.insert_left_unit(0, false).unwrap();
        assert!(std::ptr::eq(inserted.provider(), provider.as_ref()));
        assert_eq!(inserted.data().as_ptr(), source.data().as_ptr());
        let removed = inserted.remove_unit(0).unwrap();
        assert!(std::ptr::eq(removed.provider(), provider.as_ref()));
        assert_eq!(removed.data().as_ptr(), source.data().as_ptr());
        assert_eq!(removed.data(), source.data());
    }
}

#[cfg(feature = "racah-generated")]
#[test]
fn sun_checked_generic_compact_qr_preserves_provider_and_reconstructs() {
    use tenet::typed::SUNFusionRule;

    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let provider = Arc::new(SUNFusionRule::new(3).unwrap());
    let leg = GradedSpace::try_new(Arc::clone(&provider), [(vec![1, 1], 1)], false).unwrap();
    let source: TensorMap<_, f64> =
        TensorMap::from_block_fn(&runtime, [&leg], [&leg], |trees, _| {
            trees.coupled().iter().sum::<i64>() as f64 + 1.0
        })
        .unwrap();

    let (q, r) = source.qr_compact().unwrap();
    let (orth_q, orth_r) = source.left_orth().unwrap();
    assert_eq!(orth_q.data(), q.data());
    assert_eq!(orth_r.data(), r.data());
    assert!(std::ptr::eq(orth_q.provider(), q.provider()));
    assert!(std::ptr::eq(orth_r.provider(), r.provider()));
    assert_eq!(orth_q.codomain(), q.codomain());
    assert_eq!(orth_q.domain(), q.domain());
    assert_eq!(orth_r.codomain(), r.codomain());
    assert_eq!(orth_r.domain(), r.domain());
    assert!(std::ptr::eq(q.provider(), provider.as_ref()));
    assert!(std::ptr::eq(r.provider(), provider.as_ref()));
    let rebuilt = q.compose(&r).unwrap();
    assert!(rebuilt
        .data()
        .iter()
        .zip(source.data())
        .all(|(actual, expected)| (actual - expected).abs() < 1.0e-10));

    let complex = source.to_c64();
    let (complex_q, complex_r) = complex.qr_compact().unwrap();
    let (complex_orth_q, complex_orth_r) = complex.left_orth().unwrap();
    assert_eq!(complex_orth_q.data(), complex_q.data());
    assert_eq!(complex_orth_r.data(), complex_r.data());
    assert!(std::ptr::eq(
        complex_orth_q.provider(),
        complex_q.provider()
    ));
    assert!(std::ptr::eq(
        complex_orth_r.provider(),
        complex_r.provider()
    ));
    assert_eq!(complex_orth_q.codomain(), complex_q.codomain());
    assert_eq!(complex_orth_q.domain(), complex_q.domain());
    assert_eq!(complex_orth_r.codomain(), complex_r.codomain());
    assert_eq!(complex_orth_r.domain(), complex_r.domain());
    assert!(std::ptr::eq(complex_q.provider(), provider.as_ref()));
    assert!(std::ptr::eq(complex_r.provider(), provider.as_ref()));
    let complex_rebuilt = complex_q.compose(&complex_r).unwrap();
    assert!(complex_rebuilt
        .data()
        .iter()
        .zip(complex.data())
        .all(|(actual, expected)| (*actual - *expected).norm() < 1.0e-10));
}

#[cfg(feature = "racah-generated")]
#[test]
fn sun_checked_generic_compact_svd_preserves_provider_and_reconstructs() {
    use tenet::typed::SUNFusionRule;

    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let provider = Arc::new(SUNFusionRule::new(3).unwrap());
    let leg = GradedSpace::try_new(Arc::clone(&provider), [(vec![1, 1], 1)], false).unwrap();
    let source: TensorMap<_, f64> =
        TensorMap::from_block_fn(&runtime, [&leg], [&leg], |trees, _| {
            trees.coupled().iter().sum::<i64>() as f64 + 1.0
        })
        .unwrap();

    let (u, s, vh) = source.svd_compact().unwrap();
    assert!(std::ptr::eq(u.provider(), provider.as_ref()));
    assert!(std::ptr::eq(s.provider(), provider.as_ref()));
    assert!(std::ptr::eq(vh.provider(), provider.as_ref()));
    let rebuilt = u.compose(&s).unwrap().compose(&vh).unwrap();
    assert!(rebuilt
        .data()
        .iter()
        .zip(source.data())
        .all(|(actual, expected)| (actual - expected).abs() < 1.0e-10));

    let complex = source.to_c64();
    let (complex_u, complex_s, complex_vh) = complex.svd_compact().unwrap();
    assert!(std::ptr::eq(complex_u.provider(), provider.as_ref()));
    assert!(std::ptr::eq(complex_s.provider(), provider.as_ref()));
    assert!(std::ptr::eq(complex_vh.provider(), provider.as_ref()));
    let complex_rebuilt = complex_u
        .compose(&complex_s)
        .unwrap()
        .compose(&complex_vh)
        .unwrap();
    assert!(complex_rebuilt
        .data()
        .iter()
        .zip(complex.data())
        .all(|(actual, expected)| (*actual - *expected).norm() < 1.0e-10));
}

#[cfg(feature = "racah-generated")]
#[test]
fn sun_checked_generic_full_svd_preserves_provider_reconstructs_and_rejects_lazy() {
    use tenet::typed::SUNFusionRule;

    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let provider = Arc::new(SUNFusionRule::new(3).unwrap());
    let leg = GradedSpace::try_new(Arc::clone(&provider), [(vec![1, 1], 1)], false).unwrap();
    let source: TensorMap<_, f64> =
        TensorMap::from_block_fn(&runtime, [&leg], [&leg], |trees, _| {
            trees.coupled().iter().sum::<i64>() as f64 + 1.0
        })
        .unwrap();

    let (u, s, vh) = source.svd_full().unwrap();
    assert!(std::ptr::eq(u.provider(), provider.as_ref()));
    assert!(std::ptr::eq(s.provider(), provider.as_ref()));
    assert!(std::ptr::eq(vh.provider(), provider.as_ref()));
    let rebuilt = u.compose(&s).unwrap().compose(&vh).unwrap();
    assert!(rebuilt
        .data()
        .iter()
        .zip(source.data())
        .all(|(actual, expected)| (actual - expected).abs() < 1.0e-10));

    let complex = source.to_c64();
    let (complex_u, complex_s, complex_vh) = complex.svd_full().unwrap();
    let complex_rebuilt = complex_u
        .compose(&complex_s)
        .unwrap()
        .compose(&complex_vh)
        .unwrap();
    assert!(complex_rebuilt
        .data()
        .iter()
        .zip(complex.data())
        .all(|(actual, expected)| (*actual - *expected).norm() < 1.0e-10));

    let lazy = source.adjoint().unwrap();
    assert!(matches!(
        lazy.svd_full(),
        Err(GenericTensorError::Facade(_))
    ));
}

#[cfg(feature = "racah-generated")]
#[test]
fn sun_checked_generic_compact_lq_preserves_provider_and_reconstructs() {
    use tenet::typed::SUNFusionRule;

    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let provider = Arc::new(SUNFusionRule::new(3).unwrap());
    let leg = GradedSpace::try_new(Arc::clone(&provider), [(vec![1, 1], 1)], false).unwrap();
    let source: TensorMap<_, f64> =
        TensorMap::from_block_fn(&runtime, [&leg], [&leg], |trees, _| {
            trees.coupled().iter().sum::<i64>() as f64 + 1.0
        })
        .unwrap();

    let (l, q) = source.lq_compact().unwrap();
    let (orth_l, orth_q) = source.right_orth().unwrap();
    assert_eq!(orth_l.data(), l.data());
    assert_eq!(orth_q.data(), q.data());
    assert!(std::ptr::eq(orth_l.provider(), l.provider()));
    assert!(std::ptr::eq(orth_q.provider(), q.provider()));
    assert_eq!(orth_l.codomain(), l.codomain());
    assert_eq!(orth_l.domain(), l.domain());
    assert_eq!(orth_q.codomain(), q.codomain());
    assert_eq!(orth_q.domain(), q.domain());
    assert!(std::ptr::eq(l.provider(), provider.as_ref()));
    assert!(std::ptr::eq(q.provider(), provider.as_ref()));
    let rebuilt = l.compose(&q).unwrap();
    assert!(rebuilt
        .data()
        .iter()
        .zip(source.data())
        .all(|(actual, expected)| (actual - expected).abs() < 1.0e-10));

    let complex = source.to_c64();
    let (complex_l, complex_q) = complex.lq_compact().unwrap();
    let (complex_orth_l, complex_orth_q) = complex.right_orth().unwrap();
    assert_eq!(complex_orth_l.data(), complex_l.data());
    assert_eq!(complex_orth_q.data(), complex_q.data());
    assert!(std::ptr::eq(
        complex_orth_l.provider(),
        complex_l.provider()
    ));
    assert!(std::ptr::eq(
        complex_orth_q.provider(),
        complex_q.provider()
    ));
    assert_eq!(complex_orth_l.codomain(), complex_l.codomain());
    assert_eq!(complex_orth_l.domain(), complex_l.domain());
    assert_eq!(complex_orth_q.codomain(), complex_q.codomain());
    assert_eq!(complex_orth_q.domain(), complex_q.domain());
    assert!(std::ptr::eq(complex_l.provider(), provider.as_ref()));
    assert!(std::ptr::eq(complex_q.provider(), provider.as_ref()));
    let complex_rebuilt = complex_l.compose(&complex_q).unwrap();
    assert!(complex_rebuilt
        .data()
        .iter()
        .zip(complex.data())
        .all(|(actual, expected)| (*actual - *expected).norm() < 1.0e-10));
}

#[cfg(feature = "racah-generated")]
#[test]
fn sun_checked_generic_orth_aliases_reconstruct_multiplicity_fixture() {
    use tenet::typed::SUNFusionRule;

    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    macro_rules! assert_aliases {
        ($source:expr, $close:expr) => {{
            let source = $source;
            let close = $close;
            let (q, r) = source.qr_compact().unwrap();
            let (orth_q, orth_r) = source.left_orth().unwrap();
            for (actual, expected) in orth_q
                .compose(&orth_r)
                .unwrap()
                .data()
                .iter()
                .zip(source.data())
            {
                assert!(close(*actual, *expected) < 1.0e-10);
            }
            for (actual, expected) in q.compose(&r).unwrap().data().iter().zip(source.data()) {
                assert!(close(*actual, *expected) < 1.0e-10);
            }
            for (alias, lower) in [(&orth_q, &q), (&orth_r, &r)] {
                assert!(std::ptr::eq(alias.provider(), lower.provider()));
                assert_eq!(alias.codomain(), lower.codomain());
                assert_eq!(alias.domain(), lower.domain());
                assert!(alias.runtime().shares_state_with(lower.runtime()));
            }

            let (l, q) = source.lq_compact().unwrap();
            let (orth_l, orth_q) = source.right_orth().unwrap();
            for (actual, expected) in orth_l
                .compose(&orth_q)
                .unwrap()
                .data()
                .iter()
                .zip(source.data())
            {
                assert!(close(*actual, *expected) < 1.0e-10);
            }
            for (actual, expected) in l.compose(&q).unwrap().data().iter().zip(source.data()) {
                assert!(close(*actual, *expected) < 1.0e-10);
            }
            for (alias, lower) in [(&orth_l, &l), (&orth_q, &q)] {
                assert!(std::ptr::eq(alias.provider(), lower.provider()));
                assert_eq!(alias.codomain(), lower.codomain());
                assert_eq!(alias.domain(), lower.domain());
                assert!(alias.runtime().shares_state_with(lower.runtime()));
            }
        }};
    }

    for n in [3, 4] {
        let provider = Arc::new(SUNFusionRule::new(n).unwrap());
        let label = if n == 3 { vec![1, 1] } else { vec![1, 0, 1] };
        let leg = GradedSpace::try_new(Arc::clone(&provider), [(label, 1)], false).unwrap();
        let source: TensorMap<_, f64> =
            TensorMap::from_block_fn(&runtime, [&leg, &leg], [&leg], |trees, _| {
                trees.codomain_vertices()[0].get() as f64
            })
            .unwrap();
        assert_eq!(source.block_count(), 2);
        assert_aliases!(&source, |actual: f64, expected: f64| {
            (actual - expected).abs()
        });
        assert_aliases!(source.to_c64(), |actual: Complex64, expected: Complex64| {
            (actual - expected).norm()
        });
    }
}

#[cfg(feature = "racah-generated")]
#[test]
fn sun_checked_generic_full_qr_preserves_provider_and_reconstructs() {
    use tenet::typed::SUNFusionRule;

    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let provider = Arc::new(SUNFusionRule::new(3).unwrap());
    let leg = GradedSpace::try_new(Arc::clone(&provider), [(vec![1, 1], 1)], false).unwrap();
    let source: TensorMap<_, f64> =
        TensorMap::from_block_fn(&runtime, [&leg], [&leg], |trees, _| {
            trees.coupled().iter().sum::<i64>() as f64 + 1.0
        })
        .unwrap();

    let (q, r) = source.qr_full().unwrap();
    assert!(std::ptr::eq(q.provider(), provider.as_ref()));
    assert!(std::ptr::eq(r.provider(), provider.as_ref()));
    let rebuilt = q.compose(&r).unwrap();
    assert!(rebuilt
        .data()
        .iter()
        .zip(source.data())
        .all(|(actual, expected)| (actual - expected).abs() < 1.0e-10));

    let complex = source.to_c64();
    let (complex_q, complex_r) = complex.qr_full().unwrap();
    assert!(std::ptr::eq(complex_q.provider(), provider.as_ref()));
    assert!(std::ptr::eq(complex_r.provider(), provider.as_ref()));
    let complex_rebuilt = complex_q.compose(&complex_r).unwrap();
    assert!(complex_rebuilt
        .data()
        .iter()
        .zip(complex.data())
        .all(|(actual, expected)| (*actual - *expected).norm() < 1.0e-10));
}

#[cfg(feature = "racah-generated")]
#[test]
fn sun_checked_generic_svd_vals_matches_compact_spectrum() {
    use tenet::typed::SUNFusionRule;

    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let provider = Arc::new(SUNFusionRule::new(3).unwrap());
    let leg = GradedSpace::try_new(Arc::clone(&provider), [(vec![1, 1], 1)], false).unwrap();
    let source: TensorMap<_, f64> =
        TensorMap::from_block_fn(&runtime, [&leg], [&leg], |trees, _| {
            trees.coupled().iter().sum::<i64>() as f64 + 1.0
        })
        .unwrap();
    let spectra = source.svd_vals().unwrap();
    assert!(!spectra.is_empty());
    assert!(spectra.iter().all(|spectrum| spectrum
        .values
        .iter()
        .all(|value| value.is_finite() && *value >= 0.0)));

    let complex = source.to_c64();
    let complex_spectra = complex.svd_vals().unwrap();
    assert_eq!(complex_spectra, spectra);
}

#[test]
fn checked_generic_eigh_vals_preserves_spectrum_and_dtype() {
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let provider = Arc::new(CheckedOnlyToy::new(0));
    let leg = GradedSpace::try_new(Arc::clone(&provider), [(Label::X, 1)], false).unwrap();
    let source: TensorMap<_, f64> =
        TensorMap::from_block_fn(&runtime, [&leg], [&leg], |_, _| 2.5).unwrap();

    let spectra = source.eigh_vals().unwrap();
    assert_eq!(spectra.len(), 1);
    assert_eq!(spectra[0].sector, Label::X);
    assert_eq!(spectra[0].values, vec![2.5]);

    let complex = source.to_c64();
    assert_eq!(complex.eigh_vals().unwrap(), spectra);
}

#[test]
fn checked_generic_eig_vals_preserves_spectrum_and_dtype() {
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let provider = Arc::new(CheckedOnlyToy::new(0));
    let leg = GradedSpace::try_new(Arc::clone(&provider), [(Label::X, 2)], false).unwrap();
    let source: TensorMap<_, f64> =
        TensorMap::from_block_fn(&runtime, [&leg], [&leg], |_, indices| {
            if indices[0] == indices[1] {
                (indices[0] + 2) as f64
            } else {
                0.0
            }
        })
        .unwrap();

    let spectra = source.eig_vals().unwrap();
    assert_eq!(spectra.len(), 1);
    assert_eq!(spectra[0].sector, Label::X);
    assert_eq!(
        spectra[0].values,
        vec![Complex64::new(3.0, 0.0), Complex64::new(2.0, 0.0)]
    );

    let complex = source.to_c64();
    assert_eq!(complex.eig_vals().unwrap(), spectra);
}

#[test]
fn checked_generic_svd_trunc_reconstructs_and_preserves_provider() {
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let provider = Arc::new(CheckedOnlyToy::new(0));
    let leg = GradedSpace::try_new(Arc::clone(&provider), [(Label::X, 2)], false).unwrap();
    let source: TensorMap<_, f64> =
        TensorMap::from_block_fn(&runtime, [&leg], [&leg], |_, indices| {
            if indices[0] == indices[1] {
                (indices[0] + 2) as f64
            } else {
                0.0
            }
        })
        .unwrap();
    let result = source.svd_trunc(&Truncation::rank(1)).unwrap();
    assert!(std::ptr::eq(result.u.provider(), provider.as_ref()));
    assert!(std::ptr::eq(result.s.provider(), provider.as_ref()));
    assert!(std::ptr::eq(result.vh.provider(), provider.as_ref()));
    let rebuilt = result
        .u
        .compose(&result.s)
        .unwrap()
        .compose(&result.vh)
        .unwrap();
    assert!(rebuilt.data().iter().all(|value| value.is_finite()));
    assert!(result.singular_values.iter().all(|spectrum| {
        spectrum.values.len() <= 2 && spectrum.values.iter().all(|value| value.is_finite())
    }));
}

#[test]
fn checked_generic_lazy_adjoint_preserves_provider_and_reductions() {
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let provider = Arc::new(CheckedOnlyToy::new(0));
    let leg = GradedSpace::try_new(Arc::clone(&provider), [(Label::X, 2)], false).unwrap();
    let source: TensorMap<_, f64> =
        TensorMap::from_block_fn(&runtime, [&leg], [&leg], |_, indices| {
            (indices.iter().sum::<usize>() + 1) as f64
        })
        .unwrap();

    let adjoint = source.adjoint().unwrap();
    assert!(std::ptr::eq(adjoint.provider(), provider.as_ref()));
    assert!((adjoint.norm().unwrap() - source.norm().unwrap()).abs() < 1.0e-12);
    assert!((adjoint.tr().unwrap() - source.tr().unwrap()).abs() < 1.0e-12);

    let complex = source.to_c64();
    let complex_adjoint = complex.adjoint().unwrap();
    assert!(std::ptr::eq(complex_adjoint.provider(), provider.as_ref()));
    assert!((complex_adjoint.tr().unwrap() - complex.tr().unwrap().conj()).norm() < 1.0e-12);
}

#[test]
fn checked_generic_reduction_dimension_failure_is_typed_and_nonpublishing() {
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let provider = Arc::new(CheckedOnlyToy::new(0));
    let leg = GradedSpace::try_new(Arc::clone(&provider), [(Label::X, 1)], false).unwrap();
    let source: TensorMap<_, f64> =
        TensorMap::from_block_fn(&runtime, [&leg], [&leg], |_, _| 2.0).unwrap();
    let before = source.data().to_vec();
    provider.fail_algebra.store(true, Ordering::Relaxed);
    let error = source.norm().unwrap_err();
    assert!(matches!(
        error,
        GenericTensorError::Structure(CheckedGenericStructureError::Provider(ToyError::Algebra))
    ));
    assert_eq!(source.data(), before.as_slice());
}

#[test]
fn checked_generic_compact_qr_failure_is_typed_and_nonpublishing() {
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let provider = Arc::new(CheckedOnlyToy::new_product_probe(0));
    let leg = GradedSpace::try_new(Arc::clone(&provider), [(Label::X, 1)], false).unwrap();
    let source: TensorMap<_, f64> =
        TensorMap::from_block_fn(&runtime, [&leg, &leg], [&leg], |_, _| 2.0).unwrap();
    let before = source.data().to_vec();
    provider.fail_algebra.store(true, Ordering::Relaxed);
    let error = source.qr_compact().unwrap_err();
    assert!(matches!(
        error,
        GenericTensorError::Plan(tenet::typed::CheckedGenericPlanError::Provider(
            ToyError::Algebra
        ))
    ));
    assert_eq!(source.data(), before.as_slice());
    assert!(matches!(
        source.left_orth(),
        Err(GenericTensorError::Plan(
            tenet::typed::CheckedGenericPlanError::Provider(ToyError::Algebra)
        ))
    ));
    assert_eq!(source.data(), before.as_slice());
}

#[test]
fn checked_generic_compact_svd_failure_is_typed_and_nonpublishing() {
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let provider = Arc::new(CheckedOnlyToy::new_product_probe(0));
    let leg = GradedSpace::try_new(Arc::clone(&provider), [(Label::X, 1)], false).unwrap();
    let source: TensorMap<_, f64> =
        TensorMap::from_block_fn(&runtime, [&leg, &leg], [&leg], |_, _| 2.0).unwrap();
    let before = source.data().to_vec();
    provider.fail_algebra.store(true, Ordering::Relaxed);
    let error = source.svd_compact().unwrap_err();
    assert!(matches!(
        error,
        GenericTensorError::Plan(tenet::typed::CheckedGenericPlanError::Provider(
            ToyError::Algebra
        ))
    ));
    assert_eq!(source.data(), before.as_slice());
}

#[test]
fn checked_generic_compact_lq_failure_is_typed_and_nonpublishing() {
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let provider = Arc::new(CheckedOnlyToy::new_product_probe(0));
    let leg = GradedSpace::try_new(Arc::clone(&provider), [(Label::X, 1)], false).unwrap();
    let source: TensorMap<_, f64> =
        TensorMap::from_block_fn(&runtime, [&leg, &leg], [&leg], |_, _| 2.0).unwrap();
    let before = source.data().to_vec();
    provider.fail_algebra.store(true, Ordering::Relaxed);
    let error = source.lq_compact().unwrap_err();
    assert!(matches!(
        error,
        GenericTensorError::Plan(tenet::typed::CheckedGenericPlanError::Provider(
            ToyError::Algebra
        ))
    ));
    assert_eq!(source.data(), before.as_slice());
    assert!(matches!(
        source.right_orth(),
        Err(GenericTensorError::Plan(
            tenet::typed::CheckedGenericPlanError::Provider(ToyError::Algebra)
        ))
    ));
    assert_eq!(source.data(), before.as_slice());
}

#[test]
fn checked_generic_full_qr_failure_is_typed_and_nonpublishing() {
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let provider = Arc::new(CheckedOnlyToy::new_product_probe(0));
    let leg = GradedSpace::try_new(Arc::clone(&provider), [(Label::X, 1)], false).unwrap();
    let source: TensorMap<_, f64> =
        TensorMap::from_block_fn(&runtime, [&leg, &leg], [&leg], |_, _| 2.0).unwrap();
    let before = source.data().to_vec();
    provider.fail_algebra.store(true, Ordering::Relaxed);
    let error = source.qr_full().unwrap_err();
    assert!(matches!(
        error,
        GenericTensorError::Plan(tenet::typed::CheckedGenericPlanError::Provider(
            ToyError::Algebra
        ))
    ));
    assert_eq!(source.data(), before.as_slice());
}

#[test]
fn checked_generic_full_lq_reconstructs_and_preserves_provider() {
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let provider = Arc::new(CheckedOnlyToy::new_product_probe(0));
    let leg = GradedSpace::try_new(Arc::clone(&provider), [(Label::X, 1)], false).unwrap();
    let source: TensorMap<_, f64> =
        TensorMap::from_block_fn(&runtime, [&leg, &leg], [&leg], |_, _| 2.0).unwrap();
    let (l, q) = source.lq_full().unwrap();
    assert!(std::ptr::eq(l.provider(), provider.as_ref()));
    assert!(std::ptr::eq(q.provider(), provider.as_ref()));
    let rebuilt = l.compose(&q).unwrap();
    assert_eq!(rebuilt.data().len(), source.data().len());
    assert!(rebuilt
        .data()
        .iter()
        .zip(source.data())
        .all(|(actual, expected)| (actual - expected).abs() < 1e-12));
}

#[test]
fn checked_generic_full_lq_supports_complex_scalars() {
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let provider = Arc::new(CheckedOnlyToy::new_product_probe(0));
    let leg = GradedSpace::try_new(Arc::clone(&provider), [(Label::X, 1)], false).unwrap();
    let source: TensorMap<_, Complex64> =
        TensorMap::from_block_fn(&runtime, [&leg, &leg], [&leg], |_, _| {
            Complex64::new(2.0, 1.0)
        })
        .unwrap();
    let (l, q) = source.lq_full().unwrap();
    let rebuilt = l.compose(&q).unwrap();
    assert!((rebuilt.norm().unwrap() - source.norm().unwrap()).abs() < 1e-12);
}

#[test]
fn checked_generic_full_lq_failure_is_typed_and_nonpublishing() {
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let provider = Arc::new(CheckedOnlyToy::new_product_probe(0));
    let leg = GradedSpace::try_new(Arc::clone(&provider), [(Label::X, 1)], false).unwrap();
    let source: TensorMap<_, f64> =
        TensorMap::from_block_fn(&runtime, [&leg, &leg], [&leg], |_, _| 2.0).unwrap();
    let before = source.data().to_vec();
    provider.fail_algebra.store(true, Ordering::Relaxed);
    let error = source.lq_full().unwrap_err();
    assert!(matches!(
        error,
        GenericTensorError::Plan(tenet::typed::CheckedGenericPlanError::Provider(
            ToyError::Algebra
        ))
    ));
    assert_eq!(source.data(), before.as_slice());
}

#[test]
fn checked_only_contract_and_compose_keep_left_authority() {
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let left_provider = Arc::new(CheckedOnlyToy::new(0));
    let right_provider = Arc::new(CheckedOnlyToy::new(0));
    let left_leg =
        GradedSpace::try_new(Arc::clone(&left_provider), [(Label::X, 1)], false).unwrap();
    let right_leg =
        GradedSpace::try_new(Arc::clone(&right_provider), [(Label::X, 1)], false).unwrap();
    let source: TensorMap<_, f64> =
        TensorMap::from_block_fn(&runtime, [&left_leg], [&left_leg], |_, _| 1.0).unwrap();
    let nontrivial: TensorMap<_, f64> =
        TensorMap::from_block_fn(&runtime, [&left_leg, &left_leg], [&left_leg], |_, _| 1.0)
            .unwrap();
    let identity: TensorMap<_, f64> =
        TensorMap::from_block_fn(&runtime, [&right_leg], [&right_leg], |_, _| 1.0).unwrap();
    left_provider.r_queries.store(0, Ordering::Relaxed);

    for output in [
        source.contract(&identity, &[1], &[0], &[0, 1]).unwrap(),
        source
            .contract_ordered(&identity, &[1], &[0], &[0, 1])
            .unwrap(),
        source.compose(&identity).unwrap(),
    ] {
        assert!(std::ptr::eq(output.provider(), left_provider.as_ref()));
        assert_eq!(output.data(), source.data());
        for index in 0..source.block_count() {
            assert_eq!(
                output.block_fusion_trees(index).unwrap(),
                source.block_fusion_trees(index).unwrap()
            );
        }
    }
    assert_eq!(left_provider.r_queries.load(Ordering::Relaxed), 0);

    let other_runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let foreign_runtime_identity: TensorMap<_, f64> =
        TensorMap::from_block_fn(&other_runtime, [&right_leg], [&right_leg], |_, _| 1.0).unwrap();
    left_provider.algebra_queries.store(0, Ordering::Relaxed);
    right_provider.algebra_queries.store(0, Ordering::Relaxed);
    assert!(matches!(
        source.contract(&foreign_runtime_identity, &[1], &[0], &[0, 1]),
        Err(GenericTensorError::Facade(
            tenet::prelude::Error::RuntimeMismatch
        ))
    ));
    assert_eq!(left_provider.algebra_queries.load(Ordering::Relaxed), 0);
    assert_eq!(right_provider.algebra_queries.load(Ordering::Relaxed), 0);

    let wrong_provider = Arc::new(CheckedOnlyToy::new(1));
    let wrong_leg =
        GradedSpace::try_new(Arc::clone(&wrong_provider), [(Label::X, 1)], false).unwrap();
    let wrong_identity: TensorMap<_, f64> =
        TensorMap::from_block_fn(&runtime, [&wrong_leg], [&wrong_leg], |_, _| 1.0).unwrap();
    left_provider.algebra_queries.store(0, Ordering::Relaxed);
    wrong_provider.algebra_queries.store(0, Ordering::Relaxed);
    assert!(source
        .contract(&wrong_identity, &[1], &[0], &[0, 1])
        .is_err());
    assert_eq!(left_provider.algebra_queries.load(Ordering::Relaxed), 0);
    assert_eq!(wrong_provider.algebra_queries.load(Ordering::Relaxed), 0);

    left_provider.fail_algebra.store(true, Ordering::Relaxed);
    let error = nontrivial
        .contract(&identity, &[2], &[0], &[0, 1, 2])
        .unwrap_err();
    assert!(matches!(
        error,
        GenericTensorError::Plan(tenet::typed::CheckedGenericPlanError::Provider(
            ToyError::Algebra
        ))
    ));
}

#[test]
fn checked_only_identity_transforms_make_no_provider_queries() {
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let provider = Arc::new(CheckedOnlyToy::new(0));
    let leg = GradedSpace::try_new(Arc::clone(&provider), [(Label::X, 1)], false).unwrap();
    let source: TensorMap<_, f64> =
        TensorMap::from_block_fn(&runtime, [&leg, &leg], [&leg], |_, indices| {
            indices.iter().sum::<usize>() as f64 + 1.0
        })
        .unwrap();

    for counter in [
        &provider.identity_queries,
        &provider.style_queries,
        &provider.algebra_queries,
        &provider.coefficient_queries,
        &provider.f_queries,
        &provider.r_queries,
        &provider.postcommit_queries,
    ] {
        counter.store(0, Ordering::Relaxed);
    }

    for output in [
        source.permute(&[0, 1], &[2]).unwrap(),
        source.braid(&[0, 1], &[2], &[2, 1, 0]).unwrap(),
        source.transpose_axes(&[0, 1], &[2]).unwrap(),
        source.repartition(2).unwrap(),
    ] {
        assert!(std::ptr::eq(output.provider(), provider.as_ref()));
        assert_eq!(output.data().as_ptr(), source.data().as_ptr());
    }
    for counter in [
        &provider.identity_queries,
        &provider.style_queries,
        &provider.algebra_queries,
        &provider.coefficient_queries,
        &provider.f_queries,
        &provider.r_queries,
        &provider.postcommit_queries,
    ] {
        assert_eq!(counter.load(Ordering::Relaxed), 0);
    }
}

#[test]
fn checked_only_otimes_preserves_typed_late_f_failures() {
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let provider = Arc::new(CheckedOnlyToy::new_product_probe(0));
    let leg = GradedSpace::try_new(Arc::clone(&provider), [(Label::X, 1)], false).unwrap();
    let source: TensorMap<_, f64> =
        TensorMap::from_block_fn(&runtime, [&leg, &leg], [&leg, &leg], |trees, _| {
            trees.codomain_vertices()[0].get() as f64
                - 2.0 * trees.domain_vertices()[0].get() as f64
        })
        .unwrap();

    provider.f_queries.store(0, Ordering::Relaxed);
    source.otimes(&source).unwrap();
    let final_f_query = provider.f_queries.load(Ordering::Relaxed);
    assert!(final_f_query > 1);
    provider.f_queries.store(0, Ordering::Relaxed);
    provider.reset_commit_spy();
    provider
        .fail_f_on_query
        .store(final_f_query, Ordering::Relaxed);
    let error = source.otimes(&source).unwrap_err();
    assert!(matches!(
        error,
        GenericTensorError::TensorProduct(CheckedGenericTensorProductError::Provider(
            ToyError::Algebra
        ))
    ));
    assert_eq!(provider.f_queries.load(Ordering::Relaxed), final_f_query);
    assert_eq!(provider.commit_count.load(Ordering::Relaxed), 0);
    provider.fail_f_on_query.store(0, Ordering::Relaxed);

    provider.f_queries.store(0, Ordering::Relaxed);
    provider.reset_commit_spy();
    provider.malformed_f.store(true, Ordering::Relaxed);
    let error = source.otimes(&source).unwrap_err();
    assert!(matches!(
        error,
        GenericTensorError::TensorProduct(CheckedGenericTensorProductError::SymbolShape {
            symbol: "F",
            ..
        })
    ));
    assert_eq!(provider.commit_count.load(Ordering::Relaxed), 0);
    assert_eq!(provider.r_queries.load(Ordering::Relaxed), 0);
}

#[test]
fn checked_only_otimes_rejects_runtime_identity_and_style_before_algebra() {
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let other_runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let first = Arc::new(CheckedOnlyToy::new(0));
    let mismatched = Arc::new(CheckedOnlyToy::new(1));
    let first_leg = GradedSpace::try_new(Arc::clone(&first), [(Label::X, 1)], false).unwrap();
    let mismatched_leg =
        GradedSpace::try_new(Arc::clone(&mismatched), [(Label::X, 1)], false).unwrap();
    let lhs: TensorMap<_, f64> =
        TensorMap::from_block_fn(&runtime, [&first_leg, &first_leg], [&first_leg], |_, _| 1.0)
            .unwrap();
    let wrong_runtime: TensorMap<_, f64> = TensorMap::from_block_fn(
        &other_runtime,
        [&first_leg, &first_leg],
        [&first_leg],
        |_, _| 1.0,
    )
    .unwrap();
    let wrong_identity: TensorMap<_, f64> = TensorMap::from_block_fn(
        &runtime,
        [&mismatched_leg, &mismatched_leg],
        [&mismatched_leg],
        |_, _| 1.0,
    )
    .unwrap();
    first.algebra_queries.store(0, Ordering::Relaxed);
    first.coefficient_queries.store(0, Ordering::Relaxed);
    mismatched.algebra_queries.store(0, Ordering::Relaxed);
    mismatched.coefficient_queries.store(0, Ordering::Relaxed);

    assert!(matches!(
        lhs.otimes(&wrong_runtime),
        Err(GenericTensorError::Facade(_))
    ));
    assert!(matches!(
        lhs.otimes(&wrong_identity),
        Err(GenericTensorError::TensorProduct(
            CheckedGenericTensorProductError::Core(_)
        ))
    ));
    first.invalid_style.store(true, Ordering::Relaxed);
    assert!(matches!(
        lhs.otimes(&lhs),
        Err(GenericTensorError::TensorProduct(
            CheckedGenericTensorProductError::Core(_)
        ))
    ));
    assert_eq!(first.algebra_queries.load(Ordering::Relaxed), 0);
    assert_eq!(first.coefficient_queries.load(Ordering::Relaxed), 0);
    assert_eq!(mismatched.algebra_queries.load(Ordering::Relaxed), 0);
    assert_eq!(mismatched.coefficient_queries.load(Ordering::Relaxed), 0);
}

#[test]
fn checked_only_otimes_matches_fixed_heterogeneous_nonunit_oracle() {
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let first = Arc::new(CheckedOnlyToy::new_product_probe(9));
    let second = Arc::new(CheckedOnlyToy::new_product_probe(9));
    let x2 = GradedSpace::try_new(Arc::clone(&first), [(Label::X, 2)], false).unwrap();
    let x1 = GradedSpace::try_new(Arc::clone(&first), [(Label::X, 1)], false).unwrap();
    let y3 = GradedSpace::try_new(Arc::clone(&second), [(Label::One, 3)], false).unwrap();
    let y1 = GradedSpace::try_new(Arc::clone(&second), [(Label::One, 1)], false).unwrap();
    let rhs_x1 = GradedSpace::try_new(Arc::clone(&second), [(Label::X, 1)], false).unwrap();
    let lhs: TensorMap<_, f64> =
        TensorMap::from_block_fn(&runtime, [&x2, &x1], [&x1, &x1], |trees, index| {
            100.0 * trees.codomain_vertices()[0].get() as f64
                + 10.0 * trees.domain_vertices()[0].get() as f64
                + index[0] as f64
                + 1.0
        })
        .unwrap();
    let rhs: TensorMap<_, f64> =
        TensorMap::from_block_fn(&runtime, [&y3, &rhs_x1], [&y1, &rhs_x1], |_, index| {
            index[0] as f64 + 1.0
        })
        .unwrap();
    for provider in [&first, &second] {
        provider.identity_queries.store(0, Ordering::Relaxed);
        provider.style_queries.store(0, Ordering::Relaxed);
        provider.algebra_queries.store(0, Ordering::Relaxed);
        provider.coefficient_queries.store(0, Ordering::Relaxed);
        provider.r_queries.store(0, Ordering::Relaxed);
        provider.f_queries.store(0, Ordering::Relaxed);
        provider.reset_commit_spy();
    }
    let output = lhs.otimes(&rhs).unwrap();
    assert!(std::ptr::eq(output.provider(), first.as_ref()));
    assert_eq!(first.identity_queries.load(Ordering::Relaxed), 3);
    assert_eq!(second.identity_queries.load(Ordering::Relaxed), 1);
    assert_eq!(first.commit_count.load(Ordering::Relaxed), 1);
    assert_eq!(second.commit_count.load(Ordering::Relaxed), 0);
    assert_eq!(first.postcommit_queries.load(Ordering::Relaxed), 0);
    assert_eq!(second.postcommit_queries.load(Ordering::Relaxed), 0);
    assert!(first.algebra_queries.load(Ordering::Relaxed) > 0);
    assert!(first.f_queries.load(Ordering::Relaxed) > 0);
    assert_eq!(first.r_queries.load(Ordering::Relaxed), 0);
    assert_eq!(second.algebra_queries.load(Ordering::Relaxed), 0);
    assert_eq!(second.coefficient_queries.load(Ordering::Relaxed), 0);
    assert_eq!(second.f_queries.load(Ordering::Relaxed), 0);
    assert_eq!(second.r_queries.load(Ordering::Relaxed), 0);

    const EXPECTED_KEYS: [([usize; 3], [usize; 3]); 16] = [
        ([1, 1, 1], [1, 1, 1]),
        ([2, 1, 1], [1, 1, 1]),
        ([1, 1, 2], [1, 1, 1]),
        ([2, 1, 2], [1, 1, 1]),
        ([1, 1, 1], [2, 1, 1]),
        ([2, 1, 1], [2, 1, 1]),
        ([1, 1, 2], [2, 1, 1]),
        ([2, 1, 2], [2, 1, 1]),
        ([1, 1, 1], [1, 1, 2]),
        ([2, 1, 1], [1, 1, 2]),
        ([1, 1, 2], [1, 1, 2]),
        ([2, 1, 2], [1, 1, 2]),
        ([1, 1, 1], [2, 1, 2]),
        ([2, 1, 1], [2, 1, 2]),
        ([1, 1, 2], [2, 1, 2]),
        ([2, 1, 2], [2, 1, 2]),
    ];
    let keys = (0..output.block_count())
        .map(|index| {
            let trees = output.block_fusion_trees(index).unwrap();
            assert_eq!(
                trees.codomain_uncoupled(),
                &[Label::X, Label::X, Label::One, Label::X]
            );
            assert_eq!(
                trees.domain_uncoupled(),
                &[Label::X, Label::X, Label::One, Label::X]
            );
            assert_eq!(
                output.block(index).unwrap().shape(),
                &[2, 1, 3, 1, 1, 1, 1, 1]
            );
            (
                trees
                    .codomain_vertices()
                    .iter()
                    .map(|vertex| vertex.get())
                    .collect::<Vec<_>>()
                    .try_into()
                    .unwrap(),
                trees
                    .domain_vertices()
                    .iter()
                    .map(|vertex| vertex.get())
                    .collect::<Vec<_>>()
                    .try_into()
                    .unwrap(),
            )
        })
        .collect::<Vec<_>>();
    assert_eq!(keys, EXPECTED_KEYS);

    const EXPECTED_DATA: [f64; 96] = [
        555.0, 560.0, 1110.0, 1120.0, 1665.0, 1680.0, 1055.0, 1060.0, 2110.0, 2120.0, 3165.0,
        3180.0, 1221.0, 1232.0, 2442.0, 2464.0, 3663.0, 3696.0, 2321.0, 2332.0, 4642.0, 4664.0,
        6963.0, 6996.0, 605.0, 610.0, 1210.0, 1220.0, 1815.0, 1830.0, 1105.0, 1110.0, 2210.0,
        2220.0, 3315.0, 3330.0, 1331.0, 1342.0, 2662.0, 2684.0, 3993.0, 4026.0, 2431.0, 2442.0,
        4862.0, 4884.0, 7293.0, 7326.0, 1221.0, 1232.0, 2442.0, 2464.0, 3663.0, 3696.0, 2321.0,
        2332.0, 4642.0, 4664.0, 6963.0, 6996.0, 2775.0, 2800.0, 5550.0, 5600.0, 8325.0, 8400.0,
        5275.0, 5300.0, 10550.0, 10600.0, 15825.0, 15900.0, 1331.0, 1342.0, 2662.0, 2684.0, 3993.0,
        4026.0, 2431.0, 2442.0, 4862.0, 4884.0, 7293.0, 7326.0, 3025.0, 3050.0, 6050.0, 6100.0,
        9075.0, 9150.0, 5525.0, 5550.0, 11050.0, 11100.0, 16575.0, 16650.0,
    ];
    assert_eq!(output.data(), EXPECTED_DATA);

    // The first stored value has two nonzero root-multiplicity paths:
    // μ=1 contributes 111*1 and μ=2 contributes 111*4. The fixed result
    // therefore kills overwrite-instead-of-accumulate mutations.
    let colliding_path_coefficients = [1.0, 4.0];
    assert_eq!(colliding_path_coefficients.len(), 2);
    assert_eq!(
        111.0 * colliding_path_coefficients.iter().sum::<f64>(),
        EXPECTED_DATA[0]
    );

    let complex_lhs: TensorMap<_, Complex64> =
        TensorMap::from_block_fn(&runtime, [&x2, &x1], [&x1, &x1], |trees, index| {
            Complex64::new(1.0, 1.0)
                * (100.0 * trees.codomain_vertices()[0].get() as f64
                    + 10.0 * trees.domain_vertices()[0].get() as f64
                    + index[0] as f64
                    + 1.0)
        })
        .unwrap();
    let complex_rhs: TensorMap<_, Complex64> =
        TensorMap::from_block_fn(&runtime, [&y3, &rhs_x1], [&y1, &rhs_x1], |_, index| {
            Complex64::new(2.0, -3.0) * (index[0] as f64 + 1.0)
        })
        .unwrap();
    first.f_queries.store(0, Ordering::Relaxed);
    first.reset_commit_spy();
    let complex = complex_lhs.otimes(&complex_rhs).unwrap();
    assert!(std::ptr::eq(complex.provider(), first.as_ref()));
    for (actual, expected) in complex.data().iter().zip(EXPECTED_DATA) {
        assert!((*actual - Complex64::new(5.0, -1.0) * expected).norm() <= 1e-12);
    }
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

#[test]
fn checked_generic_cat_admits_once_and_queries_only_left_before_commit() {
    // What: successful catdomain admission uses the left provider Arc once;
    // admitted identity stamps keep the equal-identity right provider cold,
    // and copy planning performs no provider query after commit.
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let left_provider = Arc::new(CheckedOnlyToy::new(0));
    let right_provider = Arc::new(CheckedOnlyToy::new(0));
    let left_common =
        GradedSpace::try_new(Arc::clone(&left_provider), [(Label::X, 1)], false).unwrap();
    let right_common =
        GradedSpace::try_new(Arc::clone(&right_provider), [(Label::X, 1)], false).unwrap();
    let left_changed =
        GradedSpace::try_new(Arc::clone(&left_provider), [(Label::X, 1)], false).unwrap();
    let right_changed =
        GradedSpace::try_new(Arc::clone(&right_provider), [(Label::X, 2)], false).unwrap();
    let lhs: TensorMap<_, f64> = TensorMap::from_block_fn(
        &runtime,
        [&left_common, &left_common],
        [&left_changed],
        |trees, indices| {
            10.0 + trees.codomain_vertices()[0].get() as f64 + indices.iter().sum::<usize>() as f64
        },
    )
    .unwrap();
    let rhs: TensorMap<_, f64> = TensorMap::from_block_fn(
        &runtime,
        [&right_common, &right_common],
        [&right_changed],
        |trees, indices| {
            20.0 + trees.codomain_vertices()[0].get() as f64 + indices.iter().sum::<usize>() as f64
        },
    )
    .unwrap();
    let combined = left_changed.oplus(&right_changed).unwrap();
    for provider in [&left_provider, &right_provider] {
        for counter in [
            &provider.identity_queries,
            &provider.style_queries,
            &provider.algebra_queries,
            &provider.coefficient_queries,
            &provider.f_queries,
            &provider.r_queries,
        ] {
            counter.store(0, Ordering::Relaxed);
        }
        provider.reset_commit_spy();
    }
    let _: TensorMap<_, f64> =
        TensorMap::zeros(&runtime, [&left_common, &left_common], [&combined]).unwrap();
    let mut admission_queries = [
        left_provider.identity_queries.load(Ordering::Relaxed),
        left_provider.style_queries.load(Ordering::Relaxed),
        left_provider.algebra_queries.load(Ordering::Relaxed),
        left_provider.coefficient_queries.load(Ordering::Relaxed),
        left_provider.f_queries.load(Ordering::Relaxed),
        left_provider.r_queries.load(Ordering::Relaxed),
    ];
    let admission_query_count = left_provider.queries_since_reset.load(Ordering::Relaxed) - 3;
    // `zeros` first checks the three supplied leg authorities; cat starts from
    // already-admitted stamps, so remove exactly those three identity reads.
    admission_queries[0] -= 3;
    for counter in [
        &left_provider.identity_queries,
        &left_provider.style_queries,
        &left_provider.algebra_queries,
        &left_provider.coefficient_queries,
        &left_provider.f_queries,
        &left_provider.r_queries,
    ] {
        counter.store(0, Ordering::Relaxed);
    }
    left_provider.arm_commit_spy_after_queries(admission_query_count);

    let output = lhs.catdomain(&rhs).unwrap();

    assert!(std::ptr::eq(output.provider(), left_provider.as_ref()));
    assert!(!std::ptr::eq(output.provider(), right_provider.as_ref()));
    assert_eq!(left_provider.commit_count.load(Ordering::Relaxed), 1);
    assert_eq!(left_provider.postcommit_queries.load(Ordering::Relaxed), 0);
    assert_eq!(
        [
            left_provider.identity_queries.load(Ordering::Relaxed),
            left_provider.style_queries.load(Ordering::Relaxed),
            left_provider.algebra_queries.load(Ordering::Relaxed),
            left_provider.coefficient_queries.load(Ordering::Relaxed),
            left_provider.f_queries.load(Ordering::Relaxed),
            left_provider.r_queries.load(Ordering::Relaxed),
        ],
        admission_queries
    );
    for counter in [
        &right_provider.identity_queries,
        &right_provider.style_queries,
        &right_provider.algebra_queries,
        &right_provider.coefficient_queries,
        &right_provider.f_queries,
        &right_provider.r_queries,
        &right_provider.postcommit_queries,
    ] {
        assert_eq!(counter.load(Ordering::Relaxed), 0);
    }
}

#[test]
fn checked_generic_cat_precedence_and_admission_failure_are_typed_nonpublishing() {
    // What: admission stamps, runtime, cat arguments, then output admission
    // reject in order; every failure leaves both admitted input payloads alone.
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let other_runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let provider = Arc::new(CheckedOnlyToy::new(0));
    let equal = Arc::new(CheckedOnlyToy::new(0));
    let wrong = Arc::new(CheckedOnlyToy::new(1));
    let leg = GradedSpace::try_new(Arc::clone(&provider), [(Label::X, 1)], false).unwrap();
    let equal_leg = GradedSpace::try_new(Arc::clone(&equal), [(Label::X, 1)], false).unwrap();
    let wrong_leg = GradedSpace::try_new(Arc::clone(&wrong), [(Label::X, 1)], false).unwrap();
    let lhs: TensorMap<_, f64> =
        TensorMap::from_block_fn(&runtime, [&leg, &leg], [&leg], |_, indices| {
            1.0 + indices.iter().sum::<usize>() as f64
        })
        .unwrap();
    let valid_rhs: TensorMap<_, f64> =
        TensorMap::from_block_fn(&runtime, [&equal_leg, &equal_leg], [&equal_leg], |_, _| 2.0)
            .unwrap();
    let wrong_identity: TensorMap<_, f64> = TensorMap::zeros(
        &other_runtime,
        [&wrong_leg, &wrong_leg],
        [&wrong_leg, &wrong_leg],
    )
    .unwrap();
    let wrong_runtime: TensorMap<_, f64> = TensorMap::zeros(
        &other_runtime,
        [&equal_leg, &equal_leg],
        [&equal_leg, &equal_leg],
    )
    .unwrap();
    let bad_arguments: TensorMap<_, f64> =
        TensorMap::zeros(&runtime, [&equal_leg, &equal_leg], [&equal_leg, &equal_leg]).unwrap();
    let lhs_before = lhs.data().to_vec();
    let rhs_before = valid_rhs.data().to_vec();
    provider.fail_algebra.store(true, Ordering::Relaxed);
    for counter in [
        &provider.identity_queries,
        &provider.style_queries,
        &provider.algebra_queries,
        &provider.coefficient_queries,
        &provider.f_queries,
        &provider.r_queries,
    ] {
        counter.store(0, Ordering::Relaxed);
    }
    provider.reset_commit_spy();

    assert!(matches!(
        lhs.catdomain(&wrong_identity),
        Err(GenericTensorError::Facade(
            tenet::prelude::Error::RuleMismatch
        ))
    ));
    assert!(matches!(
        lhs.catdomain(&wrong_runtime),
        Err(GenericTensorError::Facade(
            tenet::prelude::Error::RuntimeMismatch
        ))
    ));
    assert!(matches!(
        lhs.catdomain(&bad_arguments),
        Err(GenericTensorError::Facade(
            tenet::prelude::Error::InvalidArgument(_)
        ))
    ));
    assert_eq!(provider.identity_queries.load(Ordering::Relaxed), 0);
    assert_eq!(provider.algebra_queries.load(Ordering::Relaxed), 0);

    assert!(matches!(
        lhs.catdomain(&valid_rhs),
        Err(GenericTensorError::Structure(
            CheckedGenericStructureError::Provider(ToyError::Algebra)
        ))
    ));
    assert_eq!(provider.commit_count.load(Ordering::Relaxed), 0);
    assert_eq!(lhs.data(), lhs_before);
    assert_eq!(valid_rhs.data(), rhs_before);
}

#[cfg(feature = "racah-generated")]
fn sun_cat_marker(trees: &tenet::typed::BlockFusionTrees<Vec<i64>>) -> usize {
    trees
        .codomain_vertices()
        .iter()
        .enumerate()
        .map(|(index, vertex)| (index + 1) * 100 * vertex.get())
        .chain(
            trees
                .domain_vertices()
                .iter()
                .enumerate()
                .map(|(index, vertex)| (index + 1) * 1_000 * vertex.get()),
        )
        .sum()
}

#[cfg(feature = "racah-generated")]
fn assert_sun_cat_values<D>(
    output: &TensorMap<tenet::typed::SUNFusionRule, D>,
    lhs: &TensorMap<tenet::typed::SUNFusionRule, D>,
    rhs: &TensorMap<tenet::typed::SUNFusionRule, D>,
    changed_axis: usize,
    lhs_extent: usize,
    value: impl Fn(usize) -> D,
) where
    D: Copy + fmt::Debug + PartialEq + tenet::typed::TensorScalar,
{
    let mut saw_mu_two = false;
    for output_index in 0..output.block_count() {
        let trees = output.block_fusion_trees(output_index).unwrap();
        saw_mu_two |= trees
            .codomain_vertices()
            .iter()
            .chain(trees.domain_vertices())
            .any(|vertex| vertex.get() == 2);
        assert!((0..lhs.block_count()).any(|index| lhs.block_fusion_trees(index).unwrap() == trees));
        assert!((0..rhs.block_count()).any(|index| rhs.block_fusion_trees(index).unwrap() == trees));
        let block = output.block(output_index).unwrap();
        let elements = block.shape().iter().product::<usize>();
        for linear in 0..elements {
            let mut remainder = linear;
            let mut indices = Vec::with_capacity(block.shape().len());
            let mut position = block.offset();
            for (&extent, &stride) in block.shape().iter().zip(block.strides()) {
                let index = remainder % extent;
                remainder /= extent;
                indices.push(index);
                position += index * stride;
            }
            let (base, local_changed) = if indices[changed_axis] < lhs_extent {
                (10_000, indices[changed_axis])
            } else {
                (20_000, indices[changed_axis] - lhs_extent)
            };
            indices[changed_axis] = local_changed;
            let index_marker = indices
                .iter()
                .enumerate()
                .map(|(axis, index)| (axis + 1) * index)
                .sum::<usize>();
            assert_eq!(
                output.data()[position],
                value(base + sun_cat_marker(&trees) + index_marker)
            );
        }
    }
    assert!(saw_mu_two, "SU(N) cat fixture must carry a μ=2 full key");
}

#[cfg(feature = "racah-generated")]
fn assert_sun_cat_case<D>(n: usize, label: Vec<i64>, value: impl Fn(usize) -> D + Copy)
where
    D: Copy + fmt::Debug + PartialEq + tenet::typed::TensorScalar,
{
    use tenet::typed::SUNFusionRule;

    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let left_provider = Arc::new(SUNFusionRule::new(n).unwrap());
    let right_provider = Arc::new(SUNFusionRule::new(n).unwrap());
    let left_common =
        GradedSpace::try_new(Arc::clone(&left_provider), [(label.clone(), 1)], false).unwrap();
    let right_common =
        GradedSpace::try_new(Arc::clone(&right_provider), [(label.clone(), 1)], false).unwrap();
    let left_changed =
        GradedSpace::try_new(Arc::clone(&left_provider), [(label.clone(), 1)], false).unwrap();
    let right_changed =
        GradedSpace::try_new(Arc::clone(&right_provider), [(label.clone(), 2)], false).unwrap();
    let fill = |base, trees: &tenet::typed::BlockFusionTrees<Vec<i64>>, indices: &[usize]| {
        value(
            base + sun_cat_marker(trees)
                + indices
                    .iter()
                    .enumerate()
                    .map(|(axis, index)| (axis + 1) * index)
                    .sum::<usize>(),
        )
    };

    let domain_lhs: TensorMap<_, D> = TensorMap::from_block_fn(
        &runtime,
        [&left_common, &left_common],
        [&left_changed],
        |trees, indices| fill(10_000, trees, indices),
    )
    .unwrap();
    let domain_rhs: TensorMap<_, D> = TensorMap::from_block_fn(
        &runtime,
        [&right_common, &right_common],
        [&right_changed],
        |trees, indices| fill(20_000, trees, indices),
    )
    .unwrap();
    let domain = domain_lhs.catdomain(&domain_rhs).unwrap();
    assert!(std::ptr::eq(domain.provider(), left_provider.as_ref()));
    assert!(!std::ptr::eq(domain.provider(), right_provider.as_ref()));
    assert_eq!(domain.domain()[0].degeneracy(&label).unwrap(), 3);
    assert_sun_cat_values(&domain, &domain_lhs, &domain_rhs, 2, 1, value);
    let lazy_domain = domain_lhs
        .adjoint()
        .unwrap()
        .catcodomain(&domain_rhs.adjoint().unwrap())
        .unwrap();
    assert_eq!(lazy_domain.data(), domain.adjoint().unwrap().data());

    let codomain_lhs: TensorMap<_, D> = TensorMap::from_block_fn(
        &runtime,
        [&left_changed],
        [&left_common, &left_common],
        |trees, indices| fill(10_000, trees, indices),
    )
    .unwrap();
    let codomain_rhs: TensorMap<_, D> = TensorMap::from_block_fn(
        &runtime,
        [&right_changed],
        [&right_common, &right_common],
        |trees, indices| fill(20_000, trees, indices),
    )
    .unwrap();
    let codomain = codomain_lhs.catcodomain(&codomain_rhs).unwrap();
    assert!(std::ptr::eq(codomain.provider(), left_provider.as_ref()));
    assert_eq!(codomain.codomain()[0].degeneracy(&label).unwrap(), 3);
    assert_sun_cat_values(&codomain, &codomain_lhs, &codomain_rhs, 0, 1, value);
    let lazy_codomain = codomain_lhs
        .adjoint()
        .unwrap()
        .catdomain(&codomain_rhs.adjoint().unwrap())
        .unwrap();
    assert_eq!(lazy_codomain.data(), codomain.adjoint().unwrap().data());
}

#[cfg(feature = "racah-generated")]
#[test]
fn sun_checked_generic_cat_covers_both_directions_dtypes_and_mu_two_keys() {
    // What: exact TensorKit direct-sum slab values, μ=2 full-key matching,
    // distinct equal-identity Arcs, left authority, and lazy-adjoint parity.
    for (n, label) in [(3, vec![1, 1]), (4, vec![1, 0, 1])] {
        assert_sun_cat_case(n, label.clone(), |value| value as f64);
        assert_sun_cat_case(n, label, |value| {
            Complex64::new(value as f64, -(value as f64))
        });
    }
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

        let identity: TensorMap<_, f64> =
            TensorMap::from_block_fn(&runtime, [&leg], [&leg], |_, _| 1.0).unwrap();
        for output in [
            tensor.contract(&identity, &[2], &[0], &[0, 1, 2]).unwrap(),
            tensor
                .contract_ordered(&identity, &[2], &[0], &[0, 1, 2])
                .unwrap(),
            tensor.compose(&identity).unwrap(),
        ] {
            assert!(std::ptr::eq(output.provider(), provider.as_ref()));
            assert_eq!(output.data(), tensor.data());
            for index in 0..tensor.block_count() {
                assert_eq!(
                    output.block_fusion_trees(index).unwrap(),
                    tensor.block_fusion_trees(index).unwrap()
                );
            }
        }

        let product = tensor.otimes(&tensor).unwrap();
        assert!(std::ptr::eq(product.provider(), provider.as_ref()));
        let (expected_len, expected_sum, expected_weighted, expected_prefix): (
            usize,
            f64,
            f64,
            &[f64],
        ) = match n {
            3 => (
                145,
                9.468_841_418_575_323,
                39.231_504_693_264_13,
                &[
                    0.0,
                    1.0,
                    2.0,
                    2.0,
                    4.0,
                    0.0,
                    0.0,
                    0.0,
                    0.353_553_390_593_273_6,
                    0.707_106_781_186_547_2,
                    0.0,
                    0.857_142_857_142_857,
                ],
            ),
            4 => (
                245,
                8.608_165_620_335_726,
                -1.317_392_645_553_582,
                &[
                    0.0,
                    0.0,
                    1.0,
                    2.0,
                    2.0,
                    4.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.338_873_675_850_995_87,
                    0.677_747_351_701_991_7,
                ],
            ),
            _ => unreachable!(),
        };
        assert_eq!(product.data().len(), expected_len);
        let sum = product.data().iter().sum::<f64>();
        let weighted = product
            .data()
            .iter()
            .enumerate()
            .map(|(index, value)| (index + 1) as f64 * value)
            .sum::<f64>();
        assert!((sum - expected_sum).abs() <= 1e-10);
        assert!((weighted - expected_weighted).abs() <= 1e-9);
        for (&actual, &expected) in product.data().iter().zip(expected_prefix) {
            assert!((actual - expected).abs() <= 1e-10);
        }
        let mut adjoint_root_vertices = Vec::new();
        for index in 0..product.block_count() {
            let trees = product.block_fusion_trees(index).unwrap();
            if trees.coupled() == &adjoint
                && product.data()[product.block(index).unwrap().offset()].abs() > 1e-10
            {
                assert_eq!(trees.codomain_uncoupled(), vec![adjoint.clone(); 4]);
                assert_eq!(trees.domain_uncoupled(), vec![adjoint.clone(); 2]);
                adjoint_root_vertices.push(trees.domain_vertices().last().unwrap().get());
            }
        }
        adjoint_root_vertices.sort_unstable();
        adjoint_root_vertices.dedup();
        assert_eq!(adjoint_root_vertices, [1, 2]);

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
            tensor.transpose().unwrap().transpose().unwrap(),
        ] {
            assert!(std::ptr::eq(restored.provider(), provider.as_ref()));
            assert_eq!(snapshot(&restored), source_snapshot);
            for (actual, expected) in restored.data().iter().zip(tensor.data()) {
                assert!((actual - expected).abs() <= 1e-10);
            }
        }
    }
}

#[cfg(feature = "racah-generated")]
#[test]
fn sun_checked_generic_adjoint_and_reductions_preserve_provider_and_errors() {
    use tenet::core::SUNFusionRuleError;
    use tenet::typed::SUNFusionRule;

    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    for (n, adjoint) in [(3, vec![1, 1]), (4, vec![1, 0, 1])] {
        let provider = Arc::new(SUNFusionRule::new(n).unwrap());
        let leg = GradedSpace::try_new(Arc::clone(&provider), [(adjoint, 1)], false).unwrap();
        let source: TensorMap<_, f64> =
            TensorMap::from_block_fn(&runtime, [&leg], [&leg], |trees, _| {
                trees.coupled().iter().sum::<i64>() as f64 + 1.0
            })
            .unwrap();

        let dagger = source.adjoint().unwrap();
        assert!(std::ptr::eq(dagger.provider(), provider.as_ref()));
        assert!((dagger.norm().unwrap() - source.norm().unwrap()).abs() < 1.0e-12);
        assert!((dagger.inner(&dagger).unwrap() - source.inner(&source).unwrap()).abs() < 1.0e-12);
        assert!((dagger.tr().unwrap() - source.tr().unwrap()).abs() < 1.0e-12);

        let complex = source.to_c64();
        let complex_dagger = complex.adjoint().unwrap();
        assert!(std::ptr::eq(complex_dagger.provider(), provider.as_ref()));
        assert!((complex_dagger.tr().unwrap() - complex.tr().unwrap().conj()).norm() < 1.0e-12);
    }

    let provider = SUNFusionRule::new(3).unwrap();
    let three = provider.encode_dynkin(&[1, 0]).unwrap();
    let eight = provider.encode_dynkin(&[1, 1]).unwrap();
    assert!(matches!(
        provider.try_r_symbol_generic(three, three, eight),
        Err(SUNFusionRuleError::Racah(_))
    ));
}

#[cfg(feature = "racah-generated")]
#[test]
fn sun_checked_generic_transforms_reuse_the_runtime_completed_store() {
    use tenet::typed::SUNFusionRule;

    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    for (n, adjoint) in [(3, vec![1, 1]), (4, vec![1, 0, 1])] {
        let provider = Arc::new(SUNFusionRule::new(n).unwrap());
        let leg = GradedSpace::try_new(Arc::clone(&provider), [(adjoint, 1)], false).unwrap();
        let source: TensorMap<_, f64> =
            TensorMap::from_block_fn(&runtime, [&leg, &leg], [&leg], |trees, _| {
                trees.codomain_vertices()[0].get() as f64
            })
            .unwrap();

        for operation in ["permute", "braid", "repartition"] {
            runtime.clear_tree_transform_cache();
            let apply = |tensor: &TensorMap<SUNFusionRule, f64>| match operation {
                "permute" => tensor.permute(&[1, 0], &[2]),
                "braid" => tensor.braid(&[1, 0], &[2], &[0, 1, 2]),
                "repartition" => tensor.repartition(1),
                _ => unreachable!(),
            };
            let first = apply(&source).unwrap();
            let cold = runtime.tree_transform_cache_info();
            let repeated = apply(&source).unwrap();
            let warm = runtime.tree_transform_cache_info();

            assert_eq!(cold.entries(), 1);
            assert_eq!(cold.misses(), 1);
            assert_eq!(warm.entries(), 1);
            assert_eq!(warm.misses(), 1);
            assert_eq!(warm.hits(), 1);
            assert_eq!(repeated.data(), first.data());
            assert!(std::ptr::eq(first.provider(), provider.as_ref()));
            assert!(std::ptr::eq(repeated.provider(), provider.as_ref()));
            for index in 0..first.block_count() {
                assert_eq!(
                    repeated.block_fusion_trees(index).unwrap(),
                    first.block_fusion_trees(index).unwrap()
                );
            }
        }
    }
}
