use std::fmt;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use tenet::core::{
    BraidingStyleKind, CheckedGenericAdmissionMode, CheckedGenericFusion, CheckedGenericPivotal,
    CheckedGenericRigidSymbols, FusionStyleKind, GenericFArray, GenericRMatrix, RuleIdentity,
    SectorId, SectorVec, TypedSectorAdmission,
};
use tenet::prelude::{Complex64, Error, Runtime};
use tenet::typed::{
    CheckedGenericPlanError, GenericTensorError, GradedSpace, TensorMap, TensorScalar,
};

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
enum Label {
    Unit,
    X,
}

#[derive(Clone, Debug, Eq, PartialEq)]
enum PivotalError {
    InvalidSector,
    FrobeniusSchur,
    Twist,
}

impl fmt::Display for PivotalError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{self:?}")
    }
}

impl std::error::Error for PivotalError {}

struct CheckedPivotalToy {
    identity_tag: u8,
    braiding: BraidingStyleKind,
    x_fs: f64,
    x_twist: f64,
    style_queries: AtomicUsize,
    vacuum_queries: AtomicUsize,
    fs_queries: AtomicUsize,
    twist_queries: AtomicUsize,
    other_queries: AtomicUsize,
    post_stage_queries: AtomicUsize,
    stage_after_twists: AtomicUsize,
    staged: AtomicBool,
    fail_twist_on: AtomicUsize,
    fail_fs_on: AtomicUsize,
    twist_sectors: Mutex<Vec<SectorId>>,
}

impl CheckedPivotalToy {
    fn new(identity_tag: u8, braiding: BraidingStyleKind, x_twist: f64) -> Self {
        Self::with_fs(identity_tag, braiding, 1.0, x_twist)
    }

    fn with_fs(identity_tag: u8, braiding: BraidingStyleKind, x_fs: f64, x_twist: f64) -> Self {
        Self {
            identity_tag,
            braiding,
            x_fs,
            x_twist,
            style_queries: AtomicUsize::new(0),
            vacuum_queries: AtomicUsize::new(0),
            fs_queries: AtomicUsize::new(0),
            twist_queries: AtomicUsize::new(0),
            other_queries: AtomicUsize::new(0),
            post_stage_queries: AtomicUsize::new(0),
            stage_after_twists: AtomicUsize::new(0),
            staged: AtomicBool::new(false),
            fail_twist_on: AtomicUsize::new(0),
            fail_fs_on: AtomicUsize::new(0),
            twist_sectors: Mutex::new(Vec::new()),
        }
    }

    fn unit() -> SectorId {
        SectorId::new(0)
    }

    fn x() -> SectorId {
        SectorId::new(1)
    }

    fn reset_ledger(&self, stage_after_twists: usize) {
        self.style_queries.store(0, Ordering::Relaxed);
        self.vacuum_queries.store(0, Ordering::Relaxed);
        self.fs_queries.store(0, Ordering::Relaxed);
        self.twist_queries.store(0, Ordering::Relaxed);
        self.other_queries.store(0, Ordering::Relaxed);
        self.post_stage_queries.store(0, Ordering::Relaxed);
        self.stage_after_twists
            .store(stage_after_twists, Ordering::Relaxed);
        self.staged.store(false, Ordering::Relaxed);
        self.fail_twist_on.store(0, Ordering::Relaxed);
        self.fail_fs_on.store(0, Ordering::Relaxed);
        self.twist_sectors.lock().unwrap().clear();
    }

    fn record_query(&self) {
        if self.staged.load(Ordering::Relaxed) {
            self.post_stage_queries.fetch_add(1, Ordering::Relaxed);
        }
    }

    fn record_other_query(&self) {
        self.record_query();
        self.other_queries.fetch_add(1, Ordering::Relaxed);
    }

    fn finish_observation(&self) {
        self.staged.store(false, Ordering::Relaxed);
    }

    fn fusion_channels(left: SectorId, right: SectorId) -> SectorVec {
        match (left, right) {
            (left, right) if left == Self::unit() => [right].into_iter().collect(),
            (left, right) if right == Self::unit() => [left].into_iter().collect(),
            (left, right) if left == Self::x() && right == Self::x() => {
                [Self::x()].into_iter().collect()
            }
            _ => SectorVec::new(),
        }
    }

    fn nsymbol(left: SectorId, right: SectorId, coupled: SectorId) -> usize {
        if left == Self::x() && right == Self::x() && coupled == Self::x() {
            2
        } else {
            usize::from(Self::fusion_channels(left, right).contains(&coupled))
        }
    }
}

impl CheckedGenericFusion for CheckedPivotalToy {
    type Error = PivotalError;

    fn rule_identity(&self) -> RuleIdentity {
        self.record_other_query();
        RuleIdentity::from_canonical_bytes::<Self>(
            0x951,
            Arc::<[u8]>::from(vec![
                self.identity_tag,
                self.braiding as u8,
                u8::from(self.x_fs.is_sign_negative()),
                u8::from(self.x_twist.is_sign_negative()),
            ]),
        )
    }

    fn fusion_style(&self) -> FusionStyleKind {
        self.record_other_query();
        FusionStyleKind::Generic
    }

    fn braiding_style(&self) -> BraidingStyleKind {
        self.record_query();
        self.style_queries.fetch_add(1, Ordering::Relaxed);
        self.braiding
    }

    fn vacuum(&self) -> SectorId {
        self.record_query();
        self.vacuum_queries.fetch_add(1, Ordering::Relaxed);
        Self::unit()
    }

    fn try_dual(&self, sector: SectorId) -> Result<SectorId, Self::Error> {
        self.record_other_query();
        match sector {
            sector if sector == Self::unit() || sector == Self::x() => Ok(sector),
            _ => Err(PivotalError::InvalidSector),
        }
    }

    fn try_fusion_channels(
        &self,
        left: SectorId,
        right: SectorId,
    ) -> Result<SectorVec, Self::Error> {
        self.record_other_query();
        Ok(Self::fusion_channels(left, right))
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
        self.record_other_query();
        Ok(Self::nsymbol(left, right, coupled))
    }
}

impl CheckedGenericRigidSymbols for CheckedPivotalToy {
    type Scalar = f64;

    fn try_sqrt_dim_scalar(&self, _: SectorId) -> Result<f64, Self::Error> {
        self.record_other_query();
        Ok(1.0)
    }

    fn try_inv_sqrt_dim_scalar(&self, _: SectorId) -> Result<f64, Self::Error> {
        self.record_other_query();
        Ok(1.0)
    }

    fn try_frobenius_schur_phase_scalar(&self, sector: SectorId) -> Result<f64, Self::Error> {
        self.record_query();
        let query = self.fs_queries.fetch_add(1, Ordering::Relaxed) + 1;
        if self.fail_fs_on.load(Ordering::Relaxed) == query {
            return Err(PivotalError::FrobeniusSchur);
        }
        if sector == Self::unit() {
            Ok(1.0)
        } else if sector == Self::x() {
            Ok(self.x_fs)
        } else {
            Err(PivotalError::InvalidSector)
        }
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
        self.record_other_query();
        let shape = (
            Self::nsymbol(a, b, e),
            Self::nsymbol(e, c, d),
            Self::nsymbol(b, c, f),
            Self::nsymbol(a, f, d),
        );
        let len = shape.0 * shape.1 * shape.2 * shape.3;
        Ok(GenericFArray::new(vec![0.0; len], shape))
    }

    fn try_r_symbol_generic(
        &self,
        a: SectorId,
        b: SectorId,
        c: SectorId,
    ) -> Result<GenericRMatrix<f64>, Self::Error> {
        self.record_other_query();
        let rows = Self::nsymbol(a, b, c);
        let cols = Self::nsymbol(b, a, c);
        Ok(GenericRMatrix::new(vec![0.0; rows * cols], rows, cols))
    }
}

impl CheckedGenericPivotal for CheckedPivotalToy {
    fn try_twist_scalar(&self, sector: SectorId) -> Result<f64, Self::Error> {
        self.record_query();
        let query = self.twist_queries.fetch_add(1, Ordering::Relaxed) + 1;
        self.twist_sectors.lock().unwrap().push(sector);
        if self.fail_twist_on.load(Ordering::Relaxed) == query {
            return Err(PivotalError::Twist);
        }
        let value = if sector == Self::unit() {
            1.0
        } else if sector == Self::x() {
            self.x_twist
        } else {
            return Err(PivotalError::InvalidSector);
        };
        if self.stage_after_twists.load(Ordering::Relaxed) == query {
            self.staged.store(true, Ordering::Relaxed);
        }
        Ok(value)
    }
}

impl TypedSectorAdmission for CheckedPivotalToy {
    type Sector = Label;
    type Error = PivotalError;
    type Mode = CheckedGenericAdmissionMode;

    fn typed_rule_identity(&self) -> RuleIdentity {
        CheckedGenericFusion::rule_identity(self)
    }

    fn try_encode_label(&self, sector: &Label) -> Result<SectorId, Self::Error> {
        self.record_other_query();
        Ok(match sector {
            Label::Unit => Self::unit(),
            Label::X => Self::x(),
        })
    }

    fn try_decode_label(&self, sector: SectorId) -> Result<Label, Self::Error> {
        self.record_other_query();
        if sector == Self::unit() {
            Ok(Label::Unit)
        } else if sector == Self::x() {
            Ok(Label::X)
        } else {
            Err(PivotalError::InvalidSector)
        }
    }

    fn try_dual_id(&self, sector: SectorId) -> Result<SectorId, Self::Error> {
        CheckedGenericFusion::try_dual(self, sector)
    }
}

fn label_marker(label: Label) -> usize {
    match label {
        Label::Unit => 1,
        Label::X => 7,
    }
}

fn tree_marker(trees: &tenet::typed::BlockFusionTrees<Label>) -> usize {
    10_000 * label_marker(*trees.coupled())
        + trees
            .codomain_uncoupled()
            .iter()
            .enumerate()
            .map(|(index, &label)| (index + 1) * 100 * label_marker(label))
            .sum::<usize>()
        + trees
            .codomain_innerlines()
            .iter()
            .enumerate()
            .map(|(index, &label)| (index + 1) * 1_000 * label_marker(label))
            .sum::<usize>()
        + trees
            .codomain_vertices()
            .iter()
            .enumerate()
            .map(|(index, vertex)| (index + 1) * 10 * vertex.get())
            .sum::<usize>()
        + trees
            .domain_uncoupled()
            .iter()
            .enumerate()
            .map(|(index, &label)| (index + 1) * 2_000 * label_marker(label))
            .sum::<usize>()
        + trees
            .domain_innerlines()
            .iter()
            .enumerate()
            .map(|(index, &label)| (index + 1) * 3_000 * label_marker(label))
            .sum::<usize>()
        + trees
            .domain_vertices()
            .iter()
            .enumerate()
            .map(|(index, vertex)| (index + 1) * 20 * vertex.get())
            .sum::<usize>()
}

fn fixture<D>(
    runtime: &Runtime,
    provider: &Arc<CheckedPivotalToy>,
    value: impl Fn(usize) -> D,
) -> TensorMap<CheckedPivotalToy, D>
where
    D: TensorScalar,
{
    let mixed =
        GradedSpace::try_new_with_arc(Arc::clone(provider), [(Label::Unit, 1), (Label::X, 2)])
            .unwrap();
    let x = GradedSpace::try_new_with_arc(Arc::clone(provider), [(Label::X, 1)]).unwrap();
    TensorMap::from_block_fn(runtime, [&mixed, &x], [&x], |trees, indices| {
        value(
            tree_marker(trees)
                + indices
                    .iter()
                    .enumerate()
                    .map(|(axis, index)| (axis + 1) * index)
                    .sum::<usize>(),
        )
    })
    .unwrap()
}

fn assert_values<D>(
    tensor: &TensorMap<CheckedPivotalToy, D>,
    value: impl Fn(usize) -> D,
    factor: impl Fn(&tenet::typed::BlockFusionTrees<Label>) -> f64,
) where
    D: TensorScalar + fmt::Debug + PartialEq,
{
    for block_index in 0..tensor.block_count() {
        let block = tensor.block(block_index).unwrap();
        let trees = tensor.block_fusion_trees(block_index).unwrap();
        let elements = block.shape().iter().product::<usize>();
        for linear in 0..elements {
            let mut remainder = linear;
            let mut position = block.offset();
            let mut marker = tree_marker(&trees);
            for (axis, (&extent, &stride)) in block.shape().iter().zip(block.strides()).enumerate()
            {
                let index = remainder % extent;
                remainder /= extent;
                position += index * stride;
                marker += (axis + 1) * index;
            }
            assert_eq!(
                tensor.data()[position],
                value(marker) * D::from_real(factor(&trees))
            );
        }
    }
}

fn assert_nontrivial_case<D>(value: impl Fn(usize) -> D + Copy)
where
    D: TensorScalar + fmt::Debug + PartialEq,
{
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let provider = Arc::new(CheckedPivotalToy::new(1, BraidingStyleKind::Anyonic, -1.0));
    let source = fixture(&runtime, &provider, value);
    let before = source.data().to_vec();
    let mut saw_outer_two = false;
    let mut saw_vertex_two = false;
    for index in 0..source.block_count() {
        let block = source.block(index).unwrap();
        let trees = source.block_fusion_trees(index).unwrap();
        saw_outer_two |= trees.codomain_uncoupled()[0] == Label::X && block.shape()[0] == 2;
        saw_vertex_two |= trees
            .codomain_vertices()
            .iter()
            .any(|vertex| vertex.get() == 2);
    }
    assert!(saw_outer_two);
    assert!(saw_vertex_two);

    provider.reset_ledger(2);
    let codomain = source.twist(&[0]).unwrap();
    assert_eq!(provider.style_queries.load(Ordering::Relaxed), 1);
    assert_eq!(provider.twist_queries.load(Ordering::Relaxed), 2);
    assert_eq!(provider.post_stage_queries.load(Ordering::Relaxed), 0);
    assert_eq!(provider.other_queries.load(Ordering::Relaxed), 0);
    assert_eq!(
        provider.twist_sectors.lock().unwrap().as_slice(),
        &[CheckedPivotalToy::unit(), CheckedPivotalToy::x()]
    );
    provider.finish_observation();
    assert!(std::ptr::eq(codomain.provider(), provider.as_ref()));
    assert_ne!(codomain.data().as_ptr(), source.data().as_ptr());
    assert_eq!(codomain.codomain(), source.codomain());
    assert_eq!(codomain.domain(), source.domain());
    assert_values(&codomain, value, |trees| {
        if trees.codomain_uncoupled()[0] == Label::X {
            -1.0
        } else {
            1.0
        }
    });

    provider.reset_ledger(1);
    let domain = source.twist_inverse(&[2]).unwrap();
    assert_eq!(provider.style_queries.load(Ordering::Relaxed), 1);
    assert_eq!(provider.twist_queries.load(Ordering::Relaxed), 1);
    assert_eq!(provider.post_stage_queries.load(Ordering::Relaxed), 0);
    provider.finish_observation();
    assert_values(&domain, value, |_| -1.0);

    provider.reset_ledger(2);
    let repeated = source.twist(&[0, 0]).unwrap();
    assert_eq!(provider.post_stage_queries.load(Ordering::Relaxed), 0);
    provider.finish_observation();
    assert_values(&repeated, value, |_| 1.0);

    let lazy = source.adjoint().unwrap();
    // What: logical adjoint axis 1 maps to parent codomain axis 0; logical
    // axis 0 would instead map to the uniformly-X parent domain axis 2.
    provider.reset_ledger(2);
    let lazy_twisted = lazy.twist(&[1]).unwrap();
    assert_eq!(provider.style_queries.load(Ordering::Relaxed), 1);
    assert_eq!(provider.twist_queries.load(Ordering::Relaxed), 2);
    assert_eq!(provider.post_stage_queries.load(Ordering::Relaxed), 0);
    provider.finish_observation();
    let direct = source.twist_inverse(&[0]).unwrap().adjoint().unwrap();
    assert_eq!(lazy_twisted.data(), direct.data());
    assert!(std::ptr::eq(lazy_twisted.provider(), provider.as_ref()));
    assert_eq!(source.data(), before);
}

#[test]
fn checked_generic_twist_scales_full_keys_for_real_and_complex_payloads() {
    // What: +1/-1 factors act on exact Generic tree keys, including μ=2
    // vertices and outer degeneracy two, for codomain/domain/repeated axes and
    // direct/lazy orientations without changing the input or provider Arc.
    assert_nontrivial_case(|marker| marker as f64);
    assert_nontrivial_case(|marker| Complex64::new(marker as f64, -(marker as f64) / 10.0));
}

#[test]
fn checked_generic_twist_precedence_and_late_failure_are_typed_and_nonpublishing() {
    // What: index and empty requests precede every provider query; NoBraiding
    // rejection is an early facade error, while a staged pivotal failure keeps
    // its provider type and leaves the source payload unchanged.
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let provider = Arc::new(CheckedPivotalToy::new(2, BraidingStyleKind::Anyonic, -1.0));
    let source = fixture(&runtime, &provider, |marker| marker as f64);
    let before = source.data().to_vec();

    provider.reset_ledger(0);
    assert!(matches!(
        source.twist(&[source.rank()]),
        Err(GenericTensorError::Facade(Error::InvalidArgument(_)))
    ));
    assert_eq!(provider.style_queries.load(Ordering::Relaxed), 0);
    assert_eq!(provider.twist_queries.load(Ordering::Relaxed), 0);
    assert_eq!(provider.other_queries.load(Ordering::Relaxed), 0);

    provider.reset_ledger(0);
    let empty = source.twist(&[]).unwrap();
    assert_eq!(empty.data().as_ptr(), source.data().as_ptr());
    assert_eq!(provider.style_queries.load(Ordering::Relaxed), 0);
    assert_eq!(provider.twist_queries.load(Ordering::Relaxed), 0);
    assert_eq!(provider.other_queries.load(Ordering::Relaxed), 0);

    provider.reset_ledger(2);
    provider.fail_twist_on.store(2, Ordering::Relaxed);
    assert!(matches!(
        source.twist(&[0]),
        Err(GenericTensorError::Plan(CheckedGenericPlanError::Provider(
            PivotalError::Twist
        )))
    ));
    assert_eq!(provider.style_queries.load(Ordering::Relaxed), 1);
    assert_eq!(provider.twist_queries.load(Ordering::Relaxed), 2);
    assert_eq!(provider.post_stage_queries.load(Ordering::Relaxed), 0);
    assert_eq!(source.data(), before);
}

#[test]
fn checked_generic_twist_handles_nobraiding_bosonic_and_staged_identity_sharing() {
    // What: TensorKit's NoBraiding unit carve-out and bosonic short circuit
    // share storage without pivotal queries; an anyonic all-one provider stages
    // values once and then takes the same identity-sharing path.
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();

    let no_braiding = Arc::new(CheckedPivotalToy::new(
        3,
        BraidingStyleKind::NoBraiding,
        -1.0,
    ));
    let unit = GradedSpace::try_new_with_arc(Arc::clone(&no_braiding), [(Label::Unit, 1)]).unwrap();
    let unit_tensor: TensorMap<_, f64> =
        TensorMap::from_block_fn(&runtime, [&unit], [&unit], |_, _| 3.0).unwrap();
    no_braiding.reset_ledger(0);
    let unit_twist = unit_tensor.twist(&[0, 1]).unwrap();
    assert_eq!(unit_twist.data().as_ptr(), unit_tensor.data().as_ptr());
    assert_eq!(no_braiding.style_queries.load(Ordering::Relaxed), 1);
    assert_eq!(no_braiding.vacuum_queries.load(Ordering::Relaxed), 1);
    assert_eq!(no_braiding.twist_queries.load(Ordering::Relaxed), 0);

    let nonunit = fixture(&runtime, &no_braiding, |marker| marker as f64);
    let before = nonunit.data().to_vec();
    no_braiding.reset_ledger(0);
    assert!(matches!(
        nonunit.twist(&[0]),
        Err(GenericTensorError::Facade(Error::InvalidArgument(_)))
    ));
    assert_eq!(no_braiding.style_queries.load(Ordering::Relaxed), 1);
    assert_eq!(no_braiding.vacuum_queries.load(Ordering::Relaxed), 1);
    assert_eq!(no_braiding.twist_queries.load(Ordering::Relaxed), 0);
    assert_eq!(nonunit.data(), before);

    let bosonic = Arc::new(CheckedPivotalToy::new(4, BraidingStyleKind::Bosonic, -1.0));
    let bosonic_tensor = fixture(&runtime, &bosonic, |marker| marker as f64);
    bosonic.reset_ledger(0);
    let bosonic_twist = bosonic_tensor.twist(&[0]).unwrap();
    assert_eq!(
        bosonic_twist.data().as_ptr(),
        bosonic_tensor.data().as_ptr()
    );
    assert_eq!(bosonic.style_queries.load(Ordering::Relaxed), 1);
    assert_eq!(bosonic.vacuum_queries.load(Ordering::Relaxed), 0);
    assert_eq!(bosonic.twist_queries.load(Ordering::Relaxed), 0);

    let identity = Arc::new(CheckedPivotalToy::new(5, BraidingStyleKind::Anyonic, 1.0));
    let identity_tensor = fixture(&runtime, &identity, |marker| marker as f64);
    identity.reset_ledger(2);
    let identity_twist = identity_tensor.twist(&[0]).unwrap();
    assert_eq!(identity.post_stage_queries.load(Ordering::Relaxed), 0);
    assert_eq!(identity.twist_queries.load(Ordering::Relaxed), 2);
    identity.finish_observation();
    assert_eq!(
        identity_twist.data().as_ptr(),
        identity_tensor.data().as_ptr()
    );
    assert!(std::ptr::eq(identity_twist.provider(), identity.as_ref()));
}

fn assert_flip_nontrivial_case<D>(value: impl Fn(usize) -> D + Copy)
where
    D: TensorScalar + fmt::Debug + PartialEq,
{
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let provider = Arc::new(CheckedPivotalToy::new(6, BraidingStyleKind::Anyonic, -1.0));
    let source = fixture(&runtime, &provider, value);
    let before = source.data().to_vec();
    let source_codomain = source.codomain();
    let source_domain = source.domain();

    provider.reset_ledger(2);
    let codomain = source.flip_inverse(&[0]).unwrap();
    assert_eq!(provider.fs_queries.load(Ordering::Relaxed), 2);
    assert_eq!(provider.twist_queries.load(Ordering::Relaxed), 2);
    assert_eq!(provider.post_stage_queries.load(Ordering::Relaxed), 0);
    provider.finish_observation();
    assert!(std::ptr::eq(codomain.provider(), provider.as_ref()));
    assert_eq!(
        codomain.codomain()[0].is_dual(),
        !source.codomain()[0].is_dual()
    );
    assert_values(&codomain, value, |trees| {
        if trees.codomain_uncoupled()[0] == Label::X {
            -1.0
        } else {
            1.0
        }
    });

    provider.reset_ledger(2);
    let domain = source.flip(&[2]).unwrap();
    assert_eq!(provider.post_stage_queries.load(Ordering::Relaxed), 0);
    provider.finish_observation();
    assert_eq!(domain.domain()[0].is_dual(), !source.domain()[0].is_dual());
    assert_values(&domain, value, |_| -1.0);

    provider.reset_ledger(2);
    let repeated = source.flip(&[0, 0]).unwrap();
    assert_eq!(provider.post_stage_queries.load(Ordering::Relaxed), 0);
    provider.finish_observation();
    assert_eq!(
        repeated.codomain()[0].is_dual(),
        source.codomain()[0].is_dual()
    );
    assert_values(&repeated, value, |trees| {
        if trees.codomain_uncoupled()[0] == Label::X {
            -1.0
        } else {
            1.0
        }
    });

    let lazy = source.adjoint().unwrap();
    provider.reset_ledger(2);
    let lazy_flipped = lazy.flip(&[1]).unwrap();
    assert_eq!(provider.post_stage_queries.load(Ordering::Relaxed), 0);
    provider.finish_observation();
    assert_eq!(
        lazy_flipped.domain()[0].is_dual(),
        !lazy.domain()[0].is_dual()
    );
    provider.reset_ledger(2);
    let direct = source.flip_inverse(&[0]).unwrap().adjoint().unwrap();
    provider.finish_observation();
    assert_eq!(lazy_flipped.data(), direct.data());
    assert_eq!(lazy_flipped.codomain(), direct.codomain());
    assert_eq!(lazy_flipped.domain(), direct.domain());
    assert!(std::ptr::eq(lazy_flipped.provider(), provider.as_ref()));
    assert!(lazy_flipped.runtime().shares_state_with(direct.runtime()));
    assert_eq!(source.data(), before);
    assert_eq!(source.codomain(), source_codomain);
    assert_eq!(source.domain(), source_domain);
}

#[test]
fn checked_generic_flip_scales_full_keys_and_lazy_duality_for_real_and_complex_payloads() {
    // What: checked Generic replay uses staged χ/θ factors on complete μ=2
    // keys, including repeated legs, and a lazy result owns toggled metadata.
    assert_flip_nontrivial_case(|marker| marker as f64);
    assert_flip_nontrivial_case(|marker| Complex64::new(marker as f64, -(marker as f64) / 10.0));
}

#[test]
fn checked_generic_flip_rejects_nobraiding_before_pivotal_queries() {
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let provider = Arc::new(CheckedPivotalToy::new(
        7,
        BraidingStyleKind::NoBraiding,
        -1.0,
    ));
    let source = fixture(&runtime, &provider, |marker| marker as f64);

    provider.reset_ledger(0);
    assert!(matches!(
        source.flip(&[0]),
        Err(GenericTensorError::Facade(Error::InvalidArgument(_)))
    ));
    assert_eq!(provider.style_queries.load(Ordering::Relaxed), 1);
    assert_eq!(provider.vacuum_queries.load(Ordering::Relaxed), 0);
    assert_eq!(provider.fs_queries.load(Ordering::Relaxed), 0);
    assert_eq!(provider.twist_queries.load(Ordering::Relaxed), 0);
}

#[test]
fn checked_generic_flip_precedence_and_staged_failures_are_nonpublishing() {
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let provider = Arc::new(CheckedPivotalToy::new(8, BraidingStyleKind::Anyonic, -1.0));
    let source = fixture(&runtime, &provider, |marker| marker as f64);
    let before = source.data().to_vec();

    provider.reset_ledger(0);
    assert!(matches!(
        source.flip(&[source.rank()]),
        Err(GenericTensorError::Facade(Error::InvalidArgument(_)))
    ));
    assert_eq!(provider.style_queries.load(Ordering::Relaxed), 0);
    assert_eq!(provider.fs_queries.load(Ordering::Relaxed), 0);
    assert_eq!(provider.twist_queries.load(Ordering::Relaxed), 0);

    provider.reset_ledger(0);
    let empty = source.flip(&[]).unwrap();
    assert_eq!(empty.data().as_ptr(), source.data().as_ptr());
    assert_eq!(provider.style_queries.load(Ordering::Relaxed), 0);
    assert_eq!(provider.fs_queries.load(Ordering::Relaxed), 0);
    assert_eq!(provider.twist_queries.load(Ordering::Relaxed), 0);

    provider.reset_ledger(0);
    provider.fail_fs_on.store(2, Ordering::Relaxed);
    assert!(matches!(
        source.flip(&[0]),
        Err(GenericTensorError::Plan(CheckedGenericPlanError::Provider(
            PivotalError::FrobeniusSchur
        )))
    ));
    assert_eq!(source.data(), before);

    provider.reset_ledger(0);
    provider.fail_twist_on.store(2, Ordering::Relaxed);
    assert!(matches!(
        source.flip(&[0]),
        Err(GenericTensorError::Plan(CheckedGenericPlanError::Provider(
            PivotalError::Twist
        )))
    ));
    assert_eq!(source.data(), before);
}

#[test]
fn checked_generic_flip_uses_staged_nontrivial_fs_and_twist_factors() {
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let provider = Arc::new(CheckedPivotalToy::with_fs(
        9,
        BraidingStyleKind::Anyonic,
        -1.0,
        1.0,
    ));
    let x_dual = GradedSpace::try_new_with_arc(Arc::clone(&provider), [(Label::X, 1)])
        .and_then(|space| space.try_dual())
        .unwrap();
    let source: TensorMap<_, f64> =
        TensorMap::from_block_fn(&runtime, [&x_dual], [&x_dual], |_, _| 3.0).unwrap();

    let codomain = source.flip(&[0]).unwrap();
    let domain = source.flip(&[1]).unwrap();
    assert_eq!(codomain.data(), &[-3.0]);
    assert_eq!(domain.data(), &[-3.0]);
    let roundtrip = codomain.flip_inverse(&[0]).unwrap();
    assert_eq!(roundtrip.data(), source.data());
    assert_eq!(roundtrip.codomain(), source.codomain());
    assert_eq!(roundtrip.domain(), source.domain());
}

#[cfg(feature = "racah-generated")]
fn assert_sun_identity_case<D>(n: usize, label: Vec<i64>, value: impl Fn(usize) -> D)
where
    D: TensorScalar + fmt::Debug + PartialEq,
{
    use tenet::typed::SUNFusionRule;

    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let provider = Arc::new(SUNFusionRule::new(n).unwrap());
    let legs = [2, 1, 2].map(|degeneracy| {
        GradedSpace::try_new_with_arc(Arc::clone(&provider), [(label.clone(), degeneracy)]).unwrap()
    });
    let source: TensorMap<_, D> = TensorMap::from_block_fn(
        &runtime,
        [&legs[0], &legs[1], &legs[2]],
        [],
        |trees, indices| {
            value(100 * trees.codomain_vertices()[0].get() + indices.iter().sum::<usize>())
        },
    )
    .unwrap();
    let twisted = source.twist(&[0, 1, 2]).unwrap();
    assert_eq!(twisted.data().as_ptr(), source.data().as_ptr());
    assert_eq!(twisted.data(), source.data());
    assert!(std::ptr::eq(twisted.provider(), provider.as_ref()));
    assert_eq!(twisted.codomain(), source.codomain());
    assert_eq!(twisted.domain(), source.domain());
    assert_eq!(twisted.block_count(), source.block_count());
    for index in 0..source.block_count() {
        let before = source.block(index).unwrap();
        let after = twisted.block(index).unwrap();
        assert_eq!(after.key(), before.key());
        assert_eq!(after.shape(), before.shape());
        assert_eq!(after.strides(), before.strides());
        assert_eq!(after.offset(), before.offset());
    }
}

#[cfg(feature = "racah-generated")]
#[test]
fn sun_checked_generic_twist_identity_shares_exact_provider_and_layout() {
    // What: the checked SU(3)/SU(4) pivotal implementation is θ=1, so all four
    // provider/dtype cases share payload and preserve the admitted layout.
    assert_sun_identity_case(3, vec![2, 2], |value| value as f64);
    assert_sun_identity_case(3, vec![2, 2], |value| {
        Complex64::new(value as f64, -(value as f64))
    });
    assert_sun_identity_case(4, vec![2, 0, 2], |value| value as f64);
    assert_sun_identity_case(4, vec![2, 0, 2], |value| {
        Complex64::new(value as f64, -(value as f64))
    });
}

#[cfg(feature = "racah-generated")]
fn assert_sun_flip_case<D>(n: usize, label: Vec<i64>, value: impl Fn(usize) -> D)
where
    D: TensorScalar + fmt::Debug + PartialEq,
{
    use tenet::typed::SUNFusionRule;

    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let provider = Arc::new(SUNFusionRule::new(n).unwrap());
    let legs = [2, 1, 2].map(|degeneracy| {
        GradedSpace::try_new_with_arc(Arc::clone(&provider), [(label.clone(), degeneracy)]).unwrap()
    });
    let source: TensorMap<_, D> = TensorMap::from_block_fn(
        &runtime,
        [&legs[0], &legs[1], &legs[2]],
        [],
        |trees, indices| {
            value(100 * trees.codomain_vertices()[0].get() + indices.iter().sum::<usize>())
        },
    )
    .unwrap();
    let flipped = source.flip(&[0, 2]).unwrap();
    assert!(std::ptr::eq(flipped.provider(), provider.as_ref()));
    assert!(flipped.codomain()[0].is_dual());
    assert!(flipped.codomain()[2].is_dual());
    assert_eq!(flipped.data(), source.data());
    let mut saw_vertex_two = false;
    for index in 0..source.block_count() {
        let before = source.block(index).unwrap();
        let after = flipped.block(index).unwrap();
        let before_trees = source.block_fusion_trees(index).unwrap();
        let after_trees = flipped.block_fusion_trees(index).unwrap();
        saw_vertex_two |= before_trees
            .codomain_vertices()
            .iter()
            .any(|vertex| vertex.get() > 1);
        assert_eq!(after_trees, before_trees);
        assert_eq!(after.shape(), before.shape());
        assert_eq!(after.strides(), before.strides());
        assert_eq!(after.offset(), before.offset());
    }
    assert!(saw_vertex_two);
    let roundtrip = flipped.flip_inverse(&[0, 2]).unwrap();
    assert_eq!(roundtrip.data(), source.data());
    assert_eq!(roundtrip.codomain(), source.codomain());
}

#[cfg(feature = "racah-generated")]
#[test]
fn sun_checked_generic_flip_preserves_full_key_layout_and_inverse_roundtrip() {
    assert_sun_flip_case(3, vec![2, 2], |value| value as f64);
    assert_sun_flip_case(3, vec![2, 2], |value| {
        Complex64::new(value as f64, -(value as f64))
    });
    assert_sun_flip_case(4, vec![2, 0, 2], |value| value as f64);
    assert_sun_flip_case(4, vec![2, 0, 2], |value| {
        Complex64::new(value as f64, -(value as f64))
    });
}
