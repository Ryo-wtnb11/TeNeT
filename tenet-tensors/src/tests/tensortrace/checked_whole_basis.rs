//! Checked Generic trace F3 (#2054): first-provider-error order, query and
//! validation counts, on the actual checked SU(3) provider behind a
//! recording, key-selective failure wrapper.

use super::*;
use std::cell::{Cell, RefCell};
use tenet_core::{
    BraidingStyleKind, CheckedGenericFusion, CheckedGenericPivotal, CheckedGenericRigidSymbols,
    FusionStyleKind, GenericFArray, GenericRMatrix, RuleIdentity, SUNFusionRule,
    SUNFusionRuleError, SectorVec,
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Call {
    Dual(SectorId),
    Channels(SectorId, SectorId),
    N(SectorId, SectorId, SectorId),
    Dim(SectorId),
    SqrtDim(SectorId),
    InvSqrtDim(SectorId),
    Fs(SectorId),
    F([SectorId; 6]),
    R([SectorId; 3]),
    Twist(SectorId),
}

#[derive(Debug)]
enum LedgerError {
    Injected(Call),
    Delegated(SUNFusionRuleError),
}
impl std::fmt::Display for LedgerError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Injected(call) => write!(f, "injected failure at {call:?}"),
            Self::Delegated(error) => std::fmt::Display::fmt(error, f),
        }
    }
}
impl std::error::Error for LedgerError {}

/// Records every provider query in order and fails exactly the queries in
/// `fail` plus every F or R symbol query that names `fail_symbols_with`;
/// every other answer is the checked SU(N) provider's.
///
/// Its identity is renewed whenever its answers change, so a cache-4 entry
/// built under one failure set is never a hit under another (equal
/// identity must mean equal answers).
struct Ledger {
    inner: SUNFusionRule,
    identity: RefCell<RuleIdentity>,
    calls: RefCell<Vec<Call>>,
    fail: RefCell<Vec<Call>>,
    fail_symbols_with: Cell<Option<SectorId>>,
}

impl Ledger {
    fn query<T>(
        &self,
        call: Call,
        answer: impl FnOnce(&SUNFusionRule) -> Result<T, SUNFusionRuleError>,
    ) -> Result<T, LedgerError> {
        self.calls.borrow_mut().push(call);
        let names_failing_sector = match (call, self.fail_symbols_with.get()) {
            (Call::F(sectors), Some(sector)) => sectors.contains(&sector),
            (Call::R(sectors), Some(sector)) => sectors.contains(&sector),
            _ => false,
        };
        if names_failing_sector || self.fail.borrow().contains(&call) {
            return Err(LedgerError::Injected(call));
        }
        answer(&self.inner).map_err(LedgerError::Delegated)
    }

    fn reset(&self, fail: Vec<Call>) {
        self.calls.borrow_mut().clear();
        *self.fail.borrow_mut() = fail;
        self.fail_symbols_with(None);
    }

    fn fail_symbols_with(&self, sector: Option<SectorId>) {
        self.fail_symbols_with.set(sector);
        *self.identity.borrow_mut() = RuleIdentity::new_unique::<Self>();
    }
}

impl CheckedGenericFusion for Ledger {
    type Error = LedgerError;
    fn rule_identity(&self) -> RuleIdentity {
        self.identity.borrow().clone()
    }
    fn fusion_style(&self) -> FusionStyleKind {
        self.inner.fusion_style()
    }
    fn braiding_style(&self) -> BraidingStyleKind {
        self.inner.braiding_style()
    }
    fn vacuum(&self) -> SectorId {
        self.inner.vacuum()
    }
    fn try_dual(&self, a: SectorId) -> Result<SectorId, Self::Error> {
        self.query(Call::Dual(a), |p| p.try_dual(a))
    }
    fn try_fusion_channels(&self, a: SectorId, b: SectorId) -> Result<SectorVec, Self::Error> {
        self.query(Call::Channels(a, b), |p| p.try_fusion_channels(a, b))
    }
    fn try_fusion_channels_in_table(
        &self,
        a: SectorId,
        b: SectorId,
    ) -> Result<SectorVec, Self::Error> {
        self.query(Call::Channels(a, b), |p| {
            p.try_fusion_channels_in_table(a, b)
        })
    }
    fn try_nsymbol(&self, a: SectorId, b: SectorId, c: SectorId) -> Result<usize, Self::Error> {
        self.query(Call::N(a, b, c), |p| p.try_nsymbol(a, b, c))
    }
}

impl CheckedGenericRigidSymbols for Ledger {
    type Scalar = f64;
    fn try_dim_scalar(&self, a: SectorId) -> Result<f64, Self::Error> {
        self.query(Call::Dim(a), |p| p.try_dim_scalar(a))
    }
    fn try_sqrt_dim_scalar(&self, a: SectorId) -> Result<f64, Self::Error> {
        self.query(Call::SqrtDim(a), |p| p.try_sqrt_dim_scalar(a))
    }
    fn try_inv_sqrt_dim_scalar(&self, a: SectorId) -> Result<f64, Self::Error> {
        self.query(Call::InvSqrtDim(a), |p| p.try_inv_sqrt_dim_scalar(a))
    }
    fn try_frobenius_schur_phase_scalar(&self, a: SectorId) -> Result<f64, Self::Error> {
        self.query(Call::Fs(a), |p| p.try_frobenius_schur_phase_scalar(a))
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
        self.query(Call::F([a, b, c, d, e, f]), |p| {
            p.try_f_symbol_generic(a, b, c, d, e, f)
        })
    }
    fn try_r_symbol_generic(
        &self,
        a: SectorId,
        b: SectorId,
        c: SectorId,
    ) -> Result<GenericRMatrix<f64>, Self::Error> {
        self.query(Call::R([a, b, c]), |p| p.try_r_symbol_generic(a, b, c))
    }
}

impl CheckedGenericPivotal for Ledger {
    fn try_twist_scalar(&self, a: SectorId) -> Result<f64, Self::Error> {
        self.query(Call::Twist(a), |p| p.try_twist_scalar(a))
    }
}

struct Fixture {
    provider: Arc<Ledger>,
    src: BoundDynamicFusionMapSpace<Ledger>,
    dst: BoundDynamicFusionMapSpace<Ledger>,
    vacuum: SectorId,
    adjoint: SectorId,
    fundamental: SectorId,
}

const OUTPUT: [usize; 2] = [1, 3];
const LHS: [usize; 1] = [0];
const RHS: [usize; 1] = [2];

fn axes() -> TensorTraceAxisSpec<'static> {
    TensorTraceAxisSpec::new(&OUTPUT, &LHS, &RHS)
}

/// SU(3) `V ⊗ A ← V ⊗ A` with `V = 1 ⊕ 8² ⊕ 3` and `A = 8`, traced over
/// `(0, 2)`: the `(p…, q…)` permutation `[1, 0] ← [3, 2]` braids both trees.
/// The source has five fusion groups; the `3 ⊗ 8` group shares no sector
/// with the others except `8`, so its admission and recoupling queries are
/// its own.
#[allow(clippy::arc_with_non_send_sync)]
fn fixture() -> Fixture {
    let inner = SUNFusionRule::new(3).unwrap();
    let vacuum = inner.encode_dynkin(&[0, 0]).unwrap();
    let adjoint = inner.encode_dynkin(&[1, 1]).unwrap();
    let fundamental = inner.encode_dynkin(&[1, 0]).unwrap();
    let provider = Arc::new(Ledger {
        inner,
        identity: RefCell::new(RuleIdentity::new_unique::<Ledger>()),
        calls: RefCell::new(Vec::new()),
        fail: RefCell::new(Vec::new()),
        fail_symbols_with: Cell::new(None),
    });
    let v = || SectorLeg::new([(vacuum, 1), (adjoint, 2), (fundamental, 1)], false);
    let a = || SectorLeg::new([(adjoint, 1)], false);
    let src = BoundDynamicFusionMapSpace::from_final_homspace_generic_checked(
        Arc::clone(&provider),
        FusionTreeHomSpace::new(
            FusionProductSpace::new([v(), a()]),
            FusionProductSpace::new([v(), a()]),
        ),
    )
    .unwrap();
    let preflight =
        crate::tensortrace_fusion_dyn_preflight_generic_checked(&src, axes(), 1).unwrap();
    let dst = BoundDynamicFusionMapSpace::from_final_homspace_generic_checked(
        Arc::clone(&provider),
        preflight.into_selected_homspace(),
    )
    .unwrap();
    Fixture {
        provider,
        src,
        dst,
        vacuum,
        adjoint,
        fundamental,
    }
}

fn compile(
    fixture: &Fixture,
) -> Result<TensorTraceFusionStructure<f64>, CheckedGenericPlanError<LedgerError>> {
    // Publishes nothing: each call of this helper is cold.
    crate::tensortrace::compile_fusion_dyn_generic_checked(
        &fixture.dst,
        &fixture.src,
        axes(),
        &mut crate::tree_transform::CheckedPendingCoefficients::new(),
    )
}

/// The first provider error of a compile that fails `fail` and, when
/// given, every F/R query naming `symbols_with`.
fn first_error(fixture: &Fixture, fail: Vec<Call>, symbols_with: Option<SectorId>) -> Call {
    fixture.provider.reset(fail);
    fixture.provider.fail_symbols_with(symbols_with);
    match compile(fixture) {
        Err(CheckedGenericPlanError::Provider(LedgerError::Injected(call))) => call,
        other => panic!("expected an injected provider error, got {other:?}"),
    }
}

fn group_of(structure: &BlockStructure, block: usize) -> &[usize] {
    structure
        .fusion_tree_group_slice()
        .iter()
        .map(|group| group.block_indices())
        .find(|indices| indices.contains(&block))
        .unwrap()
}

fn source_key(structure: &BlockStructure, block: usize) -> &FusionTreePairKey {
    let BlockKey::FusionTree(key) = structure.block(block).unwrap().key() else {
        unreachable!("fusion-tree source")
    };
    key
}

/// The first block of the `3 ⊗ 8` group.
fn fundamental_group_start(fixture: &Fixture) -> usize {
    let structure = fixture.src.space().structure();
    (0..structure.block_count())
        .find(|&block| {
            source_key(structure, block).codomain_tree().uncoupled()[0] == fixture.fundamental
        })
        .unwrap()
}

/// The pins rely on this shape: block 0 (coupled vacuum) opens the `8 ⊗ 8`
/// group `G0`, and the `3 ⊗ 8` group opens before `G0`'s second member, so
/// the groups interleave in block order.
#[test]
fn fixture_groups_interleave_in_block_order() {
    let fixture = fixture();
    let structure = fixture.src.space().structure();
    let first = group_of(structure, 0);
    assert!(first.len() > 2, "{first:?}");
    assert_eq!(
        source_key(structure, 0).codomain_tree().coupled(),
        fixture.vacuum
    );
    let start = fundamental_group_start(&fixture);
    assert_eq!(group_of(structure, start)[0], start);
    assert!(group_of(structure, start).len() > 1);
    assert!(start < first[1], "{start} vs {first:?}");
    assert_eq!(structure.fusion_tree_group_slice().len(), 5);
}

/// The sector-keyed failures the order pins combine: the twist is queried
/// only by a trace channel factor (lowering), and first by block 0;
/// `Channels(3, 8)` only by admitting a `3 ⊗ 8` source tree; `R(8, 8, 8)`
/// only by braiding an `8 ⊗ 8 → 8` vertex of `G0`; `N(8, 8, c)` with
/// `c ∉ {1, 8}` only by admitting an `8 ⊗ 8 → c` vertex of `G0`.
fn keys(fixture: &Fixture) -> [Call; 4] {
    let (v, a, f) = (fixture.vacuum, fixture.adjoint, fixture.fundamental);
    let structure = fixture.src.space().structure();
    let other = group_of(structure, 0)
        .iter()
        .map(|&block| source_key(structure, block).codomain_tree().coupled())
        .find(|&coupled| coupled != v && coupled != a)
        .unwrap();
    [
        Call::Twist(a),
        Call::Channels(f, a),
        Call::R([a, a, a]),
        Call::N(a, a, other),
    ]
}

#[test]
fn each_injected_query_is_reached_alone() {
    let fixture = fixture();
    for key in keys(&fixture) {
        assert_eq!(first_error(&fixture, vec![key], None), key);
    }
    let symbol = first_error(&fixture, Vec::new(), Some(fixture.fundamental));
    assert!(matches!(symbol, Call::F(_) | Call::R(_)), "{symbol:?}");
}

/// What: a source's lowering error (its trace channel factor) is reported
/// before any error of a fusion group whose first member comes later in
/// block order, whether that group fails at admission or at recoupling.
/// This is the cross-group order Option A of #2054 keeps.
#[test]
fn lowering_error_precedes_later_group_admission_and_recoupling() {
    let fixture = fixture();
    let [twist, later_admission, ..] = keys(&fixture);
    assert_eq!(
        first_error(&fixture, vec![twist, later_admission], None),
        twist
    );
    assert_eq!(
        first_error(&fixture, vec![twist], Some(fixture.fundamental)),
        twist
    );
}

/// What: the compiled checked trace terms equal an eager per-source
/// composition: each source permuted alone by the checked per-pair permute
/// (TensorKit's pair path), split at the open ranks, kept where the traced
/// trees agree, scaled by `dim(c)/dim(first)·Π twist(non-dual)`, summed per
/// (destination block, source block, destination key).
#[test]
fn checked_trace_terms_match_eager_per_source_composition() {
    let fixture = fixture();
    let provider = &*fixture.provider;
    provider.reset(Vec::new());
    let structure = compile(&fixture).unwrap();
    type Key = (usize, usize, FusionTreePairKey);
    let mut got: Vec<(Key, f64)> = Vec::new();
    let add = |entries: &mut Vec<(Key, f64)>, key: Key, value: f64| match entries
        .iter_mut()
        .find(|(k, _)| *k == key)
    {
        Some((_, sum)) => *sum += value,
        None => entries.push((key, value)),
    };
    for term in structure.terms() {
        add(
            &mut got,
            (term.dst_block(), term.src_block(), term.dst_key().clone()),
            *term.coefficient(),
        );
    }

    let src = fixture.src.space().structure();
    let dst = fixture.dst.space().structure();
    let mut want: Vec<(Key, f64)> = Vec::new();
    let mut rows_seen = 0;
    for block in 0..src.block_count() {
        let BlockKey::FusionTree(key) = src.block(block).unwrap().key() else {
            unreachable!()
        };
        let rows =
            tenet_core::generic_permute_tree_pair_checked(provider, key, &[1, 0], &[3, 2]).unwrap();
        rows_seen += rows.len();
        for (permuted, coefficient) in rows {
            let (open_codomain, traced_codomain) = tenet_core::split_fusion_tree_generic_checked(
                provider,
                permuted.codomain_tree(),
                1,
            )
            .unwrap();
            let (open_domain, traced_domain) =
                tenet_core::split_fusion_tree_generic_checked(provider, permuted.domain_tree(), 1)
                    .unwrap();
            if traced_codomain != traced_domain {
                continue;
            }
            let inner = &provider.inner;
            let first = traced_codomain.uncoupled()[0];
            let mut factor = inner.try_dim_scalar(traced_codomain.coupled()).unwrap()
                / inner.try_dim_scalar(first).unwrap();
            for (&sector, &dual) in traced_codomain
                .uncoupled()
                .iter()
                .zip(traced_codomain.is_dual())
                .skip(1)
            {
                if !dual {
                    factor *= inner.try_twist_scalar(sector).unwrap();
                }
            }
            let dst_key = FusionTreePairKey::pair(open_codomain, open_domain);
            let dst_block = dst.find_block_index_by_fusion_tree_pair(&dst_key).unwrap();
            add(&mut want, (dst_block, block, dst_key), coefficient * factor);
        }
    }
    assert!(
        rows_seen > src.block_count(),
        "some source must reach several rows"
    );
    assert_eq!(got.len(), want.len());
    for (key, value) in &want {
        let actual = got
            .iter()
            .find(|(k, _)| k == key)
            .unwrap_or_else(|| panic!("missing term {key:?}"))
            .1;
        crate::test_numerics::numerics::assert_close("trace term", actual, *value, 16);
    }
}

/// What (#2054, Option A): a fusion group is admitted and recoupled whole
/// before its first member is lowered, as TensorKit `_trace_permute!`
/// builds one permutation matrix per fusion block before any trace
/// lowering. Inside one group, a later member's admission or recoupling
/// error therefore precedes an earlier member's lowering error; the
/// cross-group order is pinned above.
#[test]
fn group_admission_and_recoupling_precede_its_first_members_lowering() {
    let fixture = fixture();
    let [twist, _, recoupling, admission] = keys(&fixture);
    assert_eq!(
        first_error(&fixture, vec![twist, recoupling], None),
        recoupling
    );
    assert_eq!(
        first_error(&fixture, vec![twist, admission], None),
        admission
    );
}

/// What: each fusion group is permuted once, at its first member; the
/// first trace provider queries after the preflight are one checked
/// validation of each member of block 0's group, in block order, and the
/// group is not validated again before its recoupling.
#[test]
fn group_admission_validates_each_member_once_in_block_order() {
    let fixture = fixture();
    let provider = &*fixture.provider;
    provider.reset(Vec::new());
    crate::tensortrace_fusion_dyn_preflight_generic_checked(&fixture.src, axes(), 1).unwrap();
    let preflight = provider.calls.borrow().len();

    let structure = fixture.src.space().structure();
    let mut per_member = Vec::new();
    for &block in group_of(structure, 0) {
        provider.reset(Vec::new());
        tenet_core::validate_generic_fusion_tree_pair_checked(
            provider,
            source_key(structure, block),
        )
        .unwrap();
        per_member.push(provider.calls.borrow().clone());
    }
    let admission = per_member.concat();

    provider.reset(Vec::new());
    crate::tensortrace::reset_trace_transform_invocations();
    compile(&fixture).unwrap();
    // One whole-group permute per fusion group, at its first member.
    let mut openers = structure
        .fusion_tree_group_slice()
        .iter()
        .map(|group| group.block_indices()[0])
        .collect::<Vec<_>>();
    openers.sort_unstable();
    assert_eq!(crate::tensortrace::take_trace_transform_sources(), openers);
    let calls = provider.calls.borrow();
    let after = preflight + admission.len();
    assert_eq!(calls[preflight..after], admission[..]);
    assert_ne!(calls[after..after + per_member[0].len()], per_member[0][..]);
}

/// What: a cold checked compile makes exactly the provider queries, in the
/// same order, and produces the same term sequence as a04e59c2 (before
/// #2072): routing through cache 4 changes nothing on a miss. The ledger is
/// exact; term values go through `portable` (racah SU(3) coefficients round
/// their last bits differently on Linux and macOS).
#[test]
fn cold_compile_ledger_and_term_bits_match_pinned_revision() {
    let fixture = fixture();
    fixture.provider.reset(Vec::new());
    let structure = compile(&fixture).unwrap();
    let calls = fixture.provider.calls.borrow();
    let ledger = super::trace_cache4::fingerprint(format!("{:?}", *calls).bytes().map(u64::from));
    let terms = super::trace_cache4::fingerprint(structure.terms().iter().flat_map(|term| {
        [
            term.dst_block() as u64,
            term.src_block() as u64,
            super::trace_cache4::portable(*term.coefficient()),
        ]
    }));
    assert_eq!(
        (calls.len(), ledger, terms),
        (2031, 12_939_473_490_702_431_790, 14_775_106_432_576_145_088)
    );
}

/// The compile-only checked trace that publishes on success
/// (`PivotalCoefficientAlgebra::trace_terms`).
fn publishing_compile(
    fixture: &Fixture,
) -> Result<TensorTraceFusionStructure<f64>, CheckedGenericPlanError<LedgerError>> {
    <tenet_core::CheckedGenericAdmissionMode as crate::PivotalCoefficientAlgebra<Ledger>>::trace_terms(
        &fixture.dst,
        &fixture.src,
        axes(),
    )
}

fn trace_activity() -> (usize, usize, usize) {
    let activity = crate::tree_transform::take_trace_column_activity();
    (activity.hits, activity.misses, activity.publications)
}

fn sorted(mut calls: Vec<Call>) -> Vec<Call> {
    calls.sort_by_key(|call| format!("{call:?}"));
    calls
}

/// What (#2072): a warm checked compile hits every fusion group and makes
/// none of the groups' admission or recoupling queries. Its ledger is the
/// cold ledger less, as a multiset, exactly what composing each group alone
/// asks (member validation, then F/R): the preflight and lowering queries
/// remain, no F or R is asked, and the term bits equal the cold ones.
#[test]
fn warm_compile_skips_exactly_group_admission_and_recoupling() {
    let _guard = crate::test_support::CACHE_TEST_LOCK
        .lock()
        .unwrap_or_else(|error| error.into_inner());
    let fixture = fixture();
    let provider = &*fixture.provider;
    let structure = fixture.src.space().structure();
    let groups = structure.fusion_tree_group_slice().len();
    provider.reset(Vec::new());
    trace_activity();
    let cold = publishing_compile(&fixture).unwrap();
    let cold_calls = provider.calls.take();
    assert_eq!(trace_activity(), (0, groups, groups));
    let warm = publishing_compile(&fixture).unwrap();
    let warm_calls = provider.calls.take();
    assert_eq!(trace_activity(), (groups, 0, 0));

    let bits = |structure: &TensorTraceFusionStructure<f64>| {
        structure
            .terms()
            .iter()
            .map(|term| {
                (
                    term.dst_block(),
                    term.src_block(),
                    term.coefficient().to_bits(),
                )
            })
            .collect::<Vec<_>>()
    };
    assert_eq!(bits(&cold), bits(&warm));
    assert!(!warm_calls
        .iter()
        .any(|call| matches!(call, Call::F(_) | Call::R(_))));
    let mut skipped = Vec::new();
    for group in structure.fusion_tree_group_slice() {
        tenet_core::generic_permute_tree_pair_block_indexed_checked(
            provider,
            structure,
            group.block_indices(),
            tenet_core::FusionTreePairOrientation::Direct,
            &[1, 0],
            &[3, 2],
        )
        .unwrap();
        skipped.extend(provider.calls.take());
    }
    assert!(skipped.iter().any(|call| matches!(call, Call::F(_))));
    let mut expected = warm_calls;
    expected.extend(skipped);
    assert_eq!(sorted(cold_calls), sorted(expected));
}

/// What (#2072): a failing call publishes no group, whether it fails in a
/// later group's admission or recoupling (after earlier groups were built)
/// or in lowering, so repeating it asks the provider exactly the same
/// queries and fails the same way.
#[test]
fn failing_compile_publishes_nothing_and_repeats_its_ledger() {
    let _guard = crate::test_support::CACHE_TEST_LOCK
        .lock()
        .unwrap_or_else(|error| error.into_inner());
    let fixture = fixture();
    let provider = &*fixture.provider;
    let [twist, later_admission, recoupling, admission] = keys(&fixture);
    for fail in [twist, later_admission, recoupling, admission] {
        provider.reset(vec![fail]);
        trace_activity();
        let first = format!("{:?}", publishing_compile(&fixture).unwrap_err());
        let first_calls = provider.calls.take();
        let (_, misses, publications) = trace_activity();
        assert!(misses > 0, "{fail:?}");
        assert_eq!(publications, 0, "{fail:?}");
        let again = format!("{:?}", publishing_compile(&fixture).unwrap_err());
        assert_eq!(again, first, "{fail:?}");
        assert_eq!(provider.calls.take(), first_calls, "{fail:?}");
        assert_eq!(trace_activity().2, 0, "{fail:?}");
    }
}

/// What (#2072): the owned checked trace publishes its groups only once
/// execution returned `Ok`; a lowering failure publishes none, and the
/// following successful call is cold, then warm.
#[test]
fn owned_checked_trace_publishes_after_success_only() {
    let _guard = crate::test_support::CACHE_TEST_LOCK
        .lock()
        .unwrap_or_else(|error| error.into_inner());
    let fixture = fixture();
    let groups = fixture
        .src
        .space()
        .structure()
        .fusion_tree_group_slice()
        .len();
    let [twist, ..] = keys(&fixture);
    let data = vec![1.0_f64; fixture.src.space().required_len().unwrap()];
    let run = || {
        crate::tensortrace_fusion_dyn_owned_generic_checked(
            &fixture.dst,
            &fixture.src,
            &data,
            axes(),
            1.0,
        )
    };
    fixture.provider.reset(vec![twist]);
    trace_activity();
    assert!(run().is_err());
    assert_eq!(trace_activity().2, 0);
    fixture.provider.reset(Vec::new());
    let cold = run().unwrap();
    assert_eq!(trace_activity(), (0, groups, groups));
    let warm = run().unwrap();
    assert_eq!(trace_activity(), (groups, 0, 0));
    assert_eq!(
        cold.iter().map(|value| value.to_bits()).collect::<Vec<_>>(),
        warm.iter().map(|value| value.to_bits()).collect::<Vec<_>>()
    );
}
