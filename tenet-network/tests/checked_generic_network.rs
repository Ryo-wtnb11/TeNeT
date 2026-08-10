#![cfg(feature = "racah-generated")]

use std::collections::HashMap;
use std::fmt::{self, Debug};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;

#[cfg(feature = "opt-path")]
use tenet::core::CheckedGenericStructureError;
use tenet::core::{
    BraidingStyleKind, CheckedGenericAdmissionMode, CheckedGenericFusion, CheckedGenericPivotal,
    CheckedGenericRigidSymbols, FusionStyleKind, GenericFArray, GenericRMatrix, RuleIdentity,
    SectorId, SectorVec, TypedSectorAdmission,
};
use tenet::prelude::{Complex64, Error, Runtime, TensorScalar};
use tenet::typed::{
    BlockFusionTrees, CheckedGenericPlanError, GenericTensorError, GradedSpace, SUNFusionRule,
    TensorMap,
};
#[cfg(feature = "opt-path")]
use tenet_network::Optimizer;
use tenet_network::{
    configure_plan_cache, plan_cache_stats, slice_plan_for, tensor, DenseCostModel,
    DenseTensorInfo, GreedyDenseOptimizer, LabelOrderDenseOptimizer, Network,
    NetworkExecutionWorkspace, NetworkIR, PlanCacheConfig, PlannedNetwork, SlicedPlan,
    TemporaryLabel, TensorId,
};

fn labels(names: &[&str]) -> Vec<TemporaryLabel> {
    names.iter().copied().map(TemporaryLabel::from).collect()
}

trait OracleScalar: TensorScalar + Copy + Debug {
    fn value(marker: usize) -> Self;
    fn distance(self, other: Self) -> f64;
}

impl OracleScalar for f64 {
    fn value(marker: usize) -> Self {
        marker as f64 / 17.0
    }

    fn distance(self, other: Self) -> f64 {
        (self - other).abs()
    }
}

impl OracleScalar for Complex64 {
    fn value(marker: usize) -> Self {
        Self::new(marker as f64 / 17.0, -(marker as f64) / 23.0)
    }

    fn distance(self, other: Self) -> f64 {
        (self - other).norm()
    }
}

fn marker(trees: &BlockFusionTrees<Vec<i64>>, indices: &[usize]) -> usize {
    trees
        .codomain_vertices()
        .iter()
        .chain(trees.domain_vertices())
        .enumerate()
        .map(|(index, vertex)| (index + 1) * 100 * vertex.get())
        .chain(
            indices
                .iter()
                .enumerate()
                .map(|(index, value)| (index + 1) * value),
        )
        .sum()
}

fn assert_same<D: OracleScalar>(
    actual: &TensorMap<SUNFusionRule, D>,
    expected: &TensorMap<SUNFusionRule, D>,
) {
    assert_eq!(actual.codomain(), expected.codomain());
    assert_eq!(actual.domain(), expected.domain());
    assert_eq!(actual.block_count(), expected.block_count());
    for index in 0..actual.block_count() {
        assert_eq!(
            actual.block_fusion_trees(index).unwrap(),
            expected.block_fusion_trees(index).unwrap()
        );
        assert_eq!(actual.block(index).unwrap(), expected.block(index).unwrap());
    }
    assert_eq!(actual.data().len(), expected.data().len());
    for (index, (&lhs, &rhs)) in actual.data().iter().zip(expected.data()).enumerate() {
        assert!(
            lhs.distance(rhs) <= 1.0e-10 * (1.0 + lhs.distance(D::value(0))),
            "payload {index} differs: {lhs:?} vs {rhs:?}"
        );
    }
}

fn authority_provider<D>(
    planned: &PlannedNetwork,
    tensors: &[&TensorMap<SUNFusionRule, D>],
) -> *const SUNFusionRule {
    let mut authorities = tensors
        .iter()
        .enumerate()
        .map(|(index, tensor)| (TensorId::new(index), tensor.provider() as *const _))
        .collect::<HashMap<_, _>>();
    for step in planned.plan().steps() {
        let authority = authorities[&step.lhs()];
        authorities.remove(&step.lhs());
        authorities.remove(&step.rhs());
        authorities.insert(step.result(), authority);
    }
    authorities[&planned.plan().steps().last().unwrap().result()]
}

fn assert_sun_network<D: OracleScalar + Send + Sync + 'static>(n: usize, label: Vec<i64>) {
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let lhs_provider = Arc::new(SUNFusionRule::new(n).unwrap());
    let rhs_provider = Arc::new(SUNFusionRule::new(n).unwrap());
    let tail_provider = Arc::new(SUNFusionRule::new(n).unwrap());
    let lhs_leg =
        GradedSpace::try_new_with_arc(Arc::clone(&lhs_provider), [(label.clone(), 1)]).unwrap();
    let rhs_leg =
        GradedSpace::try_new_with_arc(Arc::clone(&rhs_provider), [(label.clone(), 1)]).unwrap();
    let tail_leg =
        GradedSpace::try_new_with_arc(Arc::clone(&tail_provider), [(label.clone(), 1)]).unwrap();
    let lhs: TensorMap<_, D> = TensorMap::from_block_fn(
        &runtime,
        [&lhs_leg, &lhs_leg],
        [&lhs_leg],
        |trees, indices| D::value(10_000 + marker(trees, indices)),
    )
    .unwrap();
    assert!((0..lhs.block_count()).any(|index| {
        lhs.block_fusion_trees(index)
            .unwrap()
            .codomain_vertices()
            .iter()
            .any(|vertex| vertex.get() == 2)
    }));
    // What: the trace-only macro entrypoint matches the ordinary checked trace
    // for a tensor that contains a nontrivial multiplicity vertex. Repeated
    // calls reuse only the reduced plan and workspace.
    let direct_trace = lhs.trace_pairs(&[(0, 2)]).unwrap();
    let before_trace_macro = plan_cache_stats(&runtime);
    let macro_trace = tensor!([b;] = lhs[a, b; a]).unwrap();
    let after_first_trace_macro = plan_cache_stats(&runtime);
    let macro_trace_replay = tensor!([b;] = lhs[a, b; a]).unwrap();
    let after_trace_macro_replay = plan_cache_stats(&runtime);
    assert_eq!(
        after_first_trace_macro.entries,
        before_trace_macro.entries + 1
    );
    assert_eq!(
        after_trace_macro_replay.entries,
        after_first_trace_macro.entries
    );
    assert!(after_trace_macro_replay.hits > after_first_trace_macro.hits);
    assert_eq!(
        after_trace_macro_replay.workspaces_created,
        after_first_trace_macro.workspaces_created
    );
    assert_eq!(
        direct_trace.provider() as *const _,
        lhs.provider() as *const _
    );
    assert_eq!(
        macro_trace.provider() as *const _,
        lhs.provider() as *const _
    );
    assert_eq!(
        macro_trace_replay.provider() as *const _,
        lhs.provider() as *const _
    );
    assert_same(&macro_trace, &direct_trace);
    assert_same(&macro_trace_replay, &direct_trace);
    let middle: TensorMap<_, D> =
        TensorMap::from_block_fn(&runtime, [&rhs_leg], [&rhs_leg], |_, _| D::value(17)).unwrap();
    let tail: TensorMap<_, D> =
        TensorMap::from_block_fn(&runtime, [&tail_leg], [&tail_leg], |_, _| D::value(17)).unwrap();
    assert!(!std::ptr::eq(lhs.provider(), middle.provider()));

    let expected = lhs
        .contract(&middle, &[2], &[0], &[0, 1, 2])
        .unwrap()
        .contract(&tail, &[2], &[0], &[0, 1, 2])
        .unwrap()
        .permute(&[1, 0], &[2])
        .unwrap();
    let network = Network::new(
        vec![
            labels(&["a", "b", "c"]),
            labels(&["c", "d"]),
            labels(&["d", "e"]),
        ],
        vec![false; 3],
        vec![Some(2), Some(1), Some(1)],
        labels(&["b", "a", "e"]),
        Some(2),
    )
    .unwrap();
    let tensors = [&lhs, &middle, &tail];
    let explicit = network
        .plan(
            &tensors,
            &LabelOrderDenseOptimizer::new(labels(&["c", "d"])),
        )
        .unwrap();
    let greedy = network.plan(&tensors, &GreedyDenseOptimizer).unwrap();
    let mut workspace = NetworkExecutionWorkspace::default();
    for planned in [&explicit, &greedy] {
        let authority = authority_provider(planned, &tensors);
        for _ in 0..2 {
            let actual = planned
                .execute_with_workspace(&tensors, &mut workspace)
                .unwrap();
            assert_eq!(actual.provider() as *const _, authority);
            assert_same(&actual, &expected);
        }
    }
    let mut separate_a = NetworkExecutionWorkspace::default();
    let mut separate_b = NetworkExecutionWorkspace::default();
    assert_same(
        &explicit
            .execute_with_workspace(&tensors, &mut separate_a)
            .unwrap(),
        &expected,
    );
    assert_same(
        &explicit
            .execute_with_workspace(&tensors, &mut separate_b)
            .unwrap(),
        &expected,
    );

    let drift_lhs_provider = Arc::new(SUNFusionRule::new(n).unwrap());
    let drift_middle_provider = Arc::new(SUNFusionRule::new(n).unwrap());
    let drift_tail_provider = Arc::new(SUNFusionRule::new(n).unwrap());
    let drift_lhs_leg =
        GradedSpace::try_new_with_arc(Arc::clone(&drift_lhs_provider), [(label.clone(), 1)])
            .unwrap();
    let drift_middle_leg =
        GradedSpace::try_new_with_arc(Arc::clone(&drift_middle_provider), [(label.clone(), 1)])
            .unwrap();
    let drift_tail_leg =
        GradedSpace::try_new_with_arc(Arc::clone(&drift_tail_provider), [(label, 1)]).unwrap();
    let drift_lhs: TensorMap<_, D> = TensorMap::from_block_fn(
        &runtime,
        [&drift_lhs_leg, &drift_lhs_leg],
        [&drift_lhs_leg],
        |trees, indices| D::value(10_000 + marker(trees, indices)),
    )
    .unwrap();
    let drift_middle: TensorMap<_, D> = TensorMap::from_block_fn(
        &runtime,
        [&drift_middle_leg],
        [&drift_middle_leg],
        |_, _| D::value(17),
    )
    .unwrap();
    let drift_tail: TensorMap<_, D> =
        TensorMap::from_block_fn(&runtime, [&drift_tail_leg], [&drift_tail_leg], |_, _| {
            D::value(17)
        })
        .unwrap();
    let drift_refs = [&drift_lhs, &drift_middle, &drift_tail];
    let drift = explicit
        .execute_with_workspace(&drift_refs, &mut separate_a)
        .unwrap();
    assert_eq!(
        drift.provider() as *const _,
        authority_provider(&explicit, &drift_refs)
    );
    assert_same(&drift, &expected);

    let macro_first = tensor!([b, a; e] = lhs[a, b; c] * middle[c; d] * tail[d; e]).unwrap();
    let macro_replay = tensor!([b, a; e] = lhs[a, b; c] * middle[c; d] * tail[d; e]).unwrap();
    assert_eq!(
        macro_first.provider() as *const _,
        authority_provider(&greedy, &tensors)
    );
    assert_same(&macro_first, &expected);
    assert_same(&macro_replay, &expected);

    let planned = Arc::new(greedy);
    let operands = [Arc::new(lhs), Arc::new(middle), Arc::new(tail)];
    let workers = (0..2)
        .map(|_| {
            let planned = Arc::clone(&planned);
            let operands = operands.clone();
            std::thread::spawn(move || {
                planned
                    .execute(&[&operands[0], &operands[1], &operands[2]])
                    .unwrap()
            })
        })
        .collect::<Vec<_>>();
    for worker in workers {
        assert_same(&worker.join().unwrap(), &expected);
    }
}

#[test]
fn sun_checked_generic_network_matches_manual_explicit_greedy_macro_and_replay() {
    // What: provider-neutral SU(N) conformance over μ=2 keys, both payload
    // dtypes, three operands, nontrivial final order, replay, and concurrency.
    for (n, label) in [(3, vec![1, 1]), (4, vec![1, 0, 1])] {
        assert_sun_network::<f64>(n, label.clone());
        assert_sun_network::<Complex64>(n, label);
    }
}

#[test]
fn sun_checked_generic_internal_slice_preserves_outer_multiplicity_keys() {
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let provider = Arc::new(SUNFusionRule::new(3).unwrap());
    let label = vec![1, 1];
    let leg = GradedSpace::try_new_with_arc(Arc::clone(&provider), [(label, 2)]).unwrap();
    let lhs: TensorMap<_, Complex64> =
        TensorMap::from_block_fn(&runtime, [&leg, &leg], [&leg], |trees, indices| {
            Complex64::value(20_000 + marker(trees, indices))
        })
        .unwrap();
    assert!((0..lhs.block_count()).any(|index| {
        lhs.block_fusion_trees(index)
            .unwrap()
            .codomain_vertices()
            .iter()
            .any(|vertex| vertex.get() == 2)
    }));
    let rhs: TensorMap<_, Complex64> =
        TensorMap::from_block_fn(&runtime, [&leg], [&leg], |trees, indices| {
            Complex64::value(30_000 + marker(trees, indices))
        })
        .unwrap();
    let inputs = vec![labels(&["a", "b", "x"]), labels(&["x", "d"])];
    let output = labels(&["a", "b", "d"]);
    let network = Network::new(
        inputs.clone(),
        vec![false; 2],
        vec![Some(2), Some(1)],
        output.clone(),
        Some(2),
    )
    .unwrap();
    let tensors = [&lhs, &rhs];
    let planned = network.plan(&tensors, &GreedyDenseOptimizer).unwrap();
    let expected = planned.execute(&tensors).unwrap();
    let ir = NetworkIR::from_labels(inputs, output).unwrap();
    let cost = DenseCostModel::from_network(
        &ir,
        &[
            DenseTensorInfo::new(vec![2, 2, 2]),
            DenseTensorInfo::new(vec![2, 2]),
        ],
    )
    .unwrap();
    let dense = SlicedPlan::new(
        planned.plan().clone(),
        slice_plan_for(&ir, planned.plan(), &cost, &labels(&["x"])),
    );
    let sliced = network
        .lower_symmetric_sliced_plan(&tensors, dense)
        .unwrap();
    let (actual, peak) = network
        .execute_symmetric_sliced(&tensors, sliced, usize::MAX)
        .unwrap();
    assert!(peak > 0);
    assert_same(&actual, &expected);
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum InjectedError {
    Encode,
    Dual,
    Symbol,
    Provider,
}

impl fmt::Display for InjectedError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "injected {self:?} failure")
    }
}

impl std::error::Error for InjectedError {}

struct InjectedGeneric {
    inner: SUNFusionRule,
    fail_encode: AtomicBool,
    fail_dual: AtomicBool,
    dual_calls: AtomicUsize,
    fail_symbol_at: AtomicUsize,
    symbol_calls: AtomicUsize,
}

impl InjectedGeneric {
    fn new() -> Self {
        Self {
            inner: SUNFusionRule::new(3).unwrap(),
            fail_encode: AtomicBool::new(false),
            fail_dual: AtomicBool::new(false),
            dual_calls: AtomicUsize::new(0),
            fail_symbol_at: AtomicUsize::new(0),
            symbol_calls: AtomicUsize::new(0),
        }
    }

    fn reset_symbols(&self) {
        self.fail_symbol_at.store(0, Ordering::SeqCst);
        self.symbol_calls.store(0, Ordering::SeqCst);
    }

    fn arm_symbol(&self, ordinal: usize) {
        self.symbol_calls.store(0, Ordering::SeqCst);
        self.fail_symbol_at.store(ordinal, Ordering::SeqCst);
    }

    fn symbol(&self) -> Result<(), InjectedError> {
        let ordinal = self.symbol_calls.fetch_add(1, Ordering::SeqCst) + 1;
        if self.fail_symbol_at.load(Ordering::SeqCst) == ordinal {
            Err(InjectedError::Symbol)
        } else {
            Ok(())
        }
    }
}

impl CheckedGenericFusion for InjectedGeneric {
    type Error = InjectedError;

    fn rule_identity(&self) -> RuleIdentity {
        CheckedGenericFusion::rule_identity(&self.inner)
    }

    fn fusion_style(&self) -> FusionStyleKind {
        CheckedGenericFusion::fusion_style(&self.inner)
    }

    fn braiding_style(&self) -> BraidingStyleKind {
        CheckedGenericFusion::braiding_style(&self.inner)
    }

    fn vacuum(&self) -> SectorId {
        CheckedGenericFusion::vacuum(&self.inner)
    }

    fn try_dual(&self, sector: SectorId) -> Result<SectorId, Self::Error> {
        self.dual_calls.fetch_add(1, Ordering::SeqCst);
        if self.fail_dual.load(Ordering::SeqCst) {
            Err(InjectedError::Dual)
        } else {
            CheckedGenericFusion::try_dual(&self.inner, sector).map_err(|_| InjectedError::Provider)
        }
    }

    fn try_fusion_channels(
        &self,
        left: SectorId,
        right: SectorId,
    ) -> Result<SectorVec, Self::Error> {
        self.symbol()?;
        CheckedGenericFusion::try_fusion_channels(&self.inner, left, right)
            .map_err(|_| InjectedError::Provider)
    }

    fn try_fusion_channels_in_table(
        &self,
        left: SectorId,
        right: SectorId,
    ) -> Result<SectorVec, Self::Error> {
        self.symbol()?;
        CheckedGenericFusion::try_fusion_channels_in_table(&self.inner, left, right)
            .map_err(|_| InjectedError::Provider)
    }

    fn try_nsymbol(
        &self,
        left: SectorId,
        right: SectorId,
        coupled: SectorId,
    ) -> Result<usize, Self::Error> {
        self.symbol()?;
        CheckedGenericFusion::try_nsymbol(&self.inner, left, right, coupled)
            .map_err(|_| InjectedError::Provider)
    }
}

#[cfg(feature = "opt-path")]
#[derive(Clone, Copy)]
enum PlanningFailure {
    Dual,
    Dimension,
}

#[cfg(feature = "opt-path")]
fn assert_optimizer_does_not_retry_provider_failure(
    optimizer: Optimizer,
    failure: PlanningFailure,
) {
    let runtime = Runtime::builder()
        .dense_threads(1)
        .plan_cache(PlanCacheConfig {
            optimizer,
            ..PlanCacheConfig::default()
        })
        .build()
        .unwrap();
    let provider = Arc::new(InjectedGeneric::new());
    let tensors = injected_chain(&runtime, &provider);
    provider.dual_calls.store(0, Ordering::SeqCst);
    match failure {
        PlanningFailure::Dual => provider.fail_dual.store(true, Ordering::SeqCst),
        PlanningFailure::Dimension => provider.arm_symbol(1),
    }
    let first = &tensors[0];
    let second = &tensors[1];
    let third = &tensors[2];
    let fourth = &tensors[3];
    let error = tensor!(
        [a, b; f] = first[a, b; c]
            * second[c; d]
            * third[d; e]
            * fourth[e; f]
    )
    .unwrap_err();
    match failure {
        PlanningFailure::Dual => {
            assert!(matches!(
                error,
                GenericTensorError::Structure(CheckedGenericStructureError::Provider(
                    InjectedError::Dual
                ))
            ));
            assert_eq!(provider.dual_calls.load(Ordering::SeqCst), 1);
        }
        PlanningFailure::Dimension => {
            assert!(matches!(
                error,
                GenericTensorError::Structure(CheckedGenericStructureError::Provider(
                    InjectedError::Symbol
                ))
            ));
            assert_eq!(provider.symbol_calls.load(Ordering::SeqCst), 1);
        }
    }
    let stats = plan_cache_stats(&runtime);
    assert_eq!(stats.entries, 0);
    assert_eq!(stats.topology_materializations, 0);
    assert_eq!(stats.workspaces_created, 0);
}

#[cfg(feature = "opt-path")]
#[test]
fn checked_generic_optimizer_fallback_never_retries_provider_failures() {
    for optimizer in [Optimizer::DynamicProgramming, Optimizer::AutoHq] {
        for failure in [PlanningFailure::Dual, PlanningFailure::Dimension] {
            assert_optimizer_does_not_retry_provider_failure(optimizer.clone(), failure);
        }
    }
}

impl CheckedGenericRigidSymbols for InjectedGeneric {
    type Scalar = f64;

    fn try_sqrt_dim_scalar(&self, sector: SectorId) -> Result<f64, Self::Error> {
        self.symbol()?;
        CheckedGenericRigidSymbols::try_sqrt_dim_scalar(&self.inner, sector)
            .map_err(|_| InjectedError::Provider)
    }

    fn try_inv_sqrt_dim_scalar(&self, sector: SectorId) -> Result<f64, Self::Error> {
        self.symbol()?;
        CheckedGenericRigidSymbols::try_inv_sqrt_dim_scalar(&self.inner, sector)
            .map_err(|_| InjectedError::Provider)
    }

    fn try_frobenius_schur_phase_scalar(&self, sector: SectorId) -> Result<f64, Self::Error> {
        self.symbol()?;
        CheckedGenericRigidSymbols::try_frobenius_schur_phase_scalar(&self.inner, sector)
            .map_err(|_| InjectedError::Provider)
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
        self.symbol()?;
        CheckedGenericRigidSymbols::try_f_symbol_generic(&self.inner, a, b, c, d, e, f)
            .map_err(|_| InjectedError::Provider)
    }

    fn try_r_symbol_generic(
        &self,
        a: SectorId,
        b: SectorId,
        c: SectorId,
    ) -> Result<GenericRMatrix<f64>, Self::Error> {
        self.symbol()?;
        CheckedGenericRigidSymbols::try_r_symbol_generic(&self.inner, a, b, c)
            .map_err(|_| InjectedError::Provider)
    }
}

impl CheckedGenericPivotal for InjectedGeneric {
    fn try_twist_scalar(&self, sector: SectorId) -> Result<f64, Self::Error> {
        self.symbol()?;
        CheckedGenericPivotal::try_twist_scalar(&self.inner, sector)
            .map_err(|_| InjectedError::Provider)
    }
}

impl TypedSectorAdmission for InjectedGeneric {
    type Sector = Vec<i64>;
    type Error = InjectedError;
    type Mode = CheckedGenericAdmissionMode;

    fn typed_rule_identity(&self) -> RuleIdentity {
        CheckedGenericFusion::rule_identity(self)
    }

    fn try_encode_label(&self, sector: &Self::Sector) -> Result<SectorId, Self::Error> {
        if self.fail_encode.load(Ordering::SeqCst) {
            return Err(InjectedError::Encode);
        }
        self.inner
            .encode_dynkin(sector)
            .map_err(|_| InjectedError::Provider)
    }

    fn try_decode_label(&self, sector: SectorId) -> Result<Self::Sector, Self::Error> {
        self.inner
            .decode_dynkin(sector)
            .map_err(|_| InjectedError::Provider)
    }

    fn try_dual_id(&self, sector: SectorId) -> Result<SectorId, Self::Error> {
        self.try_dual(sector)
    }
}

fn injected_chain(
    runtime: &Runtime,
    provider: &Arc<InjectedGeneric>,
) -> Vec<TensorMap<InjectedGeneric, f64>> {
    let leg = GradedSpace::try_new_with_arc(Arc::clone(provider), [(vec![1, 1], 1)]).unwrap();
    vec![
        TensorMap::from_block_fn(runtime, [&leg, &leg], [&leg], |trees, _| {
            trees.codomain_vertices()[0].get() as f64
        })
        .unwrap(),
        TensorMap::from_block_fn(runtime, [&leg], [&leg], |_, _| 2.0).unwrap(),
        TensorMap::from_block_fn(runtime, [&leg], [&leg], |_, _| 3.0).unwrap(),
        TensorMap::from_block_fn(runtime, [&leg], [&leg], |_, _| 4.0).unwrap(),
    ]
}

fn injected_network(operands: usize, permute_output: bool) -> Network {
    let all_inputs = vec![
        labels(&["a", "b", "c"]),
        labels(&["c", "d"]),
        labels(&["d", "e"]),
        labels(&["e", "f"]),
    ];
    let tail = ["c", "d", "e", "f"][operands - 1];
    Network::new(
        all_inputs.into_iter().take(operands).collect(),
        vec![false; operands],
        std::iter::once(Some(2))
            .chain(std::iter::repeat_n(Some(1), operands - 1))
            .collect(),
        if permute_output {
            labels(&["b", "a", tail])
        } else {
            labels(&["a", "b", tail])
        },
        Some(2),
    )
    .unwrap()
}

fn assert_injected_recovery(
    provider: &InjectedGeneric,
    planned: &PlannedNetwork,
    tensors: &[&TensorMap<InjectedGeneric, f64>],
    ordinal: usize,
) {
    let mut workspace = NetworkExecutionWorkspace::default();
    provider.arm_symbol(ordinal);
    assert!(matches!(
        planned.execute_with_workspace(tensors, &mut workspace),
        Err(GenericTensorError::Plan(_))
    ));
    provider.reset_symbols();
    let recovered = planned
        .execute_with_workspace(tensors, &mut workspace)
        .unwrap();
    provider.reset_symbols();
    let expected = planned.execute(tensors).unwrap();
    assert_eq!(recovered.block_count(), expected.block_count());
    for (&actual, &want) in recovered.data().iter().zip(expected.data()) {
        assert!((actual - want).abs() <= 1.0e-12 * (1.0 + want.abs()));
    }
}

fn injected_plan_case(
    operands: usize,
    permute_output: bool,
) -> (
    Arc<InjectedGeneric>,
    Vec<TensorMap<InjectedGeneric, f64>>,
    PlannedNetwork,
) {
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let provider = Arc::new(InjectedGeneric::new());
    let tensors = injected_chain(&runtime, &provider);
    let refs = tensors.iter().take(operands).collect::<Vec<_>>();
    let planned = injected_network(operands, permute_output)
        .plan(
            &refs,
            &LabelOrderDenseOptimizer::new(labels(&["c", "d", "e"])[..operands - 1].to_vec()),
        )
        .unwrap();
    (provider, tensors, planned)
}

fn cold_query_count(operands: usize, permute_output: bool) -> usize {
    let (provider, tensors, planned) = injected_plan_case(operands, permute_output);
    let refs = tensors.iter().take(operands).collect::<Vec<_>>();
    provider.reset_symbols();
    planned.execute(&refs).unwrap();
    provider.symbol_calls.load(Ordering::SeqCst)
}

#[test]
fn checked_generic_failures_stay_typed_and_workspace_recovers_at_every_step() {
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let provider = Arc::new(InjectedGeneric::new());

    provider.fail_encode.store(true, Ordering::SeqCst);
    assert!(matches!(
        GradedSpace::try_new_with_arc(Arc::clone(&provider), [(vec![1, 1], 1)]),
        Err(GenericTensorError::Structure(_))
    ));
    provider.fail_encode.store(false, Ordering::SeqCst);

    let tensors = injected_chain(&runtime, &provider);
    let refs = tensors.iter().collect::<Vec<_>>();
    provider.fail_dual.store(true, Ordering::SeqCst);
    assert!(matches!(
        injected_network(4, true).plan(&refs, &GreedyDenseOptimizer),
        Err(GenericTensorError::Structure(_))
    ));
    provider.fail_dual.store(false, Ordering::SeqCst);

    let first_end = cold_query_count(2, false);
    let middle_end = cold_query_count(3, false);
    let final_end = cold_query_count(4, false);
    let permutation_end = cold_query_count(4, true);
    assert!([first_end, middle_end, final_end, permutation_end]
        .into_iter()
        .all(|queries| queries > 0));

    for (operands, permute, ordinal) in [
        (2, false, 1),
        (3, false, middle_end),
        (4, false, final_end),
        (4, true, permutation_end),
    ] {
        let (case_provider, case_tensors, case_plan) = injected_plan_case(operands, permute);
        let case_refs = case_tensors.iter().take(operands).collect::<Vec<_>>();
        assert_injected_recovery(&case_provider, &case_plan, &case_refs, ordinal);
    }
}

#[test]
fn checked_generic_static_trace_failure_stays_typed_and_does_not_publish_cache_state() {
    // What: a provider error during trace lowering returns before a static plan
    // or replay workspace is published, and the same expression can recover.
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let provider = Arc::new(InjectedGeneric::new());
    let leg = GradedSpace::try_new_with_arc(Arc::clone(&provider), [(vec![1, 1], 1)]).unwrap();
    let tensor: TensorMap<_, f64> =
        TensorMap::from_block_fn(&runtime, [&leg], [&leg], |_, _| 1.0).unwrap();
    provider.arm_symbol(1);
    let error = tensor!([;] = tensor[a; a]).unwrap_err();
    assert!(
        matches!(
            error,
            GenericTensorError::Plan(CheckedGenericPlanError::Provider(InjectedError::Symbol))
        ),
        "{error:?}"
    );
    let stats = plan_cache_stats(&runtime);
    assert_eq!(stats.entries, 0);
    assert_eq!(stats.workspaces_created, 0);

    provider.reset_symbols();
    let traced = tensor!([;] = tensor[a; a]).unwrap();
    assert_eq!(traced.provider() as *const _, provider.as_ref() as *const _);
}

#[test]
fn checked_generic_scalar_empty_outer_product_and_single_permute_follow_ordinary_ops() {
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let provider = Arc::new(InjectedGeneric::new());
    let leg = GradedSpace::try_new_with_arc(Arc::clone(&provider), [(vec![1, 1], 1)]).unwrap();
    let lhs: TensorMap<_, f64> =
        TensorMap::from_block_fn(&runtime, [&leg], [], |_, _| 2.0).unwrap();
    let rhs: TensorMap<_, f64> =
        TensorMap::from_block_fn(&runtime, [], [&leg], |_, _| 5.0).unwrap();
    let scalar = lhs.contract(&rhs, &[0], &[0], &[]).unwrap();
    let scalar_plan = Network::new(vec![vec![]], vec![false], vec![Some(0)], vec![], Some(0))
        .unwrap()
        .plan(&[&scalar], &GreedyDenseOptimizer)
        .unwrap();
    assert_eq!(
        scalar_plan.execute(&[&scalar]).unwrap().data(),
        scalar.data()
    );
    let outer = Network::new(
        vec![labels(&["a"]), labels(&["b"])],
        vec![false; 2],
        vec![Some(1), Some(0)],
        labels(&["a", "b"]),
        Some(1),
    )
    .unwrap()
    .plan(&[&lhs, &rhs], &GreedyDenseOptimizer)
    .unwrap()
    .execute(&[&lhs, &rhs])
    .unwrap();
    let expected_outer = lhs.contract(&rhs, &[], &[], &[0, 1]).unwrap();
    assert_eq!(outer.data(), expected_outer.data());

    let rank_three: TensorMap<_, f64> =
        TensorMap::from_block_fn(&runtime, [&leg, &leg], [&leg], |trees, _| {
            trees.codomain_vertices()[0].get() as f64
        })
        .unwrap();
    let permuted = Network::new(
        vec![labels(&["a", "b", "c"])],
        vec![false],
        vec![Some(2)],
        labels(&["b", "a", "c"]),
        Some(2),
    )
    .unwrap()
    .plan(&[&rank_three], &GreedyDenseOptimizer)
    .unwrap()
    .execute(&[&rank_three])
    .unwrap();
    assert_eq!(
        permuted.data(),
        rank_three.permute(&[1, 0], &[2]).unwrap().data()
    );

    let empty = GradedSpace::try_new_with_arc(
        Arc::clone(&provider),
        std::iter::empty::<(Vec<i64>, usize)>(),
    )
    .unwrap();
    let zero: TensorMap<_, f64> = TensorMap::zeros(&runtime, [&empty], []).unwrap();
    let zero_outer = Network::new(
        vec![labels(&["a"]), labels(&["b"])],
        vec![false; 2],
        vec![Some(1); 2],
        labels(&["a", "b"]),
        Some(1),
    )
    .unwrap()
    .plan(&[&zero, &zero], &GreedyDenseOptimizer)
    .unwrap()
    .execute(&[&zero, &zero])
    .unwrap();
    assert!(zero_outer.data().is_empty());
}

#[test]
fn checked_generic_cache_modes_dtype_pools_and_lazy_rejection_match_direct_authority() {
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let provider = Arc::new(InjectedGeneric::new());
    let leg = GradedSpace::try_new_with_arc(Arc::clone(&provider), [(vec![1, 1], 1)]).unwrap();
    let a64: TensorMap<_, f64> =
        TensorMap::from_block_fn(&runtime, [&leg], [&leg], |_, _| 2.0).unwrap();
    let b64: TensorMap<_, f64> =
        TensorMap::from_block_fn(&runtime, [&leg], [&leg], |_, _| 3.0).unwrap();
    let ac: TensorMap<_, Complex64> =
        TensorMap::from_block_fn(&runtime, [&leg], [&leg], |_, _| Complex64::new(2.0, 1.0))
            .unwrap();
    let bc: TensorMap<_, Complex64> =
        TensorMap::from_block_fn(&runtime, [&leg], [&leg], |_, _| Complex64::new(3.0, -1.0))
            .unwrap();
    let first = tensor!([i; k] = a64[i; j] * b64[j; k]).unwrap();
    let second = tensor!([i; k] = a64[i; j] * b64[j; k]).unwrap();
    let complex = tensor!([i; k] = ac[i; j] * bc[j; k]).unwrap();
    assert_eq!(first.data(), second.data());
    assert_eq!(
        complex.data(),
        ac.contract(&bc, &[1], &[0], &[0, 1]).unwrap().data()
    );
    let stats = plan_cache_stats(&runtime);
    assert!(stats.hits >= 2);
    assert_eq!(stats.entries, 1);
    assert_eq!(stats.workspaces_created, 2);

    configure_plan_cache(
        &runtime,
        PlanCacheConfig {
            enabled: false,
            ..runtime.plan_cache_config()
        },
    );
    let uncached = tensor!([i; k] = a64[i; j] * b64[j; k]).unwrap();
    assert_eq!(uncached.data(), first.data());

    let lazy = a64.adjoint().unwrap();
    let network = Network::new(
        vec![labels(&["i", "j"]), labels(&["j", "k"])],
        vec![false; 2],
        vec![Some(1); 2],
        labels(&["i", "k"]),
        Some(1),
    )
    .unwrap();
    let planned = network.plan(&[&lazy, &b64], &GreedyDenseOptimizer).unwrap();
    let direct = lazy.contract(&b64, &[1], &[0], &[0, 1]);
    let replay = planned.execute(&[&lazy, &b64]);
    assert!(matches!(
        direct,
        Err(GenericTensorError::Facade(Error::InvalidArgument(_)))
    ));
    assert!(matches!(
        replay,
        Err(GenericTensorError::Facade(Error::InvalidArgument(_)))
    ));

    let conjugated = Network::new(
        vec![labels(&["j", "i"]), labels(&["j", "k"])],
        vec![true, false],
        vec![Some(1); 2],
        labels(&["i", "k"]),
        Some(1),
    )
    .unwrap()
    .plan(&[&a64, &b64], &GreedyDenseOptimizer)
    .unwrap()
    .execute(&[&a64, &b64]);
    assert!(matches!(
        conjugated,
        Err(GenericTensorError::Facade(Error::InvalidArgument(_)))
    ));
}
