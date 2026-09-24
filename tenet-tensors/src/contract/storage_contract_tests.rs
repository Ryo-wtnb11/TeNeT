//! The Host compile entry of the storage (device) contraction route, and —
//! with the `cuda` feature, on a real device — its executor replaying the
//! Host `DynamicTree` artifact (G2c-1a, #1345).
//!
//! The device tests force every orientation and axis-order candidate through
//! the test-only plan builder, which the typed API cannot reach, and compare
//! the device replay of each artifact with the Host replay of the same
//! artifact. Independent oracles for the typed lowering live in
//! `tenet/tests/typed_contract_host_oracle.rs` and `typed_cuda_contract.rs`.

use std::sync::Arc;

use tenet_core::{
    FermionParityFusionRule, FusionProductSpace, FusionTreeHomSpace, MultiplicityFreeRigidSymbols,
    ProductFusionRule, ProductFusionRuleExt, SU2FusionRule, SU2Irrep, SectorId, SectorLeg,
    U1FusionRule, U1Irrep,
};
use tenet_operations::{OutputAxisOrder, TensorContractSpec};

use crate::contract::dynamic::{
    compile_dynamic_tree_execution_artifact, execute_dynamic_tree_execution_artifact,
    DynamicFusionSpaceCache, DynamicTreeExecutionArtifact,
};
use crate::contract::fusion::FusionContractOrientation;
use crate::contract::scratch::DynamicFusionScratchWorkspace;
use crate::tree_context::TreeTransformExecutionContext;
use crate::{
    BoundDynamicFusionMapSpace, DenseRecouplingScalar, DenseTreeTransformOperations, FusionOperand,
    RecouplingCoefficientAction, RuleIdentity,
};

type Context<D> = crate::TensorContractFusionExecutionContext<D, RuleIdentity>;

fn space<R: MultiplicityFreeRigidSymbols<Scalar = f64>>(
    provider: &Arc<R>,
    codomain: Vec<SectorLeg>,
    domain: Vec<SectorLeg>,
) -> BoundDynamicFusionMapSpace<R> {
    BoundDynamicFusionMapSpace::from_final_homspace_multiplicity_free(
        Arc::clone(provider),
        FusionTreeHomSpace::new(
            FusionProductSpace::new(codomain),
            FusionProductSpace::new(domain),
        ),
    )
    .unwrap()
}

fn u1_leg(dual: bool) -> SectorLeg {
    SectorLeg::new(
        [
            (U1Irrep::new(-1).sector_id(), 2),
            (U1Irrep::new(0).sector_id(), 1),
            (U1Irrep::new(1).sector_id(), 2),
        ],
        dual,
    )
}

fn su2_leg() -> SectorLeg {
    SectorLeg::new(
        [
            (SU2Irrep::from_twice_spin(0).sector_id(), 2),
            (SU2Irrep::from_twice_spin(1).sector_id(), 1),
            (SU2Irrep::from_twice_spin(2).sector_id(), 1),
        ],
        false,
    )
}

/// A rank-5 × rank-4 U(1) contraction on a codomain leg against a domain leg
/// and a domain leg against a codomain leg, contracted out of order, with a
/// permuted output: source transforms on both sides and an output transform.
struct Case<R> {
    lhs: BoundDynamicFusionMapSpace<R>,
    rhs: BoundDynamicFusionMapSpace<R>,
    lhs_axes: Vec<usize>,
    rhs_axes: Vec<usize>,
    output_axes: Vec<usize>,
}

impl<R: MultiplicityFreeRigidSymbols<Scalar = f64>> Case<R> {
    fn dst(&self) -> BoundDynamicFusionMapSpace<R> {
        BoundDynamicFusionMapSpace::contracted_multiplicity_free_ordered(
            &self.lhs,
            &self.rhs,
            &self.lhs_axes,
            &self.rhs_axes,
            OutputAxisOrder::from_axes(&self.output_axes),
        )
        .unwrap()
    }

    fn axes(&self) -> TensorContractSpec<'_> {
        TensorContractSpec::new(
            &self.lhs_axes,
            &self.rhs_axes,
            OutputAxisOrder::from_axes(&self.output_axes),
        )
    }
}

fn u1_case() -> Case<U1FusionRule> {
    let provider = Arc::new(U1FusionRule);
    let v = || u1_leg(false);
    Case {
        lhs: space(&provider, vec![v(), v(), v()], vec![v(), v()]),
        rhs: space(&provider, vec![v(), v()], vec![v(), v()]),
        lhs_axes: vec![3, 1],
        rhs_axes: vec![0, 3],
        output_axes: vec![2, 0, 4, 1, 3],
    }
}

fn su2_case() -> Case<SU2FusionRule> {
    let provider = Arc::new(SU2FusionRule);
    let s = su2_leg;
    Case {
        lhs: space(&provider, vec![s(), s()], vec![s(), s()]),
        rhs: space(&provider, vec![s(), s()], vec![s()]),
        lhs_axes: vec![3, 2],
        rhs_axes: vec![0, 1],
        output_axes: vec![1, 0, 2],
    }
}

/// `lhs` already has its contracted leg as its whole domain: under LhsRhs
/// its source transform is the identity and it is read in place; the output
/// is permuted, so an output transform remains.
fn lhs_identity_case() -> Case<U1FusionRule> {
    let provider = Arc::new(U1FusionRule);
    let v = || u1_leg(false);
    Case {
        lhs: space(&provider, vec![v(), v()], vec![v()]),
        rhs: space(&provider, vec![v(), v()], vec![v()]),
        lhs_axes: vec![2],
        rhs_axes: vec![1],
        output_axes: vec![3, 1, 0, 2],
    }
}

/// `rhs` already has its contracted leg as its whole codomain.
fn rhs_identity_case() -> Case<U1FusionRule> {
    let provider = Arc::new(U1FusionRule);
    let v = || u1_leg(false);
    Case {
        lhs: space(&provider, vec![v(), v()], vec![v()]),
        rhs: space(&provider, vec![u1_leg(true)], vec![v(), v()]),
        lhs_axes: vec![0],
        rhs_axes: vec![0],
        output_axes: vec![1, 0, 2, 3],
    }
}

/// The artifact of one forced axis-order candidate and orientation — the
/// test-only plan builder reaches both, the typed API only the scorer's
/// choice — together with its Host replay over `lhs`/`rhs` and the core-right
/// contracted axes of its plan.
fn forced_artifact<R, D>(
    case: &Case<R>,
    candidate: &crate::contract::fusion::ContractAxisOrderCandidate,
    orientation: FusionContractOrientation,
    lhs: &[D],
    rhs: &[D],
) -> (DynamicTreeExecutionArtifact<f64>, Vec<D>, Vec<usize>)
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64>
        + crate::TreeTransformRuleCacheKey<Key = RuleIdentity>,
    D: DenseRecouplingScalar + RecouplingCoefficientAction<f64>,
{
    let rule = case.lhs.provider();
    let dst = case.dst();
    let plan =
        crate::contract::prepare_tensorcontract_fusion_plan_dyn_raw_with_axis_order_and_orientation(
            rule,
            dst.space(),
            case.lhs.space(),
            case.rhs.space(),
            case.axes(),
            candidate,
            orientation,
        )
        .unwrap();
    let mut tree_context =
        TreeTransformExecutionContext::new(DenseTreeTransformOperations::default_executor());
    let mut cache = DynamicFusionSpaceCache::default();
    let artifact = compile_dynamic_tree_execution_artifact::<_, _, _, D, _, false>(
        &mut tree_context,
        &mut cache,
        rule,
        crate::contract::encoded_layout_primer::<R>,
        &plan,
        dst.space(),
        case.lhs.space(),
        case.lhs.space().structure(),
        case.rhs.space(),
        case.rhs.space().structure(),
        None,
    )
    .unwrap();
    let mut host = vec![D::zero(); dst.space().required_len().unwrap()];
    execute_dynamic_tree_execution_artifact(
        &mut tree_context,
        &mut DenseTreeTransformOperations::default(),
        &mut crate::contract::backend::TensorContractWorkspace::default(),
        &mut crate::contract::fusion_block::FusionBlockContractWorkspace::default(),
        &mut DynamicFusionScratchWorkspace::default(),
        &artifact,
        dst.space().structure(),
        &mut host,
        lhs,
        rhs,
        D::one(),
        D::zero(),
    )
    .unwrap();
    let core_right_contracting = plan.core_axes().as_spec().rhs_contracting_axes().to_vec();
    (artifact, host, core_right_contracting)
}

fn orientations() -> [FusionContractOrientation; 2] {
    [
        FusionContractOrientation::LhsRhs,
        FusionContractOrientation::RhsLhs,
    ]
}

type FermionU1 = ProductFusionRule<FermionParityFusionRule, U1FusionRule>;
type FermionSu2 = ProductFusionRule<FermionParityFusionRule, SU2FusionRule>;

const EVEN: SectorId = SectorId::new(0);
const ODD: SectorId = SectorId::new(1);

/// fZ2 x U(1), degeneracy > 1, odd sectors of both charge signs. A dual leg
/// stores the conjugate charges (the typed `GradedSpace::try_dual`).
fn fermion_u1_leg(rule: &FermionU1, dual: bool) -> SectorLeg {
    let charge = |q: i32| U1Irrep::new(if dual { -q } else { q }).sector_id();
    SectorLeg::new(
        [
            (rule.encode_sector(EVEN, charge(0)), 2),
            (rule.encode_sector(ODD, charge(1)), 2),
            (rule.encode_sector(ODD, charge(-1)), 1),
            (rule.encode_sector(EVEN, charge(1)), 1),
        ],
        dual,
    )
}

/// fZ2 (x) SU(2): the fermionic rule whose transforms recouple (Multi blocks).
fn fermion_su2_leg(rule: &FermionSu2, dual: bool) -> SectorLeg {
    SectorLeg::new(
        [
            (
                rule.encode_sector(EVEN, SU2Irrep::from_twice_spin(0).sector_id()),
                2,
            ),
            (
                rule.encode_sector(ODD, SU2Irrep::from_twice_spin(1).sector_id()),
                2,
            ),
            (
                rule.encode_sector(EVEN, SU2Irrep::from_twice_spin(2).sector_id()),
                1,
            ),
        ],
        dual,
    )
}

/// The geometry of `tenet/tests/contract_cases::fermionic_general`: rank 4
/// against rank 4 over mixed sides, B's codomain leg `v*` and B's domain leg
/// `u` contracted (dual on B's codomain side, and dual after bending from
/// its domain side unless `u` is itself dual — then θ is mixed within one
/// coupled sector).
fn fermionic_general_case<R: MultiplicityFreeRigidSymbols<Scalar = f64>>(
    provider: &Arc<R>,
    leg: impl Fn(bool) -> SectorLeg,
    u_dual: bool,
) -> Case<R> {
    Case {
        lhs: space(
            provider,
            vec![leg(false), leg(u_dual)],
            vec![leg(false), leg(false)],
        ),
        rhs: space(
            provider,
            vec![leg(false), leg(true)],
            vec![leg(false), leg(u_dual)],
        ),
        lhs_axes: vec![1, 0],
        rhs_axes: vec![3, 1],
        output_axes: vec![2, 0, 3, 1],
    }
}

/// The canonical `mul!` form with B's codomain `(v, v*)`: θ varies within one
/// coupled-sector matrix (`contract_cases::fermionic_canonical_nonuniform`).
fn fermionic_canonical_case<R: MultiplicityFreeRigidSymbols<Scalar = f64>>(
    provider: &Arc<R>,
    leg: impl Fn(bool) -> SectorLeg,
) -> Case<R> {
    Case {
        lhs: space(
            provider,
            vec![leg(false), leg(false)],
            vec![leg(false), leg(true)],
        ),
        rhs: space(provider, vec![leg(false), leg(true)], vec![leg(false)]),
        lhs_axes: vec![2, 3],
        rhs_axes: vec![0, 1],
        output_axes: vec![0, 1, 2],
    }
}

/// Which operands the `LhsRhs` layout of a [`fermionic_twist_role_cases`]
/// fixture already copies, in `(A, B)` order.
#[derive(Clone, Copy, Debug)]
enum RoleCase {
    ACopied,
    Canonical,
    BCopied,
    BothCopied,
}

/// The geometries of `tenet/tests/contract_cases::fermionic_twist_roles`:
/// one per operand-copy case of TensorKit's twist choice, each with a dual
/// leg among B's contracted legs, and A smaller than B where size decides.
fn fermionic_twist_role_cases<R: MultiplicityFreeRigidSymbols<Scalar = f64>>(
    provider: &Arc<R>,
    leg: impl Fn(bool) -> SectorLeg,
) -> [(RoleCase, Case<R>); 4] {
    let v = || leg(false);
    let v_dual = || leg(true);
    [
        (
            RoleCase::ACopied,
            Case {
                lhs: space(provider, vec![v(), v()], vec![v()]),
                rhs: space(provider, vec![v_dual()], vec![v(), v()]),
                lhs_axes: vec![1],
                rhs_axes: vec![0],
                output_axes: vec![0, 1, 2, 3],
            },
        ),
        (
            RoleCase::Canonical,
            Case {
                lhs: space(provider, vec![v()], vec![v(), v_dual()]),
                rhs: space(provider, vec![v(), v_dual()], vec![v(), v()]),
                lhs_axes: vec![1, 2],
                rhs_axes: vec![0, 1],
                output_axes: vec![0, 1, 2],
            },
        ),
        (
            RoleCase::BCopied,
            Case {
                lhs: space(provider, vec![v(), v()], vec![v_dual()]),
                rhs: space(provider, vec![v()], vec![v(), v()]),
                lhs_axes: vec![2],
                rhs_axes: vec![1],
                output_axes: vec![0, 1, 2, 3],
            },
        ),
        (
            RoleCase::BothCopied,
            Case {
                lhs: space(provider, vec![v(), v()], vec![v()]),
                rhs: space(provider, vec![v(), v_dual()], vec![v(), v()]),
                lhs_axes: vec![1, 0],
                rhs_axes: vec![3, 1],
                output_axes: vec![0, 1, 2],
            },
        ),
    ]
}

fn fermionic_cases() -> Vec<(&'static str, Case<FermionU1>)> {
    let provider = Arc::new(FermionParityFusionRule.product(U1FusionRule));
    let leg = |dual| fermion_u1_leg(&provider, dual);
    let mut cases = vec![
        ("fZ2xU1 both", fermionic_general_case(&provider, leg, false)),
        ("fZ2xU1 mixed", fermionic_general_case(&provider, leg, true)),
        ("fZ2xU1 canonical", fermionic_canonical_case(&provider, leg)),
    ];
    let names = [
        "fZ2xU1 A copied",
        "fZ2xU1 canonical small A",
        "fZ2xU1 B copied",
        "fZ2xU1 both copied small A",
    ];
    cases.extend(
        names
            .into_iter()
            .zip(fermionic_twist_role_cases(&provider, leg))
            .map(|(name, (_, case))| (name, case)),
    );
    cases
}

fn fermionic_su2_cases() -> Vec<(&'static str, Case<FermionSu2>)> {
    let provider = Arc::new(FermionParityFusionRule.product(SU2FusionRule));
    let leg = |dual| fermion_su2_leg(&provider, dual);
    let mut cases = vec![
        (
            "fZ2xSU2 both",
            fermionic_general_case(&provider, leg, false),
        ),
        (
            "fZ2xSU2 mixed",
            fermionic_general_case(&provider, leg, true),
        ),
        (
            "fZ2xSU2 canonical",
            fermionic_canonical_case(&provider, leg),
        ),
    ];
    let names = [
        "fZ2xSU2 A copied",
        "fZ2xSU2 canonical small A",
        "fZ2xSU2 B copied",
        "fZ2xSU2 both copied small A",
    ];
    cases.extend(
        names
            .into_iter()
            .zip(fermionic_twist_role_cases(&provider, leg))
            .map(|(name, (_, case))| (name, case)),
    );
    cases
}

fn host_data(space: &crate::DynamicFusionMapSpace, salt: usize) -> Vec<f64> {
    (0..space.required_len().unwrap())
        .map(|index| ((index * 7 + salt) as f64 * 0.37 + 0.1).sin())
        .collect()
}

/// The Host's per-block twist of one physical operand's transformed source,
/// recomputed from that space alone (`lhs` names the side) by the Host twist
/// compiler, as the sorted `(offset, θ)` of its non-empty twisted blocks.
fn physical_twist<R>(
    case: &Case<R>,
    artifact: &DynamicTreeExecutionArtifact<f64>,
    orientation: FusionContractOrientation,
    contracting: &[usize],
) -> Vec<(usize, f64)>
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64>,
{
    let core_right_is_lhs = orientation == FusionContractOrientation::RhsLhs;
    let lhs = artifact.twists_lhs();
    super::dynamic::contract_twist_scales(
        case.lhs.provider(),
        artifact.physical_core_space(lhs),
        artifact.physical_core_space(core_right_is_lhs).homspace(),
        lhs != core_right_is_lhs,
        contracting,
    )
    .unwrap()
}

/// The eager Host contraction of `case` over the `lhs`/`rhs` payloads: the
/// production answer every forced artifact must reproduce.
fn eager_host<R, D>(case: &Case<R>, lhs: &[D], rhs: &[D]) -> Vec<D>
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64>
        + crate::TreeTransformRuleCacheKey<Key = RuleIdentity>,
    D: DenseRecouplingScalar + RecouplingCoefficientAction<f64>,
{
    let dst = case.dst();
    let mut out = vec![D::zero(); dst.space().required_len().unwrap()];
    Context::<D>::default()
        .tensorcontract_fusion_dyn_prelowered_into(
            &dst,
            &mut out,
            FusionOperand::direct(case.lhs.space()),
            lhs,
            FusionOperand::direct(case.rhs.space()),
            rhs,
            case.axes(),
            D::one(),
            D::zero(),
        )
        .unwrap();
    out
}

/// Every forced candidate and orientation of one fermionic case:
///
/// - the destination-scale list is the Host's per-block twist of the
///   *physical* operand the artifact twists — core-right or, when only
///   core-left is already copied (or is the smaller), core-left — recomputed
///   from that space and the core-right dual flags alone, and equals the
///   artifact's own in-place twist actions;
/// - the compile-time uniform-per-Multi assertion passed;
/// - the Host replay of the forced artifact equals the eager Host
///   `contract`, so every forced orientation is tied to the production
///   answer, not only to a device replay of the same artifact.
///
/// Returns, per orientation (`[LhsRhs, RhsLhs]`), whether any artifact
/// twisted, and whether any twisted core-right transform had a Multi block.
fn check_destination_scales<R>(case: &Case<R>, what: &str) -> ([bool; 2], bool)
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64>
        + crate::TreeTransformRuleCacheKey<Key = RuleIdentity>,
{
    let lhs = host_data(case.lhs.space(), 1);
    let rhs = host_data(case.rhs.space(), 2);
    let eager = eager_host(case, &lhs, &rhs);
    let scale = eager
        .iter()
        .fold(0.0_f64, |max, value| max.max(value.abs()));
    assert!(scale > 1e-3, "{what}: vacuous all-zero eager result");
    let (mut twisted, mut multi) = ([false; 2], false);
    for candidate in
        crate::contract::contracted_axis_order_candidates(&case.lhs_axes, &case.rhs_axes)
    {
        for (slot, orientation) in orientations().into_iter().enumerate() {
            let where_ = format!("{what} {candidate:?} {orientation:?}");
            let (artifact, host, contracting) =
                forced_artifact(case, &candidate, orientation, &lhs, &rhs);
            let scales = artifact.source_twist_destination_scales();
            assert_eq!(
                scales,
                physical_twist(case, &artifact, orientation, &contracting).as_slice(),
                "{where_}: not the twisted physical operand's twist"
            );
            assert_eq!(scales, artifact.host_twist_scales().as_slice(), "{where_}");
            assert!(scales.windows(2).all(|pair| pair[0].0 < pair[1].0));
            assert!(scales.iter().all(|&(_, theta)| theta == -1.0));
            assert_eq!(host.len(), eager.len(), "{where_}");
            for (index, (&forced, &production)) in host.iter().zip(&eager).enumerate() {
                assert!(
                    (forced - production).abs() <= 1e-12 * (1.0 + scale),
                    "{where_}: element {index} is {forced}, eager Host {production}"
                );
            }
            if !scales.is_empty() {
                twisted[slot] = true;
                multi |= artifact
                    .twisted_transform_structure()
                    .blocks()
                    .iter()
                    .any(|block| {
                        matches!(block, tenet_operations::TreeTransformBlock::Multi { .. })
                    });
            }
        }
    }
    (twisted, multi)
}

#[test]
fn the_destination_scale_list_is_the_hosts_per_block_twist_in_both_orientations() {
    let mut report = Vec::new();
    let mut any_multi = false;
    let mut record = |what: &str, (twisted, multi): ([bool; 2], bool)| {
        report.push(format!(
            "{what}: LhsRhs {} RhsLhs {}",
            twisted[0], twisted[1]
        ));
        any_multi |= multi;
        twisted
    };
    let mut u1 = Vec::new();
    for (what, case) in fermionic_cases() {
        u1.push(record(what, check_destination_scales(&case, what)));
    }
    let mut su2 = Vec::new();
    for (what, case) in fermionic_su2_cases() {
        su2.push(record(what, check_destination_scales(&case, what)));
    }
    // What: every fixture twists under LhsRhs, and for each provider some
    // fixture also twists under RhsLhs, so neither orientation is covered
    // only vacuously. (With `u` not dual the lhs's contracted legs are all
    // non-dual: under RhsLhs that fixture carries no twist, and its forced
    // replays still equal the eager Host result above.)
    for twisted in [&u1, &su2] {
        assert!(twisted.iter().all(|t| t[0]), "{report:?}");
        assert!(twisted.iter().any(|t| t[0] && t[1]), "{report:?}");
    }
    // What: the uniform-per-Multi assertion was really exercised.
    assert!(any_multi, "no twisted transform recoupled");
}

#[test]
fn a_scale_list_that_varies_within_one_multi_block_is_an_internal_error() {
    let lhs_rhs = |case: &Case<FermionSu2>| {
        (
            host_data(case.lhs.space(), 1),
            host_data(case.rhs.space(), 2),
        )
    };
    let mut checked = false;
    for (what, case) in fermionic_su2_cases() {
        let (lhs, rhs) = lhs_rhs(&case);
        for candidate in
            crate::contract::contracted_axis_order_candidates(&case.lhs_axes, &case.rhs_axes)
        {
            for orientation in orientations() {
                let (artifact, _, _) = forced_artifact(&case, &candidate, orientation, &lhs, &rhs);
                let structure = artifact.twisted_transform_structure();
                let scales = artifact.source_twist_destination_scales();
                assert!(
                    super::dynamic::validate_uniform_multi_scales(structure, scales).is_ok(),
                    "{what}"
                );
                let layouts = structure.layouts();
                let Some(first) = structure.blocks().iter().find_map(|block| match *block {
                    tenet_operations::TreeTransformBlock::Multi {
                        dst_layout_start,
                        dst_count,
                        ..
                    } => {
                        let live: Vec<usize> = (dst_layout_start..dst_layout_start + dst_count)
                            .filter(|&index| layouts.entry(index).element_count != 0)
                            .map(|index| usize::try_from(layouts.entry(index).offset).unwrap())
                            .collect();
                        (live.len() >= 2).then(|| live[0])
                    }
                    tenet_operations::TreeTransformBlock::Single { .. } => None,
                }) else {
                    continue;
                };
                // Flip θ of one destination layout of that Multi block only.
                let mut tampered = scales.to_vec();
                match tampered.binary_search_by_key(&first, |&(offset, _)| offset) {
                    Ok(index) => {
                        tampered.remove(index);
                    }
                    Err(index) => tampered.insert(index, (first, -1.0)),
                }
                let error = super::dynamic::validate_uniform_multi_scales(structure, &tampered)
                    .unwrap_err();
                assert!(
                    matches!(error, crate::OperationError::InvalidArgument { .. }),
                    "{error:?}"
                );
                checked = true;
            }
        }
    }
    assert!(
        checked,
        "no fixture had a Multi block with two live destinations"
    );
}

/// TensorKit's `blas_contract!` choice of the twisted operand
/// (tensoroperations.jl:398-409 @cfaa073), in the `LhsRhs` orientation:
/// the planner's cost, the compiled artifact's borrowing and the Host
/// replay's source transforms all agree that the twist rides on the operand
/// already copied (or on the smaller one) and adds no materialization.
fn check_twist_role<R>(role: RoleCase, case: &Case<R>, what: &str) -> (usize, usize)
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64>
        + crate::TreeTransformRuleCacheKey<Key = RuleIdentity>,
{
    let rule = case.lhs.provider();
    let dst = case.dst();
    let e_a = case.lhs.space().required_len().unwrap();
    let e_b = case.rhs.space().required_len().unwrap();
    if matches!(role, RoleCase::Canonical | RoleCase::BothCopied) {
        assert!(e_a < e_b, "{what}: A must be the smaller operand");
    }
    let facts = crate::contract::fusion::prepare_tensorcontract_fusion_candidate_facts_dyn_raw(
        rule,
        dst.space(),
        case.lhs.space(),
        case.rhs.space(),
        case.axes(),
    )
    .unwrap();
    // What: this candidate is also the one the production scorer selects.
    let cheapest = facts.iter().map(|f| f.total_materialized_elements()).min();
    let facts = &facts[0];
    assert_eq!(
        cheapest,
        Some(facts.total_materialized_elements()),
        "{what}"
    );
    assert_eq!(
        facts.orientation(),
        FusionContractOrientation::LhsRhs,
        "{what}"
    );
    // (A copied by layout, B copied by layout, A twisted, total before #1351).
    let (a_copied, b_copied, twist_a, before) = match role {
        RoleCase::ACopied => (true, false, true, e_a + e_b),
        RoleCase::Canonical => (false, false, true, e_b),
        RoleCase::BCopied => (false, true, false, e_b),
        RoleCase::BothCopied => (true, true, true, e_a + e_b),
    };
    assert_eq!(facts.lhs_exact_identity_borrowable(), !a_copied, "{what}");
    assert_eq!(facts.rhs_exact_identity_borrowable(), !b_copied, "{what}");
    assert!(facts.lhs_requires_twist() == twist_a, "{what}");
    assert!(facts.rhs_requires_twist() != twist_a, "{what}");
    let lhs_len = if a_copied || twist_a { e_a } else { 0 };
    let rhs_len = if b_copied || !twist_a { e_b } else { 0 };
    assert_eq!(facts.lhs_materialized_elements(), lhs_len, "{what}");
    assert_eq!(facts.rhs_materialized_elements(), rhs_len, "{what}");

    let lhs = host_data(case.lhs.space(), 1);
    let rhs = host_data(case.rhs.space(), 2);
    let candidate =
        crate::contract::contracted_axis_order_candidates(&case.lhs_axes, &case.rhs_axes).remove(0);
    let (artifact, host, _) = forced_artifact(
        case,
        &candidate,
        FusionContractOrientation::LhsRhs,
        &lhs,
        &rhs,
    );
    assert!(artifact.requires_source_twist(), "{what}");
    assert_eq!(artifact.twists_lhs(), twist_a, "{what}");
    assert_eq!(
        artifact.borrowed_sources(),
        (lhs_len == 0, rhs_len == 0),
        "{what}"
    );

    let mut tree_context = TreeTransformExecutionContext::<f64, RuleIdentity>::new(
        DenseTreeTransformOperations::default_executor(),
    );
    let mut profile = tenet_operations::TensorContractFusionProfile::default();
    let mut profiled = vec![0.0; dst.space().required_len().unwrap()];
    super::dynamic::execute_dynamic_tree_execution_artifact_profiled(
        &mut tree_context,
        &mut DenseTreeTransformOperations::default(),
        &mut crate::contract::backend::TensorContractWorkspace::default(),
        &mut crate::contract::fusion_block::FusionBlockContractWorkspace::default(),
        &mut DynamicFusionScratchWorkspace::default(),
        &artifact,
        dst.space().structure(),
        &mut profiled,
        &lhs,
        &rhs,
        1.0,
        0.0,
        &mut profile,
    )
    .unwrap();
    assert_eq!(
        profile.lhs_transform_calls,
        usize::from(lhs_len != 0),
        "{what}"
    );
    assert_eq!(
        profile.rhs_transform_calls,
        usize::from(rhs_len != 0),
        "{what}"
    );

    let eager = eager_host(case, &lhs, &rhs);
    let scale = eager
        .iter()
        .fold(0.0_f64, |max, value| max.max(value.abs()));
    for (index, ((&forced, &replayed), &production)) in
        host.iter().zip(&profiled).zip(&eager).enumerate()
    {
        assert!(
            (forced - production).abs() <= 1e-12 * (1.0 + scale)
                && (replayed - production).abs() <= 1e-12 * (1.0 + scale),
            "{what}: element {index} is {forced}/{replayed}, eager Host {production}"
        );
    }
    (lhs_len + rhs_len, before)
}

#[test]
fn the_fermionic_twist_rides_on_the_operand_already_copied_or_the_smaller() {
    let fu1 = Arc::new(FermionParityFusionRule.product(U1FusionRule));
    let fsu2 = Arc::new(FermionParityFusionRule.product(SU2FusionRule));
    let mut totals = Vec::new();
    for (role, case) in fermionic_twist_role_cases(&fu1, |dual| fermion_u1_leg(&fu1, dual)) {
        totals.push((
            role,
            check_twist_role(role, &case, &format!("fZ2xU1 {role:?}")),
        ));
    }
    for (role, case) in fermionic_twist_role_cases(&fsu2, |dual| fermion_su2_leg(&fsu2, dual)) {
        totals.push((
            role,
            check_twist_role(role, &case, &format!("fZ2xSU2 {role:?}")),
        ));
    }
    // What: the E_B-sized materialization is gone exactly where only A was
    // copied (A copied: E_A + E_B -> E_A; canonical with A smaller:
    // E_B -> E_A); B copied and both copied are unchanged.
    for (role, (after, before)) in totals {
        match role {
            RoleCase::ACopied | RoleCase::Canonical => assert!(after < before, "{role:?}"),
            RoleCase::BCopied | RoleCase::BothCopied => assert_eq!(after, before, "{role:?}"),
        }
    }
}

/// A storage GEMM the Host storage-direct entry must reject before reaching.
struct RejectingGemm;

impl tenet_operations::fusion_replay::StorageGemm<f64, Vec<f64>, Vec<f64>, Vec<f64>>
    for RejectingGemm
{
    fn matmul_range_into(
        &mut self,
        _dst: &mut Vec<f64>,
        _dst_offset: usize,
        _lhs: &Vec<f64>,
        _lhs_offset: usize,
        _rhs: &Vec<f64>,
        _rhs_offset: usize,
        _rows: usize,
        _contracted: usize,
        _cols: usize,
    ) -> Result<(), crate::OperationError> {
        panic!("the storage-direct entry must reject a nonuniform twist before any GEMM")
    }
}

/// Why the device lifts the canonical non-uniform-θ case to DynamicTree: no
/// per-job GEMM alpha expresses a twist that varies within one coupled-sector
/// matrix. The device core route declines it; the Host storage-direct entry
/// keeps its existing error (unchanged); the compiled artifact carries the
/// twist.
#[test]
fn a_canonical_nonuniform_twist_leaves_the_device_core_route_for_dynamic_tree() {
    let (_, case) = fermionic_cases().remove(2);
    let dst = case.dst();
    let (lhs, rhs) = (
        FusionOperand::direct(case.lhs.space()),
        FusionOperand::direct(case.rhs.space()),
    );
    assert!(
        crate::try_compile_storage_contract_core_route(&dst, lhs, rhs, case.axes())
            .unwrap()
            .is_none()
    );
    let resolution = Context::<f64>::default()
        .compile_storage_contract_resolution(&dst, lhs, rhs, case.axes())
        .unwrap();
    assert!(resolution.is_dynamic_tree());
    assert!(resolution.requires_source_twist());

    let lhs_data = host_data(case.lhs.space(), 1);
    let rhs_data = host_data(case.rhs.space(), 2);
    let mut out = vec![0.0; dst.space().required_len().unwrap()];
    let error = crate::tensorcontract_fusion_dyn_prelowered_direct_on_storage(
        &mut RejectingGemm,
        &dst,
        &mut out,
        lhs,
        &lhs_data,
        rhs,
        &rhs_data,
        case.axes(),
    )
    .unwrap_err();
    assert!(
        matches!(
            error,
            crate::OperationError::UnsupportedTensorContractScope { message }
                if message.contains("nonuniform")
        ),
        "{error:?}"
    );
}

#[test]
fn identity_source_fixtures_compile_a_dynamic_tree() {
    for case in [lhs_identity_case(), rhs_identity_case()] {
        let resolution = Context::<f64>::default()
            .compile_storage_contract_resolution(
                &case.dst(),
                FusionOperand::direct(case.lhs.space()),
                FusionOperand::direct(case.rhs.space()),
                case.axes(),
            )
            .unwrap();
        assert!(resolution.is_dynamic_tree());
    }
}

#[test]
fn canonical_axes_keep_the_direct_core_route_and_others_compile_one_dynamic_tree() {
    let case = u1_case();
    let dst = case.dst();
    let mut context = Context::<f64>::default();
    let general = context
        .compile_storage_contract_resolution(
            &dst,
            FusionOperand::direct(case.lhs.space()),
            FusionOperand::direct(case.rhs.space()),
            case.axes(),
        )
        .unwrap();
    // What: arbitrary axes take the Host DynamicTree artifact, bosonic so no
    // core-right twist.
    assert!(general.is_dynamic_tree());
    assert!(!general.requires_source_twist());
    let su2 = su2_case();
    let su2_resolution = Context::<f64>::default()
        .compile_storage_contract_resolution(
            &su2.dst(),
            FusionOperand::direct(su2.lhs.space()),
            FusionOperand::direct(su2.rhs.space()),
            su2.axes(),
        )
        .unwrap();
    assert!(su2_resolution.is_dynamic_tree());

    let provider = Arc::new(U1FusionRule);
    let v = || u1_leg(false);
    let lhs = space(&provider, vec![v(), v()], vec![v()]);
    let rhs = space(&provider, vec![v()], vec![v(), v()]);
    let dst = BoundDynamicFusionMapSpace::contracted_multiplicity_free_ordered(
        &lhs,
        &rhs,
        &[2],
        &[0],
        OutputAxisOrder::identity(),
    )
    .unwrap();
    let canonical = context
        .compile_storage_contract_resolution(
            &dst,
            FusionOperand::direct(lhs.space()),
            FusionOperand::direct(rhs.space()),
            TensorContractSpec::new(&[2], &[0], OutputAxisOrder::identity()),
        )
        .unwrap();
    // What: the TensorKit `mul!` form stays on the fully-direct core route.
    assert!(!canonical.is_dynamic_tree());

    // What: a lazy adjoint with arbitrary axes compiles the prelowered
    // artifact, never the Host's dense `Structure` route.
    let adjoint_bound = lhs.adjoint_view().unwrap();
    let rhs3 = space(&provider, vec![v(), v()], vec![v()]);
    let output = [3, 0, 1, 2];
    let lazy_dst = crate::BoundDynamicFusionMapSpace::contracted_multiplicity_free_ordered(
        &adjoint_bound,
        &rhs3,
        &[1],
        &[0],
        OutputAxisOrder::from_axes(&output),
    )
    .unwrap();
    let lazy = context
        .compile_storage_contract_resolution(
            &lazy_dst,
            FusionOperand::adjoint(lhs.space()),
            FusionOperand::direct(rhs3.space()),
            TensorContractSpec::new_with_conjugation(
                &[1],
                &[0],
                OutputAxisOrder::from_axes(&output),
                true,
                false,
            ),
        )
        .unwrap();
    assert!(lazy.is_dynamic_tree());
    assert!(!lazy.requires_source_twist());
}

/// The geometry of `tenet/tests/contract_cases::su2_structure_cases`: the
/// Host resolves it to its dense `Structure` route, the storage compile
/// entry to the prelowered `DynamicTree` artifact. Pinned so the device
/// fixture cannot silently stop covering the Host-`Structure` class.
#[test]
fn a_self_dual_core_form_lazy_contraction_is_host_structure_and_device_dynamic_tree() {
    let provider = Arc::new(SU2FusionRule);
    let s = su2_leg;
    // Lazy lhs: `X^H` against `rhs` over `X^H`'s whole domain, output bent.
    let x = space(&provider, vec![s(), s()], vec![s(), s()]);
    let rhs = space(&provider, vec![s(), s()], vec![s()]);
    // Lazy rhs: `lhs` against `Y^H` over `Y^H`'s whole codomain, output swapped.
    let lhs = space(&provider, vec![s()], vec![s(), s()]);
    let y = space(&provider, vec![s()], vec![s(), s()]);
    let cases = [
        (
            x.adjoint_view().unwrap(),
            FusionOperand::adjoint(x.space()),
            rhs.clone(),
            FusionOperand::direct(rhs.space()),
            vec![2, 3],
            vec![0, 1],
            vec![2, 0, 1],
            (true, false),
        ),
        (
            lhs.clone(),
            FusionOperand::direct(lhs.space()),
            y.adjoint_view().unwrap(),
            FusionOperand::adjoint(y.space()),
            vec![1, 2],
            vec![0, 1],
            vec![1, 0],
            (false, true),
        ),
    ];
    for (lhs_logical, lhs_operand, rhs_logical, rhs_operand, lhs_axes, rhs_axes, output, conj) in
        cases
    {
        let dst = BoundDynamicFusionMapSpace::contracted_multiplicity_free_ordered(
            &lhs_logical,
            &rhs_logical,
            &lhs_axes,
            &rhs_axes,
            OutputAxisOrder::from_axes(&output),
        )
        .unwrap();
        let axes = TensorContractSpec::new_with_conjugation(
            &lhs_axes,
            &rhs_axes,
            OutputAxisOrder::from_axes(&output),
            conj.0,
            conj.1,
        );
        let lhs_data = vec![0.5; lhs_operand.storage_space().required_len().unwrap()];
        let rhs_data = vec![0.25; rhs_operand.storage_space().required_len().unwrap()];
        let mut out = vec![0.0; dst.space().required_len().unwrap()];
        let mut context = Context::<f64>::default();
        context
            .tensorcontract_fusion_dyn_prelowered_into(
                &dst,
                &mut out,
                lhs_operand,
                &lhs_data,
                rhs_operand,
                &rhs_data,
                axes,
                1.0,
                0.0,
            )
            .unwrap();
        // What: the Host took its `Structure` route.
        assert!(context.last_resolution_is_structure(), "{output:?}");
        assert!(out.iter().any(|&value| value != 0.0));
        let storage = context
            .compile_storage_contract_resolution(&dst, lhs_operand, rhs_operand, axes)
            .unwrap();
        assert!(storage.is_dynamic_tree(), "{output:?}");
    }
}

#[test]
fn a_fermionic_dual_contracted_leg_on_a_transformed_operand_reports_the_twist() {
    let provider = Arc::new(FermionParityFusionRule);
    let odd_even = |dual| SectorLeg::new([(SectorId::new(0), 2), (SectorId::new(1), 2)], dual);
    // `rhs` codomain leg 1 is dual: contracting it is a supertrace twist on
    // the core-right operand.
    let lhs = space(
        &provider,
        vec![odd_even(false), odd_even(false)],
        vec![odd_even(false)],
    );
    let rhs = space(
        &provider,
        vec![odd_even(false), odd_even(true)],
        vec![odd_even(false)],
    );
    let lhs_axes = [2, 0];
    let rhs_axes = [0, 1];
    let dst = BoundDynamicFusionMapSpace::contracted_multiplicity_free_ordered(
        &lhs,
        &rhs,
        &lhs_axes,
        &rhs_axes,
        OutputAxisOrder::identity(),
    )
    .unwrap();
    let mut context = Context::<f64>::default();
    let resolution = context
        .compile_storage_contract_resolution(
            &dst,
            FusionOperand::direct(lhs.space()),
            FusionOperand::direct(rhs.space()),
            TensorContractSpec::new(&lhs_axes, &rhs_axes, OutputAxisOrder::identity()),
        )
        .unwrap();
    // What: the twist travels in the artifact, which the device now replays
    // with destination scales (G2c-2).
    assert!(resolution.is_dynamic_tree());
    assert!(resolution.requires_source_twist());
}

/// Why the device executor needs no zero fill for a core job with
/// `contracted == 0`: no admitted space has one. A zero degeneracy is dropped
/// from the leg, so every operand block has a non-empty contracted extent and
/// a coupled sector with nothing to contract has no job at all — its
/// destination block is then an inactive block, zeroed by the core zeroing
/// rule instead.
#[test]
fn a_zero_degeneracy_sector_never_yields_a_zero_extent_core_job() {
    let provider = Arc::new(U1FusionRule);
    let leg = || {
        SectorLeg::new(
            [
                (U1Irrep::new(0).sector_id(), 1),
                (U1Irrep::new(1).sector_id(), 1),
            ],
            false,
        )
    };
    let empty = || {
        SectorLeg::new(
            [
                (U1Irrep::new(0).sector_id(), 0),
                (U1Irrep::new(1).sector_id(), 2),
            ],
            false,
        )
    };
    let case = Case {
        lhs: space(&provider, vec![leg(), leg()], vec![empty()]),
        rhs: space(&provider, vec![empty()], vec![leg()]),
        lhs_axes: vec![2],
        rhs_axes: vec![0],
        output_axes: vec![1, 0, 2],
    };
    let no_zero_extent = |space: &crate::DynamicFusionMapSpace| {
        let structure = space.structure();
        (0..structure.block_count())
            .all(|index| !structure.block(index).unwrap().shape().contains(&0))
    };
    assert!(no_zero_extent(case.lhs.space()));
    assert!(no_zero_extent(case.rhs.space()));
    let dst = case.dst();
    let resolution = Context::<f64>::default()
        .compile_storage_contract_resolution(
            &dst,
            FusionOperand::direct(case.lhs.space()),
            FusionOperand::direct(case.rhs.space()),
            case.axes(),
        )
        .unwrap();
    assert!(resolution.is_dynamic_tree());
}

/// `wide` carries charge 2 that `narrow` lacks, so a destination block
/// coupled at charge 2 has no GEMM job.
fn wide_leg() -> SectorLeg {
    SectorLeg::new(
        [
            (U1Irrep::new(0).sector_id(), 1),
            (U1Irrep::new(1).sector_id(), 2),
            (U1Irrep::new(2).sector_id(), 1),
        ],
        false,
    )
}

fn narrow_leg(dual: bool) -> SectorLeg {
    let sign = if dual { -1 } else { 1 };
    SectorLeg::new(
        [
            (U1Irrep::new(0).sector_id(), 2),
            (U1Irrep::new(sign).sector_id(), 1),
        ],
        dual,
    )
}

/// One geometry per way the device completes a destination it did not
/// allocate (`contract_overwrite_into`), with the core plan's inactive
/// destination-block count when the core GEMMs write that destination
/// directly (`None`: an output transform writes it).
fn overwrite_cases() -> Vec<(&'static str, Case<U1FusionRule>, Option<usize>)> {
    let provider = Arc::new(U1FusionRule);
    let (v, w) = (wide_leg, || narrow_leg(false));
    let case = |lhs, rhs, lhs_axes: &[usize], rhs_axes: &[usize], output_axes: &[usize]| Case {
        lhs,
        rhs,
        lhs_axes: lhs_axes.to_vec(),
        rhs_axes: rhs_axes.to_vec(),
        output_axes: output_axes.to_vec(),
    };
    vec![
        (
            "core route, inactive block",
            case(
                space(&provider, vec![v(), v()], vec![w()]),
                space(&provider, vec![w()], vec![v()]),
                &[2],
                &[0],
                &[0, 1, 2],
            ),
            Some(1),
        ),
        (
            "transformed lhs, identity output, inactive block",
            case(
                space(&provider, vec![v()], vec![w(), v()]),
                space(&provider, vec![w()], vec![v()]),
                &[1],
                &[0],
                &[0, 1, 2],
            ),
            Some(1),
        ),
        (
            "transformed rhs, identity output, fully covered",
            case(
                space(&provider, vec![v(), v()], vec![w()]),
                space(&provider, vec![v()], vec![narrow_leg(true)]),
                &[2],
                &[1],
                &[0, 1, 2],
            ),
            Some(0),
        ),
        (
            "output transform over an inactive core block",
            case(
                space(&provider, vec![v(), v()], vec![w()]),
                space(&provider, vec![w()], vec![v()]),
                &[2],
                &[0],
                &[1, 0, 2],
            ),
            None,
        ),
    ]
}

#[test]
fn each_way_a_device_overwrite_completes_its_destination_has_a_fixture() {
    // Why: the device gate replays these into a NaN-poisoned destination; it
    // proves the zeroing rule only if the Host compiler really resolves each
    // geometry to the writer it names.
    for (what, case, expected) in overwrite_cases() {
        let resolution = Context::<f64>::default()
            .compile_storage_contract_resolution(
                &case.dst(),
                FusionOperand::direct(case.lhs.space()),
                FusionOperand::direct(case.rhs.space()),
                case.axes(),
            )
            .unwrap();
        assert_eq!(
            resolution.direct_destination_inactive_blocks(),
            expected,
            "{what}"
        );
        assert_eq!(
            what.starts_with("core route"),
            !resolution.is_dynamic_tree(),
            "{what}"
        );
    }
}

#[cfg(feature = "cuda")]
mod device {
    use super::*;

    use num_complex::Complex64;
    use tenet_dense::{CudaDenseContext, CudaScalar};
    use tenet_operations::cuda::CudaStorage;
    use tenet_operations::CudaTreeTransformExecutor;

    use crate::contract::resolution::{StorageContractResolution, StorageContractRoute};
    use crate::CudaContractScratch;

    trait Payload:
        CudaScalar + DenseRecouplingScalar + RecouplingCoefficientAction<f64> + std::fmt::Debug
    {
        fn fill(index: usize) -> Self;
        fn nan() -> Self;
        fn distance(self, other: Self) -> f64;
        fn magnitude(self) -> f64;
    }

    impl Payload for f64 {
        fn fill(index: usize) -> Self {
            (index as f64 * 0.37 + 0.1).sin()
        }
        fn nan() -> Self {
            f64::NAN
        }
        fn distance(self, other: Self) -> f64 {
            (self - other).abs()
        }
        fn magnitude(self) -> f64 {
            self.abs()
        }
    }

    impl Payload for Complex64 {
        fn fill(index: usize) -> Self {
            Complex64::new(
                (index as f64 * 0.37 + 0.1).sin(),
                (index as f64 * 0.23).cos(),
            )
        }
        fn nan() -> Self {
            Complex64::new(f64::NAN, f64::NAN)
        }
        fn distance(self, other: Self) -> f64 {
            (self - other).norm()
        }
        fn magnitude(self) -> f64 {
            self.norm()
        }
    }

    fn data<D: Payload>(space: &crate::DynamicFusionMapSpace, salt: usize) -> Vec<D> {
        (0..space.required_len().unwrap())
            .map(|index| D::fill(index * 7 + salt))
            .collect()
    }

    fn assert_close<D: Payload>(actual: &[D], expected: &[D], what: &str) {
        assert_eq!(actual.len(), expected.len(), "{what}: length");
        assert!(
            expected.iter().any(|value| value.magnitude() > 1e-3),
            "{what}: vacuous all-zero reference"
        );
        for (index, (&left, &right)) in actual.iter().zip(expected).enumerate() {
            assert!(
                left.distance(right) <= 1e-12 * (1.0 + right.magnitude()),
                "{what}: element {index} is {left:?}, expected {right:?}"
            );
        }
    }

    struct Replay<D: Payload> {
        ctx: CudaDenseContext,
        transforms: CudaTreeTransformExecutor,
        scratch: CudaContractScratch,
        _payload: std::marker::PhantomData<D>,
    }

    impl<D: Payload> Replay<D> {
        fn new() -> Self {
            Self {
                ctx: CudaDenseContext::new(0).expect("a CUDA device"),
                transforms: CudaTreeTransformExecutor::default(),
                scratch: CudaContractScratch::default(),
                _payload: std::marker::PhantomData,
            }
        }

        /// Device and Host replays of one artifact for one forced candidate.
        fn run<R>(
            &mut self,
            case: &Case<R>,
            candidate: &crate::contract::fusion::ContractAxisOrderCandidate,
            orientation: FusionContractOrientation,
        ) -> (Vec<D>, Vec<D>, bool, bool)
        where
            R: MultiplicityFreeRigidSymbols<Scalar = f64>
                + crate::TreeTransformRuleCacheKey<Key = RuleIdentity>,
        {
            let (device, host, borrowed, _) = self.run_twisted(case, candidate, orientation);
            (device, host, borrowed.0, borrowed.1)
        }

        /// [`Self::run`], also reporting whether the artifact twisted.
        fn run_twisted<R>(
            &mut self,
            case: &Case<R>,
            candidate: &crate::contract::fusion::ContractAxisOrderCandidate,
            orientation: FusionContractOrientation,
        ) -> (Vec<D>, Vec<D>, (bool, bool), bool)
        where
            R: MultiplicityFreeRigidSymbols<Scalar = f64>
                + crate::TreeTransformRuleCacheKey<Key = RuleIdentity>,
        {
            let dst = case.dst();
            let lhs = data::<D>(case.lhs.space(), 1);
            let rhs = data::<D>(case.rhs.space(), 2);
            let (artifact, host, _) = forced_artifact(case, candidate, orientation, &lhs, &rhs);
            let borrowed = artifact.borrowed_sources();
            let twisted = artifact.requires_source_twist();
            let resolution = StorageContractResolution::new(StorageContractRoute::DynamicTree(
                Arc::new(artifact),
            ))
            .unwrap();
            let device = self.execute(&resolution, &dst, &lhs, &rhs);
            (device, host, borrowed, twisted)
        }

        /// Replays `resolution` into a fresh zero destination and into a
        /// retained NaN-poisoned one (`dst_is_zeroed = false`, the
        /// `contract_overwrite_into` mode), asserts the two agree, and
        /// returns the first.
        fn execute<R>(
            &mut self,
            resolution: &StorageContractResolution<f64>,
            dst: &BoundDynamicFusionMapSpace<R>,
            lhs: &[D],
            rhs: &[D],
        ) -> Vec<D> {
            let ctx = &mut self.ctx;
            let lhs = CudaStorage::<D>::upload(ctx, lhs).unwrap();
            let rhs = CudaStorage::<D>::upload(ctx, rhs).unwrap();
            let len = dst.space().required_len().unwrap();
            let mut results = Vec::new();
            for (fill, dst_is_zeroed) in [(D::ZERO, true), (D::nan(), false)] {
                let mut out = CudaStorage::<D>::upload_owned(ctx, vec![fill; len]).unwrap();
                crate::execute_storage_contract_resolution_on_cuda(
                    ctx,
                    &mut self.transforms,
                    &mut self.scratch,
                    resolution,
                    dst.space().structure(),
                    &mut out,
                    dst_is_zeroed,
                    &lhs,
                    &rhs,
                )
                .unwrap();
                results.push(out.download(ctx).unwrap());
            }
            let overwritten = results.pop().unwrap();
            let zeroed = results.pop().unwrap();
            assert_eq!(overwritten.len(), zeroed.len());
            for (index, (&left, &right)) in overwritten.iter().zip(&zeroed).enumerate() {
                assert!(
                    left.distance(right) <= 1e-12 * (1.0 + right.magnitude()),
                    "retained destination element {index} is {left:?}, fresh {right:?}"
                );
            }
            zeroed
        }
    }

    fn every_candidate_and_orientation<R, D>(case: &Case<R>, what: &str) -> (bool, bool)
    where
        R: MultiplicityFreeRigidSymbols<Scalar = f64>
            + crate::TreeTransformRuleCacheKey<Key = RuleIdentity>,
        D: Payload,
    {
        let mut replay = Replay::<D>::new();
        let mut any_borrow = (false, false);
        for candidate in
            crate::contract::contracted_axis_order_candidates(&case.lhs_axes, &case.rhs_axes)
        {
            for orientation in [
                FusionContractOrientation::LhsRhs,
                FusionContractOrientation::RhsLhs,
            ] {
                let (device, host, lhs_borrowed, rhs_borrowed) =
                    replay.run(case, &candidate, orientation);
                any_borrow.0 |= lhs_borrowed;
                any_borrow.1 |= rhs_borrowed;
                assert_close(
                    &device,
                    &host,
                    &format!("{what} {candidate:?} {orientation:?}"),
                );
            }
        }
        any_borrow
    }

    #[test]
    #[ignore = "requires a real CUDA device"]
    fn every_forced_candidate_and_orientation_matches_the_host_replay() {
        // What: both orientations of every axis-order candidate, U(1) rank 5
        // and SU(2) recoupling, real and complex, device == Host of the same
        // artifact.
        every_candidate_and_orientation::<_, f64>(&u1_case(), "U(1) f64");
        every_candidate_and_orientation::<_, Complex64>(&u1_case(), "U(1) c64");
        every_candidate_and_orientation::<_, f64>(&su2_case(), "SU(2) f64");
        every_candidate_and_orientation::<_, Complex64>(&su2_case(), "SU(2) c64");
    }

    #[test]
    #[ignore = "requires a real CUDA device"]
    fn an_identity_source_transform_borrows_either_operand_on_device() {
        // `lhs` already has its contracted legs as its whole domain in
        // order, `rhs` its as its whole codomain: under LhsRhs the lhs is
        // borrowed, and the permuted output keeps an output transform.
        let lhs_canonical = lhs_identity_case();
        let borrowed = every_candidate_and_orientation::<_, f64>(&lhs_canonical, "lhs borrow");
        let rhs_canonical = rhs_identity_case();
        let rhs_borrowed = every_candidate_and_orientation::<_, f64>(&rhs_canonical, "rhs borrow");
        // What: non-vacuity — some forced artifact really borrowed each side.
        assert!(borrowed.0, "no artifact borrowed the lhs");
        assert!(rhs_borrowed.1, "no artifact borrowed the rhs");
    }

    #[test]
    #[ignore = "requires a real CUDA device"]
    fn a_nan_poisoned_core_destination_scratch_is_zeroed_on_its_inactive_blocks() {
        // `v` carries charge 2 that `w` lacks: the core destination has a
        // (2, 2)-coupled block no GEMM job writes, so a retained scratch that
        // still holds NaN leaks unless exactly those blocks are zeroed.
        let provider = Arc::new(U1FusionRule);
        let v = || {
            SectorLeg::new(
                [
                    (U1Irrep::new(0).sector_id(), 1),
                    (U1Irrep::new(1).sector_id(), 2),
                    (U1Irrep::new(2).sector_id(), 1),
                ],
                false,
            )
        };
        let w = || {
            SectorLeg::new(
                [
                    (U1Irrep::new(0).sector_id(), 2),
                    (U1Irrep::new(1).sector_id(), 1),
                ],
                false,
            )
        };
        let case = Case {
            lhs: space(&provider, vec![v(), v()], vec![w()]),
            rhs: space(&provider, vec![w()], vec![v()]),
            lhs_axes: vec![2],
            rhs_axes: vec![0],
            output_axes: vec![1, 0, 2],
        };
        let mut replay = Replay::<f64>::new();
        let candidate = crate::contract::contracted_axis_order_candidates(&[2], &[0]).remove(0);
        let (first, host, _, _) = replay.run(&case, &candidate, FusionContractOrientation::LhsRhs);
        assert_close(&first, &host, "cold");
        let poisoned = replay
            .scratch
            .poison_core_destination::<f64>(&replay.ctx, f64::nan());
        assert!(poisoned > 0, "the first replay retained a core destination");
        let (second, host, _, _) = replay.run(&case, &candidate, FusionContractOrientation::LhsRhs);
        assert!(second.iter().all(|value| value.is_finite()), "{second:?}");
        assert_close(&second, &host, "after poisoning");
    }

    fn every_fermionic_candidate_and_orientation<R, D>(case: &Case<R>, what: &str)
    where
        R: MultiplicityFreeRigidSymbols<Scalar = f64>
            + crate::TreeTransformRuleCacheKey<Key = RuleIdentity>,
        D: Payload,
    {
        let mut replay = Replay::<D>::new();
        let mut twisted = [false; 2];
        for candidate in
            crate::contract::contracted_axis_order_candidates(&case.lhs_axes, &case.rhs_axes)
        {
            for (slot, orientation) in orientations().into_iter().enumerate() {
                let (device, host, _, twist) = replay.run_twisted(case, &candidate, orientation);
                twisted[slot] |= twist;
                assert_close(
                    &device,
                    &host,
                    &format!("{what} {candidate:?} {orientation:?}"),
                );
            }
        }
        assert!(
            twisted.iter().any(|&twist| twist),
            "{what}: nothing twisted"
        );
    }

    #[test]
    #[ignore = "requires a real CUDA device"]
    fn a_twisted_artifact_replays_with_destination_scales_as_the_host_scales_in_place() {
        // What: for every forced candidate and both orientations — the
        // twisted operand is whichever one the artifact already materializes
        // — the device's θ-scaled source transform
        // equals the Host's transform followed by its in-place twist, on the
        // same artifact; fZ2 x U(1) and fZ2 (x) SU(2) (recoupling), the twist
        // on one or both contracted legs, and the canonical non-uniform form.
        for (what, case) in fermionic_cases() {
            every_fermionic_candidate_and_orientation::<_, f64>(&case, what);
            every_fermionic_candidate_and_orientation::<_, Complex64>(&case, what);
        }
        for (what, case) in fermionic_su2_cases() {
            every_fermionic_candidate_and_orientation::<_, f64>(&case, what);
            every_fermionic_candidate_and_orientation::<_, Complex64>(&case, what);
        }
    }

    #[test]
    #[ignore = "requires a real CUDA device"]
    fn a_retained_destination_is_completed_by_every_destination_writer() {
        // What: into a NaN-poisoned destination the core route and an
        // identity output zero exactly the plan's inactive blocks, a fully
        // covered one needs none, and an output transform writes everything
        // itself — each equal to the eager Host contraction.
        fn at<D: Payload>() {
            let mut replay = Replay::<D>::new();
            for (what, case, _) in overwrite_cases() {
                let dst = case.dst();
                let lhs = data::<D>(case.lhs.space(), 1);
                let rhs = data::<D>(case.rhs.space(), 2);
                let resolution = Context::<D>::default()
                    .compile_storage_contract_resolution(
                        &dst,
                        FusionOperand::direct(case.lhs.space()),
                        FusionOperand::direct(case.rhs.space()),
                        case.axes(),
                    )
                    .unwrap();
                let device = replay.execute(&resolution, &dst, &lhs, &rhs);
                assert_close(&device, &eager_host(&case, &lhs, &rhs), what);
            }
        }
        at::<f64>();
        at::<Complex64>();
    }
}
