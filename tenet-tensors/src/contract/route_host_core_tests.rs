//! #1859 Host H2: Core and CopyC routes on the one Host executor.
//!
//! The eager characterization compares, bit for bit and on the machine
//! running the test, every eager Core / CopyC replay with `e6e9cac8`'s eager
//! step sequence spelled with the replay primitives ([`base_core`],
//! [`base_copy_c`]): alpha/beta placement, `Zeroed`, signed zeros and
//! non-finite operands. The pinned digests fold signed zeros and NaNs (their
//! signs are chosen by the dense backend's architecture-specific kernels) and
//! tie the values to `e6e9cac8` on every platform.

use super::*;

/// A U(1) fixture of each Core / CopyC shape: unswapped Core with an
/// inactive block, swapped Core, and CopyC in both orientations.
fn core_fixtures() -> Vec<(&'static str, Case<U1FusionRule>, RouteKind)> {
    let u1 = Arc::new(U1FusionRule);
    let u1_v = || u1_leg(false);
    let (_, core, _) = overwrite_cases().remove(0);
    let fixtures = vec![
        ("core, inactive block", core, RouteKind::Core),
        (
            "core, swapped",
            square_case(&u1, u1_v, [0, 1], [2, 3], [2, 3, 0, 1]),
            RouteKind::Core,
        ),
        (
            "copyC C1p",
            square_case(&u1, u1_v, [3, 2], [1, 0], [1, 0, 3, 2]),
            RouteKind::CopyC,
        ),
        (
            "copyC C2p",
            square_case(&u1, u1_v, [0, 1], [2, 3], [3, 2, 1, 0]),
            RouteKind::CopyC,
        ),
    ];
    for (name, case, route) in &fixtures {
        assert_eq!(owned_routes(case), (*route, *route), "{name}");
    }
    fixtures
}

fn eager_resolution<R>(case: &Case<R>) -> crate::contract::StorageContractResolution<f64>
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64>
        + crate::TreeTransformRuleCacheKey<Key = RuleIdentity>,
{
    Context::<f64>::default()
        .plan_contract::<crate::contract::HostEagerExecutor, _>(
            &case.dst(),
            FusionOperand::direct(case.lhs.space()),
            FusionOperand::direct(case.rhs.space()),
            &case.lhs_axes,
            &case.rhs_axes,
            &case.output_axes,
        )
        .unwrap()
}

type Operand<'a> = (&'a Arc<tenet_core::BlockStructure>, &'a [f64]);

/// `e6e9cac8`'s eager Core arm: the operands swapped for the B·A candidate,
/// then `execute_raw_zeroed(alpha)` or `execute_raw(alpha, beta)`.
#[allow(clippy::too_many_arguments)]
fn base_core(
    plan: &tenet_operations::fusion_replay::FusionBlockContractPlan<f64>,
    swapped: bool,
    dst_structure: &Arc<tenet_core::BlockStructure>,
    dst: &mut [f64],
    lhs: Operand<'_>,
    rhs: Operand<'_>,
    alpha: f64,
    init: ContractDestinationInit<f64>,
) {
    let ((left_structure, left), (right_structure, right)) =
        if swapped { (rhs, lhs) } else { (lhs, rhs) };
    let mut backend = DenseTreeTransformOperations::default();
    let mut backend_workspace = crate::contract::backend::TensorContractWorkspace::default();
    let mut gemm = BackendRank2Gemm::<_, _, f64>::new(&mut backend, &mut backend_workspace);
    let mut kernels = crate::StridedHostKernelAdapter::default();
    let mut workspace = FusionBlockContractWorkspace::default();
    match init {
        ContractDestinationInit::Zeroed => plan.execute_raw_zeroed(
            &mut kernels,
            &mut gemm,
            &mut workspace,
            dst_structure,
            dst,
            left_structure,
            left,
            right_structure,
            right,
            alpha,
        ),
        ContractDestinationInit::Axpby(beta) => plan.execute_raw(
            &mut kernels,
            &mut gemm,
            &mut workspace,
            dst_structure,
            dst,
            left_structure,
            left,
            right_structure,
            right,
            alpha,
            beta,
        ),
    }
    .unwrap();
}

/// `e6e9cac8`'s eager CopyC arm (TensorKit `blas_contract!` copyC): the core
/// `(1, 0)` into a zeroed temporary, then `overwrite(alpha)` for `Zeroed`
/// or `into(alpha, beta)`.
fn base_copy_c(
    copy: &crate::contract::CopyCRoute<f64>,
    dst_structure: &Arc<tenet_core::BlockStructure>,
    dst: &mut [f64],
    lhs: Operand<'_>,
    rhs: Operand<'_>,
    alpha: f64,
    init: ContractDestinationInit<f64>,
) {
    let mut temporary = vec![0.0; copy.temporary_len];
    base_core(
        &copy.core,
        copy.swapped,
        &copy.temporary,
        &mut temporary,
        lhs,
        rhs,
        1.0,
        ContractDestinationInit::Axpby(0.0),
    );
    let mut tree = TreeTransformExecutionContext::<f64, RuleIdentity>::new(
        DenseTreeTransformOperations::default(),
    );
    match init {
        ContractDestinationInit::Zeroed => tree.tree_transform_structure_overwrite_into_raw(
            &copy.transform,
            dst_structure,
            &copy.temporary,
            dst,
            &temporary,
            alpha,
            &[],
        ),
        ContractDestinationInit::Axpby(beta) => tree.tree_transform_structure_into_raw(
            &copy.transform,
            dst_structure,
            &copy.temporary,
            dst,
            &temporary,
            alpha,
            beta,
        ),
    }
    .unwrap();
}

/// `e6e9cac8`'s eager sequence of a Core or CopyC resolution.
#[allow(clippy::too_many_arguments)]
fn base_route(
    resolution: &crate::contract::StorageContractResolution<f64>,
    dst_structure: &Arc<tenet_core::BlockStructure>,
    dst: &mut [f64],
    lhs: Operand<'_>,
    rhs: Operand<'_>,
    alpha: f64,
    init: ContractDestinationInit<f64>,
) {
    match &resolution.route {
        crate::contract::resolution::ContractRoute::Core { plan, swapped } => {
            base_core(plan, *swapped, dst_structure, dst, lhs, rhs, alpha, init)
        }
        crate::contract::resolution::ContractRoute::CopyC(copy) => {
            base_copy_c(copy, dst_structure, dst, lhs, rhs, alpha, init)
        }
        crate::contract::resolution::ContractRoute::DynamicTree(_) => {
            panic!("Core / CopyC fixture planned DynamicTree")
        }
    }
}

/// Every eager `(alpha, init)` replay plus non-finite operand replays, each
/// bit-identical to [`base_route`]; the portable digest of all outputs.
fn eager_core_digest<R>(case: &Case<R>, route: RouteKind) -> u64
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64>
        + crate::TreeTransformRuleCacheKey<Key = RuleIdentity>,
{
    let dst = case.dst();
    let len = dst.space().required_len().unwrap();
    let lhs = exact_data(case.lhs.space().required_len().unwrap(), 1);
    let rhs = exact_data(case.rhs.space().required_len().unwrap(), 3);
    let resolution = eager_resolution(case);
    let mut digest = FNV_START;
    let mut context = Context::<f64>::default();
    let mut run = |lhs: &[f64], alpha: f64, init: ContractDestinationInit<f64>, out: Vec<f64>| {
        let mut expected = out.clone();
        base_route(
            &resolution,
            dst.space().structure(),
            &mut expected,
            (case.lhs.space().structure(), lhs),
            (case.rhs.space().structure(), &rhs),
            alpha,
            init,
        );
        let mut actual = out;
        context
            .tensorcontract_fusion_dyn_prelowered_into_with_init(
                &dst,
                &mut actual,
                FusionOperand::direct(case.lhs.space()),
                lhs,
                FusionOperand::direct(case.rhs.space()),
                &rhs,
                case.axes(),
                alpha,
                init,
            )
            .unwrap();
        assert_eq!(
            context.last_resolution_is_core(),
            route == RouteKind::Core,
            "{route:?}"
        );
        assert_eq!(bits(&actual), bits(&expected), "alpha {alpha}, {init:?}");
        fnv(&mut digest, &actual);
    };
    for alpha in [0.0, 1.0, -1.0, 2.5] {
        for (init, read) in inits() {
            let out = match (init, read) {
                (ContractDestinationInit::Zeroed, _) => vec![0.0; len],
                (_, None) => vec![f64::NAN; len],
                (_, Some(_)) => exact_data(len, 11),
            };
            run(&lhs, alpha, init, out);
        }
    }
    let mut poisoned = lhs.clone();
    poisoned[0] = f64::NAN;
    if poisoned.len() > 1 {
        poisoned[1] = f64::INFINITY;
    }
    for alpha in [0.0, 1.0] {
        run(
            &poisoned,
            alpha,
            ContractDestinationInit::Axpby(1.0),
            exact_data(len, 11),
        );
    }
    digest
}

/// Portable digests (see [`portable_bits`]) of the eager Core / CopyC
/// replays, recorded on `e6e9cac8`.
const EAGER_CORE_PINNED: [(&str, u64); 4] = [
    ("core, inactive block", 0xa9b1_d5db_d80a_fb07),
    ("core, swapped", 0xf11f_58af_16f1_4d74),
    ("copyC C1p", 0x2be7_3679_de6f_8c4e),
    ("copyC C2p", 0x9258_b27c_c589_c856),
];

#[test]
fn eager_core_and_copy_c_bits_are_pinned() {
    let observed = core_fixtures()
        .iter()
        .map(|(name, case, route)| (*name, eager_core_digest(case, *route)))
        .collect::<Vec<_>>();
    assert_eq!(
        observed, EAGER_CORE_PINNED,
        "observed digests: {observed:#x?}"
    );
}
