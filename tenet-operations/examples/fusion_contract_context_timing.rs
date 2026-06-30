use std::hint::black_box;
use std::time::{Duration, Instant};

use tenet_core::{
    FusionTensorMapSpace, FusionTreeHomSpace, SU2FusionRule, TensorMap, TensorMapSpace,
};
use tenet_operations::{
    tensorcontract_fusion_explicit_plan, tensorcontract_fusion_explicit_plan_into,
    tensorcontract_fusion_into, TensorContractAxisSpec, TensorContractFusionExecutionContext,
    TensorContractFusionExplicitPlan, TreeTransformBuiltinRuleCacheKey,
};

static LHS_CONTRACTING_AXES: [usize; 3] = [0, 1, 2];
static RHS_CONTRACTING_AXES: [usize; 3] = [1, 2, 3];

const MANUAL_ITERS: usize = 20_000;
const ONESHOT_ITERS: usize = 5_000;
const COLD_CONTEXT_ITERS: usize = 2_000;
const WARM_CONTEXT_ITERS: usize = 100_000;

fn main() {
    let fixture = Su2NoncanonicalFixture::new();
    let expected = fixture.manual_once();

    let (manual, manual_data) = fixture.time_manual(MANUAL_ITERS);
    assert_close(&manual_data, &expected);

    let (oneshot, oneshot_data) = fixture.time_oneshot(ONESHOT_ITERS);
    assert_close(&oneshot_data, &expected);

    let (context_cold, context_cold_data) = fixture.time_context_cold(COLD_CONTEXT_ITERS);
    assert_close(&context_cold_data, &expected);

    let mut context =
        TensorContractFusionExecutionContext::<f64, TreeTransformBuiltinRuleCacheKey>::default();
    assert_close(&fixture.context_warm_once(&mut context), &expected);
    let (warm, warm_data) = fixture.time_context_warm(&mut context, WARM_CONTEXT_ITERS);
    assert_close(&warm_data, &expected);

    println!("fusion contraction timing (release)");
    println!("fixture,su2_noncanonical_source_degeneracy");
    println!("manual_explicit_ns,{:.3}", nanos_per(manual, MANUAL_ITERS));
    println!("one_shot_ns,{:.3}", nanos_per(oneshot, ONESHOT_ITERS));
    println!(
        "context_cold_ns,{:.3}",
        nanos_per(context_cold, COLD_CONTEXT_ITERS)
    );
    println!("context_warm_ns,{:.3}", nanos_per(warm, WARM_CONTEXT_ITERS));
    println!(
        "context_warm_vs_manual,{:.3}",
        nanos_per(warm, WARM_CONTEXT_ITERS) / nanos_per(manual, MANUAL_ITERS)
    );
    println!(
        "context_warm_vs_oneshot,{:.3}",
        nanos_per(warm, WARM_CONTEXT_ITERS) / nanos_per(oneshot, ONESHOT_ITERS)
    );
    println!(
        "tree_plan_cache,hits={},misses={},len={}",
        context.tree_context().cache().stats().plan_hits(),
        context.tree_context().cache().stats().plan_misses(),
        context.tree_context().cache().plan_len()
    );
    println!(
        "tree_structure_cache,hits={},misses={},len={}",
        context.tree_context().cache().stats().structure_hits(),
        context.tree_context().cache().stats().structure_misses(),
        context.tree_context().cache().structure_len()
    );
    println!(
        "fusion_plan_cache,hits={},misses={},len={}",
        context.fusion_plan_cache_stats().hits(),
        context.fusion_plan_cache_stats().misses(),
        context.fusion_plan_cache_len()
    );
    println!(
        "contract_structure_cache,hits={},misses={},len={}",
        context.contract_cache().stats().structure_hits(),
        context.contract_cache().stats().structure_misses(),
        context.contract_cache().structure_len()
    );
}

struct Su2NoncanonicalFixture {
    lhs: TensorMap<f64, 3, 1>,
    rhs: TensorMap<f64, 1, 3>,
    dst_space: FusionTensorMapSpace<1, 1>,
    lhs_canonical_space: FusionTensorMapSpace<1, 3>,
    rhs_canonical_space: FusionTensorMapSpace<3, 1>,
    plan: TensorContractFusionExplicitPlan,
    initial_dst: Vec<f64>,
    alpha: f64,
    beta: f64,
}

impl Su2NoncanonicalFixture {
    fn new() -> Self {
        let rule = SU2FusionRule;
        let lhs_hom = FusionTreeHomSpace::from_sector_ids([1, 1, 1], [1]);
        let rhs_hom = FusionTreeHomSpace::from_sector_ids([1], [1, 1, 1]);
        let lhs_canonical_hom = lhs_hom
            .permute(&rule, &[3], &[0, 1, 2])
            .expect("valid lhs canonical tree-pair transform");
        let rhs_canonical_hom = rhs_hom
            .permute(&rule, &[1, 2, 3], &[0])
            .expect("valid rhs canonical tree-pair transform");
        let dst_hom = FusionTreeHomSpace::tensorcontract_homspace(
            &rule,
            &lhs_hom,
            &rhs_hom,
            axes().lhs_contracting_axes(),
            axes().rhs_contracting_axes(),
            &[0, 1],
            1,
        )
        .unwrap();

        let lhs_space = FusionTensorMapSpace::from_degeneracy_shapes(
            TensorMapSpace::<3, 1>::from_dims([2, 2, 2], [2]).unwrap(),
            lhs_hom,
            &rule,
            [vec![2, 2, 2, 2], vec![2, 2, 2, 2]],
        )
        .unwrap();
        let rhs_space = FusionTensorMapSpace::from_degeneracy_shapes(
            TensorMapSpace::<1, 3>::from_dims([2], [2, 2, 2]).unwrap(),
            rhs_hom,
            &rule,
            [vec![2, 2, 2, 2], vec![2, 2, 2, 2]],
        )
        .unwrap();
        let lhs_canonical_space = FusionTensorMapSpace::from_degeneracy_shapes(
            TensorMapSpace::<1, 3>::from_dims([2], [2, 2, 2]).unwrap(),
            lhs_canonical_hom,
            &rule,
            [vec![2, 2, 2, 2], vec![2, 2, 2, 2]],
        )
        .unwrap();
        let rhs_canonical_space = FusionTensorMapSpace::from_degeneracy_shapes(
            TensorMapSpace::<3, 1>::from_dims([2, 2, 2], [2]).unwrap(),
            rhs_canonical_hom,
            &rule,
            [vec![2, 2, 2, 2], vec![2, 2, 2, 2]],
        )
        .unwrap();
        let dst_space = FusionTensorMapSpace::from_degeneracy_shapes(
            TensorMapSpace::<1, 1>::from_dims([2], [2]).unwrap(),
            dst_hom,
            &rule,
            [vec![2, 2]],
        )
        .unwrap();
        let lhs_data = (0..32)
            .map(|index| 1.0 + 0.125 * index as f64)
            .collect::<Vec<_>>();
        let rhs_data = (0..32)
            .map(|index| -3.0 + 0.25 * index as f64)
            .collect::<Vec<_>>();
        let lhs = TensorMap::<f64, 3, 1>::from_vec_with_fusion_space(lhs_data, lhs_space).unwrap();
        let rhs = TensorMap::<f64, 1, 3>::from_vec_with_fusion_space(rhs_data, rhs_space).unwrap();
        let plan = tensorcontract_fusion_explicit_plan(
            &rule,
            &dst_space,
            lhs.fusion_space().unwrap(),
            rhs.fusion_space().unwrap(),
            axes(),
        )
        .unwrap();
        Self {
            lhs,
            rhs,
            dst_space,
            lhs_canonical_space,
            rhs_canonical_space,
            plan,
            initial_dst: vec![2.0, -1.0, 4.0, -3.0],
            alpha: -1.5,
            beta: 0.25,
        }
    }

    fn manual_once(&self) -> Vec<f64> {
        let mut dst = self.dst();
        let mut lhs_canonical = self.lhs_canonical();
        let mut rhs_canonical = self.rhs_canonical();
        self.manual_into(&mut dst, &mut lhs_canonical, &mut rhs_canonical);
        dst.data().to_vec()
    }

    fn time_manual(&self, iterations: usize) -> (Duration, Vec<f64>) {
        let mut dst = self.dst();
        let mut lhs_canonical = self.lhs_canonical();
        let mut rhs_canonical = self.rhs_canonical();
        let elapsed = time_loop(iterations, || {
            dst.data_mut().copy_from_slice(&self.initial_dst);
            self.manual_into(&mut dst, &mut lhs_canonical, &mut rhs_canonical);
            black_box(checksum(dst.data()));
        });
        (elapsed, dst.data().to_vec())
    }

    fn time_oneshot(&self, iterations: usize) -> (Duration, Vec<f64>) {
        let rule = SU2FusionRule;
        let mut dst = self.dst();
        let elapsed = time_loop(iterations, || {
            dst.data_mut().copy_from_slice(&self.initial_dst);
            tensorcontract_fusion_into(
                &rule,
                &mut dst,
                &self.lhs,
                &self.rhs,
                axes(),
                self.alpha,
                self.beta,
            )
            .unwrap();
            black_box(checksum(dst.data()));
        });
        (elapsed, dst.data().to_vec())
    }

    fn time_context_cold(&self, iterations: usize) -> (Duration, Vec<f64>) {
        let rule = SU2FusionRule;
        let mut dst = self.dst();
        let elapsed = time_loop(iterations, || {
            let mut context = TensorContractFusionExecutionContext::<
                f64,
                TreeTransformBuiltinRuleCacheKey,
            >::default();
            dst.data_mut().copy_from_slice(&self.initial_dst);
            context
                .tensorcontract_fusion_into(
                    &rule,
                    &mut dst,
                    &self.lhs,
                    &self.rhs,
                    axes(),
                    self.alpha,
                    self.beta,
                )
                .unwrap();
            black_box(checksum(dst.data()));
        });
        (elapsed, dst.data().to_vec())
    }

    fn time_context_warm(
        &self,
        context: &mut TensorContractFusionExecutionContext<f64, TreeTransformBuiltinRuleCacheKey>,
        iterations: usize,
    ) -> (Duration, Vec<f64>) {
        let rule = SU2FusionRule;
        let mut dst = self.dst();
        let elapsed = time_loop(iterations, || {
            dst.data_mut().copy_from_slice(&self.initial_dst);
            context
                .tensorcontract_fusion_into(
                    &rule,
                    &mut dst,
                    &self.lhs,
                    &self.rhs,
                    axes(),
                    self.alpha,
                    self.beta,
                )
                .unwrap();
            black_box(checksum(dst.data()));
        });
        (elapsed, dst.data().to_vec())
    }

    fn manual_into(
        &self,
        dst: &mut TensorMap<f64, 1, 1>,
        lhs_canonical: &mut TensorMap<f64, 1, 3>,
        rhs_canonical: &mut TensorMap<f64, 3, 1>,
    ) {
        let rule = SU2FusionRule;
        tensorcontract_fusion_explicit_plan_into(
            &rule,
            &self.plan,
            dst,
            lhs_canonical,
            rhs_canonical,
            &self.lhs,
            &self.rhs,
            self.alpha,
            self.beta,
        )
        .unwrap();
    }

    fn context_warm_once(
        &self,
        context: &mut TensorContractFusionExecutionContext<f64, TreeTransformBuiltinRuleCacheKey>,
    ) -> Vec<f64> {
        let rule = SU2FusionRule;
        let mut dst = self.dst();
        context
            .tensorcontract_fusion_into(
                &rule,
                &mut dst,
                &self.lhs,
                &self.rhs,
                axes(),
                self.alpha,
                self.beta,
            )
            .unwrap();
        dst.data().to_vec()
    }

    fn dst(&self) -> TensorMap<f64, 1, 1> {
        TensorMap::<f64, 1, 1>::from_vec_with_fusion_space(
            self.initial_dst.clone(),
            self.dst_space.clone(),
        )
        .unwrap()
    }

    fn lhs_canonical(&self) -> TensorMap<f64, 1, 3> {
        TensorMap::<f64, 1, 3>::from_vec_with_fusion_space(
            vec![0.0; self.lhs_canonical_space.required_len().unwrap()],
            self.lhs_canonical_space.clone(),
        )
        .unwrap()
    }

    fn rhs_canonical(&self) -> TensorMap<f64, 3, 1> {
        TensorMap::<f64, 3, 1>::from_vec_with_fusion_space(
            vec![0.0; self.rhs_canonical_space.required_len().unwrap()],
            self.rhs_canonical_space.clone(),
        )
        .unwrap()
    }
}

fn axes() -> TensorContractAxisSpec<'static> {
    TensorContractAxisSpec::canonical(&LHS_CONTRACTING_AXES, &RHS_CONTRACTING_AXES)
}

fn time_loop<F>(iterations: usize, mut f: F) -> Duration
where
    F: FnMut(),
{
    let start = Instant::now();
    for _ in 0..iterations {
        f();
    }
    start.elapsed()
}

fn nanos_per(duration: Duration, iterations: usize) -> f64 {
    duration.as_secs_f64() * 1.0e9 / iterations as f64
}

fn checksum(data: &[f64]) -> f64 {
    data.iter().copied().sum()
}

fn assert_close(actual: &[f64], expected: &[f64]) {
    for (&actual, &expected) in actual.iter().zip(expected) {
        assert!(
            (actual - expected).abs() < 1.0e-10,
            "actual {actual} expected {expected}"
        );
    }
}
