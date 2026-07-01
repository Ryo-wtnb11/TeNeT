use std::hint::black_box;
use std::time::{Duration, Instant};

use num_complex::Complex64;
use tenet_core::{
    BlockKey, BlockStructure, FermionParityFusionRule, FusionProductSpace, FusionTensorMapSpace,
    FusionTreeHomSpace, ProductFusionRule, SU2FusionRule, SU2Irrep, SectorId, SectorLeg, TensorMap,
    TensorMapSpace, U1FusionRule, U1Irrep,
};
use tenet_operations::{
    tensorcontract_fusion_explicit_plan, tensorcontract_fusion_explicit_plan_into,
    tensorcontract_fusion_explicit_plan_into_canonical_dst, tensorcontract_fusion_into,
    tree_pair_transform_into_with_context, AxisPermutation, TensorContractAxisSpec,
    TensorContractFusionExecutionContext, TensorContractFusionExplicitPlan,
    TreeTransformBuiltinRuleCacheKey, TreeTransformExecutionContext, TreeTransformRuleCacheKey,
};

static LHS_CONTRACTING_AXES: [usize; 3] = [0, 1, 2];
static RHS_CONTRACTING_AXES: [usize; 3] = [1, 2, 3];
static CANONICAL_LHS_CONTRACTING_AXES: [usize; 3] = [1, 2, 3];
static CANONICAL_RHS_CONTRACTING_AXES: [usize; 3] = [0, 1, 2];

const MANUAL_ITERS: usize = 20_000;
const ONESHOT_ITERS: usize = 5_000;
const COLD_CONTEXT_ITERS: usize = 2_000;
const WARM_CONTEXT_ITERS: usize = 100_000;

type FpU1Rule = ProductFusionRule<FermionParityFusionRule, U1FusionRule>;
type FpU1Su2Rule = ProductFusionRule<FpU1Rule, SU2FusionRule>;

fn main() {
    bench_su2_noncanonical_source();
    bench_su2_output_scratch();
    bench_product_complex();
}

fn bench_su2_noncanonical_source() {
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
    let (source_transform, source_checksum) = fixture.time_source_transforms(WARM_CONTEXT_ITERS);
    let (canonical_contract, contract_data) = fixture.time_canonical_contract(WARM_CONTEXT_ITERS);
    assert_close(&contract_data, &expected);

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
        "source_transform_warm_ns,{:.3}",
        nanos_per(source_transform, WARM_CONTEXT_ITERS)
    );
    println!(
        "canonical_contract_warm_ns,{:.3}",
        nanos_per(canonical_contract, WARM_CONTEXT_ITERS)
    );
    println!("result_checksum,{:.12}", checksum(&warm_data));
    println!("source_transform_checksum,{source_checksum:.12}");
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
        "contract_structure_cache,hits={},misses={},len={}",
        context.contract_cache().stats().structure_hits(),
        context.contract_cache().stats().structure_misses(),
        context.contract_cache().structure_len()
    );
    println!(
        "fusion_block_contract_cache,hits={},misses={},len={}",
        context.fusion_block_contract_cache_hits(),
        context.fusion_block_contract_cache_misses(),
        context.fusion_block_contract_cache_len()
    );
    println!(
        "fusion_execution_plan_cache,hits={},misses={},len={}",
        context.fusion_execution_plan_cache_hits(),
        context.fusion_execution_plan_cache_misses(),
        context.fusion_execution_plan_cache_len()
    );
}

fn bench_su2_output_scratch() {
    let fixture = Su2OutputScratchFixture::new();
    let expected = fixture.manual_once();
    let (manual, manual_data) = fixture.time_manual(MANUAL_ITERS);
    assert_close(&manual_data, &expected);
    let (context_cold, cold_data) = fixture.time_context_cold(COLD_CONTEXT_ITERS);
    assert_close(&cold_data, &expected);
    let mut context =
        TensorContractFusionExecutionContext::<f64, TreeTransformBuiltinRuleCacheKey>::default();
    assert_close(&fixture.context_warm_once(&mut context), &expected);
    let (warm, warm_data) = fixture.time_context_warm(&mut context, WARM_CONTEXT_ITERS);
    assert_close(&warm_data, &expected);
    let (output_transform, output_data) = fixture.time_output_transform(WARM_CONTEXT_ITERS);
    assert_close(&output_data, &expected);

    println!();
    println!("fixture,su2_output_transform_canonical_dst_scratch");
    println!("manual_explicit_ns,{:.3}", nanos_per(manual, MANUAL_ITERS));
    println!(
        "context_cold_ns,{:.3}",
        nanos_per(context_cold, COLD_CONTEXT_ITERS)
    );
    println!("context_warm_ns,{:.3}", nanos_per(warm, WARM_CONTEXT_ITERS));
    println!(
        "output_transform_warm_ns,{:.3}",
        nanos_per(output_transform, WARM_CONTEXT_ITERS)
    );
    println!("result_checksum,{:.12}", checksum(&warm_data));
    println!(
        "tree_plan_cache,hits={},misses={},len={}",
        context.tree_context().cache().stats().plan_hits(),
        context.tree_context().cache().stats().plan_misses(),
        context.tree_context().cache().plan_len()
    );
    println!(
        "contract_structure_cache,hits={},misses={},len={}",
        context.contract_cache().stats().structure_hits(),
        context.contract_cache().stats().structure_misses(),
        context.contract_cache().structure_len()
    );
    println!(
        "fusion_block_contract_cache,hits={},misses={},len={}",
        context.fusion_block_contract_cache_hits(),
        context.fusion_block_contract_cache_misses(),
        context.fusion_block_contract_cache_len()
    );
    println!(
        "fusion_execution_plan_cache,hits={},misses={},len={}",
        context.fusion_execution_plan_cache_hits(),
        context.fusion_execution_plan_cache_misses(),
        context.fusion_execution_plan_cache_len()
    );
}

fn bench_product_complex() {
    let fixture = ProductComplexFixture::new();
    let expected = fixture.manual_once();
    let (manual, manual_data) = fixture.time_manual(MANUAL_ITERS);
    assert_close_complex(&manual_data, &expected);
    let (oneshot, oneshot_data) = fixture.time_oneshot(ONESHOT_ITERS);
    assert_close_complex(&oneshot_data, &expected);
    let (context_cold, cold_data) = fixture.time_context_cold(COLD_CONTEXT_ITERS);
    assert_close_complex(&cold_data, &expected);
    let mut context = TensorContractFusionExecutionContext::<
        Complex64,
        <FpU1Su2Rule as TreeTransformRuleCacheKey>::Key,
    >::default();
    assert_close_complex(&fixture.context_warm_once(&mut context), &expected);
    let (warm, warm_data) = fixture.time_context_warm(&mut context, WARM_CONTEXT_ITERS);
    assert_close_complex(&warm_data, &expected);

    println!();
    println!("fixture,product_fz2_u1_su2_complex");
    println!("manual_explicit_ns,{:.3}", nanos_per(manual, MANUAL_ITERS));
    println!("one_shot_ns,{:.3}", nanos_per(oneshot, ONESHOT_ITERS));
    println!(
        "context_cold_ns,{:.3}",
        nanos_per(context_cold, COLD_CONTEXT_ITERS)
    );
    println!("context_warm_ns,{:.3}", nanos_per(warm, WARM_CONTEXT_ITERS));
    println!("result_checksum,{:.12}", checksum_complex(&warm_data));
    println!(
        "tree_plan_cache,hits={},misses={},len={}",
        context.tree_context().cache().stats().plan_hits(),
        context.tree_context().cache().stats().plan_misses(),
        context.tree_context().cache().plan_len()
    );
    println!(
        "contract_structure_cache,hits={},misses={},len={}",
        context.contract_cache().stats().structure_hits(),
        context.contract_cache().stats().structure_misses(),
        context.contract_cache().structure_len()
    );
    println!(
        "fusion_block_contract_cache,hits={},misses={},len={}",
        context.fusion_block_contract_cache_hits(),
        context.fusion_block_contract_cache_misses(),
        context.fusion_block_contract_cache_len()
    );
    println!(
        "fusion_execution_plan_cache,hits={},misses={},len={}",
        context.fusion_execution_plan_cache_hits(),
        context.fusion_execution_plan_cache_misses(),
        context.fusion_execution_plan_cache_len()
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

    fn time_source_transforms(&self, iterations: usize) -> (Duration, f64) {
        let rule = SU2FusionRule;
        let mut lhs_canonical = self.lhs_canonical();
        let mut rhs_canonical = self.rhs_canonical();
        let mut context =
            TreeTransformExecutionContext::<f64, TreeTransformBuiltinRuleCacheKey>::default();
        tree_pair_transform_into_with_context(
            &mut context,
            &rule,
            self.plan.lhs_transform().clone(),
            &mut lhs_canonical,
            &self.lhs,
            1.0,
            0.0,
        )
        .unwrap();
        tree_pair_transform_into_with_context(
            &mut context,
            &rule,
            self.plan.rhs_transform().clone(),
            &mut rhs_canonical,
            &self.rhs,
            1.0,
            0.0,
        )
        .unwrap();
        let elapsed = time_loop(iterations, || {
            tree_pair_transform_into_with_context(
                &mut context,
                &rule,
                self.plan.lhs_transform().clone(),
                &mut lhs_canonical,
                &self.lhs,
                1.0,
                0.0,
            )
            .unwrap();
            tree_pair_transform_into_with_context(
                &mut context,
                &rule,
                self.plan.rhs_transform().clone(),
                &mut rhs_canonical,
                &self.rhs,
                1.0,
                0.0,
            )
            .unwrap();
            black_box(checksum(lhs_canonical.data()) + checksum(rhs_canonical.data()));
        });
        (
            elapsed,
            checksum(lhs_canonical.data()) + checksum(rhs_canonical.data()),
        )
    }

    fn time_canonical_contract(&self, iterations: usize) -> (Duration, Vec<f64>) {
        let rule = SU2FusionRule;
        let mut lhs_canonical = self.lhs_canonical();
        let mut rhs_canonical = self.rhs_canonical();
        let mut dst = self.dst();
        self.transform_sources_into(&mut lhs_canonical, &mut rhs_canonical);
        let mut context =
            TensorContractFusionExecutionContext::<f64, TreeTransformBuiltinRuleCacheKey>::default(
            );
        context
            .tensorcontract_fusion_into(
                &rule,
                &mut dst,
                &lhs_canonical,
                &rhs_canonical,
                canonical_axes(),
                self.alpha,
                self.beta,
            )
            .unwrap();
        let elapsed = time_loop(iterations, || {
            dst.data_mut().copy_from_slice(&self.initial_dst);
            context
                .tensorcontract_fusion_into(
                    &rule,
                    &mut dst,
                    &lhs_canonical,
                    &rhs_canonical,
                    canonical_axes(),
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

    fn transform_sources_into(
        &self,
        lhs_canonical: &mut TensorMap<f64, 1, 3>,
        rhs_canonical: &mut TensorMap<f64, 3, 1>,
    ) {
        let rule = SU2FusionRule;
        let mut context =
            TreeTransformExecutionContext::<f64, TreeTransformBuiltinRuleCacheKey>::default();
        tree_pair_transform_into_with_context(
            &mut context,
            &rule,
            self.plan.lhs_transform().clone(),
            lhs_canonical,
            &self.lhs,
            1.0,
            0.0,
        )
        .unwrap();
        tree_pair_transform_into_with_context(
            &mut context,
            &rule,
            self.plan.rhs_transform().clone(),
            rhs_canonical,
            &self.rhs,
            1.0,
            0.0,
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

struct Su2OutputScratchFixture {
    lhs: TensorMap<f64, 2, 2>,
    rhs: TensorMap<f64, 0, 0>,
    dst_space: FusionTensorMapSpace<4, 0>,
    canonical_dst_space: FusionTensorMapSpace<4, 0>,
    lhs_canonical_space: FusionTensorMapSpace<4, 0>,
    rhs_canonical_space: FusionTensorMapSpace<0, 0>,
    plan: TensorContractFusionExplicitPlan,
    initial_dst: Vec<f64>,
    alpha: f64,
    beta: f64,
}

impl Su2OutputScratchFixture {
    fn new() -> Self {
        let rule = SU2FusionRule;
        let lhs_hom = FusionTreeHomSpace::from_sector_ids([1, 1], [1, 1]);
        let lhs_canonical_hom = lhs_hom
            .permute(&rule, &[0, 1, 2, 3], &[])
            .expect("valid all-open canonical transform");
        let dst_hom = lhs_hom
            .permute(&rule, &[0, 2, 1, 3], &[])
            .expect("valid nonidentity output transform");
        let rhs_hom = FusionTreeHomSpace::from_sector_ids([], []);
        let lhs_space = fusion_space_from_hom::<2, 2>(
            TensorMapSpace::<2, 2>::from_dims([1, 1], [1, 1]).unwrap(),
            lhs_hom,
            &rule,
            1,
        );
        let lhs_canonical_space = fusion_space_from_hom::<4, 0>(
            TensorMapSpace::<4, 0>::from_dims([1, 1, 1, 1], []).unwrap(),
            lhs_canonical_hom.clone(),
            &rule,
            1,
        );
        let canonical_dst_space = lhs_canonical_space.clone();
        let dst_space = fusion_space_from_hom::<4, 0>(
            TensorMapSpace::<4, 0>::from_dims([1, 1, 1, 1], []).unwrap(),
            dst_hom,
            &rule,
            1,
        );
        let rhs_canonical_space = FusionTensorMapSpace::from_degeneracy_shapes(
            TensorMapSpace::<0, 0>::from_dims([], []).unwrap(),
            rhs_hom,
            &rule,
            [vec![]],
        )
        .unwrap();
        let lhs_len = lhs_space.required_len().unwrap();
        let dst_len = dst_space.required_len().unwrap();
        let lhs_data = (0..lhs_len)
            .map(|index| 1.0 + 0.25 * index as f64)
            .collect::<Vec<_>>();
        let lhs = TensorMap::<f64, 2, 2>::from_vec_with_fusion_space(lhs_data, lhs_space).unwrap();
        let rhs = TensorMap::<f64, 0, 0>::from_vec_with_fusion_space(
            vec![2.0],
            rhs_canonical_space.clone(),
        )
        .unwrap();
        let initial_dst = (0..dst_len)
            .map(|index| 0.5 + index as f64)
            .collect::<Vec<_>>();
        let plan = tensorcontract_fusion_explicit_plan(
            &rule,
            &dst_space,
            lhs.fusion_space().unwrap(),
            rhs.fusion_space().unwrap(),
            output_axes(),
        )
        .unwrap();
        Self {
            lhs,
            rhs,
            dst_space,
            canonical_dst_space,
            lhs_canonical_space,
            rhs_canonical_space,
            plan,
            initial_dst,
            alpha: -0.75,
            beta: 0.5,
        }
    }

    fn manual_once(&self) -> Vec<f64> {
        let mut dst = self.dst();
        let mut canonical_dst = self.canonical_dst();
        let mut lhs_canonical = self.lhs_canonical();
        let mut rhs_canonical = self.rhs_canonical();
        self.manual_into(
            &mut dst,
            &mut canonical_dst,
            &mut lhs_canonical,
            &mut rhs_canonical,
        );
        dst.data().to_vec()
    }

    fn time_manual(&self, iterations: usize) -> (Duration, Vec<f64>) {
        let mut dst = self.dst();
        let mut canonical_dst = self.canonical_dst();
        let mut lhs_canonical = self.lhs_canonical();
        let mut rhs_canonical = self.rhs_canonical();
        let elapsed = time_loop(iterations, || {
            dst.data_mut().copy_from_slice(&self.initial_dst);
            self.manual_into(
                &mut dst,
                &mut canonical_dst,
                &mut lhs_canonical,
                &mut rhs_canonical,
            );
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
                    output_axes(),
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
                    output_axes(),
                    self.alpha,
                    self.beta,
                )
                .unwrap();
            black_box(checksum(dst.data()));
        });
        (elapsed, dst.data().to_vec())
    }

    fn time_output_transform(&self, iterations: usize) -> (Duration, Vec<f64>) {
        let mut dst = self.dst();
        let mut canonical_dst = self.canonical_dst();
        let mut lhs_canonical = self.lhs_canonical();
        let mut rhs_canonical = self.rhs_canonical();
        self.materialize_canonical_dst(&mut canonical_dst, &mut lhs_canonical, &mut rhs_canonical);
        let rule = SU2FusionRule;
        let mut context =
            TreeTransformExecutionContext::<f64, TreeTransformBuiltinRuleCacheKey>::default();
        tree_pair_transform_into_with_context(
            &mut context,
            &rule,
            self.plan.output_transform().clone(),
            &mut dst,
            &canonical_dst,
            1.0,
            self.beta,
        )
        .unwrap();
        let elapsed = time_loop(iterations, || {
            dst.data_mut().copy_from_slice(&self.initial_dst);
            tree_pair_transform_into_with_context(
                &mut context,
                &rule,
                self.plan.output_transform().clone(),
                &mut dst,
                &canonical_dst,
                1.0,
                self.beta,
            )
            .unwrap();
            black_box(checksum(dst.data()));
        });
        (elapsed, dst.data().to_vec())
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
                output_axes(),
                self.alpha,
                self.beta,
            )
            .unwrap();
        dst.data().to_vec()
    }

    fn manual_into(
        &self,
        dst: &mut TensorMap<f64, 4, 0>,
        canonical_dst: &mut TensorMap<f64, 4, 0>,
        lhs_canonical: &mut TensorMap<f64, 4, 0>,
        rhs_canonical: &mut TensorMap<f64, 0, 0>,
    ) {
        let rule = SU2FusionRule;
        tensorcontract_fusion_explicit_plan_into_canonical_dst(
            &rule,
            &self.plan,
            dst,
            canonical_dst,
            lhs_canonical,
            rhs_canonical,
            &self.lhs,
            &self.rhs,
            self.alpha,
            self.beta,
        )
        .unwrap();
    }

    fn materialize_canonical_dst(
        &self,
        canonical_dst: &mut TensorMap<f64, 4, 0>,
        lhs_canonical: &mut TensorMap<f64, 4, 0>,
        rhs_canonical: &mut TensorMap<f64, 0, 0>,
    ) {
        let mut dst = self.dst();
        self.manual_into(&mut dst, canonical_dst, lhs_canonical, rhs_canonical);
    }

    fn dst(&self) -> TensorMap<f64, 4, 0> {
        TensorMap::<f64, 4, 0>::from_vec_with_fusion_space(
            self.initial_dst.clone(),
            self.dst_space.clone(),
        )
        .unwrap()
    }

    fn canonical_dst(&self) -> TensorMap<f64, 4, 0> {
        TensorMap::<f64, 4, 0>::from_vec_with_fusion_space(
            vec![0.0; self.canonical_dst_space.required_len().unwrap()],
            self.canonical_dst_space.clone(),
        )
        .unwrap()
    }

    fn lhs_canonical(&self) -> TensorMap<f64, 4, 0> {
        TensorMap::<f64, 4, 0>::from_vec_with_fusion_space(
            vec![0.0; self.lhs_canonical_space.required_len().unwrap()],
            self.lhs_canonical_space.clone(),
        )
        .unwrap()
    }

    fn rhs_canonical(&self) -> TensorMap<f64, 0, 0> {
        TensorMap::<f64, 0, 0>::from_vec_with_fusion_space(
            vec![0.0; self.rhs_canonical_space.required_len().unwrap()],
            self.rhs_canonical_space.clone(),
        )
        .unwrap()
    }
}

struct ProductComplexFixture {
    rule: FpU1Su2Rule,
    lhs: TensorMap<Complex64, 2, 1>,
    rhs: TensorMap<Complex64, 0, 0>,
    dst_space: FusionTensorMapSpace<2, 1>,
    lhs_canonical_space: FusionTensorMapSpace<3, 0>,
    rhs_canonical_space: FusionTensorMapSpace<0, 0>,
    canonical_dst_space: FusionTensorMapSpace<3, 0>,
    plan: TensorContractFusionExplicitPlan,
    initial_dst: Vec<Complex64>,
    alpha: Complex64,
    beta: Complex64,
}

impl ProductComplexFixture {
    fn new() -> Self {
        let (rule, src_space, dst_space) = fz2_u1_su2_tree_pair_fixture();
        let rhs_hom = FusionTreeHomSpace::from_sector_ids([], []);
        let scalar_key = BlockKey::from(rhs_hom.fusion_tree_keys(&rule)[0].clone());
        let rhs_space = FusionTensorMapSpace::new(
            TensorMapSpace::<0, 0>::from_dims([], []).unwrap(),
            rhs_hom,
            BlockStructure::packed_column_major_with_keys(0, [(scalar_key, vec![])]).unwrap(),
        )
        .unwrap();
        let lhs_canonical_hom = src_space
            .homspace()
            .permute(&rule, &[0, 1, 2], &[])
            .unwrap();
        let lhs_canonical_space = FusionTensorMapSpace::from_degeneracy_shapes(
            TensorMapSpace::<3, 0>::from_dims([1, 1, 1], []).unwrap(),
            lhs_canonical_hom,
            &rule,
            [vec![1, 1, 1], vec![1, 1, 1]],
        )
        .unwrap();
        let rhs_canonical_space = rhs_space.clone();
        let canonical_dst_space = lhs_canonical_space.clone();
        let lhs = TensorMap::<Complex64, 2, 1>::from_vec_with_fusion_space(
            vec![Complex64::new(1.0, 2.0), Complex64::new(3.0, -1.0)],
            src_space,
        )
        .unwrap();
        let rhs = TensorMap::<Complex64, 0, 0>::from_vec_with_fusion_space(
            vec![Complex64::new(2.0, 0.5)],
            rhs_space,
        )
        .unwrap();
        let plan = tensorcontract_fusion_explicit_plan(
            &rule,
            &dst_space,
            lhs.fusion_space().unwrap(),
            rhs.fusion_space().unwrap(),
            product_axes(),
        )
        .unwrap();
        Self {
            rule,
            lhs,
            rhs,
            dst_space,
            lhs_canonical_space,
            rhs_canonical_space,
            canonical_dst_space,
            plan,
            initial_dst: vec![Complex64::new(5.0, 1.0), Complex64::new(-2.0, 4.0)],
            alpha: Complex64::new(2.0, 0.0),
            beta: Complex64::new(3.0, 0.0),
        }
    }

    fn manual_once(&self) -> Vec<Complex64> {
        let mut dst = self.dst();
        let mut canonical_dst = self.canonical_dst();
        let mut lhs_canonical = self.lhs_canonical();
        let mut rhs_canonical = self.rhs_canonical();
        self.manual_into(
            &mut dst,
            &mut canonical_dst,
            &mut lhs_canonical,
            &mut rhs_canonical,
        );
        dst.data().to_vec()
    }

    fn time_manual(&self, iterations: usize) -> (Duration, Vec<Complex64>) {
        let mut dst = self.dst();
        let mut canonical_dst = self.canonical_dst();
        let mut lhs_canonical = self.lhs_canonical();
        let mut rhs_canonical = self.rhs_canonical();
        let elapsed = time_loop(iterations, || {
            dst.data_mut().copy_from_slice(&self.initial_dst);
            self.manual_into(
                &mut dst,
                &mut canonical_dst,
                &mut lhs_canonical,
                &mut rhs_canonical,
            );
            black_box(checksum_complex(dst.data()));
        });
        (elapsed, dst.data().to_vec())
    }

    fn time_oneshot(&self, iterations: usize) -> (Duration, Vec<Complex64>) {
        let mut dst = self.dst();
        let elapsed = time_loop(iterations, || {
            dst.data_mut().copy_from_slice(&self.initial_dst);
            tensorcontract_fusion_into(
                &self.rule,
                &mut dst,
                &self.lhs,
                &self.rhs,
                product_axes(),
                self.alpha,
                self.beta,
            )
            .unwrap();
            black_box(checksum_complex(dst.data()));
        });
        (elapsed, dst.data().to_vec())
    }

    fn time_context_cold(&self, iterations: usize) -> (Duration, Vec<Complex64>) {
        let mut dst = self.dst();
        let elapsed = time_loop(iterations, || {
            let mut context = TensorContractFusionExecutionContext::<
                Complex64,
                <FpU1Su2Rule as TreeTransformRuleCacheKey>::Key,
            >::default();
            dst.data_mut().copy_from_slice(&self.initial_dst);
            context
                .tensorcontract_fusion_into(
                    &self.rule,
                    &mut dst,
                    &self.lhs,
                    &self.rhs,
                    product_axes(),
                    self.alpha,
                    self.beta,
                )
                .unwrap();
            black_box(checksum_complex(dst.data()));
        });
        (elapsed, dst.data().to_vec())
    }

    fn time_context_warm(
        &self,
        context: &mut TensorContractFusionExecutionContext<
            Complex64,
            <FpU1Su2Rule as TreeTransformRuleCacheKey>::Key,
        >,
        iterations: usize,
    ) -> (Duration, Vec<Complex64>) {
        let mut dst = self.dst();
        let elapsed = time_loop(iterations, || {
            dst.data_mut().copy_from_slice(&self.initial_dst);
            context
                .tensorcontract_fusion_into(
                    &self.rule,
                    &mut dst,
                    &self.lhs,
                    &self.rhs,
                    product_axes(),
                    self.alpha,
                    self.beta,
                )
                .unwrap();
            black_box(checksum_complex(dst.data()));
        });
        (elapsed, dst.data().to_vec())
    }

    fn context_warm_once(
        &self,
        context: &mut TensorContractFusionExecutionContext<
            Complex64,
            <FpU1Su2Rule as TreeTransformRuleCacheKey>::Key,
        >,
    ) -> Vec<Complex64> {
        let mut dst = self.dst();
        context
            .tensorcontract_fusion_into(
                &self.rule,
                &mut dst,
                &self.lhs,
                &self.rhs,
                product_axes(),
                self.alpha,
                self.beta,
            )
            .unwrap();
        dst.data().to_vec()
    }

    fn manual_into(
        &self,
        dst: &mut TensorMap<Complex64, 2, 1>,
        canonical_dst: &mut TensorMap<Complex64, 3, 0>,
        lhs_canonical: &mut TensorMap<Complex64, 3, 0>,
        rhs_canonical: &mut TensorMap<Complex64, 0, 0>,
    ) {
        tensorcontract_fusion_explicit_plan_into_canonical_dst(
            &self.rule,
            &self.plan,
            dst,
            canonical_dst,
            lhs_canonical,
            rhs_canonical,
            &self.lhs,
            &self.rhs,
            self.alpha,
            self.beta,
        )
        .unwrap();
    }

    fn dst(&self) -> TensorMap<Complex64, 2, 1> {
        TensorMap::<Complex64, 2, 1>::from_vec_with_fusion_space(
            self.initial_dst.clone(),
            self.dst_space.clone(),
        )
        .unwrap()
    }

    fn canonical_dst(&self) -> TensorMap<Complex64, 3, 0> {
        TensorMap::<Complex64, 3, 0>::from_vec_with_fusion_space(
            vec![Complex64::new(0.0, 0.0); self.canonical_dst_space.required_len().unwrap()],
            self.canonical_dst_space.clone(),
        )
        .unwrap()
    }

    fn lhs_canonical(&self) -> TensorMap<Complex64, 3, 0> {
        TensorMap::<Complex64, 3, 0>::from_vec_with_fusion_space(
            vec![Complex64::new(0.0, 0.0); self.lhs_canonical_space.required_len().unwrap()],
            self.lhs_canonical_space.clone(),
        )
        .unwrap()
    }

    fn rhs_canonical(&self) -> TensorMap<Complex64, 0, 0> {
        TensorMap::<Complex64, 0, 0>::from_vec_with_fusion_space(
            vec![Complex64::new(0.0, 0.0); self.rhs_canonical_space.required_len().unwrap()],
            self.rhs_canonical_space.clone(),
        )
        .unwrap()
    }
}

fn fusion_space_from_hom<const NOUT: usize, const NIN: usize>(
    dense_space: TensorMapSpace<NOUT, NIN>,
    homspace: FusionTreeHomSpace,
    rule: &SU2FusionRule,
    dim: usize,
) -> FusionTensorMapSpace<NOUT, NIN> {
    let count = homspace.fusion_tree_keys(rule).len();
    FusionTensorMapSpace::from_degeneracy_shapes(
        dense_space,
        homspace,
        rule,
        (0..count).map(|_| vec![dim; NOUT + NIN]),
    )
    .unwrap()
}

fn fz2_u1_su2_tree_pair_fixture() -> (
    FpU1Su2Rule,
    FusionTensorMapSpace<2, 1>,
    FusionTensorMapSpace<2, 1>,
) {
    let left_rule = FpU1Rule::default();
    let rule = FpU1Su2Rule::default();
    let even = SectorId::new(0);
    let odd = SectorId::new(1);
    let left_sector =
        |parity, charge| left_rule.encode_sector(parity, U1Irrep::new(charge).sector_id());
    let sector = |parity, charge, twice_spin| {
        rule.encode_sector(
            left_sector(parity, charge),
            SU2Irrep::from_twice_spin(twice_spin).sector_id(),
        )
    };

    let a = sector(odd, 1, 1);
    let b = sector(odd, -1, 1);
    let c0 = sector(even, 0, 0);
    let c1 = sector(even, 0, 2);
    let src_hom = FusionTreeHomSpace::new(
        FusionProductSpace::new([SectorLeg::new([a], false), SectorLeg::new([b], false)]),
        FusionProductSpace::new([SectorLeg::new([c0, c1], false)]),
    );
    let dst_hom = FusionTreeHomSpace::new(
        FusionProductSpace::new([SectorLeg::new([b], false), SectorLeg::new([a], false)]),
        FusionProductSpace::new([SectorLeg::new([c0, c1], false)]),
    );
    let src_space = FusionTensorMapSpace::from_degeneracy_shapes(
        TensorMapSpace::<2, 1>::from_dims([1, 1], [1]).unwrap(),
        src_hom,
        &rule,
        [vec![1, 1, 1], vec![1, 1, 1]],
    )
    .unwrap();
    let dst_space = FusionTensorMapSpace::from_degeneracy_shapes(
        TensorMapSpace::<2, 1>::from_dims([1, 1], [1]).unwrap(),
        dst_hom,
        &rule,
        [vec![1, 1, 1], vec![1, 1, 1]],
    )
    .unwrap();
    (rule, src_space, dst_space)
}

fn axes() -> TensorContractAxisSpec<'static> {
    TensorContractAxisSpec::canonical(&LHS_CONTRACTING_AXES, &RHS_CONTRACTING_AXES)
}

fn canonical_axes() -> TensorContractAxisSpec<'static> {
    TensorContractAxisSpec::canonical(
        &CANONICAL_LHS_CONTRACTING_AXES,
        &CANONICAL_RHS_CONTRACTING_AXES,
    )
}

fn output_axes() -> TensorContractAxisSpec<'static> {
    TensorContractAxisSpec::new(&[], &[], AxisPermutation::from_axes(&[0, 2, 1, 3]))
}

fn product_axes() -> TensorContractAxisSpec<'static> {
    TensorContractAxisSpec::new(&[], &[], AxisPermutation::from_axes(&[1, 0, 2]))
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

fn checksum_complex(data: &[Complex64]) -> f64 {
    data.iter().map(|value| value.re + 0.5 * value.im).sum()
}

fn assert_close(actual: &[f64], expected: &[f64]) {
    for (&actual, &expected) in actual.iter().zip(expected) {
        assert!(
            (actual - expected).abs() < 1.0e-10,
            "actual {actual} expected {expected}"
        );
    }
}

fn assert_close_complex(actual: &[Complex64], expected: &[Complex64]) {
    for (&actual, &expected) in actual.iter().zip(expected) {
        assert!(
            (actual - expected).norm() < 1.0e-10,
            "actual {actual} expected {expected}"
        );
    }
}
