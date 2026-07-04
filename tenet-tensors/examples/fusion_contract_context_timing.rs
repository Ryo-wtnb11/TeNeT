use std::hint::black_box;
use std::time::{Duration, Instant};

use num_complex::Complex64;
use tenet_core::{
    BlockKey, BlockStructure, FermionParityFusionRule, FusionProductSpace, FusionTensorMapSpace,
    FusionTreeHomSpace, ProductFusionRule, SU2FusionRule, SU2Irrep, SectorId, SectorLeg, TensorMap,
    TensorMapSpace, U1FusionRule, U1Irrep,
};
use tenet_tensors::{
    prepare_tensorcontract_fusion_plan, tensorcontract_fusion_into,
    tensorcontract_fusion_prepared_into, tensorcontract_fusion_prepared_into_core_dst,
    tree_transform_into_with_context, FusionContractPlan, HostTensorOperations,
    HostTreeFusionExecutionContext, OutputAxisOrder, TensorContractFusionExecutionContext,
    TensorContractFusionProfile, TensorContractSpec, TreeTransformBuiltinRuleCacheKey,
    TreeTransformExecutionContext, TreeTransformRuleCacheKey,
};

static LHS_CONTRACTING_AXES: [usize; 3] = [0, 1, 2];
static RHS_CONTRACTING_AXES: [usize; 3] = [1, 2, 3];
static CORE_LHS_CONTRACTING_AXES: [usize; 3] = [1, 2, 3];
static CORE_RHS_CONTRACTING_AXES: [usize; 3] = [0, 1, 2];

const MANUAL_ITERS: usize = 20_000;
const ONESHOT_ITERS: usize = 5_000;
const COLD_CONTEXT_ITERS: usize = 2_000;
const WARM_CONTEXT_ITERS: usize = 100_000;
const PROFILE_ITERS: usize = 20_000;

type FpU1Rule = ProductFusionRule<FermionParityFusionRule, U1FusionRule>;
type FpU1Su2Rule = ProductFusionRule<FpU1Rule, SU2FusionRule>;

fn main() {
    bench_su2_non_core_form_source();
    bench_su2_output_scratch();
    bench_product_complex();
}

fn bench_su2_non_core_form_source() {
    let fixture = Su2NoncoreFixture::new();
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
    let (source_transform_host, source_host_checksum) =
        fixture.time_source_transforms_host_tree(WARM_CONTEXT_ITERS);
    assert!((source_host_checksum - source_checksum).abs() < 1.0e-12);
    let (warm_host_tree, warm_host_tree_data) =
        fixture.time_context_warm_host_tree(WARM_CONTEXT_ITERS);
    assert_close(&warm_host_tree_data, &expected);
    let (core_contract, contract_data) = fixture.time_core_contract(WARM_CONTEXT_ITERS);
    assert_close(&contract_data, &expected);
    let (profile, profiled_data) = fixture.time_context_warm_profiled(PROFILE_ITERS);
    assert_close(&profiled_data, &expected);

    println!("fusion contraction timing (release)");
    println!("fixture,su2_non_core_form_source_degeneracy");
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
        "source_transform_host_tree_warm_ns,{:.3}",
        nanos_per(source_transform_host, WARM_CONTEXT_ITERS)
    );
    println!(
        "context_warm_host_tree_ns,{:.3}",
        nanos_per(warm_host_tree, WARM_CONTEXT_ITERS)
    );
    println!(
        "core_contract_warm_ns,{:.3}",
        nanos_per(core_contract, WARM_CONTEXT_ITERS)
    );
    print_profile_breakdown(&profile, PROFILE_ITERS);
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
        "dynamic_fusion_space_cache,hits={},fast_hits={},misses={},len={}",
        context.dynamic_fusion_space_cache_hits(),
        context.dynamic_fusion_space_cache_fast_hits(),
        context.dynamic_fusion_space_cache_misses(),
        context.dynamic_fusion_space_cache_len()
    );
    println!(
        "fusion_block_contract_cache,hits={},fast_hits={},misses={},len={}",
        context.contraction_resolution_cache_hits(),
        context.contraction_resolution_cache_fast_hits(),
        context.contraction_resolution_cache_misses(),
        context.contraction_resolution_cache_len()
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
    let (profile, profiled_data) = fixture.time_context_warm_profiled(PROFILE_ITERS);
    assert_close(&profiled_data, &expected);

    println!();
    println!("fixture,su2_output_transform_core_dst_scratch");
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
    print_profile_breakdown(&profile, PROFILE_ITERS);
    println!("result_checksum,{:.12}", checksum(&warm_data));
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
        "dynamic_fusion_space_cache,hits={},fast_hits={},misses={},len={}",
        context.dynamic_fusion_space_cache_hits(),
        context.dynamic_fusion_space_cache_fast_hits(),
        context.dynamic_fusion_space_cache_misses(),
        context.dynamic_fusion_space_cache_len()
    );
    println!(
        "fusion_block_contract_cache,hits={},fast_hits={},misses={},len={}",
        context.contraction_resolution_cache_hits(),
        context.contraction_resolution_cache_fast_hits(),
        context.contraction_resolution_cache_misses(),
        context.contraction_resolution_cache_len()
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
    let (profile, profiled_data) = fixture.time_context_warm_profiled(PROFILE_ITERS);
    assert_close_complex(&profiled_data, &expected);

    println!();
    println!("fixture,product_fz2_u1_su2_complex");
    println!("manual_explicit_ns,{:.3}", nanos_per(manual, MANUAL_ITERS));
    println!("one_shot_ns,{:.3}", nanos_per(oneshot, ONESHOT_ITERS));
    println!(
        "context_cold_ns,{:.3}",
        nanos_per(context_cold, COLD_CONTEXT_ITERS)
    );
    println!("context_warm_ns,{:.3}", nanos_per(warm, WARM_CONTEXT_ITERS));
    print_profile_breakdown(&profile, PROFILE_ITERS);
    println!("result_checksum,{:.12}", checksum_complex(&warm_data));
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
        "dynamic_fusion_space_cache,hits={},fast_hits={},misses={},len={}",
        context.dynamic_fusion_space_cache_hits(),
        context.dynamic_fusion_space_cache_fast_hits(),
        context.dynamic_fusion_space_cache_misses(),
        context.dynamic_fusion_space_cache_len()
    );
    println!(
        "contract_structure_cache,hits={},misses={},len={}",
        context.contract_cache().stats().structure_hits(),
        context.contract_cache().stats().structure_misses(),
        context.contract_cache().structure_len()
    );
    println!(
        "fusion_block_contract_cache,hits={},fast_hits={},misses={},len={}",
        context.contraction_resolution_cache_hits(),
        context.contraction_resolution_cache_fast_hits(),
        context.contraction_resolution_cache_misses(),
        context.contraction_resolution_cache_len()
    );
}

struct Su2NoncoreFixture {
    lhs: TensorMap<f64, 3, 1>,
    rhs: TensorMap<f64, 1, 3>,
    dst_space: FusionTensorMapSpace<1, 1>,
    lhs_core_space: FusionTensorMapSpace<1, 3>,
    rhs_core_space: FusionTensorMapSpace<3, 1>,
    plan: FusionContractPlan,
    initial_dst: Vec<f64>,
    alpha: f64,
    beta: f64,
}

impl Su2NoncoreFixture {
    fn new() -> Self {
        let rule = SU2FusionRule;
        let lhs_hom = FusionTreeHomSpace::from_sector_ids([(1, 2), (1, 2), (1, 2)], [(1, 2)]);
        let rhs_hom = FusionTreeHomSpace::from_sector_ids([(1, 2)], [(1, 2), (1, 2), (1, 2)]);
        let lhs_core_hom = lhs_hom
            .permute(&rule, &[3], &[0, 1, 2])
            .expect("valid lhs core tree-pair transform");
        let rhs_core_hom = rhs_hom
            .permute(&rule, &[1, 2, 3], &[0])
            .expect("valid rhs core tree-pair transform");
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
        let lhs_core_space = FusionTensorMapSpace::from_degeneracy_shapes(
            TensorMapSpace::<1, 3>::from_dims([2], [2, 2, 2]).unwrap(),
            lhs_core_hom,
            &rule,
            [vec![2, 2, 2, 2], vec![2, 2, 2, 2]],
        )
        .unwrap();
        let rhs_core_space = FusionTensorMapSpace::from_degeneracy_shapes(
            TensorMapSpace::<3, 1>::from_dims([2, 2, 2], [2]).unwrap(),
            rhs_core_hom,
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
        let plan = prepare_tensorcontract_fusion_plan(
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
            lhs_core_space,
            rhs_core_space,
            plan,
            initial_dst: vec![2.0, -1.0, 4.0, -3.0],
            alpha: -1.5,
            beta: 0.25,
        }
    }

    fn manual_once(&self) -> Vec<f64> {
        let mut dst = self.dst();
        let mut lhs_core = self.lhs_core();
        let mut rhs_core = self.rhs_core();
        self.manual_into(&mut dst, &mut lhs_core, &mut rhs_core);
        dst.data().to_vec()
    }

    fn time_manual(&self, iterations: usize) -> (Duration, Vec<f64>) {
        let mut dst = self.dst();
        let mut lhs_core = self.lhs_core();
        let mut rhs_core = self.rhs_core();
        let elapsed = time_loop(iterations, || {
            dst.data_mut().copy_from_slice(&self.initial_dst);
            self.manual_into(&mut dst, &mut lhs_core, &mut rhs_core);
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

    fn time_context_warm_profiled(
        &self,
        iterations: usize,
    ) -> (TensorContractFusionProfile, Vec<f64>) {
        let rule = SU2FusionRule;
        let mut dst = self.dst();
        let mut context =
            TensorContractFusionExecutionContext::<f64, TreeTransformBuiltinRuleCacheKey>::default(
            );
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
        let mut total_profile = TensorContractFusionProfile::default();
        for _ in 0..iterations {
            dst.data_mut().copy_from_slice(&self.initial_dst);
            let mut profile = TensorContractFusionProfile::default();
            context
                .tensorcontract_fusion_into_profiled(
                    &rule,
                    &mut dst,
                    &self.lhs,
                    &self.rhs,
                    axes(),
                    self.alpha,
                    self.beta,
                    &mut profile,
                )
                .unwrap();
            total_profile.accumulate(&profile);
            black_box(checksum(dst.data()));
        }
        (total_profile, dst.data().to_vec())
    }

    fn time_context_warm_host_tree(&self, iterations: usize) -> (Duration, Vec<f64>) {
        let rule = SU2FusionRule;
        let mut dst = self.dst();
        let mut context =
            HostTreeFusionExecutionContext::<f64, TreeTransformBuiltinRuleCacheKey>::default();
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
        let mut lhs_core = self.lhs_core();
        let mut rhs_core = self.rhs_core();
        let mut context =
            TreeTransformExecutionContext::<f64, TreeTransformBuiltinRuleCacheKey>::default();
        tree_transform_into_with_context(
            &mut context,
            &rule,
            self.plan.lhs_transform().clone(),
            &mut lhs_core,
            &self.lhs,
            1.0,
            0.0,
        )
        .unwrap();
        tree_transform_into_with_context(
            &mut context,
            &rule,
            self.plan.rhs_transform().clone(),
            &mut rhs_core,
            &self.rhs,
            1.0,
            0.0,
        )
        .unwrap();
        let elapsed = time_loop(iterations, || {
            tree_transform_into_with_context(
                &mut context,
                &rule,
                self.plan.lhs_transform().clone(),
                &mut lhs_core,
                &self.lhs,
                1.0,
                0.0,
            )
            .unwrap();
            tree_transform_into_with_context(
                &mut context,
                &rule,
                self.plan.rhs_transform().clone(),
                &mut rhs_core,
                &self.rhs,
                1.0,
                0.0,
            )
            .unwrap();
            black_box(checksum(lhs_core.data()) + checksum(rhs_core.data()));
        });
        (
            elapsed,
            checksum(lhs_core.data()) + checksum(rhs_core.data()),
        )
    }

    fn time_source_transforms_host_tree(&self, iterations: usize) -> (Duration, f64) {
        let rule = SU2FusionRule;
        let mut lhs_core = self.lhs_core();
        let mut rhs_core = self.rhs_core();
        let mut context = TreeTransformExecutionContext::<
            f64,
            TreeTransformBuiltinRuleCacheKey,
            f64,
            HostTensorOperations,
        >::default();
        tree_transform_into_with_context(
            &mut context,
            &rule,
            self.plan.lhs_transform().clone(),
            &mut lhs_core,
            &self.lhs,
            1.0,
            0.0,
        )
        .unwrap();
        tree_transform_into_with_context(
            &mut context,
            &rule,
            self.plan.rhs_transform().clone(),
            &mut rhs_core,
            &self.rhs,
            1.0,
            0.0,
        )
        .unwrap();
        let elapsed = time_loop(iterations, || {
            tree_transform_into_with_context(
                &mut context,
                &rule,
                self.plan.lhs_transform().clone(),
                &mut lhs_core,
                &self.lhs,
                1.0,
                0.0,
            )
            .unwrap();
            tree_transform_into_with_context(
                &mut context,
                &rule,
                self.plan.rhs_transform().clone(),
                &mut rhs_core,
                &self.rhs,
                1.0,
                0.0,
            )
            .unwrap();
            black_box(checksum(lhs_core.data()) + checksum(rhs_core.data()));
        });
        (
            elapsed,
            checksum(lhs_core.data()) + checksum(rhs_core.data()),
        )
    }

    fn time_core_contract(&self, iterations: usize) -> (Duration, Vec<f64>) {
        let rule = SU2FusionRule;
        let mut lhs_core = self.lhs_core();
        let mut rhs_core = self.rhs_core();
        let mut dst = self.dst();
        self.transform_sources_into(&mut lhs_core, &mut rhs_core);
        let mut context =
            TensorContractFusionExecutionContext::<f64, TreeTransformBuiltinRuleCacheKey>::default(
            );
        context
            .tensorcontract_fusion_into(
                &rule,
                &mut dst,
                &lhs_core,
                &rhs_core,
                core_axes(),
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
                    &lhs_core,
                    &rhs_core,
                    core_axes(),
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
        lhs_core: &mut TensorMap<f64, 1, 3>,
        rhs_core: &mut TensorMap<f64, 3, 1>,
    ) {
        let rule = SU2FusionRule;
        tensorcontract_fusion_prepared_into(
            &rule, &self.plan, dst, lhs_core, rhs_core, &self.lhs, &self.rhs, self.alpha, self.beta,
        )
        .unwrap();
    }

    fn transform_sources_into(
        &self,
        lhs_core: &mut TensorMap<f64, 1, 3>,
        rhs_core: &mut TensorMap<f64, 3, 1>,
    ) {
        let rule = SU2FusionRule;
        let mut context =
            TreeTransformExecutionContext::<f64, TreeTransformBuiltinRuleCacheKey>::default();
        tree_transform_into_with_context(
            &mut context,
            &rule,
            self.plan.lhs_transform().clone(),
            lhs_core,
            &self.lhs,
            1.0,
            0.0,
        )
        .unwrap();
        tree_transform_into_with_context(
            &mut context,
            &rule,
            self.plan.rhs_transform().clone(),
            rhs_core,
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

    fn lhs_core(&self) -> TensorMap<f64, 1, 3> {
        TensorMap::<f64, 1, 3>::from_vec_with_fusion_space(
            vec![0.0; self.lhs_core_space.required_len().unwrap()],
            self.lhs_core_space.clone(),
        )
        .unwrap()
    }

    fn rhs_core(&self) -> TensorMap<f64, 3, 1> {
        TensorMap::<f64, 3, 1>::from_vec_with_fusion_space(
            vec![0.0; self.rhs_core_space.required_len().unwrap()],
            self.rhs_core_space.clone(),
        )
        .unwrap()
    }
}

struct Su2OutputScratchFixture {
    lhs: TensorMap<f64, 2, 2>,
    rhs: TensorMap<f64, 0, 0>,
    dst_space: FusionTensorMapSpace<4, 0>,
    core_dst_space: FusionTensorMapSpace<4, 0>,
    lhs_core_space: FusionTensorMapSpace<4, 0>,
    rhs_core_space: FusionTensorMapSpace<0, 0>,
    plan: FusionContractPlan,
    initial_dst: Vec<f64>,
    alpha: f64,
    beta: f64,
}

impl Su2OutputScratchFixture {
    fn new() -> Self {
        let rule = SU2FusionRule;
        let lhs_hom = FusionTreeHomSpace::from_sector_ids([(1, 1), (1, 1)], [(1, 1), (1, 1)]);
        let lhs_core_hom = lhs_hom
            .permute(&rule, &[0, 1, 2, 3], &[])
            .expect("valid all-open core transform");
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
        let lhs_core_space = fusion_space_from_hom::<4, 0>(
            TensorMapSpace::<4, 0>::from_dims([1, 1, 1, 1], []).unwrap(),
            lhs_core_hom.clone(),
            &rule,
            1,
        );
        let core_dst_space = lhs_core_space.clone();
        let dst_space = fusion_space_from_hom::<4, 0>(
            TensorMapSpace::<4, 0>::from_dims([1, 1, 1, 1], []).unwrap(),
            dst_hom,
            &rule,
            1,
        );
        let rhs_core_space = FusionTensorMapSpace::from_degeneracy_shapes(
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
        let rhs =
            TensorMap::<f64, 0, 0>::from_vec_with_fusion_space(vec![2.0], rhs_core_space.clone())
                .unwrap();
        let initial_dst = (0..dst_len)
            .map(|index| 0.5 + index as f64)
            .collect::<Vec<_>>();
        let plan = prepare_tensorcontract_fusion_plan(
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
            core_dst_space,
            lhs_core_space,
            rhs_core_space,
            plan,
            initial_dst,
            alpha: -0.75,
            beta: 0.5,
        }
    }

    fn manual_once(&self) -> Vec<f64> {
        let mut dst = self.dst();
        let mut core_dst = self.core_dst();
        let mut lhs_core = self.lhs_core();
        let mut rhs_core = self.rhs_core();
        self.manual_into(&mut dst, &mut core_dst, &mut lhs_core, &mut rhs_core);
        dst.data().to_vec()
    }

    fn time_manual(&self, iterations: usize) -> (Duration, Vec<f64>) {
        let mut dst = self.dst();
        let mut core_dst = self.core_dst();
        let mut lhs_core = self.lhs_core();
        let mut rhs_core = self.rhs_core();
        let elapsed = time_loop(iterations, || {
            dst.data_mut().copy_from_slice(&self.initial_dst);
            self.manual_into(&mut dst, &mut core_dst, &mut lhs_core, &mut rhs_core);
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

    fn time_context_warm_profiled(
        &self,
        iterations: usize,
    ) -> (TensorContractFusionProfile, Vec<f64>) {
        let rule = SU2FusionRule;
        let mut dst = self.dst();
        let mut context =
            TensorContractFusionExecutionContext::<f64, TreeTransformBuiltinRuleCacheKey>::default(
            );
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
        let mut total_profile = TensorContractFusionProfile::default();
        for _ in 0..iterations {
            dst.data_mut().copy_from_slice(&self.initial_dst);
            let mut profile = TensorContractFusionProfile::default();
            context
                .tensorcontract_fusion_into_profiled(
                    &rule,
                    &mut dst,
                    &self.lhs,
                    &self.rhs,
                    output_axes(),
                    self.alpha,
                    self.beta,
                    &mut profile,
                )
                .unwrap();
            total_profile.accumulate(&profile);
            black_box(checksum(dst.data()));
        }
        (total_profile, dst.data().to_vec())
    }

    fn time_output_transform(&self, iterations: usize) -> (Duration, Vec<f64>) {
        let mut dst = self.dst();
        let mut core_dst = self.core_dst();
        let mut lhs_core = self.lhs_core();
        let mut rhs_core = self.rhs_core();
        self.materialize_core_dst(&mut core_dst, &mut lhs_core, &mut rhs_core);
        let rule = SU2FusionRule;
        let mut context =
            TreeTransformExecutionContext::<f64, TreeTransformBuiltinRuleCacheKey>::default();
        tree_transform_into_with_context(
            &mut context,
            &rule,
            self.plan.output_transform().clone(),
            &mut dst,
            &core_dst,
            1.0,
            self.beta,
        )
        .unwrap();
        let elapsed = time_loop(iterations, || {
            dst.data_mut().copy_from_slice(&self.initial_dst);
            tree_transform_into_with_context(
                &mut context,
                &rule,
                self.plan.output_transform().clone(),
                &mut dst,
                &core_dst,
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
        core_dst: &mut TensorMap<f64, 4, 0>,
        lhs_core: &mut TensorMap<f64, 4, 0>,
        rhs_core: &mut TensorMap<f64, 0, 0>,
    ) {
        let rule = SU2FusionRule;
        tensorcontract_fusion_prepared_into_core_dst(
            &rule, &self.plan, dst, core_dst, lhs_core, rhs_core, &self.lhs, &self.rhs, self.alpha,
            self.beta,
        )
        .unwrap();
    }

    fn materialize_core_dst(
        &self,
        core_dst: &mut TensorMap<f64, 4, 0>,
        lhs_core: &mut TensorMap<f64, 4, 0>,
        rhs_core: &mut TensorMap<f64, 0, 0>,
    ) {
        let mut dst = self.dst();
        self.manual_into(&mut dst, core_dst, lhs_core, rhs_core);
    }

    fn dst(&self) -> TensorMap<f64, 4, 0> {
        TensorMap::<f64, 4, 0>::from_vec_with_fusion_space(
            self.initial_dst.clone(),
            self.dst_space.clone(),
        )
        .unwrap()
    }

    fn core_dst(&self) -> TensorMap<f64, 4, 0> {
        TensorMap::<f64, 4, 0>::from_vec_with_fusion_space(
            vec![0.0; self.core_dst_space.required_len().unwrap()],
            self.core_dst_space.clone(),
        )
        .unwrap()
    }

    fn lhs_core(&self) -> TensorMap<f64, 4, 0> {
        TensorMap::<f64, 4, 0>::from_vec_with_fusion_space(
            vec![0.0; self.lhs_core_space.required_len().unwrap()],
            self.lhs_core_space.clone(),
        )
        .unwrap()
    }

    fn rhs_core(&self) -> TensorMap<f64, 0, 0> {
        TensorMap::<f64, 0, 0>::from_vec_with_fusion_space(
            vec![0.0; self.rhs_core_space.required_len().unwrap()],
            self.rhs_core_space.clone(),
        )
        .unwrap()
    }
}

struct ProductComplexFixture {
    rule: FpU1Su2Rule,
    lhs: TensorMap<Complex64, 2, 1>,
    rhs: TensorMap<Complex64, 0, 0>,
    dst_space: FusionTensorMapSpace<2, 1>,
    lhs_core_space: FusionTensorMapSpace<3, 0>,
    rhs_core_space: FusionTensorMapSpace<0, 0>,
    core_dst_space: FusionTensorMapSpace<3, 0>,
    plan: FusionContractPlan,
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
            packed_fixture_structure(0, [(scalar_key, vec![])]).unwrap(),
        )
        .unwrap();
        let lhs_core_hom = src_space
            .homspace()
            .permute(&rule, &[0, 1, 2], &[])
            .unwrap();
        let lhs_core_space = FusionTensorMapSpace::from_degeneracy_shapes(
            TensorMapSpace::<3, 0>::from_dims([1, 1, 1], []).unwrap(),
            lhs_core_hom,
            &rule,
            [vec![1, 1, 1], vec![1, 1, 1]],
        )
        .unwrap();
        let rhs_core_space = rhs_space.clone();
        let core_dst_space = lhs_core_space.clone();
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
        let plan = prepare_tensorcontract_fusion_plan(
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
            lhs_core_space,
            rhs_core_space,
            core_dst_space,
            plan,
            initial_dst: vec![Complex64::new(5.0, 1.0), Complex64::new(-2.0, 4.0)],
            alpha: Complex64::new(2.0, 0.0),
            beta: Complex64::new(3.0, 0.0),
        }
    }

    fn manual_once(&self) -> Vec<Complex64> {
        let mut dst = self.dst();
        let mut core_dst = self.core_dst();
        let mut lhs_core = self.lhs_core();
        let mut rhs_core = self.rhs_core();
        self.manual_into(&mut dst, &mut core_dst, &mut lhs_core, &mut rhs_core);
        dst.data().to_vec()
    }

    fn time_manual(&self, iterations: usize) -> (Duration, Vec<Complex64>) {
        let mut dst = self.dst();
        let mut core_dst = self.core_dst();
        let mut lhs_core = self.lhs_core();
        let mut rhs_core = self.rhs_core();
        let elapsed = time_loop(iterations, || {
            dst.data_mut().copy_from_slice(&self.initial_dst);
            self.manual_into(&mut dst, &mut core_dst, &mut lhs_core, &mut rhs_core);
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

    fn time_context_warm_profiled(
        &self,
        iterations: usize,
    ) -> (TensorContractFusionProfile, Vec<Complex64>) {
        let mut dst = self.dst();
        let mut context = TensorContractFusionExecutionContext::<
            Complex64,
            <FpU1Su2Rule as TreeTransformRuleCacheKey>::Key,
        >::default();
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
        let mut total_profile = TensorContractFusionProfile::default();
        for _ in 0..iterations {
            dst.data_mut().copy_from_slice(&self.initial_dst);
            let mut profile = TensorContractFusionProfile::default();
            context
                .tensorcontract_fusion_into_profiled(
                    &self.rule,
                    &mut dst,
                    &self.lhs,
                    &self.rhs,
                    product_axes(),
                    self.alpha,
                    self.beta,
                    &mut profile,
                )
                .unwrap();
            total_profile.accumulate(&profile);
            black_box(checksum_complex(dst.data()));
        }
        (total_profile, dst.data().to_vec())
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
        core_dst: &mut TensorMap<Complex64, 3, 0>,
        lhs_core: &mut TensorMap<Complex64, 3, 0>,
        rhs_core: &mut TensorMap<Complex64, 0, 0>,
    ) {
        tensorcontract_fusion_prepared_into_core_dst(
            &self.rule, &self.plan, dst, core_dst, lhs_core, rhs_core, &self.lhs, &self.rhs,
            self.alpha, self.beta,
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

    fn core_dst(&self) -> TensorMap<Complex64, 3, 0> {
        TensorMap::<Complex64, 3, 0>::from_vec_with_fusion_space(
            vec![Complex64::new(0.0, 0.0); self.core_dst_space.required_len().unwrap()],
            self.core_dst_space.clone(),
        )
        .unwrap()
    }

    fn lhs_core(&self) -> TensorMap<Complex64, 3, 0> {
        TensorMap::<Complex64, 3, 0>::from_vec_with_fusion_space(
            vec![Complex64::new(0.0, 0.0); self.lhs_core_space.required_len().unwrap()],
            self.lhs_core_space.clone(),
        )
        .unwrap()
    }

    fn rhs_core(&self) -> TensorMap<Complex64, 0, 0> {
        TensorMap::<Complex64, 0, 0>::from_vec_with_fusion_space(
            vec![Complex64::new(0.0, 0.0); self.rhs_core_space.required_len().unwrap()],
            self.rhs_core_space.clone(),
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
        FusionProductSpace::new([
            SectorLeg::new([(a, 1)], false),
            SectorLeg::new([(b, 1)], false),
        ]),
        FusionProductSpace::new([SectorLeg::new([(c0, 1), (c1, 1)], false)]),
    );
    let dst_hom = FusionTreeHomSpace::new(
        FusionProductSpace::new([
            SectorLeg::new([(b, 1)], false),
            SectorLeg::new([(a, 1)], false),
        ]),
        FusionProductSpace::new([SectorLeg::new([(c0, 1), (c1, 1)], false)]),
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

fn axes() -> TensorContractSpec<'static> {
    TensorContractSpec::with_default_output_order(&LHS_CONTRACTING_AXES, &RHS_CONTRACTING_AXES)
}

fn core_axes() -> TensorContractSpec<'static> {
    TensorContractSpec::with_default_output_order(
        &CORE_LHS_CONTRACTING_AXES,
        &CORE_RHS_CONTRACTING_AXES,
    )
}

fn output_axes() -> TensorContractSpec<'static> {
    TensorContractSpec::new(&[], &[], OutputAxisOrder::from_axes(&[0, 2, 1, 3]))
}

fn product_axes() -> TensorContractSpec<'static> {
    TensorContractSpec::new(&[], &[], OutputAxisOrder::from_axes(&[1, 0, 2]))
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

fn print_profile_breakdown(profile: &TensorContractFusionProfile, iterations: usize) {
    println!("profile_route,{:?}", profile.route);
    println!(
        "profile_total_ns,{:.3}",
        nanos_per(profile.total, iterations)
    );
    println!(
        "profile_typed_space_setup_ns,{:.3}",
        nanos_per(profile.typed_space_setup, iterations)
    );
    println!(
        "profile_core_route_check_ns,{:.3}",
        nanos_per(profile.core_route_check, iterations)
    );
    println!(
        "profile_dense_block_specs_ns,{:.3}",
        nanos_per(profile.dense_block_specs, iterations)
    );
    println!(
        "profile_dense_structure_lookup_ns,{:.3}",
        nanos_per(profile.dense_structure_lookup, iterations)
    );
    println!(
        "profile_dense_contract_ns,{:.3}",
        nanos_per(profile.dense_contract, iterations)
    );
    println!(
        "profile_prepared_plan_ns,{:.3}",
        nanos_per(profile.prepared_plan, iterations)
    );
    println!(
        "profile_source_space_lookup_ns,{:.3}",
        nanos_per(profile.source_space_lookup, iterations)
    );
    println!(
        "profile_lhs_scratch_prepare_ns,{:.3}",
        nanos_per(profile.lhs_scratch_prepare, iterations)
    );
    println!(
        "profile_rhs_scratch_prepare_ns,{:.3}",
        nanos_per(profile.rhs_scratch_prepare, iterations)
    );
    println!(
        "profile_lhs_transform_ns,{:.3}",
        nanos_per(profile.lhs_transform, iterations)
    );
    println!(
        "profile_rhs_transform_ns,{:.3}",
        nanos_per(profile.rhs_transform, iterations)
    );
    println!(
        "profile_core_dst_space_lookup_ns,{:.3}",
        nanos_per(profile.core_dst_space_lookup, iterations)
    );
    println!(
        "profile_dst_scratch_prepare_ns,{:.3}",
        nanos_per(profile.dst_scratch_prepare, iterations)
    );
    println!(
        "profile_fusion_block_plan_lookup_ns,{:.3}",
        nanos_per(profile.fusion_block_plan_lookup, iterations)
    );
    println!(
        "profile_core_contract_total_ns,{:.3}",
        nanos_per(profile.core_contract_total, iterations)
    );
    println!(
        "profile_core_validate_ns,{:.3}",
        nanos_per(profile.core_validate, iterations)
    );
    println!(
        "profile_core_scale_ns,{:.3}",
        nanos_per(profile.core_scale, iterations)
    );
    println!(
        "profile_core_workspace_prepare_ns,{:.3}",
        nanos_per(profile.core_workspace_prepare, iterations)
    );
    println!(
        "profile_core_pack_lhs_ns,{:.3}",
        nanos_per(profile.core_pack_lhs, iterations)
    );
    println!(
        "profile_core_pack_rhs_ns,{:.3}",
        nanos_per(profile.core_pack_rhs, iterations)
    );
    println!(
        "profile_core_matmul_ns,{:.3}",
        nanos_per(profile.core_matmul, iterations)
    );
    println!(
        "profile_core_scatter_ns,{:.3}",
        nanos_per(profile.core_scatter, iterations)
    );
    println!(
        "profile_output_transform_ns,{:.3}",
        nanos_per(profile.output_transform, iterations)
    );
    println!(
        "profile_tree_replay_total_ns,{:.3}",
        nanos_per(profile.tree_replay.total, iterations)
    );
    println!(
        "profile_tree_cache_plus_replay_ns,{:.3}",
        nanos_per(
            profile.tree_replay.cache_lookup + profile.tree_replay.total,
            iterations
        )
    );
    println!(
        "profile_tree_cache_lookup_ns,{:.3}",
        nanos_per(profile.tree_replay.cache_lookup, iterations)
    );
    println!(
        "profile_tree_validate_ns,{:.3}",
        nanos_per(profile.tree_replay.validate, iterations)
    );
    println!(
        "profile_tree_single_total_ns,{:.3}",
        nanos_per(profile.tree_replay.single_total, iterations)
    );
    println!(
        "profile_tree_multi_workspace_prepare_ns,{:.3}",
        nanos_per(profile.tree_replay.multi_workspace_prepare, iterations)
    );
    println!(
        "profile_tree_multi_pack_ns,{:.3}",
        nanos_per(profile.tree_replay.multi_pack, iterations)
    );
    println!(
        "profile_tree_multi_coefficient_prepare_ns,{:.3}",
        nanos_per(profile.tree_replay.multi_coefficient_prepare, iterations)
    );
    println!(
        "profile_tree_multi_matmul_total_ns,{:.3}",
        nanos_per(profile.tree_replay.multi_matmul_total, iterations)
    );
    println!(
        "profile_tree_multi_dense_view_setup_ns,{:.3}",
        nanos_per(profile.tree_replay.multi_dense_view_setup, iterations)
    );
    println!(
        "profile_tree_multi_dense_matmul_call_ns,{:.3}",
        nanos_per(profile.tree_replay.multi_dense_matmul_call, iterations)
    );
    println!(
        "profile_tree_multi_scalar_recoupling_ns,{:.3}",
        nanos_per(profile.tree_replay.multi_scalar_recoupling, iterations)
    );
    println!(
        "profile_tree_multi_scatter_ns,{:.3}",
        nanos_per(profile.tree_replay.multi_scatter, iterations)
    );
    println!(
        "profile_tree_strided_view_setup_ns,{:.3}",
        nanos_per(profile.tree_replay.strided_view_setup, iterations)
    );
    println!(
        "profile_tree_strided_kernel_ns,{:.3}",
        nanos_per(profile.tree_replay.strided_kernel, iterations)
    );
    println!(
        "profile_tree_single_blocks,{}",
        profile.tree_replay.single_blocks
    );
    println!(
        "profile_tree_multi_blocks,{}",
        profile.tree_replay.multi_blocks
    );
    println!(
        "profile_tree_packed_columns,{}",
        profile.tree_replay.packed_columns
    );
    println!(
        "profile_tree_scattered_columns,{}",
        profile.tree_replay.scattered_columns
    );
    println!(
        "profile_lhs_transform_calls,{}",
        profile.lhs_transform_calls
    );
    println!(
        "profile_rhs_transform_calls,{}",
        profile.rhs_transform_calls
    );
    println!(
        "profile_output_transform_calls,{}",
        profile.output_transform_calls
    );
    println!(
        "profile_core_contract_groups,{}",
        profile.core_contract_groups
    );
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

/// Fixture layout: subblocks packed contiguously in key order (not a product
/// layout; exercises the arbitrary-strided-view contract).
fn packed_fixture_structure<I, K>(
    rank: usize,
    blocks: I,
) -> Result<BlockStructure, tenet_core::CoreError>
where
    I: IntoIterator<Item = (K, Vec<usize>)>,
    K: Into<tenet_core::BlockKey>,
{
    let mut keys = Vec::new();
    let mut shapes = Vec::new();
    for (key, shape) in blocks {
        keys.push(key.into());
        shapes.push(shape);
    }
    BlockStructure::from_parts(
        tenet_core::SectorStructure::from_keys(rank, keys)?,
        tenet_core::DegeneracyStructure::packed_column_major(rank, shapes)?,
    )
}
