use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;

use tenet_core::{
    FermionParityFusionRule, FusionProductSpace, FusionTensorMapSpace, FusionTreeHomSpace,
    MultiplicityFreeRigidSymbols, ProductFusionRule, ProductSectorCodec, SU2Irrep, SectorId,
    SectorLeg, TensorKitProductCodec, TensorMap, TensorMapSpace, U1FusionRule, U1Irrep,
};
use tenet_tensors::{
    OutputAxisOrder, TensorContractFusionExecutionContext, TensorContractSpec,
    TreeTransformRuleCacheKey,
};

struct CountingAllocator;

thread_local! {
    static COUNTING: Cell<bool> = const { Cell::new(false) };
    static ALLOCATIONS: Cell<usize> = const { Cell::new(0) };
    static REALLOCATIONS: Cell<usize> = const { Cell::new(0) };
}

unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        if COUNTING.get() {
            ALLOCATIONS.set(ALLOCATIONS.get() + 1);
        }
        unsafe { System.alloc(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        if COUNTING.get() {
            REALLOCATIONS.set(REALLOCATIONS.get() + 1);
        }
        unsafe { System.realloc(ptr, layout, new_size) }
    }
}

#[global_allocator]
static GLOBAL: CountingAllocator = CountingAllocator;

const WORKLOADS: [([usize; 2], [usize; 2], [usize; 4]); 3] = [
    ([2, 3], [0, 1], [0, 1, 2, 3]),
    ([3, 2], [0, 1], [0, 1, 2, 3]),
    ([3, 2], [0, 1], [1, 0, 2, 3]),
];

fn u1_sectors() -> Vec<SectorId> {
    [-1, 0, 1]
        .into_iter()
        .map(|charge| U1Irrep::new(charge).sector_id())
        .collect()
}

fn product_sectors() -> Vec<SectorId> {
    [(-1, 1), (0, 0), (1, 1)]
        .into_iter()
        .map(|(charge, parity)| {
            TensorKitProductCodec::try_encode(
                U1Irrep::new(charge).sector_id(),
                SectorId::new(parity),
            )
            .unwrap()
        })
        .collect()
}

fn assert_replay_allocates_nothing<R>(rule: &R, sectors: &[SectorId])
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64> + TreeTransformRuleCacheKey,
    R::Key: Clone + Eq + std::hash::Hash,
{
    let leg = || SectorLeg::new(sectors.iter().map(|&sector| (sector, 1)), false);
    let homspace = || {
        FusionTreeHomSpace::new(
            FusionProductSpace::new([leg(), leg()]),
            FusionProductSpace::new([leg(), leg()]),
        )
    };
    let space = |hom: FusionTreeHomSpace| {
        let key_count = hom.fusion_tree_keys(rule).len();
        FusionTensorMapSpace::from_degeneracy_shapes(
            TensorMapSpace::<2, 2>::from_dims(
                [sectors.len(), sectors.len()],
                [sectors.len(), sectors.len()],
            )
            .unwrap(),
            hom,
            rule,
            vec![vec![1; 4]; key_count],
        )
        .unwrap()
    };
    let lhs_space = space(homspace());
    let rhs_space = space(homspace());
    let lhs = TensorMap::<f64, 2, 2>::from_vec_with_fusion_space(
        (0..lhs_space.required_len().unwrap())
            .map(|index| index as f64 * 0.25 - 2.0)
            .collect(),
        lhs_space,
    )
    .unwrap();
    let rhs = TensorMap::<f64, 2, 2>::from_vec_with_fusion_space(
        (0..rhs_space.required_len().unwrap())
            .map(|index| index as f64 * 0.5 - 3.0)
            .collect(),
        rhs_space,
    )
    .unwrap();

    for (lhs_axes, rhs_axes, output_axes) in WORKLOADS {
        let axes = || {
            TensorContractSpec::new(
                &lhs_axes,
                &rhs_axes,
                OutputAxisOrder::from_axes(&output_axes),
            )
        };
        let dst_hom = FusionTreeHomSpace::tensorcontract_homspace(
            rule,
            lhs.fusion_space().unwrap().homspace(),
            rhs.fusion_space().unwrap().homspace(),
            &lhs_axes,
            &rhs_axes,
            &output_axes,
            2,
        )
        .unwrap();
        let dst_space = space(dst_hom);
        let mut expected = TensorMap::<f64, 2, 2>::from_vec_with_fusion_space(
            vec![0.0; dst_space.required_len().unwrap()],
            dst_space,
        )
        .unwrap();
        let mut context = TensorContractFusionExecutionContext::<f64, R::Key>::default();
        context
            .tensorcontract_fusion_into(rule, &mut expected, &lhs, &rhs, axes(), 1.0, 0.0)
            .unwrap();
        let mut actual = expected.clone();
        let prepared = context
            .prepare_tensorcontract_fusion(rule, &actual, &lhs, &rhs, axes())
            .unwrap();
        for _ in 0..2 {
            context
                .execute_prepared_tensorcontract_fusion(
                    &prepared,
                    rule,
                    &mut actual,
                    &lhs,
                    &rhs,
                    1.0,
                    0.0,
                )
                .unwrap();
        }

        ALLOCATIONS.set(0);
        REALLOCATIONS.set(0);
        COUNTING.set(true);
        let result = context.execute_prepared_tensorcontract_fusion(
            &prepared,
            rule,
            &mut actual,
            &lhs,
            &rhs,
            1.0,
            0.0,
        );
        COUNTING.set(false);
        result.unwrap();

        assert_eq!(actual.data(), expected.data());
        assert_eq!(ALLOCATIONS.get(), 0, "axes={lhs_axes:?}/{output_axes:?}");
        assert_eq!(REALLOCATIONS.get(), 0, "axes={lhs_axes:?}/{output_axes:?}");
    }
}

#[test]
fn warmed_prepared_rank4_fusion_replay_allocates_nothing() {
    assert_replay_allocates_nothing(&U1FusionRule, &u1_sectors());
    assert_replay_allocates_nothing(
        &FermionParityFusionRule,
        &[SectorId::new(0), SectorId::new(1)],
    );
    assert_replay_allocates_nothing(
        &tenet_core::SU2FusionRule,
        &[
            SU2Irrep::from_twice_spin(0).sector_id(),
            SU2Irrep::from_twice_spin(1).sector_id(),
            SU2Irrep::from_twice_spin(2).sector_id(),
        ],
    );
    assert_replay_allocates_nothing(
        &ProductFusionRule::<U1FusionRule, FermionParityFusionRule>::new(
            U1FusionRule,
            FermionParityFusionRule,
        ),
        &product_sectors(),
    );
}
