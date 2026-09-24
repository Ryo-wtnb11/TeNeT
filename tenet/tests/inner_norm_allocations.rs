use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::collections::BTreeSet;
use std::fmt;
use std::hint::black_box;
use std::sync::Arc;

use tenet::core::{
    product_sector, BraidingStyleKind, CheckedGenericAdmissionMode, CheckedGenericFusion,
    CheckedGenericRigidSymbols, FermionParityFusionRule, FusionStyleKind, Fz2SectorLayout,
    GenericFArray, GenericRMatrix, PackedProductCodec, ProductFusionRule, ProductSectorLayout,
    RuleIdentity, SU2FusionRule, SU2Irrep, SectorId, SectorVec, Su2SectorLayout,
    TypedSectorAdmission, U1FusionRule, U1Irrep, U1SectorLayout, Z2Irrep,
};
use tenet::prelude::{Complex64, GradedSpace, Runtime, TensorMap};
use tenet::typed::TensorScalar;

type Fz2U1Codec = PackedProductCodec<Fz2SectorLayout, U1SectorLayout>;
type Fz2U1Layout = ProductSectorLayout<Fz2SectorLayout, U1SectorLayout>;
type Fz2U1Su2Codec = PackedProductCodec<Fz2U1Layout, Su2SectorLayout>;
type Fz2U1Rule = ProductFusionRule<FermionParityFusionRule, U1FusionRule, Fz2U1Codec>;
type Fz2U1Su2Rule = ProductFusionRule<Fz2U1Rule, SU2FusionRule, Fz2U1Su2Codec>;

struct CountingAllocator;

thread_local! {
    static COUNTING: Cell<bool> = const { Cell::new(false) };
    static ALLOCATIONS: Cell<usize> = const { Cell::new(0) };
}

unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let pointer = unsafe { System.alloc(layout) };
        if !pointer.is_null() && COUNTING.get() {
            ALLOCATIONS.set(ALLOCATIONS.get() + 1);
        }
        pointer
    }

    unsafe fn dealloc(&self, pointer: *mut u8, layout: Layout) {
        unsafe { System.dealloc(pointer, layout) };
    }

    unsafe fn realloc(&self, pointer: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        let pointer = unsafe { System.realloc(pointer, layout, new_size) };
        if !pointer.is_null() && COUNTING.get() {
            ALLOCATIONS.set(ALLOCATIONS.get() + 1);
        }
        pointer
    }
}

#[global_allocator]
static ALLOCATOR: CountingAllocator = CountingAllocator;

fn measured<T>(operation: impl FnOnce() -> T) -> (T, usize) {
    ALLOCATIONS.set(0);
    COUNTING.set(true);
    let value = operation();
    COUNTING.set(false);
    (value, ALLOCATIONS.get())
}

fn non_abelian_space() -> GradedSpace<Fz2U1Su2Rule> {
    let rule = Arc::new(Fz2U1Su2Rule::new(
        Fz2U1Rule::new(FermionParityFusionRule, U1FusionRule),
        SU2FusionRule,
    ));
    let label = |parity: u8, charge: i32, twice_spin: usize| {
        product_sector(
            product_sector(
                if parity == 0 {
                    Z2Irrep::EVEN
                } else {
                    Z2Irrep::ODD
                },
                U1Irrep::new(charge),
            ),
            SU2Irrep::from_twice_spin(twice_spin),
        )
    };
    GradedSpace::try_new_with_arc(
        rule,
        [
            (label(0, -2, 0), 4),
            (label(0, 1, 2), 3),
            (label(1, -1, 1), 4),
            (label(1, 2, 3), 2),
        ],
    )
    .unwrap()
}

#[test]
fn warmed_non_abelian_inner_and_norm_do_not_allocate() {
    // What: cold region compilation is observable, while explicit warm-up
    // leaves both public reductions allocation-free on the caller thread.
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let space = non_abelian_space();
    let lhs: TensorMap<Fz2U1Su2Rule, Complex64> =
        TensorMap::rand_with_seed(&runtime, [&space, &space], [&space], 282_401).unwrap();
    let rhs: TensorMap<Fz2U1Su2Rule, Complex64> =
        TensorMap::rand_with_seed(&runtime, [&space, &space], [&space], 282_402).unwrap();

    let (cold, cold_allocations) = measured(|| lhs.inner(&rhs).unwrap());
    eprintln!("cold coupled-region initialization: {cold_allocations} allocations");
    black_box(cold);
    black_box(lhs.norm().unwrap());

    let (inner, inner_allocations) = measured(|| lhs.inner(&rhs).unwrap());
    let (norm, norm_allocations) = measured(|| lhs.norm().unwrap());
    black_box((inner, norm));

    assert_eq!(inner_allocations, 0);
    assert_eq!(norm_allocations, 0);

    // Out-of-range payloads take the rescaling passes, which add none either.
    for scale in [1e200, 1e-200] {
        let extreme = lhs.scale(Complex64::new(scale, 0.0));
        black_box(extreme.norm().unwrap());
        let (value, allocations) = measured(|| extreme.norm().unwrap());
        assert!(value.is_finite() && value > 0.0, "{scale:e}: {value:e}");
        assert_eq!(allocations, 0, "rescaled norm at {scale:e}");
        let (value, allocations) = measured(|| extreme.norm_p(3.0).unwrap());
        assert!(value.is_finite() && value > 0.0, "{scale:e}: {value:e}");
        assert_eq!(allocations, 0, "rescaled norm_p(3) at {scale:e}");
    }
}

#[test]
fn warmed_non_abelian_trace_does_not_allocate() {
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let space = non_abelian_space();
    let owned: TensorMap<Fz2U1Su2Rule, Complex64> =
        TensorMap::rand_with_seed(&runtime, [&space, &space], [&space, &space], 282_403).unwrap();
    let lazy = owned.adjoint().unwrap();

    black_box(owned.tr().unwrap());
    black_box(lazy.tr().unwrap());
    for (row, (value, allocations)) in [
        ("owned trace", measured(|| owned.tr().unwrap())),
        ("lazy trace", measured(|| lazy.tr().unwrap())),
    ] {
        black_box(value);
        assert_eq!(allocations, 0, "{row}");
    }
}

#[test]
fn warmed_lazy_adjoint_inner_does_not_allocate_in_mixed_or_double_orientation() {
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let space = non_abelian_space();
    let lhs_parent: TensorMap<Fz2U1Su2Rule, Complex64> =
        TensorMap::rand_with_seed(&runtime, [&space, &space], [&space], 666_301).unwrap();
    let rhs_parent: TensorMap<Fz2U1Su2Rule, Complex64> =
        TensorMap::rand_with_seed(&runtime, [&space, &space], [&space], 666_302).unwrap();
    let owned: TensorMap<Fz2U1Su2Rule, Complex64> =
        TensorMap::rand_with_seed(&runtime, [&space], [&space, &space], 666_303).unwrap();
    let warm_lhs = lhs_parent.adjoint().unwrap();
    let warm_rhs = rhs_parent.adjoint().unwrap();
    black_box(warm_lhs.inner(&owned).unwrap());
    black_box(owned.inner(&warm_lhs).unwrap());
    black_box(warm_lhs.inner(&warm_rhs).unwrap());

    let lhs_mixed_left = lhs_parent.adjoint().unwrap();
    let lhs_mixed_right = lhs_parent.adjoint().unwrap();
    let lhs_double = lhs_parent.adjoint().unwrap();
    let rhs_double = rhs_parent.adjoint().unwrap();
    for (value, allocations) in [
        measured(|| lhs_mixed_left.inner(&owned).unwrap()),
        measured(|| owned.inner(&lhs_mixed_right).unwrap()),
        measured(|| lhs_double.inner(&rhs_double).unwrap()),
    ] {
        black_box(value);
        assert_eq!(allocations, 0);
    }
    for lazy in [&lhs_mixed_left, &lhs_mixed_right, &lhs_double, &rhs_double] {
        let parent_len = lhs_parent.data().len();
        let (materialized_len, allocations) = measured(|| lazy.data().len());
        assert_eq!(materialized_len, parent_len);
        assert!(allocations > 0, "inner materialized its lazy operand");
    }
}

/// Checked Generic toy with outer multiplicity two on `X (x) X -> X` and an
/// allocation-free `dim`, so the rows below measure the reduction owners and
/// not the provider.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
enum CheckedLabel {
    Vacuum,
    X,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct CheckedError;

impl fmt::Display for CheckedError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("invalid synthetic checked sector")
    }
}

impl std::error::Error for CheckedError {}

struct CheckedToy;

impl CheckedToy {
    const VACUUM: SectorId = SectorId::new(0);
    const X: SectorId = SectorId::new(1);

    fn channels(left: SectorId, right: SectorId) -> Result<SectorVec, CheckedError> {
        match (left, right) {
            (Self::VACUUM, sector) | (sector, Self::VACUUM)
                if sector == Self::VACUUM || sector == Self::X =>
            {
                Ok([sector].into_iter().collect())
            }
            (Self::X, Self::X) => Ok([Self::VACUUM, Self::X].into_iter().collect()),
            _ => Err(CheckedError),
        }
    }

    fn multiplicity(left: SectorId, right: SectorId, coupled: SectorId) -> usize {
        if (left, right, coupled) == (Self::X, Self::X, Self::X) {
            2
        } else {
            usize::from(
                Self::channels(left, right).is_ok_and(|channels| channels.contains(&coupled)),
            )
        }
    }
}

impl CheckedGenericFusion for CheckedToy {
    type Error = CheckedError;

    fn rule_identity(&self) -> RuleIdentity {
        RuleIdentity::from_canonical_bytes::<Self>(0x1219, Arc::<[u8]>::from(*b"checked-toy"))
    }

    fn fusion_style(&self) -> FusionStyleKind {
        FusionStyleKind::Generic
    }

    fn braiding_style(&self) -> BraidingStyleKind {
        BraidingStyleKind::Bosonic
    }

    fn vacuum(&self) -> SectorId {
        Self::VACUUM
    }

    fn try_dual(&self, sector: SectorId) -> Result<SectorId, Self::Error> {
        match sector {
            Self::VACUUM | Self::X => Ok(sector),
            _ => Err(CheckedError),
        }
    }

    fn try_fusion_channels(
        &self,
        left: SectorId,
        right: SectorId,
    ) -> Result<SectorVec, Self::Error> {
        Self::channels(left, right)
    }

    fn try_fusion_channels_in_table(
        &self,
        left: SectorId,
        right: SectorId,
    ) -> Result<SectorVec, Self::Error> {
        Self::channels(left, right)
    }

    fn try_nsymbol(
        &self,
        left: SectorId,
        right: SectorId,
        coupled: SectorId,
    ) -> Result<usize, Self::Error> {
        Ok(Self::multiplicity(left, right, coupled))
    }
}

impl CheckedGenericRigidSymbols for CheckedToy {
    type Scalar = f64;

    fn try_sqrt_dim_scalar(&self, sector: SectorId) -> Result<f64, Self::Error> {
        match sector {
            Self::VACUUM => Ok(1.0),
            Self::X => Ok((1.0 + 2.0_f64.sqrt()).sqrt()),
            _ => Err(CheckedError),
        }
    }

    fn try_inv_sqrt_dim_scalar(&self, sector: SectorId) -> Result<f64, Self::Error> {
        Ok(self.try_sqrt_dim_scalar(sector)?.recip())
    }

    fn try_frobenius_schur_phase_scalar(&self, sector: SectorId) -> Result<f64, Self::Error> {
        self.try_dual(sector).map(|_| 1.0)
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
        let shape = (
            Self::multiplicity(a, b, e),
            Self::multiplicity(e, c, d),
            Self::multiplicity(b, c, f),
            Self::multiplicity(a, f, d),
        );
        let rows = shape.0 * shape.1;
        let cols = shape.2 * shape.3;
        Ok(GenericFArray::new(
            (0..rows * cols)
                .map(|index| f64::from(index / cols == index % cols))
                .collect(),
            shape,
        ))
    }

    fn try_r_symbol_generic(
        &self,
        a: SectorId,
        b: SectorId,
        c: SectorId,
    ) -> Result<GenericRMatrix<f64>, Self::Error> {
        let size = Self::multiplicity(a, b, c);
        Ok(GenericRMatrix::new(
            (0..size * size)
                .map(|index| f64::from(index / size == index % size))
                .collect(),
            size,
            size,
        ))
    }
}

impl TypedSectorAdmission for CheckedToy {
    type Sector = CheckedLabel;
    type Error = CheckedError;
    type Mode = CheckedGenericAdmissionMode;

    fn typed_rule_identity(&self) -> RuleIdentity {
        CheckedGenericFusion::rule_identity(self)
    }

    fn try_encode_label(&self, sector: &Self::Sector) -> Result<SectorId, Self::Error> {
        Ok(match sector {
            CheckedLabel::Vacuum => Self::VACUUM,
            CheckedLabel::X => Self::X,
        })
    }

    fn try_decode_label(&self, sector: SectorId) -> Result<Self::Sector, Self::Error> {
        match sector {
            Self::VACUUM => Ok(CheckedLabel::Vacuum),
            Self::X => Ok(CheckedLabel::X),
            _ => Err(CheckedError),
        }
    }

    fn try_dual_id(&self, sector: SectorId) -> Result<SectorId, Self::Error> {
        self.try_dual(sector)
    }
}

fn block_value<D: TensorScalar>(
    trees: &tenet::typed::BlockFusionTrees<CheckedLabel>,
    indices: &[usize],
) -> D {
    let vertex = trees.codomain_vertices()[0].get() as f64;
    D::from_real(
        vertex
            + indices.iter().enumerate().fold(0.0, |acc, (axis, &index)| {
                acc + (index as f64 + 1.0) * (axis as f64 + 1.0)
            }),
    )
}

/// Coupled sectors `G` of a tensor; each one is exactly one packed region.
macro_rules! coupled_sector_count {
    ($tensor:expr) => {{
        let tensor = &$tensor;
        (0..tensor.block_count())
            .map(|index| {
                tensor
                    .block_fusion_trees(index)
                    .unwrap()
                    .coupled()
                    .to_owned()
            })
            .collect::<BTreeSet<_>>()
            .len()
    }};
}

#[test]
fn warmed_checked_generic_reductions_do_not_allocate() {
    // What: checked Generic reductions take their `dim(c)` weights
    // through one fallible closure and never build a per-call weight map.
    // Before #1219 every warm call allocated once (a std `HashMap`) for owned
    // and lazy-adjoint inputs alike.
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let provider = Arc::new(CheckedToy);
    let leg = GradedSpace::try_new_with_arc(
        Arc::clone(&provider),
        [(CheckedLabel::Vacuum, 1), (CheckedLabel::X, 2)],
    )
    .unwrap();
    let lhs: TensorMap<CheckedToy, Complex64> =
        TensorMap::from_block_fn(&runtime, [&leg, &leg], [&leg, &leg], block_value).unwrap();
    let rhs: TensorMap<CheckedToy, Complex64> =
        TensorMap::from_block_fn(&runtime, [&leg, &leg], [&leg, &leg], |trees, indices| {
            block_value::<Complex64>(trees, indices) * Complex64::new(0.5, -1.5)
        })
        .unwrap();
    assert!(lhs.block_count() > coupled_sector_count!(lhs));
    let lazy_lhs = lhs.adjoint().unwrap();
    let lazy_rhs = rhs.adjoint().unwrap();

    let (cold, cold_allocations) = measured(|| lhs.inner(&rhs).unwrap());
    eprintln!("cold checked coupled-region initialization: {cold_allocations} allocations");
    black_box(cold);
    black_box(lhs.norm().unwrap());
    black_box(lazy_lhs.inner(&lazy_rhs).unwrap());
    black_box(lazy_lhs.inner(&rhs).unwrap());
    black_box(lazy_lhs.norm().unwrap());
    black_box(lhs.tr().unwrap());
    black_box(lazy_lhs.tr().unwrap());

    for (row, (value, allocations)) in [
        ("owned inner", measured(|| lhs.inner(&rhs).unwrap())),
        (
            "owned norm",
            measured(|| Complex64::from(lhs.norm().unwrap())),
        ),
        (
            "lazy-lazy inner",
            measured(|| lazy_lhs.inner(&lazy_rhs).unwrap()),
        ),
        (
            "lazy-owned inner",
            measured(|| lazy_lhs.inner(&rhs).unwrap()),
        ),
        (
            "owned-lazy inner",
            measured(|| rhs.inner(&lazy_lhs).unwrap()),
        ),
        (
            "lazy norm",
            measured(|| Complex64::from(lazy_lhs.norm().unwrap())),
        ),
        ("owned trace", measured(|| lhs.tr().unwrap())),
        ("lazy trace", measured(|| lazy_lhs.tr().unwrap())),
    ] {
        black_box(value);
        assert_eq!(allocations, 0, "{row}");
    }

    // The rescaled checked norm neither allocates nor loses the value: scaling
    // commutes with the norm, so `norm(s * t) = s * norm(t)` within the
    // workspace rule taken relative to the result (the payload's entry count
    // bounds the terms of the sum).
    let norm = lhs.norm().unwrap();
    for scale in [1e200, 1e-200] {
        let extreme = lhs.scale(Complex64::new(scale, 0.0));
        let lazy = extreme.adjoint().unwrap();
        black_box((extreme.norm().unwrap(), lazy.norm().unwrap()));
        for (row, tensor) in [("owned", &extreme), ("lazy", &lazy)] {
            let (value, allocations) = measured(|| tensor.norm().unwrap());
            assert_eq!(allocations, 0, "{row} rescaled norm at {scale:e}");
            let want = scale * norm;
            let bound = 32.0 * (lhs.data().len() as f64).sqrt() * f64::EPSILON * want;
            assert!(
                (value - want).abs() <= bound,
                "{row} {scale:e}: {value:e} against {want:e}"
            );
        }
    }
}

/// SU(N) checked `dim` allocates per query (`tenet-sectors/src/sun.rs::
/// decode_dynkin`'s `Vec<i64>` plus racah's own irrep decode), so the
/// reductions still pay the provider's cost once per coupled sector; the
/// reduction owners themselves add none (before #1219: one map allocation on
/// top of the G provider queries).
#[cfg(feature = "racah-generated")]
#[test]
fn warmed_su3_checked_inner_and_norm_allocate_only_through_the_provider() {
    use tenet::typed::SUNFusionRule;

    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let provider = Arc::new(SUNFusionRule::new(3).unwrap());
    let leg = GradedSpace::try_new_with_arc(Arc::clone(&provider), [(vec![1, 0], 2)]).unwrap();
    let lhs: TensorMap<SUNFusionRule, f64> =
        TensorMap::from_block_fn(&runtime, [&leg, &leg], [&leg, &leg], |trees, indices| {
            trees.codomain_vertices()[0].get() as f64 + indices.iter().sum::<usize>() as f64
        })
        .unwrap();
    let rhs: TensorMap<SUNFusionRule, f64> =
        TensorMap::from_block_fn(&runtime, [&leg, &leg], [&leg, &leg], |trees, indices| {
            2.0 * trees.domain_vertices()[0].get() as f64 - indices[0] as f64
        })
        .unwrap();
    let coupled = (0..lhs.block_count())
        .map(|index| lhs.block_fusion_trees(index).unwrap().coupled().to_owned())
        .collect::<BTreeSet<_>>();
    let sectors = coupled.len();
    assert!(sectors >= 2);
    // The provider's own per-query cost, summed over the G coupled sectors.
    let provider_allocations: usize = coupled
        .iter()
        .map(|label| {
            let sector = provider.try_encode_label(label).unwrap();
            black_box(provider.try_sqrt_dim_scalar(sector).unwrap());
            let (weight, allocations) = measured(|| provider.try_sqrt_dim_scalar(sector).unwrap());
            black_box(weight);
            allocations
        })
        .sum();
    let lazy_lhs = lhs.adjoint().unwrap();
    let lazy_rhs = rhs.adjoint().unwrap();
    black_box(lhs.inner(&rhs).unwrap());
    black_box(lhs.norm().unwrap());
    black_box(lazy_lhs.inner(&lazy_rhs).unwrap());
    black_box(lazy_lhs.norm().unwrap());

    for (row, (value, allocations)) in [
        ("owned inner", measured(|| lhs.inner(&rhs).unwrap())),
        ("owned norm", measured(|| lhs.norm().unwrap())),
        (
            "lazy inner",
            measured(|| lazy_lhs.inner(&lazy_rhs).unwrap()),
        ),
        ("lazy norm", measured(|| lazy_lhs.norm().unwrap())),
    ] {
        black_box(value);
        eprintln!(
            "su3 {row}: {allocations} allocations for {sectors} coupled sectors \
             ({provider_allocations} from the provider's dim queries)"
        );
        assert_eq!(allocations, provider_allocations, "{row}");
    }
}
