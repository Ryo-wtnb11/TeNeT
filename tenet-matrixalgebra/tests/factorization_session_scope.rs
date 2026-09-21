//! #1361: the compact-QR matricization fallbacks (multiplicity-free and
//! Generic) and the checked-Generic compact QR enter one CPU linear-algebra
//! session per call, however many coupled sectors they factorize. The
//! direct-region routes are gated through the facade in
//! `tenet/tests/factorization_session_scope.rs`.
//!
//! This is a separate test binary because `cpu_session_stats` is a
//! process-wide counter; the mutex serializes its readers here.
//!
//! Route: an adjoint view keeps the source blocks' offsets and strides but
//! swaps their codomain and domain axes, so its structure has no coupled-sector
//! matrix regions (asserted below). Both compact-QR entries take the direct
//! route only when `coupled_sector_regions(nout)` is `Some`
//! (`build_compact_factor_plan`, `prepare_compact_factor_plan_generic_checked`),
//! so these fixtures take the per-sector matricization fallback. The
//! checked-Generic entry has no direct route: it always factorizes the
//! per-sector matrices of `generic_input_matricizations`.

#![cfg(feature = "cpu-faer")]

use std::sync::{Arc, Mutex};

use tenet_core::{
    BraidingStyleKind, CheckedGenericFusion, FusionProductSpace, FusionRule, FusionStyleKind,
    FusionTensorMapSpace, FusionTreeHomSpace, RuleIdentity, SectorId, SectorLeg, SectorVec,
    TensorMap, TensorMapSpace, U1FusionRule, U1Irrep,
};
use tenet_dense::{cpu_session_stats, DefaultDenseExecutor};
use tenet_matrixalgebra::{
    qr_compact_dyn, qr_compact_dyn_checked_generic, qr_compact_dyn_generic, BoundDynamicTensorRef,
    BoundTensorMap,
};
use tenet_tensors::BoundDynamicFusionMapSpace;

static COUNTER_LOCK: Mutex<()> = Mutex::new(());

fn sessions_during<T>(call: impl FnOnce() -> T) -> u64 {
    let before = cpu_session_stats().sessions_opened;
    let _ = call();
    cpu_session_stats().sessions_opened - before
}

/// A U(1) 2 <- 2 tensor with legs over `charges`, degeneracy 2 each.
fn u1_tensor(charges: &[i32]) -> BoundTensorMap<U1FusionRule, f64, 2, 2> {
    let rule = U1FusionRule;
    let degeneracy = 2usize;
    let leg = || {
        SectorLeg::new(
            charges
                .iter()
                .map(|&q| (U1Irrep::new(q).sector_id(), degeneracy)),
            false,
        )
    };
    let dim = charges.len() * degeneracy;
    let homspace = FusionTreeHomSpace::new(
        FusionProductSpace::new([leg(), leg()]),
        FusionProductSpace::new([leg(), leg()]),
    );
    let keys = homspace.fusion_tree_keys(&rule).len();
    let space = FusionTensorMapSpace::from_degeneracy_shapes_coupled(
        TensorMapSpace::<2, 2>::from_dims([dim, dim], [dim, dim]).unwrap(),
        homspace,
        &rule,
        vec![vec![degeneracy; 4]; keys],
    )
    .unwrap();
    let len = space.required_len().unwrap();
    let data = (0..len)
        .map(|i| ((i * 11 + 5) % 29) as f64 * 0.25 - 3.0)
        .collect();
    let tensor = TensorMap::<f64, 2, 2>::from_vec_with_fusion_space(data, space).unwrap();
    BoundTensorMap::try_new(Arc::new(rule), tensor).unwrap()
}

/// A Generic toy rule with one self-dual sector `x` whose `x ⊗ x` channel has
/// multiplicity 2: the smallest rule that takes the Generic entry.
#[derive(Clone, Copy)]
struct ToyGenericRule;

impl FusionRule for ToyGenericRule {
    fn rule_identity(&self) -> RuleIdentity {
        RuleIdentity::of_type::<Self>()
    }

    fn fusion_style(&self) -> FusionStyleKind {
        FusionStyleKind::Generic
    }

    fn braiding_style(&self) -> BraidingStyleKind {
        BraidingStyleKind::Bosonic
    }

    fn vacuum(&self) -> SectorId {
        SectorId::new(0)
    }

    fn dual(&self, sector: SectorId) -> SectorId {
        sector
    }

    fn fusion_channels(&self, left: SectorId, right: SectorId) -> SectorVec {
        match (left.id(), right.id()) {
            (0, x) | (x, 0) => [SectorId::new(x)].into_iter().collect(),
            (1, 1) => [SectorId::new(0), SectorId::new(1)].into_iter().collect(),
            _ => SectorVec::new(),
        }
    }

    fn nsymbol(&self, left: SectorId, right: SectorId, coupled: SectorId) -> usize {
        if (left.id(), right.id(), coupled.id()) == (1, 1, 1) {
            2
        } else {
            usize::from(self.fusion_channels(left, right).contains(&coupled))
        }
    }
}

/// [`ToyGenericRule`] behind the fallible checked-Generic provider surface.
struct CheckedToyRule;

impl CheckedGenericFusion for CheckedToyRule {
    type Error = std::convert::Infallible;

    fn rule_identity(&self) -> RuleIdentity {
        RuleIdentity::of_type::<Self>()
    }

    fn fusion_style(&self) -> FusionStyleKind {
        ToyGenericRule.fusion_style()
    }

    fn braiding_style(&self) -> BraidingStyleKind {
        ToyGenericRule.braiding_style()
    }

    fn vacuum(&self) -> SectorId {
        ToyGenericRule.vacuum()
    }

    fn try_dual(&self, sector: SectorId) -> Result<SectorId, Self::Error> {
        Ok(ToyGenericRule.dual(sector))
    }

    fn try_fusion_channels(
        &self,
        left: SectorId,
        right: SectorId,
    ) -> Result<SectorVec, Self::Error> {
        Ok(ToyGenericRule.fusion_channels(left, right))
    }

    fn try_fusion_channels_in_table(
        &self,
        left: SectorId,
        right: SectorId,
    ) -> Result<SectorVec, Self::Error> {
        Ok(ToyGenericRule.fusion_channels(left, right))
    }

    fn try_nsymbol(
        &self,
        left: SectorId,
        right: SectorId,
        coupled: SectorId,
    ) -> Result<usize, Self::Error> {
        Ok(ToyGenericRule.nsymbol(left, right, coupled))
    }
}

fn assert_not_matrix_layout<R: FusionRule>(space: &BoundDynamicFusionMapSpace<R>) {
    let regions = space
        .space()
        .structure()
        .coupled_sector_regions(space.space().nout())
        .unwrap();
    assert!(
        regions.is_none(),
        "fixture must not be a coupled-sector matrix layout"
    );
}

fn assert_fallback_and_checked_generic_routes_open_one_session(mut dense: DefaultDenseExecutor) {
    let _guard = COUNTER_LOCK.lock().unwrap_or_else(|p| p.into_inner());

    // U(1) charges -1, 0, 1 on each leg: five coupled sectors on a 2 <- 2 map.
    let bound = u1_tensor(&[-1, 0, 1]);
    let adjoint = bound.space().adjoint_view().unwrap();
    assert_not_matrix_layout(&adjoint);
    let input = BoundDynamicTensorRef::try_new(&adjoint, bound.data()).unwrap();
    assert_eq!(
        sessions_during(|| qr_compact_dyn(&mut dense, &input).unwrap()),
        1,
        "multiplicity-free matricization qr_compact"
    );

    // Generic: sectors {0, x} on every leg, so both coupled sectors 0 and x
    // occur, x with fusion multiplicity.
    let leg = |d0, d1| SectorLeg::new([(SectorId::new(0), d0), (SectorId::new(1), d1)], false);
    let homspace = FusionTreeHomSpace::new(
        FusionProductSpace::new([leg(2, 2), leg(1, 2)]),
        FusionProductSpace::new([leg(2, 1)]),
    );
    let space =
        BoundDynamicFusionMapSpace::from_final_homspace_generic(Arc::new(ToyGenericRule), homspace)
            .unwrap();
    let data = (0..space.space().required_len().unwrap())
        .map(|i| 1.0 + ((i * 7 + 3) % 13) as f64 / 8.0)
        .collect::<Vec<f64>>();
    let adjoint = space.adjoint_view().unwrap();
    assert_not_matrix_layout(&adjoint);
    let input = BoundDynamicTensorRef::try_new(&adjoint, &data).unwrap();
    assert_eq!(
        sessions_during(|| qr_compact_dyn_generic(&mut dense, &input).unwrap()),
        1,
        "Generic matricization qr_compact"
    );

    let checked = BoundDynamicFusionMapSpace::from_final_homspace_generic_checked(
        Arc::new(CheckedToyRule),
        space.space().homspace().clone(),
    )
    .unwrap();
    let data = (0..checked.space().required_len().unwrap())
        .map(|i| 1.0 + ((i * 5 + 2) % 11) as f64 / 8.0)
        .collect::<Vec<f64>>();
    let input = BoundDynamicTensorRef::try_new(&checked, &data).unwrap();
    assert_eq!(
        sessions_during(|| qr_compact_dyn_checked_generic(&mut dense, &input).unwrap()),
        1,
        "checked-Generic qr_compact"
    );
}

#[test]
fn qr_fallback_and_checked_generic_routes_open_one_session_at_one_thread() {
    assert_fallback_and_checked_generic_routes_open_one_session(
        DefaultDenseExecutor::with_threads(1).unwrap(),
    );
}

#[test]
fn qr_fallback_and_checked_generic_routes_open_one_session_at_default_threads() {
    assert_fallback_and_checked_generic_routes_open_one_session(DefaultDenseExecutor::new());
}
