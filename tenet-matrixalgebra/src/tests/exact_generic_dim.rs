//! The infallible Generic testing tier weighs a sector by its exact quantum
//! dimension, as the multiplicity-free tier does (#2076).
//!
//! TensorKitSectors `dim(::Sector)` is an exact categorical dimension (`Int`
//! for SU(2)); TensorKit `findtruncated(::SectorVector, ...)` compares
//! `totaldim > howmany` against it without slack (`truncation.jl:198`). The
//! oracle is SU(2) through both tiers: `SU2FusionRule` (multiplicity-free) and
//! the same data lifted to `GenericRigidSymbols`, whose `dim_scalar` is the
//! hand-written `2j + 1`.

use super::*;
use crate::seam::{decide_bond_truncation, decide_bond_truncation_generic_checked};

/// SU(2) as an infallible Generic provider: 1x1 symbol blocks lifted from
/// [`SU2FusionRule`], under its own identity.
struct Su2Generic;

impl FusionRule for Su2Generic {
    fn rule_identity(&self) -> RuleIdentity {
        RuleIdentity::of_type::<Self>()
    }
    fn fusion_style(&self) -> FusionStyleKind {
        SU2FusionRule.fusion_style()
    }
    fn braiding_style(&self) -> BraidingStyleKind {
        SU2FusionRule.braiding_style()
    }
    fn vacuum(&self) -> SectorId {
        SU2FusionRule.vacuum()
    }
    fn dual(&self, sector: SectorId) -> SectorId {
        SU2FusionRule.dual(sector)
    }
    fn fusion_channels(&self, left: SectorId, right: SectorId) -> SectorVec {
        SU2FusionRule.fusion_channels(left, right)
    }
    fn nsymbol(&self, left: SectorId, right: SectorId, coupled: SectorId) -> usize {
        SU2FusionRule.nsymbol(left, right, coupled)
    }
    fn sector_order_key(&self, sector: SectorId) -> tenet_core::SectorOrderKey {
        SU2FusionRule.sector_order_key(sector)
    }
}

impl GenericFusionSymbols for Su2Generic {
    type Scalar = f64;

    fn f_symbol_generic(
        &self,
        a: SectorId,
        b: SectorId,
        c: SectorId,
        d: SectorId,
        e: SectorId,
        f: SectorId,
    ) -> GenericFArray<f64> {
        let shape = (
            self.nsymbol(a, b, e),
            self.nsymbol(e, c, d),
            self.nsymbol(b, c, f),
            self.nsymbol(a, f, d),
        );
        let len = shape.0 * shape.1 * shape.2 * shape.3;
        let value = SU2FusionRule.f_symbol_scalar(a, b, c, d, e, f);
        GenericFArray::new(vec![value; len], shape)
    }

    fn r_symbol_generic(&self, a: SectorId, b: SectorId, c: SectorId) -> GenericRMatrix<f64> {
        let (rows, cols) = (self.nsymbol(a, b, c), self.nsymbol(b, a, c));
        let value = SU2FusionRule.r_symbol_scalar(a, b, c);
        GenericRMatrix::new(vec![value; rows * cols], rows, cols)
    }
}

impl GenericRigidSymbols for Su2Generic {
    fn dim_scalar(&self, sector: SectorId) -> f64 {
        (sector.id() + 1) as f64
    }
    fn sqrt_dim_scalar(&self, sector: SectorId) -> f64 {
        self.dim_scalar(sector).sqrt()
    }
    fn inv_sqrt_dim_scalar(&self, sector: SectorId) -> f64 {
        self.sqrt_dim_scalar(sector).recip()
    }
    fn frobenius_schur_phase_scalar(&self, sector: SectorId) -> f64 {
        SU2FusionRule.frobenius_schur_phase_scalar(sector)
    }
}

/// `(2j, values)` per sector: dims 1, 2, 3, 5, 7, 8; `sqrt(dim)^2` is off
/// for all but 1.
fn spectra() -> Vec<SectorSpectrum> {
    [
        (0, vec![0.93, 0.41]),
        (1, vec![0.87, 0.52, 0.11]),
        (2, vec![0.74, 0.29]),
        (4, vec![0.66, 0.2]),
        (6, vec![0.58]),
        (7, vec![0.47]),
    ]
    .into_iter()
    .map(|(twice, values)| SectorSpectrum {
        sector: SectorId::new(twice),
        values,
    })
    .collect()
}

#[test]
fn infallible_generic_reports_the_exact_dimension() {
    // What: the checked view's dim is `2j + 1`, not `sqrt(2j + 1)^2`, so the
    // trace channel factor `dim(c) / dim(a)` is the exact quotient.
    let provider = InfallibleGeneric::new(&Su2Generic);
    let mut inexact_squares = 0;
    for twice in 0..8 {
        let sector = SectorId::new(twice);
        let dim = (twice + 1) as f64;
        assert_eq!(
            provider.try_dim_scalar(sector).unwrap(),
            dim,
            "2j = {twice}"
        );
        let sqrt_dim = provider.try_sqrt_dim_scalar(sector).unwrap();
        inexact_squares += usize::from(sqrt_dim * sqrt_dim != dim);
    }
    // The sqrt² route would differ, so the assertions above discriminate.
    assert!(inexact_squares >= 4);
    let quotient = |c: usize, a: usize| {
        provider.try_dim_scalar(SectorId::new(c)).unwrap()
            / provider.try_dim_scalar(SectorId::new(a)).unwrap()
    };
    assert_eq!(quotient(4, 2), 5.0 / 3.0);
    assert_eq!(quotient(6, 7), 7.0 / 8.0);
}

#[test]
fn infallible_generic_keeps_a_state_that_meets_the_rank_budget_exactly() {
    // What: one spin-2 value weighs 5; a budget of 5 keeps it and 4 does not.
    // `sqrt(5)^2 = 5.000000000000001` would overflow the budget of 5.
    let provider = InfallibleGeneric::new(&Su2Generic);
    let spectra = [SectorSpectrum {
        sector: SectorId::new(4),
        values: vec![1.0],
    }];
    for (budget, kept) in [(4, false), (5, true)] {
        let decision =
            decide_bond_truncation_generic_checked(&provider, &spectra, &Truncation::rank(budget))
                .unwrap();
        assert_eq!(decision.kept, vec![vec![kept]], "budget {budget}");
    }
}

#[test]
fn multiplicity_free_and_infallible_generic_take_one_rank_decision() {
    // What: SU(2) through either tier keeps the same states and reports the
    // same discarded weight under every rank budget.
    let spectra = spectra();
    let provider = InfallibleGeneric::new(&Su2Generic);
    let total: usize = spectra
        .iter()
        .map(|entry| (entry.sector.id() + 1) * entry.values.len())
        .sum();
    for budget in 0..=total + 1 {
        let truncation = Truncation::rank(budget);
        let mf = decide_bond_truncation(&SU2FusionRule, &spectra, &truncation).unwrap();
        let generic =
            decide_bond_truncation_generic_checked(&provider, &spectra, &truncation).unwrap();
        assert_eq!(mf.kept, generic.kept, "budget {budget}");
        assert_eq!(
            mf.error.to_bits(),
            generic.error.to_bits(),
            "budget {budget}"
        );
    }
}
