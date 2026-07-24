use num_complex::Complex64;
use smallvec::smallvec;

use crate::{
    BraidingStyleKind, CheckedFusionAlgebra, FusionAlgebraError, FusionRule, FusionStyleKind,
    MultiplicityFreeFusionRule, MultiplicityFreeFusionSymbols, MultiplicityFreeRigidSymbols,
    RuleIdentity, SectorId, SectorVec,
};

// FibonacciAnyon: the simplest genuinely non-abelian anyon model (Simple
// fusion + Anyonic braiding + complex F/R symbols) — SectorId 0 = vacuum
// `𝟙`, SectorId 1 = `τ`, with `τ⊗τ = 𝟙 ⊕ τ`. All numeric F/R/dim/twist
// values below are copied verbatim from TensorKitSectors.jl's
// `FibonacciAnyon` (`~/.julia/packages/TensorKitSectors/tugbK/src/anyons.jl`,
// lines 82-146) — never "simplify" a sign or phase here without rereading
// that source (project convention: don't derive anyon conventions from
// "should be").
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Hash)]
pub struct FibonacciFusionRule;

impl FibonacciFusionRule {
    /// `false` for the vacuum (`𝟙`, SectorId 0), `true` for `τ` (SectorId 1).
    fn is_tau(sector: SectorId) -> bool {
        sector.id() != 0
    }
}

/// `dim(FibonacciAnyon)` (anyons.jl:82-83): `𝟙 -> 1`, `τ -> φ = (1+√5)/2`
/// (`Float64(MathConstants.golden)`).
fn fibonacci_quantum_dim(sector: SectorId) -> f64 {
    if FibonacciFusionRule::is_tau(sector) {
        (1.0 + 5.0_f64.sqrt()) / 2.0
    } else {
        1.0
    }
}

impl FusionRule for FibonacciFusionRule {
    fn rule_identity(&self) -> RuleIdentity {
        RuleIdentity::of_type::<Self>()
    }

    fn fusion_style(&self) -> FusionStyleKind {
        FusionStyleKind::Simple
    }

    fn braiding_style(&self) -> BraidingStyleKind {
        BraidingStyleKind::Anyonic
    }

    fn vacuum(&self) -> SectorId {
        SectorId::new(0)
    }

    // `dual(s) = s` (anyons.jl:80: `dual(s::FibonacciAnyon) = s`) is exactly
    // the `FusionRule::dual` default (identity) — no override needed.
    fn fusion_channels(&self, left: SectorId, right: SectorId) -> SectorVec {
        match (Self::is_tau(left), Self::is_tau(right)) {
            (false, _) => smallvec![right],
            (true, false) => smallvec![left],
            // τ⊗τ = 𝟙 ⊕ τ, vacuum-first to match TensorKitSectors'
            // `FibonacciAnyonProdIterator` iteration order (anyons.jl:96-109).
            (true, true) => smallvec![SectorId::new(0), SectorId::new(1)],
        }
    }
}

fn checked_fibonacci_sector(sector: SectorId) -> Result<(), FusionAlgebraError> {
    (sector.id() <= 1)
        .then_some(())
        .ok_or(FusionAlgebraError::InvalidSector { sector })
}

impl CheckedFusionAlgebra for FibonacciFusionRule {
    fn try_dual_sector(&self, sector: SectorId) -> Result<SectorId, FusionAlgebraError> {
        checked_fibonacci_sector(sector)?;
        Ok(sector)
    }

    fn try_fusion_channels(
        &self,
        left: SectorId,
        right: SectorId,
    ) -> Result<SectorVec, FusionAlgebraError> {
        checked_fibonacci_sector(left)?;
        checked_fibonacci_sector(right)?;
        Ok(self.fusion_channels(left, right))
    }

    fn try_nsymbol(
        &self,
        left: SectorId,
        right: SectorId,
        coupled: SectorId,
    ) -> Result<usize, FusionAlgebraError> {
        checked_fibonacci_sector(left)?;
        checked_fibonacci_sector(right)?;
        checked_fibonacci_sector(coupled)?;
        Ok(self.nsymbol(left, right, coupled))
    }
}

impl MultiplicityFreeFusionRule for FibonacciFusionRule {}

impl MultiplicityFreeFusionSymbols for FibonacciFusionRule {
    type Scalar = Complex64;

    fn scalar_one(&self) -> Self::Scalar {
        Complex64::new(1.0, 0.0)
    }

    fn scalar_conj(&self, value: Self::Scalar) -> Self::Scalar {
        value.conj()
    }

    // Verbatim port of `Fsymbol` (anyons.jl:115-137): four `Nsymbol` gates,
    // then the single non-trivial 2x2 block `F^{τττ}_τ` (entries ±1/φ,
    // ±1/√φ); every other allowed configuration is 1.
    fn f_symbol_scalar(
        &self,
        left: SectorId,
        middle: SectorId,
        right: SectorId,
        coupled: SectorId,
        left_coupled: SectorId,
        right_coupled: SectorId,
    ) -> Self::Scalar {
        if self.nsymbol(left, middle, left_coupled) == 0
            || self.nsymbol(left_coupled, right, coupled) == 0
            || self.nsymbol(middle, right, right_coupled) == 0
            || self.nsymbol(left, right_coupled, coupled) == 0
        {
            return Complex64::new(0.0, 0.0);
        }
        if Self::is_tau(left)
            && Self::is_tau(middle)
            && Self::is_tau(right)
            && Self::is_tau(coupled)
        {
            let phi = fibonacci_quantum_dim(SectorId::new(1));
            if !Self::is_tau(left_coupled) && !Self::is_tau(right_coupled) {
                Complex64::new(1.0 / phi, 0.0)
            } else if Self::is_tau(left_coupled) && Self::is_tau(right_coupled) {
                Complex64::new(-1.0 / phi, 0.0)
            } else {
                Complex64::new(1.0 / phi.sqrt(), 0.0)
            }
        } else {
            Complex64::new(1.0, 0.0)
        }
    }

    // Verbatim port of `Rsymbol` (anyons.jl:139-146): trivial braiding with
    // the vacuum, and the two complex phases `cispi(4/5)` / `cispi(-3/5)`
    // for `R^{ττ}_𝟙` / `R^{ττ}_τ`.
    fn r_symbol_scalar(&self, left: SectorId, right: SectorId, coupled: SectorId) -> Self::Scalar {
        if self.nsymbol(left, right, coupled) == 0 {
            return Complex64::new(0.0, 0.0);
        }
        if !Self::is_tau(left) || !Self::is_tau(right) {
            Complex64::new(1.0, 0.0)
        } else if !Self::is_tau(coupled) {
            Complex64::from_polar(1.0, std::f64::consts::PI * 4.0 / 5.0)
        } else {
            Complex64::from_polar(1.0, std::f64::consts::PI * -3.0 / 5.0)
        }
    }
}

impl MultiplicityFreeRigidSymbols for FibonacciFusionRule {
    fn dim_scalar(&self, sector: SectorId) -> Self::Scalar {
        Complex64::new(fibonacci_quantum_dim(sector), 0.0)
    }

    fn inv_dim_scalar(&self, sector: SectorId) -> Self::Scalar {
        Complex64::new(1.0 / fibonacci_quantum_dim(sector), 0.0)
    }

    fn sqrt_dim_scalar(&self, sector: SectorId) -> Self::Scalar {
        Complex64::new(fibonacci_quantum_dim(sector).sqrt(), 0.0)
    }

    fn inv_sqrt_dim_scalar(&self, sector: SectorId) -> Self::Scalar {
        Complex64::new(1.0 / fibonacci_quantum_dim(sector).sqrt(), 0.0)
    }

    // TensorKitSectors has no `FibonacciAnyon`-specific `twist` override, so
    // it falls back to the generic `twist_from_Rsymbol` (sectors.jl:646-647):
    // `twist(a) = Σ_{b ∈ a⊗a} dim(b)/dim(a) * Rsymbol(a,a,b)`. Verified
    // numerically against that formula (not guessed):
    //   twist(𝟙) = 1
    //   twist(τ) = (1/φ)·cispi(4/5) + (φ/φ)·cispi(-3/5) = cispi(-4/5)
    fn twist_scalar(&self, sector: SectorId) -> Self::Scalar {
        if Self::is_tau(sector) {
            Complex64::from_polar(1.0, std::f64::consts::PI * -4.0 / 5.0)
        } else {
            Complex64::new(1.0, 0.0)
        }
    }

    // TensorKitSectors has no override either, so this is the generic
    // `frobenius_schur_phase_from_Fsymbol` (sectors.jl:461-469):
    // `sign(Fsymbol(a, dual(a), a, a, leftunit(a), rightunit(a)))`, with
    // `leftunit`/`rightunit` defaulting to `unit(a)` = vacuum (sectors.jl:
    // 139-154). For `a = τ` (self-dual) that is `Fsymbol(τ,τ,τ,τ,𝟙,𝟙) =
    // +1/φ`, whose sign is `+1`; for `a = 𝟙` it is trivially `+1`.
    fn frobenius_schur_phase_scalar(&self, _sector: SectorId) -> Self::Scalar {
        Complex64::new(1.0, 0.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_complex_bits(actual: Complex64, expected: Complex64) {
        assert_eq!(actual.re.to_bits(), expected.re.to_bits());
        assert_eq!(actual.im.to_bits(), expected.im.to_bits());
    }

    #[test]
    fn fibonacci_provider_matches_tensorkitsectors() {
        let rule = FibonacciFusionRule;
        let vacuum = SectorId::new(0);
        let tau = SectorId::new(1);
        let invalid = SectorId::new(2);
        let phi = (1.0 + 5.0_f64.sqrt()) / 2.0;
        let cispi = |x: f64| Complex64::from_polar(1.0, std::f64::consts::PI * x);

        assert_eq!(rule.fusion_style(), FusionStyleKind::Simple);
        assert_eq!(rule.braiding_style(), BraidingStyleKind::Anyonic);
        assert!(!rule.has_trivial_associator_gauge());
        assert_eq!(rule.vacuum(), vacuum);
        assert_eq!(rule.fusion_channels(vacuum, tau).as_slice(), &[tau]);
        assert_eq!(rule.fusion_channels(tau, vacuum).as_slice(), &[tau]);
        assert_eq!(rule.fusion_channels(tau, tau).as_slice(), &[vacuum, tau]);
        assert_eq!(rule.nsymbol(vacuum, vacuum, tau), 0);
        assert_eq!(
            rule.try_fusion_channels(invalid, tau),
            Err(FusionAlgebraError::InvalidSector { sector: invalid })
        );

        for (left_coupled, right_coupled, expected) in [
            (vacuum, vacuum, 1.0 / phi),
            (vacuum, tau, 1.0 / phi.sqrt()),
            (tau, vacuum, 1.0 / phi.sqrt()),
            (tau, tau, -1.0 / phi),
        ] {
            assert_complex_bits(
                rule.f_symbol_scalar(tau, tau, tau, tau, left_coupled, right_coupled),
                Complex64::new(expected, 0.0),
            );
        }
        assert_complex_bits(
            rule.f_symbol_scalar(vacuum, tau, tau, tau, tau, tau),
            Complex64::new(1.0, 0.0),
        );
        assert_complex_bits(
            rule.f_symbol_scalar(tau, vacuum, tau, tau, tau, tau),
            Complex64::new(1.0, 0.0),
        );
        assert_eq!(
            rule.f_symbol_scalar(vacuum, vacuum, vacuum, tau, vacuum, vacuum),
            Complex64::new(0.0, 0.0)
        );
        assert_complex_bits(rule.r_symbol_scalar(tau, tau, vacuum), cispi(4.0 / 5.0));
        assert_complex_bits(rule.r_symbol_scalar(tau, tau, tau), cispi(-3.0 / 5.0));
        assert_complex_bits(rule.dim_scalar(tau), Complex64::new(phi, 0.0));
        assert_complex_bits(rule.twist_scalar(tau), cispi(-4.0 / 5.0));
        assert_eq!(
            rule.frobenius_schur_phase_scalar(vacuum),
            Complex64::new(1.0, 0.0)
        );
        assert_eq!(
            rule.frobenius_schur_phase_scalar(tau),
            Complex64::new(1.0, 0.0)
        );
    }
}
