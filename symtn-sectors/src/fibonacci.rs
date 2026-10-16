use fusion_symbols::fibonacci::{fibonacci_f_symbol, fibonacci_r_symbol, FibonacciObject};
use num_complex::Complex64;

use crate::{
    BraidingStyleKind, CanonicalUnitFusionRule, CheckedFusionAlgebra, FusionAlgebraError,
    FusionRule, FusionStyleKind, MultiplicityFreeFusionRule, MultiplicityFreeFusionSymbols,
    MultiplicityFreeRigidSymbols, RuleIdentity, SectorCodec, SectorId, SectorVec,
};

/// The Fibonacci category in closed form, `τ ⊗ τ = 𝟙 ⊕ τ`, as a symtn fusion
/// rule: `SectorId` 0 is the vacuum `𝟙`, `SectorId` 1 is `τ`.
///
/// Every value comes from [`fusion_symbols::fibonacci`], which follows
/// TensorKitSectors' `FibonacciAnyon`; this type only maps `SectorId`s to
/// `FibonacciObject`s and widens the real F, dimension and
/// Frobenius–Schur values to `Complex64` (symtn's one `Scalar`).
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Hash)]
pub struct FibonacciFusionRule;

/// A simple object of the Fibonacci fusion category, the semantic label of
/// both [`FibonacciFusionRule`] and [`crate::FibonacciTableRule`].
///
/// Why not re-export `fusion_symbols::fibonacci::FibonacciObject`: symtn's
/// public label types stay symtn's own (as `SU2Irrep` does), so a
/// symbol-layer release cannot change symtn's API.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub enum FibonacciSector {
    /// The tensor unit `𝟙`.
    Vacuum,
    /// The nontrivial Fibonacci anyon `τ`.
    Tau,
}

impl FibonacciSector {
    pub(crate) fn from_sector_id(sector: SectorId) -> Result<Self, FusionAlgebraError> {
        match sector.id() {
            0 => Ok(Self::Vacuum),
            1 => Ok(Self::Tau),
            _ => Err(FusionAlgebraError::InvalidSector { sector }),
        }
    }

    pub(crate) fn sector_id(self) -> SectorId {
        SectorId::new(match self {
            Self::Vacuum => 0,
            Self::Tau => 1,
        })
    }
}

fn checked_object(sector: SectorId) -> Result<FibonacciObject, FusionAlgebraError> {
    Ok(match FibonacciSector::from_sector_id(sector)? {
        FibonacciSector::Vacuum => FibonacciObject::Vacuum,
        FibonacciSector::Tau => FibonacciObject::Tau,
    })
}

/// The infallible symbols' domain check (#1737): an out-of-domain id panics
/// with the typed error instead of aliasing `τ`.
fn object(method: &str, sector: SectorId) -> FibonacciObject {
    checked_object(sector).unwrap_or_else(|error| panic!("{method} precondition violated: {error}"))
}

fn sector_id(object: FibonacciObject) -> SectorId {
    SectorId::new(match object {
        FibonacciObject::Vacuum => 0,
        FibonacciObject::Tau => 1,
    })
}

fn real(value: f64) -> Complex64 {
    Complex64::new(value, 0.0)
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

    fn dual(&self, sector: SectorId) -> SectorId {
        self.dual_or_panic(sector)
    }

    fn fusion_channels(&self, left: SectorId, right: SectorId) -> SectorVec {
        self.fusion_channels_or_panic(left, right)
    }

    fn nsymbol(&self, left: SectorId, right: SectorId, coupled: SectorId) -> usize {
        self.nsymbol_or_panic(left, right, coupled)
    }
}

impl CheckedFusionAlgebra for FibonacciFusionRule {
    fn try_dual_sector(&self, sector: SectorId) -> Result<SectorId, FusionAlgebraError> {
        Ok(sector_id(checked_object(sector)?.dual()))
    }

    fn try_fusion_channels(
        &self,
        left: SectorId,
        right: SectorId,
    ) -> Result<SectorVec, FusionAlgebraError> {
        let channels = checked_object(left)?.fusion(checked_object(right)?);
        Ok(channels.iter().copied().map(sector_id).collect())
    }

    fn try_nsymbol(
        &self,
        left: SectorId,
        right: SectorId,
        coupled: SectorId,
    ) -> Result<usize, FusionAlgebraError> {
        let channels = checked_object(left)?.fusion(checked_object(right)?);
        Ok(usize::from(channels.contains(&checked_object(coupled)?)))
    }
}

impl SectorCodec for FibonacciFusionRule {
    type Sector = FibonacciSector;

    fn encode_sector(&self, value: &Self::Sector) -> Result<SectorId, FusionAlgebraError> {
        Ok(value.sector_id())
    }

    fn decode_sector(&self, id: SectorId) -> Result<Self::Sector, FusionAlgebraError> {
        FibonacciSector::from_sector_id(id)
    }
}

impl MultiplicityFreeFusionRule for FibonacciFusionRule {}

impl CanonicalUnitFusionRule for FibonacciFusionRule {}

impl MultiplicityFreeFusionSymbols for FibonacciFusionRule {
    type Scalar = Complex64;

    fn f_symbol_scalar(
        &self,
        left: SectorId,
        middle: SectorId,
        right: SectorId,
        coupled: SectorId,
        left_coupled: SectorId,
        right_coupled: SectorId,
    ) -> Self::Scalar {
        let o = |sector| object("f_symbol_scalar", sector);
        real(fibonacci_f_symbol(
            o(left),
            o(middle),
            o(right),
            o(coupled),
            o(left_coupled),
            o(right_coupled),
        ))
    }

    fn r_symbol_scalar(&self, left: SectorId, right: SectorId, coupled: SectorId) -> Self::Scalar {
        let o = |sector| object("r_symbol_scalar", sector);
        fibonacci_r_symbol(o(left), o(right), o(coupled))
    }
}

impl MultiplicityFreeRigidSymbols for FibonacciFusionRule {
    fn dim_scalar(&self, sector: SectorId) -> Self::Scalar {
        real(object("dim_scalar", sector).dim())
    }

    fn inv_dim_scalar(&self, sector: SectorId) -> Self::Scalar {
        real(1.0 / object("inv_dim_scalar", sector).dim())
    }

    fn sqrt_dim_scalar(&self, sector: SectorId) -> Self::Scalar {
        real(object("sqrt_dim_scalar", sector).dim().sqrt())
    }

    fn inv_sqrt_dim_scalar(&self, sector: SectorId) -> Self::Scalar {
        real(1.0 / object("inv_sqrt_dim_scalar", sector).dim().sqrt())
    }

    fn twist_scalar(&self, sector: SectorId) -> Self::Scalar {
        object("twist_scalar", sector).twist()
    }

    fn frobenius_schur_phase_scalar(&self, sector: SectorId) -> Self::Scalar {
        real(object("frobenius_schur_phase_scalar", sector).frobenius_schur())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_complex_bits(actual: Complex64, expected: Complex64) {
        assert_eq!(actual.re.to_bits(), expected.re.to_bits());
        assert_eq!(actual.im.to_bits(), expected.im.to_bits());
    }

    /// Closed-form irrational values (`1/φ`, `1/√φ`, `φ`, `cispi`) admit more
    /// than one valid evaluation order, so they meet the TensorKitSectors
    /// closed forms under the workspace tolerance rule, not bit for bit.
    fn assert_closed_form(what: &str, actual: Complex64, expected: Complex64) {
        crate::test_numerics::numerics::assert_close(what, actual, expected, 1);
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
            assert_closed_form(
                "F(τ,τ,τ,τ)",
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
        assert_closed_form(
            "R(τ,τ,1)",
            rule.r_symbol_scalar(tau, tau, vacuum),
            cispi(4.0 / 5.0),
        );
        assert_closed_form(
            "R(τ,τ,τ)",
            rule.r_symbol_scalar(tau, tau, tau),
            cispi(-3.0 / 5.0),
        );
        assert_closed_form("dim(τ)", rule.dim_scalar(tau), Complex64::new(phi, 0.0));
        assert_closed_form("twist(τ)", rule.twist_scalar(tau), cispi(-4.0 / 5.0));
        assert_eq!(
            rule.frobenius_schur_phase_scalar(vacuum),
            Complex64::new(1.0, 0.0)
        );
        assert_eq!(
            rule.frobenius_schur_phase_scalar(tau),
            Complex64::new(1.0, 0.0)
        );
    }

    #[test]
    fn fibonacci_has_canonical_unit_channels_and_associator() {
        // What: Fibonacci's vacuum is a unique self-dual identity even though
        // its non-unit fusion has multiple channels.
        fn accepts_canonical_unit<R: CanonicalUnitFusionRule>(_rule: &R) {}

        let rule = FibonacciFusionRule;
        let vacuum = rule.vacuum();
        let tau = SectorId::new(1);
        accepts_canonical_unit(&rule);
        assert_eq!(rule.dual(vacuum), vacuum);
        assert_eq!(rule.fusion_channels(vacuum, tau).as_slice(), &[tau]);
        assert_eq!(rule.fusion_channels(tau, vacuum).as_slice(), &[tau]);
        assert_eq!(rule.nsymbol(vacuum, tau, tau), 1);
        assert_eq!(rule.nsymbol(tau, vacuum, tau), 1);
        assert_eq!(rule.nsymbol(vacuum, tau, vacuum), 0);
        for fused in rule.fusion_channels(tau, tau) {
            assert_complex_bits(
                rule.f_symbol_scalar(vacuum, tau, tau, fused, tau, fused),
                Complex64::new(1.0, 0.0),
            );
            assert_complex_bits(
                rule.f_symbol_scalar(tau, vacuum, tau, fused, tau, tau),
                Complex64::new(1.0, 0.0),
            );
            assert_complex_bits(
                rule.f_symbol_scalar(tau, tau, vacuum, fused, fused, tau),
                Complex64::new(1.0, 0.0),
            );
        }
    }

    #[test]
    fn fibonacci_codec_round_trips_the_reachable_algebra() {
        // What: the semantic labels are a lossless, non-aliasing boundary for
        // the vacuum and every sector produced by duality and fusion.
        let rule = FibonacciFusionRule;
        let vacuum = rule.encode_sector(&FibonacciSector::Vacuum).unwrap();
        let tau = rule.encode_sector(&FibonacciSector::Tau).unwrap();

        assert_ne!(vacuum, tau);
        assert_eq!(rule.decode_sector(vacuum), Ok(FibonacciSector::Vacuum));
        assert_eq!(rule.decode_sector(tau), Ok(FibonacciSector::Tau));
        assert_eq!(
            rule.decode_sector(rule.vacuum()),
            Ok(FibonacciSector::Vacuum)
        );

        for sector in [vacuum, tau] {
            let label = rule.decode_sector(sector).unwrap();
            assert_eq!(rule.encode_sector(&label), Ok(sector));
            let dual = rule.dual(sector);
            let dual_label = rule.decode_sector(dual).unwrap();
            assert_eq!(rule.encode_sector(&dual_label), Ok(dual));
        }
        for left in [vacuum, tau] {
            for right in [vacuum, tau] {
                for coupled in rule.fusion_channels(left, right) {
                    let label = rule.decode_sector(coupled).unwrap();
                    assert_eq!(rule.encode_sector(&label), Ok(coupled));
                }
            }
        }

        let invalid = SectorId::new(2);
        assert_eq!(
            rule.decode_sector(invalid),
            Err(FusionAlgebraError::InvalidSector { sector: invalid })
        );
    }
}
