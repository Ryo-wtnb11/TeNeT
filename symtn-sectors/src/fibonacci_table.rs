use std::sync::Arc;

use fusion_symbols::tables::{FibonacciTable, TableError, TableObject};
use num_complex::Complex64;

use crate::{
    BraidingStyleKind, CanonicalUnitFusionRule, CheckedFusionAlgebra, FibonacciSector,
    FusionAlgebraError, FusionRule, FusionStyleKind, MultiplicityFreeFusionRule,
    MultiplicityFreeFusionSymbols, MultiplicityFreeRigidSymbols, RuleIdentity, SectorCodec,
    SectorId, SectorVec,
};

const RANK: usize = 2;

/// Fibonacci from the pinned AnyonWiki tables of
/// [`fusion_symbols::tables::FibonacciTable`], as a symtn fusion rule.
///
/// It is the same category as the closed-form [`crate::FibonacciFusionRule`]
/// but an independent evaluation, so the two carry different
/// [`RuleIdentity`] values: recoupling data computed under one is never
/// reused for the other. The identity is built from the table's
/// `authority_fingerprint()` (provenance record plus the source bytes), so a
/// changed table, epoch or gauge is a new identity.
///
/// Sector labels are [`FibonacciSector`]s, with the closed form's ids:
/// `SectorId(i)` is the table's one-based object `TableObject(i + 1)`.
///
/// # Examples
///
/// ```
/// use symtn_sectors::{FibonacciSector, FibonacciTableRule, FusionRule, SectorCodec};
///
/// let fib = FibonacciTableRule::try_new()?;
/// let tau = fib.encode_sector(&FibonacciSector::Tau)?;
/// assert_eq!(fib.fusion_channels(tau, tau).len(), 2);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[derive(Debug)]
pub struct FibonacciTableRule {
    table: FibonacciTable,
    identity: RuleIdentity,
}

impl FibonacciTableRule {
    /// Loads and validates the embedded tables.
    ///
    /// # Errors
    ///
    /// Returns [`FibonacciTableError`] if the embedded data do not parse or
    /// violate an exact `N` axiom; for the shipped data this cannot happen.
    pub fn try_new() -> Result<Self, FibonacciTableError> {
        let table = FibonacciTable::try_new().map_err(FibonacciTableError)?;
        let bytes = table.authority_fingerprint();
        let identity = RuleIdentity::from_canonical_bytes::<Self>(fnv1a64(bytes), Arc::from(bytes));
        Ok(Self { table, identity })
    }

    fn index(sector: SectorId) -> Result<TableObject, FusionAlgebraError> {
        (sector.id() < RANK)
            .then(|| TableObject(sector.id() as u16 + 1))
            .ok_or(FusionAlgebraError::InvalidSector { sector })
    }

    /// The infallible symbols' domain check (#1737): an out-of-domain id
    /// panics with the typed error instead of reaching the table.
    fn object(method: &str, sector: SectorId) -> TableObject {
        Self::index(sector)
            .unwrap_or_else(|error| panic!("{method} precondition violated: {error}"))
    }

    fn sector_id(object: TableObject) -> SectorId {
        SectorId::new(usize::from(object.label()) - 1)
    }
}

/// Failure to load the embedded Fibonacci tables.
///
/// Opaque: the table's own error is reachable only through
/// [`std::error::Error::source`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FibonacciTableError(TableError);

impl std::fmt::Display for FibonacciTableError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("Fibonacci table authority failed to load")
    }
}

impl std::error::Error for FibonacciTableError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(&self.0)
    }
}

/// FNV-1a over 64 bits, the prehash `RuleIdentity` carries beside the bytes.
///
/// Why not `DefaultHasher`: its output is not stable across Rust releases.
/// The prehash is only a bucket key; equality compares the full bytes.
fn fnv1a64(bytes: &[u8]) -> u64 {
    bytes.iter().fold(0xcbf2_9ce4_8422_2325, |hash, &byte| {
        (hash ^ u64::from(byte)).wrapping_mul(0x0000_0100_0000_01b3)
    })
}

impl FusionRule for FibonacciTableRule {
    fn rule_identity(&self) -> RuleIdentity {
        self.identity.clone()
    }

    fn fusion_style(&self) -> FusionStyleKind {
        FusionStyleKind::Simple
    }

    fn braiding_style(&self) -> BraidingStyleKind {
        BraidingStyleKind::Anyonic
    }

    fn vacuum(&self) -> SectorId {
        Self::sector_id(self.table.unit())
    }

    // Why not `supports_unitary_braid_dagger() == true`: construction
    // validates the exact `N` axioms only; R unitarity is a consistency test
    // of the table, not a property the constructor proved.

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

impl CheckedFusionAlgebra for FibonacciTableRule {
    fn try_dual_sector(&self, sector: SectorId) -> Result<SectorId, FusionAlgebraError> {
        Ok(Self::sector_id(self.table.dual(Self::index(sector)?)))
    }

    fn try_fusion_channels(
        &self,
        left: SectorId,
        right: SectorId,
    ) -> Result<SectorVec, FusionAlgebraError> {
        let (left, right) = (Self::index(left)?, Self::index(right)?);
        Ok(self
            .table
            .fusion(left, right)
            .map(Self::sector_id)
            .collect())
    }

    fn try_nsymbol(
        &self,
        left: SectorId,
        right: SectorId,
        coupled: SectorId,
    ) -> Result<usize, FusionAlgebraError> {
        let (left, right, coupled) = (
            Self::index(left)?,
            Self::index(right)?,
            Self::index(coupled)?,
        );
        Ok(self.table.nsymbol(left, right, coupled) as usize)
    }
}

impl SectorCodec for FibonacciTableRule {
    type Sector = FibonacciSector;

    fn encode_sector(&self, value: &Self::Sector) -> Result<SectorId, FusionAlgebraError> {
        Ok(value.sector_id())
    }

    fn decode_sector(&self, id: SectorId) -> Result<Self::Sector, FusionAlgebraError> {
        FibonacciSector::from_sector_id(id)
    }
}

impl MultiplicityFreeFusionRule for FibonacciTableRule {}

// The unit fusion laws are checked exactly at construction; the unitor
// associators are asserted bitwise `1 + 0i` by the table's own tests and by
// `tests/fibonacci_providers.rs`.
impl CanonicalUnitFusionRule for FibonacciTableRule {}

impl MultiplicityFreeFusionSymbols for FibonacciTableRule {
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
        let o = |sector| Self::object("f_symbol_scalar", sector);
        self.table.f_symbol(
            o(left),
            o(middle),
            o(right),
            o(coupled),
            o(left_coupled),
            o(right_coupled),
        )
    }

    fn r_symbol_scalar(&self, left: SectorId, right: SectorId, coupled: SectorId) -> Self::Scalar {
        let o = |sector| Self::object("r_symbol_scalar", sector);
        self.table.r_symbol(o(left), o(right), o(coupled))
    }
}

impl MultiplicityFreeRigidSymbols for FibonacciTableRule {
    fn dim_scalar(&self, sector: SectorId) -> Self::Scalar {
        Complex64::new(self.table.dim(Self::object("dim_scalar", sector)), 0.0)
    }

    fn inv_dim_scalar(&self, sector: SectorId) -> Self::Scalar {
        Complex64::new(
            1.0 / self.table.dim(Self::object("inv_dim_scalar", sector)),
            0.0,
        )
    }

    fn sqrt_dim_scalar(&self, sector: SectorId) -> Self::Scalar {
        Complex64::new(
            self.table
                .dim(Self::object("sqrt_dim_scalar", sector))
                .sqrt(),
            0.0,
        )
    }

    fn inv_sqrt_dim_scalar(&self, sector: SectorId) -> Self::Scalar {
        Complex64::new(
            1.0 / self
                .table
                .dim(Self::object("inv_sqrt_dim_scalar", sector))
                .sqrt(),
            0.0,
        )
    }

    fn twist_scalar(&self, sector: SectorId) -> Self::Scalar {
        self.table.twist(Self::object("twist_scalar", sector))
    }

    fn frobenius_schur_phase_scalar(&self, sector: SectorId) -> Self::Scalar {
        self.table
            .frobenius_schur(Self::object("frobenius_schur_phase_scalar", sector))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fnv1a64_matches_the_published_reference_vectors() {
        assert_eq!(fnv1a64(b""), 0xcbf2_9ce4_8422_2325);
        assert_eq!(fnv1a64(b"a"), 0xaf63_dc4c_8601_ec8c);
        assert_eq!(fnv1a64(b"foobar"), 0x8594_4171_f739_67e8);
    }

    #[test]
    fn identity_is_bound_to_the_table_fingerprint() {
        // The prehash fusion-symbols 0.3.0 pins for these exact bytes
        // (`tables::tests::FINGERPRINT_PREHASH`). It replaces the old
        // `symtn-category-data` identity (prehash 0xee32_d986_edef_625b):
        // caches keyed on that identity miss once.
        const FINGERPRINT_PREHASH: u64 = 0x52e5_b6dc_1a15_96c5;
        let fib = FibonacciTableRule::try_new().expect("embedded tables are valid");
        let bytes = fib.table.authority_fingerprint();
        assert_eq!(fnv1a64(bytes), FINGERPRINT_PREHASH);
        assert_eq!(
            fib.rule_identity(),
            RuleIdentity::from_canonical_bytes::<FibonacciTableRule>(
                FINGERPRINT_PREHASH,
                Arc::from(bytes),
            )
        );
        assert_eq!(
            fib.rule_identity(),
            FibonacciTableRule::try_new().unwrap().rule_identity()
        );
        assert_ne!(
            fib.rule_identity(),
            crate::FibonacciFusionRule.rule_identity()
        );
    }

    #[test]
    fn table_error_is_opaque_with_the_authority_as_source() {
        use std::error::Error;
        let inner = TableError::UnknownObject {
            object: TableObject(3),
        };
        let error = FibonacciTableError(inner.clone());
        assert_eq!(
            error.to_string(),
            "Fibonacci table authority failed to load"
        );
        assert_eq!(error.source().unwrap().to_string(), inner.to_string());
    }

    #[test]
    fn embedded_tables_load_with_the_expected_structure() {
        let fib = FibonacciTableRule::try_new().expect("embedded tables are valid");
        assert_eq!(fib.vacuum(), SectorId::new(0));
        assert_eq!(fib.fusion_style(), FusionStyleKind::Simple);
        assert_eq!(fib.braiding_style(), BraidingStyleKind::Anyonic);
        assert!(!fib.supports_unitary_braid_dagger());
        assert!(!fib.has_trivial_associator_gauge());
    }
}
