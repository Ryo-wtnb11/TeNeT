//! The CategoryData Fibonacci provider.

use std::sync::Arc;

use num_complex::Complex64;
use smallvec::SmallVec;
use tenet_sectors::{
    BraidingStyleKind, CanonicalUnitFusionRule, CheckedFusionAlgebra, FusionAlgebraError,
    FusionRule, FusionStyleKind, MultiplicityFreeFusionRule, MultiplicityFreeFusionSymbols,
    MultiplicityFreeRigidSymbols, RuleIdentity, SectorCodec, SectorId, SectorVec,
};

use crate::provenance::{fnv1a64, CategoryProvenance, PROJECTION_EPOCH};
use crate::table::{CategoryDataError, CategoryObject, MultiplicityFreeTable};

const RANK: usize = 2;

// The three pinned files, embedded verbatim under their upstream paths. They
// are the identity-bearing bytes, so they are `include_str!`-ed rather than
// re-encoded: any change to them, deliberate or accidental, changes
// `rule_identity`.
const N_TEXT: &str = include_str!("../data/categorydata-v0.1.3/Nsymbols/FR_2_1_0_2.txt");
const F_TEXT: &str = include_str!("../data/categorydata-v0.1.3/Fsymbols/FR_2_1_0_2_0.txt");
const R_TEXT: &str = include_str!("../data/categorydata-v0.1.3/Rsymbols/FR_2_1_0_2_0_0.txt");

/// Fibonacci as distributed by CategoryData.jl, not as evaluated in closed form.
///
/// This provider answers exactly the questions `tenet_sectors::FibonacciFusionRule`
/// answers, from the pinned decimal tables instead of from formulas. The two
/// are mathematically the same category and agree to a few ULP, but they are
/// *not* byte-identical and deliberately carry different
/// [`RuleIdentity`] values — see the crate documentation.
///
/// # Examples
///
/// ```
/// use tenet_category_data::{CategoryDataFibonacci, CategoryObject};
/// use tenet_sectors::{FusionRule, SectorCodec};
///
/// let fib = CategoryDataFibonacci::try_new()?;
/// let tau = fib.encode_sector(&CategoryObject(2))?;
/// assert_eq!(fib.fusion_channels(tau, tau).len(), 2);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[derive(Debug)]
pub struct CategoryDataFibonacci {
    table: MultiplicityFreeTable,
    // Derived once at construction: O(rank^2) values that the engine asks for
    // per recoupling step. Deriving them is what TensorKitSectors does too;
    // caching them here is initialisation, not a cache layer with a policy.
    dims: Vec<f64>,
    frobenius_schur: Vec<Complex64>,
    twists: Vec<Complex64>,
    identity: RuleIdentity,
}

impl CategoryDataFibonacci {
    /// The frozen lineage of this provider's tables.
    pub const PROVENANCE: CategoryProvenance = CategoryProvenance {
        exact_source_repository: "https://github.com/JCBridgeman/smallRankUnitaryFusionData",
        exact_source_commit: "a6ba1c308ddcecbb03b9651a25fa0d1afcfb7546",
        exact_source_export: "exportData.nb: N[expr, 25] then DecimalForm[.., {Infinity, 20}]",
        distribution_repository: "https://github.com/lkdvos/CategoryData.jl",
        distribution_commit: "e793a08a093f6ba890a32c7e57d8e8b347441058",
        distribution_version: "0.3.6",
        artifact_tag: "data-v0.1.3",
        artifact_git_tree_sha1: "90a414867a392b7b811ba6c8092e5e1a848dc4c4",
        artifact_tarball_sha256: "586f48411e0c3aa8780959d2395724bf6d1de4e4bcf785eb84864e0d5c045cf6",
        artifact_url:
            "https://github.com/lkdvos/CategoryData.jl/archive/refs/tags/data-v0.1.3.tar.gz",
        category_key: "Fib = PMFC{2,1,0,2,0,0}",
        object_order: "source file order, one-based: 1 = vacuum, 2 = tau",
        gauge: "source gauge verbatim; only the one-based to zero-based label shift is applied",
        projection_epoch: PROJECTION_EPOCH,
    };

    /// Parses, validates and publishes the provider.
    ///
    /// There is no unchecked constructor: a provider that skipped the import
    /// checks would be indistinguishable from a correct one at every later
    /// call site.
    ///
    /// # Errors
    ///
    /// Returns [`CategoryDataError`] if the embedded tables do not parse or
    /// violate an exact `N` axiom. For the shipped data this cannot happen;
    /// the constructor is fallible because it is the import boundary, and the
    /// crate's tests are what hold that invariant.
    pub fn try_new() -> Result<Self, CategoryDataError> {
        let table = MultiplicityFreeTable::parse(RANK, N_TEXT, F_TEXT, R_TEXT)?;

        // Derived exactly as TensorKitSectors 0.3.9 derives them, from F and R
        // only. See references.md, "Derived quantities", for the per-formula
        // source lines. These are API data, not a second source of authority:
        // the tables above remain the only input.
        let unit_f = |a: usize| table.fsymbol(a, table.dual(a), a, a, 0, 0);
        let dims: Vec<f64> = (0..RANK)
            .map(|a| (Complex64::new(1.0, 0.0) / unit_f(a)).norm())
            .collect();
        let frobenius_schur: Vec<Complex64> = (0..RANK)
            .map(|a| {
                let value = unit_f(a);
                value / value.norm()
            })
            .collect();
        let twists: Vec<Complex64> = (0..RANK)
            .map(|a| {
                table
                    .channels(a, a)
                    .map(|b| Complex64::new(dims[b] / dims[a], 0.0) * table.rsymbol(a, a, b))
                    .fold(Complex64::new(0.0, 0.0), |sum, term| sum + term)
            })
            .collect();

        let mut bytes = Vec::new();
        Self::PROVENANCE.write_canonical_bytes(&mut bytes);
        // The projected coefficients are reproducible from these bytes under
        // the recorded epoch, so the source text is what the identity retains
        // — not a re-serialisation of the parsed f64s, which would hide a
        // parser change behind identical output.
        for (tag, text) in [
            ("Nsymbols", N_TEXT),
            ("Fsymbols", F_TEXT),
            ("Rsymbols", R_TEXT),
        ] {
            bytes.extend_from_slice(b"--- ");
            bytes.extend_from_slice(tag.as_bytes());
            bytes.push(b'\n');
            bytes.extend_from_slice(text.as_bytes());
        }
        let identity =
            RuleIdentity::from_canonical_bytes::<Self>(fnv1a64(&bytes), Arc::from(bytes));

        Ok(Self {
            table,
            dims,
            frobenius_schur,
            twists,
            identity,
        })
    }

    /// The category's objects in the frozen source order.
    pub fn objects(&self) -> impl Iterator<Item = CategoryObject> {
        (1..=self.table.rank() as u16).map(CategoryObject)
    }

    #[inline]
    fn index(sector: SectorId) -> Result<usize, FusionAlgebraError> {
        (sector.id() < RANK)
            .then_some(sector.id())
            .ok_or(FusionAlgebraError::InvalidSector { sector })
    }
}

impl FusionRule for CategoryDataFibonacci {
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
        SectorId::new(0)
    }

    // Why not `supports_unitary_braid_dagger() == true`: construction here
    // validates the exact `N` axioms only. R unitarity is a *consistency*
    // check living in this crate's tests, per the issue's data-authority
    // correction, so the constructor has not proved the property this flag
    // asserts. The closed-form provider reports the same `false`.

    fn dual(&self, sector: SectorId) -> SectorId {
        SectorId::new(self.table.dual(sector.id()))
    }

    fn fusion_channels(&self, left: SectorId, right: SectorId) -> SectorVec {
        self.table
            .channels(left.id(), right.id())
            .map(SectorId::new)
            .collect::<SmallVec<_>>()
    }

    fn nsymbol(&self, left: SectorId, right: SectorId, coupled: SectorId) -> usize {
        self.table.nsymbol(left.id(), right.id(), coupled.id()) as usize
    }
}

impl CheckedFusionAlgebra for CategoryDataFibonacci {
    fn try_dual_sector(&self, sector: SectorId) -> Result<SectorId, FusionAlgebraError> {
        Self::index(sector)?;
        Ok(self.dual(sector))
    }

    fn try_fusion_channels(
        &self,
        left: SectorId,
        right: SectorId,
    ) -> Result<SectorVec, FusionAlgebraError> {
        Self::index(left)?;
        Self::index(right)?;
        Ok(self.fusion_channels(left, right))
    }

    fn try_nsymbol(
        &self,
        left: SectorId,
        right: SectorId,
        coupled: SectorId,
    ) -> Result<usize, FusionAlgebraError> {
        Self::index(left)?;
        Self::index(right)?;
        Self::index(coupled)?;
        Ok(self.nsymbol(left, right, coupled))
    }
}

impl SectorCodec for CategoryDataFibonacci {
    type Sector = CategoryObject;

    fn encode_sector(&self, value: &Self::Sector) -> Result<SectorId, FusionAlgebraError> {
        if value.label() >= 1 && usize::from(value.label()) <= RANK {
            Ok(SectorId::new(usize::from(value.label()) - 1))
        } else {
            Err(FusionAlgebraError::UnrepresentableSectorLabel {
                rule: self.identity.clone(),
                label: value.to_string(),
            })
        }
    }

    fn decode_sector(&self, id: SectorId) -> Result<Self::Sector, FusionAlgebraError> {
        Self::index(id).map(|index| CategoryObject(index as u16 + 1))
    }
}

impl MultiplicityFreeFusionRule for CategoryDataFibonacci {}

// The unit laws this marker promises are checked exactly at construction
// (`MultiplicityFreeTable::check_unit_laws`), and the unit-gauge triangle
// identities are covered by the crate's consistency tests.
impl CanonicalUnitFusionRule for CategoryDataFibonacci {}

impl MultiplicityFreeFusionSymbols for CategoryDataFibonacci {
    type Scalar = Complex64;

    fn scalar_one(&self) -> Self::Scalar {
        Complex64::new(1.0, 0.0)
    }

    fn scalar_conj(&self, value: Self::Scalar) -> Self::Scalar {
        value.conj()
    }

    // Left at the `false` default: Fibonacci's associator is genuinely
    // non-trivial, so the gauge-general Artin path is the correct one.

    fn f_symbol_scalar(
        &self,
        left: SectorId,
        middle: SectorId,
        right: SectorId,
        coupled: SectorId,
        left_coupled: SectorId,
        right_coupled: SectorId,
    ) -> Self::Scalar {
        self.table.fsymbol(
            left.id(),
            middle.id(),
            right.id(),
            coupled.id(),
            left_coupled.id(),
            right_coupled.id(),
        )
    }

    fn r_symbol_scalar(&self, left: SectorId, right: SectorId, coupled: SectorId) -> Self::Scalar {
        self.table.rsymbol(left.id(), right.id(), coupled.id())
    }
}

impl MultiplicityFreeRigidSymbols for CategoryDataFibonacci {
    fn dim_scalar(&self, sector: SectorId) -> Self::Scalar {
        Complex64::new(self.dims[sector.id()], 0.0)
    }

    fn inv_dim_scalar(&self, sector: SectorId) -> Self::Scalar {
        Complex64::new(1.0 / self.dims[sector.id()], 0.0)
    }

    fn sqrt_dim_scalar(&self, sector: SectorId) -> Self::Scalar {
        Complex64::new(self.dims[sector.id()].sqrt(), 0.0)
    }

    fn inv_sqrt_dim_scalar(&self, sector: SectorId) -> Self::Scalar {
        Complex64::new(1.0 / self.dims[sector.id()].sqrt(), 0.0)
    }

    fn twist_scalar(&self, sector: SectorId) -> Self::Scalar {
        self.twists[sector.id()]
    }

    fn frobenius_schur_phase_scalar(&self, sector: SectorId) -> Self::Scalar {
        self.frobenius_schur[sector.id()]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn embedded_tables_load_and_expose_the_frozen_ordering() {
        let fib = CategoryDataFibonacci::try_new().expect("embedded tables are valid");
        assert_eq!(
            fib.objects().collect::<Vec<_>>(),
            vec![CategoryObject(1), CategoryObject(2)]
        );
        assert_eq!(fib.vacuum(), SectorId::new(0));
        assert_eq!(fib.fusion_style(), FusionStyleKind::Simple);
        assert_eq!(fib.braiding_style(), BraidingStyleKind::Anyonic);
        assert!(!fib.supports_unitary_braid_dagger());
        assert!(!fib.has_trivial_associator_gauge());
    }

    #[test]
    fn identity_is_reproducible_and_content_bound() {
        // Two independent loads of the same pinned bytes are one identity.
        let first = CategoryDataFibonacci::try_new().expect("valid");
        let second = CategoryDataFibonacci::try_new().expect("valid");
        assert_eq!(first.rule_identity(), second.rule_identity());

        // A different projection epoch is a different category identity.
        let mut baseline = Vec::new();
        CategoryDataFibonacci::PROVENANCE.write_canonical_bytes(&mut baseline);
        let mut bumped = Vec::new();
        CategoryProvenance {
            projection_epoch: "other",
            ..CategoryDataFibonacci::PROVENANCE
        }
        .write_canonical_bytes(&mut bumped);
        assert_ne!(baseline, bumped);
        assert_ne!(
            RuleIdentity::from_canonical_bytes::<CategoryDataFibonacci>(
                fnv1a64(&baseline),
                Arc::from(baseline),
            ),
            RuleIdentity::from_canonical_bytes::<CategoryDataFibonacci>(
                fnv1a64(&bumped),
                Arc::from(bumped),
            )
        );
    }

    /// The exact byte composition `try_new` hashes, with `F` swappable.
    fn identity_bytes(f_text: &str) -> Vec<u8> {
        let mut bytes = Vec::new();
        CategoryDataFibonacci::PROVENANCE.write_canonical_bytes(&mut bytes);
        for (tag, text) in [
            ("Nsymbols", N_TEXT),
            ("Fsymbols", f_text),
            ("Rsymbols", R_TEXT),
        ] {
            bytes.extend_from_slice(b"--- ");
            bytes.extend_from_slice(tag.as_bytes());
            bytes.push(b'\n');
            bytes.extend_from_slice(text.as_bytes());
        }
        bytes
    }

    fn identity_of(f_text: &str) -> RuleIdentity {
        let bytes = identity_bytes(f_text);
        RuleIdentity::from_canonical_bytes::<CategoryDataFibonacci>(
            fnv1a64(&bytes),
            Arc::from(bytes),
        )
    }

    // Golden digests of the three shipped files and of the identity they
    // produce. This is the only test that can fail on a *tamper*: every other
    // test in the crate reads the shipped bytes and would happily agree with
    // an edited table whose values still round to the same `f64`. Such an edit
    // silently moves `rule_identity` and desynchronises the SHA-256s recorded
    // in `references.md`, which is exactly the TODO-1 acceptance criterion.
    //
    // Updating a constant here is therefore a deliberate act: it means the
    // pinned source changed, and `references.md` must change with it.
    const N_TEXT_FNV: u64 = 0xaba7_7aa3_3176_9df0;
    const F_TEXT_FNV: u64 = 0xbc12_a51a_3729_3fdf;
    const R_TEXT_FNV: u64 = 0x2c13_49ae_3a17_11fe;
    const IDENTITY_PREHASH: u64 = 0xee32_d986_edef_625b;

    /// FNV-1a of `Rsymbols/FR_2_1_0_2_0_1.txt` from the same pinned artifact:
    /// the *mirror* braiding of this category. Recorded rather than shipped
    /// because the crate must not carry data it does not use. Asserting it
    /// differs from `R_TEXT_FNV` pins the braid-index-0 choice: swapping the
    /// mirror file in fails `shipped_table_bytes_are_pinned` below, and
    /// `identity_moves_on_a_source_edit_too_small_to_move_a_coefficient`
    /// separately establishes that different table bytes give a different
    /// identity.
    const MIRROR_BRAID_R_FNV: u64 = 0x7c19_0b00_80ec_d4d6;

    #[test]
    fn shipped_table_bytes_are_pinned() {
        assert_eq!(
            fnv1a64(N_TEXT.as_bytes()),
            N_TEXT_FNV,
            "Nsymbols file changed"
        );
        assert_eq!(
            fnv1a64(F_TEXT.as_bytes()),
            F_TEXT_FNV,
            "Fsymbols file changed"
        );
        assert_eq!(
            fnv1a64(R_TEXT.as_bytes()),
            R_TEXT_FNV,
            "Rsymbols file changed"
        );
        assert_ne!(
            fnv1a64(R_TEXT.as_bytes()),
            MIRROR_BRAID_R_FNV,
            "the mirror braiding was loaded instead of braid index 0"
        );

        // The identity is a pure function of the provenance bytes and these
        // three files, so pinning its prehash pins the whole composition.
        let provider = CategoryDataFibonacci::try_new().expect("valid");
        assert_eq!(
            provider.rule_identity(),
            RuleIdentity::from_canonical_bytes::<CategoryDataFibonacci>(
                IDENTITY_PREHASH,
                Arc::from(identity_bytes(F_TEXT)),
            )
        );
    }

    #[test]
    fn identity_moves_on_a_source_edit_too_small_to_move_a_coefficient() {
        // A change in the 21st decimal of a table entry rounds to the same
        // `f64`, so no coefficient observably changes. The identity must move
        // anyway: it is bound to the source bytes, which is what makes the
        // acceptance criterion "changing the source changes the identity" hold
        // below the projection's resolution as well as above it.
        // Appends one trailing zero digit to every decimal literal, e.g.
        // `0.786…607` -> `0.786…6070`. Value-preserving by construction and
        // below binary64 resolution, with no hard-coded literal to rot.
        let edited: String = F_TEXT
            .lines()
            .map(|line| {
                let fields: Vec<String> = line
                    .split_whitespace()
                    .map(|field| {
                        if field.contains('.') {
                            format!("{field}0")
                        } else {
                            field.to_owned()
                        }
                    })
                    .collect();
                format!("{}\n", fields.join(" "))
            })
            .collect();
        assert_ne!(edited, F_TEXT, "the edit did not apply");

        // Every parsed field, not one sampled field: this is the claim that
        // the edit is invisible to the projection, so it has to be checked
        // over the whole table.
        let fields = |text: &str| {
            text.split_whitespace()
                .map(|field| field.parse::<f64>().expect("numeric field").to_bits())
                .collect::<Vec<_>>()
        };
        assert_eq!(
            fields(&edited),
            fields(F_TEXT),
            "the edit moved a coefficient"
        );

        let provider = CategoryDataFibonacci::try_new().expect("valid");
        assert_eq!(identity_of(F_TEXT), provider.rule_identity());
        assert_ne!(identity_of(&edited), provider.rule_identity());
    }
}
