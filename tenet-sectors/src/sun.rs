use std::fmt;
use std::sync::Arc;

use num_traits::ToPrimitive;

use crate::{
    BraidingStyleKind, CheckedGenericFusion, CheckedGenericRigidSymbols, FusionStyleKind,
    GenericFArray, GenericRMatrix, RuleIdentity, SectorId, SectorVec, SymbolShapeError,
};

const CODEC_VERSION: &[u8] = b"tenet:sun:dynkin:graded-total-then-lex:v1";
const IDENTITY_SCHEMA: u64 = 0x5355_4e5f_434f_4445;

/// Checked Racah-backed SU(N) structural fusion adapter.
#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub struct SUNFusionRule {
    n: usize,
}

/// Failure at the SU(N) label/provider boundary.
#[derive(Debug)]
pub enum SUNFusionRuleError {
    InvalidRank {
        n: usize,
    },
    RankNotRepresentable {
        n: usize,
    },
    LabelAllocation {
        len: usize,
        source: std::collections::TryReserveError,
    },
    LabelLength {
        expected: usize,
        found: usize,
    },
    NegativeLabel {
        index: usize,
        value: i64,
    },
    UnrepresentableLabel {
        labels: Vec<i64>,
    },
    DecodeOverflow {
        sector: SectorId,
    },
    DimensionNotRepresentable {
        sector: SectorId,
    },
    UnexpectedFShape {
        expected: [usize; 4],
        found: [usize; 4],
    },
    UnexpectedRShape {
        expected: [usize; 2],
        found: [usize; 2],
    },
    MalformedSymbolData(SymbolShapeError),
    InvalidPivotalPhase {
        sector: SectorId,
        value: f64,
    },
    Racah(racah::sun::SunError),
}

impl fmt::Display for SUNFusionRuleError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidRank { n } => write!(f, "SU(N) requires N >= 2, got {n}"),
            Self::RankNotRepresentable { n } => {
                write!(f, "SU({n}) Dynkin labels exceed this platform's Vec limit")
            }
            Self::LabelAllocation { len, .. } => {
                write!(f, "could not allocate {len} SU(N) Dynkin labels")
            }
            Self::LabelLength { expected, found } => {
                write!(f, "expected {expected} Dynkin labels, got {found}")
            }
            Self::NegativeLabel { index, value } => {
                write!(f, "Dynkin label {index} is negative ({value})")
            }
            Self::UnrepresentableLabel { labels } => {
                write!(
                    f,
                    "Dynkin label is not representable as a SectorId: {labels:?}"
                )
            }
            Self::DecodeOverflow { sector } => {
                write!(
                    f,
                    "SectorId {} cannot be decoded by this SU(N) codec",
                    sector.id()
                )
            }
            Self::DimensionNotRepresentable { sector } => write!(
                f,
                "exact dimension for SectorId {} is not a finite positive f64",
                sector.id()
            ),
            Self::UnexpectedFShape { expected, found } => {
                write!(f, "Racah F shape {found:?} does not match {expected:?}")
            }
            Self::UnexpectedRShape { expected, found } => {
                write!(f, "Racah R shape {found:?} does not match {expected:?}")
            }
            Self::MalformedSymbolData(error) => write!(f, "malformed Racah symbol data: {error}"),
            Self::InvalidPivotalPhase { sector, value } => write!(
                f,
                "F-derived pivotal phase for SectorId {} is invalid ({value})",
                sector.id()
            ),
            Self::Racah(error) => write!(f, "Racah SU(N) error: {error}"),
        }
    }
}

impl std::error::Error for SUNFusionRuleError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::LabelAllocation { source, .. } => Some(source),
            Self::MalformedSymbolData(error) => Some(error),
            Self::Racah(error) => Some(error),
            _ => None,
        }
    }
}

impl SUNFusionRule {
    pub fn new(n: usize) -> Result<Self, SUNFusionRuleError> {
        if n < 2 {
            return Err(SUNFusionRuleError::InvalidRank { n });
        }
        if n - 1 > isize::MAX as usize / size_of::<i64>() {
            return Err(SUNFusionRuleError::RankNotRepresentable { n });
        }
        Ok(Self { n })
    }

    pub const fn rank(&self) -> usize {
        self.n
    }

    /// Encodes nonnegative length `N - 1` Dynkin labels by total grade then lexicographic order.
    pub fn encode_dynkin(&self, labels: &[i64]) -> Result<SectorId, SUNFusionRuleError> {
        self.check_labels(labels)?;
        if labels.iter().any(|&label| usize::try_from(label).is_err()) {
            return Err(SUNFusionRuleError::UnrepresentableLabel {
                labels: labels.to_vec(),
            });
        }
        let labels: Vec<usize> = labels.iter().map(|&x| x as usize).collect();
        let grade = labels
            .iter()
            .try_fold(0usize, |sum, &x| sum.checked_add(x))
            .ok_or_else(|| SUNFusionRuleError::UnrepresentableLabel {
                labels: labels.iter().map(|&x| x as i64).collect(),
            })?;
        let mut rank = count_before_grade(grade, self.n - 1, usize::MAX)
            .exact()
            .ok_or_else(|| SUNFusionRuleError::UnrepresentableLabel {
                labels: labels.iter().map(|&x| x as i64).collect(),
            })?;
        let mut remaining = grade;
        for (index, &label) in labels.iter().enumerate() {
            let tail = self.n - 2 - index;
            rank = rank
                .checked_add(
                    prefix_count(remaining, tail, label, usize::MAX)
                        .and_then(Capped::exact)
                        .ok_or_else(|| SUNFusionRuleError::UnrepresentableLabel {
                            labels: labels.iter().map(|&x| x as i64).collect(),
                        })?,
                )
                .ok_or_else(|| SUNFusionRuleError::UnrepresentableLabel {
                    labels: labels.iter().map(|&x| x as i64).collect(),
                })?;
            remaining -= label;
        }
        Ok(SectorId::new(rank))
    }

    /// Decodes a `SectorId` with bounded binary searches over the grade and each label.
    pub fn decode_dynkin(&self, sector: SectorId) -> Result<Vec<i64>, SUNFusionRuleError> {
        let id = sector.id();
        let r = self.n - 1;
        let mut low = 0usize;
        let mut high = id;
        while low < high {
            let mid = low + (high - low) / 2;
            if count_before_grade(mid + 1, r, id).is_exceeded() {
                high = mid;
            } else {
                low = mid + 1;
            }
        }
        let grade = low;
        let before = if grade == 0 {
            0
        } else {
            count_before_grade(grade, r, id)
                .exact()
                .ok_or(SUNFusionRuleError::DecodeOverflow { sector })?
        };
        let mut residual = id - before;
        let mut remaining = grade;
        let mut labels = Vec::new();
        labels
            .try_reserve_exact(r)
            .map_err(|source| SUNFusionRuleError::LabelAllocation { len: r, source })?;
        for index in 0..r {
            let tail = r - index - 1;
            let mut low = 0usize;
            let mut high = remaining;
            while low < high {
                let mid = low + (high - low).div_ceil(2);
                match prefix_count(remaining, tail, mid, residual) {
                    Some(Capped::Exact(count)) if count <= residual => low = mid,
                    _ => high = mid - 1,
                }
            }
            let skipped = prefix_count(remaining, tail, low, residual)
                .and_then(Capped::exact)
                .ok_or(SUNFusionRuleError::DecodeOverflow { sector })?;
            residual -= skipped;
            labels.push(
                i64::try_from(low).map_err(|_| SUNFusionRuleError::DecodeOverflow { sector })?,
            );
            remaining -= low;
        }
        if residual != 0 {
            return Err(SUNFusionRuleError::DecodeOverflow { sector });
        }
        Ok(labels)
    }

    fn check_labels(&self, labels: &[i64]) -> Result<(), SUNFusionRuleError> {
        if labels.len() != self.n - 1 {
            return Err(SUNFusionRuleError::LabelLength {
                expected: self.n - 1,
                found: labels.len(),
            });
        }
        if let Some((index, &value)) = labels.iter().enumerate().find(|(_, value)| **value < 0) {
            return Err(SUNFusionRuleError::NegativeLabel { index, value });
        }
        Ok(())
    }

    fn identity_bytes(&self) -> Arc<[u8]> {
        let mut bytes = Vec::with_capacity(racah::sun::sun_authority_fingerprint().len() + 32);
        bytes.extend_from_slice(CODEC_VERSION);
        bytes.extend_from_slice(&(usize::BITS).to_le_bytes());
        bytes.extend_from_slice(&self.n.to_le_bytes());
        bytes.extend_from_slice(b":generic:bosonic:racah-sun");
        bytes.extend_from_slice(
            b":rigid=f64-finite-exact-dim:f-axes=mu-nu-kappa-lambda:r-axes=mu-nu:pivotal=f-sign-0000",
        );
        bytes.extend_from_slice(racah::sun::sun_authority_fingerprint());
        Arc::from(bytes)
    }

    fn irrep(&self, sector: SectorId) -> Result<racah::sun::Irrep, SUNFusionRuleError> {
        let labels = self.decode_dynkin(sector)?;
        racah::sun::Irrep::from_dynkin(&labels).map_err(SUNFusionRuleError::Racah)
    }

    fn dim_scalar(&self, sector: SectorId) -> Result<f64, SUNFusionRuleError> {
        self.irrep(sector)?
            .dim()
            .to_f64()
            .filter(|value| value.is_finite() && *value > 0.0)
            .ok_or(SUNFusionRuleError::DimensionNotRepresentable { sector })
    }
}

impl CheckedGenericFusion for SUNFusionRule {
    type Error = SUNFusionRuleError;

    fn rule_identity(&self) -> RuleIdentity {
        RuleIdentity::from_canonical_bytes::<Self>(IDENTITY_SCHEMA, self.identity_bytes())
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

    fn try_dual(&self, sector: SectorId) -> Result<SectorId, Self::Error> {
        let mut labels = self.decode_dynkin(sector)?;
        labels.reverse();
        self.encode_dynkin(&labels)
    }

    fn try_fusion_channels(
        &self,
        left: SectorId,
        right: SectorId,
    ) -> Result<SectorVec, Self::Error> {
        let product = racah::sun::directproduct(&self.irrep(left)?, &self.irrep(right)?)
            .map_err(SUNFusionRuleError::Racah)?;
        let mut channels: SectorVec = product
            .keys()
            .map(|irrep| self.encode_dynkin(&irrep.dynkin()))
            .collect::<Result<_, _>>()?;
        channels.sort_unstable();
        Ok(channels)
    }

    fn try_fusion_channels_in_table(
        &self,
        left: SectorId,
        right: SectorId,
    ) -> Result<SectorVec, Self::Error> {
        self.try_fusion_channels(left, right)
    }

    fn try_nsymbol(
        &self,
        left: SectorId,
        right: SectorId,
        coupled: SectorId,
    ) -> Result<usize, Self::Error> {
        let product = racah::sun::directproduct(&self.irrep(left)?, &self.irrep(right)?)
            .map_err(SUNFusionRuleError::Racah)?;
        Ok(product.get(&self.irrep(coupled)?).copied().unwrap_or(0) as usize)
    }
}

impl CheckedGenericRigidSymbols for SUNFusionRule {
    type Scalar = f64;

    fn try_sqrt_dim_scalar(&mut self, sector: SectorId) -> Result<f64, Self::Error> {
        Ok(self.dim_scalar(sector)?.sqrt())
    }

    fn try_inv_sqrt_dim_scalar(&mut self, sector: SectorId) -> Result<f64, Self::Error> {
        Ok(self.dim_scalar(sector)?.sqrt().recip())
    }

    fn try_frobenius_schur_phase_scalar(&mut self, sector: SectorId) -> Result<f64, Self::Error> {
        let a = self.irrep(sector)?;
        let dual = a.dual();
        let unit = racah::sun::Irrep::trivial(self.n).map_err(SUNFusionRuleError::Racah)?;
        let block = racah::sun::f_symbol(&a, &dual, &a, &a, &unit, &unit)
            .map_err(SUNFusionRuleError::Racah)?;
        if block.dims() != [1, 1, 1, 1] {
            return Err(SUNFusionRuleError::UnexpectedFShape {
                expected: [1, 1, 1, 1],
                found: block.dims(),
            });
        }
        let coefficient = block.at(0, 0, 0, 0);
        if !coefficient.is_finite() || coefficient == 0.0 {
            return Err(SUNFusionRuleError::InvalidPivotalPhase {
                sector,
                value: coefficient,
            });
        }
        Ok(coefficient.signum())
    }

    fn try_f_symbol_generic(
        &mut self,
        a: SectorId,
        b: SectorId,
        c: SectorId,
        d: SectorId,
        e: SectorId,
        f: SectorId,
    ) -> Result<GenericFArray<f64>, Self::Error> {
        let expected = [
            self.try_nsymbol(a, b, e)?,
            self.try_nsymbol(e, c, d)?,
            self.try_nsymbol(b, c, f)?,
            self.try_nsymbol(a, f, d)?,
        ];
        let block = racah::sun::f_symbol(
            &self.irrep(a)?,
            &self.irrep(b)?,
            &self.irrep(c)?,
            &self.irrep(d)?,
            &self.irrep(e)?,
            &self.irrep(f)?,
        )
        .map_err(SUNFusionRuleError::Racah)?;
        if block.dims() != expected {
            return Err(SUNFusionRuleError::UnexpectedFShape {
                expected,
                found: block.dims(),
            });
        }
        GenericFArray::try_new(
            block.data().to_vec(),
            (expected[0], expected[1], expected[2], expected[3]),
        )
        .map_err(SUNFusionRuleError::MalformedSymbolData)
    }

    fn try_r_symbol_generic(
        &mut self,
        a: SectorId,
        b: SectorId,
        c: SectorId,
    ) -> Result<GenericRMatrix<f64>, Self::Error> {
        let expected = [self.try_nsymbol(a, b, c)?, self.try_nsymbol(b, a, c)?];
        let block = racah::sun::r_symbol(&self.irrep(a)?, &self.irrep(b)?, &self.irrep(c)?)
            .map_err(SUNFusionRuleError::Racah)?;
        let found = [block.dim(), block.dim()];
        if found != expected {
            return Err(SUNFusionRuleError::UnexpectedRShape { expected, found });
        }
        GenericRMatrix::try_new(block.data().to_vec(), expected[0], expected[1])
            .map_err(SUNFusionRuleError::MalformedSymbolData)
    }
}

enum Capped {
    Exact(usize),
    Exceeded,
}
impl Capped {
    fn exact(self) -> Option<usize> {
        if let Self::Exact(value) = self {
            Some(value)
        } else {
            None
        }
    }
    fn is_exceeded(&self) -> bool {
        matches!(self, Self::Exceeded)
    }
}

fn count_before_grade(grade: usize, parts: usize, cap: usize) -> Capped {
    match grade.checked_add(parts.saturating_sub(1)) {
        Some(n) => binomial(n, parts, cap),
        None => Capped::Exceeded,
    }
}

fn prefix_count(remaining: usize, tail: usize, labels_before: usize, cap: usize) -> Option<Capped> {
    if labels_before == 0 || tail == 0 {
        return Some(Capped::Exact(usize::from(labels_before > remaining)));
    }
    let upper = binomial_u128(remaining.checked_add(tail)?, tail)?;
    let lower = binomial_u128(
        remaining.checked_sub(labels_before)?.checked_add(tail)?,
        tail,
    )?;
    let difference = upper.checked_sub(lower)?;
    Some(if difference > cap as u128 {
        Capped::Exceeded
    } else {
        Capped::Exact(difference as usize)
    })
}

fn binomial(n: usize, k: usize, cap: usize) -> Capped {
    match binomial_u128(n, k) {
        Some(value) if value <= cap as u128 => Capped::Exact(value as usize),
        _ => Capped::Exceeded,
    }
}

const _: () = assert!(usize::BITS <= 64);

// Why not arbitrary precision: for a valid grade g > 0 with r parts,
// B=C(g+r-1,r)<=M. The largest prefix upper is
// U=C(g+r-1,r-1)=B*r/g<=M²<2¹²⁸ for M=usize::MAX; later states decrease.
// A count-before value overflowing u128 is therefore already greater than M.
fn binomial_u128(n: usize, k: usize) -> Option<u128> {
    if k > n {
        return Some(0);
    }
    let k = k.min(n - k);
    let mut result = 1u128;
    for i in 1..=k {
        let mut numerator = (n - k + i) as u128;
        let mut denominator = i as u128;
        let divisor = gcd_u128(numerator, denominator);
        numerator /= divisor;
        denominator /= divisor;
        let divisor = gcd_u128(result, denominator);
        result /= divisor;
        denominator /= divisor;
        if denominator != 1 {
            return None;
        }
        result = result.checked_mul(numerator)?;
    }
    Some(result)
}

const fn gcd_u128(mut left: u128, mut right: u128) -> u128 {
    while right != 0 {
        let remainder = left % right;
        left = right;
        right = remainder;
    }
    left
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn codec_is_graded_lexicographic_and_roundtrips() {
        let rule = SUNFusionRule::new(3).unwrap();
        let labels = [[0, 0], [0, 1], [1, 0], [0, 2], [1, 1], [2, 0]];
        for (id, label) in labels.into_iter().enumerate() {
            assert_eq!(rule.encode_dynkin(&label).unwrap(), SectorId::new(id));
            assert_eq!(rule.decode_dynkin(SectorId::new(id)).unwrap(), label);
        }
        let su2 = SUNFusionRule::new(2).unwrap();
        let edge = [i64::MAX];
        let id = su2.encode_dynkin(&edge).unwrap();
        assert_eq!(id, SectorId::new(i64::MAX as usize));
        assert_eq!(su2.decode_dynkin(id).unwrap(), edge);
    }

    #[test]
    fn codec_handles_bounded_binomial_cancellation() {
        if usize::BITS != 64 {
            return;
        }
        let rule = SUNFusionRule::new(39).unwrap();
        let sector = SectorId::new(usize::try_from(17_876_288_714_431_443_296u64).unwrap());
        let mut expected = vec![0; 38];
        expected[37] = 31;
        assert_eq!(rule.decode_dynkin(sector).unwrap(), expected);
        assert_eq!(rule.encode_dynkin(&expected).unwrap(), sector);
    }

    #[test]
    fn validation_and_identity_are_typed_and_stable() {
        assert!(matches!(
            SUNFusionRule::new(1),
            Err(SUNFusionRuleError::InvalidRank { .. })
        ));
        let rule = SUNFusionRule::new(3).unwrap();
        assert!(matches!(
            rule.encode_dynkin(&[1]),
            Err(SUNFusionRuleError::LabelLength { .. })
        ));
        assert!(matches!(
            rule.encode_dynkin(&[-1, 0]),
            Err(SUNFusionRuleError::NegativeLabel { .. })
        ));
        assert!(matches!(
            rule.encode_dynkin(&[i64::MAX, i64::MAX]),
            Err(SUNFusionRuleError::UnrepresentableLabel { .. })
        ));
        assert!(matches!(
            SUNFusionRule::new(usize::MAX),
            Err(SUNFusionRuleError::RankNotRepresentable { .. })
        ));
        if usize::BITS == 64 {
            let max_rank = isize::MAX as usize / size_of::<i64>() + 1;
            let rule = SUNFusionRule::new(max_rank).unwrap();
            assert!(matches!(
                rule.decode_dynkin(rule.vacuum()),
                Err(SUNFusionRuleError::LabelAllocation { .. })
            ));
        }
        for n in [3, 4] {
            let rule = SUNFusionRule::new(n).unwrap();
            let labels = rule.decode_dynkin(SectorId::new(usize::MAX)).unwrap();
            assert_eq!(
                rule.encode_dynkin(&labels).unwrap(),
                SectorId::new(usize::MAX)
            );
        }
        assert!(matches!(
            SUNFusionRule::new(2)
                .unwrap()
                .decode_dynkin(SectorId::new(usize::MAX)),
            Err(SUNFusionRuleError::DecodeOverflow { .. })
        ));
        assert_eq!(
            rule.rule_identity(),
            SUNFusionRule::new(3).unwrap().rule_identity()
        );
        assert_ne!(
            rule.rule_identity(),
            SUNFusionRule::new(4).unwrap().rule_identity()
        );
        let original = rule.identity_bytes();
        let mut expected = b"tenet:sun:dynkin:graded-total-then-lex:v1".to_vec();
        expected.extend_from_slice(&usize::BITS.to_le_bytes());
        expected.extend_from_slice(&3usize.to_le_bytes());
        expected.extend_from_slice(b":generic:bosonic:racah-sun");
        expected.extend_from_slice(
            b":rigid=f64-finite-exact-dim:f-axes=mu-nu-kappa-lambda:r-axes=mu-nu:pivotal=f-sign-0000",
        );
        expected.extend_from_slice(racah::sun::sun_authority_fingerprint());
        assert_eq!(original.as_ref(), expected);
        let mut changed = original.to_vec();
        changed[0] ^= 1;
        assert_ne!(
            RuleIdentity::from_canonical_bytes::<SUNFusionRule>(IDENTITY_SCHEMA, original),
            RuleIdentity::from_canonical_bytes::<SUNFusionRule>(
                IDENTITY_SCHEMA,
                Arc::from(changed)
            )
        );
    }

    #[test]
    fn su3_generic_structure_uses_racah() {
        let rule = SUNFusionRule::new(3).unwrap();
        let eight = rule.encode_dynkin(&[1, 1]).unwrap();
        assert_eq!(rule.fusion_style(), FusionStyleKind::Generic);
        assert_eq!(rule.braiding_style(), BraidingStyleKind::Bosonic);
        assert_eq!(rule.try_nsymbol(eight, eight, eight).unwrap(), 2);
        assert_eq!(
            rule.try_dual(rule.encode_dynkin(&[2, 1]).unwrap()).unwrap(),
            rule.encode_dynkin(&[1, 2]).unwrap()
        );
        let channels = rule.try_fusion_channels(eight, eight).unwrap();
        assert_eq!(
            channels
                .iter()
                .map(|sector| sector.id())
                .collect::<Vec<_>>(),
            [0, 4, 6, 9, 12]
        );
        assert_eq!(
            channels
                .iter()
                .map(|&sector| rule.decode_dynkin(sector).unwrap())
                .collect::<Vec<_>>(),
            [[0, 0], [1, 1], [0, 3], [3, 0], [2, 2]]
        );
        assert_eq!(
            rule.try_fusion_channels_in_table(eight, eight).unwrap(),
            channels
        );
        for (labels, multiplicity) in [
            ([0, 0], 1),
            ([1, 1], 2),
            ([0, 3], 1),
            ([3, 0], 1),
            ([2, 2], 1),
        ] {
            assert_eq!(
                rule.try_nsymbol(eight, eight, rule.encode_dynkin(&labels).unwrap())
                    .unwrap(),
                multiplicity
            );
        }
        assert_eq!(
            rule.try_nsymbol(eight, eight, rule.encode_dynkin(&[1, 0]).unwrap())
                .unwrap(),
            0
        );
    }

    #[test]
    fn su3_rigid_symbols_match_sunrepresentations_fixtures() {
        let mut rule = SUNFusionRule::new(3).unwrap();
        let three = rule.encode_dynkin(&[1, 0]).unwrap();
        let anti_three = rule.encode_dynkin(&[0, 1]).unwrap();
        let eight = rule.encode_dynkin(&[1, 1]).unwrap();

        for (sector, dimension) in [(three, 3.0), (anti_three, 3.0), (eight, 8.0)] {
            let sqrt = rule.try_sqrt_dim_scalar(sector).unwrap();
            assert!((sqrt * sqrt - dimension).abs() < 1e-12);
            assert!((sqrt * rule.try_inv_sqrt_dim_scalar(sector).unwrap() - 1.0).abs() < 1e-12);
            assert_ne!(rule.try_frobenius_schur_phase_scalar(sector).unwrap(), 0.0);
        }

        let f = rule
            .try_f_symbol_generic(eight, eight, eight, eight, eight, eight)
            .unwrap();
        assert_eq!(f.shape(), (2, 2, 2, 2));
        for (index, expected) in [
            ((0, 0, 0, 0), 0.857_142_857_142_856),
            ((0, 0, 1, 1), -0.142_857_142_857_142_63),
            ((0, 1, 1, 1), -0.383_325_938_999_963_37),
            ((1, 1, 1, 1), 0.628_571_428_571_427_4),
        ] {
            let (mu, nu, kappa, lambda) = index;
            assert!((*f.get(mu, nu, kappa, lambda) - expected).abs() < 1e-12);
        }

        let twenty_seven = rule.encode_dynkin(&[2, 2]).unwrap();
        let asymmetric = rule
            .try_f_symbol_generic(
                twenty_seven,
                twenty_seven,
                anti_three,
                anti_three,
                eight,
                rule.encode_dynkin(&[3, 1]).unwrap(),
            )
            .unwrap();
        assert_eq!(asymmetric.shape(), (2, 1, 1, 1));
        assert!((*asymmetric.get(1, 0, 0, 0) - 0.797_724_035_217_465_4).abs() < 1e-12);

        let r = rule.try_r_symbol_generic(eight, eight, eight).unwrap();
        assert_eq!(r.shape(), (2, 2));
        for (index, expected) in [
            ((0, 0), -0.285_714_285_714_285_3),
            ((0, 1), 0.958_314_847_499_908_8),
            ((1, 0), 0.958_314_847_499_908_9),
            ((1, 1), 0.285_714_285_714_285_53),
        ] {
            assert!((*r.get(index.0, index.1) - expected).abs() < 1e-12);
        }

        let left = rule.encode_dynkin(&[2, 2]).unwrap();
        let right = rule.encode_dynkin(&[2, 3]).unwrap();
        let asymmetric_r = rule.try_r_symbol_generic(left, right, right).unwrap();
        assert_eq!(asymmetric_r.shape(), (3, 3));
        for (index, expected) in [
            ((0, 1), -0.439_764_978_446_187_45),
            ((1, 0), -0.374_954_863_181_616_1),
            ((0, 2), 0.887_615_402_640_999_9),
            ((2, 0), 0.916_876_867_332_670_7),
        ] {
            assert!((*asymmetric_r.get(index.0, index.1) - expected).abs() < 1e-12);
        }
    }

    #[test]
    fn su2_fundamental_pivotal_phase_matches_f_gauge() {
        let mut rule = SUNFusionRule::new(2).unwrap();
        let fundamental = rule.encode_dynkin(&[1]).unwrap();
        assert_eq!(
            rule.try_frobenius_schur_phase_scalar(fundamental).unwrap(),
            -1.0
        );
    }

    #[test]
    fn su4_adjoint_rigid_shape_smoke() {
        let mut rule = SUNFusionRule::new(4).unwrap();
        let adjoint = rule.encode_dynkin(&[1, 0, 1]).unwrap();
        assert_eq!(rule.dim_scalar(adjoint).unwrap(), 15.0);
        assert_eq!(
            rule.try_f_symbol_generic(adjoint, adjoint, adjoint, adjoint, adjoint, adjoint)
                .unwrap()
                .shape(),
            (2, 2, 2, 2)
        );
        assert_eq!(
            rule.try_r_symbol_generic(adjoint, adjoint, adjoint)
                .unwrap()
                .shape(),
            (2, 2)
        );
    }

    #[test]
    fn rigid_failures_remain_typed() {
        let mut rule = SUNFusionRule::new(3).unwrap();
        let three = rule.encode_dynkin(&[1, 0]).unwrap();
        let eight = rule.encode_dynkin(&[1, 1]).unwrap();
        assert!(matches!(
            rule.try_r_symbol_generic(three, three, eight),
            Err(SUNFusionRuleError::Racah(
                racah::sun::SunError::ZeroFusionChannel { .. }
            ))
        ));
        assert!(matches!(
            rule.try_f_symbol_generic(three, three, three, three, eight, three),
            Err(SUNFusionRuleError::Racah(
                racah::sun::SunError::ZeroFusionChannel { .. }
            ))
        ));

        let rule = SUNFusionRule::new(78).unwrap();
        let mut labels = vec![0; 77];
        labels[36] = 19;
        let sector = rule.encode_dynkin(&labels).unwrap();
        assert!(matches!(
            rule.dim_scalar(sector),
            Err(SUNFusionRuleError::DimensionNotRepresentable { .. })
        ));
    }
}
