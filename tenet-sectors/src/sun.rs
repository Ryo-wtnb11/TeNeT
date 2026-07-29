use std::fmt;
use std::sync::Arc;

use crate::{
    BraidingStyleKind, CheckedGenericFusion, FusionStyleKind, RuleIdentity, SectorId, SectorVec,
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
    InvalidRank { n: usize },
    LabelLength { expected: usize, found: usize },
    NegativeLabel { index: usize, value: i64 },
    UnrepresentableLabel { labels: Vec<i64> },
    DecodeOverflow { sector: SectorId },
    Racah(racah::sun::SunError),
}

impl fmt::Display for SUNFusionRuleError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidRank { n } => write!(f, "SU(N) requires N >= 2, got {n}"),
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
            Self::Racah(error) => write!(f, "Racah SU(N) error: {error}"),
        }
    }
}

impl std::error::Error for SUNFusionRuleError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
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
            rank =
                rank.checked_add(prefix_count(remaining, tail, label, usize::MAX).map_err(
                    |_| SUNFusionRuleError::UnrepresentableLabel {
                        labels: labels.iter().map(|&x| x as i64).collect(),
                    },
                )?)
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
        let mut labels = Vec::with_capacity(r);
        for index in 0..r {
            let tail = r - index - 1;
            let mut low = 0usize;
            let mut high = remaining;
            while low < high {
                let mid = low + (high - low).div_ceil(2);
                let count = prefix_count(remaining, tail, mid, residual)?;
                if count <= residual {
                    low = mid;
                } else {
                    high = mid - 1;
                }
            }
            let skipped = prefix_count(remaining, tail, low, residual)?;
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
        bytes.extend_from_slice(racah::sun::sun_authority_fingerprint());
        Arc::from(bytes)
    }

    fn irrep(&self, sector: SectorId) -> Result<racah::sun::Irrep, SUNFusionRuleError> {
        let labels = self.decode_dynkin(sector)?;
        racah::sun::Irrep::from_dynkin(&labels).map_err(SUNFusionRuleError::Racah)
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

fn prefix_count(
    remaining: usize,
    tail: usize,
    labels_before: usize,
    cap: usize,
) -> Result<usize, SUNFusionRuleError> {
    if labels_before == 0 || tail == 0 {
        return Ok(usize::from(labels_before > remaining));
    }
    let upper = exact_binomial(
        remaining
            .checked_add(tail)
            .ok_or_else(|| SUNFusionRuleError::UnrepresentableLabel { labels: Vec::new() })?,
        tail,
    )?;
    let lower = exact_binomial(
        remaining
            .checked_sub(labels_before)
            .and_then(|value| value.checked_add(tail))
            .ok_or_else(|| SUNFusionRuleError::UnrepresentableLabel { labels: Vec::new() })?,
        tail,
    )?;
    let difference = upper
        .checked_sub(lower)
        .ok_or_else(|| SUNFusionRuleError::UnrepresentableLabel { labels: Vec::new() })?;
    Ok(difference.min(cap.saturating_add(1)))
}

fn exact_binomial(n: usize, k: usize) -> Result<usize, SUNFusionRuleError> {
    match binomial(n, k, usize::MAX) {
        Capped::Exact(value) => Ok(value),
        Capped::Exceeded => Err(SUNFusionRuleError::DecodeOverflow {
            sector: SectorId::new(usize::MAX),
        }),
    }
}

fn binomial(n: usize, k: usize, cap: usize) -> Capped {
    if k > n {
        return Capped::Exact(0);
    }
    let k = k.min(n - k);
    let mut result = 1usize;
    for i in 1..=k {
        let mut numerator = n - k + i;
        let mut denominator = i;
        let divisor = gcd(numerator, denominator);
        numerator /= divisor;
        denominator /= divisor;
        let divisor = gcd(result, denominator);
        result /= divisor;
        denominator /= divisor;
        debug_assert_eq!(denominator, 1);
        match result.checked_mul(numerator) {
            Some(value) if value <= cap => result = value,
            _ => return Capped::Exceeded,
        }
    }
    Capped::Exact(result)
}

const fn gcd(mut left: usize, mut right: usize) -> usize {
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
        assert_eq!(
            rule.rule_identity(),
            SUNFusionRule::new(3).unwrap().rule_identity()
        );
        assert_ne!(
            rule.rule_identity(),
            SUNFusionRule::new(4).unwrap().rule_identity()
        );
        let original = rule.identity_bytes();
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
        assert!(rule
            .try_fusion_channels(eight, eight)
            .unwrap()
            .contains(&eight));
    }
}
