//! Placement-neutral validation of axis permutations and axis subsets.
//!
//! Every crate validates axis lists through this module and maps
//! [`AxisError`] into its own error type at the call site, so a malformed list
//! is rejected by one rule regardless of the operation or storage that
//! received it.

use smallvec::SmallVec;

/// Why an axis list was rejected.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AxisError {
    /// `axis` is not below `rank`.
    OutOfRange { axis: usize, rank: usize },
    /// `axis` occurs more than once.
    Duplicate { axis: usize },
    /// A permutation of `expected` axes received `actual` entries.
    Length { expected: usize, actual: usize },
}

/// The set of axes of a rank-`rank` tensor selected so far.
///
/// Inline (allocation-free) up to rank 128.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AxisMask {
    words: SmallVec<[u64; 2]>,
    rank: usize,
}

impl AxisMask {
    /// An empty selection over `rank` axes.
    pub fn new(rank: usize) -> Self {
        Self {
            words: SmallVec::from_elem(0, rank.div_ceil(u64::BITS as usize)),
            rank,
        }
    }

    /// Selects `axis`, rejecting out-of-range and already-selected axes.
    #[inline]
    pub fn insert(&mut self, axis: usize) -> Result<(), AxisError> {
        if axis >= self.rank {
            return Err(AxisError::OutOfRange {
                axis,
                rank: self.rank,
            });
        }
        let (word, bit) = Self::locate(axis);
        if self.words[word] & bit != 0 {
            return Err(AxisError::Duplicate { axis });
        }
        self.words[word] |= bit;
        Ok(())
    }

    /// Selects every axis of `axes` in order; stops at the first rejection.
    pub fn insert_all(&mut self, axes: &[usize]) -> Result<(), AxisError> {
        axes.iter().try_for_each(|&axis| self.insert(axis))
    }

    /// Whether `axis` is selected; `false` for out-of-range axes.
    #[inline]
    pub fn contains(&self, axis: usize) -> bool {
        if axis >= self.rank {
            return false;
        }
        let (word, bit) = Self::locate(axis);
        self.words[word] & bit != 0
    }

    /// Whether every axis below `rank` is selected.
    pub fn is_full(&self) -> bool {
        (0..self.rank).all(|axis| self.contains(axis))
    }

    /// The unselected axes in increasing order.
    pub fn complement(&self) -> impl Iterator<Item = usize> + '_ {
        (0..self.rank).filter(|&axis| !self.contains(axis))
    }

    #[inline]
    fn locate(axis: usize) -> (usize, u64) {
        (
            axis / u64::BITS as usize,
            1u64 << (axis % u64::BITS as usize),
        )
    }
}

/// Checks that `axes` is a permutation of `0..rank`.
pub fn validate_permutation(axes: &[usize], rank: usize) -> Result<(), AxisError> {
    if axes.len() != rank {
        return Err(AxisError::Length {
            expected: rank,
            actual: axes.len(),
        });
    }
    validate_axis_subset(axes, rank).map(|_| ())
}

/// Checks that `axes` are distinct axes below `rank` and returns them as a
/// mask.
pub fn validate_axis_subset(axes: &[usize], rank: usize) -> Result<AxisMask, AxisError> {
    let mut mask = AxisMask::new(rank);
    mask.insert_all(axes)?;
    Ok(mask)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn permutation_accepts_bijections_and_rejects_each_misuse() {
        assert_eq!(validate_permutation(&[], 0), Ok(()));
        assert_eq!(validate_permutation(&[2, 0, 1], 3), Ok(()));
        assert_eq!(
            validate_permutation(&[0, 1], 3),
            Err(AxisError::Length {
                expected: 3,
                actual: 2
            })
        );
        assert_eq!(
            validate_permutation(&[0, 3, 1], 3),
            Err(AxisError::OutOfRange { axis: 3, rank: 3 })
        );
        assert_eq!(
            validate_permutation(&[1, 0, 1], 3),
            Err(AxisError::Duplicate { axis: 1 })
        );
    }

    #[test]
    fn subset_mask_and_complement_cross_word_boundaries() {
        let rank = 130;
        let axes = [129, 0, 64, 63];
        let mask = validate_axis_subset(&axes, rank).unwrap();
        for axis in 0..rank {
            assert_eq!(mask.contains(axis), axes.contains(&axis), "axis {axis}");
        }
        assert!(!mask.contains(rank));
        assert_eq!(mask.complement().count(), rank - axes.len());
        assert!(!mask.is_full());
        assert_eq!(
            validate_axis_subset(&[64, 64], rank),
            Err(AxisError::Duplicate { axis: 64 })
        );
        assert_eq!(
            validate_axis_subset(&[130], rank),
            Err(AxisError::OutOfRange {
                axis: 130,
                rank: 130
            })
        );
        let mut full = AxisMask::new(3);
        full.insert_all(&[2, 1, 0]).unwrap();
        assert!(full.is_full());
        assert!(AxisMask::new(0).is_full());
    }
}
