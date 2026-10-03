use super::*;

#[non_exhaustive]
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum CoreError {
    RankMismatch {
        shape: usize,
        strides: usize,
    },
    StructureRankMismatch {
        expected: usize,
        actual: usize,
    },
    FusionSpaceSplitMismatch {
        expected_nout: usize,
        expected_nin: usize,
        actual_nout: usize,
        actual_nin: usize,
    },
    DimensionMismatch {
        expected: usize,
        actual: usize,
    },
    InvalidBraidIndex {
        index: usize,
        rank: usize,
    },
    InvalidPermutation {
        permutation: Vec<usize>,
        rank: usize,
    },
    /// Derived axis-position metadata disagrees with its source operation.
    InconsistentAxisPosition {
        logical_axis: usize,
        expected_position: usize,
        actual_position: usize,
    },
    UnitLayoutCorrespondence,
    UnsupportedFusionStyle {
        expected: FusionStyleKind,
        actual: FusionStyleKind,
    },
    UnsupportedBraidingStyle {
        expected: &'static str,
        actual: BraidingStyleKind,
    },
    UnsupportedSectorBraid {
        left: SectorId,
        right: SectorId,
        style: BraidingStyleKind,
    },
    InvalidSector {
        sector: SectorId,
    },
    InvalidMultiplicityIndex {
        value: usize,
    },
    SectorMismatch {
        expected: SectorId,
        actual: SectorId,
    },
    FusionRuleMismatch {
        expected: RuleIdentity,
        actual: RuleIdentity,
    },
    MissingFusionRuleIdentity,
    /// A per-sector leg degeneracy disagrees with another authoritative
    /// source (the paired leg of a composition, or a fusion-tree degeneracy
    /// shape validated against its leg).
    LegDegeneracyMismatch {
        sector: SectorId,
        expected: usize,
        actual: usize,
    },
    FusionChannelCount {
        left: SectorId,
        right: SectorId,
        count: usize,
    },
    MalformedFusionTree {
        message: &'static str,
    },
    /// A contracted leg pair does not pair a space with its dual. Axes are
    /// the operands' external axes and the flags are their external-axis
    /// duality flags; a valid pair has opposite flags.
    ContractedLegDualityMismatch {
        lhs_axis: usize,
        rhs_axis: usize,
        lhs_is_dual: bool,
        rhs_is_dual: bool,
    },
    BlockCountMismatch {
        expected: usize,
        actual: usize,
    },
    BlockIndexOutOfBounds {
        index: usize,
        count: usize,
    },
    OverlappingBlockStorage {
        first_block: usize,
        second_block: usize,
        offset: usize,
    },
    DuplicateBlockKey {
        key: Box<BlockKey>,
    },
    MixedBlockKeyKinds {
        expected: BlockKeyKind,
        actual: BlockKeyKind,
    },
    ExpectedFusionTreePairKey {
        actual: BlockKeyKind,
    },
    MissingBlockKey {
        key: Box<BlockKey>,
    },
    MissingFusionSpace,
    /// A bounded Generic provider cannot represent the
    /// requested space/sector exactly. Carries the full human-readable
    /// diagnosis; block dimensions are either exact or this error — never
    /// silently truncated.
    FusionOutsideTable {
        message: String,
    },
    ElementCountOverflow,
    OffsetOverflow {
        value: usize,
    },
    StrideOverflow {
        value: usize,
    },
    OutOfBounds,
}

impl fmt::Display for CoreError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::RankMismatch { shape, strides } => {
                write!(
                    f,
                    "rank mismatch: shape rank {shape}, strides rank {strides}"
                )
            }
            Self::StructureRankMismatch { expected, actual } => {
                write!(
                    f,
                    "block structure rank mismatch: expected {expected}, got {actual}"
                )
            }
            Self::FusionSpaceSplitMismatch {
                expected_nout,
                expected_nin,
                actual_nout,
                actual_nin,
            } => {
                write!(
                    f,
                    "fusion-space split mismatch: hom space is {expected_nout} <- {expected_nin}, dynamic split is {actual_nout} <- {actual_nin}"
                )
            }
            Self::DimensionMismatch { expected, actual } => {
                write!(f, "dimension mismatch: expected {expected}, got {actual}")
            }
            Self::InvalidBraidIndex { index, rank } => {
                write!(
                    f,
                    "cannot braid adjacent fusion-tree outputs at index {index} for rank {rank}"
                )
            }
            Self::InvalidPermutation { permutation, rank } => {
                write!(f, "invalid permutation {permutation:?} for rank {rank}")
            }
            Self::InconsistentAxisPosition {
                logical_axis,
                expected_position,
                actual_position,
            } => {
                write!(
                    f,
                    "inconsistent axis position for logical axis {logical_axis}: expected {expected_position}, got {actual_position}"
                )
            }
            Self::UnitLayoutCorrespondence => {
                write!(
                    f,
                    "unit-leg layout does not correspond to the supplied source layout"
                )
            }
            Self::UnsupportedFusionStyle { expected, actual } => {
                write!(
                    f,
                    "unsupported fusion style {actual:?}; expected {expected:?}"
                )
            }
            Self::UnsupportedBraidingStyle { expected, actual } => {
                write!(
                    f,
                    "unsupported braiding style {actual:?}; expected {expected}"
                )
            }
            Self::UnsupportedSectorBraid { left, right, style } => {
                write!(
                    f,
                    "cannot braid non-unit sectors {left:?} and {right:?} with braiding style {style:?}"
                )
            }
            Self::InvalidSector { sector } => write!(f, "invalid sector {sector:?}"),
            Self::InvalidMultiplicityIndex { value } => {
                write!(
                    f,
                    "invalid multiplicity index {value}; labels are one-based"
                )
            }
            Self::SectorMismatch { expected, actual } => {
                write!(f, "sector mismatch: expected {expected:?}, got {actual:?}")
            }
            Self::FusionRuleMismatch { expected, actual } => {
                write!(
                    f,
                    "fusion rule mismatch: expected {expected:?}, got {actual:?}"
                )
            }
            Self::MissingFusionRuleIdentity => write!(f, "fusion space has no bound rule identity"),
            Self::LegDegeneracyMismatch {
                sector,
                expected,
                actual,
            } => {
                write!(
                    f,
                    "leg degeneracy mismatch for sector {sector:?}: expected {expected}, got {actual}"
                )
            }
            Self::FusionChannelCount { left, right, count } => {
                write!(
                    f,
                    "expected one fusion channel for {left:?} x {right:?}, got {count}"
                )
            }
            Self::MalformedFusionTree { message } => {
                write!(f, "malformed fusion tree: {message}")
            }
            Self::ContractedLegDualityMismatch {
                lhs_axis,
                rhs_axis,
                lhs_is_dual,
                rhs_is_dual,
            } => {
                write!(
                    f,
                    "contracted fusion leg duality flags do not match: lhs axis {lhs_axis} \
                     (is_dual = {lhs_is_dual}) and rhs axis {rhs_axis} (is_dual = {rhs_is_dual}) \
                     must have opposite duality flags"
                )
            }
            Self::BlockCountMismatch { expected, actual } => {
                write!(f, "block count mismatch: expected {expected}, got {actual}")
            }
            Self::BlockIndexOutOfBounds { index, count } => {
                write!(f, "block index {index} is out of bounds for {count} blocks")
            }
            Self::OverlappingBlockStorage {
                first_block,
                second_block,
                offset,
            } => {
                write!(
                    f,
                    "logical blocks {first_block} and {second_block} overlap at storage offset {offset}"
                )
            }
            Self::DuplicateBlockKey { key } => {
                write!(f, "duplicate block key {key:?}")
            }
            Self::MixedBlockKeyKinds { expected, actual } => {
                write!(
                    f,
                    "mixed block key kinds: expected {expected}, got {actual}"
                )
            }
            Self::ExpectedFusionTreePairKey { actual } => {
                write!(f, "expected a fusion-tree pair key, got {actual}")
            }
            Self::MissingBlockKey { key } => {
                write!(f, "missing matching block for key {key:?}")
            }
            Self::MissingFusionSpace => write!(f, "tensor does not carry a fusion-tree space"),
            Self::FusionOutsideTable { message } => write!(f, "{message}"),
            Self::ElementCountOverflow => write!(f, "block element count overflow"),
            Self::OffsetOverflow { value } => {
                write!(f, "block offset {value} overflows addressable layout")
            }
            Self::StrideOverflow { value } => {
                write!(f, "block stride {value} overflows addressable layout")
            }
            Self::OutOfBounds => write!(f, "block view accesses outside the buffer"),
        }
    }
}

impl std::error::Error for CoreError {}

/// Failure while deriving a fusion space through checked finite-algebra
/// operations.
///
/// The established [`CoreError`] remains the structural error vocabulary.
/// Checked algebra failures stay separate so expert infallible APIs do not
/// acquire a new error variant or a stronger fusion-rule bound.
#[non_exhaustive]
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum CheckedFusionSpaceError {
    Core(Box<CoreError>),
    FusionAlgebra(Box<FusionAlgebraError>),
}

impl fmt::Display for CheckedFusionSpaceError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Core(error) => error.fmt(formatter),
            Self::FusionAlgebra(error) => error.fmt(formatter),
        }
    }
}

impl std::error::Error for CheckedFusionSpaceError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Core(error) => Some(error.as_ref()),
            Self::FusionAlgebra(error) => Some(error.as_ref()),
        }
    }
}

impl From<CoreError> for CheckedFusionSpaceError {
    fn from(error: CoreError) -> Self {
        Self::Core(Box::new(error))
    }
}

impl From<FusionAlgebraError> for CheckedFusionSpaceError {
    fn from(error: FusionAlgebraError) -> Self {
        Self::FusionAlgebra(Box::new(error))
    }
}

pub(crate) fn validate_layout(layout: BlockLayout<'_>) -> Result<(), CoreError> {
    if layout.shape.len() != layout.strides.len() {
        return Err(CoreError::RankMismatch {
            shape: layout.shape.len(),
            strides: layout.strides.len(),
        });
    }
    if layout.is_empty() {
        return if layout.offset <= layout.len {
            Ok(())
        } else {
            Err(CoreError::OutOfBounds)
        };
    }
    if layout.offset >= layout.len {
        return Err(CoreError::OutOfBounds);
    }
    let max_delta = max_offset_delta(layout.shape, layout.strides)?;
    let last = layout
        .offset
        .checked_add(max_delta)
        .ok_or_else(|| CoreError::OffsetOverflow {
            value: layout.offset,
        })?;
    if last < layout.len {
        Ok(())
    } else {
        Err(CoreError::OutOfBounds)
    }
}

fn max_offset_delta(shape: &[usize], strides: &[usize]) -> Result<usize, CoreError> {
    shape
        .iter()
        .zip(strides)
        .try_fold(0usize, |acc, (&dim, &stride)| {
            let steps = dim.saturating_sub(1);
            let delta = steps
                .checked_mul(stride)
                .ok_or_else(|| CoreError::StrideOverflow { value: stride })?;
            acc.checked_add(delta)
                .ok_or_else(|| CoreError::ElementCountOverflow)
        })
}

pub(crate) fn storage_end_exclusive(
    shape: &[usize],
    strides: &[usize],
    offset: usize,
) -> Result<usize, CoreError> {
    if shape.len() != strides.len() {
        return Err(CoreError::RankMismatch {
            shape: shape.len(),
            strides: strides.len(),
        });
    }
    if shape.contains(&0) {
        return Ok(offset);
    }
    let max_delta = max_offset_delta(shape, strides)?;
    offset
        .checked_add(max_delta)
        .and_then(|last| last.checked_add(1))
        .ok_or_else(|| CoreError::OffsetOverflow { value: offset })
}

/// Checked product of `dims` (an element count); the TeNeT-side authority
/// that operations and matrixalgebra map into their own error types.
#[doc(hidden)]
#[inline]
pub fn checked_product(dims: &[usize]) -> Result<usize, CoreError> {
    dims.iter().try_fold(1usize, |acc, &dim| {
        acc.checked_mul(dim)
            .ok_or_else(|| CoreError::ElementCountOverflow)
    })
}

/// Feeds the column-major strides of `shape`, starting at `first`, to `push`
/// in axis order and returns `first * product(shape)`.
///
/// Every running product, including the final one, must fit in `usize`;
/// otherwise `overflow()` is returned at the first overflowing axis, after the
/// strides before it were pushed.
#[doc(hidden)]
pub fn try_for_each_column_major_stride<E>(
    first: usize,
    shape: &[usize],
    mut push: impl FnMut(usize) -> Result<(), E>,
    overflow: impl Fn() -> E,
) -> Result<usize, E> {
    shape.iter().try_fold(first, |stride, &dim| {
        push(stride)?;
        stride.checked_mul(dim).ok_or_else(&overflow)
    })
}

/// Column-major strides of `shape`. Only the strides themselves must fit in
/// `usize`; the element count is checked separately by the layout.
#[doc(hidden)]
pub fn column_major_strides(shape: &[usize]) -> Result<Vec<usize>, CoreError> {
    let mut strides = Vec::with_capacity(shape.len());
    if let Some((_, leading)) = shape.split_last() {
        let last = try_for_each_column_major_stride(
            1,
            leading,
            |stride| {
                strides.push(stride);
                Ok(())
            },
            || CoreError::ElementCountOverflow,
        )?;
        strides.push(last);
    }
    Ok(strides)
}

#[cfg(test)]
mod layout_arithmetic_tests {
    use super::*;

    #[test]
    fn column_major_stride_walk_checks_every_running_product() {
        let mut pushed = Vec::new();
        let total = try_for_each_column_major_stride(
            3,
            &[2, 4, 5],
            |stride| {
                pushed.push(stride);
                Ok::<(), ()>(())
            },
            || (),
        );
        assert_eq!((total, pushed), (Ok(120), vec![3, 6, 24]));

        // The final product must fit even though no stride is pushed for it;
        // strides before the overflow are already pushed.
        let mut pushed = Vec::new();
        let total = try_for_each_column_major_stride(
            1,
            &[2, usize::MAX],
            |stride| {
                pushed.push(stride);
                Ok::<(), &str>(())
            },
            || "overflow",
        );
        assert_eq!((total, pushed), (Err("overflow"), vec![1, 2]));

        // A push error wins over a later overflow.
        let total = try_for_each_column_major_stride(
            1,
            &[usize::MAX, 4],
            |s| if s > 1 { Err(s) } else { Ok(()) },
            || 0,
        );
        assert_eq!(total, Err(usize::MAX));
    }

    #[test]
    fn column_major_strides_does_not_require_the_element_count_to_fit() {
        assert_eq!(column_major_strides(&[]), Ok(vec![]));
        assert_eq!(column_major_strides(&[2, usize::MAX]), Ok(vec![1, 2]));
        assert_eq!(
            column_major_strides(&[usize::MAX, 2, 1]),
            Err(CoreError::ElementCountOverflow)
        );
        assert_eq!(checked_product(&[]), Ok(1));
        assert_eq!(
            checked_product(&[usize::MAX, 2]),
            Err(CoreError::ElementCountOverflow)
        );
    }
}
