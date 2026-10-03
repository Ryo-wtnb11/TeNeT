use super::*;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct PreparedArtinStep {
    pub(crate) index: usize,
    pub(crate) inverse: bool,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct PreparedTreeBraid {
    pub(crate) permutation: SmallVec<[usize; 8]>,
    pub(crate) artin_steps: SmallVec<[PreparedArtinStep; 28]>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct SimplePreparedTreeBraid {
    permutation: Arc<[usize]>,
    artin_steps: Vec<PreparedArtinStep>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum UniqueBorrowedBraidLevels<'operation> {
    Symmetric,
    Explicit {
        codomain: &'operation [usize],
        domain: &'operation [usize],
    },
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct UniqueBorrowedTreePairBraid<'operation> {
    codomain_permutation: &'operation [usize],
    domain_permutation: &'operation [usize],
    pub(super) source_codomain_rank: usize,
    pub(super) source_domain_rank: usize,
    raw_axis_positions: Option<&'operation [usize]>,
    levels: UniqueBorrowedBraidLevels<'operation>,
}

#[cfg(test)]
thread_local! {
    static UNIQUE_BORROWED_POSITION_QUERIES: std::cell::Cell<usize> =
        const { std::cell::Cell::new(0) };
}

#[cfg(test)]
pub(crate) fn reset_unique_borrowed_position_queries() {
    UNIQUE_BORROWED_POSITION_QUERIES.with(|queries| queries.set(0));
}

#[cfg(test)]
pub(crate) fn unique_borrowed_position_queries() -> usize {
    UNIQUE_BORROWED_POSITION_QUERIES.with(std::cell::Cell::get)
}

impl<'operation> UniqueBorrowedTreePairBraid<'operation> {
    pub(super) fn permutation_at(&self, position: usize) -> usize {
        linearized_tree_pair_axis_at(
            self.codomain_permutation,
            self.domain_permutation,
            self.source_codomain_rank,
            self.source_domain_rank,
            position,
        )
    }

    fn linear_position_of_axis(&self, axis: usize) -> usize {
        #[cfg(test)]
        if self.raw_axis_positions.is_some() {
            UNIQUE_BORROWED_POSITION_QUERIES.with(|queries| queries.set(queries.get() + 1));
        }
        let logical_axis = if axis < self.source_codomain_rank {
            axis
        } else {
            self.source_codomain_rank + self.source_domain_rank + self.source_codomain_rank
                - 1
                - axis
        };
        // Why not materialize the linear permutation here: the operation
        // compiler already owns this inverse table, while direct core callers
        // intentionally retain their allocation-free compatibility path.
        let raw_position = self.raw_axis_positions.map_or_else(
            || {
                self.codomain_permutation
                    .iter()
                    .chain(self.domain_permutation)
                    .position(|&candidate| candidate == logical_axis)
                    .expect("validated tree-pair axis map contains every logical axis")
            },
            |positions| positions[logical_axis],
        );
        if raw_position < self.codomain_permutation.len() {
            raw_position
        } else {
            self.codomain_permutation.len() + self.source_codomain_rank + self.source_domain_rank
                - 1
                - raw_position
        }
    }

    fn level_at(&self, axis: usize) -> usize {
        match self.levels {
            UniqueBorrowedBraidLevels::Symmetric => {
                if axis < self.source_codomain_rank {
                    axis
                } else {
                    self.source_codomain_rank + self.source_domain_rank - 1 - axis
                        + self.source_codomain_rank
                }
            }
            UniqueBorrowedBraidLevels::Explicit { codomain, domain } => {
                if axis < self.source_codomain_rank {
                    codomain[axis]
                } else {
                    domain[self.source_codomain_rank + self.source_domain_rank - 1 - axis]
                }
            }
        }
    }

    pub(super) fn artin_steps(&self) -> UniqueBorrowedArtinSteps<'_> {
        UniqueBorrowedArtinSteps {
            braid: self,
            target: 0,
            source_axis: None,
            candidate_exclusive: 0,
            next_index: 0,
        }
    }
}

pub(crate) struct UniqueBorrowedArtinSteps<'operation> {
    braid: &'operation UniqueBorrowedTreePairBraid<'operation>,
    target: usize,
    source_axis: Option<usize>,
    candidate_exclusive: usize,
    next_index: usize,
}

impl Iterator for UniqueBorrowedArtinSteps<'_> {
    type Item = PreparedArtinStep;

    fn next(&mut self) -> Option<Self::Item> {
        let rank = self.braid.source_codomain_rank + self.braid.source_domain_rank;
        loop {
            if let Some(source_axis) = self.source_axis {
                while self.candidate_exclusive > 0 {
                    self.candidate_exclusive -= 1;
                    let candidate = self.candidate_exclusive;
                    if self.braid.linear_position_of_axis(candidate) > self.target {
                        self.next_index -= 1;
                        return Some(PreparedArtinStep {
                            index: self.next_index,
                            inverse: self.braid.level_at(candidate)
                                > self.braid.level_at(source_axis),
                        });
                    }
                }
                self.source_axis = None;
                self.target += 1;
                continue;
            }
            if self.target >= rank.saturating_sub(1) {
                return None;
            }
            let source_axis = self.braid.permutation_at(self.target);
            let crossings = (0..source_axis)
                .filter(|&candidate| self.braid.linear_position_of_axis(candidate) > self.target)
                .count();
            if crossings == 0 {
                self.target += 1;
            } else {
                self.source_axis = Some(source_axis);
                self.candidate_exclusive = source_axis;
                self.next_index = self.target + crossings;
            }
        }
    }
}

impl PreparedTreeBraid {
    pub(crate) fn new(
        permutation: &[usize],
        levels: &[usize],
        rank: usize,
    ) -> Result<Self, CoreError> {
        validate_permutation_inline(permutation, rank)?;
        debug_assert_eq!(levels.len(), rank);

        let mut work = SmallVec::<[usize; 8]>::from_slice(permutation);
        let mut current_levels = SmallVec::<[usize; 8]>::from_slice(levels);
        let mut artin_steps = SmallVec::new();
        for target in 0..rank.saturating_sub(1) {
            let source = work[target];
            for index in (target..source).rev() {
                artin_steps.push(PreparedArtinStep {
                    index,
                    inverse: current_levels[index] > current_levels[index + 1],
                });
                current_levels.swap(index, index + 1);
            }
            for item in work.iter_mut().take(rank).skip(target + 1) {
                if *item < source {
                    *item += 1;
                }
            }
            work[target] = target;
        }

        Ok(Self {
            permutation: SmallVec::from_slice(permutation),
            artin_steps,
        })
    }
}

impl SimplePreparedTreeBraid {
    fn from_prepared(braid: PreparedTreeBraid) -> Self {
        Self {
            permutation: Arc::from(braid.permutation.into_vec()),
            artin_steps: braid.artin_steps.into_vec(),
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum PreparedCycleDirection {
    Clockwise,
    Anticlockwise,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum PreparedTreePairFamily {
    BraidLike,
    Transpose,
}

const PREPARED_BRAID_BLOCK_FAMILY_ERROR: &str =
    "prepared tree-pair operation is incompatible with braid block execution";
const PREPARED_TRANSPOSE_BLOCK_FAMILY_ERROR: &str =
    "prepared tree-pair operation is incompatible with transpose block execution";

#[allow(clippy::large_enum_variant)]
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) enum PreparedTreePairPlan<'operation> {
    Identity,
    Repartition,
    Braid(PreparedTreeBraid),
    SimpleBraid(SimplePreparedTreeBraid),
    UniqueBraid(UniqueBorrowedTreePairBraid<'operation>),
    Transpose {
        direction: PreparedCycleDirection,
        count: usize,
    },
}

pub(crate) enum PreparedTreePairArtinSteps<'operation> {
    Owned(std::slice::Iter<'operation, PreparedArtinStep>),
    Unique(UniqueBorrowedArtinSteps<'operation>),
}

impl Iterator for PreparedTreePairArtinSteps<'_> {
    type Item = PreparedArtinStep;

    fn next(&mut self) -> Option<Self::Item> {
        match self {
            Self::Owned(steps) => steps.next().copied(),
            Self::Unique(steps) => steps.next(),
        }
    }
}

impl PreparedTreePairPlan<'_> {
    fn owned_braid_parts(&self) -> Option<(&[usize], &[PreparedArtinStep])> {
        match self {
            Self::Braid(braid) => Some((&braid.permutation, &braid.artin_steps)),
            Self::SimpleBraid(braid) => Some((&braid.permutation, &braid.artin_steps)),
            Self::Identity | Self::Repartition | Self::UniqueBraid(_) | Self::Transpose { .. } => {
                None
            }
        }
    }

    pub(crate) fn artin_steps(&self) -> Option<PreparedTreePairArtinSteps<'_>> {
        match self {
            Self::Braid(braid) => Some(PreparedTreePairArtinSteps::Owned(braid.artin_steps.iter())),
            Self::SimpleBraid(braid) => {
                Some(PreparedTreePairArtinSteps::Owned(braid.artin_steps.iter()))
            }
            Self::UniqueBraid(braid) => {
                Some(PreparedTreePairArtinSteps::Unique(braid.artin_steps()))
            }
            Self::Identity | Self::Repartition | Self::Transpose { .. } => None,
        }
    }
}

/// Rank-dependent preparation for one fusion-tree-pair operation.
///
/// The tensor-plan compiler may prepare this once and execute it for every
/// tree in one structural block. The representation is intentionally opaque:
/// operation validation and braid/cycle lowering are core semantics.
///
/// Why not expose the individual Artin or cycle steps: doing so would let
/// downstream crates duplicate TensorKit's domain reversal and inverse-level
/// conventions, recreating the semantic split this type removes.
#[doc(hidden)]
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PreparedTreePairOperation<'operation> {
    source_codomain_rank: usize,
    source_domain_rank: usize,
    pub(super) target_codomain_rank: usize,
    requires_symmetric_braiding: bool,
    family: PreparedTreePairFamily,
    pub(crate) plan: PreparedTreePairPlan<'operation>,
}

impl<'operation> PreparedTreePairOperation<'operation> {
    pub(crate) fn is_identity(&self) -> bool {
        matches!(self.plan, PreparedTreePairPlan::Identity)
    }

    #[doc(hidden)]
    pub fn into_compiler_owned_simple_braid(self) -> Self {
        let Self {
            source_codomain_rank,
            source_domain_rank,
            target_codomain_rank,
            requires_symmetric_braiding,
            family,
            plan,
        } = self;
        let plan = match plan {
            PreparedTreePairPlan::Braid(braid) => {
                PreparedTreePairPlan::SimpleBraid(SimplePreparedTreeBraid::from_prepared(braid))
            }
            plan => plan,
        };
        Self {
            source_codomain_rank,
            source_domain_rank,
            target_codomain_rank,
            requires_symmetric_braiding,
            family,
            plan,
        }
    }

    /// Validate operation metadata that depends only on total tensor rank.
    ///
    /// The exact codomain/domain split is categorical information and remains
    /// the responsibility of the operation-specific validators below.
    #[doc(hidden)]
    pub fn validate_rank_syntax(
        total_rank: usize,
        codomain_permutation: &[usize],
        domain_permutation: &[usize],
    ) -> Result<(), CoreError> {
        validate_tree_pair_axis_map_inline(codomain_permutation, domain_permutation, total_rank, 0)
    }

    pub fn prepare_braid<R>(
        rule: &R,
        source_codomain_rank: usize,
        source_domain_rank: usize,
        codomain_permutation: &'operation [usize],
        domain_permutation: &'operation [usize],
        codomain_levels: &'operation [usize],
        domain_levels: &'operation [usize],
    ) -> Result<Self, CoreError>
    where
        R: FusionRule,
    {
        Self::validate_braid_level_lengths(
            source_codomain_rank,
            source_domain_rank,
            codomain_levels,
            domain_levels,
        )?;
        if !rule.fusion_style().is_multiplicity_free() {
            return Err(CoreError::UnsupportedFusionStyle {
                expected: FusionStyleKind::Simple,
                actual: rule.fusion_style(),
            });
        }
        Self::prepare_braid_lowered(
            rule.fusion_style() == FusionStyleKind::Unique,
            None,
            source_codomain_rank,
            source_domain_rank,
            codomain_permutation,
            domain_permutation,
            codomain_levels,
            domain_levels,
        )
    }

    #[doc(hidden)]
    #[expect(
        clippy::too_many_arguments,
        reason = "the raw braid API exposes the two tree sides, levels, and optional storage-axis projection explicitly"
    )]
    pub fn prepare_braid_with_raw_axis_positions<R>(
        rule: &R,
        source_codomain_rank: usize,
        source_domain_rank: usize,
        codomain_permutation: &'operation [usize],
        domain_permutation: &'operation [usize],
        codomain_levels: &'operation [usize],
        domain_levels: &'operation [usize],
        raw_axis_positions: &'operation [usize],
    ) -> Result<Self, CoreError>
    where
        R: FusionRule,
    {
        Self::validate_braid_level_lengths(
            source_codomain_rank,
            source_domain_rank,
            codomain_levels,
            domain_levels,
        )?;
        if !rule.fusion_style().is_multiplicity_free() {
            return Err(CoreError::UnsupportedFusionStyle {
                expected: FusionStyleKind::Simple,
                actual: rule.fusion_style(),
            });
        }
        Self::prepare_braid_lowered(
            rule.fusion_style() == FusionStyleKind::Unique,
            Some(raw_axis_positions),
            source_codomain_rank,
            source_domain_rank,
            codomain_permutation,
            domain_permutation,
            codomain_levels,
            domain_levels,
        )
    }

    #[doc(hidden)]
    pub fn validate_braid_syntax(
        source_codomain_rank: usize,
        source_domain_rank: usize,
        codomain_permutation: &[usize],
        domain_permutation: &[usize],
        codomain_levels: &[usize],
        domain_levels: &[usize],
    ) -> Result<(), CoreError> {
        Self::validate_braid_level_lengths(
            source_codomain_rank,
            source_domain_rank,
            codomain_levels,
            domain_levels,
        )?;
        validate_tree_pair_axis_map_inline(
            codomain_permutation,
            domain_permutation,
            source_codomain_rank,
            source_domain_rank,
        )
    }

    #[expect(
        clippy::too_many_arguments,
        reason = "the private lowering step keeps the two tree sides, levels, and optional raw-axis projection explicit"
    )]
    fn prepare_braid_lowered(
        unique: bool,
        raw_axis_positions: Option<&'operation [usize]>,
        source_codomain_rank: usize,
        source_domain_rank: usize,
        codomain_permutation: &'operation [usize],
        domain_permutation: &'operation [usize],
        codomain_levels: &'operation [usize],
        domain_levels: &'operation [usize],
    ) -> Result<Self, CoreError> {
        let target_codomain_rank = codomain_permutation.len();
        if tree_pair_axis_map_is_identity(
            codomain_permutation,
            domain_permutation,
            source_codomain_rank,
            source_domain_rank,
        ) {
            Self::validate_raw_axis_positions(
                raw_axis_positions,
                codomain_permutation,
                domain_permutation,
                source_codomain_rank + source_domain_rank,
            )?;
            return Ok(Self {
                source_codomain_rank,
                source_domain_rank,
                target_codomain_rank,
                requires_symmetric_braiding: false,
                family: PreparedTreePairFamily::BraidLike,
                plan: PreparedTreePairPlan::Identity,
            });
        }

        let total_rank = source_codomain_rank + source_domain_rank;
        if unique {
            validate_tree_pair_axis_map_without_scratch(
                codomain_permutation,
                domain_permutation,
                source_codomain_rank,
                source_domain_rank,
            )?;
        } else {
            validate_tree_pair_axis_map_inline(
                codomain_permutation,
                domain_permutation,
                source_codomain_rank,
                source_domain_rank,
            )?;
        }
        Self::validate_raw_axis_positions(
            raw_axis_positions,
            codomain_permutation,
            domain_permutation,
            total_rank,
        )?;
        if (0..total_rank).all(|position| {
            linearized_tree_pair_axis_at(
                codomain_permutation,
                domain_permutation,
                source_codomain_rank,
                source_domain_rank,
                position,
            ) == position
        }) {
            return Ok(Self {
                source_codomain_rank,
                source_domain_rank,
                target_codomain_rank,
                requires_symmetric_braiding: false,
                family: PreparedTreePairFamily::BraidLike,
                plan: PreparedTreePairPlan::Repartition,
            });
        }

        if unique {
            return Ok(Self {
                source_codomain_rank,
                source_domain_rank,
                target_codomain_rank,
                requires_symmetric_braiding: false,
                family: PreparedTreePairFamily::BraidLike,
                plan: PreparedTreePairPlan::UniqueBraid(UniqueBorrowedTreePairBraid {
                    codomain_permutation,
                    domain_permutation,
                    source_codomain_rank,
                    source_domain_rank,
                    raw_axis_positions,
                    levels: UniqueBorrowedBraidLevels::Explicit {
                        codomain: codomain_levels,
                        domain: domain_levels,
                    },
                }),
            });
        }
        let permutation = materialize_linearized_tree_pair_permutation(
            codomain_permutation,
            domain_permutation,
            source_codomain_rank,
            source_domain_rank,
        );
        let mut levels = SmallVec::<[usize; 8]>::with_capacity(total_rank);
        levels.extend_from_slice(codomain_levels);
        levels.extend(domain_levels.iter().rev().copied());
        let braid = PreparedTreeBraid::new(&permutation, &levels, total_rank)?;
        Ok(Self {
            source_codomain_rank,
            source_domain_rank,
            target_codomain_rank,
            requires_symmetric_braiding: false,
            family: PreparedTreePairFamily::BraidLike,
            plan: PreparedTreePairPlan::Braid(braid),
        })
    }

    fn validate_braid_level_lengths(
        source_codomain_rank: usize,
        source_domain_rank: usize,
        codomain_levels: &[usize],
        domain_levels: &[usize],
    ) -> Result<(), CoreError> {
        if codomain_levels.len() != source_codomain_rank {
            return Err(CoreError::DimensionMismatch {
                expected: source_codomain_rank,
                actual: codomain_levels.len(),
            });
        }
        if domain_levels.len() != source_domain_rank {
            return Err(CoreError::DimensionMismatch {
                expected: source_domain_rank,
                actual: domain_levels.len(),
            });
        }
        Ok(())
    }

    fn validate_raw_axis_positions(
        raw_axis_positions: Option<&[usize]>,
        codomain_permutation: &[usize],
        domain_permutation: &[usize],
        total_rank: usize,
    ) -> Result<(), CoreError> {
        if let Some(raw_axis_positions) = raw_axis_positions {
            if raw_axis_positions.len() != total_rank {
                return Err(CoreError::DimensionMismatch {
                    expected: total_rank,
                    actual: raw_axis_positions.len(),
                });
            }
            for position in 0..total_rank {
                let logical_axis =
                    raw_tree_pair_axis_at(codomain_permutation, domain_permutation, position);
                if raw_axis_positions[logical_axis] != position {
                    return Err(CoreError::InconsistentAxisPosition {
                        logical_axis,
                        expected_position: position,
                        actual_position: raw_axis_positions[logical_axis],
                    });
                }
            }
        }
        Ok(())
    }

    pub fn prepare_permute<R>(
        rule: &R,
        source_codomain_rank: usize,
        source_domain_rank: usize,
        codomain_permutation: &'operation [usize],
        domain_permutation: &'operation [usize],
    ) -> Result<Self, CoreError>
    where
        R: FusionRule,
    {
        if !rule.braiding_style().is_symmetric() {
            return Err(CoreError::UnsupportedBraidingStyle {
                expected: "symmetric braiding",
                actual: rule.braiding_style(),
            });
        }
        if !rule.fusion_style().is_multiplicity_free() {
            return Err(CoreError::UnsupportedFusionStyle {
                expected: FusionStyleKind::Simple,
                actual: rule.fusion_style(),
            });
        }
        Self::prepare_permute_lowered(
            rule.fusion_style() == FusionStyleKind::Unique,
            None,
            source_codomain_rank,
            source_domain_rank,
            codomain_permutation,
            domain_permutation,
        )
    }

    #[doc(hidden)]
    pub fn prepare_permute_with_raw_axis_positions<R>(
        rule: &R,
        source_codomain_rank: usize,
        source_domain_rank: usize,
        codomain_permutation: &'operation [usize],
        domain_permutation: &'operation [usize],
        raw_axis_positions: &'operation [usize],
    ) -> Result<Self, CoreError>
    where
        R: FusionRule,
    {
        if !rule.braiding_style().is_symmetric() {
            return Err(CoreError::UnsupportedBraidingStyle {
                expected: "symmetric braiding",
                actual: rule.braiding_style(),
            });
        }
        if !rule.fusion_style().is_multiplicity_free() {
            return Err(CoreError::UnsupportedFusionStyle {
                expected: FusionStyleKind::Simple,
                actual: rule.fusion_style(),
            });
        }
        Self::prepare_permute_lowered(
            rule.fusion_style() == FusionStyleKind::Unique,
            Some(raw_axis_positions),
            source_codomain_rank,
            source_domain_rank,
            codomain_permutation,
            domain_permutation,
        )
    }

    #[doc(hidden)]
    pub fn validate_permute_syntax(
        source_codomain_rank: usize,
        source_domain_rank: usize,
        codomain_permutation: &[usize],
        domain_permutation: &[usize],
    ) -> Result<(), CoreError> {
        validate_tree_pair_axis_map_inline(
            codomain_permutation,
            domain_permutation,
            source_codomain_rank,
            source_domain_rank,
        )
    }

    fn prepare_permute_lowered(
        unique: bool,
        raw_axis_positions: Option<&'operation [usize]>,
        source_codomain_rank: usize,
        source_domain_rank: usize,
        codomain_permutation: &'operation [usize],
        domain_permutation: &'operation [usize],
    ) -> Result<Self, CoreError> {
        let target_codomain_rank = codomain_permutation.len();
        if tree_pair_axis_map_is_identity(
            codomain_permutation,
            domain_permutation,
            source_codomain_rank,
            source_domain_rank,
        ) {
            Self::validate_raw_axis_positions(
                raw_axis_positions,
                codomain_permutation,
                domain_permutation,
                source_codomain_rank + source_domain_rank,
            )?;
            return Ok(Self {
                source_codomain_rank,
                source_domain_rank,
                target_codomain_rank,
                requires_symmetric_braiding: true,
                family: PreparedTreePairFamily::BraidLike,
                plan: PreparedTreePairPlan::Identity,
            });
        }

        let total_rank = source_codomain_rank + source_domain_rank;
        if unique {
            validate_tree_pair_axis_map_without_scratch(
                codomain_permutation,
                domain_permutation,
                source_codomain_rank,
                source_domain_rank,
            )?;
        } else {
            validate_tree_pair_axis_map_inline(
                codomain_permutation,
                domain_permutation,
                source_codomain_rank,
                source_domain_rank,
            )?;
        }
        Self::validate_raw_axis_positions(
            raw_axis_positions,
            codomain_permutation,
            domain_permutation,
            total_rank,
        )?;
        if (0..total_rank).all(|position| {
            linearized_tree_pair_axis_at(
                codomain_permutation,
                domain_permutation,
                source_codomain_rank,
                source_domain_rank,
                position,
            ) == position
        }) {
            return Ok(Self {
                source_codomain_rank,
                source_domain_rank,
                target_codomain_rank,
                requires_symmetric_braiding: true,
                family: PreparedTreePairFamily::BraidLike,
                plan: PreparedTreePairPlan::Repartition,
            });
        }
        if unique {
            return Ok(Self {
                source_codomain_rank,
                source_domain_rank,
                target_codomain_rank,
                requires_symmetric_braiding: true,
                family: PreparedTreePairFamily::BraidLike,
                plan: PreparedTreePairPlan::UniqueBraid(UniqueBorrowedTreePairBraid {
                    codomain_permutation,
                    domain_permutation,
                    source_codomain_rank,
                    source_domain_rank,
                    raw_axis_positions,
                    levels: UniqueBorrowedBraidLevels::Symmetric,
                }),
            });
        }
        let permutation = materialize_linearized_tree_pair_permutation(
            codomain_permutation,
            domain_permutation,
            source_codomain_rank,
            source_domain_rank,
        );
        let mut levels = SmallVec::<[usize; 8]>::with_capacity(total_rank);
        levels.extend(0..source_codomain_rank);
        levels.extend((source_codomain_rank..source_codomain_rank + source_domain_rank).rev());
        let braid = PreparedTreeBraid::new(&permutation, &levels, total_rank)?;
        Ok(Self {
            source_codomain_rank,
            source_domain_rank,
            target_codomain_rank,
            requires_symmetric_braiding: true,
            family: PreparedTreePairFamily::BraidLike,
            plan: PreparedTreePairPlan::Braid(braid),
        })
    }

    pub fn prepare_transpose(
        source_codomain_rank: usize,
        source_domain_rank: usize,
        codomain_permutation: &[usize],
        domain_permutation: &[usize],
    ) -> Result<Self, CoreError> {
        let permutation = Self::validated_transpose_permutation(
            codomain_permutation,
            domain_permutation,
            source_codomain_rank,
            source_domain_rank,
        )?;
        let total_rank = source_codomain_rank + source_domain_rank;
        let target_codomain_rank = codomain_permutation.len();
        let Some(position) = permutation.iter().position(|&axis| axis == 0) else {
            return Ok(Self {
                source_codomain_rank,
                source_domain_rank,
                target_codomain_rank,
                requires_symmetric_braiding: false,
                family: PreparedTreePairFamily::Transpose,
                plan: PreparedTreePairPlan::Identity,
            });
        };
        let plan = match transpose_cycles(position, total_rank) {
            None => PreparedTreePairPlan::Repartition,
            Some((direction, count)) => PreparedTreePairPlan::Transpose { direction, count },
        };
        Ok(Self {
            source_codomain_rank,
            source_domain_rank,
            target_codomain_rank,
            requires_symmetric_braiding: false,
            family: PreparedTreePairFamily::Transpose,
            plan,
        })
    }

    #[doc(hidden)]
    pub fn validate_transpose_syntax(
        source_codomain_rank: usize,
        source_domain_rank: usize,
        codomain_permutation: &[usize],
        domain_permutation: &[usize],
    ) -> Result<(), CoreError> {
        validate_cyclic_tree_pair_axis_map_inline(
            codomain_permutation,
            domain_permutation,
            source_codomain_rank,
            source_domain_rank,
        )
    }

    fn validated_transpose_permutation(
        codomain_permutation: &[usize],
        domain_permutation: &[usize],
        source_codomain_rank: usize,
        source_domain_rank: usize,
    ) -> Result<SmallVec<[usize; 8]>, CoreError> {
        validate_cyclic_tree_pair_axis_map_inline(
            codomain_permutation,
            domain_permutation,
            source_codomain_rank,
            source_domain_rank,
        )?;
        Ok(materialize_linearized_tree_pair_permutation(
            codomain_permutation,
            domain_permutation,
            source_codomain_rank,
            source_domain_rank,
        ))
    }

    fn validate_source(&self, tree_pair: &FusionTreePairKey) -> Result<(), CoreError> {
        let actual_codomain_rank = tree_pair.codomain_tree().uncoupled().len();
        if actual_codomain_rank != self.source_codomain_rank {
            return Err(CoreError::DimensionMismatch {
                expected: self.source_codomain_rank,
                actual: actual_codomain_rank,
            });
        }
        let actual_domain_rank = tree_pair.domain_tree().uncoupled().len();
        if actual_domain_rank != self.source_domain_rank {
            return Err(CoreError::DimensionMismatch {
                expected: self.source_domain_rank,
                actual: actual_domain_rank,
            });
        }
        Ok(())
    }

    fn validate_rule_capabilities<R>(&self, rule: &R) -> Result<(), CoreError>
    where
        R: FusionRule,
    {
        if self.requires_symmetric_braiding && !rule.braiding_style().is_symmetric() {
            return Err(CoreError::UnsupportedBraidingStyle {
                expected: "symmetric braiding",
                actual: rule.braiding_style(),
            });
        }
        Ok(())
    }

    pub(crate) fn validate_block_preflight<R>(
        &self,
        rule: &R,
        expected_family: PreparedTreePairFamily,
    ) -> Result<(), CoreError>
    where
        R: FusionRule,
    {
        if self.family != expected_family {
            return Err(CoreError::MalformedFusionTree {
                message: match expected_family {
                    PreparedTreePairFamily::BraidLike => PREPARED_BRAID_BLOCK_FAMILY_ERROR,
                    PreparedTreePairFamily::Transpose => PREPARED_TRANSPOSE_BLOCK_FAMILY_ERROR,
                },
            });
        }
        self.validate_rule_capabilities(rule)
    }

    pub(crate) fn validate_source_split(
        &self,
        source_codomain_rank: usize,
        source_domain_rank: usize,
    ) -> Result<(), CoreError> {
        if source_codomain_rank != self.source_codomain_rank {
            return Err(CoreError::DimensionMismatch {
                expected: self.source_codomain_rank,
                actual: source_codomain_rank,
            });
        }
        if source_domain_rank != self.source_domain_rank {
            return Err(CoreError::DimensionMismatch {
                expected: self.source_domain_rank,
                actual: source_domain_rank,
            });
        }
        Ok(())
    }
}

impl<'operation> PreparedTreePairOperation<'operation> {
    /// Execute this prepared operation on one multiplicity-free tree pair.
    ///
    /// `tree_pair` follows [`FusionTreePairKey::validate_for_rule`]'s
    /// provider-domain precondition.
    pub fn execute_multiplicity_free<R>(
        &self,
        rule: &R,
        tree_pair: &FusionTreePairKey,
    ) -> Result<Vec<(FusionTreePairKey, R::Scalar)>, CoreError>
    where
        R: MultiplicityFreeRigidSymbols,
        R::Scalar: Clone + Add<Output = R::Scalar> + Mul<Output = R::Scalar>,
    {
        self.validate_source(tree_pair)?;
        self.validate_rule_capabilities(rule)?;
        if !rule.fusion_style().is_multiplicity_free() {
            return Err(CoreError::UnsupportedFusionStyle {
                expected: FusionStyleKind::Simple,
                actual: rule.fusion_style(),
            });
        }
        let validated = validate_fusion_tree_pair_for_rule(rule, tree_pair)?;
        self.execute_multiplicity_free_validated(validated)
    }

    pub(crate) fn execute_multiplicity_free_proven<R>(
        &self,
        validated: ValidatedFusionTreePair<'_, R>,
    ) -> Result<Vec<(FusionTreePairKey, R::Scalar)>, CoreError>
    where
        R: MultiplicityFreeRigidSymbols,
        R::Scalar: Clone + Add<Output = R::Scalar> + Mul<Output = R::Scalar>,
    {
        self.validate_source(validated.key)?;
        self.validate_rule_capabilities(validated.rule)?;
        if !validated.rule.fusion_style().is_multiplicity_free() {
            return Err(CoreError::UnsupportedFusionStyle {
                expected: FusionStyleKind::Simple,
                actual: validated.rule.fusion_style(),
            });
        }
        self.execute_multiplicity_free_validated(validated)
    }

    fn execute_multiplicity_free_validated<R>(
        &self,
        validated: ValidatedFusionTreePair<'_, R>,
    ) -> Result<Vec<(FusionTreePairKey, R::Scalar)>, CoreError>
    where
        R: MultiplicityFreeRigidSymbols,
        R::Scalar: Clone + Add<Output = R::Scalar> + Mul<Output = R::Scalar>,
    {
        let rule = validated.rule;
        let tree_pair = validated.key;
        match &self.plan {
            PreparedTreePairPlan::Identity => Ok(vec![(tree_pair.clone(), R::Scalar::one())]),
            PreparedTreePairPlan::Repartition => multiplicity_free_repartition_tree_pair_validated(
                validated,
                self.target_codomain_rank,
            ),
            plan @ (PreparedTreePairPlan::Braid(_) | PreparedTreePairPlan::SimpleBraid(_)) => {
                let (permutation, artin_steps) = plan
                    .owned_braid_parts()
                    .expect("owned braid plan exposes its schedule");
                self.multiplicity_free_braid_via_codomain(validated, |rule, codomain| {
                    execute_multiplicity_free_tree_braid(rule, codomain, permutation, artin_steps)
                })
            }
            PreparedTreePairPlan::UniqueBraid(braid) => {
                self.multiplicity_free_braid_via_codomain(validated, |rule, codomain| {
                    execute_multiplicity_free_tree_braid_steps(rule, codomain, braid.artin_steps())
                })
            }
            PreparedTreePairPlan::Transpose { direction, count } => {
                let current = multiplicity_free_repartition_tree_pair_validated(
                    validated,
                    self.target_codomain_rank,
                )?;
                run_cycles(current, Some((*direction, *count)), |terms, direction| {
                    terms.then(|key| match direction {
                        PreparedCycleDirection::Clockwise => {
                            multiplicity_free_cycle_clockwise_tree_pair(rule, key)
                        }
                        PreparedCycleDirection::Anticlockwise => {
                            multiplicity_free_cycle_anticlockwise_tree_pair(rule, key)
                        }
                    })
                })
            }
        }
    }

    /// Execute this prepared operation on one unique-fusion tree pair.
    ///
    /// `tree_pair` follows [`FusionTreePairKey::validate_for_rule`]'s
    /// provider-domain precondition.
    pub fn execute_unique_rigid<R>(
        &self,
        rule: &R,
        tree_pair: &FusionTreePairKey,
    ) -> Result<(FusionTreePairKey, R::Scalar), CoreError>
    where
        R: MultiplicityFreeRigidSymbols,
        R::Scalar: Clone + Mul<Output = R::Scalar>,
    {
        self.validate_source(tree_pair)?;
        self.validate_rule_capabilities(rule)?;
        if rule.fusion_style() != FusionStyleKind::Unique {
            return Err(CoreError::UnsupportedFusionStyle {
                expected: FusionStyleKind::Unique,
                actual: rule.fusion_style(),
            });
        }
        let validated = validate_fusion_tree_pair_for_rule(rule, tree_pair)?;
        self.execute_unique_rigid_validated(validated)
    }

    pub(crate) fn execute_unique_rigid_proven<R>(
        &self,
        validated: ValidatedFusionTreePair<'_, R>,
    ) -> Result<(FusionTreePairKey, R::Scalar), CoreError>
    where
        R: MultiplicityFreeRigidSymbols,
        R::Scalar: Clone + Mul<Output = R::Scalar>,
    {
        self.validate_source(validated.key)?;
        self.validate_rule_capabilities(validated.rule)?;
        if validated.rule.fusion_style() != FusionStyleKind::Unique {
            return Err(CoreError::UnsupportedFusionStyle {
                expected: FusionStyleKind::Unique,
                actual: validated.rule.fusion_style(),
            });
        }
        self.execute_unique_rigid_validated(validated)
    }

    /// Braid as TensorKit `braid((f₁, f₂), p, levels)` does
    /// (`braiding_manipulations.jl:281`, via `fsbraid`: pair branch
    /// `:302-309`, block branch `:317-331`): repartition every leg into the
    /// codomain, braid that tree, and repartition to the target split.
    fn multiplicity_free_braid_via_codomain<R, F>(
        &self,
        validated: ValidatedFusionTreePair<'_, R>,
        braid_codomain: F,
    ) -> Result<Vec<(FusionTreePairKey, R::Scalar)>, CoreError>
    where
        R: MultiplicityFreeRigidSymbols,
        R::Scalar: Clone + Add<Output = R::Scalar> + Mul<Output = R::Scalar>,
        F: Fn(&R, &FusionTreeKey) -> Result<Vec<(FusionTreeKey, R::Scalar)>, CoreError>,
    {
        let rule = validated.rule;
        braid_terms_via_codomain(
            vec![(validated.key.clone(), R::Scalar::one())],
            self.source_codomain_rank,
            self.source_codomain_rank + self.source_domain_rank,
            self.target_codomain_rank,
            |terms, bend| multiplicity_free_bend_terms(rule, terms, bend),
            |codomain| braid_codomain(rule, codomain),
        )
    }

    /// Unique-fusion form of [`Self::multiplicity_free_braid_via_codomain`]:
    /// every step has exactly one output, so coefficients multiply directly.
    fn unique_rigid_braid_via_codomain<R, F>(
        &self,
        validated: ValidatedFusionTreePair<'_, R>,
        braid_codomain: F,
    ) -> Result<(FusionTreePairKey, R::Scalar), CoreError>
    where
        R: MultiplicityFreeRigidSymbols,
        R::Scalar: Clone + Mul<Output = R::Scalar>,
        F: FnOnce(&R, &FusionTreeKey) -> Result<(FusionTreeKey, R::Scalar), CoreError>,
    {
        let rule = validated.rule;
        let all_rank = self.source_codomain_rank + self.source_domain_rank;
        let (all_codomain, repartition_to_all) =
            unique_rigid_repartition_tree_pair_validated(validated, all_rank)?;
        let (braided_tree, braid_coefficient) = braid_codomain(rule, all_codomain.codomain_tree())?;
        let braided_pair =
            FusionTreePairKey::pair(braided_tree, all_codomain.domain_tree().clone());
        let (destination, repartition_back) = unique_rigid_repartition_tree_pair_unchecked(
            rule,
            &braided_pair,
            self.target_codomain_rank,
        )?;
        Ok((
            destination,
            repartition_to_all * braid_coefficient * repartition_back,
        ))
    }

    fn execute_unique_rigid_validated<R>(
        &self,
        validated: ValidatedFusionTreePair<'_, R>,
    ) -> Result<(FusionTreePairKey, R::Scalar), CoreError>
    where
        R: MultiplicityFreeRigidSymbols,
        R::Scalar: Clone + Mul<Output = R::Scalar>,
    {
        let rule = validated.rule;
        match &self.plan {
            PreparedTreePairPlan::Identity => Ok((validated.key.clone(), R::Scalar::one())),
            PreparedTreePairPlan::Repartition => {
                unique_rigid_repartition_tree_pair_validated(validated, self.target_codomain_rank)
            }
            plan @ (PreparedTreePairPlan::Braid(_) | PreparedTreePairPlan::SimpleBraid(_)) => {
                let (permutation, artin_steps) = plan
                    .owned_braid_parts()
                    .expect("owned braid plan exposes its schedule");
                self.unique_rigid_braid_via_codomain(validated, |rule, codomain| {
                    execute_unique_tree_braid(rule, codomain, permutation, artin_steps)
                })
            }
            PreparedTreePairPlan::UniqueBraid(braid) => self
                .unique_rigid_braid_via_codomain(validated, |rule, codomain| {
                    execute_unique_tree_braid_borrowed(rule, codomain, braid)
                }),
            PreparedTreePairPlan::Transpose { direction, count } => {
                let current = unique_rigid_repartition_tree_pair_validated(
                    validated,
                    self.target_codomain_rank,
                )?;
                run_cycles(current, Some((*direction, *count)), |term, direction| {
                    term.then(|key| match direction {
                        PreparedCycleDirection::Clockwise => {
                            unique_rigid_cycle_clockwise_tree_pair(rule, key)
                        }
                        PreparedCycleDirection::Anticlockwise => {
                            unique_rigid_cycle_anticlockwise_tree_pair(rule, key)
                        }
                    })
                })
            }
        }
    }
}
