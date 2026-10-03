use super::*;

#[doc(hidden)]
#[non_exhaustive]
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum FusionSpaceAdmission {
    Unbound,
    Subset(RuleIdentity),
    Complete(RuleIdentity),
}

impl FusionSpaceAdmission {
    #[doc(hidden)]
    pub fn rule_identity(&self) -> Option<&RuleIdentity> {
        match self {
            Self::Unbound => None,
            Self::Subset(identity) | Self::Complete(identity) => Some(identity),
        }
    }
}

#[derive(Clone, Debug)]
pub struct FusionTensorMapSpace<const NOUT: usize, const NIN: usize> {
    dense_space: TensorMapSpace<NOUT, NIN>,
    homspace: Arc<FusionTreeHomSpace>,
    subblock_structure: Arc<BlockStructure>,
    admission: FusionSpaceAdmission,
}

impl<const NOUT: usize, const NIN: usize> PartialEq for FusionTensorMapSpace<NOUT, NIN> {
    fn eq(&self, other: &Self) -> bool {
        self.dense_space == other.dense_space
            && self.homspace == other.homspace
            && self.subblock_structure == other.subblock_structure
            && self.admission.rule_identity() == other.admission.rule_identity()
    }
}

impl<const NOUT: usize, const NIN: usize> Eq for FusionTensorMapSpace<NOUT, NIN> {}

impl<const NOUT: usize, const NIN: usize> FusionTensorMapSpace<NOUT, NIN> {
    /// Expert compatibility constructor for an explicit caller-selected block
    /// structure.
    ///
    /// Use this at an import or custom-layout boundary where the
    /// [`BlockStructure`] has already been prepared. It does not derive the
    /// ordinary operation-output layout from the final hom space.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenet_core::{
    ///     BlockStructure, FusionTensorMapSpace, FusionTreeHomSpace, TensorMapSpace,
    /// };
    ///
    /// let dense = TensorMapSpace::<1, 0>::from_dims([2], []).unwrap();
    /// let hom = FusionTreeHomSpace::from_sector_ids([(0, 2)], std::iter::empty::<(usize, usize)>());
    /// let structure = BlockStructure::packed_column_major(1, [vec![2]]).unwrap();
    ///
    /// let space = FusionTensorMapSpace::new_unbound(dense, hom, structure).unwrap();
    /// assert_eq!(space.required_len().unwrap(), 2);
    /// ```
    pub fn new_unbound(
        dense_space: TensorMapSpace<NOUT, NIN>,
        homspace: FusionTreeHomSpace,
        subblock_structure: BlockStructure,
    ) -> Result<Self, CoreError> {
        Self::from_shared_subblock_structure(
            dense_space,
            homspace,
            subblock_structure.into_shared(),
        )
    }

    /// Shared-handle variant of [`Self::new_unbound`].
    ///
    /// This is the same expert compatibility/import boundary: the caller has
    /// already selected the block layout. This constructor checks that
    /// the hom-space and structure ranks match and that logical block footprints
    /// do not overlap. Key, sector, duality, and logical-shape admission remain
    /// the responsibility of [`Self::try_bind_rule`].
    pub fn from_shared_subblock_structure(
        dense_space: TensorMapSpace<NOUT, NIN>,
        homspace: FusionTreeHomSpace,
        subblock_structure: Arc<BlockStructure>,
    ) -> Result<Self, CoreError> {
        Self::validate_homspace_rank(&homspace)?;
        Self::validate_structure_rank(&subblock_structure)?;
        if subblock_structure
            .coupled_sector_regions(NOUT)?
            .is_none()
        {
            validate_block_storage_injective(&subblock_structure)?;
        }
        Ok(Self::from_admitted_shared_subblock_structure(
            dense_space,
            homspace,
            subblock_structure,
        ))
    }

    fn from_admitted_shared_subblock_structure(
        dense_space: TensorMapSpace<NOUT, NIN>,
        homspace: FusionTreeHomSpace,
        subblock_structure: Arc<BlockStructure>,
    ) -> Self {
        let subblock_structure = BlockStructure::canonicalize_shared(subblock_structure);
        Self {
            dense_space,
            homspace: Arc::new(homspace),
            subblock_structure,
            admission: FusionSpaceAdmission::Unbound,
        }
    }

    /// Expert compatibility constructor for caller-supplied fusion-tree
    /// subblock shapes.
    ///
    /// The shapes are given per fusion-tree **subblock** (one entry per
    /// fusion-tree key, in key order), not per coupled-sector matrix block.
    /// This mirrors TensorKit's block/subblock distinction: a coupled-sector
    /// matrix block is assembled from these tree-resolved degeneracy shapes.
    /// Ordinary TeNeT operation outputs instead derive their layout from the
    /// final hom space that already owns the leg degeneracies.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenet_core::{
    ///     FusionTensorMapSpace, FusionTreeHomSpace, TensorMapSpace, Z2FusionRule, Z2Irrep,
    /// };
    ///
    /// let rule = Z2FusionRule;
    /// let space = FusionTensorMapSpace::from_degeneracy_shapes(
    ///     TensorMapSpace::<1, 1>::from_dims([1], [1]).unwrap(),
    ///     FusionTreeHomSpace::from_sectors([(Z2Irrep::EVEN, 1)], [(Z2Irrep::EVEN, 1)]),
    ///     &rule,
    ///     [vec![1, 1]],
    /// )
    /// .unwrap();
    /// assert_eq!(space.required_len().unwrap(), 1);
    /// ```
    pub fn from_degeneracy_shapes<R, Shapes>(
        dense_space: TensorMapSpace<NOUT, NIN>,
        homspace: FusionTreeHomSpace,
        rule: &R,
        shapes: Shapes,
    ) -> Result<Self, CoreError>
    where
        R: MultiplicityFreeFusionRule,
        Shapes: IntoIterator,
        Shapes::Item: Into<Vec<usize>>,
    {
        Self::from_degeneracy_shapes_coupled(dense_space, homspace, rule, shapes)
    }

    /// Expert compatibility constructor for a TensorKit-style coupled-sector
    /// matrix layout from caller-supplied subblock shapes.
    ///
    /// Each coupled sector stores one contiguous column-major matrix whose
    /// rows enumerate (codomain fusion tree × codomain degeneracies) and whose
    /// columns enumerate (domain fusion tree × domain degeneracies). Fusion
    /// tree subblocks are strided views into that matrix, so the canonical
    /// (codomain | domain) matricization needs no packing. Block keys and
    /// their order are identical to [`Self::from_degeneracy_shapes`]; only
    /// strides and offsets differ. This explicit shape list is an
    /// import/compatibility boundary, not the ordinary final-hom-space output
    /// path.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenet_core::{
    ///     FusionProductSpace, FusionTensorMapSpace, FusionTreeHomSpace, SectorLeg,
    ///     TensorMapSpace, Z2FusionRule, Z2Irrep,
    /// };
    ///
    /// let rule = Z2FusionRule;
    /// let leg = || SectorLeg::new([(Z2Irrep::EVEN, 1), (Z2Irrep::ODD, 1)], false);
    /// let space = FusionTensorMapSpace::from_degeneracy_shapes_coupled(
    ///     TensorMapSpace::<1, 1>::from_dims([2], [2]).unwrap(),
    ///     FusionTreeHomSpace::new(
    ///         FusionProductSpace::new([leg()]),
    ///         FusionProductSpace::new([leg()]),
    ///     ),
    ///     &rule,
    ///     [vec![1, 1], vec![1, 1]],
    /// )
    /// .unwrap();
    /// assert_eq!(space.required_len().unwrap(), 2);
    /// ```
    pub fn from_degeneracy_shapes_coupled<R, Shapes>(
        dense_space: TensorMapSpace<NOUT, NIN>,
        homspace: FusionTreeHomSpace,
        rule: &R,
        shapes: Shapes,
    ) -> Result<Self, CoreError>
    where
        R: MultiplicityFreeFusionRule,
        Shapes: IntoIterator,
        Shapes::Item: Into<Vec<usize>>,
    {
        Self::validate_homspace_rank(&homspace)?;
        let subblock_structure = homspace.coupled_subblock_structure(rule, NOUT, shapes)?;
        Self::validate_structure_rank(&subblock_structure)?;
        Ok(Self::from_admitted_shared_subblock_structure(
            dense_space,
            homspace,
            subblock_structure,
        )
        .with_complete_rule(rule.rule_identity()))
    }

    fn validate_homspace_rank(homspace: &FusionTreeHomSpace) -> Result<(), CoreError> {
        if homspace.codomain().len() != NOUT {
            return Err(CoreError::StructureRankMismatch {
                expected: NOUT,
                actual: homspace.codomain().len(),
            });
        }
        if homspace.domain().len() != NIN {
            return Err(CoreError::StructureRankMismatch {
                expected: NIN,
                actual: homspace.domain().len(),
            });
        }
        Ok(())
    }

    fn validate_structure_rank(subblock_structure: &BlockStructure) -> Result<(), CoreError> {
        let rank = NOUT + NIN;
        if subblock_structure.rank() != rank {
            return Err(CoreError::StructureRankMismatch {
                expected: rank,
                actual: subblock_structure.rank(),
            });
        }
        Ok(())
    }

    /// Builds the adjoint metadata view without rechecking its physical
    /// footprint.
    ///
    /// Why not route through the expert constructor: swapping the two sides
    /// only permutes each admitted block's shape/stride axes, so its exact set of
    /// storage offsets is unchanged.
    pub fn adjoint_view(&self) -> Result<FusionTensorMapSpace<NIN, NOUT>, CoreError> {
        let dense_space = TensorMapSpace::<NIN, NOUT>::from_dims(
            std::array::from_fn(|index| self.dense_space.domain().dims()[index]),
            std::array::from_fn(|index| self.dense_space.codomain().dims()[index]),
        )?;
        let homspace = FusionTreeHomSpace::new(
            self.homspace.domain().clone(),
            self.homspace.codomain().clone(),
        );
        let rank = NOUT + NIN;
        let mut blocks = Vec::with_capacity(self.subblock_structure.block_count());
        for index in 0..self.subblock_structure.block_count() {
            let block = self.subblock_structure.block(index)?;
            let key = match block.key() {
                BlockKey::Dense => BlockKey::Dense,
                BlockKey::Opaque(key) => BlockKey::Opaque(key.clone()),
                BlockKey::FusionTree(tree) => BlockKey::FusionTree(FusionTreePairKey::pair(
                    tree.domain_tree().clone(),
                    tree.codomain_tree().clone(),
                )),
            };
            let mut shape = Vec::with_capacity(rank);
            shape.extend_from_slice(&block.shape()[NOUT..]);
            shape.extend_from_slice(&block.shape()[..NOUT]);
            let mut strides = Vec::with_capacity(rank);
            strides.extend_from_slice(&block.strides()[NOUT..]);
            strides.extend_from_slice(&block.strides()[..NOUT]);
            blocks.push(BlockSpec::with_key(key, shape, strides, block.offset())?);
        }
        let structure =
            BlockStructure::from_blocks_with_rank(rank, blocks)?.into_shared();
        Ok(FusionTensorMapSpace::<NIN, NOUT> {
            dense_space,
            homspace: Arc::new(homspace),
            subblock_structure: structure,
            admission: self.admission.clone(),
        })
    }

    #[inline]
    pub fn dense_space(&self) -> &TensorMapSpace<NOUT, NIN> {
        &self.dense_space
    }

    #[inline]
    pub fn homspace(&self) -> &FusionTreeHomSpace {
        &self.homspace
    }

    /// Shared handle to the hom space; lets replay caches compare spaces by
    /// pointer identity before falling back to structural equality.
    #[inline]
    pub fn homspace_arc(&self) -> &Arc<FusionTreeHomSpace> {
        &self.homspace
    }

    #[inline]
    pub fn subblock_structure(&self) -> &Arc<BlockStructure> {
        &self.subblock_structure
    }

    #[inline]
    pub fn rule_identity(&self) -> Option<RuleIdentity> {
        self.admission.rule_identity().cloned()
    }

    #[doc(hidden)]
    #[inline]
    pub fn admission(&self) -> &FusionSpaceAdmission {
        &self.admission
    }

    pub fn validate_rule<R: FusionRule>(&self, rule: &R) -> Result<(), CoreError> {
        match self.admission.rule_identity() {
            Some(expected) if expected != &rule.rule_identity() => Err(CoreError::FusionRuleMismatch {
                expected: expected.clone(),
                actual: rule.rule_identity(),
            }),
            Some(_) => Ok(()),
            None => Err(CoreError::MissingFusionRuleIdentity),
        }
    }

    /// Expert admission boundary for an imported or custom block layout.
    ///
    /// This certifies every present block against the rule and hom-space split,
    /// sector membership, duality, and logical degeneracy shape. Missing valid
    /// pairs remain structural zeros. It does not rebuild, repack, or otherwise
    /// select the caller's storage order, strides, or offsets; ordinary
    /// operation outputs derive those from their final hom space before
    /// admission.
    pub fn try_bind_rule<R: FusionRule>(mut self, rule: &R) -> Result<Self, CoreError> {
        let actual = rule.rule_identity();
        if let Some(expected) = self.admission.rule_identity() {
            if expected != &actual {
                return Err(CoreError::FusionRuleMismatch {
                    expected: expected.clone(),
                    actual,
                });
            }
            return Ok(self);
        }
        self.homspace
            .validate_subblock_structure_subset(rule, self.subblock_structure())?;
        self.admission = FusionSpaceAdmission::Subset(actual);
        Ok(self)
    }

    /// Checked finite-algebra admission for a caller-supplied block layout.
    ///
    /// An existing matching admission is deliberately revalidated because a
    /// legacy stamp does not prove checked finite-algebra closure.
    pub fn try_bind_rule_checked<R>(
        mut self,
        rule: &R,
    ) -> Result<Self, CheckedFusionSpaceError>
    where
        R: CheckedFusionAlgebra,
    {
        let actual = rule.rule_identity();
        let was_complete = matches!(self.admission, FusionSpaceAdmission::Complete(_));
        if let Some(expected) = self.admission.rule_identity() {
            if expected != &actual {
                return Err(CoreError::FusionRuleMismatch {
                    expected: expected.clone(),
                    actual,
                }
                .into());
            }
        }
        self.homspace
            .validate_subblock_structure_subset_checked(rule, self.subblock_structure())?;
        self.admission = if was_complete {
            FusionSpaceAdmission::Complete(actual)
        } else {
            FusionSpaceAdmission::Subset(actual)
        };
        Ok(self)
    }

    // Why not expose a general stamp setter: this path is reserved for structures
    // enumerated directly from the same HomSpace and rule above.
    fn with_complete_rule(mut self, identity: RuleIdentity) -> Self {
        self.admission = FusionSpaceAdmission::Complete(identity);
        self
    }

    pub fn find_subblock_index(&self, key: &FusionTreePairKey) -> Option<usize> {
        self.subblock_structure
            .find_block_index_by_fusion_tree_pair(key)
    }

    pub fn required_len(&self) -> Result<usize, CoreError> {
        self.subblock_structure.required_len()
    }
}
