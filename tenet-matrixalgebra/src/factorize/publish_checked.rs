use super::*;

pub(super) struct FactorTreeCursor<'a, M> {
    pub(super) matricizations: &'a [M],
    pub(super) ranks: &'a [SectorRank],
    pub(super) matrix: usize,
    pub(super) tree: usize,
    pub(super) valid: bool,
}

impl<'a, M: SectorGeometry> FactorTreeCursor<'a, M> {
    pub(super) fn new(matricizations: &'a [M], ranks: &'a [SectorRank]) -> Self {
        let sectors_are_canonical = matricizations
            .windows(2)
            .all(|pair| pair[0].sector() < pair[1].sector());
        Self {
            matricizations,
            ranks,
            matrix: 0,
            tree: 0,
            valid: matricizations.len() == ranks.len() && sectors_are_canonical,
        }
    }

    pub(super) fn next(&mut self, side: FactorSide) -> Option<(SectorId, &'a FusionTreeKey)> {
        while let (Some(matrix), Some(rank)) = (
            self.matricizations.get(self.matrix),
            self.ranks.get(self.matrix),
        ) {
            self.valid &= matrix.sector() == rank.sector;
            if rank.kept != 0 {
                if let Some(tree) = matrix.tree(side, self.tree) {
                    self.tree += 1;
                    return Some((matrix.sector(), tree.tree));
                }
            }
            self.matrix += 1;
            self.tree = 0;
        }
        None
    }

    pub(super) fn matches(&mut self, side: FactorSide, key: &FusionTreeKey) -> bool {
        record_generic_pair_ordered_key_validation();
        if !self.valid {
            return false;
        }
        self.next(side)
            .is_some_and(|(sector, tree)| coupled_of_generic(key) == sector && key == tree)
    }

    pub(super) fn is_exhausted(&mut self, side: FactorSide) -> bool {
        self.next(side).is_none() && self.valid
    }
}

/// Staged keys of a prepared checked layout in enumeration order. The
/// prepared structure lists its blocks exactly as the enumeration produced
/// its keys and `commit` interns them without reordering, so this sequence
/// is the key order of the committed space; the checked builders validate
/// and commit from one enumeration instead of enumerating keys separately.
pub(super) fn staged_fusion_tree_keys(
    sector: &SectorStructure,
) -> impl Iterator<Item = &FusionTreePairKey> {
    sector
        .blocks()
        .iter()
        .filter_map(|block| match block.key() {
            BlockKey::FusionTree(key) => Some(key),
            _ => None,
        })
}

pub(super) fn validate_generic_factor_keys<'a, 'k, M: SectorGeometry>(
    keys: impl IntoIterator<Item = &'k FusionTreePairKey>,
    side: FactorSide,
    mut cursor: Option<&mut FactorTreeCursor<'_, M>>,
    matricizations: &'a [M],
    matrix_by_sector: &mut Option<FxHashMap<SectorId, &'a M>>,
) -> Result<bool, OperationError> {
    let mut ordered = true;
    let mut index: Option<PlacementIndex<'a>> = None;
    for key in keys {
        // Why not build the final layout first: missing source placements must
        // fail before output construction. The cursor only skips the old
        // lookup for the unique, sector-ordered sequence produced by the
        // canonical matricization path.
        let tree = match side {
            FactorSide::Left => key.codomain_tree(),
            FactorSide::Right => key.domain_tree(),
        };
        let aligned = cursor
            .as_deref_mut()
            .map(|cursor| cursor.matches(side, tree));
        if aligned != Some(true) {
            record_generic_pair_fallback_lookup(side);
            let matrix_by_sector =
                matrix_by_sector.get_or_insert_with(|| matricization_map(matricizations));
            let sector = coupled_of_generic(tree);
            matricization_of(matrix_by_sector, sector)?;
            index
                .get_or_insert_with(|| PlacementIndex::new(matricizations, &[side]))
                .placement(sector, side, tree)?;
        }
        if let Some(aligned) = aligned {
            ordered &= aligned;
        }
    }
    if let Some(cursor) = cursor {
        ordered &= cursor.is_exhausted(side);
    }
    Ok(ordered)
}

pub(super) fn checked_extent(shape: &[usize]) -> Option<usize> {
    tenet_core::checked_product(shape).ok()
}

pub(super) fn factor_output_is_canonical<D, M: SectorGeometry>(
    structure: &BlockStructure,
    matricizations: &[M],
    pairs: &[FactorPair<D>],
    required_len: usize,
    side: FactorSide,
) -> bool {
    if matricizations.len() != pairs.len() {
        return false;
    }
    let mut block_index = 0usize;
    let mut output_offset = 0usize;
    for (matrix, pair) in matricizations.iter().zip(pairs) {
        if pair.sector != matrix.sector() {
            return false;
        }
        let (factor_len, factor_leading) = match side {
            FactorSide::Left => (pair.left.len(), pair.left_rows),
            FactorSide::Right => (pair.right.len(), pair.right_leading),
        };
        let Some(next_offset) = sector_factor_output_is_canonical(
            structure,
            &mut block_index,
            output_offset,
            matrix,
            side,
            factor_len,
            factor_leading,
            pair.kept,
            side,
        ) else {
            return false;
        };
        output_offset = next_offset;
    }
    block_index == structure.block_count() && output_offset == required_len
}

/// Proves that one sector's selected factor (`factor_len` elements, column
/// major with leading dimension `factor_leading`, bond extent `bond`) already
/// occupies the blocks of `structure` starting at `block_index`/`output_offset`
/// in the exact layout `from_bound` expects, and returns the next output
/// offset. Left output reads the source as `F[o + q + a*j]`, right output as
/// `F[j + b*(o + q)]`; `a` is the source-side extent on `source_trees`.
#[allow(clippy::too_many_arguments)]
pub(super) fn sector_factor_output_is_canonical<M: SectorGeometry>(
    structure: &BlockStructure,
    block_index: &mut usize,
    output_offset: usize,
    matrix: &M,
    source_trees: FactorSide,
    factor_len: usize,
    factor_leading: usize,
    bond: usize,
    side: FactorSide,
) -> Option<usize> {
    let source_extent = match source_trees {
        FactorSide::Left => matrix.rows(),
        FactorSide::Right => matrix.cols(),
    };
    let required_leading = match side {
        FactorSide::Left => source_extent,
        FactorSide::Right => bond,
    };
    if factor_leading != required_leading || factor_len != source_extent.checked_mul(bond)? {
        return None;
    }
    let mut tree_prefix = 0usize;
    for tree_index in 0..matrix.tree_count(source_trees) {
        let tree = matrix.tree(source_trees, tree_index)?;
        let extent = checked_extent(tree.shape)?;
        if tree.offset != tree_prefix {
            return None;
        }
        tree_prefix = tree_prefix.checked_add(extent)?;
        if bond == 0 {
            continue;
        }
        let block = structure.block(*block_index).ok()?;
        let BlockKey::FusionTree(actual_key) = block.key() else {
            return None;
        };
        let actual_tree = match side {
            FactorSide::Left => actual_key.codomain_tree(),
            FactorSide::Right => actual_key.domain_tree(),
        };
        if actual_tree != tree.tree || coupled_of(actual_tree) != matrix.sector() {
            return None;
        }
        factor_block_is_canonical(
            &block,
            tree.shape,
            tree.offset,
            output_offset,
            source_extent,
            bond,
            side,
        )?;
        *block_index += 1;
    }
    if tree_prefix != source_extent {
        return None;
    }
    output_offset.checked_add(factor_len)
}

/// Proves that `block` holds the side-tree span `[tree_offset, tree_offset +
/// |tree_shape|)` of a column-major sector factor starting at `output_offset`
/// (left `a x bond`, right `bond x a`, `a = source_extent`).
pub(super) fn factor_block_is_canonical(
    block: &BlockRef<'_>,
    tree_shape: &[usize],
    tree_offset: usize,
    output_offset: usize,
    source_extent: usize,
    bond: usize,
    side: FactorSide,
) -> Option<()> {
    let block_offset = match side {
        FactorSide::Left => output_offset.checked_add(tree_offset)?,
        FactorSide::Right => output_offset.checked_add(bond.checked_mul(tree_offset)?)?,
    };
    if block.offset() != block_offset
        || block.shape().len() != tree_shape.len() + 1
        || block.strides().len() != block.shape().len()
    {
        return None;
    }
    let shape_matches = match side {
        FactorSide::Left => {
            &block.shape()[..tree_shape.len()] == tree_shape
                && block.shape()[tree_shape.len()] == bond
        }
        FactorSide::Right => block.shape()[0] == bond && &block.shape()[1..] == tree_shape,
    };
    if !shape_matches {
        return None;
    }
    let mut stride = match side {
        FactorSide::Left => 1,
        FactorSide::Right => bond,
    };
    let stride_offset = usize::from(matches!(side, FactorSide::Right));
    if matches!(side, FactorSide::Right) && block.strides()[0] != 1 {
        return None;
    }
    for (axis, &dim) in tree_shape.iter().enumerate() {
        if block.strides()[axis + stride_offset] != stride {
            return None;
        }
        stride = stride.checked_mul(dim)?;
    }
    if matches!(side, FactorSide::Left) && block.strides()[tree_shape.len()] != source_extent {
        return None;
    }
    Some(())
}

/// Identity-completed sector (bond states but no source matricization): its
/// output blocks, in output order, must tile a column-major `bond x bond`
/// identity at `output_offset`, with the output tree order as the side basis
/// exactly as the scatter fallback numbers it. A sector with no output blocks
/// on this side (e.g. the right factor of a codomain-only full-QR sector)
/// publishes nothing. Returns the next offset and whether an identity is due.
pub(super) fn identity_sector_output_is_canonical(
    structure: &BlockStructure,
    block_index: &mut usize,
    output_offset: usize,
    sector: SectorId,
    bond: usize,
    side: FactorSide,
) -> Option<(usize, bool)> {
    let mut tree_prefix = 0usize;
    while let Ok(block) = structure.block(*block_index) {
        let BlockKey::FusionTree(key) = block.key() else {
            return None;
        };
        let tree = match side {
            FactorSide::Left => key.codomain_tree(),
            FactorSide::Right => key.domain_tree(),
        };
        if coupled_of(tree) != sector {
            break;
        }
        let tree_shape = match side {
            FactorSide::Left => block.shape().split_last()?.1,
            FactorSide::Right => block.shape().split_first()?.1,
        };
        factor_block_is_canonical(
            &block,
            tree_shape,
            tree_prefix,
            output_offset,
            bond,
            bond,
            side,
        )?;
        tree_prefix = tree_prefix.checked_add(checked_extent(tree_shape)?)?;
        *block_index += 1;
    }
    if tree_prefix == 0 {
        Some((output_offset, false))
    } else if tree_prefix == bond {
        Some((output_offset.checked_add(bond.checked_mul(bond)?)?, true))
    } else {
        None
    }
}

/// A column-major `bond x bond` identity for a sector without a source
/// matricization (MatrixAlgebraKit `one!` on a zero-extent input block),
/// published after the first `pairs_before` pairs.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct IdentitySegment {
    pub(super) pairs_before: usize,
    pub(super) bond: usize,
}

/// One-sided sibling of [`factor_output_is_canonical`]: walks the ascending
/// union of source sectors and admitted bond sectors. A populated sector must
/// carry its pair and publish it by [`sector_factor_output_is_canonical`]; a
/// populated sector without a pair must have no bond (a null sector omitted
/// as full rank); a bond-only sector must pass
/// [`identity_sector_output_is_canonical`]. The whole admitted output must be
/// exhausted and every pair consumed, so unordered, duplicate or extraneous
/// records return `None` and keep the scatter fallback. `source_trees`
/// selects the matricization side whose trees index the selected factor
/// (columns for adjoint placement).
///
/// Every pair is consumed in order, so the plan records only the identity
/// segments; the list stays unallocated when there are none, keeping the
/// all-populated case free of plan allocation.
pub(super) fn one_sided_factor_output_plan<D, M: SectorGeometry>(
    structure: &BlockStructure,
    matricizations: &[M],
    pairs: &[FactorPair<D>],
    dimensions: &BTreeMap<SectorId, usize>,
    required_len: usize,
    side: FactorSide,
    source_trees: FactorSide,
) -> Option<Vec<IdentitySegment>> {
    let mut identities = Vec::new();
    let mut matrices = matricizations.iter().peekable();
    let mut pair_records = pairs.iter().enumerate().peekable();
    let mut bonds = dimensions.iter().peekable();
    let mut previous_matrix = None;
    let mut block_index = 0usize;
    let mut output_offset = 0usize;
    loop {
        let matrix_sector = matrices.peek().map(|matrix| matrix.sector());
        let sector = match (matrix_sector, bonds.peek().map(|(&sector, _)| sector)) {
            (None, None) => break,
            (Some(matrix), Some(bond)) => matrix.min(bond),
            (Some(sector), None) | (None, Some(sector)) => sector,
        };
        let bond = bonds
            .next_if(|&(&candidate, _)| candidate == sector)
            .map(|(_, &bond)| bond);
        if matrix_sector != Some(sector) {
            let (next_offset, identity) = identity_sector_output_is_canonical(
                structure,
                &mut block_index,
                output_offset,
                sector,
                bond?,
                side,
            )?;
            if identity {
                identities.push(IdentitySegment {
                    pairs_before: pair_records.peek().map_or(pairs.len(), |&(index, _)| index),
                    bond: bond?,
                });
            }
            output_offset = next_offset;
            continue;
        }
        let matrix = matrices.next()?;
        if previous_matrix.is_some_and(|previous| previous >= sector) {
            return None;
        }
        previous_matrix = Some(sector);
        let Some((_, pair)) = pair_records.next_if(|(_, pair)| pair.sector == sector) else {
            if bond.unwrap_or(0) != 0 {
                return None;
            }
            continue;
        };
        let (factor_len, factor_leading) = match side {
            FactorSide::Left => (pair.left.len(), pair.left_rows),
            FactorSide::Right => (pair.right.len(), pair.right_leading),
        };
        output_offset = sector_factor_output_is_canonical(
            structure,
            &mut block_index,
            output_offset,
            matrix,
            source_trees,
            factor_len,
            factor_leading,
            bond?,
            side,
        )?;
    }
    if pair_records.next().is_some()
        || block_index != structure.block_count()
        || output_offset != required_len
    {
        return None;
    }
    #[cfg(test)]
    ONE_SIDED_PUBLICATION_PROBE.with(|probe| {
        let mut value = probe.get();
        value.plan_bytes += identities.capacity() * std::mem::size_of::<IdentitySegment>();
        probe.set(value);
    });
    Some(identities)
}

/// Publishes the plan proved by [`one_sided_factor_output_plan`]: the
/// selected side of each pair is moved or appended and each identity is
/// written once in place, so no output element is written twice. The
/// opposite side stays in place for its own publication.
pub(super) fn take_one_sided_factors<D: FactorScalar>(
    pairs: &mut [FactorPair<D>],
    identities: &[IdentitySegment],
    required_len: usize,
    side: FactorSide,
) -> Vec<D> {
    #[cfg(test)]
    let first = pairs
        .iter()
        .map(|pair| match side {
            FactorSide::Left => &pair.left,
            FactorSide::Right => &pair.right,
        })
        .find(|factor| !factor.is_empty())
        .map(Vec::as_ptr);
    let mut output: Option<Vec<D>> = None;
    let mut _written = 0usize;
    let mut identities = identities.iter().peekable();
    for index in 0..=pairs.len() {
        while let Some(identity) = identities.next_if(|identity| identity.pairs_before == index) {
            let bond = identity.bond;
            let data = output.get_or_insert_with(|| Vec::with_capacity(required_len));
            let start = data.len();
            data.resize(start + bond * bond, D::zero());
            for diagonal in 0..bond {
                data[start + diagonal * (bond + 1)] = D::one();
            }
            _written += bond * bond;
        }
        let Some(pair) = pairs.get_mut(index) else {
            break;
        };
        let factor = match side {
            FactorSide::Left => std::mem::take(&mut pair.left),
            FactorSide::Right => std::mem::take(&mut pair.right),
        };
        _written += append_owned_factor(&mut output, factor, required_len);
    }
    let data = output.unwrap_or_default();
    #[cfg(test)]
    ONE_SIDED_PUBLICATION_PROBE.with(|probe| {
        let mut value = probe.get();
        value.canonical_publications += 1;
        value.owner_reused +=
            usize::from(first.is_some_and(|pointer| std::ptr::eq(pointer, data.as_ptr())));
        value.appended_elements += _written;
        probe.set(value);
    });
    data
}

/// Appends `factor` to `output`, letting the first nonempty factor keep its
/// allocation (grown once to `required_len`). Returns the element count that
/// was appended to an existing owner, zero otherwise.
pub(super) fn append_owned_factor<D>(
    output: &mut Option<Vec<D>>,
    factor: Vec<D>,
    required_len: usize,
) -> usize {
    if factor.is_empty() {
        return 0;
    }
    if let Some(data) = output {
        let appended = factor.len();
        data.extend(factor);
        appended
    } else {
        let mut factor = factor;
        factor.reserve_exact(required_len - factor.len());
        *output = Some(factor);
        0
    }
}

pub(super) fn publish_generic_factor_pairs<D>(
    pairs: Vec<FactorPair<D>>,
    left_len: usize,
    right_len: usize,
) -> (Vec<D>, Vec<D>) {
    #[cfg(test)]
    let first_left = pairs
        .iter()
        .find(|pair| !pair.left.is_empty())
        .map(|pair| pair.left.as_ptr());
    #[cfg(test)]
    let first_right = pairs
        .iter()
        .find(|pair| !pair.right.is_empty())
        .map(|pair| pair.right.as_ptr());
    let mut left_data: Option<Vec<D>> = None;
    let mut right_data: Option<Vec<D>> = None;
    for pair in pairs {
        let left_appended = append_owned_factor(&mut left_data, pair.left, left_len);
        let right_appended = append_owned_factor(&mut right_data, pair.right, right_len);
        record_generic_pair_appended(left_appended, right_appended);
    }
    let left_data = left_data.unwrap_or_default();
    let right_data = right_data.unwrap_or_default();
    #[cfg(test)]
    GENERIC_PAIR_PUBLICATION_PROBE.with(|probe| {
        let mut value = probe.get();
        value.left_owner_reused += usize::from(
            first_left.is_some_and(|pointer| std::ptr::eq(pointer, left_data.as_ptr())),
        );
        value.right_owner_reused += usize::from(
            first_right.is_some_and(|pointer| std::ptr::eq(pointer, right_data.as_ptr())),
        );
        probe.set(value);
    });
    (left_data, right_data)
}

pub(super) fn build_left_right_bound_pair_generic_checked<R, D, M>(
    provider: &Arc<R>,
    homspace: &FusionTreeHomSpace,
    matricizations: &[M],
    pairs: Vec<FactorPair<D>>,
) -> Result<DynamicFactorPair<R, D>, CheckedGenericFactorPlanError<R::Error>>
where
    R: CheckedGenericFusion,
    D: FactorScalar,
    M: SectorGeometry,
{
    let ranks = pairs
        .iter()
        .map(|pair| SectorRank {
            sector: pair.sector,
            kept: pair.kept,
        })
        .collect::<Vec<_>>();
    let mut matrix_by_sector = None;
    let new_leg = SectorLeg::new(ranks.iter().map(|rank| (rank.sector, rank.kept)), false);
    // One provider-backed enumeration per side serves both the ordered key
    // validation and the committed space; the left side completes before the
    // right side starts so validation precedence is unchanged.
    let left_hom = FusionTreeHomSpace::new(
        homspace.codomain().clone(),
        FusionProductSpace::new([new_leg.clone()]),
    );
    let left_prepared = left_hom
        .prepare_coupled_subblock_structure_from_leg_degeneracies_generic_checked(provider.as_ref())
        .map_err(CheckedGenericFactorPlanError::from)?;
    let mut left_cursor = FactorTreeCursor::new(matricizations, &ranks);
    let left_ordered = validate_generic_factor_keys(
        staged_fusion_tree_keys(left_prepared.sector_structure()),
        FactorSide::Left,
        Some(&mut left_cursor),
        matricizations,
        &mut matrix_by_sector,
    )
    .map_err(CheckedGenericFactorPlanError::from)?;
    let left = BoundDynamicFusionMapSpace::from_prepared_final_homspace_generic_checked(
        Arc::clone(provider),
        left_hom,
        left_prepared,
    )
    .map_err(CheckedGenericFactorPlanError::from)?;
    let right_hom = FusionTreeHomSpace::new(
        FusionProductSpace::new([new_leg]),
        homspace.domain().clone(),
    );
    let right_prepared = right_hom
        .prepare_coupled_subblock_structure_from_leg_degeneracies_generic_checked(provider.as_ref())
        .map_err(CheckedGenericFactorPlanError::from)?;
    let mut right_cursor = FactorTreeCursor::new(matricizations, &ranks);
    let right_ordered = validate_generic_factor_keys(
        staged_fusion_tree_keys(right_prepared.sector_structure()),
        FactorSide::Right,
        Some(&mut right_cursor),
        matricizations,
        &mut matrix_by_sector,
    )
    .map_err(CheckedGenericFactorPlanError::from)?;
    let right = BoundDynamicFusionMapSpace::from_prepared_final_homspace_generic_checked(
        Arc::clone(provider),
        right_hom,
        right_prepared,
    )
    .map_err(CheckedGenericFactorPlanError::from)?;
    let left_len = left.space().required_len().map_err(|e| {
        CheckedGenericFactorPlanError::Operation(OperationError::from_core_preserving_context(e))
    })?;
    let right_len = right.space().required_len().map_err(|e| {
        CheckedGenericFactorPlanError::Operation(OperationError::from_core_preserving_context(e))
    })?;
    let canonical = left_ordered
        && right_ordered
        && factor_output_is_canonical(
            left.space().structure(),
            matricizations,
            &pairs,
            left_len,
            FactorSide::Left,
        )
        && factor_output_is_canonical(
            right.space().structure(),
            matricizations,
            &pairs,
            right_len,
            FactorSide::Right,
        );
    let (left_data, right_data) = if canonical {
        #[cfg(test)]
        GENERIC_PAIR_PUBLICATION_PROBE.with(|probe| {
            let mut value = probe.get();
            value.canonical_publications += 1;
            probe.set(value);
        });
        publish_generic_factor_pairs(pairs, left_len, right_len)
    } else {
        #[cfg(test)]
        GENERIC_PAIR_PUBLICATION_PROBE.with(|probe| {
            let mut value = probe.get();
            value.fallback_publications += 1;
            probe.set(value);
        });
        let mut left_data = vec![D::zero(); left_len];
        let mut right_data = vec![D::zero(); right_len];
        let index = PlacementIndex::new(matricizations, &[FactorSide::Left, FactorSide::Right]);
        let left_groups = SectorBlockGroups::new(left.space().structure(), FactorSide::Left)
            .map_err(CheckedGenericFactorPlanError::from)?;
        let right_groups = SectorBlockGroups::new(right.space().structure(), FactorSide::Right)
            .map_err(CheckedGenericFactorPlanError::from)?;
        for (matrix, pair) in matricizations.iter().zip(&pairs) {
            scatter_left_sector_blocks_generic(
                left.space(),
                &mut left_data,
                matrix,
                &index,
                &left_groups,
                &pair.left,
                pair.left_rows,
            )
            .map_err(CheckedGenericFactorPlanError::from)?;
            scatter_right_sector_blocks_generic(
                right.space(),
                &mut right_data,
                matrix,
                &index,
                &right_groups,
                &pair.right,
                pair.right_leading,
            )
            .map_err(CheckedGenericFactorPlanError::from)?;
        }
        (left_data, right_data)
    };
    let left_nout = left.space().nout();
    let right_nin = right.space().nin();
    Ok((
        BoundDynFactor::from_bound(left, left_data, left_nout, 1)
            .map_err(CheckedGenericFactorPlanError::from)?,
        BoundDynFactor::from_bound(right, right_data, 1, right_nin)
            .map_err(CheckedGenericFactorPlanError::from)?,
    ))
}

pub(super) fn build_checked_pair_from_input<R, D>(
    provider: &Arc<R>,
    homspace: &FusionTreeHomSpace,
    matricizations: &InputMatricizations<'_, D>,
    pairs: Vec<FactorPair<D>>,
) -> Result<DynamicFactorPair<R, D>, CheckedGenericFactorPlanError<R::Error>>
where
    R: CheckedGenericFusion,
    D: FactorScalar,
{
    match matricizations {
        InputMatricizations::Regions { regions, .. } => {
            build_left_right_bound_pair_generic_checked(provider, homspace, regions.as_ref(), pairs)
        }
        InputMatricizations::Packed(matrices) => {
            build_left_right_bound_pair_generic_checked(provider, homspace, matrices, pairs)
        }
    }
}

/// Checked generic factor materialization with identity completion for sectors
/// absent from the source matricization (the full-factor contract).
pub(super) fn build_bound_factor_generic_checked<R, D, M>(
    provider: &Arc<R>,
    homspace: &FusionTreeHomSpace,
    matricizations: &[M],
    pairs: &mut [FactorPair<D>],
    dimensions: &BTreeMap<SectorId, usize>,
    side: FactorSide,
) -> Result<BoundDynFactor<R, D>, CheckedGenericFactorPlanError<R::Error>>
where
    R: CheckedGenericFusion,
    D: FactorScalar,
    M: SectorGeometry,
{
    let bond = SectorLeg::new(
        dimensions.iter().map(|(&sector, &dim)| (sector, dim)),
        false,
    );
    let output_hom = match side {
        FactorSide::Left => {
            FusionTreeHomSpace::new(homspace.codomain().clone(), FusionProductSpace::new([bond]))
        }
        FactorSide::Right => {
            FusionTreeHomSpace::new(FusionProductSpace::new([bond]), homspace.domain().clone())
        }
    };
    let matrices = matricizations
        .iter()
        .map(|matrix| (matrix.sector(), matrix))
        .collect::<FxHashMap<_, _>>();
    // One provider-backed enumeration serves both the populated-key
    // prevalidation and the committed space; the prepared structure lists its
    // blocks in the enumeration's key order, so the first-match placement
    // semantics are those of the key sequence.
    let prepared = output_hom
        .prepare_coupled_subblock_structure_from_leg_degeneracies_generic_checked(provider.as_ref())
        .map_err(CheckedGenericFactorPlanError::from)?;
    let placements = PlacementIndex::new(matricizations, &[side]);
    for block in prepared.sector_structure().blocks() {
        let BlockKey::FusionTree(key) = block.key() else {
            continue;
        };
        let sector = match side {
            FactorSide::Left => coupled_of_generic(key.codomain_tree()),
            FactorSide::Right => coupled_of_generic(key.domain_tree()),
        };
        if matrices.contains_key(&sector) {
            let tree = match side {
                FactorSide::Left => key.codomain_tree(),
                FactorSide::Right => key.domain_tree(),
            };
            placements.placement(sector, side, tree)?;
        }
    }
    let space = BoundDynamicFusionMapSpace::from_prepared_final_homspace_generic_checked(
        Arc::clone(provider),
        output_hom,
        prepared,
    )
    .map_err(CheckedGenericFactorPlanError::from)?;
    let len = space.space().required_len().map_err(|e| {
        CheckedGenericFactorPlanError::Operation(OperationError::from_core_preserving_context(e))
    })?;
    let (nout, nin) = match side {
        FactorSide::Left => (space.space().nout(), 1),
        FactorSide::Right => (1, space.space().nin()),
    };
    if let Some(identities) = one_sided_factor_output_plan(
        space.space().structure(),
        matricizations,
        pairs,
        dimensions,
        len,
        side,
        side,
    ) {
        let data = take_one_sided_factors(pairs, &identities, len, side);
        return BoundDynFactor::from_bound(space, data, nout, nin)
            .map_err(CheckedGenericFactorPlanError::from);
    }
    record_one_sided_fallback_publication();
    let mut data = vec![D::zero(); len];
    let pairs = pairs
        .iter()
        .map(|pair| (pair.sector, pair))
        .collect::<FxHashMap<_, _>>();
    let mut missing_offsets = FxHashMap::<SectorId, usize>::default();
    let structure = Arc::clone(space.space().structure());
    for index in 0..structure.block_count() {
        let block = structure.block(index).map_err(|e| {
            CheckedGenericFactorPlanError::Operation(OperationError::from_core_preserving_context(
                e,
            ))
        })?;
        let BlockKey::FusionTree(key) = block.key() else {
            continue;
        };
        let (sector, axis) = match side {
            FactorSide::Left => (
                coupled_of_generic(key.codomain_tree()),
                block.shape().len() - 1,
            ),
            FactorSide::Right => (coupled_of_generic(key.domain_tree()), 0),
        };
        if matrices.contains_key(&sector) {
            let pair = pairs
                .get(&sector)
                .ok_or(CheckedGenericFactorPlanError::Operation(
                    OperationError::UnsupportedTensorContractScope {
                        message: "factor rank absent for a populated source sector",
                    },
                ))?;
            let tree = match side {
                FactorSide::Left => key.codomain_tree(),
                FactorSide::Right => key.domain_tree(),
            };
            let offset = placements.placement(sector, side, tree)?.0;
            let (factor, factor_rows) = match side {
                FactorSide::Left => (&pair.left, pair.left_rows),
                FactorSide::Right => (&pair.right, pair.right_leading),
            };
            scatter_matrix_block(
                &mut data,
                block.shape(),
                block.strides(),
                block.offset(),
                axis,
                side,
                factor,
                factor_rows,
                offset,
            );
            continue;
        }
        let dimension =
            *dimensions
                .get(&sector)
                .ok_or(CheckedGenericFactorPlanError::Operation(
                    OperationError::UnsupportedTensorContractScope {
                        message: "factor sector absent from the source tensor",
                    },
                ))?;
        let side_offset = missing_offsets.entry(sector).or_default();
        let extent = block
            .shape()
            .iter()
            .enumerate()
            .filter(|&(i, _)| i != axis)
            .try_fold(1usize, |acc, (_, &value)| acc.checked_mul(value))
            .ok_or(CheckedGenericFactorPlanError::Operation(
                OperationError::ElementCountOverflow,
            ))?;
        let end =
            side_offset
                .checked_add(extent)
                .ok_or(CheckedGenericFactorPlanError::Operation(
                    OperationError::ElementCountOverflow,
                ))?;
        if end > dimension {
            return Err(CheckedGenericFactorPlanError::Operation(
                OperationError::ElementCountMismatch {
                    expected: dimension,
                    actual: end,
                },
            ));
        }
        scatter_identity_matrix_block(
            &mut data,
            block.shape(),
            block.strides(),
            block.offset(),
            axis,
            dimension,
            *side_offset,
            extent,
        )?;
        *side_offset = end;
    }
    BoundDynFactor::from_bound(space, data, nout, nin).map_err(CheckedGenericFactorPlanError::from)
}

/// Generic sibling of [`scatter_left_sector_blocks`].
pub(super) fn scatter_left_sector_blocks_generic<D, M>(
    left_space: &DynamicFusionMapSpace,
    left_data: &mut [D],
    matrix: &M,
    index: &PlacementIndex<'_>,
    groups: &SectorBlockGroups,
    factor: &[D],
    factor_rows: usize,
) -> Result<(), OperationError>
where
    D: FactorScalar,
    M: SectorGeometry,
{
    let left_structure = Arc::clone(left_space.structure());
    for block_index in groups.blocks(matrix.sector()) {
        #[cfg(test)]
        record_scatter_visit(FactorSide::Left);
        let block = left_structure
            .block(block_index)
            .map_err(OperationError::from_core_preserving_context)?;
        let BlockKey::FusionTree(key) = block.key() else {
            continue;
        };
        debug_assert_eq!(coupled_of_generic(key.codomain_tree()), matrix.sector());
        let (row_offset, _) =
            index.placement(matrix.sector(), FactorSide::Left, key.codomain_tree())?;
        #[cfg(test)]
        GENERIC_PAIR_PUBLICATION_PROBE.with(|probe| {
            let mut value = probe.get();
            value.left_scatter_calls += 1;
            value.left_scattered_elements +=
                checked_extent(block.shape()).expect("admitted block extent is finite");
            probe.set(value);
        });
        scatter_matrix_block(
            left_data,
            block.shape(),
            block.strides(),
            block.offset(),
            block.shape().len() - 1,
            FactorSide::Left,
            factor,
            factor_rows,
            row_offset,
        );
    }
    Ok(())
}

/// Generic sibling of [`scatter_right_sector_blocks`].
pub(super) fn scatter_right_sector_blocks_generic<D, M>(
    right_space: &DynamicFusionMapSpace,
    right_data: &mut [D],
    matrix: &M,
    index: &PlacementIndex<'_>,
    groups: &SectorBlockGroups,
    factor: &[D],
    factor_rows: usize,
) -> Result<(), OperationError>
where
    D: FactorScalar,
    M: SectorGeometry,
{
    let right_structure = Arc::clone(right_space.structure());
    for block_index in groups.blocks(matrix.sector()) {
        #[cfg(test)]
        record_scatter_visit(FactorSide::Right);
        let block = right_structure
            .block(block_index)
            .map_err(OperationError::from_core_preserving_context)?;
        let BlockKey::FusionTree(key) = block.key() else {
            continue;
        };
        debug_assert_eq!(coupled_of_generic(key.domain_tree()), matrix.sector());
        let (col_offset, _) =
            index.placement(matrix.sector(), FactorSide::Right, key.domain_tree())?;
        #[cfg(test)]
        GENERIC_PAIR_PUBLICATION_PROBE.with(|probe| {
            let mut value = probe.get();
            value.right_scatter_calls += 1;
            value.right_scattered_elements +=
                checked_extent(block.shape()).expect("admitted block extent is finite");
            probe.set(value);
        });
        scatter_matrix_block(
            right_data,
            block.shape(),
            block.strides(),
            block.offset(),
            0,
            FactorSide::Right,
            factor,
            factor_rows,
            col_offset,
        );
    }
    Ok(())
}

/// Why not the paired builder: its bond holds only sectors that carry a
/// factor pair, so a side-only sector would be dropped from the full
/// factor's `fuse(codomain)`/`fuse(domain)` bond instead of publishing its
/// identity block, as TensorKit's `initialize_output(qr_full!/lq_full!)` does.
pub(super) fn checked_full_factor_pair<R, D, M>(
    provider: &Arc<R>,
    homspace: &FusionTreeHomSpace,
    matrices: &[M],
    mut pairs: Vec<FactorPair<D>>,
    dimensions: &BTreeMap<SectorId, usize>,
) -> Result<DynamicFactorPair<R, D>, CheckedGenericFactorPlanError<R::Error>>
where
    R: CheckedGenericFusion,
    D: FactorScalar,
    M: SectorGeometry,
{
    let left = build_bound_factor_generic_checked(
        provider,
        homspace,
        matrices,
        &mut pairs,
        dimensions,
        FactorSide::Left,
    )?;
    let right = build_bound_factor_generic_checked(
        provider,
        homspace,
        matrices,
        &mut pairs,
        dimensions,
        FactorSide::Right,
    )?;
    Ok((left, right))
}
