use super::*;

/// Where one block lands in its coupled-sector matrix, and which of its
/// sector, row tree and column tree the builder saw for the first time.
#[doc(hidden)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CoupledMatrixPlacement {
    pub sector: usize,
    pub row_offset: usize,
    pub col_offset: usize,
    pub new_sector: bool,
    pub new_row: bool,
    pub new_col: bool,
}

/// First-appearance coupled-sector matricization of a block sequence in any
/// storage layout (TensorKit `block(t, c)` stacking).
///
/// Sectors are numbered in first-appearance order. Within a sector, a
/// codomain (domain) tree takes the next row (column) offset the first time
/// it appears and keeps it, so a tree's offset is the summed extent of the
/// trees seen before it. A repeated tree keeps its first extent; callers that
/// require consistent extents compare against the shapes they recorded.
#[doc(hidden)]
#[derive(Debug, Default)]
pub struct CoupledMatricizationBuilder<'a> {
    /// First-appearance index and `(rows, cols)` extent of each sector.
    sectors: FxHashMap<SectorId, SectorExtent>,
    /// Row and column offset of each tree within each sector. Offsets are
    /// sector-local: a raw block layout may pair a domain tree with codomain
    /// trees of different coupled sectors.
    trees: FxHashMap<(usize, &'a FusionTreeKey), TreeOffsets>,
}

#[derive(Clone, Copy, Debug)]
struct SectorExtent {
    index: usize,
    rows: usize,
    cols: usize,
    row_trees: usize,
    col_trees: usize,
}

#[derive(Clone, Copy, Debug, Default)]
struct TreeOffsets {
    row: Option<TreeExtent>,
    col: Option<TreeExtent>,
}

impl<'a> CoupledMatricizationBuilder<'a> {
    pub fn new() -> Self {
        Self::default()
    }

    /// Places the block `key` whose codomain (domain) degeneracy has
    /// `row_dim` (`col_dim`) elements. The coupled sector is the codomain
    /// tree's.
    pub fn place(
        &mut self,
        key: &'a FusionTreePairKey,
        row_dim: usize,
        col_dim: usize,
    ) -> Result<CoupledMatrixPlacement, CoreError> {
        let next_index = self.sectors.len();
        let mut new_sector = false;
        let extent = self
            .sectors
            .entry(key.codomain_tree().coupled())
            .or_insert_with(|| {
                new_sector = true;
                SectorExtent {
                    index: next_index,
                    rows: 0,
                    cols: 0,
                    row_trees: 0,
                    col_trees: 0,
                }
            });
        let sector = extent.index;
        let row = &mut self
            .trees
            .entry((sector, key.codomain_tree()))
            .or_default()
            .row;
        let (row, new_row) = place_in_slot(row, &mut extent.rows, &mut extent.row_trees, row_dim)?;
        let col = &mut self
            .trees
            .entry((sector, key.domain_tree()))
            .or_default()
            .col;
        let (col, new_col) = place_in_slot(col, &mut extent.cols, &mut extent.col_trees, col_dim)?;
        Ok(CoupledMatrixPlacement {
            sector,
            row_offset: row.offset,
            col_offset: col.offset,
            new_sector,
            new_row,
            new_col,
        })
    }

    /// Row and column extents of the coupled sector `coupled`; zero when no
    /// block of it was placed.
    pub fn extents(&self, coupled: SectorId) -> (usize, usize) {
        self.sectors
            .get(&coupled)
            .map_or((0, 0), |extent| (extent.rows, extent.cols))
    }
}

fn place_in_slot(
    slot: &mut Option<TreeExtent>,
    extent: &mut usize,
    count: &mut usize,
    dim: usize,
) -> Result<(TreeExtent, bool), CoreError> {
    if let Some(placed) = *slot {
        return Ok((placed, false));
    }
    let placed = first_seen_tree(
        extent,
        *count,
        || Ok(dim),
        || CoreError::ElementCountOverflow,
    )?;
    *count += 1;
    *slot = Some(placed);
    Ok((placed, true))
}

/// One fusion tree's row or column block inside a coupled-sector matrix.
#[doc(hidden)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct TreeExtent {
    /// First-appearance position among this side's trees.
    pub index: usize,
    /// Row (column) offset from the start of the matrix.
    pub offset: usize,
    /// Number of rows (columns) the tree spans.
    pub dim: usize,
}

/// The coupled-sector placement rule, the one place it is written down: a
/// tree seen for the first time takes the side's running extent as its
/// offset and advances it by its own dimension. `dim` is evaluated only for
/// a new tree.
fn first_seen_tree<E>(
    extent: &mut usize,
    index: usize,
    dim: impl FnOnce() -> Result<usize, E>,
    overflow: impl FnOnce() -> E,
) -> Result<TreeExtent, E> {
    let dim = dim()?;
    let offset = *extent;
    *extent = offset.checked_add(dim).ok_or_else(overflow)?;
    Ok(TreeExtent { index, offset, dim })
}

/// The rows or the columns of one coupled-sector matrix: each tree's
/// [`TreeExtent`] under the first-appearance rule. The running extent stays
/// with the caller, so a side's map can be owned (moved into a contraction
/// plan and re-based there) or borrowed scratch reused across sectors.
#[doc(hidden)]
#[derive(Clone, Debug)]
pub struct CoupledMatrixSide<K> {
    trees: FxHashMap<K, TreeExtent>,
}

impl<K> Default for CoupledMatrixSide<K> {
    fn default() -> Self {
        Self {
            trees: FxHashMap::default(),
        }
    }
}

impl<K: core::hash::Hash + Eq> CoupledMatrixSide<K> {
    /// Places `tree`: a known tree returns its first extent with `false`; a
    /// new tree is placed at `*extent`, which advances by `dim()`, and
    /// returns `true`. One hash per call.
    pub fn place<E>(
        &mut self,
        tree: K,
        extent: &mut usize,
        dim: impl FnOnce() -> Result<usize, E>,
        overflow: impl FnOnce() -> E,
    ) -> Result<(TreeExtent, bool), E> {
        let index = self.trees.len();
        match self.trees.entry(tree) {
            std::collections::hash_map::Entry::Occupied(entry) => Ok((*entry.get(), false)),
            std::collections::hash_map::Entry::Vacant(entry) => {
                let placed = first_seen_tree(extent, index, dim, overflow)?;
                entry.insert(placed);
                Ok((placed, true))
            }
        }
    }

    pub fn get<Q>(&self, tree: &Q) -> Option<&TreeExtent>
    where
        K: core::borrow::Borrow<Q>,
        Q: core::hash::Hash + Eq + ?Sized,
    {
        self.trees.get(tree)
    }

    pub fn iter(&self) -> impl Iterator<Item = (&K, &TreeExtent)> {
        self.trees.iter()
    }

    pub fn len(&self) -> usize {
        self.trees.len()
    }

    pub fn is_empty(&self) -> bool {
        self.trees.is_empty()
    }

    /// Moves every tree to `offset_of(tree)`, keeping its index and
    /// dimension (a contraction aligning one operand's tree order to another).
    pub fn rebase(&mut self, mut offset_of: impl FnMut(&K) -> usize) {
        for (tree, placed) in &mut self.trees {
            placed.offset = offset_of(tree);
        }
    }

    /// Forgets every tree, keeping the allocation for the next sector.
    pub fn clear(&mut self) {
        self.trees.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn side_places_first_sight_once_and_rebases_offsets_only() {
        let mut side = CoupledMatrixSide::<&str>::default();
        let mut extent = 0usize;
        let overflow = || CoreError::ElementCountOverflow;
        let place = |side: &mut CoupledMatrixSide<&'static str>, extent: &mut usize, tree, dim| {
            side.place(tree, extent, || Ok::<_, CoreError>(dim), overflow)
        };
        assert_eq!(
            place(&mut side, &mut extent, "a", 2).unwrap(),
            (
                TreeExtent {
                    index: 0,
                    offset: 0,
                    dim: 2
                },
                true
            )
        );
        assert_eq!(
            place(&mut side, &mut extent, "b", 3).unwrap(),
            (
                TreeExtent {
                    index: 1,
                    offset: 2,
                    dim: 3
                },
                true
            )
        );
        // A known tree keeps its first extent and never evaluates `dim`.
        let known = side.place(
            "a",
            &mut extent,
            || -> Result<usize, CoreError> { unreachable!("dim is evaluated only for a new tree") },
            overflow,
        );
        assert_eq!(
            known.unwrap(),
            (
                TreeExtent {
                    index: 0,
                    offset: 0,
                    dim: 2
                },
                false
            )
        );
        assert_eq!(extent, 5);
        assert_eq!(
            place(&mut side, &mut extent, "c", usize::MAX),
            Err(CoreError::ElementCountOverflow)
        );
        assert!(side.get("c").is_none());

        side.rebase(|tree| if *tree == "a" { 3 } else { 0 });
        assert_eq!(
            side.get("a"),
            Some(&TreeExtent {
                index: 0,
                offset: 3,
                dim: 2
            })
        );
        assert_eq!(
            side.get("b"),
            Some(&TreeExtent {
                index: 1,
                offset: 0,
                dim: 3
            })
        );
    }
}
