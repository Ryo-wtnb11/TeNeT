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
}

#[derive(Clone, Copy, Debug, Default)]
struct TreeOffsets {
    row: Option<usize>,
    col: Option<usize>,
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
                }
            });
        let sector = extent.index;
        let row = &mut self
            .trees
            .entry((sector, key.codomain_tree()))
            .or_default()
            .row;
        let (row_offset, new_row) = place_offset(row, &mut extent.rows, row_dim)?;
        let col = &mut self
            .trees
            .entry((sector, key.domain_tree()))
            .or_default()
            .col;
        let (col_offset, new_col) = place_offset(col, &mut extent.cols, col_dim)?;
        Ok(CoupledMatrixPlacement {
            sector,
            row_offset,
            col_offset,
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

/// The builder keeps one map for every sector and both sides, so it applies
/// the rule to its own slot and per-sector extent rather than through a
/// [`CoupledMatrixSide`] (which owns a single extent).
fn place_offset(
    slot: &mut Option<usize>,
    extent: &mut usize,
    dim: usize,
) -> Result<(usize, bool), CoreError> {
    if let Some(offset) = *slot {
        return Ok((offset, false));
    }
    let placed = first_seen_tree(extent, 0, || Ok(dim), || CoreError::ElementCountOverflow)?;
    *slot = Some(placed.offset);
    Ok((placed.offset, true))
}

/// Where the placement rule put one fusion tree on one side of a
/// coupled-sector matrix.
#[doc(hidden)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CoupledTreePlacement {
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
) -> Result<CoupledTreePlacement, E> {
    let dim = dim()?;
    let offset = *extent;
    *extent = offset.checked_add(dim).ok_or_else(overflow)?;
    Ok(CoupledTreePlacement { index, offset, dim })
}

/// The rows or the columns of one coupled-sector matrix under the
/// first-appearance rule: its running extent and, per tree, the projection
/// `V` of the [`CoupledTreePlacement`] its owner needs (an index, an offset
/// and dimension, ...), so the map stores no more than that.
#[doc(hidden)]
#[derive(Clone, Debug)]
pub struct CoupledMatrixSide<K, V> {
    trees: FxHashMap<K, V>,
    extent: usize,
}

impl<K, V> Default for CoupledMatrixSide<K, V> {
    fn default() -> Self {
        Self {
            trees: FxHashMap::default(),
            extent: 0,
        }
    }
}

impl<K: core::hash::Hash + Eq, V: Copy> CoupledMatrixSide<K, V> {
    /// Places `tree`: a known tree returns its stored value with `false`; a
    /// new tree is placed at the side's extent, which advances by `dim()`,
    /// and stores `project(placement)` with `true`. One hash per call.
    pub fn place<E>(
        &mut self,
        tree: K,
        dim: impl FnOnce() -> Result<usize, E>,
        overflow: impl FnOnce() -> E,
        project: impl FnOnce(CoupledTreePlacement) -> V,
    ) -> Result<(V, bool), E> {
        let index = self.trees.len();
        match self.trees.entry(tree) {
            std::collections::hash_map::Entry::Occupied(entry) => Ok((*entry.get(), false)),
            std::collections::hash_map::Entry::Vacant(entry) => {
                let value = project(first_seen_tree(&mut self.extent, index, dim, overflow)?);
                entry.insert(value);
                Ok((value, true))
            }
        }
    }

    /// Rows (columns) placed so far.
    pub fn extent(&self) -> usize {
        self.extent
    }

    pub fn get<Q>(&self, tree: &Q) -> Option<&V>
    where
        K: core::borrow::Borrow<Q>,
        Q: core::hash::Hash + Eq + ?Sized,
    {
        self.trees.get(tree)
    }

    pub fn iter(&self) -> impl Iterator<Item = (&K, &V)> {
        self.trees.iter()
    }

    /// Mutable stored values, for an owner that re-bases its trees (a
    /// contraction aligning one operand's tree order to another); the
    /// extent is unchanged.
    pub fn iter_mut(&mut self) -> impl Iterator<Item = (&K, &mut V)> {
        self.trees.iter_mut()
    }

    pub fn len(&self) -> usize {
        self.trees.len()
    }

    pub fn is_empty(&self) -> bool {
        self.trees.is_empty()
    }

    /// Starts the next sector: forgets every tree and the extent, keeping
    /// the allocation.
    pub fn clear(&mut self) {
        self.trees.clear();
        self.extent = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn side_places_first_sight_once_and_projects_what_it_stores() {
        let mut side = CoupledMatrixSide::<&str, (usize, usize)>::default();
        let overflow = || CoreError::ElementCountOverflow;
        let project = |placed: CoupledTreePlacement| (placed.offset, placed.dim);
        let place = |side: &mut CoupledMatrixSide<&'static str, (usize, usize)>, tree, dim| {
            side.place(tree, || Ok::<_, CoreError>(dim), overflow, project)
        };
        assert_eq!(place(&mut side, "a", 2).unwrap(), ((0, 2), true));
        assert_eq!(place(&mut side, "b", 3).unwrap(), ((2, 3), true));
        // A known tree keeps its first placement and never evaluates `dim`.
        let known = side.place(
            "a",
            || -> Result<usize, CoreError> { unreachable!("dim is evaluated only for a new tree") },
            overflow,
            project,
        );
        assert_eq!(known.unwrap(), ((0, 2), false));
        assert_eq!(side.extent(), 5);
        assert_eq!(
            place(&mut side, "c", usize::MAX),
            Err(CoreError::ElementCountOverflow)
        );
        assert!(side.get("c").is_none());
        assert_eq!(side.extent(), 5);

        let mut indexed = CoupledMatrixSide::<&str, usize>::default();
        indexed
            .place("x", || Ok::<_, CoreError>(4), overflow, |p| p.index)
            .unwrap();
        assert_eq!(
            indexed
                .place("y", || Ok::<_, CoreError>(1), overflow, |p| p.index)
                .unwrap(),
            (1, true)
        );
        indexed.clear();
        assert_eq!((indexed.len(), indexed.extent()), (0, 0));
    }
}
