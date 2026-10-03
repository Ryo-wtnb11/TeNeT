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
        let SectorExtent {
            index: sector,
            rows,
            cols,
        } = extent;
        let sector = *sector;
        let row = &mut self
            .trees
            .entry((sector, key.codomain_tree()))
            .or_default()
            .row;
        let (row_offset, new_row) = first_seen_offset(row, rows, row_dim)?;
        let col = &mut self
            .trees
            .entry((sector, key.domain_tree()))
            .or_default()
            .col;
        let (col_offset, new_col) = first_seen_offset(col, cols, col_dim)?;
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

fn first_seen_offset(
    slot: &mut Option<usize>,
    extent: &mut usize,
    dim: usize,
) -> Result<(usize, bool), CoreError> {
    if let Some(offset) = *slot {
        return Ok((offset, false));
    }
    let offset = *extent;
    *extent = offset
        .checked_add(dim)
        .ok_or(CoreError::ElementCountOverflow)?;
    *slot = Some(offset);
    Ok((offset, true))
}
