//! Owned block outputs that every destination block overwrites exactly once
//! (#1290).
//!
//! The destination buffer is left uninitialized only when
//! [`blocks_tile_storage`] proves that the blocks' reachable offsets
//! partition `0..required_len`. Each block is then written through a
//! [`BlockOverwrite`] that derives the destination strides and offset from
//! the block itself, so a caller supplies only source geometry and cannot
//! aim a write elsewhere; [`initialize_owned`] publishes the length only after
//! every block reported a complete write. A structure without that proof is
//! written into a zeroed buffer through the initialized kernels, which is the
//! fill-then-overwrite sequence these outputs used before.

use core::mem::MaybeUninit;
use core::ops::{Add, Mul};

use num_traits::{One, Zero};
use tenet_core::{BlockRef, BlockStructure};

use crate::owned_overwrite_buffer::initialize_owned;
use crate::{CheckedBlockLayout, ConjugateValue, OperationError};

#[cfg(debug_assertions)]
thread_local! {
    static PREFILLED_ELEMENTS: core::cell::Cell<usize> = const { core::cell::Cell::new(0) };
}

/// Returns and resets the number of output elements this thread zero-filled
/// before an owned block overwrite because the layout was not proved to tile
/// its storage.
///
/// Why debug builds only: as `take_checked_block_passes`, a probe for tests.
#[cfg(debug_assertions)]
#[doc(hidden)]
pub fn take_owned_block_prefills() -> usize {
    PREFILLED_ELEMENTS.with(core::cell::Cell::take)
}

enum Storage<'a, D> {
    Uninit(&'a mut [MaybeUninit<D>]),
    Zeroed(&'a mut [D]),
}

/// The single write of one destination block of an owned output.
#[doc(hidden)]
pub struct BlockOverwrite<'a, D> {
    storage: Storage<'a, D>,
    layout: &'a mut CheckedBlockLayout,
    block: BlockRef<'a>,
    written: bool,
}

impl<D> BlockOverwrite<'_, D>
where
    D: Copy + Add<D, Output = D> + Mul<D, Output = D> + PartialEq + Zero + One + ConjugateValue,
{
    fn claim(&self) -> Result<(), OperationError> {
        if self.written {
            return Err(OperationError::InvalidArgument {
                message: "owned output block written twice",
            });
        }
        Ok(())
    }

    /// `block = alpha * op(source)`, where `source_stride(axis)` is the
    /// source stride of the block's logical `axis`, and `op` conjugates when
    /// `conjugate` is set. `alpha = 1` is a bit-exact copy.
    pub fn copy(
        &mut self,
        mut source_stride: impl FnMut(usize) -> Result<usize, OperationError>,
        source: &[D],
        source_offset: usize,
        conjugate: bool,
        alpha: D,
    ) -> Result<(), OperationError> {
        self.claim()?;
        let strides = self.block.strides();
        self.layout.fill_one(self.block.shape(), |axis| {
            Ok((strides[axis], source_stride(axis)?))
        })?;
        let dst_offset = checked_offset(self.block.offset())?;
        let src_offset = checked_offset(source_offset)?;
        match &mut self.storage {
            Storage::Uninit(dst) => self
                .layout
                .copy_uninit(dst, source, dst_offset, src_offset, conjugate, alpha)?,
            Storage::Zeroed(dst) => self.layout.tensoradd(
                dst,
                source,
                dst_offset,
                src_offset,
                conjugate,
                alpha,
                D::zero(),
            )?,
        }
        self.written = true;
        Ok(())
    }

    /// `block = alpha * op_l(lhs) + beta * op_r(rhs)` with the arithmetic of
    /// [`CheckedBlockLayout::add_two_source`]; a zero coefficient drops its
    /// source, and both zero leave the block zero. `source_strides(axis)`
    /// returns the `(lhs, rhs)` strides of logical `axis`, and both are
    /// converted before any write whichever coefficients are zero. Each
    /// source is `(data, offset, conjugate)`.
    pub fn add(
        &mut self,
        mut source_strides: impl FnMut(usize) -> Result<(usize, usize), OperationError>,
        (lhs, lhs_offset, lhs_conjugate): (&[D], usize, bool),
        (rhs, rhs_offset, rhs_conjugate): (&[D], usize, bool),
        alpha: D,
        beta: D,
    ) -> Result<(), OperationError> {
        self.claim()?;
        let strides = self.block.strides();
        // With `alpha = 0` the rhs is the layout's first source, which the
        // single-source walks read.
        let swap = alpha.is_zero();
        self.layout.fill_two(self.block.shape(), |axis| {
            let (l, r) = source_strides(axis)?;
            let (first, second) = if swap { (r, l) } else { (l, r) };
            Ok((strides[axis], first, second))
        })?;
        let dst_offset = checked_offset(self.block.offset())?;
        let lhs_offset = checked_offset(lhs_offset)?;
        let rhs_offset = checked_offset(rhs_offset)?;
        let single = match (alpha.is_zero(), beta.is_zero()) {
            (false, false) => None,
            (false, true) => Some((lhs, lhs_offset, lhs_conjugate, alpha)),
            (true, false) => Some((rhs, rhs_offset, rhs_conjugate, beta)),
            (true, true) => {
                self.write_zero()?;
                self.written = true;
                return Ok(());
            }
        };
        match (&mut self.storage, single) {
            (Storage::Uninit(dst), None) => self.layout.add_two_source_uninit(
                dst,
                lhs,
                rhs,
                dst_offset,
                lhs_offset,
                rhs_offset,
                lhs_conjugate,
                rhs_conjugate,
                alpha,
                beta,
            )?,
            (Storage::Zeroed(dst), None) => self.layout.add_two_source(
                dst,
                lhs,
                rhs,
                dst_offset,
                lhs_offset,
                rhs_offset,
                lhs_conjugate,
                rhs_conjugate,
                alpha,
                beta,
            )?,
            (Storage::Uninit(dst), Some((source, offset, conjugate, factor))) => self
                .layout
                .copy_uninit(dst, source, dst_offset, offset, conjugate, factor)?,
            (Storage::Zeroed(dst), Some((source, offset, conjugate, factor))) => {
                self.layout.tensoradd(
                    dst,
                    source,
                    dst_offset,
                    offset,
                    conjugate,
                    factor,
                    D::zero(),
                )?
            }
        }
        self.written = true;
        Ok(())
    }

    /// Leaves the block zero.
    pub fn zero(&mut self) -> Result<(), OperationError> {
        self.claim()?;
        self.write_zero()?;
        self.written = true;
        Ok(())
    }

    fn write_zero(&mut self) -> Result<(), OperationError> {
        match &mut self.storage {
            // Walks the block's own layout, as every other write does: a
            // tiled block need not be a contiguous range (coupled-sector
            // blocks are sub-matrices of their sector matrix). The source is
            // one zero read with stride zero on every axis.
            Storage::Uninit(dst) => {
                let strides = self.block.strides();
                self.layout
                    .fill_one(self.block.shape(), |axis| Ok((strides[axis], 0)))?;
                self.layout.copy_uninit(
                    dst,
                    &[D::zero()],
                    checked_offset(self.block.offset())?,
                    0,
                    false,
                    D::one(),
                )
            }
            // Why not write zeros: the buffer starts zeroed, and an unproved
            // layout may alias, where the fill-then-overwrite sequence this
            // path preserves left other blocks' writes in place.
            Storage::Zeroed(_) => Ok(()),
        }
    }
}

fn checked_offset(offset: usize) -> Result<isize, OperationError> {
    isize::try_from(offset).map_err(|_| OperationError::OffsetOverflow { value: offset })
}

/// Whether the blocks' reachable offsets partition `0..len`, so writing every
/// block once initializes every element exactly once. Sufficient, not
/// necessary; `false` only costs the zero fill. Two proofs are accepted, both
/// without allocation or hashing:
///
/// - the tiling its constructor recorded on the structure
///   ([`BlockStructure::storage_tiling_proven`]: the canonical coupled-sector
///   layout built from leg degeneracies, TensorKit's block layout);
/// - non-empty blocks that, in index order, are each compact (column-major
///   under some axis permutation) and contiguous from offset zero to `len`.
///
/// Why not `BlockStructure::coupled_sector_regions`: it compiles hashed region
/// metadata on the structure wrapper, which the interner rebuilds whenever the
/// last owner drops it, so a steady-state eager call would pay it every time.
fn blocks_tile_storage(structure: &BlockStructure, len: usize) -> Result<bool, OperationError> {
    Ok(structure.storage_tiling_proven() || compact_blocks_tile_storage(structure, len)?)
}

fn compact_blocks_tile_storage(
    structure: &BlockStructure,
    len: usize,
) -> Result<bool, OperationError> {
    let mut next = 0usize;
    for index in 0..structure.block_count() {
        let block = structure.block(index)?;
        let count = block.element_count()?;
        if count == 0 {
            continue;
        }
        if block.offset() != next || !is_compact(block.shape(), block.strides()) {
            return Ok(false);
        }
        next = next
            .checked_add(count)
            .ok_or(OperationError::ElementCountOverflow)?;
    }
    Ok(next == len)
}

/// For a block with no zero extent: whether its strides, ordered, are the
/// column-major strides of its extents, so it reaches `0..count` from its
/// offset, each once. Each step needs exactly one non-unit axis whose stride
/// is the running product; the product strictly grows, so no axis is used
/// twice. Why not sort the axes: that would spill a buffer past the inline
/// rank on the output path this module exists to keep allocation-free.
fn is_compact(shape: &[usize], strides: &[usize]) -> bool {
    let axes = || shape.iter().zip(strides).filter(|(&extent, _)| extent > 1);
    let mut expected = 1usize;
    for _ in axes() {
        let mut next = axes().filter(|(_, &stride)| stride == expected);
        let (Some((&extent, _)), None) = (next.next(), next.next()) else {
            return false;
        };
        // Cannot overflow: the block's element count was checked.
        expected *= extent;
    }
    true
}

/// Allocates `destination`'s payload and writes every block once through
/// `write_block`, in block order. Each call must complete exactly one
/// [`BlockOverwrite`] write or return an error; a block left unwritten is an
/// error, and on any error or panic no partially written buffer escapes.
#[doc(hidden)]
pub fn overwrite_owned_blocks<D>(
    destination: &BlockStructure,
    write_block: impl FnMut(BlockRef<'_>, &mut BlockOverwrite<'_, D>) -> Result<(), OperationError>,
) -> Result<Vec<D>, OperationError>
where
    D: Copy + Add<D, Output = D> + Mul<D, Output = D> + PartialEq + Zero + One + ConjugateValue,
{
    overwrite_owned_blocks_in(destination, true, write_block)
}

fn overwrite_owned_blocks_in<D>(
    destination: &BlockStructure,
    allow_uninit: bool,
    mut write_block: impl FnMut(BlockRef<'_>, &mut BlockOverwrite<'_, D>) -> Result<(), OperationError>,
) -> Result<Vec<D>, OperationError>
where
    D: Copy + Add<D, Output = D> + Mul<D, Output = D> + PartialEq + Zero + One + ConjugateValue,
{
    let len = destination.required_len()?;
    let mut layout = CheckedBlockLayout::default();
    let mut write_all = |mut storage: Storage<'_, D>| -> Result<(), OperationError> {
        for index in 0..destination.block_count() {
            let block = destination.block(index)?;
            let mut writer = BlockOverwrite {
                storage: match &mut storage {
                    Storage::Uninit(dst) => Storage::Uninit(dst),
                    Storage::Zeroed(dst) => Storage::Zeroed(dst),
                },
                layout: &mut layout,
                block,
                written: false,
            };
            write_block(block, &mut writer)?;
            if !writer.written {
                return Err(OperationError::InvalidArgument {
                    message: "owned output block left unwritten",
                });
            }
        }
        Ok(())
    };
    if allow_uninit && blocks_tile_storage(destination, len)? {
        return initialize_owned(len, |dst| write_all(Storage::Uninit(dst)));
    }
    #[cfg(debug_assertions)]
    PREFILLED_ELEMENTS.with(|count| count.set(count.get() + len));
    let mut data = vec![D::zero(); len];
    write_all(Storage::Zeroed(&mut data))?;
    Ok(data)
}

#[cfg(test)]
mod tests {
    use super::*;
    use num_complex::Complex64;
    use tenet_core::{BlockKey, BlockSpec};

    fn structure(blocks: &[(&[usize], &[usize], usize)]) -> BlockStructure {
        BlockStructure::from_blocks_with_rank(
            3,
            blocks
                .iter()
                .enumerate()
                .map(|(index, &(shape, strides, offset))| {
                    BlockSpec::with_key(
                        BlockKey::opaque([index as u64]),
                        shape.to_vec(),
                        strides.to_vec(),
                        offset,
                    )
                    .unwrap()
                })
                .collect(),
        )
        .unwrap()
    }

    /// Column-major, an empty block, an axis-permuted compact block, and
    /// extent-one axes with arbitrary strides: 23 elements, tiled in order.
    fn tiled() -> BlockStructure {
        structure(&[
            (&[2, 3, 1], &[1, 2, 6], 0),
            (&[0, 4, 2], &[1, 1, 4], 6),
            (&[3, 2, 2], &[4, 1, 2], 6),
            (&[1, 5, 1], &[7, 1, 9], 18),
        ])
    }

    fn values() -> Vec<Complex64> {
        let special = [
            Complex64::new(-0.0, 0.0),
            Complex64::new(f64::NAN, 1.0),
            Complex64::new(f64::INFINITY, -0.0),
            Complex64::new(f64::MIN_POSITIVE / 4.0, -2.5),
        ];
        (0..96)
            .map(|i| {
                special
                    .get(i % 11)
                    .copied()
                    .unwrap_or(Complex64::new(i as f64 * 0.75 - 9.0, 1.0 - i as f64 * 0.5))
            })
            .collect()
    }

    fn bits(values: &[Complex64]) -> Vec<(u64, u64)> {
        // Why canonicalize under Miri only: Miri picks arithmetic NaN sign
        // and payload nondeterministically; native runs compare them exactly.
        let bits = |value: f64| {
            if cfg!(miri) && value.is_nan() {
                f64::NAN.to_bits()
            } else {
                value.to_bits()
            }
        };
        values
            .iter()
            .map(|value| (bits(value.re), bits(value.im)))
            .collect()
    }

    /// Padded, transposed source strides (and a reversed axis order) per
    /// block, all in bounds of `values()`.
    fn source_stride(block: usize, axis: usize) -> usize {
        [[1, 5, 30], [2, 1, 9], [15, 1, 3], [40, 13, 1]][block][axis]
    }

    type Writer<'a> = BlockOverwrite<'a, Complex64>;

    fn write_copy(
        allow_uninit: bool,
        conjugate: bool,
        alpha: Complex64,
    ) -> Result<Vec<Complex64>, OperationError> {
        let source = values();
        let mut index = 0;
        overwrite_owned_blocks_in(&tiled(), allow_uninit, |_, writer: &mut Writer<'_>| {
            let block = index;
            index += 1;
            writer.copy(
                |axis| Ok(source_stride(block, axis)),
                &source,
                2 + block,
                conjugate,
                alpha,
            )
        })
    }

    fn write_add(allow_uninit: bool, alpha: Complex64, beta: Complex64) -> Vec<Complex64> {
        let lhs = values();
        let rhs: Vec<Complex64> = values().into_iter().rev().collect();
        let mut index = 0;
        overwrite_owned_blocks_in(&tiled(), allow_uninit, |_, writer: &mut Writer<'_>| {
            let block = index;
            index += 1;
            writer.add(
                |axis| Ok((source_stride(block, axis), source_stride(3 - block, axis))),
                (&lhs, block, block % 2 == 0),
                (&rhs, 1, block % 2 == 1),
                alpha,
                beta,
            )
        })
        .unwrap()
    }

    /// What: a tiled layout skips the fill and stores the same bits as the
    /// base sequence (zero fill, then the initialized kernels), for copies,
    /// scaled and conjugated copies, and every coefficient case of an add.
    #[test]
    fn tiled_blocks_skip_the_fill_and_match_the_zero_fill_sequence_bitwise() {
        let one = Complex64::new(1.0, 0.0);
        let zero = Complex64::new(0.0, 0.0);
        let scaled = Complex64::new(0.5, -0.25);
        for conjugate in [false, true] {
            for alpha in [one, scaled, zero] {
                #[cfg(debug_assertions)]
                take_owned_block_prefills();
                let uninit = write_copy(true, conjugate, alpha).unwrap();
                #[cfg(debug_assertions)]
                assert_eq!(take_owned_block_prefills(), 0);
                let base = write_copy(false, conjugate, alpha).unwrap();
                #[cfg(debug_assertions)]
                assert_eq!(take_owned_block_prefills(), 23);
                assert_eq!(bits(&uninit), bits(&base));
            }
        }
        for (alpha, beta) in [
            (one, scaled),
            (scaled, -one),
            (zero, scaled),
            (scaled, zero),
            (zero, zero),
        ] {
            #[cfg(debug_assertions)]
            take_owned_block_prefills();
            let uninit = write_add(true, alpha, beta);
            #[cfg(debug_assertions)]
            assert_eq!(take_owned_block_prefills(), 0);
            assert_eq!(bits(&uninit), bits(&write_add(false, alpha, beta)));
        }
    }

    /// A U(1) `V ⊗ V <- V ⊗ V` coupled-sector layout: several trees per
    /// coupled sector, so blocks are strided sub-matrices that tile only
    /// jointly (the case the compact proof rejects).
    fn coupled() -> std::sync::Arc<BlockStructure> {
        use tenet_core::{
            FusionProductSpace, FusionTreeHomSpace, SectorLeg, U1FusionRule, U1Irrep,
        };
        let leg = || {
            SectorLeg::new(
                [
                    (U1Irrep::new(-1).sector_id(), 2),
                    (U1Irrep::new(0).sector_id(), 1),
                    (U1Irrep::new(1).sector_id(), 3),
                ],
                false,
            )
        };
        let product = || FusionProductSpace::new([leg(), leg()]);
        FusionTreeHomSpace::new(product(), product())
            .coupled_subblock_structure_from_leg_degeneracies(&U1FusionRule)
            .unwrap()
    }

    /// What: on a coupled-sector layout whose blocks are not contiguous
    /// ranges, the unfilled output stores the base sequence's bits for a
    /// transposing conjugated copy, every add coefficient case (both zero
    /// included), and an explicit zero block.
    #[test]
    fn coupled_sector_blocks_skip_the_fill_and_match_the_zero_fill_sequence_bitwise() {
        let structure = coupled();
        let len = structure.required_len().unwrap();
        assert!((0..structure.block_count()).any(|index| {
            let block = structure.block(index).unwrap();
            block.element_count().unwrap() > 0 && !is_compact(block.shape(), block.strides())
        }));
        assert!(!compact_blocks_tile_storage(&structure, len).unwrap());
        assert!(structure.storage_tiling_proven());
        let lhs = values().into_iter().cycle().take(len).collect::<Vec<_>>();
        let rhs = lhs.iter().rev().copied().collect::<Vec<_>>();
        let one = Complex64::new(1.0, 0.0);
        let zero = Complex64::new(0.0, 0.0);
        let scaled = Complex64::new(0.5, -0.25);
        let run = |allow_uninit: bool, alpha: Complex64, beta: Complex64, mode: u8| {
            overwrite_owned_blocks_in(
                &structure,
                allow_uninit,
                |block, writer: &mut Writer<'_>| {
                    let strides = block.strides();
                    // Reversed axis order reads a transposed, still in-bounds view.
                    let rank = strides.len();
                    match mode {
                        0 => writer.copy(
                            |axis| Ok(strides[rank - 1 - axis]),
                            &lhs,
                            block.offset(),
                            true,
                            alpha,
                        ),
                        1 => writer.add(
                            |axis| Ok((strides[axis], strides[axis])),
                            (&lhs, block.offset(), false),
                            (&rhs, block.offset(), true),
                            alpha,
                            beta,
                        ),
                        _ => writer.zero(),
                    }
                },
            )
            .unwrap()
        };
        for (alpha, beta, mode) in [
            (one, zero, 0),
            (scaled, zero, 0),
            (one, scaled, 1),
            (scaled, -one, 1),
            (zero, scaled, 1),
            (scaled, zero, 1),
            (zero, zero, 1),
            (zero, zero, 2),
        ] {
            #[cfg(debug_assertions)]
            take_owned_block_prefills();
            let uninit = run(true, alpha, beta, mode);
            #[cfg(debug_assertions)]
            assert_eq!(take_owned_block_prefills(), 0);
            assert_eq!(
                bits(&uninit),
                bits(&run(false, alpha, beta, mode)),
                "mode {mode}"
            );
        }
    }

    /// What: layouts without the tiling proof keep the zero fill and the
    /// unwritten storage reads as zero.
    #[test]
    fn untiled_layouts_keep_the_zero_fill() {
        let source = [2.0f64; 16];
        for (layout, len) in [
            // A hole between the blocks.
            (
                structure(&[(&[2, 1, 1], &[1, 1, 1], 0), (&[2, 1, 1], &[1, 1, 1], 3)]),
                5,
            ),
            // Contiguous, but not in block order.
            (
                structure(&[(&[2, 1, 1], &[1, 1, 1], 2), (&[2, 1, 1], &[1, 1, 1], 0)]),
                4,
            ),
            // A padded block.
            (structure(&[(&[2, 2, 1], &[1, 3, 1], 0)]), 5),
        ] {
            #[cfg(debug_assertions)]
            take_owned_block_prefills();
            let data = overwrite_owned_blocks(&layout, |_, writer| {
                writer.copy(|_| Ok(1), &source, 0, false, 1.0)
            })
            .unwrap();
            #[cfg(debug_assertions)]
            assert_eq!(take_owned_block_prefills(), len);
            assert_eq!(data.len(), len);
            assert_eq!(data.iter().filter(|&&value| value == 0.0).count(), len - 4);
        }
    }

    /// What: a writer that leaves a block unwritten, or writes one twice,
    /// is an error, and nothing is returned.
    #[test]
    fn every_block_is_written_exactly_once() {
        let source = values();
        let unwritten = overwrite_owned_blocks(&tiled(), |block, writer: &mut Writer<'_>| {
            if block.shape() == [3, 2, 2] {
                return Ok(());
            }
            writer.zero()
        });
        assert!(matches!(
            unwritten,
            Err(OperationError::InvalidArgument { .. })
        ));
        let twice = overwrite_owned_blocks(&tiled(), |_, writer: &mut Writer<'_>| {
            writer.zero()?;
            writer.copy(|_| Ok(1), &source, 0, false, Complex64::new(1.0, 0.0))
        });
        assert!(matches!(twice, Err(OperationError::InvalidArgument { .. })));
        // A failed write leaves the block claimable, but an error returned
        // by the writer still aborts the output.
        let failed = overwrite_owned_blocks(&tiled(), |_, writer: &mut Writer<'_>| {
            writer.copy(|_| Ok(1), &source[..1], 0, false, Complex64::new(1.0, 0.0))
        });
        assert!(failed.is_err());
        let zeros =
            overwrite_owned_blocks(&tiled(), |_, writer: &mut Writer<'_>| writer.zero()).unwrap();
        assert_eq!(bits(&zeros), bits(&[Complex64::new(0.0, 0.0); 23]));
    }

    /// What: unwinding from a block writer after earlier blocks were
    /// written drops the storage without exposing it (Miri checks the
    /// partially initialized buffer is never read or published).
    #[test]
    fn panicking_block_writer_publishes_nothing() {
        let source = values();
        let result = std::panic::catch_unwind(|| {
            let mut index = 0;
            let _ = overwrite_owned_blocks(&tiled(), |_, writer: &mut Writer<'_>| {
                index += 1;
                if index == 3 {
                    panic!("injected block writer panic");
                }
                writer.copy(|_| Ok(1), &source, 0, false, Complex64::new(1.0, 0.0))
            });
        });
        assert!(result.is_err());
    }
}
