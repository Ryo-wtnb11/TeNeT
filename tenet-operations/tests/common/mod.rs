//! Fixtures and the independent oracle shared by the device tree-transform
//! tests and by the CPU-only test that proves the oracle itself.
//!
//! The oracle is an explicit index walk over the fixture's own block metadata:
//! it never reads a `TreeTransformStructure`, its baked fused layouts, or any
//! executor state, so "device matches the oracle" and "host matches the oracle"
//! are independent statements about the same fixture.

#![allow(dead_code)]

use std::sync::Arc;

use num_complex::Complex64;
use tenet_core::{BlockKey, BlockSpec, BlockStructure};
use tenet_operations::{TreeTransformBlockSpec, TreeTransformStructure};

/// Payload dtypes the fixtures are replayed with, plus the host arithmetic the
/// oracle needs. Deliberately not a TeNeT trait: the oracle must not share a
/// scalar contract with the code under test.
pub trait TestScalar: Copy + std::fmt::Debug + PartialEq + 'static {
    const NAME: &'static str;
    fn from_parts(re: f64, im: f64) -> Self;
    fn zero() -> Self {
        Self::from_parts(0.0, 0.0)
    }
    fn scale(self, factor: f64) -> Self;
    fn add(self, other: Self) -> Self;
    fn conjugate(self) -> Self;
    fn distance(self, other: Self) -> f64;
    /// Deterministic, non-degenerate sample for buffer index `index`.
    fn sample(index: usize) -> Self {
        Self::from_parts(1.0 + (index as f64) * 0.5, -0.25 + (index as f64) * 0.125)
    }
    fn is_nan(self) -> bool;
}

impl TestScalar for f64 {
    const NAME: &'static str = "f64";

    fn from_parts(re: f64, _im: f64) -> Self {
        re
    }

    fn scale(self, factor: f64) -> Self {
        self * factor
    }

    fn add(self, other: Self) -> Self {
        self + other
    }

    fn conjugate(self) -> Self {
        self
    }

    fn distance(self, other: Self) -> f64 {
        (self - other).abs()
    }

    fn is_nan(self) -> bool {
        f64::is_nan(self)
    }
}

impl TestScalar for Complex64 {
    const NAME: &'static str = "Complex64";

    fn from_parts(re: f64, im: f64) -> Self {
        Complex64::new(re, im)
    }

    fn scale(self, factor: f64) -> Self {
        self * factor
    }

    fn add(self, other: Self) -> Self {
        self + other
    }

    fn conjugate(self) -> Self {
        Complex64::new(self.re, -self.im)
    }

    fn distance(self, other: Self) -> f64 {
        (self - other).norm()
    }

    fn is_nan(self) -> bool {
        self.re.is_nan() || self.im.is_nan()
    }
}

/// One block of a fixture's storage: extents, element strides, flat offset.
#[derive(Clone, Debug)]
pub struct Block {
    pub shape: Vec<usize>,
    pub strides: Vec<usize>,
    pub offset: usize,
}

impl Block {
    pub fn packed(shape: Vec<usize>, offset: usize) -> Self {
        let mut strides = Vec::with_capacity(shape.len());
        let mut running = 1usize;
        for &dim in &shape {
            strides.push(running);
            running *= dim;
        }
        Self {
            shape,
            strides,
            offset,
        }
    }

    pub fn element_count(&self) -> usize {
        self.shape.iter().product()
    }

    fn end_exclusive(&self) -> usize {
        if self.shape.contains(&0) {
            return self.offset;
        }
        self.offset
            + self
                .shape
                .iter()
                .zip(&self.strides)
                .map(|(dim, stride)| (dim - 1) * stride)
                .sum::<usize>()
            + 1
    }

    fn spec(&self, index: usize) -> BlockSpec {
        BlockSpec::with_key(
            BlockKey::ordinal(index),
            self.shape.clone(),
            self.strides.clone(),
            self.offset,
        )
        .unwrap()
    }
}

/// One replayed block pair: `axes[dst_axis] = src_axis`.
#[derive(Clone, Debug)]
pub struct Pair {
    pub dst_block: usize,
    pub src_block: usize,
    pub axes: Vec<usize>,
    pub coefficient: f64,
}

/// A complete tree-transform fixture: two block structures, the block pairs the
/// transform replays, and whether the source is read conjugated.
#[derive(Clone, Debug)]
pub struct Fixture {
    pub name: &'static str,
    pub rank: usize,
    pub dst_blocks: Vec<Block>,
    pub src_blocks: Vec<Block>,
    pub pairs: Vec<Pair>,
    pub conjugate: bool,
}

impl Fixture {
    pub fn dst_structure(&self) -> Arc<BlockStructure> {
        Arc::new(
            BlockStructure::from_blocks_with_rank(
                self.rank,
                self.dst_blocks
                    .iter()
                    .enumerate()
                    .map(|(index, block)| block.spec(index))
                    .collect(),
            )
            .unwrap(),
        )
    }

    pub fn src_structure(&self) -> Arc<BlockStructure> {
        Arc::new(
            BlockStructure::from_blocks_with_rank(
                self.rank,
                self.src_blocks
                    .iter()
                    .enumerate()
                    .map(|(index, block)| block.spec(index))
                    .collect(),
            )
            .unwrap(),
        )
    }

    pub fn dst_len(&self) -> usize {
        self.dst_blocks
            .iter()
            .map(Block::end_exclusive)
            .max()
            .unwrap_or(0)
    }

    pub fn src_len(&self) -> usize {
        self.src_blocks
            .iter()
            .map(Block::end_exclusive)
            .max()
            .unwrap_or(0)
    }

    pub fn compile(&self) -> TreeTransformStructure<f64> {
        let specs: Vec<TreeTransformBlockSpec<f64>> = self
            .pairs
            .iter()
            .map(|pair| {
                TreeTransformBlockSpec::single(pair.dst_block, pair.src_block, pair.coefficient)
                    .with_source_axes(pair.axes.iter().copied())
            })
            .collect();
        TreeTransformStructure::compile_structures_with_storage_conjugation(
            &self.dst_structure(),
            &self.src_structure(),
            &specs,
            self.conjugate,
        )
        .unwrap()
    }

    pub fn source<T: TestScalar>(&self) -> Vec<T> {
        (0..self.src_len()).map(T::sample).collect()
    }

    /// The expected destination buffer, computed by walking every block pair's
    /// index space explicitly.
    ///
    /// `Overwrite` assigns the active elements and zeroes every destination
    /// element that belongs to a block no pair writes; accumulate (`beta == 1`)
    /// adds into `destination` and leaves untouched blocks alone. Nothing here
    /// consults the compiled structure.
    pub fn expected<T: TestScalar>(
        &self,
        source: &[T],
        destination: &[T],
        overwrite: bool,
    ) -> Vec<T> {
        let mut expected = destination.to_vec();
        if overwrite {
            let written: Vec<usize> = self.pairs.iter().map(|pair| pair.dst_block).collect();
            for (index, block) in self.dst_blocks.iter().enumerate() {
                if written.contains(&index) {
                    continue;
                }
                for position in positions(block, &(0..self.rank).collect::<Vec<_>>()) {
                    expected[position] = T::zero();
                }
            }
        }
        for pair in &self.pairs {
            let dst = &self.dst_blocks[pair.dst_block];
            let src = &self.src_blocks[pair.src_block];
            let identity: Vec<usize> = (0..self.rank).collect();
            for (dst_position, src_position) in
                positions(dst, &identity).zip(positions(src, &pair.axes))
            {
                let mut value = source[src_position];
                if self.conjugate {
                    value = value.conjugate();
                }
                let value = value.scale(pair.coefficient);
                expected[dst_position] = if overwrite {
                    value
                } else {
                    expected[dst_position].add(value)
                };
            }
        }
        expected
    }
}

/// Flat positions of `block`, walked in the *destination* axis order given by
/// `axes` (`axes[dst_axis] = block_axis`), fastest destination axis first.
fn positions<'a>(block: &'a Block, axes: &'a [usize]) -> impl Iterator<Item = usize> + 'a {
    let extents: Vec<usize> = axes.iter().map(|&axis| block.shape[axis]).collect();
    let strides: Vec<usize> = axes.iter().map(|&axis| block.strides[axis]).collect();
    let count: usize = extents.iter().product();
    (0..count).map(move |linear| {
        let mut remaining = linear;
        let mut position = block.offset;
        for (extent, stride) in extents.iter().zip(&strides) {
            position += (remaining % extent) * stride;
            remaining /= extent;
        }
        position
    })
}

fn reversed(rank: usize) -> Vec<usize> {
    (0..rank).rev().collect()
}

fn permuted(shape: &[usize], axes: &[usize]) -> Vec<usize> {
    axes.iter().map(|&axis| shape[axis]).collect()
}

/// A destination block whose extents are `axes`-permuted from the source's.
fn permuted_pair(
    src_shape: Vec<usize>,
    axes: Vec<usize>,
    coefficient: f64,
    dst_offset: usize,
    src_offset: usize,
) -> (Block, Block, Pair) {
    let dst_shape = permuted(&src_shape, &axes);
    (
        Block::packed(dst_shape, dst_offset),
        Block::packed(src_shape, src_offset),
        Pair {
            dst_block: 0,
            src_block: 0,
            axes,
            coefficient,
        },
    )
}

fn single_pair_fixture(
    name: &'static str,
    src_shape: Vec<usize>,
    axes: Vec<usize>,
    coefficient: f64,
    conjugate: bool,
) -> Fixture {
    let rank = src_shape.len();
    let (dst, src, pair) = permuted_pair(src_shape, axes, coefficient, 0, 0);
    Fixture {
        name,
        rank,
        dst_blocks: vec![dst],
        src_blocks: vec![src],
        pairs: vec![pair],
        conjugate,
    }
}

/// Rank 2–6 permutes and transposes, including a fermionic sign (coefficient
/// −1), a coefficient that is neither 1 nor −1, and a conjugated source.
pub fn rank_sweep() -> Vec<Fixture> {
    vec![
        single_pair_fixture("rank2_transpose", vec![3, 2], vec![1, 0], 1.0, false),
        single_pair_fixture("rank2_fermionic_sign", vec![3, 2], vec![1, 0], -1.0, false),
        single_pair_fixture(
            "rank3_cyclic_permute",
            vec![2, 3, 4],
            vec![2, 0, 1],
            1.0,
            false,
        ),
        single_pair_fixture(
            "rank3_coefficient_not_unit",
            vec![2, 3, 4],
            vec![1, 2, 0],
            0.625,
            false,
        ),
        single_pair_fixture(
            "rank4_reverse_conjugated",
            vec![2, 3, 2, 2],
            reversed(4),
            -1.0,
            true,
        ),
        single_pair_fixture(
            "rank5_permute",
            vec![2, 2, 3, 2, 2],
            vec![3, 0, 4, 1, 2],
            0.75,
            false,
        ),
        single_pair_fixture(
            "rank6_reverse",
            vec![2, 2, 2, 3, 2, 2],
            reversed(6),
            -1.0,
            false,
        ),
        single_pair_fixture("rank2_identity", vec![4, 5], vec![0, 1], 1.0, false),
    ]
}

/// Several blocks whose storage interleaves: the destination blocks are not in
/// the source's order and their layouts are gapped sub-boxes of a wider parent.
pub fn interleaved_multi_block() -> Fixture {
    // Destination storage: block 0 occupies the even columns of a 4x4 parent,
    // block 1 the odd ones, so the two layouts interleave in memory.
    let dst_blocks = vec![
        Block {
            shape: vec![4, 2],
            strides: vec![1, 8],
            offset: 0,
        },
        Block {
            shape: vec![4, 2],
            strides: vec![1, 8],
            offset: 4,
        },
    ];
    let src_blocks = vec![
        Block::packed(vec![2, 4], 0),
        Block::packed(vec![2, 4], 8),
        Block::packed(vec![3, 3], 16),
    ];
    Fixture {
        name: "interleaved_multi_block",
        rank: 2,
        dst_blocks,
        src_blocks,
        // Reversed pair order as well: block order is the structure's, and the
        // destinations are disjoint, so the result must not depend on it.
        pairs: vec![
            Pair {
                dst_block: 1,
                src_block: 0,
                axes: vec![1, 0],
                coefficient: -1.0,
            },
            Pair {
                dst_block: 0,
                src_block: 1,
                axes: vec![1, 0],
                coefficient: 2.5,
            },
        ],
        conjugate: false,
    }
}

/// A destination with layouts no pair writes, which `Overwrite` must zero even
/// when they hold NaN.
pub fn inactive_destination_layouts() -> Fixture {
    let dst_blocks = vec![
        Block::packed(vec![2, 3], 0),
        Block::packed(vec![3, 2], 6),
        Block::packed(vec![2, 2], 12),
    ];
    let src_blocks = vec![Block::packed(vec![3, 2], 0)];
    Fixture {
        name: "inactive_destination_layouts",
        rank: 2,
        dst_blocks,
        src_blocks,
        pairs: vec![Pair {
            dst_block: 0,
            src_block: 0,
            axes: vec![1, 0],
            coefficient: -1.0,
        }],
        conjugate: false,
    }
}

/// A structure with more distinct baked layouts than Tenferro's default
/// 64-entry cuTENSOR plan bound, so a replay that does not raise the bound
/// evicts a plan it needs again on the next block.
pub fn many_distinct_signatures(blocks: usize) -> Fixture {
    let mut dst_blocks = Vec::with_capacity(blocks);
    let mut src_blocks = Vec::with_capacity(blocks);
    let mut pairs = Vec::with_capacity(blocks);
    let mut dst_offset = 0usize;
    let mut src_offset = 0usize;
    for index in 0..blocks {
        // Distinct extents give distinct fused signatures; the source is
        // transposed so the move is never a plain copy.
        let shape = vec![2 + index % 7, 3 + index / 7];
        let dst = Block::packed(vec![shape[1], shape[0]], dst_offset);
        let src = Block::packed(shape, src_offset);
        dst_offset += dst.element_count();
        src_offset += src.element_count();
        pairs.push(Pair {
            dst_block: index,
            src_block: index,
            axes: vec![1, 0],
            coefficient: if index % 2 == 0 { 1.0 } else { -1.0 },
        });
        dst_blocks.push(dst);
        src_blocks.push(src);
    }
    Fixture {
        name: "many_distinct_signatures",
        rank: 2,
        dst_blocks,
        src_blocks,
        pairs,
        conjugate: false,
    }
}

/// A fixture containing a zero-extent block pair next to a live one.
pub fn zero_extent_block() -> Fixture {
    Fixture {
        name: "zero_extent_block",
        rank: 2,
        dst_blocks: vec![
            Block::packed(vec![2, 3], 0),
            Block {
                shape: vec![0, 2],
                strides: vec![1, 1],
                offset: 6,
            },
        ],
        src_blocks: vec![
            Block::packed(vec![3, 2], 0),
            Block {
                shape: vec![2, 0],
                strides: vec![1, 1],
                offset: 6,
            },
        ],
        pairs: vec![
            Pair {
                dst_block: 0,
                src_block: 0,
                axes: vec![1, 0],
                coefficient: 1.5,
            },
            Pair {
                dst_block: 1,
                src_block: 1,
                axes: vec![1, 0],
                coefficient: 1.0,
            },
        ],
        conjugate: false,
    }
}

/// A destination layout the host proves injective only through its exact
/// overlap fallback: dims [3, 2] with strides [2, 3] addresses 0, 2, 4, 3, 5, 7
/// — distinct, but each stride does not exceed the span of the faster axis, so
/// the device's cumulative-span rule (and Tenferro's) rejects it.
///
/// Host replays it; the device must report it as a capability boundary, and
/// must do so before it has cost the caller anything.
pub fn expert_interleaved_destination() -> Fixture {
    Fixture {
        name: "expert_interleaved_destination",
        rank: 2,
        dst_blocks: vec![Block {
            shape: vec![3, 2],
            strides: vec![2, 3],
            offset: 0,
        }],
        src_blocks: vec![Block::packed(vec![2, 3], 0)],
        pairs: vec![Pair {
            dst_block: 0,
            src_block: 0,
            axes: vec![1, 0],
            coefficient: 1.0,
        }],
        conjugate: false,
    }
}

/// The fixtures whose every coefficient is exactly 1, where an f64 device move
/// is a multiply by one and therefore bit-identical to the host's copy.
pub fn unit_coefficient_fixtures() -> Vec<Fixture> {
    all_fixtures()
        .into_iter()
        .filter(|fixture| fixture.pairs.iter().all(|pair| pair.coefficient == 1.0))
        .collect()
}

/// Every fixture the device and host suites both replay.
pub fn all_fixtures() -> Vec<Fixture> {
    let mut fixtures = rank_sweep();
    fixtures.push(interleaved_multi_block());
    fixtures.push(inactive_destination_layouts());
    fixtures.push(zero_extent_block());
    fixtures
}
