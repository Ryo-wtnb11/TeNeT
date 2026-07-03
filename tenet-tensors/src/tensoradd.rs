use core::ops::{Add, Mul};
use std::sync::Arc;

use num_traits::{One, Zero};
use tenet_core::{
    BlockKey, BlockStructure, HostReadableStorage, HostWritableStorage, TensorMap, TensorStorage,
};

use crate::axis::{permutation_axes, AxisPermutation};
use crate::error::OperationError;
use crate::strided::offset_to_isize;
use crate::structure_identity::validate_structure_identity;
use crate::{ConjugateValue, TensorOperationsBackend};

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TensorAddStructure {
    rank: usize,
    axes: Vec<usize>,
    terms: Vec<TensorAddStructureTerm>,
    descriptor: TensorAddDescriptor,
    dst_structure: Arc<BlockStructure>,
    src_structure: Arc<BlockStructure>,
}

pub fn tensoradd_structure<
    TDst,
    TSrc,
    const NOUT: usize,
    const NIN: usize,
    SDst,
    SSrc,
    DDst,
    DSrc,
>(
    dst: &TensorMap<TDst, NOUT, NIN, SDst, DDst>,
    src: &TensorMap<TSrc, NOUT, NIN, SSrc, DSrc>,
    permutation: AxisPermutation<'_>,
) -> Result<TensorAddStructure, OperationError>
where
    DDst: TensorStorage<TDst>,
    DSrc: TensorStorage<TSrc>,
{
    TensorAddStructure::compile(dst, src, permutation)
}

pub fn tensoradd_structure_with_conjugation<
    TDst,
    TSrc,
    const NOUT: usize,
    const NIN: usize,
    SDst,
    SSrc,
    DDst,
    DSrc,
>(
    dst: &TensorMap<TDst, NOUT, NIN, SDst, DDst>,
    src: &TensorMap<TSrc, NOUT, NIN, SSrc, DSrc>,
    permutation: AxisPermutation<'_>,
    source_conjugate: bool,
) -> Result<TensorAddStructure, OperationError>
where
    DDst: TensorStorage<TDst>,
    DSrc: TensorStorage<TSrc>,
{
    TensorAddStructure::compile_with_conjugation(dst, src, permutation, source_conjugate)
}

pub(crate) const PLAIN_TENSORADD_FUSION_PERMUTE_REQUIRES_TREE_TRANSFORM: &str =
    "plain tensoradd does not lower fusion-tree permutations; use tree_pair_transform_*";
pub(crate) const PLAIN_TENSORADD_FUSION_CONJUGATION_REQUIRES_CATEGORICAL_ADJOINT: &str =
    "plain tensoradd with fusion-tree conjugation requires categorical adjoint lowering";

impl TensorAddStructure {
    pub fn compile<TDst, TSrc, const NOUT: usize, const NIN: usize, SDst, SSrc, DDst, DSrc>(
        dst: &TensorMap<TDst, NOUT, NIN, SDst, DDst>,
        src: &TensorMap<TSrc, NOUT, NIN, SSrc, DSrc>,
        permutation: AxisPermutation<'_>,
    ) -> Result<Self, OperationError>
    where
        DDst: TensorStorage<TDst>,
        DSrc: TensorStorage<TSrc>,
    {
        Self::compile_with_conjugation(dst, src, permutation, false)
    }

    pub fn compile_with_conjugation<
        TDst,
        TSrc,
        const NOUT: usize,
        const NIN: usize,
        SDst,
        SSrc,
        DDst,
        DSrc,
    >(
        dst: &TensorMap<TDst, NOUT, NIN, SDst, DDst>,
        src: &TensorMap<TSrc, NOUT, NIN, SSrc, DSrc>,
        permutation: AxisPermutation<'_>,
        source_conjugate: bool,
    ) -> Result<Self, OperationError>
    where
        DDst: TensorStorage<TDst>,
        DSrc: TensorStorage<TSrc>,
    {
        let axes = permutation_axes(permutation, dst.structure().rank())?;
        let has_fusion = dst.fusion_space().is_some() || src.fusion_space().is_some();
        if source_conjugate && has_fusion {
            return Err(OperationError::UnsupportedTensorContractScope {
                message: PLAIN_TENSORADD_FUSION_CONJUGATION_REQUIRES_CATEGORICAL_ADJOINT,
            });
        }
        if !axes.iter().copied().eq(0..dst.structure().rank()) && has_fusion {
            return Err(OperationError::UnsupportedTensorContractScope {
                message: PLAIN_TENSORADD_FUSION_PERMUTE_REQUIRES_TREE_TRANSFORM,
            });
        }
        Self::compile_shared_structures(
            Arc::clone(dst.structure()),
            Arc::clone(src.structure()),
            permutation,
            source_conjugate,
        )
    }

    pub fn compile_structures(
        dst_structure: &BlockStructure,
        src_structure: &BlockStructure,
        permutation: AxisPermutation<'_>,
    ) -> Result<Self, OperationError> {
        Self::compile_shared_structures(
            Arc::new(dst_structure.clone()),
            Arc::new(src_structure.clone()),
            permutation,
            false,
        )
    }

    fn compile_shared_structures(
        dst_structure: Arc<BlockStructure>,
        src_structure: Arc<BlockStructure>,
        permutation: AxisPermutation<'_>,
        source_conjugate: bool,
    ) -> Result<Self, OperationError> {
        if dst_structure.block_count() != src_structure.block_count() {
            return Err(OperationError::BlockCountMismatch {
                dst: dst_structure.block_count(),
                src: src_structure.block_count(),
            });
        }

        let rank = dst_structure.rank();
        if src_structure.rank() != rank {
            return Err(OperationError::StructureRankMismatch {
                expected: rank,
                actual: src_structure.rank(),
            });
        }
        let axes = permutation_axes(permutation, rank)?;
        let src_for_dst = dst_structure
            .pair_block_indices_from(&src_structure)
            .map_err(OperationError::from_core_preserving_context)?;
        let mut terms = Vec::with_capacity(dst_structure.block_count());

        for dst_index in 0..dst_structure.block_count() {
            let dst_block = dst_structure.block(dst_index)?;
            if dst_block.shape().len() != rank {
                return Err(OperationError::RankMismatch {
                    expected: rank,
                    actual: dst_block.shape().len(),
                });
            }
            let src_index = src_for_dst[dst_index];
            let src_block = src_structure.block(src_index)?;
            if src_block.shape().len() != rank {
                return Err(OperationError::RankMismatch {
                    expected: rank,
                    actual: src_block.shape().len(),
                });
            }

            terms.push(TensorAddStructureTerm {
                key: dst_block.key().clone(),
                dst_block: dst_index,
                src_block: src_index,
            });
        }

        let descriptor = TensorAddDescriptor::compile(
            rank,
            &axes,
            source_conjugate,
            &terms,
            &dst_structure,
            &src_structure,
        )?;

        Ok(Self {
            rank,
            axes,
            terms,
            descriptor,
            dst_structure,
            src_structure,
        })
    }

    #[inline]
    pub fn rank(&self) -> usize {
        self.rank
    }

    #[inline]
    pub fn axes(&self) -> &[usize] {
        &self.axes
    }

    #[inline]
    pub fn terms(&self) -> &[TensorAddStructureTerm] {
        &self.terms
    }

    #[inline]
    pub(crate) fn descriptor(&self) -> &TensorAddDescriptor {
        &self.descriptor
    }

    pub(crate) fn validate_replay_structures(
        &self,
        dst_structure: &Arc<BlockStructure>,
        src_structure: &Arc<BlockStructure>,
    ) -> Result<(), OperationError> {
        validate_structure_identity("dst", &self.dst_structure, dst_structure)?;
        validate_structure_identity("src", &self.src_structure, src_structure)
    }

    pub fn execute_with<B, T, const NOUT: usize, const NIN: usize, S, DDst, DSrc>(
        &self,
        backend: &mut B,
        allocator: &mut B::Allocator,
        dst: &mut TensorMap<T, NOUT, NIN, S, DDst>,
        src: &TensorMap<T, NOUT, NIN, S, DSrc>,
        alpha: T,
        beta: T,
    ) -> Result<(), OperationError>
    where
        B: TensorOperationsBackend,
        T: Copy
            + Add<T, Output = T>
            + Mul<T, Output = T>
            + PartialEq
            + Zero
            + One
            + ConjugateValue
            + strided_kernel::MaybeSendSync,
        DDst: HostWritableStorage<T>,
        DSrc: HostReadableStorage<T>,
    {
        backend.tensoradd_structure_into(allocator, self, dst, src, alpha, beta)
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TensorAddStructureTerm {
    key: BlockKey,
    dst_block: usize,
    src_block: usize,
}

impl TensorAddStructureTerm {
    #[inline]
    pub fn key(&self) -> &BlockKey {
        &self.key
    }

    #[inline]
    pub fn dst_block(&self) -> usize {
        self.dst_block
    }

    #[inline]
    pub fn src_block(&self) -> usize {
        self.src_block
    }
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub(crate) struct TensorAddDescriptor {
    source_conjugate: bool,
    terms: Vec<TensorAddDescriptorTerm>,
    shapes: Vec<usize>,
    dst_strides: Vec<isize>,
    src_strides: Vec<isize>,
}

impl TensorAddDescriptor {
    #[inline]
    pub(crate) fn terms(&self) -> &[TensorAddDescriptorTerm] {
        &self.terms
    }

    #[inline]
    pub(crate) fn source_conjugate(&self) -> bool {
        self.source_conjugate
    }

    fn reserve(&mut self, term_count: usize, rank: usize) {
        self.terms.reserve(term_count);
        let entry_count = term_count.saturating_mul(rank);
        self.shapes.reserve(entry_count);
        self.dst_strides.reserve(entry_count);
        self.src_strides.reserve(entry_count);
    }

    pub(crate) fn shape(&self, term: &TensorAddDescriptorTerm) -> &[usize] {
        &self.shapes[term.layout_start..term.layout_start + term.rank]
    }

    pub(crate) fn dst_strides(&self, term: &TensorAddDescriptorTerm) -> &[isize] {
        &self.dst_strides[term.layout_start..term.layout_start + term.rank]
    }

    pub(crate) fn src_strides(&self, term: &TensorAddDescriptorTerm) -> &[isize] {
        &self.src_strides[term.layout_start..term.layout_start + term.rank]
    }

    fn compile(
        rank: usize,
        axes: &[usize],
        source_conjugate: bool,
        terms: &[TensorAddStructureTerm],
        dst_structure: &BlockStructure,
        src_structure: &BlockStructure,
    ) -> Result<Self, OperationError> {
        let mut descriptor = Self {
            source_conjugate,
            ..Self::default()
        };
        descriptor.reserve(terms.len(), rank);

        for term in terms {
            let dst_block = dst_structure.block(term.dst_block())?;
            let src_block = src_structure.block(term.src_block())?;
            if dst_block.shape().len() != rank {
                return Err(OperationError::RankMismatch {
                    expected: rank,
                    actual: dst_block.shape().len(),
                });
            }
            if src_block.shape().len() != rank {
                return Err(OperationError::RankMismatch {
                    expected: rank,
                    actual: src_block.shape().len(),
                });
            }

            let layout_start = descriptor.shapes.len();
            for (dst_axis, &src_axis) in axes.iter().enumerate() {
                let dst_dim = dst_block.shape()[dst_axis];
                let src_dim = src_block.shape()[src_axis];
                if dst_dim != src_dim {
                    let src_shape = axes
                        .iter()
                        .map(|&axis| src_block.shape()[axis])
                        .collect::<Vec<_>>();
                    return Err(OperationError::ShapeMismatch {
                        dst: dst_block.shape().to_vec(),
                        src: src_shape,
                    });
                }
                descriptor.shapes.push(dst_dim);
                descriptor.dst_strides.push(
                    isize::try_from(dst_block.strides()[dst_axis]).map_err(|_| {
                        OperationError::StrideOverflow {
                            value: dst_block.strides()[dst_axis],
                        }
                    })?,
                );
                descriptor.src_strides.push(
                    isize::try_from(src_block.strides()[src_axis]).map_err(|_| {
                        OperationError::StrideOverflow {
                            value: src_block.strides()[src_axis],
                        }
                    })?,
                );
            }

            descriptor.terms.push(TensorAddDescriptorTerm {
                dst_block: term.dst_block(),
                src_block: term.src_block(),
                layout_start,
                rank,
                dst_offset: offset_to_isize(dst_block.offset())?,
                src_offset: offset_to_isize(src_block.offset())?,
            });
        }

        Ok(descriptor)
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct TensorAddDescriptorTerm {
    pub(crate) dst_block: usize,
    pub(crate) src_block: usize,
    pub(crate) layout_start: usize,
    pub(crate) rank: usize,
    pub(crate) dst_offset: isize,
    pub(crate) src_offset: isize,
}
