//! Tree-transform plan data: block/group specs (destination trees, source
//! trees, recoupling matrices) and the grouped plan container. Pure data —
//! the symmetric compile layer in `tenet-tensors` builds these from fusion
//! rules; replay consumes them without any symmetry knowledge.

use std::sync::Arc;

use tenet_core::{
    BlockKey, BlockStructure, FusionTreeBlockGroup, FusionTreeGroupKey, TensorMap, TensorStorage,
};

use crate::transform_helpers::{block_indices_for_keys, fusion_tree_group_block_keys};
use crate::transform_structure::TreeTransformStructure;
use crate::OperationError;

#[derive(Clone, Debug)]
enum SpecEntries<K, T> {
    Single {
        dst: K,
        src: K,
        coefficient: T,
    },
    Multi {
        dst: Vec<K>,
        src: Vec<K>,
        coefficients: Vec<T>,
    },
}

impl<K, T> SpecEntries<K, T> {
    #[inline]
    fn dst(&self) -> &[K] {
        match self {
            Self::Single { dst, .. } => std::slice::from_ref(dst),
            Self::Multi { dst, .. } => dst,
        }
    }

    #[inline]
    fn src(&self) -> &[K] {
        match self {
            Self::Single { src, .. } => std::slice::from_ref(src),
            Self::Multi { src, .. } => src,
        }
    }

    #[inline]
    fn coefficients(&self) -> &[T] {
        match self {
            Self::Single { coefficient, .. } => std::slice::from_ref(coefficient),
            Self::Multi { coefficients, .. } => coefficients,
        }
    }
}

impl<K: PartialEq, T: PartialEq> PartialEq for SpecEntries<K, T> {
    fn eq(&self, other: &Self) -> bool {
        self.dst() == other.dst()
            && self.src() == other.src()
            && self.coefficients() == other.coefficients()
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct TreeTransformBlockSpec<T> {
    entries: SpecEntries<usize, T>,
    source_axes: Option<Arc<[usize]>>,
}

impl<T> TreeTransformBlockSpec<T> {
    pub fn single(dst_block: usize, src_block: usize, coefficient: T) -> Self {
        Self {
            entries: SpecEntries::Single {
                dst: dst_block,
                src: src_block,
                coefficient,
            },
            source_axes: None,
        }
    }

    pub fn multi(
        dst_blocks: Vec<usize>,
        src_blocks: Vec<usize>,
        recoupling_coefficients_dst_src: Vec<T>,
    ) -> Self {
        Self {
            entries: SpecEntries::Multi {
                dst: dst_blocks,
                src: src_blocks,
                coefficients: recoupling_coefficients_dst_src,
            },
            source_axes: None,
        }
    }

    pub fn with_source_axes<I>(mut self, axes: I) -> Self
    where
        I: IntoIterator<Item = usize>,
    {
        self.source_axes = Some(axes.into_iter().collect());
        self
    }

    fn with_optional_source_axes(mut self, axes: Option<Arc<[usize]>>) -> Self {
        self.source_axes = axes;
        self
    }

    #[inline]
    pub fn dst_blocks(&self) -> &[usize] {
        self.entries.dst()
    }

    #[inline]
    pub fn src_blocks(&self) -> &[usize] {
        self.entries.src()
    }

    /// Recoupling matrix coefficients stored as `U[dst, src]` in row-major
    /// destination-by-source order: `coeff[src + dst * src_count]`.
    #[inline]
    pub fn recoupling_coefficients_dst_src(&self) -> &[T] {
        self.entries.coefficients()
    }

    #[inline]
    pub fn source_axes(&self) -> Option<&[usize]> {
        self.source_axes.as_deref()
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct TreeTransformKeyBlockSpec<T> {
    entries: SpecEntries<BlockKey, T>,
    source_axes: Option<Arc<[usize]>>,
}

impl<T> TreeTransformKeyBlockSpec<T> {
    pub fn single<KDst, KSrc>(dst_key: KDst, src_key: KSrc, coefficient: T) -> Self
    where
        KDst: Into<BlockKey>,
        KSrc: Into<BlockKey>,
    {
        Self {
            entries: SpecEntries::Single {
                dst: dst_key.into(),
                src: src_key.into(),
                coefficient,
            },
            source_axes: None,
        }
    }

    pub fn multi<DstKeys, SrcKeys, KDst, KSrc>(
        dst_keys: DstKeys,
        src_keys: SrcKeys,
        recoupling_coefficients_dst_src: Vec<T>,
    ) -> Self
    where
        DstKeys: IntoIterator<Item = KDst>,
        SrcKeys: IntoIterator<Item = KSrc>,
        KDst: Into<BlockKey>,
        KSrc: Into<BlockKey>,
    {
        Self {
            entries: SpecEntries::Multi {
                dst: dst_keys.into_iter().map(Into::into).collect(),
                src: src_keys.into_iter().map(Into::into).collect(),
                coefficients: recoupling_coefficients_dst_src,
            },
            source_axes: None,
        }
    }

    pub fn with_source_axes<I>(mut self, axes: I) -> Self
    where
        I: IntoIterator<Item = usize>,
    {
        self.source_axes = Some(axes.into_iter().collect());
        self
    }

    #[inline]
    pub fn dst_keys(&self) -> &[BlockKey] {
        self.entries.dst()
    }

    #[inline]
    pub fn src_keys(&self) -> &[BlockKey] {
        self.entries.src()
    }

    /// Recoupling matrix coefficients stored as `U[dst, src]` in row-major
    /// destination-by-source order: `coeff[src + dst * src_count]`.
    #[inline]
    pub fn recoupling_coefficients_dst_src(&self) -> &[T] {
        self.entries.coefficients()
    }

    #[inline]
    pub fn source_axes(&self) -> Option<&[usize]> {
        self.source_axes.as_deref()
    }
}

impl<T: Copy> TreeTransformKeyBlockSpec<T> {
    pub(crate) fn to_indexed_spec(
        &self,
        dst_structure: &BlockStructure,
        src_structure: &BlockStructure,
    ) -> Result<TreeTransformBlockSpec<T>, OperationError> {
        let spec = match &self.entries {
            SpecEntries::Single {
                dst,
                src,
                coefficient,
            } => {
                let dst_block = dst_structure.find_block_index_by_key(dst).ok_or_else(|| {
                    OperationError::MissingBlockKey {
                        key: Box::new(dst.clone()),
                    }
                })?;
                let src_block = src_structure.find_block_index_by_key(src).ok_or_else(|| {
                    OperationError::MissingBlockKey {
                        key: Box::new(src.clone()),
                    }
                })?;
                TreeTransformBlockSpec::single(dst_block, src_block, *coefficient)
            }
            SpecEntries::Multi {
                dst,
                src,
                coefficients,
            } => TreeTransformBlockSpec::multi(
                block_indices_for_keys(dst_structure, dst)?,
                block_indices_for_keys(src_structure, src)?,
                coefficients.clone(),
            ),
        };
        Ok(spec.with_optional_source_axes(self.source_axes.clone()))
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct TreeTransformGroupBlockSpec<T> {
    group_key: FusionTreeGroupKey,
    entries: SpecEntries<BlockKey, T>,
    source_axes: Option<Arc<[usize]>>,
}

impl<T> TreeTransformGroupBlockSpec<T> {
    pub fn single<KDst, KSrc>(
        group_key: FusionTreeGroupKey,
        dst_key: KDst,
        src_key: KSrc,
        coefficient: T,
    ) -> Self
    where
        KDst: Into<BlockKey>,
        KSrc: Into<BlockKey>,
    {
        Self {
            group_key,
            entries: SpecEntries::Single {
                dst: dst_key.into(),
                src: src_key.into(),
                coefficient,
            },
            source_axes: None,
        }
    }

    pub fn multi<DstKeys, SrcKeys, KDst, KSrc>(
        group_key: FusionTreeGroupKey,
        dst_keys: DstKeys,
        src_keys: SrcKeys,
        recoupling_coefficients_dst_src: Vec<T>,
    ) -> Self
    where
        DstKeys: IntoIterator<Item = KDst>,
        SrcKeys: IntoIterator<Item = KSrc>,
        KDst: Into<BlockKey>,
        KSrc: Into<BlockKey>,
    {
        Self {
            group_key,
            entries: SpecEntries::Multi {
                dst: dst_keys.into_iter().map(Into::into).collect(),
                src: src_keys.into_iter().map(Into::into).collect(),
                coefficients: recoupling_coefficients_dst_src,
            },
            source_axes: None,
        }
    }

    pub fn with_source_axes<I>(mut self, axes: I) -> Self
    where
        I: IntoIterator<Item = usize>,
    {
        self.source_axes = Some(axes.into_iter().collect());
        self
    }

    /// Reuse one immutable source-axis map across plan entries.
    #[doc(hidden)]
    pub fn with_shared_source_axes(mut self, axes: Arc<[usize]>) -> Self {
        self.source_axes = Some(axes);
        self
    }

    pub fn from_block_groups(
        dst_structure: &BlockStructure,
        dst_group: &FusionTreeBlockGroup,
        src_structure: &BlockStructure,
        src_group: &FusionTreeBlockGroup,
        recoupling_coefficients_dst_src: Vec<T>,
    ) -> Result<Self, OperationError> {
        let dst_keys = fusion_tree_group_block_keys(dst_structure, dst_group, "dst")?;
        let src_keys = fusion_tree_group_block_keys(src_structure, src_group, "src")?;
        let expected = dst_keys
            .len()
            .checked_mul(src_keys.len())
            .ok_or(OperationError::ElementCountOverflow)?;
        if recoupling_coefficients_dst_src.len() != expected {
            return Err(OperationError::CoefficientCountMismatch {
                expected,
                actual: recoupling_coefficients_dst_src.len(),
            });
        }
        Ok(Self::multi(
            src_group.group_key().clone(),
            dst_keys,
            src_keys,
            recoupling_coefficients_dst_src,
        ))
    }

    #[inline]
    pub fn group_key(&self) -> &FusionTreeGroupKey {
        &self.group_key
    }

    #[inline]
    pub fn dst_keys(&self) -> &[BlockKey] {
        self.entries.dst()
    }

    #[inline]
    pub fn src_keys(&self) -> &[BlockKey] {
        self.entries.src()
    }

    /// Recoupling matrix coefficients stored as `U[dst, src]` in row-major
    /// destination-by-source order: `coeff[src + dst * src_count]`.
    #[inline]
    pub fn recoupling_coefficients_dst_src(&self) -> &[T] {
        self.entries.coefficients()
    }

    #[inline]
    pub fn source_axes(&self) -> Option<&[usize]> {
        self.source_axes.as_deref()
    }
}

impl<T: Copy> TreeTransformGroupBlockSpec<T> {
    pub(crate) fn to_indexed_spec(
        &self,
        dst_structure: &BlockStructure,
        src_structure: &BlockStructure,
    ) -> Result<TreeTransformBlockSpec<T>, OperationError> {
        let spec = match &self.entries {
            SpecEntries::Single {
                dst,
                src,
                coefficient,
            } => {
                let dst_block = dst_structure.find_block_index_by_key(dst).ok_or_else(|| {
                    OperationError::MissingBlockKey {
                        key: Box::new(dst.clone()),
                    }
                })?;
                let src_block = src_structure.find_block_index_by_key(src).ok_or_else(|| {
                    OperationError::MissingBlockKey {
                        key: Box::new(src.clone()),
                    }
                })?;
                TreeTransformBlockSpec::single(dst_block, src_block, *coefficient)
            }
            SpecEntries::Multi {
                dst,
                src,
                coefficients,
            } => TreeTransformBlockSpec::multi(
                block_indices_for_keys(dst_structure, dst)?,
                block_indices_for_keys(src_structure, src)?,
                coefficients.clone(),
            ),
        };
        Ok(spec.with_optional_source_axes(self.source_axes.clone()))
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct TreeTransformGroupPlan<T> {
    specs: Vec<TreeTransformGroupBlockSpec<T>>,
}

impl<T> TreeTransformGroupPlan<T> {
    pub fn new(specs: Vec<TreeTransformGroupBlockSpec<T>>) -> Self {
        Self { specs }
    }

    pub fn from_specs<I>(specs: I) -> Self
    where
        I: IntoIterator<Item = TreeTransformGroupBlockSpec<T>>,
    {
        Self::new(specs.into_iter().collect())
    }

    #[inline]
    pub fn specs(&self) -> &[TreeTransformGroupBlockSpec<T>] {
        &self.specs
    }

    pub fn into_specs(self) -> Vec<TreeTransformGroupBlockSpec<T>> {
        self.specs
    }
}

impl<T: Copy> TreeTransformGroupPlan<T> {
    pub fn compile<
        TDst,
        TSrc,
        const DST_NOUT: usize,
        const DST_NIN: usize,
        const SRC_NOUT: usize,
        const SRC_NIN: usize,
        SDst,
        SSrc,
        DDst,
        DSrc,
    >(
        &self,
        dst: &TensorMap<TDst, DST_NOUT, DST_NIN, SDst, DDst>,
        src: &TensorMap<TSrc, SRC_NOUT, SRC_NIN, SSrc, DSrc>,
    ) -> Result<TreeTransformStructure<T>, OperationError>
    where
        DDst: TensorStorage<TDst>,
        DSrc: TensorStorage<TSrc>,
    {
        TreeTransformStructure::compile_grouped(dst, src, &self.specs)
    }

    pub fn compile_structures(
        &self,
        dst_structure: &BlockStructure,
        src_structure: &BlockStructure,
    ) -> Result<TreeTransformStructure<T>, OperationError> {
        self.compile_structures_with_storage_conjugation(dst_structure, src_structure, false)
    }

    pub fn compile_structures_with_storage_conjugation(
        &self,
        dst_structure: &BlockStructure,
        src_structure: &BlockStructure,
        storage_conjugate: bool,
    ) -> Result<TreeTransformStructure<T>, OperationError> {
        TreeTransformStructure::compile_grouped_structures_with_storage_conjugation(
            dst_structure,
            src_structure,
            &self.specs,
            storage_conjugate,
        )
    }

    pub fn compile_shared_structures_with_storage_conjugation(
        &self,
        dst_structure: Arc<BlockStructure>,
        src_structure: Arc<BlockStructure>,
        storage_conjugate: bool,
    ) -> Result<TreeTransformStructure<T>, OperationError> {
        TreeTransformStructure::compile_grouped_shared_structures(
            dst_structure,
            src_structure,
            &self.specs,
            storage_conjugate,
        )
    }

    pub fn compile_shared_structures_with_storage_mapping<FBlock, FAxis>(
        &self,
        dst_structure: Arc<BlockStructure>,
        logical_src_structure: &BlockStructure,
        storage_src_structure: Arc<BlockStructure>,
        logical_to_storage_block: FBlock,
        logical_to_storage_axis: FAxis,
        storage_conjugate: bool,
    ) -> Result<TreeTransformStructure<T>, OperationError>
    where
        FBlock: Fn(usize) -> Result<usize, OperationError>,
        FAxis: Fn(usize) -> Result<usize, OperationError>,
    {
        let mut specs = Vec::with_capacity(self.specs.len());
        for spec in &self.specs {
            let indexed = spec.to_indexed_spec(&dst_structure, logical_src_structure)?;
            let entries = match indexed.entries {
                SpecEntries::Single {
                    dst,
                    src,
                    coefficient,
                } => {
                    TreeTransformBlockSpec::single(dst, logical_to_storage_block(src)?, coefficient)
                }
                SpecEntries::Multi {
                    dst,
                    src,
                    coefficients,
                } => TreeTransformBlockSpec::multi(
                    dst,
                    src.into_iter()
                        .map(&logical_to_storage_block)
                        .collect::<Result<Vec<_>, _>>()?,
                    coefficients,
                ),
            };
            let logical_axes = indexed
                .source_axes
                .as_deref()
                .map(ToOwned::to_owned)
                .unwrap_or_else(|| (0..logical_src_structure.rank()).collect());
            let storage_axes = logical_axes
                .into_iter()
                .map(&logical_to_storage_axis)
                .collect::<Result<Vec<_>, _>>()?;
            specs.push(entries.with_source_axes(storage_axes));
        }
        TreeTransformStructure::compile_indexed_shared_structures_with_storage_conjugation(
            dst_structure,
            storage_src_structure,
            &specs,
            storage_conjugate,
        )
    }
}
