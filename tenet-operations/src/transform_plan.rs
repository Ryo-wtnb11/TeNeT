//! Tree-transform plan data: block/group specs (destination trees, source
//! trees, recoupling matrices) and the grouped plan container. Pure data —
//! the symmetric compile layer in `tenet-tensors` builds these from fusion
//! rules; replay consumes them without any symmetry knowledge.

use std::borrow::Cow;
use std::sync::Arc;

use tenet_core::{
    BlockKey, BlockStructure, FusionTreeBlockGroup, FusionTreeGroupKey, FusionTreePairKey,
    TensorMap, TensorStorage,
};

use crate::transform_helpers::{
    duplicate_fusion_tree_pair_indices, fusion_tree_group_block_keys,
    fusion_tree_pair_matches_group, fusion_tree_pairs_share_group,
};
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

#[derive(Debug)]
enum ResolvedSpecEntries<'a, T> {
    Single {
        dst: usize,
        src: usize,
        coefficient: &'a T,
    },
    Multi {
        dst: Vec<usize>,
        src: Vec<usize>,
        coefficients: &'a [T],
    },
}

impl<T> ResolvedSpecEntries<'_, T> {
    #[inline]
    fn dst(&self) -> &[usize] {
        match self {
            Self::Single { dst, .. } => std::slice::from_ref(dst),
            Self::Multi { dst, .. } => dst,
        }
    }

    #[inline]
    fn src(&self) -> &[usize] {
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

    fn map_source_blocks<F>(&mut self, logical_to_storage_block: &F) -> Result<(), OperationError>
    where
        F: Fn(usize) -> Result<usize, OperationError>,
    {
        match self {
            Self::Single { src, .. } => *src = logical_to_storage_block(*src)?,
            Self::Multi { src, .. } => {
                for block in src {
                    *block = logical_to_storage_block(*block)?;
                }
            }
        }
        Ok(())
    }
}

#[derive(Debug)]
pub(crate) struct ResolvedTreeTransformBlockSpec<'a, T> {
    entries: ResolvedSpecEntries<'a, T>,
    source_axes: Option<Cow<'a, [usize]>>,
}

impl<'a, T> ResolvedTreeTransformBlockSpec<'a, T> {
    fn from_entries<K, FindIndex, EraseKey>(
        entries: &'a SpecEntries<K, T>,
        source_axes: Option<&'a [usize]>,
        dst_structure: &BlockStructure,
        src_structure: &BlockStructure,
        find_index: FindIndex,
        erase_key: EraseKey,
    ) -> Result<Self, OperationError>
    where
        FindIndex: Fn(&BlockStructure, &K) -> Option<usize>,
        EraseKey: Fn(&K) -> BlockKey,
    {
        let resolve = |structure: &BlockStructure, key: &K| {
            find_index(structure, key).ok_or_else(|| OperationError::MissingBlockKey {
                key: Box::new(erase_key(key)),
            })
        };
        let entries = match entries {
            SpecEntries::Single {
                dst,
                src,
                coefficient,
            } => ResolvedSpecEntries::Single {
                dst: resolve(dst_structure, dst)?,
                src: resolve(src_structure, src)?,
                coefficient,
            },
            SpecEntries::Multi {
                dst,
                src,
                coefficients,
            } => ResolvedSpecEntries::Multi {
                dst: dst
                    .iter()
                    .map(|key| resolve(dst_structure, key))
                    .collect::<Result<_, _>>()?,
                src: src
                    .iter()
                    .map(|key| resolve(src_structure, key))
                    .collect::<Result<_, _>>()?,
                coefficients,
            },
        };
        Ok(Self {
            entries,
            source_axes: source_axes.map(Cow::Borrowed),
        })
    }

    pub(crate) fn dst_blocks(&self) -> &[usize] {
        self.entries.dst()
    }

    pub(crate) fn src_blocks(&self) -> &[usize] {
        self.entries.src()
    }

    pub(crate) fn coefficients(&self) -> &[T] {
        self.entries.coefficients()
    }

    pub(crate) fn source_axes(&self) -> Option<&[usize]> {
        self.source_axes.as_deref()
    }

    fn map_storage<FBlock, FAxis>(
        mut self,
        logical_rank: usize,
        logical_to_storage_block: &FBlock,
        logical_to_storage_axis: &FAxis,
    ) -> Result<Self, OperationError>
    where
        FBlock: Fn(usize) -> Result<usize, OperationError>,
        FAxis: Fn(usize) -> Result<usize, OperationError>,
    {
        self.entries.map_source_blocks(logical_to_storage_block)?;
        let storage_axes = match self.source_axes.as_deref() {
            Some(logical_axes) => logical_axes
                .iter()
                .copied()
                .map(logical_to_storage_axis)
                .collect::<Result<Vec<_>, _>>()?,
            None => (0..logical_rank)
                .map(logical_to_storage_axis)
                .collect::<Result<Vec<_>, _>>()?,
        };
        self.source_axes = Some(Cow::Owned(storage_axes));
        Ok(self)
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

/// Explicit replay descriptor keyed by application block labels.
///
/// This is the namespace-neutral entry point for Dense, Opaque, and
/// FusionTree block labels. Categorical grouped recoupling uses
/// [`TreeTransformGroupBlockSpec`] instead.
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

impl<T> TreeTransformKeyBlockSpec<T> {
    pub(crate) fn resolve(
        &self,
        dst_structure: &BlockStructure,
        src_structure: &BlockStructure,
    ) -> Result<ResolvedTreeTransformBlockSpec<'_, T>, OperationError> {
        ResolvedTreeTransformBlockSpec::from_entries(
            &self.entries,
            self.source_axes(),
            dst_structure,
            src_structure,
            BlockStructure::find_block_index_by_key,
            Clone::clone,
        )
    }
}

/// Categorical grouped-recoupling descriptor.
///
/// Source and destination identities are fusion-tree pairs. Use
/// [`TreeTransformKeyBlockSpec`] when replay must address arbitrary Dense or
/// Opaque application labels.
#[derive(Clone, Debug, PartialEq)]
pub struct TreeTransformGroupBlockSpec<T> {
    group_key: FusionTreeGroupKey,
    entries: SpecEntries<FusionTreePairKey, T>,
    source_axes: Option<Arc<[usize]>>,
}

impl<T> TreeTransformGroupBlockSpec<T> {
    /// Creates one categorical row/column transform.
    ///
    /// The stored group identity is the source key's fusion-tree group.
    pub fn single(dst_key: FusionTreePairKey, src_key: FusionTreePairKey, coefficient: T) -> Self {
        let group_key = src_key.group_key();
        Self {
            group_key,
            entries: SpecEntries::Single {
                dst: dst_key,
                src: src_key,
                coefficient,
            },
            source_axes: None,
        }
    }

    /// Creates a categorical transform with coefficients ordered as `U[dst, src]`.
    ///
    /// Each side must be nonempty, internally group-coherent, and free of
    /// duplicate fusion-tree identities. The source and destination groups may
    /// differ. Violations return [`OperationError::EmptyTransformBlock`],
    /// [`OperationError::FusionTreeGroupMismatch`],
    /// [`OperationError::DuplicateTreeTransformKey`], or
    /// [`OperationError::CoefficientCountMismatch`]; an unrepresentable
    /// row/column product returns [`OperationError::ElementCountOverflow`].
    pub fn try_multi<DstKeys, SrcKeys>(
        dst_keys: DstKeys,
        src_keys: SrcKeys,
        recoupling_coefficients_dst_src: Vec<T>,
    ) -> Result<Self, OperationError>
    where
        DstKeys: IntoIterator<Item = FusionTreePairKey>,
        SrcKeys: IntoIterator<Item = FusionTreePairKey>,
    {
        let dst_keys = dst_keys.into_iter().collect::<Vec<_>>();
        let src_keys = src_keys.into_iter().collect::<Vec<_>>();
        let Some(first_src) = src_keys.first() else {
            return Err(OperationError::EmptyTransformBlock);
        };
        let Some(first_dst) = dst_keys.first() else {
            return Err(OperationError::EmptyTransformBlock);
        };
        let group_key = first_src.group_key();
        if let Some(index) = src_keys
            .iter()
            .position(|key| !fusion_tree_pair_matches_group(key, &group_key))
        {
            return Err(OperationError::FusionTreeGroupMismatch {
                tensor: "src",
                index,
            });
        }
        if let Some(index) = dst_keys
            .iter()
            .position(|key| !fusion_tree_pairs_share_group(key, first_dst))
        {
            return Err(OperationError::FusionTreeGroupMismatch {
                tensor: "dst",
                index,
            });
        }
        let (duplicate_src, duplicate_dst) =
            duplicate_fusion_tree_pair_indices(&src_keys, &dst_keys);
        if let Some(index) = duplicate_src {
            return Err(OperationError::DuplicateTreeTransformKey {
                tensor: "src",
                index,
            });
        }
        if let Some(index) = duplicate_dst {
            return Err(OperationError::DuplicateTreeTransformKey {
                tensor: "dst",
                index,
            });
        }
        Self::multi_from_validated(
            group_key,
            dst_keys,
            src_keys,
            recoupling_coefficients_dst_src,
        )
    }

    fn multi_from_validated(
        group_key: FusionTreeGroupKey,
        dst_keys: Vec<FusionTreePairKey>,
        src_keys: Vec<FusionTreePairKey>,
        recoupling_coefficients_dst_src: Vec<T>,
    ) -> Result<Self, OperationError> {
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
        Ok(Self {
            group_key,
            entries: SpecEntries::Multi {
                dst: dst_keys,
                src: src_keys,
                coefficients: recoupling_coefficients_dst_src,
            },
            source_axes: None,
        })
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
        let Some(first_src) = src_keys.first() else {
            return Err(OperationError::EmptyTransformBlock);
        };
        if dst_keys.is_empty() {
            return Err(OperationError::EmptyTransformBlock);
        }
        let (duplicate_src, duplicate_dst) =
            duplicate_fusion_tree_pair_indices(&src_keys, &dst_keys);
        if let Some(index) = duplicate_src {
            return Err(OperationError::DuplicateTreeTransformKey {
                tensor: "src",
                index,
            });
        }
        if let Some(index) = duplicate_dst {
            return Err(OperationError::DuplicateTreeTransformKey {
                tensor: "dst",
                index,
            });
        }
        Self::multi_from_validated(
            first_src.group_key(),
            dst_keys,
            src_keys,
            recoupling_coefficients_dst_src,
        )
    }

    #[inline]
    pub fn group_key(&self) -> &FusionTreeGroupKey {
        &self.group_key
    }

    #[inline]
    pub fn dst_keys(&self) -> &[FusionTreePairKey] {
        self.entries.dst()
    }

    #[inline]
    pub fn src_keys(&self) -> &[FusionTreePairKey] {
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

impl<T> TreeTransformGroupBlockSpec<T> {
    pub(crate) fn resolve(
        &self,
        dst_structure: &BlockStructure,
        src_structure: &BlockStructure,
    ) -> Result<ResolvedTreeTransformBlockSpec<'_, T>, OperationError> {
        ResolvedTreeTransformBlockSpec::from_entries(
            &self.entries,
            self.source_axes(),
            dst_structure,
            src_structure,
            BlockStructure::find_block_index_by_fusion_tree_pair,
            |key| BlockKey::FusionTree(key.clone()),
        )
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
        // Why not resolve every key first: fallible block/axis mapping completes
        // per spec before the next key lookup, preserving callback error order.
        for spec in &self.specs {
            specs.push(
                spec.resolve(&dst_structure, logical_src_structure)?
                    .map_storage(
                        logical_src_structure.rank(),
                        &logical_to_storage_block,
                        &logical_to_storage_axis,
                    )?,
            );
        }
        TreeTransformStructure::compile_resolved_shared_structures(
            dst_structure,
            storage_src_structure,
            &specs,
            storage_conjugate,
        )
    }
}
