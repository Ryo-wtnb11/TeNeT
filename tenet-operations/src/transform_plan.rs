//! Tree-transform plan data: block/group specs (destination trees, source
//! trees, recoupling matrices) and the grouped plan container. Pure data —
//! the symmetric compile layer in `tenet-tensors` builds these from fusion
//! rules; replay consumes them without any symmetry knowledge.

use std::borrow::Cow;
use std::fmt;
use std::sync::{Arc, OnceLock};

use tenet_core::{
    BlockKey, BlockStructure, FusionTreeBlockGroup, FusionTreeGroupKey, FusionTreePairKey,
    TensorMap, TensorStorage,
};

use crate::transform_helpers::{
    duplicate_fusion_tree_pair_indices, fusion_tree_group_block_keys,
    fusion_tree_pair_matches_group, fusion_tree_pairs_share_group,
};
use crate::transform_structure::{SharedTreeTransformCoefficients, TreeTransformStructure};
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
        Self::from_entries_with_resolvers(
            entries,
            source_axes,
            |key| resolve(dst_structure, key),
            |key| resolve(src_structure, key),
        )
    }

    fn from_entries_with_resolvers<K, ResolveDst, ResolveSrc>(
        entries: &'a SpecEntries<K, T>,
        source_axes: Option<&'a [usize]>,
        resolve_dst: ResolveDst,
        resolve_src: ResolveSrc,
    ) -> Result<Self, OperationError>
    where
        ResolveDst: Fn(&K) -> Result<usize, OperationError>,
        ResolveSrc: Fn(&K) -> Result<usize, OperationError>,
    {
        let entries = match entries {
            SpecEntries::Single {
                dst,
                src,
                coefficient,
            } => ResolvedSpecEntries::Single {
                dst: resolve_dst(dst)?,
                src: resolve_src(src)?,
                coefficient,
            },
            SpecEntries::Multi {
                dst,
                src,
                coefficients,
            } => ResolvedSpecEntries::Multi {
                dst: dst.iter().map(&resolve_dst).collect::<Result<_, _>>()?,
                src: src.iter().map(resolve_src).collect::<Result<_, _>>()?,
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

    fn has_matrix_payload(&self) -> bool {
        matches!(self.entries, SpecEntries::Multi { .. })
    }

    fn resolve_with_source_projection<F>(
        &self,
        dst_structure: &BlockStructure,
        source_index: &F,
    ) -> Result<ResolvedTreeTransformBlockSpec<'_, T>, OperationError>
    where
        F: Fn(&FusionTreePairKey) -> Result<usize, OperationError>,
    {
        let resolve_dst = |key: &FusionTreePairKey| {
            dst_structure
                .find_block_index_by_fusion_tree_pair(key)
                .ok_or_else(|| OperationError::MissingBlockKey {
                    key: Box::new(BlockKey::FusionTree(key.clone())),
                })
        };
        ResolvedTreeTransformBlockSpec::from_entries_with_resolvers(
            &self.entries,
            self.source_axes(),
            resolve_dst,
            source_index,
        )
    }
}

/// Immutable categorical fusion-tree transform, independent of storage layout.
///
/// Matrix-valued groups materialize one shared contiguous coefficient payload
/// on first binding. All-Single plans keep their compact inline coefficients;
/// sharing those scalars would add allocation without removing a matrix copy.
#[derive(Clone)]
pub struct TreeTransformGroupPlan<T> {
    specs: Vec<TreeTransformGroupBlockSpec<T>>,
    coefficient_payload: OnceLock<Option<SharedTreeTransformCoefficients<T>>>,
}

impl<T: fmt::Debug> fmt::Debug for TreeTransformGroupPlan<T> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("TreeTransformGroupPlan")
            .field("specs", &self.specs)
            .finish()
    }
}

impl<T> TreeTransformGroupPlan<T> {
    pub fn new(specs: Vec<TreeTransformGroupBlockSpec<T>>) -> Self {
        Self {
            specs,
            coefficient_payload: OnceLock::new(),
        }
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

impl<T: PartialEq> PartialEq for TreeTransformGroupPlan<T> {
    fn eq(&self, other: &Self) -> bool {
        self.specs == other.specs
    }
}

impl<T: Copy> TreeTransformGroupPlan<T> {
    fn coefficient_payload(
        &self,
    ) -> Result<Option<SharedTreeTransformCoefficients<T>>, OperationError> {
        if let Some(payload) = self.coefficient_payload.get() {
            return Ok(payload.clone());
        }
        let coefficient_count = self.specs.iter().try_fold(0usize, |count, spec| {
            count
                .checked_add(spec.recoupling_coefficients_dst_src().len())
                .ok_or(OperationError::ElementCountOverflow)
        })?;
        let has_matrix_payload = self
            .specs
            .iter()
            .any(TreeTransformGroupBlockSpec::has_matrix_payload);
        Ok(self
            .coefficient_payload
            .get_or_init(|| {
                if !has_matrix_payload {
                    return None;
                }
                let mut coefficients = Vec::with_capacity(coefficient_count);
                for spec in &self.specs {
                    coefficients.extend_from_slice(spec.recoupling_coefficients_dst_src());
                }
                debug_assert_eq!(coefficients.len(), coefficient_count);
                Some(SharedTreeTransformCoefficients::from_vec(coefficients))
            })
            .clone())
    }

    fn compile_shared_structures_internal(
        &self,
        dst_structure: Arc<BlockStructure>,
        src_structure: Arc<BlockStructure>,
        storage_conjugate: bool,
    ) -> Result<TreeTransformStructure<T>, OperationError> {
        let mut specs = Vec::with_capacity(self.specs.len());
        for spec in &self.specs {
            specs.push(spec.resolve(&dst_structure, &src_structure)?);
        }
        TreeTransformStructure::compile_resolved_shared_structures(
            dst_structure,
            src_structure,
            &specs,
            storage_conjugate,
            self.coefficient_payload()?,
        )
    }

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
        self.compile_shared_structures_internal(
            Arc::clone(dst.structure()),
            Arc::clone(src.structure()),
            false,
        )
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
        self.compile_shared_structures_internal(
            Arc::new(dst_structure.clone()),
            Arc::new(src_structure.clone()),
            storage_conjugate,
        )
    }

    pub fn compile_shared_structures_with_storage_conjugation(
        &self,
        dst_structure: Arc<BlockStructure>,
        src_structure: Arc<BlockStructure>,
        storage_conjugate: bool,
    ) -> Result<TreeTransformStructure<T>, OperationError> {
        self.compile_shared_structures_internal(dst_structure, src_structure, storage_conjugate)
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
            self.coefficient_payload()?,
        )
    }

    #[doc(hidden)]
    pub fn compile_shared_structures_with_source_projection<FSource, FAxis>(
        &self,
        dst_structure: Arc<BlockStructure>,
        storage_src_structure: Arc<BlockStructure>,
        logical_rank: usize,
        source_index: FSource,
        logical_to_storage_axis: FAxis,
        storage_conjugate: bool,
    ) -> Result<TreeTransformStructure<T>, OperationError>
    where
        FSource: Fn(&FusionTreePairKey) -> Result<usize, OperationError>,
        FAxis: Fn(usize) -> Result<usize, OperationError>,
    {
        let identity_block = |index| Ok(index);
        let mut specs = Vec::with_capacity(self.specs.len());
        for spec in &self.specs {
            specs.push(
                spec.resolve_with_source_projection(&dst_structure, &source_index)?
                    .map_storage(logical_rank, &identity_block, &logical_to_storage_axis)?,
            );
        }
        TreeTransformStructure::compile_resolved_shared_structures(
            dst_structure,
            storage_src_structure,
            &specs,
            storage_conjugate,
            self.coefficient_payload()?,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tenet_core::BlockSpec;

    fn tree_pair(vertex: usize) -> FusionTreePairKey {
        FusionTreePairKey::try_pair_from_sector_ids(
            [0, 0],
            [],
            0,
            [false, false],
            [],
            [],
            [],
            [vertex],
            [],
        )
        .unwrap()
    }

    fn structure_with_rank(
        keys: &[FusionTreePairKey],
        elements: usize,
        rank: usize,
    ) -> BlockStructure {
        BlockStructure::from_blocks_with_rank(
            rank,
            keys.iter()
                .enumerate()
                .map(|(index, key)| {
                    BlockSpec::column_major_with_key(
                        BlockKey::from(key.clone()),
                        std::iter::once(elements)
                            .chain(std::iter::repeat(1).take(rank - 1))
                            .collect(),
                        index * elements,
                    )
                    .unwrap()
                })
                .collect(),
        )
        .unwrap()
    }

    fn structure(keys: &[FusionTreePairKey], elements: usize) -> BlockStructure {
        structure_with_rank(keys, elements, 2)
    }

    fn matrix_plan(keys: &[FusionTreePairKey; 2]) -> TreeTransformGroupPlan<f64> {
        TreeTransformGroupPlan::new(vec![TreeTransformGroupBlockSpec::try_multi(
            keys.clone(),
            keys.clone(),
            vec![1.0_f64, 2.0, 3.0, 4.0],
        )
        .unwrap()])
    }

    #[test]
    fn categorical_coefficients_are_shared_across_layout_bindings() {
        // What: one categorical plan binds to different degeneracy layouts
        // without copying its coefficient payload into either replay plan.
        let keys = [tree_pair(1), tree_pair(2)];
        let plan = matrix_plan(&keys);
        let first_layout = structure(&keys, 2);
        let second_layout = structure(&keys, 3);

        let first = plan
            .compile_structures(&first_layout, &first_layout)
            .unwrap();
        let second = plan
            .compile_structures(&second_layout, &second_layout)
            .unwrap();

        assert!(first.shares_coefficient_payload_with(&second));
        assert_eq!(
            first.recoupling_coefficients_dst_src(),
            &[1.0, 2.0, 3.0, 4.0]
        );
        assert_eq!(
            second.recoupling_coefficients_dst_src(),
            &[1.0, 2.0, 3.0, 4.0]
        );
    }

    #[test]
    fn concurrent_cold_bindings_share_the_installed_payload() {
        // What: racing cold layout bindings retain the one payload installed
        // by OnceLock instead of each retaining its own flattened matrix.
        let keys = [tree_pair(1), tree_pair(2)];
        let plan = matrix_plan(&keys);
        let first_layout = structure(&keys, 2);
        let second_layout = structure(&keys, 3);
        let barrier = Arc::new(std::sync::Barrier::new(2));

        let (first, second) = std::thread::scope(|scope| {
            let first_barrier = Arc::clone(&barrier);
            let first_plan = &plan;
            let first_layout = &first_layout;
            let first = scope.spawn(move || {
                first_barrier.wait();
                first_plan
                    .compile_structures(first_layout, first_layout)
                    .unwrap()
            });
            let second_barrier = Arc::clone(&barrier);
            let second_plan = &plan;
            let second_layout = &second_layout;
            let second = scope.spawn(move || {
                second_barrier.wait();
                second_plan
                    .compile_structures(second_layout, second_layout)
                    .unwrap()
            });
            (first.join().unwrap(), second.join().unwrap())
        });

        assert!(first.shares_coefficient_payload_with(&second));
    }

    #[test]
    fn debug_output_ignores_lazy_payload_state_and_storage_kind() {
        // What: Debug remains the categorical specs plus the historical
        // coefficient slice, independent of lazy initialization and binding.
        let keys = [tree_pair(1), tree_pair(2)];
        let plan = matrix_plan(&keys);
        let layout = structure(&keys, 2);
        let wrong_rank = structure_with_rank(&keys, 2, 3);
        let before = format!("{plan:?}");

        let error = plan.compile_structures(&layout, &wrong_rank).unwrap_err();
        assert!(matches!(
            error,
            OperationError::StructureRankMismatch {
                expected: 2,
                actual: 3
            }
        ));
        assert_eq!(format!("{plan:?}"), before);

        let shared = plan.compile_structures(&layout, &layout).unwrap();
        assert_eq!(format!("{plan:?}"), before);
        let owned = TreeTransformStructure::compile_structures(
            &layout,
            &layout,
            &[TreeTransformBlockSpec::multi(
                vec![0, 1],
                vec![0, 1],
                vec![1.0, 2.0, 3.0, 4.0],
            )],
        )
        .unwrap();
        let shared_debug = format!("{shared:?}");
        assert_eq!(shared_debug, format!("{owned:?}"));
        assert!(!shared_debug.contains("Owned"));
        assert!(!shared_debug.contains("Shared"));
    }
}
