#[derive(Clone, Debug)]
pub struct TensorMap<T, const NOUT: usize, const NIN: usize, S = Trivial, D = Vec<T>> {
    storage: D,
    space: TensorMapSpace<NOUT, NIN>,
    structure: Arc<BlockStructure>,
    fusion_space: Option<Arc<FusionTensorMapSpace<NOUT, NIN>>>,
    _marker: PhantomData<(T, S)>,
}

pub type Tensor<T, const N: usize, S = Trivial> = TensorMap<T, N, 0, S>;

impl<T, const NOUT: usize, const NIN: usize, S> TensorMap<T, NOUT, NIN, S, Vec<T>> {
    /// Builds a dense tensor backed by `Vec<T>` and the default trivial block
    /// structure.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenet_core::{TensorMap, TensorMapSpace};
    ///
    /// let space = TensorMapSpace::<1, 1>::from_dims([2], [2]).unwrap();
    /// let tensor = TensorMap::<f64, 1, 1>::from_vec(
    ///     vec![1.0, 3.0, 2.0, 4.0],
    ///     space,
    /// )
    /// .unwrap();
    /// assert_eq!(tensor.data(), &[1.0, 3.0, 2.0, 4.0]);
    /// ```
    pub fn from_vec(data: Vec<T>, space: TensorMapSpace<NOUT, NIN>) -> Result<Self, CoreError> {
        Self::from_vec_with_structure(data, space.clone(), BlockStructure::trivial(space.dims())?)
    }

    /// Builds a dense tensor backed by `Vec<T>` and selects an explicit custom
    /// block layout.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenet_core::{BlockStructure, TensorMap, TensorMapSpace, Trivial};
    ///
    /// let space = TensorMapSpace::<1, 0>::from_dims([2], []).unwrap();
    /// let structure = BlockStructure::packed_column_major(1, [vec![2]]).unwrap();
    /// let tensor = TensorMap::<i32, 1, 0>::from_vec_with_structure(
    ///     vec![10, 20],
    ///     space,
    ///     structure,
    /// )
    /// .unwrap();
    /// assert_eq!(tensor.data(), &[10, 20]);
    /// ```
    pub fn from_vec_with_structure(
        data: Vec<T>,
        space: TensorMapSpace<NOUT, NIN>,
        structure: BlockStructure,
    ) -> Result<Self, CoreError> {
        Self::from_vec_with_shared_structure(data, space, structure.into_shared())
    }

    /// Shared-handle variant of [`Self::from_vec_with_structure`]; the caller
    /// still selects the custom block layout.
    pub fn from_vec_with_shared_structure(
        data: Vec<T>,
        space: TensorMapSpace<NOUT, NIN>,
        structure: Arc<BlockStructure>,
    ) -> Result<Self, CoreError> {
        Self::from_storage_with_shared_structure(data, space, structure)
    }

    /// Attaches `Vec<T>` data to an already-selected symmetric tensor layout.
    ///
    /// Unlike [`Self::from_vec_with_structure`], this method does not choose a
    /// block layout; `fusion_space` already owns it.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenet_core::{
    ///     FusionTensorMapSpace, FusionTreeHomSpace, TensorMap, TensorMapSpace,
    ///     Z2FusionRule, Z2Irrep,
    /// };
    ///
    /// let rule = Z2FusionRule;
    /// let fusion_space = FusionTensorMapSpace::from_degeneracy_shapes(
    ///     TensorMapSpace::<1, 1>::from_dims([1], [1]).unwrap(),
    ///     FusionTreeHomSpace::from_sectors([(Z2Irrep::EVEN, 1)], [(Z2Irrep::EVEN, 1)]),
    ///     &rule,
    ///     [vec![1, 1]],
    /// )
    /// .unwrap();
    /// let tensor = TensorMap::<f64, 1, 1>::from_vec_with_fusion_space(
    ///     vec![3.5],
    ///     fusion_space,
    /// )
    /// .unwrap();
    /// assert_eq!(tensor.data(), &[3.5]);
    /// ```
    pub fn from_vec_with_fusion_space(
        data: Vec<T>,
        fusion_space: FusionTensorMapSpace<NOUT, NIN>,
    ) -> Result<Self, CoreError> {
        Self::from_storage_with_fusion_space(data, fusion_space)
    }

    /// Shared-handle variant of [`Self::from_vec_with_fusion_space`]; the
    /// fusion space has already selected the layout.
    pub fn from_vec_with_shared_fusion_space(
        data: Vec<T>,
        fusion_space: Arc<FusionTensorMapSpace<NOUT, NIN>>,
    ) -> Result<Self, CoreError> {
        Self::from_storage_with_shared_fusion_space(data, fusion_space)
    }

    /// Builds a tensor by evaluating `fill(key, indices)` for every block
    /// element; positions not covered by any block keep `background`.
    /// Layout-independent: packed and coupled spaces produce identical
    /// tensors from the same `fill`.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenet_core::{
    ///     FusionTensorMapSpace, FusionTreeHomSpace, TensorMap, TensorMapSpace,
    ///     Z2FusionRule, Z2Irrep,
    /// };
    ///
    /// let rule = Z2FusionRule;
    /// let fusion_space = FusionTensorMapSpace::from_degeneracy_shapes(
    ///     TensorMapSpace::<1, 1>::from_dims([1], [1]).unwrap(),
    ///     FusionTreeHomSpace::from_sectors([(Z2Irrep::EVEN, 1)], [(Z2Irrep::EVEN, 1)]),
    ///     &rule,
    ///     [vec![1, 1]],
    /// )
    /// .unwrap();
    /// let tensor = TensorMap::<i32, 1, 1>::from_block_fn_with_fusion_space(
    ///     fusion_space,
    ///     0,
    ///     |_key, indices| 10 + indices[0] as i32,
    /// )
    /// .unwrap();
    /// assert_eq!(tensor.data(), &[10]);
    /// ```
    pub fn from_block_fn_with_fusion_space<F>(
        fusion_space: FusionTensorMapSpace<NOUT, NIN>,
        background: T,
        fill: F,
    ) -> Result<Self, CoreError>
    where
        T: Clone,
        F: FnMut(&BlockKey, &[usize]) -> T,
    {
        let len = fusion_space.required_len()?;
        let mut tensor = Self::from_vec_with_fusion_space(vec![background; len], fusion_space)?;
        tensor.fill_block_elements(fill)?;
        Ok(tensor)
    }
}

impl<T: Clone, const NOUT: usize, const NIN: usize, S> TensorMap<T, NOUT, NIN, S, Vec<T>> {
    /// Builds a dense tensor filled with a single value.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenet_core::{TensorMap, TensorMapSpace};
    ///
    /// let space = TensorMapSpace::<2, 0>::from_dims([2, 3], []).unwrap();
    /// let tensor = TensorMap::<f64, 2, 0>::filled(1.25, space).unwrap();
    /// assert_eq!(tensor.data(), &[1.25; 6]);
    /// ```
    pub fn filled(value: T, space: TensorMapSpace<NOUT, NIN>) -> Result<Self, CoreError> {
        Self::from_vec(vec![value; space.dense_dim()], space)
    }
}

impl<T, const NOUT: usize, const NIN: usize, S, D> TensorMap<T, NOUT, NIN, S, D>
where
    D: TensorStorage<T>,
{
    /// Builds a tensor from caller-provided storage and selects an explicit
    /// custom block layout.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenet_core::{BlockStructure, TensorMap, TensorMapSpace, Trivial};
    ///
    /// let space = TensorMapSpace::<1, 0>::from_dims([2], []).unwrap();
    /// let structure = BlockStructure::packed_column_major(1, [vec![2]]).unwrap();
    /// let tensor = TensorMap::<i32, 1, 0, Trivial, Vec<i32>>::from_storage_with_structure(
    ///     vec![1, 2],
    ///     space,
    ///     structure,
    /// )
    /// .unwrap();
    /// assert_eq!(tensor.data(), &[1, 2]);
    /// ```
    pub fn from_storage_with_structure(
        storage: D,
        space: TensorMapSpace<NOUT, NIN>,
        structure: BlockStructure,
    ) -> Result<Self, CoreError> {
        Self::from_storage_with_shared_structure(storage, space, structure.into_shared())
    }

    /// Shared-handle variant of [`Self::from_storage_with_structure`]; the
    /// caller still selects the custom block layout.
    pub fn from_storage_with_shared_structure(
        storage: D,
        space: TensorMapSpace<NOUT, NIN>,
        structure: Arc<BlockStructure>,
    ) -> Result<Self, CoreError> {
        Self::from_storage_parts(
            storage,
            space,
            BlockStructure::canonicalize_shared(structure),
            None,
        )
    }

    /// Attaches caller-provided storage to an already-selected symmetric
    /// tensor layout.
    ///
    /// Unlike [`Self::from_storage_with_structure`], this method does not
    /// choose a block layout; `fusion_space` already owns it.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenet_core::{
    ///     FusionTensorMapSpace, FusionTreeHomSpace, TensorMap, TensorMapSpace,
    ///     Trivial, Z2FusionRule, Z2Irrep,
    /// };
    ///
    /// let rule = Z2FusionRule;
    /// let fusion_space = FusionTensorMapSpace::from_degeneracy_shapes(
    ///     TensorMapSpace::<1, 1>::from_dims([1], [1]).unwrap(),
    ///     FusionTreeHomSpace::from_sectors([(Z2Irrep::EVEN, 1)], [(Z2Irrep::EVEN, 1)]),
    ///     &rule,
    ///     [vec![1, 1]],
    /// )
    /// .unwrap();
    /// let tensor = TensorMap::<f64, 1, 1, Trivial, Vec<f64>>::from_storage_with_fusion_space(
    ///     vec![2.0],
    ///     fusion_space,
    /// )
    /// .unwrap();
    /// assert_eq!(tensor.data(), &[2.0]);
    /// ```
    pub fn from_storage_with_fusion_space(
        storage: D,
        fusion_space: FusionTensorMapSpace<NOUT, NIN>,
    ) -> Result<Self, CoreError> {
        Self::from_storage_with_shared_fusion_space(storage, Arc::new(fusion_space))
    }

    /// Shared-handle variant of [`Self::from_storage_with_fusion_space`]; the
    /// fusion space has already selected the layout.
    pub fn from_storage_with_shared_fusion_space(
        storage: D,
        fusion_space: Arc<FusionTensorMapSpace<NOUT, NIN>>,
    ) -> Result<Self, CoreError> {
        Self::from_storage_parts(
            storage,
            fusion_space.dense_space().clone(),
            Arc::clone(fusion_space.subblock_structure()),
            Some(fusion_space),
        )
    }

    fn from_storage_parts(
        storage: D,
        space: TensorMapSpace<NOUT, NIN>,
        structure: Arc<BlockStructure>,
        fusion_space: Option<Arc<FusionTensorMapSpace<NOUT, NIN>>>,
    ) -> Result<Self, CoreError> {
        if structure.rank() != space.dims().len() {
            return Err(CoreError::StructureRankMismatch {
                expected: space.dims().len(),
                actual: structure.rank(),
            });
        }
        let required_len = structure.required_len()?;
        let storage_len = storage.len();
        validate_exact_storage_extent(required_len, storage_len, storage_len)?;
        Ok(Self {
            storage,
            space,
            structure,
            fusion_space,
            _marker: PhantomData,
        })
    }

    /// Borrows the concrete storage immutably.
    ///
    /// The default `Vec<T>` storage cannot be resized through this borrow. Safe
    /// element mutation for host storage remains available through
    /// [`Self::data_mut`]; custom storage with interior mutability is checked
    /// again at execution boundaries.
    ///
    /// ```compile_fail
    /// use tenet_core::{TensorMap, TensorMapSpace};
    ///
    /// let space = TensorMapSpace::<1, 0>::from_dims([2], []).unwrap();
    /// let mut tensor = TensorMap::<i32, 1, 0>::from_vec(vec![1, 2], space).unwrap();
    /// tensor.storage_mut().clear();
    /// ```
    #[inline]
    pub fn storage(&self) -> &D {
        &self.storage
    }

    /// Validates the storage extent immediately before an execution boundary.
    ///
    /// `actual_len` is the extent of the concrete view that execution will
    /// access. Host callers must pass their slice length; callers for opaque
    /// storage pass the corresponding backend-visible extent. The reported
    /// [`TensorStorage::len`] is checked independently so custom storage cannot
    /// hide a changed or inconsistent extent.
    ///
    /// Constructor-only validation or a stable-length marker would not cover
    /// safe interior mutability reachable through [`Self::storage`]. Exact
    /// execution-time validation preserves external storage support without
    /// sealing the storage traits or adding an unsafe capability contract.
    ///
    /// This stays crate-local until an execution crate has a concrete
    /// view-boundary contract that can tie `actual_len` to the view it will
    /// access.
    #[inline]
    pub(crate) fn validate_storage_extent(&self, actual_len: usize) -> Result<(), CoreError> {
        validate_exact_storage_extent(
            self.structure.required_len()?,
            self.storage.len(),
            actual_len,
        )
    }

    #[inline]
    pub fn placement(&self) -> Placement {
        self.storage.placement()
    }

    #[inline]
    pub fn similar_storage_filled(&self, len: usize, value: T) -> D::Similar
    where
        D: SimilarStorage<T>,
        T: Clone,
    {
        self.storage.similar_filled(len, value)
    }

    #[inline]
    pub fn space(&self) -> &TensorMapSpace<NOUT, NIN> {
        &self.space
    }

    #[inline]
    pub fn structure(&self) -> &Arc<BlockStructure> {
        &self.structure
    }

    #[inline]
    pub fn fusion_space(&self) -> Option<&Arc<FusionTensorMapSpace<NOUT, NIN>>> {
        self.fusion_space.as_ref()
    }

    #[inline]
    pub fn dim(&self) -> usize {
        self.storage.len()
    }

    #[inline]
    pub fn storage_dim(&self) -> usize {
        self.storage.len()
    }

    /// Full dense element count obtained by multiplying the uncoupled leg dimensions.
    ///
    /// For block-sparse/symmetric tensors this can be larger than the packed storage
    /// length returned by [`Self::dim`] / [`Self::storage_dim`].
    #[inline]
    pub fn dense_dim(&self) -> usize {
        self.space.dense_dim()
    }

    #[inline]
    pub fn dims(&self) -> &[usize] {
        self.space.dims()
    }
}

impl<T, const NOUT: usize, const NIN: usize, S, D> TensorMap<T, NOUT, NIN, S, D>
where
    D: HostReadableStorage<T>,
{
    #[inline]
    fn validated_host_data(&self) -> Result<&[T], CoreError> {
        let data = self.storage.as_slice();
        self.validate_storage_extent(data.len())?;
        Ok(data)
    }

    #[inline]
    pub fn data(&self) -> &[T] {
        self.storage.as_slice()
    }

    /// Visits every block element as `(key, indices, value)`, independent of
    /// the storage layout.
    pub fn for_each_block_element<F>(&self, mut visit: F) -> Result<(), CoreError>
    where
        F: FnMut(&BlockKey, &[usize], &T),
    {
        let structure = Arc::clone(&self.structure);
        let data = self.validated_host_data()?;
        for index in 0..structure.block_count() {
            let block = structure.block(index)?;
            let shape = block.shape();
            let strides = block.strides();
            let offset = block.offset();
            let count: usize = shape.iter().product();
            let mut indices = vec![0usize; shape.len()];
            for _ in 0..count {
                let position = offset
                    + indices
                        .iter()
                        .zip(strides)
                        .map(|(&index, &stride)| index * stride)
                        .sum::<usize>();
                visit(block.key(), &indices, &data[position]);
                for axis in 0..shape.len() {
                    indices[axis] += 1;
                    if indices[axis] < shape[axis] {
                        break;
                    }
                    indices[axis] = 0;
                }
            }
        }
        Ok(())
    }

    pub fn subblock(&self) -> Result<BlockView<'_, T>, CoreError> {
        let block = self.structure.only_block()?;
        BlockView::new(
            self.validated_host_data()?,
            block.shape(),
            block.strides(),
            block.offset(),
        )
    }

    pub fn block(&self, index: usize) -> Result<BlockView<'_, T>, CoreError> {
        let block = self.structure.block(index)?;
        BlockView::new(
            self.validated_host_data()?,
            block.shape(),
            block.strides(),
            block.offset(),
        )
    }

    pub fn block_by_key(&self, key: &BlockKey) -> Result<BlockView<'_, T>, CoreError> {
        let block = self.structure.block_by_key(key)?;
        BlockView::new(
            self.validated_host_data()?,
            block.shape(),
            block.strides(),
            block.offset(),
        )
    }

    pub fn subblock_by_tree(
        &self,
        key: &FusionTreePairKey,
    ) -> Result<BlockView<'_, T>, CoreError> {
        let block = self.structure.fusion_tree_pair_block(key)?;
        BlockView::new(
            self.validated_host_data()?,
            block.shape(),
            block.strides(),
            block.offset(),
        )
    }

    pub fn subblock_by_sectors<R>(
        &self,
        rule: &R,
        sectors: &[SectorId],
    ) -> Result<BlockView<'_, T>, CoreError>
    where
        R: MultiplicityFreeFusionRule,
    {
        let mut blocks = self.subblocks_by_sectors(rule, sectors)?;
        if blocks.len() != 1 {
            return Err(CoreError::BlockCountMismatch {
                expected: 1,
                actual: blocks.len(),
            });
        }
        Ok(blocks.remove(0))
    }

    pub fn subblocks_by_sectors<R>(
        &self,
        rule: &R,
        sectors: &[SectorId],
    ) -> Result<Vec<BlockView<'_, T>>, CoreError>
    where
        R: MultiplicityFreeFusionRule,
    {
        let fusion_space = self
            .fusion_space
            .as_ref()
            .ok_or(CoreError::MissingFusionSpace)?;
        let keys = fusion_space
            .homspace()
            .fusion_tree_keys_from_external_sectors(rule, sectors)?;
        let data = self.validated_host_data()?;
        let mut blocks = Vec::with_capacity(keys.len());
        for key in keys {
            let block = self.structure.fusion_tree_pair_block(&key)?;
            blocks.push(BlockView::new(
                data,
                block.shape(),
                block.strides(),
                block.offset(),
            )?);
        }
        Ok(blocks)
    }
}

impl<T, const NOUT: usize, const NIN: usize, S, D> TensorMap<T, NOUT, NIN, S, D>
where
    D: HostWritableStorage<T>,
{
    /// Mutates host elements without exposing a length-changing storage API.
    ///
    /// ```
    /// use tenet_core::{TensorMap, TensorMapSpace};
    ///
    /// let space = TensorMapSpace::<1, 0>::from_dims([2], []).unwrap();
    /// let mut tensor = TensorMap::<i32, 1, 0>::from_vec(vec![1, 2], space).unwrap();
    /// let len = tensor.data_mut().len();
    /// tensor.data_mut()[1] = 7;
    /// assert_eq!(tensor.data(), &[1, 7]);
    /// assert_eq!(tensor.data().len(), len);
    /// ```
    #[inline]
    pub fn data_mut(&mut self) -> &mut [T] {
        self.storage.as_mut_slice()
    }

    /// Fills every block element with `fill(key, indices)`, independent of
    /// the storage layout. The layout-safe way to enter data: constructing
    /// through this (instead of positioning raw values in a flat vector)
    /// gives identical tensors for the packed and coupled layouts.
    pub fn fill_block_elements<F>(&mut self, mut fill: F) -> Result<(), CoreError>
    where
        F: FnMut(&BlockKey, &[usize]) -> T,
    {
        let structure = Arc::clone(&self.structure);
        let data = validated_host_data_mut(structure.as_ref(), &mut self.storage)?;
        for index in 0..structure.block_count() {
            let block = structure.block(index)?;
            let shape = block.shape();
            let strides = block.strides();
            let offset = block.offset();
            let count: usize = shape.iter().product();
            let mut indices = vec![0usize; shape.len()];
            for _ in 0..count {
                let position = offset
                    + indices
                        .iter()
                        .zip(strides)
                        .map(|(&index, &stride)| index * stride)
                        .sum::<usize>();
                data[position] = fill(block.key(), &indices);
                for axis in 0..shape.len() {
                    indices[axis] += 1;
                    if indices[axis] < shape[axis] {
                        break;
                    }
                    indices[axis] = 0;
                }
            }
        }
        Ok(())
    }

    pub fn subblock_mut(&mut self) -> Result<BlockViewMut<'_, T>, CoreError> {
        let block = self.structure.only_block()?;
        BlockViewMut::new(
            validated_host_data_mut(self.structure.as_ref(), &mut self.storage)?,
            block.shape(),
            block.strides(),
            block.offset(),
        )
    }

    pub fn block_mut(&mut self, index: usize) -> Result<BlockViewMut<'_, T>, CoreError> {
        let block = self.structure.block(index)?;
        BlockViewMut::new(
            validated_host_data_mut(self.structure.as_ref(), &mut self.storage)?,
            block.shape(),
            block.strides(),
            block.offset(),
        )
    }

    pub fn block_mut_by_key(&mut self, key: &BlockKey) -> Result<BlockViewMut<'_, T>, CoreError> {
        let block = self.structure.block_by_key(key)?;
        BlockViewMut::new(
            validated_host_data_mut(self.structure.as_ref(), &mut self.storage)?,
            block.shape(),
            block.strides(),
            block.offset(),
        )
    }

    pub fn subblock_mut_by_tree(
        &mut self,
        key: &FusionTreePairKey,
    ) -> Result<BlockViewMut<'_, T>, CoreError> {
        let block = self.structure.fusion_tree_pair_block(key)?;
        BlockViewMut::new(
            validated_host_data_mut(self.structure.as_ref(), &mut self.storage)?,
            block.shape(),
            block.strides(),
            block.offset(),
        )
    }

    pub fn subblock_mut_by_sectors<R>(
        &mut self,
        rule: &R,
        sectors: &[SectorId],
    ) -> Result<BlockViewMut<'_, T>, CoreError>
    where
        R: MultiplicityFreeFusionRule,
    {
        let fusion_space = self
            .fusion_space
            .as_ref()
            .ok_or(CoreError::MissingFusionSpace)?;
        let key = fusion_space
            .homspace()
            .unique_fusion_tree_key_from_external_sectors(rule, sectors)?;
        self.subblock_mut_by_tree(&key)
    }
}

#[inline]
fn validated_host_data_mut<'a, T, D>(
    structure: &BlockStructure,
    storage: &'a mut D,
) -> Result<&'a mut [T], CoreError>
where
    D: HostWritableStorage<T>,
{
    let required_len = structure.required_len()?;
    let reported_len = storage.len();
    let data = storage.as_mut_slice();
    validate_exact_storage_extent(required_len, reported_len, data.len())?;
    Ok(data)
}

#[inline]
fn validate_exact_storage_extent(
    required_len: usize,
    reported_len: usize,
    actual_len: usize,
) -> Result<(), CoreError> {
    if reported_len != required_len {
        return Err(CoreError::DimensionMismatch {
            expected: required_len,
            actual: reported_len,
        });
    }
    if actual_len != reported_len {
        return Err(CoreError::DimensionMismatch {
            expected: reported_len,
            actual: actual_len,
        });
    }
    Ok(())
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct BlockLayout<'a> {
    len: usize,
    offset: usize,
    shape: &'a [usize],
    strides: &'a [usize],
}

impl<'a> BlockLayout<'a> {
    pub fn new(
        len: usize,
        offset: usize,
        shape: &'a [usize],
        strides: &'a [usize],
    ) -> Result<Self, CoreError> {
        let layout = Self {
            len,
            offset,
            shape,
            strides,
        };
        validate_layout(layout)?;
        Ok(layout)
    }

    #[inline]
    pub fn len(self) -> usize {
        self.len
    }

    #[inline]
    pub fn offset(self) -> usize {
        self.offset
    }

    #[inline]
    pub fn shape(self) -> &'a [usize] {
        self.shape
    }

    #[inline]
    pub fn strides(self) -> &'a [usize] {
        self.strides
    }

    #[inline]
    pub fn rank(self) -> usize {
        self.shape.len()
    }

    #[inline]
    pub fn is_empty(self) -> bool {
        self.shape.iter().any(|&dim| dim == 0)
    }
}

#[derive(Debug)]
pub struct BlockView<'a, T> {
    data: &'a [T],
    layout: BlockLayout<'a>,
}

impl<'a, T> Clone for BlockView<'a, T> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<'a, T> Copy for BlockView<'a, T> {}

impl<'a, T> BlockView<'a, T> {
    pub fn new(
        data: &'a [T],
        shape: &'a [usize],
        strides: &'a [usize],
        offset: usize,
    ) -> Result<Self, CoreError> {
        let layout = BlockLayout::new(data.len(), offset, shape, strides)?;
        Ok(Self { data, layout })
    }

    #[inline]
    pub fn data(&self) -> &'a [T] {
        self.data
    }

    #[inline]
    pub fn layout(&self) -> BlockLayout<'a> {
        self.layout
    }

    #[inline]
    pub fn shape(&self) -> &'a [usize] {
        self.layout.shape()
    }

    #[inline]
    pub fn strides(&self) -> &'a [usize] {
        self.layout.strides()
    }

    #[inline]
    pub fn offset(&self) -> usize {
        self.layout.offset()
    }
}

#[derive(Debug)]
pub struct BlockViewMut<'a, T> {
    data: &'a mut [T],
    layout: BlockLayout<'a>,
}

impl<'a, T> BlockViewMut<'a, T> {
    pub fn new(
        data: &'a mut [T],
        shape: &'a [usize],
        strides: &'a [usize],
        offset: usize,
    ) -> Result<Self, CoreError> {
        let layout = BlockLayout::new(data.len(), offset, shape, strides)?;
        Ok(Self { data, layout })
    }

    #[inline]
    pub fn data(&self) -> &[T] {
        self.data
    }

    #[inline]
    pub fn data_mut(&mut self) -> &mut [T] {
        self.data
    }

    #[inline]
    pub fn layout(&self) -> BlockLayout<'a> {
        self.layout
    }

    #[inline]
    pub fn shape(&self) -> &'a [usize] {
        self.layout.shape()
    }

    #[inline]
    pub fn strides(&self) -> &'a [usize] {
        self.layout.strides()
    }

    #[inline]
    pub fn offset(&self) -> usize {
        self.layout.offset()
    }

    #[inline]
    pub fn into_parts(self) -> (&'a mut [T], BlockLayout<'a>) {
        (self.data, self.layout)
    }
}
