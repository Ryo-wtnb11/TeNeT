use super::*;

#[cfg(feature = "cuda")]
impl<R, D: CudaPayload> TensorMap<R, D> {
    /// Uploads host ownership of a device-capable payload to this tensor's
    /// Runtime CUDA context.
    ///
    /// Dense storage uploads directly. Compact diagonal storage is expanded
    /// operation-locally and becomes dense on device; a roundtrip remains
    /// dense rather than recovering compactness. A lazy adjoint transfers only its canonical
    /// parent and rebuilds a lazy view over the device parent.
    ///
    /// Every payload of the base family uploads: `f64`, `Complex64`, `f32`
    /// and `Complex32`. Single precision moves exactly half the bytes of its
    /// double-precision twin for the same fixture, in the same number of
    /// transfer and allocation calls. What single precision does *not* open is
    /// the device factorization family — see [`CudaFactorizationPayload`].
    ///
    /// ```
    /// use num_complex::Complex32;
    /// use tenet::sector::U1FusionRule;
    /// use tenet::typed::TensorMap;
    ///
    /// fn c32_upload(tensor: &TensorMap<U1FusionRule, Complex32>) {
    ///     let _ = tensor.to_cuda();
    /// }
    /// ```
    ///
    /// ```
    /// use tenet::sector::U1FusionRule;
    /// use tenet::typed::TensorMap;
    ///
    /// fn f32_upload(tensor: &TensorMap<U1FusionRule, f32>) {
    ///     let _ = tensor.to_cuda();
    /// }
    /// ```
    pub fn to_cuda(&self) -> Result<TensorMap<R, D, CudaStorage<D>>, Error> {
        let mut lease = self.runtime.lease_cuda()?;
        let cuda = &mut *lease;
        let upload = |body: &Arc<TypedTensorBody<R, D>>| {
            let storage = match body.data.as_ref() {
                TypedData::Dense(data) => CudaStorage::upload(cuda, data)?,
                TypedData::Diagonal(spectrum) => {
                    let dense = tenet_matrixalgebra::seam::diagonal_bond_data(
                        body.space.space(),
                        spectrum,
                        &|value| value,
                    )?;
                    CudaStorage::upload_owned(cuda, dense)?
                }
            };
            Ok::<_, Error>(Arc::new(TypedTensorBody::dense(
                body.space.clone(),
                storage,
            )))
        };

        let repr = match &self.repr {
            TypedTensorRepr::Owned(body) => TypedTensorRepr::Owned(upload(body)?),
            TypedTensorRepr::Adjoint(view) => TypedTensorRepr::Adjoint(Arc::new(
                TypedAdjointView::new(upload(&view.parent)?, view.logical_space.clone()),
            )),
        };
        Ok(TensorMap {
            runtime: self.runtime.clone(),
            repr,
        })
    }
}

#[cfg(feature = "cuda")]
impl<R, D: CudaPayload> TensorMap<R, D, CudaStorage<D>> {
    /// Downloads device ownership into one final dense host buffer.
    ///
    /// A lazy adjoint downloads only its canonical parent and rebuilds a cold
    /// host lazy view. No receiver-sized logical adjoint is materialized.
    /// Device storage is never implicitly host-readable:
    ///
    /// ```compile_fail
    /// use tenet::sector::U1FusionRule;
    /// use tenet::typed::{CudaStorage, TensorMap};
    ///
    /// fn no_device_slice(tensor: &TensorMap<U1FusionRule, f64, CudaStorage>) {
    ///     let _ = tensor.dense_data();
    /// }
    /// ```
    pub fn to_host(&self) -> Result<TensorMap<R, D>, Error> {
        // ponytail: this message predates `lease_cuda`; kept byte-identical.
        let mut lease = self.runtime.lease_cuda().map_err(|_| {
            Error::InvalidArgument("this runtime was built without a CUDA device".to_string())
        })?;
        let cuda = &mut *lease;
        let download = |body: &Arc<TypedTensorBody<R, D, CudaStorage<D>>>| {
            let TypedData::Dense(storage) = body.data.as_ref() else {
                unreachable!("typed CUDA transfer never produces compact storage")
            };
            let data = storage.download(cuda)?;
            Ok::<_, Error>(Arc::new(TypedTensorBody::dense(body.space.clone(), data)))
        };

        let repr = match &self.repr {
            TypedTensorRepr::Owned(body) => TypedTensorRepr::Owned(download(body)?),
            TypedTensorRepr::Adjoint(view) => TypedTensorRepr::Adjoint(Arc::new(
                TypedAdjointView::new(download(&view.parent)?, view.logical_space.clone()),
            )),
        };
        Ok(TensorMap {
            runtime: self.runtime.clone(),
            repr,
        })
    }
}

#[cfg(feature = "cuda")]
impl<R, D> TensorMap<R, D, CudaStorage<D>>
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64> + CheckedFusionAlgebra + SectorCodec,
    D: CudaPayload,
{
    /// Lazy categorical adjoint over the same parent device allocation.
    ///
    /// A compact diagonal returns its owned conjugated diagonal instead, as
    /// on Host (#1452), so no lazy view ever holds a diagonal parent.
    pub fn adjoint(&self) -> Result<Self, Error> {
        if let Some(adjoint) = self.compact_adjoint() {
            return Ok(adjoint);
        }
        self.dense_adjoint_view()
    }

    /// Borrowed adjoint view of a device tensor. See [`TensorRef`].
    pub fn adjoint_view(&self) -> TensorRef<'_, R, D, CudaStorage<D>> {
        TensorRef {
            base: self,
            adjoint: Some(Self::adjoint),
        }
    }

    /// The Host [`TensorMap::materialize`] contract on the device: an owned
    /// dense device tensor with a fresh allocation on the same device, never
    /// sharing storage with `self`, and equal to the Host result: finite
    /// values, infinities and signed zeros bit for bit. A NaN stays NaN, but
    /// its payload and sign bits may be canonicalized (by the real strided
    /// copy, and by Tenferro's `conj`, which is a float negate).
    ///
    /// Every step moves bits rather than computing them, because a device
    /// copy that scales by a complex `1` does not: cuTENSOR's permutation and
    /// `cuda_region_axpby` both multiply by `(1, 0)`, which turns
    /// `(0, -inf)` into `(NaN, -inf)` and loses `-0`.
    /// - Owned: one Tenferro gather of the whole allocation, which allocates
    ///   the output.
    /// - Real lazy adjoint: the same gather of the parent as the output
    ///   allocation, then one strided copy per block (the transpose; a real
    ///   `1` scale is exact).
    /// - Complex lazy adjoint: one element gather of the parent through an
    ///   `O(required_len)` coordinate table that encodes the transpose, then
    ///   one elementwise conjugation of the whole buffer.
    ///
    /// Nothing is downloaded.
    ///
    /// # Cost
    ///
    /// Owned and real adjoint: one payload-sized device allocation and one
    /// 8-byte index upload; the adjoint adds one copy per block. Complex
    /// adjoint: a host coordinate table of `required_len` entries, built in
    /// one pass and uploaded (`8 * rank(buffer) * required_len` bytes, with a
    /// buffer rank of 1 or 2),
    /// and a second payload-sized allocation for the conjugation, live
    /// together with the gathered buffer. The transfer counters also charge
    /// each index upload as a device allocation. Blocks without a
    /// fusion-tree key read as zero and are zeroed with `cuda_region_zero`.
    ///
    /// # Errors
    ///
    /// [`Error::UnsupportedOnDevice`] for compact diagonal storage (which
    /// [`TensorMap::to_cuda`] never produces), [`Error::PlacementMismatch`]
    /// for a payload on another device, and CUDA backend errors.
    pub fn materialize(&self) -> Result<Self, Error> {
        let dense_err = |err| Error::from(tenet_tensors::OperationError::Dense(err));
        let region = |dims: Vec<usize>, strides: Vec<usize>, offset: usize| {
            tenet_dense::CudaRegion::new(dims, strides, offset).map_err(dense_err)
        };
        let space = self.logical_space().clone();
        let required_len = space.space().required_len()?;
        // (strided copies for a real adjoint, element table for a complex
        // one, regions to zero)
        type Copies = Vec<(tenet_dense::CudaRegion, tenet_dense::CudaRegion)>;
        type Adjoint = (Copies, Vec<usize>, Vec<tenet_dense::CudaRegion>);
        let (source, adjoint): (_, Option<Adjoint>) = match &self.repr {
            TypedTensorRepr::Owned(_) => (self.direct_cuda_storage("materialize")?, None),
            TypedTensorRepr::Adjoint(view) => {
                #[cfg(test)]
                observe_adjoint_materialization();
                let TypedData::Dense(source) = view.parent.data.as_ref() else {
                    unreachable!("TypedAdjointView::new admits only dense parents")
                };
                let parent_space = view.parent.space.space();
                if parent_space.required_len()? != required_len {
                    return Err(internal_layout_error(
                        "an adjoint layout has its parent's payload length",
                    ));
                }
                let (nout, nin) = (parent_space.nout(), parent_space.nin());
                let parent_structure = parent_space.structure();
                let structure = space.space().structure();
                let mut copies = Vec::new();
                // Per block: (shape, destination strides and offset, source
                // strides and offset, or `None` for a block that reads as zero).
                let mut walks = Vec::new();
                let mut zeros = Vec::new();
                let mut covered = 0usize;
                for index in 0..structure.block_count() {
                    let block = structure.block(index)?;
                    let shape = block.shape();
                    covered += shape.iter().product::<usize>();
                    let destination =
                        region(shape.to_vec(), block.strides().to_vec(), block.offset())?;
                    // Why zero instead of error: a non-fusion-tree block has no
                    // adjoint source and reads as zero, as on the Host.
                    let BlockKey::FusionTree(key) = block.key() else {
                        zeros.push(destination);
                        if D::IS_COMPLEX {
                            walks.push((shape, block.strides(), block.offset(), None));
                        }
                        continue;
                    };
                    let source_block = parent_structure.block(
                        parent_structure
                            .find_block_index_by_adjoint_fusion_tree_pair(key)
                            .ok_or_else(|| {
                                internal_layout_error("adjoint block has no parent block")
                            })?,
                    )?;
                    let source_strides: Vec<usize> = (0..shape.len())
                        .map(|axis| {
                            source_block.strides()[logical_adjoint_axis_to_parent(nout, nin, axis)]
                        })
                        .collect();
                    if D::IS_COMPLEX {
                        walks.push((
                            shape,
                            block.strides(),
                            block.offset(),
                            Some((source_strides, source_block.offset())),
                        ));
                    } else {
                        copies.push((
                            region(shape.to_vec(), source_strides, source_block.offset())?,
                            destination,
                        ));
                    }
                }
                if covered != required_len {
                    return Err(internal_layout_error(
                        "adjoint blocks do not tile the payload",
                    ));
                }
                // The table maps each output element to its parent element. A
                // canonical layout lays blocks out compactly in block order, so
                // the walk visits output positions 0, 1, 2, ... and each entry
                // is pushed once; any other order falls back to an indexed fill.
                let mut table = Vec::with_capacity(if D::IS_COMPLEX { required_len } else { 0 });
                let mut sequential = true;
                for (shape, strides, offset, source) in &walks {
                    for_each_block_element(shape, strides, *offset, source, |dst, src| {
                        sequential &= dst == table.len();
                        table.push(src);
                    });
                }
                if !sequential {
                    // ponytail: non-canonical output layouts only; zero-filled
                    // then overwritten, as tiling was checked above.
                    table = vec![0; required_len];
                    for (shape, strides, offset, source) in &walks {
                        for_each_block_element(shape, strides, *offset, source, |dst, src| {
                            table[dst] = src;
                        });
                    }
                }
                (source, Some((copies, table, zeros)))
            }
        };

        let mut lease = self.runtime.lease_cuda()?;
        let cuda = &mut *lease;
        if source.placement() != Placement::Cuda(cuda.device()) {
            return Err(Error::PlacementMismatch);
        }
        let output = match adjoint {
            None => source.gather_members(cuda, required_len, 1, &[0])?,
            Some((copies, table, zeros)) => {
                let mut output = if D::IS_COMPLEX {
                    source.gather_elements(cuda, &table)?.conj(cuda)?
                } else {
                    let mut output = source.gather_members(cuda, required_len, 1, &[0])?;
                    for (source_region, destination_region) in &copies {
                        tenet_dense::cuda_copy_strided_into::<D>(
                            cuda,
                            &source.0,
                            source_region,
                            &mut output.0,
                            destination_region,
                        )
                        .map_err(dense_err)?;
                    }
                    output
                };
                for zero in &zeros {
                    tenet_dense::cuda_region_zero::<D>(cuda, &mut output.0, zero)
                        .map_err(dense_err)?;
                }
                output
            }
        };
        drop(lease);
        Ok(Self {
            runtime: self.runtime.clone(),
            repr: owned_repr(TypedTensorBody::dense(space, output)),
        })
    }

    pub(super) fn direct_cuda_storage(
        &self,
        operation: &'static str,
    ) -> Result<&CudaStorage<D>, Error> {
        match &self.repr {
            TypedTensorRepr::Owned(body) => match body.data.as_ref() {
                TypedData::Dense(storage) => Ok(storage),
                TypedData::Diagonal(_) => Err(Error::UnsupportedOnDevice(format!(
                    "{operation} requires dense CUDA storage"
                ))),
            },
            TypedTensorRepr::Adjoint(_) => Err(Error::UnsupportedOnDevice(format!(
                "{operation} does not support lazy adjoint CUDA operands"
            ))),
        }
    }

    pub(super) fn validate_cuda_owned_metadata(
        expected: Placement,
        actual: Placement,
        required_len: usize,
        actual_len: usize,
    ) -> Result<(), Error> {
        if actual != expected {
            return Err(Error::PlacementMismatch);
        }
        if actual_len != required_len {
            return Err(internal_layout_error(
                "CUDA payload length does not match its admitted tensor space",
            ));
        }
        Ok(())
    }

    pub(super) fn with_owned_cuda_storage(&self, storage: CudaStorage<D>) -> Self {
        let body = self
            .owned_body()
            .expect("CUDA arithmetic output authority must be owned");
        Self {
            runtime: self.runtime.clone(),
            repr: owned_repr(TypedTensorBody::dense(body.space.clone(), storage)),
        }
    }
}
