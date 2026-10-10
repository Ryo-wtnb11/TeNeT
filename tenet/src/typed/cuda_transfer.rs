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

#[cfg(feature = "cuda")]
impl<R, D> MaterializeExec<R, D> for CudaStorage<D>
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64> + CheckedFusionAlgebra + SectorCodec,
    D: CudaPayload,
{
    fn materialize(
        t: &TensorMap<R, D, Self>,
        source: &Self,
        map: Option<&tenet_tensors::AdjointBlockMap<'_>>,
    ) -> Result<Self, Error> {
        let dense_err = |err| Error::from(tenet_tensors::OperationError::Dense(err));
        let required_len = t.logical_space().space().required_len()?;
        let adjoint = map.map(cuda_adjoint_lowering::<D>).transpose()?;

        let mut lease = t.runtime.lease_cuda()?;
        let cuda = &mut *lease;
        if source.placement() != Placement::Cuda(cuda.device()) {
            return Err(Error::PlacementMismatch);
        }
        let output = match adjoint {
            None => source.gather_members(cuda, required_len, 1, &[0])?,
            Some(CudaAdjointLowering {
                copies,
                table,
                zeros,
            }) => {
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
        Ok(output)
    }
}

/// The device lowering of an [`tenet_tensors::AdjointBlockMap`], on the Host
/// before the lease: the strided copies (real) or the element table
/// (complex) that conjugate-transpose the parent, and the regions of blocks
/// that read as zero.
#[cfg(feature = "cuda")]
pub(super) struct CudaAdjointLowering {
    /// (parent region, output region) per block of a real payload.
    pub(super) copies: Vec<(tenet_dense::CudaRegion, tenet_dense::CudaRegion)>,
    /// The parent element of each output element of a complex payload.
    pub(super) table: Vec<usize>,
    pub(super) zeros: Vec<tenet_dense::CudaRegion>,
}

#[cfg(feature = "cuda")]
pub(super) fn cuda_adjoint_lowering<D: CudaPayload>(
    map: &tenet_tensors::AdjointBlockMap<'_>,
) -> Result<CudaAdjointLowering, Error> {
    let region = |dims: Vec<usize>, strides: Vec<usize>, offset: usize| {
        tenet_dense::CudaRegion::new(dims, strides, offset)
            .map_err(|err| Error::from(tenet_tensors::OperationError::Dense(err)))
    };
    let space = map.adjoint_space();
    let required_len = space.required_len()?;
    if map.parent_space().required_len()? != required_len {
        return Err(internal_layout_error(
            "an adjoint layout has its parent's payload length",
        ));
    }
    let structure = space.structure();
    let mut copies = Vec::new();
    // Per block: (shape, destination strides and offset, source strides and
    // offset, or `None` for a block that reads as zero).
    let mut walks = Vec::new();
    let mut zeros = Vec::new();
    let mut covered = 0usize;
    for index in 0..structure.block_count() {
        let block = structure.block(index)?;
        let shape = block.shape();
        covered += shape.iter().product::<usize>();
        let destination = region(shape.to_vec(), block.strides().to_vec(), block.offset())?;
        let Some(source) = map.source(block)? else {
            zeros.push(destination);
            if D::IS_COMPLEX {
                walks.push((shape, block.strides(), block.offset(), None));
            }
            continue;
        };
        let source_strides = (0..shape.len())
            .map(|axis| source.stride(axis))
            .collect::<Result<Vec<_>, _>>()?;
        if D::IS_COMPLEX {
            walks.push((
                shape,
                block.strides(),
                block.offset(),
                Some((source_strides, source.offset())),
            ));
        } else {
            copies.push((
                region(shape.to_vec(), source_strides, source.offset())?,
                destination,
            ));
        }
    }
    if covered != required_len {
        return Err(internal_layout_error(
            "adjoint blocks do not tile the payload",
        ));
    }
    // The table maps each output element to its parent element. A canonical
    // layout lays blocks out compactly in block order, so the walk visits
    // output positions 0, 1, 2, ... and each entry is pushed once; any other
    // order falls back to an indexed fill.
    let mut table = Vec::with_capacity(if D::IS_COMPLEX { required_len } else { 0 });
    let mut sequential = true;
    for (shape, strides, offset, source) in &walks {
        for_each_block_element(shape, strides, *offset, source, |dst, src| {
            sequential &= dst == table.len();
            table.push(src);
        });
    }
    if !sequential {
        // ponytail: non-canonical output layouts only; zero-filled then
        // overwritten, as tiling was checked above.
        table = vec![0; required_len];
        for (shape, strides, offset, source) in &walks {
            for_each_block_element(shape, strides, *offset, source, |dst, src| {
                table[dst] = src;
            });
        }
    }
    Ok(CudaAdjointLowering {
        copies,
        table,
        zeros,
    })
}

#[cfg(feature = "cuda")]
/// Calls `visit(destination, source)` for every element of one block, first
/// axis fastest; a `None` source (a block that reads as zero) visits `0`.
fn for_each_block_element(
    shape: &[usize],
    strides: &[usize],
    offset: usize,
    source: &Option<(Vec<usize>, usize)>,
    mut visit: impl FnMut(usize, usize),
) {
    let mut position = vec![0usize; shape.len()];
    let at = |position: &[usize], strides: &[usize], base: usize| {
        base + position
            .iter()
            .zip(strides)
            .map(|(i, stride)| i * stride)
            .sum::<usize>()
    };
    for _ in 0..shape.iter().product::<usize>() {
        let src = source
            .as_ref()
            .map_or(0, |(strides, base)| at(&position, strides, *base));
        visit(at(&position, strides, offset), src);
        for (axis, extent) in shape.iter().enumerate() {
            position[axis] += 1;
            if position[axis] < *extent {
                break;
            }
            position[axis] = 0;
        }
    }
}
