use super::*;

/// The test probes follow the scope body: it may run on a backend worker
/// thread, and the probes are thread-local so that parallel tests stay apart.
/// Moving them in and out makes a probe observe the call, whichever thread
/// runs it.
#[cfg(test)]
macro_rules! test_probes {
    ($($field:ident: $key:ident: $ty:ty),* $(,)?) => {
        struct TestProbes {
            $($field: $ty,)*
        }

        impl TestProbes {
            pub(super) fn take() -> Self {
                Self { $($field: $key.take(),)* }
            }

            pub(super) fn put(self) {
                $($key.set(self.$field);)*
            }
        }
    };
}

#[cfg(test)]
test_probes! {
    compact_svd: COMPACT_SVD_COPY_PROBE: CompactSvdCopyProbe,
    compact_qr: COMPACT_QR_COPY_PROBE: CompactQrCopyProbe,
    eigh: EIGH_COPY_PROBE: EighCopyProbe,
    eigh_vectors: EIGH_OWNED_VECTOR_POINTERS: Vec<usize>,
    checked_eigh_pairs: CHECKED_EIGH_PAIR_POINTERS: Vec<usize>,
    checked_svd_stage: CHECKED_COMPACT_SVD_STAGE_POINTERS: Vec<(usize, usize)>,
    generic_svd_fallback: GENERIC_COMPACT_SVD_FALLBACK_POINTERS: Vec<(usize, usize)>,
    mf_svd_fallback: MF_COMPACT_SVD_FALLBACK_POINTERS: Vec<(usize, usize)>,
    compact_lq: COMPACT_LQ_COPY_PROBE: CompactLqCopyProbe,
    diagonal_bond: DIAGONAL_BOND_BUILD_PROBE: DiagonalBondBuildProbe,
    values_fallbacks: VALUES_MATRICIZATION_FALLBACKS: usize,
    checked_inputs: CHECKED_COMPACT_INPUT_OBSERVATIONS: Vec<CheckedCompactInputObservation>,
    plan_finish: GENERIC_FACTOR_PLAN_FINISH_CALLS: usize,
    pair_publication: GENERIC_PAIR_PUBLICATION_PROBE: GenericPairPublicationProbe,
    one_sided: ONE_SIDED_PUBLICATION_PROBE: OneSidedPublicationProbe,
    buffer_builds: FACTOR_BUFFER_BUILD_COUNTS: (usize, usize),
    placement_index: PLACEMENT_INDEX_PROBE: PlacementIndexProbe,
    scatter_visits: SCATTER_VISIT_PROBE: ScatterVisitProbe,
}

/// Runs one streaming per-block factorization loop inside a single executor
/// linear-algebra scope: the backend admits the call once, while the loop
/// still holds only one block's input and output at a time. Batching through
/// `factorize_batch` would instead hold every block's factors at once.
pub(super) fn in_linalg_scope<E, T>(
    dense: &mut E,
    body: impl FnOnce(&mut dyn DenseExecutor) -> Result<T, OperationError> + Send,
) -> Result<T, OperationError>
where
    E: DenseExecutor + ?Sized,
    T: Send,
{
    let mut body = Some(body);
    let mut outcome = None;
    #[cfg(test)]
    let mut probes = Some(TestProbes::take());
    let scoped = dense.with_linalg_scope(&mut |dense| {
        tenet_tensors::host_pool::observe_dense_site();
        if let Some(body) = body.take() {
            #[cfg(test)]
            if let Some(probes) = probes.take() {
                probes.put();
            }
            outcome = Some(body(dense));
            #[cfg(test)]
            {
                probes = Some(TestProbes::take());
            }
        }
        Ok(())
    });
    #[cfg(test)]
    if let Some(probes) = probes.take() {
        probes.put();
    }
    scoped.map_err(OperationError::Dense)?;
    outcome.unwrap_or_else(|| {
        Err(OperationError::Dense(DenseError::Backend {
            backend: DenseBackend::Tenferro,
            op: "with_linalg_scope",
            message: "dense executor returned without running the scope body".to_string(),
        }))
    })
}

pub(super) fn factorize_col_major_batch<E, D>(
    dense: &mut E,
    op: DenseFactorization,
    blocks: &[(&[D], usize, usize)],
) -> Result<Vec<Vec<DenseTensor>>, OperationError>
where
    E: DenseExecutor + ?Sized,
    D: FactorScalar,
{
    let layouts = blocks
        .iter()
        .map(|&(_, rows, cols)| ([rows, cols], [1usize, rows]))
        .collect::<Vec<_>>();
    let inputs = blocks
        .iter()
        .zip(&layouts)
        .map(|(&(data, _, _), (shape, strides))| {
            DenseView::new(data, shape, strides, 0).map(D::dense_read)
        })
        .collect::<Result<Vec<_>, _>>()
        .map_err(OperationError::Dense)?;
    let outputs = dense
        .factorize_batch(op, &inputs)
        .map_err(OperationError::Dense)?;
    // An overriding executor is outside this crate: a short batch would
    // otherwise drop a sector silently or panic in a consumer. Each entry's
    // factor count is checked by `compact_{qr,svd}_outputs`, with the same
    // error the per-matrix path reports.
    if outputs.len() != blocks.len() {
        return Err(OperationError::Dense(DenseError::Backend {
            backend: DenseBackend::Tenferro,
            op: "factorize_batch",
            message: format!(
                "dense factorize_batch returned {} entries for {} inputs",
                outputs.len(),
                blocks.len()
            ),
        }));
    }
    Ok(outputs)
}

pub(super) fn compact_factor_output_owned<D: FactorScalar>(
    tensor: DenseTensor,
    expected_shape: &[usize],
    op: &'static str,
) -> Result<Vec<D>, OperationError> {
    let source = D::dense_slice(&tensor).map_err(OperationError::Dense)?;
    let shape = tensor.shape();
    if shape != expected_shape {
        return Err(OperationError::Dense(DenseError::Backend {
            backend: DenseBackend::Tenferro,
            op,
            message: format!(
                "{op} output shape mismatch: source {shape:?}, destination {expected_shape:?}",
            ),
        }));
    }
    let expected_len = shape
        .iter()
        .try_fold(1usize, |acc, &dim| match acc.checked_mul(dim) {
            Some(count) => Ok(count),
            None => Err(DenseError::ElementCountOverflow),
        })
        .map_err(OperationError::Dense)?;
    if source.len() != expected_len {
        return Err(OperationError::Dense(DenseError::Backend {
            backend: DenseBackend::Tenferro,
            op,
            message: format!(
                "{op} output storage length mismatch: source {}, expected {}",
                source.len(),
                expected_len
            ),
        }));
    }
    D::dense_into_vec(tensor).map_err(OperationError::Dense)
}

pub(super) fn compact_real_spectrum_owned<D: FactorScalar>(
    tensor: DenseTensor,
    expected_shape: &[usize],
    op: &'static str,
) -> Result<Vec<f64>, OperationError> {
    let spectrum = D::real_spectrum(&tensor).map_err(OperationError::Dense)?;
    let shape = tensor.shape();
    if shape != expected_shape {
        return Err(OperationError::Dense(DenseError::Backend {
            backend: DenseBackend::Tenferro,
            op,
            message: format!(
                "{op} output shape mismatch: source {shape:?}, destination {expected_shape:?}",
            ),
        }));
    }
    let expected_len = shape
        .iter()
        .try_fold(1usize, |acc, &dim| match acc.checked_mul(dim) {
            Some(count) => Ok(count),
            None => Err(DenseError::ElementCountOverflow),
        })
        .map_err(OperationError::Dense)?;
    if spectrum.len() != expected_len {
        return Err(OperationError::Dense(DenseError::Backend {
            backend: DenseBackend::Tenferro,
            op,
            message: format!(
                "{op} output storage length mismatch: source {}, expected {}",
                spectrum.len(),
                expected_len
            ),
        }));
    }
    Ok(spectrum)
}

pub(super) fn concat_compact_factor_regions<D>(
    regions: Vec<Option<Vec<D>>>,
    required_len: usize,
) -> Vec<D> {
    #[cfg(test)]
    let first = regions
        .iter()
        .find_map(|region| region.as_ref().map(Vec::as_ptr));
    let output = concat_owned_factor_regions(regions, required_len);
    #[cfg(test)]
    COMPACT_QR_COPY_PROBE.with(|probe| {
        let mut current = probe.get();
        current.owned_output_publications += 1;
        current.owned_output_owner_reused +=
            usize::from(first.is_some_and(|pointer| std::ptr::eq(pointer, output.as_ptr())));
        probe.set(current);
    });
    output
}

pub(super) fn concat_owned_factor_regions<D>(
    regions: Vec<Option<Vec<D>>>,
    required_len: usize,
) -> Vec<D> {
    let mut output = None;
    for region in regions.into_iter().flatten() {
        append_owned_factor(&mut output, region, required_len);
    }
    output.unwrap_or_default()
}

pub(super) fn copy_col_major_strided<D: Copy>(
    source: &[D],
    rows: usize,
    cols: usize,
    source_leading: usize,
    destination: &mut [D],
    destination_leading: usize,
) {
    for col in 0..cols {
        let src_start = source_leading * col;
        let dst_start = destination_leading * col;
        destination[dst_start..dst_start + rows]
            .copy_from_slice(&source[src_start..src_start + rows]);
    }
}

pub(super) fn advance_outer_index(index: &mut [usize], shape: &[usize]) {
    for axis in 1..shape.len() {
        index[axis] += 1;
        if index[axis] < shape[axis] {
            break;
        }
        index[axis] = 0;
    }
}

#[allow(clippy::too_many_arguments)]
pub(super) fn copy_tensor_block_to_matrix<D: Copy>(
    source: &[D],
    shape: &[usize],
    strides: &[usize],
    offset: usize,
    nout: usize,
    matrix: &mut [D],
    matrix_rows: usize,
    row_offset: usize,
    col_offset: usize,
) {
    if shape.is_empty() {
        matrix[row_offset + matrix_rows * col_offset] = source[offset];
        return;
    }
    let run = shape[0];
    let src_lane_stride = strides[0];
    let dst_lane_stride = if nout > 0 { 1 } else { matrix_rows };
    let outer_count: usize = shape[1..].iter().product();
    let mut index = vec![0usize; shape.len()];
    for _ in 0..outer_count {
        let mut src_start = offset;
        let mut row = 0usize;
        let mut row_stride = if nout > 0 { shape[0] } else { 1 };
        let mut col = 0usize;
        let mut col_stride = if nout == 0 { shape[0] } else { 1 };
        for axis in 1..shape.len() {
            src_start += index[axis] * strides[axis];
            if axis < nout {
                row += index[axis] * row_stride;
                row_stride *= shape[axis];
            } else {
                col += index[axis] * col_stride;
                col_stride *= shape[axis];
            }
        }
        let dst_start = (row_offset + row) + matrix_rows * (col_offset + col);
        if src_lane_stride == 1 && dst_lane_stride == 1 {
            matrix[dst_start..dst_start + run].copy_from_slice(&source[src_start..src_start + run]);
        } else {
            for lane in 0..run {
                matrix[dst_start + lane * dst_lane_stride] =
                    source[src_start + lane * src_lane_stride];
            }
        }
        advance_outer_index(&mut index, shape);
    }
}

pub(super) fn copy_mapped_to_strided_diagonal<D, V, F>(
    data: &mut [D],
    offset: usize,
    diagonal_stride: usize,
    values: &[V],
    to_scalar: &F,
) where
    V: Copy,
    F: Fn(V) -> D + ?Sized,
{
    for (position, &value) in values.iter().enumerate() {
        data[offset + position * diagonal_stride] = to_scalar(value);
    }
}

/// Copies a dense column-major matrix region into one fusion-tree subblock.
///
/// `matrix_axis` names the block axis that walks the matrix's own leading
/// dimension side; the remaining axes enumerate the offset side column-major.
/// For `U` the matrix axis is the trailing (new leg) axis and the codomain
/// axes select rows at `side_offset`; for `Vt` the matrix axis is the leading
/// (new leg) axis and the domain axes select columns at `side_offset`.
/// `factor_side` names the factor layout (`Left`: `a x b`, element `(o, j)` at
/// `F[o + a*j]`; `Right`: `b x cols`, element `(j, o)` at `F[j + b*o]`).
#[allow(clippy::too_many_arguments)]
pub(super) fn scatter_matrix_block<D: Copy>(
    data: &mut [D],
    shape: &[usize],
    strides: &[usize],
    offset: usize,
    matrix_axis: usize,
    factor_side: FactorSide,
    matrix: &[D],
    matrix_rows: usize,
    side_offset: usize,
) {
    if shape.is_empty() {
        data[offset] = matrix[side_offset];
        return;
    }
    let rank = shape.len();
    let run = shape[0];
    let dst_lane_stride = strides[0];
    let outer_count: usize = shape[1..].iter().product();
    let mut index = vec![0usize; rank];
    for _ in 0..outer_count {
        let mut dst_start = offset;
        let mut side = 0usize;
        let mut side_stride = if matrix_axis == 0 { 1 } else { shape[0] };
        let mut matrix_index = 0usize;
        for axis in 1..rank {
            dst_start += index[axis] * strides[axis];
            if axis == matrix_axis {
                matrix_index = index[axis];
            } else {
                side += index[axis] * side_stride;
                side_stride *= shape[axis];
            }
        }
        // A rank-1 block (zero-leg tree) has `matrix_axis == 0 == rank - 1` on
        // both sides, so `matrix_axis` alone cannot encode the source layout;
        // only a Left factor walks the bond with stride `matrix_rows`.
        let (src_start, src_lane_stride) = if matrix_axis == 0 {
            if matrix_axis == rank - 1 && factor_side == FactorSide::Left {
                (side_offset + side, matrix_rows)
            } else {
                (matrix_rows * (side_offset + side), 1)
            }
        } else if matrix_axis == rank - 1 {
            ((side_offset + side) + matrix_rows * matrix_index, 1)
        } else {
            (
                matrix_index + matrix_rows * (side_offset + side),
                matrix_rows,
            )
        };
        if src_lane_stride == 1 && dst_lane_stride == 1 {
            data[dst_start..dst_start + run].copy_from_slice(&matrix[src_start..src_start + run]);
        } else {
            for lane in 0..run {
                data[dst_start + lane * dst_lane_stride] =
                    matrix[src_start + lane * src_lane_stride];
            }
        }
        advance_outer_index(&mut index, shape);
    }
}

#[allow(clippy::too_many_arguments)]
pub(super) fn scatter_identity_matrix_block<D: FactorScalar>(
    data: &mut [D],
    shape: &[usize],
    strides: &[usize],
    offset: usize,
    matrix_axis: usize,
    matrix_dimension: usize,
    side_offset: usize,
    side_extent: usize,
) -> Result<(), OperationError> {
    if shape.get(matrix_axis).copied() != Some(matrix_dimension) {
        return Err(OperationError::ElementCountMismatch {
            expected: matrix_dimension,
            actual: shape.get(matrix_axis).copied().unwrap_or(0),
        });
    }
    for local_side in 0..side_extent {
        let matrix_index = side_offset
            .checked_add(local_side)
            .ok_or(OperationError::ElementCountOverflow)?;
        let mut destination = offset + matrix_index * strides[matrix_axis];
        let mut remaining = local_side;
        for axis in 0..shape.len() {
            if axis == matrix_axis {
                continue;
            }
            let coordinate = remaining % shape[axis];
            remaining /= shape[axis];
            destination += coordinate * strides[axis];
        }
        data[destination] = D::one();
    }
    Ok(())
}
