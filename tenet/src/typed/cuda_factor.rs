use super::*;

#[cfg(all(test, feature = "cuda"))]
thread_local! {
    /// `(download_calls, device_partials_len, host_partials_len)`.
    pub(super) static CUDA_REDUCTION_BUFFER_OBSERVATION:
        std::cell::Cell<Option<(usize, usize, usize)>> = const {
            std::cell::Cell::new(None)
        };
    /// `(payload_zero_uploads, coefficient_uploads, kernels)`.
    pub(super) static CUDA_ARITHMETIC_OBSERVATION:
        std::cell::Cell<Option<(usize, usize, usize)>> = const {
            std::cell::Cell::new(None)
        };
    /// `(qr_calls, factor_copies, selector_uploads, output_uploads,
    /// assembly_gemms, live_route_scratch, peak_route_scratch)`.
    pub(super) static CUDA_QR_OBSERVATION: std::cell::Cell<Option<CudaQrObservation>> = const {
            std::cell::Cell::new(None)
        };
    /// `(successful_results, spectrum_scalars, final_storage_creations,
    /// live_route_scratch, peak_route_scratch)`.
    pub(super) static CUDA_SVD_OBSERVATION: std::cell::Cell<Option<CudaSvdObservation>> = const {
            std::cell::Cell::new(None)
        };
    /// `(stage, one-based ordinal)` for operation-local failure injection.
    pub(super) static CUDA_EIGH_FAILURE: std::cell::Cell<Option<(&'static str, usize)>> = const {
        std::cell::Cell::new(None)
    };
    /// Forces the per-tree EIGH assembly on aligned routes, so the general
    /// path can be compared with the aligned one on the same input.
    pub(super) static CUDA_EIGH_TREEWISE: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    /// Marks every compact-SVD route non-aligned, so the selector-GEMM
    /// assembly runs on a layout whose public constructions are all aligned.
    pub(super) static CUDA_SVD_TREEWISE: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    /// Forces the selector-GEMM EIGH assembly even when every order is the
    /// identity, so the column-copy path can be compared with it.
    pub(super) static CUDA_EIGH_FORCE_SELECTOR: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    /// Selector uploads made by device `eigh_full` assembly.
    pub(super) static CUDA_EIGH_SELECTOR_UPLOADS: std::cell::Cell<Option<usize>> = const {
        std::cell::Cell::new(None)
    };
}

#[cfg(all(test, feature = "cuda"))]
pub(super) type CudaQrObservation = (usize, usize, usize, usize, usize, usize, usize);
#[cfg(all(test, feature = "cuda"))]
pub(super) type CudaSvdObservation = (usize, usize, usize, usize, usize);

#[cfg(all(test, feature = "cuda"))]
pub(super) fn update_cuda_svd_observation(
    update: impl FnOnce(CudaSvdObservation) -> CudaSvdObservation,
) {
    CUDA_SVD_OBSERVATION.with(|observation| {
        if let Some(current) = observation.get() {
            observation.set(Some(update(current)));
        }
    });
}

#[cfg(all(test, feature = "cuda"))]
pub(crate) fn observe_cuda_svd_decomposition(values: usize) {
    update_cuda_svd_observation(|(results, total, creations, live, peak)| {
        (results + 1, total + values, creations, live, peak)
    });
}

#[cfg(all(test, feature = "cuda"))]
pub(super) fn observe_cuda_svd_final_storage_creation() {
    update_cuda_svd_observation(|(results, total, creations, live, peak)| {
        (results, total, creations + 1, live, peak)
    });
}

#[cfg(all(test, feature = "cuda"))]
pub(super) fn observe_cuda_arithmetic(
    zero_uploads: usize,
    coefficient_uploads: usize,
    kernels: usize,
) {
    CUDA_ARITHMETIC_OBSERVATION.with(|observation| {
        if let Some((zeros, coefficients, calls)) = observation.get() {
            observation.set(Some((
                zeros + zero_uploads,
                coefficients + coefficient_uploads,
                calls + kernels,
            )));
        }
    });
}

#[cfg(all(test, feature = "cuda"))]
pub(super) fn update_cuda_qr_observation(
    update: impl FnOnce(CudaQrObservation) -> CudaQrObservation,
) {
    CUDA_QR_OBSERVATION.with(|observation| {
        if let Some(current) = observation.get() {
            observation.set(Some(update(current)));
        }
    });
}

#[cfg(all(test, feature = "cuda"))]
pub(crate) fn observe_cuda_qr_decomposition() {
    update_cuda_qr_observation(|(qr, copies, selectors, outputs, gemms, live, peak)| {
        (qr + 1, copies, selectors, outputs, gemms, live, peak)
    });
}

/// One whole-factor device copy on a layout-aligned assembly route.
///
/// Shared by compact QR and compact SVD: both assemble through
/// [`copy_whole_factor`], so both gate tests arm and assert this observation.
#[cfg(all(test, feature = "cuda"))]
pub(crate) fn observe_cuda_factor_copy() {
    update_cuda_qr_observation(|(qr, copies, selectors, outputs, gemms, live, peak)| {
        (qr, copies + 1, selectors, outputs, gemms, live, peak)
    });
}

#[cfg(all(test, feature = "cuda"))]
pub(crate) fn observe_cuda_qr_selector_upload() {
    update_cuda_qr_observation(|(qr, copies, selectors, outputs, gemms, live, peak)| {
        (qr, copies, selectors + 1, outputs, gemms, live, peak)
    });
}

#[cfg(all(test, feature = "cuda"))]
pub(crate) fn observe_cuda_factor_assembly_gemm() {
    update_cuda_qr_observation(|(qr, copies, selectors, outputs, gemms, live, peak)| {
        (qr, copies, selectors, outputs, gemms + 1, live, peak)
    });
}

#[cfg(all(test, feature = "cuda"))]
pub(super) fn observe_cuda_qr_output_upload() {
    update_cuda_qr_observation(|(qr, copies, selectors, outputs, gemms, live, peak)| {
        (qr, copies, selectors, outputs + 1, gemms, live, peak)
    });
}

#[cfg(feature = "cuda")]
pub(crate) fn dense_err(err: tenet_dense::DenseError) -> Error {
    Error::from(tenet_tensors::OperationError::Dense(err))
}

/// Uploads a small host-built selector matrix (`rows x cols`, column-major,
/// zero except `entries`) used by the assembly GEMMs.
#[cfg(feature = "cuda")]
pub(crate) fn upload_selector<D: CudaPayload>(
    cuda: &mut CudaDenseContext,
    rows: usize,
    cols: usize,
    entries: impl Iterator<Item = (usize, usize, D)>,
) -> Result<CudaStorage<D>, Error> {
    let mut data = vec![D::ZERO; rows * cols];
    for (row, col, value) in entries {
        data[row + rows * col] = value;
    }
    let selector = CudaStorage::upload_owned(cuda, data).map_err(Error::from)?;
    #[cfg(test)]
    observe_cuda_qr_selector_upload();
    Ok(selector)
}

#[cfg(feature = "cuda")]
#[inline]
pub(crate) fn cuda_qr_region<D: CudaPayload>(
    cuda: &mut CudaDenseContext,
    source: &CudaDenseStorage,
    offset: usize,
    rows: usize,
    cols: usize,
) -> Result<(CudaDenseStorage, CudaDenseStorage), Error> {
    let factors = dense_cuda_qr_region::<D>(cuda, source, offset, rows, cols).map_err(dense_err)?;
    #[cfg(test)]
    observe_cuda_qr_decomposition();
    Ok(factors)
}

#[cfg(feature = "cuda")]
#[inline]
pub(crate) fn cuda_svd_region<D: CudaPayload>(
    cuda: &mut CudaDenseContext,
    source: &CudaDenseStorage,
    offset: usize,
    rows: usize,
    cols: usize,
) -> Result<
    (
        CudaDenseStorage,
        tenet_dense::CudaSpectrum,
        CudaDenseStorage,
    ),
    Error,
> {
    let factors =
        dense_cuda_svd_region::<D>(cuda, source, offset, rows, cols).map_err(dense_err)?;
    #[cfg(test)]
    observe_cuda_svd_decomposition(factors.1.len());
    Ok(factors)
}

#[cfg(feature = "cuda")]
#[inline]
pub(crate) fn cuda_hermitian_regions<D: CudaPayload>(
    cuda: &mut CudaDenseContext,
    source: &CudaDenseStorage,
    regions: &[(usize, usize)],
    relative_tolerance: f64,
) -> Result<Vec<bool>, Error> {
    dense_cuda_hermitian_regions::<D>(cuda, source, regions, relative_tolerance).map_err(dense_err)
}

#[cfg(feature = "cuda")]
#[inline]
pub(crate) fn typed_cuda_eigh_region<D: CudaPayload>(
    cuda: &mut CudaDenseContext,
    source: &CudaDenseStorage,
    offset: usize,
    n: usize,
) -> Result<(tenet_dense::CudaSpectrum, CudaDenseStorage), Error> {
    cuda_eigh_region::<D>(cuda, source, offset, n).map_err(dense_err)
}

/// Downloads the device spectra of one factorization call with one transfer.
#[cfg(feature = "cuda")]
#[inline]
pub(crate) fn cuda_download_spectra<D: CudaPayload>(
    cuda: &mut CudaDenseContext,
    spectra: &[tenet_dense::CudaSpectrum],
) -> Result<Vec<Vec<f64>>, Error> {
    tenet_dense::cuda_download_spectra::<D>(cuda, spectra).map_err(dense_err)
}

/// Writes `factor_rows x kept` slices of `factor * selector` into the target
/// sector region of a left factor (`codomain <- bond`), one GEMM per
/// codomain tree so correctness never relies on tree enumeration order
/// matching between the source and factor spaces.
#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
pub(crate) fn assemble_left_factor<D: CudaPayload>(
    cuda: &mut CudaDenseContext,
    dst: &mut CudaStorage<D>,
    target: &CoupledSectorRegion,
    source: &CoupledSectorRegion,
    factor: &CudaDenseStorage,
    k_full: usize,
    selector: &CudaDenseStorage,
    selector_offset: usize,
    kept: usize,
) -> Result<(), Error> {
    for target_tree in target.row_trees() {
        let sub_rows = target_tree.extent()?;
        if sub_rows == 0 {
            continue;
        }
        let src_row = source
            .row_trees()
            .iter()
            .find(|source_tree| source_tree.tree() == target_tree.tree())
            .map(|source_tree| source_tree.offset())
            .ok_or_else(|| internal_layout_error("codomain tree missing in the source sector"))?;
        cuda_gemm_region_into::<D>(
            cuda,
            &mut dst.0,
            target.range().start + target_tree.offset(),
            target.rows(),
            factor,
            src_row,
            source.rows(),
            selector,
            selector_offset,
            k_full,
            sub_rows,
            k_full,
            kept,
            D::ONE,
            D::ZERO,
        )
        .map_err(dense_err)?;
        #[cfg(test)]
        observe_cuda_factor_assembly_gemm();
    }
    Ok(())
}

/// Writes `factor * selector` into a left-factor region with one GEMM.
///
/// Only valid on a route the plan proved layout-aligned: the target's
/// codomain trees tile it exactly as the source's do, so the per-tree GEMMs
/// of [`assemble_left_factor`] are row slices of this one product.
///
/// The product is exact data movement: each selector column holds a single
/// unit entry. Why not Tenferro `gather`: at 0.7.1 it reads owned index
/// tensors only, so one index upload per call would need a device slice copy
/// per route to feed it.
#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
pub(crate) fn assemble_aligned_left_factor<D: CudaPayload>(
    cuda: &mut CudaDenseContext,
    dst: &mut CudaStorage<D>,
    target: &CoupledSectorRegion,
    factor: &CudaDenseStorage,
    k_full: usize,
    selector: &CudaStorage<D>,
    selector_offset: usize,
    kept: usize,
) -> Result<(), Error> {
    if target.rows() == 0 {
        return Ok(());
    }
    cuda_gemm_region_into::<D>(
        cuda,
        &mut dst.0,
        target.range().start,
        target.rows(),
        factor,
        0,
        target.rows(),
        &selector.0,
        selector_offset,
        k_full,
        target.rows(),
        k_full,
        kept,
        D::ONE,
        D::ZERO,
    )
    .map_err(dense_err)?;
    #[cfg(test)]
    observe_cuda_factor_assembly_gemm();
    Ok(())
}

/// Writes `kept x factor_cols` slices of `selector * factor` into the target
/// sector region of a right factor (`bond <- domain`), one GEMM per domain
/// tree.
#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
pub(crate) fn assemble_right_factor<D: CudaPayload>(
    cuda: &mut CudaDenseContext,
    dst: &mut CudaStorage<D>,
    target: &CoupledSectorRegion,
    source: &CoupledSectorRegion,
    selector: &CudaDenseStorage,
    kept: usize,
    k_full: usize,
    factor: &CudaDenseStorage,
) -> Result<(), Error> {
    for target_tree in target.col_trees() {
        let sub_cols = target_tree.extent()?;
        if sub_cols == 0 {
            continue;
        }
        let src_col = source
            .col_trees()
            .iter()
            .find(|source_tree| source_tree.tree() == target_tree.tree())
            .map(|source_tree| source_tree.offset())
            .ok_or_else(|| internal_layout_error("domain tree missing in the source sector"))?;
        cuda_gemm_region_into::<D>(
            cuda,
            &mut dst.0,
            target.range().start + target.rows() * target_tree.offset(),
            target.rows(),
            selector,
            0,
            kept,
            factor,
            k_full * src_col,
            k_full,
            kept,
            k_full,
            sub_cols,
            D::ONE,
            D::ZERO,
        )
        .map_err(dense_err)?;
        #[cfg(test)]
        observe_cuda_factor_assembly_gemm();
    }
    Ok(())
}

/// Copies a whole compact device factor into its target sector region.
///
/// Only valid on a route the plan proved layout-aligned: the target region's
/// tree sequence equals the source's (same trees, extents and offsets), so the
/// region is exactly the compact factor and the identity-selector GEMM reduces
/// to one device copy. The per-tree GEMM stays the general assembly path.
#[cfg(feature = "cuda")]
pub(crate) fn copy_whole_factor<D: CudaPayload>(
    cuda: &mut CudaDenseContext,
    dst: &mut CudaStorage<D>,
    target: &CoupledSectorRegion,
    factor: &CudaDenseStorage,
) -> Result<(), Error> {
    // `cuda_copy_region_into` accepts a source at least as long as the region
    // so one long buffer can serve every shorter region; this route instead
    // promises the factor *is* the region, so assert that here rather
    // than silently copying a prefix of a longer factor.
    if factor.len() != target.rows().saturating_mul(target.cols()) {
        return Err(internal_layout_error(
            "device factor length does not match its aligned target region",
        ));
    }
    cuda_copy_region_into::<D>(
        cuda,
        &mut dst.0,
        target.range().start,
        target.rows(),
        factor,
        target.rows(),
        target.cols(),
    )
    .map_err(dense_err)?;
    #[cfg(test)]
    observe_cuda_factor_copy();
    Ok(())
}

/// Copies every column of a full `source.rows() x cols` device factor into a
/// left-factor region, one strided copy per codomain tree: the per-tree
/// assembly of [`assemble_left_factor`] when its selector is the identity.
#[cfg(feature = "cuda")]
pub(crate) fn copy_left_factor_treewise<D: CudaPayload>(
    cuda: &mut CudaDenseContext,
    dst: &mut CudaStorage<D>,
    target: &CoupledSectorRegion,
    source: &CoupledSectorRegion,
    factor: &CudaDenseStorage,
    cols: usize,
) -> Result<(), Error> {
    let region = |rows: usize, leading: usize, offset: usize| {
        tenet_dense::CudaRegion::new(vec![rows, cols], vec![1, leading], offset).map_err(dense_err)
    };
    for target_tree in target.row_trees() {
        let sub_rows = target_tree.extent()?;
        if sub_rows == 0 {
            continue;
        }
        let src_row = source
            .row_trees()
            .iter()
            .find(|source_tree| source_tree.tree() == target_tree.tree())
            .map(|source_tree| source_tree.offset())
            .ok_or_else(|| internal_layout_error("codomain tree missing in the source sector"))?;
        tenet_dense::cuda_copy_strided_into::<D>(
            cuda,
            factor,
            &region(sub_rows, source.rows(), src_row)?,
            &mut dst.0,
            &region(
                sub_rows,
                target.rows(),
                target.range().start + target_tree.offset(),
            )?,
        )
        .map_err(dense_err)?;
        #[cfg(test)]
        observe_cuda_factor_copy();
    }
    Ok(())
}

/// Fills the diagonal of a coupled-layout `W <- W` buffer from per-sector
/// spectra, mirroring the host `diagonal_bond_tensor_dyn`.
#[cfg(feature = "cuda")]
pub(crate) fn fill_diagonal_values<D: CudaPayload>(
    structure: &BlockStructure,
    data: &mut [D],
    spectra: &[tenet_matrixalgebra::SectorSpectrum<f64>],
) -> Result<(), Error> {
    for index in 0..structure.block_count() {
        let block = structure.block(index)?;
        let BlockKey::FusionTree(tree) = block.key() else {
            continue;
        };
        let sector = tree.codomain_tree().coupled();
        let Some(entry) = spectra.iter().find(|entry| entry.sector == sector) else {
            continue;
        };
        let strides = block.strides();
        let offset = block.offset();
        let count = block.shape()[0].min(block.shape()[1]);
        for position in 0..count {
            data[offset + position * (strides[0] + strides[1])] =
                D::from_real(entry.values[position]);
        }
    }
    Ok(())
}

/// The plan's nonzero-rank routes with their `(left, right)` factor regions:
/// a zero-rank sector has nothing to decompose or publish.
#[cfg(feature = "cuda")]
pub(super) fn executed_routes(
    plan: &CompactFactorPlan,
) -> impl Iterator<Item = (&CompactFactorRoute, usize, usize)> {
    plan.routes()
        .iter()
        .filter_map(|route| Some((route, route.left_region()?, route.right_region()?)))
}

#[cfg(feature = "cuda")]
pub(super) struct TypedCudaQrScratch<D: CudaPayload> {
    pub(super) left: CudaDenseStorage,
    pub(super) right: CudaDenseStorage,
    /// `None` when both assembly routes are layout-aligned and copy instead.
    pub(super) selector: Option<CudaStorage<D>>,
}

#[cfg(feature = "cuda")]
impl<D: CudaPayload> TypedCudaQrScratch<D> {
    pub(super) fn new(
        left: CudaDenseStorage,
        right: CudaDenseStorage,
        selector: Option<CudaStorage<D>>,
    ) -> Self {
        #[cfg(test)]
        update_cuda_qr_observation(|(qr, copies, selectors, outputs, gemms, live, peak)| {
            let live = live + 1;
            (qr, copies, selectors, outputs, gemms, live, peak.max(live))
        });
        Self {
            left,
            right,
            selector,
        }
    }
}

#[cfg(all(test, feature = "cuda"))]
impl<D: CudaPayload> Drop for TypedCudaQrScratch<D> {
    fn drop(&mut self) {
        update_cuda_qr_observation(|(qr, copies, selectors, outputs, gemms, live, peak)| {
            (qr, copies, selectors, outputs, gemms, live - 1, peak)
        });
    }
}

#[cfg(feature = "cuda")]
pub(super) struct TypedCudaSvdScratch<D: CudaPayload> {
    pub(super) left: CudaDenseStorage,
    pub(super) right: CudaDenseStorage,
    /// `None` on a layout-aligned side, whose gauged factor is copied.
    pub(super) left_selector: Option<CudaDenseStorage>,
    pub(super) right_selector: Option<CudaDenseStorage>,
    payload: std::marker::PhantomData<D>,
}

#[cfg(feature = "cuda")]
impl<D: CudaPayload> TypedCudaSvdScratch<D> {
    pub(super) fn new(
        left: CudaDenseStorage,
        right: CudaDenseStorage,
        left_selector: Option<CudaDenseStorage>,
        right_selector: Option<CudaDenseStorage>,
    ) -> Self {
        #[cfg(test)]
        update_cuda_svd_observation(|(results, total, creations, live, peak)| {
            let live = live + 1;
            (results, total, creations, live, peak.max(live))
        });
        Self {
            left,
            right,
            left_selector,
            right_selector,
            payload: std::marker::PhantomData,
        }
    }

    pub(super) fn selector(
        selector: &Option<CudaDenseStorage>,
    ) -> Result<&CudaDenseStorage, Error> {
        selector.as_ref().ok_or_else(|| {
            internal_layout_error("a non-aligned factor route has no gauge selector")
        })
    }
}

#[cfg(all(test, feature = "cuda"))]
impl<D: CudaPayload> Drop for TypedCudaSvdScratch<D> {
    fn drop(&mut self) {
        update_cuda_svd_observation(|(results, total, creations, live, peak)| {
            (results, total, creations, live - 1, peak)
        });
    }
}

/// Validates the compact SVD diagonal factor against its routes and returns
/// each executed route's diagonal start: the packed `rank x rank` region's
/// offset, whose diagonal then has element stride `rank + 1`.
#[cfg(feature = "cuda")]
pub(super) fn validate_cuda_svd_middle_regions(
    plan: &CompactFactorPlan,
    middle_regions: &[CoupledSectorRegion],
) -> Result<Vec<usize>, Error> {
    let mut by_sector = HashMap::with_capacity(middle_regions.len());
    for region in middle_regions {
        if by_sector.insert(region.coupled(), region).is_some() {
            return Err(internal_layout_error(
                "compact SVD diagonal factor contains a duplicate coupled sector",
            ));
        }
    }
    let mut diagonals = Vec::with_capacity(middle_regions.len());
    for (route, _, _) in executed_routes(plan) {
        let rank = route.rank();
        let middle = by_sector.get(&route.sector()).ok_or_else(|| {
            internal_layout_error("compact SVD diagonal factor is missing a source sector")
        })?;
        let range_len = middle.range().len();
        let expected_len = rank.checked_mul(rank).ok_or_else(|| {
            internal_layout_error("compact SVD diagonal factor region length overflows")
        })?;
        if (middle.rows(), middle.cols()) != (rank, rank)
            || range_len != expected_len
            || !middle.has_aligned_diagonal()
        {
            return Err(internal_layout_error(
                "compact SVD diagonal factor region does not match its source route",
            ));
        }
        diagonals.push(middle.range().start);
    }
    if diagonals.len() != middle_regions.len() {
        return Err(internal_layout_error(
            "compact SVD diagonal factor does not have exactly one region per route",
        ));
    }
    Ok(diagonals)
}

#[cfg(feature = "cuda")]
pub(super) fn validate_cuda_reduction_placement(
    expected: Placement,
    lhs: Placement,
    rhs: Placement,
) -> Result<(), Error> {
    if lhs != expected || rhs != expected {
        return Err(Error::PlacementMismatch);
    }
    Ok(())
}

#[cfg(feature = "cuda")]
pub(super) fn download_cuda_reduction_partials<D: CudaPayload>(
    partials: &CudaStorage<D>,
    cuda: &CudaDenseContext,
) -> Result<Vec<D>, Error> {
    #[cfg(test)]
    let device_len = partials.len();
    #[cfg(test)]
    let observed_call = CUDA_REDUCTION_BUFFER_OBSERVATION.with(|observation| {
        observation.get().map(|(calls, _, _)| {
            let calls = calls + 1;
            observation.set(Some((calls, device_len, 0)));
            calls
        })
    });
    let values = partials.download(cuda)?;
    #[cfg(test)]
    if let Some(calls) = observed_call {
        CUDA_REDUCTION_BUFFER_OBSERVATION.with(|observation| {
            observation.set(Some((calls, device_len, values.len())));
        });
    }
    Ok(values)
}
