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
) -> Result<Vec<bool>, Error> {
    dense_cuda_hermitian_regions::<D>(cuda, source, regions).map_err(dense_err)
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

#[cfg(feature = "cuda")]
pub(super) fn compile_cuda_qr_plan<R>(
    space: &BoundDynamicFusionMapSpace<R>,
    source_regions: Arc<[CoupledSectorRegion]>,
) -> Result<TypedCudaQrPlan<R>, Error>
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64> + CheckedFusionAlgebra + SectorCodec,
{
    let source_space = space.space();
    let hom = source_space.homspace();
    let bond = SectorLeg::new(
        source_regions.iter().filter_map(|region| {
            let rank = region.rows().min(region.cols());
            (rank != 0).then_some((region.coupled(), rank))
        }),
        false,
    );
    let left_space = space.derive_from_final_homspace(FusionTreeHomSpace::new(
        FusionProductSpace::new(hom.codomain().legs().iter().cloned()),
        FusionProductSpace::new([bond.clone()]),
    ))?;
    let right_space = space.derive_from_final_homspace(FusionTreeHomSpace::new(
        FusionProductSpace::new([bond]),
        FusionProductSpace::new(hom.domain().legs().iter().cloned()),
    ))?;
    let left_regions = sector_regions(left_space.space().structure(), left_space.space().nout())?;
    let right_regions =
        sector_regions(right_space.space().structure(), right_space.space().nout())?;
    let index_by_sector = |regions: &[CoupledSectorRegion]| {
        let mut indices = HashMap::with_capacity(regions.len());
        for (index, region) in regions.iter().enumerate() {
            if indices.insert(region.coupled(), index).is_some() {
                return Err(internal_layout_error(
                    "compact QR factor contains a duplicate coupled sector",
                ));
            }
        }
        Ok(indices)
    };
    let left_by_sector = index_by_sector(&left_regions)?;
    let right_by_sector = index_by_sector(&right_regions)?;
    let mut routes = Vec::with_capacity(source_regions.len());
    for (source, region) in source_regions.iter().enumerate() {
        let rank = region.rows().min(region.cols());
        if rank == 0 {
            continue;
        }
        let left = *left_by_sector.get(&region.coupled()).ok_or_else(|| {
            internal_layout_error("compact QR left factor is missing a source sector")
        })?;
        let right = *right_by_sector.get(&region.coupled()).ok_or_else(|| {
            internal_layout_error("compact QR right factor is missing a source sector")
        })?;
        let left_region = &left_regions[left];
        let right_region = &right_regions[right];
        if (left_region.rows(), left_region.cols()) != (region.rows(), rank)
            || (right_region.rows(), right_region.cols()) != (rank, region.cols())
            || !cuda_qr_tree_extents_match(region.row_trees(), left_region.row_trees())?
            || !cuda_qr_tree_extents_match(region.col_trees(), right_region.col_trees())?
        {
            return Err(internal_layout_error(
                "compact QR factor region does not match its source route",
            ));
        }
        routes.push(TypedCudaQrRoute {
            source,
            left,
            right,
            rank,
            aligned_left: cuda_factor_layout_is_aligned(
                region.row_trees(),
                left_region.row_trees(),
                region.rows(),
            )?,
            aligned_right: cuda_factor_layout_is_aligned(
                region.col_trees(),
                right_region.col_trees(),
                region.cols(),
            )?,
        });
    }
    if routes.len() != left_regions.len() || routes.len() != right_regions.len() {
        return Err(internal_layout_error(
            "compact QR factor contains an unrouted coupled sector",
        ));
    }
    Ok(TypedCudaQrPlan {
        left_space,
        right_space,
        source_regions,
        left_regions,
        right_regions,
        routes,
    })
}

#[cfg(feature = "cuda")]
/// Admits the eigenvector and diagonal factor spaces for a bond of the given
/// per-sector `ranks` (`(sector, kept)`) and routes every coupled sector
/// from its source region to them. A function of the space alone, so the
/// eager call and a prepared handle compile the same plan.
pub(super) fn compile_cuda_eigh_plan<R>(
    space: &BoundDynamicFusionMapSpace<R>,
    source_regions: Arc<[CoupledSectorRegion]>,
    ranks: impl ExactSizeIterator<Item = (SectorId, usize)> + Clone,
) -> Result<TypedCudaEighPlan<R>, Error>
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64> + CheckedFusionAlgebra + SectorCodec,
{
    let source_space = space.space();
    let hom = source_space.homspace();
    let bond = SectorLeg::new(ranks.clone(), false);
    let left_space = space.derive_from_final_homspace(FusionTreeHomSpace::new(
        FusionProductSpace::new(hom.codomain().legs().iter().cloned()),
        FusionProductSpace::new([bond.clone()]),
    ))?;
    let middle_space = space.derive_from_final_homspace(FusionTreeHomSpace::new(
        FusionProductSpace::new([bond.clone()]),
        FusionProductSpace::new([bond]),
    ))?;
    let left_regions = sector_regions(left_space.space().structure(), left_space.space().nout())?;
    let middle_regions = sector_regions(
        middle_space.space().structure(),
        middle_space.space().nout(),
    )?;
    let index_by_sector = |regions: &[CoupledSectorRegion]| {
        let mut indices = HashMap::with_capacity(regions.len());
        for (index, region) in regions.iter().enumerate() {
            if indices.insert(region.coupled(), index).is_some() {
                return Err(internal_layout_error(
                    "CUDA EIGH factor contains a duplicate coupled sector",
                ));
            }
        }
        Ok(indices)
    };
    let source_by_sector = index_by_sector(&source_regions)?;
    let left_by_sector = index_by_sector(&left_regions)?;
    let middle_by_sector = index_by_sector(&middle_regions)?;
    let mut routes = Vec::with_capacity(ranks.len());
    for (sector, kept) in ranks {
        if kept == 0 {
            return Err(internal_layout_error(
                "CUDA EIGH plan retained an empty sector",
            ));
        }
        let source = *source_by_sector
            .get(&sector)
            .ok_or_else(|| internal_layout_error("CUDA EIGH factor is missing a source sector"))?;
        let source_region = &source_regions[source];
        let full_rank = source_region.rows().min(source_region.cols());
        if kept > full_rank {
            return Err(internal_layout_error(
                "CUDA EIGH rank exceeds its source route",
            ));
        }
        let left = *left_by_sector.get(&sector).ok_or_else(|| {
            internal_layout_error("CUDA EIGH eigenvector factor is missing a sector")
        })?;
        let middle = *middle_by_sector.get(&sector).ok_or_else(|| {
            internal_layout_error("CUDA EIGH diagonal factor is missing a sector")
        })?;
        let left_region = &left_regions[left];
        let middle_region = &middle_regions[middle];
        let middle_len = middle_region
            .range()
            .end
            .checked_sub(middle_region.range().start)
            .ok_or_else(|| internal_layout_error("CUDA EIGH diagonal range is invalid"))?;
        if (left_region.rows(), left_region.cols()) != (source_region.rows(), kept)
            || (middle_region.rows(), middle_region.cols()) != (kept, kept)
            || middle_len
                != kept
                    .checked_mul(kept)
                    .ok_or_else(|| internal_layout_error("CUDA EIGH diagonal length overflows"))?
            || !cuda_qr_tree_extents_match(source_region.row_trees(), left_region.row_trees())?
            || !cuda_qr_tree_extents_match(middle_region.row_trees(), middle_region.col_trees())?
        {
            return Err(internal_layout_error(
                "CUDA EIGH factor region does not match its source route",
            ));
        }
        routes.push(TypedCudaEighRoute {
            source,
            left,
            full_rank,
            kept,
            aligned: cuda_factor_layout_is_aligned(
                source_region.row_trees(),
                left_region.row_trees(),
                source_region.rows(),
            )?,
        });
    }
    if routes.len() != left_regions.len() || routes.len() != middle_regions.len() {
        return Err(internal_layout_error(
            "CUDA EIGH factor contains an unrouted coupled sector",
        ));
    }
    Ok(TypedCudaEighPlan {
        left_space,
        middle_space,
        source_regions,
        left_regions,
        routes,
    })
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

#[cfg(feature = "cuda")]
#[derive(Clone, Copy)]
pub(super) struct TypedCudaQrRoute {
    pub(super) source: usize,
    pub(super) left: usize,
    pub(super) right: usize,
    pub(super) rank: usize,
    /// The left factor's target region has the source's codomain tree layout,
    /// so it can be written by one whole-factor copy instead of a per-tree
    /// identity-selector GEMM. Proved at plan time, never a size heuristic.
    pub(super) aligned_left: bool,
    /// The same proof for the right factor's domain tree layout.
    pub(super) aligned_right: bool,
}

#[cfg(feature = "cuda")]
pub(super) struct TypedCudaQrPlan<R> {
    pub(super) left_space: BoundDynamicFusionMapSpace<R>,
    pub(super) right_space: BoundDynamicFusionMapSpace<R>,
    pub(super) source_regions: Arc<[CoupledSectorRegion]>,
    pub(super) left_regions: Arc<[CoupledSectorRegion]>,
    pub(super) right_regions: Arc<[CoupledSectorRegion]>,
    pub(super) routes: Vec<TypedCudaQrRoute>,
}

#[cfg(feature = "cuda")]
#[derive(Clone, Copy)]
pub(super) struct TypedCudaEighRoute {
    pub(super) source: usize,
    pub(super) left: usize,
    pub(super) full_rank: usize,
    pub(super) kept: usize,
    /// The eigenvector region has the source's codomain tree layout, so its
    /// column permutation is one whole-region GEMM instead of one per tree.
    pub(super) aligned: bool,
}

/// The eigenvector (`left`) and diagonal (`middle`) factor spaces and routes
/// of a device EIGH, for a bond of the given per-sector rank.
#[cfg(feature = "cuda")]
pub(super) struct TypedCudaEighPlan<R> {
    pub(super) left_space: BoundDynamicFusionMapSpace<R>,
    pub(super) middle_space: BoundDynamicFusionMapSpace<R>,
    pub(super) source_regions: Arc<[CoupledSectorRegion]>,
    pub(super) left_regions: Arc<[CoupledSectorRegion]>,
    pub(super) routes: Vec<TypedCudaEighRoute>,
}

/// Whether a factor region reproduces its source's tree layout exactly: the
/// same trees in the same order, with equal extents and equal offsets, tiling
/// `[0, extent)` without gaps.
///
/// `cuda_qr_tree_extents_match` is deliberately order-insensitive because the
/// general assembly never relies on enumeration order. This stricter predicate
/// is the proof that lets an aligned route skip the identity-selector GEMM.
#[cfg(feature = "cuda")]
pub(super) fn cuda_factor_layout_is_aligned(
    source: &[CoupledTreeExtent],
    factor: &[CoupledTreeExtent],
    extent: usize,
) -> Result<bool, Error> {
    if source.len() != factor.len() {
        return Ok(false);
    }
    let mut covered = 0usize;
    for (source_tree, factor_tree) in source.iter().zip(factor) {
        let size = source_tree.extent()?;
        if source_tree.tree() != factor_tree.tree()
            || size != factor_tree.extent()?
            || source_tree.offset() != covered
            || factor_tree.offset() != covered
        {
            return Ok(false);
        }
        covered += size;
    }
    Ok(covered == extent)
}

#[cfg(feature = "cuda")]
pub(super) fn cuda_qr_tree_extents_match(
    source: &[CoupledTreeExtent],
    factor: &[CoupledTreeExtent],
) -> Result<bool, Error> {
    if source.len() != factor.len() {
        return Ok(false);
    }
    let mut matched = vec![false; factor.len()];
    for source_tree in source {
        let source_extent = source_tree.extent()?;
        let mut match_index = None;
        for (index, factor_tree) in factor.iter().enumerate() {
            if !matched[index]
                && source_tree.tree() == factor_tree.tree()
                && source_extent == factor_tree.extent()?
            {
                match_index = Some(index);
                break;
            }
        }
        let Some(index) = match_index else {
            return Ok(false);
        };
        matched[index] = true;
    }
    Ok(matched.into_iter().all(|is_matched| is_matched))
}

/// Validates the compact SVD diagonal factor against its routes and returns
/// each route's diagonal start: the packed `rank x rank` region's offset,
/// whose diagonal then has element stride `rank + 1`.
#[cfg(feature = "cuda")]
pub(super) fn validate_cuda_svd_middle_regions<R>(
    plan: &TypedCudaQrPlan<R>,
    middle_regions: &[CoupledSectorRegion],
) -> Result<Vec<usize>, Error> {
    if middle_regions.len() != plan.routes.len() {
        return Err(internal_layout_error(
            "compact SVD diagonal factor does not have exactly one region per route",
        ));
    }
    let mut by_sector = HashMap::with_capacity(middle_regions.len());
    for region in middle_regions {
        if by_sector.insert(region.coupled(), region).is_some() {
            return Err(internal_layout_error(
                "compact SVD diagonal factor contains a duplicate coupled sector",
            ));
        }
    }
    let mut diagonals = Vec::with_capacity(plan.routes.len());
    for route in &plan.routes {
        let source = &plan.source_regions[route.source];
        let middle = by_sector.get(&source.coupled()).ok_or_else(|| {
            internal_layout_error("compact SVD diagonal factor is missing a source sector")
        })?;
        let range_len = middle
            .range()
            .end
            .checked_sub(middle.range().start)
            .ok_or_else(|| {
                internal_layout_error("compact SVD diagonal factor has an invalid region range")
            })?;
        let expected_len = route.rank.checked_mul(route.rank).ok_or_else(|| {
            internal_layout_error("compact SVD diagonal factor region length overflows")
        })?;
        if (middle.rows(), middle.cols()) != (route.rank, route.rank)
            || range_len != expected_len
            || !cuda_qr_tree_extents_match(middle.row_trees(), middle.col_trees())?
        {
            return Err(internal_layout_error(
                "compact SVD diagonal factor region does not match its source route",
            ));
        }
        diagonals.push(middle.range().start);
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
