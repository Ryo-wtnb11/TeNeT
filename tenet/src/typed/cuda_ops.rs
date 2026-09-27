#[allow(unused_imports)]
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
    /// use tenet::core::U1FusionRule;
    /// use tenet::typed::TensorMap;
    ///
    /// fn c32_upload(tensor: &TensorMap<U1FusionRule, Complex32>) {
    ///     let _ = tensor.to_cuda();
    /// }
    /// ```
    ///
    /// ```
    /// use tenet::core::U1FusionRule;
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
                    let dense = tenet_matrixalgebra::diagonal_bond_data(
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
    /// use tenet::core::U1FusionRule;
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
/// Device compact SVD (`svd_compact`) and Hermitian eigendecomposition
/// (`eigh_full`) over either device payload (`f64` or `Complex64`).
///
/// A truncated factorization is a composition, and its decision over
/// quantum-dimension-weighted spectra across all coupled sectors belongs on
/// the host; placement stays the caller's explicit choice:
///
/// ```no_run
/// use std::sync::Arc;
///
/// use tenet::core::U1Irrep;
/// use tenet::prelude::{Runtime, Svd, TensorMap, Truncation, U1FusionRule};
/// use tenet::typed::GradedSpace;
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let runtime = Runtime::builder().cuda(0).build()?;
/// let leg = GradedSpace::try_new(Arc::new(U1FusionRule), [(U1Irrep::new(0), 4)])?;
/// let device = TensorMap::<_, f64>::from_subblock_fn(&runtime, [&leg], [&leg], |_, index| {
///     (index[0] + 2 * index[1]) as f64
/// })?
/// .to_cuda()?;
///
/// let Svd { u, s, vh } = device.svd_compact(&[0], &[1])?;
/// // Until a device `restrict_leg` lands, the factors move to the host once.
/// let (u, s, vh) = (u.to_host()?, s.to_host()?, vh.to_host()?);
/// let found = s.domain()[0].find_truncated(&s.diagview()?, &Truncation::rank(2))?;
/// let u = u.restrict_leg(&[(u.codomain_rank(), &found.selection)])?;
/// let s = s.restrict_leg(&[(0, &found.selection), (1, &found.selection)])?;
/// let vh = vh.restrict_leg(&[(0, &found.selection)])?;
/// # let _ = (u, s, vh, found.error);
/// # Ok(())
/// # }
/// ```
///
/// The `d` and `v` of `eigh_full` are truncated the same way.
///
/// `u` and `vh` follow the Host largest-pivot gauge for every payload, as
/// TensorKit's (MatrixAlgebraKit `gaugefix!`) does. `eigh_full`
/// admits a block only when it equals its *conjugate* transpose, so a
/// complex-symmetric non-Hermitian block is rejected before any
/// factorization.
///
/// Compact QR is not here: it is `f64`-only and lives in its own impl below.
///
/// Checked Generic providers deliberately have no device factorizations, and
/// compact SVD has the same deliberately narrow typed CUDA surface:
///
/// ```compile_fail
/// use tenet::core::{
///     CheckedGenericAdmissionMode, CheckedGenericFusion, CheckedGenericRigidSymbols,
///     TypedSectorAdmission,
/// };
/// use tenet::typed::{CudaStorage, TensorMap};
///
/// fn no_checked_generic_cuda_svd<R>(tensor: &TensorMap<R, f64, CudaStorage>)
/// where
///     R: TypedSectorAdmission<Mode = CheckedGenericAdmissionMode>
///         + CheckedGenericFusion
///         + CheckedGenericRigidSymbols<Scalar = f64>,
/// {
///     let _ = tensor.svd_compact(&[0], &[1]);
/// }
/// ```
///
/// The device payload marker alone does not admit a device factorization, and
/// neither does it together with the *host* factorization marker: the gate is
/// [`CudaFactorizationPayload`]. Every payload implements it since #1341, but
/// a *generic* body must still name it — the two bounds below are not enough,
/// because neither says anything about the device gauge or the device
/// Hermitian rule.
///
/// ```compile_fail
/// use tenet::prelude::U1FusionRule;
/// use tenet::typed::{CudaPayload, CudaStorage, TensorMap};
///
/// fn device_payload_only<D: CudaPayload>(tensor: &TensorMap<U1FusionRule, D, CudaStorage<D>>) {
///     let _ = tensor.svd_compact(&[0], &[1]);
/// }
/// ```
///
/// ```compile_fail
/// use tenet::prelude::{FactorizationScalar, U1FusionRule};
/// use tenet::typed::{CudaPayload, CudaStorage, TensorMap};
///
/// fn device_payload_and_host_factorizing<D: CudaPayload + FactorizationScalar>(
///     tensor: &TensorMap<U1FusionRule, D, CudaStorage<D>>,
/// ) {
///     let _ = tensor.svd_compact(&[0], &[1]);
/// }
/// ```
///
/// ```
/// use tenet::prelude::U1FusionRule;
/// use tenet::typed::{CudaFactorizationPayload, CudaStorage, TensorMap};
///
/// fn device_factorizing<D: CudaFactorizationPayload>(
///     tensor: &TensorMap<U1FusionRule, D, CudaStorage<D>>,
/// ) {
///     let _ = tensor.svd_compact(&[0], &[1]);
/// }
/// ```
///
/// Concretely, at all four device payloads — the single-precision twins of the
/// pins #1336 left here, which differ only in the dtype:
///
/// ```
/// use tenet::prelude::U1FusionRule;
/// use tenet::typed::{CudaStorage, TensorMap};
///
/// fn f32_device_svd(tensor: &TensorMap<U1FusionRule, f32, CudaStorage<f32>>) {
///     let _ = tensor.svd_compact(&[0], &[1]);
/// }
///
/// fn f64_device_svd(tensor: &TensorMap<U1FusionRule, f64, CudaStorage<f64>>) {
///     let _ = tensor.svd_compact(&[0], &[1]);
/// }
/// ```
///
/// ```
/// use num_complex::{Complex32, Complex64};
/// use tenet::prelude::U1FusionRule;
/// use tenet::typed::{CudaStorage, TensorMap};
///
/// fn c32_device_eigh(tensor: &TensorMap<U1FusionRule, Complex32, CudaStorage<Complex32>>) {
///     let _ = tensor.eigh_full(&[0], &[1]);
/// }
///
/// fn c64_device_eigh(tensor: &TensorMap<U1FusionRule, Complex64, CudaStorage<Complex64>>) {
///     let _ = tensor.eigh_full(&[0], &[1]);
/// }
/// ```
impl<R, D> TensorMap<R, D, CudaStorage<D>>
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64> + CheckedFusionAlgebra + SectorCodec,
    D: CudaFactorizationPayload,
{
    /// The `k x k` identity selector the non-aligned assembly GEMMs need.
    /// A fully aligned route assembles by copy and uploads nothing.
    fn identity_selector(
        cuda: &mut CudaDenseContext,
        route: &TypedCudaQrRoute,
    ) -> Result<Option<CudaStorage<D>>, Error> {
        if route.aligned_left && route.aligned_right {
            return Ok(None);
        }
        upload_selector(
            cuda,
            route.rank,
            route.rank,
            (0..route.rank).map(|index| (index, index, D::ONE)),
        )
        .map(Some)
    }

    fn route_selector(selector: &Option<CudaStorage<D>>) -> Result<&CudaStorage<D>, Error> {
        selector.as_ref().ok_or_else(|| {
            internal_layout_error("a non-aligned factor route has no assembly selector")
        })
    }

    pub(super) fn compile_cuda_qr_plan(
        &self,
        source_regions: Arc<[CoupledSectorRegion]>,
    ) -> Result<TypedCudaQrPlan<R>, Error> {
        compile_cuda_qr_plan(self.logical_space(), source_regions)
    }

    /// Admits the eigenvector and diagonal factor spaces for `spectra` and
    /// routes every coupled sector from its source region to them.
    fn compile_cuda_eigh_plan(
        &self,
        source_regions: Arc<[CoupledSectorRegion]>,
        spectra: &[tenet_matrixalgebra::SectorSpectrum<f64>],
    ) -> Result<TypedCudaEighPlan<R>, Error> {
        compile_cuda_eigh_plan(
            self.logical_space(),
            source_regions,
            spectra
                .iter()
                .map(|entry| (entry.sector, entry.values.len())),
        )
    }

    /// Streamed compact SVD of owned dense CUDA storage.
    ///
    /// Each nonempty coupled-sector route is decomposed and assembled before
    /// its raw device factors are dropped. The returned `s` is deliberately
    /// a dense CUDA tensor because CUDA diagonal storage is not part of the
    /// typed storage contract; each route's singular values are copied into
    /// its diagonal on the device (element stride `k + 1`).
    ///
    /// Transfers, exactly: host to device, one zero upload per factor
    /// (`u`, `s`, `vh`: the dense output sizes, `s` being `Σ_c k_c²`
    /// elements, since a host zero buffer is the only device allocation
    /// path until #740) plus one upload of `max_c rows_c` `i64` gauge weights
    /// when any route is nonempty. Device to host, no tensor payload:
    /// the singular values never cross to the host, so the call ends
    /// without a spectrum download. The backend still reads O(1)
    /// solver-status metadata per route, a host barrier each.
    ///
    /// `u` and `vh` carry the Host gauge: in every column of each sector's
    /// `u`, the first largest-magnitude entry is real and non-negative, and
    /// the matching `vh` row takes the inverse phase. The fix-up runs on the
    /// device without a download (see [`tenet_dense::cuda_svd_gauge_phases`]):
    /// 13 Tenferro ops per route for the phases, then per side either a
    /// `k x k` diagonal selector that the non-aligned assembly GEMM
    /// multiplies by anyway (1 op) or a scaling of the aligned factor before
    /// its copy (2 ops), plus one `conj` for a complex left side. They are
    /// counted in [`tenet_dense::CudaTransferStats::gauge_ops`].
    ///
    /// `rows` and `cols` are the leg roles, as for the Host operation: it
    /// acts on the matrix view `self.permute(rows, cols)` (one device
    /// permute), and the current split borrows `self` without a transform.
    pub fn svd_compact(&self, rows: &[usize], cols: &[usize]) -> Result<Svd<Self>, Error> {
        self.with_cuda_leg_roles(rows, cols, Self::svd_compact_matrix)
    }

    fn svd_compact_matrix(&self) -> Result<Svd<Self>, Error> {
        let source = self.direct_cuda_storage("svd_compact")?;
        let source_space = self.logical_space().space();
        let required_len = source_space.required_len()?;
        let source_regions = sector_regions(source_space.structure(), source_space.nout())?;

        {
            // Preflight only: the ordinal is immutable, so this placement
            // check takes no device lock at all.
            let device = self.runtime.cuda_device_ordinal_checked()?;
            Self::validate_cuda_owned_metadata(
                Placement::Cuda(device),
                source.placement(),
                required_len,
                source.len(),
            )?;
        }
        // As for typed CUDA QR, all provider work and final-space admission
        // complete before the execution lock and before any output exists.
        let plan = self.compile_cuda_qr_plan(source_regions)?;
        #[cfg(test)]
        let plan = {
            let mut plan = plan;
            if CUDA_SVD_TREEWISE.with(std::cell::Cell::get) {
                for route in &mut plan.routes {
                    route.aligned_left = false;
                    route.aligned_right = false;
                }
            }
            plan
        };
        let bond = plan.left_space.space().homspace().domain().legs()[0].clone();
        let middle_space =
            self.logical_space()
                .derive_from_final_homspace(FusionTreeHomSpace::new(
                    FusionProductSpace::new([bond.clone()]),
                    FusionProductSpace::new([bond]),
                ))?;
        let middle_regions = sector_regions(
            middle_space.space().structure(),
            middle_space.space().nout(),
        )?;
        let diagonals = validate_cuda_svd_middle_regions(&plan, &middle_regions)?;
        let left_len = plan.left_space.space().required_len()?;
        let middle_len = middle_space.space().required_len()?;
        let right_len = plan.right_space.space().required_len()?;
        let (left_data, middle_data, right_data) = {
            let mut lease = self.runtime.lease_cuda()?;
            let cuda = &mut *lease;
            let mut left_data = CudaStorage::upload_owned(cuda, vec![D::ZERO; left_len])?;
            #[cfg(test)]
            observe_cuda_svd_final_storage_creation();
            let mut right_data = CudaStorage::upload_owned(cuda, vec![D::ZERO; right_len])?;
            #[cfg(test)]
            observe_cuda_svd_final_storage_creation();
            // The zero upload is the only device allocation path until #740;
            // every value then arrives by a device copy.
            let mut middle_data = CudaStorage::upload_owned(cuda, vec![D::ZERO; middle_len])?;
            #[cfg(test)]
            observe_cuda_svd_final_storage_creation();

            let max_rows = plan
                .routes
                .iter()
                .map(|route| plan.source_regions[route.source].rows())
                .max();
            let weights = max_rows
                .map(|rows| CudaSvdGaugeWeights::upload(cuda, rows))
                .transpose()
                .map_err(dense_err)?;
            for (route, &diagonal) in plan.routes.iter().zip(&diagonals) {
                let source_region = &plan.source_regions[route.source];
                let left_region = &plan.left_regions[route.left];
                let right_region = &plan.right_regions[route.right];
                let (raw_left, spectrum, raw_right) = cuda_svd_region::<D>(
                    cuda,
                    &source.0,
                    source_region.range().start,
                    source_region.rows(),
                    source_region.cols(),
                )?;
                if spectrum.len() != route.rank {
                    return Err(internal_layout_error(
                        "compact SVD spectrum length does not match its source route",
                    ));
                }
                let weights = weights.as_ref().ok_or_else(|| {
                    internal_layout_error("a compact SVD route has no gauge weights")
                })?;
                // The Host gauge (first largest-|U| entry of each column real
                // and non-negative) is folded into the selector a non-aligned
                // side multiplies by anyway; only an aligned side, which is a
                // plain copy, pays a separate scaling pass.
                let phases = cuda_svd_gauge_phases::<D>(
                    cuda,
                    &raw_left,
                    source_region.rows(),
                    route.rank,
                    weights,
                )
                .map_err(dense_err)?;
                let (left, left_selector) = if route.aligned_left {
                    let left = phases
                        .scale_left::<D>(cuda, &raw_left, source_region.rows())
                        .map_err(dense_err)?;
                    (left, None)
                } else {
                    let selector = phases.left_selector::<D>(cuda).map_err(dense_err)?;
                    (raw_left, Some(selector))
                };
                let (right, right_selector) = if route.aligned_right {
                    let right = phases
                        .scale_right::<D>(cuda, &raw_right, source_region.cols())
                        .map_err(dense_err)?;
                    (right, None)
                } else {
                    let selector = phases.right_selector::<D>(cuda).map_err(dense_err)?;
                    (raw_right, Some(selector))
                };
                drop(phases);
                // The scratch owns all route-local allocations. It is dropped
                // at the end of this iteration, bounding peak raw-factor
                // storage independently of the number of sectors.
                let scratch =
                    TypedCudaSvdScratch::<D>::new(left, right, left_selector, right_selector);
                // Compact SVD keeps the full rank by construction, so the same
                // proved layout identity applies as for QR.
                if route.aligned_left {
                    copy_whole_factor(cuda, &mut left_data, left_region, &scratch.left)?;
                } else {
                    assemble_left_factor(
                        cuda,
                        &mut left_data,
                        left_region,
                        source_region,
                        &scratch.left,
                        route.rank,
                        TypedCudaSvdScratch::<D>::selector(&scratch.left_selector)?,
                        0,
                        route.rank,
                    )?;
                }
                if route.aligned_right {
                    copy_whole_factor(cuda, &mut right_data, right_region, &scratch.right)?;
                } else {
                    assemble_right_factor(
                        cuda,
                        &mut right_data,
                        right_region,
                        source_region,
                        TypedCudaSvdScratch::<D>::selector(&scratch.right_selector)?,
                        route.rank,
                        route.rank,
                        &scratch.right,
                    )?;
                }
                tenet_dense::cuda_copy_spectrum_into::<D>(
                    cuda,
                    spectrum,
                    &mut middle_data.0,
                    diagonal,
                    route.rank + 1,
                )
                .map_err(dense_err)?;
            }
            (left_data, middle_data, right_data)
        };

        Ok(Svd {
            u: Self {
                runtime: self.runtime.clone(),
                repr: owned_repr(TypedTensorBody::dense(plan.left_space, left_data)),
            },
            s: Self {
                runtime: self.runtime.clone(),
                repr: owned_repr(TypedTensorBody::dense(middle_space, middle_data)),
            },
            vh: Self {
                runtime: self.runtime.clone(),
                repr: owned_repr(TypedTensorBody::dense(plan.right_space, right_data)),
            },
        })
    }

    /// Hermitian eigendecomposition of an owned dense CUDA endomorphism.
    ///
    /// Returns an [`Eigh`] with `self = v * d * v.adjoint()`. Both factors remain
    /// on the source device. A lazy-adjoint receiver is rejected explicitly;
    /// no receiver-sized payload is downloaded or materialized.
    ///
    /// No truncation decision is made: every eigenpair is kept. Eigenvalues are
    /// the only numerical payload that crosses to the host, where they are
    /// sorted by descending `|λ|`; the eigenvectors stay on the device and
    /// their columns are gathered into that order by the assembly selector.
    ///
    /// Transfers, exactly:
    ///
    /// - Hermiticity admission (`cuda_hermitian_regions`): at most three
    ///   downloads of gathered real scalars, one per stage that has an
    ///   undecided sector (per-sector maxima, then sums of squares and
    ///   residual maxima, then residual sums of squares; O(1) values per
    ///   sector), and one rank-0 real normalizer upload per sector entering
    ///   stage 2 (nonzero, finite input maximum) and per sector entering
    ///   stage 3 (nonzero, finite residual maximum): at most two per
    ///   sector.
    /// - Device to host: all sectors' eigenvalues (`Σ_c n_c` real values) in
    ///   one download, which the non-finite check, the host `|λ|` order and
    ///   the factor-space plan consume.
    /// - Host to device: `d` as one dense upload of `Σ_c n_c²` elements with
    ///   the sorted eigenvalues already on its diagonal, one zero upload for
    ///   `v`, and at most one packed selector upload.
    ///
    /// Why `d` is filled on the host rather than scattered on the
    /// device: the only device allocation path is a zero upload of the same
    /// `Σ_c n_c²` elements (#740), so a device scatter would add a
    /// `Σ_c n_c` upload and move nothing less.
    ///
    /// `rows` and `cols` are the leg roles, as for the Host operation: it
    /// acts on the matrix view `self.permute(rows, cols)` (one device
    /// permute), and the current split borrows `self` without a transform.
    pub fn eigh_full(&self, rows: &[usize], cols: &[usize]) -> Result<Eigh<Self>, Error> {
        self.with_cuda_leg_roles(rows, cols, Self::eigh_full_matrix)
    }

    fn eigh_full_matrix(&self) -> Result<Eigh<Self>, Error> {
        let source = self.direct_cuda_storage("eigh_full")?;
        let source_space = self.logical_space().space();
        if source_space.homspace().codomain() != source_space.homspace().domain() {
            return Err(
                tenet_tensors::OperationError::UnsupportedTensorContractScope {
                    message: "eigh requires an endomorphism (codomain == domain)",
                }
                .into(),
            );
        }
        let required_len = source_space.required_len()?;
        let source_regions = sector_regions(source_space.structure(), source_space.nout())?;
        if source_regions
            .iter()
            .any(|region| region.rows() != region.cols())
        {
            return Err(internal_layout_error(
                "CUDA EIGH source contains a non-square coupled-sector region",
            ));
        }
        tenet_matrixalgebra::validate_endomorphism_region_stacking(
            &source_regions,
            tenet_matrixalgebra::EIGH_FULL_STACKING,
        )?;

        {
            // Preflight only: the ordinal is immutable, so this placement
            // check takes no device lock at all.
            let device = self.runtime.cuda_device_ordinal_checked()?;
            Self::validate_cuda_owned_metadata(
                Placement::Cuda(device),
                source.placement(),
                required_len,
                source.len(),
            )?;
        }

        // The existing compact factor plan is the canonical source -> left
        // factor route. EIGH needs that left route only; no new plan hierarchy.
        let source_plan = self.compile_cuda_qr_plan(source_regions)?;

        // Admission is complete. Validate every block before the first EIGH so
        // a late non-Hermitian sector cannot trigger partial numerical work.
        {
            let mut lease = self.runtime.lease_cuda()?;
            let cuda = &mut *lease;
            let regions: Vec<_> = source_plan
                .source_regions
                .iter()
                .map(|region| (region.range().start, region.rows()))
                .collect();
            if cuda_hermitian_regions::<D>(cuda, &source.0, &regions)?.contains(&false) {
                return Err(
                    tenet_tensors::OperationError::UnsupportedTensorContractScope {
                        message: "eigh requires every coupled-sector block to be Hermitian",
                    }
                    .into(),
                );
            }
        }

        let (spectra, mut raw_vectors, orders) = {
            let mut lease = self.runtime.lease_cuda()?;
            let cuda = &mut *lease;
            let mut device_spectra = Vec::with_capacity(source_plan.source_regions.len());
            let mut vectors = Vec::with_capacity(source_plan.source_regions.len());
            #[cfg(test)]
            let mut decomposition_ordinal = 0;
            for region in source_plan.source_regions.iter() {
                let n = region.rows();
                if n == 0 {
                    vectors.push(None);
                    continue;
                }
                let (values, vector) =
                    typed_cuda_eigh_region::<D>(cuda, &source.0, region.range().start, n)?;
                #[cfg(test)]
                {
                    decomposition_ordinal += 1;
                    Self::inject_cuda_eigh_failure("decomposition", decomposition_ordinal)?;
                }
                device_spectra.push(values);
                vectors.push(Some(vector));
            }
            let mut downloaded = cuda_download_spectra::<D>(cuda, &device_spectra)?.into_iter();
            let mut spectra = Vec::with_capacity(source_plan.source_regions.len());
            let mut orders = Vec::with_capacity(source_plan.source_regions.len());
            for region in source_plan.source_regions.iter() {
                let n = region.rows();
                let values = if n == 0 {
                    Vec::new()
                } else {
                    downloaded.next().ok_or_else(|| {
                        internal_layout_error("CUDA EIGH spectrum count does not match its blocks")
                    })?
                };
                if values.iter().any(|value| !value.is_finite()) {
                    return Err(internal_layout_error(
                        "CUDA EIGH returned a non-finite eigenvalue",
                    ));
                }
                let mut order: Vec<_> = (0..n).collect();
                order.sort_by(|&left, &right| {
                    values[right]
                        .abs()
                        .total_cmp(&values[left].abs())
                        .then(left.cmp(&right))
                });
                let sorted = order.iter().map(|&index| values[index]).collect();
                spectra.push(tenet_matrixalgebra::SectorSpectrum {
                    sector: region.coupled(),
                    values: sorted,
                });
                orders.push(order);
            }
            (spectra, vectors, orders)
        };

        // Every factor-space admission is provider work on the host: no CUDA
        // lease is held from here until the assembly below.
        let plan = self.compile_cuda_eigh_plan(source_plan.source_regions, &spectra)?;
        let vector_len = plan.left_space.space().required_len()?;
        let diagonal_len = plan.middle_space.space().required_len()?;
        let mut diagonal_host = vec![D::ZERO; diagonal_len];
        fill_diagonal_values(
            plan.middle_space.space().structure(),
            &mut diagonal_host,
            &spectra,
        )?;

        let (diagonal_data, vector_data) = {
            let mut lease = self.runtime.lease_cuda()?;
            let cuda = &mut *lease;
            let diagonal_data = CudaStorage::upload_owned(cuda, diagonal_host)?;
            let mut vector_data = CudaStorage::upload_owned(cuda, vec![D::ZERO; vector_len])?;
            // Every route's `full_rank x kept` column selector, packed back
            // to back so the call makes one upload whatever its route count.
            let mut selector_offsets = Vec::with_capacity(plan.routes.len());
            let mut selector_len = 0usize;
            for route in plan.routes.iter() {
                if route.kept > orders[route.source].len() {
                    return Err(internal_layout_error(
                        "CUDA EIGH rank exceeds its eigenvector order",
                    ));
                }
                selector_offsets.push(selector_len);
                selector_len += route.full_rank * route.kept;
            }
            let selector = if selector_len == 0 {
                None
            } else {
                Some(upload_selector(
                    cuda,
                    selector_len,
                    1,
                    plan.routes
                        .iter()
                        .zip(&selector_offsets)
                        .flat_map(|(route, &offset)| {
                            orders[route.source][..route.kept].iter().enumerate().map(
                                move |(column, &row)| {
                                    (offset + row + route.full_rank * column, 0, D::ONE)
                                },
                            )
                        }),
                )?)
            };
            #[cfg(test)]
            if selector.is_some() {
                CUDA_EIGH_SELECTOR_UPLOADS.with(|uploads| {
                    if let Some(count) = uploads.get() {
                        uploads.set(Some(count + 1));
                    }
                });
            }
            #[cfg(test)]
            let mut assembly_ordinal = 0;
            for (route, &selector_offset) in plan.routes.iter().zip(&selector_offsets) {
                #[cfg(test)]
                {
                    assembly_ordinal += 1;
                    Self::inject_cuda_eigh_failure("assembly", assembly_ordinal)?;
                }
                let selector = selector
                    .as_ref()
                    .ok_or_else(|| internal_layout_error("CUDA EIGH route has no selector"))?;
                let raw = raw_vectors[route.source]
                    .take()
                    .ok_or_else(|| internal_layout_error("CUDA EIGH route has no eigenvectors"))?;
                let left_region = &plan.left_regions[route.left];
                #[cfg(test)]
                let aligned = route.aligned && !CUDA_EIGH_TREEWISE.with(std::cell::Cell::get);
                #[cfg(not(test))]
                let aligned = route.aligned;
                if aligned {
                    assemble_aligned_left_factor(
                        cuda,
                        &mut vector_data,
                        left_region,
                        &raw,
                        route.full_rank,
                        selector,
                        selector_offset,
                        route.kept,
                    )?;
                } else {
                    assemble_left_factor(
                        cuda,
                        &mut vector_data,
                        left_region,
                        &plan.source_regions[route.source],
                        &raw,
                        route.full_rank,
                        &selector.0,
                        selector_offset,
                        route.kept,
                    )?;
                }
            }
            (diagonal_data, vector_data)
        };

        Ok(Eigh {
            d: Self {
                runtime: self.runtime.clone(),
                repr: owned_repr(TypedTensorBody::dense(plan.middle_space, diagonal_data)),
            },
            v: Self {
                runtime: self.runtime.clone(),
                repr: owned_repr(TypedTensorBody::dense(plan.left_space, vector_data)),
            },
        })
    }

    #[cfg(test)]
    fn inject_cuda_eigh_failure(stage: &'static str, ordinal: usize) -> Result<(), Error> {
        if CUDA_EIGH_FAILURE.with(|failure| failure.get()) == Some((stage, ordinal)) {
            return Err(Error::InvalidArgument(format!(
                "injected CUDA EIGH {stage} failure"
            )));
        }
        Ok(())
    }
}

#[cfg(feature = "cuda")]
/// Device compact QR, every device factorization payload (`f64`, `f32`,
/// `Complex64`, `Complex32`).
///
/// `qr_compact` returns the Host gauge exactly: positive diagonal (`R_jj` real
/// and non-negative, phase 1 kept where `R_jj == 0`, MatrixAlgebraKit
/// `positive = true`), applied on device by the backend's own QR primitive
/// (`QrGauge::PositiveDiagonal`: `Q -> Q D`, `R -> D^H R` with
/// `D = diag(R_jj / |R_jj|)`) rather than re-derived here.
///
/// The payload dtype is the only degree of freedom; the plan, the routes and
/// the assembly are the ones the device SVD shares.
///
/// Checked Generic providers have no device QR:
///
/// ```compile_fail
/// use tenet::core::{
///     CheckedGenericAdmissionMode, CheckedGenericFusion, CheckedGenericRigidSymbols,
///     TypedSectorAdmission,
/// };
/// use tenet::typed::{CudaStorage, TensorMap};
///
/// fn no_checked_generic_cuda_qr<R>(tensor: &TensorMap<R, f64, CudaStorage>)
/// where
///     R: TypedSectorAdmission<Mode = CheckedGenericAdmissionMode>
///         + CheckedGenericFusion
///         + CheckedGenericRigidSymbols<Scalar = f64>,
/// {
///     let _ = tensor.qr_compact(&[0], &[1]);
/// }
/// ```
///
/// ```
/// use tenet::prelude::U1FusionRule;
/// use tenet::typed::{CudaFactorizationPayload, CudaStorage, TensorMap};
///
/// fn device_qr<D: CudaFactorizationPayload>(
///     tensor: &TensorMap<U1FusionRule, D, CudaStorage<D>>,
/// ) {
///     let _ = tensor.qr_compact(&[0], &[1]);
/// }
/// ```
impl<R, D> TensorMap<R, D, CudaStorage<D>>
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64> + CheckedFusionAlgebra + SectorCodec,
    D: CudaFactorizationPayload,
{
    /// Streamed compact QR of owned dense CUDA storage.
    ///
    /// Each nonempty coupled sector is gauge-fixed on device by the backend's
    /// positive-diagonal QR: `R_jj` is real and non-negative, and a zero
    /// diagonal entry keeps phase 1. Nothing of either factor crosses to the
    /// host. A route whose factor region reproduces the source's tree layout
    /// is assembled by one whole-factor device copy; any other route keeps the
    /// per-tree identity-selector GEMM.
    ///
    /// `rows` and `cols` are the leg roles, as for the Host operation: it
    /// acts on the matrix view `self.permute(rows, cols)` (one device
    /// permute), and the current split borrows `self` without a transform.
    pub fn qr_compact(&self, rows: &[usize], cols: &[usize]) -> Result<Qr<Self>, Error> {
        self.with_cuda_leg_roles(rows, cols, Self::qr_compact_matrix)
    }

    fn qr_compact_matrix(&self) -> Result<Qr<Self>, Error> {
        let source = self.direct_cuda_storage("qr_compact")?;
        let source_space = self.logical_space().space();
        let required_len = source_space.required_len()?;
        let source_regions = sector_regions(source_space.structure(), source_space.nout())?;

        {
            // Preflight only: the ordinal is immutable, so this placement
            // check takes no device lock at all.
            let device = self.runtime.cuda_device_ordinal_checked()?;
            Self::validate_cuda_owned_metadata(
                Placement::Cuda(device),
                source.placement(),
                required_len,
                source.len(),
            )?;
        }

        // Provider queries and final HomSpace admission belong outside the
        // execution lock; the plan owns every source-to-factor route.
        let plan = self.compile_cuda_qr_plan(source_regions)?;
        let left_len = plan.left_space.space().required_len()?;
        let right_len = plan.right_space.space().required_len()?;

        let (left_data, right_data) = {
            let mut lease = self.runtime.lease_cuda()?;
            let cuda = &mut *lease;
            let mut left_data = CudaStorage::upload_owned(cuda, vec![D::ZERO; left_len])?;
            #[cfg(test)]
            observe_cuda_qr_output_upload();
            let mut right_data = CudaStorage::upload_owned(cuda, vec![D::ZERO; right_len])?;
            #[cfg(test)]
            observe_cuda_qr_output_upload();
            for route in &plan.routes {
                let source_region = &plan.source_regions[route.source];
                let left_region = &plan.left_regions[route.left];
                let right_region = &plan.right_regions[route.right];
                let (raw_left, raw_right) = cuda_qr_region::<D>(
                    cuda,
                    &source.0,
                    source_region.range().start,
                    source_region.rows(),
                    source_region.cols(),
                )?;
                let selector = Self::identity_selector(cuda, route)?;
                let scratch = TypedCudaQrScratch::new(raw_left, raw_right, selector);
                if route.aligned_left {
                    copy_whole_factor(cuda, &mut left_data, left_region, &scratch.left)?;
                } else {
                    assemble_left_factor(
                        cuda,
                        &mut left_data,
                        left_region,
                        source_region,
                        &scratch.left,
                        route.rank,
                        &Self::route_selector(&scratch.selector)?.0,
                        0,
                        route.rank,
                    )?;
                }
                if route.aligned_right {
                    copy_whole_factor(cuda, &mut right_data, right_region, &scratch.right)?;
                } else {
                    assemble_right_factor(
                        cuda,
                        &mut right_data,
                        right_region,
                        source_region,
                        &Self::route_selector(&scratch.selector)?.0,
                        route.rank,
                        route.rank,
                        &scratch.right,
                    )?;
                }
            }
            (left_data, right_data)
        };

        Ok(Qr {
            q: Self {
                runtime: self.runtime.clone(),
                repr: owned_repr(TypedTensorBody::dense(plan.left_space, left_data)),
            },
            r: Self {
                runtime: self.runtime.clone(),
                repr: owned_repr(TypedTensorBody::dense(plan.right_space, right_data)),
            },
        })
    }
}

#[cfg(feature = "cuda")]
/// A device [`TensorMap::trace_pairs`] validated and compiled on the Host,
/// not yet executed: executing it is the only step that touches the device.
#[doc(hidden)]
pub struct CudaTracePairs<'a, R, D: CudaPayload> {
    runtime: &'a Runtime,
    source_space: &'a BoundDynamicFusionMapSpace<R>,
    source: &'a CudaStorage<D>,
    space: BoundDynamicFusionMapSpace<R>,
    structure: tenet_tensors::TensorTraceFusionStructure<f64>,
}

#[cfg(feature = "cuda")]
impl<R, D> CudaTracePairs<'_, R, D>
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64> + CheckedFusionAlgebra + SectorCodec,
    D: CudaPayload,
{
    /// Runs the compiled trace: the #740 output upload, then one accumulating
    /// contraction per term, under one device lease.
    pub fn execute(self) -> Result<TensorMap<R, D, CudaStorage<D>>, Error> {
        let required_len = self.space.space().required_len()?;
        let mut lease = self.runtime.lease_cuda()?;
        let (cuda, transforms) = lease.split();
        // ponytail: #740 — the device seam initializes an output by uploading
        // zeros; replace only with a measured native allocation. The zeros are
        // also what the accumulating replay starts from.
        let mut output = CudaStorage::upload_owned(cuda, vec![D::from_real(0.0); required_len])?;
        tenet_tensors::tensortrace_fusion_structure_into_on_cuda(
            cuda,
            transforms,
            &self.structure,
            self.space.space().structure(),
            &mut output,
            self.source_space.space().structure(),
            self.source,
            D::from_real(1.0),
            D::from_real(1.0),
        )?;
        drop(lease);
        Ok(TensorMap {
            runtime: self.runtime.clone(),
            repr: owned_repr(TypedTensorBody::dense(self.space, output)),
        })
    }
}

#[cfg(feature = "cuda")]
/// Device transfer, arithmetic, reductions, and contraction over a
/// [`CudaStorage`] payload.
///
/// The payload dtype is the only degree of freedom: structural coefficients
/// stay real (`R::Scalar = f64`), and operand conjugation is carried as a GEMM
/// flag, never as a materialized conjugated buffer. All four payload dtypes are
/// admitted (#1336); see [`CudaPayload`].
///
/// Checked Generic providers deliberately have no device execution methods in
/// this leaf:
///
/// ```compile_fail
/// use tenet::core::{
///     CheckedGenericAdmissionMode, CheckedGenericFusion, CheckedGenericRigidSymbols,
///     TypedSectorAdmission,
/// };
/// use tenet::typed::{CudaStorage, TensorMap};
///
/// fn no_checked_generic_cuda_operations<R>(
///     lhs: &TensorMap<R, f64, CudaStorage>,
///     rhs: &TensorMap<R, f64, CudaStorage>,
/// ) where
///     R: TypedSectorAdmission<Mode = CheckedGenericAdmissionMode>
///         + CheckedGenericFusion
///         + CheckedGenericRigidSymbols<Scalar = f64>,
/// {
///     let _ = lhs.norm(2.0);
///     let _ = lhs.inner(rhs);
///     let _ = lhs.scale(2.0);
///     let _ = lhs.axpby(2.0, rhs, -3.0);
///     let _ = lhs.zeros_like();
/// }
/// ```
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

    fn cuda_axpby_owned(
        &self,
        required_len: usize,
        lhs: (&CudaStorage<D>, D),
        rhs: Option<(&CudaStorage<D>, D)>,
    ) -> Result<CudaStorage<D>, Error> {
        let (lhs, alpha) = lhs;
        let mut lease = self.runtime.lease_cuda()?;
        let cuda = &mut *lease;
        let expected = Placement::Cuda(cuda.device());
        Self::validate_cuda_owned_metadata(expected, lhs.placement(), required_len, lhs.len())?;
        if let Some((rhs, _)) = rhs {
            Self::validate_cuda_owned_metadata(expected, rhs.placement(), required_len, rhs.len())?;
        }

        // ponytail: #740 keeps these proven Host uploads until native device
        // allocation publishes cross-stream writes correctly and wins a bench.
        let coefficient_values = match rhs {
            Some((_, beta)) => vec![alpha, beta],
            None => vec![alpha],
        };
        // Coefficients stay data operands so the descriptor scales stay 1/0.
        // A zero coefficient skips its operand, VectorInterface's
        // `scale(x, 0) = zero(x) * 0` as on the Host (#1442): the output is
        // born zero. Arithmetic does not promise signed-zero bit parity
        // across storage backends.
        let coefficients = CudaStorage::upload_owned(cuda, coefficient_values)?;
        #[cfg(test)]
        observe_cuda_arithmetic(0, 1, 0);
        let mut output = CudaStorage::upload_owned(cuda, vec![D::from_real(0.0); required_len])?;
        #[cfg(test)]
        observe_cuda_arithmetic(1, 0, 0);
        if required_len != 0 && !alpha.is_zero() {
            cuda_gemm_region_into::<D>(
                cuda,
                &mut output.0,
                0,
                required_len,
                &lhs.0,
                0,
                required_len,
                &coefficients.0,
                0,
                1,
                required_len,
                1,
                1,
                D::from_real(1.0),
                D::from_real(0.0),
            )
            .map_err(|err| Error::from(tenet_tensors::OperationError::Dense(err)))?;
            #[cfg(test)]
            observe_cuda_arithmetic(0, 0, 1);
        }
        if let Some((rhs, _)) = rhs.filter(|(_, beta)| required_len != 0 && !beta.is_zero()) {
            cuda_gemm_region_into::<D>(
                cuda,
                &mut output.0,
                0,
                required_len,
                &rhs.0,
                0,
                required_len,
                &coefficients.0,
                1,
                1,
                required_len,
                1,
                1,
                D::from_real(1.0),
                D::from_real(1.0),
            )
            .map_err(|err| Error::from(tenet_tensors::OperationError::Dense(err)))?;
            #[cfg(test)]
            observe_cuda_arithmetic(0, 0, 1);
        }
        Ok(output)
    }

    fn cuda_zeros_owned(
        &self,
        required_len: usize,
        source: &CudaStorage<D>,
    ) -> Result<CudaStorage<D>, Error> {
        let mut lease = self.runtime.lease_cuda()?;
        let cuda = &mut *lease;
        Self::validate_cuda_owned_metadata(
            Placement::Cuda(cuda.device()),
            source.placement(),
            required_len,
            source.len(),
        )?;
        let output = CudaStorage::upload_owned(cuda, vec![D::from_real(0.0); required_len])?;
        #[cfg(test)]
        observe_cuda_arithmetic(1, 0, 0);
        Ok(output)
    }

    fn with_owned_cuda_storage(&self, storage: CudaStorage<D>) -> Self {
        let body = self
            .owned_body()
            .expect("CUDA arithmetic output authority must be owned");
        Self {
            runtime: self.runtime.clone(),
            repr: owned_repr(TypedTensorBody::dense(body.space.clone(), storage)),
        }
    }

    /// Fresh device result `factor * self` for owned storage; a lazy adjoint
    /// redirects algebraically through its canonical parent. A zero factor
    /// gives exact zeros whatever `self` holds, as on the Host (TensorKit's
    /// `scale(x, 0)`); signed-zero bits are backend-local.
    pub fn scale(&self, factor: D) -> Result<Self, Error> {
        let required_len = self.logical_space().space().required_len()?;
        if let TypedTensorRepr::Adjoint(view) = &self.repr {
            let parent = Self {
                runtime: self.runtime.clone(),
                repr: TypedTensorRepr::Owned(Arc::clone(&view.parent)),
            };
            // `alpha * A^H == (conj(alpha) * A)^H`, matching the adjoint arm
            // of the Host `scale_multiplicity_free`.
            return parent.scale(FactorScalar::adjoint(factor))?.adjoint();
        }
        let source = self.direct_cuda_storage("scale")?;
        let output = self.cuda_axpby_owned(required_len, (source, factor), None)?;
        Ok(self.with_owned_cuda_storage(output))
    }

    /// Fresh device result `x.axpby(alpha, &y, beta) == alpha * x + beta * y`,
    /// named and ordered like the Host [`TensorMap::axpby`]. A zero coefficient
    /// drops its operand, NaN and Inf included, as on the Host (TensorKit's
    /// `add`); signed-zero bits are backend-local.
    ///
    /// ```compile_fail
    /// use tenet::prelude::U1FusionRule;
    /// use tenet::typed::{CudaStorage, TensorMap};
    ///
    /// fn removed(x: &TensorMap<U1FusionRule, f64, CudaStorage<f64>>) {
    ///     let _ = x.add(x, 1.0, 2.0);
    /// }
    /// ```
    pub fn axpby<'a>(
        &self,
        alpha: D,
        y: impl Into<TensorRef<'a, R, D, CudaStorage<D>>>,
        beta: D,
    ) -> Result<Self, Error> {
        let y = y.into().operand()?;
        let y = &*y;
        let required_len = self.logical_space().space().required_len()?;
        if !self.runtime.same_runtime(&y.runtime) {
            return Err(Error::RuntimeMismatch);
        }
        if self.logical_space().space() != y.logical_space().space() {
            return Err(Error::InvalidArgument(
                "tensors live on different spaces or block layouts".to_string(),
            ));
        }
        match (&self.repr, &y.repr) {
            (TypedTensorRepr::Adjoint(lhs), TypedTensorRepr::Adjoint(rhs)) => {
                let lhs = Self {
                    runtime: self.runtime.clone(),
                    repr: TypedTensorRepr::Owned(Arc::clone(&lhs.parent)),
                };
                let rhs = Self {
                    runtime: y.runtime.clone(),
                    repr: TypedTensorRepr::Owned(Arc::clone(&rhs.parent)),
                };
                // `(alpha A^H + beta B^H) == (conj(alpha) A + conj(beta) B)^H`,
                // so the folded coefficients are conjugated before the parents
                // are combined.
                return lhs
                    .axpby(
                        FactorScalar::adjoint(alpha),
                        &rhs,
                        FactorScalar::adjoint(beta),
                    )?
                    .adjoint();
            }
            (TypedTensorRepr::Adjoint(_), TypedTensorRepr::Owned(_))
            | (TypedTensorRepr::Owned(_), TypedTensorRepr::Adjoint(_)) => {
                return Err(Error::UnsupportedOnDevice(
                    "axpby does not support mixed owned/lazy CUDA operands".to_string(),
                ));
            }
            (TypedTensorRepr::Owned(_), TypedTensorRepr::Owned(_)) => {}
        }
        let lhs = self.direct_cuda_storage("axpby")?;
        let rhs = y.direct_cuda_storage("axpby")?;
        let output = self.cuda_axpby_owned(required_len, (lhs, alpha), Some((rhs, beta)))?;
        Ok(self.with_owned_cuda_storage(output))
    }

    /// `destination = alpha * self + beta * destination` on the device: the
    /// Host [`TensorMap::axpby_into`], BLAS `axpby`.
    ///
    /// One `dot_general` over the whole payload with `alpha` as its descriptor
    /// scale and `beta` in its epilogue: `beta = 0` never reads the
    /// destination, and a zero `alpha` does not read `self` (the destination
    /// becomes `beta * destination` through a zero-source move). Nothing is
    /// uploaded or allocated on a warm context.
    ///
    /// # Errors
    ///
    /// [`Error::RuntimeMismatch`]; [`Error::InvalidArgument`] for different
    /// spaces or block layouts, a destination that is not owned dense CUDA
    /// storage, or one that aliases `self`; [`Error::UnsupportedOnDevice`] for
    /// a lazy-adjoint or compact `self`, as for the returning
    /// [`Self::axpby`] with mixed operands; [`Error::DestinationShared`];
    /// [`Error::PlacementMismatch`]. Every rejection happens before any device
    /// work.
    pub fn axpby_into(&self, destination: &mut Self, alpha: D, beta: D) -> Result<(), Error> {
        if !self.runtime.same_runtime(&destination.runtime) {
            return Err(Error::RuntimeMismatch);
        }
        if self.logical_space().space() != destination.logical_space().space() {
            return Err(Error::InvalidArgument(
                "tensors live on different spaces or block layouts".to_string(),
            ));
        }
        let source = self.direct_cuda_storage("axpby_into")?;
        let destination_storage = unique_cuda_destination(
            destination,
            &self.storage_body().data,
            self.logical_space().space(),
        )?;
        let device = Placement::Cuda(self.runtime.cuda_device_ordinal_checked()?);
        if source.placement() != device || destination_storage.placement() != device {
            return Err(Error::PlacementMismatch);
        }
        let len = TensorStorage::len(source);
        let TypedTensorRepr::Owned(destination_body) = &mut destination.repr else {
            return Err(internal_layout_error(
                "ordinary CUDA destination checked above",
            ));
        };
        let destination_data = Arc::get_mut(destination_body)
            .and_then(|body| Arc::get_mut(&mut body.data))
            .ok_or_else(|| internal_layout_error("unique CUDA destination checked above"))?;
        let TypedData::Dense(destination_data) = destination_data else {
            return Err(internal_layout_error(
                "dense CUDA destination checked above",
            ));
        };
        if len == 0 {
            return Ok(());
        }
        let dense_err = |err| Error::from(tenet_tensors::OperationError::Dense(err));
        let region = tenet_dense::CudaRegion::packed(&[len], 0).map_err(dense_err)?;
        let mut lease = self.runtime.lease_cuda()?;
        let cuda = &mut *lease;
        if alpha.is_zero() {
            return tenet_dense::cuda_region_scale::<D>(
                cuda,
                &mut destination_data.0,
                &region,
                beta,
            )
            .map_err(dense_err);
        }
        let beta = if beta.is_zero() {
            tenet_dense::CudaRegionBeta::Overwrite
        } else if beta == D::from_real(1.0) {
            tenet_dense::CudaRegionBeta::Accumulate
        } else {
            tenet_dense::CudaRegionBeta::Scale(beta)
        };
        tenet_dense::cuda_region_axpby::<D>(
            cuda,
            &source.0,
            &region,
            false,
            alpha,
            tenet_dense::CudaRegionCoefficient::One,
            beta,
            &mut destination_data.0,
            &region,
        )
        .map_err(dense_err)
    }

    /// Exact positive-zero device tensor, independent of source values.
    pub fn zeros_like(&self) -> Result<Self, Error> {
        let required_len = self.logical_space().space().required_len()?;
        if let TypedTensorRepr::Adjoint(view) = &self.repr {
            let parent = Self {
                runtime: self.runtime.clone(),
                repr: TypedTensorRepr::Owned(Arc::clone(&view.parent)),
            };
            return parent.zeros_like()?.adjoint();
        }
        let source = self.direct_cuda_storage("zeros_like")?;
        let output = self.cuda_zeros_owned(required_len, source)?;
        Ok(self.with_owned_cuda_storage(output))
    }

    /// Quantum-dimension-weighted Frobenius reduction over owned CUDA storage,
    /// **conjugate-linear in `lhs`** like the Host `coupled_region_inner`,
    /// returned in the wide `f64` lane before any narrowing to `D`.
    ///
    /// The conjugation is the GEMM operand flag `MatrixOp::Adjoint`: the left
    /// row-vector view of a coupled region has extent 1 along its first axis,
    /// so the adjoint strides `[len, 1]` address exactly the same elements as
    /// the identity strides `[1, 1]`, and the orientation only sets
    /// `lhs_conj`. For `D = f64` that flag is a no-op, so the reduction has
    /// the same operands and produces the same values as the unconjugated
    /// form it replaces.
    ///
    /// The device returns one scalar per coupled sector. Category weights stay
    /// with the tensor and are applied only after releasing the Runtime lock.
    ///
    /// # Accumulation contract
    ///
    /// Like the Host `coupled_region_inner`, which widens every entry, the sum
    /// accumulates in `f64` within *and* across coupled sectors at every
    /// payload dtype, so an `f32`/`Complex32` result is finite wherever the
    /// Host's is (#1344 for `norm`, #1383 for `inner`).
    ///
    /// * **Within one coupled sector** the sum runs on the device inside the
    ///   backend GEMM, in the dtype of its operands. Tenferro 0.7.1 offers no
    ///   widening dot or reduction: `dot_general` dispatches on the one dtype
    ///   shared by both operands and the destination (tenferro-gpu 0.7.1
    ///   `src/cubecl/gemm.rs` `dot_general_read_into_accum`), and
    ///   `DotGeneralAccumulation` carries no compute type; the BLAS-1
    ///   `vdot_read` is `cublasSdot_v2`/`cublasCdotc_v2` (`src/cubecl/blas1.rs`)
    ///   and the reduce kernels are generic over one float type
    ///   (`src/kernels/reduce/launch.rs` `launch_sum_float`). A
    ///   single-precision payload is therefore widened on the device first
    ///   ([`CudaStorage::widened`], Tenferro's `cast`; exact, since every
    ///   `f32` is an `f64`) and reduced in `f64`. That costs one device
    ///   allocation of twice the payload bytes and one elementwise pass per
    ///   *distinct* operand buffer — one for `norm` or `inner(a, a)`, two for
    ///   `inner(a, b)` — and no host transfer. `f64`/`Complex64` payloads take
    ///   the reduction directly and pay nothing extra.
    /// * **Across coupled sectors** the quantum-dimension weighting and the
    ///   final sum run on the host in `WideScalar::Wide`, the accumulator the
    ///   Host reductions use.
    ///
    /// Why not TensorKit's accumulation: TensorKit 0.17.1 `dot`
    /// (`src/tensors/linalg.jl`) sums `dim(c) * dot(b1, b2)` in the payload
    /// type, a `Float32` BLAS `sdot` per block, so it saturates exactly where
    /// the pre-#1383 device did. TeNeT's Host semantics are strictly wider,
    /// and the device must agree with its own Host.
    fn weighted_inner_cuda(
        &self,
        lhs: &CudaStorage<D>,
        rhs: &CudaStorage<D>,
    ) -> Result<num_complex::Complex64, Error> {
        match D::DTYPE {
            tenet_dense::DenseDType::F32 => self.weighted_inner_widened_cuda::<f64>(lhs, rhs),
            tenet_dense::DenseDType::C32 => {
                self.weighted_inner_widened_cuda::<num_complex::Complex64>(lhs, rhs)
            }
            _ => Ok(self.weighted_inner_cuda_wide(lhs, rhs)?.widen_complex()),
        }
    }

    /// [`Self::weighted_inner_cuda`] for a single-precision payload: both
    /// operands widened to the double-precision lane `E` on the device, one
    /// cast when they are the same buffer.
    ///
    /// Why widen rather than rescale by `norm(Inf)` (#1344): the reference
    /// only needs *some* overflow-safe sum — LinearAlgebra's `generic_norm2`
    /// accumulates a `Float32` sum in `Float64` and rescales by `normInf`
    /// only when `length * maxabs^2` leaves the payload range — and the Host
    /// already meets it by accumulating every square in `f64`, where an `f32`
    /// payload can neither overflow nor underflow. Widening reproduces that
    /// Host arithmetic exactly in one extra device pass; a scaled norm would
    /// need a max-abs pass, a host round trip for the scale and a scaled copy
    /// before the same reduction, and would still round each square in `f32`.
    ///
    /// The selected reference dispatches are narrower. Julia routes a strided
    /// `BlasFloat` array of `length >= NRM2_CUTOFF` to `BLAS.nrm2`
    /// (`LinearAlgebra/src/dense.jl:107`), a scaled sum in the payload
    /// precision. TensorKit 0.17.1 `norm` (`src/tensors/linalg.jl:277`) takes
    /// `norm(t.data)` for `UniqueFusion` and otherwise `_norm` (`:261`), which
    /// adds `dim(c) * norm(b)^2` in `float(real(scalartype))` — `Float32` for
    /// a `Float32` tensor — so it returns `Inf` once one weighted block square
    /// exceeds `f32::MAX` (the "big" SU(2) fixture of
    /// `typed_cuda_single_precision`). The Host and this device norm
    /// accumulate every square and the cross-sector combine in `f64`, which is
    /// strictly wider than that dispatch. For `inner` a rescale is not even
    /// available: a cancelling sum has no scale that keeps every product in
    /// range and the result accurate.
    fn weighted_inner_widened_cuda<E: CudaPayload>(
        &self,
        lhs: &CudaStorage<D>,
        rhs: &CudaStorage<D>,
    ) -> Result<num_complex::Complex64, Error> {
        let mut lease = self.runtime.lease_cuda()?;
        let wide_lhs = lhs.widened::<E>(&mut lease)?;
        let wide_rhs = if std::ptr::eq(lhs, rhs) {
            None
        } else {
            Some(rhs.widened::<E>(&mut lease)?)
        };
        drop(lease);
        let wide_rhs = wide_rhs.as_ref().unwrap_or(&wide_lhs);
        Ok(self
            .weighted_inner_cuda_wide(&wide_lhs, wide_rhs)?
            .widen_complex())
    }

    /// [`Self::weighted_inner_cuda`] over operands already in the lane `E`: the
    /// cross-sector total in `E`'s wide accumulator.
    fn weighted_inner_cuda_wide<E: CudaPayload>(
        &self,
        lhs: &CudaStorage<E>,
        rhs: &CudaStorage<E>,
    ) -> Result<<E as tenet_tensors::WideScalar>::Wide, Error> {
        let space = self.logical_space().space();
        let regions = sector_regions(space.structure(), space.nout())?;
        let mut lease = self.runtime.lease_cuda()?;
        let cuda = &mut *lease;
        validate_cuda_reduction_placement(
            Placement::Cuda(cuda.device()),
            lhs.placement(),
            rhs.placement(),
        )?;
        // ponytail: #740 keeps the proven host-zero upload until a native
        // allocation has correct cross-stream publication and measured value.
        let mut partials =
            CudaStorage::upload_owned(cuda, vec![E::from_real(0.0); regions.len().max(1)])?;
        {
            let mut gemm = CudaStorageGemm::new(cuda);
            for (index, region) in regions.iter().enumerate() {
                let len = region.rows() * region.cols();
                if len == 0 {
                    continue;
                }
                gemm.matmul_range_with_ops_scaled_into(
                    &mut partials,
                    index,
                    lhs,
                    region.range().start,
                    rhs,
                    region.range().start,
                    1,
                    len,
                    1,
                    tenet_dense::MatrixOp::Adjoint,
                    tenet_dense::MatrixOp::Identity,
                    E::from_real(1.0),
                )?;
            }
        }
        let values = download_cuda_reduction_partials(&partials, cuda)?;
        drop(lease);

        Ok(regions
            .iter()
            .zip(values)
            .map(|(region, value)| {
                value.widen()
                    * <<E as tenet_tensors::WideScalar>::Wide as FactorScalar>::from_real(
                        self.logical_space().provider().dim_scalar(region.coupled()),
                    )
            })
            .fold(
                <<E as tenet_tensors::WideScalar>::Wide as FactorScalar>::from_real(0.0),
                |total, term| total + term,
            ))
    }

    /// Quantum-dimension-weighted Frobenius norm of a device tensor: the
    /// `p == 2` arm of the Host [`TensorMap::norm`].
    ///
    /// Only `p == 2` has a device reduction. Any other valid `p` is
    /// [`Error::UnsupportedOnDevice`] rather than a hidden Host transfer; an
    /// invalid `p` is [`Error::InvalidArgument`], as on the Host.
    ///
    /// A lazy adjoint delegates to its canonical parent because this norm is
    /// adjoint invariant; no logical-adjoint payload is materialized.
    ///
    /// # Accumulation and range
    ///
    /// The norm accumulates in `f64` at every payload dtype, within and
    /// across coupled sectors, exactly like the Host `norm`: an
    /// `f32`/`Complex32` payload is widened on the device by one cast
    /// (one device allocation of twice the payload bytes and one elementwise
    /// pass) before the per-sector reduction, because Tenferro 0.7.1 offers no
    /// widening reduction. Its result is therefore finite wherever every entry
    /// is finite, and nonzero wherever some entry is, matching the Host within
    /// `f64` rounding (#1344). `f64`/`Complex64` payloads take the unwidened
    /// reduction and pay nothing extra. [`Self::inner`] shares this reduction.
    ///
    /// Unlike the Host `norm`, an `f64`/`Complex64` device sum is not
    /// rescaled: it returns `inf` once the squared norm exceeds `f64::MAX`
    /// (entries near `1e154` and above) and loses accuracy once it falls
    /// below `f64::MIN_POSITIVE` (norms near `1e-154` and below), where the
    /// Host returns the representable norm.
    pub fn norm(&self, p: f64) -> Result<f64, Error> {
        validate_norm_p(p)?;
        if p != 2.0 {
            return Err(Error::UnsupportedOnDevice(format!(
                "device norm supports only p = 2, got {p}"
            )));
        }
        if let TypedTensorRepr::Adjoint(view) = &self.repr {
            return Self {
                runtime: self.runtime.clone(),
                repr: TypedTensorRepr::Owned(Arc::clone(&view.parent)),
            }
            .norm(p);
        }
        let storage = self.direct_cuda_storage("norm")?;
        // `<t, t>` is real up to rounding; the norm is its real part's root,
        // matching the Host `norm_multiplicity_free`.
        Ok(self.weighted_inner_cuda(storage, storage)?.re.sqrt())
    }

    /// TensorKit `dot(x, y)`: the quantum-dimension-weighted Frobenius inner
    /// product with **`self` conjugated**, matching the Host
    /// `inner_multiplicity_free`. Lazy adjoints remain an explicit
    /// unsupported device scope.
    ///
    /// Accumulates like the Host `inner`: in `f64` within and across coupled
    /// sectors at every payload dtype, then narrows the total to `D` once
    /// (`D::from_complex64`, as the Host does). An `f32`/`Complex32` result is
    /// therefore finite exactly where the Host's is, including a cancelling
    /// sum whose individual products exceed `f32::MAX`. A single-precision
    /// call widens each distinct operand on the device first; the cost is
    /// documented on `weighted_inner_cuda` (#1383).
    #[doc(alias = "dot")]
    pub fn inner<'a>(
        &self,
        other: impl Into<TensorRef<'a, R, D, CudaStorage<D>>>,
    ) -> Result<D, Error> {
        let other = other.into().operand()?;
        let other = &*other;
        if !self.runtime.same_runtime(&other.runtime) {
            return Err(Error::RuntimeMismatch);
        }
        if self.logical_space().space() != other.logical_space().space() {
            return Err(Error::InvalidArgument(
                "tensors live on different spaces or block layouts".to_string(),
            ));
        }
        let lhs = self.direct_cuda_storage("inner")?;
        let rhs = other.direct_cuda_storage("inner")?;
        self.weighted_inner_cuda(lhs, rhs).map(D::from_complex64)
    }

    fn cuda_fusion_operand(
        &self,
        operation: &'static str,
    ) -> Result<
        (
            &BoundDynamicFusionMapSpace<R>,
            tenet_tensors::FusionOperand<'_>,
            &CudaStorage<D>,
        ),
        Error,
    > {
        match &self.repr {
            TypedTensorRepr::Owned(body) => match body.data.as_ref() {
                TypedData::Dense(storage) => Ok((
                    &body.space,
                    tenet_tensors::FusionOperand::direct(body.space.space()),
                    storage,
                )),
                TypedData::Diagonal(_) => Err(Error::UnsupportedOnDevice(format!(
                    "{operation} requires dense CUDA storage"
                ))),
            },
            TypedTensorRepr::Adjoint(view) => {
                let TypedData::Dense(storage) = view.parent.data.as_ref() else {
                    unreachable!("TypedAdjointView::new admits only dense parents")
                };
                Ok((
                    &view.logical_space,
                    tenet_tensors::FusionOperand::adjoint(view.parent.space.space()),
                    storage,
                ))
            }
        }
    }

    /// TensorKit `tensorcontract` on device tensors: the Host
    /// [`TensorMap::contract`] semantics for arbitrary contracted legs, output
    /// order and codomain/domain split, on owned or lazy-adjoint operands,
    /// executed on the device.
    ///
    /// # One planning authority
    ///
    /// Everything categorical is the Host's, computed on the host before the
    /// device is touched: the destination space (by the Host's own derivation
    /// for owned or lazy operands), the route, the orientation and axis-order
    /// candidate, the source and output transform structures, the operand
    /// borrow decisions and the core plan. The device only replays that
    /// artifact — each non-identity source transform into device scratch, the
    /// coupled-sector GEMMs, the output transform — which is TensorKit's
    /// `blas_contract!` dataflow and QSpace's `contract` dataflow, so FLOPs,
    /// bytes moved and working set are the Host's. A contraction already in
    /// TensorKit `mul!` form keeps the fully-direct GEMM route over the parent
    /// buffers, with lazy adjoints as GEMM operand flags and no transform; it
    /// takes no Host context lock.
    ///
    /// # Cost
    ///
    /// A warm call uploads nothing but the output initialisation (#740):
    /// exactly one H2D of `required_len * size_of::<D>()` bytes, from one host
    /// `Vec` of that size; its only device allocation is the returned output;
    /// it downloads nothing. The first use of a transform structure uploads its
    /// coefficient payload once (as the device [`Self::permute`] does), and a
    /// new high-water mark of the operand or core-destination scratch costs one
    /// zero upload of the new size per buffer; both then stay resident on the
    /// Runtime (see [`crate::typed::Runtime::cuda_contract_scratch_bytes`]) until
    /// [`crate::typed::Runtime::clear_tree_transform_cache`]. Per call the
    /// device submits one region move per Single block / pack / scatter
    /// column, one GEMM per recoupling job and per coupled-sector job, one
    /// zero fill per inactive layout of each transform in overwrite mode, and
    /// one zero fill per core-destination block no GEMM writes (the Host
    /// clears its whole core destination instead).
    ///
    /// The warm contract holds while this Runtime's Host transform store
    /// admits the structures (see [`Self::permute`]).
    ///
    /// # Numerics
    ///
    /// Device and Host agree to dtype tolerance, never bitwise: the GEMM
    /// summation order is cuTENSOR's, a Single-block move rounds as
    /// `alpha * (c * x)`, and where the Host picks its dense `Structure` route
    /// for a conjugated operand the device runs the transform route instead.
    /// Exact zeros may differ in sign (Host `-0.0` where the device writes
    /// `+0.0`).
    ///
    /// # Fermionic twist
    ///
    /// A fermionic contraction over dual contracted legs twists the
    /// core-right operand (TensorKit's `twist!` in `blas_contract!`). The
    /// canonical form with a twist uniform per coupled sector folds it into
    /// the GEMM alpha; every other case — general axes, or a twist that
    /// varies within one coupled sector — folds `θ_b` into the descriptor
    /// alpha of the source-transform move writing core-right block `b`, where
    /// the Host scales the transformed operand in place afterwards: the same
    /// values, one pass fewer, and no extra upload.
    ///
    /// # Errors
    ///
    /// In this order, all before any device work: [`Error::RuntimeMismatch`];
    /// [`tenet_tensors::OperationError::UnsupportedTensorContractScope`] for
    /// non-symmetric (anyonic or `NoBraiding`) providers whatever the axes,
    /// as on Host — a behaviour change: the canonical anyonic (before G2c-1a)
    /// and `NoBraiding` (before #1372) device contractions were accepted;
    /// [`Error::UnsupportedOnDevice`] for
    /// diagonal storage; the Host's own errors for malformed axes, output
    /// orders or mismatched legs; [`Error::PlacementMismatch`].
    pub fn contract<'a>(
        &self,
        other: impl Into<TensorRef<'a, R, D, CudaStorage<D>>>,
        spec: &ContractSpec<'_>,
    ) -> Result<Self, Error> {
        let (lhs_axes, rhs_axes) = (spec.lhs, spec.rhs);
        let output_axes = &spec.output_axes()[..];
        let other = other.into().operand()?;
        let other = &*other;
        if !self.runtime.same_runtime(&other.runtime) {
            return Err(Error::RuntimeMismatch);
        }
        reject_non_symmetric_contraction(self.logical_space().provider().braiding_style())?;
        let (lhs_space, lhs_operand, lhs_storage) = self.cuda_fusion_operand("contract")?;
        let (rhs_space, rhs_operand, rhs_storage) = other.cuda_fusion_operand("contract")?;
        let output_order = OutputAxisOrder::from_axes(output_axes);
        // The Host derives the destination per representation, and so does
        // this: two owned operands through the owned derivation, a lazy
        // adjoint through the oriented one.
        let dst_space = if lhs_operand.storage_conjugate() || rhs_operand.storage_conjugate() {
            oriented_contract_destination(
                lhs_space,
                lhs_operand,
                rhs_space,
                rhs_operand,
                lhs_axes,
                rhs_axes,
                output_order,
                Some(spec.codomain.len()),
            )?
        } else {
            BoundDynamicFusionMapSpace::contracted_multiplicity_free_partitioned(
                lhs_space,
                rhs_space,
                lhs_axes,
                rhs_axes,
                output_order,
                spec.codomain.len(),
            )?
        };
        let axes = tenet_tensors::TensorContractSpec::new_with_conjugation(
            lhs_axes,
            rhs_axes,
            output_order,
            lhs_operand.storage_conjugate(),
            rhs_operand.storage_conjugate(),
        );
        // The canonical core route resolves from the operands alone and takes
        // no lock. Otherwise the Host context lease compiles the artifact and
        // is dropped before the device lease is taken; nothing under the
        // device lease leases again.
        let resolution = match tenet_tensors::try_compile_storage_contract_core_route(
            &dst_space,
            lhs_operand,
            rhs_operand,
            axes,
        )? {
            Some(core) => core,
            None => {
                let mut lease = self.runtime.lease_context()?;
                lease
                    .context()
                    .multiplicity_free_lane::<D>()?
                    .compile_storage_contract_dynamic_tree(
                        &dst_space,
                        lhs_operand,
                        rhs_operand,
                        axes,
                    )?
            }
        };
        let device = Placement::Cuda(self.runtime.cuda_device_ordinal_checked()?);
        if lhs_storage.placement() != device || rhs_storage.placement() != device {
            return Err(Error::PlacementMismatch);
        }
        let required_len = dst_space.space().required_len()?;

        let mut lease = self.runtime.lease_cuda()?;
        let (cuda, transforms, scratch) = lease.split_contract();
        // ponytail: #740 — the device seam initializes an output by uploading
        // zeros; replace only with a measured native allocation.
        let mut dst = CudaStorage::upload_owned(cuda, vec![D::from_real(0.0); required_len])?;
        tenet_tensors::execute_storage_contract_resolution_on_cuda(
            cuda,
            transforms,
            scratch,
            &resolution,
            dst_space.space().structure(),
            &mut dst,
            lhs_storage,
            rhs_storage,
            D::from_real(1.0),
            tenet_tensors::ContractDestinationInit::Zeroed,
        )?;
        drop(lease);
        Ok(Self {
            runtime: self.runtime.clone(),
            repr: owned_repr(TypedTensorBody::dense(dst_space, dst)),
        })
    }

    /// `destination = alpha * self.contract(other, spec) + beta * destination`
    /// on the device, preserving the destination's provider, space, body and
    /// device allocation: arbitrary contracted and output axes, owned or
    /// lazy-adjoint operands, through the same resolution as the returning
    /// [`Self::contract`].
    ///
    /// The admission sequence is the Host one ([`TensorMap::contract_into`])
    /// plus the device's own boundaries: same Runtime, the symmetric-braiding
    /// boundary of [`Self::contract`], same rule identity, an owned dense
    /// device destination that aliases neither operand's payload body, the
    /// contraction's fusion space and block layout (malformed axes report the
    /// Host's errors here), exact operand and destination lengths, unique
    /// destination ownership ([`Error::DestinationShared`]), then the compile
    /// of the Host route and placement. Every rejection — including a compile
    /// error — happens before any device work, so a rejected call leaves
    /// `destination` untouched.
    ///
    /// `alpha` and `beta` ride the epilogue of whatever writes each element:
    /// the core GEMMs (`alpha * job_alpha`, `beta`) when they write the
    /// destination directly, otherwise the output transform in `Axpby(beta)`
    /// mode. The coupled sectors no GEMM reaches become `beta * destination`
    /// through a zero-source region move — zeros for `beta = 0`, which never
    /// reads the destination, and nothing for `beta = 1`.
    ///
    /// # Cost
    ///
    /// No destination reset or separate `beta` pass over written blocks. A
    /// warm call transfers nothing and allocates nothing on the device;
    /// scratch and coefficient payloads are as for [`Self::contract`].
    pub fn contract_into<'a>(
        &self,
        other: impl Into<TensorRef<'a, R, D, CudaStorage<D>>>,
        spec: &ContractSpec<'_>,
        destination: &mut Self,
        alpha: D,
        beta: D,
    ) -> Result<(), Error> {
        let (lhs_axes, rhs_axes) = (spec.lhs, spec.rhs);
        let output_axes = &spec.output_axes()[..];
        let other = other.into().operand()?;
        let other = &*other;
        if !self.runtime.same_runtime(&other.runtime)
            || !self.runtime.same_runtime(&destination.runtime)
        {
            return Err(Error::RuntimeMismatch);
        }
        reject_non_symmetric_contraction(self.logical_space().provider().braiding_style())?;
        let identity = TypedSectorAdmission::typed_rule_identity(self.provider());
        if identity != TypedSectorAdmission::typed_rule_identity(other.provider())
            || identity != TypedSectorAdmission::typed_rule_identity(destination.provider())
        {
            return Err(Error::RuleMismatch);
        }

        // Same error kind and wording as Host (only the storage noun names the
        // placement): "the destination is not an owned dense tensor" is a
        // caller mistake on either placement, not a device capability limit.
        let (destination_body, destination_storage) = match &destination.repr {
            TypedTensorRepr::Owned(body) => match body.data.as_ref() {
                TypedData::Dense(storage) => (body, storage),
                TypedData::Diagonal(_) => {
                    return Err(Error::InvalidArgument(
                        "contraction destination must use ordinary dense CUDA storage".to_string(),
                    ))
                }
            },
            TypedTensorRepr::Adjoint(_) => {
                return Err(Error::InvalidArgument(
                    "contraction destination must use ordinary dense CUDA storage".to_string(),
                ))
            }
        };
        if Arc::ptr_eq(&destination_body.data, &self.storage_body().data)
            || Arc::ptr_eq(&destination_body.data, &other.storage_body().data)
        {
            return Err(Error::InvalidArgument(
                "destination storage must not alias an input".to_string(),
            ));
        }

        let (lhs_space, lhs_operand, lhs_storage) = self.cuda_fusion_operand("contract")?;
        let (rhs_space, rhs_operand, rhs_storage) = other.cuda_fusion_operand("contract")?;
        let output_order = OutputAxisOrder::from_axes(output_axes);
        let expected = BoundDynamicFusionMapSpace::contracted_multiplicity_free_partitioned(
            lhs_space,
            rhs_space,
            lhs_axes,
            rhs_axes,
            output_order,
            spec.codomain.len(),
        )?;
        if destination_body.space.space() != expected.space() {
            return Err(Error::InvalidArgument(
                "destination fusion space or block layout does not match the contraction result"
                    .to_string(),
            ));
        }
        let execution_destination =
            lhs_space.rebind_validated(&destination_body.space.validated_layout())?;

        let required_destination = destination_body.space.space().required_len()?;
        for (tensor, actual, required) in [
            (
                "lhs",
                TensorStorage::len(lhs_storage),
                lhs_operand.storage_space().required_len()?,
            ),
            (
                "rhs",
                TensorStorage::len(rhs_storage),
                rhs_operand.storage_space().required_len()?,
            ),
            (
                "destination",
                TensorStorage::len(destination_storage),
                required_destination,
            ),
        ] {
            if actual != required {
                return Err(Error::InvalidArgument(format!(
                    "{tensor} storage length {actual} does not match required length {required}"
                )));
            }
        }
        if Arc::strong_count(destination_body) != 1
            || Arc::strong_count(&destination_body.data) != 1
        {
            return Err(Error::DestinationShared);
        }
        let destination_placement = destination_storage.placement();

        // Resolution exactly as the returning `contract`: the lock-free core
        // route, otherwise the Host artifact under a context lease dropped
        // before the device lease.
        let axes = tenet_tensors::TensorContractSpec::new_with_conjugation(
            lhs_axes,
            rhs_axes,
            output_order,
            lhs_operand.storage_conjugate(),
            rhs_operand.storage_conjugate(),
        );
        let resolution = match tenet_tensors::try_compile_storage_contract_core_route(
            &execution_destination,
            lhs_operand,
            rhs_operand,
            axes,
        )? {
            Some(core) => core,
            None => {
                let mut lease = self.runtime.lease_context()?;
                lease
                    .context()
                    .multiplicity_free_lane::<D>()?
                    .compile_storage_contract_dynamic_tree(
                        &execution_destination,
                        lhs_operand,
                        rhs_operand,
                        axes,
                    )?
            }
        };
        let device = Placement::Cuda(self.runtime.cuda_device_ordinal_checked()?);
        if lhs_storage.placement() != device
            || rhs_storage.placement() != device
            || destination_placement != device
        {
            return Err(Error::PlacementMismatch);
        }

        let TypedTensorRepr::Owned(destination_body) = &mut destination.repr else {
            return Err(internal_layout_error(
                "ordinary CUDA destination checked above",
            ));
        };
        let destination_body = Arc::get_mut(destination_body)
            .ok_or_else(|| internal_layout_error("unique CUDA destination body checked above"))?;
        let destination_data = Arc::get_mut(&mut destination_body.data).ok_or_else(|| {
            internal_layout_error("unique CUDA destination payload checked above")
        })?;
        let TypedData::Dense(destination_data) = destination_data else {
            return Err(internal_layout_error(
                "dense CUDA destination checked above",
            ));
        };

        let mut lease = self.runtime.lease_cuda()?;
        let (cuda, transforms, scratch) = lease.split_contract();
        tenet_tensors::execute_storage_contract_resolution_on_cuda(
            cuda,
            transforms,
            scratch,
            &resolution,
            execution_destination.space().structure(),
            destination_data,
            lhs_storage,
            rhs_storage,
            alpha,
            tenet_tensors::ContractDestinationInit::Axpby(beta),
        )?;
        Ok(())
    }

    /// Tensor-map composition on owned or lazy-adjoint device tensors. This uses the
    /// twist-free composition compiler and therefore remains distinct from
    /// [`Self::contract`] for fermionic providers, and it admits every
    /// braiding style, as on Host, where `contract` requires a symmetric one.
    #[doc(alias = "mul")]
    pub fn compose<'a>(
        &self,
        other: impl Into<TensorRef<'a, R, D, CudaStorage<D>>>,
    ) -> Result<Self, Error> {
        let other = other.into().operand()?;
        let other = &*other;
        if !self.runtime.same_runtime(&other.runtime) {
            return Err(Error::RuntimeMismatch);
        }
        let (lhs_space, lhs_operand, lhs_storage) = self.cuda_fusion_operand("compose")?;
        let (rhs_space, rhs_operand, rhs_storage) = other.cuda_fusion_operand("compose")?;
        let lhs_axes: Vec<_> = (self.codomain_rank()..self.rank()).collect();
        let rhs_axes: Vec<_> = (0..other.codomain_rank()).collect();
        let dst_space = BoundDynamicFusionMapSpace::contracted_multiplicity_free(
            lhs_space, rhs_space, &lhs_axes, &rhs_axes,
        )?;
        let mut lease = self.runtime.lease_cuda()?;
        let cuda = &mut *lease;
        let expected_placement = Placement::Cuda(cuda.device());
        if lhs_storage.placement() != expected_placement
            || rhs_storage.placement() != expected_placement
        {
            return Err(Error::PlacementMismatch);
        }
        let mut dst = CudaStorage::upload_owned(
            cuda,
            vec![D::from_real(0.0); dst_space.space().required_len()?],
        )?;
        tenet_tensors::tensorcompose_fusion_dyn_prelowered_direct_on_storage(
            &mut CudaStorageGemm::new(cuda),
            &dst_space,
            &mut dst,
            lhs_operand,
            lhs_storage,
            rhs_operand,
            rhs_storage,
            &lhs_axes,
            &rhs_axes,
        )?;
        drop(lease);
        Ok(Self {
            runtime: self.runtime.clone(),
            repr: owned_repr(TypedTensorBody::dense(dst_space, dst)),
        })
    }

    /// Device lowering shared by every structural transform, mirroring Host
    /// `tree_transform_multiplicity_free_real` branch for branch.
    ///
    /// Everything categorical is the Host's, unchanged: the same
    /// [`TreeTransformOperation`], the same rule validation, the same
    /// categorical group plan, the same `RuleIdentity`-keyed transform cache
    /// and the same completed `TreeTransformStructure`. Only the executor
    /// below that structure differs, so a device result is the Host result up
    /// to dtype tolerance.
    ///
    /// Lock order — and the reason the two phases are written apart: planning
    /// runs under the pooled Host context lease, which is dropped before the
    /// device lease is taken, and nothing reached under the device lease
    /// leases again (`std::sync::Mutex` is not re-entrant). Exactly one device
    /// lease per replay.
    fn tree_transform_cuda(
        &self,
        operation_name: &'static str,
        operation: TreeTransformOperation,
    ) -> Result<Self, Error> {
        // Host order: a lazy adjoint lowers the operation onto its parent and
        // re-wraps the owned result, rather than materializing the adjoint.
        // `.adjoint()` is itself lease-free, so this recursion never nests a
        // device lease.
        if let TypedTensorRepr::Adjoint(view) = &self.repr {
            let parent_space = view.parent.space.space();
            let lowered = lower_adjoint_tree_transform_operation(
                parent_space.nout(),
                parent_space.nin(),
                &operation,
            )?;
            let parent = Self {
                runtime: self.runtime.clone(),
                repr: TypedTensorRepr::Owned(Arc::clone(&view.parent)),
            };
            return parent
                .tree_transform_cuda(operation_name, lowered)?
                .adjoint();
        }
        // Host's compact-diagonal arm is unreachable here — `to_cuda`
        // densifies a diagonal — but a device tensor that somehow carried one
        // is an explicit capability error, never a silent dense fallback.
        let src = self.direct_cuda_storage(operation_name)?;
        let body = self
            .owned_body()
            .ok_or_else(|| internal_layout_error("owned device tree transform input"))?;
        let dst_space = body.space.transformed_multiplicity_free(&operation)?;
        let required_len = dst_space.space().required_len()?;

        let structure = {
            let mut lease = self.runtime.lease_context()?;
            lease
                .context()
                .multiplicity_free_lane::<D>()?
                .tree_context_mut()
                .compile_tree_pair_structure(
                    body.space.provider(),
                    &operation,
                    dst_space.space().structure(),
                    body.space.space().structure(),
                )?
        };

        let mut lease = self.runtime.lease_cuda()?;
        let (cuda, executor) = lease.split();
        if src.placement() != Placement::Cuda(cuda.device()) {
            return Err(Error::PlacementMismatch);
        }
        // ponytail: #740 — the device seam still initializes an output by
        // uploading zeros; replace only with a measured native allocation.
        //
        // Why not report an inexpressible destination layout before spending
        // this upload: a returning transform cannot produce one. `dst_space`
        // comes from `transformed_multiplicity_free`, i.e. the canonical
        // final-homspace layout, never from the source's strides, and block
        // strides are `usize`, so both arms of the executor's layout rejection
        // are unreachable here; the buffer is this method's own and is dropped
        // on any error, so no caller-visible state is touched either way.
        // G2b-3 has no such argument — its destination is the caller's — so it
        // must validate before it writes, and must redo this reasoning if it
        // admits a non-canonical layout through `admit_exact_tree_pair_layout`.
        let mut dst = CudaStorage::upload_owned(cuda, vec![D::from_real(0.0); required_len])?;
        executor.replay(
            cuda,
            &structure,
            dst_space.space().structure(),
            body.space.space().structure(),
            &mut dst,
            src,
            D::from_real(1.0),
            CudaTreeTransformDestination::Overwrite,
        )?;
        drop(lease);
        Ok(Self {
            runtime: self.runtime.clone(),
            repr: owned_repr(TypedTensorBody::dense(dst_space, dst)),
        })
    }

    /// Shared body of the three planar device operations, mirroring Host
    /// `planar`: derive the planar axis order, let the expert layer validate
    /// it, and run it as a transpose — never as a permute, because domain
    /// trees run opposite to the planar boundary.
    fn planar_cuda(
        &self,
        operation_name: &'static str,
        kind: PlanarRequestKind<'_>,
    ) -> Result<Self, Error> {
        let operation = with_planar_axes(
            self.codomain_rank(),
            self.rank(),
            kind,
            |codomain_axes, domain_axes| {
                Ok(TreeTransformOperation::transpose(
                    codomain_axes.iter().copied(),
                    domain_axes.iter().copied(),
                ))
            },
        )?;
        self.tree_transform_cuda(operation_name, operation)
    }

    /// TensorKit `permute` on a device tensor: the Host
    /// [`TensorMap::permute`] semantics, executed on the device.
    ///
    /// # Cost
    ///
    /// A warm call uploads no structure data and downloads nothing. Its only
    /// transfer is the output initialisation (#740): exactly one H2D of
    /// `required_len * size_of::<D>()` bytes, from one host `Vec` of that
    /// size, and its only device allocation is the returned output. The first
    /// call for a given structure additionally uploads that structure's
    /// coefficient payload once and may grow the pack/scatter workspace once.
    ///
    /// On device the replay runs in overwrite mode, so it also zero-fills
    /// every *inactive* destination layout on the device — redundant work over
    /// an output that was just uploaded as zeros, costing one submission and
    /// `Σ(inactive layout elements)` device writes per call. It is kept
    /// because it is what makes the executor's destination mode independent of
    /// what the destination held, which `*_into` with `beta = 0` relies on; removing
    /// it here would need a second replay mode for no transfer saving.
    ///
    /// That warm contract holds only while this Runtime's Host transform store
    /// admits the structure: with a tree-transform cache byte budget of zero,
    /// or for a structure whose entry exceeds the store's per-entry limit, the
    /// store hands back a fresh allocation per call and the device therefore
    /// re-uploads its coefficients per call.
    ///
    /// # Numerics
    ///
    /// Inherited from the device executor: the block set written is the Host's
    /// exactly; coefficient-`1` `f64` moves are bitwise, and everything else
    /// agrees to dtype tolerance, with one rounding position moved for Single
    /// blocks (`alpha * (c * x)` where the Host folds the scales and rounds as
    /// `(alpha * c) * x`) and with the recoupling GEMM's summation order that
    /// of cuTENSOR rather than the Host kernel.
    ///
    /// # Errors
    ///
    /// [`Error::UnsupportedOnDevice`] for diagonal storage;
    /// [`Error::PlacementMismatch`] for a payload on another device; the
    /// expert layer's own [`Error::Operation`] / [`Error::Core`] /
    /// [`Error::FusionAlgebra`] for malformed axis lists — all before any
    /// device write.
    ///
    /// Checked-Generic, Generic and complex-coefficient providers are excluded
    /// by this impl's bound, so they are a compile-time boundary:
    ///
    /// ```compile_fail
    /// use tenet::core::{
    ///     CheckedGenericAdmissionMode, CheckedGenericFusion, CheckedGenericRigidSymbols,
    ///     TypedSectorAdmission,
    /// };
    /// use tenet::typed::{CudaStorage, TensorMap};
    ///
    /// fn no_checked_generic_cuda_permute<R>(tensor: &TensorMap<R, f64, CudaStorage>)
    /// where
    ///     R: TypedSectorAdmission<Mode = CheckedGenericAdmissionMode>
    ///         + CheckedGenericFusion
    ///         + CheckedGenericRigidSymbols<Scalar = f64>,
    /// {
    ///     let _ = tensor.permute(&[1], &[0]);
    /// }
    /// ```
    ///
    /// ```compile_fail
    /// use tenet::core::FibonacciFusionRule;
    /// use tenet::typed::{CudaStorage, TensorMap};
    ///
    /// fn no_generic_cuda_permute(tensor: &TensorMap<FibonacciFusionRule, f64, CudaStorage>) {
    ///     let _ = tensor.permute(&[1], &[0]);
    /// }
    /// ```
    ///
    /// # Source compatibility
    ///
    /// Enabling the `cuda` feature adds these four names to a second `impl`,
    /// so the *path* forms `TensorMap::permute`, `TensorMap::braid`,
    /// `TensorMap::transpose` and
    /// `TensorMap::repartition` become ambiguous (`E0034`) where they were not
    /// before — the same non-additivity the device `adjoint`, `norm` and
    /// `scale` already have. Method-call syntax (`tensor.permute(..)`), a
    /// closure (`|t| t.permute(..)`) or a fully qualified
    /// `TensorMap::<R, D>::permute` all keep working.
    ///
    /// The multiplicity-free twin compiles, for either device payload:
    ///
    /// ```
    /// use tenet::prelude::U1FusionRule;
    /// use tenet::typed::{CudaPayload, CudaStorage, TensorMap};
    ///
    /// fn device_permute<D: CudaPayload>(tensor: &TensorMap<U1FusionRule, D, CudaStorage<D>>) {
    ///     let _ = tensor.permute(&[1], &[0]);
    ///     let _ = tensor.braid(&[1], &[0], &[0, 1]);
    ///     let _ = tensor.transpose(&[1], &[0]);
    ///     let _ = tensor.repartition(0);
    /// }
    /// ```
    pub fn permute(&self, codomain_axes: &[usize], domain_axes: &[usize]) -> Result<Self, Error> {
        if self.axes_are_identity(codomain_axes, domain_axes) {
            return Ok(self.clone());
        }
        self.tree_transform_cuda(
            "permute",
            TreeTransformOperation::permute(
                codomain_axes.iter().copied(),
                domain_axes.iter().copied(),
            ),
        )
    }

    /// The device form of the Host leg-role composition: `op` runs on
    /// `permute(self, rows, cols)`, and the current split borrows `self`
    /// without a transform or a clone.
    fn with_cuda_leg_roles<T>(
        &self,
        rows: &[usize],
        cols: &[usize],
        op: impl FnOnce(&Self) -> Result<T, Error>,
    ) -> Result<T, Error> {
        if self.axes_are_identity(rows, cols) {
            return op(self);
        }
        op(&self.permute(rows, cols)?)
    }

    /// TensorKit `braid` on a device tensor: the Host [`TensorMap::braid`]
    /// semantics, executed on the device.
    ///
    /// Cost, numerics and provider boundary are [`Self::permute`]'s.
    ///
    /// # Errors
    ///
    /// [`Error::InvalidArgument`] when `levels` does not list one level per
    /// source axis — checked **before** the identity short circuit, as on
    /// Host, so a mis-lengthed `levels` is reported even for an identity
    /// permutation. Otherwise as [`Self::permute`].
    pub fn braid(
        &self,
        codomain_axes: &[usize],
        domain_axes: &[usize],
        levels: &[usize],
    ) -> Result<Self, Error> {
        let rank = self.rank();
        if levels.len() != rank {
            return Err(Error::InvalidArgument(format!(
                "braid levels must list one level per source axis \
                 (expected {rank}, got {})",
                levels.len()
            )));
        }
        if self.axes_are_identity(codomain_axes, domain_axes) {
            return Ok(self.clone());
        }
        let nout = self.codomain_rank();
        self.tree_transform_cuda(
            "braid",
            TreeTransformOperation::braid(
                codomain_axes.iter().copied(),
                domain_axes.iter().copied(),
                levels[..nout].iter().copied(),
                levels[nout..].iter().copied(),
            ),
        )
    }

    /// TensorKit `repartition(t, N1, N2)` on a device tensor: the Host
    /// [`TensorMap::repartition`] semantics, executed on the device.
    ///
    /// Requesting the split the tensor already has returns a clone and does no
    /// device work at all.
    ///
    /// Cost, numerics and provider boundary are [`Self::permute`]'s.
    pub fn repartition(&self, num_codomain: usize) -> Result<Self, Error> {
        if num_codomain == self.codomain_rank() {
            return Ok(self.clone());
        }
        self.planar_cuda(
            "repartition",
            PlanarRequestKind::Repartition { num_codomain },
        )
    }

    /// TensorKit `transpose(t, (p₁, p₂))` on a device tensor: the Host
    /// [`TensorMap::transpose`] semantics, executed on the device.
    ///
    /// Cost, numerics and provider boundary are [`Self::permute`]'s.
    pub fn transpose(&self, codomain_axes: &[usize], domain_axes: &[usize]) -> Result<Self, Error> {
        if self.axes_are_identity(codomain_axes, domain_axes) {
            return Ok(self.clone());
        }
        self.planar_cuda(
            "transpose",
            PlanarRequestKind::Explicit {
                codomain_axes,
                domain_axes,
            },
        )
    }

    /// `destination = alpha * self.permute(codomain_axes, domain_axes) +
    /// beta * destination` on the device, without replacing its provider,
    /// space, body, or device allocation: the Host [`TensorMap::permute_into`].
    ///
    /// The admission sequence is the Host one plus the device's own placement
    /// check: same Runtime, same rule identity, an owned dense device source
    /// (a lazy adjoint is *rejected*, not lowered onto its parent, exactly as
    /// on Host — the destination is the caller's, so there is no re-wrapping
    /// to do), an owned dense device destination that does not alias the
    /// source payload, the admitted tree-pair operation or else the
    /// transform's own fusion space and block layout, the exact destination
    /// length, and unique destination ownership ([`Error::DestinationShared`]).
    /// Every one of them, and the structure compilation, happens before the
    /// device lease is taken; the placement check is the first thing under
    /// it. After a successful replay the exact source/destination layout pair
    /// is admitted on the Runtime, so a second call with the same pair skips
    /// the layout derivation — the same store, and the same admitted path, a
    /// Host call would take.
    ///
    /// There is no identity short circuit, because Host has none here: an
    /// identity axis list still writes `alpha * self + beta * destination`.
    ///
    /// `beta` rides the `dot_general` epilogue of the move or scatter that
    /// writes each destination layout, once per element; the layouts no move
    /// writes become `beta * destination` through a zero-source move (zeros
    /// for `beta = 0`, which never reads the destination; nothing for
    /// `beta = 1`).
    ///
    /// # Cost
    ///
    /// A warm call transfers nothing in either direction and allocates no
    /// device buffer: 0 H2D, 0 D2H, 0 device allocations. Unlike the returning
    /// [`Self::permute`] it pays no output initialisation. The first call for
    /// a given structure uploads that structure's coefficient payload once,
    /// may grow the pack/scatter workspace once, and reserves the context zero
    /// template; that warm contract holds only while this Runtime's Host
    /// transform store admits the structure, as for [`Self::permute`].
    /// `alpha == 0` writes `beta * destination` over every written layout
    /// through the same context zero template, so the first zero-scale call on
    /// a context whose template is still short pays one upload — once per
    /// context, not per call.
    ///
    /// # Numerics
    ///
    /// [`Self::permute`]'s, with the caller scale applied as the executor's
    /// rule: a Single block rounds as `alpha * (c * x)` where the Host folds
    /// the scales and rounds as `(alpha * c) * x`; Multi blocks agree in order
    /// with the Host (`alpha * (U x)`). `alpha == 0` — `-0.0` included, by
    /// IEEE comparison — does not read the source (VectorInterface's
    /// `scale(x, 0) = zero(x)`), as on Host; only the sign of an exact zero
    /// may differ.
    ///
    /// # Errors
    ///
    /// The Host variants and messages, with the storage noun naming the
    /// placement (as for [`Self::contract_into`]), plus
    /// [`Error::PlacementMismatch`] for a payload on another device.
    /// Validation and plan-construction failures leave `destination`
    /// untouched — no byte of it is written before the last rejection is
    /// decided. A backend error after replay begins may leave it partially
    /// written.
    ///
    /// # Source compatibility
    ///
    /// As for [`Self::permute`], the `cuda` feature adds these names to a
    /// second `impl`, so the *path* forms such as `TensorMap::permute_into`
    /// become ambiguous (`E0034`). Method-call syntax and
    /// `TensorMap::<R, D>::permute_into` keep working.
    pub fn permute_into(
        &self,
        codomain_axes: &[usize],
        domain_axes: &[usize],
        destination: &mut Self,
        alpha: D,
        beta: D,
    ) -> Result<(), Error> {
        self.tree_transform_into_cuda(
            destination,
            alpha,
            beta,
            |_, _| {
                Ok(TreeTransformOperation::permute(
                    codomain_axes.iter().copied(),
                    domain_axes.iter().copied(),
                ))
            },
            |operation| {
                tree_operation_matches_axes(
                    operation,
                    TreeTransformOperationKind::Permute,
                    codomain_axes,
                    domain_axes,
                )
            },
        )
    }

    /// `destination = alpha * self.braid(codomain_axes, domain_axes, levels) +
    /// beta * destination` on the device: the Host [`TensorMap::braid_into`].
    /// Validation, cost, numerics and failure behavior are
    /// [`Self::permute_into`]'s.
    pub fn braid_into(
        &self,
        codomain_axes: &[usize],
        domain_axes: &[usize],
        levels: &[usize],
        destination: &mut Self,
        alpha: D,
        beta: D,
    ) -> Result<(), Error> {
        let operation = braid_operation(
            self.rank(),
            self.codomain_rank(),
            codomain_axes,
            domain_axes,
            levels,
        )?;
        let expected = operation.clone();
        self.tree_transform_into_cuda(
            destination,
            alpha,
            beta,
            |_, _| Ok(operation),
            |admitted| *admitted == expected,
        )
    }

    /// `destination = alpha * self.transpose(codomain_axes, domain_axes) +
    /// beta * destination` on the device. Validation, cost, numerics and
    /// failure behavior are [`Self::permute_into`]'s.
    pub fn transpose_into(
        &self,
        codomain_axes: &[usize],
        domain_axes: &[usize],
        destination: &mut Self,
        alpha: D,
        beta: D,
    ) -> Result<(), Error> {
        self.tree_transform_into_cuda(
            destination,
            alpha,
            beta,
            |source, _| {
                with_planar_axes(
                    source.codomain_rank(),
                    source.rank(),
                    PlanarRequestKind::Explicit {
                        codomain_axes,
                        domain_axes,
                    },
                    |codomain_axes, domain_axes| {
                        Ok(TreeTransformOperation::transpose(
                            codomain_axes.iter().copied(),
                            domain_axes.iter().copied(),
                        ))
                    },
                )
            },
            |operation| {
                tree_operation_matches_axes(
                    operation,
                    TreeTransformOperationKind::Transpose,
                    codomain_axes,
                    domain_axes,
                )
            },
        )
    }

    /// `destination = alpha * self.repartition(destination.codomain_rank()) +
    /// beta * destination` on the device. Validation, cost, numerics and
    /// failure behavior are [`Self::permute_into`]'s.
    pub fn repartition_into(&self, destination: &mut Self, alpha: D, beta: D) -> Result<(), Error> {
        let source_codomain_rank = self.codomain_rank();
        let source_rank = self.rank();
        let destination_codomain_rank = destination.codomain_rank();
        self.tree_transform_into_cuda(
            destination,
            alpha,
            beta,
            |source, destination| {
                if destination.rank() != source.rank() {
                    return Err(Error::InvalidArgument(format!(
                        "repartition destination rank {} does not match source rank {}",
                        destination.rank(),
                        source.rank()
                    )));
                }
                with_planar_axes(
                    source.codomain_rank(),
                    source.rank(),
                    PlanarRequestKind::Repartition {
                        num_codomain: destination.codomain_rank(),
                    },
                    |codomain_axes, domain_axes| {
                        Ok(TreeTransformOperation::transpose(
                            codomain_axes.iter().copied(),
                            domain_axes.iter().copied(),
                        ))
                    },
                )
            },
            |operation| {
                let planar_axis = |position: usize| {
                    if position < source_codomain_rank {
                        position
                    } else {
                        source_rank - 1 - (position - source_codomain_rank)
                    }
                };
                operation.kind() == TreeTransformOperationKind::Transpose
                    && operation
                        .codomain_permutation()
                        .iter()
                        .copied()
                        .eq((0..destination_codomain_rank).map(planar_axis))
                    && operation
                        .domain_permutation()
                        .iter()
                        .copied()
                        .eq((destination_codomain_rank..source_rank)
                            .rev()
                            .map(planar_axis))
            },
        )
    }

    /// One admission and replay boundary for every typed device `*_into`
    /// tree transform,
    /// mirroring Host `tree_transform_into` step for step.
    ///
    /// The destination is the caller's device buffer, so — unlike the
    /// returning [`Self::tree_transform_cuda`], which owns a freshly uploaded
    /// output and may therefore spend it before the executor's layout verdict
    /// — nothing here may write before every rejection has been decided. That
    /// holds without a new preflight: the executor decides Stage A, the
    /// prepared-structure layout expressibility and Stage C before its first
    /// submission, so a rejected replay leaves the destination byte-identical.
    ///
    /// Lock order is [`Self::tree_transform_cuda`]'s: the pooled Host context
    /// lease that compiles the structure is dropped before the device lease,
    /// and nothing under the device lease leases again.
    fn tree_transform_into_cuda(
        &self,
        destination: &mut Self,
        alpha: D,
        beta: D,
        operation: impl FnOnce(&Self, &Self) -> Result<TreeTransformOperation, Error>,
        admitted_operation_matches: impl FnMut(&TreeTransformOperation) -> bool,
    ) -> Result<(), Error> {
        if !self.runtime.same_runtime(&destination.runtime) {
            return Err(Error::RuntimeMismatch);
        }
        let identity = TypedSectorAdmission::typed_rule_identity(self.provider());
        if identity != TypedSectorAdmission::typed_rule_identity(destination.provider()) {
            return Err(Error::RuleMismatch);
        }

        // Same error kinds, wording and order as Host; only the storage noun
        // names the placement, as in `contract_into`.
        // A lazy adjoint source is rejected rather than lowered: Host rejects
        // it here too, because lowering would produce a result shaped like the
        // adjoint's parent, not like the caller's destination.
        let (source_body, source_storage) =
            match &self.repr {
                TypedTensorRepr::Owned(body) => match body.data.as_ref() {
                    TypedData::Dense(storage) => (body, storage),
                    TypedData::Diagonal(_) => return Err(Error::InvalidArgument(
                        "typed destination tree transform requires an ordinary dense CUDA source"
                            .to_string(),
                    )),
                },
                TypedTensorRepr::Adjoint(_) => {
                    return Err(Error::InvalidArgument(
                        "typed destination tree transform requires an ordinary dense CUDA source"
                            .to_string(),
                    ))
                }
            };
        let (destination_body, destination_storage) = match &destination.repr {
            TypedTensorRepr::Owned(body) => match body.data.as_ref() {
                TypedData::Dense(storage) => (body, storage),
                TypedData::Diagonal(_) => {
                    return Err(Error::InvalidArgument(
                        "destination must use ordinary dense CUDA storage".to_string(),
                    ))
                }
            },
            TypedTensorRepr::Adjoint(_) => {
                return Err(Error::InvalidArgument(
                    "destination must use ordinary dense CUDA storage".to_string(),
                ))
            }
        };
        if Arc::ptr_eq(&source_body.data, &destination_body.data) {
            return Err(Error::InvalidArgument(
                "destination storage must not alias an input".to_string(),
            ));
        }

        let admitted_operation = self.runtime.admitted_tree_pair_operation(
            &identity,
            &source_body.space,
            &destination_body.space,
            admitted_operation_matches,
        );
        let exact_layout_admitted = admitted_operation.is_some();
        let operation = match admitted_operation {
            Some(operation) => operation,
            None => operation(self, destination)?,
        };
        if !exact_layout_admitted {
            let expected = source_body
                .space
                .transformed_multiplicity_free(&operation)?;
            if destination_body.space.space() != expected.space() {
                return Err(Error::InvalidArgument(
                    "destination fusion space or block layout does not match the operation result"
                        .to_string(),
                ));
            }
        }
        let required = destination_body.space.space().required_len()?;
        let actual = TensorStorage::len(destination_storage);
        if actual != required {
            return Err(Error::InvalidArgument(format!(
                "destination storage length {actual} does not match required length {required}"
            )));
        }
        if Arc::strong_count(destination_body) != 1
            || Arc::strong_count(&destination_body.data) != 1
        {
            return Err(Error::DestinationShared);
        }
        let destination_placement = destination_storage.placement();

        let source_structure = Arc::clone(source_body.space.space().structure());
        let destination_structure = Arc::clone(destination_body.space.space().structure());
        // Host compiles the structure inside its replay entry point, under the
        // same pooled context lease; here it must happen before the device
        // lease is taken, as in `tree_transform_cuda`. The destination's
        // provider is the authority, exactly as Host's
        // `tree_transform_dyn_into_ref` call passes it.
        let structure = {
            let mut lease = self.runtime.lease_context()?;
            lease
                .context()
                .multiplicity_free_lane::<D>()?
                .tree_context_mut()
                .compile_tree_pair_structure(
                    destination_body.space.provider(),
                    &operation,
                    &destination_structure,
                    &source_structure,
                )?
        };

        {
            let mut lease = self.runtime.lease_cuda()?;
            let (cuda, executor) = lease.split();
            let expected_placement = Placement::Cuda(cuda.device());
            if source_storage.placement() != expected_placement
                || destination_placement != expected_placement
            {
                return Err(Error::PlacementMismatch);
            }
            let TypedTensorRepr::Owned(destination_body) = &mut destination.repr else {
                return Err(internal_layout_error(
                    "ordinary CUDA destination checked above",
                ));
            };
            let destination_body = Arc::get_mut(destination_body).ok_or_else(|| {
                internal_layout_error("unique CUDA destination body checked above")
            })?;
            let destination_data = Arc::get_mut(&mut destination_body.data).ok_or_else(|| {
                internal_layout_error("unique CUDA destination payload checked above")
            })?;
            let TypedData::Dense(destination_data) = destination_data else {
                return Err(internal_layout_error(
                    "dense CUDA destination checked above",
                ));
            };
            executor.replay(
                cuda,
                &structure,
                &destination_structure,
                &source_structure,
                destination_data,
                source_storage,
                alpha,
                CudaTreeTransformDestination::Axpby(beta),
            )?;
        }
        if !exact_layout_admitted {
            self.runtime.admit_exact_tree_pair_layout(
                identity,
                &operation,
                &source_body.space,
                destination.logical_space(),
            );
        }
        Ok(())
    }

    /// TensorKit `twist(t, inds)` on a device tensor: each fusion-tree block
    /// is multiplied by the product over `legs` (flat leg indices, codomain
    /// first) of that leg's ribbon-twist eigenvalue.
    ///
    /// Same branches, in the same order, as the Host `twist`: an out-of-range
    /// leg is rejected before the empty-list short circuit, a `NoBraiding`
    /// provider is preflighted for non-unit legs, a twist that is the identity
    /// on every block returns a body-sharing clone without touching the
    /// device, and a lazy adjoint redirects through its parent with the
    /// inverse operation. A twist does not change the space, so the result
    /// keeps the receiver's layout.
    ///
    /// Reference: TensorKit `twist!`
    /// (`src/tensors/indexmanipulations.jl:62-77` @`cfaa073`), which tests
    /// `has_shared_twist` and then `scale!(t[f₁,f₂], θ)` per fusion-tree
    /// block. QSpace has no ribbon-twist path at all — a concrete absence, not
    /// an unexamined one — so the block-scaling decomposition follows
    /// TensorKit and only the dataflow below is Rust/device specific.
    ///
    /// # Cost
    ///
    /// θ is built on the Host from the block structure alone, before the
    /// device lease, and reaches the kernel as the contraction descriptor's
    /// own scale: for every provider this impl admits (`Scalar = f64`) a
    /// ribbon twist is a real sign, never zero, so the descriptor cannot let
    /// CUDA skip a source read: NaN and real infinities propagate as on Host.
    /// An infinite *complex* entry instead becomes NaN in both components,
    /// because the device always multiplies where Host copies a factor-1 block
    /// (the #1301 deviation, disclosed by `cuda_region_axpby`); `f64` and every
    /// finite payload are exact. No θ table is uploaded. A warm call therefore transfers only the #740 output
    /// initialisation — one H2D of the output bytes — downloads nothing,
    /// allocates exactly one device buffer and submits one strided move per
    /// non-empty block. Residual: a whole-buffer bitwise device copy followed
    /// by in-place per-block scaling would submit fewer kernels, but Tenferro
    /// (0.5.0 and 0.6.0) has no in-place strided scale (`cuda_region_axpby` cannot alias
    /// its source and destination), so it is not expressible today.
    ///
    /// Flat elements that belong to no block — only reachable under a padded
    /// expert layout — are zero here, where Host copies the source bytes
    /// through; device structural results have carried that convention since
    /// the transform leaf.
    ///
    /// # Errors
    ///
    /// [`Error::UnsupportedOnDevice`] for a compact (diagonal) device payload:
    /// the Host's compact-spectrum arm needs a representation `to_cuda` never
    /// produces, and a silent dense fallback would hide the cost.
    ///
    /// Under the `cuda` feature `TensorMap<R, D, CudaStorage<D>>` and the Host
    /// `TensorMap<R, D>` both have a `twist`; they are distinct inherent
    /// methods on distinct types, so a path-form call must name the storage.
    pub fn twist(&self, legs: &[usize], direction: Direction) -> Result<Self, Error> {
        self.twist_with_inverse_cuda(legs, direction.is_inverse())
    }

    /// TensorKit `tensortrace!` on a device tensor: the Host
    /// [`TensorMap::trace_pairs`] semantics — the categorical partial trace
    /// of each mutually dual `(lhs, rhs)` pair, remaining legs kept in order
    /// and on their side, owned or lazy-adjoint — executed on the device.
    ///
    /// # One planning authority
    ///
    /// The destination space and the trace structure come from the compile
    /// the Host trace runs: the valid (source tree, destination tree) terms,
    /// each with the recoupling coefficient times `dim(c)/dim(a_1)` times the
    /// twist of every non-dual traced leg after the first (the fermionic
    /// supertrace), and every block stride. The device only replays it: one
    /// contraction per term, reading the source block through one merged
    /// diagonal axis per pair (extent `t_k`, stride `s_lhs + s_rhs`; pairs are
    /// never merged with each other) against the coefficient repeated (the
    /// context ones template for a unit coefficient, the scaled template
    /// otherwise), so each traced element is scaled before the sum as on the
    /// Host, accumulated (`beta = 1`) into the
    /// zero-initialised output, so a destination block with several producers
    /// receives their sum. A lazy adjoint traces its parent with the Host's
    /// parent axes and a conjugated read.
    ///
    /// # Cost
    ///
    /// A warm call transfers only the #740 output initialisation: one H2D of
    /// `required_len * size_of::<D>()` bytes, one device allocation (the
    /// output), no download and no new cuTENSOR plan while the trace's
    /// distinct term signatures fit the plan-cache budget the transform
    /// executor raises the bound under (about 585 plans together with the
    /// prepared transforms; beyond it a warm call rebuilds plans). The first call whose
    /// largest traced extent `prod t_k` exceeds the resident ones or scaled
    /// template grows it once (one upload each, reported by
    /// [`crate::prelude::CudaTreeTransformStats::context_scalar_operand_bytes`]).
    /// One submission per term with a non-zero coefficient, plus one device
    /// refill of the scaled template per distinct non-unit coefficient; FLOPs
    /// are the Host's `sum_terms |out| * prod t_k` multiply-adds.
    ///
    /// # Numerics
    ///
    /// Device and Host agree to dtype tolerance, never bitwise: the sum order
    /// is cuTENSOR's. Both scale each traced element by `alpha * c` before
    /// the sum and skip a zero coefficient, as TensorKit's `_trace_permute!`
    /// and TensorOperations' `tensortrace!` do (#1438).
    ///
    /// # Errors
    ///
    /// In the Host's order, all before any device work: the Host's
    /// [`Error::InvalidArgument`] for a malformed pair list and its errors for
    /// legs that are not mutually dual; [`Error::UnsupportedOnDevice`] for a
    /// compact (diagonal) device payload, where the Host takes its
    /// compact-spectrum arm; the Host compile's own errors;
    /// [`Error::PlacementMismatch`]. A rejected call leaves the device and the
    /// Runtime unchanged.
    pub fn trace_pairs(&self, pairs: &[(usize, usize)]) -> Result<Self, Error> {
        match self.prepare_trace_pairs(pairs)? {
            Some(trace) => trace.execute(),
            None => Ok(self.clone()),
        }
    }

    /// `destination = alpha * self.trace_pairs(pairs) + beta * destination` on
    /// the device: the Host [`TensorMap::trace_pairs_into`].
    ///
    /// TensorKit's `_trace_permute!` order: every destination layout first
    /// becomes `beta * destination` through a zero-source region move (zeros
    /// for `beta = 0`, which never reads the destination; nothing for
    /// `beta = 1`), then every term accumulates as in [`Self::trace_pairs`].
    /// An empty `pairs` is [`Self::axpby_into`].
    ///
    /// # Errors
    ///
    /// [`Self::trace_pairs`]'s, all before any device work, then
    /// [`Error::RuntimeMismatch`] / [`Error::RuleMismatch`] against the
    /// destination, [`Error::InvalidArgument`] for a destination that is not
    /// owned dense CUDA storage, aliases the source, or has the wrong space,
    /// layout or length, [`Error::DestinationShared`] and
    /// [`Error::PlacementMismatch`]. A rejected call leaves `destination`
    /// untouched.
    pub fn trace_pairs_into(
        &self,
        pairs: &[(usize, usize)],
        destination: &mut Self,
        alpha: D,
        beta: D,
    ) -> Result<(), Error> {
        if !self.runtime.same_runtime(&destination.runtime) {
            return Err(Error::RuntimeMismatch);
        }
        if TypedSectorAdmission::typed_rule_identity(self.provider())
            != TypedSectorAdmission::typed_rule_identity(destination.provider())
        {
            return Err(Error::RuleMismatch);
        }
        let Some(trace) = self.prepare_trace_pairs(pairs)? else {
            return self.axpby_into(destination, alpha, beta);
        };
        let destination_storage =
            unique_cuda_destination(destination, &self.storage_body().data, trace.space.space())?;
        if destination_storage.placement() != trace.source.placement() {
            return Err(Error::PlacementMismatch);
        }
        let TypedTensorRepr::Owned(destination_body) = &mut destination.repr else {
            return Err(internal_layout_error(
                "ordinary CUDA destination checked above",
            ));
        };
        let destination_data = Arc::get_mut(destination_body)
            .and_then(|body| Arc::get_mut(&mut body.data))
            .ok_or_else(|| internal_layout_error("unique CUDA destination checked above"))?;
        let TypedData::Dense(destination_data) = destination_data else {
            return Err(internal_layout_error(
                "dense CUDA destination checked above",
            ));
        };
        let mut lease = self.runtime.lease_cuda()?;
        let (cuda, transforms) = lease.split();
        tenet_tensors::tensortrace_fusion_structure_into_on_cuda(
            cuda,
            transforms,
            &trace.structure,
            trace.space.space().structure(),
            destination_data,
            trace.source_space.space().structure(),
            trace.source,
            alpha,
            beta,
        )?;
        Ok(())
    }

    /// Everything [`Self::trace_pairs`] decides before the device lease — the
    /// pair list, duality, storage kind, the Host compile and placement —
    /// with the compiled structure kept for [`CudaTracePairs::execute`];
    /// `None` for an empty pair list. Lets `tensor!` decide every trace of a
    /// network before the first one allocates, at no second compile.
    #[doc(hidden)]
    pub fn prepare_trace_pairs(
        &self,
        pairs: &[(usize, usize)],
    ) -> Result<Option<CudaTracePairs<'_, R, D>>, Error> {
        let Some(TracePairAxes {
            output_axes,
            destination_codomain_rank,
            trace_lhs,
            trace_rhs,
        }) = trace_pair_axes(self.rank(), self.codomain_rank(), pairs)?
        else {
            return Ok(None);
        };
        let mapped_output_axes;
        let mapped_trace_lhs;
        let mapped_trace_rhs;
        let (source_space, source_data, axes) = match &self.repr {
            TypedTensorRepr::Owned(body) => (
                &body.space,
                body.data.as_ref(),
                tenet_tensors::TensorTraceAxisSpec::new(&output_axes, &trace_lhs, &trace_rhs),
            ),
            TypedTensorRepr::Adjoint(view) => {
                let parent = view.parent.space.space();
                mapped_output_axes =
                    logical_adjoint_axes_to_parent(parent.nout(), parent.nin(), &output_axes);
                mapped_trace_lhs =
                    logical_adjoint_axes_to_parent(parent.nout(), parent.nin(), &trace_lhs);
                mapped_trace_rhs =
                    logical_adjoint_axes_to_parent(parent.nout(), parent.nin(), &trace_rhs);
                (
                    &view.parent.space,
                    view.parent.data.as_ref(),
                    tenet_tensors::TensorTraceAxisSpec::new_with_conjugation(
                        &mapped_output_axes,
                        &mapped_trace_lhs,
                        &mapped_trace_rhs,
                        true,
                    ),
                )
            }
        };
        let homspace = tenet_tensors::tensortrace_fusion_dyn_selected_homspace_checked(
            source_space,
            axes,
            destination_codomain_rank,
        )?;
        let space = source_space.derive_from_final_homspace(homspace)?;
        let TypedData::Dense(source) = source_data else {
            return Err(Error::UnsupportedOnDevice(
                "trace_pairs requires dense CUDA storage".to_string(),
            ));
        };
        let structure = tenet_tensors::TensorTraceFusionStructure::compile_fusion_dyn_checked(
            &space,
            source_space,
            axes,
        )?;
        if source.placement() != Placement::Cuda(self.runtime.cuda_device_ordinal_checked()?) {
            return Err(Error::PlacementMismatch);
        }
        Ok(Some(CudaTracePairs {
            runtime: &self.runtime,
            source_space,
            source,
            space,
            structure,
        }))
    }

    fn twist_with_inverse_cuda(&self, legs: &[usize], inverse: bool) -> Result<Self, Error> {
        let rank = self.rank();
        let name = if inverse { "inverse twist" } else { "twist" };
        if let Some(&leg) = legs.iter().find(|&&leg| leg >= rank) {
            return Err(Error::InvalidArgument(format!(
                "{name} leg {leg} out of range for rank {rank}"
            )));
        }
        if legs.is_empty() {
            return Ok(self.clone());
        }
        let provider = self.logical_space().provider();
        reject_unbraided_nonunit_legs(
            provider,
            self.logical_space().space().homspace(),
            legs,
            name,
            true,
        )?;
        // Host order: the lazy adjoint lowers onto its parent with the
        // opposite direction rather than materializing itself. `.adjoint()` is
        // lease-free, so this recursion never nests a device lease.
        if let TypedTensorRepr::Adjoint(view) = &self.repr {
            let parent = Self {
                runtime: self.runtime.clone(),
                repr: TypedTensorRepr::Owned(Arc::clone(&view.parent)),
            };
            let axes = logical_adjoint_axes_to_parent(
                view.parent.space.space().nout(),
                view.parent.space.space().nin(),
                legs,
            );
            return parent.twist_with_inverse_cuda(&axes, !inverse)?.adjoint();
        }
        let nout = self.codomain_rank();
        let space = self.logical_space().clone();
        // Ahead of the storage kind, as on Host, where an all-ones twist
        // leaves even a compact spectrum untouched.
        if twist_is_identity_over_blocks(provider, space.space().structure(), nout, legs)? {
            return Ok(self.clone());
        }
        let src = self.direct_cuda_storage(name)?;
        let structure = space.space().structure();
        let required_len = space.space().required_len()?;
        // The whole θ table, on the Host, before the lease: one region per
        // block with that block's factor as the descriptor scale. A block
        // whose factor is 1 still has to be moved — the output starts as
        // zeros, where the Host started from a copy of the source.
        let blocks = (0..structure.block_count())
            .map(|index| {
                let block = structure.block(index)?;
                let factor = match block.key() {
                    BlockKey::FusionTree(key) => {
                        twist_block_factor(provider, key, nout, legs, inverse)
                    }
                    _ => 1.0,
                };
                let region = tenet_dense::CudaRegion::new(
                    block.shape().to_vec(),
                    block.strides().to_vec(),
                    block.offset(),
                )
                .and_then(|region| region.validate_as_destination(name).map(|()| region))
                .map_err(|err| Error::from(tenet_tensors::OperationError::Dense(err)))?;
                Ok((region, D::from_real(factor)))
            })
            .collect::<Result<Vec<_>, Error>>()?;

        let mut lease = self.runtime.lease_cuda()?;
        let cuda = &mut *lease;
        Self::validate_cuda_owned_metadata(
            Placement::Cuda(cuda.device()),
            src.placement(),
            required_len,
            src.len(),
        )?;
        // ponytail: #740 — the device seam still initializes an output by
        // uploading zeros; replace only with a measured native allocation.
        let mut dst = CudaStorage::upload_owned(cuda, vec![D::from_real(0.0); required_len])?;
        for (region, factor) in &blocks {
            // A ribbon twist is never zero, so the factor rides the descriptor
            // scale: no coefficient buffer, and CUDA still has to read the
            // source rather than skipping it as it may for a zero alpha. That
            // keeps NaN and real infinities propagating as on Host; an infinite
            // complex entry is the disclosed #1301 difference, because Host
            // bit-copies a factor-1 block where this always multiplies.
            tenet_dense::cuda_region_axpby::<D>(
                cuda,
                &src.0,
                region,
                false,
                *factor,
                tenet_dense::CudaRegionCoefficient::One,
                tenet_dense::CudaRegionBeta::Overwrite,
                &mut dst.0,
                region,
            )
            .map_err(|err| Error::from(tenet_tensors::OperationError::Dense(err)))?;
        }
        drop(lease);
        Ok(Self {
            runtime: self.runtime.clone(),
            repr: owned_repr(TypedTensorBody::dense(space, dst)),
        })
    }
}

impl<R, D> TensorMap<R, D>
where
    D: TensorScalar,
{
    pub(super) fn fusion_operand(&self) -> tenet_tensors::FusionOperand<'_> {
        match &self.repr {
            TypedTensorRepr::Owned(body) => {
                tenet_tensors::FusionOperand::direct(body.space.space())
            }
            TypedTensorRepr::Adjoint(view) => {
                tenet_tensors::FusionOperand::adjoint(view.parent.space.space())
            }
        }
    }

    pub(super) fn fusion_operand_and_data(
        &self,
    ) -> (tenet_tensors::FusionOperand<'_>, std::borrow::Cow<'_, [D]>) {
        match &self.repr {
            TypedTensorRepr::Owned(body) => (
                tenet_tensors::FusionOperand::direct(body.space.space()),
                body.materialized_dense_data(),
            ),
            TypedTensorRepr::Adjoint(view) => (
                tenet_tensors::FusionOperand::adjoint(view.parent.space.space()),
                std::borrow::Cow::Borrowed(view.parent_data()),
            ),
        }
    }

    /// Checks that this tensor can have leg `axis` exchanged for `expected`.
    fn require_selected_leg(
        &self,
        axis: usize,
        expected: &GradedSpace<R>,
        operation: &str,
    ) -> Result<(), TypedFacadeError<R>>
    where
        R: TypedSectorAdmission,
        R::Mode: TypedTensorModeDispatch<R>,
    {
        if self.network_has_compact_payload() {
            return Err(Error::InvalidArgument(format!(
                "{operation} requires a dense Host payload"
            ))
            .into());
        }
        require_selected_leg_of(self.logical_space(), axis, expected, operation)
    }

    /// This tensor's hom-space with leg `axis` replaced, built into a root
    /// layout of the same provider.
    fn root_with_replaced_leg(
        &self,
        axis: usize,
        replacement: &SectorLeg,
    ) -> Result<BoundDynamicFusionMapSpace<R>, TypedFacadeError<R>>
    where
        R: TypedSectorAdmission,
        R::Mode: TypedTensorRootDispatch<R>,
    {
        space_with_replaced_legs(self.logical_space(), |candidate| {
            (candidate == axis).then_some(replacement)
        })
    }

    /// Restricts each listed leg to the subspace its selection names.
    ///
    /// `legs` is a set of `(axis, selection)` pairs on distinct axes; a
    /// single-leg restriction is `&[(axis, &selection)]`. Each pair is
    /// composition with the inclusion isometry `ι_σ` of [`LegSelection`] on
    /// that leg — `ι_σ^† ∘ t` for a codomain leg, `t ∘ ι_σ` for a domain leg.
    /// Isometries on distinct legs commute, so the result equals any order of
    /// one-pair restrictions, bit for bit. The tensor stays invariant, every
    /// leg keeps its dual flag, and every surviving fusion tree keeps its
    /// inner lines and vertices: `ι_σ` is the identity on irreps, so no
    /// structural coefficient, braid or fermionic sign enters. Selecting a
    /// single degeneracy index of sector `q` leaves a leg equal to the
    /// one-dimensional space of `q`, so the charge stays explicit on the leg.
    ///
    /// A compact diagonal `bond <- bond` map (the `s` of an SVD, the `d` of
    /// an `eigh`) stays compact when both legs get the same selection,
    /// `&[(0, &sel), (1, &sel)]`: the result is again an endomorphism of the
    /// kept subspace (TensorKit `truncate_diagonal!`). This is the third of
    /// the three calls that turn a compact factorization plus a
    /// [`GradedSpace::find_truncated`] decision into a truncated one.
    ///
    /// [`Self::embed_leg`] is the adjoint: `restrict_leg` after `embed_leg` is
    /// the identity, `embed_leg` after `restrict_leg` is the orthogonal
    /// projector onto the subspace.
    ///
    /// # Cost
    ///
    /// One strided copy per destination block for all `k` legs at once,
    /// `O(selected payload)` data movement, no matrix multiplication and no
    /// recoupling. One payload allocation plus `O(rank + blocks)` structural
    /// work, independent of the degeneracy dimensions. (TensorKit pays a
    /// contraction with an explicit isometry tensor for an arbitrary leg;
    /// TeNeT addresses the degeneracy axes inside each reduced block
    /// directly.) A compact input copies only the `O(sum_c k'_c)` kept
    /// values; the discarded ones are never touched.
    ///
    /// Defined for Host payloads: device storage has no such method, so an
    /// unsupported placement is a compile-time absence rather than a runtime
    /// error. A lazy adjoint is read in place.
    ///
    /// # Errors
    ///
    /// [`Error::InvalidArgument`] when `legs` is empty, an axis is out of
    /// range or appears twice, or an axis is not the leg its selection was
    /// built from; [`Error::RuleMismatch`] when a selection belongs to a
    /// different rule; [`Error::InvalidArgument`] for a compact diagonal
    /// payload with any set other than both legs and one selection (call
    /// [`Self::materialize`] first for a dense result). Nothing is allocated
    /// before every check has passed.
    pub fn restrict_leg(
        &self,
        legs: &[(usize, &LegSelection<R>)],
    ) -> Result<Self, TypedFacadeError<R>>
    where
        R: TypedSectorAdmission,
        R::Mode: TypedTensorRootDispatch<R>,
    {
        let _host_pool = self.runtime.enter_host_pool();
        require_restriction_set(self.logical_space(), legs)?;
        let compact = self
            .spectrum()
            .map(|spectrum| match legs {
                [(0, first), (1, second)] | [(1, second), (0, first)]
                    if first.entries == second.entries =>
                {
                    Ok((spectrum, *first))
                }
                _ => Err(TypedFacadeError::<R>::from(Error::InvalidArgument(
                    "restrict_leg: a compact diagonal stays compact only with one selection \
                     on both legs, &[(0, sel), (1, sel)]; call materialize() first for any \
                     other restriction"
                        .to_string(),
                ))),
            })
            .transpose()?;
        let destination = restricted_space(self.logical_space(), legs)?;
        if let Some((spectrum, selection)) = compact {
            let mut kept = Vec::with_capacity(selection.entries.len());
            for (sector, range) in &selection.entries {
                // Both lists are in canonical `SectorId` order, so this is a
                // lookup, not a scan. A violated order can only make the
                // search miss, which is the typed error below — never a match
                // on the wrong sector, because the key is compared.
                let values = spectrum
                    .binary_search_by_key(sector, |entry| entry.sector)
                    .ok()
                    .and_then(|index| spectrum[index].values.get(range.clone()))
                    .ok_or_else(|| {
                        TypedFacadeError::<R>::from(Error::InvalidArgument(format!(
                            "restrict_leg: the compact payload has no [{}, {}) for sector {:?}",
                            range.start, range.end, sector
                        )))
                    })?;
                kept.push(tenet_matrixalgebra::SectorSpectrum {
                    sector: *sector,
                    values: values.to_vec(),
                });
            }
            return Ok(self.with_spectrum_on(destination, kept));
        }
        let (source, source_data) = self.fusion_operand_and_data();
        let data = tenet_tensors::oriented_fusion_restrict_owned(
            destination.space().structure(),
            source,
            &source_data,
            &restriction_starts(self.rank(), legs),
        )
        .map_err(Error::from)?;
        Ok(Self {
            runtime: self.runtime.clone(),
            repr: owned_repr(TypedTensorBody::dense(destination, data)),
        })
    }

    /// Embeds leg `axis` back into the parent leg of `selection`, zero-padding
    /// the degeneracy indices the selection does not name.
    ///
    /// This is the adjoint of [`Self::restrict_leg`]: composition with `ι_σ`
    /// on a codomain leg and with `ι_σ^†` on a domain leg. The receiver's leg
    /// `axis` must equal [`LegSelection::subspace`]; the result carries
    /// [`LegSelection::parent`] on that axis. No target space is passed
    /// separately — the selection already owns both legs, so they cannot
    /// disagree.
    ///
    /// # Cost
    ///
    /// `O(source payload)` data movement into an allocator-zeroed output of
    /// `O(destination payload)`, no matrix multiplication and no recoupling.
    ///
    /// Defined for Host payloads only, as [`Self::restrict_leg`].
    ///
    /// # Errors
    ///
    /// [`Error::InvalidArgument`] when the payload is a compact diagonal one,
    /// `axis` is out of range, or `axis` is not [`LegSelection::subspace`];
    /// [`Error::RuleMismatch`] when the selection belongs to a different rule.
    /// Every block is preflighted before the first destination element is
    /// written.
    pub fn embed_leg(
        &self,
        axis: usize,
        selection: &LegSelection<R>,
    ) -> Result<Self, TypedFacadeError<R>>
    where
        R: TypedSectorAdmission,
        R::Mode: TypedTensorRootDispatch<R>,
    {
        self.require_selected_leg(axis, selection.subspace(), "embed_leg")?;
        let _host_pool = self.runtime.enter_host_pool();
        let destination = self.root_with_replaced_leg(axis, selection.parent().leg())?;
        let len = destination
            .space()
            .required_len()
            .map_err(Error::from)
            .map_err(TypedFacadeError::<R>::from)?;
        let mut data = tenet_tensors::zeroed_payload::<D>(len);
        let mut ranges: Vec<tenet_tensors::SectorRangeTable<'_>> = vec![None; self.rank()];
        ranges[axis] = Some(selection.entries.as_slice());
        let (source, source_data) = self.fusion_operand_and_data();
        tenet_tensors::fusion_scatter_add_assign(
            destination.space().structure(),
            &mut data,
            self.logical_space().space().structure(),
            source,
            &source_data,
            &ranges,
        )
        .map_err(Error::from)?;
        Ok(Self {
            runtime: self.runtime.clone(),
            repr: owned_repr(TypedTensorBody::dense(destination, data)),
        })
    }
    /// Restricts tensor-local logical degeneracy coordinates for network
    /// slicing without exposing provider allocation or storage orientation.
    ///
    /// `SectorId` is safe here only as an immediately validated, tensor-local
    /// argument: the authority id is decoded by this tensor's provider, dualized
    /// exactly once for a partner occurrence, and checked against the actual
    /// effective leg before any destination is allocated.
    #[doc(hidden)]
    pub fn network_restrict_degeneracies(
        &self,
        adjoint: bool,
        restrictions: &[NetworkDegeneracyRestriction],
    ) -> Result<Self, TypedFacadeError<R>>
    where
        R: TypedSectorAdmission,
        R::Mode: TypedTensorRootDispatch<R>,
    {
        let _host_pool = self.runtime.enter_host_pool();
        if matches!(
            &self.repr,
            TypedTensorRepr::Owned(body) if matches!(body.data.as_ref(), TypedData::Diagonal(_))
        ) {
            return Err(Error::InvalidArgument(
                "network degeneracy restriction requires dense Host payloads".to_string(),
            )
            .into());
        }
        let rank = self.rank();
        let codomain_rank = self.codomain_rank();
        let domain_rank = self.domain_rank();
        let homspace = self.logical_space().space().homspace();
        let mut seen = vec![false; rank];
        let mut staged = Vec::with_capacity(restrictions.len());

        // Validate the complete request before deriving a layout or allocating
        // payload. In particular, callers never reconstruct dual-sector ids.
        for restriction in restrictions {
            let NetworkDegeneracyRestriction {
                effective_axis,
                authority_sector,
                range,
                partner,
            } = restriction;
            if *effective_axis >= rank || seen[*effective_axis] {
                return Err(Error::InvalidArgument(format!(
                    "invalid or duplicate effective restriction axis {effective_axis} for rank {rank}"
                ))
                .into());
            }
            seen[*effective_axis] = true;
            if range.start >= range.end {
                return Err(Error::InvalidArgument(format!(
                    "degeneracy restriction must be nonempty, got [{}, {})",
                    range.start, range.end
                ))
                .into());
            }
            TypedSectorAdmission::try_decode_label(self.provider(), *authority_sector)
                .map_err(<R::Mode as TypedTensorModeDispatch<R>>::map_provider_error)?;
            let effective_sector = if *partner {
                TypedSectorAdmission::try_dual_id(self.provider(), *authority_sector)
                    .map_err(<R::Mode as TypedTensorModeDispatch<R>>::map_provider_error)?
            } else {
                *authority_sector
            };

            let (logical_axis, dualized) = if adjoint {
                if *effective_axis < domain_rank {
                    (codomain_rank + *effective_axis, false)
                } else {
                    (*effective_axis - domain_rank, true)
                }
            } else if *effective_axis < codomain_rank {
                (*effective_axis, false)
            } else {
                (*effective_axis, true)
            };
            let logical_sector = if dualized {
                TypedSectorAdmission::try_dual_id(self.provider(), effective_sector)
                    .map_err(<R::Mode as TypedTensorModeDispatch<R>>::map_provider_error)?
            } else {
                effective_sector
            };
            let logical_leg = if logical_axis < codomain_rank {
                &homspace.codomain().legs()[logical_axis]
            } else {
                &homspace.domain().legs()[logical_axis - codomain_rank]
            };
            let degeneracy = logical_leg.degeneracy(logical_sector).ok_or_else(|| {
                TypedFacadeError::<R>::from(Error::InvalidArgument(format!(
                    "sector {effective_sector:?} is absent from effective axis {effective_axis}"
                )))
            })?;
            if range.end > degeneracy {
                return Err(Error::InvalidArgument(format!(
                    "degeneracy restriction [{}, {}) exceeds axis {effective_axis} sector {effective_sector:?} degeneracy {degeneracy}",
                    range.start, range.end
                ))
                .into());
            }
            staged.push((
                logical_axis,
                logical_sector,
                range.start,
                range.end - range.start,
            ));
        }

        let restricted_product = |product: &FusionProductSpace, axis_base: usize| {
            product
                .legs()
                .iter()
                .enumerate()
                .map(|(axis, leg)| {
                    let logical_axis = axis_base + axis;
                    if let Some((_, sector, _, extent)) = staged
                        .iter()
                        .find(|(candidate, ..)| *candidate == logical_axis)
                    {
                        SectorLeg::try_new([(*sector, *extent)], leg.is_dual()).map_err(|error| {
                            TypedFacadeError::<R>::from(Error::InvalidArgument(error.to_string()))
                        })
                    } else {
                        Ok(leg.clone())
                    }
                })
                .collect::<Result<Vec<_>, TypedFacadeError<R>>>()
                .map(FusionProductSpace::new)
        };
        let restricted_homspace = FusionTreeHomSpace::new(
            restricted_product(homspace.codomain(), 0)?,
            restricted_product(homspace.domain(), codomain_rank)?,
        );
        let destination = <R::Mode as TypedTensorRootDispatch<R>>::build_root(
            Arc::clone(self.logical_space().provider_arc()),
            restricted_homspace,
        )?;
        // One single-entry table per restricted axis: this path always names
        // exactly one sector per axis, and the kernel reads the sector back
        // from each destination block's own key.
        let tables: Vec<[(SectorId, usize); 1]> = staged
            .iter()
            .map(|&(_, sector, start, _)| [(sector, start)])
            .collect();
        let mut starts: Vec<tenet_tensors::SectorStartTable<'_>> = vec![None; rank];
        for (table, &(axis, ..)) in tables.iter().zip(&staged) {
            starts[axis] = Some(table.as_slice());
        }
        let (source, source_data) = self.fusion_operand_and_data();
        let data = tenet_tensors::oriented_fusion_restrict_owned(
            destination.space().structure(),
            source,
            &source_data,
            &starts,
        )
        .map_err(Error::from)?;
        Ok(Self {
            runtime: self.runtime.clone(),
            repr: owned_repr(TypedTensorBody::dense(destination, data)),
        })
    }

    #[doc(hidden)]
    pub fn network_has_compact_payload(&self) -> bool {
        matches!(self.storage_body().data.as_ref(), TypedData::Diagonal(_))
    }

    #[doc(hidden)]
    pub fn network_zeros_from_effective_legs(
        &self,
        codomain: &[GradedSpace<R>],
        domain: &[GradedSpace<R>],
    ) -> Result<Self, TypedFacadeError<R>>
    where
        R: TypedSectorAdmission,
        R::Mode: TypedTensorRootDispatch<R>,
    {
        let identity = TypedSectorAdmission::typed_rule_identity(self.provider());
        if codomain
            .iter()
            .chain(domain)
            .any(|leg| TypedSectorAdmission::typed_rule_identity(leg.provider()) != identity)
        {
            return Err(Error::RuleMismatch.into());
        }
        let raw_domain = domain
            .iter()
            .map(GradedSpace::try_dual)
            .collect::<Result<Vec<_>, _>>()?;
        let homspace = FusionTreeHomSpace::new(
            FusionProductSpace::new(codomain.iter().map(|leg| leg.network_sector_leg().clone())),
            FusionProductSpace::new(
                raw_domain
                    .iter()
                    .map(|leg| leg.network_sector_leg().clone()),
            ),
        );
        let space = <R::Mode as TypedTensorRootDispatch<R>>::build_root(
            Arc::clone(self.logical_space().provider_arc()),
            homspace,
        )?;
        let len = space
            .space()
            .required_len()
            .map_err(Error::from)
            .map_err(TypedFacadeError::<R>::from)?;
        Ok(Self {
            runtime: self.runtime.clone(),
            repr: owned_repr(TypedTensorBody::dense(space, vec![D::from_real(0.0); len])),
        })
    }

    fn owned_cat_layout(&self) -> Result<CatOperandLayout<'_>, Error> {
        let space = self.logical_space().space();
        CatOperandLayout::owned(space.structure(), space.nout(), space.nin())
    }

    fn adjoint_logical_for_cat(&self) -> Result<Option<Self>, Error> {
        match &self.repr {
            TypedTensorRepr::Owned(_) => Ok(None),
            TypedTensorRepr::Adjoint(_) => self.materialized_tensor_uncached().map(Some),
        }
    }

    fn cat_operand(&self) -> Result<(CatOperandLayout<'_>, std::borrow::Cow<'_, [D]>), Error> {
        match &self.repr {
            TypedTensorRepr::Owned(body) => Ok((
                CatOperandLayout::owned(
                    body.space.space().structure(),
                    body.space.space().nout(),
                    body.space.space().nin(),
                )?,
                body.materialized_dense_data(),
            )),
            TypedTensorRepr::Adjoint(view) => Ok((
                CatOperandLayout::adjoint(
                    view.parent.space.space().structure(),
                    view.parent.space.space().nout(),
                    view.parent.space.space().nin(),
                )?,
                std::borrow::Cow::Borrowed(view.parent_data()),
            )),
        }
    }

    /// An owned dense copy with this tensor's space, values, provider,
    /// runtime and placement (TensorKit `copy`, or `TensorMap(d)` for a
    /// diagonal).
    ///
    /// The result is neither a lazy adjoint nor a compact diagonal, and its
    /// payload is freshly allocated, so writing to it never changes `self`.
    /// This is the remedy for operations that reject lazy or compact inputs,
    /// such as `*_into` sources and checked-Generic
    /// factorizations. A lazy adjoint is conjugate-transposed from its parent
    /// in one pass.
    ///
    /// Why not named `copy`: TensorKit's `copy(::DiagonalTensorMap)` stays
    /// diagonal, and `Clone` here is a shallow handle copy.
    ///
    /// # Cost
    ///
    /// One payload allocation of `required_len` elements and one pass over
    /// it, plus two fixed body wrappers. A compact diagonal is zero-filled and
    /// then writes its `O(Σ_c k_c)` values.
    ///
    /// ```
    /// use std::sync::Arc;
    /// use tenet::core::{U1FusionRule, U1Irrep};
    /// use tenet::typed::{GradedSpace, Runtime, TensorMap};
    ///
    /// let runtime = Runtime::builder().build()?;
    /// let v = GradedSpace::try_new(
    ///     Arc::new(U1FusionRule),
    ///     [(U1Irrep::new(0), 2), (U1Irrep::new(1), 1)],
    /// )?;
    /// let t: TensorMap<_, f64> = TensorMap::rand_with_seed(&runtime, [&v], [&v], 7)?;
    /// let adjoint = t.adjoint()?;
    /// let mut owned = adjoint.materialize()?;
    /// assert_eq!(owned.dense_data()?, adjoint.materialize()?.dense_data()?);
    /// owned.scale_assign(2.0);
    /// assert_ne!(owned.dense_data()?, adjoint.materialize()?.dense_data()?);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    ///
    /// # Errors
    ///
    /// [`Error::Operation`] only if an engine-internal layout invariant is
    /// broken.
    pub fn materialize(&self) -> Result<Self, Error> {
        let body = match &self.repr {
            TypedTensorRepr::Owned(body) => body,
            TypedTensorRepr::Adjoint(view) => return self.materialize_adjoint(view),
        };
        let data = match body.data.as_ref() {
            TypedData::Dense(data) => data.clone(),
            TypedData::Diagonal(spectrum) => {
                tenet_matrixalgebra::diagonal_bond_data(body.space.space(), spectrum, &|value| {
                    value
                })?
            }
        };
        Ok(self.with_data(data))
    }

    /// The payload a space-only rewrite (unit-leg insert/remove) may install
    /// in its new body: a dense payload is shared at pointer cost; a lazy
    /// adjoint or a compact spectrum goes through [`Self::materialize`] into
    /// a **fresh** dense payload (one copy) — the #613 Group 4 contract.
    ///
    /// Infallible for the reason [`TypedTensorBody::materialized_dense_data`]
    /// is: the diagonal fill is total on a bond space this module built from
    /// that same spectrum.
    pub(super) fn shareable_dense_payload(&self) -> Arc<TypedData<D>> {
        if let Some(body) = self.owned_body() {
            if matches!(body.data.as_ref(), TypedData::Dense(_)) {
                return Arc::clone(&body.data);
            }
        }
        let materialized = self
            .materialize()
            .expect("a pre-admitted typed tensor must materialize");
        Arc::clone(
            &materialized
                .owned_body()
                .expect("materialize returns an owned body")
                .data,
        )
    }

    /// Builds an operation-local logical tensor: a full receiver-sized
    /// logical payload, released with the operation. Prefer an oriented
    /// kernel or algebraic redirect when one implements the same semantics.
    pub(super) fn materialized_tensor_uncached(&self) -> Result<Self, Error> {
        let TypedTensorRepr::Adjoint(view) = &self.repr else {
            return Ok(self.clone());
        };
        self.materialize_adjoint(view)
    }

    fn materialize_adjoint(&self, view: &TypedAdjointView<R, D>) -> Result<Self, Error> {
        if view.borrowed {
            // Backstop: every refusing operation checks first, by name, with
            // `refuse_borrowed_view`.
            return Err(borrowed_view_unsupported("the operation"));
        }
        #[cfg(test)]
        observe_adjoint_materialization();
        let _host_pool = self.runtime.enter_host_pool();
        #[cfg(test)]
        UNCACHED_ADJOINT_MATERIALIZATIONS
            .set(UNCACHED_ADJOINT_MATERIALIZATIONS.get().saturating_add(1));
        let data = tenet_tensors::materialize_adjoint_data_dyn(
            view.parent.space.space(),
            view.logical_space.space(),
            view.parent_data(),
        )?;
        Ok(Self {
            runtime: self.runtime.clone(),
            repr: owned_repr(TypedTensorBody::dense(view.logical_space.clone(), data)),
        })
    }
}

impl<R, D> TensorMap<R, D>
where
    R: TypedSectorAdmission,
    R::Mode: TypedTensorRootDispatch<R>,
    D: TensorScalar,
{
    /// TensorKit `catdomain(t1, t2)` (`side = Side::Domain`) and
    /// `catcodomain(t1, t2)` (`side = Side::Codomain`): concatenate two tensor
    /// maps along their sole leg on `side`. The product spaces on the other
    /// side must match exactly; the two concatenated legs must share duality
    /// and are combined by direct sum `V = V1 ⊕ V2`; reduced data is copied
    /// into adjacent slabs per coupled sector (column slabs for the domain,
    /// row slabs for the codomain), `self` first.
    ///
    /// Rust uses one method (`t1.cat(&t2, side)`) because binary tensor
    /// operations in this API are methods and the two TensorKit functions are
    /// one operation that differs only in the side; the operand order matches
    /// TensorKit's free functions.
    ///
    /// Both operands share one `D`, so mixed-dtype widening is statically
    /// unrepresentable — widen with [`Self::convert`] first.
    /// A lazy adjoint is read from parent storage through the oriented copy
    /// plan without publishing a receiver-sized materialization. A compact
    /// diagonal operand is densified into an operation-local buffer on every
    /// call; nothing is retained.
    ///
    /// # Complexity
    ///
    /// One output admission and allocation plus a single
    /// `O(len(self) + len(other))` copy pass over the compiled per-sector slab
    /// plan. If an oriented geometry is conservatively declined, correctness
    /// falls back to operation-local materialization and retries the
    /// plan against the already-admitted output.
    ///
    /// # Errors
    ///
    /// [`Error::RuleMismatch`] on differing admitted rule identities and
    /// [`Error::RuntimeMismatch`] on differing runtimes, in that order; then
    /// [`Error::InvalidArgument`] for more than one leg on `side`, mismatched
    /// product spaces on the other side, or concatenated legs of opposite
    /// duality. Checked-Generic output-admission failures retain their typed
    /// provider error.
    ///
    /// An adjoint view `other` (`t.adjoint_view()`) returns
    /// [`Error::Unsupported`] when the concatenation plan cannot read it in
    /// place (non-monotone oriented regions): this operation would copy it.
    /// Pass `&t.adjoint()?.materialize()?` instead.
    pub fn cat<'a>(
        &self,
        other: impl Into<TensorRef<'a, R, D>>,
        side: Side,
    ) -> Result<Self, TypedFacadeError<R>> {
        let other = other.into().operand()?;
        let other = &*other;
        let lhs_space = self.logical_space().space();
        let rhs_space = other.logical_space().space();
        if lhs_space.admission().rule_identity() != rhs_space.admission().rule_identity() {
            return Err(TypedFacadeError::<R>::from(Error::RuleMismatch));
        }
        if !self.runtime.same_runtime(&other.runtime) {
            return Err(TypedFacadeError::<R>::from(Error::RuntimeMismatch));
        }
        let lhs = lhs_space.homspace();
        let rhs = rhs_space.homspace();
        let _host_pool = self.runtime.enter_host_pool();
        let (axis, homspace) = cat_homspace(
            lhs.codomain(),
            lhs.domain(),
            rhs.codomain(),
            rhs.domain(),
            side,
        )
        .map_err(TypedFacadeError::<R>::from)?;
        let space = <R::Mode as TypedTensorRootDispatch<R>>::build_root(
            Arc::clone(self.logical_space().provider_arc()),
            homspace,
        )?;
        let (lhs_layout, lhs_data) = self.cat_operand().map_err(TypedFacadeError::<R>::from)?;
        let (rhs_layout, rhs_data) = other.cat_operand().map_err(TypedFacadeError::<R>::from)?;
        let data = if let Some(plan) = compile_cat_plan(
            space.space().structure(),
            space.space().nout(),
            [lhs_layout, rhs_layout],
            axis,
            side,
        )
        .map_err(TypedFacadeError::<R>::from)?
        {
            plan.execute(&lhs_data, &rhs_data)
                .map_err(TypedFacadeError::<R>::from)?
        } else {
            // Why not recurse through `cat`: output admission has succeeded,
            // so retry only the local copy plan and never query the provider
            // or admit the same HomSpace a second time. Why the view is
            // refused only here: whether the plan declines is known only
            // after the output layout is admitted.
            other
                .refuse_borrowed_view("cat")
                .map_err(TypedFacadeError::<R>::from)?;
            // Only a lazy adjoint needs a logical payload; an owned operand
            // reuses the buffer read above, so a compact operand is not
            // densified a second time.
            let lhs_local = self
                .adjoint_logical_for_cat()
                .map_err(TypedFacadeError::<R>::from)?;
            let rhs_local = other
                .adjoint_logical_for_cat()
                .map_err(TypedFacadeError::<R>::from)?;
            let (lhs_layout, lhs_data) = match &lhs_local {
                Some(local) => local.cat_operand(),
                None => self
                    .owned_cat_layout()
                    .map(|layout| (layout, std::borrow::Cow::Borrowed(&*lhs_data))),
            }
            .map_err(TypedFacadeError::<R>::from)?;
            let (rhs_layout, rhs_data) = match &rhs_local {
                Some(local) => local.cat_operand(),
                None => other
                    .owned_cat_layout()
                    .map(|layout| (layout, std::borrow::Cow::Borrowed(&*rhs_data))),
            }
            .map_err(TypedFacadeError::<R>::from)?;
            let plan = compile_cat_plan(
                space.space().structure(),
                space.space().nout(),
                [lhs_layout, rhs_layout],
                axis,
                side,
            )
            .map_err(TypedFacadeError::<R>::from)?
            .ok_or_else(|| {
                TypedFacadeError::<R>::from(internal_layout_error(
                    "owned cat operands did not produce a copy plan",
                ))
            })?;
            plan.execute(&lhs_data, &rhs_data)
                .map_err(TypedFacadeError::<R>::from)?
        };
        Ok(Self {
            runtime: self.runtime.clone(),
            repr: owned_repr(TypedTensorBody::dense(space, data)),
        })
    }
}

impl<R, D, S> TensorMap<R, D, S>
where
    S: HostReadableStorage<D>,
{
    /// Whole reduced payload in the tensor's logical coupled-sector layout,
    /// borrowed from dense Host storage without copying.
    ///
    /// These are fusion-tree-indexed reduced block entries, not entries in the
    /// physical carrier basis. Use [`Self::to_physical_dense`] when the
    /// provider implements [`PhysicalFusionBasis`] and physical entries are
    /// required.
    ///
    /// ```
    /// use std::sync::Arc;
    ///
    /// use tenet::core::{U1FusionRule, U1Irrep};
    /// use tenet::typed::{Error, GradedSpace, Runtime, TensorMap};
    ///
    /// let runtime = Runtime::builder().build()?;
    /// let v = GradedSpace::try_new(Arc::new(U1FusionRule), [(U1Irrep::new(0), 2)])?;
    /// let t: TensorMap<_, f64> = TensorMap::rand_with_seed(&runtime, [&v], [&v], 1)?;
    /// assert_eq!(t.dense_data()?.len(), 4);
    /// let adjoint = t.adjoint()?;
    /// assert!(matches!(adjoint.dense_data(), Err(Error::Unsupported { .. })));
    /// assert_eq!(adjoint.materialize()?.dense_data()?.len(), 4);
    /// # Ok::<(), Error>(())
    /// ```
    ///
    /// # Errors
    ///
    /// [`Error::Unsupported`] with [`crate::error::Alternative::Materialize`]
    /// for a lazy adjoint or a compact diagonal, whose entries are not stored
    /// densely; call [`Self::materialize`] first. Never copies.
    pub fn dense_data(&self) -> Result<&[D], Error> {
        match &self.repr {
            TypedTensorRepr::Owned(body) => match &*body.data {
                TypedData::Dense(data) => Ok(data.as_slice()),
                TypedData::Diagonal(_) => Err(borrowed_view_unsupported("dense_data")),
            },
            TypedTensorRepr::Adjoint(_) => Err(borrowed_view_unsupported("dense_data")),
        }
    }
}

impl<R, D, S> TypedTensorBody<R, D, S>
where
    D: TensorScalar,
    S: HostReadableStorage<D>,
{
    /// The payload in the dense coupled layout: borrowed for a dense payload,
    /// densified into an operation-local buffer for a compact diagonal.
    ///
    /// Why not cached in the body: a retained densification is an implicit
    /// receiver-sized copy the caller never asked for (#1548). Every caller
    /// that reaches the compact arm documents the copy as part of its cost.
    pub(super) fn materialized_dense_data(&self) -> std::borrow::Cow<'_, [D]> {
        match &*self.data {
            TypedData::Dense(data) => std::borrow::Cow::Borrowed(data.as_slice()),
            TypedData::Diagonal(spectrum) => {
                #[cfg(test)]
                DIAGONAL_MATERIALIZATIONS.set(DIAGONAL_MATERIALIZATIONS.get().saturating_add(1));
                std::borrow::Cow::Owned(
                    tenet_matrixalgebra::diagonal_bond_data(
                        self.space.space(),
                        spectrum,
                        &|value| value,
                    )
                    .expect("diagonal fill is total on the stored bond space"),
                )
            }
        }
    }
}

impl<R, D, S> TypedAdjointView<R, D, S>
where
    S: HostReadableStorage<D>,
{
    /// The parent payload, which is dense by the [`TypedAdjointView::new`]
    /// invariant.
    pub(super) fn parent_data(&self) -> &[D] {
        match &*self.parent.data {
            TypedData::Dense(data) => data.as_slice(),
            TypedData::Diagonal(_) => {
                unreachable!("TypedAdjointView::new admits only dense parents")
            }
        }
    }
}

impl<R, D> TensorMap<R, D>
where
    R: TypedSectorAdmission,
    R::Mode: TypedTensorRootDispatch<R>,
    D: TensorScalar,
{
    /// Builds a compact diagonal map `bond <- bond` from labelled sector values.
    ///
    /// TeNeT's typed counterpart of TensorKit `DiagonalTensorMap` / `diagm`
    /// stores `O(Σ_c k_c)` values. Input labels may be permuted, but must name
    /// every nonzero bond sector exactly once; output is canonicalized to the
    /// bond's engine-sector order. Each vector must equal that sector's
    /// degeneracy. All validation precedes checked layout admission, and the
    /// supplied dual flag is preserved.
    pub fn diagonal<I>(
        runtime: &Runtime,
        bond: &GradedSpace<R>,
        spectra: I,
    ) -> Result<Self, TypedFacadeError<R>>
    where
        I: IntoIterator<Item = SectorSpectrum<R::Sector, D>>,
    {
        let spectra: Vec<_> = spectra.into_iter().collect();
        let mut labels: Vec<_> = spectra.iter().map(|entry| &entry.sector).collect();
        labels.sort_unstable();
        if let Some(duplicate) = labels.windows(2).find(|pair| pair[0] == pair[1]) {
            return Err(Error::InvalidArgument(format!(
                "sector label {:?} is declared more than once",
                duplicate[0]
            ))
            .into());
        }

        let mut encoded = Vec::with_capacity(spectra.len());
        for entry in &spectra {
            encoded.push(
                TypedSectorAdmission::try_encode_label(bond.provider(), &entry.sector)
                    .map_err(<R::Mode as TypedTensorModeDispatch<R>>::map_provider_error)?,
            );
        }
        let mut by_id: Vec<_> = encoded.iter().enumerate().collect();
        by_id.sort_unstable_by_key(|(_, id)| *id);
        if let Some(duplicate) = by_id.windows(2).find(|pair| pair[0].1 == pair[1].1) {
            return Err(Error::InvalidArgument(format!(
                "SectorCodec law violation: labels {:?} and {:?} both encode to {:?}",
                spectra[duplicate[0].0].sector, spectra[duplicate[1].0].sector, duplicate[0].1
            ))
            .into());
        }
        let mut supplied: HashMap<_, _> = encoded
            .into_iter()
            .zip(spectra)
            .map(|(sector, entry)| (sector, entry.values))
            .collect();
        if bond
            .leg()
            .sectors()
            .iter()
            .any(|sector| !supplied.contains_key(sector))
        {
            return Err(Error::InvalidArgument(
                "diagonal spectrum is missing a bond sector".into(),
            )
            .into());
        }
        if supplied.len() != bond.leg().sectors().len() {
            return Err(Error::InvalidArgument(
                "diagonal spectrum contains an unknown bond sector".into(),
            )
            .into());
        }
        let mut spectrum = Vec::with_capacity(bond.degeneracies().len());
        for (&sector, &degeneracy) in bond.leg().sectors().iter().zip(bond.degeneracies()) {
            let values = supplied
                .remove(&sector)
                .expect("complete checked spectrum contains every bond sector");
            if values.len() != degeneracy {
                return Err(Error::InvalidArgument(
                    "diagonal spectrum length does not match bond degeneracy".into(),
                )
                .into());
            }
            spectrum.push(tenet_matrixalgebra::SectorSpectrum { sector, values });
        }
        let homspace = FusionTreeHomSpace::new(
            FusionProductSpace::new([bond.leg().clone()]),
            FusionProductSpace::new([bond.leg().clone()]),
        );
        let space = <R::Mode as TypedTensorRootDispatch<R>>::build_root(
            Arc::clone(bond.provider_arc()),
            homspace,
        )?;
        Ok(Self {
            runtime: runtime.clone(),
            repr: owned_repr(TypedTensorBody::diagonal(space, spectrum)),
        })
    }

    /// The leg of a `bond <- bond` endomorphism, after proving that is what
    /// the receiver is.
    fn bond_endomorphism_leg(&self, operation: &str) -> Result<&SectorLeg, TypedFacadeError<R>> {
        let homspace = self.logical_space().space().homspace();
        if homspace.codomain().len() != 1 || homspace.domain().len() != 1 {
            return Err(Error::InvalidArgument(format!(
                "{operation} requires a rank-(1,1) bond <- bond map, got rank {}|{}",
                homspace.codomain().len(),
                homspace.domain().len()
            ))
            .into());
        }
        let codomain = &homspace.codomain().legs()[0];
        if codomain != &homspace.domain().legs()[0] {
            return Err(Error::InvalidArgument(format!(
                "{operation} requires equal codomain and domain legs"
            ))
            .into());
        }
        Ok(codomain)
    }

    /// Returns the per-coupled-sector diagonal of a `bond <- bond` map.
    ///
    /// MatrixAlgebraKit's `diagview`, and the spectrum reader for a factor that
    /// is diagonal by construction but not stored compactly — a checked-Generic
    /// `s`, or a device `s`/`d` brought back with `to_host`. Compact storage is
    /// cloned; dense storage is read one strided diagonal per block.
    /// Off-diagonal entries are never inspected, so this returns the diagonal
    /// of an arbitrary endomorphism, not a proof that it is diagonal — use
    /// [`crate::expert::is_diagonal`] for that.
    ///
    /// [`crate::expert::diagonal_spectrum`] is a different question: it
    /// answers whether the payload *is* compact.
    ///
    /// # Complexity
    ///
    /// `O(sum_c k_c)` reads and one output allocation per sector; no dense
    /// block is materialized and no payload-sized copy is made.
    ///
    /// # Errors
    ///
    /// [`Error::InvalidArgument`] for a lazy adjoint (adjoin the result of the
    /// owned parent, or read [`Self::materialize`]'s result, instead), for a
    /// receiver that is not `bond <- bond`, and
    /// for a layout whose blocks are not fusion-tree keyed.
    ///
    /// Each sector is decoded to its provider label. A provider that cannot
    /// decode one returns its own error: for a checked Generic provider that
    /// is [`GenericTensorError::Structure`] wrapping
    /// [`CheckedGenericStructureError::Provider`],
    /// and for a multiplicity-free provider [`Error::FusionAlgebra`]. No
    /// partial spectrum is returned. In a truncated factorization this is the
    /// step where a decode failure surfaces, before `find_truncated`.
    pub fn diagview(&self) -> Result<Vec<SectorSpectrum<R::Sector, D>>, TypedFacadeError<R>> {
        let body = self.owned_body().ok_or_else(|| {
            TypedFacadeError::<R>::from(Error::InvalidArgument(
                "diagview requires an owned tensor, not a lazy adjoint".to_string(),
            ))
        })?;
        self.bond_endomorphism_leg("diagview")?;
        let raw: Vec<tenet_matrixalgebra::SectorSpectrum<D>> = match body.data.as_ref() {
            TypedData::Diagonal(spectrum) => spectrum.clone(),
            TypedData::Dense(_) => {
                let structure = self.logical_space().space().structure();
                let payload = body.materialized_dense_data();
                let data: &[D] = &payload;
                let mut collected = Vec::with_capacity(structure.block_count());
                for index in 0..structure.block_count() {
                    let block = structure
                        .block(index)
                        .map_err(Error::from)
                        .map_err(TypedFacadeError::<R>::from)?;
                    let BlockKey::FusionTree(key) = block.key() else {
                        return Err(Error::InvalidArgument(
                            "diagview requires a fusion-tree block layout".to_string(),
                        )
                        .into());
                    };
                    let offset = block.offset();
                    let step = block.strides()[0] + block.strides()[1];
                    let count = block.shape()[0].min(block.shape()[1]);
                    collected.push(tenet_matrixalgebra::SectorSpectrum {
                        sector: key.codomain_tree().coupled(),
                        values: (0..count)
                            .map(|step_index| data[offset + step_index * step])
                            .collect(),
                    });
                }
                // Canonical bond-sector order, as compact storage already keeps
                // it, so the two arms are interchangeable for the caller.
                collected.sort_unstable_by_key(|entry| entry.sector);
                if let Some(pair) = collected
                    .windows(2)
                    .find(|pair| pair[0].sector == pair[1].sector)
                {
                    return Err(Error::InvalidArgument(format!(
                        "diagview: coupled sector {:?} names more than one block",
                        pair[0].sector
                    ))
                    .into());
                }
                collected
            }
        };
        raw.into_iter()
            .map(|entry| {
                Ok(SectorSpectrum {
                    sector: TypedSectorAdmission::try_decode_label(
                        self.logical_space().provider(),
                        entry.sector,
                    )
                    .map_err(<R::Mode as TypedTensorModeDispatch<R>>::map_provider_error)?,
                    values: entry.values,
                })
            })
            .collect()
    }

    /// Returns the compact diagonal spectrum without materializing dense data.
    ///
    /// This is typed `diag` readback: [`None`] means the representation has no
    /// directly stored compact spectrum (it is dense or a lazy adjoint);
    /// otherwise it clones only the `O(Σ_c k_c)` compact values in canonical
    /// bond-sector order.
    #[expect(
        clippy::type_complexity,
        reason = "the expert compact readback exposes provider-labelled sector spectra"
    )]
    pub(crate) fn diagonal_spectrum(
        &self,
    ) -> Result<Option<Vec<SectorSpectrum<R::Sector, D>>>, TypedFacadeError<R>> {
        self.spectrum()
            .map(|spectrum| {
                spectrum
                    .iter()
                    .map(|entry| {
                        Ok(SectorSpectrum {
                            sector: TypedSectorAdmission::try_decode_label(
                                self.logical_space().provider(),
                                entry.sector,
                            )
                            .map_err(<R::Mode as TypedTensorModeDispatch<R>>::map_provider_error)?,
                            values: entry.values.clone(),
                        })
                    })
                    .collect()
            })
            .transpose()
    }
}

impl<R, D> TensorMap<R, D>
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64> + CheckedFusionAlgebra + SectorCodec,
    D: TensorScalar,
{
    /// Tests a rank-one map for blockwise diagonality without materializing compact storage.
    ///
    /// This matches TensorKit `isdiag` for finite data at `tol = 0`; positive
    /// tolerance uses `max_offdiag <= tol * max(norm(Inf), 1)`. Negative and
    /// non-finite tolerances are rejected before every shortcut.
    ///
    /// Scale `tol` to the payload dtype, as [`FactorizationScalar`] describes.
    pub(crate) fn is_diagonal(&self, tol: f64) -> Result<bool, Error> {
        if !tol.is_finite() || tol < 0.0 {
            return Err(Error::InvalidArgument(
                "diagonal tolerance must be finite and nonnegative".into(),
            ));
        }
        if self.spectrum().is_some() {
            return Ok(true);
        }
        if self.rank() != 2 || self.codomain_rank() != 1 || self.domain_rank() != 1 {
            return Ok(false);
        }
        let materialized = self.materialized_tensor_uncached()?;
        let data_payload = materialized
            .owned_body()
            .expect("uncached materialization is owned")
            .materialized_dense_data();
        let data: &[D] = &data_payload;
        let mut norm = 0.0_f64;
        let mut offdiag = 0.0_f64;
        for index in 0..self.logical_space().space().structure().block_count() {
            let block = self.logical_space().space().structure().block(index)?;
            for row in 0..block.shape()[0] {
                for col in 0..block.shape()[1] {
                    let value = data
                        [block.offset() + row * block.strides()[0] + col * block.strides()[1]]
                        .widen_complex()
                        .norm();
                    norm = norm.max(value);
                    if row != col {
                        if !value.is_finite() {
                            return Ok(false);
                        }
                        offdiag = offdiag.max(value);
                    }
                }
            }
        }
        Ok(offdiag <= tol * norm.max(1.0))
    }
}

impl<R, D> TensorMap<R, D>
where
    R: TypedSectorAdmission,
    R::Mode: TypedTensorConstructionDispatch<R, D>,
    D: TensorScalar,
{
    /// Returns the first leg's provider allocation after proving that every
    /// leg has the same categorical identity.
    fn authority<'a>(legs: &[&'a GradedSpace<R>]) -> Result<&'a Arc<R>, TypedFacadeError<R>> {
        let (first, rest) = legs.split_first().ok_or_else(|| {
            TypedFacadeError::<R>::from(Error::InvalidArgument(
                "at least one leg is required to infer the fusion provider".into(),
            ))
        })?;
        // Equal identities, rather than pointer equality, let separately
        // allocated providers interoperate. This guard precedes every provider
        // query and layout-admission stage.
        let expected_identity = TypedSectorAdmission::typed_rule_identity(first.provider());
        for leg in rest {
            let actual_identity = TypedSectorAdmission::typed_rule_identity(leg.provider());
            if actual_identity != expected_identity {
                return Err(TypedFacadeError::<R>::from(Error::RuleMismatch));
            }
        }
        Ok(first.provider_arc())
    }

    /// Validation half of [`Self::build`]: admits the complete bound layout
    /// without touching payload or runtime RNG state.
    fn build_space(
        provider: Arc<R>,
        codomain: &[&GradedSpace<R>],
        domain: &[&GradedSpace<R>],
    ) -> Result<BoundDynamicFusionMapSpace<R>, TypedFacadeError<R>> {
        let homspace = FusionTreeHomSpace::new(
            FusionProductSpace::new(codomain.iter().map(|leg| leg.leg().clone())),
            FusionProductSpace::new(domain.iter().map(|leg| leg.leg().clone())),
        );
        <R::Mode as TypedTensorConstructionDispatch<R, D>>::build_construction_root(
            provider, homspace,
        )
    }

    /// Payload half of [`Self::build`]: fills only a fully admitted layout and
    /// validates its final data length before publication.
    fn fill_space(
        runtime: &Runtime,
        space: BoundDynamicFusionMapSpace<R>,
        fill: Fill<'_, D>,
    ) -> Result<Self, TypedFacadeError<R>> {
        let data = apply_fill(space.space(), fill).map_err(TypedFacadeError::<R>::from)?;
        BoundDynamicTensorRef::try_new(&space, &data)
            .map_err(Error::from)
            .map_err(TypedFacadeError::<R>::from)?;
        Ok(Self {
            runtime: runtime.clone(),
            repr: owned_repr(TypedTensorBody::dense(space, data)),
        })
    }

    fn build(
        runtime: &Runtime,
        provider: Arc<R>,
        codomain: &[&GradedSpace<R>],
        domain: &[&GradedSpace<R>],
        fill: Fill<'_, D>,
    ) -> Result<Self, TypedFacadeError<R>> {
        let space = Self::build_space(provider, codomain, domain)?;
        Self::fill_space(runtime, space, fill)
    }

    /// Zero tensor map on `codomain <- domain` (TensorKit `zeros(T, W <- V)`).
    ///
    /// Every leg must have the same rule identity; the first leg's exact
    /// provider allocation becomes the tensor's authority. Identity mismatch
    /// is reported before provider algebra is queried. Layout admission and
    /// payload validation are transactional: failure publishes no tensor.
    ///
    /// # Complexity
    ///
    /// One fusion-tree layout admission plus one `O(stored_len)` zeroed
    /// payload allocation.
    ///
    /// ```compile_fail
    /// use std::sync::Arc;
    /// use tenet::core::{FibonacciFusionRule, FibonacciSector};
    /// use tenet::typed::{GradedSpace, Runtime, TensorMap};
    /// let runtime = Runtime::builder().build().unwrap();
    /// let tau = GradedSpace::try_new(
    ///     Arc::new(FibonacciFusionRule),
    ///     [(FibonacciSector::Tau, 1)],
    /// ).unwrap();
    /// // A complex categorical coefficient cannot act on a real payload.
    /// let _: TensorMap<FibonacciFusionRule, f64> =
    ///     TensorMap::zeros(&runtime, [&tau], [&tau]).unwrap();
    /// ```
    pub fn zeros<'a, Codomain, Domain>(
        runtime: &Runtime,
        codomain: Codomain,
        domain: Domain,
    ) -> Result<Self, TypedFacadeError<R>>
    where
        Codomain: IntoIterator<Item = &'a GradedSpace<R>>,
        Domain: IntoIterator<Item = &'a GradedSpace<R>>,
        R: 'a,
    {
        let codomain: Vec<_> = codomain.into_iter().collect();
        let domain: Vec<_> = domain.into_iter().collect();
        let legs: Vec<_> = codomain.iter().chain(&domain).copied().collect();
        let provider = Arc::clone(Self::authority(&legs)?);
        Self::build(runtime, provider, &codomain, &domain, Fill::Zeros)
    }

    /// Tensor map whose every symmetry-allowed element is produced by
    /// `fill(trees, indices)`, one fusion-tree subblock at a time (the unit
    /// [`Self::subblocks`] reads back; coupled-sector blocks are
    /// [`Self::blocks`]).
    ///
    /// All block keys are decoded exactly once before the first callback.
    /// Therefore a late decode failure invokes `fill` zero times and publishes
    /// neither a partial payload nor a tensor.
    ///
    /// **No TensorKit counterpart.** TensorKit builds an uninitialized,
    /// zeroed, or random tensor and then mutates its blocks; this labelled
    /// callback is TeNeT's semantic-fixture constructor.
    ///
    /// # Complexity
    ///
    /// One layout admission, one decode per stored block, and one callback per
    /// stored element.
    pub fn from_subblock_fn<'a, Codomain, Domain, F>(
        runtime: &Runtime,
        codomain: Codomain,
        domain: Domain,
        mut fill: F,
    ) -> Result<Self, TypedFacadeError<R>>
    where
        Codomain: IntoIterator<Item = &'a GradedSpace<R>>,
        Domain: IntoIterator<Item = &'a GradedSpace<R>>,
        F: FnMut(&BlockFusionTrees<R::Sector>, &[usize]) -> D,
        R: 'a,
    {
        let codomain: Vec<_> = codomain.into_iter().collect();
        let domain: Vec<_> = domain.into_iter().collect();
        let legs: Vec<_> = codomain.iter().chain(&domain).copied().collect();
        let provider = Arc::clone(Self::authority(&legs)?);
        let space = Self::build_space(Arc::clone(&provider), &codomain, &domain)?;
        let structure = space.space().structure();
        let mut labelled = HashMap::with_capacity(structure.block_count());
        for index in 0..structure.block_count() {
            let key = structure
                .block(index)
                .map_err(Error::from)
                .map_err(TypedFacadeError::<R>::from)?
                .key()
                .clone();
            let decoded = decode_block_fusion_trees(provider.as_ref(), &key)?;
            labelled.insert(key, decoded);
        }
        let mut callback = |key: &BlockKey, indices: &[usize]| {
            fill(
                labelled
                    .get(key)
                    .expect("all admitted block keys were decoded before payload fill"),
                indices,
            )
        };
        Self::fill_space(runtime, space, Fill::BlockFn(&mut callback))
    }

    /// Random tensor map using an explicit deterministic splitmix64 seed.
    ///
    /// Every stored real component is uniform on `[-1, 1)`: an `f64` payload
    /// entry uses one draw, while a `Complex64` entry uses independent draws for
    /// its real and imaginary components. This deliberately differs from
    /// TensorKit's default `rand`, whose components are uniform on `[0, 1)`;
    /// pinned source coordinates are recorded in `tenet/references.md`.
    ///
    /// There is no seedless form: TensorKit takes the RNG from the caller, and
    /// the seed is this API's explicit counterpart of that argument.
    ///
    /// Reproducibility is defined for the same TeNeT version and layout;
    /// it is not a cross-library byte-stream contract. Semantic cross-version
    /// fixtures should use [`Self::from_subblock_fn`].
    ///
    /// # Complexity
    ///
    /// One layout admission and one `O(stored_len)` payload allocation/fill.
    pub fn rand_with_seed<'a, Codomain, Domain>(
        runtime: &Runtime,
        codomain: Codomain,
        domain: Domain,
        seed: u64,
    ) -> Result<Self, TypedFacadeError<R>>
    where
        Codomain: IntoIterator<Item = &'a GradedSpace<R>>,
        Domain: IntoIterator<Item = &'a GradedSpace<R>>,
        R: 'a,
    {
        let codomain: Vec<_> = codomain.into_iter().collect();
        let domain: Vec<_> = domain.into_iter().collect();
        let legs: Vec<_> = codomain.iter().chain(&domain).copied().collect();
        let provider = Arc::clone(Self::authority(&legs)?);
        Self::build(runtime, provider, &codomain, &domain, Fill::Rand(seed))
    }
}

impl<R, D> TensorMap<R, D>
where
    R: TypedSectorAdmission,
    R::Mode: TypedTensorModeDispatch<R>,
    D: TensorScalar,
{
    /// Provider-labelled fusion trees and borrowed values for every fusion-tree
    /// subblock: TensorKit `subblocks(t)`. For coupled-sector matrices use
    /// [`Self::blocks`].
    ///
    /// All labels are decoded and all views are validated before the iterator is
    /// returned, so iteration is infallible. In particular, a checked-provider
    /// decode failure exposes no prefix. Blocks remain in canonical stored order.
    ///
    /// Copies no numeric payload.
    ///
    /// # Errors
    ///
    /// [`Error::Unsupported`] with [`crate::error::Alternative::Materialize`]
    /// for a lazy adjoint or a compact diagonal, whose subblocks are not
    /// stored densely; call [`Self::materialize`] first. Otherwise an error
    /// if the provider cannot decode a stored fusion-tree label, or if stored
    /// block metadata does not form a valid view into the logical tensor
    /// data.
    #[expect(
        clippy::type_complexity,
        reason = "the public block iterator yields labelled trees with borrowed block views"
    )]
    pub fn subblocks<'a>(
        &'a self,
    ) -> Result<
        impl ExactSizeIterator<Item = (BlockFusionTrees<R::Sector>, BlockView<'a, D>)> + 'a,
        TypedFacadeError<R>,
    > {
        let data = self
            .dense_data()
            .map_err(|_| borrowed_view_unsupported("subblocks"))
            .map_err(TypedFacadeError::<R>::from)?;
        let structure = self.logical_space().space().structure();
        let mut labelled = Vec::with_capacity(structure.block_count());
        for index in 0..structure.block_count() {
            let block = structure
                .block(index)
                .map_err(Error::from)
                .map_err(TypedFacadeError::<R>::from)?;
            let trees = decode_block_fusion_trees(self.logical_space().provider(), block.key())?;
            labelled.push((trees, block));
        }

        let blocks = labelled
            .into_iter()
            .map(|(trees, block)| {
                BlockView::new(data, block.shape(), block.strides(), block.offset())
                    .map(|values| (trees, values))
                    .map_err(Error::from)
                    .map_err(TypedFacadeError::<R>::from)
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok(blocks.into_iter())
    }

    /// Provider-labelled fusion trees of one subblock, by its index in
    /// [`Self::subblocks`] order.
    pub fn subblock_fusion_trees(
        &self,
        index: usize,
    ) -> Result<BlockFusionTrees<R::Sector>, TypedFacadeError<R>> {
        let block = self
            .logical_space()
            .space()
            .structure()
            .block(index)
            .map_err(Error::from)
            .map_err(TypedFacadeError::<R>::from)?;
        decode_block_fusion_trees(self.logical_space().provider(), block.key())
    }
}

impl<R, D, S> TensorMap<R, D, S>
where
    R: TypedSectorAdmission,
    R::Mode: TypedTensorModeDispatch<R>,
    S: TensorStorage<D>,
{
    /// The matrix of coupled sector `coupled`: TensorKit `block(t, c)`.
    ///
    /// A borrowed view on Host and CUDA storage alike; nothing is copied,
    /// transferred or materialized. A lazy adjoint yields its parent's region
    /// flagged as conjugate-transposed, and a compact diagonal its stored
    /// values, as TensorKit returns `block(parent, c)'` and `Diagonal(view)`.
    /// Row and column order are described on [`CoupledBlock`].
    ///
    /// A sector the tensor stores no block for gives TensorKit's empty view:
    /// `blockdim(codomain, c) × blockdim(domain, c)`, where one of the two is
    /// zero.
    ///
    /// # Complexity
    ///
    /// `O(log C)` for `C` stored coupled sectors, after the per-structure
    /// region table is compiled once (`O(subblocks)`, cached and shared with
    /// [`Self::tr`] and [`Self::inner`]). An absent sector additionally folds
    /// the fused dimensions of both sides, `O(legs · sectors · channels)`.
    ///
    /// # Errors
    ///
    /// Returns the provider's error when it cannot encode `coupled`.
    pub fn block(
        &self,
        coupled: &R::Sector,
    ) -> Result<CoupledBlock<'_, R, D, S>, TypedFacadeError<R>> {
        let provider = self.logical_space().provider();
        let id = TypedSectorAdmission::try_encode_label(provider, coupled)
            .map_err(<R::Mode as TypedTensorModeDispatch<R>>::map_provider_error)?;
        let regions = self.stored_sector_regions()?;
        match regions.binary_search_by_key(&id, CoupledSectorRegion::coupled) {
            Ok(index) => self
                .coupled_block(regions, index)
                .map_err(TypedFacadeError::<R>::from),
            Err(_) => {
                let homspace = self.logical_space().space().homspace();
                let rows = <R::Mode as TypedTensorModeDispatch<R>>::coupled_block_dimension(
                    provider,
                    homspace.codomain(),
                    id,
                )?;
                let cols = <R::Mode as TypedTensorModeDispatch<R>>::coupled_block_dimension(
                    provider,
                    homspace.domain(),
                    id,
                )?;
                // A coupled sector nonempty on both sides always has a stored
                // region, so a miss with two nonzero dimensions is a broken layout.
                if rows != 0 && cols != 0 {
                    return Err(TypedFacadeError::<R>::from(internal_layout_error(
                        "a sector fused on both sides has no stored block",
                    )));
                }
                let payload = match self.storage_body().data.as_ref() {
                    TypedData::Dense(storage) => CoupledBlockPayload::Dense {
                        storage,
                        offset: 0,
                        adjoint: matches!(self.repr, TypedTensorRepr::Adjoint(_)),
                    },
                    TypedData::Diagonal(_) => CoupledBlockPayload::Diagonal(&[]),
                };
                Ok(CoupledBlock {
                    rows,
                    cols,
                    payload,
                    provider,
                    region: None,
                })
            }
        }
    }

    /// Every stored coupled sector with its matrix: TensorKit `blocks(t)`.
    ///
    /// Sectors come in ascending [`tenet_core::SectorId`] order, the storage
    /// order [`GradedSpace::sectors`] also uses, which is not TensorKit's
    /// `blocksectors` order in general. Each view is the one [`Self::block`]
    /// returns. All labels are decoded before the iterator is returned, so
    /// iteration is infallible.
    ///
    /// # Errors
    ///
    /// Fails when the provider cannot decode a stored coupled sector.
    #[expect(
        clippy::type_complexity,
        reason = "the public block iterator yields labelled sectors with borrowed matrix views"
    )]
    pub fn blocks<'a>(
        &'a self,
    ) -> Result<
        impl ExactSizeIterator<Item = (R::Sector, CoupledBlock<'a, R, D, S>)> + 'a,
        TypedFacadeError<R>,
    > {
        let regions = self.stored_sector_regions()?;
        let provider = self.logical_space().provider();
        let blocks = (0..regions.len())
            .map(|index| {
                let sector =
                    TypedSectorAdmission::try_decode_label(provider, regions[index].coupled())
                        .map_err(<R::Mode as TypedTensorModeDispatch<R>>::map_provider_error)?;
                let block = self
                    .coupled_block(Arc::clone(&regions), index)
                    .map_err(TypedFacadeError::<R>::from)?;
                Ok((sector, block))
            })
            .collect::<Result<Vec<_>, TypedFacadeError<R>>>()?;
        Ok(blocks.into_iter())
    }

    /// Coupled regions of the stored payload's own layout. For a lazy adjoint
    /// that is the parent's layout, whose sector `c` holds `block(t', c)'`.
    fn stored_sector_regions(&self) -> Result<Arc<[CoupledSectorRegion]>, TypedFacadeError<R>> {
        let space = self.storage_body().space.space();
        let regions =
            sector_regions(space.structure(), space.nout()).map_err(TypedFacadeError::<R>::from)?;
        debug_assert!(
            regions
                .windows(2)
                .all(|pair| pair[0].coupled() < pair[1].coupled()),
            "coupled regions are sorted by sector id"
        );
        Ok(regions)
    }

    fn coupled_block(
        &self,
        regions: Arc<[CoupledSectorRegion]>,
        index: usize,
    ) -> Result<CoupledBlock<'_, R, D, S>, Error> {
        let region = &regions[index];
        let adjoint = matches!(self.repr, TypedTensorRepr::Adjoint(_));
        let (rows, cols) = if adjoint {
            (region.cols(), region.rows())
        } else {
            (region.rows(), region.cols())
        };
        let payload = match self.storage_body().data.as_ref() {
            TypedData::Dense(storage) => {
                if region.range().end > storage.len() {
                    return Err(internal_layout_error("coupled region outside the payload"));
                }
                CoupledBlockPayload::Dense {
                    storage,
                    offset: region.range().start,
                    adjoint,
                }
            }
            // Every compact constructor stores the spectrum in bond-leg id
            // order, the order of the bond space's regions.
            TypedData::Diagonal(spectrum) => {
                let values = spectrum
                    .binary_search_by_key(&region.coupled(), |entry| entry.sector)
                    .map(|index| spectrum[index].values.as_slice())
                    .ok()
                    .filter(|values| values.len() == rows && rows == cols)
                    .ok_or_else(|| {
                        internal_layout_error("compact spectrum disagrees with its bond space")
                    })?;
                CoupledBlockPayload::Diagonal(values)
            }
        };
        Ok(CoupledBlock {
            rows,
            cols,
            payload,
            provider: self.logical_space().provider(),
            region: Some((regions, index, adjoint)),
        })
    }
}

impl<R, D> TensorMap<R, D>
where
    R: TypedSectorAdmission,
    R::Mode: TypedTensorConstructionDispatch<R, D>,
    D: TensorScalar,
{
    /// The fused external sector content of one side of a structural
    /// constructor (TensorKit `fuse`, `spaces/gradedspace.jl:150-158`),
    /// weighted by the provider's `N` symbols. Stored sector content is
    /// already external, so duality is dropped.
    fn fused_content(
        provider: &R,
        legs: &[&GradedSpace<R>],
    ) -> Result<Vec<(tenet_core::SectorId, usize)>, TypedFacadeError<R>> {
        let (first, rest) = legs.split_first().ok_or_else(|| {
            // Keep the empty-side failure local to the typed fusion fold.
            Error::InvalidArgument("fuse_all needs at least one space".into())
        })?;
        let mut fused: Vec<(tenet_core::SectorId, usize)> = first.leg().iter().collect();
        for leg in rest {
            let pairs: Vec<(tenet_core::SectorId, usize)> = leg.leg().iter().collect();
            fused = <R::Mode as TypedTensorModeDispatch<R>>::fuse_sector_content(
                provider, &fused, &pairs,
            )?;
        }
        Ok(fused)
    }

    /// Shared body of the structural constructors: checks the fused fit,
    /// builds zeros and writes the (partial) identity into every
    /// coupled-sector matrix.
    fn structural<'a, C, M>(
        runtime: &Runtime,
        codomain: C,
        domain: M,
        embed: bool,
        what: &str,
    ) -> Result<Self, TypedFacadeError<R>>
    where
        C: IntoIterator<Item = &'a GradedSpace<R>>,
        M: IntoIterator<Item = &'a GradedSpace<R>>,
        R: 'a,
    {
        let codomain: Vec<&GradedSpace<R>> = codomain.into_iter().collect();
        let domain: Vec<&GradedSpace<R>> = domain.into_iter().collect();
        let legs: Vec<&GradedSpace<R>> = codomain.iter().chain(&domain).copied().collect();
        let provider = Arc::clone(Self::authority(&legs)?);
        let fused_codomain = Self::fused_content(&provider, &codomain)?;
        let fused_domain = Self::fused_content(&provider, &domain)?;
        let fits = if embed {
            // TensorKit `domain ≾ codomain`: sectorwise embeddable.
            fused_domain
                .iter()
                .all(|&(sector, deg)| fused_codomain.iter().any(|&(s, d)| s == sector && d >= deg))
        } else {
            // TensorKit `domain ≅ codomain`: identical fused sector content
            // (both sides are SectorId-sorted, so slice equality is content
            // equality).
            fused_codomain == fused_domain
        };
        if !fits {
            // Keep the stable constructor diagnostic shape.
            return Err(Error::InvalidArgument(format!(
                "{what}: codomain and domain are not {} (fused sector content differs)",
                if embed {
                    "isometrically embeddable"
                } else {
                    "isomorphic"
                }
            ))
            .into());
        }
        let mut tensor = Self::build(runtime, provider, &codomain, &domain, Fill::Zeros)?;
        // TensorKit's `one!` per coupled-sector block (`tensors/linalg.jl:102-158`).
        // A block's rows and columns enumerate (fusion tree, degeneracy)
        // pairs, so for a Generic provider the diagonal is also the
        // equal-tree, equal-vertex, equal-degeneracy identity.
        write_identity_blocks_generic(&mut tensor)?;
        Ok(tensor)
    }

    /// The canonical structural isomorphism `codomain <- domain` (TensorKit
    /// `isomorphism(W ← V)`; also TensorKit `id(V)` as `isomorphism(V, V)` and
    /// `unitary`, which only adds a Euclidean inner-product check every TeNeT
    /// provider satisfies): every coupled-sector block is the identity matrix,
    /// which requires the fused codomain and domain to carry identical sector
    /// content. Multiplicity-free and checked-Generic providers alike.
    ///
    /// # Errors
    ///
    /// Everything [`Self::zeros`] reports, plus [`Error::InvalidArgument`]
    /// when the fused codomain and domain differ in sector content
    /// (TensorKit's `SpaceMismatch` on `domain ≅ codomain`).
    ///
    /// # Complexity
    ///
    /// One fused-content fold over the legs plus one `O(stored_len)` payload.
    pub fn isomorphism<'a, C, M>(
        runtime: &Runtime,
        codomain: C,
        domain: M,
    ) -> Result<Self, TypedFacadeError<R>>
    where
        C: IntoIterator<Item = &'a GradedSpace<R>>,
        M: IntoIterator<Item = &'a GradedSpace<R>>,
        R: 'a,
    {
        Self::structural(runtime, codomain, domain, false, "isomorphism")
    }

    /// The canonical isometry `codomain <- domain` (TensorKit
    /// `isometry(W ← V)`): each
    /// coupled-sector block is the partial identity (the first `cols` columns
    /// of the identity), so `t† ∘ t = isomorphism(domain, domain)`. Requires
    /// the domain to embed isometrically in the codomain (sectorwise
    /// `deg_domain <= deg_codomain` on the fused content). Multiplicity-free
    /// and checked-Generic providers alike.
    ///
    /// # Errors
    ///
    /// Everything [`Self::zeros`] reports, plus [`Error::InvalidArgument`]
    /// when the fused domain does not embed sectorwise into the fused
    /// codomain (TensorKit's `SpaceMismatch` on `domain ≾ codomain`).
    ///
    /// # Complexity
    ///
    /// One fused-content fold over the legs plus one `O(stored_len)` payload.
    pub fn isometry<'a, C, M>(
        runtime: &Runtime,
        codomain: C,
        domain: M,
    ) -> Result<Self, TypedFacadeError<R>>
    where
        C: IntoIterator<Item = &'a GradedSpace<R>>,
        M: IntoIterator<Item = &'a GradedSpace<R>>,
        R: 'a,
    {
        Self::structural(runtime, codomain, domain, true, "isometry")
    }
}

/// The destination checks every device `*_into` shares, in the Host order:
/// owned dense CUDA storage, no alias of the input payload, the result's
/// space and block layout, the exact length, then unique ownership.
#[cfg(feature = "cuda")]
fn unique_cuda_destination<'d, R, D>(
    destination: &'d TensorMap<R, D, CudaStorage<D>>,
    input: &Arc<TypedData<D, CudaStorage<D>>>,
    expected: &DynamicFusionMapSpace,
) -> Result<&'d CudaStorage<D>, Error>
where
    D: CudaPayload,
{
    let (body, storage) = match &destination.repr {
        TypedTensorRepr::Owned(body) => match body.data.as_ref() {
            TypedData::Dense(storage) => (body, storage),
            TypedData::Diagonal(_) => {
                return Err(Error::InvalidArgument(
                    "destination must use ordinary dense CUDA storage".to_string(),
                ))
            }
        },
        TypedTensorRepr::Adjoint(_) => {
            return Err(Error::InvalidArgument(
                "destination must use ordinary dense CUDA storage".to_string(),
            ))
        }
    };
    if Arc::ptr_eq(&body.data, input) {
        return Err(Error::InvalidArgument(
            "destination storage must not alias an input".to_string(),
        ));
    }
    if body.space.space() != expected {
        return Err(Error::InvalidArgument(
            "destination fusion space or block layout does not match the operation result"
                .to_string(),
        ));
    }
    let required = expected.required_len()?;
    let actual = TensorStorage::len(storage);
    if actual != required {
        return Err(Error::InvalidArgument(format!(
            "destination storage length {actual} does not match required length {required}"
        )));
    }
    if Arc::strong_count(body) != 1 || Arc::strong_count(&body.data) != 1 {
        return Err(Error::DestinationShared);
    }
    Ok(storage)
}
