#[allow(unused_imports)]
use super::*;

impl<R> TypedSpaceModeDispatch<R> for MultiplicityFreeAdmissionMode
where
    R: TypedSectorAdmission<Error = FusionAlgebraError, Mode = MultiplicityFreeAdmissionMode>
        + MultiplicityFreeRigidSymbols<Scalar = f64>
        + CheckedFusionAlgebra
        + SectorCodec,
{
    fn vacuum(provider: &R) -> SectorId {
        provider.vacuum()
    }

    fn fusion_channels(
        provider: &R,
        left: SectorId,
        right: SectorId,
    ) -> Result<Vec<SectorId>, TypedFacadeError<R>> {
        provider
            .try_fusion_channels(left, right)
            .map(|channels| channels.into_iter().collect())
            .map_err(Into::into)
    }

    fn nsymbol(
        provider: &R,
        left: SectorId,
        right: SectorId,
        coupled: SectorId,
    ) -> Result<usize, TypedFacadeError<R>> {
        provider
            .try_nsymbol(left, right, coupled)
            .map_err(Into::into)
    }

    fn dim(provider: &R, sector: SectorId) -> Result<f64, TypedFacadeError<R>> {
        Ok(provider.dim_scalar(sector))
    }
}

impl<R> TypedSpaceModeDispatch<R> for CheckedGenericAdmissionMode
where
    R: TypedSectorAdmission<
            Error = <R as CheckedGenericFusion>::Error,
            Mode = CheckedGenericAdmissionMode,
        > + CheckedGenericRigidSymbols<Scalar = f64>,
{
    fn vacuum(provider: &R) -> SectorId {
        CheckedGenericFusion::vacuum(provider)
    }

    fn fusion_channels(
        provider: &R,
        left: SectorId,
        right: SectorId,
    ) -> Result<Vec<SectorId>, TypedFacadeError<R>> {
        provider
            .try_fusion_channels(left, right)
            .map(|channels| channels.into_iter().collect())
            .map_err(<Self as TypedTensorModeDispatch<R>>::map_provider_error)
    }

    fn nsymbol(
        provider: &R,
        left: SectorId,
        right: SectorId,
        coupled: SectorId,
    ) -> Result<usize, TypedFacadeError<R>> {
        provider
            .try_nsymbol(left, right, coupled)
            .map_err(<Self as TypedTensorModeDispatch<R>>::map_provider_error)
    }

    fn dim(provider: &R, sector: SectorId) -> Result<f64, TypedFacadeError<R>> {
        provider
            .try_sqrt_dim_scalar(sector)
            .map(|sqrt_dim| sqrt_dim * sqrt_dim)
            .map_err(<Self as TypedTensorModeDispatch<R>>::map_provider_error)
    }
}

impl<R> TypedTruncationDispatch<R> for MultiplicityFreeAdmissionMode
where
    R: TypedSectorAdmission<Error = FusionAlgebraError, Mode = MultiplicityFreeAdmissionMode>
        + MultiplicityFreeRigidSymbols<Scalar = f64>
        + CheckedFusionAlgebra,
{
    fn decide_bond_truncation<V>(
        provider: &R,
        spectra: &[tenet_matrixalgebra::SectorSpectrum<V>],
        truncation: &Truncation,
    ) -> Result<tenet_matrixalgebra::TruncationDecision, Error>
    where
        V: SpectrumMagnitude,
    {
        // Magnitude-based, as MatrixAlgebraKit `findtruncated` is.
        tenet_matrixalgebra::decide_bond_truncation(provider, spectra, truncation)
            .map_err(Error::from)
    }
}

impl<R> TypedTruncationDispatch<R> for CheckedGenericAdmissionMode
where
    R: TypedSectorAdmission<
            Error = <R as CheckedGenericFusion>::Error,
            Mode = CheckedGenericAdmissionMode,
        > + CheckedGenericRigidSymbols<Scalar = f64>,
{
    fn decide_bond_truncation<V>(
        provider: &R,
        spectra: &[tenet_matrixalgebra::SectorSpectrum<V>],
        truncation: &Truncation,
    ) -> Result<tenet_matrixalgebra::TruncationDecision, Self::FacadeError>
    where
        V: SpectrumMagnitude,
    {
        tenet_matrixalgebra::decide_bond_truncation_generic_checked(provider, spectra, truncation)
            .map_err(Into::into)
    }
}

impl<R, D> TypedTensorReductionDispatch<R, D> for MultiplicityFreeAdmissionMode
where
    R: TypedSectorAdmission<Error = FusionAlgebraError, Mode = MultiplicityFreeAdmissionMode>
        + MultiplicityFreeRigidSymbols<Scalar = f64>
        + CheckedFusionAlgebra
        + SectorCodec,
    D: TensorScalar,
{
    fn inner(tensor: &TensorMap<R, D>, other: &TensorMap<R, D>) -> Result<D, Error> {
        tensor.inner_multiplicity_free(other)
    }
    fn norm(tensor: &TensorMap<R, D>, p: f64) -> Result<f64, Error> {
        tensor.norm_p_multiplicity_free(p)
    }
    fn tr(tensor: &TensorMap<R, D>) -> Result<D, Error> {
        tensor.tr_multiplicity_free()
    }
}

impl<R, D> TypedTensorAddScaleDispatch<R, D> for MultiplicityFreeAdmissionMode
where
    R: TypedSectorAdmission<Error = FusionAlgebraError, Mode = MultiplicityFreeAdmissionMode>
        + MultiplicityFreeRigidSymbols<Scalar = f64>
        + CheckedFusionAlgebra
        + SectorCodec,
    D: TensorScalar,
{
    fn axpby(
        tensor: &TensorMap<R, D>,
        alpha: D,
        other: &TensorMap<R, D>,
        beta: D,
    ) -> Result<TensorMap<R, D>, Error> {
        tensor.add_multiplicity_free(other, alpha, beta)
    }

    fn scale(tensor: &TensorMap<R, D>, factor: D) -> TensorMap<R, D> {
        tensor.scale_multiplicity_free(factor)
    }
}

impl<R, D> TypedTensorAddScaleDispatch<R, D> for CheckedGenericAdmissionMode
where
    R: TypedSectorAdmission<
            Error = <R as CheckedGenericFusion>::Error,
            Mode = CheckedGenericAdmissionMode,
        > + CheckedGenericFusion,
    D: TensorScalar,
{
    fn axpby(
        tensor: &TensorMap<R, D>,
        alpha: D,
        other: &TensorMap<R, D>,
        beta: D,
    ) -> Result<TensorMap<R, D>, GenericTensorError<<R as CheckedGenericFusion>::Error>> {
        host_add_impl(tensor, other, alpha, beta).map_err(GenericTensorError::from)
    }

    fn scale(tensor: &TensorMap<R, D>, factor: D) -> TensorMap<R, D> {
        host_scale_impl(tensor, factor)
    }
}

impl<R, D> TypedTensorAdjointDispatch<R, D> for MultiplicityFreeAdmissionMode
where
    R: TypedSectorAdmission<Error = FusionAlgebraError, Mode = MultiplicityFreeAdmissionMode>
        + MultiplicityFreeRigidSymbols<Scalar = f64>
        + CheckedFusionAlgebra
        + SectorCodec,
    D: TensorScalar,
{
    fn adjoint(tensor: &TensorMap<R, D>) -> Result<TensorMap<R, D>, Error> {
        tensor.adjoint_multiplicity_free()
    }
}

impl<R, D> TypedTensorInvDispatch<R, D> for MultiplicityFreeAdmissionMode
where
    R: TypedSectorAdmission<Error = FusionAlgebraError, Mode = MultiplicityFreeAdmissionMode>
        + MultiplicityFreeRigidSymbols<Scalar = f64>
        + CheckedFusionAlgebra
        + SectorCodec,
    D: AdvancedLinalgScalar,
{
    fn inv(tensor: &TensorMap<R, D>) -> Result<TensorMap<R, D>, Error> {
        tensor.inv_multiplicity_free()
    }
}

impl<R, D> TypedTensorSolveDispatch<R, D> for MultiplicityFreeAdmissionMode
where
    R: TypedSectorAdmission<Error = FusionAlgebraError, Mode = MultiplicityFreeAdmissionMode>
        + MultiplicityFreeRigidSymbols<Scalar = f64>
        + CheckedFusionAlgebra
        + SectorCodec,
    D: AdvancedLinalgScalar,
{
    fn solve(tensor: &TensorMap<R, D>, rhs: &TensorMap<R, D>) -> Result<TensorMap<R, D>, Error> {
        tensor.solve_multiplicity_free(rhs)
    }
}

impl<R, D> TypedTensorPinvDispatch<R, D> for MultiplicityFreeAdmissionMode
where
    R: TypedSectorAdmission<Error = FusionAlgebraError, Mode = MultiplicityFreeAdmissionMode>
        + MultiplicityFreeRigidSymbols<Scalar = f64>
        + CheckedFusionAlgebra
        + SectorCodec,
    D: AdvancedLinalgScalar,
{
    fn pinv(tensor: &TensorMap<R, D>, rcond: f64) -> Result<TensorMap<R, D>, Error> {
        tensor.pinv_multiplicity_free(rcond)
    }
}

impl<R, D> TypedTensorNullDispatch<R, D> for MultiplicityFreeAdmissionMode
where
    R: TypedSectorAdmission<Error = FusionAlgebraError, Mode = MultiplicityFreeAdmissionMode>
        + MultiplicityFreeRigidSymbols<Scalar = f64>
        + CheckedFusionAlgebra
        + SectorCodec,
    D: FactorizationScalar,
{
    fn left_null(tensor: &TensorMap<R, D>) -> Result<TensorMap<R, D>, Error> {
        tensor.left_null_multiplicity_free()
    }

    fn right_null(tensor: &TensorMap<R, D>) -> Result<TensorMap<R, D>, Error> {
        tensor.right_null_multiplicity_free()
    }
}

impl<R, D> TypedTensorPolarDispatch<R, D> for MultiplicityFreeAdmissionMode
where
    R: TypedSectorAdmission<Error = FusionAlgebraError, Mode = MultiplicityFreeAdmissionMode>
        + MultiplicityFreeRigidSymbols<Scalar = f64>
        + CheckedFusionAlgebra
        + SectorCodec,
    D: FactorizationScalar,
{
    fn left_polar(tensor: &TensorMap<R, D>) -> Result<LeftPolar<TensorMap<R, D>>, Error> {
        tensor.left_polar_multiplicity_free()
    }

    fn right_polar(tensor: &TensorMap<R, D>) -> Result<RightPolar<TensorMap<R, D>>, Error> {
        tensor.right_polar_multiplicity_free()
    }
}

impl<R, D> TypedTensorExpDispatch<R, D> for MultiplicityFreeAdmissionMode
where
    R: TypedSectorAdmission<Error = FusionAlgebraError, Mode = MultiplicityFreeAdmissionMode>
        + MultiplicityFreeRigidSymbols<Scalar = f64>
        + CheckedFusionAlgebra
        + SectorCodec,
    D: AdvancedLinalgScalar,
{
    fn exp(tensor: &TensorMap<R, D>) -> Result<TensorMap<R, D>, Error> {
        tensor.exp_multiplicity_free()
    }
}

impl<R, D> TypedTensorQrDispatch<R, D> for MultiplicityFreeAdmissionMode
where
    R: TypedSectorAdmission<Error = FusionAlgebraError, Mode = MultiplicityFreeAdmissionMode>
        + MultiplicityFreeRigidSymbols<Scalar = f64>
        + CheckedFusionAlgebra
        + SectorCodec,
    D: FactorizationScalar,
{
    fn qr_compact(tensor: &TensorMap<R, D>) -> Result<Qr<TensorMap<R, D>>, Error> {
        tensor.qr_compact_multiplicity_free()
    }
}

impl<R, D> TypedTensorSvdDispatch<R, D> for MultiplicityFreeAdmissionMode
where
    R: TypedSectorAdmission<Error = FusionAlgebraError, Mode = MultiplicityFreeAdmissionMode>
        + MultiplicityFreeRigidSymbols<Scalar = f64>
        + CheckedFusionAlgebra
        + SectorCodec,
    D: FactorizationScalar,
{
    fn svd_compact(tensor: &TensorMap<R, D>) -> Result<Svd<TensorMap<R, D>>, Error> {
        tensor.svd_compact_multiplicity_free()
    }

    fn svd_full(tensor: &TensorMap<R, D>) -> Result<Svd<TensorMap<R, D>>, Error> {
        tensor.svd_full_multiplicity_free()
    }
}

impl<R, D> TypedTensorLqDispatch<R, D> for MultiplicityFreeAdmissionMode
where
    R: TypedSectorAdmission<Error = FusionAlgebraError, Mode = MultiplicityFreeAdmissionMode>
        + MultiplicityFreeRigidSymbols<Scalar = f64>
        + CheckedFusionAlgebra
        + SectorCodec,
    D: FactorizationScalar,
{
    fn lq_compact(tensor: &TensorMap<R, D>) -> Result<Lq<TensorMap<R, D>>, Error> {
        tensor.lq_compact_multiplicity_free()
    }
}

impl<R, D> TypedTensorFullQrDispatch<R, D> for MultiplicityFreeAdmissionMode
where
    R: TypedSectorAdmission<Error = FusionAlgebraError, Mode = MultiplicityFreeAdmissionMode>
        + MultiplicityFreeRigidSymbols<Scalar = f64>
        + CheckedFusionAlgebra
        + SectorCodec,
    D: FactorizationScalar,
{
    fn qr_full(tensor: &TensorMap<R, D>) -> Result<Qr<TensorMap<R, D>>, Error> {
        tensor.qr_full_multiplicity_free()
    }
}

impl<R, D> TypedTensorFullLqDispatch<R, D> for MultiplicityFreeAdmissionMode
where
    R: TypedSectorAdmission<Error = FusionAlgebraError, Mode = MultiplicityFreeAdmissionMode>
        + MultiplicityFreeRigidSymbols<Scalar = f64>
        + CheckedFusionAlgebra
        + SectorCodec,
    D: FactorizationScalar,
{
    fn lq_full(tensor: &TensorMap<R, D>) -> Result<Lq<TensorMap<R, D>>, Error> {
        tensor.lq_full_multiplicity_free()
    }
}

impl<R, D> TypedTensorSvdValsDispatch<R, D> for MultiplicityFreeAdmissionMode
where
    R: TypedSectorAdmission<
            Error = FusionAlgebraError,
            Mode = MultiplicityFreeAdmissionMode,
            Sector = <R as SectorCodec>::Sector,
        > + MultiplicityFreeRigidSymbols<Scalar = f64>
        + CheckedFusionAlgebra
        + SectorCodec,
    D: FactorizationScalar,
{
    fn svd_vals(
        tensor: &TensorMap<R, D>,
    ) -> Result<Vec<SectorSpectrum<<R as TypedSectorAdmission>::Sector, f64>>, Error> {
        tensor.svd_vals_multiplicity_free()
    }
}

impl<R, D> TypedTensorEighValsDispatch<R, D> for MultiplicityFreeAdmissionMode
where
    R: TypedSectorAdmission<
            Error = FusionAlgebraError,
            Mode = MultiplicityFreeAdmissionMode,
            Sector = <R as SectorCodec>::Sector,
        > + MultiplicityFreeRigidSymbols<Scalar = f64>
        + CheckedFusionAlgebra
        + SectorCodec,
    D: FactorizationScalar,
{
    fn eigh_vals(
        tensor: &TensorMap<R, D>,
    ) -> Result<Vec<SectorSpectrum<<R as TypedSectorAdmission>::Sector, f64>>, Error> {
        tensor.eigh_vals_multiplicity_free()
    }
}

impl<R, D> TypedTensorEighDispatch<R, D> for MultiplicityFreeAdmissionMode
where
    R: TypedSectorAdmission<Error = FusionAlgebraError, Mode = MultiplicityFreeAdmissionMode>
        + MultiplicityFreeRigidSymbols<Scalar = f64>
        + CheckedFusionAlgebra
        + SectorCodec,
    D: FactorizationScalar,
{
    fn eigh_full(tensor: &TensorMap<R, D>) -> Result<Eigh<TensorMap<R, D>>, Error> {
        tensor.eigh_full_multiplicity_free()
    }
}

impl<R, D> TypedTensorEigValsDispatch<R, D> for MultiplicityFreeAdmissionMode
where
    R: TypedSectorAdmission<
            Error = FusionAlgebraError,
            Mode = MultiplicityFreeAdmissionMode,
            Sector = <R as SectorCodec>::Sector,
        > + MultiplicityFreeRigidSymbols<Scalar = f64>
        + CheckedFusionAlgebra
        + SectorCodec,
    D: AdvancedLinalgScalar,
    <D as FactorScalar>::Eig: TensorScalar,
{
    fn eig_vals(
        tensor: &TensorMap<R, D>,
    ) -> Result<
        Vec<SectorSpectrum<<R as TypedSectorAdmission>::Sector, num_complex::Complex64>>,
        Error,
    > {
        tensor.eig_vals_multiplicity_free()
    }
}

impl<R, D> TypedTensorEigDispatch<R, D> for MultiplicityFreeAdmissionMode
where
    R: TypedSectorAdmission<Error = FusionAlgebraError, Mode = MultiplicityFreeAdmissionMode>
        + MultiplicityFreeRigidSymbols<Scalar = f64>
        + CheckedFusionAlgebra
        + SectorCodec,
    D: AdvancedLinalgScalar,
    <D as FactorScalar>::Eig: TensorScalar,
{
    fn eig_full(
        tensor: &TensorMap<R, D>,
    ) -> Result<Eig<TensorMap<R, <D as FactorScalar>::Eig>>, Error> {
        tensor.eig_full_multiplicity_free()
    }
}

impl<R, D> TypedTensorAdjointDispatch<R, D> for CheckedGenericAdmissionMode
where
    R: TypedSectorAdmission<
            Error = <R as CheckedGenericFusion>::Error,
            Mode = CheckedGenericAdmissionMode,
        > + CheckedGenericFusion,
    D: TensorScalar,
{
    fn adjoint(
        tensor: &TensorMap<R, D>,
    ) -> Result<TensorMap<R, D>, GenericTensorError<<R as CheckedGenericFusion>::Error>> {
        // Why not a lazy view: densifying its unstored zeros through
        // conjugation would publish them as `0-0i`.
        if let Some(adjoint) = tensor.compact_adjoint() {
            return Ok(adjoint);
        }
        Ok(match &tensor.repr {
            TypedTensorRepr::Owned(parent) => {
                let logical_space =
                    tenet_tensors::adjoint_bound_space_dyn_generic_checked(&parent.space)
                        .map_err(GenericTensorError::Plan)?;
                TensorMap {
                    runtime: tensor.runtime.clone(),
                    repr: TypedTensorRepr::Adjoint(Arc::new(TypedAdjointView::new(
                        Arc::clone(parent),
                        logical_space,
                    ))),
                }
            }
            TypedTensorRepr::Adjoint(view) => TensorMap {
                runtime: tensor.runtime.clone(),
                repr: TypedTensorRepr::Owned(Arc::clone(&view.parent)),
            },
        })
    }
}

impl<R, D> TypedTensorInvDispatch<R, D> for CheckedGenericAdmissionMode
where
    R: TypedSectorAdmission<
            Error = <R as CheckedGenericFusion>::Error,
            Mode = CheckedGenericAdmissionMode,
        > + CheckedGenericFusion,
    D: AdvancedLinalgScalar,
{
    fn inv(
        tensor: &TensorMap<R, D>,
    ) -> Result<TensorMap<R, D>, GenericTensorError<<R as CheckedGenericFusion>::Error>> {
        if matches!(&tensor.repr, TypedTensorRepr::Adjoint(_)) {
            return Self::inv(&tensor.adjoint()?)?
                .adjoint()?
                .materialized_tensor_uncached()
                .map_err(GenericTensorError::from);
        }
        let source = tensor.logical_space();
        let codomain = tenet_matrixalgebra::coupled_sector_block_dimensions_generic_checked(
            source.space().homspace().codomain(),
            source.provider(),
        )?;
        let domain = tenet_matrixalgebra::coupled_sector_block_dimensions_generic_checked(
            source.space().homspace().domain(),
            source.provider(),
        )?;
        if codomain != domain {
            return Err(Error::from(
                tenet_tensors::OperationError::UnsupportedTensorContractScope {
                    message: "inv requires isomorphic codomain and domain",
                },
            )
            .into());
        }
        let homspace = FusionTreeHomSpace::new(
            source.space().homspace().domain().clone(),
            source.space().homspace().codomain().clone(),
        );
        let output = <R::Mode as TypedTensorRootDispatch<R>>::build_root(
            Arc::clone(source.provider_arc()),
            homspace,
        )?;
        let body = tensor
            .owned_body()
            .expect("checked Generic inverse input is owned after lazy dispatch");
        let mut dense = tensor.runtime.lease_dense();
        let factor = tenet_matrixalgebra::inv_direct_into_dyn(
            dense.dense(),
            &BoundDynamicTensorRef::try_new(&body.space, body.materialized_dense_data().as_ref())
                .map_err(Error::from)?,
            output,
        )
        .map_err(Error::from)?;
        Ok(wrap_factor_on(&tensor.runtime, factor))
    }
}

impl<R, D> TypedTensorSolveDispatch<R, D> for CheckedGenericAdmissionMode
where
    R: TypedSectorAdmission<
            Error = <R as CheckedGenericFusion>::Error,
            Mode = CheckedGenericAdmissionMode,
        > + CheckedGenericFusion,
    D: AdvancedLinalgScalar,
{
    fn solve(
        tensor: &TensorMap<R, D>,
        rhs: &TensorMap<R, D>,
    ) -> Result<TensorMap<R, D>, GenericTensorError<<R as CheckedGenericFusion>::Error>> {
        if !tensor.runtime.same_runtime(&rhs.runtime) {
            return Err(Error::RuntimeMismatch.into());
        }
        // `checked_generic_solve_into` materializes every lazy `rhs`.
        rhs.refuse_borrowed_view("solve")?;
        let _host_pool = tensor.runtime.enter_host_pool();
        if tensor.logical_space().space().admission().rule_identity()
            != rhs.logical_space().space().admission().rule_identity()
        {
            return Err(Error::RuleMismatch.into());
        }
        if tensor.logical_space().space().homspace().codomain()
            != rhs.logical_space().space().homspace().codomain()
        {
            return Err(Error::InvalidArgument(
                "solve requires equal divisor and right-hand-side codomains".to_string(),
            )
            .into());
        }

        let lhs_space = tensor.logical_space();
        let codomain = tenet_matrixalgebra::coupled_sector_block_dimensions_generic_checked(
            lhs_space.space().homspace().codomain(),
            lhs_space.provider(),
        )?;
        let domain = tenet_matrixalgebra::coupled_sector_block_dimensions_generic_checked(
            lhs_space.space().homspace().domain(),
            lhs_space.provider(),
        )?;
        if codomain != domain {
            return Err(Error::from(
                tenet_tensors::OperationError::UnsupportedTensorContractScope {
                    message: "solve requires an isomorphic divisor codomain and domain",
                },
            )
            .into());
        }

        let output = <R::Mode as TypedTensorRootDispatch<R>>::build_root(
            Arc::clone(lhs_space.provider_arc()),
            FusionTreeHomSpace::new(
                lhs_space.space().homspace().domain().clone(),
                rhs.logical_space().space().homspace().domain().clone(),
            ),
        )?;
        let factor =
            checked_generic_solve_into(tensor, rhs, tensor.logical_space().clone(), output)?;
        Ok(wrap_factor_on(&tensor.runtime, factor))
    }
}

pub(super) fn checked_generic_solve_into<R, D>(
    tensor: &TensorMap<R, D>,
    rhs: &TensorMap<R, D>,
    divisor_authority: BoundDynamicFusionMapSpace<R>,
    output: BoundDynamicFusionMapSpace<R>,
) -> Result<BoundDynFactor<R, D>, GenericTensorError<<R as CheckedGenericFusion>::Error>>
where
    R: TypedSectorAdmission<
            Error = <R as CheckedGenericFusion>::Error,
            Mode = CheckedGenericAdmissionMode,
        > + CheckedGenericFusion,
    D: TensorScalar,
{
    let lhs = tensor
        .materialized_tensor_uncached()
        .map_err(GenericTensorError::from)?;
    let rhs = rhs
        .materialized_tensor_uncached()
        .map_err(GenericTensorError::from)?;
    let lhs_body = lhs.owned_body().expect("uncached solve lhs is owned");
    let rhs_body = rhs.owned_body().expect("uncached solve rhs is owned");
    let mut dense = tensor.runtime.lease_dense();
    tenet_matrixalgebra::solve_left_direct_into_dyn(
        dense.dense(),
        &BoundDynamicTensorRef::try_new(
            &divisor_authority,
            lhs_body.materialized_dense_data().as_ref(),
        )
        .map_err(Error::from)?,
        &BoundDynamicTensorRef::try_new(
            &rhs_body.space,
            rhs_body.materialized_dense_data().as_ref(),
        )
        .map_err(Error::from)?,
        output,
    )
    .map_err(Error::from)
    .map_err(GenericTensorError::from)
}

pub(super) fn checked_compact_pinv_layout<R, D>(
    source: &BoundDynamicFusionMapSpace<R>,
    output: &BoundDynamicFusionMapSpace<R>,
    spectrum: &[tenet_matrixalgebra::SectorSpectrum<D>],
) -> bool {
    let source_space = source.space();
    let output_space = output.space();
    if !is_diagonal_bond_space(source_space)
        || !is_diagonal_bond_space(output_space)
        || !Arc::ptr_eq(source.provider_arc(), output.provider_arc())
        || output_space.homspace().codomain() != source_space.homspace().domain()
        || output_space.homspace().domain() != source_space.homspace().codomain()
    {
        return false;
    }
    // Leave malformed or noncanonical layouts to the existing dense seam,
    // which owns their typed error mapping and validation order.
    let (Ok(Some(source_regions)), Ok(Some(output_regions))) = (
        source_space
            .structure()
            .coupled_sector_regions(source_space.nout()),
        output_space
            .structure()
            .coupled_sector_regions(output_space.nout()),
    ) else {
        return false;
    };
    if source_regions.len() != spectrum.len() || output_regions.len() != spectrum.len() {
        return false;
    }
    let by_sector: HashMap<_, _> = spectrum.iter().map(|entry| (entry.sector, entry)).collect();
    let output_by_sector: HashMap<_, _> = output_regions
        .iter()
        .map(|region| (region.coupled(), region))
        .collect();
    if by_sector.len() != spectrum.len() || output_by_sector.len() != spectrum.len() {
        return false;
    }
    source_regions.iter().all(|source_region| {
        let Some(entry) = by_sector.get(&source_region.coupled()) else {
            return false;
        };
        let Some(output_region) = output_by_sector.get(&source_region.coupled()) else {
            return false;
        };
        source_region.has_aligned_diagonal()
            && output_region.has_aligned_diagonal()
            && source_region.rows() == source_region.cols()
            && output_region.rows() == source_region.cols()
            && output_region.cols() == source_region.rows()
            && source_region.row_trees().len() == 1
            && source_region.col_trees().len() == 1
            && output_region.row_trees() == source_region.col_trees()
            && output_region.col_trees() == source_region.row_trees()
            && entry.values.len() == source_region.rows()
    })
}

impl<R, D> TypedTensorPinvDispatch<R, D> for CheckedGenericAdmissionMode
where
    R: TypedSectorAdmission<
            Error = <R as CheckedGenericFusion>::Error,
            Mode = CheckedGenericAdmissionMode,
        > + CheckedGenericFusion,
    D: AdvancedLinalgScalar,
{
    fn pinv(
        tensor: &TensorMap<R, D>,
        rcond: f64,
    ) -> Result<TensorMap<R, D>, GenericTensorError<<R as CheckedGenericFusion>::Error>> {
        if !rcond.is_finite() || rcond < 0.0 {
            return Err(Error::InvalidArgument(
                "pinv rcond must be finite and non-negative".to_string(),
            )
            .into());
        }
        if matches!(&tensor.repr, TypedTensorRepr::Adjoint(_)) {
            return Self::pinv(&tensor.adjoint()?, rcond)?
                .adjoint()?
                .materialized_tensor_uncached()
                .map_err(GenericTensorError::from);
        }
        let source = tensor.logical_space();
        let output = <R::Mode as TypedTensorRootDispatch<R>>::build_root(
            Arc::clone(source.provider_arc()),
            FusionTreeHomSpace::new(
                source.space().homspace().domain().clone(),
                source.space().homspace().codomain().clone(),
            ),
        )?;
        let body = tensor
            .owned_body()
            .expect("checked Generic pinv input is owned after lazy dispatch");
        if let TypedData::Diagonal(spectrum) = body.data.as_ref() {
            if checked_compact_pinv_layout(&body.space, &output, spectrum) {
                let sigma_max = spectrum.iter().flat_map(|entry| &entry.values).try_fold(
                    0.0_f64,
                    |largest, &value| {
                        let magnitude = value.abs_value();
                        let rounded = D::from_real(magnitude).abs_value();
                        (magnitude.is_finite() && rounded.is_finite()).then(|| largest.max(rounded))
                    },
                );
                if let Some(sigma_max) = sigma_max {
                    let cutoff = rcond * sigma_max;
                    // The dense SVD can flush a retained subnormal to zero.
                    if spectrum
                        .iter()
                        .flat_map(|entry| &entry.values)
                        .all(|&value| {
                            D::from_real(value.abs_value()).abs_value() <= cutoff
                                || (value.abs_value() >= D::safe_minimum()
                                    && value.recip_value().abs_value().is_finite())
                        })
                    {
                        let mapped = map_spectrum(spectrum, |value| {
                            Ok(if D::from_real(value.abs_value()).abs_value() > cutoff {
                                value.recip_value()
                            } else {
                                D::from_real(0.0)
                            })
                        })?;
                        return Ok(tensor.with_spectrum_on(output, mapped));
                    }
                }
            }
        }
        let mut dense = tensor.runtime.lease_dense();
        let factor = tenet_matrixalgebra::pinv_direct_into_dyn(
            dense.dense(),
            &BoundDynamicTensorRef::try_new(&body.space, body.materialized_dense_data().as_ref())
                .map_err(Error::from)?,
            output,
            rcond,
        )
        .map_err(pinv_seam_error)?;
        Ok(wrap_factor_on(&tensor.runtime, factor))
    }
}

impl<R, D> TypedTensorNullDispatch<R, D> for CheckedGenericAdmissionMode
where
    R: TypedSectorAdmission<
            Error = <R as CheckedGenericFusion>::Error,
            Mode = CheckedGenericAdmissionMode,
        > + CheckedGenericFusion,
    D: FactorizationScalar,
{
    fn left_null(
        tensor: &TensorMap<R, D>,
    ) -> Result<TensorMap<R, D>, GenericTensorError<<R as CheckedGenericFusion>::Error>> {
        if matches!(&tensor.repr, TypedTensorRepr::Adjoint(_)) {
            return Self::right_null(&tensor.adjoint()?)?
                .adjoint()?
                .materialized_tensor_uncached()
                .map_err(GenericTensorError::from);
        }
        let body = tensor
            .owned_body()
            .expect("checked Generic left-null input is owned after lazy dispatch");
        let dimensions = if let TypedData::Diagonal(spectrum) = body.data.as_ref() {
            if is_diagonal_bond_space(body.space.space()) {
                match tenet_matrixalgebra::left_null_diagonal_dyn_checked_generic(
                    &body.space,
                    spectrum,
                )? {
                    tenet_matrixalgebra::CheckedDiagonalNullFactor::Direct(factor) => {
                        return Ok(wrap_factor_on(&tensor.runtime, factor));
                    }
                    tenet_matrixalgebra::CheckedDiagonalNullFactor::Fallback(dimensions) => {
                        dimensions
                    }
                }
            } else {
                None
            }
        } else {
            None
        };
        let payload = body.materialized_dense_data();
        let input = BoundDynamicTensorRef::try_new(&body.space, &payload).map_err(Error::from)?;
        let mut dense = tensor.runtime.lease_dense();
        let factor = tenet_matrixalgebra::left_null_dyn_checked_generic_with_dimensions(
            dense.dense(),
            &input,
            dimensions,
        )?;
        Ok(wrap_factor_on(&tensor.runtime, factor))
    }

    fn right_null(
        tensor: &TensorMap<R, D>,
    ) -> Result<TensorMap<R, D>, GenericTensorError<<R as CheckedGenericFusion>::Error>> {
        if matches!(&tensor.repr, TypedTensorRepr::Adjoint(_)) {
            return Self::left_null(&tensor.adjoint()?)?
                .adjoint()?
                .materialized_tensor_uncached()
                .map_err(GenericTensorError::from);
        }
        let body = tensor
            .owned_body()
            .expect("checked Generic right-null input is owned after lazy dispatch");
        let dimensions = if let TypedData::Diagonal(spectrum) = body.data.as_ref() {
            if is_diagonal_bond_space(body.space.space()) {
                match tenet_matrixalgebra::right_null_diagonal_dyn_checked_generic(
                    &body.space,
                    spectrum,
                )? {
                    tenet_matrixalgebra::CheckedDiagonalNullFactor::Direct(factor) => {
                        return Ok(wrap_factor_on(&tensor.runtime, factor));
                    }
                    tenet_matrixalgebra::CheckedDiagonalNullFactor::Fallback(dimensions) => {
                        dimensions
                    }
                }
            } else {
                None
            }
        } else {
            None
        };
        let payload = body.materialized_dense_data();
        let input = BoundDynamicTensorRef::try_new(&body.space, &payload).map_err(Error::from)?;
        let mut dense = tensor.runtime.lease_dense();
        let factor = tenet_matrixalgebra::right_null_dyn_checked_generic_with_dimensions(
            dense.dense(),
            &input,
            dimensions,
        )?;
        Ok(wrap_factor_on(&tensor.runtime, factor))
    }
}

impl<R, D> TypedTensorPolarDispatch<R, D> for CheckedGenericAdmissionMode
where
    R: TypedSectorAdmission<
            Error = <R as CheckedGenericFusion>::Error,
            Mode = CheckedGenericAdmissionMode,
        > + CheckedGenericFusion,
    D: FactorizationScalar,
{
    fn left_polar(
        tensor: &TensorMap<R, D>,
    ) -> Result<LeftPolar<TensorMap<R, D>>, GenericTensorError<<R as CheckedGenericFusion>::Error>>
    {
        match &tensor.repr {
            TypedTensorRepr::Owned(body) => {
                if let TypedData::Diagonal(spectrum) = body.data.as_ref() {
                    if is_diagonal_bond_space(body.space.space()) {
                        if let Some(tenet_matrixalgebra::CheckedCompactPolarFactors {
                            w_space,
                            p_space,
                            phase,
                            magnitude,
                        }) =
                            tenet_matrixalgebra::left_polar_diagonal_spectra_dyn_checked_generic(
                                &body.space,
                                spectrum,
                            )?
                        {
                            return Ok(LeftPolar {
                                w: tensor.with_spectrum_on(w_space, phase),
                                p: tensor.with_spectrum_on(p_space, magnitude),
                            });
                        }
                    }
                }
                let payload = body.materialized_dense_data();
                let input =
                    BoundDynamicTensorRef::try_new(&body.space, &payload).map_err(Error::from)?;
                let mut dense = tensor.runtime.lease_dense();
                let LeftPolar { w, p } =
                    tenet_matrixalgebra::left_polar_dyn_checked_generic(dense.dense(), &input)?;
                Ok(LeftPolar {
                    w: wrap_factor_on(&tensor.runtime, w),
                    p: wrap_factor_on(&tensor.runtime, p),
                })
            }
            TypedTensorRepr::Adjoint(view) => {
                let mut dense = tensor.runtime.lease_dense();
                let input = BoundDynamicTensorRef::try_new(&view.parent.space, view.parent_data())
                    .map_err(Error::from)?;
                let RightPolar { p, wh: w } =
                    tenet_matrixalgebra::left_polar_adjoint_parent_dyn_checked_generic(
                        dense.dense(),
                        &input,
                    )?;
                let w = wrap_factor_on(&tensor.runtime, w)
                    .adjoint()?
                    .materialized_tensor_uncached()
                    .map_err(GenericTensorError::from)?;
                Ok(LeftPolar {
                    w,
                    p: wrap_factor_on(&tensor.runtime, p),
                })
            }
        }
    }

    fn right_polar(
        tensor: &TensorMap<R, D>,
    ) -> Result<RightPolar<TensorMap<R, D>>, GenericTensorError<<R as CheckedGenericFusion>::Error>>
    {
        match &tensor.repr {
            TypedTensorRepr::Owned(body) => {
                if let TypedData::Diagonal(spectrum) = body.data.as_ref() {
                    if is_diagonal_bond_space(body.space.space()) {
                        if let Some(tenet_matrixalgebra::CheckedCompactPolarFactors {
                            w_space,
                            p_space,
                            phase,
                            magnitude,
                        }) =
                            tenet_matrixalgebra::right_polar_diagonal_spectra_dyn_checked_generic(
                                &body.space,
                                spectrum,
                            )?
                        {
                            return Ok(RightPolar {
                                p: tensor.with_spectrum_on(p_space, magnitude),
                                wh: tensor.with_spectrum_on(w_space, phase),
                            });
                        }
                    }
                }
                let payload = body.materialized_dense_data();
                let input =
                    BoundDynamicTensorRef::try_new(&body.space, &payload).map_err(Error::from)?;
                let mut dense = tensor.runtime.lease_dense();
                let RightPolar { p, wh: w } =
                    tenet_matrixalgebra::right_polar_dyn_checked_generic(dense.dense(), &input)?;
                Ok(RightPolar {
                    p: wrap_factor_on(&tensor.runtime, p),
                    wh: wrap_factor_on(&tensor.runtime, w),
                })
            }
            TypedTensorRepr::Adjoint(view) => {
                let mut dense = tensor.runtime.lease_dense();
                let input = BoundDynamicTensorRef::try_new(&view.parent.space, view.parent_data())
                    .map_err(Error::from)?;
                let LeftPolar { w, p } =
                    tenet_matrixalgebra::right_polar_adjoint_parent_dyn_checked_generic(
                        dense.dense(),
                        &input,
                    )?;
                let w = wrap_factor_on(&tensor.runtime, w)
                    .adjoint()?
                    .materialized_tensor_uncached()
                    .map_err(GenericTensorError::from)?;
                Ok(RightPolar {
                    p: wrap_factor_on(&tensor.runtime, p),
                    wh: w,
                })
            }
        }
    }
}

impl<R, D> TypedTensorExpDispatch<R, D> for CheckedGenericAdmissionMode
where
    R: TypedSectorAdmission<
            Error = <R as CheckedGenericFusion>::Error,
            Mode = CheckedGenericAdmissionMode,
        > + CheckedGenericFusion,
    D: AdvancedLinalgScalar,
{
    fn exp(
        tensor: &TensorMap<R, D>,
    ) -> Result<TensorMap<R, D>, GenericTensorError<<R as CheckedGenericFusion>::Error>> {
        if tensor.logical_space().space().homspace().codomain()
            != tensor.logical_space().space().homspace().domain()
        {
            return Err(Error::from(
                tenet_tensors::OperationError::UnsupportedTensorContractScope {
                    message: "exp requires an endomorphism (codomain == domain)",
                },
            )
            .into());
        }
        let local = matches!(&tensor.repr, TypedTensorRepr::Adjoint(_))
            .then(|| tensor.materialized_tensor_uncached())
            .transpose()
            .map_err(GenericTensorError::from)?;
        let body = local
            .as_ref()
            .and_then(TensorMap::owned_body)
            .unwrap_or_else(|| tensor.owned_body().expect("owned representation"));
        let mut dense = tensor.runtime.lease_dense();
        let factor = tenet_matrixalgebra::exp_pade13_direct_into_dyn(
            dense.dense(),
            &BoundDynamicTensorRef::try_new(&body.space, body.materialized_dense_data().as_ref())
                .map_err(Error::from)?,
        )
        .map_err(Error::from)?;
        Ok(wrap_factor_on(&tensor.runtime, factor))
    }
}

impl<R, D> TypedTensorQrDispatch<R, D> for CheckedGenericAdmissionMode
where
    R: TypedSectorAdmission<
            Error = <R as CheckedGenericFusion>::Error,
            Mode = CheckedGenericAdmissionMode,
        > + CheckedGenericFusion,
    D: FactorizationScalar,
{
    fn qr_compact(
        tensor: &TensorMap<R, D>,
    ) -> Result<Qr<TensorMap<R, D>>, GenericTensorError<<R as CheckedGenericFusion>::Error>> {
        tensor.qr_compact_checked_generic()
    }
}

impl<R, D> TypedTensorSvdDispatch<R, D> for CheckedGenericAdmissionMode
where
    R: TypedSectorAdmission<
            Error = <R as CheckedGenericFusion>::Error,
            Mode = CheckedGenericAdmissionMode,
        > + CheckedGenericFusion,
    D: FactorizationScalar,
{
    fn svd_compact(
        tensor: &TensorMap<R, D>,
    ) -> Result<Svd<TensorMap<R, D>>, GenericTensorError<<R as CheckedGenericFusion>::Error>> {
        tensor.svd_compact_checked_generic()
    }

    fn svd_full(
        tensor: &TensorMap<R, D>,
    ) -> Result<Svd<TensorMap<R, D>>, GenericTensorError<<R as CheckedGenericFusion>::Error>> {
        tensor.svd_full_checked_generic()
    }
}

impl<R, D> TypedTensorLqDispatch<R, D> for CheckedGenericAdmissionMode
where
    R: TypedSectorAdmission<
            Error = <R as CheckedGenericFusion>::Error,
            Mode = CheckedGenericAdmissionMode,
        > + CheckedGenericFusion,
    D: FactorizationScalar,
{
    fn lq_compact(
        tensor: &TensorMap<R, D>,
    ) -> Result<Lq<TensorMap<R, D>>, GenericTensorError<<R as CheckedGenericFusion>::Error>> {
        tensor.lq_compact_checked_generic()
    }
}

impl<R, D> TypedTensorFullQrDispatch<R, D> for CheckedGenericAdmissionMode
where
    R: TypedSectorAdmission<
            Error = <R as CheckedGenericFusion>::Error,
            Mode = CheckedGenericAdmissionMode,
        > + CheckedGenericFusion,
    D: FactorizationScalar,
{
    fn qr_full(
        tensor: &TensorMap<R, D>,
    ) -> Result<Qr<TensorMap<R, D>>, GenericTensorError<<R as CheckedGenericFusion>::Error>> {
        tensor.qr_full_checked_generic()
    }
}

impl<R, D> TypedTensorFullLqDispatch<R, D> for CheckedGenericAdmissionMode
where
    R: TypedSectorAdmission<
            Error = <R as CheckedGenericFusion>::Error,
            Mode = CheckedGenericAdmissionMode,
        > + CheckedGenericFusion,
    D: FactorizationScalar,
{
    fn lq_full(
        tensor: &TensorMap<R, D>,
    ) -> Result<Lq<TensorMap<R, D>>, GenericTensorError<<R as CheckedGenericFusion>::Error>> {
        let TypedTensorRepr::Owned(body) = &tensor.repr else {
            return Err(GenericTensorError::Facade(Error::InvalidArgument(
                "checked Generic lq_full does not accept lazy adjoints".to_string(),
            )));
        };
        if let TypedData::Diagonal(spectrum) = body.data.as_ref() {
            if let Some((l_space, q_space, phases, magnitudes)) =
                tenet_matrixalgebra::lq_diagonal_dyn_checked_generic(&body.space, spectrum, true)?
            {
                return Ok(Lq {
                    l: tensor.with_spectrum_on(l_space, magnitudes),
                    q: tensor.with_spectrum_on(q_space, phases),
                });
            }
        }
        let mut dense = tensor.runtime.lease_dense();
        let payload = body.materialized_dense_data();
        let input = BoundDynamicTensorRef::try_new(&body.space, &payload)
            .map_err(|error| GenericTensorError::Facade(error.into()))?;
        let Lq { l, q } = tenet_matrixalgebra::lq_full_dyn_checked_generic(dense.dense(), &input)?;
        Ok(Lq {
            l: wrap_factor_on(&tensor.runtime, l),
            q: wrap_factor_on(&tensor.runtime, q),
        })
    }
}

impl<R, D> TypedTensorSvdValsDispatch<R, D> for CheckedGenericAdmissionMode
where
    R: TypedSectorAdmission<
            Error = <R as CheckedGenericFusion>::Error,
            Mode = CheckedGenericAdmissionMode,
        > + CheckedGenericFusion,
    D: FactorizationScalar,
{
    fn svd_vals(
        tensor: &TensorMap<R, D>,
    ) -> Result<
        Vec<SectorSpectrum<<R as TypedSectorAdmission>::Sector, f64>>,
        GenericTensorError<<R as CheckedGenericFusion>::Error>,
    > {
        tensor.svd_vals_checked_generic()
    }
}

impl<R, D> TypedTensorEighValsDispatch<R, D> for CheckedGenericAdmissionMode
where
    R: TypedSectorAdmission<
            Error = <R as CheckedGenericFusion>::Error,
            Mode = CheckedGenericAdmissionMode,
        > + CheckedGenericFusion,
    D: FactorizationScalar,
{
    fn eigh_vals(
        tensor: &TensorMap<R, D>,
    ) -> Result<
        Vec<SectorSpectrum<<R as TypedSectorAdmission>::Sector, f64>>,
        GenericTensorError<<R as CheckedGenericFusion>::Error>,
    > {
        tensor.eigh_vals_checked_generic()
    }
}

impl<R, D> TypedTensorEighDispatch<R, D> for CheckedGenericAdmissionMode
where
    R: TypedSectorAdmission<
            Error = <R as CheckedGenericFusion>::Error,
            Mode = CheckedGenericAdmissionMode,
        > + CheckedGenericFusion,
    D: FactorizationScalar,
{
    fn eigh_full(
        tensor: &TensorMap<R, D>,
    ) -> Result<Eigh<TensorMap<R, D>>, GenericTensorError<<R as CheckedGenericFusion>::Error>> {
        tensor.eigh_full_checked_generic()
    }
}

impl<R, D> TypedTensorEigValsDispatch<R, D> for CheckedGenericAdmissionMode
where
    R: TypedSectorAdmission<
            Error = <R as CheckedGenericFusion>::Error,
            Mode = CheckedGenericAdmissionMode,
        > + CheckedGenericFusion,
    D: AdvancedLinalgScalar,
{
    fn eig_vals(
        tensor: &TensorMap<R, D>,
    ) -> Result<
        Vec<SectorSpectrum<<R as TypedSectorAdmission>::Sector, num_complex::Complex64>>,
        GenericTensorError<<R as CheckedGenericFusion>::Error>,
    > {
        tensor.eig_vals_checked_generic()
    }
}

impl<R, D> TypedTensorEigDispatch<R, D> for CheckedGenericAdmissionMode
where
    R: TypedSectorAdmission<
            Error = <R as CheckedGenericFusion>::Error,
            Mode = CheckedGenericAdmissionMode,
        > + CheckedGenericFusion,
    D: AdvancedLinalgScalar,
    <D as FactorScalar>::Eig: TensorScalar,
{
    fn eig_full(
        tensor: &TensorMap<R, D>,
    ) -> Result<
        Eig<TensorMap<R, <D as FactorScalar>::Eig>>,
        GenericTensorError<<R as CheckedGenericFusion>::Error>,
    > {
        tensor.eig_full_checked_generic()
    }
}

impl<R, D> TypedTensorReductionDispatch<R, D> for CheckedGenericAdmissionMode
where
    R: TypedSectorAdmission<
            Error = <R as CheckedGenericFusion>::Error,
            Mode = CheckedGenericAdmissionMode,
        > + CheckedGenericFusion
        + CheckedGenericRigidSymbols<Scalar = f64>,
    D: TensorScalar,
{
    fn inner(
        tensor: &TensorMap<R, D>,
        other: &TensorMap<R, D>,
    ) -> Result<D, GenericTensorError<<R as CheckedGenericFusion>::Error>> {
        if !tensor.runtime.same_runtime(&other.runtime) {
            return Err(Error::RuntimeMismatch.into());
        }
        let _host_pool = tensor.runtime.enter_host_pool();
        if tensor.logical_space().space() != other.logical_space().space() {
            return Err(Error::InvalidArgument(
                "tensors live on different spaces or block layouts".to_string(),
            )
            .into());
        }
        if matches!(&tensor.repr, TypedTensorRepr::Owned(body) if matches!(body.data.as_ref(), TypedData::Diagonal(_)))
            || matches!(&other.repr, TypedTensorRepr::Owned(body) if matches!(body.data.as_ref(), TypedData::Diagonal(_)))
        {
            return Err(Error::InvalidArgument(
                "checked Generic reductions require dense payloads".to_string(),
            )
            .into());
        }
        // One `dim(c)` query per coupled sector, evaluated in place. The
        // dense owner asks once per region; the oriented owner asks once per
        // logical block, and a canonical structure lists each coupled sector
        // as one contiguous run of blocks, so a one-entry run cache keeps
        // both at G queries with no allocation and no hashing. Why not a
        // `Vec<f64>` indexed by region: the oriented owner asks by `SectorId`,
        // so that would still need a sector-to-index map or a tenet-core
        // accessor, while the run cache needs neither.
        let provider = tensor.logical_space().provider();
        let mut last: Option<(SectorId, f64)> = None;
        let weight_of = move |sector: SectorId| match last {
            Some((cached, weight)) if cached == sector => Ok(weight),
            _ => {
                let weight = <R::Mode as TypedSpaceModeDispatch<R>>::dim(provider, sector)?;
                last = Some((sector, weight));
                Ok(weight)
            }
        };
        if matches!(&tensor.repr, TypedTensorRepr::Adjoint(_))
            || matches!(&other.repr, TypedTensorRepr::Adjoint(_))
        {
            let (lhs_operand, lhs_data) = tensor.fusion_operand_and_data();
            let (rhs_operand, rhs_data) = other.fusion_operand_and_data();
            return tenet_tensors::oriented_fusion_inner_with(
                tensor.logical_space().space().structure(),
                lhs_operand,
                &lhs_data,
                rhs_operand,
                &rhs_data,
                weight_of,
            );
        }
        let value = coupled_region_inner(
            tensor.logical_space().space().structure(),
            tensor.logical_space().space().nout(),
            tensor
                .owned_body()
                .expect("owned inner input")
                .materialized_dense_data()
                .as_ref(),
            other
                .owned_body()
                .expect("owned inner input")
                .materialized_dense_data()
                .as_ref(),
            weight_of,
        )?;
        Ok(D::from_complex64(value))
    }

    fn norm(
        tensor: &TensorMap<R, D>,
        p: f64,
    ) -> Result<f64, GenericTensorError<<R as CheckedGenericFusion>::Error>> {
        validate_norm_p(p)?;
        if p != 2.0 {
            return Err(Error::InvalidArgument(format!(
                "checked Generic norm supports only p = 2, got {p}"
            ))
            .into());
        }
        // The norm is adjoint invariant, so a lazy adjoint reads its parent in
        // storage order instead of pairing oriented blocks.
        if let TypedTensorRepr::Adjoint(view) = &tensor.repr {
            return Self::norm(
                &TensorMap {
                    runtime: tensor.runtime.clone(),
                    repr: TypedTensorRepr::Owned(Arc::clone(&view.parent)),
                },
                p,
            );
        }
        let body = tensor.owned_body().expect("owned norm input");
        if matches!(body.data.as_ref(), TypedData::Diagonal(_)) {
            return Err(Error::InvalidArgument(
                "checked Generic reductions require dense payloads".to_string(),
            )
            .into());
        }
        let payload = body.materialized_dense_data();
        let data: &[D] = &payload;
        let structure = tensor.logical_space().space().structure();
        let nout = tensor.logical_space().space().nout();
        let provider = tensor.logical_space().provider();
        let weight_of = |sector| <R::Mode as TypedSpaceModeDispatch<R>>::dim(provider, sector);
        // Why not `Self::inner(tensor, tensor)`: it narrows the wide sum to
        // `D`, which for `f32`/`Complex32` rounds `|t|²` to single precision
        // and can leave it subnormal, above the rescaling threshold.
        let sum = coupled_region_inner(structure, nout, data, data, weight_of)?.re;
        rescaled_power_norm(
            sum,
            2.0,
            || max_abs(data.iter().copied()),
            |max| {
                coupled_region_weighted_sum(structure, nout, data, weight_of, |value| {
                    scaled_power(value, max, 2.0)
                })
            },
        )
    }

    fn tr(
        tensor: &TensorMap<R, D>,
    ) -> Result<D, GenericTensorError<<R as CheckedGenericFusion>::Error>> {
        let hom = tensor.logical_space().space().homspace();
        if hom.codomain().legs() != hom.domain().legs() {
            return Err(Error::InvalidArgument(
                "tr() requires an endomorphism (domain == codomain)".to_string(),
            )
            .into());
        }
        if matches!(&tensor.repr, TypedTensorRepr::Owned(body) if matches!(body.data.as_ref(), TypedData::Diagonal(_)))
        {
            return Err(Error::InvalidArgument(
                "checked Generic reductions require dense payloads".to_string(),
            )
            .into());
        }
        if let TypedTensorRepr::Adjoint(view) = &tensor.repr {
            let parent = TensorMap {
                runtime: tensor.runtime.clone(),
                repr: TypedTensorRepr::Owned(Arc::clone(&view.parent)),
            };
            return Ok(FactorScalar::adjoint(Self::tr(&parent)?));
        }
        let provider = tensor.logical_space().provider();
        Ok(D::from_complex64(weighted_trace(
            tensor.logical_space().space().structure(),
            tensor.logical_space().space().nout(),
            tensor
                .owned_body()
                .expect("owned trace input")
                .materialized_dense_data()
                .as_ref(),
            |sector| <R::Mode as TypedSpaceModeDispatch<R>>::dim(provider, sector),
        )?))
    }
}

impl<R> TypedTensorModeDispatch<R> for MultiplicityFreeAdmissionMode
where
    R: TypedSectorAdmission<Error = FusionAlgebraError, Mode = MultiplicityFreeAdmissionMode>
        + CheckedFusionAlgebra,
{
    type FacadeError = Error;

    fn map_provider_error(error: <R as TypedSectorAdmission>::Error) -> Self::FacadeError {
        error.into()
    }

    fn facade_error_into_error(error: Error) -> Error {
        error
    }

    fn coupled_block_dimension(
        provider: &R,
        product: &FusionProductSpace,
        coupled: SectorId,
    ) -> Result<usize, Self::FacadeError> {
        Ok(product
            .coupled_sector_block_dimensions(provider)?
            .get(&coupled)
            .copied()
            .unwrap_or(0))
    }

    fn fuse_sector_content(
        provider: &R,
        left: &[(SectorId, usize)],
        right: &[(SectorId, usize)],
    ) -> Result<Vec<(SectorId, usize)>, Self::FacadeError> {
        fuse_sector_content(
            left,
            right,
            |a, b| CheckedFusionAlgebra::try_fusion_channels(provider, a, b),
            |a, b, c| CheckedFusionAlgebra::try_nsymbol(provider, a, b, c),
        )
        .map_err(Into::into)
    }
}

impl<R> TypedTensorRootDispatch<R> for MultiplicityFreeAdmissionMode
where
    R: TypedSectorAdmission<Error = FusionAlgebraError, Mode = MultiplicityFreeAdmissionMode>
        + MultiplicityFreeRigidSymbols<Scalar = f64>
        + CheckedFusionAlgebra
        + SectorCodec,
{
    fn build_root(
        provider: Arc<R>,
        homspace: FusionTreeHomSpace,
    ) -> Result<BoundDynamicFusionMapSpace<R>, Self::FacadeError> {
        BoundDynamicFusionMapSpace::from_final_homspace_multiplicity_free_checked(
            provider, homspace,
        )
        .map_err(Into::into)
    }
}

impl<R, D> TypedTensorConstructionDispatch<R, D> for MultiplicityFreeAdmissionMode
where
    R: TypedSectorAdmission<Error = FusionAlgebraError, Mode = MultiplicityFreeAdmissionMode>
        + MultiplicityFreeRigidSymbols
        + CheckedFusionAlgebra
        + SectorCodec,
    <R as MultiplicityFreeFusionSymbols>::Scalar:
        CategoricalScalar + tenet_tensors::DenseRecouplingScalar,
    D: TensorScalar
        + tenet_tensors::RecouplingCoefficientAction<<R as MultiplicityFreeFusionSymbols>::Scalar>,
{
    fn build_construction_root(
        provider: Arc<R>,
        homspace: FusionTreeHomSpace,
    ) -> Result<BoundDynamicFusionMapSpace<R>, Self::FacadeError> {
        BoundDynamicFusionMapSpace::from_final_homspace_multiplicity_free_checked(
            provider, homspace,
        )
        .map_err(Into::into)
    }
}

impl<R> TypedTensorModeDispatch<R> for CheckedGenericAdmissionMode
where
    R: TypedSectorAdmission<
            Error = <R as CheckedGenericFusion>::Error,
            Mode = CheckedGenericAdmissionMode,
        > + CheckedGenericFusion,
{
    type FacadeError = GenericTensorError<<R as CheckedGenericFusion>::Error>;

    fn map_provider_error(error: <R as TypedSectorAdmission>::Error) -> Self::FacadeError {
        GenericTensorError::Structure(CheckedGenericStructureError::Provider(error))
    }

    // Why lossy: `TensorRef` is provider-neutral, so its adjoint constructor
    // returns `Error`. The only non-facade failure of a checked-Generic
    // adjoint is a broken invariant of an admitted layout.
    fn facade_error_into_error(error: Self::FacadeError) -> Error {
        match error {
            GenericTensorError::Facade(error) => error,
            other => Error::InvalidArgument(format!("adjoint view: {other}")),
        }
    }

    fn coupled_block_dimension(
        provider: &R,
        product: &FusionProductSpace,
        coupled: SectorId,
    ) -> Result<usize, Self::FacadeError> {
        Ok(
            tenet_matrixalgebra::coupled_sector_block_dimensions_generic_checked(
                product, provider,
            )?
            .get(&coupled)
            .copied()
            .unwrap_or(0),
        )
    }

    fn fuse_sector_content(
        provider: &R,
        left: &[(SectorId, usize)],
        right: &[(SectorId, usize)],
    ) -> Result<Vec<(SectorId, usize)>, Self::FacadeError> {
        fuse_sector_content(
            left,
            right,
            |a, b| CheckedGenericFusion::try_fusion_channels(provider, a, b),
            |a, b, c| CheckedGenericFusion::try_nsymbol(provider, a, b, c),
        )
        .map_err(<Self as TypedTensorModeDispatch<R>>::map_provider_error)
    }
}

impl<R> TypedTensorRootDispatch<R> for CheckedGenericAdmissionMode
where
    R: TypedSectorAdmission<
            Error = <R as CheckedGenericFusion>::Error,
            Mode = CheckedGenericAdmissionMode,
        > + CheckedGenericFusion,
{
    fn build_root(
        provider: Arc<R>,
        homspace: FusionTreeHomSpace,
    ) -> Result<BoundDynamicFusionMapSpace<R>, Self::FacadeError> {
        BoundDynamicFusionMapSpace::from_final_homspace_generic_checked(provider, homspace)
            .map_err(Into::into)
    }
}

impl<R, D> TypedTensorConstructionDispatch<R, D> for CheckedGenericAdmissionMode
where
    R: TypedSectorAdmission<
            Error = <R as CheckedGenericFusion>::Error,
            Mode = CheckedGenericAdmissionMode,
        > + CheckedGenericFusion,
    D: TensorScalar,
{
    fn build_construction_root(
        provider: Arc<R>,
        homspace: FusionTreeHomSpace,
    ) -> Result<BoundDynamicFusionMapSpace<R>, Self::FacadeError> {
        BoundDynamicFusionMapSpace::from_final_homspace_generic_checked(provider, homspace)
            .map_err(Into::into)
    }
}

impl<R, D> TypedTensorTransformDispatch<R, D> for MultiplicityFreeAdmissionMode
where
    R: TypedSectorAdmission<Error = FusionAlgebraError, Mode = MultiplicityFreeAdmissionMode>
        + MultiplicityFreeRigidSymbols
        + CheckedFusionAlgebra
        + SectorCodec,
    <R as MultiplicityFreeFusionSymbols>::Scalar:
        CategoricalScalar + tenet_tensors::DenseRecouplingScalar,
    D: TensorScalar
        + MultiplicityFreeTransformExecution<R, <R as MultiplicityFreeFusionSymbols>::Scalar>,
{
    fn tree_transform(
        tensor: &TensorMap<R, D>,
        operation: TreeTransformOperation,
    ) -> Result<TensorMap<R, D>, Self::FacadeError> {
        D::execute(tensor, operation)
    }
}

impl<R, D> TypedTensorTransformDispatch<R, D> for CheckedGenericAdmissionMode
where
    R: TypedSectorAdmission<
            Error = <R as CheckedGenericFusion>::Error,
            Mode = CheckedGenericAdmissionMode,
        > + CheckedGenericRigidSymbols<Scalar = f64>,
    D: TensorScalar,
{
    fn tree_transform(
        tensor: &TensorMap<R, D>,
        operation: TreeTransformOperation,
    ) -> Result<TensorMap<R, D>, Self::FacadeError> {
        let payload;
        let input = match &tensor.repr {
            TypedTensorRepr::Owned(body) => {
                payload = body.materialized_dense_data();
                CheckedTreeTransformInput::direct(&body.space, &payload)
            }
            TypedTensorRepr::Adjoint(view) => CheckedTreeTransformInput::adjoint(
                &view.logical_space,
                &view.parent.space,
                view.parent_data(),
            ),
        };
        let mut lease = tensor.runtime.lease_context()?;
        let (space, data) = tree_transform_dyn_owned_checked_generic_input_in_context(
            lease.context().generic_lane::<D>()?.tree_context_mut(),
            operation,
            input,
            D::from_real(1.0),
        )?;
        Ok(TensorMap {
            runtime: tensor.runtime.clone(),
            repr: owned_repr(TypedTensorBody::dense(space, data)),
        })
    }
}

impl<R, D> TypedTensorTwistDispatch<R, D> for MultiplicityFreeAdmissionMode
where
    R: TypedSectorAdmission<Error = FusionAlgebraError, Mode = MultiplicityFreeAdmissionMode>
        + MultiplicityFreeRigidSymbols<Scalar = f64>
        + CheckedFusionAlgebra
        + SectorCodec,
    D: TensorScalar,
{
    fn twist(
        tensor: &TensorMap<R, D>,
        legs: &[usize],
        inverse: bool,
    ) -> Result<TensorMap<R, D>, Self::FacadeError> {
        tensor.twist_with_inverse(legs, inverse)
    }
}

impl<R, D> TypedTensorTwistDispatch<R, D> for CheckedGenericAdmissionMode
where
    R: TypedSectorAdmission<
            Error = <R as CheckedGenericFusion>::Error,
            Mode = CheckedGenericAdmissionMode,
        > + CheckedGenericPivotal<Scalar = f64>,
    D: TensorScalar,
{
    fn twist(
        tensor: &TensorMap<R, D>,
        legs: &[usize],
        inverse: bool,
    ) -> Result<TensorMap<R, D>, Self::FacadeError> {
        twist_checked_generic(tensor, legs, inverse)
    }
}

impl<R, D> TypedTensorFlipDispatch<R, D> for MultiplicityFreeAdmissionMode
where
    R: TypedSectorAdmission<Error = FusionAlgebraError, Mode = MultiplicityFreeAdmissionMode>
        + MultiplicityFreeRigidSymbols<Scalar = f64>
        + CheckedFusionAlgebra
        + SectorCodec,
    D: TensorScalar,
{
    fn flip(
        tensor: &TensorMap<R, D>,
        legs: &[usize],
        inverse: bool,
    ) -> Result<TensorMap<R, D>, Self::FacadeError> {
        tensor.flip_multiplicity_free_with_inverse(legs, inverse)
    }
}

impl<R, D> TypedTensorFlipDispatch<R, D> for CheckedGenericAdmissionMode
where
    R: TypedSectorAdmission<
            Error = <R as CheckedGenericFusion>::Error,
            Mode = CheckedGenericAdmissionMode,
        > + CheckedGenericPivotal<Scalar = f64>,
    D: TensorScalar,
{
    fn flip(
        tensor: &TensorMap<R, D>,
        legs: &[usize],
        inverse: bool,
    ) -> Result<TensorMap<R, D>, Self::FacadeError> {
        flip_checked_generic(tensor, legs, inverse)
    }
}

pub(super) fn flip_checked_generic<R, D>(
    tensor: &TensorMap<R, D>,
    legs: &[usize],
    inverse: bool,
) -> Result<TensorMap<R, D>, GenericTensorError<<R as CheckedGenericFusion>::Error>>
where
    R: TypedSectorAdmission<
            Error = <R as CheckedGenericFusion>::Error,
            Mode = CheckedGenericAdmissionMode,
        > + CheckedGenericPivotal<Scalar = f64>,
    D: TensorScalar,
{
    let rank = tensor.rank();
    let name = if inverse { "inverse flip" } else { "flip" };
    if let Some(&leg) = legs.iter().find(|&&leg| leg >= rank) {
        return Err(GenericTensorError::Facade(Error::InvalidArgument(format!(
            "{name} leg {leg} out of range for rank {rank}"
        ))));
    }
    if legs.is_empty() {
        return Ok(tensor.clone());
    }

    let provider = tensor.logical_space().provider();
    if CheckedGenericFusion::braiding_style(provider) == tenet_core::BraidingStyleKind::NoBraiding {
        let leg = legs[0];
        return Err(GenericTensorError::Facade(Error::InvalidArgument(format!(
            "{name} leg {leg} needs the twist and Frobenius-Schur coefficients but the fusion rule has no braiding"
        ))));
    }
    if let TypedTensorRepr::Adjoint(view) = &tensor.repr {
        let logical_space = checked_generic_flip_destination(tensor, legs)?.0;
        let parent = TensorMap {
            runtime: tensor.runtime.clone(),
            repr: TypedTensorRepr::Owned(Arc::clone(&view.parent)),
        };
        let axes = logical_adjoint_axes_to_parent(
            view.parent.space.space().nout(),
            view.parent.space.space().nin(),
            legs,
        );
        let flipped_parent = flip_checked_generic_owned(&parent, &axes, !inverse)?;
        let TypedTensorRepr::Owned(parent) = &flipped_parent.repr else {
            unreachable!("checked-Generic parent flip stays owned")
        };
        return Ok(TensorMap {
            runtime: tensor.runtime.clone(),
            repr: TypedTensorRepr::Adjoint(Arc::new(TypedAdjointView::new(
                Arc::clone(parent),
                logical_space,
            ))),
        });
    }
    flip_checked_generic_owned(tensor, legs, inverse)
}

pub(super) fn flip_checked_generic_owned<R, D>(
    tensor: &TensorMap<R, D>,
    legs: &[usize],
    inverse: bool,
) -> Result<TensorMap<R, D>, GenericTensorError<<R as CheckedGenericFusion>::Error>>
where
    R: TypedSectorAdmission<
            Error = <R as CheckedGenericFusion>::Error,
            Mode = CheckedGenericAdmissionMode,
        > + CheckedGenericPivotal<Scalar = f64>,
    D: TensorScalar,
{
    let nout = tensor.codomain_rank();
    let (space, occurrences) = checked_generic_flip_destination(tensor, legs)?;

    // Why stage first: any pivotal failure leaves the input and destination
    // unpublished, while replay over this table cannot query the provider.
    let provider = tensor.logical_space().provider();
    let mut staged = HashMap::<SectorId, (f64, f64)>::new();
    for block_index in 0..space.space().structure().block_count() {
        let block = space
            .space()
            .structure()
            .block(block_index)
            .map_err(Error::from)
            .map_err(GenericTensorError::Facade)?;
        let BlockKey::FusionTree(key) = block.key() else {
            continue;
        };
        for &(leg, _) in &occurrences {
            let sector = uncoupled_sector_of_leg(key, nout, leg);
            if let std::collections::hash_map::Entry::Vacant(entry) = staged.entry(sector) {
                let chi = provider
                    .try_frobenius_schur_phase_scalar(sector)
                    .map_err(|error| {
                        GenericTensorError::Plan(CheckedGenericPlanError::Provider(error))
                    })?;
                let theta = provider.try_twist_scalar(sector).map_err(|error| {
                    GenericTensorError::Plan(CheckedGenericPlanError::Provider(error))
                })?;
                entry.insert((chi, theta));
            }
        }
    }
    let block_factor = |key: &FusionTreePairKey| {
        occurrences
            .iter()
            .map(|&(leg, dual)| {
                let (chi, theta) = staged[&uncoupled_sector_of_leg(key, nout, leg)];
                if leg < nout {
                    if dual {
                        if inverse {
                            1.0
                        } else {
                            chi * theta
                        }
                    } else if inverse {
                        chi * theta
                    } else {
                        1.0
                    }
                } else if dual {
                    if inverse {
                        theta
                    } else {
                        chi
                    }
                } else if inverse {
                    chi
                } else {
                    theta
                }
            })
            .product::<f64>()
    };
    let mut data = tensor
        .owned_body()
        .expect("owned checked-Generic flip input")
        .materialized_dense_data()
        .as_ref()
        .to_vec();
    scale_blocks_impl(space.space(), &mut data, &|key| match key {
        BlockKey::FusionTree(key) => block_factor(key),
        _ => 1.0,
    })
    .map_err(GenericTensorError::Facade)?;
    Ok(TensorMap {
        runtime: tensor.runtime.clone(),
        repr: owned_repr(TypedTensorBody::dense(space, data)),
    })
}

#[expect(
    clippy::type_complexity,
    reason = "the flip staging contract returns the admitted space with occurrence metadata"
)]
pub(super) fn checked_generic_flip_destination<R, D>(
    tensor: &TensorMap<R, D>,
    legs: &[usize],
) -> Result<
    (BoundDynamicFusionMapSpace<R>, Vec<(usize, bool)>),
    GenericTensorError<<R as CheckedGenericFusion>::Error>,
>
where
    R: TypedSectorAdmission<
            Error = <R as CheckedGenericFusion>::Error,
            Mode = CheckedGenericAdmissionMode,
        > + CheckedGenericPivotal<Scalar = f64>,
    D: TensorScalar,
{
    let (homspace, occurrences) =
        flip_toggled_homspace(tensor.logical_space().space().homspace(), legs);
    let space = <R::Mode as TypedTensorRootDispatch<R>>::build_root(
        Arc::clone(tensor.logical_space().provider_arc()),
        homspace,
    )?;
    check_flip_layout_identity(
        tensor.logical_space().space().structure(),
        space.space().structure(),
    )
    .map_err(GenericTensorError::Facade)?;
    Ok((space, occurrences))
}

pub(super) fn twist_checked_generic<R, D>(
    tensor: &TensorMap<R, D>,
    legs: &[usize],
    inverse: bool,
) -> Result<TensorMap<R, D>, GenericTensorError<<R as CheckedGenericFusion>::Error>>
where
    R: TypedSectorAdmission<
            Error = <R as CheckedGenericFusion>::Error,
            Mode = CheckedGenericAdmissionMode,
        > + CheckedGenericPivotal<Scalar = f64>,
    D: TensorScalar,
{
    let rank = tensor.rank();
    let name = if inverse { "inverse twist" } else { "twist" };
    if let Some(&leg) = legs.iter().find(|&&leg| leg >= rank) {
        return Err(GenericTensorError::Facade(Error::InvalidArgument(format!(
            "{name} leg {leg} out of range for rank {rank}"
        ))));
    }
    if legs.is_empty() {
        return Ok(tensor.clone());
    }

    let provider = tensor.logical_space().provider();
    let homspace = tensor.logical_space().space().homspace();
    let braiding = CheckedGenericFusion::braiding_style(provider);
    if braiding == tenet_core::BraidingStyleKind::NoBraiding {
        let vacuum = CheckedGenericFusion::vacuum(provider);
        let nout = homspace.codomain().len();
        for &leg in legs {
            let sectors = if leg < nout {
                homspace.codomain().legs()[leg].sectors()
            } else {
                homspace.domain().legs()[leg - nout].sectors()
            };
            if sectors.iter().any(|&sector| sector != vacuum) {
                return Err(GenericTensorError::Facade(Error::InvalidArgument(format!(
                    "{name} leg {leg} carries non-unit sectors but the fusion rule has no braiding"
                ))));
            }
        }
        return Ok(tensor.clone());
    }
    if braiding.is_bosonic() {
        return Ok(tensor.clone());
    }
    if let TypedTensorRepr::Adjoint(view) = &tensor.repr {
        let parent = TensorMap {
            runtime: tensor.runtime.clone(),
            repr: TypedTensorRepr::Owned(Arc::clone(&view.parent)),
        };
        let axes = logical_adjoint_axes_to_parent(
            view.parent.space.space().nout(),
            view.parent.space.space().nin(),
            legs,
        );
        let twisted_parent = twist_checked_generic_owned(&parent, &axes, !inverse)?;
        let TypedTensorRepr::Owned(parent) = &twisted_parent.repr else {
            unreachable!("checked-Generic parent twist stays owned")
        };
        // Why not call `adjoint()`: the original view already owns the exact
        // admitted logical space, and re-deriving it would query the provider
        // after all fallible twist values had been staged.
        return Ok(TensorMap {
            runtime: tensor.runtime.clone(),
            repr: TypedTensorRepr::Adjoint(Arc::new(TypedAdjointView::new(
                Arc::clone(parent),
                view.logical_space.clone(),
            ))),
        });
    }

    twist_checked_generic_owned(tensor, legs, inverse)
}

pub(super) fn twist_checked_generic_owned<R, D>(
    tensor: &TensorMap<R, D>,
    legs: &[usize],
    inverse: bool,
) -> Result<TensorMap<R, D>, GenericTensorError<<R as CheckedGenericFusion>::Error>>
where
    R: TypedSectorAdmission<
            Error = <R as CheckedGenericFusion>::Error,
            Mode = CheckedGenericAdmissionMode,
        > + CheckedGenericPivotal<Scalar = f64>,
    D: TensorScalar,
{
    // Why stage first: a late provider failure must leave no scaled payload to
    // publish, and replay over the staged table must never call the provider.
    let provider = tensor.logical_space().provider();
    let nout = tensor.codomain_rank();
    let structure = tensor.logical_space().space().structure();
    let mut staged = HashMap::<SectorId, f64>::new();
    for block_index in 0..structure.block_count() {
        let block = structure
            .block(block_index)
            .map_err(Error::from)
            .map_err(GenericTensorError::Facade)?;
        let BlockKey::FusionTree(key) = block.key() else {
            continue;
        };
        for &leg in legs {
            let sector = uncoupled_sector_of_leg(key, nout, leg);
            if let std::collections::hash_map::Entry::Vacant(entry) = staged.entry(sector) {
                let value = provider.try_twist_scalar(sector).map_err(|error| {
                    GenericTensorError::Plan(CheckedGenericPlanError::Provider(error))
                })?;
                entry.insert(value);
            }
        }
    }
    if staged.values().all(|&value| value == 1.0) {
        return Ok(tensor.clone());
    }

    // Checked pivotal coefficients are exactly real here, so conjugation for
    // the inverse operation leaves every staged value unchanged.
    let _ = inverse;
    let block_factor = |key: &FusionTreePairKey| {
        legs.iter()
            .map(|&leg| staged[&uncoupled_sector_of_leg(key, nout, leg)])
            .product::<f64>()
    };
    let mut data = tensor
        .owned_body()
        .expect("owned checked-Generic twist input")
        .materialized_dense_data()
        .as_ref()
        .to_vec();
    scale_blocks_impl(
        tensor.logical_space().space(),
        &mut data,
        &|key| match key {
            BlockKey::FusionTree(key) => block_factor(key),
            _ => 1.0,
        },
    )
    .map_err(GenericTensorError::Facade)?;
    Ok(TensorMap {
        runtime: tensor.runtime.clone(),
        repr: owned_repr(TypedTensorBody::dense(tensor.logical_space().clone(), data)),
    })
}

impl<R, D> TypedTensorProductDispatch<R, D> for MultiplicityFreeAdmissionMode
where
    R: TypedSectorAdmission<Error = FusionAlgebraError, Mode = MultiplicityFreeAdmissionMode>
        + MultiplicityFreeRigidSymbols
        + CheckedFusionAlgebra
        + SectorCodec
        + CanonicalUnitFusionRule,
    <R as MultiplicityFreeFusionSymbols>::Scalar:
        CategoricalScalar + tenet_tensors::DenseRecouplingScalar,
    D: TensorScalar
        + tenet_tensors::RecouplingCoefficientAction<<R as MultiplicityFreeFusionSymbols>::Scalar>,
{
    fn tensor_product(
        lhs: &TensorMap<R, D>,
        rhs: &TensorMap<R, D>,
    ) -> Result<TensorMap<R, D>, Self::FacadeError> {
        let lhs_owned = lhs.materialized_tensor_uncached()?;
        let rhs_owned = rhs.materialized_tensor_uncached()?;
        let lhs_body = lhs_owned
            .owned_body()
            .expect("uncached materialization is owned");
        let rhs_body = rhs_owned
            .owned_body()
            .expect("uncached materialization is owned");
        let (space, data) = tensorproduct_owned_multiplicity_free(
            BoundDynamicTensorRef::try_new(
                &lhs_body.space,
                lhs_body.materialized_dense_data().as_ref(),
            )?,
            BoundDynamicTensorRef::try_new(
                &rhs_body.space,
                rhs_body.materialized_dense_data().as_ref(),
            )?,
        )?;
        Ok(TensorMap {
            runtime: lhs.runtime.clone(),
            repr: owned_repr(TypedTensorBody::dense(space, data)),
        })
    }
}

impl<R, D> TypedTensorProductDispatch<R, D> for CheckedGenericAdmissionMode
where
    R: TypedSectorAdmission<
            Error = <R as CheckedGenericFusion>::Error,
            Mode = CheckedGenericAdmissionMode,
        > + CheckedGenericRigidSymbols<Scalar = f64>,
    D: TensorScalar,
{
    fn tensor_product(
        lhs: &TensorMap<R, D>,
        rhs: &TensorMap<R, D>,
    ) -> Result<TensorMap<R, D>, Self::FacadeError> {
        let lhs_owned = lhs.materialized_tensor_uncached()?;
        let rhs_owned = rhs.materialized_tensor_uncached()?;
        let lhs_body = lhs_owned
            .owned_body()
            .expect("uncached materialization is owned");
        let rhs_body = rhs_owned
            .owned_body()
            .expect("uncached materialization is owned");
        let (space, data) = tensorproduct_owned_checked_generic(
            &lhs_body.space,
            lhs_body.materialized_dense_data().as_ref(),
            &rhs_body.space,
            rhs_body.materialized_dense_data().as_ref(),
        )?;
        Ok(TensorMap {
            runtime: lhs.runtime.clone(),
            repr: owned_repr(TypedTensorBody::dense(space, data)),
        })
    }
}

impl<R, D> TypedTensorContractDispatch<R, D> for MultiplicityFreeAdmissionMode
where
    R: TypedSectorAdmission<Error = FusionAlgebraError, Mode = MultiplicityFreeAdmissionMode>
        + MultiplicityFreeRigidSymbols
        + CheckedFusionAlgebra
        + SectorCodec,
    <R as MultiplicityFreeFusionSymbols>::Scalar:
        CategoricalScalar + tenet_tensors::DenseRecouplingScalar,
    D: TensorScalar
        + MultiplicityFreeContractExecution<R, <R as MultiplicityFreeFusionSymbols>::Scalar>,
{
    fn contract(
        lhs: &TensorMap<R, D>,
        rhs: &TensorMap<R, D>,
        spec: &ContractSpec<'_>,
    ) -> Result<TensorMap<R, D>, Self::FacadeError> {
        reject_non_symmetric_contraction(lhs.logical_space().provider().braiding_style())?;
        D::contract(lhs, rhs, spec)
    }

    fn compose(
        lhs: &TensorMap<R, D>,
        rhs: &TensorMap<R, D>,
    ) -> Result<TensorMap<R, D>, Self::FacadeError> {
        D::compose(lhs, rhs)
    }
}

impl<R, D> TypedTensorContractDispatch<R, D> for CheckedGenericAdmissionMode
where
    R: TypedSectorAdmission<
            Error = <R as CheckedGenericFusion>::Error,
            Mode = CheckedGenericAdmissionMode,
        > + CheckedGenericRigidSymbols<Scalar = f64>,
    D: TensorScalar,
{
    fn contract(
        lhs: &TensorMap<R, D>,
        rhs: &TensorMap<R, D>,
        spec: &ContractSpec<'_>,
    ) -> Result<TensorMap<R, D>, Self::FacadeError> {
        reject_non_symmetric_contraction(CheckedGenericFusion::braiding_style(
            lhs.logical_space().provider(),
        ))
        .map_err(Error::from)?;
        contract_checked_generic(lhs, rhs, spec)
    }

    fn compose(
        lhs: &TensorMap<R, D>,
        rhs: &TensorMap<R, D>,
    ) -> Result<TensorMap<R, D>, Self::FacadeError> {
        // The canonical axes need no braid, so composition bypasses
        // `contract`'s braiding boundaries (TensorKit `mul!`).
        let (lhs_body, rhs_body) = checked_generic_owned_bodies(lhs, rhs)?;
        let mut lease = lhs.runtime.lease_context()?;
        let (space, data) = tensorcompose_owned_checked_generic_in_context(
            lease.context().generic_lane::<D>()?,
            &lhs_body.space,
            lhs_body.materialized_dense_data().as_ref(),
            &rhs_body.space,
            rhs_body.materialized_dense_data().as_ref(),
        )?;
        Ok(TensorMap {
            runtime: lhs.runtime.clone(),
            repr: owned_repr(TypedTensorBody::dense(space, data)),
        })
    }
}

#[allow(clippy::type_complexity)]
pub(super) fn checked_generic_owned_bodies<'a, R, D>(
    lhs: &'a TensorMap<R, D>,
    rhs: &'a TensorMap<R, D>,
) -> Result<(&'a TypedTensorBody<R, D>, &'a TypedTensorBody<R, D>), Error>
where
    R: TypedSectorAdmission,
    D: TensorScalar,
{
    let (TypedTensorRepr::Owned(lhs_body), TypedTensorRepr::Owned(rhs_body)) =
        (&lhs.repr, &rhs.repr)
    else {
        return Err(Error::InvalidArgument(
            "checked Generic contraction currently requires direct owned tensors".to_string(),
        ));
    };
    Ok((lhs_body, rhs_body))
}

pub(super) fn contract_checked_generic<R, D>(
    lhs: &TensorMap<R, D>,
    rhs: &TensorMap<R, D>,
    spec: &ContractSpec<'_>,
) -> Result<TensorMap<R, D>, GenericTensorError<<R as CheckedGenericFusion>::Error>>
where
    R: TypedSectorAdmission<
            Error = <R as CheckedGenericFusion>::Error,
            Mode = CheckedGenericAdmissionMode,
        > + CheckedGenericRigidSymbols<Scalar = f64>,
    D: TensorScalar,
{
    let (lhs_body, rhs_body) = checked_generic_owned_bodies(lhs, rhs)?;
    let mut lease = lhs.runtime.lease_context()?;
    let (space, data) = tensorcontract_owned_checked_generic_in_context(
        lease.context().generic_lane::<D>()?,
        &lhs_body.space,
        lhs_body.materialized_dense_data().as_ref(),
        &rhs_body.space,
        rhs_body.materialized_dense_data().as_ref(),
        TensorContractSpec::new(
            spec.lhs,
            spec.rhs,
            OutputAxisOrder::from_axes(&spec.output_axes()),
        ),
        spec.codomain.len(),
    )?;
    Ok(TensorMap {
        runtime: lhs.runtime.clone(),
        repr: owned_repr(TypedTensorBody::dense(space, data)),
    })
}
