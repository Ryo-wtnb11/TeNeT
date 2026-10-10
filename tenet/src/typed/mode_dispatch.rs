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
        <Self as tenet_tensors::RigidCoefficientAlgebra<R>>::dim(provider, sector)
            .map_err(Into::into)
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
        <Self as tenet_tensors::RigidCoefficientAlgebra<R>>::dim(provider, sector).map_err(
            |error| match error {
                tenet_tensors::CheckedGenericPlanError::Provider(error) => {
                    <Self as TypedTensorModeDispatch<R>>::map_provider_error(error)
                }
                other => GenericTensorError::Plan(other),
            },
        )
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
        tenet_matrixalgebra::seam::decide_bond_truncation(provider, spectra, truncation)
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
        tenet_matrixalgebra::seam::decide_bond_truncation_generic_checked(
            provider, spectra, truncation,
        )
        .map_err(Into::into)
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

    fn fusion_style(provider: &R) -> tenet_core::FusionStyleKind {
        tenet_core::FusionRule::fusion_style(provider)
    }

    fn braiding_style(provider: &R) -> tenet_core::BraidingStyleKind {
        tenet_core::FusionRule::braiding_style(provider)
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

    fn build_root_with(
        provider: Arc<R>,
        build_homspace: impl FnOnce() -> Result<FusionTreeHomSpace, Self::FacadeError>,
    ) -> Result<BoundDynamicFusionMapSpace<R>, Self::FacadeError> {
        BoundDynamicFusionMapSpace::from_final_homspace_multiplicity_free_checked_with(
            provider,
            build_homspace,
        )
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

    // Why not ask the provider: admission already required the Generic
    // style (`validate_checked_generic_style`), so an op on admitted data
    // never asks again and cannot be swayed by a provider whose answer
    // changed since.
    fn fusion_style(_provider: &R) -> tenet_core::FusionStyleKind {
        tenet_core::FusionStyleKind::Generic
    }

    fn braiding_style(provider: &R) -> tenet_core::BraidingStyleKind {
        CheckedGenericFusion::braiding_style(provider)
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
            tenet_matrixalgebra::seam::coupled_sector_block_dimensions_generic_checked(
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

    fn build_root_with(
        provider: Arc<R>,
        build_homspace: impl FnOnce() -> Result<FusionTreeHomSpace, Self::FacadeError>,
    ) -> Result<BoundDynamicFusionMapSpace<R>, Self::FacadeError> {
        BoundDynamicFusionMapSpace::from_final_homspace_generic_checked_with(
            provider,
            build_homspace,
        )
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
    fn try_compact_transform(
        tensor: &TensorMap<R, D>,
        operation: &TreeTransformOperation,
    ) -> Result<Option<TensorMap<R, D>>, Error> {
        D::try_compact(tensor, operation)
    }

    fn try_lazy_adjoint_transform(
        tensor: &TensorMap<R, D>,
        operation: &TreeTransformOperation,
    ) -> Result<Option<TensorMap<R, D>>, Error> {
        D::try_lazy_adjoint(tensor, operation)
    }

    fn transform(
        tensor: &TensorMap<R, D>,
        operation: TreeTransformOperation,
    ) -> Result<(BoundDynamicFusionMapSpace<R>, Vec<D>), Error> {
        D::transform(tensor, operation)
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
    fn try_lazy_adjoint_transform(
        tensor: &TensorMap<R, D>,
        operation: &TreeTransformOperation,
    ) -> Result<Option<TensorMap<R, D>>, Self::FacadeError> {
        lazy_adjoint_of_transformed_parent(tensor, operation)
    }

    fn transform(
        tensor: &TensorMap<R, D>,
        operation: TreeTransformOperation,
    ) -> Result<(BoundDynamicFusionMapSpace<R>, Vec<D>), Self::FacadeError> {
        // Why owned: the facade routes every lazy adjoint through
        // `try_lazy_adjoint_transform` above.
        let body = tensor
            .owned_body()
            .ok_or_else(|| internal_layout_error("checked Generic transform input is owned"))?;
        let data = body.materialized_dense_data();
        let mut lease = tensor.runtime.lease_context()?;
        Ok(lease
            .context()
            .generic_lane::<D>()?
            .tree_context_mut()
            .tree_transform_owned_checked_generic_in(
                &body.space,
                None,
                data.as_ref(),
                &operation,
                D::from_real(1.0),
            )?)
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
    fn twist_values<'a>(
        provider: &'a R,
        structure: &tenet_core::BlockStructure,
        codomain_rank: usize,
        legs: &[usize],
    ) -> Result<Option<impl Fn(SectorId) -> f64 + 'a>, Error> {
        if twist_is_identity_over_blocks(provider, structure, codomain_rank, legs)? {
            return Ok(None);
        }
        Ok(Some(move |sector| provider.twist_scalar(sector)))
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
    fn twist_values<'a>(
        provider: &'a R,
        structure: &tenet_core::BlockStructure,
        codomain_rank: usize,
        legs: &[usize],
    ) -> Result<Option<impl Fn(SectorId) -> f64 + 'a>, Self::FacadeError> {
        // Why stage first: a late provider failure must leave no scaled
        // payload to publish, and replay over the staged table must never
        // call the provider.
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
                let sector = uncoupled_sector_of_leg(key, codomain_rank, leg);
                if let std::collections::hash_map::Entry::Vacant(entry) = staged.entry(sector) {
                    let value = provider.try_twist_scalar(sector).map_err(|error| {
                        GenericTensorError::Plan(CheckedGenericPlanError::Provider(error))
                    })?;
                    entry.insert(value);
                }
            }
        }
        if staged.values().all(|&value| value == 1.0) {
            return Ok(None);
        }
        Ok(Some(move |sector| staged[&sector]))
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
    fn root(
        space: &BoundDynamicFusionMapSpace<R>,
        homspace: FusionTreeHomSpace,
    ) -> Result<BoundDynamicFusionMapSpace<R>, Error> {
        Ok(space.derive_from_final_homspace(homspace)?)
    }

    fn pivotal_values<'a>(
        provider: &'a R,
        _structure: &tenet_core::BlockStructure,
        _codomain_rank: usize,
        _occurrences: &[(usize, bool)],
    ) -> Result<impl Fn(SectorId) -> (f64, f64) + 'a, Error> {
        Ok(move |sector| {
            (
                provider.frobenius_schur_phase_scalar(sector),
                provider.twist_scalar(sector),
            )
        })
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
    fn root(
        space: &BoundDynamicFusionMapSpace<R>,
        homspace: FusionTreeHomSpace,
    ) -> Result<BoundDynamicFusionMapSpace<R>, Self::FacadeError> {
        <Self as TypedTensorRootDispatch<R>>::build_root(Arc::clone(space.provider_arc()), homspace)
    }

    fn pivotal_values<'a>(
        provider: &'a R,
        structure: &tenet_core::BlockStructure,
        codomain_rank: usize,
        occurrences: &[(usize, bool)],
    ) -> Result<impl Fn(SectorId) -> (f64, f64) + 'a, Self::FacadeError> {
        // Why stage first: any pivotal failure leaves the input and
        // destination unpublished, while replay over this table cannot query
        // the provider.
        let mut staged = HashMap::<SectorId, (f64, f64)>::new();
        for block_index in 0..structure.block_count() {
            let block = structure
                .block(block_index)
                .map_err(Error::from)
                .map_err(GenericTensorError::Facade)?;
            let BlockKey::FusionTree(key) = block.key() else {
                continue;
            };
            for &(leg, _) in occurrences {
                let sector = uncoupled_sector_of_leg(key, codomain_rank, leg);
                if let std::collections::hash_map::Entry::Vacant(entry) = staged.entry(sector) {
                    let chi =
                        provider
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
        Ok(move |sector| staged[&sector])
    }
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
    fn product(
        lhs_space: &BoundDynamicFusionMapSpace<R>,
        lhs_data: &[D],
        rhs_space: &BoundDynamicFusionMapSpace<R>,
        rhs_data: &[D],
    ) -> Result<(BoundDynamicFusionMapSpace<R>, Vec<D>), Error> {
        Ok(tensorproduct_owned_multiplicity_free(
            BoundDynamicTensorRef::try_new(lhs_space, lhs_data)?,
            BoundDynamicTensorRef::try_new(rhs_space, rhs_data)?,
        )?)
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
    fn product(
        lhs_space: &BoundDynamicFusionMapSpace<R>,
        lhs_data: &[D],
        rhs_space: &BoundDynamicFusionMapSpace<R>,
        rhs_data: &[D],
    ) -> Result<(BoundDynamicFusionMapSpace<R>, Vec<D>), Self::FacadeError> {
        Ok(tensorproduct_owned_checked_generic(
            lhs_space, lhs_data, rhs_space, rhs_data,
        )?)
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
    fn try_compact_contract(
        lhs: &TensorMap<R, D>,
        rhs: &TensorMap<R, D>,
        spec: &ContractSpec<'_>,
    ) -> Result<Option<TensorMap<R, D>>, Error> {
        D::try_compact_contract(lhs, rhs, spec)
    }

    fn try_compact_compose(
        lhs: &TensorMap<R, D>,
        rhs: &TensorMap<R, D>,
    ) -> Result<Option<TensorMap<R, D>>, Error> {
        D::try_compact_compose(lhs, rhs)
    }

    fn contract(
        lhs: &TensorMap<R, D>,
        rhs: &TensorMap<R, D>,
        spec: &ContractSpec<'_>,
    ) -> Result<(BoundDynamicFusionMapSpace<R>, Vec<D>), Error> {
        D::contract(lhs, rhs, spec)
    }

    fn compose(
        lhs: &TensorMap<R, D>,
        rhs: &TensorMap<R, D>,
    ) -> Result<(BoundDynamicFusionMapSpace<R>, Vec<D>), Error> {
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
    ) -> Result<(BoundDynamicFusionMapSpace<R>, Vec<D>), Self::FacadeError> {
        let output_axes = spec.output_axes();
        let mut lease = lhs.runtime.lease_context()?;
        let (lhs_operand, lhs_data) = lhs.fusion_operand_and_data();
        let (rhs_operand, rhs_data) = rhs.fusion_operand_and_data();
        Ok(lease
            .context()
            .generic_lane::<D>()?
            .tensorcontract_checked_generic_in(
                (lhs.logical_space(), lhs_operand, &lhs_data),
                (rhs.logical_space(), rhs_operand, &rhs_data),
                (spec.lhs, spec.rhs, OutputAxisOrder::from_axes(&output_axes)),
                spec.codomain.len(),
            )?)
    }

    fn compose(
        lhs: &TensorMap<R, D>,
        rhs: &TensorMap<R, D>,
    ) -> Result<(BoundDynamicFusionMapSpace<R>, Vec<D>), Self::FacadeError> {
        let mut lease = lhs.runtime.lease_context()?;
        let (lhs_operand, lhs_data) = lhs.fusion_operand_and_data();
        let (rhs_operand, rhs_data) = rhs.fusion_operand_and_data();
        Ok(lease
            .context()
            .generic_lane::<D>()?
            .tensorcompose_checked_generic_in(
                (lhs.logical_space(), lhs_operand, &lhs_data),
                (rhs.logical_space(), rhs_operand, &rhs_data),
            )?)
    }
}
