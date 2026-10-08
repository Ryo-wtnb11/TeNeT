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
