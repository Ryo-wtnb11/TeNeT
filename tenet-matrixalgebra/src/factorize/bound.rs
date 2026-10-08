use super::*;

// ---------------------------------------------------------------------------
// Dynamic-rank representation.
// ---------------------------------------------------------------------------

#[cfg(test)]
/// Dynamic-rank factor tensor: an expert-layer space handle plus flat data in
/// the coupled-sector matrix layout (the same pair `tenet_tensors::adjoint_dyn`
/// returns).
pub(crate) type DynFactor<D> = (DynamicFusionMapSpace, Vec<D>);

#[cfg(test)]
/// Typed tensor plus the sole provider authority accepted by provider-sensitive
/// factorization and matrix-function APIs.
pub(crate) struct BoundTensorMapRef<'a, R, D, const NOUT: usize, const NIN: usize> {
    pub(super) space: &'a BoundDynamicFusionMapSpace<R>,
    pub(super) tensor: &'a TensorMap<D, NOUT, NIN>,
}

#[cfg(test)]
/// Owned typed tensor that retains the provider authority for its fusion
/// space. Provider-sensitive operations consume this type or its borrowed
/// view instead of accepting an independently supplied rule.
///
/// The fields are private so a tensor and an unrelated provider-bound space
/// cannot be paired without validation.
///
/// ```compile_fail
/// use tenet_core::TensorMap;
/// use tenet_matrixalgebra::BoundTensorMap;
/// use tenet_tensors::BoundDynamicFusionMapSpace;
///
/// fn forge<R, D>(
///     space: BoundDynamicFusionMapSpace<R>,
///     tensor: TensorMap<D, 1, 1>,
/// ) -> BoundTensorMap<R, D, 1, 1> {
///     BoundTensorMap { space, tensor }
/// }
/// ```
#[derive(Clone, Debug)]
pub(crate) struct BoundTensorMap<R, D, const NOUT: usize, const NIN: usize> {
    pub(super) space: BoundDynamicFusionMapSpace<R>,
    pub(super) tensor: TensorMap<D, NOUT, NIN>,
}

#[cfg(test)]
impl<R, D, const NOUT: usize, const NIN: usize> BoundTensorMap<R, D, NOUT, NIN>
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64>,
{
    pub fn try_new(
        provider: Arc<R>,
        tensor: TensorMap<D, NOUT, NIN>,
    ) -> Result<Self, OperationError> {
        let space =
            BoundDynamicFusionMapSpace::bind_multiplicity_free(dyn_space_of(&tensor)?, provider)?;
        BoundDynamicTensorRef::try_new(&space, tensor.data())?;
        Ok(Self { space, tensor })
    }

    pub fn space(&self) -> &BoundDynamicFusionMapSpace<R> {
        &self.space
    }

    pub fn tensor(&self) -> &TensorMap<D, NOUT, NIN> {
        &self.tensor
    }

    pub fn data(&self) -> &[D] {
        self.tensor.data()
    }

    pub fn as_ref(&self) -> BoundTensorMapRef<'_, R, D, NOUT, NIN> {
        BoundTensorMapRef {
            space: &self.space,
            tensor: &self.tensor,
        }
    }

    pub fn into_parts(self) -> (BoundDynamicFusionMapSpace<R>, TensorMap<D, NOUT, NIN>) {
        (self.space, self.tensor)
    }
}

#[cfg(test)]
impl<R, D, const NOUT: usize, const NIN: usize> std::ops::Deref
    for BoundTensorMap<R, D, NOUT, NIN>
{
    type Target = TensorMap<D, NOUT, NIN>;

    fn deref(&self) -> &Self::Target {
        &self.tensor
    }
}

#[cfg(test)]
impl<'a, R, D, const NOUT: usize, const NIN: usize> BoundTensorMapRef<'a, R, D, NOUT, NIN>
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64>,
{
    pub fn space(&self) -> &'a BoundDynamicFusionMapSpace<R> {
        self.space
    }

    pub fn tensor(&self) -> &'a TensorMap<D, NOUT, NIN> {
        self.tensor
    }

    pub(crate) fn dynamic(&self) -> BoundDynamicTensorRef<'_, R, D> {
        // Why not expose an unchecked constructor: `BoundTensorMap::try_new`
        // establishes this invariant once, while external dynamic inputs must
        // continue through `BoundDynamicTensorRef::try_new`.
        BoundDynamicTensorRef::try_new(self.space, self.tensor.data())
            .expect("BoundTensorMap preserves its validated dynamic layout")
    }
}

/// Owned dynamic factor that retains the provider used to create its complete
/// fusion space.
pub struct BoundDynFactor<R, D> {
    pub(super) space: BoundDynamicFusionMapSpace<R>,
    pub(super) data: Vec<D>,
}

pub(super) type DynamicFactorPair<R, D> = (BoundDynFactor<R, D>, BoundDynFactor<R, D>);

impl<R, D> fmt::Debug for BoundDynFactor<R, D> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("BoundDynFactor")
            .field("space", &self.space)
            .field("data_len", &self.data.len())
            .finish()
    }
}

impl<R, D> Clone for BoundDynFactor<R, D>
where
    D: Clone,
{
    fn clone(&self) -> Self {
        Self {
            space: self.space.clone(),
            data: self.data.clone(),
        }
    }
}

impl<R, D> BoundDynFactor<R, D> {
    pub(crate) fn from_bound(
        space: BoundDynamicFusionMapSpace<R>,
        data: Vec<D>,
        expected_nout: usize,
        expected_nin: usize,
    ) -> Result<Self, OperationError> {
        if space.space().nout() != expected_nout || space.space().nin() != expected_nin {
            return Err(OperationError::RankMismatch {
                expected: expected_nout + expected_nin,
                actual: space.space().rank(),
            });
        }
        let expected = space
            .space()
            .required_len()
            .map_err(OperationError::from_core_preserving_context)?;
        if data.len() != expected {
            return Err(OperationError::from_core_preserving_context(
                CoreError::DimensionMismatch {
                    expected,
                    actual: data.len(),
                },
            ));
        }
        Ok(Self { space, data })
    }

    pub fn space(&self) -> &BoundDynamicFusionMapSpace<R> {
        &self.space
    }

    pub fn data(&self) -> &[D] {
        &self.data
    }

    #[cfg(test)]
    pub(crate) fn data_mut(&mut self) -> &mut [D] {
        &mut self.data
    }

    pub fn into_parts(self) -> (BoundDynamicFusionMapSpace<R>, Vec<D>) {
        (self.space, self.data)
    }
}

#[cfg(test)]
pub(crate) fn adjoint_bound_factor<R, D>(
    factor: &BoundDynFactor<R, D>,
) -> Result<BoundDynFactor<R, D>, OperationError>
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64>,
    D: FactorScalar,
{
    let (space, data) = tenet_tensors::adjoint_bound_dyn(factor.space(), factor.data())?;
    let nout = space.space().nout();
    let nin = space.space().nin();
    BoundDynFactor::from_bound(space, data, nout, nin)
}

#[cfg(test)]
/// Rank-erases the fusion space of a typed tensor (shared handles, no copy).
pub(crate) fn dyn_space_of<D, const NOUT: usize, const NIN: usize>(
    tensor: &TensorMap<D, NOUT, NIN>,
) -> Result<DynamicFusionMapSpace, OperationError> {
    Ok(DynamicFusionMapSpace::from_typed(
        tensor
            .fusion_space()
            .ok_or(OperationError::Core(CoreError::MissingFusionSpace))?,
    ))
}

#[cfg(test)]
/// Rebuilds a typed tensor from a dynamic factor: the subblock structure and
/// hom space are shared as-is (identical layout by construction), only the
/// dense bookkeeping dims are recomputed as per-axis degeneracy totals.
pub(crate) fn typed_from_dyn<R, D, const NOUT: usize, const NIN: usize>(
    rule: &R,
    (space, data): DynFactor<D>,
) -> Result<TensorMap<D, NOUT, NIN>, OperationError>
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64>,
{
    if space.nout() != NOUT || space.nin() != NIN {
        return Err(OperationError::RankMismatch {
            expected: NOUT + NIN,
            actual: space.rank(),
        });
    }
    let axis_dim = |leg: &SectorLeg| {
        leg.degeneracies().iter().try_fold(0usize, |total, &dim| {
            total
                .checked_add(dim)
                .ok_or(CoreError::ElementCountOverflow)
        })
    };
    let mut codomain_dims = [0usize; NOUT];
    for (dim, leg) in codomain_dims
        .iter_mut()
        .zip(space.homspace().codomain().legs())
    {
        *dim = axis_dim(leg).map_err(OperationError::from_core_preserving_context)?;
    }
    let mut domain_dims = [0usize; NIN];
    for (dim, leg) in domain_dims.iter_mut().zip(space.homspace().domain().legs()) {
        *dim = axis_dim(leg).map_err(OperationError::from_core_preserving_context)?;
    }
    // Why not recover axis dimensions from populated fusion-tree blocks:
    // SectorLeg is the complete axis-space authority, including sectors with
    // no participating tree. External relabeling is irrelevant to a dimension
    // sum and can fail for a finite encoded dual after dense factorization.
    let typed_space = FusionTensorMapSpace::from_shared_subblock_structure(
        TensorMapSpace::<NOUT, NIN>::from_dims(codomain_dims, domain_dims)
            .map_err(OperationError::from_core_preserving_context)?,
        space.homspace().clone(),
        Arc::clone(space.structure()),
    )
    .map_err(OperationError::from_core_preserving_context)?
    .try_bind_rule(rule)
    .map_err(OperationError::from_core_preserving_context)?;
    TensorMap::from_vec_with_fusion_space(data, typed_space)
        .map_err(OperationError::from_core_preserving_context)
}

#[cfg(test)]
pub(crate) fn typed_from_bound_factor<R, D, const NOUT: usize, const NIN: usize>(
    factor: BoundDynFactor<R, D>,
) -> Result<BoundTensorMap<R, D, NOUT, NIN>, OperationError>
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64>,
{
    let (space, data) = factor.into_parts();
    let provider = Arc::clone(space.provider_arc());
    let tensor = typed_from_dyn(provider.as_ref(), (space.space().clone(), data))?;
    Ok(BoundTensorMap { space, tensor })
}
