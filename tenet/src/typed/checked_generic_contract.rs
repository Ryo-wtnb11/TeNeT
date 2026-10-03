use super::*;

impl<R, D> TypedTensorTraceDispatch<R, D> for MultiplicityFreeAdmissionMode
where
    R: TypedSectorAdmission<Error = FusionAlgebraError, Mode = MultiplicityFreeAdmissionMode>
        + MultiplicityFreeRigidSymbols<Scalar = f64>
        + CheckedFusionAlgebra
        + SectorCodec,
    D: TensorScalar,
{
    fn trace_pairs(
        tensor: &TensorMap<R, D>,
        pairs: &[(usize, usize)],
    ) -> Result<TensorMap<R, D>, Self::FacadeError> {
        tensor.trace_pairs_multiplicity_free(pairs)
    }
}

impl<R, D> TypedTensorTraceDispatch<R, D> for CheckedGenericAdmissionMode
where
    R: TypedSectorAdmission<
            Error = <R as CheckedGenericFusion>::Error,
            Mode = CheckedGenericAdmissionMode,
        > + CheckedGenericPivotal<Scalar = f64>,
    D: TensorScalar + tenet_tensors::RecouplingCoefficientAction<f64>,
{
    fn trace_pairs(
        tensor: &TensorMap<R, D>,
        pairs: &[(usize, usize)],
    ) -> Result<TensorMap<R, D>, Self::FacadeError> {
        trace_pairs_checked_generic(tensor, pairs)
    }
}

/// The axis lists a validated `trace_pairs` pair list stands for.
pub(super) struct TracePairAxes {
    pub(super) output_axes: Vec<usize>,
    pub(super) destination_codomain_rank: usize,
    pub(super) trace_lhs: Vec<usize>,
    pub(super) trace_rhs: Vec<usize>,
}

/// Validates a `trace_pairs` pair list and derives the untraced output axes in
/// order; `None` for an empty list, whose trace is a clone. Shared by every
/// storage so the Host and device reject a malformed list with one message.
///
/// Why this validation is kept rather than left to the seam: `seen` is not a
/// check, it is the derivation of `output_axes` — the seam cannot supply it,
/// and a malformed list would otherwise produce a silently wrong output order
/// rather than an error. Same precedent as `braid`'s levels pre-check.
pub(super) fn trace_pair_axes(
    rank: usize,
    codomain_rank: usize,
    pairs: &[(usize, usize)],
) -> Result<Option<TracePairAxes>, Error> {
    let mut seen = tenet_core::axes::AxisMask::new(rank);
    for &(lhs, rhs) in pairs {
        if seen.insert(lhs).and_then(|()| seen.insert(rhs)).is_err() {
            return Err(Error::InvalidArgument(format!(
                "invalid trace pair list {pairs:?} for rank {rank} \
                 (axes must be in range and distinct)"
            )));
        }
    }
    if pairs.is_empty() {
        return Ok(None);
    }
    let output_axes: Vec<usize> = seen.complement().collect();
    let destination_codomain_rank = output_axes
        .iter()
        .filter(|&&axis| axis < codomain_rank)
        .count();
    Ok(Some(TracePairAxes {
        output_axes,
        destination_codomain_rank,
        trace_lhs: pairs.iter().map(|&(lhs, _)| lhs).collect(),
        trace_rhs: pairs.iter().map(|&(_, rhs)| rhs).collect(),
    }))
}

pub(super) fn trace_pairs_checked_generic<R, D>(
    tensor: &TensorMap<R, D>,
    pairs: &[(usize, usize)],
) -> Result<TensorMap<R, D>, TypedFacadeError<R>>
where
    R: TypedSectorAdmission<
            Error = <R as CheckedGenericFusion>::Error,
            Mode = CheckedGenericAdmissionMode,
        > + CheckedGenericPivotal<Scalar = f64>,
    D: TensorScalar + tenet_tensors::RecouplingCoefficientAction<f64>,
{
    let Some(TracePairAxes {
        output_axes,
        destination_codomain_rank,
        trace_lhs,
        trace_rhs,
    }) = trace_pair_axes(tensor.rank(), tensor.codomain_rank(), pairs)?
    else {
        return Ok(tensor.clone());
    };
    let mapped_output_axes;
    let mapped_trace_lhs;
    let mapped_trace_rhs;
    let payload;
    let (source_space, source_data, axes) = match &tensor.repr {
        TypedTensorRepr::Owned(body) => {
            payload = body.materialized_dense_data();
            (
                &body.space,
                &*payload,
                tenet_tensors::TensorTraceAxisSpec::new(&output_axes, &trace_lhs, &trace_rhs),
            )
        }
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
                view.parent_data(),
                tenet_tensors::TensorTraceAxisSpec::new_with_conjugation(
                    &mapped_output_axes,
                    &mapped_trace_lhs,
                    &mapped_trace_rhs,
                    true,
                ),
            )
        }
    };
    let homspace = tenet_tensors::tensortrace_fusion_dyn_selected_homspace_generic_checked(
        source_space,
        axes,
        destination_codomain_rank,
    )?;
    let prepared = source_space
        .prepare_final_homspace_generic_with_checked(source_space.provider(), homspace)
        .map_err(CheckedGenericPlanError::from)?;
    let space = source_space
        .commit_final_homspace_generic_bound_checked(prepared)
        .map_err(CheckedGenericPlanError::Operation)?;
    let structure = <tenet_core::CheckedGenericAdmissionMode as tenet_tensors::PivotalCoefficientAlgebra<R>>::trace_terms(&space, source_space, axes)?;
    let data = tenet_tensors::tensortrace_fusion_dyn_structure_owned(
        &structure,
        space.space(),
        source_space.space(),
        source_data,
        D::from_real(1.0),
    )
    .map_err(CheckedGenericPlanError::Operation)?;
    Ok(TensorMap {
        runtime: tensor.runtime.clone(),
        repr: owned_repr(TypedTensorBody::dense(space, data)),
    })
}

impl<R, D> TensorMap<R, D>
where
    R: TypedSectorAdmission,
    R::Mode: TypedTensorTraceDispatch<R, D>,
    D: TensorScalar,
{
    /// Traces mutually dual axis pairs and returns the tensor on the remaining
    /// legs, preserving their order and codomain/domain side.
    ///
    /// Each pair uses flat zero-based axes with codomain axes first. Every axis
    /// must be in range and appear at most once; `pairs.is_empty()` returns a
    /// clone. This is the categorical contraction trace, so provider pivotal
    /// coefficients and fermionic twists are included. It can therefore be a
    /// supertrace and need not equal [`Self::tr`].
    ///
    /// Dense input runs the sectorwise trace engine. A multiplicity-free
    /// compact rank-`(1, 1)` factor traced over its only pair reduces directly
    /// in `O(sum_c k_c)`; other compact cases materialize. Checked Generic
    /// requires [`crate::sector::CheckedGenericPivotal`] and admits the output
    /// with the source's exact provider `Arc`. Lazy adjoints are read through
    /// their parent without materializing.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidArgument`] for an out-of-range or repeated axis.
    /// Non-dual legs and trace execution failures return their corresponding
    /// [`Error`] variant. If required pivotal data is unavailable, the original
    /// provider error is available as the source. A failed trace returns no
    /// output tensor.
    ///
    /// ```
    /// use std::sync::Arc;
    /// use tenet::sector::{U1FusionRule, U1Irrep};
    /// use tenet::typed::{GradedSpace, Runtime, TensorMap};
    ///
    /// let runtime = Runtime::builder().build()?;
    /// let v = GradedSpace::try_new(Arc::new(U1FusionRule), [(U1Irrep::new(0), 2)])?;
    /// let id: TensorMap<_, f64> = TensorMap::isomorphism(&runtime, [&v], [&v])?;
    /// assert_eq!(id.trace_pairs(&[(0, 1)])?.scalar()?, 2.0);
    /// # Ok::<(), tenet::typed::Error>(())
    /// ```
    pub fn trace_pairs(&self, pairs: &[(usize, usize)]) -> Result<Self, TypedFacadeError<R>> {
        <R::Mode as TypedTensorTraceDispatch<R, D>>::trace_pairs(self, pairs)
    }
}

pub(super) fn contract_multiplicity_free<R, D>(
    lhs: &TensorMap<R, D>,
    rhs: &TensorMap<R, D>,
    spec: &ContractSpec<'_>,
) -> Result<TensorMap<R, D>, Error>
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64> + CheckedFusionAlgebra + SectorCodec,
    D: TensorScalar,
{
    let (lhs_axes, rhs_axes) = (spec.lhs, spec.rhs);
    let output_axes = &spec.output_axes()[..];
    let codomain_rank = spec.codomain.len();
    if let Some(compact) =
        lhs.try_contract_diagonal(rhs, lhs_axes, rhs_axes, output_axes, codomain_rank)?
    {
        return Ok(compact);
    }
    let output_order = OutputAxisOrder::from_axes(output_axes);
    let destination = contract_destination(
        lhs,
        rhs,
        lhs_axes,
        rhs_axes,
        output_order,
        Some(codomain_rank),
    )?;
    let Some(copy_c) = CopyC::plan(lhs, rhs, spec, destination.space())? else {
        return contract_multiplicity_free_into(
            lhs,
            rhs,
            lhs_axes,
            rhs_axes,
            output_order,
            destination,
        );
    };
    let mut lease = lhs.runtime.lease_context()?;
    let lane = lease.context().multiplicity_free_lane::<D>()?;
    let temporary = copy_c.contract_temporary(lane)?;
    let data = tree_transform_owned_multiplicity_free_into(
        lane,
        BoundDynamicTensorRef::try_new(&copy_c.temporary_space, temporary.as_slice())?,
        copy_c.operation.clone(),
        &destination,
    )?;
    lane.restore_copy_c_scratch(temporary);
    Ok(TensorMap {
        runtime: lhs.runtime.clone(),
        repr: owned_repr(TypedTensorBody::dense(destination, data)),
    })
}

/// TensorKit `blas_contract!`'s `copyC` for `lhs·rhs` under `spec`: the
/// zero-copy candidate contraction with its own default output into a
/// temporary, then one permuting transform (`tensoradd!(C, Cnew, pAB, false,
/// α, β)`) into the result. Shared by the owned [`TensorMap::contract`] and
/// the destination [`TensorMap::contract_into`] routes, so both make the same
/// choice.
///
/// Why not the one-call ordered route when this applies: the predicate found
/// it costlier under TensorKit's memcost. It would copy a source (a lazy
/// adjoint always), while the zero-copy candidate borrows both operands and
/// only `C` moves.
pub(super) struct CopyC<'a, R, D> {
    first: &'a TensorMap<R, D>,
    second: &'a TensorMap<R, D>,
    first_axes: &'a [usize],
    second_axes: &'a [usize],
    pub(super) temporary_space: BoundDynamicFusionMapSpace<R>,
    temporary_len: usize,
    /// Permutes the temporary into the requested output order and split.
    pub(super) operation: TreeTransformOperation,
}

/// The only CopyC decision shared by owned stacks and ordinary tensors.
/// Axis pairing stays in the caller's original order after orientation.
pub(super) struct CopyCGeometry {
    orientation: tenet_tensors::FusionContractOrientation,
    pub(super) operation: TreeTransformOperation,
}

impl CopyCGeometry {
    pub(super) fn is_swapped(&self) -> bool {
        self.orientation == tenet_tensors::FusionContractOrientation::RhsLhs
    }
    pub(super) fn new(
        orientation: tenet_tensors::FusionContractOrientation,
        lhs_rank: usize,
        rhs_rank: usize,
        lhs_contract: &[usize],
        rhs_contract: &[usize],
        output_axes: &[usize],
        codomain_rank: usize,
    ) -> Self {
        let lhs_open = lhs_rank - lhs_contract.len();
        let rhs_open = rhs_rank - rhs_contract.len();
        let (lhs_offset, rhs_offset) = match orientation {
            tenet_tensors::FusionContractOrientation::LhsRhs => (0, 0),
            tenet_tensors::FusionContractOrientation::RhsLhs => (rhs_open, lhs_open),
        };
        let position = |axis: usize| {
            if axis < lhs_open {
                axis + lhs_offset
            } else {
                axis - rhs_offset
            }
        };
        let (codomain, domain) = output_axes.split_at(codomain_rank);
        Self {
            orientation,
            operation: TreeTransformOperation::permute(
                codomain.iter().copied().map(position),
                domain.iter().copied().map(position),
            ),
        }
    }

    pub(super) fn oriented<'a, T>(
        &self,
        lhs: &'a T,
        rhs: &'a T,
        lhs_axes: &'a [usize],
        rhs_axes: &'a [usize],
    ) -> (&'a T, &'a T, &'a [usize], &'a [usize]) {
        match self.orientation {
            tenet_tensors::FusionContractOrientation::LhsRhs => (lhs, rhs, lhs_axes, rhs_axes),
            tenet_tensors::FusionContractOrientation::RhsLhs => (rhs, lhs, rhs_axes, lhs_axes),
        }
    }
}

impl<'a, R, D> CopyC<'a, R, D>
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64> + CheckedFusionAlgebra + SectorCodec,
    D: TensorScalar,
{
    /// The copyC plan when TensorKit's memcost choice takes it for a result
    /// of space `destination`, else `None`. Metadata only: no lease, no
    /// scratch, no write.
    pub(super) fn plan(
        lhs: &'a TensorMap<R, D>,
        rhs: &'a TensorMap<R, D>,
        spec: &'a ContractSpec<'_>,
        destination: &DynamicFusionMapSpace,
    ) -> Result<Option<Self>, Error> {
        let (lhs_axes, rhs_axes) = (spec.lhs, spec.rhs);
        let output_axes = &spec.output_axes()[..];
        let Some(order) = tenet_tensors::zero_copy_contract_order_for_output_permute(
            lhs.logical_space().provider(),
            destination,
            lhs.fusion_operand(),
            rhs.fusion_operand(),
            lhs_axes,
            rhs_axes,
            output_axes,
        ) else {
            return Ok(None);
        };
        let geometry = CopyCGeometry::new(
            order,
            lhs.rank(),
            rhs.rank(),
            lhs_axes,
            rhs_axes,
            output_axes,
            spec.codomain.len(),
        );
        let (first, second, first_axes, second_axes) =
            geometry.oriented(lhs, rhs, lhs_axes, rhs_axes);
        let temporary_space = contract_destination(
            first,
            second,
            first_axes,
            second_axes,
            OutputAxisOrder::identity(),
            None,
        )?;
        let temporary_len = temporary_space.space().required_len()?;
        Ok(Some(Self {
            first,
            second,
            first_axes,
            second_axes,
            temporary_space,
            temporary_len,
            operation: geometry.operation,
        }))
    }

    /// Contracts into the lane's pooled `copyC` scratch and lends it out; the
    /// caller transforms it into its result and hands it back with
    /// `restore_copy_c_scratch`.
    ///
    /// TensorKit allocates `Cnew` as a temporary (`tensoralloc_add(...,
    /// Val(true), allocator)`) and frees it after the permute; the pooled
    /// scratch plays that role, so a warm call allocates no output-sized
    /// temporary. Stale pooled values are never read: `Axpby(0)` is a strong
    /// zero that assigns every block, inactive ones included.
    pub(super) fn contract_temporary(
        &self,
        lane: &mut crate::runtime::Ctx<D, tenet_core::RuleIdentity>,
    ) -> Result<tenet_operations::host_scratch::HostScratchBuffer<D>, Error> {
        let mut temporary = lane.take_copy_c_scratch();
        temporary.resize_filled(self.temporary_len, D::from_real(0.0));
        contract_multiplicity_free_into_slice(
            lane,
            self.first,
            self.second,
            self.first_axes,
            self.second_axes,
            OutputAxisOrder::identity(),
            &self.temporary_space,
            temporary.as_mut_slice(),
            tenet_tensors::ContractDestinationInit::Axpby(D::from_real(0.0)),
        )?;
        Ok(temporary)
    }
}

/// The destination space of `lhs·rhs` in `output_order`, split after
/// `codomain_rank` output axes (`None`: after every open lhs axis), derived as
/// the contraction route does: owned operands through the owned derivation, a
/// lazy adjoint through the oriented one.
pub(super) fn contract_destination<R, D>(
    lhs: &TensorMap<R, D>,
    rhs: &TensorMap<R, D>,
    lhs_axes: &[usize],
    rhs_axes: &[usize],
    output_order: OutputAxisOrder<'_>,
    codomain_rank: Option<usize>,
) -> Result<BoundDynamicFusionMapSpace<R>, Error>
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64> + CheckedFusionAlgebra + SectorCodec,
    D: TensorScalar,
{
    Ok(
        if let (TypedTensorRepr::Owned(lhs_body), TypedTensorRepr::Owned(rhs_body)) =
            (&lhs.repr, &rhs.repr)
        {
            match codomain_rank {
                Some(codomain_rank) => {
                    BoundDynamicFusionMapSpace::contracted_multiplicity_free_partitioned(
                        &lhs_body.space,
                        &rhs_body.space,
                        lhs_axes,
                        rhs_axes,
                        output_order,
                        codomain_rank,
                    )?
                }
                None => BoundDynamicFusionMapSpace::contracted_multiplicity_free_ordered(
                    &lhs_body.space,
                    &rhs_body.space,
                    lhs_axes,
                    rhs_axes,
                    output_order,
                )?,
            }
        } else {
            oriented_contract_destination(
                lhs.logical_space(),
                lhs.fusion_operand(),
                rhs.logical_space(),
                rhs.fusion_operand(),
                lhs_axes,
                rhs_axes,
                output_order,
                codomain_rank,
            )?
        },
    )
}

/// Contracts into `destination`, which [`contract_destination`] derived for
/// the same operands, axes and order.
pub(super) fn contract_multiplicity_free_into<R, D>(
    lhs: &TensorMap<R, D>,
    rhs: &TensorMap<R, D>,
    lhs_axes: &[usize],
    rhs_axes: &[usize],
    output_order: OutputAxisOrder<'_>,
    destination: BoundDynamicFusionMapSpace<R>,
) -> Result<TensorMap<R, D>, Error>
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64> + CheckedFusionAlgebra + SectorCodec,
    D: TensorScalar,
{
    let mut lease = lhs.runtime.lease_context()?;
    let mut data = tenet_tensors::zeroed_payload(destination.space().required_len()?);
    contract_multiplicity_free_into_slice(
        lease.context().multiplicity_free_lane::<D>()?,
        lhs,
        rhs,
        lhs_axes,
        rhs_axes,
        output_order,
        &destination,
        &mut data,
        tenet_tensors::ContractDestinationInit::Zeroed,
    )?;
    Ok(TensorMap {
        runtime: lhs.runtime.clone(),
        repr: owned_repr(TypedTensorBody::dense(destination, data)),
    })
}

/// [`contract_multiplicity_free_into`] writing the caller's `data`, initialized
/// per `init`, through the caller's lane.
#[allow(clippy::too_many_arguments)]
fn contract_multiplicity_free_into_slice<R, D>(
    context: &mut crate::runtime::Ctx<D, tenet_core::RuleIdentity>,
    lhs: &TensorMap<R, D>,
    rhs: &TensorMap<R, D>,
    lhs_axes: &[usize],
    rhs_axes: &[usize],
    output_order: OutputAxisOrder<'_>,
    destination: &BoundDynamicFusionMapSpace<R>,
    data: &mut [D],
    init: tenet_tensors::ContractDestinationInit<D>,
) -> Result<(), Error>
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64> + CheckedFusionAlgebra + SectorCodec,
    D: TensorScalar,
{
    if let (TypedTensorRepr::Owned(lhs_body), TypedTensorRepr::Owned(rhs_body)) =
        (&lhs.repr, &rhs.repr)
    {
        tensorcontract_owned_multiplicity_free_into_slice(
            context,
            destination,
            data,
            BoundDynamicTensorRef::try_new(
                &lhs_body.space,
                lhs_body.materialized_dense_data().as_ref(),
            )?,
            BoundDynamicTensorRef::try_new(
                &rhs_body.space,
                rhs_body.materialized_dense_data().as_ref(),
            )?,
            lhs_axes,
            rhs_axes,
            output_order,
            init,
        )?;
    } else {
        let (lhs_operand, lhs_data) = lhs.fusion_operand_and_data();
        let (rhs_operand, rhs_data) = rhs.fusion_operand_and_data();
        tensorcontract_oriented_multiplicity_free_into_slice(
            context,
            destination,
            data,
            lhs_operand,
            &lhs_data,
            rhs_operand,
            &rhs_data,
            lhs_axes,
            rhs_axes,
            output_order,
            OrientedContractionKind::Contract,
            init,
        )?;
    }
    Ok(())
}

pub(super) fn compose_multiplicity_free<R, D>(
    lhs: &TensorMap<R, D>,
    rhs: &TensorMap<R, D>,
) -> Result<TensorMap<R, D>, Error>
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64> + CheckedFusionAlgebra + SectorCodec,
    D: TensorScalar,
{
    if let Some(compact) = lhs.compose_compact(rhs)? {
        return Ok(compact);
    }
    compose_multiplicity_free_with_lane(lhs, rhs)
}

#[allow(private_bounds)]
pub(super) fn compose_multiplicity_free_with_lane<R, D>(
    lhs: &TensorMap<R, D>,
    rhs: &TensorMap<R, D>,
) -> Result<TensorMap<R, D>, Error>
where
    R: MultiplicityFreeRigidSymbols + CheckedFusionAlgebra + SectorCodec,
    R::Scalar: CategoricalScalar + tenet_tensors::DenseRecouplingScalar,
    D: TensorScalar + crate::runtime::MultiplicityFreeCoefficientLane<R::Scalar>,
{
    let lhs_axes = (lhs.codomain_rank()..lhs.rank()).collect::<Vec<_>>();
    let rhs_axes = (0..rhs.codomain_rank()).collect::<Vec<_>>();
    let mut lease = lhs.runtime.lease_context()?;
    let (space, data) =
        if let (TypedTensorRepr::Owned(lhs_body), TypedTensorRepr::Owned(rhs_body)) =
            (&lhs.repr, &rhs.repr)
        {
            tensorcompose_owned_multiplicity_free(
                D::lane(lease.context())?,
                BoundDynamicTensorRef::try_new(
                    &lhs_body.space,
                    lhs_body.materialized_dense_data().as_ref(),
                )?,
                BoundDynamicTensorRef::try_new(
                    &rhs_body.space,
                    rhs_body.materialized_dense_data().as_ref(),
                )?,
                &lhs_axes,
                &rhs_axes,
            )?
        } else {
            let (lhs_operand, lhs_data) = lhs.fusion_operand_and_data();
            let (rhs_operand, rhs_data) = rhs.fusion_operand_and_data();
            tensorcontract_oriented_multiplicity_free(
                D::lane(lease.context())?,
                lhs.logical_space(),
                lhs_operand,
                &lhs_data,
                rhs.logical_space(),
                rhs_operand,
                &rhs_data,
                &lhs_axes,
                &rhs_axes,
                OutputAxisOrder::identity(),
                OrientedContractionKind::Compose,
            )?
        };
    Ok(TensorMap {
        runtime: lhs.runtime.clone(),
        repr: owned_repr(TypedTensorBody::dense(space, data)),
    })
}

impl<R, D> MultiplicityFreeContractExecution<R, f64> for D
where
    R: TypedSectorAdmission
        + MultiplicityFreeRigidSymbols<Scalar = f64>
        + CheckedFusionAlgebra
        + SectorCodec,
    D: TensorScalar,
{
    fn contract(
        lhs: &TensorMap<R, Self>,
        rhs: &TensorMap<R, Self>,
        spec: &ContractSpec<'_>,
    ) -> Result<TensorMap<R, Self>, Error> {
        contract_multiplicity_free(lhs, rhs, spec)
    }

    fn compose(
        lhs: &TensorMap<R, Self>,
        rhs: &TensorMap<R, Self>,
    ) -> Result<TensorMap<R, Self>, Error> {
        compose_multiplicity_free(lhs, rhs)
    }
}

impl<R> MultiplicityFreeContractExecution<R, Complex64> for Complex64
where
    R: TypedSectorAdmission
        + MultiplicityFreeRigidSymbols<Scalar = Complex64>
        + CheckedFusionAlgebra
        + SectorCodec,
{
    fn contract(
        _lhs: &TensorMap<R, Self>,
        _rhs: &TensorMap<R, Self>,
        _spec: &ContractSpec<'_>,
    ) -> Result<TensorMap<R, Self>, Error> {
        Err(
            tenet_tensors::OperationError::UnsupportedTensorContractScope {
                message: "ordinary complex-coefficient contraction is outside the admitted scope",
            }
            .into(),
        )
    }

    fn compose(
        lhs: &TensorMap<R, Self>,
        rhs: &TensorMap<R, Self>,
    ) -> Result<TensorMap<R, Self>, Error> {
        compose_multiplicity_free_with_lane(lhs, rhs)
    }
}

pub(crate) type TypedFacadeError<R> =
    <<R as TypedSectorAdmission>::Mode as TypedTensorModeDispatch<R>>::FacadeError;

pub(super) fn write_identity_blocks_generic<R, D>(tensor: &mut TensorMap<R, D>) -> Result<(), Error>
where
    D: TensorScalar,
{
    let TypedTensorRepr::Owned(body) = &mut tensor.repr else {
        unreachable!()
    };
    write_dense_identity_blocks(Arc::get_mut(body).expect("fresh identity body"))
}

pub(super) fn write_dense_identity_blocks<R, D>(
    body: &mut TypedTensorBody<R, D>,
) -> Result<(), Error>
where
    D: TensorScalar,
{
    let regions = {
        let space = body.space.space();
        sector_regions(space.structure(), space.nout())?
    };
    let payload = Arc::get_mut(&mut body.data).expect("fresh identity payload");
    let TypedData::Dense(data) = payload else {
        unreachable!()
    };
    for region in regions.iter() {
        for i in 0..region.rows().min(region.cols()) {
            data[region.range().start + i * (region.rows() + 1)] = D::from_real(1.0);
        }
    }
    Ok(())
}

#[cfg(test)]
mod copy_c_scratch_tests {
    use std::sync::Arc;

    use tenet_core::{U1FusionRule, U1Irrep};

    use crate::typed::{ContractSpec, GradedSpace, Runtime, TensorMap, TypedData, TypedTensorRepr};

    /// Grows the pooled copyC scratch to a length no temporary here reaches;
    /// a shorter length afterwards proves the copyC route ran.
    const MARKER_LEN: usize = 100_000;

    fn mark_scratch(runtime: &Runtime) {
        let mut lease = runtime.lease_context().unwrap();
        let lane = lease.context().multiplicity_free_lane::<f64>().unwrap();
        let mut scratch = lane.take_copy_c_scratch();
        scratch.resize_filled(MARKER_LEN, 0.0);
        lane.restore_copy_c_scratch(scratch);
    }

    fn retained_len(runtime: &Runtime) -> usize {
        let mut lease = runtime.lease_context().unwrap();
        lease
            .context()
            .multiplicity_free_lane::<f64>()
            .unwrap()
            .copy_c_scratch_len()
    }

    /// The copyC temporary lives in the leased context and is sized by the
    /// latest temporary, so a Runtime retains at most one largest temporary
    /// per idle context — the same bound as the context's other scratch.
    #[test]
    fn copy_c_scratch_holds_the_latest_temporary_only() {
        let runtime = Runtime::builder().dense_threads(1).build().unwrap();
        let space =
            |n| GradedSpace::try_new(Arc::new(U1FusionRule), [(U1Irrep::new(0), n)]).unwrap();
        let spec = ContractSpec {
            lhs: &[2],
            rhs: &[0],
            codomain: &[0, 2],
            domain: &[1],
        };
        let run = |n: usize| {
            let (p, c) = (space(n), space(3));
            let a = TensorMap::<U1FusionRule, f64>::rand_with_seed(&runtime, [&p, &p], [&c], 1)
                .unwrap();
            let b = TensorMap::rand_with_seed(&runtime, [&c], [&p], 2).unwrap();
            drop(a.contract(&b, &spec).unwrap());
            retained_len(&runtime)
        };
        assert_eq!(retained_len(&runtime), 0);
        assert_eq!(run(5), 125);
        assert_eq!(run(7), 343);
        assert_eq!(run(5), 125);
    }

    /// The pooled temporary is written with a strong-zero `Axpby(0)`, so a
    /// stale buffer never leaks: a NaN-poisoned buffer longer than the
    /// temporary (so resizing writes nothing) still gives exactly the
    /// contract-then-permute result, including the temporary's inactive block.
    #[test]
    fn stale_copy_c_scratch_never_reaches_the_result() {
        let runtime = Runtime::builder().dense_threads(1).build().unwrap();
        let provider = Arc::new(U1FusionRule);
        // `c` carries charge 0 only, so the temporary `p ⊗ q ← r` block of
        // coupled charge 1 receives no GEMM contribution.
        let open = GradedSpace::try_new(
            Arc::clone(&provider),
            [(U1Irrep::new(0), 2), (U1Irrep::new(1), 3)],
        )
        .unwrap();
        let bond = GradedSpace::try_new(provider, [(U1Irrep::new(0), 4)]).unwrap();
        let a =
            TensorMap::<U1FusionRule, f64>::rand_with_seed(&runtime, [&open, &open], [&bond], 3)
                .unwrap();
        let b = TensorMap::rand_with_seed(&runtime, [&bond], [&open], 4).unwrap();
        let spec = ContractSpec {
            lhs: &[2],
            rhs: &[0],
            codomain: &[0, 2],
            domain: &[1],
        };
        let expected = a
            .contract(
                &b,
                &ContractSpec {
                    lhs: &[2],
                    rhs: &[0],
                    codomain: &[0, 1],
                    domain: &[2],
                },
            )
            .unwrap()
            .permute(&[0, 2], &[1])
            .unwrap();
        const POISONED_LEN: usize = 100_000;
        {
            let mut lease = runtime.lease_context().unwrap();
            let lane = lease.context().multiplicity_free_lane::<f64>().unwrap();
            let mut scratch = lane.take_copy_c_scratch();
            scratch.resize_filled(POISONED_LEN, f64::NAN);
            lane.restore_copy_c_scratch(scratch);
        }
        let actual = a.contract(&b, &spec).unwrap();
        // What: the copyC route ran on the poisoned buffer.
        assert!(retained_len(&runtime) < POISONED_LEN);
        assert_eq!(actual.codomain(), expected.codomain());
        assert_eq!(actual.domain(), expected.domain());
        assert_eq!(actual.dense_data().unwrap(), expected.dense_data().unwrap());
    }

    /// The partial-coverage fixture of `stale_copy_c_scratch_never_reaches_the_result`:
    /// `c` carries charge 0 only, so the result's coupled-charge-1 block of
    /// `(p, r | q)` is reached by no GEMM.
    fn partial_copy_c_fixture(
        runtime: &Runtime,
    ) -> (TensorMap<U1FusionRule, f64>, TensorMap<U1FusionRule, f64>) {
        let provider = Arc::new(U1FusionRule);
        let open = GradedSpace::try_new(
            Arc::clone(&provider),
            [(U1Irrep::new(0), 2), (U1Irrep::new(1), 3)],
        )
        .unwrap();
        let bond = GradedSpace::try_new(provider, [(U1Irrep::new(0), 4)]).unwrap();
        (
            TensorMap::rand_with_seed(runtime, [&open, &open], [&bond], 3).unwrap(),
            TensorMap::rand_with_seed(runtime, [&bond], [&open], 4).unwrap(),
        )
    }

    const MOVED: ContractSpec<'static> = ContractSpec {
        lhs: &[2],
        rhs: &[0],
        codomain: &[0, 2],
        domain: &[1],
    };

    /// #1631: on the copyC route the output transform writes every element,
    /// so an unreached one becomes `alpha * (+0) + beta * dst` — a `-0.0`
    /// becomes `+0.0` under `beta = 1` and an infinite `alpha` gives NaN, as
    /// TensorKit's `tensoradd!` — while `alpha = 0` still never reads the
    /// operands.
    #[test]
    fn copy_c_contract_into_writes_unreached_elements_through_the_transform() {
        let runtime = Runtime::builder().dense_threads(1).build().unwrap();
        let (a, b) = partial_copy_c_fixture(&runtime);
        let result = a.contract(&b, &MOVED).unwrap();
        let unreached: Vec<usize> = result
            .dense_data()
            .unwrap()
            .iter()
            .enumerate()
            .filter(|(_, value)| **value == 0.0)
            .map(|(index, _)| index)
            .collect();
        assert!(!unreached.is_empty(), "the fixture must miss a sector");
        let run =
            |seed: f64, alpha: f64, beta: f64, operands: (&TensorMap<_, _>, &TensorMap<_, _>)| {
                let mut destination = result.scale(0.0);
                let TypedTensorRepr::Owned(body) = &mut destination.repr else {
                    unreachable!("a scaled dense tensor is owned")
                };
                let Some(TypedData::Dense(data)) =
                    Arc::get_mut(body).and_then(|body| Arc::get_mut(&mut body.data))
                else {
                    unreachable!("a fresh scaled tensor is uniquely owned and dense")
                };
                data.fill(seed);
                mark_scratch(&runtime);
                operands
                    .0
                    .contract_into(operands.1, &MOVED, &mut destination, alpha, beta)
                    .unwrap();
                assert!(
                    retained_len(&runtime) < MARKER_LEN,
                    "the copyC route must run"
                );
                destination
            };
        let negative_zero = run(-0.0, 1.0, 1.0, (&a, &b));
        for &index in &unreached {
            let value = negative_zero.dense_data().unwrap()[index];
            assert_eq!(value.to_bits(), 0.0f64.to_bits(), "-0.0 at {index}");
        }
        let infinite = run(1.0, f64::INFINITY, 1.0, (&a, &b));
        for &index in &unreached {
            assert!(infinite.dense_data().unwrap()[index].is_nan(), "{index}");
        }
        let (nan_a, nan_b) = (a.scale(f64::NAN), b.scale(f64::NAN));
        let silent = run(0.5, 0.0, 2.0, (&nan_a, &nan_b));
        assert!(silent
            .dense_data()
            .unwrap()
            .iter()
            .all(|&value| value == 1.0));
    }

    /// #1631: a compact diagonal operand on the copyC route is densified
    /// once, for the temporary's contraction; the length check reads the
    /// stored representation.
    #[test]
    fn copy_c_contract_into_densifies_a_compact_operand_once() {
        let runtime = Runtime::builder().dense_threads(1).build().unwrap();
        let provider = Arc::new(U1FusionRule);
        let space =
            |n| GradedSpace::try_new(Arc::clone(&provider), [(U1Irrep::new(0), n)]).unwrap();
        let (p, q, c) = (space(2), space(3), space(4));
        let t =
            TensorMap::<U1FusionRule, f64>::rand_with_seed(&runtime, [&p, &q], [&c], 5).unwrap();
        let spectra = c
            .sectors()
            .unwrap()
            .into_iter()
            .map(|sector| crate::typed::SectorSpectrum {
                values: vec![1.5; c.degeneracy(&sector).unwrap()],
                sector,
            })
            .collect::<Vec<_>>();
        let d = TensorMap::diagonal(&runtime, &c, spectra).unwrap();
        let mut destination = t.contract(&d, &MOVED).unwrap().scale(1.0);
        mark_scratch(&runtime);
        crate::typed::DIAGONAL_MATERIALIZATIONS.set(0);
        t.contract_into(&d, &MOVED, &mut destination, 1.0, 0.0)
            .unwrap();
        assert_eq!(crate::typed::DIAGONAL_MATERIALIZATIONS.get(), 1);
        assert!(
            retained_len(&runtime) < MARKER_LEN,
            "the copyC route must run"
        );
        let expected = t.contract(&d.materialize().unwrap(), &MOVED).unwrap();
        assert_eq!(
            destination.dense_data().unwrap(),
            expected.dense_data().unwrap()
        );
    }
}
