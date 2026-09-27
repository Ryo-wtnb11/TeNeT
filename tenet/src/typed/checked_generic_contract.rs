#[allow(unused_imports)]
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

/// The error message of [`reject_non_symmetric_contraction`], shared with the
/// device network preflight so a network rejection and a typed `contract`
/// rejection are indistinguishable.
#[doc(hidden)]
pub const NON_SYMMETRIC_CONTRACTION_UNSUPPORTED: &str =
    "ordinary contraction requires symmetric (bosonic or fermionic) braiding; \
     use compose for the composition of morphisms (planar contraction is TeNeT#1070)";

/// The ordinary-contraction boundary of every `contract` entry, returning or
/// overwriting, Host or device, typed or `tensor!`: only a symmetric braiding
/// (TensorKit `SymmetricBraiding`: Bosonic, Fermionic) is admitted, as by
/// TensorKit `blas_contract!` before any layout test.
///
/// Why not admit the canonical axes of a `NoBraiding` or `Anyonic` rule:
/// general-axes contraction is defined by braiding legs into place, which is
/// well defined for arbitrary permutations only under a symmetric braiding.
/// The one axis pattern that needs no braid is exactly [`TensorMap::compose`]
/// (TensorKit `mul!`), which stays admitted for every braiding style; planar
/// contraction for non-symmetric categories is a separate named operation
/// (#1070).
#[doc(hidden)]
pub fn reject_non_symmetric_contraction(
    braiding: tenet_core::BraidingStyleKind,
) -> Result<(), tenet_tensors::OperationError> {
    if !braiding.is_symmetric() {
        return Err(
            tenet_tensors::OperationError::UnsupportedTensorContractScope {
                message: NON_SYMMETRIC_CONTRACTION_UNSUPPORTED,
            },
        );
    }
    Ok(())
}

/// The axis lists a validated `trace_pairs` pair list stands for.
pub(super) struct TracePairAxes {
    output_axes: Vec<usize>,
    destination_codomain_rank: usize,
    trace_lhs: Vec<usize>,
    trace_rhs: Vec<usize>,
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
    let mut seen = vec![false; rank];
    for &(lhs, rhs) in pairs {
        for axis in [lhs, rhs] {
            if axis >= rank || seen[axis] {
                return Err(Error::InvalidArgument(format!(
                    "invalid trace pair list {pairs:?} for rank {rank} \
                     (axes must be in range and distinct)"
                )));
            }
            seen[axis] = true;
        }
    }
    if pairs.is_empty() {
        return Ok(None);
    }
    let output_axes: Vec<usize> = (0..rank).filter(|&axis| !seen[axis]).collect();
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
    let (source_space, source_data, axes) = match &tensor.repr {
        TypedTensorRepr::Owned(body) => (
            &body.space,
            body.materialized_dense_data(),
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
                view.parent.materialized_dense_data(),
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
    let data = tenet_tensors::tensortrace_fusion_dyn_owned_generic_checked(
        &space,
        source_space,
        source_data,
        axes,
        D::from_real(1.0),
    )?;
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
    /// requires [`tenet_core::CheckedGenericPivotal`] and admits the output
    /// with the source's exact provider `Arc`. Lazy adjoints are read through
    /// their parent without filling the receiver cache.
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
    /// use tenet::core::{U1FusionRule, U1Irrep};
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
    lhs_axes: &[usize],
    rhs_axes: &[usize],
    output_axes: &[usize],
) -> Result<TensorMap<R, D>, Error>
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64> + CheckedFusionAlgebra + SectorCodec,
    D: TensorScalar,
{
    if let Some(compact) = lhs.try_contract_diagonal(rhs, lhs_axes, rhs_axes, output_axes)? {
        return Ok(compact);
    }
    let output_order = OutputAxisOrder::from_axes(output_axes);
    let destination = contract_destination(lhs, rhs, lhs_axes, rhs_axes, output_order)?;
    let Some(order) = tenet_tensors::zero_copy_contract_order_for_output_permute(
        lhs.logical_space().provider(),
        destination.space(),
        lhs.fusion_operand(),
        rhs.fusion_operand(),
        lhs_axes,
        rhs_axes,
        output_axes,
    ) else {
        return contract_multiplicity_free_into(
            lhs,
            rhs,
            lhs_axes,
            rhs_axes,
            output_order,
            destination,
        );
    };
    // Why not the one-call ordered route here: the predicate found it
    // costlier under TensorKit's memcost. It would copy a source (a lazy
    // adjoint always), while the zero-copy candidate borrows both operands
    // and only `C` moves. This is TensorKit `blas_contract!`'s `copyC`: a
    // temporary, then a permuting `tensoradd!` into `destination`.
    let lhs_open = lhs.rank() - lhs_axes.len();
    let rhs_open = rhs.rank() - rhs_axes.len();
    let identity = OutputAxisOrder::identity();
    let (temporary, lhs_offset, rhs_offset) = match order {
        tenet_tensors::FusionContractOrientation::LhsRhs => (
            contract_multiplicity_free_ordered(lhs, rhs, lhs_axes, rhs_axes, identity)?,
            0,
            0,
        ),
        tenet_tensors::FusionContractOrientation::RhsLhs => (
            contract_multiplicity_free_ordered(rhs, lhs, rhs_axes, lhs_axes, identity)?,
            rhs_open,
            lhs_open,
        ),
    };
    // `temporary` lists the rhs open axes first under `RhsLhs`.
    let position = |axis: usize| {
        if axis < lhs_open {
            axis + lhs_offset
        } else {
            axis - rhs_offset
        }
    };
    let (codomain, domain) = output_axes.split_at(lhs_open);
    let body = temporary
        .owned_body()
        .expect("contraction results are owned");
    let mut lease = lhs.runtime.lease_context()?;
    let data = tree_transform_owned_multiplicity_free_into(
        lease.context().multiplicity_free_lane::<D>()?,
        BoundDynamicTensorRef::try_new(&body.space, body.materialized_dense_data())?,
        TreeTransformOperation::permute(
            codomain.iter().copied().map(position),
            domain.iter().copied().map(position),
        ),
        &destination,
    )?;
    Ok(TensorMap {
        runtime: lhs.runtime.clone(),
        repr: owned_repr(TypedTensorBody::dense(destination, data)),
    })
}

/// The destination space of `lhs·rhs` in `output_order`, derived as the
/// contraction route does: owned operands through the owned derivation, a
/// lazy adjoint through the oriented one.
pub(super) fn contract_destination<R, D>(
    lhs: &TensorMap<R, D>,
    rhs: &TensorMap<R, D>,
    lhs_axes: &[usize],
    rhs_axes: &[usize],
    output_order: OutputAxisOrder<'_>,
) -> Result<BoundDynamicFusionMapSpace<R>, Error>
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64> + CheckedFusionAlgebra + SectorCodec,
    D: TensorScalar,
{
    Ok(
        if let (TypedTensorRepr::Owned(lhs_body), TypedTensorRepr::Owned(rhs_body)) =
            (&lhs.repr, &rhs.repr)
        {
            BoundDynamicFusionMapSpace::contracted_multiplicity_free_ordered(
                &lhs_body.space,
                &rhs_body.space,
                lhs_axes,
                rhs_axes,
                output_order,
            )?
        } else {
            oriented_contract_destination(
                lhs.logical_space(),
                lhs.fusion_operand(),
                rhs.logical_space(),
                rhs.fusion_operand(),
                lhs_axes,
                rhs_axes,
                output_order,
            )?
        },
    )
}

pub(super) fn contract_multiplicity_free_ordered<R, D>(
    lhs: &TensorMap<R, D>,
    rhs: &TensorMap<R, D>,
    lhs_axes: &[usize],
    rhs_axes: &[usize],
    output_order: OutputAxisOrder<'_>,
) -> Result<TensorMap<R, D>, Error>
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64> + CheckedFusionAlgebra + SectorCodec,
    D: TensorScalar,
{
    let destination = contract_destination(lhs, rhs, lhs_axes, rhs_axes, output_order)?;
    contract_multiplicity_free_into(lhs, rhs, lhs_axes, rhs_axes, output_order, destination)
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
    let data = if let (TypedTensorRepr::Owned(lhs_body), TypedTensorRepr::Owned(rhs_body)) =
        (&lhs.repr, &rhs.repr)
    {
        tensorcontract_owned_multiplicity_free_into(
            lease.context().multiplicity_free_lane::<D>()?,
            &destination,
            BoundDynamicTensorRef::try_new(&lhs_body.space, lhs_body.materialized_dense_data())?,
            BoundDynamicTensorRef::try_new(&rhs_body.space, rhs_body.materialized_dense_data())?,
            lhs_axes,
            rhs_axes,
            output_order,
        )?
    } else {
        let (lhs_operand, lhs_data) = lhs.fusion_operand_and_data();
        let (rhs_operand, rhs_data) = rhs.fusion_operand_and_data();
        tensorcontract_oriented_multiplicity_free_into(
            lease.context().multiplicity_free_lane::<D>()?,
            &destination,
            lhs_operand,
            lhs_data,
            rhs_operand,
            rhs_data,
            lhs_axes,
            rhs_axes,
            output_order,
            OrientedContractionKind::Contract,
        )?
    };
    Ok(TensorMap {
        runtime: lhs.runtime.clone(),
        repr: owned_repr(TypedTensorBody::dense(destination, data)),
    })
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
                    lhs_body.materialized_dense_data(),
                )?,
                BoundDynamicTensorRef::try_new(
                    &rhs_body.space,
                    rhs_body.materialized_dense_data(),
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
                lhs_data,
                rhs.logical_space(),
                rhs_operand,
                rhs_data,
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
        lhs_axes: &[usize],
        rhs_axes: &[usize],
        output_axes: &[usize],
    ) -> Result<TensorMap<R, Self>, Error> {
        contract_multiplicity_free(lhs, rhs, lhs_axes, rhs_axes, output_axes)
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
        _lhs_axes: &[usize],
        _rhs_axes: &[usize],
        _output_axes: &[usize],
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

pub(super) fn write_dense_identity_blocks<R, D>(body: &mut TypedTensorBody<R, D>) -> Result<(), Error>
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

impl<R, D> TensorMap<R, D>
where
    R: TypedSectorAdmission,
    R::Mode: TypedTensorInvDispatch<R, D>,
    D: AdvancedLinalgScalar,
{
    /// TensorKit 0.17 / MatrixAlgebraKit `inv`: the true inverse `t^-1` of a
    /// nonsingular map, defined by `t * t^-1 = id` on the codomain and
    /// `t^-1 * t = id` on the domain. Computed per coupled sector as the exact
    /// dense solve `t_c X_c = 1`, not as a spectral function — there is no
    /// truncation policy to apply and no factor tensor to build.
    ///
    /// # Domain
    ///
    /// TensorKit asks for `codomain ≅ domain` — **isomorphic, not equal** —
    /// and returns a map `domain <- codomain`. This facade's seam agrees: a
    /// rank-one codomain and a rank-two domain with the same coupled-sector
    /// dimensions are accepted, and the result carries the two spaces swapped.
    /// The pin is `inv_accepts_isomorphic_but_unequal_codomain_and_domain`.
    ///
    /// # Errors
    ///
    /// - [`Error::Operation`] when the two sides are not isomorphic, and when a
    ///   coupled-sector block is singular — the dense solve is where that
    ///   surfaces, so it comes back as an execution error rather than an
    ///   argument one. Never a panic.
    /// - [`Error::InvalidArgument`] from the compact arm below, whose zero
    ///   entry is visible before any solve runs and is therefore reported as
    ///   the caller mistake it is. The two storages of one singular tensor
    ///   consequently report different variants; both are pinned by
    ///   `inv_reports_a_singular_input_as_a_typed_error`.
    ///
    /// # Complexity
    ///
    /// Dense input: `O(Σ_c n_c³)`, one LU solve per coupled sector. Compact
    /// multiplicity-free compact input (a spectrum factor, TensorKit's
    /// `DiagonalTensorMap`): the
    /// **O(rank) elementwise-reciprocal arm**, `1/s_i` over the `Σ_c k_c`
    /// stored values, and the result stays compact — matching TensorKit's
    /// `inv(::DiagonalTensorMap)`, which is `inv.(d.data)`. Nothing dense is
    /// built on either side of that arm. A host lazy adjoint solves its owned
    /// parent and returns a detached owned adjoint of that inverse; it does not
    /// allocate or publish a separate receiver-materialization payload.
    ///
    /// Checked Generic performs the same isomorphism preflight, admits the
    /// swapped output with the source provider `Arc` before output allocation
    /// or dense work, and preserves provider admission failures as typed errors.
    /// Standalone compact construction is supported, but checked `inv` has no
    /// elementwise compact arm: it materializes that input and publishes a dense
    /// result.
    pub fn inv(&self) -> Result<Self, TypedFacadeError<R>> {
        <R::Mode as TypedTensorInvDispatch<R, D>>::inv(self)
    }
}

impl<R, D> TensorMap<R, D>
where
    R: TypedSectorAdmission,
    R::Mode: TypedTensorExpDispatch<R, D>,
    D: AdvancedLinalgScalar,
{
    /// The matrix exponential `exp(t) = Σ_k t^k / k!`, evaluated per coupled
    /// sector — TensorKit's `exp`, which copies and calls `exp!`: check
    /// `domain == codomain`, then exponentiate every block.
    ///
    /// # Domain
    ///
    /// Any endomorphism, of any dtype. Multiplicity-free tensors retain the
    /// original two dense routes:
    ///
    /// - **Hermitian blocks** take the spectral function `V exp(D) Vᴴ` of the
    ///   Hermitian eigendecomposition.
    /// - **Everything else** takes blockwise scaling-and-squaring Padé [13/13]
    ///   (Higham 2005). Non-normal, defective and complex non-Hermitian blocks
    ///   are all in domain; nothing is symmetrized.
    ///
    /// The mathematical and blockwise contract matches the referenced
    /// implementation, but bitwise output does not. TeNeT's general dense route
    /// always uses Padé [13/13], while the referenced implementation delegates
    /// dense algorithm selection and may choose a lower degree for a small-norm
    /// block. The results agree only up to their numerical approximation; pinned
    /// source coordinates are recorded in `tenet/references.md`.
    ///
    /// The multiplicity-free **compact** arm is TensorKit's
    /// `exp(::DiagonalTensorMap)`: unconditionally elementwise, with no
    /// hermiticity gate. Standalone checked-Generic compact construction is
    /// supported, but checked `exp` has no compact dispatch: it materializes
    /// that input and publishes the dense Padé result.
    ///
    /// # Errors
    ///
    /// - [`Error::Operation`] when the input is not an endomorphism
    ///   (`codomain != domain`), when a general block holds a nonfinite entry,
    ///   when a general block's column 1-norm overflows to infinity although
    ///   every entry is finite, or when the backend fails. Nothing is published
    ///   unless every coupled sector succeeded.
    /// - [`Error::Core`] / [`Error::FusionAlgebra`] from the multiplicity-free
    ///   Hermitian composition route.
    ///
    /// # Complexity
    ///
    /// Dense input is `O(Σ_c n_c³)`: multiplicity-free Hermitian blocks use one
    /// eigendecomposition plus a composition; all general blocks, including
    /// checked-Generic, use six GEMMs, one solve and the necessary Padé
    /// squarings per sector with `O(max_c n_c²)` workspace. Coupled sectors are
    /// never mixed. A dense lazy adjoint builds one operation-local logical
    /// payload per call without publishing its receiver cache. Compact
    /// multiplicity-free input remains `O(rank)` elementwise.
    ///
    /// TensorKit's diagonal implementation is the reference for the compact
    /// branch; this method never panics for a supported tensor contract.
    ///
    /// ```
    /// use std::sync::Arc;
    /// use tenet::core::{U1FusionRule, U1Irrep};
    /// use tenet::typed::{GradedSpace, Runtime, TensorMap};
    /// let runtime = Runtime::builder().build()?;
    /// let v = GradedSpace::try_new(Arc::new(U1FusionRule), [(U1Irrep::new(0), 2), (U1Irrep::new(1), 1)])?;
    /// let zero: TensorMap<_, f64> = TensorMap::zeros(&runtime, [&v], [&v])?;
    /// let id: TensorMap<_, f64> = TensorMap::isomorphism(&runtime, [&v], [&v])?;
    /// assert!(zero.exp()?.dense_data()?.iter().zip(id.dense_data()?).all(|(a, b)| (a - b).abs() < 1e-15));
    /// # Ok::<(), tenet::typed::Error>(())
    /// ```
    pub fn exp(&self) -> Result<Self, TypedFacadeError<R>> {
        <R::Mode as TypedTensorExpDispatch<R, D>>::exp(self)
    }
}
