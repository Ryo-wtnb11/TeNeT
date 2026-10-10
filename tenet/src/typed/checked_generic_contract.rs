use super::*;

impl<R, D> TypedTensorTraceDispatch<R, D> for MultiplicityFreeAdmissionMode
where
    R: TypedSectorAdmission<Error = FusionAlgebraError, Mode = MultiplicityFreeAdmissionMode>
        + MultiplicityFreeRigidSymbols<Scalar = f64>
        + CheckedFusionAlgebra
        + SectorCodec,
    D: TensorScalar,
{
    fn try_compact_trace(
        tensor: &TensorMap<R, D>,
        space: &BoundDynamicFusionMapSpace<R>,
        axes: tenet_tensors::TensorTraceAxisSpec<'_>,
        dst_nout: usize,
    ) -> Result<Option<TensorMap<R, D>>, Error> {
        compact_arms::full_trace_spectrum(tensor, space, axes, dst_nout)
    }

    fn trace<P: AsRef<[D]>>(
        space: &BoundDynamicFusionMapSpace<R>,
        payload: impl FnOnce() -> P,
        axes: tenet_tensors::TensorTraceAxisSpec<'_>,
        dst_nout: usize,
    ) -> Result<(BoundDynamicFusionMapSpace<R>, Vec<D>), Error> {
        let stage = tenet_tensors::tensortrace_stage_multiplicity_free(space, axes, dst_nout)?;
        Ok(tenet_tensors::tensortrace_multiplicity_free_in(
            stage,
            space,
            payload,
            axes,
            D::from_real(1.0),
        )?)
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
    fn trace<P: AsRef<[D]>>(
        space: &BoundDynamicFusionMapSpace<R>,
        payload: impl FnOnce() -> P,
        axes: tenet_tensors::TensorTraceAxisSpec<'_>,
        dst_nout: usize,
    ) -> Result<(BoundDynamicFusionMapSpace<R>, Vec<D>), Self::FacadeError> {
        let stage = tenet_tensors::tensortrace_stage_checked_generic(space, axes, dst_nout)?;
        Ok(tenet_tensors::tensortrace_checked_generic_in(
            stage,
            space,
            payload,
            axes,
            D::from_real(1.0),
        )?)
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
fn trace_pair_axes(
    rank: usize,
    codomain_rank: usize,
    pairs: &[(usize, usize)],
) -> Result<Option<TracePairAxes>, Error> {
    let mut seen = tenet_core::axes::AxisMask::new(rank);
    for &(lhs, rhs) in pairs {
        if seen.insert(lhs).and_then(|()| seen.insert(rhs)).is_err() {
            // The traced axes are an axis subset, reported as every other
            // subset misuse (#1873), with the pairs flattened in order.
            return Err(tenet_tensors::OperationError::InvalidAxisSet {
                tensor: "trace pairs",
                axes: pairs.iter().flat_map(|&(lhs, rhs)| [lhs, rhs]).collect(),
                rank,
            }
            .into());
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

/// A trace's source in storage orientation: an owned body as it is, or a
/// lazy adjoint's parent with the pair axes mapped onto the parent's legs
/// and the source conjugated. Shared by every trace entry point and
/// placement, so they lower an adjoint identically.
pub(super) struct TraceSource<'t, R, D, S> {
    pub(super) body: &'t Arc<TypedTensorBody<R, D, S>>,
    pub(super) axes: TracePairAxes,
    pub(super) conjugate: bool,
}

impl<R, D, S> TraceSource<'_, R, D, S> {
    pub(super) fn spec(&self) -> tenet_tensors::TensorTraceAxisSpec<'_> {
        tenet_tensors::TensorTraceAxisSpec::new_with_conjugation(
            &self.axes.output_axes,
            &self.axes.trace_lhs,
            &self.axes.trace_rhs,
            self.conjugate,
        )
    }
}

/// Gates the provider's braiding, validates `pairs` and lowers `tensor` to
/// its trace source; `None` for an empty list.
///
/// The braiding gate comes first, as in TensorKit `trace_permute!`, which
/// rejects a non-symmetric braiding before it looks at the indices, even for
/// an empty trace.
pub(super) fn trace_source<'t, R, D, S>(
    tensor: &'t TensorMap<R, D, S>,
    braiding: tenet_core::BraidingStyleKind,
    pairs: &[(usize, usize)],
) -> Result<Option<TraceSource<'t, R, D, S>>, Error> {
    tenet_tensors::require_symmetric_braiding(braiding, tenet_tensors::SymmetricBraidingOp::Trace)?;
    let Some(mut axes) = trace_pair_axes(tensor.rank(), tensor.codomain_rank(), pairs)? else {
        return Ok(None);
    };
    Ok(Some(match &tensor.repr {
        TypedTensorRepr::Owned(body) => TraceSource {
            body,
            axes,
            conjugate: false,
        },
        TypedTensorRepr::Adjoint(view) => {
            let parent = view.parent.space.space();
            let (nout, nin) = (parent.nout(), parent.nin());
            for list in [
                &mut axes.output_axes,
                &mut axes.trace_lhs,
                &mut axes.trace_rhs,
            ] {
                *list = logical_adjoint_axes_to_parent(nout, nin, list);
            }
            TraceSource {
                body: &view.parent,
                axes,
                conjugate: true,
            }
        }
    }))
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
    /// must be in range and appear at most once; past the braiding gate,
    /// `pairs.is_empty()` returns a clone. This is the categorical contraction trace, so provider pivotal
    /// coefficients and fermionic twists are included. It can therefore be a
    /// supertrace and need not equal [`Self::tr`].
    ///
    /// Checked Generic requires [`crate::sector::CheckedGenericPivotal`] and
    /// admits the output with the source's exact provider `Arc`. Lazy
    /// adjoints are read through their parent without materializing.
    ///
    /// # Complexity
    ///
    /// Dense storage runs the partial-trace engine over the whole payload. A
    /// multiplicity-free compact spectrum factor traced over its only pair
    /// reduces the stored spectrum in `O(Σ_c k_c)` without materializing
    /// (#604), with a deliberately narrow guard: one pair on a rank-(1,1)
    /// source, where the destination tree is empty and the coefficient
    /// collapses to a per-sector scalar, `dim(c) · θ(c)` on a direct traced
    /// codomain leg and `dim(c)` on a dual one. Other compact cases
    /// materialize inside the runtime's Host pool.
    ///
    /// # Errors
    ///
    /// In TensorKit `trace_permute!`'s order: `UnsupportedTensorContractScope`
    /// for a non-symmetric braiding (even with an empty `pairs`), then
    /// [`Error::Operation`] with `InvalidAxisSet { tensor: "trace pairs" }`
    /// for an out-of-range or repeated axis, then
    /// `StructureMismatch { "trace axes" }` for legs that are not mutually
    /// dual, in both admission modes. Trace execution failures return their
    /// corresponding [`Error`] variant. If required pivotal data is unavailable, the original
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
        let _host_pool = self.runtime.enter_host_pool();
        let braiding = <R::Mode as TypedTensorModeDispatch<R>>::braiding_style(self.provider());
        let Some(source) = trace_source(self, braiding, pairs)? else {
            return Ok(self.clone());
        };
        let (space, axes) = (&source.body.space, source.spec());
        let dst_nout = source.axes.destination_codomain_rank;
        if let Some(compact) = <R::Mode as TypedTensorTraceDispatch<R, D>>::try_compact_trace(
            self, space, axes, dst_nout,
        )? {
            return Ok(compact);
        }
        let (space, data) = <R::Mode as TypedTensorTraceDispatch<R, D>>::trace(
            space,
            || source.body.materialized_dense_data(),
            axes,
            dst_nout,
        )?;
        Ok(self.published(space, data))
    }
}

#[allow(private_bounds)]
fn compose_multiplicity_free<R, D>(
    lhs: &TensorMap<R, D>,
    rhs: &TensorMap<R, D>,
) -> Result<(BoundDynamicFusionMapSpace<R>, Vec<D>), Error>
where
    R: MultiplicityFreeRigidSymbols + CheckedFusionAlgebra + SectorCodec,
    R::Scalar: CategoricalScalar + tenet_tensors::DenseRecouplingScalar,
    D: TensorScalar + crate::runtime::MultiplicityFreeCoefficientLane<R::Scalar>,
{
    let mut lease = lhs.runtime.lease_context()?;
    let (lhs_operand, lhs_data) = lhs.fusion_operand_and_data();
    let (rhs_operand, rhs_data) = rhs.fusion_operand_and_data();
    Ok(
        D::lane(lease.context())?.tensorcompose_multiplicity_free_in(
            (lhs.logical_space(), lhs_operand, &lhs_data),
            (rhs.logical_space(), rhs_operand, &rhs_data),
        )?,
    )
}

impl<R, D> MultiplicityFreeContractExecution<R, f64> for D
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64> + CheckedFusionAlgebra + SectorCodec,
    D: TensorScalar,
{
    fn contract(
        lhs: &TensorMap<R, Self>,
        rhs: &TensorMap<R, Self>,
        spec: &ContractSpec<'_>,
    ) -> Result<(BoundDynamicFusionMapSpace<R>, Vec<Self>), Error> {
        let output_axes = spec.output_axes();
        let mut lease = lhs.runtime.lease_context()?;
        let context = lease.context().multiplicity_free_lane::<Self>()?;
        let (lhs_operand, lhs_data) = lhs.fusion_operand_and_data();
        let (rhs_operand, rhs_data) = rhs.fusion_operand_and_data();
        #[cfg(test)]
        crate::tensor_core::observe_contract_seam_call();
        Ok(context.tensorcontract_multiplicity_free_in(
            (lhs.logical_space(), lhs_operand, &lhs_data),
            (rhs.logical_space(), rhs_operand, &rhs_data),
            (spec.lhs, spec.rhs, OutputAxisOrder::from_axes(&output_axes)),
            spec.codomain.len(),
        )?)
    }

    fn compose(
        lhs: &TensorMap<R, Self>,
        rhs: &TensorMap<R, Self>,
    ) -> Result<(BoundDynamicFusionMapSpace<R>, Vec<Self>), Error> {
        compose_multiplicity_free(lhs, rhs)
    }
}

impl<R> MultiplicityFreeContractExecution<R, Complex64> for Complex64
where
    R: MultiplicityFreeRigidSymbols<Scalar = Complex64> + CheckedFusionAlgebra + SectorCodec,
{
    fn contract(
        _lhs: &TensorMap<R, Self>,
        _rhs: &TensorMap<R, Self>,
        _spec: &ContractSpec<'_>,
    ) -> Result<(BoundDynamicFusionMapSpace<R>, Vec<Self>), Error> {
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
    ) -> Result<(BoundDynamicFusionMapSpace<R>, Vec<Self>), Error> {
        compose_multiplicity_free(lhs, rhs)
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
