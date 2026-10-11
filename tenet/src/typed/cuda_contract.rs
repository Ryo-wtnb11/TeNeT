use super::*;

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
impl<R, D> TensorMap<R, D, CudaStorage<D>>
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64> + CheckedFusionAlgebra + SectorCodec,
    D: CudaPayload,
{
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
    /// [`crate::typed::CudaTreeTransformStats::context_scalar_operand_bytes`]).
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
    /// In the Host's order, all before any device work: the braiding gate; the
    /// Host's [`Error::Operation`] with `InvalidAxisSet { tensor: "trace pairs" }`
    /// for a malformed pair list and its errors for legs that are not mutually
    /// dual; [`Error::UnsupportedOnDevice`] for a
    /// compact (diagonal) device payload, where the Host takes its
    /// compact-spectrum arm; the Host compile's own errors;
    /// [`Error::PlacementMismatch`]. A rejected call leaves the device and the
    /// Runtime unchanged.
    pub fn trace_pairs(&self, pairs: &[(usize, usize)]) -> Result<Self, Error> {
        match self.prepare_trace_pairs(pairs, None)? {
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
    /// Past the braiding gate, an empty `pairs` is [`Self::axpby_into`].
    ///
    /// # Errors
    ///
    /// The Host [`TensorMap::trace_pairs_into`]'s order:
    /// [`Error::RuntimeMismatch`] / [`Error::RuleMismatch`] against the
    /// destination; the braiding gate and pair-list errors of
    /// [`Self::trace_pairs`]; `SpaceMismatch` for a destination
    /// whose space or layout is not the result's; the duality error; then
    /// [`Self::trace_pairs`]'s device errors, [`Error::InvalidArgument`] for a
    /// destination that is not owned dense CUDA storage, aliases the source,
    /// or has the wrong length, [`Error::DestinationShared`] and
    /// [`Error::PlacementMismatch`].
    ///
    /// # Failure
    ///
    /// A validation or capability error leaves `destination` bit-identical:
    /// every check completes before the first write. On failure after
    /// validation (a backend or other execution error) the destination's
    /// contents are unspecified, while its space and block structure stay
    /// intact.
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
        let Some(trace) =
            self.prepare_trace_pairs(pairs, Some(destination.logical_space().space()))?
        else {
            return self.axpby_into(destination, alpha, beta);
        };
        let destination_storage = unique_dense_destination(
            destination,
            &self.storage_body().data,
            trace.space.space(),
            "CUDA",
        )?;
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
    /// `None` for an empty pair list.
    pub(crate) fn prepare_trace_pairs(
        &self,
        pairs: &[(usize, usize)],
        destination_space: Option<&tenet_tensors::DynamicFusionMapSpace>,
    ) -> Result<Option<CudaTracePairs<'_, R, D>>, Error> {
        let braiding = self.provider().braiding_style();
        let Some(traced) = trace_source(self, braiding, pairs)? else {
            return Ok(None);
        };
        let destination_codomain_rank = traced.axes.destination_codomain_rank;
        let source_space = &traced.body.space;
        let source_data = traced.body.data.as_ref();
        let axes = traced.spec();
        let preflight = tenet_tensors::tensortrace_fusion_dyn_preflight_checked(
            source_space,
            axes,
            destination_codomain_rank,
        )?;
        let dual_pairs = preflight.require_dual_pairs();
        let space = source_space.derive_from_final_homspace(preflight.into_selected_homspace())?;
        // TensorKit `trace_permute!`'s order: the destination space, then the
        // pair duality; the representation and storage checks are TeNeT's own.
        if let Some(destination_space) = destination_space {
            require_destination_space(destination_space, space.space())?;
        }
        dual_pairs?;
        let TypedData::Dense(source) = source_data else {
            return Err(Error::UnsupportedOnDevice(
                "trace_pairs requires dense CUDA storage".to_string(),
            ));
        };
        let structure = <tenet_core::MultiplicityFreeAdmissionMode as tenet_tensors::PivotalCoefficientAlgebra<R>>::trace_terms(&space, source_space, axes)?;
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
}

/// The device contraction kernel: the route planned on the Host, replayed on
/// the device. Every transform structure, core plan, borrow decision and the
/// destination come from the one planner the Host uses, with the
/// fully-direct-core capabilities of the device executor.
#[cfg(feature = "cuda")]
impl<R, D> ContractExec<R, D> for CudaStorage<D>
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64> + CheckedFusionAlgebra + SectorCodec,
    D: CudaPayload,
{
    type Caps = tenet_tensors::DirectCoreExecutor;

    fn contract_owned<C: tenet_tensors::DenseBlockScalar>(
        replay: ContractReplay<'_, D, C>,
        lhs: &TensorMap<R, D, Self>,
        rhs: &TensorMap<R, D, Self>,
    ) -> Result<Self, TransformExecError>
    where
        D: RecouplingCoefficientAction<C>,
    {
        let (lhs, rhs) = (cuda_contract_operand(lhs)?, cuda_contract_operand(rhs)?);
        let ContractReplay {
            runtime,
            lease,
            resolution,
            destination,
            destination_len,
            lhs: lhs_structure,
            rhs: rhs_structure,
            ..
        } = replay;
        drop(lease);
        let mut lease = runtime.lease_cuda()?;
        let (cuda, transforms, scratch) = lease.split_contract();
        let device = Placement::Cuda(cuda.device());
        if lhs.placement() != device || rhs.placement() != device {
            return Err(Error::PlacementMismatch.into());
        }
        // ponytail: #740 — the device seam initializes an output by uploading
        // zeros; replace only with a measured native allocation.
        let mut dst = CudaStorage::upload_owned(cuda, vec![D::from_real(0.0); destination_len])
            .map_err(Error::from)?;
        tenet_tensors::execute_storage_contract_resolution_on_cuda(
            cuda,
            transforms,
            scratch,
            resolution,
            (destination, &mut dst),
            (lhs_structure, lhs),
            (rhs_structure, rhs),
            D::from_real(1.0),
            tenet_tensors::ContractDestinationInit::Zeroed,
        )
        .map_err(TransformExecError::Replay)?;
        drop(lease);
        Ok(dst)
    }

    /// The route executor admits every region before its first submission,
    /// so a rejected replay leaves the caller's destination byte-identical.
    fn contract_into<C: tenet_tensors::DenseBlockScalar>(
        replay: ContractReplay<'_, D, C>,
        lhs: &TensorMap<R, D, Self>,
        rhs: &TensorMap<R, D, Self>,
        destination: &mut Self,
        alpha: D,
        beta: D,
    ) -> Result<(), TransformExecError>
    where
        D: RecouplingCoefficientAction<C>,
    {
        let (lhs, rhs) = (cuda_contract_operand(lhs)?, cuda_contract_operand(rhs)?);
        let ContractReplay {
            runtime,
            lease,
            resolution,
            destination: destination_structure,
            lhs: lhs_structure,
            rhs: rhs_structure,
            ..
        } = replay;
        drop(lease);
        let mut lease = runtime.lease_cuda()?;
        let (cuda, transforms, scratch) = lease.split_contract();
        let device = Placement::Cuda(cuda.device());
        if lhs.placement() != device
            || rhs.placement() != device
            || destination.placement() != device
        {
            return Err(Error::PlacementMismatch.into());
        }
        tenet_tensors::execute_storage_contract_resolution_on_cuda(
            cuda,
            transforms,
            scratch,
            resolution,
            (destination_structure, destination),
            (lhs_structure, lhs),
            (rhs_structure, rhs),
            alpha,
            tenet_tensors::ContractDestinationInit::Axpby(beta),
        )
        .map_err(TransformExecError::Replay)
    }
}

/// The dense device payload a contraction operand stores: its own, or a
/// lazy adjoint's parent's. Device tensors hold no compact data — `to_cuda`
/// densifies and every device kernel outputs dense.
#[cfg(feature = "cuda")]
fn cuda_contract_operand<R, D: CudaPayload>(
    t: &TensorMap<R, D, CudaStorage<D>>,
) -> Result<&CudaStorage<D>, Error> {
    match t.storage_body().data.as_ref() {
        TypedData::Dense(storage) => Ok(storage),
        TypedData::Diagonal(_) => Err(internal_layout_error("CUDA tensors hold no compact data")),
    }
}
