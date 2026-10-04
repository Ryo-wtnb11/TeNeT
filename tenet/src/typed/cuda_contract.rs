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
    /// borrow decisions and the core plan, chosen by the one contraction
    /// planner (`plan_contract`) the batched [`crate::typed::ContractPlan`]
    /// also uses. The device only replays that route — each non-identity
    /// source transform into device scratch, the coupled-sector GEMMs, the
    /// output transform — which is TensorKit's `blas_contract!` dataflow and
    /// QSpace's `contract` dataflow, so FLOPs, bytes moved and working set are
    /// the Host's. Where TensorKit's `_contract_memcost` takes `copyC` (a
    /// zero-copy core whose result needs only a permute), the core writes the
    /// core-destination scratch and one transform writes the output. A
    /// contraction already in
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
    /// [`crate::typed::OperationError::UnsupportedTensorContractScope`] for
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
        // `plan_contract` in its two halves: the canonical core route resolves
        // from the operands alone and takes no lock. Otherwise the Host context
        // lease compiles the `CopyC` or `DynamicTree` route and is dropped
        // before the device lease is taken; nothing under the device lease
        // leases again.
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
                    .plan_contract_beyond_core(
                        &dst_space,
                        lhs_space,
                        lhs_operand,
                        rhs_space,
                        rhs_operand,
                        lhs_axes,
                        rhs_axes,
                        output_axes,
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
    /// of the Host route and placement. A compile error counts as validation.
    ///
    /// `alpha` and `beta` ride the epilogue of whatever writes each element:
    /// the core GEMMs (`alpha * job_alpha`, `beta`) when they write the
    /// destination directly, otherwise the output transform in `Axpby(beta)`
    /// mode. On the core route the coupled sectors no GEMM reaches become
    /// `beta * destination` through a zero-source region move — zeros for
    /// `beta = 0`, which never reads the destination, and nothing for
    /// `beta = 1`. An element an output transform writes becomes
    /// `alpha * source + beta * destination` even where the source is `+0`,
    /// so there a `-0.0` turns into `+0.0` for `beta = 1`, as on the Host
    /// ([`TensorMap::contract_into`]) and in TensorKit's `tensoradd!`.
    ///
    /// # Cost
    ///
    /// No destination reset or separate `beta` pass over written blocks. A
    /// warm call transfers nothing and allocates nothing on the device;
    /// scratch and coefficient payloads are as for [`Self::contract`].
    ///
    /// # Failure
    ///
    /// A validation or capability error leaves `destination` bit-identical:
    /// every check completes before the first write. On failure after
    /// validation (a backend or other execution error) the destination's
    /// contents are unspecified, while its space and block structure stay
    /// intact.
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
        // route, otherwise the rest of `plan_contract` under a context lease
        // dropped before the device lease.
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
                    .plan_contract_beyond_core(
                        &execution_destination,
                        lhs_space,
                        lhs_operand,
                        rhs_space,
                        rhs_operand,
                        lhs_axes,
                        rhs_axes,
                        output_axes,
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
    /// Host's [`Error::InvalidArgument`] for a malformed pair list and its
    /// errors for legs that are not mutually dual; [`Error::UnsupportedOnDevice`] for a
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
    /// [`Self::trace_pairs`]; [`Error::InvalidArgument`] for a destination
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
    /// `None` for an empty pair list. Lets `tensor!` decide every trace of a
    /// network before the first one allocates, at no second compile.
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
