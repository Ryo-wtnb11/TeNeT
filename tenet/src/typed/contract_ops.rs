use super::*;

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
}

impl<R, D> TensorMap<R, D>
where
    R: TypedSectorAdmission<Error = FusionAlgebraError, Mode = MultiplicityFreeAdmissionMode>
        + MultiplicityFreeRigidSymbols<Scalar = f64>
        + CheckedFusionAlgebra
        + SectorCodec,
    D: TensorScalar,
{
    /// `destination = alpha * self.trace_pairs(pairs) + beta * destination`,
    /// TensorKit `tensortrace!(C, A, p, q, false, α, β)`.
    ///
    /// TensorKit's `_trace_permute!` order: every destination layout becomes
    /// `beta * destination` first (a strong zero for `beta = 0`, which never
    /// reads the destination; nothing for `beta = 1`), then every trace term
    /// adds `alpha * coefficient * trace(block)`. That `beta` pass is the
    /// reference's own: several source blocks feed one destination block, so
    /// no single term's write can carry it. An empty `pairs` is exactly
    /// [`Self::axpby_into`], with its rules and errors — including its
    /// acceptance of a compact diagonal source. A lazy-adjoint source is read
    /// through its parent.
    ///
    /// # Errors
    ///
    /// [`Error::RuntimeMismatch`], [`Error::RuleMismatch`], the pair-list and
    /// duality errors of [`Self::trace_pairs`], [`Error::Unsupported`] for a
    /// compact (diagonal) source with a non-empty `pairs`, [`Error::InvalidArgument`] for a destination
    /// that is not owned dense host storage, aliases the source, or has the
    /// wrong space, layout or length, and [`Error::DestinationShared`].
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
        let Some(TracePairAxes {
            output_axes,
            destination_codomain_rank,
            trace_lhs,
            trace_rhs,
        }) = trace_pair_axes(self.rank(), self.codomain_rank(), pairs)?
        else {
            return host_axpby_into(self, destination, alpha, beta);
        };
        let mapped_output_axes;
        let mapped_trace_lhs;
        let mapped_trace_rhs;
        let (source_space, axes) = match &self.repr {
            TypedTensorRepr::Owned(body) => (
                &body.space,
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
                    tenet_tensors::TensorTraceAxisSpec::new_with_conjugation(
                        &mapped_output_axes,
                        &mapped_trace_lhs,
                        &mapped_trace_rhs,
                        true,
                    ),
                )
            }
        };
        let TypedData::Dense(source_data) = self.storage_body().data.as_ref() else {
            return Err(Error::Unsupported {
                operation: "trace_pairs_into",
                alternative: Alternative::Materialize,
            });
        };
        let homspace = tenet_tensors::tensortrace_fusion_dyn_selected_homspace_checked(
            source_space,
            axes,
            destination_codomain_rank,
        )?;
        let space = source_space.derive_from_final_homspace(homspace)?;
        let destination_body = match &destination.repr {
            TypedTensorRepr::Owned(body) if matches!(body.data.as_ref(), TypedData::Dense(_)) => {
                body
            }
            _ => {
                return Err(Error::InvalidArgument(
                    "destination must use ordinary dense host storage".to_string(),
                ))
            }
        };
        if Arc::ptr_eq(&destination_body.data, &self.storage_body().data) {
            return Err(Error::InvalidArgument(
                "destination storage must not alias an input".to_string(),
            ));
        }
        if destination_body.space.space() != space.space() {
            return Err(Error::InvalidArgument(
                "destination fusion space or block layout does not match the trace result"
                    .to_string(),
            ));
        }
        if Arc::strong_count(destination_body) != 1
            || Arc::strong_count(&destination_body.data) != 1
        {
            return Err(Error::DestinationShared);
        }
        let _host_pool = self.runtime.enter_host_pool();
        let TypedTensorRepr::Owned(destination_body) = &mut destination.repr else {
            return Err(internal_layout_error("ordinary destination checked above"));
        };
        let destination_data = Arc::get_mut(destination_body)
            .and_then(|body| Arc::get_mut(&mut body.data))
            .ok_or_else(|| internal_layout_error("unique destination checked above"))?;
        let TypedData::Dense(destination_data) = destination_data else {
            return Err(internal_layout_error("dense destination checked above"));
        };
        tenet_tensors::tensortrace_fusion_dyn_into_checked(
            &space,
            destination_data,
            source_space,
            source_data,
            axes,
            alpha,
            beta,
        )?;
        Ok(())
    }

    /// `destination = alpha * self.contract(other, spec) + beta * destination`,
    /// TensorKit `tensorcontract!(C, A, pA, false, B, pB, false, pAB, α, β)`,
    /// preserving the destination's provider, space, body and dense host
    /// allocation. `destination` must have the space of that result, including
    /// its codomain/domain split. A compact diagonal operand is densified into
    /// an operation-local buffer where [`Self::contract`] has no scaling arm.
    ///
    /// `alpha` and `beta` ride the epilogue of whatever writes each element:
    /// the core GEMMs (TensorKit's `mul!(C, A, B, α, β)`) or an output
    /// transform's add (`tensoradd!(C, Cnew, pAB, false, α, β)`). `beta == 0`
    /// never reads `destination` (NaN does not survive; unreached blocks
    /// become `+0`), and `alpha == 0` never reads the operands. Nothing clears
    /// the destination first.
    ///
    /// A coupled sector no GEMM reaches: on the core route, where the GEMMs
    /// write the destination directly, it becomes `beta * destination`
    /// (TensorKit's `rmul!(C, β)`), so `beta == 1` leaves it bit for bit. On a
    /// route with an output transform (the one-call route's C transform, or
    /// copyC below) the transform writes every element, so an unreached one
    /// becomes `alpha * (+0) + beta * destination`: IEEE addition turns a
    /// `-0.0` into `+0.0` even for `beta == 1`, and a non-finite `alpha`
    /// gives NaN — as TensorKit's `tensoradd!`, and as the eager
    /// [`Self::contract`] followed by [`Self::axpby`].
    ///
    /// **Route.** The same memcost choice as [`Self::contract`]: when a
    /// zero-copy candidate plus one output permute is cheaper, the product is
    /// written with its own output order into Runtime-pooled scratch and one
    /// tree transform adds it into `destination` with `alpha`/`beta`
    /// (TensorKit `blas_contract!`'s `copyC`); no operand is rebuilt.
    ///
    /// # Errors
    ///
    /// [`Error::RuntimeMismatch`]; then
    /// [`crate::typed::OperationError::UnsupportedTensorContractScope`] for
    /// non-symmetric (anyonic or `NoBraiding`) providers, as for
    /// [`Self::contract`]; [`Error::RuleMismatch`]; [`Error::InvalidArgument`]
    /// for a destination that is not owned dense host storage, aliases an
    /// operand, or has the wrong space, layout or length;
    /// [`Error::DestinationShared`] when `destination` shares its storage
    /// with a clone. Runtime-context leasing counts as validation.
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
        other: impl Into<TensorRef<'a, R, D>>,
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

        let destination_body = match &destination.repr {
            TypedTensorRepr::Owned(body) if matches!(body.data.as_ref(), TypedData::Dense(_)) => {
                body
            }
            _ => {
                return Err(Error::InvalidArgument(
                    "contraction destination must use ordinary dense host storage".to_string(),
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

        let output_order = OutputAxisOrder::from_axes(output_axes);
        let expected = BoundDynamicFusionMapSpace::contracted_multiplicity_free_partitioned(
            self.logical_space(),
            other.logical_space(),
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
        let execution_destination = self
            .logical_space()
            .rebind_validated(&destination_body.space.validated_layout())?;

        // Why not measure `fusion_operand_and_data()`: it densifies a compact
        // operand, and the copyC route densifies it again for its own
        // contraction. A compact payload densifies to exactly the required
        // length by construction, so only stored dense lengths are checked.
        let stored_dense_len = |tensor: &Self| match &tensor.repr {
            TypedTensorRepr::Owned(body) => match body.data.as_ref() {
                TypedData::Dense(data) => Some(data.len()),
                TypedData::Diagonal(_) => None,
            },
            TypedTensorRepr::Adjoint(view) => Some(view.parent_data().len()),
        };
        let required_destination = destination_body.space.space().required_len()?;
        let actual_destination = match destination_body.data.as_ref() {
            TypedData::Dense(data) => data.len(),
            TypedData::Diagonal(_) => unreachable!("dense destination checked above"),
        };
        for (tensor, actual, required) in [
            (
                "lhs",
                stored_dense_len(self),
                self.fusion_operand().storage_space().required_len()?,
            ),
            (
                "rhs",
                stored_dense_len(other),
                other.fusion_operand().storage_space().required_len()?,
            ),
            (
                "destination",
                Some(actual_destination),
                required_destination,
            ),
        ] {
            let Some(actual) = actual else {
                continue;
            };
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
        let copy_c =
            super::checked_generic_contract::CopyC::plan(self, other, spec, expected.space())?;

        let mut lease = self.runtime.lease_context()?;
        let context = lease.context().multiplicity_free_lane::<D>()?;
        let TypedTensorRepr::Owned(destination_body) = &mut destination.repr else {
            unreachable!("ordinary destination checked above")
        };
        let destination_body =
            Arc::get_mut(destination_body).expect("unique destination body checked above");
        let destination_provider = destination_body.space.provider();
        let destination_structure = destination_body.space.space().structure();
        let destination_data = Arc::get_mut(&mut destination_body.data)
            .expect("unique destination payload checked above");
        let TypedData::Dense(destination_data) = destination_data else {
            unreachable!("dense destination checked above")
        };
        if let Some(copy_c) = copy_c {
            // TensorKit `blas_contract!` with `C` as the destination:
            // `mul!(Cnew, A, B)`, then `tensoradd!(C, Cnew, pAB, false, α, β)`.
            // The temporary is written before `destination` is touched.
            let temporary = copy_c.contract_temporary(context)?;
            context.tree_context_mut().tree_transform_dyn_into_ref(
                destination_provider,
                &copy_c.operation,
                destination_structure,
                copy_c.temporary_space.space().structure(),
                destination_data.as_mut_slice(),
                temporary.as_slice(),
                alpha,
                beta,
            )?;
            context.restore_copy_c_scratch(temporary);
            return Ok(());
        }
        let (lhs, lhs_data) = self.fusion_operand_and_data();
        let (rhs, rhs_data) = other.fusion_operand_and_data();
        context.tensorcontract_fusion_dyn_prelowered_into(
            &execution_destination,
            destination_data,
            lhs,
            &lhs_data,
            rhs,
            &rhs_data,
            TensorContractSpec::new_with_conjugation(
                lhs_axes,
                rhs_axes,
                output_order,
                lhs.storage_conjugate(),
                rhs.storage_conjugate(),
            ),
            alpha,
            beta,
        )?;
        Ok(())
    }
}

impl<R, D> TensorMap<R, D>
where
    R: TypedSectorAdmission,
    R::Mode: TypedTensorProductDispatch<R, D>,
    D: TensorScalar,
{
    /// Tensor product in one category, ordered as
    /// `codomain(self), codomain(other); domain(self), domain(other)`.
    ///
    /// The two codomain trees and the two domain trees are merged
    /// independently with F moves. No legs cross and no R symbol is needed,
    /// including for a `NoBraiding` provider.
    ///
    /// Equal provider identities are sufficient; the two tensors may own
    /// different `Arc` allocations. The output always retains `self`'s exact
    /// provider allocation. A compact diagonal operand is densified into an
    /// operation-local buffer first; the output is dense either way.
    ///
    /// # Errors
    ///
    /// [`Error::RuntimeMismatch`] is reported before provider work. Checked
    /// Generic providers preserve algebra and malformed-F failures in
    /// [`GenericTensorError::TensorProduct`].
    ///
    /// ```compile_fail
    /// use tenet::sector::FibonacciFusionRule;
    /// use tenet::typed::TensorMap;
    /// fn unavailable(tensor: &TensorMap<FibonacciFusionRule, f64>) {
    ///     let _ = tensor.otimes(tensor);
    /// }
    /// ```
    ///
    /// ```compile_fail
    /// use tenet::sector::FibonacciFusionRule;
    /// use tenet::typed::Complex64;
    /// use tenet::typed::TensorMap;
    /// fn unavailable(tensor: &TensorMap<FibonacciFusionRule, Complex64>) {
    ///     let _ = tensor.tr();
    /// }
    /// ```
    ///
    /// ```compile_fail
    /// use tenet::sector::FibonacciFusionRule;
    /// use tenet::typed::Complex64;
    /// use tenet::typed::TensorMap;
    /// fn unavailable(tensor: &TensorMap<FibonacciFusionRule, Complex64>) {
    ///     let _ = tensor.svd_full(&[0], &[1]);
    /// }
    /// ```
    ///
    ///
    /// An adjoint view `other` (`t.adjoint_view()`) returns
    /// [`Error::Unsupported`]: this operation would copy it. Pass
    /// `&t.adjoint()?.materialize()?` instead.
    pub fn otimes<'a>(
        &self,
        other: impl Into<TensorRef<'a, R, D>>,
    ) -> Result<Self, TypedFacadeError<R>> {
        let other = other.into().operand()?;
        let other = &*other;
        other.refuse_borrowed_view("otimes")?;
        if !self.runtime.same_runtime(&other.runtime) {
            return Err(Error::RuntimeMismatch.into());
        }
        <R::Mode as TypedTensorProductDispatch<R, D>>::tensor_product(self, other)
    }
}

/// The leg roles of a pairwise contraction (TensorOperations
/// `tensorcontract!`'s `pA[2]`, `pB[1]` and `pAB`): which legs are contracted,
/// and how the open legs are ordered and split into the result's codomain and
/// domain.
///
/// Open legs are numbered `0..open_rank`, the open legs of the left operand in
/// ascending axis order first, then those of the right operand.
/// `codomain ++ domain` must be a permutation of `0..open_rank`.
///
/// The result is defined as the contraction that puts every open leg of the
/// left operand in the codomain and every open leg of the right one in the
/// domain, followed by [`TensorMap::permute`] onto `(codomain, domain)`; the
/// operation performs it without that separate permute.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ContractSpec<'a> {
    /// Contracted axes of the left operand, paired in order with `rhs`.
    pub lhs: &'a [usize],
    /// Contracted axes of the right operand.
    pub rhs: &'a [usize],
    /// Open legs forming the result's codomain, in order.
    pub codomain: &'a [usize],
    /// Open legs forming the result's domain, in order.
    pub domain: &'a [usize],
}

impl ContractSpec<'_> {
    /// `codomain ++ domain`, the single output order the engine takes with
    /// [`Self::codomain`]'s length as the split.
    pub(super) fn output_axes(&self) -> SmallVec<[usize; 8]> {
        self.codomain.iter().chain(self.domain).copied().collect()
    }
}

impl<R, D> TensorMap<R, D>
where
    R: TypedSectorAdmission,
    R::Mode: TypedTensorContractDispatch<R, D>,
    D: TensorScalar,
{
    /// Contracts `spec.lhs` of `self` with `spec.rhs` of `other` (pairwise, in
    /// list order) and returns the open legs as `spec.codomain ← spec.domain`
    /// (TensorKit `tensorcontract!` with `pAB = (codomain, domain)`).
    ///
    /// The result equals the contraction whose codomain is every open axis of
    /// `self` and whose domain is every open axis of `other`, followed by
    /// [`Self::permute`] onto `(spec.codomain, spec.domain)` — see
    /// [`ContractSpec`]. A leg moved across the split is dualized exactly as
    /// that permute dualizes it.
    ///
    /// **Cost.** No permute pass follows the contraction. As in TensorKit's
    /// `blas_contract!`, the GEMMs write the result directly when its layout
    /// allows, and otherwise write a Runtime-pooled temporary that one tree
    /// transform moves into the result, inside this call — never slower than
    /// the contraction followed by the explicit permute.
    ///
    /// **Braiding scope**: ordinary contraction is available only for
    /// symmetric braiding (Bosonic, Fermionic), as TensorKit `blas_contract!`.
    /// An anyonic or unbraided (`NoBraiding`) provider is rejected even when
    /// these axes are the canonical, crossing-free ones: general axes are
    /// defined by braiding legs into place, and the call carries no planar
    /// embedding or braid direction. The crossing-free case is exactly
    /// [`Self::compose`], which every braiding style admits; general planar
    /// contraction is a separate operation (#1070). Behaviour change (#1372):
    /// a canonical `NoBraiding` contraction was accepted before.
    ///
    /// For fermionic symmetric braiding this **twists**
    /// dual contracted legs with the fermionic supertrace twist — unlike
    /// composition (TensorKit `A * B` / `mul!`), which never does. Bosonic
    /// rules are unaffected; fermionic rules can differ by signs.
    /// [`Self::compose`] is the other semantics, and its documentation states
    /// the exact relation between the two.
    ///
    /// # Compact fast paths
    ///
    /// Contracting **one** leg against a factor in compact diagonal storage —
    /// an `s` from [`Self::svd_compact`], a `d` from [`Self::eigh_full`] — is
    /// a per-leg bond scaling and is run as one: the other operand's contracted
    /// leg is multiplied by the spectrum in place of a GEMM, and the result is
    /// laid out with a single [`Self::permute`]. Mathematically it is the same
    /// tensor the dense route computes, so this is a cost question only, and any
    /// pattern that does not fit falls through to the dense route rather than
    /// being refused.
    ///
    /// **Complexity.** In `docs/complexity_parity_policy.md`'s parameters — `d`
    /// the per-sector bond degeneracy, `n` the other operand's *open*-leg size,
    /// so its blocks hold `d·n` entries — the dense route materializes the
    /// spectrum as a `Σ_c d_c²` block-diagonal buffer and multiplies it in, at
    /// O(d²) storage and O(d²·n) work. The scaling route touches each of those
    /// `d·n` entries once, at O(d) storage and O(d·n) work, which is the order
    /// that policy's row requires. `D · D` multiplies the two spectra
    /// elementwise and stays compact, at O(d).
    ///
    /// **TensorKit correspondence.** This is what TensorKit's
    /// `DiagonalTensorMap` gets from its type: `block(D, c)` is a `Diagonal`, so
    /// LinearAlgebra dispatches the multiplication to `lmul!`/`rmul!` scaling
    /// (`diagonal.jl`), with no braiding or recoupling of its own.
    ///
    /// **Which patterns.** Exactly the two geometries that are a composition on
    /// the contracted leg, in either order — the contracted leg of the compact
    /// operand is its bond, and the other operand's is a leg on the side that
    /// faces it (`t`'s domain against `D`'s codomain, or `D`'s domain against a
    /// codomain leg of `t`, at any position). A leg on the far side, more than
    /// one contracted leg, or an output order that would move the surviving
    /// bond of a `D · D` product across the codomain/domain split all take the
    /// dense route: the first two are not proved geometries, and the last is not
    /// equivalent to rebinding the product spectrum (checked in #453). A
    /// supertrace twist on a dual contracted leg of `other` would also decline,
    /// and cannot currently arise — see `try_contract_diagonal`.
    ///
    /// The result is bound to `self`'s provider allocation, the same
    /// left-authority rule [`Self::zeros`] uses for its first leg: the two
    /// operands must agree on
    /// [`crate::sector::FusionRule::rule_identity`], which makes the choice of
    /// allocation immaterial to the algebra.
    ///
    /// # Errors
    ///
    /// - [`Error::RuntimeMismatch`] when the operands belong to different
    ///   runtimes.
    /// - [`Error::Operation`] with
    ///   [`crate::typed::OperationError::UnsupportedTensorContractScope`] for
    ///   non-symmetric (anyonic or `NoBraiding`) providers, whatever the axes.
    /// - [`Error::Operation`] / [`Error::Core`] / [`Error::FusionAlgebra`] for
    ///   malformed axis lists, a `codomain ++ domain` that is not a
    ///   permutation of the open axes, mismatched contracted legs, or operands whose
    ///   providers report different rule identities. Those all come back from
    ///   the expert layer, which owns the rules; re-checking them here would
    ///   be a second copy free to drift.
    ///   Checked Generic providers preserve provider and replay failures in
    ///   [`GenericTensorError::Plan`] and currently accept direct-owned inputs.
    ///
    /// ```
    /// use std::sync::Arc;
    ///
    /// use tenet::sector::{U1FusionRule, U1Irrep};
    /// use tenet::typed::{ContractSpec, GradedSpace, Runtime, TensorMap};
    ///
    /// let runtime = Runtime::builder().build()?;
    /// let v = GradedSpace::try_new(
    ///     Arc::new(U1FusionRule),
    ///     [(U1Irrep::new(0), 2), (U1Irrep::new(1), 1)],
    /// )?;
    /// let t: TensorMap<_, f64> = TensorMap::rand_with_seed(&runtime, [&v], [&v], 8)?;
    /// let id = TensorMap::isomorphism(&runtime, [&v], [&v])?;
    ///
    /// // Contracting the domain leg with the identity's codomain leg is a
    /// // no-op on the payload; `[0] ← [1]` keeps the open axes in place.
    /// let spec = ContractSpec { lhs: &[1], rhs: &[0], codomain: &[0], domain: &[1] };
    /// let out = t.contract(&id, &spec)?;
    /// assert_eq!(out.dense_data()?, t.dense_data()?);
    ///
    /// // Both open legs in the codomain: the same as permuting afterwards.
    /// let spec = ContractSpec { lhs: &[1], rhs: &[0], codomain: &[0, 1], domain: &[] };
    /// assert_eq!(out.permute(&[0, 1], &[])?.dense_data()?, t.contract(&id, &spec)?.dense_data()?);
    /// # Ok::<(), tenet::typed::Error>(())
    /// ```
    pub fn contract<'a>(
        &self,
        other: impl Into<TensorRef<'a, R, D>>,
        spec: &ContractSpec<'_>,
    ) -> Result<Self, TypedFacadeError<R>> {
        let other = other.into().operand()?;
        let other = &*other;
        // The one check the expert layer cannot make: it never sees the two
        // runtimes, and mixing execution state across them is a trust-boundary
        // violation rather than an algebra error. Scalar type and placement
        // need no arm here — `D` and `S` are type parameters.
        if !self.runtime.same_runtime(&other.runtime) {
            return Err(Error::RuntimeMismatch.into());
        }
        <R::Mode as TypedTensorContractDispatch<R, D>>::contract(self, other, spec)
    }
}

impl<R, D> TensorMap<R, D>
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64> + CheckedFusionAlgebra + SectorCodec,
    D: TensorScalar,
{
    /// TensorKit `deligneproduct`: embeds `self` as `(a, 𝟙)` and `other` as
    /// `(𝟙, b)` in the supplied ordered product category, then combines the
    /// embedded tensors with the F-only [`Self::otimes`] route.
    ///
    /// This operation is typed-only and keeps the payload type `D` unchanged.
    /// The caller supplies the exact [`ProductFusionRule`], including its
    /// component providers and codec; both component [`crate::sector::RuleIdentity`] values
    /// must match the operands, and the codec participates in the product
    /// identity. [`CanonicalUnitFusionRule`] is required for both components
    /// because TeNeT stores no separate unitor data. Factor order and nested
    /// association are preserved exactly rather than reassociated or swapped.
    ///
    /// Validation reports [`Error::RuntimeMismatch`] before component
    /// [`Error::RuleMismatch`]. Both vacuum embeddings, codec decodes, and
    /// source/target fusion-tree bijections are prepared before either
    /// embedded `TensorMap` or layout is published. After that transaction
    /// succeeds, the operation builds the two embedded tensors by copying
    /// their dense data (materializing a compact operand when necessary).
    ///
    /// An adjoint view `other` (`t.adjoint_view()`) returns
    /// [`Error::Unsupported`]: this operation would copy it. Pass
    /// `&t.adjoint()?.materialize()?` instead.
    pub fn deligne_product<'a, R2, C>(
        &self,
        other: impl Into<TensorRef<'a, R2, D>>,
        product: Arc<ProductFusionRule<R, R2, C>>,
    ) -> Result<TensorMap<ProductFusionRule<R, R2, C>, D>, Error>
    where
        R: CanonicalUnitFusionRule,
        R2: MultiplicityFreeRigidSymbols<Scalar = f64>
            + CheckedFusionAlgebra
            + SectorCodec
            + CanonicalUnitFusionRule,
        C: ProductSectorCodec + Sync + 'static,
    {
        let other = other.into().operand()?;
        let other = &*other;
        other.refuse_borrowed_view("deligne_product")?;
        if !self.runtime.same_runtime(&other.runtime) {
            return Err(Error::RuntimeMismatch);
        }
        if product.left_rule().rule_identity() != self.logical_space().provider().rule_identity()
            || product.right_rule().rule_identity()
                != other.logical_space().provider().rule_identity()
        {
            return Err(Error::RuleMismatch);
        }
        let left_vacuum = product
            .left_rule()
            .decode_sector(product.left_rule().vacuum())?;
        let right_vacuum = product
            .right_rule()
            .decode_sector(product.right_rule().vacuum())?;
        let left = prepare_product_operand(
            self,
            Arc::clone(&product),
            |sector| ProductSector::new(sector, right_vacuum.clone()),
            |sector| sector.left().clone(),
        )?;
        let right = prepare_product_operand(
            other,
            product,
            |sector| ProductSector::new(left_vacuum.clone(), sector),
            |sector| sector.right().clone(),
        )?;
        let left = left.commit()?;
        let right = right.commit()?;
        left.otimes(&right)
    }
}

impl<R, D> TensorMap<R, D>
where
    R: TypedSectorAdmission,
    R::Mode: TypedTensorContractDispatch<R, D>,
    D: TensorScalar,
{
    /// Categorical composition of two tensor maps, TensorKit `A * B` / `mul!`:
    /// `self`'s whole domain is contracted against `other`'s whole codomain,
    /// leaving `self.codomain() <- other.domain()`.
    ///
    /// **Fermionic semantics**: unlike [`Self::contract`] (TensorKit
    /// `tensorcontract!` / `@tensor`), composition never twists dual
    /// contracted legs — there is no supertrace here. Bosonic rules cannot
    /// tell the two apart; a fermionic one differs by a sign on every dual
    /// contracted leg carrying an odd sector, so the exact relation is
    /// `self.compose(other) == self.contract(twist(other, other's dual
    /// codomain legs), ..)`. Reach for `compose` when you mean operator
    /// multiplication of tensor maps, and for `contract` when you mean
    /// index-notation contraction.
    ///
    /// **Anyonic and unbraided semantics**: composition remains
    /// coupled-sector block multiplication for every braiding style,
    /// `NoBraiding` included, as TensorKit `mul!` checks none. The fixed
    /// domain/codomain boundary supplies the whole geometry, so no legs are
    /// exchanged and no R symbol is used; this is why `compose` is admitted
    /// where [`Self::contract`] rejects the same canonical axes.
    ///
    /// The axes are not arguments, deliberately: composition is defined by the
    /// codomain/domain split itself, and TensorKit's `*` takes none.
    ///
    /// The result is bound to `self`'s provider allocation — the same
    /// left-authority rule as [`Self::contract`] and [`Self::zeros`] — with one
    /// exemption: the `D * t` compact arm below returns `t`'s own space and
    /// runtime handle, because that space *is* the destination and rebuilding
    /// it under the left allocation would be a copy for nothing. The two
    /// allocations must already agree on
    /// [`crate::sector::FusionRule::rule_identity`] for the composition to be
    /// legal at all, so the choice is immaterial to the algebra.
    ///
    /// # Compact fast paths
    ///
    /// When either operand carries compact diagonal storage — an `s` from
    /// [`Self::svd_compact`], a `d` from [`Self::eigh_full`] — and the
    /// destination is representable, this takes TensorKit's
    /// `DiagonalTensorMap` route instead of a GEMM: `t * D` and `D * t` scale
    /// one bond axis per block (`rmul!` / `lmul!`), and `D * D` multiplies the
    /// two spectra elementwise and stays compact. Verified twist-free against
    /// TK's `diagonal.jl`: `block(D, c)` is a `Diagonal`, so LinearAlgebra
    /// dispatches to scaling, with no braiding or recoupling. The result is the
    /// same tensor the dense route computes, so this is a cost question only,
    /// and any operand or destination that does not fit falls through to the
    /// dense path rather than being refused. That path, and every
    /// checked-Generic composition, densifies a compact operand into an
    /// operation-local buffer first.
    ///
    /// # Errors
    ///
    /// - [`Error::RuntimeMismatch`] when the operands belong to different
    ///   runtimes, as for [`Self::contract`].
    /// - [`Error::Operation`] / [`Error::Core`] / [`Error::FusionAlgebra`]
    ///   when the two are not composable — mismatched ranks, legs that are not
    ///   mutually dual, or providers reporting different rule identities.
    ///   Those come back from the expert layer, which owns the rules.
    ///   Checked Generic failures use [`GenericTensorError::Plan`].
    ///
    /// ```compile_fail
    /// use tenet::sector::FibonacciFusionRule;
    /// use tenet::typed::TensorMap;
    /// fn unavailable(tensor: &TensorMap<FibonacciFusionRule, f64>) {
    ///     let _ = tensor.compose(tensor);
    /// }
    /// ```
    #[doc(alias = "mul")]
    pub fn compose<'a>(
        &self,
        other: impl Into<TensorRef<'a, R, D>>,
    ) -> Result<Self, TypedFacadeError<R>> {
        let other = other.into().operand()?;
        let other = &*other;
        // Runtime first, exactly as `contract`: crossing runtimes is a
        // trust-boundary violation rather than an algebra error, and the
        // expert layer never sees the two runtimes.
        if !self.runtime.same_runtime(&other.runtime) {
            return Err(Error::RuntimeMismatch.into());
        }
        <R::Mode as TypedTensorContractDispatch<R, D>>::compose(self, other)
    }
}

impl<R, D> TensorMap<R, D>
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64> + CheckedFusionAlgebra + SectorCodec,
    D: TensorScalar,
{
    /// The compact arms of [`Self::compose`], or `None` when the operands or
    /// the destination cannot support one and the dense route must run.
    ///
    /// Each arm proves its destination rather than deriving it. Composition
    /// glues `self.codomain <- other.domain`, so:
    ///
    /// - `D * D` — both operands are bond spaces (`codomain == domain`), so
    ///   when the two spaces are equal the destination *is* that space, and
    ///   [`is_diagonal_bond_space`] certifies it can hold a compact result.
    /// - `t * D` — the destination is `t.codomain <- D.domain`; `D` is a bond
    ///   space, so `D.domain == D.codomain`, and requiring that to equal
    ///   `t.domain` makes the destination `t`'s own space. `t`'s payload is
    ///   then `t`'s data with each block's trailing axis scaled.
    /// - `D * t` — the mirror image, scaling `t`'s leading axis.
    ///
    /// Without those equalities the destination is a different space (a dual
    /// leg on the contracted side is the reachable case) and reusing an
    /// operand's would silently produce a tensor on the wrong space, so the
    /// arm declines and the expert layer decides — including by rejecting a
    /// composition that is not one at all.
    pub(super) fn compose_compact(&self, other: &Self) -> Result<Option<Self>, Error> {
        if !self.same_rule(other) {
            return Ok(None);
        }
        let (left, right) = (self.logical_space().space(), other.logical_space().space());
        match (self.spectrum(), other.spectrum()) {
            (Some(lhs), Some(rhs)) => {
                // Both clauses are unreachable today and stay for the reason
                // [`is_diagonal_bond_space`] gives. `left != right` is the
                // weaker one: two compact payloads on unequal bond spaces
                // necessarily carry spectra that differ in their sectors or
                // their lengths, so the elementwise product below would refuse
                // them anyway — just with `spectra_disagree`'s message instead
                // of the expert layer's. Removing it would change which error a
                // caller sees, not whether one is reported.
                if left != right || !is_diagonal_bond_space(left) {
                    return Ok(None);
                }
                if lhs.len() != rhs.len() {
                    return Err(spectra_disagree());
                }
                let product = lhs
                    .iter()
                    .zip(rhs)
                    .map(|(left, right)| {
                        if left.sector != right.sector || left.values.len() != right.values.len() {
                            return Err(spectra_disagree());
                        }
                        Ok(tenet_matrixalgebra::SectorSpectrum {
                            sector: left.sector,
                            values: left
                                .values
                                .iter()
                                .zip(&right.values)
                                .map(|(&a, &b)| a * b)
                                .collect(),
                        })
                    })
                    .collect::<Result<Vec<_>, Error>>()?;
                Ok(Some(self.with_spectrum(product)))
            }
            // `t * D`: scale each block's trailing axis (TensorKit `rmul!`).
            (None, Some(spectrum)) => {
                if !is_diagonal_bond_space(right)
                    || left.homspace().domain().legs() != right.homspace().codomain().legs()
                {
                    return Ok(None);
                }
                self.scaled_axis(None, spectrum).map(Some)
            }
            // `D * t`: scale each block's leading axis (TensorKit `lmul!`).
            (Some(spectrum), None) => {
                if !is_diagonal_bond_space(left)
                    || right.homspace().codomain().legs() != left.homspace().domain().legs()
                {
                    return Ok(None);
                }
                other.scaled_axis(Some(0), spectrum).map(Some)
            }
            (None, None) => Ok(None),
        }
    }

    /// The compact arm of [`Self::contract`] (issue #584), or `None` when the
    /// operands, the axis pattern or the output order do not fit one and the
    /// dense route must run.
    ///
    /// A one-axis contraction against a compact operand is a bond scaling, so
    /// the spectrum multiplies the *other* operand's contracted leg — O(d·n)
    /// against the dense route's O(d²·n) GEMM on a materialized `Σ_c d_c²`
    /// buffer — and one [`Self::permute`] lays the result out. The permute is
    /// what carries every recoupling and bend, so this adds no mathematics of
    /// its own; it is a scale followed by one permutation.
    ///
    /// # Which patterns, and why only those
    ///
    /// The engine admits a contracted pair only when the two legs agree on
    /// their raw duality flag on the compose-shaped pairing (one operand's
    /// domain leg against the other's codomain leg), and a compact operand's
    /// leg *is* its bond on both sides. So each arm requires exactly that
    /// pairing and compares the two legs itself: raw equality is the engine's
    /// admissibility condition here, so a mismatch is a contraction the dense
    /// route must reject rather than one this arm may answer, and the arm
    /// declines so the expert layer reports it in its own words. `D · D` is
    /// handed to [`Self::compose_compact`], which is the same product and
    /// already proves its destination.
    ///
    /// # The twist, and why it is not folded
    ///
    /// [`Self::contract`] applies the fermionic supertrace twist to a **dual**
    /// contracted leg of `other`, where [`Self::compose`] does not. Here the case
    /// cannot arise, so the arm declines instead of carrying arithmetic no test
    /// could reach: a compact payload's bond leg is built non-dual
    /// (`diagonal_bond_bound_space_like`), the arms pair it with a *codomain*
    /// leg of `other` whose external duality is exactly its raw flag, and
    /// admissibility forces that flag to equal the bond's. The guard stays
    /// because the first constructor of a compact payload on a dual bond leg —
    /// or of an arm pairing a domain leg of `other` — should decline rather
    /// than silently return a wrong sign.
    pub(super) fn try_contract_diagonal(
        &self,
        other: &Self,
        lhs_axes: &[usize],
        rhs_axes: &[usize],
        output_axes: &[usize],
        codomain_rank: usize,
    ) -> Result<Option<Self>, Error> {
        if lhs_axes.len() != 1 || rhs_axes.len() != 1 || !self.same_rule(other) {
            return Ok(None);
        }
        let (lhs_axis, rhs_axis) = (lhs_axes[0], rhs_axes[0]);
        if lhs_axis >= self.rank() || rhs_axis >= other.rank() {
            return Ok(None);
        }
        // Why the provider rather than a stored flag: `braiding_style` is the
        // rule's own answer, and `R` is concrete here.
        let fermionic = self.logical_space().provider().braiding_style()
            == tenet_core::BraidingStyleKind::Fermionic;
        if fermionic
            && other
                .logical_space()
                .space()
                .homspace()
                .external_axis_is_dual(rhs_axis)
                != Some(false)
        {
            return Ok(None);
        }
        let (left, right) = (self.logical_space().space(), other.logical_space().space());
        let (left_home, right_home) = (left.homspace(), right.homspace());
        match (self.spectrum(), other.spectrum()) {
            // `D · D`: the same product as `D * D`, which already knows how to
            // stay compact and which destinations may hold the result.
            (Some(_), Some(_)) => {
                if lhs_axis != 1
                    || rhs_axis != 0
                    || codomain_rank != 1
                    || output_axes.iter().copied().ne(0..2)
                {
                    // Why not a reordered output: `pAB` can move the surviving
                    // bond across the codomain/domain split, and rebinding the
                    // product spectrum there is not equivalent to a permute
                    // (#453).
                    return Ok(None);
                }
                self.compose_compact(other)
            }
            // `t · D` (TensorKit `rmul!`): scale `t`'s contracted domain leg,
            // then move it to where the contraction's output order wants it.
            (None, Some(spectrum)) => {
                if rhs_axis != 0 || lhs_axis < self.codomain_rank() {
                    return Ok(None);
                }
                if left_home.domain().legs()[lhs_axis - self.codomain_rank()]
                    != right_home.codomain().legs()[0]
                {
                    return Ok(None);
                }
                let mut source: Vec<usize> = (0..self.rank()).filter(|&a| a != lhs_axis).collect();
                source.push(lhs_axis);
                self.scaled_axis(Some(lhs_axis), spectrum)?
                    .permuted_to_output(&source, output_axes, codomain_rank)
            }
            // `D · t` (TensorKit `lmul!`): the mirror image, scaling the
            // contracted codomain leg of `t` at whatever position it sits.
            (Some(spectrum), None) => {
                if lhs_axis != 1 || rhs_axis >= other.codomain_rank() {
                    return Ok(None);
                }
                if left_home.domain().legs()[0] != right_home.codomain().legs()[rhs_axis] {
                    return Ok(None);
                }
                let mut source = vec![rhs_axis];
                source.extend((0..other.rank()).filter(|&a| a != rhs_axis));
                let Some(completed) = other
                    .scaled_axis(Some(rhs_axis), spectrum)?
                    .permuted_to_output(&source, output_axes, codomain_rank)?
                else {
                    return Ok(None);
                };
                // Scaling and permutation deliberately run on `other`; only
                // after both succeed do we rebind their validated owned-dense
                // result to the public contract's exact left authority.
                let TensorMap { repr, .. } = completed;
                let TypedTensorRepr::Owned(body) = repr else {
                    return Ok(None);
                };
                if !matches!(body.data.as_ref(), TypedData::Dense(_)) {
                    return Ok(None);
                }
                let space = self
                    .logical_space()
                    .rebind_validated(&body.space.validated_layout())?;
                Ok(Some(Self {
                    runtime: self.runtime.clone(),
                    repr: owned_repr(TypedTensorBody::with_shared_payload(
                        space,
                        Arc::clone(&body.data),
                    )),
                }))
            }
            (None, None) => Ok(None),
        }
    }

    /// This tensor's axes, listed in `source[output_axes[..]]` order and split
    /// at `codomain_rank`, or `None` when `output_axes` is not a permutation of
    /// `0..source.len()`.
    ///
    /// `source` is the contraction's default output order expressed as axes of
    /// the scaled operand. An `output_axes` that is not a
    /// permutation declines rather than errors: the dense route validates it
    /// and reports it, and one error message beats two.
    fn permuted_to_output(
        &self,
        source: &[usize],
        output_axes: &[usize],
        codomain_rank: usize,
    ) -> Result<Option<Self>, Error> {
        let mut sorted = output_axes.to_vec();
        sorted.sort_unstable();
        if sorted.iter().copied().ne(0..source.len()) {
            return Ok(None);
        }
        let ordered: Vec<usize> = output_axes.iter().map(|&axis| source[axis]).collect();
        if codomain_rank == self.codomain_rank()
            && ordered[..codomain_rank]
                .iter()
                .copied()
                .eq(0..codomain_rank)
            && ordered[codomain_rank..]
                .iter()
                .copied()
                .eq(codomain_rank..source.len())
        {
            return Ok(Some(self.clone()));
        }
        self.tree_transform_multiplicity_free_real(TreeTransformOperation::permute(
            ordered[..codomain_rank].iter().copied(),
            ordered[codomain_rank..].iter().copied(),
        ))
        .map(Some)
    }

    /// This tensor with one bond axis of every block scaled by `spectrum`,
    /// on its own space. `axis = None` scales the trailing axis, `Some(0)` the
    /// leading one, exactly as the seam names them.
    fn scaled_axis(
        &self,
        axis: Option<usize>,
        spectrum: &[tenet_matrixalgebra::SectorSpectrum<D>],
    ) -> Result<Self, Error> {
        let _host_pool = self.runtime.enter_host_pool();
        let mut data = if matches!(&self.repr, TypedTensorRepr::Adjoint(_)) {
            let (operand, source) = self.fusion_operand_and_data();
            tenet_tensors::oriented_fusion_add_owned(
                self.logical_space().space().structure(),
                operand,
                &source,
                operand,
                &source,
                D::from_real(1.0),
                D::from_real(0.0),
            )?
        } else {
            self.owned_body()
                .expect("owned scaled-axis input")
                .materialized_dense_data()
                .as_ref()
                .to_vec()
        };
        tenet_matrixalgebra::seam::scale_axis_by_spectrum_mapped(
            self.logical_space().space(),
            &mut data,
            axis,
            spectrum,
            |value| value,
        )?;
        Ok(self.with_data(data))
    }

    /// Whether two operands' providers are the same rule. The compact paths
    /// below skip the expert layer, which is where a mismatch would otherwise
    /// be caught, so they have to ask themselves.
    pub(super) fn same_rule(&self, other: &Self) -> bool {
        self.logical_space().provider().rule_identity()
            == other.logical_space().provider().rule_identity()
    }

    /// Partial trace over pairs of mutually dual legs (TensorKit
    /// `tensortrace!` / TensorOperations `@tensor a[i, i; j]`).
    ///
    /// Each `(lhs, rhs)` pair of flat axis numbers (`0..rank`, codomain axes
    /// first) is traced away; the remaining legs keep their order and their
    /// codomain/domain side. Tracing nothing returns the source.
    ///
    /// This is the **tensor-contraction** trace: it applies the categorical
    /// trace coefficients, including a fermionic rule's twists, so it is the
    /// supertrace there. [`Self::tr`] is the dimension-weighted block trace,
    /// and the two genuinely disagree for a fermionic provider.
    ///
    /// TensorKit's native parallel-list `Index2Tuple` is what the seam takes
    /// internally; the Rust API uses `&[(usize, usize)]`.
    ///
    /// # Complexity
    ///
    /// Dense storage runs the partial-trace engine over the whole payload. A
    /// compact spectrum factor traced over its only pair reduces the stored
    /// spectrum in `O(Σ_c k_c)` without materializing (#604), with a
    /// deliberately narrow guard: one pair on a rank-(1,1) source, where the
    /// destination tree is empty and the coefficient collapses to a per-sector
    /// scalar,
    /// `dim(c) · θ(c)` on a direct traced codomain leg and `dim(c)` on a dual
    /// one. That twist is what makes this the supertrace and not [`Self::tr`];
    /// the coefficient is checked numerically against the engine route by the
    /// oracle sweeps in `tests/typed_facade.rs`.
    ///
    /// # Errors
    ///
    /// [`Error::InvalidArgument`] when the pair list is malformed — an axis out
    /// of range, or one named twice.
    /// Otherwise [`Error::Operation`] / [`Error::Core`] /
    /// [`Error::FusionAlgebra`] from the seam, which owns the rest of the
    /// validation (legs that are not mutually dual, above all).
    pub(super) fn trace_pairs_multiplicity_free(
        &self,
        pairs: &[(usize, usize)],
    ) -> Result<Self, Error> {
        let _host_pool = self.runtime.enter_host_pool();
        let rank = self.rank();
        let Some(TracePairAxes {
            output_axes,
            destination_codomain_rank,
            trace_lhs,
            trace_rhs,
        }) = trace_pair_axes(rank, self.codomain_rank(), pairs)?
        else {
            return Ok(self.clone());
        };
        let mapped_output_axes;
        let mapped_trace_lhs;
        let mapped_trace_rhs;
        let (source_space, source_data, axes) = match &self.repr {
            TypedTensorRepr::Owned(body) => (
                &body.space,
                match &*body.data {
                    TypedData::Dense(data) => Some(data.as_slice()),
                    TypedData::Diagonal(_) => None,
                },
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
                    Some(view.parent_data()),
                    tenet_tensors::TensorTraceAxisSpec::new_with_conjugation(
                        &mapped_output_axes,
                        &mapped_trace_lhs,
                        &mapped_trace_rhs,
                        true,
                    ),
                )
            }
        };
        // Preflight first: the checked homspace selection must fail before any
        // destination layout is derived, so a rejected trace publishes no state.
        let homspace = tenet_tensors::tensortrace_fusion_dyn_selected_homspace_checked(
            source_space,
            axes,
            destination_codomain_rank,
        )?;
        let space = source_space.derive_from_final_homspace(homspace)?;
        // Compact arm (#604): the full trace of a rank-(1,1) spectrum factor
        // over its only pair is a reduction of the stored spectrum, so there
        // is nothing to materialize. This is
        // the *categorical* trace, not `tr()`'s — the engine's
        // `trace_channel_factor` carries the quantum dimension of the traced
        // channel and, exactly where the traced leg is *not* dual, its
        // fermionic twist, which is what makes this the supertrace for a
        // fermionic rule and the coefficient `dim(c) · θ(c)` rather than
        // `tr()`'s unconditional `dim(c)`. The guard is this narrow because
        // with one pair and rank two the destination is the empty tree, so the
        // traced channel is a single uncoupled sector and the coefficient
        // collapses to a per-sector scalar; any wider geometry leaves an open
        // destination tree whose recoupling is not a per-sector scaling.
        // Today the geometric conditions are implied by the Group 4 contract
        // (`TypedData::Diagonal` lives on bond spaces only), so they are
        // defensive, not a reachable branch. A
        // lazy dense adjoint has no compact spectrum and therefore goes through
        // the parent-oriented trace seam; a compact adjoint remains an owned
        // compact tensor. The coefficient is pinned against the engine route
        // by the `compact_full_trace_*` oracle sweeps in
        // `tests/typed_facade.rs`.
        if let Some(spectrum) = self.spectrum() {
            if rank == 2 && self.codomain_rank() == 1 && pairs.len() == 1 {
                // The dense arm's compile rejects these; a spectrum reduction
                // must not answer where the categorical trace is undefined
                // (TensorKit `trace_permute!` requires symmetric braiding).
                if !self.provider().braiding_style().is_symmetric() {
                    return Err(
                        tenet_tensors::OperationError::UnsupportedTensorContractScope {
                            message: tenet_tensors::FUSION_TENSORTRACE_REQUIRES_SYMMETRIC_BRAIDING,
                        }
                        .into(),
                    );
                }
                let traced_leg_is_dual: bool =
                    self.logical_space().space().homspace().codomain().legs()[0].is_dual();
                let provider: &R = self.logical_space().provider();
                // Accumulated in `Complex64` and narrowed once through the
                // #568 `UserScalar` surface, with the same per-sector reduction
                // order as compact `tr`. The typed spectrum already
                // stores `SectorSpectrum<D>`, and the coefficient is the
                // provider's real scalar, so the result is a plain `D`.
                let mut total: num_complex::Complex64 = num_complex::Complex64::new(0.0, 0.0);
                for entry in spectrum {
                    let coefficient: f64 = if traced_leg_is_dual {
                        provider.dim_scalar(entry.sector)
                    } else {
                        provider.dim_scalar(entry.sector) * provider.twist_scalar(entry.sector)
                    };
                    let mut partial = D::Wide::from_real(0.0);
                    for &value in &entry.values {
                        partial = partial + value.widen();
                    }
                    total += partial.widen_complex() * coefficient;
                }
                // A fully traced rank-(1,1) destination is one scalar.
                if space.space().required_len()? != 1 {
                    return Err(internal_layout_error(
                        "a fully traced rank-one destination is not a single scalar",
                    ));
                }
                let value: D = D::from_complex64(total);
                return Ok(Self {
                    runtime: self.runtime.clone(),
                    repr: owned_repr(TypedTensorBody::dense(space, vec![value])),
                });
            }
        }
        let owned_payload;
        let source_data = match source_data {
            Some(data) => data,
            None => {
                owned_payload = self
                    .owned_body()
                    .expect("owned trace input")
                    .materialized_dense_data();
                &owned_payload
            }
        };
        let data = tenet_tensors::tensortrace_fusion_dyn_owned_checked(
            &space,
            source_space,
            source_data,
            axes,
            D::from_real(1.0),
        )?;
        Ok(Self {
            runtime: self.runtime.clone(),
            repr: owned_repr(TypedTensorBody::dense(space, data)),
        })
    }
}
