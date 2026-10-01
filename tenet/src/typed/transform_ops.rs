#[allow(unused_imports)]
use super::*;

pub(super) fn tree_operation_matches_axes(
    operation: &TreeTransformOperation,
    kind: TreeTransformOperationKind,
    codomain_axes: &[usize],
    domain_axes: &[usize],
) -> bool {
    operation.kind() == kind
        && operation.codomain_permutation() == codomain_axes
        && operation.domain_permutation() == domain_axes
}

/// The braid operation of `braid`/`braid_into`, with the facade's one own
/// check: one level per source axis.
pub(super) fn braid_operation(
    rank: usize,
    codomain_rank: usize,
    codomain_axes: &[usize],
    domain_axes: &[usize],
    levels: &[usize],
) -> Result<TreeTransformOperation, Error> {
    if levels.len() != rank {
        return Err(Error::InvalidArgument(format!(
            "braid levels must list one level per source axis \
             (expected {rank}, got {})",
            levels.len()
        )));
    }
    Ok(TreeTransformOperation::braid(
        codomain_axes.iter().copied(),
        domain_axes.iter().copied(),
        levels[..codomain_rank].iter().copied(),
        levels[codomain_rank..].iter().copied(),
    ))
}

impl<R, D> TensorMap<R, D>
where
    R: TypedSectorAdmission<Error = FusionAlgebraError, Mode = MultiplicityFreeAdmissionMode>
        + MultiplicityFreeRigidSymbols<Scalar = f64>
        + CheckedFusionAlgebra
        + SectorCodec,
    D: TensorScalar,
{
    /// `destination = alpha * self.permute(codomain_axes, domain_axes) +
    /// beta * destination`, TensorKit `permute!(tdst, tsrc, p, α, β)`, without
    /// replacing the destination's provider, space, body or host allocation.
    ///
    /// `beta` rides the add that writes each destination block, so it is
    /// applied once per element and never as a separate pass. `beta == 0`
    /// (IEEE, `-0.0` too) never reads `destination`: a NaN there does not
    /// survive, and the blocks the transform does not reach become `+0`.
    /// `beta == 1` leaves those blocks untouched. `alpha == 0` does not read
    /// the source (VectorInterface's `scale(x, 0) = zero(x)`).
    ///
    /// # Errors
    ///
    /// [`Error::RuntimeMismatch`], [`Error::RuleMismatch`], then
    /// [`Error::InvalidArgument`] for a lazy-adjoint or compact source, a
    /// destination that is not owned dense host storage, one that aliases
    /// the source, or one whose space, block layout or length does not match
    /// the result; [`Error::DestinationShared`] when `destination` shares its
    /// storage with a clone. Plan-construction failures count as validation.
    ///
    /// # Failure
    ///
    /// A validation or capability error leaves `destination` bit-identical:
    /// every check completes before the first write. On failure after
    /// validation (a backend or other execution error) the destination's
    /// contents are unspecified, while its space and block structure stay
    /// intact.
    pub fn permute_into(
        &self,
        codomain_axes: &[usize],
        domain_axes: &[usize],
        destination: &mut Self,
        alpha: D,
        beta: D,
    ) -> Result<(), Error> {
        self.tree_transform_into(
            destination,
            alpha,
            beta,
            |_, _| {
                Ok(TreeTransformOperation::permute(
                    codomain_axes.iter().copied(),
                    domain_axes.iter().copied(),
                ))
            },
            |operation| {
                tree_operation_matches_axes(
                    operation,
                    TreeTransformOperationKind::Permute,
                    codomain_axes,
                    domain_axes,
                )
            },
        )
    }

    /// `destination = alpha * self.braid(codomain_axes, domain_axes, levels) +
    /// beta * destination`, TensorKit `braid!(tdst, tsrc, p, levels, α, β)`.
    /// `levels` must list one level per source axis, as for
    /// [`Self::braid`]; otherwise destination rules and errors are
    /// [`Self::permute_into`]'s.
    ///
    /// # Failure
    ///
    /// A validation or capability error leaves `destination` bit-identical:
    /// every check completes before the first write. On failure after
    /// validation (a backend or other execution error) the destination's
    /// contents are unspecified, while its space and block structure stay
    /// intact.
    pub fn braid_into(
        &self,
        codomain_axes: &[usize],
        domain_axes: &[usize],
        levels: &[usize],
        destination: &mut Self,
        alpha: D,
        beta: D,
    ) -> Result<(), Error> {
        let operation = braid_operation(
            self.rank(),
            self.codomain_rank(),
            codomain_axes,
            domain_axes,
            levels,
        )?;
        let expected = operation.clone();
        self.tree_transform_into(
            destination,
            alpha,
            beta,
            |_, _| Ok(operation),
            |admitted| *admitted == expected,
        )
    }

    /// `destination = alpha * self.transpose(codomain_axes, domain_axes) +
    /// beta * destination`, TensorKit `transpose!(tdst, tsrc, p, α, β)`.
    /// Destination rules and errors are [`Self::permute_into`]'s.
    ///
    /// # Failure
    ///
    /// A validation or capability error leaves `destination` bit-identical:
    /// every check completes before the first write. On failure after
    /// validation (a backend or other execution error) the destination's
    /// contents are unspecified, while its space and block structure stay
    /// intact.
    pub fn transpose_into(
        &self,
        codomain_axes: &[usize],
        domain_axes: &[usize],
        destination: &mut Self,
        alpha: D,
        beta: D,
    ) -> Result<(), Error> {
        self.tree_transform_into(
            destination,
            alpha,
            beta,
            |source, _| {
                with_planar_axes(
                    source.codomain_rank(),
                    source.rank(),
                    PlanarRequestKind::Explicit {
                        codomain_axes,
                        domain_axes,
                    },
                    |codomain_axes, domain_axes| {
                        Ok(TreeTransformOperation::transpose(
                            codomain_axes.iter().copied(),
                            domain_axes.iter().copied(),
                        ))
                    },
                )
            },
            |operation| {
                tree_operation_matches_axes(
                    operation,
                    TreeTransformOperationKind::Transpose,
                    codomain_axes,
                    domain_axes,
                )
            },
        )
    }

    /// `destination = alpha * self.repartition(destination.codomain_rank()) +
    /// beta * destination`, TensorKit `repartition!(tdst, tsrc, α, β)`: the
    /// target split is the destination's own. Destination rules and errors
    /// are [`Self::permute_into`]'s.
    ///
    /// # Failure
    ///
    /// A validation or capability error leaves `destination` bit-identical:
    /// every check completes before the first write. On failure after
    /// validation (a backend or other execution error) the destination's
    /// contents are unspecified, while its space and block structure stay
    /// intact.
    pub fn repartition_into(&self, destination: &mut Self, alpha: D, beta: D) -> Result<(), Error> {
        let source_codomain_rank = self.codomain_rank();
        let source_rank = self.rank();
        let destination_codomain_rank = destination.codomain_rank();
        self.tree_transform_into(
            destination,
            alpha,
            beta,
            |source, destination| {
                if destination.rank() != source.rank() {
                    return Err(Error::InvalidArgument(format!(
                        "repartition destination rank {} does not match source rank {}",
                        destination.rank(),
                        source.rank()
                    )));
                }
                with_planar_axes(
                    source.codomain_rank(),
                    source.rank(),
                    PlanarRequestKind::Repartition {
                        num_codomain: destination.codomain_rank(),
                    },
                    |codomain_axes, domain_axes| {
                        Ok(TreeTransformOperation::transpose(
                            codomain_axes.iter().copied(),
                            domain_axes.iter().copied(),
                        ))
                    },
                )
            },
            |operation| {
                let planar_axis = |position: usize| {
                    if position < source_codomain_rank {
                        position
                    } else {
                        source_rank - 1 - (position - source_codomain_rank)
                    }
                };
                operation.kind() == TreeTransformOperationKind::Transpose
                    && operation
                        .codomain_permutation()
                        .iter()
                        .copied()
                        .eq((0..destination_codomain_rank).map(planar_axis))
                    && operation
                        .domain_permutation()
                        .iter()
                        .copied()
                        .eq((destination_codomain_rank..source_rank)
                            .rev()
                            .map(planar_axis))
            },
        )
    }

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

    /// One admission and replay boundary for every typed Host
    /// beta-accumulating tree transform.
    fn tree_transform_into(
        &self,
        destination: &mut Self,
        alpha: D,
        beta: D,
        operation: impl FnOnce(&Self, &Self) -> Result<TreeTransformOperation, Error>,
        admitted_operation_matches: impl FnMut(&TreeTransformOperation) -> bool,
    ) -> Result<(), Error> {
        if !self.runtime.same_runtime(&destination.runtime) {
            return Err(Error::RuntimeMismatch);
        }
        let identity = TypedSectorAdmission::typed_rule_identity(self.provider());
        if identity != TypedSectorAdmission::typed_rule_identity(destination.provider()) {
            return Err(Error::RuleMismatch);
        }

        let source_body = match &self.repr {
            TypedTensorRepr::Owned(body) if matches!(body.data.as_ref(), TypedData::Dense(_)) => {
                body
            }
            _ => {
                return Err(Error::InvalidArgument(
                    "typed destination tree transform requires an ordinary dense host source"
                        .to_string(),
                ))
            }
        };
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
        if Arc::ptr_eq(&source_body.data, &destination_body.data) {
            return Err(Error::InvalidArgument(
                "destination storage must not alias an input".to_string(),
            ));
        }

        let admitted_operation = self.runtime.admitted_tree_pair_operation(
            &identity,
            &source_body.space,
            &destination_body.space,
            admitted_operation_matches,
        );
        let exact_layout_admitted = admitted_operation.is_some();
        let operation = match admitted_operation {
            Some(operation) => operation,
            None => operation(self, destination)?,
        };
        if !exact_layout_admitted {
            let expected = source_body
                .space
                .transformed_multiplicity_free(&operation)?;
            if destination_body.space.space() != expected.space() {
                return Err(Error::InvalidArgument(
                    "destination fusion space or block layout does not match the operation result"
                        .to_string(),
                ));
            }
        }
        let required = destination_body.space.space().required_len()?;
        let actual = match destination_body.data.as_ref() {
            TypedData::Dense(data) => data.len(),
            TypedData::Diagonal(_) => unreachable!("dense destination checked above"),
        };
        if actual != required {
            return Err(Error::InvalidArgument(format!(
                "destination storage length {actual} does not match required length {required}"
            )));
        }
        if Arc::strong_count(destination_body) != 1
            || Arc::strong_count(&destination_body.data) != 1
        {
            return Err(Error::DestinationShared);
        }

        let source_structure = source_body.space.space().structure();
        let source_data = match source_body.data.as_ref() {
            TypedData::Dense(data) => data.as_slice(),
            TypedData::Diagonal(_) => unreachable!("dense source checked above"),
        };
        {
            let mut lease = self.runtime.lease_context()?;
            let context = lease
                .context()
                .multiplicity_free_lane::<D>()?
                .tree_context_mut();
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
            context.tree_transform_dyn_into_ref(
                destination_provider,
                &operation,
                destination_structure,
                source_structure,
                destination_data.as_mut_slice(),
                source_data,
                alpha,
                beta,
            )?;
        }
        if !exact_layout_admitted {
            self.runtime.admit_exact_tree_pair_layout(
                identity,
                &operation,
                &source_body.space,
                destination.logical_space(),
            );
        }
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
    R::Mode: TypedTensorTransformDispatch<R, D>,
    D: TensorScalar,
{
    /// TensorKit `permute`: re-arranges legs with symmetric braiding.
    ///
    /// `codomain_axes` and `domain_axes` list source axis numbers (`0..rank`,
    /// codomain axes first) for the new codomain and domain.
    ///
    /// # Compact storage
    ///
    /// An exact identity returns a compact diagonal factor unchanged. A
    /// rank-(1,1) leg swap through [`Self::permute`] or [`Self::transpose`]
    /// keeps it compact. An admitted rank-(1,1) [`Self::braid`] reads its
    /// spectrum directly and publishes a dense result. Other non-identity
    /// `permute`/`transpose`/`repartition` cases and unadmitted braids
    /// materialize the compact source into an operation-local buffer and
    /// publish a dense `Σ_c k_c²` result.
    ///
    /// # Errors
    ///
    /// [`Error::Operation`] / [`Error::Core`] / [`Error::FusionAlgebra`] when
    /// the axis lists are malformed (out of range, repeated, or not a
    /// partition of `0..rank`) or the provider cannot support the braiding the
    /// requested motion needs. The expert layer's own typed errors are the
    /// contract here: re-validating the axes at this layer would be a second
    /// copy of a rule that already exists one call down, free to drift.
    /// Checked Generic providers return [`GenericTensorError::Plan`] with the
    /// concrete provider error preserved as its source.
    ///
    /// ```
    /// use std::sync::Arc;
    ///
    /// use tenet::sector::{U1FusionRule, U1Irrep};
    /// use tenet::typed::{GradedSpace, Runtime, TensorMap};
    ///
    /// let runtime = Runtime::builder().build()?;
    /// let v = GradedSpace::try_new(
    ///     Arc::new(U1FusionRule),
    ///     [(U1Irrep::new(0), 1), (U1Irrep::new(1), 2)],
    /// )?;
    /// let w = GradedSpace::try_new(
    ///     Arc::new(U1FusionRule),
    ///     [(U1Irrep::new(0), 1), (U1Irrep::new(1), 1)],
    /// )?;
    /// let t: TensorMap<_, f64> = TensorMap::rand_with_seed(&runtime, [&v], [&w], 7)?;
    /// assert_eq!(t.leg_dims()?, [3, 2]);
    ///
    /// let swapped = t.permute(&[1], &[0])?;
    /// assert_eq!(swapped.leg_dims()?, [2, 3]);
    /// // A bosonic two-leg swap is an involution: swapping back restores the
    /// // payload exactly.
    /// assert_eq!(swapped.permute(&[1], &[0])?.dense_data()?, t.dense_data()?);
    /// # Ok::<(), tenet::typed::Error>(())
    /// ```
    pub fn permute(
        &self,
        codomain_axes: &[usize],
        domain_axes: &[usize],
    ) -> Result<Self, TypedFacadeError<R>> {
        if self.axes_are_identity(codomain_axes, domain_axes) {
            return Ok(self.clone());
        }
        <R::Mode as TypedTensorTransformDispatch<R, D>>::tree_transform(
            self,
            TreeTransformOperation::permute(
                codomain_axes.iter().copied(),
                domain_axes.iter().copied(),
            ),
        )
    }

    /// Runs `op` on the matrix view `permute(self, rows, cols)`: the leg roles
    /// of every factorization and matrix function.
    ///
    /// The current split borrows `self`, so it costs no transform and no
    /// clone. Any other split is one [`Self::permute`], whose typed errors
    /// (malformed axes, non-symmetric braiding) are returned before `op` runs.
    pub(super) fn with_leg_roles<T>(
        &self,
        rows: &[usize],
        cols: &[usize],
        op: impl FnOnce(&Self) -> Result<T, TypedFacadeError<R>>,
    ) -> Result<T, TypedFacadeError<R>> {
        if self.axes_are_identity(rows, cols) {
            return op(self);
        }
        op(&self.permute(rows, cols)?)
    }

    /// TensorKit `braid`: re-arranges legs with an explicit braid, one level
    /// per source axis.
    ///
    /// `codomain_axes` and `domain_axes` name source axes exactly as for
    /// [`Self::permute`]. `levels` is per source *strand*, one entry for every
    /// axis in `0..rank` — codomain axes first — and it is split by the
    /// **source** codomain rank, so entry `i` always describes source axis `i`
    /// regardless of where that axis ends up. The levels decide which strand
    /// crosses above at each transposition; for a symmetric (bosonic) braiding
    /// they cannot change the result, and this is then [`Self::permute`].
    ///
    /// A compact rank-(1,1) diagonal is read directly when the compiled braid
    /// has one term per source block, its coefficient remains finite and
    /// nonzero in the payload dtype, and destination coverage is complete;
    /// otherwise it uses the ordinary dense replay.
    ///
    /// # Errors
    ///
    /// [`Error::InvalidArgument`] when `levels` does not have one entry per
    /// source axis — the one check this facade makes itself, because the axis
    /// lists and the levels are validated by different layers and a
    /// mis-lengthed `levels` would otherwise be split silently.
    ///
    /// Otherwise [`Error::Operation`] / [`Error::Core`] /
    /// [`Error::FusionAlgebra`] straight from the expert layer for malformed
    /// axis lists or a provider that cannot support the requested braiding.
    /// As for [`Self::permute`], those errors are the contract; this layer does
    /// not re-validate axes.
    /// Checked Generic failures use [`GenericTensorError::Plan`].
    pub fn braid(
        &self,
        codomain_axes: &[usize],
        domain_axes: &[usize],
        levels: &[usize],
    ) -> Result<Self, TypedFacadeError<R>> {
        // Keep the established axis/level pre-check and diagnostic.
        let rank = self.rank();
        if levels.len() != rank {
            return Err(Error::InvalidArgument(format!(
                "braid levels must list one level per source axis \
                 (expected {rank}, got {})",
                levels.len()
            ))
            .into());
        }
        if self.axes_are_identity(codomain_axes, domain_axes) {
            return Ok(self.clone());
        }
        let nout = self.codomain_rank();
        <R::Mode as TypedTensorTransformDispatch<R, D>>::tree_transform(
            self,
            TreeTransformOperation::braid(
                codomain_axes.iter().copied(),
                domain_axes.iter().copied(),
                levels[..nout].iter().copied(),
                levels[nout..].iter().copied(),
            ),
        )
    }

    /// TensorKit `repartition(t, N₁, N₂)`: moves the planar boundary so the
    /// codomain holds `num_codomain` legs and the domain holds the rest.
    ///
    /// The planar order — codomain followed by reversed domain — is preserved;
    /// legs that cross the boundary are bent, and so arrive with their dual
    /// flag flipped and their sectors dualized, without any braid being
    /// introduced. The identity repartition returns its input unchanged; a
    /// compact diagonal whose boundary moves is densified into an
    /// operation-local buffer first.
    ///
    /// # Errors
    ///
    /// [`Error::InvalidArgument`] when `num_codomain` exceeds the rank, and
    /// otherwise [`Error::Operation`] / [`Error::Core`] /
    /// [`Error::FusionAlgebra`] from the expert layer, which owns the
    /// validation this facade passes through. Checked Generic failures use
    /// [`GenericTensorError::Plan`].
    pub fn repartition(&self, num_codomain: usize) -> Result<Self, TypedFacadeError<R>> {
        if num_codomain == self.codomain_rank() {
            return Ok(self.clone());
        }
        self.planar(PlanarRequestKind::Repartition { num_codomain })
    }

    /// TensorKit `transpose(t, (p₁, p₂))`: the planar transpose with an
    /// explicit cyclic axis map.
    ///
    /// `codomain_axes` and `domain_axes` are flat source axis numbers
    /// (`0..rank`, codomain axes first), exactly as for [`Self::permute`], but
    /// together they must describe one **cyclic rotation** of the planar source
    /// order (codomain axes followed by the domain axes reversed).
    ///
    /// Planar means it **never braids**: legs are bent across the boundary, and
    /// bending conjugates them, so legs that cross it carry flipped dual flags.
    /// Spelling this as a [`Self::permute`] of the same axis order would be
    /// wrong for any provider whose braiding is not symmetric — the two agree
    /// only up to the R-symbols a permute inserts and this does not.
    ///
    /// TensorKit's argument-free `transpose(t)` is the full rotation
    /// `codomain_axes = (nout..rank).rev()`, `domain_axes = (0..nout).rev()`,
    /// which carries every codomain leg across the boundary and every domain
    /// leg back; it is its own inverse. There is no argument-free overload:
    /// the axes are the operation's leg roles and are always stated.
    /// [`Self::repartition`] is the other planar parameterization, by the
    /// target codomain rank.
    ///
    /// # Errors
    ///
    /// [`Error::Operation`] / [`Error::Core`] / [`Error::FusionAlgebra`] when
    /// the axis lists are malformed or are not a cyclic rotation of the planar
    /// order — a re-arrangement that would need a braid is refused rather than
    /// silently braided. As everywhere in this
    /// facade the expert layer owns that validation; it is not repeated here.
    pub fn transpose(
        &self,
        codomain_axes: &[usize],
        domain_axes: &[usize],
    ) -> Result<Self, TypedFacadeError<R>> {
        if self.axes_are_identity(codomain_axes, domain_axes) {
            return Ok(self.clone());
        }
        self.planar(PlanarRequestKind::Explicit {
            codomain_axes,
            domain_axes,
        })
    }

    /// Shared body of the three planar operations: derive the planar axis
    /// order, let the expert layer check it, and run it as a transpose.
    ///
    /// The shared axis derivation defines what "planar" means for each request
    /// kind; duplicating it here would allow the definitions to drift.
    fn planar(&self, kind: PlanarRequestKind<'_>) -> Result<Self, TypedFacadeError<R>> {
        let operation = with_planar_axes(
            self.codomain_rank(),
            self.rank(),
            kind,
            |codomain_axes, domain_axes| {
                // Why `transpose` and not `permute` even when the axes happen
                // to be a plain permutation: domain trees run opposite to the
                // planar boundary, so flattening them into a permute would
                // braid a different leg across it.
                Ok(TreeTransformOperation::transpose(
                    codomain_axes.iter().copied(),
                    domain_axes.iter().copied(),
                ))
            },
        )
        .map_err(TypedFacadeError::<R>::from)?;
        <R::Mode as TypedTensorTransformDispatch<R, D>>::tree_transform(self, operation)
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

#[allow(private_bounds)]
impl<R, D> TensorMap<R, D>
where
    R: MultiplicityFreeRigidSymbols + CheckedFusionAlgebra + SectorCodec,
    <R as MultiplicityFreeFusionSymbols>::Scalar:
        CategoricalScalar + tenet_tensors::DenseRecouplingScalar,
    D: TensorScalar
        + crate::runtime::MultiplicityFreeCoefficientLane<
            <R as MultiplicityFreeFusionSymbols>::Scalar,
        >,
{
    fn tree_transform_multiplicity_free_owned(
        &self,
        operation: TreeTransformOperation,
    ) -> Result<Self, Error> {
        // Leasing rather than locking: independent operations on one runtime
        // must not serialize behind each other.
        let mut lease = self.runtime.lease_context()?;
        let body = self.owned_body().expect("owned tree transform input");
        let (space, data) = tree_transform_owned_multiplicity_free(
            D::lane(lease.context())?,
            BoundDynamicTensorRef::try_new(&body.space, body.materialized_dense_data().as_ref())?,
            operation,
        )?;
        Ok(Self {
            runtime: self.runtime.clone(),
            repr: owned_repr(TypedTensorBody::dense(space, data)),
        })
    }
}

impl<R, D> TensorMap<R, D>
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64> + CheckedFusionAlgebra + SectorCodec,
    D: TensorScalar,
{
    /// Runs one prepared real-coefficient tree transform, retaining the
    /// established compact and lazy-adjoint fast paths.
    fn tree_transform_multiplicity_free_real(
        &self,
        operation: TreeTransformOperation,
    ) -> Result<Self, Error> {
        if let Some(spectrum) = self.spectrum() {
            if crate::tensor_core::is_rank_one_diagonal_swap(
                self.codomain_rank(),
                self.rank() - self.codomain_rank(),
                &operation,
            ) {
                let destination = self
                    .logical_space()
                    .transformed_multiplicity_free(&operation)?;
                let transformed = crate::tensor_core::transform_rank_one_diagonal_spectrum(
                    self.logical_space().provider(),
                    self.logical_space().space(),
                    destination.space(),
                    &operation,
                    spectrum,
                )?;
                return Ok(self.with_spectrum_on(destination, transformed));
            }
            if crate::tensor_core::is_rank_one_diagonal_braid(
                self.codomain_rank(),
                self.rank() - self.codomain_rank(),
                &operation,
            ) {
                if let Ok(destination) = self
                    .logical_space()
                    .transformed_multiplicity_free(&operation)
                {
                    let compiled = {
                        let mut lease = self.runtime.lease_context()?;
                        lease
                            .context()
                            .multiplicity_free_lane::<D>()?
                            .tree_context_mut()
                            .compile_tree_pair_structure(
                                self.logical_space().provider(),
                                &operation,
                                destination.space().structure(),
                                self.logical_space().space().structure(),
                            )
                            .ok()
                    };
                    if let Some(compiled) = compiled {
                        if let Some(data) = crate::tensor_core::try_braid_rank_one_diagonal_data(
                            self.logical_space().space(),
                            destination.space(),
                            &compiled,
                            spectrum,
                        ) {
                            return Ok(Self {
                                runtime: self.runtime.clone(),
                                repr: owned_repr(TypedTensorBody::dense(destination, data)),
                            });
                        }
                    }
                }
            }
        }
        if let TypedTensorRepr::Adjoint(view) = &self.repr {
            let parent_space = view.parent.space.space();
            let lowered = lower_adjoint_tree_transform_operation(
                parent_space.nout(),
                parent_space.nin(),
                &operation,
            )?;
            let parent = Self {
                runtime: self.runtime.clone(),
                repr: TypedTensorRepr::Owned(Arc::clone(&view.parent)),
            };
            return parent
                .tree_transform_multiplicity_free_real(lowered)?
                .adjoint();
        }
        let mut lease = self.runtime.lease_context()?;
        let body = self.owned_body().expect("owned tree transform input");
        let (space, data) = tree_transform_owned_multiplicity_free(
            lease.context().multiplicity_free_lane::<D>()?,
            BoundDynamicTensorRef::try_new(&body.space, body.materialized_dense_data().as_ref())?,
            operation,
        )?;
        Ok(Self {
            runtime: self.runtime.clone(),
            repr: owned_repr(TypedTensorBody::dense(space, data)),
        })
    }
}

impl<R> TensorMap<R, num_complex::Complex64>
where
    R: MultiplicityFreeRigidSymbols<Scalar = num_complex::Complex64>
        + CheckedFusionAlgebra
        + SectorCodec,
{
    fn tree_transform_multiplicity_free_complex(
        &self,
        operation: TreeTransformOperation,
    ) -> Result<Self, Error> {
        self.materialized_tensor_uncached()?
            .tree_transform_multiplicity_free_owned(operation)
    }
}

impl<R, D> MultiplicityFreeTransformExecution<R, f64> for D
where
    R: TypedSectorAdmission
        + MultiplicityFreeRigidSymbols<Scalar = f64>
        + CheckedFusionAlgebra
        + SectorCodec,
    D: TensorScalar,
{
    fn execute(
        tensor: &TensorMap<R, Self>,
        operation: TreeTransformOperation,
    ) -> Result<TensorMap<R, Self>, Error> {
        tensor.tree_transform_multiplicity_free_real(operation)
    }
}

impl<R> MultiplicityFreeTransformExecution<R, num_complex::Complex64> for num_complex::Complex64
where
    R: TypedSectorAdmission
        + MultiplicityFreeRigidSymbols<Scalar = num_complex::Complex64>
        + CheckedFusionAlgebra
        + SectorCodec,
{
    fn execute(
        tensor: &TensorMap<R, Self>,
        operation: TreeTransformOperation,
    ) -> Result<TensorMap<R, Self>, Error> {
        tensor.tree_transform_multiplicity_free_complex(operation)
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
        tenet_matrixalgebra::scale_axis_by_spectrum_mapped(
            self.logical_space().space(),
            &mut data,
            axis,
            spectrum,
            |value| value,
        )?;
        Ok(self.with_data(data))
    }

    /// Wraps one factor the matrix-algebra seam produced into a typed tensor
    /// map. `BoundDynFactor::into_parts` hands back exactly the pair
    /// [`TypedTensorBody`] stores, so there is nothing to validate here — the
    /// seam already certified the space against its own data.
    fn wrap_bound_factor(&self, factor: BoundDynFactor<R, D>) -> Self {
        wrap_factor_on(&self.runtime, factor)
    }

    /// Wraps a seam spectrum as a factor in compact diagonal storage: the bond
    /// space is derived from the spectrum itself, but the payload stays the
    /// `Σ_c k_c` values rather than the `Σ_c k_c²` block-diagonal buffer they
    /// would fill (TensorKit's `DiagonalTensorMap`).
    ///
    /// The spectrum is stored raw — engine [`crate::sector::SectorId`]s, values in
    /// the payload dtype `D`. Decoding belongs to the caller-facing spectrum
    /// fields, not to storage; a stored payload never leaves this module.
    ///
    /// Sorted by sector id first because the bond leg is built from this order.
    fn diagonal_factor<V: Copy>(
        &self,
        spectrum: &mut [tenet_matrixalgebra::SectorSpectrum<V>],
        to_scalar: impl Fn(V) -> D,
    ) -> Result<Self, Error> {
        diagonal_factor_on(&self.runtime, self.logical_space(), spectrum, to_scalar)
    }

    /// Decodes a seam spectrum into provider labels and sorts it by label.
    ///
    /// Every id here came out of the engine's own coupled-sector enumeration,
    /// so a decode failure is the provider breaking [`SectorCodec`]'s
    /// decode-totality law — same contract as [`decode_block_fusion_trees`].
    fn decode_spectrum<V>(
        &self,
        raw: Vec<tenet_matrixalgebra::SectorSpectrum<V>>,
    ) -> Result<Vec<SectorSpectrum<R::Sector, V>>, Error> {
        let provider = self.logical_space().provider();
        let mut decoded: Vec<SectorSpectrum<R::Sector, V>> = raw
            .into_iter()
            .map(|entry| {
                Ok(SectorSpectrum {
                    sector: provider.decode_sector(entry.sector)?,
                    values: entry.values,
                })
            })
            .collect::<Result<_, Error>>()?;
        // Public label order, not the engine's opaque sector-id order.
        decoded.sort_by(|left, right| left.sector.cmp(&right.sector));
        Ok(decoded)
    }

    /// The bound space and dense payload of this owned tensor map; a compact
    /// diagonal is densified operation-locally.
    #[allow(clippy::type_complexity)]
    fn bound_payload(
        &self,
    ) -> Result<(&BoundDynamicFusionMapSpace<R>, std::borrow::Cow<'_, [D]>), Error> {
        let body = self.owned_body().ok_or_else(|| {
            internal_layout_error("factorization input must be owned after adjoint dispatch")
        })?;
        Ok((&body.space, body.materialized_dense_data()))
    }

    /// TensorKit 0.17 / MatrixAlgebraKit `svd_compact`: `t = u * s * vh` with
    /// the bond `min(rows, cols)` per coupled sector.
    ///
    /// Returns an [`Svd`] with `u : codomain <- bond`, `s : bond <- bond`
    /// and `vh : bond <- domain`.
    ///
    /// # Storage
    ///
    /// `s` is held in compact diagonal storage — `Σ_c k_c` values, not the
    /// `Σ_c k_c²` block-diagonal buffer — matching the `DiagonalTensorMap`
    /// TensorKit's own `svd_compact` returns. A downstream `u.compose(&s)` or
    /// `s.compose(&vh)` takes the O(d·n) bond-scaling path rather than a dense
    /// GEMM. [`Self::materialize`] builds the dense buffer on request; a
    /// caller who only needs the values should reach for
    /// [`Self::svd_vals`], which builds no factor at all.
    /// An owned compact-diagonal input with representable magnitudes is sorted
    /// directly by sector:
    /// no dense input or dense SVD is needed. The dense `u` and `vh` permutation
    /// factors still require `Σ_c k_c²` storage and writes; sorting costs
    /// `O(Σ_c k_c log k_c)`. Nonfinite or unrepresentable spectra retain the
    /// dense solver's error behavior.
    ///
    /// # Errors
    ///
    /// [`Error::Operation`] / [`Error::Core`] / [`Error::FusionAlgebra`]
    /// straight from the matrix-algebra seam. As everywhere in this facade
    /// there are no pre-checks here: the seam owns the rules, and a second copy
    /// would be free to drift.
    ///
    /// ```
    /// use std::sync::Arc;
    ///
    /// use tenet::sector::{U1FusionRule, U1Irrep};
    /// use tenet::typed::{GradedSpace, Runtime, Svd, TensorMap};
    ///
    /// let runtime = Runtime::builder().build()?;
    /// let v = GradedSpace::try_new(
    ///     Arc::new(U1FusionRule),
    ///     [(U1Irrep::new(0), 2), (U1Irrep::new(1), 2)],
    /// )?;
    /// let t: TensorMap<_, f64> = TensorMap::rand_with_seed(&runtime, [&v], [&v], 9)?;
    ///
    /// let Svd { u, s, vh } = t.svd_compact(&[0], &[1])?;
    /// // `u` is an isometry: `u† ∘ u` is the identity on its domain.
    /// let gram = u.adjoint()?.compose(&u)?;
    /// let identity = TensorMap::isomorphism(&runtime, &u.domain(), &u.domain())?;
    /// assert!(gram.axpby(1.0, &identity, -1.0)?.norm(2.0)? <= 1e-12 * gram.norm(2.0)?.max(1.0));
    /// let rebuilt = u.compose(&s)?.compose(&vh)?;
    /// let max_err = rebuilt
    ///     .dense_data()?
    ///     .iter()
    ///     .zip(t.dense_data()?)
    ///     .map(|(a, b)| (a - b).abs())
    ///     .fold(0.0f64, f64::max);
    /// assert!(max_err < 1e-12);
    /// # Ok::<(), tenet::typed::Error>(())
    /// ```
    pub(super) fn svd_compact_multiplicity_free(&self) -> Result<Svd<Self>, Error>
    where
        D: FactorizationScalar,
    {
        let compact = match &self.repr {
            TypedTensorRepr::Owned(body) => match body.data.as_ref() {
                TypedData::Diagonal(spectrum) => {
                    tenet_matrixalgebra::svd_compact_diagonal_factors_dyn(&body.space, spectrum)?
                }
                TypedData::Dense(_) => None,
            },
            TypedTensorRepr::Adjoint(_) => None,
        };
        let (u, vh, mut spectrum) = if let Some(factors) = compact {
            factors
        } else {
            // The ordinary route keeps its dense-only lease and its compact-S
            // factor seam, including the established nonfinite error behavior.
            let mut dense = self.runtime.lease_dense();
            match &self.repr {
                TypedTensorRepr::Adjoint(view) => {
                    tenet_matrixalgebra::svd_compact_adjoint_factors_dyn(
                        dense.dense(),
                        &BoundDynamicTensorRef::try_new(&view.parent.space, view.parent_data())?,
                    )?
                }
                TypedTensorRepr::Owned(_) => {
                    let (bound_space, bound_payload) = self.bound_payload()?;
                    tenet_matrixalgebra::svd_compact_factors_dyn(
                        dense.dense(),
                        &BoundDynamicTensorRef::try_new(bound_space, &bound_payload)?,
                    )?
                }
            }
        };
        Ok(Svd {
            u: self.wrap_bound_factor(u),
            s: self.diagonal_factor(&mut spectrum, D::from_real)?,
            vh: self.wrap_bound_factor(vh),
        })
    }

    /// TensorKit 0.17 / MatrixAlgebraKit `svd_full`: `t = u * s * vh` with
    /// square unitaries and a rectangular `s` per coupled sector.
    ///
    /// Returns an [`Svd`] with `u : codomain <- W`, `s : W <- W'` and
    /// `vh : W' <- domain`.
    ///
    /// `s` is dense here where [`Self::svd_compact`]'s is diagonal, and that is
    /// TK-exact rather than a residual gap: TensorKit's own `svd_full` builds
    /// `s` as a dense rectangular tensor
    /// (`similar(t, real(scalartype(t)), V_cod <- V_dom)`). TensorKit's
    /// diagonal-`S` `svd_full!` applies to diagonal *inputs*, which is a
    /// different operation.
    ///
    /// # Errors
    ///
    /// As [`Self::svd_compact`]: the seam's own errors, unfiltered.
    pub(super) fn svd_full_multiplicity_free(&self) -> Result<Svd<Self>, Error>
    where
        D: FactorizationScalar,
    {
        let mut dense = self.runtime.lease_dense();
        let out = match &self.repr {
            TypedTensorRepr::Adjoint(view) => tenet_matrixalgebra::svd_full_adjoint_dyn(
                dense.dense(),
                &BoundDynamicTensorRef::try_new(&view.parent.space, view.parent_data())?,
            )?,
            TypedTensorRepr::Owned(_) => {
                let (bound_space, bound_payload) = self.bound_payload()?;
                tenet_matrixalgebra::svd_full_dyn(
                    dense.dense(),
                    &BoundDynamicTensorRef::try_new(bound_space, &bound_payload)?,
                )?
            }
        };
        let (u, s, vh, _) = out.into_parts();
        Ok(Svd {
            u: self.wrap_bound_factor(u),
            s: self.wrap_bound_factor(s),
            vh: self.wrap_bound_factor(vh),
        })
    }

    /// TensorKit 0.17 / MatrixAlgebraKit `svd_vals`: the singular values per
    /// coupled sector, and nothing else.
    ///
    /// No factor tensor and no bond space is built at all, so this is cheaper
    /// still than reading [`Self::svd_compact`]'s compact `s`.
    /// Finite owned compact-diagonal inputs are sorted sectorwise without
    /// materializing the input or calling a dense SVD.
    ///
    /// # Errors
    ///
    /// [`Error::Operation`] / [`Error::Core`] from the seam, plus
    /// [`Error::FusionAlgebra`] when the provider cannot decode a coupled
    /// sector its own algebra produced.
    pub(super) fn svd_vals_multiplicity_free(
        &self,
    ) -> Result<Vec<SectorSpectrum<R::Sector, f64>>, Error>
    where
        D: FactorizationScalar,
    {
        if let TypedTensorRepr::Owned(body) = &self.repr {
            if let TypedData::Diagonal(spectrum) = body.data.as_ref() {
                if let Some(raw) =
                    tenet_matrixalgebra::svd_vals_compact_diagonal_dyn(&body.space, spectrum)?
                {
                    return self.decode_spectrum(raw);
                }
            }
        }
        let mut dense = self.runtime.lease_dense();
        // Singular values and coupled-sector ids are invariant under adjoint,
        // so an oriented input or logical-payload copy cannot change this output.
        let raw = match &self.repr {
            TypedTensorRepr::Adjoint(view) => tenet_matrixalgebra::svd_vals_dyn(
                dense.dense(),
                &BoundDynamicTensorRef::try_new(&view.parent.space, view.parent_data())?,
            )?,
            TypedTensorRepr::Owned(_) => {
                let (bound_space, bound_payload) = self.bound_payload()?;
                tenet_matrixalgebra::svd_vals_dyn(
                    dense.dense(),
                    &BoundDynamicTensorRef::try_new(bound_space, &bound_payload)?,
                )?
            }
        };
        self.decode_spectrum(raw)
    }

    /// TensorKit 0.17 / MatrixAlgebraKit `qr_compact`: `t = q * r` with `q`
    /// carrying orthonormal columns per coupled sector.
    ///
    /// # Errors
    ///
    /// As [`Self::svd_compact`]: the seam's own errors, unfiltered.
    ///
    /// # Complexity
    ///
    /// `O(Σ_c n_c³)` — sectorwise cubic; the seam runs one dense QR per
    /// coupled-sector matrix. A lazy adjoint first allocates its whole logical
    /// dense payload as an operation-local owned tensor, released with the
    /// operation, and the returned factors are owned. A compact-diagonal
    /// payload (TensorKit's `DiagonalTensorMap`) is densified into an
    /// operation-local coupled buffer first, as for [`Self::left_polar`].
    /// TensorKit 0.17 *does* keep a diagonal QR compact
    /// (MatrixAlgebraKit's `DiagonalAlgorithm`); that fast path is not adopted
    /// here — the issue #613 Group 4 contract requires every compact fast path
    /// to be re-proven individually, the same deferral the polars record.
    pub(super) fn qr_compact_multiplicity_free(&self) -> Result<Qr<Self>, Error>
    where
        D: FactorizationScalar,
    {
        if matches!(&self.repr, TypedTensorRepr::Adjoint(_)) {
            return self
                .materialized_tensor_uncached()?
                .qr_compact_multiplicity_free();
        }
        let mut dense = self.runtime.lease_dense();
        let (bound_space, bound_payload) = self.bound_payload()?;
        let Qr { q, r } = tenet_matrixalgebra::qr_compact_dyn(
            dense.dense(),
            &BoundDynamicTensorRef::try_new(bound_space, &bound_payload)?,
        )?;
        Ok(Qr {
            q: self.wrap_bound_factor(q),
            r: self.wrap_bound_factor(r),
        })
    }

    /// TensorKit 0.17 / MatrixAlgebraKit `qr_full`: `t = q * r` with a square
    /// `q` per coupled sector.
    ///
    /// # Errors
    ///
    /// As [`Self::svd_compact`]: the seam's own errors, unfiltered.
    ///
    /// # Complexity
    ///
    /// For sector shape `m_c x n_c`, dense work is `O(m_c² n_c)` when
    /// `m_c <= n_c`, and `O(m_c²(n_c + m_c))` when completion is required.
    /// Source packing and owned factor publication are additional costs. A
    /// lazy adjoint also allocates its whole logical payload for the call. A
    /// compact-diagonal payload is densified into an operation-local buffer
    /// first (TensorKit's `DiagonalAlgorithm` covers `qr_full!` too —
    /// same non-adoption, same #613 Group 4 deferral).
    pub(super) fn qr_full_multiplicity_free(&self) -> Result<Qr<Self>, Error>
    where
        D: FactorizationScalar,
    {
        if matches!(&self.repr, TypedTensorRepr::Adjoint(_)) {
            return self
                .materialized_tensor_uncached()?
                .qr_full_multiplicity_free();
        }
        let mut dense = self.runtime.lease_dense();
        let (bound_space, bound_payload) = self.bound_payload()?;
        let Qr { q, r } = tenet_matrixalgebra::qr_full_dyn(
            dense.dense(),
            &BoundDynamicTensorRef::try_new(bound_space, &bound_payload)?,
        )?;
        Ok(Qr {
            q: self.wrap_bound_factor(q),
            r: self.wrap_bound_factor(r),
        })
    }

    /// TensorKit 0.17 / MatrixAlgebraKit `lq_compact`: `t = l * q` with `q`
    /// carrying orthonormal rows per coupled sector.
    ///
    /// # Errors
    ///
    /// As [`Self::svd_compact`]: the seam's own errors, unfiltered.
    ///
    /// # Complexity
    ///
    /// Sectorwise cubic. A lazy adjoint runs compact QR on its owned parent,
    /// reverses and adjoints the factors, then materializes both outputs into
    /// detached owned tensors, retaining neither parent factor buffer. A
    /// compact-diagonal payload is densified into an operation-local buffer
    /// first (TensorKit's `DiagonalAlgorithm` covers the LQ pair as well
    /// — same non-adoption, same #613 Group 4 deferral).
    pub(super) fn lq_compact_multiplicity_free(&self) -> Result<Lq<Self>, Error>
    where
        D: FactorizationScalar,
    {
        if matches!(&self.repr, TypedTensorRepr::Adjoint(_)) {
            let Qr { q, r } = self.adjoint()?.qr_compact_multiplicity_free()?;
            return Ok(Lq {
                l: r.adjoint()?.materialized_tensor_uncached()?,
                q: q.adjoint()?.materialized_tensor_uncached()?,
            });
        }
        let mut dense = self.runtime.lease_dense();
        let (bound_space, bound_payload) = self.bound_payload()?;
        let Lq { l, q } = tenet_matrixalgebra::lq_compact_dyn(
            dense.dense(),
            &BoundDynamicTensorRef::try_new(bound_space, &bound_payload)?,
        )?;
        Ok(Lq {
            l: self.wrap_bound_factor(l),
            q: self.wrap_bound_factor(q),
        })
    }

    /// TensorKit 0.17 / MatrixAlgebraKit `lq_full`: `t = l * q` with a square
    /// `q` per coupled sector.
    ///
    /// # Errors
    ///
    /// As [`Self::svd_compact`]: the seam's own errors, unfiltered.
    ///
    /// # Complexity
    ///
    /// For sector shape `m_c x n_c`, dense work is `O(n_c² m_c)` when
    /// `n_c <= m_c`, and `O(n_c²(m_c + n_c))` when completion is required.
    /// Source packing, the sectorwise adjoint, and owned factor publication are
    /// additional costs. A lazy adjoint uses the parent full-QR route and two
    /// detached owned output payloads. A compact-diagonal payload is
    /// materialized dense first.
    pub(super) fn lq_full_multiplicity_free(&self) -> Result<Lq<Self>, Error>
    where
        D: FactorizationScalar,
    {
        if matches!(&self.repr, TypedTensorRepr::Adjoint(_)) {
            let Qr { q, r } = self.adjoint()?.qr_full_multiplicity_free()?;
            return Ok(Lq {
                l: r.adjoint()?.materialized_tensor_uncached()?,
                q: q.adjoint()?.materialized_tensor_uncached()?,
            });
        }
        let mut dense = self.runtime.lease_dense();
        let (bound_space, bound_payload) = self.bound_payload()?;
        let Lq { l, q } = tenet_matrixalgebra::lq_full_dyn(
            dense.dense(),
            &BoundDynamicTensorRef::try_new(bound_space, &bound_payload)?,
        )?;
        Ok(Lq {
            l: self.wrap_bound_factor(l),
            q: self.wrap_bound_factor(q),
        })
    }

    /// TensorKit 0.17 / MatrixAlgebraKit `left_null`: `n : codomain <- W` with
    /// `n^H * t = 0`.
    ///
    /// # Null bond
    ///
    /// `W` is a fresh non-dual single-leg bond space carrying, per coupled
    /// sector `c`, the `rows_c − rank_c` null directions; `rank_c` is the
    /// numerical rank the seam takes from that sector's compact SVD, counting
    /// `σ > ε(dtype) · max(rows_c, cols_c) · σ_max,c` as nonzero. A sector
    /// with no null directions is absent from `W`, so `W` is empty for a
    /// numerically full-rank tensor. Note this is *not*
    /// TensorKit/MatrixAlgebraKit's default `left_null`, which without a
    /// truncation argument is QR-based and counts only the structural nullity
    /// `rows_c − min(rows_c, cols_c)` (MatrixAlgebraKit
    /// `interface/orthnull.jl`, the `alg::Nothing` mode); the seam's behavior
    /// corresponds to their SVD mode with a tolerance.
    ///
    /// # Errors
    ///
    /// As [`Self::svd_compact`]: the seam's own errors, unfiltered.
    ///
    /// # Complexity
    ///
    /// Sectorwise cubic — one compact SVD per coupled sector plus an
    /// orthonormal completion of the sectors that keep null directions; a
    /// compact-diagonal payload is materialized dense first, as for
    /// [`Self::qr_compact`]. A lazy adjoint runs the owned parent's
    /// [`Self::right_null`] and returns its detached adjoint, without
    /// materializing the receiver.
    pub(super) fn left_null_multiplicity_free(&self) -> Result<Self, Error>
    where
        D: FactorizationScalar,
    {
        if matches!(&self.repr, TypedTensorRepr::Adjoint(_)) {
            return self
                .adjoint()?
                .right_null_multiplicity_free()?
                .adjoint()?
                .materialized_tensor_uncached();
        }
        let mut dense = self.runtime.lease_dense();
        let (bound_space, bound_payload) = self.bound_payload()?;
        let out = tenet_matrixalgebra::left_null_dyn(
            dense.dense(),
            &BoundDynamicTensorRef::try_new(bound_space, &bound_payload)?,
        )?;
        Ok(self.wrap_bound_factor(out))
    }

    /// TensorKit 0.17 / MatrixAlgebraKit `right_null`: `n : W <- domain` with
    /// `t * n^H = 0`.
    ///
    /// # Null bond
    ///
    /// As [`Self::left_null`], mirrored: `W` is a fresh non-dual single-leg
    /// bond space with `cols_c − rank_c` directions per coupled sector under
    /// the same SVD numerical-rank cutoff, sectors with none absent — and the
    /// same divergence from TensorKit/MatrixAlgebraKit's QR-based default
    /// applies.
    ///
    /// # Errors
    ///
    /// As [`Self::svd_compact`]: the seam's own errors, unfiltered.
    ///
    /// # Complexity
    ///
    /// As [`Self::left_null`]: sectorwise cubic, compact-diagonal payload
    /// materialized dense first. A lazy adjoint mirrors the parent redirect
    /// described there.
    pub(super) fn right_null_multiplicity_free(&self) -> Result<Self, Error>
    where
        D: FactorizationScalar,
    {
        if matches!(&self.repr, TypedTensorRepr::Adjoint(_)) {
            return self
                .adjoint()?
                .left_null_multiplicity_free()?
                .adjoint()?
                .materialized_tensor_uncached();
        }
        let mut dense = self.runtime.lease_dense();
        let (bound_space, bound_payload) = self.bound_payload()?;
        let out = tenet_matrixalgebra::right_null_dyn(
            dense.dense(),
            &BoundDynamicTensorRef::try_new(bound_space, &bound_payload)?,
        )?;
        Ok(self.wrap_bound_factor(out))
    }

    /// TensorKit 0.17 / MatrixAlgebraKit `left_polar`: the polar decomposition
    /// `t = w ∘ p`, returned as a [`LeftPolar`] — `w` isometric (`w† ∘ w = id` on the
    /// domain) and `p` Hermitian positive semidefinite.
    ///
    /// Factor spaces per TensorKit 0.17: `w` lives on the input's
    /// own space `codomain <- domain`, `p` on `domain <- domain`. TensorKit
    /// also exposes algorithm kinds for the polars; TeNeT deliberately does
    /// not. A lazy typed adjoint executes
    /// the opposite polar on its exact owned parent, keeps the already-owned
    /// positive factor, and returns an owned adjoint of the isometry without
    /// materializing the receiver.
    ///
    /// # Errors
    ///
    /// As [`Self::svd_compact`]: the seam's own errors, unfiltered — in
    /// particular [`Error::Operation`] when some coupled-sector matrix has
    /// fewer rows than columns (the left polar needs every sector at least as
    /// tall as it is wide).
    ///
    /// # Complexity
    ///
    /// `O(Σ_c n_c³)` — sectorwise cubic, no global materialization; the seam
    /// factorizes each coupled sector on its own. A compact-diagonal payload
    /// (TensorKit's `DiagonalTensorMap`) is densified into an operation-local
    /// buffer first, as for [`Self::qr_compact`]: TensorKit 0.17 has
    /// no diagonal polar specialization either (its `DiagonalAlgorithm`
    /// table gives `DiagonalTensorMap` only `copy_input` for the polars, so
    /// it dispatches dense per block), and the
    /// issue #613 Group 4 contract requires any compact fast path to be
    /// individually re-proven — out of scope here.
    pub(super) fn left_polar_multiplicity_free(&self) -> Result<LeftPolar<Self>, Error>
    where
        D: FactorizationScalar,
    {
        if let TypedTensorRepr::Adjoint(view) = &self.repr {
            let mut dense = self.runtime.lease_dense();
            let mut lease = self.runtime.lease_context()?;
            let LeftPolar { w, p } = tenet_matrixalgebra::left_polar_adjoint_parent_dyn(
                dense.dense(),
                lease.context().multiplicity_free_lane::<D>()?,
                &BoundDynamicTensorRef::try_new(&view.parent.space, view.parent_data())?,
            )?;
            return Ok(LeftPolar {
                w: self.wrap_bound_factor(w),
                p: self.wrap_bound_factor(p),
            });
        }
        // Dense lease before the context lease — the polar seam recouples
        // internally, so unlike QR/LQ/null it takes the context lane; the
        // lease order matches every existing site that takes both lanes.
        let mut dense = self.runtime.lease_dense();
        let mut lease = self.runtime.lease_context()?;
        let (bound_space, bound_payload) = self.bound_payload()?;
        let LeftPolar { w, p } = tenet_matrixalgebra::left_polar_dyn(
            dense.dense(),
            lease.context().multiplicity_free_lane::<D>()?,
            &BoundDynamicTensorRef::try_new(bound_space, &bound_payload)?,
        )?;
        Ok(LeftPolar {
            w: self.wrap_bound_factor(w),
            p: self.wrap_bound_factor(p),
        })
    }

    /// TensorKit 0.17 / MatrixAlgebraKit `right_polar`: the polar
    /// decomposition `t = p ∘ wh`, returned as a [`RightPolar`] — `p`
    /// Hermitian positive semidefinite and `wh` a coisometry
    /// (`wh ∘ wh† = id` on the codomain).
    ///
    /// Factor spaces per TensorKit 0.17: `p` on
    /// `codomain <- codomain`, `wh` on the input's own space
    /// `codomain <- domain`. Everything [`Self::left_polar`] says about
    /// algorithm kinds, adjoint views and the compact-diagonal route holds
    /// here unchanged.
    ///
    /// # Errors
    ///
    /// As [`Self::left_polar`], mirrored: [`Error::Operation`] when some
    /// coupled-sector matrix has fewer columns than rows.
    ///
    /// # Complexity
    ///
    /// As [`Self::left_polar`]: `O(Σ_c n_c³)`, sectorwise, with a
    /// compact-diagonal payload materialized first.
    pub(super) fn right_polar_multiplicity_free(&self) -> Result<RightPolar<Self>, Error>
    where
        D: FactorizationScalar,
    {
        if let TypedTensorRepr::Adjoint(view) = &self.repr {
            let mut dense = self.runtime.lease_dense();
            let mut lease = self.runtime.lease_context()?;
            let RightPolar { p, wh: w } = tenet_matrixalgebra::right_polar_adjoint_parent_dyn(
                dense.dense(),
                lease.context().multiplicity_free_lane::<D>()?,
                &BoundDynamicTensorRef::try_new(&view.parent.space, view.parent_data())?,
            )?;
            return Ok(RightPolar {
                p: self.wrap_bound_factor(p),
                wh: self.wrap_bound_factor(w),
            });
        }
        // See `left_polar` for the lease order rationale.
        let mut dense = self.runtime.lease_dense();
        let mut lease = self.runtime.lease_context()?;
        let (bound_space, bound_payload) = self.bound_payload()?;
        let RightPolar { p, wh: w } = tenet_matrixalgebra::right_polar_dyn(
            dense.dense(),
            lease.context().multiplicity_free_lane::<D>()?,
            &BoundDynamicTensorRef::try_new(bound_space, &bound_payload)?,
        )?;
        Ok(RightPolar {
            p: self.wrap_bound_factor(p),
            wh: self.wrap_bound_factor(w),
        })
    }

    /// TensorKit 0.17 / MatrixAlgebraKit `eigh_full`: the Hermitian
    /// eigendecomposition `t = v * d * v^H` of an endomorphism, returned as an
    /// [`Eigh`].
    ///
    /// `d : bond <- bond` carries the eigenvalues in compact diagonal storage
    /// (TensorKit's `DiagonalTensorMap`), so `v.compose(&d)` takes the
    /// bond-scaling path; `v : codomain <- bond` is the eigenbasis. The
    /// eigenvalues are real for both payload dtypes — TensorKit's Hermitian `D`
    /// is real too — but `d` keeps the payload dtype `D` so it composes with
    /// `v` directly.
    ///
    /// # Errors
    ///
    /// [`Error::Operation`] when the tensor is not an endomorphism or its
    /// coupled blocks are not Hermitian, and otherwise
    /// [`Error::Core`] / [`Error::FusionAlgebra`] from the seam — which owns
    /// those rules, so they are not re-checked here.
    pub(super) fn eigh_full_multiplicity_free(&self) -> Result<Eigh<Self>, Error>
    where
        D: FactorizationScalar,
    {
        if matches!(&self.repr, TypedTensorRepr::Adjoint(_)) {
            return self
                .materialized_tensor_uncached()?
                .eigh_full_multiplicity_free();
        }
        let mut dense = self.runtime.lease_dense();
        let (bound_space, bound_payload) = self.bound_payload()?;
        let out = tenet_matrixalgebra::eigh_full_dyn(
            dense.dense(),
            &BoundDynamicTensorRef::try_new(bound_space, &bound_payload)?,
        )?;
        let (v, mut eigenvalues) = out.into_parts();
        Ok(Eigh {
            d: self.diagonal_factor(&mut eigenvalues, D::from_real)?,
            v: self.wrap_bound_factor(v),
        })
    }

    /// TensorKit 0.17 / MatrixAlgebraKit `eigh_vals`: the Hermitian eigenvalues
    /// per coupled sector, and nothing else.
    ///
    /// No factor and no bond space is built, so this is the cheap way to ask
    /// about a spectrum — the [`Self::svd_vals`] of the eigendecompositions.
    ///
    /// # Errors
    ///
    /// [`Self::eigh_full`]'s, plus [`Error::FusionAlgebra`] when the provider
    /// cannot decode a coupled sector its own algebra produced.
    pub(super) fn eigh_vals_multiplicity_free(
        &self,
    ) -> Result<Vec<SectorSpectrum<R::Sector>>, Error>
    where
        D: FactorizationScalar,
    {
        if matches!(&self.repr, TypedTensorRepr::Adjoint(_)) {
            return self
                .materialized_tensor_uncached()?
                .eigh_vals_multiplicity_free();
        }
        let mut dense = self.runtime.lease_dense();
        let (bound_space, bound_payload) = self.bound_payload()?;
        let raw = tenet_matrixalgebra::eigh_vals_dyn(
            dense.dense(),
            &BoundDynamicTensorRef::try_new(bound_space, &bound_payload)?,
        )?;
        self.decode_spectrum(raw)
    }

    /// TensorKit 0.17 / MatrixAlgebraKit `eig_full`: the general
    /// (non-Hermitian) eigendecomposition `t = v * d * v^-1` of an
    /// endomorphism, returned as an [`Eig`].
    ///
    /// Both factors are complex whatever `D` is: a real matrix's eigenpairs are
    /// complex in general, and TensorKit's `eigen` likewise returns
    /// `ComplexF64` `D` and `V` for a real argument. `d` carries the spectrum
    /// in compact diagonal storage.
    ///
    /// # The `D::Eig` bound
    ///
    /// The `where` clause is vacuous for the payload types this facade
    /// admits — `f64` and `Complex64` have `Eig = Complex64`, `f32` and
    /// `Complex32` have `Eig = Complex32`, all [`TensorScalar`]s. It is written
    /// out because
    /// [`tenet_matrixalgebra::FactorScalar::Eig`] is the wider seam's associated
    /// type and is not constrained to this facade's scalars, so without it the
    /// factors could not be `TensorMap`s at all. Per-method rather than on the
    /// impl block, so nothing outside the `eig_*` row pays for it.
    ///
    /// # Errors
    ///
    /// [`Error::Operation`] when the tensor is not an endomorphism, and
    /// otherwise [`Error::Core`] / [`Error::FusionAlgebra`] from the seam.
    #[allow(clippy::type_complexity)]
    pub(super) fn eig_full_multiplicity_free(
        &self,
    ) -> Result<Eig<TensorMap<R, <D as FactorScalar>::Eig>>, Error>
    where
        D: AdvancedLinalgScalar,
        <D as FactorScalar>::Eig: TensorScalar,
    {
        if matches!(&self.repr, TypedTensorRepr::Adjoint(_)) {
            return self
                .materialized_tensor_uncached()?
                .eig_full_multiplicity_free();
        }
        let mut dense = self.runtime.lease_dense();
        let (bound_space, bound_payload) = self.bound_payload()?;
        let out = tenet_matrixalgebra::eig_full_dyn(
            dense.dense(),
            &BoundDynamicTensorRef::try_new(bound_space, &bound_payload)?,
        )?;
        let (v, mut eigenvalues) = out.into_parts();
        Ok(Eig {
            d: diagonal_factor_on(
                &self.runtime,
                self.logical_space(),
                &mut eigenvalues,
                <<D as FactorScalar>::Eig as FactorScalar>::from_complex64,
            )?,
            v: wrap_factor_on(&self.runtime, v),
        })
    }

    /// TensorKit 0.17 / MatrixAlgebraKit `eig_vals`: the general eigenvalues
    /// per coupled sector, and nothing else. `Complex64` for every payload dtype.
    ///
    /// # Errors
    ///
    /// [`Self::eig_full`]'s, plus [`Error::FusionAlgebra`] when the provider
    /// cannot decode a coupled sector its own algebra produced.
    pub(super) fn eig_vals_multiplicity_free(
        &self,
    ) -> Result<Vec<SectorSpectrum<R::Sector, num_complex::Complex64>>, Error>
    where
        D: AdvancedLinalgScalar,
        // Carried across the whole row even though this member builds no
        // factor: the three are one API surface, and a caller who can spell two
        // of them but not the third would be reading an accident.
        <D as FactorScalar>::Eig: TensorScalar,
    {
        if matches!(&self.repr, TypedTensorRepr::Adjoint(_)) {
            return self
                .materialized_tensor_uncached()?
                .eig_vals_multiplicity_free();
        }
        let mut dense = self.runtime.lease_dense();
        let (bound_space, bound_payload) = self.bound_payload()?;
        let raw = tenet_matrixalgebra::eig_vals_dyn(
            dense.dense(),
            &BoundDynamicTensorRef::try_new(bound_space, &bound_payload)?,
        )?;
        self.decode_spectrum(raw)
    }

    /// Multiplicity-free implementation of the public mode-dispatched exponential.
    pub(super) fn exp_multiplicity_free(&self) -> Result<Self, Error>
    where
        D: AdvancedLinalgScalar,
    {
        if let Some(spectrum) = self.spectrum() {
            // Why the spectrum is exponentiated unconditionally while the dense
            // arm asks about hermiticity: the dense question picks an algorithm
            // (spectral or Padé), not a domain, and a diagonal is already in its
            // eigenbasis so neither answer would change what happens here.
            // TensorKit splits the same way (#576, #578).
            return Ok(self.with_spectrum(map_spectrum(spectrum, |value| Ok(value.exp_value()))?));
        }
        let mut dense = self.runtime.lease_dense();
        let mut lease = self.runtime.lease_context()?;
        let local = matches!(&self.repr, TypedTensorRepr::Adjoint(_))
            .then(|| self.materialized_tensor_uncached())
            .transpose()?;
        let body = local
            .as_ref()
            .and_then(Self::owned_body)
            .unwrap_or_else(|| self.owned_body().expect("owned representation"));
        let out = tenet_matrixalgebra::exp_dyn(
            dense.dense(),
            lease.context().multiplicity_free_lane::<D>()?,
            &BoundDynamicTensorRef::try_new(&body.space, body.materialized_dense_data().as_ref())?,
        )?;
        Ok(self.wrap_bound_factor(out))
    }

    /// Multiplicity-free implementation of the public mode-dispatched inverse.
    pub(super) fn inv_multiplicity_free(&self) -> Result<Self, Error>
    where
        D: AdvancedLinalgScalar,
    {
        if let Some(spectrum) = self.spectrum() {
            // Why `== 0` and not a tolerance: the dense arm has none either
            // (the solve either fails or it does not), and a compact arm that
            // refused near-zero entries would let storage change the answer.
            // Match the dense arm: exact zero is singular.
            return Ok(self.with_spectrum(map_spectrum(spectrum, |value| {
                if value.abs_value() == 0.0 {
                    Err(Error::InvalidArgument(
                        "inv of a singular diagonal (zero entry)".to_string(),
                    ))
                } else {
                    Ok(value.recip_value())
                }
            })?));
        }
        if matches!(&self.repr, TypedTensorRepr::Adjoint(_)) {
            // (A†)^-1 = (A^-1)†. Avoid materializing the receiver by
            // solving the owned parent, then detach the final adjoint so the
            // result retains neither the parent inverse nor its payload.
            return self
                .adjoint()?
                .inv_multiplicity_free()?
                .adjoint()?
                .materialized_tensor_uncached();
        }
        let mut dense = self.runtime.lease_dense();
        let (bound_space, bound_payload) = self.bound_payload()?;
        let out = tenet_matrixalgebra::inv_direct_dyn(
            dense.dense(),
            &BoundDynamicTensorRef::try_new(bound_space, &bound_payload)?,
        )?;
        Ok(self.wrap_bound_factor(out))
    }

    /// Solves `self * x = rhs` sector by sector without forming an inverse.
    ///
    /// The two codomains must be exactly equal and `self` must have isomorphic
    /// codomain and domain. The result is `domain(self) <- domain(rhs)` and
    /// keeps `self`'s exact provider allocation. Dense blocks are written
    /// directly into the final output; compact diagonal divisors reuse the
    /// elementwise reciprocal and bond-scaling path.
    pub(super) fn solve_multiplicity_free(&self, rhs: &Self) -> Result<Self, Error>
    where
        D: AdvancedLinalgScalar,
    {
        if !self.runtime.same_runtime(&rhs.runtime) {
            return Err(Error::RuntimeMismatch);
        }
        if !self.same_rule(rhs) {
            return Err(Error::RuleMismatch);
        }
        if self.logical_space().space().homspace().codomain()
            != rhs.logical_space().space().homspace().codomain()
        {
            return Err(Error::InvalidArgument(
                "solve requires equal divisor and right-hand-side codomains".to_string(),
            ));
        }
        if !self.logical_space().codomain_isomorphic_to_domain()? {
            return Err(Error::from(
                tenet_tensors::OperationError::UnsupportedTensorContractScope {
                    message: "solve requires an isomorphic divisor codomain and domain",
                },
            ));
        }
        // Only a compact divisor reads a lazy `rhs` in place (through
        // `compose`); the dense route materializes both operands.
        if self.spectrum().is_none() {
            rhs.refuse_borrowed_view("solve")?;
        }

        if let Some(spectrum) = self.spectrum() {
            if spectrum
                .iter()
                .flat_map(|entry| &entry.values)
                .any(|value| value.abs_value() == 0.0)
            {
                return Err(Error::from(tenet_tensors::OperationError::Dense(
                    tenet_dense::DenseError::NumericalFailure {
                        backend: tenet_dense::DenseBackend::Tenferro,
                        op: "solve_into",
                        message: "singular compact diagonal divisor".to_string(),
                    },
                )));
            }
            let solved = self.inv_multiplicity_free()?.compose(rhs)?;
            let TypedTensorRepr::Owned(body) = solved.repr else {
                return Err(internal_layout_error(
                    "compact solve must produce an owned result",
                ));
            };
            let space = self
                .logical_space()
                .rebind_validated(&body.space.validated_layout())?;
            return Ok(Self {
                runtime: self.runtime.clone(),
                repr: owned_repr(TypedTensorBody::with_shared_payload(
                    space,
                    Arc::clone(&body.data),
                )),
            });
        }

        let lhs_local = matches!(&self.repr, TypedTensorRepr::Adjoint(_))
            .then(|| self.materialized_tensor_uncached())
            .transpose()?;
        let rhs_local = matches!(&rhs.repr, TypedTensorRepr::Adjoint(_))
            .then(|| rhs.materialized_tensor_uncached())
            .transpose()?;
        let lhs = lhs_local.as_ref().unwrap_or(self);
        let rhs = rhs_local.as_ref().unwrap_or(rhs);
        let (lhs_space, lhs_payload) = lhs.bound_payload()?;
        let (rhs_space, rhs_payload) = rhs.bound_payload()?;
        let mut dense = self.runtime.lease_dense();
        let out = tenet_matrixalgebra::solve_left_direct_dyn(
            dense.dense(),
            &BoundDynamicTensorRef::try_new(lhs_space, &lhs_payload)?,
            &BoundDynamicTensorRef::try_new(rhs_space, &rhs_payload)?,
        )?;
        Ok(self.wrap_bound_factor(out))
    }

    /// Multiplicity-free implementation of [`Self::pinv`].
    pub(super) fn pinv_multiplicity_free(&self, rcond: f64) -> Result<Self, Error>
    where
        D: AdvancedLinalgScalar,
    {
        // Ahead of the storage split, so both arms answer alike: the seam
        // repeats this check for its own callers, but the compact arm never
        // reaches the seam.
        if !rcond.is_finite() || rcond < 0.0 {
            return Err(Error::InvalidArgument(
                "pinv rcond must be finite and non-negative".to_string(),
            ));
        }
        if let Some(spectrum) = self.spectrum() {
            // A non-finite entry is rejected rather than folded: `f64::max`
            // would drop a NaN and `NaN > cutoff` would then zero it, a
            // silent finite answer (the dense arm's `pinv_cutoff` contract).
            let sigma_max = spectrum
                .iter()
                .flat_map(|entry| entry.values.iter())
                .try_fold(0.0f64, |largest, &value| {
                    let magnitude = value.abs_value();
                    magnitude.is_finite().then(|| largest.max(magnitude))
                })
                .ok_or_else(|| {
                    Error::InvalidArgument("pinv singular values must be finite".to_string())
                })?;
            let cutoff = rcond * sigma_max;
            // Strict `>`, matching the dense fold: a
            // value exactly on the cutoff is cut. Changing it to `>=` is what
            // `pinv_cuts_a_singular_value_sitting_exactly_on_the_cutoff` kills.
            return Ok(self.with_spectrum(map_spectrum(spectrum, |value| {
                Ok(if value.abs_value() > cutoff {
                    value.recip_value()
                } else {
                    D::from_real(0.0)
                })
            })?));
        }
        let mut dense = self.runtime.lease_dense();
        let mut lease = self.runtime.lease_context()?;
        let out = match &self.repr {
            TypedTensorRepr::Adjoint(view) => tenet_matrixalgebra::pinv_adjoint_parent_dyn(
                dense.dense(),
                lease.context().multiplicity_free_lane::<D>()?,
                &BoundDynamicTensorRef::try_new(&view.parent.space, view.parent_data())?,
                rcond,
            )
            .map_err(pinv_seam_error)?,
            TypedTensorRepr::Owned(_) => {
                let (bound_space, bound_payload) = self.bound_payload()?;
                tenet_matrixalgebra::pinv_dyn(
                    dense.dense(),
                    lease.context().multiplicity_free_lane::<D>()?,
                    &BoundDynamicTensorRef::try_new(bound_space, &bound_payload)?,
                    rcond,
                )
                .map_err(pinv_seam_error)?
            }
        };
        Ok(self.wrap_bound_factor(out))
    }

    /// Whether two operands' providers are the same rule. The compact paths
    /// below skip the expert layer, which is where a mismatch would otherwise
    /// be caught, so they have to ask themselves.
    fn same_rule(&self, other: &Self) -> bool {
        self.logical_space().provider().rule_identity()
            == other.logical_space().provider().rule_identity()
    }

    /// The dimension-weighted inner product of two compact spectra,
    /// `Σ_c dim(c) * Σ_i conj(a_i) b_i` — [`weighted_inner`]'s reduction with
    /// the zeros left out, since a bond space's dense form is zero off the
    /// per-sector diagonal.
    fn compact_inner(
        lhs: &[tenet_matrixalgebra::SectorSpectrum<D>],
        rhs: &[tenet_matrixalgebra::SectorSpectrum<D>],
        provider: &R,
    ) -> Result<num_complex::Complex64, Error> {
        if lhs.len() != rhs.len() {
            return Err(spectra_disagree());
        }
        let mut total = num_complex::Complex64::new(0.0, 0.0);
        for (left, right) in lhs.iter().zip(rhs) {
            if left.sector != right.sector || left.values.len() != right.values.len() {
                return Err(spectra_disagree());
            }
            // `D::Wide` for the same reason as `coupled_region_inner`: the
            // identity for the double pair, double precision for the single
            // one, so a compact `norm`/`inner` answers like its dense twin.
            let mut partial = D::Wide::from_real(0.0);
            for (&a, &b) in left.values.iter().zip(&right.values) {
                partial = partial + FactorScalar::adjoint(a.widen()) * b.widen();
            }
            total += partial.widen_complex() * provider.dim_scalar(left.sector);
        }
        Ok(total)
    }

    /// Mixed compact/dense inner over the stored diagonal only. A compact
    /// operand defines structural zeros off diagonal, so those dense entries
    /// are not part of this reduction (including non-finite values).
    fn compact_dense_inner(
        spectrum: &[tenet_matrixalgebra::SectorSpectrum<D>],
        dense: &Self,
        compact_is_lhs: bool,
    ) -> Result<num_complex::Complex64, Error> {
        let logical_structure = dense.logical_space().space().structure();
        let (data_structure, data, dense_is_adjoint) = match &dense.repr {
            TypedTensorRepr::Owned(body) => {
                let TypedData::Dense(data) = body.data.as_ref() else {
                    unreachable!("mixed compact/dense dispatch requires a dense operand")
                };
                (body.space.space().structure(), data.as_slice(), false)
            }
            TypedTensorRepr::Adjoint(view) => (
                view.parent.space.space().structure(),
                view.parent_data(),
                true,
            ),
        };
        if logical_structure.block_count() != spectrum.len()
            || data_structure.required_len()? != data.len()
        {
            return Err(spectra_disagree());
        }
        // A compact operand proves a rank-(1,1) bond endomorphism. Its equal
        // codomain and domain make the adjoint parent's admitted structure
        // content-identical to the logical structure, including block order.
        // Compare that contract once in O(G), then traverse both in lockstep;
        // per-block indexed lookup would make this O(G log G).
        if dense_is_adjoint && logical_structure != data_structure {
            return Err(internal_layout_error(
                "compact/dense inner adjoint parent has a different bond layout",
            ));
        }

        let mut total = num_complex::Complex64::new(0.0, 0.0);
        for (index, entry) in spectrum.iter().enumerate() {
            let logical_block = logical_structure.block(index)?;
            let Some(pair) = logical_block.key().as_fusion_tree_pair() else {
                return Err(spectra_disagree());
            };
            if pair.codomain_tree().coupled() != entry.sector
                || logical_block.shape().len() != 2
                || logical_block.shape()[0] != logical_block.shape()[1]
                || logical_block.shape()[0] != entry.values.len()
            {
                return Err(spectra_disagree());
            }
            let data_block = data_structure.block(index)?;
            if data_block.shape().len() != 2
                || data_block.shape()[0] != data_block.shape()[1]
                || data_block.shape()[0] != entry.values.len()
            {
                return Err(spectra_disagree());
            }
            let stride = data_block.strides()[0] + data_block.strides()[1];
            let mut partial = D::Wide::from_real(0.0);
            for (i, &compact_value) in entry.values.iter().enumerate() {
                let dense_value = *data.get(data_block.offset() + i * stride).ok_or_else(|| {
                    internal_layout_error("compact/dense inner diagonal exceeds scalar storage")
                })?;
                let compact_value = compact_value.widen();
                let dense_value = dense_value.widen();
                partial = partial
                    + match (compact_is_lhs, dense_is_adjoint) {
                        (true, false) => FactorScalar::adjoint(compact_value) * dense_value,
                        (true, true) => {
                            FactorScalar::adjoint(compact_value)
                                * FactorScalar::adjoint(dense_value)
                        }
                        (false, false) => FactorScalar::adjoint(dense_value) * compact_value,
                        (false, true) => dense_value * compact_value,
                    };
            }
            total +=
                partial.widen_complex() * dense.logical_space().provider().dim_scalar(entry.sector);
        }
        Ok(total)
    }

    /// The linear combination `alpha * self + beta * other`.
    ///
    /// Both operands must live on the same runtime and on the same space —
    /// identical hom space and block layout — since the combination is
    /// element-wise on the shared storage order.
    ///
    /// # False friend
    ///
    /// VectorInterface's `add(y, x, α, β)` is `y * β + x * α`: its **first**
    /// coefficient belongs to its **second** argument. Here `alpha` belongs to
    /// `self` and `beta` to `other`. Callers
    /// arriving from Julia should go by the argument order, not by the
    /// coefficient names.
    ///
    /// # Errors
    ///
    /// - [`Error::RuntimeMismatch`] when the operands belong to different
    ///   runtimes, as for [`Self::contract`].
    /// - [`Error::InvalidArgument`] when they do not live on the same space.
    ///   The space comparison already covers rule identity, so a separate
    ///   check would only re-report the same disagreement.
    ///
    /// ```
    /// use std::sync::Arc;
    ///
    /// use tenet::sector::{FermionParityFusionRule, Z2Irrep};
    /// use tenet::typed::{GradedSpace, Runtime, TensorMap};
    ///
    /// let runtime = Runtime::builder().build()?;
    /// let v = GradedSpace::try_new(
    ///     Arc::new(FermionParityFusionRule),
    ///     [(Z2Irrep::EVEN, 1), (Z2Irrep::ODD, 1)],
    /// )?;
    /// let t: TensorMap<_, f64> = TensorMap::rand_with_seed(&runtime, [&v], [&v], 10)?;
    ///
    /// // `1.0 * t + 1.0 * t` doubles every entry, so the norm doubles too.
    /// let doubled = t.axpby(1.0, &t, 1.0)?;
    /// assert!((doubled.norm(2.0)? - 2.0 * t.norm(2.0)?).abs() < 1e-12);
    /// # Ok::<(), tenet::typed::Error>(())
    /// ```
    pub(super) fn add_multiplicity_free(
        &self,
        other: &Self,
        alpha: D,
        beta: D,
    ) -> Result<Self, Error> {
        // Runtime first, exactly as `contract` does: crossing runtimes is a
        // trust-boundary violation rather than an algebra error.
        if !self.runtime.same_runtime(&other.runtime) {
            return Err(Error::RuntimeMismatch);
        }
        let _host_pool = self.runtime.enter_host_pool();
        // `DynamicFusionMapSpace: PartialEq` covers the hom space, the
        // codomain/domain split and the block structure, which is exactly what
        // makes the zipped element-wise combination below meaningful.
        if self.logical_space().space() != other.logical_space().space() {
            return Err(Error::InvalidArgument(
                "tensors live on different spaces or block layouts".to_string(),
            ));
        }
        if matches!(&self.repr, TypedTensorRepr::Adjoint(_))
            || matches!(&other.repr, TypedTensorRepr::Adjoint(_))
        {
            if let (Some(spectrum), TypedTensorRepr::Adjoint(_)) = (self.spectrum(), &other.repr) {
                let (operand, dense) = other.fusion_operand_and_data();
                let mut data = tenet_tensors::oriented_fusion_add_owned(
                    self.logical_space().space().structure(),
                    operand,
                    &dense,
                    operand,
                    &dense,
                    beta,
                    D::from_real(0.0),
                )?;
                add_spectrum_into(self.logical_space().space(), &mut data, spectrum, alpha)?;
                return Ok(self.with_data(data));
            }
            if let (TypedTensorRepr::Adjoint(_), Some(spectrum)) = (&self.repr, other.spectrum()) {
                let (operand, dense) = self.fusion_operand_and_data();
                let mut data = tenet_tensors::oriented_fusion_add_owned(
                    self.logical_space().space().structure(),
                    operand,
                    &dense,
                    operand,
                    &dense,
                    alpha,
                    D::from_real(0.0),
                )?;
                add_spectrum_into(self.logical_space().space(), &mut data, spectrum, beta)?;
                return Ok(self.with_data(data));
            }
            let (lhs, lhs_data) = self.fusion_operand_and_data();
            let (rhs, rhs_data) = other.fusion_operand_and_data();
            let data = tenet_tensors::oriented_fusion_add_owned(
                self.logical_space().space().structure(),
                lhs,
                &lhs_data,
                rhs,
                &rhs_data,
                alpha,
                beta,
            )?;
            return Ok(self.with_data(data));
        }
        match (self.spectrum(), other.spectrum()) {
            // Two spectra on one bond space: the sum is diagonal too.
            (Some(lhs), Some(rhs)) => {
                if lhs.len() != rhs.len() {
                    return Err(spectra_disagree());
                }
                let sum = lhs
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
                                .map(|(&x, &y)| scale_value(x, alpha) + scale_value(y, beta))
                                .collect(),
                        })
                    })
                    .collect::<Result<Vec<_>, Error>>()?;
                return Ok(self.with_spectrum(sum));
            }
            // Mixed: the result is dense, but the *diagonal operand* is never
            // materialized to get there — the one O(n²) buffer this needs is
            // the owned result, which scatters the spectrum onto its own
            // diagonal. Materializing first would allocate a second.
            (Some(diagonal), None) => {
                return Ok(self.with_data(scatter_spectrum(
                    self.logical_space().space(),
                    other
                        .owned_body()
                        .expect("owned add input")
                        .materialized_dense_data()
                        .as_ref(),
                    beta,
                    diagonal,
                    alpha,
                )?))
            }
            (None, Some(diagonal)) => {
                return Ok(self.with_data(scatter_spectrum(
                    self.logical_space().space(),
                    self.owned_body()
                        .expect("owned add input")
                        .materialized_dense_data()
                        .as_ref(),
                    alpha,
                    diagonal,
                    beta,
                )?))
            }
            (None, None) => {}
        }
        Ok(self.with_data(
            self.owned_body()
                .expect("owned add input")
                .materialized_dense_data()
                .as_ref()
                .iter()
                .zip(
                    other
                        .owned_body()
                        .expect("owned add input")
                        .materialized_dense_data()
                        .as_ref(),
                )
                .map(|(&x, &y)| scale_value(x, alpha) + scale_value(y, beta))
                .collect(),
        ))
    }

    /// `factor * self` (TensorKit `scale`).
    ///
    /// Infallible because the host payload dtype is the type parameter `D`;
    /// `factor` is simply another `D`.
    ///
    /// Compact diagonal storage is preserved: scaling a spectrum factor stays
    /// `Σ_c k_c` values rather than densifying.
    pub(super) fn scale_multiplicity_free(&self, factor: D) -> Self {
        if let Some(spectrum) = self.spectrum() {
            return self.with_spectrum(
                spectrum
                    .iter()
                    .map(|entry| tenet_matrixalgebra::SectorSpectrum {
                        sector: entry.sector,
                        values: entry
                            .values
                            .iter()
                            .map(|&value| scale_value(value, factor))
                            .collect(),
                    })
                    .collect(),
            );
        }
        if let TypedTensorRepr::Adjoint(view) = &self.repr {
            let parent = Self {
                runtime: self.runtime.clone(),
                repr: TypedTensorRepr::Owned(Arc::clone(&view.parent)),
            };
            return parent
                .scale(FactorScalar::adjoint(factor))
                .adjoint()
                .expect("scaling a pre-admitted adjoint must preserve its layout");
        }
        self.with_data(
            self.owned_body()
                .expect("owned scale input")
                .materialized_dense_data()
                .as_ref()
                .iter()
                .map(|&value| scale_value(value, factor))
                .collect(),
        )
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

    /// TensorKit `adjoint` (dagger): swaps codomain and domain and
    /// conjugate-transposes every block. Real payloads are transposed only;
    /// c64 entries are conjugated as well.
    ///
    /// Dense storage is a lazy parent-backed view, matching TensorKit's
    /// `AdjointTensorMap`: metadata swaps immediately, and only
    /// [`Self::materialize`] builds the whole logical payload.
    /// Compact diagonal storage keeps its established `O(Σ_c k_c)` owned
    /// conjugation path and never becomes a lazy view.
    ///
    /// # Errors
    ///
    /// [`Error::Operation`] / [`Error::Core`] / [`Error::FusionAlgebra`]
    /// straight from the seam, which owns the bend the dagger performs.
    pub(super) fn adjoint_multiplicity_free(&self) -> Result<Self, Error> {
        if let Some(adjoint) = self.compact_adjoint() {
            return Ok(adjoint);
        }
        self.dense_adjoint_view()
    }

    /// TensorKit `norm`: the Frobenius norm weighted by the coupled sectors'
    /// quantum dimensions, `norm(t)^2 = Σ_c dim(c) * |block_c|^2`.
    ///
    /// Always real, for both payload dtypes. For an abelian rule every
    /// `dim(c)` is one and this is the plain Frobenius norm; for a non-abelian
    /// one it is not.
    ///
    /// # Errors
    ///
    /// [`Error::Core`] when the block structure cannot be walked, which is an
    /// engine-internal invariant rather than a caller mistake.
    fn norm_multiplicity_free(&self) -> Result<f64, Error> {
        // Keep the weighted reduction on the shared helper; a second copy
        // could drift from the block semantics it centralizes.
        let provider = self.logical_space().provider();
        if let Some(spectrum) = self.spectrum() {
            return rescaled_power_norm(
                Self::compact_inner(spectrum, spectrum, provider)?.re,
                2.0,
                || Self::spectrum_max_abs(spectrum),
                |max| {
                    Ok(Self::spectrum_weighted_sum(spectrum, provider, |value| {
                        scaled_power(value, max, 2.0)
                    }))
                },
            );
        }
        if let TypedTensorRepr::Adjoint(view) = &self.repr {
            let parent = Self {
                runtime: self.runtime.clone(),
                repr: TypedTensorRepr::Owned(Arc::clone(&view.parent)),
            };
            return parent.norm_multiplicity_free();
        }
        let data_payload = self
            .owned_body()
            .expect("owned norm input")
            .materialized_dense_data();
        let data: &[D] = &data_payload;
        rescaled_power_norm(
            self.weighted_self_inner()?.re,
            2.0,
            || max_abs(data.iter().copied()),
            |max| {
                coupled_region_weighted_sum(
                    self.logical_space().space().structure(),
                    self.logical_space().space().nout(),
                    data,
                    |coupled| Ok::<_, Error>(provider.dim_scalar(coupled)),
                    |value| scaled_power(value, max, 2.0),
                )
            },
        )
    }

    /// The largest stored magnitude of a compact spectrum, as [`max_abs`].
    fn spectrum_max_abs(spectrum: &[tenet_matrixalgebra::SectorSpectrum<D>]) -> f64 {
        max_abs(
            spectrum
                .iter()
                .flat_map(|entry| entry.values.iter().copied()),
        )
    }

    /// `Σ_c dim(c) · Σ_k term(s_{c,k})` over a compact spectrum: the stored
    /// diagonal of each coupled block, whose off-diagonal zeros add nothing
    /// to any power sum with `p > 0`.
    fn spectrum_weighted_sum(
        spectrum: &[tenet_matrixalgebra::SectorSpectrum<D>],
        provider: &R,
        term: impl Fn(D) -> f64,
    ) -> f64 {
        spectrum
            .iter()
            .map(|entry| {
                provider.dim_scalar(entry.sector)
                    * entry.values.iter().map(|&value| term(value)).sum::<f64>()
            })
            .sum()
    }

    /// The `p == Inf` arm of [`Self::norm`]: the largest stored magnitude,
    /// not dimension weighted, NaN-propagating, `+0.0` without entries.
    fn norm_inf_multiplicity_free(&self) -> Result<f64, Error> {
        if let Some(spectrum) = self.spectrum() {
            return Ok(Self::spectrum_max_abs(spectrum));
        }
        if let TypedTensorRepr::Adjoint(view) = &self.repr {
            return Ok(max_abs(view.parent_data().iter().copied()));
        }
        Ok(max_abs(
            self.owned_body()
                .expect("owned norm input")
                .materialized_dense_data()
                .as_ref()
                .iter()
                .copied(),
        ))
    }

    /// [`Self::norm`]'s exponent dispatch. `p == 2` and `p == Inf` take the
    /// Frobenius and maximum arms, so no exponent has two reductions.
    pub(super) fn norm_p_multiplicity_free(&self, p: f64) -> Result<f64, Error> {
        // Checked before any dispatch so an invalid `p` is rejected the same
        // way on compact and dense storage.
        validate_norm_p(p)?;
        if p == 2.0 {
            return self.norm_multiplicity_free();
        }
        if p.is_infinite() {
            return self.norm_inf_multiplicity_free();
        }
        if let TypedTensorRepr::Adjoint(view) = &self.repr {
            let parent = Self {
                runtime: self.runtime.clone(),
                repr: TypedTensorRepr::Owned(Arc::clone(&view.parent)),
            };
            return parent.norm_p_multiplicity_free(p);
        }
        let provider = self.logical_space().provider();
        let power = |value: D| value.widen_complex().norm().powf(p);
        if let Some(spectrum) = self.spectrum() {
            return rescaled_power_norm(
                Self::spectrum_weighted_sum(spectrum, provider, power),
                p,
                || Self::spectrum_max_abs(spectrum),
                |max| {
                    Ok(Self::spectrum_weighted_sum(spectrum, provider, |value| {
                        scaled_power(value, max, p)
                    }))
                },
            );
        }
        let structure = self.logical_space().space().structure();
        let nout = self.logical_space().space().nout();
        let data_payload = self
            .owned_body()
            .expect("owned norm input")
            .materialized_dense_data();
        let data: &[D] = &data_payload;
        let weight_of = |coupled| Ok::<_, Error>(provider.dim_scalar(coupled));
        rescaled_power_norm(
            coupled_region_weighted_sum(structure, nout, data, weight_of, power)?,
            p,
            || max_abs(data.iter().copied()),
            |max| {
                coupled_region_weighted_sum(structure, nout, data, weight_of, |value| {
                    scaled_power(value, max, p)
                })
            },
        )
    }

    /// The dimension-weighted inner product of this tensor with itself, the
    /// body of [`Self::norm`].
    fn weighted_self_inner(&self) -> Result<num_complex::Complex64, Error> {
        weighted_inner(
            self.logical_space().provider(),
            self.logical_space().space().structure(),
            self.logical_space().space().nout(),
            self.owned_body()
                .expect("owned norm input")
                .materialized_dense_data()
                .as_ref(),
            self.owned_body()
                .expect("owned norm input")
                .materialized_dense_data()
                .as_ref(),
        )
    }

    /// TensorKit `dot(x, y)`: the quantum-dimension-weighted Frobenius inner
    /// product `Σ_c dim(c) * <a_c, b_c>` with **`self` conjugated** — the
    /// product is conjugate-linear in its first argument.
    /// A compact diagonal operand contributes structural zeros off diagonal;
    /// those positions of a dense operand are not read, even when non-finite.
    ///
    /// `t.inner(&t)?` is `t.norm(2.0)?²` up to floating point, and for `D = f64`
    /// the result is exactly real.
    ///
    /// # Errors
    ///
    /// Exactly [`Self::axpby`]'s — the operands must share a runtime and a space
    /// — plus [`Error::Core`] from the block-structure walk, as for
    /// [`Self::norm`].
    pub(super) fn inner_multiplicity_free(&self, other: &Self) -> Result<D, Error> {
        if !self.runtime.same_runtime(&other.runtime) {
            return Err(Error::RuntimeMismatch);
        }
        let _host_pool = self.runtime.enter_host_pool();
        if self.logical_space().space() != other.logical_space().space() {
            return Err(Error::InvalidArgument(
                "tensors live on different spaces or block layouts".to_string(),
            ));
        }
        // Compact operands reduce without materializing their structural zeros.
        if let (Some(lhs), Some(rhs)) = (self.spectrum(), other.spectrum()) {
            let provider = self.logical_space().provider();
            return Ok(D::from_complex64(Self::compact_inner(lhs, rhs, provider)?));
        }
        if let Some(lhs) = self.spectrum() {
            return Ok(D::from_complex64(Self::compact_dense_inner(
                lhs, other, true,
            )?));
        }
        if let Some(rhs) = other.spectrum() {
            return Ok(D::from_complex64(Self::compact_dense_inner(
                rhs, self, false,
            )?));
        }
        if matches!(&self.repr, TypedTensorRepr::Adjoint(_))
            || matches!(&other.repr, TypedTensorRepr::Adjoint(_))
        {
            let provider = self.logical_space().provider();
            let (lhs_operand, lhs_data) = self.fusion_operand_and_data();
            let (rhs_operand, rhs_data) = other.fusion_operand_and_data();
            let value = match (&self.repr, &other.repr) {
                (TypedTensorRepr::Adjoint(lhs), TypedTensorRepr::Adjoint(rhs)) => {
                    tenet_tensors::oriented_fusion_inner(
                        lhs.parent.space.space().structure(),
                        tenet_tensors::FusionOperand::direct(rhs.parent.space.space()),
                        rhs.parent_data(),
                        tenet_tensors::FusionOperand::direct(lhs.parent.space.space()),
                        lhs.parent_data(),
                        |sector| provider.dim_scalar(sector),
                    )?
                }
                _ => tenet_tensors::oriented_fusion_inner(
                    self.logical_space().space().structure(),
                    lhs_operand,
                    &lhs_data,
                    rhs_operand,
                    &rhs_data,
                    |sector| provider.dim_scalar(sector),
                )?,
            };
            return Ok(value);
        }
        // `D::from_complex64` is `.re` for the real scalar and the identity for
        // the complex one, so one static conversion covers both scalar types.
        Ok(D::from_complex64(weighted_inner(
            self.logical_space().provider(),
            self.logical_space().space().structure(),
            self.logical_space().space().nout(),
            self.owned_body()
                .expect("owned inner input")
                .materialized_dense_data()
                .as_ref(),
            other
                .owned_body()
                .expect("owned inner input")
                .materialized_dense_data()
                .as_ref(),
        )?))
    }

    /// TensorKit `tr`: the full trace of an endomorphism
    /// (`domain == codomain`), pairing codomain leg `i` with domain leg `i`.
    ///
    /// This is TensorKit's quantum-dimension-weighted block trace:
    /// `Σ_c dim(c) * tr(b_c)`. It is *not* the supertrace — a fermionic rule's
    /// twists belong to tensor contraction, and [`Self::trace_pairs`] is where
    /// they appear. The two therefore disagree for a fermionic provider by
    /// design.
    ///
    /// # Errors
    ///
    /// [`Error::InvalidArgument`] when the tensor is not an endomorphism, and
    /// [`Error::Core`] when the block structure cannot be walked.
    pub(super) fn tr_multiplicity_free(&self) -> Result<D, Error> {
        let hom = self.logical_space().space().homspace();
        // The weighted trace below indexes codomain axis `i` together with
        // domain axis `nout + i` and would be meaningless without this check.
        if hom.codomain().legs() != hom.domain().legs() {
            return Err(Error::InvalidArgument(
                "tr() requires an endomorphism (domain == codomain)".to_string(),
            ));
        }
        if let Some(spectrum) = self.spectrum() {
            // `Σ_c dim(c) * Σ_i d_i`: TensorKit's block trace on a
            // `DiagonalTensorMap`, read straight off the stored values.
            let provider = self.logical_space().provider();
            let mut total = num_complex::Complex64::new(0.0, 0.0);
            for entry in spectrum {
                let mut partial = D::Wide::from_real(0.0);
                for &value in &entry.values {
                    partial = partial + value.widen();
                }
                total += partial.widen_complex() * provider.dim_scalar(entry.sector);
            }
            return Ok(D::from_complex64(total));
        }
        if let TypedTensorRepr::Adjoint(view) = &self.repr {
            let parent = Self {
                runtime: self.runtime.clone(),
                repr: TypedTensorRepr::Owned(Arc::clone(&view.parent)),
            };
            return Ok(FactorScalar::adjoint(parent.tr()?));
        }
        let provider = self.logical_space().provider();
        Ok(D::from_complex64(weighted_trace(
            self.logical_space().space().structure(),
            self.logical_space().space().nout(),
            self.owned_body()
                .expect("owned trace input")
                .materialized_dense_data()
                .as_ref(),
            |sector| Ok::<_, Error>(provider.dim_scalar(sector)),
        )?))
    }

    /// The single element of a rank-0 (scalar) tensor, e.g. the result of
    /// contracting every leg — TensorKit `scalar` (an empty payload reads
    /// as zero there too).
    ///
    /// Returns `D` directly: the value is the sum of the coupled payload. A
    /// lazy adjoint is materialized operation-locally; a rank-0 payload holds
    /// at most one value per coupled sector.
    ///
    /// # Errors
    ///
    /// [`Error::InvalidArgument`] on a tensor with legs.
    pub fn scalar(&self) -> Result<D, Error> {
        if self.rank() != 0 {
            return Err(Error::InvalidArgument(format!(
                "scalar() requires a rank-0 tensor, got rank {}",
                self.rank()
            )));
        }
        // A rank-0 payload holds at most one element; summing gives the empty
        // payload its zero for free.
        let materialized = self.materialized_tensor_uncached()?;
        Ok(materialized
            .owned_body()
            .expect("uncached materialization is owned")
            .materialized_dense_data()
            .as_ref()
            .iter()
            .fold(D::from_real(0.0), |acc, &value| acc + value))
    }

    pub(super) fn twist_with_inverse(&self, legs: &[usize], inverse: bool) -> Result<Self, Error> {
        let rank = self.rank();
        let name = if inverse { "inverse twist" } else { "twist" };
        if let Some(&leg) = legs.iter().find(|&&leg| leg >= rank) {
            return Err(Error::InvalidArgument(format!(
                "{name} leg {leg} out of range for rank {rank}"
            )));
        }
        if legs.is_empty() {
            return Ok(self.clone());
        }
        let provider = self.logical_space().provider();
        // NoBraiding preflight (PR #620 review): before the compact arm and
        // before any θ evaluation — see `reject_unbraided_nonunit_legs`.
        reject_unbraided_nonunit_legs(
            provider,
            self.logical_space().space().homspace(),
            legs,
            name,
            true,
        )?;
        if let TypedTensorRepr::Adjoint(view) = &self.repr {
            let parent = Self {
                runtime: self.runtime.clone(),
                repr: TypedTensorRepr::Owned(Arc::clone(&view.parent)),
            };
            let axes = logical_adjoint_axes_to_parent(
                view.parent.space.space().nout(),
                view.parent.space.space().nin(),
                legs,
            );
            return parent.twist_with_inverse(&axes, !inverse)?.adjoint();
        }
        let nout = self.codomain_rank();
        if let Some(spectrum) = self.spectrum() {
            // Compact arm: a bond space's two legs both carry the block's
            // coupled sector, so
            // the per-block factor collapses to θ(sector)^|legs|. The space
            // is unchanged, so the payload may stay compact.
            let sector_factor = |sector: tenet_core::SectorId| -> f64 {
                let factor = legs.iter().map(|_| provider.twist_scalar(sector)).product();
                twist_factor_with_inverse(factor, inverse)
            };
            if spectrum
                .iter()
                .all(|entry| sector_factor(entry.sector) == 1.0)
            {
                return Ok(self.clone());
            }
            let scaled = spectrum
                .iter()
                .map(|entry| {
                    let factor = D::from_real(sector_factor(entry.sector));
                    tenet_matrixalgebra::SectorSpectrum {
                        sector: entry.sector,
                        values: entry.values.iter().map(|&value| value * factor).collect(),
                    }
                })
                .collect();
            return Ok(self.with_spectrum_on(self.logical_space().clone(), scaled));
        }
        if twist_is_identity_over_blocks(
            provider,
            self.logical_space().space().structure(),
            nout,
            legs,
        )? {
            return Ok(self.clone());
        }
        let mut data = self
            .owned_body()
            .expect("owned twist input")
            .materialized_dense_data()
            .as_ref()
            .to_vec();
        scale_blocks_impl(self.logical_space().space(), &mut data, &|key| match key {
            BlockKey::FusionTree(key) => twist_block_factor(provider, key, nout, legs, inverse),
            _ => 1.0,
        })?;
        Ok(Self {
            runtime: self.runtime.clone(),
            repr: owned_repr(TypedTensorBody::dense(self.logical_space().clone(), data)),
        })
    }

    pub(super) fn flip_multiplicity_free_with_inverse(
        &self,
        legs: &[usize],
        inverse: bool,
    ) -> Result<Self, Error> {
        let rank = self.rank();
        let name = if inverse { "inverse flip" } else { "flip" };
        if let Some(&leg) = legs.iter().find(|&&leg| leg >= rank) {
            return Err(Error::InvalidArgument(format!(
                "{name} leg {leg} out of range for rank {rank}"
            )));
        }
        if legs.is_empty() {
            return Ok(self.clone());
        }
        let hom = self.logical_space().space().homspace();
        // NoBraiding preflight (PR #620 review): flip's coefficients are
        // built from the same θ/χ — see `reject_unbraided_nonunit_legs`.
        reject_unbraided_nonunit_legs(self.logical_space().provider(), hom, legs, name, false)?;
        if let TypedTensorRepr::Adjoint(view) = &self.repr {
            let parent = Self {
                runtime: self.runtime.clone(),
                repr: TypedTensorRepr::Owned(Arc::clone(&view.parent)),
            };
            let axes = logical_adjoint_axes_to_parent(
                view.parent.space.space().nout(),
                view.parent.space.space().nin(),
                legs,
            );
            return parent
                .flip_multiplicity_free_with_inverse(&axes, !inverse)?
                .adjoint();
        }
        let nout = hom.codomain().len();
        // Sequential semantics for repeated legs, centralized in the shared
        // helper from #580 PR 5.
        let (new_hom, occurrences) = flip_toggled_homspace(hom, legs);
        let space = self.logical_space().derive_from_final_homspace(new_hom)?;
        check_flip_layout_identity(
            self.logical_space().space().structure(),
            space.space().structure(),
        )?;
        let provider = self.logical_space().provider();
        let mut data = self
            .owned_body()
            .expect("owned flip input")
            .materialized_dense_data()
            .as_ref()
            .to_vec();
        scale_blocks_impl(space.space(), &mut data, &|key| match key {
            BlockKey::FusionTree(key) => {
                flip_block_factor(provider, key, nout, &occurrences, inverse)
            }
            _ => 1.0,
        })?;
        Ok(Self {
            runtime: self.runtime.clone(),
            repr: owned_repr(TypedTensorBody::dense(space, data)),
        })
    }

    /// A zero tensor on the same spaces and dtype as `self` (TensorKit
    /// `zerovector`). Dense and compact payloads are freshly initialized to
    /// exact positive zero, independently of non-finite source values. A lazy
    /// adjoint zeros its canonical parent and stays a lazy adjoint.
    pub fn zeros_like(&self) -> Self {
        if let TypedTensorRepr::Adjoint(view) = &self.repr {
            let parent = Self {
                runtime: self.runtime.clone(),
                repr: TypedTensorRepr::Owned(Arc::clone(&view.parent)),
            };
            return parent
                .zeros_like()
                .adjoint()
                .expect("zeroing a pre-admitted adjoint must preserve its layout");
        }
        if let Some(spectrum) = self.spectrum() {
            return self.with_spectrum(
                spectrum
                    .iter()
                    .map(|entry| tenet_matrixalgebra::SectorSpectrum {
                        sector: entry.sector,
                        values: vec![D::from_real(0.0); entry.values.len()],
                    })
                    .collect(),
            );
        }
        self.with_data(vec![
            D::from_real(0.0);
            self.owned_body()
                .expect("owned zero input")
                .materialized_dense_data()
                .as_ref()
                .len()
        ])
    }
}

impl<R, D> TensorMap<R, D>
where
    R: TypedSectorAdmission,
    R::Mode: TypedTensorNullDispatch<R, D> + TypedTensorTransformDispatch<R, D>,
    D: FactorizationScalar,
{
    /// Returns an orthonormal basis `n : codomain(self) <- W` for the numerical
    /// left null space, satisfying `n^H * self ~= 0`.
    ///
    /// In sector `c`, singular values count as nonzero only when
    /// `sigma > epsilon(dtype) * max(m_c, n_c) * sigma_max,c`. The fresh
    /// non-dual one-leg bond `W` contains `m_c - rank_c` directions; sectors
    /// with zero nullity are absent. This intentionally differs from
    /// TensorKit/MatrixAlgebraKit's QR-based default, which reports structural
    /// nullity rather than this SVD numerical rank. [`Self::right_null`]
    /// returns the corresponding basis on the domain side.
    ///
    /// The compact SVD costs
    /// `O(sum_c m_c * n_c * min(m_c, n_c))`, plus an orthonormal completion in
    /// sectors that keep null directions. Compact diagonal input is
    /// materialized first. Lazy adjoints use the opposite null space of their
    /// owned parent and return a detached result without filling the receiver
    /// cache. Checked results use the same provider instance as `self`. TeNeT
    /// creates the output bond only after every sector succeeds; otherwise it
    /// returns an error and no tensor.
    ///
    /// ```
    /// use std::sync::Arc;
    /// use tenet::sector::{U1FusionRule, U1Irrep};
    /// use tenet::typed::{GradedSpace, Runtime, TensorMap};
    ///
    /// let runtime = Runtime::builder().build()?;
    /// let v = GradedSpace::try_new(Arc::new(U1FusionRule), [(U1Irrep::new(0), 2)])?;
    /// let zero: TensorMap<_, f64> = TensorMap::zeros(&runtime, [&v], [&v])?;
    /// let n = zero.left_null(&[0], &[1])?;
    /// assert!(n.adjoint()?.compose(&zero)?.norm(2.0)? < 1e-12);
    /// # Ok::<(), tenet::typed::Error>(())
    /// ```
    ///
    /// `rows` and `cols` are the leg roles: the operation acts on the matrix
    /// view `self.permute(rows, cols)`, and the current split costs nothing
    /// extra (see [`Self::svd_compact`]'s *Leg roles*).
    pub fn left_null(&self, rows: &[usize], cols: &[usize]) -> Result<Self, TypedFacadeError<R>> {
        self.with_leg_roles(
            rows,
            cols,
            <R::Mode as TypedTensorNullDispatch<R, D>>::left_null,
        )
    }

    /// Returns an orthonormal-row basis `n : W <- domain(self)` for the
    /// numerical right null space, satisfying `self * n^H ~= 0`.
    ///
    /// The fresh bond contains `n_c - rank_c` directions per sector. It uses
    /// the same numerical cutoff and cost as [`Self::left_null`], with rows and
    /// columns exchanged. Compact inputs are materialized first; a lazy
    /// adjoint uses the left null space of its owned parent without
    /// materializing the receiver. Checked results use the source provider instance, and a
    /// failure returns no tensor.
    ///
    /// `rows` and `cols` are the leg roles: the operation acts on the matrix
    /// view `self.permute(rows, cols)`, and the current split costs nothing
    /// extra (see [`Self::svd_compact`]'s *Leg roles*).
    pub fn right_null(&self, rows: &[usize], cols: &[usize]) -> Result<Self, TypedFacadeError<R>> {
        self.with_leg_roles(
            rows,
            cols,
            <R::Mode as TypedTensorNullDispatch<R, D>>::right_null,
        )
    }
}

impl<R, D> TensorMap<R, D>
where
    R: TypedSectorAdmission,
    R::Mode: TypedTensorPolarDispatch<R, D> + TypedTensorTransformDispatch<R, D>,
    D: FactorizationScalar,
{
    /// Returns the left polar decomposition `self = w * p` as a [`LeftPolar`].
    ///
    /// `w` lives on the input space `codomain(self) <- domain(self)` and is an
    /// isometry (`w^H * w = id` on the domain). The positive-semidefinite
    /// factor `p : domain(self) <- domain(self)` is `V * S * V^H` from the
    /// compact SVD. Every coupled-sector block must be at least as tall as it
    /// is wide. [`Self::right_polar`] handles wide blocks.
    ///
    /// Cost is `O(sum_c m_c * n_c * min(m_c, n_c))` plus sectorwise
    /// composition. Compact diagonal input is materialized first. A lazy
    /// adjoint runs the opposite decomposition on its owned parent and returns
    /// detached owned factors without materializing the receiver. Checked factors
    /// use the same provider instance as `self`. If that provider rejects an
    /// output space or any sector computation fails, no factors are returned.
    ///
    /// ```
    /// use std::sync::Arc;
    /// use tenet::sector::{U1FusionRule, U1Irrep};
    /// use tenet::typed::{GradedSpace, LeftPolar, Runtime, TensorMap};
    ///
    /// let runtime = Runtime::builder().build()?;
    /// let v = GradedSpace::try_new(Arc::new(U1FusionRule), [(U1Irrep::new(0), 2)])?;
    /// let a: TensorMap<_, f64> = TensorMap::isomorphism(&runtime, [&v], [&v])?.scale(2.0);
    /// let LeftPolar { w, p } = a.left_polar(&[0], &[1])?;
    /// assert!(w.compose(&p)?.axpby(1.0, &a, -1.0)?.norm(2.0)? < 1e-12);
    /// # Ok::<(), tenet::typed::Error>(())
    /// ```
    ///
    /// `rows` and `cols` are the leg roles: the operation acts on the matrix
    /// view `self.permute(rows, cols)`, and the current split costs nothing
    /// extra (see [`Self::svd_compact`]'s *Leg roles*).
    pub fn left_polar(
        &self,
        rows: &[usize],
        cols: &[usize],
    ) -> Result<LeftPolar<Self>, TypedFacadeError<R>> {
        self.with_leg_roles(
            rows,
            cols,
            <R::Mode as TypedTensorPolarDispatch<R, D>>::left_polar,
        )
    }

    /// Returns the right polar decomposition `self = p * wh` as a
    /// [`RightPolar`].
    ///
    /// `p : codomain(self) <- codomain(self)` is positive semidefinite and
    /// `wh` lives on the input space as a coisometry
    /// (`wh * wh^H = id` on the codomain). Every coupled-sector block must be
    /// at least as wide as it is tall. Factor spaces are stated above;
    /// [`Self::left_polar`] describes the corresponding storage routes, lazy
    /// input handling, and cost. Checked factors use the source provider
    /// instance, and a failure returns no factors.
    ///
    /// `rows` and `cols` are the leg roles: the operation acts on the matrix
    /// view `self.permute(rows, cols)`, and the current split costs nothing
    /// extra (see [`Self::svd_compact`]'s *Leg roles*).
    pub fn right_polar(
        &self,
        rows: &[usize],
        cols: &[usize],
    ) -> Result<RightPolar<Self>, TypedFacadeError<R>> {
        self.with_leg_roles(
            rows,
            cols,
            <R::Mode as TypedTensorPolarDispatch<R, D>>::right_polar,
        )
    }
}

impl<R, D> TensorMap<R, D>
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64> + CheckedFusionAlgebra + SectorCodec,
    D: TensorScalar,
{
    fn insert_unit_multiplicity_free(&self, insertion: UnitLegInsertion) -> Result<Self, Error>
    where
        R: CanonicalUnitFusionRule,
    {
        let (UnitLegInsertion::Left { position, .. } | UnitLegInsertion::Right { position, .. }) =
            insertion;
        if position > self.rank() {
            return Err(Error::InvalidArgument(format!(
                "TensorMap::insert_unit: position {position} exceeds rank {}",
                self.rank()
            )));
        }
        let provider = self.logical_space().provider();
        let source_hom = self.logical_space().space().homspace();
        let homspace = match insertion {
            UnitLegInsertion::Left { position, dual } => {
                source_hom.insert_left_unit(provider, position, dual)?
            }
            UnitLegInsertion::Right { position, dual } => {
                source_hom.insert_right_unit(provider, position, dual)?
            }
        };
        let destination = self.logical_space().derive_from_final_homspace(homspace)?;
        validate_unit_layout_correspondence_checked(
            provider,
            (source_hom, self.logical_space().space().structure()),
            (
                destination.space().homspace(),
                destination.space().structure(),
            ),
            insertion,
        )
        .map_err(map_checked_unit_layout_error)?;
        let data = self.shareable_dense_payload();
        Ok(Self {
            runtime: self.runtime.clone(),
            repr: owned_repr(TypedTensorBody::with_shared_payload(destination, data)),
        })
    }

    fn remove_unit_multiplicity_free(&self, axis: usize) -> Result<Self, Error>
    where
        R: CanonicalUnitFusionRule,
    {
        if axis >= self.rank() {
            return Err(Error::InvalidArgument(format!(
                "TensorMap::remove_unit: axis {axis} is out of range for rank {}",
                self.rank()
            )));
        }
        let provider = self.logical_space().provider();
        let source_hom = self.logical_space().space().homspace();
        let nout = source_hom.codomain().len();
        let leg = if axis < nout {
            &source_hom.codomain().legs()[axis]
        } else {
            &source_hom.domain().legs()[axis - nout]
        };
        if leg.sectors() != [provider.vacuum()] || leg.degeneracy(provider.vacuum()) != Some(1) {
            return Err(Error::InvalidArgument(format!(
                "TensorMap::remove_unit: axis {axis} is not a canonical unit leg"
            )));
        }
        // The insertion that this removal undoes, for the correspondence
        // validator: a codomain leg is the right seam's insertion, a domain
        // leg the left seam's.
        let insertion = if axis < nout {
            UnitLegInsertion::Right {
                position: axis,
                dual: leg.is_dual(),
            }
        } else {
            UnitLegInsertion::Left {
                position: axis,
                dual: leg.is_dual(),
            }
        };
        let homspace = source_hom.remove_unit(provider, axis)?;
        let destination = self.logical_space().derive_from_final_homspace(homspace)?;
        validate_unit_layout_correspondence_checked(
            provider,
            (
                destination.space().homspace(),
                destination.space().structure(),
            ),
            (source_hom, self.logical_space().space().structure()),
            insertion,
        )
        .map_err(map_checked_unit_layout_error)?;
        let data = self.shareable_dense_payload();
        Ok(Self {
            runtime: self.runtime.clone(),
            repr: owned_repr(TypedTensorBody::with_shared_payload(destination, data)),
        })
    }
}

/// Canonical-unit leg execution selected by the admitted provider mode.
#[doc(hidden)]
pub trait TypedTensorUnitDispatch<R, D>: TypedTensorModeDispatch<R>
where
    R: TypedSectorAdmission,
    D: TensorScalar,
{
    fn insert_unit(
        tensor: &TensorMap<R, D>,
        insertion: UnitLegInsertion,
    ) -> Result<TensorMap<R, D>, Self::FacadeError>;

    fn remove_unit(
        tensor: &TensorMap<R, D>,
        axis: usize,
    ) -> Result<TensorMap<R, D>, Self::FacadeError>;
}

impl<R, D> TypedTensorUnitDispatch<R, D> for MultiplicityFreeAdmissionMode
where
    R: TypedSectorAdmission<Error = FusionAlgebraError, Mode = MultiplicityFreeAdmissionMode>
        + MultiplicityFreeRigidSymbols<Scalar = f64>
        + CheckedFusionAlgebra
        + SectorCodec
        + CanonicalUnitFusionRule,
    D: TensorScalar,
{
    fn insert_unit(
        tensor: &TensorMap<R, D>,
        insertion: UnitLegInsertion,
    ) -> Result<TensorMap<R, D>, Self::FacadeError> {
        tensor.insert_unit_multiplicity_free(insertion)
    }

    fn remove_unit(
        tensor: &TensorMap<R, D>,
        axis: usize,
    ) -> Result<TensorMap<R, D>, Self::FacadeError> {
        tensor.remove_unit_multiplicity_free(axis)
    }
}

impl<R, D> TypedTensorUnitDispatch<R, D> for CheckedGenericAdmissionMode
where
    R: TypedSectorAdmission<
            Error = <R as CheckedGenericFusion>::Error,
            Mode = CheckedGenericAdmissionMode,
        > + CheckedCanonicalUnitFusionRule,
    D: TensorScalar,
{
    fn insert_unit(
        tensor: &TensorMap<R, D>,
        insertion: UnitLegInsertion,
    ) -> Result<TensorMap<R, D>, Self::FacadeError> {
        generic_insert_unit(tensor, insertion)
    }

    fn remove_unit(
        tensor: &TensorMap<R, D>,
        axis: usize,
    ) -> Result<TensorMap<R, D>, Self::FacadeError> {
        generic_remove_unit(tensor, axis)
    }
}

impl<R, D> TensorMap<R, D>
where
    R: TypedSectorAdmission,
    R::Mode: TypedTensorUnitDispatch<R, D>,
    D: TensorScalar,
{
    /// TensorKit `insertleftunit(t, i; dual)` (`seam = Side::Domain`) and
    /// `insertrightunit(t, i; dual)` (`seam = Side::Codomain`): inserts the
    /// canonical unit leg — the vacuum with degeneracy one, on the dual space
    /// for [`Duality::Dual`] — at zero-based flat external slot `position`.
    ///
    /// `seam` decides only the codomain/domain boundary slot
    /// `position == self.codomain_rank()`: the unit joins the side `seam`
    /// names. Every other position lies strictly inside one side, and both
    /// seams insert there identically. The trivial sector adds no block and
    /// reorders nothing, so the stored values are untouched.
    ///
    /// O(1) for a dense payload: the new body shares the payload allocation,
    /// exactly as TensorKit's `copy = false` default shares `t.data` for an
    /// ordinary `TensorMap`. A lazy dense adjoint is first converted to a
    /// fresh dense tensor. A compact spectrum factor materializes
    /// into a fresh dense payload first (one copy) — the #613 Group 4
    /// contract; TensorKit routes its `DiagonalTensorMap` through the generic
    /// similar+block-copy branch for the same reason. No device arm.
    ///
    /// The provider must certify that its vacuum obeys the canonical unit
    /// laws (`CanonicalUnitFusionRule`, or `CheckedCanonicalUnitFusionRule`
    /// for checked Generic); the hom-space transform and the layout validator
    /// both demand it. Results retain the same provider instance.
    ///
    /// # Errors
    ///
    /// [`Error::InvalidArgument`] when `position` exceeds the rank. Otherwise
    /// the layout derivation's and unit-correspondence validator's own
    /// classes; checked-Generic failures retain their typed
    /// [`GenericTensorError`] variants.
    pub fn insert_unit(
        &self,
        position: usize,
        seam: Side,
        dual: Duality,
    ) -> Result<Self, TypedFacadeError<R>> {
        let dual = dual.is_dual();
        let insertion = match seam {
            Side::Domain => UnitLegInsertion::Left { position, dual },
            Side::Codomain => UnitLegInsertion::Right { position, dual },
        };
        <R::Mode as TypedTensorUnitDispatch<R, D>>::insert_unit(self, insertion)
    }

    /// TensorKit `removeunit(t, i)`: removes the canonical unit leg at flat
    /// external axis `axis`. The selected leg must contain exactly the vacuum
    /// sector with degeneracy one. This undoes [`Self::insert_unit`]; sharing
    /// and compact materialization exactly as there — a dense insert→remove
    /// round trip returns to the original spaces on the original payload
    /// allocation.
    ///
    /// # Errors
    ///
    /// [`Error::InvalidArgument`] when `axis` is out of range or the leg is
    /// not a canonical unit leg. Otherwise the layout derivation's and
    /// validator's own classes; checked-Generic failures retain their typed
    /// [`GenericTensorError`] variants.
    pub fn remove_unit(&self, axis: usize) -> Result<Self, TypedFacadeError<R>> {
        <R::Mode as TypedTensorUnitDispatch<R, D>>::remove_unit(self, axis)
    }
}

pub(super) fn generic_insert_unit<R, D>(
    tensor: &TensorMap<R, D>,
    insertion: UnitLegInsertion,
) -> Result<TensorMap<R, D>, GenericTensorError<R::Error>>
where
    R: CheckedCanonicalUnitFusionRule,
    D: TensorScalar,
{
    let (UnitLegInsertion::Left { position, .. } | UnitLegInsertion::Right { position, .. }) =
        insertion;
    if position > tensor.rank() {
        return Err(GenericTensorError::Facade(Error::InvalidArgument(format!(
            "TensorMap::insert_unit: position {position} exceeds rank {}",
            tensor.rank()
        ))));
    }
    let provider = tensor.logical_space().provider();
    let source_hom = tensor.logical_space().space().homspace();
    let homspace = match insertion {
        UnitLegInsertion::Left { position, dual } => source_hom
            .insert_left_unit_checked(provider, position, dual)
            .map_err(|error| GenericTensorError::Facade(error.into()))?,
        UnitLegInsertion::Right { position, dual } => source_hom
            .insert_right_unit_checked(provider, position, dual)
            .map_err(|error| GenericTensorError::Facade(error.into()))?,
    };
    let destination = BoundDynamicFusionMapSpace::from_final_homspace_generic_checked(
        Arc::clone(tensor.logical_space().provider_arc()),
        homspace,
    )
    .map_err(GenericTensorError::Structure)?;
    validate_unit_layout_correspondence_generic_checked(
        provider,
        (source_hom, tensor.logical_space().space().structure()),
        (
            destination.space().homspace(),
            destination.space().structure(),
        ),
        insertion,
    )
    .map_err(GenericTensorError::Structure)?;
    let data = tensor.shareable_dense_payload();
    Ok(TensorMap {
        runtime: tensor.runtime.clone(),
        repr: owned_repr(TypedTensorBody::with_shared_payload(destination, data)),
    })
}

pub(super) fn generic_remove_unit<R, D>(
    tensor: &TensorMap<R, D>,
    axis: usize,
) -> Result<TensorMap<R, D>, GenericTensorError<R::Error>>
where
    R: CheckedCanonicalUnitFusionRule,
    D: TensorScalar,
{
    if axis >= tensor.rank() {
        return Err(GenericTensorError::Facade(Error::InvalidArgument(format!(
            "TensorMap::remove_unit: axis {axis} is out of range for rank {}",
            tensor.rank()
        ))));
    }
    let provider = tensor.logical_space().provider();
    let source_hom = tensor.logical_space().space().homspace();
    let nout = source_hom.codomain().len();
    let leg = if axis < nout {
        &source_hom.codomain().legs()[axis]
    } else {
        &source_hom.domain().legs()[axis - nout]
    };
    let vacuum = CheckedGenericFusion::vacuum(provider);
    if leg.sectors() != [vacuum] || leg.degeneracy(vacuum) != Some(1) {
        return Err(GenericTensorError::Facade(Error::InvalidArgument(format!(
            "TensorMap::remove_unit: axis {axis} is not a canonical unit leg"
        ))));
    }
    let insertion = if axis < nout {
        UnitLegInsertion::Right {
            position: axis,
            dual: leg.is_dual(),
        }
    } else {
        UnitLegInsertion::Left {
            position: axis,
            dual: leg.is_dual(),
        }
    };
    let homspace = source_hom
        .remove_unit_checked(provider, axis)
        .map_err(|error| GenericTensorError::Facade(error.into()))?;
    let destination = BoundDynamicFusionMapSpace::from_final_homspace_generic_checked(
        Arc::clone(tensor.logical_space().provider_arc()),
        homspace,
    )
    .map_err(GenericTensorError::Structure)?;
    validate_unit_layout_correspondence_generic_checked(
        provider,
        (
            destination.space().homspace(),
            destination.space().structure(),
        ),
        (source_hom, tensor.logical_space().space().structure()),
        insertion,
    )
    .map_err(GenericTensorError::Structure)?;
    let data = tensor.shareable_dense_payload();
    Ok(TensorMap {
        runtime: tensor.runtime.clone(),
        repr: owned_repr(TypedTensorBody::with_shared_payload(destination, data)),
    })
}

// The `re`/`im` gates are law checks (`re(t) + i·im(t)` rebuilds `t`).
// TensorKit's real-input
// branches (`real(t) = t`, `imag(t) = zerovector(t)` for a real scalartype)
// are statically unrepresentable here: these methods exist on the `Complex64`
// impl only, and `to_c64().re()` covers the round trip.
impl<R> TensorMap<R, num_complex::Complex64> {
    /// The element-wise real part, as an f64 tensor map on the same spaces
    /// (TensorKit `Base.real`: blockwise element-wise, result scalartype
    /// real).
    ///
    /// A compact spectrum maps spectrum-to-spectrum and stays compact. A cold
    /// lazy adjoint uses an operation-local payload for the `O(stored_len)`
    /// output; the logical space is shared.
    pub fn re(&self) -> TensorMap<R, f64> {
        self.map_parts(|value| value.re)
    }

    /// The element-wise imaginary part, as an f64 tensor map on the same
    /// spaces (TensorKit `Base.imag`).
    ///
    /// A compact spectrum maps spectrum-to-spectrum and stays compact. A cold
    /// lazy adjoint uses an operation-local payload for the `O(stored_len)`
    /// output; the logical space is shared.
    pub fn im(&self) -> TensorMap<R, f64> {
        self.map_parts(|value| value.im)
    }

    /// The shared owned-input route of [`Self::re`] / [`Self::im`].
    fn map_parts(&self, part: impl Fn(num_complex::Complex64) -> f64) -> TensorMap<R, f64> {
        let materialized = self
            .materialized_tensor_uncached()
            .expect("a pre-admitted typed adjoint must materialize");
        let source = materialized
            .owned_body()
            .expect("uncached materialization is owned");
        let body = match source.data.as_ref() {
            TypedData::Dense(data) => TypedTensorBody::dense(
                source.space.clone(),
                data.iter().map(|&value| part(value)).collect(),
            ),
            TypedData::Diagonal(spectrum) => {
                TypedTensorBody::diagonal(source.space.clone(), map_spectrum_dtype(spectrum, part))
            }
        };
        TensorMap {
            runtime: self.runtime.clone(),
            repr: owned_repr(body),
        }
    }
}

/// [`map_spectrum`]'s cross-dtype sibling: the same sector-and-length
/// preserving value map, for the conversions whose output dtype differs from
/// the input's — which is exactly why the two cannot share one signature.
pub(super) fn map_spectrum_dtype<A: Copy, B>(
    spectrum: &[tenet_matrixalgebra::SectorSpectrum<A>],
    value_of: impl Fn(A) -> B,
) -> Vec<tenet_matrixalgebra::SectorSpectrum<B>> {
    spectrum
        .iter()
        .map(|entry| tenet_matrixalgebra::SectorSpectrum {
            sector: entry.sector,
            values: entry.values.iter().map(|&value| value_of(value)).collect(),
        })
        .collect()
}
