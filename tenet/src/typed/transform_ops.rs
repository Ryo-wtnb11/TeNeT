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
    pub(super) fn tree_transform_multiplicity_free_real(
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

impl<R, D> TensorMap<R, D>
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64> + CheckedFusionAlgebra + SectorCodec,
    D: TensorScalar,
{
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
                    tenet_matrixalgebra::seam::svd_compact_diagonal_factors_dyn(
                        &body.space,
                        spectrum,
                    )?
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
                    tenet_matrixalgebra::seam::svd_compact_adjoint_factors_dyn(
                        dense.dense(),
                        &BoundDynamicTensorRef::try_new(&view.parent.space, view.parent_data())?,
                    )?
                }
                TypedTensorRepr::Owned(_) => {
                    let (bound_space, bound_payload) = self.bound_payload()?;
                    tenet_matrixalgebra::seam::svd_compact_factors_dyn(
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
    /// `s` is compact when the constructed row and column bond legs coincide
    /// exactly, its spectrum covers every bond sector, and the rank-(1,1)
    /// compact layout is admitted. Otherwise `s` is dense and may be
    /// rectangular. The direct owned compact-diagonal input route also avoids
    /// dense input materialization and a solver call. Use
    /// [`Self::materialize`] when a dense singular-value buffer is required.
    ///
    /// # Errors
    ///
    /// As [`Self::svd_compact`]: the seam's own errors, unfiltered.
    pub(super) fn svd_full_multiplicity_free(&self) -> Result<Svd<Self>, Error>
    where
        D: FactorizationScalar,
    {
        if let TypedTensorRepr::Owned(body) = &self.repr {
            if let TypedData::Diagonal(spectrum) = body.data.as_ref() {
                // A diagonal input has square sectors, so compact and full
                // factor spaces coincide, including their nondual bond W.
                if let Some((u, vh, mut spectrum)) =
                    tenet_matrixalgebra::seam::svd_compact_diagonal_factors_dyn(
                        &body.space,
                        spectrum,
                    )?
                {
                    return Ok(Svd {
                        u: self.wrap_bound_factor(u),
                        s: self.diagonal_factor(&mut spectrum, D::from_real)?,
                        vh: self.wrap_bound_factor(vh),
                    });
                }
            }
        }
        let mut dense = self.runtime.lease_dense();
        let factors = match &self.repr {
            TypedTensorRepr::Adjoint(view) => {
                tenet_matrixalgebra::seam::svd_full_adjoint_factors_dyn(
                    dense.dense(),
                    &BoundDynamicTensorRef::try_new(&view.parent.space, view.parent_data())?,
                )?
            }
            TypedTensorRepr::Owned(_) => {
                let (bound_space, bound_payload) = self.bound_payload()?;
                tenet_matrixalgebra::seam::svd_full_factors_dyn(
                    dense.dense(),
                    &BoundDynamicTensorRef::try_new(bound_space, &bound_payload)?,
                )?
            }
        };
        let (u, vh, mut spectrum, row_dimensions, col_dimensions) = factors.into_parts();
        if full_svd_compact_bond(&u, &vh, &spectrum) {
            let space = tenet_matrixalgebra::seam::diagonal_bond_bound_space_like(
                self.logical_space(),
                &spectrum,
            )?;
            if full_svd_compact_layout(&space, &spectrum) {
                return Ok(Svd {
                    u: self.wrap_bound_factor(u),
                    s: diagonal_factor_on_bound(&self.runtime, space, &mut spectrum, D::from_real),
                    vh: self.wrap_bound_factor(vh),
                });
            }
        }
        let s = tenet_matrixalgebra::seam::rectangular_diagonal_bond_tensor(
            self.logical_space(),
            &spectrum,
            &row_dimensions,
            &col_dimensions,
        )?;
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
                    tenet_matrixalgebra::seam::svd_vals_compact_diagonal_dyn(&body.space, spectrum)?
                {
                    return self.decode_spectrum(raw);
                }
            }
        }
        let mut dense = self.runtime.lease_dense();
        // Singular values and coupled-sector ids are invariant under adjoint,
        // so an oriented input or logical-payload copy cannot change this output.
        let raw = match &self.repr {
            TypedTensorRepr::Adjoint(view) => tenet_matrixalgebra::seam::svd_vals_dyn(
                dense.dense(),
                &BoundDynamicTensorRef::try_new(&view.parent.space, view.parent_data())?,
            )?,
            TypedTensorRepr::Owned(_) => {
                let (bound_space, bound_payload) = self.bound_payload()?;
                tenet_matrixalgebra::seam::svd_vals_dyn(
                    dense.dense(),
                    &BoundDynamicTensorRef::try_new(bound_space, &bound_payload)?,
                )?
            }
        };
        self.decode_spectrum(raw)
    }

    fn try_qr_diagonal(&self) -> Option<Qr<Self>>
    where
        D: FactorizationScalar,
    {
        let TypedTensorRepr::Owned(body) = &self.repr else {
            return None;
        };
        let TypedData::Diagonal(spectrum) = body.data.as_ref() else {
            return None;
        };
        let Qr { q, r } = tenet_matrixalgebra::seam::qr_diagonal_dyn(&body.space, spectrum)?;
        let wrap = |values| Self {
            runtime: self.runtime.clone(),
            repr: owned_repr(TypedTensorBody::diagonal(body.space.clone(), values)),
        };
        Some(Qr {
            q: wrap(q),
            r: wrap(r),
        })
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
    /// operation, and the returned factors are owned. An admitted owned compact
    /// diagonal uses O(Σ_c k_c) spectrum work/storage and no dense QR. Both
    /// factors preserve the input bond, including dual orientation.
    pub(super) fn qr_compact_multiplicity_free(&self) -> Result<Qr<Self>, Error>
    where
        D: FactorizationScalar,
    {
        if let Some(factors) = self.try_qr_diagonal() {
            return Ok(factors);
        }
        if matches!(&self.repr, TypedTensorRepr::Adjoint(_)) {
            return self
                .materialized_tensor_uncached()?
                .qr_compact_multiplicity_free();
        }
        let mut dense = self.runtime.lease_dense();
        let (bound_space, bound_payload) = self.bound_payload()?;
        let Qr { q, r } = tenet_matrixalgebra::seam::qr_compact_dyn(
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
    /// compact diagonal uses the same O(Σ_c k_c) route as compact QR.
    pub(super) fn qr_full_multiplicity_free(&self) -> Result<Qr<Self>, Error>
    where
        D: FactorizationScalar,
    {
        if let Some(factors) = self.try_qr_diagonal() {
            return Ok(factors);
        }
        if matches!(&self.repr, TypedTensorRepr::Adjoint(_)) {
            return self
                .materialized_tensor_uncached()?
                .qr_full_multiplicity_free();
        }
        let mut dense = self.runtime.lease_dense();
        let (bound_space, bound_payload) = self.bound_payload()?;
        let Qr { q, r } = tenet_matrixalgebra::seam::qr_full_dyn(
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
    /// compact diagonal uses the QR spectrum route with exchanged factors,
    /// in O(Σ_c k_c) work/storage.
    pub(super) fn lq_compact_multiplicity_free(&self) -> Result<Lq<Self>, Error>
    where
        D: FactorizationScalar,
    {
        if let Some(Qr { q, r }) = self.try_qr_diagonal() {
            return Ok(Lq { l: r, q });
        }
        if matches!(&self.repr, TypedTensorRepr::Adjoint(_)) {
            let Qr { q, r } = self.adjoint()?.qr_compact_multiplicity_free()?;
            return Ok(Lq {
                l: r.adjoint()?.materialized_tensor_uncached()?,
                q: q.adjoint()?.materialized_tensor_uncached()?,
            });
        }
        let mut dense = self.runtime.lease_dense();
        let (bound_space, bound_payload) = self.bound_payload()?;
        let Lq { l, q } = tenet_matrixalgebra::seam::lq_compact_dyn(
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
    /// detached owned output payloads. An admitted owned compact diagonal uses
    /// the same O(Σ_c k_c) spectrum route as compact LQ.
    pub(super) fn lq_full_multiplicity_free(&self) -> Result<Lq<Self>, Error>
    where
        D: FactorizationScalar,
    {
        if let Some(Qr { q, r }) = self.try_qr_diagonal() {
            return Ok(Lq { l: r, q });
        }
        if matches!(&self.repr, TypedTensorRepr::Adjoint(_)) {
            let Qr { q, r } = self.adjoint()?.qr_full_multiplicity_free()?;
            return Ok(Lq {
                l: r.adjoint()?.materialized_tensor_uncached()?,
                q: q.adjoint()?.materialized_tensor_uncached()?,
            });
        }
        let mut dense = self.runtime.lease_dense();
        let (bound_space, bound_payload) = self.bound_payload()?;
        let Lq { l, q } = tenet_matrixalgebra::seam::lq_full_dyn(
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
    /// numerical rank under the same cutoff as the sector's compact SVD, counting
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
    /// Sectorwise cubic for dense input: one compact SVD per coupled sector
    /// plus an orthonormal completion where needed. An admitted owned Host
    /// compact diagonal with exact zero and well-separated nonzero entries
    /// reads its spectrum directly and writes rectangular coordinate factors
    /// in O(Σ k_c + Σ k_c q_c) work and storage, where q_c is nullity.
    /// Positive magnitudes at or below
    /// `max(ε k_c, sqrt(ε)) σ_max,c` conservatively use the existing SVD
    /// route, as do nonfinite, subnormal-scaled, or unsupported layouts. This margin is an
    /// optimization gate, not a promise of bitwise provider-rank agreement.
    /// A lazy adjoint runs the owned parent's
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
        if let TypedTensorRepr::Owned(body) = &self.repr {
            if let TypedData::Diagonal(spectrum) = body.data.as_ref() {
                if let Some(out) =
                    tenet_matrixalgebra::seam::left_null_diagonal_dyn(&body.space, spectrum)?
                {
                    return Ok(self.wrap_bound_factor(out));
                }
            }
        }
        let mut dense = self.runtime.lease_dense();
        let (bound_space, bound_payload) = self.bound_payload()?;
        let out = tenet_matrixalgebra::seam::left_null_dyn(
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
    /// As [`Self::left_null`], including its direct compact-diagonal route
    /// and conservative SVD fallback. A lazy adjoint mirrors the parent
    /// redirect described there.
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
        if let TypedTensorRepr::Owned(body) = &self.repr {
            if let TypedData::Diagonal(spectrum) = body.data.as_ref() {
                if let Some(out) =
                    tenet_matrixalgebra::seam::right_null_diagonal_dyn(&body.space, spectrum)?
                {
                    return Ok(self.wrap_bound_factor(out));
                }
            }
        }
        let mut dense = self.runtime.lease_dense();
        let (bound_space, bound_payload) = self.bound_payload()?;
        let out = tenet_matrixalgebra::seam::right_null_dyn(
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
    /// Dense input costs `O(Σ_c n_c³)` sectorwise. An admitted owned compact
    /// diagonal uses `O(Σ_c n_c)` compact factor values and no dense SVD.
    pub(super) fn left_polar_multiplicity_free(&self) -> Result<LeftPolar<Self>, Error>
    where
        D: FactorizationScalar,
    {
        if let TypedTensorRepr::Owned(body) = &self.repr {
            if let TypedData::Diagonal(spectrum) = body.data.as_ref() {
                if let Some(LeftPolar { w, p }) =
                    tenet_matrixalgebra::seam::left_polar_diagonal_spectra_dyn(
                        &body.space,
                        spectrum,
                    )?
                {
                    return Ok(LeftPolar {
                        w: self.with_spectrum(w),
                        p: self.with_spectrum(p),
                    });
                }
            }
        }
        if let TypedTensorRepr::Adjoint(view) = &self.repr {
            let mut dense = self.runtime.lease_dense();
            let mut lease = self.runtime.lease_context()?;
            let LeftPolar { w, p } = tenet_matrixalgebra::seam::left_polar_adjoint_parent_dyn(
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
        let LeftPolar { w, p } = tenet_matrixalgebra::seam::left_polar_dyn(
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
    /// As [`Self::left_polar`], with compact factor values for an admitted
    /// owned compact diagonal.
    pub(super) fn right_polar_multiplicity_free(&self) -> Result<RightPolar<Self>, Error>
    where
        D: FactorizationScalar,
    {
        if let TypedTensorRepr::Owned(body) = &self.repr {
            if let TypedData::Diagonal(spectrum) = body.data.as_ref() {
                if let Some(RightPolar { p, wh }) =
                    tenet_matrixalgebra::seam::right_polar_diagonal_spectra_dyn(
                        &body.space,
                        spectrum,
                    )?
                {
                    return Ok(RightPolar {
                        p: self.with_spectrum(p),
                        wh: self.with_spectrum(wh),
                    });
                }
            }
        }
        if let TypedTensorRepr::Adjoint(view) = &self.repr {
            let mut dense = self.runtime.lease_dense();
            let mut lease = self.runtime.lease_context()?;
            let RightPolar { p, wh: w } =
                tenet_matrixalgebra::seam::right_polar_adjoint_parent_dyn(
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
        let RightPolar { p, wh: w } = tenet_matrixalgebra::seam::right_polar_dyn(
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
    /// An admitted owned compact diagonal builds the sorted spectrum and dense
    /// permutation eigenbasis without materializing a dense input.
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
        if let TypedTensorRepr::Owned(body) = &self.repr {
            if let TypedData::Diagonal(spectrum) = body.data.as_ref() {
                if let Some(out) =
                    tenet_matrixalgebra::seam::eigh_full_diagonal_dyn(&body.space, spectrum)?
                {
                    let (v, mut eigenvalues) = out.into_parts();
                    return Ok(Eigh {
                        d: self.diagonal_factor(&mut eigenvalues, D::from_real)?,
                        v: self.wrap_bound_factor(v),
                    });
                }
            }
        }
        if matches!(&self.repr, TypedTensorRepr::Adjoint(_)) {
            return self
                .materialized_tensor_uncached()?
                .eigh_full_multiplicity_free();
        }
        let mut dense = self.runtime.lease_dense();
        let (bound_space, bound_payload) = self.bound_payload()?;
        let out = tenet_matrixalgebra::seam::eigh_full_dyn(
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
    /// An owned compact diagonal with finite, exactly real entries is read
    /// directly without materializing input blocks or invoking a dense solver.
    /// Other inputs retain the usual Hermiticity admission and dense path.
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
        if let TypedTensorRepr::Owned(body) = &self.repr {
            if let TypedData::Diagonal(spectrum) = body.data.as_ref() {
                if let Some(raw) =
                    tenet_matrixalgebra::seam::eigh_vals_diagonal_dyn(&body.space, spectrum)?
                {
                    return self.decode_spectrum(raw);
                }
            }
        }
        if matches!(&self.repr, TypedTensorRepr::Adjoint(_)) {
            return self
                .materialized_tensor_uncached()?
                .eigh_vals_multiplicity_free();
        }
        let mut dense = self.runtime.lease_dense();
        let (bound_space, bound_payload) = self.bound_payload()?;
        let raw = tenet_matrixalgebra::seam::eigh_vals_dyn(
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
    /// An admitted owned compact diagonal builds the sorted spectrum and dense
    /// permutation eigenbasis without materializing a dense input.
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
        if let TypedTensorRepr::Owned(body) = &self.repr {
            if let TypedData::Diagonal(spectrum) = body.data.as_ref() {
                if let Some(out) =
                    tenet_matrixalgebra::seam::eig_full_diagonal_dyn(&body.space, spectrum)?
                {
                    let (v, mut eigenvalues) = out.into_parts();
                    return Ok(Eig {
                        d: diagonal_factor_on(
                            &self.runtime,
                            self.logical_space(),
                            &mut eigenvalues,
                            <<D as FactorScalar>::Eig as FactorScalar>::from_complex64,
                        )?,
                        v: wrap_factor_on(&self.runtime, v),
                    });
                }
            }
        }
        if matches!(&self.repr, TypedTensorRepr::Adjoint(_)) {
            return self
                .materialized_tensor_uncached()?
                .eig_full_multiplicity_free();
        }
        let mut dense = self.runtime.lease_dense();
        let (bound_space, bound_payload) = self.bound_payload()?;
        let out = tenet_matrixalgebra::seam::eig_full_dyn(
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
    /// An admitted owned compact diagonal is read without dense materialization.
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
        if let TypedTensorRepr::Owned(body) = &self.repr {
            if let TypedData::Diagonal(spectrum) = body.data.as_ref() {
                if let Some(raw) =
                    tenet_matrixalgebra::seam::eig_vals_diagonal_dyn(&body.space, spectrum)?
                {
                    return self.decode_spectrum(raw);
                }
            }
        }
        if matches!(&self.repr, TypedTensorRepr::Adjoint(_)) {
            return self
                .materialized_tensor_uncached()?
                .eig_vals_multiplicity_free();
        }
        let mut dense = self.runtime.lease_dense();
        let (bound_space, bound_payload) = self.bound_payload()?;
        let raw = tenet_matrixalgebra::seam::eig_vals_dyn(
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
            return Ok(self.with_spectrum(exp_spectrum(spectrum)?));
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
        let out = tenet_matrixalgebra::seam::exp_dyn(
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
            return Ok(self.with_spectrum(inv_spectrum(spectrum)?));
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
        let out = tenet_matrixalgebra::seam::inv_direct_dyn(
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
            reject_singular_compact_divisor(spectrum)?;
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
        let out = tenet_matrixalgebra::seam::solve_left_direct_dyn(
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
            TypedTensorRepr::Adjoint(view) => tenet_matrixalgebra::seam::pinv_adjoint_parent_dyn(
                dense.dense(),
                lease.context().multiplicity_free_lane::<D>()?,
                &BoundDynamicTensorRef::try_new(&view.parent.space, view.parent_data())?,
                rcond,
            )
            .map_err(pinv_seam_error)?,
            TypedTensorRepr::Owned(_) => {
                let (bound_space, bound_payload) = self.bound_payload()?;
                tenet_matrixalgebra::seam::pinv_dyn(
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
    /// The dense-input compact SVD costs
    /// `O(sum_c m_c * n_c * min(m_c, n_c))`, plus an orthonormal completion in
    /// sectors that keep null directions. An admitted Host compact diagonal
    /// with exact zero and well-separated nonzero entries uses a direct
    /// coordinate basis in `O(sum_c k_c + sum_c k_c q_c)` work and storage,
    /// where `k_c` is sector size and `q_c` is nullity. Positive magnitudes
    /// within `max(epsilon(dtype) * k_c, sqrt(epsilon(dtype))) * sigma_max,c`
    /// retain the SVD route, as do nonfinite, subnormal-scaled, and unsupported layouts. This
    /// conservative gate does not promise bitwise agreement with any dense
    /// provider at the cutoff. Lazy adjoints use the opposite null space of their
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
    /// columns exchanged. Admitted Host compact diagonals use the direct
    /// coordinate route described above; a lazy
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
    /// Dense-input cost is `O(sum_c m_c * n_c * min(m_c, n_c))` plus sectorwise
    /// composition. An admitted owned compact diagonal stores both factors
    /// compactly; use [`Self::materialize`] for dense buffers. A lazy
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

impl<R, D> TensorMap<R, D>
where
    R: TypedSectorAdmission,
    R::Mode: TypedTensorAdjointDispatch<R, D>,
    D: TensorScalar,
{
    /// Returns the adjoint `self^H`, swapping codomain and domain and
    /// conjugate-transposing every coupled-sector block.
    ///
    /// Dense storage becomes a lazy parent-backed view: its logical space is
    /// available immediately, and only [`Self::materialize`] builds the
    /// whole logical payload. Applying `adjoint` twice
    /// returns the original owned parent. Compact diagonal storage instead
    /// performs an owned `O(sum_c k_c)` conjugation and stays compact, as
    /// TensorKit `adjoint(::DiagonalTensorMap)` does.
    /// Every result keeps the source's exact provider `Arc`.
    ///
    /// If layout construction or checked pivotal data fails, that layout or
    /// provider error is returned before a view is created. For real payloads
    /// this is a transpose; complex payloads are conjugated as well.
    ///
    /// ```
    /// use std::sync::Arc;
    /// use tenet::sector::{U1FusionRule, U1Irrep};
    /// use tenet::typed::{GradedSpace, Runtime, TensorMap};
    ///
    /// let runtime = Runtime::builder().build()?;
    /// let v = GradedSpace::try_new(Arc::new(U1FusionRule), [(U1Irrep::new(0), 2)])?;
    /// let t: TensorMap<_, f64> = TensorMap::rand_with_seed(&runtime, [&v], [&v], 2)?;
    /// assert_eq!(t.adjoint()?.adjoint()?.dense_data()?, t.dense_data()?);
    /// # Ok::<(), tenet::typed::Error>(())
    /// ```
    pub fn adjoint(&self) -> Result<Self, TypedFacadeError<R>> {
        <R::Mode as TypedTensorAdjointDispatch<R, D>>::adjoint(self)
    }

    /// Borrowed adjoint view: the operand [`Self::adjoint`] would give,
    /// passed without building it. See [`TensorRef`].
    pub fn adjoint_view(&self) -> TensorRef<'_, R, D> {
        TensorRef {
            base: self,
            adjoint: Some(|tensor| {
                <R::Mode as TypedTensorAdjointDispatch<R, D>>::adjoint(tensor)
                    .map_err(<R::Mode as TypedTensorModeDispatch<R>>::facade_error_into_error)
            }),
        }
    }
}
