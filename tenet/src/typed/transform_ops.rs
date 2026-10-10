use super::*;

/// The exact-layout probe of a repartition: the planar axes of
/// `with_planar_axes`'s `Repartition` arm, laid out on the stack so a warm
/// typed `repartition_into` allocates nothing. Ranks above the inline bound
/// take the full path (no probe).
pub(super) fn repartition_probe(
    source_codomain_rank: usize,
    source_rank: usize,
    num_codomain: usize,
    probe: &mut dyn FnMut(tenet_tensors::TreeTransformOperationView<'_>),
) {
    const INLINE_RANK: usize = 64;
    if source_rank > INLINE_RANK || num_codomain > source_rank {
        return;
    }
    let mut axes = [0usize; INLINE_RANK];
    for (slot, axis) in axes
        .iter_mut()
        .zip((0..source_codomain_rank).chain((source_codomain_rank..source_rank).rev()))
    {
        *slot = axis;
    }
    axes[num_codomain..source_rank].reverse();
    probe(tree_operation_view(
        TreeTransformOperationKind::Transpose,
        &axes[..num_codomain],
        &axes[num_codomain..source_rank],
    ));
}

/// The borrowed exact-layout probe of a permutation request without levels.
pub(super) fn tree_operation_view<'a>(
    kind: TreeTransformOperationKind,
    codomain_axes: &'a [usize],
    domain_axes: &'a [usize],
) -> tenet_tensors::TreeTransformOperationView<'a> {
    tenet_tensors::TreeTransformOperationView::new(kind, codomain_axes, domain_axes, &[], &[])
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
    /// the source, or one whose length does not match the result,
    /// `SpaceMismatch` for one whose space or block layout does not; [`Error::DestinationShared`] when `destination` shares its
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
            |_, _, probe| {
                probe(tree_operation_view(
                    TreeTransformOperationKind::Permute,
                    codomain_axes,
                    domain_axes,
                ))
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
        let probe_operation = operation.clone();
        self.tree_transform_into(
            destination,
            alpha,
            beta,
            |_, _| Ok(operation),
            |_, _, probe| {
                probe(tenet_tensors::TreeTransformOperationView::of(
                    &probe_operation,
                ))
            },
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
            |_, _, probe| {
                probe(tree_operation_view(
                    TreeTransformOperationKind::Transpose,
                    codomain_axes,
                    domain_axes,
                ))
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
            |source, destination, probe| {
                if destination.rank() == source.rank() {
                    repartition_probe(
                        source_codomain_rank,
                        source_rank,
                        destination_codomain_rank,
                        probe,
                    );
                }
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
        exact_layout_probe: impl FnOnce(
            &Self,
            &Self,
            &mut dyn FnMut(tenet_tensors::TreeTransformOperationView<'_>),
        ),
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

        // A destination proved once to be this operation's result space of
        // this source hits the completed transformer directly: the owned
        // operation and the result space are not rebuilt.
        let mut exact_layout_hit = None;
        exact_layout_probe(self, destination, &mut |view| {
            exact_layout_hit = crate::runtime::Runtime::exact_layout_tree_pair_hit(
                &identity,
                view,
                &source_body.space,
                &destination_body.space,
            );
        });
        let operation = match exact_layout_hit {
            Some(_) => None,
            None => {
                let operation = operation(self, destination)?;
                let expected = source_body
                    .space
                    .transformed_multiplicity_free(&operation)?;
                if destination_body.space.space() != expected.space() {
                    return Err(Error::from(tenet_tensors::OperationError::SpaceMismatch {
                        message: "destination fusion space or block layout does not match the operation result",
                    }));
                }
                Some(operation)
            }
        };
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
            match (&exact_layout_hit, &operation) {
                (Some(structure), _) => context.replay_tree_transform_dyn_into(
                    structure,
                    destination_structure,
                    source_structure,
                    destination_data.as_mut_slice(),
                    source_data,
                    alpha,
                    beta,
                )?,
                (None, Some(operation)) => context.tree_transform_dyn_into_ref(
                    destination_provider,
                    operation,
                    destination_structure,
                    source_structure,
                    destination_data.as_mut_slice(),
                    source_data,
                    alpha,
                    beta,
                )?,
                (None, None) => unreachable!("a miss builds the operation"),
            }
        }
        if let Some(operation) = &operation {
            // Retention is an optimization: a lookup-only key keeps no proof.
            let _ = crate::runtime::Runtime::admit_exact_tree_pair_layout(
                &identity,
                operation,
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
    /// [`Error::Operation`] with
    /// [`crate::typed::OperationError::InvalidPermutation`] when the axis lists
    /// are malformed (out of range, repeated, or not a partition of `0..rank`),
    /// as at every axis-taking entry point; [`Error::Operation`] /
    /// [`Error::Core`] / [`Error::FusionAlgebra`] when the provider cannot
    /// support the braiding the requested motion needs. The expert layer's own typed errors are the
    /// contract here: re-validating the axes at this layer would be a second
    /// copy of a rule that already exists one call down, free to drift.
    /// Checked Generic providers report an axis misuse in
    /// [`GenericTensorError::Facade`] with the same [`Error`], and other
    /// failures in [`GenericTensorError::Plan`] with the concrete provider error
    /// preserved as its source.
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
        self.tree_transform(TreeTransformOperation::permute(
            codomain_axes.iter().copied(),
            domain_axes.iter().copied(),
        ))
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
    /// Otherwise, as for [`Self::permute`]: [`Error::Operation`] with
    /// [`crate::typed::OperationError::InvalidPermutation`] for malformed axis
    /// lists, and [`Error::Operation`] / [`Error::Core`] /
    /// [`Error::FusionAlgebra`] from the expert layer for a provider that
    /// cannot support the requested braiding; this layer does not re-validate
    /// axes. Checked Generic reports an axis misuse in
    /// [`GenericTensorError::Facade`] and other failures in
    /// [`GenericTensorError::Plan`].
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
        self.tree_transform(TreeTransformOperation::braid(
            codomain_axes.iter().copied(),
            domain_axes.iter().copied(),
            levels[..nout].iter().copied(),
            levels[nout..].iter().copied(),
        ))
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
    /// otherwise, as for [`Self::permute`], [`Error::Operation`] /
    /// [`Error::Core`] / [`Error::FusionAlgebra`] from the expert layer, which
    /// owns the validation this facade passes through; an axis misuse is
    /// [`Error::Operation`] with
    /// [`crate::typed::OperationError::InvalidPermutation`]. Checked Generic
    /// reports an axis misuse in [`GenericTensorError::Facade`] and other
    /// failures in [`GenericTensorError::Plan`].
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
        self.tree_transform(operation)
    }

    /// The one body of every admitted permutation, braid and planar
    /// transpose: the compact rank-(1,1) arm and the mode's lazy-adjoint arm,
    /// else one dense transform published as an owned tensor.
    pub(super) fn tree_transform(
        &self,
        operation: TreeTransformOperation,
    ) -> Result<Self, TypedFacadeError<R>> {
        if let Some(compact) = compact_arms::transform_rank_one_diagonal(self, &operation)? {
            return Ok(compact);
        }
        if let TypedTensorRepr::Adjoint(_) = &self.repr {
            if let Some(lazy) =
                <R::Mode as TypedTensorTransformDispatch<R, D>>::try_lazy_adjoint_transform(
                    self, &operation,
                )?
            {
                return Ok(lazy);
            }
        }
        let (space, data) =
            <R::Mode as TypedTensorTransformDispatch<R, D>>::transform(self, operation)?;
        Ok(self.published(space, data))
    }
}

/// The owned multiplicity-free transform of either coefficient lane: one
/// destination derivation and one replay (or overwrite) into a fresh payload.
#[allow(private_bounds)]
fn tree_transform_multiplicity_free_owned<R, D>(
    tensor: &TensorMap<R, D>,
    operation: TreeTransformOperation,
) -> Result<(BoundDynamicFusionMapSpace<R>, Vec<D>), Error>
where
    R: MultiplicityFreeRigidSymbols + CheckedFusionAlgebra + SectorCodec,
    <R as MultiplicityFreeFusionSymbols>::Scalar:
        CategoricalScalar + tenet_tensors::DenseRecouplingScalar,
    D: TensorScalar
        + crate::runtime::MultiplicityFreeCoefficientLane<
            <R as MultiplicityFreeFusionSymbols>::Scalar,
        >,
{
    // Leasing rather than locking: independent operations on one runtime
    // must not serialize behind each other.
    let mut lease = tensor.runtime.lease_context()?;
    // Why owned: a lazy adjoint takes the lane's lazy arm, and a
    // complex-coefficient rule cannot hold one.
    let body = tensor
        .owned_body()
        .ok_or_else(|| internal_layout_error("multiplicity-free transform input is owned"))?;
    let context = D::lane(lease.context())?;
    let data = body.materialized_dense_data();
    let input = BoundDynamicTensorRef::try_new(&body.space, data.as_ref())?;
    #[cfg(test)]
    crate::tensor_core::observe_tree_transform_seam_call();
    Ok(context
        .tree_context_mut()
        .tree_transform_owned_multiplicity_free_in(
            input.space(),
            input.data(),
            &operation,
            D::from_real(1.0),
        )?)
}

/// The multiplicity-free [`TypedTensorTransformDispatch::transform_bond_spectrum`]
/// of either coefficient lane.
#[allow(private_bounds)]
fn transform_bond_spectrum_multiplicity_free<R, D>(
    tensor: &TensorMap<R, D>,
    spectrum: &[tenet_matrixalgebra::SectorSpectrum<D>],
    operation: &TreeTransformOperation,
    require_representable: bool,
) -> Result<BondTransform<R, D>, Error>
where
    R: MultiplicityFreeRigidSymbols + CheckedFusionAlgebra + SectorCodec,
    <R as MultiplicityFreeFusionSymbols>::Scalar:
        CategoricalScalar + tenet_tensors::DenseRecouplingScalar,
    D: TensorScalar
        + crate::runtime::MultiplicityFreeCoefficientLane<
            <R as MultiplicityFreeFusionSymbols>::Scalar,
        >,
{
    let source = tensor.logical_space();
    let mut lease = tensor.runtime.lease_context()?;
    let (destination, outcome) = D::lane(lease.context())?
        .tree_context_mut()
        .tree_transform_structure_multiplicity_free_in(
            source,
            operation,
            |structure, preview| {
                crate::tensor_core::bond_spectrum(
                    source.space().structure(),
                    preview,
                    structure,
                    spectrum,
                    require_representable,
                )
            },
            || tenet_matrixalgebra::seam::diagonal_bond_data(source.space(), spectrum, &|v| v),
        )?;
    Ok(BondTransform::from_outcome(destination, outcome))
}

impl<R, D> BondTransform<R, D> {
    pub(super) fn from_outcome(
        destination: BoundDynamicFusionMapSpace<R>,
        outcome: tenet_tensors::StructureOutcome<Vec<tenet_matrixalgebra::SectorSpectrum<D>>, D>,
    ) -> Self {
        let output = match outcome {
            tenet_tensors::StructureOutcome::Read(spectrum) => BondOutput::Spectrum(spectrum),
            tenet_tensors::StructureOutcome::Replayed(data) => BondOutput::Dense(data),
        };
        Self {
            destination,
            output,
        }
    }
}

/// TensorKit `permute(t') = adjoint(permute(parent, adjointtensorindices(..)))`
/// and its `braid`/`transpose`/`repartition` twins: the lazy adjoint of the
/// parent's owned transform, one body for every mode. `None` unless `tensor`
/// is a lazy adjoint.
pub(super) fn lazy_adjoint_of_transformed_parent<R, D>(
    tensor: &TensorMap<R, D>,
    operation: &TreeTransformOperation,
) -> Result<Option<TensorMap<R, D>>, TypedFacadeError<R>>
where
    R: TypedSectorAdmission,
    R::Mode: TypedTensorTransformDispatch<R, D> + TypedAdjointSpace<R>,
    D: TensorScalar,
{
    let TypedTensorRepr::Adjoint(view) = &tensor.repr else {
        return Ok(None);
    };
    let parent_space = view.parent.space.space();
    let lowered =
        lower_adjoint_tree_transform_operation(parent_space.nout(), parent_space.nin(), operation)?;
    TensorMap {
        runtime: tensor.runtime.clone(),
        repr: TypedTensorRepr::Owned(Arc::clone(&view.parent)),
    }
    .tree_transform(lowered)?
    .adjoint()
    .map(Some)
}

impl<R, D> MultiplicityFreeTransformExecution<R, f64> for D
where
    R: TypedSectorAdmission<Error = FusionAlgebraError, Mode = MultiplicityFreeAdmissionMode>
        + MultiplicityFreeRigidSymbols<Scalar = f64>
        + CheckedFusionAlgebra
        + SectorCodec,
    D: TensorScalar,
{
    fn transform_bond_spectrum(
        tensor: &TensorMap<R, Self>,
        spectrum: &[tenet_matrixalgebra::SectorSpectrum<Self>],
        operation: &TreeTransformOperation,
        require_representable: bool,
    ) -> Result<BondTransform<R, Self>, Error> {
        transform_bond_spectrum_multiplicity_free(
            tensor,
            spectrum,
            operation,
            require_representable,
        )
    }

    fn try_lazy_adjoint(
        tensor: &TensorMap<R, Self>,
        operation: &TreeTransformOperation,
    ) -> Result<Option<TensorMap<R, Self>>, Error> {
        lazy_adjoint_of_transformed_parent(tensor, operation)
    }

    fn transform(
        tensor: &TensorMap<R, Self>,
        operation: TreeTransformOperation,
    ) -> Result<(BoundDynamicFusionMapSpace<R>, Vec<Self>), Error> {
        tree_transform_multiplicity_free_owned(tensor, operation)
    }
}

impl<R> MultiplicityFreeTransformExecution<R, num_complex::Complex64> for num_complex::Complex64
where
    R: MultiplicityFreeRigidSymbols<Scalar = num_complex::Complex64>
        + CheckedFusionAlgebra
        + SectorCodec,
{
    fn transform_bond_spectrum(
        tensor: &TensorMap<R, Self>,
        spectrum: &[tenet_matrixalgebra::SectorSpectrum<Self>],
        operation: &TreeTransformOperation,
        require_representable: bool,
    ) -> Result<BondTransform<R, Self>, Error> {
        transform_bond_spectrum_multiplicity_free(
            tensor,
            spectrum,
            operation,
            require_representable,
        )
    }

    fn transform(
        tensor: &TensorMap<R, Self>,
        operation: TreeTransformOperation,
    ) -> Result<(BoundDynamicFusionMapSpace<R>, Vec<Self>), Error> {
        tree_transform_multiplicity_free_owned(tensor, operation)
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
    // TensorKit cfaa073e4d1e3eb2167edcbdc3be9872f41e7d91 derives the result
    // space before sharing `t.data` in `insertleftunit` / `insertrightunit`:
    // https://github.com/Jutho/TensorKit.jl/blob/cfaa073e4d1e3eb2167edcbdc3be9872f41e7d91/src/tensors/indexmanipulations.jl#L103-L172
    // Rust's checked provider can still reject the derived layout, so keep it
    // staged until the exact source/destination correspondence is proved.
    let validation_homspace = homspace.clone();
    let prepared = tensor
        .logical_space()
        .prepare_final_homspace_generic_with_checked(provider, homspace)
        .map_err(GenericTensorError::Structure)?;
    validate_unit_layout_correspondence_generic_checked(
        provider,
        (source_hom, tensor.logical_space().space().structure()),
        (&validation_homspace, prepared.structure()),
        insertion,
    )
    .map_err(GenericTensorError::Structure)?;
    let destination = tensor
        .logical_space()
        .commit_final_homspace_generic_bound_checked(prepared)
        .map_err(map_checked_unit_commit_error)?;
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
    // TensorKit cfaa073e4d1e3eb2167edcbdc3be9872f41e7d91 keeps the payload
    // only after deriving the smaller space in `removeunit`:
    // https://github.com/Jutho/TensorKit.jl/blob/cfaa073e4d1e3eb2167edcbdc3be9872f41e7d91/src/tensors/indexmanipulations.jl#L174-L197
    // The checked Rust path additionally delays publication until its layout
    // correspondence with the already-admitted larger source is proved.
    let validation_homspace = homspace.clone();
    let prepared = tensor
        .logical_space()
        .prepare_final_homspace_generic_with_checked(provider, homspace)
        .map_err(GenericTensorError::Structure)?;
    validate_unit_layout_correspondence_generic_checked(
        provider,
        (&validation_homspace, prepared.structure()),
        (source_hom, tensor.logical_space().space().structure()),
        insertion,
    )
    .map_err(GenericTensorError::Structure)?;
    let destination = tensor
        .logical_space()
        .commit_final_homspace_generic_bound_checked(prepared)
        .map_err(map_checked_unit_commit_error)?;
    let data = tensor.shareable_dense_payload();
    Ok(TensorMap {
        runtime: tensor.runtime.clone(),
        repr: owned_repr(TypedTensorBody::with_shared_payload(destination, data)),
    })
}

fn map_checked_unit_commit_error<E>(error: OperationError) -> GenericTensorError<E> {
    match error {
        OperationError::Core(error) => {
            GenericTensorError::Structure(CheckedGenericStructureError::Core(error))
        }
        error => GenericTensorError::from(error),
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

impl<R, D> TensorMap<R, D>
where
    R: TypedSectorAdmission,
    R::Mode: TypedAdjointSpace<R>,
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
        // Why not a lazy view of a compact diagonal: densifying its unstored
        // zeros through conjugation would publish them as `0-0i`.
        if let Some(adjoint) = self.compact_adjoint() {
            return Ok(adjoint);
        }
        Ok(match &self.repr {
            TypedTensorRepr::Owned(parent) => {
                let logical_space =
                    <R::Mode as TypedAdjointSpace<R>>::adjoint_space(&parent.space)?;
                debug_assert!(Arc::ptr_eq(
                    parent.space.provider_arc(),
                    logical_space.provider_arc()
                ));
                Self {
                    runtime: self.runtime.clone(),
                    repr: TypedTensorRepr::Adjoint(Arc::new(TypedAdjointView::new(
                        Arc::clone(parent),
                        logical_space,
                    ))),
                }
            }
            TypedTensorRepr::Adjoint(view) => Self {
                runtime: self.runtime.clone(),
                repr: TypedTensorRepr::Owned(Arc::clone(&view.parent)),
            },
        })
    }

    /// Borrowed adjoint view: the operand [`Self::adjoint`] would give,
    /// passed without building it. See [`TensorRef`].
    pub fn adjoint_view(&self) -> TensorRef<'_, R, D> {
        TensorRef {
            base: self,
            adjoint: Some(|tensor| {
                tensor
                    .adjoint()
                    .map_err(<R::Mode as TypedTensorModeDispatch<R>>::facade_error_into_error)
            }),
        }
    }
}
