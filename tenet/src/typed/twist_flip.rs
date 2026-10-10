use super::*;

impl<R, D> TensorMap<R, D>
where
    R: TypedSectorAdmission,
    R::Mode: TypedTensorTwistDispatch<R, D>,
    D: TensorScalar,
{
    /// TensorKit `twist(t, inds; inv)` (and its in-place `twist!`): multiplies
    /// each fusion-tree block by the product over `legs` (flat leg indices,
    /// codomain first) of that leg's ribbon-twist eigenvalue, or by its
    /// inverse for [`Direction::Inverse`].
    ///
    /// A bosonic provider, a NoBraiding provider restricted to unit legs, or
    /// selected sectors whose staged twists are all one returns a body-sharing
    /// clone, matching TensorKit's `copy = false` behavior. Otherwise the
    /// operation publishes one fresh scaled payload on the exact admitted
    /// space and provider allocation. A lazy adjoint redirects through its
    /// parent with the inverse operation. Multiplicity-free compact spectra
    /// retain their existing representation-preserving path; a checked-Generic
    /// compact factor is densified into an operation-local buffer first.
    ///
    /// # Errors
    ///
    /// An out-of-range leg is rejected before the empty-list short circuit.
    /// Non-unit NoBraiding legs are invalid. Checked-Generic pivotal failures
    /// retain their typed provider error, and no result is published until all
    /// selected twist values have been staged successfully.
    pub fn twist(&self, legs: &[usize], direction: Direction) -> Result<Self, TypedFacadeError<R>> {
        let inverse = direction.is_inverse();
        let rank = self.rank();
        let name = if inverse { "inverse twist" } else { "twist" };
        if let Some(&leg) = legs.iter().find(|&&leg| leg >= rank) {
            return Err(Error::InvalidArgument(format!(
                "{name} leg {leg} out of range for rank {rank}"
            ))
            .into());
        }
        if legs.is_empty() {
            return Ok(self.clone());
        }
        let provider = self.provider();
        let braiding = <R::Mode as TypedTensorModeDispatch<R>>::braiding_style(provider);
        if braiding == tenet_core::BraidingStyleKind::NoBraiding {
            // A unit leg twists by one; any other has no twist to apply.
            let homspace = self.logical_space().space().homspace();
            let vacuum = <R::Mode as TypedSpaceModeDispatch<R>>::vacuum(provider);
            let nout = homspace.codomain().len();
            for &leg in legs {
                let sectors = if leg < nout {
                    homspace.codomain().legs()[leg].sectors()
                } else {
                    homspace.domain().legs()[leg - nout].sectors()
                };
                if sectors.iter().any(|&sector| sector != vacuum) {
                    return Err(Error::InvalidArgument(format!(
                        "{name} leg {leg} carries non-unit sectors but the fusion rule has no braiding"
                    ))
                    .into());
                }
            }
            return Ok(self.clone());
        }
        if braiding.is_bosonic() {
            return Ok(self.clone());
        }
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
            let twisted_parent = parent.twist_owned(&axes, !inverse)?;
            let TypedTensorRepr::Owned(parent) = &twisted_parent.repr else {
                return Err(internal_layout_error("a parent twist stays owned").into());
            };
            // Why not call `adjoint()`: the original view already owns the
            // exact admitted logical space, and re-deriving it would query
            // the provider after all fallible twist values had been staged.
            return Ok(Self {
                runtime: self.runtime.clone(),
                repr: TypedTensorRepr::Adjoint(Arc::new(TypedAdjointView::new(
                    Arc::clone(parent),
                    view.logical_space.clone(),
                ))),
            });
        }
        self.twist_owned(legs, inverse)
    }

    /// The twist of an owned tensor on its own space: a compact arm, else one
    /// scaled copy of the dense payload unless every twist value is one.
    fn twist_owned(&self, legs: &[usize], inverse: bool) -> Result<Self, TypedFacadeError<R>> {
        if self.spectrum().is_some() {
            if let Some(compact) =
                <R::Mode as TypedTensorTwistDispatch<R, D>>::try_compact_twist(self, legs, inverse)?
            {
                return Ok(compact);
            }
        }
        let nout = self.codomain_rank();
        let space = self.logical_space();
        let Some(theta) = <R::Mode as TypedTensorTwistDispatch<R, D>>::twist_values(
            space.provider(),
            space.space().structure(),
            nout,
            legs,
        )?
        else {
            return Ok(self.clone());
        };
        let mut data = self
            .owned_body()
            .ok_or_else(|| internal_layout_error("twist input is owned"))?
            .materialized_dense_data()
            .as_ref()
            .to_vec();
        scale_blocks_impl(space.space(), &mut data, &|key| match key {
            BlockKey::FusionTree(key) => twist_factor_with_inverse(
                legs.iter()
                    .map(|&leg| theta(uncoupled_sector_of_leg(key, nout, leg)))
                    .product(),
                inverse,
            ),
            _ => 1.0,
        })?;
        Ok(self.published(space.clone(), data))
    }
}

impl<R, D> TensorMap<R, D>
where
    R: TypedSectorAdmission,
    R::Mode: TypedTensorFlipDispatch<R, D>,
    D: TensorScalar,
{
    /// TensorKit `flip(t, I; inv)`:
    /// return a tensor isomorphic to `self` where the duality flag of each
    /// leg in `legs` (flat indices, codomain first; a leg listed twice is
    /// flipped twice, sequentially) is toggled,
    /// `space(t', i) = flip(space(t, i))`. The stored sectors and the block
    /// layout are unchanged; each fusion-tree block picks up the
    /// Z-isomorphism phase of TensorKit's fusion-tree `flip`
    /// per flipped leg with uncoupled sector `a` and pre-flip duality `d`
    /// (χ = Frobenius–Schur phase, θ = ribbon twist; both real for every
    /// rule in scope): codomain leg → `d ? χ·θ : 1`; domain leg →
    /// `d ? χ : θ`.
    ///
    /// Like TensorKit's, this `flip` is *not* an involution: flipping the
    /// same leg twice returns to the original spaces but can scale odd
    /// blocks (e.g. by θ = −1 on fermionic legs); only `flip⁴ = id` in
    /// general. [`Direction::Inverse`] applies the inverse isomorphism.
    ///
    /// One scaled copy of the dense payload into a fresh body, O(len); a
    /// compact spectrum factor materializes first (the flipped space is no
    /// longer a bond space, so the result cannot stay compact). The same
    /// facade narrowings as [`Self::twist`] apply: a lazy dense adjoint
    /// redirects through the parent with the inverse categorical map without
    /// materializing; there is no device arm, and checked Generic follows its
    /// provider-mode dispatch.
    ///
    /// # Errors
    ///
    /// [`Error::InvalidArgument`] when a leg is out of range, before the
    /// empty-list short-circuit; empty `legs` returns an identical clone.
    /// Otherwise [`Error::Operation`] /
    /// [`Error::Core`] from the layout derivation of the toggled hom space.
    /// Checked-Generic admission and pivotal failures retain their typed
    /// [`GenericTensorError`] variants.
    pub fn flip(&self, legs: &[usize], direction: Direction) -> Result<Self, TypedFacadeError<R>> {
        let inverse = direction.is_inverse();
        let rank = self.rank();
        let name = if inverse { "inverse flip" } else { "flip" };
        if let Some(&leg) = legs.iter().find(|&&leg| leg >= rank) {
            return Err(Error::InvalidArgument(format!(
                "{name} leg {leg} out of range for rank {rank}"
            ))
            .into());
        }
        if legs.is_empty() {
            return Ok(self.clone());
        }
        if <R::Mode as TypedTensorModeDispatch<R>>::braiding_style(self.provider())
            == tenet_core::BraidingStyleKind::NoBraiding
        {
            let leg = legs[0];
            return Err(Error::InvalidArgument(format!(
                "{name} leg {leg} needs the twist and Frobenius-Schur coefficients but the fusion rule has no braiding"
            ))
            .into());
        }
        if let TypedTensorRepr::Adjoint(view) = &self.repr {
            let logical_space = self.flip_destination(legs)?.0;
            let parent = Self {
                runtime: self.runtime.clone(),
                repr: TypedTensorRepr::Owned(Arc::clone(&view.parent)),
            };
            let axes = logical_adjoint_axes_to_parent(
                view.parent.space.space().nout(),
                view.parent.space.space().nin(),
                legs,
            );
            let flipped_parent = parent.flip_owned(&axes, !inverse)?;
            let TypedTensorRepr::Owned(parent) = &flipped_parent.repr else {
                return Err(internal_layout_error("a parent flip stays owned").into());
            };
            return Ok(Self {
                runtime: self.runtime.clone(),
                repr: TypedTensorRepr::Adjoint(Arc::new(TypedAdjointView::new(
                    Arc::clone(parent),
                    logical_space,
                ))),
            });
        }
        self.flip_owned(legs, inverse)
    }

    /// The flip of an owned tensor: one scaled copy of its dense payload on
    /// the toggled space.
    fn flip_owned(&self, legs: &[usize], inverse: bool) -> Result<Self, TypedFacadeError<R>> {
        let nout = self.codomain_rank();
        let (space, occurrences) = self.flip_destination(legs)?;
        let data = {
            let values = <R::Mode as TypedTensorFlipDispatch<R, D>>::pivotal_values(
                self.provider(),
                space.space().structure(),
                nout,
                &occurrences,
            )?;
            let mut data = self
                .owned_body()
                .ok_or_else(|| internal_layout_error("flip input is owned"))?
                .materialized_dense_data()
                .as_ref()
                .to_vec();
            scale_blocks_impl(space.space(), &mut data, &|key| match key {
                BlockKey::FusionTree(key) => {
                    flip_block_factor(&values, key, nout, &occurrences, inverse)
                }
                _ => 1.0,
            })?;
            data
        };
        Ok(self.published(space, data))
    }

    /// The admitted duality-toggled space of a flip of `legs`, with each
    /// flip occurrence's pre-flip duality. The stored layout must not change.
    #[expect(
        clippy::type_complexity,
        reason = "the flip destination carries its occurrence metadata"
    )]
    fn flip_destination(
        &self,
        legs: &[usize],
    ) -> Result<(BoundDynamicFusionMapSpace<R>, Vec<(usize, bool)>), TypedFacadeError<R>> {
        let (homspace, occurrences) =
            flip_toggled_homspace(self.logical_space().space().homspace(), legs);
        let space =
            <R::Mode as TypedTensorFlipDispatch<R, D>>::root(self.logical_space(), homspace)?;
        check_flip_layout_identity(
            self.logical_space().space().structure(),
            space.space().structure(),
        )?;
        Ok((space, occurrences))
    }
}
