use super::*;

impl<R, D> TensorMap<R, D>
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64> + CheckedFusionAlgebra + SectorCodec,
    D: TensorScalar,
{
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
}

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
        <R::Mode as TypedTensorTwistDispatch<R, D>>::twist(self, legs, direction.is_inverse())
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
        <R::Mode as TypedTensorFlipDispatch<R, D>>::flip(self, legs, direction.is_inverse())
    }
}
