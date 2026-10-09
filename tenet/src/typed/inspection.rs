use super::*;

impl<R, D> TensorMap<R, D>
where
    D: TensorScalar,
{
    /// An owned dense copy with this tensor's space, values, provider,
    /// runtime and placement (TensorKit `copy`, or `TensorMap(d)` for a
    /// diagonal).
    ///
    /// The result is neither a lazy adjoint nor a compact diagonal, and its
    /// payload is freshly allocated, so writing to it never changes `self`.
    /// This is the remedy for operations that reject lazy or compact inputs,
    /// such as `*_into` sources and checked-Generic
    /// factorizations. A lazy adjoint is conjugate-transposed from its parent
    /// in one pass.
    ///
    /// Why not named `copy`: TensorKit's `copy(::DiagonalTensorMap)` stays
    /// diagonal, and `Clone` here is a shallow handle copy.
    ///
    /// # Cost
    ///
    /// One payload allocation of `required_len` elements and one pass over
    /// it, plus two fixed body wrappers. A compact diagonal is zero-filled and
    /// then writes its `O(Σ_c k_c)` values.
    ///
    /// ```
    /// use std::sync::Arc;
    /// use tenet::sector::{U1FusionRule, U1Irrep};
    /// use tenet::typed::{GradedSpace, Runtime, TensorMap};
    ///
    /// let runtime = Runtime::builder().build()?;
    /// let v = GradedSpace::try_new(
    ///     Arc::new(U1FusionRule),
    ///     [(U1Irrep::new(0), 2), (U1Irrep::new(1), 1)],
    /// )?;
    /// let t: TensorMap<_, f64> = TensorMap::rand_with_seed(&runtime, [&v], [&v], 7)?;
    /// let adjoint = t.adjoint()?;
    /// let mut owned = adjoint.materialize()?;
    /// assert_eq!(owned.dense_data()?, adjoint.materialize()?.dense_data()?);
    /// owned.scale_assign(2.0)?;
    /// assert_ne!(owned.dense_data()?, adjoint.materialize()?.dense_data()?);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    ///
    /// # Errors
    ///
    /// [`Error::Operation`] only if an engine-internal layout invariant is
    /// broken.
    pub fn materialize(&self) -> Result<Self, Error> {
        let body = match &self.repr {
            TypedTensorRepr::Owned(body) => body,
            TypedTensorRepr::Adjoint(view) => return self.materialize_adjoint(view),
        };
        let data = match body.data.as_ref() {
            TypedData::Dense(data) => data.clone(),
            TypedData::Diagonal(spectrum) => tenet_matrixalgebra::seam::diagonal_bond_data(
                body.space.space(),
                spectrum,
                &|value| value,
            )?,
        };
        Ok(self.with_data(data))
    }

    /// The payload a space-only rewrite (unit-leg insert/remove) may install
    /// in its new body: a dense payload is shared at pointer cost; a lazy
    /// adjoint or a compact spectrum goes through [`Self::materialize`] into
    /// a **fresh** dense payload (one copy) — the #613 Group 4 contract.
    ///
    /// Infallible for the reason [`TypedTensorBody::materialized_dense_data`]
    /// is: the diagonal fill is total on a bond space this module built from
    /// that same spectrum.
    pub(super) fn shareable_dense_payload(&self) -> Arc<TypedData<D>> {
        if let Some(body) = self.owned_body() {
            if matches!(body.data.as_ref(), TypedData::Dense(_)) {
                return Arc::clone(&body.data);
            }
        }
        let materialized = self
            .materialize()
            .expect("a pre-admitted typed tensor must materialize");
        Arc::clone(
            &materialized
                .owned_body()
                .expect("materialize returns an owned body")
                .data,
        )
    }

    /// Builds an operation-local logical tensor: a full receiver-sized
    /// logical payload, released with the operation. Prefer an oriented
    /// kernel or algebraic redirect when one implements the same semantics.
    pub(super) fn materialized_tensor_uncached(&self) -> Result<Self, Error> {
        let TypedTensorRepr::Adjoint(view) = &self.repr else {
            return Ok(self.clone());
        };
        self.materialize_adjoint(view)
    }

    fn materialize_adjoint(&self, view: &TypedAdjointView<R, D>) -> Result<Self, Error> {
        if view.borrowed {
            // Backstop: every refusing operation checks first, by name, with
            // `refuse_borrowed_view`.
            return Err(borrowed_view_unsupported("the operation"));
        }
        #[cfg(test)]
        observe_adjoint_materialization();
        let _host_pool = self.runtime.enter_host_pool();
        #[cfg(test)]
        observe_materialization_pool();
        #[cfg(test)]
        UNCACHED_ADJOINT_MATERIALIZATIONS
            .set(UNCACHED_ADJOINT_MATERIALIZATIONS.get().saturating_add(1));
        let data = tenet_tensors::materialize_adjoint_data_dyn(
            view.parent.space.space(),
            view.logical_space.space(),
            view.parent_data(),
        )?;
        Ok(Self {
            runtime: self.runtime.clone(),
            repr: owned_repr(TypedTensorBody::dense(view.logical_space.clone(), data)),
        })
    }
}

impl<R, D, S> TensorMap<R, D, S>
where
    S: HostReadableStorage<D>,
{
    /// Whole reduced payload in the tensor's logical coupled-sector layout,
    /// borrowed from dense Host storage without copying.
    ///
    /// These are fusion-tree-indexed reduced block entries, not entries in the
    /// physical carrier basis. Use [`Self::to_physical_dense`] when the
    /// provider implements [`PhysicalFusionBasis`] and physical entries are
    /// required.
    ///
    /// ```
    /// use std::sync::Arc;
    ///
    /// use tenet::sector::{U1FusionRule, U1Irrep};
    /// use tenet::typed::{Error, GradedSpace, Runtime, TensorMap};
    ///
    /// let runtime = Runtime::builder().build()?;
    /// let v = GradedSpace::try_new(Arc::new(U1FusionRule), [(U1Irrep::new(0), 2)])?;
    /// let t: TensorMap<_, f64> = TensorMap::rand_with_seed(&runtime, [&v], [&v], 1)?;
    /// assert_eq!(t.dense_data()?.len(), 4);
    /// let adjoint = t.adjoint()?;
    /// assert!(matches!(adjoint.dense_data(), Err(Error::Unsupported { .. })));
    /// assert_eq!(adjoint.materialize()?.dense_data()?.len(), 4);
    /// # Ok::<(), Error>(())
    /// ```
    ///
    /// # Errors
    ///
    /// [`Error::Unsupported`] with [`crate::error::Alternative::Materialize`]
    /// for a lazy adjoint or a compact diagonal, whose entries are not stored
    /// densely; call [`Self::materialize`] first. Never copies.
    pub fn dense_data(&self) -> Result<&[D], Error> {
        match &self.repr {
            TypedTensorRepr::Owned(body) => match &*body.data {
                TypedData::Dense(data) => Ok(data.as_slice()),
                TypedData::Diagonal(_) => Err(borrowed_view_unsupported("dense_data")),
            },
            TypedTensorRepr::Adjoint(_) => Err(borrowed_view_unsupported("dense_data")),
        }
    }
}

impl<R, D, S> TypedTensorBody<R, D, S>
where
    D: TensorScalar,
    S: HostReadableStorage<D>,
{
    /// The payload in the dense coupled layout: borrowed for a dense payload,
    /// densified into an operation-local buffer for a compact diagonal.
    ///
    /// Why not cached in the body: a retained densification is an implicit
    /// receiver-sized copy the caller never asked for (#1548). Every caller
    /// that reaches the compact arm documents the copy as part of its cost.
    pub(super) fn materialized_dense_data(&self) -> std::borrow::Cow<'_, [D]> {
        match &*self.data {
            TypedData::Dense(data) => std::borrow::Cow::Borrowed(data.as_slice()),
            TypedData::Diagonal(spectrum) => {
                #[cfg(test)]
                DIAGONAL_MATERIALIZATIONS.set(DIAGONAL_MATERIALIZATIONS.get().saturating_add(1));
                #[cfg(test)]
                observe_materialization_pool();
                std::borrow::Cow::Owned(
                    tenet_matrixalgebra::seam::diagonal_bond_data(
                        self.space.space(),
                        spectrum,
                        &|value| value,
                    )
                    .expect("diagonal fill is total on the stored bond space"),
                )
            }
        }
    }
}

impl<R, D, S> TypedAdjointView<R, D, S>
where
    S: HostReadableStorage<D>,
{
    /// The parent payload, which is dense by the [`TypedAdjointView::new`]
    /// invariant.
    pub(super) fn parent_data(&self) -> &[D] {
        match &*self.parent.data {
            TypedData::Dense(data) => data.as_slice(),
            TypedData::Diagonal(_) => {
                unreachable!("TypedAdjointView::new admits only dense parents")
            }
        }
    }
}

impl<R, D> TensorMap<R, D>
where
    R: TypedSectorAdmission,
    R::Mode: TypedTensorConstructionDispatch<R, D>,
    D: TensorScalar,
{
    /// Builds a compact diagonal map `bond <- bond` from labelled sector values.
    ///
    /// TeNeT's typed counterpart of TensorKit `DiagonalTensorMap` / `diagm`
    /// stores `O(Σ_c k_c)` values. Input labels may be permuted, but must name
    /// every nonzero bond sector exactly once; output is canonicalized to the
    /// bond's engine-sector order. Each vector must equal that sector's
    /// degeneracy. All validation precedes checked layout admission, and the
    /// supplied dual flag is preserved.
    pub fn diagonal<I>(
        runtime: &Runtime,
        bond: &GradedSpace<R>,
        spectra: I,
    ) -> Result<Self, TypedFacadeError<R>>
    where
        I: IntoIterator<Item = SectorSpectrum<R::Sector, D>>,
    {
        let spectra: Vec<_> = spectra.into_iter().collect();
        let mut labels: Vec<_> = spectra.iter().map(|entry| &entry.sector).collect();
        labels.sort_unstable();
        if let Some(duplicate) = labels.windows(2).find(|pair| pair[0] == pair[1]) {
            return Err(Error::InvalidArgument(format!(
                "sector label {:?} is declared more than once",
                duplicate[0]
            ))
            .into());
        }

        let mut encoded = Vec::with_capacity(spectra.len());
        for entry in &spectra {
            encoded.push(
                TypedSectorAdmission::try_encode_label(bond.provider(), &entry.sector)
                    .map_err(<R::Mode as TypedTensorModeDispatch<R>>::map_provider_error)?,
            );
        }
        let mut by_id: Vec<_> = encoded.iter().enumerate().collect();
        by_id.sort_unstable_by_key(|(_, id)| *id);
        if let Some(duplicate) = by_id.windows(2).find(|pair| pair[0].1 == pair[1].1) {
            return Err(Error::InvalidArgument(format!(
                "SectorCodec law violation: labels {:?} and {:?} both encode to {:?}",
                spectra[duplicate[0].0].sector, spectra[duplicate[1].0].sector, duplicate[0].1
            ))
            .into());
        }
        let mut supplied: HashMap<_, _> = encoded
            .into_iter()
            .zip(spectra)
            .map(|(sector, entry)| (sector, entry.values))
            .collect();
        if bond
            .leg()
            .sectors()
            .iter()
            .any(|sector| !supplied.contains_key(sector))
        {
            return Err(Error::InvalidArgument(
                "diagonal spectrum is missing a bond sector".into(),
            )
            .into());
        }
        if supplied.len() != bond.leg().sectors().len() {
            return Err(Error::InvalidArgument(
                "diagonal spectrum contains an unknown bond sector".into(),
            )
            .into());
        }
        let mut spectrum = Vec::with_capacity(bond.degeneracies().len());
        for (&sector, &degeneracy) in bond.leg().sectors().iter().zip(bond.degeneracies()) {
            let values = supplied
                .remove(&sector)
                .expect("complete checked spectrum contains every bond sector");
            if values.len() != degeneracy {
                return Err(Error::InvalidArgument(
                    "diagonal spectrum length does not match bond degeneracy".into(),
                )
                .into());
            }
            spectrum.push(tenet_matrixalgebra::SectorSpectrum { sector, values });
        }
        let homspace = FusionTreeHomSpace::new(
            FusionProductSpace::new([bond.leg().clone()]),
            FusionProductSpace::new([bond.leg().clone()]),
        );
        let space = <R::Mode as TypedTensorConstructionDispatch<R, D>>::build_construction_root(
            Arc::clone(bond.provider_arc()),
            homspace,
        )?;
        Ok(Self {
            runtime: runtime.clone(),
            repr: owned_repr(TypedTensorBody::diagonal(space, spectrum)),
        })
    }
}

impl<R, D> TensorMap<R, D>
where
    R: TypedSectorAdmission,
    R::Mode: TypedTensorRootDispatch<R>,
    D: TensorScalar,
{
    /// The leg of a `bond <- bond` endomorphism, after proving that is what
    /// the receiver is.
    fn require_bond_map(&self, operation: &str) -> Result<(), TypedFacadeError<R>> {
        let homspace = self.logical_space().space().homspace();
        if homspace.codomain().len() != 1 || homspace.domain().len() != 1 {
            return Err(Error::InvalidArgument(format!(
                "{operation} requires a rank-(1,1) bond <- bond map, got rank {}|{}",
                homspace.codomain().len(),
                homspace.domain().len()
            ))
            .into());
        }
        Ok(())
    }

    /// Returns the per-coupled-sector diagonal of a one-leg `W_out <- W_in`
    /// map.
    ///
    /// MatrixAlgebraKit's `diagview`, and the spectrum reader for a factor that
    /// is diagonal by construction but not stored compactly — an explicitly
    /// materialized compact-SVD `s`, the rectangular `s` of
    /// [`Self::svd_full`], or a device `s`/`d` brought back with `to_host`.
    /// A rectangular sector block `m_c x n_c` contributes its leading
    /// `min(m_c, n_c)` diagonal entries; a sector present on only one leg has
    /// no block and no entry ([`GradedSpace::extend_spectrum`] pads such a
    /// spectrum to a full leg). Compact storage is cloned; dense storage is read one strided
    /// diagonal per block.
    /// Off-diagonal entries are never inspected, so this returns the diagonal
    /// of an arbitrary endomorphism, not a proof that it is diagonal — use
    /// [`crate::expert::is_diagonal`] for that.
    ///
    /// [`crate::expert::diagonal_spectrum`] is a different question: it
    /// answers whether the payload *is* compact.
    ///
    /// # Complexity
    ///
    /// `O(sum_c k_c)` reads and one output allocation per sector; no dense
    /// block is materialized and no payload-sized copy is made.
    ///
    /// # Errors
    ///
    /// [`Error::InvalidArgument`] for a lazy adjoint (adjoin the result of the
    /// owned parent, or read [`Self::materialize`]'s result, instead), for a
    /// receiver that is not a rank-(1,1) map, and
    /// for a layout whose blocks are not fusion-tree keyed.
    ///
    /// Each sector is decoded to its provider label. A provider that cannot
    /// decode one returns its own error: for a checked Generic provider that
    /// is [`GenericTensorError::Structure`] wrapping
    /// [`CheckedGenericStructureError::Provider`],
    /// and for a multiplicity-free provider [`Error::FusionAlgebra`]. No
    /// partial spectrum is returned. In a truncated factorization this is the
    /// step where a decode failure surfaces, before `find_truncated`.
    pub fn diagview(&self) -> Result<Vec<SectorSpectrum<R::Sector, D>>, TypedFacadeError<R>> {
        let body = self.owned_body().ok_or_else(|| {
            TypedFacadeError::<R>::from(Error::InvalidArgument(
                "diagview requires an owned tensor, not a lazy adjoint".to_string(),
            ))
        })?;
        self.require_bond_map("diagview")?;
        let raw: Vec<tenet_matrixalgebra::SectorSpectrum<D>> = match body.data.as_ref() {
            TypedData::Diagonal(spectrum) => spectrum.clone(),
            TypedData::Dense(_) => {
                let structure = self.logical_space().space().structure();
                let payload = body.materialized_dense_data();
                let data: &[D] = &payload;
                let mut collected = Vec::with_capacity(structure.block_count());
                for index in 0..structure.block_count() {
                    let block = structure
                        .block(index)
                        .map_err(Error::from)
                        .map_err(TypedFacadeError::<R>::from)?;
                    let BlockKey::FusionTree(key) = block.key() else {
                        return Err(Error::InvalidArgument(
                            "diagview requires a fusion-tree block layout".to_string(),
                        )
                        .into());
                    };
                    let offset = block.offset();
                    let step = block.strides()[0] + block.strides()[1];
                    let count = block.shape()[0].min(block.shape()[1]);
                    collected.push(tenet_matrixalgebra::SectorSpectrum {
                        sector: key.codomain_tree().coupled(),
                        values: (0..count)
                            .map(|step_index| data[offset + step_index * step])
                            .collect(),
                    });
                }
                // Canonical bond-sector order, as compact storage already keeps
                // it, so the two arms are interchangeable for the caller.
                collected.sort_unstable_by_key(|entry| entry.sector);
                if let Some(pair) = collected
                    .windows(2)
                    .find(|pair| pair[0].sector == pair[1].sector)
                {
                    return Err(Error::InvalidArgument(format!(
                        "diagview: coupled sector {:?} names more than one block",
                        pair[0].sector
                    ))
                    .into());
                }
                collected
            }
        };
        raw.into_iter()
            .map(|entry| {
                Ok(SectorSpectrum {
                    sector: TypedSectorAdmission::try_decode_label(
                        self.logical_space().provider(),
                        entry.sector,
                    )
                    .map_err(<R::Mode as TypedTensorModeDispatch<R>>::map_provider_error)?,
                    values: entry.values,
                })
            })
            .collect()
    }

    /// Returns the compact diagonal spectrum without materializing dense data.
    ///
    /// This is typed `diag` readback: [`None`] means the representation has no
    /// directly stored compact spectrum (it is dense or a lazy adjoint);
    /// otherwise it clones only the `O(Σ_c k_c)` compact values in canonical
    /// bond-sector order.
    #[expect(
        clippy::type_complexity,
        reason = "the expert compact readback exposes provider-labelled sector spectra"
    )]
    pub(crate) fn diagonal_spectrum(
        &self,
    ) -> Result<Option<Vec<SectorSpectrum<R::Sector, D>>>, TypedFacadeError<R>> {
        self.spectrum()
            .map(|spectrum| {
                spectrum
                    .iter()
                    .map(|entry| {
                        Ok(SectorSpectrum {
                            sector: TypedSectorAdmission::try_decode_label(
                                self.logical_space().provider(),
                                entry.sector,
                            )
                            .map_err(<R::Mode as TypedTensorModeDispatch<R>>::map_provider_error)?,
                            values: entry.values.clone(),
                        })
                    })
                    .collect()
            })
            .transpose()
    }
}

impl<R, D> TensorMap<R, D>
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64> + CheckedFusionAlgebra + SectorCodec,
    D: TensorScalar,
{
    /// Tests a rank-one map for blockwise diagonality without materializing compact storage.
    ///
    /// This matches TensorKit `isdiag` for finite data at `tol = 0`; positive
    /// tolerance uses `max_offdiag <= tol * max(norm(Inf), 1)`. Negative and
    /// non-finite tolerances are rejected before every shortcut.
    ///
    /// Scale `tol` to the payload dtype, as [`FactorizationScalar`] describes.
    pub(crate) fn is_diagonal(&self, tol: f64) -> Result<bool, Error> {
        if !tol.is_finite() || tol < 0.0 {
            return Err(Error::InvalidArgument(
                "diagonal tolerance must be finite and nonnegative".into(),
            ));
        }
        if self.spectrum().is_some() {
            return Ok(true);
        }
        if self.rank() != 2 || self.codomain_rank() != 1 || self.domain_rank() != 1 {
            return Ok(false);
        }
        let materialized = self.materialized_tensor_uncached()?;
        let data_payload = materialized
            .owned_body()
            .expect("uncached materialization is owned")
            .materialized_dense_data();
        let data: &[D] = &data_payload;
        let mut norm = 0.0_f64;
        let mut offdiag = 0.0_f64;
        for index in 0..self.logical_space().space().structure().block_count() {
            let block = self.logical_space().space().structure().block(index)?;
            for row in 0..block.shape()[0] {
                for col in 0..block.shape()[1] {
                    let value = data
                        [block.offset() + row * block.strides()[0] + col * block.strides()[1]]
                        .widen_complex()
                        .norm();
                    norm = norm.max(value);
                    if row != col {
                        if !value.is_finite() {
                            return Ok(false);
                        }
                        offdiag = offdiag.max(value);
                    }
                }
            }
        }
        Ok(offdiag <= tol * norm.max(1.0))
    }
}

impl<R, D> TensorMap<R, D>
where
    R: TypedSectorAdmission,
    R::Mode: TypedTensorModeDispatch<R>,
    D: TensorScalar,
{
    /// Provider-labelled fusion trees and borrowed values for every fusion-tree
    /// subblock: TensorKit `subblocks(t)`. For coupled-sector matrices use
    /// [`Self::blocks`].
    ///
    /// All labels are decoded and all views are validated before the iterator is
    /// returned, so iteration is infallible. In particular, a checked-provider
    /// decode failure exposes no prefix. Blocks remain in canonical stored order.
    ///
    /// Copies no numeric payload.
    ///
    /// # Errors
    ///
    /// [`Error::Unsupported`] with [`crate::error::Alternative::Materialize`]
    /// for a lazy adjoint or a compact diagonal, whose subblocks are not
    /// stored densely; call [`Self::materialize`] first. Otherwise an error
    /// if the provider cannot decode a stored fusion-tree label, or if stored
    /// block metadata does not form a valid view into the logical tensor
    /// data.
    #[expect(
        clippy::type_complexity,
        reason = "the public block iterator yields labelled trees with borrowed block views"
    )]
    pub fn subblocks<'a>(
        &'a self,
    ) -> Result<
        impl ExactSizeIterator<Item = (BlockFusionTrees<R::Sector>, BlockView<'a, D>)> + 'a,
        TypedFacadeError<R>,
    > {
        let data = self
            .dense_data()
            .map_err(|_| borrowed_view_unsupported("subblocks"))
            .map_err(TypedFacadeError::<R>::from)?;
        let structure = self.logical_space().space().structure();
        let mut labelled = Vec::with_capacity(structure.block_count());
        for index in 0..structure.block_count() {
            let block = structure
                .block(index)
                .map_err(Error::from)
                .map_err(TypedFacadeError::<R>::from)?;
            let trees = decode_block_fusion_trees(self.logical_space().provider(), block.key())?;
            labelled.push((trees, block));
        }

        let blocks = labelled
            .into_iter()
            .map(|(trees, block)| {
                BlockView::new(data, block.shape(), block.strides(), block.offset())
                    .map(|values| (trees, values))
                    .map_err(Error::from)
                    .map_err(TypedFacadeError::<R>::from)
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok(blocks.into_iter())
    }

    /// Provider-labelled fusion trees of one subblock, by its index in
    /// [`Self::subblocks`] order.
    pub fn subblock_fusion_trees(
        &self,
        index: usize,
    ) -> Result<BlockFusionTrees<R::Sector>, TypedFacadeError<R>> {
        let block = self
            .logical_space()
            .space()
            .structure()
            .block(index)
            .map_err(Error::from)
            .map_err(TypedFacadeError::<R>::from)?;
        decode_block_fusion_trees(self.logical_space().provider(), block.key())
    }
}

impl<R, D, S> TensorMap<R, D, S>
where
    R: TypedSectorAdmission,
    R::Mode: TypedTensorModeDispatch<R>,
    S: TensorStorage<D>,
{
    /// The matrix of coupled sector `coupled`: TensorKit `block(t, c)`.
    ///
    /// A borrowed view on Host and CUDA storage alike; nothing is copied,
    /// transferred or materialized. A lazy adjoint yields its parent's region
    /// flagged as conjugate-transposed, and a compact diagonal its stored
    /// values, as TensorKit returns `block(parent, c)'` and `Diagonal(view)`.
    /// Row and column order are described on [`CoupledBlock`].
    ///
    /// A sector the tensor stores no block for gives TensorKit's empty view:
    /// `blockdim(codomain, c) × blockdim(domain, c)`, where one of the two is
    /// zero.
    ///
    /// # Complexity
    ///
    /// `O(log C)` for `C` stored coupled sectors, after the per-structure
    /// region table is compiled once (`O(subblocks)`, cached and shared with
    /// [`Self::tr`] and [`Self::inner`]). An absent sector additionally folds
    /// the fused dimensions of both sides, `O(legs · sectors · channels)`.
    ///
    /// # Errors
    ///
    /// Returns the provider's error when it cannot encode `coupled`.
    pub fn block(
        &self,
        coupled: &R::Sector,
    ) -> Result<CoupledBlock<'_, R, D, S>, TypedFacadeError<R>> {
        let provider = self.logical_space().provider();
        let id = TypedSectorAdmission::try_encode_label(provider, coupled)
            .map_err(<R::Mode as TypedTensorModeDispatch<R>>::map_provider_error)?;
        let regions = self.stored_sector_regions()?;
        match regions.binary_search_by_key(&id, CoupledSectorRegion::coupled) {
            Ok(index) => self
                .coupled_block(regions, index)
                .map_err(TypedFacadeError::<R>::from),
            Err(_) => {
                let homspace = self.logical_space().space().homspace();
                let rows = <R::Mode as TypedTensorModeDispatch<R>>::coupled_block_dimension(
                    provider,
                    homspace.codomain(),
                    id,
                )?;
                let cols = <R::Mode as TypedTensorModeDispatch<R>>::coupled_block_dimension(
                    provider,
                    homspace.domain(),
                    id,
                )?;
                // A coupled sector nonempty on both sides always has a stored
                // region, so a miss with two nonzero dimensions is a broken layout.
                if rows != 0 && cols != 0 {
                    return Err(TypedFacadeError::<R>::from(internal_layout_error(
                        "a sector fused on both sides has no stored block",
                    )));
                }
                let payload = match self.storage_body().data.as_ref() {
                    TypedData::Dense(storage) => CoupledBlockPayload::Dense {
                        storage,
                        offset: 0,
                        adjoint: matches!(self.repr, TypedTensorRepr::Adjoint(_)),
                    },
                    TypedData::Diagonal(_) => CoupledBlockPayload::Diagonal(&[]),
                };
                Ok(CoupledBlock {
                    rows,
                    cols,
                    payload,
                    provider,
                    region: None,
                })
            }
        }
    }

    /// Every stored coupled sector with its matrix: TensorKit `blocks(t)`.
    ///
    /// Sectors come in ascending [`crate::sector::SectorId`] order, the storage
    /// order [`GradedSpace::sectors`] also uses, which is not TensorKit's
    /// `blocksectors` order in general. Each view is the one [`Self::block`]
    /// returns. All labels are decoded before the iterator is returned, so
    /// iteration is infallible.
    ///
    /// # Errors
    ///
    /// Fails when the provider cannot decode a stored coupled sector.
    #[expect(
        clippy::type_complexity,
        reason = "the public block iterator yields labelled sectors with borrowed matrix views"
    )]
    pub fn blocks<'a>(
        &'a self,
    ) -> Result<
        impl ExactSizeIterator<Item = (R::Sector, CoupledBlock<'a, R, D, S>)> + 'a,
        TypedFacadeError<R>,
    > {
        let regions = self.stored_sector_regions()?;
        let provider = self.logical_space().provider();
        let blocks = (0..regions.len())
            .map(|index| {
                let sector =
                    TypedSectorAdmission::try_decode_label(provider, regions[index].coupled())
                        .map_err(<R::Mode as TypedTensorModeDispatch<R>>::map_provider_error)?;
                let block = self
                    .coupled_block(Arc::clone(&regions), index)
                    .map_err(TypedFacadeError::<R>::from)?;
                Ok((sector, block))
            })
            .collect::<Result<Vec<_>, TypedFacadeError<R>>>()?;
        Ok(blocks.into_iter())
    }

    /// Coupled regions of the stored payload's own layout. For a lazy adjoint
    /// that is the parent's layout, whose sector `c` holds `block(t', c)'`.
    fn stored_sector_regions(&self) -> Result<Arc<[CoupledSectorRegion]>, TypedFacadeError<R>> {
        let space = self.storage_body().space.space();
        let regions =
            sector_regions(space.structure(), space.nout()).map_err(TypedFacadeError::<R>::from)?;
        debug_assert!(
            regions
                .windows(2)
                .all(|pair| pair[0].coupled() < pair[1].coupled()),
            "coupled regions are sorted by sector id"
        );
        Ok(regions)
    }

    fn coupled_block(
        &self,
        regions: Arc<[CoupledSectorRegion]>,
        index: usize,
    ) -> Result<CoupledBlock<'_, R, D, S>, Error> {
        let region = &regions[index];
        let adjoint = matches!(self.repr, TypedTensorRepr::Adjoint(_));
        let (rows, cols) = if adjoint {
            (region.cols(), region.rows())
        } else {
            (region.rows(), region.cols())
        };
        let payload = match self.storage_body().data.as_ref() {
            TypedData::Dense(storage) => {
                if region.range().end > storage.len() {
                    return Err(internal_layout_error("coupled region outside the payload"));
                }
                CoupledBlockPayload::Dense {
                    storage,
                    offset: region.range().start,
                    adjoint,
                }
            }
            // Every compact constructor stores the spectrum in bond-leg id
            // order, the order of the bond space's regions.
            TypedData::Diagonal(spectrum) => {
                let values = spectrum
                    .binary_search_by_key(&region.coupled(), |entry| entry.sector)
                    .map(|index| spectrum[index].values.as_slice())
                    .ok()
                    .filter(|values| values.len() == rows && rows == cols)
                    .ok_or_else(|| {
                        internal_layout_error("compact spectrum disagrees with its bond space")
                    })?;
                CoupledBlockPayload::Diagonal(values)
            }
        };
        Ok(CoupledBlock {
            rows,
            cols,
            payload,
            provider: self.logical_space().provider(),
            region: Some((regions, index, adjoint)),
        })
    }
}
