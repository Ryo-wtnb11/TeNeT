use super::*;

impl<R, D> TensorMap<R, D>
where
    R: TypedSectorAdmission,
    R::Mode: TypedTensorConstructionDispatch<R, D>,
    D: TensorScalar,
{
    /// Returns the first leg's provider allocation after proving that every
    /// leg has the same categorical identity.
    fn authority<'a>(legs: &[&'a GradedSpace<R>]) -> Result<&'a Arc<R>, TypedFacadeError<R>> {
        let (first, rest) = legs.split_first().ok_or_else(|| {
            TypedFacadeError::<R>::from(Error::InvalidArgument(
                "at least one leg is required to infer the fusion provider".into(),
            ))
        })?;
        // Equal identities, rather than pointer equality, let separately
        // allocated providers interoperate. This guard precedes every provider
        // query and layout-admission stage.
        let expected_identity = TypedSectorAdmission::typed_rule_identity(first.provider());
        for leg in rest {
            let actual_identity = TypedSectorAdmission::typed_rule_identity(leg.provider());
            if actual_identity != expected_identity {
                return Err(TypedFacadeError::<R>::from(Error::RuleMismatch));
            }
        }
        Ok(first.provider_arc())
    }

    /// Validation half of [`Self::build`]: admits the complete bound layout
    /// without touching payload or runtime RNG state.
    fn build_space(
        provider: Arc<R>,
        codomain: &[&GradedSpace<R>],
        domain: &[&GradedSpace<R>],
    ) -> Result<BoundDynamicFusionMapSpace<R>, TypedFacadeError<R>> {
        let homspace = FusionTreeHomSpace::new(
            FusionProductSpace::new(codomain.iter().map(|leg| leg.leg().clone())),
            FusionProductSpace::new(domain.iter().map(|leg| leg.leg().clone())),
        );
        <R::Mode as TypedTensorConstructionDispatch<R, D>>::build_construction_root(
            provider, homspace,
        )
    }

    /// Payload half of [`Self::build`]: fills only a fully admitted layout and
    /// validates its final data length before publication.
    fn fill_space(
        runtime: &Runtime,
        space: BoundDynamicFusionMapSpace<R>,
        fill: Fill<'_, D>,
    ) -> Result<Self, TypedFacadeError<R>> {
        let data = apply_fill(space.space(), fill).map_err(TypedFacadeError::<R>::from)?;
        BoundDynamicTensorRef::try_new(&space, &data)
            .map_err(Error::from)
            .map_err(TypedFacadeError::<R>::from)?;
        Ok(Self {
            runtime: runtime.clone(),
            repr: owned_repr(TypedTensorBody::dense(space, data)),
        })
    }

    fn build(
        runtime: &Runtime,
        provider: Arc<R>,
        codomain: &[&GradedSpace<R>],
        domain: &[&GradedSpace<R>],
        fill: Fill<'_, D>,
    ) -> Result<Self, TypedFacadeError<R>> {
        let space = Self::build_space(provider, codomain, domain)?;
        Self::fill_space(runtime, space, fill)
    }

    /// Zero tensor map on `codomain <- domain` (TensorKit `zeros(T, W <- V)`).
    ///
    /// Every leg must have the same rule identity; the first leg's exact
    /// provider allocation becomes the tensor's authority. Identity mismatch
    /// is reported before provider algebra is queried. Layout admission and
    /// payload validation are transactional: failure publishes no tensor.
    ///
    /// # Complexity
    ///
    /// One fusion-tree layout admission plus one `O(stored_len)` zeroed
    /// payload allocation.
    ///
    /// ```compile_fail
    /// use std::sync::Arc;
    /// use tenet::sector::{FibonacciFusionRule, FibonacciSector};
    /// use tenet::typed::{GradedSpace, Runtime, TensorMap};
    /// let runtime = Runtime::builder().build().unwrap();
    /// let tau = GradedSpace::try_new(
    ///     Arc::new(FibonacciFusionRule),
    ///     [(FibonacciSector::Tau, 1)],
    /// ).unwrap();
    /// // A complex categorical coefficient cannot act on a real payload.
    /// let _: TensorMap<FibonacciFusionRule, f64> =
    ///     TensorMap::zeros(&runtime, [&tau], [&tau]).unwrap();
    /// ```
    pub fn zeros<'a, Codomain, Domain>(
        runtime: &Runtime,
        codomain: Codomain,
        domain: Domain,
    ) -> Result<Self, TypedFacadeError<R>>
    where
        Codomain: IntoIterator<Item = &'a GradedSpace<R>>,
        Domain: IntoIterator<Item = &'a GradedSpace<R>>,
        R: 'a,
    {
        let codomain: Vec<_> = codomain.into_iter().collect();
        let domain: Vec<_> = domain.into_iter().collect();
        let legs: Vec<_> = codomain.iter().chain(&domain).copied().collect();
        let provider = Arc::clone(Self::authority(&legs)?);
        Self::build(runtime, provider, &codomain, &domain, Fill::Zeros)
    }

    /// Tensor map whose every symmetry-allowed element is produced by
    /// `fill(trees, indices)`, one fusion-tree subblock at a time (the unit
    /// [`Self::subblocks`] reads back; coupled-sector blocks are
    /// [`Self::blocks`]).
    ///
    /// All block keys are decoded exactly once before the first callback.
    /// Therefore a late decode failure invokes `fill` zero times and publishes
    /// neither a partial payload nor a tensor.
    ///
    /// **No TensorKit counterpart.** TensorKit builds an uninitialized,
    /// zeroed, or random tensor and then mutates its blocks; this labelled
    /// callback is TeNeT's semantic-fixture constructor.
    ///
    /// # Complexity
    ///
    /// One layout admission, one decode per stored block, and one callback per
    /// stored element.
    pub fn from_subblock_fn<'a, Codomain, Domain, F>(
        runtime: &Runtime,
        codomain: Codomain,
        domain: Domain,
        mut fill: F,
    ) -> Result<Self, TypedFacadeError<R>>
    where
        Codomain: IntoIterator<Item = &'a GradedSpace<R>>,
        Domain: IntoIterator<Item = &'a GradedSpace<R>>,
        F: FnMut(&BlockFusionTrees<R::Sector>, &[usize]) -> D,
        R: 'a,
    {
        let codomain: Vec<_> = codomain.into_iter().collect();
        let domain: Vec<_> = domain.into_iter().collect();
        let legs: Vec<_> = codomain.iter().chain(&domain).copied().collect();
        let provider = Arc::clone(Self::authority(&legs)?);
        let space = Self::build_space(Arc::clone(&provider), &codomain, &domain)?;
        let structure = space.space().structure();
        let mut labelled = HashMap::with_capacity(structure.block_count());
        for index in 0..structure.block_count() {
            let key = structure
                .block(index)
                .map_err(Error::from)
                .map_err(TypedFacadeError::<R>::from)?
                .key()
                .clone();
            let decoded = decode_block_fusion_trees(provider.as_ref(), &key)?;
            labelled.insert(key, decoded);
        }
        let mut callback = |key: &BlockKey, indices: &[usize]| {
            fill(
                labelled
                    .get(key)
                    .expect("all admitted block keys were decoded before payload fill"),
                indices,
            )
        };
        Self::fill_space(runtime, space, Fill::BlockFn(&mut callback))
    }

    /// Random tensor map using an explicit deterministic splitmix64 seed.
    ///
    /// Every stored real component is uniform on `[-1, 1)`: an `f64` payload
    /// entry uses one draw, while a `Complex64` entry uses independent draws for
    /// its real and imaginary components. This deliberately differs from
    /// TensorKit's default `rand`, whose components are uniform on `[0, 1)`;
    /// pinned source coordinates are recorded in `tenet/references.md`.
    ///
    /// There is no seedless form: TensorKit takes the RNG from the caller, and
    /// the seed is this API's explicit counterpart of that argument.
    ///
    /// Reproducibility is defined for the same TeNeT version and layout;
    /// it is not a cross-library byte-stream contract. Semantic cross-version
    /// fixtures should use [`Self::from_subblock_fn`].
    ///
    /// # Complexity
    ///
    /// One layout admission and one `O(stored_len)` payload allocation/fill.
    pub fn rand_with_seed<'a, Codomain, Domain>(
        runtime: &Runtime,
        codomain: Codomain,
        domain: Domain,
        seed: u64,
    ) -> Result<Self, TypedFacadeError<R>>
    where
        Codomain: IntoIterator<Item = &'a GradedSpace<R>>,
        Domain: IntoIterator<Item = &'a GradedSpace<R>>,
        R: 'a,
    {
        let codomain: Vec<_> = codomain.into_iter().collect();
        let domain: Vec<_> = domain.into_iter().collect();
        let legs: Vec<_> = codomain.iter().chain(&domain).copied().collect();
        let provider = Arc::clone(Self::authority(&legs)?);
        Self::build(runtime, provider, &codomain, &domain, Fill::Rand(seed))
    }
}

impl<R, D> TensorMap<R, D>
where
    R: TypedSectorAdmission,
    R::Mode: TypedTensorConstructionDispatch<R, D>,
    D: TensorScalar,
{
    /// The fused external sector content of one side of a structural
    /// constructor (TensorKit `fuse`, `spaces/gradedspace.jl:150-158`),
    /// weighted by the provider's `N` symbols. Stored sector content is
    /// already external, so duality is dropped.
    fn fused_content(
        provider: &R,
        legs: &[&GradedSpace<R>],
    ) -> Result<Vec<(tenet_core::SectorId, usize)>, TypedFacadeError<R>> {
        let (first, rest) = legs.split_first().ok_or_else(|| {
            // Keep the empty-side failure local to the typed fusion fold.
            Error::InvalidArgument("fuse_all needs at least one space".into())
        })?;
        let mut fused: Vec<(tenet_core::SectorId, usize)> = first.leg().iter().collect();
        for leg in rest {
            let pairs: Vec<(tenet_core::SectorId, usize)> = leg.leg().iter().collect();
            fused = <R::Mode as TypedTensorModeDispatch<R>>::fuse_sector_content(
                provider, &fused, &pairs,
            )?;
        }
        Ok(fused)
    }

    /// Shared body of the structural constructors: checks the fused fit,
    /// builds zeros and writes the (partial) identity into every
    /// coupled-sector matrix.
    fn structural<'a, C, M>(
        runtime: &Runtime,
        codomain: C,
        domain: M,
        embed: bool,
        what: &str,
    ) -> Result<Self, TypedFacadeError<R>>
    where
        C: IntoIterator<Item = &'a GradedSpace<R>>,
        M: IntoIterator<Item = &'a GradedSpace<R>>,
        R: 'a,
    {
        let codomain: Vec<&GradedSpace<R>> = codomain.into_iter().collect();
        let domain: Vec<&GradedSpace<R>> = domain.into_iter().collect();
        let legs: Vec<&GradedSpace<R>> = codomain.iter().chain(&domain).copied().collect();
        let provider = Arc::clone(Self::authority(&legs)?);
        let fused_codomain = Self::fused_content(&provider, &codomain)?;
        let fused_domain = Self::fused_content(&provider, &domain)?;
        let fits = if embed {
            // TensorKit `domain ≾ codomain`: sectorwise embeddable.
            fused_domain
                .iter()
                .all(|&(sector, deg)| fused_codomain.iter().any(|&(s, d)| s == sector && d >= deg))
        } else {
            // TensorKit `domain ≅ codomain`: identical fused sector content
            // (both sides are SectorId-sorted, so slice equality is content
            // equality).
            fused_codomain == fused_domain
        };
        if !fits {
            // Keep the stable constructor diagnostic shape.
            return Err(Error::InvalidArgument(format!(
                "{what}: codomain and domain are not {} (fused sector content differs)",
                if embed {
                    "isometrically embeddable"
                } else {
                    "isomorphic"
                }
            ))
            .into());
        }
        let mut tensor = Self::build(runtime, provider, &codomain, &domain, Fill::Zeros)?;
        // TensorKit's `one!` per coupled-sector block (`tensors/linalg.jl:102-158`).
        // A block's rows and columns enumerate (fusion tree, degeneracy)
        // pairs, so for a Generic provider the diagonal is also the
        // equal-tree, equal-vertex, equal-degeneracy identity.
        write_identity_blocks_generic(&mut tensor)?;
        Ok(tensor)
    }

    /// The canonical structural isomorphism `codomain <- domain` (TensorKit
    /// `isomorphism(W ← V)`; also TensorKit `id(V)` as `isomorphism(V, V)` and
    /// `unitary`, which only adds a Euclidean inner-product check every TeNeT
    /// provider satisfies): every coupled-sector block is the identity matrix,
    /// which requires the fused codomain and domain to carry identical sector
    /// content. Multiplicity-free and checked-Generic providers alike.
    ///
    /// # Errors
    ///
    /// Everything [`Self::zeros`] reports, plus [`Error::InvalidArgument`]
    /// when the fused codomain and domain differ in sector content
    /// (TensorKit's `SpaceMismatch` on `domain ≅ codomain`).
    ///
    /// # Complexity
    ///
    /// One fused-content fold over the legs plus one `O(stored_len)` payload.
    pub fn isomorphism<'a, C, M>(
        runtime: &Runtime,
        codomain: C,
        domain: M,
    ) -> Result<Self, TypedFacadeError<R>>
    where
        C: IntoIterator<Item = &'a GradedSpace<R>>,
        M: IntoIterator<Item = &'a GradedSpace<R>>,
        R: 'a,
    {
        Self::structural(runtime, codomain, domain, false, "isomorphism")
    }

    /// The canonical isometry `codomain <- domain` (TensorKit
    /// `isometry(W ← V)`): each
    /// coupled-sector block is the partial identity (the first `cols` columns
    /// of the identity), so `t† ∘ t = isomorphism(domain, domain)`. Requires
    /// the domain to embed isometrically in the codomain (sectorwise
    /// `deg_domain <= deg_codomain` on the fused content). Multiplicity-free
    /// and checked-Generic providers alike.
    ///
    /// # Errors
    ///
    /// Everything [`Self::zeros`] reports, plus [`Error::InvalidArgument`]
    /// when the fused domain does not embed sectorwise into the fused
    /// codomain (TensorKit's `SpaceMismatch` on `domain ≾ codomain`).
    ///
    /// # Complexity
    ///
    /// One fused-content fold over the legs plus one `O(stored_len)` payload.
    pub fn isometry<'a, C, M>(
        runtime: &Runtime,
        codomain: C,
        domain: M,
    ) -> Result<Self, TypedFacadeError<R>>
    where
        C: IntoIterator<Item = &'a GradedSpace<R>>,
        M: IntoIterator<Item = &'a GradedSpace<R>>,
        R: 'a,
    {
        Self::structural(runtime, codomain, domain, true, "isometry")
    }
}

impl<R, D> TensorMap<R, D>
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64> + CheckedFusionAlgebra + SectorCodec,
    D: TensorScalar,
{
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
