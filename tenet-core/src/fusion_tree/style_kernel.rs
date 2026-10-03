//! Fusion-style kernels: the only mode-specific part of a fusion-tree move.
//!
//! Every move has one surgery (which sectors a swap reads, which innerline it
//! rewrites, which external legs it exchanges). A [`StyleKernel`] supplies
//! only what TensorKit branches on `FusionStyle` inside the coefficient
//! (`braiding_manipulations.jl:132, 157`; `duality_manipulations.jl:89-112`
//! for bends): admission, the coefficient, and, for Generic fusion, the
//! vertex labels each output carries.
//!
//! The kernels are sealed and statically dispatched: [`UniqueK`] (one channel
//! per fusion, in-place trees), [`SimpleK`] (multiplicity-free channel
//! enumeration) and [`GenericK`] (outer multiplicity, fallible provider).

use super::*;

mod sealed {
    pub trait Sealed {}
}

/// Mode-specific coefficient emission for the shared fusion-tree surgery.
///
/// Why not one F-move channel enumeration for every kernel: admissibility
/// differs. Multiplicity-free moves enumerate complete `fusion_channels`
/// (grouped and prefix-filtered in the multi-F-move), while Generic moves
/// visit only channels inside the provider's checked table
/// (`fusion_channels_in_table`), whose frontier channels are provably dead on
/// admitted structures. Channel enumeration is therefore part of emission.
pub(crate) trait StyleKernel: sealed::Sealed {
    type S: CategoricalScalar;
    type E: From<CoreError>;

    fn vacuum(&self) -> SectorId;
    fn braiding_style(&self) -> BraidingStyleKind;

    /// The fusion-style gate of an Artin swap.
    fn admit_artin(&self) -> Result<(), Self::E>;

    /// The first-pair swap `a ⊗ b ← c` (TensorKit `artin_braid`, `i == 1`):
    /// one term per output vertex label.
    fn artin_first<T: TreeView, W: ArtinWriter<Self::S, Self::E>>(
        &self,
        left: SectorId,
        right: SectorId,
        coupled: SectorId,
        inverse: bool,
        tree: &T,
        out: &mut W,
    ) -> Result<(), Self::E>;

    /// A swap past the first pair over the inner-extended lines
    /// `[a, b, c, d, e]`: one term per new innerline `c′` and vertex labels.
    fn artin_general<T: TreeView, W: ArtinWriter<Self::S, Self::E>>(
        &self,
        sectors: [SectorId; 5],
        inverse: bool,
        index: usize,
        tree: &T,
        out: &mut W,
    ) -> Result<(), Self::E>;
}

/// Read access to the tree a fusion-tree move acts on.
pub(crate) trait TreeView {
    fn coupled(&self) -> SectorId;
    fn innerlines(&self) -> &[SectorId];
    /// The vertex label at `position`; only Generic kernels read it.
    fn vertex(&self, position: usize) -> Option<MultiplicityIndex>;
}

impl TreeView for UnhashedFusionTree {
    fn coupled(&self) -> SectorId {
        MultiplicityFreeTreeLocalData::coupled(self)
    }
    fn innerlines(&self) -> &[SectorId] {
        MultiplicityFreeTreeLocalData::innerlines(self)
    }
    fn vertex(&self, position: usize) -> Option<MultiplicityIndex> {
        self.vertex_at(position)
    }
}

impl TreeView for MultiplicityFreeTreeLocal {
    #[inline(always)]
    fn coupled(&self) -> SectorId {
        self.coupled
    }
    #[inline(always)]
    fn innerlines(&self) -> &[SectorId] {
        &self.innerlines
    }
    #[inline(always)]
    fn vertex(&self, _: usize) -> Option<MultiplicityIndex> {
        None
    }
}

impl TreeView for FusionTreeKey {
    fn coupled(&self) -> SectorId {
        FusionTreeKey::coupled(self)
    }
    fn innerlines(&self) -> &[SectorId] {
        FusionTreeKey::innerlines(self)
    }
    fn vertex(&self, position: usize) -> Option<MultiplicityIndex> {
        self.vertices().get(position).copied()
    }
}

/// How an Artin output's vertex labels relate to the source tree's.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum ArtinVertices {
    /// Unchanged.
    Keep,
    /// A unit crossing past the first pair: vertices `index - 1` and `index`
    /// exchange.
    SwapUnit,
    /// The first vertex becomes the zero-based label `nu`.
    First { nu: usize },
    /// Vertices `index - 1` and `index` become the zero-based `sigma`, `lambda`.
    Pair { sigma: usize, lambda: usize },
}

/// Where an Artin output goes. `begin` validates and stages the structural
/// edit (the innerline update `(position, sector)` and the vertex relation);
/// `finish` attaches the coefficient. Each kernel calls them in its
/// established order relative to the coefficient, so malformed-input errors
/// keep their order per mode.
pub(crate) trait ArtinWriter<S, E> {
    type Slot;
    /// A capacity hint for outputs the kernel is about to emit.
    fn reserve(&mut self, _additional: usize) {}
    fn begin(
        &mut self,
        innerline: Option<(usize, SectorId)>,
        vertices: ArtinVertices,
    ) -> Result<Self::Slot, E>;
    fn finish(&mut self, slot: Self::Slot, coefficient: S) -> Result<(), E>;
}

/// The position-dependent part of one Artin swap, fixed before any tree is
/// read: TensorKit `artin_braid(f, i; inv)` entry checks
/// (`braiding_manipulations.jl:18-60`).
pub(crate) struct ArtinSite {
    rank: usize,
    first: SectorId,
    index: usize,
    inverse: bool,
    left: SectorId,
    right: SectorId,
}

impl ArtinSite {
    pub(crate) fn new<K: StyleKernel>(
        kernel: &K,
        uncoupled: &[SectorId],
        index: usize,
        inverse: bool,
    ) -> Result<Self, K::E> {
        kernel.admit_artin()?;
        let rank = uncoupled.len();
        if index + 1 >= rank {
            return Err(CoreError::InvalidBraidIndex { index, rank }.into());
        }
        let left = uncoupled[index];
        let right = uncoupled[index + 1];
        if left != kernel.vacuum()
            && right != kernel.vacuum()
            && !kernel.braiding_style().has_braiding()
        {
            return Err(CoreError::UnsupportedSectorBraid {
                left,
                right,
                style: kernel.braiding_style(),
            }
            .into());
        }
        Ok(Self {
            rank,
            first: uncoupled[0],
            index,
            inverse,
            left,
            right,
        })
    }

    #[inline(always)]
    fn inner_extended<T: TreeView>(&self, tree: &T, index: usize) -> Result<SectorId, CoreError> {
        if index == 0 {
            return Ok(self.first);
        }
        if index + 1 == self.rank {
            return Ok(tree.coupled());
        }
        tree.innerlines()
            .get(index - 1)
            .copied()
            .ok_or(CoreError::MalformedFusionTree {
                message: "inner-extended tree is missing an innerline",
            })
    }
}

/// The one Artin swap surgery (TensorKit `artin_braid`,
/// `braiding_manipulations.jl:18-198`): pick the unit, first-pair or general
/// case and read the inner-extended lines; the kernel emits each output into
/// `out`, which owns its representation (an in-place tree, a compact local,
/// or a full key).
#[inline(always)]
pub(crate) fn artin_surgery<K, T, W>(
    kernel: &K,
    site: &ArtinSite,
    tree: &T,
    out: &mut W,
) -> Result<(), K::E>
where
    K: StyleKernel,
    T: TreeView,
    W: ArtinWriter<K::S, K::E>,
{
    let index = site.index;
    let (left, right) = (site.left, site.right);
    let vacuum = kernel.vacuum();
    if left == vacuum || right == vacuum {
        out.reserve(1);
        let slot = if index == 0 {
            out.begin(None, ArtinVertices::Keep)?
        } else {
            let inner_source = if left == vacuum {
                site.inner_extended(tree, index + 1)?
            } else {
                site.inner_extended(tree, index - 1)?
            };
            out.begin(Some((index - 1, inner_source)), ArtinVertices::SwapUnit)?
        };
        return out.finish(slot, K::S::one());
    }

    if index == 0 {
        let coupled = if site.rank > 2 {
            tree.innerlines()
                .first()
                .copied()
                .ok_or(CoreError::MalformedFusionTree {
                    message: "first braid of a rank > 2 tree requires the first innerline",
                })?
        } else {
            tree.coupled()
        };
        return kernel.artin_first(left, right, coupled, site.inverse, tree, out);
    }

    let a = site.inner_extended(tree, index - 1)?;
    let c = site.inner_extended(tree, index)?;
    let e = site.inner_extended(tree, index + 1)?;
    kernel.artin_general([a, left, c, right, e], site.inverse, index, tree, out)
}

/// The innerline message a writer reports when the update position is absent.
pub(crate) fn artin_innerline_message(vertices: ArtinVertices) -> &'static str {
    match vertices {
        ArtinVertices::SwapUnit => "unit braid past the first adjacent pair requires an innerline",
        _ => "non-first braid requires an innerline to update",
    }
}

/// The position-dependent part of one `bendright`, fixed by the external
/// legs: TensorKit `_bendright_treepair` (`duality_manipulations.jl:33-54`).
pub(crate) struct BendSite {
    pub(crate) codomain_rank: usize,
    pub(crate) domain_rank: usize,
    codomain_first: SectorId,
    pub(crate) bent_sector: SectorId,
    bent_is_dual: Option<bool>,
}

/// The lines one `bendright` reads from a tree pair: `c` (coupled), `a`
/// (the left part's coupled line) and the bent leg's duality.
#[derive(Clone, Copy)]
pub(crate) struct BendLines {
    pub(crate) coupled: SectorId,
    pub(crate) left_coupled: SectorId,
    pub(crate) bent_is_dual: bool,
}

impl BendSite {
    pub(crate) fn new(
        codomain_uncoupled: &[SectorId],
        codomain_is_dual: &[bool],
        domain_rank: usize,
    ) -> Result<Self, CoreError> {
        let codomain_rank = codomain_uncoupled.len();
        if codomain_rank == 0 {
            return Err(CoreError::MalformedFusionTree {
                message: "bendright requires at least one codomain leg",
            });
        }
        Ok(Self {
            codomain_rank,
            domain_rank,
            codomain_first: codomain_uncoupled[0],
            bent_sector: codomain_uncoupled[codomain_rank - 1],
            bent_is_dual: codomain_is_dual.get(codomain_rank - 1).copied(),
        })
    }

    pub(crate) fn bent_is_dual(&self) -> Result<bool, CoreError> {
        self.bent_is_dual.ok_or(CoreError::MalformedFusionTree {
            message: "codomain tree is missing a duality flag",
        })
    }

    /// `a = N₁ == 1 ? unit : N₁ == 2 ? uncoupled[1] : innerlines[end]`
    /// (`duality_manipulations.jl:37`), after the coupled-sector check.
    #[inline(always)]
    pub(crate) fn lines<C, D>(
        &self,
        vacuum: SectorId,
        codomain: &C,
        domain: &D,
    ) -> Result<BendLines, CoreError>
    where
        C: TreeView + ?Sized,
        D: TreeView + ?Sized,
    {
        let coupled = codomain.coupled();
        if self.domain_rank != 0 && domain.coupled() != coupled {
            return Err(CoreError::MalformedFusionTree {
                message: "fusion tree pair requires matching coupled sectors",
            });
        }
        let left_coupled = match self.codomain_rank {
            1 => vacuum,
            2 => self.codomain_first,
            _ => codomain
                .innerlines()
                .last()
                .copied()
                .ok_or(CoreError::MalformedFusionTree {
                    message: "bendright requires the last codomain innerline",
                })?,
        };
        Ok(BendLines {
            coupled,
            left_coupled,
            bent_is_dual: self.bent_is_dual()?,
        })
    }
}

/// The `bendright` coefficient of a kernel whose provider carries rigidity
/// data: `coeff₀ = √d_c/√d_a · conj(κ_{b̄})^{[b dual]}` times the `B` symbol
/// (`duality_manipulations.jl:62-65, 88-110`).
pub(crate) trait BendKernel: StyleKernel {
    /// The emitted `B` data: a scalar without multiplicity, a row `B[μ, :]`
    /// with it.
    type Row;

    fn bend_row<T: TreeView + ?Sized>(
        &self,
        site: &BendSite,
        lines: &BendLines,
        codomain: &T,
    ) -> Result<Self::Row, Self::E>;
}

/// One `bendright`: the lines it reads and the kernel's `B` emission. Every
/// mode reads the same lines; only the coefficient and, for Generic fusion,
/// the output vertex labels differ.
#[inline(always)]
pub(crate) fn bend_surgery<K, C, D>(
    kernel: &K,
    site: &BendSite,
    codomain: &C,
    domain: &D,
) -> Result<(BendLines, K::Row), K::E>
where
    K: BendKernel,
    C: TreeView + ?Sized,
    D: TreeView + ?Sized,
{
    let lines = site.lines(kernel.vacuum(), codomain, domain)?;
    let row = kernel.bend_row(site, &lines, codomain)?;
    Ok((lines, row))
}

/// The multiplicity-free bend coefficient, shared by `UniqueK` and `SimpleK`.
#[inline(always)]
fn mf_bend_coefficient<R>(rule: &R, bent: SectorId, lines: &BendLines) -> R::Scalar
where
    R: MultiplicityFreeRigidSymbols,
{
    let coefficient = rule.sqrt_dim_scalar(lines.coupled)
        * rule.inv_sqrt_dim_scalar(lines.left_coupled)
        * rule.b_symbol_scalar(lines.left_coupled, bent, lines.coupled);
    if lines.bent_is_dual {
        coefficient * rule.frobenius_schur_phase_scalar(rule.dual(bent)).conj()
    } else {
        coefficient
    }
}

impl<R: MultiplicityFreeRigidSymbols> BendKernel for UniqueK<'_, R> {
    type Row = R::Scalar;

    #[inline(always)]
    fn bend_row<T: TreeView + ?Sized>(
        &self,
        site: &BendSite,
        lines: &BendLines,
        _: &T,
    ) -> Result<R::Scalar, CoreError> {
        Ok(mf_bend_coefficient(self.0, site.bent_sector, lines))
    }
}

impl<R: MultiplicityFreeRigidSymbols> BendKernel for SimpleK<'_, R> {
    type Row = R::Scalar;

    #[inline(always)]
    fn bend_row<T: TreeView + ?Sized>(
        &self,
        site: &BendSite,
        lines: &BendLines,
        _: &T,
    ) -> Result<R::Scalar, CoreError> {
        Ok(mf_bend_coefficient(self.0, site.bent_sector, lines))
    }
}

/// `coeff₀ · B[μ, :]` and the dual of the bent leg, which joins the domain.
pub(crate) struct GenericBendRow<S> {
    pub(crate) bent_dual: SectorId,
    coeff0: S,
    bmat: GenericRMatrix<S>,
    mu: usize,
}

impl<S: CategoricalScalar> GenericBendRow<S> {
    /// Every nonzero `coeff₀ · B[μ, ν]` with its zero-based `ν`
    /// (`duality_manipulations.jl:104-106`).
    pub(crate) fn terms(&self) -> impl Iterator<Item = (usize, S)> + '_ {
        (0..self.bmat.shape().1)
            .map(|nu| (nu, self.coeff0.clone() * self.bmat.get(self.mu, nu).clone()))
            .filter(|(_, coefficient)| !coefficient.is_zero())
    }
}

impl<C: GenericRigidAccess> BendKernel for GenericK<'_, C> {
    type Row = GenericBendRow<C::Scalar>;

    // GenericFusion branch (duality_manipulations.jl:97-112): Bmat =
    // Bsymbol(a, b, c) and μ = N₁ > 1 ? vertices[end] : 1.
    fn bend_row<T: TreeView + ?Sized>(
        &self,
        site: &BendSite,
        lines: &BendLines,
        codomain: &T,
    ) -> Result<Self::Row, Self::E> {
        let rule = self.0;
        let provider = CheckedGenericSymbolError::Provider;
        let bent = site.bent_sector;
        let bent_dual = rule.try_dual(bent).map_err(provider)?;
        let mut coeff0 = rule.try_sqrt_dim_scalar(lines.coupled).map_err(provider)?
            * rule
                .try_inv_sqrt_dim_scalar(lines.left_coupled)
                .map_err(provider)?;
        if lines.bent_is_dual {
            let dual_bent = rule.try_dual(bent).map_err(provider)?;
            coeff0 = coeff0
                * rule
                    .try_frobenius_schur_phase_scalar(dual_bent)
                    .map_err(provider)?
                    .conj();
        }
        let bmat = rule.try_b_symbol_generic(lines.left_coupled, bent, lines.coupled)?;
        let mu = if site.codomain_rank > 1 {
            mu_index(codomain, site.codomain_rank - 2)?
        } else {
            0
        };
        let (rows, cols) = bmat.shape();
        if mu >= rows {
            return Err(CheckedGenericSymbolError::Shape {
                symbol: "B",
                expected: vec![mu + 1, cols],
                actual: vec![rows, cols],
            });
        }
        Ok(GenericBendRow {
            bent_dual,
            coeff0,
            bmat,
            mu,
        })
    }
}

/// The multiplicity-free first-pair output, shared by `UniqueK` and `SimpleK`.
#[inline(always)]
fn emit_mf_artin_first<R, W>(
    rule: &R,
    left: SectorId,
    right: SectorId,
    coupled: SectorId,
    inverse: bool,
    out: &mut W,
) -> Result<(), CoreError>
where
    R: MultiplicityFreeFusionSymbols,
    W: ArtinWriter<R::Scalar, CoreError>,
{
    let coefficient = mf_artin_first_coefficient(rule, left, right, coupled, inverse);
    let slot = out.begin(None, ArtinVertices::Keep)?;
    out.finish(slot, coefficient)
}

/// Unique fusion: one channel per fusion, in-place trees, infallible symbols.
pub(crate) struct UniqueK<'r, R>(pub(crate) &'r R);
/// Multiplicity-free (Unique or Simple) channel enumeration on compact locals.
pub(crate) struct SimpleK<'r, R>(pub(crate) &'r R);
/// Outer-multiplicity fusion over a checked provider.
pub(crate) struct GenericK<'c, C>(pub(crate) &'c C);

impl<R> sealed::Sealed for UniqueK<'_, R> {}
impl<R> sealed::Sealed for SimpleK<'_, R> {}
impl<C> sealed::Sealed for GenericK<'_, C> {}

impl<R> StyleKernel for UniqueK<'_, R>
where
    R: MultiplicityFreeFusionSymbols,
{
    type S = R::Scalar;
    type E = CoreError;

    fn vacuum(&self) -> SectorId {
        self.0.vacuum()
    }
    fn braiding_style(&self) -> BraidingStyleKind {
        self.0.braiding_style()
    }
    fn admit_artin(&self) -> Result<(), CoreError> {
        if self.0.fusion_style() != FusionStyleKind::Unique {
            return Err(CoreError::UnsupportedFusionStyle {
                expected: FusionStyleKind::Unique,
                actual: self.0.fusion_style(),
            });
        }
        Ok(())
    }
    fn artin_first<T: TreeView, W: ArtinWriter<R::Scalar, CoreError>>(
        &self,
        left: SectorId,
        right: SectorId,
        coupled: SectorId,
        inverse: bool,
        _: &T,
        out: &mut W,
    ) -> Result<(), CoreError> {
        emit_mf_artin_first(self.0, left, right, coupled, inverse, out)
    }
    fn artin_general<T: TreeView, W: ArtinWriter<R::Scalar, CoreError>>(
        &self,
        [a, b, c, d, e]: [SectorId; 5],
        inverse: bool,
        index: usize,
        _: &T,
        out: &mut W,
    ) -> Result<(), CoreError> {
        let c_prime = only_fusion_channel(self.0, a, d)?;
        let slot = out.begin(Some((index - 1, c_prime)), ArtinVertices::Keep)?;
        out.finish(
            slot,
            mf_artin_coefficient(self.0, [a, b, c, d, e, c_prime], inverse),
        )
    }
}

impl<R> StyleKernel for SimpleK<'_, R>
where
    R: MultiplicityFreeFusionSymbols,
{
    type S = R::Scalar;
    type E = CoreError;

    fn vacuum(&self) -> SectorId {
        self.0.vacuum()
    }
    fn braiding_style(&self) -> BraidingStyleKind {
        self.0.braiding_style()
    }
    fn admit_artin(&self) -> Result<(), CoreError> {
        if !self.0.fusion_style().is_multiplicity_free() {
            return Err(CoreError::UnsupportedFusionStyle {
                expected: FusionStyleKind::Simple,
                actual: self.0.fusion_style(),
            });
        }
        Ok(())
    }
    #[inline(always)]
    fn artin_first<T: TreeView, W: ArtinWriter<R::Scalar, CoreError>>(
        &self,
        left: SectorId,
        right: SectorId,
        coupled: SectorId,
        inverse: bool,
        _: &T,
        out: &mut W,
    ) -> Result<(), CoreError> {
        emit_mf_artin_first(self.0, left, right, coupled, inverse, out)
    }
    #[inline(always)]
    fn artin_general<T: TreeView, W: ArtinWriter<R::Scalar, CoreError>>(
        &self,
        [a, b, c, d, e]: [SectorId; 5],
        inverse: bool,
        index: usize,
        _: &T,
        out: &mut W,
    ) -> Result<(), CoreError> {
        for c_prime in self.0.fusion_channels(a, d) {
            if self.0.nsymbol(c_prime, b, e) == 0 {
                continue;
            }
            let slot = out.begin(Some((index - 1, c_prime)), ArtinVertices::Keep)?;
            out.finish(
                slot,
                mf_artin_coefficient(self.0, [a, b, c, d, e, c_prime], inverse),
            )?;
        }
        Ok(())
    }
}

/// The zero-based outer-multiplicity label of the vertex at `position`.
/// [`MultiplicityIndex`] stores the one-based categorical label, and
/// TensorKit's `Rmat[μ, ν]` / `Fmat[κ, λ, μ, ρ]` are one-based Julia indices.
pub(crate) fn mu_index<T: TreeView + ?Sized>(
    tree: &T,
    position: usize,
) -> Result<usize, CoreError> {
    Ok(tree
        .vertex(position)
        .ok_or(CoreError::MalformedFusionTree {
            message: "Generic fusion tree requires a vertex label at the read position",
        })?
        .get()
        - 1)
}

impl<C> StyleKernel for GenericK<'_, C>
where
    C: GenericFRAccess,
{
    type S = C::Scalar;
    type E = CheckedGenericSymbolError<C::Error>;

    fn vacuum(&self) -> SectorId {
        self.0.vacuum()
    }
    fn braiding_style(&self) -> BraidingStyleKind {
        self.0.braiding_style()
    }
    // `has_multiplicity()` is exactly the `FusionStyle(I) isa GenericFusion`
    // predicate TensorKit branches on (braiding_manipulations.jl:137,170).
    fn admit_artin(&self) -> Result<(), Self::E> {
        if !self.0.fusion_style().has_multiplicity() {
            return Err(CoreError::UnsupportedFusionStyle {
                expected: FusionStyleKind::Generic,
                actual: self.0.fusion_style(),
            }
            .into());
        }
        Ok(())
    }

    // GenericFusion i == 1 branch (braiding_manipulations.jl:137-148):
    // R = Rmat[μ, ν] with Rmat = inv ? Rsymbol(b,a,c)' : Rsymbol(a,b,c); the
    // adjoint is taken at the element read.
    fn artin_first<T: TreeView, W: ArtinWriter<Self::S, Self::E>>(
        &self,
        left: SectorId,
        right: SectorId,
        coupled: SectorId,
        inverse: bool,
        tree: &T,
        out: &mut W,
    ) -> Result<(), Self::E> {
        let mu0 = mu_index(tree, 0)?;
        let rmat = if inverse {
            checked_generic_r_symbol(self.0, right, left, coupled)?
        } else {
            checked_generic_r_symbol(self.0, left, right, coupled)?
        };
        let n_nu = self
            .0
            .try_nsymbol(right, left, coupled)
            .map_err(CheckedGenericSymbolError::Provider)?;
        out.reserve(n_nu);
        for nu in 0..n_nu {
            let r = if inverse {
                rmat.get(nu, mu0).conj()
            } else {
                rmat.get(mu0, nu).clone()
            };
            if r.is_zero() {
                continue;
            }
            let slot = out.begin(None, ArtinVertices::First { nu })?;
            out.finish(slot, r)?;
        }
        Ok(())
    }

    // case i > 1 (braiding_manipulations.jl:151-187): for c′ in
    // intersect(a ⊗ d, e ⊗ conj(b)), coeff[σ, λ] = Σ_{ρ,κ} Rmat1[ν,ρ] ·
    // conj(Fmat[κ,λ,μ,ρ]) · conj(Rmat2[σ,κ]). `fusion_channels_in_table`:
    // frontier c′ of a bounded table are provably dead on admitted structures.
    fn artin_general<T: TreeView, W: ArtinWriter<Self::S, Self::E>>(
        &self,
        [a, b, c, d, e]: [SectorId; 5],
        inverse: bool,
        index: usize,
        tree: &T,
        out: &mut W,
    ) -> Result<(), Self::E> {
        let rule = self.0;
        let provider = CheckedGenericSymbolError::Provider;
        let mu0 = mu_index(tree, index - 1)?;
        let nu0 = mu_index(tree, index)?;
        for c_prime in rule.try_fusion_channels_in_table(a, d).map_err(provider)? {
            if rule.try_nsymbol(c_prime, b, e).map_err(provider)? == 0 {
                continue;
            }
            let rmat1 = if inverse {
                checked_generic_r_symbol(rule, d, c, e)?
            } else {
                checked_generic_r_symbol(rule, c, d, e)?
            };
            let rmat2 = if inverse {
                checked_generic_r_symbol(rule, d, a, c_prime)?
            } else {
                checked_generic_r_symbol(rule, a, d, c_prime)?
            };
            let fmat = checked_generic_f_symbol(rule, d, a, b, e, c_prime, c)?;
            let n_sigma = rule.try_nsymbol(a, d, c_prime).map_err(provider)?;
            let n_lambda = rule.try_nsymbol(c_prime, b, e).map_err(provider)?;
            let n_rho = rule.try_nsymbol(d, c, e).map_err(provider)?;
            let n_kappa = rule.try_nsymbol(d, a, c_prime).map_err(provider)?;
            for sigma in 0..n_sigma {
                for lambda in 0..n_lambda {
                    let mut coeff = C::Scalar::zero();
                    for rho in 0..n_rho {
                        for kappa in 0..n_kappa {
                            // Adjoint element reads: Rmat1[ν,ρ] is base[ν,ρ] or
                            // conj(base[ρ,ν]); conj(Rmat2[σ,κ]) is conj(base[σ,κ])
                            // or base[κ,σ] (the double conjugate cancels).
                            let r1 = if inverse {
                                rmat1.get(rho, nu0).conj()
                            } else {
                                rmat1.get(nu0, rho).clone()
                            };
                            let f_conj = fmat.get(kappa, lambda, mu0, rho).conj();
                            let r2_conj = if inverse {
                                rmat2.get(kappa, sigma).clone()
                            } else {
                                rmat2.get(sigma, kappa).conj()
                            };
                            coeff = coeff + r1 * f_conj * r2_conj;
                        }
                    }
                    if coeff.is_zero() {
                        continue;
                    }
                    let slot = out.begin(
                        Some((index - 1, c_prime)),
                        ArtinVertices::Pair { sigma, lambda },
                    )?;
                    out.finish(slot, coeff)?;
                }
            }
        }
        Ok(())
    }
}
