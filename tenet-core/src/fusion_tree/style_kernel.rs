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
    #[inline]
    fn coupled(&self) -> SectorId {
        FusionTreeKey::coupled(self)
    }
    #[inline]
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
        match tree.innerlines().get(index - 1) {
            Some(&line) => Ok(line),
            None => Err(CoreError::MalformedFusionTree {
                message: "inner-extended tree is missing an innerline",
            }),
        }
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
            match tree.innerlines().first() {
                Some(&line) => line,
                None => {
                    return Err(CoreError::MalformedFusionTree {
                        message: "first braid of a rank > 2 tree requires the first innerline",
                    }
                    .into())
                }
            }
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
        match self.bent_is_dual {
            Some(is_dual) => Ok(is_dual),
            None => Err(CoreError::MalformedFusionTree {
                message: "codomain tree is missing a duality flag",
            }),
        }
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
            _ => match codomain.innerlines().last() {
                Some(&line) => line,
                None => {
                    return Err(CoreError::MalformedFusionTree {
                        message: "bendright requires the last codomain innerline",
                    })
                }
            },
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
    let Some(label) = tree.vertex(position) else {
        return Err(CoreError::MalformedFusionTree {
            message: "Generic fusion tree requires a vertex label at the read position",
        });
    };
    Ok(label.get() - 1)
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

/// The first codomain leg a `foldright` moves: TensorKit `foldright`
/// (`duality_manipulations.jl:220-293`) reads `a = f₁.uncoupled[1]` and its
/// duality.
pub(crate) struct FoldSite {
    pub(crate) first: SectorId,
    pub(crate) first_is_dual: bool,
}

impl FoldSite {
    pub(crate) fn new(
        codomain_uncoupled: &[SectorId],
        codomain_is_dual: &[bool],
    ) -> Result<Self, CoreError> {
        let Some(&first) = codomain_uncoupled.first() else {
            return Err(CoreError::MalformedFusionTree {
                message: "foldright requires at least one codomain leg",
            });
        };
        let Some(&first_is_dual) = codomain_is_dual.first() else {
            return Err(CoreError::MalformedFusionTree {
                message: "codomain tree is missing the first duality flag",
            });
        };
        Ok(Self {
            first,
            first_is_dual,
        })
    }
}

/// The `foldright` coefficient of a kernel, and the multi-F-moves it composes:
/// `(f₁′, coeff₁) ∈ multi_Fmove(f₁)`, `(f₂′, coeff₂) ∈
/// multi_Fmove_inv(ā, b, f₂, !isdual(a))`, coefficient
/// `√d_c/√d_b · coeff₂' · A(a, b, c) · coeff₁ · κ_a^{[a dual]}`.
pub(crate) trait FoldKernel: StyleKernel {
    /// The site-fixed part (`κ_a`, `a`, its duality).
    type Fold;
    /// The part fixed by `(b, c)`: `√d_c/√d_b` and `A(a, b, c)`.
    type Factors;
    /// A multi-F-move coefficient: a scalar, or a vector over the top vertex.
    type FCoeff;

    /// The fold data and `ā`.
    fn fold_begin(&self, site: &FoldSite) -> Result<(Self::Fold, SectorId), Self::E>;
    fn fold_factors(
        &self,
        fold: &Self::Fold,
        tail_coupled: SectorId,
        coupled: SectorId,
    ) -> Result<Self::Factors, Self::E>;
    fn fold_coefficient(
        &self,
        fold: &Self::Fold,
        factors: &Self::Factors,
        codomain: &Self::FCoeff,
        domain: &Self::FCoeff,
    ) -> Result<Self::S, Self::E>;
}

/// The multi-F-moves a keyed fold composes, on keys: the surgery itself for
/// keyed kernels, projected and materialized for multiplicity-free locals.
pub(crate) trait KeyedMultiFKernel: FoldKernel {
    type Moves: IntoIterator<Item = (FusionTreeKey, Self::FCoeff)>;
    fn multi_fmove(&self, tree: &FusionTreeKey) -> Result<Self::Moves, Self::E>;
    fn multi_fmove_inv(
        &self,
        leading: SectorId,
        coupled: SectorId,
        tree: &FusionTreeKey,
        leading_is_dual: bool,
    ) -> Result<Self::Moves, Self::E>;
}

/// One `foldright` of a keyed tree pair; every output goes to `emit`.
#[inline(always)]
pub(crate) fn fold_surgery<K, F>(
    kernel: &K,
    tree_pair: &FusionTreePairKey,
    mut emit: F,
) -> Result<(), K::E>
where
    K: KeyedMultiFKernel,
    F: FnMut(FusionTreePairKey, K::S),
{
    let codomain = tree_pair.codomain_tree();
    let site = FoldSite::new(codomain.uncoupled(), codomain.is_dual())?;
    let (fold, dual_first) = kernel.fold_begin(&site)?;
    let coupled = codomain.coupled();
    for (codomain_prime, coeff1) in kernel.multi_fmove(codomain)? {
        let tail_coupled = codomain_prime.coupled();
        let factors = kernel.fold_factors(&fold, tail_coupled, coupled)?;
        // The last output takes `codomain_prime` by move instead of a clone.
        let mut inverse = kernel
            .multi_fmove_inv(
                dual_first,
                tail_coupled,
                tree_pair.domain_tree(),
                !site.first_is_dual,
            )?
            .into_iter();
        let mut next = inverse.next();
        while let Some((domain_prime, coeff2)) = next {
            let coefficient = kernel.fold_coefficient(&fold, &factors, &coeff1, &coeff2)?;
            next = inverse.next();
            if next.is_none() {
                emit(
                    FusionTreePairKey::pair(codomain_prime, domain_prime),
                    coefficient,
                );
                break;
            }
            emit(
                FusionTreePairKey::pair(codomain_prime.clone(), domain_prime),
                coefficient,
            );
        }
    }
    Ok(())
}

/// Every multiplicity-free fold, unique ones included, reads its
/// coefficient through the one authority, [`MultiplicityFreeFoldCoefficient`].
impl<R: MultiplicityFreeRigidSymbols> FoldKernel for SimpleK<'_, R> {
    type Fold = MultiplicityFreeFoldCoefficient<R::Scalar>;
    type Factors = (R::Scalar, R::Scalar);
    type FCoeff = R::Scalar;

    #[inline(always)]
    fn fold_begin(&self, site: &FoldSite) -> Result<(Self::Fold, SectorId), CoreError> {
        Ok((
            MultiplicityFreeFoldCoefficient::new(self.0, site.first, site.first_is_dual),
            self.0.dual(site.first),
        ))
    }
    #[inline(always)]
    fn fold_factors(
        &self,
        fold: &Self::Fold,
        tail_coupled: SectorId,
        coupled: SectorId,
    ) -> Result<Self::Factors, CoreError> {
        Ok(fold.sector_factors(self.0, tail_coupled, coupled))
    }
    #[inline(always)]
    fn fold_coefficient(
        &self,
        fold: &Self::Fold,
        factors: &Self::Factors,
        codomain: &R::Scalar,
        domain: &R::Scalar,
    ) -> Result<R::Scalar, CoreError> {
        Ok(fold.coefficient(factors, codomain, domain))
    }
}

pub(crate) struct GenericFold<S> {
    first: SectorId,
    first_is_dual: bool,
    kappa: S,
}

impl<C: GenericRigidAccess> FoldKernel for GenericK<'_, C> {
    type Fold = GenericFold<C::Scalar>;
    type Factors = (C::Scalar, GenericRMatrix<C::Scalar>);
    type FCoeff = Vec<C::Scalar>;

    fn fold_begin(&self, site: &FoldSite) -> Result<(Self::Fold, SectorId), Self::E> {
        let provider = CheckedGenericSymbolError::Provider;
        let kappa = self
            .0
            .try_frobenius_schur_phase_scalar(site.first)
            .map_err(provider)?;
        let dual_first = self.0.try_dual(site.first).map_err(provider)?;
        Ok((
            GenericFold {
                first: site.first,
                first_is_dual: site.first_is_dual,
                kappa,
            },
            dual_first,
        ))
    }
    // GenericFusion branch (duality_manipulations.jl:268-284): A(a, b, c),
    // then coeff₀ = √d_c/√d_b.
    fn fold_factors(
        &self,
        fold: &Self::Fold,
        tail_coupled: SectorId,
        coupled: SectorId,
    ) -> Result<Self::Factors, Self::E> {
        let provider = CheckedGenericSymbolError::Provider;
        let a_matrix = self
            .0
            .try_a_symbol_generic(fold.first, tail_coupled, coupled)?;
        let coeff0 = self.0.try_sqrt_dim_scalar(coupled).map_err(provider)?
            * self
                .0
                .try_inv_sqrt_dim_scalar(tail_coupled)
                .map_err(provider)?;
        Ok((coeff0, a_matrix))
    }
    // coeff₀ · (coeff₂' · (Aᵀ · coeff₁)) · κ_a^{[a dual]}.
    fn fold_coefficient(
        &self,
        fold: &Self::Fold,
        (coeff0, a_matrix): &Self::Factors,
        coeff1: &Vec<C::Scalar>,
        coeff2: &Vec<C::Scalar>,
    ) -> Result<C::Scalar, Self::E> {
        let (rows, cols) = a_matrix.shape();
        if coeff1.len() != rows || coeff2.len() != cols {
            return Err(CoreError::MalformedFusionTree {
                message: "foldright: coefficient-vector length disagrees with A-matrix shape",
            }
            .into());
        }
        let mut inner = C::Scalar::zero();
        for (j, coeff2_j) in coeff2.iter().enumerate() {
            let mut column = C::Scalar::zero();
            for (i, coeff1_i) in coeff1.iter().enumerate() {
                column = column + a_matrix.get(i, j).clone() * coeff1_i.clone();
            }
            inner = inner + coeff2_j.conj() * column;
        }
        let coefficient = coeff0.clone() * inner;
        Ok(if fold.first_is_dual {
            coefficient * fold.kappa.clone()
        } else {
            coefficient
        })
    }
}

impl<C: GenericRigidAccess> KeyedMultiFKernel for GenericK<'_, C> {
    type Moves = GenericFmoveTerms<C::Scalar>;

    fn multi_fmove(&self, tree: &FusionTreeKey) -> Result<Self::Moves, Self::E> {
        multi_fmove_surgery(self, tree)
    }
    fn multi_fmove_inv(
        &self,
        leading: SectorId,
        coupled: SectorId,
        tree: &FusionTreeKey,
        leading_is_dual: bool,
    ) -> Result<Self::Moves, Self::E> {
        multi_fmove_inv_surgery(self, &(leading, leading_is_dual), coupled, tree)
    }
}

/// A tree with its externals, as a multi-F-move reads it: a key, or a
/// multiplicity-free frame with one local.
pub(crate) trait FramedTree: TreeView {
    fn uncoupled(&self) -> &[SectorId];
    fn is_dual(&self) -> &[bool];
}

impl FramedTree for FusionTreeKey {
    #[inline]
    fn uncoupled(&self) -> &[SectorId] {
        FusionTreeKey::uncoupled(self)
    }
    #[inline]
    fn is_dual(&self) -> &[bool] {
        FusionTreeKey::is_dual(self)
    }
}

/// A multiplicity-free local read against its frame.
pub(crate) struct FramedLocal<'a> {
    pub(crate) frame: &'a MultiplicityFreeTreeFrame,
    pub(crate) local: &'a MultiplicityFreeTreeLocal,
}

impl TreeView for FramedLocal<'_> {
    #[inline(always)]
    fn coupled(&self) -> SectorId {
        self.local.coupled
    }
    #[inline(always)]
    fn innerlines(&self) -> &[SectorId] {
        &self.local.innerlines
    }
    #[inline(always)]
    fn vertex(&self, _: usize) -> Option<MultiplicityIndex> {
        None
    }
}

impl FramedTree for FramedLocal<'_> {
    #[inline(always)]
    fn uncoupled(&self) -> &[SectorId] {
        &self.frame.uncoupled
    }
    #[inline(always)]
    fn is_dual(&self) -> &[bool] {
        &self.frame.is_dual
    }
}

/// The per-style part of TensorKit `multi_Fmove` / `multi_Fmove_inv`
/// (`basic_manipulations.jl:191-328, 353-470`): Stage 1 tree enumeration and
/// Stage 2 associator products, which differ by fusion style in what is
/// enumerated (complete channels with prefix pruning, the provider's checked
/// table, or the one unique channel) and in the coefficient's shape.
pub(crate) trait MultiFKernel: StyleKernel {
    /// The destination trees with their coefficients.
    type Moves;
    /// What fixes an inverse move's externals besides the source tree: the
    /// leading leg and its duality, or a prepared output frame.
    type Lift;

    /// External checks preceding the rank dispatch.
    fn check_externals<T: FramedTree + ?Sized>(&self, _tree: &T) -> Result<(), Self::E> {
        Ok(())
    }
    /// `N == 1`: the empty tail at the unit (TK `:218-220`).
    fn unit_tail<T: FramedTree + ?Sized>(&self, tree: &T) -> Result<Self::Moves, Self::E> {
        self.tails(tree)
    }
    /// `N == 2`: the one-leg tail (TK `:221-233`).
    fn single_tail<T: FramedTree + ?Sized>(&self, tree: &T) -> Result<Self::Moves, Self::E> {
        self.tails(tree)
    }
    /// Every tail and its associator (TK Stage 1 and Stage 2, or the
    /// `UniqueFusion` branch at `:209-213` for every rank).
    fn tails<T: FramedTree + ?Sized>(&self, tree: &T) -> Result<Self::Moves, Self::E>;

    fn lift_leading(lift: &Self::Lift) -> SectorId;
    /// External checks preceding the `c ∈ a ⊗ b` admission.
    fn check_lift<T: FramedTree + ?Sized>(
        &self,
        _lift: &Self::Lift,
        _tree: &T,
    ) -> Result<(), Self::E> {
        Ok(())
    }
    /// TK `c ∈ a ⊗ b || throw(SectorMismatch)` (`:355-356`).
    fn admit_lift(
        &self,
        leading: SectorId,
        tree_coupled: SectorId,
        coupled: SectorId,
    ) -> Result<(), Self::E>;
    /// Every lifted tree and its conjugated associator.
    fn lifts<T: FramedTree + ?Sized>(
        &self,
        lift: &Self::Lift,
        coupled: SectorId,
        tree: &T,
    ) -> Result<Self::Moves, Self::E>;
}

/// One `multi_Fmove`: split the first leg off `tree`.
#[inline(always)]
pub(crate) fn multi_fmove_surgery<K, T>(kernel: &K, tree: &T) -> Result<K::Moves, K::E>
where
    K: MultiFKernel,
    T: FramedTree + ?Sized,
{
    kernel.check_externals(tree)?;
    match tree.uncoupled().len() {
        0 => Err(CoreError::MalformedFusionTree {
            message: "multi_Fmove requires at least one uncoupled sector",
        }
        .into()),
        1 => kernel.unit_tail(tree),
        2 => kernel.single_tail(tree),
        _ => kernel.tails(tree),
    }
}

/// One `multi_Fmove_inv`: fuse the lift's leading leg onto `tree` at
/// `coupled`.
#[inline(always)]
pub(crate) fn multi_fmove_inv_surgery<K, T>(
    kernel: &K,
    lift: &K::Lift,
    coupled: SectorId,
    tree: &T,
) -> Result<K::Moves, K::E>
where
    K: MultiFKernel,
    T: FramedTree + ?Sized,
{
    kernel.check_lift(lift, tree)?;
    kernel.admit_lift(K::lift_leading(lift), tree.coupled(), coupled)?;
    kernel.lifts(lift, coupled, tree)
}
