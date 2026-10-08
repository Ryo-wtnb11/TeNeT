use super::*;

impl<R, D> TensorMap<R, D>
where
    R: TypedSectorAdmission,
    R::Mode: FusionMode<R> + TypedTensorTransformDispatch<R, D>,
    D: AdvancedLinalgScalar,
{
    /// Solves `self * x = rhs` independently in every coupled sector, without
    /// forming an inverse.
    ///
    /// `rows` and `cols` select the receiver's matrix view, while
    /// `rhs_rows` and `rhs_cols` select the right-hand side's matrix view.
    /// Each pair names source axes in [`Self::permute`] order. The current
    /// split borrows its operand without a transform.
    ///
    /// The permuted operands must share a runtime and fusion-rule identity,
    /// their codomains must be exactly equal, and `self` must have isomorphic
    /// codomain and domain. The result has space
    /// `domain(permuted self) <- domain(permuted rhs)` and keeps `self`'s exact provider `Arc`.
    /// This is TensorKit's left solve `self \\ rhs`.
    /// TensorKit's right solve `self / rhs` can be composed from adjoints
    /// and this left solve with roles chosen in the adjointed axis order.
    ///
    /// Dense input uses one linear solve per coupled sector. A compact
    /// diagonal divisor instead applies its elementwise reciprocal (TensorKit
    /// `D \ t`): a compact right-hand side on the same bond gives a compact
    /// quotient on the divisor's space, and any other one — dense, in any
    /// layout, or a lazy adjoint read in place — lands in the output space
    /// with its leading (bond) axis scaled, `O(Σ_c k_c m_c)` with no LU, in
    /// both fusion modes. Every compact divisor, in any bond layout, is
    /// admitted through one shared structural check. Nonfinite divisor
    /// entries map through the reciprocal. An exact zero divisor entry is the
    /// dense route's singular-block operation error in every mode. A compact
    /// right-hand side of a dense divisor is densified into the solve buffer.
    /// On the dense route, lazy adjoints are materialized only for this call.
    ///
    /// # Errors
    ///
    /// In one order in every fusion mode (#1995), as TensorKit's `\` checks:
    /// [`Error::RuntimeMismatch`], then [`Error::RuleMismatch`] for
    /// incompatible operands, then an operation `SpaceMismatch` for unequal
    /// codomains, then one when the divisor is not isomorphic,
    /// then the borrowed-view refusal below; an operation
    /// `Dense(NumericalFailure)` when a sector is singular comes after all of
    /// them. If a checked provider rejects the output space,
    /// its original error is available as the source. If any preflight, sector
    /// solve, or output-space creation fails, no result tensor is returned.
    /// Non-identity roles also return the existing [`Self::permute`] errors
    /// for malformed axes or unsupported braiding.
    ///
    /// ```
    /// use std::sync::Arc;
    /// use tenet::sector::{U1FusionRule, U1Irrep};
    /// use tenet::typed::{GradedSpace, Runtime, TensorMap};
    ///
    /// let runtime = Runtime::builder().build()?;
    /// let v = GradedSpace::try_new(Arc::new(U1FusionRule), [(U1Irrep::new(0), 2)])?;
    /// let a: TensorMap<_, f64> = TensorMap::isomorphism(&runtime, [&v], [&v])?;
    /// let b: TensorMap<_, f64> = TensorMap::rand_with_seed(&runtime, [&v], [&v], 1)?;
    /// assert_eq!(a.solve(&[0], &[1], &b, &[0], &[1])?.dense_data()?, b.dense_data()?);
    /// # Ok::<(), tenet::typed::Error>(())
    /// ```
    ///
    /// An adjoint view `rhs` (`t.adjoint_view()`) is accepted by the current
    /// split of a compact divisor, which reads it directly, in both fusion
    /// modes. A dense divisor or either moved role pair would copy the view
    /// and returns [`Error::Unsupported`]. Pass `&t.adjoint()?.materialize()?`
    /// instead. Moved roles refuse the view after the runtime and rule checks
    /// (which precede the leg roles' own [`Self::permute`] errors) but before
    /// the shape checks, which need the permuted operands.
    pub fn solve<'a>(
        &self,
        rows: &[usize],
        cols: &[usize],
        rhs: impl Into<TensorRef<'a, R, D>>,
        rhs_rows: &[usize],
        rhs_cols: &[usize],
    ) -> Result<Self, TypedFacadeError<R>> {
        let rhs = rhs.into().operand()?;
        let rhs = &*rhs;
        // Runtime and rule first, ahead of the leg roles: the permute cannot
        // change them, so the first error does not depend on the roles or on
        // the rhs's ownership. Permuting a borrowed view would copy it, so
        // moved roles refuse it next; the shape checks need the permuted
        // operands. A moved compact divisor also becomes dense.
        self.require_solve_operands(rhs)?;
        if !self.axes_are_identity(rows, cols) || !rhs.axes_are_identity(rhs_rows, rhs_cols) {
            rhs.refuse_borrowed_view("solve")?;
        }
        self.with_leg_roles(rows, cols, |lhs| {
            rhs.with_leg_roles(rhs_rows, rhs_cols, |right| lhs.factor_solve(right))
        })
    }
}

impl<R, D> TensorMap<R, D>
where
    R: TypedSectorAdmission,
    R::Mode: FusionMode<R> + TypedAdjointSpace<R> + TypedTensorTransformDispatch<R, D>,
    D: AdvancedLinalgScalar,
{
    /// TensorKit 0.17 / MatrixAlgebraKit `pinv`: the Moore-Penrose
    /// thresholded pseudo-inverse `t⁺ = V S⁺ Uᴴ`, where `t = U S Vᴴ` is the compact SVD and
    /// `S⁺` inverts every singular value above the cutoff and sends the rest to
    /// zero. This is the exact Moore-Penrose inverse of the hard-thresholded
    /// effective-rank tensor `t_r`. It is the Moore-Penrose inverse of `t`
    /// itself only when no genuinely nonzero singular value is discarded, and
    /// then satisfies `t t⁺ t = t`. It reduces to [`Self::inv`] when `t` is
    /// nonsingular and `rcond` keeps every singular value.
    ///
    /// # Tolerance, and the divergence from TensorKit
    ///
    /// The cutoff is `rcond * σ_max` with **one global `σ_max` taken across all
    /// coupled sectors**, and the comparison is strict: a singular value
    /// sitting exactly on the cutoff is discarded. TensorKit instead takes
    /// per-block `atol`/`rtol` keywords, so its relative tolerance is measured
    /// against each block's own largest singular value. That is a deliberate
    /// divergence, not a gap: a per-block relative tolerance cannot cut
    /// anything in a one-dimensional sector however small that sector's
    /// contribution to the tensor is. TensorKit's `DiagonalTensorMap` branch
    /// is deliberately **not** mirrored either: there `rtol` is ignored
    /// whenever `atol` is nonzero, the default is no cutoff at all, and its
    /// comparison (`abs(x) < tol` discards) *keeps* a value sitting exactly on
    /// the cutoff — the opposite of the strict `>` above.
    ///
    /// # Errors
    ///
    /// - [`Error::InvalidArgument`] when `rcond` is not finite or is negative,
    ///   checked before any provider work or dense allocation.
    /// - [`Error::InvalidArgument`] (`pinv singular values must be finite`)
    ///   for a nonfinite compact entry, in every fusion mode: the dense
    ///   cutoff's own refusal of a nonfinite singular value.
    /// - [`Error::Operation`] / [`Error::Core`] from dense SVD or recomposition.
    ///
    /// There is no singular-input failure: sending the offending directions to
    /// zero is what a pseudo-inverse is for.
    ///
    /// # Complexity and storage
    ///
    /// Dense input uses one compact SVD per nonempty coupled sector,
    /// `O(Σ_c n_c³)`, then folds `S⁺` into a column scaling and recomposes with
    /// one local GEMM. A Host compact diagonal input stays compact in every
    /// fusion mode and bond layout (TensorKit `pinv(::DiagonalTensorMap)`
    /// keeps `d.domain`): its magnitudes are compared unrounded, in `f64`,
    /// against the same global cutoff, and a retained subnormal inverts to its
    /// IEEE reciprocal (`Inf`). The dense route compares singular values at
    /// the payload's precision, so for `f32`/`Complex32` a value whose rounded
    /// magnitude lands on the other side of the cutoff is cut on one route and
    /// kept on the other. For `K` stored values and `G` sectors, its
    /// elementwise cutoff and reciprocal take `O(K + G)` time and result
    /// space.
    ///
    /// Checked Generic admits the dense route's swapped output with the
    /// source's exact provider `Arc` and validates its identity, HomSpace,
    /// rank, and layout before any SVD/GEMM. A lazy adjoint is read through its parent's SVD in the
    /// dense stage, `(A^H)^+ = U S^+ Vh` for `A = U S Vh`, in both modes.
    ///
    /// `rows` and `cols` are the leg roles: the operation acts on the matrix
    /// view `self.permute(rows, cols)`, and the current split costs nothing
    /// extra (see [`Self::svd_compact`]'s *Leg roles*).
    pub fn pinv(
        &self,
        rows: &[usize],
        cols: &[usize],
        rcond: f64,
    ) -> Result<Self, TypedFacadeError<R>> {
        self.with_leg_roles(rows, cols, |t| t.factor_pinv(rcond))
    }
}

impl<R, D> TensorMap<R, D>
where
    R: TypedSectorAdmission,
    R::Mode: FusionMode<R> + TypedTensorTransformDispatch<R, D>,
    D: FactorizationScalar,
{
    /// Returns the compact QR factorization `self = q * r` as a [`Qr`].
    ///
    /// In coupled sector `c`, with block shape `m_c x n_c`, the bond dimension
    /// is `k_c = min(m_c, n_c)`. Thus `q : codomain(self) <- W` has
    /// orthonormal columns and `r : W <- domain(self)`. The diagonal of `r` is
    /// fixed to be real and non-negative, matching the default positive gauge.
    /// [`Self::qr_full`] instead uses `dim(W_c) = m_c`, making `q` square and
    /// `r` upper trapezoidal in every sector.
    ///
    /// An owned Host compact diagonal on `V <- V` is factorized directly and
    /// returns both factors in compact storage with `W = V`, including dual
    /// orientation, for multiplicity-free and checked-Generic providers alike
    /// (TensorKit's diagonal dispatch). Any other input, including a
    /// materialized diagonal, takes the dense route, whose `W` is a fresh
    /// nondual bond (TensorKit `fuse`); a dual `V` thus yields `W = V` or its
    /// nondual flip depending on storage. Swapped leg roles are one
    /// [`Self::permute`] first, so the rule follows the mode's rank-(1,1)
    /// swap of a compact diagonal: multiplicity-free rules with a real
    /// categorical scalar keep it compact, so the direct arm applies with the
    /// swapped leg as `W`; checked Generic densifies it and takes the dense
    /// route with a fresh nondual `W`.
    /// Phase is +1 at zero; work and output storage are `O(sum_c k_c)` after
    /// sector/layout validation. Use [`Self::diagview`] to read the factors,
    /// or [`Self::materialize`] before [`Self::dense_data`] for a dense buffer.
    /// A nonfinite entry, dense or diagonal, returns [`Error::Operation`] (an
    /// `InvalidArgument`: `qr`/`lq input components must be finite`) before
    /// any work.
    ///
    /// Other inputs run one dense QR per sector, with cost
    /// `O(sum_c m_c * n_c * min(m_c, n_c))`. Lazy adjoints are materialized
    /// only for the operation (LQ is itself the QR of the adjoint, so the
    /// copy is the one LQ would make). Checked factors use the same provider
    /// instance as `self`.
    /// If any sector fails or the provider rejects an output space, no factors
    /// are returned.
    ///
    /// # Errors
    ///
    /// Dense execution returns [`Error::Operation`], while factor-layout
    /// failures return [`Error::Core`] where applicable. If a checked provider
    /// rejects an output space, its original error is available as the source.
    ///
    /// ```
    /// use std::sync::Arc;
    /// use tenet::sector::{U1FusionRule, U1Irrep};
    /// use tenet::typed::{GradedSpace, Qr, Runtime, TensorMap};
    ///
    /// let runtime = Runtime::builder().build()?;
    /// let v = GradedSpace::try_new(Arc::new(U1FusionRule), [(U1Irrep::new(0), 2)])?;
    /// let a: TensorMap<_, f64> = TensorMap::rand_with_seed(&runtime, [&v], [&v], 3)?;
    /// let Qr { q, r } = a.qr_compact(&[0], &[1])?;
    /// let rebuilt = q.compose(&r)?;
    /// assert!(rebuilt.axpby(1.0, &a, -1.0)?.norm(2.0)? < 1e-12);
    /// # Ok::<(), tenet::typed::Error>(())
    /// ```
    ///
    /// `rows` and `cols` are the leg roles: the operation acts on the matrix
    /// view `self.permute(rows, cols)`, and the current split costs nothing
    /// extra (see [`Self::svd_compact`]'s *Leg roles*).
    pub fn qr_compact(
        &self,
        rows: &[usize],
        cols: &[usize],
    ) -> Result<Qr<Self>, TypedFacadeError<R>> {
        self.with_leg_roles(rows, cols, |t| {
            t.factor_qr(FactorOp::QrCompact, |lease, source| {
                tenet_matrixalgebra::seam::qr_compact_from_source::<R::Mode, _, _, _, _>(
                    lease, source,
                )
            })
        })
    }
}

impl<R, D> TensorMap<R, D>
where
    R: TypedSectorAdmission,
    R::Mode: FusionMode<R> + TypedTensorTransformDispatch<R, D>,
    D: FactorizationScalar,
{
    /// Returns the compact singular-value decomposition
    /// `self = u * s * vh` as an [`Svd`], which states the spectrum order and
    /// the storage routes of `s`.
    ///
    /// Sector `c` uses `k_c = min(m_c, n_c)`, with
    /// `u : codomain(self) <- W`, `s : W <- W`, and
    /// `vh : W <- domain(self)`. `u` has orthonormal columns, `vh` has
    /// orthonormal rows, and each sector's singular values are non-negative and
    /// descending. [`Self::svd_full`] instead returns square outer factors and
    /// a generally rectangular `s`.
    ///
    /// On Host, compact `s` stores only `sum_c k_c` diagonal values for both
    /// multiplicity-free and checked-Generic providers. `s.materialize()`
    /// produces dense storage when needed.
    /// All checked factors retain the source's exact provider `Arc`.
    ///
    /// The bond `W` of `s` is TensorKit's `fuse(codomain)`: a nondual bond of
    /// the singular-value sectors, which a nondual `V <- V` input already is.
    ///
    /// Dense inputs cost `O(sum_c m_c * n_c * min(m_c, n_c))`. An owned
    /// compact diagonal is sorted directly by sector, without
    /// a dense input or dense SVD call; its dense `u` and `vh` still require
    /// `O(sum_c k_c²)` output storage and writes. A lazy adjoint is factored
    /// from its parent without materializing it, in the same gauge as its
    /// materialized adjoint. A compact diagonal uses the
    /// same direct per-sector sorting and factor publication in both modes; a
    /// nonfinite entry, here or in dense input, returns
    /// [`Error::Operation`] (an `InvalidArgument`: `svd input components must be finite`)
    /// before any work.
    /// Any sector, layout, or provider failure returns no factors.
    ///
    /// ```
    /// use std::sync::Arc;
    /// use tenet::sector::{U1FusionRule, U1Irrep};
    /// use tenet::typed::{GradedSpace, Runtime, Svd, TensorMap};
    ///
    /// let runtime = Runtime::builder().build()?;
    /// let v = GradedSpace::try_new(Arc::new(U1FusionRule), [(U1Irrep::new(0), 2)])?;
    /// let a: TensorMap<_, f64> = TensorMap::rand_with_seed(&runtime, [&v], [&v], 4)?;
    /// let Svd { u, s, vh } = a.svd_compact(&[0], &[1])?;
    /// let rebuilt = u.compose(&s)?.compose(&vh)?;
    /// assert!(rebuilt.axpby(1.0, &a, -1.0)?.norm(2.0)? < 1e-12);
    /// # Ok::<(), tenet::typed::Error>(())
    /// ```
    ///
    /// # Leg roles
    ///
    /// `rows` and `cols` list source axes (`0..rank`, codomain first) exactly
    /// as for [`Self::permute`]. They are the codomain and the domain of the
    /// matrix this operation acts on, and the result is the operation applied
    /// to `self.permute(rows, cols)`; every space, factor and error stated
    /// here refers to that matrix view. The current split (`rows = 0..nout`,
    /// `cols = nout..rank`) borrows `self` and runs no transform. Any other
    /// split costs that one permute and nothing more: the permuted tensor is
    /// consumed in place of `self`, and its packed coupled-sector regions are
    /// read by the solver directly. Every factorization and matrix function
    /// (`svd_*`, `qr_*`, `lq_*`, the polar and null-space families, `eigh_*`,
    /// `eig_*`, `exp`, `inv`, `pinv`) takes its leg roles this way. A split
    /// other than the current one needs a symmetric braiding; otherwise the
    /// permute's `UnsupportedBraidingStyle` error is returned before any
    /// factorization work, as are its errors for malformed axes.
    pub fn svd_compact(
        &self,
        rows: &[usize],
        cols: &[usize],
    ) -> Result<Svd<Self>, TypedFacadeError<R>> {
        self.with_leg_roles(rows, cols, Self::factor_svd_compact)
    }

    /// Returns the full SVD `self = u * s * vh` with square outer factors.
    ///
    /// In sector `c`, `u` is `m_c x m_c`, `vh` is `n_c x n_c`, and `s` is
    /// the rectangular `m_c x n_c` diagonal matrix. The spaces are
    /// `u : codomain <- W_out`, `s : W_out <- W_in`, and
    /// `vh : W_in <- domain`. It accepts the same inputs as
    /// [`Self::svd_compact`], but its square outer factors can require more
    /// dense storage. On Host, `s` uses compact diagonal storage on the bond
    /// `fuse(codomain)` exactly when the constructed `W_out` and `W_in` legs
    /// coincide and every positive bond sector has a complete spectrum.
    /// Otherwise it is dense. An owned compact diagonal input returns its
    /// [`Self::svd_compact`], with which its full SVD coincides (TensorKit
    /// `svd_compact!(::DiagonalAlgorithm)` is `svd_full!`), in every fusion
    /// mode: no dense input materialization or solver call.
    /// Call `s.materialize()` before `dense_data()` when needed.
    /// Checked factors use the source provider instance; a failure returns no
    /// factors.
    ///
    /// `rows` and `cols` are the leg roles: the operation acts on the matrix
    /// view `self.permute(rows, cols)`, and the current split costs nothing
    /// extra (see [`Self::svd_compact`]'s *Leg roles*).
    pub fn svd_full(
        &self,
        rows: &[usize],
        cols: &[usize],
    ) -> Result<Svd<Self>, TypedFacadeError<R>> {
        self.with_leg_roles(rows, cols, Self::factor_svd_full)
    }
}

impl<R, D> TensorMap<R, D>
where
    R: TypedSectorAdmission,
    R::Mode: FusionMode<R> + TypedAdjointSpace<R> + TypedTensorTransformDispatch<R, D>,
    D: FactorizationScalar,
{
    /// Returns the compact LQ factorization `self = l * q` as an [`Lq`].
    ///
    /// Sector `c` uses bond dimension `k_c = min(m_c, n_c)`, with
    /// `l : codomain(self) <- W` and `q : W <- domain(self)`. The rows of `q`
    /// are orthonormal and the diagonal of `l` is real and non-negative.
    /// [`Self::lq_full`] instead uses `dim(W_c) = n_c`, so `q` is square and
    /// `l` is lower trapezoidal.
    ///
    /// Its cost and compact storage contract are the same as [`Self::qr_compact`]:
    /// owned Host compact diagonals preserve `W = V` and both factors
    /// are compact, including on a dual `V`.
    /// Checked factors use the source provider instance, and a failure returns
    /// no factors. A lazy adjoint is materialized only for the operation.
    ///
    /// ```
    /// use std::sync::Arc;
    /// use tenet::sector::{U1FusionRule, U1Irrep};
    /// use tenet::typed::{GradedSpace, Lq, Runtime, TensorMap};
    ///
    /// let runtime = Runtime::builder().build()?;
    /// let v = GradedSpace::try_new(Arc::new(U1FusionRule), [(U1Irrep::new(0), 2)])?;
    /// let a: TensorMap<_, f64> = TensorMap::rand_with_seed(&runtime, [&v], [&v], 5)?;
    /// let Lq { l, q } = a.lq_compact(&[0], &[1])?;
    /// assert!(l.compose(&q)?.axpby(1.0, &a, -1.0)?.norm(2.0)? < 1e-12);
    /// # Ok::<(), tenet::typed::Error>(())
    /// ```
    ///
    /// `rows` and `cols` are the leg roles: the operation acts on the matrix
    /// view `self.permute(rows, cols)`, and the current split costs nothing
    /// extra (see [`Self::svd_compact`]'s *Leg roles*).
    pub fn lq_compact(
        &self,
        rows: &[usize],
        cols: &[usize],
    ) -> Result<Lq<Self>, TypedFacadeError<R>> {
        self.with_leg_roles(rows, cols, |t| {
            t.factor_lq(FactorOp::LqCompact, |lease, source| {
                tenet_matrixalgebra::seam::lq_compact_from_source::<R::Mode, _, _, _, _>(
                    lease, source,
                )
            })
        })
    }
}

impl<R, D> TensorMap<R, D>
where
    R: TypedSectorAdmission,
    D: TensorScalar,
{
    /// Applies `f` to every stored value of a compact diagonal tensor and
    /// returns a compact diagonal on the same space.
    ///
    /// This is TensorKit 0.17's `DiagonalTensorMap($f.(d.data), d.domain)`
    /// (`tensors/diagonal.jl:384-390`) with the elementwise function supplied
    /// by the caller: `s.map_diagonal(f64::sqrt)` splits singular values for a
    /// Vidal gauge, `s.map_diagonal(|x| 1.0 / x)` inverts them. A spectral map
    /// never mixes sectors or changes a bond dimension, so the result keeps the
    /// receiver's space. `f` sees the values directly; for a real payload it
    /// decides what a negative entry means (`f64::sqrt` yields `NaN`).
    ///
    /// # Domain
    ///
    /// The receiver must store a compact diagonal: a Host
    /// [`Self::svd_compact`] `s`, an eigendecomposition's `d`, or a tensor
    /// built by [`Self::diagonal`]. A dense tensor is rejected even when its
    /// blocks happen to be diagonal; this method never scans or builds a
    /// dense buffer. A diagonal that is stored densely is made compact explicitly with
    /// `TensorMap::diagonal(runtime, bond, t.diagview()?)`.
    ///
    /// # Errors
    ///
    /// [`Error::InvalidArgument`] when the receiver has dense storage,
    /// including a lazy adjoint.
    ///
    /// # Complexity
    ///
    /// `Σ_c k_c` calls of `f` over the stored values and one owned compact
    /// output of that length.
    ///
    /// ```
    /// use std::sync::Arc;
    ///
    /// use tenet::sector::{U1FusionRule, U1Irrep};
    /// use tenet::typed::{GradedSpace, Runtime, SectorSpectrum, TensorMap};
    ///
    /// let runtime = Runtime::builder().build()?;
    /// let v = GradedSpace::try_new(Arc::new(U1FusionRule), [(U1Irrep::new(0), 2)])?;
    /// let values = SectorSpectrum { sector: U1Irrep::new(0), values: vec![4.0, 9.0] };
    /// let s: TensorMap<_, f64> = TensorMap::diagonal(&runtime, &v, [values])?;
    /// assert_eq!(s.map_diagonal(f64::sqrt)?.diagview()?[0].values, [2.0, 3.0]);
    /// let dense: TensorMap<_, f64> = TensorMap::isomorphism(&runtime, [&v], [&v])?;
    /// assert!(dense.map_diagonal(f64::sqrt).is_err());
    /// # Ok::<(), tenet::typed::Error>(())
    /// ```
    pub fn map_diagonal(&self, f: impl Fn(D) -> D) -> Result<Self, Error> {
        let Some(spectrum) = self.spectrum() else {
            return Err(Error::InvalidArgument(
                "map_diagonal requires a compact diagonal tensor (a compact factor or a \
                 TensorMap::diagonal), but this tensor has dense storage"
                    .to_string(),
            ));
        };
        Ok(self.with_spectrum(map_spectrum(spectrum, |value| Ok(f(value)))?))
    }
}

impl<R, D> TensorMap<R, D>
where
    R: TypedSectorAdmission,
    R::Mode: FusionMode<R> + TypedTensorTransformDispatch<R, D>,
    D: FactorizationScalar,
{
    /// Returns the full QR factorization `self = q * r` as a [`Qr`].
    ///
    /// Sector `c` has a square `m_c x m_c` unitary `q` and an
    /// `m_c x n_c` upper-trapezoidal `r`; the intermediate bond therefore has
    /// the codomain's coupled-sector dimensions. The diagonal of `r` is real
    /// and non-negative. For sector shape `m_c x n_c`, dense work is
    /// `O(m_c² n_c)` when `m_c <= n_c`; when `m_c > n_c`, completing the
    /// square `q` costs `O(m_c²(n_c + m_c))`. Source packing and owned factor
    /// publication are additional costs. An owned Host compact diagonal
    /// uses `W = V` and two compact factors under the boundary documented by
    /// [`Self::qr_compact`].
    /// See [`Self::qr_compact`] for the
    /// compact alternative, storage and lazy-input behavior, errors, and example.
    ///
    /// `rows` and `cols` are the leg roles: the operation acts on the matrix
    /// view `self.permute(rows, cols)`, and the current split costs nothing
    /// extra (see [`Self::svd_compact`]'s *Leg roles*).
    pub fn qr_full(&self, rows: &[usize], cols: &[usize]) -> Result<Qr<Self>, TypedFacadeError<R>> {
        self.with_leg_roles(rows, cols, |t| {
            t.factor_qr(FactorOp::QrFull, |lease, source| {
                tenet_matrixalgebra::seam::qr_full_from_source::<R::Mode, _, _, _, _>(lease, source)
            })
        })
    }
}

impl<R, D> TensorMap<R, D>
where
    R: TypedSectorAdmission,
    R::Mode: FusionMode<R> + TypedAdjointSpace<R> + TypedTensorTransformDispatch<R, D>,
    D: FactorizationScalar,
{
    /// Returns the full LQ factorization `self = l * q` as an [`Lq`].
    ///
    /// Sector `c` has an `m_c x n_c` lower-trapezoidal `l` and a square
    /// `n_c x n_c` unitary `q`; the intermediate bond therefore has the
    /// domain's coupled-sector dimensions. The diagonal of `l` is real and
    /// non-negative. For sector shape `m_c x n_c`, dense work is
    /// `O(n_c² m_c)` when `n_c <= m_c`; when `n_c > m_c`, completing the
    /// square `q` costs `O(n_c²(m_c + n_c))`. Source packing, the sectorwise
    /// adjoint, and owned factor publication are additional costs. An owned
    /// Host compact diagonal uses `W = V` and two compact factors under
    /// the boundary documented by [`Self::lq_compact`]. See [`Self::lq_compact`] for the compact alternative, storage and
    /// lazy-input behavior, errors, and example.
    ///
    /// `rows` and `cols` are the leg roles: the operation acts on the matrix
    /// view `self.permute(rows, cols)`, and the current split costs nothing
    /// extra (see [`Self::svd_compact`]'s *Leg roles*).
    pub fn lq_full(&self, rows: &[usize], cols: &[usize]) -> Result<Lq<Self>, TypedFacadeError<R>> {
        self.with_leg_roles(rows, cols, |t| {
            t.factor_lq(FactorOp::LqFull, |lease, source| {
                tenet_matrixalgebra::seam::lq_full_from_source::<R::Mode, _, _, _, _>(lease, source)
            })
        })
    }
}

impl<R, D> TensorMap<R, D>
where
    R: TypedSectorAdmission,
    R::Mode: FusionMode<R> + TypedTensorTransformDispatch<R, D>,
    D: FactorizationScalar,
{
    /// Returns only the singular values, grouped by provider-labelled coupled
    /// sector and descending within each sector.
    ///
    /// No factor tensor or intermediate bond is built. This is the least
    /// allocating member of the SVD family when only the spectrum is needed.
    /// Lazy adjoints are read through their owned parent,
    /// and an owned compact diagonal input is read and sorted directly without
    /// a dense solver for both multiplicity-free and checked-Generic providers;
    /// a nonfinite entry, here or in dense input, returns
    /// [`Error::Operation`] (an `InvalidArgument`: `svd input components must be finite`).
    /// A dense failure returns
    /// [`Error::Operation`]; if a provider cannot decode a sector label, its
    /// original error is available as the source. See [`Self::svd_compact`] for
    /// the decomposition contract and representative example.
    /// For the direct path, validating `B` source blocks, sorting `k_c` values
    /// in each of `G` sectors and sorting public labels costs
    /// `O(B + Σ_c k_c log k_c + G log G)` time and `O(Σ_c k_c + G)` space.
    ///
    /// `rows` and `cols` are the leg roles: the operation acts on the matrix
    /// view `self.permute(rows, cols)`, and the current split costs nothing
    /// extra (see [`Self::svd_compact`]'s *Leg roles*).
    pub fn svd_vals(
        &self,
        rows: &[usize],
        cols: &[usize],
    ) -> Result<Vec<SectorSpectrum<<R as TypedSectorAdmission>::Sector, f64>>, TypedFacadeError<R>>
    {
        self.with_leg_roles(rows, cols, |t| {
            t.factor_values(FactorOp::SvdVals, |lease, source| {
                tenet_matrixalgebra::seam::svd_vals_from_source::<R::Mode, _, _, _, _>(
                    lease, source,
                )
            })
        })
    }
}

impl<R, D> TensorMap<R, D>
where
    R: TypedSectorAdmission,
    R::Mode: FusionMode<R> + TypedTensorTransformDispatch<R, D>,
    D: FactorizationScalar,
{
    /// Returns only the real Hermitian eigenvalues, grouped by
    /// provider-labelled coupled sector and descending by absolute value.
    ///
    /// No eigenvector factor or bond space is built. The input must be an
    /// endomorphism and every sector must pass the same Hermiticity admission
    /// at `hermitian_tol` as [`Self::eigh_full`]. An owned Host compact diagonal is read directly for
    /// both multiplicity-free and checked-Generic providers: a nonfinite entry
    /// returns [`Error::Operation`] (an `InvalidArgument`: `eigh input components must be
    /// finite`; dense input too, after the endomorphism and stacking checks
    /// and before the Hermiticity check), the dense route's Hermiticity check is applied to the diagonal,
    /// and the eigenvalues are its real parts. Lazy adjoints use an
    /// operation-local dense payload. Dense failures return
    /// [`Error::Operation`],
    /// layout failures return [`Error::Core`], and an original provider or
    /// label-decoding error is available as the source. No spectrum is returned
    /// unless every sector succeeds.
    ///
    /// `rows` and `cols` are the leg roles: the operation acts on the matrix
    /// view `self.permute(rows, cols)`, and the current split costs nothing
    /// extra (see [`Self::svd_compact`]'s *Leg roles*).
    pub fn eigh_vals(
        &self,
        rows: &[usize],
        cols: &[usize],
        hermitian_tol: HermitianTol,
    ) -> Result<Vec<SectorSpectrum<<R as TypedSectorAdmission>::Sector, f64>>, TypedFacadeError<R>>
    {
        self.with_leg_roles(rows, cols, |t| {
            t.factor_values(FactorOp::EighVals, |lease, source| {
                tenet_matrixalgebra::seam::eigh_vals_from_source::<R::Mode, _, _, _, _>(
                    lease,
                    source,
                    hermitian_tol,
                )
            })
        })
    }
}

impl<R, D> TensorMap<R, D>
where
    R: TypedSectorAdmission,
    R::Mode: FusionMode<R> + TypedTensorTransformDispatch<R, D>,
    D: FactorizationScalar,
{
    /// Returns the Hermitian eigendecomposition `self = v * d * v^H` as an
    /// [`Eigh`], which states the spectrum order and the storage routes of `d`.
    ///
    /// The input must be an endomorphism whose every coupled-sector block `A`
    /// is Hermitian to `hermitian_tol` (MatrixAlgebraKit's `hermitian_tol`):
    /// `‖(A − Aᴴ)/2‖_F ≤ tol · ‖A‖_F`, where [`HermitianTol::DEFAULT`] is
    /// `eps(real(D))^(3/4)`, MatrixAlgebraKit's level at unit norm. Unlike
    /// MatrixAlgebraKit's absolute `atol`, the tolerance is relative, so the
    /// decision does not change when `self` is rescaled. In each coupled sector,
    /// `v : codomain(self) <- W` is unitary and `d : W <- W` holds the signed
    /// real eigenvalues. Eigenvalues are stable-sorted by
    /// descending absolute value; each eigenvector's phase is fixed by making
    /// its largest-magnitude component real and non-negative. Bases inside an
    /// exactly degenerate eigenspace remain backend-dependent.
    ///
    /// Both multiplicity-free and checked-Generic `d` factors use compact
    /// diagonal storage. Checked factors retain the exact source provider
    /// `Arc`. An owned Host compact diagonal is read directly into a
    /// permutation eigenbasis, for both multiplicity-free and checked-Generic
    /// providers, after the same finite-input and Hermiticity checks as
    /// [`Self::eigh_vals`]; its dense output still occupies `Σ_c k_c²`
    /// elements. Lazy adjoints use an operation-local dense payload.
    ///
    /// # Errors and cost
    ///
    /// A non-endomorphism, failed Hermiticity check, or dense numerical failure
    /// returns [`Error::Operation`]. Layout failures retain [`Error::Core`],
    /// and multiplicity-free algebra failures retain [`Error::FusionAlgebra`]
    /// where applicable. For checked Generic, the original layout or provider
    /// error is available as the source. It also validates identical full
    /// row/column fusion-tree stacking. Factors are returned only after every
    /// sector succeeds; otherwise the method returns an error and no factors.
    /// The direct compact path sorts in `O(Σ_c k_c log k_c)` time and writes
    /// `O(Σ_c k_c²)` eigenvector elements; the general dense route costs
    /// `O(Σ_c n_c³)` time.
    ///
    /// ```
    /// use std::sync::Arc;
    /// use tenet::sector::{U1FusionRule, U1Irrep};
    /// use tenet::typed::{Eigh, GradedSpace, HermitianTol, Runtime, TensorMap};
    ///
    /// let runtime = Runtime::builder().build()?;
    /// let v = GradedSpace::try_new(Arc::new(U1FusionRule), [(U1Irrep::new(0), 2)])?;
    /// let a: TensorMap<_, f64> = TensorMap::isomorphism(&runtime, [&v], [&v])?.scale(2.0);
    /// let Eigh { d, v: eigenvectors } = a.eigh_full(&[0], &[1], HermitianTol::DEFAULT)?;
    /// let rebuilt = eigenvectors.compose(&d)?.compose(&eigenvectors.adjoint()?)?;
    /// assert!(rebuilt.axpby(1.0, &a, -1.0)?.norm(2.0)? < 1e-12);
    /// # Ok::<(), tenet::typed::Error>(())
    /// ```
    ///
    /// `rows` and `cols` are the leg roles: the operation acts on the matrix
    /// view `self.permute(rows, cols)`, and the current split costs nothing
    /// extra (see [`Self::svd_compact`]'s *Leg roles*).
    pub fn eigh_full(
        &self,
        rows: &[usize],
        cols: &[usize],
        hermitian_tol: HermitianTol,
    ) -> Result<Eigh<Self>, TypedFacadeError<R>> {
        self.with_leg_roles(rows, cols, |t| t.factor_eigh_full(hermitian_tol))
    }
}

impl<R, D> TensorMap<R, D>
where
    R: TypedSectorAdmission,
    R::Mode: FusionMode<R> + TypedTensorTransformDispatch<R, D>,
    D: AdvancedLinalgScalar,
    // Carried across the whole eig row even though this member builds no
    // factor: the three are one API surface.
    <D as FactorScalar>::Eig: TensorScalar,
{
    /// Returns only the general eigenvalues as `Complex64` (at every payload
    /// dtype, like every spectrum of this crate), grouped by
    /// provider-labelled sector and descending by magnitude.
    ///
    /// No eigenvector factor or bond space is built. The input must be an
    /// endomorphism. An owned Host compact diagonal is read directly for both
    /// multiplicity-free and checked-Generic providers: a nonfinite entry
    /// returns [`Error::Operation`] (an `InvalidArgument`: `eig input components must be
    /// finite`; dense input too, after the endomorphism and stacking checks),
    /// and an eigenvalue of infinite magnitude fails the dense
    /// route's eigenvalue check. Lazy adjoints use an operation-local dense
    /// payload.
    ///
    /// `rows` and `cols` are the leg roles: the operation acts on the matrix
    /// view `self.permute(rows, cols)`, and the current split costs nothing
    /// extra (see [`Self::svd_compact`]'s *Leg roles*).
    pub fn eig_vals(
        &self,
        rows: &[usize],
        cols: &[usize],
    ) -> Result<
        Vec<SectorSpectrum<<R as TypedSectorAdmission>::Sector, num_complex::Complex64>>,
        TypedFacadeError<R>,
    > {
        self.with_leg_roles(rows, cols, |t| {
            t.factor_values(FactorOp::EigVals, |lease, source| {
                tenet_matrixalgebra::seam::eig_vals_from_source::<R::Mode, _, _, _, _>(
                    lease, source,
                )
            })
        })
    }
}

impl<R, D> TensorMap<R, D>
where
    R: TypedSectorAdmission,
    R::Mode: FusionMode<R> + TypedTensorTransformDispatch<R, D>,
    D: AdvancedLinalgScalar,
    <D as FactorScalar>::Eig: TensorScalar,
{
    /// Returns the general eigendecomposition `self * v = v * d` as an
    /// [`Eig`] of complex factors.
    ///
    /// For a diagonalizable input this gives
    /// `self = v * d * v^-1`. Both factors are complex even when the input is
    /// real, in the payload's own precision: `D::Eig` is `Complex64` for `f64`
    /// and `Complex64`, and `Complex32` for `f32` and `Complex32`. `v : codomain(self) <- W` holds right eigenvectors and compact
    /// `d : W <- W` holds their eigenvalues. Values are stable-sorted by
    /// descending magnitude, and the largest-magnitude component of each
    /// eigenvector is made real and non-negative. Degenerate eigenbases remain
    /// backend-dependent.
    ///
    /// Both fusion modes promise `self * v ≈ v * d` to backward error and
    /// certify no eigenvector rank, as MatrixAlgebraKit `eig_full!`: a finite
    /// defective input returns its computed, possibly nearly parallel,
    /// eigenvectors. Checked Generic retains the exact source provider `Arc`.
    /// An owned Host compact diagonal reads its eigenvalues directly and
    /// builds a dense permutation factor, for both multiplicity-free and
    /// checked-Generic providers, after the same finite-input and eigenvalue
    /// checks as [`Self::eig_vals`]. Lazy adjoints use an operation-local
    /// dense payload.
    ///
    /// A non-endomorphism, non-finite input, invalid/non-finite eigenvalue,
    /// factor-layout failure, or provider failure returns no factors.
    /// The direct compact path sorts in `O(Σ_c n_c log n_c)` time and writes
    /// `O(Σ_c n_c²)` eigenvector elements; the dense route costs
    /// `O(Σ_c n_c³)` time.
    ///
    /// ```
    /// use std::sync::Arc;
    /// use tenet::sector::{U1FusionRule, U1Irrep};
    /// use tenet::typed::{Eig, GradedSpace, Runtime, TensorMap};
    ///
    /// let runtime = Runtime::builder().build()?;
    /// let v = GradedSpace::try_new(Arc::new(U1FusionRule), [(U1Irrep::new(0), 2)])?;
    /// let a: TensorMap<_, f64> = TensorMap::isomorphism(&runtime, [&v], [&v])?.scale(2.0);
    /// let Eig { d, v: eigenvectors } = a.eig_full(&[0], &[1])?;
    /// let rebuilt = eigenvectors.compose(&d)?.compose(&eigenvectors.inv(&[0], &[1])?)?;
    /// assert!(rebuilt.axpby(1.0.into(), &a.convert::<num_complex::Complex64>(), (-1.0).into())?.norm(2.0)? < 1e-12);
    /// # Ok::<(), tenet::typed::Error>(())
    /// ```
    ///
    /// `rows` and `cols` are the leg roles: the operation acts on the matrix
    /// view `self.permute(rows, cols)`, and the current split costs nothing
    /// extra (see [`Self::svd_compact`]'s *Leg roles*).
    pub fn eig_full(
        &self,
        rows: &[usize],
        cols: &[usize],
    ) -> Result<Eig<TensorMap<R, <D as FactorScalar>::Eig>>, TypedFacadeError<R>> {
        self.with_leg_roles(rows, cols, Self::factor_eig_full)
    }
}

impl<R, D> TensorMap<R, D>
where
    R: TypedSectorAdmission,
    R::Mode: FusionMode<R> + TypedAdjointSpace<R> + TypedTensorTransformDispatch<R, D>,
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
    /// sectors that keep null directions. An owned Host compact diagonal uses a
    /// direct coordinate basis in `O(sum_c k_c + sum_c k_c q_c)` work and
    /// storage, where `k_c` is sector size and `q_c` is nullity: the dense
    /// route's rank cutoff is applied to the magnitudes `|a_i|` directly, and
    /// the null directions are the unit vectors of the magnitudes at or below
    /// it, in descending magnitude. A nonfinite entry, dense or diagonal,
    /// returns [`Error::Operation`] (an `InvalidArgument`: `null input components must be
    /// finite`).
    /// Lazy adjoints use the opposite null space of their
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
        self.with_leg_roles(rows, cols, |t| {
            t.factor_null(
                FactorOp::LeftNull,
                |parent| {
                    parent.factor_null(
                        FactorOp::RightNull,
                        |_| {
                            // A lazy adjoint's parent is owned, so it is
                            // never redirected again.
                            Err(internal_layout_error(
                                "an owned null-space parent was redirected",
                            )
                            .into())
                        },
                        |lease, source| {
                            tenet_matrixalgebra::seam::right_null_from_source::<R::Mode, _, _, _, _>(
                                lease, source,
                            )
                        },
                    )
                },
                |lease, source| {
                    tenet_matrixalgebra::seam::left_null_from_source::<R::Mode, _, _, _, _>(
                        lease, source,
                    )
                },
            )
        })
    }

    /// Returns an orthonormal-row basis `n : W <- domain(self)` for the
    /// numerical right null space, satisfying `self * n^H ~= 0`.
    ///
    /// The fresh bond contains `n_c - rank_c` directions per sector. It uses
    /// the same numerical cutoff and cost as [`Self::left_null`], with rows and
    /// columns exchanged. Host compact diagonals use the direct
    /// coordinate route described above; a lazy
    /// adjoint uses the left null space of its owned parent without
    /// materializing the receiver. Checked results use the source provider instance, and a
    /// failure returns no tensor.
    ///
    /// `rows` and `cols` are the leg roles: the operation acts on the matrix
    /// view `self.permute(rows, cols)`, and the current split costs nothing
    /// extra (see [`Self::svd_compact`]'s *Leg roles*).
    pub fn right_null(&self, rows: &[usize], cols: &[usize]) -> Result<Self, TypedFacadeError<R>> {
        self.with_leg_roles(rows, cols, |t| {
            t.factor_null(
                FactorOp::RightNull,
                |parent| {
                    parent.factor_null(
                        FactorOp::LeftNull,
                        |_| {
                            // A lazy adjoint's parent is owned, so it is
                            // never redirected again.
                            Err(
                                internal_layout_error("an owned null-space parent was redirected")
                                    .into(),
                            )
                        },
                        |lease, source| {
                            tenet_matrixalgebra::seam::left_null_from_source::<R::Mode, _, _, _, _>(
                                lease, source,
                            )
                        },
                    )
                },
                |lease, source| {
                    tenet_matrixalgebra::seam::right_null_from_source::<R::Mode, _, _, _, _>(
                        lease, source,
                    )
                },
            )
        })
    }
}

impl<R, D> TensorMap<R, D>
where
    R: TypedSectorAdmission,
    R::Mode: FusionMode<R> + TypedTensorTransformDispatch<R, D>,
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
    /// composition. An owned compact diagonal stores both factors
    /// compactly; use [`Self::materialize`] for dense buffers. A lazy
    /// adjoint runs the opposite decomposition on its owned parent and returns
    /// detached owned factors without materializing the receiver. Checked factors
    /// use the same provider instance as `self`. If that provider rejects an
    /// output space or any sector computation fails, no factors are returned.
    /// After the block-shape check, a nonfinite entry returns
    /// [`Error::Operation`] (an `InvalidArgument`: `polar input components must be finite`).
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
        self.with_leg_roles(rows, cols, Self::factor_left_polar)
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
        self.with_leg_roles(rows, cols, Self::factor_right_polar)
    }
}

impl<R, D> TensorMap<R, D>
where
    R: TypedSectorAdmission,
    R::Mode: FusionMode<R> + TypedAdjointSpace<R> + TypedTensorTransformDispatch<R, D>,
    D: AdvancedLinalgScalar,
{
    /// TensorKit 0.17 / MatrixAlgebraKit `inv`: the true inverse `t^-1` of a
    /// nonsingular map, defined by `t * t^-1 = id` on the codomain and
    /// `t^-1 * t = id` on the domain. Computed per coupled sector as the exact
    /// dense solve `t_c X_c = 1`, not as a spectral function — there is no
    /// truncation policy to apply and no factor tensor to build.
    ///
    /// # Domain
    ///
    /// TensorKit asks for `codomain ≅ domain` — **isomorphic, not equal** —
    /// and returns a map `domain <- codomain`. This facade's seam agrees: a
    /// rank-one codomain and a rank-two domain with the same coupled-sector
    /// dimensions are accepted, and the result carries the two spaces swapped.
    /// The pin is `inv_accepts_isomorphic_but_unequal_codomain_and_domain`.
    ///
    /// # Errors
    ///
    /// - [`Error::Operation`] with `SpaceMismatch` when the two sides are not
    ///   isomorphic (TensorKit `SpaceMismatch`).
    /// - [`Error::Operation`] with `Dense(NumericalFailure)` when a
    ///   coupled-sector block is singular, in both storages: the dense solve
    ///   fails, and the compact arm finds a zero entry before any solve runs.
    ///   Never a panic; pinned by
    ///   `inv_reports_a_singular_input_as_a_typed_error`.
    ///
    /// # Complexity
    ///
    /// Dense input: `O(Σ_c n_c³)`, one LU solve per coupled sector. Compact
    /// input (a spectrum factor, TensorKit's
    /// `DiagonalTensorMap`): the
    /// **O(rank) elementwise-reciprocal arm**, `1/s_i` over the `Σ_c k_c`
    /// stored values, and the result stays compact — matching TensorKit's
    /// `inv(::DiagonalTensorMap)`, which is `inv.(d.data)`. Nothing dense is
    /// built on either side of that arm. A host lazy adjoint solves its owned
    /// parent and returns a detached owned adjoint of that inverse; it does not
    /// allocate or publish a separate receiver-materialization payload.
    ///
    /// Checked Generic performs the same isomorphism preflight, admits the
    /// swapped output with the source provider `Arc` before output allocation
    /// or dense work, and preserves provider admission failures as typed errors.
    /// Every compact input, in any bond layout and fusion mode, takes the same
    /// elementwise-reciprocal arm, with the same zero-entry error and the same
    /// NaN/infinity pass-through, and stays compact on its own space.
    ///
    /// `rows` and `cols` are the leg roles: the operation acts on the matrix
    /// view `self.permute(rows, cols)`, and the current split costs nothing
    /// extra (see [`Self::svd_compact`]'s *Leg roles*).
    pub fn inv(&self, rows: &[usize], cols: &[usize]) -> Result<Self, TypedFacadeError<R>> {
        self.with_leg_roles(rows, cols, Self::factor_inv)
    }
}

impl<R, D> TensorMap<R, D>
where
    R: TypedSectorAdmission,
    R::Mode: FusionMode<R> + TypedTensorTransformDispatch<R, D>,
    D: AdvancedLinalgScalar,
{
    /// The matrix exponential `exp(t) = Σ_k t^k / k!`, evaluated per coupled
    /// sector — TensorKit's `exp`, which copies and calls `exp!`: check
    /// `domain == codomain`, then exponentiate every block.
    ///
    /// # Domain
    ///
    /// Any endomorphism, of any dtype. Multiplicity-free tensors retain the
    /// original two dense routes:
    ///
    /// - **Hermitian blocks** take the spectral function `V exp(D) Vᴴ` of the
    ///   Hermitian eigendecomposition.
    /// - **Everything else** takes blockwise scaling-and-squaring Padé [13/13]
    ///   (Higham 2005). Non-normal, defective and complex non-Hermitian blocks
    ///   are all in domain; nothing is symmetrized.
    ///
    /// The mathematical and blockwise contract matches the referenced
    /// implementation, but bitwise output does not. TeNeT's general dense route
    /// always uses Padé [13/13], while the referenced implementation delegates
    /// dense algorithm selection and may choose a lower degree for a small-norm
    /// block. The results agree only up to their numerical approximation; pinned
    /// source coordinates are recorded in `tenet/references.md`.
    ///
    /// The **compact** arm is TensorKit's
    /// `exp(::DiagonalTensorMap)`: unconditionally elementwise, with no
    /// hermiticity gate, for every compact input in any bond layout and
    /// fusion mode. Nonfinite entries map through `exp` without an error.
    ///
    /// # Errors
    ///
    /// - [`Error::Operation`] when the input is not an endomorphism
    ///   (`codomain != domain`), when a general block holds a nonfinite entry,
    ///   when a general block's column 1-norm overflows to infinity although
    ///   every entry is finite, or when the backend fails. Nothing is published
    ///   unless every coupled sector succeeded.
    /// - [`Error::Core`] / [`Error::FusionAlgebra`] from the multiplicity-free
    ///   Hermitian composition route.
    ///
    /// # Complexity
    ///
    /// Dense input is `O(Σ_c n_c³)`: multiplicity-free Hermitian blocks use one
    /// eigendecomposition plus a composition; all general blocks, including
    /// checked-Generic, use six GEMMs, one solve and the necessary Padé
    /// squarings per sector with `O(max_c n_c²)` workspace. Coupled sectors are
    /// never mixed. A dense lazy adjoint builds one operation-local logical
    /// payload per call, released with the call. Compact input remains
    /// `O(rank)` elementwise in every fusion mode.
    ///
    /// TensorKit's diagonal implementation is the reference for the compact
    /// branch; this method never panics for a supported tensor contract.
    ///
    /// ```
    /// use std::sync::Arc;
    /// use tenet::sector::{U1FusionRule, U1Irrep};
    /// use tenet::typed::{GradedSpace, Runtime, TensorMap};
    /// let runtime = Runtime::builder().build()?;
    /// let v = GradedSpace::try_new(Arc::new(U1FusionRule), [(U1Irrep::new(0), 2), (U1Irrep::new(1), 1)])?;
    /// let zero: TensorMap<_, f64> = TensorMap::zeros(&runtime, [&v], [&v])?;
    /// let id: TensorMap<_, f64> = TensorMap::isomorphism(&runtime, [&v], [&v])?;
    /// assert!(zero.exp(&[0], &[1])?.dense_data()?.iter().zip(id.dense_data()?).all(|(a, b)| (a - b).abs() < 1e-15));
    /// # Ok::<(), tenet::typed::Error>(())
    /// ```
    ///
    /// `rows` and `cols` are the leg roles: the operation acts on the matrix
    /// view `self.permute(rows, cols)`, and the current split costs nothing
    /// extra (see [`Self::svd_compact`]'s *Leg roles*).
    pub fn exp(&self, rows: &[usize], cols: &[usize]) -> Result<Self, TypedFacadeError<R>> {
        self.with_leg_roles(rows, cols, Self::factor_exp)
    }
}
