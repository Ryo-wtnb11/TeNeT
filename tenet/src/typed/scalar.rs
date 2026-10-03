use super::*;

/// Scalar payloads supported by [`TensorMap`], base capability.
///
/// Admits the payload-dtype-independent half of the typed API: construction
/// and inspection, [`TensorMap::adjoint`], `scale`/`axpby`, the
/// reductions (`norm`, `inner`, `tr`),
/// contraction/`compose`/`otimes`/`cat`, the structural transforms
/// (`permute`, `braid`, `transpose`, `repartition`, `twist`, `flip`),
/// `restrict_leg`/`embed_leg`/`diagview`, trace, and
/// `tensor!` network execution.
///
/// Factorizations need [`FactorizationScalar`]; matrix functions, inverses,
/// solves and the general eigendecomposition need [`AdvancedLinalgScalar`].
///
/// This trait is sealed; the supported scalar types are `f64`,
/// [`num_complex::Complex64`], `f32` and [`num_complex::Complex32`].
/// Admission is split across markers rather than carried by one trait so that
/// a payload dtype joins one family at a time, with its own review and its own
/// tolerance evidence (<https://github.com/Ryo-wtnb11/TeNeT/issues/1065>);
/// single precision has since joined every host family.
///
/// # Precision
///
/// The structural coefficients of a symmetry (F/R symbols, quantum
/// dimensions, fermionic signs) stay `f64` for every payload dtype and act on
/// a single-precision payload through the existing coefficient actions, so an
/// `f32` tensor carries `f32` data recoupled by exactly-computed `f64`
/// algebraic numbers. This follows TensorKit, whose
/// `promote_permute(Float32, I)` stays `Float32` for a real sector scalar
/// type (`indexmanipulations.jl`). A provider with *complex* structural
/// coefficients and a single-precision payload remains a compile-time
/// boundary.
///
/// [`TensorMap::norm`] returns `f64` for every payload dtype and accumulate in `f64`;
/// [`TensorMap::inner`] and [`TensorMap::tr`] return the
/// payload type, as TensorKit's do, but also sum in double precision.
///
/// Tolerances are the caller's: the predicates that take a `tol` have no
/// default, and a tolerance chosen for `f64` (`1e-12` in the examples of this
/// crate) rejects everything at `f32`. Scale it by the payload's own epsilon
/// — `f32::EPSILON` is about `1.2e-7` — as TensorKit's `rtoldefault` does.
///
/// Execution state is per payload dtype, but the *plan* caches are not: plans
/// are structural, so [`Runtime::tree_transform_cache_info`] and
/// [`Runtime::clear_tree_transform_cache`] report and clear single- and
/// double-precision activity together, in one shared store.
///
/// # Annotate the payload dtype
///
/// With `f64` the only real payload, a float literal used to select it on its
/// own. With two, inference waits for the end-of-function `f64` fallback,
/// which arrives too late for method resolution: a payload dtype taken
/// *solely* from float literals, whose result then has a method called on it,
/// now fails with `E0689 ambiguous numeric type {float}`. It affects
/// [`TensorMap::diagonal`], [`TensorMap::from_subblock_fn`] closures returning
/// literals, and `scale`/`axpby` coefficients on an un-annotated
/// `zeros`/`isomorphism`/`rand_with_seed`:
///
/// ```
/// use std::sync::Arc;
/// use tenet::sector::{U1FusionRule, U1Irrep};
/// use tenet::typed::{GradedSpace, Runtime, SectorSpectrum, TensorMap};
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let runtime = Runtime::builder().dense_threads(1).build()?;
/// let leg = GradedSpace::try_new(Arc::new(U1FusionRule), [(U1Irrep::new(0), 2)])?;
/// // Annotated, so the literals have a type to be.
/// let t: TensorMap<U1FusionRule, f64> = TensorMap::diagonal(
///     &runtime,
///     &leg,
///     [SectorSpectrum { sector: U1Irrep::new(0), values: vec![2.0, 3.0] }],
/// )?;
/// assert!((t.tr()?.abs() - 5.0).abs() < 1e-12);
/// # Ok(())
/// # }
/// ```
///
/// Passing such a result to an `f64` parameter, comparing it, or never
/// pinning it still compiles, because the fallback does arrive. Only a method
/// call on it does not.
///
/// # What single precision does *not* reach
///
/// Each rejection below is paired with the same body at `f64`, which is what
/// shows the `compile_fail` is the payload dtype and not an unrelated mistake.
///
/// Factorizations are *not* in this list: single precision reaches
/// [`FactorizationScalar`] under
/// <https://github.com/Ryo-wtnb11/TeNeT/issues/1324>, so `svd_compact` and its
/// siblings compile at `f32` and [`num_complex::Complex32`]. Neither is
/// persistence: the wire format has `f32` and `Complex32` tags since
/// <https://github.com/Ryo-wtnb11/TeNeT/issues/1325> (see
/// [`TypedPersistenceCodec`]).
///
/// Nor is the advanced family: matrix functions, `inv`/`pinv`/`solve` and the
/// general eigendecomposition reach both single-precision payloads under
/// <https://github.com/Ryo-wtnb11/TeNeT/issues/1459> (see
/// [`AdvancedLinalgScalar`]). What stays closed is a provider with complex
/// structural coefficients, described above.
///
/// A device payload is *not* one of the closed gates: every dtype of this
/// marker uploads (<https://github.com/Ryo-wtnb11/TeNeT/issues/1336>) and, since
/// <https://github.com/Ryo-wtnb11/TeNeT/issues/1341>, factorizes on device.
#[cfg_attr(
    feature = "cuda",
    doc = "```
use tenet::sector::U1FusionRule;
use tenet::typed::TensorMap;

fn single_precision_upload(tensor: &TensorMap<U1FusionRule, f32>) {
    let _ = tensor.to_cuda();
}

fn double_precision_upload(tensor: &TensorMap<U1FusionRule, f64>) {
    let _ = tensor.to_cuda();
}
```"
)]
#[allow(private_bounds)]
pub trait TensorScalar: ScalarOps {}

impl TensorScalar for f64 {}
impl TensorScalar for num_complex::Complex64 {}
impl TensorScalar for f32 {}
impl TensorScalar for num_complex::Complex32 {}

/// Scalar payloads admitted to the factorization family.
///
/// Adds, on top of [`TensorScalar`]: QR/LQ (compact and full), SVD (compact,
/// full, values), Hermitian eigendecomposition (full, values), left/right
/// null spaces, and left/right polar.
/// [`GradedSpace::find_truncated`] carries no payload and stays on
/// [`TensorScalar`].
///
/// Sealed through [`TensorScalar`]: implemented for `f64`,
/// [`num_complex::Complex64`], `f32` and [`num_complex::Complex32`]. Matrix
/// functions, `inv`/`pinv`/`solve` and the general eigendecomposition are on
/// [`AdvancedLinalgScalar`].
///
/// # Single-precision tolerances
///
/// Every factorization here is backward stable in the payload's own working
/// precision, so a result computed at `f32`/[`num_complex::Complex32`] agrees
/// with the `f64`/[`num_complex::Complex64`] factorization of the *same*
/// (exactly widened) input only to `eps(real(D))`, not to `f64::EPSILON`:
/// `f32::EPSILON` is `1.19e-7`, about `9e8` times coarser. Concretely,
///
/// * every `tol` is the caller's and has no default. A residual test such as
///   `t.axpby(1, &t.adjoint()?, -1)?.norm(2.0)? <= tol * t.norm(2.0)?.max(1.0)`
///   (TensorKit `ishermitian`) sees a residual that an `f64` payload keeps
///   near `1e-16` sit near `1e-7` in `f32`, so a `tol` chosen for `f64` —
///   `1e-12` in most of the doctests here — rejects a perfectly good
///   single-precision result.
///   Scale it with the payload: MatrixAlgebraKit's `defaulttol`
///   (`src/common/defaults.jl`) is `eps(real(T))^(2/3)`, which is `3.7e-11` at
///   `f64` and `2.4e-5` at `f32`; TensorKit's `isapprox` default is
///   `sqrt(eps(real(T)))`, `1.5e-8` and `3.5e-4`;
/// * the truncation policies that take tolerances
///   ([`Truncation::relative_cutoff`], [`Truncation::relative_error`]) are the same
///   story: a cutoff below the payload's own noise floor keeps noise. The
///   spectra themselves, the truncation `error` and every norm this crate
///   returns are `f64` at *every* payload dtype (they are widened, not
///   recomputed), so the `f64` type of a tolerance says nothing about the
///   precision of the values it is compared against;
/// * **the kept/discarded sets near a tie may legitimately differ from the
///   double-precision run of the same physics.** The singular values and
///   Hermitian eigenvalues of a single-precision block carry a relative error
///   of order `eps(f32)` times the condition number, so two values separated
///   by less than that are not ordered reliably, and a magnitude-driven
///   policy ([`Truncation::rank`], [`Truncation::relative_cutoff`],
///   [`Truncation::relative_error`]) can act on a different order.
///
///   What holds at every payload dtype is the policy's **postcondition
///   against the spectrum that run actually computed**: a kept value is at or
///   above the threshold of that run, and the reported `error` is the weighted
///   2-norm of what that run discarded, within its budget. What does *not*
///   carry across dtypes is a comparison of the two runs' outcomes, and how
///   far it fails depends on the policy:
///
///   * [`Truncation::rank`] at a tie swaps two interchangeable states, so the
///     kept count is the budget either way and the discarded weight agrees to
///     the accuracy of the values themselves;
///   * [`Truncation::relative_cutoff`] and [`Truncation::relative_error`] have
///     a *boundary*, not a tie: a value within noise of the threshold, or a
///     tail whose cumulative weight is within noise of the budget, is kept by
///     one run and dropped by the other. Then the kept count differs by one
///     state and the discarded weight differs by that whole state's weight —
///     not by `eps * cond`.
///
///   Separate the spectrum by more than `eps(real(D)) * cond`, and keep the
///   budget away from a cumulative-weight boundary by the same margin, if the
///   outcome has to be reproducible across dtypes.
///
/// # Source compatibility
///
/// The `E0689` note on [`TensorScalar`] applies here too, one level further
/// in: a payload dtype taken *solely* from float literals used to be pinned to
/// `f64` by calling a factorization on it, because `f64` and
/// [`num_complex::Complex64`] were the only implementors. With four, inference
/// waits for the end-of-function fallback, which is too late for a method call
/// on the result — `t.eigh_full(&[0], &[1])?.d.diagview()?[0].values[0].abs()` on an
/// un-annotated `from_subblock_fn` tensor now needs the payload dtype written
/// down. The same holds for [`GradedSpace::find_truncated`], whose spectrum
/// type is `SpectrumMagnitude` and now has four implementors.
///
/// A caller generic over the base marker cannot reach a factorization:
///
/// ```compile_fail
/// use tenet::sector::U1FusionRule;
/// use tenet::typed::{TensorMap, TensorScalar};
///
/// fn base_only<D: TensorScalar>(tensor: &TensorMap<U1FusionRule, D>) {
///     let _ = tensor.svd_compact(&[0], &[1]);
/// }
/// ```
///
/// The same body compiles once the caller asks for this marker, which is what
/// shows the rejection above is the bound and not an unrelated mistake:
///
/// ```
/// use tenet::sector::U1FusionRule;
/// use tenet::typed::{FactorizationScalar, TensorMap};
///
/// fn factorizing<D: FactorizationScalar>(tensor: &TensorMap<U1FusionRule, D>) {
///     let _ = tensor.svd_compact(&[0], &[1]);
/// }
/// ```
///
/// The same pair for `qr_compact`:
///
/// ```compile_fail
/// use tenet::sector::U1FusionRule;
/// use tenet::typed::{TensorMap, TensorScalar};
///
/// fn base_only<D: TensorScalar>(tensor: &TensorMap<U1FusionRule, D>) {
///     let _ = tensor.qr_compact(&[0], &[1]);
/// }
/// ```
///
/// ```
/// use tenet::sector::U1FusionRule;
/// use tenet::typed::{FactorizationScalar, TensorMap};
///
/// fn factorizing<D: FactorizationScalar>(tensor: &TensorMap<U1FusionRule, D>) {
///     let _ = tensor.qr_compact(&[0], &[1]);
/// }
/// ```
///
/// This marker still does not *by itself* carry the device factorizations:
/// those are `CudaFactorizationPayload`, which every payload of this marker now
/// implements (<https://github.com/Ryo-wtnb11/TeNeT/issues/1341>), device QR
/// included (<https://github.com/Ryo-wtnb11/TeNeT/issues/1271>). The examples below
/// only compile under the `cuda` feature, so a host-only CI run does not
/// exercise them.
#[cfg_attr(
    feature = "cuda",
    doc = "```
use tenet::sector::U1FusionRule;
use tenet::typed::{CudaStorage, TensorMap};

fn f32_device_svd(tensor: &TensorMap<U1FusionRule, f32, CudaStorage<f32>>) {
    let _ = tensor.svd_compact(&[0], &[1]);
}

fn f64_device_svd(tensor: &TensorMap<U1FusionRule, f64, CudaStorage<f64>>) {
    let _ = tensor.svd_compact(&[0], &[1]);
}
```

```
use num_complex::Complex32;
use tenet::sector::U1FusionRule;
use tenet::typed::{CudaStorage, TensorMap};

fn c32_device_qr(tensor: &TensorMap<U1FusionRule, Complex32, CudaStorage<Complex32>>) {
    let _ = tensor.qr_compact(&[0], &[1]);
}
```"
)]
pub trait FactorizationScalar: TensorScalar {}

impl FactorizationScalar for f64 {}
impl FactorizationScalar for num_complex::Complex64 {}
impl FactorizationScalar for f32 {}
impl FactorizationScalar for num_complex::Complex32 {}

/// Scalar payloads admitted to the advanced linear-algebra family.
///
/// Adds, on top of [`FactorizationScalar`]: the matrix functions
/// ([`TensorMap::exp`]), [`TensorMap::inv`], [`TensorMap::pinv`],
/// [`TensorMap::solve`], and the general (non-Hermitian)
/// eigendecomposition (`eig_full`, `eig_vals`).
///
/// These are the operations whose accuracy depends on conditioning rather than
/// on one backend call, so they are the last family a new payload dtype joins.
///
/// Sealed through [`TensorScalar`]: implemented for `f64`,
/// [`num_complex::Complex64`], `f32` and [`num_complex::Complex32`] (single
/// precision since <https://github.com/Ryo-wtnb11/TeNeT/issues/1459>).
///
/// # Single precision
///
/// Every operation runs in the payload's own precision; nothing is silently
/// widened. The general eigendecomposition returns its factors in
/// `D::Eig` — [`num_complex::Complex32`] for an `f32` or
/// [`num_complex::Complex32`] payload — on both the multiplicity-free and the
/// Checked-Generic dispatch, while the reported eigenvalue list (`eig_vals`)
/// stays `Complex64` at every payload dtype, as every spectrum and norm of
/// this crate does.
///
/// A single-precision result agrees with the double-precision result of the
/// same input only to `eps(f32)` times the conditioning of the operation:
/// `inv`, `solve`, `pinv` and negative powers carry the condition number of
/// the divisor, and eigenvalues that of the eigenbasis. `pinv`'s `rcond`
/// is the caller's; one below `eps(f32)` keeps noise. Unlike TensorKit, whose
/// ordinary test matrix skips `\` at `Float32`/`ComplexF32`, `solve` is
/// admitted: it is one LU solve per coupled block, backward stable at any
/// working precision, and tested against the widened oracle.
///
/// Neither the base marker nor [`FactorizationScalar`] reaches a matrix
/// function:
///
/// ```compile_fail
/// use tenet::sector::U1FusionRule;
/// use tenet::typed::{TensorMap, TensorScalar};
///
/// fn base_only<D: TensorScalar>(tensor: &TensorMap<U1FusionRule, D>) {
///     let _ = tensor.exp(&[0], &[1]);
/// }
/// ```
///
/// ```compile_fail
/// use tenet::sector::U1FusionRule;
/// use tenet::typed::{FactorizationScalar, TensorMap};
///
/// fn factorizing_only<D: FactorizationScalar>(tensor: &TensorMap<U1FusionRule, D>) {
///     let _ = tensor.exp(&[0], &[1]);
/// }
/// ```
///
/// This marker does:
///
/// ```
/// use tenet::sector::U1FusionRule;
/// use tenet::typed::{AdvancedLinalgScalar, TensorMap};
///
/// fn advanced<D: AdvancedLinalgScalar>(tensor: &TensorMap<U1FusionRule, D>) {
///     let _ = tensor.exp(&[0], &[1]);
///     let _ = tensor.inv(&[0], &[1]);
///     let _ = tensor.solve(&[0], &[1], tensor, &[0], &[1]);
/// }
/// ```
///
/// [`TensorMap::inv`] on its own, so the pair above cannot pass on `exp` alone:
///
/// ```compile_fail
/// use tenet::sector::U1FusionRule;
/// use tenet::typed::{FactorizationScalar, TensorMap};
///
/// fn factorizing_only<D: FactorizationScalar>(tensor: &TensorMap<U1FusionRule, D>) {
///     let _ = tensor.inv(&[0], &[1]);
/// }
/// ```
///
/// ```
/// use tenet::sector::U1FusionRule;
/// use tenet::typed::{AdvancedLinalgScalar, TensorMap};
///
/// fn advanced<D: AdvancedLinalgScalar>(tensor: &TensorMap<U1FusionRule, D>) {
///     let _ = tensor.inv(&[0], &[1]);
/// }
/// ```
///
/// And the general eigendecomposition, whose `D::Eig: TensorScalar` bound is
/// held constant across the pair so that only the marker differs:
///
/// ```compile_fail
/// use tenet::sector::U1FusionRule;
/// use tenet::typed::{FactorizationScalar, TensorMap, TensorScalar};
///
/// fn factorizing_only<D>(tensor: &TensorMap<U1FusionRule, D>)
/// where
///     D: FactorizationScalar,
///     D::Eig: TensorScalar,
/// {
///     let _ = tensor.eig_full(&[0], &[1]);
/// }
/// ```
///
/// ```
/// use tenet::sector::U1FusionRule;
/// use tenet::typed::{AdvancedLinalgScalar, TensorMap, TensorScalar};
///
/// fn advanced<D>(tensor: &TensorMap<U1FusionRule, D>)
/// where
///     D: AdvancedLinalgScalar,
///     D::Eig: TensorScalar,
/// {
///     let _ = tensor.eig_full(&[0], &[1]);
/// }
/// ```
pub trait AdvancedLinalgScalar: FactorizationScalar {}

impl AdvancedLinalgScalar for f64 {}
impl AdvancedLinalgScalar for num_complex::Complex64 {}
impl AdvancedLinalgScalar for f32 {}
impl AdvancedLinalgScalar for num_complex::Complex32 {}

/// Scalar payloads a [`CudaStorage`] device buffer can own.
///
/// This bundles the typed payload trait [`TensorScalar`] (whose `ScalarOps`
/// half selects the matching multiplicity-free execution context) with the
/// device dtype [`crate::typed::CudaScalar`]. All four payload dtypes of the
/// base family are admitted: `f64`, [`num_complex::Complex64`], `f32` and
/// [`num_complex::Complex32`].
///
/// This marker carries the *base* device family only — transfer, arithmetic,
/// the reductions, contraction/`compose` and the structural transforms. The
/// device factorizations, QR included, sit behind [`CudaFactorizationPayload`]
/// on top of it.
#[cfg(feature = "cuda")]
#[doc(hidden)]
pub trait CudaPayload: TensorScalar + tenet_dense::CudaScalar {}

#[cfg(feature = "cuda")]
impl CudaPayload for f64 {}
#[cfg(feature = "cuda")]
impl CudaPayload for num_complex::Complex64 {}
#[cfg(feature = "cuda")]
impl CudaPayload for f32 {}
#[cfg(feature = "cuda")]
impl CudaPayload for num_complex::Complex32 {}

/// Device payloads admitted to the *device* factorization family
/// (`svd_compact`, `eigh_full`, `qr_compact`).
///
/// The marker exists so that admitting a dtype to the device *base* family and
/// to the *host* factorization family cannot, together, open the device
/// factorizations by accident: those two admissions say nothing about the
/// device SVD gauge, the device Hermitian admission rule or the host-side
/// truncation composition at that dtype. All four payloads are admitted here
/// since <https://github.com/Ryo-wtnb11/TeNeT/issues/1341> (leaf C4) supplied
/// that evidence for `f32` and [`num_complex::Complex32`].
/// Device `qr_compact` joined all four in
/// <https://github.com/Ryo-wtnb11/TeNeT/issues/1271>.
///
/// Sealed through [`CudaPayload`].
#[cfg(feature = "cuda")]
#[doc(hidden)]
pub trait CudaFactorizationPayload: CudaPayload + FactorizationScalar {}

#[cfg(feature = "cuda")]
impl CudaFactorizationPayload for f64 {}
#[cfg(feature = "cuda")]
impl CudaFactorizationPayload for num_complex::Complex64 {}
#[cfg(feature = "cuda")]
impl CudaFactorizationPayload for f32 {}
#[cfg(feature = "cuda")]
impl CudaFactorizationPayload for num_complex::Complex32 {}

/// Internal scalar operations shared by typed tensor execution.
/// The reduction accumulator is [`tenet_tensors::WideScalar::Wide`], which
/// owns that decision for every crate: `Self` for the double-precision pair
/// (so those reductions are unchanged, down to the emitted arithmetic) and the
/// double-precision scalar of the same field for a single-precision payload.
/// `Wide: FactorScalar` because the typed reductions form conjugated products
/// in the accumulator and widen it to `Complex64` at the end.
///
/// TensorKit reaches comparable accuracy differently — `LinearAlgebra.norm`
/// scales each block — so accumulating wide is a deliberate TeNeT choice,
/// recorded in <https://github.com/Ryo-wtnb11/TeNeT/issues/1315>.
pub(crate) trait ScalarOps:
    FactorScalar
    + tenet_tensors::RecouplingCoefficientAction<f64>
    + tenet_tensors::ZeroBytes
    + tenet_tensors::WideScalar<Wide: FactorScalar>
{
    /// Whether `FactorScalar::adjoint` is the identity (a real dtype).
    const CONJUGATION_IS_IDENTITY: bool;

    /// Returns the execution lane for this payload dtype, building it if the
    /// runtime has not needed it yet.
    ///
    /// # Errors
    ///
    /// Propagates the dense-executor construction error of a deferred lane.
    fn ctx_of<Key: Clone + Eq + Hash + Send + Sync + 'static>(
        ctxs: &mut Ctxs<Key>,
    ) -> Result<&mut Ctx<Self, Key>, Error>;
    fn rand_unit(state: &mut u64) -> Self;
    fn abs_value(self) -> f64;
    fn exp_value(self) -> Self;
    fn recip_value(self) -> Self;
}

pub(super) type CheckedGenericSpectrumResult<R, V> = Result<
    Vec<SectorSpectrum<<R as TypedSectorAdmission>::Sector, V>>,
    GenericTensorError<<R as CheckedGenericFusion>::Error>,
>;

impl<R, D> TensorMap<R, D>
where
    R: TypedSectorAdmission,
    R::Mode: TypedTensorSolveDispatch<R, D> + TypedTensorTransformDispatch<R, D>,
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
    /// diagonal divisor instead applies its elementwise reciprocal: a compact
    /// right-hand side on the same bond gives a compact quotient, and a dense
    /// one has its leading (bond) axis scaled, `O(Σ_c k_c m_c)` with no LU.
    /// Checked Generic takes that arm only for an aligned bond layout whose
    /// right-hand side space is the destination, and otherwise materializes
    /// both operands for the dense route. An exact zero divisor entry is the
    /// dense route's singular-block operation error in every mode. A compact
    /// right-hand side of a dense divisor is densified into the solve buffer.
    /// Lazy adjoints are materialized only for this call.
    ///
    /// # Errors
    ///
    /// Returns [`Error::RuntimeMismatch`] or [`Error::RuleMismatch`] for
    /// incompatible operands, [`Error::InvalidArgument`] for unequal
    /// codomains, and an operation error when the divisor is not isomorphic or
    /// a sector is singular. If a checked provider rejects the output space,
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
    /// split of a compact divisor, which reads it directly. A dense divisor
    /// or either moved role pair would copy the view and returns
    /// [`Error::Unsupported`]. Pass `&t.adjoint()?.materialize()?` instead.
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
        // Permuting a borrowed view would copy it before the solve dispatch
        // reaches its own view guard. A moved compact divisor also becomes dense.
        if !self.axes_are_identity(rows, cols) || !rhs.axes_are_identity(rhs_rows, rhs_cols) {
            let refusal = rhs.refuse_borrowed_view("solve");
            if refusal.is_err() && !self.runtime.same_runtime(&rhs.runtime) {
                return Err(Error::RuntimeMismatch.into());
            }
            refusal?;
        }
        self.with_leg_roles(rows, cols, |lhs| {
            rhs.with_leg_roles(rhs_rows, rhs_cols, |right| {
                <R::Mode as TypedTensorSolveDispatch<R, D>>::solve(lhs, right)
            })
        })
    }
}

impl<R, D> TensorMap<R, D>
where
    R: MultiplicityFreeRigidSymbols
        + PhysicalFusionBasis<Scalar = <R as MultiplicityFreeFusionSymbols>::Scalar>,
    D: TensorScalar
        + tenet_tensors::RecouplingCoefficientAction<<R as MultiplicityFreeFusionSymbols>::Scalar>,
{
    /// Expands the reduced coupled-layout payload into an owned physical-basis
    /// tensor on the Host.
    ///
    /// The returned axes and entries follow [`PhysicalDense`]. The provider's
    /// carrier basis and fusion coefficients determine the embedding; no
    /// symmetry-specific dispatch occurs in this method.
    ///
    /// # Cost
    ///
    /// A lazy adjoint or a compact diagonal is first densified into an
    /// operation-local reduced buffer, released on return: one copy of the
    /// reduced payload, never larger than the physical output. TensorKit's
    /// `convert(Array, t)` pays the same copy through `t[f₁, f₂]`.
    ///
    /// # Index order matches TensorKit
    ///
    /// Each axis is ordered as TensorKit's `convert(Array, t)` orders it: a
    /// dual space `V'` is listed in `V`'s order, so bending a leg with
    /// [`Self::permute`] or [`Self::repartition`] leaves its index order
    /// unchanged. For the built-in U(1) and SU(2) providers, whose braiding is
    /// bosonic, the expansion of a permuted tensor is exactly the matching
    /// axis permutation of this array.
    ///
    /// ```
    /// use std::sync::Arc;
    /// use tenet::sector::{U1FusionRule, U1Irrep};
    /// use tenet::typed::{Error, GradedSpace, Runtime, TensorMap};
    ///
    /// let rt = Runtime::builder().build()?;
    /// let v = GradedSpace::try_new(Arc::new(U1FusionRule), [-1, 0, 1].map(|q| (U1Irrep::new(q), 1)))?;
    /// let id = TensorMap::<U1FusionRule, f64>::isomorphism(&rt, [&v], [&v])?;
    ///
    /// // `V <- V`.
    /// let matrix = id.to_physical_dense().expect("U(1) has a physical basis");
    /// assert_eq!(matrix.data, [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]);
    ///
    /// // `V ⊗ V' <- ()`: the same entries.
    /// let bent = id.permute(&[0, 1], &[])?;
    /// let vector = bent.to_physical_dense().expect("U(1) has a physical basis");
    /// assert_eq!(vector.data, matrix.data);
    /// # Ok::<(), Error>(())
    /// ```
    ///
    /// Providers opt in at compile time by implementing
    /// [`PhysicalFusionBasis`]:
    ///
    /// ```compile_fail
    /// use tenet::sector::Z2FusionRule;
    /// use tenet::typed::TensorMap;
    ///
    /// fn unsupported(tensor: &TensorMap<Z2FusionRule, f64>) {
    ///     let _ = tensor.to_physical_dense();
    /// }
    /// ```
    pub fn to_physical_dense(
        &self,
    ) -> Result<PhysicalDense<D>, PhysicalDenseError<<R as PhysicalFusionBasis>::Error>> {
        // Why an operation-local copy rather than `Unsupported`: the physical
        // output is never smaller than the reduced payload, so the copy at
        // most doubles the bytes this conversion already writes, and
        // `PhysicalDenseError` has no `Unsupported` variant to refuse with.
        let adjoint;
        let payload = match &self.repr {
            TypedTensorRepr::Owned(body) => body.materialized_dense_data(),
            TypedTensorRepr::Adjoint(view) => {
                #[cfg(test)]
                observe_adjoint_materialization();
                adjoint = tenet_tensors::materialize_adjoint_data_dyn(
                    view.parent.space.space(),
                    view.logical_space.space(),
                    view.parent_data(),
                )?;
                std::borrow::Cow::Borrowed(adjoint.as_slice())
            }
        };
        let source = BoundDynamicTensorRef::try_new(self.logical_space(), &payload)?;
        let (shape, data) = expand_physical_host(source)?;
        Ok(PhysicalDense { shape, data })
    }

    /// Projects physical-basis data into this tensor's exact reduced schema.
    ///
    /// `self` supplies the runtime, provider allocation, fusion-tree layout,
    /// offsets, and strides. Its numeric payload is not read or modified. All
    /// validation and provider queries finish before the returned tensor is
    /// constructed.
    pub fn project_physical_dense(
        &self,
        physical: &PhysicalDense<D>,
    ) -> Result<Self, PhysicalDenseError<<R as PhysicalFusionBasis>::Error>> {
        let data = project_physical_host(
            self.logical_space(),
            physical.shape.as_slice(),
            physical.data.as_slice(),
        )?;
        Ok(Self {
            runtime: self.runtime.clone(),
            repr: owned_repr(TypedTensorBody::dense(self.logical_space().clone(), data)),
        })
    }
}

impl<R, D> TensorMap<R, D>
where
    R: TypedSectorAdmission,
    R::Mode: TypedTensorPinvDispatch<R, D> + TypedTensorTransformDispatch<R, D>,
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
    /// - A non-finite compact entry on checked Generic retains the dense SVD
    ///   route and its typed [`Error::Operation`] failure; the multiplicity-free
    ///   compact arm returns [`Error::InvalidArgument`]. Checked finite entries
    ///   that require the dense route inherit its dense error behavior.
    /// - [`Error::Operation`] / [`Error::Core`] from dense SVD or recomposition.
    ///
    /// There is no singular-input failure: sending the offending directions to
    /// zero is what a pseudo-inverse is for.
    ///
    /// # Complexity and storage
    ///
    /// Dense input uses one compact SVD per nonempty coupled sector,
    /// `O(Σ_c n_c³)`, then folds `S⁺` into a column scaling and recomposes with
    /// one local GEMM. An admitted finite Host compact diagonal input stays
    /// compact in multiplicity-free mode. Checked Generic also requires a
    /// matching source/output layout and retained values with normal magnitudes
    /// and finite reciprocals; otherwise it follows the dense route. For `K`
    /// stored values, `B` checked
    /// source/output blocks, and `G` sectors, its elementwise cutoff and
    /// reciprocal take `O(K + B + G)` time and `O(K + G)` result space.
    ///
    /// Checked Generic admits the swapped output with the source's exact
    /// provider `Arc` and validates its identity, HomSpace, rank, and layout
    /// before any SVD/GEMM. The compact arm also checks both source and
    /// swapped-output coupled-sector layouts. Ineligible layouts retain the
    /// dense route. A checked lazy adjoint is materialized operation-locally.
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
        self.with_leg_roles(rows, cols, |t| {
            <R::Mode as TypedTensorPinvDispatch<R, D>>::pinv(t, rcond)
        })
    }
}

impl<R, D> TensorMap<R, D>
where
    R: TypedSectorAdmission<
            Error = <R as CheckedGenericFusion>::Error,
            Mode = CheckedGenericAdmissionMode,
        > + CheckedGenericFusion,
    D: AdvancedLinalgScalar,
{
    /// Checked-Generic general eigenvalues for owned host tensors.
    pub(super) fn eig_vals_checked_generic(
        &self,
    ) -> CheckedGenericSpectrumResult<R, num_complex::Complex64> {
        let TypedTensorRepr::Owned(body) = &self.repr else {
            return Err(GenericTensorError::Facade(Error::InvalidArgument(
                "checked Generic eig_vals does not accept lazy adjoints".to_string(),
            )));
        };
        let direct = if let TypedData::Diagonal(spectrum) = body.data.as_ref() {
            if is_diagonal_bond_space(body.space.space()) {
                tenet_matrixalgebra::seam::eig_vals_diagonal_dyn(&body.space, spectrum).map_err(
                    |error| GenericTensorError::Plan(CheckedGenericPlanError::Operation(error)),
                )?
            } else {
                None
            }
        } else {
            None
        };
        let raw = if let Some(raw) = direct {
            raw
        } else {
            let mut dense = self.runtime.lease_dense();
            let payload = body.materialized_dense_data();
            let input = BoundDynamicTensorRef::try_new(&body.space, &payload)
                .map_err(|error| GenericTensorError::Facade(error.into()))?;
            tenet_matrixalgebra::seam::eig_vals_dyn_checked_generic(dense.dense(), &input)?
        };
        let provider = self.logical_space().provider();
        let mut decoded = raw
            .into_iter()
            .map(|entry| {
                Ok(SectorSpectrum {
                    sector: provider.try_decode_label(entry.sector)?,
                    values: entry.values,
                })
            })
            .collect::<Result<Vec<_>, <R as TypedSectorAdmission>::Error>>()
            .map_err(|error| GenericTensorError::Plan(CheckedGenericPlanError::Provider(error)))?;
        decoded.sort_by(|left, right| left.sector.cmp(&right.sector));
        Ok(decoded)
    }
}

impl<R, D> TensorMap<R, D>
where
    R: TypedSectorAdmission<
            Error = <R as CheckedGenericFusion>::Error,
            Mode = CheckedGenericAdmissionMode,
        > + CheckedGenericFusion,
    D: FactorizationScalar,
{
    /// Checked-Generic Hermitian eigenvalues for owned host tensors.
    pub(super) fn eigh_vals_checked_generic(&self) -> CheckedGenericSpectrumResult<R, f64> {
        let TypedTensorRepr::Owned(body) = &self.repr else {
            return Err(GenericTensorError::Facade(Error::InvalidArgument(
                "checked Generic eigh_vals does not accept lazy adjoints".to_string(),
            )));
        };
        let direct = if let TypedData::Diagonal(spectrum) = body.data.as_ref() {
            if is_diagonal_bond_space(body.space.space()) {
                tenet_matrixalgebra::seam::eigh_vals_diagonal_dyn(&body.space, spectrum).map_err(
                    |error| GenericTensorError::Plan(CheckedGenericPlanError::Operation(error)),
                )?
            } else {
                None
            }
        } else {
            None
        };
        let raw = if let Some(raw) = direct {
            raw
        } else {
            let mut dense = self.runtime.lease_dense();
            let payload = body.materialized_dense_data();
            let input = BoundDynamicTensorRef::try_new(&body.space, &payload)
                .map_err(|error| GenericTensorError::Facade(error.into()))?;
            tenet_matrixalgebra::seam::eigh_vals_dyn_checked_generic(dense.dense(), &input)?
        };
        let provider = self.logical_space().provider();
        let mut decoded = raw
            .into_iter()
            .map(|entry| {
                Ok(SectorSpectrum {
                    sector: provider.try_decode_label(entry.sector)?,
                    values: entry.values,
                })
            })
            .collect::<Result<Vec<_>, <R as TypedSectorAdmission>::Error>>()
            .map_err(|error| GenericTensorError::Plan(CheckedGenericPlanError::Provider(error)))?;
        decoded.sort_by(|left, right| left.sector.cmp(&right.sector));
        Ok(decoded)
    }
}

impl<R, D> TensorMap<R, D>
where
    R: TypedSectorAdmission<
            Error = <R as CheckedGenericFusion>::Error,
            Mode = CheckedGenericAdmissionMode,
        > + CheckedGenericFusion,
    D: FactorizationScalar,
{
    pub(super) fn eigh_full_checked_generic(
        &self,
    ) -> Result<Eigh<Self>, GenericTensorError<<R as CheckedGenericFusion>::Error>> {
        if matches!(&self.repr, TypedTensorRepr::Adjoint(_)) {
            return self
                .materialized_tensor_uncached()
                .map_err(GenericTensorError::Facade)?
                .eigh_full_checked_generic();
        }
        let body = self.owned_body().expect("owned checked Generic EIGH input");
        let direct = match body.data.as_ref() {
            TypedData::Diagonal(spectrum) if is_diagonal_bond_space(body.space.space()) => {
                tenet_matrixalgebra::seam::eigh_full_diagonal_dyn_checked_generic(
                    &body.space,
                    spectrum,
                )?
            }
            _ => None,
        };
        let out = if let Some(out) = direct {
            out
        } else {
            let mut dense = self.runtime.lease_dense();
            let payload = body.materialized_dense_data();
            let input = BoundDynamicTensorRef::try_new(&body.space, &payload)
                .map_err(|error| GenericTensorError::Facade(error.into()))?;
            tenet_matrixalgebra::seam::eigh_full_dyn_checked_generic(dense.dense(), &input)?
        };
        let (v, mut eigenvalues) = out.into_parts();
        let d = diagonal_factor_on_checked(
            &self.runtime,
            Arc::clone(body.space.provider_arc()),
            &mut eigenvalues,
            D::from_real,
        )?;
        Ok(Eigh {
            d,
            v: wrap_factor_on(&self.runtime, v),
        })
    }
}

impl<R, D> TensorMap<R, D>
where
    R: TypedSectorAdmission<
            Error = <R as CheckedGenericFusion>::Error,
            Mode = CheckedGenericAdmissionMode,
        > + CheckedGenericFusion,
    D: AdvancedLinalgScalar,
    <D as FactorScalar>::Eig: TensorScalar,
{
    #[expect(
        clippy::type_complexity,
        reason = "the checked eigensolver returns its diagonal and eigenvector factors together"
    )]
    pub(super) fn eig_full_checked_generic(
        &self,
    ) -> Result<
        Eig<TensorMap<R, <D as FactorScalar>::Eig>>,
        GenericTensorError<<R as CheckedGenericFusion>::Error>,
    > {
        if matches!(&self.repr, TypedTensorRepr::Adjoint(_)) {
            return self
                .materialized_tensor_uncached()
                .map_err(GenericTensorError::Facade)?
                .eig_full_checked_generic();
        }
        let body = self.owned_body().expect("owned checked Generic EIG input");
        let direct = match body.data.as_ref() {
            TypedData::Diagonal(spectrum) if is_diagonal_bond_space(body.space.space()) => {
                tenet_matrixalgebra::seam::eig_full_diagonal_dyn_checked_generic(
                    &body.space,
                    spectrum,
                )?
            }
            _ => None,
        };
        let out = if let Some(out) = direct {
            out
        } else {
            let mut dense = self.runtime.lease_dense();
            let payload = body.materialized_dense_data();
            let input = BoundDynamicTensorRef::try_new(&body.space, &payload)
                .map_err(|error| GenericTensorError::Facade(error.into()))?;
            tenet_matrixalgebra::seam::eig_full_dyn_checked_generic(dense.dense(), &input)?
        };
        let (v, mut eigenvalues) = out.into_parts();
        let d = diagonal_factor_on_checked(
            &self.runtime,
            Arc::clone(body.space.provider_arc()),
            &mut eigenvalues,
            <<D as FactorScalar>::Eig as FactorScalar>::from_complex64,
        )?;
        Ok(Eig {
            d,
            v: wrap_factor_on(&self.runtime, v),
        })
    }
}

impl<R, D> TensorMap<R, D>
where
    R: TypedSectorAdmission<
            Error = <R as CheckedGenericFusion>::Error,
            Mode = CheckedGenericAdmissionMode,
        > + CheckedGenericFusion,
    D: FactorizationScalar,
{
    /// Checked-Generic full QR for owned host tensors.
    pub(super) fn qr_full_checked_generic(
        &self,
    ) -> Result<Qr<Self>, GenericTensorError<<R as CheckedGenericFusion>::Error>> {
        let TypedTensorRepr::Owned(body) = &self.repr else {
            return Err(GenericTensorError::Facade(Error::InvalidArgument(
                "checked Generic qr_full does not accept lazy adjoints".to_string(),
            )));
        };
        if let TypedData::Diagonal(spectrum) = body.data.as_ref() {
            if let Some((q_space, r_space, phases, magnitudes)) =
                tenet_matrixalgebra::seam::qr_diagonal_dyn_checked_generic(
                    &body.space,
                    spectrum,
                    true,
                )?
            {
                return Ok(Qr {
                    q: self.with_spectrum_on(q_space, phases),
                    r: self.with_spectrum_on(r_space, magnitudes),
                });
            }
        }
        let mut dense = self.runtime.lease_dense();
        let payload = body.materialized_dense_data();
        let input = BoundDynamicTensorRef::try_new(&body.space, &payload)
            .map_err(|error| GenericTensorError::Facade(error.into()))?;
        let Qr { q, r } =
            tenet_matrixalgebra::seam::qr_full_dyn_checked_generic(dense.dense(), &input)?;
        Ok(Qr {
            q: wrap_factor_on(&self.runtime, q),
            r: wrap_factor_on(&self.runtime, r),
        })
    }
}

impl<R, D> TensorMap<R, D>
where
    R: TypedSectorAdmission<
            Error = <R as CheckedGenericFusion>::Error,
            Mode = CheckedGenericAdmissionMode,
        > + CheckedGenericFusion,
    D: FactorizationScalar,
{
    /// Checked-Generic singular values only for owned host tensors.
    pub(super) fn svd_vals_checked_generic(&self) -> CheckedGenericSpectrumResult<R, f64> {
        let TypedTensorRepr::Owned(body) = &self.repr else {
            return Err(GenericTensorError::Facade(Error::InvalidArgument(
                "checked Generic svd_vals does not accept lazy adjoints".to_string(),
            )));
        };
        let direct = if let TypedData::Diagonal(spectrum) = body.data.as_ref() {
            if is_diagonal_bond_space(body.space.space()) {
                tenet_matrixalgebra::seam::svd_vals_compact_diagonal_dyn(&body.space, spectrum)
                    .map_err(|error| {
                        GenericTensorError::Plan(CheckedGenericPlanError::Operation(error))
                    })?
            } else {
                None
            }
        } else {
            None
        };
        let raw = if let Some(raw) = direct {
            raw
        } else {
            let mut dense = self.runtime.lease_dense();
            let payload = body.materialized_dense_data();
            let input = BoundDynamicTensorRef::try_new(&body.space, &payload)
                .map_err(|error| GenericTensorError::Facade(error.into()))?;
            tenet_matrixalgebra::seam::svd_vals_dyn_checked_generic(dense.dense(), &input)?
        };
        let provider = self.logical_space().provider();
        let mut decoded = raw
            .into_iter()
            .map(|entry| {
                Ok(SectorSpectrum {
                    sector: provider.try_decode_label(entry.sector)?,
                    values: entry.values,
                })
            })
            .collect::<Result<Vec<_>, <R as TypedSectorAdmission>::Error>>()
            .map_err(|error| GenericTensorError::Plan(CheckedGenericPlanError::Provider(error)))?;
        decoded.sort_by(|left, right| left.sector.cmp(&right.sector));
        Ok(decoded)
    }
}

impl<R, D> TensorMap<R, D>
where
    R: TypedSectorAdmission<
            Error = <R as CheckedGenericFusion>::Error,
            Mode = CheckedGenericAdmissionMode,
        > + CheckedGenericFusion,
    D: FactorizationScalar,
{
    /// Checked-Generic compact LQ for owned host tensors.
    pub(super) fn lq_compact_checked_generic(
        &self,
    ) -> Result<Lq<Self>, GenericTensorError<<R as CheckedGenericFusion>::Error>> {
        let TypedTensorRepr::Owned(body) = &self.repr else {
            return Err(GenericTensorError::Facade(Error::InvalidArgument(
                "checked Generic lq_compact does not accept lazy adjoints".to_string(),
            )));
        };
        if let TypedData::Diagonal(spectrum) = body.data.as_ref() {
            if let Some((l_space, q_space, phases, magnitudes)) =
                tenet_matrixalgebra::seam::lq_diagonal_dyn_checked_generic(
                    &body.space,
                    spectrum,
                    false,
                )?
            {
                return Ok(Lq {
                    l: self.with_spectrum_on(l_space, magnitudes),
                    q: self.with_spectrum_on(q_space, phases),
                });
            }
        }
        let mut dense = self.runtime.lease_dense();
        let payload = body.materialized_dense_data();
        let input = BoundDynamicTensorRef::try_new(&body.space, &payload)
            .map_err(|error| GenericTensorError::Facade(error.into()))?;
        let Lq { l, q } =
            tenet_matrixalgebra::seam::lq_compact_dyn_checked_generic(dense.dense(), &input)?;
        Ok(Lq {
            l: wrap_factor_on(&self.runtime, l),
            q: wrap_factor_on(&self.runtime, q),
        })
    }
}

impl<R, D> TensorMap<R, D>
where
    R: TypedSectorAdmission<
            Error = <R as CheckedGenericFusion>::Error,
            Mode = CheckedGenericAdmissionMode,
        > + CheckedGenericFusion,
    D: FactorizationScalar,
{
    pub(super) fn svd_full_checked_generic(
        &self,
    ) -> Result<Svd<Self>, GenericTensorError<<R as CheckedGenericFusion>::Error>> {
        let TypedTensorRepr::Owned(body) = &self.repr else {
            return Err(GenericTensorError::Facade(Error::InvalidArgument(
                "checked Generic svd_full does not accept lazy adjoints".to_string(),
            )));
        };
        let admission = if let TypedData::Diagonal(spectrum) = body.data.as_ref() {
            Some(
                tenet_matrixalgebra::seam::svd_full_diagonal_factors_dyn_checked_generic(
                    &body.space,
                    spectrum,
                )?,
            )
        } else {
            None
        };
        let factors = match admission {
            Some(tenet_matrixalgebra::seam::CheckedDiagonalFullSvdFactors::Direct(factors)) => {
                factors
            }
            admission => {
                let mut dense = self.runtime.lease_dense();
                let payload = body.materialized_dense_data();
                let input = BoundDynamicTensorRef::try_new(&body.space, &payload)
                    .map_err(|error| GenericTensorError::Facade(error.into()))?;
                match admission {
                    Some(tenet_matrixalgebra::seam::CheckedDiagonalFullSvdFactors::Fallback(
                        dimensions,
                    )) => {
                        tenet_matrixalgebra::seam::svd_full_factors_dyn_checked_generic_with_dimensions(
                            dense.dense(),
                            &input,
                            Some(dimensions),
                        )?
                    }
                    _ => tenet_matrixalgebra::seam::svd_full_factors_dyn_checked_generic(
                        dense.dense(),
                        &input,
                    )?,
                }
            }
        };
        let (u, vh, mut spectrum, row_dimensions, col_dimensions) = factors.into_parts();
        if full_svd_compact_bond(&u, &vh, &spectrum) {
            let space = tenet_matrixalgebra::seam::diagonal_bond_bound_space_generic_checked(
                Arc::clone(body.space.provider_arc()),
                &spectrum,
            )?;
            if full_svd_compact_layout(&space, &spectrum) {
                return Ok(Svd {
                    u: wrap_factor_on(&self.runtime, u),
                    s: diagonal_factor_on_bound(&self.runtime, space, &mut spectrum, D::from_real),
                    vh: wrap_factor_on(&self.runtime, vh),
                });
            }
        }
        let s = tenet_matrixalgebra::seam::rectangular_diagonal_bond_tensor_generic_checked(
            Arc::clone(body.space.provider_arc()),
            &spectrum,
            &row_dimensions,
            &col_dimensions,
            &D::from_real,
        )?;
        Ok(Svd {
            u: wrap_factor_on(&self.runtime, u),
            s: wrap_factor_on(&self.runtime, s),
            vh: wrap_factor_on(&self.runtime, vh),
        })
    }

    /// Checked-Generic compact SVD for owned host tensors.
    pub(super) fn svd_compact_checked_generic(
        &self,
    ) -> Result<Svd<Self>, GenericTensorError<<R as CheckedGenericFusion>::Error>> {
        let TypedTensorRepr::Owned(body) = &self.repr else {
            return Err(GenericTensorError::Facade(Error::InvalidArgument(
                "checked Generic svd_compact does not accept lazy adjoints".to_string(),
            )));
        };
        if let TypedData::Diagonal(spectrum) = body.data.as_ref() {
            if let Some((u, vh, mut singular_values)) =
                tenet_matrixalgebra::seam::svd_compact_diagonal_factors_dyn_checked_generic(
                    &body.space,
                    spectrum,
                )?
            {
                let s = diagonal_factor_on_checked(
                    &self.runtime,
                    Arc::clone(body.space.provider_arc()),
                    &mut singular_values,
                    D::from_real,
                )?;
                return Ok(Svd {
                    u: wrap_factor_on(&self.runtime, u),
                    s,
                    vh: wrap_factor_on(&self.runtime, vh),
                });
            }
        }
        let mut dense = self.runtime.lease_dense();
        let payload = body.materialized_dense_data();
        let input = BoundDynamicTensorRef::try_new(&body.space, &payload)
            .map_err(|error| GenericTensorError::Facade(error.into()))?;
        let (u, vh, mut singular_values) =
            tenet_matrixalgebra::seam::svd_compact_factors_with_spectrum_dyn_checked_generic(
                dense.dense(),
                &input,
            )?;
        let s = diagonal_factor_on_checked(
            &self.runtime,
            Arc::clone(input.space().provider_arc()),
            &mut singular_values,
            D::from_real,
        )?;
        Ok(Svd {
            u: wrap_factor_on(&self.runtime, u),
            s,
            vh: wrap_factor_on(&self.runtime, vh),
        })
    }
}

impl<R, D> TensorMap<R, D>
where
    R: TypedSectorAdmission<
            Error = <R as CheckedGenericFusion>::Error,
            Mode = CheckedGenericAdmissionMode,
        > + CheckedGenericFusion,
    D: FactorizationScalar,
{
    /// Checked-Generic compact QR for owned host tensors.
    ///
    /// The first Generic decomposition leaf deliberately rejects lazy-adjoint
    /// inputs; no operation-local whole-payload fallback is introduced here.
    pub(super) fn qr_compact_checked_generic(
        &self,
    ) -> Result<Qr<Self>, GenericTensorError<<R as CheckedGenericFusion>::Error>> {
        let TypedTensorRepr::Owned(body) = &self.repr else {
            return Err(GenericTensorError::Facade(Error::InvalidArgument(
                "checked Generic qr_compact does not accept lazy adjoints".to_string(),
            )));
        };
        if let TypedData::Diagonal(spectrum) = body.data.as_ref() {
            if let Some((q_space, r_space, phases, magnitudes)) =
                tenet_matrixalgebra::seam::qr_diagonal_dyn_checked_generic(
                    &body.space,
                    spectrum,
                    false,
                )?
            {
                return Ok(Qr {
                    q: self.with_spectrum_on(q_space, phases),
                    r: self.with_spectrum_on(r_space, magnitudes),
                });
            }
        }
        let mut dense = self.runtime.lease_dense();
        let payload = body.materialized_dense_data();
        let input = BoundDynamicTensorRef::try_new(&body.space, &payload)
            .map_err(|error| GenericTensorError::Facade(error.into()))?;
        let Qr { q, r } =
            tenet_matrixalgebra::seam::qr_compact_dyn_checked_generic(dense.dense(), &input)?;
        Ok(Qr {
            q: wrap_factor_on(&self.runtime, q),
            r: wrap_factor_on(&self.runtime, r),
        })
    }
}

impl<R, D> TensorMap<R, D>
where
    R: TypedSectorAdmission,
    R::Mode: TypedTensorQrDispatch<R, D> + TypedTensorTransformDispatch<R, D>,
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
    /// An admitted finite owned Host compact diagonal on `V <- V` returns
    /// both factors in compact storage with `W = V`, including dual
    /// orientation, for multiplicity-free and checked-Generic providers alike
    /// (TensorKit's diagonal dispatch). Any other input, including a
    /// materialized diagonal or swapped leg roles, takes the dense route,
    /// whose `W` is a fresh nondual bond (TensorKit `fuse`); a dual `V` thus
    /// yields `W = V` or its nondual flip depending on storage.
    /// Phase is +1 at zero; work and output storage are `O(sum_c k_c)` after
    /// sector/layout validation. Use [`Self::diagview`] to read the factors,
    /// or [`Self::materialize`] before [`Self::dense_data`] for a dense buffer.
    /// Nonfinite or unrepresentable magnitudes retain the dense provider route.
    ///
    /// Other inputs run one dense QR per sector, with cost
    /// `O(sum_c m_c * n_c * min(m_c, n_c))`. Multiplicity-free lazy adjoints
    /// are materialized only for the operation; checked-Generic QR requires
    /// an owned input. Checked factors use the same provider instance as `self`.
    /// If any sector fails or the provider rejects an output space, no factors
    /// are returned.
    ///
    /// # Errors
    ///
    /// Dense execution returns [`Error::Operation`], while factor-layout
    /// failures return [`Error::Core`] where applicable. If a checked provider
    /// rejects an output space, its original error is available as the source.
    /// Passing a lazy adjoint there returns [`Error::InvalidArgument`].
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
        self.with_leg_roles(
            rows,
            cols,
            <R::Mode as TypedTensorQrDispatch<R, D>>::qr_compact,
        )
    }
}

impl<R, D> TensorMap<R, D>
where
    R: TypedSectorAdmission,
    R::Mode: TypedTensorSvdDispatch<R, D> + TypedTensorTransformDispatch<R, D>,
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
    /// a generally rectangular `s`; its storage depends on the admitted bond.
    ///
    /// On Host, compact `s` stores only `sum_c k_c` diagonal values for both
    /// multiplicity-free and checked-Generic providers. `s.materialize()`
    /// produces dense storage when needed.
    /// All checked factors retain the source's exact provider `Arc`.
    ///
    /// Dense inputs cost `O(sum_c m_c * n_c * min(m_c, n_c))`. An owned
    /// multiplicity-free compact diagonal with representable magnitudes is
    /// sorted directly by sector, without
    /// a dense input or dense SVD call; its dense `u` and `vh` still require
    /// `O(sum_c k_c²)` output storage and writes. A multiplicity-free lazy
    /// adjoint is handled from its parent without materializing it.
    /// Checked-Generic SVD requires owned input. An admitted square compact
    /// diagonal uses the same direct per-sector sorting and factor publication;
    /// other layouts and nonfinite or unrepresentable spectra use the ordinary
    /// dense solver path.
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
        self.with_leg_roles(
            rows,
            cols,
            <R::Mode as TypedTensorSvdDispatch<R, D>>::svd_compact,
        )
    }

    /// Returns the full SVD `self = u * s * vh` with square outer factors.
    ///
    /// In sector `c`, `u` is `m_c x m_c`, `vh` is `n_c x n_c`, and `s` is
    /// the rectangular `m_c x n_c` diagonal matrix. The spaces are
    /// `u : codomain <- W_out`, `s : W_out <- W_in`, and
    /// `vh : W_in <- domain`. It accepts the same inputs as
    /// [`Self::svd_compact`], but its square outer factors can require more
    /// dense storage. On Host, `s` uses compact diagonal storage exactly when
    /// the constructed `W_out` and `W_in` legs coincide, every positive bond
    /// sector has a complete spectrum, and the compact layout is admitted.
    /// Otherwise it is dense. An admitted owned compact input avoids dense
    /// input materialization and a solver call. For a checked-Generic provider,
    /// this requires a rank-(1,1) square aligned source with one tree per side
    /// and complete checked row and column bond maps exactly equal to the
    /// source spectrum sectors and dimensions; other checked inputs use the
    /// dense route.
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
        self.with_leg_roles(
            rows,
            cols,
            <R::Mode as TypedTensorSvdDispatch<R, D>>::svd_full,
        )
    }
}

impl<R, D> TensorMap<R, D>
where
    R: TypedSectorAdmission,
    R::Mode: TypedTensorLqDispatch<R, D> + TypedTensorTransformDispatch<R, D>,
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
    /// admitted owned Host compact diagonals preserve `W = V` and both factors
    /// are compact, including on a dual `V`. Checked Generic requires an owned
    /// input.
    /// Checked factors use the source provider instance, and a failure returns
    /// no factors. A multiplicity-free lazy adjoint runs QR on its owned
    /// parent and returns detached owned factors without materializing the
    /// receiver.
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
        self.with_leg_roles(
            rows,
            cols,
            <R::Mode as TypedTensorLqDispatch<R, D>>::lq_compact,
        )
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
    R::Mode: TypedTensorFullQrDispatch<R, D> + TypedTensorTransformDispatch<R, D>,
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
    /// publication are additional costs. An admitted owned Host compact diagonal
    /// uses `W = V` and two compact factors under the boundary documented by
    /// [`Self::qr_compact`].
    /// See [`Self::qr_compact`] for the
    /// compact alternative, storage and lazy-input behavior, errors, and example.
    ///
    /// `rows` and `cols` are the leg roles: the operation acts on the matrix
    /// view `self.permute(rows, cols)`, and the current split costs nothing
    /// extra (see [`Self::svd_compact`]'s *Leg roles*).
    pub fn qr_full(&self, rows: &[usize], cols: &[usize]) -> Result<Qr<Self>, TypedFacadeError<R>> {
        self.with_leg_roles(
            rows,
            cols,
            <R::Mode as TypedTensorFullQrDispatch<R, D>>::qr_full,
        )
    }
}

impl<R, D> TensorMap<R, D>
where
    R: TypedSectorAdmission,
    R::Mode: TypedTensorFullLqDispatch<R, D> + TypedTensorTransformDispatch<R, D>,
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
    /// adjoint, and owned factor publication are additional costs. An admitted
    /// owned Host compact diagonal uses `W = V` and two compact factors under
    /// the boundary documented by [`Self::lq_compact`]. See [`Self::lq_compact`] for the compact alternative, storage and
    /// lazy-input behavior, errors, and example.
    ///
    /// `rows` and `cols` are the leg roles: the operation acts on the matrix
    /// view `self.permute(rows, cols)`, and the current split costs nothing
    /// extra (see [`Self::svd_compact`]'s *Leg roles*).
    pub fn lq_full(&self, rows: &[usize], cols: &[usize]) -> Result<Lq<Self>, TypedFacadeError<R>> {
        self.with_leg_roles(
            rows,
            cols,
            <R::Mode as TypedTensorFullLqDispatch<R, D>>::lq_full,
        )
    }
}

impl<R, D> TensorMap<R, D>
where
    R: TypedSectorAdmission,
    R::Mode: TypedTensorSvdValsDispatch<R, D> + TypedTensorTransformDispatch<R, D>,
    D: FactorizationScalar,
{
    /// Returns only the singular values, grouped by provider-labelled coupled
    /// sector and descending within each sector.
    ///
    /// No factor tensor or intermediate bond is built. This is the least
    /// allocating member of the SVD family when only the spectrum is needed.
    /// Multiplicity-free lazy adjoints are read through their owned parent,
    /// and an owned compact diagonal input with finite, representable magnitudes
    /// is read and sorted directly without a dense solver for both
    /// multiplicity-free and checked-Generic providers. Other compact cases
    /// retain the dense solver's behavior. Checked Generic requires an owned
    /// input and returns [`Error::InvalidArgument`] for a lazy adjoint. A dense failure returns
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
        self.with_leg_roles(
            rows,
            cols,
            <R::Mode as TypedTensorSvdValsDispatch<R, D>>::svd_vals,
        )
    }
}

impl<R, D> TensorMap<R, D>
where
    R: TypedSectorAdmission,
    R::Mode: TypedTensorEighValsDispatch<R, D> + TypedTensorTransformDispatch<R, D>,
    D: FactorizationScalar,
{
    /// Returns only the real Hermitian eigenvalues, grouped by
    /// provider-labelled coupled sector and descending by absolute value.
    ///
    /// No eigenvector factor or bond space is built. The input must be an
    /// endomorphism and every sector must satisfy the same Hermiticity check as
    /// [`Self::eigh_full`]. An owned Host compact diagonal with finite, exactly
    /// real entries and an admitted endomorphism sector layout is read directly
    /// for both multiplicity-free and checked-Generic providers. The checked
    /// route also requires the canonical bond space and a bijection between
    /// stored spectra and aligned square sector regions. Other compact inputs
    /// use the dense route; checked Generic requires owned input and rejects
    /// lazy adjoints for this values-only method. Dense failures return
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
    ) -> Result<Vec<SectorSpectrum<<R as TypedSectorAdmission>::Sector, f64>>, TypedFacadeError<R>>
    {
        self.with_leg_roles(
            rows,
            cols,
            <R::Mode as TypedTensorEighValsDispatch<R, D>>::eigh_vals,
        )
    }
}

impl<R, D> TensorMap<R, D>
where
    R: TypedSectorAdmission,
    R::Mode: TypedTensorEighDispatch<R, D> + TypedTensorTransformDispatch<R, D>,
    D: FactorizationScalar,
{
    /// Returns the Hermitian eigendecomposition `self = v * d * v^H` as an
    /// [`Eigh`], which states the spectrum order and the storage routes of `d`.
    ///
    /// The input must be an endomorphism. In each coupled sector,
    /// `v : codomain(self) <- W` is unitary and `d : W <- W` holds the signed
    /// real eigenvalues. Eigenvalues are stable-sorted by
    /// descending absolute value; each eigenvector's phase is fixed by making
    /// its largest-magnitude component real and non-negative. Bases inside an
    /// exactly degenerate eigenspace remain backend-dependent.
    ///
    /// Both multiplicity-free and checked-Generic `d` factors use compact
    /// diagonal storage. Checked factors retain the exact source provider
    /// `Arc`. An owned Host compact diagonal with finite, exactly real entries
    /// and one aligned square region per coupled sector is read directly into
    /// a permutation eigenbasis, for both multiplicity-free and checked-Generic
    /// providers; its dense output still occupies `Σ_c k_c²` elements. Other
    /// compact inputs and lazy adjoints use an operation-local dense payload.
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
    /// use tenet::typed::{Eigh, GradedSpace, Runtime, TensorMap};
    ///
    /// let runtime = Runtime::builder().build()?;
    /// let v = GradedSpace::try_new(Arc::new(U1FusionRule), [(U1Irrep::new(0), 2)])?;
    /// let a: TensorMap<_, f64> = TensorMap::isomorphism(&runtime, [&v], [&v])?.scale(2.0);
    /// let Eigh { d, v: eigenvectors } = a.eigh_full(&[0], &[1])?;
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
    ) -> Result<Eigh<Self>, TypedFacadeError<R>> {
        self.with_leg_roles(
            rows,
            cols,
            <R::Mode as TypedTensorEighDispatch<R, D>>::eigh_full,
        )
    }
}

impl<R, D> TensorMap<R, D>
where
    R: TypedSectorAdmission,
    R::Mode: TypedTensorEigValsDispatch<R, D> + TypedTensorTransformDispatch<R, D>,
    D: AdvancedLinalgScalar,
{
    /// Returns only the general eigenvalues as `Complex64` (at every payload
    /// dtype, like every spectrum of this crate), grouped by
    /// provider-labelled sector and descending by magnitude.
    ///
    /// No eigenvector factor or bond space is built. The input must be an
    /// endomorphism. An owned Host compact diagonal with finite eigenvalue
    /// magnitudes and an admitted endomorphism sector layout is read directly
    /// for both multiplicity-free and checked-Generic providers. The checked
    /// route also requires the canonical bond space and a bijection between
    /// stored spectra and aligned square sector regions. For multiplicity-free
    /// providers, other compact inputs and lazy adjoints use an operation-local
    /// dense payload. Checked Generic uses that dense route only for owned
    /// inputs and rejects lazy adjoints for this values-only method. Unlike
    /// [`Self::eig_full`], no eigenvector-rank gate is needed because no
    /// eigenbasis is returned.
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
        self.with_leg_roles(
            rows,
            cols,
            <R::Mode as TypedTensorEigValsDispatch<R, D>>::eig_vals,
        )
    }
}

impl<R, D> TensorMap<R, D>
where
    R: TypedSectorAdmission,
    R::Mode: TypedTensorEigDispatch<R, D> + TypedTensorTransformDispatch<R, D>,
    D: AdvancedLinalgScalar,
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
    /// Checked Generic retains the exact source provider `Arc` and rejects an
    /// eigenvector matrix that is not numerically full rank under
    /// `n * epsilon * sigma_max`. This is an operational gate on the computed
    /// matrix, not a universal detector for every defective floating-point
    /// input. The multiplicity-free path forwards the dense backend result
    /// without this additional rank gate. An admitted owned Host compact
    /// diagonal with finite values of finite norm and one aligned square
    /// region per coupled sector reads its eigenvalues directly and builds a
    /// dense permutation factor, for both multiplicity-free and checked-Generic
    /// providers; the rank gate is not evaluated there because a permutation
    /// has unit singular values. Lazy adjoints and other inputs use an
    /// operation-local dense payload.
    ///
    /// A non-endomorphism, invalid/non-finite dense result, checked rank-gate
    /// failure, factor-layout failure, or provider failure returns no factors.
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
        self.with_leg_roles(
            rows,
            cols,
            <R::Mode as TypedTensorEigDispatch<R, D>>::eig_full,
        )
    }
}
