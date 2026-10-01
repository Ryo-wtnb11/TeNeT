#[allow(unused_imports)]
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

/// One tensor-local restriction used by the internal network slice executor.
#[doc(hidden)]
pub struct NetworkDegeneracyRestriction {
    pub effective_axis: usize,
    pub authority_sector: SectorId,
    pub range: std::ops::Range<usize>,
    pub partner: bool,
}

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

pub(super) fn host_add_impl<R, D>(
    tensor: &TensorMap<R, D>,
    other: &TensorMap<R, D>,
    alpha: D,
    beta: D,
) -> Result<TensorMap<R, D>, Error>
where
    R: TypedSectorAdmission,
    D: TensorScalar,
{
    if !tensor.runtime.same_runtime(&other.runtime) {
        return Err(Error::RuntimeMismatch);
    }
    if tensor.logical_space().space() != other.logical_space().space() {
        return Err(Error::InvalidArgument(
            "tensors live on different spaces or block layouts".to_string(),
        ));
    }
    let _host_pool = tensor.runtime.enter_host_pool();
    if matches!(&tensor.repr, TypedTensorRepr::Adjoint(_))
        || matches!(&other.repr, TypedTensorRepr::Adjoint(_))
    {
        let (lhs, lhs_data) = tensor.fusion_operand_and_data();
        let (rhs, rhs_data) = other.fusion_operand_and_data();
        let data = tenet_tensors::oriented_fusion_add_owned(
            tensor.logical_space().space().structure(),
            lhs,
            &lhs_data,
            rhs,
            &rhs_data,
            alpha,
            beta,
        )?;
        return Ok(tensor.with_data(data));
    }
    match (tensor.spectrum(), other.spectrum()) {
        (Some(lhs), Some(rhs)) => {
            if lhs.len() != rhs.len() {
                return Err(spectra_disagree());
            }
            let sum = lhs
                .iter()
                .zip(rhs)
                .map(|(left, right)| {
                    if left.sector != right.sector || left.values.len() != right.values.len() {
                        return Err(spectra_disagree());
                    }
                    Ok(tenet_matrixalgebra::SectorSpectrum {
                        sector: left.sector,
                        values: left
                            .values
                            .iter()
                            .zip(&right.values)
                            .map(|(&x, &y)| scale_value(x, alpha) + scale_value(y, beta))
                            .collect(),
                    })
                })
                .collect::<Result<Vec<_>, Error>>()?;
            return Ok(tensor.with_spectrum(sum));
        }
        (Some(diagonal), None) => {
            return Ok(tensor.with_data(scatter_spectrum(
                tensor.logical_space().space(),
                other
                    .owned_body()
                    .expect("owned add input")
                    .materialized_dense_data()
                    .as_ref(),
                beta,
                diagonal,
                alpha,
            )?))
        }
        (None, Some(diagonal)) => {
            return Ok(tensor.with_data(scatter_spectrum(
                tensor.logical_space().space(),
                tensor
                    .owned_body()
                    .expect("owned add input")
                    .materialized_dense_data()
                    .as_ref(),
                alpha,
                diagonal,
                beta,
            )?))
        }
        (None, None) => {}
    }
    Ok(tensor.with_data(
        tensor
            .owned_body()
            .expect("owned add input")
            .materialized_dense_data()
            .as_ref()
            .iter()
            .zip(
                other
                    .owned_body()
                    .expect("owned add input")
                    .materialized_dense_data()
                    .as_ref(),
            )
            .map(|(&x, &y)| scale_value(x, alpha) + scale_value(y, beta))
            .collect(),
    ))
}

pub(super) fn host_scale_impl<R, D>(tensor: &TensorMap<R, D>, factor: D) -> TensorMap<R, D>
where
    R: TypedSectorAdmission,
    R::Mode: TypedTensorAdjointDispatch<R, D>,
    D: TensorScalar,
{
    if let Some(spectrum) = tensor.spectrum() {
        return tensor.with_spectrum(
            spectrum
                .iter()
                .map(|entry| tenet_matrixalgebra::SectorSpectrum {
                    sector: entry.sector,
                    values: entry
                        .values
                        .iter()
                        .map(|&value| scale_value(value, factor))
                        .collect(),
                })
                .collect(),
        );
    }
    if let TypedTensorRepr::Adjoint(view) = &tensor.repr {
        let parent = TensorMap {
            runtime: tensor.runtime.clone(),
            repr: TypedTensorRepr::Owned(Arc::clone(&view.parent)),
        };
        return host_scale_impl(&parent, FactorScalar::adjoint(factor))
            .adjoint()
            .expect("scaling a pre-admitted adjoint must preserve its layout");
    }
    tensor.with_data(
        tensor
            .owned_body()
            .expect("owned scale input")
            .materialized_dense_data()
            .as_ref()
            .iter()
            .map(|&value| scale_value(value, factor))
            .collect(),
    )
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
    /// Dense input uses one linear solve per coupled sector. A
    /// multiplicity-free compact diagonal divisor instead applies its
    /// elementwise reciprocal and bond scaling; checked Generic materializes
    /// compact inputs for the dense route, and a compact right-hand side of a
    /// dense divisor is densified into the solve buffer. Lazy adjoints are
    /// materialized only for this call.
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
    /// - [`Error::InvalidArgument`] for a non-finite singular value (compact
    ///   entry magnitude), rejected rather than cut, on every storage arm.
    /// - [`Error::Operation`] / [`Error::Core`] from dense SVD or recomposition.
    ///
    /// There is no singular-input failure: sending the offending directions to
    /// zero is what a pseudo-inverse is for.
    ///
    /// # Complexity and storage
    ///
    /// Dense input uses one compact SVD per nonempty coupled sector,
    /// `O(Σ_c n_c³)`, then folds `S⁺` into a column scaling and recomposes with
    /// one local GEMM. Multiplicity-free compact input uses an **O(rank)
    /// elementwise cutoff-and-reciprocal arm** over the stored diagonal values
    /// and stays compact.
    ///
    /// Checked Generic admits the swapped output with the source's exact
    /// provider `Arc` and validates its identity, HomSpace, rank, and layout
    /// before any SVD/GEMM. Standalone compact construction is supported, but
    /// checked `pinv` has no elementwise compact arm: it materializes that input
    /// for the dense path and publishes a dense result. A checked lazy adjoint
    /// is likewise materialized operation-locally.
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
    R: TypedSectorAdmission,
    R::Mode: TypedTensorReductionDispatch<R, D>,
    D: TensorScalar,
{
    /// TensorKit `norm(t, p)`: the entrywise `p`-norm of the reduced blocks,
    ///
    /// ```text
    /// p == 2       -> sqrt(sum_c dim(c) * sum_ij |self_c[i,j]|^2)
    /// p == Inf     -> max_c max_ij |self_c[i,j]|          (not dim-weighted)
    /// finite p > 0 -> (sum_c dim(c) * sum_ij |self_c[i,j]|^p)^(1/p)
    /// ```
    ///
    /// `p == 2.0` is the quantum-dimension-weighted Frobenius norm and is the
    /// only exponent every storage supports; `p` is an `f64` exactly as in
    /// TensorKit, so each norm has one spelling. The entrywise norm is never
    /// an operator norm, matrices included.
    ///
    /// For `p == 2`, abelian providers have `dim(c) = 1`, giving the ordinary Frobenius
    /// norm. Multiplicity-free compact diagonal input is reduced directly in
    /// `O(sum_c k_c)`; dense input is one pass over the payload. Lazy adjoints
    /// read their parent orientation without materializing.
    /// Checked-Generic reductions currently require dense payloads.
    ///
    /// The Host norm does not overflow or underflow while the norm itself is
    /// representable: when the unscaled sum of squares leaves the `f64` range,
    /// two more passes rescale every entry by the largest magnitude, as Julia's
    /// `LinearAlgebra.generic_norm2` does. An entry of NaN magnitude `|x|` gives
    /// NaN; otherwise an infinite magnitude gives `inf`.
    ///
    /// `p == Inf` follows Julia's NaN-propagating `max`: a payload holding
    /// any NaN, including a complex entry whose real or imaginary part alone
    /// is NaN, returns NaN; an infinite entry returns `+inf`; a tensor with no
    /// stored entries returns `+0.0`. Every exponent is one pass over the
    /// payload, `O(sum_c k_c)` on compact diagonal storage.
    ///
    /// # Errors
    ///
    /// - [`Error::InvalidArgument`] when `p` is NaN, zero, negative, or
    ///   `-inf`; TensorKit throws `ArgumentError` over the same domain.
    /// - [`Error::InvalidArgument`] for `p != 2` on a checked-Generic
    ///   provider, which has only the Frobenius reduction.
    /// - If a checked provider cannot supply a quantum dimension, its original
    ///   error is available as the source. An invalid coupled-sector layout
    ///   returns [`Error::Core`].
    ///
    /// ```
    /// use std::sync::Arc;
    /// use tenet::sector::{U1FusionRule, U1Irrep};
    /// use tenet::typed::{GradedSpace, Runtime, TensorMap};
    ///
    /// let runtime = Runtime::builder().build()?;
    /// let v = GradedSpace::try_new(Arc::new(U1FusionRule), [(U1Irrep::new(0), 2)])?;
    /// let id: TensorMap<_, f64> = TensorMap::isomorphism(&runtime, [&v], [&v])?;
    /// let twice = id.axpby(1.0, &id, 1.0)?;
    /// assert!((twice.norm(2.0)? - 2.0_f64.sqrt() * 2.0).abs() < 1e-12);
    /// assert_eq!(twice.norm(f64::INFINITY)?, 2.0);
    /// assert_eq!(id.inner(&id)?, 2.0);
    /// assert_eq!(id.tr()?, 2.0);
    /// # Ok::<(), tenet::typed::Error>(())
    /// ```
    pub fn norm(&self, p: f64) -> Result<f64, TypedFacadeError<R>> {
        <R::Mode as TypedTensorReductionDispatch<R, D>>::norm(self, p)
    }
    /// Returns the quantum-dimension-weighted Frobenius inner product
    /// `sum_c dim(c) * sum_ij conj(self_c[i,j]) * other_c[i,j]`.
    ///
    /// The product is conjugate-linear in `self`, and `self.inner(&self)` is
    /// `self.norm(2.0)^2` up to floating-point error. Both tensors must share the
    /// same runtime, hom space, and block layout. A multiplicity-free compact
    /// diagonal operand is reduced directly from its stored spectrum in
    /// `O(sum_c k_c)` payload reads, including against a dense lazy adjoint;
    /// it is never densified. Its off-diagonal entries are structural zeros,
    /// so matching dense off-diagonal values are not read even when they are
    /// `NaN` or infinite. This deliberately differs from TensorKit 0.17's
    /// current generic mixed-block reduction, which visits those stored dense
    /// positions and therefore propagates their non-finite values.
    /// Checked-Generic reductions currently require dense payloads. See
    /// [`Self::norm`] for the weighting, lazy behavior, and example.
    #[doc(alias = "dot")]
    pub fn inner<'a>(
        &self,
        other: impl Into<TensorRef<'a, R, D>>,
    ) -> Result<D, TypedFacadeError<R>> {
        let other = other.into().operand()?;
        let other = &*other;
        <R::Mode as TypedTensorReductionDispatch<R, D>>::inner(self, other)
    }
    /// Returns the quantum-dimension-weighted block trace
    /// `sum_c dim(c) * Tr(self_c)` of an endomorphism.
    ///
    /// The codomain and domain legs must be exactly equal. This is TensorKit's
    /// `tr`; it is not a positivity claim, and the result may be negative or
    /// complex. It also is not the fermionic supertrace: no twist factor is
    /// applied. Use [`Self::trace_pairs`] for the categorical contraction
    /// trace, which includes the provider's pivotal/twist data.
    ///
    /// Multiplicity-free compact diagonal input is summed directly in
    /// `O(sum_c k_c)`. Checked-Generic reductions require dense payloads. A
    /// lazy adjoint returns the conjugate of its parent's trace without
    /// materializing. See [`Self::norm`] for a runnable example.
    pub fn tr(&self) -> Result<D, TypedFacadeError<R>> {
        <R::Mode as TypedTensorReductionDispatch<R, D>>::tr(self)
    }
}

impl<R, D> TensorMap<R, D>
where
    R: TypedSectorAdmission,
    R::Mode: TypedTensorAddScaleDispatch<R, D>,
    D: TensorScalar,
{
    /// Returns the host-side linear combination
    /// `x.axpby(alpha, &y, beta) == alpha * x + beta * y` on the operands'
    /// common tensor space, where `x` is the receiver.
    ///
    /// The name is BLAS `axpby`: each coefficient sits next to the operand it
    /// multiplies. TeNeT has no `add` with coefficients because
    /// VectorInterface's `add(y, x, α, β)` computes `β·y + α·x`, binding the
    /// coefficients the other way round; a ported `add` call would compile and
    /// silently swap them.
    ///
    /// Both operands must share a runtime and exactly the same tensor space and
    /// block layout. Two compact diagonal inputs stay compact. A mixed
    /// compact/dense pair allocates only the dense result, and lazy inputs are
    /// read in their logical orientation without filling their caches.
    ///
    /// Returns [`Error::RuntimeMismatch`] or [`Error::InvalidArgument`] before
    /// producing a result. See [`Self::norm`] for a runnable example.
    ///
    /// ```compile_fail
    /// use tenet::sector::U1FusionRule;
    /// use tenet::typed::TensorMap;
    ///
    /// fn removed(x: &TensorMap<U1FusionRule, f64>) {
    ///     let _ = x.add(x, 1.0, 2.0);
    /// }
    /// ```
    pub fn axpby<'a>(
        &self,
        alpha: D,
        y: impl Into<TensorRef<'a, R, D>>,
        beta: D,
    ) -> Result<Self, TypedFacadeError<R>> {
        let y = y.into().operand()?;
        let y = &*y;
        <R::Mode as TypedTensorAddScaleDispatch<R, D>>::axpby(self, alpha, y, beta)
    }

    /// Returns `factor * self` in host storage.
    ///
    /// The operation is infallible because `factor` already has the payload
    /// type `D`. Compact diagonal storage stays compact. A dense lazy adjoint
    /// remains lazy by scaling its parent with the conjugated factor, so no
    /// receiver materialization is cached.
    pub fn scale(&self, factor: D) -> Self {
        <R::Mode as TypedTensorAddScaleDispatch<R, D>>::scale(self, factor)
    }

    /// `self = factor * self` in place, VectorInterface `scale!(t, α)`, on
    /// the receiver's own storage: a dense payload element by element, a
    /// compact diagonal on its stored values (it stays compact, TensorKit's
    /// `scale!` over a `DiagonalTensorMap`'s blocks), and a lazy adjoint on
    /// its parent with the conjugated factor (it stays lazy, as
    /// [`Self::scale`] does). Nothing is allocated.
    ///
    /// # Errors
    ///
    /// [`Error::DestinationShared`] when the receiver shares its storage
    /// with another handle (a shallow `Clone`, or the parent of a lazy
    /// adjoint): writing it would change that handle too, and replacing it
    /// would silently allocate, so the call does neither and leaves the
    /// receiver unchanged. The scaling itself cannot fail.
    ///
    /// ```
    /// use std::sync::Arc;
    ///
    /// use tenet::sector::{U1FusionRule, U1Irrep};
    /// use tenet::typed::{Error, GradedSpace, Runtime, TensorMap};
    ///
    /// let runtime = Runtime::builder().build()?;
    /// let v = GradedSpace::try_new(Arc::new(U1FusionRule), [(U1Irrep::new(0), 2)])?;
    /// let mut t: TensorMap<_, f64> = TensorMap::rand_with_seed(&runtime, [&v], [&v], 1)?;
    /// let expected = t.scale(2.0);
    /// t.scale_assign(2.0)?;
    /// assert_eq!(t.dense_data()?, expected.dense_data()?);
    ///
    /// let shared = t.clone();
    /// assert_eq!(t.scale_assign(2.0), Err(Error::DestinationShared));
    /// drop(shared);
    /// # Ok::<(), tenet::typed::Error>(())
    /// ```
    pub fn scale_assign(&mut self, factor: D) -> Result<(), TypedFacadeError<R>> {
        let (body, factor) = match &mut self.repr {
            TypedTensorRepr::Owned(body) => (Arc::get_mut(body), factor),
            TypedTensorRepr::Adjoint(view) => (
                Arc::get_mut(view).and_then(|view| Arc::get_mut(&mut view.parent)),
                FactorScalar::adjoint(factor),
            ),
        };
        let data = body
            .and_then(|body| Arc::get_mut(&mut body.data))
            .ok_or(Error::DestinationShared)?;
        let scale = |value: &mut D| *value = scale_value(*value, factor);
        match data {
            TypedData::Dense(data) => data.iter_mut().for_each(scale),
            TypedData::Diagonal(spectrum) => spectrum
                .iter_mut()
                .flat_map(|entry| entry.values.iter_mut())
                .for_each(scale),
        }
        Ok(())
    }

    /// `destination = alpha * self + beta * destination`: BLAS `axpby`,
    /// TensorKit `add!(ty, tx, α, β)`, in place on the destination's dense
    /// host payload.
    ///
    /// Each element is `scale(destination, beta) + scale(self, alpha)` with
    /// VectorInterface's strong zero, in one pass: `beta == 0` does not let a
    /// NaN in `destination` through, and `alpha == 0` does not read `self`.
    /// A lazy-adjoint `self` is read in its logical orientation without
    /// filling its cache; a compact diagonal `self` is added onto the
    /// diagonal after `destination` is scaled.
    ///
    /// # Errors
    ///
    /// [`Error::RuntimeMismatch`]; [`Error::InvalidArgument`] when the two
    /// tensors live on different spaces or block layouts, when `destination`
    /// is not owned dense host storage, or when it aliases `self`;
    /// [`Error::DestinationShared`] when `destination` shares its storage with
    /// a clone.
    ///
    /// # Failure
    ///
    /// A validation or capability error leaves `destination` bit-identical:
    /// every check completes before the first write. On failure after
    /// validation (a backend or other execution error) the destination's
    /// contents are unspecified, while its space and block structure stay
    /// intact.
    ///
    /// ```
    /// use std::sync::Arc;
    ///
    /// use tenet::sector::{U1FusionRule, U1Irrep};
    /// use tenet::typed::{Error, GradedSpace, Runtime, TensorMap};
    ///
    /// let runtime = Runtime::builder().build()?;
    /// let v = GradedSpace::try_new(Arc::new(U1FusionRule), [(U1Irrep::new(0), 2)])?;
    /// let x: TensorMap<_, f64> = TensorMap::rand_with_seed(&runtime, [&v], [&v], 1)?;
    /// let mut y = x.zeros_like();
    /// x.axpby_into(&mut y, 2.0, 0.0)?;
    /// assert_eq!(y.dense_data()?, x.scale(2.0).dense_data()?);
    ///
    /// // A clone shares the payload, so it is not a valid destination.
    /// let shared = y.clone();
    /// assert_eq!(x.axpby_into(&mut y, 1.0, 1.0), Err(Error::DestinationShared));
    /// drop(shared);
    /// # Ok::<(), tenet::typed::Error>(())
    /// ```
    pub fn axpby_into(
        &self,
        destination: &mut Self,
        alpha: D,
        beta: D,
    ) -> Result<(), TypedFacadeError<R>> {
        host_axpby_into(self, destination, alpha, beta).map_err(TypedFacadeError::<R>::from)
    }
}

/// The host `axpby_into` body, shared with the empty-pair `trace_pairs_into`.
pub(super) fn host_axpby_into<R, D>(
    x: &TensorMap<R, D>,
    destination: &mut TensorMap<R, D>,
    alpha: D,
    beta: D,
) -> Result<(), Error>
where
    R: TypedSectorAdmission,
    D: TensorScalar,
{
    if !x.runtime.same_runtime(&destination.runtime) {
        return Err(Error::RuntimeMismatch);
    }
    if x.logical_space().space() != destination.logical_space().space() {
        return Err(Error::InvalidArgument(
            "tensors live on different spaces or block layouts".to_string(),
        ));
    }
    let destination_body = match &destination.repr {
        TypedTensorRepr::Owned(body) if matches!(body.data.as_ref(), TypedData::Dense(_)) => body,
        _ => {
            return Err(Error::InvalidArgument(
                "destination must use ordinary dense host storage".to_string(),
            ))
        }
    };
    if Arc::ptr_eq(&destination_body.data, &x.storage_body().data) {
        return Err(Error::InvalidArgument(
            "destination storage must not alias an input".to_string(),
        ));
    }
    if Arc::strong_count(destination_body) != 1 || Arc::strong_count(&destination_body.data) != 1 {
        return Err(Error::DestinationShared);
    }
    let _host_pool = x.runtime.enter_host_pool();
    let TypedTensorRepr::Owned(destination_body) = &mut destination.repr else {
        return Err(internal_layout_error("ordinary destination checked above"));
    };
    let destination_data = Arc::get_mut(destination_body)
        .and_then(|body| Arc::get_mut(&mut body.data))
        .ok_or_else(|| internal_layout_error("unique destination checked above"))?;
    let TypedData::Dense(destination_data) = destination_data else {
        return Err(internal_layout_error("dense destination checked above"));
    };
    match &x.repr {
        TypedTensorRepr::Owned(body) => match body.data.as_ref() {
            TypedData::Dense(source) => {
                if source.len() != destination_data.len() {
                    return Err(Error::InvalidArgument(
                        "tensors have different dense payload lengths".to_string(),
                    ));
                }
                for (value, &source) in destination_data.iter_mut().zip(source) {
                    *value = scale_value(*value, beta) + scale_value(source, alpha);
                }
            }
            TypedData::Diagonal(spectrum) => {
                for value in destination_data.iter_mut() {
                    *value = scale_value(*value, beta);
                }
                add_spectrum_into(x.logical_space().space(), destination_data, spectrum, alpha)?;
            }
        },
        TypedTensorRepr::Adjoint(_) => {
            let (operand, source) = x.fusion_operand_and_data();
            tenet_tensors::oriented_fusion_axpby_into(
                x.logical_space().space().structure(),
                destination_data,
                operand,
                &source,
                alpha,
                beta,
            )?;
        }
    }
    Ok(())
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
        let mut dense = self.runtime.lease_dense();
        let payload = body.materialized_dense_data();
        let input = BoundDynamicTensorRef::try_new(&body.space, &payload)
            .map_err(|error| GenericTensorError::Facade(error.into()))?;
        let raw = tenet_matrixalgebra::eig_vals_dyn_checked_generic(dense.dense(), &input)?;
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
        let mut dense = self.runtime.lease_dense();
        let payload = body.materialized_dense_data();
        let input = BoundDynamicTensorRef::try_new(&body.space, &payload)
            .map_err(|error| GenericTensorError::Facade(error.into()))?;
        let raw = tenet_matrixalgebra::eigh_vals_dyn_checked_generic(dense.dense(), &input)?;
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
        let mut dense = self.runtime.lease_dense();
        let payload = body.materialized_dense_data();
        let input = BoundDynamicTensorRef::try_new(&body.space, &payload)
            .map_err(|error| GenericTensorError::Facade(error.into()))?;
        let out = tenet_matrixalgebra::eigh_full_dyn_checked_generic(dense.dense(), &input)?;
        let (v, mut eigenvalues) = out.into_parts();
        let d = diagonal_factor_on_checked(
            &self.runtime,
            Arc::clone(input.space().provider_arc()),
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
        let mut dense = self.runtime.lease_dense();
        let payload = body.materialized_dense_data();
        let input = BoundDynamicTensorRef::try_new(&body.space, &payload)
            .map_err(|error| GenericTensorError::Facade(error.into()))?;
        let out = tenet_matrixalgebra::eig_full_dyn_checked_generic(dense.dense(), &input)?;
        let (v, mut eigenvalues) = out.into_parts();
        let d = diagonal_factor_on_checked(
            &self.runtime,
            Arc::clone(input.space().provider_arc()),
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
        let mut dense = self.runtime.lease_dense();
        let payload = body.materialized_dense_data();
        let input = BoundDynamicTensorRef::try_new(&body.space, &payload)
            .map_err(|error| GenericTensorError::Facade(error.into()))?;
        let Qr { q, r } = tenet_matrixalgebra::qr_full_dyn_checked_generic(dense.dense(), &input)?;
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
        let mut dense = self.runtime.lease_dense();
        let payload = body.materialized_dense_data();
        let input = BoundDynamicTensorRef::try_new(&body.space, &payload)
            .map_err(|error| GenericTensorError::Facade(error.into()))?;
        let raw = tenet_matrixalgebra::svd_vals_dyn_checked_generic(dense.dense(), &input)?;
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
        let mut dense = self.runtime.lease_dense();
        let payload = body.materialized_dense_data();
        let input = BoundDynamicTensorRef::try_new(&body.space, &payload)
            .map_err(|error| GenericTensorError::Facade(error.into()))?;
        let Lq { l, q } =
            tenet_matrixalgebra::lq_compact_dyn_checked_generic(dense.dense(), &input)?;
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
        let mut dense = self.runtime.lease_dense();
        let payload = body.materialized_dense_data();
        let input = BoundDynamicTensorRef::try_new(&body.space, &payload)
            .map_err(|error| GenericTensorError::Facade(error.into()))?;
        let out = tenet_matrixalgebra::svd_full_dyn_checked_generic(dense.dense(), &input)?;
        let (u, s, vh, _) = out.into_parts();
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
        let mut dense = self.runtime.lease_dense();
        let payload = body.materialized_dense_data();
        let input = BoundDynamicTensorRef::try_new(&body.space, &payload)
            .map_err(|error| GenericTensorError::Facade(error.into()))?;
        let Svd { u, s, vh } =
            tenet_matrixalgebra::svd_compact_dyn_checked_generic(dense.dense(), &input)?;
        Ok(Svd {
            u: wrap_factor_on(&self.runtime, u),
            s: wrap_factor_on(&self.runtime, s),
            vh: wrap_factor_on(&self.runtime, vh),
        })
    }
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
        let mut dense = self.runtime.lease_dense();
        let payload = body.materialized_dense_data();
        let input = BoundDynamicTensorRef::try_new(&body.space, &payload)
            .map_err(|error| GenericTensorError::Facade(error.into()))?;
        let Qr { q, r } =
            tenet_matrixalgebra::qr_compact_dyn_checked_generic(dense.dense(), &input)?;
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
    /// Each sector runs one dense QR, with cost
    /// `O(sum_c m_c * n_c * min(m_c, n_c))`. Compact diagonal input is
    /// materialized first. Multiplicity-free lazy adjoints are materialized
    /// only for the operation; checked-Generic QR requires an
    /// owned input. Checked factors use the same provider instance as `self`.
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
    /// a rectangular dense `s`.
    ///
    /// For multiplicity-free providers, compact `s` stores only
    /// `sum_c k_c` diagonal values and bond composition becomes a scaling.
    /// Checked-Generic compact SVD currently returns a dense `s`; checked EIGH
    /// and EIG are the Generic factorizations whose diagonal factor is compact.
    /// All checked factors retain the source's exact provider `Arc`.
    ///
    /// Dense inputs cost `O(sum_c m_c * n_c * min(m_c, n_c))`. An owned
    /// multiplicity-free compact diagonal with representable magnitudes is
    /// sorted directly by sector, without
    /// a dense input or dense SVD call; its dense `u` and `vh` still require
    /// `O(sum_c k_c²)` output storage and writes. A multiplicity-free lazy
    /// adjoint is handled from its parent without materializing it.
    /// Checked-Generic SVD requires owned input and still densifies compact
    /// diagonals. Nonfinite or unrepresentable compact spectra use the ordinary
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
    /// the dense rectangular `m_c x n_c` diagonal matrix. The spaces are
    /// `u : codomain <- W_out`, `s : W_out <- W_in`, and
    /// `vh : W_in <- domain`. It accepts the same inputs as
    /// [`Self::svd_compact`], including the operation-local densification of
    /// a compact diagonal input, but its square outer factors can require more
    /// dense storage. Checked factors use the source provider instance, and a
    /// failure returns no factors.
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
    /// Its cost is the same as [`Self::qr_compact`]. Compact inputs are
    /// materialized first, and checked Generic requires an owned input.
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
    /// Adds every source block into the matching destination block rectangle.
    #[doc(hidden)]
    pub fn network_scatter_add_assign(
        &mut self,
        source: &Self,
        ranges: &[Option<std::ops::Range<usize>>],
    ) -> Result<(), Error> {
        if !self.runtime.same_runtime(&source.runtime) {
            return Err(Error::RuntimeMismatch);
        }
        let destination_space = self.logical_space().space();
        let _host_pool = self.runtime.enter_host_pool();
        let source_space = source.logical_space().space();
        if destination_space.nout() != source_space.nout()
            || destination_space.nin() != source_space.nin()
            || ranges.len() != destination_space.rank()
            || !Arc::ptr_eq(
                self.logical_space().provider_arc(),
                source.logical_space().provider_arc(),
            )
        {
            return Err(Error::InvalidArgument(
                "network slice partial does not match accumulator rank/provider".to_string(),
            ));
        }
        let destination_legs = destination_space
            .homspace()
            .codomain()
            .legs()
            .iter()
            .chain(destination_space.homspace().domain().legs());
        let source_legs = source_space
            .homspace()
            .codomain()
            .legs()
            .iter()
            .chain(source_space.homspace().domain().legs());
        let mut sliced = Vec::new();
        for (axis, ((destination_leg, source_leg), range)) in
            destination_legs.zip(source_legs).zip(ranges).enumerate()
        {
            match range {
                None if source_leg == destination_leg => {}
                None => {
                    return Err(Error::InvalidArgument(
                        "network slice unsliced output leg differs from destination".to_string(),
                    ));
                }
                Some(range) => {
                    let Some((sector, degeneracy)) = source_leg.iter().next() else {
                        return Err(Error::InvalidArgument(
                            "network sliced output leg must select one sector".to_string(),
                        ));
                    };
                    if source_leg.iter().nth(1).is_some()
                        || source_leg.is_dual() != destination_leg.is_dual()
                        || range.end.checked_sub(range.start) != Some(degeneracy)
                        || destination_leg
                            .degeneracy(sector)
                            .is_none_or(|destination| range.end > destination)
                    {
                        return Err(Error::InvalidArgument(
                            "network sliced output leg does not match destination range"
                                .to_string(),
                        ));
                    }
                    sliced.push((axis, [(sector, range.clone())]));
                }
            }
        }
        // The sliced source leg carries exactly one sector (checked above), so
        // each restricted axis needs a single-entry table; the kernel reads the
        // sector back from each source block's own key.
        let mut scatter_ranges: Vec<tenet_tensors::SectorRangeTable<'_>> = vec![None; ranges.len()];
        for (axis, table) in &sliced {
            scatter_ranges[*axis] = Some(table.as_slice());
        }
        let source_is_dense = match &source.repr {
            TypedTensorRepr::Owned(body) => matches!(body.data.as_ref(), TypedData::Dense(_)),
            TypedTensorRepr::Adjoint(_) => true,
        };
        if !source_is_dense {
            return Err(Error::InvalidArgument(
                "network slice partial must have dense payload".to_string(),
            ));
        }
        let destination_structure = destination_space.structure().clone();
        let source_structure = source_space.structure();
        let (source_operand, source_data) = source.fusion_operand_and_data();
        let TypedTensorRepr::Owned(destination_body) = &mut self.repr else {
            return Err(Error::InvalidArgument(
                "network slice accumulator must be direct owned".to_string(),
            ));
        };
        let destination_body = Arc::get_mut(destination_body).ok_or_else(|| {
            Error::InvalidArgument("network slice accumulator payload is shared".to_string())
        })?;
        let TypedData::Dense(destination_data) = Arc::get_mut(&mut destination_body.data)
            .ok_or_else(|| {
                Error::InvalidArgument("network slice accumulator data is shared".to_string())
            })?
        else {
            return Err(Error::InvalidArgument(
                "network slice accumulator must have dense payload".to_string(),
            ));
        };
        tenet_tensors::fusion_scatter_add_assign(
            &destination_structure,
            destination_data,
            source_structure,
            source_operand,
            &source_data,
            &scatter_ranges,
        )
        .map_err(Error::from)
    }

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
    /// The receiver must store a compact diagonal: a multiplicity-free
    /// [`Self::svd_compact`] `s`, an eigendecomposition's `d`, or a tensor
    /// built by [`Self::diagonal`]. A dense tensor is rejected even when its
    /// blocks happen to be diagonal; this method never scans or builds a
    /// dense buffer. A diagonal that is stored densely (a checked-Generic
    /// `svd_compact` `s`) is made compact explicitly with
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
    /// publication are additional costs. See [`Self::qr_compact`] for the
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
    /// adjoint, and owned factor publication are additional costs. See
    /// [`Self::lq_compact`] for the compact alternative, storage and
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
    /// is read and sorted directly without a dense solver. Other compact cases
    /// retain the dense solver's behavior;
    /// checked Generic currently requires an owned input and returns
    /// [`Error::InvalidArgument`] for a lazy adjoint. A dense failure returns
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
    /// [`Self::eigh_full`]. An owned Host multiplicity-free compact diagonal
    /// with finite, exactly real entries is read directly; other compact
    /// inputs and lazy adjoints use the dense route. Checked Generic currently
    /// requires owned input for this values-only method. Dense failures return
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
    /// `Arc`. An owned Host multiplicity-free compact diagonal with finite,
    /// exactly real entries is read directly into a permutation eigenbasis;
    /// its dense output still occupies `Σ_c k_c²` elements. Other compact
    /// inputs and lazy adjoints use an operation-local dense payload.
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
    /// endomorphism. An owned Host multiplicity-free compact diagonal with
    /// finite eigenvalue magnitudes is read directly. Other compact inputs and
    /// lazy adjoints use an operation-local dense payload; checked Generic
    /// currently requires owned input for this values-only method. Unlike
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
    /// without this additional rank gate. An admitted owned Host
    /// multiplicity-free compact diagonal reads finite eigenvalues directly
    /// and builds a dense permutation factor. Lazy adjoints and other inputs
    /// use an operation-local dense payload.
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
