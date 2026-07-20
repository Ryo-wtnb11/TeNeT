//! User-layer symmetric tensor: dynamic rank, rule-erased, runtime-carrying.
//!
//! A [`Tensor`] stores one enum-erased provider-bound dynamic fusion space plus
//! flat scalar storage (`f64` or `Complex64`, chosen at construction) in the
//! TensorKit-equivalent coupled-sector matrix layout. The scalar type is
//! erased behind an internal storage enum; rank is fully dynamic (no ceiling),
//! matching TensorKit's `tensorcontract!`. CPU operations briefly acquire a
//! per-operation context and/or executor lease, then run with that resource
//! exclusively without holding the [`Runtime`]'s coarse shared-state lock.
//! They dispatch on the stored rule and dtype once per call (never per block)
//! and forward the bound authority to the expert layer.

use std::hash::Hash;
use std::sync::{Arc, Mutex, OnceLock};

use num_complex::Complex64;
use smallvec::SmallVec;
use tenet_core::{
    BlockKey, BlockStructure, CheckedFusionAlgebra, CoupledSectorRegion, FusionProductSpace,
    FusionRule, FusionTreeHomSpace, LoweredMultiplicityFreeAlgebra, MultiplicityFreeRigidSymbols,
    Placement, SectorId, Su3FusionRule,
};
#[cfg(feature = "cuda")]
use tenet_core::{SectorLeg, TensorStorage};
#[cfg(feature = "cuda")]
use tenet_dense::{
    cuda_eigh_region, cuda_gemm_region_into, cuda_qr_region, cuda_svd_region, CudaDenseContext,
    CudaDenseStorage,
};
#[cfg(feature = "cuda")]
use tenet_matrixalgebra::{select_truncation, validate_hermitian_regions, WeightedSpectrum};
use tenet_matrixalgebra::{
    BoundDynFactor, BoundDynamicTensorRef, FactorScalar, SectorSpectrum, Truncation,
};
#[cfg(feature = "cuda")]
use tenet_tensors::cuda::{CudaStorage, CudaStorageGemm};
use tenet_tensors::{
    BoundDynamicFusionMapSpace, DynamicFusionMapSpace, OperationError, OutputAxisOrder,
    RecouplingCoefficientAction, TensorContractSpec, TreeTransformOperation,
    TreeTransformRuleCacheKey,
};

use crate::error::Error;
use crate::runtime::{rule_lanes, Ctx, Ctxs, Runtime, RuntimeExecutionConfig, RuntimeIdentity};
use crate::space::{Fz2U1Su2Rule, RuleKind, Space, U1Fz2Rule, UserRuleContext};

mod diagonal;
use diagonal::{axpby_dense_c64, axpby_dense_real, compact_inner, dense_inner};

#[cfg(test)]
thread_local! {
    static PERMUTE_PRE_REPLAY_POISON: std::cell::Cell<Option<bool>> =
        const { std::cell::Cell::new(None) };
    static ORDERED_CONTRACT_FUSED_ROUTE: std::cell::Cell<Option<bool>> =
        const { std::cell::Cell::new(None) };
    static SELECTED_RESULT_LAYOUT_BUILDS: std::cell::Cell<Option<usize>> =
        const { std::cell::Cell::new(None) };
}

#[cfg(test)]
fn observe_permute_pre_replay_poison(is_poisoned: bool) {
    PERMUTE_PRE_REPLAY_POISON.with(|observation| {
        if observation.get().is_some() {
            observation.set(Some(is_poisoned));
        }
    });
}

#[cfg(test)]
fn observe_ordered_contract_fused_route() {
    ORDERED_CONTRACT_FUSED_ROUTE.with(|observation| {
        if observation.get().is_some() {
            observation.set(Some(true));
        }
    });
}

#[cfg(test)]
fn observe_selected_result_layout_build() {
    SELECTED_RESULT_LAYOUT_BUILDS.with(|observation| {
        if let Some(builds) = observation.get() {
            observation.set(Some(builds + 1));
        }
    });
}

/// The scalar type a [`Tensor`] stores, fixed at construction.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum Dtype {
    /// Real double precision (`f64`).
    F64,
    /// Complex double precision ([`Complex64`]).
    C64,
}

/// A scalar produced by a [`Tensor`] reduction ([`Tensor::scalar`],
/// [`Tensor::inner`], [`Tensor::tr`]): the variant matches the producing
/// tensor's [`Dtype`], mirroring TensorKit, where `dot`/`tr` on a real
/// tensor return a real scalar. Non-exhaustive so future precisions
/// (f32/c32) can add variants.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Scalar {
    /// Real double precision.
    F64(f64),
    /// Complex double precision.
    C64(Complex64),
}

impl Scalar {
    /// The real part (the value itself for real variants).
    pub fn re(self) -> f64 {
        match self {
            Self::F64(value) => value,
            Self::C64(value) => value.re,
        }
    }

    /// The imaginary part (exactly `0.0` for real variants).
    pub fn im(self) -> f64 {
        match self {
            Self::F64(_) => 0.0,
            Self::C64(value) => value.im,
        }
    }

    /// The value as `f64`; [`Error::DtypeMismatch`] on complex variants.
    /// Use [`Self::re`] when you deliberately want the real part of a
    /// complex scalar.
    pub fn try_f64(self) -> Result<f64, Error> {
        match self {
            Self::F64(value) => Ok(value),
            Self::C64(_) => Err(Error::DtypeMismatch),
        }
    }

    /// Widens to [`Complex64`] (exact for every variant).
    pub fn to_c64(self) -> Complex64 {
        match self {
            Self::F64(value) => Complex64::new(value, 0.0),
            Self::C64(value) => value,
        }
    }
}

impl std::fmt::Display for Scalar {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::F64(value) => write!(f, "{value}"),
            Self::C64(value) => write!(f, "{value}"),
        }
    }
}

/// Dtype-erased flat storage in the coupled-sector matrix layout. The
/// device variant shares the immutable buffer behind an `Arc` (operations
/// always write fresh destinations), keeping `Tensor: Clone` cheap and the
/// host paths untouched.
#[derive(Clone, Debug)]
pub enum Data {
    F64(Vec<f64>),
    C64(Vec<Complex64>),
    /// Compact O(rank) diagonal storage for spectrum tensors (SVD `s`, eigh/eig
    /// `d`): only the per-sector diagonal values are held, not the dense
    /// block-diagonal matrix (issue #55). Storage-local operations consume this
    /// representation directly; dense-only operations use
    /// [`Tensor::coupled_data`] as the explicit materialization boundary.
    Diagonal(DiagonalData),
    #[cfg(feature = "cuda")]
    CudaF64(Arc<CudaStorage>),
}

/// The values of a [`Data::Diagonal`] tensor plus how they materialize into a
/// dense buffer, chosen to reproduce the former dense diagonal bit-for-bit:
/// SVD singular values / Hermitian eigenvalues are real (`RealF64`, or `RealC64`
/// when the source tensor is complex), general eigenvalues are complex (`C64`).
#[derive(Clone, Debug)]
pub enum DiagonalData {
    RealF64(Vec<SectorSpectrum<f64>>),
    RealC64(Vec<SectorSpectrum<f64>>),
    C64(Vec<SectorSpectrum<Complex64>>),
}

impl DiagonalData {
    fn dtype(&self) -> Dtype {
        match self {
            DiagonalData::RealF64(_) => Dtype::F64,
            DiagonalData::RealC64(_) | DiagonalData::C64(_) => Dtype::C64,
        }
    }

    /// TensorKit `tr` on compact diagonal storage: sum the reduced diagonal
    /// values with their quantum dimensions. Why not reuse `trace_pairs`: that
    /// contraction API intentionally inserts orientation-dependent fermionic
    /// twists, while matrix trace uses TensorKit's positive trace formalism.
    fn ordinary_trace_with(&self, dim: impl Fn(SectorId) -> f64) -> Complex64 {
        let trace_real = |spectra: &[SectorSpectrum<f64>]| {
            spectra
                .iter()
                .map(|entry| entry.values.iter().sum::<f64>() * dim(entry.sector))
                .sum::<f64>()
        };
        match self {
            Self::RealF64(spectra) | Self::RealC64(spectra) => {
                Complex64::new(trace_real(spectra), 0.0)
            }
            Self::C64(spectra) => spectra
                .iter()
                .map(|entry| entry.values.iter().sum::<Complex64>() * dim(entry.sector))
                .sum(),
        }
    }

    fn ordinary_trace<R>(&self, rule: &R) -> Complex64
    where
        R: MultiplicityFreeRigidSymbols<Scalar = f64>,
    {
        self.ordinary_trace_with(|sector| rule.dim_scalar(sector))
    }

    fn elementwise_product(&self, rhs: &Self) -> Option<Self> {
        fn multiply<V: Copy>(
            lhs: &[SectorSpectrum<V>],
            rhs: &[SectorSpectrum<V>],
            mul: impl Fn(V, V) -> V,
        ) -> Option<Vec<SectorSpectrum<V>>> {
            if lhs.len() != rhs.len() {
                return None;
            }
            lhs.iter()
                .zip(rhs)
                .map(|(lhs, rhs)| {
                    if lhs.sector != rhs.sector || lhs.values.len() != rhs.values.len() {
                        return None;
                    }
                    Some(SectorSpectrum {
                        sector: lhs.sector,
                        values: lhs
                            .values
                            .iter()
                            .copied()
                            .zip(rhs.values.iter().copied())
                            .map(|(lhs, rhs)| mul(lhs, rhs))
                            .collect(),
                    })
                })
                .collect()
        }

        fn real_complex_product(
            real: &[SectorSpectrum<f64>],
            complex: &[SectorSpectrum<Complex64>],
        ) -> Option<Vec<SectorSpectrum<Complex64>>> {
            if real.len() != complex.len() {
                return None;
            }
            real.iter()
                .zip(complex)
                .map(|(real, complex)| {
                    if real.sector != complex.sector || real.values.len() != complex.values.len() {
                        return None;
                    }
                    Some(SectorSpectrum {
                        sector: real.sector,
                        values: real
                            .values
                            .iter()
                            .copied()
                            .zip(complex.values.iter().copied())
                            .map(|(real, complex)| real * complex)
                            .collect(),
                    })
                })
                .collect()
        }

        match (self, rhs) {
            (Self::RealF64(lhs), Self::RealF64(rhs)) => {
                multiply(lhs, rhs, |lhs, rhs| lhs * rhs).map(Self::RealF64)
            }
            (Self::RealC64(lhs), Self::RealC64(rhs)) => {
                multiply(lhs, rhs, |lhs, rhs| lhs * rhs).map(Self::RealC64)
            }
            (Self::C64(lhs), Self::C64(rhs)) => {
                multiply(lhs, rhs, |lhs, rhs| lhs * rhs).map(Self::C64)
            }
            (Self::RealC64(real), Self::C64(complex)) => {
                real_complex_product(real, complex).map(Self::C64)
            }
            (Self::C64(complex), Self::RealC64(real)) => {
                real_complex_product(real, complex).map(Self::C64)
            }
            _ => None,
        }
    }

    /// Multiplies every stored value by a real factor, preserving the variant —
    /// so scaling a diagonal factor (e.g. itebd's `λ / |λ|`) keeps O(rank)
    /// storage instead of densifying.
    fn scaled(&self, factor: f64) -> DiagonalData {
        fn map_real(spectra: &[SectorSpectrum<f64>], factor: f64) -> Vec<SectorSpectrum<f64>> {
            spectra
                .iter()
                .map(|entry| SectorSpectrum {
                    sector: entry.sector,
                    values: entry.values.iter().map(|&value| value * factor).collect(),
                })
                .collect()
        }
        match self {
            DiagonalData::RealF64(spectra) => DiagonalData::RealF64(map_real(spectra, factor)),
            DiagonalData::RealC64(spectra) => DiagonalData::RealC64(map_real(spectra, factor)),
            DiagonalData::C64(spectra) => DiagonalData::C64(
                spectra
                    .iter()
                    .map(|entry| SectorSpectrum {
                        sector: entry.sector,
                        values: entry.values.iter().map(|&value| value * factor).collect(),
                    })
                    .collect(),
            ),
        }
    }

    /// The largest `|entry|` over all sectors (for `pinv`'s relative cutoff).
    fn max_abs(&self) -> f64 {
        match self {
            DiagonalData::RealF64(s) | DiagonalData::RealC64(s) => s
                .iter()
                .flat_map(|e| e.values.iter())
                .fold(0.0f64, |m, &v| m.max(v.abs())),
            DiagonalData::C64(s) => s
                .iter()
                .flat_map(|e| e.values.iter())
                .fold(0.0f64, |m, &v| m.max(v.norm())),
        }
    }

    /// Elementwise reciprocal — the diagonal `inv` (TensorKit `inv.(d.data)`).
    /// Errors on a zero entry, like the dense `inv` on a rank-deficient block.
    fn try_recip(&self) -> Result<DiagonalData, Error> {
        fn recip_real(s: &[SectorSpectrum<f64>]) -> Result<Vec<SectorSpectrum<f64>>, Error> {
            s.iter()
                .map(|e| {
                    Ok(SectorSpectrum {
                        sector: e.sector,
                        values: e
                            .values
                            .iter()
                            .map(|&v| {
                                if v == 0.0 {
                                    Err(Error::InvalidArgument(
                                        "inv of a singular diagonal (zero entry)".to_string(),
                                    ))
                                } else {
                                    Ok(1.0 / v)
                                }
                            })
                            .collect::<Result<_, Error>>()?,
                    })
                })
                .collect()
        }
        Ok(match self {
            DiagonalData::RealF64(s) => DiagonalData::RealF64(recip_real(s)?),
            DiagonalData::RealC64(s) => DiagonalData::RealC64(recip_real(s)?),
            DiagonalData::C64(s) => DiagonalData::C64(
                s.iter()
                    .map(|e| {
                        Ok(SectorSpectrum {
                            sector: e.sector,
                            values: e
                                .values
                                .iter()
                                .map(|&v| {
                                    if v == Complex64::new(0.0, 0.0) {
                                        Err(Error::InvalidArgument(
                                            "inv of a singular diagonal (zero entry)".to_string(),
                                        ))
                                    } else {
                                        Ok(Complex64::new(1.0, 0.0) / v)
                                    }
                                })
                                .collect::<Result<_, Error>>()?,
                        })
                    })
                    .collect::<Result<_, Error>>()?,
            ),
        })
    }

    /// Elementwise pseudo-inverse with an `rcond * max|entry|` cutoff (TensorKit
    /// `pinv` on a diagonal): entries at or below the cutoff map to 0, the rest
    /// to `1/entry`. Same variant (`1/entry` of a real entry stays real).
    fn pinv(&self, rcond: f64) -> DiagonalData {
        let cutoff = rcond * self.max_abs();
        fn map_real(s: &[SectorSpectrum<f64>], cutoff: f64) -> Vec<SectorSpectrum<f64>> {
            s.iter()
                .map(|e| SectorSpectrum {
                    sector: e.sector,
                    values: e
                        .values
                        .iter()
                        .map(|&v| if v.abs() > cutoff { 1.0 / v } else { 0.0 })
                        .collect(),
                })
                .collect()
        }
        match self {
            DiagonalData::RealF64(s) => DiagonalData::RealF64(map_real(s, cutoff)),
            DiagonalData::RealC64(s) => DiagonalData::RealC64(map_real(s, cutoff)),
            DiagonalData::C64(s) => DiagonalData::C64(
                s.iter()
                    .map(|e| SectorSpectrum {
                        sector: e.sector,
                        values: e
                            .values
                            .iter()
                            .map(|&v| {
                                if v.norm() > cutoff {
                                    Complex64::new(1.0, 0.0) / v
                                } else {
                                    Complex64::new(0.0, 0.0)
                                }
                            })
                            .collect(),
                    })
                    .collect(),
            ),
        }
    }

    /// Elementwise principal square root (TensorKit `sqrt.(d.data)`). A real
    /// (`RealF64`) diagonal errors on a negative entry (like the dense f64
    /// `sqrt`); a complex-typed real spectrum (`RealC64`) takes the complex root
    /// and promotes to `C64`, matching the dense c64 `sqrt`.
    fn try_sqrt(&self) -> Result<DiagonalData, Error> {
        let map_c64 = |s: &[SectorSpectrum<Complex64>]| -> Vec<SectorSpectrum<Complex64>> {
            s.iter()
                .map(|e| SectorSpectrum {
                    sector: e.sector,
                    values: e.values.iter().map(|&v| v.sqrt()).collect(),
                })
                .collect()
        };
        Ok(match self {
            DiagonalData::RealF64(s) => DiagonalData::RealF64(
                s.iter()
                    .map(|e| {
                        Ok(SectorSpectrum {
                            sector: e.sector,
                            values: e
                                .values
                                .iter()
                                .map(|&v| {
                                    if v < 0.0 {
                                        Err(Error::InvalidArgument(format!(
                                            "sqrt of a negative diagonal entry {v}; convert to \
                                             c64 with to_c64() for the complex square root"
                                        )))
                                    } else {
                                        Ok(v.sqrt())
                                    }
                                })
                                .collect::<Result<_, Error>>()?,
                        })
                    })
                    .collect::<Result<_, Error>>()?,
            ),
            DiagonalData::RealC64(s) => DiagonalData::C64(map_c64(
                &s.iter()
                    .map(|e| SectorSpectrum {
                        sector: e.sector,
                        values: e.values.iter().map(|&v| Complex64::new(v, 0.0)).collect(),
                    })
                    .collect::<Vec<_>>(),
            )),
            DiagonalData::C64(s) => DiagonalData::C64(map_c64(s)),
        })
    }
}

/// Explicit "no device kernel yet" error; device tensors never fall back
/// to host execution silently.
#[cfg(feature = "cuda")]
fn device_unsupported(what: &str) -> Error {
    Error::UnsupportedOnDevice(format!(
        "{what} has no device implementation yet; move the tensor to the \
         host explicitly with to_host()"
    ))
}

/// The scalar types a [`Tensor`] can store: `f64` and [`Complex64`]. This
/// trait is **sealed** (its supertrait is crate-private); it exists so
/// [`Tensor::from_block_fn`] can infer the constructed dtype from the fill
/// closure's return type.
pub trait TensorScalar: UserScalar {}

impl TensorScalar for f64 {}
impl TensorScalar for Complex64 {}

/// The scalar types the user layer stores: the expert-layer scalar machinery
/// plus the glue to lift typed data into the erased [`Data`] storage and to
/// pick the matching per-scalar execution context. Crate-private supertrait
/// sealing [`TensorScalar`].
pub trait UserScalar: FactorScalar + RecouplingCoefficientAction<f64> {
    fn lift(data: Vec<Self>) -> Data;
    fn ctx_of<Key: Clone + Eq + Hash + Send + Sync + 'static>(
        ctxs: &mut Ctxs<Key>,
    ) -> &mut Ctx<Self, Key>;
    fn rand_unit(state: &mut u64) -> Self;
}

impl UserScalar for f64 {
    fn lift(data: Vec<Self>) -> Data {
        Data::F64(data)
    }

    fn ctx_of<Key: Clone + Eq + Hash + Send + Sync + 'static>(
        ctxs: &mut Ctxs<Key>,
    ) -> &mut Ctx<Self, Key> {
        &mut ctxs.f64
    }

    fn rand_unit(state: &mut u64) -> Self {
        rand_unit(state)
    }
}

impl UserScalar for Complex64 {
    fn lift(data: Vec<Self>) -> Data {
        Data::C64(data)
    }

    fn ctx_of<Key: Clone + Eq + Hash + Send + Sync + 'static>(
        ctxs: &mut Ctxs<Key>,
    ) -> &mut Ctx<Self, Key> {
        &mut ctxs.c64
    }

    fn rand_unit(state: &mut u64) -> Self {
        Complex64::new(rand_unit(state), rand_unit(state))
    }
}

macro_rules! define_tensor_execution_context {
    ($( $field:ident: $key:ty ),+ $(,)?) => {
        /// Caller-owned host execution state for dynamic destination operations.
        /// [`Self::default`] is independent of a [`Runtime`]; use
        /// [`Self::for_runtime`] when execution must inherit and remain bound to a
        /// runtime's backend configuration.
        #[derive(Default)]
        pub struct TensorExecutionContext {
            runtime: Option<Runtime>,
            runtime_identity: Option<RuntimeIdentity>,
            $($field: Ctxs<$key>,)+
        }

        impl TensorExecutionContext {
            // Why not retain a Runtime here: pooled contexts live inside that
            // Runtime, so the back-reference would form an Arc cycle.
            pub(crate) fn for_config(config: &RuntimeExecutionConfig) -> Result<Self, Error> {
                let mut context = Self {
                    runtime: None,
                    runtime_identity: None,
                    $($field: Ctxs::with_config(&config.shared_ctx, config.gemm_kind)?,)+
                };
                if let Some(threads) = config.recoupling_threads {
                    context.set_recoupling_threads(threads);
                }
                if let Some(backend) = config.transpose_backend {
                    context.set_transpose_backend(backend);
                }
                Ok(context)
            }

            fn set_recoupling_threads(&mut self, threads: usize) {
                $(self.$field.set_recoupling_threads(threads);)+
            }

            fn set_transpose_backend(&mut self, backend: tenet_tensors::TransposeBackend) {
                $(self.$field.set_transpose_backend(backend);)+
            }

            #[doc(hidden)]
            pub fn release_runtime_binding(&mut self) {
                self.runtime = None;
            }

            #[doc(hidden)]
            pub fn bind_runtime(&mut self, runtime: &Runtime) -> Result<(), Error> {
                if self
                    .runtime_identity
                    .as_ref()
                    .is_some_and(|identity| !identity.matches(runtime))
                {
                    return Err(Error::RuntimeMismatch);
                }
                self.runtime_identity = Some(runtime.identity());
                self.runtime = Some(runtime.clone());
                Ok(())
            }

            #[cfg(test)]
            fn recoupling_threads_are(&mut self, expected: usize) -> bool {
                true $(&& self.$field.recoupling_threads_are(expected))+
            }

            #[cfg(test)]
            fn shares_cpu_context(&mut self, shared: &tenet_dense::SharedCpuContext) -> bool {
                true $(&& self.$field.shares_cpu_context(shared))+
            }
        }
    };
}

rule_lanes!(define_tensor_execution_context);

#[doc(hidden)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum OverwriteOutcome {
    Written,
    Incompatible,
}

// Usage: `TensorExecutionContext::try_contract_overwrite_into`, matching on
// `OverwriteOutcome::Incompatible` to fall back to an owned allocation (see
// that method's doc example). Stays `#[doc(hidden)]` alongside `try_*`
// itself: un-hiding just the method would put a hidden cache type in its
// public signature, so both move together or not at all (issue #150).
#[doc(hidden)]
#[derive(Default)]
pub struct ContractOverwriteCache {
    prepared: Option<PreparedContractOverwrite>,
    preparations: u64,
    structural_comparisons: u64,
}

struct PreparedContractOverwrite {
    lhs_space: Arc<UserBoundSpace>,
    rhs_space: Arc<UserBoundSpace>,
    lhs_axes: Vec<usize>,
    rhs_axes: Vec<usize>,
    output_axes: Vec<usize>,
    expected: Arc<UserBoundSpace>,
}

// Usage: `TensorExecutionContext::try_permute_overwrite_into`, same
// Incompatible/Written pattern as `ContractOverwriteCache` above.
#[doc(hidden)]
#[derive(Default)]
pub struct PermuteOverwriteCache {
    prepared: Option<PreparedPermuteOverwrite>,
    preparations: u64,
    structural_comparisons: u64,
}

struct PreparedPermuteOverwrite {
    source_space: Arc<UserBoundSpace>,
    codomain_axes: Vec<usize>,
    domain_axes: Vec<usize>,
    operation: TreeTransformOperation,
    expected: Arc<UserBoundSpace>,
}

enum PreparedPermuteOperation<'a> {
    Owned(TreeTransformOperation),
    Borrowed(&'a TreeTransformOperation),
}

fn same_dynamic_space_counted(
    lhs: &Arc<UserBoundSpace>,
    rhs: &Arc<UserBoundSpace>,
    structural_comparisons: &mut u64,
) -> bool {
    if Arc::ptr_eq(lhs, rhs) {
        true
    } else {
        *structural_comparisons += 1;
        lhs.as_ref() == rhs.as_ref()
    }
}

impl ContractOverwriteCache {
    #[doc(hidden)]
    pub fn preparations(&self) -> u64 {
        self.preparations
    }

    #[doc(hidden)]
    pub fn structural_comparisons(&self) -> u64 {
        self.structural_comparisons
    }
}

impl PermuteOverwriteCache {
    #[doc(hidden)]
    pub fn preparations(&self) -> u64 {
        self.preparations
    }

    #[doc(hidden)]
    pub fn structural_comparisons(&self) -> u64 {
        self.structural_comparisons
    }
}

impl TensorExecutionContext {
    /// Builds caller-owned execution state with the same CPU execution
    /// configuration as `runtime`, bound to it for runtime validation.
    pub fn for_runtime(runtime: &Runtime) -> Result<Self, Error> {
        let mut context = Self::for_config(runtime.execution_config())?;
        context.runtime_identity = Some(runtime.identity());
        context.runtime = Some(runtime.clone());
        Ok(context)
    }

    fn accepts_runtime(&self, tensor: &Tensor) -> bool {
        self.runtime
            .as_ref()
            .map(|runtime| runtime.shares_state_with(tensor.runtime()))
            .or_else(|| {
                self.runtime_identity
                    .as_ref()
                    .map(|identity| identity.matches(tensor.runtime()))
            })
            .unwrap_or(true)
    }

    fn validate_runtime(&self, tensor: &Tensor) -> Result<(), Error> {
        if self.accepts_runtime(tensor) {
            Ok(())
        } else {
            Err(Error::InvalidArgument(
                "execution context is bound to a different runtime".to_string(),
            ))
        }
    }
}

/// Dispatches once on the stored dtype of `$tensor`, binding `$data` to the
/// typed data vector in both arms; `$body` must be dtype-generic (the expert
/// entry points are generic over the scalar).
macro_rules! with_data {
    ($tensor:expr, $data:ident, $body:expr) => {
        match $tensor.coupled_data()? {
            Data::F64($data) => $body,
            Data::C64($data) => $body,
            Data::Diagonal(_) => unreachable!("coupled_data materializes Data::Diagonal"),
            #[cfg(feature = "cuda")]
            Data::CudaF64(_) => return Err(device_unsupported("this operation")),
        }
    };
}

/// Result of [`Tensor::svd_trunc`]: `t ~ u * s * vh` with the truncated bond
/// (TensorKit 0.17 / MatrixAlgebraKit `svd_trunc`). `singular_values` holds
/// the kept per-sector spectra and `error` the quantum-dimension-weighted
/// 2-norm of everything discarded.
#[derive(Clone, Debug)]
pub struct SvdTrunc {
    /// Left isometry `U` (codomain legs `<- bond`).
    pub u: Tensor,
    /// Diagonal singular-value tensor `S` (`bond <- bond`).
    pub s: Tensor,
    /// Right isometry `V†` (`bond <- domain legs`).
    pub vh: Tensor,
    /// Kept singular values per coupled sector.
    pub singular_values: Vec<SectorSpectrum>,
    /// Quantum-dimension-weighted 2-norm of the discarded singular values.
    pub error: f64,
}

/// Result of [`Tensor::eigh_trunc`]: `t ~ v * d * v^H` with the truncated
/// bond; `error` is the quantum-dimension-weighted 2-norm of the discarded
/// eigenvalues.
#[derive(Clone, Debug)]
pub struct EighTrunc {
    /// Diagonal eigenvalue tensor `D` (`bond <- bond`), real for Hermitian input.
    pub d: Tensor,
    /// Eigenvector isometry `V` (codomain legs `<- bond`).
    pub v: Tensor,
    /// Kept eigenvalues per coupled sector.
    pub eigenvalues: Vec<SectorSpectrum>,
    /// Quantum-dimension-weighted 2-norm of the discarded eigenvalues.
    pub error: f64,
}

/// Result of [`Tensor::eig_trunc`]: `t ~ v * d * v^-1` with the truncated
/// bond. `d` and `v` are always c64 (the general eigendecomposition is
/// complex-valued even for real input); `error` is the
/// quantum-dimension-weighted 2-norm of the discarded `|eigenvalues|`.
#[derive(Clone, Debug)]
pub struct EigTrunc {
    /// Diagonal eigenvalue tensor `D` (`bond <- bond`), always c64.
    pub d: Tensor,
    /// Eigenvector tensor `V` (codomain legs `<- bond`), always c64.
    pub v: Tensor,
    /// Kept (complex) eigenvalues per coupled sector.
    pub eigenvalues: Vec<SectorSpectrum<Complex64>>,
    /// Quantum-dimension-weighted 2-norm of the discarded `|eigenvalues|`.
    pub error: f64,
}

/// How a freshly built tensor is filled.
enum Fill<'f, D> {
    Zeros,
    Rand(u64),
    BlockFn(&'f mut dyn FnMut(&BlockKey, &[usize]) -> D),
}

/// splitmix64: small deterministic RNG for [`Tensor::rand`]; no external
/// dependency, values uniform in `[-1, 1)`.
fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

fn rand_unit(state: &mut u64) -> f64 {
    ((splitmix64(state) >> 11) as f64) / ((1u64 << 52) as f64) - 1.0
}

fn build_bound_space<
    R: MultiplicityFreeRigidSymbols<Scalar = f64>
        + LoweredMultiplicityFreeAlgebra
        + tenet_core::CheckedFusionAlgebra,
>(
    provider: Arc<R>,
    hom: FusionTreeHomSpace,
) -> Result<BoundDynamicFusionMapSpace<R>, Error> {
    BoundDynamicFusionMapSpace::from_final_homspace_multiplicity_free_lowered(provider, hom)
        .map_err(Into::into)
}

fn build_bound_space_like<
    R: MultiplicityFreeRigidSymbols<Scalar = f64>
        + LoweredMultiplicityFreeAlgebra
        + tenet_core::CheckedFusionAlgebra,
>(
    authority: &BoundDynamicFusionMapSpace<R>,
    hom: FusionTreeHomSpace,
) -> Result<BoundDynamicFusionMapSpace<R>, Error> {
    authority
        .derive_from_final_homspace(hom)
        .map_err(Into::into)
}

fn build_bound_space_generic<R: FusionRule>(
    provider: Arc<R>,
    hom: FusionTreeHomSpace,
) -> Result<BoundDynamicFusionMapSpace<R>, Error> {
    let leg_deg = |leg: &tenet_core::SectorLeg, sector: SectorId| -> Result<usize, Error> {
        leg.degeneracy(sector).ok_or_else(|| {
            Error::InvalidArgument(format!("sector {sector:?} not present on this leg"))
        })
    };
    let keys = hom
        .fusion_tree_keys_generic(provider.as_ref())
        .map_err(tenet_tensors::OperationError::from_core_preserving_context)?;
    let mut shapes = Vec::with_capacity(keys.len());
    for key in keys.iter() {
        let mut shape = Vec::with_capacity(hom.rank());
        for (leg, &sector) in hom.codomain().legs().iter().zip(key.codomain_uncoupled()) {
            shape.push(leg_deg(leg, sector)?);
        }
        for (leg, &sector) in hom.domain().legs().iter().zip(key.domain_uncoupled()) {
            shape.push(leg_deg(leg, sector)?);
        }
        shapes.push(shape);
    }
    BoundDynamicFusionMapSpace::from_degeneracy_shapes_generic(provider, hom, shapes)
        .map_err(Into::into)
}

/// Fills a freshly-built coupled space (rule-agnostic: only touches the block
/// structure). Shared by the mult-free and SU(3) construction paths.
fn apply_fill<S: UserScalar>(
    space: &DynamicFusionMapSpace,
    fill: Fill<'_, S>,
) -> Result<Vec<S>, Error> {
    let len = space.required_len()?;
    let mut data = vec![S::from_real(0.0); len];
    match fill {
        Fill::Zeros => {}
        Fill::Rand(seed) => {
            let mut state = seed;
            for value in &mut data {
                *value = S::rand_unit(&mut state);
            }
        }
        Fill::BlockFn(fill) => {
            fill_block_elements(space.structure(), &mut data, fill)?;
        }
    }
    Ok(data)
}

/// Fills every symmetry-allowed block element via `fill(key, indices)`,
/// mirroring [`tenet_core::TensorMap::from_block_fn_with_fusion_space`]
/// (degeneracy coordinates local to the block, codomain axes first, first
/// axis fastest).
fn fill_block_elements<D: UserScalar>(
    structure: &BlockStructure,
    data: &mut [D],
    fill: &mut dyn FnMut(&BlockKey, &[usize]) -> D,
) -> Result<(), Error> {
    for index in 0..structure.block_count() {
        let block = structure.block(index)?;
        let shape = block.shape();
        let strides = block.strides();
        let offset = block.offset();
        let count: usize = shape.iter().product();
        let mut indices = vec![0usize; shape.len()];
        for _ in 0..count {
            let position = offset
                + indices
                    .iter()
                    .zip(strides)
                    .map(|(&i, &s)| i * s)
                    .sum::<usize>();
            data[position] = fill(block.key(), &indices);
            for axis in 0..shape.len() {
                indices[axis] += 1;
                if indices[axis] < shape[axis] {
                    break;
                }
                indices[axis] = 0;
            }
        }
    }
    Ok(())
}

/// Scales every fusion-tree block of `data` in place by the real factor
/// `factor_of(key)` (skipping factor-1 blocks). Backs [`Tensor::twist`] and
/// [`Tensor::flip`], whose effect on the storage is exactly a per-block
/// phase.
fn scale_blocks_impl<D: UserScalar>(
    space: &DynamicFusionMapSpace,
    data: &mut [D],
    factor_of: &dyn Fn(&BlockKey) -> f64,
) -> Result<(), Error> {
    let structure = space.structure();
    for index in 0..structure.block_count() {
        let block = structure.block(index)?;
        let factor = factor_of(block.key());
        if factor == 1.0 {
            continue;
        }
        let factor = D::from_real(factor);
        let shape = block.shape();
        let strides = block.strides();
        let offset = block.offset();
        let count: usize = shape.iter().product();
        let mut indices = vec![0usize; shape.len()];
        for _ in 0..count {
            let position = offset
                + indices
                    .iter()
                    .zip(strides)
                    .map(|(&i, &s)| i * s)
                    .sum::<usize>();
            data[position] = data[position] * factor;
            for axis in 0..shape.len() {
                indices[axis] += 1;
                if indices[axis] < shape[axis] {
                    break;
                }
                indices[axis] = 0;
            }
        }
    }
    Ok(())
}

/// Quantum-dimension-weighted Frobenius inner product over the stored
/// blocks: `sum_c dim(c) * <a_c, b_c>` with the first argument conjugated,
/// matching TensorKit's `dot` (which conjugates its first argument). Real
/// tensors produce an exactly-real result.
fn weighted_inner<R, D>(
    rule: &R,
    structure: &BlockStructure,
    nout: usize,
    a: &[D],
    b: &[D],
) -> Result<Complex64, Error>
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64>,
    D: UserScalar,
{
    // Abelian (UniqueFusion) fast path: every `dim(c) == 1`, and the coupled
    // buffer is a padding-free concatenation of the per-sector blocks, so the
    // weighted per-block sum collapses to one whole-buffer conjugated dot —
    // no `dim(c)` weights and no per-element odometer. Mirrors TensorKit's
    // `inner(tx.data, ty.data)` / `norm(t.data)` UniqueFusion specialization
    // (vectorinterface.jl:124, linalg.jl:277). Non-abelian keeps the weighted
    // block loop below, where `dim(c) != 1`.
    if rule.fusion_style() == tenet_core::FusionStyleKind::Unique {
        let mut total = D::from_real(0.0);
        for (&ai, &bi) in a.iter().zip(b) {
            total = total + FactorScalar::adjoint(ai) * bi;
        }
        return Ok(total.widen_complex());
    }
    coupled_region_inner(structure, nout, a, b, |coupled| rule.dim_scalar(coupled))
}

/// Generic-fusion (Stage B3c-1) sibling of [`weighted_inner`] for an
/// outer-multiplicity rule (SU(N)): `sum_c dim(c) * <a_c, b_c>`. Identical block
/// loop; the only difference is the coupled-sector weight. [`GenericRigidSymbols`]
/// exposes `sqrt_dim` rather than `dim`, and `dim(c) = sqrt_dim(c)^2` (an
/// integer for SU(N)) — so a Frobenius norm sums every multiplicity block
/// (e.g. both SU(3) `N(8,8,8)=2` vertices) weighted by the quantum dimension,
/// exactly matching TensorKit's `norm`. No Unique fast path: Generic is never
/// abelian.
fn weighted_inner_generic<R, D>(
    rule: &R,
    structure: &BlockStructure,
    nout: usize,
    a: &[D],
    b: &[D],
) -> Result<Complex64, Error>
where
    R: tenet_core::GenericRigidSymbols<Scalar = f64>,
    D: UserScalar,
{
    coupled_region_inner(structure, nout, a, b, |coupled| {
        let sqrt = rule.sqrt_dim_scalar(coupled);
        sqrt * sqrt
    })
}

/// Generic-fusion (Stage B3c-2) sibling of [`weighted_trace`] for an
/// outer-multiplicity rule (SU(N)): identical diagonal-block walk; the weight
/// is `dim(c) = sqrt_dim(c)²`.
/// A vertex-labelled (OM) block contributes only when its codomain and domain
/// trees coincide INCLUDING the vertex labels — off-diagonal vertex pairs are
/// off the coupled-block diagonal exactly like any other tree mismatch.
fn weighted_trace_generic<R, D>(
    rule: &R,
    structure: &BlockStructure,
    nout: usize,
    data: &[D],
) -> Result<Complex64, Error>
where
    R: tenet_core::GenericRigidSymbols<Scalar = f64>,
    D: UserScalar,
{
    let mut total = Complex64::new(0.0, 0.0);
    for index in 0..structure.block_count() {
        let block = structure.block(index)?;
        let key = match block.key() {
            BlockKey::FusionTree(key) => key,
            _ => {
                return Err(Error::InvalidArgument(
                    "tr() requires fusion-tree blocks".to_string(),
                ))
            }
        };
        if key.codomain_tree() != key.domain_tree() {
            continue;
        }
        let coupled = key.codomain_tree().coupled();
        let sqrt = rule.sqrt_dim_scalar(coupled);
        let weight = sqrt * sqrt;
        let shape = block.shape();
        let strides = block.strides();
        let offset = block.offset();
        let count: usize = shape[..nout].iter().product();
        let mut partial = D::from_real(0.0);
        for linear in 0..count {
            let mut remainder = linear;
            let mut position = offset;
            for axis in 0..nout {
                let coordinate = remainder % shape[axis];
                remainder /= shape[axis];
                position += coordinate * (strides[axis] + strides[nout + axis]);
            }
            partial = partial + data[position];
        }
        total += partial.widen_complex() * weight;
    }
    Ok(total)
}

/// Quantum-dimension-weighted block trace of an endomorphism:
/// `sum_c dim(c) * tr(b_c)`, matching TensorKit's `tr` (`linalg.jl`, the
/// native `sum_c dim(c) * tr(block)`). Only fusion-tree blocks whose codomain
/// and domain trees coincide sit on the coupled-block diagonal; every other
/// block is off-diagonal in `b_c` and contributes nothing. Within a diagonal
/// block the trace pairs codomain degeneracy axis `i` with domain axis
/// `nout + i` (equal degeneracies, since the spaces match). Real tensors give
/// an exactly-real result. Fermionic twists belong to `trace_pairs` / tensor
/// contractions and are not part of this matrix trace.
fn weighted_trace<R, D>(
    rule: &R,
    structure: &BlockStructure,
    nout: usize,
    data: &[D],
) -> Result<Complex64, Error>
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64>,
    D: UserScalar,
{
    let mut total = Complex64::new(0.0, 0.0);
    for index in 0..structure.block_count() {
        let block = structure.block(index)?;
        let key = match block.key() {
            BlockKey::FusionTree(key) => key,
            _ => {
                return Err(Error::InvalidArgument(
                    "tr() requires fusion-tree blocks".to_string(),
                ))
            }
        };
        // Off the coupled-block diagonal (codomain tree != domain tree): not on
        // the matrix diagonal of b_c, so it drops out of tr(b_c).
        if key.codomain_tree() != key.domain_tree() {
            continue;
        }
        let coupled = key.codomain_tree().coupled();
        let weight = rule.dim_scalar(coupled);
        let shape = block.shape();
        let strides = block.strides();
        let offset = block.offset();
        // Walk only the codomain degeneracy multi-index and index both halves
        // diagonally (axis i and axis nout+i share the index) — the degeneracy
        // trace of this coupled sub-block.
        let count: usize = shape[..nout].iter().product();
        let mut partial = D::from_real(0.0);
        for linear in 0..count {
            let mut remainder = linear;
            let mut position = offset;
            for axis in 0..nout {
                let coordinate = remainder % shape[axis];
                remainder /= shape[axis];
                position += coordinate * (strides[axis] + strides[nout + axis]);
            }
            partial = partial + data[position];
        }
        total += partial.widen_complex() * weight;
    }
    Ok(total)
}

/// Which tree transform a leg re-arrangement uses.
enum TransformKind<'a> {
    Permute,
    Braid { levels: &'a [usize] },
    Transpose,
}

struct LoweredAdjointTransformRequest {
    codomain_axes: Vec<usize>,
    domain_axes: Vec<usize>,
    levels: Vec<usize>,
}

fn lower_adjoint_transform_request(
    parent_codomain_rank: usize,
    parent_domain_rank: usize,
    logical_codomain_axes: &[usize],
    logical_domain_axes: &[usize],
    kind: &TransformKind<'_>,
) -> Result<LoweredAdjointTransformRequest, Error> {
    let rank = parent_codomain_rank
        .checked_add(parent_domain_rank)
        .ok_or_else(|| Error::InvalidArgument("tensor rank overflow".to_string()))?;
    let logical_axes = logical_codomain_axes
        .iter()
        .chain(logical_domain_axes)
        .copied()
        .collect::<Vec<_>>();
    validate_axis_permutation(&logical_axes, rank)?;

    let logical_to_parent = |axis: usize| {
        if axis < parent_domain_rank {
            parent_codomain_rank + axis
        } else {
            axis - parent_domain_rank
        }
    };
    let codomain_axes = logical_domain_axes
        .iter()
        .copied()
        .map(logical_to_parent)
        .collect();
    let domain_axes = logical_codomain_axes
        .iter()
        .copied()
        .map(logical_to_parent)
        .collect();
    let levels = match kind {
        TransformKind::Braid { levels } => {
            // Why not reflect the level values: the outer adjoint conjugates
            // coefficients; TensorKit only reindexes levels into parent order.
            levels[parent_domain_rank..]
                .iter()
                .chain(&levels[..parent_domain_rank])
                .copied()
                .collect()
        }
        TransformKind::Permute | TransformKind::Transpose => Vec::new(),
    };

    Ok(LoweredAdjointTransformRequest {
        codomain_axes,
        domain_axes,
        levels,
    })
}

fn validate_contracted_axes(contracted: &[usize], rank: usize) -> Result<(), Error> {
    let mut seen = SmallVec::<[bool; 16]>::new();
    seen.resize(rank, false);
    for &axis in contracted {
        if axis >= rank || seen[axis] {
            return Err(Error::InvalidArgument(format!(
                "invalid contracted axis list {contracted:?} for rank {rank}"
            )));
        }
        seen[axis] = true;
    }
    Ok(())
}

fn validate_axis_permutation(axes: &[usize], rank: usize) -> Result<(), Error> {
    if axes.len() == rank && validate_contracted_axes(axes, rank).is_ok() {
        return Ok(());
    }
    Err(
        OperationError::Core(tenet_core::CoreError::InvalidPermutation {
            permutation: axes.to_vec(),
            rank,
        })
        .into(),
    )
}

// ---------------------------------------------------------------------------
// Public tensor type.
// ---------------------------------------------------------------------------

/// A block-sparse symmetric tensor with dynamic rank, tied to a [`Runtime`].
///
/// `Tensor` is the user-layer face of the expert layer's dynamic-rank
/// machinery: the fusion rule (U1 / Z2 / fZ2 / SU2 / U1 x fZ2) is fixed per
/// tensor by the [`Space`]s it was built from, and the codomain/domain split
/// is a runtime property with no rank ceiling. Mixing tensors of different
/// rules or different runtimes in one operation is an error.
///
/// Scalar type: each tensor stores either real `f64` or complex
/// [`Complex64`] data, fixed at construction (the [`Dtype`] token of
/// [`Self::rand`], [`Self::zeros`] and so on; [`Self::from_block_fn`]
/// infers it from the fill closure) and reported by [`Self::dtype`].
/// Operations dispatch on the stored dtype internally; mixing dtypes in one
/// operation is [`Error::DtypeMismatch`] (widen explicitly with
/// [`Self::to_c64`]).
///
/// # Examples
///
/// ```
/// use tenet::prelude::*;
///
/// let rt = Runtime::builder().build()?;
/// let v = Space::z2([(0, 1), (1, 1)]);
///
/// // Same numbers as the tutorial's expert-layer Z2 example.
/// let a = Tensor::from_block_fn(&rt, [&v], [&v], |key, _| match key {
///     BlockKey::FusionTree(key) if key.codomain_uncoupled()[0].id() == 0 => 2.0,
///     _ => 3.0,
/// })?;
/// let b = Tensor::from_block_fn(&rt, [&v], [&v], |key, _| match key {
///     BlockKey::FusionTree(key) if key.codomain_uncoupled()[0].id() == 0 => 5.0,
///     _ => 7.0,
/// })?;
/// let c = a.compose(&b)?;
/// assert_eq!(c.data(), &[10.0, 21.0]);
/// # Ok::<(), tenet::prelude::Error>(())
/// ```
#[derive(Debug)]
enum UserBoundSpace {
    U1(BoundDynamicFusionMapSpace<tenet_core::U1FusionRule>),
    Z2(BoundDynamicFusionMapSpace<tenet_core::Z2FusionRule>),
    FZ2(BoundDynamicFusionMapSpace<tenet_core::FermionParityFusionRule>),
    SU2(BoundDynamicFusionMapSpace<tenet_core::SU2FusionRule>),
    U1FZ2(BoundDynamicFusionMapSpace<U1Fz2Rule>),
    FZ2U1SU2(BoundDynamicFusionMapSpace<Fz2U1Su2Rule>),
    Su3(BoundDynamicFusionMapSpace<Su3FusionRule>),
}

trait IntoUserBoundDynamicSpace: FusionRule + Sized {
    fn into_user_bound(
        expected: &UserBoundSpace,
        bound: BoundDynamicFusionMapSpace<Self>,
    ) -> Result<UserBoundSpace, Error>;
}

macro_rules! impl_into_user_bound {
    ($rule:ty, $context:ident, $inner:ident) => {
        impl IntoUserBoundDynamicSpace for $rule {
            fn into_user_bound(
                expected: &UserBoundSpace,
                bound: BoundDynamicFusionMapSpace<Self>,
            ) -> Result<UserBoundSpace, Error> {
                let UserBoundSpace::$inner(existing) = expected else {
                    return Err(Error::InvalidArgument(
                        "SVD factor provider type does not match tensor context".to_string(),
                    ));
                };
                if !Arc::ptr_eq(existing.provider_arc(), bound.provider_arc())
                    || existing.provider().rule_identity() != bound.provider().rule_identity()
                {
                    return Err(Error::InvalidArgument(
                        "SVD factor provider identity does not match tensor context".to_string(),
                    ));
                }
                Ok(UserBoundSpace::$inner(bound))
            }
        }
    };
}

impl_into_user_bound!(tenet_core::U1FusionRule, U1, U1);
impl_into_user_bound!(tenet_core::Z2FusionRule, Z2, Z2);
impl_into_user_bound!(tenet_core::FermionParityFusionRule, FZ2, FZ2);
impl_into_user_bound!(tenet_core::SU2FusionRule, SU2, SU2);
impl_into_user_bound!(U1Fz2Rule, U1FZ2, U1FZ2);
impl_into_user_bound!(Fz2U1Su2Rule, FZ2U1SU2, FZ2U1SU2);
impl_into_user_bound!(Su3FusionRule, Su3, Su3);

impl PartialEq for UserBoundSpace {
    fn eq(&self, other: &Self) -> bool {
        self.identity() == other.identity() && self.raw() == other.raw()
    }
}

impl Eq for UserBoundSpace {}

impl UserBoundSpace {
    fn from_bound<R>(expected: &Self, bound: BoundDynamicFusionMapSpace<R>) -> Result<Self, Error>
    where
        R: IntoUserBoundDynamicSpace,
    {
        R::into_user_bound(expected, bound)
    }

    fn contracted(
        &self,
        rhs: &Self,
        lhs_axes: &[usize],
        rhs_axes: &[usize],
    ) -> Result<Self, Error> {
        macro_rules! contract {
            ($lhs:expr, $rhs:expr, $variant:ident, $method:ident) => {
                Ok(UserBoundSpace::$variant(
                    BoundDynamicFusionMapSpace::$method($lhs, $rhs, lhs_axes, rhs_axes)?,
                ))
            };
        }
        match (self, rhs) {
            (Self::U1(lhs), Self::U1(rhs)) => {
                contract!(lhs, rhs, U1, contracted_multiplicity_free_lowered)
            }
            (Self::Z2(lhs), Self::Z2(rhs)) => {
                contract!(lhs, rhs, Z2, contracted_multiplicity_free_lowered)
            }
            (Self::FZ2(lhs), Self::FZ2(rhs)) => {
                contract!(lhs, rhs, FZ2, contracted_multiplicity_free_lowered)
            }
            (Self::SU2(lhs), Self::SU2(rhs)) => {
                contract!(lhs, rhs, SU2, contracted_multiplicity_free_lowered)
            }
            (Self::U1FZ2(lhs), Self::U1FZ2(rhs)) => {
                contract!(lhs, rhs, U1FZ2, contracted_multiplicity_free_lowered)
            }
            (Self::FZ2U1SU2(lhs), Self::FZ2U1SU2(rhs)) => {
                contract!(lhs, rhs, FZ2U1SU2, contracted_multiplicity_free_lowered)
            }
            (Self::Su3(lhs), Self::Su3(rhs)) => {
                contract!(lhs, rhs, Su3, contracted_generic)
            }
            _ => Err(Error::RuleMismatch),
        }
    }

    fn contracted_with_output_order(
        &self,
        rhs: &Self,
        lhs_axes: &[usize],
        rhs_axes: &[usize],
        output_order: OutputAxisOrder<'_>,
    ) -> Result<Self, Error> {
        let OutputAxisOrder::Axes(output_axes) = output_order else {
            return self.contracted(rhs, lhs_axes, rhs_axes);
        };

        // The Generic facade stays on its separately proved sequential path.
        if matches!((self, rhs), (Self::Su3(_), Self::Su3(_))) {
            let default = self.contracted(rhs, lhs_axes, rhs_axes)?;
            validate_axis_permutation(output_axes, default.raw().rank())?;
            let split = default.raw().nout();
            return default.transformed(&TreeTransformOperation::permute(
                output_axes[..split].iter().copied(),
                output_axes[split..].iter().copied(),
            ));
        }

        let output_rank = match self
            .raw()
            .rank()
            .checked_sub(lhs_axes.len())
            .and_then(|lhs_open| {
                rhs.raw()
                    .rank()
                    .checked_sub(rhs_axes.len())
                    .and_then(|rhs_open| lhs_open.checked_add(rhs_open))
            }) {
            Some(rank) => rank,
            None => {
                self.validate_contracted_homspace(rhs, lhs_axes, rhs_axes)?;
                return Err(Error::InvalidArgument(
                    "contracted axis count exceeds tensor rank".to_string(),
                ));
            }
        };
        if let Err(output_error) = validate_axis_permutation(output_axes, output_rank) {
            // Preserve historical contraction-before-pAB error precedence
            // without building the default coupled layout. Valid pAB skips
            // this cold check and reaches the cache lookup immediately.
            self.validate_contracted_homspace(rhs, lhs_axes, rhs_axes)?;
            return Err(output_error);
        }
        macro_rules! contract {
            ($lhs:expr, $rhs:expr, $variant:ident) => {
                Ok(UserBoundSpace::$variant(
                    BoundDynamicFusionMapSpace::contracted_multiplicity_free_ordered_lowered(
                        $lhs,
                        $rhs,
                        lhs_axes,
                        rhs_axes,
                        OutputAxisOrder::from_axes(output_axes),
                    )?,
                ))
            };
        }
        match (self, rhs) {
            (Self::U1(lhs), Self::U1(rhs)) => contract!(lhs, rhs, U1),
            (Self::Z2(lhs), Self::Z2(rhs)) => contract!(lhs, rhs, Z2),
            (Self::FZ2(lhs), Self::FZ2(rhs)) => contract!(lhs, rhs, FZ2),
            (Self::SU2(lhs), Self::SU2(rhs)) => contract!(lhs, rhs, SU2),
            (Self::U1FZ2(lhs), Self::U1FZ2(rhs)) => contract!(lhs, rhs, U1FZ2),
            (Self::FZ2U1SU2(lhs), Self::FZ2U1SU2(rhs)) => {
                contract!(lhs, rhs, FZ2U1SU2)
            }
            _ => Err(Error::RuleMismatch),
        }
    }

    fn validate_contracted_homspace(
        &self,
        rhs: &Self,
        lhs_axes: &[usize],
        rhs_axes: &[usize],
    ) -> Result<(), Error> {
        macro_rules! validate {
            ($lhs:expr, $rhs:expr) => {
                BoundDynamicFusionMapSpace::validate_contracted_homspace_multiplicity_free(
                    $lhs, $rhs, lhs_axes, rhs_axes,
                )
                .map_err(Into::into)
            };
        }
        match (self, rhs) {
            (Self::U1(lhs), Self::U1(rhs)) => validate!(lhs, rhs),
            (Self::Z2(lhs), Self::Z2(rhs)) => validate!(lhs, rhs),
            (Self::FZ2(lhs), Self::FZ2(rhs)) => validate!(lhs, rhs),
            (Self::SU2(lhs), Self::SU2(rhs)) => validate!(lhs, rhs),
            (Self::U1FZ2(lhs), Self::U1FZ2(rhs)) => validate!(lhs, rhs),
            (Self::FZ2U1SU2(lhs), Self::FZ2U1SU2(rhs)) => validate!(lhs, rhs),
            _ => Err(Error::RuleMismatch),
        }
    }

    fn transformed(&self, operation: &TreeTransformOperation) -> Result<Self, Error> {
        macro_rules! transform {
            ($space:expr, $variant:ident, $method:ident) => {
                Ok(UserBoundSpace::$variant($space.$method(operation)?))
            };
        }
        match self {
            Self::U1(space) => transform!(space, U1, transformed_multiplicity_free_lowered),
            Self::Z2(space) => transform!(space, Z2, transformed_multiplicity_free_lowered),
            Self::FZ2(space) => transform!(space, FZ2, transformed_multiplicity_free_lowered),
            Self::SU2(space) => transform!(space, SU2, transformed_multiplicity_free_lowered),
            Self::U1FZ2(space) => transform!(space, U1FZ2, transformed_multiplicity_free_lowered),
            Self::FZ2U1SU2(space) => {
                transform!(space, FZ2U1SU2, transformed_multiplicity_free_lowered)
            }
            Self::Su3(space) => transform!(space, Su3, transformed_generic),
        }
    }

    fn adjoint_space(&self) -> Result<Self, Error> {
        macro_rules! adjoint {
            ($space:expr, $variant:ident, $function:ident) => {
                Ok(UserBoundSpace::$variant(tenet_tensors::$function($space)?))
            };
        }
        match self {
            Self::U1(space) => adjoint!(space, U1, adjoint_bound_space_dyn_lowered),
            Self::Z2(space) => adjoint!(space, Z2, adjoint_bound_space_dyn_lowered),
            Self::FZ2(space) => adjoint!(space, FZ2, adjoint_bound_space_dyn_lowered),
            Self::SU2(space) => adjoint!(space, SU2, adjoint_bound_space_dyn_lowered),
            Self::U1FZ2(space) => adjoint!(space, U1FZ2, adjoint_bound_space_dyn_lowered),
            Self::FZ2U1SU2(space) => {
                adjoint!(space, FZ2U1SU2, adjoint_bound_space_dyn_lowered)
            }
            Self::Su3(space) => adjoint!(space, Su3, adjoint_bound_space_dyn_generic),
        }
    }

    fn from_homspace(&self, homspace: FusionTreeHomSpace) -> Result<Self, Error> {
        macro_rules! build {
            ($space:expr, $variant:ident) => {
                Ok(UserBoundSpace::$variant(build_bound_space_like(
                    $space, homspace,
                )?))
            };
        }
        match self {
            Self::U1(space) => build!(space, U1),
            Self::Z2(space) => build!(space, Z2),
            Self::FZ2(space) => build!(space, FZ2),
            Self::SU2(space) => build!(space, SU2),
            Self::U1FZ2(space) => build!(space, U1FZ2),
            Self::FZ2U1SU2(space) => build!(space, FZ2U1SU2),
            Self::Su3(space) => Ok(UserBoundSpace::Su3(build_bound_space_generic(
                Arc::clone(space.provider_arc()),
                homspace,
            )?)),
        }
    }

    fn from_selected_homspace(&self, homspace: FusionTreeHomSpace) -> Result<Self, Error> {
        #[cfg(test)]
        observe_selected_result_layout_build();
        macro_rules! build {
            ($space:expr, $variant:ident) => {
                Ok(UserBoundSpace::$variant(
                    $space.derive_from_final_homspace(homspace)?,
                ))
            };
        }
        match self {
            Self::U1(space) => build!(space, U1),
            Self::Z2(space) => build!(space, Z2),
            Self::FZ2(space) => build!(space, FZ2),
            Self::SU2(space) => build!(space, SU2),
            Self::U1FZ2(space) => build!(space, U1FZ2),
            Self::FZ2U1SU2(space) => build!(space, FZ2U1SU2),
            Self::Su3(space) => Ok(UserBoundSpace::Su3(
                BoundDynamicFusionMapSpace::from_final_homspace_generic(
                    Arc::clone(space.provider_arc()),
                    homspace,
                )?,
            )),
        }
    }

    fn raw(&self) -> &DynamicFusionMapSpace {
        match self {
            UserBoundSpace::U1(space) => space.space(),
            UserBoundSpace::Z2(space) => space.space(),
            UserBoundSpace::FZ2(space) => space.space(),
            UserBoundSpace::SU2(space) => space.space(),
            UserBoundSpace::U1FZ2(space) => space.space(),
            UserBoundSpace::FZ2U1SU2(space) => space.space(),
            UserBoundSpace::Su3(space) => space.space(),
        }
    }

    fn context(&self) -> UserRuleContext {
        match self {
            UserBoundSpace::U1(space) => UserRuleContext::U1(Arc::clone(space.provider_arc())),
            UserBoundSpace::Z2(space) => UserRuleContext::Z2(Arc::clone(space.provider_arc())),
            UserBoundSpace::FZ2(space) => UserRuleContext::FZ2(Arc::clone(space.provider_arc())),
            UserBoundSpace::SU2(space) => UserRuleContext::SU2(Arc::clone(space.provider_arc())),
            UserBoundSpace::U1FZ2(space) => {
                UserRuleContext::U1FZ2(Arc::clone(space.provider_arc()))
            }
            UserBoundSpace::FZ2U1SU2(space) => {
                UserRuleContext::FZ2U1SU2(Arc::clone(space.provider_arc()))
            }
            UserBoundSpace::Su3(space) => UserRuleContext::Su3(Arc::clone(space.provider_arc())),
        }
    }

    fn kind(&self) -> RuleKind {
        match self {
            UserBoundSpace::U1(_) => RuleKind::U1,
            UserBoundSpace::Z2(_) => RuleKind::Z2,
            UserBoundSpace::FZ2(_) => RuleKind::FZ2,
            UserBoundSpace::SU2(_) => RuleKind::SU2,
            UserBoundSpace::U1FZ2(_) => RuleKind::U1FZ2,
            UserBoundSpace::FZ2U1SU2(_) => RuleKind::FZ2U1SU2,
            UserBoundSpace::Su3(_) => RuleKind::Su3,
        }
    }

    fn identity(&self) -> tenet_core::RuleIdentity {
        match self {
            UserBoundSpace::U1(space) => space.provider().rule_identity(),
            UserBoundSpace::Z2(space) => space.provider().rule_identity(),
            UserBoundSpace::FZ2(space) => space.provider().rule_identity(),
            UserBoundSpace::SU2(space) => space.provider().rule_identity(),
            UserBoundSpace::U1FZ2(space) => space.provider().rule_identity(),
            UserBoundSpace::FZ2U1SU2(space) => space.provider().rule_identity(),
            UserBoundSpace::Su3(space) => space.provider().rule_identity(),
        }
    }

    #[cfg(test)]
    fn provider_matches_context_allocation(&self, context: &UserRuleContext) -> bool {
        match (self, context) {
            (Self::U1(space), UserRuleContext::U1(provider)) => {
                Arc::ptr_eq(space.provider_arc(), provider)
            }
            (Self::Z2(space), UserRuleContext::Z2(provider)) => {
                Arc::ptr_eq(space.provider_arc(), provider)
            }
            (Self::FZ2(space), UserRuleContext::FZ2(provider)) => {
                Arc::ptr_eq(space.provider_arc(), provider)
            }
            (Self::SU2(space), UserRuleContext::SU2(provider)) => {
                Arc::ptr_eq(space.provider_arc(), provider)
            }
            (Self::U1FZ2(space), UserRuleContext::U1FZ2(provider)) => {
                Arc::ptr_eq(space.provider_arc(), provider)
            }
            (Self::FZ2U1SU2(space), UserRuleContext::FZ2U1SU2(provider)) => {
                Arc::ptr_eq(space.provider_arc(), provider)
            }
            (Self::Su3(space), UserRuleContext::Su3(provider)) => {
                Arc::ptr_eq(space.provider_arc(), provider)
            }
            _ => false,
        }
    }
}

impl std::ops::Deref for UserBoundSpace {
    type Target = DynamicFusionMapSpace;

    fn deref(&self) -> &Self::Target {
        self.raw()
    }
}

macro_rules! with_bound_multiplicity_free {
    ($space:expr, $bound:ident, $body:expr) => {
        match $space.as_ref() {
            UserBoundSpace::U1($bound) => $body,
            UserBoundSpace::Z2($bound) => $body,
            UserBoundSpace::FZ2($bound) => $body,
            UserBoundSpace::SU2($bound) => $body,
            UserBoundSpace::U1FZ2($bound) => $body,
            UserBoundSpace::FZ2U1SU2($bound) => $body,
            UserBoundSpace::Su3(_) => {
                unreachable!("generic provider uses the dedicated SVD path")
            }
        }
    };
}

/// Static dispatch from the tensor's sole bound authority. Why not rebuild a
/// `UserRuleContext`: ordinary operations only need a provider borrow, and an
/// enum reconstruction plus Arc refcount traffic would make the hot path pay
/// for a user-facing `Space` view it never creates.
macro_rules! with_user_rule {
    ($space:expr, $rule:ident, $body:expr) => {
        match $space.as_ref() {
            UserBoundSpace::U1(bound) => {
                let $rule = bound.provider();
                $body
            }
            UserBoundSpace::Z2(bound) => {
                let $rule = bound.provider();
                $body
            }
            UserBoundSpace::FZ2(bound) => {
                let $rule = bound.provider();
                $body
            }
            UserBoundSpace::SU2(bound) => {
                let $rule = bound.provider();
                $body
            }
            UserBoundSpace::U1FZ2(bound) => {
                let $rule = bound.provider();
                $body
            }
            UserBoundSpace::FZ2U1SU2(bound) => {
                let $rule = bound.provider();
                $body
            }
            UserBoundSpace::Su3(_) => {
                unreachable!("generic provider requires a dedicated operation path")
            }
        }
    };
}

macro_rules! with_user_rule_ctx {
    ($space:expr, $state:expr, $rule:ident, $ctxs:ident, $body:expr) => {
        match $space.as_ref() {
            UserBoundSpace::U1(bound) => {
                let $rule = bound.provider();
                let $ctxs = &mut $state.u1;
                $body
            }
            UserBoundSpace::Z2(bound) => {
                let $rule = bound.provider();
                let $ctxs = &mut $state.z2;
                $body
            }
            UserBoundSpace::FZ2(bound) => {
                let $rule = bound.provider();
                let $ctxs = &mut $state.fz2;
                $body
            }
            UserBoundSpace::SU2(bound) => {
                let $rule = bound.provider();
                let $ctxs = &mut $state.su2;
                $body
            }
            UserBoundSpace::U1FZ2(bound) => {
                let $rule = bound.provider();
                let $ctxs = &mut $state.u1_fz2;
                $body
            }
            UserBoundSpace::FZ2U1SU2(bound) => {
                let $rule = bound.provider();
                let $ctxs = &mut $state.fz2_u1_su2;
                $body
            }
            UserBoundSpace::Su3(_) => {
                unreachable!("generic provider requires a dedicated operation path")
            }
        }
    };
}

macro_rules! with_bound_ctx {
    ($space:expr, $state:expr, $bound:ident, $ctxs:ident, $body:expr) => {
        match $space.as_ref() {
            UserBoundSpace::U1($bound) => {
                let $ctxs = &mut $state.u1;
                $body
            }
            UserBoundSpace::Z2($bound) => {
                let $ctxs = &mut $state.z2;
                $body
            }
            UserBoundSpace::FZ2($bound) => {
                let $ctxs = &mut $state.fz2;
                $body
            }
            UserBoundSpace::SU2($bound) => {
                let $ctxs = &mut $state.su2;
                $body
            }
            UserBoundSpace::U1FZ2($bound) => {
                let $ctxs = &mut $state.u1_fz2;
                $body
            }
            UserBoundSpace::FZ2U1SU2($bound) => {
                let $ctxs = &mut $state.fz2_u1_su2;
                $body
            }
            UserBoundSpace::Su3(_) => unreachable!("generic provider is unsupported"),
        }
    };
}

#[derive(Clone, Debug)]
struct TensorBody {
    space: Arc<UserBoundSpace>,
    data: Arc<Data>,
}

#[derive(Debug)]
struct AdjointView {
    // Why not retain an adjoint space here: deriving its block grid is the
    // O(blocks) work this view exists to defer. The parent remains the sole
    // semantic authority; logical_space is only its reproducible derived view.
    parent: TensorBody,
    logical_space: OnceLock<Arc<UserBoundSpace>>,
    materialized: OnceLock<Arc<TensorBody>>,
    // Why not rely on OnceLock::set races: losing builders would still repeat
    // the expensive block-grid/data work before publication.
    init: Mutex<()>,
    #[cfg(test)]
    logical_space_builds: std::sync::atomic::AtomicUsize,
    #[cfg(test)]
    materialized_body_builds: std::sync::atomic::AtomicUsize,
}

#[derive(Debug)]
enum TensorRepr {
    Owned(TensorBody),
    Adjoint(Arc<AdjointView>),
}

#[derive(Clone, Copy)]
enum TensorOrientation {
    Owned,
    Adjoint,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ContractionSemantics {
    TensorContract,
    Composition,
}

struct TensorMetadataView<'a> {
    body: &'a TensorBody,
    orientation: TensorOrientation,
}

impl TensorMetadataView<'_> {
    fn codomain(&self) -> &FusionProductSpace {
        match self.orientation {
            TensorOrientation::Owned => self.body.space.homspace().codomain(),
            TensorOrientation::Adjoint => self.body.space.homspace().domain(),
        }
    }

    fn domain(&self) -> &FusionProductSpace {
        match self.orientation {
            TensorOrientation::Owned => self.body.space.homspace().domain(),
            TensorOrientation::Adjoint => self.body.space.homspace().codomain(),
        }
    }

    fn nout(&self) -> usize {
        self.codomain().len()
    }

    fn nin(&self) -> usize {
        self.domain().len()
    }

    fn rank(&self) -> usize {
        self.nout() + self.nin()
    }
}

/// A block-sparse symmetric tensor map `codomain <- domain` with dynamic rank,
/// carrying its [`Runtime`] and a rule-erased fusion space. This is the
/// everyday user-layer type; see the crate-level docs for the execution model.
#[derive(Debug)]
pub struct Tensor {
    rt: Runtime,
    repr: TensorRepr,
    compact_dense: OnceLock<Arc<Data>>,
}

/// Opaque tensor authority parked without owning its runtime.
#[doc(hidden)]
pub struct RuntimeDetachedTensor {
    // Why not store `Runtime`: idle plan-cache buffers would point back to the
    // cache owner and keep the entire Runtime alive.
    runtime: RuntimeIdentity,
    repr: TensorRepr,
    compact_dense: OnceLock<Arc<Data>>,
}

impl Clone for Tensor {
    fn clone(&self) -> Self {
        let compact_dense = OnceLock::new();
        if let Some(data) = self.compact_dense.get() {
            let _ = compact_dense.set(Arc::clone(data));
        }
        let repr = match &self.repr {
            TensorRepr::Owned(body) => TensorRepr::Owned(body.clone()),
            TensorRepr::Adjoint(view) => TensorRepr::Adjoint(Arc::clone(view)),
        };
        Self {
            rt: self.rt.clone(),
            repr,
            compact_dense,
        }
    }
}

impl Tensor {
    fn owned(rt: Runtime, space: Arc<UserBoundSpace>, data: Arc<Data>) -> Self {
        Self {
            rt,
            repr: TensorRepr::Owned(TensorBody { space, data }),
            compact_dense: OnceLock::new(),
        }
    }

    fn metadata(&self) -> TensorMetadataView<'_> {
        match &self.repr {
            TensorRepr::Owned(body) => TensorMetadataView {
                body,
                orientation: TensorOrientation::Owned,
            },
            TensorRepr::Adjoint(view) => TensorMetadataView {
                body: &view.parent,
                orientation: TensorOrientation::Adjoint,
            },
        }
    }

    fn parent_body_for_lowering(&self) -> &TensorBody {
        match &self.repr {
            TensorRepr::Owned(body) => body,
            TensorRepr::Adjoint(view) => &view.parent,
        }
    }

    fn ordinary_body(&self) -> &TensorBody {
        // Why not return the adjoint parent: its space and bytes describe a
        // different tensor, which is exactly the incoherent pair this split
        // prevents from reaching general consumers.
        match &self.repr {
            TensorRepr::Owned(body) => body,
            TensorRepr::Adjoint(_) => {
                panic!("internal: an adjoint view reached an owned-only consumer")
            }
        }
    }

    fn stored_data(&self) -> &Data {
        // Dtype, placement, and view-native lowering inspect physical storage;
        // they do not interpret it in the logical adjoint block layout.
        self.parent_body_for_lowering().data.as_ref()
    }

    fn stored_data_arc(&self) -> &Arc<Data> {
        &self.parent_body_for_lowering().data
    }

    fn rule_authority_space(&self) -> &Arc<UserBoundSpace> {
        // Fusion-rule/provider identity is adjoint-invariant. Why not use this
        // for layouts: only logical_space/materialized_body may supply those.
        &self.parent_body_for_lowering().space
    }

    fn owned_body_mut(&mut self) -> Option<&mut TensorBody> {
        let TensorRepr::Owned(body) = &mut self.repr else {
            return None;
        };
        Some(body)
    }

    fn is_adjoint_view(&self) -> bool {
        matches!(self.repr, TensorRepr::Adjoint(_))
    }

    #[cfg(test)]
    fn has_cached_materialization(&self) -> bool {
        match &self.repr {
            TensorRepr::Owned(_) => self.compact_dense.get().is_some(),
            TensorRepr::Adjoint(view) => view.materialized.get().is_some(),
        }
    }

    #[cfg(test)]
    fn adjoint_build_counts(&self) -> (usize, usize) {
        let TensorRepr::Adjoint(view) = &self.repr else {
            return (0, 0);
        };
        (
            view.logical_space_builds
                .load(std::sync::atomic::Ordering::Relaxed),
            view.materialized_body_builds
                .load(std::sync::atomic::Ordering::Relaxed),
        )
    }

    #[doc(hidden)]
    pub fn detach_runtime(self) -> RuntimeDetachedTensor {
        let Self {
            rt,
            repr,
            compact_dense,
        } = self;
        RuntimeDetachedTensor {
            runtime: rt.identity(),
            repr,
            compact_dense,
        }
    }

    fn rule_kind(&self) -> RuleKind {
        self.rule_authority_space().kind()
    }

    fn rule_context(&self) -> UserRuleContext {
        self.rule_authority_space().context()
    }

    fn su3_rule(&self) -> &Su3FusionRule {
        match self.rule_authority_space().as_ref() {
            UserBoundSpace::Su3(space) => space.provider(),
            _ => unreachable!("SU(3) dispatch requires an SU(3) tensor context"),
        }
    }

    fn build<'a, C, D, S>(
        rt: &Runtime,
        codomain: C,
        domain: D,
        fill: Fill<'_, S>,
    ) -> Result<Self, Error>
    where
        C: IntoIterator<Item = &'a Space>,
        D: IntoIterator<Item = &'a Space>,
        S: UserScalar,
    {
        let codomain: Vec<&Space> = codomain.into_iter().collect();
        let domain: Vec<&Space> = domain.into_iter().collect();
        let mut spaces = codomain.iter().chain(domain.iter());
        let context = Arc::clone(
            spaces
                .next()
                .ok_or_else(|| {
                    Error::InvalidArgument(
                        "at least one leg is required to infer the fusion rule".to_string(),
                    )
                })?
                .rule_context(),
        );
        if spaces.any(|space| {
            !Arc::ptr_eq(space.rule_context(), &context)
                && space.rule_context().identity() != context.identity()
        }) {
            return Err(Error::RuleMismatch);
        }

        let hom = FusionTreeHomSpace::new(
            FusionProductSpace::new(codomain.iter().map(|space| space.sector_leg())),
            FusionProductSpace::new(domain.iter().map(|space| space.sector_leg())),
        );
        macro_rules! build {
            ($provider:expr, $variant:ident) => {{
                let bound = build_bound_space(Arc::clone($provider), hom)?;
                let data = S::lift(apply_fill(bound.space(), fill)?);
                Ok::<_, Error>((UserBoundSpace::$variant(bound), data))
            }};
        }
        let (space, data) = match context.as_ref() {
            UserRuleContext::U1(provider) => build!(provider, U1),
            UserRuleContext::Z2(provider) => build!(provider, Z2),
            UserRuleContext::FZ2(provider) => build!(provider, FZ2),
            UserRuleContext::SU2(provider) => build!(provider, SU2),
            UserRuleContext::U1FZ2(provider) => build!(provider, U1FZ2),
            UserRuleContext::FZ2U1SU2(provider) => build!(provider, FZ2U1SU2),
            UserRuleContext::Su3(provider) => {
                let bound = build_bound_space_generic(Arc::clone(provider), hom)?;
                let data = S::lift(apply_fill(bound.space(), fill)?);
                Ok((UserBoundSpace::Su3(bound), data))
            }
        }?;
        Ok(Self::owned(rt.clone(), Arc::new(space), Arc::new(data)))
    }

    /// Zero tensor of the given [`Dtype`] on `codomain <- domain`
    /// (TensorKit `zeros(Float64, W ← V)` / `zeros(ComplexF64, W ← V)`).
    /// All spaces must share one fusion rule.
    pub fn zeros<'a, C, D>(
        rt: &Runtime,
        dtype: Dtype,
        codomain: C,
        domain: D,
    ) -> Result<Self, Error>
    where
        C: IntoIterator<Item = &'a Space>,
        D: IntoIterator<Item = &'a Space>,
    {
        match dtype {
            Dtype::F64 => Self::build::<_, _, f64>(rt, codomain, domain, Fill::Zeros),
            Dtype::C64 => Self::build::<_, _, Complex64>(rt, codomain, domain, Fill::Zeros),
        }
    }

    /// Random tensor of the given [`Dtype`] on `codomain <- domain`
    /// (TensorKit `rand(Float64, W ← V)` / `rand(ComplexF64, W ← V)`):
    /// entries (real and imaginary parts for [`Dtype::C64`]) uniform in
    /// `[-1, 1)`.
    ///
    /// Deterministic per runtime: the n-th `rand`-family call on a given
    /// runtime always produces the same tensor. Use [`Self::rand_with_seed`]
    /// for an explicit stream.
    pub fn rand<'a, C, D>(rt: &Runtime, dtype: Dtype, codomain: C, domain: D) -> Result<Self, Error>
    where
        C: IntoIterator<Item = &'a Space>,
        D: IntoIterator<Item = &'a Space>,
    {
        Self::rand_with_seed(rt, dtype, codomain, domain, rt.next_rand_seed())
    }

    /// Random tensor with an explicit seed (splitmix64 stream, entries
    /// uniform in `[-1, 1)`).
    ///
    /// Reproducibility is defined for the same TeNeT version and tensor
    /// layout. The stream fills internal storage order, so a sector codec or
    /// block-layout migration can produce a different semantic tensor from
    /// the same seed. Cross-version fixtures should use
    /// [`Self::from_block_fn`] with semantic [`BlockKey`] labels.
    pub fn rand_with_seed<'a, C, D>(
        rt: &Runtime,
        dtype: Dtype,
        codomain: C,
        domain: D,
        seed: u64,
    ) -> Result<Self, Error>
    where
        C: IntoIterator<Item = &'a Space>,
        D: IntoIterator<Item = &'a Space>,
    {
        match dtype {
            Dtype::F64 => Self::build::<_, _, f64>(rt, codomain, domain, Fill::Rand(seed)),
            Dtype::C64 => Self::build::<_, _, Complex64>(rt, codomain, domain, Fill::Rand(seed)),
        }
    }

    /// Tensor filled block-by-block: `fill(key, indices)` is called for
    /// every element of every symmetry-allowed block, with `indices` local
    /// to the block (degeneracy coordinates, codomain axes first). Mirrors
    /// [`tenet_core::TensorMap::from_block_fn_with_fusion_space`].
    ///
    /// The constructed dtype follows the closure's return type (`f64` or
    /// [`Complex64`], the two [`TensorScalar`] impls) — no dtype token
    /// needed.
    ///
    /// The fusion-tree `key` labels domain legs with the domain `Space`'s
    /// own sectors (TensorKit's `f2.uncoupled`), not their duals; on both
    /// sides the uncoupled sectors fuse to the tree's coupled sector.
    pub fn from_block_fn<'a, C, D, S, F>(
        rt: &Runtime,
        codomain: C,
        domain: D,
        mut fill: F,
    ) -> Result<Self, Error>
    where
        C: IntoIterator<Item = &'a Space>,
        D: IntoIterator<Item = &'a Space>,
        S: TensorScalar,
        F: FnMut(&BlockKey, &[usize]) -> S,
    {
        Self::build(rt, codomain, domain, Fill::BlockFn(&mut fill))
    }

    /// Shared core of [`Self::id`] / [`Self::isomorphism`] /
    /// [`Self::isometry`]: checks that the domain fits in the codomain
    /// (exactly for `embed == false`, isometric embedding for
    /// `embed == true`) and fills every coupled-sector matrix with the
    /// (partial) identity, exactly TensorKit's `one!` per coupled block
    /// (`tensors/linalg.jl:102-158`).
    fn structural(
        rt: &Runtime,
        dtype: Dtype,
        codomain: Vec<&Space>,
        domain: Vec<&Space>,
        embed: bool,
        what: &str,
    ) -> Result<Self, Error> {
        let fused_codomain = Space::fuse_all(&codomain)?;
        let fused_domain = Space::fuse_all(&domain)?;
        let fits = if embed {
            // TensorKit `domain ≾ codomain`: sectorwise embeddable.
            fused_domain.sectors.iter().all(|&(sector, deg)| {
                fused_codomain
                    .sectors
                    .iter()
                    .any(|&(s, d)| s == sector && d >= deg)
            })
        } else {
            // TensorKit `domain ≅ codomain`: identical fused sector content.
            fused_codomain.sectors == fused_domain.sectors
        };
        if !fits {
            return Err(Error::InvalidArgument(format!(
                "{what}: codomain and domain are not {} (fused sector content differs)",
                if embed {
                    "isometrically embeddable"
                } else {
                    "isomorphic"
                }
            )));
        }
        let mut t = Self::build::<_, _, f64>(rt, codomain, domain, Fill::Zeros)?;
        let regions = sector_regions(
            t.ordinary_body().space.structure(),
            t.ordinary_body().space.nout(),
        )?;
        let body = t.owned_body_mut().ok_or_else(|| {
            Error::InvalidArgument(
                "fresh structural tensor unexpectedly shares its owned authority".to_string(),
            )
        })?;
        let Data::F64(data) = Arc::make_mut(&mut body.data) else {
            unreachable!("structural constructors build f64 host tensors");
        };
        for region in regions.iter() {
            for i in 0..region.rows().min(region.cols()) {
                data[region.range().start + i * (region.rows() + 1)] = 1.0;
            }
        }
        Ok(match dtype {
            Dtype::F64 => t,
            Dtype::C64 => t.to_c64(),
        })
    }

    /// The identity endomorphism on `spaces <- spaces` (TensorKit `id(V)`,
    /// `tensors/linalg.jl:75-82`): every coupled-sector block is the
    /// identity matrix.
    ///
    /// ```
    /// use tenet::prelude::*;
    ///
    /// let rt = Runtime::builder().build()?;
    /// let v = Space::u1([(0, 2), (1, 1)]);
    /// let t = Tensor::rand(&rt, Dtype::F64, [&v], [&v])?;
    /// let id = Tensor::id(&rt, Dtype::F64, [&v])?;
    /// assert_eq!(id.compose(&t)?.data(), t.data());
    /// # Ok::<(), tenet::prelude::Error>(())
    /// ```
    pub fn id<'a, S>(rt: &Runtime, dtype: Dtype, spaces: S) -> Result<Self, Error>
    where
        S: IntoIterator<Item = &'a Space>,
    {
        let spaces: Vec<&Space> = spaces.into_iter().collect();
        Self::structural(rt, dtype, spaces.clone(), spaces, false, "id")
    }

    /// The canonical structural isomorphism `codomain <- domain` (TensorKit
    /// `isomorphism(W ← V)`, `tensors/linalg.jl:102-109`): every
    /// coupled-sector block is the identity matrix, which requires the fused
    /// codomain and domain to carry identical sector content. The
    /// finite-torus norm fuser is `isomorphism(fuse(dual(l) ⊗ l) ←
    /// dual(l) ⊗ l)`.
    ///
    /// ```
    /// use tenet::prelude::*;
    ///
    /// let rt = Runtime::builder().build()?;
    /// let v = Space::u1([(0, 1), (1, 1)]);
    /// let fused = v.dual().fuse(&v)?;
    /// let f = Tensor::isomorphism(&rt, Dtype::F64, [&fused], [&v.dual(), &v])?;
    /// // Unitary: f† ∘ f is the identity on the product space.
    /// let roundtrip = f.adjoint()?.compose(&f)?;
    /// assert_eq!(roundtrip.data(), Tensor::id(&rt, Dtype::F64, [&v.dual(), &v])?.data());
    /// # Ok::<(), tenet::prelude::Error>(())
    /// ```
    pub fn isomorphism<'a, C, D>(
        rt: &Runtime,
        dtype: Dtype,
        codomain: C,
        domain: D,
    ) -> Result<Self, Error>
    where
        C: IntoIterator<Item = &'a Space>,
        D: IntoIterator<Item = &'a Space>,
    {
        Self::structural(
            rt,
            dtype,
            codomain.into_iter().collect(),
            domain.into_iter().collect(),
            false,
            "isomorphism",
        )
    }

    /// TensorKit `unitary(W ← V)` (`tensors/linalg.jl:129-132`): identical
    /// to [`Self::isomorphism`] — TensorKit only adds a Euclidean
    /// inner-product check, which every tenet fusion rule satisfies.
    pub fn unitary<'a, C, D>(
        rt: &Runtime,
        dtype: Dtype,
        codomain: C,
        domain: D,
    ) -> Result<Self, Error>
    where
        C: IntoIterator<Item = &'a Space>,
        D: IntoIterator<Item = &'a Space>,
    {
        Self::structural(
            rt,
            dtype,
            codomain.into_iter().collect(),
            domain.into_iter().collect(),
            false,
            "unitary",
        )
    }

    /// The canonical isometry `codomain <- domain` (TensorKit
    /// `isometry(W ← V)`, `tensors/linalg.jl:149-158`): each coupled-sector
    /// block is the partial identity (the first `cols` columns of the
    /// identity), so `t† ∘ t = id(domain)`. Requires the domain to embed
    /// isometrically in the codomain (sectorwise `deg_domain <=
    /// deg_codomain`).
    ///
    /// ```
    /// use tenet::prelude::*;
    ///
    /// let rt = Runtime::builder().build()?;
    /// let small = Space::su2([(0, 1), (1, 1)]);
    /// let big = Space::su2([(0, 2), (1, 3), (2, 1)]);
    /// let w = Tensor::isometry(&rt, Dtype::F64, [&big], [&small])?;
    /// let id = Tensor::id(&rt, Dtype::F64, [&small])?;
    /// assert_eq!(w.adjoint()?.compose(&w)?.data(), id.data());
    /// # Ok::<(), tenet::prelude::Error>(())
    /// ```
    pub fn isometry<'a, C, D>(
        rt: &Runtime,
        dtype: Dtype,
        codomain: C,
        domain: D,
    ) -> Result<Self, Error>
    where
        C: IntoIterator<Item = &'a Space>,
        D: IntoIterator<Item = &'a Space>,
    {
        Self::structural(
            rt,
            dtype,
            codomain.into_iter().collect(),
            domain.into_iter().collect(),
            true,
            "isometry",
        )
    }

    /// TensorKit `twist(t, inds)` (`tensors/indexmanipulations.jl:62-97`):
    /// multiplies each fusion-tree block by the product over `legs` (flat
    /// leg indices, codomain first) of the ribbon-twist eigenvalue θ of that
    /// leg's uncoupled sector on the block's fusion tree. θ = −1 for odd
    /// fermionic sectors and +1 for every bosonic sector, so this is a no-op
    /// on purely bosonic legs and an involution on fermionic ones.
    ///
    /// ```
    /// use tenet::prelude::*;
    ///
    /// let rt = Runtime::builder().build()?;
    /// let v = Space::fz2([(0, 1), (1, 1)]);
    /// let t = Tensor::rand(&rt, Dtype::F64, [&v], [&v])?;
    /// let twisted = t.twist(&[0])?;
    /// assert_eq!(twisted.twist(&[0])?.data(), t.data()); // θ² = 1
    /// # Ok::<(), tenet::prelude::Error>(())
    /// ```
    pub fn twist(&self, legs: &[usize]) -> Result<Self, Error> {
        if self.is_adjoint_view() {
            return self.materialized_tensor()?.twist(legs);
        }
        let rank = self.rank();
        if let Some(&leg) = legs.iter().find(|&&leg| leg >= rank) {
            return Err(Error::InvalidArgument(format!(
                "twist leg {leg} out of range for rank {rank}"
            )));
        }
        if legs.is_empty() {
            return Ok(self.clone());
        }
        self.reject_unwired_su3("Tensor::twist")?;
        let nout = self.codomain_rank();
        if let Data::Diagonal(diagonal) = self.stored_data() {
            return with_user_rule!(self.ordinary_body().space, rule, {
                let sector_factor = |sector| {
                    legs.iter()
                        .map(|_| rule.twist_scalar(sector))
                        .product::<f64>()
                };
                if diagonal.sectors_all(|sector| sector_factor(sector) == 1.0) {
                    Ok(self.clone())
                } else {
                    Ok(self.with_diagonal(diagonal.scaled_by_sector(sector_factor)))
                }
            });
        }
        // TensorKit `has_shared_twist` (`indexmanipulations.jl`): the twist is
        // the identity when every requested leg carries theta = 1 on every
        // block. Bosonic rules are all-theta=1 by construction (O(1)
        // short-circuit — the Z2/U1/SU2 finite-torus paths); a
        // fermionic/anyonic tensor still shares its buffer when no requested
        // leg touches a twisted sector. Either way, skip the whole-buffer
        // clone-and-scale-by-1 and return the shared data.
        let twist_is_identity = with_user_rule!(self.ordinary_body().space, rule, {
            rule.braiding_style().is_bosonic() || {
                let structure = self.ordinary_body().space.structure();
                (0..structure.block_count()).try_fold(true, |noop, index| {
                    let block = structure.block(index)?;
                    Ok::<_, Error>(
                        noop && match block.key() {
                            BlockKey::FusionTree(key) => legs.iter().all(|&leg| {
                                let sector = if leg < nout {
                                    key.codomain_uncoupled()[leg]
                                } else {
                                    key.domain_uncoupled()[leg - nout]
                                };
                                rule.twist_scalar(sector) == 1.0
                            }),
                            _ => true,
                        },
                    )
                })?
            }
        });
        if twist_is_identity {
            return Ok(self.clone());
        }
        self.scaled_blocks(&self.ordinary_body().space, &|key| match key {
            BlockKey::FusionTree(key) => with_user_rule!(self.ordinary_body().space, rule, {
                legs.iter()
                    .map(|&leg| {
                        rule.twist_scalar(if leg < nout {
                            key.codomain_uncoupled()[leg]
                        } else {
                            key.domain_uncoupled()[leg - nout]
                        })
                    })
                    .product()
            }),
            _ => 1.0,
        })
    }

    /// TensorKit `flip(t, I)` (`tensors/indexmanipulations.jl:8-29`): return
    /// a tensor isomorphic to `self` where the duality flag of each leg in
    /// `legs` (flat indices, codomain first; a leg listed twice is flipped
    /// twice, sequentially) is toggled, `space(t', i) = flip(space(t, i))`.
    /// The stored sectors and the block layout are unchanged; each
    /// fusion-tree block picks up the Z-isomorphism phase of
    /// `fusiontrees/braiding_manipulations.jl:384-414` per flipped leg with
    /// uncoupled sector `a` and pre-flip duality `d` (χ = Frobenius-Schur
    /// phase, θ = ribbon twist; both real for every rule in scope):
    /// codomain leg → `d ? χ·θ : 1`; domain leg → `d ? χ : θ`.
    ///
    /// Like TensorKit's, this `flip` is *not* an involution: flipping the
    /// same leg twice returns to the original spaces but can scale odd
    /// blocks (e.g. by θ = −1 on fermionic legs); only `flip⁴ = id` in
    /// general.
    ///
    /// ```
    /// use tenet::prelude::*;
    ///
    /// let rt = Runtime::builder().build()?;
    /// let v = Space::fz2([(0, 1), (1, 1)]);
    /// let t = Tensor::from_block_fn(&rt, [&v], [&v], |key, _| match key {
    ///     BlockKey::FusionTree(key) if key.codomain_uncoupled()[0].id() == 0 => 2.0,
    ///     _ => 3.0,
    /// })?;
    /// // TensorKit 0.17: flip(t, 2) on V ← V negates the odd block (θ = −1)
    /// // and re-orients the domain leg (see the flip oracle test).
    /// let flipped = t.flip(&[1])?;
    /// assert_eq!(flipped.data(), &[2.0, -3.0]);
    /// assert!(!flipped.space(1)?.is_dual()); // was dual: space(t, 1) = dual(v)
    /// # Ok::<(), tenet::prelude::Error>(())
    /// ```
    pub fn flip(&self, legs: &[usize]) -> Result<Self, Error> {
        if self.is_adjoint_view() {
            return self.materialized_tensor()?.flip(legs);
        }
        let rank = self.rank();
        if let Some(&leg) = legs.iter().find(|&&leg| leg >= rank) {
            return Err(Error::InvalidArgument(format!(
                "flip leg {leg} out of range for rank {rank}"
            )));
        }
        if legs.is_empty() {
            return Ok(self.clone());
        }
        self.reject_unwired_su3("Tensor::flip")?;
        let hom = self.ordinary_body().space.homspace();
        let nout = hom.codomain().len();
        let leg_of = |leg: usize| {
            if leg < nout {
                &hom.codomain().legs()[leg]
            } else {
                &hom.domain().legs()[leg - nout]
            }
        };
        // Sequential semantics for repeated legs (TensorKit flips one index
        // at a time): record the duality each occurrence sees.
        let mut flip_count = vec![0usize; rank];
        let occurrences: Vec<(usize, bool)> = legs
            .iter()
            .map(|&leg| {
                let dual = leg_of(leg).is_dual() ^ (flip_count[leg] % 2 == 1);
                flip_count[leg] += 1;
                (leg, dual)
            })
            .collect();

        let toggled_leg = |index: usize, leg: &tenet_core::SectorLeg| {
            if flip_count[index] % 2 == 1 {
                tenet_core::SectorLeg::new(leg.iter(), !leg.is_dual())
            } else {
                leg.clone()
            }
        };
        let new_hom = FusionTreeHomSpace::new(
            FusionProductSpace::new(
                hom.codomain()
                    .legs()
                    .iter()
                    .enumerate()
                    .map(|(index, leg)| toggled_leg(index, leg)),
            ),
            FusionProductSpace::new(
                hom.domain()
                    .legs()
                    .iter()
                    .enumerate()
                    .map(|(index, leg)| toggled_leg(nout + index, leg)),
            ),
        );
        let new_space = self.ordinary_body().space.from_homspace(new_hom)?;
        // Flipping preserves the stored sectors, so the flipped space must
        // reproduce the block layout exactly; anything else is a bug.
        let old_structure = self.ordinary_body().space.structure();
        let new_structure = new_space.structure();
        if new_structure.block_count() != old_structure.block_count() {
            return Err(internal_layout_error("flip changed the block count"));
        }
        for index in 0..old_structure.block_count() {
            let old_block = old_structure.block(index)?;
            let new_block = new_structure.block(index)?;
            if old_block.offset() != new_block.offset() || old_block.shape() != new_block.shape() {
                return Err(internal_layout_error("flip changed the block layout"));
            }
        }

        let flipped = self.scaled_blocks(new_space.raw(), &|key| match key {
            BlockKey::FusionTree(key) => with_user_rule!(self.ordinary_body().space, rule, {
                occurrences
                    .iter()
                    .map(|&(leg, dual)| {
                        let sector = if leg < nout {
                            key.codomain_uncoupled()[leg]
                        } else {
                            key.domain_uncoupled()[leg - nout]
                        };
                        let chi = rule.frobenius_schur_phase_scalar(sector);
                        let theta = rule.twist_scalar(sector);
                        // TensorKit 0.17 flip coefficients (forward, real χ/θ).
                        if leg < nout {
                            if dual {
                                chi * theta
                            } else {
                                1.0
                            }
                        } else if dual {
                            chi
                        } else {
                            theta
                        }
                    })
                    .product()
            }),
            _ => 1.0,
        })?;
        Ok(Self::owned(
            flipped.rt.clone(),
            Arc::new(new_space),
            Arc::clone(&flipped.ordinary_body().data),
        ))
    }

    /// Clones the storage scaled block-wise by `factor_of(key)` (evaluated
    /// on the blocks of `structure_space`, whose layout must match the
    /// stored one), shared by [`Self::twist`] and [`Self::flip`].
    fn scaled_blocks(
        &self,
        structure_space: &DynamicFusionMapSpace,
        factor_of: &dyn Fn(&BlockKey) -> f64,
    ) -> Result<Self, Error> {
        let data = match self.coupled_data()? {
            Data::F64(data) => {
                let mut out = data.clone();
                scale_blocks_impl(structure_space, &mut out, factor_of)?;
                Data::F64(out)
            }
            Data::C64(data) => {
                let mut out = data.clone();
                scale_blocks_impl(structure_space, &mut out, factor_of)?;
                Data::C64(out)
            }
            Data::Diagonal(_) => unreachable!("coupled_data materializes Data::Diagonal"),
            #[cfg(feature = "cuda")]
            Data::CudaF64(_) => return Err(device_unsupported("twist/flip")),
        };
        Ok(Self::owned(
            self.rt.clone(),
            Arc::clone(&self.ordinary_body().space),
            Arc::new(data),
        ))
    }

    fn with_bound(&self, space: UserBoundSpace, data: Data) -> Result<Self, Error> {
        Ok(Self::owned(
            self.rt.clone(),
            Arc::new(space),
            Arc::new(data),
        ))
    }

    /// Resolves the stored representation into this tensor's dense coupled
    /// layout. Why not require every operation to call this: compact-aware
    /// operations must preserve O(r) storage and therefore inspect
    /// [`Data::Diagonal`] before reaching this dense fallback.
    fn coupled_data(&self) -> Result<&Data, Error> {
        let body = self.materialized_body()?;
        if let Data::Diagonal(diagonal) = body.data.as_ref() {
            return Ok(self
                .compact_dense
                .get_or_init(|| Arc::new(Self::materialize_diagonal(body, diagonal)))
                .as_ref());
        }
        Ok(body.data.as_ref())
    }

    fn materialized_body(&self) -> Result<&TensorBody, Error> {
        let TensorRepr::Adjoint(view) = &self.repr else {
            return Ok(self.ordinary_body());
        };
        if let Some(body) = view.materialized.get() {
            return Ok(body);
        }
        let _guard = view
            .init
            .lock()
            .map_err(|_| Error::InvalidArgument("adjoint initializer was poisoned".to_string()))?;
        if let Some(body) = view.materialized.get() {
            return Ok(body);
        }
        let logical_space = Self::initialize_logical_space(view)?;
        let built = Self::build_adjoint_body(&view.parent, logical_space)?;
        #[cfg(test)]
        view.materialized_body_builds
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let _ = view.materialized.set(built);
        view.materialized.get().map(Arc::as_ref).ok_or_else(|| {
            Error::InvalidArgument(
                "adjoint materialization completed without publishing its owned body".to_string(),
            )
        })
    }

    fn materialized_tensor(&self) -> Result<Self, Error> {
        let body = self.materialized_body()?;
        Ok(Self::owned(
            self.rt.clone(),
            Arc::clone(&body.space),
            Arc::clone(&body.data),
        ))
    }

    fn logical_space(&self) -> Result<&UserBoundSpace, Error> {
        let TensorRepr::Adjoint(view) = &self.repr else {
            return Ok(self.ordinary_body().space.as_ref());
        };
        if let Some(space) = view.logical_space.get() {
            return Ok(space);
        }
        let _guard = view
            .init
            .lock()
            .map_err(|_| Error::InvalidArgument("adjoint initializer was poisoned".to_string()))?;
        Self::initialize_logical_space(view).map(Arc::as_ref)
    }

    fn initialize_logical_space(view: &AdjointView) -> Result<&Arc<UserBoundSpace>, Error> {
        if let Some(space) = view.logical_space.get() {
            return Ok(space);
        }
        let space = Arc::new(view.parent.space.adjoint_space()?);
        #[cfg(test)]
        view.logical_space_builds
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let _ = view.logical_space.set(space);
        view.logical_space.get().ok_or_else(|| {
            Error::InvalidArgument(
                "adjoint logical-space initialization completed without publishing it".to_string(),
            )
        })
    }

    fn build_adjoint_body(
        parent: &TensorBody,
        logical_space: &Arc<UserBoundSpace>,
    ) -> Result<Arc<TensorBody>, Error> {
        macro_rules! materialize {
            ($space:expr, $variant:ident, $function:ident, $data:expr, $lift:ident) => {{
                let (derived_space, data) = tenet_tensors::$function($space, $data)?;
                let derived_space = UserBoundSpace::$variant(derived_space);
                if derived_space != **logical_space {
                    return Err(internal_layout_error(
                        "adjoint data materialization disagrees with cached logical space",
                    ));
                }
                Ok::<_, Error>(Arc::new(TensorBody {
                    space: Arc::clone(logical_space),
                    data: Arc::new(Data::$lift(data)),
                }))
            }};
        }
        match (parent.space.as_ref(), parent.data.as_ref()) {
            (UserBoundSpace::U1(space), Data::F64(data)) => {
                materialize!(space, U1, adjoint_bound_dyn_lowered, data, F64)
            }
            (UserBoundSpace::U1(space), Data::C64(data)) => {
                materialize!(space, U1, adjoint_bound_dyn_lowered, data, C64)
            }
            (UserBoundSpace::Z2(space), Data::F64(data)) => {
                materialize!(space, Z2, adjoint_bound_dyn_lowered, data, F64)
            }
            (UserBoundSpace::Z2(space), Data::C64(data)) => {
                materialize!(space, Z2, adjoint_bound_dyn_lowered, data, C64)
            }
            (UserBoundSpace::FZ2(space), Data::F64(data)) => {
                materialize!(space, FZ2, adjoint_bound_dyn_lowered, data, F64)
            }
            (UserBoundSpace::FZ2(space), Data::C64(data)) => {
                materialize!(space, FZ2, adjoint_bound_dyn_lowered, data, C64)
            }
            (UserBoundSpace::SU2(space), Data::F64(data)) => {
                materialize!(space, SU2, adjoint_bound_dyn_lowered, data, F64)
            }
            (UserBoundSpace::SU2(space), Data::C64(data)) => {
                materialize!(space, SU2, adjoint_bound_dyn_lowered, data, C64)
            }
            (UserBoundSpace::U1FZ2(space), Data::F64(data)) => {
                materialize!(space, U1FZ2, adjoint_bound_dyn_lowered, data, F64)
            }
            (UserBoundSpace::U1FZ2(space), Data::C64(data)) => {
                materialize!(space, U1FZ2, adjoint_bound_dyn_lowered, data, C64)
            }
            (UserBoundSpace::FZ2U1SU2(space), Data::F64(data)) => {
                materialize!(space, FZ2U1SU2, adjoint_bound_dyn_lowered, data, F64)
            }
            (UserBoundSpace::FZ2U1SU2(space), Data::C64(data)) => {
                materialize!(space, FZ2U1SU2, adjoint_bound_dyn_lowered, data, C64)
            }
            (UserBoundSpace::Su3(space), Data::F64(data)) => {
                materialize!(space, Su3, adjoint_bound_dyn_generic, data, F64)
            }
            (UserBoundSpace::Su3(space), Data::C64(data)) => {
                materialize!(space, Su3, adjoint_bound_dyn_generic, data, C64)
            }
            (_, Data::Diagonal(_)) => Err(Error::InvalidArgument(
                "compact diagonal tensors do not use the lazy adjoint representation".to_string(),
            )),
            #[cfg(feature = "cuda")]
            (_, Data::CudaF64(_)) => {
                Err(device_unsupported("materializing an adjoint device tensor"))
            }
        }
    }

    /// A non-diagonal clone: `Data::Diagonal` materialized into its dense
    /// equivalent, everything else shared by `Arc` (cheap). Why not use this in
    /// compact-aware arithmetic: it would recreate the O(r²) payload those
    /// paths are designed to avoid.
    fn densified_if_diagonal(&self) -> Self {
        if !matches!(self.stored_data(), Data::Diagonal(_)) {
            return self.clone();
        }
        let data = self
            .coupled_data()
            .expect("a valid compact diagonal tensor has a total dense representation")
            .clone();
        Self::owned(
            self.rt.clone(),
            Arc::clone(&self.ordinary_body().space),
            Arc::new(data),
        )
    }

    /// Rebuilds the dense block-diagonal buffer of a [`Data::Diagonal`] tensor in
    /// its own (`space`) layout. This is the eager fallback for dense-only
    /// consumers and reproduces the former dense diagonal tensor bit-for-bit via
    /// [`tenet_matrixalgebra::diagonal_bond_data`].
    fn materialize_diagonal(body: &TensorBody, diagonal: &DiagonalData) -> Data {
        match diagonal {
            DiagonalData::RealF64(spectrum) => Data::F64(
                tenet_matrixalgebra::diagonal_bond_data(&body.space, spectrum, &|value| value)
                    .expect("diagonal fill is total on the stored bond space"),
            ),
            DiagonalData::RealC64(spectrum) => Data::C64(
                tenet_matrixalgebra::diagonal_bond_data(&body.space, spectrum, &|value| {
                    Complex64::new(value, 0.0)
                })
                .expect("diagonal fill is total on the stored bond space"),
            ),
            DiagonalData::C64(spectrum) => Data::C64(
                tenet_matrixalgebra::diagonal_bond_data(&body.space, spectrum, &|value| value)
                    .expect("diagonal fill is total on the stored bond space"),
            ),
        }
    }

    /// The scalar type this tensor stores.
    pub fn dtype(&self) -> Dtype {
        // Discriminant only; dtype is adjoint-invariant, so read the stored
        // buffer directly (no need to materialize a lazy adjoint).
        match self.stored_data() {
            Data::F64(_) => Dtype::F64,
            Data::C64(_) => Dtype::C64,
            // Diagonal storage carries its own dtype tag (no materialization).
            Data::Diagonal(diagonal) => diagonal.dtype(),
            #[cfg(feature = "cuda")]
            Data::CudaF64(_) => Dtype::F64,
        }
    }

    /// Where this tensor's data lives: [`Placement::Host`] or
    /// [`Placement::Cuda`] with the device ordinal. Transfers are always
    /// explicit (`to_cuda()` / `to_host()`).
    pub fn placement(&self) -> Placement {
        match self.stored_data() {
            Data::F64(_) | Data::C64(_) | Data::Diagonal(_) => Placement::Host,
            #[cfg(feature = "cuda")]
            Data::CudaF64(storage) => storage.placement(),
        }
    }

    /// Uploads an f64 host tensor to the runtime's CUDA device (built with
    /// `Runtime::builder().cuda(device)`); a cheap clone when already
    /// device-resident. Explicit errors: c64 tensors (no device c64 storage
    /// yet) and runtimes built without a CUDA device.
    #[cfg(feature = "cuda")]
    pub fn to_cuda(&self) -> Result<Self, Error> {
        if let TensorRepr::Adjoint(view) = &self.repr {
            let parent = Self::owned(
                self.rt.clone(),
                Arc::clone(&view.parent.space),
                Arc::clone(&view.parent.data),
            )
            .to_cuda()?;
            return parent.adjoint();
        }
        let data = match self.coupled_data()? {
            Data::CudaF64(storage) => Data::CudaF64(Arc::clone(storage)),
            Data::Diagonal(_) => unreachable!("coupled_data materializes Data::Diagonal"),
            Data::C64(_) => {
                return Err(device_unsupported("uploading a c64 tensor"));
            }
            Data::F64(host) => {
                let mut state = self.rt.lock();
                let cuda = state.cuda.as_mut().ok_or_else(|| {
                    Error::InvalidArgument(
                        "this runtime was built without a CUDA device; use \
                         Runtime::builder().cuda(device)"
                            .to_string(),
                    )
                })?;
                Data::CudaF64(Arc::new(CudaStorage::upload(cuda, host)?))
            }
        };
        Ok(Self::owned(
            self.rt.clone(),
            Arc::clone(&self.ordinary_body().space),
            Arc::new(data),
        ))
    }

    /// Downloads a device tensor back to host storage; a plain copy when
    /// already host-resident.
    #[cfg(feature = "cuda")]
    pub fn to_host(&self) -> Result<Self, Error> {
        if let TensorRepr::Adjoint(view) = &self.repr {
            let parent = Self::owned(
                self.rt.clone(),
                Arc::clone(&view.parent.space),
                Arc::clone(&view.parent.data),
            )
            .to_host()?;
            return parent.adjoint();
        }
        let data = match self.coupled_data()? {
            Data::F64(_) | Data::C64(_) => self.coupled_data()?.clone(),
            Data::Diagonal(_) => unreachable!("coupled_data materializes Data::Diagonal"),
            Data::CudaF64(storage) => {
                let mut state = self.rt.lock();
                let cuda = state.cuda.as_mut().ok_or_else(|| {
                    Error::InvalidArgument(
                        "this runtime was built without a CUDA device".to_string(),
                    )
                })?;
                Data::F64(storage.download(cuda)?)
            }
        };
        Ok(Self::owned(
            self.rt.clone(),
            Arc::clone(&self.ordinary_body().space),
            Arc::new(data),
        ))
    }

    /// The [`Runtime`] this tensor was created from (a shared handle).
    pub fn runtime(&self) -> &Runtime {
        &self.rt
    }

    /// Number of codomain legs.
    pub fn codomain_rank(&self) -> usize {
        self.metadata().nout()
    }

    /// Number of domain legs.
    pub fn domain_rank(&self) -> usize {
        self.metadata().nin()
    }

    /// Total number of legs.
    pub fn rank(&self) -> usize {
        self.metadata().rank()
    }

    /// Number of codomain (output) legs. TensorKit `numout`; alias of
    /// [`Self::codomain_rank`].
    pub fn numout(&self) -> usize {
        self.codomain_rank()
    }

    /// Number of domain (input) legs. TensorKit `numin`; alias of
    /// [`Self::domain_rank`].
    pub fn numin(&self) -> usize {
        self.domain_rank()
    }

    /// Total number of legs. TensorKit `numind`; alias of [`Self::rank`].
    pub fn numind(&self) -> usize {
        self.rank()
    }

    /// Number of tensors currently sharing this tensor's storage allocation.
    #[doc(hidden)]
    pub fn storage_strong_count(&self) -> usize {
        Arc::strong_count(self.stored_data_arc())
    }

    /// Flat `f64` storage in the TensorKit-equivalent coupled-sector matrix
    /// layout (column-major inside each coupled block).
    ///
    /// This is an **internal-packing inspection API** (tests, debugging,
    /// oracle comparisons), not a general element-access API:
    ///
    /// - The slice is the internal buffer in the coupled-sector matrix
    ///   layout; element positions depend on block order, the fusion-tree
    ///   basis, and column-major packing.
    /// - That layout is **not a stable ABI**: it may change between
    ///   versions without notice.
    /// - There are no implicit device copies: on a device tensor this
    ///   panics — download explicitly with `to_host()` first.
    /// - For semantic access, prefer the operation APIs (contractions,
    ///   [`Self::scalar`], norms); a stable block iterator / dense export
    ///   would be a separate future API.
    ///
    /// # Panics
    ///
    /// Panics if the tensor stores c64 data (use [`Self::data_c64`]) or is
    /// device-resident (use `to_host()`). Both are legal tensor states, so
    /// prefer [`Self::try_data`] when the dtype/placement is not statically
    /// known — this method is the panicking half of that pair (#128).
    pub fn data(&self) -> &[f64] {
        self.try_data()
            .expect("data(): tensor is not host f64; use try_data()/data_c64()/to_host()")
    }

    /// Flat host `f64` storage, or a typed error when the tensor is not in that
    /// state. The recoverable counterpart of [`Self::data`]: a c64 tensor
    /// yields [`Error::DtypeMismatch`] and a device tensor
    /// [`Error::PlacementMismatch`] instead of panicking (#128). Same
    /// internal-packing caveats as [`Self::data`].
    pub fn try_data(&self) -> Result<&[f64], Error> {
        if self.placement() != Placement::Host {
            return Err(Error::PlacementMismatch);
        }
        match self.coupled_data()? {
            Data::F64(data) => Ok(data),
            Data::C64(_) => Err(Error::DtypeMismatch),
            Data::Diagonal(_) => unreachable!("coupled_data materializes Data::Diagonal"),
            #[cfg(feature = "cuda")]
            Data::CudaF64(_) => Err(Error::PlacementMismatch),
        }
    }

    /// Flat [`Complex64`] storage in the coupled-sector matrix layout.
    ///
    /// The same caveats as [`Self::data`] apply: this inspects the internal
    /// coupled-sector packing (layout-dependent, not a stable ABI, no
    /// implicit device copies; intended for tests and debugging).
    ///
    /// # Panics
    ///
    /// Panics if the tensor stores f64 data (use [`Self::data`]) or is
    /// device-resident (use `to_host()`). Both are legal tensor states, so
    /// prefer [`Self::try_data_c64`] when the dtype/placement is not
    /// statically known — this method is the panicking half of that pair
    /// (#128).
    pub fn data_c64(&self) -> &[Complex64] {
        self.try_data_c64()
            .expect("data_c64(): tensor is not host c64; use try_data_c64()/data()/to_host()")
    }

    /// Flat host [`Complex64`] storage, or a typed error when the tensor is not
    /// in that state. The recoverable counterpart of [`Self::data_c64`]: an
    /// f64 tensor yields [`Error::DtypeMismatch`] and a device tensor
    /// [`Error::PlacementMismatch`] instead of panicking (#128). Same
    /// internal-packing caveats as [`Self::data`].
    pub fn try_data_c64(&self) -> Result<&[Complex64], Error> {
        if self.placement() != Placement::Host {
            return Err(Error::PlacementMismatch);
        }
        match self.coupled_data()? {
            Data::C64(data) => Ok(data),
            Data::F64(_) => Err(Error::DtypeMismatch),
            Data::Diagonal(_) => unreachable!("coupled_data materializes Data::Diagonal"),
            #[cfg(feature = "cuda")]
            Data::CudaF64(_) => Err(Error::PlacementMismatch),
        }
    }

    /// Widens to a c64 tensor (imaginary parts zero); a cheap clone when the
    /// tensor already stores c64 data.
    ///
    /// # Panics
    ///
    /// Panics if the tensor is device-resident (a legal state); prefer
    /// [`Self::try_to_c64`], the recoverable half of this pair (#128).
    pub fn to_c64(&self) -> Self {
        self.try_to_c64()
            .expect("to_c64(): tensor is device-resident; use try_to_c64()/to_host()")
    }

    /// Widens to a c64 tensor, or a typed error when widening is not possible
    /// in place: a device-resident tensor yields [`Error::PlacementMismatch`]
    /// instead of panicking (#128). The recoverable counterpart of
    /// [`Self::to_c64`].
    pub fn try_to_c64(&self) -> Result<Self, Error> {
        if let Data::Diagonal(diagonal) = self.stored_data() {
            return Ok(self.with_diagonal(diagonal.to_c64_storage()));
        }
        let data = match self.coupled_data()? {
            Data::F64(data) => Data::C64(
                data.iter()
                    .map(|&value| Complex64::new(value, 0.0))
                    .collect(),
            ),
            Data::C64(data) => Data::C64(data.clone()),
            Data::Diagonal(_) => unreachable!("coupled_data materializes Data::Diagonal"),
            #[cfg(feature = "cuda")]
            Data::CudaF64(_) => return Err(Error::PlacementMismatch),
        };
        Ok(Self::owned(
            self.rt.clone(),
            Arc::clone(&self.materialized_body()?.space),
            Arc::new(data),
        ))
    }

    /// A zero tensor on the same spaces and dtype as `self` (TensorKit
    /// `zerovector` / `zero`). Cheapest same-shape constructor: scales the
    /// storage by zero rather than re-deriving the block structure.
    pub fn zeros_like(&self) -> Result<Self, Error> {
        self.scale(0.0)
    }

    /// Quantum-dimension-weighted total dimension of every leg, in flat
    /// order (codomain legs first, then domain legs). This is the same
    /// notion as [`crate::prelude::Space::dim`] per leg; contraction
    /// planners use it as a size/FLOP proxy.
    pub fn leg_dims(&self) -> Result<Vec<usize>, Error> {
        let metadata = self.metadata();
        if self.rule_kind() == RuleKind::Su3 {
            use tenet_core::GenericRigidSymbols;
            let rule = self.su3_rule();
            return Ok(metadata
                .codomain()
                .legs()
                .iter()
                .chain(metadata.domain().legs())
                .map(|leg| {
                    leg.iter()
                        .map(|(sector, deg)| {
                            let sqrt = rule.sqrt_dim_scalar(sector);
                            deg * (sqrt * sqrt).round() as usize
                        })
                        .sum()
                })
                .collect());
        }
        with_user_rule!(self.rule_authority_space(), rule, {
            Ok(metadata
                .codomain()
                .legs()
                .iter()
                .chain(metadata.domain().legs())
                .map(|leg| {
                    leg.iter()
                        .map(|(sector, deg)| {
                            (deg as f64 * rule.dim_scalar(sector)).round() as usize
                        })
                        .sum()
                })
                .collect())
        })
    }

    /// Quantum-dimension-weighted size of one flat leg.
    pub fn leg_dim(&self, axis: usize) -> Result<usize, Error> {
        let metadata = self.metadata();
        let leg = if axis < metadata.nout() {
            &metadata.codomain().legs()[axis]
        } else if axis < metadata.rank() {
            &metadata.domain().legs()[axis - metadata.nout()]
        } else {
            return Err(Error::InvalidArgument(format!(
                "axis {axis} out of range for rank {}",
                metadata.rank()
            )));
        };
        if self.rule_kind() == RuleKind::Su3 {
            use tenet_core::GenericRigidSymbols;
            let rule = self.su3_rule();
            return Ok(leg
                .iter()
                .map(|(sector, deg)| {
                    let sqrt = rule.sqrt_dim_scalar(sector);
                    deg * (sqrt * sqrt).round() as usize
                })
                .sum());
        }
        with_user_rule!(self.rule_authority_space(), rule, {
            Ok(leg
                .iter()
                .map(|(sector, deg)| (deg as f64 * rule.dim_scalar(sector)).round() as usize)
                .sum())
        })
    }

    /// The user-facing [`Space`] of flat leg `axis`, following TensorKit's
    /// `space(t, i)` convention: `codomain[i]` for `i < codomain_rank()`,
    /// `dual(domain[i - codomain_rank()])` otherwise.
    pub fn space(&self, axis: usize) -> Result<Space, Error> {
        let metadata = self.metadata();
        let nout = metadata.nout();
        if axis < nout {
            Ok(Space::from_leg(
                Arc::new(self.rule_context()),
                &metadata.codomain().legs()[axis],
            ))
        } else if axis < metadata.rank() {
            Ok(Space::from_leg(
                Arc::new(self.rule_context()),
                &metadata.domain().legs()[axis - nout],
            )
            .dual())
        } else {
            Err(Error::InvalidArgument(format!(
                "axis {axis} out of range for rank {}",
                metadata.rank()
            )))
        }
    }

    /// The codomain spaces, in leg order.
    pub fn codomain_spaces(&self) -> Vec<Space> {
        let metadata = self.metadata();
        let context = Arc::new(self.rule_context());
        metadata
            .codomain()
            .legs()
            .iter()
            .map(|leg| Space::from_leg(Arc::clone(&context), leg))
            .collect()
    }

    /// The domain spaces, in leg order (the spaces as written, i.e. *not*
    /// dualized; `t.space(codomain_rank() + i)` is their dual).
    pub fn domain_spaces(&self) -> Vec<Space> {
        let metadata = self.metadata();
        let context = Arc::new(self.rule_context());
        metadata
            .domain()
            .legs()
            .iter()
            .map(|leg| Space::from_leg(Arc::clone(&context), leg))
            .collect()
    }

    fn check_rank0(&self) -> Result<(), Error> {
        if self.rank() != 0 {
            return Err(Error::InvalidArgument(format!(
                "scalar() requires a rank-0 tensor, got rank {}",
                self.rank()
            )));
        }
        Ok(())
    }

    /// The single element of a rank-0 (scalar) tensor, e.g. the result of
    /// contracting every leg. The returned [`Scalar`] variant matches
    /// [`Self::dtype`] (`F64` for f64 tensors, `C64` for c64 tensors);
    /// errors on tensors with legs.
    pub fn scalar(&self) -> Result<Scalar, Error> {
        self.check_rank0()?;
        match self.coupled_data()? {
            Data::F64(data) => Ok(Scalar::F64(data.iter().sum())),
            Data::C64(data) => Ok(Scalar::C64(data.iter().sum())),
            Data::Diagonal(_) => unreachable!("coupled_data materializes Data::Diagonal"),
            #[cfg(feature = "cuda")]
            Data::CudaF64(_) => Err(device_unsupported("scalar()")),
        }
    }

    fn check_same_world(&self, other: &Self) -> Result<(), Error> {
        if self.rule_authority_space().identity() != other.rule_authority_space().identity() {
            return Err(Error::RuleMismatch);
        }
        if !self.rt.same_runtime(&other.rt) {
            return Err(Error::RuntimeMismatch);
        }
        if self.placement() != other.placement() {
            return Err(Error::PlacementMismatch);
        }
        if self.dtype() != other.dtype() {
            return Err(Error::DtypeMismatch);
        }
        Ok(())
    }

    fn validate_host_destination(&self, input: &Self) -> Result<(), Error> {
        self.check_same_world(input)?;
        if self.placement() != Placement::Host {
            return Err(Error::PlacementMismatch);
        }
        if self.is_adjoint_view() || matches!(self.stored_data(), Data::Diagonal(_)) {
            return Err(Error::InvalidArgument(
                "destination must use ordinary dense host storage".to_string(),
            ));
        }
        if Arc::ptr_eq(self.stored_data_arc(), input.stored_data_arc()) {
            return Err(Error::InvalidArgument(
                "destination storage must not alias an input".to_string(),
            ));
        }
        Ok(())
    }

    fn validate_exact_destination_space(
        &self,
        expected: &DynamicFusionMapSpace,
    ) -> Result<(), Error> {
        if self.ordinary_body().space.raw() != expected {
            return Err(Error::InvalidArgument(
                "destination fusion space or block layout does not match the operation result"
                    .to_string(),
            ));
        }
        self.validate_destination_storage_len(expected)
    }

    fn validate_exact_destination_space_arc(
        &self,
        expected: &Arc<UserBoundSpace>,
    ) -> Result<(), Error> {
        if self.ordinary_body().space.raw() != expected.raw() {
            return Err(Error::InvalidArgument(
                "destination fusion space or block layout does not match the operation result"
                    .to_string(),
            ));
        }
        self.validate_destination_storage_len(expected.raw())
    }

    fn validate_destination_storage_len(
        &self,
        expected: &DynamicFusionMapSpace,
    ) -> Result<(), Error> {
        let required = expected.required_len()?;
        let actual = match self.stored_data() {
            Data::F64(data) => data.len(),
            Data::C64(data) => data.len(),
            Data::Diagonal(_) => 0,
            #[cfg(feature = "cuda")]
            Data::CudaF64(_) => 0,
        };
        if actual != required {
            return Err(Error::InvalidArgument(format!(
                "destination storage length {actual} does not match required length {required}"
            )));
        }
        Ok(())
    }

    /// Categorical composition `self * rhs`: contracts `self`'s domain with
    /// `rhs`'s codomain, leg by leg. TensorKit `A * B` (`mul!` on coupled
    /// blocks); also available as the `&a * &b` operator (see the
    /// [`std::ops::Mul`] impl, which panics instead of returning `Result`).
    ///
    /// # Fermionic semantics: `compose` vs `contract`
    ///
    /// `compose` / `&a * &b` is TensorKit's `A * B` / `mul!`: **no**
    /// fermionic supertrace twist is inserted on dual composed legs.
    /// [`Self::contract`] and the `tensor!` macro are TensorKit's
    /// `tensorcontract!` / `@tensor`: dual contracted legs **are** twisted
    /// (TensorKit `tensoroperations.jl:388-420` twists only in
    /// `blas_contract!`, never in `mul!`). For bosonic rules the two agree
    /// exactly; for fermionic rules (fZ2 and products containing it) they
    /// can differ by signs. Worked example — the odd sector flips sign:
    ///
    /// ```
    /// use tenet::prelude::*;
    ///
    /// let rt = Runtime::builder().build()?;
    /// let v = Space::fz2([(0, 1), (1, 1)]);
    /// // a : v <- v*, b : v* <- v; the composed legs are dual (v*).
    /// let odd = |key: &BlockKey| matches!(key, BlockKey::FusionTree(k)
    ///     if k.codomain_uncoupled()[0].id() == 1);
    /// let a = Tensor::from_block_fn(&rt, [&v], [&v.dual()], |k, _| if odd(k) { 2.0 } else { 5.0 })?;
    /// let b = Tensor::from_block_fn(&rt, [&v.dual()], [&v], |k, _| if odd(k) { 3.0 } else { 7.0 })?;
    /// let composed = a.compose(&b)?;                    // mul! semantics: no twist
    /// let contracted = a.contract(&b, &[1], &[0])?;     // tensorcontract!: twist on v*
    /// assert_eq!(composed.data()[0], contracted.data()[0]);  // even sector agrees
    /// assert_eq!(composed.data()[1], -contracted.data()[1]); // odd sector: sign flip
    /// # Ok::<(), tenet::prelude::Error>(())
    /// ```
    ///
    /// Rule of thumb: use `compose` when you mean operator/matrix
    /// multiplication of tensor maps (TensorKit `A * B`); use
    /// [`Self::contract`] / `tensor!` when you mean index-notation
    /// contraction (TensorKit `@tensor`). Bosonic results are identical.
    #[doc(alias = "mul")]
    #[doc(alias = "matmul")]
    pub fn compose(&self, rhs: &Self) -> Result<Self, Error> {
        if self.domain_rank() != rhs.codomain_rank() {
            return Err(Error::InvalidArgument(format!(
                "compose shape mismatch: lhs domain rank {} vs rhs codomain rank {}",
                self.domain_rank(),
                rhs.codomain_rank()
            )));
        }
        self.check_same_world(rhs)?;
        let lhs_axes: Vec<usize> = (self.codomain_rank()..self.rank()).collect();
        let rhs_axes: Vec<usize> = (0..rhs.codomain_rank()).collect();
        let mut diagonal_dst = if self.diagonal_data().is_some() || rhs.diagonal_data().is_some() {
            Some(self.contraction_output_space(rhs, &lhs_axes, &rhs_axes)?)
        } else {
            None
        };
        // Why not send a proven diagonal composition through GEMM: TensorKit's
        // `DiagonalTensorMap` `rmul!`/`lmul!` shows it is only per-block bond
        // scaling. The compact product is valid only after the derived output
        // proves the same rank-2 diagonal invariant.
        match (self.diagonal_data(), rhs.diagonal_data()) {
            (Some(lhs), Some(rhs_diagonal)) => {
                let dst_space = diagonal_dst
                    .take()
                    .expect("diagonal destination prepared when both operands are diagonal");
                if Self::is_diagonal_bond_space(dst_space.raw()) {
                    if let Some(product) = lhs.elementwise_product(rhs_diagonal) {
                        return self.with_bound(dst_space, Data::Diagonal(product));
                    }
                }
            }
            // `t * D`: scale `self`'s trailing bond axis (columns). `self.domain`
            // is the single bond leg == `D.codomain`, so the space is `self`'s.
            (None, Some(diagonal))
                if diagonal_dst.as_ref().map(UserBoundSpace::raw)
                    == Some(self.logical_space()?.raw()) =>
            {
                return self.scaled_axis_copy_diagonal(None, diagonal);
            }
            // `D * t`: scale `rhs`'s leading bond axis (rows). `rhs.codomain` is
            // the single bond leg == `D.domain`, so the space is `rhs`'s.
            (Some(diagonal), None)
                if diagonal_dst.as_ref().map(UserBoundSpace::raw)
                    == Some(rhs.logical_space()?.raw()) =>
            {
                return rhs.scaled_axis_copy_diagonal(Some(0), diagonal);
            }
            _ => {}
        }
        let fermionic = self.rule_kind() != RuleKind::Su3
            && with_user_rule!(self.rule_authority_space(), rule, {
                rule.braiding_style() == tenet_core::BraidingStyleKind::Fermionic
            });
        if fermionic {
            match (self.stored_data(), rhs.stored_data()) {
                (Data::F64(_), Data::F64(_)) | (Data::C64(_), Data::C64(_)) => {
                    return self.compose_host_fusion_impl(rhs, &lhs_axes, &rhs_axes);
                }
                _ => {}
            }
        }
        self.contract(rhs, &lhs_axes, &rhs_axes)
    }

    /// Contracts `lhs_axes` of `self` with `rhs_axes` of `rhs` (pairwise, in
    /// list order), with the default output order: `self`'s open axes
    /// ascending become the codomain, `rhs`'s open axes ascending become the
    /// domain. TensorKit `tensorcontract!` with default `pAB`.
    ///
    /// **Fermionic semantics**: like TensorKit `tensorcontract!` / `@tensor`
    /// (and the `tensor!` macro), this **twists** dual contracted legs with
    /// the fermionic supertrace twist — unlike [`Self::compose`] / `&a * &b`
    /// (TensorKit `A * B` / `mul!`), which never does. Bosonic rules are
    /// unaffected; fermionic rules can differ by signs. See the worked
    /// example on [`Self::compose`].
    pub fn contract(
        &self,
        rhs: &Self,
        lhs_axes: &[usize],
        rhs_axes: &[usize],
    ) -> Result<Self, Error> {
        self.check_same_world(rhs)?;
        if lhs_axes.len() != rhs_axes.len() {
            return Err(Error::InvalidArgument(format!(
                "contracted axis lists differ in length: {} vs {}",
                lhs_axes.len(),
                rhs_axes.len()
            )));
        }
        validate_contracted_axes(lhs_axes, self.rank())?;
        validate_contracted_axes(rhs_axes, rhs.rank())?;
        // SU(N) (Generic): a lazy-adjoint operand is materialized to plain
        // coupled data BEFORE contracting, rather than folded into the GEMM seam.
        // The mult-free seam folds a conjugate via its Structure route, whose
        // non-self-dual coupled-sector mislabel was the historical bug; SU(3) is
        // non-self-dual (3 <-> 3̄), so we route it through the mislabel-proof
        // eager materialization (`materialize_adjoint`, generic block-relabel)
        // instead. `scale(1.0)` reads `coupled_data` — the materialization point —
        // and returns an ordinary (non-lazy) tensor, so both operands then take
        // the direct core/compose GEMM with no conjugate flag. Gated on `Su3` to
        // keep the mult-free seam byte-for-byte unchanged (the χ32 guarantee).
        if self.rule_kind() == RuleKind::Su3 && (self.is_adjoint_view() || rhs.is_adjoint_view()) {
            let lhs = if self.is_adjoint_view() {
                self.scale(1.0)?
            } else {
                self.clone()
            };
            let rhs = if rhs.is_adjoint_view() {
                rhs.scale(1.0)?
            } else {
                rhs.clone()
            };
            return lhs.contract(&rhs, lhs_axes, rhs_axes);
        }
        // Order-parity fast path for a real or complex diagonal operand (#75): instead of
        // densifying it to an O(d²) block-diagonal and running an O(d²·n) GEMM,
        // scale the OTHER operand's contracted leg by the spectrum (O(d·n)) and
        // `permute` the result into the contract output arrangement (O(n)). The
        // `permute` reuses the tested recoupling/repartition machinery, so the
        // result space — including leg duality and the codomain/domain split — is
        // correct for the proven canonical geometries at any leg position within
        // the preserved partition side. This is the same
        // scale + one-permute structure TensorKit runs (a `Diagonal` block scales
        // the recoupled operand); see docs/complexity_parity_policy.md.
        //
        // `contract` (tensorcontract!) applies a supertrace twist to `rhs`'s
        // externally dual contracted legs; `mul!` does not. The canonical
        // diagonal routes below therefore fold that RHS twist into the scaled
        // operand. θ = ±1 by charge parity, identity for bosonic rules.
        // SU(N) (Generic) is bosonic and cannot ride the mult-free `with_rule!`
        // binding; short-circuit the twist probe (the diagonal fast path below
        // never fires for it — SU(N) has no `Data::Diagonal` factors yet).
        let fermionic = self.rule_kind() != RuleKind::Su3
            && with_user_rule!(self.rule_authority_space(), rule, {
                rule.braiding_style() == tenet_core::BraidingStyleKind::Fermionic
            });
        if lhs_axes.len() == 1 && rhs_axes.len() == 1 {
            let twist_rhs_leg = fermionic && rhs.external_axis_is_dual(rhs_axes[0])?;
            let diagonal_dst = if self.diagonal_data().is_some() || rhs.diagonal_data().is_some() {
                Some(self.contraction_output_space(rhs, lhs_axes, rhs_axes)?)
            } else {
                None
            };
            // Why not scale a noncanonical diagonal axis: crossing the
            // codomain/domain partition can dualize its sector label, while
            // block-local scaling indexes the compact spectrum by raw labels.
            // Keep those routes dense until scaling carries an explicit relabel.
            match (self.diagonal_data(), rhs.diagonal_data()) {
                (Some(lhs), Some(rhs_diagonal))
                    if self.rule_kind() != RuleKind::Su3 && lhs_axes == [1] && rhs_axes == [0] =>
                {
                    let folded_rhs = self.twist_folded_diagonal(rhs_diagonal, twist_rhs_leg);
                    let dst_space = diagonal_dst
                        .expect("diagonal destination prepared when both operands are diagonal");
                    if Self::is_diagonal_bond_space(dst_space.raw()) {
                        if let Some(product) = lhs.elementwise_product(&folded_rhs) {
                            return self.with_bound(dst_space, Data::Diagonal(product));
                        }
                    }
                }
                // A * D: scale A's contracted leg by the (twist-folded) spectrum,
                // then repartition to the output arrangement (A's open axes ->
                // codomain, the scaled leg -> domain).
                (None, Some(diagonal))
                    if lhs_axes[0] >= self.codomain_rank() && rhs_axes[0] == 0 =>
                {
                    let leg = lhs_axes[0];
                    let folded = self.twist_folded_diagonal(diagonal, twist_rhs_leg);
                    let scaled = self.scaled_axis_copy_diagonal(Some(leg), &folded)?;
                    let codomain: Vec<usize> = (0..self.rank()).filter(|&a| a != leg).collect();
                    let output = scaled.permute(&codomain, &[leg])?;
                    debug_assert_eq!(
                        Some(output.ordinary_body().space.raw()),
                        diagonal_dst.as_ref().map(UserBoundSpace::raw)
                    );
                    return Ok(output);
                }
                // D * A: pre-twist A's dual contracted leg, scale it, then
                // repartition (the scaled leg -> codomain 0, A's open -> domain).
                (Some(diagonal), None) if lhs_axes[0] == 1 && rhs_axes[0] < rhs.codomain_rank() => {
                    let leg = rhs_axes[0];
                    let pretwisted = if twist_rhs_leg {
                        rhs.twist(&[leg])?
                    } else {
                        rhs.clone()
                    };
                    let scaled = pretwisted.scaled_axis_copy_diagonal(Some(leg), diagonal)?;
                    let domain: Vec<usize> = (0..rhs.rank()).filter(|&a| a != leg).collect();
                    let output = scaled.permute(&[leg], &domain)?;
                    debug_assert_eq!(
                        Some(output.ordinary_body().space.raw()),
                        diagonal_dst.as_ref().map(UserBoundSpace::raw)
                    );
                    return Ok(output);
                }
                _ => {}
            }
        }
        // Why not generalize compact storage to every diagonal contraction: a
        // zero-axis outer product is rank 4 and a two-axis contraction is scalar,
        // neither fits `DiagonalData`'s rank-2 bond invariant. Those shapes and
        // any unproved rank-2 layout retain the ordinary dense fallback.
        if matches!(self.stored_data(), Data::Diagonal(_))
            || matches!(rhs.stored_data(), Data::Diagonal(_))
        {
            return self.densified_if_diagonal().contract(
                &rhs.densified_if_diagonal(),
                lhs_axes,
                rhs_axes,
            );
        }
        // SU(N) (Generic), Stage B3c-2 source-transform route: the direct GEMM
        // engine only accepts core/compose form (lhs contracted axes == its whole
        // domain in order, rhs contracted axes == its whole codomain in order).
        // Any other arrangement is canonicalized here by composing ALREADY
        // TK-pinned primitives — one generic permute per operand (the B3a tree
        // transform, which carries all the recoupling) followed by the core
        // contract — so this wiring adds no mathematics of its own; the route-
        // equivalence gate pins `contract == explicit permute + core contract`.
        // Recursion is bounded: the permuted arrangement is canonical by
        // construction, so the recursive call falls through to the seam below.
        // Placed after the diagonal fast paths so a diagonal bond operand keeps
        // its O(d·n) scaling route, and after the adjoint materialization so
        // operands here are plain dense tensors.
        if self.rule_kind() == RuleKind::Su3 {
            let canonical_lhs = (self.codomain_rank()..self.rank()).collect::<Vec<_>>();
            let canonical_rhs = (0..rhs.codomain_rank()).collect::<Vec<_>>();
            if lhs_axes != canonical_lhs.as_slice() || rhs_axes != canonical_rhs.as_slice() {
                let lhs_open: Vec<usize> =
                    (0..self.rank()).filter(|a| !lhs_axes.contains(a)).collect();
                let rhs_open: Vec<usize> =
                    (0..rhs.rank()).filter(|a| !rhs_axes.contains(a)).collect();
                let lhs = self.permute(&lhs_open, lhs_axes)?;
                let rhs = rhs.permute(rhs_axes, &rhs_open)?;
                let contracted = (lhs_open.len()..lhs.rank()).collect::<Vec<_>>();
                let rhs_contracted = (0..rhs_axes.len()).collect::<Vec<_>>();
                return lhs.contract(&rhs, &contracted, &rhs_contracted);
            }
        }
        // Fold a lazy adjoint into contraction without copying its blocks.
        // Planning keeps the logical adjoint space for categorical geometry,
        // while execution maps only referenced blocks and strides onto the
        // shared parent storage and conjugates their numerical values.
        //
        // Why not pass only the parent space and remap user axes here: doing so
        // loses non-self-dual sector relabeling before the fusion-tree plan is
        // built. Keeping logical and storage layouts separate matches
        // TensorKit's AdjointTensorMap boundary.
        match (self.stored_data(), rhs.stored_data()) {
            (Data::F64(_), Data::F64(_)) | (Data::C64(_), Data::C64(_)) => {
                self.contract_host_fusion_impl(rhs, lhs_axes, rhs_axes, OutputAxisOrder::identity())
            }
            #[cfg(feature = "cuda")]
            (Data::CudaF64(a), Data::CudaF64(b)) => {
                if self.is_adjoint_view() || rhs.is_adjoint_view() {
                    return Err(device_unsupported(
                        "contracting a lazy adjoint device tensor",
                    ));
                }
                self.contract_cuda_impl(rhs, a, b, lhs_axes, rhs_axes)
            }
            _ => Err(Error::DtypeMismatch),
        }
    }

    fn contract_host_fusion_impl(
        &self,
        rhs: &Self,
        lhs_axes: &[usize],
        rhs_axes: &[usize],
        output_order: OutputAxisOrder<'_>,
    ) -> Result<Self, Error> {
        self.contract_host_fusion_impl_with_semantics(
            rhs,
            lhs_axes,
            rhs_axes,
            output_order,
            ContractionSemantics::TensorContract,
        )
    }

    fn compose_host_fusion_impl(
        &self,
        rhs: &Self,
        lhs_axes: &[usize],
        rhs_axes: &[usize],
    ) -> Result<Self, Error> {
        self.contract_host_fusion_impl_with_semantics(
            rhs,
            lhs_axes,
            rhs_axes,
            OutputAxisOrder::identity(),
            ContractionSemantics::Composition,
        )
    }

    fn contract_host_fusion_impl_with_semantics(
        &self,
        rhs: &Self,
        lhs_axes: &[usize],
        rhs_axes: &[usize],
        output_order: OutputAxisOrder<'_>,
        semantics: ContractionSemantics,
    ) -> Result<Self, Error> {
        let (lhs_logical, lhs_storage, lhs_conj) = self.seam_operand()?;
        let (rhs_logical, rhs_storage, rhs_conj) = rhs.seam_operand()?;
        // The seam always consumes the raw stored buffer (it never materializes):
        // for a lazy adjoint that buffer is the shared parent, conjugated by the
        // flag; for an ordinary tensor it is just the stored data.
        match (self.stored_data(), rhs.stored_data()) {
            (Data::F64(a), Data::F64(b)) => self.contract_impl(
                lhs_logical,
                lhs_storage,
                a,
                lhs_conj,
                rhs_logical,
                rhs_storage,
                b,
                rhs_conj,
                rhs,
                lhs_axes,
                rhs_axes,
                output_order,
                semantics,
            ),
            (Data::C64(a), Data::C64(b)) => self.contract_impl(
                lhs_logical,
                lhs_storage,
                a,
                lhs_conj,
                rhs_logical,
                rhs_storage,
                b,
                rhs_conj,
                rhs,
                lhs_axes,
                rhs_axes,
                output_order,
                semantics,
            ),
            _ => Err(Error::DtypeMismatch),
        }
    }

    /// Returns categorical layout, physical storage layout, and numeric
    /// conjugation separately. Why not remap user axes here: lower planning
    /// must enumerate the logical adjoint geometry before mapping only the
    /// referenced storage blocks and strides.
    fn seam_operand(&self) -> Result<(&UserBoundSpace, &UserBoundSpace, bool), Error> {
        match &self.repr {
            TensorRepr::Owned(body) => Ok((body.space.as_ref(), body.space.as_ref(), false)),
            TensorRepr::Adjoint(view) => {
                Ok((self.logical_space()?, view.parent.space.as_ref(), true))
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn contract_impl<D: UserScalar>(
        &self,
        lhs_logical: &UserBoundSpace,
        lhs_storage: &UserBoundSpace,
        lhs_data: &[D],
        lhs_conj: bool,
        rhs_logical: &UserBoundSpace,
        rhs_storage: &UserBoundSpace,
        rhs_data: &[D],
        rhs_conj: bool,
        rhs: &Self,
        lhs_axes: &[usize],
        rhs_axes: &[usize],
        output_order: OutputAxisOrder<'_>,
        semantics: ContractionSemantics,
    ) -> Result<Self, Error> {
        // Lease a per-rule context so independent operations on one runtime do
        // not serialize while bound spaces remain the fusion authority.
        let mut lease = self.rt.lease_context()?;
        let context = lease.context();
        let dst_bound = self.logical_space()?.contracted_with_output_order(
            rhs.logical_space()?,
            lhs_axes,
            rhs_axes,
            output_order,
        )?;
        let mut data = vec![D::from_real(0.0); dst_bound.raw().required_len()?];
        let spec = TensorContractSpec::new_with_conjugation(
            lhs_axes,
            rhs_axes,
            output_order,
            lhs_conj,
            rhs_conj,
        );
        macro_rules! contract_bound {
            ($contexts:expr, $dst:expr, $lhs_logical:expr, $lhs_storage:expr, $rhs_logical:expr, $rhs_storage:expr) => {{
                // Why not use the generalized prelowered route unconditionally:
                // ordinary operands must retain the established accumulation
                // order and bitwise output; only lazy operands need categorical
                // geometry separated from their parent storage.
                if semantics == ContractionSemantics::Composition {
                    let lhs = if lhs_conj {
                        tenet_tensors::FusionOperand::prelowered_adjoint(
                            $lhs_logical.space(),
                            $lhs_storage.space(),
                        )?
                    } else {
                        tenet_tensors::FusionOperand::direct($lhs_logical.space())
                    };
                    let rhs = if rhs_conj {
                        tenet_tensors::FusionOperand::prelowered_adjoint(
                            $rhs_logical.space(),
                            $rhs_storage.space(),
                        )?
                    } else {
                        tenet_tensors::FusionOperand::direct($rhs_logical.space())
                    };
                    D::ctx_of($contexts).tensorcompose_fusion_dyn_into_lowered(
                        $dst,
                        &mut data,
                        lhs,
                        lhs_data,
                        rhs,
                        rhs_data,
                        lhs_axes,
                        rhs_axes,
                        D::from_real(1.0),
                        D::from_real(0.0),
                    )
                } else if !lhs_conj && !rhs_conj {
                    D::ctx_of($contexts).tensorcontract_fusion_dyn_into_lowered(
                        $dst,
                        &mut data,
                        $lhs_logical,
                        lhs_data,
                        $rhs_logical,
                        rhs_data,
                        TensorContractSpec::new(lhs_axes, rhs_axes, output_order),
                        D::from_real(1.0),
                        D::from_real(0.0),
                    )
                } else {
                    let lhs = if lhs_conj {
                        tenet_tensors::FusionOperand::prelowered_adjoint(
                            $lhs_logical.space(),
                            $lhs_storage.space(),
                        )?
                    } else {
                        tenet_tensors::FusionOperand::direct($lhs_logical.space())
                    };
                    let rhs = if rhs_conj {
                        tenet_tensors::FusionOperand::prelowered_adjoint(
                            $rhs_logical.space(),
                            $rhs_storage.space(),
                        )?
                    } else {
                        tenet_tensors::FusionOperand::direct($rhs_logical.space())
                    };
                    D::ctx_of($contexts).tensorcontract_fusion_dyn_prelowered_into_lowered(
                        $dst,
                        &mut data,
                        lhs,
                        lhs_data,
                        rhs,
                        rhs_data,
                        spec,
                        D::from_real(1.0),
                        D::from_real(0.0),
                    )
                }
            }};
        }
        match (
            &dst_bound,
            lhs_logical,
            lhs_storage,
            rhs_logical,
            rhs_storage,
        ) {
            (
                UserBoundSpace::U1(dst),
                UserBoundSpace::U1(lhs_logical),
                UserBoundSpace::U1(lhs_storage),
                UserBoundSpace::U1(rhs_logical),
                UserBoundSpace::U1(rhs_storage),
            ) => contract_bound!(
                &mut context.u1,
                dst,
                lhs_logical,
                lhs_storage,
                rhs_logical,
                rhs_storage
            ),
            (
                UserBoundSpace::Z2(dst),
                UserBoundSpace::Z2(lhs_logical),
                UserBoundSpace::Z2(lhs_storage),
                UserBoundSpace::Z2(rhs_logical),
                UserBoundSpace::Z2(rhs_storage),
            ) => contract_bound!(
                &mut context.z2,
                dst,
                lhs_logical,
                lhs_storage,
                rhs_logical,
                rhs_storage
            ),
            (
                UserBoundSpace::FZ2(dst),
                UserBoundSpace::FZ2(lhs_logical),
                UserBoundSpace::FZ2(lhs_storage),
                UserBoundSpace::FZ2(rhs_logical),
                UserBoundSpace::FZ2(rhs_storage),
            ) => contract_bound!(
                &mut context.fz2,
                dst,
                lhs_logical,
                lhs_storage,
                rhs_logical,
                rhs_storage
            ),
            (
                UserBoundSpace::SU2(dst),
                UserBoundSpace::SU2(lhs_logical),
                UserBoundSpace::SU2(lhs_storage),
                UserBoundSpace::SU2(rhs_logical),
                UserBoundSpace::SU2(rhs_storage),
            ) => contract_bound!(
                &mut context.su2,
                dst,
                lhs_logical,
                lhs_storage,
                rhs_logical,
                rhs_storage
            ),
            (
                UserBoundSpace::U1FZ2(dst),
                UserBoundSpace::U1FZ2(lhs_logical),
                UserBoundSpace::U1FZ2(lhs_storage),
                UserBoundSpace::U1FZ2(rhs_logical),
                UserBoundSpace::U1FZ2(rhs_storage),
            ) => contract_bound!(
                &mut context.u1_fz2,
                dst,
                lhs_logical,
                lhs_storage,
                rhs_logical,
                rhs_storage
            ),
            (
                UserBoundSpace::FZ2U1SU2(dst),
                UserBoundSpace::FZ2U1SU2(lhs_logical),
                UserBoundSpace::FZ2U1SU2(lhs_storage),
                UserBoundSpace::FZ2U1SU2(rhs_logical),
                UserBoundSpace::FZ2U1SU2(rhs_storage),
            ) => contract_bound!(
                &mut context.fz2_u1_su2,
                dst,
                lhs_logical,
                lhs_storage,
                rhs_logical,
                rhs_storage
            ),
            (
                UserBoundSpace::Su3(dst),
                UserBoundSpace::Su3(lhs),
                UserBoundSpace::Su3(_),
                UserBoundSpace::Su3(rhs),
                UserBoundSpace::Su3(_),
            ) => {
                if lhs_conj || rhs_conj {
                    return Err(Error::InvalidArgument(
                        "internal: SU(N) contraction reached the seam with a conjugate flag"
                            .to_string(),
                    ));
                }
                D::ctx_of(&mut context.su3).tensorcontract_fusion_dyn_into_generic(
                    dst,
                    &mut data,
                    lhs,
                    lhs_data,
                    rhs,
                    rhs_data,
                    TensorContractSpec::new(lhs_axes, rhs_axes, output_order),
                    D::from_real(1.0),
                    D::from_real(0.0),
                )
            }
            _ => return Err(Error::RuleMismatch),
        }?;
        let data = D::lift(data);
        self.with_bound(dst_bound, data)
    }

    /// Device contraction: same plan compilation and resolution cache as the
    /// host path (spaces are host-side metadata), replayed directly on the
    /// device buffers via one offset GEMM per coupled-sector matrix.
    /// Phase-1 scope: only the canonical fully-direct route (exactly
    /// `contract`'s `alpha = 1`, `beta = 0` semantics); contractions that
    /// resolve to dynamic tree transforms return an explicit error.
    #[cfg(feature = "cuda")]
    fn contract_cuda_impl(
        &self,
        rhs: &Self,
        lhs_data: &CudaStorage,
        rhs_data: &CudaStorage,
        lhs_axes: &[usize],
        rhs_axes: &[usize],
    ) -> Result<Self, Error> {
        let mut guard = self.rt.lock();
        let state = &mut *guard;
        let cuda = state.cuda.as_mut().ok_or_else(|| {
            Error::InvalidArgument(
                "this runtime was built without a CUDA device; use \
                 Runtime::builder().cuda(device)"
                    .to_string(),
            )
        })?;
        if self.is_adjoint_view() || rhs.is_adjoint_view() {
            return Err(device_unsupported(
                "contracting a lazy adjoint device tensor",
            ));
        }
        let dst_bound =
            self.logical_space()?
                .contracted(rhs.logical_space()?, lhs_axes, rhs_axes)?;
        // ponytail: destination allocated by uploading host zeros; a
        // device-side alloc/memset seam replaces this if upload cost
        // ever matters (the direct route overwrites every element).
        let mut dst = CudaStorage::upload(cuda, &vec![0.0; dst_bound.raw().required_len()?])?;
        let spec = TensorContractSpec::with_default_output_order(lhs_axes, rhs_axes);
        macro_rules! contract_cuda_bound {
            ($contexts:expr, $dst:expr, $lhs:expr, $rhs:expr) => {
                $contexts.f64.tensorcontract_fusion_dyn_direct_on_storage(
                    &mut CudaStorageGemm::new(cuda),
                    $dst,
                    &mut dst,
                    $lhs,
                    lhs_data,
                    $rhs,
                    rhs_data,
                    spec,
                )
            };
        }
        match (
            &dst_bound,
            self.ordinary_body().space.as_ref(),
            rhs.ordinary_body().space.as_ref(),
        ) {
            (UserBoundSpace::U1(dst), UserBoundSpace::U1(lhs), UserBoundSpace::U1(rhs)) => {
                contract_cuda_bound!(&mut state.u1, dst, lhs, rhs)
            }
            (UserBoundSpace::Z2(dst), UserBoundSpace::Z2(lhs), UserBoundSpace::Z2(rhs)) => {
                contract_cuda_bound!(&mut state.z2, dst, lhs, rhs)
            }
            (UserBoundSpace::FZ2(dst), UserBoundSpace::FZ2(lhs), UserBoundSpace::FZ2(rhs)) => {
                contract_cuda_bound!(&mut state.fz2, dst, lhs, rhs)
            }
            (UserBoundSpace::SU2(dst), UserBoundSpace::SU2(lhs), UserBoundSpace::SU2(rhs)) => {
                contract_cuda_bound!(&mut state.su2, dst, lhs, rhs)
            }
            (
                UserBoundSpace::U1FZ2(dst),
                UserBoundSpace::U1FZ2(lhs),
                UserBoundSpace::U1FZ2(rhs),
            ) => contract_cuda_bound!(&mut state.u1_fz2, dst, lhs, rhs),
            (
                UserBoundSpace::FZ2U1SU2(dst),
                UserBoundSpace::FZ2U1SU2(lhs),
                UserBoundSpace::FZ2U1SU2(rhs),
            ) => contract_cuda_bound!(&mut state.fz2_u1_su2, dst, lhs, rhs),
            (UserBoundSpace::Su3(_), UserBoundSpace::Su3(_), UserBoundSpace::Su3(_)) => {
                return Err(Error::InvalidArgument(
                    "CUDA contraction is not yet supported for SU(3) tensors".to_string(),
                ));
            }
            _ => return Err(Error::RuleMismatch),
        }?;
        let data = Data::CudaF64(Arc::new(dst));
        drop(guard);
        self.with_bound(dst_bound, data)
    }

    /// Like [`Self::contract`], but with an explicit output axis order
    /// (`pAB`): `output_axes[i]` picks, for output position `i`, an index
    /// into the default output order (`self` open axes ascending, then
    /// `rhs` open axes ascending). The codomain/domain split of the result
    /// keeps `self`'s open-leg count on the codomain side.
    pub fn contract_ordered(
        &self,
        rhs: &Self,
        lhs_axes: &[usize],
        rhs_axes: &[usize],
        output_axes: &[usize],
    ) -> Result<Self, Error> {
        self.check_same_world(rhs)?;
        if lhs_axes.len() != rhs_axes.len() {
            return Err(Error::InvalidArgument(format!(
                "contracted axis lists differ in length: {} vs {}",
                lhs_axes.len(),
                rhs_axes.len()
            )));
        }
        validate_contracted_axes(lhs_axes, self.rank())?;
        validate_contracted_axes(rhs_axes, rhs.rank())?;
        let open_rank = self.rank() - lhs_axes.len() + rhs.rank() - rhs_axes.len();

        let host_mult_free_dense = self.rule_kind() != RuleKind::Su3
            && self.placement() == Placement::Host
            && !matches!(self.stored_data(), Data::Diagonal(_))
            && !matches!(rhs.stored_data(), Data::Diagonal(_));
        if !host_mult_free_dense {
            // Why not force generic fusion, compact diagonal, or device storage
            // through the multiplicity-free host plan: those routes have distinct
            // complexity or placement contracts. Preserve their proven sequential
            // operation, including validation order, until each backend can consume
            // pAB directly.
            let contracted = self.contract(rhs, lhs_axes, rhs_axes)?;
            if output_axes.len() != contracted.rank() {
                return Err(Error::InvalidArgument(format!(
                    "output axis list length {} does not match open rank {}",
                    output_axes.len(),
                    contracted.rank()
                )));
            }
            let split = contracted.codomain_rank();
            if output_axes.iter().copied().eq(0..contracted.rank()) {
                return Ok(contracted);
            }
            return contracted.permute(&output_axes[..split], &output_axes[split..]);
        }

        if output_axes.len() != open_rank {
            // Why not report pAB first: historically `contract_ordered` ran the
            // contraction before inspecting pAB, so an incompatible contracted
            // pair must retain precedence when both inputs are invalid. This
            // compatibility-only path is cold because valid pAB skips it.
            self.logical_space()?.validate_contracted_homspace(
                rhs.logical_space()?,
                lhs_axes,
                rhs_axes,
            )?;
            return Err(Error::InvalidArgument(format!(
                "output axis list length {} does not match open rank {}",
                output_axes.len(),
                open_rank
            )));
        }
        if output_axes.iter().copied().eq(0..open_rank) {
            return self.contract(rhs, lhs_axes, rhs_axes);
        }

        #[cfg(test)]
        observe_ordered_contract_fused_route();
        self.contract_host_fusion_impl(
            rhs,
            lhs_axes,
            rhs_axes,
            OutputAxisOrder::from_axes(output_axes),
        )
    }

    /// TensorKit `permute`: re-arranges legs with symmetric braiding.
    /// `codomain_axes` and `domain_axes` list source axis numbers
    /// (`0..rank`, codomain axes first) for the new codomain and domain.
    pub fn permute(&self, codomain_axes: &[usize], domain_axes: &[usize]) -> Result<Self, Error> {
        self.transformed(codomain_axes, domain_axes, TransformKind::Permute)
    }

    /// TensorKit `braid`: explicit braid with one level per source axis
    /// (levels decide which strand crosses above at each transposition).
    pub fn braid(
        &self,
        codomain_axes: &[usize],
        domain_axes: &[usize],
        levels: &[usize],
    ) -> Result<Self, Error> {
        self.transformed(codomain_axes, domain_axes, TransformKind::Braid { levels })
    }

    /// TensorKit `transpose`: the planar transpose `codomain <- domain`
    /// to `domain' <- codomain'`, i.e. cyclic leg rotation without
    /// braiding. Equivalent to
    /// `transpose_into` with reversed domain axes as the new codomain and
    /// reversed codomain axes as the new domain.
    pub fn transpose(&self) -> Result<Self, Error> {
        let codomain_axes: Vec<usize> = (self.codomain_rank()..self.rank()).rev().collect();
        let domain_axes: Vec<usize> = (0..self.codomain_rank()).rev().collect();
        self.transformed(&codomain_axes, &domain_axes, TransformKind::Transpose)
    }

    /// TensorKit `repartition(t, N₁, N₂)`: move the planar boundary so the
    /// codomain holds `num_codomain` legs and the domain holds the rest. The
    /// boundary order is codomain followed by reversed domain; legs which cross
    /// the boundary are bent without introducing a symmetric braid.
    pub fn repartition(&self, num_codomain: usize) -> Result<Self, Error> {
        if num_codomain > self.rank() {
            return Err(Error::InvalidArgument(format!(
                "repartition: num_codomain {num_codomain} exceeds rank {}",
                self.rank()
            )));
        }
        if num_codomain == self.codomain_rank() {
            return Ok(self.clone());
        }

        let mut axes = (0..self.codomain_rank())
            .chain((self.codomain_rank()..self.rank()).rev())
            .collect::<Vec<_>>();
        axes[num_codomain..].reverse();
        let (codomain_axes, domain_axes) = axes.split_at(num_codomain);

        // Why not identity `permute`: domain trees run opposite to the planar
        // boundary, and flattening them would braid a different leg across it.
        self.transformed(codomain_axes, domain_axes, TransformKind::Transpose)
    }

    fn transformed(
        &self,
        codomain_axes: &[usize],
        domain_axes: &[usize],
        kind: TransformKind<'_>,
    ) -> Result<Self, Error> {
        let rank = self.rank();
        let nout = self.codomain_rank();
        if let TransformKind::Braid { levels } = &kind {
            if levels.len() != rank {
                return Err(Error::InvalidArgument(format!(
                    "braid levels must list one level per source axis \
                     (expected {rank}, got {})",
                    levels.len()
                )));
            }
        }
        // Identity permutes and braids have no axis motion or adjacent braid
        // swaps, so return the tensor unchanged and share its owned storage.
        // Levels cannot contribute a phase when there is no crossing. Why not
        // include Transpose: its planar boundary/cycle semantics stay on the
        // general path; same-split repartition already has its own no-op.
        let shares_identity_storage =
            matches!(&kind, TransformKind::Permute | TransformKind::Braid { .. })
                && codomain_axes.iter().copied().eq(0..nout)
                && domain_axes.iter().copied().eq(nout..rank);
        if shares_identity_storage {
            return Ok(self.clone());
        }
        if let TensorRepr::Adjoint(view) = &self.repr {
            let parent_nout = view.parent.space.homspace().codomain().len();
            let parent_nin = view.parent.space.homspace().domain().len();
            let lowered = lower_adjoint_transform_request(
                parent_nout,
                parent_nin,
                codomain_axes,
                domain_axes,
                &kind,
            )?;
            let parent = Self::owned(
                self.rt.clone(),
                Arc::clone(&view.parent.space),
                Arc::clone(&view.parent.data),
            );
            let transformed = match kind {
                TransformKind::Permute => parent.transformed(
                    &lowered.codomain_axes,
                    &lowered.domain_axes,
                    TransformKind::Permute,
                ),
                TransformKind::Braid { .. } => parent.transformed(
                    &lowered.codomain_axes,
                    &lowered.domain_axes,
                    TransformKind::Braid {
                        levels: &lowered.levels,
                    },
                ),
                TransformKind::Transpose => parent.transformed(
                    &lowered.codomain_axes,
                    &lowered.domain_axes,
                    TransformKind::Transpose,
                ),
            }?;
            return transformed.adjoint();
        }
        let operation = match kind {
            TransformKind::Permute => TreeTransformOperation::permute(
                codomain_axes.iter().copied(),
                domain_axes.iter().copied(),
            ),
            TransformKind::Braid { levels } => TreeTransformOperation::braid(
                codomain_axes.iter().copied(),
                domain_axes.iter().copied(),
                levels[..nout].iter().copied(),
                levels[nout..].iter().copied(),
            ),
            TransformKind::Transpose => TreeTransformOperation::transpose(
                codomain_axes.iter().copied(),
                domain_axes.iter().copied(),
            ),
        };

        with_data!(self, data, self.transformed_impl(data, operation))
    }

    fn transformed_impl<D: UserScalar>(
        &self,
        src_data: &[D],
        operation: TreeTransformOperation,
    ) -> Result<Self, Error> {
        // Tree transforms use a leased context and retain the source provider
        // proof in the derived destination.
        let mut lease = self.rt.lease_context()?;
        let context = lease.context();
        let dst_bound = self.ordinary_body().space.transformed(&operation)?;
        // SU(3) (Generic): dedicated non-macro path — build the generic result
        // space and drive the non-memoized generic tree-transform. The recoupling
        // coefficient scalar is f64 for either data dtype, so the generic braid
        // math is identical to the tree-level layer this stage proved against TK.
        if self.rule_kind() == RuleKind::Su3 {
            let rule = self.su3_rule();
            let dst_space = dst_bound.raw();
            let mut data = vec![D::from_real(0.0); dst_space.required_len()?];
            D::ctx_of(&mut context.su3)
                .tree_context_mut()
                .tree_transform_dyn_into_generic(
                    rule,
                    operation,
                    &Arc::clone(dst_space.structure()),
                    self.ordinary_body().space.structure(),
                    &mut data,
                    src_data,
                    D::from_real(1.0),
                    D::from_real(0.0),
                )?;
            return self.with_bound(dst_bound, D::lift(data));
        }
        let data = with_user_rule_ctx!(self.ordinary_body().space, context, rule, ctxs, {
            let dst_space = dst_bound.raw();
            let required_len = dst_space.required_len()?;
            let owned = D::ctx_of(ctxs)
                .tree_context_mut()
                .try_tree_transform_dyn_overwrite_owned(
                    rule,
                    &operation,
                    &Arc::clone(dst_space.structure()),
                    self.ordinary_body().space.structure(),
                    dst_space.nout(),
                    src_data,
                    D::from_real(1.0),
                )?;
            let data = if let Some(data) = owned {
                data
            } else {
                let mut data = vec![D::from_real(0.0); required_len];
                D::ctx_of(ctxs).tree_context_mut().tree_transform_dyn_into(
                    rule,
                    operation,
                    &Arc::clone(dst_space.structure()),
                    self.ordinary_body().space.structure(),
                    &mut data,
                    src_data,
                    D::from_real(1.0),
                    D::from_real(0.0),
                )?;
                data
            };
            Ok::<_, Error>(D::lift(data))
        })?;
        self.with_bound(dst_bound, data)
    }

    /// Partial trace over pairs of mutually dual legs (TensorKit
    /// `tensortrace!` / TensorOperations `@tensor a[i, i; j]` semantics):
    /// each `(lhs, rhs)` pair of flat leg indices is traced, the remaining
    /// legs keep their order and codomain/domain sides. Symmetric fusion
    /// rules apply the categorical trace coefficients (quantum-dimension
    /// factors, and twists for fermionic rules: the supertrace).
    pub fn trace_pairs(&self, pairs: &[(usize, usize)]) -> Result<Self, Error> {
        if self.is_adjoint_view() {
            return self.materialized_tensor()?.trace_pairs(pairs);
        }
        // SU(N) (Generic): the partial-trace engine rides the mult-free
        // recoupling (`multiplicity_free_permute_tree_pair`); its generic
        // sibling is Stage B3c-3. Full trace (`tr`) IS wired generically.
        if self.rule_kind() == RuleKind::Su3 {
            return Err(Error::UnsupportedForRule {
                operation: "Tensor::trace_pairs",
                rule: "SU(3)",
            });
        }
        let rank = self.rank();
        let mut seen = vec![false; rank];
        for &(lhs, rhs) in pairs {
            for axis in [lhs, rhs] {
                if axis >= rank || seen[axis] {
                    return Err(Error::InvalidArgument(format!(
                        "invalid trace pair list {pairs:?} for rank {rank} \
                         (axes must be in range and distinct)"
                    )));
                }
                seen[axis] = true;
            }
        }
        let output_axes: Vec<usize> = (0..rank).filter(|&axis| !seen[axis]).collect();
        let dst_codomain_rank = output_axes
            .iter()
            .filter(|&&axis| axis < self.codomain_rank())
            .count();
        let trace_lhs: Vec<usize> = pairs.iter().map(|&(lhs, _)| lhs).collect();
        let trace_rhs: Vec<usize> = pairs.iter().map(|&(_, rhs)| rhs).collect();
        with_data!(self, data, {
            self.trace_pairs_impl(
                data,
                &output_axes,
                dst_codomain_rank,
                &trace_lhs,
                &trace_rhs,
            )
        })
    }

    fn trace_pairs_impl<D: UserScalar>(
        &self,
        src_data: &[D],
        output_axes: &[usize],
        dst_codomain_rank: usize,
        trace_lhs: &[usize],
        trace_rhs: &[usize],
    ) -> Result<Self, Error> {
        let hom = with_user_rule!(self.ordinary_body().space, rule, {
            let hom = self.ordinary_body().space.homspace().select(
                rule,
                &output_axes[..dst_codomain_rank],
                &output_axes[dst_codomain_rank..],
            )?;
            Ok::<_, Error>(hom)
        })?;
        let dst_bound = self.ordinary_body().space.from_selected_homspace(hom)?;
        let mut data = vec![D::from_real(0.0); dst_bound.raw().required_len()?];
        macro_rules! trace_bound {
            ($dst:expr, $src:expr) => {
                tenet_tensors::tensortrace_fusion_dyn_into(
                    $dst,
                    &mut data,
                    $src,
                    src_data,
                    tenet_tensors::TensorTraceAxisSpec::new(output_axes, trace_lhs, trace_rhs),
                    D::from_real(1.0),
                    D::from_real(0.0),
                )
            };
        }
        match (&dst_bound, self.ordinary_body().space.as_ref()) {
            (UserBoundSpace::U1(dst), UserBoundSpace::U1(src)) => trace_bound!(dst, src),
            (UserBoundSpace::Z2(dst), UserBoundSpace::Z2(src)) => trace_bound!(dst, src),
            (UserBoundSpace::FZ2(dst), UserBoundSpace::FZ2(src)) => trace_bound!(dst, src),
            (UserBoundSpace::SU2(dst), UserBoundSpace::SU2(src)) => trace_bound!(dst, src),
            (UserBoundSpace::U1FZ2(dst), UserBoundSpace::U1FZ2(src)) => {
                trace_bound!(dst, src)
            }
            (UserBoundSpace::FZ2U1SU2(dst), UserBoundSpace::FZ2U1SU2(src)) => {
                trace_bound!(dst, src)
            }
            (UserBoundSpace::Su3(_), UserBoundSpace::Su3(_)) => {
                unreachable!("partial SU(3) trace is rejected while selecting its homspace")
            }
            _ => return Err(Error::RuleMismatch),
        }?;
        let data = D::lift(data);
        self.with_bound(dst_bound, data)
    }

    /// TensorKit `tr`: full trace of an endomorphism (`domain == codomain`)
    /// to a scalar, pairing codomain leg `i` with domain leg `i`. The
    /// returned [`Scalar`] variant matches [`Self::dtype`]. This is TensorKit's
    /// positive/ordinary trace; [`Self::trace_pairs`] retains the fermionic
    /// supertrace semantics used by tensor contractions.
    pub fn tr(&self) -> Result<Scalar, Error> {
        if let TensorRepr::Adjoint(view) = &self.repr {
            let parent = Self::owned(
                self.rt.clone(),
                Arc::clone(&view.parent.space),
                Arc::clone(&view.parent.data),
            );
            return Ok(match parent.tr()? {
                Scalar::F64(value) => Scalar::F64(value),
                Scalar::C64(value) => Scalar::C64(value.conj()),
            });
        }
        let hom = self.ordinary_body().space.homspace();
        if hom.codomain().legs() != hom.domain().legs() {
            return Err(Error::InvalidArgument(
                "tr() requires an endomorphism (domain == codomain)".to_string(),
            ));
        }
        // Block-local weighted trace (TensorKit `tr`): sum the coupled-block
        // diagonals weighted by quantum dimension, directly on the stored
        // blocks. Avoids the generic partial-trace engine's per-call recoupling
        // compile, rank-0 destination allocation, and kernel dispatch that the
        // former `trace_pairs` route paid to produce a single scalar.
        let nout = self.codomain_rank();
        if let Data::Diagonal(diagonal) = self.stored_data() {
            let value = if self.rule_kind() == RuleKind::Su3 {
                let rule = self.su3_rule();
                diagonal.ordinary_trace_with(|sector| {
                    let sqrt = tenet_core::GenericRigidSymbols::sqrt_dim_scalar(rule, sector);
                    sqrt * sqrt
                })
            } else {
                with_user_rule!(
                    self.ordinary_body().space,
                    rule,
                    diagonal.ordinary_trace(rule)
                )
            };
            return Ok(match diagonal {
                DiagonalData::RealF64(_) => Scalar::F64(value.re),
                DiagonalData::RealC64(_) | DiagonalData::C64(_) => Scalar::C64(value),
            });
        }
        // SU(N) (Generic): same block-local weighted trace through the
        // generic-dim sibling (mult-free `with_rule!` cannot host it).
        if self.rule_kind() == RuleKind::Su3 {
            let rule = self.su3_rule();
            return match self.coupled_data()? {
                Data::F64(data) => {
                    weighted_trace_generic(rule, self.ordinary_body().space.structure(), nout, data)
                        .map(|v| Scalar::F64(v.re))
                }
                Data::C64(data) => {
                    weighted_trace_generic(rule, self.ordinary_body().space.structure(), nout, data)
                        .map(Scalar::C64)
                }
                Data::Diagonal(_) => unreachable!("coupled_data materializes Data::Diagonal"),
                #[cfg(feature = "cuda")]
                Data::CudaF64(_) => Err(device_unsupported("tr()")),
            };
        }
        match self.coupled_data()? {
            Data::F64(data) => with_user_rule!(self.ordinary_body().space, rule, {
                weighted_trace(rule, self.ordinary_body().space.structure(), nout, data)
                    .map(|v| Scalar::F64(v.re))
            }),
            Data::C64(data) => with_user_rule!(self.ordinary_body().space, rule, {
                weighted_trace(rule, self.ordinary_body().space.structure(), nout, data)
                    .map(Scalar::C64)
            }),
            Data::Diagonal(_) => unreachable!("coupled_data materializes Data::Diagonal"),
            #[cfg(feature = "cuda")]
            Data::CudaF64(_) => Err(device_unsupported("tr()")),
        }
    }

    /// TensorKit `adjoint` (dagger): swaps codomain and domain and
    /// conjugate-transposes every block (real scalars: transpose only, c64:
    /// entries conjugated).
    ///
    /// Lazy, exactly like TensorKit's `AdjointTensorMap`: no data is copied or
    /// conjugated here. The result shares the parent buffer and reverses the
    /// borrowed space orientation in O(1). Consumers not yet lowered for the
    /// view (`data`, `svd`, ...) materialize one shared owned adjoint on demand.
    /// Compact diagonal storage is handled directly and does not use this lazy
    /// dense-adjoint route.
    pub fn adjoint(&self) -> Result<Self, Error> {
        if let Data::Diagonal(diagonal) = self.stored_data() {
            // Why not use the lazy dense-adjoint wrapper: real compact spectra
            // are self-adjoint and can share their Data Arc; only genuinely
            // complex entries require an owned O(r) conjugated result.
            return Ok(match diagonal {
                DiagonalData::RealF64(_) | DiagonalData::RealC64(_) => self.clone(),
                DiagonalData::C64(_) => self.with_diagonal(diagonal.conjugated_complex()?),
            });
        }
        Ok(match &self.repr {
            TensorRepr::Owned(parent) => Self {
                rt: self.rt.clone(),
                repr: TensorRepr::Adjoint(Arc::new(AdjointView {
                    parent: parent.clone(),
                    logical_space: OnceLock::new(),
                    materialized: OnceLock::new(),
                    init: Mutex::new(()),
                    #[cfg(test)]
                    logical_space_builds: std::sync::atomic::AtomicUsize::new(0),
                    #[cfg(test)]
                    materialized_body_builds: std::sync::atomic::AtomicUsize::new(0),
                })),
                compact_dense: OnceLock::new(),
            },
            TensorRepr::Adjoint(view) => Self {
                rt: self.rt.clone(),
                repr: TensorRepr::Owned(view.parent.clone()),
                compact_dense: OnceLock::new(),
            },
        })
    }

    /// Frobenius norm, weighted by coupled-sector quantum dimensions
    /// (`norm(t)^2 = sum_c dim(c) * |block_c|^2`), matching TensorKit's
    /// `norm`. Always real, for both dtypes.
    pub fn norm(&self) -> Result<f64, Error> {
        if let TensorRepr::Adjoint(view) = &self.repr {
            return Self::owned(
                self.rt.clone(),
                Arc::clone(&view.parent.space),
                Arc::clone(&view.parent.data),
            )
            .norm();
        }
        #[cfg(feature = "cuda")]
        if let Data::CudaF64(storage) = self.stored_data() {
            return Ok(self.weighted_inner_cuda(storage, storage)?.re.sqrt());
        }
        if self.rule_kind() != RuleKind::Su3 {
            if let Data::Diagonal(diagonal) = self.stored_data() {
                let value = with_user_rule!(self.ordinary_body().space, rule, {
                    match diagonal {
                        DiagonalData::RealF64(spectrum) => {
                            compact_inner(rule, spectrum, spectrum, |value| value, |value| value)
                        }
                        DiagonalData::RealC64(spectrum) => compact_inner(
                            rule,
                            spectrum,
                            spectrum,
                            |value| Complex64::new(value, 0.0),
                            |value| Complex64::new(value, 0.0),
                        ),
                        DiagonalData::C64(spectrum) => {
                            compact_inner(rule, spectrum, spectrum, |value| value, |value| value)
                        }
                    }
                })
                .ok_or_else(|| {
                    internal_layout_error("a diagonal spectrum is incompatible with itself")
                })?;
                return Ok(value.re.sqrt());
            }
        }
        // SU(N) (Generic): dedicated non-macro path — the Frobenius norm is a
        // storage-level block sum weighted by dim(c) = sqrt_dim(c)², so it needs
        // only `GenericRigidSymbols`, no contract. Sums over OM vertices.
        if self.rule_kind() == RuleKind::Su3 {
            let value = with_data!(self, data, {
                weighted_inner_generic(
                    self.su3_rule(),
                    self.ordinary_body().space.structure(),
                    self.ordinary_body().space.nout(),
                    data,
                    data,
                )
            })?;
            return Ok(value.re.sqrt());
        }
        let value = with_data!(self, data, {
            with_user_rule!(self.ordinary_body().space, rule, {
                weighted_inner(
                    rule,
                    self.ordinary_body().space.structure(),
                    self.ordinary_body().space.nout(),
                    data,
                    data,
                )
            })
        })?;
        Ok(value.re.sqrt())
    }

    /// Entrywise infinity norm over TensorKit tensor blocks:
    /// `maximum(norm(block, Inf) for block in blocks(t))`.
    ///
    /// Julia's `norm(array, Inf)` is the maximum absolute element, including
    /// for matrices. TensorKit applies that to each block, so the coupled
    /// storage equivalent is the maximum absolute stored entry. Unlike
    /// [`Self::norm`], this is not quantum-dimension weighted.
    pub fn norm_inf(&self) -> Result<f64, Error> {
        #[cfg(feature = "cuda")]
        if let Data::CudaF64(_) = self.stored_data() {
            return Err(device_unsupported("norm_inf()"));
        }
        if let Data::Diagonal(diagonal) = self.stored_data() {
            return Ok(diagonal.max_abs());
        }
        match self.coupled_data()? {
            Data::F64(data) => Ok(data.iter().map(|value| value.abs()).fold(0.0, f64::max)),
            Data::C64(data) => Ok(data.iter().map(|value| value.norm()).fold(0.0, f64::max)),
            Data::Diagonal(_) => unreachable!("coupled_data materializes Data::Diagonal"),
            #[cfg(feature = "cuda")]
            Data::CudaF64(_) => unreachable!("returned above"),
        }
    }

    /// Returns `factor * self` (real factor, both dtypes). Use
    /// [`Self::scale_c64`] for a complex factor.
    pub fn scale(&self, factor: f64) -> Result<Self, Error> {
        // Scaling a diagonal stays diagonal (O(rank)); itebd normalizes λ this
        // way, and keeping it diagonal lets the next contract scale the bond.
        if let Data::Diagonal(diagonal) = self.stored_data() {
            return Ok(self.with_diagonal(diagonal.scaled(factor)));
        }
        let data = match self.coupled_data()? {
            Data::F64(data) => Data::F64(data.iter().map(|&value| value * factor).collect()),
            Data::C64(data) => Data::C64(data.iter().map(|&value| value * factor).collect()),
            Data::Diagonal(_) => unreachable!("coupled_data materializes Data::Diagonal"),
            #[cfg(feature = "cuda")]
            Data::CudaF64(storage) => {
                Data::CudaF64(Arc::new(self.axpby_cuda(factor, storage, None)?))
            }
        };
        Ok(Self::owned(
            self.rt.clone(),
            Arc::clone(&self.materialized_body()?.space),
            Arc::new(data),
        ))
    }

    /// Returns `factor * self` for a c64 tensor. Errors with
    /// [`Error::DtypeMismatch`] on f64 tensors (widen with
    /// [`Self::to_c64`] first).
    pub fn scale_c64(&self, factor: Complex64) -> Result<Self, Error> {
        if let Data::Diagonal(diagonal) = self.stored_data() {
            return Ok(self.with_diagonal(diagonal.scaled_c64(factor)?));
        }
        match self.coupled_data()? {
            Data::C64(data) => Ok(Self::owned(
                self.rt.clone(),
                Arc::clone(&self.materialized_body()?.space),
                Arc::new(Data::C64(
                    data.iter().map(|&value| value * factor).collect(),
                )),
            )),
            Data::F64(_) => Err(Error::DtypeMismatch),
            Data::Diagonal(_) => unreachable!("coupled_data materializes Data::Diagonal"),
            #[cfg(feature = "cuda")]
            Data::CudaF64(_) => Err(device_unsupported("scale_c64()")),
        }
    }

    /// Returns `alpha * self + beta * other` (real coefficients, both
    /// dtypes). Both tensors must live on the same spaces (identical hom
    /// space and block layout) and store the same dtype.
    pub fn add(&self, other: &Self, alpha: f64, beta: f64) -> Result<Self, Error> {
        if self.is_adjoint_view() || other.is_adjoint_view() {
            return self
                .materialized_tensor()?
                .add(&other.materialized_tensor()?, alpha, beta);
        }
        self.check_same_space(other)?;
        match (self.diagonal_data(), other.diagonal_data()) {
            (Some(lhs), Some(rhs)) => {
                let data = lhs.axpby_real(rhs, alpha, beta).ok_or_else(|| {
                    internal_layout_error("equal diagonal spaces carry incompatible spectra")
                })?;
                return Ok(self.with_diagonal(data));
            }
            (Some(diagonal), None) => {
                // Why not materialize `diagonal`: the owned dense result is the
                // only O(n²) allocation required by diagonal+dense addition.
                let data = axpby_dense_real(
                    &self.ordinary_body().space,
                    other.coupled_data()?,
                    diagonal,
                    beta,
                    alpha,
                )?;
                return Ok(self.with_same_data(data));
            }
            (None, Some(diagonal)) => {
                let data = axpby_dense_real(
                    &self.ordinary_body().space,
                    self.coupled_data()?,
                    diagonal,
                    alpha,
                    beta,
                )?;
                return Ok(self.with_same_data(data));
            }
            (None, None) => {}
        }
        let data = match (self.coupled_data()?, other.coupled_data()?) {
            (Data::F64(a), Data::F64(b)) => Data::F64(
                a.iter()
                    .zip(b)
                    .map(|(&x, &y)| alpha * x + beta * y)
                    .collect(),
            ),
            (Data::C64(a), Data::C64(b)) => Data::C64(
                a.iter()
                    .zip(b)
                    .map(|(&x, &y)| x * alpha + y * beta)
                    .collect(),
            ),
            #[cfg(feature = "cuda")]
            (Data::CudaF64(a), Data::CudaF64(b)) => {
                Data::CudaF64(Arc::new(self.axpby_cuda(alpha, a, Some((beta, b)))?))
            }
            _ => return Err(Error::DtypeMismatch),
        };
        Ok(Self::owned(
            self.rt.clone(),
            Arc::clone(&self.materialized_body()?.space),
            Arc::new(data),
        ))
    }

    /// Returns `alpha * self + beta * other` with complex coefficients; both
    /// tensors must be c64 (widen with [`Self::to_c64`] first).
    pub fn add_c64(&self, other: &Self, alpha: Complex64, beta: Complex64) -> Result<Self, Error> {
        if self.is_adjoint_view() || other.is_adjoint_view() {
            return self
                .materialized_tensor()?
                .add_c64(&other.materialized_tensor()?, alpha, beta);
        }
        self.check_same_space(other)?;
        match (self.diagonal_data(), other.diagonal_data()) {
            (Some(lhs), Some(rhs)) => {
                let data = lhs
                    .axpby_c64(rhs, alpha, beta)
                    .ok_or(Error::DtypeMismatch)?;
                return Ok(self.with_diagonal(data));
            }
            (Some(diagonal), None) => {
                let data = axpby_dense_c64(
                    &self.ordinary_body().space,
                    other.coupled_data()?,
                    diagonal,
                    beta,
                    alpha,
                )?;
                return Ok(self.with_same_data(data));
            }
            (None, Some(diagonal)) => {
                let data = axpby_dense_c64(
                    &self.ordinary_body().space,
                    self.coupled_data()?,
                    diagonal,
                    alpha,
                    beta,
                )?;
                return Ok(self.with_same_data(data));
            }
            (None, None) => {}
        }
        match (self.coupled_data()?, other.coupled_data()?) {
            (Data::C64(a), Data::C64(b)) => Ok(Self::owned(
                self.rt.clone(),
                Arc::clone(&self.materialized_body()?.space),
                Arc::new(Data::C64(
                    a.iter()
                        .zip(b)
                        .map(|(&x, &y)| alpha * x + beta * y)
                        .collect(),
                )),
            )),
            #[cfg(feature = "cuda")]
            (Data::CudaF64(_), _) | (_, Data::CudaF64(_)) => Err(device_unsupported("add_c64()")),
            _ => Err(Error::DtypeMismatch),
        }
    }

    /// Frobenius inner product `<self, other>` with `self` conjugated,
    /// weighted by coupled-sector quantum dimensions, matching TensorKit's
    /// `dot(x, y)`. The returned [`Scalar`] variant matches the operands'
    /// dtype: f64 tensors give `Scalar::F64` (the result is exactly real),
    /// so `t.inner(&t)?.re() == t.norm()?.powi(2)` up to floating-point
    /// error. Both tensors must live on the same spaces and store the same
    /// dtype.
    pub fn inner(&self, other: &Self) -> Result<Scalar, Error> {
        if self.is_adjoint_view() || other.is_adjoint_view() {
            return self
                .materialized_tensor()?
                .inner(&other.materialized_tensor()?);
        }
        self.check_same_space(other)?;
        self.reject_unwired_su3("Tensor::inner")?;
        match (self.diagonal_data(), other.diagonal_data()) {
            (Some(lhs), Some(rhs)) => {
                let value = with_user_rule!(self.ordinary_body().space, rule, {
                    match (lhs, rhs) {
                        (DiagonalData::RealF64(lhs), DiagonalData::RealF64(rhs)) => {
                            compact_inner(rule, lhs, rhs, |value| value, |value| value)
                                .map(|value| Scalar::F64(value.re))
                        }
                        (DiagonalData::RealC64(lhs), DiagonalData::RealC64(rhs)) => compact_inner(
                            rule,
                            lhs,
                            rhs,
                            |value| Complex64::new(value, 0.0),
                            |value| Complex64::new(value, 0.0),
                        )
                        .map(Scalar::C64),
                        (DiagonalData::RealC64(lhs), DiagonalData::C64(rhs)) => compact_inner(
                            rule,
                            lhs,
                            rhs,
                            |value| Complex64::new(value, 0.0),
                            |value| value,
                        )
                        .map(Scalar::C64),
                        (DiagonalData::C64(lhs), DiagonalData::RealC64(rhs)) => compact_inner(
                            rule,
                            lhs,
                            rhs,
                            |value| value,
                            |value| Complex64::new(value, 0.0),
                        )
                        .map(Scalar::C64),
                        (DiagonalData::C64(lhs), DiagonalData::C64(rhs)) => {
                            compact_inner(rule, lhs, rhs, |value| value, |value| value)
                                .map(Scalar::C64)
                        }
                        _ => None,
                    }
                })
                .ok_or(Error::DtypeMismatch)?;
                return Ok(value);
            }
            (Some(diagonal), None) => {
                let value = with_user_rule!(self.ordinary_body().space, rule, {
                    match (diagonal, other.coupled_data()?) {
                        (DiagonalData::RealF64(spectrum), Data::F64(dense)) => dense_inner(
                            rule,
                            &self.ordinary_body().space,
                            spectrum,
                            dense,
                            true,
                            |value| value,
                        )
                        .map(|value| Scalar::F64(value.re)),
                        (DiagonalData::RealC64(spectrum), Data::C64(dense)) => dense_inner(
                            rule,
                            &self.ordinary_body().space,
                            spectrum,
                            dense,
                            true,
                            |value| Complex64::new(value, 0.0),
                        )
                        .map(Scalar::C64),
                        (DiagonalData::C64(spectrum), Data::C64(dense)) => dense_inner(
                            rule,
                            &self.ordinary_body().space,
                            spectrum,
                            dense,
                            true,
                            |value| value,
                        )
                        .map(Scalar::C64),
                        _ => Err(Error::DtypeMismatch),
                    }
                })?;
                return Ok(value);
            }
            (None, Some(diagonal)) => {
                let value = with_user_rule!(self.ordinary_body().space, rule, {
                    match (self.coupled_data()?, diagonal) {
                        (Data::F64(dense), DiagonalData::RealF64(spectrum)) => dense_inner(
                            rule,
                            &self.ordinary_body().space,
                            spectrum,
                            dense,
                            false,
                            |value| value,
                        )
                        .map(|value| Scalar::F64(value.re)),
                        (Data::C64(dense), DiagonalData::RealC64(spectrum)) => dense_inner(
                            rule,
                            &self.ordinary_body().space,
                            spectrum,
                            dense,
                            false,
                            |value| Complex64::new(value, 0.0),
                        )
                        .map(Scalar::C64),
                        (Data::C64(dense), DiagonalData::C64(spectrum)) => dense_inner(
                            rule,
                            &self.ordinary_body().space,
                            spectrum,
                            dense,
                            false,
                            |value| value,
                        )
                        .map(Scalar::C64),
                        _ => Err(Error::DtypeMismatch),
                    }
                })?;
                return Ok(value);
            }
            (None, None) => {}
        }
        match (self.coupled_data()?, other.coupled_data()?) {
            (Data::F64(a), Data::F64(b)) => with_user_rule!(self.ordinary_body().space, rule, {
                weighted_inner(
                    rule,
                    self.ordinary_body().space.structure(),
                    self.ordinary_body().space.nout(),
                    a,
                    b,
                )
                .map(|v| Scalar::F64(v.re))
            }),
            (Data::C64(a), Data::C64(b)) => with_user_rule!(self.ordinary_body().space, rule, {
                weighted_inner(
                    rule,
                    self.ordinary_body().space.structure(),
                    self.ordinary_body().space.nout(),
                    a,
                    b,
                )
                .map(Scalar::C64)
            }),
            #[cfg(feature = "cuda")]
            (Data::CudaF64(a), Data::CudaF64(b)) => {
                self.weighted_inner_cuda(a, b).map(|v| Scalar::F64(v.re))
            }
            _ => Err(Error::DtypeMismatch),
        }
    }

    /// Frobenius inner product `<self, other>` with `self` conjugated — an
    /// alias for [`Self::inner`], matching `LinearAlgebra.dot` / TensorKit's
    /// `dot(x, y)`. Provided for callers who reach for the `dot` name; the
    /// semantics (conjugate-linear in the first argument, quantum-dimension
    /// weighted) are identical.
    ///
    /// ```
    /// use tenet::prelude::*;
    ///
    /// let rt = Runtime::builder().build()?;
    /// let v = Space::u1([(0, 2), (1, 1)]);
    /// let t = Tensor::rand(&rt, Dtype::F64, [&v], [&v])?;
    /// assert_eq!(t.dot(&t)?.re(), t.inner(&t)?.re());
    /// # Ok::<(), tenet::prelude::Error>(())
    /// ```
    pub fn dot(&self, other: &Self) -> Result<Scalar, Error> {
        self.inner(other)
    }

    /// Returns `self / norm(self)`, the unit-norm tensor pointing the same way
    /// (TensorKit's `normalize`, LinearAlgebra's 2-norm normalization). The
    /// norm is the quantum-dimension-weighted Frobenius norm from
    /// [`Self::norm`]; the result satisfies `t.normalize()?.norm()? == 1`.
    /// Works for both dtypes (a c64 tensor is scaled by the real reciprocal
    /// norm).
    ///
    /// Like TensorKit, a zero-norm tensor is not special-cased: normalizing it
    /// divides by zero and yields non-finite entries. Guard the caller if that
    /// input is reachable.
    ///
    /// ```
    /// use tenet::prelude::*;
    ///
    /// let rt = Runtime::builder().build()?;
    /// let v = Space::u1([(0, 2), (1, 1)]);
    /// let t = Tensor::rand(&rt, Dtype::F64, [&v], [&v])?;
    /// let unit = t.normalize()?;
    /// assert!((unit.norm()? - 1.0).abs() < 1e-12);
    /// # Ok::<(), tenet::prelude::Error>(())
    /// ```
    pub fn normalize(&self) -> Result<Self, Error> {
        self.scale(1.0 / self.norm()?)
    }

    /// Tests whether the tensor equals its own adjoint within `tol`, relative
    /// to its norm (TensorKit `ishermitian`). Non-endomorphisms (codomain and
    /// domain spaces differ) are never Hermitian and return `false` without
    /// error, unlike TensorKit which throws — the predicate form is friendlier.
    pub fn is_hermitian(&self, tol: f64) -> Result<bool, Error> {
        if self.codomain_spaces() != self.domain_spaces() {
            return Ok(false);
        }
        let diff = self.add(&self.adjoint()?, 1.0, -1.0)?.norm()?;
        Ok(diff <= tol * self.norm()?.max(1.0))
    }

    /// Tests whether `adjoint(t) ∘ t` is the identity on the domain within
    /// `tol` (TensorKit `isisometric`): the columns are orthonormal. Works for
    /// any rectangular shape with `codomain_dim >= domain_dim`.
    pub fn is_isometric(&self, tol: f64) -> Result<bool, Error> {
        let gram = self.adjoint()?.compose(self)?;
        let identity = Self::id(&self.rt, self.dtype(), &self.domain_spaces())?;
        Ok(gram.add(&identity, 1.0, -1.0)?.norm()? <= tol * gram.norm()?.max(1.0))
    }

    /// Tests whether the tensor is unitary within `tol` (TensorKit
    /// `isunitary`): isometric in both directions, i.e. `adjoint(t) ∘ t` and
    /// `t ∘ adjoint(t)` are both identities.
    pub fn is_unitary(&self, tol: f64) -> Result<bool, Error> {
        Ok(self.is_isometric(tol)? && self.adjoint()?.is_isometric(tol)?)
    }

    /// Tests whether the tensor is Hermitian and positive definite (TensorKit
    /// `isposdef`, which is Cholesky-based and strict): every Hermitian
    /// eigenvalue must exceed `tol * max(norm, 1)`. Positive *semi*definite
    /// spectra (an eigenvalue at zero) return `false`; with `tol = 0.0` the
    /// check is exact strict positivity up to floating point.
    pub fn is_posdef(&self, tol: f64) -> Result<bool, Error> {
        if !self.is_hermitian(tol)? {
            return Ok(false);
        }
        let threshold = tol * self.norm()?.max(1.0);
        Ok(self
            .eigh_vals()?
            .iter()
            .flat_map(|spectrum| spectrum.values.iter())
            .all(|&lambda| lambda > threshold))
    }

    /// Tests whether the tensor equals minus its own adjoint within `tol`,
    /// relative to its norm (TensorKit `isantihermitian`). Non-endomorphisms
    /// return `false` without error (cf. [`Self::is_hermitian`]).
    pub fn is_antihermitian(&self, tol: f64) -> Result<bool, Error> {
        if self.codomain_spaces() != self.domain_spaces() {
            return Ok(false);
        }
        let sum = self.add(&self.adjoint()?, 1.0, 1.0)?.norm()?;
        Ok(sum <= tol * self.norm()?.max(1.0))
    }

    /// The Hermitian part `(t + t†)/2` (TensorKit `project_hermitian`), the
    /// nearest Hermitian tensor. Requires an endomorphism.
    pub fn project_hermitian(&self) -> Result<Self, Error> {
        self.add(&self.adjoint()?, 0.5, 0.5)
    }

    /// The anti-Hermitian part `(t - t†)/2` (TensorKit `project_antihermitian`).
    /// Requires an endomorphism.
    pub fn project_antihermitian(&self) -> Result<Self, Error> {
        self.add(&self.adjoint()?, 0.5, -0.5)
    }

    fn check_same_space(&self, other: &Self) -> Result<(), Error> {
        self.check_same_world(other)?;
        if self.logical_space()? != other.logical_space()? {
            return Err(Error::InvalidArgument(
                "tensors live on different spaces or block layouts".to_string(),
            ));
        }
        Ok(())
    }

    /// Stops Generic rules before they reach multiplicity-free-only dispatch.
    fn reject_unwired_su3(&self, operation: &'static str) -> Result<(), Error> {
        if self.rule_kind() == RuleKind::Su3 {
            return Err(Error::UnsupportedForRule {
                operation,
                rule: "SU(3)",
            });
        }
        Ok(())
    }

    // -----------------------------------------------------------------------
    // Decompositions and matrix functions (TensorKit 0.17 / MatrixAlgebraKit
    // names, transparently over the tenet-matrixalgebra dynamic cores).
    // -----------------------------------------------------------------------

    fn from_bound_factor<R, D>(&self, factor: BoundDynFactor<R, D>) -> Result<Self, Error>
    where
        R: IntoUserBoundDynamicSpace,
        D: UserScalar,
    {
        let (space, data) = factor.into_parts();
        Ok(Self::owned(
            self.rt.clone(),
            Arc::new(UserBoundSpace::from_bound(
                self.materialized_body()?.space.as_ref(),
                space,
            )?),
            Arc::new(D::lift(data)),
        ))
    }

    fn from_bound_factors<R, D>(
        &self,
        factors: (
            BoundDynFactor<R, D>,
            BoundDynFactor<R, D>,
            Vec<SectorSpectrum>,
        ),
        complex: bool,
    ) -> Result<(Self, Self, Self), Error>
    where
        R: IntoUserBoundDynamicSpace,
        D: UserScalar,
    {
        let (u, vh, spectrum) = factors;
        Ok((
            self.from_bound_factor(u)?,
            self.from_diagonal_real_spectrum(spectrum, complex)?,
            self.from_bound_factor(vh)?,
        ))
    }

    fn from_svd_trunc_dyn<R, D>(
        &self,
        output: tenet_matrixalgebra::SvdTruncDyn<R, D>,
        complex: bool,
    ) -> Result<SvdTrunc, Error>
    where
        R: IntoUserBoundDynamicSpace,
        D: UserScalar,
    {
        let (u, _s, vh, singular_values, error) = output.into_parts();
        Ok(SvdTrunc {
            u: self.from_bound_factor(u)?,
            s: self.from_diagonal_real_spectrum(singular_values.clone(), complex)?,
            vh: self.from_bound_factor(vh)?,
            singular_values,
            error,
        })
    }

    /// Wraps a real per-sector spectrum (svd `S`, eigh `D`) as a diagonal-storage
    /// tensor: the bond space is built eagerly, but the values stay O(rank) in
    /// `Data::Diagonal` instead of a dense O(rank²) block-diagonal buffer (issue
    /// #55). `complex` preserves the public dtype: a complex input yields a
    /// complex-valued but real-magnitude `S` (`RealC64`), while a real input
    /// yields `RealF64`.
    fn from_diagonal_real_spectrum(
        &self,
        mut spectrum: Vec<SectorSpectrum<f64>>,
        complex: bool,
    ) -> Result<Self, Error> {
        spectrum.sort_unstable_by_key(|entry| entry.sector);
        // SU(N) (Generic): the bond space is a rank-1/rank-1 hom whose trees
        // are trivial, but the key enumeration must still be the generic one.
        let space = if let UserBoundSpace::Su3(bound) = self.ordinary_body().space.as_ref() {
            let space = tenet_matrixalgebra::diagonal_bond_bound_space_generic(
                Arc::clone(bound.provider_arc()),
                &spectrum,
            )?;
            UserBoundSpace::from_bound(self.ordinary_body().space.as_ref(), space)?
        } else {
            with_bound_multiplicity_free!(self.ordinary_body().space, bound, {
                let space = tenet_matrixalgebra::diagonal_bond_bound_space_like(bound, &spectrum)?;
                UserBoundSpace::from_bound(self.ordinary_body().space.as_ref(), space)
            })?
        };
        let data = if complex {
            DiagonalData::RealC64(spectrum)
        } else {
            DiagonalData::RealF64(spectrum)
        };
        Ok(Self::owned(
            self.rt.clone(),
            Arc::new(space),
            Arc::new(Data::Diagonal(data)),
        ))
    }

    /// Wraps a complex per-sector spectrum (eig `D`) as diagonal storage. The
    /// general eigendecomposition is complex-valued even for real input, so `d`
    /// is always c64 and stays compact through block-local scaling/products.
    fn from_diagonal_complex_spectrum(
        &self,
        mut spectrum: Vec<SectorSpectrum<Complex64>>,
    ) -> Result<Self, Error> {
        spectrum.sort_unstable_by_key(|entry| entry.sector);
        with_bound_multiplicity_free!(self.ordinary_body().space, bound, {
            let space = tenet_matrixalgebra::diagonal_bond_bound_space_like(bound, &spectrum)?;
            let space = UserBoundSpace::from_bound(self.ordinary_body().space.as_ref(), space)?;
            self.with_bound(space, Data::Diagonal(DiagonalData::C64(spectrum)))
        })
    }

    fn with_same_data(&self, data: Data) -> Self {
        Self::owned(
            self.rt.clone(),
            Arc::clone(&self.ordinary_body().space),
            Arc::new(data),
        )
    }

    /// Reuse this tensor's space with a new diagonal payload (elementwise
    /// scale/inv/pinv/sqrt keep the same bond space).
    fn with_diagonal(&self, data: DiagonalData) -> Self {
        Self::owned(
            self.rt.clone(),
            Arc::clone(&self.ordinary_body().space),
            Arc::new(Data::Diagonal(data)),
        )
    }

    fn diagonal_data(&self) -> Option<&DiagonalData> {
        match self.stored_data() {
            Data::Diagonal(diagonal) => Some(diagonal),
            _ => None,
        }
    }

    fn is_diagonal_bond_space(space: &DynamicFusionMapSpace) -> bool {
        let homspace = space.homspace();
        space.nout() == 1
            && space.nin() == 1
            && homspace.codomain().legs() == homspace.domain().legs()
    }

    fn contraction_output_space(
        &self,
        rhs: &Self,
        lhs_axes: &[usize],
        rhs_axes: &[usize],
    ) -> Result<UserBoundSpace, Error> {
        self.logical_space()?
            .contracted(rhs.logical_space()?, lhs_axes, rhs_axes)
    }

    fn external_axis_is_dual(&self, axis: usize) -> Result<bool, Error> {
        self.logical_space()?
            .homspace()
            .external_axis_is_dual(axis)
            .ok_or_else(|| {
                Error::InvalidArgument(format!(
                    "axis {axis} is out of range for rank {}",
                    self.rank()
                ))
            })
    }

    /// Folds the supertrace twist into compact values. Why not call `twist` on
    /// a temporary tensor: this helper already has the spectrum and avoids an
    /// extra compact result allocation before contraction.
    fn twist_folded_diagonal(&self, diagonal: &DiagonalData, apply: bool) -> DiagonalData {
        if !apply {
            return diagonal.clone();
        }
        fn fold<V: Copy>(
            spectrum: &[SectorSpectrum<V>],
            factor: impl Fn(SectorId, V) -> V,
        ) -> Vec<SectorSpectrum<V>> {
            spectrum
                .iter()
                .map(|entry| SectorSpectrum {
                    sector: entry.sector,
                    values: entry
                        .values
                        .iter()
                        .copied()
                        .map(|value| factor(entry.sector, value))
                        .collect(),
                })
                .collect()
        }
        with_user_rule!(self.ordinary_body().space, rule, {
            match diagonal {
                DiagonalData::RealF64(spectrum) => {
                    DiagonalData::RealF64(fold(spectrum, |sector, value| {
                        value * rule.twist_scalar(sector)
                    }))
                }
                DiagonalData::RealC64(spectrum) => {
                    DiagonalData::RealC64(fold(spectrum, |sector, value| {
                        value * rule.twist_scalar(sector)
                    }))
                }
                DiagonalData::C64(spectrum) => {
                    DiagonalData::C64(fold(spectrum, |sector, value| {
                        value * rule.twist_scalar(sector)
                    }))
                }
            }
        })
    }

    /// Why not materialize a diagonal matrix: TensorKit `lmul!`/`rmul!` only
    /// scales the selected block-local axis for every compact scalar variant.
    fn scaled_axis_copy_diagonal(
        &self,
        axis: Option<usize>,
        diagonal: &DiagonalData,
    ) -> Result<Self, Error> {
        let space = Arc::clone(&self.materialized_body()?.space);
        match (self.coupled_data()?, diagonal) {
            (Data::F64(data), DiagonalData::RealF64(spectrum)) => {
                let mut buf = data.clone();
                tenet_matrixalgebra::scale_axis_by_spectrum_mapped(
                    &space,
                    &mut buf,
                    axis,
                    spectrum,
                    |value| value,
                )?;
                Ok(Self::owned(
                    self.rt.clone(),
                    Arc::clone(&space),
                    Arc::new(Data::F64(buf)),
                ))
            }
            (Data::C64(data), DiagonalData::RealC64(spectrum)) => {
                let mut buf = data.clone();
                tenet_matrixalgebra::scale_axis_by_spectrum_mapped(
                    &space,
                    &mut buf,
                    axis,
                    spectrum,
                    |value| Complex64::new(value, 0.0),
                )?;
                Ok(Self::owned(
                    self.rt.clone(),
                    Arc::clone(&space),
                    Arc::new(Data::C64(buf)),
                ))
            }
            (Data::C64(data), DiagonalData::C64(spectrum)) => {
                let mut buf = data.clone();
                tenet_matrixalgebra::scale_axis_by_spectrum_mapped(
                    &space,
                    &mut buf,
                    axis,
                    spectrum,
                    |value| value,
                )?;
                Ok(Self::owned(
                    self.rt.clone(),
                    Arc::clone(&space),
                    Arc::new(Data::C64(buf)),
                ))
            }
            (Data::F64(_) | Data::C64(_), _) => Err(Error::DtypeMismatch),
            (Data::Diagonal(_), _) => Err(Error::InvalidArgument(
                "internal: diagonal scaling requires a non-diagonal operand".to_string(),
            )),
            #[cfg(feature = "cuda")]
            (Data::CudaF64(_), _) => Err(device_unsupported("diagonal scaling")),
        }
    }

    /// Compact SVD `t = u * s * vh` (MatrixAlgebraKit `svd_compact`):
    /// per coupled sector the bond is `min(rows, cols)`.
    pub fn svd_compact(&self) -> Result<(Self, Self, Self), Error> {
        if self.is_adjoint_view() {
            return self.materialized_tensor()?.svd_compact();
        }
        #[cfg(feature = "cuda")]
        if let Data::CudaF64(storage) = self.stored_data() {
            let out = self.svd_cuda(storage, None)?;
            return Ok((out.u, out.s, out.vh));
        }
        let complex = self.dtype() == Dtype::C64;
        // Lease a dense executor for this op instead of the coarse runtime lock,
        // so concurrent factorizations on a shared runtime run in parallel
        // (#155); byte-identical single-threaded.
        let mut dense = self.rt.lease_dense();
        with_data!(self, data, {
            // SU(N) (Generic): the block-level SVD engine is symmetry-agnostic;
            // only the factor-space builders differ (multiplicity-aware keys).
            if let UserBoundSpace::Su3(bound) = self.ordinary_body().space.as_ref() {
                let factors = tenet_matrixalgebra::svd_compact_factors_dyn_generic(
                    dense.dense(),
                    &BoundDynamicTensorRef::try_new(&bound, data)?,
                )?;
                self.from_bound_factors(factors, complex)
            } else {
                with_bound_multiplicity_free!(self.ordinary_body().space, bound, {
                    let factors = tenet_matrixalgebra::svd_compact_factors_dyn(
                        dense.dense(),
                        &BoundDynamicTensorRef::try_new(&bound, data)?,
                    )?;
                    self.from_bound_factors(factors, complex)
                })
            }
        })
    }

    /// Full SVD `t = u * s * vh` (MatrixAlgebraKit `svd_full`): square
    /// unitaries per sector, rectangular diagonal `s`.
    pub fn svd_full(&self) -> Result<(Self, Self, Self), Error> {
        if self.is_adjoint_view() {
            return self.materialized_tensor()?.svd_full();
        }
        // Why not dispatch SU(3): the square-unitary completion path has no
        // generic sibling yet. Compact and truncated SVD are supported, but
        // silently using the multiplicity-free builder would produce an
        // invalid generic fusion-tree space.
        if self.rule_kind() == RuleKind::Su3 {
            return Err(Error::UnsupportedForRule {
                operation: "Tensor::svd_full",
                rule: "SU(3)",
            });
        }
        // Lease a dense executor for this op instead of the coarse runtime lock,
        // so concurrent factorizations on a shared runtime run in parallel
        // (#155); byte-identical single-threaded.
        let mut dense = self.rt.lease_dense();
        with_data!(self, data, {
            with_bound_multiplicity_free!(self.ordinary_body().space, bound, {
                let out = tenet_matrixalgebra::svd_full_dyn(
                    dense.dense(),
                    &BoundDynamicTensorRef::try_new(bound, data)?,
                )?;
                let (u, s, vh, _) = out.into_parts();
                Ok::<_, Error>((
                    self.from_bound_factor(u)?,
                    self.from_bound_factor(s)?,
                    self.from_bound_factor(vh)?,
                ))
            })
        })
    }

    /// Truncated SVD (MatrixAlgebraKit `svd_trunc`); see [`SvdTrunc`].
    pub fn svd_trunc(&self, truncation: &Truncation) -> Result<SvdTrunc, Error> {
        if self.is_adjoint_view() {
            return self.materialized_tensor()?.svd_trunc(truncation);
        }
        #[cfg(feature = "cuda")]
        if let Data::CudaF64(storage) = self.stored_data() {
            return self.svd_cuda(storage, Some(truncation));
        }
        // Singular values are real => `s` is a real diagonal in O(rank) storage
        // (see `svd_compact`). `out.singular_values` is also returned, so it is
        // cloned into the diagonal factor.
        let complex = self.dtype() == Dtype::C64;
        // Lease a dense executor for this op instead of the coarse runtime lock,
        // so concurrent factorizations on a shared runtime run in parallel
        // (#155); byte-identical single-threaded.
        let mut dense = self.rt.lease_dense();
        with_data!(self, data, {
            // SU(N) (Generic): same engine and generic factor spaces; the
            // sqrt_dim² truncation weight remains a real quantum dimension.
            if let UserBoundSpace::Su3(bound) = self.ordinary_body().space.as_ref() {
                let output = tenet_matrixalgebra::svd_trunc_dyn_generic(
                    dense.dense(),
                    &BoundDynamicTensorRef::try_new(&bound, data)?,
                    truncation,
                )?;
                self.from_svd_trunc_dyn(output, complex)
            } else {
                with_bound_multiplicity_free!(self.ordinary_body().space, bound, {
                    let output = tenet_matrixalgebra::svd_trunc_dyn(
                        dense.dense(),
                        &BoundDynamicTensorRef::try_new(&bound, data)?,
                        truncation,
                    )?;
                    self.from_svd_trunc_dyn(output, complex)
                })
            }
        })
    }

    /// All singular values per coupled sector, descending (MatrixAlgebraKit
    /// `svd_vals`). Real for both dtypes.
    pub fn svd_vals(&self) -> Result<Vec<SectorSpectrum>, Error> {
        if self.is_adjoint_view() {
            return self.materialized_tensor()?.svd_vals();
        }
        // Lease a dense executor for this op instead of the coarse runtime lock,
        // so concurrent factorizations on a shared runtime run in parallel
        // (#155); byte-identical single-threaded.
        let mut dense = self.rt.lease_dense();
        with_data!(self, data, {
            if let UserBoundSpace::Su3(bound) = self.ordinary_body().space.as_ref() {
                tenet_matrixalgebra::svd_vals_dyn_generic(
                    dense.dense(),
                    &BoundDynamicTensorRef::try_new(&bound, data)?,
                )
            } else {
                with_bound_multiplicity_free!(self.ordinary_body().space, bound, {
                    tenet_matrixalgebra::svd_vals_dyn(
                        dense.dense(),
                        &BoundDynamicTensorRef::try_new(&bound, data)?,
                    )
                })
            }
            .map_err(Into::into)
        })
    }

    /// Compact QR `t = q * r` (MatrixAlgebraKit `qr_compact`): `q` has
    /// orthonormal columns per coupled sector.
    pub fn qr_compact(&self) -> Result<(Self, Self), Error> {
        if self.is_adjoint_view() {
            return self.materialized_tensor()?.qr_compact();
        }
        #[cfg(feature = "cuda")]
        if let Data::CudaF64(storage) = self.stored_data() {
            return self.qr_cuda(storage);
        }
        // Lease a dense executor for this op instead of the coarse runtime lock,
        // so concurrent factorizations on a shared runtime run in parallel
        // (#155); byte-identical single-threaded.
        let mut dense = self.rt.lease_dense();
        with_data!(self, data, {
            if let UserBoundSpace::Su3(bound) = self.ordinary_body().space.as_ref() {
                let (q, r) = tenet_matrixalgebra::qr_compact_dyn_generic(
                    dense.dense(),
                    &BoundDynamicTensorRef::try_new(bound, data)?,
                )?;
                Ok((self.from_bound_factor(q)?, self.from_bound_factor(r)?))
            } else {
                with_bound_multiplicity_free!(self.ordinary_body().space, bound, {
                    let (q, r) = tenet_matrixalgebra::qr_compact_dyn(
                        dense.dense(),
                        &BoundDynamicTensorRef::try_new(bound, data)?,
                    )?;
                    Ok::<_, Error>((self.from_bound_factor(q)?, self.from_bound_factor(r)?))
                })
            }
        })
    }

    /// Full QR `t = q * r` (MatrixAlgebraKit `qr_full`): square `q` per
    /// sector.
    pub fn qr_full(&self) -> Result<(Self, Self), Error> {
        if self.is_adjoint_view() {
            return self.materialized_tensor()?.qr_full();
        }
        // ponytail: see svd_full — the square-Q completion has no generic
        // sibling yet (B3c-3); qr_compact covers left_orth and the workflows.
        if self.rule_kind() == RuleKind::Su3 {
            return Err(Error::UnsupportedForRule {
                operation: "Tensor::qr_full",
                rule: "SU(3)",
            });
        }
        // Lease a dense executor for this op instead of the coarse runtime lock,
        // so concurrent factorizations on a shared runtime run in parallel
        // (#155); byte-identical single-threaded.
        let mut dense = self.rt.lease_dense();
        with_data!(self, data, {
            with_bound_multiplicity_free!(self.ordinary_body().space, bound, {
                let (q, r) = tenet_matrixalgebra::qr_full_dyn(
                    dense.dense(),
                    &BoundDynamicTensorRef::try_new(bound, data)?,
                )?;
                Ok::<_, Error>((self.from_bound_factor(q)?, self.from_bound_factor(r)?))
            })
        })
    }

    /// Compact LQ `t = l * q` (MatrixAlgebraKit `lq_compact`): `q` has
    /// orthonormal rows per coupled sector.
    pub fn lq_compact(&self) -> Result<(Self, Self), Error> {
        if self.is_adjoint_view() {
            return self.materialized_tensor()?.lq_compact();
        }
        // Lease a dense executor for this op instead of the coarse runtime lock,
        // so concurrent factorizations on a shared runtime run in parallel
        // (#155); byte-identical single-threaded.
        let mut dense = self.rt.lease_dense();
        with_data!(self, data, {
            if let UserBoundSpace::Su3(bound) = self.ordinary_body().space.as_ref() {
                let (l, q) = tenet_matrixalgebra::lq_compact_dyn_generic(
                    dense.dense(),
                    &BoundDynamicTensorRef::try_new(bound, data)?,
                )?;
                Ok((self.from_bound_factor(l)?, self.from_bound_factor(q)?))
            } else {
                with_bound_multiplicity_free!(self.ordinary_body().space, bound, {
                    let (l, q) = tenet_matrixalgebra::lq_compact_dyn(
                        dense.dense(),
                        &BoundDynamicTensorRef::try_new(bound, data)?,
                    )?;
                    Ok::<_, Error>((self.from_bound_factor(l)?, self.from_bound_factor(q)?))
                })
            }
        })
    }

    /// Full LQ `t = l * q` (MatrixAlgebraKit `lq_full`): square `q` per
    /// sector.
    pub fn lq_full(&self) -> Result<(Self, Self), Error> {
        if self.is_adjoint_view() {
            return self.materialized_tensor()?.lq_full();
        }
        // ponytail: see svd_full/qr_full (B3c-3); lq_compact covers right_orth.
        if self.rule_kind() == RuleKind::Su3 {
            return Err(Error::UnsupportedForRule {
                operation: "Tensor::lq_full",
                rule: "SU(3)",
            });
        }
        // Lease a dense executor for this op instead of the coarse runtime lock,
        // so concurrent factorizations on a shared runtime run in parallel
        // (#155); byte-identical single-threaded.
        let mut dense = self.rt.lease_dense();
        with_data!(self, data, {
            with_bound_multiplicity_free!(self.ordinary_body().space, bound, {
                let (l, q) = tenet_matrixalgebra::lq_full_dyn(
                    dense.dense(),
                    &BoundDynamicTensorRef::try_new(bound, data)?,
                )?;
                Ok::<_, Error>((self.from_bound_factor(l)?, self.from_bound_factor(q)?))
            })
        })
    }

    /// Left isometry factorization `t = v * c` (TensorKit 0.17 `left_orth`,
    /// default QR kind): `v` isometric, `c` the corestriction.
    pub fn left_orth(&self) -> Result<(Self, Self), Error> {
        self.qr_compact()
    }

    /// Right isometry factorization `t = c * vh` (TensorKit 0.17
    /// `right_orth`, default LQ kind): `vh` has orthonormal rows.
    pub fn right_orth(&self) -> Result<(Self, Self), Error> {
        self.lq_compact()
    }

    /// Left null space `n : codomain <- W` with `n^H * t = 0` (MatrixAlgebraKit
    /// `left_null`).
    pub fn left_null(&self) -> Result<Self, Error> {
        if self.is_adjoint_view() {
            return self.materialized_tensor()?.left_null();
        }
        self.reject_unwired_su3("Tensor::left_null")?;
        // Lease a dense executor for this op instead of the coarse runtime lock,
        // so concurrent factorizations on a shared runtime run in parallel
        // (#155); byte-identical single-threaded.
        let mut dense = self.rt.lease_dense();
        with_data!(self, data, {
            with_bound_multiplicity_free!(self.ordinary_body().space, bound, {
                let out = tenet_matrixalgebra::left_null_dyn(
                    dense.dense(),
                    &BoundDynamicTensorRef::try_new(bound, data)?,
                )?;
                self.from_bound_factor(out)
            })
        })
    }

    /// Right null space `n : W <- domain` with `t * n^H = 0` (MatrixAlgebraKit
    /// `right_null`).
    pub fn right_null(&self) -> Result<Self, Error> {
        if self.is_adjoint_view() {
            return self.materialized_tensor()?.right_null();
        }
        self.reject_unwired_su3("Tensor::right_null")?;
        // Lease a dense executor for this op instead of the coarse runtime lock,
        // so concurrent factorizations on a shared runtime run in parallel
        // (#155); byte-identical single-threaded.
        let mut dense = self.rt.lease_dense();
        with_data!(self, data, {
            with_bound_multiplicity_free!(self.ordinary_body().space, bound, {
                let out = tenet_matrixalgebra::right_null_dyn(
                    dense.dense(),
                    &BoundDynamicTensorRef::try_new(bound, data)?,
                )?;
                self.from_bound_factor(out)
            })
        })
    }

    /// Left polar decomposition `t = w * p` (MatrixAlgebraKit `left_polar`):
    /// `w` isometric, `p` positive on the domain. Every coupled-sector matrix
    /// must have at least as many rows as columns.
    pub fn left_polar(&self) -> Result<(Self, Self), Error> {
        if self.is_adjoint_view() {
            return self.materialized_tensor()?.left_polar();
        }
        self.reject_unwired_su3("Tensor::left_polar")?;
        with_data!(self, data, self.left_polar_impl(data))
    }

    fn left_polar_impl<D: UserScalar>(&self, data: &[D]) -> Result<(Self, Self), Error> {
        let mut dense = self.rt.lease_dense();
        let mut lease = self.rt.lease_context()?;
        let context = lease.context();
        with_bound_ctx!(self.ordinary_body().space, context, bound, ctxs, {
            let (w, p) = tenet_matrixalgebra::left_polar_dyn(
                dense.dense(),
                D::ctx_of(ctxs),
                &BoundDynamicTensorRef::try_new(bound, data)?,
            )?;
            Ok::<_, Error>((self.from_bound_factor(w)?, self.from_bound_factor(p)?))
        })
    }

    /// Right polar decomposition `t = p * w` (MatrixAlgebraKit
    /// `right_polar`): `p` positive on the codomain, `w` isometric. Every
    /// coupled-sector matrix must have at least as many columns as rows.
    pub fn right_polar(&self) -> Result<(Self, Self), Error> {
        if self.is_adjoint_view() {
            return self.materialized_tensor()?.right_polar();
        }
        self.reject_unwired_su3("Tensor::right_polar")?;
        with_data!(self, data, self.right_polar_impl(data))
    }

    fn right_polar_impl<D: UserScalar>(&self, data: &[D]) -> Result<(Self, Self), Error> {
        let mut dense = self.rt.lease_dense();
        let mut lease = self.rt.lease_context()?;
        let context = lease.context();
        with_bound_ctx!(self.ordinary_body().space, context, bound, ctxs, {
            let (p, w) = tenet_matrixalgebra::right_polar_dyn(
                dense.dense(),
                D::ctx_of(ctxs),
                &BoundDynamicTensorRef::try_new(bound, data)?,
            )?;
            Ok::<_, Error>((self.from_bound_factor(p)?, self.from_bound_factor(w)?))
        })
    }

    /// Full Hermitian eigendecomposition `t = v * d * v^H` (MatrixAlgebraKit
    /// `eigh_full`), returned as `(d, v)`. Requires an endomorphism with
    /// Hermitian coupled blocks. The eigenvalues are real for both dtypes
    /// (TensorKit: real `D`); `d` keeps the input dtype so it composes with
    /// `v` directly.
    pub fn eigh_full(&self) -> Result<(Self, Self), Error> {
        if self.is_adjoint_view() {
            return self.materialized_tensor()?.eigh_full();
        }
        self.reject_unwired_su3("Tensor::eigh_full")?;
        #[cfg(feature = "cuda")]
        if let Data::CudaF64(storage) = self.stored_data() {
            let out = self.eigh_cuda(storage, None)?;
            return Ok((out.d, out.v));
        }
        // eigh eigenvalues are real, so `d` is a real diagonal (`RealC64` for
        // c64 input). Build it as O(rank) diagonal storage from the spectrum;
        // `eigh_full_dyn` returns only the spectrum + eigenvectors (no dense d),
        // so nothing O(rank²) is materialized and discarded here (#56 item N).
        let complex = self.dtype() == Dtype::C64;
        // Lease a dense executor for this op instead of the coarse runtime lock,
        // so concurrent factorizations on a shared runtime run in parallel
        // (#155); byte-identical single-threaded.
        let mut dense = self.rt.lease_dense();
        with_data!(self, data, {
            with_bound_multiplicity_free!(self.ordinary_body().space, bound, {
                let out = tenet_matrixalgebra::eigh_full_dyn(
                    dense.dense(),
                    &BoundDynamicTensorRef::try_new(bound, data)?,
                )?;
                let (v, eigenvalues) = out.into_parts();
                Ok::<_, Error>((
                    self.from_diagonal_real_spectrum(eigenvalues, complex)?,
                    self.from_bound_factor(v)?,
                ))
            })
        })
    }

    /// Truncated Hermitian eigendecomposition (MatrixAlgebraKit
    /// `eigh_trunc`); see [`EighTrunc`].
    pub fn eigh_trunc(&self, truncation: &Truncation) -> Result<EighTrunc, Error> {
        if self.is_adjoint_view() {
            return self.materialized_tensor()?.eigh_trunc(truncation);
        }
        self.reject_unwired_su3("Tensor::eigh_trunc")?;
        #[cfg(feature = "cuda")]
        if let Data::CudaF64(storage) = self.stored_data() {
            return self.eigh_cuda(storage, Some(truncation));
        }
        // Real eigenvalues => real diagonal `d` in O(rank) storage (see
        // `eigh_full`). `out.eigenvalues` is also returned to the caller, so it
        // is cloned into the diagonal factor.
        let complex = self.dtype() == Dtype::C64;
        // Lease a dense executor for this op instead of the coarse runtime lock,
        // so concurrent factorizations on a shared runtime run in parallel
        // (#155); byte-identical single-threaded.
        let mut dense = self.rt.lease_dense();
        with_data!(self, data, {
            with_bound_multiplicity_free!(self.ordinary_body().space, bound, {
                let out = tenet_matrixalgebra::eigh_trunc_dyn(
                    dense.dense(),
                    &BoundDynamicTensorRef::try_new(bound, data)?,
                    truncation,
                )?;
                let (v, eigenvalues, error) = out.into_parts();
                Ok::<_, Error>(EighTrunc {
                    d: self.from_diagonal_real_spectrum(eigenvalues.clone(), complex)?,
                    v: self.from_bound_factor(v)?,
                    eigenvalues,
                    error,
                })
            })
        })
    }

    /// All Hermitian eigenvalues per coupled sector, descending by magnitude
    /// (MatrixAlgebraKit `eigh_vals`). Real for both dtypes.
    pub fn eigh_vals(&self) -> Result<Vec<SectorSpectrum>, Error> {
        if self.is_adjoint_view() {
            return self.materialized_tensor()?.eigh_vals();
        }
        self.reject_unwired_su3("Tensor::eigh_vals")?;
        // Lease a dense executor for this op instead of the coarse runtime lock,
        // so concurrent factorizations on a shared runtime run in parallel
        // (#155); byte-identical single-threaded.
        let mut dense = self.rt.lease_dense();
        with_data!(self, data, {
            with_bound_multiplicity_free!(self.ordinary_body().space, bound, {
                tenet_matrixalgebra::eigh_vals_dyn(
                    dense.dense(),
                    &BoundDynamicTensorRef::try_new(bound, data)?,
                )
            })
            .map_err(Into::into)
        })
    }

    /// Full general (non-Hermitian) eigendecomposition `t = v * d * v^-1`
    /// (MatrixAlgebraKit `eig_full`), returned as `(d, v)`. Requires an
    /// endomorphism. The output tensors are always c64, even for f64 input
    /// (real matrices have complex eigenpairs), matching TensorKit's
    /// `eigen`, whose `D` and `V` are `ComplexF64` for real input.
    pub fn eig_full(&self) -> Result<(Self, Self), Error> {
        if self.is_adjoint_view() {
            return self.materialized_tensor()?.eig_full();
        }
        self.reject_unwired_su3("Tensor::eig_full")?;
        // Lease a dense executor for this op instead of the coarse runtime lock,
        // so concurrent factorizations on a shared runtime run in parallel
        // (#155); byte-identical single-threaded.
        let mut dense = self.rt.lease_dense();
        with_data!(self, data, {
            with_bound_multiplicity_free!(self.ordinary_body().space, bound, {
                let out = tenet_matrixalgebra::eig_full_dyn(
                    dense.dense(),
                    &BoundDynamicTensorRef::try_new(bound, data)?,
                )?;
                let (v, eigenvalues) = out.into_parts();
                Ok::<_, Error>((
                    self.from_diagonal_complex_spectrum(eigenvalues)?,
                    self.from_bound_factor(v)?,
                ))
            })
        })
    }

    /// Truncated general eigendecomposition (MatrixAlgebraKit `eig_trunc`,
    /// kept by descending `|eigenvalue|`); see [`EigTrunc`]. Output tensors
    /// are always c64.
    pub fn eig_trunc(&self, truncation: &Truncation) -> Result<EigTrunc, Error> {
        if self.is_adjoint_view() {
            return self.materialized_tensor()?.eig_trunc(truncation);
        }
        self.reject_unwired_su3("Tensor::eig_trunc")?;
        // Lease a dense executor for this op instead of the coarse runtime lock,
        // so concurrent factorizations on a shared runtime run in parallel
        // (#155); byte-identical single-threaded.
        let mut dense = self.rt.lease_dense();
        with_data!(self, data, {
            with_bound_multiplicity_free!(self.ordinary_body().space, bound, {
                let out = tenet_matrixalgebra::eig_trunc_dyn(
                    dense.dense(),
                    &BoundDynamicTensorRef::try_new(bound, data)?,
                    truncation,
                )?;
                let (v, eigenvalues, error) = out.into_parts();
                Ok::<_, Error>(EigTrunc {
                    d: self.from_diagonal_complex_spectrum(eigenvalues.clone())?,
                    v: self.from_bound_factor(v)?,
                    eigenvalues,
                    error,
                })
            })
        })
    }

    /// All general eigenvalues per coupled sector, descending by magnitude
    /// (MatrixAlgebraKit `eig_vals`). Complex for both dtypes.
    pub fn eig_vals(&self) -> Result<Vec<SectorSpectrum<Complex64>>, Error> {
        if self.is_adjoint_view() {
            return self.materialized_tensor()?.eig_vals();
        }
        self.reject_unwired_su3("Tensor::eig_vals")?;
        // Lease a dense executor for this op instead of the coarse runtime lock,
        // so concurrent factorizations on a shared runtime run in parallel
        // (#155); byte-identical single-threaded.
        let mut dense = self.rt.lease_dense();
        with_data!(self, data, {
            with_bound_multiplicity_free!(self.ordinary_body().space, bound, {
                tenet_matrixalgebra::eig_vals_dyn(
                    dense.dense(),
                    &BoundDynamicTensorRef::try_new(bound, data)?,
                )
            })
            .map_err(Into::into)
        })
    }

    /// Matrix exponential of a Hermitian endomorphism, `exp(t) = v exp(d)
    /// v^H` (TensorKit `exp`, via the eigendecomposition).
    pub fn exp(&self) -> Result<Self, Error> {
        if self.is_adjoint_view() {
            return self.materialized_tensor()?.exp();
        }
        self.reject_unwired_su3("Tensor::exp")?;
        with_data!(self, data, self.exp_impl(data))
    }

    fn exp_impl<D: UserScalar>(&self, data: &[D]) -> Result<Self, Error> {
        let mut dense = self.rt.lease_dense();
        let mut lease = self.rt.lease_context()?;
        let context = lease.context();
        with_bound_ctx!(self.ordinary_body().space, context, bound, ctxs, {
            let out = tenet_matrixalgebra::exp_dyn(
                dense.dense(),
                D::ctx_of(ctxs),
                &BoundDynamicTensorRef::try_new(bound, data)?,
            )?;
            self.from_bound_factor(out)
        })
    }

    /// True inverse of a full-rank endomorphism (MatrixAlgebraKit-style
    /// `inv`); fails when any coupled block is rank-deficient at working
    /// precision.
    pub fn inv(&self) -> Result<Self, Error> {
        if self.is_adjoint_view() {
            return self.materialized_tensor()?.inv();
        }
        // A diagonal inverse is elementwise (O(rank)), not a block inversion;
        // keep it diagonal so the next contract still scales the bond.
        if let Data::Diagonal(diagonal) = self.stored_data() {
            return Ok(self.with_diagonal(diagonal.try_recip()?));
        }
        self.reject_unwired_su3("Tensor::inv")?;
        with_data!(self, data, self.inv_impl(data))
    }

    fn inv_impl<D: UserScalar>(&self, data: &[D]) -> Result<Self, Error> {
        let mut dense = self.rt.lease_dense();
        let mut lease = self.rt.lease_context()?;
        let context = lease.context();
        with_bound_ctx!(self.ordinary_body().space, context, bound, ctxs, {
            let out = tenet_matrixalgebra::inv_dyn(
                dense.dense(),
                D::ctx_of(ctxs),
                &BoundDynamicTensorRef::try_new(bound, data)?,
            )?;
            self.from_bound_factor(out)
        })
    }

    /// Moore-Penrose pseudo-inverse `t^+ = v s^+ u^H` (MatrixAlgebraKit
    /// `pinv`) with an `rcond * sigma_max` cutoff on the singular values.
    pub fn pinv(&self, rcond: f64) -> Result<Self, Error> {
        if self.is_adjoint_view() {
            return self.materialized_tensor()?.pinv(rcond);
        }
        if !rcond.is_finite() || rcond < 0.0 {
            return Err(Error::InvalidArgument(
                "pinv rcond must be finite and non-negative".to_string(),
            ));
        }
        // A diagonal pseudo-inverse is an elementwise cutoff+reciprocal on the
        // spectrum (O(rank)) — its own singular values are |entry| — so skip the
        // SVD and keep it diagonal (itebd's `l_out.pinv` fires this).
        if let Data::Diagonal(diagonal) = self.stored_data() {
            return Ok(self.with_diagonal(diagonal.pinv(rcond)));
        }
        self.reject_unwired_su3("Tensor::pinv")?;
        with_data!(self, data, self.pinv_impl(data, rcond))
    }

    fn pinv_impl<D: UserScalar>(&self, data: &[D], rcond: f64) -> Result<Self, Error> {
        let mut dense = self.rt.lease_dense();
        let mut lease = self.rt.lease_context()?;
        let context = lease.context();
        with_bound_ctx!(self.ordinary_body().space, context, bound, ctxs, {
            let out = tenet_matrixalgebra::pinv_dyn(
                dense.dense(),
                D::ctx_of(ctxs),
                &BoundDynamicTensorRef::try_new(bound, data)?,
                rcond,
            )?;
            self.from_bound_factor(out)
        })
    }

    /// Elementwise square root of a diagonal bond tensor, i.e. the
    /// TensorKit 0.17 `sqrt(::DiagonalTensorMap)` idiom
    /// (`tensors/diagonal.jl:384-390`: `sqrt.(d.data)` on the diagonal)
    /// used to split singular values as `√S · √S = S` in Vidal-gauge /
    /// gate-application updates.
    ///
    /// The receiver must be a diagonal bond tensor as produced by the
    /// factorization paths ([`Self::svd_trunc`]'s `s`, eigenvalue factors):
    /// one codomain leg equal to the one domain leg and every stored block
    /// diagonal (off-diagonal entries exactly zero). Anything else — the
    /// analog of calling this on a non-`DiagonalTensorMap` — is an
    /// [`Error::InvalidArgument`]. For f64 tensors every diagonal entry must
    /// be `>= 0` (Julia's real `sqrt` throws a `DomainError` there too;
    /// convert with [`Self::to_c64`] first for the complex branch); c64
    /// tensors take the principal complex square root.
    ///
    /// ```
    /// use tenet::prelude::*;
    ///
    /// let rt = Runtime::builder().build()?;
    /// let v = Space::u1([(0, 2), (1, 2)]);
    /// let t = Tensor::rand_with_seed(&rt, Dtype::F64, [&v, &v], [&v], 7)?;
    /// let s = t.svd_trunc(&Truncation::Full)?.s;
    /// let sqrt_s = s.sqrt()?;
    /// let composed = sqrt_s.compose(&sqrt_s)?;
    /// let max_err = composed
    ///     .data()
    ///     .iter()
    ///     .zip(s.data())
    ///     .map(|(a, b)| (a - b).abs())
    ///     .fold(0.0f64, f64::max);
    /// assert!(max_err < 1e-12);
    /// assert!(t.sqrt().is_err()); // not a diagonal bond tensor
    /// # Ok::<(), tenet::prelude::Error>(())
    /// ```
    pub fn sqrt(&self) -> Result<Self, Error> {
        if self.is_adjoint_view() {
            return self.materialized_tensor()?.sqrt();
        }
        let hom = self.ordinary_body().space.homspace();
        if hom.codomain().len() != 1
            || hom.domain().len() != 1
            || hom.codomain().legs() != hom.domain().legs()
        {
            return Err(Error::InvalidArgument(
                "sqrt requires a diagonal bond tensor `[v] <- [v]` (equal single \
                 codomain and domain legs), like the `s` factor of svd_trunc"
                    .to_string(),
            ));
        }
        // Diagonal storage: sqrt is elementwise on the spectrum (O(rank)) and
        // stays diagonal, so √S · √S = S keeps scaling the bond.
        if let Data::Diagonal(diagonal) = self.stored_data() {
            return Ok(self.with_diagonal(diagonal.try_sqrt()?));
        }
        let data = match self.coupled_data()? {
            Data::F64(data) => Data::F64(sqrt_diagonal_impl(
                &self.ordinary_body().space,
                data,
                &|value| {
                    if value < 0.0 {
                        Err(Error::InvalidArgument(format!(
                            "sqrt of a negative diagonal entry {value}; convert to c64 \
                         with to_c64() for the complex square root"
                        )))
                    } else {
                        Ok(value.sqrt())
                    }
                },
            )?),
            Data::C64(data) => Data::C64(sqrt_diagonal_impl(
                &self.ordinary_body().space,
                data,
                &|value| Ok(value.sqrt()),
            )?),
            Data::Diagonal(_) => unreachable!("coupled_data materializes Data::Diagonal"),
            #[cfg(feature = "cuda")]
            Data::CudaF64(_) => return Err(device_unsupported("sqrt")),
        };
        Ok(Self::owned(
            self.rt.clone(),
            Arc::clone(&self.materialized_body()?.space),
            Arc::new(data),
        ))
    }
}

impl RuntimeDetachedTensor {
    #[doc(hidden)]
    pub fn matches_runtime(&self, runtime: &Runtime) -> bool {
        self.runtime.matches(runtime)
    }

    #[doc(hidden)]
    pub fn attach_runtime(self, runtime: &Runtime) -> Result<Tensor, Error> {
        if !self.matches_runtime(runtime) {
            return Err(Error::RuntimeMismatch);
        }
        let Self {
            runtime: _,
            repr,
            compact_dense,
        } = self;
        Ok(Tensor {
            rt: runtime.clone(),
            repr,
            compact_dense,
        })
    }
}

impl TensorExecutionContext {
    // Why not keep `can_contract_overwrite_into`/`can_permute_overwrite_into`:
    // #144 moved every real caller onto `try_contract_overwrite_into`/
    // `try_permute_overwrite_into` (check-and-write-once), and a workspace +
    // external-consumer grep found zero remaining callers of the two-call
    // check-then-write predicates. Keeping the unused, non-hidden predicates
    // around made rustdoc advertise a dead TOCTOU-shaped pattern as the
    // correct one (issue #150).

    #[allow(clippy::too_many_arguments)]
    fn write_contract_prepared(
        &mut self,
        dst: &mut Tensor,
        lhs: &Tensor,
        rhs: &Tensor,
        lhs_axes: &[usize],
        rhs_axes: &[usize],
        output_order: OutputAxisOrder<'_>,
        alpha: Scalar,
        expected: &UserBoundSpace,
    ) -> Result<(), Error> {
        let rule_kind = dst.ordinary_body().space.kind();
        let dst_space = Arc::clone(&dst.ordinary_body().space);
        let dst_data = dst
            .owned_body_mut()
            .and_then(|body| Arc::get_mut(&mut body.data));
        match (dst_data, lhs.stored_data(), rhs.stored_data(), alpha) {
            (
                Some(Data::F64(dst_data)),
                Data::F64(lhs_data),
                Data::F64(rhs_data),
                Scalar::F64(alpha),
            ) => {
                // SU(3)'s generic plan omits destinations with no GEMM, while
                // built-in plans fully overwrite and should avoid the fill.
                if rule_kind == RuleKind::Su3 {
                    dst_data.fill(0.0);
                }
                dispatch_contract_into(
                    self,
                    dst_space.as_ref(),
                    expected,
                    dst_data,
                    lhs,
                    lhs_data,
                    rhs,
                    rhs_data,
                    lhs_axes,
                    rhs_axes,
                    output_order,
                    alpha,
                    0.0,
                )
            }
            (
                Some(Data::C64(dst_data)),
                Data::C64(lhs_data),
                Data::C64(rhs_data),
                Scalar::C64(alpha),
            ) => {
                if rule_kind == RuleKind::Su3 {
                    dst_data.fill(Complex64::new(0.0, 0.0));
                }
                dispatch_contract_into(
                    self,
                    dst_space.as_ref(),
                    expected,
                    dst_data,
                    lhs,
                    lhs_data,
                    rhs,
                    rhs_data,
                    lhs_axes,
                    rhs_axes,
                    output_order,
                    alpha,
                    Complex64::new(0.0, 0.0),
                )
            }
            (None, _, _, _) => Err(Error::InvalidArgument(
                "destination storage must be uniquely owned".to_string(),
            )),
            _ => Err(Error::DtypeMismatch),
        }
    }

    fn write_permute_prepared(
        &mut self,
        dst: &mut Tensor,
        src: &Tensor,
        alpha: Scalar,
        operation: PreparedPermuteOperation<'_>,
        expected: &DynamicFusionMapSpace,
    ) -> Result<(), Error> {
        // Why not clear the destination here: the explicit overwrite replay
        // writes active and structurally inactive logical blocks itself.
        let dst_space = Arc::clone(&dst.ordinary_body().space);
        let dst_data = dst
            .owned_body_mut()
            .and_then(|body| Arc::get_mut(&mut body.data));
        match (dst_data, src.stored_data(), alpha) {
            (Some(Data::F64(dst_data)), Data::F64(src_data), Scalar::F64(alpha)) => {
                #[cfg(test)]
                observe_permute_pre_replay_poison(dst_data.iter().all(|value| value.is_nan()));
                dispatch_prepared_permute_into(
                    self,
                    dst_space.as_ref(),
                    operation,
                    expected,
                    dst_data,
                    src,
                    src_data,
                    alpha,
                )
            }
            (Some(Data::C64(dst_data)), Data::C64(src_data), Scalar::C64(alpha)) => {
                #[cfg(test)]
                observe_permute_pre_replay_poison(
                    dst_data
                        .iter()
                        .all(|value| value.re.is_nan() && value.im.is_nan()),
                );
                dispatch_prepared_permute_into(
                    self,
                    dst_space.as_ref(),
                    operation,
                    expected,
                    dst_data,
                    src,
                    src_data,
                    alpha,
                )
            }
            (None, _, _) => Err(Error::InvalidArgument(
                "destination storage must be uniquely owned".to_string(),
            )),
            _ => Err(Error::DtypeMismatch),
        }
    }

    /// Attempts to overwrite `dst` in place with `alpha * contract(lhs, rhs)`,
    /// reusing the destination-space check cached in `cache` across repeated
    /// calls with the same shapes. Returns `OverwriteOutcome::Incompatible`
    /// (leaving `dst` unchanged) when `dst` cannot be reused in place — the
    /// caller matches on the outcome and falls back to an owned allocation:
    ///
    /// ```
    /// use tenet::prelude::*;
    ///
    /// let runtime = Runtime::builder().build().unwrap();
    /// let space = Space::u1([(-1, 2), (0, 3), (1, 2)]);
    /// let lhs = Tensor::rand_with_seed(&runtime, Dtype::F64, [&space], [&space], 1).unwrap();
    /// let rhs = Tensor::rand_with_seed(&runtime, Dtype::F64, [&space], [&space], 2).unwrap();
    /// let owned = lhs.contract(&rhs, &[1], &[0]).unwrap();
    /// let mut dst = owned.scale(f64::NAN).unwrap();
    /// let mut context = TensorExecutionContext::default();
    /// let mut cache = ContractOverwriteCache::default();
    ///
    /// let result = match context
    ///     .try_contract_overwrite_into(&mut cache, &mut dst, &lhs, &rhs, &[1], &[0], Scalar::F64(1.0))
    ///     .unwrap()
    /// {
    ///     OverwriteOutcome::Written => dst,
    ///     OverwriteOutcome::Incompatible => lhs.contract(&rhs, &[1], &[0]).unwrap(),
    /// };
    /// assert_eq!(result.data().len(), owned.data().len());
    /// ```
    #[doc(hidden)]
    #[allow(clippy::too_many_arguments)]
    pub fn try_contract_overwrite_into(
        &mut self,
        cache: &mut ContractOverwriteCache,
        dst: &mut Tensor,
        lhs: &Tensor,
        rhs: &Tensor,
        lhs_axes: &[usize],
        rhs_axes: &[usize],
        alpha: Scalar,
    ) -> Result<OverwriteOutcome, Error> {
        self.try_contract_overwrite_with_order(
            cache,
            dst,
            lhs,
            rhs,
            lhs_axes,
            rhs_axes,
            OutputAxisOrder::identity(),
            &[],
            alpha,
        )
    }

    /// Ordered counterpart used by compiled network replay. `output_axes`
    /// has the same pAB meaning as [`Tensor::contract_ordered`].
    #[doc(hidden)]
    #[allow(clippy::too_many_arguments)]
    pub fn try_contract_ordered_overwrite_into(
        &mut self,
        cache: &mut ContractOverwriteCache,
        dst: &mut Tensor,
        lhs: &Tensor,
        rhs: &Tensor,
        lhs_axes: &[usize],
        rhs_axes: &[usize],
        output_axes: &[usize],
        alpha: Scalar,
    ) -> Result<OverwriteOutcome, Error> {
        if lhs.rule_kind() == RuleKind::Su3
            || rhs.rule_kind() == RuleKind::Su3
            || dst.rule_kind() == RuleKind::Su3
        {
            return Ok(OverwriteOutcome::Incompatible);
        }
        self.try_contract_overwrite_with_order(
            cache,
            dst,
            lhs,
            rhs,
            lhs_axes,
            rhs_axes,
            OutputAxisOrder::from_axes(output_axes),
            output_axes,
            alpha,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn try_contract_overwrite_with_order(
        &mut self,
        cache: &mut ContractOverwriteCache,
        dst: &mut Tensor,
        lhs: &Tensor,
        rhs: &Tensor,
        lhs_axes: &[usize],
        rhs_axes: &[usize],
        output_order: OutputAxisOrder<'_>,
        cache_output_axes: &[usize],
        alpha: Scalar,
    ) -> Result<OverwriteOutcome, Error> {
        if !self.accepts_runtime(lhs)
            || !self.accepts_runtime(rhs)
            || !self.accepts_runtime(dst)
            || lhs.check_same_world(rhs).is_err()
            || dst.check_same_world(lhs).is_err()
            || dst.check_same_world(rhs).is_err()
            || dst.placement() != Placement::Host
            || lhs.is_adjoint_view()
            || rhs.is_adjoint_view()
            || dst.is_adjoint_view()
            || matches!(lhs.stored_data(), Data::Diagonal(_))
            || matches!(rhs.stored_data(), Data::Diagonal(_))
            || matches!(dst.stored_data(), Data::Diagonal(_))
            || Arc::ptr_eq(&dst.ordinary_body().data, &lhs.ordinary_body().data)
            || Arc::ptr_eq(&dst.ordinary_body().data, &rhs.ordinary_body().data)
        {
            return Ok(OverwriteOutcome::Incompatible);
        }
        if lhs_axes.len() != rhs_axes.len() {
            return Err(Error::InvalidArgument(format!(
                "contracted axis lists differ in length: {} vs {}",
                lhs_axes.len(),
                rhs_axes.len()
            )));
        }
        validate_contracted_axes(lhs_axes, lhs.rank())?;
        validate_contracted_axes(rhs_axes, rhs.rank())?;

        let cache_matches = cache.prepared.as_ref().is_some_and(|prepared| {
            same_dynamic_space_counted(
                &prepared.lhs_space,
                &lhs.ordinary_body().space,
                &mut cache.structural_comparisons,
            ) && same_dynamic_space_counted(
                &prepared.rhs_space,
                &rhs.ordinary_body().space,
                &mut cache.structural_comparisons,
            ) && prepared.lhs_axes == lhs_axes
                && prepared.rhs_axes == rhs_axes
                && prepared.output_axes == cache_output_axes
        });
        if !cache_matches {
            let expected = lhs.ordinary_body().space.contracted_with_output_order(
                &rhs.ordinary_body().space,
                lhs_axes,
                rhs_axes,
                output_order,
            )?;
            cache.prepared = Some(PreparedContractOverwrite {
                lhs_space: Arc::clone(&lhs.ordinary_body().space),
                rhs_space: Arc::clone(&rhs.ordinary_body().space),
                lhs_axes: lhs_axes.to_vec(),
                rhs_axes: rhs_axes.to_vec(),
                output_axes: cache_output_axes.to_vec(),
                expected: Arc::new(expected),
            });
            cache.preparations += 1;
        } else {
            let prepared = cache.prepared.as_mut().expect("matched above");
            if !Arc::ptr_eq(&prepared.lhs_space, &lhs.ordinary_body().space) {
                prepared.lhs_space = Arc::clone(&lhs.ordinary_body().space);
            }
            if !Arc::ptr_eq(&prepared.rhs_space, &rhs.ordinary_body().space) {
                prepared.rhs_space = Arc::clone(&rhs.ordinary_body().space);
            }
        }
        {
            let prepared = cache.prepared.as_mut().expect("prepared above");
            if !Arc::ptr_eq(&dst.ordinary_body().space, &prepared.expected) {
                cache.structural_comparisons += 1;
            }
            if dst
                .validate_exact_destination_space_arc(&prepared.expected)
                .is_err()
                || Arc::strong_count(&dst.ordinary_body().data) != 1
            {
                return Ok(OverwriteOutcome::Incompatible);
            }
            if !Arc::ptr_eq(&dst.ordinary_body().space, &prepared.expected) {
                prepared.expected = Arc::clone(&dst.ordinary_body().space);
            }
        }
        let prepared = cache.prepared.as_ref().expect("prepared above");
        self.write_contract_prepared(
            dst,
            lhs,
            rhs,
            lhs_axes,
            rhs_axes,
            output_order,
            alpha,
            prepared.expected.as_ref(),
        )?;
        Ok(OverwriteOutcome::Written)
    }

    #[doc(hidden)]
    pub fn try_permute_overwrite_into(
        &mut self,
        cache: &mut PermuteOverwriteCache,
        dst: &mut Tensor,
        src: &Tensor,
        codomain_axes: &[usize],
        domain_axes: &[usize],
        alpha: Scalar,
    ) -> Result<OverwriteOutcome, Error> {
        if !self.accepts_runtime(src)
            || !self.accepts_runtime(dst)
            || dst.check_same_world(src).is_err()
            || dst.placement() != Placement::Host
            || src.is_adjoint_view()
            || dst.is_adjoint_view()
            || matches!(src.stored_data(), Data::Diagonal(_))
            || matches!(dst.stored_data(), Data::Diagonal(_))
            || Arc::ptr_eq(&dst.ordinary_body().data, &src.ordinary_body().data)
        {
            return Ok(OverwriteOutcome::Incompatible);
        }
        let cache_matches = cache.prepared.as_ref().is_some_and(|prepared| {
            same_dynamic_space_counted(
                &prepared.source_space,
                &src.ordinary_body().space,
                &mut cache.structural_comparisons,
            ) && prepared.codomain_axes == codomain_axes
                && prepared.domain_axes == domain_axes
        });
        if !cache_matches {
            let operation = TreeTransformOperation::permute(
                codomain_axes.iter().copied(),
                domain_axes.iter().copied(),
            );
            let expected = src.ordinary_body().space.transformed(&operation)?;
            cache.prepared = Some(PreparedPermuteOverwrite {
                source_space: Arc::clone(&src.ordinary_body().space),
                codomain_axes: codomain_axes.to_vec(),
                domain_axes: domain_axes.to_vec(),
                operation,
                expected: Arc::new(expected),
            });
            cache.preparations += 1;
        } else {
            let prepared = cache.prepared.as_mut().expect("matched above");
            if !Arc::ptr_eq(&prepared.source_space, &src.ordinary_body().space) {
                prepared.source_space = Arc::clone(&src.ordinary_body().space);
            }
        }
        {
            let prepared = cache.prepared.as_mut().expect("prepared above");
            if !Arc::ptr_eq(&dst.ordinary_body().space, &prepared.expected) {
                cache.structural_comparisons += 1;
            }
            if dst
                .validate_exact_destination_space_arc(&prepared.expected)
                .is_err()
                || Arc::strong_count(&dst.ordinary_body().data) != 1
            {
                return Ok(OverwriteOutcome::Incompatible);
            }
            if !Arc::ptr_eq(&dst.ordinary_body().space, &prepared.expected) {
                prepared.expected = Arc::clone(&dst.ordinary_body().space);
            }
        }
        let prepared = cache.prepared.as_ref().expect("prepared above");
        self.write_permute_prepared(
            dst,
            src,
            alpha,
            PreparedPermuteOperation::Borrowed(&prepared.operation),
            prepared.expected.raw(),
        )?;
        Ok(OverwriteOutcome::Written)
    }

    /// Overwrites an exact-layout dense host destination with
    /// `alpha * contract(lhs, rhs)`. Validation errors leave `dst` unchanged;
    /// an error returned after backend execution begins may leave it partially
    /// overwritten.
    pub fn contract_overwrite_into(
        &mut self,
        dst: &mut Tensor,
        lhs: &Tensor,
        rhs: &Tensor,
        lhs_axes: &[usize],
        rhs_axes: &[usize],
        alpha: Scalar,
    ) -> Result<(), Error> {
        self.validate_runtime(lhs)?;
        self.validate_runtime(rhs)?;
        self.validate_runtime(dst)?;
        lhs.check_same_world(rhs)?;
        dst.validate_host_destination(lhs)?;
        dst.validate_host_destination(rhs)?;
        if lhs.is_adjoint_view()
            || rhs.is_adjoint_view()
            || matches!(lhs.stored_data(), Data::Diagonal(_))
            || matches!(rhs.stored_data(), Data::Diagonal(_))
        {
            return Err(Error::InvalidArgument(
                "dynamic destination contraction requires ordinary dense inputs".to_string(),
            ));
        }
        if lhs_axes.len() != rhs_axes.len() {
            return Err(Error::InvalidArgument(format!(
                "contracted axis lists differ in length: {} vs {}",
                lhs_axes.len(),
                rhs_axes.len()
            )));
        }
        validate_contracted_axes(lhs_axes, lhs.rank())?;
        validate_contracted_axes(rhs_axes, rhs.rank())?;

        let expected =
            lhs.ordinary_body()
                .space
                .contracted(&rhs.ordinary_body().space, lhs_axes, rhs_axes)?;
        dst.validate_exact_destination_space(expected.raw())?;

        self.write_contract_prepared(
            dst,
            lhs,
            rhs,
            lhs_axes,
            rhs_axes,
            OutputAxisOrder::identity(),
            alpha,
            &expected,
        )
    }

    /// Overwrites an exact-layout dense host destination with
    /// `alpha * permute(src)`. Validation errors leave `dst` unchanged; an
    /// error returned after backend execution begins may leave it partially
    /// overwritten.
    pub fn permute_overwrite_into(
        &mut self,
        dst: &mut Tensor,
        src: &Tensor,
        codomain_axes: &[usize],
        domain_axes: &[usize],
        alpha: Scalar,
    ) -> Result<(), Error> {
        self.validate_runtime(src)?;
        self.validate_runtime(dst)?;
        dst.validate_host_destination(src)?;
        if src.is_adjoint_view() || matches!(src.stored_data(), Data::Diagonal(_)) {
            return Err(Error::InvalidArgument(
                "dynamic destination permutation requires an ordinary dense input".to_string(),
            ));
        }
        let operation = TreeTransformOperation::permute(
            codomain_axes.iter().copied(),
            domain_axes.iter().copied(),
        );
        let expected = src.ordinary_body().space.transformed(&operation)?;
        dst.validate_exact_destination_space(expected.raw())?;

        self.write_permute_prepared(
            dst,
            src,
            alpha,
            PreparedPermuteOperation::Owned(operation),
            expected.raw(),
        )
    }
}

#[allow(clippy::too_many_arguments)]
fn contract_into_bound<R, D, Key>(
    contexts: &mut Ctxs<Key>,
    dst_space: &BoundDynamicFusionMapSpace<R>,
    dst_data: &mut [D],
    lhs_space: &BoundDynamicFusionMapSpace<R>,
    lhs_data: &[D],
    rhs_space: &BoundDynamicFusionMapSpace<R>,
    rhs_data: &[D],
    lhs_axes: &[usize],
    rhs_axes: &[usize],
    output_order: OutputAxisOrder<'_>,
    alpha: D,
    beta: D,
) -> Result<(), Error>
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64>
        + LoweredMultiplicityFreeAlgebra
        + CheckedFusionAlgebra
        + TreeTransformRuleCacheKey<Key = Key>,
    D: UserScalar,
    Key: Clone + Eq + Hash + Send + Sync + 'static,
{
    D::ctx_of(contexts)
        .tensorcontract_fusion_dyn_into_lowered(
            dst_space,
            dst_data,
            lhs_space,
            lhs_data,
            rhs_space,
            rhs_data,
            TensorContractSpec::new(lhs_axes, rhs_axes, output_order),
            alpha,
            beta,
        )
        .map_err(Into::into)
}

#[allow(clippy::too_many_arguments)]
fn dispatch_contract_into<D: UserScalar>(
    context: &mut TensorExecutionContext,
    authority: &UserBoundSpace,
    dst_space: &UserBoundSpace,
    dst_data: &mut [D],
    lhs: &Tensor,
    lhs_data: &[D],
    rhs: &Tensor,
    rhs_data: &[D],
    lhs_axes: &[usize],
    rhs_axes: &[usize],
    output_order: OutputAxisOrder<'_>,
    alpha: D,
    beta: D,
) -> Result<(), Error> {
    match (
        authority,
        dst_space,
        lhs.ordinary_body().space.as_ref(),
        rhs.ordinary_body().space.as_ref(),
    ) {
        (
            UserBoundSpace::U1(_),
            UserBoundSpace::U1(dst),
            UserBoundSpace::U1(lhs_space),
            UserBoundSpace::U1(rhs_space),
        ) => contract_into_bound(
            &mut context.u1,
            dst,
            dst_data,
            lhs_space,
            lhs_data,
            rhs_space,
            rhs_data,
            lhs_axes,
            rhs_axes,
            output_order,
            alpha,
            beta,
        ),
        (
            UserBoundSpace::Z2(_),
            UserBoundSpace::Z2(dst),
            UserBoundSpace::Z2(lhs_space),
            UserBoundSpace::Z2(rhs_space),
        ) => contract_into_bound(
            &mut context.z2,
            dst,
            dst_data,
            lhs_space,
            lhs_data,
            rhs_space,
            rhs_data,
            lhs_axes,
            rhs_axes,
            output_order,
            alpha,
            beta,
        ),
        (
            UserBoundSpace::FZ2(_),
            UserBoundSpace::FZ2(dst),
            UserBoundSpace::FZ2(lhs_space),
            UserBoundSpace::FZ2(rhs_space),
        ) => contract_into_bound(
            &mut context.fz2,
            dst,
            dst_data,
            lhs_space,
            lhs_data,
            rhs_space,
            rhs_data,
            lhs_axes,
            rhs_axes,
            output_order,
            alpha,
            beta,
        ),
        (
            UserBoundSpace::SU2(_),
            UserBoundSpace::SU2(dst),
            UserBoundSpace::SU2(lhs_space),
            UserBoundSpace::SU2(rhs_space),
        ) => contract_into_bound(
            &mut context.su2,
            dst,
            dst_data,
            lhs_space,
            lhs_data,
            rhs_space,
            rhs_data,
            lhs_axes,
            rhs_axes,
            output_order,
            alpha,
            beta,
        ),
        (
            UserBoundSpace::U1FZ2(_),
            UserBoundSpace::U1FZ2(dst),
            UserBoundSpace::U1FZ2(lhs_space),
            UserBoundSpace::U1FZ2(rhs_space),
        ) => contract_into_bound(
            &mut context.u1_fz2,
            dst,
            dst_data,
            lhs_space,
            lhs_data,
            rhs_space,
            rhs_data,
            lhs_axes,
            rhs_axes,
            output_order,
            alpha,
            beta,
        ),
        (
            UserBoundSpace::FZ2U1SU2(_),
            UserBoundSpace::FZ2U1SU2(dst),
            UserBoundSpace::FZ2U1SU2(lhs_space),
            UserBoundSpace::FZ2U1SU2(rhs_space),
        ) => contract_into_bound(
            &mut context.fz2_u1_su2,
            dst,
            dst_data,
            lhs_space,
            lhs_data,
            rhs_space,
            rhs_data,
            lhs_axes,
            rhs_axes,
            output_order,
            alpha,
            beta,
        ),
        (
            UserBoundSpace::Su3(_),
            UserBoundSpace::Su3(dst),
            UserBoundSpace::Su3(lhs_space),
            UserBoundSpace::Su3(rhs_space),
        ) => D::ctx_of(&mut context.su3)
            .tensorcontract_fusion_dyn_into_generic(
                dst,
                dst_data,
                lhs_space,
                lhs_data,
                rhs_space,
                rhs_data,
                TensorContractSpec::new(lhs_axes, rhs_axes, output_order),
                alpha,
                beta,
            )
            .map_err(Into::into),
        _ => Err(Error::RuleMismatch),
    }
}

#[allow(clippy::too_many_arguments)]
fn permute_into_with_rule<R, D, Key>(
    contexts: &mut Ctxs<Key>,
    rule: &R,
    operation: &TreeTransformOperation,
    dst_space: &DynamicFusionMapSpace,
    dst_data: &mut [D],
    src: &Tensor,
    src_data: &[D],
    alpha: D,
) -> Result<(), Error>
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64> + TreeTransformRuleCacheKey<Key = Key>,
    D: UserScalar,
    Key: Clone + Eq + Hash + Send + Sync + 'static,
{
    D::ctx_of(contexts)
        .tree_context_mut()
        .tree_transform_dyn_overwrite_into_ref(
            rule,
            operation,
            dst_space.structure(),
            src.ordinary_body().space.structure(),
            dst_data,
            src_data,
            alpha,
        )
        .map_err(Into::into)
}

#[allow(clippy::too_many_arguments)]
fn dispatch_prepared_permute_into<D: UserScalar>(
    context: &mut TensorExecutionContext,
    authority: &UserBoundSpace,
    operation: PreparedPermuteOperation<'_>,
    dst_space: &DynamicFusionMapSpace,
    dst_data: &mut [D],
    src: &Tensor,
    src_data: &[D],
    alpha: D,
) -> Result<(), Error> {
    match operation {
        PreparedPermuteOperation::Owned(operation) => dispatch_permute_into(
            context, authority, operation, dst_space, dst_data, src, src_data, alpha,
        ),
        PreparedPermuteOperation::Borrowed(operation) => dispatch_permute_into_ref(
            context, authority, operation, dst_space, dst_data, src, src_data, alpha,
        ),
    }
}

#[allow(clippy::too_many_arguments)]
fn dispatch_permute_into<D: UserScalar>(
    context: &mut TensorExecutionContext,
    authority: &UserBoundSpace,
    operation: TreeTransformOperation,
    dst_space: &DynamicFusionMapSpace,
    dst_data: &mut [D],
    src: &Tensor,
    src_data: &[D],
    alpha: D,
) -> Result<(), Error> {
    if let UserBoundSpace::Su3(space) = authority {
        return D::ctx_of(&mut context.su3)
            .tree_context_mut()
            .tree_transform_dyn_overwrite_into_generic(
                space.provider(),
                operation,
                dst_space.structure(),
                src.ordinary_body().space.structure(),
                dst_data,
                src_data,
                alpha,
            )
            .map_err(Into::into);
    }
    dispatch_permute_into_ref(
        context, authority, &operation, dst_space, dst_data, src, src_data, alpha,
    )
}

#[allow(clippy::too_many_arguments)]
fn dispatch_permute_into_ref<D: UserScalar>(
    context: &mut TensorExecutionContext,
    authority: &UserBoundSpace,
    operation: &TreeTransformOperation,
    dst_space: &DynamicFusionMapSpace,
    dst_data: &mut [D],
    src: &Tensor,
    src_data: &[D],
    alpha: D,
) -> Result<(), Error> {
    macro_rules! apply {
        ($contexts:expr, $rule:expr) => {
            permute_into_with_rule(
                $contexts, $rule, operation, dst_space, dst_data, src, src_data, alpha,
            )
        };
    }
    match authority {
        UserBoundSpace::U1(space) => apply!(&mut context.u1, space.provider()),
        UserBoundSpace::Z2(space) => apply!(&mut context.z2, space.provider()),
        UserBoundSpace::FZ2(space) => apply!(&mut context.fz2, space.provider()),
        UserBoundSpace::SU2(space) => apply!(&mut context.su2, space.provider()),
        UserBoundSpace::U1FZ2(space) => apply!(&mut context.u1_fz2, space.provider()),
        UserBoundSpace::FZ2U1SU2(space) => {
            apply!(&mut context.fz2_u1_su2, space.provider())
        }
        UserBoundSpace::Su3(space) => D::ctx_of(&mut context.su3)
            .tree_context_mut()
            .tree_transform_dyn_overwrite_into_generic(
                space.provider(),
                operation.clone(),
                dst_space.structure(),
                src.ordinary_body().space.structure(),
                dst_data,
                src_data,
                alpha,
            )
            .map_err(Into::into),
    }
}

/// TensorKit `A * B` as an operator: `&a * &b` is exactly
/// [`Tensor::compose`] (categorical composition / `mul!` on coupled blocks,
/// **no** fermionic supertrace twist — see the fermionic-semantics note on
/// [`Tensor::compose`]).
///
/// # Panics
///
/// Panics on any composition error (space/rule/runtime/dtype mismatch),
/// printing both hom spaces — mirroring TensorKit, where `A * B` throws
/// `SpaceMismatch` (nalgebra and ndarray panic on shape mismatch the same
/// way). Use [`Tensor::compose`] directly when you want the `Result`.
impl std::ops::Mul<&Tensor> for &Tensor {
    type Output = Tensor;

    fn mul(self, rhs: &Tensor) -> Tensor {
        match self.compose(rhs) {
            Ok(out) => out,
            Err(err) => panic!(
                "Tensor * Tensor (compose) failed: {err}\n  lhs: {:?} <- {:?}\n  rhs: {:?} <- {:?}",
                self.codomain_spaces(),
                self.domain_spaces(),
                rhs.codomain_spaces(),
                rhs.domain_spaces(),
            ),
        }
    }
}

/// Takes the elementwise square root of the diagonal of every `[n, n]`
/// block, verifying that all off-diagonal entries are exactly zero (the
/// storage invariant of the diagonal bond tensors built by the
/// factorization paths).
fn sqrt_diagonal_impl<D: UserScalar + PartialEq>(
    space: &DynamicFusionMapSpace,
    data: &[D],
    sqrt_of: &dyn Fn(D) -> Result<D, Error>,
) -> Result<Vec<D>, Error> {
    let zero = D::from_real(0.0);
    let mut out = vec![zero; data.len()];
    let structure = space.structure();
    for index in 0..structure.block_count() {
        let block = structure.block(index)?;
        let shape = block.shape();
        let strides = block.strides();
        let offset = block.offset();
        debug_assert_eq!(shape.len(), 2);
        for row in 0..shape[0] {
            for col in 0..shape[1] {
                let position = offset + row * strides[0] + col * strides[1];
                if row == col {
                    out[position] = sqrt_of(data[position])?;
                } else if data[position] != zero {
                    return Err(Error::InvalidArgument(format!(
                        "sqrt requires a diagonal bond tensor, but block {:?} has a \
                         nonzero off-diagonal entry at ({row}, {col})",
                        block.key()
                    )));
                }
            }
        }
    }
    Ok(out)
}

type SectorRegion = CoupledSectorRegion;

fn internal_layout_error(what: &str) -> Error {
    Error::InvalidArgument(format!(
        "internal coupled-layout invariant violated ({what}); please report this"
    ))
}

/// Enumerates the coupled-sector matrix regions of a coupled-layout block
/// structure and verifies that every fusion-tree block addresses exactly the
/// packed column-major sector matrix. The structural constructors and the
/// device paths rely on these offsets, so any other layout is an explicit
/// error, never silent misaddressing.
fn sector_regions(structure: &BlockStructure, nout: usize) -> Result<Arc<[SectorRegion]>, Error> {
    structure
        .coupled_sector_regions(nout)?
        .ok_or_else(|| internal_layout_error("non-packed coupled-sector layout"))
}

fn coupled_region_inner<D, W>(
    structure: &BlockStructure,
    nout: usize,
    a: &[D],
    b: &[D],
    mut weight_of: W,
) -> Result<Complex64, Error>
where
    D: UserScalar,
    W: FnMut(SectorId) -> f64,
{
    let regions = sector_regions(structure, nout)?;
    let required_len = structure.required_len()?;
    if a.len() != required_len || b.len() != required_len {
        return Err(internal_layout_error(
            "coupled-sector regions do not cover the scalar buffers",
        ));
    }

    let mut total = Complex64::new(0.0, 0.0);
    for region in regions.iter() {
        let range = region.range();
        let lhs = a.get(range.clone()).ok_or_else(|| {
            internal_layout_error("coupled-sector region exceeds the left scalar buffer")
        })?;
        let rhs = b.get(range).ok_or_else(|| {
            internal_layout_error("coupled-sector region exceeds the right scalar buffer")
        })?;
        let mut partial = D::from_real(0.0);
        for (&ai, &bi) in lhs.iter().zip(rhs) {
            partial = partial + FactorScalar::adjoint(ai) * bi;
        }
        total += partial.widen_complex() * weight_of(region.coupled());
    }
    Ok(total)
}

#[cfg(test)]
fn odometer_inner_oracle<D, W>(
    structure: &BlockStructure,
    a: &[D],
    b: &[D],
    mut weight_of: W,
) -> Result<Complex64, Error>
where
    D: UserScalar,
    W: FnMut(SectorId) -> f64,
{
    let mut total = Complex64::new(0.0, 0.0);
    for index in 0..structure.block_count() {
        let block = structure.block(index)?;
        let BlockKey::FusionTree(key) = block.key() else {
            return Err(internal_layout_error(
                "inner-product oracle requires fusion-tree blocks",
            ));
        };
        let coupled = key.codomain_tree().coupled();
        let shape = block.shape();
        let strides = block.strides();
        let count: usize = shape.iter().product();
        let mut indices = vec![0usize; shape.len()];
        let mut partial = D::from_real(0.0);
        for _ in 0..count {
            let position = block.offset()
                + indices
                    .iter()
                    .zip(strides)
                    .map(|(&i, &stride)| i * stride)
                    .sum::<usize>();
            partial = partial + FactorScalar::adjoint(a[position]) * b[position];
            for axis in 0..shape.len() {
                indices[axis] += 1;
                if indices[axis] < shape[axis] {
                    break;
                }
                indices[axis] = 0;
            }
        }
        total += partial.widen_complex() * weight_of(coupled);
    }
    Ok(total)
}

#[cfg(test)]
mod coupled_region_inner_tests {
    use super::*;
    use tenet_core::GenericRigidSymbols;

    fn assert_close(actual: Complex64, expected: Complex64) {
        assert!(
            (actual - expected).norm() <= 1.0e-11 * (1.0 + expected.norm()),
            "actual={actual:?}, expected={expected:?}"
        );
    }

    fn assert_multiplicity_free_oracle(space: Space, seed: u64) {
        let runtime = Runtime::builder().dense_threads(1).build().unwrap();
        for dtype in [Dtype::F64, Dtype::C64] {
            let lhs =
                Tensor::rand_with_seed(&runtime, dtype, [&space, &space], [&space], seed).unwrap();
            let rhs = Tensor::rand_with_seed(&runtime, dtype, [&space, &space], [&space], seed + 1)
                .unwrap();
            let structure = lhs.ordinary_body().space.structure();
            let expected = match (lhs.coupled_data().unwrap(), rhs.coupled_data().unwrap()) {
                (Data::F64(a), Data::F64(b)) => {
                    with_user_rule!(lhs.ordinary_body().space, rule, {
                        odometer_inner_oracle(structure, a, b, |coupled| rule.dim_scalar(coupled))
                    })
                }
                (Data::C64(a), Data::C64(b)) => {
                    with_user_rule!(lhs.ordinary_body().space, rule, {
                        odometer_inner_oracle(structure, a, b, |coupled| rule.dim_scalar(coupled))
                    })
                }
                _ => unreachable!(),
            }
            .unwrap();
            let actual = lhs.inner(&rhs).unwrap().to_c64();
            assert_close(actual, expected);
            assert_close(rhs.inner(&lhs).unwrap().to_c64(), actual.conj());
            assert_close(
                lhs.inner(&lhs).unwrap().to_c64(),
                Complex64::new(lhs.norm().unwrap().powi(2), 0.0),
            );
        }
    }

    #[test]
    fn non_abelian_regions_match_the_block_odometer_oracle() {
        // What: contiguous coupled-sector reduction preserves every block of
        // multi-sector, multi-tree SU(2) and fermionic product tensors.
        assert_multiplicity_free_oracle(Space::su2([(0, 2), (1, 2), (2, 1), (3, 1)]), 282_001);
        assert_multiplicity_free_oracle(
            Space::fz2_u1_su2([
                ((0, -2, 0), 2),
                ((0, 1, 2), 1),
                ((1, -1, 1), 2),
                ((1, 2, 3), 1),
            ])
            .unwrap(),
            282_101,
        );
    }

    #[test]
    fn generic_outer_multiplicity_norm_matches_the_block_odometer_oracle() {
        // What: SU(3) norm includes every outer-multiplicity vertex with the
        // generic sqrt-dimension-squared weight.
        let runtime = Runtime::builder().dense_threads(1).build().unwrap();
        let space = Space::su3([((1, 1), 2)]).unwrap();
        let tensor =
            Tensor::rand_with_seed(&runtime, Dtype::C64, [&space], [&space, &space], 282_201)
                .unwrap();
        let rule = tensor.su3_rule();
        let data = tensor.data_c64();
        let expected = odometer_inner_oracle(
            tensor.ordinary_body().space.structure(),
            data,
            data,
            |coupled| {
                let sqrt = rule.sqrt_dim_scalar(coupled);
                sqrt * sqrt
            },
        )
        .unwrap();
        assert_close(
            Complex64::new(tensor.norm().unwrap().powi(2), 0.0),
            expected,
        );
    }

    #[test]
    fn malformed_scalar_range_is_a_typed_internal_layout_error() {
        // What: the lowest contiguous-region boundary rejects a scalar buffer
        // that cannot be covered exactly instead of entering an odometer path.
        let runtime = Runtime::builder().dense_threads(1).build().unwrap();
        let space = Space::su2([(0, 2), (1, 2), (2, 1)]);
        let tensor =
            Tensor::rand_with_seed(&runtime, Dtype::F64, [&space], [&space], 282_301).unwrap();
        let data = tensor.data();
        let error = coupled_region_inner(
            tensor.ordinary_body().space.structure(),
            tensor.ordinary_body().space.nout(),
            &data[..data.len() - 1],
            data,
            |_| 1.0,
        )
        .unwrap_err();
        assert!(matches!(error, Error::InvalidArgument(message) if
            message.contains("internal coupled-layout invariant violated")));
    }

    #[test]
    fn empty_and_non_fusion_structures_keep_explicit_boundary_semantics() {
        // What: a canonical empty structure reduces to zero, while an ordinal
        // dense structure is rejected as non-packed coupled-sector storage.
        let empty = BlockStructure::empty(3);
        assert_eq!(
            coupled_region_inner::<f64, _>(&empty, 1, &[], &[], |_| 7.0).unwrap(),
            Complex64::new(0.0, 0.0)
        );

        let trivial = BlockStructure::trivial(&[2, 2]).unwrap();
        let error = coupled_region_inner(&trivial, 1, &[1.0; 4], &[1.0; 4], |_| 1.0).unwrap_err();
        assert!(matches!(error, Error::InvalidArgument(message) if
            message.contains("non-packed coupled-sector layout")));
    }
}

// ---------------------------------------------------------------------------
// CUDA device paths (f64 only).
//
// The user-layer storage is always the TensorKit-equivalent coupled-sector
// matrix layout, so every coupled sector is one contiguous column-major
// matrix region of the flat device buffer. All device work is expressed as
// (a) per-sector cuSOLVER decompositions on those regions and (b) region
// GEMMs against small host-built selector matrices (identity / prefix /
// sign / permutation) that also perform factor assembly into freshly
// allocated coupled-layout buffers. Only scalars ever cross PCIe implicitly:
// per-sector reduction partials, singular/eigen values and R diagonals
// (truncation and gauge decisions are host scalar logic).
// ---------------------------------------------------------------------------

#[cfg(feature = "cuda")]
fn dense_err(err: tenet_dense::DenseError) -> Error {
    Error::from(OperationError::Dense(err))
}

#[cfg(feature = "cuda")]
fn require_cuda(cuda: Option<&mut CudaDenseContext>) -> Result<&mut CudaDenseContext, Error> {
    cuda.ok_or_else(|| {
        Error::InvalidArgument(
            "this runtime was built without a CUDA device; use \
             Runtime::builder().cuda(device)"
                .to_string(),
        )
    })
}

#[cfg(feature = "cuda")]
fn coupled_sector_of(region: &SectorRegion) -> SectorId {
    region.coupled()
}

#[cfg(feature = "cuda")]
fn find_source<'a>(
    regions: &'a [SectorRegion],
    target: &SectorRegion,
) -> Result<(usize, &'a SectorRegion), Error> {
    regions
        .iter()
        .enumerate()
        .find(|(_, region)| region.coupled() == target.coupled())
        .ok_or_else(|| internal_layout_error("factor bond sector missing in the source tensor"))
}

/// Shared host-side truncation decision (exactly the host cores' flow:
/// `select_truncation` over quantum-dimension-weighted magnitudes, kept
/// prefixes, empty sectors dropped, discarded weighted 2-norm as `error`;
/// a no-op decision keeps the full factorization with `error == 0`).
#[cfg(feature = "cuda")]
fn decide_kept<R: MultiplicityFreeRigidSymbols<Scalar = f64>>(
    rule: &R,
    spectra: &[SectorSpectrum],
    truncation: Option<&Truncation>,
) -> (Vec<SectorSpectrum>, f64) {
    let Some(truncation) = truncation else {
        return (spectra.to_vec(), 0.0);
    };
    let magnitudes: Vec<Vec<f64>> = spectra
        .iter()
        .map(|entry| entry.values.iter().map(|value| value.abs()).collect())
        .collect();
    let weighted: Vec<WeightedSpectrum<'_>> = spectra
        .iter()
        .zip(&magnitudes)
        .map(|(entry, values)| WeightedSpectrum {
            weight: rule.dim_scalar(entry.sector),
            values,
        })
        .collect();
    let decision = select_truncation(&weighted, truncation);
    if spectra
        .iter()
        .zip(&decision.kept)
        .all(|(entry, &count)| entry.values.len() == count)
    {
        return (spectra.to_vec(), 0.0);
    }
    let mut kept = spectra.to_vec();
    for (entry, &count) in kept.iter_mut().zip(&decision.kept) {
        entry.values.truncate(count);
    }
    kept.retain(|entry| !entry.values.is_empty());
    (kept, decision.error)
}

/// Uploads a small host-built selector matrix (`rows x cols`, column-major,
/// zero except `entries`) used by the assembly GEMMs.
#[cfg(feature = "cuda")]
fn upload_selector(
    cuda: &mut CudaDenseContext,
    rows: usize,
    cols: usize,
    entries: impl Iterator<Item = (usize, usize, f64)>,
) -> Result<CudaStorage, Error> {
    let mut data = vec![0.0; rows * cols];
    for (row, col, value) in entries {
        data[row + rows * col] = value;
    }
    CudaStorage::upload(cuda, &data).map_err(Error::from)
}

/// Writes `factor_rows x kept` slices of `factor * selector` into the target
/// sector region of a left factor (`codomain <- bond`), one GEMM per
/// codomain tree so correctness never relies on tree enumeration order
/// matching between the source and factor spaces.
#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
fn assemble_left_factor(
    cuda: &mut CudaDenseContext,
    dst: &mut CudaStorage,
    target: &SectorRegion,
    source: &SectorRegion,
    factor: &CudaDenseStorage,
    k_full: usize,
    selector: &CudaStorage,
    kept: usize,
) -> Result<(), Error> {
    for target_tree in target.row_trees() {
        let sub_rows = target_tree.extent()?;
        if sub_rows == 0 {
            continue;
        }
        let src_row = source
            .row_trees()
            .iter()
            .find(|source_tree| source_tree.tree() == target_tree.tree())
            .map(|source_tree| source_tree.offset())
            .ok_or_else(|| internal_layout_error("codomain tree missing in the source sector"))?;
        cuda_gemm_region_into(
            cuda,
            &mut dst.0,
            target.range().start + target_tree.offset(),
            target.rows(),
            factor,
            src_row,
            source.rows(),
            &selector.0,
            0,
            k_full,
            sub_rows,
            k_full,
            kept,
            1.0,
            0.0,
        )
        .map_err(dense_err)?;
    }
    Ok(())
}

/// Writes `kept x factor_cols` slices of `selector * factor` into the target
/// sector region of a right factor (`bond <- domain`), one GEMM per domain
/// tree.
#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
fn assemble_right_factor(
    cuda: &mut CudaDenseContext,
    dst: &mut CudaStorage,
    target: &SectorRegion,
    source: &SectorRegion,
    selector: &CudaStorage,
    kept: usize,
    k_full: usize,
    factor: &CudaDenseStorage,
) -> Result<(), Error> {
    for target_tree in target.col_trees() {
        let sub_cols = target_tree.extent()?;
        if sub_cols == 0 {
            continue;
        }
        let src_col = source
            .col_trees()
            .iter()
            .find(|source_tree| source_tree.tree() == target_tree.tree())
            .map(|source_tree| source_tree.offset())
            .ok_or_else(|| internal_layout_error("domain tree missing in the source sector"))?;
        cuda_gemm_region_into(
            cuda,
            &mut dst.0,
            target.range().start + target.rows() * target_tree.offset(),
            target.rows(),
            &selector.0,
            0,
            kept,
            factor,
            k_full * src_col,
            k_full,
            kept,
            k_full,
            sub_cols,
            1.0,
            0.0,
        )
        .map_err(dense_err)?;
    }
    Ok(())
}

/// Fills the diagonal of a coupled-layout `W <- W` buffer from per-sector
/// spectra, mirroring the host `diagonal_bond_tensor_dyn`.
#[cfg(feature = "cuda")]
fn fill_diagonal_values(
    structure: &BlockStructure,
    data: &mut [f64],
    spectra: &[SectorSpectrum],
) -> Result<(), Error> {
    for index in 0..structure.block_count() {
        let block = structure.block(index)?;
        let BlockKey::FusionTree(tree) = block.key() else {
            continue;
        };
        let sector = tree.codomain_tree().coupled();
        let Some(entry) = spectra.iter().find(|entry| entry.sector == sector) else {
            continue;
        };
        let strides = block.strides();
        let offset = block.offset();
        let count = block.shape()[0].min(block.shape()[1]);
        for position in 0..count {
            data[offset + position * (strides[0] + strides[1])] = entry.values[position];
        }
    }
    Ok(())
}

#[cfg(feature = "cuda")]
impl Tensor {
    /// Device weighted Frobenius inner product: one `[1, len] x [len, 1]`
    /// region GEMM per coupled sector into a device partials buffer, then a
    /// single download of the per-sector scalars, weighted by quantum
    /// dimensions on the host.
    fn weighted_inner_cuda(&self, a: &CudaStorage, b: &CudaStorage) -> Result<Complex64, Error> {
        let regions = sector_regions(
            self.ordinary_body().space.structure(),
            self.ordinary_body().space.nout(),
        )?;
        let mut guard = self.rt.lock();
        let state = &mut *guard;
        let cuda = require_cuda(state.cuda.as_mut())?;
        let mut partials = CudaStorage::upload(cuda, &vec![0.0; regions.len().max(1)])?;
        for (index, region) in regions.iter().enumerate() {
            let len = region.rows() * region.cols();
            if len == 0 {
                continue;
            }
            cuda_gemm_region_into(
                cuda,
                &mut partials.0,
                index,
                1,
                &a.0,
                region.range().start,
                1,
                &b.0,
                region.range().start,
                len,
                1,
                len,
                1,
                1.0,
                0.0,
            )
            .map_err(dense_err)?;
        }
        let values = partials.download(cuda)?;
        drop(guard);
        let total = with_user_rule!(self.ordinary_body().space, rule, {
            regions
                .iter()
                .zip(&values)
                .map(|(region, &value)| value * rule.dim_scalar(coupled_sector_of(region)))
                .sum::<f64>()
        });
        Ok(Complex64::new(total, 0.0))
    }

    /// Device `alpha * x (+ beta * y)`: whole-buffer region GEMVs against a
    /// `[1, 1]` ones operand (tenferro has no axpby/scale primitive; this
    /// stays on the one proven dot-general seam).
    fn axpby_cuda(
        &self,
        alpha: f64,
        x: &CudaStorage,
        other: Option<(f64, &CudaStorage)>,
    ) -> Result<CudaStorage, Error> {
        let len = TensorStorage::<f64>::len(x);
        let mut guard = self.rt.lock();
        let state = &mut *guard;
        let cuda = require_cuda(state.cuda.as_mut())?;
        let ones = CudaStorage::upload(cuda, &[1.0])?;
        // ponytail: destination allocated by uploading host zeros, same seam
        // as the device contraction path; replace with a device alloc if the
        // upload ever shows up in profiles.
        let mut dst = CudaStorage::upload(cuda, &vec![0.0; len])?;
        if len > 0 {
            cuda_gemm_region_into(
                cuda, &mut dst.0, 0, len, &x.0, 0, len, &ones.0, 0, 1, len, 1, 1, alpha, 0.0,
            )
            .map_err(dense_err)?;
            if let Some((beta, y)) = other {
                cuda_gemm_region_into(
                    cuda, &mut dst.0, 0, len, &y.0, 0, len, &ones.0, 0, 1, len, 1, 1, beta, 1.0,
                )
                .map_err(dense_err)?;
            }
        }
        drop(guard);
        Ok(dst)
    }

    /// Device SVD: per-sector cuSOLVER SVD on the packed regions, values
    /// downloaded for the (shared, host-side) truncation decision, factors
    /// assembled on device through prefix selectors. `truncation: None` is
    /// `svd_compact`.
    fn svd_cuda(
        &self,
        storage: &CudaStorage,
        truncation: Option<&Truncation>,
    ) -> Result<SvdTrunc, Error> {
        let regions = sector_regions(
            self.ordinary_body().space.structure(),
            self.ordinary_body().space.nout(),
        )?;
        let mut guard = self.rt.lock();
        let state = &mut *guard;
        let cuda = require_cuda(state.cuda.as_mut())?;
        let out = with_bound_multiplicity_free!(self.ordinary_body().space, bound, {
            let rule = bound.provider();
            let mut spectra: Vec<SectorSpectrum> = Vec::with_capacity(regions.len());
            let mut factors: Vec<Option<(CudaDenseStorage, CudaDenseStorage)>> =
                Vec::with_capacity(regions.len());
            for region in regions.iter() {
                let sector = coupled_sector_of(region);
                if region.rows() == 0 || region.cols() == 0 {
                    spectra.push(SectorSpectrum {
                        sector,
                        values: Vec::new(),
                    });
                    factors.push(None);
                    continue;
                }
                let (u, s, vt) = cuda_svd_region(
                    cuda,
                    &storage.0,
                    region.range().start,
                    region.rows(),
                    region.cols(),
                )
                .map_err(dense_err)?;
                spectra.push(SectorSpectrum { sector, values: s });
                factors.push(Some((u, vt)));
            }
            let (kept_spectra, error) = decide_kept(rule, &spectra, truncation);

            let hom = self.ordinary_body().space.homspace();
            let bond_leg = SectorLeg::new(
                kept_spectra
                    .iter()
                    .map(|entry| (entry.sector, entry.values.len())),
                false,
            );
            let build_output_space = |hom| {
                let space = build_bound_space_like(bound, hom)?;
                UserBoundSpace::from_bound(self.ordinary_body().space.as_ref(), space)
            };
            let u_space = build_output_space(FusionTreeHomSpace::new(
                FusionProductSpace::new(hom.codomain().legs().iter().cloned()),
                FusionProductSpace::new([bond_leg.clone()]),
            ))?;
            let s_space = build_output_space(FusionTreeHomSpace::new(
                FusionProductSpace::new([bond_leg.clone()]),
                FusionProductSpace::new([bond_leg.clone()]),
            ))?;
            let vh_space = build_output_space(FusionTreeHomSpace::new(
                FusionProductSpace::new([bond_leg]),
                FusionProductSpace::new(hom.domain().legs().iter().cloned()),
            ))?;

            let mut u_data = CudaStorage::upload(cuda, &vec![0.0; u_space.required_len()?])?;
            for target in sector_regions(u_space.structure(), u_space.nout())?.iter() {
                let kept = target.cols();
                if kept == 0 {
                    continue;
                }
                let (index, source) = find_source(&regions, target)?;
                let Some((u_dev, _)) = &factors[index] else {
                    return Err(internal_layout_error("kept sector without a device factor"));
                };
                let k_full = source.rows().min(source.cols());
                let selector = upload_selector(cuda, k_full, kept, (0..kept).map(|j| (j, j, 1.0)))?;
                assemble_left_factor(
                    cuda,
                    &mut u_data,
                    target,
                    source,
                    u_dev,
                    k_full,
                    &selector,
                    kept,
                )?;
            }

            let mut vh_data = CudaStorage::upload(cuda, &vec![0.0; vh_space.required_len()?])?;
            for target in sector_regions(vh_space.structure(), vh_space.nout())?.iter() {
                let kept = target.rows();
                if kept == 0 {
                    continue;
                }
                let (index, source) = find_source(&regions, target)?;
                let Some((_, vt_dev)) = &factors[index] else {
                    return Err(internal_layout_error("kept sector without a device factor"));
                };
                let k_full = source.rows().min(source.cols());
                let selector = upload_selector(cuda, kept, k_full, (0..kept).map(|j| (j, j, 1.0)))?;
                assemble_right_factor(
                    cuda,
                    &mut vh_data,
                    target,
                    source,
                    &selector,
                    kept,
                    k_full,
                    vt_dev,
                )?;
            }

            let mut s_host = vec![0.0; s_space.required_len()?];
            fill_diagonal_values(s_space.structure(), &mut s_host, &kept_spectra)?;
            let s_data = CudaStorage::upload(cuda, &s_host)?;

            Ok::<_, Error>(SvdTrunc {
                u: self.with_bound(u_space, Data::CudaF64(Arc::new(u_data)))?,
                s: self.with_bound(s_space, Data::CudaF64(Arc::new(s_data)))?,
                vh: self.with_bound(vh_space, Data::CudaF64(Arc::new(vh_data)))?,
                singular_values: kept_spectra,
                error,
            })
        })?;
        drop(guard);
        Ok(out)
    }

    /// Device compact QR with the host's positive-diagonal gauge: only `R`'s
    /// diagonal crosses to the host (sign decisions), the gauge is applied by
    /// the sign-selector assembly GEMMs.
    fn qr_cuda(&self, storage: &CudaStorage) -> Result<(Self, Self), Error> {
        let regions = sector_regions(
            self.ordinary_body().space.structure(),
            self.ordinary_body().space.nout(),
        )?;
        let mut guard = self.rt.lock();
        let state = &mut *guard;
        let cuda = require_cuda(state.cuda.as_mut())?;
        let out = with_bound_multiplicity_free!(self.ordinary_body().space, bound, {
            let rule = bound.provider();
            let mut factors: Vec<Option<(CudaDenseStorage, CudaDenseStorage, Vec<f64>)>> =
                Vec::with_capacity(regions.len());
            let mut bond_pairs: Vec<(SectorId, usize)> = Vec::with_capacity(regions.len());
            for region in regions.iter() {
                let sector = coupled_sector_of(region);
                if region.rows() == 0 || region.cols() == 0 {
                    bond_pairs.push((sector, 0));
                    factors.push(None);
                    continue;
                }
                let (q, r, diag) = cuda_qr_region(
                    cuda,
                    &storage.0,
                    region.range().start,
                    region.rows(),
                    region.cols(),
                )
                .map_err(dense_err)?;
                // Positive-diagonal gauge (host `positive_diagonal_gauge`,
                // real scalars): flip where R's diagonal is negative, leave
                // exact zeros untouched.
                let signs: Vec<f64> = diag
                    .iter()
                    .map(|&value| if value < 0.0 { -1.0 } else { 1.0 })
                    .collect();
                bond_pairs.push((sector, region.rows().min(region.cols())));
                factors.push(Some((q, r, signs)));
            }

            let hom = self.ordinary_body().space.homspace();
            let bond_leg = SectorLeg::new(bond_pairs.iter().copied(), false);
            let build_output_space = |hom| {
                let space = build_bound_space_like(bound, hom)?;
                UserBoundSpace::from_bound(self.ordinary_body().space.as_ref(), space)
            };
            let q_space = build_output_space(FusionTreeHomSpace::new(
                FusionProductSpace::new(hom.codomain().legs().iter().cloned()),
                FusionProductSpace::new([bond_leg.clone()]),
            ))?;
            let r_space = build_output_space(FusionTreeHomSpace::new(
                FusionProductSpace::new([bond_leg]),
                FusionProductSpace::new(hom.domain().legs().iter().cloned()),
            ))?;

            let mut q_data = CudaStorage::upload(cuda, &vec![0.0; q_space.required_len()?])?;
            for target in sector_regions(q_space.structure(), q_space.nout())?.iter() {
                let kept = target.cols();
                if kept == 0 {
                    continue;
                }
                let (index, source) = find_source(&regions, target)?;
                let Some((q_dev, _, signs)) = &factors[index] else {
                    return Err(internal_layout_error("kept sector without a device factor"));
                };
                let selector = upload_selector(
                    cuda,
                    kept,
                    kept,
                    signs.iter().enumerate().map(|(j, &sign)| (j, j, sign)),
                )?;
                assemble_left_factor(
                    cuda,
                    &mut q_data,
                    target,
                    source,
                    q_dev,
                    kept,
                    &selector,
                    kept,
                )?;
            }

            let mut r_data = CudaStorage::upload(cuda, &vec![0.0; r_space.required_len()?])?;
            for target in sector_regions(r_space.structure(), r_space.nout())?.iter() {
                let kept = target.rows();
                if kept == 0 {
                    continue;
                }
                let (index, source) = find_source(&regions, target)?;
                let Some((_, r_dev, signs)) = &factors[index] else {
                    return Err(internal_layout_error("kept sector without a device factor"));
                };
                let selector = upload_selector(
                    cuda,
                    kept,
                    kept,
                    signs.iter().enumerate().map(|(j, &sign)| (j, j, sign)),
                )?;
                assemble_right_factor(
                    cuda,
                    &mut r_data,
                    target,
                    source,
                    &selector,
                    kept,
                    kept,
                    r_dev,
                )?;
            }

            Ok::<_, Error>((
                self.with_bound(q_space, Data::CudaF64(Arc::new(q_data)))?,
                self.with_bound(r_space, Data::CudaF64(Arc::new(r_data)))?,
            ))
        })?;
        drop(guard);
        Ok(out)
    }

    /// Device Hermitian eigendecomposition: eigenvalues cross to the host
    /// (descending-by-magnitude ordering and truncation are host decisions),
    /// eigenvectors are reordered / truncated on device via a permutation
    /// selector. `truncation: None` is `eigh_full`.
    fn eigh_cuda(
        &self,
        storage: &CudaStorage,
        truncation: Option<&Truncation>,
    ) -> Result<EighTrunc, Error> {
        {
            let hom = self.ordinary_body().space.homspace();
            if hom.codomain() != hom.domain() {
                return Err(Error::InvalidArgument(
                    "eigh requires an endomorphism (codomain == domain)".to_string(),
                ));
            }
        }
        let regions = sector_regions(
            self.ordinary_body().space.structure(),
            self.ordinary_body().space.nout(),
        )?;
        let mut guard = self.rt.lock();
        let state = &mut *guard;
        let cuda = require_cuda(state.cuda.as_mut())?;
        // No device validator exists; skipping this copy lets cuSOLVER silently trust one triangle.
        let host_data = storage.0.download_f64(cuda).map_err(dense_err)?;
        validate_hermitian_regions(&host_data, &regions)?;
        let out = with_bound_multiplicity_free!(self.ordinary_body().space, bound, {
            let rule = bound.provider();
            let mut spectra: Vec<SectorSpectrum> = Vec::with_capacity(regions.len());
            let mut factors: Vec<Option<(CudaDenseStorage, Vec<usize>)>> =
                Vec::with_capacity(regions.len());
            for region in regions.iter() {
                let sector = coupled_sector_of(region);
                let n = region.rows();
                if n == 0 {
                    spectra.push(SectorSpectrum {
                        sector,
                        values: Vec::new(),
                    });
                    factors.push(None);
                    continue;
                }
                let (values, vectors) = cuda_eigh_region(cuda, &storage.0, region.range().start, n)
                    .map_err(dense_err)?;
                // Host ordering contract: descending by |eigenvalue|,
                // stable on ties (mirrors `eigh_full_dyn`).
                let mut order: Vec<usize> = (0..n).collect();
                order.sort_by(|&a, &b| {
                    values[b]
                        .abs()
                        .partial_cmp(&values[a].abs())
                        .expect("finite eigenvalues")
                        .then(a.cmp(&b))
                });
                let sorted: Vec<f64> = order.iter().map(|&index| values[index]).collect();
                spectra.push(SectorSpectrum {
                    sector,
                    values: sorted,
                });
                factors.push(Some((vectors, order)));
            }
            let (kept_spectra, error) = decide_kept(rule, &spectra, truncation);

            let hom = self.ordinary_body().space.homspace();
            let bond_leg = SectorLeg::new(
                kept_spectra
                    .iter()
                    .map(|entry| (entry.sector, entry.values.len())),
                false,
            );
            let build_output_space = |hom| {
                let space = build_bound_space_like(bound, hom)?;
                UserBoundSpace::from_bound(self.ordinary_body().space.as_ref(), space)
            };
            let v_space = build_output_space(FusionTreeHomSpace::new(
                FusionProductSpace::new(hom.codomain().legs().iter().cloned()),
                FusionProductSpace::new([bond_leg.clone()]),
            ))?;
            let d_space = build_output_space(FusionTreeHomSpace::new(
                FusionProductSpace::new([bond_leg.clone()]),
                FusionProductSpace::new([bond_leg]),
            ))?;

            let mut v_data = CudaStorage::upload(cuda, &vec![0.0; v_space.required_len()?])?;
            for target in sector_regions(v_space.structure(), v_space.nout())?.iter() {
                let kept = target.cols();
                if kept == 0 {
                    continue;
                }
                let (index, source) = find_source(&regions, target)?;
                let Some((v_dev, order)) = &factors[index] else {
                    return Err(internal_layout_error("kept sector without a device factor"));
                };
                let n = source.rows();
                let selector = upload_selector(
                    cuda,
                    n,
                    kept,
                    order
                        .iter()
                        .take(kept)
                        .enumerate()
                        .map(|(j, &original)| (original, j, 1.0)),
                )?;
                assemble_left_factor(cuda, &mut v_data, target, source, v_dev, n, &selector, kept)?;
            }

            let mut d_host = vec![0.0; d_space.required_len()?];
            fill_diagonal_values(d_space.structure(), &mut d_host, &kept_spectra)?;
            let d_data = CudaStorage::upload(cuda, &d_host)?;

            Ok::<_, Error>(EighTrunc {
                d: self.with_bound(d_space, Data::CudaF64(Arc::new(d_data)))?,
                v: self.with_bound(v_space, Data::CudaF64(Arc::new(v_data)))?,
                eigenvalues: kept_spectra,
                error,
            })
        })?;
        drop(guard);
        Ok(out)
    }
}

/// Concise, explicit tensor constructors on the runtime itself: `rt.zeros(…)`
/// is exactly `Tensor::zeros(&rt, …)`, one per common builder. Use these when
/// juggling several runtimes (each call names its own); use the argument-free
/// free functions ([`zeros`], [`rand`], …) when one default runtime suffices.
impl Runtime {
    /// [`Tensor::zeros`] on this runtime.
    pub fn zeros<'a, C, D>(&self, dtype: Dtype, codomain: C, domain: D) -> Result<Tensor, Error>
    where
        C: IntoIterator<Item = &'a Space>,
        D: IntoIterator<Item = &'a Space>,
    {
        Tensor::zeros(self, dtype, codomain, domain)
    }

    /// [`Tensor::rand`] on this runtime.
    pub fn rand<'a, C, D>(&self, dtype: Dtype, codomain: C, domain: D) -> Result<Tensor, Error>
    where
        C: IntoIterator<Item = &'a Space>,
        D: IntoIterator<Item = &'a Space>,
    {
        Tensor::rand(self, dtype, codomain, domain)
    }

    /// [`Tensor::rand_with_seed`] on this runtime.
    pub fn rand_with_seed<'a, C, D>(
        &self,
        dtype: Dtype,
        codomain: C,
        domain: D,
        seed: u64,
    ) -> Result<Tensor, Error>
    where
        C: IntoIterator<Item = &'a Space>,
        D: IntoIterator<Item = &'a Space>,
    {
        Tensor::rand_with_seed(self, dtype, codomain, domain, seed)
    }

    /// [`Tensor::id`] on this runtime.
    pub fn id<'a, S>(&self, dtype: Dtype, spaces: S) -> Result<Tensor, Error>
    where
        S: IntoIterator<Item = &'a Space>,
    {
        Tensor::id(self, dtype, spaces)
    }
}

/// Zero tensor on the calling thread's default runtime — [`Tensor::zeros`]
/// without the runtime argument. Set the default once with
/// [`crate::set_default_runtime`] / [`crate::default!`]; errors if none is set.
pub fn zeros<'a, C, D>(dtype: Dtype, codomain: C, domain: D) -> Result<Tensor, Error>
where
    C: IntoIterator<Item = &'a Space>,
    D: IntoIterator<Item = &'a Space>,
{
    Tensor::zeros(&crate::runtime::default_runtime()?, dtype, codomain, domain)
}

/// Random tensor on the calling thread's default runtime; see [`zeros`] and
/// [`Tensor::rand`].
pub fn rand<'a, C, D>(dtype: Dtype, codomain: C, domain: D) -> Result<Tensor, Error>
where
    C: IntoIterator<Item = &'a Space>,
    D: IntoIterator<Item = &'a Space>,
{
    Tensor::rand(&crate::runtime::default_runtime()?, dtype, codomain, domain)
}

/// Seeded random tensor on the calling thread's default runtime; see [`zeros`]
/// and [`Tensor::rand_with_seed`].
pub fn rand_with_seed<'a, C, D>(
    dtype: Dtype,
    codomain: C,
    domain: D,
    seed: u64,
) -> Result<Tensor, Error>
where
    C: IntoIterator<Item = &'a Space>,
    D: IntoIterator<Item = &'a Space>,
{
    Tensor::rand_with_seed(
        &crate::runtime::default_runtime()?,
        dtype,
        codomain,
        domain,
        seed,
    )
}

/// Identity tensor on the calling thread's default runtime; see [`zeros`] and
/// [`Tensor::id`].
pub fn id<'a, S>(dtype: Dtype, spaces: S) -> Result<Tensor, Error>
where
    S: IntoIterator<Item = &'a Space>,
{
    Tensor::id(&crate::runtime::default_runtime()?, dtype, spaces)
}

#[cfg(test)]
mod compact_diagonal_tests;

#[cfg(test)]
mod runtime_detached_tests {
    use super::*;

    #[test]
    fn runtime_detached_roundtrip_preserves_host_authority() {
        for dtype in [Dtype::F64, Dtype::C64] {
            let runtime = Runtime::builder().build().unwrap();
            let other = Runtime::builder().build().unwrap();
            let space = Space::u1([(-1, 1), (0, 2), (1, 1)]);
            let tensor =
                Tensor::rand_with_seed(&runtime, dtype, [&space], [&space], 247_001).unwrap();
            let expected_f64 = (dtype == Dtype::F64).then(|| tensor.data().to_vec());
            let expected_c64 = (dtype == Dtype::C64).then(|| tensor.data_c64().to_vec());
            let data = Arc::clone(&tensor.ordinary_body().data);
            let bound_space = Arc::clone(&tensor.ordinary_body().space);

            let detached = tensor.detach_runtime();
            assert!(!detached.matches_runtime(&other));
            assert!(detached.matches_runtime(&runtime));
            let restored = detached.attach_runtime(&runtime).unwrap();

            assert!(Arc::ptr_eq(&restored.ordinary_body().data, &data));
            assert!(Arc::ptr_eq(&restored.ordinary_body().space, &bound_space));
            if let Some(expected) = expected_f64 {
                assert_eq!(restored.data(), expected);
            }
            if let Some(expected) = expected_c64 {
                assert_eq!(restored.data_c64(), expected);
            }
        }
    }

    #[test]
    fn runtime_detached_roundtrip_preserves_lazy_adjoint_state() {
        let runtime = Runtime::builder().build().unwrap();
        let space = Space::u1([(-1, 1), (0, 2), (1, 1)]);
        let source =
            Tensor::rand_with_seed(&runtime, Dtype::C64, [&space], [&space], 247_002).unwrap();
        let lazy = source.adjoint().unwrap();
        let expected = lazy.data_c64().to_vec();
        assert!(lazy.is_adjoint_view());
        assert!(lazy.has_cached_materialization());

        let restored = lazy.detach_runtime().attach_runtime(&runtime).unwrap();

        assert!(restored.is_adjoint_view());
        assert!(restored.has_cached_materialization());
        assert_eq!(restored.data_c64(), expected);
    }

    #[test]
    fn runtime_detached_roundtrip_keeps_lazy_adjoint_unmaterialized_until_read() {
        let runtime = Runtime::builder().build().unwrap();
        let space = Space::u1([(-1, 1), (0, 2), (1, 1)]);
        let source =
            Tensor::rand_with_seed(&runtime, Dtype::C64, [&space], [&space], 247_004).unwrap();
        let expected = source.adjoint().unwrap().data_c64().to_vec();
        let lazy = source.adjoint().unwrap();
        assert!(lazy.is_adjoint_view());
        assert!(!lazy.has_cached_materialization());

        let restored = lazy.detach_runtime().attach_runtime(&runtime).unwrap();
        assert!(restored.is_adjoint_view());
        assert!(!restored.has_cached_materialization());

        assert_eq!(restored.data_c64(), expected);
        assert!(restored.has_cached_materialization());
    }

    #[test]
    fn runtime_detached_authority_rejects_a_different_runtime() {
        let runtime = Runtime::builder().build().unwrap();
        let other = Runtime::builder().build().unwrap();
        let space = Space::u1([(0, 2)]);
        let detached = Tensor::rand_with_seed(&runtime, Dtype::F64, [&space], [&space], 247_003)
            .unwrap()
            .detach_runtime();

        assert_eq!(
            detached.attach_runtime(&other).unwrap_err(),
            Error::RuntimeMismatch
        );
    }
}

#[cfg(test)]
mod adjoint_parent_view_tests {
    use super::*;

    fn assert_close(actual: &Tensor, expected: &Tensor) {
        assert_eq!(actual.codomain_spaces(), expected.codomain_spaces());
        assert_eq!(actual.domain_spaces(), expected.domain_spaces());
        assert_eq!(actual.dtype(), expected.dtype());
        match (
            actual.coupled_data().unwrap(),
            expected.coupled_data().unwrap(),
        ) {
            (Data::F64(actual), Data::F64(expected)) => {
                assert_eq!(actual.len(), expected.len());
                for (&actual, &expected) in actual.iter().zip(expected) {
                    assert!((actual - expected).abs() < 1e-11);
                }
            }
            (Data::C64(actual), Data::C64(expected)) => {
                assert_eq!(actual.len(), expected.len());
                for (&actual, &expected) in actual.iter().zip(expected) {
                    assert!((actual - expected).norm() < 1e-11);
                }
            }
            _ => panic!("dtype mismatch"),
        }
    }

    fn assert_metadata_and_materialization(space: Space, seed: u64) {
        let runtime = Runtime::builder().dense_threads(1).build().unwrap();
        let source =
            Tensor::rand_with_seed(&runtime, Dtype::C64, [&space, &space], [&space], seed).unwrap();
        let adjoint = source.adjoint().unwrap();
        assert!(adjoint.is_adjoint_view());
        assert_eq!(adjoint.codomain_rank(), 1);
        assert_eq!(adjoint.domain_rank(), 2);
        assert_eq!(adjoint.rank(), 3);
        assert_eq!(adjoint.dtype(), source.dtype());
        assert_eq!(adjoint.placement(), source.placement());
        assert!(adjoint.runtime().same_runtime(source.runtime()));
        assert_eq!(adjoint.codomain_spaces(), source.domain_spaces());
        assert_eq!(adjoint.domain_spaces(), source.codomain_spaces());
        assert_eq!(adjoint.adjoint_build_counts(), (0, 0));

        tenet_tensors::reset_global_operation_caches();
        assert_eq!(adjoint.space(0).unwrap(), source.domain_spaces()[0]);
        assert_eq!(adjoint.leg_dims().unwrap().len(), 3);
        assert_eq!(adjoint.adjoint_build_counts(), (0, 0));

        let clone = adjoint.clone();
        let expected = clone.try_data_c64().unwrap().to_vec();
        assert_eq!(adjoint.adjoint_build_counts(), (1, 1));
        assert_eq!(adjoint.try_data_c64().unwrap(), expected);
        assert_eq!(adjoint.adjoint_build_counts(), (1, 1));

        let round_trip = adjoint.adjoint().unwrap();
        assert!(!round_trip.is_adjoint_view());
        assert!(Arc::ptr_eq(
            &round_trip.ordinary_body().data,
            &source.ordinary_body().data
        ));
        assert!(Arc::ptr_eq(
            &round_trip.ordinary_body().space,
            &source.ordinary_body().space
        ));
    }

    #[test]
    fn metadata_and_shared_materialization_cover_supported_rules() {
        // What: an adjoint view swaps only logical orientation until one owned consumer reads it.
        assert_metadata_and_materialization(Space::u1([(-1, 1), (0, 2), (1, 1)]), 261_001);
        assert_metadata_and_materialization(Space::fz2([(0, 2), (1, 2)]), 261_002);
        assert_metadata_and_materialization(Space::su2([(0, 2), (1, 2), (2, 1)]), 261_003);
        assert_metadata_and_materialization(
            Space::product([((-1, 0), 1), ((0, 1), 2), ((1, 0), 1)]).unwrap(),
            261_004,
        );
        assert_metadata_and_materialization(
            Space::su3([((1, 0), 1), ((0, 1), 1)]).unwrap(),
            261_005,
        );
    }

    #[test]
    fn asymmetric_u1_consumers_match_an_eager_adjoint_oracle() {
        // What: transform, trace, and rectangular SVD consume one coherent adjoint body.
        let runtime = Runtime::builder().dense_threads(1).build().unwrap();
        let left = Space::u1([(-2, 1), (-1, 2), (0, 1), (1, 2)]);
        let right = Space::u1([(-1, 1), (0, 3), (2, 1)]);
        let source =
            Tensor::rand_with_seed(&runtime, Dtype::C64, [&left], [&right], 261_101).unwrap();
        let lazy = source.adjoint().unwrap();
        let eager = source.adjoint().unwrap().materialized_tensor().unwrap();

        assert_close(
            &lazy.permute(&[0], &[1]).unwrap(),
            &eager.permute(&[0], &[1]).unwrap(),
        );
        let (lazy_u, lazy_s, lazy_vh) = lazy.svd_compact().unwrap();
        let (eager_u, eager_s, eager_vh) = eager.svd_compact().unwrap();
        assert_close(&lazy_u, &eager_u);
        assert_close(&lazy_s, &eager_s);
        assert_close(&lazy_vh, &eager_vh);

        let endomorphism =
            Tensor::rand_with_seed(&runtime, Dtype::C64, [&left], [&left], 261_102).unwrap();
        let trace = endomorphism.tr().unwrap().to_c64();
        let adjoint_trace = endomorphism.adjoint().unwrap().tr().unwrap().to_c64();
        assert!((adjoint_trace - trace.conj()).norm() < 1e-12);
    }

    fn assert_lowered_transform_matches_eager_oracle(
        lazy: &Tensor,
        actual: Tensor,
        expected: Tensor,
    ) {
        assert_eq!(actual.codomain_spaces(), expected.codomain_spaces());
        assert_eq!(actual.domain_spaces(), expected.domain_spaces());
        assert_eq!(actual.dtype(), expected.dtype());
        assert!(actual.is_adjoint_view());
        assert_eq!(actual.adjoint_build_counts(), (0, 0));
        assert_eq!(lazy.adjoint_build_counts(), (0, 0));

        let expected_parent = expected.adjoint().unwrap().materialized_tensor().unwrap();
        let actual_parent = actual.adjoint().unwrap();
        assert_close(&actual_parent, &expected_parent);

        assert_eq!(actual.adjoint_build_counts(), (0, 0));
        assert_eq!(lazy.adjoint_build_counts(), (0, 0));
    }

    fn assert_adjoint_transforms_stay_parent_lowered(space: Space, dtype: Dtype, seed: u64) {
        let runtime = Runtime::builder()
            .dense_threads(1)
            .recoupling_threads(1)
            .build()
            .unwrap();
        let parent =
            Tensor::rand_with_seed(&runtime, dtype, [&space, &space, &space], [&space], seed)
                .unwrap();
        if dtype == Dtype::C64 {
            assert!(parent
                .data_c64()
                .iter()
                .any(|value| value.im.abs() > f64::EPSILON));
        }
        let lazy = parent.adjoint().unwrap();
        let eager = parent.adjoint().unwrap().materialized_tensor().unwrap();
        let codomain_axes = [3, 0, 2];
        let domain_axes = [1];
        let levels = [17, 3, 11, 5];

        assert_lowered_transform_matches_eager_oracle(
            &lazy,
            lazy.permute(&codomain_axes, &domain_axes).unwrap(),
            eager.permute(&codomain_axes, &domain_axes).unwrap(),
        );
        assert_lowered_transform_matches_eager_oracle(
            &lazy,
            lazy.braid(&codomain_axes, &domain_axes, &levels).unwrap(),
            eager.braid(&codomain_axes, &domain_axes, &levels).unwrap(),
        );
        assert_lowered_transform_matches_eager_oracle(
            &lazy,
            lazy.repartition(3).unwrap(),
            eager.repartition(3).unwrap(),
        );
        assert_lowered_transform_matches_eager_oracle(
            &lazy,
            lazy.transpose().unwrap(),
            eager.transpose().unwrap(),
        );

        let involution = lazy.adjoint().unwrap();
        assert!(!involution.is_adjoint_view());
        assert!(Arc::ptr_eq(
            &involution.ordinary_body().space,
            &parent.ordinary_body().space
        ));
        assert!(Arc::ptr_eq(
            &involution.ordinary_body().data,
            &parent.ordinary_body().data
        ));
        assert_eq!(lazy.adjoint_build_counts(), (0, 0));
    }

    fn nested_product_space(degeneracies: [usize; 4]) -> Space {
        Space::fz2_u1_su2([
            ((0, 0, 0), degeneracies[0]),
            ((0, 0, 2), degeneracies[1]),
            ((1, -1, 1), degeneracies[2]),
            ((1, 1, 1), degeneracies[3]),
        ])
        .unwrap()
    }

    #[test]
    fn adjoint_transforms_match_eager_oracles_without_building_adjoint_grids() {
        // What: non-self-dual, fermionic, SU2 inner-line, and product
        // transforms lower to the parent for real and genuinely complex data.
        let spaces = [
            Space::u1([(-2, 1), (-1, 2), (0, 1), (1, 1)]),
            Space::fz2([(1, 2)]),
            Space::su2([(0, 1), (1, 2), (2, 1)]),
            nested_product_space([1, 1, 1, 1]),
        ];
        for (case, space) in spaces.into_iter().enumerate() {
            assert_adjoint_transforms_stay_parent_lowered(
                space.clone(),
                Dtype::F64,
                261_300 + case as u64 * 10,
            );
            assert_adjoint_transforms_stay_parent_lowered(
                space,
                Dtype::C64,
                261_301 + case as u64 * 10,
            );
        }
    }

    #[test]
    fn asymmetric_product_lazy_transpose_matches_eager_materialization() {
        // What: transpose of a complex 3|1 product adjoint preserves the exact
        // reversed split without materializing either lazy transform result.
        let runtime = Runtime::builder()
            .dense_threads(1)
            .recoupling_threads(1)
            .build()
            .unwrap();
        let first = nested_product_space([1, 1, 1, 1]);
        let second = nested_product_space([2, 1, 1, 1]);
        let third = nested_product_space([1, 2, 1, 1]);
        let domain = nested_product_space([1, 1, 2, 1]);
        let parent = Tensor::rand_with_seed(
            &runtime,
            Dtype::C64,
            [&first, &second, &third],
            [&domain],
            261_380,
        )
        .unwrap();
        assert!(parent
            .data_c64()
            .iter()
            .any(|value| value.im.abs() > f64::EPSILON));
        let lazy = parent.adjoint().unwrap();
        let eager = parent.adjoint().unwrap().materialized_tensor().unwrap();

        let actual = lazy.transpose().unwrap();
        let expected = eager.transpose().unwrap();

        assert_eq!(
            actual.codomain_spaces(),
            vec![third.dual(), second.dual(), first.dual()]
        );
        assert_eq!(actual.domain_spaces(), vec![domain.dual()]);
        assert_lowered_transform_matches_eager_oracle(&lazy, actual, expected);
    }

    #[test]
    fn adjoint_braid_levels_follow_tensorkit_parent_axis_order() {
        // What: a 3|1 parent's logical levels map to [3, 11, 5, 17], with
        // unchanged values, while the output tuples swap around the adjoint.
        let levels = [17, 3, 11, 5];
        let kind = TransformKind::Braid { levels: &levels };
        let lowered = lower_adjoint_transform_request(3, 1, &[3, 0, 2], &[1], &kind).unwrap();

        assert_eq!(lowered.codomain_axes, [0]);
        assert_eq!(lowered.domain_axes, [2, 3, 1]);
        assert_eq!(lowered.levels, [3, 11, 5, 17]);
    }

    #[test]
    fn lowered_adjoint_braid_preserves_the_exact_fermionic_swap_sign() {
        // What: swapping two odd fZ2 legs through a lazy adjoint keeps the
        // TensorKit fermionic minus sign and does not build an adjoint grid.
        let runtime = Runtime::builder().dense_threads(1).build().unwrap();
        let odd = Space::fz2([(1, 1)]);
        let parent = Tensor::from_block_fn(
            &runtime,
            [&odd, &odd],
            std::iter::empty::<&Space>(),
            |_, _| 1.0,
        )
        .unwrap();
        let lazy = parent.adjoint().unwrap();

        let transformed = lazy.braid(&[], &[1, 0], &[0, 1]).unwrap();
        let transformed_parent = transformed.adjoint().unwrap();

        assert!(transformed_parent.data().iter().all(|&value| value == -1.0));
        assert_eq!(lazy.adjoint_build_counts(), (0, 0));
        assert_eq!(transformed.adjoint_build_counts(), (0, 0));
    }

    #[test]
    fn malformed_adjoint_transform_errors_precede_any_view_build() {
        // What: level-count errors precede axis errors, and all invalid
        // transform requests leave the lazy adjoint representation untouched.
        let runtime = Runtime::builder().dense_threads(1).build().unwrap();
        let space = Space::u1([(-2, 1), (0, 2), (1, 1)]);
        let lazy = Tensor::rand_with_seed(
            &runtime,
            Dtype::C64,
            [&space, &space, &space],
            [&space],
            261_390,
        )
        .unwrap()
        .adjoint()
        .unwrap();

        let bad_levels_and_axes = lazy.braid(&[4, 0, 2], &[1], &[17, 3, 11]).unwrap_err();
        assert!(matches!(bad_levels_and_axes, Error::InvalidArgument(_)));
        assert_eq!(lazy.adjoint_build_counts(), (0, 0));

        let bad_axes = lazy.permute(&[4, 0, 2], &[1]).unwrap_err();
        let Error::Operation(error) = bad_axes else {
            panic!("invalid axes returned the wrong error layer");
        };
        assert!(matches!(
            error.as_ref(),
            OperationError::Core(tenet_core::CoreError::InvalidPermutation { .. })
        ));
        assert_eq!(lazy.adjoint_build_counts(), (0, 0));

        assert!(matches!(
            lazy.repartition(5),
            Err(Error::InvalidArgument(_))
        ));
        assert_eq!(lazy.adjoint_build_counts(), (0, 0));
    }

    fn assert_concurrent_raw_reads_initialize_one_shared_body(dtype: Dtype, seed: u64) {
        let runtime = Runtime::builder().dense_threads(1).build().unwrap();
        let space = Space::u1((-8..=8).map(|charge| (charge, 2)));
        let adjoint = Tensor::rand_with_seed(&runtime, dtype, [&space, &space], [&space], seed)
            .unwrap()
            .adjoint()
            .unwrap();
        let barrier = Arc::new(std::sync::Barrier::new(8));
        std::thread::scope(|scope| {
            for _ in 0..8 {
                let clone = adjoint.clone();
                let barrier = Arc::clone(&barrier);
                scope.spawn(move || {
                    barrier.wait();
                    match dtype {
                        Dtype::F64 => assert!(!clone.try_data().unwrap().is_empty()),
                        Dtype::C64 => assert!(!clone.try_data_c64().unwrap().is_empty()),
                    }
                });
            }
        });
        assert_eq!(adjoint.adjoint_build_counts(), (1, 1));
    }

    #[test]
    fn concurrent_raw_reads_initialize_one_shared_body() {
        // What: f64 and c64 clones racing on their first raw read publish each
        // derived logical space and materialized body exactly once.
        assert_concurrent_raw_reads_initialize_one_shared_body(Dtype::F64, 261_103);
        assert_concurrent_raw_reads_initialize_one_shared_body(Dtype::C64, 261_105);
    }

    fn assert_host_contract_stays_view_native(dtype: Dtype, seed: u64) {
        let runtime = Runtime::builder().dense_threads(2).build().unwrap();
        let space = Space::u1([(-2, 1), (-1, 2), (0, 3), (1, 2), (2, 1)]);
        let lhs = Tensor::rand_with_seed(&runtime, dtype, [&space, &space], [&space], seed)
            .unwrap()
            .adjoint()
            .unwrap();
        let rhs =
            Tensor::rand_with_seed(&runtime, dtype, [&space, &space], [&space], seed + 1).unwrap();

        let barrier = Arc::new(std::sync::Barrier::new(4));
        std::thread::scope(|scope| {
            for _ in 0..4 {
                let lhs = lhs.clone();
                let rhs = rhs.clone();
                let barrier = Arc::clone(&barrier);
                scope.spawn(move || {
                    barrier.wait();
                    assert_eq!(lhs.compose(&rhs).unwrap().rank(), 2);
                });
            }
        });
        assert_eq!(lhs.adjoint_build_counts(), (1, 0));

        for _ in 0..3 {
            let output = lhs.compose(&rhs).unwrap();
            assert_eq!(output.rank(), 2);
        }
        assert_eq!(lhs.adjoint_build_counts(), (1, 0));
    }

    #[test]
    fn fermionic_compose_keeps_lazy_lhs_and_rhs_parent_native() {
        // What: A† * B and A† * B† over the non-Abelian product read both
        // parent buffers through the Core batch without building owned adjoints.
        let runtime = Runtime::builder().dense_threads(1).build().unwrap();
        let space = Space::fz2_u1_su2([((0, 0, 0), 2), ((1, 0, 1), 2), ((1, 1, 2), 1)]).unwrap();
        let lhs_parent =
            Tensor::rand_with_seed(&runtime, Dtype::C64, [&space.dual()], [&space], 353_801)
                .unwrap();
        let rhs = Tensor::rand_with_seed(&runtime, Dtype::C64, [&space.dual()], [&space], 353_802)
            .unwrap();
        let rhs_parent =
            Tensor::rand_with_seed(&runtime, Dtype::C64, [&space], [&space.dual()], 353_803)
                .unwrap();
        let lhs = lhs_parent.adjoint().unwrap();
        let rhs_lazy = rhs_parent.adjoint().unwrap();

        let lhs_result = lhs.compose(&rhs).unwrap();
        assert_eq!(lhs.adjoint_build_counts(), (1, 0));
        let both_result = lhs.compose(&rhs_lazy).unwrap();
        assert_eq!(lhs.adjoint_build_counts(), (1, 0));
        assert_eq!(rhs_lazy.adjoint_build_counts(), (1, 0));

        let eager_lhs = lhs_parent.adjoint().unwrap().materialized_tensor().unwrap();
        let eager_rhs = rhs_parent.adjoint().unwrap().materialized_tensor().unwrap();
        assert_close(&lhs_result, &eager_lhs.compose(&rhs).unwrap());
        assert_close(&both_result, &eager_lhs.compose(&eager_rhs).unwrap());
    }

    fn assert_lazy_contract_matches_eager_oracle(space: Space, seed: u64) {
        let runtime = Runtime::builder()
            .dense_threads(1)
            .recoupling_threads(1)
            .build()
            .unwrap();
        let lhs_parent =
            Tensor::rand_with_seed(&runtime, Dtype::C64, [&space], [&space, &space], seed).unwrap();
        let rhs_parent =
            Tensor::rand_with_seed(&runtime, Dtype::C64, [&space], [&space, &space], seed + 1)
                .unwrap();
        let plain =
            Tensor::rand_with_seed(&runtime, Dtype::C64, [&space, &space], [&space], seed + 2)
                .unwrap();
        let lhs_eager = lhs_parent.adjoint().unwrap().materialized_tensor().unwrap();
        let rhs_eager = rhs_parent.adjoint().unwrap().materialized_tensor().unwrap();
        let output_axes = [2, 0, 3, 1];

        for (lhs, rhs, eager_lhs, eager_rhs) in [
            (
                lhs_parent.adjoint().unwrap(),
                plain.clone(),
                lhs_eager.clone(),
                plain.clone(),
            ),
            (
                plain.clone(),
                rhs_parent.adjoint().unwrap(),
                plain.clone(),
                rhs_eager.clone(),
            ),
            (
                lhs_parent.adjoint().unwrap(),
                rhs_parent.adjoint().unwrap(),
                lhs_eager.clone(),
                rhs_eager.clone(),
            ),
        ] {
            let expected = eager_lhs
                .contract_ordered(&eager_rhs, &[2], &[0], &output_axes)
                .unwrap();
            let actual = lhs
                .contract_ordered(&rhs, &[2], &[0], &output_axes)
                .unwrap();
            assert_close(&actual, &expected);
            assert_eq!(lhs.adjoint_build_counts().1, 0);
            assert_eq!(rhs.adjoint_build_counts().1, 0);
        }
    }

    fn assert_lazy_core_adjoint_matches_eager(
        rows: Space,
        contracted: Space,
        cols: Space,
        dtype: Dtype,
        seed: u64,
    ) {
        let runtime = Runtime::builder().dense_threads(1).build().unwrap();
        let lhs_parent =
            Tensor::rand_with_seed(&runtime, dtype, [&contracted], [&rows], seed).unwrap();
        let rhs_parent =
            Tensor::rand_with_seed(&runtime, dtype, [&cols], [&contracted], seed + 1).unwrap();
        let lhs_direct =
            Tensor::rand_with_seed(&runtime, dtype, [&rows], [&contracted], seed + 2).unwrap();
        let rhs_direct =
            Tensor::rand_with_seed(&runtime, dtype, [&contracted], [&cols], seed + 3).unwrap();

        let lhs_lazy = lhs_parent.adjoint().unwrap();
        let rhs_lazy = rhs_parent.adjoint().unwrap();
        let lhs_eager = lhs_lazy.materialized_tensor().unwrap();
        let rhs_eager = rhs_lazy.materialized_tensor().unwrap();
        for (lhs, rhs, eager_lhs, eager_rhs) in [
            (
                lhs_lazy.clone(),
                rhs_direct.clone(),
                lhs_eager.clone(),
                rhs_direct.clone(),
            ),
            (
                lhs_direct.clone(),
                rhs_lazy.clone(),
                lhs_direct.clone(),
                rhs_eager.clone(),
            ),
            (lhs_lazy, rhs_lazy, lhs_eager, rhs_eager),
        ] {
            let expected = eager_lhs.compose(&eager_rhs).unwrap();
            let actual = lhs.compose(&rhs).unwrap();
            assert_close(&actual, &expected);
        }
    }

    #[test]
    fn repeated_and_parallel_host_contractions_do_not_materialize_adjoint_data() {
        // What: f64 and c64 contraction reuse one logical space and parent storage.
        assert_host_contract_stays_view_native(Dtype::F64, 261_104);
        assert_host_contract_stays_view_native(Dtype::C64, 261_106);
    }

    #[test]
    fn lazy_contraction_matches_eager_oracles_for_supported_rule_families() {
        // What: lhs, rhs, and double adjoints preserve non-self-dual labels,
        // recoupling coefficients, fermionic signs, and crossed output order.
        assert_lazy_contract_matches_eager_oracle(Space::u1([(-2, 1), (-1, 2), (1, 3)]), 261_201);
        assert_lazy_contract_matches_eager_oracle(Space::su2([(0, 2), (1, 3), (2, 1)]), 261_211);
        assert_lazy_contract_matches_eager_oracle(Space::fz2([(0, 2), (1, 3)]), 261_221);
        assert_lazy_contract_matches_eager_oracle(
            Space::fz2_u1_su2([
                ((0, 0, 0), 2),
                ((1, -1, 1), 2),
                ((1, 1, 1), 1),
                ((0, 2, 2), 1),
            ])
            .unwrap(),
            261_231,
        );
        assert_lazy_contract_matches_eager_oracle(
            Space::fz2_u1_su2([((0, 0, 0), 2), ((1, 0, 1), 2), ((0, 0, 2), 1)]).unwrap(),
            261_241,
        );
    }

    #[test]
    fn lazy_core_adjoint_handles_rectangular_real_and_complex_blocks() {
        // What: lhs, rhs, and both-adjoint Core replay transposes rectangular
        // parent matrices and conjugates complex values exactly once.
        for (dtype, seed) in [(Dtype::F64, 272_100), (Dtype::C64, 272_200)] {
            assert_lazy_core_adjoint_matches_eager(
                Space::su2([(0, 2), (1, 1)]),
                Space::su2([(0, 1), (1, 3)]),
                Space::su2([(0, 3), (1, 2)]),
                dtype,
                seed,
            );
            assert_lazy_core_adjoint_matches_eager(
                Space::u1([(-1, 2), (0, 1), (1, 1)]),
                Space::u1([(-1, 1), (0, 2), (1, 3)]),
                Space::u1([(-1, 3), (0, 1), (1, 2)]),
                dtype,
                seed + 20,
            );
            assert_lazy_core_adjoint_matches_eager(
                Space::fz2([(0, 2), (1, 1)]),
                Space::fz2([(0, 1), (1, 3)]),
                Space::fz2([(0, 3), (1, 2)]),
                dtype,
                seed + 40,
            );
            assert_lazy_core_adjoint_matches_eager(
                Space::fz2_u1_su2([((0, 0, 0), 2), ((1, -1, 1), 1), ((1, 1, 1), 1)]).unwrap(),
                Space::fz2_u1_su2([((0, 0, 0), 1), ((1, -1, 1), 3), ((1, 1, 1), 2)]).unwrap(),
                Space::fz2_u1_su2([((0, 0, 0), 3), ((1, -1, 1), 1), ((1, 1, 1), 2)]).unwrap(),
                dtype,
                seed + 60,
            );
        }
    }
}

#[cfg(test)]
mod shared_context_tests {
    use super::*;

    /// Every runtime-minted executor shares one CPU context, avoiding one
    /// eager rayon pool per rule, dtype, and concurrent lease.
    #[test]
    fn runtime_and_leased_contexts_share_one_cpu_context() {
        let rt = Runtime::builder().build().expect("runtime");
        let shared = rt.execution_config().shared_ctx.clone();
        {
            let mut state = rt.lock();
            assert!(state.shares_cpu_context(&shared));
        }

        let mut lease = rt.lease_context().expect("lease");
        assert!(lease.context().shares_cpu_context(&shared));
        let mut network_context = TensorExecutionContext::for_runtime(&rt).expect("context");
        assert!(network_context.shares_cpu_context(&shared));
    }

    #[test]
    fn runtime_builder_recoupling_threads_reach_every_runtime_and_context_lane() {
        fn assert_runtime_and_context(runtime: &Runtime, expected: usize) {
            {
                let mut state = runtime.lock();
                assert!(state.recoupling_threads_are(expected));
            }

            let mut context = TensorExecutionContext::for_runtime(runtime).expect("context");
            assert!(context.recoupling_threads_are(expected));
        }

        let configured = Runtime::builder()
            .recoupling_threads(3)
            .build()
            .expect("configured runtime");
        assert_runtime_and_context(&configured, 3);

        let default = Runtime::builder().build().expect("default runtime");
        assert_runtime_and_context(&default, 1);
    }
}

#[cfg(test)]
mod bound_provider_tests {
    use super::*;

    #[test]
    fn construction_and_svd_factors_share_one_provider_allocation() {
        // What: construction and owned factors retain the originating provider allocation.
        let runtime = Runtime::builder().build().unwrap();
        let space = Space::z2([(0, 2), (1, 1)]);
        let provider = space.rule_context().as_ref().clone();
        let tensor = Tensor::rand_with_seed(&runtime, Dtype::F64, [&space], [&space], 7).unwrap();
        assert!(tensor
            .ordinary_body()
            .space
            .provider_matches_context_allocation(&provider));

        let (u, s, vh) = tensor.svd_compact().unwrap();
        for factor in [&u, &s, &vh] {
            assert!(factor
                .ordinary_body()
                .space
                .provider_matches_context_allocation(&provider));
        }
    }

    #[test]
    fn destination_execution_preserves_its_provider_allocation() {
        // What: overwrite execution never replaces the destination's provider allocation.
        let runtime = Runtime::builder().build().unwrap();
        let space = Space::product([((0, 0), 2), ((1, 1), 1)]).unwrap();
        let lhs = Tensor::rand_with_seed(&runtime, Dtype::F64, [&space], [&space], 11).unwrap();
        let rhs = Tensor::rand_with_seed(&runtime, Dtype::F64, [&space], [&space], 13).unwrap();
        let mut dst = lhs.contract(&rhs, &[1], &[0]).unwrap();
        let before = dst.ordinary_body().space.context();
        let mut execution = TensorExecutionContext::default();

        execution
            .contract_overwrite_into(&mut dst, &lhs, &rhs, &[1], &[0], Scalar::F64(1.0))
            .unwrap();

        assert!(dst
            .ordinary_body()
            .space
            .provider_matches_context_allocation(&before));
    }

    #[test]
    fn permute_overwrite_forwards_poisoned_destination_to_replay() {
        // What: the top-level destination boundary does not clear logical data
        // before handing it to the explicit overwrite replay.
        let runtime = Runtime::builder().build().unwrap();
        let space = Space::su2([(0, 2), (1, 2), (2, 1)]);
        let source =
            Tensor::rand_with_seed(&runtime, Dtype::F64, [&space, &space], [&space], 17).unwrap();
        let expected = source.permute(&[1], &[2, 0]).unwrap();
        let mut destination = expected.scale(f64::NAN).unwrap();
        let mut execution = TensorExecutionContext::for_runtime(&runtime).unwrap();

        PERMUTE_PRE_REPLAY_POISON.with(|observation| observation.set(Some(false)));
        execution
            .permute_overwrite_into(&mut destination, &source, &[1], &[2, 0], Scalar::F64(1.0))
            .unwrap();
        let observed = PERMUTE_PRE_REPLAY_POISON.with(|observation| observation.replace(None));

        assert_eq!(observed, Some(true));
        assert_eq!(destination.data(), expected.data());
    }

    #[test]
    fn contract_cache_reuses_semantically_equal_spaces_and_retains_destination_authority() {
        // What: cache reuse follows semantic layout identity while writes retain a distinct destination Arc.
        let runtime = Runtime::builder().build().unwrap();
        let lhs_space = Space::z2([(0, 2), (1, 1)]);
        let rhs_space = Space::z2([(0, 2), (1, 1)]);
        let dst_space = Space::z2([(0, 2), (1, 1)]);
        let lhs =
            Tensor::rand_with_seed(&runtime, Dtype::F64, [&lhs_space], [&lhs_space], 21).unwrap();
        let rhs =
            Tensor::rand_with_seed(&runtime, Dtype::F64, [&rhs_space], [&rhs_space], 22).unwrap();
        let oracle = lhs.contract(&rhs, &[1], &[0]).unwrap();
        let mut dst = Tensor::zeros(&runtime, Dtype::F64, [&dst_space], [&dst_space]).unwrap();
        let destination_provider = dst.ordinary_body().space.context();
        let mut execution = TensorExecutionContext::for_runtime(&runtime).unwrap();
        let mut cache = ContractOverwriteCache::default();

        assert_eq!(
            execution
                .try_contract_overwrite_into(
                    &mut cache,
                    &mut dst,
                    &lhs,
                    &rhs,
                    &[1],
                    &[0],
                    Scalar::F64(1.0),
                )
                .unwrap(),
            OverwriteOutcome::Written
        );
        assert_eq!(dst.data(), oracle.data());
        assert!(dst
            .ordinary_body()
            .space
            .provider_matches_context_allocation(&destination_provider));

        let lhs_space_2 = Space::z2([(0, 2), (1, 1)]);
        let rhs_space_2 = Space::z2([(0, 2), (1, 1)]);
        let lhs_2 =
            Tensor::rand_with_seed(&runtime, Dtype::F64, [&lhs_space_2], [&lhs_space_2], 23)
                .unwrap();
        let rhs_2 =
            Tensor::rand_with_seed(&runtime, Dtype::F64, [&rhs_space_2], [&rhs_space_2], 24)
                .unwrap();
        let oracle_2 = lhs_2.contract(&rhs_2, &[1], &[0]).unwrap();
        assert_eq!(
            execution
                .try_contract_overwrite_into(
                    &mut cache,
                    &mut dst,
                    &lhs_2,
                    &rhs_2,
                    &[1],
                    &[0],
                    Scalar::F64(1.0),
                )
                .unwrap(),
            OverwriteOutcome::Written
        );
        assert_eq!(cache.preparations, 1);
        assert_eq!(dst.data(), oracle_2.data());
        assert!(dst
            .ordinary_body()
            .space
            .provider_matches_context_allocation(&destination_provider));
    }

    #[test]
    fn permute_cache_reuses_semantically_equal_spaces_and_retains_destination_authority() {
        // What: permutation cache reuse is allocation-independent and never rebinds destination authority.
        let runtime = Runtime::builder().build().unwrap();
        let source_space = Space::z2([(0, 2), (1, 1)]);
        let destination_space = Space::z2([(0, 2), (1, 1)]);
        let source = Tensor::rand_with_seed(
            &runtime,
            Dtype::F64,
            [&source_space, &source_space],
            [&source_space],
            31,
        )
        .unwrap();
        let oracle = source.permute(&[1, 0], &[2]).unwrap();
        let mut dst = Tensor::zeros(
            &runtime,
            Dtype::F64,
            [&destination_space, &destination_space],
            [&destination_space],
        )
        .unwrap();
        let destination_provider = dst.ordinary_body().space.context();
        let mut execution = TensorExecutionContext::for_runtime(&runtime).unwrap();
        let mut cache = PermuteOverwriteCache::default();
        assert_eq!(
            execution
                .try_permute_overwrite_into(
                    &mut cache,
                    &mut dst,
                    &source,
                    &[1, 0],
                    &[2],
                    Scalar::F64(1.0),
                )
                .unwrap(),
            OverwriteOutcome::Written
        );
        assert_eq!(dst.data(), oracle.data());

        let source_space_2 = Space::z2([(0, 2), (1, 1)]);
        let source_2 = Tensor::rand_with_seed(
            &runtime,
            Dtype::F64,
            [&source_space_2, &source_space_2],
            [&source_space_2],
            32,
        )
        .unwrap();
        let oracle_2 = source_2.permute(&[1, 0], &[2]).unwrap();
        assert_eq!(
            execution
                .try_permute_overwrite_into(
                    &mut cache,
                    &mut dst,
                    &source_2,
                    &[1, 0],
                    &[2],
                    Scalar::F64(1.0),
                )
                .unwrap(),
            OverwriteOutcome::Written
        );
        assert_eq!(cache.preparations, 1);
        assert_eq!(dst.data(), oracle_2.data());
        assert!(dst
            .ordinary_body()
            .space
            .provider_matches_context_allocation(&destination_provider));
    }
}

#[cfg(test)]
mod tk_user_api_tests {
    use super::*;
    use std::collections::{BTreeMap, BTreeSet};
    use tenet_core::{
        Fz2SectorLayout, PackedProductCodec, ProductSectorCodec, ProductSectorLayout,
        Su2SectorLayout, U1Irrep, U1SectorLayout,
    };

    type NestedLabel = (usize, i32, usize);
    type NestedInnerCodec = PackedProductCodec<Fz2SectorLayout, U1SectorLayout>;
    type NestedInnerLayout = ProductSectorLayout<Fz2SectorLayout, U1SectorLayout>;
    type NestedOuterCodec = PackedProductCodec<NestedInnerLayout, Su2SectorLayout>;

    const E: NestedLabel = (0, 0, 0);
    const O: NestedLabel = (1, 0, 1);
    const T: NestedLabel = (0, 0, 2);

    #[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
    struct NestedSemanticElement {
        codomain: Vec<NestedLabel>,
        domain: Vec<NestedLabel>,
        codomain_inner: Vec<NestedLabel>,
        domain_inner: Vec<NestedLabel>,
        codomain_is_dual: Vec<bool>,
        domain_is_dual: Vec<bool>,
        coupled: NestedLabel,
        indices: Vec<usize>,
    }

    fn nested_element(
        codomain: &[NestedLabel],
        domain: &[NestedLabel],
        codomain_inner: &[NestedLabel],
        domain_inner: &[NestedLabel],
        coupled: NestedLabel,
        indices: &[usize],
    ) -> NestedSemanticElement {
        NestedSemanticElement {
            codomain: codomain.to_vec(),
            domain: domain.to_vec(),
            codomain_inner: codomain_inner.to_vec(),
            domain_inner: domain_inner.to_vec(),
            codomain_is_dual: vec![false; codomain.len()],
            domain_is_dual: vec![false; domain.len()],
            coupled,
            indices: indices.to_vec(),
        }
    }

    fn nested_label(sector: SectorId) -> NestedLabel {
        let (inner, spin) = NestedOuterCodec::decode(sector).unwrap();
        let (parity, charge) = NestedInnerCodec::decode(inner).unwrap();
        (
            parity.id(),
            U1Irrep::from_sector_id(charge).unwrap().charge(),
            spin.id(),
        )
    }

    fn normalized_nested_element(key: &BlockKey, indices: &[usize]) -> NestedSemanticElement {
        let BlockKey::FusionTree(key) = key else {
            panic!("nested product fixture must use fusion-tree blocks");
        };
        let labels = |sectors: &[SectorId]| {
            sectors
                .iter()
                .copied()
                .map(nested_label)
                .collect::<Vec<_>>()
        };
        NestedSemanticElement {
            codomain: labels(key.codomain_uncoupled()),
            domain: labels(key.domain_uncoupled()),
            codomain_inner: labels(key.codomain_innerlines()),
            domain_inner: labels(key.domain_innerlines()),
            codomain_is_dual: key.codomain_is_dual().to_vec(),
            domain_is_dual: key.domain_is_dual().to_vec(),
            coupled: nested_label(key.coupled()),
            indices: indices.to_vec(),
        }
    }

    // Why not derive these orders through the legacy Cantor codec: the
    // coefficient oracle must survive another internal SectorId encoding
    // change. These normalized keys are copied from the TensorKit oracle.
    fn nested_source_order() -> Vec<NestedSemanticElement> {
        vec![
            nested_element(&[E, E], &[E, E], &[], &[], E, &[0, 0, 0, 0]),
            nested_element(&[O, O], &[E, E], &[], &[], E, &[0, 0, 0, 0]),
            nested_element(&[E, E], &[O, O], &[], &[], E, &[0, 0, 0, 0]),
            nested_element(&[O, O], &[O, O], &[], &[], E, &[0, 0, 0, 0]),
            nested_element(&[E, E], &[O, O], &[], &[], E, &[0, 0, 0, 1]),
            nested_element(&[O, O], &[O, O], &[], &[], E, &[0, 0, 0, 1]),
            nested_element(&[O, O], &[O, O], &[], &[], T, &[0, 0, 0, 0]),
            nested_element(&[O, O], &[O, O], &[], &[], T, &[0, 0, 0, 1]),
            nested_element(&[O, E], &[O, E], &[], &[], O, &[0, 0, 0, 0]),
            nested_element(&[E, O], &[O, E], &[], &[], O, &[0, 0, 0, 0]),
            nested_element(&[O, E], &[E, O], &[], &[], O, &[0, 0, 0, 0]),
            nested_element(&[E, O], &[E, O], &[], &[], O, &[0, 0, 0, 0]),
            nested_element(&[O, E], &[E, O], &[], &[], O, &[0, 0, 0, 1]),
            nested_element(&[E, O], &[E, O], &[], &[], O, &[0, 0, 0, 1]),
        ]
    }

    fn nested_repartition_3_order() -> Vec<NestedSemanticElement> {
        let mut order = vec![
            nested_element(&[E, E, E], &[E], &[E], &[], E, &[0, 0, 0, 0]),
            nested_element(&[O, O, E], &[E], &[E], &[], E, &[0, 0, 0, 0]),
            nested_element(&[O, E, O], &[E], &[O], &[], E, &[0, 0, 0, 0]),
            nested_element(&[O, E, O], &[E], &[O], &[], E, &[0, 0, 1, 0]),
            nested_element(&[E, O, O], &[E], &[O], &[], E, &[0, 0, 0, 0]),
            nested_element(&[E, O, O], &[E], &[O], &[], E, &[0, 0, 1, 0]),
            nested_element(&[O, E, E], &[O], &[O], &[], O, &[0, 0, 0, 0]),
            nested_element(&[E, O, E], &[O], &[O], &[], O, &[0, 0, 0, 0]),
            nested_element(&[E, E, O], &[O], &[E], &[], O, &[0, 0, 0, 0]),
            nested_element(&[E, E, O], &[O], &[E], &[], O, &[0, 0, 1, 0]),
            nested_element(&[O, O, O], &[O], &[E], &[], O, &[0, 0, 0, 0]),
            nested_element(&[O, O, O], &[O], &[E], &[], O, &[0, 0, 1, 0]),
            nested_element(&[O, O, O], &[O], &[T], &[], O, &[0, 0, 0, 0]),
            nested_element(&[O, O, O], &[O], &[T], &[], O, &[0, 0, 1, 0]),
        ];
        for element in &mut order {
            element.codomain_is_dual = vec![false, false, true];
        }
        order
    }

    fn nested_repartition_1_order() -> Vec<NestedSemanticElement> {
        let mut order = vec![
            nested_element(&[E], &[E, E, E], &[], &[E], E, &[0, 0, 0, 0]),
            nested_element(&[E], &[O, O, E], &[], &[E], E, &[0, 0, 0, 0]),
            nested_element(&[E], &[O, O, E], &[], &[E], E, &[0, 0, 1, 0]),
            nested_element(&[E], &[O, E, O], &[], &[O], E, &[0, 0, 0, 0]),
            nested_element(&[E], &[E, O, O], &[], &[O], E, &[0, 0, 0, 0]),
            nested_element(&[E], &[E, O, O], &[], &[O], E, &[0, 0, 1, 0]),
            nested_element(&[O], &[O, E, E], &[], &[O], O, &[0, 0, 0, 0]),
            nested_element(&[O], &[E, O, E], &[], &[O], O, &[0, 0, 0, 0]),
            nested_element(&[O], &[E, O, E], &[], &[O], O, &[0, 0, 1, 0]),
            nested_element(&[O], &[E, E, O], &[], &[E], O, &[0, 0, 0, 0]),
            nested_element(&[O], &[O, O, O], &[], &[E], O, &[0, 0, 0, 0]),
            nested_element(&[O], &[O, O, O], &[], &[E], O, &[0, 0, 1, 0]),
            nested_element(&[O], &[O, O, O], &[], &[T], O, &[0, 0, 0, 0]),
            nested_element(&[O], &[O, O, O], &[], &[T], O, &[0, 0, 1, 0]),
        ];
        for element in &mut order {
            element.domain_is_dual = vec![false, false, true];
        }
        order
    }

    fn nested_repartition_0_order() -> Vec<NestedSemanticElement> {
        let mut order = vec![
            nested_element(&[], &[E, E, E, E], &[], &[E, E], E, &[0, 0, 0, 0]),
            nested_element(&[], &[O, O, E, E], &[], &[E, E], E, &[0, 0, 0, 0]),
            nested_element(&[], &[O, O, E, E], &[], &[E, E], E, &[0, 1, 0, 0]),
            nested_element(&[], &[O, E, O, E], &[], &[O, E], E, &[0, 0, 0, 0]),
            nested_element(&[], &[E, O, O, E], &[], &[O, E], E, &[0, 0, 0, 0]),
            nested_element(&[], &[E, O, O, E], &[], &[O, E], E, &[0, 1, 0, 0]),
            nested_element(&[], &[O, E, E, O], &[], &[O, O], E, &[0, 0, 0, 0]),
            nested_element(&[], &[E, O, E, O], &[], &[O, O], E, &[0, 0, 0, 0]),
            nested_element(&[], &[E, O, E, O], &[], &[O, O], E, &[0, 1, 0, 0]),
            nested_element(&[], &[E, E, O, O], &[], &[E, O], E, &[0, 0, 0, 0]),
            nested_element(&[], &[O, O, O, O], &[], &[E, O], E, &[0, 0, 0, 0]),
            nested_element(&[], &[O, O, O, O], &[], &[E, O], E, &[0, 1, 0, 0]),
            nested_element(&[], &[O, O, O, O], &[], &[T, O], E, &[0, 0, 0, 0]),
            nested_element(&[], &[O, O, O, O], &[], &[T, O], E, &[0, 1, 0, 0]),
        ];
        for element in &mut order {
            element.domain_is_dual = vec![false, false, true, true];
        }
        order
    }

    fn nested_repartition_4_order() -> Vec<NestedSemanticElement> {
        let mut order = vec![
            nested_element(&[E, E, E, E], &[], &[E, E], &[], E, &[0, 0, 0, 0]),
            nested_element(&[O, O, E, E], &[], &[E, E], &[], E, &[0, 0, 0, 0]),
            nested_element(&[O, E, O, E], &[], &[O, E], &[], E, &[0, 0, 0, 0]),
            nested_element(&[O, E, O, E], &[], &[O, E], &[], E, &[0, 0, 1, 0]),
            nested_element(&[E, O, O, E], &[], &[O, E], &[], E, &[0, 0, 0, 0]),
            nested_element(&[E, O, O, E], &[], &[O, E], &[], E, &[0, 0, 1, 0]),
            nested_element(&[O, E, E, O], &[], &[O, O], &[], E, &[0, 0, 0, 0]),
            nested_element(&[E, O, E, O], &[], &[O, O], &[], E, &[0, 0, 0, 0]),
            nested_element(&[E, E, O, O], &[], &[E, O], &[], E, &[0, 0, 0, 0]),
            nested_element(&[E, E, O, O], &[], &[E, O], &[], E, &[0, 0, 1, 0]),
            nested_element(&[O, O, O, O], &[], &[E, O], &[], E, &[0, 0, 0, 0]),
            nested_element(&[O, O, O, O], &[], &[E, O], &[], E, &[0, 0, 1, 0]),
            nested_element(&[O, O, O, O], &[], &[T, O], &[], E, &[0, 0, 0, 0]),
            nested_element(&[O, O, O, O], &[], &[T, O], &[], E, &[0, 0, 1, 0]),
        ];
        for element in &mut order {
            element.codomain_is_dual = vec![false, false, true, true];
        }
        order
    }

    fn nested_semantic_sequence_tensor(
        rt: &Runtime,
        codomain: &[&Space],
        domain: &[&Space],
    ) -> Tensor {
        // What: TensorKit's sequential fixture values are attached to
        // normalized fusion-tree elements rather than TeNeT storage offsets.
        let order = nested_source_order();
        let assignments = order
            .iter()
            .cloned()
            .enumerate()
            .map(|(index, semantic)| (semantic, (index + 1) as f64))
            .collect::<BTreeMap<_, _>>();
        assert_eq!(
            assignments.len(),
            order.len(),
            "TensorKit source oracle contains duplicate elements"
        );
        let mut visited = BTreeSet::new();
        let tensor = Tensor::from_block_fn(
            rt,
            codomain.iter().copied(),
            domain.iter().copied(),
            |key, indices| {
                let element = normalized_nested_element(key, indices);
                assert!(
                    visited.insert(element.clone()),
                    "TeNeT enumerated a duplicate source semantic element"
                );
                assignments
                    .get(&element)
                    .copied()
                    .expect("TensorKit source oracle covers every semantic element")
            },
        )
        .unwrap();
        assert_eq!(
            visited,
            assignments.keys().cloned().collect(),
            "TensorKit and TeNeT source semantic element sets differ"
        );
        tensor
    }

    fn assert_nested_semantic_fixture(
        actual: &Tensor,
        order: &[NestedSemanticElement],
        expected: &[f64],
    ) {
        // What: each expected coefficient/sign is selected by normalized
        // fusion-tree labels, dual flags, and local degeneracy indices, with
        // exact coverage and no duplicate or extra elements.
        assert_eq!(order.len(), expected.len());
        let expected = order
            .iter()
            .cloned()
            .zip(expected.iter().copied())
            .collect::<BTreeMap<_, _>>();
        assert_eq!(
            expected.len(),
            order.len(),
            "TensorKit semantic oracle contains duplicate elements"
        );

        let structure = actual.ordinary_body().space.structure();
        let mut observed = BTreeMap::new();
        for block_index in 0..structure.block_count() {
            let block = structure.block(block_index).unwrap();
            let mut indices = vec![0usize; block.shape().len()];
            let count: usize = block.shape().iter().product();
            for _ in 0..count {
                let position = block.offset()
                    + indices
                        .iter()
                        .zip(block.strides())
                        .map(|(&index, &stride)| index * stride)
                        .sum::<usize>();
                let semantic = normalized_nested_element(block.key(), &indices);
                assert!(
                    observed.insert(semantic, actual.data()[position]).is_none(),
                    "TeNeT produced a duplicate normalized semantic element"
                );
                for (axis, index) in indices.iter_mut().enumerate() {
                    *index += 1;
                    if *index < block.shape()[axis] {
                        break;
                    }
                    *index = 0;
                }
            }
        }
        assert_eq!(observed.len(), actual.data().len());
        assert_eq!(
            observed.keys().collect::<Vec<_>>(),
            expected.keys().collect::<Vec<_>>(),
            "TensorKit and TeNeT semantic element sets differ"
        );
        for (semantic, expected) in expected {
            let actual = observed[&semantic];
            assert!(
                (actual - expected).abs() < 1.0e-12,
                "TensorKit semantic fixture mismatch for {semantic:?}: \
                 actual={actual}, expected={expected}"
            );
        }
    }

    fn sequential_f64_tensor(rt: &Runtime, codomain: &[&Space], domain: &[&Space]) -> Tensor {
        let mut tensor = Tensor::zeros(
            rt,
            Dtype::F64,
            codomain.iter().copied(),
            domain.iter().copied(),
        )
        .unwrap();
        let body = tensor.owned_body_mut().unwrap();
        let Data::F64(data) = Arc::get_mut(&mut body.data).unwrap() else {
            unreachable!("requested f64 tensor")
        };
        for (index, value) in data.iter_mut().enumerate() {
            *value = (index + 1) as f64;
        }
        tensor
    }

    fn assert_external_axis_order(output: &Tensor, source: &Tensor, axes: &[usize]) {
        assert_eq!(output.rank(), axes.len());
        for (output_axis, &source_axis) in axes.iter().enumerate() {
            assert_eq!(
                output.space(output_axis).unwrap(),
                source.space(source_axis).unwrap()
            );
        }
    }

    fn assert_tensorkit_fixture(actual: &[f64], expected: &[f64]) {
        assert_eq!(actual.len(), expected.len());
        for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
            assert!(
                (actual - expected).abs() < 1.0e-12,
                "TensorKit fixture mismatch at {index}: actual={actual}, expected={expected}"
            );
        }
    }

    #[test]
    fn index_count_aliases_match_rank_accessors() {
        // What: numout/numin/numind are exact TK-named aliases of the rank accessors.
        let rt = Runtime::builder().build().unwrap();
        let v = Space::u1([(0, 2), (1, 1)]);
        let t = Tensor::rand(&rt, Dtype::F64, [&v, &v], [&v]).unwrap();
        assert_eq!(t.numout(), t.codomain_rank());
        assert_eq!(t.numin(), t.domain_rank());
        assert_eq!(t.numind(), t.rank());
        assert_eq!((t.numout(), t.numin(), t.numind()), (2, 1, 3));
    }

    #[test]
    fn repartition_moves_the_split_and_round_trips() {
        // What: repartition re-splits legs at the given codomain count, invertibly.
        let rt = Runtime::builder().build().unwrap();
        let v = Space::u1([(0, 2), (1, 1)]);
        let t = Tensor::rand(&rt, Dtype::F64, [&v, &v], [&v]).unwrap();
        let r = t.repartition(1).unwrap();
        assert_eq!((r.codomain_rank(), r.domain_rank()), (1, 2));
        // Back to the original split recovers the original data (planar move).
        let back = r.repartition(2).unwrap();
        assert_eq!(back.data(), t.data());
        assert!(t.repartition(4).is_err());
    }

    #[test]
    fn repartition_uses_tensorkit_planar_axis_order_for_heterogeneous_u1_legs() {
        // What: a 2|2 -> 3|1 repartition moves the last domain leg across the
        // boundary and matches `tensorkit_semantic_oracle.out` section 4,
        // `U1 2|2 -> 3|1`.
        let rt = Runtime::builder().build().unwrap();
        let a = Space::u1([(0, 1)]);
        let b = Space::u1([(0, 2)]);
        let c = Space::u1([(0, 3)]);
        let d = Space::u1([(0, 4)]);
        let source = sequential_f64_tensor(&rt, &[&a, &b], &[&c, &d]);

        let output = source.repartition(3).unwrap();

        assert_eq!((output.codomain_rank(), output.domain_rank()), (3, 1));
        assert_external_axis_order(&output, &source, &[0, 1, 3, 2]);
        assert_tensorkit_fixture(
            output.data(),
            &[
                1.0, 2.0, 7.0, 8.0, 13.0, 14.0, 19.0, 20.0, 3.0, 4.0, 9.0, 10.0, 15.0, 16.0, 21.0,
                22.0, 5.0, 6.0, 11.0, 12.0, 17.0, 18.0, 23.0, 24.0,
            ],
        );
    }

    #[test]
    fn repartition_same_split_shares_storage_without_transforming() {
        // What: repartitioning to the current split is a zero-copy no-op.
        let rt = Runtime::builder().build().unwrap();
        let v = Space::su2([(0, 1), (1, 2)]);
        let source = Tensor::rand_with_seed(&rt, Dtype::F64, [&v, &v], [&v], 191).unwrap();

        let output = source.repartition(source.codomain_rank()).unwrap();

        assert!(Arc::ptr_eq(
            &output.ordinary_body().space,
            &source.ordinary_body().space
        ));
        assert!(Arc::ptr_eq(
            &output.ordinary_body().data,
            &source.ordinary_body().data
        ));
    }

    #[test]
    fn identity_braid_shares_storage_for_multiplicity_free_rules() {
        // What: exact-axis braids share owned storage for fermionic,
        // non-Abelian, and nested-product tensors even with nonmonotone levels.
        let rt = Runtime::builder().build().unwrap();
        let spaces = [
            Space::fz2([(0, 1), (1, 2)]),
            Space::su2([(0, 1), (1, 2), (2, 1)]),
            Space::fz2_u1_su2([((0, 0, 0), 1), ((1, 1, 1), 2)]).unwrap(),
        ];

        for (case, space) in spaces.iter().enumerate() {
            let source =
                Tensor::rand_with_seed(&rt, Dtype::F64, [space, space], [space], 200 + case as u64)
                    .unwrap();
            let output = source.braid(&[0, 1], &[2], &[17, 3, 11]).unwrap();

            assert!(
                Arc::ptr_eq(&output.ordinary_body().space, &source.ordinary_body().space),
                "case {case}"
            );
            assert!(
                Arc::ptr_eq(&output.ordinary_body().data, &source.ordinary_body().data),
                "case {case}"
            );
        }
    }

    #[test]
    fn identity_braid_validates_levels_before_sharing() {
        // What: malformed braid levels remain an error even when the axis map
        // itself is the identity.
        let rt = Runtime::builder().build().unwrap();
        let space = Space::fz2([(0, 1), (1, 1)]);
        let source =
            Tensor::rand_with_seed(&rt, Dtype::F64, [&space, &space], [&space], 203).unwrap();

        assert!(source.braid(&[0, 1], &[2], &[7, 5]).is_err());
    }

    #[test]
    fn identity_braid_shares_rank_zero_storage() {
        // What: the empty axis map is a zero-copy identity braid for a scalar.
        let rt = Runtime::builder().build().unwrap();
        let space = Space::u1([(0, 1)]);
        let vector =
            Tensor::rand_with_seed(&rt, Dtype::F64, [&space], std::iter::empty::<&Space>(), 204)
                .unwrap();
        let scalar = vector
            .contract(&vector.adjoint().unwrap(), &[0], &[0])
            .unwrap();

        let output = scalar.braid(&[], &[], &[]).unwrap();

        assert!(Arc::ptr_eq(
            &output.ordinary_body().space,
            &scalar.ordinary_body().space
        ));
        assert!(Arc::ptr_eq(
            &output.ordinary_body().data,
            &scalar.ordinary_body().data
        ));
    }

    #[test]
    fn nonidentity_braid_keeps_fermionic_odd_swap_sign() {
        // What: the identity shortcut does not absorb a real crossing of two
        // odd fZ2 legs, whose reduced data acquires the fermionic minus sign.
        let rt = Runtime::builder().build().unwrap();
        let odd = Space::fz2([(1, 1)]);
        let source =
            Tensor::from_block_fn(&rt, [&odd, &odd], std::iter::empty::<&Space>(), |_, _| 1.0)
                .unwrap();

        let output = source.braid(&[1, 0], &[], &[0, 1]).unwrap();

        assert!(output.data().iter().all(|&value| value == -1.0));
        assert!(!Arc::ptr_eq(
            &output.ordinary_body().data,
            &source.ordinary_body().data
        ));
    }

    #[test]
    fn repartition_matches_tensorkit_for_fermion_odd_sectors() {
        // What: a planar boundary move preserves TensorKit's fZ2 odd-sector
        // signs from semantic oracle section 4, `fZ2 2|2 -> 3|1`.
        let rt = Runtime::builder().build().unwrap();
        let a = Space::fz2([(0, 1), (1, 1)]);
        let b = Space::fz2([(0, 2), (1, 1)]);
        let c = Space::fz2([(0, 1), (1, 2)]);
        let d = Space::fz2([(0, 2), (1, 2)]);
        let source = sequential_f64_tensor(&rt, &[&a, &b], &[&c, &d]);

        let output = source.repartition(3).unwrap();

        assert_external_axis_order(&output, &source, &[0, 1, 3, 2]);
        assert_tensorkit_fixture(
            output.data(),
            &[
                1.0, 2.0, 4.0, 5.0, 3.0, 6.0, 31.0, 32.0, 34.0, 35.0, 33.0, 36.0, 19.0, 20.0, 25.0,
                26.0, 21.0, 27.0, 7.0, 8.0, 13.0, 14.0, 9.0, 15.0, 22.0, 23.0, 28.0, 29.0, 24.0,
                30.0, 10.0, 11.0, 16.0, 17.0, 12.0, 18.0,
            ],
        );
    }

    #[test]
    fn repartition_matches_tensorkit_for_su2_recoupling() {
        // What: SU2 repartition with nontrivial inner lines reproduces the
        // TensorKit F-move coefficients from semantic oracle section 4,
        // `SU2 2|2 -> 3|1`.
        let rt = Runtime::builder().build().unwrap();
        let a = Space::su2([(0, 1), (1, 1)]);
        let b = Space::su2([(0, 1), (1, 1)]);
        let c = Space::su2([(0, 1), (1, 1)]);
        let d = Space::su2([(0, 1), (1, 2)]);
        let source = sequential_f64_tensor(&rt, &[&a, &b], &[&c, &d]);

        let output = source.repartition(3).unwrap();

        assert_external_axis_order(&output, &source, &[0, 1, 3, 2]);
        assert_tensorkit_fixture(
            output.data(),
            &[
                1.0,
                2.0,
                12.727_922_061_357_859,
                15.556_349_186_104_049,
                14.142_135_623_730_955,
                16.970_562_748_477_143,
                7.000_000_000_000_002,
                8.000_000_000_000_002,
                -2.121_320_343_559_643,
                -3.535_533_905_932_737_8,
                -2.828_427_124_746_190_3,
                -4.242_640_687_119_286,
                15.921_683_328_090_658,
                17.146_428_199_482_248,
            ],
        );
        assert!(output
            .ordinary_body()
            .space
            .structure()
            .sector_structure()
            .blocks()
            .iter()
            .any(|block| matches!(block.key(), BlockKey::FusionTree(key) if !key.codomain_innerlines().is_empty())));
    }

    #[test]
    fn repartition_matches_tensorkit_for_nested_fz2_u1_su2() {
        // What: nested product coefficients retain both odd parity and SU2
        // recoupling semantics from semantic oracle section 4,
        // `fZ2xU1xSU2 2|2 -> 3|1`.
        let rt = Runtime::builder().build().unwrap();
        let base = [((0, 0, 0), 1), ((1, 0, 1), 1)];
        let a = Space::fz2_u1_su2(base).unwrap();
        let b = Space::fz2_u1_su2(base).unwrap();
        let c = Space::fz2_u1_su2(base).unwrap();
        let d = Space::fz2_u1_su2([((0, 0, 0), 1), ((1, 0, 1), 2)]).unwrap();
        let source = nested_semantic_sequence_tensor(&rt, &[&a, &b], &[&c, &d]);

        let output = source.repartition(3).unwrap();

        assert_external_axis_order(&output, &source, &[0, 1, 3, 2]);
        assert_nested_semantic_fixture(
            &output,
            &nested_repartition_3_order(),
            &[
                1.0,
                2.0,
                15.556_349_186_104_049,
                18.384_776_310_850_24,
                16.970_562_748_477_143,
                19.798_989_873_223_334,
                9.000_000_000_000_002,
                10.000_000_000_000_002,
                -2.121_320_343_559_643,
                -3.535_533_905_932_737_8,
                -2.828_427_124_746_190_3,
                -4.242_640_687_119_286,
                8.573_214_099_741_124,
                9.797_958_971_132_713,
            ],
        );
    }

    #[test]
    fn threaded_owned_transform_fallback_preserves_nonabelian_results() {
        // What: configuring threaded recoupling leaves the new serial-only
        // owned writer and reproduces the initialized SU2/product path for
        // both real and complex storage.
        let serial = Runtime::builder().build().unwrap();
        let threaded = Runtime::builder().recoupling_threads(2).build().unwrap();

        let su2 = Space::su2([(0, 1), (1, 2)]);
        let serial_su2 =
            Tensor::rand_with_seed(&serial, Dtype::C64, [&su2, &su2], [&su2, &su2], 226)
                .unwrap()
                .repartition(3)
                .unwrap();
        let threaded_su2 =
            Tensor::rand_with_seed(&threaded, Dtype::C64, [&su2, &su2], [&su2, &su2], 226)
                .unwrap()
                .repartition(3)
                .unwrap();
        assert_eq!(threaded_su2.data_c64(), serial_su2.data_c64());

        let product = Space::fz2_u1_su2([((0, 0, 0), 1), ((1, 0, 1), 2)]).unwrap();
        let serial_product = Tensor::rand_with_seed(
            &serial,
            Dtype::F64,
            [&product, &product],
            [&product, &product],
            227,
        )
        .unwrap()
        .repartition(1)
        .unwrap();
        let threaded_product = Tensor::rand_with_seed(
            &threaded,
            Dtype::F64,
            [&product, &product],
            [&product, &product],
            227,
        )
        .unwrap()
        .repartition(1)
        .unwrap();
        assert_eq!(threaded_product.data(), serial_product.data());
    }

    #[test]
    fn repartition_decreasing_boundary_matches_tensorkit_for_fermion_odd_sectors() {
        // What: moving the boundary in the opposite direction preserves the
        // fZ2 oracle signs from section 4, `fZ2 2|2 -> 1|3`.
        let rt = Runtime::builder().build().unwrap();
        let a = Space::fz2([(0, 1), (1, 1)]);
        let b = Space::fz2([(0, 2), (1, 1)]);
        let c = Space::fz2([(0, 1), (1, 2)]);
        let d = Space::fz2([(0, 2), (1, 2)]);
        let source = sequential_f64_tensor(&rt, &[&a, &b], &[&c, &d]);

        let output = source.repartition(1).unwrap();

        assert_eq!((output.codomain_rank(), output.domain_rank()), (1, 3));
        assert_external_axis_order(&output, &source, &[0, 2, 3, 1]);
        assert_tensorkit_fixture(
            output.data(),
            &[
                1.0, 4.0, 2.0, 5.0, 7.0, 10.0, 13.0, 16.0, 8.0, 11.0, 14.0, 17.0, 21.0, 24.0, 27.0,
                30.0, 33.0, 36.0, 19.0, 22.0, 25.0, 28.0, 20.0, 23.0, 26.0, 29.0, 31.0, 34.0, 32.0,
                35.0, 3.0, 6.0, 9.0, 12.0, 15.0, 18.0,
            ],
        );
    }

    #[test]
    fn repartition_decreasing_boundary_matches_tensorkit_for_nested_product() {
        // What: decreasing the boundary retains the nested product's odd
        // parity and SU2 coefficients from semantic oracle section 4,
        // `fZ2xU1xSU2 2|2 -> 1|3`.
        let rt = Runtime::builder().build().unwrap();
        let base = [((0, 0, 0), 1), ((1, 0, 1), 1)];
        let a = Space::fz2_u1_su2(base).unwrap();
        let b = Space::fz2_u1_su2(base).unwrap();
        let c = Space::fz2_u1_su2(base).unwrap();
        let d = Space::fz2_u1_su2([((0, 0, 0), 1), ((1, 0, 1), 2)]).unwrap();
        let source = nested_semantic_sequence_tensor(&rt, &[&a, &b], &[&c, &d]);

        let output = source.repartition(1).unwrap();

        assert_eq!((output.codomain_rank(), output.domain_rank()), (1, 3));
        assert_external_axis_order(&output, &source, &[0, 2, 3, 1]);
        assert_nested_semantic_fixture(
            &output,
            &nested_repartition_1_order(),
            &[
                1.0,
                3.0,
                5.0,
                14.142_135_623_730_95,
                16.970_562_748_477_14,
                19.798_989_873_223_33,
                9.000_000_000_000_002,
                11.0,
                13.0,
                -std::f64::consts::SQRT_2,
                -2.0 * std::f64::consts::SQRT_2,
                -3.0 * std::f64::consts::SQRT_2,
                8.573_214_099_741_124,
                9.797_958_971_132_713,
            ],
        );
    }

    #[test]
    fn repartition_supports_empty_codomain_empty_domain_and_rank_zero() {
        // What: N=0 and N=rank match the nested-product endpoint fixtures in
        // semantic oracle section 4, while rank zero remains a shared no-op.
        let rt = Runtime::builder().build().unwrap();
        let a = Space::u1([(0, 1)]);
        let b = Space::u1([(0, 2)]);
        let c = Space::u1([(0, 3)]);
        let d = Space::u1([(0, 4)]);
        let source = sequential_f64_tensor(&rt, &[&a, &b], &[&c, &d]);

        let all_domain = source.repartition(0).unwrap();
        assert_eq!(
            (all_domain.codomain_rank(), all_domain.domain_rank()),
            (0, 4)
        );
        assert_external_axis_order(&all_domain, &source, &[2, 3, 1, 0]);

        let all_codomain = source.repartition(source.rank()).unwrap();
        assert_eq!(
            (all_codomain.codomain_rank(), all_codomain.domain_rank()),
            (4, 0)
        );
        assert_external_axis_order(&all_codomain, &source, &[0, 1, 3, 2]);

        let base = [((0, 0, 0), 1), ((1, 0, 1), 1)];
        let na = Space::fz2_u1_su2(base).unwrap();
        let nb = Space::fz2_u1_su2(base).unwrap();
        let nc = Space::fz2_u1_su2(base).unwrap();
        let nd = Space::fz2_u1_su2([((0, 0, 0), 1), ((1, 0, 1), 2)]).unwrap();
        let nested = nested_semantic_sequence_tensor(&rt, &[&na, &nb], &[&nc, &nd]);

        let nested_all_domain = nested.repartition(0).unwrap();
        assert_external_axis_order(&nested_all_domain, &nested, &[2, 3, 1, 0]);
        assert_nested_semantic_fixture(
            &nested_all_domain,
            &nested_repartition_0_order(),
            &[
                1.0,
                3.0,
                5.0,
                14.142_135_623_730_95,
                16.970_562_748_477_14,
                19.798_989_873_223_33,
                12.727_922_061_357_86,
                15.556_349_186_104_05,
                18.384_776_310_850_24,
                -2.0,
                -4.000_000_000_000_001,
                -6.000_000_000_000_002,
                12.124_355_652_982_14,
                13.856_406_460_551_02,
            ],
        );

        let nested_all_codomain = nested.repartition(nested.rank()).unwrap();
        assert_external_axis_order(&nested_all_codomain, &nested, &[0, 1, 3, 2]);
        assert_nested_semantic_fixture(
            &nested_all_codomain,
            &nested_repartition_4_order(),
            &[
                1.0,
                2.0,
                15.556_349_186_104_05,
                18.384_776_310_850_24,
                16.970_562_748_477_14,
                19.798_989_873_223_33,
                12.727_922_061_357_86,
                14.142_135_623_730_96,
                -3.000_000_000_000_001,
                -5.000_000_000_000_001,
                -4.000_000_000_000_001,
                -6.000_000_000_000_002,
                12.124_355_652_982_14,
                13.856_406_460_551_02,
            ],
        );

        let vector =
            Tensor::rand_with_seed(&rt, Dtype::F64, [&a], std::iter::empty::<&Space>(), 192)
                .unwrap();
        let scalar = vector
            .contract(&vector.adjoint().unwrap(), &[0], &[0])
            .unwrap();
        let repartitioned_scalar = scalar.repartition(0).unwrap();
        assert_eq!(repartitioned_scalar.rank(), 0);
        assert!(Arc::ptr_eq(
            &repartitioned_scalar.ordinary_body().space,
            &scalar.ordinary_body().space
        ));
        assert!(Arc::ptr_eq(
            &repartitioned_scalar.ordinary_body().data,
            &scalar.ordinary_body().data
        ));
    }

    #[test]
    fn zeros_like_is_a_same_shape_zero() {
        // What: zeros_like keeps spaces/dtype and zeroes every entry.
        let rt = Runtime::builder().build().unwrap();
        let v = Space::u1([(0, 2), (1, 1)]);
        let t = Tensor::rand(&rt, Dtype::C64, [&v], [&v]).unwrap();
        let z = t.zeros_like().unwrap();
        assert_eq!(z.dtype(), Dtype::C64);
        assert_eq!(z.codomain_spaces(), t.codomain_spaces());
        assert_eq!(z.norm().unwrap(), 0.0);
    }

    #[test]
    fn identity_is_hermitian_isometric_unitary_posdef() {
        // What: the identity endomorphism satisfies every structural predicate.
        let rt = Runtime::builder().build().unwrap();
        let v = Space::u1([(0, 2), (1, 1)]);
        let id = Tensor::id(&rt, Dtype::F64, [&v, &v]).unwrap();
        assert!(id.is_hermitian(1e-12).unwrap());
        assert!(id.is_isometric(1e-12).unwrap());
        assert!(id.is_unitary(1e-12).unwrap());
        assert!(id.is_posdef(1e-12).unwrap());
    }

    #[test]
    fn non_endomorphism_is_not_hermitian() {
        // What: a rectangular map returns false rather than erroring.
        let rt = Runtime::builder().build().unwrap();
        let v = Space::u1([(0, 2), (1, 1)]);
        let w = Space::u1([(0, 3), (1, 2)]);
        let t = Tensor::rand(&rt, Dtype::F64, [&v], [&w]).unwrap();
        assert!(!t.is_hermitian(1e-12).unwrap());
    }

    #[test]
    fn negative_identity_is_hermitian_but_not_posdef() {
        // What: is_posdef rejects a Hermitian tensor with a negative eigenvalue.
        let rt = Runtime::builder().build().unwrap();
        let v = Space::u1([(0, 2), (1, 1)]);
        let minus_id = Tensor::id(&rt, Dtype::F64, [&v])
            .unwrap()
            .scale(-1.0)
            .unwrap();
        assert!(minus_id.is_hermitian(1e-12).unwrap());
        assert!(!minus_id.is_posdef(1e-12).unwrap());
    }

    #[test]
    fn zero_tensor_is_not_posdef() {
        // What: a zero spectrum is positive SEMIdefinite, so strict posdef
        // (TK isposdef = Cholesky) must reject it.
        let rt = Runtime::builder().build().unwrap();
        let v = Space::u1([(0, 2), (1, 1)]);
        let zero = Tensor::zeros(&rt, Dtype::F64, [&v], [&v]).unwrap();
        assert!(zero.is_hermitian(1e-12).unwrap());
        assert!(!zero.is_posdef(1e-12).unwrap());
    }

    #[test]
    fn hermitian_projectors_split_a_general_endomorphism() {
        // What: t = project_hermitian(t) + project_antihermitian(t), and each
        // part satisfies its predicate.
        let rt = Runtime::builder().build().unwrap();
        let v = Space::u1([(0, 2), (1, 1)]);
        let t = Tensor::rand(&rt, Dtype::C64, [&v], [&v]).unwrap();
        let herm = t.project_hermitian().unwrap();
        let anti = t.project_antihermitian().unwrap();
        assert!(herm.is_hermitian(1e-10).unwrap());
        assert!(anti.is_antihermitian(1e-10).unwrap());
        // Reassembled parts recover the original tensor.
        let recomposed = herm.add(&anti, 1.0, 1.0).unwrap();
        assert!(recomposed.add(&t, 1.0, -1.0).unwrap().norm().unwrap() < 1e-10);
    }
}

#[cfg(test)]
mod ordered_contract_route_tests {
    use super::*;

    #[test]
    fn contracted_axis_validation_handles_high_rank_and_late_duplicates() {
        let valid = (0..64).rev().collect::<Vec<_>>();
        validate_contracted_axes(&valid, 64).unwrap();
        let mut duplicate = valid;
        duplicate[63] = duplicate[0];
        // What: validation remains correct after the inline common-rank mark
        // storage spills to its linear-time high-rank fallback.
        assert!(validate_contracted_axes(&duplicate, 64).is_err());
        assert!(validate_contracted_axes(&[64], 64).is_err());
    }

    #[test]
    fn ordinary_multiplicity_free_ordered_contract_uses_fused_plan_route() {
        // What: a crossed SU2 pAB is handed to the contraction plan instead of
        // returning a default-order owned tensor to a second public permute.
        let runtime = Runtime::builder().build().unwrap();
        let space = Space::su2([(0, 2), (1, 2), (2, 1)]);
        let lhs = Tensor::rand_with_seed(
            &runtime,
            Dtype::F64,
            [&space, &space],
            [&space, &space],
            224_501,
        )
        .unwrap();
        let rhs = Tensor::rand_with_seed(
            &runtime,
            Dtype::F64,
            [&space, &space],
            [&space, &space],
            224_502,
        )
        .unwrap();

        ORDERED_CONTRACT_FUSED_ROUTE.with(|observation| observation.set(Some(false)));
        let _ = lhs
            .contract_ordered(&rhs, &[3, 2], &[0, 1], &[2, 0, 3, 1])
            .unwrap();
        let observed = ORDERED_CONTRACT_FUSED_ROUTE.with(|observation| observation.replace(None));

        assert_eq!(observed, Some(true));
    }

    #[test]
    fn compact_diagonal_ordered_contract_keeps_sequential_fallback() {
        // What: compact diagonal complexity dispatch is not bypassed by the
        // new host fusion route.
        let runtime = Runtime::builder().build().unwrap();
        let space = Space::u1([(0, 2), (1, 2)]);
        let source =
            Tensor::rand_with_seed(&runtime, Dtype::F64, [&space], [&space], 224_503).unwrap();
        let diagonal = source.svd_compact().unwrap().1;

        ORDERED_CONTRACT_FUSED_ROUTE.with(|observation| observation.set(Some(false)));
        let _ = diagonal
            .contract_ordered(&diagonal, &[1], &[0], &[1, 0])
            .unwrap();
        let observed = ORDERED_CONTRACT_FUSED_ROUTE.with(|observation| observation.replace(None));

        assert_eq!(observed, Some(false));
    }

    #[test]
    fn generic_fusion_ordered_contract_keeps_sequential_fallback() {
        // What: outer-multiplicity-capable generic fusion remains on its
        // separately proved contract and permute implementations.
        let runtime = Runtime::builder().build().unwrap();
        let space = Space::su3([((1, 0), 1), ((0, 1), 1)]).unwrap();
        let lhs =
            Tensor::rand_with_seed(&runtime, Dtype::F64, [&space], [&space], 224_504).unwrap();
        let rhs =
            Tensor::rand_with_seed(&runtime, Dtype::F64, [&space], [&space], 224_505).unwrap();

        ORDERED_CONTRACT_FUSED_ROUTE.with(|observation| observation.set(Some(false)));
        let actual = lhs.contract_ordered(&rhs, &[1], &[0], &[1, 0]).unwrap();
        let observed = ORDERED_CONTRACT_FUSED_ROUTE.with(|observation| observation.replace(None));
        let expected = lhs
            .contract(&rhs, &[1], &[0])
            .unwrap()
            .permute(&[1], &[0])
            .unwrap();

        assert_eq!(observed, Some(false));
        assert_eq!(actual.data().len(), expected.data().len());
        for (&actual, &expected) in actual.data().iter().zip(expected.data()) {
            assert!((actual - expected).abs() < 1.0e-11);
        }
    }

    #[test]
    fn partial_trace_builds_selected_result_layout_once() {
        // What: nested-product partial trace enters the selected-result layout
        // builder once and returns the expected rank-zero tensor.
        let runtime = Runtime::builder().build().unwrap();
        let space = Space::fz2_u1_su2([((0, 0, 0), 2), ((1, -1, 1), 1), ((1, 1, 1), 1)]).unwrap();
        let tensor =
            Tensor::rand_with_seed(&runtime, Dtype::F64, [&space], [&space], 224_506).unwrap();

        SELECTED_RESULT_LAYOUT_BUILDS.with(|observation| observation.set(Some(0)));
        let traced = tensor.trace_pairs(&[(0, 1)]).unwrap();
        let builds = SELECTED_RESULT_LAYOUT_BUILDS.with(|observation| observation.replace(None));

        assert_eq!(builds, Some(1));
        assert_eq!(traced.rank(), 0);
    }
}
