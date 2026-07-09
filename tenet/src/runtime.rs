//! User-layer runtime: owns the per-rule execution/cache state so everyday
//! tensor code never passes explicit contexts around.

use std::any::Any;
use std::hash::Hash;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};

use num_complex::Complex64;
use tenet_tensors::{
    DenseTreeTransformOperations, TensorContractFusionExecutionContext,
    TreeTransformBuiltinRuleCacheKey, TreeTransformProductRuleCacheKey,
};

use crate::error::Error;
use crate::plancache::{Optimizer, PlanCacheConfig};

pub type Ctx<D, Key> = TensorContractFusionExecutionContext<D, Key>;
pub(crate) type BuiltinKey = TreeTransformBuiltinRuleCacheKey;
pub(crate) type ProductKey = TreeTransformProductRuleCacheKey<BuiltinKey, BuiltinKey>;
/// Cache key of the left-associated triple product `(fZ2 ⊠ U1) ⊠ SU2`.
pub(crate) type TripleKey = TreeTransformProductRuleCacheKey<ProductKey, BuiltinKey>;

/// The pair of per-scalar execution contexts for one fusion rule: tensor
/// operations dispatch on the stored dtype once per call and pick one side.
pub struct Ctxs<Key: Clone + Eq + Hash + Send + Sync + 'static> {
    pub(crate) f64: Ctx<f64, Key>,
    pub(crate) c64: Ctx<Complex64, Key>,
}

impl<Key: Clone + Eq + Hash + Send + Sync + 'static> Default for Ctxs<Key> {
    fn default() -> Self {
        Self {
            f64: Ctx::default(),
            c64: Ctx::default(),
        }
    }
}

/// Builds one contraction/recoupling backend for the requested thread count and
/// CPU GEMM provider. Both `None` yields the faer default; a `gemm_kind` of
/// `Blas` fails if no `cpu-blas`/`blas-*` provider was compiled in.
fn make_transform_ops(
    threads: Option<usize>,
    gemm_kind: Option<tenet_dense::CpuBackendKind>,
) -> Result<DenseTreeTransformOperations, Error> {
    let ops = match (threads, gemm_kind) {
        (Some(threads), Some(kind)) => {
            DenseTreeTransformOperations::with_threads_and_kind(threads, kind)
        }
        (None, Some(kind)) => DenseTreeTransformOperations::with_kind(kind),
        (Some(threads), None) => DenseTreeTransformOperations::with_threads(threads),
        (None, None) => Ok(DenseTreeTransformOperations::default_executor()),
    };
    Ok(ops?)
}

impl<Key: Clone + Eq + Hash + Send + Sync + 'static> Ctxs<Key> {
    /// Builds the per-scalar contexts with an explicit thread count and/or CPU
    /// GEMM provider for the contraction/recoupling backend. Passing `(None,
    /// None)` reproduces [`Ctxs::default`] but through the same seam.
    fn with_config(
        threads: Option<usize>,
        gemm_kind: Option<tenet_dense::CpuBackendKind>,
    ) -> Result<Self, Error> {
        Ok(Self {
            f64:
                Ctx::with_parts(
                    tenet_tensors::TreeTransformExecutionContext::new(make_transform_ops(
                        threads, gemm_kind,
                    )?),
                    make_transform_ops(threads, gemm_kind)?,
                    <DenseTreeTransformOperations as tenet_tensors::TensorContractBackend<
                        f64,
                        f64,
                    >>::Workspace::default(),
                    tenet_tensors::TensorContractCache::new(),
                ),
            c64: Ctx::with_parts(
                tenet_tensors::TreeTransformExecutionContext::new(make_transform_ops(
                    threads, gemm_kind,
                )?),
                make_transform_ops(threads, gemm_kind)?,
                <DenseTreeTransformOperations as tenet_tensors::TensorContractBackend<
                    Complex64,
                    f64,
                >>::Workspace::default(),
                tenet_tensors::TensorContractCache::new(),
            ),
        })
    }
}

/// Per-rule expert-layer execution contexts (contraction resolution caches,
/// tree-transform replay caches, dense backends and workspaces).
///
/// One field per supported rule; each context is created eagerly because the
/// empty contexts are cheap, and filled lazily by use.
pub(crate) struct RuntimeState {
    pub(crate) u1: Ctxs<BuiltinKey>,
    pub(crate) z2: Ctxs<BuiltinKey>,
    pub(crate) fz2: Ctxs<BuiltinKey>,
    pub(crate) su2: Ctxs<BuiltinKey>,
    pub(crate) u1_fz2: Ctxs<ProductKey>,
    pub(crate) fz2_u1_su2: Ctxs<TripleKey>,
    /// Rule-independent dense-factorization executor (SVD / QR / eigh on the
    /// coupled-sector matrices), shared by all decomposition methods. Boxed
    /// behind [`tenet_dense::DenseExecutor`] so the CPU linear-algebra backend
    /// is selectable at [`RuntimeBuilder::with_dense_executor`]; the default is
    /// the faer-backed [`tenet_dense::DefaultDenseExecutor`].
    pub(crate) dense: Box<dyn tenet_dense::DenseExecutor + Send>,
    /// CUDA device context when the runtime was built with
    /// [`RuntimeBuilder::cuda`]; `None` on CPU-only runtimes.
    #[cfg(feature = "cuda")]
    pub(crate) cuda: Option<tenet_dense::CudaDenseContext>,
    /// Contraction-plan cache configuration (the cache state itself lives
    /// in `extension_slot`).
    pub(crate) plan_cache_config: PlanCacheConfig,
    /// Type-erased downstream extension slot. Currently holds the
    /// contraction-plan cache: the cache and plan types live in
    /// `tenet-network`, which depends on this crate, so the runtime can only
    /// hold them behind `dyn Any`; `tenet-network` claims and downcasts the
    /// slot on first use.
    pub(crate) extension_slot: Option<Box<dyn Any + Send>>,
}

impl RuntimeState {
    fn new(dense: Box<dyn tenet_dense::DenseExecutor + Send>) -> Self {
        Self {
            u1: Ctxs::default(),
            z2: Ctxs::default(),
            fz2: Ctxs::default(),
            su2: Ctxs::default(),
            u1_fz2: Ctxs::default(),
            fz2_u1_su2: Ctxs::default(),
            dense,
            #[cfg(feature = "cuda")]
            cuda: None,
            plan_cache_config: PlanCacheConfig::default(),
            extension_slot: None,
        }
    }

    fn with_config(
        dense: Box<dyn tenet_dense::DenseExecutor + Send>,
        threads: Option<usize>,
        gemm_kind: Option<tenet_dense::CpuBackendKind>,
    ) -> Result<Self, Error> {
        Ok(Self {
            u1: Ctxs::with_config(threads, gemm_kind)?,
            z2: Ctxs::with_config(threads, gemm_kind)?,
            fz2: Ctxs::with_config(threads, gemm_kind)?,
            su2: Ctxs::with_config(threads, gemm_kind)?,
            u1_fz2: Ctxs::with_config(threads, gemm_kind)?,
            fz2_u1_su2: Ctxs::with_config(threads, gemm_kind)?,
            dense,
            #[cfg(feature = "cuda")]
            cuda: None,
            plan_cache_config: PlanCacheConfig::default(),
            extension_slot: None,
        })
    }

    /// Applies the tree-transform replay worker count to every per-rule
    /// execution context (parallelism is a property of the backend, so the
    /// setting lives on each context's transform backend).
    fn set_recoupling_threads(&mut self, threads: usize) {
        fn apply<Key: Clone + Eq + Hash + Send + Sync + 'static>(
            ctxs: &mut Ctxs<Key>,
            threads: usize,
        ) {
            ctxs.f64
                .tree_context_mut()
                .backend_mut()
                .set_recoupling_threads(threads);
            ctxs.c64
                .tree_context_mut()
                .backend_mut()
                .set_recoupling_threads(threads);
        }
        apply(&mut self.u1, threads);
        apply(&mut self.z2, threads);
        apply(&mut self.fz2, threads);
        apply(&mut self.su2, threads);
        apply(&mut self.u1_fz2, threads);
        apply(&mut self.fz2_u1_su2, threads);
    }
}

impl Default for RuntimeState {
    fn default() -> Self {
        Self::new(Box::new(tenet_dense::DefaultDenseExecutor::default()))
    }
}

/// Dispatches on a [`crate::space::RuleKind`], binding `$rule` to the
/// concrete fusion rule and `$ctx` to the matching per-scalar execution
/// context pair ([`Ctxs`]) of a [`RuntimeState`].
macro_rules! with_rule_ctx {
    ($kind:expr, $state:expr, $rule:ident, $ctx:ident, $body:expr) => {
        match $kind {
            $crate::space::RuleKind::U1 => {
                let $rule = &tenet_core::U1FusionRule;
                let $ctx = &mut $state.u1;
                $body
            }
            $crate::space::RuleKind::Z2 => {
                let $rule = &tenet_core::Z2FusionRule;
                let $ctx = &mut $state.z2;
                $body
            }
            $crate::space::RuleKind::FZ2 => {
                let $rule = &tenet_core::FermionParityFusionRule;
                let $ctx = &mut $state.fz2;
                $body
            }
            $crate::space::RuleKind::SU2 => {
                let $rule = &tenet_core::SU2FusionRule;
                let $ctx = &mut $state.su2;
                $body
            }
            $crate::space::RuleKind::U1FZ2 => {
                let $rule = &tenet_core::ProductFusionRule::<
                    tenet_core::U1FusionRule,
                    tenet_core::FermionParityFusionRule,
                >::new(
                    tenet_core::U1FusionRule,
                    tenet_core::FermionParityFusionRule,
                );
                let $ctx = &mut $state.u1_fz2;
                $body
            }
            $crate::space::RuleKind::FZ2U1SU2 => {
                let $rule = &tenet_core::ProductFusionRule::<
                    tenet_core::ProductFusionRule<
                        tenet_core::FermionParityFusionRule,
                        tenet_core::U1FusionRule,
                    >,
                    tenet_core::SU2FusionRule,
                >::new(
                    tenet_core::ProductFusionRule::new(
                        tenet_core::FermionParityFusionRule,
                        tenet_core::U1FusionRule,
                    ),
                    tenet_core::SU2FusionRule,
                );
                let $ctx = &mut $state.fz2_u1_su2;
                $body
            }
        }
    };
}
pub(crate) use with_rule_ctx;

struct RuntimeInner {
    state: Mutex<RuntimeState>,
    rand_counter: AtomicU64,
}

/// Execution runtime for the user-layer [`crate::prelude::Tensor`] API.
///
/// A `Runtime` is built once via [`Runtime::builder`] and then carried
/// implicitly by every tensor created from it; operations reuse the
/// runtime's expert-layer caches (contraction plans, tree-transform replays,
/// dense workspaces) without explicit context arguments. Cloning a `Runtime`
/// clones a shared handle, not the state.
///
/// Concurrency: the internal state sits behind one coarse `Mutex`, locked
/// once per tensor operation. The user layer is designed for
/// single-threaded driving code (backend parallelism lives below), so the
/// lock is uncontended in practice.
///
/// # Examples
///
/// ```
/// use tenet::prelude::*;
///
/// let rt = Runtime::builder().build()?;
/// let v = Space::z2([(0, 1), (1, 1)]);
/// let a = Tensor::zeros(&rt, Dtype::F64, [&v], [&v])?;
/// assert_eq!(a.norm()?, 0.0);
/// # Ok::<(), tenet::prelude::Error>(())
/// ```
#[derive(Clone)]
pub struct Runtime {
    inner: Arc<RuntimeInner>,
}

impl Runtime {
    /// Starts building a runtime. The default runtime uses the CPU backend;
    /// feature-gated device options such as CUDA are attached through the
    /// builder so tensor construction keeps the same shape.
    pub fn builder() -> RuntimeBuilder {
        RuntimeBuilder::default()
    }

    pub(crate) fn lock(&self) -> MutexGuard<'_, RuntimeState> {
        // ponytail: poisoning treated as fatal; no operation leaves the
        // caches half-written in a way worth recovering from.
        self.inner
            .state
            .lock()
            .expect("tenet runtime state poisoned")
    }

    pub(crate) fn same_runtime(&self, other: &Runtime) -> bool {
        Arc::ptr_eq(&self.inner, &other.inner)
    }

    /// Snapshot of this runtime's contraction-plan-cache configuration.
    pub fn plan_cache_config(&self) -> PlanCacheConfig {
        self.lock().plan_cache_config.clone()
    }

    /// Replaces the contraction-plan-cache configuration.
    pub fn set_plan_cache_config(&self, config: PlanCacheConfig) {
        self.lock().plan_cache_config = config;
    }

    /// Locked access to the type-erased downstream extension slot
    /// (currently the contraction-plan cache: the cache type lives in
    /// `tenet-network`, which claims and downcasts the slot on first use).
    /// Expert seam for `tenet-network`; do not hold tensors' operations
    /// inside `f` (the runtime state mutex is held for its duration).
    #[doc(hidden)]
    pub fn with_extension_slot<R>(
        &self,
        f: impl FnOnce(&mut Option<Box<dyn Any + Send>>) -> R,
    ) -> R {
        f(&mut self.lock().extension_slot)
    }

    /// Deterministic per-runtime stream position for [`crate::prelude::Tensor::rand`].
    pub(crate) fn next_rand_seed(&self) -> u64 {
        // Fixed base seed: runs are reproducible, consecutive `rand` calls
        // still produce distinct tensors.
        0x9E37_79B9_7F4A_7C15 ^ self.inner.rand_counter.fetch_add(1, Ordering::Relaxed)
    }
}

impl std::fmt::Debug for Runtime {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Runtime").finish_non_exhaustive()
    }
}

/// Selects the CPU linear-algebra provider for dense per-coupled-sector
/// factorizations (SVD / QR / eigh / GEMM), chosen via
/// [`RuntimeBuilder::linalg_backend`]. Backend choice changes performance
/// only — results stay TensorKit-equivalent across providers.
///
/// The *specific* BLAS/LAPACK behind [`LinalgBackend::Blas`] (OpenBLAS, MKL,
/// Accelerate, or an injected provider) is a compile-time choice via the
/// `blas-*` cargo features; at runtime you only pick faer vs the linked BLAS.
/// Selecting `Blas` when no `cpu-blas`/`blas-*` provider was compiled in fails
/// at [`RuntimeBuilder::build`].
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum LinalgBackend {
    /// Pure-Rust faer provider (the default; always available).
    Faer,
    /// System BLAS/LAPACK linked through a `blas-*` cargo feature.
    Blas,
}

impl LinalgBackend {
    fn to_kind(self) -> tenet_dense::CpuBackendKind {
        match self {
            LinalgBackend::Faer => tenet_dense::CpuBackendKind::Faer,
            LinalgBackend::Blas => tenet_dense::CpuBackendKind::Blas,
        }
    }
}

/// Builder for [`Runtime`]; see [`Runtime::builder`].
///
/// Not `Clone`/`Debug`-derivable: an injected dense executor
/// ([`Self::with_dense_executor`]) is a `Box<dyn DenseExecutor>`, which is
/// neither cloneable nor `Debug`. A manual [`std::fmt::Debug`] is provided that
/// reports the executor's presence without touching it.
#[derive(Default)]
pub struct RuntimeBuilder {
    #[cfg(feature = "cuda")]
    cuda_device: Option<usize>,
    plan_cache: PlanCacheConfig,
    dense_threads: Option<usize>,
    recoupling_threads: Option<usize>,
    /// User-injected CPU linear-algebra backend; `None` uses the faer default.
    dense_executor: Option<Box<dyn tenet_dense::DenseExecutor + Send>>,
    /// Selected built-in CPU provider for dense factorizations (SVD/QR/eigh);
    /// `None` uses faer. Ignored when [`Self::dense_executor`] is set.
    linalg_backend: Option<LinalgBackend>,
    /// Selected built-in CPU provider for the contraction/recoupling GEMM;
    /// `None` uses faer. Independent of [`Self::linalg_backend`].
    gemm_backend: Option<LinalgBackend>,
}

impl std::fmt::Debug for RuntimeBuilder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut s = f.debug_struct("RuntimeBuilder");
        #[cfg(feature = "cuda")]
        s.field("cuda_device", &self.cuda_device);
        s.field("plan_cache", &self.plan_cache)
            .field("dense_threads", &self.dense_threads)
            .field("recoupling_threads", &self.recoupling_threads)
            .field("dense_executor", &self.dense_executor.is_some())
            .field("linalg_backend", &self.linalg_backend)
            .field("gemm_backend", &self.gemm_backend)
            .finish()
    }
}

impl RuntimeBuilder {
    /// Attaches a CUDA device (by ordinal) to the runtime. Tensors stay on
    /// the host until moved explicitly with
    /// [`crate::prelude::Tensor::to_cuda`]; there are no implicit
    /// transfers. Device initialization happens in [`Self::build`].
    #[cfg(feature = "cuda")]
    pub fn cuda(mut self, device: usize) -> Self {
        self.cuda_device = Some(device);
        self
    }

    /// Sets the contraction-plan-cache configuration (capacity, replan
    /// policy, default optimizer) for this runtime.
    pub fn plan_cache(mut self, config: PlanCacheConfig) -> Self {
        self.plan_cache = config;
        self
    }

    /// Sets the default contraction-order [`Optimizer`] for network
    /// contractions (`tensor!`); shorthand for setting it on
    /// [`Self::plan_cache`]'s config.
    pub fn optimizer(mut self, optimizer: Optimizer) -> Self {
        self.plan_cache.optimizer = optimizer;
        self
    }

    /// Sets the global CPU worker count used by dense/strided kernels that
    /// run on rayon's global pool. This must be configured before any rayon
    /// work has initialized the pool; later calls are best-effort no-ops.
    ///
    /// If unset, [`Self::build`] also checks `TENET_DENSE_THREADS`. A value
    /// of 1 keeps tiny-block workloads serial while still allowing outer
    /// application-level parallelism.
    pub fn dense_threads(mut self, threads: usize) -> Self {
        self.dense_threads = Some(threads.max(1));
        self
    }

    /// Selects the CPU linear-algebra backend (SVD / QR / eigh / GEMM on the
    /// coupled-sector matrices) by injecting a [`tenet_dense::DenseExecutor`].
    /// Unset uses the faer-backed default. This is the seam for a system
    /// BLAS/LAPACK or MKL backend: implement `DenseExecutor` and pass it here —
    /// no operator or decomposition code changes.
    ///
    /// The injected executor carries its own thread configuration;
    /// [`Self::dense_threads`] then only sizes the shared rayon pool and the
    /// recoupling contexts, not the injected backend.
    pub fn with_dense_executor(
        mut self,
        executor: Box<dyn tenet_dense::DenseExecutor + Send>,
    ) -> Self {
        self.dense_executor = Some(executor);
        self
    }

    /// Selects a built-in CPU provider ([`LinalgBackend::Faer`] or
    /// [`LinalgBackend::Blas`]) for the dense **factorizations** — SVD / QR /
    /// eigh / eig / inv / exp (the LAPACK-style work). Unset uses faer. The
    /// contraction GEMM (BLAS-style work) is chosen separately with
    /// [`Self::gemm_backend`].
    ///
    /// This is the ergonomic counterpart to [`Self::with_dense_executor`] for
    /// the shipped providers; an explicitly injected executor takes precedence.
    /// Choosing [`LinalgBackend::Blas`] without a compiled `cpu-blas`/`blas-*`
    /// provider fails in [`Self::build`].
    ///
    /// # Examples
    ///
    /// ```
    /// use tenet::prelude::*;
    ///
    /// // Explicit faer provider (also the default). Every tensor created from
    /// // this runtime factorizes on the chosen backend — no per-call argument.
    /// let rt = Runtime::builder()
    ///     .linalg_backend(LinalgBackend::Faer)
    ///     .build()?;
    /// let v = Space::u1([(-1, 1), (0, 2), (1, 1)]);
    /// let t = Tensor::rand_with_seed(&rt, Dtype::F64, [&v, &v], [&v, &v], 7)?;
    /// let (_u, _s, _vh) = t.svd_compact()?;
    ///
    /// // Switch to the system BLAS/LAPACK linked via a `blas-*` cargo feature
    /// // (OpenBLAS / MKL / Accelerate). Results are identical to faer up to
    /// // floating-point rounding; only performance differs. Without a linked
    /// // provider this returns an error, so fall back to faer:
    /// let rt = Runtime::builder()
    ///     .linalg_backend(LinalgBackend::Blas)
    ///     .build()
    ///     .or_else(|_| Runtime::builder().build())?;
    /// # let _ = rt;
    /// # Ok::<(), tenet::prelude::Error>(())
    /// ```
    pub fn linalg_backend(mut self, backend: LinalgBackend) -> Self {
        self.linalg_backend = Some(backend);
        self
    }

    /// Selects a built-in CPU provider ([`LinalgBackend::Faer`] or
    /// [`LinalgBackend::Blas`]) for the coupled-block **contraction GEMM**
    /// (`compose` / `contract` and the recoupling replays — the BLAS-style
    /// work). Unset uses faer. Independent of [`Self::linalg_backend`]: the
    /// factorizations and the contraction GEMM can run on different providers
    /// (e.g. faer GEMM with BLAS/LAPACK factorizations, or the reverse).
    /// Choosing [`LinalgBackend::Blas`] without a compiled `cpu-blas`/`blas-*`
    /// provider fails in [`Self::build`].
    ///
    /// # Examples
    ///
    /// ```
    /// use tenet::prelude::*;
    ///
    /// // faer everywhere is the default; this is explicit and equivalent.
    /// let rt = Runtime::builder()
    ///     .gemm_backend(LinalgBackend::Faer)
    ///     .build()?;
    /// # let _ = rt;
    /// # Ok::<(), tenet::prelude::Error>(())
    /// ```
    pub fn gemm_backend(mut self, backend: LinalgBackend) -> Self {
        self.gemm_backend = Some(backend);
        self
    }

    /// Sets the CPU worker count for symmetry recoupling replays
    /// (permute/braid/transpose tree transforms — the cold-path cost of
    /// SU(2) workloads; **not** BLAS threads). Default is 1 (serial); values
    /// above 1 parallelize replays past the backend's size gate on the
    /// shared rayon pool.
    pub fn recoupling_threads(mut self, threads: usize) -> Self {
        self.recoupling_threads = Some(threads);
        self
    }

    /// Finishes the build; fails when a requested backend (e.g. the CUDA
    /// device) cannot be initialized.
    pub fn build(self) -> Result<Runtime, Error> {
        let dense_threads = self.dense_threads.or_else(dense_threads_from_env);
        if let Some(threads) = dense_threads {
            // rayon's global pool can be initialized only once per process.
            // Runtime construction is often repeated in tests/examples, so
            // keep this knob best-effort after the first user.
            let _ = rayon::ThreadPoolBuilder::new()
                .num_threads(threads.max(1))
                .build_global();
        }
        // Injected backend wins; otherwise build the selected provider (faer by
        // default), honoring the dense-thread count when set.
        let dense: Box<dyn tenet_dense::DenseExecutor + Send> = match self.dense_executor {
            Some(executor) => executor,
            None => {
                let kind = self.linalg_backend.map(LinalgBackend::to_kind);
                match (kind, dense_threads) {
                    (Some(kind), Some(threads)) => Box::new(
                        tenet_dense::DefaultDenseExecutor::with_threads_and_kind(threads, kind)
                            .map_err(tenet_tensors::OperationError::Dense)?,
                    ),
                    (Some(kind), None) => Box::new(
                        tenet_dense::DefaultDenseExecutor::with_kind(kind)
                            .map_err(tenet_tensors::OperationError::Dense)?,
                    ),
                    (None, Some(threads)) => Box::new(
                        tenet_dense::DefaultDenseExecutor::with_threads(threads)
                            .map_err(tenet_tensors::OperationError::Dense)?,
                    ),
                    (None, None) => Box::new(tenet_dense::DefaultDenseExecutor::default()),
                }
            }
        };
        let gemm_kind = self.gemm_backend.map(LinalgBackend::to_kind);
        let mut state = if dense_threads.is_some() || gemm_kind.is_some() {
            RuntimeState::with_config(dense, dense_threads, gemm_kind)?
        } else {
            RuntimeState::new(dense)
        };
        state.plan_cache_config = self.plan_cache;
        if let Some(threads) = self.recoupling_threads {
            state.set_recoupling_threads(threads);
        }
        #[cfg(feature = "cuda")]
        if let Some(device) = self.cuda_device {
            state.cuda = Some(
                tenet_dense::CudaDenseContext::new(device)
                    .map_err(tenet_tensors::OperationError::Dense)?,
            );
        }
        Ok(Runtime {
            inner: Arc::new(RuntimeInner {
                state: Mutex::new(state),
                rand_counter: AtomicU64::new(0),
            }),
        })
    }
}

fn dense_threads_from_env() -> Option<usize> {
    std::env::var("TENET_DENSE_THREADS")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .map(|threads| threads.max(1))
}

#[cfg(test)]
mod tests {
    use super::*;

    // All four provider/thread combinations of the contraction-backend builder
    // must construct on faer (always compiled). Guards the `with_config` /
    // `make_transform_ops` matrix, incl. the plain-default `(None, None)` arm
    // that the builder's fast path would otherwise never exercise.
    #[test]
    fn transform_ops_builds_for_every_faer_config() {
        let faer = tenet_dense::CpuBackendKind::Faer;
        assert!(make_transform_ops(None, None).is_ok());
        assert!(make_transform_ops(Some(1), None).is_ok());
        assert!(make_transform_ops(None, Some(faer)).is_ok());
        assert!(make_transform_ops(Some(1), Some(faer)).is_ok());
    }
}
