//! User-layer runtime: owns the per-rule execution/cache state so everyday
//! tensor code never passes explicit contexts around.

use std::any::Any;
use std::cell::RefCell;
use std::hash::Hash;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};

use num_complex::Complex64;
use tenet_tensors::{
    DenseTreeTransformOperations, TensorContractFusionExecutionContext,
    TreeTransformBuiltinRuleCacheKey, TreeTransformProductRuleCacheKey,
    TreeTransformSu3RuleCacheKey,
};

/// Re-exported for the prelude: the transpose-kernel selection consumed by
/// [`RuntimeBuilder::transpose_backend`] (defined next to the kernel adapter
/// it configures; see its docs for the opt-in rationale).
pub use tenet_tensors::TransposeBackend;

use crate::error::Error;
use crate::plancache::{Optimizer, PlanCacheConfig};

pub type Ctx<D, Key> = TensorContractFusionExecutionContext<D, Key>;
pub(crate) type BuiltinKey = TreeTransformBuiltinRuleCacheKey;
pub(crate) type ProductKey = TreeTransformProductRuleCacheKey<BuiltinKey, BuiltinKey>;
/// Cache key of the left-associated triple product `(fZ2 ⊠ U1) ⊠ SU2`.
pub(crate) type TripleKey = TreeTransformProductRuleCacheKey<ProductKey, BuiltinKey>;
/// Cache key of the Stage B3b SU(3) table provider (its provenance hash).
pub(crate) type Su3Key = TreeTransformSu3RuleCacheKey;

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

/// Builds one contraction/recoupling backend on the runtime's shared CPU
/// context (one rayon pool per runtime — see `RuntimeExecutionConfig::
/// shared_ctx`); a `gemm_kind` of `Blas` fails if no `cpu-blas`/`blas-*`
/// provider was compiled in.
fn make_transform_ops(
    ctx: &tenet_dense::SharedCpuContext,
    gemm_kind: Option<tenet_dense::CpuBackendKind>,
) -> Result<DenseTreeTransformOperations, Error> {
    Ok(DenseTreeTransformOperations::new(
        tenet_dense::DefaultDenseExecutor::with_shared_context(ctx, gemm_kind)
            .map_err(tenet_tensors::OperationError::Dense)?,
    ))
}

impl<Key: Clone + Eq + Hash + Send + Sync + 'static> Ctxs<Key> {
    /// Builds the per-scalar contexts on the runtime's shared CPU context,
    /// optionally with an explicit CPU GEMM provider.
    pub(crate) fn with_config(
        ctx: &tenet_dense::SharedCpuContext,
        gemm_kind: Option<tenet_dense::CpuBackendKind>,
    ) -> Result<Self, Error> {
        Ok(Self {
            f64:
                Ctx::with_parts(
                    tenet_tensors::TreeTransformExecutionContext::new(make_transform_ops(
                        ctx, gemm_kind,
                    )?),
                    make_transform_ops(ctx, gemm_kind)?,
                    <DenseTreeTransformOperations as tenet_tensors::TensorContractBackend<
                        f64,
                        f64,
                    >>::Workspace::default(),
                    tenet_tensors::TensorContractCache::new(),
                ),
            c64: Ctx::with_parts(
                tenet_tensors::TreeTransformExecutionContext::new(make_transform_ops(
                    ctx, gemm_kind,
                )?),
                make_transform_ops(ctx, gemm_kind)?,
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
    /// Stage B3b: SU(3) generic-fusion execution context (permute/braid/
    /// transpose). Keyed by the table provenance hash, so a swapped table never
    /// reuses another table's compiled plans.
    pub(crate) su3: Ctxs<Su3Key>,
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
}

/// Contraction-plan cache home, behind its own mutex in [`RuntimeInner`] rather
/// than the coarse `state` mutex (#155): the network hot path locks only this,
/// never contending with standalone ops, and reads config + accesses the slot
/// in one acquisition (see [`Runtime::with_plan_cache`]).
struct PlanCacheHome {
    /// Contraction-plan cache configuration (the cache state itself lives
    /// in `slot`).
    config: PlanCacheConfig,
    /// Type-erased downstream extension slot. Currently holds the
    /// contraction-plan cache: the cache and plan types live in
    /// `tenet-network`, which depends on this crate, so the runtime can only
    /// hold them behind `dyn Any`; `tenet-network` claims and downcasts the
    /// slot on first use.
    slot: Option<Box<dyn Any + Send>>,
}

impl RuntimeState {
    // Why no `Ctxs::default()` fast path anymore: even the "default" runtime
    // must route every executor through the shared CPU context — the default
    // constructors each build a private env-driven rayon pool, and 28 of them
    // per runtime is exactly the thread explosion #155's pools amplified.
    fn with_config(
        dense: Box<dyn tenet_dense::DenseExecutor + Send>,
        ctx: &tenet_dense::SharedCpuContext,
        gemm_kind: Option<tenet_dense::CpuBackendKind>,
    ) -> Result<Self, Error> {
        Ok(Self {
            u1: Ctxs::with_config(ctx, gemm_kind)?,
            z2: Ctxs::with_config(ctx, gemm_kind)?,
            fz2: Ctxs::with_config(ctx, gemm_kind)?,
            su2: Ctxs::with_config(ctx, gemm_kind)?,
            u1_fz2: Ctxs::with_config(ctx, gemm_kind)?,
            fz2_u1_su2: Ctxs::with_config(ctx, gemm_kind)?,
            su3: Ctxs::with_config(ctx, gemm_kind)?,
            dense,
            #[cfg(feature = "cuda")]
            cuda: None,
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

    /// Applies the transpose-kernel selection to every per-rule execution
    /// context. Unlike the replay worker count (a tree-transform-only
    /// property), the transpose kernel drives pure permuted copies in BOTH
    /// backends each context holds — the tree-transform backend (replay
    /// pack/scatter) and the contraction backend (fusion-block pack/scatter)
    /// — so both are set, for every rule including SU(3).
    fn set_transpose_backend(&mut self, backend: TransposeBackend) {
        fn apply<Key: Clone + Eq + Hash + Send + Sync + 'static>(
            ctxs: &mut Ctxs<Key>,
            backend: TransposeBackend,
        ) {
            ctxs.f64
                .tree_context_mut()
                .backend_mut()
                .set_transpose_backend(backend);
            ctxs.f64
                .contract_backend_mut()
                .set_transpose_backend(backend);
            ctxs.c64
                .tree_context_mut()
                .backend_mut()
                .set_transpose_backend(backend);
            ctxs.c64
                .contract_backend_mut()
                .set_transpose_backend(backend);
        }
        apply(&mut self.u1, backend);
        apply(&mut self.z2, backend);
        apply(&mut self.fz2, backend);
        apply(&mut self.su2, backend);
        apply(&mut self.u1_fz2, backend);
        apply(&mut self.fz2_u1_su2, backend);
        apply(&mut self.su3, backend);
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
            // See `with_rule!`: SU(3) is Generic and takes dedicated `*_generic`
            // paths that branch on `RuleKind::Su3` before reaching this macro.
            $crate::space::RuleKind::Su3 => {
                unimplemented!(
                    "this operation is not yet supported for SU(3) tensors \
                     (Stage B3b implements permute/braid/transpose)"
                )
            }
        }
    };
}
pub(crate) use with_rule_ctx;

struct RuntimeInner {
    // The coarse state mutex is now cold on the CPU hot paths: standalone ops
    // lease from `context_pool`/`executor_pool` (below) and the network path
    // uses per-plan workspace pools, so this is held only for CUDA device state,
    // plan-cache config, and the non-mintable injected-executor fallback (#155).
    state: Mutex<RuntimeState>,
    rand_counter: AtomicU64,
    execution_config: RuntimeExecutionConfig,
    /// Standalone-op parallelism (#155): rather than hold the coarse `state`
    /// mutex for a whole `contract`/`permute`/factorization, each op leases a
    /// per-rule execution context (and, for factorizations, a dense executor)
    /// for its duration and returns it, so ops on a shared `Runtime` run
    /// concurrently. Both pools mirror the network `WorkspacePool`: mint on
    /// empty, bounded idle count, quarantine-on-panic.
    ///
    /// Why not one shared `Sync` executor instead of a pool: `DenseExecutor`
    /// takes `&mut self` (per-call scratch), and a future CUDA per-stream
    /// executor would carry non-`Sync` device state — a pool of owned
    /// executors is the share strategy that survives that, a `&`-shared one is
    /// not.
    context_pool: Mutex<Vec<crate::tensor::TensorExecutionContext>>,
    executor_pool: Mutex<Vec<Box<dyn tenet_dense::DenseExecutor + Send>>>,
    /// `false` when a caller injected a custom (non-mintable) executor via
    /// `with_dense_executor`: the pool cannot reproduce it, so factorizations
    /// fall back to the `state` lock and its single executor.
    executor_mintable: bool,
    /// Idle-pool cap: keep up to one warm context/executor per core so a
    /// data-parallel driver (one thread per core) reuses them instead of
    /// re-minting each call. ponytail: cores, not a tunable; raise only if a
    /// workload oversubscribes cores and re-mint churn shows up in a profile.
    max_idle: usize,
    /// Contraction-plan cache, behind its own mutex so the network hot path
    /// never contends with standalone ops on the `state` mutex (#155).
    plan_cache: Mutex<PlanCacheHome>,
}

/// Mints a dense executor identical to the one `RuntimeBuilder::build` created
/// from the same config. Only called when no custom executor was injected
/// (`executor_mintable`), and that config already built successfully once, so a
/// re-mint cannot fail. ponytail: CPU executors are cheap to construct; the pool
/// exists to skip per-call construction, not because minting is expensive.
fn mint_dense(config: &RuntimeExecutionConfig) -> Box<dyn tenet_dense::DenseExecutor + Send> {
    Box::new(
        tenet_dense::DefaultDenseExecutor::with_shared_context(
            &config.shared_ctx,
            config.linalg_kind,
        )
        .expect("dense executor config validated at Runtime build time"),
    )
}

/// RAII lease of a pooled per-rule execution context (#155). Returns it to the
/// pool on drop; on panic it is dropped instead of returned (quarantine —
/// mirrors `tenet_network`'s `WorkspaceLease`).
pub(crate) struct ContextLease<'a> {
    pool: &'a Mutex<Vec<crate::tensor::TensorExecutionContext>>,
    max_idle: usize,
    context: Option<crate::tensor::TensorExecutionContext>,
}

impl ContextLease<'_> {
    pub(crate) fn context(&mut self) -> &mut crate::tensor::TensorExecutionContext {
        self.context
            .as_mut()
            .expect("context lease always owns a context")
    }
}

impl Drop for ContextLease<'_> {
    fn drop(&mut self) {
        if std::thread::panicking() {
            // A panic mid-op may have left the context's caches half-written;
            // do not return it to the pool.
            self.context.take();
            return;
        }
        if let Some(context) = self.context.take() {
            let mut available = self.pool.lock().expect("context pool poisoned");
            if available.len() < self.max_idle {
                available.push(context);
            }
        }
    }
}

/// RAII lease of a dense executor (#155): a pooled executor for mintable
/// configs, or the `state` lock for a non-mintable injected executor.
pub(crate) enum DenseLease<'a> {
    Pooled {
        pool: &'a Mutex<Vec<Box<dyn tenet_dense::DenseExecutor + Send>>>,
        max_idle: usize,
        executor: Option<Box<dyn tenet_dense::DenseExecutor + Send>>,
    },
    Locked(MutexGuard<'a, RuntimeState>),
}

impl DenseLease<'_> {
    pub(crate) fn dense(&mut self) -> &mut (dyn tenet_dense::DenseExecutor + Send) {
        match self {
            DenseLease::Pooled { executor, .. } => &mut **executor
                .as_mut()
                .expect("dense lease always owns an executor"),
            DenseLease::Locked(guard) => &mut *guard.dense,
        }
    }
}

impl Drop for DenseLease<'_> {
    fn drop(&mut self) {
        if let DenseLease::Pooled {
            pool,
            max_idle,
            executor,
        } = self
        {
            if std::thread::panicking() {
                executor.take();
                return;
            }
            if let Some(executor) = executor.take() {
                let mut available = pool.lock().expect("executor pool poisoned");
                if available.len() < *max_idle {
                    available.push(executor);
                }
            }
        }
    }
}

// No `dense_threads` field: the thread count is baked into `shared_ctx` at
// build time, so executors minted later cannot drift from it.
#[derive(Clone)]
pub(crate) struct RuntimeExecutionConfig {
    pub(crate) gemm_kind: Option<tenet_dense::CpuBackendKind>,
    pub(crate) recoupling_threads: Option<usize>,
    pub(crate) transpose_backend: Option<TransposeBackend>,
    /// CPU provider for dense factorizations (SVD/QR/eigh). Kept here so the
    /// standalone-op executor pool can re-mint an executor identical to the one
    /// `RuntimeBuilder::build` created (issue #155). `None` uses faer.
    pub(crate) linalg_kind: Option<tenet_dense::CpuBackendKind>,
    /// THE runtime's CPU context: one rayon pool shared by every executor this
    /// runtime mints — the state's, the executor pool's, and all 28 transform
    /// backends of every pooled `TensorExecutionContext`. Without it each
    /// executor built its own eager env-sized pool, and the #155 context pool
    /// multiplied that into a process-thread-cap failure (macOS `WouldBlock`)
    /// under concurrent leases.
    pub(crate) shared_ctx: tenet_dense::SharedCpuContext,
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

    #[doc(hidden)]
    pub fn shares_state_with(&self, other: &Runtime) -> bool {
        self.same_runtime(other)
    }

    pub(crate) fn execution_config(&self) -> &RuntimeExecutionConfig {
        &self.inner.execution_config
    }

    /// Leases a per-rule execution context for one standalone op (#155): pop a
    /// warm one or mint a fresh config-bound one. The op runs on the leased
    /// context, not under the coarse `state` lock, so ops on a shared runtime
    /// run concurrently. Byte-identical to the old locked path: the per-rule
    /// machinery is the same `Ctxs`, and single-threaded use reuses one pooled
    /// context (its caches warm exactly as the runtime state's did).
    pub(crate) fn lease_context(&self) -> Result<ContextLease<'_>, Error> {
        let pooled = self
            .inner
            .context_pool
            .lock()
            .expect("context pool poisoned")
            .pop();
        let context = match pooled {
            Some(context) => context,
            None => {
                crate::tensor::TensorExecutionContext::for_config(&self.inner.execution_config)?
            }
        };
        Ok(ContextLease {
            pool: &self.inner.context_pool,
            max_idle: self.inner.max_idle,
            context: Some(context),
        })
    }

    /// Leases a dense executor for one factorization (#155). Pooled for a
    /// mintable config; otherwise falls back to the `state` lock and its single
    /// injected executor (which cannot be reproduced for a pool).
    pub(crate) fn lease_dense(&self) -> DenseLease<'_> {
        if !self.inner.executor_mintable {
            return DenseLease::Locked(self.lock());
        }
        let executor = self
            .inner
            .executor_pool
            .lock()
            .expect("executor pool poisoned")
            .pop()
            .unwrap_or_else(|| mint_dense(&self.inner.execution_config));
        DenseLease::Pooled {
            pool: &self.inner.executor_pool,
            max_idle: self.inner.max_idle,
            executor: Some(executor),
        }
    }

    fn lock_plan_cache(&self) -> MutexGuard<'_, PlanCacheHome> {
        self.inner
            .plan_cache
            .lock()
            .expect("tenet plan-cache poisoned")
    }

    /// Snapshot of this runtime's contraction-plan-cache configuration.
    pub fn plan_cache_config(&self) -> PlanCacheConfig {
        self.lock_plan_cache().config.clone()
    }

    /// Replaces the contraction-plan-cache configuration.
    pub fn set_plan_cache_config(&self, config: PlanCacheConfig) {
        self.lock_plan_cache().config = config;
    }

    /// Locked access to the type-erased downstream extension slot
    /// (currently the contraction-plan cache: the cache type lives in
    /// `tenet-network`, which claims and downcasts the slot on first use).
    /// Expert seam for `tenet-network`; do not hold tensors' operations
    /// inside `f` (the plan-cache mutex is held for its duration).
    #[doc(hidden)]
    pub fn with_extension_slot<R>(
        &self,
        f: impl FnOnce(&mut Option<Box<dyn Any + Send>>) -> R,
    ) -> R {
        f(&mut self.lock_plan_cache().slot)
    }

    /// Reads the plan-cache config AND accesses the slot under ONE plan-cache
    /// lock (#155): the network hot path resolves enable/replan policy and the
    /// cache lookup in a single acquisition instead of two.
    #[doc(hidden)]
    pub fn with_plan_cache<R>(
        &self,
        f: impl FnOnce(&PlanCacheConfig, &mut Option<Box<dyn Any + Send>>) -> R,
    ) -> R {
        let mut home = self.lock_plan_cache();
        let home = &mut *home;
        f(&home.config, &mut home.slot)
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
    /// Selected transpose kernel for pure permuted copies; `None` keeps the
    /// fused-loop default (dispatch-identical to not having the knob).
    transpose_backend: Option<TransposeBackend>,
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
            .field("transpose_backend", &self.transpose_backend)
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

    /// Selects the CPU **transpose kernel** for pure permuted copies (the
    /// pack / assign-scatter of tree-transform replay and fusion-block
    /// contraction). Unset keeps [`TransposeBackend::FusedLoops`], which is
    /// byte- and dispatch-identical to not having the knob at all.
    ///
    /// [`TransposeBackend::StridedPerm`] is **opt-in on measured numbers**
    /// (issue #114): it loses badly on small-degeneracy SU(2) replay (d=4
    /// transposes ~2x slower — its per-call plan build cannot amortize over
    /// many tiny blocks) and only wins on large-block abelian
    /// transpose-heavy workloads. Backend choice never changes results; see
    /// `docs/backend_policy.md` for the regime table.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenet::prelude::*;
    ///
    /// let rt = Runtime::builder()
    ///     .transpose_backend(TransposeBackend::StridedPerm)
    ///     .build()?;
    /// # let _ = rt;
    /// # Ok::<(), tenet::prelude::Error>(())
    /// ```
    pub fn transpose_backend(mut self, backend: TransposeBackend) -> Self {
        self.transpose_backend = Some(backend);
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
        // A custom injected executor cannot be re-minted for the pool; those
        // runtimes fall back to the state lock for factorizations (#155).
        let executor_mintable = self.dense_executor.is_none();
        let linalg_kind = self.linalg_backend.map(LinalgBackend::to_kind);
        // THE runtime's one CPU context (rayon pool): every executor below —
        // factorization, executor-pool mints, all per-rule transform backends,
        // and every pooled TensorExecutionContext — shares it. dense_threads
        // pins the count; otherwise the environment decides once, here, instead
        // of once per executor.
        let shared_ctx = match dense_threads {
            Some(threads) => tenet_dense::SharedCpuContext::with_threads(threads)
                .map_err(tenet_tensors::OperationError::Dense)?,
            None => tenet_dense::SharedCpuContext::from_env(),
        };
        // Injected backend wins; otherwise build the selected provider (faer by
        // default) on the shared context.
        let dense: Box<dyn tenet_dense::DenseExecutor + Send> = match self.dense_executor {
            Some(executor) => executor,
            None => Box::new(
                tenet_dense::DefaultDenseExecutor::with_shared_context(&shared_ctx, linalg_kind)
                    .map_err(tenet_tensors::OperationError::Dense)?,
            ),
        };
        let gemm_kind = self.gemm_backend.map(LinalgBackend::to_kind);
        let mut state = RuntimeState::with_config(dense, &shared_ctx, gemm_kind)?;
        let plan_cache = PlanCacheHome {
            config: self.plan_cache,
            slot: None,
        };
        if let Some(threads) = self.recoupling_threads {
            state.set_recoupling_threads(threads);
        }
        if let Some(transpose) = self.transpose_backend {
            state.set_transpose_backend(transpose);
        }
        #[cfg(feature = "cuda")]
        if let Some(device) = self.cuda_device {
            state.cuda = Some(
                tenet_dense::CudaDenseContext::new(device)
                    .map_err(tenet_tensors::OperationError::Dense)?,
            );
        }
        // One warm context/executor per core covers a thread-per-core driver;
        // fall back to a small count if the core count is unavailable.
        let max_idle = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(4)
            .max(2);
        Ok(Runtime {
            inner: Arc::new(RuntimeInner {
                state: Mutex::new(state),
                rand_counter: AtomicU64::new(0),
                execution_config: RuntimeExecutionConfig {
                    gemm_kind,
                    recoupling_threads: self.recoupling_threads,
                    transpose_backend: self.transpose_backend,
                    linalg_kind,
                    shared_ctx,
                },
                context_pool: Mutex::new(Vec::new()),
                executor_pool: Mutex::new(Vec::new()),
                executor_mintable,
                max_idle,
                plan_cache: Mutex::new(plan_cache),
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

thread_local! {
    static DEFAULT_RUNTIME: RefCell<Option<Runtime>> = const { RefCell::new(None) };
}

/// Sets the calling thread's default [`Runtime`], used by the argument-free
/// tensor constructors ([`crate::prelude::zeros`], [`crate::prelude::rand`], …).
///
/// The default is **thread-local**: it is not shared with other threads, and
/// passing a runtime explicitly (`Tensor::zeros(&rt, …)` or `rt.zeros(…)`)
/// always works regardless of it — that is the escape hatch for using several
/// runtimes at once (e.g. per MPI rank, or comparing backends). Call once near
/// program start; a later call overwrites the default on this thread.
///
/// [`default!`](crate::default) is shorthand: `default!(rt)` == `set_default_runtime(&rt)`.
///
/// # Examples
///
/// ```
/// use tenet::prelude::*;
///
/// let rt = Runtime::builder().build()?;
/// default!(rt); // set once for this thread; equivalently set_default_runtime(&rt)
///
/// let v = Space::u1([(0, 1), (1, 1)]);
/// let a = zeros(Dtype::F64, [&v], [&v])?; // no runtime argument
/// assert_eq!(a.norm()?, 0.0);
///
/// // Explicit still works for a second runtime (e.g. another backend / rank):
/// let rt2 = Runtime::builder().build()?;
/// let b = rt2.zeros(Dtype::F64, [&v], [&v])?;
/// # let _ = (a, b);
/// # Ok::<(), tenet::prelude::Error>(())
/// ```
pub fn set_default_runtime(rt: &Runtime) {
    DEFAULT_RUNTIME.with(|cell| *cell.borrow_mut() = Some(rt.clone()));
}

/// Returns the calling thread's default [`Runtime`] (a cheap handle clone), or
/// an error if none was set with [`set_default_runtime`] / [`default!`](crate::default).
pub fn default_runtime() -> Result<Runtime, Error> {
    DEFAULT_RUNTIME.with(|cell| {
        cell.borrow().clone().ok_or_else(|| {
            Error::InvalidArgument(
                "no default runtime set on this thread; call set_default_runtime(&rt) \
                 (or default!(rt)), or pass a runtime explicitly"
                    .to_string(),
            )
        })
    })
}

/// Clears the calling thread's default [`Runtime`] (mainly for test isolation).
pub fn clear_default_runtime() {
    DEFAULT_RUNTIME.with(|cell| *cell.borrow_mut() = None);
}

/// Sets the calling thread's default runtime: `default!(rt)` is shorthand for
/// [`set_default_runtime`]`(&rt)`.
#[macro_export]
macro_rules! default {
    ($rt:expr) => {
        $crate::set_default_runtime(&$rt)
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    // All context/provider combinations of the contraction-backend builder
    // must construct on faer (always compiled), and each must land on the
    // context it was given (the one-pool-per-Runtime invariant, #155).
    #[test]
    fn transform_ops_builds_for_every_faer_config() {
        let faer = tenet_dense::CpuBackendKind::Faer;
        let serial = tenet_dense::SharedCpuContext::with_threads(1).expect("serial context");
        let env = tenet_dense::SharedCpuContext::from_env();
        for ctx in [&serial, &env] {
            let ops = make_transform_ops(ctx, None).expect("default transform ops");
            assert!(ops.dense().shares_cpu_context(ctx));
            // Explicit-kind arm: build must succeed; context sharing holds
            // whenever the kind IS the compiled default (all-faer builds), but
            // an explicit non-default kind keeps a private context (see
            // `DefaultDenseExecutor::with_shared_context`), so no sharing
            // assert here — it would flip on blas-featured builds.
            let ops = make_transform_ops(ctx, Some(faer)).expect("faer transform ops");
            drop(ops);
        }
    }

    /// `RuntimeBuilder::transpose_backend(StridedPerm)` must reach BOTH
    /// backends (tree-transform and contraction) of every per-rule context —
    /// the backends whose getters feed every kernel-adapter construction in
    /// the replay/contraction drivers. Unset must stay FusedLoops. This is
    /// the builder→backend leg of the route-selection chain; the
    /// backend→dispatch leg is pinned by tenet-operations'
    /// `transpose_backend_field_switches_the_route`.
    #[test]
    fn builder_transpose_backend_reaches_every_context_backend() {
        fn assert_all<Key: Clone + Eq + Hash + Send + Sync + 'static>(
            ctxs: &mut Ctxs<Key>,
            expected: TransposeBackend,
        ) {
            assert_eq!(
                ctxs.f64
                    .tree_context_mut()
                    .backend_mut()
                    .transpose_backend(),
                expected
            );
            assert_eq!(ctxs.f64.contract_backend().transpose_backend(), expected);
            assert_eq!(
                ctxs.c64
                    .tree_context_mut()
                    .backend_mut()
                    .transpose_backend(),
                expected
            );
            assert_eq!(ctxs.c64.contract_backend().transpose_backend(), expected);
        }
        fn assert_state(runtime: &Runtime, expected: TransposeBackend) {
            let mut state = runtime.inner.state.lock().unwrap();
            let state = &mut *state;
            assert_all(&mut state.u1, expected);
            assert_all(&mut state.z2, expected);
            assert_all(&mut state.fz2, expected);
            assert_all(&mut state.su2, expected);
            assert_all(&mut state.u1_fz2, expected);
            assert_all(&mut state.fz2_u1_su2, expected);
            assert_all(&mut state.su3, expected);
        }

        let selected = Runtime::builder()
            .transpose_backend(TransposeBackend::StridedPerm)
            .build()
            .unwrap();
        assert_state(&selected, TransposeBackend::StridedPerm);

        let default = Runtime::builder().build().unwrap();
        assert_state(&default, TransposeBackend::FusedLoops);
    }
}
