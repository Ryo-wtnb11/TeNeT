//! User-layer runtime: owns the per-rule execution/cache state so everyday
//! tensor code never passes explicit contexts around.

use std::any::Any;
use std::hash::Hash;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};

use num_complex::Complex64;
use tenet_tensors::{
    TensorContractFusionExecutionContext, TreeTransformBuiltinRuleCacheKey,
    TreeTransformProductRuleCacheKey,
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

/// Per-rule expert-layer execution contexts (contraction resolution caches,
/// tree-transform replay caches, dense backends and workspaces).
///
/// One field per supported rule; each context is created eagerly because the
/// empty contexts are cheap, and filled lazily by use.
#[derive(Default)]
pub(crate) struct RuntimeState {
    pub(crate) u1: Ctxs<BuiltinKey>,
    pub(crate) z2: Ctxs<BuiltinKey>,
    pub(crate) fz2: Ctxs<BuiltinKey>,
    pub(crate) su2: Ctxs<BuiltinKey>,
    pub(crate) u1_fz2: Ctxs<ProductKey>,
    pub(crate) fz2_u1_su2: Ctxs<TripleKey>,
    /// Rule-independent dense-factorization executor (SVD / QR / eigh on the
    /// coupled-sector matrices), shared by all decomposition methods.
    pub(crate) dense: tenet_dense::DefaultDenseExecutor,
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

/// Builder for [`Runtime`]; see [`Runtime::builder`].
#[derive(Clone, Debug, Default)]
pub struct RuntimeBuilder {
    #[cfg(feature = "cuda")]
    cuda_device: Option<usize>,
    plan_cache: PlanCacheConfig,
    recoupling_threads: Option<usize>,
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
        let mut state = RuntimeState::default();
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
