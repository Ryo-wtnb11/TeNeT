//! User-layer runtime: owns the per-rule execution/cache state so everyday
//! tensor code never passes explicit contexts around.

use std::hash::Hash;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};

use num_complex::Complex64;
use tenet_tensors::{
    TensorContractFusionExecutionContext, TreeTransformBuiltinRuleCacheKey,
    TreeTransformProductRuleCacheKey,
};

use crate::error::Error;

pub(crate) type Ctx<D, Key> = TensorContractFusionExecutionContext<D, Key>;
pub(crate) type BuiltinKey = TreeTransformBuiltinRuleCacheKey;
pub(crate) type ProductKey = TreeTransformProductRuleCacheKey<BuiltinKey, BuiltinKey>;
/// Cache key of the left-associated triple product `(fZ2 ⊠ U1) ⊠ SU2`.
pub(crate) type TripleKey = TreeTransformProductRuleCacheKey<ProductKey, BuiltinKey>;

/// The pair of per-scalar execution contexts for one fusion rule: tensor
/// operations dispatch on the stored dtype once per call and pick one side.
pub(crate) struct Ctxs<Key: Clone + Eq + Hash> {
    pub(crate) f64: Ctx<f64, Key>,
    pub(crate) c64: Ctx<Complex64, Key>,
}

impl<Key: Clone + Eq + Hash> Default for Ctxs<Key> {
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
/// let a = Tensor::zeros(&rt, [&v], [&v])?;
/// assert_eq!(a.norm()?, 0.0);
/// # Ok::<(), tenet::prelude::Error>(())
/// ```
#[derive(Clone)]
pub struct Runtime {
    inner: Arc<RuntimeInner>,
}

impl Runtime {
    /// Starts building a runtime. Currently the default CPU backend is the
    /// only choice; the builder exists so device/cache options (e.g. CUDA)
    /// can land without breaking the construction pattern.
    pub fn builder() -> RuntimeBuilder {
        RuntimeBuilder {}
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
pub struct RuntimeBuilder {}

impl RuntimeBuilder {
    /// Finishes the build. Infallible today; returns `Result` so backend
    /// probing (GPU init, BLAS discovery) can fail here later without an
    /// API break.
    pub fn build(self) -> Result<Runtime, Error> {
        Ok(Runtime {
            inner: Arc::new(RuntimeInner {
                state: Mutex::new(RuntimeState::default()),
                rand_counter: AtomicU64::new(0),
            }),
        })
    }
}
