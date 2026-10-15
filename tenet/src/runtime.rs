//! User-layer runtime: owns shared execution and cache state so everyday
//! tensor code never passes explicit contexts around.

use std::any::Any;
use std::hash::Hash;
use std::num::NonZeroUsize;
#[cfg(feature = "cuda")]
use std::sync::PoisonError;
use std::sync::{Arc, Mutex, MutexGuard, Weak};

use num_complex::{Complex32, Complex64};
use tenet_core::{HomSpaceId, RuleIdentity};
use tenet_tensors::{
    BoundDynamicFusionMapSpace, DenseTreeTransformOperations, TensorContractFusionExecutionContext,
    TreeTransformOperation,
};

use crate::error::Error;
use crate::typed::ScalarOps;

#[cfg(feature = "cuda")]
mod cuda;
mod host_scratch;
mod lanes;
mod leases;
#[cfg(test)]
mod tests;

#[cfg(feature = "cuda")]
pub(crate) use self::cuda::*;
#[cfg(feature = "cuda")]
pub use self::cuda::CudaTreeTransformStats;
pub(crate) use self::lanes::*;
pub use self::lanes::{Ctx, Ctxs};
pub(crate) use self::leases::*;

/// Type-keyed storage for state a downstream crate keeps on a [`Runtime`].
///
/// Each value lives under its own type, so a lookup returns the value of the
/// requested type or nothing. Why not one `Option<Box<dyn Any>>`: a value of
/// another type in that single slot left its owner a choice between a panic
/// and silently dropping someone else's state.
#[doc(hidden)]
#[derive(Default)]
pub struct ExtensionSlot {
    values: Vec<Box<dyn Any + Send>>,
}

impl ExtensionSlot {
    /// The stored value of type `T`, if one was inserted.
    pub fn get_mut<T: Any + Send>(&mut self) -> Option<&mut T> {
        self.values
            .iter_mut()
            .find_map(|value| value.as_mut().downcast_mut::<T>())
    }

    /// The stored value of type `T`, inserting `init()` first if there is none.
    pub fn get_or_insert_with<T: Any + Send>(&mut self, init: impl FnOnce() -> T) -> &mut T {
        let index = match self
            .values
            .iter()
            .position(|value| value.as_ref().is::<T>())
        {
            Some(index) => index,
            None => {
                self.values.push(Box::new(init()));
                self.values.len() - 1
            }
        };
        self.values[index]
            .as_mut()
            .downcast_mut::<T>()
            .expect("the value at `index` was found or inserted as a `T`")
    }
}

struct RuntimeInner {
    // Standalone CPU ops normally lease from `context_pool`/`executor_pool`,
    // while network execution uses per-plan workspace pools. Pool checkout,
    // cache access, provider resources, and a non-mintable injected executor
    // can still synchronize. Device operations do not take this lock: they
    // serialize on the device-local `cuda` mutex below instead.
    state: Mutex<RuntimeState>,
    execution_config: RuntimeExecutionConfig,
    /// Standalone operations lease an execution context and, for factorization,
    /// a dense executor. Both pools mint on empty, bound their idle count, and
    /// quarantine a resource that unwinds while leased.
    ///
    /// `DenseExecutor` takes `&mut self` for per-call scratch, so leases own
    /// mutable executor state rather than sharing one executor by `&`.
    context_pool: Mutex<Vec<PooledContext>>,
    executor_pool: Mutex<Vec<Box<dyn tenet_dense::DenseExecutor + Send>>>,
    /// `false` when a caller injected a custom (non-mintable) executor via
    /// `with_dense_executor`: the pool cannot reproduce it, so factorizations
    /// fall back to the `state` lock and its single executor.
    executor_mintable: bool,
    /// Idle-pool cap derived from available parallelism, with a minimum of two
    /// and a fallback of four when the system does not report a value.
    max_idle: usize,
    /// Contraction-plan cache behind its own mutex, separate from `state` but
    /// still synchronized while cache work is in progress.
    extensions: Mutex<ExtensionSlot>,
    /// The single CUDA context of this runtime and its device's process-wide
    /// lock. Device operations serialize on both (see [`CudaLease`]) instead
    /// of on `state`.
    #[cfg(feature = "cuda")]
    cuda: Option<CudaHome>,
    /// Device ordinal of `cuda`, fixed at build time and readable without a
    /// lock so placement preflight takes no device lock at all.
    #[cfg(feature = "cuda")]
    cuda_device: Option<usize>,
}

/// Pool entry for `RuntimeInner::context_pool`. Boxed: a context owns both the
/// multiplicity-free and Generic-fusion lanes, each with `f64` and `c64`
/// state. `Vec::pop`/`push` must move only a pointer rather than that complete
/// execution state; profiling showed by-value moves were ~70% of a small
/// standalone op's cost (issue #228). The pointer-size canary test below
/// prevents that regression.
type PooledContext = Box<TensorExecutionContext>;

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

// No `dense_threads` field: the thread count is baked into `shared_ctx` at
// build time, so executors minted later cannot drift from it.
#[derive(Clone)]
pub(crate) struct RuntimeExecutionConfig {
    pub(crate) gemm_kind: Option<tenet_dense::CpuBackendKind>,
    pub(crate) recoupling_threads: Option<NonZeroUsize>,
    /// CPU provider for dense factorizations (SVD/QR/eigh). Kept here so the
    /// standalone-op executor pool can re-mint an executor identical to the one
    /// `RuntimeBuilder::build` created (issue #155). `None` uses Tenferro's
    /// resolved compiled provider default.
    pub(crate) linalg_kind: Option<tenet_dense::CpuBackendKind>,
    /// Runtime CPU context shared by built-in executors that use the compiled
    /// default kind: the state, mintable executor pool, and transform backends.
    /// An explicitly requested nondefault kind uses its own provider context;
    /// an injected executor owns its own configuration.
    pub(crate) shared_ctx: tenet_dense::SharedCpuContext,
}

/// Execution runtime for the user-layer [`crate::typed::TensorMap`] API.
///
/// A `Runtime` is built once via [`Runtime::builder`] and then carried
/// implicitly by every tensor created from it; operations reuse the
/// Runtime-owned completed tree-transform store without explicit context
/// arguments. Context-local operation caches stay disabled; cloning a
/// `Runtime` clones a shared handle, not the state.
///
/// Concurrency: standalone tensor operations can lease independent execution
/// contexts instead of holding the coarse state mutex for their full duration.
/// Pool checkout, shared stores, the plan cache, and dense providers retain
/// their own synchronization and capability limits. Device operations take
/// only the device lock (below) and a mutex over this runtime's single CUDA
/// context, not the state mutex, so Host work is not blocked by device work.
///
/// CUDA: every device operation of every `Runtime` built for the same device
/// ordinal holds one process-wide lock for that device while it submits work,
/// so device operations on one device serialize their host-side enqueue across
/// Runtimes (the GPU work itself stays asynchronous; the lock adds no host
/// sync). A host sync that already happens under a lease — `to_host`, scalar
/// and spectrum downloads — therefore
/// stalls every Runtime on the device, not only the caller. This is what makes
/// a fresh device output, returned by any operation that creates a new
/// tensor, safe to read through another Runtime on the same device (#1384).
///
/// A buffer bound in an earlier lease and written again later (a
/// `*_into` destination, or reused scratch) is ordered by the
/// device's single CubeCL stream instead (#1391): the first device Runtime of
/// the process sets CubeCL's `streaming.max_streams` to 1, so all device work
/// of every thread, TeNeT or not, runs in enqueue order and never overlaps on
/// the GPU. Building a device Runtime fails with an unsupported error if
/// CubeCL's configuration was already loaded with more streams.
/// Process-wide side effects: a `cubecl.toml` `streaming.max_streams` value is
/// overridden without notice, the setting stays fixed even if opening the
/// device then fails, every other CubeCL client in the process (wgpu
/// included) also gets one stream, and a later `CubeClRuntimeConfig::set`
/// panics. One ordering residual, independent of the stream count, is tracked
/// in tensor4all/tenferro-rs#1868 (see `docs/backend_policy.md`).
///
/// # Examples
///
/// ```
/// use std::sync::Arc;
/// use tenet::sector::{Z2FusionRule, Z2Irrep};
/// use tenet::typed::{GradedSpace, Runtime, TensorMap};
///
/// let rt = Runtime::builder().build()?;
/// let v = GradedSpace::try_new(
///     Arc::new(Z2FusionRule),
///     [(Z2Irrep::EVEN, 1), (Z2Irrep::ODD, 1)],
/// )?;
/// let a: TensorMap<_, f64> = TensorMap::zeros(&rt, [&v], [&v])?;
/// assert_eq!(a.norm(2.0)?, 0.0);
/// # Ok::<(), tenet::typed::Error>(())
/// ```
#[derive(Clone)]
pub struct Runtime {
    inner: Arc<RuntimeInner>,
}

/// Non-owning identity for internal state parked outside an active execution.
///
/// Equality and hashing use the Runtime allocation's address. The held `Weak`
/// keeps that allocation reserved, so a later Runtime can never reuse the
/// address while this identity is live.
#[doc(hidden)]
#[derive(Clone, Debug)]
pub struct RuntimeIdentity {
    inner: Weak<RuntimeInner>,
}

impl PartialEq for RuntimeIdentity {
    fn eq(&self, other: &Self) -> bool {
        Weak::ptr_eq(&self.inner, &other.inner)
    }
}

impl Eq for RuntimeIdentity {}

impl std::hash::Hash for RuntimeIdentity {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.inner.as_ptr().hash(state);
    }
}

impl RuntimeIdentity {
    #[doc(hidden)]
    pub fn matches(&self, runtime: &Runtime) -> bool {
        self.inner
            .upgrade()
            .is_some_and(|inner| Arc::ptr_eq(&inner, &runtime.inner))
    }

    #[doc(hidden)]
    pub fn is_alive(&self) -> bool {
        self.inner.strong_count() != 0
    }
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

    pub(crate) fn identity(&self) -> RuntimeIdentity {
        RuntimeIdentity {
            inner: Arc::downgrade(&self.inner),
        }
    }

    #[cfg(test)]
    pub(crate) fn execution_config(&self) -> &RuntimeExecutionConfig {
        &self.inner.execution_config
    }

    /// The resolved transformer of a typed Host `*_into` request whose spaces
    /// were proved to match once before (the exact-layout memo on the
    /// process-global entry of coefficient type `C`); `None` takes the full
    /// path.
    pub(crate) fn exact_layout_tree_pair_hit_of<R, C: 'static + Send + Sync>(
        rule: &RuleIdentity,
        operation: tenet_tensors::TreeTransformOperationView<'_>,
        source: &BoundDynamicFusionMapSpace<R>,
        destination: &BoundDynamicFusionMapSpace<R>,
    ) -> Option<tenet_tensors::TreeTransformStructure<C>> {
        let (source_homspace, source_layout) = bound_layout_identity(source);
        let (destination_homspace, destination_layout) = bound_layout_identity(destination);
        tenet_tensors::exact_layout_tree_pair_hit::<C>(
            rule,
            operation,
            destination.space().structure(),
            source.space().structure(),
            (&source_homspace, source_layout),
            (&destination_homspace, destination_layout),
        )
    }

    pub(crate) fn admit_exact_tree_pair_layout_of<R, C: 'static + Send + Sync>(
        rule: &RuleIdentity,
        operation: &TreeTransformOperation,
        source: &BoundDynamicFusionMapSpace<R>,
        destination: &BoundDynamicFusionMapSpace<R>,
    ) -> bool {
        let (source_homspace, source_layout) = bound_layout_identity(source);
        let (destination_homspace, destination_layout) = bound_layout_identity(destination);
        tenet_tensors::admit_exact_tree_pair_layout::<C>(
            rule,
            operation,
            destination.space().structure(),
            source.space().structure(),
            (&source_homspace, source_layout),
            (&destination_homspace, destination_layout),
        )
    }

    /// Enters this runtime's CPU pool as the calling thread's Host pool until
    /// the guard drops. Every Host eager operation runs inside one: through
    /// its execution lease, or directly for the strided-only operations.
    pub(crate) fn enter_host_pool(&self) -> HostPoolEntry {
        HostPoolEntry {
            _guard: tenet_operations::host_pool::enter_host_pool(
                &self.inner.execution_config.shared_ctx,
            ),
            #[cfg(test)]
            _probe: HostPoolDepth::enter(),
        }
    }

    /// Leases an execution context for one standalone op: pop an idle one or
    /// mint a fresh config-bound one. Each context owns one
    /// `RuleIdentity`-keyed multiplicity-free lane and a separate Generic-fusion
    /// lane. The coarse `state` lock is not held during that lease, but pool
    /// checkout and other shared resources remain synchronized.
    pub(crate) fn lease_context(&self) -> Result<ContextLease<'_>, Error> {
        let pooled = self
            .inner
            .context_pool
            .lock()
            .expect("context pool poisoned")
            .pop();
        let context = match pooled {
            Some(context) => context,
            None => Box::new(TensorExecutionContext::for_config(
                &self.inner.execution_config,
            )?),
        };
        Ok(ContextLease {
            pool: &self.inner.context_pool,
            max_idle: self.inner.max_idle,
            context: Some(context),
            _host_pool: self.enter_host_pool(),
        })
    }

    /// Leases a dense executor for one factorization (#155). Pooled for a
    /// mintable config; otherwise falls back to the `state` lock and its single
    /// injected executor (which cannot be reproduced for a pool).
    pub(crate) fn lease_dense(&self) -> DenseLease<'_> {
        #[cfg(test)]
        crate::typed::probe::hit(crate::typed::probe::Event::DenseLease);
        if !self.inner.executor_mintable {
            return DenseLease::Locked {
                state: self.lock(),
                _host_pool: self.enter_host_pool(),
            };
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
            _host_pool: self.enter_host_pool(),
        }
    }

    /// Locked access to the type-erased downstream extension slot (the
    /// contraction-plan cache and its configuration: the types live in
    /// `tenet-network`, which depends on this crate and claims and downcasts
    /// its entry on first use). Behind its own mutex rather than the coarse
    /// `state` one (#155), so the network hot path never contends with
    /// standalone ops. Expert seam for `tenet-network`; the mutex is held for
    /// the duration of `f`, so do not run tensor operations inside it.
    pub(crate) fn with_extension_slot<R>(&self, f: impl FnOnce(&mut ExtensionSlot) -> R) -> R {
        f(&mut self
            .inner
            .extensions
            .lock()
            .expect("tenet extension slot poisoned"))
    }

    /// CUDA device ordinal fixed when this runtime was built. Lock-free: the
    /// ordinal is immutable, so placement preflight needs no device lock.
    #[cfg(feature = "cuda")]
    pub(crate) fn cuda_device_ordinal(&self) -> Option<usize> {
        self.inner.cuda_device
    }

    /// The device ordinal, or the missing-device error every device operation
    /// reports. Preflight-only sites use this instead of a lease.
    #[cfg(feature = "cuda")]
    pub(crate) fn cuda_device_ordinal_checked(&self) -> Result<usize, Error> {
        self.inner.cuda_device.ok_or_else(missing_cuda_device)
    }

    /// Leases the device state for a path that only drops or reads it.
    ///
    /// Unlike [`Self::lease_cuda`], a poisoned mutex is recovered rather than
    /// re-panicked: clearing prepared device state and reading its byte
    /// counters are safe whatever a panicking device operation left behind —
    /// every entry is a dtype conversion of data the structure still owns, and
    /// the workspaces are fully rewritten before they are read. Turning
    /// another thread's panic into a panic in a cache-clear or a statistics
    /// read would be the worse contract.
    #[cfg(feature = "cuda")]
    pub(crate) fn lease_cuda_for_maintenance(&self) -> Option<CudaLease<'_>> {
        self.inner.cuda.as_ref().map(|cuda| {
            let device = lock_device(cuda.device_lock);
            CudaLease {
                state: cuda.state.lock().unwrap_or_else(PoisonError::into_inner),
                _device: device,
            }
        })
    }

    /// Leases this runtime's single CUDA context for one device operation.
    ///
    /// Device operations validate first, then lease, then execute: they hold
    /// only the device lock and this Runtime's device state, never the coarse
    /// `state` lock, so Host work on the same runtime is not blocked by device
    /// work. Device operations of every Runtime on this device serialize here.
    #[cfg(feature = "cuda")]
    pub(crate) fn lease_cuda(&self) -> Result<CudaLease<'_>, Error> {
        // ponytail: poisoning treated as fatal, as for `lock()` and the pools.
        self.inner
            .cuda
            .as_ref()
            .ok_or_else(missing_cuda_device)
            .map(|cuda| {
                let device = lock_device(cuda.device_lock);
                CudaLease {
                    state: cuda.state.lock().expect("tenet cuda context poisoned"),
                    _device: device,
                }
            })
    }

    /// Device state this Runtime's tree-transform executor and device context
    /// retain, or `None` when the Runtime has no device.
    ///
    /// Read-only observation; it takes the device lease, so it waits behind a
    /// device operation in progress. A device lock poisoned by an earlier
    /// panic is recovered and the recovered state's counters are returned,
    /// rather than panicking inside a statistics read.
    #[cfg(feature = "cuda")]
    pub fn cuda_tree_transform_stats(&self) -> Option<CudaTreeTransformStats> {
        let mut lease = self.lease_cuda_for_maintenance()?;
        let (dense, executor) = lease.split();
        Some(CudaTreeTransformStats {
            prepared_structures: executor.prepared_structures(),
            executor_bytes: executor.executor_device_bytes(),
            workspace_bytes: executor.workspace_device_bytes(),
            context_scalar_operand_bytes: dense.scalar_operand_bytes(),
            required_plan_entries: executor.required_plan_entries(),
        })
    }

    /// Device bytes the general contraction's operand and core-destination
    /// scratch holds, or `None` when the Runtime has no device.
    ///
    /// Read-only and grow-only: the scratch keeps its high-water allocation
    /// per payload dtype until [`Self::clear_cuda_caches`]. It is
    /// reported apart from [`CudaTreeTransformStats`] so each device byte is
    /// counted once, and is not charged to
    /// `tenet-network`'s `PlanCacheConfig::workspace_budget_bytes`. Takes the device lease like
    /// [`Self::cuda_tree_transform_stats`].
    #[cfg(feature = "cuda")]
    pub fn cuda_contract_scratch_bytes(&self) -> Option<usize> {
        let mut lease = self.lease_cuda_for_maintenance()?;
        Some(lease.split_contract().2.device_bytes())
    }

    /// Snapshot of the cuTENSOR plan cache of this Runtime's device context,
    /// or `Ok(None)` when the Runtime has no device.
    ///
    /// Read-only observation for warm-cost contracts: every region move,
    /// zero fill and GEMM a device operation submits builds or reuses one
    /// cuTENSOR plan per operand signature, so `misses` not growing across a
    /// warm call shows it rebuilt no plan, and `evictions` growing shows
    /// thrash. Nothing reads it back to decide execution. Takes the device
    /// lease like [`Self::cuda_tree_transform_stats`]; an error is the
    /// backend's own statistics query failing.
    #[cfg(feature = "cuda")]
    pub fn cuda_plan_cache_stats(&self) -> Result<Option<tenet_dense::CudaPlanCacheStats>, Error> {
        let Some(lease) = self.lease_cuda_for_maintenance() else {
            return Ok(None);
        };
        lease
            .plan_cache_stats()
            .map(Some)
            .map_err(|error| Error::from(tenet_operations::OperationError::Dense(error)))
    }

    /// Drops this Runtime's retained device execution state; a no-op when the
    /// Runtime has no device.
    ///
    /// Clears the tree-transform executor's prepared structures (uploaded
    /// coefficient and recoupling vectors), its pack/scatter workspaces and
    /// the plan-entry reservation it holds on the device context, and the
    /// general contraction's operand and core-destination scratch. After it,
    /// [`CudaTreeTransformStats`]'s `prepared_structures`, `executor_bytes`,
    /// `workspace_bytes` and `required_plan_entries` and
    /// [`Self::cuda_contract_scratch_bytes`] read 0; the next device operation
    /// re-prepares what it needs, so this is a memory decision, never a
    /// correctness one.
    ///
    /// It does not touch the process-global semantic caches shared by every
    /// Runtime ([`crate::cache::clear`] does that, and leaves device state
    /// alone), the device context's shared scalar operands
    /// (`context_scalar_operand_bytes`) or its cuTENSOR plan cache, which the
    /// context owns and bounds itself, or `tenet-network`'s per-Runtime plan
    /// cache (`clear_plan_cache`).
    ///
    /// Takes the device lease, so it waits behind a device operation in
    /// progress. A device lock poisoned by an earlier panic is recovered
    /// rather than propagated: dropping prepared device state is always safe.
    #[cfg(feature = "cuda")]
    pub fn clear_cuda_caches(&self) {
        if let Some(mut lease) = self.lease_cuda_for_maintenance() {
            let (dense, executor, scratch) = lease.split_contract();
            executor.clear(dense);
            scratch.clear();
        }
    }
}

fn bound_layout_identity<R>(space: &BoundDynamicFusionMapSpace<R>) -> (HomSpaceId, [usize; 3]) {
    let space = space.space();
    (
        space.homspace().id(),
        [
            space.structure().content_id(),
            space.homspace().codomain().len(),
            space.homspace().domain().len(),
        ],
    )
}

impl std::fmt::Debug for Runtime {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Runtime").finish_non_exhaustive()
    }
}

/// Names a CPU linear-algebra provider. [`RuntimeBuilder::linalg_backend`]
/// uses it for the dense per-coupled-sector factorizations (SVD / QR / eigh /
/// eig / inv / exp); [`RuntimeBuilder::gemm_backend`] uses it, independently,
/// for contraction GEMM. Backend choice changes performance
/// only — results stay TensorKit-equivalent across providers.
///
/// The *specific* BLAS/LAPACK behind [`LinalgBackend::Blas`] (OpenBLAS, MKL,
/// Accelerate, or an injected provider) is a compile-time choice via the
/// `blas-*` cargo features; at runtime you only pick faer vs the linked BLAS.
/// Selecting `Blas` when no `cpu-blas`/`blas-*` provider was compiled in fails
/// at [`RuntimeBuilder::build`].
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum LinalgBackend {
    /// Pure-Rust faer provider, available with the `cpu-faer` feature.
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

/// Contradictory or invalid [`RuntimeBuilder`] settings, rejected by
/// [`RuntimeBuilder::build`] instead of being silently adjusted.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RuntimeConfigError {
    /// [`RuntimeBuilder::dense_threads`] was given zero.
    ZeroDenseThreads,
    /// [`RuntimeBuilder::recoupling_threads`] was given zero.
    ZeroRecouplingThreads,
    /// Both [`RuntimeBuilder::with_dense_executor`] and
    /// [`RuntimeBuilder::linalg_backend`] select the factorization provider.
    DenseExecutorWithLinalgBackend,
}

impl std::fmt::Display for RuntimeConfigError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::ZeroDenseThreads => "dense_threads must be at least 1",
            Self::ZeroRecouplingThreads => "recoupling_threads must be at least 1",
            Self::DenseExecutorWithLinalgBackend => {
                "with_dense_executor and linalg_backend both select the factorization provider"
            }
        })
    }
}

impl std::error::Error for RuntimeConfigError {}

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
    dense_threads: Option<usize>,
    recoupling_threads: Option<usize>,
    /// User-injected CPU linear-algebra backend. When absent, the selected
    /// built-in `linalg_backend` is used; when that is absent too, its compiled
    /// provider default is used.
    dense_executor: Option<Box<dyn tenet_dense::DenseExecutor + Send>>,
    /// Selected built-in CPU provider for dense factorizations (SVD/QR/eigh);
    /// `None` uses the compiled provider default. Mutually exclusive with
    /// [`Self::dense_executor`].
    linalg_backend: Option<LinalgBackend>,
    /// Selected built-in CPU provider for the contraction/recoupling GEMM;
    /// `None` uses the compiled provider default. Independent of
    /// [`Self::linalg_backend`].
    gemm_backend: Option<LinalgBackend>,
}

impl std::fmt::Debug for RuntimeBuilder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut s = f.debug_struct("RuntimeBuilder");
        #[cfg(feature = "cuda")]
        s.field("cuda_device", &self.cuda_device);
        s.field("dense_threads", &self.dense_threads)
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
    /// [`crate::typed::TensorMap::to_cuda`]; there are no implicit
    /// transfers. Device initialization happens in [`Self::build`].
    #[cfg(feature = "cuda")]
    pub fn cuda(mut self, device: usize) -> Self {
        self.cuda_device = Some(device);
        self
    }

    /// Sets the worker count of this runtime's CPU pool. The runtime owns one
    /// pool; its Host eager operations run dense kernels, tree-transform
    /// replay, plan compile and strided kernels on it, and building a runtime
    /// never touches Rayon's process-global pool. Runtimes with different
    /// counts coexist in one process.
    ///
    /// Unset, the pool uses the process's available parallelism; the runtime
    /// reads no environment variable. A value of 1 creates no worker pool:
    /// every Host operation of this runtime runs on its calling thread. Zero
    /// makes [`Self::build`] fail with [`RuntimeConfigError::ZeroDenseThreads`].
    /// The count does not configure provider-internal (BLAS) or
    /// injected-executor threads.
    pub fn dense_threads(mut self, threads: usize) -> Self {
        self.dense_threads = Some(threads);
        self
    }

    /// Selects the CPU factorization backend (SVD / QR / eigh / eig / inv / exp
    /// on the coupled-sector matrices) by injecting a [`crate::expert::DenseExecutor`].
    /// When no executor is injected, the selected built-in `linalg_backend` is
    /// used; when it is unset, the provider follows Tenferro's resolved compiled
    /// default: BLAS when its CPU build enables `cpu-blas`, otherwise faer. This
    /// is the seam for a system BLAS/LAPACK or MKL backend: implement
    /// `DenseExecutor` and pass it here — no operator or decomposition code changes.
    /// Contraction GEMM is not delegated to the injected executor; it is
    /// selected by [`Self::gemm_backend`].
    ///
    /// The injected executor owns its thread configuration. [`Self::dense_threads`]
    /// sizes the runtime CPU pool, which still runs this runtime's replay,
    /// plan compile and strided work; it does not reconfigure the injected
    /// backend.
    pub fn with_dense_executor(
        mut self,
        executor: Box<dyn tenet_dense::DenseExecutor + Send>,
    ) -> Self {
        self.dense_executor = Some(executor);
        self
    }

    /// Selects a built-in CPU provider ([`LinalgBackend::Faer`] or
    /// [`LinalgBackend::Blas`]) for the dense **factorizations** — SVD / QR /
    /// eigh / eig / inv / exp (the LAPACK-style work). Unset uses Tenferro's
    /// resolved compiled provider default: BLAS when its CPU build enables
    /// `cpu-blas`, otherwise faer. The contraction GEMM (BLAS-style work) is
    /// chosen separately with [`Self::gemm_backend`].
    ///
    /// This is the ergonomic counterpart to [`Self::with_dense_executor`] for
    /// the shipped providers; setting both makes [`Self::build`] fail with
    /// [`RuntimeConfigError::DenseExecutorWithLinalgBackend`].
    /// Choosing [`LinalgBackend::Blas`] without a compiled `cpu-blas`/`blas-*`
    /// provider fails in [`Self::build`].
    ///
    /// # Examples
    ///
    /// ```
    /// use std::sync::Arc;
    /// use tenet::sector::{U1FusionRule, U1Irrep};
    /// use tenet::typed::{GradedSpace, LinalgBackend, Runtime, Svd, TensorMap};
    ///
    /// // Explicit faer provider. Every tensor created from this runtime
    /// // factorizes on the chosen backend — no per-call argument.
    /// let rt = Runtime::builder()
    ///     .linalg_backend(LinalgBackend::Faer)
    ///     .build()?;
    /// let v = GradedSpace::try_new(
    ///     Arc::new(U1FusionRule),
    ///     [(-1, 1), (0, 2), (1, 1)].map(|(q, n)| (U1Irrep::new(q), n)),
    /// )?;
    /// let t: TensorMap<_, f64> = TensorMap::rand_with_seed(&rt, [&v, &v], [&v, &v], 7)?;
    /// let Svd { u: _u, s: _s, vh: _vh } = t.svd_compact(&[0, 1], &[2, 3])?;
    ///
    /// // Switch to the system BLAS/LAPACK linked via a `blas-*` cargo feature
    /// // (OpenBLAS / MKL / Accelerate). Results are identical to faer up to
    /// // floating-point rounding; only performance differs. Without a linked
    /// // provider this returns an error, so return to the compiled default:
    /// let rt = Runtime::builder()
    ///     .linalg_backend(LinalgBackend::Blas)
    ///     .build()
    ///     .or_else(|_| Runtime::builder().build())?;
    /// # let _ = rt;
    /// # Ok::<(), tenet::typed::Error>(())
    /// ```
    pub fn linalg_backend(mut self, backend: LinalgBackend) -> Self {
        self.linalg_backend = Some(backend);
        self
    }

    /// Selects a built-in CPU provider ([`LinalgBackend::Faer`] or
    /// [`LinalgBackend::Blas`]) for the coupled-block **contraction GEMM**
    /// (`compose` / `contract` and the recoupling replays — the BLAS-style
    /// work). Unset uses Tenferro's resolved compiled provider default: BLAS
    /// when its CPU build enables `cpu-blas`, otherwise faer. Independent of
    /// [`Self::linalg_backend`]: the
    /// factorizations and the contraction GEMM can run on different providers
    /// (e.g. faer GEMM with BLAS/LAPACK factorizations, or the reverse).
    /// Choosing [`LinalgBackend::Blas`] without a compiled `cpu-blas`/`blas-*`
    /// provider fails in [`Self::build`].
    ///
    /// # Examples
    ///
    /// ```
    /// use tenet::typed::{LinalgBackend, Runtime};
    ///
    /// // This explicitly selects faer, regardless of the compiled default.
    /// let rt = Runtime::builder()
    ///     .gemm_backend(LinalgBackend::Faer)
    ///     .build()?;
    /// # let _ = rt;
    /// # Ok::<(), tenet::typed::Error>(())
    /// ```
    pub fn gemm_backend(mut self, backend: LinalgBackend) -> Self {
        self.gemm_backend = Some(backend);
        self
    }

    /// Sets the CPU worker count for symmetry recoupling replays
    /// (permute/braid/transpose tree transforms — the cold-path cost of
    /// SU(2) workloads; **not** BLAS threads) and for tree-transform plan
    /// compile. Default is 1 (serial); values above 1 request parallelism
    /// past the backend's size gate, capped by and run on this runtime's CPU
    /// pool ([`Self::dense_threads`]). Zero makes [`Self::build`] fail with
    /// [`RuntimeConfigError::ZeroRecouplingThreads`].
    pub fn recoupling_threads(mut self, threads: usize) -> Self {
        self.recoupling_threads = Some(threads);
        self
    }

    /// Finishes the build; fails with [`RuntimeConfigError`] on contradictory
    /// or invalid settings, and when a requested backend (e.g. the CUDA
    /// device) cannot be initialized.
    pub fn build(self) -> Result<Runtime, Error> {
        if self.dense_threads == Some(0) {
            return Err(RuntimeConfigError::ZeroDenseThreads.into());
        }
        let recoupling_threads = self
            .recoupling_threads
            .map(|threads| {
                NonZeroUsize::new(threads).ok_or(RuntimeConfigError::ZeroRecouplingThreads)
            })
            .transpose()?;
        if self.dense_executor.is_some() && self.linalg_backend.is_some() {
            return Err(RuntimeConfigError::DenseExecutorWithLinalgBackend.into());
        }
        // A custom injected executor cannot be re-minted for the pool; those
        // runtimes fall back to the state lock for factorizations (#155).
        let executor_mintable = self.dense_executor.is_none();
        let linalg_kind = self.linalg_backend.map(LinalgBackend::to_kind);
        // Built-in executors using the compiled default kind share this runtime
        // CPU context. Explicit nondefault providers receive a private context
        // in `with_shared_context`; injected executors own their configuration.
        let shared_ctx = match self.dense_threads {
            Some(threads) => tenet_dense::SharedCpuContext::with_threads(threads),
            None => tenet_dense::SharedCpuContext::with_available_parallelism(),
        }
        .map_err(tenet_tensors::OperationError::Dense)?;
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
        if let Some(threads) = recoupling_threads {
            state.set_recoupling_threads(threads);
        }
        #[cfg(feature = "cuda")]
        let cuda = match self.cuda_device {
            // The backend libraries (cuTENSOR, cuSOLVER/cuBLAS) initialize
            // lazily inside tenferro, so without this the first user operation
            // on this Runtime pays a one-time ~0.2 s unrelated to its size.
            // Construction is where that cost belongs; a failure here is a
            // build failure, never a half-initialized Runtime.
            Some(device) => {
                let mut cuda = tenet_dense::CudaDenseContext::new(device)
                    .map_err(tenet_tensors::OperationError::Dense)?;
                // Registered only once the ordinal opened, so the registry
                // stays bounded by real devices. Warm-up submits work, so it
                // runs under the device lock like any other submission.
                let device_lock = cuda_device_lock(device);
                {
                    let _device = lock_device(device_lock);
                    cuda.warm_up()
                        .map_err(tenet_tensors::OperationError::Dense)?;
                }
                // The executor's prepared-structure count is a device-side
                // bound of its own: the host completed-transformer cache is
                // process-global and byte-bounded only. It bounds memory,
                // never correctness.
                Some(CudaHome {
                    device_lock,
                    state: Mutex::new(CudaDeviceState {
                        dense: cuda,
                        tree_transform: tenet_operations::CudaTreeTransformExecutor::new(
                            tenet_operations::DEFAULT_COEFFICIENT_BUDGET_BYTES,
                            tenet_operations::DEFAULT_PLAN_CACHE_BUDGET_BYTES,
                        ),
                        contract_scratch: tenet_tensors::CudaContractScratch::default(),
                    }),
                })
            }
            None => None,
        };
        // One warm context/executor per core covers a thread-per-core driver;
        // fall back to a small count if the core count is unavailable.
        let max_idle = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(4)
            .max(2);
        Ok(Runtime {
            inner: Arc::new(RuntimeInner {
                state: Mutex::new(state),
                execution_config: RuntimeExecutionConfig {
                    gemm_kind,
                    recoupling_threads,
                    linalg_kind,
                    shared_ctx,
                },
                context_pool: Mutex::new(Vec::new()),
                executor_pool: Mutex::new(Vec::new()),
                executor_mintable,
                max_idle,
                extensions: Mutex::new(ExtensionSlot::default()),
                #[cfg(feature = "cuda")]
                cuda,
                #[cfg(feature = "cuda")]
                cuda_device: self.cuda_device,
            }),
        })
    }
}
