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
pub use tenet_tensors::RuntimeTreeTransformCacheInfo;

/// Snapshot returned by the deprecated [`Runtime::tree_transform_cache_info`];
/// removed with it. Observability only.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
#[non_exhaustive]
pub struct TreeTransformCacheInfo {
    /// The process-global completed-transformer cache.
    pub structures: RuntimeTreeTransformCacheInfo,
    /// Always empty: the categorical-plan tier was removed in #2014-4
    /// (plans are assembled from the composed-coefficient cache).
    pub plans: RuntimeTreeTransformCacheInfo,
    /// The process-global composed-coefficient cache (per fusion-tree group,
    /// every fusion style), shared by every Runtime since #2014-4.
    pub groups: RuntimeTreeTransformCacheInfo,
}
use tenet_tensors::{
    BoundDynamicFusionMapSpace, DenseTreeTransformOperations, TensorContractFusionExecutionContext,
    TreeTransformOperation,
};

use crate::error::Error;
use crate::typed::ScalarOps;

mod host_scratch;
pub(crate) type CoefficientCtx<D, Key, C> = TensorContractFusionExecutionContext<
    D,
    Key,
    DenseTreeTransformOperations,
    DenseTreeTransformOperations,
    C,
>;
pub type Ctx<D, Key> = CoefficientCtx<D, Key, f64>;

mod coefficient_lane_private {
    pub trait Sealed<C> {}

    impl<T: crate::typed::ScalarOps> Sealed<f64> for T {}
    impl Sealed<num_complex::Complex64> for num_complex::Complex64 {}
}

/// Selects one of the three payload/coefficient pairs owned by a runtime.
///
/// This stays private to the crate: provider scalar compatibility is a static
/// execution detail, not another public context type.
pub(crate) trait MultiplicityFreeCoefficientLane<C: tenet_tensors::DenseBlockScalar>:
    ScalarOps + tenet_tensors::RecouplingCoefficientAction<C> + coefficient_lane_private::Sealed<C>
{
    fn lane(
        context: &mut TensorExecutionContext,
    ) -> Result<&mut CoefficientCtx<Self, RuleIdentity, C>, Error>;
}
/// The supported payload/coefficient execution contexts for one cache-key
/// namespace. Existing operations dispatch on the stored dtype to the
/// real-coefficient lanes.
///
/// The double-precision lanes are built with the runtime; the single-precision
/// ones are built on first use. Every admitted payload dtype needs its own
/// dense executors and contract workspace, so building all four eagerly would
/// charge every runtime for dtypes the program never names. `lane_config`
/// carries what [`Ctxs::with_config`] was given, plus the settings applied
/// afterwards, so a lane created later is configured exactly like an eager one.
pub struct Ctxs<Key: Clone + Eq + Hash + Send + Sync + 'static> {
    pub(crate) f64: Ctx<f64, Key>,
    pub(crate) c64: Ctx<Complex64, Key>,
    pub(crate) f32: Option<Box<Ctx<f32, Key>>>,
    pub(crate) c32: Option<Box<Ctx<Complex32, Key>>>,
    lane_config: LaneConfig,
}

/// What a deferred [`Ctxs`] lane needs to be built exactly like an eager one.
#[derive(Clone)]
struct LaneConfig {
    /// `None` for a `Default` [`Ctxs`], whose lanes own private env-driven
    /// pools rather than a runtime's shared CPU context.
    shared: Option<(
        tenet_dense::SharedCpuContext,
        Option<tenet_dense::CpuBackendKind>,
    )>,
    recoupling_threads: Option<NonZeroUsize>,
}

impl<Key: Clone + Eq + Hash + Send + Sync + 'static> Default for Ctxs<Key> {
    fn default() -> Self {
        Self {
            f64: Ctx::default(),
            c64: Ctx::default(),
            f32: None,
            c32: None,
            lane_config: LaneConfig {
                shared: None,
                recoupling_threads: None,
            },
        }
    }
}

/// Builds one contraction/recoupling backend. A compiled-default `gemm_kind`
/// uses the runtime's shared CPU context; an explicit nondefault kind uses a
/// private provider context. Both remain subject to provider synchronization.
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
            ),
            f32: None,
            c32: None,
            lane_config: LaneConfig {
                shared: Some((ctx.clone(), gemm_kind)),
                recoupling_threads: None,
            },
        })
    }

    /// Builds one deferred lane the way [`Self::with_config`] built the eager
    /// ones, then replays the settings applied since.
    fn make_lane<D: ScalarOps>(config: &LaneConfig) -> Result<Box<Ctx<D, Key>>, Error> {
        let mut lane =
            match &config.shared {
                Some((ctx, gemm_kind)) => Ctx::with_parts(
                    tenet_tensors::TreeTransformExecutionContext::new(make_transform_ops(
                        ctx, *gemm_kind,
                    )?),
                    make_transform_ops(ctx, *gemm_kind)?,
                    <DenseTreeTransformOperations as tenet_tensors::TensorContractBackend<
                        D,
                        f64,
                    >>::Workspace::default(),
                ),
                None => Ctx::default(),
            };
        if let Some(threads) = config.recoupling_threads {
            lane.tree_context_mut()
                .backend_mut()
                .set_recoupling_threads(threads);
        }
        Ok(Box::new(lane))
    }

    pub(crate) fn f32_lane(&mut self) -> Result<&mut Ctx<f32, Key>, Error> {
        let lane = match self.f32.take() {
            Some(lane) => lane,
            None => Self::make_lane(&self.lane_config)?,
        };
        Ok(self.f32.insert(lane))
    }

    pub(crate) fn c32_lane(&mut self) -> Result<&mut Ctx<Complex32, Key>, Error> {
        let lane = match self.c32.take() {
            Some(lane) => lane,
            None => Self::make_lane(&self.lane_config)?,
        };
        Ok(self.c32.insert(lane))
    }

    pub(crate) fn set_recoupling_threads(&mut self, threads: NonZeroUsize) {
        self.lane_config.recoupling_threads = Some(threads);
        self.f64
            .tree_context_mut()
            .backend_mut()
            .set_recoupling_threads(threads);
        self.c64
            .tree_context_mut()
            .backend_mut()
            .set_recoupling_threads(threads);
        if let Some(lane) = self.f32.as_mut() {
            lane.tree_context_mut()
                .backend_mut()
                .set_recoupling_threads(threads);
        }
        if let Some(lane) = self.c32.as_mut() {
            lane.tree_context_mut()
                .backend_mut()
                .set_recoupling_threads(threads);
        }
    }

    #[cfg(test)]
    pub(crate) fn recoupling_threads_are(&mut self, expected: usize) -> bool {
        self.f64
            .tree_context_mut()
            .backend_mut()
            .recoupling_threads()
            .get()
            == expected
            && self
                .c64
                .tree_context_mut()
                .backend_mut()
                .recoupling_threads()
                .get()
                == expected
            && self.f32.as_mut().is_none_or(|lane| {
                lane.tree_context_mut()
                    .backend_mut()
                    .recoupling_threads()
                    .get()
                    == expected
            })
            && self.c32.as_mut().is_none_or(|lane| {
                lane.tree_context_mut()
                    .backend_mut()
                    .recoupling_threads()
                    .get()
                    == expected
            })
    }

    #[cfg(test)]
    pub(crate) fn shares_cpu_context(&mut self, shared: &tenet_dense::SharedCpuContext) -> bool {
        self.f64
            .tree_context_mut()
            .backend_mut()
            .dense()
            .shares_cpu_context(shared)
            && self
                .f64
                .contract_backend()
                .dense()
                .shares_cpu_context(shared)
            && self
                .c64
                .tree_context_mut()
                .backend_mut()
                .dense()
                .shares_cpu_context(shared)
            && self
                .c64
                .contract_backend()
                .dense()
                .shares_cpu_context(shared)
            && self.f32.as_mut().is_none_or(|lane| {
                lane.tree_context_mut()
                    .backend_mut()
                    .dense()
                    .shares_cpu_context(shared)
                    && lane.contract_backend().dense().shares_cpu_context(shared)
            })
            && self.c32.as_mut().is_none_or(|lane| {
                lane.tree_context_mut()
                    .backend_mut()
                    .dense()
                    .shares_cpu_context(shared)
                    && lane.contract_backend().dense().shares_cpu_context(shared)
            })
    }

    /// Builds both deferred lanes, so a test can assert that a lane created
    /// after the runtime carries the same configuration as an eager one.
    #[cfg(test)]
    pub(crate) fn force_single_precision_lanes(&mut self) -> Result<(), Error> {
        self.f32_lane()?;
        self.c32_lane()?;
        Ok(())
    }
}

fn make_complex_multiplicity_free_ctx(
    ctx: &tenet_dense::SharedCpuContext,
    gemm_kind: Option<tenet_dense::CpuBackendKind>,
) -> Result<CoefficientCtx<Complex64, RuleIdentity, Complex64>, Error> {
    Ok(CoefficientCtx::with_parts(
        tenet_tensors::TreeTransformExecutionContext::new(make_transform_ops(ctx, gemm_kind)?),
        make_transform_ops(ctx, gemm_kind)?,
        <DenseTreeTransformOperations as tenet_tensors::TensorContractBackend<
            Complex64,
            Complex64,
        >>::Workspace::default(),
    ))
}

macro_rules! rule_lanes {
    ($callback:ident) => {
        $callback! {
            mf: tenet_core::RuleIdentity,
            generic: tenet_core::RuleIdentity,
        }
    };
}

macro_rules! define_tensor_execution_context {
    ($( $field:ident: $key:ty ),+ $(,)?) => {
        /// Runtime-owned host tensor execution state.
        pub(crate) struct TensorExecutionContext {
            $(pub(crate) $field: Ctxs<$key>,)+
            mf_c64_coeff_c64: CoefficientCtx<Complex64, RuleIdentity, Complex64>,
            #[cfg(all(test, feature = "racah-generated"))]
            generic_lane_uses: usize,
        }

        impl TensorExecutionContext {
            // Why not retain a Runtime here: pooled contexts live inside that
            // Runtime, so the back-reference would form an Arc cycle.
            pub(crate) fn for_config(config: &RuntimeExecutionConfig) -> Result<Self, Error> {
                let mut context = Self {
                    $($field: Ctxs::with_config(&config.shared_ctx, config.gemm_kind)?,)+
                    mf_c64_coeff_c64: make_complex_multiplicity_free_ctx(
                        &config.shared_ctx,
                        config.gemm_kind,
                    )?,
                    #[cfg(all(test, feature = "racah-generated"))]
                    generic_lane_uses: 0,
                };
                if let Some(threads) = config.recoupling_threads {
                    context.set_recoupling_threads(threads);
                }
                Ok(context)
            }

            fn set_recoupling_threads(&mut self, threads: NonZeroUsize) {
                $(self.$field.set_recoupling_threads(threads);)+
                self.mf_c64_coeff_c64
                    .tree_context_mut()
                    .backend_mut()
                    .set_recoupling_threads(threads);
            }

            /// Returns the multiplicity-free lane matching scalar `D`.
            ///
            /// Only this lane is exposed to the typed facade; Generic-fusion
            /// execution remains behind its provider-specific boundary.
            pub(crate) fn multiplicity_free_lane<D: ScalarOps>(
                &mut self,
            ) -> Result<&mut Ctx<D, tenet_core::RuleIdentity>, Error> {
                D::ctx_of(&mut self.mf)
            }

            /// Returns the statically compatible complex-payload,
            /// complex-coefficient multiplicity-free lane.
            #[allow(
                dead_code,
                reason = "runtime foundation for the #592 typed-facade follow-up"
            )]
            pub(crate) fn complex_multiplicity_free_lane(
                &mut self,
            ) -> &mut CoefficientCtx<Complex64, RuleIdentity, Complex64> {
                &mut self.mf_c64_coeff_c64
            }

            pub(crate) fn generic_lane<D: ScalarOps>(
                &mut self,
            ) -> Result<&mut Ctx<D, tenet_core::RuleIdentity>, Error> {
                #[cfg(all(test, feature = "racah-generated"))]
                {
                    self.generic_lane_uses += 1;
                }
                D::ctx_of(&mut self.generic)
            }

            #[cfg(all(test, feature = "racah-generated"))]
            pub(crate) fn generic_lane_uses(&self) -> usize {
                self.generic_lane_uses
            }

            #[cfg(test)]
            pub(crate) fn recoupling_threads_are(&mut self, expected: usize) -> bool {
                true $(&& self.$field.recoupling_threads_are(expected))+
                    && self
                        .mf_c64_coeff_c64
                        .tree_context_mut()
                        .backend_mut()
                        .recoupling_threads()
                        .get()
                        == expected
            }

            #[cfg(test)]
            pub(crate) fn shares_cpu_context(
                &mut self,
                shared: &tenet_dense::SharedCpuContext,
            ) -> bool {
                true $(&& self.$field.shares_cpu_context(shared))+
                    && self
                        .mf_c64_coeff_c64
                        .tree_context_mut()
                        .backend_mut()
                        .dense()
                        .shares_cpu_context(shared)
                    && self
                        .mf_c64_coeff_c64
                        .contract_backend()
                        .dense()
                        .shares_cpu_context(shared)
            }

        }
    };
}

impl<D: ScalarOps> MultiplicityFreeCoefficientLane<f64> for D {
    fn lane(
        context: &mut TensorExecutionContext,
    ) -> Result<&mut CoefficientCtx<Self, RuleIdentity, f64>, Error> {
        Self::ctx_of(&mut context.mf)
    }
}

impl MultiplicityFreeCoefficientLane<Complex64> for Complex64 {
    fn lane(
        context: &mut TensorExecutionContext,
    ) -> Result<&mut CoefficientCtx<Self, RuleIdentity, Complex64>, Error> {
        Ok(&mut context.mf_c64_coeff_c64)
    }
}

rule_lanes!(define_tensor_execution_context);

macro_rules! define_runtime_state {
    ($( $field:ident: $key:ty ),+ $(,)?) => {
        /// Expert-layer execution contexts for the multiplicity-free and
        /// Generic-fusion namespaces, plus the rule-independent dense executor.
        ///
        /// CPU state only: a CUDA device context lives behind its own
        /// device lease on `RuntimeInner`, so a device operation never holds
        /// this lock.
        pub(crate) struct RuntimeState {
            $(pub(crate) $field: Ctxs<$key>,)+
            mf_c64_coeff_c64: CoefficientCtx<Complex64, RuleIdentity, Complex64>,
            pub(crate) dense: Box<dyn tenet_dense::DenseExecutor + Send>,
        }

        impl RuntimeState {
            // Why not use `Ctxs::default()`: each default context creates a
            // private env-driven pool instead of sharing the runtime CPU pool.
            fn with_config(
                dense: Box<dyn tenet_dense::DenseExecutor + Send>,
                ctx: &tenet_dense::SharedCpuContext,
                gemm_kind: Option<tenet_dense::CpuBackendKind>,
            ) -> Result<Self, Error> {
                Ok(Self {
                    $($field: Ctxs::with_config(ctx, gemm_kind)?,)+
                    mf_c64_coeff_c64: make_complex_multiplicity_free_ctx(ctx, gemm_kind)?,
                    dense,
                })
            }

            fn set_recoupling_threads(&mut self, threads: NonZeroUsize) {
                $(self.$field.set_recoupling_threads(threads);)+
                self.mf_c64_coeff_c64
                    .tree_context_mut()
                    .backend_mut()
                    .set_recoupling_threads(threads);
            }

            #[cfg(test)]
            pub(crate) fn recoupling_threads_are(&mut self, expected: usize) -> bool {
                true $(&& self.$field.recoupling_threads_are(expected))+
                    && self
                        .mf_c64_coeff_c64
                        .tree_context_mut()
                        .backend_mut()
                        .recoupling_threads()
                        .get()
                        == expected
            }

            #[cfg(test)]
            pub(crate) fn shares_cpu_context(
                &mut self,
                shared: &tenet_dense::SharedCpuContext,
            ) -> bool {
                true $(&& self.$field.shares_cpu_context(shared))+
                    && self
                        .mf_c64_coeff_c64
                        .tree_context_mut()
                        .backend_mut()
                        .dense()
                        .shares_cpu_context(shared)
                    && self
                        .mf_c64_coeff_c64
                        .contract_backend()
                        .dense()
                        .shares_cpu_context(shared)
            }
        }
    };
}

rule_lanes!(define_runtime_state);

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

/// The device-local state one CUDA device operation may need: the dense
/// context every device kernel is submitted through, the tree-transform
/// executor that holds the per-structure device state (uploaded coefficient
/// vectors, pack/scatter workspaces), and the general contraction's operand
/// and core-destination scratch.
///
/// The executor lives here rather than beside the host caches because a replay
/// needs `&mut CudaDenseContext` and `&mut CudaTreeTransformExecutor` at the
/// same instant, and because its entries are keyed by the device context's own
/// identity: they are meaningless without it. Putting both behind one mutex is
/// what lets a device transform take exactly one lock.
#[cfg(feature = "cuda")]
pub(crate) struct CudaDeviceState {
    dense: tenet_dense::CudaDenseContext,
    tree_transform: tenet_operations::CudaTreeTransformExecutor,
    /// Grow-only execution scratch, bounded by the largest contraction run
    /// and observable through [`Runtime::cuda_contract_scratch_bytes`]. Like
    /// the executor's workspaces it is not charged to
    /// `tenet-network`'s `PlanCacheConfig::workspace_budget_bytes`, which admits idle network
    /// workspaces and has no eviction protocol for a Runtime singleton.
    contract_scratch: tenet_tensors::CudaContractScratch,
}

/// A Runtime's device state and the lock of the device it lives on.
///
/// The per-Runtime mutex is kept although the device lock already excludes
/// every other lease on it: it is what hands out `&mut CudaDeviceState`
/// safely, and it is never contended, so it costs one uncontended atomic
/// pair. Replacing it by an `UnsafeCell` guarded by the device lock would buy
/// nothing measurable for an `unsafe` invariant.
#[cfg(feature = "cuda")]
struct CudaHome {
    device_lock: &'static Mutex<()>,
    state: Mutex<CudaDeviceState>,
}

/// The process-wide lock of CUDA device `ordinal`, shared by every Runtime on
/// that device.
///
/// Why a lock across Runtimes (#1384): CubeCL's client is process-wide per
/// device and records a binding's stream cursor only when it is bound, not
/// when a later kernel writes it (tensor4all/cubecl#16). A stale read of an
/// output `O` needs some other submission that syncs a stream past `O`'s bind
/// cursor between that bind and `O`'s last write. For a fresh output — bound
/// and fully written inside one lease, as in every operation that returns a
/// new tensor — this lock excludes every other lease on the device, so no
/// TeNeT submission can fall in that window.
/// Any later reader's sync to the writer's stream then either happened before
/// the writer's lease (its synced cursor is below `O`'s bind cursor, so it
/// waits on a fresh event) or after it (the event already covers every write
/// into `O`). No host synchronization is added: the lock orders enqueue only.
/// A buffer bound in one lease and written in a later one
/// (`*_into` destinations, `CudaContractScratch`, pooled
/// network intermediates) spans two leases and is ordered instead by the
/// single CubeCL stream `CudaDenseContext::new` pins (#1391).
///
/// One leaked entry per ordinal that built a device context, so the registry
/// is bounded by the device count. The lock guards no data, so a poison left
/// by a panicking holder is ignored.
#[cfg(feature = "cuda")]
fn cuda_device_lock(ordinal: usize) -> &'static Mutex<()> {
    static LOCKS: Mutex<std::collections::BTreeMap<usize, &'static Mutex<()>>> =
        Mutex::new(std::collections::BTreeMap::new());
    LOCKS
        .lock()
        .unwrap_or_else(PoisonError::into_inner)
        .entry(ordinal)
        .or_insert_with(|| Box::leak(Box::new(Mutex::new(()))))
}

#[cfg(feature = "cuda")]
fn lock_device(lock: &'static Mutex<()>) -> MutexGuard<'static, ()> {
    lock.lock().unwrap_or_else(PoisonError::into_inner)
}

/// RAII lease of this runtime's single CUDA context, held only for the device
/// portion of one device operation. A device context cannot be pooled: device
/// tensors are bound to one backend instance, so the guard is the lease.
///
/// It dereferences to the [`tenet_dense::CudaDenseContext`], so every device
/// operation that needs nothing else reads exactly as it did before the
/// tree-transform executor moved in beside it; a transform takes both halves
/// through [`Self::split`].
///
/// It holds the device's process-wide lock ([`cuda_device_lock`]) first and
/// this Runtime's state second, so every lease on one device, from any
/// Runtime, excludes every other. That protects what is bound and fully
/// written within one lease; a buffer written again in a later lease is not
/// covered (see [`cuda_device_lock`]). A host sync under the lease stalls
/// every Runtime on the device.
///
/// Nothing reached while this lease is held may lease again — on this Runtime
/// or on any other Runtime of the same device: neither mutex is re-entrant.
/// Host-side planning — structure compilation, cache lookup, space
/// derivation — is finished, and the host context lease dropped, before this
/// one is taken, and the code run under it only sees the dense context and
/// executor, never a `Runtime`.
#[cfg(feature = "cuda")]
pub(crate) struct CudaLease<'a> {
    // Declared first so it is released before the device lock.
    state: MutexGuard<'a, CudaDeviceState>,
    _device: MutexGuard<'static, ()>,
}

#[cfg(feature = "cuda")]
impl CudaLease<'_> {
    /// Borrows the dense context and the tree-transform executor at once, which
    /// a replay needs because the executor submits its work through the
    /// context.
    pub(crate) fn split(
        &mut self,
    ) -> (
        &mut tenet_dense::CudaDenseContext,
        &mut tenet_operations::CudaTreeTransformExecutor,
    ) {
        let state = &mut *self.state;
        (&mut state.dense, &mut state.tree_transform)
    }

    /// [`Self::split`] plus the contraction scratch, which a general device
    /// contraction needs beside the transform executor it replays through.
    pub(crate) fn split_contract(
        &mut self,
    ) -> (
        &mut tenet_dense::CudaDenseContext,
        &mut tenet_operations::CudaTreeTransformExecutor,
        &mut tenet_tensors::CudaContractScratch,
    ) {
        let state = &mut *self.state;
        (
            &mut state.dense,
            &mut state.tree_transform,
            &mut state.contract_scratch,
        )
    }
}

#[cfg(feature = "cuda")]
impl std::ops::Deref for CudaLease<'_> {
    type Target = tenet_dense::CudaDenseContext;

    fn deref(&self) -> &Self::Target {
        &self.state.dense
    }
}

#[cfg(feature = "cuda")]
impl std::ops::DerefMut for CudaLease<'_> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.state.dense
    }
}

/// Device bytes this Runtime's tree-transform executor and its device context
/// hold for reuse, each counted once.
///
/// Read-only: nothing here is a knob. `executor_bytes` is what the executor
/// retains (uploaded coefficient and recoupling vectors plus the pack/scatter
/// workspaces) and is released by the deprecated
/// [`Runtime::clear_tree_transform_cache`]. `context_scalar_operand_bytes` is
/// the device context's own ones and zero templates, which many device
/// operations share and which the executor does not own; the two are reported
/// separately so neither is charged twice. Neither is charged to
/// `tenet-network`'s `PlanCacheConfig::workspace_budget_bytes`, which admits idle *network*
/// workspaces and has no eviction protocol for a Runtime singleton.
#[cfg(feature = "cuda")]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct CudaTreeTransformStats {
    /// Structures with device state currently prepared.
    pub prepared_structures: usize,
    /// Device bytes the executor retains: coefficient vectors and workspaces.
    pub executor_bytes: usize,
    /// Of `executor_bytes`, what the pack/scatter workspaces hold.
    pub workspace_bytes: usize,
    /// Device bytes the shared context scalar operands hold, owned by the
    /// device context rather than by the executor.
    pub context_scalar_operand_bytes: usize,
    /// Distinct cuTENSOR operand signatures the prepared structures submit.
    pub required_plan_entries: usize,
}

/// The one error a device operation reports when its runtime has no device.
#[cfg(feature = "cuda")]
fn missing_cuda_device() -> Error {
    Error::InvalidArgument(
        "this runtime was built without a CUDA device; use \
         Runtime::builder().cuda(device)"
            .to_string(),
    )
}

/// A runtime's Host pool entry on the calling thread (see
/// [`Runtime::enter_host_pool`]); the pool stays entered until it drops.
#[must_use = "the pool is entered only while the entry lives"]
pub(crate) struct HostPoolEntry {
    _guard: tenet_operations::host_pool::HostPoolGuard,
    #[cfg(test)]
    _probe: HostPoolDepth,
}

#[cfg(test)]
thread_local! {
    static HOST_POOL_DEPTH: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

/// Test mirror of the entered runtime pools on this thread, so a probe can
/// ask whether a data-movement site runs inside one.
#[cfg(test)]
struct HostPoolDepth;

#[cfg(test)]
impl HostPoolDepth {
    fn enter() -> Self {
        HOST_POOL_DEPTH.set(HOST_POOL_DEPTH.get() + 1);
        Self
    }
}

#[cfg(test)]
impl Drop for HostPoolDepth {
    fn drop(&mut self) {
        HOST_POOL_DEPTH.set(HOST_POOL_DEPTH.get() - 1);
    }
}

/// Whether a runtime Host pool is entered on this thread.
#[cfg(test)]
pub(crate) fn host_pool_entered() -> bool {
    HOST_POOL_DEPTH.get() > 0
}

#[cfg(test)]
thread_local! {
    /// Dense leases taken on this thread: the #1996 lease-count probe.
    pub(crate) static DENSE_LEASES: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

/// RAII lease of a pooled execution context (#155). Returns it to the
/// pool on drop; on panic it is dropped instead of returned (quarantine —
/// mirrors `tenet_network`'s `WorkspaceLease`). While it lives, the runtime's
/// CPU pool is the calling thread's Host pool, so the operation's replay,
/// plan compile and strided regions run on it.
pub(crate) struct ContextLease<'a> {
    pool: &'a Mutex<Vec<PooledContext>>,
    max_idle: usize,
    context: Option<PooledContext>,
    _host_pool: HostPoolEntry,
}

impl ContextLease<'_> {
    pub(crate) fn context(&mut self) -> &mut TensorExecutionContext {
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
/// configs, or the `state` lock for a non-mintable injected executor. Like
/// [`ContextLease`], it enters the runtime's CPU pool while it lives.
pub(crate) enum DenseLease<'a> {
    Pooled {
        pool: &'a Mutex<Vec<Box<dyn tenet_dense::DenseExecutor + Send>>>,
        max_idle: usize,
        executor: Option<Box<dyn tenet_dense::DenseExecutor + Send>>,
        _host_pool: HostPoolEntry,
    },
    Locked {
        state: MutexGuard<'a, RuntimeState>,
        _host_pool: HostPoolEntry,
    },
}

impl DenseLease<'_> {
    pub(crate) fn dense(&mut self) -> &mut (dyn tenet_dense::DenseExecutor + Send) {
        match self {
            DenseLease::Pooled { executor, .. } => &mut **executor
                .as_mut()
                .expect("dense lease always owns an executor"),
            DenseLease::Locked { state, .. } => &mut *state.dense,
        }
    }
}

impl Drop for DenseLease<'_> {
    fn drop(&mut self) {
        if let DenseLease::Pooled {
            pool,
            max_idle,
            executor,
            ..
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

    /// Deprecated (one release): use [`crate::cache::stats`].
    ///
    /// `structures` reports the process-global completed-transformer cache
    /// and `groups` the process-global composed-coefficient cache, both
    /// shared by every Runtime: `entry_capacity` = `usize::MAX` (byte-bounded
    /// only), `misses` = builds offered for admission, `admission_bypasses` =
    /// oversize rejections. `plans` is always empty: that tier was removed
    /// in #2014-4. The caches are sampled one after another, not atomically.
    #[deprecated(note = "use `tenet::cache::stats`; see the method documentation")]
    pub fn tree_transform_cache_info(&self) -> TreeTransformCacheInfo {
        let global = |kind| {
            RuntimeTreeTransformCacheInfo::from_structure_cache(tenet_core::structure_cache_info(
                kind,
            ))
        };
        TreeTransformCacheInfo {
            structures: global(tenet_core::StructureCacheKind::CompletedTreeTransformer),
            plans: RuntimeTreeTransformCacheInfo::default(),
            groups: global(tenet_core::StructureCacheKind::TreeTransformCoefficients),
        }
    }

    /// Deprecated (one release): clears the process-global structure caches,
    /// composed coefficients included, for **every** Runtime
    /// ([`crate::cache::clear`]), then this Runtime's prepared device
    /// transform state and contraction scratch. The steps run one after
    /// another, never nested.
    ///
    /// The device state is dropped under the maintenance lease, after the
    /// host clears have returned, so this never inverts a lock order; on a
    /// Runtime with a device it blocks behind a device operation in
    /// progress. A device lock poisoned by an earlier panic is recovered
    /// rather than propagated: dropping prepared device state is always safe.
    #[deprecated(
        note = "use `tenet::cache::clear`; this also clears every Runtime's structure caches"
    )]
    pub fn clear_tree_transform_cache(&self) {
        crate::cache::clear();
        #[cfg(feature = "cuda")]
        if let Some(mut lease) = self.lease_cuda_for_maintenance() {
            let (dense, executor, scratch) = lease.split_contract();
            executor.clear(dense);
            scratch.clear();
        }
    }

    /// [`Self::exact_layout_tree_pair_hit_of`] of the real-coefficient lane.
    #[cfg(feature = "cuda")]
    pub(crate) fn exact_layout_tree_pair_hit<R>(
        rule: &RuleIdentity,
        operation: tenet_tensors::TreeTransformOperationView<'_>,
        source: &BoundDynamicFusionMapSpace<R>,
        destination: &BoundDynamicFusionMapSpace<R>,
    ) -> Option<tenet_tensors::TreeTransformStructure<f64>> {
        Self::exact_layout_tree_pair_hit_of(rule, operation, source, destination)
    }

    /// [`Self::admit_exact_tree_pair_layout_of`] of the real-coefficient lane.
    #[cfg(feature = "cuda")]
    pub(crate) fn admit_exact_tree_pair_layout<R>(
        rule: &RuleIdentity,
        operation: &TreeTransformOperation,
        source: &BoundDynamicFusionMapSpace<R>,
        destination: &BoundDynamicFusionMapSpace<R>,
    ) -> bool {
        Self::admit_exact_tree_pair_layout_of::<R, f64>(rule, operation, source, destination)
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
        DENSE_LEASES.set(DENSE_LEASES.get() + 1);
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
    /// per payload dtype until the deprecated
    /// [`Self::clear_tree_transform_cache`]. It is
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

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};

    use super::*;
    use crate::typed::{GradedSpace, TensorMap};
    use tenet_core::{
        BraidingStyleKind, CheckedFusionAlgebra, FibonacciFusionRule, FibonacciSector,
        FusionAlgebraError, FusionProductSpace, FusionRule, FusionStyleKind, FusionTreeHomSpace,
        MultiplicityFreeFusionRule, MultiplicityFreeFusionSymbols, MultiplicityFreeRigidSymbols,
        SU2FusionRule, SU2Irrep, SectorCodec, SectorId, SectorLeg, SectorVec,
    };

    #[test]
    fn network_destination_dual_stays_inside_complete_publication_epoch() {
        const FILTER: &str =
            "runtime::tests::network_destination_dual_stays_inside_complete_publication_epoch";
        if std::env::var_os("TENET_NETWORK_PRODUCER_RESET").is_none() {
            let output = std::process::Command::new(std::env::current_exe().unwrap())
                .args(["--exact", FILTER])
                .env("TENET_NETWORK_PRODUCER_RESET", "1")
                .output()
                .unwrap();
            let stdout = String::from_utf8_lossy(&output.stdout);
            assert!(
                output.status.success() && stdout.contains("test result: ok. 1 passed; 0 failed;"),
                "{stdout} {}",
                String::from_utf8_lossy(&output.stderr)
            );
            return;
        }
        let runtime = Runtime::builder().dense_threads(1).build().unwrap();
        let provider = Arc::new(NetworkResetU1 {
            reset_on_dual: std::sync::atomic::AtomicBool::new(false),
        });
        let leg = GradedSpace::try_new(Arc::clone(&provider), [(tenet_core::U1Irrep::new(1), 2)])
            .unwrap();
        let source: TensorMap<_, f64> =
            TensorMap::from_subblock_fn(&runtime, [&leg], [&leg], |_, _| 1.0).unwrap();
        tenet_core::clear_structure_caches();
        provider.reset_on_dual.store(true, Ordering::SeqCst);
        let stale = source
            .network_zeros_from_effective_legs(
                std::slice::from_ref(&leg),
                std::slice::from_ref(&leg),
            )
            .unwrap();
        assert!(!provider.reset_on_dual.load(Ordering::SeqCst));
        assert_eq!(
            tenet_core::structure_cache_info(tenet_core::StructureCacheKind::DegeneracyStructure)
                .entries(),
            0
        );
        let fresh = source
            .network_zeros_from_effective_legs(
                std::slice::from_ref(&leg),
                std::slice::from_ref(&leg),
            )
            .unwrap();
        assert_eq!(stale.dense_data().unwrap(), fresh.dense_data().unwrap());
        assert_eq!(
            tenet_core::structure_cache_info(tenet_core::StructureCacheKind::DegeneracyStructure)
                .entries(),
            1
        );
    }

    struct NetworkResetU1 {
        reset_on_dual: std::sync::atomic::AtomicBool,
    }
    impl FusionRule for NetworkResetU1 {
        fn rule_identity(&self) -> RuleIdentity {
            RuleIdentity::of_type::<Self>()
        }
        fn fusion_style(&self) -> FusionStyleKind {
            tenet_core::U1FusionRule.fusion_style()
        }
        fn braiding_style(&self) -> BraidingStyleKind {
            tenet_core::U1FusionRule.braiding_style()
        }
        fn vacuum(&self) -> SectorId {
            tenet_core::U1FusionRule.vacuum()
        }
        fn dual(&self, s: SectorId) -> SectorId {
            tenet_core::U1FusionRule.dual(s)
        }
        fn fusion_channels(&self, a: SectorId, b: SectorId) -> SectorVec {
            tenet_core::U1FusionRule.fusion_channels(a, b)
        }
    }
    impl MultiplicityFreeFusionRule for NetworkResetU1 {}
    impl CheckedFusionAlgebra for NetworkResetU1 {
        fn try_dual_sector(&self, s: SectorId) -> Result<SectorId, FusionAlgebraError> {
            if self.reset_on_dual.swap(false, Ordering::SeqCst) {
                tenet_core::clear_structure_caches();
            }
            tenet_core::U1FusionRule.try_dual_sector(s)
        }
        fn try_fusion_channels(
            &self,
            a: SectorId,
            b: SectorId,
        ) -> Result<SectorVec, FusionAlgebraError> {
            tenet_core::U1FusionRule.try_fusion_channels(a, b)
        }
        fn try_nsymbol(
            &self,
            a: SectorId,
            b: SectorId,
            c: SectorId,
        ) -> Result<usize, FusionAlgebraError> {
            tenet_core::U1FusionRule.try_nsymbol(a, b, c)
        }
    }
    impl SectorCodec for NetworkResetU1 {
        type Sector = tenet_core::U1Irrep;
        fn encode_sector(&self, s: &Self::Sector) -> Result<SectorId, FusionAlgebraError> {
            tenet_core::U1FusionRule.encode_sector(s)
        }
        fn decode_sector(&self, s: SectorId) -> Result<Self::Sector, FusionAlgebraError> {
            tenet_core::U1FusionRule.decode_sector(s)
        }
    }
    impl MultiplicityFreeFusionSymbols for NetworkResetU1 {
        type Scalar = f64;
        fn f_symbol_scalar(
            &self,
            a: SectorId,
            b: SectorId,
            c: SectorId,
            d: SectorId,
            e: SectorId,
            f: SectorId,
        ) -> f64 {
            tenet_core::U1FusionRule.f_symbol_scalar(a, b, c, d, e, f)
        }
        fn r_symbol_scalar(&self, a: SectorId, b: SectorId, c: SectorId) -> f64 {
            tenet_core::U1FusionRule.r_symbol_scalar(a, b, c)
        }
    }
    impl MultiplicityFreeRigidSymbols for NetworkResetU1 {
        fn dim_scalar(&self, s: SectorId) -> f64 {
            tenet_core::U1FusionRule.dim_scalar(s)
        }
        fn inv_dim_scalar(&self, s: SectorId) -> f64 {
            tenet_core::U1FusionRule.inv_dim_scalar(s)
        }
        fn sqrt_dim_scalar(&self, s: SectorId) -> f64 {
            tenet_core::U1FusionRule.sqrt_dim_scalar(s)
        }
        fn inv_sqrt_dim_scalar(&self, s: SectorId) -> f64 {
            tenet_core::U1FusionRule.inv_sqrt_dim_scalar(s)
        }
        fn twist_scalar(&self, s: SectorId) -> f64 {
            tenet_core::U1FusionRule.twist_scalar(s)
        }
        fn frobenius_schur_phase_scalar(&self, s: SectorId) -> f64 {
            tenet_core::U1FusionRule.frobenius_schur_phase_scalar(s)
        }
    }

    #[derive(Clone)]
    struct CountingFibonacci {
        structural_calls: Arc<AtomicUsize>,
        layout_calls: Arc<AtomicUsize>,
        malformed_channels: bool,
    }

    impl CountingFibonacci {
        fn new() -> Self {
            Self {
                structural_calls: Arc::new(AtomicUsize::new(0)),
                layout_calls: Arc::new(AtomicUsize::new(0)),
                malformed_channels: false,
            }
        }

        fn structural_calls(&self) -> usize {
            self.structural_calls.load(Ordering::Relaxed)
        }

        fn layout_calls(&self) -> usize {
            self.layout_calls.load(Ordering::Relaxed)
        }
    }

    impl CheckedFusionAlgebra for CountingFibonacci {
        fn try_dual_sector(&self, sector: SectorId) -> Result<SectorId, FusionAlgebraError> {
            self.layout_calls.fetch_add(1, Ordering::Relaxed);
            FibonacciFusionRule.try_dual_sector(sector)
        }

        fn try_fusion_channels(
            &self,
            left: SectorId,
            right: SectorId,
        ) -> Result<SectorVec, FusionAlgebraError> {
            self.layout_calls.fetch_add(1, Ordering::Relaxed);
            if self.malformed_channels && left == SectorId::new(1) && right == SectorId::new(1) {
                return Ok([SectorId::new(0), SectorId::new(2)].into_iter().collect());
            }
            FibonacciFusionRule.try_fusion_channels(left, right)
        }

        fn try_nsymbol(
            &self,
            left: SectorId,
            right: SectorId,
            coupled: SectorId,
        ) -> Result<usize, FusionAlgebraError> {
            self.layout_calls.fetch_add(1, Ordering::Relaxed);
            FibonacciFusionRule.try_nsymbol(left, right, coupled)
        }
    }

    impl SectorCodec for CountingFibonacci {
        type Sector = FibonacciSector;

        fn encode_sector(&self, value: &Self::Sector) -> Result<SectorId, FusionAlgebraError> {
            FibonacciFusionRule.encode_sector(value)
        }

        fn decode_sector(&self, id: SectorId) -> Result<Self::Sector, FusionAlgebraError> {
            FibonacciFusionRule.decode_sector(id)
        }
    }

    impl FusionRule for CountingFibonacci {
        fn rule_identity(&self) -> RuleIdentity {
            FibonacciFusionRule.rule_identity()
        }

        fn fusion_style(&self) -> FusionStyleKind {
            FibonacciFusionRule.fusion_style()
        }

        fn braiding_style(&self) -> BraidingStyleKind {
            FibonacciFusionRule.braiding_style()
        }

        fn vacuum(&self) -> SectorId {
            FibonacciFusionRule.vacuum()
        }

        fn supports_unitary_braid_dagger(&self) -> bool {
            FibonacciFusionRule.supports_unitary_braid_dagger()
        }

        fn dual(&self, sector: SectorId) -> SectorId {
            self.layout_calls.fetch_add(1, Ordering::Relaxed);
            FibonacciFusionRule.dual(sector)
        }

        fn fusion_channels(&self, left: SectorId, right: SectorId) -> SectorVec {
            self.layout_calls.fetch_add(1, Ordering::Relaxed);
            FibonacciFusionRule.fusion_channels(left, right)
        }

        fn nsymbol(&self, left: SectorId, right: SectorId, coupled: SectorId) -> usize {
            self.layout_calls.fetch_add(1, Ordering::Relaxed);
            FibonacciFusionRule.nsymbol(left, right, coupled)
        }
    }

    impl MultiplicityFreeFusionRule for CountingFibonacci {}

    impl MultiplicityFreeFusionSymbols for CountingFibonacci {
        type Scalar = Complex64;

        fn f_symbol_scalar(
            &self,
            left: SectorId,
            middle: SectorId,
            right: SectorId,
            coupled: SectorId,
            left_coupled: SectorId,
            right_coupled: SectorId,
        ) -> Self::Scalar {
            self.structural_calls.fetch_add(1, Ordering::Relaxed);
            FibonacciFusionRule.f_symbol_scalar(
                left,
                middle,
                right,
                coupled,
                left_coupled,
                right_coupled,
            )
        }

        fn r_symbol_scalar(
            &self,
            left: SectorId,
            right: SectorId,
            coupled: SectorId,
        ) -> Self::Scalar {
            self.structural_calls.fetch_add(1, Ordering::Relaxed);
            FibonacciFusionRule.r_symbol_scalar(left, right, coupled)
        }
    }

    impl MultiplicityFreeRigidSymbols for CountingFibonacci {
        fn dim_scalar(&self, sector: SectorId) -> Self::Scalar {
            self.structural_calls.fetch_add(1, Ordering::Relaxed);
            FibonacciFusionRule.dim_scalar(sector)
        }

        fn inv_dim_scalar(&self, sector: SectorId) -> Self::Scalar {
            self.structural_calls.fetch_add(1, Ordering::Relaxed);
            FibonacciFusionRule.inv_dim_scalar(sector)
        }

        fn sqrt_dim_scalar(&self, sector: SectorId) -> Self::Scalar {
            self.structural_calls.fetch_add(1, Ordering::Relaxed);
            FibonacciFusionRule.sqrt_dim_scalar(sector)
        }

        fn inv_sqrt_dim_scalar(&self, sector: SectorId) -> Self::Scalar {
            self.structural_calls.fetch_add(1, Ordering::Relaxed);
            FibonacciFusionRule.inv_sqrt_dim_scalar(sector)
        }

        fn twist_scalar(&self, sector: SectorId) -> Self::Scalar {
            self.structural_calls.fetch_add(1, Ordering::Relaxed);
            FibonacciFusionRule.twist_scalar(sector)
        }

        fn frobenius_schur_phase_scalar(&self, sector: SectorId) -> Self::Scalar {
            self.structural_calls.fetch_add(1, Ordering::Relaxed);
            FibonacciFusionRule.frobenius_schur_phase_scalar(sector)
        }
    }

    // What: the context pool must hold pointer-sized entries. Pooling the
    // complete multi-lane execution state by value made Vec pop/push memmoves
    // ~70% of a small standalone op's cost (issue #228). Reverting
    // `PooledContext` to a by-value alias fails here instead of silently
    // reintroducing that tax.
    #[test]
    fn pooled_context_is_pointer_sized() {
        assert_eq!(
            std::mem::size_of::<PooledContext>(),
            std::mem::size_of::<usize>()
        );
    }

    // What: every Runtime on one ordinal must share one device lock (#1384),
    // and distinct ordinals must not serialize on each other.
    #[cfg(feature = "cuda")]
    #[test]
    fn cuda_device_lock_is_one_per_ordinal() {
        assert!(std::ptr::eq(cuda_device_lock(7), cuda_device_lock(7)));
        assert!(!std::ptr::eq(cuda_device_lock(7), cuda_device_lock(8)));
    }

    // The default-feature graph includes faer. This control verifies that its
    // explicit provider path constructs; the adapter's private route test pins
    // the unset compiled-default selection separately.
    #[cfg(feature = "cpu-faer")]
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

    #[test]
    fn runtime_and_leased_contexts_share_one_cpu_context() {
        let runtime = Runtime::builder().build().expect("runtime");
        let shared = runtime.execution_config().shared_ctx.clone();
        {
            let mut state = runtime.lock();
            assert!(state.shares_cpu_context(&shared));
        }

        let mut lease = runtime.lease_context().expect("lease");
        assert!(lease.context().shares_cpu_context(&shared));
        let mut network_context =
            TensorExecutionContext::for_config(runtime.execution_config()).expect("context");
        assert!(network_context.shares_cpu_context(&shared));
    }

    // What: a lane built after the runtime carries the configuration the
    // eager lanes were given. Deferring construction is the whole reason a
    // program that never touches single precision pays nothing for it, and the
    // deferral is only sound if the late lane is indistinguishable.
    #[test]
    fn lazily_built_single_precision_lanes_inherit_the_runtime_configuration() {
        let runtime = Runtime::builder()
            .recoupling_threads(3)
            .build()
            .expect("runtime");
        let shared = runtime.execution_config().shared_ctx.clone();

        let mut state = runtime.lock();
        state.mf.force_single_precision_lanes().expect("mf lanes");
        state
            .generic
            .force_single_precision_lanes()
            .expect("generic lanes");
        assert!(state.recoupling_threads_are(3));
        assert!(state.shares_cpu_context(&shared));
        drop(state);

        let mut context =
            TensorExecutionContext::for_config(runtime.execution_config()).expect("context");
        context.mf.force_single_precision_lanes().expect("mf lanes");
        context
            .generic
            .force_single_precision_lanes()
            .expect("generic lanes");
        assert!(context.recoupling_threads_are(3));
        assert!(context.shares_cpu_context(&shared));
    }

    #[test]
    fn runtime_builder_recoupling_threads_reach_every_runtime_and_context_lane() {
        fn assert_runtime_and_context(runtime: &Runtime, expected: usize) {
            {
                let mut state = runtime.lock();
                assert!(state.recoupling_threads_are(expected));
            }

            let mut context =
                TensorExecutionContext::for_config(runtime.execution_config()).expect("context");
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

    #[test]
    fn fibonacci_complex_lane_reuses_nonreal_structural_plan() {
        if crate::test_cache::run_isolated_or_return(
            "TENET_FIBONACCI_LANE_ISOLATED",
            "runtime::tests::fibonacci_complex_lane_reuses_nonreal_structural_plan",
        ) {
            return;
        }
        // What: the private Complex64<-Complex64 Runtime lane compiles one
        // Fibonacci braid, then warm replay avoids every structural-symbol
        // query. Layout queries are counted separately and are not claimed to
        // disappear.
        let runtime = Runtime::builder().build().expect("runtime");
        let rule = Arc::new(CountingFibonacci::new());
        let tau = SectorId::new(1);
        let leg = || SectorLeg::new([(tau, 1)], false);
        let homspace = FusionTreeHomSpace::new(
            FusionProductSpace::new([leg(), leg()]),
            FusionProductSpace::new([]),
        );
        let source = BoundDynamicFusionMapSpace::from_final_homspace_multiplicity_free(
            Arc::clone(&rule),
            homspace,
        )
        .expect("Fibonacci source");
        let operation = TreeTransformOperation::braid([1, 0], [], [0, 1], []);
        let destination = source
            .transformed_multiplicity_free(&operation)
            .expect("Fibonacci braid destination");
        rule.structural_calls.store(0, Ordering::Relaxed);
        rule.layout_calls.store(0, Ordering::Relaxed);
        crate::cache::clear();

        let source_structure = Arc::clone(source.space().structure());
        let destination_structure = Arc::clone(destination.space().structure());
        let source_data = vec![Complex64::new(1.0, 0.0); source.space().required_len().unwrap()];
        let mut destination_data =
            vec![Complex64::new(0.0, 0.0); destination.space().required_len().unwrap()];
        let mut lease = runtime.lease_context().expect("context");
        let context = lease
            .context()
            .complex_multiplicity_free_lane()
            .tree_context_mut();

        context
            .tree_transform_dyn_overwrite_into_ref(
                rule.as_ref(),
                &operation,
                &destination_structure,
                &source_structure,
                &mut destination_data,
                &source_data,
                Complex64::new(1.0, 0.0),
            )
            .expect("cold Fibonacci braid");
        let cold_calls = rule.structural_calls();
        let cold_layout_calls = rule.layout_calls();
        assert!(cold_calls > 0);
        assert!(cold_layout_calls > 0);
        assert!(destination_data.iter().any(|value| value.im != 0.0));
        let cold_destination_data = destination_data.clone();
        let cold = crate::test_cache::completed();
        assert_eq!(cold.entries(), 1);
        assert_eq!(cold.misses(), 1);
        assert_eq!(cold.hits(), 0);

        destination_data.fill(Complex64::new(0.0, 0.0));
        context
            .tree_transform_dyn_overwrite_into_ref(
                rule.as_ref(),
                &operation,
                &destination_structure,
                &source_structure,
                &mut destination_data,
                &source_data,
                Complex64::new(1.0, 0.0),
            )
            .expect("warm Fibonacci braid");
        assert_eq!(destination_data, cold_destination_data);
        assert!(destination_data.iter().any(|value| value.im != 0.0));
        assert_eq!(rule.structural_calls(), cold_calls);
        // Layout/admission queries are observed separately; only structural
        // F/R replay is promised to disappear on a warm cache hit.
        assert!(rule.layout_calls() >= cold_layout_calls);
        let warm = crate::test_cache::completed();
        assert_eq!(warm.entries(), 1);
        assert_eq!(warm.misses(), 1);
        assert_eq!(warm.hits(), 1);
    }

    #[test]
    fn fibonacci_checked_construction_failures_publish_nothing() {
        if crate::test_cache::run_isolated_or_return(
            "TENET_FIBONACCI_FAILURES_ISOLATED",
            "runtime::tests::fibonacci_checked_construction_failures_publish_nothing",
        ) {
            return;
        }
        let runtime = Runtime::builder().build().expect("runtime");
        let rule = Arc::new(CountingFibonacci {
            malformed_channels: true,
            ..CountingFibonacci::new()
        });
        let tau = GradedSpace::try_new(Arc::clone(&rule), [(FibonacciSector::Tau, 1)])
            .expect("label admission");
        crate::cache::clear();
        let cache_before = crate::test_cache::completed();
        let callbacks = AtomicUsize::new(0);

        let late = TensorMap::<CountingFibonacci, Complex64>::from_subblock_fn(
            &runtime,
            [&tau, &tau, &tau],
            [&tau],
            |_, _| {
                callbacks.fetch_add(1, Ordering::Relaxed);
                Complex64::new(1.0, 0.0)
            },
        );
        let late = late.unwrap_err();
        assert!(format!("{late:?}").contains("InvalidSector"));
        assert_eq!(callbacks.load(Ordering::Relaxed), 0);
        assert_eq!(crate::test_cache::completed(), cache_before);

        let early = TensorMap::<CountingFibonacci, Complex64>::from_subblock_fn(
            &runtime,
            std::iter::empty(),
            std::iter::empty(),
            |_, _| {
                callbacks.fetch_add(1, Ordering::Relaxed);
                Complex64::new(1.0, 0.0)
            },
        );
        assert!(matches!(
            early,
            Err(Error::InvalidArgument(message))
                if message == "at least one leg is required to infer the fusion provider"
        ));
        assert_eq!(callbacks.load(Ordering::Relaxed), 0);
        assert_eq!(crate::test_cache::completed(), cache_before);

        assert!(TensorMap::<CountingFibonacci, Complex64>::rand_with_seed(
            &runtime,
            [&tau, &tau, &tau],
            [&tau],
            0x9E37_79B9_7F4A_7C15,
        )
        .is_err());
        assert_eq!(crate::test_cache::completed(), cache_before);
    }

    #[test]
    #[allow(deprecated)]
    fn runtimes_share_completed_transformers_and_expired_weak_handles_run_eagerly() {
        if crate::test_cache::run_isolated_or_return(
            "TENET_RUNTIMES_SHARE_ISOLATED",
            "runtime::tests::runtimes_share_completed_transformers_and_expired_weak_handles_run_eagerly",
        ) {
            return;
        }
        let runtime_a = Runtime::builder().build().unwrap();
        let runtime_b = Runtime::builder().build().unwrap();
        let provider = Arc::new(SU2FusionRule);
        let space = GradedSpace::try_new(
            provider,
            [
                (SU2Irrep::from_twice_spin(0), 2),
                (SU2Irrep::from_twice_spin(1), 2),
                (SU2Irrep::from_twice_spin(2), 1),
            ],
        )
        .unwrap();
        let source_a: TensorMap<SU2FusionRule, f64> =
            TensorMap::rand_with_seed(&runtime_a, [&space, &space], [&space], 475_002).unwrap();
        let source_b: TensorMap<SU2FusionRule, f64> =
            TensorMap::rand_with_seed(&runtime_b, [&space, &space], [&space], 475_002).unwrap();
        crate::cache::clear();
        let expected_a = source_a.permute(&[1], &[2, 0]).unwrap();
        let after_a = crate::test_cache::completed();
        let expected_b = source_b.permute(&[1], &[2, 0]).unwrap();
        let after_b = crate::test_cache::completed();
        // What: one process-global entry serves both Runtimes (#2014-3).
        assert_eq!((after_a.entries(), after_a.misses()), (1, 1));
        assert_eq!((after_b.entries(), after_b.misses()), (1, 1));
        assert_eq!(after_b.hits(), after_a.hits() + 1);
        assert_eq!(
            expected_a.dense_data().unwrap(),
            expected_b.dense_data().unwrap()
        );

        // The deprecated per-Runtime clear clears the global caches for
        // every Runtime, and reports the shared cache through either one.
        runtime_a.clear_tree_transform_cache();
        assert_eq!(crate::test_cache::completed().entries(), 0);
        assert_eq!(
            runtime_b.tree_transform_cache_info().structures.entries(),
            0
        );
        assert_eq!(
            runtime_b
                .tree_transform_cache_info()
                .structures
                .entry_capacity(),
            usize::MAX
        );

        let mut context = TensorExecutionContext::for_config(runtime_b.execution_config()).unwrap();
        let source_space = source_b.test_bound_space();
        let destination_space = expected_b.test_bound_space();
        let rule = Arc::clone(source_space.provider_arc());
        let source_structure = Arc::clone(source_space.space().structure());
        let destination_structure = Arc::clone(destination_space.space().structure());
        let source_data = source_b.dense_data().unwrap().to_vec();
        let expected_data = expected_b.dense_data().unwrap().to_vec();
        let operation = TreeTransformOperation::permute([1], [2, 0]);
        drop(source_a);
        drop(source_b);
        drop(expected_a);
        drop(expected_b);
        drop(runtime_a);
        drop(runtime_b);

        let mut actual = vec![f64::NAN; expected_data.len()];
        context
            .mf
            .f64
            .tree_context_mut()
            .tree_transform_dyn_overwrite_into_ref(
                rule.as_ref(),
                &operation,
                &destination_structure,
                &source_structure,
                &mut actual,
                &source_data,
                1.0,
            )
            .unwrap();
        assert_eq!(actual, expected_data);
    }

    /// Compile-level gate for the device-state ownership of #1322: the lease is
    /// still a `CudaDenseContext` for every caller that wants only the dense
    /// context, and a transform reaches both halves through `split()` without
    /// leasing twice. It is never executed — a device would be needed to
    /// obtain a lease — but it fails the build if either shape regresses.
    #[cfg(feature = "cuda")]
    #[allow(dead_code)]
    fn cuda_lease_still_derefs_to_the_dense_context_and_splits(mut lease: CudaLease<'_>) {
        let dense: &mut tenet_dense::CudaDenseContext = &mut lease;
        let _ = dense.device();
        let (dense, executor) = lease.split();
        let _ = executor.retained_device_bytes(dense);
    }
}
