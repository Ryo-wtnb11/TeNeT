//! The Host CPU pool that an eager operation's parallel regions run on.
//!
//! A `Runtime` owns one CPU context ([`SharedCpuContext`]); its Rayon pool
//! runs the runtime's dense kernels. An eager operation enters that context on
//! the calling thread with [`with_host_pool`]. Every parallel region below it
//! (tree-transform replay fan-out, plan compile, strided kernels) then runs
//! inside the same pool through [`install_region`] or [`strided`], and every
//! degree cap reads [`current_threads`].
//!
//! Why the regions are installed, not the whole operation: a Rayon worker that
//! waits in `join` or in a nested `install` runs other queued jobs of its
//! pool. A whole operation picked up that way could enter Tenferro's backend
//! while another operation's dense call owns the same pool (Tenferro panics on
//! that re-entry), or block on a runtime lock held lower on the same stack.
//! Regions hold no lock and never call the dense backend, so running one of
//! them on a waiting worker is harmless.
//!
//! With no entered pool (direct callers of this crate without a `Runtime`),
//! regions use the ambient Rayon pool as before.

use std::cell::RefCell;
use std::marker::PhantomData;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Mutex;

use strided_kernel::{with_execution_policy, ExecutionPolicy};
use tenet_dense::SharedCpuContext;

thread_local! {
    static CURRENT: RefCell<Option<SharedCpuContext>> = const { RefCell::new(None) };
}

/// Keeps a pool entered on the calling thread until dropped (see
/// [`enter_host_pool`]). Not `Send`: the entry is thread-local.
#[must_use = "the pool is entered only while the guard lives"]
pub struct HostPoolGuard {
    entered: usize,
    previous: Option<SharedCpuContext>,
    _thread_bound: PhantomData<*const ()>,
}

impl Drop for HostPoolGuard {
    fn drop(&mut self) {
        let previous = self.previous.take();
        CURRENT.with(|current| {
            let mut current = current.borrow_mut();
            // Why the identity check: a guard dropped out of nesting order must
            // not re-enter a pool that an inner guard has already left.
            if current.as_ref().map(SharedCpuContext::identity) == Some(self.entered) {
                *current = previous;
            }
        });
    }
}

/// Enters `pool` as the calling thread's Host pool until the guard drops.
/// Entries nest. Used by owners whose lifetime is lexical but not a closure,
/// such as a runtime's execution leases.
pub fn enter_host_pool(pool: &SharedCpuContext) -> HostPoolGuard {
    HostPoolGuard {
        entered: pool.identity(),
        previous: CURRENT.with(|current| current.replace(Some(pool.clone()))),
        _thread_bound: PhantomData,
    }
}

/// Runs `op` with `pool` as the calling thread's Host pool.
pub fn with_host_pool<R>(pool: &SharedCpuContext, op: impl FnOnce() -> R) -> R {
    let _entered = enter_host_pool(pool);
    op()
}

fn current_pool() -> Option<SharedCpuContext> {
    CURRENT.with(|current| current.borrow().clone())
}

/// Worker count a parallel region may use: the entered pool's size, or the
/// ambient Rayon pool's when no pool is entered.
pub fn current_threads() -> usize {
    CURRENT
        .with(|current| current.borrow().as_ref().map(SharedCpuContext::num_threads))
        .unwrap_or_else(rayon::current_num_threads)
        .max(1)
}

/// Runs a parallel region inside the entered pool. A one-thread pool runs it
/// on the calling thread with strided fan-out disabled. `op` must not take a
/// runtime lock or call the dense backend (see the module docs).
pub fn install_region<R: Send>(site: HostPoolSite, op: impl FnOnce() -> R + Send) -> R {
    match current_pool() {
        None => {
            observe(site, None);
            op()
        }
        Some(pool) if pool.num_threads() == 1 => {
            with_execution_policy(ExecutionPolicy::Sequential, || {
                observe(site, Some(&pool));
                op()
            })
        }
        Some(pool) => pool.install(|| {
            with_host_pool(&pool, || {
                observe(site, Some(&pool));
                op()
            })
        }),
    }
}

/// Runs one strided-kernel call of `len` elements. At or below strided's own
/// threading gate the kernel is serial whichever pool is current, so it runs
/// in place without an install; above it the call is a parallel region.
pub fn strided<R: Send>(len: usize, op: impl FnOnce() -> R + Send) -> R {
    if len <= strided_kernel::execution::MINTHREADLENGTH {
        return match CURRENT.with(|current| current.borrow().is_some()) {
            // Why Sequential rather than ambient: a strided entry may query the
            // ambient pool before its own size gate, which would initialize
            // Rayon's global pool from inside a runtime operation.
            true => with_execution_policy(ExecutionPolicy::Sequential, op),
            false => op(),
        };
    }
    install_region(HostPoolSite::Strided, op)
}

/// Where a Host pool observation was taken.
#[doc(hidden)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum HostPoolSite {
    Replay,
    PlanCompile,
    Strided,
    Dense,
}

/// One observation of the pool a Host parallel site ran in.
#[doc(hidden)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct HostPoolObservation {
    pub site: HostPoolSite,
    /// [`SharedCpuContext::identity`] of the entered pool, if any.
    pub entered: Option<usize>,
    /// `rayon::current_num_threads()` when the site ran on a Rayon worker;
    /// `None` when it ran on the calling (non-pool) thread.
    pub worker_pool_threads: Option<usize>,
}

static OBSERVING: AtomicBool = AtomicBool::new(false);
static OBSERVATIONS: Mutex<Vec<HostPoolObservation>> = Mutex::new(Vec::new());

/// Test hook: starts or stops recording [`HostPoolObservation`]s. Off by
/// default; when off a site pays one relaxed atomic load.
#[doc(hidden)]
pub fn observe_host_pools(enabled: bool) {
    OBSERVING.store(enabled, Ordering::Relaxed);
}

/// Test hook: drains the recorded observations.
#[doc(hidden)]
pub fn take_host_pool_observations() -> Vec<HostPoolObservation> {
    std::mem::take(&mut *OBSERVATIONS.lock().unwrap_or_else(|p| p.into_inner()))
}

/// Records where a dense linear-algebra scope body runs (it is installed by
/// Tenferro, not by [`install_region`]).
#[doc(hidden)]
pub fn observe_dense_site() {
    let pool = current_pool();
    observe(HostPoolSite::Dense, pool.as_ref());
}

fn observe(site: HostPoolSite, pool: Option<&SharedCpuContext>) {
    if !OBSERVING.load(Ordering::Relaxed) {
        return;
    }
    let observation = HostPoolObservation {
        site,
        entered: pool.map(SharedCpuContext::identity),
        worker_pool_threads: rayon::current_thread_index().map(|_| rayon::current_num_threads()),
    };
    OBSERVATIONS
        .lock()
        .unwrap_or_else(|p| p.into_inner())
        .push(observation);
}
