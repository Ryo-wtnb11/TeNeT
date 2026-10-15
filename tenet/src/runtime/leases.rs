use super::*;

/// A runtime's Host pool entry on the calling thread (see
/// [`Runtime::enter_host_pool`]); the pool stays entered until it drops.
#[must_use = "the pool is entered only while the entry lives"]
pub(crate) struct HostPoolEntry {
    pub(super) _guard: tenet_operations::host_pool::HostPoolGuard,
    #[cfg(test)]
    pub(super) _probe: HostPoolDepth,
}

/// Test mirror of the entered runtime pools on this thread, so a probe can
/// ask whether a data-movement site runs inside one.
#[cfg(test)]
pub(super) struct HostPoolDepth;

#[cfg(test)]
impl HostPoolDepth {
    pub(super) fn enter() -> Self {
        crate::typed::probe::enter(crate::typed::probe::Gauge::HostPool);
        Self
    }
}

#[cfg(test)]
impl Drop for HostPoolDepth {
    fn drop(&mut self) {
        crate::typed::probe::leave(crate::typed::probe::Gauge::HostPool);
    }
}

/// RAII lease of a pooled execution context (#155). Returns it to the
/// pool on drop; on panic it is dropped instead of returned (quarantine —
/// mirrors `tenet_network`'s `WorkspaceLease`). While it lives, the runtime's
/// CPU pool is the calling thread's Host pool, so the operation's replay,
/// plan compile and strided regions run on it.
pub(crate) struct ContextLease<'a> {
    pub(super) pool: &'a Mutex<Vec<PooledContext>>,
    pub(super) max_idle: usize,
    pub(super) context: Option<PooledContext>,
    pub(super) _host_pool: HostPoolEntry,
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
