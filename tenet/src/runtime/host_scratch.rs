use super::*;

impl<Key: Clone + Eq + Hash + Send + Sync + 'static> Ctxs<Key> {
    fn retained_host_scratch_bytes(&self) -> usize {
        let mut bytes = self
            .f64
            .retained_host_scratch_bytes()
            .saturating_add(self.c64.retained_host_scratch_bytes());
        if let Some(lane) = &self.f32 {
            bytes = bytes.saturating_add(lane.retained_host_scratch_bytes());
        }
        if let Some(lane) = &self.c32 {
            bytes = bytes.saturating_add(lane.retained_host_scratch_bytes());
        }
        bytes
    }

    fn trim_host_scratch(&mut self) {
        self.f64.trim_host_scratch();
        self.c64.trim_host_scratch();
        if let Some(lane) = &mut self.f32 {
            lane.trim_host_scratch();
        }
        if let Some(lane) = &mut self.c32 {
            lane.trim_host_scratch();
        }
    }
}

impl TensorExecutionContext {
    fn retained_host_scratch_bytes(&self) -> usize {
        self.mf
            .retained_host_scratch_bytes()
            .saturating_add(self.generic.retained_host_scratch_bytes())
            .saturating_add(self.mf_c64_coeff_c64.retained_host_scratch_bytes())
    }

    fn trim_host_scratch(&mut self) {
        self.mf.trim_host_scratch();
        self.generic.trim_host_scratch();
        self.mf_c64_coeff_c64.trim_host_scratch();
    }
}

impl Runtime {
    /// Allocated Host capacity retained by idle standalone contraction contexts.
    ///
    /// The snapshot sums fusion lhs/rhs/destination and `copyC` buffers and
    /// the tree-transform workspaces (one per contraction stage plus the
    /// context's own, coefficient packs included) across every instantiated
    /// payload/coefficient lane in the idle pool. Active leases and the
    /// lock-held expert contexts are excluded; this method does not wait for
    /// operations. Fusion-block and dense-provider workspaces are outside this
    /// counter.
    pub fn host_contract_scratch_bytes(&self) -> usize {
        self.inner
            .context_pool
            .lock()
            .expect("context pool poisoned")
            .iter()
            .fold(0usize, |bytes, context| {
                bytes.saturating_add(context.retained_host_scratch_bytes())
            })
    }

    /// Releases Host contraction scratch held by currently idle contexts.
    ///
    /// A concurrently active lease is unchanged and may return its buffers to
    /// the idle pool after this call; call trim again after operations finish
    /// to release those buffers. Device state is unaffected.
    pub fn trim_host_contract_scratch(&self) {
        let mut contexts = self
            .inner
            .context_pool
            .lock()
            .expect("context pool poisoned");
        for context in contexts.iter_mut() {
            context.trim_host_scratch();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tenet_operations::host_scratch::HostScratchBuffer;

    fn seed<D, C>(lane: &mut CoefficientCtx<D, RuleIdentity, C>, len: usize) -> usize
    where
        D: ScalarOps + tenet_tensors::RecouplingCoefficientAction<C>,
        C: tenet_tensors::DenseBlockScalar,
    {
        let mut scratch = HostScratchBuffer::default();
        scratch.resize_filled(len, D::from_real(0.0));
        let bytes = scratch.capacity().saturating_mul(std::mem::size_of::<D>());
        lane.restore_copy_c_scratch(scratch);
        bytes
    }

    fn seed_all_lanes(context: &mut TensorExecutionContext) -> usize {
        context.mf.force_single_precision_lanes().unwrap();
        context.generic.force_single_precision_lanes().unwrap();
        let mut bytes = seed(&mut context.mf.f64, 11);
        bytes += seed(&mut context.mf.c64, 12);
        bytes += seed(context.mf.f32.as_deref_mut().unwrap(), 13);
        bytes += seed(context.mf.c32.as_deref_mut().unwrap(), 14);
        bytes += seed(&mut context.generic.f64, 15);
        bytes += seed(&mut context.generic.c64, 16);
        bytes += seed(context.generic.f32.as_deref_mut().unwrap(), 17);
        bytes += seed(context.generic.c32.as_deref_mut().unwrap(), 18);
        bytes + seed(&mut context.mf_c64_coeff_c64, 19)
    }

    #[test]
    fn host_scratch_observation_counts_every_idle_context_and_instantiated_lane() {
        let runtime = Runtime::builder().dense_threads(1).build().unwrap();
        let mut first = runtime.lease_context().unwrap();
        let mut second = runtime.lease_context().unwrap();
        let expected = seed_all_lanes(first.context()) + seed_all_lanes(second.context());
        assert_eq!(runtime.host_contract_scratch_bytes(), 0);
        drop(first);
        drop(second);

        assert_eq!(runtime.host_contract_scratch_bytes(), expected);
        runtime.trim_host_contract_scratch();
        assert_eq!(runtime.host_contract_scratch_bytes(), 0);
    }

    #[test]
    fn trim_does_not_mutate_a_concurrently_active_lease() {
        let runtime = Runtime::builder().dense_threads(1).build().unwrap();
        let worker_runtime = runtime.clone();
        let leased = Arc::new(std::sync::Barrier::new(2));
        let release = Arc::new(std::sync::Barrier::new(2));
        let worker = {
            let leased = Arc::clone(&leased);
            let release = Arc::clone(&release);
            std::thread::spawn(move || {
                let mut lease = worker_runtime.lease_context().unwrap();
                let expected = seed(lease.context().multiplicity_free_lane::<f64>().unwrap(), 23);
                leased.wait();
                release.wait();
                expected
            })
        };
        leased.wait();

        assert_eq!(runtime.host_contract_scratch_bytes(), 0);
        runtime.trim_host_contract_scratch();
        release.wait();
        let expected = worker.join().unwrap();
        assert_eq!(runtime.host_contract_scratch_bytes(), expected);
        runtime.trim_host_contract_scratch();
        assert_eq!(runtime.host_contract_scratch_bytes(), 0);
    }
}
