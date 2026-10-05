use super::*;

mod network_operand_sealed {
    pub trait Sealed {}
}

/// A tensor type [`Network::contract`] accepts: the typed Host tensor and,
/// with the `cuda` feature, the typed CUDA tensor. Every operand of one call
/// has the same provider, scalar and storage, which Rust checks.
pub trait NetworkOperand: network_operand_sealed::Sealed + Sized {
    /// The error [`Network::contract`] returns for this operand type.
    type Error;

    #[doc(hidden)]
    fn contract_network(network: &Network, tensors: &[&Self]) -> Result<Self, Self::Error>;
}

impl Network {
    /// Contracts `tensors` eagerly through the Runtime's plan cache.
    ///
    /// The plan is looked up by the network's topology — labels, `conj`
    /// markers, written `;` splits, output, the operands' codomain ranks —
    /// and the optimizer of the operands' Runtime
    /// ([`PlanCacheConfig::optimizer`](crate::PlanCacheConfig)), so a repeated
    /// call reuses the plan and a pooled workspace. Leg dimensions are not
    /// part of the key: [`ReplanPolicy`](crate::ReplanPolicy) decides when a
    /// dimension change replans. The cache's statistics, persistence and
    /// configuration are [`plan_cache_stats`](crate::plan_cache_stats),
    /// [`save_plan_cache`](crate::save_plan_cache) /
    /// [`load_plan_cache`](crate::load_plan_cache) and
    /// [`configure_plan_cache`](crate::configure_plan_cache).
    ///
    /// Use [`Network::plan`] and [`PlannedNetwork::execute`] instead to pick
    /// the optimizer per call or to own the plan and workspace.
    ///
    /// Every metadata rejection — operand count, labels against ranks and
    /// `;` splits, contracted spaces, Runtime and rule identity, braiding,
    /// and on a device the placement and dense payloads — is decided before
    /// the plan lookup, so a rejected call leaves the cache untouched.
    ///
    /// # Errors
    ///
    /// The typed error of the operands' mode, as [`PlannedNetwork::execute`]
    /// returns it, or [`Error`] for device operands.
    pub fn contract<T: NetworkOperand>(&self, tensors: &[&T]) -> Result<T, T::Error> {
        T::contract_network(self, tensors)
    }

    fn validate_operand_count(&self, operands: usize) -> Result<(), Error> {
        if operands != self.inputs.len() {
            return Err(invalid(format!(
                "network has {} operands but {} tensors were given",
                self.inputs.len(),
                operands
            )));
        }
        Ok(())
    }

    fn contract_preflight<R, D, S>(
        &self,
        tensors: &[&TensorMap<R, D, S>],
    ) -> Result<(), HostNetworkError<R>>
    where
        R: TypedSectorAdmission,
        R::Mode: HostNetworkModeDispatch<R, D>,
        D: TensorScalar,
        S: TensorStorage<D>,
    {
        static_operand_preflight(
            tensors,
            &self.inputs,
            &self.conj,
            &self.codomain_splits,
            &[],
            Some(&self.contracted),
        )
    }
}

fn runtime_optimizer<R, D, S>(tensors: &[&TensorMap<R, D, S>]) -> crate::Optimizer
where
    S: TensorStorage<D>,
{
    tensors
        .first()
        .map(|tensor| tensor.runtime().plan_cache_config().optimizer)
        .unwrap_or_default()
}

impl<R, D> network_operand_sealed::Sealed for TensorMap<R, D>
where
    R: TypedSectorAdmission + Send + Sync,
    R::Mode: HostNetworkModeDispatch<R, D>,
    D: TensorScalar + Send + Sync + 'static,
{
}

impl<R, D> NetworkOperand for TensorMap<R, D>
where
    R: TypedSectorAdmission + Send + Sync,
    R::Mode: HostNetworkModeDispatch<R, D>,
    D: TensorScalar + Send + Sync + 'static,
{
    type Error = HostNetworkError<R>;

    fn contract_network(network: &Network, tensors: &[&Self]) -> Result<Self, Self::Error> {
        network.validate_operand_count(tensors.len())?;
        reject_non_symmetric_network(tensors, tensors.len() > 1)?;
        network.contract_preflight(tensors)?;
        crate::plancache::get_or_plan_network(network, tensors, &runtime_optimizer(tensors))?
            .execute_host(tensors)
    }
}

#[cfg(feature = "cuda")]
impl<R, D> network_operand_sealed::Sealed for TensorMap<R, D, CudaStorage<D>>
where
    R: TypedSectorAdmission<Error = FusionAlgebraError, Mode = MultiplicityFreeAdmissionMode>
        + MultiplicityFreeRigidSymbols<Scalar = f64>
        + CheckedFusionAlgebra
        + SectorCodec
        + Send
        + Sync,
    D: CudaPayload,
{
}

#[cfg(feature = "cuda")]
impl<R, D> NetworkOperand for TensorMap<R, D, CudaStorage<D>>
where
    R: TypedSectorAdmission<Error = FusionAlgebraError, Mode = MultiplicityFreeAdmissionMode>
        + MultiplicityFreeRigidSymbols<Scalar = f64>
        + CheckedFusionAlgebra
        + SectorCodec
        + Send
        + Sync,
    D: CudaPayload,
{
    type Error = Error;

    fn contract_network(network: &Network, tensors: &[&Self]) -> Result<Self, Error> {
        network.validate_operand_count(tensors.len())?;
        cuda_operand_admission(tensors, tensors.len() > 1)?;
        network.contract_preflight(tensors)?;
        crate::plancache::get_or_plan_network(network, tensors, &runtime_optimizer(tensors))?
            .execute_cuda(tensors)
    }
}
