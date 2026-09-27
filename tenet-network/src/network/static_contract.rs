use super::*;

mod static_operand_sealed {
    pub trait Sealed {}
}

/// Closed dispatch surface used by the `tensor!` expansion.
///
/// Every operand in one invocation must have the same provider, scalar, and
/// supported storage type. These mismatches are rejected by Rust before a
/// network can be planned or executed:
///
/// ```compile_fail
/// use tenet::core::{SU2FusionRule, U1FusionRule};
/// use tenet::typed::TensorMap;
/// use tenet_network::tensor;
///
/// fn mixed_provider(
///     a: &TensorMap<U1FusionRule, f64>,
///     b: &TensorMap<SU2FusionRule, f64>,
/// ) {
///     let _ = tensor!([i; k] = a[i; j] * b[j; k]);
/// }
/// ```
///
/// ```compile_fail
/// use tenet::core::U1FusionRule;
/// use tenet::prelude::Complex64;
/// use tenet::typed::TensorMap;
/// use tenet_network::tensor;
///
/// fn mixed_scalar(
///     a: &TensorMap<U1FusionRule, f64>,
///     b: &TensorMap<U1FusionRule, Complex64>,
/// ) {
///     let _ = tensor!([i; k] = a[i; j] * b[j; k]);
/// }
/// ```
///
/// ```compile_fail
/// use tenet::core::{Placement, TensorStorage, U1FusionRule};
/// use tenet::typed::TensorMap;
/// use tenet_network::tensor;
///
/// struct DeviceStorage;
/// impl TensorStorage<f64> for DeviceStorage {
///     fn len(&self) -> usize { 0 }
///     fn placement(&self) -> Placement { Placement::Cuda(0) }
/// }
///
/// fn mixed_storage(
///     a: &TensorMap<U1FusionRule, f64>,
///     b: &TensorMap<U1FusionRule, f64, DeviceStorage>,
/// ) {
///     let _ = tensor!([i; k] = a[i; j] * b[j; k]);
/// }
/// ```
///
/// Checked Generic providers use the same Host dispatch surface and preserve
/// their typed provider errors. The macro uses [`StaticTraceNetworkOperand`]
/// only when an operand contains a trace, so ordinary networks do not require
/// pivotal provider data.
#[doc(hidden)]
pub trait StaticNetworkOperand: static_operand_sealed::Sealed + Sized {
    type Error;

    fn contract_static(
        tensors: &[&Self],
        spec: &'static StaticTopologySpec,
    ) -> Result<Self, Self::Error>;
}

/// Static network dispatch with the ordinary typed trace operation available.
#[doc(hidden)]
pub trait StaticTraceNetworkOperand: StaticNetworkOperand {
    fn contract_static_trace(
        tensors: &[&Self],
        spec: &'static StaticTopologySpec,
    ) -> Result<Self, Self::Error>;
}

/// Preserves each macro operand's concrete typed tensor without conversion.
#[doc(hidden)]
pub fn normalize_tensor_operand<R, D, S, O>(tensor: &O) -> &TensorMap<R, D, S>
where
    O: AsRef<TensorMap<R, D, S>> + ?Sized,
{
    tensor.as_ref()
}

/// Typed Host/CUDA dispatch used by `tensor!`.
#[doc(hidden)]
pub fn contract_static_network<T: StaticNetworkOperand>(
    tensors: &[&T],
    spec: &'static StaticTopologySpec,
) -> Result<T, T::Error> {
    T::contract_static(tensors, spec)
}

/// Typed Host/CUDA dispatch used by `tensor!` when an operand contains a trace.
#[doc(hidden)]
pub fn contract_static_trace_network<T: StaticTraceNetworkOperand>(
    tensors: &[&T],
    spec: &'static StaticTopologySpec,
) -> Result<T, T::Error> {
    T::contract_static_trace(tensors, spec)
}

fn validate_static_shape<T>(tensors: &[&T], spec: &StaticTopologySpec) -> Result<(), Error> {
    if tensors.len() != spec.inputs.len() {
        return Err(invalid(format!(
            "network has {} operands but {} tensors were given",
            spec.inputs.len(),
            tensors.len()
        )));
    }
    if spec.conj.len() != spec.inputs.len() || spec.codomain_splits.len() != spec.inputs.len() {
        return Err(invalid(
            "static topology marker lists must match operand count",
        ));
    }
    Ok(())
}

impl<R, D> static_operand_sealed::Sealed for TensorMap<R, D>
where
    R: TypedSectorAdmission + Send + Sync,
    R::Mode: HostNetworkModeDispatch<R, D>,
    D: TensorScalar + Send + Sync + 'static,
{
}

impl<R, D> StaticNetworkOperand for TensorMap<R, D>
where
    R: TypedSectorAdmission + Send + Sync,
    R::Mode: HostNetworkModeDispatch<R, D>,
    D: TensorScalar + Send + Sync + 'static,
{
    type Error = HostNetworkError<R>;

    fn contract_static(
        tensors: &[&Self],
        spec: &'static StaticTopologySpec,
    ) -> Result<Self, Self::Error> {
        validate_static_shape(tensors, spec)?;
        reject_non_symmetric_network(tensors, tensors.len() > 1)?;
        static_network_operand_preflight(tensors, spec)?;
        let codomain_ranks = tensors
            .iter()
            .map(|tensor| tensor.codomain_rank())
            .collect::<Vec<_>>();
        let optimizer = tensors
            .first()
            .map(|tensor| tensor.runtime().plan_cache_config().optimizer)
            .unwrap_or_default();
        crate::plancache::get_or_plan_static(spec, tensors, &codomain_ranks, &optimizer, || {
            spec.network()
        })?
        .execute_host(tensors)
    }
}

impl<R, D> StaticTraceNetworkOperand for TensorMap<R, D>
where
    R: TypedSectorAdmission + Send + Sync,
    R::Mode: HostNetworkModeDispatch<R, D> + TypedTensorTraceDispatch<R, D>,
    D: TensorScalar + Send + Sync + 'static,
{
    fn contract_static_trace(
        tensors: &[&Self],
        spec: &'static StaticTopologySpec,
    ) -> Result<Self, Self::Error> {
        validate_static_shape(tensors, spec)?;
        // The trace pre-step keeps the operand count, so whether the reduced
        // network contracts is known now.
        reject_non_symmetric_network(tensors, tensors.len() > 1)?;
        let codomain_ranks = tensors
            .iter()
            .map(|tensor| tensor.codomain_rank())
            .collect::<Vec<_>>();
        let optimizer = tensors
            .first()
            .map(|tensor| tensor.runtime().plan_cache_config().optimizer)
            .unwrap_or_default();
        let lowering = StaticTraceLowering::new(tensors, spec)?;
        lowering.preflight(tensors)?;
        let mut lowered = Vec::with_capacity(tensors.len());
        for (tensor, trace) in tensors.iter().zip(&lowering.traces) {
            lowered.push(match trace {
                None => None,
                Some((adjoint, pairs)) => {
                    let value = if *adjoint {
                        tensor.adjoint()?
                    } else {
                        (*tensor).clone()
                    };
                    Some(value.trace_pairs(pairs)?)
                }
            });
        }
        let reduced = tensors
            .iter()
            .zip(&lowered)
            .map(|(tensor, traced)| traced.as_ref().unwrap_or(tensor))
            .collect::<Vec<_>>();
        crate::plancache::get_or_plan_static(spec, &reduced, &codomain_ranks, &optimizer, || {
            lowering.network(spec)
        })?
        .execute_host(&reduced)
    }
}

#[cfg(feature = "cuda")]
impl<R, D> static_operand_sealed::Sealed for TensorMap<R, D, CudaStorage<D>>
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
impl<R, D> StaticNetworkOperand for TensorMap<R, D, CudaStorage<D>>
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

    fn contract_static(
        tensors: &[&Self],
        spec: &'static StaticTopologySpec,
    ) -> Result<Self, Error> {
        validate_static_shape(tensors, spec)?;
        cuda_operand_admission(tensors, tensors.len() > 1)?;
        static_network_operand_preflight(tensors, spec)?;
        let codomain_ranks = tensors
            .iter()
            .map(|tensor| tensor.codomain_rank())
            .collect::<Vec<_>>();
        let optimizer = tensors
            .first()
            .map(|tensor| tensor.runtime().plan_cache_config().optimizer)
            .unwrap_or_default();
        crate::plancache::get_or_plan_static(spec, tensors, &codomain_ranks, &optimizer, || {
            spec.network()
        })?
        .execute_cuda(tensors)
    }
}

#[cfg(feature = "cuda")]
impl<R, D> StaticTraceNetworkOperand for TensorMap<R, D, CudaStorage<D>>
where
    R: TypedSectorAdmission<Error = FusionAlgebraError, Mode = MultiplicityFreeAdmissionMode>
        + MultiplicityFreeRigidSymbols<Scalar = f64>
        + CheckedFusionAlgebra
        + SectorCodec
        + Send
        + Sync,
    D: CudaPayload,
{
    /// The Host trace pre-step with device `trace_pairs`. Every trace of the
    /// network is validated and compiled on the Host before the first one
    /// runs, so a rejection on any operand leaves the device untouched; the
    /// reduced network then takes the ordinary device path.
    fn contract_static_trace(
        tensors: &[&Self],
        spec: &'static StaticTopologySpec,
    ) -> Result<Self, Self::Error> {
        validate_static_shape(tensors, spec)?;
        // The trace pre-step keeps the operand count, so a reduced network
        // that contracts is decided here; a non-symmetric trace alone is
        // rejected below with the Host's trace error.
        cuda_operand_admission(tensors, tensors.len() > 1)?;
        let codomain_ranks = tensors
            .iter()
            .map(|tensor| tensor.codomain_rank())
            .collect::<Vec<_>>();
        let optimizer = tensors
            .first()
            .map(|tensor| tensor.runtime().plan_cache_config().optimizer)
            .unwrap_or_default();
        let lowering = StaticTraceLowering::new(tensors, spec)?;
        lowering.preflight(tensors)?;
        let sources = tensors
            .iter()
            .zip(&lowering.traces)
            .map(|(tensor, trace)| {
                trace
                    .as_ref()
                    .map(|(adjoint, pairs)| {
                        let value = if *adjoint {
                            tensor.adjoint()?
                        } else {
                            (*tensor).clone()
                        };
                        Ok((value, pairs))
                    })
                    .transpose()
            })
            .collect::<Result<Vec<_>, Error>>()?;
        let prepared = sources
            .iter()
            .map(|source| match source {
                Some((value, pairs)) => value.prepare_trace_pairs(pairs),
                None => Ok(None),
            })
            .collect::<Result<Vec<_>, Error>>()?;
        let lowered = prepared
            .into_iter()
            .map(|trace| trace.map(|trace| trace.execute()).transpose())
            .collect::<Result<Vec<_>, Error>>()?;
        let reduced = tensors
            .iter()
            .zip(&lowered)
            .map(|(tensor, traced)| traced.as_ref().unwrap_or(tensor))
            .collect::<Vec<_>>();
        crate::plancache::get_or_plan_static(spec, &reduced, &codomain_ranks, &optimizer, || {
            lowering.network(spec)
        })?
        .execute_cuda(&reduced)
    }
}

/// The call-local trace lowering of a `tensor!` expression, decided from its
/// labels and the operands' ranks alone: the reduced network's operand
/// labels, and per traced operand whether it is read through its adjoint and
/// its intra-operand pairs. Host and device execute the same lowering.
struct StaticTraceLowering {
    inputs: Vec<Vec<TemporaryLabel>>,
    conj: Vec<bool>,
    splits: Vec<Option<usize>>,
    traces: Vec<Option<StaticTrace>>,
}

/// One operand's trace: whether it is read through its adjoint, and its pairs.
type StaticTrace = (bool, Vec<(usize, usize)>);

impl StaticTraceLowering {
    fn new<R, D, S>(
        tensors: &[&TensorMap<R, D, S>],
        spec: &StaticTopologySpec,
    ) -> Result<Self, Error>
    where
        D: TensorScalar,
        S: TensorStorage<D>,
    {
        let mut lowering = Self {
            inputs: Vec::with_capacity(tensors.len()),
            conj: Vec::with_capacity(tensors.len()),
            splits: Vec::with_capacity(tensors.len()),
            traces: Vec::with_capacity(tensors.len()),
        };
        for (index, tensor) in tensors.iter().enumerate() {
            let written = spec.inputs[index]
                .iter()
                .map(|label| TemporaryLabel::from(*label))
                .collect::<Vec<_>>();
            if !has_intra_operand_pair(&written) {
                lowering.inputs.push(written);
                lowering.conj.push(spec.conj[index]);
                lowering.splits.push(spec.codomain_splits[index]);
                lowering.traces.push(None);
                continue;
            }
            if written.len() != tensor.rank() {
                return Err(invalid(format!(
                    "operand {index} has {} labels but tensor rank {}",
                    written.len(),
                    tensor.rank()
                )));
            }
            if let Some(split) = spec.codomain_splits[index] {
                if split != tensor.codomain_rank() {
                    return Err(invalid(format!(
                        "operand {index} puts {split} label(s) before `;` but the tensor's codomain rank is {}",
                        tensor.codomain_rank()
                    )));
                }
            }
            let labels = if spec.conj[index] {
                rotate(&written, tensor.codomain_rank())
            } else {
                written
            };
            let (pairs, reduced) = split_trace_pairs(index, &labels)?;
            lowering.inputs.push(reduced);
            lowering.conj.push(false);
            lowering.splits.push(None);
            lowering.traces.push(Some((spec.conj[index], pairs)));
        }
        Ok(lowering)
    }

    /// [`static_operand_preflight`] of the reduced network, before any trace
    /// runs.
    fn preflight<R, D, S>(&self, tensors: &[&TensorMap<R, D, S>]) -> Result<(), HostNetworkError<R>>
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
            &self.splits,
            &self.traces,
            None,
        )
    }

    fn network(&self, spec: &StaticTopologySpec) -> Result<Network, Error> {
        Network::new(
            self.inputs.clone(),
            self.conj.clone(),
            self.splits.clone(),
            spec.output
                .iter()
                .map(|label| TemporaryLabel::from(*label))
                .collect(),
            spec.output_codomain_rank,
        )
    }
}

fn has_intra_operand_pair(labels: &[TemporaryLabel]) -> bool {
    labels
        .iter()
        .enumerate()
        .any(|(i, l)| labels[..i].contains(l))
}

/// Splits an operand's (conj-lowered) labels into intra-operand trace pairs
/// (first occurrence, second occurrence) and the surviving open labels in
/// written order. A label written three or more times on one operand is
/// rejected (the macro already rejects it at compile time; this guards the
/// direct API).
#[expect(
    clippy::type_complexity,
    reason = "the trace parser returns paired axes and surviving labels together"
)]
fn split_trace_pairs(
    operand: usize,
    labels: &[TemporaryLabel],
) -> Result<(Vec<(usize, usize)>, Vec<TemporaryLabel>), Error> {
    let mut pairs: Vec<(usize, usize)> = Vec::new();
    let mut traced = vec![false; labels.len()];
    for (second, label) in labels.iter().enumerate() {
        let occurrences: Vec<usize> = labels[..second]
            .iter()
            .enumerate()
            .filter(|(_, l)| *l == label)
            .map(|(i, _)| i)
            .collect();
        match occurrences.len() {
            0 => {}
            1 => {
                pairs.push((occurrences[0], second));
                traced[occurrences[0]] = true;
                traced[second] = true;
            }
            _ => {
                return Err(invalid(format!(
                    "label `{label}` appears more than twice on operand {operand}"
                )))
            }
        }
    }
    let reduced = labels
        .iter()
        .enumerate()
        .filter(|&(i, _)| !traced[i])
        .map(|(_, l)| l.clone())
        .collect();
    Ok((pairs, reduced))
}
