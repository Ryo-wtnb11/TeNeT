use super::*;

/// Runtime and rule identity of a network's operands, the first check of
/// every lowering and preflight; `None` for no operand.
fn typed_operand_identity<R, D, S>(
    tensors: &[&TensorMap<R, D, S>],
) -> Result<Option<RuleIdentity>, HostNetworkError<R>>
where
    R: TypedSectorAdmission,
    R::Mode: HostNetworkModeDispatch<R, D>,
    D: TensorScalar,
    S: TensorStorage<D>,
{
    let Some(first) = tensors.first() else {
        return Ok(None);
    };
    let runtime = first.runtime().identity();
    let identity = TypedSectorAdmission::typed_rule_identity(first.provider());
    for (index, tensor) in tensors.iter().enumerate().skip(1) {
        if !runtime.matches(tensor.runtime()) {
            return Err(invalid(format!("operand {index} uses a different Runtime")).into());
        }
        if identity != TypedSectorAdmission::typed_rule_identity(tensor.provider()) {
            return Err(Error::RuleMismatch.into());
        }
    }
    Ok(Some(identity))
}

/// [`StaticTopologySpec::contracted`] as the preflight borrows it.
type ContractedAxisPairs<'a> = &'a [&'a [Option<(usize, usize)>]];

/// The metadata preflight of a network, run by `tensor!` before any trace,
/// plan lookup, lease or transfer and by every typed lowering: each operand's
/// written rank and `;` split, then every contracted leg against the dual of
/// its partner. A traced operand (`traces[i]`) enters with its reduced labels
/// and its adjoint-read legs without the traced axes, which are the trace
/// output's legs; `StaticTraceLowering::new` checked its rank and split. The
/// caller decides braiding and on the device placement first.
///
/// Allocation-free, because a warm plan-cache hit runs it on every call and
/// the cache key holds no sectors: legs are borrowed from the operands and
/// compared through the dual map, and no label map is built.
///
/// `contracted` is the pairing a `tensor!` spec resolved at macro expansion
/// ([`StaticTopologySpec::contracted`]), which makes the pass O(N) in the
/// total lowered legs N. Without it — a runtime [`Network`], or a traced
/// lowering, whose pairs are not static — each axis rediscovers its partner
/// by a scan over the earlier axes, which is O(N²) overall.
fn static_operand_preflight<R, D, S, L>(
    tensors: &[&TensorMap<R, D, S>],
    inputs: &[impl AsRef<[L]>],
    conj: &[bool],
    splits: &[Option<usize>],
    traces: &[Option<StaticTrace>],
    contracted: Option<ContractedAxisPairs<'_>>,
) -> Result<(), HostNetworkError<R>>
where
    R: TypedSectorAdmission,
    R::Mode: HostNetworkModeDispatch<R, D>,
    D: TensorScalar,
    S: TensorStorage<D>,
    L: PartialEq + std::fmt::Display,
{
    typed_operand_identity(tensors)?;
    let trace = |operand: usize| traces.get(operand).and_then(Option::as_ref);
    for (index, &tensor) in tensors.iter().enumerate() {
        if trace(index).is_some() {
            continue;
        }
        let labels = inputs[index].as_ref();
        if labels.len() != tensor.rank() {
            return Err(invalid(format!(
                "operand {index} has {} labels but tensor rank {}",
                labels.len(),
                tensor.rank()
            ))
            .into());
        }
        if let Some(split) = splits[index] {
            if split != tensor.codomain_rank() {
                return Err(invalid(format!(
                    "operand {index} puts {split} label(s) before `;` but the tensor's codomain rank is {}",
                    tensor.codomain_rank()
                ))
                .into());
            }
        }
    }
    let Some(first) = tensors.first() else {
        return Ok(());
    };
    // Lowered (conj-rotated, trace-reduced) axis `axis` of `operand`: its label,
    // and the stored leg with whether the lowered leg is that leg's dual.
    let rank = |operand: usize| {
        trace(operand).map_or(tensors[operand].rank(), |_| inputs[operand].as_ref().len())
    };
    // A lowered axis and its written axis are the same, except on a `conj`
    // operand, which is read through its adjoint: lowered axis `axis` is
    // written axis `(axis + codomain_rank) % rank`.
    let rotated = |operand: usize| trace(operand).is_none() && conj[operand];
    let written_axis = |operand: usize, axis: usize| {
        let tensor = tensors[operand];
        if rotated(operand) {
            (axis + tensor.codomain_rank()) % tensor.rank()
        } else {
            axis
        }
    };
    let lowered_axis = |operand: usize, written: usize| {
        let tensor = tensors[operand];
        if rotated(operand) {
            (written + tensor.rank() - tensor.codomain_rank()) % tensor.rank()
        } else {
            written
        }
    };
    let label =
        |operand: usize, axis: usize| &inputs[operand].as_ref()[written_axis(operand, axis)];
    // The partner of a lowered axis found by scanning the earlier axes: the
    // fallback when the caller has no precomputed pairing, and the
    // cross-check of one that it has.
    let scan = |operand: usize, axis: usize| {
        let written = label(operand, axis);
        (0..=operand)
            .flat_map(|previous| {
                let end = if previous == operand {
                    axis
                } else {
                    rank(previous)
                };
                (0..end).map(move |previous_axis| (previous, previous_axis))
            })
            .find(|&(previous, previous_axis)| label(previous, previous_axis) == written)
    };
    let leg = |operand: usize, axis: usize| {
        let tensor = tensors[operand];
        let (adjoint, axis) = match trace(operand) {
            Some((adjoint, pairs)) => (
                *adjoint,
                (0..tensor.rank())
                    .filter(|&kept| !pairs.iter().any(|&(a, b)| kept == a || kept == b))
                    .nth(axis),
            ),
            None => (conj[operand], Some(axis)),
        };
        let codomain_rank = tensor.codomain_rank();
        let missing =
            || HostNetworkError::<R>::from(invalid(format!("operand {operand} has no leg")));
        let axis = axis.ok_or_else(missing)?;
        let source = if adjoint {
            (axis + codomain_rank) % tensor.rank()
        } else {
            axis
        };
        let stored = tensor.network_source_leg(source).ok_or_else(missing)?;
        Ok::<_, HostNetworkError<R>>((stored, (source >= codomain_rank) != adjoint))
    };
    // A pairing this pass does not accept would skip a space check or panic,
    // so it is dropped for the scan here rather than indexed blindly below.
    // Rejected in one O(N) sweep: a shape that is not the operands', an
    // endpoint outside the pairing, and a pair inside one operand — which
    // the written and the lowered order of a `conj` operand enumerate from
    // opposite ends, so its endpoints would not be the scan's. An operand
    // with an intra-operand pair is a trace, and `tensor!` lowers it through
    // `StaticTrace`s and no static pairing.
    let contracted = contracted.filter(|pairs| {
        traces.is_empty()
            && pairs.len() == tensors.len()
            && pairs
                .iter()
                .zip(inputs)
                .enumerate()
                .all(|(operand, (operand_pairs, labels))| {
                    operand_pairs.len() == labels.as_ref().len()
                        && operand_pairs.iter().flatten().all(|&(previous, written)| {
                            previous < operand && written < pairs[previous].len()
                        })
                })
    });
    for operand in 0..tensors.len() {
        for axis in 0..rank(operand) {
            let pair = match contracted {
                Some(pairs) => pairs[operand][written_axis(operand, axis)]
                    .map(|(previous, written)| (previous, lowered_axis(previous, written))),
                None => scan(operand, axis),
            };
            debug_assert!(
                contracted.is_none() || pair == scan(operand, axis),
                "precomputed pairing {pair:?} of operand {operand} lowered axis {axis} \
                 disagrees with the label scan {:?}",
                scan(operand, axis)
            );
            let Some((previous_operand, previous_axis)) = pair else {
                continue;
            };
            let written = label(operand, axis);
            if !legs_contract(
                first.provider(),
                leg(previous_operand, previous_axis)?,
                leg(operand, axis)?,
            )? {
                return Err(invalid(format!(
                    "space mismatch for contracted label `{written}` between operand {previous_operand} leg {previous_axis} and operand {operand} leg {axis}"
                ))
                .into());
            }
        }
    }
    Ok(())
}

/// Whether two lowered legs, each a stored leg and whether it is read as that
/// leg's dual, may be contracted: the second must be the dual of the first,
/// as `GradedSpace::try_dual` would build it. Only the parity of the two
/// dualisations matters: when exactly one side is dualised the stored legs
/// must be equal, which is TensorKit's `space(A, i) == space(B, j)'` with a
/// structural dual.
fn legs_contract<R>(
    provider: &R,
    (lhs, lhs_dualised): (&SectorLeg, bool),
    (rhs, rhs_dualised): (&SectorLeg, bool),
) -> Result<bool, HostNetworkError<R>>
where
    R: TypedSectorAdmission,
    R::Mode: TypedTensorModeDispatch<R>,
{
    if lhs_dualised != rhs_dualised {
        return Ok(lhs == rhs);
    }
    is_dual_of(provider, lhs, rhs)
}

/// `into == dual(from)` for sorted, duplicate-free legs: the opposite duality
/// flag, as many sectors, and the dual of every sector of `from` in `into`
/// with the same degeneracy. The round trip `dual(dual(s)) == s` proves the
/// dual injective on `from`, so its image fills `into` exactly; a provider
/// whose dual is not an involution is rejected with the error
/// `GradedSpace::try_dual` raises for a non-injective dual, not accepted.
fn is_dual_of<R>(
    provider: &R,
    from: &SectorLeg,
    into: &SectorLeg,
) -> Result<bool, HostNetworkError<R>>
where
    R: TypedSectorAdmission,
    R::Mode: TypedTensorModeDispatch<R>,
{
    if from.is_dual() == into.is_dual() || from.sectors().len() != into.sectors().len() {
        return Ok(false);
    }
    let dual_of = |sector| {
        TypedSectorAdmission::try_dual_id(provider, sector)
            .map_err(<R::Mode as TypedTensorModeDispatch<R>>::map_provider_error)
    };
    for (&sector, &degeneracy) in from.sectors().iter().zip(from.degeneracies()) {
        let dual = dual_of(sector)?;
        if dual_of(dual)? != sector {
            return Err(invalid(format!(
                "dual map is not injective: sector {dual:?} appears multiple times"
            ))
            .into());
        }
        match into.sectors().binary_search(&dual) {
            Ok(position) if into.degeneracies()[position] == degeneracy => {}
            _ => return Ok(false),
        }
    }
    Ok(true)
}

/// [`static_operand_preflight`] of an untraced `tensor!` network, as both
/// placements run it before the plan-cache lookup; public only so its
/// allocation-free contract can be tested from a counting-allocator binary.
#[doc(hidden)]
pub fn static_network_operand_preflight<R, D, S>(
    tensors: &[&TensorMap<R, D, S>],
    spec: &StaticTopologySpec,
) -> Result<(), HostNetworkError<R>>
where
    R: TypedSectorAdmission,
    R::Mode: HostNetworkModeDispatch<R, D>,
    D: TensorScalar,
    S: TensorStorage<D>,
{
    static_operand_preflight(
        tensors,
        spec.inputs,
        spec.conj,
        spec.codomain_splits,
        &[],
        Some(spec.contracted),
    )
}

/// The typed `contract`'s braiding boundary (TensorKit `blas_contract!`
/// requires symmetric braiding), decided once for a network whose schedule
/// contracts: the same error the first contraction step would raise, without
/// the permutes, trace pre-step, sliced accumulator or plan lookup that would
/// otherwise run before it.
fn reject_non_symmetric_network<R, D, S>(
    tensors: &[&TensorMap<R, D, S>],
    contracts: bool,
) -> Result<(), HostNetworkError<R>>
where
    R: TypedSectorAdmission,
    R::Mode: HostNetworkModeDispatch<R, D, S>,
    D: TensorScalar,
    S: NetworkPayloadStorage<D>,
{
    let Some(first) = tensors.first().filter(|_| contracts) else {
        return Ok(());
    };
    let braiding = <R::Mode as HostNetworkModeDispatch<R, D, S>>::braiding_style(first.provider());
    tenet::typed::reject_non_symmetric_contraction(braiding)
        .map_err(|error| HostNetworkError::<R>::from(Error::from(error)))
}

fn validate_typed_contracted_pairs<R, D, S>(
    tensors: &[TensorMap<R, D, S>],
    pairs: &[InputLegPair],
) -> Result<(), HostNetworkError<R>>
where
    R: TypedSectorAdmission,
    R::Mode: HostNetworkModeDispatch<R, D, S>,
    D: TensorScalar,
    S: NetworkPayloadStorage<D>,
{
    let spaces = tensors
        .iter()
        .map(typed_flat_spaces)
        .collect::<Result<Vec<_>, _>>()?;
    for &((lhs_slot, lhs_axis), (rhs_slot, rhs_axis)) in pairs {
        if spaces[rhs_slot][rhs_axis] != spaces[lhs_slot][lhs_axis].try_dual()? {
            return Err(invalid(format!(
                "contracted input spaces mismatch between operand {lhs_slot} leg {lhs_axis} and operand {rhs_slot} leg {rhs_axis}"
            ))
            .into());
        }
    }
    Ok(())
}

fn typed_flat_spaces<R, D, S>(
    tensor: &TensorMap<R, D, S>,
) -> Result<Vec<GradedSpace<R>>, HostNetworkError<R>>
where
    R: TypedSectorAdmission,
    R::Mode: TypedTensorModeDispatch<R>,
    D: TensorScalar,
    S: TensorStorage<D>,
{
    let mut spaces = tensor.codomain();
    spaces.extend(
        tensor
            .domain()
            .iter()
            .map(GradedSpace::try_dual)
            .collect::<Result<Vec<_>, _>>()?,
    );
    Ok(spaces)
}

fn typed_effective_spaces<R, D, S>(
    tensor: &TensorMap<R, D, S>,
    adjoint: bool,
) -> Result<Vec<GradedSpace<R>>, HostNetworkError<R>>
where
    R: TypedSectorAdmission,
    R::Mode: TypedTensorModeDispatch<R>,
    D: TensorScalar,
    S: TensorStorage<D>,
{
    if !adjoint {
        return typed_flat_spaces(tensor);
    }
    let mut spaces = tensor.domain();
    spaces.extend(
        tensor
            .codomain()
            .iter()
            .map(GradedSpace::try_dual)
            .collect::<Result<Vec<_>, _>>()?,
    );
    Ok(spaces)
}

fn rotate<T: Clone>(items: &[T], split: usize) -> Vec<T> {
    items[split..]
        .iter()
        .chain(items[..split].iter())
        .cloned()
        .collect()
}
