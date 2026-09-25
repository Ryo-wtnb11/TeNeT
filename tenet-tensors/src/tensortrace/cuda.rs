//! Device executor of a Host-compiled [`TensorTraceFusionStructure`].
//!
//! Nothing categorical happens here: the valid (source tree, destination
//! tree) terms, their coefficients (recoupling row x `dim(c)/dim(a_1)` x the
//! fermionic twist of every non-dual traced leg after the first) and every
//! stride were fixed on the host by the compile the Host trace itself runs.
//! Each descriptor term with a non-zero coefficient becomes one
//! [`cuda_region_trace_accumulate`]: the source block read through one merged
//! diagonal axis per traced pair, contracted against `α′ = alpha *
//! coefficient` repeated (the ones template when `α′ = 1`), so every traced
//! element is scaled before the sum, and accumulated into its destination
//! block. A zero coefficient is skipped and a zero `α′` adds nothing, as on
//! the host (#1438).
//!
//! TensorKit `_trace_permute!` (tensoroperations.jl:242/267 @cfaa073) is the
//! same loop — `scale!(tdst, β)`, then one dense `tensortrace!` per valid tree
//! pair (`iszero(coeff) && continue`) with `α′ = α·coeff`, `β = One()`,
//! which `stridedtensortrace!` applies as `Scaler(α′)` to each element before
//! the `Adder()` reduction — and TensorOperations' strided and
//! cuTENSOR `tensortrace!` build the same diagonal-stride view
//! (`newstrides = strides(q₁) .+ strides(q₂)`). QSpace `QSpace::trace`
//! (QSpace.cc:4487 @dd2cc7e) filters the source blocks whose traced labels
//! match, traces each dense block (`wbarray::trace`, one joint index per pair,
//! wbarray.cc:3558) and adds it into its destination block.

use std::collections::HashSet;
use std::sync::Arc;

use tenet_core::BlockStructure;
use tenet_dense::{cuda_region_trace_accumulate, CudaDenseContext, CudaRegion, CudaScalar};
use tenet_operations::cuda::CudaStorage;
use tenet_operations::CudaTreeTransformExecutor;

use super::{validate_trace_data_extents, TensorTraceFusionStructure};
use crate::{OperationError, RecouplingCoefficientAction};

/// `dst += alpha * trace(src)` over every term of `structure`, on the device.
///
/// Accumulation rule: every term adds (`beta = 1`), so a destination block
/// with several producers — several source blocks, or several recoupling
/// channels of one — receives their sum. The caller passes a zeroed
/// destination for an overwriting trace; this is the Host fallback writer's
/// zero-then-accumulate order (`tensortrace_fusion_dyn_structure_into_raw`
/// with `beta = 0`), and the returning typed trace gets it for free from the
/// output's zero initialisation.
///
/// Every region is built and every structure identity and length checked
/// before the first submission, so a rejected call touches neither buffer.
/// The ones and scaled templates are reserved to the largest traced extent
/// first, so a replay uploads at most once each and a warm one never; each
/// distinct non-unit `α′` costs one device refill of the scaled template. The
/// transform executor reserves plan entries for the distinct term signatures
/// (a high-water mark, under its plan-cache budget), so a warm replay of up
/// to that budget rebuilds no plan.
#[doc(hidden)]
#[allow(clippy::too_many_arguments)]
pub fn tensortrace_fusion_structure_accumulate_on_cuda<C, D>(
    ctx: &mut CudaDenseContext,
    transforms: &mut CudaTreeTransformExecutor,
    structure: &TensorTraceFusionStructure<C>,
    dst_structure: &Arc<BlockStructure>,
    dst: &mut CudaStorage<D>,
    src_structure: &Arc<BlockStructure>,
    src: &CudaStorage<D>,
    alpha: D,
) -> Result<(), OperationError>
where
    C: Copy,
    D: CudaScalar + RecouplingCoefficientAction<C>,
{
    structure.validate_replay_structures(dst_structure, src_structure)?;
    validate_trace_data_extents(dst_structure, dst.0.len(), src_structure, src.0.len())?;
    let descriptor = structure.descriptor();
    if descriptor.terms().len() != structure.terms().len() {
        return Err(OperationError::CoefficientCountMismatch {
            expected: descriptor.terms().len(),
            actual: structure.terms().len(),
        });
    }

    let mut moves = Vec::with_capacity(descriptor.terms().len());
    let mut largest_unit = 0usize;
    let mut largest_scaled = 0usize;
    for (term, fusion_term) in descriptor.terms().iter().zip(structure.terms()) {
        if term.dst_block != fusion_term.dst_block() || term.src_block != fusion_term.src_block() {
            return Err(OperationError::StructureMismatch {
                tensor: "trace term",
            });
        }
        let output_shape = descriptor.output_shape(term);
        let trace_shape = descriptor.trace_shape(term);
        let mut src_dims = Vec::with_capacity(output_shape.len() + trace_shape.len());
        src_dims.extend_from_slice(output_shape);
        src_dims.extend_from_slice(trace_shape);
        let src_strides = descriptor
            .src_output_strides(term)
            .iter()
            .chain(descriptor.src_trace_strides(term))
            .map(|&stride| unsigned(stride))
            .collect::<Result<Vec<_>, _>>()?;
        let dst_strides = descriptor
            .dst_strides(term)
            .iter()
            .map(|&stride| unsigned(stride))
            .collect::<Result<Vec<_>, _>>()?;
        let src_region = CudaRegion::new(src_dims, src_strides, unsigned(term.src_offset)?)
            .map_err(OperationError::Dense)?;
        let dst_region = CudaRegion::new(
            output_shape.to_vec(),
            dst_strides,
            unsigned(term.dst_offset)?,
        )
        .map_err(OperationError::Dense)?;
        let trace_len = trace_shape
            .iter()
            .try_fold(1usize, |count, &dim| count.checked_mul(dim))
            .ok_or(OperationError::ElementCountOverflow)?;
        if D::coefficient_as_data(fusion_term.coefficient) == D::ZERO {
            continue;
        }
        let scale = alpha.scale_by_coefficient(fusion_term.coefficient);
        if scale == D::ONE {
            largest_unit = largest_unit.max(trace_len);
        } else if scale != D::ZERO {
            largest_scaled = largest_scaled.max(trace_len);
        }
        moves.push((src_region, dst_region, scale, trace_len));
    }

    // A plan is keyed on the three operand layouts: the source and destination
    // regions (offsets excepted, alignment being `size_of::<D>()` for every
    // view) and the ones view, which the source's traced extents fix. The
    // conjugation flag and the scale are the same for every term or not in
    // the key, and a scaled term reads the scaled template through a view of
    // the same metadata as the ones template.
    // Each refill of the scaled template is one more packed `[len]` move.
    let fills = moves
        .iter()
        .filter(|(_, _, scale, _)| *scale != D::ONE)
        .map(|&(_, _, _, len)| len)
        .collect::<HashSet<_>>()
        .len();
    let signatures = moves
        .iter()
        .map(|(src, dst, _, _)| (src.dims(), src.strides(), dst.strides()))
        .collect::<HashSet<_>>()
        .len()
        + fills;
    transforms.reserve_plan_entries_for_additional(ctx, signatures)?;
    ctx.reserve_ones_template::<D>(largest_unit)
        .map_err(OperationError::Dense)?;
    ctx.reserve_scaled_template::<D>(largest_scaled)
        .map_err(OperationError::Dense)?;
    // Terms sharing `α′` run together, longest traced extent first, so the
    // scaled template is refilled once per distinct `α′`: one sort on the
    // bit pattern, O(T log T). The order of terms adding into one
    // destination changes rounding only.
    let mut order: Vec<usize> = (0..moves.len()).collect();
    order.sort_by_key(|&term| {
        let (_, _, scale, len) = &moves[term];
        (scale.bit_pattern(), std::cmp::Reverse(*len))
    });
    let conjugate = descriptor.source_conjugate();
    for (src_region, dst_region, scale, _) in order.into_iter().map(|term| &moves[term]) {
        cuda_region_trace_accumulate::<D>(
            ctx, &src.0, src_region, conjugate, *scale, &mut dst.0, dst_region,
        )
        .map_err(OperationError::Dense)?;
    }
    Ok(())
}

/// Host block strides and offsets are `usize`, so a negative descriptor entry
/// cannot come from a compiled structure; it is reported, never wrapped.
fn unsigned(value: isize) -> Result<usize, OperationError> {
    usize::try_from(value).map_err(|_| OperationError::InvalidArgument {
        message: "trace descriptor stride or offset is negative",
    })
}
