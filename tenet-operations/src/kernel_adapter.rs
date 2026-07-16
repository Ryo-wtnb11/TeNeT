use core::ops::{Add, Mul};
use std::cell::RefCell;

use num_traits::{One, Zero};
use strided_kernel::{
    axpy_conj_raw, axpy_raw, copy_scale_conj_raw, copy_scale_raw, RawStridedMut, RawStridedRef,
};

use crate::{
    axpby_raw_strided_kernel_trusted, scale_raw_strided_kernel_trusted,
    tensoradd_raw_strided_kernel_trusted, ConjugateValue, OperationError,
    RecouplingCoefficientAction,
};

const FUSED_RANK_LIMIT: usize = 8;

/// Transpose kernel selection for pure permuted copies (pack / assign-scatter)
/// in [`StridedHostKernelAdapter`], chosen per-runtime via
/// `Runtime::builder().transpose_backend(...)` (see `docs/backend_policy.md`).
/// Backend choice is a performance knob only — routed copies are byte-identical
/// across backends (checksum-verified in the #114 A/B).
///
/// `StridedPerm` is **opt-in, not the default**, on measured numbers (issue
/// #114, ported from prototype commit b3ca6e5): its per-call plan build loses
/// badly on the many tiny blocks of small-degeneracy SU(2) replay (d=4 swap
/// +92%, swap+out +95%; profile: pack x6.8, scatter x6.1) and only wins above
/// noise on large-block abelian transposes (fZ2 swap+out d=16 -6.5%). The name
/// says `strided_perm` rather than HPTT because the kernel is `strided_perm`'s
/// HPTT-*inspired* col-major copy reached through `strided-kernel` (already a
/// dependency), not a literal HPTT (Springer et al.) binding.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum TransposeBackend {
    /// The zero-alloc fused loop nest (the default; byte- and
    /// dispatch-identical to the pre-#114 behavior).
    #[default]
    FusedLoops,
    /// `strided_perm`'s HPTT-style blocked micro-kernel transpose, applied to
    /// eligible pure permuted copies; ineligible layouts fall back to the
    /// fused loop.
    StridedPerm,
}

/// strided-perm-backed permuted copy `dst = src` over strided views (no scale,
/// no conjugate). Returns `Ok(true)` when the strided-perm route handled the
/// copy, `Ok(false)` when the layout is outside its supported class and the
/// caller must fall back to the fused loop.
///
/// The transpose micro-kernel assumes each side has a genuine stride-1 axis
/// (the classic row-major↔transpose case). A pack that gathers a strided
/// *slice* of a larger storage can have no stride-1 axis once extent-1 axes
/// are dropped (its contiguous axis was a singleton) — the kernel would then
/// treat the smallest-stride axis as if it were stride-1 and silently corrupt.
/// We detect that and decline. (Guard rationale from prototype b3ca6e5: this
/// exact pattern corrupted the energy -1.803 vs -1.772 before the guard.)
#[allow(clippy::too_many_arguments)]
fn strided_perm_copy<T: Copy + strided_kernel::MaybeSendSync>(
    dst_data: &mut [T],
    src_data: &[T],
    shape: &[usize],
    dst_strides: &[isize],
    src_strides: &[isize],
    dst_offset: isize,
    src_offset: isize,
) -> Result<bool, OperationError> {
    // Drop extent-1 axes (their strides are irrelevant, and the planner is not
    // robust to extent-1 axes carrying colliding strides). `fused_pair` does the
    // same normalization.
    use smallvec::SmallVec;
    let mut rshape: SmallVec<[usize; 8]> = SmallVec::new();
    let mut rdst: SmallVec<[isize; 8]> = SmallVec::new();
    let mut rsrc: SmallVec<[isize; 8]> = SmallVec::new();
    for axis in 0..shape.len() {
        if shape[axis] != 1 {
            rshape.push(shape[axis]);
            rdst.push(dst_strides[axis]);
            rsrc.push(src_strides[axis]);
        }
    }
    if rshape.is_empty() {
        // single element (all axes extent 1)
        dst_data[dst_offset as usize] = src_data[src_offset as usize];
        return Ok(true);
    }
    // Eligibility: each side needs a real stride-1 axis for the micro-kernel.
    if !rsrc.contains(&1) || !rdst.contains(&1) {
        return Ok(false);
    }
    // View construction validates the same bounding box fused_pair accesses
    // (max index = offset + Σ(dim-1)·stride < len), so with the positive-stride
    // guard it only errors on a genuine out-of-bounds layout — an upstream bug.
    // Propagating that as a clean error beats declining into a fused panic.
    let src = strided_kernel::StridedView::new(src_data, &rshape, &rsrc, src_offset)
        .map_err(crate::strided::error)?;
    let mut dst = strided_kernel::StridedViewMut::new(dst_data, &rshape, &rdst, dst_offset)
        .map_err(crate::strided::error)?;
    strided_kernel::copy_into_col_major(&mut dst, &src).map_err(crate::strided::error)?;
    Ok(true)
}

/// Allocation-free fused loop layout for one (destination, source) view pair.
///
/// Axes with extent 1 are dropped, the rest are ordered by destination stride
/// and adjacent axes are fused when both stride patterns are contiguous, so
/// small replay copies avoid per-call heap allocation and plan building.
#[derive(Clone, Copy, Debug)]
pub(crate) struct FusedPairLayout {
    pub(crate) rank: usize,
    pub(crate) dims: [usize; FUSED_RANK_LIMIT],
    pub(crate) dst_strides: [isize; FUSED_RANK_LIMIT],
    pub(crate) src_strides: [isize; FUSED_RANK_LIMIT],
}

/// Borrowed view of a prebaked fused loop layout (issue #232).
///
/// Holds the exact `(dims, dst_strides, src_strides)` that [`fuse_pair_layout`]
/// would return for one (block, role) stride pair, computed once at compile
/// time in the immutable `TreeTransformLayoutTable` and reused across every
/// replay call instead of recomputed. The slices live in that table's arena;
/// `apply_fused_pair_slices` consumes them directly. dtype-independent — one
/// baked layout serves f64 and c64 alike (the normalization never inspects
/// values).
#[derive(Clone, Copy, Debug)]
pub struct BakedFusedLayout<'a> {
    pub dims: &'a [usize],
    pub dst_strides: &'a [isize],
    pub src_strides: &'a [isize],
}

pub(crate) fn fuse_pair_layout(
    shape: &[usize],
    dst_strides: &[isize],
    src_strides: &[isize],
) -> Option<FusedPairLayout> {
    if shape.len() > FUSED_RANK_LIMIT {
        return None;
    }
    let mut layout = FusedPairLayout {
        rank: 0,
        dims: [1; FUSED_RANK_LIMIT],
        dst_strides: [0; FUSED_RANK_LIMIT],
        src_strides: [0; FUSED_RANK_LIMIT],
    };
    for axis in 0..shape.len() {
        if shape[axis] == 1 {
            continue;
        }
        if shape[axis] == 0 {
            return Some(FusedPairLayout {
                rank: 1,
                dims: [0; FUSED_RANK_LIMIT],
                dst_strides: [0; FUSED_RANK_LIMIT],
                src_strides: [0; FUSED_RANK_LIMIT],
            });
        }
        let mut position = layout.rank;
        while position > 0 && layout.dst_strides[position - 1] > dst_strides[axis] {
            layout.dims[position] = layout.dims[position - 1];
            layout.dst_strides[position] = layout.dst_strides[position - 1];
            layout.src_strides[position] = layout.src_strides[position - 1];
            position -= 1;
        }
        layout.dims[position] = shape[axis];
        layout.dst_strides[position] = dst_strides[axis];
        layout.src_strides[position] = src_strides[axis];
        layout.rank += 1;
    }
    if layout.rank == 0 {
        layout.rank = 1;
        layout.dims[0] = 1;
    }
    let mut fused = 0usize;
    for axis in 1..layout.rank {
        let extent = layout.dims[fused] as isize;
        if layout.dst_strides[fused] * extent == layout.dst_strides[axis]
            && layout.src_strides[fused] * extent == layout.src_strides[axis]
        {
            layout.dims[fused] *= layout.dims[axis];
        } else {
            fused += 1;
            layout.dims[fused] = layout.dims[axis];
            layout.dst_strides[fused] = layout.dst_strides[axis];
            layout.src_strides[fused] = layout.src_strides[axis];
        }
    }
    layout.rank = fused + 1;
    Some(layout)
}

/// Applies `dst = apply(dst, op(src))` over a fixed-capacity stack layout with a
/// plain loop nest. Test-only since #41 moved the non-baked replay onto the
/// strided-rs raw kernels; retained as the byte-identical reference the fused /
/// strided-perm parity tests compare against (it drives the same
/// `apply_fused_pair_slices` loop the baked #232 replay uses in production).
#[cfg(test)]
#[allow(clippy::too_many_arguments)]
fn apply_fused_pair<T, Apply, ElementOp>(
    dst_data: &mut [T],
    src_data: &[T],
    layout: &FusedPairLayout,
    dst_offset: isize,
    src_offset: isize,
    apply: Apply,
    op: ElementOp,
) where
    T: Copy,
    Apply: Fn(&mut T, T),
    ElementOp: Fn(T) -> T,
{
    apply_fused_pair_slices(
        dst_data,
        src_data,
        &layout.dims[..layout.rank],
        &layout.dst_strides[..layout.rank],
        &layout.src_strides[..layout.rank],
        dst_offset,
        src_offset,
        apply,
        op,
    );
}

/// Loop-nest core of [`apply_fused_pair`] over borrowed layout slices, so both
/// the freshly-recomputed [`FusedPairLayout`] (stack arrays) and a prebaked
/// [`BakedFusedLayout`] (arena slices, issue #232) drive the identical kernel.
/// The slices are already normalized (extent-1 axes dropped, ordered by
/// destination stride, contiguous runs fused); rank == `dims.len()` and is
/// bounded by `FUSED_RANK_LIMIT`.
#[allow(clippy::too_many_arguments)]
fn apply_fused_pair_slices<T, Apply, ElementOp>(
    dst_data: &mut [T],
    src_data: &[T],
    dims: &[usize],
    dst_strides: &[isize],
    src_strides: &[isize],
    dst_offset: isize,
    src_offset: isize,
    apply: Apply,
    op: ElementOp,
) where
    T: Copy,
    Apply: Fn(&mut T, T),
    ElementOp: Fn(T) -> T,
{
    let rank = dims.len();
    debug_assert!(rank <= FUSED_RANK_LIMIT);
    if rank == 0 || dims.iter().any(|&dim| dim == 0) {
        return;
    }
    let inner_len = dims[0];
    let inner_dst = dst_strides[0];
    let inner_src = src_strides[0];
    let mut index = [0usize; FUSED_RANK_LIMIT];
    let mut dst_base = dst_offset;
    let mut src_base = src_offset;
    loop {
        if inner_dst == 1 && inner_src == 1 {
            let dst_start = dst_base as usize;
            let src_start = src_base as usize;
            let dst = &mut dst_data[dst_start..dst_start + inner_len];
            let src = &src_data[src_start..src_start + inner_len];
            for position in 0..inner_len {
                apply(&mut dst[position], op(src[position]));
            }
        } else {
            for position in 0..inner_len {
                let dst_position = (dst_base + position as isize * inner_dst) as usize;
                let src_position = (src_base + position as isize * inner_src) as usize;
                apply(&mut dst_data[dst_position], op(src_data[src_position]));
            }
        }
        let mut axis = 1;
        loop {
            if axis >= rank {
                return;
            }
            index[axis] += 1;
            dst_base += dst_strides[axis];
            src_base += src_strides[axis];
            if index[axis] < dims[axis] {
                break;
            }
            dst_base -= dims[axis] as isize * dst_strides[axis];
            src_base -= dims[axis] as isize * src_strides[axis];
            index[axis] = 0;
            axis += 1;
        }
    }
}

thread_local! {
    /// Reused fused-loop scratch for the rank > FUSED_RANK_LIMIT tail only.
    /// Those high-rank contraction intermediates dominate warm replay, and
    /// reusing one buffer per thread keeps them alloc-free after warmup (warm
    /// chi16 -58%, chi32 -64%; commit 12748cf). Rank <= FUSED_RANK_LIMIT never
    /// reaches this path — it delegates to the strided-rs #140 raw kernels.
    ///
    /// Why-not delegate rank > 8 too (issue #41): strided-rs's raw kernels fall
    /// back to the *allocating* view kernels above RAW_FUSED_RANK_LIMIT, which
    /// would break the warm zero-alloc contract pinned by tenet-network's
    /// rank_nine_cached_permutation_has_no_caller_thread_operation_allocation.
    /// This path goes away when strided-rs grows an alloc-free rank>8 route.
    static FUSE_SCRATCH: RefCell<FuseScratch> = const { RefCell::new(FuseScratch::new()) };
}

#[derive(Default)]
struct FuseScratch {
    dims: Vec<usize>,
    dst_strides: Vec<isize>,
    src_strides: Vec<isize>,
    index: Vec<usize>,
}

impl FuseScratch {
    const fn new() -> Self {
        Self {
            dims: Vec::new(),
            dst_strides: Vec::new(),
            src_strides: Vec::new(),
            index: Vec::new(),
        }
    }
}

/// Hand-rolled rank > FUSED_RANK_LIMIT strided walk over the reused
/// thread_local scratch (the pre-#41 `fused_pair` scratch branch, kept only for
/// this tail; see [`FUSE_SCRATCH`] for why it cannot delegate yet). Runs the
/// identical layout algorithm as the delegated rank <= 8 path (extent-1 axes
/// dropped, axes ordered by destination stride, adjacent contiguous axes
/// fused), so the produced values are byte-identical across the rank split.
#[allow(clippy::too_many_arguments)]
fn fused_pair_high_rank<T, Apply, ElementOp>(
    dst_data: &mut [T],
    src_data: &[T],
    shape: &[usize],
    dst_strides: &[isize],
    src_strides: &[isize],
    dst_offset: isize,
    src_offset: isize,
    apply: Apply,
    op: ElementOp,
) where
    T: Copy,
    Apply: Fn(&mut T, T),
    ElementOp: Fn(T) -> T,
{
    FUSE_SCRATCH.with(|cell| {
        let mut scratch = cell.borrow_mut();
        let FuseScratch {
            dims,
            dst_strides: fdst,
            src_strides: fsrc,
            index,
        } = &mut *scratch;
        dims.clear();
        fdst.clear();
        fsrc.clear();
        // insertion sort by destination stride, dropping extent-1 axes
        for axis in 0..shape.len() {
            let extent = shape[axis];
            if extent == 1 {
                continue;
            }
            let d = dst_strides[axis];
            let s = src_strides[axis];
            let mut position = dims.len();
            while position > 0 && fdst[position - 1] > d {
                position -= 1;
            }
            dims.insert(position, extent);
            fdst.insert(position, d);
            fsrc.insert(position, s);
        }
        if dims.is_empty() {
            dims.push(1);
            fdst.push(0);
            fsrc.push(0);
        }
        // fuse adjacent contiguous axes
        let mut fused = 0usize;
        for axis in 1..dims.len() {
            let extent = dims[fused] as isize;
            if fdst[fused] * extent == fdst[axis] && fsrc[fused] * extent == fsrc[axis] {
                dims[fused] *= dims[axis];
            } else {
                fused += 1;
                dims[fused] = dims[axis];
                fdst[fused] = fdst[axis];
                fsrc[fused] = fsrc[axis];
            }
        }
        let rank = fused + 1;
        dims.truncate(rank);
        fdst.truncate(rank);
        fsrc.truncate(rank);
        index.clear();
        index.resize(rank, 0);

        let inner_len = dims[0];
        let inner_dst = fdst[0];
        let inner_src = fsrc[0];
        let mut dst_base = dst_offset;
        let mut src_base = src_offset;
        loop {
            if inner_dst == 1 && inner_src == 1 {
                let dst_start = dst_base as usize;
                let src_start = src_base as usize;
                let dst = &mut dst_data[dst_start..dst_start + inner_len];
                let src = &src_data[src_start..src_start + inner_len];
                for position in 0..inner_len {
                    apply(&mut dst[position], op(src[position]));
                }
            } else {
                for position in 0..inner_len {
                    let dst_position = (dst_base + position as isize * inner_dst) as usize;
                    let src_position = (src_base + position as isize * inner_src) as usize;
                    apply(&mut dst_data[dst_position], op(src_data[src_position]));
                }
            }
            let mut axis = 1;
            loop {
                if axis >= rank {
                    return;
                }
                index[axis] += 1;
                dst_base += fdst[axis];
                src_base += fsrc[axis];
                if index[axis] < dims[axis] {
                    break;
                }
                dst_base -= dims[axis] as isize * fdst[axis];
                src_base -= dims[axis] as isize * fsrc[axis];
                index[axis] = 0;
                axis += 1;
            }
        }
    });
}

/// Non-baked rank <= FUSED_RANK_LIMIT strided copy/scale/axpy, delegated to
/// strided-rs #140 raw kernels (issue #41). The four
/// `(assign | accumulate) x (plain | conj)` combinations map 1:1 onto
/// `copy_scale_raw` / `copy_scale_conj_raw` / `axpy_raw` / `axpy_conj_raw`,
/// which run the *identical* fuse-order-odometer walk TeNeT used to hand-roll
/// (strided-rs #139 ported this crate's `fuse_pair_layout` /
/// `apply_fused_pair` verbatim), so results are byte-identical — pinned by the
/// differential tests below — and allocation-free for rank <= 8. Callers route
/// rank > FUSED_RANK_LIMIT to [`fused_pair_high_rank`] instead (see
/// [`FUSE_SCRATCH`] for why that tail cannot delegate yet).
///
/// Accepted constant-factor cost (issue #41, measured in
/// `bench_41_fused_vs_raw_vs_plan`): the checked `RawStrided::new` re-validates
/// the bounding box per call, ~+30% on 21x rank-4 d=4 blocks vs the removed
/// hand-rolled loop. Same complexity order; the duplicated kernel code it
/// deletes is worth the constant. The warm hot path is unaffected (it takes the
/// #232 baked route).
///
/// Why `new` (checked) and not `new_unchecked`: the removed hand-rolled loop
/// relied on safe slice indexing to turn an out-of-bounds layout into a *panic*
/// rather than UB. `RawStrided{Ref,Mut}::new` runs the same O(rank) bounding-box
/// test the trusted axpby path already runs in debug and returns a clean `Err`
/// on OOB, which the adapter propagates — strictly safer than a panic and never
/// reached by a valid replay layout. (Observable change vs the old panic is
/// documented at the one probe test that pinned it.)
#[allow(clippy::too_many_arguments)]
fn delegate_raw_copy<T>(
    dst_data: &mut [T],
    src_data: &[T],
    shape: &[usize],
    dst_strides: &[isize],
    src_strides: &[isize],
    dst_offset: isize,
    src_offset: isize,
    assign: bool,
    source_conjugate: bool,
    alpha: T,
) -> Result<(), OperationError>
where
    T: Copy
        + Add<T, Output = T>
        + Mul<T, Output = T>
        + ConjugateValue
        + strided_kernel::MaybeSendSync,
{
    let src = RawStridedRef::new(src_data, shape, src_strides, src_offset)
        .map_err(crate::strided::error)?;
    let mut dst = RawStridedMut::new(dst_data, shape, dst_strides, dst_offset)
        .map_err(crate::strided::error)?;
    // ConjugateValue::maybe_conj(true) is exactly ElementOpApply::conj for the
    // scalar types (real = identity, complex = num-complex conj), so branching
    // on source_conjugate onto the plain/conj raw kernels reproduces the old
    // `alpha * value.maybe_conj(source_conjugate)` element op.
    match (assign, source_conjugate) {
        (true, false) => copy_scale_raw(&mut dst, &src, alpha),
        (true, true) => copy_scale_conj_raw(&mut dst, &src, alpha),
        (false, false) => axpy_raw(&mut dst, &src, alpha),
        (false, true) => axpy_conj_raw(&mut dst, &src, alpha),
    }
    .map_err(crate::strided::error)
}

/// `dst = alpha * op(src)` (assign) or `dst += alpha * op(src)` (accumulate)
/// over one strided pair, taking the issue-#232 prebaked layout when present.
///
/// `Some(baked)`: the compile-time-normalized arena slices drive
/// `apply_fused_pair_slices` directly — strided-rs keeps its `fuse_pair_layout`
/// / `apply_fused_pair` `pub(crate)`, so the baked replay cannot delegate and
/// stays hand-rolled here (that is why `fuse_pair_layout` / `FusedPairLayout`
/// remain in this module). `None`: delegate to the #140 raw kernels via
/// [`delegate_raw_copy`]. The zero-extent short-circuit stays ahead of both so a
/// baked empty marker and an unbaked empty shape are both no-ops.
#[allow(clippy::too_many_arguments)]
fn copy_or_axpy<T>(
    baked: Option<BakedFusedLayout<'_>>,
    dst_data: &mut [T],
    src_data: &[T],
    shape: &[usize],
    dst_strides: &[isize],
    src_strides: &[isize],
    dst_offset: isize,
    src_offset: isize,
    assign: bool,
    source_conjugate: bool,
    alpha: T,
) -> Result<(), OperationError>
where
    T: Copy
        + Add<T, Output = T>
        + Mul<T, Output = T>
        + ConjugateValue
        + strided_kernel::MaybeSendSync,
{
    if shape.contains(&0) {
        return Ok(());
    }
    match baked {
        Some(baked) => {
            apply_fused_pair_slices(
                dst_data,
                src_data,
                baked.dims,
                baked.dst_strides,
                baked.src_strides,
                dst_offset,
                src_offset,
                move |dst: &mut T, value: T| {
                    if assign {
                        *dst = value;
                    } else {
                        *dst = *dst + value;
                    }
                },
                move |value: T| alpha * value.maybe_conj(source_conjugate),
            );
            Ok(())
        }
        None if shape.len() > FUSED_RANK_LIMIT => {
            // Rank > 8 stays hand-rolled: the #140 raw kernels would fall back
            // to allocating view kernels here, breaking the warm zero-alloc
            // contract (see FUSE_SCRATCH).
            fused_pair_high_rank(
                dst_data,
                src_data,
                shape,
                dst_strides,
                src_strides,
                dst_offset,
                src_offset,
                move |dst: &mut T, value: T| {
                    if assign {
                        *dst = value;
                    } else {
                        *dst = *dst + value;
                    }
                },
                move |value: T| alpha * value.maybe_conj(source_conjugate),
            );
            Ok(())
        }
        None => delegate_raw_copy(
            dst_data,
            src_data,
            shape,
            dst_strides,
            src_strides,
            dst_offset,
            src_offset,
            assign,
            source_conjugate,
            alpha,
        ),
    }
}

/// Backend-neutral low-level kernel adapter for host-slice replay.
///
/// Replay drivers (tree-transform pack/recoupling/scatter, fusion-block
/// pack/scatter/scale) call these primitives instead of concrete kernel
/// functions, so the low-level execution backend (scalar loops, strided-rs,
/// BLAS, future C++ kernels) is replaceable behind one boundary.
///
/// The data contract is host slices. Device replay needs a separate
/// storage-aware adapter; device storage must not be hidden behind this trait.
pub trait HostKernelAdapter<T> {
    /// `dst = alpha * op(src) + beta * dst` over strided views, where `op` is
    /// conjugation when `source_conjugate` is set (tensoradd / single-block
    /// tree replay primitive).
    #[allow(clippy::too_many_arguments)]
    fn add_strided(
        &mut self,
        zero_strides: &mut Vec<isize>,
        dst_data: &mut [T],
        src_data: &[T],
        shape: &[usize],
        dst_strides: &[isize],
        src_strides: &[isize],
        dst_offset: isize,
        src_offset: isize,
        source_conjugate: bool,
        alpha: T,
        beta: T,
    ) -> Result<(), OperationError>;

    /// `dst = alpha * src + beta * dst` over strided views without
    /// conjugation (scatter primitive).
    #[allow(clippy::too_many_arguments)]
    fn axpby_strided(
        &mut self,
        dst_data: &mut [T],
        src_data: &[T],
        shape: &[usize],
        dst_strides: &[isize],
        src_strides: &[isize],
        dst_offset: isize,
        src_offset: isize,
        alpha: T,
        beta: T,
    ) -> Result<(), OperationError>;

    /// `dst = alpha * op(src)` over strided views (pack primitive).
    #[allow(clippy::too_many_arguments)]
    fn copy_scale_strided(
        &mut self,
        dst_data: &mut [T],
        src_data: &[T],
        shape: &[usize],
        dst_strides: &[isize],
        src_strides: &[isize],
        dst_offset: isize,
        src_offset: isize,
        source_conjugate: bool,
        alpha: T,
    ) -> Result<(), OperationError>;

    /// [`add_strided`](Self::add_strided) with an optional prebaked fused layout
    /// (issue #232). The default ignores `baked` and forwards to `add_strided`,
    /// so adapters that do not fuse (test doubles) need no change; the strided
    /// host adapter overrides it to skip `fuse_pair_layout` on the `beta ∈ {0,1}`
    /// fast path. `baked` is a pure function of the (block, role) stride pair, so
    /// it is dtype-independent and correctness-neutral versus recomputation.
    #[allow(clippy::too_many_arguments)]
    fn add_strided_baked(
        &mut self,
        zero_strides: &mut Vec<isize>,
        dst_data: &mut [T],
        src_data: &[T],
        shape: &[usize],
        dst_strides: &[isize],
        src_strides: &[isize],
        dst_offset: isize,
        src_offset: isize,
        source_conjugate: bool,
        alpha: T,
        beta: T,
        baked: Option<BakedFusedLayout<'_>>,
    ) -> Result<(), OperationError> {
        let _ = baked;
        self.add_strided(
            zero_strides,
            dst_data,
            src_data,
            shape,
            dst_strides,
            src_strides,
            dst_offset,
            src_offset,
            source_conjugate,
            alpha,
            beta,
        )
    }

    /// [`axpby_strided`](Self::axpby_strided) with an optional prebaked fused
    /// layout (issue #232). See [`add_strided_baked`](Self::add_strided_baked)
    /// for the default/override contract.
    #[allow(clippy::too_many_arguments)]
    fn axpby_strided_baked(
        &mut self,
        dst_data: &mut [T],
        src_data: &[T],
        shape: &[usize],
        dst_strides: &[isize],
        src_strides: &[isize],
        dst_offset: isize,
        src_offset: isize,
        alpha: T,
        beta: T,
        baked: Option<BakedFusedLayout<'_>>,
    ) -> Result<(), OperationError> {
        let _ = baked;
        self.axpby_strided(
            dst_data,
            src_data,
            shape,
            dst_strides,
            src_strides,
            dst_offset,
            src_offset,
            alpha,
            beta,
        )
    }

    /// [`copy_scale_strided`](Self::copy_scale_strided) with an optional
    /// prebaked fused layout (issue #232). See
    /// [`add_strided_baked`](Self::add_strided_baked) for the default/override
    /// contract.
    #[allow(clippy::too_many_arguments)]
    fn copy_scale_strided_baked(
        &mut self,
        dst_data: &mut [T],
        src_data: &[T],
        shape: &[usize],
        dst_strides: &[isize],
        src_strides: &[isize],
        dst_offset: isize,
        src_offset: isize,
        source_conjugate: bool,
        alpha: T,
        baked: Option<BakedFusedLayout<'_>>,
    ) -> Result<(), OperationError> {
        let _ = baked;
        self.copy_scale_strided(
            dst_data,
            src_data,
            shape,
            dst_strides,
            src_strides,
            dst_offset,
            src_offset,
            source_conjugate,
            alpha,
        )
    }

    /// `dst = beta * dst` over a strided block (inactive-block scale
    /// primitive).
    fn scale_strided(
        &mut self,
        dst_data: &mut [T],
        shape: &[usize],
        dst_strides: &[isize],
        dst_offset: isize,
        beta: T,
    ) -> Result<(), OperationError>;

    /// `destination[:, dst] = Σ_src coefficient[dst, src] * source[:, src]`
    /// over packed tree columns.
    ///
    /// TensorKit's dense-vector GenericTreeTransformer uses `U[dst, src]` and
    /// computes `buffer_dst = buffer_src * transpose(U)` after packing source
    /// trees as columns. This is the BLAS/GEMM replacement point for the
    /// recoupling matrix application.
    #[allow(clippy::too_many_arguments)]
    fn recoupling_src_times_u_transpose<C>(
        &mut self,
        destination: &mut [T],
        source: &[T],
        recoupling_coefficients_dst_src: &[C],
        coefficient_start: usize,
        element_count: usize,
        src_count: usize,
        dst_count: usize,
    ) -> Result<(), OperationError>
    where
        C: Copy,
        T: RecouplingCoefficientAction<C>;
}

/// Default host kernel adapter backed by the strided-rs style raw kernels.
///
/// The recoupling matrix application is currently a scalar loop; swapping it
/// for a BLAS/GEMM call happens by replacing this adapter, not by editing the
/// replay drivers.
#[derive(Clone, Copy, Debug, Default)]
pub struct StridedHostKernelAdapter {
    /// Selected transpose kernel for pure permuted copies (pack /
    /// assign-scatter); [`TransposeBackend::FusedLoops`] by default. See
    /// [`TransposeBackend`] for why `StridedPerm` is opt-in.
    pub transpose_backend: TransposeBackend,
}

impl StridedHostKernelAdapter {
    /// Adapter with an explicit transpose kernel; `Default` is `FusedLoops`.
    #[inline]
    pub fn with_transpose_backend(transpose_backend: TransposeBackend) -> Self {
        Self { transpose_backend }
    }
}

impl<T> HostKernelAdapter<T> for StridedHostKernelAdapter
where
    T: Copy
        + Add<T, Output = T>
        + Mul<T, Output = T>
        + PartialEq
        + Zero
        + One
        + ConjugateValue
        + strided_kernel::MaybeSendSync,
{
    fn add_strided(
        &mut self,
        zero_strides: &mut Vec<isize>,
        dst_data: &mut [T],
        src_data: &[T],
        shape: &[usize],
        dst_strides: &[isize],
        src_strides: &[isize],
        dst_offset: isize,
        src_offset: isize,
        source_conjugate: bool,
        alpha: T,
        beta: T,
    ) -> Result<(), OperationError> {
        self.add_strided_baked(
            zero_strides,
            dst_data,
            src_data,
            shape,
            dst_strides,
            src_strides,
            dst_offset,
            src_offset,
            source_conjugate,
            alpha,
            beta,
            None,
        )
    }

    fn add_strided_baked(
        &mut self,
        zero_strides: &mut Vec<isize>,
        dst_data: &mut [T],
        src_data: &[T],
        shape: &[usize],
        dst_strides: &[isize],
        src_strides: &[isize],
        dst_offset: isize,
        src_offset: isize,
        source_conjugate: bool,
        alpha: T,
        beta: T,
        baked: Option<BakedFusedLayout<'_>>,
    ) -> Result<(), OperationError> {
        if beta.is_zero() || beta.is_one() {
            let assign = beta.is_zero();
            copy_or_axpy(
                baked,
                dst_data,
                src_data,
                shape,
                dst_strides,
                src_strides,
                dst_offset,
                src_offset,
                assign,
                source_conjugate,
                alpha,
            )?;
            zero_strides.clear();
            return Ok(());
        }
        tensoradd_raw_strided_kernel_trusted(
            zero_strides,
            dst_data,
            src_data,
            shape,
            dst_strides,
            src_strides,
            dst_offset,
            src_offset,
            source_conjugate,
            alpha,
            beta,
        )
    }

    fn axpby_strided(
        &mut self,
        dst_data: &mut [T],
        src_data: &[T],
        shape: &[usize],
        dst_strides: &[isize],
        src_strides: &[isize],
        dst_offset: isize,
        src_offset: isize,
        alpha: T,
        beta: T,
    ) -> Result<(), OperationError> {
        self.axpby_strided_baked(
            dst_data,
            src_data,
            shape,
            dst_strides,
            src_strides,
            dst_offset,
            src_offset,
            alpha,
            beta,
            None,
        )
    }

    fn axpby_strided_baked(
        &mut self,
        dst_data: &mut [T],
        src_data: &[T],
        shape: &[usize],
        dst_strides: &[isize],
        src_strides: &[isize],
        dst_offset: isize,
        src_offset: isize,
        alpha: T,
        beta: T,
        baked: Option<BakedFusedLayout<'_>>,
    ) -> Result<(), OperationError> {
        // Assign-scatter (beta=0, alpha=1, positive strides) is a pure permuted
        // copy: route through the strided-perm blocked transpose when that
        // backend was selected and the layout is eligible; otherwise fall
        // through to the fused loop. Opt-in, never default — see
        // `TransposeBackend` for the #114 A/B numbers behind that. The baked
        // layout only serves the fused fallback; the strided-perm kernel builds
        // its own plan and ignores it.
        if self.transpose_backend == TransposeBackend::StridedPerm
            && beta.is_zero()
            && alpha.is_one()
            && src_strides.iter().all(|&s| s >= 0)
            && dst_strides.iter().all(|&s| s >= 0)
            && strided_perm_copy(
                dst_data,
                src_data,
                shape,
                dst_strides,
                src_strides,
                dst_offset,
                src_offset,
            )?
        {
            return Ok(());
        }
        if beta.is_zero() || beta.is_one() {
            let assign = beta.is_zero();
            // axpby_strided carries no conjugation (scatter primitive), so pass
            // source_conjugate = false onto the shared copy/axpy path.
            return copy_or_axpy(
                baked,
                dst_data,
                src_data,
                shape,
                dst_strides,
                src_strides,
                dst_offset,
                src_offset,
                assign,
                false,
                alpha,
            );
        }
        axpby_raw_strided_kernel_trusted(
            dst_data,
            src_data,
            shape,
            dst_strides,
            src_strides,
            dst_offset,
            src_offset,
            alpha,
            beta,
        )
    }

    fn copy_scale_strided(
        &mut self,
        dst_data: &mut [T],
        src_data: &[T],
        shape: &[usize],
        dst_strides: &[isize],
        src_strides: &[isize],
        dst_offset: isize,
        src_offset: isize,
        source_conjugate: bool,
        alpha: T,
    ) -> Result<(), OperationError> {
        self.copy_scale_strided_baked(
            dst_data,
            src_data,
            shape,
            dst_strides,
            src_strides,
            dst_offset,
            src_offset,
            source_conjugate,
            alpha,
            None,
        )
    }

    fn copy_scale_strided_baked(
        &mut self,
        dst_data: &mut [T],
        src_data: &[T],
        shape: &[usize],
        dst_strides: &[isize],
        src_strides: &[isize],
        dst_offset: isize,
        src_offset: isize,
        source_conjugate: bool,
        alpha: T,
        baked: Option<BakedFusedLayout<'_>>,
    ) -> Result<(), OperationError> {
        // Pack is a pure permuted copy (alpha=1, no conjugate, positive
        // strides): route it through the strided-perm blocked transpose when
        // that backend was selected and the layout is eligible; otherwise fall
        // through to the fused loop. Opt-in, never default — see
        // `TransposeBackend` for the #114 A/B numbers behind that. The baked
        // layout only serves the fused fallback.
        if self.transpose_backend == TransposeBackend::StridedPerm
            && alpha.is_one()
            && !source_conjugate
            && src_strides.iter().all(|&s| s >= 0)
            && dst_strides.iter().all(|&s| s >= 0)
            && strided_perm_copy(
                dst_data,
                src_data,
                shape,
                dst_strides,
                src_strides,
                dst_offset,
                src_offset,
            )?
        {
            return Ok(());
        }
        // Pack always assigns (dst = alpha * op(src)).
        copy_or_axpy(
            baked,
            dst_data,
            src_data,
            shape,
            dst_strides,
            src_strides,
            dst_offset,
            src_offset,
            true,
            source_conjugate,
            alpha,
        )
    }

    fn scale_strided(
        &mut self,
        dst_data: &mut [T],
        shape: &[usize],
        dst_strides: &[isize],
        dst_offset: isize,
        beta: T,
    ) -> Result<(), OperationError> {
        scale_raw_strided_kernel_trusted(dst_data, shape, dst_strides, dst_offset, beta)
    }

    fn recoupling_src_times_u_transpose<C>(
        &mut self,
        destination: &mut [T],
        source: &[T],
        recoupling_coefficients_dst_src: &[C],
        coefficient_start: usize,
        element_count: usize,
        src_count: usize,
        dst_count: usize,
    ) -> Result<(), OperationError>
    where
        C: Copy,
        T: RecouplingCoefficientAction<C>,
    {
        validate_recoupling_lens(
            destination.len(),
            source.len(),
            recoupling_coefficients_dst_src.len(),
            coefficient_start,
            element_count,
            src_count,
            dst_count,
        )?;
        for dst_index in 0..dst_count {
            let dst_column_start = dst_index * element_count;
            let coefficient_row_start = coefficient_start + dst_index * src_count;
            for element in 0..element_count {
                let mut sum = T::zero();
                for src_index in 0..src_count {
                    let coeff = recoupling_coefficients_dst_src[coefficient_row_start + src_index];
                    let src_value = source[element + src_index * element_count];
                    sum = sum + src_value.scale_by_coefficient(coeff);
                }
                destination[dst_column_start + element] = sum;
            }
        }
        Ok(())
    }
}

/// Shared dimension validation for recoupling matrix application.
///
/// All adapter implementations should validate against the same packed-column
/// layout before touching data.
pub fn validate_recoupling_lens(
    destination_len: usize,
    source_len: usize,
    coefficient_len: usize,
    coefficient_start: usize,
    element_count: usize,
    src_count: usize,
    dst_count: usize,
) -> Result<(), OperationError> {
    let expected_source_len = element_count
        .checked_mul(src_count)
        .ok_or(OperationError::ElementCountOverflow)?;
    let expected_destination_len = element_count
        .checked_mul(dst_count)
        .ok_or(OperationError::ElementCountOverflow)?;
    let coefficient_count = src_count
        .checked_mul(dst_count)
        .ok_or(OperationError::ElementCountOverflow)?;
    let coefficient_end = coefficient_start
        .checked_add(coefficient_count)
        .ok_or(OperationError::ElementCountOverflow)?;

    if source_len != expected_source_len {
        return Err(OperationError::ElementCountMismatch {
            expected: expected_source_len,
            actual: source_len,
        });
    }
    if destination_len != expected_destination_len {
        return Err(OperationError::ElementCountMismatch {
            expected: expected_destination_len,
            actual: destination_len,
        });
    }
    if coefficient_len < coefficient_end {
        return Err(OperationError::CoefficientCountMismatch {
            expected: coefficient_end,
            actual: coefficient_len,
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn strided_host_adapter_add_strided_matches_axpby_semantics() {
        let mut adapter = StridedHostKernelAdapter::default();
        let mut zero_strides = Vec::new();
        let mut dst = [10.0_f64, 20.0];
        let src = [2.0_f64, 3.0];

        adapter
            .add_strided(
                &mut zero_strides,
                &mut dst,
                &src,
                &[2],
                &[1],
                &[1],
                0,
                0,
                false,
                2.0,
                3.0,
            )
            .unwrap();

        assert_eq!(dst, [34.0, 66.0]);
    }

    #[test]
    fn strided_host_adapter_recoupling_applies_u_transpose() {
        let mut adapter = StridedHostKernelAdapter::default();
        // Two source columns of two elements, two destination columns:
        // destination[:, d] = sum_s U[d, s] * source[:, s].
        let source = [1.0_f64, 2.0, 10.0, 20.0];
        let mut destination = [0.0_f64; 4];
        let coefficients = [1.0_f64, 0.5, -1.0, 2.0];

        adapter
            .recoupling_src_times_u_transpose(&mut destination, &source, &coefficients, 0, 2, 2, 2)
            .unwrap();

        assert_eq!(destination, [6.0, 12.0, 19.0, 38.0]);
    }

    #[test]
    fn recoupling_len_validation_rejects_mismatched_columns() {
        let err = validate_recoupling_lens(4, 3, 4, 0, 2, 2, 2).unwrap_err();
        assert_eq!(
            err,
            OperationError::ElementCountMismatch {
                expected: 4,
                actual: 3,
            }
        );
    }

    // --- hybrid fused-pair dispatch tests (issue #103) ---

    fn layout(shape: &[usize], dst: &[isize], src: &[isize]) -> FusedPairLayout {
        fuse_pair_layout(shape, dst, src).expect("rank within FUSED_RANK_LIMIT")
    }

    /// Row-major strides for a shape (last axis fastest).
    fn row_major(shape: &[usize]) -> Vec<isize> {
        let mut strides = vec![1isize; shape.len()];
        for axis in (0..shape.len().saturating_sub(1)).rev() {
            strides[axis] = strides[axis + 1] * shape[axis + 1] as isize;
        }
        strides
    }

    /// Naive odometer reference for `dst[i...] = src[i...]` over strided views.
    fn reference_copy(
        dst: &mut [f64],
        src: &[f64],
        shape: &[usize],
        dst_strides: &[isize],
        src_strides: &[isize],
    ) {
        let total: usize = shape.iter().product();
        let mut index = vec![0usize; shape.len()];
        for _ in 0..total {
            let dst_pos: isize = index
                .iter()
                .zip(dst_strides)
                .map(|(&i, &s)| i as isize * s)
                .sum();
            let src_pos: isize = index
                .iter()
                .zip(src_strides)
                .map(|(&i, &s)| i as isize * s)
                .sum();
            dst[dst_pos as usize] = src[src_pos as usize];
            for axis in (0..shape.len()).rev() {
                index[axis] += 1;
                if index[axis] < shape[axis] {
                    break;
                }
                index[axis] = 0;
            }
        }
    }

    #[test]
    fn fuse_pair_layout_drops_extent_one_axes_and_fuses_contiguous_runs() {
        // Extent-1 axis dropped regardless of its (garbage) strides, then the
        // two remaining contiguous axes fuse into one 6-element run.
        let fused = layout(&[2, 1, 3], &[1, 999, 2], &[1, -7, 2]);
        assert_eq!(fused.rank, 1);
        assert_eq!(fused.dims[0], 6);
        assert_eq!(fused.dst_strides[0], 1);
        assert_eq!(fused.src_strides[0], 1);
    }

    #[test]
    fn fuse_pair_layout_orders_axes_by_destination_stride_without_fusing_mismatched_source() {
        // Axes arrive in descending destination-stride order and must be
        // reordered ascending; destination strides are contiguous (1 * 2 == 2)
        // but source strides are not (3 * 2 != 1), so the axes must NOT fuse.
        let unfused = layout(&[3, 2], &[2, 1], &[1, 3]);
        assert_eq!(unfused.rank, 2);
        assert_eq!(&unfused.dims[..2], &[2, 3]);
        assert_eq!(&unfused.dst_strides[..2], &[1, 2]);
        assert_eq!(&unfused.src_strides[..2], &[3, 1]);
    }

    #[test]
    fn fuse_pair_layout_zero_extent_collapses_to_empty_marker() {
        let empty = layout(&[2, 0, 3], &[1, 2, 4], &[1, 2, 4]);
        assert_eq!(empty.rank, 1);
        assert_eq!(empty.dims[0], 0);
    }

    #[test]
    fn fuse_pair_layout_all_extent_one_collapses_to_scalar() {
        let scalar = layout(&[1, 1], &[5, 3], &[2, 8]);
        assert_eq!(scalar.rank, 1);
        assert_eq!(scalar.dims[0], 1);
        assert_eq!(scalar.dst_strides[0], 0);
        assert_eq!(scalar.src_strides[0], 0);
    }

    #[test]
    fn fuse_pair_layout_rejects_rank_above_limit() {
        // The gate is on raw shape length, before extent-1 dropping.
        let shape = [1usize; FUSED_RANK_LIMIT + 1];
        let strides = [0isize; FUSED_RANK_LIMIT + 1];
        assert!(fuse_pair_layout(&shape, &strides, &strides).is_none());
    }

    #[test]
    fn apply_fused_pair_copies_transposed_layout_exactly() {
        // dst[j * 2 + i] = 2 * src[i * 3 + j] over a logical [2, 3] iteration:
        // exact element placement through non-fusable permuted strides.
        let src = [1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0];
        let mut dst = [0.0_f64; 6];
        let transposed = layout(&[2, 3], &[1, 2], &[3, 1]);
        apply_fused_pair(
            &mut dst,
            &src,
            &transposed,
            0,
            0,
            |dst, value| *dst = value,
            |value| 2.0 * value,
        );
        assert_eq!(dst, [2.0, 8.0, 4.0, 10.0, 6.0, 12.0]);
    }

    #[test]
    fn apply_fused_pair_accumulates_with_offsets() {
        let src = [0.0_f64, 1.0, 2.0];
        let mut dst = [10.0_f64, 20.0, 30.0];
        let contiguous = layout(&[2], &[1], &[1]);
        apply_fused_pair(
            &mut dst,
            &src,
            &contiguous,
            1,
            1,
            |dst, value| *dst = *dst + value,
            |value| 3.0 * value,
        );
        assert_eq!(dst, [10.0, 23.0, 36.0]);
    }

    #[test]
    fn apply_fused_pair_zero_extent_is_a_noop() {
        let empty = layout(&[2, 0], &[1, 2], &[1, 2]);
        let src = [1.0_f64; 4];
        let mut dst = [7.0_f64; 4];
        apply_fused_pair(
            &mut dst,
            &src,
            &empty,
            0,
            0,
            |dst, value| *dst = value,
            |value| value,
        );
        assert_eq!(dst, [7.0; 4]);
    }

    #[test]
    fn delegate_raw_copy_rank8_stack_matches_reference() {
        // #41: the non-baked assign path now runs strided-rs copy_scale_raw. At
        // rank 8 it stays on the stack fuse path; the produced values must match
        // the naive odometer reference (byte-identical delegation).
        let shape8 = [2usize; 8];
        let src_strides8 = row_major(&shape8);
        let dst_strides8: Vec<isize> = src_strides8.iter().rev().copied().collect();
        let src: Vec<f64> = (0..256).map(|value| value as f64 * 0.5 + 1.0).collect();

        let mut dst = vec![0.0_f64; 256];
        delegate_raw_copy(
            &mut dst,
            &src,
            &shape8,
            &dst_strides8,
            &src_strides8,
            0,
            0,
            true,
            false,
            1.0,
        )
        .unwrap();

        let mut dst_reference = vec![0.0_f64; 256];
        reference_copy(
            &mut dst_reference,
            &src,
            &shape8,
            &dst_strides8,
            &src_strides8,
        );
        assert_eq!(dst, dst_reference);
    }

    #[test]
    fn rank9_routes_to_scratch_path_and_matches_reference() {
        // Rank 9 is above FUSED_RANK_LIMIT, so copy_or_axpy routes to the kept
        // hand-rolled fused_pair_high_rank (thread_local scratch, alloc-free
        // warm) instead of delegating — the #140 raw kernels would allocate
        // through the view fallback here. Values must match the naive reference.
        let shape = [2usize; 9];
        let src_strides = row_major(&shape);
        let dst_strides: Vec<isize> = src_strides.iter().rev().copied().collect();
        let src: Vec<f64> = (0..512).map(|value| value as f64 - 100.0).collect();

        let mut dst = vec![0.0_f64; 512];
        copy_or_axpy(
            None,
            &mut dst,
            &src,
            &shape,
            &dst_strides,
            &src_strides,
            0,
            0,
            true,
            false,
            1.0,
        )
        .unwrap();

        let mut dst_reference = vec![0.0_f64; 512];
        reference_copy(&mut dst_reference, &src, &shape, &dst_strides, &src_strides);
        assert_eq!(dst, dst_reference);
    }

    #[test]
    fn delegate_raw_copy_preserves_enumerated_contracts() {
        // #41 contract pin: the delegated raw path reproduces the old fused_pair
        // behavior for every enumerated edge case — zero extent (no-op),
        // negative source stride, broadcast (stride-0) source, and a
        // non-injective destination (last write wins, per-call raw kernels do
        // NOT reject it — only CopyPlan.compile would).

        // Zero extent: no-op, no bounds error.
        let src = [1.0_f64; 4];
        let mut dst = [9.0_f64; 4];
        delegate_raw_copy(
            &mut dst,
            &src,
            &[2, 0],
            &[1, 2],
            &[1, 2],
            0,
            0,
            true,
            false,
            1.0,
        )
        .unwrap();
        assert_eq!(dst, [9.0; 4]);

        // Negative source stride: reversed read.
        let src = [1.0_f64, 2.0, 3.0, 4.0];
        let mut dst = [0.0_f64; 4];
        delegate_raw_copy(&mut dst, &src, &[4], &[1], &[-1], 0, 3, true, false, 1.0).unwrap();
        assert_eq!(dst, [4.0, 3.0, 2.0, 1.0]);

        // Broadcast (stride-0) source: every logical index reads src[0].
        let src = [7.0_f64];
        let mut dst = [0.0_f64; 3];
        delegate_raw_copy(&mut dst, &src, &[3], &[1], &[0], 0, 0, true, false, 2.0).unwrap();
        assert_eq!(dst, [14.0, 14.0, 14.0]);

        // Non-injective destination (dst stride 0): allowed, last write wins.
        let src = [1.0_f64, 2.0, 3.0];
        let mut dst = [0.0_f64];
        delegate_raw_copy(&mut dst, &src, &[3], &[0], &[1], 0, 0, true, false, 1.0).unwrap();
        assert_eq!(dst, [3.0]);
    }

    #[test]
    fn delegate_raw_copy_accumulate_and_conjugate_match_reference() {
        use num_complex::Complex64;
        // Accumulate (axpy_raw): dst += alpha * src.
        let src = [1.0_f64, 2.0];
        let mut dst = [10.0_f64, 20.0];
        delegate_raw_copy(&mut dst, &src, &[2], &[1], &[1], 0, 0, false, false, 3.0).unwrap();
        assert_eq!(dst, [13.0, 26.0]);

        // Conjugating assign (copy_scale_conj_raw): dst = alpha * conj(src).
        let src = [Complex64::new(1.0, 2.0), Complex64::new(-3.0, 4.0)];
        let mut dst = [Complex64::default(); 2];
        delegate_raw_copy(
            &mut dst,
            &src,
            &[2],
            &[1],
            &[1],
            0,
            0,
            true,
            true,
            Complex64::new(2.0, 0.0),
        )
        .unwrap();
        assert_eq!(dst[0], Complex64::new(2.0, -4.0));
        assert_eq!(dst[1], Complex64::new(-6.0, -8.0));

        // Conjugating accumulate (axpy_conj_raw): dst += alpha * conj(src).
        let src = [Complex64::new(1.0, 1.0)];
        let mut dst = [Complex64::new(5.0, 5.0)];
        delegate_raw_copy(
            &mut dst,
            &src,
            &[1],
            &[1],
            &[1],
            0,
            0,
            false,
            true,
            Complex64::new(1.0, 0.0),
        )
        .unwrap();
        assert_eq!(dst[0], Complex64::new(6.0, 4.0));
    }

    #[test]
    fn delegate_raw_copy_rejects_out_of_bounds_as_clean_error() {
        // #41 observable-behavior note: the removed hand-rolled loop panicked on
        // an out-of-bounds layout (safe indexing); the checked RawStrided::new
        // now returns a clean Err instead. Pinned so the panic->Err change is
        // deliberate, not accidental.
        let src = [1.0_f64; 4];
        let mut dst = [0.0_f64; 3]; // one element too short for shape [4]
        let err = delegate_raw_copy(&mut dst, &src, &[4], &[1], &[1], 0, 0, true, false, 1.0);
        assert!(err.is_err());
    }

    #[test]
    fn strided_host_adapter_fused_beta_branches_match_axpby_semantics() {
        let mut adapter = StridedHostKernelAdapter::default();
        let mut zero_strides = Vec::new();
        let src = [2.0_f64, 3.0];

        // add_strided beta = 1: accumulate through the fused path.
        let mut dst = [10.0_f64, 20.0];
        adapter
            .add_strided(
                &mut zero_strides,
                &mut dst,
                &src,
                &[2],
                &[1],
                &[1],
                0,
                0,
                false,
                2.0,
                1.0,
            )
            .unwrap();
        assert_eq!(dst, [14.0, 26.0]);

        // add_strided beta = 0: assign through the fused path.
        adapter
            .add_strided(
                &mut zero_strides,
                &mut dst,
                &src,
                &[2],
                &[1],
                &[1],
                0,
                0,
                false,
                2.0,
                0.0,
            )
            .unwrap();
        assert_eq!(dst, [4.0, 6.0]);

        // axpby_strided beta = 1 then beta = 0.
        let mut dst = [1.0_f64, 2.0];
        adapter
            .axpby_strided(&mut dst, &src, &[2], &[1], &[1], 0, 0, 3.0, 1.0)
            .unwrap();
        assert_eq!(dst, [7.0, 11.0]);
        adapter
            .axpby_strided(&mut dst, &src, &[2], &[1], &[1], 0, 0, 3.0, 0.0)
            .unwrap();
        assert_eq!(dst, [6.0, 9.0]);

        // copy_scale_strided always assigns.
        let mut dst = [99.0_f64, 99.0];
        adapter
            .copy_scale_strided(&mut dst, &src, &[2], &[1], &[1], 0, 0, false, -1.0)
            .unwrap();
        assert_eq!(dst, [-2.0, -3.0]);

        // scale_strided scales in place.
        adapter.scale_strided(&mut dst, &[2], &[1], 0, 2.0).unwrap();
        assert_eq!(dst, [-4.0, -6.0]);
    }

    // #41 perf probe (ignored; run explicitly):
    //   cargo test --release -p tenet-operations -- --ignored --nocapture bench_41
    // Small-block repeated copy at the SU(2) replay regime (21 rank-4 d=4
    // transposed blocks): the pre-#41 hand-rolled rank<=8 path (fuse_pair_layout
    // + apply_fused_pair_slices, NO bounds check) vs #140 `copy_scale_raw`
    // (per-call fuse + 2x validate_bounds) vs #142 `CopyPlan` (compile once,
    // execute many; still validates bounds per execute). Documents that the
    // delegation trades ~30% on this NON-baked path for the per-call
    // RawStrided::new bounds validation — the warm hot replay path is unaffected
    // because it takes the #232 baked route (apply_fused_pair_slices, unchanged).
    // Reaching baseline would need a validation-free prepared execute in
    // strided-rs; new_unchecked is blocked by this crate's #![deny(unsafe_code)].
    #[test]
    #[ignore]
    fn bench_41_fused_vs_raw_vs_plan() {
        use std::time::Instant;
        use strided_kernel::{copy_scale_raw, CopyPlan, RawStridedMut, RawStridedRef};

        const BLOCKS: usize = 21;
        let dims = [4usize, 4, 4, 4];
        let src_strides = [1isize, 4, 16, 64]; // column-major src
        let dst_strides = [64isize, 16, 4, 1]; // transposed dst (non-contiguous)
        let elems = 256usize;
        let src: Vec<f64> = (0..elems).map(|i| i as f64 * 0.5 - 3.0).collect();
        let mut dst = vec![0.0f64; elems];
        let iters = 20_000usize;

        let median = |mut v: Vec<f64>| {
            v.sort_by(|a, b| a.partial_cmp(b).unwrap());
            v[v.len() / 2]
        };

        // (a) hand-rolled baseline: the exact pre-#41 rank<=8 path
        // (fuse_pair_layout + apply_fused_pair_slices), with NO bounds check —
        // this is what `fused_pair` ran for rank <= FUSED_RANK_LIMIT.
        let mut fused_ns = Vec::new();
        for _ in 0..5 {
            let t = Instant::now();
            for _ in 0..iters {
                for _ in 0..BLOCKS {
                    let layout = fuse_pair_layout(&dims, &dst_strides, &src_strides).unwrap();
                    apply_fused_pair_slices(
                        &mut dst,
                        &src,
                        &layout.dims[..layout.rank],
                        &layout.dst_strides[..layout.rank],
                        &layout.src_strides[..layout.rank],
                        0,
                        0,
                        |d: &mut f64, v: f64| *d = v,
                        |v: f64| v,
                    );
                }
            }
            fused_ns.push(t.elapsed().as_nanos() as f64 / (iters * BLOCKS) as f64);
        }

        // (b) #140 copy_scale_raw, per-call (RawStrided::new + fuse each block)
        let mut raw_ns = Vec::new();
        for _ in 0..5 {
            let t = Instant::now();
            for _ in 0..iters {
                for _ in 0..BLOCKS {
                    let s = RawStridedRef::new(&src, &dims, &src_strides, 0).unwrap();
                    let mut d = RawStridedMut::new(&mut dst, &dims, &dst_strides, 0).unwrap();
                    copy_scale_raw(&mut d, &s, 1.0).unwrap();
                }
            }
            raw_ns.push(t.elapsed().as_nanos() as f64 / (iters * BLOCKS) as f64);
        }

        // (c) #142 CopyPlan, compile once then execute_scale per block
        let plan = CopyPlan::compile(&dims, &dst_strides, &src_strides).unwrap();
        let compile_t = Instant::now();
        for _ in 0..iters {
            let _ = CopyPlan::compile(&dims, &dst_strides, &src_strides).unwrap();
        }
        let compile_ns = compile_t.elapsed().as_nanos() as f64 / iters as f64;
        let mut plan_ns = Vec::new();
        for _ in 0..5 {
            let t = Instant::now();
            for _ in 0..iters {
                for _ in 0..BLOCKS {
                    let s = RawStridedRef::new(&src, &dims, &src_strides, 0).unwrap();
                    let mut d = RawStridedMut::new(&mut dst, &dims, &dst_strides, 0).unwrap();
                    plan.execute_scale(&mut d, &s, 1.0).unwrap();
                }
            }
            plan_ns.push(t.elapsed().as_nanos() as f64 / (iters * BLOCKS) as f64);
        }

        println!("\n#41 small-block (21x rank4 d4 transposed) per-block ns, median-of-5:");
        println!("  fused_pair (hand-rolled): {:.1}", median(fused_ns));
        println!("  copy_scale_raw (#140):    {:.1}", median(raw_ns));
        println!(
            "  CopyPlan.execute (#142):  {:.1}  (compile once: {:.1} ns)",
            median(plan_ns),
            compile_ns
        );
    }
}

/// Parity tests for the strided-perm transpose route (ported from prototype
/// commit b3ca6e5). These call `strided_perm_copy` and `fused_pair` directly
/// and assert byte-equality, independent of any runtime backend selection —
/// the routing contract (routing never changes results) holds for every
/// [`TransposeBackend`] value.
#[cfg(test)]
mod strided_perm_probe {
    use super::{
        apply_fused_pair, fuse_pair_layout, strided_perm_copy, StridedHostKernelAdapter,
        TransposeBackend,
    };
    use crate::kernel_adapter::HostKernelAdapter;

    /// Byte-identical reference for the fused assign copy `dst = src`, built from
    /// the kept `fuse_pair_layout` + `apply_fused_pair` (the same normalization
    /// the production baked replay and the strided-rs raw kernels both run). Used
    /// in place of the removed hand-rolled `fused_pair` (#41). Only covers the
    /// rank <= FUSED_RANK_LIMIT layouts these probes exercise.
    fn reference_fused(
        dst: &mut [f64],
        src: &[f64],
        shape: &[usize],
        dst_strides: &[isize],
        src_strides: &[isize],
        dst_off: isize,
        src_off: isize,
    ) {
        let layout = fuse_pair_layout(shape, dst_strides, src_strides)
            .expect("probe layouts stay within FUSED_RANK_LIMIT");
        apply_fused_pair(
            dst,
            src,
            &layout,
            dst_off,
            src_off,
            |d, v| *d = v,
            |v: f64| v,
        );
    }

    /// Inclusive max linear index a positive-stride layout reaches.
    fn span(shape: &[usize], strides: &[isize], offset: isize) -> usize {
        let mut hi = offset;
        for (&d, &s) in shape.iter().zip(strides) {
            hi += (d as isize - 1).max(0) * s;
        }
        (hi + 1) as usize
    }

    /// strided-perm-path output must equal the fused-loop output for every
    /// layout it accepts (the routing contract: routing never changes results).
    fn assert_strided_perm_matches_fused(
        shape: &[usize],
        dst_strides: &[isize],
        src_strides: &[isize],
        dst_off: isize,
        src_off: isize,
    ) {
        let src: Vec<f64> = (0..span(shape, src_strides, src_off))
            .map(|i| (i as f64) * 0.5 - 3.0)
            .collect();
        let dlen = span(shape, dst_strides, dst_off);
        let mut route = vec![0.0f64; dlen];
        let mut fused = vec![0.0f64; dlen];
        let handled = strided_perm_copy(
            &mut route,
            &src,
            shape,
            dst_strides,
            src_strides,
            dst_off,
            src_off,
        )
        .unwrap();
        assert!(
            handled,
            "expected strided-perm to handle shape={shape:?} ds={dst_strides:?} ss={src_strides:?}"
        );
        reference_fused(
            &mut fused,
            &src,
            shape,
            dst_strides,
            src_strides,
            dst_off,
            src_off,
        );
        assert_eq!(
            route, fused,
            "strided-perm != fused for shape={shape:?} ds={dst_strides:?} ss={src_strides:?} \
             doff={dst_off} soff={src_off}"
        );
    }

    #[test]
    #[allow(clippy::type_complexity)] // fixed-shape test-case table, not a public type
    fn strided_perm_matches_fused_across_layouts() {
        // (shape, dst_strides, src_strides, dst_off, src_off) — every case has a
        // genuine stride-1 axis on each side and positive strides.
        let cases: &[(&[usize], &[isize], &[isize], isize, isize)] = &[
            (&[4, 4], &[1, 4], &[4, 1], 0, 0), // 2D square transpose
            (&[3, 5], &[1, 3], &[5, 1], 0, 0), // 2D rectangular transpose
            (&[3, 3], &[1, 3], &[3, 1], 5, 7), // 2D with offsets
            (&[3, 4, 2], &[1, 3, 12], &[1, 10, 40], 0, 0), // 3D gapped sub-block, stride-1 axis0
            (&[3, 4, 2], &[1, 3, 12], &[4, 1, 16], 0, 0), // 3D, src stride-1 on axis1
            (&[2, 3, 2, 2], &[1, 2, 6, 12], &[24, 8, 1, 4], 0, 0), // 4D permuted
            (&[1, 4, 1, 4], &[1, 1, 4, 4], &[9, 1, 9, 8], 0, 0), // extent-1 mixed, reduces to [4,4]
            (&[6, 6], &[1, 6], &[6, 1], 0, 0), // larger transpose (hits blocking)
        ];
        for &(shape, ds, ss, doff, soff) in cases {
            assert_strided_perm_matches_fused(shape, ds, ss, doff, soff);
        }
    }

    #[test]
    fn declines_strided_slice_without_stride1_axis() {
        // Strided slice of a larger buffer: reduced strides [168,42] have no
        // stride-1 axis (its contiguous axis was a singleton). The kernel cannot
        // do this correctly, so the wrapper must decline (return false) and leave
        // dst untouched. This is the exact pattern that corrupted the energy
        // (-1.803 vs -1.772) before the guard existed.
        let src = vec![0.0f64; 4096];
        let shape = [4usize, 4];
        let mut dst = [7.0f64; 16];
        let handled = strided_perm_copy(&mut dst, &src, &shape, &[1, 4], &[168, 42], 0, 0).unwrap();
        assert!(!handled, "must decline the no-stride-1 strided slice");
        assert!(
            dst.iter().all(|&x| x == 7.0),
            "dst must be untouched when declined"
        );
    }

    #[test]
    fn declines_when_only_dst_has_stride1() {
        // dst has a stride-1 axis but src does not (both sides are required).
        let src = vec![1.0f64; 4096];
        let handled =
            strided_perm_copy(&mut [0.0f64; 16], &src, &[4, 4], &[1, 4], &[10, 40], 0, 0).unwrap();
        assert!(!handled, "must decline when src lacks a stride-1 axis");
    }

    #[test]
    fn all_extent1_copies_single_element() {
        let src = [42.0f64, 99.0];
        let mut dst = [0.0f64, 0.0];
        let handled = strided_perm_copy(&mut dst, &src, &[1, 1], &[1, 1], &[1, 1], 1, 1).unwrap();
        assert!(handled);
        assert_eq!(
            dst[1], src[1],
            "single-element copy must move src[off] to dst[off]"
        );
    }

    #[test]
    fn transposes_genuine_case_correctly() {
        // Genuine transpose against a hand-computed reference (not just fused):
        // src row-major [3,1], dst col-major [1,2]. dst[i+2j] == src[3i+j].
        let src = [1.0f64, 2.0, 3.0, 4.0, 5.0, 6.0];
        let mut dst = [0.0f64; 6];
        let handled = strided_perm_copy(&mut dst, &src, &[2, 3], &[1, 2], &[3, 1], 0, 0).unwrap();
        assert!(handled, "genuine transpose must be handled by strided-perm");
        let mut expect = [0.0f64; 6];
        for i in 0..2 {
            for j in 0..3 {
                expect[i + 2 * j] = src[3 * i + j];
            }
        }
        assert_eq!(
            dst, expect,
            "strided-perm transpose disagrees with manual copy"
        );
    }

    /// #104-style differential test through the PUBLIC adapter: the default
    /// (`FusedLoops`) adapter and a `StridedPerm` adapter must byte-match on an
    /// eligible transpose layout, for both `copy_scale_strided` (pack) and
    /// `axpby_strided` (assign-scatter) — pins that selecting the backend
    /// changes nothing but which kernel runs.
    #[test]
    fn adapter_backends_byte_match_on_eligible_transpose() {
        let shape = [6usize, 5];
        let ss = [5isize, 1]; // src row-major
        let ds = [1isize, 6]; // dst col-major => transpose, both sides stride-1
        let src: Vec<f64> = (0..30).map(|i| i as f64 * 0.25 - 2.0).collect();

        // Reference: the raw route (also asserts the layout is eligible).
        let mut route = vec![0.0f64; 30];
        assert!(
            strided_perm_copy(&mut route, &src, &shape, &ds, &ss, 0, 0).unwrap(),
            "layout must be route-eligible"
        );

        let mut fused_adapter = StridedHostKernelAdapter::default();
        let mut perm_adapter =
            StridedHostKernelAdapter::with_transpose_backend(TransposeBackend::StridedPerm);
        for adapter in [&mut fused_adapter, &mut perm_adapter] {
            let backend = adapter.transpose_backend;
            let mut pack = vec![0.0f64; 30];
            adapter
                .copy_scale_strided(&mut pack, &src, &shape, &ds, &ss, 0, 0, false, 1.0)
                .unwrap();
            assert_eq!(pack, route, "copy_scale_strided mismatch for {backend:?}");

            let mut scatter = vec![0.0f64; 30];
            adapter
                .axpby_strided(&mut scatter, &src, &shape, &ds, &ss, 0, 0, 1.0, 0.0)
                .unwrap();
            assert_eq!(scatter, route, "axpby_strided mismatch for {backend:?}");
        }
    }

    /// #41 observable-behavior pin. Before delegation the default (FusedLoops)
    /// route *panicked* on an out-of-bounds destination (safe indexing) while the
    /// StridedPerm route returned a clean `Err` (StridedView validation). Now the
    /// FusedLoops route delegates to strided-rs `copy_scale_raw`, whose checked
    /// `RawStridedMut::new` also rejects OOB with a clean `Err` — so both backends
    /// now reject the same out-of-bounds layout uniformly, with no panic. This
    /// pins that the panic->Err change is deliberate and that neither route
    /// aborts the process on a malformed layout.
    #[test]
    fn both_backends_reject_out_of_bounds_without_panic() {
        let shape = [4usize, 4];
        let ss = [4isize, 1];
        let ds = [1isize, 4]; // eligible transpose layout
        let src: Vec<f64> = (0..16).map(|i| i as f64).collect();

        for backend in [TransposeBackend::FusedLoops, TransposeBackend::StridedPerm] {
            let mut adapter = StridedHostKernelAdapter::with_transpose_backend(backend);
            let mut short_dst = vec![0.0f64; 15]; // one element too short for [4,4]
            assert!(
                adapter
                    .copy_scale_strided(&mut short_dst, &src, &shape, &ds, &ss, 0, 0, false, 1.0)
                    .is_err(),
                "{backend:?} route must reject the out-of-bounds layout as a clean Err"
            );
        }
    }

    // Crossover micro-bench (ignored; run explicitly):
    //   cargo test --release -p tenet-operations -- --ignored --nocapture strided_perm_crossover
    // Prints fused vs strided-perm per-call ns over square transposes of growing
    // N so the block size where the blocked micro-kernel overtakes the fused loop
    // (amortizing its per-call plan build) is visible.
    #[test]
    #[ignore]
    fn strided_perm_crossover_bench() {
        use std::time::Instant;
        println!("\n  N     fused(ns)   strided-perm(ns)   speedup");
        for &n in &[4usize, 8, 16, 32, 64, 128, 256, 512] {
            let src: Vec<f64> = (0..n * n).map(|i| i as f64).collect();
            let shape = [n, n];
            let ss = [n as isize, 1]; // src row-major (stride-1 axis1)
            let ds = [1, n as isize]; // dst col-major (stride-1 axis0) => transpose
            let mut dst = vec![0.0f64; n * n];
            let iters = (1usize << 24 >> (2 * (n as f64).log2() as usize)).max(50);
            // warm caches
            reference_fused(&mut dst, &src, &shape, &ds, &ss, 0, 0);
            let _ = strided_perm_copy(&mut dst, &src, &shape, &ds, &ss, 0, 0);
            let t = Instant::now();
            for _ in 0..iters {
                reference_fused(&mut dst, &src, &shape, &ds, &ss, 0, 0);
            }
            let fused_ns = t.elapsed().as_nanos() as f64 / iters as f64;
            let t = Instant::now();
            for _ in 0..iters {
                let _ = strided_perm_copy(&mut dst, &src, &shape, &ds, &ss, 0, 0);
            }
            let route_ns = t.elapsed().as_nanos() as f64 / iters as f64;
            println!(
                "{n:4}  {fused_ns:11.1}  {route_ns:15.1}   {:.2}x {}",
                fused_ns / route_ns,
                if route_ns < fused_ns {
                    "<- route wins"
                } else {
                    ""
                }
            );
        }
    }
}
