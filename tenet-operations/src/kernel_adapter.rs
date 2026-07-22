use core::ops::{Add, Mul};

use num_traits::{One, Zero};

use crate::{
    axpby_raw_strided_kernel_trusted, scale_raw_strided_kernel_trusted,
    tensoradd_raw_strided_kernel_trusted, ConjugateValue, OperationError,
    RecouplingCoefficientAction,
};

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

#[derive(Debug, Default)]
pub(crate) struct FusedLayoutScratch {
    dims: Vec<usize>,
    dst_strides: Vec<isize>,
    src_strides: Vec<isize>,
}

impl FusedLayoutScratch {
    pub(crate) fn dims(&self) -> &[usize] {
        &self.dims
    }

    pub(crate) fn dst_strides(&self) -> &[isize] {
        &self.dst_strides
    }

    pub(crate) fn src_strides(&self) -> &[isize] {
        &self.src_strides
    }
}

#[derive(Debug, Default)]
struct StridedKernelScratch {
    layout: FusedLayoutScratch,
    index: Vec<usize>,
}

/// Borrowed view of a prebaked fused loop layout (issue #232).
///
/// Holds the exact `(dims, dst_strides, src_strides)` that [`normalize_fused_layout`]
/// would return for one (block, role) stride pair, computed once at compile
/// time in the immutable `TreeTransformLayoutTable` and reused across every
/// replay call instead of recomputed. The slices live in that table's arena;
/// `apply_fused_pair_slices` consumes them directly. dtype-independent — one
/// baked layout serves f64 and c64 alike (the normalization never inspects
/// values).
///
/// Safe downstream adapters retain direct read access to the normalized slices:
///
/// ```
/// use tenet_operations::BakedFusedLayout;
///
/// fn inspect(layout: BakedFusedLayout<'_>) {
///     let _ = (layout.dims, layout.dst_strides, layout.src_strides);
/// }
/// ```
///
/// Construction remains sealed:
///
/// ```compile_fail
/// use tenet_operations::BakedFusedLayout;
///
/// let _ = BakedFusedLayout {
///     dims: &[2],
///     dst_strides: &[1],
///     src_strides: &[1],
/// };
/// ```
#[derive(Clone, Copy, Debug)]
pub struct BakedFusedLayout<'a> {
    pub dims: &'a [usize],
    pub dst_strides: &'a [isize],
    pub src_strides: &'a [isize],
    // Why not expose the seal: readable fields preserve custom-adapter
    // compatibility, while construction must remain tied to normalized data.
    _sealed: (),
}

impl<'a> BakedFusedLayout<'a> {
    pub(crate) fn try_from_normalized_slices(
        dims: &'a [usize],
        dst_strides: &'a [isize],
        src_strides: &'a [isize],
    ) -> Result<Self, OperationError> {
        if dims.is_empty() {
            return Err(OperationError::RankMismatch {
                expected: 1,
                actual: 0,
            });
        }
        if dims.len() != dst_strides.len() {
            return Err(OperationError::RankMismatch {
                expected: dims.len(),
                actual: dst_strides.len(),
            });
        }
        if dims.len() != src_strides.len() {
            return Err(OperationError::RankMismatch {
                expected: dims.len(),
                actual: src_strides.len(),
            });
        }
        if dims == [0] && dst_strides == [0] && src_strides == [0] {
            return Ok(Self {
                dims,
                dst_strides,
                src_strides,
                _sealed: (),
            });
        }
        if dims == [1] && dst_strides == [0] && src_strides == [0] {
            return Ok(Self {
                dims,
                dst_strides,
                src_strides,
                _sealed: (),
            });
        }
        if dims.iter().any(|&dim| dim <= 1) {
            return Err(OperationError::InvalidArgument {
                message: "baked fused layout is not normalized",
            });
        }
        dims.iter().try_fold(1usize, |product, &dim| {
            product
                .checked_mul(dim)
                .ok_or(OperationError::ElementCountOverflow)
        })?;
        Ok(Self {
            dims,
            dst_strides,
            src_strides,
            _sealed: (),
        })
    }

    #[inline]
    pub fn dims(&self) -> &'a [usize] {
        self.dims
    }

    #[inline]
    pub fn dst_strides(&self) -> &'a [isize] {
        self.dst_strides
    }

    #[inline]
    pub fn src_strides(&self) -> &'a [isize] {
        self.src_strides
    }
}

#[inline]
fn validate_strided_ranks(
    shape: &[usize],
    dst_strides: &[isize],
    src_strides: &[isize],
) -> Result<(), OperationError> {
    if shape.len() != dst_strides.len() {
        return Err(OperationError::RankMismatch {
            expected: shape.len(),
            actual: dst_strides.len(),
        });
    }
    if shape.len() != src_strides.len() {
        return Err(OperationError::RankMismatch {
            expected: shape.len(),
            actual: src_strides.len(),
        });
    }
    Ok(())
}

pub(crate) fn normalize_fused_layout(
    shape: &[usize],
    dst_strides: &[isize],
    src_strides: &[isize],
    scratch: &mut FusedLayoutScratch,
) -> Result<(), OperationError> {
    validate_strided_ranks(shape, dst_strides, src_strides)?;
    scratch.dims.clear();
    scratch.dst_strides.clear();
    scratch.src_strides.clear();

    if shape.contains(&0) {
        scratch.dims.push(0);
        scratch.dst_strides.push(0);
        scratch.src_strides.push(0);
        return Ok(());
    }
    shape.iter().try_fold(1usize, |product, &dim| {
        product
            .checked_mul(dim)
            .ok_or(OperationError::ElementCountOverflow)
    })?;

    for axis in 0..shape.len() {
        if shape[axis] == 1 {
            continue;
        }
        let mut position = scratch.dims.len();
        while position > 0 && scratch.dst_strides[position - 1] > dst_strides[axis] {
            position -= 1;
        }
        scratch.dims.insert(position, shape[axis]);
        scratch.dst_strides.insert(position, dst_strides[axis]);
        scratch.src_strides.insert(position, src_strides[axis]);
    }
    if scratch.dims.is_empty() {
        scratch.dims.push(1);
        scratch.dst_strides.push(0);
        scratch.src_strides.push(0);
    }
    let mut fused = 0usize;
    for axis in 1..scratch.dims.len() {
        let extent = scratch.dims[fused] as isize;
        if scratch.dst_strides[fused] * extent == scratch.dst_strides[axis]
            && scratch.src_strides[fused] * extent == scratch.src_strides[axis]
        {
            scratch.dims[fused] = scratch.dims[fused]
                .checked_mul(scratch.dims[axis])
                .ok_or(OperationError::ElementCountOverflow)?;
        } else {
            fused += 1;
            scratch.dims[fused] = scratch.dims[axis];
            scratch.dst_strides[fused] = scratch.dst_strides[axis];
            scratch.src_strides[fused] = scratch.src_strides[axis];
        }
    }
    let rank = fused + 1;
    scratch.dims.truncate(rank);
    scratch.dst_strides.truncate(rank);
    scratch.src_strides.truncate(rank);
    Ok(())
}

pub(crate) fn for_each_fused_span<F>(
    dims: &[usize],
    dst_strides: &[isize],
    src_strides: &[isize],
    dst_offset: isize,
    src_offset: isize,
    index: &mut [usize],
    mut visit: F,
) where
    F: FnMut(isize, isize, usize, isize, isize),
{
    let rank = dims.len();
    debug_assert_eq!(rank, dst_strides.len());
    debug_assert_eq!(rank, src_strides.len());
    if rank == 0 || dims.contains(&0) {
        return;
    }
    debug_assert_eq!(rank, index.len());
    let inner_len = dims[0];
    let inner_dst = dst_strides[0];
    let inner_src = src_strides[0];
    index.fill(0);
    let mut dst_base = dst_offset;
    let mut src_base = src_offset;
    loop {
        visit(dst_base, src_base, inner_len, inner_dst, inner_src);
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

#[allow(clippy::too_many_arguments)]
fn fused_pair<T, Apply, ElementOp>(
    scratch: &mut StridedKernelScratch,
    dst_data: &mut [T],
    src_data: &[T],
    shape: &[usize],
    dst_strides: &[isize],
    src_strides: &[isize],
    dst_offset: isize,
    src_offset: isize,
    apply: Apply,
    op: ElementOp,
) -> Result<(), OperationError>
where
    T: Copy,
    Apply: Fn(&mut T, T),
    ElementOp: Fn(T) -> T,
{
    let StridedKernelScratch { layout, index } = scratch;
    normalize_fused_layout(shape, dst_strides, src_strides, layout)?;
    index.resize(layout.dims.len(), 0);
    apply_fused_pair_slices(
        dst_data,
        src_data,
        &layout.dims,
        &layout.dst_strides,
        &layout.src_strides,
        dst_offset,
        src_offset,
        index.as_mut_slice(),
        apply,
        op,
    );
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn apply_fused_pair_slices<T, Apply, ElementOp>(
    dst_data: &mut [T],
    src_data: &[T],
    dims: &[usize],
    dst_strides: &[isize],
    src_strides: &[isize],
    dst_offset: isize,
    src_offset: isize,
    index: &mut [usize],
    apply: Apply,
    op: ElementOp,
) where
    T: Copy,
    Apply: Fn(&mut T, T),
    ElementOp: Fn(T) -> T,
{
    for_each_fused_span(
        dims,
        dst_strides,
        src_strides,
        dst_offset,
        src_offset,
        index,
        |dst_base, src_base, inner_len, inner_dst, inner_src| {
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
        },
    );
}

#[allow(clippy::too_many_arguments)]
fn fused_pair_baked<T, Apply, ElementOp>(
    scratch: &mut StridedKernelScratch,
    baked: Option<BakedFusedLayout<'_>>,
    dst_data: &mut [T],
    src_data: &[T],
    shape: &[usize],
    dst_strides: &[isize],
    src_strides: &[isize],
    dst_offset: isize,
    src_offset: isize,
    apply: Apply,
    op: ElementOp,
) -> Result<(), OperationError>
where
    T: Copy,
    Apply: Fn(&mut T, T),
    ElementOp: Fn(T) -> T,
{
    match baked {
        Some(baked) => {
            scratch.index.resize(baked.dims().len(), 0);
            apply_fused_pair_slices(
                dst_data,
                src_data,
                baked.dims(),
                baked.dst_strides(),
                baked.src_strides(),
                dst_offset,
                src_offset,
                scratch.index.as_mut_slice(),
                apply,
                op,
            );
            Ok(())
        }
        None => fused_pair(
            scratch,
            dst_data,
            src_data,
            shape,
            dst_strides,
            src_strides,
            dst_offset,
            src_offset,
            apply,
            op,
        ),
    }
}

#[allow(clippy::too_many_arguments)]
fn fused_pair_baked_with_index<T, Apply, ElementOp>(
    layout: &mut FusedLayoutScratch,
    baked: Option<BakedFusedLayout<'_>>,
    index: &mut [usize],
    dst_data: &mut [T],
    src_data: &[T],
    shape: &[usize],
    dst_strides: &[isize],
    src_strides: &[isize],
    dst_offset: isize,
    src_offset: isize,
    apply: Apply,
    op: ElementOp,
) -> Result<(), OperationError>
where
    T: Copy,
    Apply: Fn(&mut T, T),
    ElementOp: Fn(T) -> T,
{
    match baked {
        Some(baked) => {
            let rank = baked.dims().len();
            let Some(index) = index.get_mut(..rank) else {
                return Err(OperationError::InvalidArgument {
                    message: "fused traversal scratch is shorter than the normalized rank",
                });
            };
            apply_fused_pair_slices(
                dst_data,
                src_data,
                baked.dims(),
                baked.dst_strides(),
                baked.src_strides(),
                dst_offset,
                src_offset,
                index,
                apply,
                op,
            );
        }
        None => {
            normalize_fused_layout(shape, dst_strides, src_strides, layout)?;
            let rank = layout.dims.len();
            let Some(index) = index.get_mut(..rank) else {
                return Err(OperationError::InvalidArgument {
                    message: "fused traversal scratch is shorter than the normalized rank",
                });
            };
            apply_fused_pair_slices(
                dst_data,
                src_data,
                &layout.dims,
                &layout.dst_strides,
                &layout.src_strides,
                dst_offset,
                src_offset,
                index,
                apply,
                op,
            );
        }
    }
    Ok(())
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
    /// host adapter overrides it to skip layout normalization on the `beta ∈ {0,1}`
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

    /// [`add_strided_baked`](Self::add_strided_baked) with caller-owned
    /// traversal scratch. Custom adapters keep their existing behavior through
    /// this additive default; compiled host replay uses the override to retain
    /// runtime-rank state in its execution workspace.
    #[allow(clippy::too_many_arguments)]
    fn add_strided_baked_with_index(
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
        index: &mut [usize],
    ) -> Result<(), OperationError> {
        let _ = index;
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
            baked,
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

    /// [`axpby_strided_baked`](Self::axpby_strided_baked) with caller-owned
    /// traversal scratch. See
    /// [`add_strided_baked_with_index`](Self::add_strided_baked_with_index).
    #[allow(clippy::too_many_arguments)]
    fn axpby_strided_baked_with_index(
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
        index: &mut [usize],
    ) -> Result<(), OperationError> {
        let _ = index;
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
            baked,
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

    /// [`copy_scale_strided_baked`](Self::copy_scale_strided_baked) with
    /// caller-owned traversal scratch. See
    /// [`add_strided_baked_with_index`](Self::add_strided_baked_with_index).
    #[allow(clippy::too_many_arguments)]
    fn copy_scale_strided_baked_with_index(
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
        index: &mut [usize],
    ) -> Result<(), OperationError> {
        let _ = index;
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
            baked,
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
///
/// Why not `Copy` or externally constructible: direct/unbaked calls retain
/// mutable normalization and traversal scratch. Compiled replay supplies its
/// traversal indices from the execution workspace; clones preserve
/// configuration but never share either scratch source.
#[derive(Debug, Default)]
pub struct StridedHostKernelAdapter {
    /// Selected transpose kernel for pure permuted copies (pack /
    /// assign-scatter); [`TransposeBackend::FusedLoops`] by default. See
    /// [`TransposeBackend`] for why `StridedPerm` is opt-in.
    pub transpose_backend: TransposeBackend,
    scratch: StridedKernelScratch,
}

impl Clone for StridedHostKernelAdapter {
    fn clone(&self) -> Self {
        Self::with_transpose_backend(self.transpose_backend)
    }
}

impl StridedHostKernelAdapter {
    /// Adapter with an explicit transpose kernel; `Default` is `FusedLoops`.
    #[inline]
    pub fn with_transpose_backend(transpose_backend: TransposeBackend) -> Self {
        Self {
            transpose_backend,
            scratch: StridedKernelScratch::default(),
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn fused_pair_baked_dispatch<T, Apply, ElementOp>(
        &mut self,
        index: Option<&mut [usize]>,
        baked: Option<BakedFusedLayout<'_>>,
        dst_data: &mut [T],
        src_data: &[T],
        shape: &[usize],
        dst_strides: &[isize],
        src_strides: &[isize],
        dst_offset: isize,
        src_offset: isize,
        apply: Apply,
        op: ElementOp,
    ) -> Result<(), OperationError>
    where
        T: Copy,
        Apply: Fn(&mut T, T),
        ElementOp: Fn(T) -> T,
    {
        match index {
            Some(index) => fused_pair_baked_with_index(
                &mut self.scratch.layout,
                baked,
                index,
                dst_data,
                src_data,
                shape,
                dst_strides,
                src_strides,
                dst_offset,
                src_offset,
                apply,
                op,
            ),
            None => fused_pair_baked(
                &mut self.scratch,
                baked,
                dst_data,
                src_data,
                shape,
                dst_strides,
                src_strides,
                dst_offset,
                src_offset,
                apply,
                op,
            ),
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn add_strided_baked_impl<T>(
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
        index: Option<&mut [usize]>,
    ) -> Result<(), OperationError>
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
        validate_strided_ranks(shape, dst_strides, src_strides)?;
        if beta.is_zero() || beta.is_one() {
            let assign = beta.is_zero();
            self.fused_pair_baked_dispatch(
                index,
                baked,
                dst_data,
                src_data,
                shape,
                dst_strides,
                src_strides,
                dst_offset,
                src_offset,
                move |dst, value| {
                    if assign {
                        *dst = value;
                    } else {
                        *dst = *dst + value;
                    }
                },
                move |value: T| alpha * value.maybe_conj(source_conjugate),
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

    #[allow(clippy::too_many_arguments)]
    fn axpby_strided_baked_impl<T>(
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
        index: Option<&mut [usize]>,
    ) -> Result<(), OperationError>
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
        validate_strided_ranks(shape, dst_strides, src_strides)?;
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
            return self.fused_pair_baked_dispatch(
                index,
                baked,
                dst_data,
                src_data,
                shape,
                dst_strides,
                src_strides,
                dst_offset,
                src_offset,
                move |dst, value| {
                    if assign {
                        *dst = value;
                    } else {
                        *dst = *dst + value;
                    }
                },
                move |value: T| alpha * value,
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

    #[allow(clippy::too_many_arguments)]
    fn copy_scale_strided_baked_impl<T>(
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
        index: Option<&mut [usize]>,
    ) -> Result<(), OperationError>
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
        validate_strided_ranks(shape, dst_strides, src_strides)?;
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
        self.fused_pair_baked_dispatch(
            index,
            baked,
            dst_data,
            src_data,
            shape,
            dst_strides,
            src_strides,
            dst_offset,
            src_offset,
            |dst, value| *dst = value,
            move |value: T| alpha * value.maybe_conj(source_conjugate),
        )
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
        self.add_strided_baked_impl(
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
            baked,
            None,
        )
    }

    fn add_strided_baked_with_index(
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
        index: &mut [usize],
    ) -> Result<(), OperationError> {
        self.add_strided_baked_impl(
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
            baked,
            Some(index),
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
        self.axpby_strided_baked_impl(
            dst_data,
            src_data,
            shape,
            dst_strides,
            src_strides,
            dst_offset,
            src_offset,
            alpha,
            beta,
            baked,
            None,
        )
    }

    fn axpby_strided_baked_with_index(
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
        index: &mut [usize],
    ) -> Result<(), OperationError> {
        self.axpby_strided_baked_impl(
            dst_data,
            src_data,
            shape,
            dst_strides,
            src_strides,
            dst_offset,
            src_offset,
            alpha,
            beta,
            baked,
            Some(index),
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
        self.copy_scale_strided_baked_impl(
            dst_data,
            src_data,
            shape,
            dst_strides,
            src_strides,
            dst_offset,
            src_offset,
            source_conjugate,
            alpha,
            baked,
            None,
        )
    }

    fn copy_scale_strided_baked_with_index(
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
        index: &mut [usize],
    ) -> Result<(), OperationError> {
        self.copy_scale_strided_baked_impl(
            dst_data,
            src_data,
            shape,
            dst_strides,
            src_strides,
            dst_offset,
            src_offset,
            source_conjugate,
            alpha,
            baked,
            Some(index),
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

    fn layout(shape: &[usize], dst: &[isize], src: &[isize]) -> FusedLayoutScratch {
        let mut layout = FusedLayoutScratch::default();
        normalize_fused_layout(shape, dst, src, &mut layout).unwrap();
        layout
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
    fn normalized_layout_drops_extent_one_axes_and_fuses_contiguous_runs() {
        // Extent-1 axis dropped regardless of its (garbage) strides, then the
        // two remaining contiguous axes fuse into one 6-element run.
        let fused = layout(&[2, 1, 3], &[1, 999, 2], &[1, -7, 2]);
        assert_eq!(fused.dims(), &[6]);
        assert_eq!(fused.dst_strides(), &[1]);
        assert_eq!(fused.src_strides(), &[1]);
    }

    #[test]
    fn normalized_layout_orders_axes_without_fusing_mismatched_source() {
        // Axes arrive in descending destination-stride order and must be
        // reordered ascending; destination strides are contiguous (1 * 2 == 2)
        // but source strides are not (3 * 2 != 1), so the axes must NOT fuse.
        let unfused = layout(&[3, 2], &[2, 1], &[1, 3]);
        assert_eq!(unfused.dims(), &[2, 3]);
        assert_eq!(unfused.dst_strides(), &[1, 2]);
        assert_eq!(unfused.src_strides(), &[3, 1]);
    }

    #[test]
    fn normalized_layout_zero_extent_collapses_to_empty_marker() {
        let empty = layout(&[2, 0, 3], &[1, 2, 4], &[1, 2, 4]);
        assert_eq!(empty.dims(), &[0]);
    }

    #[test]
    fn normalized_layout_all_extent_one_collapses_to_scalar() {
        let scalar = layout(&[1, 1], &[5, 3], &[2, 8]);
        assert_eq!(scalar.dims(), &[1]);
        assert_eq!(scalar.dst_strides(), &[0]);
        assert_eq!(scalar.src_strides(), &[0]);
    }

    #[test]
    fn zero_extent_replaces_prior_normalization_state() {
        let mut scratch = layout(&[2, 3], &[1, 2], &[1, 2]);
        normalize_fused_layout(&[2, 0, 3], &[1, 2, 4], &[1, 2, 4], &mut scratch).unwrap();
        assert_eq!(scratch.dims(), &[0]);
        assert_eq!(scratch.dst_strides(), &[0]);
        assert_eq!(scratch.src_strides(), &[0]);
    }

    #[test]
    fn normalized_layout_reports_element_count_overflow() {
        // What: normalization and sealed baked construction report the same
        // typed overflow instead of panicking or admitting an invalid token.
        let mut scratch = FusedLayoutScratch::default();
        assert_eq!(
            normalize_fused_layout(&[usize::MAX, 2], &[1, 2], &[1, 2], &mut scratch,).unwrap_err(),
            OperationError::ElementCountOverflow
        );
        assert_eq!(
            BakedFusedLayout::try_from_normalized_slices(&[usize::MAX, 2], &[1, 2], &[1, 2],)
                .unwrap_err(),
            OperationError::ElementCountOverflow
        );
        assert_eq!(
            normalize_fused_layout(&[usize::MAX, 2], &[1], &[1, 2], &mut scratch).unwrap_err(),
            OperationError::RankMismatch {
                expected: 2,
                actual: 1,
            }
        );
        assert_eq!(
            BakedFusedLayout::try_from_normalized_slices(&[usize::MAX, 2], &[1], &[1, 2],)
                .unwrap_err(),
            OperationError::RankMismatch {
                expected: 2,
                actual: 1,
            }
        );
    }

    #[test]
    fn fused_span_order_keeps_axis_one_fastest_with_negative_strides() {
        // What: the shared odometer preserves exact visit order and signed
        // offset arithmetic while using caller-owned runtime-rank scratch.
        let mut index = [0; 3];
        let mut visits = Vec::new();
        for_each_fused_span(
            &[3, 2, 2],
            &[1, 10, -100],
            &[2, -20, 200],
            300,
            60,
            &mut index,
            |dst, src, len, dst_stride, src_stride| {
                visits.push((dst, src, len, dst_stride, src_stride));
            },
        );
        assert_eq!(
            visits,
            [
                (300, 60, 3, 1, 2),
                (310, 40, 3, 1, 2),
                (200, 260, 3, 1, 2),
                (210, 240, 3, 1, 2),
            ]
        );
    }

    #[test]
    fn baked_fused_layout_accepts_runtime_rank_and_rejects_length_mismatches() {
        let empty_dims = [];
        let empty_strides = [];
        let dynamic_dims = [2usize; 9];
        let dynamic_strides = [1isize; 9];

        // What: only nonempty normalized slices with one stride per dimension
        // can become trusted replay tokens.
        assert_eq!(
            BakedFusedLayout::try_from_normalized_slices(
                &empty_dims,
                &empty_strides,
                &empty_strides
            )
            .unwrap_err(),
            OperationError::RankMismatch {
                expected: 1,
                actual: 0,
            }
        );
        assert!(BakedFusedLayout::try_from_normalized_slices(
            &dynamic_dims,
            &dynamic_strides,
            &dynamic_strides
        )
        .is_ok());
        assert_eq!(
            BakedFusedLayout::try_from_normalized_slices(&[0], &[1], &[0]).unwrap_err(),
            OperationError::InvalidArgument {
                message: "baked fused layout is not normalized",
            }
        );
        assert_eq!(
            BakedFusedLayout::try_from_normalized_slices(&[1], &[1], &[0]).unwrap_err(),
            OperationError::InvalidArgument {
                message: "baked fused layout is not normalized",
            }
        );

        let dims = [2usize, 3];
        let short = [1isize];
        let complete = [1isize, 2];
        assert_eq!(
            BakedFusedLayout::try_from_normalized_slices(&dims, &short, &complete).unwrap_err(),
            OperationError::RankMismatch {
                expected: 2,
                actual: 1,
            }
        );
        assert_eq!(
            BakedFusedLayout::try_from_normalized_slices(&dims, &complete, &short).unwrap_err(),
            OperationError::RankMismatch {
                expected: 2,
                actual: 1,
            }
        );
    }

    fn assert_baked_copy_matches_recomputed(shape: &[usize], strides: &[isize]) {
        let backing_len = shape
            .iter()
            .zip(strides)
            .map(|(&dim, &stride)| (dim - 1) * stride as usize)
            .sum::<usize>()
            + 1;
        let src = (0..backing_len)
            .map(|index| index as f64 + 0.25)
            .collect::<Vec<_>>();
        let mut expected = vec![-1.0; backing_len];
        let mut actual = expected.clone();
        let mut adapter = StridedHostKernelAdapter::default();

        adapter
            .copy_scale_strided(
                &mut expected,
                &src,
                shape,
                strides,
                strides,
                0,
                0,
                false,
                2.0,
            )
            .unwrap();
        let normalized = layout(shape, strides, strides);
        let baked = BakedFusedLayout::try_from_normalized_slices(
            normalized.dims(),
            normalized.dst_strides(),
            normalized.src_strides(),
        )
        .unwrap();
        adapter
            .copy_scale_strided_baked(
                &mut actual,
                &src,
                shape,
                strides,
                strides,
                0,
                0,
                false,
                2.0,
                Some(baked),
            )
            .unwrap();

        assert_eq!(actual, expected);
    }

    #[test]
    fn baked_fused_layout_matches_eager_for_dynamic_rank() {
        assert_baked_copy_matches_recomputed(
            &[2, 2, 2, 2, 2, 2, 2, 2, 2],
            &[1, 3, 9, 27, 81, 243, 729, 2187, 6561],
        );
    }

    #[test]
    fn public_baked_methods_reject_raw_rank_mismatches_before_dispatch() {
        let mut adapter = StridedHostKernelAdapter::default();
        let mut zero_strides = Vec::new();
        let mut dst = [0.0_f64; 2];
        let src = [1.0_f64; 2];
        let shape = [2usize];
        let strides = [1isize];
        let missing = [];
        let dst_error = OperationError::RankMismatch {
            expected: 1,
            actual: 0,
        };

        // What: all three public baked entry points validate both raw stride
        // ranks even without a baked token and on general-beta paths.
        assert_eq!(
            adapter
                .add_strided_baked(
                    &mut zero_strides,
                    &mut dst,
                    &src,
                    &shape,
                    &missing,
                    &strides,
                    0,
                    0,
                    false,
                    1.0,
                    2.0,
                    None,
                )
                .unwrap_err(),
            dst_error
        );
        assert_eq!(
            adapter
                .add_strided_baked(
                    &mut zero_strides,
                    &mut dst,
                    &src,
                    &shape,
                    &strides,
                    &missing,
                    0,
                    0,
                    false,
                    1.0,
                    2.0,
                    None,
                )
                .unwrap_err(),
            dst_error
        );
        assert_eq!(
            adapter
                .axpby_strided_baked(
                    &mut dst, &src, &shape, &missing, &strides, 0, 0, 1.0, 2.0, None,
                )
                .unwrap_err(),
            dst_error
        );
        assert_eq!(
            adapter
                .axpby_strided_baked(
                    &mut dst, &src, &shape, &strides, &missing, 0, 0, 1.0, 2.0, None,
                )
                .unwrap_err(),
            dst_error
        );
        assert_eq!(
            adapter
                .copy_scale_strided_baked(
                    &mut dst, &src, &shape, &missing, &strides, 0, 0, false, 1.0, None,
                )
                .unwrap_err(),
            dst_error
        );
        assert_eq!(
            adapter
                .copy_scale_strided_baked(
                    &mut dst, &src, &shape, &strides, &missing, 0, 0, false, 1.0, None,
                )
                .unwrap_err(),
            dst_error
        );
    }

    #[test]
    fn apply_fused_pair_copies_transposed_layout_exactly() {
        // dst[j * 2 + i] = 2 * src[i * 3 + j] over a logical [2, 3] iteration:
        // exact element placement through non-fusable permuted strides.
        let src = [1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0];
        let mut dst = [0.0_f64; 6];
        let mut scratch = StridedKernelScratch::default();
        fused_pair(
            &mut scratch,
            &mut dst,
            &src,
            &[2, 3],
            &[1, 2],
            &[3, 1],
            0,
            0,
            |dst, value| *dst = value,
            |value| 2.0 * value,
        )
        .unwrap();
        assert_eq!(dst, [2.0, 8.0, 4.0, 10.0, 6.0, 12.0]);
    }

    #[test]
    fn apply_fused_pair_accumulates_with_offsets() {
        let src = [0.0_f64, 1.0, 2.0];
        let mut dst = [10.0_f64, 20.0, 30.0];
        let mut scratch = StridedKernelScratch::default();
        fused_pair(
            &mut scratch,
            &mut dst,
            &src,
            &[2],
            &[1],
            &[1],
            1,
            1,
            |dst, value| *dst += value,
            |value| 3.0 * value,
        )
        .unwrap();
        assert_eq!(dst, [10.0, 23.0, 36.0]);
    }

    #[test]
    fn apply_fused_pair_zero_extent_is_a_noop() {
        let empty = layout(&[2, 0], &[1, 2], &[1, 2]);
        let src = [1.0_f64; 4];
        let mut dst = [7.0_f64; 4];
        let mut index = Vec::new();
        apply_fused_pair_slices(
            &mut dst,
            &src,
            empty.dims(),
            empty.dst_strides(),
            empty.src_strides(),
            0,
            0,
            &mut index,
            |dst, value| *dst = value,
            |value| value,
        );
        assert_eq!(dst, [7.0; 4]);
    }

    #[test]
    fn fused_pair_dynamic_rank_matches_reference() {
        let shape = [2usize; 9];
        let src_strides = row_major(&shape);
        let dst_strides: Vec<isize> = src_strides.iter().rev().copied().collect();
        let src: Vec<f64> = (0..512).map(|value| value as f64 - 100.0).collect();

        let mut dst = vec![0.0_f64; 512];
        let mut scratch = StridedKernelScratch::default();
        fused_pair(
            &mut scratch,
            &mut dst,
            &src,
            &shape,
            &dst_strides,
            &src_strides,
            0,
            0,
            |dst, value| *dst = value,
            |value| value,
        )
        .unwrap();

        let mut dst_reference = vec![0.0_f64; 512];
        reference_copy(&mut dst_reference, &src, &shape, &dst_strides, &src_strides);
        assert_eq!(dst, dst_reference);
    }

    #[test]
    fn fused_pair_zero_extent_shape_is_a_noop() {
        let src = [1.0_f64; 4];
        let mut dst = [9.0_f64; 4];
        let mut scratch = StridedKernelScratch::default();
        fused_pair(
            &mut scratch,
            &mut dst,
            &src,
            &[2, 0],
            &[1, 2],
            &[1, 2],
            0,
            0,
            |dst, value| *dst = value,
            |value| value,
        )
        .unwrap();
        assert_eq!(dst, [9.0; 4]);
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
}

/// Parity tests for the strided-perm transpose route (ported from prototype
/// commit b3ca6e5). These call `strided_perm_copy` and `fused_pair` directly
/// and assert byte-equality, independent of any runtime backend selection —
/// the routing contract (routing never changes results) holds for every
/// [`TransposeBackend`] value.
#[cfg(test)]
mod strided_perm_probe {
    use super::{
        fused_pair, strided_perm_copy, StridedHostKernelAdapter, StridedKernelScratch,
        TransposeBackend,
    };
    use crate::kernel_adapter::HostKernelAdapter;

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
        let mut scratch = StridedKernelScratch::default();
        fused_pair(
            &mut scratch,
            &mut fused,
            &src,
            shape,
            dst_strides,
            src_strides,
            dst_off,
            src_off,
            |d, v| *d = v,
            |v: f64| v,
        )
        .unwrap();
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

    /// Observes which route the adapter's `transpose_backend` field actually
    /// selects. Routed copies are byte-identical by design, so the routes are
    /// distinguished at the error boundary instead: on an eligible layout whose
    /// destination is one element too short, the strided-perm route returns a
    /// clean `Err` (StridedView bounding-box validation) while the fused loop
    /// panics on its safe indexing. Each behavior is unique to its route, so
    /// this pins that the adapter field — not an env var, not a global —
    /// switches the dispatch.
    #[test]
    fn transpose_backend_field_switches_the_route() {
        let shape = [4usize, 4];
        let ss = [4isize, 1];
        let ds = [1isize, 4]; // eligible transpose layout
        let src: Vec<f64> = (0..16).map(|i| i as f64).collect();

        // StridedPerm: view validation rejects the short destination cleanly.
        let mut perm_adapter =
            StridedHostKernelAdapter::with_transpose_backend(TransposeBackend::StridedPerm);
        let mut short_dst = vec![0.0f64; 15];
        assert!(
            perm_adapter
                .copy_scale_strided(&mut short_dst, &src, &shape, &ds, &ss, 0, 0, false, 1.0)
                .is_err(),
            "StridedPerm route must reject the out-of-bounds layout as an error"
        );

        // FusedLoops (default): the same call reaches the fused loop, whose
        // safe indexing panics on the out-of-bounds destination instead.
        let panicked = std::panic::catch_unwind(move || {
            let mut fused_adapter = StridedHostKernelAdapter::default();
            let mut short_dst = vec![0.0f64; 15];
            let _ = fused_adapter.copy_scale_strided(
                &mut short_dst,
                &src,
                &shape,
                &ds,
                &ss,
                0,
                0,
                false,
                1.0,
            );
        })
        .is_err();
        assert!(
            panicked,
            "FusedLoops route must take the fused (panicking) path"
        );
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
            let mut scratch = StridedKernelScratch::default();
            let iters = (1usize << 24 >> (2 * (n as f64).log2() as usize)).max(50);
            // warm caches
            fused_pair(
                &mut scratch,
                &mut dst,
                &src,
                &shape,
                &ds,
                &ss,
                0,
                0,
                |d, v| *d = v,
                |v: f64| v,
            )
            .unwrap();
            let _ = strided_perm_copy(&mut dst, &src, &shape, &ds, &ss, 0, 0);
            let t = Instant::now();
            for _ in 0..iters {
                fused_pair(
                    &mut scratch,
                    &mut dst,
                    &src,
                    &shape,
                    &ds,
                    &ss,
                    0,
                    0,
                    |d, v| *d = v,
                    |v: f64| v,
                )
                .unwrap();
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
