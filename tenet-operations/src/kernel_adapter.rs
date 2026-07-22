use core::ops::{Add, Mul};

use num_traits::{One, Zero};

use crate::{
    axpby_raw_strided_kernel_trusted, scale_raw_strided_kernel_trusted,
    tensoradd_raw_strided_kernel_trusted, ConjugateValue, OperationError,
    RecouplingCoefficientAction,
};

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
    pub(crate) fn from_compiled_normalized_slices(
        dims: &'a [usize],
        dst_strides: &'a [isize],
        src_strides: &'a [isize],
    ) -> Self {
        // Why not revalidate here: the compiler validates these exact arena
        // slices before publishing the immutable layout table.
        Self {
            dims,
            dst_strides,
            src_strides,
            _sealed: (),
        }
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
        if scratch.dst_strides[fused].checked_mul(extent) == Some(scratch.dst_strides[axis])
            && scratch.src_strides[fused].checked_mul(extent) == Some(scratch.src_strides[axis])
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
    scratch: StridedKernelScratch,
}

impl Clone for StridedHostKernelAdapter {
    fn clone(&self) -> Self {
        Self::default()
    }
}

impl StridedHostKernelAdapter {
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
