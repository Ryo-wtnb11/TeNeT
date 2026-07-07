#![forbid(unsafe_code)]

//! Dense block execution boundary for TeNeT.
//!
//! Symmetric tensor algorithms lower to this crate through TeNeT-owned storage
//! views and executors. The storage placement determines the execution path:
//! host views use host kernels, and future device views should use device
//! kernels without exposing concrete runtimes to TensorMap-level code.

use core::fmt;

use num_complex::{Complex32, Complex64};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DenseDType {
    F32,
    F64,
    I32,
    I64,
    Bool,
    C32,
    C64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DenseBackend {
    Tenferro,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DensePlacement {
    Host,
}

#[derive(Debug)]
pub struct DenseView<'a, T> {
    data: &'a [T],
    shape: &'a [usize],
    strides: &'a [usize],
    offset: usize,
}

impl<'a, T> Clone for DenseView<'a, T> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<'a, T> Copy for DenseView<'a, T> {}

impl<'a, T> DenseView<'a, T> {
    pub fn new(
        data: &'a [T],
        shape: &'a [usize],
        strides: &'a [usize],
        offset: usize,
    ) -> Result<Self, DenseError> {
        validate_dense_layout(data.len(), offset, shape, strides)?;
        Ok(Self {
            data,
            shape,
            strides,
            offset,
        })
    }

    /// Trusted constructor: the caller guarantees the layout was validated
    /// when the owning plan was compiled (replay-side counterpart of the
    /// `*_trusted` kernel entry points). Layout errors are still memory-safe
    /// (worst case an index panic downstream); debug builds re-validate.
    #[inline]
    pub fn new_trusted(
        data: &'a [T],
        shape: &'a [usize],
        strides: &'a [usize],
        offset: usize,
    ) -> Self {
        debug_assert!(validate_dense_layout(data.len(), offset, shape, strides).is_ok());
        Self {
            data,
            shape,
            strides,
            offset,
        }
    }

    #[inline]
    pub fn data(&self) -> &'a [T] {
        self.data
    }

    #[inline]
    pub fn shape(&self) -> &'a [usize] {
        self.shape
    }

    #[inline]
    pub fn strides(&self) -> &'a [usize] {
        self.strides
    }

    #[inline]
    pub fn offset(&self) -> usize {
        self.offset
    }

    #[inline]
    pub fn placement(&self) -> DensePlacement {
        DensePlacement::Host
    }
}

#[derive(Debug)]
pub struct DenseViewMut<'a, T> {
    data: &'a mut [T],
    shape: &'a [usize],
    strides: &'a [usize],
    offset: usize,
}

impl<'a, T> DenseViewMut<'a, T> {
    pub fn new(
        data: &'a mut [T],
        shape: &'a [usize],
        strides: &'a [usize],
        offset: usize,
    ) -> Result<Self, DenseError> {
        validate_dense_layout(data.len(), offset, shape, strides)?;
        Ok(Self {
            data,
            shape,
            strides,
            offset,
        })
    }

    /// Trusted constructor; see [`DenseView::new_trusted`].
    #[inline]
    pub fn new_trusted(
        data: &'a mut [T],
        shape: &'a [usize],
        strides: &'a [usize],
        offset: usize,
    ) -> Self {
        debug_assert!(validate_dense_layout(data.len(), offset, shape, strides).is_ok());
        Self {
            data,
            shape,
            strides,
            offset,
        }
    }

    #[inline]
    pub fn data(&self) -> &[T] {
        self.data
    }

    #[inline]
    pub fn data_mut(&mut self) -> &mut [T] {
        self.data
    }

    #[inline]
    pub fn shape(&self) -> &'a [usize] {
        self.shape
    }

    #[inline]
    pub fn strides(&self) -> &'a [usize] {
        self.strides
    }

    #[inline]
    pub fn offset(&self) -> usize {
        self.offset
    }

    #[inline]
    pub fn placement(&self) -> DensePlacement {
        DensePlacement::Host
    }
}

#[derive(Clone, Copy, Debug)]
pub enum DenseRead<'a> {
    F32(DenseView<'a, f32>),
    F64(DenseView<'a, f64>),
    I32(DenseView<'a, i32>),
    I64(DenseView<'a, i64>),
    Bool(DenseView<'a, bool>),
    C32(DenseView<'a, Complex32>),
    C64(DenseView<'a, Complex64>),
}

impl DenseRead<'_> {
    pub fn dtype(&self) -> DenseDType {
        match self {
            Self::F32(_) => DenseDType::F32,
            Self::F64(_) => DenseDType::F64,
            Self::I32(_) => DenseDType::I32,
            Self::I64(_) => DenseDType::I64,
            Self::Bool(_) => DenseDType::Bool,
            Self::C32(_) => DenseDType::C32,
            Self::C64(_) => DenseDType::C64,
        }
    }

    pub fn shape(&self) -> &[usize] {
        match self {
            Self::F32(view) => view.shape(),
            Self::F64(view) => view.shape(),
            Self::I32(view) => view.shape(),
            Self::I64(view) => view.shape(),
            Self::Bool(view) => view.shape(),
            Self::C32(view) => view.shape(),
            Self::C64(view) => view.shape(),
        }
    }

    pub fn placement(&self) -> DensePlacement {
        match self {
            Self::F32(view) => view.placement(),
            Self::F64(view) => view.placement(),
            Self::I32(view) => view.placement(),
            Self::I64(view) => view.placement(),
            Self::Bool(view) => view.placement(),
            Self::C32(view) => view.placement(),
            Self::C64(view) => view.placement(),
        }
    }
}

#[derive(Debug)]
pub enum DenseWrite<'a> {
    F32(DenseViewMut<'a, f32>),
    F64(DenseViewMut<'a, f64>),
    I32(DenseViewMut<'a, i32>),
    I64(DenseViewMut<'a, i64>),
    Bool(DenseViewMut<'a, bool>),
    C32(DenseViewMut<'a, Complex32>),
    C64(DenseViewMut<'a, Complex64>),
}

impl DenseWrite<'_> {
    pub fn dtype(&self) -> DenseDType {
        match self {
            Self::F32(_) => DenseDType::F32,
            Self::F64(_) => DenseDType::F64,
            Self::I32(_) => DenseDType::I32,
            Self::I64(_) => DenseDType::I64,
            Self::Bool(_) => DenseDType::Bool,
            Self::C32(_) => DenseDType::C32,
            Self::C64(_) => DenseDType::C64,
        }
    }

    pub fn shape(&self) -> &[usize] {
        match self {
            Self::F32(view) => view.shape(),
            Self::F64(view) => view.shape(),
            Self::I32(view) => view.shape(),
            Self::I64(view) => view.shape(),
            Self::Bool(view) => view.shape(),
            Self::C32(view) => view.shape(),
            Self::C64(view) => view.shape(),
        }
    }

    pub fn placement(&self) -> DensePlacement {
        match self {
            Self::F32(view) => view.placement(),
            Self::F64(view) => view.placement(),
            Self::I32(view) => view.placement(),
            Self::I64(view) => view.placement(),
            Self::Bool(view) => view.placement(),
            Self::C32(view) => view.placement(),
            Self::C64(view) => view.placement(),
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DenseDotConfig {
    lhs_contracting_dims: Vec<usize>,
    rhs_contracting_dims: Vec<usize>,
    lhs_batch_dims: Vec<usize>,
    rhs_batch_dims: Vec<usize>,
    // Elementwise conjugation of the operands, folded into the contraction
    // kernel (BLAS Aᴴ forms / conjugating accumulator) rather than
    // materialized. Flags are in dot-operand order, i.e. `lhs_conj` applies to
    // the first operand passed to `dot_general_into`, which may be the caller's
    // rhs after a route-order swap — the caller is responsible for that mapping.
    lhs_conj: bool,
    rhs_conj: bool,
}

impl DenseDotConfig {
    pub fn new(
        lhs_contracting_dims: Vec<usize>,
        rhs_contracting_dims: Vec<usize>,
        lhs_batch_dims: Vec<usize>,
        rhs_batch_dims: Vec<usize>,
    ) -> Self {
        Self {
            lhs_contracting_dims,
            rhs_contracting_dims,
            lhs_batch_dims,
            rhs_batch_dims,
            lhs_conj: false,
            rhs_conj: false,
        }
    }

    /// Set elementwise conjugation of the operands (dot-operand order).
    pub fn with_conjugation(mut self, lhs_conj: bool, rhs_conj: bool) -> Self {
        self.lhs_conj = lhs_conj;
        self.rhs_conj = rhs_conj;
        self
    }

    pub fn matmul() -> Self {
        Self::new(vec![1], vec![0], Vec::new(), Vec::new())
    }

    #[inline]
    pub fn lhs_conj(&self) -> bool {
        self.lhs_conj
    }

    #[inline]
    pub fn rhs_conj(&self) -> bool {
        self.rhs_conj
    }

    #[inline]
    pub fn lhs_contracting_dims(&self) -> &[usize] {
        &self.lhs_contracting_dims
    }

    #[inline]
    pub fn rhs_contracting_dims(&self) -> &[usize] {
        &self.rhs_contracting_dims
    }

    #[inline]
    pub fn lhs_batch_dims(&self) -> &[usize] {
        &self.lhs_batch_dims
    }

    #[inline]
    pub fn rhs_batch_dims(&self) -> &[usize] {
        &self.rhs_batch_dims
    }
}

#[derive(Clone, Debug)]
pub struct DenseTensor {
    backend: DenseBackend,
    inner: DenseTensorInner,
}

#[derive(Clone, Debug)]
enum DenseTensorInner {
    #[cfg(feature = "tenferro")]
    Tenferro(tenferro_tensor::Tensor),
    #[cfg(not(feature = "tenferro"))]
    #[allow(dead_code)]
    Empty(std::convert::Infallible),
}

impl DenseTensor {
    #[inline]
    pub fn backend(&self) -> DenseBackend {
        self.backend
    }

    pub fn dtype(&self) -> DenseDType {
        match &self.inner {
            #[cfg(feature = "tenferro")]
            DenseTensorInner::Tenferro(tensor) => dense_dtype_from_tenferro(tensor.dtype()),
            #[cfg(not(feature = "tenferro"))]
            DenseTensorInner::Empty(inner) => match *inner {},
        }
    }

    pub fn shape(&self) -> &[usize] {
        match &self.inner {
            #[cfg(feature = "tenferro")]
            DenseTensorInner::Tenferro(tensor) => tensor.shape(),
            #[cfg(not(feature = "tenferro"))]
            DenseTensorInner::Empty(inner) => match *inner {},
        }
    }

    pub fn as_f32_slice(&self) -> Result<&[f32], DenseError> {
        match &self.inner {
            #[cfg(feature = "tenferro")]
            DenseTensorInner::Tenferro(tensor) => tensor
                .as_slice::<f32>()
                .map_err(|err| tenferro_error("DenseTensor::as_f32_slice", err)),
            #[cfg(not(feature = "tenferro"))]
            DenseTensorInner::Empty(inner) => match *inner {},
        }
    }

    pub fn as_f64_slice(&self) -> Result<&[f64], DenseError> {
        match &self.inner {
            #[cfg(feature = "tenferro")]
            DenseTensorInner::Tenferro(tensor) => tensor
                .as_slice::<f64>()
                .map_err(|err| tenferro_error("DenseTensor::as_f64_slice", err)),
            #[cfg(not(feature = "tenferro"))]
            DenseTensorInner::Empty(inner) => match *inner {},
        }
    }

    pub fn as_c32_slice(&self) -> Result<&[Complex32], DenseError> {
        match &self.inner {
            #[cfg(feature = "tenferro")]
            DenseTensorInner::Tenferro(tensor) => tensor
                .as_slice::<Complex32>()
                .map_err(|err| tenferro_error("DenseTensor::as_c32_slice", err)),
            #[cfg(not(feature = "tenferro"))]
            DenseTensorInner::Empty(inner) => match *inner {},
        }
    }

    pub fn as_c64_slice(&self) -> Result<&[Complex64], DenseError> {
        match &self.inner {
            #[cfg(feature = "tenferro")]
            DenseTensorInner::Tenferro(tensor) => tensor
                .as_slice::<Complex64>()
                .map_err(|err| tenferro_error("DenseTensor::as_c64_slice", err)),
            #[cfg(not(feature = "tenferro"))]
            DenseTensorInner::Empty(inner) => match *inner {},
        }
    }

    #[cfg(feature = "tenferro")]
    fn from_tenferro(tensor: tenferro_tensor::Tensor) -> Self {
        Self {
            backend: DenseBackend::Tenferro,
            inner: DenseTensorInner::Tenferro(tensor),
        }
    }
}

/// Dtype-erased GEMM scalar for the accumulate-form matmul seam
/// (`C = alpha * A * B + beta * C`). Mirrors the BLAS/cuTENSOR parameter
/// shape so backends can consume it without generics.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum DenseScalar {
    F32(f32),
    F64(f64),
    C32(Complex32),
    C64(Complex64),
}

impl DenseScalar {
    pub fn is_one(&self) -> bool {
        match self {
            Self::F32(value) => *value == 1.0,
            Self::F64(value) => *value == 1.0,
            Self::C32(value) => *value == Complex32::new(1.0, 0.0),
            Self::C64(value) => *value == Complex64::new(1.0, 0.0),
        }
    }

    pub fn is_zero(&self) -> bool {
        match self {
            Self::F32(value) => *value == 0.0,
            Self::F64(value) => *value == 0.0,
            Self::C32(value) => *value == Complex32::new(0.0, 0.0),
            Self::C64(value) => *value == Complex64::new(0.0, 0.0),
        }
    }
}

/// One GEMM of a batched matmul over shared flat buffers: the column-major
/// `rows x cols` destination block at `dst_offset` receives
/// `alpha * lhs_block * rhs_block + beta * dst_block`. Offsets are element
/// offsets relative to the corresponding view's own offset. Callers guarantee
/// the destination blocks of a batch are pairwise disjoint, so executors may
/// run jobs in any order or concurrently.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DenseGemmBatchJob {
    pub dst_offset: usize,
    pub lhs_offset: usize,
    pub rhs_offset: usize,
    pub rows: usize,
    pub contracted: usize,
    pub cols: usize,
}

pub trait DenseExecutor {
    fn svd(&mut self, input: DenseRead<'_>) -> Result<Vec<DenseTensor>, DenseError>;
    fn qr(&mut self, input: DenseRead<'_>) -> Result<Vec<DenseTensor>, DenseError>;
    fn eigh(&mut self, input: DenseRead<'_>) -> Result<Vec<DenseTensor>, DenseError>;

    fn svd_into(
        &mut self,
        input: DenseRead<'_>,
        u: DenseWrite<'_>,
        s: DenseWrite<'_>,
        vt: DenseWrite<'_>,
    ) -> Result<(), DenseError> {
        let outputs = self.svd(input)?;
        if outputs.len() != 3 {
            return Err(DenseError::Backend {
                backend: DenseBackend::Tenferro,
                op: "svd_into",
                message: "dense SVD must return exactly (U, S, Vt)".to_string(),
            });
        }
        copy_dense_tensor_into(&outputs[0], u, "svd_into")?;
        copy_dense_tensor_into(&outputs[1], s, "svd_into")?;
        copy_dense_tensor_into(&outputs[2], vt, "svd_into")
    }

    fn qr_into(
        &mut self,
        input: DenseRead<'_>,
        q: DenseWrite<'_>,
        r: DenseWrite<'_>,
    ) -> Result<(), DenseError> {
        let outputs = self.qr(input)?;
        if outputs.len() != 2 {
            return Err(DenseError::Backend {
                backend: DenseBackend::Tenferro,
                op: "qr_into",
                message: "dense QR must return exactly (Q, R)".to_string(),
            });
        }
        copy_dense_tensor_into(&outputs[0], q, "qr_into")?;
        copy_dense_tensor_into(&outputs[1], r, "qr_into")
    }

    fn eigh_into(
        &mut self,
        input: DenseRead<'_>,
        values: DenseWrite<'_>,
        vectors: DenseWrite<'_>,
    ) -> Result<(), DenseError> {
        let outputs = self.eigh(input)?;
        if outputs.len() != 2 {
            return Err(DenseError::Backend {
                backend: DenseBackend::Tenferro,
                op: "eigh_into",
                message: "dense EIGH must return exactly (values, vectors)".to_string(),
            });
        }
        copy_dense_tensor_into(&outputs[0], values, "eigh_into")?;
        copy_dense_tensor_into(&outputs[1], vectors, "eigh_into")
    }

    /// General (non-Hermitian) eigendecomposition `(values, vectors)`; both
    /// outputs are complex regardless of the input scalar.
    fn eig(&mut self, input: DenseRead<'_>) -> Result<Vec<DenseTensor>, DenseError> {
        let _ = input;
        Err(DenseError::Backend {
            backend: DenseBackend::Tenferro,
            op: "eig",
            message: "executor does not implement the general eigendecomposition".to_string(),
        })
    }

    fn dot_general_into(
        &mut self,
        output: DenseWrite<'_>,
        lhs: DenseRead<'_>,
        rhs: DenseRead<'_>,
        config: &DenseDotConfig,
    ) -> Result<(), DenseError>;

    fn matmul_into(
        &mut self,
        output: DenseWrite<'_>,
        lhs: DenseRead<'_>,
        rhs: DenseRead<'_>,
    ) -> Result<(), DenseError> {
        self.dot_general_into(output, lhs, rhs, &DenseDotConfig::matmul())
    }

    /// Accumulate-form matmul: `output = alpha * lhs * rhs + beta * output`
    /// (BLAS gemm semantics). The default supports only the overwrite case
    /// `alpha = 1, beta = 0`; accumulate-capable backends override it.
    fn matmul_axpby_into(
        &mut self,
        output: DenseWrite<'_>,
        lhs: DenseRead<'_>,
        rhs: DenseRead<'_>,
        alpha: DenseScalar,
        beta: DenseScalar,
    ) -> Result<(), DenseError> {
        if alpha.is_one() && beta.is_zero() {
            return self.matmul_into(output, lhs, rhs);
        }
        Err(DenseError::Backend {
            backend: DenseBackend::Tenferro,
            op: "matmul_axpby_into",
            message: "executor does not implement the accumulate-form matmul".to_string(),
        })
    }

    /// Batched accumulate-form matmul over shared flat buffers: for each job,
    /// the destination block receives `alpha * lhs_block * rhs_block + beta *
    /// dst_block` (column-major, BLAS gemm semantics; see
    /// [`DenseGemmBatchJob`]). The default executes the jobs serially through
    /// `matmul_axpby_into`; batch-capable backends override it.
    fn matmul_batch_axpby_into(
        &mut self,
        output: DenseWrite<'_>,
        lhs: DenseRead<'_>,
        rhs: DenseRead<'_>,
        jobs: &[DenseGemmBatchJob],
        alpha: DenseScalar,
        beta: DenseScalar,
    ) -> Result<(), DenseError> {
        match (output, lhs, rhs) {
            (DenseWrite::F32(out), DenseRead::F32(lhs), DenseRead::F32(rhs)) => {
                matmul_batch_axpby_serial(
                    self,
                    out,
                    lhs,
                    rhs,
                    jobs,
                    alpha,
                    beta,
                    |view: DenseViewMut<'_, f32>| DenseWrite::F32(view),
                    |view: DenseView<'_, f32>| DenseRead::F32(view),
                )
            }
            (DenseWrite::F64(out), DenseRead::F64(lhs), DenseRead::F64(rhs)) => {
                matmul_batch_axpby_serial(
                    self,
                    out,
                    lhs,
                    rhs,
                    jobs,
                    alpha,
                    beta,
                    |view: DenseViewMut<'_, f64>| DenseWrite::F64(view),
                    |view: DenseView<'_, f64>| DenseRead::F64(view),
                )
            }
            (DenseWrite::C32(out), DenseRead::C32(lhs), DenseRead::C32(rhs)) => {
                matmul_batch_axpby_serial(
                    self,
                    out,
                    lhs,
                    rhs,
                    jobs,
                    alpha,
                    beta,
                    |view: DenseViewMut<'_, Complex32>| DenseWrite::C32(view),
                    |view: DenseView<'_, Complex32>| DenseRead::C32(view),
                )
            }
            (DenseWrite::C64(out), DenseRead::C64(lhs), DenseRead::C64(rhs)) => {
                matmul_batch_axpby_serial(
                    self,
                    out,
                    lhs,
                    rhs,
                    jobs,
                    alpha,
                    beta,
                    |view: DenseViewMut<'_, Complex64>| DenseWrite::C64(view),
                    |view: DenseView<'_, Complex64>| DenseRead::C64(view),
                )
            }
            _ => Err(DenseError::Backend {
                backend: DenseBackend::Tenferro,
                op: "matmul_batch_axpby_into",
                message: "batched matmul requires matching f32/f64/c32/c64 operands".to_string(),
            }),
        }
    }
}

fn copy_dense_tensor_into(
    tensor: &DenseTensor,
    output: DenseWrite<'_>,
    op: &'static str,
) -> Result<(), DenseError> {
    match output {
        DenseWrite::F32(output) => {
            copy_contiguous_tensor_into_view(tensor.as_f32_slice()?, tensor.shape(), output, op)
        }
        DenseWrite::F64(output) => {
            copy_contiguous_tensor_into_view(tensor.as_f64_slice()?, tensor.shape(), output, op)
        }
        DenseWrite::C32(output) => {
            copy_contiguous_tensor_into_view(tensor.as_c32_slice()?, tensor.shape(), output, op)
        }
        DenseWrite::C64(output) => {
            copy_contiguous_tensor_into_view(tensor.as_c64_slice()?, tensor.shape(), output, op)
        }
        DenseWrite::I32(_) | DenseWrite::I64(_) | DenseWrite::Bool(_) => Err(DenseError::Backend {
            backend: DenseBackend::Tenferro,
            op,
            message: format!("{op} outputs require f32/f64/c32/c64 destination views"),
        }),
    }
}

fn copy_contiguous_tensor_into_view<T: Copy>(
    source: &[T],
    source_shape: &[usize],
    mut output: DenseViewMut<'_, T>,
    op: &'static str,
) -> Result<(), DenseError> {
    if source_shape != output.shape() {
        return Err(DenseError::Backend {
            backend: DenseBackend::Tenferro,
            op,
            message: format!(
                "{op} output shape mismatch: source {:?}, destination {:?}",
                source_shape,
                output.shape()
            ),
        });
    }
    let expected = source_shape.iter().try_fold(1usize, |acc, &dim| {
        acc.checked_mul(dim).ok_or(DenseError::ElementCountOverflow)
    })?;
    if source.len() != expected {
        return Err(DenseError::Backend {
            backend: DenseBackend::Tenferro,
            op,
            message: format!(
                "{op} output storage length mismatch: source {}, expected {}",
                source.len(),
                expected
            ),
        });
    }
    if expected == 0 {
        return Ok(());
    }
    if source_shape.is_empty() {
        let offset = output.offset();
        output.data_mut()[offset] = source[0];
        return Ok(());
    }

    let shape = output.shape().to_vec();
    let strides = output.strides().to_vec();
    let offset = output.offset();
    let run = shape[0];
    let outer_count = shape[1..].iter().product::<usize>();
    let mut index = vec![0usize; shape.len()];
    let data = output.data_mut();
    for outer in 0..outer_count {
        let src_start = outer * run;
        let mut dst_start = offset;
        for axis in 1..shape.len() {
            dst_start += index[axis] * strides[axis];
        }
        if strides[0] == 1 {
            data[dst_start..dst_start + run].copy_from_slice(&source[src_start..src_start + run]);
        } else {
            for lane in 0..run {
                data[dst_start + lane * strides[0]] = source[src_start + lane];
            }
        }
        for axis in 1..shape.len() {
            index[axis] += 1;
            if index[axis] < shape[axis] {
                break;
            }
            index[axis] = 0;
        }
    }
    Ok(())
}

fn batch_offset(base: usize, offset: usize) -> Result<usize, DenseError> {
    base.checked_add(offset)
        .ok_or(DenseError::OffsetOverflow { value: offset })
}

fn matrix_len(rows: usize, cols: usize) -> Result<usize, DenseError> {
    rows.checked_mul(cols)
        .ok_or(DenseError::ElementCountOverflow)
}

fn same_gemm_shape(lhs: &DenseGemmBatchJob, rhs: &DenseGemmBatchJob) -> bool {
    lhs.rows == rhs.rows && lhs.contracted == rhs.contracted && lhs.cols == rhs.cols
}

fn strided_batch_run_len(jobs: &[DenseGemmBatchJob], start: usize) -> usize {
    let Some(first) = jobs.get(start) else {
        return 0;
    };
    let Some(second) = jobs.get(start + 1) else {
        return 1;
    };
    if !same_gemm_shape(first, second) {
        return 1;
    }
    let Some(dst_stride) = second.dst_offset.checked_sub(first.dst_offset) else {
        return 1;
    };
    if dst_stride == 0 {
        return 1;
    }
    let Some(lhs_stride) = second.lhs_offset.checked_sub(first.lhs_offset) else {
        return 1;
    };
    let Some(rhs_stride) = second.rhs_offset.checked_sub(first.rhs_offset) else {
        return 1;
    };

    let mut len = 2usize;
    while let Some(next) = jobs.get(start + len) {
        let prev = &jobs[start + len - 1];
        if !same_gemm_shape(first, next) {
            break;
        }
        if prev.dst_offset.checked_add(dst_stride) != Some(next.dst_offset)
            || prev.lhs_offset.checked_add(lhs_stride) != Some(next.lhs_offset)
            || prev.rhs_offset.checked_add(rhs_stride) != Some(next.rhs_offset)
        {
            break;
        }
        len += 1;
    }
    len
}

fn has_strided_batch_run(jobs: &[DenseGemmBatchJob]) -> bool {
    let mut start = 0usize;
    while start < jobs.len() {
        let run_len = strided_batch_run_len(jobs, start);
        if run_len > 1 {
            return true;
        }
        start += run_len;
    }
    false
}

/// Serial fallback for [`DenseExecutor::matmul_batch_axpby_into`]: one
/// `matmul_axpby_into` per job over rank-2 sub-views of the shared buffers.
#[allow(clippy::too_many_arguments)]
fn matmul_batch_axpby_serial<E, T, W, R>(
    executor: &mut E,
    mut output: DenseViewMut<'_, T>,
    lhs: DenseView<'_, T>,
    rhs: DenseView<'_, T>,
    jobs: &[DenseGemmBatchJob],
    alpha: DenseScalar,
    beta: DenseScalar,
    wrap_write: W,
    wrap_read: R,
) -> Result<(), DenseError>
where
    E: DenseExecutor + ?Sized,
    W: for<'x> Fn(DenseViewMut<'x, T>) -> DenseWrite<'x>,
    R: for<'x> Fn(DenseView<'x, T>) -> DenseRead<'x>,
{
    for job in jobs {
        let lhs_shape = [job.rows, job.contracted];
        let lhs_strides = [1, job.rows];
        let rhs_shape = [job.contracted, job.cols];
        let rhs_strides = [1, job.contracted];
        let dst_shape = [job.rows, job.cols];
        let dst_strides = [1, job.rows];
        let lhs_view = DenseView::new(
            lhs.data(),
            &lhs_shape,
            &lhs_strides,
            batch_offset(lhs.offset(), job.lhs_offset)?,
        )?;
        let rhs_view = DenseView::new(
            rhs.data(),
            &rhs_shape,
            &rhs_strides,
            batch_offset(rhs.offset(), job.rhs_offset)?,
        )?;
        let dst_offset = batch_offset(output.offset(), job.dst_offset)?;
        let dst_view = DenseViewMut::new(output.data_mut(), &dst_shape, &dst_strides, dst_offset)?;
        executor.matmul_axpby_into(
            wrap_write(dst_view),
            wrap_read(lhs_view),
            wrap_read(rhs_view),
            alpha,
            beta,
        )?;
    }
    Ok(())
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum DenseError {
    RankMismatch {
        shape: usize,
        strides: usize,
    },
    ElementCountOverflow,
    StrideOverflow {
        value: usize,
    },
    OffsetOverflow {
        value: usize,
    },
    OutOfBounds,
    Backend {
        backend: DenseBackend,
        op: &'static str,
        message: String,
    },
}

impl fmt::Display for DenseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::RankMismatch { shape, strides } => {
                write!(
                    f,
                    "rank mismatch: shape rank {shape}, strides rank {strides}"
                )
            }
            Self::ElementCountOverflow => write!(f, "dense view element count overflow"),
            Self::StrideOverflow { value } => {
                write!(f, "dense view stride {value} does not fit in isize")
            }
            Self::OffsetOverflow { value } => {
                write!(f, "dense view offset {value} does not fit in isize")
            }
            Self::OutOfBounds => write!(f, "dense view accesses outside the buffer"),
            Self::Backend {
                backend,
                op,
                message,
            } => {
                write!(f, "{backend:?} backend error in {op}: {message}")
            }
        }
    }
}

impl std::error::Error for DenseError {}

fn validate_dense_layout(
    len: usize,
    offset: usize,
    shape: &[usize],
    strides: &[usize],
) -> Result<(), DenseError> {
    if shape.len() != strides.len() {
        return Err(DenseError::RankMismatch {
            shape: shape.len(),
            strides: strides.len(),
        });
    }
    if shape.iter().any(|&dim| dim == 0) {
        return if offset <= len {
            Ok(())
        } else {
            Err(DenseError::OutOfBounds)
        };
    }
    if offset >= len {
        return Err(DenseError::OutOfBounds);
    }
    let max_delta = max_offset_delta(shape, strides)?;
    let last = offset
        .checked_add(max_delta)
        .ok_or(DenseError::OffsetOverflow { value: offset })?;
    if last < len {
        Ok(())
    } else {
        Err(DenseError::OutOfBounds)
    }
}

fn max_offset_delta(shape: &[usize], strides: &[usize]) -> Result<usize, DenseError> {
    shape
        .iter()
        .zip(strides)
        .try_fold(0usize, |acc, (&dim, &stride)| {
            let steps = dim.saturating_sub(1);
            let delta = steps
                .checked_mul(stride)
                .ok_or(DenseError::StrideOverflow { value: stride })?;
            acc.checked_add(delta)
                .ok_or(DenseError::ElementCountOverflow)
        })
}

#[cfg(feature = "tenferro")]
pub use tenferro_adapter::DefaultDenseExecutor;

#[cfg(feature = "cuda")]
pub use cuda_adapter::{
    cuda_eigh_region, cuda_gemm_region_into, cuda_matmul_region_into, cuda_qr_region,
    cuda_svd_region, CudaDenseContext, CudaDenseStorage,
};

#[cfg(feature = "cuda")]
mod cuda_adapter;

#[cfg(feature = "tenferro")]
mod tenferro_adapter {
    use super::*;

    use tenferro_cpu::CpuBackend;
    use tenferro_linalg::LinalgBackend;
    use tenferro_tensor::backend::{GroupedGemmConfig, GroupedGemmJob};
    use tenferro_tensor::{
        BackendCachedDot, BackendRuntimeCache, DotGeneralConfig, Tensor, TensorDot, TensorRead,
        TensorView, TensorViewMut, TensorWrite, TypedTensorView, TypedTensorViewMut,
    };

    #[derive(Debug)]
    pub struct DefaultDenseExecutor {
        backend: CpuBackend,
        matmul_config: DotGeneralConfig,
        strided_batch_matmul_config: DotGeneralConfig,
        grouped_cache: <CpuBackend as BackendRuntimeCache>::RuntimeCache,
        grouped_jobs: Vec<GroupedGemmJob>,
        #[cfg(test)]
        logical_gemm_dispatches: usize,
    }

    impl DefaultDenseExecutor {
        pub fn new() -> Self {
            Self::from_backend(CpuBackend::new())
        }

        pub fn with_threads(threads: usize) -> Result<Self, DenseError> {
            CpuBackend::with_threads(threads)
                .map(Self::from_backend)
                .map_err(|err| tenferro_error("CpuBackend::with_threads", err))
        }

        fn from_backend(backend: CpuBackend) -> Self {
            Self {
                backend,
                matmul_config: DotGeneralConfig {
                    lhs_contracting_dims: vec![1],
                    rhs_contracting_dims: vec![0],
                    lhs_batch_dims: Vec::new(),
                    rhs_batch_dims: Vec::new(),
                },
                strided_batch_matmul_config: DotGeneralConfig {
                    lhs_contracting_dims: vec![1],
                    rhs_contracting_dims: vec![0],
                    lhs_batch_dims: vec![2],
                    rhs_batch_dims: vec![2],
                },
                grouped_cache: <CpuBackend as BackendRuntimeCache>::RuntimeCache::default(),
                grouped_jobs: Vec::new(),
                #[cfg(test)]
                logical_gemm_dispatches: 0,
            }
        }

        #[cfg(test)]
        pub(crate) fn reset_logical_gemm_dispatches(&mut self) {
            self.logical_gemm_dispatches = 0;
        }

        #[cfg(test)]
        pub(crate) fn logical_gemm_dispatches(&self) -> usize {
            self.logical_gemm_dispatches
        }

        #[allow(clippy::too_many_arguments)]
        fn matmul_batch_axpby_grouped(
            &mut self,
            output: DenseWrite<'_>,
            lhs: DenseRead<'_>,
            rhs: DenseRead<'_>,
            jobs: &[DenseGemmBatchJob],
            alpha: DenseScalar,
            beta: DenseScalar,
        ) -> Result<(), DenseError> {
            #[cfg(test)]
            {
                self.logical_gemm_dispatches += jobs.len();
            }
            let lhs = TensorRead::from_view(tenferro_view(lhs)?);
            let rhs = TensorRead::from_view(tenferro_view(rhs)?);
            let output = TensorWrite::from_view(tenferro_view_mut(output)?);
            self.grouped_jobs.clear();
            self.grouped_jobs.extend(jobs.iter().map(|job| {
                GroupedGemmJob::new(
                    job.dst_offset,
                    job.lhs_offset,
                    job.rhs_offset,
                    job.rows,
                    job.contracted,
                    job.cols,
                )
            }));
            let accumulation = tenferro_tensor::DotGeneralAccumulation {
                lhs_conj: false,
                rhs_conj: false,
                alpha: tenferro_scalar(alpha),
                beta: tenferro_scalar(beta),
            };
            let config = GroupedGemmConfig::new(&self.grouped_jobs, accumulation);
            BackendCachedDot::grouped_gemm_cached(
                &mut self.backend,
                &mut self.grouped_cache,
                None,
                lhs,
                rhs,
                &config,
                output,
            )
            .map_err(|err| tenferro_error("grouped_gemm", err))
        }

        #[allow(clippy::too_many_arguments)]
        fn matmul_batch_axpby_strided_typed<T, W, R>(
            &mut self,
            output: &mut DenseViewMut<'_, T>,
            lhs: DenseView<'_, T>,
            rhs: DenseView<'_, T>,
            jobs: &[DenseGemmBatchJob],
            alpha: DenseScalar,
            beta: DenseScalar,
            wrap_write: W,
            wrap_read: R,
        ) -> Result<bool, DenseError>
        where
            T: 'static,
            W: for<'x> Fn(DenseViewMut<'x, T>) -> DenseWrite<'x> + Copy,
            R: for<'x> Fn(DenseView<'x, T>) -> DenseRead<'x> + Copy,
        {
            if jobs.len() < 2 {
                return Ok(false);
            }

            if has_strided_batch_run(jobs) {
                self.matmul_batch_axpby_strided_runs_typed(
                    output, lhs, rhs, jobs, 0, alpha, beta, wrap_write, wrap_read,
                )?;
                return Ok(true);
            }

            Ok(false)
        }

        #[allow(clippy::too_many_arguments)]
        fn matmul_batch_axpby_strided_runs_typed<T, W, R>(
            &mut self,
            output: &mut DenseViewMut<'_, T>,
            lhs: DenseView<'_, T>,
            rhs: DenseView<'_, T>,
            jobs: &[DenseGemmBatchJob],
            cache_slot_base: usize,
            alpha: DenseScalar,
            beta: DenseScalar,
            wrap_write: W,
            wrap_read: R,
        ) -> Result<(), DenseError>
        where
            T: 'static,
            W: for<'x> Fn(DenseViewMut<'x, T>) -> DenseWrite<'x> + Copy,
            R: for<'x> Fn(DenseView<'x, T>) -> DenseRead<'x> + Copy,
        {
            let mut start = 0usize;
            while start < jobs.len() {
                let run_len = strided_batch_run_len(jobs, start);
                self.matmul_strided_batch_run_typed(
                    output,
                    lhs,
                    rhs,
                    &jobs[start..start + run_len],
                    cache_slot_base + start,
                    alpha,
                    beta,
                    wrap_write,
                    wrap_read,
                )?;
                start += run_len;
            }
            Ok(())
        }

        #[allow(clippy::too_many_arguments)]
        fn matmul_strided_batch_run_typed<T, W, R>(
            &mut self,
            output: &mut DenseViewMut<'_, T>,
            lhs: DenseView<'_, T>,
            rhs: DenseView<'_, T>,
            run: &[DenseGemmBatchJob],
            cache_slot: usize,
            alpha: DenseScalar,
            beta: DenseScalar,
            wrap_write: W,
            wrap_read: R,
        ) -> Result<(), DenseError>
        where
            T: 'static,
            W: for<'x> Fn(DenseViewMut<'x, T>) -> DenseWrite<'x>,
            R: for<'x> Fn(DenseView<'x, T>) -> DenseRead<'x>,
        {
            #[cfg(test)]
            {
                self.logical_gemm_dispatches += 1;
            }
            let first = &run[0];
            let run_len = run.len();
            let (lhs_batch_stride, rhs_batch_stride, dst_batch_stride) = if run_len > 1 {
                let next = &run[1];
                (
                    next.lhs_offset.checked_sub(first.lhs_offset).ok_or(
                        DenseError::OffsetOverflow {
                            value: first.lhs_offset,
                        },
                    )?,
                    next.rhs_offset.checked_sub(first.rhs_offset).ok_or(
                        DenseError::OffsetOverflow {
                            value: first.rhs_offset,
                        },
                    )?,
                    next.dst_offset.checked_sub(first.dst_offset).ok_or(
                        DenseError::OffsetOverflow {
                            value: first.dst_offset,
                        },
                    )?,
                )
            } else {
                (
                    matrix_len(first.rows, first.contracted)?,
                    matrix_len(first.contracted, first.cols)?,
                    matrix_len(first.rows, first.cols)?,
                )
            };
            let lhs_shape = [first.rows, first.contracted, run_len];
            let lhs_strides = [1, first.rows, lhs_batch_stride];
            let rhs_shape = [first.contracted, first.cols, run_len];
            let rhs_strides = [1, first.contracted, rhs_batch_stride];
            let dst_shape = [first.rows, first.cols, run_len];
            let dst_strides = [1, first.rows, dst_batch_stride];
            let lhs_view = DenseView::new(
                lhs.data(),
                &lhs_shape,
                &lhs_strides,
                batch_offset(lhs.offset(), first.lhs_offset)?,
            )?;
            let rhs_view = DenseView::new(
                rhs.data(),
                &rhs_shape,
                &rhs_strides,
                batch_offset(rhs.offset(), first.rhs_offset)?,
            )?;
            let dst_offset = batch_offset(output.offset(), first.dst_offset)?;
            let dst_view =
                DenseViewMut::new(output.data_mut(), &dst_shape, &dst_strides, dst_offset)?;
            let lhs = TensorRead::from_view(tenferro_view(wrap_read(lhs_view))?);
            let rhs = TensorRead::from_view(tenferro_view(wrap_read(rhs_view))?);
            let output = TensorWrite::from_view(tenferro_view_mut(wrap_write(dst_view))?);
            let accumulation = tenferro_tensor::DotGeneralAccumulation {
                lhs_conj: false,
                rhs_conj: false,
                alpha: tenferro_scalar(alpha),
                beta: tenferro_scalar(beta),
            };
            BackendCachedDot::dot_general_read_into_accum_cached(
                &mut self.backend,
                &mut self.grouped_cache,
                Some(cache_slot),
                lhs,
                rhs,
                &self.strided_batch_matmul_config,
                accumulation,
                output,
            )
            .map_err(|err| tenferro_error("strided_batch_gemm", err))
        }
    }

    impl Default for DefaultDenseExecutor {
        fn default() -> Self {
            Self::new()
        }
    }

    impl DenseExecutor for DefaultDenseExecutor {
        fn svd(&mut self, input: DenseRead<'_>) -> Result<Vec<DenseTensor>, DenseError> {
            let input = tenferro_view(input)?;
            self.backend
                .svd_read(input)
                .map(wrap_outputs)
                .map_err(|err| tenferro_error("svd_read", err))
        }

        fn qr(&mut self, input: DenseRead<'_>) -> Result<Vec<DenseTensor>, DenseError> {
            let input = tenferro_view(input)?;
            self.backend
                .qr_read(input)
                .map(wrap_outputs)
                .map_err(|err| tenferro_error("qr_read", err))
        }

        fn eig(&mut self, input: DenseRead<'_>) -> Result<Vec<DenseTensor>, DenseError> {
            let input = tenferro_view(input)?;
            self.backend
                .eig_read(input)
                .map(wrap_outputs)
                .map_err(|err| tenferro_error("eig_read", err))
        }

        fn eigh(&mut self, input: DenseRead<'_>) -> Result<Vec<DenseTensor>, DenseError> {
            let input = tenferro_view(input)?;
            self.backend
                .eigh_read(input)
                .map(wrap_outputs)
                .map_err(|err| tenferro_error("eigh_read", err))
        }

        fn dot_general_into(
            &mut self,
            output: DenseWrite<'_>,
            lhs: DenseRead<'_>,
            rhs: DenseRead<'_>,
            config: &DenseDotConfig,
        ) -> Result<(), DenseError> {
            let lhs = TensorRead::from_view(tenferro_view(lhs)?);
            let rhs = TensorRead::from_view(tenferro_view(rhs)?);
            let output = TensorWrite::from_view(tenferro_view_mut(output)?);
            let dot_config = tenferro_dot_config(config);
            // Non-conjugating path stays byte-identical to the plain read_into
            // (which itself just wraps an overwrite accumulation). Conjugation
            // is folded into the kernel via the accumulation's conj flags — no
            // conjugated operand copy — instead of falling back to a scalar loop.
            if config.lhs_conj() || config.rhs_conj() {
                let mut accumulation =
                    tenferro_tensor::DotGeneralAccumulation::overwrite(lhs.dtype())
                        .map_err(|err| tenferro_error("dot_general_accum", err))?;
                accumulation.lhs_conj = config.lhs_conj();
                accumulation.rhs_conj = config.rhs_conj();
                self.backend
                    .dot_general_read_into_accum(lhs, rhs, &dot_config, accumulation, output)
                    .map_err(|err| tenferro_error("dot_general_accum", err))
            } else {
                self.backend
                    .dot_general_read_into(lhs, rhs, &dot_config, output)
                    .map_err(|err| tenferro_error("dot_general_read_into", err))
            }
        }

        fn matmul_into(
            &mut self,
            output: DenseWrite<'_>,
            lhs: DenseRead<'_>,
            rhs: DenseRead<'_>,
        ) -> Result<(), DenseError> {
            // GEMM backend selection is owned by tenferro; this seam only
            // lowers views and reuses the cached rank-2 contraction config.
            let lhs = TensorRead::from_view(tenferro_view(lhs)?);
            let rhs = TensorRead::from_view(tenferro_view(rhs)?);
            let output = TensorWrite::from_view(tenferro_view_mut(output)?);
            self.backend
                .dot_general_read_into(lhs, rhs, &self.matmul_config, output)
                .map_err(|err| tenferro_error("dot_general_read_into", err))
        }

        fn matmul_axpby_into(
            &mut self,
            output: DenseWrite<'_>,
            lhs: DenseRead<'_>,
            rhs: DenseRead<'_>,
            alpha: DenseScalar,
            beta: DenseScalar,
        ) -> Result<(), DenseError> {
            // Overwrite case keeps the cached-config fast path.
            if alpha.is_one() && beta.is_zero() {
                return self.matmul_into(output, lhs, rhs);
            }
            let lhs = TensorRead::from_view(tenferro_view(lhs)?);
            let rhs = TensorRead::from_view(tenferro_view(rhs)?);
            let output = TensorWrite::from_view(tenferro_view_mut(output)?);
            let accumulation = tenferro_tensor::DotGeneralAccumulation {
                lhs_conj: false,
                rhs_conj: false,
                alpha: tenferro_scalar(alpha),
                beta: tenferro_scalar(beta),
            };
            self.backend
                .dot_general_read_into_accum(lhs, rhs, &self.matmul_config, accumulation, output)
                .map_err(|err| tenferro_error("dot_general_accum", err))
        }

        fn matmul_batch_axpby_into(
            &mut self,
            output: DenseWrite<'_>,
            lhs: DenseRead<'_>,
            rhs: DenseRead<'_>,
            jobs: &[DenseGemmBatchJob],
            alpha: DenseScalar,
            beta: DenseScalar,
        ) -> Result<(), DenseError> {
            match (output, lhs, rhs) {
                (DenseWrite::F32(mut out), DenseRead::F32(lhs), DenseRead::F32(rhs)) => {
                    if self.matmul_batch_axpby_strided_typed(
                        &mut out,
                        lhs,
                        rhs,
                        jobs,
                        alpha,
                        beta,
                        |view: DenseViewMut<'_, f32>| DenseWrite::F32(view),
                        |view: DenseView<'_, f32>| DenseRead::F32(view),
                    )? {
                        return Ok(());
                    }
                    self.matmul_batch_axpby_grouped(
                        DenseWrite::F32(out),
                        DenseRead::F32(lhs),
                        DenseRead::F32(rhs),
                        jobs,
                        alpha,
                        beta,
                    )
                }
                (DenseWrite::F64(mut out), DenseRead::F64(lhs), DenseRead::F64(rhs)) => {
                    if self.matmul_batch_axpby_strided_typed(
                        &mut out,
                        lhs,
                        rhs,
                        jobs,
                        alpha,
                        beta,
                        |view: DenseViewMut<'_, f64>| DenseWrite::F64(view),
                        |view: DenseView<'_, f64>| DenseRead::F64(view),
                    )? {
                        return Ok(());
                    }
                    self.matmul_batch_axpby_grouped(
                        DenseWrite::F64(out),
                        DenseRead::F64(lhs),
                        DenseRead::F64(rhs),
                        jobs,
                        alpha,
                        beta,
                    )
                }
                (DenseWrite::C32(mut out), DenseRead::C32(lhs), DenseRead::C32(rhs)) => {
                    if self.matmul_batch_axpby_strided_typed(
                        &mut out,
                        lhs,
                        rhs,
                        jobs,
                        alpha,
                        beta,
                        |view: DenseViewMut<'_, Complex32>| DenseWrite::C32(view),
                        |view: DenseView<'_, Complex32>| DenseRead::C32(view),
                    )? {
                        return Ok(());
                    }
                    self.matmul_batch_axpby_grouped(
                        DenseWrite::C32(out),
                        DenseRead::C32(lhs),
                        DenseRead::C32(rhs),
                        jobs,
                        alpha,
                        beta,
                    )
                }
                (DenseWrite::C64(mut out), DenseRead::C64(lhs), DenseRead::C64(rhs)) => {
                    if self.matmul_batch_axpby_strided_typed(
                        &mut out,
                        lhs,
                        rhs,
                        jobs,
                        alpha,
                        beta,
                        |view: DenseViewMut<'_, Complex64>| DenseWrite::C64(view),
                        |view: DenseView<'_, Complex64>| DenseRead::C64(view),
                    )? {
                        return Ok(());
                    }
                    self.matmul_batch_axpby_grouped(
                        DenseWrite::C64(out),
                        DenseRead::C64(lhs),
                        DenseRead::C64(rhs),
                        jobs,
                        alpha,
                        beta,
                    )
                }
                _ => Err(DenseError::Backend {
                    backend: DenseBackend::Tenferro,
                    op: "matmul_batch_axpby_into",
                    message: "batched matmul requires matching f32/f64/c32/c64 operands"
                        .to_string(),
                }),
            }
        }
    }

    fn tenferro_scalar(value: DenseScalar) -> tenferro_tensor::ContractionScalar {
        match value {
            DenseScalar::F32(value) => tenferro_tensor::ContractionScalar::F32(value),
            DenseScalar::F64(value) => tenferro_tensor::ContractionScalar::F64(value),
            DenseScalar::C32(value) => tenferro_tensor::ContractionScalar::C32(value),
            DenseScalar::C64(value) => tenferro_tensor::ContractionScalar::C64(value),
        }
    }

    fn wrap_outputs(outputs: Vec<Tensor>) -> Vec<DenseTensor> {
        outputs
            .into_iter()
            .map(DenseTensor::from_tenferro)
            .collect()
    }

    fn tenferro_view(input: DenseRead<'_>) -> Result<TensorView<'_>, DenseError> {
        match input {
            DenseRead::F32(view) => typed_tenferro_view(view).map(TensorView::F32),
            DenseRead::F64(view) => typed_tenferro_view(view).map(TensorView::F64),
            DenseRead::I32(view) => typed_tenferro_view(view).map(TensorView::I32),
            DenseRead::I64(view) => typed_tenferro_view(view).map(TensorView::I64),
            DenseRead::Bool(view) => typed_tenferro_view(view).map(TensorView::Bool),
            DenseRead::C32(view) => typed_tenferro_view(view).map(TensorView::C32),
            DenseRead::C64(view) => typed_tenferro_view(view).map(TensorView::C64),
        }
    }

    fn tenferro_view_mut(output: DenseWrite<'_>) -> Result<TensorViewMut<'_>, DenseError> {
        match output {
            DenseWrite::F32(view) => typed_tenferro_view_mut(view).map(TensorViewMut::F32),
            DenseWrite::F64(view) => typed_tenferro_view_mut(view).map(TensorViewMut::F64),
            DenseWrite::I32(view) => typed_tenferro_view_mut(view).map(TensorViewMut::I32),
            DenseWrite::I64(view) => typed_tenferro_view_mut(view).map(TensorViewMut::I64),
            DenseWrite::Bool(view) => typed_tenferro_view_mut(view).map(TensorViewMut::Bool),
            DenseWrite::C32(view) => typed_tenferro_view_mut(view).map(TensorViewMut::C32),
            DenseWrite::C64(view) => typed_tenferro_view_mut(view).map(TensorViewMut::C64),
        }
    }

    fn typed_tenferro_view<'a, T: 'static>(
        view: DenseView<'a, T>,
    ) -> Result<TypedTensorView<'a, T>, DenseError> {
        let strides = strides_to_isize(view.strides())?;
        let offset = isize::try_from(view.offset()).map_err(|_| DenseError::OffsetOverflow {
            value: view.offset(),
        })?;
        TypedTensorView::from_slice(view.shape(), strides, offset, view.data())
            .map_err(|err| tenferro_error("TypedTensorView::from_slice", err))
    }

    fn typed_tenferro_view_mut<'a, T: 'static>(
        view: DenseViewMut<'a, T>,
    ) -> Result<TypedTensorViewMut<'a, T>, DenseError> {
        let DenseViewMut {
            data,
            shape,
            strides,
            offset,
        } = view;
        let strides = strides_to_isize(strides)?;
        let offset =
            isize::try_from(offset).map_err(|_| DenseError::OffsetOverflow { value: offset })?;
        TypedTensorViewMut::from_slice(shape, strides, offset, data)
            .map_err(|err| tenferro_error("TypedTensorViewMut::from_slice", err))
    }

    fn tenferro_dot_config(config: &DenseDotConfig) -> DotGeneralConfig {
        DotGeneralConfig {
            lhs_contracting_dims: config.lhs_contracting_dims().to_vec(),
            rhs_contracting_dims: config.rhs_contracting_dims().to_vec(),
            lhs_batch_dims: config.lhs_batch_dims().to_vec(),
            rhs_batch_dims: config.rhs_batch_dims().to_vec(),
        }
    }
}

#[cfg(feature = "tenferro")]
fn strides_to_isize(strides: &[usize]) -> Result<Vec<isize>, DenseError> {
    strides
        .iter()
        .map(|&stride| {
            isize::try_from(stride).map_err(|_| DenseError::StrideOverflow { value: stride })
        })
        .collect()
}

#[cfg(feature = "tenferro")]
fn dense_dtype_from_tenferro(dtype: tenferro_tensor::DType) -> DenseDType {
    match dtype {
        tenferro_tensor::DType::F32 => DenseDType::F32,
        tenferro_tensor::DType::F64 => DenseDType::F64,
        tenferro_tensor::DType::I32 => DenseDType::I32,
        tenferro_tensor::DType::I64 => DenseDType::I64,
        tenferro_tensor::DType::Bool => DenseDType::Bool,
        tenferro_tensor::DType::C32 => DenseDType::C32,
        tenferro_tensor::DType::C64 => DenseDType::C64,
    }
}

#[cfg(feature = "tenferro")]
fn tenferro_error(op: &'static str, err: tenferro_tensor::Error) -> DenseError {
    DenseError::Backend {
        backend: DenseBackend::Tenferro,
        op,
        message: err.to_string(),
    }
}

#[cfg(test)]
mod tests {
    #![allow(dead_code)]

    use super::*;

    fn assert_f64_close(actual: f64, expected: f64, tol: f64) {
        assert!(
            (actual - expected).abs() <= tol,
            "expected {expected}, got {actual}, tol={tol}"
        );
    }

    fn assert_f32_close(actual: f32, expected: f32, tol: f32) {
        assert!(
            (actual - expected).abs() <= tol,
            "expected {expected}, got {actual}, tol={tol}"
        );
    }

    fn assert_c32_close(actual: Complex32, expected: Complex32, tol: f32) {
        assert_f32_close(actual.re, expected.re, tol);
        assert_f32_close(actual.im, expected.im, tol);
    }

    fn assert_c64_close(actual: Complex64, expected: Complex64, tol: f64) {
        assert_f64_close(actual.re, expected.re, tol);
        assert_f64_close(actual.im, expected.im, tol);
    }

    // Regression guard for the conjugated-contraction fast path: a conj flag on
    // `dot_general_into` must fold conjugation into the kernel and produce
    // exactly what contracting an elementwise-conjugated operand would — and it
    // must actually change the result (so the flag can't be silently dropped
    // back to a no-op or a bypassed scalar loop).
    #[test]
    fn dot_general_conjugation_flag_matches_materialized_conjugate() {
        let c = |re: f64, im: f64| Complex64::new(re, im);
        let shape = [2usize, 2];
        let strides = [1usize, 2]; // column-major
        let lhs = vec![c(1.0, 1.0), c(3.0, 2.0), c(2.0, -1.0), c(4.0, -3.0)];
        let rhs = vec![c(5.0, -2.0), c(7.0, -4.0), c(6.0, 1.0), c(8.0, 2.0)];

        let run = |lhs_data: &[Complex64], lhs_conj: bool, rhs_conj: bool| -> Vec<Complex64> {
            let mut out = vec![c(0.0, 0.0); 4];
            let mut executor = DefaultDenseExecutor::new();
            executor
                .dot_general_into(
                    DenseWrite::C64(DenseViewMut::new(&mut out, &shape, &strides, 0).unwrap()),
                    DenseRead::C64(DenseView::new(lhs_data, &shape, &strides, 0).unwrap()),
                    DenseRead::C64(DenseView::new(&rhs, &shape, &strides, 0).unwrap()),
                    &DenseDotConfig::matmul().with_conjugation(lhs_conj, rhs_conj),
                )
                .unwrap();
            out
        };

        let via_flag = run(&lhs, true, false);
        let lhs_conjugated: Vec<Complex64> = lhs.iter().map(|z| z.conj()).collect();
        let via_materialized = run(&lhs_conjugated, false, false);
        for (actual, expected) in via_flag.iter().zip(&via_materialized) {
            assert_c64_close(*actual, *expected, 1.0e-12);
        }

        let plain = run(&lhs, false, false);
        assert!(
            via_flag
                .iter()
                .zip(&plain)
                .any(|(a, b)| (a - b).norm() > 1.0e-9),
            "conjugation flag had no effect on the result"
        );
    }

    fn col_major_index(rows: usize, row: usize, col: usize) -> usize {
        row + col * rows
    }

    fn transpose_f32(mat: &[f32], rows: usize, cols: usize) -> Vec<f32> {
        let mut out = vec![0.0; rows * cols];
        for j in 0..cols {
            for i in 0..rows {
                out[col_major_index(cols, j, i)] = mat[col_major_index(rows, i, j)];
            }
        }
        out
    }

    fn transpose_f64(mat: &[f64], rows: usize, cols: usize) -> Vec<f64> {
        let mut out = vec![0.0; rows * cols];
        for j in 0..cols {
            for i in 0..rows {
                out[col_major_index(cols, j, i)] = mat[col_major_index(rows, i, j)];
            }
        }
        out
    }

    fn transpose_c32(mat: &[Complex32], rows: usize, cols: usize) -> Vec<Complex32> {
        let mut out = vec![Complex32::new(0.0, 0.0); rows * cols];
        for j in 0..cols {
            for i in 0..rows {
                out[col_major_index(cols, j, i)] = mat[col_major_index(rows, i, j)];
            }
        }
        out
    }

    fn transpose_c64(mat: &[Complex64], rows: usize, cols: usize) -> Vec<Complex64> {
        let mut out = vec![Complex64::new(0.0, 0.0); rows * cols];
        for j in 0..cols {
            for i in 0..rows {
                out[col_major_index(cols, j, i)] = mat[col_major_index(rows, i, j)];
            }
        }
        out
    }

    fn conjugate_transpose_c32(mat: &[Complex32], rows: usize, cols: usize) -> Vec<Complex32> {
        let mut out = vec![Complex32::new(0.0, 0.0); rows * cols];
        for j in 0..cols {
            for i in 0..rows {
                out[col_major_index(cols, j, i)] = mat[col_major_index(rows, i, j)].conj();
            }
        }
        out
    }

    fn conjugate_transpose_c64(mat: &[Complex64], rows: usize, cols: usize) -> Vec<Complex64> {
        let mut out = vec![Complex64::new(0.0, 0.0); rows * cols];
        for j in 0..cols {
            for i in 0..rows {
                out[col_major_index(cols, j, i)] = mat[col_major_index(rows, i, j)].conj();
            }
        }
        out
    }

    fn matmul_f32(lhs: &[f32], rhs: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
        let mut out = vec![0.0; m * n];
        for j in 0..n {
            for p in 0..k {
                let rhs_pj = rhs[col_major_index(k, p, j)];
                for i in 0..m {
                    out[col_major_index(m, i, j)] += lhs[col_major_index(m, i, p)] * rhs_pj;
                }
            }
        }
        out
    }

    fn matmul_f64(lhs: &[f64], rhs: &[f64], m: usize, k: usize, n: usize) -> Vec<f64> {
        let mut out = vec![0.0; m * n];
        for j in 0..n {
            for p in 0..k {
                let rhs_pj = rhs[col_major_index(k, p, j)];
                for i in 0..m {
                    out[col_major_index(m, i, j)] += lhs[col_major_index(m, i, p)] * rhs_pj;
                }
            }
        }
        out
    }

    fn matmul_c32(
        lhs: &[Complex32],
        rhs: &[Complex32],
        m: usize,
        k: usize,
        n: usize,
    ) -> Vec<Complex32> {
        let mut out = vec![Complex32::new(0.0, 0.0); m * n];
        for j in 0..n {
            for p in 0..k {
                let rhs_pj = rhs[col_major_index(k, p, j)];
                for i in 0..m {
                    out[col_major_index(m, i, j)] += lhs[col_major_index(m, i, p)] * rhs_pj;
                }
            }
        }
        out
    }

    fn matmul_c64(
        lhs: &[Complex64],
        rhs: &[Complex64],
        m: usize,
        k: usize,
        n: usize,
    ) -> Vec<Complex64> {
        let mut out = vec![Complex64::new(0.0, 0.0); m * n];
        for j in 0..n {
            for p in 0..k {
                let rhs_pj = rhs[col_major_index(k, p, j)];
                for i in 0..m {
                    out[col_major_index(m, i, j)] += lhs[col_major_index(m, i, p)] * rhs_pj;
                }
            }
        }
        out
    }

    fn diag_f32(values: &[f32]) -> Vec<f32> {
        let mut out = vec![0.0; values.len() * values.len()];
        for (i, value) in values.iter().enumerate() {
            out[col_major_index(values.len(), i, i)] = *value;
        }
        out
    }

    fn diag_f64(values: &[f64]) -> Vec<f64> {
        let mut out = vec![0.0; values.len() * values.len()];
        for (i, value) in values.iter().enumerate() {
            out[col_major_index(values.len(), i, i)] = *value;
        }
        out
    }

    fn diag_c32_from_real(values: &[f32]) -> Vec<Complex32> {
        let mut out = vec![Complex32::new(0.0, 0.0); values.len() * values.len()];
        for (i, value) in values.iter().enumerate() {
            out[col_major_index(values.len(), i, i)] = Complex32::new(*value, 0.0);
        }
        out
    }

    fn diag_c64_from_real(values: &[f64]) -> Vec<Complex64> {
        let mut out = vec![Complex64::new(0.0, 0.0); values.len() * values.len()];
        for (i, value) in values.iter().enumerate() {
            out[col_major_index(values.len(), i, i)] = Complex64::new(*value, 0.0);
        }
        out
    }

    #[test]
    fn dense_view_rejects_out_of_bounds_layout() {
        let data = [0.0; 6];
        let shape = [2, 3];
        let strides = [1, 4];
        let err = DenseView::new(&data, &shape, &strides, 0).unwrap_err();
        assert_eq!(err, DenseError::OutOfBounds);
    }

    #[cfg(feature = "tenferro")]
    #[test]
    fn default_executor_matmul_into_matches_tensorkit_recoupling_view_for_all_gemm_dtypes() {
        let lhs_shape = [2, 3];
        let lhs_strides = [1, 2];
        let rhs_shape = [3, 2];
        let rhs_strides = [1, 3];
        let out_shape = [2, 2];
        let out_strides = [1, 4];
        let out_offset = 1;

        let mut executor = DefaultDenseExecutor::new();

        let lhs_f32 = vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let u_f32 = vec![10.0_f32, 100.0, 1000.0, 20.0, 200.0, 2000.0];
        let mut out_f32 = vec![-1.0_f32; 8];
        executor
            .matmul_into(
                DenseWrite::F32(
                    DenseViewMut::new(&mut out_f32, &out_shape, &out_strides, out_offset).unwrap(),
                ),
                DenseRead::F32(DenseView::new(&lhs_f32, &lhs_shape, &lhs_strides, 0).unwrap()),
                DenseRead::F32(DenseView::new(&u_f32, &rhs_shape, &rhs_strides, 0).unwrap()),
            )
            .unwrap();
        assert_eq!(
            out_f32,
            vec![-1.0, 5310.0, 6420.0, -1.0, -1.0, 10620.0, 12840.0, -1.0]
        );

        let lhs_f64 = vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0];
        let u_f64 = vec![10.0_f64, 100.0, 1000.0, 20.0, 200.0, 2000.0];
        let mut out_f64 = vec![-1.0_f64; 8];
        executor
            .matmul_into(
                DenseWrite::F64(
                    DenseViewMut::new(&mut out_f64, &out_shape, &out_strides, out_offset).unwrap(),
                ),
                DenseRead::F64(DenseView::new(&lhs_f64, &lhs_shape, &lhs_strides, 0).unwrap()),
                DenseRead::F64(DenseView::new(&u_f64, &rhs_shape, &rhs_strides, 0).unwrap()),
            )
            .unwrap();
        assert_eq!(
            out_f64,
            vec![-1.0, 5310.0, 6420.0, -1.0, -1.0, 10620.0, 12840.0, -1.0]
        );

        let lhs_c32 = lhs_f32
            .iter()
            .map(|&value| Complex32::new(value, 0.0))
            .collect::<Vec<_>>();
        let u_c32 = u_f32
            .iter()
            .map(|&value| Complex32::new(value, 0.0))
            .collect::<Vec<_>>();
        let mut out_c32 = vec![Complex32::new(-1.0, -2.0); 8];
        executor
            .matmul_into(
                DenseWrite::C32(
                    DenseViewMut::new(&mut out_c32, &out_shape, &out_strides, out_offset).unwrap(),
                ),
                DenseRead::C32(DenseView::new(&lhs_c32, &lhs_shape, &lhs_strides, 0).unwrap()),
                DenseRead::C32(DenseView::new(&u_c32, &rhs_shape, &rhs_strides, 0).unwrap()),
            )
            .unwrap();
        assert_eq!(
            out_c32,
            vec![
                Complex32::new(-1.0, -2.0),
                Complex32::new(5310.0, 0.0),
                Complex32::new(6420.0, 0.0),
                Complex32::new(-1.0, -2.0),
                Complex32::new(-1.0, -2.0),
                Complex32::new(10620.0, 0.0),
                Complex32::new(12840.0, 0.0),
                Complex32::new(-1.0, -2.0),
            ]
        );

        let lhs_c64 = lhs_f64
            .iter()
            .map(|&value| Complex64::new(value, 0.0))
            .collect::<Vec<_>>();
        let u_c64 = u_f64
            .iter()
            .map(|&value| Complex64::new(value, 0.0))
            .collect::<Vec<_>>();
        let mut out_c64 = vec![Complex64::new(-1.0, -2.0); 8];
        executor
            .matmul_into(
                DenseWrite::C64(
                    DenseViewMut::new(&mut out_c64, &out_shape, &out_strides, out_offset).unwrap(),
                ),
                DenseRead::C64(DenseView::new(&lhs_c64, &lhs_shape, &lhs_strides, 0).unwrap()),
                DenseRead::C64(DenseView::new(&u_c64, &rhs_shape, &rhs_strides, 0).unwrap()),
            )
            .unwrap();
        assert_eq!(
            out_c64,
            vec![
                Complex64::new(-1.0, -2.0),
                Complex64::new(5310.0, 0.0),
                Complex64::new(6420.0, 0.0),
                Complex64::new(-1.0, -2.0),
                Complex64::new(-1.0, -2.0),
                Complex64::new(10620.0, 0.0),
                Complex64::new(12840.0, 0.0),
                Complex64::new(-1.0, -2.0),
            ]
        );
    }

    #[cfg(feature = "tenferro")]
    #[test]
    fn default_executor_fuses_same_shape_strided_batch_jobs_for_all_gemm_dtypes() {
        let mut lhs = Vec::new();
        let mut rhs = Vec::new();
        for block in 0..2 {
            let base = block as f64;
            lhs.extend_from_slice(&[1.0 + base, 2.0 + base, 3.0 + base, 4.0 + base]);
            rhs.extend_from_slice(&[5.0 + base, 6.0 + base, 7.0 + base, 8.0 + base]);
        }
        let mut output = vec![-99.0; 2 * 4];
        let jobs = [0usize, 1]
            .into_iter()
            .map(|block| DenseGemmBatchJob {
                dst_offset: block * 4,
                lhs_offset: block * 4,
                rhs_offset: block * 4,
                rows: 2,
                contracted: 2,
                cols: 2,
            })
            .collect::<Vec<_>>();
        let flat_shape = [2 * 4];
        let flat_strides = [1usize];

        let mut executor = DefaultDenseExecutor::new();
        executor.reset_logical_gemm_dispatches();
        executor
            .matmul_batch_axpby_into(
                DenseWrite::F64(
                    DenseViewMut::new(&mut output, &flat_shape, &flat_strides, 0).unwrap(),
                ),
                DenseRead::F64(DenseView::new(&lhs, &flat_shape, &flat_strides, 0).unwrap()),
                DenseRead::F64(DenseView::new(&rhs, &flat_shape, &flat_strides, 0).unwrap()),
                &jobs,
                DenseScalar::F64(1.0),
                DenseScalar::F64(0.0),
            )
            .unwrap();

        assert_eq!(
            executor.logical_gemm_dispatches(),
            1,
            "same-shape strided batch submitted {} logical GEMM dispatches for {} jobs",
            executor.logical_gemm_dispatches(),
            jobs.len()
        );
        assert!(
            executor.logical_gemm_dispatches() < jobs.len(),
            "batched GEMM logical dispatch count must not scale with same-shape job count"
        );
        for block in 0..2 {
            let start = block * 4;
            let expected = matmul_f64(&lhs[start..start + 4], &rhs[start..start + 4], 2, 2, 2);
            for (actual, expected) in output[start..start + 4].iter().zip(expected) {
                assert_f64_close(*actual, expected, 1.0e-12);
            }
        }

        let lhs_f32 = lhs.iter().map(|&value| value as f32).collect::<Vec<_>>();
        let rhs_f32 = rhs.iter().map(|&value| value as f32).collect::<Vec<_>>();
        let mut output_f32 = vec![-99.0_f32; 2 * 4];
        let mut executor = DefaultDenseExecutor::new();
        executor
            .matmul_batch_axpby_into(
                DenseWrite::F32(
                    DenseViewMut::new(&mut output_f32, &flat_shape, &flat_strides, 0).unwrap(),
                ),
                DenseRead::F32(DenseView::new(&lhs_f32, &flat_shape, &flat_strides, 0).unwrap()),
                DenseRead::F32(DenseView::new(&rhs_f32, &flat_shape, &flat_strides, 0).unwrap()),
                &jobs,
                DenseScalar::F32(1.0),
                DenseScalar::F32(0.0),
            )
            .unwrap();
        assert_eq!(executor.logical_gemm_dispatches(), 1);
        for block in 0..2 {
            let start = block * 4;
            let expected = matmul_f32(
                &lhs_f32[start..start + 4],
                &rhs_f32[start..start + 4],
                2,
                2,
                2,
            );
            for (actual, expected) in output_f32[start..start + 4].iter().zip(expected) {
                assert_f32_close(*actual, expected, 1.0e-4);
            }
        }

        let lhs_c32 = lhs_f32
            .iter()
            .map(|&value| Complex32::new(value, 0.25 * value))
            .collect::<Vec<_>>();
        let rhs_c32 = rhs_f32
            .iter()
            .map(|&value| Complex32::new(value, -0.125 * value))
            .collect::<Vec<_>>();
        let mut output_c32 = vec![Complex32::new(-99.0, -99.0); 2 * 4];
        let mut executor = DefaultDenseExecutor::new();
        executor
            .matmul_batch_axpby_into(
                DenseWrite::C32(
                    DenseViewMut::new(&mut output_c32, &flat_shape, &flat_strides, 0).unwrap(),
                ),
                DenseRead::C32(DenseView::new(&lhs_c32, &flat_shape, &flat_strides, 0).unwrap()),
                DenseRead::C32(DenseView::new(&rhs_c32, &flat_shape, &flat_strides, 0).unwrap()),
                &jobs,
                DenseScalar::C32(Complex32::new(1.0, 0.0)),
                DenseScalar::C32(Complex32::new(0.0, 0.0)),
            )
            .unwrap();
        assert_eq!(executor.logical_gemm_dispatches(), 1);
        for block in 0..2 {
            let start = block * 4;
            let expected = matmul_c32(
                &lhs_c32[start..start + 4],
                &rhs_c32[start..start + 4],
                2,
                2,
                2,
            );
            for (actual, expected) in output_c32[start..start + 4].iter().zip(expected) {
                assert_c32_close(*actual, expected, 1.0e-3);
            }
        }

        let lhs_c64 = lhs
            .iter()
            .map(|&value| Complex64::new(value, 0.25 * value))
            .collect::<Vec<_>>();
        let rhs_c64 = rhs
            .iter()
            .map(|&value| Complex64::new(value, -0.125 * value))
            .collect::<Vec<_>>();
        let mut output_c64 = vec![Complex64::new(-99.0, -99.0); 2 * 4];
        let mut executor = DefaultDenseExecutor::new();
        executor
            .matmul_batch_axpby_into(
                DenseWrite::C64(
                    DenseViewMut::new(&mut output_c64, &flat_shape, &flat_strides, 0).unwrap(),
                ),
                DenseRead::C64(DenseView::new(&lhs_c64, &flat_shape, &flat_strides, 0).unwrap()),
                DenseRead::C64(DenseView::new(&rhs_c64, &flat_shape, &flat_strides, 0).unwrap()),
                &jobs,
                DenseScalar::C64(Complex64::new(1.0, 0.0)),
                DenseScalar::C64(Complex64::new(0.0, 0.0)),
            )
            .unwrap();
        assert_eq!(executor.logical_gemm_dispatches(), 1);
        for block in 0..2 {
            let start = block * 4;
            let expected = matmul_c64(
                &lhs_c64[start..start + 4],
                &rhs_c64[start..start + 4],
                2,
                2,
                2,
            );
            for (actual, expected) in output_c64[start..start + 4].iter().zip(expected) {
                assert_c64_close(*actual, expected, 1.0e-12);
            }
        }
    }

    #[cfg(feature = "tenferro")]
    #[test]
    fn default_executor_qr_reads_transposed_views_for_all_linalg_dtypes() {
        let f32_data = vec![1.0_f32, -2.0, 3.0, 0.5, -1.0, 4.0];
        let f64_data = vec![1.0_f64, -2.0, 3.0, 0.5, -1.0, 4.0];
        let c32_data = vec![
            Complex32::new(1.0, 0.5),
            Complex32::new(-2.0, 1.0),
            Complex32::new(3.0, -0.25),
            Complex32::new(0.5, -1.0),
            Complex32::new(-1.0, 0.75),
            Complex32::new(4.0, 1.5),
        ];
        let c64_data = vec![
            Complex64::new(1.0, 0.5),
            Complex64::new(-2.0, 1.0),
            Complex64::new(3.0, -0.25),
            Complex64::new(0.5, -1.0),
            Complex64::new(-1.0, 0.75),
            Complex64::new(4.0, 1.5),
        ];
        let shape = [3, 2];
        let strides = [2, 1];
        let mut executor = DefaultDenseExecutor::new();

        let outputs = executor
            .qr(DenseRead::F32(
                DenseView::new(&f32_data, &shape, &strides, 0).unwrap(),
            ))
            .unwrap();
        assert_eq!(outputs[0].dtype(), DenseDType::F32);
        let recon = matmul_f32(
            outputs[0].as_f32_slice().unwrap(),
            outputs[1].as_f32_slice().unwrap(),
            3,
            2,
            2,
        );
        let expected = transpose_f32(&f32_data, 2, 3);
        for (actual, expected) in recon.iter().zip(expected.iter()) {
            assert_f32_close(*actual, *expected, 1.0e-5);
        }

        let outputs = executor
            .qr(DenseRead::F64(
                DenseView::new(&f64_data, &shape, &strides, 0).unwrap(),
            ))
            .unwrap();
        assert_eq!(outputs[0].dtype(), DenseDType::F64);
        let recon = matmul_f64(
            outputs[0].as_f64_slice().unwrap(),
            outputs[1].as_f64_slice().unwrap(),
            3,
            2,
            2,
        );
        let expected = transpose_f64(&f64_data, 2, 3);
        for (actual, expected) in recon.iter().zip(expected.iter()) {
            assert_f64_close(*actual, *expected, 1.0e-9);
        }

        let outputs = executor
            .qr(DenseRead::C32(
                DenseView::new(&c32_data, &shape, &strides, 0).unwrap(),
            ))
            .unwrap();
        assert_eq!(outputs[0].dtype(), DenseDType::C32);
        let recon = matmul_c32(
            outputs[0].as_c32_slice().unwrap(),
            outputs[1].as_c32_slice().unwrap(),
            3,
            2,
            2,
        );
        let expected = transpose_c32(&c32_data, 2, 3);
        for (actual, expected) in recon.iter().zip(expected.iter()) {
            assert_c32_close(*actual, *expected, 1.0e-5);
        }

        let outputs = executor
            .qr(DenseRead::C64(
                DenseView::new(&c64_data, &shape, &strides, 0).unwrap(),
            ))
            .unwrap();
        assert_eq!(outputs[0].dtype(), DenseDType::C64);
        let recon = matmul_c64(
            outputs[0].as_c64_slice().unwrap(),
            outputs[1].as_c64_slice().unwrap(),
            3,
            2,
            2,
        );
        let expected = transpose_c64(&c64_data, 2, 3);
        for (actual, expected) in recon.iter().zip(expected.iter()) {
            assert_c64_close(*actual, *expected, 1.0e-9);
        }
    }

    #[cfg(feature = "tenferro")]
    #[test]
    fn default_executor_eigh_reads_transposed_views_for_all_linalg_dtypes() {
        let f32_data = vec![4.0_f32, 1.0, 1.0, 3.0];
        let f64_data = vec![4.0_f64, 1.0, 1.0, 3.0];
        let c32_data = vec![
            Complex32::new(4.0, 0.0),
            Complex32::new(1.0, -0.5),
            Complex32::new(1.0, 0.5),
            Complex32::new(3.0, 0.0),
        ];
        let c64_data = vec![
            Complex64::new(4.0, 0.0),
            Complex64::new(1.0, -0.5),
            Complex64::new(1.0, 0.5),
            Complex64::new(3.0, 0.0),
        ];
        let shape = [2, 2];
        let strides = [2, 1];
        let mut executor = DefaultDenseExecutor::new();

        let outputs = executor
            .eigh(DenseRead::F32(
                DenseView::new(&f32_data, &shape, &strides, 0).unwrap(),
            ))
            .unwrap();
        assert_eq!(outputs[0].dtype(), DenseDType::F32);
        assert_eq!(outputs[1].dtype(), DenseDType::F32);
        let values = outputs[0].as_f32_slice().unwrap();
        let vectors = outputs[1].as_f32_slice().unwrap();
        let recon = matmul_f32(
            &matmul_f32(vectors, &diag_f32(values), 2, 2, 2),
            &transpose_f32(vectors, 2, 2),
            2,
            2,
            2,
        );
        let expected = transpose_f32(&f32_data, 2, 2);
        for (actual, expected) in recon.iter().zip(expected.iter()) {
            assert_f32_close(*actual, *expected, 1.0e-5);
        }

        let outputs = executor
            .eigh(DenseRead::F64(
                DenseView::new(&f64_data, &shape, &strides, 0).unwrap(),
            ))
            .unwrap();
        assert_eq!(outputs[0].dtype(), DenseDType::F64);
        assert_eq!(outputs[1].dtype(), DenseDType::F64);
        let values = outputs[0].as_f64_slice().unwrap();
        let vectors = outputs[1].as_f64_slice().unwrap();
        let recon = matmul_f64(
            &matmul_f64(vectors, &diag_f64(values), 2, 2, 2),
            &transpose_f64(vectors, 2, 2),
            2,
            2,
            2,
        );
        let expected = transpose_f64(&f64_data, 2, 2);
        for (actual, expected) in recon.iter().zip(expected.iter()) {
            assert_f64_close(*actual, *expected, 1.0e-10);
        }

        let outputs = executor
            .eigh(DenseRead::C32(
                DenseView::new(&c32_data, &shape, &strides, 0).unwrap(),
            ))
            .unwrap();
        assert_eq!(outputs[0].dtype(), DenseDType::F32);
        assert_eq!(outputs[1].dtype(), DenseDType::C32);
        let values = outputs[0].as_f32_slice().unwrap();
        let vectors = outputs[1].as_c32_slice().unwrap();
        let recon = matmul_c32(
            &matmul_c32(vectors, &diag_c32_from_real(values), 2, 2, 2),
            &conjugate_transpose_c32(vectors, 2, 2),
            2,
            2,
            2,
        );
        let expected = transpose_c32(&c32_data, 2, 2);
        for (actual, expected) in recon.iter().zip(expected.iter()) {
            assert_c32_close(*actual, *expected, 1.0e-5);
        }

        let outputs = executor
            .eigh(DenseRead::C64(
                DenseView::new(&c64_data, &shape, &strides, 0).unwrap(),
            ))
            .unwrap();
        assert_eq!(outputs[0].dtype(), DenseDType::F64);
        assert_eq!(outputs[1].dtype(), DenseDType::C64);
        let values = outputs[0].as_f64_slice().unwrap();
        let vectors = outputs[1].as_c64_slice().unwrap();
        let recon = matmul_c64(
            &matmul_c64(vectors, &diag_c64_from_real(values), 2, 2, 2),
            &conjugate_transpose_c64(vectors, 2, 2),
            2,
            2,
            2,
        );
        let expected = transpose_c64(&c64_data, 2, 2);
        for (actual, expected) in recon.iter().zip(expected.iter()) {
            assert_c64_close(*actual, *expected, 1.0e-10);
        }
    }

    #[cfg(feature = "tenferro")]
    #[test]
    fn default_executor_svd_accepts_all_supported_linalg_dtypes() {
        let f32_data = [1.0_f32, -2.0, 0.5, 4.0];
        let f64_data = [1.0_f64, -2.0, 0.5, 4.0];
        let c32_data = [
            Complex32::new(1.0, 0.5),
            Complex32::new(-2.0, 1.0),
            Complex32::new(0.5, -0.25),
            Complex32::new(4.0, 1.5),
        ];
        let c64_data = [
            Complex64::new(1.0, 0.5),
            Complex64::new(-2.0, 1.0),
            Complex64::new(0.5, -0.25),
            Complex64::new(4.0, 1.5),
        ];
        let shape = [2, 2];
        let strides = [2, 1];

        let mut executor = DefaultDenseExecutor::new();
        for (input, dtype) in [
            (
                DenseRead::F32(DenseView::new(&f32_data, &shape, &strides, 0).unwrap()),
                DenseDType::F32,
            ),
            (
                DenseRead::F64(DenseView::new(&f64_data, &shape, &strides, 0).unwrap()),
                DenseDType::F64,
            ),
            (
                DenseRead::C32(DenseView::new(&c32_data, &shape, &strides, 0).unwrap()),
                DenseDType::C32,
            ),
            (
                DenseRead::C64(DenseView::new(&c64_data, &shape, &strides, 0).unwrap()),
                DenseDType::C64,
            ),
        ] {
            let outputs = executor.svd(input).unwrap();
            assert_eq!(outputs[0].dtype(), dtype);
            assert!(matches!(
                (dtype, outputs[1].dtype()),
                (DenseDType::F32, DenseDType::F32)
                    | (DenseDType::F64, DenseDType::F64)
                    | (DenseDType::C32, DenseDType::F32)
                    | (DenseDType::C64, DenseDType::F64)
            ));
            assert_eq!(outputs[2].dtype(), dtype);
        }
    }

    #[cfg(feature = "tenferro")]
    #[test]
    fn default_executor_svd_into_writes_strided_destination_views() {
        let data = [1.0_f64, -2.0, 0.5, 4.0];
        let input_shape = [2, 2];
        let input_strides = [1, 2];
        let input = DenseRead::F64(DenseView::new(&data, &input_shape, &input_strides, 0).unwrap());

        let mut executor = DefaultDenseExecutor::new();
        let expected = executor.svd(input).unwrap();

        let mut u = vec![-99.0; 8];
        let mut s = vec![-99.0; 4];
        let mut vt = vec![-99.0; 8];
        let matrix_shape = [2, 2];
        let matrix_strides = [1, 3];
        let s_shape = [2];
        let s_strides = [2];
        executor
            .svd_into(
                input,
                DenseWrite::F64(
                    DenseViewMut::new(&mut u, &matrix_shape, &matrix_strides, 1).unwrap(),
                ),
                DenseWrite::F64(DenseViewMut::new(&mut s, &s_shape, &s_strides, 0).unwrap()),
                DenseWrite::F64(
                    DenseViewMut::new(&mut vt, &matrix_shape, &matrix_strides, 1).unwrap(),
                ),
            )
            .unwrap();

        let expected_u = expected[0].as_f64_slice().unwrap();
        let expected_s = expected[1].as_f64_slice().unwrap();
        let expected_vt = expected[2].as_f64_slice().unwrap();
        for col in 0..2 {
            for row in 0..2 {
                assert_f64_close(u[1 + row + 3 * col], expected_u[row + 2 * col], 1e-12);
                assert_f64_close(vt[1 + row + 3 * col], expected_vt[row + 2 * col], 1e-12);
            }
        }
        for index in 0..2 {
            assert_f64_close(s[2 * index], expected_s[index], 1e-12);
        }
    }

    #[cfg(feature = "tenferro")]
    #[test]
    fn default_executor_qr_into_writes_strided_destination_views() {
        let data = [
            Complex64::new(1.0, 0.5),
            Complex64::new(-2.0, 1.0),
            Complex64::new(0.5, -0.25),
            Complex64::new(4.0, 1.5),
        ];
        let input_shape = [2, 2];
        let input_strides = [1, 2];
        let input = DenseRead::C64(DenseView::new(&data, &input_shape, &input_strides, 0).unwrap());

        let mut executor = DefaultDenseExecutor::new();
        let expected = executor.qr(input).unwrap();

        let sentinel = Complex64::new(-99.0, 0.0);
        let mut q = vec![sentinel; 8];
        let mut r = vec![sentinel; 8];
        let matrix_shape = [2, 2];
        let matrix_strides = [1, 3];
        executor
            .qr_into(
                input,
                DenseWrite::C64(
                    DenseViewMut::new(&mut q, &matrix_shape, &matrix_strides, 1).unwrap(),
                ),
                DenseWrite::C64(
                    DenseViewMut::new(&mut r, &matrix_shape, &matrix_strides, 1).unwrap(),
                ),
            )
            .unwrap();

        let expected_q = expected[0].as_c64_slice().unwrap();
        let expected_r = expected[1].as_c64_slice().unwrap();
        for col in 0..2 {
            for row in 0..2 {
                assert_c64_close(q[1 + row + 3 * col], expected_q[row + 2 * col], 1e-12);
                assert_c64_close(r[1 + row + 3 * col], expected_r[row + 2 * col], 1e-12);
            }
        }
    }

    #[cfg(feature = "tenferro")]
    #[test]
    fn default_executor_eigh_into_writes_strided_destination_views() {
        let data = [
            Complex64::new(4.0, 0.0),
            Complex64::new(1.0, 0.5),
            Complex64::new(1.0, -0.5),
            Complex64::new(3.0, 0.0),
        ];
        let input_shape = [2, 2];
        let input_strides = [1, 2];
        let input = DenseRead::C64(DenseView::new(&data, &input_shape, &input_strides, 0).unwrap());

        let mut executor = DefaultDenseExecutor::new();
        let expected = executor.eigh(input).unwrap();

        let mut values = vec![-99.0; 4];
        let sentinel = Complex64::new(-99.0, 0.0);
        let mut vectors = vec![sentinel; 8];
        let values_shape = [2];
        let values_strides = [2];
        let matrix_shape = [2, 2];
        let matrix_strides = [1, 3];
        executor
            .eigh_into(
                input,
                DenseWrite::F64(
                    DenseViewMut::new(&mut values, &values_shape, &values_strides, 1).unwrap(),
                ),
                DenseWrite::C64(
                    DenseViewMut::new(&mut vectors, &matrix_shape, &matrix_strides, 1).unwrap(),
                ),
            )
            .unwrap();

        let expected_values = expected[0].as_f64_slice().unwrap();
        let expected_vectors = expected[1].as_c64_slice().unwrap();
        for index in 0..2 {
            assert_f64_close(values[1 + 2 * index], expected_values[index], 1e-12);
        }
        for col in 0..2 {
            for row in 0..2 {
                assert_c64_close(
                    vectors[1 + row + 3 * col],
                    expected_vectors[row + 2 * col],
                    1e-12,
                );
            }
        }
    }

    #[cfg(feature = "tenferro")]
    #[test]
    fn default_executor_rejects_integer_linalg_view() {
        let data = [1_i32, 0, 0, 1];
        let shape = [2, 2];
        let strides = [1, 2];
        let view = DenseView::new(&data, &shape, &strides, 0).unwrap();

        let mut executor = DefaultDenseExecutor::new();
        let err = executor.qr(DenseRead::I32(view)).unwrap_err();

        assert!(matches!(
            err,
            DenseError::Backend {
                backend: DenseBackend::Tenferro,
                op: "qr_read",
                ref message,
            } if message.contains("unsupported dtype")
        ));
    }
}
