//! Tensor-trace replay extension of the symmetry-free
//! [`TensorOperationsBackend`]: trace structures are compiled with fusion
//! rules, so their replay entry points live in the symmetric crate.

use core::ops::{Add, Mul};

use num_traits::{One, Zero};
use tenet_core::{HostReadableStorage, HostWritableStorage, TensorMap};

use crate::tensortrace::{
    tensortrace_fusion_structure_with_strided_kernel, tensortrace_structure_with_strided_kernel,
};
use crate::{
    ConjugateValue, HostTensorOperations, OperationError, RecouplingCoefficientAction,
    TensorOperationsBackend, TensorTraceFusionStructure, TensorTraceStructure,
};

pub trait TensorTraceOperationsBackend: TensorOperationsBackend {
    fn tensortrace_structure_into<
        T,
        const DST_NOUT: usize,
        const DST_NIN: usize,
        const SRC_NOUT: usize,
        const SRC_NIN: usize,
        SDst,
        SSrc,
        DDst,
        DSrc,
    >(
        &mut self,
        allocator: &mut Self::Allocator,
        structure: &TensorTraceStructure,
        dst: &mut TensorMap<T, DST_NOUT, DST_NIN, SDst, DDst>,
        src: &TensorMap<T, SRC_NOUT, SRC_NIN, SSrc, DSrc>,
        alpha: T,
        beta: T,
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
        DDst: HostWritableStorage<T>,
        DSrc: HostReadableStorage<T>;

    fn tensortrace_fusion_structure_into<
        T,
        C,
        const DST_NOUT: usize,
        const DST_NIN: usize,
        const SRC_NOUT: usize,
        const SRC_NIN: usize,
        SDst,
        SSrc,
        DDst,
        DSrc,
    >(
        &mut self,
        allocator: &mut Self::Allocator,
        structure: &TensorTraceFusionStructure<C>,
        dst: &mut TensorMap<T, DST_NOUT, DST_NIN, SDst, DDst>,
        src: &TensorMap<T, SRC_NOUT, SRC_NIN, SSrc, DSrc>,
        alpha: T,
        beta: T,
    ) -> Result<(), OperationError>
    where
        T: Copy
            + Add<T, Output = T>
            + Mul<T, Output = T>
            + PartialEq
            + Zero
            + One
            + ConjugateValue
            + RecouplingCoefficientAction<C>
            + strided_kernel::MaybeSendSync,
        C: Copy,
        DDst: HostWritableStorage<T>,
        DSrc: HostReadableStorage<T>;
}

impl TensorTraceOperationsBackend for HostTensorOperations {
    fn tensortrace_structure_into<
        T,
        const DST_NOUT: usize,
        const DST_NIN: usize,
        const SRC_NOUT: usize,
        const SRC_NIN: usize,
        SDst,
        SSrc,
        DDst,
        DSrc,
    >(
        &mut self,
        allocator: &mut Self::Allocator,
        structure: &TensorTraceStructure,
        dst: &mut TensorMap<T, DST_NOUT, DST_NIN, SDst, DDst>,
        src: &TensorMap<T, SRC_NOUT, SRC_NIN, SSrc, DSrc>,
        alpha: T,
        beta: T,
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
        DDst: HostWritableStorage<T>,
        DSrc: HostReadableStorage<T>,
    {
        tensortrace_structure_with_strided_kernel(allocator, structure, dst, src, alpha, beta)
    }

    fn tensortrace_fusion_structure_into<
        T,
        C,
        const DST_NOUT: usize,
        const DST_NIN: usize,
        const SRC_NOUT: usize,
        const SRC_NIN: usize,
        SDst,
        SSrc,
        DDst,
        DSrc,
    >(
        &mut self,
        allocator: &mut Self::Allocator,
        structure: &TensorTraceFusionStructure<C>,
        dst: &mut TensorMap<T, DST_NOUT, DST_NIN, SDst, DDst>,
        src: &TensorMap<T, SRC_NOUT, SRC_NIN, SSrc, DSrc>,
        alpha: T,
        beta: T,
    ) -> Result<(), OperationError>
    where
        T: Copy
            + Add<T, Output = T>
            + Mul<T, Output = T>
            + PartialEq
            + Zero
            + One
            + ConjugateValue
            + RecouplingCoefficientAction<C>
            + strided_kernel::MaybeSendSync,
        C: Copy,
        DDst: HostWritableStorage<T>,
        DSrc: HostReadableStorage<T>,
    {
        tensortrace_fusion_structure_with_strided_kernel(
            allocator, structure, dst, src, alpha, beta,
        )
    }
}
