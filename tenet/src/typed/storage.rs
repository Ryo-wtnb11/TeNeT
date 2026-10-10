//! Sealed storage execution traits: one typed operation body per family,
//! with each storage supplying only its kernels (#1756).

use super::*;

pub(super) mod sealed {
    pub trait Sealed {}
}

impl<D> sealed::Sealed for Vec<D> {}

/// One storage's placement and capability set for rule `R`, payload `D`.
///
/// The family execution traits (e.g. [`ReduceExec`]) are its subtraits. The
/// CUDA impl's bounds state today's device capability once; a rule or payload
/// outside them has no device method at all.
#[doc(hidden)]
pub trait TypedStorage<R, D>: TensorStorage<D> + sealed::Sealed + Sized + 'static {
    /// The compact-representation bridge: `Some(t)` on Host, `None` on CUDA.
    ///
    /// Only compact arms may branch on it; CUDA tensors never hold compact
    /// data, since `to_cuda` densifies and every device kernel outputs dense.
    fn as_host(t: &TensorMap<R, D, Self>) -> Option<&TensorMap<R, D>>;
}

impl<R, D: 'static> TypedStorage<R, D> for Vec<D> {
    fn as_host(t: &TensorMap<R, D, Self>) -> Option<&TensorMap<R, D>> {
        Some(t)
    }
}
