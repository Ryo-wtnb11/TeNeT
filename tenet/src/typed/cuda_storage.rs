use super::storage::sealed;
use super::*;

impl<D: CudaPayload> sealed::Sealed for CudaStorage<D> {}

impl<R, D> TypedStorage<R, D> for CudaStorage<D>
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64> + CheckedFusionAlgebra + SectorCodec,
    D: CudaPayload,
{
    fn as_host(_: &TensorMap<R, D, Self>) -> Option<&TensorMap<R, D>> {
        None
    }
}
