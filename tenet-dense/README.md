# tenet-dense

Expert dense-backend boundary. Symmetric layers use `DenseExecutor`,
`DenseView`, `DenseViewMut`, `DenseTensor`, `DensePlacement`, and `DenseError`
without depending on a concrete kernel provider. Ordinary applications choose
their backend through `tenet::prelude::RuntimeBuilder` instead.

`tenferro` supplies the default CPU executor; `cuda` exports CUDA context,
storage, and kernels. CPU BLAS selection is supplied by the relevant tenferro
feature.
