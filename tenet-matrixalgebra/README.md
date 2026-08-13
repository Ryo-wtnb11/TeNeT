# tenet-matrixalgebra

Expert blockwise factorizations, solves, matrix functions, and spectrum
truncation for fusion tensors. Key entries are the SVD/QR/eigen APIs,
`solve_left_direct_dyn`, `Truncation`, `TruncationSpace`, and `SectorSpectrum`.
Ordinary applications use the corresponding `tenet::prelude::TensorMap`
methods.

The `cuda` feature forwards CUDA support to the dense and tensor layers.
Diagnostic examples are not user documentation.
