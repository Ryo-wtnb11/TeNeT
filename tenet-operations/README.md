# tenet-operations

Symmetry-free execution layer beneath `tenet-tensors`; it deliberately does
not consume fusion rules or enumerate fusion trees. Expert entry points include
operation specifications, `OperationError`, backend traits, replay structures,
and `DenseTreeTransformOperations`.

The default host backend is `blas-openblas`; `--no-default-features` with
`blas-accelerate`, `blas-mkl`, `cpu-blas` or `provider-inject` selects another. The `cuda` feature still requires a host backend for replay;
raw kernels are not a user-level API.
