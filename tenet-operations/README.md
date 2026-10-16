# tenet-operations

Symmetry-free execution layer beneath `tenet-tensors`; it deliberately does
not consume fusion rules or enumerate fusion trees. Expert entry points include
operation specifications, `OperationError`, backend traits, replay structures,
and `DenseTreeTransformOperations`.

Enable at least one host backend: `cpu-faer`, `cpu-blas`, or a `blas-*` provider.
`provider-inject` is not a complete backend choice: `tenet-dense` refuses every
factorization and solve under it (`DenseError::Unsupported`). The `cuda`
feature still requires a host backend for replay;
raw kernels are not a user-level API.
