# tenet-krylov

Private algorithm-layer support, outside TeNeT's initial public API closure.
It provides real-`f64` Conjugate Gradient through `cg`, `CgOptions`, and
`CgResult`, over the `KrylovVector` and `LinearOperator` traits. A damped
system `(A + d I) x = b` is solved by passing an operator closure that adds
`d * x`. It has no crate-specific features and is not used by the tensor
layer.
