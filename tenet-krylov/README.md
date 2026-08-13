# tenet-krylov

Private algorithm-layer support, outside TeNeT's initial public API closure.
It provides real-`f64` Conjugate Gradient through `cg`, `CgOptions`, and
`CgResult`, over the `KrylovVector` and `LinearOperator` traits. It has no
crate-specific features and is not used by the tensor layer.
