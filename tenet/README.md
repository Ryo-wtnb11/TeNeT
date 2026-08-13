# tenet

The public TeNeT facade: provider-typed `Runtime`, `GradedSpace`, `TensorMap`,
tensor operations, and factorization results.

`GradedSpace<R>` and `TensorMap<R, D, S>` keep the fusion-rule provider `R`
concrete. `R` owns sector labels and categorical coefficients; payload scalar
`D` and storage `S` are independent. Host operations are admitted by the
provider and operation-specific capability bounds. Product providers are
built with `left.product(right)` and use nested `ProductSector` labels; factor
order and association are part of the provider type.

Fibonacci and other complex-coefficient providers have partial Host typed
coverage where the operation bounds support them. Typed CUDA is currently a
narrower multiplicity-free `f64` surface.

Start with the crate tutorial in [`src/tutorial.md`](src/tutorial.md).
