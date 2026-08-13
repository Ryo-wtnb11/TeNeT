# tenet-tensors

Compiler and executor for symmetric operations over `tenet-core` fusion-tree
layouts. It exposes expert `*_into` operations, execution contexts, transform
and contraction plans, and backend traits; ordinary code should use
`tenet::prelude::TensorMap` methods.

`cuda` enables device seams and `racah-generated` forwards generated SUN
support. Its examples are diagnostics, not a user tutorial.
