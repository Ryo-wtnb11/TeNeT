# Numerical comparisons in tests

A test asserts bit equality only when the promised result is exact:

- **Data movement:** copy, permute, scatter, restrict/embed, conjugation,
  sign application, and results left untouched after an error.
- **Determinism:** replaying the same plan or cache entry, or the same
  schedule with the same inputs, gives identical results.
- **Combinatorial output:** block and tree identity, signs, coefficient and
  truncation selection, and values that are exact by construction (for
  example small-integer or dyadic fixtures whose every partial sum is
  representable, so every summation order gives the same bits).

Any other floating-point result may be computed in more than one valid order
(GEMM, reductions, recoupling sums, factorizations, scaled accumulation). Such
a test compares against an **independent oracle** — a dense expansion, a hand
or closed-form value, a reference fixture, a reconstruction residual such as
`‖A − U S Vᴴ‖`, or orthogonality — within a dtype tolerance. When two paths
agreeing is itself the contract (for example full and compact factorizations
of the same input), the test compares them within the tolerance and says so in
its name or comment.

Comparing against the bits of a previous implementation is not a contract. It
freezes one accumulation order and is weaker evidence than an oracle.

## Tolerance

Use `tests/support/numerics.rs`, included with
`#[path = "<relative path>/tests/support/numerics.rs"] mod numerics;` from a
module that has `Complex32` and `Complex64` in scope. It applies

    K * sqrt(terms) * eps(dtype) * max(1, scale),    K = 32,

the rule of `tenet/tests/single_precision_oracle/mod.rs`:

- `eps` is `f32::EPSILON` for `f32`/`Complex32` and `f64::EPSILON` for
  `f64`/`Complex64`; complex values are compared by the modulus of the error;
- `terms` is the number of floating terms that reach one compared entry
  (the contracted length of a GEMM, the length of a reduction). When the
  exact count depends on recoupling, use a bound computed from the fixture
  shape (for example the product of the contracted legs' degeneracies) and
  say so at the call site;
- `scale` is the largest oracle magnitude.

Factorizations and other results whose error grows with conditioning pass an
explicit conditioning factor (`*_scaled`) and state where it comes from. Do not
loosen a bound to make a test pass; a failure under the rule is a finding.
