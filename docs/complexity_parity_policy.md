# Structured-complexity policy

TeNeT's public storage forms and operations carry complexity contracts. An
implementation must preserve the appropriate asymptotic FLOP count and
working-set storage for the structure it accepts. Producing the right values by
unnecessarily densifying a compact operand or materializing an avoidable
intermediate is a defect, even when the result is numerically correct.

External implementations may demonstrate an attainable bound or supply a
comparison oracle. They are supporting evidence, not the definition of this
contract.

## The rule

> For every operation and supported storage form, TeNeT must preserve the
> asymptotic advantage of the structure it accepts. An implementation that
> discards known structure and thereby adds a factor in a size parameter is a
> bug to be fixed, not merely a different implementation.

Different kernels may have different constant factors. A different asymptotic
order is not an implementation detail.

## Why this needs stating

TeNeT deliberately exposes one provider-typed `TensorMap` while retaining
compact diagonal and lazy-adjoint storage internally. Generic dense lowering
cannot infer every structured fast path. TeNeT therefore places a small number
of explicit routes at the layer where the structure is still visible, with a
numerically correct fallback for other geometry. A fallback whose structured
complexity is not yet preserved must be classified as a gap; numerical support
alone is not a claim of complexity compliance.

## Diagonal storage as the worked example (#55)

Let `d` = per-sector bond degeneracy (the diagonal's essential size, `O(d)`),
`n` = the other operand's open-leg size.

| Path | Required order | TeNeT status |
|------|----------------|--------------|
| `compose` / `U*S*Vh` | `O(d·n)` work; `O(d)` compact diagonal payload; unavoidable other operand/output storage is `O(d·n)`; no `O(d²)` diagonal materialization | **compliant** — explicit block scaling (#72) |
| single-axis composition-equivalent `contract` / `tensor!` with a diagonal | `O(d·n)` work; `O(d)` compact diagonal payload; unavoidable other operand/output storage is `O(d·n)`; no `O(d²)` diagonal materialization | **compliant** — explicit provider-typed block scaling (#584) and typed macro execution (#750) |
| other accepted diagonal contraction geometries | derive per geometry | **gap / unproved** — the dense fallback may add a factor `d`; no general order-correct claim |

That row was a genuine order regression — densifying to `O(d²)` and GEMMing
`O(d²·n)`, a factor `d` in both FLOPs and transient storage — and it was closed
as a structured-order obligation rather than a performance nicety. The current
provider-typed path from #584 scales the *other* operand's contracted leg by
the spectrum and lays the result out with one `permute`, which is where all the
recoupling stays. Constant-factor payoff is separate from the structural order
obligation.

The fast path covers only the listed single-axis geometries that are a
composition on the contracted leg, in either order. Other accepted geometries
decline to the dense route and are not classified as complexity-compliant by
this document. Reach for `compose` when the shape allows it; the two proved
routes are pinned to the same values and destination space in
`tenet/tests/typed_facade.rs`.

## Checklist for an operation

1. What FLOP and working-set orders follow from each supported storage form and
   operation geometry? Cite an external implementation only when it supplies
   relevant evidence.
2. Does TeNeT match both orders? If a fallback densifies a structured operand
   or materializes an avoidable intermediate, that is a violation.
3. For a new geometry, add one explicit fast path at the paying layer, not
   scattered runtime branches or a combinatorial type zoo. If the correct order
   cannot land immediately, classify the accepted fallback as a
   complexity-order gap and route it to an issue rather than calling it
   compliant; provide order-correct interim guidance where one exists.
