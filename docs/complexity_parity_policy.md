# Complexity-parity policy

TeNeT ports TensorKit. The **mechanism** may be Rust-idiomatic and differ from
TensorKit's, but the **asymptotic complexity (FLOPs and storage) must match
TensorKit's**. Reproducing a TK behavior at a worse order is a defect, not an
acceptable simplification — even when the result is numerically correct.

## The rule

> For every operation, TeNeT's FLOP count and working-set storage must be the
> same order as TensorKit's for that operation. If TK is `O(f(n))`, a TeNeT
> implementation that is `O(f(n)·k)` for some size parameter `k` is a bug to be
> fixed, not a "different implementation".

"Same order, different constant factor" is fine and expected (Rust vs Julia,
faer vs OpenBLAS, explicit loops vs BLAS). "Different order" is not.

## Why this needs stating: Rust has no multiple dispatch

Several TK efficiencies are *emergent* from Julia's type system + multiple
dispatch, so TK gets them "for free" with no special code. The canonical case is
`DiagonalTensorMap`: its `block(D, c)` returns a Julia `Diagonal`, and
`LinearAlgebra.mul!(::Matrix, ::Diagonal, …)` is already a scaling — so a
diagonal in *any* multiplication (including a general permuting `@tensor`
contraction) runs as an `O(rank)` scaling, never a dense `O(rank²)` GEMM, with
zero lines of TK code dedicated to it.

Rust (and TeNeT's deliberate single-`Tensor` design) has no such automatic
dispatch. The same complexity must therefore be produced **explicitly**. The
Rust-idiomatic shape is *not* "reproduce Julia's pervasive free dispatch"
(a separate `DiagonalTensor` type → combinatorial `impl`s, or a runtime
`match Data::Diagonal` scattered through every op). It is **a small number of
explicit fast paths placed at the layer where they pay**, plus a correct
fallback everywhere else — where "correct" still means *order-correct*.

So the language difference changes *where and how* the fast path is written; it
does **not** license dropping to a worse order in the fallback.

## Diagonal storage as the worked example (#55)

Let `d` = per-sector bond degeneracy (the diagonal's essential size, `O(d)`),
`n` = the other operand's open-leg size.

| Path | TK | TeNeT status |
|------|----|--------------|
| `compose` / `U*S*Vh` (mul!/lmul!/rmul!) | `O(d·n)` scale, `O(d)` store | **compliant** — explicit block scaling (#72) |
| general `contract` / `tensor!` with a diagonal | `O(d·n)` scale, `O(d)` store | **compliant** — explicit block scaling on the erased facade (#75) and on the provider-typed one (#584) |

That row was a genuine order regression — densifying to `O(d²)` and GEMMing
`O(d²·n)`, a factor `d` in both FLOPs and transient storage — and it was closed
as an order-parity obligation rather than a performance nicety: #75 on the erased
facade, #584 on the provider-typed one. Both scale the *other* operand's
contracted leg by the spectrum and lay the result out with one `permute`, which
is where all the recoupling stays. The modest *constant-factor* payoff for SU(2)
(where leg-permute cost dominates and is unavoidable, matching TK) was never the
point: order parity is the obligation; constant-factor wins are separate.

Both fast paths cover the single-axis geometries that are a composition on the
contracted leg, in either order, and *decline* to the dense route otherwise — a
decline costs the factor `d` back, so reach for `compose` (mul!) when the shape
allows it, which needs no guard to hit the scaling. The two routes are pinned to
the same values and the same destination space in `tenet/tests/typed_facade.rs`.

## Checklist when porting a TK operation

1. What is TK's FLOP order and working-set order for this op? (Read the source;
   note any `Diagonal`/structured-type dispatch that makes it cheap for free.)
2. Does the TeNeT port match both orders? If a fallback densifies a structured
   operand or materializes an intermediate TK never forms, that is a violation.
3. If Rust can't get it "for free", add one explicit fast path at the paying
   layer (not scattered runtime branches, not a combinatorial type zoo) and keep
   the fallback order-correct.
4. If order parity can't land immediately, file an issue tagged as an
   order-parity gap (not a perf nice-to-have) and add order-correct interim
   guidance.
