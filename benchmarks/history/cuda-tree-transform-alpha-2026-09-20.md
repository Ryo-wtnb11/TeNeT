# Caller scale in the device tree-transform executor (G2b-1, issue #1318)

Base: origin/main `89b1cde6`. Branch `g2b1-executor-alpha`. No dependency change.

## What the scale does

`CudaTreeTransformExecutor::replay(.., alpha, mode)` reproduces the host's
`alpha` (`transform_replay.rs`: Single `(alpha*c)*x` at :5010/:4236/:2733→:3596,
Multi pack with `T::one()` :5284, GEMM alpha=1 beta=0 :5113-5121, alpha at the
scatter only :5228-5239/:5304-5380/:3644; no short circuit —
`kernel_adapter.rs:1036` always computes `alpha * value`, its `is_zero` tests
are on beta at :899/:964).

| submission | descriptor alpha | 1x1 data operand |
|---|---|---|
| Single move, `alpha != 0` | `alpha` | block coefficient |
| Multi scatter, `alpha != 0` | `alpha` | context `1` |
| Multi pack, recoupling GEMM | `1` | context `1` / recoupling matrix |
| Single move + Multi scatter, `alpha == 0` | `1` | context zero template[0] |
| inactive-layout zero fill | `1` | unchanged (scale ignored) |

`alpha == 0` is IEEE `==`, so `-0.0` is a zero scale. It is submitted as a zero
*operand*, never as a descriptor alpha, because a descriptor alpha of zero lets
cuTENSOR skip the source read and erase the NaN/Inf the host propagates.

## Ordering decision (deviation from the design text)

`g2b-design.md` §10 asks for the zero template to be reserved "before Stage C".
That conflicts with the rejection-order contract repaired in G2a-3 (no upload,
allocation, plan-cap raise or template reservation before the admission
verdict). The reservation therefore stays where G2a-3 put it — immediately
after Stage C, before any submission, now sized
`max(max_zero_len, alpha == 0)`. Nothing the zero operand needs is lost: it only
has to exist before the first submission.

## Plan-signature accounting

`required_plan_entries` is unchanged by the scale. The scale is an
execution-time argument of `cutensor.contract` and not in
`CutensorContractionKey`; the zero-scale operand is a `[1,1]/[1,1]` view like
every other coefficient operand and carries the same `size_of::<D>()`
alignment, so it is the same plan. Pinned on device: a cold `alpha = 0` replay
of an already prepared structure adds **0** plan-cache misses over the
`alpha = 1` one, and a warm sweep over `{1, -2.5, 0, -0.0, 0.5}` adds 0 misses
and 0 evictions (`a_warm_replay_is_transfer_free_and_plan_stable_for_every_caller_scale`).

## Disclosed differences vs host

- Single blocks round as `alpha*(c*x)` where the host rounds `(alpha*c)*x`; one
  rounding position, and the overflow position moves with it. Multi blocks keep
  the host's order exactly (`alpha*(U x)`).
- At `alpha == 0` the written zeros carry the sign of `0*x` alone; the host's
  carry `sign(alpha)*sign(c)` too. Compares equal, not bitwise.
- `alpha = 1` with coefficient-1 f64 blocks stays bitwise identical to the host
  (`a_coefficient_of_one_moves_f64_payloads_bitwise`).

## Evidence

CI (CPU, no device): the structure walker (`tenet-operations/tests/common`) and
the categorical walker (`tenet-tensors/tests/categorical_recoupling`) both gained
`alpha` and are pinned against the **host** executor for
`alpha in {1, -2.5, 0, -0.0, 0.5-1.25i}` x {f64, Complex64} x {Overwrite,
Axpby(1)} x every fixture, conjugated sources included. Negative controls:
applying the scale at the pack as well (`alpha^2`, expressed by folding it into
`U`, and on provider 6j data by evaluating the oracle at `alpha*alpha`) must
differ from the host result; a zero scale over a NaN source must stay NaN on the
host.

Device (A100, `CUDA_VISIBLE_DEVICES=1`, cuTENSOR 2.5.0, CUDA 12.6, debug):

```
tests/cuda_region_axpby.rs        14 passed; 0 failed   (3.70s)
tests/cuda_tree_transform.rs      22 passed; 0 failed   (6.34s)
tests/cuda_tree_transform_categorical.rs 4 passed; 0 failed (1.69s)
full device suite                122 passed; 0 failed   (exit 0)
```

New device tests: `every_caller_scale_matches_the_host_and_the_oracle`,
`every_caller_scale_matches_the_host_on_provider_compiled_recoupling`,
`a_zero_caller_scale_multiplies_rather_than_skipping_the_source` (NaN source →
NaN, asserted against the host, not against an assumed pattern),
`a_zero_caller_scale_over_a_poisoned_destination_matches_the_host`,
`a_warm_replay_is_transfer_free_and_plan_stable_for_every_caller_scale`
(h2d/d2h/allocs = 0 for every scale including 0; prepared structures, workspace
bytes and plan requirement unchanged), `a_zero_caller_scale_is_rejected_in_the_same_order`.

Build note: the shared `gl2-target` produced a stale-artifact compile failure
for this revision (`tenet-tensors` linked an older `tenet-operations` whose
`replay` still took 7 arguments); the run used a private
`/data2/ryo-w/gpu-phase/g2b1-target`, removed afterwards.
