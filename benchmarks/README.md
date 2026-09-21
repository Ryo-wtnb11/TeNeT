# Benchmarks

Benchmark harnesses are executable measurement tools. Recorded reports are
revision-specific evidence, not current performance claims.

## Current harnesses

- `operation_matrix.sh` and `operation_matrix.md`: Host operation matrix,
  environment capture, and comparison protocol.
- `tensorkit_semantic_oracle.jl` and `tensorkit_tsvd_crosscheck.jl`: semantic
  and factorization oracle programs with pinned Julia environments.
- `qspace_su2_oracle.m`: recorded SU(2) oracle source and output.
- `eager_overhead_ledger.sh` (`tenet/examples/eager_overhead_ledger.rs`,
  `tensorkit_eager_overhead.jl`, `eager_overhead_phases.py`): per-call time,
  allocations, thread layouts, and sampled phase attribution of the eager
  primitives on small many-block tensors against TensorKit.

Run a harness at the revision being evaluated and keep its correctness checks
enabled. A historical result does not become current because its script still
exists.

## Historical results

- [July 2026 contraction microbenchmark](history/contraction-microbench-2026-07-11.md)
- [Eager per-call overhead ledger](history/eager-overhead-ledger-2026-09-21.md)
- [Generic-path performance review](issue_1007_current_main.md)
- [Checked Generic values input borrowing](history/checked-generic-values-borrow-2026-09-13.md)
- [Warm-cache audit](issue_118_warm_cache.md)
- [Allocation evidence](issue_124_allocation_evidence.md)
- [Fusion-layout cache benchmark](issue_245_layout_cache.md)

The issue-specific shell script `issue_118_audit.sh` is retained only to
reproduce its linked historical report and external-checkout setup.
