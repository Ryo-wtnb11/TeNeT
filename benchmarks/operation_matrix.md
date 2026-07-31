# Operation matrix

`operation_matrix.sh` launches one dependency-free Rust executable. Each row
uses a fresh `Runtime` and constructs its fixture before timing. Owned rows run
one cold call followed by warm-up and repeated calls in that same process.
Here `cold` means that the fresh `Runtime` tree-transform store is empty;
process-global interned structure may already exist.
Destination rows require an exact output from the owned operation; they report
`first_after_setup` and `warm_after_setup`, never a false cold sample.
Validation assertions run after the timer.

The initial executable covers U1 and SU2 controls for owned and actual
caller-owned-destination forms of `permute` and arbitrary-axis `contract`.
Runtime tree-transform counters are reported as cold and warm deltas. The
destination rows call the stable public `permute_overwrite_into` and
`contract_overwrite_into` APIs; their internal preparation/comparison counters
are not public and are therefore `NA`. Unsupported instrumentation is also
printed as `NA`; the harness does not infer allocations, scratch, provider
queries, dense kernel calls, or transfers from elapsed time.

```sh
OP_MATRIX_MIN_MS=20 benchmarks/operation_matrix.sh
```

The remaining #724 rows (transpose/repartition, compose, ordered contract,
trace, add/inner/norm, compact SVD/QR, compact diagonal, and lazy adjoint) are
not substituted with other operations. Add each only with its real public form
and available counters. This diagnostic harness remains outside required CI;
semantic coverage belongs in the existing user API tests.
