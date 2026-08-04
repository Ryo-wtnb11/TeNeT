# Operation matrix

`operation_matrix.sh` launches one dependency-free Rust executable. Each row
uses a fresh `Runtime` and constructs its fixture before timing. Owned rows run
one cold call followed by warm-up and repeated calls in that same process.
Here `cold` means that the fresh `Runtime` tree-transform store is empty;
process-global interned structure may already exist.
Destination rows require an exact output from the owned operation; they report
`first_after_setup` and `warm_after_setup`, never a false cold sample.
Validation assertions run after the timer.

The executable covers U1 and SU2 controls for owned and actual
caller-owned-destination forms of `permute` and arbitrary-axis `contract`.
The contraction rows distinguish canonical input order, contracted-input
swap, and contracted-input plus output swap. An owned `compose` row checks
that its result equals the canonical contraction before reporting it.
Runtime tree-transform counters are reported as cold and warm deltas. The same
snapshots report process-global fusion-layout and complete-HomSpace cache
deltas, with charged bytes before and after each phase. The fusion-layout cache
does not expose a hit counter, so only its available miss, eviction, bypass,
entry, and charged-byte fields are printed.

The destination rows call the stable public `permute_overwrite_into` and
`contract_overwrite_into` APIs. Exact-layout admission is attached to the
Runtime tree-transform entry, but it has no public activity counter, so the
`exact_layout_admission` column is literal `NA`. The 37-column CSV schema omits
the old erased-only destination preparation/comparison fields. Unsupported
instrumentation is also printed as `NA`; the harness does not infer
allocations, scratch, provider queries, dense kernel calls, or transfers from
elapsed time.

```sh
OP_MATRIX_MIN_MS=20 benchmarks/operation_matrix.sh
```

`OP_MATRIX_DEGENERACY` selects the common per-sector degeneracy (default 8).
`OP_MATRIX_GEMM_BACKEND` is `faer` by default. A macOS BLAS control uses the
same Apple Accelerate provider as the pinned TensorKit environment:

```sh
OP_MATRIX_GEMM_BACKEND=blas \
OP_MATRIX_CARGO_FEATURES=blas-accelerate \
benchmarks/operation_matrix.sh

julia --project=benchmarks/tensorkit_benchmark \
    benchmarks/tensorkit_microbench.jl 8 300
```

The TensorKit script records its exact revision, Julia and AppleAccelerate
versions, BLAS configuration, first-call timing, warm timing, and Julia's
per-call allocated bytes. Its first-call row can include JIT compilation and
may observe process-global TensorKit caches warmed by earlier rows; it is not
directly comparable to TeNeT's fresh-`Runtime` cold row. Only matching warm
rows under the recorded one-thread BLAS configuration are timing controls.

The remaining #724 rows (transpose/repartition, ordered contract,
trace, add/inner/norm, compact SVD/QR, compact diagonal, and lazy adjoint) are
not substituted with other operations. Add each only with its real public form
and available counters. This diagnostic harness remains outside required CI;
semantic coverage belongs in the existing user API tests.
