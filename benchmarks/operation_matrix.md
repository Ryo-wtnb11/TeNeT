# Operation matrix

`operation_matrix.sh` launches one dependency-free Rust executable. Each row
uses a fresh `Runtime` and constructs its fixture before timing. Owned rows run
one cold call followed by warm-up and repeated calls in that same process.
The wrapper records the OS, architecture, CPU name, Rust compiler, Cargo,
TeNeT, and Tenferro authorities before the CSV. TensorKit records the Julia
kernel/machine/CPU report alongside its pinned package and BLAS authorities.
Here `cold` means that the fresh `Runtime` tree-transform store is empty;
process-global interned structure may already exist.
Destination rows require an exact output from the owned operation; they report
`first_after_setup` and `warm_after_setup`, never a false cold sample.
Validation assertions run after the timer.

The executable covers U1 and SU2 controls for owned and actual
caller-owned-destination forms of `permute`, planar `transpose`,
`repartition`, owned partial trace (direct and lazy-adjoint input), and
arbitrary-axis `contract`. Owned `scale`/`add` and scalar `norm`/`inner` rows
cover both direct tensors and pre-built lazy adjoints.
The contraction rows distinguish canonical input order, contracted-input
swap, and contracted-input plus output swap. An owned `compose` row checks
that its result equals the canonical contraction before reporting it.
Runtime tree-transform counters are reported as cold and warm deltas. The same
snapshots report process-global fusion-layout and complete-HomSpace cache
deltas, with charged bytes before and after each phase. The fusion-layout cache
does not expose a hit counter, so only its available miss, eviction, bypass,
entry, and charged-byte fields are printed.

The destination rows call the stable public `permute_overwrite_into`,
`transpose_overwrite_into`, `repartition_overwrite_into`, and
`contract_overwrite_into` APIs. Exact-layout admission is attached to the
Runtime tree-transform entry, but it has no public activity counter, so the
`exact_layout_admission` column is literal `NA`. The 38-column CSV schema omits
the old erased-only destination preparation/comparison fields. Caller-thread
Rust allocation calls and requested bytes are measured directly for each
phase; divide warm totals by `iterations` for a per-call value. They exclude
worker-thread and native-BLAS allocation and do not represent frees, live
bytes, or peak memory. Scratch, provider queries, dense kernel calls, and
transfers remain `NA` rather than being inferred from elapsed time.

```sh
OP_MATRIX_MIN_MS=20 benchmarks/operation_matrix.sh
```

`OP_MATRIX_DEGENERACY` selects the common per-sector degeneracy (default 8).
`OP_MATRIX_GEMM_BACKEND` is `faer` by default. A macOS BLAS control uses the
same Apple Accelerate provider as the pinned TensorKit environment.
`OP_MATRIX_OPERATION` can select one exact operation name for focused profiling.
`OP_MATRIX_FORM` similarly selects `owned` or `destination`.
`OP_MATRIX_PROFILE_PAUSE_MS` pauses after that row's warm phase so an external
profiler can inspect the live process; it is outside every reported timer.

```sh
OP_MATRIX_OPERATION=contract_input_swap \
OP_MATRIX_FORM=destination \
OP_MATRIX_PROFILE_PAUSE_MS=30000 \
OP_MATRIX_MIN_MS=0 benchmarks/operation_matrix.sh
```

The Apple Accelerate control is:

```sh
OP_MATRIX_GEMM_BACKEND=blas \
OP_MATRIX_CARGO_FEATURES=blas-accelerate \
benchmarks/operation_matrix.sh

julia --project=benchmarks/tensorkit_benchmark \
    benchmarks/tensorkit_microbench.jl 8 300
```

The TensorKit script records its exact revision, Julia and AppleAccelerate
versions, BLAS configuration, first-call timing, warm timing, and Julia's
per-call allocated bytes. It uses the same rank-3 fixture and axis placement
for TeNeT's `permute`, planar `transpose`, and `repartition` rows, and the same
rank-4 fixture and middle trace pair for the direct/lazy-adjoint trace rows.
The same rank-4 values feed matched direct/lazy-adjoint `scale`, `add`, `norm`,
and `inner` rows; both implementations validate the same norm identities
outside the timer.
It also reports TensorKit's real caller-destination trace form;
the canonical TeNeT typed facade has no matching destination method. Its
first-call row can include JIT compilation and may observe
process-global TensorKit caches warmed by earlier rows; it is not directly
comparable to TeNeT's fresh-`Runtime` cold row. Only matching warm rows under
the recorded one-thread BLAS configuration are timing controls.

The remaining #9 rows (ordered contract, compact SVD/QR, compact diagonal,
and other lazy-adjoint consumers) are
not substituted with other operations. Add each only with its real public form
and available counters. This diagnostic harness remains outside required CI;
semantic coverage belongs in the existing user API tests.
