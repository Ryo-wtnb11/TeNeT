# Operation matrix

`operation_matrix.sh` launches one dependency-free Rust executable. Each row
uses a fresh `Runtime` and constructs its fixture before timing. Owned rows run
one cold call followed by warm-up and repeated calls in that same process.
The wrapper records the OS, architecture, CPU name, Rust compiler, Cargo,
full TeNeT SHA and dirty state, lock SHA-256, selected package/features/backend,
and the built Tenferro and Racah package versions, Cargo sources, active
features, and available registry VCS revisions before the CSV. TensorKit
records the Julia kernel/machine/CPU report alongside its pinned package and
BLAS authorities.
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
With the default `racah-generated` feature it also runs exact checked-Generic
SU3 `[1,1]` and SU4 `[1,0,1]` fixtures for public owned transforms, reductions,
compose, and contract, plus compact QR for those fixtures and a one-sector SU3
vacuum control. Compact QR coverage is limited to these checked-SUN owned rows;
no trivial or multiplicity-free row is implied. Transform/compose/contract
checks are public-call self-consistency checks comparing provider authority, spaces, block layout,
full fusion-tree keys, and payload; scale/add also check elementary payload
laws, norm/inner check scalar identities, and compact QR reconstructs the input
outside timing. On this base those fixtures
print an explicit trace exclusion because this exact `SUNFusionRule` lacks the
`SectorCodec` bound required by checked trace dispatch, and a destination
exclusion because it lacks the required multiplicity-free dispatch bounds.
The checked-Generic API exists; these operations are neither emulated nor
replaced for the exact SUN fixtures.
The `qr_compact_generic_layout` selector adds one lower-level public owned-QR
control for an infallible Generic provider. It compares the same two-sector
rank-(1,1) tensor in two padded expert layouts: canonical key order and the
same blocks in reverse order. Both inputs therefore use staged matrix assembly;
only the reversed order requires output scatter. Its two sector matrices have
shapes `d x 2d` and `2d x d`, where `d` is `OP_MATRIX_DEGENERACY`. Fixture
construction, keyed source-coordinate checks, factor-coordinate comparison,
and direct `Q * R` reconstruction are outside the timers. Because that
preflight calls QR, its first timed phase is labeled `first_after_setup`; the
measurement is a matched production-build layout control, not a cold-start
claim.
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

Install the reviewed benchmark lock explicitly in a fresh checkout, then run
the wrapper:

```sh
cp benchmarks/cpu_shape_reuse.Cargo.lock Cargo.lock
OP_MATRIX_MIN_MS=20 benchmarks/operation_matrix.sh
```

The runner requires that root lock. Before timing, it builds the exact selected
release example and takes Tenferro features from Cargo's compiler-artifact
records. Metadata supplies package source and manifest provenance. The build
and all three samples use `--locked --offline`; fetch the locked dependencies
before disconnecting if they are not already in the local Cargo cache.

`OP_MATRIX_DEGENERACY` selects the common per-sector degeneracy (default 8).
`OP_MATRIX_GEMM_BACKEND` is `faer` by default. A macOS BLAS control uses the
same Apple Accelerate provider as the pinned TensorKit environment.
`OP_MATRIX_OPERATION` can select one exact operation name for focused profiling.
`OP_MATRIX_FORM` similarly selects `owned` or `destination`.
`OP_MATRIX_PROFILE_PAUSE_MS` pauses after that row's warm phase so an external
profiler can inspect the live process; it is outside every reported timer.
`OP_MATRIX_CACHE=disabled` constructs each Runtime with a zero tree-transform
byte budget, disabling completed tree-transform admission for a cache-disabled
control. The default is `enabled`.
The executable emits raw per-process samples. `operation_matrix.sh` launches
three fresh processes, preserves every raw row as a `# raw_sample` record, and
then appends the complete median-time row for each CSV key; its warm value is
the mean within that selected child batch, not a cross-child mean.

```sh
OP_MATRIX_OPERATION=contract_input_swap \
OP_MATRIX_FORM=destination \
OP_MATRIX_PROFILE_PAUSE_MS=30000 \
OP_MATRIX_MIN_MS=0 benchmarks/operation_matrix.sh
```

The matched Generic compact-QR layout control is:

```sh
OP_MATRIX_OPERATION=qr_compact_generic_layout \
OP_MATRIX_FORM=owned \
OP_MATRIX_MIN_MS=100 \
benchmarks/operation_matrix.sh
```

The checked-Generic compact input-lowering group runs public compact QR, SVD,
and LQ on the same source geometry. It covers `f64` and genuinely complex
`Complex64` payloads, and compares a canonical contiguous layout with a padded,
reversed-block fallback. `few-large` uses two sectors with alternating matrix
shapes `d x 2d` and `2d x d`; `many-small` uses labels 0 through 15 with the
same alternating geometry and `max(1,d/16)` in place of `d`. Fixture
construction, provider binding, literal
reconstruction (`Q*R`, `U*S*Vh`, `L*Q`), QR/LQ/SVD orthogonality, singular-value
ordering, and source-immutability checks are outside timing. Every invocation
owns fresh returned factors; warm-loop temporaries are dropped per iteration,
while the first returned value is retained until its sample has been printed.
That ownership and teardown scope is identical in baseline and candidate. The
first phase is therefore `first_after_setup`, not a process-cold claim.

```sh
OP_MATRIX_OPERATION=checked_compact_input \
OP_MATRIX_FORM=owned \
OP_MATRIX_DEGENERACY=32 \
OP_MATRIX_MIN_MS=100 \
benchmarks/operation_matrix.sh
```

The benchmark-local XOR provider has a fixed 16-label domain; labels 0 and 1
retain the previous fixture behavior. The `many-small` case has `G=T=16` and 16
source blocks, but it is only a finite cardinality/dataflow control and makes no
physical non-Abelian claim. Existing generated SUN QR rows supply a physical
many-tree supplement. Rank-4, multiplicity, and broader G/T correctness sweeps,
plus direct borrow/pack probes, belong to tests because the public benchmark API
exposes no input-copy counter. Reported allocation totals remain caller-thread
requested calls/bytes; peak, live, native, and worker-thread memory are
unavailable.

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

The remaining #9 rows (ordered contract, compact SVD, compact diagonal,
the multiplicity-free and other compact QR forms, and other lazy-adjoint consumers) are
not substituted with other operations. Add each only with its real public form
and available counters. This diagnostic harness remains outside required CI;
semantic coverage belongs in the existing user API tests.
