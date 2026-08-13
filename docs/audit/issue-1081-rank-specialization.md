# Issue #1081: effective-rank specialization measurement

Date: 2026-08-13.  Benchmark commit under test:
`40f5570518176f3a0860abdc43d12253789ad394`.  Host: Apple arm64, macOS 15.5,
Rust 1.96.0, `cpu-faer` default feature.

## Question

The relevant rank is not a tensor's logical rank.  `TreeTransformStructure`
normalizes each Single block to a `BakedFusedLayout` whose effective rank is
the number of non-fusible `(dimension, destination stride, source stride)`
coordinates.  An identity layout is effective rank 1 at logical ranks 2, 4,
8, and 16; a reverse layout remains effective rank 2, 4, 8, and 16
respectively.  An adjacent transpose is effective rank 2 for logical rank 2
and rank 3 for logical ranks 4, 8, and 16 in this column-major fixture.

`tree_transform_lowering` constructs Single-block fixtures for logical ranks
2/4/8/16, identity/adjacent-transpose/reverse permutations, and 32/8/2 block
regimes.  Dimensions are equal within each block, so all tested permutations
preserve the shape.  The target is about 2 Mi elements per fixture; integer
power-of-two dimensions make some mixed/few-large cases smaller.  The bench
prints logical/effective rank, block count, bytes, and warm allocation counts.
Every fixture first checks production replay against a local generic carry loop
bit-for-bit, with NaN sentinels.  The rank-2 and rank-4 prototype loops are
then checked against the same result before timing.  The prototype consumes
the equivalent normalized descriptor locally because the compiled baked layout
is intentionally crate-private; no production API was widened for a rejected
optimization experiment.

## Representative Criterion results

Command (Criterion uses 10 samples and reports its confidence interval):

```text
cargo bench -p tenet-operations --bench tree_transform_lowering -- \
  'rank_2/adjacent_transpose/blocks_32'
cargo bench -p tenet-operations --bench tree_transform_lowering -- \
  'rank_4/reverse/blocks_32'
```

| effective layout | production warm strided replay | fixed-loop prototype |
| --- | ---: | ---: |
| rank 2 adjacent transpose, 32 blocks, 16 MiB | 1.3106–1.3389 ms | 1.3710–1.4601 ms |
| rank 4 reverse, 32 blocks, 16 MiB | 1.5457–1.5774 ms | 1.6141–1.6987 ms |

The fixed rank loops are slower in both representative non-contiguous cases;
they are not an end-to-end improvement, much less the required 20% strided or
10% warm-total win.  The full fixture matrix is constructed and checked when
the benchmark binary starts.  Every printed warm replay allocation result was
`calls=0 bytes=0`, including the rank-16 reverse cases.

## Decision

Reject TeNeT logical-rank specialization and fixed effective-rank stride
specialization.  The existing generic fused carry loop already has the needed
runtime descriptor and has no warm allocation.  Adding rank-specific code
would increase maintenance and code size without meeting the admission gate.

This does not reject an upstream strided-kernel optimization: if an upstream
kernel provides a measured fast path that meets the same gates, TeNeT should
call it through `StridedHostKernelAdapter`, rather than duplicate it or own a
separate rank-specialized kernel family.

No production executable changed, so a release binary-size or compile-time
regression gate is not applicable to this measurement-only change.  The
reproducible compilation check is:

```text
cargo check -p tenet-operations --benches
```
