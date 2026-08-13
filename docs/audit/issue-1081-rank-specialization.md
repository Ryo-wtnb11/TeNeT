# Issue #1081: effective-rank specialization measurement

Date: 2026-08-13.  Benchmark commits under test:
`40f5570518176f3a0860abdc43d12253789ad394` and
`fe5dae8785461ba543b066abc4298f4abc82061c`.  Host: Apple arm64, macOS 15.5,
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
preserve the shape.  The three measured regimes are deliberately 32 x 256,
8 x 4096, and 2 x 65536 elements where the rank admits those power-of-two
shapes: many-small, mixed, and few-large.  The bench prints
logical/effective rank, block count, bytes, and warm allocation counts.
Every fixture first checks production replay against a local generic carry loop
bit-for-bit, with NaN sentinels.  The rank-2 and rank-4 prototype loops are
then checked against the same result before timing.  The prototype consumes
the equivalent normalized descriptor locally because the compiled baked layout
is intentionally crate-private; no production API was widened for this
measurement.  Both production and prototype timing black-box their input and
output buffers.

## Representative Criterion results

Each row below is three independent invocations with Criterion's ten samples
per invocation.  Values are raw 95% confidence intervals in microseconds.
The rank-2 and rank-4 reverse rows include the local fixed-loop experiment;
the rank-4 adjacent case has effective rank 3 and is a generic-replay control.

Command:

```text
cargo bench -p tenet-operations --bench tree_transform_lowering -- \
  'rank_2/adjacent_transpose|rank_4/adjacent_transpose|rank_4/reverse' \
  --sample-size 10 --measurement-time 0.2 --warm-up-time 0.2
```

| effective layout and regime | production runs 1 / 2 / 3 | prototype runs 1 / 2 / 3 |
| --- | --- | --- |
| rank 2 adjacent, 32 x 256 | 3.1931–3.5634 / 3.2433–3.3370 / 3.1568–3.2767 | 2.7540–2.8386 / 2.7493–2.9559 / 2.7704–2.9159 |
| rank 2 adjacent, 8 x 4096 | 10.131–10.479 / 10.054–10.185 / 10.311–11.320 | 10.917–11.212 / 10.922–12.068 / 10.783–10.916 |
| rank 2 adjacent, 2 x 65536 | 77.370–87.593 / 79.615–79.930 / 76.584–77.885 | 82.955–83.971 / 78.824–80.312 / 79.594–80.548 |
| logical 4 / effective 3 adjacent, 32 x 256 | 4.7804–4.8777 / 4.7769–4.9579 / 4.7283–4.7761 | n/a |
| logical 4 / effective 3 adjacent, 8 x 4096 | 12.595–13.416 / 12.351–12.836 / 12.426–13.223 | n/a |
| logical 4 / effective 3 adjacent, 2 x 65536 | 49.532–56.392 / 45.815–49.398 / 45.870–52.892 | n/a |
| rank 4 reverse, 32 x 256 | 5.0337–5.1379 / 5.0137–5.3419 / 5.0077–5.0756 | 4.7793–4.9737 / 4.5885–4.6201 / 4.6169–4.7092 |
| rank 4 reverse, 8 x 4096 | 13.597–14.009 / 12.988–13.049 / 13.254–13.902 | 13.328–13.376 / 13.558–14.022 / 13.328–13.865 |
| rank 4 reverse, 2 x 65536 | 94.244–95.118 / 91.517–92.261 / 94.438–95.563 | 94.763–95.962 / 95.110–97.218 / 94.192–101.77 |

The rank-2 many-small prototype is consistently faster, but by roughly
11–18%, below the 20% strided gate.  The rank-4 reverse prototype has a
roughly 6–9% many-small difference and no consistent sign in the other two
regimes.  These prototype timings omit replay admission and therefore cannot
establish the required 10% end-to-end warm-total result.  They provide no
evidence to adopt a TeNeT rank-specialized path.  Every measured warm replay
allocation result was `calls=0 bytes=0`; rank 8/16 are constructed and
bitwise-checked but not timed as specialization candidates.

## Decision

Do not adopt TeNeT logical-rank or fixed effective-rank specialization from
this evidence.  It does not meet the stated strided or end-to-end gates, and
the generic fused carry loop already has no warm allocation.  This is not a
blanket rejection of a future, measured specialization.

The Multi pack/GEMM/scatter path has no isolated fixed-rank prototype through
the current public/internal benchmark boundary.  It is therefore unmeasured
here.  #1081 must remain open unless its acceptance criteria are explicitly
narrowed to Single-block strided replay; this document alone cannot close it.

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
