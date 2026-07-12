# Issue 118 warm-cache audit

## Identity and complexity

`HomSpaceId` is a collision-safe semantic identity, not a counter token. It
stores a cached prehash and an `Arc` to the full immutable hom-space key.
Normal `Hash` is O(1). Construction hashes the full space, and equality falls
back to a full semantic comparison when prehashes match and the `Arc`s differ;
those paths are not strictly O(1).

The bounded hom-space LRU only canonicalizes the `Arc` pointer. Eviction does
not change semantic equality or hashing, so an old and a re-interned equal
space still hit the scratch and adjoint caches. Scratch metadata is bounded and
resettable. Tensor buffers remain owned by execution contexts and placements.

## Reproduction

Run from the TeNeT checkout with the finite-torus migration checkout available:

```sh
RAYON_NUM_THREADS=1 \
OPENBLAS_NUM_THREADS=1 \
OMP_NUM_THREADS=1 \
MKL_NUM_THREADS=1 \
FINITE_TORUS_DIR=/path/to/finite-torus/migrate \
TENFERRO_DIR=/path/to/tenferro \
RUNS=5 WARMLOOP=40 \
sh benchmarks/issue_118_audit.sh | tee issue-118-audit.txt
```

The finite-torus manifest must resolve TeNeT from the checkout running the
script; the runner rejects stale binaries or another path dependency. Cargo
metadata also establishes the resolved tenferro checkout and rejects a
different or invalid `TENFERRO_DIR`. Every resolved `tenferro-*` package and
manifest is reported, and all must map to one physical git root. It prints
TeNeT, tenferro, and finite-torus SHAs; the governing TeNeT and finite-torus
lock hashes; the feature-tree hash; and the toolchain, system, thread
environment, and every raw result. All Cargo resolution/build commands use
`--locked`, and the runner rejects lock changes. Hashing is portable across
hosts with either `sha256sum` or `shasum`.

The finite-torus executable path comes from Cargo's JSON artifact, so workspace
`target_directory` and `CARGO_TARGET_DIR` overrides do not redirect the runner
to a stale hard-coded binary.

RSS is sampled `RUNS` times for an empty filtered test process and the existing
9,000-space eviction test. Their difference is a process-level upper estimate,
not an allocator-exact cache size. Loader, allocator page-retention, and test
harness noise remain in each observation.

An earlier single-observation probe on main `0bbc56a` measured 1,232 KiB at
count 0, 8,000 KiB at 8,192, and 8,112 KiB at 9,000. These values are retained
only as historical context, not a distribution or acceptance gate. The runner
above supersedes that probe with repeated baseline/churn observations.

## Measured result

Machine: Apple Silicon macOS, `cpu-faer`, dense and recoupling threads 1,
`WARMLOOP=40`; baseline `55b8274`, cached implementation `d99e9b2`. Each entry
is five independent processes.

| gate | baseline median (range) | cached median (range) |
|---|---:|---:|
| chi16 warm allocations | 369,767 (369,759-369,781) | 329,917 (329,881-329,941) |
| chi32 warm allocations | 443,871 (443,865-443,895) | 399,926 (399,902-399,960) |
| chi16 seconds/eval | 0.0944 (0.0913-0.1060) | 0.0974 (0.0857-0.1065) |
| chi32 seconds/eval | 0.2665 (0.2651-0.2831) | 0.2638 (0.2565-0.2817) |
| chi16 cold allocations | 6,346,466 median | 6,320,656 median |
| chi32 cold allocations | 8,955,412 median | 8,924,248 median |

The proven result is a 10.8% chi16 and 9.9% chi32 warm-allocation reduction
with exact energy parity (`-1.9249576438`, `-1.7715916178`). Timing ranges
overlap, so this audit makes no speedup claim. The cold path did not regress in
allocation count. Deep-key PR-3 work remains gated on a separately isolated
timing hotspot.
