# Issue #1124: tree-transform lowering measurement

## Scope

Measured commit: `8aa0b85fb8b2b8843ca54756a027572c872c15c4` (base
`9fee5cfacdb1c28d0df6f3805589d1bb81ef1bc2`), 2026-08-13, arm64 macOS
15.5, Rust 1.96.0, default `cpu-faer` feature, Criterion 0.5.1. Results are
local and indicative, not performance guarantees.

`TreeTransformTaskView` is a borrowed `Copy` view. `TreeTransformStructure`
owns blocks, layouts, coefficients, GEMM jobs/runs, and workspace requirements.
The benchmark does not rebuild it inside replay timing. At this Host replay
admission seam, existing
`TreeTransformReplayProfile::validate` covers task-view construction plus
structure, exact-length, and workspace arithmetic validation;
`multi_coefficient_prepare` separates workspace-local coefficient preparation;
the remaining phase timers cover numerical replay. No provider/F/R work occurs
at this seam.

Tenferro 0.3 does not expose exact GEMM-analysis hit/miss statistics through
the current TeNeT adapter. Fresh versus reused executor is therefore only a
proxy, not a cache-hit count.

## Fixtures and results

Command: `cargo bench -p tenet-operations --bench tree_transform_lowering`.
Each Criterion case used 3 seconds warm-up, 30 samples, and 3 seconds requested
measurement time. Times below are Criterion sample medians.

| Fixture | Tasks/jobs/runs | Structure bytes | Workspace payload lower bound | Fresh workspace + executor | Reused workspace + fresh executor | Reused workspace + executor A/A | Reused workspace + executor A/B |
|---|---:|---:|---:|---:|---:|---:|---:|
| many-small: 32 independent 2x2 recouplings, 8 elements/block | 32/32/1 | 20,824 | 9,216 | 1,204.4 us | 1,204.6 us | 655.3 us | 697.5 us |
| few-large: 2 independent 8x8 recouplings, 256 elements/block | 2/2/1 | 5,656 | 66,560 | 598.6 us | 631.8 us | 50.1 us | 32.0 us |

The A/B case alternates two compiled structures with equal layouts but distinct
coefficient payloads, so the existing workspace-local one-slot coefficient
identity misses every iteration. The inconsistent A/A versus A/B direction
across fixtures is evidence against claiming a stable retention benefit from
these noisy local measurements.

One separately printed cold/warm profiled sample per fixture observed admission
at 0--125 ns and coefficient preparation at 0--1,000 ns, versus numerical replay
phases at 32--757 us. Cold fresh workspace/executor requested 340--346 caller-
thread Rust allocations and 119,368--151,400 bytes; warmed reused A/A requested
0 allocations for few-large and 1 allocation/1,520 bytes for many-small. These
counts exclude worker threads, native libraries, frees, and live/peak bytes, so
they do not establish allocation-free dense replay. The focused unit regression
does establish zero warmed caller-thread Rust allocations for task-view
construction and Host replay admission validation alone; it makes no claim
about opaque Stage A/C admission.

## Decision

Keep the current operation-local borrowed task view and existing workspace-local
one-slot coefficient reuse. Do not add retained lowering metadata or a new cache:
admission is already allocation-free and negligible beside replay, while this
measurement shows no repeatable A/A advantage that could justify ownership,
keying, invalidation, and byte-budget machinery. Reused executors matter in
these fixtures, but Tenferro owns that analysis and its policy.
