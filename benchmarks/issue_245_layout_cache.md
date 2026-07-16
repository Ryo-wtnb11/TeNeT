# Issue #245 fusion-layout cache benchmark

This report records an exploratory non-regression check for commit
`bb3edb57e0eb129ce5ced951c8d0ead253080d04` against `origin/main` at
`7825e085de7ed3d2c3f5921721f7701e4d637a86`.

The machine was not quiet: Zoom, Dia, WindowServer, and `coreaudiod` were using
measurable CPU during the run. The samples therefore do not constitute the
formal machine-quiet <=3% proof. They are retained as reproducible evidence
that did not expose a regression under the available conditions.

## Source and commands

Both revisions used the exact source in
`tenet-core/examples/issue_245_layout_cache.rs`. The source was copied into the
detached main worktree without changing the main revision.

```console
cargo build --release -p tenet-core --example issue_245_layout_cache
CARGO_TARGET_DIR=/private/tmp/tenet-245-main-perf-target cargo build --offline --release -p tenet-core --example issue_245_layout_cache
for sample in 1 2 3 4 5 6 7; do echo "sample,$sample,main"; /private/tmp/tenet-245-main-perf-target/release/examples/issue_245_layout_cache; echo "sample,$sample,branch"; /private/tmp/tenet-245-layout-identity/target/release/examples/issue_245_layout_cache; done
```

The program reports warm coupled-block layout lookup time in nanoseconds per
operation for U(1) spaces with 1, 8, and 64 sectors.

## Raw samples

| sample | revision | 1 sector (ns) | 8 sectors (ns) | 64 sectors (ns) |
|---:|:---|---:|---:|---:|
| 1 | main | 1216 | 5106 | 35587 |
| 1 | branch | 1216 | 5069 | 35366 |
| 2 | main | 1125 | 5036 | 34649 |
| 2 | branch | 1121 | 5077 | 35232 |
| 3 | main | 1119 | 5127 | 36117 |
| 3 | branch | 1132 | 5129 | 35079 |
| 4 | main | 1137 | 5200 | 35473 |
| 4 | branch | 1117 | 5059 | 35643 |
| 5 | main | 1116 | 5130 | 36535 |
| 5 | branch | 1124 | 5077 | 35341 |
| 6 | main | 1135 | 5147 | 35328 |
| 6 | branch | 1147 | 5115 | 35380 |
| 7 | main | 1125 | 5180 | 35713 |
| 7 | branch | 1116 | 5127 | 35419 |

## Medians

| sectors | main (ns) | branch (ns) | change |
|---:|---:|---:|---:|
| 1 | 1125 | 1124 | -0.1% |
| 8 | 5130 | 5077 | -1.0% |
| 64 | 35587 | 35366 | -0.6% |

## Provenance

```text
OS: macOS 15.5 (24F74), Darwin 24.5.0, arm64
rustc: 1.96.0 (ac68faa20 2026-05-25), LLVM 22.1.2
cargo: 1.96.0 (30a34c682 2026-05-25)
main: 7825e085de7ed3d2c3f5921721f7701e4d637a86
branch code: bb3edb57e0eb129ce5ced951c8d0ead253080d04
machine-quiet: no
```
