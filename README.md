# TeNeT

This is the active TeNeT rebuild workspace.

The previous implementation has been frozen as `../tenet-legacy` and should be
treated as a reference/fixture source, not as the active development target.

Implementation policy:

- Read `../AGENTS.md` before editing.
- Read `../reviews/2026-06-29-tenet-rebuild-implementation-policy.md` before
  changing architecture.
- Keep public APIs Rust-native and ergonomic.
- Keep low-level tensor execution structurally aligned with the TensorKit
  ecosystem: TensorKit, TensorKitSectors, TensorOperations, MatrixAlgebraKit,
  Strided.jl/StridedViews.jl, and Rust `strided-rs`.

Initial crate layout:

- `tenet-dense`: dense block executor boundary. The default executor currently
  adapts tenferro while keeping tenferro types out of higher tensor algorithms.
- `tenet-operations`: TensorOperations-style tensoradd/contract/permute
  lowering. It uses `strided-rs` internally at the same granularity as the
  TensorKit ecosystem uses Strided.jl/StridedViews.jl.
- `tenet`: public facade crate.
