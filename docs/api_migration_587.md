# Expert facade migration (pre-release)

TeNeT is pre-release (`0.1.0`, with no published TeNeT release), so the
facade cleanup may remove paths. Runtime behavior is unchanged.

| Removed facade path | Direct replacement |
|---|---|
| `tenet::operations::X` (outside the curated allow-list) | `tenet_tensors::X` |
| `tenet::matrixalgebra::X` (outside the curated allow-list) | `tenet_matrixalgebra::X` |

The direct crates are broader and unstable. No compatibility wrappers are
provided; callers should migrate imports before the first stable release.
