# Audit artifacts

These files are revision-pinned evidence, not current capability authority.
Each Markdown artifact names the revision it inspected or measured. Refresh an
artifact at the same stable path; use its Git history to retain earlier
snapshots. `public-api.jsonl` is generated evidence tied to the revision named
by `artifact-classification.md` and must not be hand-edited.

| Artifact | Authority | Later outcome |
| --- | --- | --- |
| [Artifact classification](artifact-classification.md) | `8999ec3678b7f894a3547b1a930802a1adfe90a0` | Documentation reconciliation: [history index](../history.md). |
| [Operation matrix](operation-matrix.md) | `eb99cc405bc57c24c9755d4a9c30b2fcc5aeec2b` | No automatic current-main claim; rerun the matrix for a new authority. |
| [Edge-case matrix](phase-b-edge-matrix.md) | `e9c5d35c45b022f93999bf8be63c27cd08fda1a3` | No automatic current-main claim; use current tests for current support. |
| [Runtime architecture review](symmetric-runtime-architecture-review.md) | TeNeT `5390f64a4e58b76f011f58e1576f632bfd569cf0` and the reference revisions listed in the report | Checked-Generic resource reuse: [#1079](https://github.com/Ryo-wtnb11/TeNeT/pull/1079); replay boundary: [#1080](https://github.com/Ryo-wtnb11/TeNeT/issues/1080); subsequent investigations: [#1081](https://github.com/Ryo-wtnb11/TeNeT/issues/1081), [#1082](https://github.com/Ryo-wtnb11/TeNeT/issues/1082), [#1083](https://github.com/Ryo-wtnb11/TeNeT/issues/1083), and [#1084](https://github.com/Ryo-wtnb11/TeNeT/issues/1084). |
| [Rank-specialization measurement](issue-1081-rank-specialization.md) | `40f5570518176f3a0860abdc43d12253789ad394` and `fe5dae8785461ba543b066abc4298f4abc82061c` | Generic replay retained; no production specialization was added ([#1129](https://github.com/Ryo-wtnb11/TeNeT/pull/1129)). |
| [Lowering-retention measurement](issue-1124-lowering-measurement.md) | `b0984695cfa218cef71ba42ca6c672a6c234e6d2` | No retained TeNeT backend sidecar was justified ([#1128](https://github.com/Ryo-wtnb11/TeNeT/pull/1128)). |

Later outcomes are indexed here rather than inserted into the original audit
narrative. Current source, tests, crate documentation, and normative policy
documents remain the authority for current behavior.
