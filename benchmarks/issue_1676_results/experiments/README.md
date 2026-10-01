# Diagnostic runs (excluded from the final comparison)

All CSVs in this directory measure throwaway variants or the pre-fix candidate.
They are not inputs to `issue_1676_summarize.py`.

- `before-adjoint-fix`: baseline 34aef370 vs candidate ffa882ea, five alternating
  pairs. The diagonal route improved, but real dense compact LQ regressed by
  approximately 3–5%. This result was not accepted as the final outcome.
- `ablation`: remove only the new diagonal admission check in compact LQ. The
  dense regression persisted, ruling out that check as its cause.
- `boundary`: force an out-of-line `extend_adjoint_col_major` boundary. The real
  dense regression persisted.
- `indexed`: use nested `Vec::push` with identical reserve and traversal order.
  This worsened large real and complex dense cases and was discarded.
- `range`: retain `Vec::extend`, replacing StepBy/Take with an exact-size index
  range. Real dense timing recovered without observed complex-control harm.
  This bounded change was adopted in 122ef8c8 and independently reviewed;
  the root directory contains its separately rebuilt final full comparison.

Each variant was separately rebuilt after scoped Release package cleaning.
The two alternating pairs in each variant directory are diagnostic evidence,
not a standalone speed claim. Early process-first cases sometimes show noise;
raw runs are retained, including non-improvements.

The code-generation investigation used `nm` and `llvm-objdump`: real baseline
adjoint append helpers used scalar load loops, whereas the pre-fix candidate
retained an out-of-line StepBy `try_fold` call per output row. The same pattern
remained with the forced outer boundary. The exact-size range removes this
iterator composition while preserving column-major order, conjugation, and
one reserve. Explicit pushes showed why removing the iterator alone was not
sufficient.

An earlier, invalid build pair reused the same executable from the shared
Cargo target (both SHA256 started `68e1f6`). All resulting CSVs were discarded
from the comparison and retained only in `/tmp/1676-excluded-stale`. Cargo
freshness across worktrees was not sufficient evidence of a rebuild. Every
accepted pair therefore uses scoped `cargo clean --release -p
 tenet-matrixalgebra -p tenet-rs`, distinct saved executable hashes, and a
semantic probe that distinguishes baseline dense outputs from candidate
compact outputs. No speed estimate in the final report uses the invalid runs.
