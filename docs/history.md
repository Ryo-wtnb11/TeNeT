# Historical design records

These documents preserve decisions and evidence from specific revisions. They
are not current API, capability, dependency, or performance authority. Use the
current source, tests, crate documentation, and normative policy documents for
those claims.

## Design and migration snapshots

| Record | What it preserves |
| --- | --- |
| [Early project roadmap](roadmap.md) | The July 2026 implementation checklist and goals. |
| [SU(2) authority migration](su2_authority.md) | The migration to Racah and its recorded compatibility sweep. |
| [API naming decisions](tensorkit_compatibility_table.md) | The July 2026 naming migration table. |
| [Backend compatibility design log](tensorkit_compatibility_todo.md) | Earlier storage and execution-boundary notes. |
| [Semantic naming audit](tensorkit_semantic_naming_audit.md) | The analysis that preceded the naming migration. |
| [Expert facade migration](api_migration_587.md) | The pre-release facade cleanup manifest. |
| [Fusion-tree key design](fusion_tree_key_design.md) | The key and persistence design exploration. |

## Evidence snapshots

- [Audit index](audit/README.md) lists commit-pinned audits and later outcomes.
- [Benchmark index](../benchmarks/README.md) separates runnable harnesses from
  historical measurements.

Historical conclusions remain in their original records. Later outcomes belong
in these indexes instead of being backported into a snapshot.
