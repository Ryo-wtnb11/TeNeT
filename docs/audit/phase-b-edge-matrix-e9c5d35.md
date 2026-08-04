# Phase B edge-case matrix at `e9c5d35`

This is a compact evidence index for issue #9. It records existing executable
tests; it does not upgrade an operation to `PROVED` for providers, dtypes, or
storage forms that are not exercised.

## Executed current-main cells

| Edge or authority | Current test | Result |
| --- | --- | --- |
| Multiplicity and checked provider admission | `checked_generic_facade` (11 tests) | PROVED for the tested checked-only provider and SU(N) multiplicity fixtures |
| Direct and lazy-adjoint disjoint/null-space behavior | `tk_disjoint_null` | PROVED for the tested U(1) fixture and TensorKit invariant |
| Complex scalar, SU(2) structure | `tk_complex_su2` | PROVED for the tested f64/c64 structural invariants |
| Fermionic signs and closed contractions | `tk_fermionic_correspondence` | PROVED for the tested fZ2 TensorKit invariants |
| QSpace SU(2) closed norm | `qspace_su2_correspondence` | PROVED for the recorded equivalent scalar |
| Basic semantic identities and decomposition reconstruction | `semantic_suite` | PROVED for its listed U(1), Z2, SU(2), fZ2, and product fixtures |
| TensorKit invariant streams | `semantic_suite -- --ignored` | PROVED for U(1), SU(2), and product fixtures at this revision |

Commands used from the TeNeT workspace:

```text
cargo test -p tenet --test semantic_suite --test tk_disjoint_null \
  --test tk_complex_su2 --test tk_fermionic_correspondence \
  --test qspace_su2_correspondence --test checked_generic_facade
cargo test -p tenet --release --test semantic_suite -- --ignored
```

The first command ran 31 tests with no failures (three invariant streams are
intentionally ignored in the default profile). The second ran all three
streams successfully. The reference pins and host configuration remain the
ones recorded in #9 comments #854--#874.

## Still not closed by this matrix

- The cells above are split across operation-specific tests; there is no
  machine-readable operation × edge-case result table yet.
- The checked Generic provider still has no reduction/decomposition/network
  rows; those are `UNSUPPORTED` and owned by #640, not inferred from these
  core-operation tests.
- Physical dense expansion/projection is `UNSUPPORTED` (#861), so reduced
  invariant agreement is not a physical dense-value oracle.
- Allocation/retained-memory definitions and cold/first-repeat/warm/cache-
  disabled/plan-replay phases are measured by selected harnesses, not emitted
  uniformly for every populated cell.
- QSpace is recorded only where an equivalent public route exists; absence of
  a QSpace row is `NOT-COMPARABLE`, not a parity claim.

The next Phase B change should fill the missing matrix/counter cells only. No
cache, prepared API, or provider-specific production optimization follows from
this document.
