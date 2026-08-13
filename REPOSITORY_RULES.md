# TeNeT repository rules

These rules apply to every change in this repository.

- **Prove capabilities.** Treat capability claims as evidence, not naming: point to the implementation and a focused test (or benchmark) that exercises the advertised path. Keep unsupported capabilities out of public docs and APIs.
- **Keep execution backends explicit.** Host and CUDA paths must have explicit, equivalent semantics and independent coverage where both are supported. Never silently fall back from CUDA to Host; return a clear error when the requested backend is unavailable.
- **Respect Racah ownership.** Racah owns generated F/R/CGC data and bounded coefficient caches. TeNeT must not grow broad ordinary-operation caches or duplicate generated data without measured evidence and an explicit ownership decision.
- **Use TensorKit/QSpace as an oracle.** For tensor, symmetry, fusion, and block-structure changes, compare against TensorKit/QSpace (including adversarial sectors and nontrivial dimensions), and record the oracle case in tests or review evidence.
- **Keep documentation consistent.** Update API docs, examples, READMEs, and feature declarations together. Documentation must describe the actual supported backend, error, and numerical semantics at the commit being reviewed.
- **Audit the exact commit independently.** Final review lanes must inspect the same immutable commit (record its full SHA), independently cover implementation/API, tests/backend parity, and docs/provenance, and report their results before merge. A branch-status or older-SHA check is not final evidence.

