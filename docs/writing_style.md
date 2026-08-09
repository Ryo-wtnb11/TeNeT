# Writing style

TeNeT documentation describes the current code. Current source and executable
tests are the primary evidence; older issues, roadmaps, benchmarks, and external
libraries are supporting evidence, not substitutes for the current
implementation.

Use plain, direct English:

- name the concrete API, provider family, storage, and device scope;
- prefer a measured or tested comparison to broad claims such as "fast" or
  "compatible";
- keep API terms precise: a provider supplies categorical data, a
  `GradedSpace` describes a leg, and `TensorMap` uses `codomain <- domain`;
- use references to support a stated TeNeT convention or comparison, not to
  define TeNeT's structure;
- state limitations with a source, test, issue, or measurement, and separate a
  missing proof from an unsupported API.

Short examples:

| Avoid | Prefer |
| --- | --- |
| "All providers are fully supported." | "The checked Generic Host operations covered by current tests are listed in the operation matrix." |
| "This is much faster." | "Benchmark X measured Y on the stated fixture and revision." |
| "TensorKit defines this layout." | "TeNeT uses this layout; the cited TensorKit source is the comparison." |

Choose words from the context, not by mechanical replacement:

| Context | Prefer | Reserve for |
| --- | --- | --- |
| A current list of supported operations | "capability inventory" | "census" only when population counting is the point |
| A public API limit | "capability boundary" or "provider interface" | "seam" for an exact internal handoff between components |
| Building a checked public value | "validated construction" | "admission" when naming the actual admission mode or API contract |
| A failed checked operation | "returns an error and no output tensor" | "publication" in internal documentation that defines when a staged result becomes visible |
| Running a stored network plan | "executes the plan" or "reuses the plan" | "replay" when the implementation specifically replays compiled operations |
