# TeNeT current-main audit report

Status: **Phase A complete; Phase B independent correctness in progress;
benchmark phases not started.**

This report is the durable summary for issue #9. It separates facts established
from current source/tests from work that still needs an independent oracle or
measurement. It must not be read as a release or performance claim.

## 1. Executive summary

TeNeT has a coherent canonical typed Host path for U(1), Z2, fZ2, SU(2), and
multiplicity-free product symmetries. Construction, index transforms,
contraction/composition, reductions, decompositions, matrix functions, and
typed network execution exist across that path, although the proof depth is
uneven by provider and operation.

Checked Generic/Racah providers use the same ordinary `GradedSpace<R>` and
`TensorMap<R,D>` ownership model and currently support construction/readback,
permute/braid/transpose/repartition, `otimes`, and arbitrary
contraction/composition. Reductions, decompositions, lazy adjoint, several
index operations, and network execution remain unsupported there. SU(N) must
therefore not yet be described as a generally complete tensor provider.

CUDA is a narrower explicit `f64` multiplicity-free execution surface. It has
real-device tests for selected transfer, arithmetic, reductions, canonical
contraction, QR/SVD/EIGH, and canonical network paths. General recoupling,
noncanonical contraction, c64, checked Generic execution, and much of the Host
matrix-algebra surface remain unsupported. No silent general Host fallback was
identified in the inspected public paths.

The cache/ownership review is substantially stronger than the documentation
and release state. Primitive generated coefficients remain provider/Racah
owned; immutable completed transforms and network plans have bounded Runtime
owners; idle network workspaces have a Runtime-wide byte budget; ordinary
numerical results are not implicitly cached. The lazy-adjoint whole-payload
`OnceLock` is compatibility storage for stable `data() -> &[D]`, not an
execution cache, and ordinary operations do not publish it.

TeNeT is not release-ready: a TeNeT-only checkout cannot resolve its required
Tenferro sibling, CI normally follows mutable Tenferro main, no toolchain/MSRV
is pinned, package metadata is incomplete, and `cargo package` fails. Current
documentation also overstates Fibonacci reachability and contains other stale
support/performance material already classified by the Phase A census.

Provisional project assessment: **Promising but not yet justified.** This is not
the requested final verdict. Current source establishes meaningful Rust
ownership, typed errors, bounded structural reuse, and selected CUDA
capability. Phase B now has pinned current TensorKit and executable QSpace
correctness rows, but no neutral performance comparison over representative
applications.

## 2. Scope and methodology

Phase A treats current production source, executable tests, manifests/features,
and current CI as authority. README, tutorials, examples, benchmark scripts,
recorded output, architecture prose, and previous comparisons are inputs only
until separately validated.

The census follows each supported row through:

```text
public entry
  -> provider/tensor/space ownership
  -> destination HomSpace and layout derivation
  -> coefficient authority and structural planning
  -> pack/transform/dense execution/scatter
  -> output publication and retained state
```

The main artifacts are:

- `docs/audit/public-api.jsonl`: rustdoc-JSON-derived public API;
- `docs/audit/artifact-classification.md`: stale-artifact census;
- `docs/audit/operation-matrix.md`: capability and call-path matrix at
  current main `d612869`;
- issue #9 comment `5176167235`: ordered task registry.

## 3. Current API snapshot

The machine-readable snapshot records all 11 workspace library packages,
their declared features and targets, and 2,918 default public-item records.
Feature deltas are recorded for `racah-generated`, `opt-path`,
`cotengra-python`, `provider-inject`, and `cuda`.

Syntactic presence is not treated as operational support. In particular:

- `FibonacciFusionRule` is re-exported but is not a canonical typed provider;
- `provider-inject` changes construction capability without adding public
  items;
- `cuda` expands the item surface but does not make every Host operation a
  device operation;
- no canonical public trivial/no-symmetry provider was identified.

## 4. Legacy artifact assessment

The Phase A inventory classified 49 tracked README/documentation/tutorial/
example/benchmark artifacts:

| Class | Count | Meaning |
|---|---:|---|
| Valid | 17 | Current API and stated scope verified at the pinned revision |
| Salvageable | 17 | Useful material, but requires repinning, rerun, or substantial rewrite |
| Historical | 9 | Revision/migration evidence; not current user guidance |
| Misleading | 6 | Demonstrably stale current-facing API, authority, support, or performance claim |

Phase B has since promoted the TensorKit semantic script and output to Valid:
they reproduce under Julia 1.11.6 and pinned TensorKit main `f87ca7f`. Current
counts, including the new Project, Manifest, and SHA-256 record, are 22 Valid,
16 Salvageable, 9 Historical, and 5 Misleading.

The remaining five misleading artifacts carry historical/audit-pending
treatment.
The root README remains Salvageable rather than Valid and will be rebuilt only
after the operation matrix and Phase B evidence stabilize.

## 5. Installation and reproducibility

Established at TeNeT `8999ec3`, local paired Tenferro `c89ce28`, Racah manifest
pin `86a540f`, and rustc 1.96.0 on macOS arm64:

- a TeNeT-only checkout fails even at `cargo metadata` because of sibling
  `../tenferro-rs` path dependencies;
- with the paired sibling present, default Host build/tests/rustdoc succeed;
- no tracked lockfile, `rust-toolchain*`, or package `rust-version` identifies
  the exact dependency/compiler graph;
- ordinary CI checks out mutable Tenferro main;
- internal path dependencies lack publishable version requirements;
- workspace repository metadata remains a placeholder;
- `cargo package -p tenet --no-verify` fails.

Issue #129 owns this release blocker. It is independent of numerical
correctness and should not be hidden by a successful developer checkout.

## 6. Feature matrix

The canonical matrix is
`docs/audit/operation-matrix.md` (current main `d612869`). Its
important boundaries are:

| Family | Current level |
|---|---|
| U(1), Z2 Host | Broad typed operation surface; strongest current proof coverage |
| fZ2, SU(2), multiplicity-free products Host | Broad surface; selected advanced rows still need provider-specific proof |
| ZN, CU(1) Host | Construction and selected transforms proved; many operation rows remain `NEEDS-PROOF` |
| checked Generic / SU(N) Host | Construction/readback, transforms, otimes, contract/compose, reductions, arithmetic, spectra, and current SVD/QR/LQ leaves; unit/cat, factor-returning EIG/EIGH, null/polar, matrix functions, and network remain unsupported |
| Fibonacci | Expert category data exists; canonical typed tensors unsupported |
| trivial/no symmetry | No canonical public provider identified |
| CUDA f64 multiplicity-free | Selected explicit operations and canonical network execution |
| CUDA c64 / checked Generic execution | Unsupported |
| serialization | Unsupported |

`PROVED` is row-local and revision-local. It does not imply exhaustive ranks,
sector distributions, dtypes, or hardware.

## 7. Correctness findings

Current tests provide meaningful evidence for:

- U(1), SU(2), fZ2, and product contraction identities;
- checked-Generic multiplicity-two construction, transforms, `otimes`,
  contraction/composition, provider ownership, and failure nonpublication;
- compact/full/truncated decomposition and matrix-function behavior on selected
  multiplicity-free fixtures;
- direct and lazy-adjoint operation correspondence;
- selected real-device Host/CUDA parity.

PR #852 adds the independent `tenet/tests/current_public_api_smoke.rs` gate. It
constructs U(1), SU(2), and product-provider tensors through public labels and
executes arbitrary contraction, permutation round trip, double adjoint, trace,
rank-truncated SVD reconstruction, and repeated explicit network-plan replay.
All three smoke tests pass in required Ubuntu CI. No trivial-provider example
is fabricated: that public path remains `UNSUPPORTED`.

The first Phase B reference refresh reran
`benchmarks/tensorkit_semantic_oracle.jl` against TensorKit main `f87ca7f`
(Project 0.17.1). The symbol/coherence, invariant-stream, and planar-repartition
output is numerically unchanged from the historical file; the old file differed
only by its four-line unverified banner. A committed Julia Project/Manifest and
SHA-256 manifest now make this TensorKit oracle reproducible. This validates
only the operations emitted by that script, not the remaining matrix rows.

The pinned TensorKit oracle has since added current-source rows for fZ2 closed
loops, the fZ2 x U(1) x SU(2) invariant stream, real/complex p-norm and matrix
exponential, disjoint/non-self-dual U(1) null-space completion for direct and
lazy-adjoint inputs, complex SU(2) permutation/adjoint-composition/SVD, and
quantum-dimension-weighted U(1)/SU(2) rank truncation. The adjacent Rust tests
compare only gauge-invariant quantities or exact data where the basis is fixed.

QSpace public master `e87ccd1` is now executable with MATLAB R2026a Update 4,
Xcode 16.4 build 16F6, Apple Clang 17.0.0, and the macOS 15.5 SDK. The
checksum-bound `benchmarks/qspace_su2_oracle.m` fixture constructs the SU(2)
spin-half vector operator, records its reduced coefficient `-sqrt(3)/2`, and
checks the Casimir contraction. `tenet/tests/qspace_su2_correspondence.rs`
matches the resulting gauge-invariant closed norm squared `3/2`. This is one
QSpace row, not broad QSpace parity.

The remaining physical dense-reference requirement is currently
`UNSUPPORTED`, not merely untested. TeNeT exposes reduced-block `data()` but no
public physical dense expansion or symmetric projection and no provider CGC
embedding capability. #861 owns that missing boundary. Checked Generic/SU(N)
reductions remain `UNSUPPORTED` under #640, so the executable QSpace SU(3)
defining-representation probe cannot yet become a TeNeT closed-reduction row.

The current unsupported or overstated correctness boundaries are tracked by:

- #640: checked-Generic operation and generated-family completion;
- #662: typed fallible-provider error/nonpublication contract;
- #592 and #633: complex-coefficient Fibonacci admission and planar oracle;
- #861: explicit physical dense expansion and symmetric projection;
- #3: Host/device semantic parity;
- #635: typed right solve.

## 8. Performance results

No neutral current-main TeNeT-versus-TensorKit-versus-QSpace performance claim
has been accepted in this audit. Historical numbers are not baselines.

Current internal measurements justify only narrow decisions:

- checked-Generic completed transforms had material repeated structural work;
  #828 routed them through the existing bounded Runtime store;
- repeated dense SVD/QR/EIGH workspace setup did not materially dominate total
  cost, so persistent factorization workspace/prepared hierarchy was rejected;
- network idle workspace retention was unbounded in bytes and now has a 128 MiB
  Runtime-wide default budget, with zero disabling pooling;
- Racah direct-product reuse belongs in Racah rather than a TeNeT mirror cache.

Phase B must rebuild the benchmark harness around rows that first pass
correctness, record exact environment/authority, and separate cold setup,
planning, cache lookup, packing, dense execution, assembly, allocation, and
retained memory.

## 9. Memory and cache results

| Retained object | Current decision |
|---|---|
| Racah dimensions/channels/CGC/F/R | Provider-owned bounded mathematical data; no TeNeT mirror |
| Canonical HomSpace/layout/completed transform | TeNeT may boundedly reuse immutable structure with complete keys |
| Network topology plan | Runtime-owned bounded reuse |
| Idle network workspace | Runtime-owned scratch under a byte budget and observable stats |
| Factorization result | Not cached |
| Dense factorization scratch | Request-local; persistent retention rejected by measurement |
| Lazy-adjoint whole payload | Receiver compatibility storage populated only by explicit `data()` |
| Operation-local adjoint payload | Temporary fallback, not a cache; optimize only per operation with proof and measurement |

The completed #783 review is the architecture authority. The remaining weak
owned `*_into` factorization copy boundary is recorded in closed #145 with a
narrow reopen gate.

## 10. Application benchmark results

Not yet available under the renewed methodology. Existing iTEBD and diagnostic
examples establish API reachability only. Neutral MPS two-site update,
effective-operator application, and small PEPS benchmarks wait until their
component correctness rows pass Phase B.

## 11. API usability

The canonical API has one typed tensor hierarchy rather than parallel checked,
SU(N)-specific, and device placement enums. Provider authority and Runtime
ownership are explicit. This is structurally preferable to the removed erased
hierarchy.

Usability still needs broader independent evaluation. The Phase A smoke program
confirms that current construction requires a `Runtime`, provider ownership
(normally `Arc<R>`), `GradedSpace`, and an explicit tensor constructor. Missing
trivial symmetry, incomplete checked Generic operations, and feature/backend
availability can make otherwise simple programs non-obvious. Documentation
must record these facts rather than explain them away.

## 12. Documentation status

The current documentation set is not ready for broad user claims. The durable
source/provenance and current tutorial code are partly usable, but README,
roadmap, compatibility tables, benchmark landing pages, and some authority
documents require rebuilding or archival treatment.

Phase E will rebuild README/tutorial/rustdoc/Web documentation from the
stabilized typed API. Every user-facing Rust snippet must compile or run in CI.

## 13. Release readiness

Current classification: **not release-ready**.

Blockers include standalone dependency topology, reproducible Tenferro/toolchain
pinning, package metadata/version requirements, package dry-runs, clippy policy,
tested-platform declarations, and honest supported-operation documentation.
The workspace can be developed and tested in its expected sibling layout, but
that is not a distributable installation contract.

## 14. Critical issues and ownership map

| Issue | Current role |
|---|---|
| #9 | Audit/conformance umbrella and matrix registry |
| #129 | Standalone build, dependency pinning, MSRV/package/release topology |
| #640 | Checked Generic/SU(N) reductions, decompositions, lazy/index coverage, network, B/C/D |
| #662 | Typed fallible-provider errors and transactional nonpublication across reachable rows |
| #592 / #633 | Complex structural coefficients/Fibonacci reachability, then planar anyon oracle |
| #3 | Host/device parity and missing CUDA capabilities |
| #740 | Blocked CUDA destination-zero optimization pending CubeCL publication semantics |
| #582 | Workspace clippy policy/current Rust 1.96 failures |
| #635 | Typed right solve through the shared left-solve authority |
| #596 | Measurement gate for typed axpy-style destination forms |
| #6 | Later memory-bounded network slicing semantics and execution |
| #39 | Cotengra subprocess timeout/cancellation |
| #436 | Bounded/diagnosable CUDA CI setup |
| #651 | Future Spin(N) spinor admission, blocked by Racah and #640 |

No second audit umbrella was created. #783 is closed as the completed cache
architecture review. #145 is closed with its broad workspace/prepared proposal
superseded by #783 measurements. Stale erased-facade acceptance text was removed
from #594, #635, #596, and #662.

## 15. Recommended roadmap

1. Keep the Phase A matrix, smoke gate, issue map, and this report synchronized
   as capabilities change.
2. Complete the remaining Phase B correctness rows and classify unsupported
   dense/Generic boundaries before new performance work.
3. Revalidate and implement #640 in small operation-family leaves, pairing
   every shared boundary with multiplicity-free regression review.
4. Extend #662 alongside each newly reachable checked operation.
5. Build neutral micro/application benchmarks only for correctness-proved rows.
6. Address #129 before claiming installability or release readiness.
7. Rebuild user documentation and final project assessment from the stabilized
   API and current measurements.

## 16. Overall assessment

The final category is intentionally deferred. The current provisional category
is **Promising but not yet justified**:

| Axis | Provisional assessment |
|---|---|
| correctness confidence | Moderate with pinned external evidence for selected multiplicity-free Host rows; partial elsewhere |
| warm numerical performance | Not neutrally compared yet |
| cold setup overhead | Structural layers identified; full neutral measurement pending |
| memory efficiency | Explicit ownership and bounded key caches are promising; application peak memory pending |
| contraction planning/cache reuse | Concrete Runtime functionality exists; neutral workload evaluation pending |
| non-Abelian recoupling | SU(2) strong; checked Generic core operations partial |
| product symmetries | Broad multiplicity-free support; representative application proof pending |
| outer multiplicity | Real SU(3)/SU(4) core fixtures work; reductions/decomposition/network incomplete |
| multithreading | Runtime/context design exists; representative scaling evidence pending |
| GPU potential | Selected real capability exists; broad parity incomplete |
| Rust integration | Strong ownership/error direction; packaging currently blocks distribution |
| API usability | One canonical typed hierarchy; independent usability pass incomplete |
| documentation quality | Mixed; artifact census completed, rebuild pending |
| installation difficulty | High outside the repository sibling layout |
| maintenance burden | Improved by erased removal and provider-neutral routing; CI/release debt remains |

Facts, measurements, interpretations, and hypotheses must remain separated as
later phases update this report.
