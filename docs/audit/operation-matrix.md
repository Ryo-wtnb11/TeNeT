# Operation matrix

Audited at: `f58c8f4b8325ae5bebfaf9d90bca5afe8fc8c21d`

This is the current-main capability census for #938. The pinned source and
executable tests are authoritative; an export, a satisfiable trait bound, or an
old benchmark is not proof.

## Status

- **PROVED**: the canonical public path has a current executable semantic test.
- **MEASURED-GAP**: the path works and a current measurement proves a remaining
  production cost.
- **UNSUPPORTED**: the public capability boundary does not expose the path.
- **INTENTIONAL-DIFFERENCE**: the path is supported with a deliberately
  different public or numerical contract.
- **NEEDS-PROOF**: the path is reachable, but the exact provider/operation cell
  lacks an independent public-API oracle.

`PROVED` is deliberately fixture-scoped. It does not claim exhaustive ranks,
sector distributions, products, devices, or checked providers.

## Provider families on default Host storage

All cells refer to `TensorMap<R, D, Vec<D>>`. The ZN column is the built-in
`ZNFusionRule::new(3)` fixture. The two product columns name the exact fixtures
exercised by the suite; none of these columns proves arbitrary moduli, products,
or recursively nested `ProductFusionRule` values.

| Operation family | U(1) | Z2 | ZN(3) | CU(1) | fZ2 | SU(2) | fZ2 x U(1) | (fZ2 x U(1)) x SU(2) | checked Generic seam [1] | Fibonacci |
|---|---|---|---|---|---|---|---|---|---|---|
| Space/tensor construction and labelled block readback | PROVED | PROVED | PROVED | PROVED | PROVED | PROVED | PROVED | PROVED | PROVED | UNSUPPORTED |
| Physical dense expansion and symmetric projection [2] | UNSUPPORTED | UNSUPPORTED | UNSUPPORTED | UNSUPPORTED | UNSUPPORTED | UNSUPPORTED | UNSUPPORTED | UNSUPPORTED | UNSUPPORTED | UNSUPPORTED |
| `adjoint` | PROVED | PROVED | PROVED | PROVED | PROVED | PROVED | PROVED | PROVED | PROVED | UNSUPPORTED |
| `permute`, `transpose`, `repartition` | PROVED | PROVED | PROVED | PROVED | PROVED | PROVED | PROVED | PROVED | PROVED | UNSUPPORTED |
| `braid`, `twist` | PROVED | PROVED | PROVED | PROVED | PROVED | PROVED | PROVED | PROVED | PROVED | UNSUPPORTED |
| `flip` | PROVED | PROVED | PROVED | PROVED | PROVED | PROVED | PROVED | PROVED | PROVED | UNSUPPORTED |
| Unit insertion/removal | PROVED | PROVED | PROVED | PROVED | PROVED | PROVED | PROVED | PROVED | PROVED | UNSUPPORTED |
| `catdomain`, `catcodomain`, `absorb` | PROVED | PROVED | PROVED | PROVED | PROVED | PROVED | PROVED | PROVED | PROVED | UNSUPPORTED |
| `otimes`, arbitrary/ordered `contract`, `compose` | PROVED | PROVED | PROVED | PROVED | PROVED | PROVED | PROVED | PROVED | PROVED | UNSUPPORTED |
| `add`, `scale`, `norm`, `trace_pairs`, `tr` | PROVED | PROVED | PROVED | PROVED | PROVED | PROVED | PROVED | PROVED | PROVED | UNSUPPORTED |
| `inner` [3] | [MEASURED-GAP](https://github.com/Ryo-wtnb11/TeNeT/issues/875) | PROVED | PROVED | PROVED | PROVED | [MEASURED-GAP](https://github.com/Ryo-wtnb11/TeNeT/issues/875) | PROVED | PROVED | PROVED | UNSUPPORTED |
| Compact QR; `left_orth`, `right_orth`; SVD values | PROVED | PROVED | PROVED | PROVED | PROVED | PROVED | PROVED | PROVED | PROVED | UNSUPPORTED |
| Compact/full/truncated SVD; full QR; compact/full LQ | PROVED | PROVED | PROVED | PROVED | PROVED | PROVED | PROVED | PROVED | PROVED | UNSUPPORTED |
| Left/right numerical null spaces [4] | PROVED | PROVED | PROVED | PROVED | PROVED | PROVED | PROVED | PROVED | INTENTIONAL-DIFFERENCE | UNSUPPORTED |
| Left/right polar factors | PROVED | PROVED | PROVED | PROVED | PROVED | PROVED | PROVED | PROVED | PROVED | UNSUPPORTED |
| EIGH full/truncated/values | PROVED | PROVED | PROVED | PROVED | PROVED | PROVED | PROVED | PROVED | PROVED | UNSUPPORTED |
| EIG full/truncated/values | PROVED | PROVED | PROVED | PROVED | PROVED | PROVED | PROVED | PROVED | PROVED | UNSUPPORTED |
| `inv`, `exp`, `powi`, left/right `solve` | PROVED | PROVED | PROVED | PROVED | PROVED | PROVED | PROVED | PROVED | PROVED | UNSUPPORTED |
| `pinv` [5] | PROVED | PROVED | PROVED | PROVED | PROVED | PROVED | PROVED | PROVED | INTENTIONAL-DIFFERENCE | UNSUPPORTED |
| Dense diagonal `sqrt` | PROVED | PROVED | PROVED | PROVED | PROVED | PROVED | PROVED | PROVED | PROVED | UNSUPPORTED |
| Network ordinary planning, contraction/permute replay [6] | PROVED | PROVED | PROVED | PROVED | PROVED | PROVED | PROVED | PROVED | PROVED | UNSUPPORTED |
| Network intra-operand trace [6] | PROVED | PROVED | PROVED | PROVED | PROVED | PROVED | PROVED | PROVED | PROVED | UNSUPPORTED |
| Network payload-destination reuse [6] | PROVED | PROVED | PROVED | PROVED | PROVED | PROVED | PROVED | PROVED | [INTENTIONAL-DIFFERENCE](https://github.com/Ryo-wtnb11/TeNeT/issues/1005) | UNSUPPORTED |
| v1 Host snapshot (`f64`/`Complex64`; admitted dense, compact, and lazy forms) [7] | PROVED | PROVED | PROVED | PROVED | PROVED | PROVED | PROVED | PROVED | PROVED | UNSUPPORTED |

Notes:

1. “Checked Generic seam” means provider-neutral dispatch proved with synthetic
   failure-injection providers and exact SU(3) `[1,1]` / SU(4) `[1,0,1]`
   fixtures, for `f64` and `Complex64` where applicable. SU(N) is evidence, not
   a named-group implementation branch. B/C/D provider work is excluded and
   remains in [#989](https://github.com/Ryo-wtnb11/TeNeT/issues/989).
2. Physical expansion/projection is a separate API project, [#861](https://github.com/Ryo-wtnb11/TeNeT/issues/861).
3. Only the U(1)/SU(2) Host `inner` production gap has a current measurement;
   [#875](https://github.com/Ryo-wtnb11/TeNeT/issues/875) owns it. The
   values-only risk in [#880](https://github.com/Ryo-wtnb11/TeNeT/issues/880)
   is not evidence for another `MEASURED-GAP` cell.
4. TeNeT publishes numerical null spaces using its existing numerical-rank
   policy. This is not a claim of a canonical basis or a TensorKit/QSpace gauge.
5. TeNeT `pinv` is a tolerance-defined numerical pseudoinverse using one strict
   global `rcond * sigma_max`; TensorKit applies its tolerance block-locally.
   The Moore-Penrose identities for the original input hold when the cutoff
   removes only numerical-null directions. Discarding an arbitrary nonzero mode
   instead gives the pseudoinverse of the thresholded effective-rank tensor.
6. For an expression with a repeated label, the macro calls the ordinary
   checked trace during call-local lowering, before it looks up the reduced
   contraction/permutation plan. The trace is recomputed on each call; only the
   reduced plan and its workspace are reused. The macro asks for pivotal
   provider data only for these trace-bearing expressions. Host MF workspaces
   can reuse compatible intermediate payload buffers; checked Generic
   workspaces reuse plan and workspace containers but create newly admitted
   intermediates. The final
   tensor leaves the workspace in both modes, and neither mode accepts a
   caller-owned payload destination. That public destination boundary remains
   an [intentional difference](https://github.com/Ryo-wtnb11/TeNeT/issues/1005)
   from TensorKit's mutating `@tensor C[...] = ...` form.
   Lazy/conjugated checked inputs reject at the same ordinary-operation seam and
   are not silently rerouted through a weaker fallback.
7. Persistence is provider-neutral; reconstruction dispatches through the
   provider's admission mode. For this row, the MF cells combine the U(1)/SU(2)
   persistence fixtures with the existing MF provider-admission proofs. They do
   not claim that TeNeT ships or tests an application codec for every provider
   column. The caller supplies a stable provider key, a resolver returning the
   exact provider `Arc`, and semantic sector-label encoding through
   `TypedPersistenceCodec`; TeNeT does not ship a fixed provider registry. Tests
   also cover checked-Generic vertex multiplicity, both scalar types, and every
   admitted representation kind. MF compact adjoints normalize to owned compact
   tensors; checked Generic preserves a lazy adjoint over a compact parent.
   Direct device snapshots and storage types other than Host `Vec<D>` are
   unsupported; use explicit `to_host()` before encoding a device tensor.

Fibonacci remains expert-layer categorical data, not a canonical typed
provider: its coefficient scalar/codec boundary does not satisfy the ordinary
typed admission contract. Complex categorical composition and Fibonacci are
owned by [#592](https://github.com/Ryo-wtnb11/TeNeT/issues/592) and
[#633](https://github.com/Ryo-wtnb11/TeNeT/issues/633); spinors by
[#651](https://github.com/Ryo-wtnb11/TeNeT/issues/651). Slicing is separately
tracked in [#6](https://github.com/Ryo-wtnb11/TeNeT/issues/6). No public
trivial/dense provider exists.

## Storage and device matrix

| Capability | Host MF `Vec<D>` | Host checked Generic `Vec<D>` | other Host-readable `S` | CUDA f64 MF | CUDA f64 checked Generic | CUDA c64 |
|---|---|---|---|---|---|---|
| Metadata, provider ownership, handle clone | PROVED | PROVED | PROVED | PROVED | PROVED | UNSUPPORTED |
| Stable `data() -> &[D]` | PROVED | PROVED | INTENTIONAL-DIFFERENCE | UNSUPPORTED | UNSUPPORTED | UNSUPPORTED |
| Physical expansion/projection [2] | UNSUPPORTED | UNSUPPORTED | UNSUPPORTED | UNSUPPORTED | UNSUPPORTED | UNSUPPORTED |
| Explicit Host/device transfer | PROVED | PROVED | [NEEDS-PROOF](https://github.com/Ryo-wtnb11/TeNeT/issues/3) | PROVED | PROVED | UNSUPPORTED |
| Lazy adjoint | PROVED | PROVED | [NEEDS-PROOF](https://github.com/Ryo-wtnb11/TeNeT/issues/3) | PROVED | UNSUPPORTED | UNSUPPORTED |
| Permute/braid/recoupling | PROVED | PROVED | UNSUPPORTED | UNSUPPORTED | UNSUPPORTED | UNSUPPORTED |
| Canonical contraction/compose | PROVED | PROVED | UNSUPPORTED | PROVED | UNSUPPORTED | UNSUPPORTED |
| Arithmetic/reductions | PROVED | PROVED | UNSUPPORTED | PROVED | UNSUPPORTED | UNSUPPORTED |
| QR/SVD/EIGH | PROVED | PROVED | UNSUPPORTED | PROVED | UNSUPPORTED | UNSUPPORTED |
| EIG/null/polar/solve/matrix functions | PROVED | PROVED | UNSUPPORTED | UNSUPPORTED | UNSUPPORTED | UNSUPPORTED |
| Network ordinary replay | PROVED | PROVED | INTENTIONAL-DIFFERENCE | PROVED | UNSUPPORTED | UNSUPPORTED |
| v1 typed snapshot (`f64`/`Complex64`) [7] | PROVED | PROVED | UNSUPPORTED | UNSUPPORTED | UNSUPPORTED | UNSUPPORTED |

The two storage `NEEDS-PROOF` cells describe future storage implementations,
not provider conformance. Host/device checked-Generic parity belongs to
[#3](https://github.com/Ryo-wtnb11/TeNeT/issues/3). CUDA `PROVED` means a
real-device test exists and is ignored without CUDA; default CI is not claimed
to run it. Release/feature topology remains [#129](https://github.com/Ryo-wtnb11/TeNeT/issues/129).

Standalone checked-Generic compact construction is `PROVED` by
[#1004](https://github.com/Ryo-wtnb11/TeNeT/issues/1004). Compact diagonal
factors produced by EIGH/EIG remain `PROVED`; checked SVD currently publishes
its diagonal factor densely.

## Current call-path authority

| Public family | Admission/planning | Execution/publication |
|---|---|---|
| Construction | `TypedTensorRootDispatch` selected by `R::Mode` | Complete bound layout before payload allocation; publish one owned body |
| Transform/index operations | `TypedTensorTransformDispatch`, twist/flip dispatch, checked tree context | Stage fallible provider work, then replay to an owned output; identity cases may share storage |
| Product/contraction/trace | Mode dispatch and bound output-space derivation | Ordinary MF or checked lower seam; destination forms preserve caller ownership where exposed |
| Reductions | Mode-dispatched qdim weights | Scalar result only; no tensor/result cache |
| Decompositions/matrix functions | Factor-space admission and sector matricization | Request-local dense calls; publish all factors only after successful staging |
| Left/right solve | Receiver-owned output HomSpace and exact receiver provider `Arc` | One sector solve per populated block; direct/lazy work remains operation-local |
| Network | `Network::plan` plus runtime plan cache | Each step calls ordinary typed contraction/permutation; bounded workspace reuse |
| CUDA | MF-only typed device impl | Explicit transfer/device kernels; unsupported scopes reject before publication |

Key source anchors at the pinned revision:

- `tenet/src/typed.rs`: `TypedTensor*Dispatch`, `TypedAdjointView`,
  `materialized_tensor_uncached`, checked solve/null/polar/EIGH/EIG dispatch.
- `tenet-matrixalgebra/src/factorize.rs`:
  `*_dyn_checked_generic`, checked factor-space builders, sector
  matricizations, truncation and numerical-rank gates.
- `tenet-network/src/network.rs`: `Network`, `PlannedNetwork`, ordinary replay.
- `tenet-network/src/plancache.rs`: bounded plan/workspace ownership and stats.

All checked-provider admissions use the tensor's exact provider `Arc`.
Fallible output admission and dense work are staged before a result is returned.
Provider-owned append-only catalog warming is not rolled back, but a failed
operation publishes no TeNeT output tensor or partial factor tuple.

## Lazy materialization

| Mechanism | Owner/trigger | Contract |
|---|---|---|
| Parent-backed lazy view | `TypedAdjointView` from `adjoint()` | Metadata plus canonical parent; no payload copy |
| Receiver-retained logical payload | lazy view `OnceLock`, only on `data()` | Stable borrowed slice; compatibility storage, not an execution cache |
| Operation-local logical payload | `materialized_tensor_uncached()` | Temporary fallback; never populates the receiver cache |

Orientation-aware transforms, contractions and algebraic adjoint redirects use
the parent where their operation law proves it. QR/LQ, EIGH/EIG logical
orientation, `exp`, solves, `sqrt`, selected cat/absorb conversions and network
rejection checks may use or encounter the operation-local seam. A cold receiver
cache proves only nonpublication; it does not prove zero peak-copy cost. The
ownership/cache audit is #783.

## Executable evidence

- `tenet/tests/typed_serialization.rs`: deterministic v1 Host snapshots,
  exact provider restoration, semantic fusion-tree keys, scalar bit patterns,
  dense/compact/lazy representation boundaries, decode limits, and malformed
  input ordering for multiplicity-free and checked-Generic modes.
- `tenet/tests/mf_structural_conformance.rs`: exact ZN(3), CU(1), SU(2),
  fermionic and exact-product structural, pivotal, contraction and reduction
  laws, including nontrivial CU(1) exchange phase and quantum-dimension oracles.
- `tenet/tests/mf_linalg_conformance.rs`: rectangular and square
  factorization, spectral and matrix-function laws for the remaining provider
  fixtures in #1002, with exact provider-`Arc` preservation.
- `tenet-network/tests/mf_provider_conformance.rs`: explicit and macro network
  execution, static trace and workspace reuse for the exact provider fixtures.
- `tenet/tests/checked_generic_facade.rs`: synthetic failure staging plus
  SU(3)/SU(4) full-key `f64`/`Complex64` oracles for reductions,
  decompositions, null/polar, EIGH/EIG, solve and matrix functions.
- `tenet/tests/checked_generic_twist.rs` and
  `tenet/tests/checked_generic_absorb.rs`: checked pivotal/index/cat/absorb
  layout, exact-Arc, lazy and failure contracts.
- `tenet-network/tests/checked_generic_network.rs`: explicit/greedy/macro
  replay, provider authority, cache/workspace recovery, trace and
  lazy/conjugated rejection boundaries.
- `tenet/tests/semantic_suite.rs` and `tenet/tests/typed_facade.rs`: U(1), Z2,
  fZ2, SU(2), CU(1), external finite-provider and exact product fixtures.
- `tenet/tests/inner_norm_allocations.rs` and
  `tenet/tests/permute_overwrite_allocations.rs`: reduction and destination
  allocation contracts.
- `tenet/tests/typed_cuda_transfer.rs` and
  `tenet-network/tests/typed_cuda_network.rs`: real-device MF transfer,
  operation and canonical-network gates.

TensorKit's `matrixalgebrakit.jl` wrappers and MatrixAlgebraKit's factorization
interfaces are the API/law comparison, not proof of TeNeT execution. QSpace is
used only where a representation/coefficient oracle exists; no QSpace gauge or
provider-wide performance conclusion is inferred from SU(3)/SU(4) fixtures.

## Findings and routing

1. Checked Generic now has the ordinary Host path through network replay and
   static intra-operand trace. Network execution still returns a new tensor;
   caller-owned payload destinations remain an intentional difference recorded
   in [#1005](https://github.com/Ryo-wtnb11/TeNeT/issues/1005). Standalone compact
   construction is [#1004](https://github.com/Ryo-wtnb11/TeNeT/issues/1004).
   Lazy/conjugated contraction still rejects without fallback, and checked CUDA
   remains with [#3](https://github.com/Ryo-wtnb11/TeNeT/issues/3).
2. Exact fixture tests now cover the remaining multiplicity-free evidence gaps
   for U(1), Z2, built-in ZN(3), CU(1), fZ2, SU(2), and the two named product
   fixtures. This does not prove arbitrary moduli or arbitrary products.
3. B/C/D are not closure criteria for #938 and remain isolated in
   [#989](https://github.com/Ryo-wtnb11/TeNeT/issues/989).
4. The only current measured provider-operation gap in this census is U(1)/SU(2)
   Host `inner`, [#875](https://github.com/Ryo-wtnb11/TeNeT/issues/875).
5. Fibonacci, storage/device scope, physical dense conversion, slicing and
   release topology remain with their narrow owners listed above. The obsolete
   CUDA doctest finding is removed: current compile-fail examples use checked
   Generic bounds rather than U(1).

After the checked Generic provider route is complete, performance should be
reviewed as a separate measured phase. This matrix does not manufacture a
performance gap from call-path inspection alone.
